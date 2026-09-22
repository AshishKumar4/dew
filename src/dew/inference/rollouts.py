"""Rollout servers: one submission interface over Dew's own server and an OpenAI-compatible engine.

A `RolloutServer` takes one prompt of token ids at a time and resolves a
future `Draw` with the sampled ids, their likelihoods and the policy version
the request was submitted under. The trainer pushes new weights with
`load(variables, version)`, and generation keeps running across the push:
requests already in flight finish on whichever weights are resident when
each token is drawn. That is why a draw carries its submission version, the
oldest weights that may have produced any of its tokens.

Two backends fill the interface. `NativeRolloutServer` drives Dew's
continuous-batching `Server` on a background thread and loads weights in
process, copying the trainer's tree onto the served device. `OpenAIRolloutServer`
posts token ids to a vLLM completions endpoint through `OpenAICompletion` and
reloads weights from disk: `SafetensorsReload` writes the policy in its Hugging
Face layout with `Pretrained.save` and asks the engine to reload it.

A remote engine reports one log-probability per token, of the distribution it
was configured to report, in its own numerics. The draw records it as the
behavior likelihood when that distribution is the sampling one, and records
no raw-policy likelihood: the trainer rescoring the tokens is the only raw
policy it has.
"""

from __future__ import annotations

import os
import shutil
import threading
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from dew.objectives.base import Variables, thaw
from dew.sampling.text import Generation, Sampling

from .clients import OpenAICompletion
from .serving import Server

if TYPE_CHECKING:
    from dew.interop.pretrained import Pretrained


@dataclass(frozen=True)
class Draw:
    """One sampled continuation of one prompt.

    `tokens` are the sampled actions, EOS included when the draw terminated
    on it. `behavior_log_probs` is the likelihood of each action under the
    distribution that drew it; `raw_log_probs` is the unmodified model's,
    or None when the backend cannot report it. `version` is the policy
    version the request was submitted under.
    """

    prompt: tuple[int, ...]
    tokens: tuple[int, ...]
    behavior_log_probs: tuple[float, ...]
    raw_log_probs: tuple[float, ...] | None
    terminated: bool
    version: int

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("a draw continues a nonempty prompt")
        if len(self.behavior_log_probs) != len(self.tokens) or (
                self.raw_log_probs is not None and len(self.raw_log_probs) != len(self.tokens)):
            raise ValueError("every drawn token needs its likelihoods")
        if self.terminated and not self.tokens:
            raise ValueError("a terminated draw ends on its EOS token")


class RolloutServer(Protocol):
    """Sample continuations of token prompts under a versioned, reloadable policy.

    `submit` never blocks on generation. `load` replaces the served weights
    and sets `version`; submissions after it return draws stamped with it.
    """

    @property
    def sampling(self) -> Sampling: ...

    @property
    def version(self) -> int: ...

    def submit(self, prompt: Sequence[int], max_new_tokens: int, *, seed: int) -> Future[Draw]: ...

    def load(self, variables: Variables, version: int) -> None: ...

    def close(self) -> None: ...


def _stops(sampling: Sampling) -> tuple[int, ...]:
    eos = sampling.eos_id
    return () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)


def _prompt(prompt: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(prompt)
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("a rollout prompt is a nonempty row of nonnegative integer token ids")
    return ids


def _budget(max_new_tokens: int) -> int:
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("a rollout draws at least one token")
    return max_new_tokens


def _native_draw(prompt: tuple[int, ...], generation: Generation[np.ndarray], version: int) -> Draw:
    count = int(generation.lengths[0])
    width = len(prompt)
    return Draw(prompt, tuple(int(token) for token in generation.tokens[0, width:width + count]),
                tuple(float(value) for value in generation.behavior_log_probs[0, :count]),
                tuple(float(value) for value in generation.raw_log_probs[0, :count]),
                bool(generation.terminated[0]), version)


class NativeRolloutServer:
    """Serve rollouts from Dew's `Server` on a background stepping thread.

    Submissions and weight loads are serialized with the server's steps by
    one lock; a load lands between two steps. The server's own sampling
    policy and transforms decide the draws, and a draw keeps both the raw
    and the behavior likelihood the server records.
    """

    def __init__(self, server: Server, *, version: int = 0):
        self._server = server
        self._version = version
        self._lock = threading.Condition()
        self._incoming: deque[tuple[tuple[int, ...], int, int, int, Future[Draw]]] = deque()
        self._outstanding: set[Future[Draw]] = set()
        self._closed = False
        self._failure: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="dew-rollout-server", daemon=True)
        self._thread.start()

    @property
    def sampling(self) -> Sampling:
        return self._server.sampling

    @property
    def version(self) -> int:
        return self._version

    def submit(self, prompt: Sequence[int], max_new_tokens: int, *, seed: int) -> Future[Draw]:
        ids, budget = _prompt(prompt), _budget(max_new_tokens)
        future: Future[Draw] = Future()
        with self._lock:
            if self._failure is not None:
                raise RuntimeError("the rollout server stopped") from self._failure
            if self._closed:
                raise RuntimeError("the rollout server is closed")
            self._incoming.append((ids, budget, seed, self._version, future))
            self._outstanding.add(future)
            self._lock.notify()
        return future

    def load(self, variables: Variables, version: int) -> None:
        with self._lock:
            self._server.reload(thaw(variables))
            self._version = version

    def _admit(self) -> None:
        """Hand every queued submission to the server; called under the lock."""
        while self._incoming:
            ids, budget, seed, version, future = self._incoming.popleft()
            ticket = self._server.submit(np.asarray(ids, np.int32), budget, seed=seed)

            def resolve(done, ids=ids, version=version, future=future) -> None:
                self._outstanding.discard(future)
                failure = done.exception()
                if failure is not None:
                    future.set_exception(failure)
                else:
                    future.set_result(_native_draw(ids, done.result(), version))

            ticket.add_done_callback(resolve)

    def _run(self) -> None:
        try:
            while True:
                with self._lock:
                    while not (self._closed or self._incoming
                               or self._server.occupancy or self._server.queued):
                        self._lock.wait()
                    if self._closed:
                        return
                    self._admit()
                    self._server.step()
        except BaseException as failure:
            with self._lock:
                self._failure = failure
                for future in [*self._outstanding, *(entry[-1] for entry in self._incoming)]:
                    if not future.done():
                        future.set_exception(failure)
                self._outstanding.clear()
                self._incoming.clear()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._lock.notify()
        self._thread.join()
        cancelled = RuntimeError("the rollout server closed before this draw finished")
        for future in [*self._outstanding, *(entry[-1] for entry in self._incoming)]:
            if not future.done():
                future.set_exception(cancelled)


def _served(variables: Variables, dtype: jnp.dtype) -> Variables:
    """Cast floating leaves to the serving dtype, leaving integer state alone."""
    return jax.tree.map(lambda leaf: leaf.astype(dtype) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
                        thaw(variables))


@dataclass(frozen=True)
class SafetensorsReload:
    """Write the policy as safetensors in its source layout, then hot-reload the engine.

    `source` is the `Pretrained` the trainer's model was loaded from; its
    `save` writes the weights, the config it derives and the tokenizer
    files, which is the directory the engine was launched on. Files are
    staged beside `directory` and moved in with `os.replace`, so the engine
    never reads a half-written file. `base_url` is the engine's root, not
    its `/v1` API.

    The vLLM reload is its development endpoint (`VLLM_SERVER_DEV_MODE=1`):
    `POST /collective_rpc {"method": "reload_weights"}`, which reloads from
    the served directory, then `POST /reset_prefix_cache?reset_running_requests=true`.
    That reset preempts running requests and recomputes them, so no cached
    prefix outlives the weights that computed it; without the flag vLLM
    answers 200 and skips the reset whenever a request holds KV blocks. A
    reset vLLM still cannot make answers non-200 and fails the push.
    """

    source: Pretrained
    directory: Path
    base_url: str
    dtype: str = "bfloat16"
    timeout: float = 600.0

    def write(self, variables: Variables) -> None:
        """Write `variables` into `directory`, file by file atomically."""
        directory = Path(self.directory)
        staging = directory.with_name(f".{directory.name}.staging")
        shutil.rmtree(staging, ignore_errors=True)
        self.source.save(staging, variables=_served(variables, jnp.dtype(self.dtype)))
        directory.mkdir(parents=True, exist_ok=True)
        for written in staging.iterdir():
            os.replace(written, directory / written.name)
        staging.rmdir()

    def __call__(self, variables: Variables) -> None:
        import httpx

        self.write(variables)
        root = self.base_url.rstrip("/")
        for path, body in (("/collective_rpc", {"method": "reload_weights"}), ("/reset_prefix_cache?reset_running_requests=true", None)):
            response = httpx.post(root + path, json=body, timeout=self.timeout)
            if response.status_code != 200:
                raise RuntimeError(f"{path.split('?')[0]} answered {response.status_code}: {response.text}")


class WeightSync(Protocol):
    def __call__(self, variables: Variables) -> None: ...


class OpenAIRolloutServer:
    """Serve rollouts from a vLLM OpenAI-compatible completions endpoint.

    Each submission is one completion request of token ids, carrying the
    `Sampling` policy as vLLM request controls, a seed, one reported
    log-probability per sampled token and the ids themselves
    (`return_tokens_as_token_ids`). `workers` requests are in flight at once;
    the engine batches them.

    vLLM reports raw model log-probabilities unless it runs with
    `--logprobs-mode processed_logprobs`. Those are the behavior likelihoods
    only when the sampling applies no transform (temperature one, no top-k,
    top-p or min-p), so a transforming policy needs `processed_logprobs=True`
    to say the engine was started that way.
    """

    def __init__(self, completion: OpenAICompletion, sampling: Sampling, weights: WeightSync, *,
                 version: int = 0, workers: int = 64, processed_logprobs: bool = False):
        if completion.provider != "vllm":
            raise ValueError("token rollouts need provider='vllm': ids, seeds and EOS controls are vLLM request fields")
        if sampling.transforms() and not processed_logprobs:
            raise ValueError(
                "vLLM reports raw log-probabilities by default, which are not the behavior likelihoods "
                "of a transforming Sampling; start it with --logprobs-mode processed_logprobs and pass "
                "processed_logprobs=True, or sample at temperature one without filters")
        if sampling.temperature == 0:
            raise ValueError("a rollout samples; greedy decoding gives every group member the same draw")
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive number of concurrent requests")
        self._completion = completion
        self._sampling = sampling
        self._weights = weights
        self._version = version
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dew-rollout-request")

    @property
    def sampling(self) -> Sampling:
        return self._sampling

    @property
    def version(self) -> int:
        return self._version

    def submit(self, prompt: Sequence[int], max_new_tokens: int, *, seed: int) -> Future[Draw]:
        return self._pool.submit(self._draw, _prompt(prompt), _budget(max_new_tokens), seed, self._version)

    def _draw(self, prompt: tuple[int, ...], budget: int, seed: int, version: int) -> Draw:
        # The pad id shapes Dew's packed rows; it is not a request field.
        completion = self._completion([list(prompt)], budget, seed=seed, sampling=replace(self._sampling, pad_id=0),
                                      logprobs=0, extra_body={"return_tokens_as_token_ids": True})
        tokens, probabilities = completion.tokens[0], completion.log_probs[0]
        if tokens is None or probabilities is None:
            raise ValueError("the engine reported no sampled token ids; is it vLLM?")
        stops = _stops(self._sampling)
        terminated = bool(tokens) and tokens[-1] in stops
        reason = completion.finish_reasons[0]
        if reason != ("stop" if terminated else "length") or (not terminated and len(tokens) != budget):
            raise ValueError(f"finish reason {reason!r} disagrees with {len(tokens)} drawn ids ending {tokens[-1:]}")
        return Draw(prompt, tokens, probabilities, None, terminated, version)

    def load(self, variables: Variables, version: int) -> None:
        self._weights(variables)
        self._version = version

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
