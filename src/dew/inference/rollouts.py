"""Rollout servers: one submission interface over Dew's own server and OpenAI-compatible engines.

A `RolloutServer` takes one prompt of token ids at a time and resolves a
future `Draw` with the sampled ids, their likelihoods and the policy version
the request was submitted under. The trainer pushes new weights with
`load(variables, version)`, and generation keeps running across the push:
requests already in flight finish on whichever weights are resident when
each token is drawn. That is why a draw carries its submission version, the
oldest weights that may have produced any of its tokens.

Two backends fill the interface. `NativeRolloutServer` drives Dew's
continuous-batching `Server` on a background thread and loads weights in
process, copying the trainer's tree onto the served device.
`OpenAIRolloutServer` posts token ids to a vLLM or SGLang completions
endpoint through `OpenAICompletion` and reloads weights from disk:
`SafetensorsReload` writes the policy in its Hugging Face layout with
`Pretrained.save` and asks the engine to reload it. The request, the
response and the draw are the same for both engines; they differ in the
field that returns sampled ids, the distribution their log-probabilities
describe, and the reload calls.

A remote engine reports one log-probability per token, of the distribution it
was configured to report, in its own numerics. The draw records it as the
behavior likelihood when that distribution is the sampling one, and records
no raw-policy likelihood: the trainer rescoring the tokens is the only raw
policy it has.
"""

from __future__ import annotations

import math
import os
import shutil
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from dew.artifacts import agreed, collective_host
from dew.objectives.base import Variables, thaw
from dew.records import JSON
from dew.sampling.text import Generation, Sampling

from .clients import OpenAICompletion
from .serving import Server

if TYPE_CHECKING:
    import httpx

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
        for probabilities in (self.behavior_log_probs, self.raw_log_probs or ()):
            if not all(math.isfinite(value) for value in probabilities):
                raise ValueError("drawn likelihoods must be finite")
        if self.terminated and not self.tokens:
            raise ValueError("a terminated draw ends on its EOS token")

    def check_stops(self, stops: tuple[int, ...]) -> Draw:
        """This draw, refused unless it ends on EOS exactly when it terminated and holds no earlier EOS."""
        if self.terminated != bool(self.tokens and self.tokens[-1] in stops) or any(
                token in stops for token in self.tokens[:-1]):
            raise ValueError("the draw's termination disagrees with its EOS tokens")
        return self


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
        """Queue one draw; a request the server refuses raises here and leaves the rest running."""
        ids, budget = _prompt(prompt), _budget(max_new_tokens)
        future: Future[Draw] = Future()
        with self._lock:
            if self._failure is not None:
                raise RuntimeError("the rollout server stopped") from self._failure
            if self._closed:
                raise RuntimeError("the rollout server is closed")
            ticket = self._server.submit(np.asarray(ids, np.int32), budget, seed=seed)
            self._outstanding.add(future)
            version = self._version

            def resolve(done: Future[Generation[np.ndarray]]) -> None:
                self._outstanding.discard(future)
                failure = done.exception()
                if failure is not None:
                    future.set_exception(failure)
                else:
                    future.set_result(_native_draw(ids, done.result(), version).check_stops(self.sampling.stops))

            ticket.add_done_callback(resolve)
            self._lock.notify()
        return future

    def load(self, variables: Variables, version: int) -> None:
        with self._lock:
            self._server.reload(thaw(variables))
            self._version = version

    def _run(self) -> None:
        try:
            while True:
                with self._lock:
                    while not (self._closed or self._server.occupancy or self._server.queued):
                        self._lock.wait()
                    if self._closed:
                        return
                    self._server.step()
        except BaseException as failure:
            with self._lock:
                self._failure = failure
                for future in self._outstanding:
                    if not future.done():
                        future.set_exception(failure)
                self._outstanding.clear()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._lock.notify()
        self._thread.join()
        cancelled = RuntimeError("the rollout server closed before this draw finished")
        for future in list(self._outstanding):
            if not future.done():
                future.set_exception(cancelled)


def _served(variables: Variables, dtype: jnp.dtype) -> Variables:
    """Cast floating leaves to the serving dtype, leaving integer state alone."""
    return jax.tree.map(lambda leaf: leaf.astype(dtype) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
                        thaw(variables))


@dataclass(frozen=True)
class SafetensorsReload:
    """Publish a policy version to a set of engine replicas through safetensors on disk.

    `source` is the `Pretrained` the trainer's model was loaded from; its
    `save` writes the weights, the config it derives and the tokenizer
    files, which is the directory every replica was launched on (a shared
    filesystem when the replicas are on several hosts). Files are staged
    beside `directory` and moved in with `os.replace`, so no engine reads a
    half-written file. `engines` are the replicas' roots, not their `/v1`
    APIs. An engine call fails the push when it answers other than 200, or
    when a call that reports its outcome carries `success` that is not true:
    both engines report some failures as a 200 with `{"success": false}`.

    One push of version `v` writes the directory once, then runs the
    replica sequence on every replica concurrently. vLLM (`engine="vllm"`,
    checked against v0.30.0), through its development endpoints
    (`VLLM_SERVER_DEV_MODE=1`): `POST /pause?mode=wait`, which lets
    in-flight requests finish and schedules no new ones; `POST
    /collective_rpc {"method": "reload_weights"}`, which reloads from the
    served directory; `POST /reset_prefix_cache`, which must answer
    `{"success": true}` so no cached prefix outlives the weights that
    computed it; `POST /update_weight_version {"new_version": "v"}`, which
    must answer `{"success": true}`; and `POST /resume`. In-flight draws
    therefore finish wholly on the old weights. A replica that fails after
    the pause stays paused, so no draw is sampled from weights the push may
    have half loaded; the next push that succeeds resumes it.

    SGLang (`engine="sglang"`, checked against v0.5.20) runs one call per
    replica, `POST /update_weights_from_disk {"model_path": directory,
    "flush_cache": true, "weight_version": "v"}`. SGLang admits it only once
    every in-flight request has finished, holds new requests until it
    returns, and flushes the radix cache before answering, so in-flight
    draws finish wholly on the old weights and no prefix computed by them
    survives. A load that fails answers 400 with `success: false`; SGLang's
    rollback re-reads the same directory, so the replica then serves
    whatever that directory holds.

    `gateway`, when set, is a recording gateway's root (rllm-model-gateway):
    once every replica serves `v`, `POST /admin/weight_version
    {"weight_version": v}` makes it stamp `v` on the calls it records from
    then on. The gateway reads its stamp when a request arrives, so the
    stamp must never run ahead of any replica: a call routed to a replica
    still on `v - 1` would claim weights it was not sampled from. Stamping
    after the whole set has moved is conservative instead: a call submitted
    between a replica's resume and the stamp is recorded as `v - 1`, older
    than the weights that served it. A push with any failed replica leaves
    the stamp where it was.

    A multi-process trainer calls the push on every process. The pool
    gathers the served tree to host memory on every process
    (`collective_host`), process 0 writes and publishes, and every process
    learns the outcome at an agreement point, so a failed push raises on
    all of them instead of leaving the others to hang at the next
    collective.
    """

    source: Pretrained
    directory: Path
    engines: tuple[str, ...]
    engine: Literal["vllm", "sglang"]
    gateway: str | None = None
    dtype: str = "bfloat16"
    timeout: float = 600.0

    def __post_init__(self) -> None:
        if self.engine not in ("vllm", "sglang"):
            raise ValueError("engine must be vllm or sglang")
        if (not isinstance(self.engines, tuple) or not self.engines
                or not all(isinstance(root, str) and root for root in self.engines)):
            raise ValueError("engines is a nonempty tuple of replica root URLs")
        if len(set(self.engines)) != len(self.engines):
            raise ValueError("every replica is published to once")

    def write(self, variables: Variables) -> None:
        """Write `variables` into `directory`, file by file atomically; every process of a pool calls it."""
        served = collective_host(_served(variables, jnp.dtype(self.dtype)), phase="weight export gather")
        agreed("weight export", lambda: self._save(served) if jax.process_index() == 0 else None)

    def _save(self, variables: Variables) -> None:
        directory = Path(self.directory)
        staging = directory.with_name(f".{directory.name}.staging")
        shutil.rmtree(staging, ignore_errors=True)
        self.source.save(staging, variables=variables)
        directory.mkdir(parents=True, exist_ok=True)
        for written in staging.iterdir():
            os.replace(written, directory / written.name)
        staging.rmdir()

    def __call__(self, variables: Variables, version: int) -> None:
        if type(version) is not int or version < 0:
            raise ValueError("a published policy version is a nonnegative integer")
        self.write(variables)
        agreed("weight publication", lambda: self._publish(version) if jax.process_index() == 0 else None)

    def _publish(self, version: int) -> None:
        import httpx

        with ThreadPoolExecutor(max_workers=len(self.engines), thread_name_prefix="dew-weight-push") as pool:
            pushes = [(root, pool.submit(self._replica, root, version)) for root in self.engines]
            failures = [(root, push.exception()) for root, push in pushes if push.exception() is not None]
        if failures:
            raise RuntimeError(f"version {version} reached {len(self.engines) - len(failures)} of "
                               f"{len(self.engines)} replicas; the gateway keeps its stamp. "
                               + "; ".join(f"{root}: {failure}" for root, failure in failures))
        if self.gateway is not None:
            response = httpx.post(self.gateway.rstrip("/") + "/admin/weight_version",
                                  json={"weight_version": version}, timeout=self.timeout)
            if response.status_code != 200 or response.json().get("weight_version") != version:
                raise RuntimeError(f"the gateway refused version {version}: "
                                   f"{response.status_code} {response.text}")

    def _replica(self, root: str, version: int) -> None:
        import httpx

        root = root.rstrip("/")
        # (path, JSON body, whether the answer must say {"success": true})
        if self.engine == "vllm":
            calls = (("/pause?mode=wait", None, False), ("/collective_rpc", {"method": "reload_weights"}, False),
                     ("/reset_prefix_cache", None, True),
                     ("/update_weight_version", {"new_version": str(version)}, True), ("/resume", None, False))
        else:
            calls = (("/update_weights_from_disk", {"model_path": str(Path(self.directory).resolve()),
                                                    "flush_cache": True, "abort_all_requests": False,
                                                    "weight_version": str(version)}, True),)
        for path, body, reports in calls:
            response = httpx.post(root + path, json=body, timeout=self.timeout)
            if response.status_code != 200 or (reports and not _succeeded(response)):
                raise RuntimeError(f"{path.split('?')[0]} answered {response.status_code}: {response.text}")


def _succeeded(response: httpx.Response) -> bool:
    """Whether the answer is a JSON object whose `success` is true."""
    if not response.headers.get("content-type", "").startswith("application/json"):
        return False
    answer = response.json()
    return isinstance(answer, dict) and answer.get("success") is True


class WeightSync(Protocol):
    """Make the engines serve `variables` as policy `version`."""

    def __call__(self, variables: Variables, version: int) -> None: ...


# The request field that makes each engine return the sampled ids themselves.
_RETURN_IDS: dict[str, dict[str, JSON]] = {"vllm": {"return_tokens_as_token_ids": True},
                                           "sglang": {"return_token_ids": True}}


class OpenAIRolloutServer:
    """Serve rollouts from a vLLM or SGLang OpenAI-compatible completions endpoint.

    Each submission is one completion request of token ids, carrying the
    `Sampling` policy as engine request controls, a seed, one reported
    log-probability per sampled token and the ids themselves. `workers`
    requests are in flight at once; the engine batches them. The engine is
    `completion.provider`.

    The reported log-probabilities are the behavior likelihoods only for
    some policies. vLLM reports raw model log-probabilities unless it runs
    with `--logprobs-mode processed_logprobs`, so a transforming policy
    (temperature other than one, top-k, top-p or min-p) needs
    `processed_logprobs=True` to say the engine was started that way.
    SGLang's `/v1/completions` reports the temperature-scaled distribution
    before its top-k, top-p and min-p filters and has no field for the
    filtered one, so this server takes any temperature but no filter.
    (SGLang's native `/generate` reports the filtered likelihood under
    `return_sampling_mask`, for a finite top-k; that is the route a filtered
    policy would need.) `SGLANG_RETURN_ORIGINAL_LOGPROB` switches the report
    to raw log-probabilities and must stay unset.

    SGLang honors a request's seed only under `--enable-deterministic-inference`;
    otherwise draws are unseeded.
    """

    def __init__(self, completion: OpenAICompletion, sampling: Sampling, weights: WeightSync, *,
                 version: int = 0, workers: int = 64, processed_logprobs: bool = False):
        engine = completion.provider
        if engine not in _RETURN_IDS:
            raise ValueError("token rollouts need provider='vllm' or 'sglang': ids, seeds and EOS controls "
                             "are engine request fields")
        if sampling.temperature == 0:
            raise ValueError("a rollout samples; greedy decoding gives every group member the same draw")
        if engine == "vllm" and sampling.transforms() and not processed_logprobs:
            raise ValueError(
                "vLLM reports raw log-probabilities by default, which are not the behavior likelihoods "
                "of a transforming Sampling; start it with --logprobs-mode processed_logprobs and pass "
                "processed_logprobs=True, or sample at temperature one without filters")
        if engine == "sglang" and processed_logprobs:
            raise ValueError("processed_logprobs names a vLLM mode; SGLang's completions route has none")
        if engine == "sglang" and replace(sampling, temperature=1.0).transforms():
            raise ValueError("SGLang's /v1/completions reports log-probabilities before top-k, top-p and min-p, "
                             "which are not the behavior likelihoods of a filtering Sampling, and has no field "
                             "for the filtered ones (native /generate's return_sampling_mask does); "
                             "sample without filters")
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive number of concurrent requests")
        self._completion = completion
        self._return_ids = _RETURN_IDS[engine]
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
                                      logprobs=0, extra_body=self._return_ids)
        tokens, probabilities = completion.tokens[0], completion.log_probs[0]
        if tokens is None or probabilities is None:
            raise ValueError(f"the engine reported no sampled token ids; is it {self._completion.provider}?")
        stops = self._sampling.stops
        terminated = bool(tokens) and tokens[-1] in stops
        reason = completion.finish_reasons[0]
        if reason != ("stop" if terminated else "length") or (not terminated and len(tokens) != budget):
            raise ValueError(f"finish reason {reason!r} disagrees with {len(tokens)} drawn ids ending {tokens[-1:]}")
        return Draw(prompt, tokens, probabilities, None, terminated, version).check_stops(stops)

    def load(self, variables: Variables, version: int) -> None:
        self._weights(variables, version)
        self._version = version

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
