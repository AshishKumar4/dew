"""Rollout servers: one submission interface over Dew's own server and OpenAI-compatible engines.

A `RolloutServer` takes one prompt of token ids at a time and resolves a
future `Draw` with the sampled ids, their likelihoods and the policy version
the request was submitted under. `load(variables, version)` pushes new
weights while generation keeps running, so in-flight requests finish on
whichever weights are resident when each token is drawn, and a draw's version
is the oldest weights that may have produced any of its tokens.

`NativeRolloutServer` drives Dew's continuous-batching `Server` on a
background thread and loads weights in process. `OpenAIRolloutServer` posts
token ids to a vLLM or SGLang completions endpoint, and `SafetensorsReload`
publishes weights to it through disk. A remote engine reports one
log-probability per token in its own numerics; the draw records it as the
behavior likelihood when that is the sampling distribution, and no raw-policy
likelihood, which only the trainer's rescoring has.
"""

from __future__ import annotations

import math
import os
import shutil
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from dew.coordination import agreed, collective_host, stop_at_exit
from dew.nn.inputs import key_seed, request_key
from dew.objectives.base import Variables, thaw
from dew.records import JSON
from dew.sampling.text import Generation, Sampling

from .clients import OpenAICompletion, decode_routed_experts
from .serving import Server

if TYPE_CHECKING:
    import httpx

    from dew.interop.pretrained import Pretrained


@dataclass(frozen=True)
class Draw:
    """One sampled continuation of one prompt.

    `tokens` are the sampled actions, with EOS included when the draw
    terminated on it. `behavior_log_probs` is the likelihood of each action
    under the distribution that drew it. `raw_log_probs` is the unmodified
    model's likelihood, or None when the backend cannot report it. `version`
    is the policy version the request was submitted under. When requested,
    `routed_experts` is the engine's mixture routing for every id it
    forwarded, and `support` is the ids its sampler kept for each drawn
    token (see `sessions.Call.routed_experts` and `sessions.Call.support`).
    Likelihoods that do not match the tokens one to one, or are not finite,
    raise ValueError.
    """

    prompt: tuple[int, ...]
    tokens: tuple[int, ...]
    behavior_log_probs: tuple[float, ...]
    raw_log_probs: tuple[float, ...] | None
    terminated: bool
    version: int
    routed_experts: np.ndarray | None = None
    support: tuple[tuple[int, ...], ...] | None = None

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
        if self.support is not None and len(self.support) != len(self.tokens):
            raise ValueError("a draw's support holds one set of kept ids per drawn token")

    def check_stops(self, stops: tuple[int, ...]) -> Draw:
        """Return this draw after checking its EOS tokens against the stop ids `stops`.

        Raises ValueError unless the draw ends on a stop id exactly when it
        terminated, with no stop id before its last token.
        """
        if self.terminated != bool(self.tokens and self.tokens[-1] in stops) or any(
                token in stops for token in self.tokens[:-1]):
            raise ValueError("the draw's termination disagrees with its EOS tokens")
        return self


class RolloutServer(Protocol):
    """Samples continuations of token prompts under a versioned policy whose weights can be reloaded.

    `submit` returns a future without waiting for generation. `load`
    replaces the served weights and sets `version`, and draws submitted
    after it report that version.
    """

    @property
    def sampling(self) -> Sampling: ...

    @property
    def version(self) -> int: ...

    def submit(self, prompt: Sequence[int], max_new_tokens: int, *, key: int | jax.Array) -> Future[Draw]: ...

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
    """Serves rollouts from Dew's `Server`, stepping it on a background thread.

    One lock orders submissions and weight loads with the server's steps,
    so a load happens between two steps. The server's own sampling policy
    and transforms decide the draws, and a draw keeps both the raw and the
    behavior likelihoods the server records. If the program ends without
    `close`, an exit hook stops the stepping after the step in progress.
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
        # A step under way at exit runs to its end inside jaxlib, so the bound is a step's, not a poll's.
        self._withdraw = stop_at_exit(self._thread, self._stop, timeout=60.0)

    @property
    def sampling(self) -> Sampling:
        return self._server.sampling

    @property
    def version(self) -> int:
        return self._version

    def submit(self, prompt: Sequence[int], max_new_tokens: int, *, key: int | jax.Array) -> Future[Draw]:
        """Queue one draw and return its future.

        A request the server refuses raises here, and the other requests keep
        running.
        """
        request = request_key(key)
        ids, budget = _prompt(prompt), _budget(max_new_tokens)
        future: Future[Draw] = Future()
        with self._lock:
            if self._failure is not None:
                raise RuntimeError("the rollout server stopped") from self._failure
            if self._closed:
                raise RuntimeError("the rollout server is closed")
            ticket = self._server.submit(np.asarray(ids, np.int32), budget, key=request)
            self._outstanding.add(future)
            version = self._version

            def resolve(done: Future[Generation[np.ndarray]]) -> None:
                self._outstanding.discard(future)
                failure = done.exception()
                if failure is not None:
                    future.set_exception(failure)
                else:
                    future.set_result(
                        _native_draw(ids, done.result(), version).check_stops(self.sampling.stops)
                    )

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

    def _stop(self) -> None:
        with self._lock:
            self._closed = True
            self._lock.notify()

    def close(self) -> None:
        self._stop()
        self._thread.join()
        self._withdraw()
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
    """Publishes a policy version to engine replicas through safetensors on disk.

    `source.save` writes the weights, the derived config and the tokenizer
    files into `directory`, the directory every replica was launched on (a
    shared one when replicas span hosts). Floating-point leaves are cast to
    `dtype` first. The files are staged beside `directory` and moved in with
    `os.replace`, so no engine reads a half-written file. `engines` are the
    replicas' root URLs, not their `/v1` API URLs, and `timeout` bounds each
    HTTP call in seconds. The push fails if a call answers with any status
    but 200, or, for a call that reports its outcome, with a `success` that
    is not true. Both engines report some failures as a 200 with
    `{"success": false}`.

    For vLLM (`engine="vllm"`, v0.30.0, `VLLM_SERVER_DEV_MODE=1`), the push
    pauses with `mode=wait`, reloads, resets the prefix cache, sets the
    weight version and resumes. In-flight draws therefore finish entirely on
    the old weights, and no cached prefix outlives them. A replica that fails
    after the pause stays paused until a later push succeeds. For SGLang
    (`engine="sglang"`, v0.5.20), the push is one `/update_weights_from_disk`
    call, which waits for in-flight requests, holds new ones and flushes the
    radix cache. A failed SGLang load rolls back by re-reading the same
    directory.

    A failed push raises an error that names every replica that did not take
    the version. Every process of a pool calls the push.
    `collective_host(held_by="first")` gathers the served tree to process 0,
    which writes and publishes it, and an agreement point raises a failure on
    every process, so the others do not hang at the next collective.
    """

    source: Pretrained
    directory: Path
    engines: tuple[str, ...]
    engine: Literal["vllm", "sglang"]
    dtype: str = "bfloat16"
    timeout: float = 600.0

    def __post_init__(self) -> None:
        if self.engine not in ("vllm", "sglang"):
            raise ValueError("engine must be vllm or sglang")

    def write(self, variables: Variables) -> None:
        """Write `variables` into `directory`, replacing each file atomically.

        Every process of a pool calls it.
        """
        served = collective_host(_served(variables, jnp.dtype(self.dtype)), phase="weight export gather",
                                 held_by="first")
        agreed("weight export", lambda: None if served is None else self._save(served))

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
        self.write(variables)
        agreed("weight publication", lambda: self._publish(version) if jax.process_index() == 0 else None)

    def _publish(self, version: int) -> None:
        with ThreadPoolExecutor(max_workers=len(self.engines), thread_name_prefix="dew-weight-push") as pool:
            pushes = [(root, pool.submit(self._replica, root, version)) for root in self.engines]
            failures = [(root, push.exception()) for root, push in pushes if push.exception() is not None]
        if failures:
            raise RuntimeError(f"version {version} reached {len(self.engines) - len(failures)} of "
                               f"{len(self.engines)} replicas: "
                               + "; ".join(f"{root}: {failure}" for root, failure in failures))

    def _replica(self, root: str, version: int) -> None:
        root = root.rstrip("/")
        # (path, JSON body, whether the answer must say {"success": true})
        calls: tuple[tuple[str, JSON, bool], ...]
        if self.engine == "vllm":
            calls = (
                ("/pause?mode=wait", None, False),
                ("/collective_rpc", {"method": "reload_weights"}, False),
                ("/reset_prefix_cache", None, True),
                ("/update_weight_version", {"new_version": str(version)}, True),
                ("/resume", None, False),
            )
        else:
            calls = (("/update_weights_from_disk", {"model_path": str(Path(self.directory).resolve()),
                                                    "flush_cache": True, "abort_all_requests": False,
                                                    "weight_version": str(version)}, True),)
        for path, body, reports in calls:
            _post(root, path, body, self.timeout, reports=reports)


def _post(root: str, path: str, body: JSON, timeout: float, *, reports: bool = False) -> httpx.Response:
    """POST `body`, refusing any answer but 200 and, when the call `reports`, a `success` that is not true."""
    import httpx

    response = httpx.post(root + path, json=body, timeout=timeout)
    if response.status_code != 200 or (reports and not _succeeded(response)):
        raise RuntimeError(f"{path.split('?')[0]} answered {response.status_code}: {response.text}")
    return response


def _succeeded(response: httpx.Response) -> bool:
    """Whether the answer is a JSON object whose `success` is true."""
    if not response.headers.get("content-type", "").startswith("application/json"):
        return False
    answer = response.json()
    return isinstance(answer, dict) and answer.get("success") is True


class WeightSync(Protocol):
    """Make the engines serve `variables` as policy `version`."""

    def __call__(self, variables: Variables, version: int) -> None: ...


class Publication:
    """Publishes weights to an engine fleet under a version, for use as a scheduler's `Publisher`.

    `load(variables, version)` pushes the weights, stamps the version, then
    updates `version`. `weights` does the push (`SafetensorsReload`, or any
    `WeightSync`). `stamp`, when set, labels later calls with a version, as
    a recording gateway does (`dew.interop.harbor.Gateway.stamp`). The stamp
    runs only after every replica serves the new version, because a gateway
    stamps a call when it arrives, and a stamp ahead of a replica would claim
    weights the call was not sampled from. A failed push or stamp raises and
    leaves `version` unchanged. The constructor stamps the launch `version`,
    so a gateway left at a higher version by an earlier run cannot mislabel
    this run's first calls. Every process calls `load`, and process 0
    stamps.
    """

    def __init__(self, weights: WeightSync, *, version: int = 0, stamp: Callable[[int], None] | None = None):
        self._weights = weights
        self._stamp = stamp
        self._stamped(version)
        self._version = version

    @property
    def version(self) -> int:
        return self._version

    def load(self, variables: Variables, version: int) -> None:
        self._weights(variables, version)
        self._stamped(version)
        self._version = version

    def _stamped(self, version: int) -> None:
        stamp = self._stamp
        if stamp is not None:
            agreed("version stamp", lambda: stamp(version) if jax.process_index() == 0 else None)


# The request field that makes each engine return the sampled ids themselves.
_RETURN_IDS: dict[str, dict[str, JSON]] = {"vllm": {"return_tokens_as_token_ids": True},
                                           "sglang": {"return_token_ids": True}}


class _RequestServer:
    """The request pool, version and weight sync the HTTP rollout servers share."""

    def __init__(self, sampling: Sampling, weights: WeightSync, version: int, workers: int):
        if sampling.temperature == 0:
            raise ValueError("a rollout samples; greedy decoding gives every group member the same draw")
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive number of concurrent requests")
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

    def submit(self, prompt: Sequence[int], max_new_tokens: int, *, key: int | jax.Array) -> Future[Draw]:
        seed = key_seed(key)
        assert seed is not None
        return self._pool.submit(self._draw, _prompt(prompt), _budget(max_new_tokens), seed, self._version)

    def _draw(self, prompt: tuple[int, ...], budget: int, seed: int, version: int) -> Draw:
        raise NotImplementedError

    def load(self, variables: Variables, version: int) -> None:
        self._weights(variables, version)
        self._version = version

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


class OpenAIRolloutServer(_RequestServer):
    """Serves rollouts from the OpenAI-compatible completions endpoint of a vLLM or SGLang engine.

    Each submission is one completion request with token ids as the prompt.
    It includes the `Sampling` policy and a seed, and asks for one
    log-probability per sampled token and for the sampled ids themselves.
    Up to `workers` requests are in flight at once. The engine is
    `completion.provider`, which must be `vllm` or `sglang`.

    The reported log-probabilities are behavior likelihoods only for some
    policies. vLLM reports raw ones unless started with
    `--logprobs-mode processed_logprobs`; pass `processed_logprobs=True`
    when it is. vLLM returns a filtering policy's kept ids only on its token
    route, so use `VLLMGenerateServer` for a filtering policy. SGLang's
    `/v1/completions` reports the temperature-scaled distribution before its
    filters, so with SGLang any temperature works but no filter does, and
    `SGLANG_RETURN_ORIGINAL_LOGPROB` must stay unset. The constructor raises
    ValueError for the combinations these rules exclude, and for a
    temperature of zero. SGLang honors a seed only under
    `--enable-deterministic-inference`.

    With `routing=True`, every draw records vLLM's routed experts for
    routing replay (`dew.nn.moe.Routes`); the engine must run with
    `--enable-return-routed-experts`.
    """

    def __init__(self, completion: OpenAICompletion, sampling: Sampling, weights: WeightSync, *,
                 version: int = 0, workers: int = 64, processed_logprobs: bool = False,
                 routing: bool = False):
        engine = completion.provider
        if engine not in _RETURN_IDS:
            raise ValueError("token rollouts need provider='vllm' or 'sglang': ids, seeds and EOS controls "
                             "are engine request fields")
        if engine == "vllm" and replace(sampling, temperature=1.0).transforms():
            raise ValueError("a filtering Sampling trains on the kept ids of every draw, which vLLM returns "
                             "only on its token route; use VLLMGenerateServer")
        if engine == "vllm" and sampling.transforms() and not processed_logprobs:
            raise ValueError(
                "vLLM reports raw log-probabilities by default, which are not the behavior likelihoods "
                "of a transforming Sampling; start it with --logprobs-mode processed_logprobs and pass "
                "processed_logprobs=True, or sample at temperature one without filters")
        if engine == "sglang" and processed_logprobs:
            raise ValueError("processed_logprobs names a vLLM mode; SGLang's completions route has none")
        if engine == "sglang" and replace(sampling, temperature=1.0).transforms():
            raise ValueError(
                "SGLang's /v1/completions reports log-probabilities before top-k, top-p and min-p, "
                "which are not the behavior likelihoods of a filtering Sampling, and has no field "
                "for the filtered ones (native /generate's return_sampling_mask does); "
                "sample without filters"
            )
        if routing and engine != "vllm":
            raise ValueError(
                "routing reads vLLM's per-choice routed_experts (--enable-return-routed-experts)"
            )
        super().__init__(sampling, weights, version, workers)
        self._completion = completion
        self._return_ids = _RETURN_IDS[engine]
        self._routing = routing

    def _draw(self, prompt: tuple[int, ...], budget: int, seed: int, version: int) -> Draw:
        # The pad id shapes Dew's packed rows; it is not a request field.
        completion = self._completion(
            [list(prompt)],
            budget,
            key=seed,
            sampling=replace(self._sampling, pad_token_id=0),
            logprobs=0,
            extra_body=self._return_ids,
        )
        tokens, probabilities = completion.tokens[0], completion.log_probs[0]
        if tokens is None or probabilities is None:
            raise ValueError(f"the engine reported no sampled token ids; is it {self._completion.provider}?")
        stops = self._sampling.stops
        terminated = bool(tokens) and tokens[-1] in stops
        reason = completion.finish_reasons[0]
        if reason != ("stop" if terminated else "length") or (not terminated and len(tokens) != budget):
            raise ValueError(
                f"finish reason {reason!r} disagrees with {len(tokens)} drawn ids ending {tokens[-1:]}"
            )
        routed = completion.routed_experts[0] if completion.routed_experts else None
        if self._routing and routed is None:
            raise ValueError("vLLM returned no routed_experts; start it with --enable-return-routed-experts")
        return Draw(prompt, tokens, probabilities, None, terminated, version,
                    routed if self._routing else None).check_stops(stops)


class VLLMGenerateServer(_RequestServer):
    """Serves rollouts from vLLM's token route, `POST /inference/v1/generate`.

    This is the one vLLM route that returns, beside the sampled ids and
    their likelihoods, the ids the sampler kept for each of them
    (`GenerateResponseChoice.sampling_mask`). So a top-k or top-p policy
    trains on its recorded support (`support_log_probs`). Start the engine
    with:

    - `--enable-scale-out` (or `--tokens-only`);
    - `--return-sampling-mask`, which needs Model Runner V2 and no
      speculative decoding;
    - `--logprobs-mode processed_logprobs`, so the reported likelihoods are
      the filtered ones;
    - `--enable-return-routed-experts` when `routing` is set.

    vLLM builds the mask only under a finite top-k, so a `Sampling` without
    `top_k` raises ValueError. The route draws under temperature, top-k,
    top-p and min-p alone, so a `Sampling` that sets a repetition, presence
    or frequency penalty, `no_repeat_ngram_size`, `min_new_tokens`,
    `typical_p` or `stop` raises ValueError too.
    """

    def __init__(self, base_url: str, sampling: Sampling, weights: WeightSync, *, version: int = 0,
                 workers: int = 64, routing: bool = False, timeout: float = 600.0):
        if sampling.top_k is None:
            raise ValueError("vLLM returns a sampling mask only under a finite top-k")
        unmatched = sampling.active(("repetition_penalty", "presence_penalty", "frequency_penalty",
                                     "no_repeat_ngram_size", "min_new_tokens", "typical_p", "stop"))
        if unmatched:
            raise ValueError(f"the token route draws under temperature, top-k, top-p and min-p alone; "
                             f"{unmatched} would not shape its draws")
        super().__init__(sampling, weights, version, workers)
        self._root = base_url.rstrip("/")
        self._routing = routing
        self._timeout = timeout

    def _draw(self, prompt: tuple[int, ...], budget: int, seed: int, version: int) -> Draw:
        sampling = self._sampling
        parameters: dict[str, JSON] = {"temperature": sampling.temperature, "top_p": sampling.top_p,
                                       "top_k": sampling.top_k, "min_p": sampling.min_p,
                                       "max_tokens": budget, "seed": seed, "logprobs": 0}
        if sampling.eos_token_ids is not None:
            parameters["stop_token_ids"] = list(sampling.stops)
        response = _post(self._root, "/inference/v1/generate",
                         {"token_ids": list(prompt), "sampling_params": parameters}, self._timeout)
        (choice,) = response.json()["choices"]
        tokens = tuple(choice["token_ids"])
        probabilities = tuple(float(entry["logprob"]) for entry in choice["logprobs"]["content"])
        # vLLM defines sampling_mask in entrypoints/scale_out/token_in_token_out (at 1c0eee9).
        mask = choice.get("sampling_mask")
        if mask is None:
            raise ValueError("vLLM returned no sampling_mask; start it with --return-sampling-mask")
        routed = choice.get("routed_experts")
        if self._routing and routed is None:
            raise ValueError("vLLM returned no routed_experts; start it with --enable-return-routed-experts")
        stops = sampling.stops
        terminated = bool(tokens) and tokens[-1] in stops
        if choice["finish_reason"] != ("stop" if terminated else "length"):
            raise ValueError(f"finish reason {choice['finish_reason']!r} disagrees with the drawn ids")
        return Draw(prompt, tokens, probabilities, None, terminated, version,
                    decode_routed_experts(routed) if self._routing else None,
                    tuple(tuple(kept) for kept in mask)).check_stops(stops)
