"""Rollout servers: one submission interface over Dew's own server and OpenAI-compatible engines.

A `RolloutServer` takes one prompt of token ids at a time and resolves a
future `Draw` with the sampled ids, their likelihoods and the policy version
the request was submitted under. `load(variables, version)` pushes new
weights while generation keeps running, so in-flight requests finish on
whichever weights are resident when each token is drawn, and a draw's version
is the oldest weights that may have produced any of its tokens.

A draw ends where its policy ends one (`check_ending`): on one of the
`Sampling`'s EOS ids, on the id that completes one of its stop strings, or
at its budget. vLLM and SGLang both return the ids through the one that
completes a stop string and say which string matched; Dew's own server keeps
them too (`dew.sampling.decoding.StopStrings`).

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
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from dew.coordination import agreed, collective_host, stop_at_exit
from dew.nn.inputs import key_seed, request_key, token_id
from dew.objectives.base import Variables, thaw
from dew.records import JSON
from dew.sampling.decoding import EndOfSequence, StopStrings
from dew.sampling.text import Generation, Sampling

from .clients import (
    OpenAICompletion,
    _engine_stops,
    decode_routed_experts,
    listed_ids,
    matched_stop,
    reported_log_probs,
)
from .engine import settled
from .serving import Server

if TYPE_CHECKING:
    import httpx

    from dew.interop.pretrained import Pretrained


class InfrastructureError(RuntimeError):
    """A rollout's infrastructure failed: a connection, an engine, a sandbox, a runtime or a service.

    A session source turns it, like any `OSError`, into an `INFRA_ERROR`
    session, which the scheduler retries; any other error is broken code or
    a broken contract and fails the rollout. The remote rollout servers
    raise it when the engine cannot be reached, times out, or answers 5xx
    or 429, and Dew's sandbox runners when they cannot run a program. An
    environment, verifier or reward raises it for a failure of its own
    external boundary that is not an `OSError`, such as an httpx transport
    error.
    """


@dataclass(frozen=True)
class Draw:
    """One sampled continuation of one prompt.

    `tokens` are the sampled actions, with the id that ended the draw
    included when it terminated: an EOS id, or the id completing a stop
    string (`check_ending`). `behavior_log_probs` is the likelihood of each action
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
            raise ValueError("a terminated draw ends on the id that ended it")
        if self.support is not None and len(self.support) != len(self.tokens):
            raise ValueError("a draw's support holds one set of kept ids per drawn token")

    def checked(self, sampling: Sampling) -> Draw:
        """Return this draw after checking that it ends as `sampling` ends one (`check_ending`)."""
        check_ending(self.tokens, self.terminated, sampling)
        return self

    def response(self, sampling: Sampling) -> tuple[int, ...]:
        """The drawn ids a reward reads decoded (`response`)."""
        return response(self.tokens, sampling)


def response(tokens: Sequence[int], sampling: Sampling) -> tuple[int, ...]:
    """A draw's ids without the EOS id it ended on, as a reward reads them decoded: a stop string's ids
    stay, as the engines return them and Dew's own tasks keep them."""
    ids = tuple(tokens)
    return ids[:-1] if ids and ids[-1] in sampling.stops else ids


def through_stop(tokens: Sequence[int], decode: Callable[[Sequence[int]], str],
                 strings: Sequence[str]) -> int:
    """The length of the shortest prefix of `tokens` whose decoded text holds one of `strings`: where a
    draw that a stop string ended ends, as the engines end one in the step whose text first holds it.

    Every prefix is decoded whole, from the first, since a tokenizer's text for a longer prefix need
    not extend a shorter one's (Transformers' cleanup turns " ." into "."), and ids that decode to
    nothing may sit inside a string. Raises ValueError where no prefix holds one: the engine and the
    tokenizer disagree."""
    for count in range(1, len(tokens) + 1):
        text = decode(tokens[:count])
        if any(string in text for string in strings):
            return count
    raise ValueError(f"the engine ended the draw on one of {list(strings)}, which no prefix of its "
                     f"{len(tokens)} ids decodes to; is `decode` the served model's tokenizer?")

def check_ending(tokens: Sequence[int], terminated: bool, sampling: Sampling) -> None:
    """Raise ValueError unless a draw of `tokens` that `terminated` or not ends as `sampling` ends one.

    An EOS id (`Sampling.stops`) ends a draw, so it is only ever the draw's last id, and a draw that
    ends on one terminated. A stop string (`Sampling.stop`) ends a draw on the id that completes it,
    which may be any id, so a terminated draw that ends on no EOS id needs a policy with stop
    strings. A draw that did not terminate ran to its budget, or a criterion that truncates it cut it.
    """
    stops = sampling.stops
    if any(token in stops for token in tokens[:-1]):
        raise ValueError("a draw continues past one of its policy's EOS ids")
    on_eos = bool(tokens) and tokens[-1] in stops
    if on_eos and not terminated:
        raise ValueError("a draw that ends on one of its policy's EOS ids terminated")
    if terminated and not on_eos and not sampling.stop:
        raise ValueError("a terminated draw ends on one of its policy's EOS ids or completes one of its "
                         "stop strings, and this policy has no stop strings")


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
    ids = tuple(token_id(token, where="a rollout prompt") for token in prompt)
    if not ids:
        raise ValueError("a rollout prompt is a nonempty row of token ids")
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

    One lock orders weight loads with the server's steps, so a load happens
    between two steps. A submission waits for a load in progress but not for a
    step. The server takes it while a step runs and seats it at the next one.
    The server's own sampling policy and transforms decide the draws, and a
    draw keeps both the raw and the behavior likelihoods the server records.
    The stepping thread starts at the first submission, so constructing the
    server starts nothing. If the program ends without `close`, an exit hook
    stops the stepping after the step in progress.

    A draw ends on one of the policy's EOS ids or on the id completing one of
    its stop strings, and `Draw.terminated` tells those from a draw cut at its
    budget or by a criterion that truncates it (`MaxNewTokens`). A server
    whose task ends rows on a criterion of its own besides those, which a
    draw's policy cannot name, raises ValueError.
    """

    def __init__(self, server: Server, *, version: int = 0):
        ending, sampling = server.stopping.ending, server.sampling
        named = [criterion for criterion in ending if isinstance(criterion, EndOfSequence)
                 and set(np.asarray(criterion.eos).reshape(-1).tolist()) == set(sampling.stops)]
        strings = [criterion for criterion in ending if isinstance(criterion, StopStrings)]
        # The policy's EOS criterion and the one its stop strings compile to, and nothing beside them.
        if (len(named), len(strings)) != (len(sampling.criteria()), int(bool(sampling.stop))) or (
                len(ending) != len(named) + len(strings)):
            raise ValueError("a rollout's draws end on its policy's EOS ids or stop strings, or are cut "
                             "short; this server's task also ends rows on criteria of its own, which no "
                             "draw's policy names")
        self._server = server
        self._version = version
        # Reentrant, so a draw's callback may load weights from inside a step.
        self._lock = threading.RLock()
        self._loading = threading.Lock()
        self._futures = threading.Lock()
        self._arrived = threading.Event()
        self._outstanding: set[Future[Draw]] = set()
        self._closed = False
        self._failure: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._withdraw: Callable[[], None] | None = None

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
        with self._futures:
            if self._failure is not None:
                raise RuntimeError("the rollout server stopped") from self._failure
            if self._closed:
                raise RuntimeError("the rollout server is closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="dew-rollout-server", daemon=True)
                self._thread.start()
                # A step under way at exit runs to its end inside jaxlib, so the bound is a step's, not a
                # poll's. The stepping thread stops after the step in progress, which may be a long first
                # compile; it holds no external call, so exit waits for it rather than finalizing under it.
                self._withdraw = stop_at_exit(self._thread, self._stop, timeout=None)
            self._outstanding.add(future)
        try:
            with self._loading:
                ticket = self._server.submit(np.asarray(ids, np.int32), budget, key=request)
                version = self._version
        except BaseException:
            self._claimed(future)
            raise

        def resolve(done: Future[Generation[np.ndarray]]) -> None:
            if not self._claimed(future):  # failed with the stepping thread or by `close`
                return
            try:
                draw = _native_draw(ids, done.result(), version).checked(self.sampling)
            except BaseException as failure:
                # The generation's own failure, or a draw that fails its checks, such as non-finite
                # likelihoods: the claimed future takes it, since a done-callback's error reaches no one.
                settled(future, failure)
                return
            settled(future, draw)

        ticket.add_done_callback(resolve)
        self._arrived.set()
        return future

    def _claimed(self, future: Future[Draw]) -> bool:
        """Whether this call takes `future` from the outstanding draws. The one
        caller that takes it completes it, and the rest leave it."""
        with self._futures:
            if future not in self._outstanding:
                return False
            self._outstanding.remove(future)
            return True

    def _detached(self, failure: BaseException | None = None) -> list[Future[Draw]]:
        """Take every outstanding draw, recording `failure` as the server's."""
        with self._futures:
            if failure is not None:
                self._failure = failure
            detached = list(self._outstanding)
            self._outstanding.clear()
        return detached

    def load(self, variables: Variables, version: int) -> None:
        with self._lock, self._loading:
            self._server.reload(thaw(variables))
            self._version = version

    def _run(self) -> None:
        try:
            while True:
                # Cleared before the look, so a submission after it wakes the wait.
                self._arrived.clear()
                with self._lock:
                    if self._closed:
                        return
                    busy = bool(self._server.occupancy or self._server.queued)
                    if busy:
                        self._server.step()
                if not busy:
                    self._arrived.wait()
        except BaseException as failure:
            for future in self._detached(failure):
                settled(future, failure)

    def _stop(self) -> None:
        with self._futures:
            self._closed = True
        self._arrived.set()

    def close(self) -> None:
        self._stop()
        with self._futures:
            thread, withdraw = self._thread, self._withdraw
        if thread is not None and withdraw is not None:
            thread.join()
            withdraw()
        cancelled = RuntimeError("the rollout server closed before this draw finished")
        for future in self._detached():
            settled(future, cancelled)


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
    `dtype` first. The source's writers replace each file whole (`dew.files.replacing`) and
    publish a sharded index last, so no engine reads a half-written file. `engines` are the
    replicas' root URLs, not their `/v1` API URLs, and `timeout` bounds each
    HTTP call in seconds. The push fails if a call answers with any status
    but 200, or, for a call that reports its outcome, with a `success` that
    is not true. Both engines report some failures as a 200 with
    `{"success": false}`.

    For vLLM (`engine="vllm"`, v0.30.0, `VLLM_SERVER_DEV_MODE=1`), the push
    reloads the directory through `/collective_rpc` inside the publication
    lifecycle it shares with `NCCLPush`: every replica pauses with
    `mode=wait`, takes the weights, resets its prefix cache, sets the weight
    version and resumes, and a phase that fails on any replica stops the
    push before the next phase, leaving the paused replicas paused until a
    later push succeeds. In-flight draws therefore finish entirely on
    the old weights, and no cached prefix outlives them. For SGLang
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
        self.source.save(self.directory, variables=variables)

    def __call__(self, variables: Variables, version: int) -> None:
        self.write(variables)
        agreed("weight publication", lambda: self._publish(version) if jax.process_index() == 0 else None)

    def _publish(self, version: int) -> None:
        def reload(root: str) -> None:
            _post(root, "/collective_rpc", {"method": "reload_weights"}, self.timeout)

        def load(root: str) -> None:
            _post(root, "/update_weights_from_disk", {
                "model_path": str(Path(self.directory).resolve()), "flush_cache": True,
                "abort_all_requests": False, "weight_version": str(version)}, self.timeout, reports=True)

        if self.engine == "vllm":
            _vllm_publication(self.engines, version, self.timeout, lambda: _on_every(self.engines, reload))
        else:
            _refused(version, self.engines, _on_every(self.engines, load))


def _on_every(engines: Sequence[str], call: Callable[[str], None]) -> dict[str, BaseException]:
    """Run `call` on every replica root at once, and return each failure by its replica."""
    failures: dict[str, BaseException] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(engines)), thread_name_prefix="dew-weight-push") as pool:
        for root, run in [(root, pool.submit(call, root)) for root in engines]:
            failure = run.exception()
            if failure is not None:
                failures[root] = failure
    return failures


def _refused(version: int, engines: Sequence[str], failures: Mapping[str, BaseException]) -> None:
    """Raise an error naming every replica in `failures` that did not take `version`, if any."""
    if failures:
        raise RuntimeError(f"version {version} did not reach {len(failures)} of {len(engines)} replicas: "
                           + "; ".join(f"{root}: {failure}" for root, failure in failures.items()))


def _vllm_publication(engines: Sequence[str], version: int, timeout: float,
                      deliver: Callable[[], Mapping[str, BaseException]], *, stamped: bool = False) -> None:
    """Publish `version` to vLLM replicas (v0.30.0, `VLLM_SERVER_DEV_MODE=1`) around a transport's `deliver`.

    Every replica pauses with `mode=wait`, so its in-flight draws finish on
    the old weights and no new one starts. `deliver` then loads the weights
    into every replica and returns the failures by replica. Each replica
    resets its prefix cache, takes the version through
    `/update_weight_version` unless the transport `stamped` it while
    delivering, and resumes. The reset and the stamp must answer
    `{"success": true}`, which vLLM withholds from one that did not happen.

    A phase that fails on any replica raises before the next phase starts,
    naming every replica it failed on. So no replica resumes on weights the
    others may not hold, and a replica the push paused stays paused until a
    later push succeeds.
    """
    def pause(root: str) -> None:
        _post(root, "/pause?mode=wait", None, timeout)

    def reset(root: str) -> None:
        _post(root, "/reset_prefix_cache", None, timeout, reports=True)

    def stamp(root: str) -> None:
        _post(root, "/update_weight_version", {"new_version": str(version)}, timeout, reports=True)

    def resume(root: str) -> None:
        _post(root, "/resume", None, timeout)

    _refused(version, engines, _on_every(engines, pause))
    _refused(version, engines, deliver())
    for step in (reset, resume) if stamped else (reset, stamp, resume):
        _refused(version, engines, _on_every(engines, step))


def _post(root: str, path: str, body: JSON, timeout: float, *, reports: bool = False) -> httpx.Response:
    """POST `body`, refusing any answer but 200 and, when the call `reports`, a `success` that is not true.

    An engine that cannot be reached, times out, or answers 5xx or 429 raises
    `InfrastructureError`; any other refusal raises RuntimeError.
    """
    import httpx

    route = path.split("?")[0]
    try:
        response = httpx.post(root.rstrip("/") + path, json=body, timeout=timeout)
    except httpx.TransportError as error:
        raise InfrastructureError(f"{route} failed: {type(error).__name__}: {error}") from error
    if response.status_code >= 500 or response.status_code == 429:
        raise InfrastructureError(f"{route} answered {response.status_code}: {response.text}")
    if response.status_code != 200 or (reports and not _succeeded(response)):
        raise RuntimeError(f"{route} answered {response.status_code}: {response.text}")
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
    a recording gateway does (`dew.interop.gateway.Gateway.stamp`). The stamp
    runs only after every replica serves the new version, because a gateway
    stamps a call when it arrives, and a stamp ahead of a replica would claim
    weights the call was not sampled from. A failed push or stamp raises and
    leaves `version` unchanged. The launch `version` is stamped when the
    version is first read, which a scheduler does before it submits a draw
    under it, so a gateway left at a higher version by an earlier run cannot
    mislabel this run's first calls, and constructing the publication touches
    no gateway. Every process calls `load`, and process 0 stamps.
    """

    def __init__(self, weights: WeightSync, *, version: int = 0, stamp: Callable[[int], None] | None = None):
        self._weights = weights
        self._stamp = stamp
        self._version = version
        # One lock orders the launch stamp against a load, so the gateway always ends at the version
        # the engines hold.
        self._published = threading.Lock()
        self._launch: int | None = version

    @property
    def version(self) -> int:
        with self._published:
            if self._launch is not None:
                self._stamped(self._launch)
                self._launch = None
            return self._version

    def load(self, variables: Variables, version: int) -> None:
        with self._published:
            self._weights(variables, version)
            self._stamped(version)
            self._launch = None
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
    `SGLANG_RETURN_ORIGINAL_LOGPROB` must stay unset. An SGLang draw streams,
    drawn as the policy without its EOS ids and cut at the first of them
    (`_streamed`). Both engines end a draw on the policy's stop strings
    themselves, matched on the text they decode, and return its ids through
    the one completing the string, but vLLM keeps a speculative block's ids
    past it (vllm/v1/engine/detokenizer.py:118-131 at v0.30.0). So a policy
    with stop strings takes `decode`, the policy's tokenizer's, and its draws
    are cut at the first id whose text completes one (`through_stop`). The
    constructor raises
    ValueError for the combinations these rules exclude, and for a
    temperature of zero. SGLang honors a seed only under
    `--enable-deterministic-inference`.

    With `routing=True`, every draw records vLLM's routed experts for
    routing replay (`dew.nn.protocols.Routes`); the engine must run with
    `--enable-return-routed-experts`.
    """

    def __init__(self, completion: OpenAICompletion, sampling: Sampling, weights: WeightSync, *,
                 version: int = 0, workers: int = 64, processed_logprobs: bool = False,
                 routing: bool = False, decode: Callable[[Sequence[int]], str] | None = None):
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
        if sampling.stop and decode is None:
            raise ValueError("a stop string ends a draw where its decoded text completes it; pass "
                             "decode=, the policy's tokenizer's, to find that id")
        super().__init__(sampling, weights, version, workers)
        self._completion = completion
        self._return_ids = _RETURN_IDS[engine]
        self._routing = routing
        self._decode = decode

    def _draw(self, prompt: tuple[int, ...], budget: int, seed: int, version: int) -> Draw:
        import openai

        try:
            if self._completion.provider == "sglang":
                return self._streamed(prompt, budget, seed, version)
            completion = self._completion(
                [list(prompt)],
                budget,
                key=seed,
                sampling=replace(self._sampling, stop=()),
                logprobs=0,
                extra_body=self._return_ids,
                **self._strings(),
            )
        except (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError) as error:
            # Unreachable, timed out, or 5xx or 429 after the client's own retries.
            raise InfrastructureError(f"the completion failed: {type(error).__name__}: {error}") from error
        tokens, probabilities = completion.tokens[0], completion.log_probs[0]
        if tokens is None or probabilities is None:
            raise ValueError(f"the engine reported no sampled token ids; is it {self._completion.provider}?")
        (answer,) = completion.responses
        assert isinstance(answer, openai.types.Completion)
        (choice,) = answer.choices
        matched = matched_stop(choice.to_dict())
        terminated = _finished(completion.finish_reasons[0], matched, tokens, budget, self._sampling)
        routed = completion.routed_experts[0] if completion.routed_experts else None
        if self._routing and routed is None:
            raise ValueError("vLLM returned no routed_experts; start it with --enable-return-routed-experts")
        if terminated and matched in self._sampling.stop:
            assert self._decode is not None
            # A speculative block's ids past the completing one (the class docstring) are dropped, with
            # their likelihoods and the routing of the ids forwarded after it.
            kept = through_stop(tokens, self._decode, self._sampling.stop)
            tokens, probabilities = tokens[:kept], probabilities[:kept]
            routed = None if routed is None else routed[:len(prompt) + kept - 1]
        return Draw(prompt, tokens, probabilities, None, terminated, version,
                    routed if self._routing else None).checked(self._sampling)

    def _strings(self) -> dict[str, JSON]:
        """The request field that ends a draw on the policy's stop strings, which the engine matches on
        the text it decodes. Dew's text completions refuse `Sampling.stop`, since the engine cuts the
        string from the text it returns; a rollout reads the ids, which run through it."""
        return {"stop": list(self._sampling.stop)} if self._sampling.stop else {}

    def _streamed(self, prompt: tuple[int, ...], budget: int, seed: int, version: int) -> Draw:
        """One SGLang draw, streamed, and cut at the policy's first EOS id.

        SGLang ends a request that names stop ids on its model's EOS ids as
        well, and its one field that turns those off, `ignore_eos`, turns off
        the named stops too (sglang/srt/managers/schedule_batch.py:1707-1729
        at v0.5.20). So the policy is drawn as one without EOS ids, which ends
        on its budget or a stop string, and the draw is cut where the policy
        stops: the same distribution, since no later id shapes an earlier one.
        `ignore_eos` leaves the stop strings to end the request
        (`_check_str_based_finish`, beside the token check it skips, at
        v0.5.20). The draw is streamed so that closing the stream at the stop
        ends the request there, rather than when the budget runs out: SGLang
        aborts a request whose client disconnects.
        """
        import openai

        stops = self._sampling.stops
        tokens: list[int] = []
        probabilities: list[float] = []
        reason: str | None = None
        matched: str | int | None = None
        ended: int | None = None
        unstopped = replace(self._sampling, eos_token_ids=None, stop=())
        with self._completion.stream([list(prompt)], budget, key=seed, sampling=unstopped, logprobs=0,
                                     extra_body=self._return_ids, **self._strings()) as chunks:
            for chunk in _engine_chunks(chunks, openai.APIError):
                # A chunk's ids may run ahead of its log-probabilities (SGLang slices the two apart),
                # so each is read whole and they are matched by position.
                for entry in chunk.choices:
                    choice = entry.to_dict()
                    ids, reported = listed_ids(choice), reported_log_probs(choice)
                    if ids is None:
                        raise ValueError("SGLang streamed no sampled token ids; is return_token_ids set?")
                    if ended is None:
                        stopped = next((index for index, token in enumerate(ids) if token in stops), None)
                        ended = None if stopped is None else len(tokens) + stopped
                    tokens.extend(ids)
                    probabilities.extend(() if reported is None else reported[1])
                    reason = entry.finish_reason or reason
                    reported_stop = matched_stop(choice)
                    matched = matched if reported_stop is None else reported_stop
                if ended is not None and len(probabilities) > ended:
                    break
        if ended is not None:
            tokens, probabilities = tokens[:ended + 1], probabilities[:ended + 1]
        terminated = ended is not None or _finished(reason, matched, tokens, budget, self._sampling)
        if ended is None and terminated:
            # SGLang returns a string's ids through the completing one (`_locate_str_stop_finished_len`
            # at v0.5.20); the cut holds it to the one rule all servers keep.
            assert self._decode is not None
            kept = through_stop(tokens, self._decode, self._sampling.stop)
            tokens, probabilities = tokens[:kept], probabilities[:kept]
        if len(probabilities) != len(tokens):
            raise ValueError(f"SGLang streamed {len(tokens)} ids and {len(probabilities)} log-probabilities")
        return Draw(prompt, tuple(tokens), tuple(probabilities), None, terminated,
                    version).checked(self._sampling)


def _finished(reason: str | None, matched: str | int | None, tokens: Sequence[int], budget: int,
              sampling: Sampling) -> bool:
    """Whether an engine's draw of `tokens` terminated, read from its finish reason and the stop it says
    matched (vLLM's `stop_reason`, SGLang's `matched_stop`): a stopped draw ends on an EOS id of the
    policy or on one of its stop strings, and a draw that did not stop ran to its budget."""
    if reason == "length" and len(tokens) == budget:
        return False
    if reason == "stop" and ((tokens and tokens[-1] in sampling.stops) or matched in sampling.stop):
        return True
    raise ValueError(f"finish reason {reason!r}, matching {matched!r}, disagrees with {len(tokens)} drawn "
                     f"ids ending {list(tokens[-1:])} under a budget of {budget}")


def _engine_chunks[Chunk](chunks: Iterable[Chunk], failure: type[Exception]) -> Iterator[Chunk]:
    """`chunks` as a stream yields them; a `failure` while it streams (its connection dropped, or
    the engine's error event, which SGLang sends for an abort, a timeout or an out-of-memory) is
    an infrastructure failure."""
    try:
        yield from chunks
    except failure as error:
        raise InfrastructureError(f"the stream failed: {type(error).__name__}: {error}") from error


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
                                       "max_tokens": budget, "seed": seed, "logprobs": 0,
                                       **_engine_stops(sampling, "vllm")}
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
        terminated = _finished(choice["finish_reason"], matched_stop(choice), tokens, budget, sampling)
        return Draw(prompt, tokens, probabilities, None, terminated, version,
                    decode_routed_experts(routed) if self._routing else None,
                    tuple(tuple(kept) for kept in mask)).checked(sampling)
