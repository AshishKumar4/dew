"""Harbor trials as a session source, each model call recorded by rllm-model-gateway.

`HarborSource` runs one Harbor trial per sample: Harbor starts the task's
sandbox, installs and runs the harness (mini-swe-agent by default), then
runs the task's verifier. The harness reaches the model only through a
recording gateway, `rllm-model-gateway`, at a base URL that carries the
session in its path, `/sessions/{session}/v1`. The gateway forwards each
call to a vLLM or SGLang replica, asks it for token ids and
log-probabilities, and records them under that session. Attribution rides
in the URL, not in a header a CLI harness may drop (memo section 5.7).

Harbor and the gateway are separate services in their own environments:
Dew starts `harbor trials start` as a subprocess, reads the trial's
`result.json`, and reads the session's traces over HTTP. Nothing of either
is imported here. The gateway must persist traces before it answers
(`sync_traces: true`), so a session's traces are complete once its harness
has exited.

`calls` maps gateway traces onto `Call` records: the engine's prompt ids,
sampled ids and behavior log-probabilities as recorded, in submission order,
with the version the gateway stamped when the request arrived (the
publication stamps it, see `SafetensorsReload`). vLLM lists the ids as
`prompt_token_ids` and `choices[0].token_ids`, which the gateway records.
SGLang 0.5.20 answers the gateway's `return_token_ids` with
`choices[0].prompt_token_ids` and `choices[0].response_token_ids`; the
gateway extracts the prompt ids but not `response_token_ids`, so those are
read from the raw response it keeps. `sglext.input_ids`/`output_ids`, which
SGLang fills only when asked, are read as a last resort.

`outcome` decides how a trial ended from Harbor's result and the calls:

- the verifier scored a trial whose harness exited cleanly: `COMPLETED`;
- the harness exited nonzero and the verifier still scored it:
  `AGENT_ERROR` with that score;
- the agent ran out of time, context or output budget, the harness hit its
  own step limit, or the last call stopped at its length limit:
  `TRUNCATED`, masked rather than scored;
- anything else: a sandbox, gateway, engine or verifier failure, a trace
  without ids or likelihoods, an aborted call, or a session with no call:
  `INFRA_ERROR`, which the scheduler retries and never trains on.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import itertools
import json
import logging
import os
import queue
import re
import secrets
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dew.records import JSON

from .sessions import Call, Session, Status, Task

if TYPE_CHECKING:
    import httpx

_logger = logging.getLogger(__name__)

HARBOR_KEY = "harbor"
"""`Task.data[HARBOR_KEY]` is the Harbor task directory the trial runs."""

# Harbor's exception class names (harbor.trial.errors, harbor.agents.installed.base).
_TRUNCATIONS = frozenset({"AgentTimeoutError", "ContextWindowExceededError", "OutputTokenExceededError"})
_AGENT_FAILURES = frozenset({"NonZeroAgentExitCodeError"})
# mini-swe-agent's exit statuses for a run it stopped at its own step or time limit, or that the
# model call's context overflowed (mini-swe-agent records the raised exception's class name).
_HARNESS_LIMITS = frozenset({"LimitsExceeded", "TimeExceeded", "ContextWindowExceededError"})
# Exit statuses naming a litellm or OpenAI client exception: the harness's model call failed after its
# own retries. Harbor's error patterns do not match litellm's wording, so it reports only a nonzero exit.
_CLIENT_FAILURES = frozenset({"APIConnectionError", "APIError", "APIResponseValidationError", "BadGatewayError",
                              "InternalServerError", "RateLimitError", "ServiceUnavailableError", "Timeout",
                              "APITimeoutError"})
_SESSION = re.compile(r"[A-Za-z0-9._:-]+")


class Gateway:
    """The parts of an rllm-model-gateway a session source reads.

    `url` is the gateway's root as Dew reaches it; `sandbox_url` is where the
    sandboxes reach it, which is what a harness's base URL is built from.

    rllm-model-gateway (3b40c37) has no authentication, and one port serves
    the model proxy beside `GET /sessions`, every session's traces, `POST
    /traces/query`, session deletes, `POST /admin/workers` and `POST
    /admin/weight_version`. A sandbox runs the policy's own commands, so a
    policy that reaches that port can read its group members' transcripts,
    reset the version stamp the staleness bound reads, or register a worker
    that returns fabricated ids and likelihoods. Therefore:

    - `url` is on an interface only Dew reaches (bind the gateway to
      loopback or a private trainer network);
    - `sandbox_url` is a reverse proxy in front of it that forwards only
      `POST /sessions/<session>/v1/chat/completions` (nginx `location ~
      ^/sessions/[^/]+/v1/chat/completions$`) and answers 403 to everything
      else, on its own address;
    - the task's agent phase allows that address and nothing else: Harbor's
      `[agent] network_mode = "allowlist"` with `allowed_hosts` naming the
      proxy, or `--allow-agent-host`, on a provider that supports allowlists
      (Harbor's `tasks/network-policy` page; Docker needs nftables `fib`
      support), and `[verifier] network_mode = "no-network"`. Harbor's
      allowlist filters by host, not path, which is why the proxy is needed.

    `sandbox_url` defaults to `url` only for trusted harnesses and tests.
    """

    def __init__(self, url: str, *, sandbox_url: str | None = None, timeout: float = 30.0,
                 client: httpx.Client | None = None):
        import httpx

        self.url = url.rstrip("/")
        self.sandbox_url = (sandbox_url or url).rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    def session(self, session: str) -> str:
        """The OpenAI base URL a harness uses so its calls are recorded under `session`."""
        if not _SESSION.fullmatch(session):
            raise ValueError(f"session ids are URL-path safe: letters, digits and ._:-; got {session!r}")
        return f"{self.sandbox_url}/sessions/{session}/v1"

    def traces(self, session: str) -> list[JSON]:
        response = self._client.get(f"{self.url}/sessions/{session}/traces")
        response.raise_for_status()
        traces = response.json()
        if not isinstance(traces, list):
            raise ValueError(f"the gateway's traces are a JSON list, got {type(traces).__name__}")
        return traces

    def ready(self, timeout: float, *, poll: float = 2.0) -> None:
        """Wait until the gateway routes to a healthy worker that answers, or raise after `timeout` seconds.

        Ready means `/health/workers` counts a healthy worker and `GET /v1/models`, proxied through
        the gateway to an engine, answers 200.

        rllm-model-gateway marks a worker dead after three failed health checks, as happens to an
        engine still loading when the gateway starts, and until a later check revives it every
        proxied call answers a plain-text 500 that leaves no trace. A session submitted then would
        fail for the gateway's reasons, not the policy's.
        """
        import httpx

        url = f"{self.url}/health/workers"
        deadline, seen = time.monotonic() + timeout, "no answer"
        while True:
            try:
                response = self._client.get(url)
                health = _object(response.json()) if response.status_code == 200 else {}
                seen = f"{response.status_code} {response.text[:500]}"
                if isinstance(healthy := health.get("healthy"), int) and healthy > 0:
                    # The gateway calls a worker healthy until its first probe fails, so also see the
                    # engine answer through it; a request without a session leaves no trace.
                    listing = self._client.get(f"{self.url}/v1/models")
                    seen = f"GET /v1/models: {listing.status_code} {listing.text[:500]}"
                    if listing.status_code == 200:
                        return
            except (httpx.HTTPError, ValueError) as error:
                seen = f"{type(error).__name__}: {error}"
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"the gateway reported no healthy worker within {timeout:g} s at {url} (last: {seen}); "
                    "check that the engines are up and registered, or raise ready_timeout")
            time.sleep(poll)

    def stamp(self, version: int) -> None:
        """Make the gateway label the calls it records from now on with `version` (a `Publication` stamp)."""
        response = self._client.post(f"{self.url}/admin/weight_version", json={"weight_version": version})
        if response.status_code != 200 or _object(response.json()).get("weight_version") != version:
            raise RuntimeError(f"the gateway refused version {version}: {response.status_code} {response.text}")

    def forget(self, session: str) -> None:
        self._client.delete(f"{self.url}/sessions/{session}").raise_for_status()


def _object(value: JSON) -> dict[str, JSON]:
    """`value` when it is a JSON object, else an empty one: an absent or null field reads as empty."""
    return value if isinstance(value, dict) else {}


def _number(name: str, value: JSON) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not a number: {value!r}")
    return float(value)


def _ids(name: str, values: JSON) -> tuple[int, ...]:
    if not isinstance(values, list) or not all(type(value) is int for value in values):
        raise ValueError(f"the trace's {name} are not a list of token ids")
    return tuple(value for value in values if type(value) is int)


@dataclass(frozen=True)
class Recorded:
    """One session as the gateway recorded it: its model calls, and the engine errors it answered instead."""

    calls: tuple[Call, ...]
    errors: tuple[str, ...]


def calls(traces: Sequence[JSON], *, unstamped: int) -> Recorded:
    """The gateway's traces of one session as `Call`s in submission order, and its engine errors.

    A trace records its arrival indirectly: `timestamp` is when the answer
    was stored and `latency_ms` how long the engine took, so the calls are
    ordered by their difference. A trace without a version stamp, from a
    gateway no publication has stamped, takes `unstamped`, the version the
    session was submitted under: no later push can have served it anything
    older. The gateway also records an engine's error reply (a prompt past
    the context length, an engine fault): such a trace carries `error` in its
    raw response and is an event of the session, returned as its message, not
    a call. A successful reply without ids, with one likelihood too few or
    too many, or with a field of the wrong JSON type raises `ValueError`;
    training on it would mean guessing or re-tokenizing text.
    """
    checked = []
    for trace in traces:
        if not isinstance(trace, dict):
            raise ValueError(f"a gateway trace is a JSON object, got {type(trace).__name__}")
        raw = trace.get("raw_response")
        if raw is not None and not isinstance(raw, dict):
            raise ValueError(f"a trace's raw response is a JSON object, got {type(raw).__name__}")
        arrival = _number("timestamp", trace.get("timestamp")) - _number("latency_ms", trace.get("latency_ms")) / 1000
        checked.append((arrival, trace, _object(raw)))
    checked.sort(key=lambda entry: entry[0])
    records, errors = [], []
    for _, trace, raw in checked:
        # vLLM answers {"error": {"message": ...}}; SGLang {"object": "error", "message": ...}.
        error = raw.get("error") or (raw.get("message") if raw.get("object") == "error" else None)
        if error:
            errors.append(str(_object(error).get("message", error)))
            continue
        prompt, sampled = trace.get("prompt_token_ids") or [], trace.get("completion_token_ids") or []
        # SGLang lists the ids on the choice as prompt_token_ids and response_token_ids; the gateway
        # extracts only vLLM's token_ids, so they stay in the raw response.
        choices = raw.get("choices")
        choice = _object(choices[0]) if isinstance(choices, list) and choices else {}
        extension = _object(raw.get("sglext"))
        outputs = extension.get("output_ids")
        prompt = prompt or choice.get("prompt_token_ids") or extension.get("input_ids") or []
        sampled = sampled or choice.get("response_token_ids") or (
            outputs[0] if isinstance(outputs, list) and outputs else [])
        if not prompt:
            raise ValueError("a trace carries no prompt ids: the engine was not asked for them or cannot list them")
        version = trace.get("weight_version")
        if version is not None and type(version) is not int:
            raise ValueError(f"a trace's weight version is an integer, got {version!r}")
        likelihoods = trace.get("logprobs") or []
        if not isinstance(likelihoods, list):
            raise ValueError("a trace's log-probabilities are a list")
        records.append(Call(_ids("prompt ids", prompt), _ids("sampled ids", sampled),
                            tuple(_number("a log-probability", value) for value in likelihoods),
                            str(trace.get("finish_reason")), unstamped if version is None else version))
    return Recorded(tuple(records), tuple(errors))


# How vLLM 0.30.0 ("This model's maximum context length is ...") and SGLang 0.5.20 ("The input (N
# tokens) is longer than the model's context length (M tokens).") word a prompt past the context length;
# vLLM's input processor, which token-id prompts reach, says "... is longer than the maximum model length of M".
_OVERFLOW = re.compile(r"maximum context length|longer than the model's context length"
                       r"|longer than the maximum model length", re.IGNORECASE)


def _reward(rewards: Mapping[str, float], key: str) -> float:
    if key in rewards:
        return rewards[key]
    if len(rewards) == 1:
        return next(iter(rewards.values()))
    raise ValueError(f"the verifier reported {sorted(rewards)} and no {key!r}")


def outcome(result: JSON, records: tuple[Call, ...], *, errors: Sequence[str] = (),
            harness_exit: str | None = None, reward_key: str = "reward") -> tuple[Status, float | None, dict[str, float], str]:
    """Status, reward, reward components and failure detail of one finished trial.

    `result` is Harbor's `TrialResult` as JSON, `records` the session's calls,
    `errors` the engine errors the gateway recorded for it, and `harness_exit`
    the harness's own exit status when it reports one. An engine that refused
    an overflowing prompt truncated the rollout; any other engine error makes
    it infra, even when the harness retried and went on. A result whose
    fields have the wrong JSON types is infra: nothing in it can be trusted
    to score.
    """
    trial = _object(result)
    failure, verdict = trial.get("exception_info"), trial.get("verifier_result")
    if not isinstance(result, dict) or not isinstance(failure, (dict, type(None))) \
            or not isinstance(verdict, (dict, type(None))):
        return Status.INFRA_ERROR, None, {}, f"Harbor's result is malformed: {result!r:.200}"
    kind = _object(failure).get("exception_type")
    kind = kind if isinstance(kind, str) else None
    detail = f"{kind}: {_object(failure).get('exception_message', '')}".strip() if kind else ""
    reported = _object(verdict).get("rewards") or {}
    try:
        if not isinstance(reported, dict):
            raise ValueError(f"the verifier's rewards are not an object: {reported!r}")
        rewards = {name: _number(f"reward {name!r}", value) for name, value in reported.items()}
        reward = _reward(rewards, reward_key) if rewards else None
    except ValueError as error:
        return Status.INFRA_ERROR, None, {}, str(error)
    if any(_OVERFLOW.search(error) for error in errors):
        return Status.TRUNCATED, reward, rewards, f"the engine refused a prompt: {errors[0]}"
    if errors:
        return Status.INFRA_ERROR, reward, rewards, f"the engine answered an error: {errors[0]}"
    if any(call.finish_reason == "abort" for call in records):
        return Status.INFRA_ERROR, reward, rewards, "the engine aborted a call"
    if (kind in _TRUNCATIONS or harness_exit in _HARNESS_LIMITS
            or (records and records[-1].finish_reason == "length")):
        return Status.TRUNCATED, reward, rewards, detail or harness_exit or "the last call stopped at its length limit"
    if not records:
        return Status.INFRA_ERROR, reward, rewards, detail or "no model call reached the gateway session"
    if harness_exit in _CLIENT_FAILURES:
        return Status.INFRA_ERROR, reward, rewards, f"the harness's model client failed: {harness_exit}"
    if reward is not None and kind is None:
        return Status.COMPLETED, reward, rewards, ""
    if reward is not None and kind in _AGENT_FAILURES:
        return Status.AGENT_ERROR, reward, rewards, detail
    return Status.INFRA_ERROR, reward, rewards, detail or "the verifier reported no reward"


class HarborSource:
    """A `SessionSource` that runs each sample as one Harbor trial behind a recording gateway.

    `harbor` is the Harbor executable (in its own environment), `agent` and
    `model` its `--agent` and `--model`, and `trials` the directory trials
    are written under. `environment` is passed to the harness as agent
    environment (`--ae`), beside the per-trial `OPENAI_BASE_URL` that
    carries the session; `arguments` are further `harbor trials start`
    options (environment provider, timeouts, agent kwargs). `workers` trials
    run at once; Harbor's own sandbox limits apply inside each. The harness
    must speak the OpenAI chat API through `OPENAI_BASE_URL`, as Harbor's
    mini-swe-agent does.

    Gateway sessions are named `{task}:{group}:{sample}:{token}`, where
    `group` is unique to one `submit` across runs and `token` is 128 secret
    bits, so a sandbox can address only the session it was handed. Every
    future resolves to a `Session`: a failure of Harbor, the sandbox, the
    gateway or the engine is an `INFRA_ERROR` session, not an exception, and
    a cancelled trial is a
    `CANCELLED` one. `attempt` is always 0: a retry is a fresh `submit`,
    relabelled by the scheduler that owns group identity.

    The first `submit` waits, up to `ready_timeout` seconds, until the gateway
    reports a healthy worker (`Gateway.ready`), so a source started beside
    engines that are still loading launches no trial the gateway would answer
    with a traceless 500.

    Use it as a context manager, or call `close`, to cancel every trial on
    the way out. A trainer that exits without either (an uncaught
    KeyboardInterrupt) still stops its trials: trials run on daemon threads,
    and an atexit hook cancels queued trials and interrupts, then kills after
    `grace`, the running ones.
    """

    def __init__(self, gateway: Gateway, *, harbor: str | os.PathLike[str], model: str, trials: os.PathLike[str],
                 agent: str = "mini-swe-agent", environment: Mapping[str, str] | None = None,
                 arguments: Sequence[str] = (), workers: int = 8, reward_key: str = "reward", grace: float = 60.0,
                 ready_timeout: float = 900.0, ready_poll: float = 2.0):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive number of concurrent trials")
        if not grace > 0:
            raise ValueError("grace is a positive number of seconds")
        self._grace = grace
        self._ready_timeout, self._ready_poll = ready_timeout, ready_poll
        self._ready = False
        self._gateway = gateway
        self._command = [os.fspath(harbor), "trials", "start", "-a", agent, "-m", model, *arguments]
        self._environment = dict(environment or {})
        self._trials = Path(trials)
        self._reward_key = reward_key
        self._run = uuid.uuid4().hex[:8]
        self._serial = itertools.count()
        # Daemon workers, not a ThreadPoolExecutor: concurrent.futures joins its pool before atexit
        # hooks run, which would wait out every running trial and start every queued one on exit.
        self._queue: queue.SimpleQueue[Callable[[], None] | None] = queue.SimpleQueue()
        self._workers = [threading.Thread(target=self._work, name=f"dew-harbor-trial-{index}", daemon=True)
                         for index in range(workers)]
        for worker in self._workers:
            worker.start()
        self._lock = threading.Lock()
        # One record per submitted, unresolved future, queued or running; a done-callback drops it.
        self._records: dict[Future[Session], _Trial] = {}
        # Trials run in their own process groups, so a Ctrl-C of the trainer does not reach them: at
        # interpreter exit this hook cancels every record and stops every live trial.
        self._exit_hook = functools.partial(_stop_all, self._records, self._lock, grace)
        atexit.register(self._exit_hook)

    def submit(self, task: Task, samples: int, *, version: int) -> list[Future[Session]]:
        if type(samples) is not int or samples < 1:
            raise ValueError("a submission runs at least one sample")
        directory = task.data.get(HARBOR_KEY)
        if not isinstance(directory, (str, os.PathLike)) or not Path(directory).is_dir():
            raise ValueError(f"task {task.id!r} names no Harbor task directory under data[{HARBOR_KEY!r}]")
        if not _SESSION.fullmatch(task.id):
            raise ValueError(f"a Harbor task id is URL-path safe (letters, digits and ._-); got {task.id!r}")
        with self._lock:
            if not self._ready:
                self._gateway.ready(self._ready_timeout, poll=self._ready_poll)
                self._ready = True
        group = f"{self._run}-{next(self._serial)}"
        futures: list[Future[Session]] = []
        for sample in range(samples):
            future: Future[Session] = Future()
            with self._lock:
                self._records[future] = _Trial()
            future.add_done_callback(self._forget_record)
            futures.append(future)
            self._queue.put(functools.partial(self._trial, future, task, Path(directory), group, sample, version))
        return futures

    def _forget_record(self, future: Future[Session]) -> None:
        with self._lock:
            self._records.pop(future, None)

    def cancel(self, futures: Sequence[Future[Session]]) -> None:
        """Interrupt the named trials; Harbor tears their sandboxes down, and each resolves `CANCELLED`."""
        with self._lock:
            for future in futures:
                record = self._records.get(future)
                if record is None or record.cancelled:
                    continue
                record.cancelled = True
                if record.process is not None and record.process.poll() is None:
                    _signal(record.process, signal.SIGINT)
                    record.timer = _timer(self._grace, self._escalate, record, signal.SIGTERM)

    def _escalate(self, record: _Trial, sent: signal.Signals) -> None:
        """Harbor ignored the last signal for a whole grace period: send the next, SIGTERM then SIGKILL."""
        with self._lock:
            if record.process is None or record.process.poll() is not None:
                return
            _logger.warning("Harbor trial %d outlived its grace; sending %s", record.process.pid, sent.name)
            _signal(record.process, sent)
            if sent is signal.SIGTERM:
                record.timer = _timer(self._grace, self._escalate, record, signal.SIGKILL)

    def close(self) -> None:
        """Cancel every unresolved trial, queued or running, and wait for the running ones to tear down."""
        with self._lock:
            pending = list(self._records)
        self.cancel(pending)
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join()
        atexit.unregister(self._exit_hook)

    def _work(self) -> None:
        while (trial := self._queue.get()) is not None:
            trial()

    def __enter__(self) -> HarborSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _trial(self, future: Future[Session], task: Task, directory: Path, group: str, sample: int,
               version: int) -> None:
        # The token keeps a sandbox from addressing its group siblings' sessions through the gateway.
        session = f"{task.id}:{group}:{sample}:{secrets.token_hex(16)}"
        name = f"dew-{group}-{sample}"

        def ended(status: Status, records: tuple[Call, ...] = (), reward: float | None = None,
                    components: Mapping[str, float] | None = None, detail: str = "") -> Session:
            return Session(task.id, group, sample, 0, records, status, reward, components or {}, detail)

        try:
            with self._lock:
                record = self._records[future]
                process = None if record.cancelled else subprocess.Popen(
                    [*self._command, "-p", os.fspath(directory), "--trial-name", name,
                     "--trials-dir", os.fspath(self._trials),
                     *itertools.chain.from_iterable(("--ae", f"{key}={value}") for key, value in
                                                    {**self._environment,
                                                     "OPENAI_BASE_URL": self._gateway.session(session)}.items())],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                record.process = process
            if process is None:
                # Resolved outside the lock: the future's done-callback takes it.
                future.set_result(ended(Status.CANCELLED))
                return
            log, _ = process.communicate()
            with self._lock:
                record.process = None
                cancelled = record.cancelled
                if record.timer is not None:
                    record.timer.cancel()
            try:
                future.set_result(self._verdict(ended, session, self._trials / name, log, process.returncode,
                                                cancelled, version))
            finally:
                # The session's traces leave the gateway on every exit path; the future already has its verdict.
                try:
                    self._gateway.forget(session)
                except Exception as error:
                    _logger.warning("could not delete gateway session %s: %s", session, error)
        except BaseException as error:
            if not future.done():
                future.set_exception(error)

    def _verdict(self, ended: Callable[..., Session], session: str, trial: Path, log: str, returncode: int,
                 cancelled: bool, version: int) -> Session:
        try:
            recorded = calls(self._gateway.traces(session), unstamped=version)
        except Exception as error:
            return ended(Status.CANCELLED if cancelled else Status.INFRA_ERROR,
                           detail=f"{trial}: unusable gateway traces: {error}")
        records = recorded.calls
        if cancelled:
            return ended(Status.CANCELLED, records, detail=str(trial))
        if not (trial / "result.json").is_file():
            return ended(Status.INFRA_ERROR, records, detail=f"harbor exited {returncode} with no result: {log[-2000:]}")
        status, reward, components, detail = outcome(
            json.loads((trial / "result.json").read_text()), records, errors=recorded.errors,
            harness_exit=_harness_exit(trial), reward_key=self._reward_key)
        return ended(status, records, reward, components, f"{trial}: {detail}".rstrip(": "))


@dataclass
class _Trial:
    """What `cancel` needs of one submitted sample: whether it was cancelled and its running Harbor."""

    cancelled: bool = False
    process: subprocess.Popen[str] | None = None
    timer: threading.Timer | None = None


def _timer(seconds: float, action: Callable[..., None], *arguments: object) -> threading.Timer:
    """A daemon timer: an escalation still pending must not hold the interpreter open at exit."""
    timer = threading.Timer(seconds, action, arguments)
    timer.daemon = True
    timer.start()
    return timer


def _signal(process: subprocess.Popen[str], sent: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, sent)


def _stop_all(records: dict[Future[Session], _Trial], lock: threading.Lock, grace: float) -> None:
    """At interpreter exit: cancel every trial, interrupt the live ones, and kill any still alive after `grace`."""
    with lock:
        live = []
        for record in records.values():
            record.cancelled = True
            if record.process is not None and record.process.poll() is None:
                live.append(record.process)
    for process in live:
        _signal(process, signal.SIGINT)
    deadline = time.monotonic() + grace
    for process in live:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(max(0.0, deadline - time.monotonic()))
        if process.poll() is None:
            _signal(process, signal.SIGKILL)


def _harness_exit(trial: Path) -> str | None:
    """mini-swe-agent's own exit status for the trial, when its trajectory is there."""
    path = trial / "agent" / "mini-swe-agent.trajectory.json"
    if not path.is_file():
        return None
    try:
        status = _object(_object(json.loads(path.read_text())).get("info")).get("exit_status")
        return status if isinstance(status, str) else None
    except (OSError, ValueError):
        return None
