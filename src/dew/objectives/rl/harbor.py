"""Harbor trials as a rollout source, each model call recorded by rllm-model-gateway.

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
`prompt_token_ids` and `choices[0].token_ids`, which the gateway records;
SGLang's chat route lists them under `sglext.input_ids` and
`sglext.output_ids`, which the gateway keeps only in the raw response, so
they are read there.

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

import itertools
import json
import os
import re
import signal
import subprocess
import threading
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .rollouts import Call, Rollout, Status, Task

if TYPE_CHECKING:
    import httpx

HARBOR_KEY = "harbor"
"""`Task.data[HARBOR_KEY]` is the Harbor task directory the trial runs."""

# Harbor's exception class names (harbor.trial.errors, harbor.agents.installed.base).
_TRUNCATIONS = frozenset({"AgentTimeoutError", "ContextWindowExceededError", "OutputTokenExceededError"})
_AGENT_FAILURES = frozenset({"NonZeroAgentExitCodeError"})
# mini-swe-agent's exit statuses for a run it stopped at its own step or time limit.
_HARNESS_LIMITS = frozenset({"LimitsExceeded", "TimeExceeded"})
_SESSION = re.compile(r"[A-Za-z0-9._:/-]+")


class Gateway:
    """The parts of an rllm-model-gateway a rollout source reads.

    `url` is the gateway's root as Dew reaches it; `sandbox_url` is the same
    gateway as the sandboxes reach it (a Docker bridge address, a cluster
    service name), which is what a harness's base URL is built from.
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
            raise ValueError(f"session ids are URL-path safe: letters, digits and ._:/-; got {session!r}")
        return f"{self.sandbox_url}/sessions/{session}/v1"

    def traces(self, session: str) -> list[dict[str, Any]]:
        response = self._client.get(f"{self.url}/sessions/{session}/traces")
        response.raise_for_status()
        return response.json()

    def forget(self, session: str) -> None:
        self._client.delete(f"{self.url}/sessions/{session}").raise_for_status()


def _ids(name: str, values: object) -> tuple[int, ...]:
    if not isinstance(values, list) or not all(type(value) is int for value in values):
        raise ValueError(f"the trace's {name} are not a list of token ids")
    return tuple(values)


def calls(traces: Sequence[Mapping[str, Any]], *, unstamped: int) -> tuple[Call, ...]:
    """The gateway's traces of one session as `Call`s, in submission order.

    A trace records its arrival indirectly: `timestamp` is when the answer
    was stored and `latency_ms` how long the engine took, so the calls are
    ordered by their difference. A trace without a version stamp, from a
    gateway no publication has stamped, takes `unstamped`, the version the
    session was submitted under: no later push can have served it anything
    older. A trace without ids or with one likelihood too few or too many
    raises `ValueError`; training on it would mean re-tokenizing text.
    """
    ordered = sorted(traces, key=lambda trace: float(trace["timestamp"]) - float(trace["latency_ms"]) / 1000)
    records = []
    for trace in ordered:
        prompt, sampled = trace.get("prompt_token_ids") or [], trace.get("completion_token_ids") or []
        extension = ((trace.get("raw_response") or {}).get("sglext") or {})
        if not prompt and extension.get("input_ids"):
            prompt = extension["input_ids"]
        if not sampled and extension.get("output_ids"):
            sampled = extension["output_ids"][0]
        if not prompt:
            raise ValueError("a trace carries no prompt ids: the engine was not asked for them or cannot list them")
        version = trace.get("weight_version")
        records.append(Call(_ids("prompt ids", prompt), _ids("sampled ids", sampled),
                            tuple(float(value) for value in trace.get("logprobs") or ()),
                            str(trace.get("finish_reason")), unstamped if version is None else int(version)))
    return tuple(records)


def _reward(rewards: Mapping[str, float], key: str) -> float:
    if key in rewards:
        return float(rewards[key])
    if len(rewards) == 1:
        return float(next(iter(rewards.values())))
    raise ValueError(f"the verifier reported {sorted(rewards)} and no {key!r}")


def outcome(result: Mapping[str, Any], records: tuple[Call, ...], *, harness_exit: str | None = None,
            reward_key: str = "reward") -> tuple[Status, float | None, dict[str, float], str]:
    """Status, reward, reward components and failure detail of one finished trial.

    `result` is Harbor's `TrialResult` as JSON, `records` the session's calls
    and `harness_exit` the harness's own exit status when it reports one.
    """
    failure = result.get("exception_info") or {}
    kind = failure.get("exception_type")
    detail = f"{kind}: {failure.get('exception_message', '')}".strip() if kind else ""
    rewards = dict((result.get("verifier_result") or {}).get("rewards") or {})
    try:
        reward = _reward(rewards, reward_key) if rewards else None
    except ValueError as error:
        return Status.INFRA_ERROR, None, rewards, str(error)
    if any(call.finish_reason == "abort" for call in records):
        return Status.INFRA_ERROR, reward, rewards, "the engine aborted a call"
    if (kind in _TRUNCATIONS or harness_exit in _HARNESS_LIMITS
            or (records and records[-1].finish_reason == "length")):
        return Status.TRUNCATED, reward, rewards, detail or harness_exit or "the last call stopped at its length limit"
    if not records:
        return Status.INFRA_ERROR, reward, rewards, detail or "no model call reached the gateway session"
    if reward is not None and kind is None:
        return Status.COMPLETED, reward, rewards, ""
    if reward is not None and kind in _AGENT_FAILURES:
        return Status.AGENT_ERROR, reward, rewards, detail
    return Status.INFRA_ERROR, reward, rewards, detail or "the verifier reported no reward"


class HarborSource:
    """A `RolloutSource` that runs each sample as one Harbor trial behind a recording gateway.

    `harbor` is the Harbor executable (in its own environment), `agent` and
    `model` its `--agent` and `--model`, and `trials` the directory trials
    are written under. `environment` is passed to the harness as agent
    environment (`--ae`), beside the per-trial `OPENAI_BASE_URL` that
    carries the session; `arguments` are further `harbor trials start`
    options (environment provider, timeouts, agent kwargs). `workers` trials
    run at once; Harbor's own sandbox limits apply inside each. The harness
    must speak the OpenAI chat API through `OPENAI_BASE_URL`, as Harbor's
    mini-swe-agent does.

    Sessions are named `{task}:{group}:{sample}`, where `group` is unique to
    one `submit` across runs. Every future resolves to a `Rollout`: a
    failure of Harbor, the sandbox, the gateway or the engine is an
    `INFRA_ERROR` rollout, not an exception, and a cancelled trial is a
    `CANCELLED` one. `attempt` is always 0: a retry is a fresh `submit`,
    relabelled by the scheduler that owns group identity.
    """

    def __init__(self, gateway: Gateway, *, harbor: str | os.PathLike[str], model: str, trials: os.PathLike[str],
                 agent: str = "mini-swe-agent", environment: Mapping[str, str] | None = None,
                 arguments: Sequence[str] = (), workers: int = 8, reward_key: str = "reward"):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive number of concurrent trials")
        self._gateway = gateway
        self._command = [os.fspath(harbor), "trials", "start", "-a", agent, "-m", model, *arguments]
        self._environment = dict(environment or {})
        self._trials = Path(trials)
        self._reward_key = reward_key
        self._run = uuid.uuid4().hex[:8]
        self._serial = itertools.count()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dew-harbor-trial")
        self._lock = threading.Lock()
        self._running: dict[Future[Rollout], subprocess.Popen[str]] = {}
        self._cancelled: set[Future[Rollout]] = set()

    def submit(self, task: Task, samples: int, *, version: int) -> list[Future[Rollout]]:
        if type(samples) is not int or samples < 1:
            raise ValueError("a submission runs at least one sample")
        directory = task.data.get(HARBOR_KEY)
        if not isinstance(directory, (str, os.PathLike)) or not Path(directory).is_dir():
            raise ValueError(f"task {task.id!r} names no Harbor task directory under data[{HARBOR_KEY!r}]")
        group = f"{self._run}-{next(self._serial)}"
        futures: list[Future[Rollout]] = []
        for sample in range(samples):
            future: Future[Rollout] = Future()
            futures.append(future)
            self._pool.submit(self._trial, future, task, Path(directory), group, sample, version)
        return futures

    def cancel(self, futures: Sequence[Future[Rollout]]) -> None:
        """Interrupt the named trials; Harbor tears their sandboxes down, and each resolves `CANCELLED`."""
        with self._lock:
            self._cancelled.update(futures)
            running = [self._running[future] for future in futures if future in self._running]
        for process in running:
            os.killpg(process.pid, signal.SIGINT)

    def close(self) -> None:
        with self._lock:
            pending = list(self._running)
        self.cancel(pending)
        self._pool.shutdown(wait=True, cancel_futures=False)

    def _trial(self, future: Future[Rollout], task: Task, directory: Path, group: str, sample: int,
               version: int) -> None:
        session = f"{task.id}:{group}:{sample}"
        name = f"dew-{group}-{sample}"

        def rollout(status: Status, records: tuple[Call, ...] = (), reward: float | None = None,
                    components: Mapping[str, float] | None = None, detail: str = "") -> Rollout:
            return Rollout(task.id, group, sample, 0, records, status, reward, components or {}, detail)

        try:
            with self._lock:
                if future in self._cancelled:
                    future.set_result(rollout(Status.CANCELLED))
                    return
                process = subprocess.Popen(
                    [*self._command, "-p", os.fspath(directory), "--trial-name", name,
                     "--trials-dir", os.fspath(self._trials),
                     *itertools.chain.from_iterable(("--ae", f"{key}={value}") for key, value in
                                                    {**self._environment,
                                                     "OPENAI_BASE_URL": self._gateway.session(session)}.items())],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                self._running[future] = process
            log, _ = process.communicate()
            with self._lock:
                del self._running[future]
                cancelled = future in self._cancelled
            trial = self._trials / name
            try:
                records = calls(self._gateway.traces(session), unstamped=version)
            except Exception as error:
                future.set_result(rollout(Status.CANCELLED if cancelled else Status.INFRA_ERROR,
                                          detail=f"{trial}: unusable gateway traces: {error}"))
                return
            if cancelled:
                future.set_result(rollout(Status.CANCELLED, records, detail=str(trial)))
                return
            if not (trial / "result.json").is_file():
                future.set_result(rollout(Status.INFRA_ERROR, records,
                                          detail=f"harbor exited {process.returncode} with no result: {log[-2000:]}"))
                return
            status, reward, components, detail = outcome(
                json.loads((trial / "result.json").read_text()), records,
                harness_exit=_harness_exit(trial), reward_key=self._reward_key)
            future.set_result(rollout(status, records, reward, components, f"{trial}: {detail}".rstrip(": ")))
            self._gateway.forget(session)
        except BaseException as error:
            if not future.done():
                future.set_exception(error)


def _harness_exit(trial: Path) -> str | None:
    """mini-swe-agent's own exit status for the trial, when its trajectory is there."""
    path = trial / "agent" / "mini-swe-agent.trajectory.json"
    if not path.is_file():
        return None
    try:
        return (json.loads(path.read_text()).get("info") or {}).get("exit_status")
    except (OSError, ValueError):
        return None
