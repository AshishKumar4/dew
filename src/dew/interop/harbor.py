"""Harbor trials as a harness: each sample one `harbor trials start`, its model calls recorded.

`Harbor` is the `Harness` (`dew.interop.harness`) that runs a sample as one
Harbor trial. Harbor starts the task's sandbox, installs and runs the agent
harness (mini-swe-agent by default), then runs the task's verifier. Its
agent reaches the model through the trial's gateway session
(`dew.interop.gateway`), given to it as the agent environment's
`OPENAI_BASE_URL`. Harbor is a separate service in its own environment: Dew
runs `harbor trials start` as a subprocess and reads the trial's
`result.json`; it imports nothing of Harbor.

`outcome` decides how a trial ended from Harbor's result, the harness's own
exit status and the calls, as `dew.interop.harness` lays out: Harbor's
exception names say which limit an agent hit, mini-swe-agent's exit status
says when it stopped at its own limit or its model client failed, and the
engine's own events come first.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from dew._records import JSON
from dew.objectives.rl.sessions import Status, Task

from .gateway import Recorded, engine_failure, number, refused_prompt
from .harness import Judged, Trial

HARBOR_KEY = "harbor"
"""The `Task.data` key that holds the Harbor task directory a trial runs."""

# Harbor's exception class names (harbor.trial.errors, harbor.agents.installed.base).
_TRUNCATIONS = frozenset({"AgentTimeoutError", "ContextWindowExceededError", "OutputTokenExceededError"})
_AGENT_FAILURES = frozenset({"NonZeroAgentExitCodeError"})
# mini-swe-agent's exit statuses for a run it stopped at its own step or time limit, or that the
# model call's context overflowed (mini-swe-agent records the raised exception's class name).
_HARNESS_LIMITS = frozenset({"LimitsExceeded", "TimeExceeded", "ContextWindowExceededError"})
# Exit statuses naming a litellm or OpenAI client exception: the harness's model call failed after its
# own retries. Harbor's error patterns do not match litellm's wording, so it reports only a nonzero exit.
_CLIENT_FAILURES = frozenset(
    {
        "APIConnectionError",
        "APIError",
        "APIResponseValidationError",
        "BadGatewayError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
        "APITimeoutError",
    }
)


@dataclass(frozen=True)
class _Judged:
    """What Harbor's result says of one trial: its exception, if any, and the verifier's rewards."""

    kind: str | None
    detail: str
    rewards: dict[str, float]


def _judged(trial: object) -> _Judged:
    """Read Harbor's `TrialResult` JSON, refusing any field that is not the type Harbor writes."""
    if not isinstance(trial, dict):
        raise ValueError(f"Harbor's result is a JSON object, got {type(trial).__name__}")
    failure, verdict = trial.get("exception_info") or {}, trial.get("verifier_result") or {}
    if not isinstance(failure, dict) or not isinstance(verdict, dict):
        raise ValueError(f"Harbor's result is malformed: {trial!r:.200}")
    kind = failure.get("exception_type")
    if kind is not None and not isinstance(kind, str):
        raise ValueError(f"Harbor's exception type is not a name: {kind!r}")
    reported = verdict.get("rewards") or {}
    if not isinstance(reported, dict):
        raise ValueError(f"the verifier's rewards are not an object: {reported!r}")
    rewards = {str(name): number(f"reward {name!r}", value) for name, value in reported.items()}
    if len(rewards) > 1 and "reward" not in rewards:
        raise ValueError(f"the verifier reported {sorted(rewards)} and no 'reward'")
    return _Judged(kind, f"{kind}: {failure.get('exception_message', '')}".strip() if kind else "", rewards)


def outcome(trial: JSON, recorded: Recorded, *, harness_exit: str | None = None) -> Judged:
    """How one finished trial ended: its status, reward, reward components and failure detail.

    `trial` is Harbor's `TrialResult` as JSON, `recorded` the session's calls
    and engine errors, and `harness_exit` the harness's own exit status when
    it reports one. An engine that refused an overflowing prompt truncated
    the rollout; any other engine error makes it infra, even when the harness
    retried and went on, and so does an aborted call or a failed model
    client, whatever limit the trial also hit. A result whose fields have the
    wrong JSON types is infra: nothing in it can be trusted to score.
    """
    try:
        judged = _judged(trial)
    except ValueError as error:
        return Judged(Status.INFRA_ERROR, detail=str(error))
    kind, detail, rewards = judged.kind, judged.detail, judged.rewards
    reward = rewards.get("reward", next(iter(rewards.values()), None))
    records = recorded.calls
    failed = engine_failure(recorded)
    if failed:
        return Judged(Status.INFRA_ERROR, reward, rewards, failed)
    if harness_exit in _CLIENT_FAILURES:
        return Judged(Status.INFRA_ERROR, reward, rewards,
                      f"the harness's model client failed: {harness_exit}")
    refused = refused_prompt(recorded)
    if refused:
        return Judged(Status.TRUNCATED, reward, rewards, refused)
    if (kind in _TRUNCATIONS or harness_exit in _HARNESS_LIMITS
            or (records and records[-1].finish_reason == "length")):
        return Judged(Status.TRUNCATED, reward, rewards,
                      detail or harness_exit or "the last call stopped at its length limit")
    if not records:
        return Judged(Status.INFRA_ERROR, reward, rewards,
                      detail or "no model call reached the gateway session")
    if reward is not None and kind is None:
        return Judged(Status.COMPLETED, reward, rewards)
    if reward is not None and kind in _AGENT_FAILURES:
        return Judged(Status.AGENT_ERROR, reward, rewards, detail)
    return Judged(Status.INFRA_ERROR, reward, rewards, detail or "the verifier reported no reward")


@dataclass(frozen=True)
class Harbor:
    """Runs each sample as one Harbor trial (a `Harness` for `HarnessSource`).

    - `harbor` is the Harbor executable, installed in its own environment.
    - `agent` and `model` are its `--agent` and `--model`.
    - `trials` is the directory trials are written under.
    - `environment` is passed to the harness as agent environment (`--ae`),
      beside the per-trial `OPENAI_BASE_URL` that names the session.
    - `arguments` are further `harbor trials start` options, such as the
      environment provider, timeouts and agent kwargs. Harbor bounds the
      environment build, the agent's setup and run, and the verifier itself
      (`--agent-timeout`, `--verifier-timeout`, `--timeout-multiplier`), but
      no option of its own bounds the whole trial: `HarnessSource`'s
      `trial_timeout` does.

    A task names its Harbor task directory under `data[HARBOR_KEY]`. The
    agent must speak the OpenAI chat API through `OPENAI_BASE_URL`, as
    Harbor's mini-swe-agent does.
    """

    harbor: str | os.PathLike[str]
    model: str
    trials: os.PathLike[str]
    agent: str = "mini-swe-agent"
    environment: Mapping[str, str] = field(default_factory=dict)
    arguments: Sequence[str] = ()

    def check(self, task: Task) -> None:
        directory = task.data.get(HARBOR_KEY)
        if not isinstance(directory, (str, os.PathLike)) or not Path(directory).is_dir():
            raise ValueError(f"task {task.id!r} names no Harbor task directory under data[{HARBOR_KEY!r}]")

    def trial(self, task: Task, url: str, name: str) -> Trial:
        directory = task.data[HARBOR_KEY]
        assert isinstance(directory, (str, os.PathLike)), "check refused a task without one"
        agent = {**self.environment, "OPENAI_BASE_URL": url}
        return Trial((os.fspath(self.harbor), "trials", "start", "-a", self.agent, "-m", self.model,
                      *self.arguments, "-p", os.fspath(directory), "--trial-name", name,
                      "--trials-dir", os.fspath(self.trials),
                      *(part for key, value in agent.items() for part in ("--ae", f"{key}={value}"))),
                     Path(self.trials) / name)

    def judge(self, task: Task, trial: Trial, log: str, returncode: int, recorded: Recorded) -> Judged:
        trial_result = trial.artifacts / "result.json"
        if not trial_result.is_file():
            return Judged(Status.INFRA_ERROR,
                          detail=f"harbor exited {returncode} with no result: {log[-2000:]}")
        try:
            harness_exit = _harness_exit(trial.artifacts)
        except ValueError as error:
            return Judged(Status.INFRA_ERROR, detail=f"unreadable harness trajectory: {error}")
        try:
            reported = json.loads(trial_result.read_text())
        except json.JSONDecodeError as error:
            return Judged(Status.INFRA_ERROR, detail=f"unreadable result.json: {error}")
        return outcome(reported, recorded, harness_exit=harness_exit)


def _harness_exit(trial: Path) -> str | None:
    """mini-swe-agent's own exit status for the trial, when its trajectory
    is there; ValueError when unreadable."""
    path = trial / "agent" / "mini-swe-agent.trajectory.json"
    if not path.is_file():
        return None
    trajectory = json.loads(path.read_text())
    summary = trajectory.get("info") if isinstance(trajectory, dict) else None
    status = summary.get("exit_status") if isinstance(summary, dict) else None
    if status is not None and not isinstance(status, str):
        raise ValueError(f"mini-swe-agent's exit status is not a name: {status!r}")
    return status


__all__ = ["HARBOR_KEY", "Harbor", "outcome"]
