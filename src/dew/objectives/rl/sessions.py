"""Engine-sourced sessions and the strict prefix-merge packer.

A `Call` is what any recording gateway can supply for one model call: the
prompt ids the engine read, the ids it sampled, their behavior
log-probabilities, the finish reason and the policy version served when the
request was submitted. A `Session` is one harness session: its calls in
submission order, with the verifier's verdict. A `SessionSource` turns tasks
into sessions; Harbor runners, Polar clients and the in-process episode
collector all sit behind it.

`pack` turns sessions into one fixed-shape training batch. It merges call
k + 1 into the row of call k only when call k + 1's prompt ids start with
every id of that row so far: call 1's prompt, then each sampled id and each
interstitial id (tool output, template glue) since. Any other history starts
a new chain; nothing is re-tokenized and no lenient rule is tried, because a
prompt-only prefix check splices sampled ids into a context the model did
not see (research memo section 6). Chains are then packed first-fit into
`[rows, width]` with per-chain segment ids and positions, the layout
`LMObjective.token_scores` reads for packed documents.

Only sampled ids of trained sessions carry loss mass. `COMPLETED` and
`AGENT_ERROR` sessions (the agent broke; the verifier's reward says how
badly) always train on their reward. `INFRA_ERROR` and `CANCELLED` sessions
never do: they take no rows and enter no baseline, and the scheduler retries
them. A `TRUNCATED` session follows the `truncation` policy:

- `mask` (the default) drops it like an infrastructure failure. A context,
  turn or wall-clock limit says little about the task, so agentic runs mask
  them (DeepSWE's compact filtering, SkyRL-Agent, memo section 1.3).
- `score` trains it on its verifier reward, as a completed session. Fits
  single-turn RLVR where a budget-hit completion can still hold a gradable
  answer, and keeps short-budget runs from training on nothing.
- `zero` trains it on reward 0 whatever the verifier said, a penalty on
  running long (SkyRL's `zero_reward_on_non_stop`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Protocol

import jax.numpy as jnp
import numpy as np

from dew.rl import group_advantage, rloo_advantage

IDS_KEY = "input_ids"
RESPONSE_MASK_KEY = "response_mask"
"""1 on every sampled id of a trainable session, the tokens that carry loss mass."""
OLD_LOG_PROBS_KEY = "old_log_probs"
"""The proximal policy's log-probabilities, recorded or rescored before the update.

GRPO's PPO ratio compares the current raw policy with this one. When a batch
carries none, the behavior log-probabilities stand in.
"""
BEHAVIOR_LOG_PROBS_KEY = "behavior_log_probs"
"""Actual sampling log-probabilities, including temperature/top-k and greedy selection."""
ADVANTAGES_KEY = "advantages"
SEGMENT_IDS_KEY = "text_segment_ids"
"""Which chain of its row each token belongs to, from 1; 0 is padding."""
POSITIONS_KEY = "text_positions"
"""Each token's position inside its chain, from 0."""
VERSIONS_KEY = "versions"
"""On a sampled id, the policy version its call was submitted under; -1 elsewhere."""
SESSION_INDEX_KEY = "session_index"
"""Which of `pack`'s sessions each chain token comes from; -1 on padding."""
CALL_INDEX_KEY = "call_index"
"""On a sampled id, which call of its session sampled it; -1 elsewhere."""
SESSION_WEIGHTS_KEY = "session_weights"
"""On a trainable id, one over its session's trainable-id count; 0 elsewhere.
Summed over a batch it counts sessions, so `sum(weights * terms)` over
`sum(weights)` is the mean over sessions of each session's token mean."""

FINISH_REASONS = frozenset({"stop", "tool_calls", "length", "abort"})
TRUNCATIONS = ("mask", "score", "zero")
"""What a `TRUNCATED` session trains on; see the module docstring."""
ESTIMATORS = ("group", "mean", "rloo")
"""Advantage families: `group` centres on the group mean and divides by its
deviation (GRPO), `mean` only centres (Dr.GRPO), `rloo` subtracts the mean
of the other members."""


class Status(Enum):
    """How a session ended, which decides whether it trains."""

    COMPLETED = "completed"
    TRUNCATED = "truncated"
    AGENT_ERROR = "agent_error"
    INFRA_ERROR = "infra_error"
    CANCELLED = "cancelled"

    @property
    def trainable(self) -> bool:
        """Whether a session with this status is scored and carries loss mass."""
        return self in (Status.COMPLETED, Status.AGENT_ERROR)


def _finite(message: str, value: object) -> float:
    """Parse a boundary number: any finite real, numpy's included, as a float.

    Records arrive from gateways, JSON and numpy arrays, so anything may land
    here. A bool is a flag, not a score; it and every non-number, infinity
    or nan raise a ValueError carrying `message`.
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{message}, got {value!r}")
    return float(value)


def _token_ids(name: str, ids: object) -> None:
    if not isinstance(ids, tuple) or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError(f"{name} must be a tuple of nonnegative integer token ids")


@dataclass(frozen=True)
class Call:
    """One model call as the engine served it.

    `sampled_ids` includes EOS when `finish_reason` is a natural stop, and
    `behavior_log_probs` holds the engine-reported log-probability of each
    sampled id. `version` is the served policy version when the request was
    submitted, the oldest policy that may have produced any of its ids.
    """

    prompt_ids: tuple[int, ...]
    sampled_ids: tuple[int, ...]
    behavior_log_probs: tuple[float, ...]
    finish_reason: str
    version: int

    def __post_init__(self) -> None:
        _token_ids("prompt_ids", self.prompt_ids)
        _token_ids("sampled_ids", self.sampled_ids)
        if not self.prompt_ids:
            raise ValueError("a model call reads at least one prompt id")
        if (not isinstance(self.behavior_log_probs, tuple)
                or len(self.behavior_log_probs) != len(self.sampled_ids)):
            raise ValueError("a call needs one behavior log-probability per sampled id")
        object.__setattr__(self, "behavior_log_probs", tuple(
            _finite("behavior log-probabilities must be finite numbers", value)
            for value in self.behavior_log_probs))
        if self.finish_reason not in FINISH_REASONS:
            raise ValueError(f"finish_reason must be one of {sorted(FINISH_REASONS)}, "
                             f"got {self.finish_reason!r}")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("a call's policy version is a nonnegative integer")


@dataclass(frozen=True)
class Task:
    """One unit of work a session source runs: an identity and its source-specific payload."""

    id: str
    data: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class Session:
    """One harness session: its calls in submission order and how it ended.

    `(task, group)` names the advantage group; `sample` and `attempt` tell
    members and retries apart. `reward` is the verifier's score, required
    when the status is trainable. `components` holds verifier sub-scores for
    logging and `detail` the verifier or failure provenance.
    """

    task: str
    group: str
    sample: int
    attempt: int
    calls: tuple[Call, ...]
    status: Status
    reward: float | None
    components: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, Status):
            raise TypeError("session status must be a Status")
        if not isinstance(self.calls, tuple) or not all(isinstance(call, Call) for call in self.calls):
            raise TypeError("session calls must be a tuple of Call records")
        if self.reward is not None:
            object.__setattr__(self, "reward", _finite("a session reward is a finite number or None",
                                                       self.reward))
        if self.status.trainable and self.reward is None:
            raise ValueError(f"a {self.status.name} session is scored, so it needs a reward")
        for name in ("sample", "attempt"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"session {name} must be a nonnegative integer")


class SessionSource(Protocol):
    """Anything that turns tasks into sessions.

    `submit` starts `samples` sessions of one task under the served policy
    `version` and returns one future per session; `cancel` stops sessions
    whose results are no longer wanted.
    """

    def submit(self, task: Task, samples: int, *, version: int) -> Sequence[Future[Session]]: ...

    def cancel(self, futures: Sequence[Future[Session]]) -> None: ...


@dataclass
class _Chain:
    """One append-only token chain of a session, as it is being built."""

    session: int
    tokens: list[int] = field(default_factory=list)
    behavior: list[float] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    calls: list[int] = field(default_factory=list)
    """Per token, the call that sampled it, or -1 for a prompt or interstitial id."""

    def extend(self, call: Call, index: int, start: int) -> None:
        """Append `call`'s prompt from `start`, then its sampled ids."""
        tail = len(call.prompt_ids) - start
        self.tokens.extend(call.prompt_ids[start:])
        self.behavior.extend([0.0] * tail)
        self.versions.extend([-1] * tail)
        self.calls.extend([-1] * tail)
        self.tokens.extend(call.sampled_ids)
        self.behavior.extend(call.behavior_log_probs)
        self.versions.extend([call.version] * len(call.sampled_ids))
        self.calls.extend([index] * len(call.sampled_ids))


def merges(chain: Sequence[int], call: Call) -> bool:
    """Whether `call` continues `chain` under the strict rule.

    The call's prompt must start with every id of the chain, sampled ids
    included. A prompt that only starts with the previous prompt is refused.
    """
    size = len(chain)
    return len(call.prompt_ids) >= size and tuple(call.prompt_ids[:size]) == tuple(chain)


def _chains(session: Session, index: int, width: int) -> list[_Chain]:
    """Split one session's calls into strict append-only chains no wider than `width`.

    A call that does not continue the current chain starts a new chain from
    its own full prompt. A single call wider than `width` is refused: dropping it would hide a session the
    scheduler admitted. A merged chain is never wider than its last call,
    whose prompt holds the whole chain, so fitting each call fits the chain.
    """
    built: list[_Chain] = []
    for number, call in enumerate(session.calls):
        size = len(call.prompt_ids) + len(call.sampled_ids)
        if size > width:
            raise ValueError(
                f"session {session.task}/{session.group}/{session.sample} call {number} holds "
                f"{size} ids, wider than the {width}-id row")
        current = built[-1] if built else None
        if current is not None and merges(current.tokens, call):
            current.extend(call, number, len(current.tokens))
        else:
            chain = _Chain(index)
            chain.extend(call, number, 0)
            built.append(chain)
    return built


def check_truncation(truncation: str) -> None:
    """Refuse a truncation policy `pack` does not apply."""
    if truncation not in TRUNCATIONS:
        raise ValueError(f"truncation must be one of {TRUNCATIONS}, got {truncation!r}")


def trained_reward(session: Session, truncation: str = "mask") -> float | None:
    """The reward `session` trains on under `truncation`, or None when it carries no loss."""
    if session.status.trainable:
        return session.reward
    if session.status is not Status.TRUNCATED or truncation == "mask":
        return None
    if truncation == "zero":
        return 0.0
    if session.reward is None:
        raise ValueError(f"truncation='score' trains session {session.task}/{session.group}/{session.sample} "
                         "on its reward, and it has none")
    return session.reward


def check_estimator(estimator: str) -> None:
    """Refuse an advantage family `advantages` does not compute."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")


def chains(session: Session, width: int) -> tuple[tuple[int, ...], ...]:
    """The ids of each strict append-only chain `pack` builds from `session`'s calls."""
    return tuple(tuple(chain.tokens) for chain in _chains(session, 0, width))


def advantages(sessions: Sequence[Session], estimator: str = "group", *,
               truncation: str = "mask") -> np.ndarray:
    """One advantage per session from the rewards of its `(task, group)`.

    The baseline reads the group's trained members only, each at its
    `trained_reward`. A masked member has no score to compare against, and
    a group with fewer than two trained members has no baseline, so every
    member there gets zero.
    """
    check_estimator(estimator)
    check_truncation(truncation)
    members: dict[tuple[str, str], list[int]] = {}
    rewards_of = [trained_reward(session, truncation) for session in sessions]
    for index, session in enumerate(sessions):
        if rewards_of[index] is not None:
            members.setdefault((session.task, session.group), []).append(index)
    per_session = np.zeros(len(sessions), np.float32)
    by_size: dict[int, list[list[int]]] = {}
    for group in members.values():
        if len(group) >= 2:
            by_size.setdefault(len(group), []).append(group)
    for size, groups in by_size.items():
        order = [index for group in groups for index in group]
        rewards = jnp.asarray([rewards_of[index] for index in order], jnp.float32)
        if estimator == "rloo":
            values = rloo_advantage(rewards, size)
        else:
            values = group_advantage(rewards, size, normalise_by_std=estimator == "group")
        per_session[order] = np.asarray(values, np.float32)
    return per_session


def _built(sessions: Sequence[Session], width: int, truncation: str) -> list[_Chain]:
    """The chains of every session that trains under `truncation`, in session order."""
    return [chain for index, session in enumerate(sessions) if trained_reward(session, truncation) is not None
            for chain in _chains(session, index, width)]


def _place(built: Sequence[_Chain], width: int) -> list[list[int]]:
    """Place chains first-fit in decreasing length, a stable order: per row, its chain numbers."""
    order = sorted(range(len(built)), key=lambda number: -len(built[number].tokens))
    fill: list[int] = []
    placed: list[list[int]] = []
    for number in order:
        size = len(built[number].tokens)
        row = next((row for row, used in enumerate(fill) if used + size <= width), None)
        if row is None:
            fill.append(0)
            placed.append([])
            row = len(fill) - 1
        placed[row].append(number)
        fill[row] += size
    return placed


def rows_needed(sessions: Sequence[Session], width: int, *, truncation: str = "mask") -> int:
    """How many `width`-id rows `pack` fills with `sessions`' trained chains."""
    check_truncation(truncation)
    return len(_place(_built(sessions, width, truncation), width))


def pack(sessions: Sequence[Session], width: int, *, rows: int | None = None,
         estimator: str = "group", truncation: str = "mask") -> dict[str, np.ndarray]:
    """Strictly merge each trained session's calls, then pack the chains into `[rows, width]`.

    Every array is `[rows, width]` and aligned with `input_ids`: entry t
    describes id t. `response_mask` is 1 on sampled ids alone;
    `behavior_log_probs`, `versions` and `call_index` are set on them;
    `advantages` repeats the session's advantage over its chain tokens;
    `session_weights` is described at `SESSION_WEIGHTS_KEY`. Chains are
    placed first-fit in decreasing length, a stable order, and `rows` pads
    the batch to a fixed count, refusing chains that need more
    (`rows_needed` counts them). `truncation` decides whether TRUNCATED
    sessions train, as the module docstring describes.
    """
    if type(width) is not int or width < 2:
        raise ValueError("a packed row holds at least two ids")
    if rows is not None and (type(rows) is not int or rows < 1):
        raise ValueError("rows is a positive integer, or None for as many as the chains need")
    values = advantages(sessions, estimator, truncation=truncation)
    built = _built(sessions, width, truncation)
    placed = _place(built, width)
    count = len(placed) if rows is None else rows
    if len(placed) > count:
        raise ValueError(f"the chains need {len(placed)} rows of {width} ids, more than the {rows} asked for")
    count = max(count, 1)
    shape = (count, width)
    ids = np.zeros(shape, np.int32)
    segments = np.zeros(shape, np.int32)
    positions = np.zeros(shape, np.int32)
    mask = np.zeros(shape, np.float32)
    behavior = np.zeros(shape, np.float32)
    versions = np.full(shape, -1, np.int32)
    session_index = np.full(shape, -1, np.int32)
    call_index = np.full(shape, -1, np.int32)
    advantage = np.zeros(shape, np.float32)
    trainable = np.zeros(len(sessions), np.int64)
    for chain in built:
        trainable[chain.session] += sum(1 for call in chain.calls if call >= 0)
    weights = np.zeros(shape, np.float32)
    for row, numbers in enumerate(placed):
        start = 0
        for segment, number in enumerate(numbers, start=1):
            chain = built[number]
            stop = start + len(chain.tokens)
            span = slice(start, stop)
            sampled = np.asarray(chain.calls) >= 0
            ids[row, span] = chain.tokens
            segments[row, span] = segment
            positions[row, span] = np.arange(len(chain.tokens))
            mask[row, span] = sampled
            behavior[row, span] = chain.behavior
            versions[row, span] = chain.versions
            session_index[row, span] = chain.session
            call_index[row, span] = chain.calls
            advantage[row, span] = values[chain.session]
            weights[row, span] = sampled / max(int(trainable[chain.session]), 1)
            start = stop
    return {
        IDS_KEY: ids, SEGMENT_IDS_KEY: segments, POSITIONS_KEY: positions,
        RESPONSE_MASK_KEY: mask, BEHAVIOR_LOG_PROBS_KEY: behavior, VERSIONS_KEY: versions,
        SESSION_INDEX_KEY: session_index, CALL_INDEX_KEY: call_index,
        ADVANTAGES_KEY: advantage, SESSION_WEIGHTS_KEY: weights,
    }


def sampled_values(batch: Mapping[str, np.ndarray],
                   values: Callable[[int, int], Sequence[float]]) -> np.ndarray:
    """Scatter per-call host values, one per sampled id, into the packed layout.

    `values(index, call)` returns one float per sampled id of call `call`
    of the `index`-th session handed to `pack`, such as the raw-policy
    log-probabilities an in-process sampler recorded. A call's sampled ids
    sit in one chain in order, so row-major order of its
    `(session_index, call_index)` positions is their order in the call.
    """
    session_index = np.asarray(batch[SESSION_INDEX_KEY])
    call_index = np.asarray(batch[CALL_INDEX_KEY])
    out = np.zeros(session_index.shape, np.float32)
    flat_session, flat_call = session_index.reshape(-1), call_index.reshape(-1)
    where = np.flatnonzero(flat_call >= 0)
    keys = flat_session[where].astype(np.int64) * (1 << 32) + flat_call[where]
    order = np.argsort(keys, kind="stable")
    boundaries = np.flatnonzero(np.diff(keys[order])) + 1
    flat = out.reshape(-1)
    for group in np.split(order, boundaries):
        if not group.size:
            continue
        position = where[group[0]]
        given = np.asarray(values(int(flat_session[position]), int(flat_call[position])), np.float32)
        if given.shape != (group.size,):
            raise ValueError("values must return one float per sampled id of the call")
        flat[where[group]] = given
    return out


def session_metrics(sessions: Sequence[Session], batch: Mapping[str, np.ndarray], *,
                    source: Callable[[Session], str] | None = None,
                    latencies: Sequence[float] | None = None,
                    version: int | None = None, truncation: str = "mask") -> dict[str, float]:
    """Host-side agentic telemetry for one packed batch and the sessions behind it.

    - `merge/calls_per_chain`: trainable calls over packed chains, 1.0 when
      nothing merged; `pack/fill` is the share of row slots holding ids.
    - `status/<name>`: share of sessions per status, and
      `masked/<name>`: share of all sampled ids that status masked, under
      the `truncation` policy `pack` used.
    - `reward/mean` over trained sessions at their trained reward, `reward/<source>` per
      `source(session)`, `reward/component/<name>` per verifier component.
    - `latency/p50`, `p90`, `p99`, `max` over `latencies`, seconds per session.
    - `lag/mean`, `lag/max`: `version` minus each trainable id's version.

    Trainer-versus-engine mismatch is the loss's own metric (`mismatch/*`),
    computed where the proximal policy is known.
    """
    check_truncation(truncation)
    rewards_of = [trained_reward(session, truncation) for session in sessions]
    metrics: dict[str, float] = {}
    total = max(len(sessions), 1)
    sampled = dict.fromkeys(Status, 0)
    for session in sessions:
        sampled[session.status] += sum(len(call.sampled_ids) for call in session.calls)
    everything = max(sum(sampled.values()), 1)
    for status in Status:
        metrics[f"status/{status.value}"] = sum(session.status == status for session in sessions) / total
        if not status.trainable and not (status is Status.TRUNCATED and truncation != "mask"):
            metrics[f"masked/{status.value}"] = sampled[status] / everything
    segments = np.asarray(batch[SEGMENT_IDS_KEY])
    rows = np.arange(segments.shape[0])[:, None] * (segments.shape[1] + 1) + segments
    chain_count = np.unique(rows[segments > 0]).size
    calls = sum(len(session.calls) for session, reward in zip(sessions, rewards_of, strict=True)
                if reward is not None)
    metrics["merge/calls_per_chain"] = calls / chain_count if chain_count else 0.0
    metrics["pack/fill"] = float(np.mean(segments > 0))
    scored = [(session, reward) for session, reward in zip(sessions, rewards_of, strict=True)
              if reward is not None]
    if scored:
        metrics["reward/mean"] = float(np.mean([reward for _, reward in scored]))
        by_source: dict[str, list[float]] = {}
        components: dict[str, list[float]] = {}
        for session, reward in scored:
            if source is not None:
                by_source.setdefault(source(session), []).append(float(reward))
            for name, value in session.components.items():
                components.setdefault(name, []).append(float(value))
        metrics.update({f"reward/{name}": float(np.mean(values)) for name, values in by_source.items()})
        metrics.update({f"reward/component/{name}": float(np.mean(values)) for name, values in components.items()})
    if latencies:
        seconds = np.asarray(latencies, np.float64)
        for label, quantile in (("p50", 50), ("p90", 90), ("p99", 99)):
            metrics[f"latency/{label}"] = float(np.percentile(seconds, quantile))
        metrics["latency/max"] = float(seconds.max())
    mask = np.asarray(batch[RESPONSE_MASK_KEY]) != 0
    if version is not None and mask.any():
        lag = version - np.asarray(batch[VERSIONS_KEY])[mask]
        metrics["lag/mean"], metrics["lag/max"] = float(lag.mean()), float(lag.max())
    return metrics
