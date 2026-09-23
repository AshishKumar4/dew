"""Engine-sourced rollouts and the strict prefix-merge packer.

A `Call` is what any recording gateway can supply for one model call: the
prompt ids the engine read, the ids it sampled, their behavior
log-probabilities, the finish reason and the policy version served when the
request was submitted. A `Rollout` is one harness session's calls in
submission order, with the verifier's verdict. A `RolloutSource` turns tasks
into rollouts; Harbor runners, Polar clients and the in-process episode
collector all sit behind it.

`pack` turns rollouts into one fixed-shape training batch. It merges call
k + 1 into the row of call k only when call k + 1's prompt ids start with
every id of that row so far: call 1's prompt, then each sampled id and each
interstitial id (tool output, template glue) since. Any other history starts
a new chain; nothing is re-tokenized and no lenient rule is tried, because a
prompt-only prefix check splices sampled ids into a context the model did
not see (research memo section 6). Chains are then packed first-fit into
`[rows, width]` with per-chain segment ids and positions, the layout
`LMObjective.token_scores` reads for packed documents.

Only sampled ids of trainable rollouts carry loss mass. A rollout is
trainable when its verifier scored it: `COMPLETED`, or `AGENT_ERROR` (the
agent broke; the verifier's reward says how badly). `TRUNCATED`,
`INFRA_ERROR` and `CANCELLED` rollouts are masked, so they take no rows and
enter no baseline: a truncation is not scored zero and an infrastructure
failure is retried by the scheduler, never trained on.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np

from dew.rl import group_advantage, rloo_advantage

IDS_KEY = "input_ids"
RESPONSE_MASK_KEY = "response_mask"
"""1 on every sampled id of a trainable rollout, the tokens that carry loss mass."""
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
ROLLOUT_INDEX_KEY = "rollout_index"
"""Which of `pack`'s rollouts each chain token comes from; -1 on padding."""
CALL_INDEX_KEY = "call_index"
"""On a sampled id, which call of its rollout sampled it; -1 elsewhere."""
ROLLOUT_WEIGHTS_KEY = "rollout_weights"
"""On a trainable id, one over its rollout's trainable-id count; 0 elsewhere.
Summed over a batch it counts rollouts, so `sum(weights * terms)` over
`sum(weights)` is the mean over rollouts of each rollout's token mean."""

FINISH_REASONS = frozenset({"stop", "tool_calls", "length", "abort"})
ESTIMATORS = ("group", "mean", "rloo")
"""Advantage families: `group` centres on the group mean and divides by its
deviation (GRPO), `mean` only centres (Dr.GRPO), `rloo` subtracts the mean
of the other members."""


class Status(Enum):
    """How a rollout ended, which decides whether it trains."""

    COMPLETED = "completed"
    TRUNCATED = "truncated"
    AGENT_ERROR = "agent_error"
    INFRA_ERROR = "infra_error"
    CANCELLED = "cancelled"

    @property
    def trainable(self) -> bool:
        """Whether a rollout with this status is scored and carries loss mass."""
        return self in (Status.COMPLETED, Status.AGENT_ERROR)


def _real(value: float) -> bool:
    """Whether `value` is a finite number; a bool is a flag, not a score."""
    return type(value) is not bool and math.isfinite(value)


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
        if not all(_real(value) for value in self.behavior_log_probs):
            raise ValueError("behavior log-probabilities must be finite numbers")
        if self.finish_reason not in FINISH_REASONS:
            raise ValueError(f"finish_reason must be one of {sorted(FINISH_REASONS)}, "
                             f"got {self.finish_reason!r}")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("a call's policy version is a nonnegative integer")


@dataclass(frozen=True)
class Task:
    """One unit of work a rollout source runs: an identity and its source-specific payload."""

    id: str
    data: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class Rollout:
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
            raise TypeError("rollout status must be a Status")
        if not isinstance(self.calls, tuple) or not all(isinstance(call, Call) for call in self.calls):
            raise TypeError("rollout calls must be a tuple of Call records")
        if self.reward is not None and not _real(self.reward):
            raise ValueError("a rollout reward is a finite number or None")
        if self.status.trainable and self.reward is None:
            raise ValueError(f"a {self.status.name} rollout is scored, so it needs a reward")
        for name in ("sample", "attempt"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"rollout {name} must be a nonnegative integer")


class RolloutSource(Protocol):
    """Anything that turns tasks into rollouts.

    `submit` starts `samples` sessions of one task under the served policy
    `version` and returns one future per session; `cancel` stops sessions
    whose results are no longer wanted.
    """

    def submit(self, task: Task, samples: int, *, version: int) -> Sequence[Future[Rollout]]: ...

    def cancel(self, futures: Sequence[Future[Rollout]]) -> None: ...


@dataclass
class _Chain:
    """One append-only token chain of a rollout, as it is being built."""

    rollout: int
    tokens: list[int] = field(default_factory=list)
    behavior: list[float] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    calls: list[int] = field(default_factory=list)
    """Per token, the call that sampled it, or -1 for a prompt or interstitial id."""
    merged: int = 0
    """How many calls this chain holds."""

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
        self.merged += 1


def merges(chain: Sequence[int], call: Call) -> bool:
    """Whether `call` continues `chain` under the strict rule.

    The call's prompt must start with every id of the chain, sampled ids
    included. A prompt that only starts with the previous prompt is refused.
    """
    size = len(chain)
    return len(call.prompt_ids) >= size and tuple(call.prompt_ids[:size]) == tuple(chain)


def chains(rollout: Rollout, index: int, width: int) -> list[_Chain]:
    """Split one rollout's calls into strict append-only chains no wider than `width`.

    A call that does not continue the current chain starts a new chain from
    its own full prompt. A single call
    wider than `width` is refused: dropping it would hide a rollout the
    scheduler admitted. A merged chain is never wider than its last call,
    whose prompt holds the whole chain, so fitting each call fits the chain.
    """
    built: list[_Chain] = []
    for number, call in enumerate(rollout.calls):
        size = len(call.prompt_ids) + len(call.sampled_ids)
        if size > width:
            raise ValueError(
                f"rollout {rollout.task}/{rollout.group}/{rollout.sample} call {number} holds "
                f"{size} ids, wider than the {width}-id row")
        current = built[-1] if built else None
        if current is not None and merges(current.tokens, call):
            current.extend(call, number, len(current.tokens))
        else:
            chain = _Chain(index)
            chain.extend(call, number, 0)
            built.append(chain)
    return built


def advantages(rollouts: Sequence[Rollout], estimator: str = "group") -> np.ndarray:
    """One advantage per rollout from the rewards of its `(task, group)`.

    The baseline reads the group's trainable members only. A masked member
    has no score to compare against, and a group with fewer than two scored
    members has no baseline, so every member there gets zero.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    members: dict[tuple[str, str], list[int]] = {}
    for index, rollout in enumerate(rollouts):
        if rollout.status.trainable:
            members.setdefault((rollout.task, rollout.group), []).append(index)
    per_rollout = np.zeros(len(rollouts), np.float32)
    by_size: dict[int, list[list[int]]] = {}
    for group in members.values():
        if len(group) >= 2:
            by_size.setdefault(len(group), []).append(group)
    for size, groups in by_size.items():
        order = [index for group in groups for index in group]
        rewards = jnp.asarray([rollouts[index].reward for index in order], jnp.float32)
        if estimator == "rloo":
            values = rloo_advantage(rewards, size)
        else:
            values = group_advantage(rewards, size, normalise_by_std=estimator == "group")
        per_rollout[order] = np.asarray(values, np.float32)
    return per_rollout


def pack(rollouts: Sequence[Rollout], width: int, *, rows: int | None = None,
         estimator: str = "group") -> dict[str, np.ndarray]:
    """Strictly merge each trainable rollout's calls, then pack the chains into `[rows, width]`.

    Every array is `[rows, width]` and aligned with `input_ids`: entry t
    describes id t. `response_mask` is 1 on sampled ids alone;
    `behavior_log_probs`, `versions` and `call_index` are set on them;
    `advantages` repeats the rollout's advantage over its chain tokens;
    `rollout_weights` is described at `ROLLOUT_WEIGHTS_KEY`. Chains are
    placed first-fit in decreasing length, a stable order, and `rows` pads
    the batch to a fixed count, refusing chains that need more.
    """
    if type(width) is not int or width < 2:
        raise ValueError("a packed row holds at least two ids")
    if rows is not None and (type(rows) is not int or rows < 1):
        raise ValueError("rows is a positive integer, or None for as many as the chains need")
    values = advantages(rollouts, estimator)
    built = [chain for index, rollout in enumerate(rollouts) if rollout.status.trainable
             for chain in chains(rollout, index, width)]
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
    count = len(fill) if rows is None else rows
    if len(fill) > count:
        raise ValueError(f"the chains need {len(fill)} rows of {width} ids, more than the {rows} asked for")
    count = max(count, 1)
    shape = (count, width)
    ids = np.zeros(shape, np.int32)
    segments = np.zeros(shape, np.int32)
    positions = np.zeros(shape, np.int32)
    mask = np.zeros(shape, np.float32)
    behavior = np.zeros(shape, np.float32)
    versions = np.full(shape, -1, np.int32)
    rollout_index = np.full(shape, -1, np.int32)
    call_index = np.full(shape, -1, np.int32)
    advantage = np.zeros(shape, np.float32)
    trainable = np.zeros(len(rollouts), np.int64)
    for chain in built:
        trainable[chain.rollout] += sum(1 for call in chain.calls if call >= 0)
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
            rollout_index[row, span] = chain.rollout
            call_index[row, span] = chain.calls
            advantage[row, span] = values[chain.rollout]
            weights[row, span] = sampled / max(int(trainable[chain.rollout]), 1)
            start = stop
    return {
        IDS_KEY: ids, SEGMENT_IDS_KEY: segments, POSITIONS_KEY: positions,
        RESPONSE_MASK_KEY: mask, BEHAVIOR_LOG_PROBS_KEY: behavior, VERSIONS_KEY: versions,
        ROLLOUT_INDEX_KEY: rollout_index, CALL_INDEX_KEY: call_index,
        ADVANTAGES_KEY: advantage, ROLLOUT_WEIGHTS_KEY: weights,
    }


def sampled_values(batch: Mapping[str, np.ndarray],
                   values: Callable[[int, int], Sequence[float]]) -> np.ndarray:
    """Scatter per-call host values, one per sampled id, into the packed layout.

    `values(index, call)` returns one float per sampled id of call `call`
    of the `index`-th rollout handed to `pack`, such as the raw-policy
    log-probabilities an in-process sampler recorded. A call's sampled ids
    sit in one chain in order, so row-major order of its
    `(rollout_index, call_index)` positions is their order in the call.
    """
    rollout_index = np.asarray(batch[ROLLOUT_INDEX_KEY])
    call_index = np.asarray(batch[CALL_INDEX_KEY])
    out = np.zeros(rollout_index.shape, np.float32)
    flat_rollout, flat_call = rollout_index.reshape(-1), call_index.reshape(-1)
    where = np.flatnonzero(flat_call >= 0)
    keys = flat_rollout[where].astype(np.int64) * (1 << 32) + flat_call[where]
    order = np.argsort(keys, kind="stable")
    boundaries = np.flatnonzero(np.diff(keys[order])) + 1
    flat = out.reshape(-1)
    for group in np.split(order, boundaries):
        if not group.size:
            continue
        position = where[group[0]]
        given = np.asarray(values(int(flat_rollout[position]), int(flat_call[position])), np.float32)
        if given.shape != (group.size,):
            raise ValueError("values must return one float per sampled id of the call")
        flat[where[group]] = given
    return out


def rollout_metrics(rollouts: Sequence[Rollout], batch: Mapping[str, np.ndarray], *,
                    source: Callable[[Rollout], str] | None = None,
                    latencies: Sequence[float] | None = None,
                    version: int | None = None) -> dict[str, float]:
    """Host-side agentic telemetry for one packed batch and the rollouts behind it.

    - `merge/calls_per_chain`: trainable calls over packed chains, 1.0 when
      nothing merged; `pack/fill` is the share of row slots holding ids.
    - `status/<name>`: share of rollouts per status, and
      `masked/<name>`: share of all sampled ids that status masked.
    - `reward/mean` over scored rollouts, `reward/<source>` per
      `source(rollout)`, `reward/component/<name>` per verifier component.
    - `latency/p50`, `p90`, `p99`, `max` over `latencies`, seconds per rollout.
    - `lag/mean`, `lag/max`: `version` minus each trainable id's version.
    - `mismatch/k3_kl`: mean `r - log r - 1` of proximal over behavior on
      trainable ids, when the batch carries a proximal rescoring.
    """
    metrics: dict[str, float] = {}
    total = max(len(rollouts), 1)
    sampled = dict.fromkeys(Status, 0)
    for rollout in rollouts:
        sampled[rollout.status] += sum(len(call.sampled_ids) for call in rollout.calls)
    everything = max(sum(sampled.values()), 1)
    for status in Status:
        metrics[f"status/{status.value}"] = sum(rollout.status == status for rollout in rollouts) / total
        if not status.trainable:
            metrics[f"masked/{status.value}"] = sampled[status] / everything
    segments = np.asarray(batch[SEGMENT_IDS_KEY])
    rows = np.arange(segments.shape[0])[:, None] * (segments.shape[1] + 1) + segments
    chain_count = np.unique(rows[segments > 0]).size
    calls = sum(len(rollout.calls) for rollout in rollouts if rollout.status.trainable)
    metrics["merge/calls_per_chain"] = calls / chain_count if chain_count else 0.0
    metrics["pack/fill"] = float(np.mean(segments > 0))
    scored = [(rollout, rollout.reward) for rollout in rollouts
              if rollout.status.trainable and rollout.reward is not None]
    if scored:
        metrics["reward/mean"] = float(np.mean([reward for _, reward in scored]))
        by_source: dict[str, list[float]] = {}
        components: dict[str, list[float]] = {}
        for rollout, reward in scored:
            if source is not None:
                by_source.setdefault(source(rollout), []).append(float(reward))
            for name, value in rollout.components.items():
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
    if OLD_LOG_PROBS_KEY in batch and mask.any():
        log_ratio = (np.asarray(batch[OLD_LOG_PROBS_KEY], np.float64)
                     - np.asarray(batch[BEHAVIOR_LOG_PROBS_KEY], np.float64))[mask]
        metrics["mismatch/k3_kl"] = float(np.mean(np.exp(log_ratio) - log_ratio - 1))
    return metrics
