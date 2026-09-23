"""Keep agent rollouts in flight and admit complete groups within a staleness bound.

`RolloutScheduler` is the trainer's `Rollout` for any `RolloutSource`: an
in-process environment, a single-turn prompt set, or a harness behind a
recording gateway. Wrap the task dataset with `scheduler.tasks(dataset)`:
the wrapped stream registers each task batch as the trainer's prefetch reads
it, so when the trainer hands over batch `i` the scheduler submits batches
`i + 1 ... i + ahead` under the version the engines serve now. Batch `i`'s
own rollouts were submitted `ahead` calls earlier and have been running,
across weight pushes, since. Nothing is read ahead of the trainer's own
prefetch, so the checkpointed data position stays the trainer's: a resumed
run re-reads and resubmits whatever was in flight, and reopening the stream
cancels what the old one left running.

Each task becomes one group of `groups` rollouts, relabelled with the
scheduler's own group id, sample index and attempt, so a resubmitted sample
rejoins its group. Admission is per rollout, by status:

- COMPLETED and AGENT_ERROR are admitted; the verifier scored them.
- TRUNCATED is admitted and `pack` masks it: a context, turn or token limit
  says nothing about the task.
- INFRA_ERROR and CANCELLED are never scored. The sample is submitted again
  under the served weights, up to `max_attempts` failures per sample, after
  which the group is abandoned rather than trained incomplete.
- A rollout whose oldest call is more than `max_lag` updates behind is
  discarded and resubmitted. One still running whose submission is already
  past the bound is cancelled before anyone waits on it: its first call was
  made under that version.

A source that raises instead of returning a status is broken; the exception
propagates after the batch's work is cancelled.

The long tail is cut two ways, both keeping the batch shape fixed.
`oversample` extra samples run per group and a group is admitted when its
first `groups` rollouts finish; the rest are cancelled (APRIL's active
partial rollouts, arXiv:2509.18521, without the carry-over). `admit` below
the task batch size admits the first `admit` groups to complete and cancels
the others (slime's over-sampling batch). Both select by completion time,
which favors short rollouts; that bias is the price of not waiting on the
tail. Rollouts running when a weight push lands keep running: their later
calls carry the new version and the rollout's staleness is its oldest call's
(Kimi K2's partial rollouts, arXiv:2507.20534 section 3.3.4).

Weights are pushed through `weights` when the served version falls
`sync_every` updates behind, so a first submission is at most
`ahead + sync_every - 1` updates stale at consumption, and construction
refuses a `max_lag` below that. Admitted groups are packed by `pack` into
fixed `[rows, width]` rows; `pack` computes the per-rollout advantages and
masks. The proximal policy is the trainer's current weights, rescored over
the packed rows with `GRPOObjective.packed_log_probs` (decoupled PPO, AReaL
arXiv:2505.24298): sources report behavior likelihoods only, and the
objective's `behavior_importance_cap` or `behavior_band` weights each token
by proximal over behavior.

One trainer process owns the scheduler; multi-process trainers are refused.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from typing import Protocol

import jax
import numpy as np

from dew.data.dataset import Batch, Dataset, tapped
from dew.nn.inputs import local_rows
from dew.objectives.base import Variables
from dew.training.state import TrainState

from .grpo import GRPOObjective
from .rollout import OLD_LOG_PROBS_KEY, RESPONSE_MASK_KEY
from .rollouts import Rollout, RolloutSource, Status, Task, pack, rollout_metrics

TASK_ID_KEY = "task_id"


class Publisher(Protocol):
    """Where the trainer's weights go: `load` serves them under `version`.

    A `RolloutServer` is one; so is an engine fleet's publish sequence.
    """

    @property
    def version(self) -> int: ...

    def load(self, variables: Variables, version: int) -> None: ...


@dataclass(frozen=True)
class SchedulerRecord:
    """What one trainer call consumed and what it cost.

    `version` and `lag` are the oldest admitted call's; `groups` counts
    admitted groups; `resubmitted` counts resubmissions by cause
    (`infra_error`, `cancelled`, `stale`); `cancelled` counts in-flight
    rollouts cancelled as surplus, stale or abandoned; `abandoned` counts
    groups given up after `max_attempts`; `waited` is the seconds the
    trainer waited. `metrics` is `rollout_metrics` over the admitted
    rollouts and their packed batch: merge ratio, status shares and masked
    shares, mean reward and reward components, submission-to-finish
    latency tail, token lag and proximal-behavior mismatch.
    """

    updates: int
    version: int
    lag: int
    groups: int
    resubmitted: Mapping[str, int]
    cancelled: int
    abandoned: int
    waited: float
    metrics: Mapping[str, float]


def task_ids(batch: Batch) -> list[Task]:
    """One task per integer `task_id` row, named by its decimal id."""
    ids = local_rows(batch[TASK_ID_KEY])
    if ids.ndim != 1 or not ids.size or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("task_id must be a nonempty vector of integer task identities")
    return [Task(str(int(value))) for value in ids]


@dataclass(eq=False)
class _Sample:
    index: int
    attempt: int
    submitted: int
    future: Future[Rollout]
    started: float = field(default_factory=time.perf_counter)
    finished: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.future.add_done_callback(lambda _: self.finished.append(time.perf_counter()))


@dataclass
class _Group:
    task: Task
    label: str
    live: list[_Sample] = field(default_factory=list)
    done: list[Rollout] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    failures: Counter[int] = field(default_factory=Counter)
    abandoned: bool = False


@dataclass
class _Entry:
    serial: int
    tasks: tuple[Task, ...]
    groups: list[_Group] = field(default_factory=list)
    complete: list[_Group] = field(default_factory=list)


@dataclass
class _Tally:
    resubmitted: Counter[str] = field(default_factory=Counter)
    cancelled: int = 0
    abandoned: int = 0


class RolloutScheduler:
    """Train on complete rollout groups from `source`, `ahead` task batches early.

    `tasks` turns one registered batch into its tasks (`task_ids` for
    integer `task_id` rows). `width` and `rows` fix the packed batch shape;
    `estimator` is `pack`'s advantage family. `log`, when given, receives
    a `SchedulerRecord` per call.
    """

    def __init__(self, objective: GRPOObjective, source: RolloutSource, weights: Publisher, *,
                 width: int, rows: int, tasks: Callable[[Batch], Sequence[Task]] = task_ids,
                 groups: int = 4, oversample: int = 0, admit: int | None = None,
                 max_lag: int = 1, ahead: int = 1, sync_every: int = 1, max_attempts: int = 3,
                 estimator: str = "group", log: Callable[[SchedulerRecord], None] | None = None):
        for name, value, least in (("width", width, 2), ("rows", rows, 1), ("groups", groups, 2),
                                   ("oversample", oversample, 0), ("ahead", ahead, 0),
                                   ("sync_every", sync_every, 1), ("max_lag", max_lag, 0),
                                   ("max_attempts", max_attempts, 1)):
            if type(value) is not int or value < least:
                raise ValueError(f"{name} must be an integer of at least {least}")
        if admit is not None and (type(admit) is not int or admit < 1):
            raise ValueError("admit must be a positive number of groups, or None to admit every task")
        if ahead + sync_every - 1 > max_lag:
            raise ValueError(
                f"ahead={ahead} and sync_every={sync_every} let a batch fall {ahead + sync_every - 1} "
                f"updates behind, past max_lag={max_lag}")
        if max_lag > 0 and objective.behavior_importance_cap is None and objective.behavior_band is None:
            raise ValueError("stale rollouts need the objective's behavior_importance_cap or behavior_band: "
                             "the proximal-to-behavior importance weight is the off-policy correction")
        self.objective, self.source, self.weights, self.tasks_of = objective, source, weights, tasks
        self.width, self.rows, self.groups, self.oversample, self.admit = width, rows, groups, oversample, admit
        self.max_lag, self.ahead, self.sync_every, self.max_attempts = max_lag, ahead, sync_every, max_attempts
        self.estimator, self.log = estimator, log
        self._lock = threading.Lock()
        self._registered: deque[_Entry] = deque()
        self._serial = 0
        self._rescore = jax.jit(objective.packed_log_probs)

    def tasks(self, dataset: Dataset) -> Dataset:
        """`dataset` with a training stream that registers each task batch ahead of the step.

        Opening the stream again, as a resume does, cancels every rollout
        the previous stream left in flight.
        """
        opened = tapped(dataset.train, self._register)

        def train() -> Iterator[Batch]:
            self._drop_registered()
            return opened()

        return dataclasses.replace(dataset, train=train)

    def close(self) -> None:
        """Cancel every rollout in flight; the source belongs to the caller."""
        self._drop_registered()

    def _drop_registered(self) -> None:
        with self._lock:
            entries = list(self._registered)
            self._registered.clear()
        self._cancel([sample for entry in entries for group in entry.groups for sample in group.live])

    def _register(self, batch: Batch) -> None:
        tasks = tuple(self.tasks_of(batch))
        with self._lock:
            self._registered.append(_Entry(self._serial, tasks))
            self._serial += 1

    def _cancel(self, samples: Sequence[_Sample]) -> None:
        if samples:
            self.source.cancel([sample.future for sample in samples])

    def _submit(self, group: _Group, samples: Sequence[tuple[int, int]]) -> None:
        """Submit `(index, attempt)` samples of one group under the served version."""
        version = self.weights.version
        futures = self.source.submit(group.task, len(samples), version=version)
        if len(futures) != len(samples):
            raise ValueError(f"the source returned {len(futures)} rollouts for {len(samples)} samples")
        group.live.extend(_Sample(index, attempt, version, future)
                          for (index, attempt), future in zip(samples, futures, strict=True))

    def _open(self, entry: _Entry) -> None:
        width = self.groups + self.oversample
        for row, task in enumerate(entry.tasks):
            group = _Group(task, f"{entry.serial}/{row}")
            entry.groups.append(group)
            self._submit(group, [(index, 0) for index in range(width)])

    def _publish(self, state: TrainState, updates: int) -> None:
        """Push the weights when the served version is `sync_every` behind, twice if the first did not take.

        A served version ahead of `updates` is out of date as well: a fit
        restored from an earlier checkpoint must not sample from weights
        newer than its own.
        """
        for _ in range(2):
            if 0 <= updates - self.weights.version < self.sync_every:
                return
            self.weights.load(state.params, updates)
        if not 0 <= updates - self.weights.version < self.sync_every:
            raise RuntimeError(f"the weights pushed at update {updates} did not take: the engines still "
                               f"serve version {self.weights.version}")

    def _replace(self, group: _Group, sample: _Sample, cause: str, tally: _Tally) -> None:
        """Resubmit a sample that cannot join its group, or abandon the group."""
        if cause != "stale":
            group.failures[sample.index] += 1
            if group.failures[sample.index] >= self.max_attempts:
                group.abandoned = True
                tally.abandoned += 1
                tally.cancelled += len(group.live)
                self._cancel(group.live)
                group.live.clear()
                return
        if len(group.done) + len(group.live) >= self.groups:
            return  # an over-sampled spare takes its place
        tally.resubmitted[cause] += 1
        self._submit(group, [(sample.index, sample.attempt + 1)])

    def _harvest(self, entry: _Entry, group: _Group, updates: int, tally: _Tally) -> None:
        """Move one group's finished rollouts into it, replacing what cannot be admitted.

        A group stops taking rollouts once it is full or abandoned; its
        remaining samples are left for `_admit`'s final cancel.
        """
        stale = [sample for sample in group.live
                 if not sample.future.done() and updates - sample.submitted > self.max_lag]
        if stale:
            tally.cancelled += len(stale)
            self._cancel(stale)
            group.live = [sample for sample in group.live if sample not in stale]
        for sample in [sample for sample in group.live if sample.future.done()]:
            if group.abandoned or len(group.done) >= self.groups:
                return
            group.live.remove(sample)
            if sample.future.cancelled():
                self._replace(group, sample, "cancelled", tally)
            else:
                self._settle(entry, group, sample, sample.future.result(), updates, tally)
        for sample in stale:
            if group.abandoned or len(group.done) >= self.groups:
                return
            self._replace(group, sample, "stale", tally)

    def _settle(self, entry: _Entry, group: _Group, sample: _Sample, rollout: Rollout, updates: int,
                tally: _Tally) -> None:
        """Admit one finished rollout into its group, or replace it."""
        if rollout.status in (Status.INFRA_ERROR, Status.CANCELLED):
            self._replace(group, sample, rollout.status.value, tally)
            return
        oldest = min((call.version for call in rollout.calls), default=sample.submitted)
        if updates - oldest > self.max_lag:
            self._replace(group, sample, "stale", tally)
            return
        group.done.append(dataclasses.replace(rollout, task=group.task.id, group=group.label,
                                              sample=sample.index, attempt=sample.attempt))
        # The future is done; its callback may still be on the resolving thread.
        finished = sample.finished[0] if sample.finished else time.perf_counter()
        group.latencies.append(finished - sample.started)
        if len(group.done) == self.groups:
            entry.complete.append(group)

    def _admit(self, entry: _Entry, updates: int, tally: _Tally) -> list[_Group]:
        """Wait for the entry's first `admit` complete groups, then cancel the rest."""
        target = len(entry.groups) if self.admit is None else min(self.admit, len(entry.groups))
        try:
            while True:
                pending = [group for group in entry.groups
                           if not group.abandoned and len(group.done) < self.groups]
                for group in pending:
                    self._harvest(entry, group, updates, tally)
                pending = [group for group in pending if not group.abandoned and len(group.done) < self.groups]
                if len(entry.complete) >= target or not pending:
                    break
                wait([sample.future for group in pending for sample in group.live], return_when=FIRST_COMPLETED)
        finally:
            leftover = [sample for group in entry.groups for sample in group.live]
            tally.cancelled += len(leftover)
            self._cancel(leftover)
            for group in entry.groups:
                group.live.clear()
        if not entry.complete:
            raise RuntimeError(f"no group of batch {entry.serial} completed: every one was abandoned after "
                               f"{self.max_attempts} failed attempts of one sample")
        return entry.complete[:target]

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> dict[str, np.ndarray]:
        del key  # sources own their sampling seeds
        if jax.process_count() != 1:
            raise ValueError("RolloutScheduler coordinates one trainer process")
        updates = int(state.updates)
        self._publish(state, updates)
        with self._lock:
            if not self._registered:
                raise ValueError("a RolloutScheduler batch comes from the stream of RolloutScheduler.tasks(dataset)")
            entry = self._registered.popleft()
            upcoming = list(self._registered)[:self.ahead]
        if tuple(self.tasks_of(batch)) != entry.tasks:
            self._cancel([sample for group in entry.groups for sample in group.live])
            raise ValueError("the trainer's batch is not the next registered task batch")
        for pending in (entry, *upcoming):
            if not pending.groups:
                self._open(pending)
        tally = _Tally()
        began = time.perf_counter()
        admitted = self._admit(entry, updates, tally)
        waited = time.perf_counter() - began
        rollouts = [rollout for group in admitted for rollout in group.done[:self.groups]]
        latencies = [latency for group in admitted for latency in group.latencies[:self.groups]]
        packed = pack(rollouts, self.width, rows=self.rows, estimator=self.estimator)
        proximal = np.asarray(self._rescore(state.params, packed), np.float32)
        packed[OLD_LOG_PROBS_KEY] = proximal * packed[RESPONSE_MASK_KEY]
        if self.log is not None:
            versions = [call.version for rollout in rollouts for call in rollout.calls]
            oldest = min(versions, default=updates)
            self.log(SchedulerRecord(
                updates, oldest, updates - oldest, len(admitted), dict(tally.resubmitted), tally.cancelled,
                tally.abandoned, waited, rollout_metrics(rollouts, packed, latencies=latencies, version=updates)))
        return packed
