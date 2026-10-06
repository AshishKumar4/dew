"""Keep agent rollouts in flight and admit complete groups within a staleness bound.

`RolloutScheduler` is the trainer's `Rollout` for any `SessionSource`, such
as an in-process environment, a single-turn prompt set, or a harness behind
a recording gateway. Wrap the task dataset with `scheduler.tasks(dataset)`.
The wrapped stream registers each task batch when the trainer's prefetch
reads it. When the trainer then passes batch `i` to the scheduler, the
scheduler submits batches `i + 1 ... i + ahead` under the weight version
the engines serve at that moment. Batch `i`'s own rollouts were submitted
`ahead` calls earlier and have kept running since, through any weight
pushes in between. The scheduler reads no data beyond the trainer's own
prefetch, so the data position in a checkpoint is still the trainer's. A
resumed run reads again and resubmits whatever was in flight, and reopening
the stream cancels the rollouts the old stream left running.

Each task becomes one group of `groups` rollouts. The scheduler relabels
each rollout with its own group id, sample index and attempt number, so a
resubmitted sample rejoins its group. Each finished rollout is admitted or
replaced according to its status:

- COMPLETED and AGENT_ERROR are admitted, because the verifier scored them.
- TRUNCATED is admitted, and `truncation` sets how it trains: `mask` (the
  default, for agentic context, turn and wall-clock limits), `score` on its
  verifier reward (single-turn RLVR), or `zero` (a length penalty); see
  `dew.objectives.rl.sessions`. Under `score`, a truncated rollout without
  a reward is resubmitted as a failed attempt (cause `unscored`), like an
  INFRA_ERROR. Its verifier never ran, so masking it would drop a sample
  for a reason unrelated to the policy, and there is no reward to train it
  on.
- INFRA_ERROR and CANCELLED are never scored. The sample is submitted again
  under the weights being served. After `max_attempts` failures of one
  sample, the scheduler abandons the whole group instead of training it
  incomplete.
- A rollout whose oldest call is more than `max_lag` updates behind is
  discarded and resubmitted. A rollout still running when its submission
  falls past that bound is cancelled without waiting for it to finish,
  because its first call was made under the submission's version.
- A rollout still running `timeout` seconds after its submission is
  cancelled and resubmitted as a failed attempt, like an INFRA_ERROR.
  Cancelling only asks the source to stop. The scheduler cannot reclaim a
  thread stuck inside an environment step, so environments must bound
  their own step time.

A source that raises an exception instead of returning a status is broken.
The scheduler cancels the batch's remaining rollouts and lets the exception
propagate.

Two options cut the long tail of slow rollouts, and both keep the batch
shape fixed. With `oversample`, each group runs that many extra samples and
is admitted once its first `groups` rollouts finish; the scheduler cancels
the rest. This is APRIL's active partial rollouts (arXiv:2509.18521)
without the carry-over. With `admit` below the task batch size, the
scheduler admits the first `admit` groups to complete and cancels the
others, as slime's over-sampling batch does. Both select by completion
time, which favors short rollouts; that bias is the price of not waiting on
the tail. Rollouts that are running when new weights are pushed keep
running. Their later calls are stamped with the new version, and a
rollout's staleness is that of its oldest call, as in Kimi K2's partial
rollouts (arXiv:2507.20534 section 3.3.4).

The scheduler pushes weights through `weights` when the served version
falls `sync_every` updates behind the trainer, or is ahead of it, as after
a resume from an earlier checkpoint. A first submission is therefore at
most `ahead + sync_every - 1` updates stale when the trainer consumes it,
and the constructor refuses a `max_lag` below that. `pack` packs the
admitted groups into fixed `[rows, width]` arrays and computes the
per-rollout advantages and masks. If a complete group's chains do not fit
in `rows` beside the groups admitted before it, the scheduler cuts that
group, so an overfull batch never reaches the step. The proximal policy is
the trainer's current weights. The scheduler rescores the packed rows under
them with `GRPOObjective.packed_log_probs` (decoupled PPO, AReaL
arXiv:2505.24298). Sources report only behavior likelihoods, and the
objective's `behavior_importance` weights each token by the ratio of its
proximal to its behavior likelihood.

In a multi-process trainer, every process runs its own scheduler on its own
source, over the task rows its data stream reads, and packs its own `rows`.
The step's batch is all the processes' rows together, sharded as the
trainer shards any batch, and no rollout moves between processes. The
processes still act together at a few points in each call. The weight push
is one call that every process makes, so the publisher has to agree on it
across processes. If admission fails on one process, every process raises
at the admission agreement point, so none is left waiting in the
rescoring. The proximal rescoring runs once over the combined batch, and
each process reads its own rows back. When a tensor or sequence axis spans
processes, several processes read the same share of the data. Then only
the share's first reader samples, and the others train on the rows it
packed (`first_reader_batch`). That way their devices hold the same rows,
which independent draws would not give them.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

import jax
import numpy as np
from jax.sharding import Mesh

from dew.coordination import agreed
from dew.data.dataset import Batch, DataPartition, Dataset, tapped
from dew.nn.inputs import local_rows, mesh_of
from dew.objectives.base import Shown, Variables
from dew.training.distributed import first_reader_batch, shard_batch
from dew.training.state import TrainState

from .grpo import GRPOObjective
from .sessions import (
    OLD_LOG_PROBS_KEY,
    RESPONSE_MASK_KEY,
    Session,
    SessionSource,
    Status,
    Task,
    chain_lengths,
    check_truncation,
    pack,
    rows_needed,
    session_metrics,
)


class Publisher(Protocol):
    """Serves the trainer's weights to the rollout engines.

    `load(variables, version)` makes the engines serve `variables` under
    `version`, and the `version` property is the version they serve now. A
    `RolloutServer` is a publisher, and so is an engine fleet's publish
    sequence.
    """

    @property
    def version(self) -> int: ...

    def load(self, variables: Variables, version: int) -> None: ...


@dataclass(frozen=True)
class SchedulerRecord:
    """What one trainer call consumed and what it cost.

    - `updates` is the trainer's update count at the call.
    - `version` is the version of the oldest admitted call, and `lag` is
      how many updates it is behind `updates`.
    - `groups` counts the admitted groups.
    - `resubmitted` counts resubmissions by cause (`infra_error`,
      `cancelled`, `stale`, `timeout`, `unscored`).
    - `cancelled` counts in-flight rollouts cancelled because they were
      surplus, stale, past their `timeout` or in an abandoned group.
    - `abandoned` counts groups given up after `max_attempts` failures.
    - `cut` counts complete groups left out because their chains did not
      fit the batch's `rows` beside the groups admitted before them.
    - `waited` is the number of seconds the trainer waited.
    - `metrics` is `session_metrics` over the admitted rollouts and their
      packed batch: merge ratio, status shares and masked shares, mean
      reward and reward components, the submission-to-finish latency tail
      and token lag.

    The trainer-engine mismatch is not in this record; the loss reports it.
    """

    updates: int
    version: int
    lag: int
    groups: int
    resubmitted: Mapping[str, int]
    cancelled: int
    abandoned: int
    cut: int
    waited: float
    metrics: Mapping[str, float]


def task_ids(batch: Batch) -> list[Task]:
    """Return one task per integer `task_id` row of `batch`, named by the id in decimal.

    Raises ValueError unless this process's `task_id` rows are a nonempty
    vector of integers.
    """
    ids = local_rows(batch["task_id"])
    if ids.ndim != 1 or not ids.size or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("task_id must be a nonempty vector of integer task identities")
    return [Task(str(int(value))) for value in ids]


@dataclass(eq=False)
class _Sample:
    index: int
    attempt: int
    submitted: int
    future: Future[Session]
    started: float = field(default_factory=time.perf_counter)
    finished: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.future.add_done_callback(lambda _: self.finished.append(time.perf_counter()))


@dataclass
class _Group:
    task: Task
    label: str
    live: list[_Sample] = field(default_factory=list)
    done: list[Session] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    """Once complete, the lengths of the chains its sessions pack into."""
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
    cut: int = 0


class RolloutScheduler:
    """Runs rollouts from `source` ahead of the trainer and packs complete groups into its batches.

    The module docstring describes `groups`, `oversample`, `admit`,
    `max_lag`, `ahead`, `sync_every` and `max_attempts`. The other
    arguments:

    - `tasks` turns one registered batch into its tasks (`task_ids` for
      integer `task_id` rows).
    - `timeout` is each rollout's deadline in seconds from its submission.
    - `width` and `rows` fix the packed batch shape. `rows` is this
      process's share of a multi-process trainer's batch.
    - `estimator` and `truncation` are `pack`'s advantage family and
      truncation policy, and `support_capacity` is its per-row support
      length, which a filtered-sampling source requires.
    - `log`, when given, receives a `SchedulerRecord` of this process's
      rollouts after each call. A process that is not the first reader of
      its share samples nothing and logs nothing.

    A `max_lag` above 0 needs the objective's `behavior_importance` (a TIS
    cap or an IcePop band), and the constructor raises ValueError without
    it. `metrics` holds the latest call's numbers, which the trainer logs as
    `rollout/<name>`.
    """

    shown: ClassVar[dict[str, Shown]] = {
        "reward/mean": Shown(better="higher"),
        "lag/mean": Shown(better="lower"),
        "pack/fill": Shown(better="higher", percent=True),
        **{f"status/{status.value}": Shown(percent=True) for status in Status},
    }

    def __init__(self, objective: GRPOObjective, source: SessionSource, weights: Publisher, *,
                 width: int, rows: int, tasks: Callable[[Batch], Sequence[Task]] = task_ids,
                 groups: int = 4, oversample: int = 0, admit: int | None = None,
                 max_lag: int = 1, ahead: int = 1, sync_every: int = 1, max_attempts: int = 3,
                 timeout: float | None = None, estimator: str = "group", truncation: str = "mask",
                 support_capacity: int | None = None, log: Callable[[SchedulerRecord], None] | None = None):
        for name, value, least in (("width", width, 2), ("rows", rows, 1), ("groups", groups, 2),
                                   ("oversample", oversample, 0), ("ahead", ahead, 0),
                                   ("sync_every", sync_every, 1), ("max_lag", max_lag, 0),
                                   ("max_attempts", max_attempts, 1)):
            if type(value) is not int or value < least:
                raise ValueError(f"{name} must be an integer of at least {least}")
        if timeout is not None and not timeout > 0:
            raise ValueError(
                "timeout must be a positive number of seconds, or None to wait without a deadline"
            )
        check_truncation(truncation)
        if admit is not None and (type(admit) is not int or admit < 1):
            raise ValueError("admit must be a positive number of groups, or None to admit every task")
        if ahead + sync_every - 1 > max_lag:
            raise ValueError(
                f"ahead={ahead} and sync_every={sync_every} let a batch fall {ahead + sync_every - 1} "
                f"updates behind, past max_lag={max_lag}")
        if max_lag > 0 and objective.behavior_importance is None:
            raise ValueError(
                "stale rollouts need the objective's behavior_importance, a TIS cap or an IcePop band: "
                "the proximal-to-behavior importance weight is the off-policy correction"
            )
        self.objective, self.source, self.weights, self.tasks_of = objective, source, weights, tasks
        self.width, self.rows, self.groups, self.oversample, self.admit = (
            width,
            rows,
            groups,
            oversample,
            admit,
        )
        self.max_lag, self.ahead, self.sync_every, self.max_attempts = (
            max_lag,
            ahead,
            sync_every,
            max_attempts,
        )
        self.timeout, self.estimator, self.truncation, self.log = timeout, estimator, truncation, log
        self.support_capacity = support_capacity
        self.metrics: dict[str, float] = {}
        self._lock = threading.Lock()
        self._registered: deque[_Entry] = deque()
        self._serial = 0
        self._partition = DataPartition()
        self._rescore = jax.jit(objective.packed_log_probs)

    def tasks(self, dataset: Dataset) -> Dataset:
        """Return `dataset` with a training stream that registers each task batch ahead of the step.

        Opening the stream again, as a resume does, cancels every rollout
        the previous stream left in flight.
        """
        opened = tapped(dataset.train, self._register)

        def train(partition: DataPartition) -> Iterator[Batch]:
            self._drop_registered()
            self._partition = partition
            return opened(partition)

        return dataclasses.replace(dataset, train=train)

    def close(self) -> None:
        """Cancel every rollout in flight. The caller closes the source; this method does not."""
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
        group.live.extend(_Sample(index, attempt, version, future)
                          for (index, attempt), future in zip(samples, futures, strict=True))

    def _open(self, entry: _Entry) -> None:
        width = self.groups + self.oversample
        for row, task in enumerate(entry.tasks):
            group = _Group(task, f"{entry.serial}/{row}")
            entry.groups.append(group)
            self._submit(group, [(index, 0) for index in range(width)])

    def _publish(self, state: TrainState, updates: int) -> None:
        """Push the weights when the served version is `sync_every` behind.

        A served version ahead of `updates` is out of date as well: a fit
        restored from an earlier checkpoint must not sample from weights
        newer than its own.
        """
        if 0 <= updates - self.weights.version < self.sync_every:
            return
        self.weights.load(state.variables, updates)
        if self.weights.version != updates:
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
        now = time.perf_counter()
        overdue = {}
        for sample in group.live:
            if sample.future.done():
                continue
            if updates - sample.submitted > self.max_lag:
                overdue[sample] = "stale"
            elif self.timeout is not None and now - sample.started >= self.timeout:
                overdue[sample] = "timeout"
        if overdue:
            tally.cancelled += len(overdue)
            self._cancel(list(overdue))
            group.live = [sample for sample in group.live if sample not in overdue]
        finished = [(sample, None) for sample in group.live if sample.future.done()]
        for sample, cause in [*finished, *overdue.items()]:
            if group.abandoned or len(group.done) >= self.groups:
                return
            if cause is not None:
                self._replace(group, sample, cause, tally)
                continue
            group.live.remove(sample)
            if sample.future.cancelled():
                self._replace(group, sample, "cancelled", tally)
            else:
                self._settle(entry, group, sample, sample.future.result(), updates, tally)

    def _settle(self, entry: _Entry, group: _Group, sample: _Sample, rollout: Session, updates: int,
                tally: _Tally) -> None:
        """Admit one finished rollout into its group, or replace it."""
        if rollout.status in (Status.INFRA_ERROR, Status.CANCELLED):
            self._replace(group, sample, rollout.status.value, tally)
            return
        if rollout.status is Status.TRUNCATED and rollout.reward is None and self.truncation == "score":
            self._replace(group, sample, "unscored", tally)
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
            self._fit(entry, group, tally)

    def _fit(self, entry: _Entry, group: _Group, tally: _Tally) -> None:
        """Admit a complete group when its chains fit `rows` beside those
        admitted before it; cut it otherwise.

        A group that cannot fit `rows` on its own is refused with a
        ValueError: `rows` is too small for the run, not for this batch.

        A session whose calls do not extend each other packs as one chain
        per call, so the rows a group needs are known only once it is done.
        """
        group.lengths = chain_lengths(group.done, self.width, truncation=self.truncation)
        alone = rows_needed(group.lengths, self.width)
        if alone > self.rows:
            raise ValueError(
                f"one group of task {group.task.id} needs {alone} rows of {self.width} ids, "
                f"more than rows={self.rows}; size rows for sessions whose calls split into chains"
            )
        admitted = [length for complete in entry.complete for length in complete.lengths]
        if rows_needed([*admitted, *group.lengths], self.width) <= self.rows:
            entry.complete.append(group)
            return
        tally.cut += 1
        tally.cancelled += len(group.live)
        self._cancel(group.live)
        group.live.clear()

    def _admit(self, entry: _Entry, updates: int, tally: _Tally) -> list[_Group]:
        """Wait for the entry's first `admit` complete groups, then cancel the rest."""
        target = len(entry.groups) if self.admit is None else min(self.admit, len(entry.groups))
        try:
            while True:
                pending = [group for group in entry.groups
                           if not group.abandoned and len(group.done) < self.groups]
                for group in pending:
                    self._harvest(entry, group, updates, tally)
                pending = [
                    group for group in pending if not group.abandoned and len(group.done) < self.groups
                ]
                if len(entry.complete) >= target or not pending:
                    break
                live = [sample for group in pending for sample in group.live]
                deadline = None if self.timeout is None else max(
                    min(sample.started for sample in live) + self.timeout - time.perf_counter(), 0.0)
                wait([sample.future for sample in live], timeout=deadline, return_when=FIRST_COMPLETED)
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
        updates = int(state.updates)
        self._publish(state, updates)
        rollouts, latencies, groups, tally, waited = agreed(
            "rollout admission", lambda: self._admitted(batch, updates))
        sampling = self._partition.reader == 0
        packed = (
            pack(
                rollouts,
                self.width,
                rows=self.rows,
                estimator=self.estimator,
                truncation=self.truncation,
                support_capacity=self.support_capacity,
            )
            if sampling
            else {}
        )
        mesh = mesh_of(state.variables)
        if mesh is not None:
            packed = first_reader_batch(mesh, packed)
        packed[OLD_LOG_PROBS_KEY] = self._proximal(state.variables, packed, mesh) * packed[RESPONSE_MASK_KEY]
        if sampling:
            versions = [call.version for rollout in rollouts for call in rollout.calls]
            oldest = min(versions, default=updates)
            metrics = session_metrics(rollouts, packed, latencies=latencies, version=updates,
                                      truncation=self.truncation)
            record = SchedulerRecord(updates, oldest, updates - oldest, groups, dict(tally.resubmitted),
                                     tally.cancelled, tally.abandoned, tally.cut, waited, metrics)
            self.metrics = {**metrics, "lag": float(record.lag), "groups": float(groups),
                            "resubmitted": float(sum(record.resubmitted.values())),
                            "waited_seconds": waited}
            if self.log is not None:
                self.log(record)
        return packed

    def _admitted(self, batch: Batch, updates: int) -> tuple[list[Session], list[float], int, _Tally, float]:
        """This process's admitted rollouts for `batch`, their latencies,
        the group count, the tally and the wait."""
        with self._lock:
            if not self._registered:
                raise ValueError(
                    "a RolloutScheduler batch comes from the stream of RolloutScheduler.tasks(dataset)"
                )
            entry = self._registered.popleft()
            upcoming = list(self._registered)[:self.ahead]
        if tuple(self.tasks_of(batch)) != entry.tasks:
            self._cancel([sample for group in entry.groups for sample in group.live])
            raise ValueError("the trainer's batch is not the next registered task batch")
        if self._partition.reader:
            return [], [], 0, _Tally(), 0.0
        for pending in (entry, *upcoming):
            if not pending.groups:
                self._open(pending)
        tally = _Tally()
        began = time.perf_counter()
        admitted = self._admit(entry, updates, tally)
        waited = time.perf_counter() - began
        return ([rollout for group in admitted for rollout in group.done[:self.groups]],
                [latency for group in admitted for latency in group.latencies[:self.groups]],
                len(admitted), tally, waited)

    def _proximal(self, params: Variables, packed: dict[str, np.ndarray], mesh: Mesh | None) -> np.ndarray:
        """The trainer's likelihoods of this process's packed rows, `[rows, width - 1]`.

        Over a mesh the rows are placed as the step places the batch, every
        process's rows together, so the rescoring runs once over the pool's
        batch and each process reads its own rows back.
        """
        scored = self._rescore(params, packed if mesh is None else shard_batch(mesh, packed))
        return local_rows(scored).astype(np.float32)


__all__ = ["Publisher", "RolloutScheduler", "SchedulerRecord", "task_ids"]
