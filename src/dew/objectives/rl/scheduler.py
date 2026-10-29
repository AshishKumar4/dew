"""Collect rollouts and admit complete groups into the trainer's batches, synchronously or ahead of it.

`RolloutScheduler` is the trainer's `Rollout` for token sessions. Each task
becomes one group of `groups` rollouts, which the scheduler admits, fits
into its rows, packs (`sessions.pack`) and reports, whichever way they were
acquired:

- `Synchronous(collector)` draws every task's sessions at the call from the
  variables the step trains (`sessions.Collector`): `PolicyCollector` for
  single-turn prompts, `EpisodeCollector` for tool episodes in lock step.
  Its draws are one program every process of the trainer's mesh enters, so
  every process collects and the first reader of each data share packs.
  Their raw likelihoods are the proximal policy's, so `old_log_probs` are
  those and no rescoring runs. A collected sample is never drawn again: a
  group with an INFRA_ERROR or CANCELLED session is abandoned.
  `RolloutScheduler.sampled` and `RolloutScheduler.episodes` build this
  case.
- `Asynchronous(source, weights, objective, ...)` keeps rollouts in flight
  from any `SessionSource` (an in-process environment, a single-turn prompt
  set, a harness behind a recording gateway) within a staleness bound, as
  below.

Asynchronously, wrap the task dataset with `scheduler.tasks(dataset)`. The
wrapped stream registers each task batch when the trainer's prefetch reads
it. When the trainer then passes batch `i` to the scheduler, the scheduler
submits batches `i + 1 ... i + ahead` under the weight version the engines
serve at that moment. Batch `i`'s own rollouts were submitted `ahead` calls
earlier and have kept running since, through any weight pushes in between.
The scheduler reads no data beyond the trainer's own prefetch, so the data
position in a checkpoint is still the trainer's. A resumed run reads again
and resubmits whatever was in flight, and reopening the stream cancels the
rollouts the old stream left running.

The scheduler relabels each rollout with its own group id, sample index and
attempt number, so a resubmitted sample rejoins its group. Each finished
rollout is admitted or replaced according to its status:

- COMPLETED and AGENT_ERROR are admitted, because the verifier scored them.
- TRUNCATED is admitted, and `truncation` sets how it trains: `mask` (the
  default, for agentic context, turn and wall-clock limits), `score` on its
  verifier reward (single-turn RLVR), or `zero` (a length penalty); see
  `dew.objectives.rl.sessions`. Under `score`, a truncated rollout without
  a reward is resubmitted as a failed attempt (cause `unscored`), like an
  INFRA_ERROR. Its verifier never ran, so masking it would drop a sample
  for a reason unrelated to the policy, and there is no reward to train it
  on.
- INFRA_ERROR and CANCELLED are never scored. Asynchronously, the sample is
  submitted again under the weights being served. After `max_attempts`
  failures of one sample, the scheduler abandons the whole group instead of
  training it incomplete.
- A rollout whose oldest call is more than `max_lag` updates behind is
  discarded and resubmitted. A rollout still running when its submission
  falls past that bound is cancelled without waiting for it to finish,
  because its first call was made under the submission's version.

The scheduler keeps no clock of its own: a deadline belongs to the
operation that can stop when it passes. The remote rollout servers bound
each request, a sandboxed environment (`SubprocessEnvironment`) its
`wall_seconds`, and `HarnessSource` each trial (`trial_timeout`). A rollout
that runs out of time there comes back as an INFRA_ERROR, which the
scheduler retries. A reward's program past its sandbox's wall clock is a
TIMEOUT verdict that the reward scores, not a failure.

A source that raises an exception instead of returning a status is broken.
The scheduler cancels the batch's remaining rollouts and lets the exception
propagate.

Two options cut the long tail of slow rollouts, and both keep the batch
shape fixed. With `oversample`, each group runs that many extra samples and
is admitted once its first `groups` rollouts finish; the scheduler cancels
the rest. This is APRIL's active partial rollouts (arXiv:2509.18521)
without the carry-over. With `admit` below the task batch size, the
scheduler admits the first `admit` groups to complete and fit, and cancels
the others, as slime's over-sampling batch does. Both select by completion
time, which favors short rollouts; that bias is the price of not waiting on
the tail. Rollouts that are running when new weights are pushed keep
running. Their later calls are stamped with the new version, and a
rollout's staleness is that of its oldest call, as in Kimi K2's partial
rollouts (arXiv:2507.20534 section 3.3.4).

Asynchronously, the scheduler pushes weights through `weights` when the
served version falls `sync_every` updates behind the trainer, or is ahead
of it, as after a resume from an earlier checkpoint. A first submission is
therefore at most `ahead + sync_every - 1` updates stale when the trainer
consumes it, and the constructor refuses a `max_lag` below that. If a
complete group's chains do not fit in `rows` beside the groups admitted
before it, the scheduler cuts that group, so an overfull batch never
reaches the step. The proximal policy is the trainer's current weights. The
scheduler rescores the packed rows under them with
`GRPOObjective.packed_log_probs` (decoupled PPO, AReaL arXiv:2505.24298).
Sources report only behavior likelihoods, and the objective's
`behavior_importance` weights each token by the ratio of its proximal to
its behavior likelihood.

In a multi-process trainer, every process runs its own scheduler over the
task rows its data stream reads, and packs its own rows. The step's batch
is all the processes' rows together, sharded as the trainer shards any
batch, and no rollout moves between processes. The processes act together
at a few points in each call, each an agreement point over the trainer's
mesh where every process raises if one failed, in this order: the call's
preparation, the acquisition and its admission, the packing, the engine
records' layout, the first reader's rows and the proximal rescoring. A
weight push is one call every process makes, so the publisher has to agree
on it across processes. When a tensor or sequence axis spans processes,
several processes read the same share of the data. Then only the share's
first reader packs, and the others train on the rows it packed
(`first_reader_batch`), its recorded likelihoods included, which
independent draws would not give them. An asynchronous source samples on
the first reader alone; a collector's draw is one program, which every
reader enters. Without a mesh nothing is agreed, and each process collects
on its own.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

import jax
import numpy as np

from dew._coordination import agreed
from dew.batch import Batch, Variables
from dew.cleanup import attempted
from dew.data.dataset import DataPartition, Loader, tapped
from dew.distributed.batches import first_reader_batch, shard_batch
from dew.nn.inputs import local_rows, mesh_of
from dew.objectives.base import Shown
from dew.sampling.text import Sampling
from dew.state import TrainState

from ..lm import TokenScorer
from .episodes import EnvironmentFactory, EpisodeCollector, EpisodeInference, EpisodeRecorder, Verifier
from .grpo import GRPOObjective
from .journal import EpisodeJournal
from .rollout import PolicyCollector, Reward, prompt_tasks
from .sessions import (
    OLD_LOG_PROBS_KEY,
    RESPONSE_MASK_KEY,
    Collector,
    Session,
    SessionSource,
    Snapshot,
    Status,
    Task,
    agreed_records,
    chain_lengths,
    check_credit,
    check_estimator,
    check_truncation,
    local_task_ids,
    pack,
    rows_needed,
    sampled_values,
    session_metrics,
    trained_reward,
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
      `cancelled`, `stale`, `unscored`).
    - `cancelled` counts in-flight rollouts cancelled because they were
      surplus, stale or in an abandoned group.
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
    """Return one task per integer `task_id` row of `batch` (`local_task_ids`), named by the id in
    decimal."""
    return [Task(str(int(value))) for value in local_task_ids(batch)]


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
    uniform: int = 0
    """Complete groups left out whose trained returns all agreed (`RolloutScheduler.varied`)."""


def _done(session: Session) -> Future[Session]:
    future: Future[Session] = Future()
    future.set_result(session)
    return future


class Synchronous:
    """Draws each call's groups at the call from the trainer's own variables through `collector`
    (the module docstring's synchronous case); `Snapshot` is what the collector draws from."""

    lag = 0
    rescore = None
    """None: the collected calls' raw likelihoods are the proximal policy's, drawn from the variables
    the step trains, so `old_log_probs` are those and no rescoring forward runs."""

    def __init__(self, collector: Collector):
        self.collector = collector
        self._serial = 0

    @property
    def width(self) -> int:
        return self.collector.width

    def rows(self, tasks: int, samples: int) -> int:
        return self.collector.rows(tasks, samples)

    def tasks(self, dataset: Loader, tasks_of: Callable[[Batch], Sequence[Task]]) -> Loader:
        """`dataset` itself: a call's tasks are its own batch's, and nothing is opened ahead."""
        return dataset

    def close(self) -> None:
        """Nothing is in flight between calls."""

    def begin(self, snapshot: Snapshot, partition: DataPartition) -> None:
        """Nothing is published: the collector draws from the snapshot itself."""

    def entries(self, batch: Batch, tasks_of: Callable[[Batch], Sequence[Task]]) -> list[_Entry]:
        entry = _Entry(self._serial, tuple(tasks_of(batch)))
        self._serial += 1
        return [entry]

    def acquires(self, partition: DataPartition) -> bool:
        """Every process: the draw is one program every process of the mesh enters."""
        return True

    def open(self, entry: _Entry, samples: int, snapshot: Snapshot) -> None:
        """Collect the entry's groups. A share's later reader enters the draw, one program for the
        share, and its collector returns nothing past it (`sessions.Collector`)."""
        # Every session was submitted when the collection began.
        began = time.perf_counter()
        collected = self.collector.collect(snapshot, entry.tasks, samples)
        if not collected:
            return
        for row, (task, sessions) in enumerate(zip(entry.tasks, collected, strict=True)):
            if len(sessions) != samples:
                raise ValueError(f"the collector drew {len(sessions)} sessions of task {task.id}, "
                                 f"not {samples}")
            entry.groups.append(_Group(task, f"{entry.serial}/{row}", [
                _Sample(index, 0, snapshot.updates, _done(session), began)
                for index, session in enumerate(sessions)]))

    def redrawn(self, group: _Group, sample: _Sample, cause: str) -> None:
        """None: the draw was one program every process entered, which no process reruns alone."""
        return

    def cancel(self, samples: Sequence[_Sample]) -> None:
        """Nothing runs: every collected session is finished."""

    def stale(self, sample: _Sample, updates: int) -> bool:
        return False


class Asynchronous:
    """Keeps rollouts from `source` in flight ahead of the trainer (the module docstring's
    asynchronous case), pushing the trainer's weights through `weights`.

    `width` and `rows` fix the packed batch shape; `rows` is this process's
    share of a multi-process trainer's batch. `objective` rescores the
    packed rows under the trainer's weights, the proximal policy, and a
    `max_lag` above 0 needs its `behavior_importance` (a TIS cap or an
    IcePop band); the constructor raises ValueError without it.
    """

    def __init__(self, source: SessionSource, weights: Publisher, objective: GRPOObjective, *,
                 width: int, rows: int, ahead: int = 1, max_lag: int = 1, sync_every: int = 1,
                 max_attempts: int = 3):
        for name, value, least in (("width", width, 2), ("rows", rows, 1), ("ahead", ahead, 0),
                                   ("sync_every", sync_every, 1), ("max_lag", max_lag, 0),
                                   ("max_attempts", max_attempts, 1)):
            if type(value) is not int or value < least:
                raise ValueError(f"{name} must be an integer of at least {least}")
        if ahead + sync_every - 1 > max_lag:
            raise ValueError(
                f"ahead={ahead} and sync_every={sync_every} let a batch fall {ahead + sync_every - 1} "
                f"updates behind, past max_lag={max_lag}")
        if max_lag > 0 and objective.behavior_importance is None:
            raise ValueError(
                "stale rollouts need the objective's behavior_importance, a TIS cap or an IcePop band: "
                "the proximal-to-behavior importance weight is the off-policy correction"
            )
        self.source, self.weights, self.objective = source, weights, objective
        self.width, self._rows = width, rows
        self.ahead, self.lag, self.sync_every, self.max_attempts = ahead, max_lag, sync_every, max_attempts
        self._lock = threading.Lock()
        self._registered: deque[_Entry] = deque()
        self._serial = 0
        self._tasks_of: Callable[[Batch], Sequence[Task]] | None = None
        # The rollout key of the call in progress and its submissions so far.
        self._key: jax.Array = jax.random.key(0)
        self._submissions = 0

    def rows(self, tasks: int, samples: int) -> int:
        return self._rows

    def tasks(self, dataset: Loader, tasks_of: Callable[[Batch], Sequence[Task]]) -> Loader:
        """`dataset` with a training stream that registers each task batch ahead of the step.

        Opening the stream again, as a resume does, cancels every rollout
        the previous stream left in flight.
        """
        self._tasks_of = tasks_of
        opened = tapped(dataset.train, self._register)

        def train(partition: DataPartition) -> Iterator[Batch]:
            self.close()
            return opened(partition)

        return dataclasses.replace(dataset, train=train)

    def close(self) -> None:
        """Cancel every rollout in flight. The caller closes the source; this method does not."""
        with self._lock:
            entries = list(self._registered)
            self._registered.clear()
        self.cancel([sample for entry in entries for group in entry.groups for sample in group.live])

    def _register(self, batch: Batch) -> None:
        assert self._tasks_of is not None, "tasks() names the conversion before its stream opens"
        tasks = tuple(self._tasks_of(batch))
        with self._lock:
            self._registered.append(_Entry(self._serial, tasks))
            self._serial += 1

    def begin(self, snapshot: Snapshot, partition: DataPartition) -> None:
        """Fold the call's key for the engines and push the weights where the served version is
        `sync_every` behind the trainer's, or ahead of it.

        The engines draw with JAX's default PRNG on devices of their own, so
        every word of the trainer's key, of whatever implementation, folds
        into an uncommitted default key, and this data share's index after
        them, so two shares' readers draw apart. A served version ahead of
        `updates` is out of date as well: a fit restored from an earlier
        checkpoint must not sample from weights newer than its own.
        """
        words = [int(word)
                 for word in np.asarray(jax.device_get(jax.random.key_data(snapshot.key))).reshape(-1)]
        self._key = functools.reduce(jax.random.fold_in, (*words, partition.index), jax.random.key(0))
        self._submissions = 0
        updates = snapshot.updates
        if 0 <= updates - self.weights.version < self.sync_every:
            return
        self.weights.load(snapshot.variables, updates)
        if self.weights.version != updates:
            raise RuntimeError(f"the weights pushed at update {updates} did not take: the engines still "
                               f"serve version {self.weights.version}")

    def entries(self, batch: Batch, tasks_of: Callable[[Batch], Sequence[Task]]) -> list[_Entry]:
        with self._lock:
            if not self._registered:
                raise ValueError(
                    "an asynchronous RolloutScheduler's batch comes from the stream of "
                    "RolloutScheduler.tasks(dataset)")
            entry = self._registered.popleft()
            upcoming = list(self._registered)[:self.ahead]
        if tuple(tasks_of(batch)) != entry.tasks:
            self.cancel([sample for group in entry.groups for sample in group.live])
            raise ValueError("the trainer's batch is not the next registered task batch")
        return [entry, *upcoming]

    def acquires(self, partition: DataPartition) -> bool:
        """The share's first reader: independent draws would hand its readers different rows."""
        return partition.reader == 0

    def open(self, entry: _Entry, samples: int, snapshot: Snapshot) -> None:
        """Submit the entry's groups, unless an earlier call opened them; the snapshot's weights
        reach the engines through `begin`."""
        if entry.groups:
            return
        for row, task in enumerate(entry.tasks):
            group = _Group(task, f"{entry.serial}/{row}")
            entry.groups.append(group)
            self._submit(group, [(index, 0) for index in range(samples)])

    def _submit(self, group: _Group, samples: Sequence[tuple[int, int]]) -> None:
        """Submit `(index, attempt)` samples of one group under the served
        version, each submission of a call drawing its own fold of the call's key."""
        version = self.weights.version
        key = jax.random.fold_in(self._key, self._submissions)
        self._submissions += 1
        futures = self.source.submit(group.task, len(samples), version=version, key=key)
        group.live.extend(_Sample(index, attempt, version, future)
                          for (index, attempt), future in zip(samples, futures, strict=True))

    def redrawn(self, group: _Group, sample: _Sample, cause: str) -> Callable[[], None] | None:
        """A resubmission under the weights being served; None once one sample has failed
        `max_attempts` times. A stale sample is not a failure."""
        if cause != "stale":
            group.failures[sample.index] += 1
            if group.failures[sample.index] >= self.max_attempts:
                return None
        return functools.partial(self._submit, group, [(sample.index, sample.attempt + 1)])

    def cancel(self, samples: Sequence[_Sample]) -> None:
        if samples:
            self.source.cancel([sample.future for sample in samples])

    def stale(self, sample: _Sample, updates: int) -> bool:
        return updates - sample.submitted > self.lag

    def rescore(self, variables: Variables, packed: Mapping[str, np.ndarray]) -> np.ndarray:
        """The packed ids' proximal log-probabilities under `variables`, the trainer's weights.

        Over a mesh the rows are placed as the step places the batch, every
        process's rows together, so the rescoring runs once over the pool's
        batch and each process reads its own rows back."""
        mesh = mesh_of(variables)
        rescore = self.objective.program("packed log probs", self.objective.packed_log_probs)
        scored = rescore(variables, packed if mesh is None else shard_batch(mesh, dict(packed)))
        return local_rows(scored).astype(np.float32) * packed[RESPONSE_MASK_KEY]


def _within[T](mesh, phase: str, operation: Callable[[], T]) -> T:
    """`operation`, agreed across the processes of `mesh` (`dew._coordination.agreed`); a process
    without one runs it alone."""
    return operation() if mesh is None else agreed(phase, operation)


class RolloutScheduler:
    """Admits complete groups of rollouts from `acquisition` and packs them into the trainer's batches.

    The module docstring describes `groups`, `oversample`, `admit` and the
    two acquisitions. The other arguments:

    - `tasks` turns one batch into its tasks (`task_ids` for integer
      `task_id` rows, `prompt_tasks` for prompt rows).
    - `estimator`, `truncation` and `credit` are `pack`'s advantage family,
      truncation policy and credit (per session, or per turn from the step
      rewards still ahead), and `support_capacity` is its per-row support
      length, which a filtered-sampling source requires.
    - `log`, when given, receives a `SchedulerRecord` of this process's
      rollouts after each call. A process that is not the first reader of
      its share packs nothing and logs nothing.

    `metrics` holds the latest call's numbers, which the trainer logs as
    `rollout/<name>`.
    """

    shown: ClassVar[dict[str, Shown]] = {
        "reward/mean": Shown(better="higher"),
        "lag/mean": Shown(better="lower"),
        "pack/fill": Shown(better="higher", percent=True),
        **{f"status/{status.value}": Shown(percent=True) for status in Status},
    }

    def __init__(self, acquisition: Synchronous | Asynchronous, *,
                 tasks: Callable[[Batch], Sequence[Task]] = task_ids, groups: int = 4, oversample: int = 0,
                 admit: int | None = None, estimator: str = "group", truncation: str = "mask",
                 credit: str = "session", support_capacity: int | None = None,
                 log: Callable[[SchedulerRecord], None] | None = None, varied: bool = False,
                 overlong: tuple[int, int] | None = None):
        for name, value, least in (("groups", groups, 2), ("oversample", oversample, 0)):
            if type(value) is not int or value < least:
                raise ValueError(f"{name} must be an integer of at least {least}")
        check_truncation(truncation)
        check_credit(credit)
        check_estimator(estimator)
        if admit is not None and (type(admit) is not int or admit < 1):
            raise ValueError("admit must be a positive number of groups, or None to admit every task")
        self.acquisition, self.tasks_of = acquisition, tasks
        self.groups, self.oversample, self.admit = groups, oversample, admit
        self.estimator, self.truncation, self.credit, self.log = estimator, truncation, credit, log
        self.support_capacity = support_capacity
        # DAPO's dynamic sampling (arXiv:2503.14476, equation 11): a group whose members' trained returns
        # all agree gives every member a zero advantage, so it is left out and admission waits for one
        # whose returns vary.
        self.varied = varied
        # DAPO's soft overlong punishment: the budget and the cache before it (`sessions.overlong_penalty`).
        if overlong is not None and not (len(overlong) == 2 and 0 < overlong[1] <= overlong[0]):
            raise ValueError("overlong is (L_max, L_cache), a positive cache within the budget")
        self.overlong = overlong
        self.metrics: dict[str, float] = {}
        self._branch: str | None = None

    @classmethod
    def sampled(cls, objective: TokenScorer, reward: Reward, *,
                decode: Callable[[Sequence[int]], str] = lambda ids: " ".join(str(token) for token in ids),
                groups: int = 4, max_new_tokens: int = 32, estimator: str = "group",
                truncation: str = "score", sampling: Sampling | None = None,
                log: Callable[[SchedulerRecord], None] | None = None, varied: bool = False,
                overlong: tuple[int, int] | None = None) -> RolloutScheduler:
        """Draw `groups` completions of each prompt row from `objective`'s policy at every call and
        train on them (`PolicyCollector` over `prompt_tasks`; GRPO's synchronous loop). A completion
        that runs out of `max_new_tokens` trains on its reward, as `truncation='score'` says.

        The calls' raw likelihoods are the proximal policy's (`Synchronous.rescore`), the policy at
        temperature 1 over the whole vocabulary: a GRPO objective whose ratio reads its policy at another
        `sampling_temperature` is refused, since its unchanged weights would not give a ratio of 1."""
        if isinstance(objective, GRPOObjective) and objective.sampling_temperature != 1.0:
            raise ValueError(
                "a synchronous collection's proximal likelihoods are the policy's raw ones, at temperature "
                f"1, and this objective reads its ratio at sampling_temperature="
                f"{objective.sampling_temperature}; set it to 1.0, or collect asynchronously, where the "
                "proximal policy is rescored")
        collector = PolicyCollector(objective, reward, decode, max_new_tokens,
                                    Sampling() if sampling is None else sampling)
        return cls(Synchronous(collector), tasks=prompt_tasks, groups=groups, estimator=estimator,
                   truncation=truncation, log=log, varied=varied, overlong=overlong)

    @classmethod
    def episodes(cls, policy: EpisodeInference, environment: EnvironmentFactory, verifier: Verifier, *,
                 max_prompt_tokens: int, max_new_tokens: int, max_turns: int, sampling: Sampling,
                 groups: int = 2, credit: str = "session", record: EpisodeRecorder | None = None,
                 journal: EpisodeJournal | None = None,
                 log: Callable[[SchedulerRecord], None] | None = None) -> RolloutScheduler:
        """Collect `groups` tool episodes of each `task_id` row at every call, in lock step under the
        step's variables, and train on them (`EpisodeCollector`; truncated episodes are masked)."""
        collector = EpisodeCollector(policy, environment, verifier, max_prompt_tokens, max_new_tokens,
                                     max_turns, sampling, record, journal)
        return cls(Synchronous(collector), groups=groups, credit=credit, log=log)

    def tasks(self, dataset: Loader) -> Loader:
        """`dataset` as the acquisition reads it: asynchronously, a stream that registers each task
        batch ahead of the step; synchronously, `dataset` itself."""
        return self.acquisition.tasks(dataset, self.tasks_of)

    def close(self) -> None:
        """Cancel every rollout in flight. The caller closes the source; this method does not."""
        self.acquisition.close()

    @contextlib.contextmanager
    def branch(self, name: str) -> Iterator[None]:
        """The calls of the cooldown branch `name` (`dew.training.rollout`): drawn under the branch's
        states, on its own branch of a collector's records (`Snapshot.branch`: an `EpisodeJournal`'s
        cohorts, which a resumed branch reads back as its own), its records reach no `log`, and the
        run's latest `metrics` stand after it. The run closes its own stream for the branch and opens
        it again at its position after (`Trainer.fit`), and an asynchronous acquisition's stream
        cancels the rollouts in flight as it opens (`tasks`): the run's as the branch's opens, which
        the branch's weights would otherwise finish, and the branch's, cancelled here, as it ends. The
        engines serve the branch's weights meanwhile, and the run's again from its next call
        (`Asynchronous.begin`: a served version ahead of the run's updates is out of date)."""
        metrics, log, branch = self.metrics, self.log, self._branch
        self.log, self._branch = None, name
        try:
            yield
        except BaseException as failure:
            # The branch's failure stays the one raised; a cancellation's is noted on it.
            attempted([("cancelling the branch's rollouts", self.close)], primary=failure)
            raise
        else:
            self.close()
        finally:
            self.metrics, self.log, self._branch = metrics, log, branch

    def _replace(self, group: _Group, sample: _Sample, cause: str, tally: _Tally) -> None:
        """Draw a sample that cannot join its group again, or abandon the group where the
        acquisition may not (`Asynchronous.redrawn`, `Synchronous.redrawn`)."""
        redraw = self.acquisition.redrawn(group, sample, cause)
        if redraw is None:
            group.abandoned = True
            tally.abandoned += 1
            tally.cancelled += len(group.live)
            self.acquisition.cancel(group.live)
            group.live.clear()
            return
        if len(group.done) + len(group.live) >= self.groups:
            return  # an over-sampled spare takes its place
        tally.resubmitted[cause] += 1
        redraw()

    def _harvest(self, entry: _Entry, group: _Group, updates: int, tally: _Tally) -> None:
        """Move one group's finished rollouts into it, replacing what cannot be admitted.

        A group stops taking rollouts once it is full or abandoned; its
        remaining samples are left for `_admit`'s final cancel.
        """
        stale = [sample for sample in group.live
                 if not sample.future.done() and self.acquisition.stale(sample, updates)]
        if stale:
            tally.cancelled += len(stale)
            self.acquisition.cancel(stale)
            group.live = [sample for sample in group.live if sample not in stale]
        finished = [(sample, None) for sample in group.live if sample.future.done()]
        for sample, cause in [*finished, *((sample, "stale") for sample in stale)]:
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
        if updates - oldest > self.acquisition.lag:
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
        """Admit a complete group when its chains fit the batch's rows beside those admitted before
        it; cut it otherwise.

        A group that cannot fit the rows on its own is refused with a
        ValueError: the rows are too few for the run, not for this batch.

        A session whose calls do not extend each other packs as one chain
        per call, so the rows a group needs are known only once it is done.
        """
        width, rows = self.acquisition.width, self.acquisition.rows(len(entry.tasks), self.groups)
        group.lengths = chain_lengths(group.done, width, truncation=self.truncation)
        alone = rows_needed(group.lengths, width)
        if alone > rows:
            raise ValueError(
                f"one group of task {group.task.id} needs {alone} rows of {width} ids, "
                f"more than rows={rows}; size rows for sessions whose calls split into chains"
            )
        if self.varied and len({trained_reward(session, self.truncation) for session in group.done}) < 2:
            tally.uniform += 1
            return
        admitted = [length for complete in entry.complete for length in complete.lengths]
        if rows_needed([*admitted, *group.lengths], width) <= rows:
            entry.complete.append(group)
            return
        tally.cut += 1
        tally.cancelled += len(group.live)
        self.acquisition.cancel(group.live)
        group.live.clear()

    def _admit(self, entry: _Entry, updates: int, tally: _Tally) -> list[_Group]:
        """Wait for the entry's first `admit` complete groups that fit, then cancel the rest."""
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
                live = [sample.future for group in pending for sample in group.live]
                wait(live, return_when=FIRST_COMPLETED)
        finally:
            leftover = [sample for group in entry.groups for sample in group.live]
            tally.cancelled += len(leftover)
            self.acquisition.cancel(leftover)
            for group in entry.groups:
                group.live.clear()
        if not entry.complete:
            raise RuntimeError(f"no group of batch {entry.serial} completed: every one was abandoned")
        return entry.complete[:target]

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> dict[str, np.ndarray]:
        mesh = mesh_of(state.variables)
        partition = DataPartition() if mesh is None else DataPartition.of(mesh)
        snapshot = Snapshot(state.variables, int(state.step), int(state.updates), key, self._branch)
        self.acquisition.begin(snapshot, partition)
        entries = _within(mesh, "rollout preparation", lambda: self.acquisition.entries(batch, self.tasks_of))
        acquiring = self.acquisition.acquires(partition)
        rollouts, latencies, groups, tally, waited = _within(
            mesh, "rollout admission",
            lambda: self._admitted(entries, snapshot, partition) if acquiring else ([], [], 0, _Tally(), 0.0))
        sampling = partition.reader == 0
        # A share's packing that fails (a support past its capacity) fails every process, not one.
        packed = _within(mesh, "rollout packing",
                         lambda: self._packed(entries[0], rollouts) if sampling else {})
        if mesh is not None:
            packed = first_reader_batch(mesh, agreed_records(packed, self.support_capacity))
        if self.acquisition.rescore is not None:
            packed[OLD_LOG_PROBS_KEY] = self.acquisition.rescore(state.variables, packed)
        if sampling:
            updates = snapshot.updates
            versions = [call.version for rollout in rollouts for call in rollout.calls]
            oldest = min(versions, default=updates)
            metrics = session_metrics(rollouts, packed, latencies=latencies, version=updates,
                                      truncation=self.truncation)
            record = SchedulerRecord(updates, oldest, updates - oldest, groups, dict(tally.resubmitted),
                                     tally.cancelled, tally.abandoned, tally.cut, waited, metrics)
            self.metrics = {**metrics, "lag": float(record.lag), "groups": float(groups),
                            "resubmitted": float(sum(record.resubmitted.values())),
                            "waited_seconds": waited, "uniform_groups": float(tally.uniform)}
            if self.log is not None:
                self.log(record)
        return packed

    def _admitted(self, entries: Sequence[_Entry], snapshot: Snapshot, partition: DataPartition
                  ) -> tuple[list[Session], list[float], int, _Tally, float]:
        """This process's admitted rollouts for the call's entry, their latencies, the group count,
        the tally and the wait, the entries ahead of it opened as well. A share's later reader opens
        what the first does and admits nothing: it trains on the first reader's rows."""
        # The wait counts a synchronous collection, which opening its entry runs.
        began = time.perf_counter()
        for pending in entries:
            self.acquisition.open(pending, self.groups + self.oversample, snapshot)
        if partition.reader != 0:
            return [], [], 0, _Tally(), time.perf_counter() - began
        tally = _Tally()
        admitted = self._admit(entries[0], snapshot.updates, tally)
        waited = time.perf_counter() - began
        return ([rollout for group in admitted for rollout in group.done[:self.groups]],
                [latency for group in admitted for latency in group.latencies[:self.groups]],
                len(admitted), tally, waited)

    def _packed(self, entry: _Entry, rollouts: Sequence[Session]) -> dict[str, np.ndarray]:
        """The admitted rollouts packed into this call's rows, with the proximal likelihoods the
        calls recorded where the acquisition drew them from the step's variables."""
        packed = pack(rollouts, self.acquisition.width,
                      rows=self.acquisition.rows(len(entry.tasks), self.groups),
                      estimator=self.estimator, truncation=self.truncation, credit=self.credit,
                      support_capacity=self.support_capacity, overlong=self.overlong)
        if self.acquisition.rescore is None:
            packed[OLD_LOG_PROBS_KEY] = sampled_values(
                packed, lambda index, call: _recorded(rollouts[index], call))
        return packed


def _recorded(rollout: Session, call: int) -> tuple[float, ...]:
    """The raw likelihoods call `call` of `rollout` recorded."""
    likelihoods = rollout.calls[call].raw_log_probs
    if likelihoods is None:
        raise ValueError(f"call {call} of {rollout.task}/{rollout.group}/{rollout.sample} recorded no raw "
                         "likelihoods, which a synchronous collection's proximal policy reads")
    return likelihoods


__all__ = ["Asynchronous", "Publisher", "RolloutScheduler", "SchedulerRecord", "Synchronous",
           "task_ids"]
