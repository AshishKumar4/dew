"""One `Trainer.fit` call's records: what it was asked for (`FitPlan`, its validation splits resolved by
`validation_suites`), what its loop carries from step to step (`FitRun`), and how the loop reads its
training stream (`waited_read`)."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import functools
import math
import time
import types
from collections.abc import Callable, Iterator, Mapping, Sequence

import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.early_stopping import EarlyStopping
from jax.sharding import Mesh

from dew import position
from dew._coordination import agreed, agreed_next, agreed_reads
from dew._records import JSON, integers
from dew.batch import CLOSES_WINDOW, CONSUMED_COUNT, EXAMPLES_CONSUMED, VALID_ROWS, Batch, ConsumedCount
from dew.checkpoints import Checkpoints
from dew.data.compile import Handed, native
from dew.data.dataset import Checkpointable, Closeable, DataPartition, Loader, PhasedStream, Reader, rows_of
from dew.data.pipeline import Dataset
from dew.distributed.batches import DevicePrefetchIterator, Signature, restore_position
from dew.distributed.placement import MeshSpec, Placement
from dew.objectives.base import Initializer, Metric, Objective, TrainingScalar
from dew.optim import OptimConfig, reclocked
from dew.state import TrainState
from dew.telemetry.profile import Profiler, region
from dew.training.budget import Pause, Plateau, every_step, recorded
from dew.training.cadence import Cadence, Counted
from dew.training.cooldown import Branch
from dew.training.evaluation import EvalSuite, Evaluation, Summary
from dew.training.interval import Interval, Taken, TokenClock
from dew.training.model_flops import TOKENS
from dew.training.observe import Observe
from dew.training.profiling import ProfileWindow, start_capture
from dew.training.runtime import PreemptionNotice
from dew.training.selection import Best, HeldBest


def validation_suites(dataset: Loader, validation: Mapping[str, Reader | EvalSuite] | None,
                      metrics: Sequence[Metric], eval_every: Cadence) -> dict[str, EvalSuite]:
    """Every split `fit` validates, by name, with its metrics and cadence
    resolved: `dataset.val` alone is the one suite `val`."""
    if validation is None:
        if metrics and dataset.val is None:
            # Else the metrics would never be scored, and the run would not say so.
            raise ValueError("metrics score validation data: give the dataset a val split, or validation=")
        # `val` is read only when a pass is scheduled; a dataset need not hold one.
        validation = {"val": dataset.val} if (eval_every or metrics) and dataset.val is not None else {}
    elif not validation:
        raise ValueError("validation must contain a split")
    suites = {}
    for name, entry in validation.items():
        suite = entry if isinstance(entry, EvalSuite) else EvalSuite(entry)
        suites[name] = EvalSuite(suite.data, tuple(suite.metrics) or tuple(metrics),
                                 eval_every if suite.every is None else suite.every)
    return suites


def split_metrics(suites: Mapping[str, EvalSuite]) -> dict[str, tuple[Metric, ...]]:
    """Each split's own metrics, by its name: two suites may each hold their own instance of one
    metric."""
    return {name: tuple(suite.metrics) for name, suite in suites.items()}


def declarations(suites: Mapping[str, EvalSuite], metrics: Sequence[Metric],
                 summaries: Sequence[Summary], *, loss: bool) -> dict[str, tuple[Metric | Summary, ...]]:
    """Every score a fit declares, by split: each suite's metrics (fit's own `metrics` as `val` where it
    validates no suite) and the summaries in their namespace (`Summary.SPLIT`), which no split can be
    named. Its selections resolve against this and its rankings read their aggregates' scores off it,
    one declaration for both. A summary is refused where it repeats another's name or weighs a score
    no evaluation records (`Summary.check`, the objective's validation loss only where it defines one,
    `loss`)."""
    names = [summary.name for summary in summaries]
    if len(names) != len(set(names)):
        raise ValueError(f"summaries are named apart, and {names} repeats one")
    for summary in summaries:
        summary.check(suites, loss)
    declared: dict[str, tuple[Metric | Summary, ...]] = {**(split_metrics(suites) or {'val': tuple(metrics)})}
    if summaries:
        declared[Summary.SPLIT] = tuple(summaries)
    return declared


@dataclasses.dataclass(frozen=True)
class FitPlan:
    """What one `Trainer.fit` call was asked for."""

    dataset: Loader
    steps: int | None
    log_every: int | Counted
    eval_every: Cadence
    tokens: int | None
    checkpoint_every: Cadence | datetime.timedelta
    checkpoint_final: bool
    local_every: Cadence
    metrics: Sequence[Metric]
    preview: bool
    scored_to: str | None = None
    """Where each evaluation writes every validation batch's scoring artifacts (`Evaluation.run`)."""
    best: tuple[Best, ...] = ()
    stop: Plateau | None = None
    pause: tuple[int | None, int | None, int | None] = (None, None, None)
    """The step, the real tokens and the consumed examples a `Pause` ends the run at (`Pause.at`)."""
    suites: Mapping[str, EvalSuite] = dataclasses.field(default_factory=dict)
    held_best: bool = False  # a restore_best without Checkpoints (`HeldBest`)
    """Every validation split by name, its metrics and cadence resolved."""
    summaries: tuple[Summary, ...] = ()
    """The scalars each step whose evaluations score every split they weigh forms (`Summary`)."""
    branches: tuple[Branch, ...] = ()
    """The run's cooldown branches (`dew.training.cooldown`)."""
    declared: Mapping[str, tuple[Metric | Summary, ...]] = dataclasses.field(default_factory=dict)
    """Every score the fit declares, by split (`declarations`)."""
    observe: Observe | None = None

    @property
    def validation(self) -> bool:
        return bool(self.suites)

    def finished(self, step: int, real: int) -> bool:
        """Whether either run-length cap has been reached, after `real` tokens."""
        return ((self.steps is not None and step >= self.steps)
                or (self.tokens is not None and real >= self.tokens))

    def paused(self, step: int, real: int, consumed: int = 0) -> bool:
        """Whether the run has reached where its `Pause` ends it, after `real` tokens and `consumed`
        examples."""
        at, tokens, examples = self.pause
        return ((at is not None and step >= at) or (tokens is not None and real >= tokens)
                or (examples is not None and consumed >= examples))

    def ended(self, step: int, real: int, consumed: int = 0) -> bool:
        """Whether this fit has nothing left to train: its budget reached, or its pause."""
        return self.finished(step, real) or self.paused(step, real, consumed)

    def horizon(self, real: int) -> int | None:
        """The least count of committed tokens past `real` at which something this plan reads of the
        clock changes: its budget, its pause, the next multiple of each cadence of tokens, a cooldown
        branch's tokens, or the end of a schedule's stage. None where nothing it reads does."""
        cadences = (self.log_every, self.eval_every, self.checkpoint_every, self.local_every,
                    *(suite.every for suite in self.suites.values()))
        nexts = [(real // every.every + 1) * every.every for every in cadences
                 if isinstance(every, Counted) and every.clock == "tokens"]
        branches = [branch.at.every for branch in self.branches
                    if isinstance(branch.at, Counted) and branch.at.clock == "tokens"]
        ends = (self.tokens, self.pause[1], *(self.dataset.token_ends or ()), *nexts, *branches)
        return min((end for end in ends if end is not None and end > real), default=None)

    def committed(self, interval: Interval, clock: TokenClock, state: TrainState) -> int:
        """The run's committed tokens after its latest step, as this plan reads them: read off the
        devices, with the tokens `state`'s open window holds, where the clock's bound reaches the next
        count the plan reads (`horizon`), and otherwise the count of its last read, which falls on no
        budget, cadence or stage the latest step's count could not, so the loop does not wait on that
        step."""
        horizon, bound = self.horizon(clock.read), clock.bound
        if horizon is not None and (bound is None or bound >= horizon):
            clock.reread(*interval.clocked(state.accumulation))
        return clock.read

    @property
    def scored(self) -> tuple[Metric, ...]:
        """Fit's metrics and every suite's, each instance once: two suites' instances of one metric
        are both a selection's to name."""
        every = (*self.metrics, *(metric for suite in self.suites.values() for metric in suite.metrics))
        return tuple({id(metric): metric for metric in every}.values())

    @property
    def cadenced(self) -> bool:
        """Whether any evaluation runs before the end."""
        return bool(self.eval_every or any(suite.every for suite in self.suites.values()))

    @property
    def evaluates_midway(self) -> bool:
        """Whether an evaluation can run before the end's: on a cadence shorter than the run, or to
        rank a checkpoint written before its end."""
        def before_end(every: Cadence | str | datetime.timedelta) -> bool:
            if isinstance(every, Counted):
                length = {"tokens": self.tokens, "examples": self.dataset.examples}[every.clock]
                return length is None or every.every < length
            return bool(every) and (self.steps is None or not isinstance(every, int) or every < self.steps)

        cadences = (self.eval_every, *(suite.every for suite in self.suites.values()))
        return any(map(before_end, cadences)) or bool(
            (self.validation or self.best or self.stop) and before_end(self.checkpoint_every))

    @staticmethod
    def falls(every: Cadence | str | datetime.timedelta, step: int,
              counts: Mapping[str, tuple[int, int]] | None = None) -> bool:
        """Whether a cadence of steps (`int`) or of a counted clock (`Counted`, the step having brought
        each clock's count from the second of its `counts` to the first) falls on `step`; a duration or
        none never does here."""
        if isinstance(every, Counted):
            return every.due(counts or {})
        return isinstance(every, int) and bool(every) and step % every == 0

    @staticmethod
    def cadences(loader: Loader, validation: Mapping[str, Reader | EvalSuite] | None,
                 metrics: Sequence[Metric], steps: int | None, tokens: int | None, accumulation: int,
                 written: tuple[int | str, int | str | None, int | str | datetime.timedelta | None],
                 checkpoints: Checkpoints | None = None, examples: int | None = None) -> tuple[
                     int | Counted, Cadence, Cadence | datetime.timedelta, Cadence,
                     dict[str, EvalSuite]]:
        """`fit`'s log, validation and checkpoint cadences, its `checkpoints`' local one
        (`Checkpoints.local_every`), and its suites' (`validation_suites`), each
        written as steps, a duration or a share of the run such as "5%", resolved once against its planned
        length: the whole updates nearest that share of its `steps` microsteps, `accumulation` an update,
        at least one; of a budget of `tokens` alone, each time the committed tokens pass a multiple of that
        share of them, and of a run bounded to `examples` no pass counts steps ahead of, each time its
        consumed examples pass a multiple of that share (`dew.training.budget.every_step`). A schedule's
        warmup may be a share too
        (`dew.optim.Cosine.warmup_steps`), and the start records what each resolved to
        (`recorded`)."""
        def cadence(value, name: str) -> Cadence:
            return every_step(value, steps, tokens, accumulation, name=name, examples=examples)

        log_every, eval_every, checkpoint_every = written
        logged, evaluated = cadence(log_every, "log_every"), cadence(eval_every, "eval_every")
        if not logged:
            raise ValueError(f"log_every must be a positive cadence, got {log_every!r}")
        saved = checkpoint_every if isinstance(checkpoint_every, datetime.timedelta) else cadence(
            checkpoint_every, "checkpoint_every")
        suites = {name: dataclasses.replace(suite, every=cadence(suite.every, f"{name}'s every"))
                  for name, suite in validation_suites(loader, validation, metrics, evaluated).items()}
        local = None if checkpoints is None else cadence(checkpoints.local_every, "local_every")
        return logged, evaluated, saved, local, suites

    @property
    def example_points(self) -> tuple[int, ...]:
        """The consumed examples its cooldown branches of examples begin at (`Counted`)."""
        return tuple(branch.at.every for branch in self.branches
                     if isinstance(branch.at, Counted) and branch.at.clock == "examples")

    def recorded(self, config: OptimConfig | None, accumulation: int, *, clocked: bool = False
                 ) -> dict[str, JSON]:
        """The cadences this plan resolved and the warmups of `config`'s schedules over its updates,
        `accumulation` microsteps each, or as written over a `clocked` run's progress, as the run's
        start records them (`dew.telemetry.FitStarted`)."""
        warmups = {} if config is None else config.warmups(
            None if self.steps is None else self.steps // accumulation, clocked=clocked)
        suites = {f"{name}.every": recorded(suite.every) for name, suite in self.suites.items()
                  if suite.every != self.eval_every}
        local = {} if self.local_every is None else {"local_every": recorded(self.local_every)}
        if self.branches:
            local["cooldowns.at"] = [branch.recorded() for branch in self.branches]
        return {"log_every": recorded(self.log_every), "eval_every": recorded(self.eval_every),
                "checkpoint_every": recorded(self.checkpoint_every), **local, **suites,
                **{f"warmup_steps.{name}": value for name, value in warmups.items()}}

    def plateaued(self, run: FitRun, scores: Mapping[str, float]) -> bool:
        """Whether the run's `Plateau` stops it at this step's `scores`: its count of evaluations
        without improvement held in the run's control, so a resume counts on (`Checkpoints.control`)."""
        stop = self.stop
        if stop is None:
            return False
        assert isinstance(stop.metric, (Metric, TrainingScalar, Summary)), (
            "fit binds the stopping selector before stepping"
        )
        name = f'{stop.split}/{stop.metric.name}'
        if name not in scores:
            return False
        value = scores[name] if stop.mode == 'min' else -scores[name]
        if not math.isfinite(value):
            return False
        key = f'plateau:{name}'
        held = run.stop_control.get(key)
        rule = {'mode': stop.mode, 'evals': stop.evals, 'min_delta': stop.min_delta}
        if held is not None and held.get('rule') != rule:
            raise ValueError(
                "Plateau policy differs from the resumed checkpoint; resume with the same stopping rule"
            )
        if held is not None and held.get('step') == run.current:
            return held['bad'] >= stop.evals
        # Flax's recurrence, which stops at its `patience`-th bad evaluation past the first.
        stopping = EarlyStopping(min_delta=stop.min_delta, patience=stop.evals - 1)
        if held is not None:
            stopping = stopping.replace(best_metric=held['best'], patience_count=held['bad'])
        stopping = stopping.update(value)
        run.stop_control[key] = {'best': stopping.best_metric, 'bad': stopping.patience_count,
                                 'step': run.current, 'rule': rule}
        return stopping.should_stop

    def due(self, step: int, counts: Mapping[str, tuple[int, int]] | None = None) -> tuple[str, ...]:
        """The suites whose own cadence falls on `step`, which brought each counted clock's count as
        `counts` says (`falls`)."""
        return tuple(name for name, suite in self.suites.items() if self.falls(suite.every, step, counts))


@dataclasses.dataclass
class FitRun:
    """One `Trainer.fit` call's progress: what its loop carries from step
    to step and what its cleanup reads however the run ends."""

    started: float
    current: int = 0
    loss: jax.Array | None = None
    evaluation: Evaluation | None = None
    last_checkpoint: float = 0.0
    checkpoint_due: bool = False
    local_due: bool = False
    """Whether the local checkpoint cadence falls on the run's latest step (`FitPlan.local_every`)."""
    clock: TokenClock = dataclasses.field(default_factory=TokenClock)
    """The run's committed tokens as its budget, cadences and schedule read them (`FitPlan.committed`)."""
    stop_control: dict = dataclasses.field(default_factory=dict)
    last_local_saved: int | None = None
    stopped: bool = False
    training: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    source: Iterator[Batch] | None = None
    train: DevicePrefetchIterator | None = None
    partition: DataPartition | None = None
    """The share of the training data this process reads, where the stream opens again at a stage of
    a schedule (`Trainer._scheduled`)."""
    tracing: bool = False
    traced: int = 0
    other: float = 0.0
    first_step: float | None = None
    preempted: int | None = None
    paused: int | None = None
    notice: PreemptionNotice | None = None
    endless: bool = False
    taken: Taken | None = None
    scored: frozenset[str] = frozenset()
    """The suites `evaluation` scored at its step, which a run whose stream ends there scores once."""
    ahead: tuple[Batch | None] | None = None
    """The next batch, or None for the stream's end, read before the step that takes it, where a
    save had to know whether the run ends there (`ends_here`)."""
    pooled: bool = False
    """Whether the run is on the examples clock in a pool, whose shares end apart: each reads on, a share
    that ran out with a batch of the pool's tree emptied (`emptied`), until every one has, the examples
    each step consumed counted over the pool (`pooled_read`)."""
    emptied: Emptied | None = None
    """A pooled run's batches of the pool's tree with nothing in them, which a share that ran out steps
    with (`pooled_read`)."""
    best: HeldBest = dataclasses.field(default_factory=HeldBest)
    """The log taken at the latest logging step and not yet read."""
    stepped: tuple[jax.Array, dict[str, jax.Array], jax.Array] | None = None
    """The latest step's loss, objective metrics and acceptance, which the log of the interval it
    ends takes."""
    reads: dict[Signature, tuple[bool, ...]] = dataclasses.field(default_factory=dict)
    """Which leaves of a batch of each signature its compiled step reads
    (`step_reads`), which the prefetcher alone places."""
    compiled: dict[Signature, tuple] = dataclasses.field(default_factory=dict)
    """The step compiled for each batch signature the run met, with its measured FLOPs
    (`Trainer._compiled_for`), which its cooldown branches step too."""
    cooling: Mapping[str, object] | None = None
    """The cooldown branch the run is to run at its current step before it reads on, resolved
    (`dew.training.cooldown.Pending`): one its checkpoints kept unfinished, as the pool agreed it, or
    one due at its last step, which runs after the step's final save (`Trainer._wind_down`)."""

    def open_window(self, profile: ProfileWindow | None, profiler: Profiler | None, seen: int,
                    compiled: bool) -> None:
        """Open the profile window's capture before an iteration's first read, so the step row records
        the read it waits on, once its warmup steps have run and the run's first step has `compiled`:
        the window holds steps, not the compile."""
        if (profiler is not None and profile is not None and not self.tracing and self.traced == 0
                and seen >= profile.warmup and compiled):
            start_capture(profiler)
            self.tracing = True

    def committed_examples(self, interval: Interval, points: Sequence[int], state: TrainState,
                           window: int) -> tuple[int, int]:
        """Where the state's update window of `window` microsteps has just committed (read off `state`
        only while one of the cooldown `points` of examples lies past the last commit), the consumed
        examples now and at that commit, which a point is crossed between
        (`dew.training.cooldown.Branch.due`), so its branch starts where an update has committed, a
        rejected microbatch's window included; none crossed otherwise."""
        if not any(interval.consumed_committed < point <= interval.consumed for point in points):
            return 0, 0
        if int(state.microstep) % window:
            return 0, 0
        crossed = interval.consumed, interval.consumed_committed
        interval.consumed_committed = interval.consumed
        return crossed

    def ends_here(self, interval: Interval) -> bool:
        """Whether a stream that may run out has, so this step is the run's last: its next batch is
        read now, agreed as every such read is, and kept for the step that takes it (`ahead`), the
        position the step saves taken before it. A run whose stream is closed (`Trainer._wind_down`)
        ends on this step without another read, and an endless stream never ends here otherwise."""
        if self.ahead is None and self.train is None:
            return True
        if self.endless:
            return False
        if self.ahead is None:
            self.ahead = (waited_read(self, interval, agree=True),)
        return self.ahead[0] is None

    @property
    def ran_out(self) -> bool:
        """Whether the stream ran out after the batch the run steps now (`ends_here`'s read found its
        end), not closed before it ran out."""
        return self.ahead is not None and self.ahead[0] is None

    def closing(self, interval: Interval, batch: Batch, examples: int | None) -> tuple[Batch, bytes | None]:
        """`batch` marked where it is the last of a run on the examples clock (`examples`, its bound;
        `CLOSES_WINDOW`), so the update it ends commits with the microbatches its window holds, known by
        reading ahead (`ends_here`), every batch of the run marked so the step keeps one program; and
        the stream's position after it, taken before that read, which the step saves. Where its stream
        ran out, at its bound (`clocked`), the last also consumes the examples the bound read that no
        batch counted (`drained`). A run on another clock's batch is as read."""
        if examples is None:
            return batch, None
        position = None if self.train is None else self.train.source_state
        last = self.ends_here(interval)
        if self.ran_out:
            batch = drained(batch, interval.consumed, examples)
        return closes_window(batch, examples, last=last), position

    def held(self) -> contextlib.AbstractContextManager[None]:
        """The training stream's placements held for a block of device work beside the steps
        (`DevicePrefetchIterator.held`), none to hold once the stream is closed."""
        return contextlib.nullcontext() if self.train is None else self.train.held()

    def further(self, interval: Interval) -> Callable[[], Batch | None]:
        """A read of the training stream's next batch for a rollout that draws past its step's
        (`Rollout.draws_further`): from inside the step's hold on placements (`read_held`), agreed as a
        step's read is, its wait counted in `interval`; None where the stream ran out on every process.
        Its position is the stream's after it, which the step saves as any read's."""
        def read() -> Batch | None:
            assert self.train is not None
            asked = time.perf_counter()
            batch = agreed_next("rollout read", iter(self.train.read_held, None))
            interval.waited += time.perf_counter() - asked
            return batch

        return read

    def read(self) -> Batch:
        """The training stream's next batch, read through this run each time:
        a caller's alias would keep the closed stream reachable from a failed
        run's traceback."""
        assert self.train is not None
        return next(self.train)


CLOCK_LIMIT = 2 ** 31 - 1
"""The most examples a clock counts: the devices hold it as int32 (`TrainState.consumed`)."""


def clocked(examples: int | None, steps: int | None, *, custom: bool, rollout: bool,
            bound_ends: bool = True) -> int | None:
    """The examples bound of a run that no plan counts steps ahead of, which keeps its progress in the
    examples its batches consume (`TrainState.consumed`), its schedule and cadences reading it; None for
    a run counted otherwise. The transaction hands the run's optimizer that progress and the batches
    carry the count, which a `custom` step or a `rollout`'s batches do not: refused there. Its progress
    reaches its end where its read ends at its bound (`Loader.bound_ends`), so a take or slice after
    the bound, which would end it first, is refused, as is a bound past what the clock counts."""
    if examples is None or steps is not None:
        return None
    if custom or rollout:
        raise ValueError("a run bounded in examples that no plan counts ahead reads its progress off the "
                         "examples its batches consume, which a custom step= or a rollout does not carry; "
                         "give steps=, or read the data in a global order its plan counts")
    if not bound_ends:
        raise ValueError("a run bounded in examples that no plan counts ahead ends where its bound does, "
                         "which its schedule's end reads; a take or slice after the bound would end it "
                         "first: name the run's length once, by examples= or by the take")
    if examples > CLOCK_LIMIT:
        raise ValueError(f"examples={examples} is past the {CLOCK_LIMIT} a run's clock counts on its "
                         "devices; bound the run in steps or tokens")
    return examples


def drained(batch: Batch, consumed: int, bound: int) -> Batch:
    """A clocked run's last `batch`, its stream run out, holding the examples its bound read that no row
    it dispatched carried: an example rejected, or of no rows, after the last row kept. The stream's end
    consumes them, so the run ends at its bound (`TrainState.consumed`). `consumed` is what the batches
    before it consumed (`Interval.consumed`), and the batch's count (`ConsumedCount`) what it consumes,
    in a pool the whole pool's (`pooled_read`): every process counts them, and the batch's first row
    carries them (`credited`), so the step's sum over the pool's rows takes them once."""
    held = counted(batch)
    remainder = bound - consumed - held.examples
    assert remainder >= 0, f"the run's batches consumed {consumed + held.examples} of its {bound} examples"
    if not remainder:
        return batch
    return {**batch, EXAMPLES_CONSUMED: credited(batch[EXAMPLES_CONSUMED], remainder),
            CONSUMED_COUNT: ConsumedCount(held.examples + remainder)}


def credited(column: jax.Array | np.ndarray, examples: int) -> jax.Array | np.ndarray:
    """A batch's `column` of consumed examples with `examples` more at its first row: a placed column shard
    by shard where it is, the shard that holds the first row taking them and the others as they were, so
    every process of a pool does the same and no transfer waits on another process; a column on the host
    is one process's whole batch."""
    if not isinstance(column, jax.Array):
        owed = np.zeros(np.shape(column), np.int32)
        owed[0] = examples
        return column + owed

    def credit(shard: jax.Shard) -> jax.Array:
        if shard.index and (shard.index[0].start or 0):
            return shard.data
        owed = np.zeros(shard.data.shape, np.int32)
        owed[0] = examples
        return shard.data + jax.device_put(owed, shard.device)

    return jax.make_array_from_single_device_arrays(column.shape, column.sharding,
                                                    [credit(shard) for shard in column.addressable_shards])


def refuse_unclocked(state: TrainState, resume: int, examples: int | None) -> None:
    """Refuse to resume a run on the examples clock (`examples`) from a checkpoint of a run on another
    clock, which kept no examples consumed (`TrainState.consumed`) for its schedule to read on from."""
    if examples is not None and state.consumed is None:
        raise ValueError(
            f"step {resume}'s checkpoint is of a run counted in steps or tokens, which kept no examples "
            "consumed, and this run reads its schedule off them: resume it on the clock it was written "
            "on, or start a fresh run from its weights")


def afresh(initial: TrainState, made: optax.OptState, examples: int | None) -> TrainState:
    """A state supplied to a fit, which reads its data afresh: its clock of examples at none consumed
    where the run is on that clock (`examples`), none on another, and its schedules in the form the run's
    own optimizer makes (`made`, its init's shapes; `dew.optim.reclocked`)."""
    return dataclasses.replace(initial, consumed=None if examples is None else jnp.zeros((), jnp.int32),
                               opt_state=reclocked(initial.opt_state, made))


def validation_read(name: str, suite: EvalSuite, *, read: bool, previewed: bool) -> None:
    """Refuse a validation cadence whose batches nothing would read: no metric, no loss `read` (the
    objective's validation loss, a checkpointer's or a held best's) and no preview a tracker takes
    (`previewed`). Only an objective with no validation loss (Flow-GRPO's) and none of the rest would
    open nothing and report nothing, and that contradiction is refused before the run does any work."""
    if not suite.metrics and not read and not previewed:
        raise ValueError(
            f"{name} is scored every {suite.every} steps and nothing consumes it: "
            "pass metrics to score it, preview=True with a tracker to sample from it, "
            "or leave eval_every unset")


def consumption_counted(batch: Batch, *, clocked: bool) -> Batch:
    """A clocked run's `batch` with the examples it consumes counted on the host (`ConsumedCount`) as it is
    prepared; one a step after the bound rebuilt without its count is refused, as the run would stay at
    its start, and each marks its real rows (`VALID_ROWS`). A run on another clock, one
    bounded in examples whose plan counts its steps ahead, keeps no count: none rides to its step."""
    if not clocked:
        return batch if EXAMPLES_CONSUMED not in batch else {
            name: value for name, value in batch.items() if name != EXAMPLES_CONSUMED}
    if EXAMPLES_CONSUMED not in batch:
        raise ValueError("a step after the examples bound dropped the batch's count of the examples it "
                         f"consumes ({EXAMPLES_CONSUMED!r}), which this run's progress reads: a native Grain "
                         "step that rebuilds its records keeps no count")
    counted_batch = {**batch, CONSUMED_COUNT: ConsumedCount(int(np.sum(batch[EXAMPLES_CONSUMED])))}
    if VALID_ROWS not in batch:
        # A clocked read ends at its bound, in a short batch filled and marked: each of its batches marks
        # its real rows, so every one is the same tree, the run's one program.
        counted_batch[VALID_ROWS] = np.ones(len(np.asarray(batch[EXAMPLES_CONSUMED])), bool)
    return counted_batch


def waited_read(run: FitRun, interval: Interval, *, agree: bool, peek: bool = False) -> Batch | None:
    """The training stream's next batch, its wait counted in `interval`, or None where it has run out
    on every process, which ends the run as its length would. Where `agree`, the read is agreed
    (`agreed_next`): one that fails, or runs out on some processes only, stops every process, as
    does a stream with no batch for a run at step 0. An endless stream's later reads are not agreed,
    as it never runs out. A batch read ahead (`FitRun.ahead`) is this read's. Where `peek`, an agreed
    read of the stream's first batch as its source gives it, on the host, placed at the run's first
    read (`DevicePrefetchIterator.peek`)."""
    if run.ahead is not None:
        (batch,), run.ahead = run.ahead, None
        return batch
    assert run.train is not None
    # The prefetcher's peek, as a stream of one batch that ends where the source holds none.
    stream = iter(run.train.peek, None) if peek else run.train
    with region("input.wait"):
        asked = time.perf_counter()
        if run.pooled:
            batch = pooled_read(run, stream, peek=peek)
        else:
            batch = agreed_next("training read", stream) if agree or peek else run.read()
        interval.waited += time.perf_counter() - asked
    if batch is None and not run.current:
        raise ValueError("the training data holds no batch to train on")
    return batch


def pooled(examples: int | None) -> bool:
    """Whether a run on the examples clock (bounded to `examples`) is a pool's, whose shares end apart
    (`FitRun.pooled`)."""
    return examples is not None and jax.process_count() > 1


def pooled_read(run: FitRun, stream: Iterator[Batch], *, phase: str = "training read",
                peek: bool = False) -> Batch | None:
    """A pooled run's next batch (`FitRun.pooled`): this share's, or one of the pool's tree emptied where
    its stream has run out and another share's reads on (`emptied`), from the run's first read on; None
    once every share's has run out. Each read counts the examples the whole pool's batches consume, each
    share's once, by its first reader (`DataPartition.reader`), whose readers hold the same rows, so every
    process's clock is the pool's; one round of the coordination service a read (`agreed_reads`). A
    `peek` reads the stream's first batch on the host, which counts nothing: the read that takes it does.
    An emptied batch holds as many rows as the others' batches of the step, a ramp's stage's included."""
    partition, empty = run.partition, run.emptied
    assert partition is not None and empty is not None

    def said(batch: Batch) -> list[int]:
        # Its share's rows: a placed batch's are the step's, every share holding as many of them.
        rows = rows_of(batch) if peek else rows_of(batch) // partition.count
        return [counted(batch).examples if partition.reader == 0 else 0, rows]

    read, words = agreed_reads(phase, stream, said)
    live = [integers(word, "a pool's read") for word in words if word is not None]
    if not live:
        return None
    if read is None:
        # Every share's batch of the step holds as many rows (`DataPartition.rows`), a ramp's stage's.
        read = empty(live[0][1], placed=not peek)
    return read if peek else {**read, CONSUMED_COUNT: ConsumedCount(sum(examples for examples, _ in live))}


def counted(batch: Batch) -> ConsumedCount:
    """The examples a clocked run's `batch` consumes (`consumption_counted`)."""
    held = batch[CONSUMED_COUNT]
    assert isinstance(held, ConsumedCount), "a clocked run's batch is counted (`consumption_counted`)"
    return held


@dataclasses.dataclass
class Emptied:
    """A pooled run's batches of the pool's tree with nothing in them (`emptied`), which a share that ran out
    steps with (`pooled_read`): each of a share's first batch (`read`, read afresh once) at as many rows as
    the step's other batches hold (`resized`), prepared (`prepare`) and placed (`place`) as the run's
    prefetcher prepares and places its own, made once for each count of rows and placement. It is asked
    for only once this share's stream has run out, its prefetcher preparing nothing more."""

    read: Callable[[], Batch]
    prepare: Callable[[Batch], Batch]
    place: Callable[[Batch], Batch]
    first: Batch | None = None
    made: dict[tuple[int, bool], Batch] = dataclasses.field(default_factory=dict)

    def __call__(self, rows: int, *, placed: bool) -> Batch:
        if (rows, placed) not in self.made:
            if self.first is None:
                self.first = self.read()
            batch = emptied(self.prepare(resized(self.first, rows)))
            self.made[rows, placed] = self.place(batch) if placed else batch
        return self.made[rows, placed]


def pooled_first(loader: Loader, partition: DataPartition) -> Batch:
    """The first batch of the first of `partition`'s pool's shares whose training stream holds one, read
    afresh and kept past its stream (`Emptied.read`): one share's stream has a batch where another's has
    run out."""
    batch = _pooled_first(loader, partition, kept=True)
    assert batch is not None, "a share that ran out steps on only while another share's stream reads"
    return batch


def resized(batch: Batch, rows: int) -> Batch:
    """`batch` holding `rows` rows, its own repeated or cut to that many: each field with a row a leaf of
    its first dimension, the rest as they are."""
    held = rows_of(batch)

    def cut(leaf):
        if np.ndim(leaf) and np.shape(leaf)[0] == held:
            return np.take(np.asarray(leaf), np.arange(rows) % held, axis=0)
        return leaf

    return jax.tree.map(cut, batch)


EMPTIED = (VALID_ROWS, EXAMPLES_CONSUMED, TOKENS)
"""The fields a batch a share steps with once its stream has run out holds zero (`emptied`): no row real,
none consuming an example, none counting a token."""


def emptied(batch: Batch) -> Batch:
    """A share's prepared `batch` of the pool's tree on the host, with no row real, none consuming an
    example and none counting a token (`EMPTIED`): what the share steps with once its stream has run out
    and another share's reads on, so it trains nothing and its pool's programs and collectives stay one.
    Its rows' other fields stand as they were, masked."""
    return {**batch, **{name: jax.tree.map(np.zeros_like, batch[name]) for name in EMPTIED if name in batch},
            CONSUMED_COUNT: ConsumedCount(0)}


def closes_window(batch: Batch, examples: int | None, *, last: bool = False) -> Batch:
    """`batch` as a run on the examples clock (bounded to `examples`) hands its step: marked whether it is
    the run's `last`, closing its update window (`CLOSES_WINDOW`), every batch of such a run marked, so
    every step it compiles is one program, fit's, one compiled ahead (`dew.training.ahead.step_batch`)
    and a cooldown's; a run on another clock's batch as it is."""
    return batch if examples is None else {**batch, CLOSES_WINDOW: np.bool_(last)}


def fit_policies[LossT, EffectsT](objective: Objective[LossT, EffectsT], best, stop, scored,
                                 checkpoint_every, *, steps: int | None, tokens: int | None,
                                 accumulation: int, checkpointed: bool, examples: int | None = None):
    """A fit's best selection and its stop policy over `objective`: a `Plateau` with its metric bound
    to the objective, or the step, tokens and examples a `Pause` of a run of `steps` and `tokens`, or
    bounded to `examples` no pass counts steps ahead of, updates of `accumulation`, ends it at
    (`Pause.at`), which needs the run `checkpointed` to resume from."""
    if isinstance(checkpoint_every, datetime.timedelta) and checkpoint_every.total_seconds() <= 0:
        raise ValueError("checkpoint_every duration must be positive")
    if isinstance(stop, Pause) and not checkpointed:
        raise ValueError("a Pause ends a run its checkpoint resumes, and this trainer has no "
                         "checkpointer; pass Checkpoints(directory)")
    pause = stop.at(steps, tokens, accumulation, examples) if isinstance(stop, Pause) else (None, None, None)
    plateau = stop if isinstance(stop, Plateau) else None
    raw = (
        ()
        if best is None
        else tuple(best)
        if isinstance(best, Sequence) and not isinstance(best, str)
        else (best,)
    )

    def owned(choice):
        if isinstance(choice, types.MethodType):
            if choice.__name__ == 'loss' and choice.__self__ is objective:
                return objective.scalars.loss
            raise TypeError("a training selector must be this objective's loss or declared scalar")
        if isinstance(choice, TrainingScalar) and choice.owner is not objective:
            raise ValueError("training scalar belongs to a different objective")
        if isinstance(choice, Best):
            source = choice._source
            if source is None and (choice.metric.startswith('train/') or choice.split == 'train'):
                source = objective.scalars[choice.metric.removeprefix('train/')]
            return dataclasses.replace(choice, _source=owned(source) if source is not None else None)
        return choice

    selection = tuple(
        (chosen if isinstance(chosen, Best) else Best(chosen)).resolved(scored)
        for chosen in (owned(Best(choice) if isinstance(choice, str) else choice) for choice in raw)
    )
    labels = [(choice.split, choice.metric) for choice in selection
              if isinstance(choice._source, (Metric, TrainingScalar))]
    if len(labels) != len(set(labels)):
        raise ValueError("select each metric once; one Best.top already keeps all its winners")
    if plateau is not None:
        # A name binds as a Best of it does, to the metric fit scores under it, its split and direction.
        metric = plateau.metric if isinstance(plateau.metric, str) else owned(plateau.metric)
        if not isinstance(metric, (str, Metric, TrainingScalar, Summary)):
            raise TypeError(
                "Plateau selects a declared metric object or summary, not a Best policy or score function"
            )
        chosen = owned(Best(metric, mode=plateau.mode, split=plateau.split))
        stopping = (chosen if isinstance(chosen, Best) else Best(chosen, mode=plateau.mode,
                                                                 split=plateau.split)).resolved(scored)
        plateau = dataclasses.replace(plateau, metric=stopping._source, mode=stopping.mode,
                                      split=stopping.split)
    return selection, plateau, pause


def first_batch(loader: Loader, mesh: Mesh, position: bytes | None = None) -> Batch:
    """The first batch `loader`'s training stream gives this process's share of `mesh`, from the saved
    `position` a resume continues at where given, the stream then closed: what an objective that names
    its input alone reads the shape off (`Trainer._declare_input`), once, before the run opens its stream
    where it starts, and the batch a step compiled ahead is compiled over (`AheadTarget.compiled`)."""
    partition = DataPartition.of(mesh)
    batch = _first_read(loader, partition, position)
    # A share a pool's bound in examples deals none of its first examples reads no batch of its own, and
    # every share's batch is of the pool's tree (`pooled_read`).
    if batch is None and partition.count > 1:
        batch = _pooled_first(loader, partition)
    if batch is None:
        raise ValueError("the training data holds no batch to read the model's input shape off")
    return batch


def _pooled_first(loader: Loader, partition: DataPartition, *, kept: bool = False) -> Batch | None:
    """The first batch of the first of `partition`'s pool's shares whose training stream holds one, read
    afresh (and `kept`, `_first_read`); None where none does."""
    shares = (dataclasses.replace(partition, index=index) for index in range(partition.count))
    reads = (_first_read(loader, share, kept=kept) for share in shares)
    return next((read for read in reads if read is not None), None)


def _first_read(loader: Loader, partition: DataPartition, position: bytes | None = None, *,
                kept: bool = False) -> Batch | None:
    """The first batch `loader`'s training stream gives `partition`'s share, from `position` where given,
    the stream then closed; None where it holds none. A batch `kept` holds copies of its arrays, which
    outlive the stream, as a batch a stream yields does only until its next read (`Loader`)."""
    stream = loader.train(partition)
    try:
        if position is not None:
            if not isinstance(stream, Checkpointable):
                raise TypeError("the training stream keeps no position a resume reads its batch from")
            restore_position(stream, position)
        batch = next(iter(stream), None)
        if kept and batch is not None:
            batch = jax.tree.map(lambda leaf: np.array(leaf) if isinstance(leaf, np.ndarray) else leaf, batch)
        return batch
    finally:
        if isinstance(stream, Closeable):
            stream.close()


type HandedSplits = (Dataset | pygrain.MapDataset | pygrain.IterDataset
                     | Mapping[str, Dataset | Handed | Reader | EvalSuite])
"""`fit`'s validation: a chain or a Grain dataset, the split `val`, or splits by name, each a chain, a
Grain dataset, NumPy arrays, a reader or an evaluation suite."""


def handed_splits(
        validation: HandedSplits | None) -> Dataset | Mapping[str, Dataset | Reader | EvalSuite] | None:
    """`fit`'s validation with what was handed as it is read as handed (`dew.data.compile.native`): a
    Grain dataset as the split `val`, and a named split's Grain dataset or NumPy arrays as that split.
    Arrays are a split by name alone, as a mapping of splits is one by its names."""
    if isinstance(validation, pygrain.MapDataset | pygrain.IterDataset):
        return {"val": native(validation, training=False)}
    if validation is None or isinstance(validation, Dataset):
        return validation
    return {name: _handed_split(entry) for name, entry in validation.items()}


def _handed_split(entry: Dataset | Handed | Reader | EvalSuite) -> Dataset | Reader | EvalSuite:
    """One split of `handed_splits`: read as handed where it is a Grain dataset or arrays."""
    if isinstance(entry, Mapping | pygrain.MapDataset | pygrain.IterDataset):
        return native(entry, training=False)
    return entry


def check_inputs(trainer, device: Batch) -> None:
    """Refuse a training batch's device fields (`split_fields`) the objective's declared inputs do not
    take, as every process agrees, before its first step compiles: the batch its device steps make,
    which the objective reads (`Trainer._model_batch`)."""
    inputs = trainer.objective.inputs
    if inputs is None:
        return
    with jax.set_mesh(trainer.device_mesh):
        checked = trainer._model_shapes(device) if trainer._device else device
    agreed("training input declaration", functools.partial(inputs.check, checked))


def unrecorded_parts(objective: str, dataset: Dataset | Loader | Handed, *, optimized: bool,
                     seeded: bool, mesh: MeshSpec | Mesh | None, given: Mapping[str, object]) -> list[str]:
    """Why a run's record cannot rebuild a trainer's run on `dataset`, each a part a
    `dew.config.RunConfig` states otherwise, read from the trainer's own fields
    (`Checkpoints.python_run`): an objective, of the class `objective` names, built in Python, an optax
    optimizer (not `optimized` from an OptimConfig), a JAX key (no integer seed), a native mesh, a
    compiled Loader, and each part `given` to the trainer that a run's record does not state. A run
    built as a RunConfig writes its own record, which rebuilds it, before it trains."""
    reasons = [f"the objective, {objective}, was built in Python; a run's record names it "
               f"by an ObjectiveConfig over a ModelConfig"]
    if not optimized:
        reasons.append("the optimizer is an optax transformation; a run's record states an OptimConfig")
    if not seeded:
        reasons.append("the key is a JAX key; a run's record states an integer seed")
    if isinstance(mesh, Mesh):
        reasons.append("the mesh is a jax.sharding.Mesh; a run's record states a MeshSpec")
    if isinstance(dataset, Loader):
        reasons.append("the data is a compiled Loader; a run's record states its chain")
    elif not isinstance(dataset, Dataset):
        reasons.append("the data was handed as it is, a Grain dataset or arrays; a run's record states "
                       "a chain")
    reasons += [f"{name} is given to the trainer, which a run's record does not state"
                for name, part in given.items() if part is not None]
    return reasons


def stage_reached(plan: FitPlan, run: FitRun) -> tuple[PhasedStream, bytes, int] | None:
    """Where the run's commits reached the tokens of the schedule's stage it stands in
    (`dew.data.pipeline.schedule`), on every process alike, as the committed tokens are the pool's: its
    stream opened again where the step that reached them left it, the stages that step's commit passed
    ended there, reading nothing, and the stage it now reads; what the prefetch worker read ahead of
    the end is dropped untrained. The last stage's tokens, where they end the data, are the run's token
    budget (`dew.training.budget.run_tokens`). None before a stage's end, which the run's clock reads
    off the devices where its steps may have reached it (`FitPlan.committed`), as a token budget's."""
    ends = plan.dataset.token_ends
    state = None if run.train is None else run.train.source_state
    if ends is None or state is None:
        return None
    stage = len(position.read(state).completed)
    if stage >= len(ends) - 1 or (committed := run.clock.read) < (end := ends[stage] or 0):
        return None
    assert run.train is not None and run.partition is not None
    run.train.close()
    run.train = None
    run.source = source = plan.dataset.train(run.partition)
    if not isinstance(source, PhasedStream):
        raise TypeError("a schedule reads its stages as phases")
    while stage < len(ends) - 1 and committed >= end:
        state, stage = source.ending(state), stage + 1
        end = ends[stage] or 0
    return source, state, stage


@dataclasses.dataclass(frozen=True)
class Start:
    """What `Trainer.place` reads before it places the state (`Trainer._started`): the objective's
    held tree and the key, the state's shapes and which frozen leaves its init passes through
    (`traced_start`), the checkpoint step a run resumes from, and the state's placement."""

    initializer: Initializer
    key: jax.Array
    abstract: TrainState
    through: dict[int, int]
    resume: int | None
    shardings: Placement[TrainState]
