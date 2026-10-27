"""One `Trainer.fit` call's records: what it was asked for (`FitPlan`, its validation splits resolved by
`validation_suites`), what its loop carries from step to step (`FitRun`), and how the loop reads its
training stream (`waited_read`)."""

from __future__ import annotations

import dataclasses
import datetime
import math
import time
from collections.abc import Iterator, Mapping, Sequence

import jax
from flax.training.early_stopping import EarlyStopping
from jax.sharding import Mesh

from dew import position
from dew._coordination import agreed_next
from dew._records import JSON
from dew.batch import Batch
from dew.checkpoints import Checkpoints
from dew.data.dataset import Closeable, DataPartition, Loader, PhasedStream, Reader
from dew.data.pipeline import Dataset
from dew.distributed.batches import DevicePrefetchIterator, Signature
from dew.distributed.placement import MeshSpec, Placement
from dew.objectives.base import Initializer, Metric, TrainingScalar
from dew.optim import OptimConfig
from dew.state import TrainState
from dew.telemetry.profile import region
from dew.training.budget import Plateau, every_step, recorded
from dew.training.cadence import Cadence, Tokens
from dew.training.evaluation import EvalSuite, Evaluation, Summary
from dew.training.interval import Interval, Taken, TokenClock
from dew.training.observe import Observe
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
    log_every: int | Tokens
    eval_every: Cadence
    tokens: int | None
    checkpoint_every: Cadence | datetime.timedelta
    checkpoint_final: bool
    local_every: Cadence
    metrics: Sequence[Metric]
    preview: bool
    best: tuple[Best, ...] = ()
    stop: Plateau | None = None
    pause: tuple[int | None, int | None] = (None, None)
    """The step and the real tokens a `Pause` ends the run at (`Pause.at`)."""
    suites: Mapping[str, EvalSuite] = dataclasses.field(default_factory=dict)
    held_best: bool = False  # a restore_best without Checkpoints (`HeldBest`)
    """Every validation split by name, its metrics and cadence resolved."""
    summaries: tuple[Summary, ...] = ()
    """The scalars each step whose evaluations score every split they weigh forms (`Summary`)."""
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

    def paused(self, step: int, real: int) -> bool:
        """Whether the run has reached where its `Pause` ends it, after `real` tokens."""
        at, tokens = self.pause
        return (at is not None and step >= at) or (tokens is not None and real >= tokens)

    def ended(self, step: int, real: int) -> bool:
        """Whether this fit has nothing left to train: its budget reached, or its pause."""
        return self.finished(step, real) or self.paused(step, real)

    def horizon(self, real: int) -> int | None:
        """The least count of committed tokens past `real` at which something this plan reads of the
        clock changes: its budget, its pause, the next multiple of each cadence of tokens, or the end
        of a schedule's stage. None where nothing it reads does."""
        cadences = (self.log_every, self.eval_every, self.checkpoint_every, self.local_every,
                    *(suite.every for suite in self.suites.values()))
        nexts = [(real // every.every + 1) * every.every for every in cadences if isinstance(every, Tokens)]
        ends = (self.tokens, self.pause[1], *(self.dataset.token_ends or ()), *nexts)
        return min((end for end in ends if end is not None and end > real), default=None)

    def committed(self, interval: Interval, clock: TokenClock) -> int:
        """The run's committed tokens after its latest step, as this plan reads them: read off the
        devices where the clock's bound reaches the next count the plan reads (`horizon`), and
        otherwise the count of its last read, which falls on no budget, cadence or stage the latest
        step's count could not, so the loop does not wait on that step."""
        horizon, bound = self.horizon(clock.read), clock.bound
        if horizon is not None and (bound is None or bound >= horizon):
            clock.reread(interval.real_tokens)
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
            if isinstance(every, Tokens):
                return self.tokens is None or every.every < self.tokens
            return bool(every) and (self.steps is None or not isinstance(every, int) or every < self.steps)

        cadences = (self.eval_every, *(suite.every for suite in self.suites.values()))
        return any(map(before_end, cadences)) or bool(
            (self.validation or self.best or self.stop) and before_end(self.checkpoint_every))

    @staticmethod
    def falls(every: Cadence | str | datetime.timedelta, step: int, real: int, before: int) -> bool:
        """Whether a cadence of steps (`int`) or of committed tokens (`Tokens`, the step having brought them
        from `before` to `real`) falls on `step`; a duration or none never does here."""
        if isinstance(every, Tokens):
            return every.due(real, before)
        return isinstance(every, int) and bool(every) and step % every == 0

    @staticmethod
    def cadences(loader: Loader, validation: Mapping[str, Reader | EvalSuite] | None,
                 metrics: Sequence[Metric], steps: int | None, tokens: int | None, accumulation: int,
                 written: tuple[int | str, int | str | None, int | str | datetime.timedelta | None],
                 checkpoints: Checkpoints | None = None) -> tuple[
                     int | Tokens, Cadence, Cadence | datetime.timedelta, Cadence, dict[str, EvalSuite]]:
        """`fit`'s log, validation and checkpoint cadences, its `checkpoints`' local one
        (`Checkpoints.local_every`), and its suites' (`validation_suites`), each
        written as steps, a duration or a share of the run such as "5%", resolved once against its planned
        length: the whole updates nearest that share of its `steps` microsteps, `accumulation` an update,
        at least one; of a budget of `tokens` alone, each time the committed tokens pass a multiple of that
        share of them (`dew.training.budget.every_step`). A schedule's warmup may be a share too
        (`dew.optim.Cosine.warmup_steps`), and the start records what each resolved to
        (`recorded`)."""
        def cadence(value, name: str) -> Cadence:
            return every_step(value, steps, tokens, accumulation, name=name)

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

    def recorded(self, config: OptimConfig | None, accumulation: int) -> dict[str, JSON]:
        """The cadences this plan resolved and the warmups of `config`'s schedules over its updates,
        `accumulation` microsteps each, as the run's start records them (`dew.telemetry.FitStarted`)."""
        warmups = {} if config is None else config.warmups(
            None if self.steps is None else self.steps // accumulation)
        suites = {f"{name}.every": recorded(suite.every) for name, suite in self.suites.items()
                  if suite.every != self.eval_every}
        local = {} if self.local_every is None else {"local_every": recorded(self.local_every)}
        return {"log_every": recorded(self.log_every), "eval_every": recorded(self.eval_every),
                "checkpoint_every": recorded(self.checkpoint_every), **local, **suites,
                **{f"warmup_steps.{name}": value for name, value in warmups.items()}}

    def plateaued(self, run: FitRun, scores: Mapping[str, float]) -> bool:
        """Whether the run's `Plateau` stops it at its current step on these `scores`, its count of
        evaluations without improvement kept in the checkpoint's control (`FitRun.stop_control`)."""
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

    def due(self, step: int, real: int = 0, before: int = 0) -> tuple[str, ...]:
        """The suites whose own cadence falls on `step`, which brought the committed tokens from
        `before` to `real` (`falls`)."""
        return tuple(name for name, suite in self.suites.items()
                     if self.falls(suite.every, step, real, before))


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
    save had to know whether the run ends there (`Trainer._ends_here`)."""
    best: HeldBest = dataclasses.field(default_factory=HeldBest)
    """The log taken at the latest logging step and not yet read."""
    stepped: tuple[jax.Array, dict[str, jax.Array], jax.Array] | None = None
    """The latest step's loss, objective metrics and acceptance, which the log of the interval it
    ends takes."""
    reads: dict[Signature, tuple[bool, ...]] = dataclasses.field(default_factory=dict)
    """Which leaves of a batch of each signature its compiled step reads
    (`step_reads`), which the prefetcher alone places."""

    def read(self) -> Batch:
        """The training stream's next batch, read through this run each time:
        a caller's alias would keep the closed stream reachable from a failed
        run's traceback."""
        assert self.train is not None
        return next(self.train)


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
        batch = agreed_next("training read", stream) if agree or peek else run.read()
        interval.waited += time.perf_counter() - asked
    if batch is None and not run.current:
        raise ValueError("the training data holds no batch to train on")
    return batch


def first_batch(loader: Loader, mesh: Mesh) -> Batch:
    """The first batch `loader`'s training stream gives this process's share of `mesh`, the stream then
    closed: what an objective that names its input alone reads the shape off (`Objective.shaped`), once,
    before the run opens its stream where it starts."""
    stream = loader.train(DataPartition.of(mesh))
    try:
        batch = next(iter(stream), None)
    finally:
        if isinstance(stream, Closeable):
            stream.close()
    if batch is None:
        raise ValueError("the training data holds no batch to read the model's input shape off")
    return batch


def unrecorded_parts(objective: str, dataset: Dataset | Loader, *, optimized: bool, seeded: bool,
                     mesh: MeshSpec | Mesh | None, given: Mapping[str, object]) -> list[str]:
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
