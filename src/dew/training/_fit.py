"""One `Trainer.fit` call's records: what it was asked for (`FitPlan`), what its loop carries from step
to step (`FitRun`), and how the loop reads its training stream (`waited_read`)."""

from __future__ import annotations

import dataclasses
import datetime
import time
from collections.abc import Iterator, Mapping, Sequence

import jax

from dew.coordination import agreed_next
from dew.data.dataset import Loader
from dew.objectives.base import Batch, Metric
from dew.telemetry.profile import region
from dew.training.budget import Plateau
from dew.training.distributed import DevicePrefetchIterator, Signature
from dew.training.evaluation import EvalSuite, Evaluation, Summary
from dew.training.interval import Interval, Taken
from dew.training.observe import Observe
from dew.training.runtime import PreemptionNotice
from dew.training.selection import Best, HeldBest


def split_metrics(suites: Mapping[str, EvalSuite]) -> dict[str, tuple[Metric, ...]]:
    """Each split's own metrics, by its name: two suites may each hold their own instance of one
    metric."""
    return {name: tuple(suite.metrics) for name, suite in suites.items()}


def declarations(suites: Mapping[str, EvalSuite], metrics: Sequence[Metric],
                 summaries: Sequence[Summary]) -> dict[str, tuple[Metric | Summary, ...]]:
    """Every score a fit declares, by split: each suite's metrics (fit's own `metrics` as `val` where it
    validates no suite) and the summaries under theirs (`Summary.SPLIT`). Its selections resolve against
    this and its rankings read their aggregates' scores off it, one declaration for both."""
    declared: dict[str, tuple[Metric | Summary, ...]] = {**(split_metrics(suites) or {'val': tuple(metrics)})}
    if summaries:
        declared[Summary.SPLIT] = tuple(summaries)
    return declared


@dataclasses.dataclass(frozen=True)
class FitPlan:
    """What one `Trainer.fit` call was asked for."""

    dataset: Loader
    steps: int | None
    log_every: int
    eval_every: int | None
    tokens: int | None
    checkpoint_every: int | datetime.timedelta | None
    checkpoint_final: bool
    local_every: int | None
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
    """The scalars each evaluation that scores every split they weigh forms (`Summary`)."""
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

    def real(self, interval: Interval) -> int:
        """The run's real tokens where a token budget reads them, else 0: the count is read from the
        devices, which a run without one need not wait on."""
        return interval.real_tokens if self.tokens is not None else 0

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
        def before_end(every: int | datetime.timedelta | None) -> bool:
            return bool(every) and (self.steps is None or not isinstance(every, int) or every < self.steps)

        cadences = (self.eval_every, *(suite.every for suite in self.suites.values()))
        return any(map(before_end, cadences)) or bool(
            (self.validation or self.best or self.stop) and before_end(self.checkpoint_every))

    def due(self, step: int) -> tuple[str, ...]:
        """The suites whose own cadence falls on `step`."""
        return tuple(name for name, suite in self.suites.items() if suite.every and step % suite.every == 0)


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
    stop_control: dict = dataclasses.field(default_factory=dict)
    last_local_saved: int | None = None
    stopped: bool = False
    training: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    source: Iterator[Batch] | None = None
    train: DevicePrefetchIterator | None = None
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


def waited_read(run: FitRun, interval: Interval, *, agree: bool) -> Batch | None:
    """The training stream's next batch, its wait counted in `interval`, or None where it has run out
    on every process, which ends the run as its length would. Where `agree`, the read is agreed
    (`agreed_next`): one that fails, or runs out on some processes only, stops every process, as
    does a stream with no batch for a run at step 0. An endless stream's later reads are not agreed,
    as it never runs out. A batch read ahead (`FitRun.ahead`) is this read's."""
    if run.ahead is not None:
        (batch,), run.ahead = run.ahead, None
        return batch
    assert run.train is not None
    with region("input.wait"):
        asked = time.perf_counter()
        batch = agreed_next("training read", run.train) if agree else run.read()
        interval.waited += time.perf_counter() - asked
    if batch is None and not run.current:
        raise ValueError("the training data holds no batch to train on")
    return batch
