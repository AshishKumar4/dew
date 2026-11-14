"""Coordinated evaluation of trained variables, independent of optimization."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import itertools
import operator
import os
import sys
import time
import types
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import jax
import numpy as np
from flax import linen as nn
from jax.sharding import Mesh
from typing_extensions import TypeIs

from dew._coordination import (
    agree_process_phase,
    agreed,
    agreed_same,
    broadcast_from_process_zero,
    collective_host,
    from_every_process,
)
from dew.artifacts import Artifact, Artifacts, FixedRows
from dew.batch import VALID_ROWS, Batch, Variables
from dew.data.dataset import Closeable, DataPartition, Reader, row_field, rows_of, take_rows
from dew.data.pipeline import Dataset
from dew.distributed.batches import (
    batch_signature,
    placed_abstract,
    row_origins,
    shard_batch,
    shared_host_fields,
    split_fields,
    with_host,
)
from dew.distributed.placement import Layout, MeshSpec, Placement
from dew.nn.kernels.options import program, shapes_of
from dew.nn.sharding import Link, measured_links, pipeline_microbatches
from dew.objectives.base import Effects, Loss, Metric, Objective, Ratio, Shown, Step, TrainingScalar, Weighted
from dew.objectives.passes import PassReports
from dew.telemetry.profile import region
from dew.training.cadence import Counted
from dew.training.staging import fetched_state
from dew.training.tracker import Tracker, write_artifact
from dew.training.transaction import with_ema

if TYPE_CHECKING:
    from dew.inputs.spec import InputSpec
    from dew.state import TrainState
    from dew.training._fit import FitPlan
    from dew.training.display import TrainingDisplay


@dataclass(frozen=True)
class Flown:
    """The scores a split's episodes earned in one evaluation, by name: the artifact every
    `EpisodeScore` of the split reads."""

    scores: Mapping[str, float]


@dataclass(frozen=True)
class EpisodeScore:
    """A score a split's episodes report (`Episodes`), declared as any metric is, so a selection, a stop
    or a summary reads it as a data split's (`Best("flight/return")`): its `name` and how it shows
    (`shown`, which way is better). A pass holds one artifact, the episodes' `Flown` scores, which it
    reads its own entry of."""

    name: str
    shown: Shown

    @property
    def reads(self) -> type:
        return Flown

    def __call__(self, artifact: Artifact, batch: Batch, /) -> float:
        if not isinstance(artifact, Flown):
            raise TypeError(f"{self.name} reads a split's episodes (Flown), not a {type(artifact).__name__}")
        return float(artifact.scores[self.name])

    def merge(self, accumulated: float, contribution: float, /) -> float:
        return contribution

    def finalize(self, accumulated: float, /) -> float:
        return accumulated


@dataclass(frozen=True)
class Episodes:
    """A split scored by flying a policy, not by reading records: `score(variables, key)` returns the
    split's scores by name (`dew.rl.evaluation.episodes`), each declared with how it shows (`scores`,
    the split's metrics, `EpisodeScore`). It goes where a reader goes (`EvalSuite(Episodes(...))`), and
    `Evaluation.run` scores it on the averaged weights where the run keeps them, on every process and
    agreed (a scorer that fails on one, or reports other scores than `scores` declares, fails the
    evaluation on all), process 0's scores the pool's, each of the metrics asked for (all of `metrics`
    unless a suite names fewer) named `<split>/<name>` as a data split's; a warming pass flies to compile
    and keeps none. Its key is
    the evaluation's event key (`_event`: the run's key folded with 'EVAL' and the step) folded with
    'EPIS'. It reads no batches, so a run's data metrics and loss are not its."""

    score: Callable[[Variables, jax.Array], Mapping[str, float]]
    scores: Mapping[str, Shown]

    @functools.cached_property
    def metrics(self) -> tuple[Metric, ...]:
        """The split's metrics: one `EpisodeScore` a declared score, made once, so a selection given one
        (`Best(split.metrics[0])`) reads the very metric the run declares."""
        return tuple(EpisodeScore(name, shown) for name, shown in self.scores.items())


type Split = Reader | Episodes
"""What a validation split is scored from: the batches a reader opens, or the episodes a policy flies."""


def episodic(reader: Reader | Dataset | Episodes | None) -> TypeIs[Episodes]:
    """Whether a split is scored by its episodes (`Episodes`) rather than by the batches a reader
    opens: the one test every reader of a suite's data makes."""
    return isinstance(reader, Episodes)


@dataclass(frozen=True)
class EvalSuite:
    """A validation split `fit` scores with its own metrics on its own cadence.

    Empty `metrics` scores fit's, and `every` None scores it on fit's
    `eval_every`. Every suite is also scored at a checkpoint that ranks the
    run's steps and at the end, so no ranking reads an earlier score. A split
    scored by its episodes holds an `Episodes` as its `data`.
    """

    data: Split
    metrics: Sequence[Metric] = ()
    every: int | str | Counted | None = None


@dataclass(frozen=True, eq=False)
class SplitLoss:
    """A validation split's loss, which its evaluation records beside the split's metrics
    (`<split>/loss`) wherever the objective defines one (`Objective.defines_validation_loss`): a score
    the fit declares for each split it validates whose suite holds no metric named `loss` of its own, so
    `Best` and `Plateau` rank it by its name, lower better, as they rank a metric. Each split's is its
    own object, as each suite's instance of one metric is."""

    name = "loss"
    shown = Shown(better="lower")


type Declared = Metric | Summary | SplitLoss
"""A score a fit declares (`dew.training._fit.declarations`), which its selections resolve against."""


@dataclass(frozen=True)
class Summary:
    """One scalar a run forms from each evaluation's split scores by a declared weighting, a board's
    weighted index: `sum(weight * score) / sum(weight)` over `weights`, each score named as an evaluation
    records it (`val/accuracy`, `test/loss`) and each weight positive. It is formed once a step, where the
    step's scores are joined (`Trainer.fit`), and only at a step that scored every split it weighs, so a
    step a rarer suite skips forms none rather than mix in its stale scores. It is logged as
    `evaluation/summary/<name>`, which `Best` and `Plateau` rank by, `better` the way it improves."""

    name: str
    weights: Mapping[str, float]
    better: Literal["higher", "lower"]

    SPLIT: ClassVar[str] = "evaluation/summary"
    """The namespace a summary's score is recorded under, in a score's split's place:
    `evaluation/summary/<name>`. It holds a '/', which no split's name or metric's name can
    (`Evaluation.run`), so no split's score is named in it."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", dict(self.weights))
        if not self.name or "/" in self.name:
            raise ValueError(f"a summary is named without '/', and {self.name!r} is not")
        if not self.weights:
            raise ValueError(f"summary {self.name!r} weighs no score")
        for score, weight in self.weights.items():
            split, _, metric = score.rpartition("/")
            if not split or not metric:
                raise ValueError(f"summary {self.name!r} weighs {score!r}, which names no split/metric")
            if not (weight > 0 and np.isfinite(weight)):
                raise ValueError(f"summary {self.name!r} weighs {score!r} by {weight}, which is not positive")
        if self.better not in ("higher", "lower"):
            raise ValueError(f"summary {self.name!r} improves 'higher' or 'lower', not {self.better!r}")

    @property
    def shown(self) -> Shown:
        """Its direction, as a metric declares its own (`Best` reads it)."""
        return Shown(better=self.better)

    def formed(self, scores: Mapping[str, float]) -> dict[str, float]:
        """`{evaluation/summary/<name>: value}` from one step's `scores`, where they hold every score it
        weighs, as a step that scored every split it weighs does (`check`), and nothing where not."""
        if not all(score in scores for score in self.weights):
            return {}
        total = sum(self.weights.values())
        value = sum(weight * float(scores[score]) for score, weight in self.weights.items()) / total
        return {f"{self.SPLIT}/{self.name}": value}

    def check(self, suites: Mapping[str, EvalSuite], loss: bool) -> None:
        """Refuse a score no evaluation records: one of a split `suites` does not hold, or a metric its
        suite does not score, its loss where the objective defines none (`loss`)."""
        for score in self.weights:
            split, _, metric = score.rpartition("/")
            if split not in suites:
                raise ValueError(f"summary {self.name!r} weighs {score!r}, and the run validates no split "
                                 f"{split!r} ({sorted(suites)})")
            declared = {scored.name for scored in suites[split].metrics}
            if metric not in declared and not (metric == "loss" and loss):
                raise ValueError(f"summary {self.name!r} weighs {score!r}, which split {split!r} does not "
                                 f"score ({sorted(declared | ({'loss'} if loss else set()))})")


EXAMPLES_SCALAR = "evaluation/examples_consumed"
"""The scalar an evaluation of a run on the examples clock records the examples it had consumed under,
the coordinate the run's own clock places it at (`Evaluation.examples`)."""

COOLDOWN = "cooldown"
"""What a cooldown branch's evaluation is named under (`Evaluation.branched`): each split it scores,
`cooldown/<split>`, which no split's own name can be (a split's name holds no `/`), and the examples the
branch had consumed, `cooldown/examples_consumed`, apart from the run's own at the step it branched at
(`EXAMPLES_SCALAR`)."""


@dataclass(frozen=True)
class Evaluation:
    """The result of one evaluation, with its previews kept on process 0.

    Every process holds the same `scores`, row counts and RNG identity
    (`event_key`). Each score's name starts with the split, as in `val/loss`.
    `coordinated_batches` is the number of batches every process scored, and
    `records` the number of real rows in them, which covers every record of the
    split once. `elapsed_seconds` is process 0's wall time, including closing the
    batch iterator. `examples` is the source examples the run had consumed when it
    evaluated, on the examples clock, its own coordinate there; None on another. The
    metric accumulators and the validation batches are not kept.
    """

    step: int
    split: str
    scores: dict[str, float]
    coordinated_batches: int
    records: int
    event_key: tuple[int, ...]
    elapsed_seconds: float
    previews: tuple[Artifact, ...]
    examples: int | None = None
    real_token_share: float | None = None
    """Of the tokens the split's batches computed (each row's `Objective.row_tokens`, padding and a
    drained rank's repeats included), the share the loss scored (`Objective.scored_tokens` of its real
    rows); None where the objective counts no tokens or its rows declare none."""

    @property
    def scalars(self) -> dict[str, float]:
        """Return the metric values, plus the split's counts if a batch was scored
        (`evaluation/counts/<split>/records`, each split's its own at a step that scores several, apart
        from every summary's, `Summary.SPLIT`, and its `real_token_share` where known), and on the
        examples clock the examples the run had consumed (`EXAMPLES_SCALAR`), a cooldown branch's under its
        own name (`COOLDOWN`)."""
        named = f"{COOLDOWN}/examples_consumed" if self.split.startswith(f"{COOLDOWN}/") else EXAMPLES_SCALAR
        clocked = {} if self.examples is None else {named: float(self.examples)}
        if not self.coordinated_batches:
            return {**self.scores, **clocked}
        share = ({} if self.real_token_share is None else
                 {f"evaluation/counts/{self.split}/real_token_share": self.real_token_share})
        return {**self.scores,
                f"evaluation/counts/{self.split}/coordinated_batches": float(self.coordinated_batches),
                f"evaluation/counts/{self.split}/records": float(self.records), **share, **clocked}

    def _report(self, display: TrainingDisplay, tracker: Tracker | None) -> None:
        """Print this evaluation on rank zero (`display`) and log its previews and scores there to `tracker`,
        every process agreeing it was reported: the trainer's, whose display is its own."""
        def report() -> None:
            if jax.process_index() != 0:
                return
            display.evaluation(self)
            if tracker is not None:
                for artifact in self.previews:
                    tracker.artifact(artifact, self.step)
                scalars = self.scalars
                if scalars:
                    tracker.log(scalars, self.step)

        agreed("evaluation reporting", report)

    def branched(self) -> Evaluation:
        """This evaluation as a cooldown branch's: its split and every score named under `COOLDOWN`."""
        return replace(self, split=f"{COOLDOWN}/{self.split}",
                       scores={f"{COOLDOWN}/{name}": value for name, value in self.scores.items()})

    @classmethod
    def run(cls, objective: Objective[Loss, Effects], variables: Variables,
            batches: Reader | Dataset | Episodes | None, *,
            key: int | jax.Array, metrics: Sequence[Metric] = (),
            step: int | jax.Array = 0, averaged: Variables | None = None,
            preview: bool = False, mesh: Mesh | None = None, split: str = "val",
            schedule_step: int | jax.Array | np.ndarray | None = None, loss: bool = False,
            training: Mapping[str, jax.Array] | None = None, warm: bool = False,
            updates: int | jax.Array | None = None,
            scored_to: str | os.PathLike[str] | None = None) -> Evaluation:
        """Evaluate `variables` on the batches every process can read, without an optimizer or a tracker.

        Every process calls this with the same objective, metrics and numerical
        settings. `batches` is called with this process's share of the split
        (`DataPartition.of(mesh)`) and returns a fresh iterator, which this call
        opens and closes; you can pass `Loader.val` directly, or a chain
        (`Dataset`), which compiles here with the objective's own evaluation
        stages and reads in its own batches (`dew.data.compile.held`). Every batch holds
        `VALID_ROWS` (all True when the reader marked none), and the objective's loss
        and the metrics count only the real rows. A process whose share runs out
        before the others' scores a copy of its last batch in which every row is a
        repeat, so every process scores the same number of batches and the pass
        reads every record of every share. With no `metrics`, no `loss` and no
        preview, no iterator is opened and the objective does no work, and a preview
        alone reads at most the first batch. Only process 0's `preview` flag counts,
        and the preview is made once per call.

        `variables` is the complete Flax variables tree. `averaged`, when given, is
        the complete set of averaged weights the objective sees as `Step.ema`, not an
        optimizer state; to evaluate those weights directly, pass `state.averaged` as
        `variables`. `step` keys the evaluation's RNG and labels the result.
        `schedule_step` is the step the objective's schedules read, and defaults to
        `step`. When training attempts were rejected, pass the number of accepted
        microbatches (`TrainState.microstep`, as `fit` does) so the schedule follows
        only the work training accepted. `updates` is the updates training
        committed (`Step.updates`), and defaults to the schedule's step.
        `warm` scores at most `WARM_BATCHES` batches and finalizes no score, so
        the evaluation holds none: a pass over them runs every program a longer
        pass does, the loss's reduction included, which a run compiles this way
        before its first step (`Trainer.fit`), and no score of a part of the
        split is checked or read.

        `scored_to`, a directory, takes every batch's scoring artifacts, process 0 writing each batch's
        real rows (a decision run's `Decisions`) as `<scored_to>/<split>-<batch>.npz`
        (`dew.training.tracker.write_artifact`), so a pass's metrics pool exactly offline, across
        batches and splits; a warming pass writes none. Process 0's `scored_to` is the pool's: every
        process scores the batches it writes, whatever it was given.

        The scores are broadcast to every process. Previews stay on process 0, and
        there is at most one objective preview, however long the validation split
        is. Reporting the result is up to the caller. In a process pool, the callers
        must agree on whether reporting failed before they enter their next
        collective, and a device failure in flight still requires ending the
        distributed runtime.
        """
        from dew.data.compile import held
        from dew.nn.inputs import request_key
        key = request_key(key)
        if episodic(batches):
            # A policy flies off the mesh, reading a stage-local store as its layers (`off_stages`).
            return cls._episodes(batches, averaged if averaged is not None else variables, key=key, step=step,
                                 schedule_step=schedule_step, updates=updates, split=split, warm=warm,
                                 metrics=metrics or batches.metrics, training=training)
        if isinstance(batches, Dataset):
            batches = held(batches, objective=objective.data())
        started = time.perf_counter()
        root = jax.process_index() == 0
        _agree_configuration(metrics, batches, split, loss=loss, training=training is not None, warm=warm)
        preview_enabled = bool(broadcast_from_process_zero(root and preview))
        event_step, context, event_words, training_scores = _event(
            key, step, schedule_step, averaged, training, updates)
        score_key, preview_key = _folded(context.key, _SCORE), _folded(context.key, _PREVIEW)
        scores: dict[str, float] = {}
        previews: tuple[Artifact, ...] = ()
        scored = records = 0
        share = None
        written = None if scored_to is None or warm else Path(scored_to) / split
        # Root writes what it is asked to keep, and every process scores the batches it writes, so each
        # runs the same collectives: root's ask is the pool's.
        kept = bool(broadcast_from_process_zero(root and written is not None))
        if batches is not None and (metrics or preview_enabled or loss or kept):
            scores, previews, scored, records, share = _score_split(
                objective, variables, batches, context, mesh, metrics=metrics, split=split,
                root=root, preview_enabled=preview_enabled, score_key=score_key,
                preview_key=preview_key, loss=loss, warm=warm, written=written if root else None,
                kept=kept)
        scores.update(training_scores)
        elapsed = time.perf_counter() - started
        scores, elapsed = broadcast_from_process_zero((scores, elapsed))
        return cls(event_step, split, scores, scored, records, event_words, elapsed, previews,
                   real_token_share=share)


    @classmethod
    def _episodes(cls, episodes: Episodes, variables: Variables, *, key: jax.Array, step: int | jax.Array,
                  schedule_step: int | jax.Array | np.ndarray | None, updates: int | jax.Array | None,
                  split: str,
                  warm: bool, metrics: Sequence[Metric], training: Mapping[str, jax.Array] | None
                  ) -> Evaluation:
        """`run`'s evaluation of a split scored by its episodes: its metrics checked and agreed as a data
        split's are (`_agree_configuration`), each one reading the flight's scores (`Flown`), its clocks,
        key and training values as well (`_event`); then, on every process and agreed, the flight, its
        scores held to those the split declares and each metric asked for read off them (`EpisodeScore`)
        and named `<split>/<name>`, process 0's the pool's; a warming pass flies once to compile and keeps
        none."""
        started = time.perf_counter()
        _agree_configuration(metrics, episodes, split, loss=False, training=training is not None, warm=warm)
        unread = [metric.name for metric in metrics if not issubclass(metric.reads, Flown)]
        if unread:
            raise ValueError(f"the {split} split is scored by its episodes, which its metrics read (Flown); "
                             f"{unread} read none")
        event_step, context, event_words, training_scores = _event(key, step, schedule_step, None, training,
                                                                   updates)

        def flown() -> dict[str, float]:
            reported = Flown(dict(episodes.score(variables, _folded(context.key, 0x45504953))))
            if set(reported.scores) != set(episodes.scores):
                raise ValueError(f"the {split} split's episodes report {sorted(reported.scores)}, and it "
                                 f"declares {sorted(episodes.scores)}")
            # A split's episodes read no batch: each of its metrics reads the one artifact the flight made.
            return {} if warm else {f"{split}/{metric.name}": metric.finalize(metric(reported, {}))
                                    for metric in metrics}

        scores = {**agreed(f"{split} episodes", flown), **training_scores}
        scores, elapsed = broadcast_from_process_zero((scores, time.perf_counter() - started))
        return cls(event_step, split, scores, 0, 0, event_words, elapsed, ())


def _pick(artifacts: tuple[Artifact, ...], reads: type):
    """Find the one artifact a metric reads, refusing an ambiguous report."""
    matching = [artifact for artifact in artifacts if isinstance(artifact, reads)]
    if len(matching) != 1:
        raise ValueError(
            f"a metric reads {reads.__name__}, and the objective's evaluation produced "
            f"{[type(a).__name__ for a in artifacts]}")
    return matching[0]


def _artifacts(value: Artifacts | None) -> tuple[Artifact, ...]:
    """Read an objective's report as a tuple, whether it returned one or many."""
    return () if value is None else value if isinstance(value, tuple) else (value,)


class _LossSums:
    """The validation loss's statistics, summed on the devices in batch order by a program of Dew's
    (`_summed`), reduced there (`reduced`) and read once after the split, every rank agreeing on the
    value: each a program a run compiled ahead takes (`warmed_ahead`)."""

    def __init__(self) -> None:
        self._total = None

    def add(self, statistics) -> None:
        if statistics is None:
            return
        self._total = statistics if self._total is None else _summed(self._total, statistics)

    def value(self, objective: Objective[Loss, Effects], mesh: Mesh) -> float | None:
        reduced = self.reduced(objective, mesh)
        if reduced is None:
            return None
        value, valid = reduced
        if not bool(valid) or not np.isfinite(float(value)):
            raise ValueError("validation loss has no finite statistical support")
        return float(value)

    def reduced(self, objective: Objective[Loss, Effects], mesh: Mesh
                ) -> tuple[np.ndarray, np.ndarray] | None:
        """The sums' loss and whether it has support, unchecked, reduced where the sums are (`reduction`)
        and read home on every rank; None where nothing was summed."""
        reduction = self.reduction(objective, mesh)
        return None if reduction is None else collective_host(reduction, phase="validation loss")

    def reduction(self, objective: Objective[Loss, Effects], mesh: Mesh):
        """The sums reduced on the devices of `mesh` that hold them (`Objective.reduce_loss`), traced under
        that mesh as the batches' scoring is, so its program is one whatever devices it is lowered for
        (`warmed_ahead`), its shapes where the sums are shapes; None where nothing was summed."""
        if self._total is None:
            return None
        with jax.set_mesh(mesh):
            return objective._validation_reduction(self._total)


@program
def _summed(total, statistics):
    """Two batches' loss statistics summed, leaf by leaf (`_LossSums`)."""
    return jax.tree.map(operator.add, total, statistics)


class _Accumulators:
    """Each metric's running accumulator over the batches scored so far.

    A metric owns its accumulator type; the pass only carries one per metric
    and hands it back to the same metric to merge and finalize.
    """

    def __init__(self) -> None:
        self._held: dict[str, Any] = {}

    def add[S](self, metric: Metric[S], contribution: S) -> None:
        held = self._held.get(metric.name)
        self._held[metric.name] = contribution if held is None else metric.merge(held, contribution)

    def finalize[S](self, metric: Metric[S]) -> float:
        return float(metric.finalize(self._held[metric.name]))


def _agree_configuration(
    metrics: Sequence[Metric], batches, split: str, *, loss: bool = False, training: bool = False,
    warm: bool = False,
) -> None:
    """Check this rank's evaluation settings, then agree they match root's.

    Ranks that disagree about the split, the metrics, whether there is a
    validation stream or whether the pass warms would walk different phases
    below, and hang at a collective one of them never reaches.
    """
    def checked() -> list[bool | str | list[list[str]]]:
        names = [metric.name for metric in metrics]
        if len(names) != len(set(names)):
            raise ValueError("evaluation metric names must be unique")
        if not split or "/" in split:
            raise ValueError("evaluation split must be a nonempty name without '/'")
        if any("/" in name for name in names):
            raise ValueError(f"evaluation metric names hold no '/', which a score's name puts after its "
                             f"split's: {names}")
        # Each metric's name and the module and qualified name of the artifact it reads.
        return [batches is not None, split,
                [[metric.name, metric.reads.__module__, metric.reads.__qualname__] for metric in metrics]]

    agreed_same("validation availability, split, ordered metric names and types, loss and warming",
                lambda: [checked(), loss, training, warm])


def _event(key: jax.Array, step: int | jax.Array, schedule_step: int | jax.Array | np.ndarray | None,
           averaged: Variables | None, training: Mapping[str, jax.Array] | None = None,
           updates: int | jax.Array | None = None) -> tuple[int, Step, tuple[int, ...], dict[str, float]]:
    """Draw the event's clocks and RNG, and the identity every rank reports.

    The clocks come home through a collective and every rank takes root's,
    so a rank whose own step differs uses root's step and schedule step. The
    key folds in 'EVAL' and that step, so an evaluation never draws what a
    training step at the same clock drew; a rank whose key differs from its
    peers' is refused, by the identity they compare.
    """
    scheduled = step if schedule_step is None else schedule_step
    step_home, schedule_home, updates_home, reports = collective_host(
        (step, scheduled, scheduled if updates is None else updates, dict(training or {})),
        phase="evaluation clocks",
    )

    def root_clocks() -> list | None:
        # Root's clocks as root holds them; a clock root cannot read is refused on every rank.
        schedule = np.asarray(schedule_home)
        return ([int(step_home), schedule.item(), schedule.dtype.name, int(updates_home)]
                if jax.process_index() == 0 else None)

    at, schedule_at, dtype, updates_at = broadcast_from_process_zero(agreed("evaluation clocks", root_clocks))

    def event() -> tuple[Step, jax.Array]:
        context = _context(key, at, jax.device_put(np.asarray(schedule_at, np.dtype(dtype))),
                           jax.device_put(np.asarray(updates_at, np.dtype(dtype))), averaged)
        return context, jax.random.key_data(context.key)

    context, event_data = agreed("evaluation context", event)
    event_words = agreed_same("evaluation key", lambda: [int(word) for word in np.asarray(collective_host(
        event_data, phase="event identity"))])
    return at, context, tuple(event_words), {name: float(value) for name, value in reports.items()}


_EVENT, _SCORE, _PREVIEW = 0x4556414C, 0x53434F52, 0x50524556
"""The words an evaluation's keys fold in ('EVAL', 'SCOR', 'PREV'): its event's, from the run's key and its
step (`_context`), and its scoring's and preview's, from the event's."""


def _context(key: jax.Array, at: int, schedule: jax.Array, committed: jax.Array,
             averaged: Variables | None) -> Step:
    """The step an evaluation at step `at` hands its objective: its key folded from `key` with 'EVAL' and
    `at`, the step its schedules read and the updates committed (`schedule`, `committed`, placed), and the
    averaged weights. An evaluation's model calls train nothing, its metrics, loss and preview alike, so a
    model that reads the run's progress reads its end (`Objective.progress`) whatever the pass reports."""
    return Step(step=schedule, key=_folded(_folded(key, _EVENT), at), ema=averaged, training=False,
                committed=committed)


def _folded(key: jax.Array | jax.ShapeDtypeStruct, word: int) -> jax.Array:
    """`jax.random.fold_in(key, word)`, the host integer placed beside the
    key by name rather than moved there implicitly by the fold: on the key's
    devices, or uncommitted with an uncommitted key. Folded by a program of
    Dew's (`_fold`), as every program a run runs is, so one compiled ahead
    runs in its place (`dew.nn.kernels.options.taking`); a key's shape, as
    a pass compiled ahead holds it (`warmed_ahead`), folds over the word's."""
    if isinstance(key, jax.ShapeDtypeStruct):
        return _fold(key, jax.ShapeDtypeStruct((), np.uint32, sharding=key.sharding))
    return _fold(key, jax.device_put(np.uint32(word), key.sharding if key.committed else None))


@program
def _fold(key: jax.Array, word: jax.Array) -> jax.Array:
    return jax.random.fold_in(key, word)


def _score_split(objective: Objective[Loss, Effects], variables: Variables, batches,
                 context: Step, mesh: Mesh | None, *, metrics: Sequence[Metric],
                 split: str, root: bool, preview_enabled: bool,
                 score_key: jax.Array, preview_key: jax.Array, loss: bool = False,
                 warm: bool = False, written: Path | None = None, kept: bool = False
                 ) -> tuple[dict[str, float], tuple[Artifact, ...], int, int, float | None]:
    """Score every record of a validation split, or its first `WARM_BATCHES`
    batches unfinalized where `warm`.

    Returns the finalized scores, root's previews, and what every rank agrees
    on: the batches scored, the real records they held, and the share of the
    tokens they computed their loss scored (`Evaluation.real_token_share`).

    Every rank walks these phases in the same order. The pass ends when no
    rank has a batch left; until then a rank that has drained scores a copy
    of its last batch whose rows are all repeats (`_covered`), so it meets
    its peers at every collective. Where the pool keeps the pass's decisions
    (`kept`, root's ask), every rank scores each batch and root writes it
    to `written`. Where the objective reports each pass of a looped model
    (`PassReports`), every metric and the loss score each pass too, under
    `<split>/pass_<k>/`.
    """
    summaries = _Accumulators()
    loss_sums = _LossSums()
    # Each pass's metrics and loss where the objective reports its passes, as many as its first report holds.
    passes: list[tuple[_Accumulators, _LossSums]] = []
    scores: dict[str, float] = {}
    previews: tuple[Artifact, ...] = ()
    source = iterator = None
    last = None
    scored = records = 0
    # The tokens the batches' loss scored and those they computed, on the host, for a split the loss
    # scores, whose batches hold what it reads; None where undeclared.
    tokens: list[int] | None = [0, 0] if objective.counts_tokens and loss else None
    seen: set = set()
    try:
        def open_source() -> None:
            # Each name is bound as it is built, so a failure part way
            # through still leaves the cleanup below what to close.
            nonlocal mesh, source, iterator
            mesh = MeshSpec().build() if mesh is None else mesh
            source = batches(DataPartition.of(mesh))
            iterator = iter(source)

        agreed("iterator construction", open_source)
        assert iterator is not None and mesh is not None
        # A pass that writes its scoring artifacts evaluates its batches, whatever metrics it scores.
        evaluating = bool(metrics) or kept
        while True:
            held, available = _next_batch(iterator, scored)
            if not available:
                break
            held, last, batch = _covered(mesh, held, last, scored, objective.inputs)
            _admitted(objective, batch, seen)
            # This rank's own real rows, counted on the host; the pool's count
            # is summed once after the split.
            records += int(np.sum(held[VALID_ROWS]))
            tokens = _counted_tokens(objective, held, tokens)
            produced = None
            if evaluating or loss:
                # The objective scores under the mesh, as the step trains
                # under it: the model's placements and its sequence and stage
                # splits read it. A preview decodes, which neither split does.
                with jax.set_mesh(mesh):
                    statistics, produced = agreed(
                        f"scoring batch {scored}",
                        lambda batch=batch, scored=scored: _dispatched(
                            objective, variables, batch, context, scored,
                            loss=loss, evaluate=evaluating, score_key=score_key))
                    loss_sums.add(statistics)
                    reported = _pass_reports(produced)
                    passes.extend((_Accumulators(), _LossSums()) for _ in range(len(reported) - len(passes)))
                    for (pass_statistics, _), (_, sums) in zip(reported, passes, strict=True):
                        sums.add(pass_statistics)
                    if evaluating:
                        produced = _merged(produced, batch, held, scored, mesh=mesh, metrics=metrics,
                                           written=written,
                                           summaries=summaries, root=root,
                                           passes=tuple(held for held, _ in passes))
            if scored == 0 and preview_enabled:
                previews = _previewed(objective, variables, batch, context, preview_key=preview_key,
                                      scored=produced, root=root)
            produced = batch = None
            scored += 1
            if (not metrics and not loss and not kept) or (warm and scored == WARM_BATCHES):
                break
        # Processes on one data share (a replicated stage or sequence axis)
        # read the same rows, so the pool counts each share once.
        # Each share's records and tokens (-1 where its rows declare none), counted once a share.
        shares = from_every_process([int(DataPartition.of(mesh).index), int(records),
                                     *([-1, -1] if tokens is None else tokens)])
        pooled = {index: counts for index, *counts in shares}.values()
        records = sum(counts[0] for counts in pooled)
        tokens = None if any(counts[1] < 0 for counts in pooled) else [
            sum(counts[1] for counts in pooled), sum(counts[2] for counts in pooled)]
        if scored and warm:
            # The reduction compiles here; a part of the split's loss is no score to check.
            for sums in (loss_sums, *(sums for _, sums in passes)):
                sums.reduced(objective, mesh)
        elif scored:
            scores = _finalized(metrics, summaries, split=split, root=root)
            loss_value = loss_sums.value(objective, mesh)
            if loss_value is not None and f'{split}/loss' not in scores:
                scores[f'{split}/loss'] = loss_value
            for at, (held, sums) in enumerate(passes, start=1):
                named = f"{split}/pass_{at}"
                scores.update(_finalized(metrics, held, split=named, root=root))
                pass_loss = sums.value(objective, mesh)
                if pass_loss is not None:
                    scores[f"{named}/loss"] = pass_loss
    finally:
        _close_source(iterator if iterator is not None else source)
    share = None if tokens is None or not tokens[1] else tokens[0] / tokens[1]
    return scores, previews, scored, records, share


def _counted_tokens(objective: Objective[Loss, Effects], held: Batch, tokens: list[int] | None
                    ) -> list[int] | None:
    """`tokens`, the split's scored and computed tokens so far, with `held`'s: its real rows' scored
    tokens (`Objective.scored_tokens`) and every row's computed ones (`Objective.row_tokens`, from its
    shape); None from the first batch whose rows declare none."""
    per_row = None if tokens is None else objective.row_tokens(held)
    if tokens is None or per_row is None:
        return None
    real = np.asarray(held[VALID_ROWS]).reshape(-1) > 0
    scored = np.asarray(objective.scored_tokens(held), np.int64).reshape(-1)
    return [tokens[0] + int(np.sum(np.where(real, scored, 0))), tokens[1] + rows_of(held) * per_row]


def _admitted(objective: Objective[Loss, Effects], batch: Batch, seen: set) -> None:
    """Check a batch of a shape the split has not shown (`seen`) against what the objective declares
    (`InputSpec.check`), as a training step checks its own (`Trainer.fit`)."""
    inputs = objective.inputs
    device, _ = split_fields(batch)
    signature = batch_signature(device)
    if inputs is None or signature in seen:
        return
    seen.add(signature)
    # A split may carry fewer fields than training, as an objective that scores from prompts reads.
    agreed("validation input declaration", lambda: inputs.check(device, whole=False))


def admit_split(objective: Objective[Loss, Effects], batches: Reader | Episodes, mesh: Mesh,
                split: str) -> None:
    """Read a validation split's first batch and check it against what the objective declares
    (`_admitted`), before a run that scores the split only past its first step trains on: a split of
    undeclared inputs is refused before an update, not after training. Its later batches are checked
    as they are scored. A split scored by its episodes (`Episodes`) reads none."""
    if objective.inputs is None or episodic(batches):
        return
    source = iterator = None
    try:
        def open_source() -> None:
            nonlocal source, iterator
            source = batches(DataPartition.of(mesh))
            iterator = iter(source)

        agreed(f"{split} iterator construction", open_source)
        assert iterator is not None
        # The phases `_score_split` walks for its first batch, so every rank meets its peers.
        held, available = _next_batch(iterator, 0)
        if available:
            _, _, batch = _covered(mesh, held, None, 0, objective.inputs)
            _admitted(objective, batch, set())
    finally:
        _close_source(iterator if iterator is not None else source)


WARM_BATCHES = 2
"""The batches a warming pass scores (`Evaluation.run`'s `warm`): the first,
then a later one, which the accumulators pool into it."""


def warmed_ahead(objective: Objective[Loss, Effects], variables: Variables, batches: Split | Dataset | None,
                 mesh: Mesh, *, key: jax.Array, schedule: jax.Array,
                 averaged: Variables | None, metrics: Sequence[Metric], loss: bool,
                 kept: bool = False) -> None:
    """The programs a warming pass over `batches` runs (`Evaluation.run`'s `warm`), called over shapes where
    a run compiled ahead collects them (`dew.nn.kernels.options.collecting`, `dew.training.ahead.started`):
    its event's context (`_context`) and scoring key, and over the split's first `WARM_BATCHES` batches as
    this process reads and places them (`placed_abstract`), each batch's scoring (`_dispatched`) and the
    loss's sum and reduction (`_LossSums`), each where `_score_split` calls it. `variables`, `averaged` and
    `key` are the state's as placed, and `schedule` the shape of the step the objective's schedules read,
    the committed updates' too; `kept`, a pass that writes what it scores (`scored_to`), which reports its
    artifacts beside its loss, as `_score_split` scores it. The metrics' merges are the host's. A
    scoring that runs an `evaluate` of the objective's own (`Objective._reports_by_program`), which may
    read the batch on the host, a
    preview, which a run compiled ahead has no tracker to show, and an episodic split's flights compile
    where the run makes them."""
    context = _context(key, 0, schedule, schedule, averaged)
    evaluating = bool(metrics) or kept
    if (batches is None or episodic(batches) or not (evaluating or loss)
            or not objective._reports_by_program(loss=loss, evaluate=evaluating)):
        return
    if isinstance(batches, Dataset):
        from dew.data.compile import held
        batches = held(batches, objective=objective.data())
    score_key = _folded(context.key, _SCORE)
    sums = _LossSums()
    iterator = iter(batches(DataPartition.of(mesh)))
    try:
        for index, batch in zip(range(WARM_BATCHES), iterator, strict=False):
            with jax.set_mesh(mesh):
                statistics, _ = _dispatched(objective, variables, placed_abstract(mesh, _validated(batch)),
                                            context, index, loss=loss, evaluate=evaluating,
                                            score_key=score_key)
                sums.add(statistics)
        sums.reduction(objective, mesh)
    finally:
        _close_source(iterator)


def _next_batch(iterator: Iterator, index: int) -> tuple[Batch | None, int]:
    """Read one batch, and agree how many ranks still had one to read.

    The count is how many ranks hold a batch, so every rank ends the
    coordinated prefix at the same batch.
    """
    error = None
    batch = None
    try:
        # A drained iterator is the end of the split, which the phase below
        # agrees on; anything else is a failure.
        with contextlib.suppress(StopIteration):
            batch = next(iterator)
    except BaseException as failure:
        error = failure
    available = agree_process_phase(
        error, phase=f"iterator next batch {index}", available=batch is not None)
    return batch, available


def _covered(mesh: Mesh, batch: Batch | None, last: Batch | None, index: int, inputs: InputSpec | None
             ) -> tuple[Batch, Batch, Batch]:
    """The batch this rank scores at `index`, the one it keeps to cover the
    next, and the first placed on the mesh, under one agreement.

    Its own batch carries `VALID_ROWS`, every row real where the reader
    marked none, and is refused on the host before it is placed where its token ids fall outside its
    model's table (`InputSpec.bounded`, as each training batch is), inside the agreement, so every
    process refuses with the one that holds the id. A rank whose share has run out while a peer's has
    not scores a copy of its last batch with every row a repeat.
    """
    def cover() -> tuple[Batch, Batch, Batch]:
        if batch is not None:
            if inputs is not None:
                inputs.bounded(batch)
            held = kept = _validated(batch)
        elif last is None:
            raise ValueError(
                f"process {jax.process_index()}'s reader yields no batch while a peer's yields one; "
                "Dew's readers give an empty share one batch whose rows are all repeats "
                "(`VALID_ROWS` False), and a reader of your own has to as well")
        else:
            held, kept = {**last, VALID_ROWS: np.zeros(rows_of(last), bool)}, last
        return held, kept, shard_batch(mesh, held)

    return agreed(f"batch cover and placement {index}", cover)


def _validated(batch: Batch) -> Batch:
    """`batch` carrying `VALID_ROWS`, every row real where the reader marked none."""
    return batch if VALID_ROWS in batch else {**batch, VALID_ROWS: np.ones(rows_of(batch), bool)}


def _cut(leaf, keep: np.ndarray):
    """`leaf` without the rows `keep` marks as repeats, when its leading axis
    is the batch's rows: an array, a per-row tuple of captions or texts, or a
    per-row list of opaque records (`dew.data.dataset.take_rows`)."""
    if not row_field(leaf, keep.size):
        return leaf
    return take_rows(leaf if isinstance(leaf, list | tuple) else np.asarray(leaf), np.flatnonzero(keep))


def _real_artifact[A: Artifact](artifact: A, keep: np.ndarray) -> A:
    """`artifact` with every per-row field cut to the real rows; one whose metrics read their rows through
    a mask of its own (`FixedRows`) keeps its batch's fixed shape with that mask cleared on its padding
    rows, so a metric reads every batch of a split, a short tail among
    them, in one shape."""
    if isinstance(artifact, FixedRows):
        return artifact.unscored(~keep)
    return dataclasses.replace(artifact, **{field.name: _cut(getattr(artifact, field.name), keep)
                                            for field in dataclasses.fields(artifact)})


def _real_batch(batch: Batch, keep: np.ndarray) -> Batch:
    """`batch` cut to the real rows, without `VALID_ROWS`: a column of opaque records, a list, is
    cut as one column, as `dew.data.dataset.take_batch` selects its rows."""
    return jax.tree.map(lambda leaf: _cut(leaf, keep),
                        {name: leaf for name, leaf in batch.items() if name != VALID_ROWS},
                        is_leaf=lambda leaf: isinstance(leaf, list))


def _dispatched(objective: Objective[Loss, Effects], variables: Variables, batch: Batch, context: Step,
                index: int, *, loss: bool, evaluate: bool, score_key: jax.Array):
    """One batch's device work, dispatched under one key: the validation-loss
    statistics where `loss`, and what the objective produces where `evaluate`."""
    keyed = replace(context, key=_folded(score_key, index))
    # The objective's programs take what a device holds; the metrics read the rest (`_merged`).
    device, _ = split_fields(batch)
    return objective._validation_report(variables, device, keyed, loss=loss, evaluate=evaluate)


def _model_output(produced):
    """An objective's report without its passes' (`PassReports`): what it scored of its model's output."""
    if not isinstance(produced, tuple):
        return produced
    kept = tuple(artifact for artifact in produced if not isinstance(artifact, PassReports))
    return kept[0] if len(kept) == 1 else kept


def _as_placed(leaf) -> np.ndarray:
    """`leaf` as placing it and reading it back gives it: in JAX's canonical
    dtype, where float64 is float32 unless x64 is on."""
    host = np.asarray(jax.device_get(leaf))
    return host.astype(jax.dtypes.canonicalize_dtype(host.dtype), copy=False)


def _pass_reports(produced) -> tuple[tuple[Ratio | Weighted | None, Artifacts], ...]:
    """Each pass's `(loss statistics, artifacts)` an objective's report holds (`PassReports`), none where
    it reports its model's output alone."""
    return next((artifact.passes for artifact in _artifacts(produced) if isinstance(artifact, PassReports)),
                ())


def _merged(produced, batch: Batch, held: Batch, index: int, *, mesh: Mesh, metrics: Sequence[Metric],
            summaries: _Accumulators, root: bool, written: Path | None = None,
            passes: Sequence[_Accumulators] = ()):
    """Merge one batch into every metric's accumulator on root, and each pass's artifacts into that pass's
    (`passes`, where the objective reports them, `PassReports`), returning what the objective produced
    about its model's output, hosted.

    The artifacts come home, and so does the batch the metrics read: on one
    process that is the host batch it was placed from, as the devices hold
    it, so nothing copies back; a batch spread over processes comes home
    with the artifacts, its host fields from every share (`_shares_host`).
    One agreement covers every metric, so a failure on root stops its peers
    there. Where `written` names a place, root writes each of the batch's scoring artifacts there,
    `<written>-<batch>`, its real rows alone (`write_artifact`).
    """
    if all(leaf.is_fully_addressable for leaf in jax.tree.leaves(batch) if isinstance(leaf, jax.Array)):
        produced = collective_host(produced, phase=f"scoring batch {index}")
        device, host = split_fields(held)
        home = with_host(jax.tree.map(_as_placed, device), host)
    else:
        device, host = split_fields(batch)
        if host:
            rows = rows_of(held)
            produced, home, origins = collective_host(
                (produced, device, row_origins(mesh, rows)), phase=f"scoring batch {index}")
            home = with_host(home, shared_host_fields(mesh, host, np.asarray(origins), rows))
        else:
            produced, home = collective_host((produced, device), phase=f"scoring batch {index}")

    reported = _pass_reports(produced)
    produced = _model_output(produced)

    def merge() -> None:
        if not root:
            return
        # The metrics read the real rows alone, so a repeat filling out the
        # split's last batch scores nothing.
        keep = np.asarray(home[VALID_ROWS], bool)
        artifacts = tuple(_real_artifact(artifact, keep) for artifact in _artifacts(produced))
        real = _real_batch(home, keep)
        for metric in metrics:
            summaries.add(metric, metric(_pick(artifacts, metric.reads), real))
        for (_, pass_artifacts), held_pass in zip(reported, passes, strict=True):
            cut = tuple(_real_artifact(artifact, keep) for artifact in _artifacts(pass_artifacts))
            for metric in metrics:
                held_pass.add(metric, metric(_pick(cut, metric.reads), real))
        if written is not None:
            # Every batch's scoring artifacts, its real rows alone, for pooling a pass offline.
            written.parent.mkdir(parents=True, exist_ok=True)
            for at, artifact in enumerate(artifacts):
                suffix = f"-{at}" if len(artifacts) > 1 else ""
                write_artifact(artifact, written.with_name(f"{written.name}-{index:05d}{suffix}"))

    agreed(f"metrics batch {index}", merge)
    return produced


def _previewed(objective: Objective[Loss, Effects], variables: Variables, batch: Batch,
               context: Step, *, preview_key: jax.Array, scored,
               root: bool) -> tuple[Artifact, ...]:
    """Ask the objective for one preview of the first batch, hosted on root.

    Every rank takes part in the generation agreement and the transfer that
    brings the artifacts home; only root keeps them.
    """
    produced = agreed("preview generation/decoding", lambda: objective.preview(
        variables, batch, replace(context, key=preview_key), scored=scored))
    produced = collective_host(produced, phase="preview artifacts", held_by="first")
    return _artifacts(produced) if root else ()


def _finalized(metrics: Sequence[Metric], summaries: _Accumulators, *,
               split: str, root: bool) -> dict[str, float]:
    """Reduce each metric's accumulator to one number, agreeing per metric.

    Only root holds accumulators, so its peers reach each agreement with
    nothing to reduce; a failure on root stops them here.
    """
    scores: dict[str, float] = {}
    for metric in metrics:
        def finalize(metric=metric) -> None:
            if root:
                scores[f"{split}/{metric.name}"] = summaries.finalize(metric)

        agreed(f"finalizing metric {metric.name}", finalize)
    return scores


def _close_source(held) -> None:
    """Close the validation iterator, agreeing a failure the pass itself had not.

    A cleanup failure on top of a failing pass becomes a note on that
    failure: the pass's own error is the one worth raising, and the peers
    are already unwinding with it.
    """
    primary = sys.exception()
    error = None
    try:
        if isinstance(held, Closeable):
            held.close()
    except BaseException as failure:
        if primary is not None:
            primary.add_note(f"Validation iterator cleanup failed: {failure!r}")
        else:
            error = failure
    if primary is None:
        agree_process_phase(error, phase="iterator cleanup")


def recorded_name[S, A](key, names: Collection[str], owner: Objective[S, A] | None = None) -> str:
    """The recorded name a run's score object names, one rule for the scores `best=` and `stop=` read
    live and the ones a checkpoint keeps: a `TrainingScalar` the training split's (of the objective
    `owner` where it is given, by identity), a `Metric` the validation split's ('val'), a `Summary` its
    namespace's (`Summary.SPLIT`), and a `(split, metric)` pair that split's. A metric its split did not
    record raises with the splits that did."""
    split = Summary.SPLIT if isinstance(key, Summary) else "val"
    if isinstance(key, tuple) and len(key) == 2:
        split, key = key
    if isinstance(key, TrainingScalar):
        if owner is not None and key.owner is not owner:
            raise KeyError("training score belongs to another objective")
        return f"train/{key.name}"
    if isinstance(key, (Metric, Summary)):
        if f"{split}/{key.name}" in names:
            return f"{split}/{key.name}"
        splits = sorted(name.rpartition("/")[0] for name in names if name.rpartition("/")[2] == key.name)
        raise KeyError(f"metric {key.name!r} has no {split} score" + (
            f"; it is recorded under {', '.join(splits)}: ask (split, metric)" if splits else ""))
    raise KeyError(key)


class MetricScores[Statistics, Additions](Mapping):
    """A fit's scores keyed by the metric objects, objective scalars and
    `(split, metric)` pairs that `best=` and `stop=` name, or by recorded name.
    Each is resolved against what the fit declares (`declared`, each split's
    metrics and loss, and the objective's training scalars): a declared score
    this evaluation did not record, as a split evaluated on a cadence of its
    own, reads as NaN, which ranks nothing; one the fit does not declare, or a
    metric `fit` was not given, raises."""

    def __init__(
        self,
        declared: Mapping[str, Sequence[Declared]],
        scores: Mapping[str, float],
        objective: Objective[Statistics, Additions],
    ):
        self.objective = objective
        self.declared = declared
        self.training = {f"train/{name}": objective.scalars[name] for name in ('loss', *objective.shown)}
        every = {id(metric): metric for metrics in declared.values() for metric in metrics}
        self.metrics = (*every.values(), *self.training.values())
        # Each split records its loss beside its metrics; the summaries' namespace records none.
        self.names = {*scores, *self.training,
                      *(f"{split}/loss" for split in declared if split != Summary.SPLIT),
                      *(f"{split}/{metric.name}" for split, metrics in declared.items()
                        for metric in metrics)}
        self.scores = scores

    def __getitem__(self, metric):
        if (
            isinstance(metric, types.MethodType)
            and metric.__name__ == "loss"
            and metric.__self__ is self.objective
        ):
            name = 'train/loss'
        elif isinstance(metric, str):
            name = metric
        else:
            named = metric[1] if isinstance(metric, tuple) and len(metric) == 2 else metric
            if not isinstance(named, TrainingScalar) and not any(known is named for known in self.metrics):
                raise KeyError("the score refers to a metric not passed to fit")
            name = recorded_name(metric, self.names, self.objective)
        if name not in self.names:
            raise KeyError(f"{name!r} is not a score this fit records")
        return self.scores.get(name, float("nan"))

    def __iter__(self):
        # Each score by the key that reads it back (`recorded_name`): a training scalar, its split's own
        # metric (the validation split's alone, another's with its split), else its recorded name, as a
        # split's loss and a summary are.
        for name in self.scores:
            split, _, label = name.rpartition('/')
            metric = next((metric for metric in self.declared.get(split, ())
                           if isinstance(metric, Metric) and metric.name == label), None)
            owned = name if metric is None else metric if split == "val" else (split, metric)
            yield self.training.get(name, owned)

    def __len__(self):
        return len(self.scores)


def schedule_step(state: TrainState, accumulation: int) -> np.ndarray:
    """The step `state`'s schedules read, as training reads it (`Transaction`: a window of window_size
    microbatches is `accumulation` steps), on the host, where an evaluation reads its clocks
    (`Evaluation.run`), in its clock's dtype: no program runs it on the devices."""
    microstep = np.asarray(state.microstep)
    return microstep * microstep.dtype.type(accumulation) // np.asarray(state.window_size)


@dataclass(frozen=True)
class Evaluator[L, E]:
    """A run's evaluations of its suites (`FitPlan.suites`) between its steps and at its end: what each
    scores, the rules it traces under, and the reports it writes. It holds what a run states once: its
    objective, its layout and meshes (`mesh` the devices', `state_mesh` where a CPU-owned run keeps its
    state), the microbatches a pipelined step feeds, the steps a window pools (`accumulation`), where it
    reports (`display`, `tracker`), and whether the run ranks its checkpoints (`ranks`); a state, its
    placement and the links its step traced with (`Link`) are each call's."""

    objective: Objective[L, E]
    layout: Layout
    mesh: Mesh
    state_mesh: Mesh
    microbatches: int
    accumulation: int
    display: TrainingDisplay
    tracker: Tracker | None
    ranks: bool

    def warm(self, plan: FitPlan, state: TrainState, shardings: Placement[TrainState],
             links: Mapping[str, Link], *, first: bool) -> None:
        """Compile the programs the run's evaluations run beside a step's compile, where any runs
        between steps (`FitPlan.evaluates_midway`): every suite warms (`Evaluation.run`'s `warm`),
        its preview included, and nothing is reported.

        An evaluation's programs compile at their first call, which would otherwise be an
        evaluation's inside the run, its seconds holding the compile (7.5 s at the first of 45
        records), counted against the run's goodput. A new training shape can change the modules
        the programs run (the fit ladder's rung), so each step compiled warms them again. The
        training values a ranking reads ride on the first evaluation that reads them, whose gather
        of them compiles there. Where none runs between steps, the run's `first` compile reads each
        split's first batch for its declaration alone (`admit_split`)."""
        if plan.evaluates_midway:
            with region("evaluate"):
                self.evaluation(plan, state, shardings, links, {}, warm=True)
        elif first:
            for split, suite in plan.suites.items():
                admit_split(self.objective, suite.data, self.mesh, split)

    def warmed_ahead(self, plan: FitPlan, state: TrainState, shardings: Placement[TrainState],
                     links: Mapping[str, Link]) -> None:
        """The programs `warm` compiles, called over `state`, the run's first state as shapes placed by
        `shardings`, where a run compiled ahead collects them (`dew.training.ahead.started`): each suite's
        warming pass (`warmed_ahead`) where it evaluates midway, and its evaluations' own where those score
        otherwise (writing what it scores, `scored_to`), the event alone where no suite is scored, over the
        weights, averaged weights and key `evaluate` reads, under the rules it scores under (`scoring`): a
        run that evaluates at its end alone runs those as well."""
        if not plan.evaluates:
            return
        # A warming pass writes nothing; an evaluation writes what it scores where the run says where.
        kinds = {*((False,) if plan.evaluates_midway else ()), plan.scored_to is not None}
        averaged = with_ema(state.variables, fetched_state(state, shardings, layout=self.layout).ema)
        # The step the objective's schedules read, of the dtype `schedule_step` reads it in.
        counters = {name: np.ones(getattr(state, name).shape, getattr(state, name).dtype)
                    for name in ("microstep", "window_size")}
        schedule = shapes_of(jax.ShapeDtypeStruct((), schedule_step(replace(state, **counters),
                                                                    self.accumulation).dtype))
        scored = self.scored(plan) or [(None, None, False)]
        with self.scoring(links):
            for (_, suite, loss), kept in itertools.product(scored, sorted(kinds)):
                warmed_ahead(self.objective, state.variables, None if suite is None else suite.data,
                             self.mesh, key=state.key, schedule=schedule, averaged=averaged,
                             metrics=() if suite is None else suite.metrics, loss=loss, kept=kept)

    def scored(self, plan: FitPlan, names: Sequence[str] | None = None) -> list[tuple[str, EvalSuite, bool]]:
        """The suites in `names`, every suite when None, each by its split and whether an evaluation scores
        its loss: wherever the objective defines one, and where the loss ranks the run's checkpoints or
        the state it holds, which preflight checks it can, unless a metric of the suite is the loss."""
        ranks = (self.ranks or plan.held_best) and not plan.best
        return [(split, suite, (ranks or self.objective.defines_validation_loss)
                 and not any(metric.name == 'loss' for metric in suite.metrics))
                for split, suite in ((split, plan.suites[split])
                                     for split in (plan.suites if names is None else names))]

    @contextlib.contextmanager
    def scoring(self, links: Mapping[str, Link]) -> Iterator[None]:
        """The rules an evaluation traces its programs under: the microbatch count, the layout's rules and
        the mesh's links (`evaluate`, `warmed_ahead`)."""
        with (pipeline_microbatches(self.microbatches),
              nn.logical_axis_rules(self.layout.axis_rules), measured_links(links)):
            yield

    def evaluation(self, plan: FitPlan, state: TrainState, shardings: Placement[TrainState],
                   links: Mapping[str, Link], training: Mapping[str, jax.Array],
                   names: Sequence[str] | None = None, *, warm: bool = False,
                   joined: Mapping[str, float] | None = None, examples: int | None = None,
                   cooled: bool = False) -> Evaluation:
        """Score the suites in `names`, every suite when None, as one evaluation, joined to the scores
        the step's earlier evaluation of the others recorded (`joined`).

        With no suite at all, the evaluation carries the training values a
        ranking or a stopping rule reads. `warm` warms every suite instead and
        reports nothing (`warm`)."""
        ranks = self.ranks or plan.best or plan.stop or plan.held_best
        reports = training if ranks and not warm else None
        chosen = self.scored(plan, names)
        if not chosen:
            return self.evaluate(state, shardings, links, None, (), plan.preview, split="val",
                                 training=reports, warm=warm, examples=examples, cooled=cooled)
        evaluations = []
        for split, suite, loss in chosen:
            evaluations.append(self.evaluate(state, shardings, links, suite.data, suite.metrics, plan.preview,
                                             loss=loss, split=split, training=reports, warm=warm,
                                             scored_to=plan.scored_to, examples=examples, cooled=cooled))
        scores = {**(joined or {}),
                  **{name: value for report in evaluations for name, value in report.scores.items()}}
        # Each summary is formed once a step, here, by the evaluation that completes the splits it weighs.
        formed = {} if warm else {name: value for summary in plan.summaries
                                  for name, value in summary.formed(scores).items() if name not in scores}
        if formed:
            replace(evaluations[0], split=Summary.SPLIT, scores=formed, previews=(),
                    coordinated_batches=0)._report(self.display, self.tracker)
        return replace(evaluations[0], scores={**scores, **formed},
                       elapsed_seconds=sum(report.elapsed_seconds for report in evaluations))

    def evaluate(self, state: TrainState, shardings: Placement[TrainState], links: Mapping[str, Link],
                 reader: Split | None, metrics: Sequence[Metric], preview: bool, *, loss: bool = False,
                 split: str = "val", training: Mapping[str, jax.Array] | None = None,
                 warm: bool = False,
                 scored_to: str | None = None, examples: int | None = None,
                 cooled: bool = False) -> Evaluation:
        """Score the validation split with this state's variables and report
        it, returning the seconds it took: the run's, but not its steps', which is what the goodput fraction
        is measured against. `warm` warms the split (`Evaluation.run`'s `warm`) and reports nothing.

        A CPU-owned run evaluates on the accelerator, over the same snapshot
        a step realizes, so validation reads the weights where the loss does."""
        paused = time.perf_counter()
        if not warm:
            self.display.status("evaluating")
        mesh, params = self.mesh, state.variables
        averaged = with_ema(state.variables, fetched_state(state, shardings, layout=self.layout).ema)
        key = state.key
        if "variables" in self.layout.host:
            from dew.training.execution import HostExecution, stack_sites
            execution = HostExecution(self.objective, self.layout, mesh, self.state_mesh,
                                      stack_sites(self.objective, state.variables))
            with jax.set_mesh(self.state_mesh):
                params, averaged = execution.snapshot(params), execution.snapshot(averaged)
            key = execution.on_accelerator(key)
        assert params is not None, "evaluation always has model variables"
        # The rules and the microbatch count; `Evaluation.run` scores under the mesh itself, and
        # previews decode outside it. The objective's schedules read steps, as training does
        # (`Transaction`): a window of window_size microbatches is `accumulation` steps.
        with self.scoring(links):
            evaluation = Evaluation.run(
                self.objective, params, reader, metrics=metrics, key=key,
                step=state.step, schedule_step=schedule_step(state, self.accumulation),
                averaged=averaged, preview=preview, mesh=mesh, loss=loss, split=split, training=training,
                warm=warm, updates=state.updates,
                # A cooldown branch's beside the run's at the step it branched at (`COOLDOWN`).
                scored_to=None if scored_to is None else os.path.join(
                    scored_to, f"step-{int(state.step)}", *((COOLDOWN,) if cooled else ())))
        # On the examples clock an evaluation is placed by the examples the run had consumed, as its host
        # counted them, its record their coordinate beside its step.
        if examples is not None:
            evaluation = replace(evaluation, examples=examples)
        # A cooldown branch's scores, named under `cooldown/` as its report and its ranking read them.
        evaluation = evaluation.branched() if cooled else evaluation
        if not warm:
            # The training values ride along for ranking (`_ranking`); the log interval reports
            # them, so the evaluation's own report leaves them out.
            replace(evaluation, scores={
                name: value for name, value in evaluation.scores.items() if name not in (training or {})}
            )._report(self.display, self.tracker)
        return replace(evaluation, elapsed_seconds=time.perf_counter() - paused)
