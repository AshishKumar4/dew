"""Coordinated evaluation of trained variables, independent of optimization."""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from dew.artifacts import (
    Artifact,
    Artifacts,
    agree_process_phase,
    agreed,
    broadcast_from_process_zero,
    collective_host,
)
from dew.data.dataset import Closeable, Reader
from dew.objectives.base import Batch, Effects, Loss, Metric, Objective, Step, Variables

from .distributed import build_mesh, data_partition, shard_batch


@dataclass(frozen=True)
class Evaluation:
    """Holds one evaluation event, with bounded hosted previews on rank zero.

    Scores, logical row counts and RNG identity agree across ranks. Scores
    include the split prefix. elapsed_seconds is rank zero's wall time through
    iterator cleanup. Metric accumulators and validation batches are not kept.
    """

    step: int
    split: str
    scores: dict[str, float]
    coordinated_batches: int
    records: int
    uneven_shards: bool
    event_key: tuple[int, ...]
    elapsed_seconds: float
    previews: tuple[Artifact, ...]

    @property
    def scalars(self) -> dict[str, float]:
        """Return the metric values, plus the evaluation/* counts if a batch was scored."""
        if not self.coordinated_batches:
            return dict(self.scores)
        return {**self.scores,
                "evaluation/coordinated_batches": float(self.coordinated_batches),
                "evaluation/records": float(self.records),
                "evaluation/uneven_shards": float(self.uneven_shards)}


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


def evaluate(objective: Objective[Loss, Effects], variables: Variables,
             batches: Reader | None, *,
             key: jax.Array, metrics: Sequence[Metric] = (),
             step: int | jax.Array = 0, averaged: Variables | None = None,
             preview: bool = False, mesh: Mesh | None = None, split: str = "val",
             schedule_step: int | jax.Array | None = None) -> Evaluation:
    """Evaluate a finite coordinated prefix without an optimizer or tracker.

    batches opens a fresh iterator over this process's share of the split
    (`data_partition` of the mesh), owned and closed by this call;
    Dataset.val can be passed directly. Every rank calls evaluate with
    the same objective, metrics and numerical settings. Only root's preview
    flag controls the once-per-event display. No consumers means no iterator
    or objective work; preview alone consumes at most the first coordinated batch.

    variables is the complete Flax variables tree. averaged, when supplied,
    is the complete overlay seen as Step.ema, not an optimizer state. Passing
    state.averaged as variables evaluates those weights directly. step tags
    the event RNG and report; schedule_step defaults to step and preserves an
    objective's accepted-work schedule when training attempts were rejected.

    Scalars are broadcast to every rank. Previews remain hosted on root and
    are bounded by one objective preview, independent of validation length.
    Reporting is the caller's job; pool callers must agree reporting failures
    before entering their next collective. In-flight device failures still
    require distributed runtime termination.
    """
    started = time.perf_counter()
    root = jax.process_index() == 0
    _agree_configuration(metrics, batches, split)
    preview_enabled = bool(broadcast_from_process_zero(root and preview))
    event_step, context, event_words = _event(key, step, schedule_step, averaged)
    score_key = jax.random.fold_in(context.key, 0x53434F52)
    preview_key = jax.random.fold_in(context.key, 0x50524556)
    scores: dict[str, float] = {}
    previews: tuple[Artifact, ...] = ()
    scored = records = 0
    uneven = False
    if batches is not None and (metrics or preview_enabled):
        scores, previews, scored, records, uneven = _score_split(
            objective, variables, batches, context, mesh, metrics=metrics, split=split,
            root=root, preview_enabled=preview_enabled, score_key=score_key,
            preview_key=preview_key)
    elapsed = time.perf_counter() - started
    scores, elapsed = broadcast_from_process_zero((scores, elapsed))
    return Evaluation(event_step, split, scores, scored, records, uneven, event_words, elapsed, previews)


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


@dataclass(frozen=True)
class _Configuration:
    """What every rank must agree on before a validation pass walks its phases."""
    validation: bool
    split: str
    metrics: tuple[tuple[str, str, str], ...]
    """Each metric's name and the module and qualified name of the artifact it reads."""

    def broadcast(self) -> list[bool | str | list[list[str]]]:
        """The record as rank zero's ranks see it, through the JSON broadcast."""
        return [self.validation, self.split, [list(entry) for entry in self.metrics]]


def _agree_configuration(metrics: Sequence[Metric], batches, split: str) -> None:
    """Check this rank's evaluation settings, then agree they match root's.

    Ranks that disagree about the split, the metrics or whether there is a
    validation stream would walk different phases below, and hang at a
    collective one of them never reaches.
    """
    def checked() -> _Configuration:
        names = [metric.name for metric in metrics]
        if len(names) != len(set(names)):
            raise ValueError("evaluation metric names must be unique")
        if not split or "/" in split:
            raise ValueError("evaluation split must be a nonempty name without '/'")
        return _Configuration(
            validation=batches is not None, split=split,
            metrics=tuple((metric.name, metric.reads.__module__, metric.reads.__qualname__)
                          for metric in metrics))

    configuration = agreed("configuration", checked).broadcast()
    root_configuration = broadcast_from_process_zero(configuration)
    error = None if configuration == root_configuration else ValueError(
        "validation availability, split and ordered metric names/types must agree across ranks")
    agree_process_phase(error, phase="configuration agreement")


def _event(key: jax.Array, step: int | jax.Array, schedule_step: int | jax.Array | None,
           averaged: Variables | None) -> tuple[int, Step, tuple[int, ...]]:
    """Draw the event's clocks and RNG, and the identity every rank reports.

    The clocks come home through a collective, so a rank whose own step
    differs uses root's. The key folds in 'EVAL' and that step, so an
    evaluation never draws what a training step at the same clock drew.
    """
    step_home, schedule_home = collective_host(
        (step, step if schedule_step is None else schedule_step), phase="evaluation clocks")

    def event() -> tuple[int, Step, jax.Array]:
        at = int(step_home)
        event_key = jax.random.fold_in(jax.random.fold_in(key, 0x4556414C), at)
        return (at, Step(step=jnp.asarray(schedule_home), key=event_key, ema=averaged),
                jax.random.key_data(event_key))

    event_step, context, event_data = agreed("evaluation context", event)
    event_words = tuple(int(word) for word in np.asarray(collective_host(
        event_data, phase="event identity")))
    return event_step, context, event_words


def _score_split(objective: Objective[Loss, Effects], variables: Variables, batches,
                 context: Step, mesh: Mesh | None, *, metrics: Sequence[Metric],
                 split: str, root: bool, preview_enabled: bool,
                 score_key: jax.Array, preview_key: jax.Array,
                 ) -> tuple[dict[str, float], tuple[Artifact, ...], int, int, bool]:
    """Score the coordinated prefix of a validation split.

    Returns the finalized scores, root's previews, and the three counts every
    rank agrees on: the batches scored, the records read, and whether the
    prefix ended because some ranks ran out of batches before others.

    Every rank walks these phases in the same order, so a rank that drains
    early stops the pool at the batch agreement rather than at a collective
    its peers have already left.
    """
    summaries = _Accumulators()
    scores: dict[str, float] = {}
    previews: tuple[Artifact, ...] = ()
    source = iterator = None
    scored = records = 0
    uneven = False
    try:
        def open_source() -> None:
            # Each name is bound as it is built, so a failure part way
            # through still leaves the cleanup below what to close.
            nonlocal mesh, source, iterator
            mesh = build_mesh() if mesh is None else mesh
            source = batches(data_partition(mesh))
            iterator = iter(source)

        agreed("iterator construction", open_source)
        assert iterator is not None and mesh is not None
        while True:
            batch, available = _next_batch(iterator, scored)
            if available != jax.process_count():
                uneven = available > 0
                batch = None
                break
            assert batch is not None
            batch, rows = _placed_batch(mesh, batch, scored)
            records += rows
            produced = None
            if metrics:
                # The objective scores under the mesh, as the step trains
                # under it: the model's placements and its sequence and stage
                # splits read it. A preview decodes, which neither split does.
                with jax.set_mesh(mesh):
                    produced = _scored_batch(objective, variables, batch, context, scored,
                                             metrics=metrics, summaries=summaries,
                                             score_key=score_key, root=root)
            if scored == 0 and preview_enabled:
                previews = _previewed(objective, variables, batch, context,
                                      preview_key=preview_key, scored=produced, root=root)
            produced = batch = None
            scored += 1
            if not metrics:
                break
        if scored:
            scores = _finalized(metrics, summaries, split=split, root=root)
    finally:
        _close_source(iterator if iterator is not None else source)
    return scores, previews, scored, records, uneven


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


def _placed_batch(mesh: Mesh, batch: Batch, index: int) -> tuple[Batch, int]:
    """Shard one validation batch onto the mesh, and count the rows it holds."""
    def place() -> tuple[Batch, int]:
        placed = shard_batch(mesh, batch)
        rows = next((leaf.shape[0] for leaf in jax.tree.leaves(placed) if leaf.ndim), None)
        if rows is None:
            raise ValueError("validation batch has no row-bearing array")
        return placed, int(rows)

    return agreed(f"batch placement {index}", place)


def _scored_batch(objective: Objective[Loss, Effects], variables: Variables, batch: Batch,
                  context: Step, index: int, *, metrics: Sequence[Metric],
                  summaries: _Accumulators, score_key: jax.Array, root: bool):
    """Score one batch into `summaries`, returning what the objective produced.

    The report and the batch come home together, so each metric on root
    reads hosted arrays. Every rank reaches every metric's agreement,
    whether or not it holds an accumulator.
    """
    produced = agreed(f"scoring batch {index}", lambda: objective.evaluate(
        variables, batch, replace(context, key=jax.random.fold_in(score_key, index))))
    produced, home = collective_host((produced, batch), phase=f"scoring batch {index}")
    artifacts = _artifacts(produced)
    for metric in metrics:
        def merge() -> None:
            if not root:
                return
            summaries.add(metric, metric(_pick(artifacts, metric.reads), home))

        agreed(f"metric {metric.name} batch {index}", merge)
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
    produced = collective_host(produced, phase="preview artifacts")
    return _artifacts(produced) if root else ()


def _finalized(metrics: Sequence[Metric], summaries: _Accumulators, *,
               split: str, root: bool) -> dict[str, float]:
    """Reduce each metric's accumulator to one number, agreeing per metric.

    Only root holds accumulators, so its peers reach each agreement with
    nothing to reduce; a failure on root stops them here.
    """
    scores: dict[str, float] = {}
    for metric in metrics:
        def finalize() -> None:
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
