"""Coordinated evaluation of trained variables, independent of optimization."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from dew.artifacts import Artifact, agree_process_phase, broadcast_from_process_zero, collective_host
from dew.objectives.base import Batch, Effects, Loss, Metric, Objective, Step, Variables
from .distributed import build_mesh, shard_batch


@dataclass(frozen=True)
class Evaluation:
    """One evaluation event, with bounded hosted previews on rank zero only.

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
        """Metric values and the existing evaluation count keys for reporting."""
        if not self.coordinated_batches:
            return dict(self.scores)
        return {**self.scores,
                "evaluation/coordinated_batches": float(self.coordinated_batches),
                "evaluation/records": float(self.records),
                "evaluation/uneven_shards": float(self.uneven_shards)}


def _pick(artifacts: tuple[Artifact, ...], reads: type):
    matching = [artifact for artifact in artifacts if isinstance(artifact, reads)]
    if len(matching) != 1:
        raise ValueError(
            f"a metric reads {reads.__name__}, and the objective's evaluation produced "
            f"{[type(a).__name__ for a in artifacts]}")
    return matching[0]


def evaluate(objective: Objective[Loss, Effects], variables: Variables,
             batches: Callable[[], Iterator[Batch]] | None, *,
             key: jax.Array, metrics: Sequence[Metric] = (),
             step: int | jax.Array = 0, averaged: Variables | None = None,
             preview: bool = False, mesh: Mesh | None = None, split: str = "val",
             schedule_step: int | jax.Array | None = None) -> Evaluation:
    """Evaluate a finite coordinated prefix without an optimizer or tracker.

    batches opens a fresh process-local iterator owned and closed by this
    call; Dataset.val can be passed directly. Every rank calls evaluate with
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
    configuration = None
    error = None
    try:
        names = [metric.name for metric in metrics]
        if len(names) != len(set(names)):
            raise ValueError("evaluation metric names must be unique")
        if not split or "/" in split:
            raise ValueError("evaluation split must be a nonempty name without '/'")
        configuration = {"validation": batches is not None, "split": split,
                         "metrics": [[metric.name, metric.reads.__module__, metric.reads.__qualname__]
                                     for metric in metrics]}
    except BaseException as failure:
        error = failure
    agree_process_phase(error, phase="configuration")
    root_configuration = broadcast_from_process_zero(configuration)
    error = None if configuration == root_configuration else ValueError(
        "validation availability, split and ordered metric names/types must agree across ranks")
    agree_process_phase(error, phase="configuration agreement")
    preview_enabled = bool(broadcast_from_process_zero(root and preview))
    step_home, schedule_home = collective_host(
        (step, step if schedule_step is None else schedule_step), phase="evaluation clocks")
    error = None
    event_step = 0
    context = event_data = None
    try:
        event_step = int(step_home)
        event_key = jax.random.fold_in(jax.random.fold_in(key, 0x4556414C), event_step)
        context = Step(step=jnp.asarray(schedule_home), key=event_key, ema=averaged)
        event_data = jax.random.key_data(event_key)
    except BaseException as failure:
        error = failure
    agree_process_phase(error, phase="evaluation context")
    assert context is not None and event_data is not None
    event_words = tuple(int(word) for word in np.asarray(collective_host(
        event_data, phase="event identity")))
    score_key = jax.random.fold_in(context.key, 0x53434F52)
    preview_key = jax.random.fold_in(context.key, 0x50524556)
    summaries: dict[str, object] = {}
    scores: dict[str, float] = {}
    previews: tuple[Artifact, ...] = ()
    source = iterator = None
    scored = records = 0
    uneven = False
    if batches is not None and (metrics or preview_enabled):
        try:
            error = None
            try:
                mesh = build_mesh() if mesh is None else mesh
                source = batches()
                iterator = iter(source)
            except BaseException as failure:
                error = failure
            agree_process_phase(error, phase="iterator construction")
            assert iterator is not None and mesh is not None
            while True:
                error = None
                batch = None
                try:
                    batch = next(iterator)
                except StopIteration:
                    pass
                except BaseException as failure:
                    error = failure
                available = agree_process_phase(
                    error, phase=f"iterator next batch {scored}", available=batch is not None)
                if available != jax.process_count():
                    uneven = available > 0
                    batch = None
                    break
                assert batch is not None
                error = None
                try:
                    batch = shard_batch(mesh, batch)
                    rows = next((leaf.shape[0] for leaf in jax.tree.leaves(batch) if leaf.ndim), None)
                    if rows is None:
                        raise ValueError("validation batch has no row-bearing array")
                    records += int(rows)
                except BaseException as failure:
                    error = failure
                agree_process_phase(error, phase=f"batch placement {scored}")
                produced = None
                if metrics:
                    error = None
                    try:
                        info = replace(context, key=jax.random.fold_in(score_key, scored))
                        produced = objective.evaluate(variables, batch, info)
                    except BaseException as failure:
                        error = failure
                    agree_process_phase(error, phase=f"scoring batch {scored}")
                    produced, home = collective_host((produced, batch), phase=f"scoring batch {scored}")
                    artifacts = (() if produced is None else produced
                                 if isinstance(produced, tuple) else (produced,))
                    for metric in metrics:
                        error = None
                        if root:
                            try:
                                contribution = metric(_pick(artifacts, metric.reads), home)
                                summaries[metric.name] = (
                                    metric.merge(summaries[metric.name], contribution)
                                    if metric.name in summaries else contribution)
                                del contribution
                            except BaseException as failure:
                                error = failure
                        agree_process_phase(error, phase=f"metric {metric.name} batch {scored}")
                    del home, artifacts
                if scored == 0 and preview_enabled:
                    error = None
                    produced_preview = None
                    try:
                        info = replace(context, key=preview_key)
                        produced_preview = objective.preview(variables, batch, info, scored=produced)
                    except BaseException as failure:
                        error = failure
                    agree_process_phase(error, phase="preview generation/decoding")
                    produced_preview = collective_host(produced_preview, phase="preview artifacts")
                    if root:
                        previews = (() if produced_preview is None else produced_preview
                                    if isinstance(produced_preview, tuple) else (produced_preview,))
                    produced_preview = None
                produced = batch = None
                scored += 1
                if not metrics:
                    break
            if scored:
                for metric in metrics:
                    error = None
                    if root:
                        try:
                            scores[f"{split}/{metric.name}"] = float(metric.finalize(summaries[metric.name]))
                        except BaseException as failure:
                            error = failure
                    agree_process_phase(error, phase=f"finalizing metric {metric.name}")
        finally:
            primary = sys.exception()
            error = None
            try:
                close = getattr(iterator if iterator is not None else source, "close", None)
                if close is not None:
                    close()
            except BaseException as failure:
                if primary is not None:
                    primary.add_note(f"Validation iterator cleanup failed: {failure!r}")
                else:
                    error = failure
            if primary is None:
                agree_process_phase(error, phase="iterator cleanup")
    elapsed = time.perf_counter() - started
    scores, elapsed = broadcast_from_process_zero((scores, elapsed))
    return Evaluation(event_step, split, scores, scored, records, uneven, event_words, elapsed, previews)
