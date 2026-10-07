"""Coordinated evaluation of trained variables, independent of optimization."""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh

from dew.artifacts import Artifact, Artifacts
from dew.coordination import agree_process_phase, agreed, broadcast_from_process_zero, collective_host
from dew.data.dataset import Closeable, DataPartition, Reader, rows_of
from dew.objectives.base import VALID_ROWS, Batch, Effects, Loss, Metric, Objective, Step, Variables

from .distributed import MeshSpec, shard_batch


@dataclass(frozen=True)
class Evaluation:
    """The result of one evaluation, with its previews kept on process 0.

    Every process holds the same `scores`, row counts and RNG identity
    (`event_key`). Each score's name starts with the split, as in `val/loss`.
    `coordinated_batches` is the number of batches every process scored, and
    `records` the number of real rows in them, which covers every record of the
    split once. `elapsed_seconds` is process 0's wall time, including closing the
    batch iterator. The metric accumulators and the validation batches are not
    kept.
    """

    step: int
    split: str
    scores: dict[str, float]
    coordinated_batches: int
    records: int
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
                "evaluation/records": float(self.records)}

    @classmethod
    def run(cls, objective: Objective[Loss, Effects], variables: Variables,
            batches: Reader | None, *,
            key: int | jax.Array, metrics: Sequence[Metric] = (),
            step: int | jax.Array = 0, averaged: Variables | None = None,
            preview: bool = False, mesh: Mesh | None = None, split: str = "val",
            schedule_step: int | jax.Array | None = None, loss: bool = False,
            training: Mapping[str, jax.Array] | None = None) -> Evaluation:
        """Evaluate `variables` on the batches every process can read, without an optimizer or a tracker.

        Every process calls this with the same objective, metrics and numerical
        settings. `batches` is called with this process's share of the split
        (`DataPartition.of(mesh)`) and returns a fresh iterator, which this call
        opens and closes; you can pass `Dataset.val` directly. Every batch holds
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
        only the work training accepted.

        The scores are broadcast to every process. Previews stay on process 0, and
        there is at most one objective preview, however long the validation split
        is. Reporting the result is up to the caller. In a process pool, the callers
        must agree on whether reporting failed before they enter their next
        collective, and a device failure in flight still requires ending the
        distributed runtime.
        """
        from dew.nn.inputs import request_key
        key = request_key(key)
        started = time.perf_counter()
        root = jax.process_index() == 0
        _agree_configuration(metrics, batches, split, loss=loss, training=training is not None)
        preview_enabled = bool(broadcast_from_process_zero(root and preview))
        event_step, context, event_words, training_scores = _event(
            key, step, schedule_step, averaged, training)
        score_key = _folded(context.key, 0x53434F52)
        preview_key = _folded(context.key, 0x50524556)
        scores: dict[str, float] = {}
        previews: tuple[Artifact, ...] = ()
        scored = records = 0
        if batches is not None and (metrics or preview_enabled or loss):
            scores, previews, scored, records = _score_split(
                objective, variables, batches, context, mesh, metrics=metrics, split=split,
                root=root, preview_enabled=preview_enabled, score_key=score_key,
                preview_key=preview_key, loss=loss)
        scores.update(training_scores)
        elapsed = time.perf_counter() - started
        scores, elapsed = broadcast_from_process_zero((scores, elapsed))
        return cls(event_step, split, scores, scored, records, event_words, elapsed, previews)


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
    metrics: Sequence[Metric], batches, split: str, *, loss: bool = False, training: bool = False
) -> None:
    """Check this rank's evaluation settings, then agree they match root's.

    Ranks that disagree about the split, the metrics or whether there is a
    validation stream would walk different phases below, and hang at a
    collective one of them never reaches.
    """
    def checked() -> list[bool | str | list[list[str]]]:
        names = [metric.name for metric in metrics]
        if len(names) != len(set(names)):
            raise ValueError("evaluation metric names must be unique")
        if not split or "/" in split:
            raise ValueError("evaluation split must be a nonempty name without '/'")
        # Each metric's name and the module and qualified name of the artifact it reads.
        return [batches is not None, split,
                [[metric.name, metric.reads.__module__, metric.reads.__qualname__] for metric in metrics]]

    configuration = [agreed("configuration", checked), loss, training]
    root_configuration = broadcast_from_process_zero(configuration)
    error = None if configuration == root_configuration else ValueError(
        "validation availability, split and ordered metric names/types must agree across ranks")
    agree_process_phase(error, phase="configuration agreement")


def _event(key: jax.Array, step: int | jax.Array, schedule_step: int | jax.Array | None,
           averaged: Variables | None, training: Mapping[str, jax.Array] | None = None
           ) -> tuple[int, Step, tuple[int, ...], dict[str, float]]:
    """Draw the event's clocks and RNG, and the identity every rank reports.

    The clocks come home through a collective, so a rank whose own step
    differs uses root's. The key folds in 'EVAL' and that step, so an
    evaluation never draws what a training step at the same clock drew.
    """
    step_home, schedule_home, reports = collective_host(
        (step, step if schedule_step is None else schedule_step, dict(training or {})),
        phase="evaluation clocks",
    )

    def event() -> tuple[int, Step, jax.Array]:
        at = int(step_home)
        event_key = _folded(_folded(key, 0x4556414C), at)
        return (at, Step(step=jax.device_put(schedule_home), key=event_key, ema=averaged),
                jax.random.key_data(event_key))

    event_step, context, event_data = agreed("evaluation context", event)
    event_words = tuple(int(word) for word in np.asarray(collective_host(
        event_data, phase="event identity")))
    return event_step, context, event_words, {name: float(value) for name, value in reports.items()}


def _folded(key: jax.Array, word: int) -> jax.Array:
    """`jax.random.fold_in(key, word)`, the host integer placed beside the
    key by name rather than moved there implicitly by the fold: on the key's
    devices, or uncommitted with an uncommitted key."""
    return jax.random.fold_in(key, jax.device_put(np.uint32(word), key.sharding if key.committed else None))


def _score_split(objective: Objective[Loss, Effects], variables: Variables, batches,
                 context: Step, mesh: Mesh | None, *, metrics: Sequence[Metric],
                 split: str, root: bool, preview_enabled: bool,
                 score_key: jax.Array, preview_key: jax.Array, loss: bool = False,
                 ) -> tuple[dict[str, float], tuple[Artifact, ...], int, int]:
    """Score every record of a validation split.

    Returns the finalized scores, root's previews, and the two counts every
    rank agrees on: the batches scored and the real records they held.

    Every rank walks these phases in the same order. The pass ends when no
    rank has a batch left; until then a rank that has drained scores a copy
    of its last batch whose rows are all repeats (`_covered`), so it meets
    its peers at every collective.
    """
    summaries = _Accumulators()
    loss_stats = None
    scores: dict[str, float] = {}
    previews: tuple[Artifact, ...] = ()
    source = iterator = None
    last = None
    scored = records = 0
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
        loss_variables = (context.ema if context.ema is not None and not objective._ema_is_reference
                          else variables)
        while True:
            held, available = _next_batch(iterator, scored)
            if not available:
                break
            held, last, batch = _covered(mesh, held, last, scored)
            # This rank's own real rows, counted on the host; the pool's count
            # is summed once after the split.
            records += int(np.sum(held[VALID_ROWS]))
            produced = None
            if metrics or loss:
                # The objective scores under the mesh, as the step trains
                # under it: the model's placements and its sequence and stage
                # splits read it. A preview decodes, which neither split does.
                with jax.set_mesh(mesh):
                    statistics, produced = agreed(
                        f"scoring batch {scored}",
                        lambda batch=batch, scored=scored: _dispatched(
                            objective, variables, loss_variables, batch, context, scored,
                            loss=loss, evaluate=bool(metrics), score_key=score_key))
                    if statistics is not None:
                        # The statistics stay on the devices, summed there in
                        # batch order, and come home once after the split.
                        loss_stats = statistics if loss_stats is None else jax.tree.map(
                            lambda total, value: total + value, loss_stats, statistics)
                    if metrics:
                        produced = _merged(produced, batch, held, scored, metrics=metrics,
                                           summaries=summaries, root=root)
            if scored == 0 and preview_enabled:
                previews = _previewed(objective, variables, batch, context,
                                      preview_key=preview_key, scored=produced, root=root)
            produced = batch = None
            scored += 1
            if not metrics and not loss:
                break
        # Processes on one data share (a replicated stage or sequence axis)
        # read the same rows, so the pool counts each share once.
        shares = np.asarray(multihost_utils.process_allgather(
            np.asarray([DataPartition.of(mesh).index, records], np.int64))).reshape(-1, 2)
        records = int(sum(dict(shares.tolist()).values()))
        if scored:
            scores = _finalized(metrics, summaries, split=split, root=root)
            if loss_stats is not None:
                loss_stats = collective_host(loss_stats, phase="validation loss")
            if loss_stats is not None and f'{split}/loss' not in scores:
                value, valid = jax.device_get(objective._validation_reduction(jax.device_put(loss_stats)))
                if not bool(valid) or not np.isfinite(float(value)):
                    raise ValueError("validation loss has no finite statistical support")
                scores[f'{split}/loss'] = float(value)
    finally:
        _close_source(iterator if iterator is not None else source)
    return scores, previews, scored, records


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


def _covered(mesh: Mesh, batch: Batch | None, last: Batch | None, index: int
             ) -> tuple[Batch, Batch, Batch]:
    """The batch this rank scores at `index`, the one it keeps to cover the
    next, and the first placed on the mesh, under one agreement.

    Its own batch carries `VALID_ROWS`, every row real where the reader
    marked none. A rank whose share has run out while a peer's has not
    scores a copy of its last batch with every row a repeat.
    """
    def cover() -> tuple[Batch, Batch, Batch]:
        if batch is not None:
            held = batch if VALID_ROWS in batch else {**batch, VALID_ROWS: np.ones(rows_of(batch), bool)}
            kept = held
        elif last is None:
            raise ValueError(
                f"process {jax.process_index()}'s reader yields no batch while a peer's yields one; "
                "Dew's readers give an empty share one batch whose rows are all repeats "
                "(`VALID_ROWS` False), and a reader of your own has to as well")
        else:
            held, kept = {**last, VALID_ROWS: np.zeros(rows_of(last), bool)}, last
        return held, kept, shard_batch(mesh, held)

    return agreed(f"batch cover and placement {index}", cover)


def _cut(leaf, keep: np.ndarray):
    """`leaf` without the rows `keep` marks as repeats, when its leading axis
    is the batch's rows: an array, or a per-row tuple of captions or texts."""
    if isinstance(leaf, tuple) and len(leaf) == keep.size:
        return tuple(text for text, kept in zip(leaf, keep, strict=True) if kept)
    array = np.asarray(leaf) if isinstance(leaf, np.ndarray | jax.Array) else None
    return array[keep] if array is not None and array.ndim and array.shape[0] == keep.size else leaf


def _real_artifact[A: Artifact](artifact: A, keep: np.ndarray) -> A:
    """`artifact` with every per-row field cut to the real rows."""
    return dataclasses.replace(artifact, **{field.name: _cut(getattr(artifact, field.name), keep)
                                            for field in dataclasses.fields(artifact)})


def _real_batch(batch: Batch, keep: np.ndarray) -> Batch:
    """`batch` cut to the real rows, without `VALID_ROWS`."""
    return jax.tree.map(lambda leaf: _cut(leaf, keep),
                        {name: leaf for name, leaf in batch.items() if name != VALID_ROWS})


def _dispatched(objective: Objective[Loss, Effects], variables: Variables, loss_variables: Variables,
                batch: Batch, context: Step, index: int, *, loss: bool, evaluate: bool,
                score_key: jax.Array):
    """One batch's device work, dispatched under one key: the validation-loss
    statistics where `loss`, and what the objective produces where `evaluate`."""
    keyed = replace(context, key=_folded(score_key, index))
    statistics = objective._validation_loss(loss_variables, batch, keyed) if loss else None
    return statistics, objective.evaluate(variables, batch, keyed) if evaluate else None


def _as_placed(leaf) -> np.ndarray:
    """`leaf` as placing it and reading it back gives it: in JAX's canonical
    dtype, where float64 is float32 unless x64 is on."""
    host = np.asarray(jax.device_get(leaf))
    return host.astype(jax.dtypes.canonicalize_dtype(host.dtype), copy=False)


def _merged(produced, batch: Batch, held: Batch, index: int, *, metrics: Sequence[Metric],
            summaries: _Accumulators, root: bool):
    """Merge one batch into every metric's accumulator on root, returning what
    the objective produced, hosted.

    The artifacts come home, and so does the batch the metrics read: on one
    process that is the host batch it was placed from, as the devices hold
    it, so nothing copies back; a batch spread over processes comes home
    with the artifacts. One agreement covers every metric, so a failure on
    root stops its peers there.
    """
    if all(leaf.is_fully_addressable for leaf in jax.tree.leaves(batch) if isinstance(leaf, jax.Array)):
        produced = collective_host(produced, phase=f"scoring batch {index}")
        home = jax.tree.map(_as_placed, held)
    else:
        produced, home = collective_host((produced, batch), phase=f"scoring batch {index}")

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
