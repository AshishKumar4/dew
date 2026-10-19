"""A memory read's eager stages, run once over the whole source.

`_materialize` runs the stages a memory read holds before its first other
stage (`_Lowering.eager`): the record steps, token splits, packing and
windows, after filling the cache before them when there is one. It keeps
the accepted rows, each with the source position it came from, as a
`_Memory` source, which the Grain lowering then reads by index like any
other. Record steps, token splits, Grain's packers and cache reads run by
the Grain lowering's own definitions (`dew.data.lowering`); first-fit
packing and windows run by their indexed plans (`plan_windows`, read as
`PackedSource` reads it, and `IndexedWindows`), so the plan can
fingerprint them. It counts what it attempted, accepted and rejected, and
records the derived order facts a resume checks.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence

import grain.python as pygrain
import numpy as np

from dew.data.analysis import _analyzed, _identity, _name
from dew.data.dataset import Batch, Records
from dew.data.lowering import _CachedOutputs, _fill, _Locality, _outputs, _packed, _TokenSplit
from dew.data.mixing import CorpusCounts
from dew.data.pipeline import (
    FlatMap,
    Map,
    Mixture,
    Pack,
    Phases,
    Pipeline,
    Record,
    Rejected,
    Stage,
    TokenSplit,
    Windows,
)
from dew.data.plan import fingerprint
from dew.data.sources.base import Read, ShardManifest, Source, _uncounted
from dew.data.steps.generic import Fields
from dew.data.steps.text import CountTargets, ValidTargets, targets_changed
from dew.records import JSON, integer, json_value, record

_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _Memory:
    """Accepted rows and their facts, keeping measured targets through field projections."""

    original: Source
    held: tuple[Batch | bytes, ...]
    measured: np.ndarray | None
    derived: Mapping[str, str]
    facts: Mapping[str, object] = dataclasses.field(default_factory=dict)
    identifiers: tuple[tuple[int, ...], ...] = ()
    inputs: int = 0
    ran: tuple[Stage, ...] = ()
    target_counts: tuple[int, ...] | None = None
    documents: int | None = None
    """The most documents a window holds where `ran` packs them, which its
    batches declare (`Coordinates.max_segments`)."""

    def describe(self) -> str:
        return self.original.describe()

    def fingerprint(self) -> Mapping[str, object]:
        """The rows as a memory read made them: the source it read, the
        derived order facts and the steps it ran, by what they compute. The
        manifest records those steps' modules in its environment."""
        return {
            "materialized": self.original.fingerprint(),
            "derived": self.derived,
            "steps": [_identity(stage).semantic() for stage in self.ran],
        }

    def supports(self, read: Read) -> bool:
        return read in ("memory", "indexed")

    def records(self, read: Read) -> tuple[Batch | bytes, ...]:
        return self.held

    def metadata(self) -> Mapping[str, object]:
        """Its source's facts, the counts only where it ran no stage over the source's rows, then
        the counts it measured."""
        stated = self.original.metadata()
        return {**(_uncounted(stated) if self.ran else stated), **self.facts, "rows": len(self.held),
                "accepted": len(self.held)}

    def lengths(self) -> np.ndarray | None:
        return self.measured

    def locality(self) -> tuple[tuple[int, ...], bool]:
        if not isinstance(self.original, _Locality) or any(
            len(identifier) != 1 for identifier in self.identifiers
        ):
            raise ValueError("block shuffle needs an unchanged physical shard index")
        sizes, remote = self.original.locality()
        ends = np.cumsum(sizes)
        selected = np.bincount(
            [int(np.searchsorted(ends, identifier[0], side="right")) for identifier in self.identifiers],
            minlength=len(sizes),
        )
        return tuple(int(size) for size in selected), remote

    def manifest(self) -> ShardManifest:
        raise ValueError("materialized records are read by index")


def _materialize(
    pipeline: Pipeline,
    counts: dict[str, CorpusCounts],
    *,
    training: bool = True,
) -> Pipeline:
    if isinstance(pipeline.source, Phases):
        raise ValueError("materialize each phase before constructing its global order")
    if isinstance(pipeline.source, (Mixture, Phases)):
        return _memory_mixture(pipeline, pipeline.source, counts, training=training)
    source = pipeline.source
    stored = source.records("memory")
    counter = counts.setdefault(_name(pipeline), CorpusCounts(_name(pipeline)))
    if not isinstance(source, _Memory):
        counter.attempted = len(stored)
    lowering = _analyzed(pipeline, "memory", validation=not training)
    rows, consumed = _memory_inputs(pipeline, source, stored, counter, lowering.cache)
    return _memory_stages(
        pipeline, source, stored, rows, consumed, lowering.eager, counter, training=training
    )


def _memory_inputs(
    pipeline: Pipeline, source: Source, stored: Records, counter: CorpusCounts, artifact_at: int | None
) -> tuple[list[tuple[tuple[int, ...], Record]], int]:
    rows: list[tuple[tuple[int, ...], Record]] = []
    consumed = 0
    if artifact_at is not None:
        filled, manifest, prefix = _fill(pipeline, source, artifact_at, "memory")
        cached = filled.mapped(pygrain.MapDataset.source(stored), manifest, transform=prefix)
        try:
            for index in range(len(cached)):
                outputs = cached[index]
                if not isinstance(outputs, tuple):
                    raise TypeError("cached input must return its tuple of expansion outputs")
                for child, value in enumerate(_CachedOutputs().flat_map(outputs)):
                    if isinstance(value, Rejected):
                        counter.rejected[value.reason] += 1
                    else:
                        rows.append(((index, child), value))
            cached.finish()
        finally:
            cached.close()
        consumed = artifact_at + 1
    else:
        for index in range(len(stored)):
            value = stored[index]
            if value is None:
                counter.rejected["missing"] += 1
            else:
                identifier = source.identifiers[index] if isinstance(source, _Memory) else (index,)
                rows.append((identifier, value))
    return rows, consumed


def _memory_stages(
    pipeline: Pipeline,
    source: Source,
    stored: Records,
    rows: list[tuple[tuple[int, ...], Record]],
    consumed: int,
    eager: int,
    counter: CorpusCounts,
    *,
    training: bool,
) -> Pipeline:
    derived: dict[str, str] = {}
    targets: Mapping[str, int] | None = None
    documents = source.documents if isinstance(source, _Memory) else None
    counted: dict[tuple[int, ...], int] | None = (
        dict(zip(source.identifiers, source.target_counts, strict=True))
        if isinstance(source, _Memory) and source.target_counts is not None
        else None
    )
    for stage in pipeline.stages[consumed:eager]:
        if isinstance(stage, TokenSplit):
            split = _TokenSplit(pygrain.MapDataset.source([value for _, value in rows]), stage)
            if counted is not None:
                counted = {(*rows[row][0], first, last): counted[rows[row][0]]
                           for row, first, last in split.spans}
            spans = enumerate(split.spans)
            rows = [((*rows[row][0], first, last), split[index]) for index, (row, first, last) in spans]
        elif isinstance(stage, Windows):
            from dew.data.steps.windows import IndexedWindows

            derived[f"accepted_before_windows_{consumed}"] = fingerprint(
                [list(identifier) for identifier, _ in rows]
            )
            windows = IndexedWindows.from_records(
                tuple(record(value, "window document") for _, value in rows), stage
            )
            derived["windows"] = fingerprint([windows.size, stage.seq_len, stage.stride])
            rows, targets = _targeted([((index,), windows[index]) for index in range(len(windows))], counter,
                                      stage.fields[0])
        elif isinstance(stage, Pack):
            from dew.data.packing import _Held, _PackedRows, refuse_overlong

            derived[f"accepted_before_pack_{consumed}"] = fingerprint(
                [list(identifier) for identifier, _ in rows]
            )
            kept = Fields(stage.packed)
            documents = tuple(kept.map(record(value, "packing document")) for _, value in rows)
            sizes = np.asarray([len(value[stage.fields[0]]) for value in documents], np.int64)
            refuse_overlong(sizes, stage, source.describe())
            if stage.policy == "first-fit" and not stage.along:
                # The plan and windows reader of an indexed read of documents that state
                # their lengths (`PackedSource.planned`): it reads the windows that hold a
                # target, and the plan counts the rest as zero_targets.
                plan = _Held(_Memory(source, documents, sizes, {}), stage)
                planned = plan.arrays
                derived["packing"] = fingerprint([planned["order"].tolist(), planned["starts"].tolist()])
                packed = _PackedRows(documents, plan)
                rows = [((plan.at("accepted", index),), packed[index]) for index in range(len(packed))]
                counter.rejected.update(["zero_targets"] * plan.at("totals", 1))
                targets = {"valid_targets": plan.at("totals", 0)}
            else:
                # Only first-fit of tokens alone has an indexed plan; the rest is Grain's
                # own packer, run once over the corpus as an indexed read streams it.
                packed = list(_packed(pygrain.MapDataset.source(documents).to_iter_dataset(), stage))
                derived["packing"] = fingerprint(sizes)
                rows, targets = _targeted([((index,), value) for index, value in enumerate(packed)], counter,
                                          stage.fields[0])
            ids = f"{stage.fields[0]}_segment_ids"
            documents = max((int(np.asarray(record(value, "packed window")[ids]).max(initial=0))
                             for _, value in rows), default=0)
        else:
            expanded: list[tuple[tuple[int, ...], Record]] = []
            expanded_counts: dict[tuple[int, ...], int] = {}
            counts_targets = isinstance(stage, Map) and isinstance(stage.step, CountTargets)
            for identifier, value in rows:
                outputs = _outputs(stage, identifier[0], value, counts=counter)
                for child, output in enumerate(outputs):
                    if isinstance(output, Rejected):
                        counter.rejected[output.reason] += 1
                    else:
                        output_id = (*identifier, child) if isinstance(stage, FlatMap) else identifier
                        expanded.append((output_id, output))
                        if counts_targets:
                            expanded_counts[output_id] = ValidTargets()(record(output, "counted row"))
                        elif counted is not None:
                            # A row's count holds while the step left the fields it was counted from;
                            # where it changed one, the row's own targets, as `recounted` carries them.
                            given, made = record(value, "counted row"), record(output, "counted row")
                            expanded_counts[output_id] = (ValidTargets()(made) if "text" in made
                                                          and targets_changed(given, made)
                                                          else counted[identifier])
            rows = expanded
            if counts_targets or counted is not None:
                counted = expanded_counts
        consumed += 1
    if counted is not None:
        targets = {"valid_targets": sum(counted.values())}
    if targets is not None and targets["valid_targets"] == 0 and training:
        raise ValueError(f"corpus {_name(pipeline)} has zero valid target tokens")
    inputs = source.inputs if isinstance(source, _Memory) else len(stored)
    return _memory_order(pipeline, rows, consumed, source, inputs, counter, derived, targets, counted,
                         documents)


def _targeted(
    rows: list[tuple[tuple[int, ...], Record]], counter: CorpusCounts, field: str
) -> tuple[list[tuple[tuple[int, ...], Record]], Mapping[str, int]]:
    """The windows a pack or windows stage made that hold a valid target of
    their token `field`, and their counts before record steps rename or remove
    their token fields."""

    kept, total = [], 0
    for identifier, value in rows:
        count = ValidTargets(field)(record(value, "packed window"))
        if count:
            kept.append((identifier, value))
            total += count
        else:
            counter.rejected["zero_targets"] += 1
    return kept, {"valid_targets": total}


def _measured_targets(child: Pipeline, mixture: Mixture) -> Pipeline:
    """A corpus a mixture weighs by tokens (`Mixture.tokens`), its valid targets measured where the
    memory read ran steps over its source's rows, whose stated counts then describe other rows
    (`_Memory.metadata`): the read holds every row the mixture reads, so it counts their targets
    by the mixture's own rule, and the mixture weighs the corpus by them as by a source's stated
    counts."""
    source = child.source
    if (mixture.tokens is None or not isinstance(source, _Memory) or "valid_targets" in source.metadata()
            or any(isinstance(value, bytes) for value in source.held)):
        return child
    counted = sum(mixture.tokens(record(value, "mixed row")) for value in source.held)
    return dataclasses.replace(child, source=dataclasses.replace(source, facts={**source.facts,
                                                                              "valid_targets": counted}))


def _memory_mixture(
    pipeline: Pipeline,
    mixture: Mixture,
    counts: dict[str, CorpusCounts],
    *,
    training: bool,
) -> Pipeline:
    children = tuple(_measured_targets(_materialize(child, counts, training=training), mixture)
                     for child in mixture.pipelines)
    kept = tuple(
        (child, weight)
        for child, weight in zip(children, mixture.weights, strict=True)
        if isinstance(child.source, (Mixture, Phases)) or len(child.source.records("memory"))
    )
    omitted = [_name(child) for child in children if all(child is not accepted for accepted, _ in kept)]
    if omitted:
        _log.warning("left out of the mixture: %s (zero accepted rows)", ", ".join(omitted))
    if not kept:
        if not training:
            return dataclasses.replace(children[0], stages=(*children[0].stages, *pipeline.stages))
        raise ValueError(
            f"no accepted row of {', '.join(_name(child) for child in children)}; raise the row budgets"
        )
    children, weights = zip(*kept, strict=True)
    return dataclasses.replace(
        pipeline, source=dataclasses.replace(mixture, pipelines=tuple(children), weights=tuple(weights))
    )


def _memory_order(
    pipeline: Pipeline,
    collected: Sequence[tuple[tuple[int, ...], Record]],
    consumed: int,
    source: Source,
    inputs: int,
    counter: CorpusCounts,
    derived: dict[str, str],
    packed: Mapping[str, int] | None,
    counted: Mapping[tuple[int, ...], int] | None,
    documents: int | None,
) -> Pipeline:
    rows = [
        (identifier, value if isinstance(value, bytes) else record(value, "materialized record"))
        for identifier, value in collected
    ]
    remaining = pipeline.stages[consumed:]
    lengths = source.lengths() if consumed == 0 else None
    fanouts = np.zeros(inputs, np.int64)
    for identifier, _ in rows:
        if identifier[0] < len(fanouts):
            fanouts[identifier[0]] += 1
    derived["fanouts"] = fingerprint(fanouts)
    if lengths is not None:
        derived["lengths"] = fingerprint(lengths)
    counter.accepted = len(rows)
    counter.truncated = sum(
        bool(value.get("_dew_truncated", False)) for _, value in rows if not isinstance(value, bytes)
    )
    facts: Batch = _measured_bounds(
        [record(value, "measured row") for _, value in rows if not isinstance(value, bytes)]
    )
    if packed is not None:
        facts = {**facts, **packed}
        derived["valid_targets"] = fingerprint(facts)
    derived["accepted_ids"] = fingerprint([list(identifier) for identifier, _ in rows])
    return dataclasses.replace(
        pipeline,
        source=_Memory(
            source,
            tuple(value for _, value in rows),
            lengths,
            derived,
            facts,
            tuple(identifier for identifier, _ in rows),
            inputs,
            pipeline.stages[:consumed],
            None if counted is None else tuple(counted[identifier] for identifier, _ in rows),
            documents,
        ),
        stages=remaining,
    )


def _measured_bounds(rows: Sequence[Batch]) -> Mapping[str, object]:
    bounds: dict[str, JSON] = {}

    def merge(target: dict[str, JSON], given: Mapping[str, object]) -> None:
        for key, value in given.items():
            if isinstance(value, Mapping):
                previous = {
                    name: json_value(datum, name) for name, datum in record(target.get(key, {}), key).items()
                }
                merge(previous, value)
                target[key] = previous
            else:
                target[key] = max(integer(value, key), integer(target.get(key, 0), key))

    for row in rows:
        merge(bounds, record(row.get("_dew_bounds", {}), "record bounds"))
    return bounds
