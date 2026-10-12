"""A memory read's eager stages, run once over the whole source.

`_materialize` runs the stages a memory read holds before its first other
stage (`_Lowering.eager`): the record steps, token splits, packing and
windows, after filling the cache before them when there is one. It keeps
the accepted rows, each with the source position it came from, as a
`_Memory` source, which the Grain lowering then reads by index like any
other. Record steps, token splits, Grain's packers and cache reads run by
the Grain lowering's own definitions (`dew.data.lowering`); first-fit
packing and windows run by their indexed plans (`PackedWindows`,
`WindowRecords`), so the plan can fingerprint them. It counts what it
attempted, accepted and rejected, and records the derived order facts a
resume checks.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence

import grain.python as pygrain
import numpy as np

from dew.data.analysis import _analyzed, _identity, _name
from dew.data.dataset import Batch, LengthGroups, Records
from dew.data.lowering import _CachedOutputs, _fill, _Locality, _outputs, _packed, _Payload, _TokenSplit
from dew.data.mix import CorpusCounts
from dew.data.pipeline import (
    FlatMap,
    Group,
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
from dew.data.sources.base import Read, ShardManifest, Source
from dew.records import JSON, integer, json_value, record

_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _Memory:
    original: Source
    held: tuple[Batch | bytes, ...]
    measured: np.ndarray | None
    derived: Mapping[str, str]
    facts: Mapping[str, object] = dataclasses.field(default_factory=dict)
    identifiers: tuple[tuple[int, ...], ...] = ()
    inputs: int = 0
    ran: tuple[Stage, ...] = ()

    def describe(self) -> str:
        return self.original.describe()

    def snapshot(self) -> Mapping[str, object]:
        """The rows as a memory read made them: the source it read, the
        derived order facts and the steps it ran, by what they compute. The
        manifest records those steps' modules in its environment."""
        return {
            "materialized": self.original.snapshot(),
            "derived": self.derived,
            "steps": [_identity(stage).semantic() for stage in self.ran],
        }

    def supports(self, read: Read) -> bool:
        return read in ("memory", "indexed")

    def records(self, read: Read) -> tuple[Batch | bytes, ...]:
        return self.held

    def metadata(self) -> Mapping[str, object]:
        return {**self.original.metadata(), **self.facts, "accepted": len(self.held)}

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
    groups: LengthGroups | None = None,
    *,
    training: bool = True,
) -> Pipeline:
    groups = next((stage.groups for stage in pipeline.stages if isinstance(stage, Group)), groups)
    if isinstance(pipeline.source, Phases):
        raise ValueError("materialize each phase before constructing its global order")
    if isinstance(pipeline.source, (Mixture, Phases)):
        return _memory_mixture(pipeline, pipeline.source, counts, groups, training=training)
    source = pipeline.source
    stored = source.records("memory")
    counter = counts.setdefault(_name(pipeline), CorpusCounts(_name(pipeline)))
    if not isinstance(source, _Memory):
        counter.attempted = len(stored)
    lowering = _analyzed(pipeline, "memory", validation=not training)
    rows, consumed = _memory_inputs(pipeline, source, stored, counter, lowering.cache)
    return _memory_stages(
        pipeline, source, stored, rows, consumed, lowering.eager, counter, groups, training=training
    )


def _memory_inputs(
    pipeline: Pipeline, source: Source, stored: Records, counter: CorpusCounts, artifact_at: int | None
) -> tuple[list[tuple[tuple[int, ...], Record]], int]:
    rows: list[tuple[tuple[int, ...], Record]] = []
    consumed = 0
    if artifact_at is not None:
        filled, manifest, prefix = _fill(pipeline, source, artifact_at, "memory")
        cached = filled.mapped(pygrain.MapDataset.source(stored), manifest, source, transform=prefix)
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
    groups: LengthGroups | None,
    *,
    training: bool,
) -> Pipeline:
    derived: dict[str, str] = {}
    targets: Mapping[str, int] | None = None
    for stage in pipeline.stages[consumed:eager]:
        if isinstance(stage, TokenSplit):
            split = _TokenSplit(pygrain.MapDataset.source([value for _, value in rows]), stage)
            spans = enumerate(split.spans)
            rows = [((*rows[row][0], first, last), split[index]) for index, (row, first, last) in spans]
        elif isinstance(stage, Windows):
            from dew.data.steps.windows import WindowRecords

            derived[f"accepted_before_windows_{consumed}"] = fingerprint(
                [list(identifier) for identifier, _ in rows]
            )
            windows = WindowRecords(tuple(record(value, "window document") for _, value in rows), stage)
            derived["windows"] = fingerprint([windows.size, stage.seq_len, stage.stride])
            rows, targets = _targeted([((index,), windows[index]) for index in range(len(windows))], counter)
        elif isinstance(stage, Pack):
            from dew.data.tokens import PackedWindows

            derived[f"accepted_before_pack_{consumed}"] = fingerprint(
                [list(identifier) for identifier, _ in rows]
            )
            documents = tuple(_Payload()(record(value, "packing document")) for _, value in rows)
            sizes = np.asarray([len(value[stage.fields[0]]) for value in documents], np.int64)
            if stage.policy == "first-fit":
                packed = PackedWindows(
                    pygrain.MapDataset.source(documents), sizes, stage.seq_len, stage.bins, source.describe()
                )
                derived["packing"] = fingerprint([packed.order.tolist(), packed.starts.tolist()])
            else:
                # Only first-fit has an indexed plan; the other policies are Grain's
                # own packer, run once over the corpus as an indexed read streams it.
                packed = list(_packed(pygrain.MapDataset.source(documents).to_iter_dataset(), stage))
                derived["packing"] = fingerprint(sizes)
            rows, targets = _targeted([((index,), packed[index]) for index in range(len(packed))], counter)
        else:
            expanded: list[tuple[tuple[int, ...], Record]] = []
            for identifier, value in rows:
                for child, output in enumerate(_outputs(stage, identifier[0], value)):
                    if isinstance(output, Rejected):
                        counter.rejected[output.reason] += 1
                    else:
                        expanded.append(
                            ((*identifier, child) if isinstance(stage, FlatMap) else identifier, output)
                        )
            rows = expanded
        consumed += 1
    if targets is not None and targets["valid_targets"] == 0 and training:
        raise ValueError(f"corpus {_name(pipeline)} has zero valid target tokens")
    inputs = source.inputs if isinstance(source, _Memory) else len(stored)
    return _memory_order(pipeline, rows, consumed, source, inputs, counter, derived, groups, targets)


def _targeted(
    rows: list[tuple[tuple[int, ...], Record]], counter: CorpusCounts
) -> tuple[list[tuple[tuple[int, ...], Record]], Mapping[str, int]]:
    """The windows a pack or windows stage made that hold a valid target, and
    their count and targets, taken where the windows are made so later steps
    cannot change them."""
    from dew.data.steps.text import ValidTargets

    kept, total = [], 0
    for identifier, value in rows:
        count = ValidTargets()(record(value, "packed window"))
        if count:
            kept.append((identifier, value))
            total += count
        else:
            counter.rejected["zero_targets"] += 1
    return kept, {"windows": len(kept), "valid_targets": total}


def _memory_mixture(
    pipeline: Pipeline,
    mixture: Mixture,
    counts: dict[str, CorpusCounts],
    groups: LengthGroups | None,
    *,
    training: bool,
) -> Pipeline:
    children = tuple(_materialize(child, counts, groups, training=training) for child in mixture.pipelines)
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
    if len(kept) == 1:
        child = kept[0][0]
        return dataclasses.replace(child, stages=(*child.stages, *pipeline.stages))
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
    groups: LengthGroups | None,
    packed: Mapping[str, int] | None,
) -> Pipeline:
    rows = [
        (identifier, value if isinstance(value, bytes) else record(value, "materialized record"))
        for identifier, value in collected
    ]
    remaining = pipeline.stages[consumed:]
    lengths = (
        np.asarray([groups.length(record(value, "grouped record")) for _, value in rows], np.int64)
        if groups is not None
        else source.lengths()
        if consumed == 0
        else None
    )
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
