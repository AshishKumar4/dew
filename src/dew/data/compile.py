"""Compile the same pipeline to an elastic order or an exact per-share reader."""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import math
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Literal, Protocol, runtime_checkable

import grain.experimental as gx
import grain.python as pygrain
import numpy as np

from dew.data.dataset import (
    Batch,
    Checkpointable,
    Closeable,
    DataPartition,
    Dataset,
    FilledBatch,
    Forwarding,
    GlobalStream,
    GlobalReader,
    PhasedStream,
    LengthGroups,
    Loading,
    Position,
    Reader,
    Records,
    _Grouped,
    _WorkerBatches,
    weight_counts,
    rows_of,
)
from dew.data.mix import Accepted, CorpusCounts, PrimedMix, SearchBudget, TokenDeficit, primed_maps, state_counts, token_weights
from dew.data.pipeline import (
    Artifact,
    BatchStage,
    Configured,
    Expansion,
    Filter,
    FlatMap,
    Group,
    Geometry,
    HostStep,
    Interleave,
    Limit,
    Map,
    MapBatch,
    MapWithIndex,
    Mixture,
    ObjectiveData,
    Pack,
    Phases,
    Pipeline,
    Prefetch,
    RandomMap,
    Rebatch,
    Record,
    Rejected,
    Repeat,
    ShardByHost,
    Shuffle,
    SingleReader,
    Slice,
    Stage,
    TokenSplit,
    TokenBudget,
    TokenGeometry,
    Volatile,
    Windows,
    Workers,
    _rejects,
)
from dew.data.plan import Manifest, Plan, StepIdentity, fingerprint
from dew.data.sources.base import Read, ShardManifest, Source
from dew.records import JSON, integer, json_value, record

type Rows = pygrain.MapDataset | pygrain.IterDataset
_log = logging.getLogger(__name__)
_DEFAULT_LOADING = Loading()
_DEFAULT_BUDGET = SearchBudget()


@runtime_checkable
class _Provisionable(Protocol):
    def provision(self) -> Source: ...


@runtime_checkable
class _Finishable(Protocol):
    def finish(self) -> None: ...


@runtime_checkable
class _Locality(Protocol):
    def locality(self) -> tuple[tuple[int, ...], bool]: ...


@runtime_checkable
class _OrderFacts(Protocol):
    def order_fingerprints(self) -> Mapping[str, str]: ...


def _leaves(pipeline: Pipeline) -> Iterator[Pipeline]:
    if isinstance(pipeline.source, (Mixture, Phases)):
        for child in pipeline.source.pipelines:
            yield from _leaves(child)
    else:
        yield pipeline


def _steps(pipeline: Pipeline) -> Iterator[Stage]:
    if isinstance(pipeline.source, (Mixture, Phases)):
        for child in pipeline.source.pipelines:
            yield from _steps(child)
    yield from pipeline.stages


def _name(pipeline: Pipeline) -> str:
    if pipeline.name is not None:
        return pipeline.name
    return "mixture" if isinstance(pipeline.source, (Mixture, Phases)) else pipeline.source.describe()


def _identity(stage: Stage, *, cache: bool = False) -> StepIdentity:
    if isinstance(stage, Artifact):
        return StepIdentity(f"dew.data.cache.{type(stage.cache).__name__}", stage.cache.url, exact=True)
    if isinstance(stage, Configured):
        return StepIdentity.of(stage.factory, cache=cache)
    if isinstance(stage, (Map, RandomMap, MapWithIndex, FlatMap, MapBatch)):
        identity = StepIdentity.of(stage.step, stage.identity, cache=cache)
        if isinstance(stage, (Map, RandomMap, FlatMap)) and stage.configured:
            identity = dataclasses.replace(identity, config=json.dumps(
                {"step": identity.config, "configured": stage.configured}, sort_keys=True))
        return identity
    if isinstance(stage, Filter):
        return StepIdentity.of(stage.predicate, stage.identity, cache=cache)
    return StepIdentity.of(stage)


def _cache_manifest(source: Source, prefix: Sequence[Stage], read: Read) -> Manifest:
    return Manifest.of(source.snapshot(), tuple(_identity(stage, cache=True) for stage in prefix),
                       fill={"read": read, "row_ids": "sharded" if read == "streamed" else "indexed"})


def _cache_references(pipeline: Pipeline, read: Read) -> tuple[str, ...]:
    references = []
    for leaf in _leaves(pipeline):
        if isinstance(leaf.source, (Mixture, Phases)):
            continue
        for index, stage in enumerate(leaf.stages):
            if isinstance(stage, Artifact):
                references.append(stage.cache.reference(_cache_manifest(leaf.source, leaf.stages[:index], read)))
    return tuple(references)


def _promoted(pipeline: Pipeline) -> tuple[Pipeline, tuple[str, ...]]:
    from dew.data.cache import Generation

    source = pipeline.source
    references: tuple[str, ...] = ()
    if isinstance(source, (Mixture, Phases)):
        branches = [_promoted(child) for child in source.pipelines]
        source = dataclasses.replace(source, pipelines=tuple(child for child, _ in branches))
        references = tuple(reference for _, held in branches for reference in held)
    for index, stage in enumerate(pipeline.stages):
        if isinstance(stage, Artifact) and isinstance(stage.cache, Generation):
            from dew.data.packing import PackedSource, plan_name

            suffix = pipeline.stages[index + 1:]
            promoted: Source = stage.cache.source()
            if suffix and isinstance(suffix[0], Pack):
                pack = suffix[0]
                plans = record(stage.cache.manifest.get("plan", {}), "published plans")
                if plan_name(pack, "accepted") in plans:
                    promoted = PackedSource(promoted, stage.cache, pack)
                    suffix = suffix[1:]
            return (dataclasses.replace(pipeline, source=promoted, stages=suffix),
                    (*references, f"generation {stage.cache.url}"))
    return dataclasses.replace(pipeline, source=source), references


def _check(pipeline: Pipeline, read: Read, *, validation: bool = False) -> None:
    if isinstance(pipeline.source, (Mixture, Phases)):
        for child in pipeline.source.pipelines:
            _check(child, read, validation=validation)
    ancestry = (
        tuple(stage for child in pipeline.source.pipelines for stage in _steps(child))
        if isinstance(pipeline.source, (Mixture, Phases))
        else ()
    )
    random = next((_identity(stage).name for stage in ancestry if isinstance(stage, RandomMap)), None)
    indexed, batched, workers, host = read != "streamed", False, False, False
    canonical = True
    prefix = list(ancestry)
    for stage in pipeline.stages:
        name = _name(pipeline)
        if isinstance(stage, TokenSplit) and read != "memory":
            raise ValueError(f"token holdout in {name} needs read='memory' or a producer-published token split")
        if isinstance(stage, RandomMap) or isinstance(stage, Configured) and stage.random:
            random = _identity(stage).name
            if validation and not canonical:
                raise ValueError(f"validation random step {random} must use the canonical record index")
        if isinstance(stage, (Shuffle, Repeat, Slice, Filter, FlatMap, Pack, Windows, Group, Limit)):
            canonical = False
        if isinstance(stage, (Map, MapWithIndex)) and stage.rejects:
            canonical = False
        if isinstance(stage, Artifact):
            if random is not None:
                raise ValueError(
                    f"cache prefix in {name} contains random step {random}; cache before stochastic steps"
                )
            for earlier in prefix:
                _identity(earlier, cache=True)
        if isinstance(stage, Shuffle) and stage.window is None and not indexed:
            raise ValueError(f"global shuffle in {name} needs an indexed pipeline; give window=")
        if isinstance(stage, (Interleave, Prefetch, Rebatch, Limit)) or (
            isinstance(stage, (Filter, FlatMap, Pack, Windows)) and (read != "memory" or random is not None)
        ):
            indexed = False
        if isinstance(stage, (Map, MapWithIndex, RandomMap)) and stage.rejects and read != "memory":
            indexed = False
        if isinstance(stage, (BatchStage, TokenBudget)):
            if batched:
                raise ValueError(f"pipeline {name} is already batched; use rebatch")
            batched, indexed = True, False
        if isinstance(stage, Workers):
            if workers:
                raise ValueError(f"pipeline {name} contains workers more than once")
            if host:
                raise ValueError(f"workers in {name} must precede map_batch")
            workers, indexed = True, False
        if isinstance(stage, MapBatch):
            if not batched:
                raise ValueError(f"map_batch in {name} must follow batch")
            host = True
        elif host:
            raise ValueError(f"{type(stage).__name__} in {name} follows the host map_batch stage")
        prefix.append(stage)


def _capabilities(pipelines: Sequence[Pipeline], read: Read, provision: bool) -> None:
    missing: list[Source] = []
    for pipeline in pipelines:
        for leaf in _leaves(pipeline):
            source = leaf.source
            if not isinstance(source, (Mixture, Phases)) and not source.supports(read):
                missing.append(source)
    if provision:
        for source in missing:
            if isinstance(source, _Provisionable):
                source.provision()
    refused = [source.describe() for source in missing if not source.supports(read)]
    if refused:
        raise ValueError(
            f"read={read!r} is unavailable for {', '.join(refused)}; "
            "provision the named artifact with dew data provision or provision=True"
        )


def _tokenizer_contract(pipelines: Sequence[Pipeline]) -> None:
    declared: dict[str, JSON] = {}
    for pipeline in pipelines:
        for leaf in _leaves(pipeline):
            if isinstance(leaf.source, (Mixture, Phases)):
                continue
            metadata = leaf.source.metadata()
            for key in ("tokenizer", "vocab_size", "eos_id", "tokenizer_revision"):
                if key not in metadata:
                    continue
                value = json_value(metadata[key], key)
                if key in declared and value != declared[key]:
                    raise ValueError(f"corpus {_name(leaf)} has {key}={value!r}, expected {declared[key]!r}; "
                                     "mixed corpora must share the tokenizer contract")
                declared[key] = value


def _configured(pipeline: Pipeline, shared: Mapping[str, object], read: Read,
                 scope: Literal["source", "run"] | None = None) -> Pipeline:
    source = pipeline.source
    if isinstance(source, (Mixture, Phases)):
        children = tuple(_configured(child, shared, read, scope) for child in source.pipelines)
        source = dataclasses.replace(source, pipelines=children)
        local = shared
    else:
        local = source.metadata()
    stages: list[Stage] = []
    for stage in pipeline.stages:
        if not isinstance(stage, Configured) or scope is not None and stage.scope != scope:
            stages.append(stage)
            continue
        metadata = shared if stage.scope == "run" else local
        missing = sorted(set(stage.needs) - metadata.keys())
        if missing:
            raise ValueError(f"{_name(pipeline)} step {_identity(stage).name} needs metadata {missing}; "
                             "provide producer metadata, recipe fields or read='memory'")
        binding = {"factory": dataclasses.asdict(_identity(stage)), "scope": stage.scope,
                   "values": {key: metadata[key] for key in stage.needs}}
        resolved = stage.factory(metadata)
        if isinstance(resolved, pygrain.RandomMapTransform) != stage.random:
            raise ValueError(f"Configured {_identity(stage).name} random={stage.random} disagrees with its step")
        if isinstance(resolved, Expansion):
            stages.append(FlatMap(resolved, configured=binding))
        elif isinstance(resolved, pygrain.RandomMapTransform):
            stages.append(RandomMap(resolved, rejects=_rejects(resolved), configured=binding))
        else:
            stages.append(Map(resolved, rejects=_rejects(resolved), configured=binding))
    return dataclasses.replace(pipeline, source=source, stages=tuple(stages))


@dataclasses.dataclass(frozen=True)
class _Apply(pygrain.MapTransform):
    stage: Map

    def map(self, element: Record) -> Record:
        if isinstance(element, Rejected):
            return element
        if element is None:
            return Rejected("missing")
        step = self.stage.step
        if isinstance(step, pygrain.MapTransform):
            return step.map(element)
        return step(record(element, "map input"))


@dataclasses.dataclass(frozen=True)
class _Random(pygrain.RandomMapTransform):
    stage: RandomMap

    def random_map(self, element: Record, rng: np.random.Generator) -> Record:
        if isinstance(element, Rejected):
            return element
        step = self.stage.step
        return (step.random_map(element, rng) if isinstance(step, pygrain.RandomMapTransform)
                else step(record(element, "random map input"), rng))


@dataclasses.dataclass(frozen=True)
class _Predicate(pygrain.MapTransform):
    stage: Filter

    def map(self, element: Record) -> Record:
        if isinstance(element, Rejected):
            return element
        return element if self.stage.predicate(record(element, "filter input")) else Rejected(self.stage.reason)


@dataclasses.dataclass(frozen=True)
class _Expand:
    stage: FlatMap

    def flat_map(self, element: Record) -> Sequence[Record]:
        if isinstance(element, Rejected):
            return (element,)
        outputs = self.stage.step.flat_map(record(element, "flat_map input"))
        return outputs or (Rejected("empty_expansion"),)


@dataclasses.dataclass(frozen=True)
class _Prefix:
    stages: tuple[Stage, ...]

    @property
    def volatile(self) -> bool:
        return any(
            isinstance(stage, Map) and isinstance(stage.step, Volatile) and stage.step.volatile
            for stage in self.stages
        )

    def __call__(self, index: int, raw: Record) -> tuple[Record, ...]:
        held: tuple[Record, ...] = (raw,)
        for stage in self.stages:
            emitted: list[Record] = []
            for value in held:
                if isinstance(value, Rejected):
                    emitted.append(value)
                elif isinstance(stage, Map):
                    emitted.append(_Apply(stage).map(value))
                elif isinstance(stage, MapWithIndex):
                    emitted.append(stage.step(index, record(value, "indexed map input")))
                elif isinstance(stage, Filter):
                    emitted.append(_Predicate(stage).map(value))
                elif isinstance(stage, FlatMap):
                    emitted.extend(stage.step.flat_map(record(value, "cached expansion input")))
                else:
                    raise ValueError(f"cache prefix contains non-record stage {type(stage).__name__}")
            held = tuple(emitted)
        return held


@dataclasses.dataclass(frozen=True)
class _CachedOutputs:
    def flat_map(self, outputs: Sequence[Batch | Rejected]) -> Sequence[Batch | Rejected]:
        return tuple(Rejected("missing") if value is None else value for value in outputs) or (Rejected("empty_expansion"),)


class _CacheIterator(pygrain.DatasetIterator[Record]):
    def __init__(self, parent: pygrain.DatasetIterator[Record], owner: Closeable):
        super().__init__(parent)
        self._read, self._owner = parent, owner

    def __next__(self) -> Record:
        try:
            return next(self._read)
        except StopIteration:
            if isinstance(self._owner, _Finishable):
                self._owner.finish()
            raise

    def get_state(self) -> Mapping[str, object]:
        return self._read.get_state()

    def set_state(self, state: Mapping[str, object]) -> None:
        self._read.set_state(state)

    def close(self) -> None:
        try:
            self._read.close()
        finally:
            self._owner.close()


class _CacheRows(pygrain.IterDataset[Record]):
    def __init__(self, parent: pygrain.IterDataset[Record], owner: Closeable):
        super().__init__(parent)
        self._records, self._owner = parent, owner

    def __iter__(self) -> pygrain.DatasetIterator[Record]:
        return _CacheIterator(iter(self._records), self._owner)


@dataclasses.dataclass(frozen=True)
class _Memory:
    original: Source
    held: tuple[Batch | bytes, ...]
    measured: np.ndarray | None
    derived: Mapping[str, str]
    facts: Mapping[str, object] = dataclasses.field(default_factory=dict)
    identifiers: tuple[tuple[int, ...], ...] = ()
    inputs: int = 0

    def describe(self) -> str:
        return self.original.describe()

    def snapshot(self) -> Mapping[str, object]:
        return {"materialized": self.original.snapshot(), "derived": self.derived}

    def supports(self, read: Read) -> bool:
        return read in ("memory", "indexed")

    def records(self, read: Read) -> tuple[Batch | bytes, ...]:
        return self.held

    def metadata(self) -> Mapping[str, object]:
        return {**self.original.metadata(), **self.facts, "accepted": len(self.held)}

    def lengths(self) -> np.ndarray | None:
        return self.measured

    def locality(self) -> tuple[tuple[int, ...], bool]:
        if not isinstance(self.original, _Locality) or any(len(identifier) != 1
                                                          for identifier in self.identifiers):
            raise ValueError("block shuffle needs an unchanged physical shard index")
        sizes, remote = self.original.locality()
        ends = np.cumsum(sizes)
        selected = np.bincount([int(np.searchsorted(ends, identifier[0], side="right"))
                               for identifier in self.identifiers], minlength=len(sizes))
        return tuple(int(size) for size in selected), remote

    def manifest(self) -> ShardManifest:
        raise ValueError("materialized records are read by index")


def _materialize(pipeline: Pipeline, counts: dict[str, CorpusCounts],
                 groups: LengthGroups | None = None, *, training: bool = True) -> Pipeline:
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
    rows, consumed = _memory_inputs(pipeline, source, stored, counter)
    return _memory_stages(pipeline, source, stored, rows, consumed, counter, groups, training=training)


def _memory_inputs(pipeline: Pipeline, source: Source, stored: Records, counter: CorpusCounts
                    ) -> tuple[list[tuple[tuple[int, ...], Record]], int]:
    rows: list[tuple[tuple[int, ...], Record]] = []
    consumed = 0
    artifact_at = next(
        (index for index, stage in enumerate(pipeline.stages) if isinstance(stage, Artifact)), None
    )
    if artifact_at is not None and any(isinstance(stage, Configured) for stage in pipeline.stages[:artifact_at]):
        artifact_at = None
    if artifact_at is not None:
        artifact = pipeline.stages[artifact_at]
        if not isinstance(artifact, Artifact):
            raise RuntimeError("cache stage changed in an immutable pipeline")
        prefix = pipeline.stages[:artifact_at]
        manifest = _cache_manifest(source, prefix, "memory")
        cached = artifact.cache.mapped(
            pygrain.MapDataset.source(stored), manifest, source, transform=_Prefix(prefix)
        )
        try:
            for index in range(len(cached)):
                outputs = cached[index]
                if not isinstance(outputs, tuple):
                    raise TypeError("cached input must return its tuple of expansion outputs")
                for child, value in enumerate(outputs):
                    if isinstance(value, Rejected):
                        counter.rejected[value.reason] += 1
                    else:
                        rows.append(((index, child), value))
            if isinstance(cached, _Finishable):
                cached.finish()
        finally:
            if isinstance(cached, Closeable):
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


def _memory_stages(pipeline: Pipeline, source: Source, stored: Records,
                    rows: list[tuple[tuple[int, ...], Record]], consumed: int,
                    counter: CorpusCounts, groups: LengthGroups | None, *, training: bool) -> Pipeline:
    derived: dict[str, str] = {}
    for stage in pipeline.stages[consumed:]:
        if not isinstance(stage, (Map, MapWithIndex, Filter, FlatMap, Pack, Windows, TokenSplit)):
            break
        if isinstance(stage, TokenSplit):
            rows = _split_tokens(rows, stage)
        elif isinstance(stage, Windows):
            from dew.data.steps.windows import WindowRecords

            derived[f"accepted_before_windows_{consumed}"] = fingerprint([list(identifier) for identifier, _ in rows])
            windows = WindowRecords(tuple(record(value, "window document") for _, value in rows), stage)
            derived["windows"] = fingerprint([windows.size, stage.seq_len, stage.stride])
            rows = [((index,), windows[index]) for index in range(len(windows))]
        elif isinstance(stage, Pack):
            from dew.data.tokens import PackedWindows, chunk_lengths, first_fit

            derived[f"accepted_before_pack_{consumed}"] = fingerprint([list(identifier) for identifier, _ in rows])
            documents = tuple(_Payload()(record(value, "packing document")) for _, value in rows)
            sizes = np.asarray([len(value[stage.fields[0]]) for value in documents], np.int64)
            packed = PackedWindows(
                pygrain.MapDataset.source(documents), sizes, stage.seq_len, stage.bins, source.describe()
            )
            plan, starts = first_fit(chunk_lengths(sizes, stage.seq_len), stage.seq_len, stage.bins)
            derived["packing"] = fingerprint([plan.tolist(), starts.tolist()])
            rows = [((index,), packed[index]) for index in range(len(packed))]
        else:
            expanded: list[tuple[tuple[int, ...], Record]] = []
            for index, (identifier, value) in enumerate(rows):
                if isinstance(stage, FlatMap):
                    outputs = stage.step.flat_map(record(value, "expansion input"))
                    if not outputs:
                        counter.rejected["empty_expansion"] += 1
                elif isinstance(stage, MapWithIndex):
                    outputs = (stage.step(index, record(value, "indexed map input")),)
                else:
                    outputs = ((_Apply(stage) if isinstance(stage, Map) else _Predicate(stage)).map(value),)
                for child, output in enumerate(outputs):
                    if isinstance(output, Rejected):
                        counter.rejected[output.reason] += 1
                    else:
                        expanded.append(
                            ((*identifier, child) if isinstance(stage, FlatMap) else identifier, output)
                        )
            rows = expanded
        consumed += 1
    inputs = source.inputs if isinstance(source, _Memory) else len(stored)
    return _memory_order(pipeline, rows, consumed, source, inputs, counter, derived, groups,
                         training=training)


def _split_tokens(rows: Sequence[tuple[tuple[int, ...], Record]], split: TokenSplit
                  ) -> list[tuple[tuple[int, ...], Record]]:
    documents = [(identifier, record(value, "token split document")) for identifier, value in rows]
    arrays = [{name: np.asarray(value[name]) for name in split.fields} for _, value in documents]
    if any(column.ndim != 1 for fields in arrays for column in fields.values()):
        raise ValueError("token split fields must be one-dimensional token arrays")
    total = sum(len(fields[split.fields[0]]) for fields in arrays)
    boundary = min(int(total * split.fraction), max(0, total - 1))
    start, stop = (0, boundary) if split.held else (boundary, total)
    selected = []
    offset = 0
    for (identifier, value), fields in zip(documents, arrays, strict=True):
        size = len(fields[split.fields[0]])
        first, last = max(0, start - offset), min(size, stop - offset)
        if first < last:
            selected.append(((*identifier, first, last), {
                key: fields[key][first:last] if key in fields else column for key, column in value.items()}))
        offset += size
    return selected


def _memory_mixture(pipeline: Pipeline, mixture: Mixture, counts: dict[str, CorpusCounts],
                    groups: LengthGroups | None, *, training: bool) -> Pipeline:
    children = tuple(_materialize(child, counts, groups, training=training) for child in mixture.pipelines)
    kept = tuple((child, weight) for child, weight in zip(children, mixture.weights, strict=True)
                 if isinstance(child.source, (Mixture, Phases)) or len(child.source.records("memory")))
    omitted = [_name(child) for child in children if all(child is not accepted for accepted, _ in kept)]
    if omitted:
        _log.warning("left out of the mixture: %s (zero accepted rows)", ", ".join(omitted))
    if not kept:
        if not training:
            return dataclasses.replace(children[0], stages=(*children[0].stages, *pipeline.stages))
        raise ValueError(f"no accepted row of {', '.join(_name(child) for child in children)}; raise the row budgets")
    if len(kept) == 1:
        child = kept[0][0]
        return dataclasses.replace(child, stages=(*child.stages, *pipeline.stages))
    children, weights = zip(*kept, strict=True)
    return dataclasses.replace(pipeline, source=dataclasses.replace(mixture, pipelines=tuple(children),
                                                                   weights=tuple(weights)))


def _memory_order(pipeline: Pipeline, collected: Sequence[tuple[tuple[int, ...], Record]], consumed: int,
                  source: Source, inputs: int, counter: CorpusCounts, derived: dict[str, str],
                  groups: LengthGroups | None, *, training: bool) -> Pipeline:
    rows = [(identifier, value if isinstance(value, bytes) else record(value, "materialized record"))
            for identifier, value in collected]
    remaining = pipeline.stages[consumed:]
    lengths = (
        np.asarray([groups.length(record(value, "grouped record")) for _, value in rows], np.int64)
        if groups is not None
        else source.lengths()
        if consumed == 0
        else None
    )
    derived["accepted_ids"] = fingerprint([list(identifier) for identifier, _ in rows])
    fanouts = np.zeros(inputs, np.int64)
    for identifier, _ in rows:
        if identifier[0] < len(fanouts):
            fanouts[identifier[0]] += 1
    derived["fanouts"] = fingerprint(fanouts)
    if lengths is not None:
        derived["lengths"] = fingerprint(lengths)
    counter.accepted = len(rows)
    counter.truncated = sum(bool(value.get("_dew_truncated", False)) for _, value in rows
                            if not isinstance(value, bytes))
    facts: Batch = _measured_bounds([record(value, "measured row") for _, value in rows
                                     if not isinstance(value, bytes)])
    if any(isinstance(stage, (Pack, Windows)) for stage in pipeline.stages[:consumed]):
        from dew.data.steps.text import ValidTargets

        targets = ValidTargets()
        kept = []
        for identifier, value in rows:
            count = targets(record(value, "packed window"))
            if count:
                kept.append((identifier, value))
            else:
                counter.rejected["zero_targets"] += 1
        rows = kept
        facts = {**facts, "windows": len(rows), "valid_targets": sum(targets(record(value, "packed window")) for _, value in rows)}
        if facts["valid_targets"] == 0 and training:
            raise ValueError(f"corpus {_name(pipeline)} has zero valid target tokens")
        derived["valid_targets"] = fingerprint(facts)
        counter.accepted = len(rows)
    derived["accepted_ids"] = fingerprint([list(identifier) for identifier, _ in rows])
    return dataclasses.replace(
        pipeline,
        source=_Memory(source, tuple(value for _, value in rows), lengths, derived, facts,
                       tuple(identifier for identifier, _ in rows), inputs),
        stages=remaining,
    )


def _measured_bounds(rows: Sequence[Batch]) -> Mapping[str, object]:
    bounds: dict[str, JSON] = {}

    def merge(target: dict[str, JSON], given: Mapping[str, object]) -> None:
        for key, value in given.items():
            if isinstance(value, Mapping):
                previous = {name: json_value(datum, name)
                            for name, datum in record(target.get(key, {}), key).items()}
                merge(previous, value)
                target[key] = previous
            else:
                target[key] = max(integer(value, key), integer(target.get(key, 0), key))

    for row in rows:
        merge(bounds, record(row.get("_dew_bounds", {}), "record bounds"))
    return bounds


class _GroupedIterator(pygrain.DatasetIterator[Batch]):
    def __init__(self, parent: pygrain.DatasetIterator[Batch], groups: LengthGroups, rows: int, seed: int):
        super().__init__(parent)
        self._read, self._groups, self._rows, self._seed = parent, groups, rows, seed
        self._before = parent.get_state()
        self._buffer: list[Batch] = []
        self._emitted, self._window = 0, 0

    def _fill(self) -> None:
        self._before = self._read.get_state()
        held = []
        for _ in range(self._rows * self._groups.window):
            try:
                held.append(next(self._read))
            except StopIteration:
                break
        ranked = sorted(held, key=self._groups.length)
        batches = [ranked[start : start + self._rows] for start in range(0, len(ranked), self._rows)]
        order = np.random.default_rng([self._seed, self._window]).permutation(len(batches))
        self._buffer = [value for index in order for value in batches[index]]
        self._emitted = 0

    def __next__(self) -> Batch:
        if self._emitted == len(self._buffer):
            if self._buffer:
                self._window += 1
            self._fill()
        if not self._buffer:
            raise StopIteration
        value = self._buffer[self._emitted]
        self._emitted += 1
        return value

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._before, "emitted": self._emitted, "window": self._window}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._read.set_state(record(state["parent"], "group parent"))
        self._window = integer(state["window"], "group window")
        self._fill()
        self._emitted = integer(state["emitted"], "group emitted")
        if not 0 <= self._emitted <= len(self._buffer):
            raise ValueError("saved grouping position exceeds its window")

    def close(self) -> None:
        self._read.close()


class _GroupedRows(pygrain.IterDataset[Batch]):
    def __init__(self, parent: pygrain.IterDataset[Batch], groups: LengthGroups, rows: int, seed: int):
        super().__init__(parent)
        self._records, self._groups, self._rows, self._seed = parent, groups, rows, seed

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        return _GroupedIterator(iter(self._records), self._groups, self._rows, self._seed)


class _Host(Forwarding):
    def __init__(self, source: Iterator[Batch], steps: tuple[HostStep, ...]):
        self._source, self._steps = source, steps

    def _forwarded(self) -> Iterator[Batch]:
        return self._source

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        value = next(self._source)
        for step in self._steps:
            value = step.map(value) if isinstance(step, pygrain.MapTransform) else step(value)
        return value

    def get_state(self) -> Position:
        if not isinstance(self._source, Checkpointable):
            raise TypeError("map_batch parent does not expose a checkpoint position")
        return self._source.get_state()

    def set_state(self, state: Position) -> None:
        if not isinstance(self._source, Checkpointable):
            raise TypeError("map_batch parent does not expose a checkpoint position")
        self._source.set_state(state)

    @property
    def records(self) -> int:
        if not isinstance(self._source, GlobalReader):
            raise TypeError("host-stage parent has no global record count")
        return self._source.records

    @property
    def order(self) -> str:
        if not isinstance(self._source, GlobalReader):
            raise TypeError("host-stage parent has no global order")
        return self._source.order


class _CanonicalPass:
    def __init__(self, reader: Reader, partition: DataPartition, batch: int):
        import jax

        self._partition = partition
        self._batch = batch
        self._source = reader(DataPartition()) if jax.process_index() == 0 else None

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        import jax
        from jax.experimental import multihost_utils

        from dew.coordination import agreed, broadcast_from_process_zero
        from dew.data.sources.codec import decode, encode

        root = jax.process_index() == 0
        held = agreed("canonical validation read", lambda: next(self._source, None) if self._source is not None else None)
        payload = b"" if held is None else encode(held)
        size = broadcast_from_process_zero(len(payload))
        if not size:
            raise StopIteration
        if jax.process_count() > 1:
            buffer = np.frombuffer(payload, np.uint8).copy() if root else np.zeros(size, np.uint8)
            payload = np.asarray(multihost_utils.broadcast_one_to_all(buffer, is_source=root)).tobytes()
        whole = record(decode(payload), "canonical validation batch")
        return _FillShare(self._partition, self._batch)(whole)

    def close(self) -> None:
        if isinstance(self._source, Closeable):
            self._source.close()
        self._source = None


class _PerShare(Forwarding):
    @property
    def endless(self) -> bool:
        return self._endless

    def __init__(
        self,
        open_reader: Callable[[], pygrain.DatasetIterator[Batch]],
        plan: Plan,
        resume: Literal["exact", "non-exact"],
        *, endless: bool = True,
    ):
        self._open, self._plan = open_reader, plan
        self._resume: Literal["exact", "non-exact"] = resume
        self._endless = endless
        self._read: pygrain.DatasetIterator[Batch] | None = None
        self._saved: Mapping[str, object] | None = None

    def _forwarded(self) -> pygrain.DatasetIterator[Batch]:
        if self._read is None:
            self._read = self._open()
            if self._saved is not None:
                self._read.set_state(self._saved)
        return self._read

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        return next(self._forwarded())

    def get_state(self) -> bytes:
        return json.dumps(
            {"dew_data_plan": self._plan.record(), "iterator": self._forwarded().get_state()}
        ).encode()

    def set_state(self, state: bytes) -> None:
        decoded = record(json.loads(state), "saved per-share position")
        self._plan.check(record(decoded.get("dew_data_plan"), "saved data plan"), resume=self._resume)
        saved = record(decoded.get("iterator"), "saved Grain iterator")
        self.close()
        self._saved = saved

    def identity(self) -> str:
        return fingerprint(self._plan.record())

    def statistics(self) -> Mapping[str, float]:
        if self._read is None:
            return {}
        return {key: value for counts in state_counts(self._read.get_state()).values()
                for key, value in counts.metrics().items()}

    def close(self) -> None:
        if self._read is not None:
            self._read.close()
            self._read = None


class _Statistics:
    def __init__(self, counts: Mapping[str, CorpusCounts]):
        self._counts = counts
        self._readers: list[weakref.ReferenceType[_PerShare]] = []

    def observe(self, stream: _PerShare) -> None:
        self._readers.append(weakref.ref(stream))

    def __call__(self) -> Mapping[str, float]:
        scalars = {key: value for counts in self._counts.values() for key, value in counts.metrics().items()}
        for reference in self._readers:
            stream = reference()
            if stream is not None:
                for name, value in stream.statistics().items():
                    scalars[name] = max(scalars.get(name, 0.), value)
        return scalars


def _per_share(pipeline: Pipeline, read: Read) -> bool:
    if isinstance(pipeline.source, Phases):
        return any(_per_share(child, read) for child in pipeline.source.pipelines)
    if read == "streamed":
        return True
    iterable = False
    for stage in pipeline.stages:
        if isinstance(stage, Prefetch):
            iterable = True
        elif iterable and isinstance(stage, (RandomMap, MapWithIndex, Group)):
            return True
    if isinstance(pipeline.source, (Mixture, Phases)):
        if pipeline.source.tokens is not None:
            for child in pipeline.source.pipelines:
                if isinstance(child.source, (Mixture, Phases)) or "valid_targets" not in child.source.metadata():
                    return True
        if any(_per_share(child, read) for child in pipeline.source.pipelines):
            return True
    return any(
        isinstance(stage, (Filter, FlatMap, Pack, Windows, Interleave, Artifact, Rebatch, TokenBudget))
        or (isinstance(stage, (Map, MapWithIndex, RandomMap)) and stage.rejects)
        for stage in pipeline.stages
    )


def _as_iter(rows: Rows, loading: Loading) -> pygrain.IterDataset:
    return (
        rows.to_iter_dataset(pygrain.ReadOptions(loading.threads, loading.read_buffer))
        if isinstance(rows, pygrain.MapDataset)
        else rows
    )


@dataclasses.dataclass(frozen=True)
class _Build:
    read: Read
    loading: Loading
    seed: int
    batch: int
    budget: SearchBudget
    global_order: bool
    training: bool

    def source(self, pipeline: Pipeline, partition: DataPartition) -> tuple[Rows, pygrain.MapDataset | None]:
        source = pipeline.source
        if isinstance(source, Phases):
            raise ValueError("phases need the global phased lowering")
        if isinstance(source, (Mixture, Phases)):
            built = [self.records(child, partition) for child in source.pipelines]
            parents, lengths = zip(*built, strict=True)
            if all(isinstance(parent, pygrain.MapDataset) for parent in parents):
                mapped = [parent for parent in parents if isinstance(parent, pygrain.MapDataset)]
                weights = source.weights
                if source.tokens is not None:
                    metadata = [
                        child.source.metadata()
                        for child in source.pipelines
                        if not isinstance(child.source, (Mixture, Phases))
                    ]
                    weights = token_weights(
                        [_name(child) for child in source.pipelines],
                        weights,
                        [integer(stated["windows"], "windows") for stated in metadata],
                        [integer(stated["valid_targets"], "valid_targets") for stated in metadata],
                    )
                measured = [length for length in lengths if length is not None]
                checked = primed_maps(mapped, [_name(child) for child in source.pipelines])
                return pygrain.MapDataset.mix(checked, weights).seed(self.seed), (
                    pygrain.MapDataset.mix(measured, weights) if len(measured) == len(parents) else None
                )
            streams = [
                Accepted(_as_iter(parent, self.loading), _name(child), budget=self.budget)
                for parent, child in zip(parents, source.pipelines, strict=True)
            ]
            if source.tokens is not None:
                return TokenDeficit(
                    streams,
                    source.weights,
                    tokens=source.tokens,
                    names=[_name(child) for child in source.pipelines],
                    budget=self.budget,
                ), None
            return PrimedMix(streams, source.weights, [_name(child) for child in source.pipelines]).seed(self.seed), None
        if self.read == "streamed":
            from dew.data.shards import streamed

            return streamed(
                source.manifest(),
                partition=partition,
                seed=self.seed if self.training else None,
                cycle_length=1,
                loading=self.loading,
                epochs=_epochs(pipeline) if self.training else 1,
            ).seed(self.seed), None
        stored = source.records(self.read)
        rows = pygrain.MapDataset.source(stored).seed(self.seed)
        lengths = source.lengths()
        measured = None if lengths is None else pygrain.MapDataset.source(lengths)
        if not self.global_order and self.training:
            rows = rows[partition.index :: partition.count]
            measured = None if measured is None else measured[partition.index :: partition.count]
            if not any(isinstance(stage, Repeat) for stage in pipeline.stages):
                rows = rows.repeat(None)
        return rows, measured

    def records(self, pipeline: Pipeline, partition: DataPartition,
                offset: int = 0) -> tuple[Rows, pygrain.MapDataset | None]:
        source = pipeline.source
        artifact_at = next(
            (index for index, stage in enumerate(pipeline.stages) if isinstance(stage, Artifact)), None
        )
        consumed = 0
        if artifact_at is not None and not isinstance(source, (Mixture, Phases)):
            artifact = pipeline.stages[artifact_at]
            if not isinstance(artifact, Artifact):
                raise RuntimeError("cache stage changed while compiling an immutable pipeline")
            prefix = pipeline.stages[:artifact_at]
            if any(not isinstance(stage, (Map, MapWithIndex, Filter, FlatMap)) for stage in prefix):
                raise ValueError(
                    f"cache prefix in {_name(pipeline)} must contain only deterministic record steps"
                )
            manifest = _cache_manifest(source, prefix, self.read)
            transform = _Prefix(prefix)
            if self.read == "streamed":
                from dew.data.shards import streamed

                shards = source.manifest()
                reader = artifact.cache.streamed(shards.reader, manifest, shards, transform=transform)
                shards = dataclasses.replace(shards, reader=reader)
                cycle = next(
                    (stage.cycle_length for stage in pipeline.stages if isinstance(stage, Interleave)), 1
                )
                rows = streamed(
                    shards,
                    partition=partition,
                    seed=self.seed if self.training else None,
                    cycle_length=cycle,
                    loading=self.loading,
                    epochs=_epochs(pipeline) if self.training else 1,
                ).seed(self.seed)
            else:
                rows = artifact.cache.mapped(
                    pygrain.MapDataset.source(source.records(self.read)).seed(self.seed),
                    manifest,
                    source,
                    transform=transform,
                )
                if isinstance(rows, Closeable):
                    rows = _CacheRows(_as_iter(rows, self.loading), rows)
            rows = Accepted(gx.FlatMapIterDataset(_as_iter(rows, self.loading), _CachedOutputs()),
                            _name(pipeline), budget=self.budget)
            lengths = None
            consumed = artifact_at + 1
        else:
            rows, lengths = self.source(pipeline, partition)
        for stage in pipeline.stages[consumed:]:
            if isinstance(stage, (BatchStage, TokenBudget, Workers, MapBatch, Rebatch)):
                break
            if self.read == "streamed" and isinstance(stage, Repeat) and _root_repeat(pipeline):
                continue
            if isinstance(stage, Prefetch) and isinstance(rows, pygrain.MapDataset) and self.global_order:
                if self.training and not any(isinstance(step, Repeat) for step in pipeline.stages):
                    rows = rows.repeat(None)
                rows = rows[offset + partition.index::partition.count]
                workers = _workers(pipeline, self.loading)
                if workers:
                    rows = _WorkerBatches(rows, partition.rows(self.batch), workers,
                                          remainder=not self.training)
            rows, lengths = self.stage(rows, lengths, stage, pipeline, partition)
        if not self.global_order:
            rows = Accepted(_as_iter(rows, self.loading), _name(pipeline), budget=self.budget)
        return rows, lengths

    def stage(
        self,
        rows: Rows,
        lengths: pygrain.MapDataset | None,
        stage: Stage,
        pipeline: Pipeline,
        partition: DataPartition,
    ) -> tuple[Rows, pygrain.MapDataset | None]:
        if isinstance(stage, (Map, RandomMap, MapWithIndex, Filter, FlatMap)):
            return self.record_stage(rows, lengths, stage, pipeline)
        if isinstance(stage, Shuffle):
            if stage.block_shards is not None:
                return self.block_shuffle(rows, lengths, stage, pipeline, partition)
            if isinstance(rows, pygrain.MapDataset):
                rows = (
                    rows.shuffle(stage.seed)
                    if stage.window is None
                    else gx.WindowShuffleMapDataset(rows, window_size=stage.window, seed=stage.seed)
                )
                if lengths is not None:
                    lengths = (
                        lengths.shuffle(stage.seed)
                        if stage.window is None
                        else gx.WindowShuffleMapDataset(lengths, window_size=stage.window, seed=stage.seed)
                    )
            elif stage.window is not None:
                rows = gx.WindowShuffleIterDataset(rows, window_size=stage.window, seed=stage.seed)
            else:
                raise ValueError(f"global shuffle needs an indexed pipeline: {_name(pipeline)}")
        elif isinstance(stage, Repeat):
            rows = (
                rows.repeat(stage.epochs)
                if isinstance(rows, pygrain.MapDataset)
                else gx.RepeatIterDataset(rows, stage.epochs)
            )
            lengths = None if lengths is None else lengths.repeat(stage.epochs)
        elif isinstance(stage, Slice):
            if not isinstance(rows, pygrain.MapDataset):
                raise ValueError(f"slice needs an indexed pipeline: {_name(pipeline)}")
            rows = rows.slice(stage.selection)
            lengths = None if lengths is None else lengths.slice(stage.selection)
        elif isinstance(stage, ShardByHost):
            if not isinstance(rows, pygrain.MapDataset):
                raise ValueError("shard_by_host must be applied at the indexed or shard-manifest root")
        elif isinstance(stage, Prefetch):
            rows = gx.ThreadPrefetchIterDataset(
                _as_iter(rows, self.loading), prefetch_buffer_size=stage.buffer
            )
        elif isinstance(stage, Limit):
            if isinstance(rows, pygrain.MapDataset):
                rows = rows[:min(stage.count, len(rows))]
                lengths = None if lengths is None else lengths[:min(stage.count, len(lengths))]
            else:
                rows = gx.LimitIterDataset(rows, stage.count)
        elif isinstance(stage, Pack):
            parent = _as_iter(rows, self.loading).map(_Payload())
            sizes = dict.fromkeys(stage.fields, stage.seq_len)
            if stage.policy == "concat":
                rows = gx.ConcatThenSplitIterDataset(parent, length_struct=sizes)
            else:
                packer = (
                    gx.FirstFitPackIterDataset if stage.policy == "first-fit" else gx.BestFitPackIterDataset
                )
                rows = packer(parent, length_struct=sizes, num_packing_bins=stage.bins, shuffle_bins=False)
            lengths = None
        elif isinstance(stage, Windows):
            from dew.data.steps.windows import CausalWindows

            rows = CausalWindows(_as_iter(rows, self.loading), stage)
            lengths = None
        elif isinstance(stage, Group):
            if isinstance(rows, pygrain.MapDataset):
                if lengths is None:
                    raise ValueError(
                        f"length grouping of {_name(pipeline)} needs producer lengths or read='memory'"
                    )
                rows = _Grouped(rows, stage.groups, self.batch, self.seed, length_at=_LengthAt(lengths))
            else:
                rows = _GroupedRows(rows, stage.groups, partition.rows(self.batch), self.seed)
            lengths = None
        elif isinstance(stage, Artifact):
            if isinstance(pipeline.source, (Mixture, Phases)):
                raise ValueError("cache belongs on each corpus before mix")
            prefix = pipeline.stages[: pipeline.stages.index(stage)]
            manifest = _cache_manifest(pipeline.source, prefix, self.read)
            if not isinstance(rows, pygrain.MapDataset):
                raise ValueError("streamed cache must be applied to its shard reader before interleave")
            rows = stage.cache.mapped(rows, manifest, pipeline.source)
        elif isinstance(stage, Interleave):
            if self.read != "streamed":
                raise ValueError("interleave needs a shard-manifest source")
        return rows, lengths

    def block_shuffle(self, rows: Rows, lengths: pygrain.MapDataset | None, stage: Shuffle,
                      pipeline: Pipeline, partition: DataPartition) -> tuple[Rows, pygrain.MapDataset | None]:
        from dew.data.ordering import BlockShuffle, PermutedLengths

        if not isinstance(rows, pygrain.MapDataset) or not isinstance(pipeline.source, _Locality):
            raise ValueError(f"block shuffle of {_name(pipeline)} needs an indexed source with physical shard counts")
        if stage.block_shards is None:
            raise ValueError("block shuffle needs a block_shards size")
        sizes, _ = pipeline.source.locality()
        if not self.global_order and self.training:
            starts = np.cumsum((0, *sizes[:-1]))
            sizes = tuple(max(0, math.ceil((int(start) + size - partition.index) / partition.count)
                                 - math.ceil((int(start) - partition.index) / partition.count))
                          for start, size in zip(starts, sizes, strict=True))
        blocked = BlockShuffle(rows, sizes, stage.block_shards, stage.seed)
        lengths = None if lengths is None else PermutedLengths(lengths, blocked.index)
        return blocked, lengths

    def record_stage(self, rows: Rows, lengths: pygrain.MapDataset | None,
                     stage: Map | RandomMap | MapWithIndex | Filter | FlatMap,
                     pipeline: Pipeline) -> tuple[Rows, pygrain.MapDataset | None]:
        if isinstance(stage, Map):
            rows = rows.map(_Apply(stage))
            if stage.rejects:
                rows = Accepted(_as_iter(rows, self.loading), _name(pipeline), budget=self.budget)
                lengths = None
        elif isinstance(stage, RandomMap):
            rows = rows.random_map(_Random(stage))
            if stage.rejects:
                rows = Accepted(_as_iter(rows, self.loading), _name(pipeline), budget=self.budget)
                lengths = None
        elif isinstance(stage, MapWithIndex):
            rows = rows.map_with_index(stage.step)
            if stage.rejects:
                rows = Accepted(_as_iter(rows, self.loading), _name(pipeline), budget=self.budget)
                lengths = None
        elif isinstance(stage, Filter):
            rows = Accepted(_as_iter(rows, self.loading).map(_Predicate(stage)),
                            _name(pipeline), budget=self.budget)
            lengths = None
        elif isinstance(stage, FlatMap):
            rows = Accepted(gx.FlatMapIterDataset(_as_iter(rows, self.loading), _Expand(stage)),
                            _name(pipeline), budget=self.budget)
            lengths = None
        return rows, lengths

    def batches(
        self, pipeline: Pipeline, partition: DataPartition, offset: int = 0, *, fill: bool = True
    ) -> pygrain.DatasetIterator[Batch]:
        budget = next((stage for stage in pipeline.stages if isinstance(stage, TokenBudget)), None)
        if budget is not None:
            return self.token_batches(pipeline, partition, budget)
        canonical = not self.training and not self.global_order
        rows, _ = self.records(pipeline, DataPartition() if canonical else partition, offset)
        batch_stage = next(stage for stage in pipeline.stages if isinstance(stage, BatchStage))
        shared_size = partition.rows(batch_stage.size)
        size = batch_stage.size if canonical else shared_size
        workers = _workers(pipeline, self.loading)
        empty = False
        if isinstance(rows, pygrain.MapDataset):
            if self.training and not any(isinstance(stage, (Repeat, Limit)) for stage in _steps(pipeline)):
                rows = rows.repeat(None)
            elif self.training and self.global_order and not _endless(pipeline):
                rows = rows[:len(rows) // batch_stage.size * batch_stage.size]
            if self.global_order or not self.training:
                whole = rows
                rows = rows[offset + partition.index :: partition.count]
                empty = not self.training and not len(rows) and bool(len(whole))
                if empty:
                    rows = whole[:1]
            if workers:
                rows = _WorkerBatches(rows, size, workers, remainder=not self.training)
        stream = _as_iter(rows, self.loading).map(_Payload()).batch(
            size,
            drop_remainder=batch_stage.drop_remainder if self.training else False,
            batch_fn=batch_stage.batch_fn,
        )
        if not self.training and fill:
            stream = stream.map(_FillShare(partition, batch_stage.size) if canonical
                                else FilledBatch(size, real=not empty))
        explicit_workers = False
        for stage in pipeline.stages[pipeline.stages.index(batch_stage) + 1 :]:
            if isinstance(stage, Workers):
                stream = stream.mp_prefetch(
                    pygrain.MultiprocessingOptions(
                        num_workers=stage.count, per_worker_buffer_size=stage.buffer
                    )
                )
                explicit_workers = True
            elif isinstance(stage, Prefetch):
                stream = gx.ThreadPrefetchIterDataset(stream, prefetch_buffer_size=stage.buffer)
            elif isinstance(stage, Rebatch):
                stream = gx.RebatchIterDataset(
                    stream, partition.rows(stage.size), drop_remainder=self.training
                )
            elif isinstance(stage, Limit):
                stream = gx.LimitIterDataset(stream, stage.count)
            elif not isinstance(stage, MapBatch):
                stream, _ = self.stage(stream, None, stage, pipeline, partition)
                stream = _as_iter(stream, self.loading)
        if self.loading.workers and not explicit_workers:
            stream = stream.mp_prefetch(
                pygrain.MultiprocessingOptions(
                    num_workers=self.loading.worker_count, per_worker_buffer_size=self.loading.worker_buffer
                )
            )
        return iter(stream)

    def token_batches(self, pipeline: Pipeline, partition: DataPartition,
                      budget: TokenBudget) -> pygrain.DatasetIterator[Batch]:
        from dew.data.batching import TokenBatches

        if partition.count > 1:
            raise ValueError("token-budget batching requires one share until a synchronized shape schedule is supplied")
        rows, _ = self.records(pipeline, partition)
        stream = TokenBatches(_as_iter(rows, self.loading), budget, partition, self.seed)
        workers = _workers(pipeline, self.loading)
        if workers:
            stream = stream.mp_prefetch(pygrain.MultiprocessingOptions(
                num_workers=workers, per_worker_buffer_size=self.loading.worker_buffer))
        return iter(stream)


def _order(pipeline: Pipeline) -> Mapping[str, object]:
    source = pipeline.source
    if isinstance(source, Phases):
        return {"phases": [_order(child) for child in source.pipelines], "until_steps": list(source.until_steps)}
    children = (
        {
            "corpora": [_order(child) for child in source.pipelines],
            "weights": list(source.weights),
            "shares": "tokens" if source.tokens is not None else "accepted rows",
        }
        if isinstance(source, (Mixture, Phases))
        else {"source": source.describe()}
    )
    return {
        **children,
        "stages": [
            repr(stage)
            for stage in pipeline.stages
            if isinstance(stage, (Shuffle, Repeat, Slice, Limit, Pack, BatchStage))
        ],
        "groups": [stage.groups.window for stage in pipeline.stages if isinstance(stage, Group)],
        "token_budgets": [dataclasses.asdict(stage) for stage in pipeline.stages if isinstance(stage, TokenBudget)],
    }


def _geometry(pipeline: Pipeline) -> Geometry | None:
    declared = [stage.step for stage in pipeline.stages if isinstance(stage, (Map, RandomMap))
                and isinstance(stage.step, TokenGeometry)]
    if declared:
        last = declared[-1]
        return Geometry(last.max_length, last.token_fields, last.length_field)
    if isinstance(pipeline.source, (Mixture, Phases)):
        children = [_geometry(child) for child in pipeline.source.pipelines]
        if not children or any(child is None for child in children):
            return None
        known = [child for child in children if child is not None]
        if any(child.token_fields != known[0].token_fields or child.length_field != known[0].length_field
               for child in known):
            raise ValueError("mixed token geometries need matching token fields and length columns")
        return dataclasses.replace(known[0], max_length=max(child.max_length for child in known))
    return None


def _token_geometry(pipeline: Pipeline, fallback: Geometry | None = None) -> Pipeline:
    if not any(isinstance(stage, TokenBudget) for stage in pipeline.stages):
        return pipeline
    geometry = _geometry(pipeline) or fallback
    if geometry is None:
        raise ValueError(f"token-budget pipeline {_name(pipeline)} needs a record step declaring TokenGeometry")
    return dataclasses.replace(pipeline, stages=tuple(dataclasses.replace(stage, geometry=geometry)
                               if isinstance(stage, TokenBudget) else stage for stage in pipeline.stages))


def _budget_rows(budget: TokenBudget) -> int:
    from dew.data.batching import bucket_lengths

    lengths = bucket_lengths(budget)
    return budget.tokens // min(lengths)


def _derived(pipeline: Pipeline, seed: int, read: Read) -> Mapping[str, str]:
    arrays: dict[str, str] = {}
    for index, leaf in enumerate(_leaves(pipeline)):
        source = leaf.source
        if isinstance(source, (Mixture, Phases)):
            continue
        if isinstance(source, _Memory):
            arrays.update({f"{index}/{key}": value for key, value in source.derived.items()})
        lengths = source.lengths()
        if isinstance(source, _OrderFacts):
            arrays.update({f"{index}/{key}": value for key, value in source.order_fingerprints().items()})
        if lengths is not None and f"{index}/lengths" not in arrays:
            arrays[f"{index}/lengths"] = fingerprint(lengths)
        if any(isinstance(stage, Shuffle) and stage.block_shards is not None for stage in leaf.stages):
            if not isinstance(source, _Locality):
                raise ValueError(f"source {_name(leaf)} does not state physical shard locality")
            sizes, _ = source.locality()
            arrays[f"{index}/physical_shards"] = fingerprint(list(sizes))
        if read != "streamed":
            arrays[f"{index}/index"] = fingerprint([len(source.records(read)), seed])
        metadata = source.metadata()
        if "valid_targets" in metadata:
            arrays[f"{index}/valid_targets"] = fingerprint(
                integer(metadata["valid_targets"], "valid_targets")
            )
    return arrays


def compile(
    train: Pipeline,
    *,
    validation: Pipeline | None = None,
    host: tuple[HostStep, ...] = (),
    objective: ObjectiveData | None = None,
    read: Read = "indexed",
    provision: bool = False,
    loading: Loading = _DEFAULT_LOADING,
    seed: int = 0,
    rejections: Literal["drop", "mask"] = "drop",
    budget: SearchBudget = _DEFAULT_BUDGET,
    resume: Literal["exact", "non-exact"] = "exact",
) -> Dataset:
    """Check every leaf before opening readers, then build both branches."""
    if isinstance(train.source, Phases):
        return _compile_phases(train, validation=validation, host=host, objective=objective, read=read,
                               provision=provision, loading=loading, seed=seed, rejections=rejections,
                               budget=budget, resume=resume)
    if read not in ("memory", "indexed", "streamed"):
        raise ValueError(f"unknown read mode {read!r}")
    if rejections == "mask":
        raise ValueError(
            "rejections='mask' needs an objective proven mask-safe; no objective currently declares it"
        )
    pipelines = (train,) if validation is None else (train, validation)
    for pipeline in pipelines:
        _check(pipeline, read, validation=pipeline is validation)
    train, generations = _promoted(train)
    if validation is not None:
        validation, _ = _promoted(validation)
    pipelines = (train,) if validation is None else (train, validation)
    _capabilities(pipelines, read, provision)
    _tokenizer_contract(pipelines)
    counts: dict[str, CorpusCounts] = {}
    validation_counts: dict[str, CorpusCounts] = {}
    if objective is not None:
        train = _objective_branch(train, objective, train=True, read=read)
        validation = None if validation is None else _objective_branch(validation, objective, train=False, read=read)
    combined = (train,) if validation is None else (train, validation)
    if any(isinstance(stage, Configured) for branch in combined for stage in _steps(branch)):
        metadata = _metadata(combined, read)
        if read == "memory":
            train = _materialize(_configured(train, metadata, read, "source"), counts)
            validation = None if validation is None else _materialize(
                _configured(validation, metadata, read, "source"), validation_counts, training=False)
            metadata = _metadata((train,) if validation is None else (train, validation), read)
        train = _configured(train, metadata, read)
        validation = None if validation is None else _configured(validation, metadata, read)
        if objective is not None:
            train = _objective_order(train, objective)
    _check(train, read)
    if validation is not None:
        _check(validation, read, validation=True)
    train = _token_geometry(train)
    if validation is not None:
        validation = _token_geometry(validation, _geometry(train))
    batches = [stage for stage in train.stages if isinstance(stage, (BatchStage, TokenBudget))]
    if len(batches) != 1:
        raise ValueError("compile needs one batch(global_size) stage on the training pipeline")
    batching = batches[0]
    batch = batching.size if isinstance(batching, BatchStage) else _budget_rows(batching)
    batch_tokens = None if isinstance(batching, BatchStage) else batching.tokens
    original = train
    if read == "memory":
        train = _materialize(train, counts)
        validation = None if validation is None else _materialize(validation, validation_counts, training=False)
        if validation_counts and not any(count.accepted for count in validation_counts.values()):
            validation = None
    global_order = not _per_share(train, read)
    manifest = Manifest.of(
        {
            "sources": [
                leaf.source.snapshot() for leaf in _leaves(original) if not isinstance(leaf.source, (Mixture, Phases))
            ]
        },
        (*(_identity(stage) for stage in _steps(original) if not isinstance(stage, (Workers, Prefetch))),
         *(StepIdentity.of(step) for step in host)),
    )
    plan = Plan(
        "global" if global_order else "per-share",
        manifest,
        {"read": read, "seed": seed, **_order(train)},
        _derived(train, seed, read),
        cache=(*generations, *_cache_references(original, read)),
        volatile=any(
            isinstance(stage.step, Volatile) and stage.step.volatile
            for stage in _steps(train)
            if isinstance(stage, Map)
        ),
    )
    build = _Build(read, loading, seed, batch, budget, global_order, training=True)
    if loading.workers == "auto":
        tuning, _ = build.records(train, DataPartition())
        loading = loading.resolved(_as_iter(tuning, dataclasses.replace(loading, workers=0)))
        build = dataclasses.replace(build, loading=loading)
    train_host = tuple(stage.step for stage in train.stages if isinstance(stage, MapBatch)) + host
    statistics = _Statistics(counts)
    training = functools.partial(_training_reader, pipeline=train, build=build, plan=plan,
                                 host=train_host, resume=resume, statistics=statistics)
    held_reader = None if validation is None else functools.partial(
        _validation_reader, pipeline=validation, build=dataclasses.replace(build, training=False),
        host=tuple(stage.step for stage in validation.stages if isinstance(stage, MapBatch)) + host)
    records = _epoch_records(train, read) if global_order else None
    if records == 0:
        raise ValueError(f"corpus {_name(train)} has no accepted rows")
    held_out = sum(integer(leaf.source.metadata().get("held_out", 0), "held_out") for leaf in _leaves(train)
                   if not isinstance(leaf.source, (Mixture, Phases)))
    return Dataset(training, held_reader, records, batch, held_out=held_out, counts=counts,
                   validation_counts=validation_counts, plan=plan, statistics=statistics, batch_tokens=batch_tokens)


def _compile_phases(train: Pipeline, *, validation: Pipeline | None, host: tuple[HostStep, ...],
                    objective: ObjectiveData | None, read: Read, provision: bool, loading: Loading,
                    seed: int, rejections: Literal["drop", "mask"], budget: SearchBudget,
                    resume: Literal["exact", "non-exact"]) -> Dataset:
    source = train.source
    if not isinstance(source, Phases):
        raise TypeError("phased compilation needs a Phases source")
    _capabilities((train,) if validation is None else (train, validation), read, provision)
    _tokenizer_contract((train,) if validation is None else (train, validation))
    consumed: dict[str, int] = {}
    built: list[tuple[Dataset, int | None]] = []
    previous = 0
    for index, (pipeline, until) in enumerate(zip(source.pipelines, source.until_steps, strict=True)):
        continued = _continued(pipeline, consumed).then(*train.stages)
        loaded = compile(continued, validation=validation if index == 0 else None, host=host,
                         objective=objective, read=read, provision=provision, loading=loading, seed=seed,
                         rejections=rejections, budget=budget, resume=resume)
        if loaded.plan is None or loaded.plan.lowering != "global":
            raise ValueError(f"phase {index} needs a global order; streamed, rejecting and online-packed phases are refused")
        end = None if until is None else until * loaded.batch
        built.append((loaded, end))
        if end is not None:
            _delivered(pipeline, end - previous, consumed)
            previous = end

    def training(partition: DataPartition) -> Iterator[Batch]:
        def global_reader(reader: Reader) -> GlobalReader:
            opened = reader(partition)
            if not isinstance(opened, GlobalReader):
                if isinstance(opened, Closeable):
                    opened.close()
                raise ValueError("a phase must expose a global record-count position")
            return opened

        phases = [(functools.partial(global_reader, loaded.train), end) for loaded, end in built]
        return PhasedStream(phases, loading.stop_seconds)

    return dataclasses.replace(built[0][0], train=training)


def _continued(pipeline: Pipeline, consumed: Mapping[str, int]) -> Pipeline:
    if isinstance(pipeline.source, Phases):
        raise ValueError("nested phases are not an indexed record order")
    if isinstance(pipeline.source, Mixture):
        children = tuple(_continued(child, consumed) for child in pipeline.source.pipelines)
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
    offset = consumed.get(_name(pipeline), 0)
    if not offset:
        return pipeline
    at = next((index + 1 for index, stage in enumerate(pipeline.stages) if isinstance(stage, Repeat)), None)
    if at is None:
        pipeline = pipeline.repeat()
        at = len(pipeline.stages)
    return dataclasses.replace(pipeline, stages=(*pipeline.stages[:at], Slice(slice(offset, None)),
                                                 *pipeline.stages[at:]))


def _delivered(pipeline: Pipeline, count: int, consumed: dict[str, int]) -> None:
    source = pipeline.source
    if isinstance(source, Phases):
        raise ValueError("nested phases have no single corpus share")
    if isinstance(source, Mixture):
        weights = tuple(weight / sum(source.weights) for weight in source.weights)
        for child, taken in zip(source.pipelines, weight_counts(weights, count), strict=True):
            _delivered(child, taken, consumed)
    else:
        name = _name(pipeline)
        consumed[name] = consumed.get(name, 0) + count


def _training_reader(partition: DataPartition, *, pipeline: Pipeline, build: _Build,
                     plan: Plan, host: tuple[HostStep, ...], resume: Literal["exact", "non-exact"],
                     statistics: _Statistics
                     ) -> Iterator[Batch]:
    if plan.volatile and partition.readers > 1:
        raise ValueError("volatile pipeline requires one reader per share until its bytes are persisted")
    for step in host:
        if partition.count > 1 and isinstance(step, SingleReader) and step.single_reader:
            raise ValueError(f"host stage {type(step).__name__} requires one reader for its batch shape")
    opened: Iterator[Batch]
    if build.global_order:
        opened = GlobalStream(functools.partial(build.batches, pipeline, partition), build.batch,
                              json.dumps(plan.order, sort_keys=True), build.loading,
                              plan=plan, resume=resume, endless=_endless(pipeline))
    else:
        own_plan = dataclasses.replace(plan, topology={"shares": partition.count, "share": partition.index,
                                                       "workers": _workers(pipeline, build.loading),
                                                       "row_multiple": partition.row_multiple,
                                                       "cycle_lengths": [stage.cycle_length for stage in _steps(pipeline)
                                                                         if isinstance(stage, Interleave)]})
        opened = _PerShare(functools.partial(build.batches, pipeline, partition), own_plan, resume,
                           endless=_endless(pipeline))
        statistics.observe(opened)
    return _Host(opened, host) if host else opened


def _validation_reader(partition: DataPartition, *, pipeline: Pipeline, build: _Build,
                       host: tuple[HostStep, ...]) -> Iterator[Batch]:
    build = dataclasses.replace(build, global_order=not _per_share(pipeline, build.read))
    opened: Iterator[Batch]
    if not build.global_order:
        build = dataclasses.replace(build, loading=dataclasses.replace(build.loading, workers=0))
    if build.read == "streamed":
        build = dataclasses.replace(build, loading=dataclasses.replace(build.loading, workers=0))
        opened = _CanonicalPass(functools.partial(build.batches, pipeline, fill=False), partition, build.batch)
    else:
        opened = build.batches(pipeline, partition)
    return _Host(opened, host) if host else opened


def _epoch_records(train: Pipeline, read: Read) -> int | None:
    records = None
    if isinstance(train.source, Phases):
        return _epoch_records(train.source.pipelines[0], read)
    if not _per_share(train, read):
        source = train.source
        if isinstance(source, (Mixture, Phases)):
            sizes = [len(child.source.records(read)) for child in source.pipelines
                     if not isinstance(child.source, (Mixture, Phases))]
            weights = source.weights
            if source.tokens is not None:
                facts = [child.source.metadata() for child in source.pipelines if not isinstance(child.source, (Mixture, Phases))]
                weights = token_weights([_name(child) for child in source.pipelines], weights, sizes,
                                        [integer(stated["valid_targets"], "valid_targets") for stated in facts])
            records = max(
                math.ceil(size * sum(weights) / weight)
                for size, weight in zip(sizes, weights, strict=True)
            )
        else:
            records = len(source.records(read))
    return records


@dataclasses.dataclass(frozen=True)
class _ShareBatch:
    partition: DataPartition

    def __call__(self, batch: Batch) -> Batch:
        return {name: value[self.partition.index :: self.partition.count] for name, value in batch.items()}


@dataclasses.dataclass(frozen=True)
class _FillShare:
    partition: DataPartition
    batch: int

    def __call__(self, batch: Batch) -> Batch:
        import jax

        mine = jax.tree.map(lambda value: value[self.partition.index::self.partition.count], batch)
        empty = rows_of(mine) == 0
        if empty:
            mine = jax.tree.map(lambda value: value[:1], batch)
        return FilledBatch(self.partition.rows(self.batch), real=not empty).map(mine)


def _workers(pipeline: Pipeline, loading: Loading) -> int:
    return next((stage.count for stage in pipeline.stages if isinstance(stage, Workers)), loading.worker_count)


def _endless(pipeline: Pipeline) -> bool:
    stages = tuple(_steps(pipeline))
    if any(isinstance(stage, Limit) for stage in stages):
        return False
    repeats = [stage.epochs for stage in stages if isinstance(stage, Repeat)]
    return not repeats or any(epochs is None for epochs in repeats)


def _root_repeat(pipeline: Pipeline) -> bool:
    for stage in pipeline.stages:
        if isinstance(stage, RandomMap):
            return False
        if isinstance(stage, Repeat):
            return True
    return True


def _epochs(pipeline: Pipeline) -> int | None:
    return next((stage.epochs for stage in pipeline.stages if isinstance(stage, Repeat)), None) if _root_repeat(pipeline) else 1


@dataclasses.dataclass(frozen=True)
class _Payload:
    def __call__(self, row: Batch) -> Batch:
        return {name: value for name, value in row.items() if not name.startswith("_dew_")}


@dataclasses.dataclass(frozen=True)
class _LengthAt:
    lengths: pygrain.MapDataset[int]

    def __call__(self, index: int) -> int:
        measured = self.lengths[index]
        if measured is None:
            raise ValueError(f"length metadata is missing index {index}")
        return int(measured)


def objective_steps(pipeline: Pipeline, objective: ObjectiveData, *, train: bool, read: Read) -> Pipeline:
    """Put objective record steps on each corpus before its mixture."""
    if isinstance(pipeline.source, (Mixture, Phases)):
        children = tuple(objective_steps(child, objective, train=train, read=read)
                         for child in pipeline.source.pipelines)
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
    boundary = (Shuffle, Repeat, Group, BatchStage, TokenBudget) if read == "memory" else (Group, BatchStage, TokenBudget)
    at = next((index for index, stage in enumerate(pipeline.stages) if isinstance(stage, boundary)),
              len(pipeline.stages))
    suffix = pipeline.stages[at:]
    pipeline = dataclasses.replace(pipeline, stages=pipeline.stages[:at])
    for step in objective.train if train else objective.evaluate:
        if isinstance(step, Expansion):
            pipeline = pipeline.flat_map(step)
        elif isinstance(step, pygrain.RandomMapTransform):
            pipeline = pipeline.random_map(step)
        else:
            pipeline = pipeline.map(step)
    return dataclasses.replace(pipeline, stages=(*pipeline.stages, *suffix))


def _objective_order(pipeline: Pipeline, objective: ObjectiveData) -> Pipeline:
    def shared(stage: Stage) -> bool:
        return isinstance(stage, RandomMap) and (stage.configured.get("scope") == "run" or
                                                 any(stage.step is step for step in objective.train))

    selected: tuple[Stage, ...] = ()
    source = pipeline.source
    if isinstance(source, (Mixture, Phases)):
        choices = [tuple(stage for stage in child.stages if shared(stage)) for child in source.pipelines]
        if choices and all(stages == choices[0] for stages in choices):
            selected = choices[0]
            children = tuple(dataclasses.replace(child, stages=tuple(stage for stage in child.stages
                                                                     if not shared(stage)))
                             for child in source.pipelines)
            source = dataclasses.replace(source, pipelines=children)
    stages = tuple(stage for stage in pipeline.stages if not shared(stage))
    selected = (*selected, *(stage for stage in pipeline.stages if shared(stage)))
    at = next((index for index, stage in enumerate(stages) if isinstance(stage, (BatchStage, TokenBudget))), len(stages))
    return dataclasses.replace(pipeline, source=source, stages=(*stages[:at], *selected, *stages[at:]))


def _objective_branch(pipeline: Pipeline, objective: ObjectiveData, *, train: bool, read: Read) -> Pipeline:
    boundary = (Shuffle, Repeat, Group, BatchStage, TokenBudget) if read == "memory" else (Group, BatchStage, TokenBudget)
    before = next((index for index, stage in enumerate(pipeline.stages) if isinstance(stage, boundary)),
                  len(pipeline.stages))
    prefix = dataclasses.replace(pipeline, stages=pipeline.stages[:before])
    prefix = objective_steps(prefix, objective, train=train, read=read)
    branch = dataclasses.replace(prefix, stages=(*prefix.stages, *pipeline.stages[before:]))
    if isinstance(branch.source, Mixture):
        children = branch.source.pipelines
        suffixes = [
            tuple(stage for stage in child.stages if isinstance(stage, RandomMap)) for child in children
        ]
        if suffixes and all(suffix == suffixes[0] for suffix in suffixes):
            children = tuple(
                dataclasses.replace(
                    child, stages=tuple(stage for stage in child.stages if not isinstance(stage, RandomMap))
                )
                for child in children
            )
            branch = dataclasses.replace(
                branch,
                source=dataclasses.replace(branch.source, pipelines=children),
                stages=(*suffixes[0], *branch.stages),
            )
    ordering = tuple(stage for stage in branch.stages if isinstance(stage, (Shuffle, Repeat)))
    if ordering and read == "memory":
        first_random = next(
            (index for index, stage in enumerate(branch.stages) if isinstance(stage, RandomMap)), None
        )
        if first_random is not None:
            without = tuple(stage for stage in branch.stages if not isinstance(stage, (Shuffle, Repeat)))
            first_random = next(index for index, stage in enumerate(without) if isinstance(stage, RandomMap))
            branch = dataclasses.replace(
                branch, stages=(*without[:first_random], *ordering, *without[first_random:])
            )
    if train and objective.group is not None and not any(isinstance(stage, TokenBudget) for stage in branch.stages):
        at = next(
            index for index, stage in enumerate(branch.stages) if isinstance(stage, (RandomMap, BatchStage, TokenBudget))
        )
        branch = dataclasses.replace(
            branch, stages=(*branch.stages[:at], Group(objective.group), *branch.stages[at:])
        )
    return branch.map_batch(*(objective.train_batch if train else objective.evaluate_batch))


def _metadata(pipelines: Sequence[Pipeline], read: Read) -> Mapping[str, object]:
    combined: Batch = {}
    slots: dict[str, int] = {}
    for pipeline in pipelines:
        for leaf in _leaves(pipeline):
            source = leaf.source
            if isinstance(source, (Mixture, Phases)):
                continue
            if isinstance(source, _Memory) and not source.held:
                continue
            if read == "memory":
                source.records(read)
            facts = source.metadata()
            for key, value in facts.items():
                if key == "slots":
                    for name, size in record(value, "slots").items():
                        slots[name] = max(slots.get(name, 0), integer(size, f"slots.{name}"))
                elif key not in combined:
                    combined[key] = value
                elif combined[key] != value:
                    combined.pop(key, None)
    return {**combined, "slots": slots} if slots else combined


def iterate(
    pipeline: Pipeline,
    *,
    read: Read = "streamed",
    provision: bool = False,
    loading: Loading = _DEFAULT_LOADING,
    budget: SearchBudget = _DEFAULT_BUDGET,
) -> pygrain.IterDataset:
    """A finite record pass through the same lowering, for publication and export."""
    if any(
        isinstance(stage, (RandomMap, BatchStage, MapBatch, Workers, Repeat)) for stage in _steps(pipeline)
    ):
        raise ValueError("publication needs deterministic, unbatched pipelines without repeat")
    _check(pipeline, read)
    _capabilities((pipeline,), read, provision)
    if any(isinstance(stage, Configured) for stage in _steps(pipeline)):
        pipeline = _configured(pipeline, _metadata((pipeline,), read), read)
        _check(pipeline, read)
    if read == "memory":
        pipeline = _materialize(pipeline, {}, training=False)
    build = _Build(read, loading, 0, 1, budget, not _per_share(pipeline, read), training=False)
    rows, _ = build.records(pipeline, DataPartition())
    return _as_iter(rows, loading)


def validation(pipeline: Pipeline, *, batch: int, objective: ObjectiveData | None = None,
               host: tuple[HostStep, ...] = (), read: Read = "indexed", provision: bool = False,
               loading: Loading = _DEFAULT_LOADING, seed: int = 0,
               budget: SearchBudget = _DEFAULT_BUDGET) -> Reader:
    """Compile one finite evaluation branch without constructing a training stream."""
    if not any(isinstance(stage, BatchStage) for stage in pipeline.stages):
        pipeline = pipeline.batch(batch, drop_remainder=False)
    _check(pipeline, read, validation=True)
    _capabilities((pipeline,), read, provision)
    if objective is not None:
        pipeline = _objective_branch(pipeline, objective, train=False, read=read)
    if any(isinstance(stage, Configured) for stage in _steps(pipeline)):
        pipeline = _configured(pipeline, _metadata((pipeline,), read), read)
    _check(pipeline, read, validation=True)
    if read == "memory":
        pipeline = _materialize(pipeline, {}, training=False)
    build = _Build(read, loading, seed, batch, budget, not _per_share(pipeline, read), training=False)
    if loading.workers == "auto":
        rows, _ = build.records(pipeline, DataPartition())
        build = dataclasses.replace(build, loading=loading.resolved(_as_iter(rows, loading)))
    stages = tuple(stage.step for stage in pipeline.stages if isinstance(stage, MapBatch)) + host
    return functools.partial(_validation_reader, pipeline=pipeline, build=build, host=stages)


def publish(pipeline: Pipeline, cache, *, read: Read = "streamed", provision: bool = False,
            loading: Loading = _DEFAULT_LOADING):
    """Fill and seal a deterministic prefix, then publish its certified generation."""
    from dew.data.cache import Cache

    if not isinstance(cache, Cache):
        raise TypeError("publish needs a Cache destination")
    if isinstance(pipeline.source, (Mixture, Phases)):
        raise ValueError("publish each source prefix before mixing")
    _capabilities((pipeline,), read, provision)
    if any(isinstance(stage, Configured) for stage in pipeline.stages):
        pipeline = _configured(pipeline, _metadata((pipeline,), read), read)
    planned = next((index for index, stage in enumerate(pipeline.stages) if isinstance(stage, Pack)), None)
    builder = None
    if planned is not None:
        from dew.data.packing import PackingPlan

        packing = pipeline.stages[planned]
        if not isinstance(packing, Pack) or planned != len(pipeline.stages) - 1:
            raise ValueError("publication accepts one terminal packing stage after the cached prefix")
        builder = PackingPlan(packing)
        pipeline = dataclasses.replace(pipeline, stages=pipeline.stages[:planned])
    cached = pipeline.cache(cache)
    _check(cached, read)
    manifest = _cache_manifest(pipeline.source, pipeline.stages, read)
    iterator = iter(iterate(cached, read=read, provision=provision, loading=loading))
    try:
        for _ in iterator:
            pass
    finally:
        iterator.close()
    return cache.publish(manifest, pipeline.source, plan=builder)
