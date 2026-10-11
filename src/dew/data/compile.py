"""Compile the same pipeline to an elastic order or an exact per-share reader."""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import math
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
    LengthGroups,
    Loading,
    Position,
    Reader,
    _Grouped,
    _WorkerBatches,
)
from dew.data.mix import Accepted, CorpusCounts, SearchBudget, TokenDeficit, token_weights
from dew.data.pipeline import (
    Artifact,
    BatchStage,
    Expansion,
    Filter,
    FlatMap,
    Group,
    HostStep,
    Interleave,
    Limit,
    Map,
    MapBatch,
    MapWithIndex,
    Mixture,
    ObjectiveData,
    Pack,
    Pipeline,
    Prefetch,
    RandomMap,
    Rebatch,
    Record,
    Rejected,
    Repeat,
    ShardByHost,
    Shuffle,
    Slice,
    Stage,
    Volatile,
    Windows,
    Workers,
)
from dew.data.plan import Manifest, Plan, StepIdentity, fingerprint
from dew.data.sources.base import Read, ShardManifest, Source
from dew.records import integer, record

type Rows = pygrain.MapDataset | pygrain.IterDataset
_log = logging.getLogger(__name__)
_DEFAULT_LOADING = Loading()
_DEFAULT_BUDGET = SearchBudget()


@runtime_checkable
class _Provisionable(Protocol):
    def provision(self) -> Source: ...


def _leaves(pipeline: Pipeline) -> Iterator[Pipeline]:
    if isinstance(pipeline.source, Mixture):
        for child in pipeline.source.pipelines:
            yield from _leaves(child)
    else:
        yield pipeline


def _steps(pipeline: Pipeline) -> Iterator[Stage]:
    if isinstance(pipeline.source, Mixture):
        for child in pipeline.source.pipelines:
            yield from _steps(child)
    yield from pipeline.stages


def _name(pipeline: Pipeline) -> str:
    if pipeline.name is not None:
        return pipeline.name
    return "mixture" if isinstance(pipeline.source, Mixture) else pipeline.source.describe()


def _identity(stage: Stage, *, cache: bool = False) -> StepIdentity:
    if isinstance(stage, Artifact):
        return StepIdentity(f"dew.data.cache.{type(stage.cache).__name__}", stage.cache.url, exact=True)
    if isinstance(stage, (Map, RandomMap, MapWithIndex, FlatMap, MapBatch)):
        return StepIdentity.of(stage.step, stage.identity, cache=cache)
    if isinstance(stage, Filter):
        return StepIdentity.of(stage.predicate, stage.identity, cache=cache)
    return StepIdentity.of(stage)


def _check(pipeline: Pipeline, read: Read, *, validation: bool = False) -> None:
    if isinstance(pipeline.source, Mixture):
        for child in pipeline.source.pipelines:
            _check(child, read, validation=validation)
    ancestry = (
        tuple(stage for child in pipeline.source.pipelines for stage in _steps(child))
        if isinstance(pipeline.source, Mixture)
        else ()
    )
    random = next((_identity(stage).name for stage in ancestry if isinstance(stage, RandomMap)), None)
    indexed, batched, workers, host = read != "streamed", False, False, False
    prefix = list(ancestry)
    for stage in pipeline.stages:
        name = _name(pipeline)
        if isinstance(stage, RandomMap):
            random = _identity(stage).name
            if validation:
                raise ValueError(f"validation pipeline {name} contains random step {random}")
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
        if isinstance(stage, Map) and stage.rejects and read != "memory":
            indexed = False
        if isinstance(stage, BatchStage):
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
            if not isinstance(source, Mixture) and not source.supports(read):
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

    def random_map(self, element: Batch | Rejected, rng: np.random.Generator) -> Batch | Rejected:
        if isinstance(element, Rejected):
            return element
        step = self.stage.step
        return (
            step.random_map(element, rng)
            if isinstance(step, pygrain.RandomMapTransform)
            else step(element, rng)
        )


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
        return (element,) if isinstance(element, Rejected) else self.stage.step.flat_map(record(element, "flat_map input"))


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
                    emitted.extend(_Expand(stage).flat_map(value))
                else:
                    raise ValueError(f"cache prefix contains non-record stage {type(stage).__name__}")
            held = tuple(emitted)
        return held


@dataclasses.dataclass(frozen=True)
class _CachedOutputs:
    def flat_map(self, outputs: Sequence[Batch | Rejected]) -> Sequence[Batch | Rejected]:
        return outputs


@dataclasses.dataclass(frozen=True)
class _Memory:
    original: Source
    held: tuple[Batch, ...]
    measured: np.ndarray | None
    derived: Mapping[str, str]
    facts: Mapping[str, object] = dataclasses.field(default_factory=dict)

    def describe(self) -> str:
        return self.original.describe()

    def snapshot(self) -> Mapping[str, object]:
        return self.original.snapshot()

    def supports(self, read: Read) -> bool:
        return read in ("memory", "indexed")

    def records(self, read: Read) -> tuple[Batch, ...]:
        return self.held

    def metadata(self) -> Mapping[str, object]:
        return {**self.original.metadata(), **self.facts, "accepted": len(self.held)}

    def lengths(self) -> np.ndarray | None:
        return self.measured

    def manifest(self) -> ShardManifest:
        raise ValueError("materialized records are read by index")


def _materialize(pipeline: Pipeline, counts: dict[str, CorpusCounts],
                 groups: LengthGroups | None = None) -> Pipeline:
    groups = next((stage.groups for stage in pipeline.stages if isinstance(stage, Group)), groups)
    if isinstance(pipeline.source, Mixture):
        return _memory_mixture(pipeline, pipeline.source, counts, groups)
    source = pipeline.source
    stored = source.records("memory")
    rows: list[tuple[tuple[int, ...], Record]] = []
    counter = counts.setdefault(_name(pipeline), CorpusCounts(_name(pipeline)))
    counter.attempted = len(stored)
    consumed = 0
    artifact_at = next(
        (index for index, stage in enumerate(pipeline.stages) if isinstance(stage, Artifact)), None
    )
    if artifact_at is not None:
        artifact = pipeline.stages[artifact_at]
        if not isinstance(artifact, Artifact):
            raise RuntimeError("cache stage changed in an immutable pipeline")
        prefix = pipeline.stages[:artifact_at]
        manifest = Manifest.of(source.snapshot(), tuple(_identity(stage, cache=True) for stage in prefix))
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
                rows.append(((index,), value))
    derived: dict[str, str] = {}
    for stage in pipeline.stages[consumed:]:
        if not isinstance(stage, (Map, MapWithIndex, Filter, FlatMap, Pack, Windows)):
            break
        if isinstance(stage, Windows):
            from dew.data.steps.windows import WindowRecords

            windows = WindowRecords(tuple(record(value, "window document") for _, value in rows), stage)
            derived["windows"] = fingerprint([windows.size, stage.seq_len, stage.stride])
            rows = [((index,), windows[index]) for index in range(len(windows))]
        elif isinstance(stage, Pack):
            from dew.data.tokens import PackedWindows, first_fit

            documents = tuple(_Payload()(record(value, "packing document")) for _, value in rows)
            sizes = np.asarray([len(value[stage.fields[0]]) for value in documents], np.int64)
            packed = PackedWindows(
                pygrain.MapDataset.source(documents), sizes, stage.seq_len, stage.bins, source.describe()
            )
            plan, starts = first_fit(sizes, stage.seq_len, stage.bins)
            derived["packing"] = fingerprint([plan.tolist(), starts.tolist()])
            rows = [((index,), packed[index]) for index in range(len(packed))]
        else:
            expanded: list[tuple[tuple[int, ...], Record]] = []
            for index, (identifier, value) in enumerate(rows):
                if isinstance(stage, FlatMap):
                    outputs = stage.step.flat_map(record(value, "expansion input"))
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
    return _memory_order(pipeline, rows, consumed, source, len(stored), counter, derived, groups)


def _memory_mixture(pipeline: Pipeline, mixture: Mixture, counts: dict[str, CorpusCounts],
                    groups: LengthGroups | None) -> Pipeline:
    children = tuple(_materialize(child, counts, groups) for child in mixture.pipelines)
    kept = tuple((child, weight) for child, weight in zip(children, mixture.weights, strict=True)
                 if isinstance(child.source, Mixture) or len(child.source.records("memory")))
    omitted = [_name(child) for child in children if all(child is not accepted for accepted, _ in kept)]
    if omitted:
        _log.warning("left out of the mixture: %s (zero accepted rows)", ", ".join(omitted))
    if not kept:
        raise ValueError(f"no accepted row of {', '.join(_name(child) for child in children)}; raise the row budgets")
    if len(kept) == 1:
        child = kept[0][0]
        return dataclasses.replace(child, stages=(*child.stages, *pipeline.stages))
    children, weights = zip(*kept, strict=True)
    return dataclasses.replace(pipeline, source=dataclasses.replace(mixture, pipelines=tuple(children),
                                                                   weights=tuple(weights)))


def _memory_order(pipeline: Pipeline, collected: Sequence[tuple[tuple[int, ...], Record]], consumed: int,
                  source: Source, inputs: int, counter: CorpusCounts, derived: dict[str, str],
                  groups: LengthGroups | None) -> Pipeline:
    rows = [(identifier, record(value, "materialized record")) for identifier, value in collected]
    remaining = pipeline.stages[consumed:]
    lengths = (
        np.asarray([groups.length(value) for _, value in rows], np.int64)
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
    counter.truncated = sum(bool(value.get("_dew_truncated", False)) for _, value in rows)
    facts: Batch = {}
    if any(isinstance(stage, (Pack, Windows)) for stage in pipeline.stages[:consumed]):
        from dew.data.steps.text import ValidTargets

        targets = ValidTargets()
        kept = []
        for identifier, value in rows:
            count = targets(value)
            if count:
                kept.append((identifier, value))
            else:
                counter.rejected["zero_targets"] += 1
        rows = kept
        facts = {"windows": len(rows), "valid_targets": sum(targets(value) for _, value in rows)}
        if facts["valid_targets"] == 0:
            raise ValueError(f"corpus {_name(pipeline)} has zero valid target tokens")
        derived["valid_targets"] = fingerprint(facts)
        counter.accepted = len(rows)
    derived["accepted_ids"] = fingerprint([list(identifier) for identifier, _ in rows])
    return dataclasses.replace(
        pipeline,
        source=_Memory(source, tuple(value for _, value in rows), lengths, derived, facts),
        stages=remaining,
    )


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


class _PerShare(Forwarding):
    @property
    def endless(self) -> bool:
        return True

    def __init__(
        self,
        open_reader: Callable[[], pygrain.DatasetIterator[Batch]],
        plan: Plan,
        resume: Literal["exact", "non-exact"],
    ):
        self._open, self._plan = open_reader, plan
        self._resume: Literal["exact", "non-exact"] = resume
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

    def close(self) -> None:
        if self._read is not None:
            self._read.close()
            self._read = None


def _per_share(pipeline: Pipeline, read: Read) -> bool:
    if read == "streamed":
        return True
    if isinstance(pipeline.source, Mixture):
        if pipeline.source.tokens is not None:
            for child in pipeline.source.pipelines:
                if isinstance(child.source, Mixture) or "valid_targets" not in child.source.metadata():
                    return True
        if any(_per_share(child, read) for child in pipeline.source.pipelines):
            return True
    return any(
        isinstance(stage, (Filter, FlatMap, Pack, Windows, Interleave, Artifact))
        or (isinstance(stage, Map) and stage.rejects)
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
        if isinstance(source, Mixture):
            built = [self.records(child, partition) for child in source.pipelines]
            parents, lengths = zip(*built, strict=True)
            if all(isinstance(parent, pygrain.MapDataset) for parent in parents):
                mapped = [parent for parent in parents if isinstance(parent, pygrain.MapDataset)]
                weights = source.weights
                if source.tokens is not None:
                    metadata = [
                        child.source.metadata()
                        for child in source.pipelines
                        if not isinstance(child.source, Mixture)
                    ]
                    weights = token_weights(
                        [_name(child) for child in source.pipelines],
                        weights,
                        [integer(stated["windows"], "windows") for stated in metadata],
                        [integer(stated["valid_targets"], "valid_targets") for stated in metadata],
                    )
                measured = [length for length in lengths if length is not None]
                return pygrain.MapDataset.mix(mapped, weights).seed(self.seed), (
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
            return pygrain.IterDataset.mix(streams, source.weights), None
        if self.read == "streamed":
            from dew.data.shards import streamed

            return streamed(
                source.manifest(),
                partition=partition,
                seed=self.seed if self.training else None,
                cycle_length=1,
                loading=self.loading,
            ), None
        stored = source.records(self.read)
        rows = pygrain.MapDataset.source(stored).seed(self.seed)
        lengths = source.lengths()
        measured = None if lengths is None else pygrain.MapDataset.source(lengths.tolist())
        if not self.global_order and self.training:
            rows = rows[partition.index :: partition.count]
            measured = None if measured is None else measured[partition.index :: partition.count]
        return rows, measured

    def records(self, pipeline: Pipeline, partition: DataPartition) -> tuple[Rows, pygrain.MapDataset | None]:
        source = pipeline.source
        artifact_at = next(
            (index for index, stage in enumerate(pipeline.stages) if isinstance(stage, Artifact)), None
        )
        consumed = 0
        if artifact_at is not None and not isinstance(source, Mixture):
            artifact = pipeline.stages[artifact_at]
            if not isinstance(artifact, Artifact):
                raise RuntimeError("cache stage changed while compiling an immutable pipeline")
            prefix = pipeline.stages[:artifact_at]
            if any(not isinstance(stage, (Map, MapWithIndex, Filter, FlatMap)) for stage in prefix):
                raise ValueError(
                    f"cache prefix in {_name(pipeline)} must contain only deterministic record steps"
                )
            manifest = Manifest.of(source.snapshot(), tuple(_identity(stage, cache=True) for stage in prefix))
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
                )
            else:
                rows = artifact.cache.mapped(
                    pygrain.MapDataset.source(source.records(self.read)).seed(self.seed),
                    manifest,
                    source,
                    transform=transform,
                )
            rows = gx.FlatMapIterDataset(_as_iter(rows, self.loading), _CachedOutputs())
            lengths = None
            consumed = artifact_at + 1
        else:
            rows, lengths = self.source(pipeline, partition)
        for stage in pipeline.stages[consumed:]:
            if isinstance(stage, (BatchStage, Workers, MapBatch, Rebatch)):
                break
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
            rows = gx.LimitIterDataset(_as_iter(rows, self.loading), stage.count)
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
            if isinstance(pipeline.source, Mixture):
                raise ValueError("cache belongs on each corpus before mix")
            prefix = pipeline.stages[: pipeline.stages.index(stage)]
            manifest = Manifest.of(
                pipeline.source.snapshot(), tuple(_identity(step, cache=True) for step in prefix)
            )
            if not isinstance(rows, pygrain.MapDataset):
                raise ValueError("streamed cache must be applied to its shard reader before interleave")
            rows = stage.cache.mapped(rows, manifest, pipeline.source)
        elif isinstance(stage, Interleave):
            if self.read != "streamed":
                raise ValueError("interleave needs a shard-manifest source")
        return rows, lengths

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
        elif isinstance(stage, MapWithIndex):
            rows = rows.map_with_index(stage.step)
        elif isinstance(stage, Filter):
            rows = Accepted(_as_iter(rows, self.loading).map(_Predicate(stage)),
                            _name(pipeline), budget=self.budget)
            lengths = None
        elif isinstance(stage, FlatMap):
            rows = gx.FlatMapIterDataset(_as_iter(rows, self.loading), _Expand(stage))
            lengths = None
        return rows, lengths

    def batches(
        self, pipeline: Pipeline, partition: DataPartition, offset: int = 0
    ) -> pygrain.DatasetIterator[Batch]:
        canonical = not self.training and not self.global_order
        rows, _ = self.records(pipeline, DataPartition() if canonical else partition)
        batch_stage = next(stage for stage in pipeline.stages if isinstance(stage, BatchStage))
        shared_size = partition.rows(batch_stage.size)
        size = batch_stage.size if canonical else shared_size
        if isinstance(rows, pygrain.MapDataset):
            if self.training and not any(isinstance(stage, Repeat) for stage in _steps(pipeline)):
                rows = rows.repeat(None)
            if self.global_order or not self.training:
                rows = rows[offset + partition.index :: partition.count]
            if self.loading.workers:
                rows = _WorkerBatches(rows, size, self.loading.worker_count, remainder=not self.training)
        stream = _as_iter(rows, self.loading).map(_Payload()).batch(
            size,
            drop_remainder=batch_stage.drop_remainder if self.training else False,
            batch_fn=batch_stage.batch_fn,
        )
        if not self.training:
            stream = stream.map(FilledBatch(size))
            if canonical and partition.count > 1:
                stream = stream.map(_ShareBatch(partition))
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


def _order(pipeline: Pipeline) -> Mapping[str, object]:
    source = pipeline.source
    children = (
        {
            "corpora": [_order(child) for child in source.pipelines],
            "weights": list(source.weights),
            "shares": "tokens" if source.tokens is not None else "accepted rows",
        }
        if isinstance(source, Mixture)
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
    }


def _derived(pipeline: Pipeline, seed: int) -> Mapping[str, str]:
    arrays: dict[str, str] = {}
    for index, leaf in enumerate(_leaves(pipeline)):
        source = leaf.source
        if isinstance(source, Mixture):
            continue
        if isinstance(source, _Memory):
            arrays.update({f"{index}/{key}": value for key, value in source.derived.items()})
        lengths = source.lengths()
        if lengths is not None:
            arrays[f"{index}/lengths"] = fingerprint(lengths)
        if source.supports("indexed"):
            arrays[f"{index}/index"] = fingerprint([len(source.records("indexed")), seed])
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
    objective: ObjectiveData | Callable[[Mapping[str, object]], ObjectiveData] | None = None,
    read: Read = "indexed",
    provision: bool = False,
    loading: Loading = _DEFAULT_LOADING,
    seed: int = 0,
    rejections: Literal["drop", "mask"] = "drop",
    budget: SearchBudget = _DEFAULT_BUDGET,
    resume: Literal["exact", "non-exact"] = "exact",
) -> Dataset:
    """Check every leaf before opening readers, then build both branches."""
    if read not in ("memory", "indexed", "streamed"):
        raise ValueError(f"unknown read mode {read!r}")
    if rejections == "mask":
        raise ValueError(
            "rejections='mask' needs an objective proven mask-safe; no objective currently declares it"
        )
    pipelines = (train,) if validation is None else (train, validation)
    for pipeline in pipelines:
        _check(pipeline, read, validation=pipeline is validation)
    _capabilities(pipelines, read, provision)
    if objective is not None:
        shaped = objective if isinstance(objective, ObjectiveData) else objective(_metadata(pipelines, read))
        train = _objective_branch(train, shaped, train=True)
        validation = None if validation is None else _objective_branch(validation, shaped, train=False)
        _check(train, read)
        if validation is not None:
            _check(validation, read, validation=True)
    batches = [stage.size for stage in train.stages if isinstance(stage, BatchStage)]
    if len(batches) != 1:
        raise ValueError("compile needs one batch(global_size) stage on the training pipeline")
    batch = batches[0]
    counts: dict[str, CorpusCounts] = {}
    original = train
    if read == "memory":
        train = _materialize(train, counts)
        validation = None if validation is None else _materialize(validation, {})
    global_order = not _per_share(train, read)
    manifest = Manifest.of(
        {
            "sources": [
                leaf.source.snapshot() for leaf in _leaves(original) if not isinstance(leaf.source, Mixture)
            ]
        },
        tuple(_identity(stage) for stage in _steps(original) if not isinstance(stage, (Workers, Prefetch))),
    )
    plan = Plan(
        "global" if global_order else "per-share",
        manifest,
        _order(train),
        _derived(train, seed),
        cache=tuple(stage.cache.url for stage in _steps(original) if isinstance(stage, Artifact)),
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

    def training(partition: DataPartition) -> Iterator[Batch]:
        if plan.volatile and partition.readers > 1:
            raise ValueError("volatile pipeline requires one reader per share until its bytes are persisted")
        opened: Iterator[Batch]
        if global_order:
            opened = GlobalStream(
                functools.partial(build.batches, train, partition),
                batch,
                json.dumps(plan.order, sort_keys=True),
                loading,
                plan=plan,
                resume=resume,
            )
        else:
            own_plan = dataclasses.replace(
                plan,
                topology={"shares": partition.count, "share": partition.index, "workers": loading.workers},
            )
            opened = _PerShare(functools.partial(build.batches, train, partition), own_plan, resume)
        return _Host(opened, train_host) if train_host else opened

    held_reader: Reader | None = None
    if validation is not None:
        held_build = dataclasses.replace(build, training=False, global_order=not _per_share(validation, read))
        held_host = tuple(stage.step for stage in validation.stages if isinstance(stage, MapBatch)) + host

        def validating(partition: DataPartition) -> Iterator[Batch]:
            held = held_build.batches(validation, partition)
            return _Host(held, held_host) if held_host else held

        held_reader = validating
    records = None
    if global_order:
        source = train.source
        if isinstance(source, Mixture):
            sizes = [len(child.source.records(read)) for child in source.pipelines
                     if not isinstance(child.source, Mixture)]
            weights = source.weights
            if source.tokens is not None:
                facts = [child.source.metadata() for child in source.pipelines if not isinstance(child.source, Mixture)]
                weights = token_weights([_name(child) for child in source.pipelines], weights, sizes,
                                        [integer(stated["valid_targets"], "valid_targets") for stated in facts])
            records = max(
                math.ceil(size * sum(weights) / weight)
                for size, weight in zip(sizes, weights, strict=True)
            )
        else:
            records = len(source.records(read))
    return Dataset(training, held_reader, records, batch, counts=counts, plan=plan)


@dataclasses.dataclass(frozen=True)
class _ShareBatch:
    partition: DataPartition

    def __call__(self, batch: Batch) -> Batch:
        return {name: value[self.partition.index :: self.partition.count] for name, value in batch.items()}


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
        return measured


def objective_steps(pipeline: Pipeline, objective: ObjectiveData, *, train: bool) -> Pipeline:
    """Put objective record steps on each corpus before its mixture."""
    if isinstance(pipeline.source, Mixture):
        children = tuple(
            objective_steps(child, objective, train=train) for child in pipeline.source.pipelines
        )
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
    at = next(
        (
            index
            for index, stage in enumerate(pipeline.stages)
            if isinstance(stage, (Shuffle, Repeat, Group, BatchStage))
        ),
        len(pipeline.stages),
    )
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


def _objective_branch(pipeline: Pipeline, objective: ObjectiveData, *, train: bool) -> Pipeline:
    before = next(
        (
            index
            for index, stage in enumerate(pipeline.stages)
            if isinstance(stage, (Shuffle, Repeat, Group, BatchStage))
        ),
        len(pipeline.stages),
    )
    prefix = dataclasses.replace(pipeline, stages=pipeline.stages[:before])
    prefix = objective_steps(prefix, objective, train=train)
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
    if ordering:
        first_random = next(
            (index for index, stage in enumerate(branch.stages) if isinstance(stage, RandomMap)), None
        )
        if first_random is not None:
            without = tuple(stage for stage in branch.stages if not isinstance(stage, (Shuffle, Repeat)))
            first_random = next(index for index, stage in enumerate(without) if isinstance(stage, RandomMap))
            branch = dataclasses.replace(
                branch, stages=(*without[:first_random], *ordering, *without[first_random:])
            )
    if train and objective.group is not None:
        at = next(
            index for index, stage in enumerate(branch.stages) if isinstance(stage, (RandomMap, BatchStage))
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
            if isinstance(source, Mixture):
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
    if read == "memory":
        pipeline = _materialize(pipeline, {})
    build = _Build(read, loading, 0, 1, budget, not _per_share(pipeline, read), training=False)
    rows, _ = build.records(pipeline, DataPartition())
    return _as_iter(rows, loading)
