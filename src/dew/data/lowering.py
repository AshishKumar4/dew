"""The Grain lowering of an analyzed pipeline.

Each stage kind has one definition here, which every read applies and the
memory executor (`dew.data.materialize`) calls for the stages it runs over
the whole source: `_outputs` for a record step, `_reordered` for an order
stage over indexed rows, `_TokenSplit` for a token split, `_packed` for
Grain's packers, and `_fill` with `_CachedOutputs` for opening a cache and
reading what it holds.

`_Build` lowers a pipeline's stages in the order they are written, as the
analysis (`dew.data.analysis`) says to cut them. A global order is built
indexed and whole, and `batches` cuts it into shares just before the batch;
a per-share order is cut at its read and streams this share's accepted
rows. A row carries its source position in a private field from the read to
the last `map_with_index`, which is the only step that reads it.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

import grain.experimental as gx
import grain.python as pygrain
import numpy as np

from dew.data.analysis import (
    _LOCAL,
    _at,
    _cache_manifest,
    _drops,
    _identity,
    _in_memory,
    _Lowering,
    _name,
    _until,
)
from dew.data.dataset import (
    SHARED_BYTES,
    Batch,
    DataPartition,
    FilledBatch,
    Loading,
    Records,
    record_batch,
    stacked,
    take_batch,
)
from dew.data.mixing import (
    Accepted,
    Chunked,
    CorpusCounts,
    PrimedMix,
    SearchBudget,
    TokenDeficit,
    primed_maps,
    token_weights,
)
from dew.data.pipeline import (
    Artifact,
    Batched,
    BatchStage,
    Expansion,
    Filter,
    FlatMap,
    Interleave,
    Limit,
    Map,
    MapBatch,
    MapWithIndex,
    Mixture,
    Native,
    Pack,
    Phases,
    Pipeline,
    Prefetch,
    RandomMap,
    Rebatch,
    Record,
    Rejected,
    Repeat,
    Shuffle,
    Slice,
    Stage,
    TokenSplit,
    Volatile,
    Windows,
    Workers,
)
from dew.data.plan import Manifest
from dew.data.readers import _FillShare
from dew.data.sources.base import Read, Shard, ShardManifest, Source
from dew.data.steps.generic import Fields
from dew.data.steps.text import ValidTargets
from dew.data.text import Chunks
from dew.data.tokens import _WrappingDataset
from dew.records import integer, record

if TYPE_CHECKING:
    from dew.data.cache import Cache

type Rows = pygrain.MapDataset | pygrain.IterDataset


@runtime_checkable
class _Locality(Protocol):
    """A source that states the rows of each of its physical shards and
    whether they are remote, which a block shuffle reads."""

    def locality(self) -> tuple[tuple[int, ...], bool]: ...


@runtime_checkable
class _Materialized(Protocol):
    """Rows a memory read holds, which state the source position each came
    from (`dew.data.materialize`)."""

    identifiers: tuple[tuple[int, ...], ...]


def _outputs(
    stage: Stage, position: int | None, value: Record, rng: np.random.Generator | None = None,
    *, counts: CorpusCounts | None = None,
) -> tuple[Record, ...]:
    """What a record stage makes of one row, the single definition the cache
    fill, memory materialization and Grain lowering all apply: its outputs,
    with a `Rejected` marker in place of each row it drops. A marker passes
    through and a missing row is rejected. `position` is the row's source
    position, as `Pipeline.map_with_index` defines it. When `counts` is given,
    a chunk expansion counts its document and fan-out as it runs.
    """
    if isinstance(value, Rejected):
        return (value,)
    if value is None:
        return (Rejected("missing"),)
    if isinstance(stage, Filter):
        return (value if stage.predicate(record(value, "filter input")) else Rejected(stage.reason),)
    if isinstance(stage, FlatMap):
        element = record(value, "flat_map input")
        outputs = stage.step.flat_map(element) if isinstance(stage.step, Expansion) else stage.step(element)
        outputs = tuple(outputs)
        if counts is not None and isinstance(stage.step, Chunks):
            counts.expanded(outputs)
        return outputs or (Rejected("empty_expansion"),)
    if isinstance(stage, RandomMap):
        if rng is None:
            raise ValueError(f"random step {_identity(stage).name} needs a record rng")
        step = stage.step
        output = (step.random_map(value, rng) if isinstance(step, pygrain.RandomMapTransform)
                  else step(record(value, "random map input"), rng))
    elif isinstance(stage, Map):
        step = stage.step
        output = step.map(value) if isinstance(step, pygrain.MapTransform) else step(record(value, "map"))
    elif isinstance(stage, MapWithIndex):
        if position is None:
            raise ValueError(f"map_with_index step {_identity(stage).name} read a row without its position")
        output = stage.step(position, record(value, "indexed map input"))
    else:
        raise ValueError(f"{type(stage).__name__} is not a record stage")
    if isinstance(output, Rejected) and not stage.rejects:
        raise ValueError(f"step {_identity(stage).name} dropped a record ({output.reason}) under map, which "
                         "keeps every record; apply a step that drops records with filter_map")
    return (output,)


_POSITION = "_dew_position"


def _placed(value: Record, position: int) -> Record:
    return value if value is None or isinstance(value, (bytes, Rejected)) else {**value, _POSITION: position}


def _position(value: Record) -> int | None:
    if value is None or isinstance(value, (bytes, Rejected)) or _POSITION not in value:
        return None
    return int(value[_POSITION])


def _unplaced(value: Record) -> Record:
    if value is None or isinstance(value, (bytes, Rejected)) or _POSITION not in value:
        return value
    return {name: column for name, column in value.items() if name != _POSITION}


@dataclasses.dataclass(frozen=True)
class _Placed:
    """The source position of each row a read numbers from `start`, or of each
    output a cache holds for it; a materialized source states its rows'
    positions (`identifiers`). An indexed root numbers `rows` rows, and Grain
    counts its index on across the epochs a repeat reads, so the position is
    that of the row the index reads in its epoch."""

    start: int = 0
    identifiers: tuple[tuple[int, ...], ...] = ()
    rows: int | None = None

    def __call__(self, index: int, value: Record | tuple[Record, ...]) -> Record | tuple[Record, ...]:
        index = index if self.rows is None else index % self.rows
        position = self.identifiers[index][0] if self.identifiers else self.start + index
        if isinstance(value, tuple):
            return tuple(_placed(output, position) for output in value)
        return _placed(value, position)


@dataclasses.dataclass(frozen=True)
class _Rest:
    """The first batch of a resumed read without its first `read` rows, which the run read before
    it saved; the batches after it whole."""

    read: int

    def __call__(self, index: int, batch: Batch) -> Batch:
        return take_batch(batch, slice(self.read, None)) if index == 0 else batch


@dataclasses.dataclass(frozen=True)
class _PlacedShards:
    reader: Callable[[Shard], pygrain.IterDataset]
    starts: Mapping[str, int]

    def __call__(self, shard: Shard) -> pygrain.IterDataset:
        return self.reader(shard).map_with_index(_Placed(self.starts[shard.id]))


def _positioned(manifest: ShardManifest) -> ShardManifest:
    rows = [shard.rows for shard in manifest.shards if shard.rows is not None]
    if len(rows) != len(manifest.shards):
        raise ValueError("map_with_index over a streamed read needs every shard to state its rows")
    starts = np.cumsum([0, *rows[:-1]])
    placed = {shard.id: int(start) for shard, start in zip(manifest.shards, starts, strict=True)}
    return dataclasses.replace(manifest, reader=_PlacedShards(manifest.reader, placed))


@dataclasses.dataclass(frozen=True)
class _Rows(gx.FlatMapTransform):
    """Grain's form of a record stage: `_outputs` over each element, keeping
    the source position an element carries out of the step's sight."""

    stage: Map | RandomMap | MapWithIndex | Filter | FlatMap
    counts: CorpusCounts | None = None

    def __call__(self, element: Record, rng: np.random.Generator | None = None) -> Record:
        return self.flat_map(element, rng)[0]

    def flat_map(self, element: Record, rng: np.random.Generator | None = None) -> tuple[Record, ...]:
        position = _position(element)
        if position is None:
            return _outputs(self.stage, None, element, rng, counts=self.counts)
        outputs = _outputs(self.stage, position, _unplaced(element), rng, counts=self.counts)
        return tuple(_placed(output, position) for output in outputs)


def _expanded(rows: pygrain.IterDataset, stage: FlatMap, name: str) -> pygrain.IterDataset:
    return (Chunked(rows, name, functools.partial(_Rows, stage)) if isinstance(stage.step, Chunks)
            else gx.FlatMapIterDataset(rows, _Rows(stage)))


@dataclasses.dataclass(frozen=True)
class _Groups:
    """A `map_batches` step over one group: its records, as Grain's flat map takes them."""

    stage: Batched
    max_fan_out: int = 1 << 20

    def flat_map(self, group: list[Record]) -> Sequence[Record]:
        rows = [record(value, "batched input") for value in group]
        columns = {name: [row[name] for row in rows] for name in rows[0]}
        made = dict(self.stage.step(columns))
        counts = {len(values) for values in made.values()}
        if len(counts) > 1:
            raise ValueError(f"a batched step returned columns of different lengths, {sorted(counts)}")
        count = counts.pop() if counts else 0
        if self.stage.same_rows:
            if count != len(rows):
                raise ValueError(f"map(batched=True) keeps the {len(rows)} rows it reads, and its step "
                                 f"returned {count}; flat_map(batched=True) changes the count")
            return [{**row, **{name: values[index] for name, values in made.items()}}
                    for index, row in enumerate(rows)]
        return [{name: values[index] for name, values in made.items()} for index in range(count)]


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
        position = _position(raw)
        if position is not None:
            index, raw = position, _unplaced(raw)
        held: tuple[Record, ...] = (raw,)
        for stage in self.stages:
            held = tuple(output for value in held for output in _outputs(stage, index, value))
        return held


@dataclasses.dataclass(frozen=True)
class _ShardSteps:
    """A shard reader that runs the record steps written before an interleave
    inside each shard, after flattening a cache's outputs, so the interleave
    flattens the shards where it stands. Rejected rows stay marked for the
    accepting reader after it."""

    reader: Callable[[Shard], pygrain.IterDataset]
    stages: tuple[Map | MapWithIndex | Filter | FlatMap, ...]
    cached: bool
    name: str

    def __call__(self, shard: Shard) -> pygrain.IterDataset:
        rows = self.reader(shard)
        rows = gx.FlatMapIterDataset(rows, _CachedOutputs()) if self.cached else rows
        for stage in self.stages:
            rows = (_expanded(rows, stage, self.name) if isinstance(stage, FlatMap)
                    else rows.map(_Rows(stage)))
        return rows


@dataclasses.dataclass(frozen=True)
class _CachedOutputs:
    """What a cache holds for one input, as rows: its outputs, a missing one
    and an empty expansion each rejected, which every read of a cache uses."""

    def flat_map(self, outputs: Sequence[Record]) -> tuple[Record, ...]:
        held = tuple(Rejected("missing") if value is None else value for value in outputs)
        return held if held else (Rejected("empty_expansion"),)


def _filled(stage: Artifact) -> Cache:
    """The cache an artifact stage fills. A generation is read as a source,
    which `_promoted` puts in its place before anything opens."""
    from dew.data.cache import Cache

    if not isinstance(stage.cache, Cache):
        raise ValueError(f"generation {stage.cache.url} must be promoted to a source before it is read")
    return stage.cache


def _fill(pipeline: Pipeline, source: Source, at: int, read: Read) -> tuple[Cache, Manifest, _Prefix]:
    """The cache stage at `at`, the manifest its fill is keyed by and the
    record steps before it that fill it, from which memory, indexed and
    streamed reads all open the cache."""
    artifact, prefix = pipeline.stages[at], pipeline.stages[:at]
    if not isinstance(artifact, Artifact):
        raise ValueError(f"stage {at} of {_name(pipeline)} is not a cache")
    return _filled(artifact), _cache_manifest(source, prefix, read), _Prefix(prefix)


def _token_spans(documents: Iterable[Record], split: TokenSplit) -> list[tuple[int, int, int]]:
    """The documents a token split keeps, in the order it reads them, as
    (document, first, last) token spans. The held side is the first
    `fraction` of all their tokens and the trained side the rest, so the two
    are disjoint and together are the corpus."""
    sizes = []
    for document in documents:
        columns = [np.asarray(record(document, "token split document")[name]) for name in split.fields]
        if any(column.ndim != 1 for column in columns):
            raise ValueError("token split fields must be one-dimensional token arrays")
        sizes.append(len(columns[0]))
    total = sum(sizes)
    boundary = min(int(total * split.fraction), max(0, total - 1))
    start, stop = (0, boundary) if split.held else (boundary, total)
    spans = []
    offset = 0
    for index, size in enumerate(sizes):
        first, last = max(0, start - offset), min(size, stop - offset)
        if first < last:
            spans.append((index, first, last))
        offset += size
    return spans


class _TokenSplit(_WrappingDataset):
    """A token split of a finite indexed order, such as a shuffled corpus:
    the spans `_token_spans` keeps, computed once over the order as it reads."""

    def __init__(self, parent: pygrain.MapDataset, split: TokenSplit):
        super().__init__(parent)
        self._split = split
        self.spans = _token_spans((parent[index] for index in range(len(parent))), split)

    def __len__(self) -> int:
        return len(self.spans)

    def record(self, index: int) -> Batch:
        document, first, last = self.spans[index]
        return {
            key: np.asarray(column)[first:last] if key in self._split.fields else column
            for key, column in record(self._parent[document], "token split document").items()
        }


def _dies_with_parent(index: int, count: int) -> None:
    """Ends this Grain worker once the process that started it has ended, however it ended.

    A worker reads until its parent sets a stop event, so a parent killed, timed out or crashed
    left its workers reading into a queue no one empties: 81 of them on one host, 47 GB between
    them, a day later. A spawned worker holds its parent's sentinel, a pipe only the parent
    writes, which reads as ended once the parent has exited; a thread waits on it and ends the
    worker."""
    import multiprocessing.connection
    import os
    import threading

    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        multiprocessing.connection.wait([parent.sentinel])
        os._exit(1)

    watcher = f"dew worker {index} of {count} watching its parent"
    threading.Thread(target=watch, name=watcher, daemon=True).start()


def _as_iter(rows: Rows, loading: Loading) -> pygrain.IterDataset:
    return (
        rows.to_iter_dataset(pygrain.ReadOptions(loading.threads, loading.read_buffer))
        if isinstance(rows, pygrain.MapDataset)
        else rows
    )


def _packed(documents: pygrain.IterDataset, stage: Pack) -> pygrain.IterDataset:
    """`documents` packed by Grain's own packer for the stage's policy."""
    sizes = dict.fromkeys(stage.fields, stage.seq_len) | dict(stage.along)
    if stage.policy == "concat":
        return gx.ConcatThenSplitIterDataset(documents, length_struct=sizes)
    packer = gx.FirstFitPackIterDataset if stage.policy == "first-fit" else gx.BestFitPackIterDataset
    # Fields cut alike with the first share its segments and positions, as the indexed planner's
    # windows do (`packed_window`), so Grain writes no copy of them for the rest.
    return packer(documents, length_struct=sizes, num_packing_bins=stage.bins, shuffle_bins=False,
                  max_sequences_per_bin=stage.segments, meta_features=stage.fields[1:])


@overload
def _repeated(rows: pygrain.MapDataset, epochs: int | None, name: str, loading: Loading, *,
              reseed: bool = True) -> pygrain.MapDataset: ...


@overload
def _repeated(rows: Rows, epochs: int | None, name: str, loading: Loading, *,
              reseed: bool = True) -> Rows: ...


def _repeated(rows: Rows, epochs: int | None, name: str, loading: Loading, *, reseed: bool = True) -> Rows:
    """`rows` read `epochs` times over, or endlessly, by Grain's repeat. An indexed repeat draws
    each pass afresh (`reseed`) as Grain's does, unless it stands for a repeat of a stream, whose
    passes replay the first. An endless repeat of a pass that holds or yields nothing would read
    nothing forever, so it is refused: an indexed pass by its length, a streamed one when a pass
    ends without a row (`_Passes`), as a pass of batches does when its records fill no batch."""
    if isinstance(rows, pygrain.MapDataset):
        if epochs is None and not len(rows):
            raise ValueError(f"a pass of {name} holds no rows, so repeating it reads nothing forever")
        return rows.repeat(epochs, reseed_each_epoch=reseed)
    parent = _as_iter(rows, loading)
    return gx.RepeatIterDataset(parent if epochs is not None else _Passes(parent, name), epochs)


class _Passes(pygrain.IterDataset):
    """A stream an endless repeat reads pass after pass, refusing a pass that yields no row."""

    def __init__(self, parent: pygrain.IterDataset, name: str):
        super().__init__(parent)
        self._rows, self._name = parent, name

    def __iter__(self) -> pygrain.DatasetIterator:
        return _PassesIterator(self._rows.__iter__(), self._name)


class _PassesIterator(pygrain.DatasetIterator):
    """Its parent's rows and position. A pass that ends without a row is refused; the repeat
    restarts a pass by restoring the parent right after one ends, and any other restore lands
    inside a pass the run already read from."""

    def __init__(self, parent: pygrain.DatasetIterator, name: str):
        super().__init__(parent)
        self._rows, self._name = parent, name
        self._yielded = self._ended = False

    def __next__(self):
        try:
            value = next(self._rows)
        except StopIteration:
            if not self._yielded:
                raise ValueError(f"a pass of {self._name} yields no rows, so repeating it reads nothing "
                                 "forever") from None
            self._ended = True
            raise
        self._yielded = True
        return value

    def get_state(self):
        return self._rows.get_state()

    def set_state(self, state) -> None:
        self._rows.set_state(state)
        self._yielded, self._ended = not self._ended, False


def _reordered(stage: Shuffle | Slice | Limit, rows: pygrain.MapDataset) -> pygrain.MapDataset:
    """An order stage over an indexed dataset."""
    if isinstance(stage, Shuffle):
        if stage.window is None:
            return rows.shuffle(stage.seed)
        return gx.WindowShuffleMapDataset(rows, window_size=stage.window, seed=stage.seed)
    if isinstance(stage, Slice):
        return rows.slice(stage.selection)
    return rows[: min(stage.count, len(rows))]


_UNPLACED = (Prefetch, Workers, MapBatch)
"""The stages after a batch that hold no place in a global order: they buffer, move or host its
batches, so a resume cuts the batches before them (`_Build.batched`)."""


@dataclasses.dataclass(frozen=True)
class _Build:
    read: Read
    loading: Loading
    seed: int
    batch: int
    budget: SearchBudget
    lowering: _Lowering
    training: bool
    corpus: tuple[int, ...] = ()
    """Where the corpus this builds sits in the mixtures above it, an index per mixture."""

    @property
    def global_order(self) -> bool:
        return not self.lowering.per_share

    def child(self, index: int) -> _Build:
        """The build of the `index`th corpus a mixture or phases reads, by its own analysis."""
        return dataclasses.replace(self, lowering=self.lowering.children[index], corpus=(*self.corpus, index))

    def stage_seed(self, pipeline: Pipeline, at: int) -> int:
        """The seed of the random stage written at `at` in `pipeline`: the run's seed, the corpus's
        place in its mixtures and how many random stages its corpus wrote before this one. No stage
        a read leaves out or runs ahead draws (a cache's prefix, a published generation's steps,
        the steps a memory read runs over the whole source), so this count is the same in every
        read. Grain's default seed counts the datasets between a step and the nearest seeded one
        instead, which those reads and the wrappers a read adds change."""
        drawn = sum(isinstance(stage, RandomMap) for stage in pipeline.stages[:at])
        return int(np.random.SeedSequence((self.seed, *self.corpus, drawn)).generate_state(1, np.uint32)[0])

    def mixed(self, pipeline: Pipeline, partition: DataPartition) -> Rows:
        source = pipeline.source
        if not isinstance(source, Mixture):
            raise ValueError("phases need the global phased lowering")
        parents = [self.child(index).records(child, partition)
                   for index, child in enumerate(source.pipelines)]
        if all(isinstance(parent, pygrain.MapDataset) for parent in parents):
            mapped = [parent for parent in parents if isinstance(parent, pygrain.MapDataset)]
            checked = primed_maps(mapped, [_name(child) for child in source.pipelines])
            return pygrain.MapDataset.mix(checked, _weights(source))
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
            )
        return PrimedMix(streams, source.weights, [_name(child) for child in source.pipelines])

    def indexed(
        self, pipeline: Pipeline, source: Source, partition: DataPartition, last: int
    ) -> tuple[Rows, int]:
        """The root of an indexed read, cut to its share by index (`shared`),
        and through its cache when it has one, with the stages it consumed.

        Without a cache the root is dense: an order batched by index needs a
        record at every position. A source held in memory may hold missing
        records (None), so its root is the records it holds, each keeping its
        source position (`_Present`); a cache rejects a missing record as it
        reads it (`_CachedOutputs`)."""
        records = source.records(self.read)
        stored = pygrain.MapDataset.source(records)
        cache = self.lowering.cache
        if cache is None:
            identifiers = source.identifiers if isinstance(source, _Materialized) else ()
            present = _Present.of(records) if _in_memory(source) else None
            if present is not None:
                stored = pygrain.MapDataset.source(present)
                identifiers = tuple((int(position),) for position in present.positions)
            placed = _Placed(identifiers=identifiers, rows=len(stored))
            rows = stored.map_with_index(placed) if last >= 0 else stored
            return self.shared(rows, pipeline, partition), 0
        filled, manifest, prefix = _fill(pipeline, source, cache, self.read)
        cached = filled.mapped(stored, manifest, transform=prefix)
        placed = cached.map_with_index(_Placed(rows=len(cached))) if last > cache else cached
        view = self.shared(placed, pipeline, partition)
        rows = cached.iterated(view, pygrain.ReadOptions(self.loading.threads, self.loading.read_buffer))
        flat = gx.FlatMapIterDataset(rows, _CachedOutputs())
        return Accepted(flat, _name(pipeline), budget=self.budget), cache + 1

    def sharded(
        self, pipeline: Pipeline, source: Source, partition: DataPartition, last: int
    ) -> tuple[pygrain.IterDataset, int]:
        """The root of a streamed read and the stages it consumed: the cache
        prefix runs inside the cache's shard readers, and the record steps
        before an interleave run inside each shard, which the interleave
        flattens where it is written."""
        stages = pipeline.stages
        cache, interleave = self.lowering.cache, self.lowering.interleave
        start = 0 if cache is None else cache + 1
        shards = source.manifest()
        if cache is not None:
            filled, manifest, prefix = _fill(pipeline, source, cache, self.read)
            shards = _positioned(shards) if 0 <= last < cache else shards
            reader = filled.streamed(shards.reader, manifest, transform=prefix)
            shards = dataclasses.replace(shards, reader=reader)
        shards = _positioned(shards) if last >= start else shards
        written = stages[start : start if interleave is None else interleave]
        local = tuple(stage for stage in written if isinstance(stage, _LOCAL))
        shards = dataclasses.replace(
            shards, reader=_ShardSteps(shards.reader, local, cache is not None, _name(pipeline))
        )
        rows = self.streamed(shards, pipeline, partition)
        if cache is not None or any(_drops(stage) for stage in local):
            rows = Accepted(rows, _name(pipeline), budget=self.budget)
        return rows, start if interleave is None else interleave + 1

    def shared(
        self, rows: pygrain.MapDataset, pipeline: Pipeline, partition: DataPartition
    ) -> pygrain.MapDataset:
        """The share of an indexed root that a per-share training read takes,
        repeated unless the pipeline repeats it. Both are views by index, so a
        cache under them still keys each row by its place in the source."""
        if self.global_order or not self.training:
            return rows
        rows = rows[partition.index :: partition.count]
        if not self.lowering.repeated:
            rows = _repeated(rows, None, _name(pipeline), self.loading)
        return rows

    def streamed(
        self, shards: ShardManifest, pipeline: Pipeline, partition: DataPartition
    ) -> pygrain.IterDataset:
        from dew.data.shards import streamed

        interleave = None if self.lowering.interleave is None else pipeline.stages[self.lowering.interleave]
        return streamed(
            shards,
            partition=partition,
            seed=self.seed if self.training else None,
            cycle_length=interleave.cycle_length if isinstance(interleave, Interleave) else 1,
            loading=self.loading,
            epochs=self.lowering.epochs if self.training else 1,
        )

    def records(self, pipeline: Pipeline, partition: DataPartition) -> Rows:
        """The rows before the batch. A global order comes back indexed,
        whole, for `batches` to cut into shares; a per-share order comes back
        as this share's stream of accepted rows."""
        source = pipeline.source
        placed = [index for index, stage in enumerate(pipeline.stages) if isinstance(stage, MapWithIndex)]
        last = max(placed, default=-1)
        if isinstance(source, (Mixture, Phases)):
            rows, consumed = self.mixed(pipeline, partition), 0
        elif self.read == "streamed":
            rows, consumed = self.sharded(pipeline, source, partition, last)
        else:
            rows, consumed = self.indexed(pipeline, source, partition, last)
        for index, stage in enumerate(pipeline.stages[consumed:], start=consumed):
            if isinstance(stage, (BatchStage, MapBatch, Rebatch)):
                break
            if index == self.lowering.absorbed:
                continue
            if isinstance(stage, Prefetch) and self.global_order:
                continue
            rows = self.stage(rows, stage, pipeline, partition, index)
            if index == last:
                rows = rows.map(_unplaced)
        if not self.global_order:
            rows = Accepted(_as_iter(rows, self.loading), _name(pipeline), budget=self.budget)
        return rows

    def stage(self, rows: Rows, stage: Stage, pipeline: Pipeline, partition: DataPartition, at: int) -> Rows:
        """`rows` through `stage`, written at `at` in `pipeline`."""
        if isinstance(stage, (Map, RandomMap, MapWithIndex, Filter, FlatMap)):
            return self.record_stage(rows, stage, pipeline, at)
        if isinstance(stage, Shuffle) and stage.block_shards is not None:
            return self.block_shuffle(rows, stage, pipeline, partition)
        if isinstance(stage, Repeat):
            return _repeated(rows, stage.epochs, _name(pipeline), self.loading)
        if isinstance(stage, (Shuffle, Slice, Limit)) and isinstance(rows, pygrain.MapDataset):
            return _reordered(stage, rows)
        if isinstance(stage, Shuffle) and stage.window is not None:
            rows = gx.WindowShuffleIterDataset(
                _as_iter(rows, self.loading), window_size=stage.window, seed=stage.seed
            )
        elif isinstance(stage, (Shuffle, Slice)):
            if not isinstance(rows, pygrain.MapDataset):
                raise ValueError(f"{type(stage).__name__} in {_name(pipeline)} needs an indexed pipeline")
        elif isinstance(stage, Limit):
            rows = gx.LimitIterDataset(_as_iter(rows, self.loading), stage.count)
        elif isinstance(stage, Prefetch):
            rows = gx.ThreadPrefetchIterDataset(
                _as_iter(rows, self.loading), prefetch_buffer_size=stage.buffer
            )
        elif isinstance(stage, Workers):
            # A column too small for a shared segment of its own crosses pickled (`SHARED_BYTES`).
            options = gx.DatasetOptions(min_shm_size=SHARED_BYTES)
            rows = gx.WithOptionsIterDataset(_as_iter(rows, self.loading), options).mp_prefetch(
                pygrain.MultiprocessingOptions(
                    num_workers=stage.count, per_worker_buffer_size=stage.buffer
                ),
                worker_init_fn=_dies_with_parent,
            )
        elif isinstance(stage, Rebatch):
            rows = gx.RebatchIterDataset(
                _as_iter(rows, self.loading), partition.rows(stage.size), drop_remainder=self.training
            )
        elif isinstance(stage, Native):
            rows = stage.function(rows)
        elif isinstance(stage, Batched):
            groups = _as_iter(rows, self.loading).batch(stage.size, drop_remainder=False, batch_fn=list)
            rows = gx.FlatMapIterDataset(groups, _Groups(stage))
        else:
            return self.layout_stage(rows, stage, pipeline)
        return rows

    def layout_stage(self, rows: Rows, stage: Stage, pipeline: Pipeline) -> Rows:
        if isinstance(stage, TokenSplit):
            if not isinstance(rows, pygrain.MapDataset):
                raise ValueError(f"token split in {_name(pipeline)} needs a finite indexed order")
            rows = _TokenSplit(rows, stage)
        elif isinstance(stage, Pack):
            rows = self.targeted(_packed(_as_iter(rows, self.loading).map(Fields(stage.packed)), stage),
                                 stage.fields[0], pipeline)
        elif isinstance(stage, Windows):
            from dew.data.steps.windows import CausalWindows

            windows = CausalWindows(_as_iter(rows, self.loading), stage)
            rows = self.targeted(windows, stage.fields[0], pipeline)
        elif isinstance(stage, Artifact):
            if not isinstance(rows, pygrain.MapDataset) or isinstance(pipeline.source, (Mixture, Phases)):
                raise ValueError("a cache belongs on each corpus's indexed rows, before mix and interleave")
            prefix = pipeline.stages[: pipeline.stages.index(stage)]
            manifest = _cache_manifest(pipeline.source, prefix, self.read)
            rows = _filled(stage).mapped(rows, manifest)
        elif not isinstance(stage, Interleave):
            raise ValueError(f"{type(stage).__name__} in {_name(pipeline)} has no lowering here")
        return rows

    def targeted(self, windows: pygrain.IterDataset, field: str, pipeline: Pipeline) -> pygrain.IterDataset:
        """The windows that hold a target of their token `field`, as a memory read's and a packing
        plan's are (`materialize._targeted`, `packing.plan_windows`): one with nothing to predict
        is counted as zero_targets and left out, so how a run reads does not change its steps."""
        return Accepted(windows, _name(pipeline), budget=self.budget, tokens=ValidTargets(field))

    def block_shuffle(self, rows: Rows, stage: Shuffle, pipeline: Pipeline, partition: DataPartition) -> Rows:
        from dew.data.ordering import BlockShuffle

        if not isinstance(rows, pygrain.MapDataset) or not isinstance(pipeline.source, _Locality):
            raise ValueError(
                f"block shuffle of {_name(pipeline)} needs an indexed source with physical shard counts"
            )
        if stage.block_shards is None:
            raise ValueError("block shuffle needs a block_shards size")
        sizes, _ = pipeline.source.locality()
        if not self.global_order and self.training:
            starts = np.cumsum((0, *sizes[:-1]))
            sizes = tuple(
                max(
                    0,
                    math.ceil((int(start) + size - partition.index) / partition.count)
                    - math.ceil((int(start) - partition.index) / partition.count),
                )
                for start, size in zip(starts, sizes, strict=True)
            )
        return BlockShuffle(rows, sizes, stage.block_shards, stage.seed)

    def record_stage(
        self, rows: Rows, stage: Map | RandomMap | MapWithIndex | Filter | FlatMap, pipeline: Pipeline,
        at: int,
    ) -> Rows:
        lowered = _Rows(stage)
        if isinstance(stage, FlatMap):
            rows = _expanded(_as_iter(rows, self.loading), stage, _name(pipeline))
        elif isinstance(stage, RandomMap):
            rows = rows.random_map(lowered, seed=self.stage_seed(pipeline, at))
        else:
            rows = rows.map(lowered)
        if isinstance(stage, (Filter, FlatMap)) or stage.rejects:
            rows = Accepted(_as_iter(rows, self.loading), _name(pipeline), budget=self.budget)
        return rows

    def batches(
        self, pipeline: Pipeline, partition: DataPartition, offset: int = 0, *, fill: bool = True
    ) -> pygrain.DatasetIterator[Batch]:
        """This share's batches, through the stages after the batch."""
        stream, after = self.batched(pipeline, partition, offset, fill=fill)
        return iter(self.after_batch(stream, after, pipeline, partition))

    def tuned(self, pipeline: Pipeline) -> _Build:
        """This build with automatic loading resolved on the whole order
        (`Loading.resolved`), and at most one worker per batch of a pass that
        ends where its rows are indexed. Grain sizes workers by memory alone,
        and a validation pass read with training's workers, so a pass of five
        batches had one per core, 26 on an H100 host. Each worker past the
        last batch reads nothing, yet starts a process that imports the
        program and keeps about six pipes open until the reader is collected."""
        loading = self.loading
        bare = dataclasses.replace(self, loading=dataclasses.replace(loading, workers=0))
        if loading.workers == "auto":
            loading = loading.resolved(bare.batched(pipeline, DataPartition())[0])
        if loading.worker_count and (not self.training or not self.lowering.endless):
            rows = bare.records(pipeline, DataPartition())
            if isinstance(rows, pygrain.MapDataset):
                size = next(stage for stage in pipeline.stages if isinstance(stage, BatchStage)).size
                batches = -(-len(rows) // size)
                loading = dataclasses.replace(loading, workers=min(loading.worker_count, batches))
        return dataclasses.replace(self, loading=loading)

    def batched(
        self, pipeline: Pipeline, partition: DataPartition, offset: int = 0, *, fill: bool = True
    ) -> tuple[pygrain.IterDataset, tuple[int, ...]]:
        """This share's batches as the reader reads them, which automatic
        loading also sizes its buffers by (`Loading.resolved`), and where the
        stages after the batch left to run on them stand. A global order is batched by
        index before the reader starts, as Grain's own indexed pipelines are,
        so each read in flight is a whole batch and each worker reads whole
        batches of its own; any other order is batched as its rows arrive.

        A global order also runs by index every stage after the batch that
        holds a place in its order (all but `_UNPLACED`), so a resume `offset`
        records in seeks into its batches as the run read them: past a limit's
        count, a window shuffle's draws and a repeat's passes. A repeat there
        replays its first pass, as a repeat of the batched stream does."""
        canonical = not self.training and not self.global_order
        rows = self.records(pipeline, DataPartition() if canonical else partition)
        batch_stage = next(stage for stage in pipeline.stages if isinstance(stage, BatchStage))
        size = batch_stage.size if canonical else partition.rows(batch_stage.size)
        drop = batch_stage.drop_remainder if self.training else False
        batched_at = pipeline.stages.index(batch_stage)
        after = tuple(range(batched_at + 1, len(pipeline.stages)))
        if not isinstance(rows, pygrain.MapDataset):
            stream = _as_iter(rows, self.loading).batch(
                size, drop_remainder=drop, batch_fn=_Batched(batch_stage.batch_fn)
            )
            if not drop and fill:
                filled = _FillShare(partition, batch_stage.size) if canonical else FilledBatch(size)
                stream = stream.map(filled)
            return stream, after
        if self.training and self.lowering.endless and not self.lowering.repeated:
            rows = _repeated(rows, None, _name(pipeline), self.loading)
        elif self.training and not self.lowering.endless and drop:
            rows = rows[: len(rows) // batch_stage.size * batch_stage.size]
        whole = rows
        rows = rows[partition.index :: partition.count]
        empty = not self.training and not len(rows) and bool(len(whole))
        if empty:
            rows = whole[:1]
        batches = rows.batch(size, drop_remainder=drop, batch_fn=_Batched(batch_stage.batch_fn))
        if not drop and fill:
            batches = batches.map(FilledBatch(size, real=not empty))
        for at in after:
            stage = pipeline.stages[at]
            if isinstance(stage, _UNPLACED):
                continue
            ordered = (_repeated(batches, stage.epochs, _name(pipeline), self.loading, reseed=False)
                       if isinstance(stage, Repeat) else self.stage(batches, stage, pipeline, partition, at))
            if not isinstance(ordered, pygrain.MapDataset):
                raise ValueError(f"{type(stage).__name__} after the batch of {_name(pipeline)} has no "
                                 "place in a global order")
            batches = ordered
        batches = batches[offset // batch_stage.size :]
        # A count of records that ends inside a batch, as a batch ramp saves, resumes past the
        # records of that batch the run read, in the order it read its batches: this share's rows
        # among the batch's first `read`, which the shares hold in turn.
        read = offset % batch_stage.size
        if read > partition.index:
            batches = batches.map_with_index(_Rest(-(-(read - partition.index) // partition.count)))
        stream = _as_iter(batches, self.loading)
        for at, stage in enumerate(pipeline.stages[:batched_at]):
            if isinstance(stage, Prefetch):
                stream = _as_iter(self.stage(stream, stage, pipeline, partition, at), self.loading)
        return stream, tuple(at for at in after if isinstance(pipeline.stages[at], _UNPLACED))

    def after_batch(
        self, stream: pygrain.IterDataset, after: tuple[int, ...], pipeline: Pipeline,
        partition: DataPartition,
    ) -> pygrain.IterDataset:
        """The stages after the batch at `after`, those `batched` left, each as `stage` lowers it,
        with the loading's own workers last when the pipeline names none. Host steps run in the
        reader that wraps this stream."""
        stages = [(at, pipeline.stages[at]) for at in after]
        if self.loading.workers and _at(pipeline.stages, Workers) is None:
            workers = Workers(self.loading.worker_count, self.loading.worker_buffer)
            stages.append((len(pipeline.stages), workers))
        for at, stage in stages:
            if not isinstance(stage, MapBatch):
                stream = _as_iter(self.stage(stream, stage, pipeline, partition, at), self.loading)
        return stream



def _epoch_records(train: Pipeline, build: _Build) -> int | None:
    """The records in one epoch of a global order: each corpus's indexed order
    up to its first repeat, as the lowering builds it, so a slice, a limit and
    a token split count exactly; a mixture, nested or not, reads until its
    first part would come round."""
    if not build.global_order:
        return None

    def counted(pipeline: Pipeline, build: _Build) -> int | None:
        source = pipeline.source
        if isinstance(source, Mixture):
            sizes = [counted(child, build.child(index)) for index, child in enumerate(source.pipelines)]
            known = [size for size in sizes if size is not None]
            if len(known) != len(sizes):
                return None
            weights = _weights(source)
            shares = zip(known, weights, strict=True)
            return max(math.ceil(size * sum(weights) / weight) for size, weight in shares)
        at = _until(pipeline.stages, (Repeat,))
        rows = build.records(dataclasses.replace(pipeline, stages=pipeline.stages[:at]), DataPartition())
        return len(rows) if isinstance(rows, pygrain.MapDataset) else None

    return counted(train, build)


def _split_spans(pipeline: Pipeline, build: _Build) -> Iterator[list[tuple[int, int, int]]]:
    """The spans each token split in the lowered order keeps
    (`_Lowering.splits`), those of the corpora a pipeline mixes first: the
    lowering's own split (`_TokenSplit`) of the indexed order before it, built
    with the analysis of the pipeline it is written in, so they are the
    documents and token offsets a reader of the plan trains on."""
    if isinstance(pipeline.source, (Mixture, Phases)):
        for index, child in enumerate(pipeline.source.pipelines):
            yield from _split_spans(child, build.child(index))
    for at, split in enumerate(pipeline.stages):
        if at in build.lowering.splits and isinstance(split, TokenSplit):
            before = dataclasses.replace(pipeline, stages=pipeline.stages[:at])
            rows = build.records(before, DataPartition())
            if not isinstance(rows, pygrain.MapDataset):
                raise ValueError(f"token split in {_name(pipeline)} needs a finite indexed order")
            yield _TokenSplit(rows, split).spans


def _weights(mixture: Mixture) -> Sequence[float]:
    """A mixture's weights, as shares of its corpora's stated valid target
    tokens when it mixes by tokens, from the rows and valid targets each
    corpus states as it is read."""
    if mixture.tokens is None:
        return mixture.weights
    leaves = [child.source for child in mixture.pipelines if not isinstance(child.source, (Mixture, Phases))]
    facts = [source.metadata() for source in leaves]
    return token_weights(
        [_name(child) for child in mixture.pipelines],
        mixture.weights,
        [integer(stated["rows"], "rows") for stated in facts],
        [integer(stated["valid_targets"], "valid_targets") for stated in facts],
    )


def _workers(pipeline: Pipeline, loading: Loading) -> int:
    return next(
        (stage.count for stage in pipeline.stages if isinstance(stage, Workers)), loading.worker_count
    )


@dataclasses.dataclass(frozen=True)
class _Payload:
    """A row without the private fields Dew's own steps carry to the batch."""

    def __call__(self, row: Batch) -> Batch:
        return {name: value for name, value in row.items() if not name.startswith("_dew_")}


@dataclasses.dataclass(frozen=True)
class _Present:
    """The records a source holds at `positions`, those not missing, read
    densely by index."""

    records: Records
    positions: np.ndarray

    @classmethod
    def of(cls, records: Records) -> _Present | None:
        """The present records of `records`, or None when none is missing."""
        held = [index for index in range(len(records)) if records[index] is not None]
        positions = np.asarray(held, np.int64)
        return None if len(positions) == len(records) else cls(records, positions)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> Batch | bytes | None:
        return self.records[int(self.positions[index])]


@runtime_checkable
class _SharedOutput(Protocol):
    """A batch function that can write its batches into shared memory, which
    Grain asks of a batch read by worker processes."""

    def enable_shared_memory_output(self) -> None: ...


@dataclasses.dataclass
class _Batched:
    """The batch step: each row's private fields dropped (`_Payload`), then
    the rows stacked (`stacked`) by the pipeline's `batch_fn`, or by
    `record_batch`, into shared memory once Grain asks for it. The strip runs
    inside the batch, so it adds no Grain stage of its own; rows carry their
    private fields up to it, for the grouping and counting that read them."""

    batch_fn: Callable[[Sequence[Batch]], Batch] | None
    shared: bool = False

    def enable_shared_memory_output(self) -> None:
        if self.batch_fn is None:
            self.shared = True
        elif isinstance(self.batch_fn, _SharedOutput):
            self.batch_fn.enable_shared_memory_output()

    def __call__(self, rows: Sequence[Batch]) -> Batch:
        if any(row is None for row in rows):
            raise ValueError(
                "an order batched by index read a missing record; a source read by index must hold "
                "every record, or be read with read='memory', which leaves missing records out"
            )
        payload = [_Payload()(row) for row in rows]
        if self.batch_fn is not None:
            return stacked(payload, self.batch_fn)
        return stacked(payload, functools.partial(record_batch, shared=self.shared))
