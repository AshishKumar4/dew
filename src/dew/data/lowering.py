"""The Grain lowering of an analyzed pipeline.

Each stage kind has one definition here, which every read applies and the
memory executor (`dew.data.materialize`) calls for the stages it runs over
the whole source: `_outputs` for a record step, `_reordered` for an order
stage over indexed rows, `_TokenSplit` for a token split, `_packed` for
Grain's packers, and `_fill` with `_CachedOutputs` for opening a cache and
reading what it holds.

`_Build` lowers a pipeline's stages in the order they are written, as the
analysis (`dew.data.analysis`) says to cut them, each run of adjacent record
steps as one Grain operator (`_run`). A global order is built
indexed and whole, and `batches` cuts it into shares just before the batch;
a per-share order is cut at its read and streams this share's accepted
rows. A row carries its source position in a private field from the read to
the last `map_with_index`, which is the only step that reads it.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import functools
import hashlib
import itertools
import logging
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

import grain.experimental as gx
import grain.python as pygrain
import numpy as np

from dew._records import boolean, integer, integers, record, strings
from dew.batch import Batch
from dew.data.analysis import (
    _LOCAL,
    _VIEWS,
    _at,
    _cache_manifest,
    _cache_steps,
    _drops,
    _identity,
    _in_memory,
    _label,
    _Lowering,
    _member_keys,
    _name,
    _steps,
    _until,
    _volatile,
    ends,
)
from dew.data.cache import _Mapped
from dew.data.concurrent import Concurrently
from dew.data.dataset import (
    SHARED_BYTES,
    DataPartition,
    FilledBatch,
    Loading,
    Ranges,
    Records,
    Tail,
    _column,
    _shareable,
    _stacked,
    logged_run,
    record_batch,
    stacked,
    take_batch,
)
from dew.data.mixing import (
    Accepted,
    Chunked,
    CorpusCounts,
    PrimedMix,
    ProportionalSelection,
    Proportions,
    TokenDeficit,
    primed_maps,
    rewound,
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
    Record,
    Rejected,
    Repeat,
    Shuffle,
    Slice,
    Stage,
    TokenBudget,
    TokenSplit,
    Windows,
    Workers,
    children,
)
from dew.data.plan import Manifest
from dew.data.readers import _FillShare
from dew.data.sources.base import Read, Shard, ShardManifest, Source, materialized
from dew.data.sources.urls import Fetch
from dew.data.steps.text import LayoutRule, ValidTargets, counted_window, recounted
from dew.data.text import Chunks
from dew.data.tokens import _WrappingDataset

if TYPE_CHECKING:
    from dew.data.cache import Cache

type Rows = pygrain.MapDataset | pygrain.IterDataset

_log = logging.getLogger(__name__)


@runtime_checkable
class _Locality(Protocol):
    """A source that states the rows of each of its physical shards and
    whether they are remote, which a block shuffle reads."""

    def locality(self) -> tuple[tuple[int, ...], bool]: ...


def _dropping(pipeline: Pipeline, lowering: _Lowering) -> bool:
    """Whether a read can drop records before its batch: a step that drops or expands records, a
    Grain step of the caller's, or a cache read as each input's outputs, in it or in a corpus it
    mixes."""
    if lowering.cache is not None and not lowering.one_to_one:
        return True
    if any(_dropping(child, analyzed)
           for child, analyzed in zip(children(pipeline), lowering.children, strict=True)):
        return True
    batched = _until(pipeline.stages, (BatchStage,))
    return any(_drops(stage) or isinstance(stage, Native) for stage in pipeline.stages[:batched])


def _carried(pipeline: Pipeline, consumed: int, *, marked: bool) -> int:
    """Where a pooled read's loading workers hand its rows to the reading process (`_Build.pooled`):
    at its first stage from `consumed` that can't carry a dropped record as a marker. Record steps
    and repeats carry them. A Grain step of the caller's reads no marker, so it runs in the workers
    only before any: over indexed rows, where what it drops reads as None (`_Marked`); over a
    stream, or where it makes one, the workers hand on around it (`_Build.records`). A view of the
    inputs (a shuffle, slice or take) reads markers as records, so it ends the workers' part once a
    step may have dropped (`marked`). Every other stage lays out, mixes or batches several records
    into one and ends it."""
    for at, stage in enumerate(pipeline.stages[consumed:], start=consumed):
        if isinstance(stage, (Map, RandomMap, MapWithIndex, Filter, FlatMap)):
            marked = marked or _drops(stage)
        elif isinstance(stage, Repeat) or (isinstance(stage, Native) and not marked):
            # The caller's Grain step reads no marker: it runs in the workers before any.
            marked = marked or isinstance(stage, Native)
        elif not (isinstance(stage, (Shuffle, Slice, Limit, Repeat)) and not marked
                  and not (isinstance(stage, Shuffle) and stage.window is not None)):
            return at
    return len(pipeline.stages)


def _shared_pool(pool: int, members: int, index: int) -> int:
    """The loading workers the `index`th of `members` corpora reads in, of a mixture's `pool`: an
    equal share, the first ones taking what is left over; one given none reads in the reading
    process."""
    return pool // members + int(index < pool % members)


class _Marked(pygrain.MapDataset):
    """Indexed records with a marker (`Rejected(reason)`) in place of each that reads as None, a
    store's missing input or one a Grain step dropped, which Grain would pass over unseen: a loading
    worker whose inputs are all None still hands one on for each (`_Build.pooled`)."""

    def __init__(self, records: pygrain.MapDataset, reason: str = "missing"):
        super().__init__(records)
        self._records, self._reason = records, reason

    def __len__(self) -> int:
        return len(self._records)

    @overload
    def __getitem__(self, index: slice) -> pygrain.MapDataset: ...

    @overload
    def __getitem__(self, index: int) -> Record: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.slice(index)
        value = self._records[index]
        return Rejected(self._reason) if value is None else value


def prepared_rows(rows: Records, stages: Sequence[Stage]) -> list[Batch | bytes]:
    """`rows`, a store's records by index, through the record steps among `stages` as a read makes
    them, for a fact measured over what reaches a configured step (`Configured.measures`). A step
    drawing from a record's generator, a configured or Grain step, or one that lays records out
    makes rows no measure before the run can read, and is refused; a view of the inputs changes no
    record and is passed over, so the measure reads them all."""
    current = [(index, rows[index]) for index in range(len(rows))]
    for stage in stages:
        if isinstance(stage, (Shuffle, Repeat, Slice, Limit, Prefetch, Workers)):
            continue
        if not isinstance(stage, (Map, MapWithIndex, Filter, FlatMap)):
            raise ValueError(f"a fact measured after {_identity(stage).name} reads rows that step makes "
                             "only as the run reads them; give the fact itself")
        current = [(position, output) for position, row in current if row is not None
                   for output in _outputs(stage, position, row) if not isinstance(output, Rejected)]
    return [row for _, row in current if isinstance(row, (Mapping, bytes))]


def _cache_inputs(source: Source, records: Records) -> pygrain.MapDataset:
    """The records a cache over `source` reads by index, the rows a memory read made each at the
    source position it came from (`_Placed`), where a step among the cached ones that reads its
    record's index (`map_with_index`) reads it, as it does without the cache."""
    stored = pygrain.MapDataset.source(records)
    if materialized(source) and source.identifiers:
        return stored.map_with_index(_Placed(identifiers=source.identifiers, rows=len(stored)))
    return stored


def _threads(stages: Sequence[Stage]) -> int | None:
    """The most records a fetch among `stages` runs at once on a pool of its own (`Fetch.threads`),
    or None where none has one."""
    pools = [stage.step.threads for stage in stages
             if isinstance(stage, Map) and isinstance(stage.step, Fetch) and stage.step.threads is not None]
    return max(pools, default=None)


def _keyed(source: Source) -> Source:
    """The source a cache over `source` keys its inputs by: under the memory reads that ran no step
    over its rows (a compile may materialize a source twice, `_prepared` and `_staged`), the source
    they hold the rows of, as an indexed read and a publication key it; else `source` itself, the
    rows a memory read made of it before the first view of its inputs (`_analyzed`)."""
    while materialized(source) and not source.ran:
        source = source.source
    return source


def _outputs(
    stage: Stage, position: int | None, value: Record, rng: np.random.Generator | None = None,
    *, counts: CorpusCounts | None = None,
) -> tuple[Record, ...]:
    """What a record stage makes of one row, the single definition the cache
    fill, memory materialization and Grain lowering all apply: its outputs,
    with a `Rejected` marker in place of each row it drops, each carrying the
    targets counted on the row as `recounted` keeps them. A marker passes
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
        if not outputs:
            return (Rejected("empty_expansion"),)
        return tuple(_counted(element, output) for output in outputs)
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
    return (_counted(value, output),)


def _counted(given: Record, output: Record) -> Record:
    """`output` as `recounted` carries the targets counted on `given`, where both are records."""
    if (given is None or output is None or isinstance(given, (bytes, Rejected))
            or isinstance(output, (bytes, Rejected))):
        return output
    return recounted(given, output)


_POSITION = "_dew_position"


def _placed(value: Record, position: int) -> Record:
    return value if value is None or isinstance(value, (bytes, Rejected)) else {**value, _POSITION: position}


def _position(value: Record) -> int | None:
    if value is None or isinstance(value, (bytes, Rejected)) or _POSITION not in value:
        return None
    return int(value[_POSITION])


def _counts_own_rows(pipeline: Pipeline) -> bool:
    """Whether a per-share read counts `pipeline`'s rows as they leave it: a corpus does, and a
    mixture does only where a Grain step of its own may drop rows, which no count of its own
    follows; else its rows are its corpora's, each counted."""
    if not isinstance(pipeline.source, (Mixture, Phases)):
        return True
    return any(isinstance(stage, Native) for stage in pipeline.stages)


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
    shards = Ranges(tuple(rows))
    placed = {shard.id: shards.start(index) for index, shard in enumerate(manifest.shards)}
    return dataclasses.replace(manifest, reader=_PlacedShards(manifest.reader, placed))


type RecordStage = Map | RandomMap | MapWithIndex | Filter | FlatMap

_RECORD = (Map, RandomMap, MapWithIndex, Filter, FlatMap)


def _chained(
    stages: Sequence[Stage], position: int | None, value: Record, rng: np.random.Generator | None = None,
    *, counts: CorpusCounts | None = None,
) -> tuple[Record, ...]:
    """What adjacent record `stages` make of one row: each stage's `_outputs` over every output of
    the one before, a marker passing through the rest."""
    held: tuple[Record, ...] = (value,)
    for stage in stages:
        held = tuple(made for given in held for made in _outputs(stage, position, given, rng, counts=counts))
    return held


def _run(
    stages: Sequence[Stage], start: int, stop: int, *, marked: bool, indexed: bool
) -> tuple[RecordStage, ...]:
    """The run of adjacent record stages from `start`, before `stop`, that lowers as one Grain
    operator (`_Rows`), as tf.data fuses adjacent maps and filters: one call per record for the
    run, not an operator and a copy of the record for each step. A step written after a filter
    then runs where the steps before it do, on the read's threads or in its workers.

    The run's one random stage draws from the rng Grain hands its operator, keyed by the record's
    place in what that operator reads, as it would lowered alone: so a stage before it that drops
    records (counted out by `Accepted`, unless they go on `marked`) or expands them ends the run,
    as does a second random stage. A fetch on its own pool (`Fetch.threads`) runs alone on it.
    Over `indexed` rows the steps before an expansion run by index, on the read's threads, and the
    expansion and the steps after it read them as a stream (`_expanded`)."""
    run: list[RecordStage] = []
    drawn = counted = False
    for stage in stages[start:stop]:
        if not isinstance(stage, _RECORD):
            break
        if _threads((stage,)) is not None:
            return tuple(run) or (stage,)
        if isinstance(stage, RandomMap):
            if drawn or counted:
                break
            drawn = True
        if isinstance(stage, FlatMap):
            # A corpus counts its source examples before an objective's step makes rows of them.
            if drawn or (indexed and run) or (stage.examples and run):
                break
            counted = True
        elif not marked and _drops(stage):
            counted = True
        run.append(stage)
    return tuple(run)


@dataclasses.dataclass(frozen=True)
class _Rows(gx.FlatMapTransform):
    """Grain's form of a run of adjacent record stages (`_run`): `_chained` over each element,
    keeping the source position an element carries out of the steps' sight, and dropping it after
    the run where it holds the last step that reads it (`unplaced`)."""

    stages: tuple[RecordStage, ...]
    counts: CorpusCounts | None = None
    unplaced: bool = False

    def __call__(self, element: Record, rng: np.random.Generator | None = None) -> Record:
        return self.flat_map(element, rng)[0]

    def flat_map(self, element: Record, rng: np.random.Generator | None = None) -> tuple[Record, ...]:
        position = _position(element)
        if position is None:
            return _chained(self.stages, None, element, rng, counts=self.counts)
        outputs = _chained(self.stages, position, _unplaced(element), rng, counts=self.counts)
        return outputs if self.unplaced else tuple(_placed(output, position) for output in outputs)


def _expanded(rows: pygrain.IterDataset, stages: tuple[RecordStage, ...], name: str, label: str, *,
              unplaced: bool = False) -> pygrain.IterDataset:
    """`rows` through a run of record stages that expands them, one Grain flat map; a chunk
    expansion's counted at its fan-out (`Chunked`)."""
    expand = functools.partial(_Rows, stages, unplaced=unplaced)
    if any(isinstance(stage, FlatMap) and isinstance(stage.step, Chunks) for stage in stages):
        return Chunked(rows, name, label, expand)
    return gx.FlatMapIterDataset(rows, expand())


def _through(rows: Rows, run: tuple[RecordStage, ...], name: str, label: str, loading: Loading, *,
             seed: int | None = None, unplaced: bool = False) -> Rows:
    """`rows` through a run of record stages (`_run`) as one Grain operator: on the pool of its
    fetch, as an expansion, as a draw from `seed` where it holds its random stage, or as a map."""
    lowered = _Rows(run, unplaced=unplaced)
    threads = _threads(run)
    if threads is not None:
        # A step with a pool of its own: its records in flight at once, delivered in order.
        return Concurrently(_as_iter(rows, loading), lowered, threads)
    if any(isinstance(stage, FlatMap) for stage in run):
        return _expanded(_as_iter(rows, loading), run, name, label, unplaced=unplaced)
    return rows.map(lowered) if seed is None else rows.random_map(lowered, seed=seed)


type Field = Batch | np.ndarray | np.generic | str | bytes | int | float | bool | None | list | tuple
"""A record's field, as the steps before a batch leave it."""


@dataclasses.dataclass
class _Grouped:
    """A loading worker's records and the markers of those its steps dropped, a batch's count at a
    time (`_Build.handed`): where the records hold the same fields, field by field, each field's
    arrays of one shape and type stacked as a batch's are, into shared memory once Grain asks for
    it, and any other field's values listed; the markers beside them."""

    shared: bool = False

    def enable_shared_memory_output(self) -> None:
        self.shared = True

    def __call__(self, elements: Sequence[Record]) -> Batch:
        markers = [element for element in elements if isinstance(element, Rejected)]
        rows = [element for element in elements if not isinstance(element, Rejected)]
        records = [row for row in rows if isinstance(row, Mapping)]
        alike = (bool(rows) and len(records) == len(rows)
                 and all(row.keys() == records[0].keys() for row in records))
        held = {name: self.column([row[name] for row in records]) for name in records[0]} if alike else rows
        return {"held": held, "alike": alike, "rows": len(rows), "markers": markers}

    def column(self, values: list[Field]) -> np.ndarray | list[Field]:
        """One field's values across the group: stacked where they are arrays of one shape and type,
        else as they are."""
        # What a batch stacks (`_stacked`), of one shape and type; the rest as it is.
        if not all(_stacked(value) for value in values):
            return values
        arrays = [np.asarray(value) for value in values]
        if any(array.shape != arrays[0].shape or array.dtype != arrays[0].dtype for array in arrays):
            return values
        stacked = _column(arrays, shared=self.shared and _shareable(arrays))
        if not isinstance(stacked, np.ndarray):
            raise TypeError(f"arrays of one shape stacked into a {type(stacked).__name__}")
        return stacked


@dataclasses.dataclass(frozen=True)
class _Ungrouped:
    """A worker's group (`_Grouped`) as the markers and records it holds, for the reading process to
    drop the markers and read the records."""

    max_fan_out: int

    def flat_map(self, group: Batch) -> Sequence[Record]:
        markers = list(group["markers"])
        held, count = group["held"], integer(group["rows"], "grouped rows")
        if not group["alike"]:
            return [*markers, *held]
        columns: dict[str, np.ndarray | list] = {}
        for name, column in record(held, "grouped records").items():
            # A batch's column: an array, or a list of a field's values that don't stack.
            if not isinstance(column, (np.ndarray, list)):
                raise TypeError(f"a grouped column {name!r} is a {type(column).__name__}, not a batch column")
            columns[name] = column
        rows = ({name: column[index] for name, column in columns.items()} for index in range(count))
        return [*markers, *rows]


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
    def threads(self) -> int | None:
        """The records a fetch among these steps runs at once on its own pool (`Fetch.threads`)."""
        return _threads(self.stages)

    @property
    def volatile(self) -> bool:
        return any(_volatile(stage) for stage in self.stages)

    def __call__(self, index: int, raw: Record) -> tuple[Record, ...]:
        position = _position(raw)
        if position is not None:
            index, raw = position, _unplaced(raw)
        return _chained(self.stages, index, raw)


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
    label: str
    loading: Loading

    def __call__(self, shard: Shard) -> pygrain.IterDataset:
        rows = self.reader(shard)
        rows = gx.FlatMapIterDataset(rows, _CachedOutputs()) if self.cached else rows
        start = 0
        while start < len(self.stages):
            run = _run(self.stages, start, len(self.stages), marked=True, indexed=False)
            rows = _as_iter(_through(rows, run, self.name, self.label, self.loading), self.loading)
            start += len(run)
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
    return _filled(artifact), _cache_manifest(source, prefix, read), _Prefix(_cache_steps(prefix))


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
    """`rows` read in order, as every stage that streams them reads them: indexed rows become an
    iterator here and nowhere else, which owns the cache fills they read (`_Filled`) as Grain
    carries it into each loading worker."""
    if not isinstance(rows, pygrain.MapDataset):
        return rows
    iterated = rows.to_iter_dataset(pygrain.ReadOptions(loading.threads, loading.read_buffer))
    fills = _fills(rows)
    return _Filled(iterated, fills) if fills else iterated


def _fills(rows: Rows) -> tuple[_Mapped, ...]:
    """The cache reads among `rows` and the datasets they read, each once: the fills an iterator
    of them starts, finishes when it runs out and closes with it. A cache's own parents are the
    records it reads, which hold none."""
    found: dict[int, _Mapped] = {}
    pending: list[pygrain.MapDataset | pygrain.IterDataset] = [rows]
    seen: set[int] = set()
    while pending:
        dataset = pending.pop()
        if id(dataset) in seen:
            continue
        seen.add(id(dataset))
        if isinstance(dataset, _Mapped):
            found[id(dataset)] = dataset
            continue
        pending.extend(dataset.parents)
    return tuple(found.values())


@contextlib.contextmanager
def _closing[Read: (pygrain.MapDataset, pygrain.IterDataset, Rows)](rows: Read) -> Iterator[Read]:
    """`rows` read by a probe of the order, its cache fills finished and closed when it is done."""
    fills = _fills(rows)
    try:
        yield rows
        for fill in fills:
            fill.finish()
    finally:
        for fill in fills:
            fill.close()


class _Filled(pygrain.IterDataset):
    """An iterator of indexed rows that owns the cache fills they read: it starts them, finishes
    them when it runs out, so a fill that read every input publishes, and closes them with it."""

    def __init__(self, parent: pygrain.IterDataset, fills: tuple[_Mapped, ...]):
        super().__init__(parent)
        self._rows, self._fills = parent, fills

    def __iter__(self) -> pygrain.DatasetIterator:
        for fill in self._fills:
            fill.start()
        return _FilledIterator(self._rows.__iter__(), self._fills)


class _FilledIterator(pygrain.DatasetIterator):
    def __init__(self, parent: pygrain.DatasetIterator, fills: tuple[_Mapped, ...]):
        super().__init__(parent)
        self._rows, self._fills = parent, fills

    def __next__(self):
        try:
            return next(self._rows)
        except StopIteration:
            for fill in self._fills:
                fill.finish()
            raise

    def get_state(self) -> Mapping[str, object]:
        return self._rows.get_state()

    def set_state(self, state: Mapping[str, object]) -> None:
        self._rows.set_state(state)

    def close(self) -> None:
        self._rows.close()
        for fill in self._fills:
            fill.close()


class _Inputs(pygrain.MapDataset):
    """A cache read by input, as the source's rows are: by index, at `positions` (the inputs a store
    holds, those it misses skipped, as an uncached read skips them), each input's outputs, or the
    one output of each where the cache holds one record for each input (`single`). The cache writes
    the missing inputs' completions itself as its fill starts (`Cache.mapped(missing=)`)."""

    def __init__(self, cached: _Mapped, positions: np.ndarray | None, *, single: bool):
        super().__init__(cached)
        self._cached, self._positions, self._single = cached, positions, single

    def __len__(self) -> int:
        return len(self._cached) if self._positions is None else len(self._positions)

    @overload
    def __getitem__(self, index: slice) -> pygrain.MapDataset: ...

    @overload
    def __getitem__(self, index: int) -> Record | tuple[Record, ...]: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.slice(index)
        position = index if self._positions is None else int(self._positions[index % len(self)])
        outputs = self._cached[position]
        return outputs[0] if self._single and isinstance(outputs, tuple) else outputs


def _packed(documents: pygrain.IterDataset, stage: Pack) -> pygrain.IterDataset:
    """`documents` packed by Grain's own packer for the stage's policy."""
    sizes = dict.fromkeys(stage.fields, stage.seq_len) | dict(stage.along)
    if stage.policy == "concat":
        cut = _OverlappedCuts(documents, stage) if stage.overlap else documents
        return _Concatenated(gx.ConcatThenSplitIterDataset(cut, length_struct=sizes))
    packer = gx.FirstFitPackIterDataset if stage.policy == "first-fit" else gx.BestFitPackIterDataset
    # Fields cut alike with the first share its segments and positions, as the indexed planner's
    # windows do (`packed_window`), so Grain writes no copy of them for the rest.
    return packer(documents, length_struct=sizes, num_packing_bins=stage.bins, shuffle_bins=False,
                  max_sequences_per_bin=stage.segments, meta_features=stage.fields[1:])


class _Ruled(pygrain.IterDataset):
    """The windows a pack or windows stage makes of `documents`, each counted by the one rule its
    documents' targets were counted by (`counted_window`), as a memory read counts them
    (`materialize._targeted`): a document's count describes no window."""

    def __init__(self, documents: pygrain.IterDataset, stage: Pack | Windows):
        super().__init__(documents)
        self._documents, self._stage = documents, stage

    def __iter__(self) -> pygrain.DatasetIterator:
        from dew.data.steps.windows import CausalWindows

        seen = LayoutRule()
        documents = self._documents.map(seen)
        windows = (_packed(documents.map(self._stage.document), self._stage)
                   if isinstance(self._stage, Pack) else CausalWindows(documents, self._stage))
        return _RuledIterator(iter(windows), seen, self._stage.fields[0])


class _RuledIterator(pygrain.DatasetIterator):
    def __init__(self, windows: pygrain.DatasetIterator, seen: LayoutRule, field: str):
        super().__init__(windows)
        self._windows, self._seen, self._field = windows, seen, field

    def __next__(self) -> Batch:
        return counted_window(next(self._windows), self._seen.rule, self._field)

    def get_state(self) -> Mapping[str, object]:
        return {"windows": self._windows.get_state(), "rule": self._seen.state()}

    def set_state(self, state: Mapping[str, object]) -> None:
        # The rule first: a packer restored reads its held documents again.
        self._seen.restore(record(state["rule"], "layout count rule"))
        self._windows.set_state(record(state["windows"], "layout windows"))

    def close(self) -> None:
        self._windows.close()


class _Concatenated(pygrain.IterDataset):
    """Grain's concatenating packer, restored from a position read back from JSON, as every saved
    position is. Its saved state marks a window's remainder by its slices and none by the tuple
    (-1, -1), which JSON reads back as a list the packer does not take for that tuple, so a restore
    reads the next document as a remainder cut to nothing and drops it (grain 0.2.18 to its main of
    2026-10-18, `packing_concat_then_split._CtsState.has_remainder`). The slices go back as the
    tuples the packer wrote."""

    def __init__(self, parent: pygrain.IterDataset):
        super().__init__(parent)
        self._packed = parent

    def __iter__(self) -> pygrain.DatasetIterator:
        return _ConcatenatedIterator(self._packed.__iter__())


class _ConcatenatedIterator(pygrain.DatasetIterator):
    def __init__(self, parent: pygrain.DatasetIterator):
        super().__init__(parent)
        self._packed = parent

    def __next__(self) -> Batch:
        return next(self._packed)

    def get_state(self) -> Mapping[str, object]:
        return self._packed.get_state()

    def set_state(self, state: Mapping[str, object]) -> None:
        slices = record(state["remainder_slices"], "packed remainder")
        self._packed.set_state({**state, "remainder_slices": {
            name: integers(bounds, name) for name, bounds in slices.items()}})


class _OverlappedCuts(pygrain.IterDataset):
    """Documents as a concatenating pack (`Pack.overlap`) reads them: the tokens before each cut
    through a document written again after it, so the window the cut ends and the next both hold
    them. Grain's concatenating packer cuts the documents it joins every `seq_len` tokens, so how
    full this stream leaves its window places the next cut."""

    def __init__(self, parent: pygrain.IterDataset, stage: Pack):
        super().__init__(parent)
        self._documents, self._stage = parent, stage

    def __iter__(self) -> pygrain.DatasetIterator:
        return _OverlappedCutsIterator(self._documents.__iter__(), self._stage)


class _OverlappedCutsIterator(pygrain.DatasetIterator):
    """Each document with its cuts overlapped, and where the last one left its window: the room
    left in it, saved with the parent's position."""

    def __init__(self, parent: pygrain.DatasetIterator, stage: Pack):
        super().__init__(parent)
        self._documents, self._stage = parent, stage
        self._room = stage.seq_len

    def __next__(self) -> Batch:
        document = next(self._documents)
        count = len(document[self._stage.fields[0]])
        pieces, start, room = [], 0, self._room
        while count - start > room:
            pieces.append(slice(start, start + room))
            start, room = max(start + room - self._stage.overlap, 0), self._stage.seq_len
        pieces.append(slice(start, count))
        self._room = room - (count - start) or self._stage.seq_len
        if len(pieces) == 1:
            return document
        return {name: np.concatenate([value[piece] for piece in pieces]) if name in self._stage.fields
                else value for name, value in document.items()}

    def get_state(self) -> Mapping[str, object]:
        return {"documents": self._documents.get_state(), "room": self._room}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._documents.set_state(record(state["documents"], "packed documents"))
        self._room = integer(state["room"], "packed window room")


@overload
def _repeated(rows: pygrain.MapDataset, epochs: int | None, name: str, loading: Loading, *,
              reseed: bool = True, certain: bool = True, barren: bool = False) -> pygrain.MapDataset: ...


@overload
def _repeated(rows: Rows, epochs: int | None, name: str, loading: Loading, *,
              reseed: bool = True, certain: bool = True, barren: bool = False) -> Rows: ...


def _repeated(rows: Rows, epochs: int | None, name: str, loading: Loading, *, reseed: bool = True,
              certain: bool = True, barren: bool = False) -> Rows:
    """`rows` read `epochs` times over, or endlessly, by Grain's repeat. An indexed repeat draws
    each pass afresh (`reseed`) as Grain's does, unless it stands for a repeat of a stream, whose
    passes replay the first. An endless repeat of a pass that holds nothing is refused, as it
    would read nothing forever: an indexed pass by its length, a stream's when a pass ends without
    a row and nothing before the repeat drops one (`certain`), so the pass is the source's own
    rows. A pass a step emptied may yield rows next time (a fetch that failed for now), so the
    repeat reads on, logging its empty passes. Every pass of a stream goes on counting its corpus
    (`_Passes`). A worker's pass of its inputs and the markers of what they dropped that holds
    nothing (`barren`) holds no input at all, which ends it (`_Barren`)."""
    if isinstance(rows, pygrain.MapDataset):
        if epochs is None and not len(rows):
            raise ValueError(f"a pass of {name} holds no rows, so repeating it reads nothing forever")
        return rows.repeat(epochs, reseed_each_epoch=reseed)
    parent = _as_iter(rows, loading)
    passes = _Passes(parent, name, endless=epochs is None, certain=certain, barren=barren)
    return gx.RepeatIterDataset(passes, epochs)


class _Passes(pygrain.IterDataset):
    """A stream a repeat reads pass after pass, each going on counting its corpus. Endlessly read
    (`endless`), a pass that yields no row is refused where that is certain to repeat (`certain`),
    and logged at each doubling of a run of them otherwise."""

    def __init__(self, parent: pygrain.IterDataset, name: str, *, endless: bool, certain: bool,
                 barren: bool = False):
        super().__init__(parent)
        self._rows, self._name, self._endless, self._certain = parent, name, endless, certain
        self._barren = barren

    def __iter__(self) -> pygrain.DatasetIterator:
        return _PassesIterator(self._rows.__iter__(), self._name, self._endless, self._certain, self._barren)


class _PassesIterator(pygrain.DatasetIterator):
    """Its parent's rows and position. The repeat restarts a pass by restoring the parent right
    after one ends, and any other restore lands inside a pass the run already read from."""

    def __init__(self, parent: pygrain.DatasetIterator, name: str, endless: bool, certain: bool,
                 barren: bool = False):
        super().__init__(parent)
        self._rows, self._name, self._endless, self._certain = parent, name, endless, certain
        self._barren = barren
        self._yielded = self._ended = False
        self._empty = 0

    def __next__(self):
        try:
            value = next(self._rows)
        except StopIteration:
            if not self._yielded and self._barren:
                raise _Barren(self._name) from None
            if not self._yielded and self._endless:
                if self._certain:
                    raise ValueError(f"a pass of {self._name} yields no rows, so repeating it reads "
                                     "nothing forever") from None
                self._empty += 1
                # From the first: a pass of no rows is no ordinary filter's run.
                if logged_run(self._empty):
                    _log.warning("%s yielded no row in its last %d passes; the read goes on until one does",
                                 self._name, self._empty)
            self._ended = True
            raise
        self._yielded, self._empty = True, 0
        return value

    def get_state(self):
        return self._rows.get_state()

    def set_state(self, state) -> None:
        # Right after a pass ends, this is the repeat rewinding it, which keeps the corpus's counts.
        self._rows.set_state(rewound(state, self._rows.get_state()) if self._ended else state)
        self._yielded, self._ended = not self._ended, False


class _Barren(Exception):
    """A pass of a loading worker's inputs, each read as a row or the marker of a dropped record,
    that held nothing: the worker holds no input (`_repeated`)."""


def _elastic_cut(stages: Sequence[Stage]) -> int:
    """Where an elastic read's own rows begin (`_Build.elastic_batches`): after the last step before
    the batch that is not a `Map` of each record alone (or a `Prefetch`, which reads none ahead
    there). Every reader runs the steps before it over the whole order, as each of them decides
    which rows there are (a filter, an expansion, a pack), which draws they take or what their place
    gives them, which only the whole order states; a map after them computes a row from it alone."""
    return max((at + 1 for at, stage in enumerate(stages) if not isinstance(stage, (Map, Prefetch))),
               default=0)


class _ElasticBatches(pygrain.IterDataset):
    """An elastic read's batches (`_Build.elastic_batches`): `rows`, the order one reader reads,
    taken a whole batch's `size` at a time, of which this reader keeps its share (`partition.held`),
    each through `maps` on `threads` threads, batched by `batch_fn`. A last batch of fewer rows is
    dropped where `drop`, or else filled to the share's rows (`FilledBatch`), its rows not real where
    the share holds none of them, so every reader reads a batch where any does. Its position is the
    order's, the same on every reader at every batch, so any reader count resumes it."""

    def __init__(self, rows: pygrain.IterDataset, maps: tuple[Map, ...], partition: DataPartition, size: int,
                 *, drop: bool, fill: bool, batch_fn: _Batched, threads: int):
        super().__init__(rows)
        self.rows, self.maps, self.partition, self.size = rows, maps, partition, size
        self.drop, self.fill, self.batch_fn, self.threads = drop, fill, batch_fn, threads

    def mapped(self, row: Record) -> Batch:
        (made,) = _chained(self.maps, None, row)
        assert isinstance(made, Mapping), "a row a batch stacks is a record of fields"
        return made

    def __iter__(self) -> pygrain.DatasetIterator:
        return _ElasticIterator(self.rows.__iter__(), self)


class _ElasticIterator(pygrain.DatasetIterator):
    def __init__(self, parent: pygrain.DatasetIterator, batches: _ElasticBatches):
        super().__init__(parent)
        self._rows, self._batches = parent, batches
        threads = batches.threads if batches.maps else 1
        self._pool = (None if threads <= 1
                      else concurrent.futures.ThreadPoolExecutor(threads, thread_name_prefix="dew-elastic"))

    def __next__(self) -> Batch:
        batches = self._batches
        group = list(itertools.islice(self._rows, batches.size))
        if not group or (batches.drop and len(group) < batches.size):
            raise StopIteration
        held, real = batches.partition.held(len(group))
        mine = group[held]
        pool = self._pool
        made = [batches.mapped(row) for row in mine] if pool is None else list(pool.map(batches.mapped, mine))
        batch = batches.batch_fn(made)
        if not batches.fill:
            return batch
        return FilledBatch(batches.partition.rows(batches.size), real=real).map(batch)

    def get_state(self) -> Mapping[str, object]:
        return self._rows.get_state()

    def set_state(self, state: Mapping[str, object]) -> None:
        self._rows.set_state(state)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
        self._rows.close()


class _Ended(pygrain.IterDataset):
    """A repeat of a loading worker's inputs, which ends where a pass of them holds none (`_Barren`)
    instead of repeating nothing forever, so the other workers read on."""

    def __init__(self, parent: pygrain.IterDataset):
        super().__init__(parent)
        self._rows = parent

    def __iter__(self) -> pygrain.DatasetIterator:
        return _EndedIterator(self._rows.__iter__())


class _EndedIterator(pygrain.DatasetIterator):
    def __init__(self, parent: pygrain.DatasetIterator):
        super().__init__(parent)
        self._rows, self._ended = parent, False

    def __next__(self):
        if self._ended:
            raise StopIteration
        try:
            return next(self._rows)
        except _Barren:
            self._ended = True
            raise StopIteration from None

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._rows.get_state(), "ended": self._ended}

    def set_state(self, state: Mapping[str, object]) -> None:
        # An ended repeat reads no more, so only one that reads on restores where it was.
        self._ended = boolean(state["ended"], "repeat ended")
        if not self._ended:
            self._rows.set_state(record(state["parent"], "repeat position"))


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


def _member_key(members: Sequence[str], index: int) -> int:
    """A mixture member's key for its random draws (`_member_keys`), as a seed."""
    key = _member_keys(members)[index]
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


@dataclasses.dataclass(frozen=True)
class _Build:
    read: Read
    loading: Loading
    seed: int
    batch: int
    lowering: _Lowering
    training: bool
    corpus: tuple[int, ...] = ()
    """The key of the corpus this builds in each mixture above it (`_member_key`)."""
    elastic: bool = False
    """Whether the training read is elastic (`Pipeline.elastic`): every reader reads the order one
    reader reads, and keeps its share of each batch (`elastic_batches`)."""
    pool: Workers | None = None
    """The loading workers of a read whose steps drop records (`pooled`), each a process that reads
    its corpus's steps up to the first that can't carry a dropped record as a marker (`_carried`),
    so a worker whose inputs are all dropped still delivers; the reading process drops the markers
    and runs the rest. A mixture's corpora each read in a share of them, or in the workers a corpus
    writes itself. None where the workers read the whole order after its batch, as a read that drops
    nothing is read."""

    @property
    def global_order(self) -> bool:
        return not self.lowering.per_share

    def child(self, index: int) -> _Build:
        """The build of the `index`th corpus a mixture or phases reads, by its own analysis, keyed
        by the name it goes by there (`_member_key`)."""
        members = strings(self.lowering.order["members"], "mixture members")
        pool = (None if self.pool is None
                else Workers(_shared_pool(self.pool.count, len(members), index), self.pool.buffer))
        loading = self.loading if pool is None else dataclasses.replace(self.loading, workers=pool.count)
        return dataclasses.replace(self, lowering=self.lowering.children[index], loading=loading, pool=pool,
                                   corpus=(*self.corpus, _member_key(members, index)))

    def pooled(self, pipeline: Pipeline) -> _Build:
        """This build with its loading workers moved before the first step that can't carry what it
        drops (`pool`), for a training read cut per share whose steps can drop records: a worker
        whose share of the inputs was all dropped would deliver nothing, and Grain's workers take
        turns, so the read would wait on it forever. Grain advises a read that filters in its
        workers to filter after them instead (`MapDataset.mp_prefetch`). The workers are the ones
        written (`.workers(n)`), else the loading's; a corpus of a mixture that writes its own reads
        in those."""
        count = _workers(pipeline, self.loading)
        written = any(isinstance(stage, Workers) for stage in _steps(pipeline))
        if (self.global_order or not self.training or not (count or written)
                or not _dropping(pipeline, self.lowering)):
            return self
        pool = next((stage for stage in pipeline.stages if isinstance(stage, Workers)),
                    Workers(count, self.loading.worker_buffer))
        return dataclasses.replace(self, loading=dataclasses.replace(self.loading, workers=count), pool=pool)

    def owning(self, pipeline: Pipeline) -> _Build:
        """This build reading `pipeline` in the workers it writes, a corpus or a mixture whose
        corpora share them, or else in its share of the read's (`pool`)."""
        written = next((stage for stage in pipeline.stages if isinstance(stage, Workers)), None)
        if self.pool is None or written is None:
            return self
        loading = dataclasses.replace(self.loading, workers=written.count)
        return dataclasses.replace(self, pool=written, loading=loading)

    def pools(self, pipeline: Pipeline) -> dict[str, int]:
        """The loading workers each corpus reads in, as `pooled` shares them out, by the corpus's
        place among the mixtures above it, which a position saved under one sharing restores only
        under the same, a corpus a mixture keeps across an edit included."""
        build = self.owning(pipeline)
        source = pipeline.source
        if build.pool is None:
            return {}
        if not isinstance(source, (Mixture, Phases)):
            return {"/".join(str(key) for key in build.corpus): build.pool.count}
        return {key: count for index, child in enumerate(source.pipelines)
                for key, count in build.child(index).pools(child).items()}

    def stage_seed(self, pipeline: Pipeline, at: int, partition: DataPartition) -> int:
        """The seed of the random stage written at `at` in `pipeline`: the run's seed, the corpus's
        key in each of its mixtures and how many random stages its corpus wrote before this one. No stage
        a read leaves out or runs ahead draws (a cache's prefix, a published generation's steps,
        the steps a memory read runs over the whole source), so this count is the same in every
        read. Grain's default seed counts the datasets between a step and the nearest seeded one
        instead, which those reads and the wrappers a read adds change. A native step is a random
        stage too: the dataset it is handed seeds the random steps it adds by that default.

        A stage after the batch draws for this share's part of each batch, which other shares'
        parts must not repeat, so its seed adds the share where there is more than one."""
        drawn = sum(isinstance(stage, (RandomMap, Native)) for stage in pipeline.stages[:at])
        batched = any(isinstance(stage, BatchStage) for stage in pipeline.stages[:at])
        share = (partition.index, partition.count) if batched and partition.count > 1 else ()
        key = (self.seed, *self.corpus, drawn, *share)
        return int(np.random.SeedSequence(key).generate_state(1, np.uint32)[0])

    def mixed(self, pipeline: Pipeline, partition: DataPartition) -> Rows:
        source = pipeline.source
        if not isinstance(source, Mixture):
            raise ValueError("phases need the global phased lowering")
        parents = [self.child(index).records(child, partition)
                   for index, child in enumerate(source.pipelines)]
        if all(isinstance(parent, pygrain.MapDataset) for parent in parents):
            mapped = [parent for parent in parents if isinstance(parent, pygrain.MapDataset)]
            checked = primed_maps(mapped)
            selection = ProportionalSelection([len(parent) for parent in checked], _weights(source))
            return pygrain.MapDataset.select_from_datasets(checked, selection)
        streams = [
            Accepted(_as_iter(parent, self.loading), _name(child), _label(child))
            for parent, child in zip(parents, source.pipelines, strict=True)
        ]
        # A streamed mix saves each member's position under its key, so a member that joins starts
        # and one that left is dropped while the others resume (`Plan.check`).
        keys = _member_keys(strings(self.lowering.order["members"], "mixture members"))
        if source.tokens is not None:
            return TokenDeficit(
                streams,
                source.weights,
                keys,
                tokens=source.tokens,
                names=[_name(child) for child in source.pipelines],
                labels=[_label(child) for child in source.pipelines],
            )
        return PrimedMix(streams, source.weights, keys)

    def indexed(
        self, pipeline: Pipeline, source: Source, partition: DataPartition, last: int, *, marked: bool = False
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
            identifiers = source.identifiers if materialized(source) else ()
            present = _Present.of(records) if _in_memory(source) else None
            if present is not None:
                stored = pygrain.MapDataset.source(present)
                identifiers = tuple((int(position),) for position in present.positions)
            elif marked:
                stored = _Marked(stored)
            placed = _Placed(identifiers=identifiers, rows=len(stored))
            rows = stored.map_with_index(placed) if last >= 0 else stored
            return self.shared(rows, pipeline, partition), 0
        keyed = _keyed(source)
        if materialized(source) and keyed is not source:
            # A memory read that ran no step before the cache holds the rows of the source it keys,
            # at the places there it states: the cache reads those, its missing inputs skipped.
            records = keyed.records(self.read)
            held = np.asarray([identifier[0] for identifier in source.identifiers], np.int64)
            positions = None if len(held) == len(records) else held
        else:
            present = _Present.of(records) if _in_memory(keyed) else None
            positions = None if present is None else present.positions
        stored = _cache_inputs(keyed, records)
        filled, manifest, prefix = _fill(pipeline, keyed, cache, self.read)
        # Read by input, as the source's own rows are: its missing inputs skipped, as the uncached
        # root skips them, and written by the fill; one record for each input reads by index and in
        # the uncached chain's order, the rest as a stream of each input's outputs.
        missing = () if positions is None else np.setdiff1d(np.arange(len(stored)), positions).tolist()
        cached = filled.mapped(stored, manifest, transform=prefix, source=keyed, missing=missing)
        inputs = _Inputs(cached, positions, single=self.lowering.one_to_one)
        identifiers = source.identifiers if materialized(source) else ()
        if positions is not None:
            identifiers = tuple((int(position),) for position in positions)
        placed = (inputs.map_with_index(_Placed(identifiers=identifiers, rows=len(inputs)))
                  if last > cache else inputs)
        view = self.viewed(self.shared(placed, pipeline, partition), pipeline, partition, cache)
        if self.lowering.one_to_one:
            return view, cache + 1
        # A fetch with a pool of its own among the cached steps reads that many inputs at once.
        threads = max(self.loading.threads, prefix.threads or 0)
        read = dataclasses.replace(self.loading, threads=threads,
                                   read_buffer=max(self.loading.read_buffer, threads))
        flat = gx.FlatMapIterDataset(_as_iter(view, read), _CachedOutputs())
        return (flat if marked else Accepted(flat, _name(pipeline), _label(pipeline))), cache + 1

    def sharded(
        self, pipeline: Pipeline, source: Source, partition: DataPartition, last: int, *, marked: bool = False
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
            reader = filled.streamed(shards.reader, manifest, transform=prefix, threads=prefix.threads)
            shards = dataclasses.replace(shards, reader=reader)
        shards = _positioned(shards) if last >= start else shards
        written = stages[start : start if interleave is None else interleave]
        local = tuple(stage for stage in written if isinstance(stage, _LOCAL))
        steps = _ShardSteps(shards.reader, local, cache is not None, _name(pipeline), _label(pipeline),
                            self.loading)
        shards = dataclasses.replace(shards, reader=steps)
        rows = self.streamed(shards, pipeline, partition)
        if not marked and (cache is not None or any(_drops(stage) for stage in local)):
            rows = Accepted(rows, _name(pipeline), _label(pipeline))
        return rows, start if interleave is None else interleave + 1

    def shared(
        self, rows: pygrain.MapDataset, pipeline: Pipeline, partition: DataPartition
    ) -> pygrain.MapDataset:
        """The share of an indexed root that a per-share training read takes,
        repeated unless the pipeline repeats it. Both are views by index, so a
        cache under them still keys each row by its place in the source."""
        if self.global_order or not self.training:
            return rows
        rows = rows[partition.cut()]
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
            name=_label(pipeline),
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
        # This corpus's own workers, where they read its first steps (`pooled`): those steps keep
        # each record they drop as a marker, which the reading process drops after them.
        pool = self.owning(pipeline).pool
        pooled = pool is not None and pool.count > 0 and not isinstance(source, (Mixture, Phases))
        # A Grain step of the caller's reads its rows as Grain gives them, without a marker.
        native = next((isinstance(stage, Native) for stage in pipeline.stages
                       if not isinstance(stage, (Shuffle, Slice, Limit, Repeat))), False)
        if isinstance(source, (Mixture, Phases)):
            rows, consumed = self.owning(pipeline).mixed(pipeline, partition), 0
        elif self.read == "streamed":
            rows, consumed = self.sharded(pipeline, source, partition, last, marked=pooled)
        else:
            rows, consumed = self.indexed(pipeline, source, partition, last, marked=pooled and not native)
        # A cache read as a stream of each input's outputs marks what it drops from the start, as do
        # the steps a streamed read runs inside each shard.
        marked = self.lowering.cache is not None and (self.read == "streamed" or not self.lowering.one_to_one)
        marked = marked or any(_drops(stage) for stage in pipeline.stages[:consumed])
        carried = _carried(pipeline, consumed, marked=marked) if pooled and pool is not None else -1
        ran, counting = consumed, _counts_own_rows(pipeline)
        for index, stage in enumerate(pipeline.stages[consumed:], start=consumed):
            if index < ran:
                continue
            if index < carried and isinstance(stage, Native) and not isinstance(rows, pygrain.MapDataset):
                # What a Grain step drops from a stream leaves no marker: the workers hand on before it.
                carried = index
            if index == carried and pool is not None:
                rows = self.handed(rows, pipeline, partition, pool)
            if isinstance(stage, (BatchStage, MapBatch)):
                break
            if index == self.lowering.absorbed:
                continue
            if isinstance(stage, Prefetch) and self.global_order:
                continue
            if isinstance(stage, Workers) and pool is not None:
                continue
            if isinstance(stage, FlatMap) and stage.examples and not self.global_order and counting:
                # The corpus's counts are of its source examples: they are taken before an objective's
                # step makes rows of them (`FlatMap.examples`), and not again after.
                rows = Accepted(_as_iter(rows, self.loading), _name(pipeline), _label(pipeline))
                counting = False
            if isinstance(stage, _RECORD):
                # The record steps from here run as one operator (`_run`), up to where the workers
                # hand their rows on.
                run = _run(pipeline.stages, index, carried if index < carried else len(pipeline.stages),
                           marked=index < carried, indexed=isinstance(rows, pygrain.MapDataset))
                ran = index + len(run)
                rows = self.record_stage(rows, run, pipeline, partition, index, marked=index < carried,
                                         unplaced=index <= last < ran)
            elif index < carried and isinstance(stage, Repeat) and not isinstance(rows, pygrain.MapDataset):
                # A worker's inputs hold a marker for each record dropped, so a pass with nothing
                # in it holds no input: that worker reads no more, and the others read on.
                repeated = _repeated(rows, stage.epochs, _name(pipeline), self.loading, barren=True)
                rows = _Ended(_as_iter(repeated, self.loading))
            elif index < carried and isinstance(stage, Native):
                rows = self.stage(rows, stage, pipeline, partition, index)
                if isinstance(rows, pygrain.MapDataset):
                    # What a Grain step drops from indexed rows reads as None, which a marker stands for.
                    rows = _Marked(rows, "grain")
                elif pool is not None:
                    # It made a stream, whose drops leave no marker: the workers hand on after it, and
                    # a worker whose inputs it drops whole holds the others up (docs/design/data.md).
                    rows = self.handed(rows, pipeline, partition, pool)
                    carried = -1
            else:
                rows = self.stage(rows, stage, pipeline, partition, index)
        if carried == len(pipeline.stages) and pool is not None:
            rows = self.handed(rows, pipeline, partition, pool)
        if not self.global_order and counting:
            rows = Accepted(_as_iter(rows, self.loading), _name(pipeline), _label(pipeline))
        return rows

    def stage(self, rows: Rows, stage: Stage, pipeline: Pipeline, partition: DataPartition, at: int) -> Rows:
        """`rows` through `stage`, written at `at` in `pipeline`."""
        if isinstance(stage, _RECORD):
            return self.record_stage(rows, (stage,), pipeline, partition, at)
        if isinstance(stage, Shuffle) and stage.block_shards is not None:
            return self.block_shuffle(rows, stage, pipeline, partition)
        if isinstance(stage, Repeat):
            certain = not any(_drops(written) for written in pipeline.stages[:at])
            return _repeated(rows, stage.epochs, _name(pipeline), self.loading, certain=certain)
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
        elif isinstance(stage, Native):
            # A Grain step reads cache fills through an iterator that owns them (`_Filled`).
            given = _as_iter(rows, self.loading) if _fills(rows) else rows
            rows = stage.function(given.seed(self.stage_seed(pipeline, at, partition)))
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
        elif isinstance(stage, (Pack, Windows)):
            rows = self.targeted(_Ruled(_as_iter(rows, self.loading), stage), stage.fields[0], pipeline)
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
        return Accepted(windows, _name(pipeline), _label(pipeline), tokens=ValidTargets(field))

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
            # Each shard's records this share holds: every count-th from its index on.
            shards = Ranges(tuple(sizes))
            sizes = tuple(
                max(
                    0,
                    math.ceil((shards.start(index) + size - partition.index) / partition.count)
                    - math.ceil((shards.start(index) - partition.index) / partition.count),
                )
                for index, size in enumerate(sizes)
            )
        return BlockShuffle(rows, sizes, stage.block_shards, stage.seed)

    def viewed(self, rows: Rows, pipeline: Pipeline, partition: DataPartition, cache: int) -> Rows:
        """`rows`, a cache's inputs, through the views of them written before the cache
        (`_VIEWS`: shuffles, repeats, slices, takes), where and as the uncached chain applies them:
        each input's cached outputs follow it, as its record steps would make them after the view."""
        for at, stage in enumerate(pipeline.stages[:cache]):
            if isinstance(stage, _VIEWS):
                viewed = self.stage(rows, stage, pipeline, partition, at)
                if not isinstance(viewed, pygrain.MapDataset):
                    raise ValueError(f"{type(stage).__name__} before the cache of {_name(pipeline)} reads "
                                     "its inputs as a stream, and the cache is read by index: write it "
                                     "after the cache")
                rows = viewed
        return rows

    def handed(self, rows: Rows, pipeline: Pipeline, partition: DataPartition,
               pool: Workers) -> pygrain.IterDataset:
        """`rows`, read by this corpus's loading workers (`pool`), each handing on every row with
        a marker in place of each record its steps dropped, which the reading process drops here,
        counting each by its reason."""
        # A worker hands on its rows and markers a share's batch count at a time, the fields that
        # stack stacked as a batch's are, as many groups in flight as it had batches.
        size = partition.rows(self.batch)
        grouped = _as_iter(rows, self.loading).batch(size, drop_remainder=False, batch_fn=_Grouped())
        handed = self.stage(grouped, pool, pipeline, partition, len(pipeline.stages))
        flat = gx.FlatMapIterDataset(_as_iter(handed, self.loading), _Ungrouped(size))
        return Accepted(flat, _name(pipeline), _label(pipeline))

    def record_stage(
        self, rows: Rows, run: tuple[RecordStage, ...], pipeline: Pipeline, partition: DataPartition, at: int,
        *, marked: bool = False, unplaced: bool = False,
    ) -> Rows:
        """`rows` through a run of adjacent record stages written from `at` (`_run`), as one Grain
        operator. A run that can drop records drops them after it, counted (`Accepted`), unless its
        rows are `marked`: then each dropped record goes on as a marker for the reading process to
        drop (`handed`). Its random stage draws with the seed it has lowered alone (`stage_seed`)."""
        drawn = next((offset for offset, stage in enumerate(run) if isinstance(stage, RandomMap)), None)
        seed = None if drawn is None else self.stage_seed(pipeline, at + drawn, partition)
        rows = _through(rows, run, _name(pipeline), _label(pipeline), self.loading, seed=seed,
                        unplaced=unplaced)
        if not marked and any(_drops(stage) for stage in run):
            rows = Accepted(_as_iter(rows, self.loading), _name(pipeline), _label(pipeline))
        return rows

    def batches(
        self, pipeline: Pipeline, partition: DataPartition, offset: int = 0, *, fill: bool = True
    ) -> pygrain.DatasetIterator[Batch]:
        """This share's batches, through the stages after the batch."""
        stream, after = self.batched(pipeline, partition, offset, fill=fill)
        return iter(self.after_batch(stream, after, pipeline, partition))

    def tuned(self, pipeline: Pipeline) -> _Build:
        """This build with automatic loading resolved on the whole order from the host's CPUs and
        memory (`Loading.resolved`), and at most one worker per batch of a pass that ends where
        its rows are indexed. A validation pass is read with training's workers, so a pass of
        five batches had one per core, 26 on an H100 host. Each worker past the last batch reads
        nothing, yet starts a process that imports the program and keeps about six pipes open
        until the reader is collected."""
        loading = self.loading
        bare = dataclasses.replace(self, loading=dataclasses.replace(loading, workers=0))
        size = next(stage for stage in pipeline.stages if isinstance(stage, BatchStage)).size
        if loading.workers == "auto":
            # A global order is read by the batch, any other order by the record (`batch_order`).
            with _closing(bare.batched(pipeline, DataPartition())[0]) as sampled:
                loading = loading.resolved(sampled, reads=1 if self.global_order else size)
        if loading.worker_count and (not self.training or not self.lowering.endless):
            with _closing(bare.records(pipeline, DataPartition())) as rows:
                if isinstance(rows, pygrain.MapDataset):
                    batches = -(-len(rows) // size)
                    loading = dataclasses.replace(loading, workers=min(loading.worker_count, batches))
        return dataclasses.replace(self, loading=loading)

    def batched(
        self, pipeline: Pipeline, partition: DataPartition, offset: int = 0, *, fill: bool = True
    ) -> tuple[pygrain.IterDataset, tuple[int, ...]]:
        """This share's batches as the reader reads them (`batch_order`), read in turn, a global
        order's with the prefetch its stages before the batch name, and where the stages after the
        batch left to run on them stand."""
        batches, after = self.batch_order(pipeline, partition, offset, fill=fill)
        if not isinstance(batches, pygrain.MapDataset):
            return batches, after
        stream = _as_iter(batches, self.loading)
        batched_at = next(at for at, stage in enumerate(pipeline.stages) if isinstance(stage, BatchStage))
        for at, stage in enumerate(pipeline.stages[:batched_at]):
            if isinstance(stage, Prefetch):
                stream = _as_iter(self.stage(stream, stage, pipeline, partition, at), self.loading)
        return stream, after

    def batch_order(
        self, pipeline: Pipeline, partition: DataPartition, offset: int = 0, *, fill: bool = True
    ) -> tuple[pygrain.MapDataset | pygrain.IterDataset, tuple[int, ...]]:
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
        if self.elastic and self.training and not self.global_order:
            return self.elastic_batches(pipeline, partition, fill=fill)
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
        if self.training and self.lowering.rooted:
            rows = _repeated(rows, None, _name(pipeline), self.loading)
        elif self.training and not (self.lowering.records_unending or self.lowering.rooted) and drop:
            rows = rows[: len(rows) // batch_stage.size * batch_stage.size]
        whole = rows
        # The records after a resume's count go to the shares in turn from the first of them on
        # (`GlobalStream`), so a count that ends inside a batch hands this share the rows the share
        # that many places on holds.
        share = partition.first(offset)
        held, real = (partition.cut(offset), True) if self.training else partition.held(len(whole), offset)
        rows = whole[held]
        batches = rows.batch(size, drop_remainder=drop, batch_fn=_Batched(batch_stage.batch_fn))
        if not drop and fill:
            batches = batches.map(FilledBatch(size, real=real))
        batches = self.placed(batches, pipeline, partition, after)
        batches = batches[offset // batch_stage.size :]
        # A count of records that ends inside a batch, as a batch ramp saves, resumes past the
        # records of that batch the run read, in the order it read its batches: the share's rows
        # among the batch's first `read`, which the shares hold in turn.
        read = offset % batch_stage.size
        if read > share:
            batches = batches.map_with_index(_Rest(-(-(read - share) // partition.count)))
        return batches, tuple(at for at in after if isinstance(pipeline.stages[at], _UNPLACED))

    def elastic_batches(
        self, pipeline: Pipeline, partition: DataPartition, *, fill: bool = True
    ) -> tuple[pygrain.IterDataset, tuple[int, ...]]:
        """An elastic read's batches (`Pipeline.elastic`) and the stages after its batch. Every reader
        reads the order one reader reads (`DataPartition()`) through the last step that decides which
        rows there are, what they draw or what their place gives them (`_elastic_cut`), and keeps its
        share of each batch's rows, every count-th (`DataPartition.cut`), as a validation pass keeps
        them; the maps written after that step run on those rows alone (`_ElasticBatches`)."""
        batch_stage = next(stage for stage in pipeline.stages if isinstance(stage, BatchStage))
        batched_at = pipeline.stages.index(batch_stage)
        cut = _elastic_cut(pipeline.stages[:batched_at])
        rows = self.records(dataclasses.replace(pipeline, stages=pipeline.stages[:cut]), DataPartition())
        maps = tuple(stage for stage in pipeline.stages[cut:batched_at] if isinstance(stage, Map))
        drop = batch_stage.drop_remainder
        stream = _ElasticBatches(_as_iter(rows, self.loading), maps, partition, batch_stage.size, drop=drop,
                                 fill=fill and not drop, batch_fn=_Batched(batch_stage.batch_fn),
                                 threads=self.loading.threads)
        return stream, tuple(range(batched_at + 1, len(pipeline.stages)))

    def placed[Item](self, batches: pygrain.MapDataset[Item], pipeline: Pipeline, partition: DataPartition,
                     after: tuple[int, ...]) -> pygrain.MapDataset[Item]:
        """A global order's `batches` through the stages after its batch at `after` that hold a
        place in it (all but `_UNPLACED`), by index: a repeat replaying its first pass, a slice, a
        limit, a window shuffle. A run's batches and their sizes (`_order_records`) go through it."""
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
        return batches

    def after_batch(
        self, stream: pygrain.IterDataset, after: tuple[int, ...], pipeline: Pipeline,
        partition: DataPartition,
    ) -> pygrain.IterDataset:
        """The stages after the batch at `after`, those `batched` left, each as `stage` lowers it,
        with the loading's own workers last when the pipeline names none. Host steps run in the
        reader that wraps this stream."""
        stages = [(at, pipeline.stages[at]) for at in after]
        if self.loading.workers and self.pool is None and _at(pipeline.stages, Workers) is None:
            workers = Workers(self.loading.worker_count, self.loading.worker_buffer)
            stages.append((len(pipeline.stages), workers))
        for at, stage in stages:
            # A pooled read's workers read before its batch (`pooled`).
            if not isinstance(stage, MapBatch) and not (isinstance(stage, Workers) and self.pool is not None):
                stream = _as_iter(self.stage(stream, stage, pipeline, partition, at), self.loading)
        return stream



def _epoch_records(train: Pipeline, build: _Build) -> int | None:
    """The records in one epoch of a global order: each corpus's indexed order
    up to its first repeat, as the lowering builds it, so a slice, a limit and
    a token split count exactly. A mixture read once counts the records its mix
    reads before it ends, as the lowering builds it. One read on past its end,
    by a repeat that never ends, a training order's own (`_Build.batched`), or a
    mix of parts that never end, reads each part on from where it was, so it
    counts until every part has come round once (`Proportions.covering`), the
    parts it mixes counted alike; a part a slice or a limit bounds comes round
    once its bounded records are read (`covered`). An order that ends counts no
    more than it reads, whatever bounds it after its repeat. A repeat written before a cache counts the
    inputs a pass reads where the cache keeps one record for each, and none
    otherwise, ahead of reading them; phases are counted by none."""
    if not build.global_order:
        return None

    def passed(pipeline: Pipeline, build: _Build) -> int | None:
        """The records `pipeline` reads up to its first repeat, as the lowering builds them."""
        once = pipeline.stages[:_until(pipeline.stages, (Repeat,))]
        return held(dataclasses.replace(pipeline, stages=once), build)

    def held(pipeline: Pipeline, build: _Build) -> int | None:
        """The records `pipeline` holds through all its stages, as the lowering builds them."""
        cache = build.lowering.cache
        if cache is not None and cache >= len(pipeline.stages):
            # A cache after the stages counted, as a preset's cache follows the order written before
            # it: a pass reads each input once, and a cache of one record for each input holds that
            # many, where one that drops or expands them counts none ahead of reading them.
            if not build.lowering.one_to_one:
                return None
            build = dataclasses.replace(build, lowering=dataclasses.replace(build.lowering, cache=None))
        # A token split reads its rows as it is built, filling a cache before it; the probe seals it.
        with _closing(build.records(pipeline, DataPartition())) as rows:
            return len(rows) if isinstance(rows, pygrain.MapDataset) else None

    def viewed(pipeline: Pipeline) -> int | None:
        """Where the first repeat after the last slice or limit of `pipeline` reads the view they
        bound, past its stages where none follows; None where nothing bounds it."""
        stages = pipeline.stages
        bounds = [at for at, stage in enumerate(stages) if ends(stage)]
        if not bounds:
            return None
        return next((at for at, stage in enumerate(stages) if at > bounds[-1] and isinstance(stage, Repeat)),
                    len(stages))

    def covering(pipeline: Pipeline, build: _Build) -> int | None:
        """The records of `pipeline` in which every part it mixes, read on past its end, comes round
        once (`covered`); a corpus's pass."""
        source = pipeline.source
        if not isinstance(source, Mixture):
            return passed(pipeline, build)
        sizes = [covered(child, build.child(index)) for index, child in enumerate(source.pipelines)]
        known = [size for size in sizes if size is not None]
        if len(known) != len(sizes):
            return None
        return Proportions.of(_weights(source)).covering(known)

    def covered(pipeline: Pipeline, build: _Build) -> int | None:
        """The records of a part a mix reads on past its end in which it comes round once: a view a
        slice or a limit bounds once its bounded records are read, those the first repeat after its
        last bound reads or the bounded order itself where none follows; else its `covering`."""
        at = viewed(pipeline)
        if at is None:
            return covering(pipeline, build)
        return held(dataclasses.replace(pipeline, stages=pipeline.stages[:at]), build)

    repeated, batched = _until(train.stages, (Repeat,)), _until(train.stages, (BatchStage, TokenBudget))
    order = dataclasses.replace(train, stages=train.stages[:batched])
    # Whether its records never end, and whether they are read on past an end on the way: by a repeat
    # without a count, or as a mix of parts that never end.
    unending, onward = build.lowering.records_unending, build.lowering.onward
    # A training order that repeats nowhere repeats for ever at its root (`_Build.batched`).
    rooted = build.training and build.lowering.rooted
    if onward or rooted:
        # A view the order's repeat reads is its epoch, the read's own root repeat reading the whole
        # order; a bound after its last written repeat ends the order.
        at = viewed(order)
        whole = at == len(order.stages) and not rooted
        count = (covering(train, build) if at is None or whole
                 else held(dataclasses.replace(order, stages=order.stages[:at]), build))
    else:
        count = passed(train, build)
    if count is None or rooted or unending or (not onward and repeated >= batched):
        return count
    # An order that ends holds no more than it reads before its batch, whatever bounds it after its repeat.
    length = _order_length(train, build)
    return count if length is None else min(count, length)


def _order_length(train: Pipeline, build: _Build) -> int | None:
    """The records `train`'s order holds before its batch, all its passes, as the lowering builds
    them; None where its read counts none ahead."""
    order = dataclasses.replace(train, stages=train.stages[:_until(train.stages, (BatchStage, TokenBudget))])
    with _closing(build.records(order, DataPartition())) as rows:
        return len(rows) if isinstance(rows, pygrain.MapDataset) else None


def _order_records(train: Pipeline, build: _Build) -> int | None:
    """The records a training read of `train`'s global order hands out where it ends, the same on
    every share (`Loader.order_records`): the rows of every batch its read yields, through the
    stages after its batch, a repeat, a slice or a limit (`_Build.placed`), a kept short batch inside
    the order with its filled rows, which it marks, and the order's last batch with its real rows
    alone; None where it never ends or its read counts none ahead. The batches' sizes are what the
    stages place, so no record is read."""
    stage = next((stage for stage in train.stages if isinstance(stage, BatchStage)), None)
    if not build.global_order or build.lowering.endless or build.lowering.records_unending or stage is None:
        return None
    length = _order_length(train, build)
    if length is None:
        return None
    whole, short = divmod(length, stage.size)
    count = whole + (1 if short and not stage.drop_remainder else 0)
    if not count:
        # No batch to place: the read refuses an order of none, as it reads.
        return 0
    after = tuple(range(train.stages.index(stage) + 1, len(train.stages)))
    sizes = pygrain.MapDataset.range(count).map(lambda index: stage.size if index < whole else short)
    placed = build.placed(sizes, train, DataPartition(), after)
    if not len(placed):
        return 0
    return (len(placed) - 1) * stage.size + integer(placed[len(placed) - 1], "a batch's size")


def _tail(train: Pipeline, build: _Build) -> Tail:
    """What a training read of `train` does with the records an epoch leaves short of a batch: reads
    them, where its records never end (`_Lowering.records_unending`) or the read repeats them at its root
    (`_Build.batched`), so they lead the next epoch's batch, or where its batch keeps a short one;
    drops them where its order ends and its batch drops it."""
    drops = next((stage.drop_remainder for stage in train.stages if isinstance(stage, BatchStage)), True)
    rooted = build.training and build.lowering.rooted
    return "read" if build.lowering.records_unending or rooted or not drops else "dropped"


def _split_spans(pipeline: Pipeline, build: _Build) -> Iterator[list[tuple[int, int, int]]]:
    """The spans each token split in the lowered order keeps
    (`_Lowering.splits`), those of the corpora a pipeline mixes first: the
    lowering's own split (`_TokenSplit`) of the indexed order before it, built
    with the analysis of the pipeline it is written in, so they are the
    documents and token offsets a reader of the plan trains on."""
    for index, child in enumerate(children(pipeline)):
        yield from _split_spans(child, build.child(index))
    for at, split in enumerate(pipeline.stages):
        if at in build.lowering.splits and isinstance(split, TokenSplit):
            before = dataclasses.replace(pipeline, stages=pipeline.stages[:at])
            rows = build.records(before, DataPartition())
            if not isinstance(rows, pygrain.MapDataset):
                raise ValueError(f"token split in {_name(pipeline)} needs a finite indexed order")
            with _closing(rows) as read:
                spans = _TokenSplit(read, split).spans
            yield spans


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
