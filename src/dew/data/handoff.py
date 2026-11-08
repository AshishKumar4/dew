"""Where a pooled read's loading workers hand its rows to the reading process
(`dew.data.lowering._Build.records`) and what crosses: the stage they hand on at (`_carried`; an in-order
read's, `_handoff`), the groups of records and markers a worker sends (`_Grouped`, `_Ungrouped`), an input's
rows its workers expand (`_Spread`), and the repeat of a worker's inputs that ends where a pass holds none
(`_Ended`)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import grain.python as pygrain
import numpy as np

from dew._records import boolean, integer, record
from dew.batch import Batch
from dew.data.analysis import _drops
from dew.data.dataset import _column, _shareable, _stacked
from dew.data.pipeline import (
    ACCOUNTS,
    Filter,
    FlatMap,
    Limit,
    Map,
    MapWithIndex,
    Native,
    Pipeline,
    Prefetch,
    RandomMap,
    Record,
    Rejected,
    Repeat,
    Shuffle,
    Slice,
    Stage,
    consumed_forward,
)
from dew.data.sources.urls import Fetch
from dew.data.text import Chunks

type RecordStage = Map | RandomMap | MapWithIndex | Filter | FlatMap


_RECORD = (Map, RandomMap, MapWithIndex, Filter, FlatMap)


def _threads(stages: Sequence[Stage]) -> int | None:
    """The most records a fetch among `stages` runs at once on a pool of its own (`Fetch.threads`),
    or None where none has one."""
    pools = [stage.step.threads for stage in stages
             if isinstance(stage, Map) and isinstance(stage.step, Fetch) and stage.step.threads is not None]
    return max(pools, default=None)


def _carried(pipeline: Pipeline, consumed: int, *, marked: bool, windows: bool = False) -> int:
    """Where a pooled read's loading workers hand its rows to the reading process (`lowering._Build.pooled`):
    at its first stage from `consumed` that can't carry a dropped record as a marker. Record steps
    and repeats carry them. A Grain step of the caller's reads no marker, so it runs in the workers
    only before any: over indexed rows, where what it drops reads as None (`_Marked`); over a
    stream, or where it makes one, the workers hand on around it (`lowering._Build.records`). A view of the
    inputs (a shuffle, slice or take) reads markers as records, so it ends the workers' part once a
    step may have dropped (`marked`); a window shuffle ends it, but where `windows`, an in-order
    read's, whose workers read the view of its indexed rows as one reader does
    (`lowering._Build.in_order`). Every other stage lays out, mixes or batches several records into
    one and ends it."""
    for at, stage in enumerate(pipeline.stages[consumed:], start=consumed):
        if isinstance(stage, (Map, RandomMap, MapWithIndex, Filter, FlatMap)):
            marked = marked or _drops(stage)
        elif isinstance(stage, Repeat) or (isinstance(stage, Native) and not marked):
            # The caller's Grain step reads no marker: it runs in the workers before any.
            marked = marked or isinstance(stage, Native)
        elif not (isinstance(stage, (Shuffle, Slice, Limit, Repeat)) and not marked
                  and (windows or not (isinstance(stage, Shuffle) and stage.window is not None))):
            return at
    return len(pipeline.stages)


def _handoff(stages: Sequence[Stage], start: int, *, marked: bool, elastic: bool = True,
             spread: bool = False) -> int:
    """Where an elastic read's workers hand its rows on (`elastic_order._InputOrder`), at the latest:
    before the first step that may make its indexed rows a stream (an expansion, a window shuffle, a Grain
    step), which no index reads, and before a random step after one that may drop, which one
    reader keys by its place among the rows kept, not among the inputs with their markers. The
    reading process runs the rest in that order, so its rows and draws are one reader's. A read cut
    per share, not `elastic`, runs a Grain step before it, over its indexed rows, and hands on after
    one that makes a stream (`lowering._Build.records`); a window shuffle of its indexed rows before any drop
    there too, a view of them as any shuffle is (`lowering._reordered`); and hands on before a repeat after a
    step that may drop, which a read in none repeats as the stream of the rows it kept, each pass alike,
    where indexed inputs would be drawn again each pass (Grain's shuffle). A read that `spread`s each input
    (`_Spread`) runs its expansion and the record steps after it in its workers too, each draw keyed by
    its input and its row in it, and hands on before the first stage after its expansion that is no
    record step, or before a window shuffle or a Grain step."""
    def windowed(stage: Stage, streamed: bool) -> bool:
        return isinstance(stage, Shuffle) and stage.window is not None and (elastic or streamed)

    kept = (RandomMap,) if elastic else (RandomMap, Repeat)
    if spread:
        expanded = False
        for at, stage in enumerate(stages[start:], start=start):
            pooled = isinstance(stage, _RECORD) and _threads((stage,)) is not None
            # A step of each row's index reads a row's place in the stream, which an input's rows share.
            placed = expanded and isinstance(stage, MapWithIndex)
            grain = isinstance(stage, Native) and (elastic or expanded)
            # Before the expansion, a draw after a drop is keyed by its place among the rows kept.
            drawn = not expanded and marked and isinstance(stage, kept)
            # An expansion whose counts the reading process keeps (a cut into chunks, `Chunked`; a later
            # objective's of the source examples it makes rows of, `FlatMap.examples`) runs there.
            counted = isinstance(stage, FlatMap) and (
                isinstance(stage.step, Chunks) or (expanded and stage.examples))
            if windowed(stage, expanded or marked) or pooled or placed or grain or drawn or counted or (
                    expanded and not isinstance(stage, (*_RECORD, Native))):
                return at
            expanded = expanded or isinstance(stage, FlatMap)
            marked = marked or _drops(stage)
        return len(stages)
    for at, stage in enumerate(stages[start:], start=start):
        grain = elastic and isinstance(stage, Native)
        if windowed(stage, marked) or isinstance(stage, FlatMap) or grain or (
                marked and isinstance(stage, kept)):
            return at
        marked = marked or _drops(stage)
    return len(stages)


def _elastic_cut(stages: Sequence[Stage]) -> int:
    """Where an elastic read's own rows begin (`lowering._Build.elastic_batches`): after the last step before
    the batch that is not a `Map` of each record alone, one record for one (or a `Prefetch`, which
    reads none ahead there). Every reader runs the steps before it over the whole order, as each of
    them decides which rows there are (a filter, a map that rejects records, an expansion, a pack),
    which draws they take or what their place gives them, which only the whole order states; a map
    after them computes a row from it alone."""
    def one_for_one(stage: Stage) -> bool:
        return isinstance(stage, Prefetch) or (isinstance(stage, Map) and not stage.rejects)
    return max((at + 1 for at, stage in enumerate(stages) if not one_for_one(stage)), default=0)


type Field = Batch | np.ndarray | np.generic | str | bytes | int | float | bool | None | list | tuple
"""A record's field, as the steps before a batch leave it."""


class _Spread(tuple):
    """What a run of record stages from an expansion makes of one input in a read's workers
    (`lowering._PerInput`): its rows and the markers of what the steps after the expansion dropped, in
    order."""

    __slots__ = ()


def _input_draws(seed: int | None, index: int, row: int) -> np.random.Generator | None:
    """A spread input's random stage's draws for its `row`th row there (`lowering._PerInput`), from the
    stage's seed and the input's `index`; none for a stage that draws none."""
    return None if seed is None else np.random.default_rng((seed, index, row))


@dataclasses.dataclass(frozen=True)
class _Unspread:
    """An input's spread rows and markers (`_Spread`), in the reading process."""

    max_fan_out: int = 1 << 20

    def flat_map(self, element: Record | _Spread) -> Sequence[Record]:
        return tuple(element) if isinstance(element, _Spread) else (element,)


@dataclasses.dataclass
class _Grouped:
    """A loading worker's records and the markers of those its steps dropped, a batch's count at a
    time (`lowering._Build.handed`), none of the records a store misses: where the records hold the same
    fields, field by field, each field's arrays of one shape and type stacked as a batch's are, into
    shared memory once Grain asks for it, and any other field's values listed; the markers beside
    them."""

    shared: bool = False

    def enable_shared_memory_output(self) -> None:
        self.shared = True

    def __call__(self, elements: Sequence[Record | None]) -> Batch:
        # A record a store read by index misses is none, which the read leaves out, as Grain's iterating
        # a dataset of them does.
        elements = consumed_forward([element for element in elements if element is not None])
        markers = [element for element in elements if isinstance(element, Rejected)]
        rows = [element for element in elements if not isinstance(element, Rejected)]
        records = [row for row in rows if isinstance(row, Mapping)]
        alike = (bool(rows) and len(records) == len(rows)
                 and all(row.keys() == records[0].keys() for row in records))
        # The source examples each row consumes are the run's accounting, no batch column: they cross as
        # the integers they are, which the reading process counts (`_Ungrouped`, `Accepted`).
        held = ({name: [row[name] for row in records] if name in ACCOUNTS else
                 self.column([row[name] for row in records]) for name in records[0]} if alike else rows)
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
        markers = [marker for marker in group["markers"] if not marker.consumed]
        # A marker of the examples the group's last rows were the last of follows its rows.
        trailing = [marker for marker in group["markers"] if marker.consumed]
        held, count = group["held"], integer(group["rows"], "grouped rows")
        if not group["alike"]:
            return [*markers, *held, *trailing]
        columns: dict[str, np.ndarray | list] = {}
        for name, column in record(held, "grouped records").items():
            # A batch's column: an array, or a list of a field's values that don't stack.
            if not isinstance(column, (np.ndarray, list)):
                raise TypeError(f"a grouped column {name!r} is a {type(column).__name__}, not a batch column")
            columns[name] = column
        rows = ({name: column[index] for name, column in columns.items()} for index in range(count))
        return [*markers, *rows, *trailing]


class _Barren(Exception):
    """A pass of a loading worker's inputs, each read as a row or the marker of a dropped record,
    that held nothing: the worker holds no input (`lowering._repeated`)."""


class _Ended(pygrain.IterDataset):
    """A repeat of a loading worker's inputs, which ends where a pass of them holds none (`_Barren`)
    instead of repeating nothing forever, so the other workers read on."""

    def __init__(self, parent: pygrain.IterDataset):
        super().__init__(parent)
        self._rows = parent

    def __iter__(self) -> pygrain.DatasetIterator:
        return _EndedIterator(self._rows.__iter__())


class _EndedIterator(pygrain.DatasetIterator):
    """Its parent's rows until a pass of them holds none. Restored ended, it reads no more and
    keeps the position it was given, the counts of its corpus's stages with it."""

    def __init__(self, parent: pygrain.DatasetIterator):
        super().__init__(parent)
        self._rows, self._ended = parent, False
        self._held: Mapping[str, object] | None = None

    def __next__(self):
        if self._ended:
            raise StopIteration
        try:
            return next(self._rows)
        except _Barren:
            self._ended = True
            raise StopIteration from None

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._rows.get_state() if self._held is None else self._held, "ended": self._ended}

    def set_state(self, state: Mapping[str, object]) -> None:
        # An ended repeat reads no more, so only one that reads on restores where it was.
        self._ended = boolean(state["ended"], "repeat ended")
        parent = record(state["parent"], "repeat position")
        self._held = parent if self._ended else None
        if not self._ended:
            self._rows.set_state(parent)
