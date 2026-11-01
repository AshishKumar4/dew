"""A run bounded in examples counts each source example where the last row it trains in is delivered
(`dew.data.pipeline.CONSUMED`). A concatenating pack or windows over a stream lay one example across
several windows, its count at the window holding its last token; a shuffle after them draws the windows in
another order, which could deliver that window before the rest of the example. Here each count waits until
every window holding a piece of its example is delivered.

The windows a share reads, in the order the layout made them, are its units. An example's pieces lie in the
units from the last one before its count that carried a count of its own (where the example before it
ended) through the one carrying its count, so a unit's count is released once every unit of that span is
drawn. A later shuffle draws units no layout ordered, so a count it draws waits for every unit before its
own (`prefix`). A stream's units are numbered before the shuffle and released after it (`Numbered`,
`Completed`), holding no more than the shuffle's window of them; an indexed read's are released at the
place its order draws the span's last unit, which a table of that order states (`Released`). A memory read
whose shares each read their own order of the windows it holds deals them by whole examples
(`whole_examples`), so no example lies across two."""

from __future__ import annotations

import dataclasses
import heapq
from collections.abc import Mapping, Sequence
from typing import overload

import grain.python as pygrain
import numpy as np

from dew._records import integer, integers, listed, record
from dew.batch import Batch
from dew.data.pipeline import CONSUMED, REFUSED, Record, Rejected

UNIT = "_dew_unit"
"""A window's place among the units its share's layout made, in their order (`Numbered`)."""

SINCE = "_dew_since"
"""The first unit that can hold a piece of the examples a unit counts: the last one before it that carried
a count, or the stream's first, which every unit after a shuffle the layout did not order counts from
(`Numbered`)."""

CLOSES = "_dew_closes"
"""Whether a window a memory read holds ends where an example ends, so the window after it holds none of the
examples it holds (`whole_examples`)."""


def accounts(row: Record) -> tuple[int, int]:
    """The source examples `row` consumes and the part of them refused (`CONSUMED`, `REFUSED`): a rejected
    row's are all refused, as the row kept after it is paid them (`dew.data.pipeline.Owed`)."""
    if isinstance(row, Rejected):
        return row.consumed, row.consumed
    if isinstance(row, Mapping):
        return (integer(row.get(CONSUMED, 0), "a row's consumed examples"),
                integer(row.get(REFUSED, 0), "a row's refused examples"))
    return 0, 0


def closes(window: Batch) -> bool:
    """Whether `window`, a layout's window holding its tokens' counts of examples token by token (each at
    its example's last token), ends with an example: its last token holds a count."""
    return any(int(np.asarray(window[name])[-1]) for name in (CONSUMED, REFUSED) if len(window[name]))


def whole_examples(closed: Sequence[bool], share: int, count: int) -> list[int]:
    """The windows of share `share` of `count`, as their indices among `closed`, each saying whether its
    window closes (`CLOSES`): the runs of windows that hold whole examples, each ending at a window that
    closes, dealt to the shares in turn, so each example's windows are one share's, in their order."""
    held, run, start = [], 0, 0
    for index, ends in enumerate(closed):
        if ends or index == len(closed) - 1:
            if run % count == share:
                held.extend(range(start, index + 1))
            run, start = run + 1, index + 1
    return held


class Numbered(pygrain.IterDataset):
    """A share's windows, each with its place among them (`UNIT`) and the first unit its examples' pieces
    can lie in (`SINCE`), for `Completed` to read after a shuffle: the stream's first for every unit where
    the windows are not in their layout's order (`prefix`)."""

    def __init__(self, parent: pygrain.IterDataset, *, prefix: bool = False):
        super().__init__(parent)
        self._rows, self._prefix = parent, prefix

    def __iter__(self) -> pygrain.DatasetIterator:
        return _NumberedIterator(self._rows.__iter__(), prefix=self._prefix)


class _NumberedIterator(pygrain.DatasetIterator):
    def __init__(self, parent: pygrain.DatasetIterator, *, prefix: bool):
        super().__init__(parent)
        self._rows, self._next, self._since, self._prefix = parent, 0, 0, prefix

    def __next__(self) -> Batch:
        row = record(next(self._rows), "a laid-out window")
        unit, since = self._next, self._next
        self._next += 1
        if any(accounts(row)):
            since = self._since
            if not self._prefix:
                self._since = unit
        return {**row, UNIT: unit, SINCE: since}

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._rows.get_state(), "next": self._next, "since": self._since}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._rows.set_state(record(state["parent"], "numbered windows"))
        self._next = integer(state["next"], "the next window's unit")
        self._since = integer(state["since"], "the last counting window's unit")

    def close(self) -> None:
        self._rows.close()


class Completed(pygrain.IterDataset):
    """Numbered windows (`Numbered`) as a shuffle draws them, each with the counts of the examples whose
    every unit is drawn by it: its own where its span is, else held until the span's last unit is drawn,
    whose window takes them."""

    def __init__(self, parent: pygrain.IterDataset):
        super().__init__(parent)
        self._rows = parent

    def __iter__(self) -> pygrain.DatasetIterator:
        return _CompletedIterator(self._rows.__iter__())


@dataclasses.dataclass
class _Held:
    """A unit's counts, held until every unit from `since` through it is drawn, `missing` of them not yet."""

    since: int
    consumed: int
    refused: int
    missing: int = 0


class _CompletedIterator(pygrain.DatasetIterator):
    """Below `floor` every unit is drawn, and of the units past it, those in `drawn`: as a window shuffle
    draws them, no more of either than its window holds, nor of the counts held (`held`)."""

    def __init__(self, parent: pygrain.DatasetIterator):
        super().__init__(parent)
        self._rows, self._floor, self._drawn = parent, 0, set[int]()
        self._held: dict[int, _Held] = {}
        # The units each held count waits for, by the unit it waits for; a count whose span starts at or
        # below the floor waits for the floor to pass it (`_floored`), the units below it all drawn.
        self._waiting: dict[int, list[int]] = {}
        self._floored: list[int] = []

    def __next__(self) -> Batch:
        window = dict(record(next(self._rows), "a numbered window"))
        unit = integer(window.pop(UNIT), "a window's unit")
        since = integer(window.pop(SINCE), "a window's first unit")
        self._drawn.add(unit)
        while self._floor in self._drawn:
            self._drawn.remove(self._floor)
            self._floor += 1
        released = [0, 0]
        while self._floored and self._floored[0] < self._floor:
            self._release(heapq.heappop(self._floored), released)
        for waiting in self._waiting.pop(unit, ()):
            held = self._held[waiting]
            held.missing -= 1
            if not held.missing:
                self._release(waiting, released)
        consumed, refused = accounts(window)
        if consumed or refused:
            self._held[unit] = _Held(since, consumed, refused)
            if self._wait(unit):
                self._release(unit, released)
        return {**window, CONSUMED: released[0], REFUSED: released[1]}

    def _wait(self, unit: int) -> bool:
        """Hold `unit`'s counts until the units of its span not yet drawn are; whether none is left."""
        held = self._held[unit]
        if unit < self._floor:
            return True
        if held.since <= self._floor:
            heapq.heappush(self._floored, unit)
            return False
        for missing in range(held.since, unit):
            if missing not in self._drawn:
                self._waiting.setdefault(missing, []).append(unit)
                held.missing += 1
        return not held.missing

    def _release(self, unit: int, released: list[int]) -> None:
        held = self._held.pop(unit)
        released[0] += held.consumed
        released[1] += held.refused

    def get_state(self) -> Mapping[str, object]:
        held = [[unit, counts.since, counts.consumed, counts.refused] for unit, counts in self._held.items()]
        return {"parent": self._rows.get_state(), "floor": self._floor, "drawn": sorted(self._drawn),
                "held": held}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._rows.set_state(record(state["parent"], "completed windows"))
        self._floor = integer(state["floor"], "the units all drawn")
        self._drawn = set(integers(state["drawn"], "the units drawn"))
        self._held, self._waiting, self._floored = {}, {}, []
        for unit, since, consumed, refused in listed(state["held"], "held counts", read=integers,
                                                     what="a list of held counts"):
            self._held[unit] = _Held(since, consumed, refused)
            self._wait(unit)

    def close(self) -> None:
        self._rows.close()


class Released(pygrain.MapDataset):
    """Indexed windows `rows`, a share's units in the order its layout made them (or in another, `prefix`),
    read in `order`, the same reorder of their indices (`order[i]` the unit read at place `i`): each
    example's counts at the first kept row from the place where its span's last unit is read, a rejected
    row carrying none of its own. Each pass of the order (`len(rows)` places) is tabled once, as it is
    first read."""

    def __init__(self, rows: pygrain.MapDataset, order: pygrain.MapDataset, *, prefix: bool = False):
        super().__init__(rows)
        self._rows, self._order, self._prefix = rows, order, prefix
        self._spans: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        # The passes tabled, the one read last and the one before, as a read ahead crosses between them.
        self._tables: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._order)

    @overload
    def __getitem__(self, index: slice) -> pygrain.MapDataset: ...

    @overload
    def __getitem__(self, index: int) -> Record: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.slice(index)
        row = self._rows[int(self._order[index])]
        if isinstance(row, Rejected):
            return dataclasses.replace(row, consumed=0)
        if not isinstance(row, Mapping):
            return row
        epoch, place = divmod(index, len(self._rows))
        table = self._tables.get(epoch)
        if table is None:
            table = self._released(epoch)
            self._tables = {held: tabled for held, tabled in self._tables.items() if held == epoch - 1}
            self._tables[epoch] = table
        return {**row, CONSUMED: int(table[0, place]), REFUSED: int(table[1, place])}

    def _units(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each unit's counts, consumed and refused, and whether its row is kept."""
        if self._spans is None:
            rows = [self._rows[unit] for unit in range(len(self._rows))]
            counts = np.asarray([accounts(row) for row in rows], np.int64).reshape(-1, 2)
            self._spans = counts[:, 0], counts[:, 1], np.asarray([isinstance(row, Mapping) for row in rows])
        return self._spans

    def _released(self, epoch: int) -> np.ndarray:
        """The counts each place of pass `epoch` of the order releases, consumed and refused: a counting
        unit's at the place its span's last unit is read, each span running from the counting unit before
        it (or the first unit, `prefix`) through it. A count with no kept row from its place on is the
        stream's last, which its end credits (`dew.training._fit.drained`)."""
        consumed, refused, kept = self._units()
        units = len(consumed)
        drawn = np.asarray([self._order[epoch * units + place] for place in range(units)], np.int64)
        place_of = np.empty(units, np.int64)
        place_of[drawn] = np.arange(units)
        counting = np.flatnonzero((consumed != 0) | (refused != 0))
        released = np.zeros((2, units), np.int64)
        if not len(counting):
            return released
        if self._prefix:
            last = np.maximum.accumulate(place_of)[counting]
        else:
            # Span k runs from counting unit k - 1 (the first unit for k = 0) through counting unit k.
            starts = np.concatenate([[0], counting[:-1]])
            last = np.maximum(np.maximum.reduceat(place_of[:counting[-1] + 1], starts), place_of[counting])
        kept_places = np.flatnonzero(kept[drawn])
        at = np.searchsorted(kept_places, last)
        placed = at < len(kept_places)
        targets = kept_places[at[placed]]
        np.add.at(released[0], targets, consumed[counting][placed])
        np.add.at(released[1], targets, refused[counting][placed])
        return released
