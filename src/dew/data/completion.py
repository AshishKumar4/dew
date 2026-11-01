"""A run bounded in examples counts each source example where the last row it trains in is delivered
(`dew.data.pipeline.CONSUMED`). A concatenating pack or windows over a stream lay one example across
several windows, its count at the last window holding a piece of it; a shuffle after them draws the windows
in another order, which could deliver that window before the rest of the example. Here each count waits
until every window holding a piece of its example is drawn.

The windows a share lays, in the layout's order, are its units, and each example's windows are one share's:
a share lays its own examples (a memory read holds the bound before the layout, `analysis._held_bound`). An
example's pieces lie in the units from the last one before its count that carried a count of its own (where
the example before it ended), less the units windows that overlap reach back (`reach`), through the one
carrying its count, so a unit's count is released once every unit of that span is drawn. A later shuffle,
or a step after the layout that may make several rows of a window, leaves no such order, so a count waits
for every unit before its own (`reach` None). A stream's units are numbered before the shuffle and released
after it (`Numbered`, `Completed`), holding no more than the shuffle's window of them; a memory read's,
which it can read twice, by a table of the place its shuffle draws each unit (`released`)."""

from __future__ import annotations

import collections
import dataclasses
import heapq
from collections.abc import Mapping, Sequence

import grain.experimental as gx
import grain.python as pygrain
import numpy as np

from dew._records import integer, integers, listed, record
from dew.batch import Batch
from dew.data.pipeline import CONSUMED, REFUSED, Pack, Rejected, Stage, Windows

UNIT = "_dew_unit"
"""A window's place among the units its share's layout made, in their order (`Numbered`)."""

SINCE = "_dew_since"
"""The first unit that can hold a piece of the examples a unit counts (`Numbered`)."""


def accounts(row: Batch | Rejected) -> tuple[int, int]:
    """The source examples `row` consumes and the part of them refused (`CONSUMED`, `REFUSED`): a rejected
    row's are all refused, as the row kept after it is paid them (`dew.data.pipeline.Owed`)."""
    if isinstance(row, Rejected):
        return row.consumed, row.consumed
    return (integer(row.get(CONSUMED, 0), "a row's consumed examples"),
            integer(row.get(REFUSED, 0), "a row's refused examples"))


def cuts(stages: Sequence[Stage]) -> bool:
    """Whether `stages` lay a source example across windows: a concatenating pack, which cuts documents
    where its windows end, or windows over a stream of them. A first-fit or best-fit pack places each whole
    in one window."""
    return any(isinstance(stage, Windows) or (isinstance(stage, Pack) and stage.policy == "concat")
               for stage in stages)


def reach(layout: Pack | Windows) -> int:
    """How many units before the last one holding a count of its own the next example's pieces can start
    in: none for a pack, whose windows share no token; for windows `stride` apart, each holding
    `seq_len + 1` tokens, the others that can hold a token with the window after that count."""
    if isinstance(layout, Pack):
        return 0
    stride = layout.stride or layout.seq_len
    return max(0, -(-(layout.seq_len + 1) // stride) - 1)


class Numbered(pygrain.IterDataset):
    """A share's windows, each with its place among them (`UNIT`) and the first unit its examples' pieces
    can lie in (`SINCE`): `reach` units before the last one that held a count, or the stream's first unit
    where the windows reach the shuffle out of their layout's order (`reach` None)."""

    def __init__(self, parent: pygrain.IterDataset, *, reach: int | None):
        super().__init__(parent)
        self._rows, self._reach = parent, reach

    def __iter__(self) -> pygrain.DatasetIterator:
        return _NumberedIterator(self._rows.__iter__(), reach=self._reach)


class _NumberedIterator(pygrain.DatasetIterator):
    def __init__(self, parent: pygrain.DatasetIterator, *, reach: int | None):
        super().__init__(parent)
        self._rows, self._next, self._since, self._reach = parent, 0, 0, reach

    def __next__(self) -> Batch:
        row = record(next(self._rows), "a laid-out window")
        unit, since = self._next, self._next
        self._next += 1
        if any(accounts(row)):
            since = 0 if self._reach is None else max(0, self._since - self._reach)
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


def released(units: Numbered, window: int | None, seed: int) -> pygrain.IterDataset:
    """`units`, a memory read's numbered windows (`Numbered`), shuffled `window` at a time with `seed` (all
    of them, `window` None), each with the counts released where it is drawn. The read can draw its units
    twice, so a first pass reads each unit's counts and span (`SINCE`), and the shuffle's draws of as many
    numbers, which depend on its seed and how many it draws and not on what they are, place each unit: a
    unit's counts are released with the unit drawn where its span's last unit is. No position holds more
    than the shuffle's own."""
    rows = (record(row, "a numbered window") for row in units)
    held = [(integer(row[UNIT], "a window's unit"), integer(row[SINCE], "a window's first unit"),
             *accounts(row)) for row in rows]
    units_count = len(held)
    size = window or max(units_count, 1)
    drawn = np.fromiter(gx.WindowShuffleIterDataset(pygrain.MapDataset.range(units_count).to_iter_dataset(),
                                                    window_size=size, seed=seed), np.int64, units_count)
    place_of = np.empty(units_count, np.int64)
    place_of[drawn] = np.arange(units_count)
    table = np.zeros((units_count, 2), np.int64)
    # The latest place among each span's units, the spans' first units never falling back (`Numbered`):
    # the units still in the latest span, by falling place.
    latest: collections.deque[int] = collections.deque()
    for unit, since, consumed, refused in held:
        while latest and place_of[latest[-1]] <= place_of[unit]:
            latest.pop()
        latest.append(unit)
        if consumed or refused:
            while latest[0] < since:
                latest.popleft()
            table[drawn[place_of[latest[0]]]] += (consumed, refused)
    shuffled = gx.WindowShuffleIterDataset(units, window_size=size, seed=seed)
    return shuffled.map(_Release(table))


@dataclasses.dataclass(frozen=True)
class _Release:
    """A drawn unit with the counts its draw releases (`released`), in place of its own."""

    table: np.ndarray

    def __call__(self, row: Batch) -> Batch:
        window = dict(row)
        unit = integer(window.pop(UNIT), "a window's unit")
        window.pop(SINCE)
        consumed, refused = self.table[unit]
        return {**window, CONSUMED: int(consumed), REFUSED: int(refused)}
