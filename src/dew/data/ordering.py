"""A locality order over physical shards, without materializing record permutations; a streamed read's
windowed shuffle, its windows read as the one before is given (`WindowShuffle`)."""

from __future__ import annotations

import copy
import threading
from collections.abc import Mapping
from typing import overload

import grain.experimental as gx
import grain.python as grain
import numpy as np

from dew.batch import Batch
from dew.data.dataset import Ranges


class BlockShuffle(grain.MapDataset[Batch]):
    def __init__(self, parent: grain.MapDataset[Batch], sizes: tuple[int, ...], block: int, seed: int):
        super().__init__(parent)
        if not sizes or any(size < 0 for size in sizes) or sum(sizes) == 0:
            raise ValueError("block shuffle needs authoritative positive physical shard counts")
        self._rows, self._sizes, self._block, self._seed = parent, sizes, block, seed
        self._shards = Ranges(sizes)
        self._total = sum(sizes)
        self._epochs: dict[int, tuple[tuple[int, ...], Ranges]] = {}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        del state["_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._rows)

    def _epoch(self, epoch: int) -> tuple[tuple[int, ...], Ranges]:
        cached = self._epochs.get(epoch)
        if cached is None:
            with self._lock:
                cached = self._epochs.get(epoch)
                if cached is None:
                    order = tuple(int(index) for index in np.random.default_rng(
                        [self._seed, epoch]).permutation(len(self._sizes)))
                    blocks = Ranges(tuple(
                        sum(self._sizes[index] for index in order[start : start + self._block])
                        for start in range(0, len(order), self._block)
                    ))
                    cached = order, blocks
                    # Published dictionaries never change under concurrent readers.
                    self._epochs = {**dict(list(self._epochs.items())[-1:]), epoch: cached}
        return cached

    def index(self, index: int) -> int:
        epoch, position = divmod(index, self._total)
        order, blocks = self._epoch(epoch)
        block, place = blocks.locate(position)
        members = order[block * self._block : (block + 1) * self._block]
        seed = int(np.random.SeedSequence([self._seed, epoch, block]).generate_state(1)[0])
        # Grain's shuffle of an integer range, `MapDataset.range(size).shuffle(seed)`, at one index
        # of its first epoch (`ShuffleMapDataset._shuffled_index`).
        offset = gx.index_shuffle(place, max_index=blocks.counts[block] - 1, seed=seed, rounds=4)
        for member in members:
            if offset < self._sizes[member]:
                return epoch * self._total + self._shards.start(member) + offset
            offset -= self._sizes[member]
        raise RuntimeError("physical shard block does not cover its record index")

    @overload
    def __getitem__(self, index: slice) -> grain.MapDataset[Batch]: ...
    @overload
    def __getitem__(self, index: int) -> Batch | None: ...
    def __getitem__(self, index: int | slice) -> Batch | grain.MapDataset[Batch] | None:
        return self.slice(index) if isinstance(index, slice) else self._rows[self.index(index)]


class WindowShuffle(grain.IterDataset[Batch]):
    """A streamed read's records shuffled within each `window` of them, in the order Grain's window shuffle
    gives (`grain.experimental.WindowShuffleIterDataset`: window k of the parent's records drawn by
    `index_shuffle(position, window - 1, seed + k, rounds=4)` and given from the last), its position the
    one Grain saves, so a saved run reads and resumes as it did. Each window after the first is read as
    the one before is given, a record of it for each record given, where Grain read each whole as its first
    record was asked: a 16k-record window's read stalled the step that asked it 10 s and held the GIL
    against the steps' dispatch (row 4740)."""

    def __init__(self, parent: grain.IterDataset[Batch], *, window: int, seed: int):
        super().__init__(parent)
        self._rows, self._window, self._seed = parent, window, seed

    def __iter__(self) -> _WindowShuffled:
        return _WindowShuffled(self._rows.__iter__(), window=self._window, seed=self._seed)


class _WindowShuffled(grain.DatasetIterator[Batch]):
    def __init__(self, parent: grain.DatasetIterator[Batch], *, window: int, seed: int):
        super().__init__(parent)
        self._read, self._size, self._seed = parent, window, seed
        self._index = self._given = 0
        self._window: list[Batch] = []
        self._start: Mapping[str, object] = parent.get_state()
        # The next window's start, captured once the current one is read, and its records read so far, one
        # a record given; None where the parent ended or no window was read yet.
        self._next: Mapping[str, object] | None = None
        self._ahead: list[Batch] = []
        self._ended = False

    def _read_ahead(self, count: int) -> None:
        while count > 0 and not self._ended:
            try:
                self._ahead.append(next(self._read))
            except StopIteration:
                self._ended = True
            count -= 1

    def _drawn(self, records: list[Batch]) -> list[Batch]:
        """`records` in window `_index`'s order, Grain's (`_reshuffle_window`)."""
        last = len(records) - 1
        return [records[gx.index_shuffle(position, max_index=last, seed=self._seed + self._index, rounds=4)]
                for position in range(len(records))]

    def _opened(self) -> None:
        """The next window, read whole where none was read ahead (the first, a restored one's next)."""
        if self._next is None:
            self._start = self._read.get_state()
            self._read_ahead(self._size)
        else:
            self._start = self._next
            self._read_ahead(self._size - len(self._ahead))
        records, self._ahead = self._ahead, []
        self._window, self._given = self._drawn(records), 0
        self._next = None if self._ended else self._read.get_state()

    def __next__(self) -> Batch:
        if not self._window:
            if self._ended and self._next is None and not self._ahead:
                raise StopIteration
            self._opened()
            if not self._window:
                raise StopIteration
        if self._next is not None:
            self._read_ahead(1)
        self._given += 1
        record = self._window.pop()
        if not self._window:
            self._index += 1
        return record

    def get_state(self) -> Mapping[str, object]:
        return {"parent_window_start_state": copy.deepcopy(self._start), "window_index": self._index,
                "pos_in_window": self._given}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._start = copy.deepcopy(state["parent_window_start_state"])
        self._read.set_state(self._start)
        self._index, self._given = int(state["window_index"]), int(state["pos_in_window"])
        self._ended, self._ahead, self._next = False, [], None
        self._read_ahead(self._size)
        records, self._ahead = self._ahead, []
        self._window = self._drawn(records)
        for _ in range(min(self._given, len(self._window))):
            self._window.pop()
        self._next = None if self._ended else self._read.get_state()

