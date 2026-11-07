"""A locality order over physical shards, without materializing record permutations."""

from __future__ import annotations

import threading
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
