"""A locality order over physical shards, without materializing record permutations."""

from __future__ import annotations

import threading
from typing import overload

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch


class BlockShuffle(grain.MapDataset[Batch]):
    def __init__(self, parent: grain.MapDataset[Batch], sizes: tuple[int, ...], block: int, seed: int):
        super().__init__(parent)
        if not sizes or any(size < 0 for size in sizes) or sum(sizes) == 0:
            raise ValueError("block shuffle needs authoritative positive physical shard counts")
        self._rows, self._sizes, self._block, self._seed = parent, sizes, block, seed
        self._starts = np.cumsum((0, *sizes[:-1]))
        self._total = sum(sizes)
        self._epochs: dict[int, tuple[tuple[int, ...], np.ndarray]] = {}
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

    def _epoch(self, epoch: int) -> tuple[tuple[int, ...], np.ndarray]:
        cached = self._epochs.get(epoch)
        if cached is None:
            with self._lock:
                cached = self._epochs.get(epoch)
                if cached is None:
                    order = tuple(int(index) for index in np.random.default_rng(
                        [self._seed, epoch]).permutation(len(self._sizes)))
                    ends = np.cumsum([
                        sum(self._sizes[index] for index in order[start : start + self._block])
                        for start in range(0, len(order), self._block)
                    ])
                    cached = order, ends
                    # Published dictionaries never change under concurrent readers.
                    self._epochs = {**dict(list(self._epochs.items())[-1:]), epoch: cached}
        return cached

    def index(self, index: int) -> int:
        epoch, position = divmod(index, self._total)
        order, ends = self._epoch(epoch)
        block = int(np.searchsorted(ends, position, side="right"))
        members = order[block * self._block : (block + 1) * self._block]
        start = 0 if block == 0 else int(ends[block - 1])
        size = int(ends[block]) - start
        seed = int(np.random.SeedSequence([self._seed, epoch, block]).generate_state(1)[0])
        picked = grain.MapDataset.range(size).shuffle(seed)[position - start]
        if picked is None:
            raise RuntimeError("Grain index shuffle filtered an integer range")
        offset = int(picked)
        for member in members:
            if offset < self._sizes[member]:
                return epoch * self._total + int(self._starts[member]) + offset
            offset -= self._sizes[member]
        raise RuntimeError("physical shard block does not cover its record index")

    @overload
    def __getitem__(self, index: slice) -> grain.MapDataset[Batch]: ...
    @overload
    def __getitem__(self, index: int) -> Batch | None: ...
    def __getitem__(self, index: int | slice) -> Batch | grain.MapDataset[Batch] | None:
        return self.slice(index) if isinstance(index, slice) else self._rows[self.index(index)]
