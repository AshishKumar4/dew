"""Grain's shuffles: its window shuffle, whose iterator `dew.data.ordering` extends (its permutation of a
window, `_reshuffle_window`, and the iterator state it keeps between windows), and its shuffle of a whole
dataset, whose place of an index `dew.data.shards._Shuffled` answers once a process."""

from collections.abc import Mapping, Sequence

from grain.python import DatasetIterator, MapDataset

def _reshuffle_window[T](window: Sequence[T], seed: int) -> list[T]: ...

class _WindowShuffleDatasetIterator[T](DatasetIterator[T]):
    _parent: DatasetIterator[T]
    _window_size: int
    _global_seed: int
    _window_index: int
    _window: list[T]
    _parent_exhausted: bool
    _parent_window_start_iter_state: Mapping[str, object]
    def __init__(self, parent: DatasetIterator[T], window_size: int, seed: int) -> None: ...
    def _fill_and_shuffle_window(self) -> None: ...

class ShuffleMapDataset[T](MapDataset[T]):
    _parent: MapDataset[T]
    _seed: int
    def __init__(self, parent: MapDataset[T], *, seed: int | None = ...) -> None: ...
    def _shuffled_index(self, index: int) -> int: ...
