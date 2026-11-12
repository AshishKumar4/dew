"""Grain's window shuffle, whose iterator `dew.data.ordering` extends: its permutation of a window
(`_reshuffle_window`) and the iterator state it keeps between windows."""

from collections.abc import Mapping, Sequence

from grain.python import DatasetIterator

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
