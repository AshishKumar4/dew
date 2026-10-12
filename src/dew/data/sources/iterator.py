"""Any resumable iterator read as a Dew store."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch, Records
from dew.data.pipeline import Dataset

from .base import Read, Shard, ShardManifest
from .memory import _record


@dataclass(frozen=True, eq=False)
class Iterated(Dataset):
    """Records from `factory()`, a fresh iterator each call, read as one
    stream. Its position is what `state(iterator)` returns and
    `load_state(iterator, saved)` restores, as WebDataset, torchdata,
    Mosaic Streaming and Ray iterators each keep their own; without them, a
    resumed reader replays the stream from the start, skipping the records
    already read. `name` is its identity in a run's plan and its logs."""

    factory: Callable[[], Iterator[Batch]]
    name: str
    state: Callable[[Iterator[Batch]], object] | None = None
    load_state: Callable[[Iterator[Batch], object], None] | None = None

    def __post_init__(self) -> None:
        if (self.state is None) != (self.load_state is None):
            raise ValueError("from_iter takes state= and load_state= together")

    def describe(self) -> str:
        return f"iter:{self.name}"

    def snapshot(self) -> Mapping[str, object]:
        return {"store": "iterator", "name": self.name}

    def supports(self, read: Read) -> bool:
        return read == "streamed"

    def records(self, read: Read) -> Records:
        raise ValueError(f"{self.describe()} is an iterator, read only as a stream")

    def manifest(self) -> ShardManifest:
        return ShardManifest((Shard(self.describe(), self.describe()),), self._read_shard)

    def _read_shard(self, shard: Shard) -> grain.IterDataset:
        return _Stream(self)

    def metadata(self) -> Mapping[str, object]:
        return {}

    def lengths(self) -> np.ndarray | None:
        return None


class _Stream(grain.IterDataset):
    def __init__(self, source: Iterated):
        super().__init__()
        self._source = source

    def __iter__(self) -> grain.DatasetIterator:
        return _StreamIterator(self._source)


class _StreamIterator(grain.DatasetIterator):
    """The iterator `source.factory` makes, its position the iterator's own
    state, or the count of records read where it keeps none."""

    def __init__(self, source: Iterated):
        super().__init__()
        self._source = source
        self._rows = source.factory()
        self._read = 0

    def __next__(self) -> Batch:
        row = _record(next(self._rows))
        self._read += 1
        return row

    def get_state(self) -> dict:
        if self._source.state is None:
            return {"read": self._read}
        return {"read": self._read, "state": self._source.state(self._rows)}

    def set_state(self, state: dict) -> None:
        self._rows, self._read = self._source.factory(), 0
        if self._source.load_state is not None:
            self._source.load_state(self._rows, state["state"])
            self._read = int(state["read"])
            return
        for _ in range(int(state["read"])):
            next(self)


def from_iter(factory: Callable[[], Iterator[Batch]], *, name: str,
              state: Callable[[Iterator[Batch]], object] | None = None,
              load_state: Callable[[Iterator[Batch], object], None] | None = None) -> Iterated:
    """The records `factory()` iterates as a Dew dataset under `name`, resumed
    through the iterator's own `state` and `load_state` where it has them
    (`Iterated`)."""
    return Iterated(factory, name, state, load_state)
