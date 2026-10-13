"""Any resumable iterator read as a Dew store."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch, Records
from dew.data.pipeline import Dataset

from .base import Read, Shard, ShardManifest
from .codec import Value, decode, encode
from .memory import _record


@dataclass(frozen=True, eq=False)
class Iterated(Dataset):
    """Records from `factory()`, a fresh iterator each call, read as one
    stream. Its position is what `state(iterator)` returns and
    `load_state(iterator, saved)` restores, as WebDataset, torchdata,
    Mosaic Streaming and Ray iterators each keep their own; without them, a
    resumed reader replays the stream from the start, skipping the records
    already read. `name` declares its content version in a run's plan and its logs."""

    factory: Callable[[], Iterator[Batch]]
    name: str
    state: Callable[[Iterator[Batch]], Value] | None = None
    load_state: Callable[[Iterator[Batch], Value], None] | None = None

    def __post_init__(self) -> None:
        if (self.state is None) != (self.load_state is None):
            raise ValueError("from_iter takes state= and load_state= together")

    def describe(self) -> str:
        return f"iter:{self.name}"

    def fingerprint(self) -> Mapping[str, object]:
        return {"store": "iterator", "identity_required": True, "identity": self.name}

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

    def __next__(self) -> Batch | bytes:
        row = None
        while row is None:
            row = _record(next(self._rows))
            self._read += 1
        return row

    def get_state(self) -> dict[str, int | str]:
        if self._source.state is None:
            return {"read": self._read}
        # Through the artifact codec, so arrays, NumPy scalars and bytes reach load_state as they were.
        saved = encode(self._source.state(self._rows))
        return {"read": self._read, "state": base64.b64encode(saved).decode()}

    def set_state(self, state: Mapping[str, object]) -> None:
        read, saved = state["read"], state.get("state")
        if not isinstance(read, int) or read < 0:
            raise ValueError(f"an iterator's position is the count of records it read, not {read!r}")
        self._rows, self._read = self._source.factory(), 0
        if self._source.load_state is not None:
            if not isinstance(saved, str):
                raise ValueError("this iterator resumes through load_state, and its position holds no state")
            self._source.load_state(self._rows, decode(base64.b64decode(saved)))
            self._read = read
            return
        while self._read < read:
            next(self)


def from_iter(factory: Callable[[], Iterator[Batch]], *, name: str,
              state: Callable[[Iterator[Batch]], Value] | None = None,
              load_state: Callable[[Iterator[Batch], Value], None] | None = None) -> Iterated:
    """The records `factory()` iterates as a Dew dataset under `name`, resumed
    through the iterator's own `state` and `load_state` where it has them
    (`Iterated`)."""
    return Iterated(factory, name, state, load_state)
