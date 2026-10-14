"""Any resumable iterator read as a Dew store."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch, Closeable, Records, Replayed, Stoppable
from dew.data.pipeline import Dataset

from .base import Read, Shard, ShardManifest
from .codec import Value, decode, encode
from .memory import _record


@dataclass(frozen=True, eq=False)
class Iterated(Dataset):
    """Records from `factory()`, a fresh iterator each call, read as one
    stream. Its position is what `state(iterator)` returns and
    `load_state(iterator, saved)` restores, as WebDataset, torchdata,
    Mosaic Streaming and Ray iterators each keep their own. `replay`
    declares instead that `factory()` reads the same records each time it
    opens, and a resume reads them again to its position (`Replayed`); a
    stream with neither trains, and its resume is refused. `name` declares
    its content version in a run's plan and its logs."""

    factory: Callable[[], Iterator[Batch]]
    name: str
    state: Callable[[Iterator[Batch]], Value] | None = None
    load_state: Callable[[Iterator[Batch], Value], None] | None = None
    replay: bool = False

    def __post_init__(self) -> None:
        if (self.state is None) != (self.load_state is None):
            raise ValueError("from_iter takes state= and load_state= together")
        if self.replay and self.state is not None:
            raise ValueError("from_iter resumes through state= and load_state= or by replay=True, not both")

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
    state, its replay (`Replayed`), or for neither the count of records read,
    from which it cannot resume."""

    def __init__(self, source: Iterated):
        super().__init__()
        self._closed = True
        self._source = source
        self._rows = Replayed(source.factory) if source.replay else source.factory()
        # The records read, which the position of a stream that does not replay carries.
        self._read = 0
        self._closed = False

    def __next__(self) -> Batch | bytes:
        row = None
        while row is None:
            row = _record(next(self._rows))
            self._read += 1
        return row

    def get_state(self) -> Mapping[str, object]:
        if isinstance(self._rows, Replayed):
            return self._rows.get_state()
        state = self._source.state
        if state is None:
            return {"read": self._read}
        # Through the artifact codec, so arrays, NumPy scalars and bytes reach load_state as they were.
        saved = encode(state(self._rows))
        return {"read": self._read, "state": base64.b64encode(saved).decode()}

    def set_state(self, state: Mapping[str, object]) -> None:
        if isinstance(self._rows, Replayed):
            self._rows.set_state(state)
            return
        if self._source.load_state is None:
            raise ValueError(
                f"{self._source.describe()} keeps no position and declares no replay, so it cannot "
                f"resume; pass replay=True if it reads the same records each time it opens, or "
                f"state= and load_state=")
        read, saved = state["read"], state.get("state")
        if not isinstance(read, int) or read < 0:
            raise ValueError(f"an iterator's position is the count of records it read, not {read!r}")
        if not isinstance(saved, str):
            raise ValueError("this iterator resumes through load_state, and its position holds no state")
        position = decode(base64.b64decode(saved))
        self.close()
        self._rows = self._source.factory()
        self._closed = False
        self._source.load_state(self._rows, position)
        self._read = read

    def request_stop(self) -> None:
        if not self._closed and isinstance(self._rows, Stoppable):
            self._rows.request_stop()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if isinstance(self._rows, Closeable):
                self._rows.close()
        finally:
            super().close()


def from_iter(factory: Callable[[], Iterator[Batch]], *, name: str,
              state: Callable[[Iterator[Batch]], Value] | None = None,
              load_state: Callable[[Iterator[Batch], Value], None] | None = None,
              replay: bool = False) -> Iterated:
    """The records `factory()` iterates as a Dew dataset under `name`, resumed
    through the iterator's own `state` and `load_state` where it has them,
    or by replay where it declares `replay` (`Iterated`)."""
    return Iterated(factory, name, state, load_state, replay)
