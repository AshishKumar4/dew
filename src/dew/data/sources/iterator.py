"""Any resumable iterator read as a Dew store."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import grain.python as grain
import numpy as np

from dew.batch import Batch
from dew.data.dataset import Closeable, Records, Replayed, Stoppable
from dew.data.pipeline import Dataset

from .base import Read, Shard, ShardManifest, unnamed
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
    its content version in a run's plan and its logs; with none, a resume
    needs `resume="non-exact"` (`Manifest.exact`)."""

    factory: Callable[[], Iterator[Batch]]
    name: str | None = None
    state: Callable[[Iterator[Batch]], Value] | None = None
    load_state: Callable[[Iterator[Batch], Value], None] | None = None
    replay: bool = False
    place: int = field(default_factory=unnamed, repr=False)

    def __post_init__(self) -> None:
        if self.name == "":
            raise ValueError("an iterator's name declares its content version; give one, or none")
        if (self.state is None) != (self.load_state is None):
            raise ValueError("from_iter takes state= and load_state= together")
        if self.replay and self.state is not None:
            raise ValueError("from_iter resumes through state= and load_state= or by replay=True, not both")

    def label(self) -> str:
        return self.name or "iterator"

    def describe(self) -> str:
        return f"iter:{self.name}" if self.name is not None else f"iter:unnamed-{self.place}"

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
        source = self._source
        return _StreamIterator(source.factory, source.describe(), state=source.state,
                               load_state=source.load_state, replay=source.replay)


class _StreamIterator[Rows: Iterator[Batch]](grain.DatasetIterator):
    """The iterator `factory` makes, read as the stream `label` names, its position the iterator's
    own (`state`, restored by `load_state`), its replay (`Replayed`), or for neither the count of
    records read, from which it cannot resume: an iterator store's (`Iterated`) and a Hugging Face
    stream's (`hf._Streamed`)."""

    def __init__(self, factory: Callable[[], Rows], label: str, *,
                 state: Callable[[Rows], Value] | None = None,
                 load_state: Callable[[Rows, Value], None] | None = None, replay: bool = False):
        super().__init__()
        self._closed = True
        self._factory, self._label, self._state, self._load_state = factory, label, state, load_state
        self._rows: Rows | Replayed = Replayed(factory) if replay else factory()
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
        state = self._state
        if state is None:
            return {"read": self._read}
        # Through the artifact codec, so arrays, NumPy scalars and bytes reach load_state as they were.
        saved = encode(state(self._rows))
        return {"read": self._read, "state": base64.b64encode(saved).decode()}

    def set_state(self, state: Mapping[str, object]) -> None:
        if isinstance(self._rows, Replayed):
            self._rows.set_state(state)
            return
        if self._load_state is None:
            raise ValueError(
                f"{self._label} keeps no position and declares no replay, so it cannot "
                f"resume; pass replay=True if it reads the same records each time it opens, or "
                f"state= and load_state=")
        read, saved = state.get("read"), state.get("state")
        if not isinstance(read, int) or read < 0:
            raise ValueError(f"an iterator's position is the count of records it read, not {read!r}")
        if not isinstance(saved, str):
            raise ValueError("this iterator resumes through load_state, and its position holds no state")
        position = decode(base64.b64decode(saved))
        self.close()
        rows = self._factory()
        self._rows, self._closed = rows, False
        self._load_state(rows, position)
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
            self._closed = True
            super().close()


def from_iter(factory: Callable[[], Iterator[Batch]], *, name: str | None = None,
              state: Callable[[Iterator[Batch]], Value] | None = None,
              load_state: Callable[[Iterator[Batch], Value], None] | None = None,
              replay: bool = False) -> Iterated:
    """The records `factory()` iterates as a Dew dataset, under `name` where given, resumed
    through the iterator's own `state` and `load_state` where it has them,
    or by replay where it declares `replay` (`Iterated`)."""
    return Iterated(factory, name, state, load_state, replay)
