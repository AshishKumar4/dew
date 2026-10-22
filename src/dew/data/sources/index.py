"""Read-only NumPy indexes backed by local files or pinned bounded mirrors."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from dew.data.storage import Store, local


@dataclass(frozen=True)
class ArrayIndex:
    url: str
    storage_options: Mapping[str, object] | None = field(default=None, repr=False)
    metadata: Mapping[str, object] | None = field(default=None, repr=False)

    @cached_property
    def pin(self):
        store = Store(self.url, self.storage_options)
        fs, path = store.resolve()
        snapshot = None
        if self.metadata is not None:
            snapshot = {"size": self.metadata.get("bytes", self.metadata.get("size")), **self.metadata}
        if not local(fs) or self.metadata is not None:
            fs, path = store.cached().resolve()
        opened = fs.open(path, "rb", source_snapshot=snapshot)
        if not isinstance(opened.name, str):
            opened.close()
            raise ValueError(f"index {self.url} needs a mirror large enough to hold its NumPy file")
        return opened

    @cached_property
    def values(self) -> np.ndarray:
        values = np.load(self.pin.name, mmap_mode="r", allow_pickle=False)
        if not isinstance(values, np.ndarray):
            raise ValueError(f"index {self.url} must contain one NumPy array")
        return values

    def verify(self, reference: Mapping[str, object]) -> None:
        self.pin.seek(0, 2)
        size = self.pin.tell()
        self.pin.seek(0)
        with open(self.pin.name, "rb") as reader:
            checksum = hashlib.file_digest(reader, "sha256").hexdigest()
        self.pin.seek(0)
        if size != reference["bytes"] or checksum != reference["sha256"]:
            raise OSError(f"committed NumPy index is corrupt: {self.url}")

    def close(self) -> None:
        self.__dict__.pop("values", None)
        pin = self.__dict__.pop("pin", None)
        if pin is not None:
            pin.close()

    def __getstate__(self) -> Mapping[str, object]:
        return {"url": self.url, "storage_options": self.storage_options, "metadata": self.metadata}


@dataclass(frozen=True)
class Ranges:
    """Consecutive ranges of `counts` items each, as an indexed store's shards hold their records:
    which range holds an item (`locate`), where a range starts (`start`) and the items of them all.
    A range of no items holds none, so an item is never located in one."""

    counts: tuple[int, ...]

    @cached_property
    def ends(self) -> np.ndarray:
        return np.cumsum(np.asarray(self.counts, np.int64), dtype=np.int64)

    def __len__(self) -> int:
        return int(self.ends[-1]) if self.counts else 0

    def start(self, shard: int) -> int:
        return int(self.ends[shard - 1]) if shard else 0

    def locate(self, item: int) -> tuple[int, int]:
        """The range that holds `item`, and its place in that range."""
        if not 0 <= item < len(self):
            raise IndexError(item)
        shard = int(np.searchsorted(self.ends, item, side="right"))
        return shard, item - self.start(shard)
