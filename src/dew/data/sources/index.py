"""Read-only NumPy indexes backed by local files or pinned bounded mirrors."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from dew.data.storage import Store


@dataclass(frozen=True)
class ArrayIndex:
    url: str
    storage_options: Mapping[str, object] | None = field(default=None, repr=False)

    @cached_property
    def pin(self):
        store = Store(self.url, self.storage_options)
        fs, path = store.resolve()
        protocols = (fs.protocol,) if isinstance(fs.protocol, str) else fs.protocol
        if "file" not in protocols:
            fs, path = store.cached().resolve()
        opened = fs.open(path, "rb")
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
        return {"url": self.url, "storage_options": self.storage_options}
