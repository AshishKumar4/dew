"""Read-only NumPy indexes memory-mapped from a file on this host: a local index's own, or its copy in Dew's
mirror, pinned while the index is open."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import BinaryIO

import numpy as np

from dew.data.declared import Declared, mirrored
from dew.data.storage import Store, local


@dataclass(frozen=True)
class ArrayIndex:
    """The NumPy array at `url`, as its producer declared it where one did (`declared`: read from the file
    `Declared.leased` checks, a local one's own or its checked copy in Dew's mirror), else from a local
    file or its copy in the mirror."""

    url: str
    storage_options: Mapping[str, object] | None = field(default=None, repr=False)
    declared: Declared | None = field(default=None, repr=False)

    @cached_property
    def lease(self) -> tuple[str, BinaryIO | None]:
        """A path on this host to the index's bytes, and the open file that holds it there, none for an
        undeclared local file, read where it is."""
        if self.declared is not None:
            return self.declared.leased()
        store = Store(self.url, self.storage_options)
        fs, path = store.resolve()
        if local(fs):
            return path, None
        held = mirrored(store)
        return held.name, held

    @cached_property
    def values(self) -> np.ndarray:
        values = np.load(self.lease[0], mmap_mode="r", allow_pickle=False)
        if not isinstance(values, np.ndarray):
            raise ValueError(f"index {self.url} must contain one NumPy array")
        return values

    def close(self) -> None:
        self.__dict__.pop("values", None)
        lease = self.__dict__.pop("lease", None)
        if lease is not None and lease[1] is not None:
            lease[1].close()

    def __getstate__(self) -> Mapping[str, object]:
        return {"url": self.url, "storage_options": self.storage_options, "declared": self.declared}
