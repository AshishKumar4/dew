"""Sources whose records are already resident in the caller's memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch, Records

from .base import Read, Shard, ShardManifest
from .codec import fingerprint


@dataclass(frozen=True)
class _Columns:
    values: Mapping[str, Sequence]

    def __post_init__(self) -> None:
        if len({len(column) for column in self.values.values()}) > 1:
            raise ValueError("memory columns have different row counts")

    def __len__(self) -> int:
        return len(next(iter(self.values.values()))) if self.values else 0

    def __getitem__(self, index: int) -> Batch:
        return {name: values[index] for name, values in self.values.items()}


@dataclass(frozen=True)
class Memory:
    """Records, column arrays or an Arrow table, with immutable positional IDs."""

    values: Records | Sequence[Batch] | Mapping[str, Sequence]
    identity: str | None = None
    shard_rows: int = 1024

    def __post_init__(self) -> None:
        if self.shard_rows < 1:
            raise ValueError("memory shard_rows must be positive")

    def describe(self) -> str:
        return "memory:" + (self.identity or self.snapshot()["sha256"])

    def snapshot(self) -> dict[str, object]:
        rows = self.records("indexed")
        return {"store": "memory", "identity": self.identity, "rows": len(rows),
                "sha256": fingerprint(rows[index] for index in range(len(rows)))}

    def supports(self, read: Read) -> bool:
        return True

    def records(self, read: Read) -> Records:
        if isinstance(self.values, Mapping):
            return _Columns(self.values)
        return self.values

    def manifest(self) -> ShardManifest:
        count = len(self.records("indexed"))
        return ShardManifest(tuple(Shard(f"memory:{start}", "memory", start,
                                         min(start + self.shard_rows, count),
                                         min(self.shard_rows, count - start))
                                   for start in range(0, count, self.shard_rows)), self._read_shard)

    def _read_shard(self, shard: Shard) -> grain.IterDataset:
        return grain.MapDataset.source(self.records("indexed"))[shard.start:shard.stop].to_iter_dataset()

    def metadata(self) -> Mapping[str, object]:
        return {"rows": len(self.records("indexed"))}

    def lengths(self) -> np.ndarray | None:
        if isinstance(self.values, Mapping) and "length" in self.values:
            return np.asarray(self.values["length"], dtype=np.int64)
        return None
