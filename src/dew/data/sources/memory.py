"""Sources whose records are already resident in the caller's memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch, Records
from dew.data.pipeline import Dataset

from .base import Read, Shard, ShardManifest
from .codec import fingerprint


class ArrowTable(Protocol):
    def to_pylist(self) -> list[Batch]: ...


class SparkFrame(Protocol):
    def toArrow(self) -> ArrowTable: ...


def _record(value: object) -> Batch | bytes | None:
    if value is None or isinstance(value, bytes):
        return value
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return value
    raise TypeError(f"source feature record must have named fields or bytes, got {type(value).__name__}")


@dataclass(frozen=True)
class _Columns:
    values: Mapping[str, Sequence]

    def __post_init__(self) -> None:
        lengths = {name: len(column) for name, column in self.values.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"memory columns have different row counts: {lengths}")

    def __len__(self) -> int:
        return len(next(iter(self.values.values()))) if self.values else 0

    def __getitem__(self, index: int) -> Batch:
        return {name: values[index] for name, values in self.values.items()}


@dataclass(frozen=True)
class Memory(Dataset):
    """Records, column arrays or an Arrow table, with immutable positional IDs."""

    values: Records | Sequence[Batch] | Mapping[str, Sequence]
    identity: str | None = None
    shard_rows: int = 1024

    def __post_init__(self) -> None:
        if self.shard_rows < 1:
            raise ValueError("memory shard_rows must be positive")

    @classmethod
    def from_arrow(cls, table: ArrowTable) -> Memory:
        return cls(table.to_pylist())

    @classmethod
    def from_spark(cls, frame: SparkFrame) -> Memory:
        """Gather a small pyspark 4.0+ frame through its Arrow handoff."""
        return cls.from_arrow(frame.toArrow())

    def describe(self) -> str:
        return f"memory:{self.identity or self.fingerprint()['sha256']}"

    def fingerprint(self) -> Mapping[str, object]:
        rows = self.records("indexed")
        return {
            "store": "memory",
            "identity": self.identity,
            "rows": len(rows),
            "shard_rows": self.shard_rows,
            "sha256": fingerprint(rows[index] for index in range(len(rows))),
        }

    def supports(self, read: Read) -> bool:
        return True

    def records(self, read: Read) -> Records:
        if isinstance(self.values, Mapping):
            return _Columns(self.values)
        return self.values

    def manifest(self) -> ShardManifest:
        count = len(self.records("indexed"))
        return ShardManifest(
            tuple(
                Shard(
                    f"memory:{start}",
                    "memory",
                    start,
                    min(start + self.shard_rows, count),
                    min(self.shard_rows, count - start),
                )
                for start in range(0, count, self.shard_rows)
            ),
            self._read_shard,
        )

    def _read_shard(self, shard: Shard) -> grain.IterDataset:
        return grain.MapDataset.source(self.records("indexed"))[shard.start : shard.stop].to_iter_dataset()

    def metadata(self) -> Mapping[str, object]:
        return {"rows": len(self.records("indexed"))}

    def lengths(self) -> np.ndarray | None:
        if isinstance(self.values, Mapping) and "length" in self.values:
            return np.asarray(self.values["length"], dtype=np.int64)
        return None
