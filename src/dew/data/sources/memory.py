"""Sources whose records are already resident in the caller's memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np

from dew.data.dataset import Batch, Records
from dew.data.pipeline import Dataset

from .base import Read, ShardManifest, _indexed_manifest
from .codec import fingerprint


@runtime_checkable
class DataFrame(Protocol):
    """A table that hands itself over through the dataframe interchange protocol: a pandas or
    polars DataFrame, or an Arrow table."""

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> object: ...


class SparkFrame(Protocol):
    def toArrow(self) -> DataFrame: ...


def _record(value: object) -> Batch | bytes | None:
    if value is None or isinstance(value, bytes):
        return value
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return value
    raise TypeError(f"source feature record must have named fields or bytes, got {type(value).__name__}")


def _table_records(table: DataFrame) -> list[Batch]:
    """A table's rows as records of its columns, through Arrow's interchange reader."""
    from pyarrow.interchange import from_dataframe

    rows = from_dataframe(table, allow_copy=True).to_pylist()
    for row in rows[:1]:
        _record(row)
    return rows


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
    """Records, column arrays, or a DataFrame or Arrow table (`DataFrame`), with immutable
    positional IDs. A table's rows become records of its columns, its index none of them."""

    values: Records | Sequence[Batch] | Mapping[str, Sequence] | DataFrame
    identity: str | None = None
    shard_rows: int = 1024

    def __post_init__(self) -> None:
        if self.shard_rows < 1:
            raise ValueError("memory shard_rows must be positive")

    @classmethod
    def from_spark(cls, frame: SparkFrame) -> Memory:
        """Gather a small pyspark 4.0+ frame through its Arrow handoff."""
        return cls(frame.toArrow())

    def label(self) -> str:
        return self.identity or "memory"

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
        return self._held

    @cached_property
    def _held(self) -> Records:
        """Its records: the rows given, the rows of columns or a table's rows, read once."""
        if isinstance(self.values, DataFrame):
            return _table_records(self.values)
        if isinstance(self.values, Mapping):
            return _Columns(self.values)
        return self.values

    def manifest(self) -> ShardManifest:
        return _indexed_manifest(self, "memory", "memory", self.shard_rows)

    def metadata(self) -> Mapping[str, object]:
        return {"rows": len(self.records("indexed")), "count_rules": self._count_rules}

    @cached_property
    def _count_rules(self) -> list[list[int | None]]:
        """The rules its rows' targets were counted by, read off the rows it holds."""
        from dew.data.steps.text import count_rules

        records = self.records("indexed")
        return count_rules(records[index] for index in range(len(records)))

    def lengths(self) -> np.ndarray | None:
        if isinstance(self.values, Mapping) and "length" in self.values:
            return np.asarray(self.values["length"], dtype=np.int64)
        return None
