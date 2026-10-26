"""Sources whose records are already resident in the caller's memory."""

from __future__ import annotations

import sys
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np

from dew.batch import Batch
from dew.data.compilation import once
from dew.data.dataset import Records
from dew.data.pipeline import Dataset

from .base import Read, ShardManifest, _indexed_manifest
from .codec import fingerprint


class _Interchanged(Protocol):
    """A table as the dataframe interchange protocol hands it over."""

    def num_columns(self) -> int: ...


@runtime_checkable
class DataFrame(Protocol):
    """A table that hands itself over through the dataframe interchange protocol: a pandas or
    polars DataFrame, or an Arrow table."""

    def __dataframe__(self) -> _Interchanged: ...


class SparkFrame(Protocol):
    def toArrow(self) -> DataFrame: ...


def _record(value: object) -> Batch | bytes | None:
    if value is None or isinstance(value, bytes):
        return value
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return value
    raise TypeError(f"source feature record must have named fields or bytes, got {type(value).__name__}")


@runtime_checkable
class _ArrowStream(Protocol):
    """A table that hands itself to Arrow as a stream of record batches: an Arrow or polars table."""

    def __arrow_c_stream__(self, requested_schema: None = None, /) -> Hashable: ...


def _table_records(table: DataFrame) -> list[Batch]:
    """A table's rows as records of its columns, through Arrow, which keeps lists, bytes, nested
    records and nulls as `Memory(records)` holds them: a pandas DataFrame by its own converter, its
    index none of them; a table that streams itself to Arrow as it does; any other through the
    interchange protocol."""
    import pyarrow as pa

    pandas = sys.modules.get("pandas")
    if pandas is not None and isinstance(table, pandas.DataFrame):
        arrow = pa.Table.from_pandas(table, preserve_index=False)
    elif isinstance(table, _ArrowStream):
        arrow = pa.table(table)
    else:
        from pyarrow.interchange import from_dataframe

        arrow = from_dataframe(table, allow_copy=True)
    rows = arrow.to_pylist()
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
        """Its content, hashed once a compilation (`dew.data.plan.once`), which a plan, a cache key and
        a description each read, its records the caller's to change between two."""
        return once(self, self._fingerprinted)

    def _fingerprinted(self) -> Mapping[str, object]:
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
        return _indexed_manifest(self, "memory", self.shard_rows)

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
