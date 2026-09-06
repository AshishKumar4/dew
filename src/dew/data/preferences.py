"""Preference pairs for DPO: chosen and rejected rows with their masks.

A `PreferencePairs` source reads a parquet file or in-memory JSON rows of
`chosen` and `rejected` token-id lists with `chosen_mask` and
`rejected_mask` marking the completion tokens, and stacks each batch in
TRL's layout: chosen rows over rejected rows, padded to the longest row,
with `completion_mask` alongside the ids. A mask must run as long as its
ids; anything else fails naming the row.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence

import numpy as np

from .dataset import (Batch, Dataset, DatasetSpec, Loading, local_batch, train_stream,
                      validation_pass)
from dew.registry import datasets

from .tokens import bounded

IDS_KEY = "input_ids"
"""Batch key holding the `[B, 2, S]` pair ids, chosen at index 0."""

MASK_KEY = "completion_mask"
"""Batch key holding the `[B, 2, S]` completion marks, 1 on completion tokens."""

FIELDS = ("chosen", "rejected", "chosen_mask", "rejected_mask")


def _ids(values: object, name: str, where: str) -> list[int]:
    """One id list: integers only, non-empty."""
    if not isinstance(values, list) or not values:
        raise ValueError(f"{where}: {name} is a non-empty id list, got {values!r}")
    ids: list[int] = []
    for token in values:
        if not isinstance(token, int):
            raise ValueError(f"{where}: {name} holds token ids, got {token!r}")
        ids.append(token)
    return ids


def _mask(values: object, name: str, length: int, where: str) -> list[int]:
    """One 0/1 mask as long as its ids."""
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(
            f"{where}: {name} runs as long as its ids ({length}), "
            f"got {values!r}")
    mask: list[int] = []
    for mark in values:
        if mark not in (0, 1):
            raise ValueError(f"{where}: {name} marks completion tokens 0/1, got {mark!r}")
        mask.append(int(mark))
    return mask


class PreferenceSource:
    """Random access over validated pair rows, stacked on read.

    Rows come back as the TRL stack: row i's chosen half at i, its rejected
    half at i + n, padded to the longest row of the two halves with `pad_id`
    and 0. The repr names the origin file for grain's checkpoint matching, or
    the record count for in-memory rows.
    """

    def __init__(self, rows: Sequence[Mapping[str, object]], origin: str, pad_id: int):
        normalized: list[tuple[list[int], list[int], list[int], list[int]]] = []
        for index, row in enumerate(rows):
            where = f"{origin} row {index}"
            if not isinstance(row, Mapping):
                raise ValueError(f"{where}: a row is an object, got {row!r}")
            unknown = sorted(key for key in row if key not in FIELDS)
            if unknown:
                raise ValueError(
                    f"{where}: unknown fields {unknown}; the fields are {list(FIELDS)}")
            if row.get("chosen") is None or row.get("rejected") is None:
                raise ValueError(f"{where}: a pair needs both chosen and rejected ids")
            chosen = _ids(row["chosen"], "chosen", where)
            rejected = _ids(row["rejected"], "rejected", where)
            normalized.append((
                chosen, rejected,
                _mask(row.get("chosen_mask", [1] * len(chosen)), "chosen_mask",
                      len(chosen), where),
                _mask(row.get("rejected_mask", [1] * len(rejected)), "rejected_mask",
                      len(rejected), where)))
        if not normalized:
            raise ValueError(f"{origin} holds no pairs")
        self._rows = normalized
        self._origin = origin
        self._pad_id = pad_id

    @classmethod
    def from_parquet(cls, path: str, pad_id: int) -> PreferenceSource:
        """The file's rows: `chosen` and `rejected` are required, the masks
        default to all-completion when absent."""
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ImportError(
                "reading preference parquet needs pyarrow: pip install pyarrow") from exc
        names = [field.name for field in parquet.read_schema(path)]
        for column in ("chosen", "rejected"):
            if column not in names:
                raise ValueError(
                    f"{path}: the {column} column is required, the file has {names}")
        table = parquet.read_table(path, columns=[name for name in FIELDS if name in names])
        return cls(table.to_pylist(), path, pad_id)

    @classmethod
    def from_records(cls, records: tuple[str, ...], pad_id: int) -> PreferenceSource:
        """In-memory rows as JSON, for tests and small sweeps."""
        rows = []
        for index, record in enumerate(records):
            try:
                rows.append(json.loads(record))
            except json.JSONDecodeError as exc:
                raise ValueError(f"record {index} is not JSON: {exc}") from exc
        return cls(rows, f"{len(rows)} in-memory records", pad_id)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(origin={self._origin!r})"

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Batch:
        chosen, rejected, chosen_mask, rejected_mask = self._rows[index]
        width = max(len(chosen), len(rejected))
        ids = np.full((2, width), self._pad_id, np.int32)
        mask = np.zeros((2, width), np.int32)
        ids[0, :len(chosen)] = chosen
        ids[1, :len(rejected)] = rejected
        mask[0, :len(chosen_mask)] = chosen_mask
        mask[1, :len(rejected_mask)] = rejected_mask
        return {IDS_KEY: ids, MASK_KEY: mask}


@datasets("preference_pairs")
@dataclasses.dataclass(frozen=True)
class PreferencePairs(DatasetSpec):
    """Chosen and rejected completions in TRL's stacked layout.

    `path` is a parquet file; `records` is JSON rows for tests and small
    sweeps; exactly one of the two is set. Each batch holds `input_ids` with
    the B chosen rows over the B rejected rows and the matching
    `completion_mask`. `val_path` is a second parquet file scored as one
    pass; None trains without validation.
    """

    path: str | None = None
    records: tuple[str, ...] = ()
    val_path: str | None = None
    pad_id: int = 0
    val_batches: int | None = 4
    seed: int = 0
    loading: Loading = Loading()

    def load(self, *, batch: int) -> Dataset:
        if (self.path is None) == (not self.records):
            raise ValueError(
                "PreferencePairs reads one source: --data.path names a parquet file, "
                "or records holds JSON rows")
        if self.path is not None:
            source = PreferenceSource.from_parquet(self.path, self.pad_id)
        else:
            source = PreferenceSource.from_records(self.records, self.pad_id)
        per_process = local_batch(batch)
        val = None
        if self.val_path is not None:
            val_source = PreferenceSource.from_parquet(self.val_path, self.pad_id)
            val = bounded(validation_pass(
                val_source, [], batch=per_process, seed=self.seed,
                loading=self.loading), self.val_batches)
        return Dataset(
            train=train_stream(source, [], batch=per_process, seed=self.seed,
                               loading=self.loading),
            val=val,
            records=len(source),
            batch=batch,
        )
