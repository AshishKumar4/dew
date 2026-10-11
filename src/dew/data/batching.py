"""Fixed-token batches with a bounded number of shapes and bounded replay."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import grain.experimental as gx
import grain.python as grain
import numpy as np

from dew.data.dataset import Batch, DataPartition
from dew.data.pipeline import Geometry, TokenBudget, TokenField
from dew.records import integer, record
from dew.objectives.base import VALID_ROWS

TOKEN_COUNTS = "_dew_token_counts"


def bucket_lengths(budget: TokenBudget, *, row_multiple: int = 1) -> tuple[int, ...]:
    geometry = budget.geometry
    if geometry is None:
        raise ValueError("token batching needs a record step declaring TokenGeometry")
    if budget.lengths:
        lengths = budget.lengths
    else:
        divisors = {divisor for candidate in range(1, math.isqrt(budget.tokens) + 1)
                    if budget.tokens % candidate == 0 for divisor in (candidate, budget.tokens // candidate)}
        lengths = tuple(sorted(length for length in divisors
                                if min(256, geometry.max_length) <= length <= geometry.max_length
                                and (budget.tokens // length) % row_multiple == 0)[-6:])
    if not lengths or lengths[-1] != geometry.max_length:
        raise ValueError(f"token budget {budget.tokens} needs a largest bucket equal to "
                         f"max_length={geometry.max_length}, with rows divisible by {row_multiple}")
    if any((budget.tokens // length) % row_multiple for length in lengths):
        raise ValueError(f"token budget bucket rows must divide across {row_multiple} row shards")
    return lengths


def _length(row: Batch, geometry: Geometry) -> int:
    measured = np.asarray(row[geometry.length_field])
    if measured.ndim or measured.dtype.kind not in "iu":
        raise ValueError(f"token length field {geometry.length_field!r} must be an integer scalar")
    length = int(measured.item())
    if not 0 < length <= geometry.max_length:
        raise ValueError(f"token length {length} is outside 1..{geometry.max_length}")
    return length


def _field(value: np.ndarray, size: int, field: TokenField) -> np.ndarray:
    extent = size + field.extra
    if value.ndim < 1:
        raise ValueError(f"token field {field.name!r} must have a sequence axis")
    selected = value[-extent:] if field.side == "left" else value[:extent]
    missing = extent - len(selected)
    if missing:
        padding = ((missing, 0) if field.side == "left" else (0, missing), *((0, 0),) * (value.ndim - 1))
        selected = np.pad(selected, padding, constant_values=field.pad)
    return selected


def _batch(rows: Sequence[Batch], length: int, geometry: Geometry, size: int) -> Batch:
    fields = {field.name: field for field in geometry.token_fields}
    cropped = [{name: _field(np.asarray(value), length, fields[name]) if name in fields else value
                for name, value in row.items() if not name.startswith("_dew_")} for row in rows]
    batched = gx.batch_and_pad(cropped, batch_size=size)
    if not isinstance(batched, Mapping):
        raise TypeError("Grain token batching must preserve a record mapping")
    counts = np.zeros(size, np.int32)
    counts[:len(rows)] = [_length(row, geometry) for row in rows]
    mask = {} if len(rows) == size else {VALID_ROWS: np.arange(size) < len(rows)}
    return {**batched, TOKEN_COUNTS: counts, **mask}


def _feasible(sizes: tuple[int, ...], count: int) -> tuple[bool, ...]:
    possible = [True] + [False] * count
    for remaining in range(1, count + 1):
        possible[remaining] = any(remaining >= size and possible[remaining - size] for size in sizes)
    return tuple(possible)


class _TokenBatches(grain.DatasetIterator[Batch]):
    def __init__(self, parent: grain.DatasetIterator[Batch], budget: TokenBudget,
                 partition: DataPartition, seed: int):
        super().__init__(parent)
        if partition.count > 1:
            raise ValueError("token-budget batching needs one share: independent per-share queues can choose different shapes")
        if budget.geometry is None:
            raise ValueError("token budget has no declared record geometry")
        self._read, self._budget, self._geometry = parent, budget, budget.geometry
        self._lengths = bucket_lengths(budget, row_multiple=partition.row_multiple)
        self._sizes = tuple(partition.rows(budget.tokens // length) for length in self._lengths)
        self._window_size = math.ceil(max(self._sizes) * budget.window / min(self._sizes)) * min(self._sizes)
        self._possible = tuple(_feasible(self._sizes[index:], self._window_size)
                               for index in range(len(self._sizes)))
        self._seed, self._window, self._emitted = seed, 0, 0
        self._before = parent.get_state()
        self._batches: list[Batch] = []
        self._shapes: set[tuple[int, int]] = set()

    def _fill(self) -> None:
        self._before = self._read.get_state()
        rows = []
        for _ in range(self._window_size):
            try:
                rows.append(next(self._read))
            except StopIteration:
                break
        rows.sort(key=lambda row: _length(row, self._geometry), reverse=True)
        formed: list[Batch] = []
        at = 0
        while at < len(rows):
            first = next(index for index, length in enumerate(self._lengths)
                         if length >= _length(rows[at], self._geometry))
            remaining = len(rows) - at
            choice = next((index for index in range(first, len(self._sizes)) if self._sizes[index] <= remaining
                           and self._possible[first][remaining - self._sizes[index]]), None)
            if choice is None:
                if self._budget.drop_remainder:
                    break
                choice = first
            size, length = self._sizes[choice], self._lengths[choice]
            formed.append(_batch(rows[at:at + size], length, self._geometry, size))
            self._shapes.add((size, length))
            at += size
        order = np.random.default_rng([self._seed, self._window]).permutation(len(formed))
        self._batches = [formed[index] for index in order]
        self._emitted = 0

    def __next__(self) -> Batch:
        if self._emitted == len(self._batches):
            if self._batches:
                self._window += 1
            self._fill()
        if not self._batches:
            raise StopIteration
        batch = self._batches[self._emitted]
        self._emitted += 1
        return batch

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self._before, "window": self._window, "emitted": self._emitted,
                "lengths": self._lengths, "shapes": sorted(self._shapes)}

    def set_state(self, state: Mapping[str, object]) -> None:
        if tuple(state["lengths"]) != self._lengths:
            raise ValueError("saved token-budget bucket lengths changed")
        self._read.set_state(record(state["parent"], "token budget parent"))
        self._window = integer(state["window"], "token budget window")
        self._fill()
        self._emitted = integer(state["emitted"], "token budget emitted")
        if not 0 <= self._emitted <= len(self._batches):
            raise ValueError("saved token-budget batch position exceeds its window")

    def close(self) -> None:
        self._read.close()


class TokenBatches(grain.IterDataset[Batch]):
    def __init__(self, parent: grain.IterDataset[Batch], budget: TokenBudget,
                 partition: DataPartition, seed: int):
        super().__init__(parent)
        self._records, self._budget, self._partition, self._seed = parent, budget, partition, seed

    def __iter__(self) -> grain.DatasetIterator[Batch]:
        return _TokenBatches(iter(self._records), self._budget, self._partition, self._seed)
