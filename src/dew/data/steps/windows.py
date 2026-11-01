"""Overlapping causal windows, with the boundary target retained.

Grain's concat-and-split packer has disjoint windows. Causal LM windows
share their boundary token so every target is predicted exactly once.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import ClassVar, Protocol

import grain.python as grain
import numpy as np

from dew._records import integer, integers, record
from dew.batch import Batch
from dew.data.pipeline import CONSUMED, Windows
from dew.data.sources.base import Read, Rows, ShardManifest, View, _indexed_manifest
from dew.data.sources.tokens import TokenFiles
from dew.data.steps.generic import aligned


class TokenSpan(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, span: slice) -> np.ndarray: ...


@dataclasses.dataclass(frozen=True)
class IndexedWindows:
    """Complete causal windows of aligned fields backed by contiguous token spans."""

    tokens: Mapping[str, TokenSpan | np.ndarray]
    length: int
    stride: int
    dtype: str | None = None

    @classmethod
    def from_records(cls, rows: Sequence[Batch], windows: Windows) -> IndexedWindows:
        held = [aligned(row, windows.fields) for row in rows]
        tokens = {
            name: np.concatenate([row[name] for row in held]) if held else np.zeros(0, np.int32)
            for name in windows.fields
        }
        return cls(tokens, windows.seq_len, windows.stride or windows.seq_len)

    @property
    def size(self) -> int:
        return len(next(iter(self.tokens.values())))

    def __len__(self) -> int:
        return max(0, (self.size - self.length - 1) // self.stride + 1)

    def __getitem__(self, index: int) -> Batch:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.stride
        return {name: np.asarray(tokens[start : start + self.length + 1], dtype=self.dtype)
                for name, tokens in self.tokens.items()}


@dataclasses.dataclass(frozen=True)
class WindowSource(View[TokenFiles]):
    """Causal windows of `length` tokens, `stride` apart, over a token store's tokens."""

    rows: ClassVar[Rows] = "derived"
    source: TokenFiles
    tokens: TokenSpan
    length: int
    stride: int

    def describe(self) -> str:
        return f"{self.source.describe()} causal windows={self.length},stride={self.stride}"

    @property
    def max_length(self) -> int:
        """The inputs each window holds, which a token budget divides into rows
        (`dew.data.pipeline.TokenGeometry`)."""
        return self.length

    def fingerprint(self) -> Mapping[str, object]:
        return {**self.source.fingerprint(), "window": self.length, "stride": self.stride}

    def can_provision(self, read: Read) -> bool:
        return self.source.can_provision(read)

    def provision(self) -> WindowSource:
        """Provision the token store, whose length index these windows need
        to read it by span."""
        self.source.provision()
        return self

    def records(self, read: Read) -> IndexedWindows:
        if read == "memory":
            self.source.records(read)
        return IndexedWindows({"text": self.tokens}, self.length, self.stride, "int32")

    def metadata(self) -> Mapping[str, object]:
        count = len(self.records("indexed"))
        return {**super().metadata(), "rows": count, "valid_targets": count * self.length}

    def lengths(self) -> np.ndarray | None:
        return None

    def manifest(self) -> ShardManifest:
        return _indexed_manifest(self, self.describe(), 1024)


class _WindowIterator(grain.DatasetIterator[Batch]):
    def __init__(self, parent: grain.DatasetIterator[Batch], windows: Windows):
        super().__init__(parent)
        self._read, self._windows = parent, windows
        self._held: Batch = {name: np.zeros(0, np.int32) for name in windows.fields}
        self._skip = 0
        # The leading held tokens whose consumed examples a window already took (`CONSUMED`).
        self._credited = 0

    def __next__(self) -> Batch:
        while True:
            windows = IndexedWindows(
                {name: value[self._skip :] for name, value in self._held.items()},
                self._windows.seq_len,
                self._windows.stride or self._windows.seq_len,
            )
            if len(windows):
                break
            row = aligned(next(self._read), self._windows.fields)
            self._held = {
                name: np.concatenate([self._held[name], row[name]])
                for name in self._windows.fields
            }
        emitted = windows[0]
        # The window holds its view's first `length + 1` tokens, its inputs and their targets.
        end, kept = self._skip + windows.length + 1, self._skip + windows.stride
        if CONSUMED in self._held:
            # Each token's count goes to the first window that holds it, the tokens a stride skips going
            # to the window after them; a token past every window the stream makes is consumed by none.
            owned = self._held[CONSUMED][self._credited : end]
            emitted = {**emitted, CONSUMED: int(np.sum(owned))}
            self._credited = max(0, end - kept)
        self._held = {name: value[kept:] for name, value in self._held.items()}
        self._skip = max(0, windows.stride - windows.size)
        return emitted

    def get_state(self) -> Mapping[str, object]:
        return {
            "parent": self._read.get_state(),
            "skip": self._skip,
            "credited": self._credited,
            "held": {
                name: {"dtype": value.dtype.str, "tokens": value.tolist()}
                for name, value in self._held.items()
            },
        }

    def set_state(self, state: Mapping[str, object]) -> None:
        self._read.set_state(record(state["parent"], "window parent"))
        held = record(state["held"], "window carry")
        self._held = {
            name: np.asarray(
                integers(record(tokens, name)["tokens"], name), dtype=str(record(tokens, name)["dtype"])
            )
            for name, tokens in held.items()
        }
        skip = state["skip"]
        if not isinstance(skip, int) or skip < 0:
            raise ValueError("saved causal window skip is invalid")
        self._skip = skip
        # A position saved before windows took counts credits none of its carry.
        self._credited = integer(state.get("credited", 0), "saved causal window credit")

    def close(self) -> None:
        self._read.close()


class CausalWindows(grain.IterDataset[Batch]):
    def __init__(self, parent: grain.IterDataset[Batch], windows: Windows):
        super().__init__(parent)
        self._records, self._windows = parent, windows

    def __iter__(self) -> grain.DatasetIterator[Batch]:
        return _WindowIterator(iter(self._records), self._windows)
