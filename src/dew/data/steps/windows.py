"""Overlapping causal windows, with the boundary target retained.

Grain's concat-and-split packer has disjoint windows. Causal LM windows
share their boundary token so every target is predicted exactly once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.pipeline import Windows
from dew.records import integers, record


class WindowRecords:
    def __init__(self, rows: Sequence[Batch], windows: Windows):
        self.windows = windows
        self.tokens = {
            name: np.concatenate([np.asarray(row[name]) for row in rows]) if rows else np.zeros(0, np.int32)
            for name in windows.fields
        }
        sizes = {len(value) for value in self.tokens.values()}
        if len(sizes) != 1:
            raise ValueError("causal window fields have different token counts")
        self.size = next(iter(sizes))

    def __len__(self) -> int:
        stride = self.windows.stride or self.windows.seq_len
        return max(0, (self.size - self.windows.seq_len - 1) // stride + 1)

    def __getitem__(self, index: int) -> Batch:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * (self.windows.stride or self.windows.seq_len)
        return {
            name: values[start : start + self.windows.seq_len + 1] for name, values in self.tokens.items()
        }


class _WindowIterator(grain.DatasetIterator[Batch]):
    def __init__(self, parent: grain.DatasetIterator[Batch], windows: Windows):
        super().__init__(parent)
        self._read, self._windows = parent, windows
        self._held: Batch = {name: np.zeros(0, np.int32) for name in windows.fields}
        self._skip = 0

    def __next__(self) -> Batch:
        size = self._windows.seq_len + 1
        while len(self._held[self._windows.fields[0]]) < size + self._skip:
            row = next(self._read)
            counts = {len(row[name]) for name in self._windows.fields}
            if len(counts) != 1:
                raise ValueError("causal window fields have different record lengths")
            self._held = {
                name: np.concatenate([self._held[name], np.asarray(row[name])])
                for name in self._windows.fields
            }
        if self._skip:
            self._held = {name: value[self._skip :] for name, value in self._held.items()}
        emitted = {name: value[:size] for name, value in self._held.items()}
        stride = self._windows.stride or self._windows.seq_len
        available = len(self._held[self._windows.fields[0]])
        self._held = {name: value[stride:] for name, value in self._held.items()}
        self._skip = max(0, stride - available)
        return emitted

    def get_state(self) -> Mapping[str, object]:
        return {
            "parent": self._read.get_state(),
            "skip": self._skip,
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

    def close(self) -> None:
        self._read.close()


class CausalWindows(grain.IterDataset[Batch]):
    def __init__(self, parent: grain.IterDataset[Batch], windows: Windows):
        if windows.seq_len < 1 or (windows.stride is not None and windows.stride < 1):
            raise ValueError("causal window lengths and strides must be positive")
        super().__init__(parent)
        self._records, self._windows = parent, windows

    def __iter__(self) -> grain.DatasetIterator[Batch]:
        return _WindowIterator(iter(self._records), self._windows)
