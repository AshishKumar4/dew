"""Tokenization and explicit over-length policies."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.data.text import tokenizer_for, tokenizer_snapshot


@dataclasses.dataclass(frozen=True)
class Tokenize(grain.MapTransform):
    tokenizer: str = "byte"
    field: str = "text"
    eos: bool = False

    def dependencies(self) -> Mapping[str, object]:
        return {"tokenizer": tokenizer_snapshot(self.tokenizer)}

    def map(self, element: Batch) -> Batch:
        value = element[self.field]
        if not isinstance(value, str):
            raise ValueError(f"Tokenize field {self.field!r} must contain text")
        tokenizer = tokenizer_for(self.tokenizer)
        ids = tokenizer.encode(value)
        if self.eos and ids:
            if tokenizer.eos_id is None:
                raise ValueError(f"tokenizer {self.tokenizer} has no EOS id")
            ids.append(tokenizer.eos_id)
        return {"text": np.asarray(ids, np.int32), "_dew_length": len(ids)}


@dataclasses.dataclass(frozen=True)
class LengthPolicy(grain.MapTransform):
    length: int
    field: str = "text"
    long: Literal["truncate", "drop"] = "truncate"
    side: Literal["left", "right"] = "right"
    aligned: tuple[str, ...] = ()
    rejects = True

    def map(self, element: Batch) -> Batch | Rejected:
        count = len(element[self.field])
        if count <= self.length:
            return element
        if self.long == "drop":
            return Rejected("over_length")
        span = slice(count - self.length, None) if self.side == "left" else slice(self.length)
        return {
            key: value[span] if key in (self.field, *self.aligned) else value
            for key, value in element.items()
        } | {"_dew_truncated": True, "_dew_length": self.length}


@dataclasses.dataclass(frozen=True)
class Chunks:
    length: int
    fields: tuple[str, ...] = ("text",)

    def flat_map(self, element: Batch) -> Sequence[Batch]:
        count = len(element[self.fields[0]])
        return tuple(
            {
                key: value[start : start + self.length] if key in self.fields else value
                for key, value in element.items()
            }
            for start in range(0, count, self.length)
        )


@dataclasses.dataclass(frozen=True)
class ValidTargets:
    field: str = "text"
    role: int | None = None
    pad_id: int | None = None
    measured: bool = True

    def __call__(self, element: Batch) -> int:
        if self.measured and "_dew_valid_targets" in element:
            return int(element["_dew_valid_targets"])
        segments = element.get(f"{self.field}_segment_ids")
        if segments is None:
            valid = np.ones(max(0, len(element[self.field]) - 1), bool)
        else:
            segments = np.asarray(segments)
            valid = (segments[1:] > 0) & (segments[1:] == segments[:-1])
        if self.role is not None:
            valid &= np.asarray(element[f"{self.field}_roles"])[1:] == self.role
        if self.pad_id is not None:
            valid &= np.asarray(element[self.field])[1:] != self.pad_id
        return int(valid.sum())


@dataclasses.dataclass(frozen=True)
class CountTargets(grain.MapTransform):
    role: int | None = None
    pad_id: int | None = None
    rejects = True

    def map(self, element: Batch) -> Batch | Rejected:
        count = ValidTargets(role=self.role, pad_id=self.pad_id, measured=False)(element)
        return Rejected("zero_targets") if count == 0 else {**element, "_dew_valid_targets": count}
