"""Over-length policies, fixed-length checks and target counts of token records."""

from __future__ import annotations

import dataclasses
from typing import Literal

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.pipeline import Rejected


@dataclasses.dataclass(frozen=True)
class LengthPolicy(grain.MapTransform):
    length: int
    field: str = "text"
    long: Literal["truncate", "drop"] = "truncate"
    side: Literal["left", "right"] = "right"
    aligned: tuple[str, ...] = ()

    @property
    def rejects(self) -> bool:
        """Whether a long record is dropped (`filter_map`) rather than cut (`map`)."""
        return self.long == "drop"

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
class ValidTargets:
    field: str = "text"
    role: int | None = None
    pad_id: int | None = None
    measured: bool = True

    def __call__(self, element: Batch) -> int:
        """The record's targets the loss scores, by the LM objective's own rule (`scored_targets`)."""
        if self.measured and "_dew_valid_targets" in element:
            return int(element["_dew_valid_targets"])
        from dew.objectives.lm.objective import scored_targets

        segments = element.get(f"{self.field}_segment_ids")
        roles = None if self.role is None else np.asarray(element[f"{self.field}_roles"])[None]
        return int(scored_targets(np, np.asarray(element[self.field])[None], self.pad_id, None,
                                  None if segments is None else np.asarray(segments)[None],
                                  roles=roles, role=self.role).sum())


@dataclasses.dataclass(frozen=True)
class CountTargets(grain.MapTransform):
    role: int | None = None
    pad_id: int | None = None
    rejects = True

    def map(self, element: Batch) -> Batch | Rejected:
        count = ValidTargets(role=self.role, pad_id=self.pad_id, measured=False)(element)
        return Rejected("zero_targets") if count == 0 else {**element, "_dew_valid_targets": count}


@dataclasses.dataclass(frozen=True)
class FixedTokens(grain.MapTransform):
    """Rows of exactly `max_length + extra` tokens of `field`, the
    `max_length` a token budget divides into rows (`TokenGeometry`)."""

    max_length: int
    extra: int = 0
    field: str = "text"

    def map(self, element: Batch) -> Batch:
        size = len(element[self.field])
        if size != self.max_length + self.extra:
            raise ValueError(f"fixed tokens need {self.max_length + self.extra} values, got {size}")
        return element
