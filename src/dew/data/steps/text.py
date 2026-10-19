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
    """Each record's targets the loss scores, counted once and carried on the record with the rule
    that counted them (`recounted`), so a token mixture weighs it without counting it again."""

    role: int | None = None
    pad_id: int | None = None
    rejects = True

    def map(self, element: Batch) -> Batch | Rejected:
        count = ValidTargets(role=self.role, pad_id=self.pad_id, measured=False)(element)
        if count == 0:
            return Rejected("zero_targets")
        rule = {name: value for name, value in (("_dew_target_role", self.role),
                                                ("_dew_target_pad", self.pad_id)) if value is not None}
        return {**element, "_dew_valid_targets": count, **rule}


_COUNTED = ("text", "text_segment_ids", "text_roles")
"""The fields a record's targets are counted from (`ValidTargets`)."""


def targets_changed(given: Batch, output: Batch) -> bool:
    """Whether a step that made `output` of `given` changed a field its targets are counted from."""
    return any(output.get(name) is not given.get(name) for name in _COUNTED)


def recounted(given: Batch, output: Batch) -> Batch:
    """`output` with the targets counted on `given` (`CountTargets`): kept while the step left every
    field they were counted from as it was, as a map of a record's other fields does, and counted
    again by the rule that counted them where it changed one, as a crop does. A record left without
    the field it was counted from, as a renaming moves it, keeps its count."""
    if "_dew_valid_targets" not in output or "text" not in output or not targets_changed(given, output):
        return output
    rule = ValidTargets(role=output.get("_dew_target_role"), pad_id=output.get("_dew_target_pad"),
                        measured=False)
    return {**output, "_dew_valid_targets": rule(output)}


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
