"""Tokenization and explicit over-length policies."""

from __future__ import annotations

import dataclasses
import functools
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Literal

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.data.text import ByteTokenizer, HFTokenizer, tokenizer_for


@dataclasses.dataclass(frozen=True)
class Tokenize(grain.MapTransform):
    """The text in `field` as token ids under "text", each document ended by
    the tokenizer's EOS with `eos`, or where the tokenizer has one with None."""

    tokenizer: str = "byte"
    field: str = "text"
    eos: bool | None = False

    @functools.cached_property
    def vocabulary(self) -> ByteTokenizer | HFTokenizer:
        """The tokenizer `tokenizer` names, which resolves its name once
        (`HFTokenizer.pinned`): the step's identity and its encoding both read
        that one resolution, in this process and in any worker it is sent to."""
        return tokenizer_for(self.tokenizer)

    def dependencies(self) -> Mapping[str, object]:
        return {"tokenizer": self.vocabulary.dependencies()}

    def map(self, element: Batch) -> Batch:
        value = element[self.field]
        if not isinstance(value, str):
            raise ValueError(f"Tokenize field {self.field!r} must contain text")
        tokenizer = self.vocabulary
        ids = tokenizer.encode(value)
        if (self.eos or (self.eos is None and tokenizer.eos_id is not None)) and ids:
            if tokenizer.eos_id is None:
                raise ValueError(f"tokenizer {self.tokenizer} has no EOS id")
            ids.append(tokenizer.eos_id)
        kept = {key: value for key, value in element.items() if key != self.field}
        return {**kept, "text": np.asarray(ids, np.int32), "_dew_length": len(ids)}


@dataclasses.dataclass(frozen=True)
class Dropped(grain.RandomMapTransform):
    """The text in `field` emptied with probability `p`."""

    field: str = "caption"
    p: float = 0.1

    def __post_init__(self) -> None:
        if not 0 <= self.p <= 1:
            raise ValueError(f"a drop probability is in [0, 1], not {self.p}")

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        return {**element, self.field: ""} if rng.random() < self.p else element


@dataclasses.dataclass(frozen=True)
class MinChars:
    """Whether the text in `field` holds at least `count` characters."""

    count: int
    field: str = "text"
    reason = "too_short"

    def __call__(self, element: Batch) -> bool:
        return len(str(element[self.field])) >= self.count


@dataclasses.dataclass(frozen=True)
class Normalized(grain.MapTransform):
    """The text in `field` in Unicode normal form `form` (`unicodedata.normalize`); other fields
    pass through."""

    form: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFKC"
    field: str = "text"

    def map(self, element: Batch) -> Batch:
        text = element[self.field]
        text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
        return {**element, self.field: unicodedata.normalize(self.form, text)}


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
class Chunks:
    """A document's token `fields` cut into pieces of at most `length`, each
    starting `length - overlap` after the last, so with `overlap=1` every
    token's next-token target stays in one piece."""

    length: int
    fields: tuple[str, ...] = ("text",)
    overlap: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < self.length:
            raise ValueError(f"chunks of {self.length} overlap by 0 to {self.length - 1}, not {self.overlap}")

    def flat_map(self, element: Batch) -> Sequence[Batch]:
        count = len(element[self.fields[0]])
        stride = self.length - self.overlap
        starts = range(0, max(count - self.overlap, 1), stride)
        return tuple(
            {
                key: value[start : start + self.length] if key in self.fields else value
                for key, value in element.items()
            }
            | {"_dew_length": min(self.length, count - start)}
            for start in starts
        )


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
