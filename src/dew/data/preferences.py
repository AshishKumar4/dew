"""Preference pairs for DPO: chosen and rejected rows with their masks.

A `PreferencePair` step reads records of
`chosen` and `rejected` token ids, lists or one-dimensional arrays, with
`chosen_mask` and `rejected_mask` marking the completion tokens. Each batch stacks them in
TRL's layout: chosen rows over rejected rows, padded to the longest row,
with `completion_mask` alongside the ids. A mask must run as long as its
ids; anything else fails naming the row.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING

import grain.python as grain
import numpy as np

from dew._records import record
from dew.batch import token_ids
from dew.data.facts import shaped, unshaped

if TYPE_CHECKING:
    from dew.batch import Batch

IDS_KEY = "input_ids"
"""Batch key holding the `[B, 2, S]` pair ids, chosen at index 0."""

MASK_KEY = "completion_mask"
"""Batch key holding the `[B, 2, S]` completion marks, 1 on completion tokens."""

FIELDS = ("chosen", "rejected", "chosen_mask", "rejected_mask")


def _mask(values: object, name: str, length: int, where: str) -> list[int]:
    """One 0/1 mask as long as its ids, a list, a tuple or a one-dimensional array as its ids are."""
    marks = values.tolist() if isinstance(values, np.ndarray) and values.ndim == 1 else values
    if not isinstance(marks, (list, tuple)) or len(marks) != length:
        raise ValueError(
            f"{where}: {name} runs as long as its ids ({length}), "
            f"got {values!r}")
    mask: list[int] = []
    for mark in marks:
        if mark not in (0, 1):
            raise ValueError(f"{where}: {name} marks completion tokens 0/1, got {mark!r}")
        mask.append(int(mark))
    return mask


@dataclasses.dataclass(frozen=True)
class PreferencePair(grain.MapTransform):
    """A chosen and a rejected completion's ids and loss masks laid out as two
    `seq_len` rows."""

    seq_len: int = 256
    _: dataclasses.KW_ONLY
    pad_id: int = 0

    def __post_init__(self) -> None:
        if self.seq_len < 1:
            raise ValueError("preference seq_len must be positive")

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """A chosen and a rejected row of `seq_len` ids and their completion mask, in place of the
        completions."""
        rows = shaped(unshaped(facts, *FIELDS), IDS_KEY, [2, self.seq_len])
        return {**shaped(rows, MASK_KEY, [2, self.seq_len]), "seq_len": self.seq_len}

    def map(self, element: Batch) -> Batch:
        element = record(element, "preference row")
        if "chosen" not in element or "rejected" not in element:
            raise ValueError("a preference pair needs both chosen and rejected ids")
        ids = np.full((2, self.seq_len), self.pad_id, np.int32)
        mask = np.zeros((2, self.seq_len), np.int32)
        for row, name in enumerate(("chosen", "rejected")):
            tokens = token_ids(element.get(name), where=f"preference: {name}")
            if len(tokens) > self.seq_len:
                raise ValueError(f"preference {name} has {len(tokens)} tokens for seq_len={self.seq_len}")
            completion = _mask(
                element.get(f"{name}_mask", [1] * len(tokens)), f"{name}_mask", len(tokens), "preference"
            )
            ids[row, : len(tokens)], mask[row, : len(tokens)] = tokens, completion
        kept = {key: value for key, value in element.items() if key not in FIELDS}
        return {**kept, IDS_KEY: ids, MASK_KEY: mask}
