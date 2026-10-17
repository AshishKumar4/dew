"""Preference pairs for DPO: chosen and rejected rows with their masks.

A `PreferencePair` step reads records of
`chosen` and `rejected` token ids, lists or one-dimensional arrays, with
`chosen_mask` and `rejected_mask` marking the completion tokens. Each batch stacks them in
TRL's layout: chosen rows over rejected rows, padded to the longest row,
with `completion_mask` alongside the ids. A mask must run as long as its
ids; anything else fails naming the row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dew.data.steps.chat import PreferencePair

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


def pair(seq_len: int = 256, *, pad_id: int = 0) -> PreferencePair:
    """A record step that lays a chosen and a rejected completion's ids and
    loss masks out as two `seq_len` rows."""
    from dew.data.steps.chat import PreferencePair

    return PreferencePair(seq_len, pad_id)
