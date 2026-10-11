"""Preference pairs for DPO: chosen and rejected rows with their masks.

A `PreferencePairs` source reads a parquet file or in-memory JSON rows of
`chosen` and `rejected` token-id lists, with `chosen_mask` and
`rejected_mask` marking the completion tokens. Each batch stacks them in
TRL's layout: chosen rows over rejected rows, padded to the longest row,
with `completion_mask` alongside the ids. A mask must run as long as its
ids; anything else fails naming the row.
"""

from __future__ import annotations



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


