"""Document chunk tables, packed windows, first-fit plans and a row's shares of its room."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import grain.python as pygrain
import numpy as np

from .dataset import Batch


def share_tokens(lengths: Sequence[int], room: int, *, caps: Sequence[int | None] | None = None,
                 whole: Sequence[bool] | None = None) -> list[int]:
    """How many of its own tokens each segment of a row keeps of the row's `room`: the one allocation
    segments that share a row read, as a decision row's copies of its state do.

    A segment kept whole (`whole`) takes all of its `lengths`, past the room too, which the row's own
    check then judges. The others share what that leaves, never below zero, each wanting its length
    up to its cap (`caps`, None for none). Where their wants exceed it, the share is max-min within one
    token: in a stable ascending order of want, each takes the least of its want and an equal split of
    what is left, floored, so a floor's remainder goes to the later segments in that order ([10, 10]
    into 7 keeps [3, 4], [100, 10, 10] into 4 keeps [2, 1, 1]). No share is below zero, the cut ones
    sum to at most what they share, and none of it is left while a segment wants more."""
    count = len(lengths)
    caps = [None] * count if caps is None else list(caps)
    whole = [False] * count if whole is None else list(whole)
    if len(caps) != count or len(whole) != count:
        raise ValueError(f"a cap and a whole flag for each of {count} segments, got {len(caps)} and "
                         f"{len(whole)}")
    cut = [at for at in range(count) if not whole[at]]
    wants = {at: lengths[at] if (cap := caps[at]) is None else min(cap, lengths[at]) for at in cut}
    left = max(0, room - sum(length for length, kept in zip(lengths, whole, strict=True) if kept))
    if sum(wants.values()) <= left:
        return [lengths[at] if whole[at] else wants[at] for at in range(count)]
    shares = [lengths[at] if whole[at] else 0 for at in range(count)]
    remaining = len(cut)
    for at in sorted(cut, key=lambda at: wants[at]):
        shares[at] = min(wants[at], left // remaining)
        left -= shares[at]
        remaining -= 1
    return shares


def chunk_table(lengths, chunk_len: int, overlap: int) -> tuple[np.ndarray, np.ndarray]:
    """The document each chunk is cut from and where in it the chunk starts, both int64 and one
    entry per chunk, for documents of `lengths` cut into pieces of at most `chunk_len` tokens, each
    starting `chunk_len - overlap` after the last (`dew.data.text.Chunks`).

    A window holds nothing longer than itself, so a document that outgrows it is cut first, and
    each chunk becomes its own segment of the window, which keeps attention inside the chunk and
    RoPE running from the chunk's own 0. A document of `chunk_len` tokens or fewer is one chunk; a
    document of no tokens is no chunks."""
    lengths = np.asarray(lengths, np.int64)
    stride = chunk_len - overlap
    counts = np.where(lengths > 0, -(-np.maximum(lengths - overlap, 1) // stride), 0)
    documents = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    return documents, (np.arange(len(documents), dtype=np.int64) - first) * stride


def chunk_lengths(lengths, chunk_len: int, overlap: int) -> np.ndarray:
    """The token count of each chunk `chunk_table` cuts documents of `lengths` into."""
    documents, starts = chunk_table(lengths, chunk_len, overlap)
    return np.minimum(chunk_len, np.asarray(lengths, np.int64)[documents] - starts)


def packed_window(chunks: Sequence[Batch], window: int) -> Batch:
    """`chunks`, each holding the same per-token fields, laid end to end in a
    zero-padded window of `window` tokens.

    Named after the first field (the ids), `<field>_segment_ids` says which
    chunk of the window each token came from, counted from 1, and 0 for the
    padding at the end, and `<field>_positions` the token's place inside its
    chunk. A block-diagonal mask and per-document RoPE read that pair.
    grain's packer writes one pair per packed feature
    (`grain/_src/python/dataset/transformations/packing_packed_batch.py:116-117`);
    chunks are cut the same way in every field here, so one pair describes
    them all and nothing transfers a second copy to the device.
    """
    fields = {key: np.zeros(window, value.dtype) for key, value in chunks[0].items()}
    segment_ids = np.zeros(window, np.int32)
    positions = np.zeros(window, np.int32)
    filled = 0
    for segment, chunk in enumerate(chunks, start=1):
        length = len(next(iter(chunk.values())))
        for key, value in chunk.items():
            fields[key][filled:filled + length] = value
        segment_ids[filled:filled + length] = segment
        positions[filled:filled + length] = np.arange(length, dtype=np.int32)
        filled += length
    ids = next(iter(fields))
    return {**fields, f"{ids}_segment_ids": segment_ids, f"{ids}_positions": positions}


def first_fit(sizes: np.ndarray, window: int, bins: int,
              segments: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The plan grain's first-fit packer would follow over chunks of `sizes`.

    Grain's packer holds `bins` open windows, adds each record to the first
    with room and fewer than `segments` records, and emits all of them when
    a record fits in none (`grain/_src/python/dataset/transformations/packing.py:341-355`,
    `packing_packed_batch.py` `FirstFitPackedBatch`). Run
    over the lengths alone it is a plan rather than a pass: which window
    every chunk belongs to, without reading a token. That plan is what lets
    the packing run ahead of the shard, so a window holds the same chunks at
    any process count.

    Returns the chunk indices grouped by window, in the order the packer
    added them, and where each window starts in that grouping. A window the
    packer never added anything to is dropped, since it would be a batch row
    of pure padding. Only the last set of bins can hold one, because a record
    goes to the first window with room and an empty window has room for
    anything.
    """
    if bins < 1:
        raise ValueError(f"a packer fills at least one window at a time, got {bins}")
    room = [window] * bins
    held = [0] * bins
    plan = np.empty(len(sizes), np.int64)
    closed = 0
    for index, size in enumerate(sizes.tolist()):
        chosen = -1
        for candidate, free in enumerate(room):
            if free >= size and (segments is None or held[candidate] < segments):
                chosen = candidate
                break
        if chosen < 0:
            # Every open window is short of room or full, so the packer emits
            # all of them and starts this chunk in the first of a fresh set.
            closed += bins
            room, held = [window] * bins, [0] * bins
            chosen = 0
        plan[index] = closed + chosen
        room[chosen] -= size
        held[chosen] += 1
    windows = closed + bins - room.count(window)
    starts = np.concatenate([np.zeros(1, np.int64),
                             np.cumsum(np.bincount(plan, minlength=windows))])
    return np.argsort(plan, kind="stable"), starts


class _WrappingDataset(pygrain.MapDataset[Batch]):
    """A dataset read by index, whose index wraps into its own length.

    grain's slice is the sharding and windowing API (`ds[shard::count]`), and
    an index past the end wraps, so `repeat` is a length change. Both are
    answered here and a subclass reads one record through `record`.
    """

    def record(self, index: int) -> Batch:
        """The record at `index`, which is inside this dataset's length."""
        raise NotImplementedError

    @overload
    def __getitem__(self, index: slice) -> pygrain.MapDataset[Batch]: ...

    @overload
    def __getitem__(self, index: int) -> Batch: ...

    def __getitem__(self, index: int | slice) -> Batch | pygrain.MapDataset[Batch]:
        if isinstance(index, slice):
            return self.slice(index)
        return self.record(index % len(self))
