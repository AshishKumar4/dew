"""Document chunk tables, packed windows, first-fit plans and a row's shares of its room."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import overload

import grain.python as pygrain
import numpy as np

from dew.batch import Batch


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


def packed_window(chunks: Sequence[Batch], window: int, along: Mapping[str, int] | None = None) -> Batch:
    """`chunks`, each holding the same fields, laid end to end in zero-padded
    windows, as grain's packer lays them: the token fields, which hold one
    entry a token and so one length, in a window of `window` tokens, and each
    field `along` an axis of its own in a window of its own length. Each field
    keeps its trailing shape (a token's row of an aligned auxiliary field).

    `<field>_segment_ids` says which chunk of the window each entry came from,
    counted from 1, and 0 for the padding at the end, and `<field>_positions`
    the entry's place inside its chunk: one pair named after the first token
    field, which describes every token field, as chunks are cut the same way in
    each, and one for each field along an axis of its own, as grain's packer
    writes one for each field it does not take for a meta feature
    (`grain/_src/python/dataset/transformations/packing_packed_batch.py:85-115`).
    A block-diagonal mask and per-document RoPE read the token pair.

    It is grain's assembly, held to it field for field (tests/test_grain_packing.py), and not grain's
    own: on armada's CPUs grain's `PackedBatch` assembles a twelve-chunk window in 112 us and its
    public packer in 3.6 ms where this takes 28 us, and an indexed read assembles a window a read
    (row 1873).
    """
    lengths = {key: window if along is None else along.get(key, window) for key in chunks[0]}
    fields = {key: np.zeros((lengths[key], *np.shape(value)[1:]), np.asarray(value).dtype)
              for key, value in chunks[0].items()}
    tokens = next(key for key in fields if along is None or key not in along)
    paired = (tokens, *(() if along is None else (key for key in fields if key in along)))
    segment_ids = {key: np.zeros(lengths[key], np.int32) for key in paired}
    positions = {key: np.zeros(lengths[key], np.int32) for key in paired}
    filled = dict.fromkeys(fields, 0)
    for segment, chunk in enumerate(chunks, start=1):
        for key, value in chunk.items():
            start = filled[key]
            filled[key] = start + len(value)
            fields[key][start:filled[key]] = value
            if key in segment_ids:
                segment_ids[key][start:filled[key]] = segment
                positions[key][start:filled[key]] = np.arange(filled[key] - start, dtype=np.int32)
    described = {f"{key}_{part}": value for key in paired
                 for part, value in (("segment_ids", segment_ids[key]), ("positions", positions[key]))}
    return {**fields, **described}


class FirstFit:
    """Grain's first-fit placement (`FirstFitPackedBatch.try_add_to_batch`): `bins` open windows,
    each with room for `rooms` entries of each field, a document going to the first window with
    room for every field of it and fewer than `segments` documents. Where none has, the packer
    emits every window that holds a document, in order, and starts afresh
    (`grain/_src/python/dataset/transformations/packing.py:250-266`).

    The one placement both of Dew's first-fit packers follow: a plan over documents' lengths alone
    (`first_fit`), and the packer of a stream of them (`dew.data.packing.FirstFitPacked`)."""

    def __init__(self, rooms: Sequence[int], bins: int, segments: int | None = None):
        if bins < 1:
            raise ValueError(f"a packer fills at least one window at a time, got {bins}")
        self.rooms, self.bins, self.segments = tuple(rooms), bins, segments
        self.restart()

    def restart(self) -> None:
        """Every window open and empty, as after the packer emits them."""
        self.free = [list(self.rooms) for _ in range(self.bins)]
        self.held = [0] * self.bins

    def place(self, sizes: Sequence[int]) -> int | None:
        """The window a document of `sizes` entries in each field goes to, which takes them, or None
        where no open window has room for it. One longer than a window is refused, as grain refuses
        it: no window ever has room for it."""
        if any(size > room for size, room in zip(sizes, self.rooms, strict=True)):
            raise ValueError(f"a document of {list(sizes)} entries is longer than the windows of "
                             f"{list(self.rooms)} it packs into; cut it first")
        for window, (free, held) in enumerate(zip(self.free, self.held, strict=True)):
            if (self.segments is None or held <= self.segments) and all(
                    size <= room for size, room in zip(sizes, free, strict=True)):
                self.free[window] = [room - size for size, room in zip(sizes, free, strict=True)]
                self.held[window] += 1
                return window
        return None

    def holding(self) -> int:
        """How many windows hold a document: the first ones, since an empty window takes any document
        a window has room for."""
        return sum(held > 0 for held in self.held)


def first_fit(sizes: np.ndarray, window: int, bins: int,
              segments: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The plan grain's first-fit packer would follow over chunks of `sizes` (`FirstFit`).

    Run over the lengths alone it is a plan rather than a pass: which window
    every chunk belongs to, without reading a token. That plan is what lets
    the packing run ahead of the shard, so a window holds the same chunks at
    any process count.

    It is grain's placement, held to it window for window (tests/test_grain_packing.py), and not
    grain's packer run over the lengths: over 200,000 chunks on armada's CPUs this plans in 0.15 s
    and grain's packer, fed zero-strided stand-ins of each length, in 88 s, the same plan (row 1873).

    Returns the chunk indices grouped by window, in the order the packer
    added them, and where each window starts in that grouping. A window the
    packer never added anything to is dropped, since it would be a batch row
    of pure padding. Only the last set of bins can hold one, because a record
    goes to the first window with room and an empty window has room for
    anything.
    """
    placement = FirstFit((window,), bins, segments)
    plan = np.empty(len(sizes), np.int64)
    closed = 0
    for index, size in enumerate(sizes.tolist()):
        chosen = placement.place((size,))
        if chosen is None:
            # Every open window is short of room or full, so the packer emits
            # all of them and starts this chunk in the first of a fresh set.
            closed += bins
            placement.restart()
            chosen = placement.place((size,))
            assert chosen == 0
        plan[index] = closed + chosen
    windows = closed + placement.holding()
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
