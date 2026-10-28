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


def _dtype(value) -> np.dtype:
    """The dtype grain's packer lays a field in: its array's, or int64 for a Python int."""
    return np.dtype(np.int64) if isinstance(value, int) else np.asarray(value).dtype


def entries(value) -> int:
    """How many entries of a window a document's field takes: its length, or one for a scalar, as
    grain's packer counts it (`PackedBatch._get_element_lengths_flat`)."""
    return 1 if np.ndim(value) == 0 else len(value)


class PackedWindows:
    """`bins` windows documents are laid into end to end, as grain's packer lays them
    (`grain/_src/python/dataset/transformations/packing_packed_batch.py`, `PackedBatch`): each
    field in a zero-padded window of its own length (`lengths`), allocated at the dtype and trailing
    shape `template`, the first document of the set, holds (a token's row of an aligned auxiliary
    field), and each document copied in as it is added, so nothing it reads from is held.

    `<field>_segment_ids` says which document of the window each entry came from, counted from 1,
    and 0 for the padding at the end, and `<field>_positions` the entry's place inside its
    document, for each of the `paired` fields: the first token field, whose pair describes every
    token field, as documents are cut the same way in each, and each field along an axis of its
    own, as grain writes one for each field it does not take for a meta feature. A block-diagonal
    mask and per-document RoPE read the token pair."""

    def __init__(self, template: Batch, lengths: Mapping[str, int], paired: Sequence[str], bins: int):
        self.values = {key: np.zeros((bins, lengths[key], *np.shape(template[key])[1:]),
                                     _dtype(template[key])) for key in lengths}
        self.segment_ids = {key: np.zeros((bins, lengths[key]), np.int32) for key in paired}
        self.positions = {key: np.zeros((bins, lengths[key]), np.int32) for key in paired}
        self.filled = {key: [0] * bins for key in lengths}
        self.held = [0] * bins

    def add(self, window: int, document: Batch) -> None:
        """`document` laid after what window `window` holds."""
        self.held[window] += 1
        for key, values in self.values.items():
            value = document[key]
            start = self.filled[key][window]
            end = self.filled[key][window] = start + entries(value)
            values[window, start:end] = value
            if key in self.segment_ids:
                self.segment_ids[key][window, start:end] = self.held[window]
                self.positions[key][window, start:end] = np.arange(end - start, dtype=np.int32)

    def windows(self) -> list[Batch]:
        """Every window that holds a document, in order: the first ones, since an empty window takes
        any document a window has room for."""
        return [{**{key: values[window] for key, values in self.values.items()},
                 **{f"{key}_{part}": laid[key][window] for key in self.segment_ids
                    for part, laid in (("segment_ids", self.segment_ids), ("positions", self.positions))}}
                for window in range(sum(held > 0 for held in self.held))]


def packed_window(chunks: Sequence[Batch], window: int) -> Batch:
    """`chunks`, each holding the same per-token fields, laid end to end in a
    zero-padded window of `window` tokens as grain's packer lays them
    (`PackedWindows`), with one pair of segment ids and positions named after
    the first field.

    It is grain's assembly, held to it field for field (tests/test_grain_packing.py), and not grain's
    own packer, whose public form lays a twelve-chunk window in 3.6 ms on armada's CPUs where this
    takes 35 us, and an indexed read assembles a window a read (rows 1873, 1064).
    """
    laid = PackedWindows(chunks[0], dict.fromkeys(chunks[0], window), (next(iter(chunks[0])),), 1)
    for chunk in chunks:
        laid.add(0, chunk)
    (packed,) = laid.windows()
    return packed


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
        # The first field's room apart from the rest's: a plan of lengths places one field, compared
        # as one number, over a corpus of 200,000 chunks.
        self.free = [self.rooms[0]] * self.bins
        self.others = [list(self.rooms[1:]) for _ in range(self.bins)]
        self.held = [0] * self.bins

    def place(self, sizes: Sequence[int]) -> int | None:
        """The window a document of `sizes` entries in each field goes to, which takes them, or None
        where no open window has room for it. One longer than a window is refused, as grain refuses
        it: no window ever has room for it."""
        size, rest, cap, held = sizes[0], sizes[1:], self.segments, self.held
        if size > self.rooms[0] or (rest and any(
                other > room for other, room in zip(rest, self.rooms[1:], strict=True))):
            raise ValueError(f"a document of {list(sizes)} entries is longer than the windows of "
                             f"{list(self.rooms)} it packs into; cut it first")
        for window, free in enumerate(self.free):
            if free >= size and (cap is None or held[window] < cap) and (not rest or all(
                    other <= room for other, room in zip(rest, self.others[window], strict=True))):
                self.free[window] = free - size
                if rest:
                    self.others[window] = [room - other for other, room in zip(rest, self.others[window],
                                                                               strict=True)]
                held[window] += 1
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
    grain's packer run over the lengths: over 200,000 chunks on armada's CPUs this plans in 0.15 to
    0.22 s (windows of 2048 and 4096 tokens, 1 to 16 bins) and grain's packer, fed zero-strided
    stand-ins of each length, in 88 s, the same plan (rows 1873, 1064).

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
