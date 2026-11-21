"""Grain's packers' window placements over documents' lengths alone: first-fit's and best-fit's, which a
plan of a pack's windows follows (`first_fit`) and a stream of documents packs by (`dew.data.packing`).
Apart from `dew.data.tokens`, whose helpers every reader of token rows imports, so a change of placement
reaches only the code that places, as the test cache keys what each test imports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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
        if sizes[0] > self.rooms[0] or any(
                other > room for other, room in zip(sizes[1:], self.rooms[1:], strict=True)):
            raise ValueError(f"a document of {list(sizes)} entries is longer than the windows of "
                             f"{list(self.rooms)} it packs into; cut it first")
        window = self._chosen(sizes)
        if window is not None:
            self.free[window] -= sizes[0]
            if len(sizes) > 1:
                self.others[window] = [room - other for other, room in zip(sizes[1:], self.others[window],
                                                                           strict=True)]
            self.held[window] += 1
        return window

    def _chosen(self, sizes: Sequence[int]) -> int | None:
        """The first window with room for every field of the document and fewer than `segments`."""
        size, rest, cap, held = sizes[0], sizes[1:], self.segments, self.held
        for window, free in enumerate(self.free):
            if free >= size and (cap is None or held[window] < cap) and (not rest or all(
                    other <= room for other, room in zip(rest, self.others[window], strict=True))):
                return window
        return None

    def holding(self) -> int:
        """How many windows hold a document: the first ones, since an empty window takes any document
        a window has room for."""
        return sum(held > 0 for held in self.held)


class BestFit(FirstFit):
    """Grain's best-fit placement over one field's lengths (`BestFitPackedBatch.try_add_to_batch`, grain
    0.2.18), which a published best-fit plan follows: a document goes to the open window it fills most, the
    first where they tie; where none has room, the packer emits every window that holds a document and
    starts afresh, as first-fit's does. A window takes a document only once every window before it has, so
    the windows holding one are the first ones (`holding`)."""

    def __init__(self, room: int, bins: int, segments: int | None = None):
        super().__init__((room,), bins, segments)

    def _chosen(self, sizes: Sequence[int]) -> int | None:
        (size,), room, cap = sizes, self.rooms[0], self.segments
        chosen, fullest = None, -1
        for window, free in enumerate(self.free):
            used = room - free
            # A window at its documents' cap takes one only where it then holds no token, as grain adds
            # its capacity to such a window's (`max_sequences_per_bin`): an empty document beside empty
            # ones.
            capped = cap is not None and self.held[window] >= cap
            if used + size <= (0 if capped else room) and used > fullest:
                chosen, fullest = window, used
        return chosen


def first_fit(sizes: np.ndarray, window: int, bins: int,
              segments: int | None = None, *, policy: str = "first-fit") -> tuple[np.ndarray, np.ndarray]:
    """The plan grain's first-fit packer would follow over chunks of `sizes` (`FirstFit`), or its best-fit
    packer's where `policy` is "best-fit" (`BestFit`, a published best-fit pack's plan: row 9053).

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
    placement = (BestFit(window, bins, segments) if policy == "best-fit"
                 else FirstFit((window,), bins, segments))
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
