"""A memory-mapped checkpoint tensor's pages handed back to the kernel once it is placed (`evict`): apart
from `dew.distributed.host`, so a loader that only evicts imports none of the host step's machinery, nor is
reached by its changes (the test cache keys what each test imports)."""

from __future__ import annotations

import mmap

import numpy as np


def evict(array: np.ndarray) -> None:
    """Drop a memory-mapped checkpoint tensor's pages from this process.

    A loader maps a checkpoint file once and hands out views into it, so
    letting a view go frees nothing while another view is alive. The pages a
    placed tensor was read from stay resident, and for a base that fills the
    host that is one copy too many.

    The pages the view covers whole are given back to the kernel here. A
    boundary page shared with a neighbour stays until the neighbour is
    placed. A later read of an evicted page faults it back from the file. An
    array that is not such a view is left alone."""
    root: np.ndarray = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    mapping = root.base
    if not isinstance(root, np.memmap) or not isinstance(mapping, mmap.mmap) or array.nbytes == 0:
        return
    # The map starts at the allocation granule below the memmap's file
    # offset; the memmap's data sits that remainder into it.
    start = root.__array_interface__["data"][0] - root.offset % mmap.ALLOCATIONGRANULARITY
    offset = array.__array_interface__["data"][0] - start
    first = -(-offset // mmap.PAGESIZE) * mmap.PAGESIZE
    last = (offset + array.nbytes) // mmap.PAGESIZE * mmap.PAGESIZE
    if last > first:
        mapping.madvise(mmap.MADV_DONTNEED, first, last - first)
