"""What tools/armada/ci.py's GPU lane takes a test file to hold on a GPU: the share its process is given over
its recorded peak, and of the records its runs left, the one that held the most."""

import concurrent.futures
from collections.abc import Callable, Hashable, Iterable

GPU_HEADROOM = (1.25, 2 ** 30)
"""A file's share of the GPU over its recorded peak, a factor and bytes beside it: the allocator's
fragmentation, which can leave a share its own size short of a block it needs."""


def gpu_needs(recorded: dict | None, budget: float) -> tuple[float, float]:
    """The GPU share a test file's process is given, as `XLA_PYTHON_CLIENT_MEM_FRACTION` of it, and what
    the file holds with its children, by its last run's `recorded` peak (tools/armada/executed.py's
    `gpu_peak`, or a child's that took its fraction, the larger) and `GPU_HEADROOM`: a child that took
    the share's fraction holds one more share, and one that named its own fraction the pool it held.
    A file no run measured, one that started a process it did not follow, and one whose last run left
    no report take all of `budget`."""
    if recorded is None or recorded["depends"] is None or recorded["peak"] is None:
        return budget, budget
    factor, slack = GPU_HEADROOM
    share = min(recorded["peak"] * factor + slack, budget)
    return share, min(share * (1 + recorded.get("shares", 0)) + recorded.get("pools", 0), budget)


def largest(keys: Iterable[str], read: Callable[[str], dict | None],
            measures: Callable[[str], Hashable | None]) -> dict:
    """Of the records at `keys` (`read`: one's record, None where there is none), for each thing a record
    `measures` (None: nothing asked for), the one whose run measured its GPU peak that holds the most
    with its children (`gpu_needs`). A thing no such record measures is absent."""
    asked = [(key, of) for key in keys if (of := measures(key)) is not None]

    def holds(found: dict) -> float:
        return gpu_needs(found, float("inf"))[1]

    most: dict = {}
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        for (_, of), found in zip(asked, pool.map(read, [key for key, _ in asked]), strict=True):
            if found is None or found.get("depends") is None or found.get("peak") is None:
                continue
            if of not in most or holds(found) > holds(most[of]):
                most[of] = found
    return most
