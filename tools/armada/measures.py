"""What tools/armada/ci.py's GPU lane takes a test file to hold on a GPU: the share its process is given over
its recorded peak, and of the records its runs left, the one that held the most."""

import concurrent.futures
from collections.abc import Callable, Hashable, Iterable

GPU_HEADROOM = (1.25, 2 ** 30)
"""A file's share of the GPU over its recorded peak, a factor and bytes beside it: the allocator's
fragmentation, which can leave a share its own size short of a block it needs."""


def measured(recorded: dict | None) -> bool:
    """Whether `recorded`, a run's record, measured what its file held on the GPU: a run that followed every
    process it started and left its report."""
    return recorded is not None and recorded.get("depends") is not None and recorded.get("peak") is not None


def gpu_needs(recorded: dict | None, budget: float) -> tuple[float, float]:
    """The GPU share a test file's process is given, as `XLA_PYTHON_CLIENT_MEM_FRACTION` of it, and what
    the file holds with its children, by its last run's `recorded` peak (tools/armada/executed.py's
    `gpu_peak`, or a child's that took its fraction, the larger) and `GPU_HEADROOM`: a child that took
    the share's fraction holds one more share, and one that named its own fraction the pool it held.
    A file no run measured, one that started a process it did not follow, and one whose last run left
    no report (`measured`) take all of `budget`."""
    if not measured(recorded):
        return budget, budget
    factor, slack = GPU_HEADROOM
    share = min(recorded["peak"] * factor + slack, budget)
    return share, min(share * (1 + recorded.get("shares", 0)) + recorded.get("pools", 0), budget)


def largest(keys: Iterable[str], read: Callable[[str], dict | None],
            measures: Callable[[str], Hashable | None]) -> dict:
    """Of the records at `keys` (`read`: one's record, None where there is none), for each thing a record
    `measures` (None: nothing asked for), the most those whose runs measured it held (`measured`), as one
    record `gpu_needs` reads: the largest peak, the most children that took its fraction, and the largest
    pools its children named, each of whichever run, so its process is given no less than any run's and holds
    no less with its children (one record chosen by what it held gave a process the 6 GiB of a run with a
    child over the 11 GiB of a run without). A thing no such record measures is absent."""
    asked = [(key, of) for key in keys if (of := measures(key)) is not None]
    most: dict = {}
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        for (_, of), found in zip(asked, pool.map(read, [key for key, _ in asked]), strict=True):
            if measured(found):
                held = most.setdefault(of, {"depends": [], "peak": 0, "shares": 0, "pools": 0})
                for each in ("peak", "shares", "pools"):
                    held[each] = max(held[each], found.get(each, 0))
    return most
