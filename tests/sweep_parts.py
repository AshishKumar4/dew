"""Stand-in shares of a sweep record for `tools/sweep_grouped_gemm._timed_apart`'s tests, importable by the
processes it spawns: one that returns its results, one that dies without them."""
import os


def returning_or_dying(name, product, implementation, sample, part, parts, lock, wanted, results) -> None:
    """Part 0 puts an empty share's results; every other part exits at once, as a process the kernel
    killed out of memory does, its results never put."""
    del name, product, implementation, sample, parts, lock, wanted
    if part == 0:
        results.put((part, {}, {}, {}))
        return
    os._exit(9)
