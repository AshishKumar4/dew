"""A function mapped over items in forked processes, in order, for the gates whose readings parse every
file of the tree (tools/duplicates.py's text, tools/lint_slop.py's rules): one worker a processor this
process may use, each a fork of the parent, which must hold no threads (a script's own process, not a
test's under JAX)."""

import gc
import multiprocessing
import os
from collections.abc import Callable


def collecting_rarely() -> None:
    """A worker's collector, run a thousandth as often: the trees these readings build are cycles enough
    that the default thresholds collect thousands of times a run."""
    gc.set_threshold(700_000, 10, 10)


def forked_map[Item, Result](function: Callable[[Item], Result], items: list[Item]) -> list[Result]:
    """`function` over `items` in forked processes, one a processor this process may use, in order. The
    parent runs it once first, so what every call loads once (a parser's modules and caches) is loaded before
    the fork and inherited, not loaded again in each worker, and freezes what it holds, which no worker's
    collector then traces: both tripled a reading's time."""
    for item in items[:1]:
        function(item)
    gc.freeze()
    context = multiprocessing.get_context("fork")
    with context.Pool(len(os.sched_getaffinity(0)), initializer=collecting_rarely) as pool:
        return pool.map(function, items, chunksize=8)
