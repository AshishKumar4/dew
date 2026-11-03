"""Wall-clock timing of calls: `samples`, which every benchmark tool in Dew measures independent
calls with, `carried`, which times calls that carry a state from one to the next, `benchmark`,
which lays out many calls' timings as a table, and `compared`, which reads two calls against each
other by their paired ratio; and the process's resident memory as the kernel reports it
(`resident_memory`), which the memory benchmarks read.

A run is timed from the call to the moment its output is ready on the devices
(`jax.block_until_ready`), so an asynchronous dispatch is timed as the work it
starts, and reading the output back to the host is not timed. The warm-up runs,
which compile what the call traces, are not kept. The caller passes what is
timed, a jitted function bound to its inputs: nothing is jitted here.
"""

from __future__ import annotations

import dataclasses
import html
import statistics
import time
from collections.abc import Callable, Hashable, Mapping

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class Benchmark[T]:
    """What `benchmark` measured: each call's timed runs in seconds, in order, and its last output.

    `table()` and a notebook's display show each call's median. Labels that are all pairs,
    `(row, column)`, lay out as a grid of rows by columns, a sampler against its step counts say.
    """

    seconds: Mapping[Hashable, tuple[float, ...]]
    outputs: Mapping[Hashable, T]

    def median(self, label: Hashable) -> float:
        return statistics.median(self.seconds[label])

    def _grid(self) -> tuple[list[str], list[tuple[str, list[float | None]]]]:
        """The column names and each row's medians: a grid for pair labels, one column otherwise."""
        labels = list(self.seconds)
        pairs = [label for label in labels if isinstance(label, tuple) and len(label) == 2]
        if labels and len(pairs) == len(labels):
            rows = list(dict.fromkeys(row for row, _ in pairs))
            columns = list(dict.fromkeys(column for _, column in pairs))
            return [str(column) for column in columns], [
                (str(row), [self.median((row, column)) if (row, column) in self.seconds else None
                            for column in columns]) for row in rows]
        return ["median"], [(str(label), [self.median(label)]) for label in labels]

    def _cells(self) -> tuple[list[str], list[tuple[str, list[str]]]]:
        """The grid's medians written in one unit: milliseconds, or seconds once any reaches ten."""
        columns, rows = self._grid()
        known = [value for _, values in rows for value in values if value is not None]
        scale, unit = (1.0, "s") if known and max(known) >= 10 else (1000.0, "ms")
        return columns, [(name, ["" if value is None else f"{value * scale:,.1f} {unit}" for value in values])
                         for name, values in rows]

    def table(self) -> str:
        """The medians as aligned plain text, one row a label (or a row of pair labels)."""
        columns, rows = self._cells()
        first = max([len(name) for name, _ in rows] + [0])
        widths = [max([len(column)] + [len(values[index]) for _, values in rows])
                  for index, column in enumerate(columns)]
        def line(name: str, cells: list[str]) -> str:
            return f"{name:<{first}}" + "".join(f"  {cell:>{width}}" for cell, width in zip(cells, widths,
                                                                                            strict=True))

        return "\n".join([line("", columns), *(line(name, values) for name, values in rows)])

    def _repr_html_(self) -> str:
        columns, rows = self._cells()
        head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        body = "".join(f"<tr><th>{html.escape(name)}</th>" + "".join(
            f'<td style="text-align:right">{html.escape(value)}</td>' for value in values) + "</tr>"
            for name, values in rows)
        return f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>"

    def __str__(self) -> str:
        return self.table()


def resident_memory() -> dict[str, int]:
    """This process's memory fields of Linux's /proc/self/status (VmRSS, RssAnon, RssFile, VmHWM,
    ...), each in bytes, under the kernel's names; a benchmark keeps the fields and unit it reports.

    The resident counts are the kernel's running counters, which since Linux 6.2 each CPU batches
    before they reach the process's total: a reading lags the pages just written by up to a batch
    for each CPU the writing threads ran on, on a machine of many CPUs tens of MiB."""
    fields = {}
    with open("/proc/self/status") as handle:
        for line in handle:
            name, _, value = line.partition(":")
            match value.split():
                case [amount, "kB"]:
                    fields[name] = int(amount) * 1024
    return fields


def largest_difference(left, right) -> float:
    """The largest elementwise difference between two pytrees of arrays, in fp32: how far one
    implementation's outputs sit from another's."""
    return max(float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))
               for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def samples[T](fn: Callable[..., T], *args, warmup: int = 3, repeats: int = 10,
               calls: int = 1) -> list[float]:
    """Seconds per call of `fn(*args)`, one sample a synchronized group of
    `calls` calls, after `warmup` groups. With one call a group each sample is
    a call's latency, and its result is released before the next call starts,
    so no two calls' outputs are ever held at once; more calls a group queue
    their dispatches and synchronize once, which measures throughput. The
    caller takes the statistic and units it reports."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    measured = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = fn(*args)
        for _ in range(calls - 1):
            outputs = fn(*args)
        jax.block_until_ready(outputs)
        measured.append((time.perf_counter() - start) / calls)
        del outputs
    return measured


def carried[S](step: Callable[[S], S], state: S, count: int, *, each: bool = False) -> tuple[S, list[float]]:
    """`count` calls of `step`, each handed the state the last returned (a training step's state, a
    decoder's cache), and the seconds they took, with the state the last call returned. With `each`,
    one sample a call, synchronized before the next starts, which times every call alone; otherwise
    one sample the window, its calls dispatched as a run dispatches them and synchronized once at its
    end, which times their throughput. A carried state is never released between calls, so its
    windows are timed apart from `samples`, whose calls are independent. Refuses no call."""
    if count < 1:
        raise ValueError(f"a measured window takes at least one call, not {count}")
    if each:
        seconds = []
        for _ in range(count):
            start = time.perf_counter()
            state = jax.block_until_ready(step(state))
            seconds.append(time.perf_counter() - start)
        return state, seconds
    start = time.perf_counter()
    for _ in range(count):
        state = step(state)
    state = jax.block_until_ready(state)
    return state, [time.perf_counter() - start]


@dataclasses.dataclass(frozen=True)
class Comparison:
    """What `compared` measured: each pair's ratio of the first call's seconds to the second's, the median
    ratio and its bootstrap interval, and the second call against itself (its A/A), whose interval is the
    placement's own noise: a difference is one where `interval` lies apart from 1 and outside `alone`."""

    ratios: tuple[float, ...]
    ratio: float
    interval: tuple[float, float]
    alone: tuple[float, float]


def compared[T](first: Callable[[], T], second: Callable[[], T], *, pairs: int = 400, warmup: int = 3,
             level: float = 0.95, resamples: int = 2000, seed: int = 0) -> Comparison:
    """`first` against `second`, each timed as `samples` times one call, in `pairs` interleaved pairs whose
    order swaps each pair (a call that runs second in a pair runs on whatever the first left), and
    `second` against itself in as many pairs, read as the median of each pair's ratio of seconds and its
    bootstrap interval at `level` over `resamples` resamples of the pairs: only paired ratios carry from
    one machine to another, and one-sided runs of an unchanged call moved -0.6% to +0.6% (the kernel
    challenge's measurements, row 3616). The caller binds each call to fresh inputs where its time depends
    on its data (a MoE's routing)."""
    for _ in range(warmup):
        jax.block_until_ready(first())
        jax.block_until_ready(second())

    def timed(call: Callable[[], T]) -> float:
        start = time.perf_counter()
        jax.block_until_ready(call())
        return time.perf_counter() - start

    def paired(left: Callable[[], T], right: Callable[[], T]) -> np.ndarray:
        ratios = []
        for index in range(pairs):
            if index % 2:
                right_seconds, left_seconds = timed(right), timed(left)
            else:
                left_seconds, right_seconds = timed(left), timed(right)
            ratios.append(left_seconds / right_seconds)
        return np.asarray(ratios)

    rng = np.random.default_rng(seed)

    def bounds(ratios: np.ndarray) -> tuple[float, float]:
        medians = np.median(rng.choice(ratios, (resamples, len(ratios))), axis=1)
        tail = (1.0 - level) / 2.0
        return float(np.quantile(medians, tail)), float(np.quantile(medians, 1.0 - tail))

    ratios = paired(first, second)
    return Comparison(tuple(float(ratio) for ratio in ratios), float(np.median(ratios)), bounds(ratios),
                      bounds(paired(second, second)))


def benchmark[T](calls: Mapping[Hashable, Callable[[], T]], *, warmup: int = 1,
                 repeats: int = 3) -> Benchmark[T]:
    """Time each of `calls`, in order, `repeats` times after `warmup` untimed runs (`samples`), and
    return the runs' seconds and each call's last output (`Benchmark`); no two of a call's outputs
    are held at once. Raises `ValueError` for no timed run or a negative warm-up."""
    if repeats < 1 or warmup < 0:
        raise ValueError(f"a benchmark times at least one run after none or more warm-ups, not "
                         f"repeats={repeats} and warmup={warmup}")
    seconds: dict[Hashable, tuple[float, ...]] = {}
    outputs: dict[Hashable, T] = {}
    for label, call in calls.items():
        ran = 0

        def kept(label=label, call=call):
            # The last run's output stays: `samples` releases each one before the next run.
            nonlocal ran
            ran += 1
            output = call()
            if ran == warmup + repeats:
                outputs[label] = output
            return output

        seconds[label] = tuple(samples(kept, warmup=warmup, repeats=repeats))
    return Benchmark(seconds, outputs)


__all__ = ["Benchmark", "Comparison", "benchmark", "carried", "compared", "samples"]
