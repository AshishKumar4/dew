"""Timing calls: `benchmark` runs each call so many times and keeps the wall time of each run.

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


@dataclasses.dataclass(frozen=True)
class Benchmark:
    """What `benchmark` measured: each call's timed runs in seconds, in order, and its last output.

    `table()` and a notebook's display show each call's median. Labels that are all pairs,
    `(row, column)`, lay out as a grid of rows by columns, a sampler against its step counts say.
    """

    seconds: Mapping[Hashable, tuple[float, ...]]
    outputs: Mapping[Hashable, object]

    def median(self, label: Hashable) -> float:
        return statistics.median(self.seconds[label])

    def _grid(self) -> tuple[list[str], list[tuple[str, list[float | None]]]]:
        """The column names and each row's medians: a grid for pair labels, one column otherwise."""
        labels = list(self.seconds)
        if labels and all(isinstance(label, tuple) and len(label) == 2 for label in labels):
            rows = list(dict.fromkeys(label[0] for label in labels))
            columns = list(dict.fromkeys(label[1] for label in labels))
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


def benchmark(calls: Mapping[Hashable, Callable[[], object]], *, warmup: int = 1,
              repeats: int = 3) -> Benchmark:
    """Run each of `calls`, in order, `warmup` times untimed and then `repeats` times timed, each
    run from the call until its output is ready (`jax.block_until_ready`), and return the runs'
    seconds and each call's last output (`Benchmark`). Raises `ValueError` for no timed run or a
    negative warm-up."""
    if repeats < 1 or warmup < 0:
        raise ValueError(f"a benchmark times at least one run after none or more warm-ups, not "
                         f"repeats={repeats} and warmup={warmup}")
    seconds: dict[Hashable, tuple[float, ...]] = {}
    outputs: dict[Hashable, object] = {}
    for label, call in calls.items():
        for _ in range(warmup):
            jax.block_until_ready(call())
        runs = []
        for _ in range(repeats):
            began = time.perf_counter()
            output = jax.block_until_ready(call())
            runs.append(time.perf_counter() - began)
        seconds[label], outputs[label] = tuple(runs), output
    return Benchmark(seconds, outputs)


__all__ = ["Benchmark", "benchmark"]
