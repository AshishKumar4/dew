"""`dew.telemetry.benchmark`: calls timed from the call until their output is ready, after warm-up
runs that compile what they trace, laid out by label as tutorial 04 lays out its samplers."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.telemetry.benchmark import benchmark


class Slow:
    """An output that is ready a known time after it is returned, as a dispatched computation is."""

    def __init__(self, seconds):
        self.seconds = seconds

    def block_until_ready(self):
        time.sleep(self.seconds)
        return self


def test_a_benchmark_times_each_call_until_its_output_is_ready_after_untimed_warm_ups():
    traced = []

    @jax.jit
    def step(x):
        traced.append(1)
        return jnp.cumsum(x) * 2

    x = jnp.arange(8.0)
    measured = benchmark({"cumsum": lambda: step(x), "slow": lambda: Slow(0.05)}, warmup=2, repeats=3)
    assert len(traced) == 1, "the warm-up compiles once and no timed run traces again"
    assert [len(measured.seconds[label]) for label in ("cumsum", "slow")] == [3, 3]
    assert min(measured.seconds["slow"]) >= 0.05, "the wait for the output is timed"
    np.testing.assert_array_equal(np.asarray(measured.outputs["cumsum"]), np.cumsum(np.arange(8.0)) * 2)
    assert measured.median("slow") == sorted(measured.seconds["slow"])[1]
    with pytest.raises(ValueError):
        benchmark({"cumsum": lambda: step(x)}, repeats=0)


def test_pair_labels_lay_out_as_rows_by_columns_in_one_unit():
    """Medians laid out by label: pairs as a grid of rows by columns, a pair that never ran left
    empty, and one unit a table, milliseconds until a median reaches ten seconds."""
    from dew.telemetry.benchmark import Benchmark

    measured = Benchmark({("euler", "4 steps"): (0.002, 0.001, 0.003), ("euler", "8 steps"): (0.004,),
                          ("heun", "8 steps"): (1.0115,)}, {})
    # Each column as wide as its widest cell, right-aligned, two spaces from the next.
    assert measured.table().splitlines() == [
        "       4 steps     8 steps",
        "euler   2.0 ms      4.0 ms",
        "heun            1,011.5 ms"]
    html = measured._repr_html_()
    assert "<th>heun</th>" in html and "1,011.5 ms" in html and html.count("<td") == 4
    slow = Benchmark({"sample": (12.0, 14.0, 13.0), "<script>": (1.0,)}, {})
    assert slow.table().splitlines() == ["          median", "sample    13.0 s", "<script>   1.0 s"]
    assert "&lt;script&gt;" in slow._repr_html_()
