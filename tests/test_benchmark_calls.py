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
    measured = benchmark({("euler", "4 steps"): lambda: Slow(0.0), ("euler", "8 steps"): lambda: Slow(0.0),
                          ("heun", "8 steps"): lambda: Slow(0.011)}, warmup=0, repeats=1)
    lines = measured.table().splitlines()
    assert lines[0].split() == ["4", "steps", "8", "steps"]
    assert lines[1].split()[0] == "euler" and lines[1].count("ms") == 2
    # Heun ran no 4 steps: its cell there is empty, and its 8 steps are at least 11 ms.
    assert lines[2].split()[0] == "heun" and lines[2].count("ms") == 1
    assert float(lines[2].split()[1]) >= 11.0
    assert "<table>" in measured._repr_html_() and "heun" in measured._repr_html_()
    single = benchmark({"slow": lambda: Slow(0.0)}, warmup=0, repeats=1)
    assert single.table().splitlines()[0].split() == ["median"]
