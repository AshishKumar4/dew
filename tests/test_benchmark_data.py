"""tools/benchmark_data.py: per-step latencies without crashing on warmup=0.

`measure` walks the loader for warmup + steps batches and times the steps
after warmup. The count it pulls is the contract: a loader that ends early
is reported, not silently extrapolated.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _measure():
    spec = importlib.util.spec_from_file_location(
        "benchmark_data_under_test", REPO_ROOT / "tools" / "benchmark_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.measure


def _pulls(batches):
    seen = []

    def stream():
        for batch in batches:
            seen.append(batch)
            yield batch

    return stream(), seen


def test_measure_times_every_step_when_nothing_is_warmup():
    """Warmup 0 means every pulled batch is timed: the first step has no
    previous tick, and `last` was never set for it."""
    measure = _measure()
    stream, seen = _pulls(range(5))

    assert len(measure(stream, steps=3, warmup=0)) == 3
    assert seen == [0, 1, 2]


def test_measure_pulls_exactly_warmup_plus_steps_batches():
    """The loader is pulled warmup + steps times and no more: what is timed
    is what was read, and an endless stream is left alone after that."""
    measure = _measure()

    def endless():
        index = 0
        while True:
            yield index
            index += 1

    stream, seen = _pulls(endless())

    assert len(measure(stream, steps=4, warmup=2)) == 4
    assert seen == [0, 1, 2, 3, 4, 5]
