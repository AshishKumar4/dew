"""tools/benchmark_step.py drives the Trainer's step surface directly.

The tool is the reproduction command behind every number in
docs/benchmarks.md and docs/performance.md, and it calls the trainer's
internals (`compile`, `shardings`, `device_mesh`, `DevicePrefetchIterator`)
directly, without `fit`, so one cpu-smoke case runs here, end to end, on
every suite run.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]


def _benchmark_step():
    spec = importlib.util.spec_from_file_location(
        "benchmark_step_under_test", REPO_ROOT / "tools" / "benchmark_step.py")
    if spec is None or spec.loader is None:
        raise ImportError("tools/benchmark_step.py is not importable as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_smoke_case_measures_a_finite_step_through_the_trainer():
    """The tool's row is the trainer's own compiled step run for real: a
    finite loss out of it, timed over the steps asked for."""
    tool = _benchmark_step()
    config = tool.BenchmarkConfig(preset='cpu-smoke', architectures=['causal_transformer'],
                                  warmup=1, steps=2, dtype='float32')
    (case,) = tool.build_cases(config)

    row = tool.measure(case, config)

    assert row["finite"] and np.isfinite(row["loss"])
    assert row["measured_steps"] == 2
    assert row["ms_per_step"] > 0 and row["p50_ms"] > 0


@pytest.mark.parametrize("intervals,busy,window", [
    ([(0, 10), (2, 3)], 10, 10),
    ([(0, 4), (2, 6)], 6, 6),
    ([(0, 2), (4, 6)], 4, 6),
])
def test_device_timeline_covers_nested_and_disjoint_kernels(
    tmp_path, monkeypatch, intervals, busy, window,
):
    from types import SimpleNamespace
    import jax.profiler

    trace = tmp_path / "trace.xplane.pb"
    trace.touch()
    events = [SimpleNamespace(name="kernel", start_ns=start * 1_000_000,
                              end_ns=end * 1_000_000) for start, end in intervals]
    profile = SimpleNamespace(planes=[SimpleNamespace(
        name="/device:GPU:0", lines=[SimpleNamespace(events=events)])])
    monkeypatch.setattr(jax.profiler, "ProfileData", SimpleNamespace(
        from_file=lambda path: profile))

    row = _benchmark_step().device_timeline(str(tmp_path), steps=1)
    assert row["device_busy_ms_per_step"] == pytest.approx(busy)
    assert row["device_window_ms_per_step"] == pytest.approx(window)
    assert row["device_busy_percent"] == pytest.approx(100 * busy / window)
