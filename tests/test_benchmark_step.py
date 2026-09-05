"""tools/benchmark_step.py drives the Trainer's step surface directly.

The tool is the reproduction command behind every number in
docs/benchmarks.md and docs/performance.md, and it calls the trainer's
internals (`compile`, `shardings`, `device_mesh`, `DevicePrefetchIterator`)
rather than `fit`. A rename on that surface broke it once without any test
noticing (`batch_sharding` left the trainer in 128a903 and the tool kept
calling it), so one cpu-smoke case runs here, end to end, on every suite run.
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
