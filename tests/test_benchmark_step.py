"""tools/benchmark_step.py drives the Trainer's step surface directly.

The tool is the reproduction command behind every number in
docs/benchmarks.md and docs/performance.md, and it calls the trainer's
internals (`compile`, `shardings`, `device_mesh`, `DevicePrefetchIterator`)
directly, without `fit`, so every cpu-smoke case runs here, end to end, on
every suite run.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from tool_support import REPO_ROOT, load

from tools.benchmark_cases import cpu_smoke_cases

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh


@pytest.mark.parametrize("architecture, fixed_batch", [
    *[(case.architecture, False) for case in cpu_smoke_cases()], ("causal_transformer", True)])
def test_a_cpu_smoke_case_measures_a_finite_step_through_the_trainer(architecture, fixed_batch, monkeypatch):
    """The tool's row is the trainer's own compiled step run for real: a
    finite loss out of it, timed over the steps asked for, for every case of
    the preset, and whether each step reads a fresh placement or reuses one.
    Its MFU numerator is the objective's training FLOPs on the case's batch,
    the trainer's train/mfu count, or None where the model has none."""
    from dew.nn.flops import MissingFlopsCount

    tool = load("benchmark_step")
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    import benchmark_models as models  # the objective and batch the tool builds its step from
    config = tool.BenchmarkConfig(preset='cpu-smoke', architectures=[architecture],
                                  warmup=1, steps=2, dtype='float32', fixed_batch=fixed_batch)
    (case,) = tool.build_cases(config)

    row = tool.measure(case, config, None)

    assert row["finite"] and np.isfinite(row["loss"])
    assert row["measured_steps"] == 2 and row["fixed_batch"] is fixed_batch
    assert row["ms_per_step"] > 0 and row["p50_ms"] > 0
    try:
        counted = models.build_objective(case).training_flops(None, models.global_batch(case))
    except MissingFlopsCount:
        counted = None
    assert row["training_flops"] == counted


def test_a_cases_peak_is_its_devices_measured_high_water_at_least_the_bytes_xla_planned(tmp_path):
    """The tool's peak memory is each device's allocator high-water over the case's steps, read in a
    process of the case's own that loaded every program from a cache and compiled nothing; it holds at
    least what XLA planned for the step, which stands beside it. Run as a person runs it, on one CPU
    device, whose allocator keeps a high-water (the eight simulated ones keep none)."""
    out = tmp_path / "rows.json"
    environment = {name: value for name, value in os.environ.items() if name != "XLA_FLAGS"}
    subprocess.run([sys.executable, str(REPO_ROOT / "tools/benchmark_step.py"), "--preset", "cpu-smoke",
                    "--architectures", "causal_transformer", "--warmup", "1", "--steps", "2",
                    "--dtype", "float32", "--mesh", "{}", "--json-out", str(out)],
                   env={**environment, "JAX_PLATFORMS": "cpu"}, check=True, capture_output=True, timeout=900)
    (row,) = json.loads(out.read_text())
    assert row["devices"] == 1 and len(row["peak_bytes_by_device"]) == 1
    assert row["peak_device_bytes"] == row["peak_bytes_by_device"][0] >= row["planned_device_bytes"] > 0
