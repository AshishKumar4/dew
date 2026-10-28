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

import jax
import jax.numpy as jnp
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


def peak_row(tmp_path, platform: str, *arguments: str, cases: str | None = None,
             settings: dict[str, str] | None = None) -> dict:
    """The causal_transformer smoke case's row, or `cases`', the tool run as a person runs it on one
    device of `platform` with `arguments` and the environment's `settings`, its peak measured in a
    process of the case's own."""
    out = tmp_path / "rows.json"
    environment = {name: value for name, value in os.environ.items() if name != "XLA_FLAGS"}
    environment |= settings or {}
    tool = str(REPO_ROOT / "tools/benchmark_step.py")
    named = (["--cases", cases] if cases is not None else
             ["--preset", "cpu-smoke", "--architectures", "causal_transformer"])
    done = subprocess.run([sys.executable, tool, *named, "--warmup", "1", "--steps", "2",
                           "--dtype", "float32", "--mesh", "{}", "--json-out", str(out), *arguments],
                          env={**environment, "JAX_PLATFORMS": platform,
                               "PYTHONPATH": os.pathsep.join((str(REPO_ROOT / "src"), str(tmp_path)))},
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
    (row,) = json.loads(out.read_text())
    return row


def test_a_cases_peak_is_read_in_a_process_of_its_own_inside_the_cases_context(tmp_path):
    """The tool's peak memory comes from a process of the case's own that loaded every program from a cache
    and compiled nothing (a compile there refuses the run), one high-water a device, the bytes XLA planned
    beside it; that process runs inside the context a diagnostic names (`--within`), as the case does."""
    (tmp_path / "marking.py").write_text(
        "import contextlib, os\n"
        "@contextlib.contextmanager\n"
        "def marked():\n"
        f"    with open({str(tmp_path / 'entered')!r}, 'a') as file: file.write(f'{{os.getpid()}}\\n')\n"
        "    yield\n")
    row = peak_row(tmp_path, "cpu", "--within", "marking:marked")
    assert row["devices"] == 1 and len(row["peak_bytes_by_device"]) == 1 and row["planned_device_bytes"] > 0
    assert len(set((tmp_path / "entered").read_text().split())) == 3, "this process and both peak processes"


def test_a_released_decoders_peak_is_read_in_a_process_that_resolves_its_family(tmp_path):
    """A case naming a released decoder (`Case.family`) hands its peak process the case as it was
    named, which resolves the family itself: the resolved config holds native records JSON does not
    carry, and the run measured nothing where it tried to (row 1308, TL-h100-dory4b-1308)."""
    cases = json.dumps([{"architecture": "causal_transformer", "family": "dory-tiny", "batch_size": 8,
                         "seq_len": 32, "optimizer": {"state_dtype": "bfloat16"}}])
    row = peak_row(tmp_path, "cpu", cases=cases)
    assert row["finite"] and len(row["peak_bytes_by_device"]) == 1 and row["planned_device_bytes"] > 0


def test_a_cache_its_caller_names_keeps_what_the_step_compiled_and_a_run_alone_compiles_cold(tmp_path):
    """perf_gate names each tree's compile cache, so its later rounds load what its first compiled: the step
    compiled into it; run alone, the tool compiles cold, writing nothing to Dew's own cache (row 1577)."""
    named, own = tmp_path / "tree-cache", tmp_path / "home-cache"
    peak_row(tmp_path, "cpu", settings={
        "JAX_COMPILATION_CACHE_DIR": str(named), "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1", "XDG_CACHE_HOME": str(own)})
    assert [entry for entry in named.iterdir() if entry.name.startswith("jit_")]
    peak_row(tmp_path, "cpu", settings={"XDG_CACHE_HOME": str(own)})
    assert not list(own.rglob("jit_*"))


@pytest.mark.gpu(reason="a GPU's allocator keeps the high-water the measured peak reads")
@pytest.mark.mesh(devices=1)
def test_a_cases_measured_peak_holds_at_least_the_bytes_xla_planned(tmp_path):
    """On a GPU the measured peak is each device's allocator high-water over the case's run, its native
    workspaces and its profiled steps included (their traces apart): at least what XLA planned."""
    row = peak_row(tmp_path, "cuda", "--profile-dir", str(tmp_path / "traces"), "--profile-steps", "1")
    assert row["peak_device_bytes"] == row["peak_bytes_by_device"][0] >= row["planned_device_bytes"] > 0
    assert any((tmp_path / "traces/peak-run").rglob("*.xplane.pb"))


def test_a_released_decoders_case_trains_its_committed_shape_under_the_optimizer_it_names(monkeypatch):
    """A case naming a released decoder (`Case.family`) trains the model its committed config
    translates to, as a pretrained load reads it, under the optimizer it names (`Case.optimizer`):
    a looped nGPT decoder's, under AdamW with its moments in bf16, a finite step through the
    trainer."""
    tool = load("benchmark_step")
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    import benchmark_models as models
    from benchmark_cases import Case

    config = tool.BenchmarkConfig(
        cases=[Case("causal_transformer", family="dory-tiny", batch_size=8, seq_len=32,
                    optimizer={"state_dtype": "bfloat16"})], warmup=1, steps=2, dtype="float32")
    (case,) = tool.build_cases(config)
    released = json.loads((REPO_ROOT / "tests/fixtures/hf/dory-tiny/config.json").read_text())

    trainer = models.build_trainer(case)
    model = trainer.objective.model
    assert (model.num_layers, model.loop.steps) == (released["num_hidden_layers"],
                                                    released["n_recurrent_loops"])
    moments = trainer.optimizer.init({"kernel": jnp.zeros((4, 4), jnp.float32)})
    assert jnp.dtype(jnp.bfloat16) in {leaf.dtype for leaf in jax.tree.leaves(moments)}
    row = tool.measure(case, config, None)
    assert row["finite"] and np.isfinite(row["loss"])
