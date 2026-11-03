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


TINY_QWEN35 = {
    "model_type": "qwen3_5_text", "attention_bias": False, "attention_dropout": 0.0, "attn_output_gate": True,
    "dtype": "float32", "full_attention_interval": 4, "head_dim": 16, "hidden_act": "silu", "hidden_size": 64,
    "intermediate_size": 128, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 16,
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_value_head_dim": 16, "max_position_embeddings": 64,
    "mlp_only_layers": [], "mtp_num_hidden_layers": 0, "num_attention_heads": 4, "num_hidden_layers": 4,
    "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "tie_word_embeddings": True, "vocab_size": 128,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [1, 1, 0], "rope_type": "default",
                        "rope_theta": 10000000, "partial_rotary_factor": 0.25}}
"""Qwen3.5's text config at a test's size: a hybrid of gated delta nets and attention, whose translation holds
records of its layer kinds (`dew.interop.config_records.NativeFields`)."""


def test_a_case_of_a_translated_config_is_read_back_as_the_case_it_builds():
    """A case whose config is a family's translated config is written as the process measuring its peak reads
    it (`cases_to_json`, `cases_from_json`) and builds the model the case builds: its records written as the
    fields they state, which the model's declared field types rebuild (row 3742)."""
    from dew.interop.hf_decoders import translate_config
    from dew.registry import models, to_record

    step = load("benchmark_step")
    case = step.Case("causal_transformer", dict(translate_config(TINY_QWEN35)), dtype="float32", batch_size=2,
                     seq_len=8)
    [read] = step.cases_from_json(step.cases_to_json([case]))
    assert (to_record(models.build(read.architecture, **read.config), object)
            == to_record(models.build(case.architecture, **case.config), object))


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
    # The CPU step runs no launch its count leaves out; one that does is reported (row 3023).
    assert row["unsized_kernels"] == 0


def test_a_steps_count_some_launches_work_is_missing_from_reads_as_a_lower_bound():
    """A row whose step runs launches no count sizes (`StepFlops.unsized`) shows its FLOPs and
    utilization as the lower bounds they are, as the trainer names them train/hfu_lower_bound; a
    complete count shows as it is (row 3023)."""
    tool = load("benchmark_step")
    row = {"architecture": "causal_transformer", "flops_per_step": 2e9, "utilization": 0.25, "mfu": 0.2}
    counted, uncounted = (tool.format_table([{**row, "unsized_kernels": unsized}]).splitlines()[-1].split()
                          for unsized in (0.0, 2.0))
    assert [cell for cell in counted if cell in ("2.0", "25.0", "20.0")] == ["2.0", "25.0", "20.0"]
    assert [cell for cell in uncounted if cell.endswith(("2.0", "25.0", "20.0"))] == [
        ">=2.0", ">=25.0", "20.0"]


def peak_row(tmp_path, platform: str, *arguments: str, settings: dict[str, str] | None = None) -> dict:
    """The causal_transformer smoke case's row, the tool run as a person runs it on one device of
    `platform` with `arguments` and the environment's `settings`, its peak measured in a process of the
    case's own."""
    out = tmp_path / "rows.json"
    environment = {name: value for name, value in os.environ.items() if name != "XLA_FLAGS"}
    environment |= settings or {}
    tool = str(REPO_ROOT / "tools/benchmark_step.py")
    done = subprocess.run([sys.executable, tool, "--preset", "cpu-smoke",
                           "--architectures", "causal_transformer", "--warmup", "1", "--steps", "2",
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


def a_cases_measured_peak_holds_at_least_the_bytes_xla_planned(tmp_path):
    """On a GPU the measured peak is each device's allocator high-water over the case's run, its native
    workspaces and its profiled steps included (their traces apart): at least what XLA planned."""
    row = peak_row(tmp_path, "cuda", "--profile-dir", str(tmp_path / "traces"), "--profile-steps", "1")
    assert row["peak_device_bytes"] == row["peak_bytes_by_device"][0] >= row["planned_device_bytes"] > 0
    assert any((tmp_path / "traces/peak-run").rglob("*.xplane.pb"))
