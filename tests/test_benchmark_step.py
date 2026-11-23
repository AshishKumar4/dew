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
from pathlib import Path

import numpy as np
import pytest
from lane_environment import outside_suite
from tool_support import REPO_ROOT, load

from tools.benchmark_cases import cpu_smoke_cases

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh


FIXTURES = REPO_ROOT / "tests" / "fixtures" / "hf"
"""Tiny checkpoints of registered decoder families, each a directory holding its config.json."""

TRANSLATED = ["qwen35-tiny", "nemotron-h-tiny", "mamba2-tiny"]
"""Families whose translated configs hold records of their layer kinds (Qwen3.5's,
`dew.interop.config_records.NativeFields`) or values already built (Nemotron-H's and Mamba-2's mixers)."""


@pytest.mark.parametrize("name", TRANSLATED)
def test_a_case_of_a_translated_config_is_read_back_as_the_case_it_builds(name):
    """A case whose config is a family's translated config is written as the process measuring its peak reads
    it (`cases_to_json`, `cases_from_json`) and builds the model the case builds: its records and built values
    written as the records of the fields the model declares (rows 3742, 3754)."""
    from dew.interop.hf_decoders import translate_config
    from dew.registry import models, to_record

    step = load("benchmark_step")
    config = translate_config(json.loads((FIXTURES / name / "config.json").read_text()))
    case = step.Case("causal_transformer", dict(config), dtype="float32", batch_size=2, seq_len=8)
    [read] = step.cases_from_json(step.cases_to_json([case]))
    assert (to_record(models.build(read.architecture, **read.config), object)
            == to_record(models.build(case.architecture, **case.config), object))


@pytest.mark.parametrize("named", ["directory", "config.json"])
@pytest.mark.parametrize("name", ["qwen35-tiny", "qwen35-native-tiny"])
def test_a_case_names_a_checkpoints_config_on_the_command_line(name, named):
    """`--cases` names a checkpoint's config (`hf_config`, its directory or its config.json): a decoder's own,
    or a multimodal wrapper's whose language model a text-only load reads (`decoder_config`), which the case's
    model is built from as the family translates it, the case's own `config` fields over the translated ones
    (row 3755)."""
    from dew.interop.hf_decoders import decoder_config, translate_config
    from dew.registry import models, to_record

    step = load("benchmark_step")
    models_module = load("benchmark_models")
    directory = FIXTURES / name
    checkpoint = directory if named == "directory" else directory / "config.json"
    [case] = step.cases_from_json(json.dumps([{"architecture": "causal_transformer",
                                              "hf_config": str(checkpoint), "config": {"max_seq_len": 128},
                                              "dtype": "float32", "batch_size": 2, "seq_len": 8}]))
    translated = translate_config(decoder_config(json.loads((directory / "config.json").read_text())))
    assert translated["max_seq_len"] != 128
    expected = models.build("causal_transformer", **{**translated, "max_seq_len": 128}, dtype="float32",
                            attention_impl="xla")
    built = models_module.build_objective(case, "xla").program_key()[0].module
    assert to_record(built, object) == to_record(expected, object)


def test_a_case_names_a_quantized_wrapper_whose_selection_reads_its_tensors_names(tmp_path):
    """A wrapper checkpoint whose quantized selection is a pattern over its modules (compressed-tensors'
    `re:`) is read with its tensors' names, from their headers, as a text-only load reads it
    (`decoder_config`'s inventory): the case builds the decoder that selection, renamed to the text
    checkpoint's modules, gives (row 3777)."""
    import shutil

    from dew.interop.hf_decoders import decoder_config, translate_config
    from dew.interop.safetensors_io import read_file
    from dew.registry import models, to_record

    source = FIXTURES / "qwen35-native-tiny"
    wrapper = Path(shutil.copytree(source, tmp_path / "wrapper"))
    config = json.loads((source / "config.json").read_text())
    quantization = {"quant_method": "compressed-tensors", "format": "float-quantized",
                    "config_groups": {"fp8": {
                        "targets": ["Linear", "re:.*layers\\.0\\.mlp\\.(up|down)_proj$"],
                        "weights": {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False}}},
                    "ignore": ["lm_head", "model.language_model.layers.1"]}
    quantized = {**config, "quantization_config": quantization}
    (wrapper / "config.json").write_text(json.dumps(quantized))
    names, _ = read_file(source / "model.safetensors")
    case = load("benchmark_step").Case("causal_transformer", hf_config=str(wrapper), dtype="float32",
                                       batch_size=2, seq_len=8)
    expected = translate_config(decoder_config(quantized, lambda: names.keys()))
    assert (to_record(models.build("causal_transformer", **case.trunk_fields()), object)
            == to_record(models.build("causal_transformer", **expected), object))


def test_a_case_naming_another_file_than_a_checkpoints_config_is_refused(tmp_path):
    """A checkpoint is named by its directory or its config.json, which a load reads: a case naming another
    file of a directory (a tweaked config beside the checkpoint's own) is refused rather than measured as the
    directory's config.json."""
    step = load("benchmark_step")
    config = json.loads((FIXTURES / "qwen35-tiny" / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tweaked.json").write_text(json.dumps({**config, "num_hidden_layers": 1}))
    case = step.Case("causal_transformer", hf_config=str(tmp_path / "tweaked.json"), dtype="float32",
                     batch_size=2, seq_len=8)
    with pytest.raises(ValueError):
        case.trunk_fields()


def test_a_case_naming_a_checkpoint_is_identified_by_the_fields_it_holds_now(tmp_path):
    """What keys a cache of a case's results (`Case.resolved`, layout_parity's references) is the checkpoint's
    translated fields, not the path naming it: the same file with a changed norm epsilon is another case."""
    step = load("benchmark_step")
    config = json.loads((FIXTURES / "qwen35-tiny" / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config))
    case = step.Case("causal_transformer", hf_config=str(tmp_path), dtype="float32", batch_size=2, seq_len=8)
    before = case.resolved().record()
    (tmp_path / "config.json").write_text(json.dumps({**config, "rms_norm_eps": config["rms_norm_eps"] * 10}))
    assert case.resolved().record() != before


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

    row = tool.measure(case, config)

    assert row["finite"] and np.isfinite(row["loss"])
    assert row["measured_steps"] == 2 and row["fixed_batch"] is fixed_batch
    assert row["ms_per_step"] > 0 and row["p50_ms"] > 0
    # The timed window's parts, each read with its host dispatch, and every synced step's own time.
    window, synced = row["window_ms_series"], row["synced_ms_series"]
    assert len(window) == len(row["window_dispatch_ms_series"]) == len(synced) == 2
    assert all(ms > 0 for ms in (*window, *synced))
    try:
        counted = models.build_objective(case).training_flops(None, models.global_batch(case))
    except MissingFlopsCount:
        counted = None
    assert row["training_flops"] == counted
    # The CPU step runs no launch its count leaves out, nor a loop counted at its bound: its count is the
    # step's own; one that does is reported (row 3023).
    assert (row["unsized_kernels"], row["utilization_bound"]) == (0, "")


def test_the_tools_step_is_the_program_a_runs_first_step_compiles(monkeypatch):
    """The step the tool times (`benchmark_models.compiled_step`) is the program a run's first step compiles:
    lowered for a GPU (`lower_for`), the tool's compile and a fit over the same batches lower one program
    (`dew.nn.kernels.options.program_digest`), the tool handing its compile each batch as fit hands its
    step, its rows marked, its work counted and its host fields apart (`dew.training.ahead.step_batch`, row
    3700)."""
    from dew.data.dataset import Loader
    from dew.nn.kernels.options import program_digest
    from dew.training._lowering import Lowered

    tool = load("benchmark_step")
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    import benchmark_models as models
    config = tool.BenchmarkConfig(preset='cpu-smoke', architectures=['causal_transformer'], warmup=1, steps=1,
                                  dtype='float32')
    (case,) = tool.build_cases(config)

    def lowered(run) -> str:
        trainer = models.build_trainer(case)
        trainer.lower_for = "sm90"
        with pytest.raises(Lowered) as raised:
            run(trainer)
        return program_digest(raised.value.lowered)

    def timed(trainer):
        with models.compiled_step(case, trainer):
            pass

    def fitted(trainer):
        data = Loader(lambda partition: models.batches(case, trainer.device_mesh), None,
                      batch=case.batch_size, records=None)
        trainer.fit(data, steps=1)

    assert lowered(timed) == lowered(fitted)


def test_a_prepare_that_cannot_compile_a_case_ahead_fails_and_a_run_handed_its_directory_refuses_one_missing(
        tmp_path, monkeypatch):
    """`--compile-ahead` keeps each case it compiled and fails where this host refused one (`Unlowerable`),
    so the job's prepare fails; and a run handed a directory of steps compiled ahead refuses a case the
    directory lacks, never compiling its own on the GPU in its place (row 10796: the prepare exited 0 without
    the case's step, and the run compiled it)."""
    from dew.nn.kernels.selection import Unlowerable

    tool = load("benchmark_step")
    ahead = tmp_path / "ahead"
    config = tool.BenchmarkConfig(preset='cpu-smoke', architectures=['causal_transformer', 'simple_dit'],
                                  warmup=1, steps=1, dtype='float32', compile_ahead=str(ahead),
                                  generation="sm90", capacity_gib=80)
    compiled, refused = tool.build_cases(config)

    class Ahead:
        def save(self, path):
            Path(path).write_text("a step")

    def ahead_of(case, *_):
        if case.architecture == refused.architecture:
            raise Unlowerable("attention", "flash", "library unavailable")
        return Ahead()

    monkeypatch.setattr(tool, "ahead_of", ahead_of)
    with pytest.raises(SystemExit):
        tool.main(config)
    assert sorted(path.name for path in ahead.iterdir()) == [tool.ahead_name(compiled)]
    # A run of the refused case, handed the directory, refuses it before it builds a trainer.
    monkeypatch.setattr(tool, "build_trainer", lambda *_, **__: pytest.fail("compiled in its place"))
    run = tool.BenchmarkConfig(preset='cpu-smoke', architectures=[refused.architecture], warmup=1, steps=1,
                               dtype='float32', ahead=str(ahead))
    with pytest.raises(SystemExit):
        tool.measure(*tool.build_cases(run), run)


def test_a_steps_count_reads_as_the_bound_it_is():
    """A row's FLOPs and utilization show as the bound their count is (`StepFlops.bound`, `BOUND_MARKS`), as
    the trainer names its train/hfu: at least the value where a launch's work went uncounted (row 3023), at
    most it where a loop was counted at its stated bound, about it where both (train 1025's F3: the table
    marked a bound from the uncounted launches alone, an upper bound shown as the count), the count itself
    as it is; the MFU, the objective's own count, never marked."""
    from dew.telemetry.instrumentation import BOUND_MARKS

    tool = load("benchmark_step")
    row = {"architecture": "causal_transformer", "flops_per_step": 2e9, "utilization": 0.25, "mfu": 0.2}
    for bound, mark in BOUND_MARKS.items():
        cells = tool.format_table([{**row, "utilization_bound": bound}]).splitlines()[-1].split()
        assert [cell for cell in cells if cell.endswith(("2.0", "25.0", "20.0"))] == [
            f"{mark}2.0", f"{mark}25.0", "20.0"]


SMOKE_CASE = ("--preset", "cpu-smoke", "--architectures", "causal_transformer", "--dtype", "float32",
              "--mesh", "{}")
"""The causal_transformer smoke case's arguments to tools/benchmark_step.py: its row's (`peak_row`) and a
prepare's that compiles the same case's program ahead (tests/test_attribute_step.py)."""


def peak_row(tmp_path, platform: str, *arguments: str, settings: dict[str, str] | None = None) -> dict:
    """The causal_transformer smoke case's row, the tool run as a person runs it on one device of
    `platform` with `arguments` and the environment's `settings`, the case measured in processes of its
    own."""
    out = tmp_path / "rows.json"
    environment = outside_suite(os.environ) | (settings or {})
    tool = str(REPO_ROOT / "tools/benchmark_step.py")
    done = subprocess.run([sys.executable, tool, *SMOKE_CASE, "--warmup", "1", "--steps", "2",
                           "--json-out", str(out), *arguments],
                          env={**environment, "JAX_PLATFORMS": platform,
                               "PYTHONPATH": os.pathsep.join((str(REPO_ROOT / "src"), str(tmp_path)))},
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
    (row,) = json.loads(out.read_text())
    return row


def test_a_case_is_measured_in_processes_of_its_own_inside_the_cases_context_alone_and_cold(tmp_path):
    """The tool's row comes from processes of the case's own: the first compiles, its compile the row's,
    the second loads every program from its cache and compiles nothing (a compile there refuses the run),
    timing the step and reading one high-water a device, the bytes XLA planned beside it; both run inside
    the context a diagnostic names (`--within`) and the tool's own process runs no case. Run alone, the
    tool compiles cold, writing nothing to Dew's own cache (row 1577)."""
    own = tmp_path / "home-cache"
    (tmp_path / "marking.py").write_text(
        "import contextlib, os\n"
        "@contextlib.contextmanager\n"
        "def marked():\n"
        f"    with open({str(tmp_path / 'entered')!r}, 'a') as file: file.write(f'{{os.getpid()}}\\n')\n"
        "    yield\n")
    row = peak_row(tmp_path, "cpu", "--within", "marking:marked", settings={"XDG_CACHE_HOME": str(own)})
    assert row["devices"] == 1 and len(row["peak_bytes_by_device"]) == 1 and row["planned_device_bytes"] > 0
    assert row["ms_per_step"] > 0 and row["compile_seconds"] > 0 and row["ahead_step"] is False
    assert row["compiled_while_measured"] == 0 and row["compile_cached"] is False
    assert len(set((tmp_path / "entered").read_text().split())) == 2, "the case's two processes alone"
    assert not list(own.rglob("jit_*"))


def test_a_case_whose_measured_run_compiled_keeps_its_timing_with_no_peak(tmp_path):
    """A program the measured run did not load from its cache (its cache switched off there alone) is
    compiled there: its compile joined the high-water, so the row has no peak and counts the compiles, its
    timing kept."""
    (tmp_path / "uncached.py").write_text(
        "import contextlib, os, jax\n"
        "@contextlib.contextmanager\n"
        "def uncached():\n"
        "    if os.environ.get('DEW_PEAK_STAGE') == 'measure':\n"
        "        jax.config.update('jax_enable_compilation_cache', False)\n"
        "    yield\n")
    row = peak_row(tmp_path, "cpu", "--within", "uncached:uncached")
    assert row["compiled_while_measured"] >= 1 and row["peak_device_bytes"] is None
    assert row["peak_bytes_by_device"] is None and row["ms_per_step"] > 0 and row["finite"]


def test_a_cache_its_caller_names_keeps_what_the_step_compiled(tmp_path):
    """perf_gate names each arm's case cache (tools/peak_memory.py's PEAK_CACHE), so its later rounds load
    what its first compiled: the step compiled into it (run alone, the tool writes nothing to Dew's own
    cache: `test_a_case_is_measured_in_processes_of_its_own_inside_the_cases_context_alone_and_cold`)."""
    named, own = tmp_path / "arm-cache", tmp_path / "home-cache"
    peak_row(tmp_path, "cpu", settings={"DEW_PEAK_CACHE": str(named), "XDG_CACHE_HOME": str(own)})
    assert [entry for entry in named.iterdir() if entry.name.startswith("jit_")]


def a_cases_measured_peak_holds_at_least_the_bytes_xla_planned(tmp_path):
    """On a GPU the measured peak is each device's allocator high-water over the case's run, its native
    workspaces and its profiled steps included (their traces apart): at least what XLA planned."""
    row = peak_row(tmp_path, "cuda", "--profile-dir", str(tmp_path / "traces"), "--profile-steps", "1")
    assert row["peak_device_bytes"] == row["peak_bytes_by_device"][0] >= row["planned_device_bytes"] > 0
    # The measuring process profiles; the one that filled its cache does not.
    (traced,) = (tmp_path / "traces").iterdir()
    assert any(traced.rglob("*.xplane.pb")) and "device_busy_percent" in row


@pytest.mark.parametrize(("steps", "parts"), [(30, [5] * 6), (2, [1, 1]), (7, [2, 1, 1, 1, 1, 1]), (1, [1])])
def test_a_timed_window_is_cut_into_parts_that_hold_every_step(steps, parts):
    """A timed window of `steps` is cut into at most six parts, none empty, holding every step once."""
    assert load("benchmark_step").window_parts(steps) == parts


def test_the_tuning_a_process_compiles_with_is_named_by_its_results_digest(tmp_path, monkeypatch):
    """The per-fusion autotuning results XLA_FLAGS names are counted and digested by their names and contents:
    a result added or changed changes the digest; no directory named, no digest."""
    tool = load("benchmark_step")
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    assert tool.tuning_digest() is None
    monkeypatch.setenv("XLA_FLAGS", f"--xla_gpu_per_fusion_autotune_cache_dir={tmp_path}")
    (tmp_path / "a.textproto").write_text("pick a")
    first = tool.tuning_digest()
    assert first["results"] == 1 and first["directory"] == str(tmp_path)
    (tmp_path / "a.textproto").write_text("pick b")
    changed = tool.tuning_digest()
    (tmp_path / "b.textproto").write_text("pick c")
    assert len({first["sha256"], changed["sha256"], tool.tuning_digest()["sha256"]}) == 3


def test_a_device_samplers_summary_holds_each_gpus_span_and_every_clock_event_reason_any_sample_held():
    """The GPUs sampled under a window's load summarized per GPU: the samples' count, each reading's least
    and greatest, and the clock event reasons any sample held, or'd, GpuIdle (between steps) not among
    them."""
    sampler = load("benchmark_step").DeviceSampler()
    import datetime

    sampler.lines = ("2026/11/14 16:00:00.000, 0, 1410, 144.0, 45, 0x0000000000000001\n"
                     "2026/11/14 16:00:01.000, 0, 1980, 410.5, 51, 0x0000000000000001\n"
                     "2026/11/14 16:00:01.250, 0, 1755, 690.25, 58, 0x0000000000000004\n"
                     "2026/11/14 16:00:01.500, 0, 1830, 650.0, 57, 0x0000000000000000\n")
    window = (datetime.datetime(2026, 11, 14, 16, 0, 0, 500000), datetime.datetime(2026, 11, 14, 16, 0, 2))
    # The idle sample before the window is not among its samples.
    assert sampler.summary(window) == {"0": {"samples": 3, "sm_mhz": [1755.0, 1980.0],
                                             "power_w": [410.5, 690.25], "temperature_c": [51.0, 58.0],
                                             "event_reasons": "0x4"}}
    assert sampler.summary()["0"]["samples"] == 4
    sampler.lines = ""
    assert sampler.summary(window) is None


def test_a_windows_parts_are_stamped_as_each_becomes_ready_whether_dispatch_or_device_is_slower():
    """Each part's stamp is its last step's readiness, read by a thread as the part is dispatched: a step
    whose dispatch is the slower (its work done before the next is dispatched) reads each part's own time,
    none 0, where awaiting the parts after the whole dispatch reads the first part as all of it."""
    import time

    tool = load("benchmark_step")

    def step(state):
        time.sleep(0.02)  # the host's dispatch, slower than the device's work
        return state + 1, np.float32(state), True

    *_, parts = tool.timed_parts(step, 0, [2, 2, 2])
    assert [size for size, _, _ in parts] == [2, 2, 2]
    assert all(seconds >= 0.03 for _, seconds, _ in parts), parts
    # The host's dispatch was the slow part: each part's dispatch takes its two steps' sleeps.
    assert all(sent >= 0.03 for _, _, sent in parts), parts


def test_a_window_reads_invalid_where_any_of_its_steps_went_non_finite_not_only_its_last():
    """A compiled step's windows keep every step's loss and finiteness on the device and read them once the
    window is over: one step that went non-finite (its own flag, or its loss) makes the window's count
    nonzero, though the steps after it recovered; the count is of the steps since the last read (audit,
    2026-11-14: only the last step's finiteness was kept, so a NaN mid-window was timed)."""
    import jax.numpy as jnp

    compiled = load("benchmark_models").CompiledStep(None, None, 0.0)
    for loss, finite in ((1.0, True), (float("nan"), True), (2.0, False), (0.5, True)):
        compiled.taken.append((jnp.float32(loss), jnp.asarray(finite)))
    assert compiled.nonfinite() == 2
    compiled.taken.append((jnp.float32(0.4), jnp.asarray(1, dtype=bool)))
    assert compiled.nonfinite() == 0


def test_a_benchmarks_rows_reach_a_tracker_after_its_cases_one_step_a_case(tmp_path):
    """`--tracked` logs each case's row, its numbers by its architecture, into a tracker once every case is
    measured, one step a case in order: an operational run's MFU and step time where a training run's land
    (row 285); a row's text and flags are no metric, and a row whose steps went non-finite is logged not at
    all (row 11745)."""
    from dew.training.tracker import LocalTracker

    step = load("benchmark_step")
    rows = [{"architecture": "simple_dit", "ms_per_step": 12.5, "mfu": 0.31, "device_kind": "cpu",
             "fixed_batch": False, "finite": True},
            {"architecture": "unet", "ms_per_step": 30.0, "mfu": None, "devices": 1, "finite": True},
            {"architecture": "dit", "ms_per_step": 8.0, "mfu": 0.5, "finite": False}]
    step.tracked(rows, LocalTracker(tmp_path))
    assert LocalTracker(tmp_path).scalars() == [
        (0, {"simple_dit/ms_per_step": 12.5, "simple_dit/mfu": 0.31}),
        (1, {"unet/ms_per_step": 30.0, "unet/devices": 1.0})]


@pytest.mark.parametrize("mesh, split", [({}, 1), ({"fsdp": 2}, 1), ({"stage": 2}, "auto"),
                                         ({"stage": 2, "microbatches": 4}, 4)],
                         ids=["single", "fsdp", "staged", "staged-microbatches"])
def test_a_cases_step_is_read_whole_and_its_pipeline_in_the_microbatches_it_names(mesh, split):
    """A case's step is read whole (`FitPolicy(split=1)`), as the benchmark has always timed it, the remat
    ladder its one way to fit, never split behind the timing's back; under pipeline stages it is read in the
    microbatches its mesh record names, the plan's where it names none; microbatches where it stages none
    are refused (review of API01: every case's split was left to the plan)."""
    fields = load("benchmark_models").trainer_mesh(mesh)
    assert fields["fit"].split == split
    with pytest.raises(ValueError, match="stages none"):
        load("benchmark_models").trainer_mesh({"microbatches": 4})
