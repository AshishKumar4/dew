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


def test_the_tools_step_is_the_program_a_runs_first_step_compiles(monkeypatch):
    """The step the tool times (`benchmark_models.compiled_step`) is the program a run's first step compiles:
    lowered for a GPU (`lower_for`), the tool's compile and a fit over the same batches lower one program
    (`dew.training.ahead.program_of`), the tool handing its compile each batch as fit hands its step, its
    rows marked, its work counted and its host fields apart (`dew.training.ahead.step_batch`, row 3700)."""
    from dew.data.dataset import Loader
    from dew.training._lowering import Lowered
    from dew.training.ahead import program_of

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
        return program_of(raised.value.lowered)

    def timed(trainer):
        with models.compiled_step(case, trainer):
            pass

    def fitted(trainer):
        data = Loader(lambda partition: models.batches(case, trainer.device_mesh), None,
                      batch=case.batch_size, records=None)
        trainer.fit(data, steps=1)

    assert lowered(timed) == lowered(fitted)


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


def test_a_cases_peak_is_read_in_a_process_of_its_own_inside_the_cases_context_alone_and_cold(tmp_path):
    """The tool's peak memory comes from a process of the case's own that loaded every program from a cache
    and compiled nothing (a compile there refuses the run), one high-water a device, the bytes XLA planned
    beside it; that process runs inside the context a diagnostic names (`--within`), as the case does. Run
    alone, the tool compiles cold, writing nothing to Dew's own cache (row 1577)."""
    own = tmp_path / "home-cache"
    (tmp_path / "marking.py").write_text(
        "import contextlib, os\n"
        "@contextlib.contextmanager\n"
        "def marked():\n"
        f"    with open({str(tmp_path / 'entered')!r}, 'a') as file: file.write(f'{{os.getpid()}}\\n')\n"
        "    yield\n")
    row = peak_row(tmp_path, "cpu", "--within", "marking:marked", settings={"XDG_CACHE_HOME": str(own)})
    assert row["devices"] == 1 and len(row["peak_bytes_by_device"]) == 1 and row["planned_device_bytes"] > 0
    assert len(set((tmp_path / "entered").read_text().split())) == 3, "this process and both peak processes"
    assert not list(own.rglob("jit_*"))


def test_a_cache_its_caller_names_keeps_what_the_step_compiled(tmp_path):
    """perf_gate names each tree's compile cache, so its later rounds load what its first compiled: the step
    compiled into it (run alone, the tool writes nothing to Dew's own cache:
    `test_a_cases_peak_is_read_in_a_process_of_its_own_inside_the_cases_context_alone_and_cold`)."""
    named, own = tmp_path / "tree-cache", tmp_path / "home-cache"
    peak_row(tmp_path, "cpu", settings={
        "JAX_COMPILATION_CACHE_DIR": str(named), "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1", "XDG_CACHE_HOME": str(own)})
    assert [entry for entry in named.iterdir() if entry.name.startswith("jit_")]


def a_cases_measured_peak_holds_at_least_the_bytes_xla_planned(tmp_path):
    """On a GPU the measured peak is each device's allocator high-water over the case's run, its native
    workspaces and its profiled steps included (their traces apart): at least what XLA planned."""
    row = peak_row(tmp_path, "cuda", "--profile-dir", str(tmp_path / "traces"), "--profile-steps", "1")
    assert row["peak_device_bytes"] == row["peak_bytes_by_device"][0] >= row["planned_device_bytes"] > 0
    assert any((tmp_path / "traces/peak-run").rglob("*.xplane.pb"))
