"""Dew's language-model fine-tune step against transformers + torch's.

The reference runs in tools/reference_runs compare whole fine-tunes on real
weights and GPUs. This is their loop at a size the suite replays: the
committed qwen3-tiny checkpoint, eight steps of fixed token rows, cross
entropy over every shifted target, global-norm clipping at 1.0 and AdamW
(0.9/0.95, eps 1e-8, decoupled decay 0.1 on all parameters) on optax's warmup-cosine
schedule. tools/reference_runs/lm_steps_fixture.py ran it under transformers 5.16.1
and torch 2.14 on CPU in float64, the truth, and in float32, the reference.
Dew runs the trainer's own compiled step at fp32 and is held to the rule of
tests/reference_error.py over the steps: its RMS distance from float64 at
most twice torch's. A different decay, clip, schedule or loss normalisation
moves every step after the first by far more than fp32 rounding does.

Observed RMS distance from float64 over the eight steps against torch's
4.8e-07 (loss) and 2.2e-07 (gradient norm): Dew on one CPU device 3.3e-07
and 2.0e-07, on an RTX 3090 (JAX_PLATFORMS=cuda) 4.0e-07 and 3.0e-07. On
either, Dew without the decay lands at 1.1e-02 (ratio 22,000) and Dew
without the clip at 1.9e-03 (ratio 4,000).
"""

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_as_exact_as_the_reference
from tool_support import REPO_ROOT, load

from dew.config import OptimConfig
from dew.interop import Pretrained
from dew.objectives.lm import TEXT_KEY, LMObjective
from dew.training import MeshSpec, Trainer
from dew.training.optim import Cosine, ParamGroup

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class NormState(NamedTuple):
    norm: jax.Array


def recorded_norm() -> optax.GradientTransformation:
    """Pass the gradient on and keep its global norm, the value torch's
    `clip_grad_norm_` returns, in the optimizer state."""
    return optax.GradientTransformation(
        lambda params: NormState(jnp.zeros((), jnp.float32)),
        lambda updates, state, params=None: (updates, NormState(optax.tree.norm(updates))))


def test_lm_fine_tune_steps_are_as_exact_as_torch():
    fixture = np.load(FIXTURES / "reference_runs" / "qwen3-tiny-steps.npz")
    tokens = fixture["tokens"]
    steps, _, width = tokens.shape
    options = {"lr_peak": 1e-2, "lr_init": 1e-3, "lr_end": 1e-3, "warmup": 2}
    pretrained = Pretrained.load(str(FIXTURES / "hf" / "qwen3-tiny"), dtype="float32",
                                 attention_impl="xla")
    objective = LMObjective(pretrained, width - 1, ema_decay=None)
    schedule = Cosine(peak=options["lr_peak"], warmup_steps=options["warmup"], end=options["lr_end"],
                      init=options["lr_init"])
    solver = OptimConfig(
        optimizer="adamw", b1=0.9, b2=0.95, optimizer_opts={"eps": 1e-8}, schedule=schedule,
        weight_decay=0.1, clip_grads=1.0, param_groups=(ParamGroup("all", ("*",)),)).build(steps)
    trainer = Trainer(objective, optax.chain(recorded_norm(), solver), key=jax.random.key(0),
                      mesh=MeshSpec(), checkpoints=None, tracker=None)
    state, _, _ = trainer.place()
    step = trainer.compile(state, {TEXT_KEY: tokens[0]})
    losses, norms = [], []
    for rows in tokens:
        state, loss, _, _, _, _ = step(state, {TEXT_KEY: rows})
        losses.append(float(loss))
        norms.append(float(state.opt_state[0].norm))

    np.testing.assert_allclose([schedule.schedule(steps)(index) for index in range(steps)],
                               fixture["lr"], rtol=1e-6)
    assert_as_exact_as_the_reference(losses, fixture["f32_loss"], fixture["f64_loss"], "loss")
    assert_as_exact_as_the_reference(norms, fixture["f32_grad_norm"], fixture["f64_grad_norm"],
                                     "gradient norm")


@pytest.mark.parametrize("dew, torch, state, passed", [
    ([100.0] * 5, [100.0] * 5, "level", True),
    ([110.0] * 5, [100.0] * 5, "faster", True),
    ([72.0] * 5, [100.0] * 5, "slower", False),
    ([99.0] * 5, [99.0, 100.0, 100.0, 100.0, 101.0], "level", True),
    ([98.0] * 5, [100.0] * 5, "slower", False),
])
def test_qwen35_pair_gates_throughput_at_its_measured_spread(dew, torch, state, passed):
    """A clearly slower Dew cannot pass; overlapping noise means level, not a speed claim."""
    pair = load("reference_runs/qwen35_pair")
    decision = pair.performance_verdict(dew, torch)
    assert decision["verdict"] == state
    assert decision["passed"] is passed
    assert decision["dew_over_torch"] == pytest.approx(np.median(dew) / np.median(torch))
    expected = max((max(samples) - min(samples)) / np.median(samples)
                   for samples in (dew, torch))
    assert decision["tolerance"] == pytest.approx(expected)


@pytest.mark.parametrize("samples", [[], [1.0], [0.0, 0.0], [-1.0, -1.0],
                                     [float("nan")] * 2, [float("inf")] * 2])
def test_qwen35_pair_refuses_missing_or_invalid_measurements(samples):
    pair = load("reference_runs/qwen35_pair")
    with pytest.raises(ValueError, match="finite positive"):
        pair.performance_verdict(samples, [100.0, 100.0])
    with pytest.raises(ValueError, match="finite positive"):
        pair.performance_verdict([100.0, 100.0], samples)


def qwen35_records(smoke=False):
    """A matched pair with deliberately conflicting per-step diagnostic rates."""
    import copy

    shared = load("reference_runs/qwen35")
    shape = (128, 32, 2, 16, 3, 4) if smoke else (248320, 1024, 8, 3072, 20, 200)
    vocab, width, batch, sequence, warmup, steps = shape
    config = {**shared.workload(17, vocab, width, batch, sequence, smoke, warmup, steps),
              "parameter_count": 23564}
    record = {
        "session": "session", "smoke": smoke, "commit": "commit", "config": config,
        "work": {"training_flops_per_step": 1000.0, "tokens_per_step": batch * sequence,
                 "basis": "analytic, real tokens", "model_identity": "matched-native-config",
                 "batch_shape": [batch, sequence]},
        "host": {"hostname": "same-host", "cuda_visible_devices": None},
        "environment": {"device": "same-device", "packages": {}},
        "checks": {"loss_first": 0.39884394, "finite_losses": True, "no_timing_window_compilation": True},
        "memory": {"whole_peak_bytes": 1000, "whole_peak_scope": shared.WHOLE_PEAK_SCOPE},
        "steady": {"window_tokens_per_s": 100.0, "window_mfu": 0.4, "tokens_per_s": 100000.0},
    }
    records = {framework: [copy.deepcopy(record), copy.deepcopy(record)] for framework in ("dew", "torch")}
    for framework, rounds in records.items():
        for row in rounds:
            row["framework"] = framework
            if framework == "dew":
                row["steady"].update(window_tokens_per_s=110.0, tokens_per_s=1.0)
    return records


def test_qwen35_pair_uses_window_rates_and_retains_scoped_peaks_and_mfus():
    pair = load("reference_runs/qwen35_pair")
    decision = pair.compare(qwen35_records(), "session", smoke=False)
    assert decision["dew_over_torch"] == 1.1
    assert decision["performance_claim_passed"]
    assert decision["whole_peak_bytes"] == {"dew": [1000, 1000], "torch": [1000, 1000]}
    assert decision["mfu"] == {"dew": [0.4, 0.4], "torch": [0.4, 0.4]}
    assert decision["loss_parity"]


@pytest.mark.parametrize("section, field, bad", [
    ("config", "tokens_sha256", "different"), ("config", "projection_sha256", "different"),
    ("config", "parameter_count", 23563), ("config", "steps", 10),
    ("config", "gradient_checkpointing", True), ("config", "betas", [0.9, 0.95]),
    ("checks", "loss_first", 0.6), ("checks", "finite_losses", False),
    ("checks", "no_timing_window_compilation", False),
    ("memory", "whole_peak_scope", "steady window"), ("memory", "whole_peak_bytes", None),
    ("host", "hostname", "another-host"), ("host", "cuda_visible_devices", "1"),
    ("environment", "device", "another-device"),
])
def test_qwen35_pair_rejects_mismatched_records(section, field, bad):
    pair = load("reference_runs/qwen35_pair")
    records = qwen35_records()
    records["dew"][1][section][field] = bad
    with pytest.raises(ValueError):
        pair.compare(records, "session", smoke=False)


def test_qwen35_cpu_smoke_proof_is_never_a_performance_claim():
    pair = load("reference_runs/qwen35_pair")
    decision = pair.compare(qwen35_records(smoke=True), "session", smoke=True)
    assert decision["passed"] and decision["loss_parity"]
    assert decision["performance_claim_passed"] is False
    with pytest.raises(SystemExit):
        pair.arguments(["--torch-python", "python", "--out", "out.json", "--smoke", "--claim"])


def test_qwen35_full_claim_refuses_a_matched_but_smaller_workload():
    pair = load("reference_runs/qwen35_pair")
    records = qwen35_records(smoke=True)
    for rounds in records.values():
        for row in rounds:
            row["smoke"] = False
    with pytest.raises(ValueError, match="full 8x3072"):
        pair.compare(records, "session", smoke=False)


@pytest.mark.parametrize("field, value", [("session", "other-session"), ("commit", "other-commit")])
def test_qwen35_pair_cannot_mix_sessions_or_commits(field, value):
    pair = load("reference_runs/qwen35_pair")
    records = qwen35_records()
    records["dew"][1][field] = value
    with pytest.raises(ValueError):
        pair.compare(records, "session", smoke=False)


@pytest.mark.parametrize("rate, code", [(72.0, 1), (110.0, 0)])
def test_qwen35_pair_run_exits_nonzero_when_dew_is_slower(tmp_path, monkeypatch, rate, code):
    """The process runner must carry a failed arithmetic verdict to its actual exit status."""
    import json
    import subprocess

    pair = load("reference_runs/qwen35_pair")
    records = qwen35_records()
    for row in records["dew"]:
        row["steady"]["window_tokens_per_s"] = rate
    calls = []
    real_run = pair.subprocess.run

    def child(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        if "work.py" in command[1]:
            Path(command[command.index("--out") + 1]).write_text(json.dumps(records["torch"][0]["work"]))
            calls.append("work")
            return subprocess.CompletedProcess(command, 0)
        framework = "torch" if "torch_qwen35_step.py" in command[1] else "dew"
        index = sum(call == framework for call in calls)
        record = records[framework][index]
        record["session"] = command[command.index("--session") + 1]
        Path(command[command.index("--out") + 1]).write_text(json.dumps(record))
        calls.append(framework)
        assert kwargs["env"]["PYTHONPATH"].endswith("/src")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pair.subprocess, "run", child)
    args = pair.arguments(["--torch-python", "torch-python", "--out", str(tmp_path / "pair.json")])
    assert pair.run(args) == code
    assert calls == ["work", "torch", "dew", "torch", "dew"]
    written = json.loads(args.out.read_text())
    assert written["comparison"]["dew_over_torch"] == rate / 100
    assert written["comparison"]["performance_claim_passed"] is (code == 0)
    assert len(written["commands"]) == 5


@pytest.mark.parametrize("family", ["qwen3-tiny", "mamba2-tiny", "mixtral-tiny", "qwen3-moe-tiny",
                                    "qwen35-tiny"])
def test_reference_work_is_the_trainers_training_flops_at_the_same_shape(tmp_path, monkeypatch, family):
    """A shifted target column is real data, but contributes no extra model token."""
    from dew.interop import translate_config
    from dew.telemetry.workload import WorkloadWork
    from dew.training.model_flops import TRAINING_FLOPS, prepare_counts

    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools" / "reference_runs"))
    writer = load("reference_runs/work")
    args = writer.arguments(["lm", "--config", str(FIXTURES / "hf" / family / "config.json"),
                             "--batch", "2", "--seq", "4", "--dtype", "float32",
                             "--attention", "xla", "--out", str(tmp_path / "work.json")])
    work = args.count(args)
    import json

    fields = translate_config(json.loads(Path(args.config).read_text()))
    fields.update(dtype="float32", attention_impl="xla")
    objective = LMObjective(fields.value, seq_len=4, ema_decay=None)
    batch = {TEXT_KEY: np.zeros((2, 5), np.int32)}
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0), tracker=None, checkpoints=None)
    assert work.training_flops_per_step == trainer.objective.training_flops(None, batch)
    counted = prepare_counts(trainer.objective, batch, count_model=True)
    assert float(np.sum(counted[TRAINING_FLOPS])) == work.training_flops_per_step
    assert work.tokens_per_step == 8
    assert work.batch_shape == (2, 5)
    work.write(args.out)
    assert WorkloadWork.read(args.out) == work
    identity = json.loads(work.model_identity)
    assert identity["fields"]["emb_features"] == fields.value.emb_features


def test_dit_work_includes_the_native_conditioning_geometry(monkeypatch):
    from dew.nn.backbones.dit import SimpleDiT

    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools" / "reference_runs"))
    runner = load("reference_runs/diffusion")
    writer = load("reference_runs/work")
    objective = runner.counting_objective((8, 8, 3), "float32", "xla")
    batch = {"image": np.zeros((2, 8, 8, 3), np.uint8)}
    work = writer.measure(objective, batch)
    assert work.training_flops_per_step == objective.training_flops(None, batch)
    assert work.tokens_per_step == 8
    assert work.examples_per_step == 2
    model = SimpleDiT(**runner.MODEL)
    assert work.training_flops_per_step == 3 * model.image_flops(np.array([4, 4]), 3)


def test_qwen35_work_counts_the_shared_smoke_geometry(monkeypatch, tmp_path):
    """The config-only count must see the convolution that HF builds by default."""
    from dew.nn.flops import counted_flops
    from dew.telemetry.workload import WorkloadWork

    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools" / "reference_runs"))
    writer = load("reference_runs/work")
    args = writer.arguments(["qwen35", "--smoke", "--dtype", "float32",
                             "--out", str(tmp_path / "work.json")])
    work = args.count(args)
    shared = load("reference_runs/qwen35")
    from dew.interop import translate_config

    model = translate_config(shared.config(smoke=True)).value
    assert work.training_flops_per_step == 3 * counted_flops(model, np.array([16, 16]))
    assert work.tokens_per_step == 32
    assert work.batch_shape == (2, 16)
    work.write(args.out)
    assert WorkloadWork.read(args.out) == work


@pytest.mark.parametrize("runner, argv", [
    ("torch_lm", ["--model", "model", "--data", "data", "--out", "out", "--batch", "2",
                  "--precision", "fp32"]),
    ("maxtext_run", ["--model", "model", "--model-name", "qwen3-0.6b", "--out", "out",
                     "--batch", "2", "--mesh", "data"]),
])
def test_reference_runner_requires_the_native_work_record(monkeypatch, capsys, runner, argv):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools" / "reference_runs"))
    tool = load(f"reference_runs/{runner}")
    with pytest.raises(SystemExit) as stopped:
        tool.arguments(argv)
    assert stopped.value.code == 2
    assert "the following arguments are required: --work" in capsys.readouterr().err
    assert tool.arguments([*argv, "--work", "work.json"]).work == "work.json"


def test_reference_reads_work_without_importing_dew_or_jax(tmp_path):
    import subprocess
    import sys

    from dew.telemetry.workload import WorkloadWork

    work = WorkloadWork(1234.0, 8, "analytic, real tokens, causal half", "native-config", (2, 5))
    path = tmp_path / "work.json"
    work.write(path)
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); from common import WorkloadWork; "
        "record = WorkloadWork.read(sys.argv[2]); "
        "assert record.training_flops_per_step == 1234.0; "
        "assert record.tokens_per_step == 8; assert record.batch_shape == (2, 5); "
        "assert not ({\"dew\", \"jax\", \"flax\"} & sys.modules.keys())"
    )
    result = subprocess.run([sys.executable, "-c", script, str(REPO_ROOT / "tools" / "reference_runs"),
                             str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_torch_reference_keeps_whole_run_peak_beside_reset_window_peak(monkeypatch):
    """A large warmup allocation remains visible even if steady-state memory is smaller."""
    from types import SimpleNamespace

    import torch

    shared = load("reference_runs/common")
    window = shared.TorchTracedWindow(torch.device("cpu"), 4, 2, 0, reset_peak=True)
    window.device = SimpleNamespace(type="cuda")
    allocated = {"peak": 500}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: allocated["peak"])
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: allocated["peak"] + 100)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: allocated.update(peak=300))
    window.before(2)
    window._end()
    assert window.peak_allocated == 300
    assert window.peak_reserved == 400
    assert window.whole_peak_allocated == 500
