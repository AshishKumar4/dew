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

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_as_exact_as_the_reference
from tool_support import REPO_ROOT, load

from dew.batch import TEXT_KEY
from dew.interop import Pretrained
from dew.objectives.lm import LMObjective
from dew.optim import Cosine, OptimConfig, ParamGroup
from dew.training import MeshSpec, Trainer

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
    schedule = Cosine(peak_value=options["lr_peak"], warmup_steps=options["warmup"],
                      end_value=options["lr_end"],
                      init_value=options["lr_init"])
    solver = OptimConfig(
        optimizer="adamw", b1=0.9, b2=0.95, optimizer_opts={"eps": 1e-8}, learning_rate=schedule,
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


def test_packed_documents_give_every_step_the_work_counted_for_one(monkeypatch):
    """`common.documents` packs every window with the same documents, numbered
    from 1, each LOW to HIGH tokens long but the last, which the row's end
    cuts; so Dew's count of one batch of packed windows is every step's, and
    it counts attention within each document only."""
    import json

    from dew.batch import coordinates
    from dew.interop import translate_config

    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools" / "reference_runs"))
    common = load("reference_runs/common")
    segments = common.documents((6, 33), "3:7")
    assert (segments == segments[0]).all()
    ids, lengths = np.unique(segments[0], return_counts=True)
    assert ids.tolist() == list(range(1, len(ids) + 1))
    assert all(3 <= length <= 7 for length in lengths[:-1]) and 1 <= lengths[-1] <= 7
    with pytest.raises(ValueError):
        common.documents((1, 4), "5:3")

    fields = translate_config(json.loads((FIXTURES / "hf" / "llama-tiny" / "config.json").read_text()))
    fields.update(dtype="float32", attention_impl="xla")
    objective = LMObjective(fields.value, seq_len=32, ema_decay=None)
    windows = np.random.default_rng(0).integers(0, 256, (6, 33), dtype=np.int32)

    def counted(rows, packed=True):
        batch = {TEXT_KEY: windows[rows]}
        if packed:
            batch["text_coordinates"] = coordinates(segments=segments[rows], max_segments=int(segments.max()),
                                                    xp=np)
        return objective.training_flops(None, batch)

    assert counted([0, 1]) == counted([4, 5]) < counted([0, 1], packed=False)


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


@pytest.mark.parametrize("model, size, vocabulary", [
    ("causal_transformer", "small", None), ("causal_transformer", "small", 8192),
    ("causal_transformer", "large", None), ("simple_dit", "small", None), ("simple_dit", "large", None),
    ("unet", "small", None)])
def test_a_parity_case_is_counted_as_the_trainer_counts_its_batch(model, size, vocabulary, tmp_path,
                                                                  monkeypatch):
    """`work.py case` writes the count `tools/benchmark_torch.py` divides by: the
    parity case the torch port steps, at the torch side's batch size and
    vocabulary, counted as the trainer counts that case's batch for
    train/mfu. A U-Net, whose convolutions have no token geometry, counts no
    tokens rather than zero."""
    from dew.training.model_flops import TRAINING_FLOPS, prepare_counts

    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools" / "reference_runs"))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    writer = load("reference_runs/work")
    cases, models = load("benchmark_cases"), load("benchmark_models")
    vocab = [] if vocabulary is None else ["--vocab-size", str(vocabulary)]
    args = writer.arguments(["case", "--model", model, "--size", size, "--batch-size", "2", *vocab,
                             "--out", str(tmp_path / "work.json")])
    work = args.count(args)
    case = cases.parity_case(model, size, batch_size=2, vocab_size=vocabulary)
    case = dataclasses.replace(case, dtype="bfloat16")
    counted = prepare_counts(models.build_objective(case), models.global_batch(case), count_model=True)
    assert work.training_flops_per_step == pytest.approx(float(np.sum(counted[TRAINING_FLOPS])), rel=1e-6)
    assert work.examples_per_step == 2
    assert (work.tokens_per_step is None) == (model == "unet")


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


def test_the_dew_qwen35_runner_trains_its_compiled_step(tmp_path):
    """The Dew side of the matched pair runs the trainer's compiled step as it returns: a tiny
    Qwen3.5 backbone on one device, as the runner requires, warms up, times and samples a step
    each, updating and staying finite."""
    script = """
import sys
from types import SimpleNamespace
from pathlib import Path
import jax, jax.numpy as jnp
sys.path[:0] = [sys.argv[1] + "/tests", sys.argv[1] + "/tools/reference_runs"]
from tool_support import load
from dew.interop import translate_config
runner, shapes = load("reference_runs/dew_qwen35_step"), load("reference_runs/qwen35")
model = translate_config(shapes.config(smoke=True)).value.clone(remat=None)
variables = model.init(jax.random.key(17), jnp.zeros((1, 16), jnp.int32))
args = SimpleNamespace(smoke=True, seed=17, work=Path(sys.argv[2]), warmup=1, steps=1)
trainer, state, compiled, batch, _, _ = runner.compile_step(args, model, variables)
seconds, _, samples, record = runner.train(args, trainer, state, compiled, batch)
assert seconds > 0 and len(samples) == 1 and record["finite_losses"]
"""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": "--xla_force_host_platform_device_count=1"}
    done = subprocess.run([sys.executable, "-c", script, str(REPO_ROOT), str(tmp_path / "work.json")],
                          env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
