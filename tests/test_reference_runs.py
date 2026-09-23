"""Dew's language-model fine-tune step against transformers + torch's.

The reference runs in tools/reference_runs compare whole fine-tunes on real
weights and GPUs. This is their loop at a size the suite replays: the
committed qwen3-tiny checkpoint, eight steps of fixed token rows, cross
entropy over every shifted target, global-norm clipping at 1.0 and AdamW
(0.9/0.95, eps 1e-8, decoupled decay 0.1) on optax's warmup-cosine schedule.
tools/reference_runs/lm_steps_fixture.py ran it under transformers 5.16.1
and torch 2.14 on CPU in float64, the truth, and in float32, the reference.
Dew runs the trainer's own compiled step at fp32 and is held to the rule of
tests/reference_error.py over the steps: its RMS distance from float64 at
most twice torch's. A different decay, clip, schedule or loss normalisation
moves every step after the first by far more than fp32 rounding does.

Observed RMS distance from float64 over the eight steps, Dew on one CPU
device / torch: loss 3.3e-07 / 4.8e-07, gradient norm 2.0e-07 / 2.2e-07.
On an A100, where the test passes too, Dew without the decay lands at
1.1e-02 (ratio 22,000) and Dew without the clip at 1.9e-03 (ratio 4,000).
"""

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from reference_error import assert_as_exact_as_the_reference

from dew.config import OptimConfig
from dew.interop import load_pretrained
from dew.objectives.lm import TEXT_KEY, LMObjective
from dew.training import MeshSpec, Trainer
from dew.training.optim import Cosine, build_optimizer

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
    pretrained = load_pretrained(str(FIXTURES / "hf" / "qwen3-tiny"), dtype="float32",
                                 attention_impl="xla")
    objective = LMObjective(pretrained.model, width - 1, ema_decay=None,
                            pretrained=pretrained.variables)
    schedule = Cosine(peak=options["lr_peak"], warmup_steps=options["warmup"], end=options["lr_end"],
                      init=options["lr_init"])
    solver = build_optimizer(OptimConfig(
        optimizer="adamw", optimizer_opts={"b1": 0.9, "b2": 0.95, "eps": 1e-8}, schedule=schedule,
        weight_decay=0.1, clip_grads=1.0), steps)
    trainer = Trainer(objective, optax.chain(recorded_norm(), solver), key=jax.random.key(0),
                      mesh=MeshSpec(), checkpoints=None, tracker=None)
    state, _, _ = trainer.place()
    step = trainer.compile(state, {TEXT_KEY: tokens[0]})
    losses, norms = [], []
    for rows in tokens:
        state, loss, _, _, _ = step(state, {TEXT_KEY: rows})
        losses.append(float(loss))
        norms.append(float(state.opt_state[0].norm))

    np.testing.assert_allclose([schedule.schedule(steps)(index) for index in range(steps)],
                               fixture["lr"], rtol=1e-6)
    assert_as_exact_as_the_reference(losses, fixture["f32_loss"], fixture["f64_loss"], "loss")
    assert_as_exact_as_the_reference(norms, fixture["f32_grad_norm"], fixture["f64_grad_norm"],
                                     "gradient norm")
