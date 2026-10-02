"""LADD with ADD's distillation term: the hinge losses from their equations,
each side's gradient from its own loss, and a run config over a saved
teacher."""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_diffusion_run_sources import batch_for

from dew.checkpoints import Checkpoints
from dew.config import ModelConfig, TrainerConfig
from dew.data import TFDSImages
from dew.diffusion.presets import Flow
from dew.objectives.base import Step
from dew.objectives.diffusion import (
    AdversarialDistillation,
    AdversarialDistillationObjective,
    DiffusionRunConfig,
    TextCondition,
)
from dew.objectives.diffusion.adversarial import Head, hinge_discriminator, hinge_generator, r1_penalty
from dew.objectives.diffusion.objective import DISCRIMINATOR, SPECTRAL, TEACHER
from dew.sampling import Consistency, Euler, TextToImage
from dew.training import Trainer

HEAD = np.load(Path(__file__).resolve().parent / "fixtures" / "stylegan_t" / "head.npz")


def test_a_head_is_stylegan_ts_at_grid_height_one():
    """StyleGAN-T's DiscHead in training mode (tools/stylegan_t_reference.py)
    on 16 sequences of 12 tokens: LADD's 2D head over a grid of height one,
    with a (1, 9) kernel, from the same weights and spectral-norm vectors.
    Both run in float64, so the gap is a few roundings of O(1) logits."""
    with jax.enable_x64(new_val=True):
        def conv(name):
            return {"kernel": jnp.asarray(HEAD[f"{name}.kernel"]), "bias": jnp.asarray(HEAD[f"{name}.bias"])}

        def norm(name):
            return {"weight": jnp.asarray(HEAD[f"{name}.weight"]), "bias": jnp.asarray(HEAD[f"{name}.bias"])}

        condition_width = HEAD["c"].shape[-1]
        params = {"block_0": {"conv": conv("main.0.0"), "norm": norm("main.0.1")},
                  "block_1": {"conv": conv("main.1.fn.0"), "norm": norm("main.1.fn.1")},
                  "cls": conv("cls"),
                  "cmapper_weight": jnp.asarray(HEAD["cmapper.weight"]) * np.sqrt(condition_width),
                  "cmapper_bias": jnp.asarray(HEAD["cmapper.bias"])}
        spectral = {"block_0": {"conv": {"u": jnp.asarray(HEAD["main.0.0.u"])}},
                    "block_1": {"conv": {"u": jnp.asarray(HEAD["main.1.fn.0.u"])}},
                    "cls": {"u": jnp.asarray(HEAD["cls.u"])}}
        logits, updated = Head(kernel_size=(1, 9)).apply(
            {"params": params, SPECTRAL: spectral}, jnp.asarray(HEAD["x"]), jnp.asarray(HEAD["c"]),
            update=True,
            mutable=[SPECTRAL])
        np.testing.assert_allclose(np.asarray(logits), HEAD["logits"], rtol=1e-10, atol=1e-12)
        assert not np.allclose(np.asarray(updated[SPECTRAL]["cls"]["u"]), HEAD["cls.u"])


def test_the_hinge_losses_are_the_papers():
    """relu(1 - D(real)) + relu(1 + D(fake)) for the discriminator and
    -D(fake) for the generator, meaned over every head's every logit."""
    real = [jnp.asarray([[2.0, 0.5], [-1.0, 0.0]]), jnp.asarray([[0.2], [3.0]])]
    fake = [jnp.asarray([[-2.0, 0.5], [1.0, -0.5]]), jnp.asarray([[0.0], [-1.5]])]
    np.testing.assert_allclose(np.asarray(hinge_discriminator(real, fake)),
                               [(0 + 0.5 + 0.8) / 3 + (0 + 1.5 + 1) / 3, (2 + 1 + 0) / 3 + (2 + 0.5 + 0) / 3])
    np.testing.assert_allclose(np.asarray(hinge_generator(fake)), [-(-2 + 0.5 + 0) / 3, -(1 - 0.5 - 1.5) / 3])


def test_r1_is_the_squared_gradient_of_each_heads_mean_logit_at_its_input():
    """ADD's R1 on each head's input, against the gradient written out for a
    quadratic head: d/dx mean(x^2 w) = 2 x w / n."""
    features = [jnp.asarray([[1.0, 2.0], [0.5, -1.0]]), jnp.asarray([[3.0], [-2.0]])]
    weights = [jnp.asarray([0.5, 2.0]), jnp.asarray([1.0])]

    def score(features):
        return [jnp.square(f) * w for f, w in zip(features, weights, strict=True)]
    expected = [np.sum(np.square(2 * np.asarray(f) * np.asarray(w) / f.shape[1]), axis=1)
                for f, w in zip(features, weights, strict=True)]
    np.testing.assert_allclose(np.asarray(r1_penalty(score, features)), expected[0] + expected[1], rtol=1e-6)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ladd")
    teacher = DiffusionRunConfig(
        model=ModelConfig("simple_dit",
                          {"patch_size": 2, "emb_features": 16, "num_layers": 2, "num_heads": 2},
                          dtype="float32", attention_impl="xla"),
        data=TFDSImages(image_size=4), preset=Flow(), solver=Euler(), guidance=None,
        sampling_steps=2, ema_decay=None, val_metrics=(), trainer=TrainerConfig(checkpoint_dir=str(root)),
        text=TextCondition(encoder="char_table", checkpoint="char_table"))
    objective = teacher.build()
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.PRNGKey(0))
    state = trainer.initial_state()
    batch = batch_for(objective, 4)
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(root / "teacher"))
    checkpoints.save(1, state, None, artifact=objective.inference_record)
    checkpoints.wait()
    teacher.save(str(root / "teacher"))
    student = dataclasses.replace(teacher, solver=Consistency(), adversarial=AdversarialDistillation(
        teacher=str(root / "teacher"), feature_layers=("dit_block_0", "dit_block_1"), cmap_dim=8,
        kernel_size=(3, 3)))
    return student, batch


def test_each_side_trains_on_its_own_loss(runs):
    """The heads' gradient is the discriminator loss's alone and the
    student's the generator and distillation losses' alone; the teacher is
    not a parameter."""
    config, batch = runs
    task = config.build()
    assert isinstance(task, AdversarialDistillationObjective)
    params = task.init(jax.random.PRNGKey(1))
    assert TEACHER not in params["params"]
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)

    def part(name):
        def loss(tree):
            total, aux = task.loss({**params, "params": tree}, batch, step)
            if name is None:
                return total.total
            return sum(aux.metrics[key] * (task.r1_weight if key == "r1" else 1) for key in name) * total.mass
        return jax.grad(loss)(params["params"])

    everything = part(None)
    critic, student = part(("discriminator", "r1")), part(("generator", "distillation"))
    for got, want in zip(jax.tree.leaves(everything[DISCRIMINATOR]), jax.tree.leaves(critic[DISCRIMINATOR]),
                         strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-7)
    for name in everything:
        if name != DISCRIMINATOR:
            for got, want in zip(jax.tree.leaves(everything[name]), jax.tree.leaves(student[name]),
                                 strict=True):
                np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-7)
    assert float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(critic[DISCRIMINATOR]))) > 0


def test_a_saved_student_samples_in_one_step(runs, tmp_path):
    config, batch = runs
    task = config.build()
    trainer = Trainer(task, optax.adam(1e-3), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(tmp_path / "student"))
    checkpoints.save(1, state, None, artifact=task.inference_record)
    checkpoints.wait()
    config.save(str(tmp_path / "student"))
    restored = TextToImage.from_run(str(tmp_path / "student"))
    assert TEACHER not in restored.params and DISCRIMINATOR not in restored.params["params"]
    expected = task.pipeline(state, ema=False)(["a red bird"], seed=9).host().images
    np.testing.assert_array_equal(restored(["a red bird"], seed=9).host().images, expected)
