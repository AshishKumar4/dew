"""LADD with ADD's distillation term: the hinge losses from their equations,
each side's gradient from its own loss, and a run config over a saved
teacher."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_diffusion_run_sources import batch_for

from dew.checkpoints import Checkpoints
from dew.config import ModelConfig, TrainerConfig
from dew.data import OxfordFlowers
from dew.objectives.base import Step
from dew.objectives.diffusion import (
    AdversarialDistillation,
    AdversarialDistillationObjective,
    DiffusionRunConfig,
    TextCondition,
)
from dew.objectives.diffusion.adversarial import hinge_discriminator, hinge_generator
from dew.objectives.diffusion.objective import DISCRIMINATOR, TEACHER
from dew.registry import presets, samplers
from dew.sampling import TextToImage
from dew.training import Trainer


def test_the_hinge_losses_are_the_papers():
    """relu(1 - D(real)) + relu(1 + D(fake)) for the discriminator and
    -D(fake) for the generator, meaned over tokens and summed over heads."""
    real = [jnp.asarray([[2.0, 0.5], [-1.0, 0.0]]), jnp.asarray([[0.2], [3.0]])]
    fake = [jnp.asarray([[-2.0, 0.5], [1.0, -0.5]]), jnp.asarray([[0.0], [-1.5]])]
    np.testing.assert_allclose(np.asarray(hinge_discriminator(real, fake)),
                               [(0 + 0.5) / 2 + (0 + 1.5) / 2 + 0.8 + 1.0, (2 + 1) / 2 + (2 + 0.5) / 2 + 0 + 0])
    np.testing.assert_allclose(np.asarray(hinge_generator(fake)), [0.75 - 0.0, -0.25 + 1.5])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ladd")
    teacher = DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 2, "num_heads": 2},
                          dtype="float32", attention_impl="xla"),
        data=OxfordFlowers(image_size=4), preset=presets.Flow(), sampler=samplers.Euler(), guidance=None,
        sampling_steps=2, ema_decay=None, val_metrics=(), trainer=TrainerConfig(checkpoint_dir=str(root)),
        text=TextCondition(encoder="char_table", checkpoint="char_table"))
    objective = teacher.build()
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.PRNGKey(0))
    state = trainer.initial_state()
    batch = batch_for(objective, 4)
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(root / "teacher"))
    checkpoints.save(1, state, None)
    checkpoints.wait()
    teacher.save(str(root / "teacher"))
    student = dataclasses.replace(teacher, sampler=samplers.Consistency(), adversarial=AdversarialDistillation(
        teacher=str(root / "teacher"), feature_layers=("dit_block_0", "dit_block_1"), distillation_weight=2.5,
        head_width=8))
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
            return total.total if name is None else sum(aux.metrics[key] for key in name) * total.mass
        return jax.grad(loss)(params["params"])

    everything = part(None)
    critic, student = part(("discriminator",)), part(("generator", "distillation"))
    for got, want in zip(jax.tree.leaves(everything[DISCRIMINATOR]), jax.tree.leaves(critic[DISCRIMINATOR]),
                         strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-7)
    for name in everything:
        if name != DISCRIMINATOR:
            for got, want in zip(jax.tree.leaves(everything[name]), jax.tree.leaves(student[name]), strict=True):
                np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-7)
    assert float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(critic[DISCRIMINATOR]))) > 0


def test_a_saved_student_samples_in_one_step(runs, tmp_path):
    config, batch = runs
    task = config.build()
    trainer = Trainer(task, optax.adam(1e-3), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(tmp_path / "student"))
    checkpoints.save(1, state, None)
    checkpoints.wait()
    config.save(str(tmp_path / "student"))
    restored = TextToImage.from_run(str(tmp_path / "student"))
    assert TEACHER not in restored.params and DISCRIMINATOR not in restored.params["params"]
    expected = task.pipeline(state, ema=False)(["a red bird"], seed=9).host().images
    np.testing.assert_array_equal(restored(["a red bird"], seed=9).host().images, expected)
