"""Guidance distillation (Meng et al. 2023, stage one) on the committed tiny
Flux pipeline's text towers: the loss written out from the paper's equation,
and a saved student run restored."""

import dataclasses
import tarfile
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
from dew.diffusion import broadcast_rates
from dew.diffusion.presets import Flow
from dew.inputs import unit_range
from dew.objectives.base import Step
from dew.objectives.diffusion import (
    DiffusionRunConfig,
    GuidanceDistillation,
    GuidanceDistillationObjective,
    TextCondition,
)
from dew.objectives.diffusion.guidance_distillation import with_guidance
from dew.objectives.diffusion.objective import TEACHER
from dew.sampling import Euler, TextToImage
from dew.training import Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FLUX = {
    "in_channels": 12,
    "out_channels": 12,
    "num_layers": 1,
    "num_single_layers": 1,
    "heads": 2,
    "head_dim": 12,
    "joint_attention_dim": 16,
    "pooled_projection_dim": 10,
    "guidance_embeds": True,
    "axes_dims_rope": [4, 4, 4],
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A tiny guidance-embedded Flux teacher, trained a step and saved, and
    the config of a student that distills it."""
    root = tmp_path_factory.mktemp("guidance")
    with tarfile.open(FIXTURES / "flux_source.tar.xz") as archive:
        archive.extractall(root / "flux", filter="data")
    teacher = DiffusionRunConfig(
        model=ModelConfig("flux_transformer", FLUX, dtype="float32", attention_impl="xla"),
        data=TFDSImages(image_size=8), preset=Flow(), solver=Euler(), guidance=None,
        sampling_steps=2, ema_decay=None, val_metrics=(), trainer=TrainerConfig(checkpoint_dir=str(root)),
        text=TextCondition(encoder="diffusion_text", checkpoint=str(root / "flux" / "pipeline")))
    objective = teacher.build()
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.PRNGKey(0))
    state = trainer.initial_state()
    batch = batch_for(objective, 8)
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(root / "teacher"))
    checkpoints.save(1, state, None)
    checkpoints.wait()
    teacher.save(str(root / "teacher"))
    student = dataclasses.replace(
        teacher, mode=GuidanceDistillation(teacher=str(root / "teacher"), scales=(1.0, 6.0))
    )
    return root, student, batch


def test_the_loss_is_the_papers_equation(runs):
    """Per row a scale w ~ U(1, 6); the student reads it as its guidance and
    matches u + w (c - u) of the teacher's two branches on the same noised
    sample, in raw velocity, under Dew's halved L2."""
    _, config, batch = runs
    task = config.build()
    assert isinstance(task, GuidanceDistillationObjective)
    params = task.init(jax.random.PRNGKey(1))
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)
    loss, _ = task.loss(params, batch, step)

    _, drop_key, time_key, noise_key, scale_key = jax.random.split(step.key, 5)
    assert task.autoencoder is None
    samples = unit_range(batch["image"])
    count = samples.shape[0]
    scale = jax.random.uniform(scale_key, (count,), minval=1.0, maxval=6.0)
    schedule = task.process.schedule
    t = schedule.sample_t(time_key, count)
    noise = jax.random.normal(noise_key, samples.shape)
    alpha, sigma = broadcast_rates(schedule, t, samples)
    x = alpha * samples + sigma * noise
    teacher = task.teacher
    given, blank = teacher._conditions(params[TEACHER], batch, drop_key, dropout=False)
    variables = teacher.model_variables(params[TEACHER])
    conditional = teacher.model.apply(variables, x, schedule.model_time(t), **given)
    unconditional = teacher.process.denoiser(teacher.model, variables, given, blank).raw_both(x, t)[1]
    target = unconditional + scale.reshape(-1, 1, 1, 1) * (conditional - unconditional)
    student_given, _ = task._conditions(params, batch, drop_key, dropout=False)
    output = task.model.apply(task.model_variables(params), x, schedule.model_time(t),
                              **with_guidance(student_given, scale))
    expected = jnp.mean(0.5 * jnp.square(output - target))
    assert float(loss.total / loss.mass) == pytest.approx(float(expected), rel=1e-5)
    modules = [program.module.clone() for program in task.program_key()]
    task.substitute(modules)
    assert all(program.module is module for program, module in zip(task.program_key(), modules, strict=True))


def test_a_saved_student_samples_one_branch_at_its_conditioners_guidance(runs, tmp_path):
    _, config, batch = runs
    task = config.build()
    trainer = Trainer(task, optax.adam(1e-2), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(tmp_path / "student"))
    checkpoints.save(1, state, None, artifact=task.inference_record())
    checkpoints.wait()
    config.save(str(tmp_path / "student"))
    restored = TextToImage.from_run(str(tmp_path / "student"))
    assert restored.guidance is None and TEACHER not in restored.variables
    expected = task.pipeline(state, ema=False)(["a red bird"], key=9).host().images
    np.testing.assert_array_equal(restored(["a red bird"], key=9).host().images, expected)
