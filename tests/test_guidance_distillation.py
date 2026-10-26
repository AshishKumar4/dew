"""Guidance distillation (Meng et al. 2023, stage one) on the committed tiny
Flux pipeline's text towers: the loss written out from the paper's equation,
and a saved student run restored."""

import dataclasses
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffusion_stubs import batch_for
from diffusion_support import images_of, saved_after_one_step

from dew.config import ObjectiveConfig, TrainerConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.diffusion import broadcast_rates
from dew.diffusion.presets import EDM, Flow
from dew.diffusion.process import aligned_conditions
from dew.inference import TextToImage
from dew.inputs import TextCondition, unit_range
from dew.objectives.base import TEACHER, Step
from dew.objectives.diffusion import GuidanceDistillationObjective
from dew.objectives.diffusion.guidance_distillation import with_guidance
from dew.registry import ModelConfig
from dew.sampling import Euler

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


@pytest.fixture(scope="module", params=[Flow(), EDM(regime="pixel")], ids=["flow", "edm"])
def runs(tmp_path_factory, request):
    """A tiny guidance-embedded Flux teacher, trained a step and saved, and
    the config of a student that distills it: under flow matching, whose
    training and sampling schedules are one, and under EDM, whose training
    times are standard-normal draws its Karras sampling grid reads otherwise."""
    root = tmp_path_factory.mktemp("guidance")
    with tarfile.open(FIXTURES / "flux_source.tar.xz") as archive:
        archive.extractall(root / "flux", filter="data")
    teacher = DiffusionRunConfig(
        model=ModelConfig("flux_transformer", {**FLUX, "dtype": "float32", "attention_impl": "xla"}),
        data=images_of("tfds://oxford_flowers102", 8), preset=request.param, val_metrics=(),
        trainer=TrainerConfig(checkpoint_dir=str(root)),
        text=TextCondition(encoder="diffusion_text", checkpoint=str(root / "flux" / "pipeline")),
        objective=ObjectiveConfig("diffusion",
                                  {"solver": Euler(), "guidance": None, "steps": 2, "ema_decay": None}))
    _, _, batch = saved_after_one_step(teacher, lambda task: batch_for(task, 8), root / "teacher", 1e-2,
                                       key=0)
    student = dataclasses.replace(teacher, teacher_run=str(root / "teacher"), objective=ObjectiveConfig(
        "guidance_distillation", {"scales": [1.0, 6.0], "solver": Euler(), "steps": 2, "ema_decay": None}))
    return root, student, batch


def test_a_student_whose_process_reads_raw_outputs_otherwise_is_refused(runs):
    """The student regresses onto the teacher's raw output, so their schedules and prediction
    transforms are one configuration, not only one class: a student whose EDM sigma_data or flow
    shift differs from its teacher run's is refused, and the student as configured builds."""
    _, student, _ = runs
    preset = student.preset
    changed = (dataclasses.replace(preset, sigma_data=0.25) if isinstance(preset, EDM)
               else dataclasses.replace(preset, shift=3.0))
    with pytest.raises(ValueError):
        dataclasses.replace(student, preset=changed).build()
    assert isinstance(student.build(), GuidanceDistillationObjective)


def test_the_loss_is_the_papers_equation(runs):
    """Per row a scale w ~ U(1, 6); the student reads it as its guidance and
    matches u + w (c - u) of the teacher's two branches on the same noised
    sample, in raw output, under Dew's halved L2 weighted as the process
    weighs the time. Teacher and student read the sample at the time drawn
    on the training schedule, input-scaled for it."""
    _, config, batch = runs
    task = config.build()
    assert isinstance(task, GuidanceDistillationObjective)
    params = task.init(jax.random.PRNGKey(1))
    _, drop_key, time_key, noise_key, scale_key = jax.random.split(jax.random.PRNGKey(2), 5)
    assert task.autoencoder is None
    samples = unit_range(batch["image"])
    count = samples.shape[0]
    scale = jax.random.uniform(scale_key, (count,), minval=1.0, maxval=6.0)
    schedule = task.process.schedule
    t = schedule.sample_t(time_key, count)
    noise = jax.random.normal(noise_key, samples.shape)
    # The step reads these scales, times and noise (`Step.tape`); its conditions' key is its own.
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None,
                tape={"guidance scale": scale, "times": t, "noise": noise})
    loss, _ = task.loss(params, batch, step)
    alpha, sigma = broadcast_rates(schedule, t, samples)
    x = (alpha * samples + sigma * noise) * task.process.prediction.get_input_scale((alpha, sigma))
    time = schedule.model_time(t)
    teacher = task.teacher
    given, blank = teacher._conditions(params[TEACHER], batch, drop_key)
    variables = teacher.model_variables(params[TEACHER])
    null = jax.tree.map(lambda value, empty: jnp.broadcast_to(empty, value.shape), given,
                        aligned_conditions(given, blank))
    conditional = teacher.model.apply(variables, x, time, **given)
    unconditional = teacher.model.apply(variables, x, time, **null)
    target = unconditional + scale.reshape(-1, 1, 1, 1) * (conditional - unconditional)
    student_given, _ = task._conditions(params, batch, drop_key)
    output = task.model.apply(task.model_variables(params), x, time, **with_guidance(student_given, scale))
    # Weighted per row as the process weighs its times: one for flow matching, EDM's lambda(sigma).
    expected = jnp.mean(0.5 * jnp.square(output - target) * task.process.weight(t).reshape(-1, 1, 1, 1))
    assert float(loss.total / loss.mass) == pytest.approx(float(expected), rel=1e-5)
    modules = [program.module.clone() for program in task.program_key()]
    task.substitute(modules)
    assert all(program.module is module for program, module in zip(task.program_key(), modules, strict=True))


def test_a_saved_student_samples_one_branch_at_its_conditioners_guidance(runs, tmp_path):
    _, config, batch = runs
    task, state, _ = saved_after_one_step(config, batch, tmp_path / "student", 1e-2)
    restored = TextToImage.from_run(str(tmp_path / "student"))
    assert restored.guidance is None and TEACHER not in restored.variables
    expected = task.pipeline(state, ema=False)(["a red bird"], key=9).host().images
    np.testing.assert_array_equal(restored(["a red bird"], key=9).host().images, expected)
