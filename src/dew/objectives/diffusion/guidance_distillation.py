"""Guidance distillation: a student that reads the guidance scale as an input.

Meng et al. 2023 ("On Distillation of Guided Diffusion Models", stage one)
train a student ε_s(z_t, w) to match the teacher's classifier-free guided
prediction at scale w, drawn per example: one student evaluation then
stands for the teacher's two. FLUX.1 [dev] ships such a student, its
guidance embedded beside the time; no training code is published for it,
so this follows the paper's equation.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.diffusion.process import DenoisingCondition, Process
from dew.diffusion.schedules import expand
from dew.diffusion.transforms import broadcast_rates
from dew.inputs import InputSpec, unit_range
from dew.objectives.base import Aux, Mean, Step, Variables
from dew.registry import objectives

from .objective import TEACHER, DiffusionObjective


def with_guidance(conditions: dict, scale: jax.Array) -> dict:
    """`conditions` with every conditioning record's guidance set to `scale`,
    one value per row."""
    return {keyword: replace(value, guidance=scale) if isinstance(value, DenoisingCondition) else value
            for keyword, value in conditions.items()}


def guided_target(conditional: jax.Array, unconditional: jax.Array, scale: jax.Array) -> jax.Array:
    """The teacher's guided raw output at a per-row scale,
    unconditional + w (conditional - unconditional)."""
    return unconditional + expand(scale, conditional) * (conditional - unconditional)


@objectives("guidance_distillation")
class GuidanceDistillationObjective(DiffusionObjective):
    """Distill `teacher`'s classifier-free guidance into this model.

    `teacher` is the teacher's objective and `teacher_variables` its whole
    variables tree, held frozen under `TEACHER`; the teacher reads its own
    conditioning of the batch, as its run trained on it. Each row draws a scale
    uniformly from `scales`; the student reads it as its conditioning
    record's `guidance` and regresses its raw output onto the teacher's
    guided raw output at that scale, both on the same noised sample, under
    the process's weighting. The two share the process and the data's
    latent geometry, and the student's conditioning must carry a guidance
    input, as a guidance-embedded Flux's does.
    """

    def __init__(self, model: nn.Module, process: Process, inputs: InputSpec, *,
                 teacher: DiffusionObjective, teacher_variables: Variables,
                 scales: tuple[float, float] = (1.0, 8.0), **kwargs):
        if (type(teacher.process.schedule) is not type(process.schedule)
                or type(teacher.process.prediction) is not type(process.prediction)):
            raise ValueError("guidance distillation regresses onto the teacher's raw output, so the "
                             "two share the process's schedule and prediction")
        unused = sorted(
            key for key in ("uncertainty", "alignment", "end_to_end") if kwargs.get(key) is not None
        )
        if unused:
            raise ValueError(f"guidance distillation trains on its own loss, which reads none of {unused}")
        kwargs.setdefault("guidance", None)
        super().__init__(model, process, inputs, **kwargs)
        if teacher.latent_shape != self.latent_shape:
            raise ValueError(f"the teacher denoises {teacher.latent_shape} and the student "
                             f"{self.latent_shape}; they share one latent geometry")
        self.teacher = teacher
        self.teacher_variables = teacher_variables
        self.scales = scales

    def held_variables(self) -> Variables:
        return {**super().held_variables(), TEACHER: self.teacher_variables}

    def init(self, key, variables: Variables | None = None) -> Variables:
        state = dict(super().init(key, variables))
        state.setdefault(TEACHER, self.teacher_variables)
        return state

    def loss(self, params, batch, step: Step):
        samples = unit_range(batch[self.inputs.sample.key])
        encode_key, drop_key, time_key, noise_key, scale_key = jax.random.split(step.key, 5)
        if self.autoencoder is not None:
            samples = self.autoencoder.encode(params["autoencoder"], samples, encode_key)
        count = samples.shape[0]
        low, high = self.scales
        scale = jax.random.uniform(scale_key, (count,), minval=low, maxval=high)
        schedule = self.process.schedule
        t = schedule.sample_t(time_key, count)
        noise = jax.random.normal(noise_key, samples.shape, dtype=jnp.float32)
        rates = broadcast_rates(schedule, t, samples)
        noisy, c_in, _ = self.process.prediction.forward_diffusion(samples, noise, rates)

        teacher = self.teacher
        frozen = jax.lax.stop_gradient(params[TEACHER])
        given, blank = teacher._conditions(frozen, batch, drop_key, dropout=False)
        denoise = teacher.process.denoiser(teacher.model, teacher.trainable(frozen), given, blank)
        conditional, unconditional = denoise.raw_both(noisy, t)
        target = jax.lax.stop_gradient(guided_target(conditional, unconditional, scale))

        conditions, _ = self._conditions(params, batch, drop_key, dropout=False)
        if not any(isinstance(value, DenoisingCondition) and value.guidance is not None
                   for value in conditions.values()):
            raise ValueError("the student's conditioning carries no guidance input to distill into")
        output = self.model.apply(self.trainable(params), noisy * c_in, schedule.model_time(t),
                                  **with_guidance(conditions, scale.astype(jnp.float32)), train=True,
                                  rngs={"dropout": jax.random.fold_in(step.key, 1)})
        assert isinstance(output, jax.Array)
        losses = optax.l2_loss(output, target)
        weighted = losses * expand(self.process.weight(t), losses)
        return Mean(jnp.sum(weighted), jnp.asarray(losses.size, jnp.float32)), Aux(metrics={})


__all__ = ["GuidanceDistillationObjective", "guided_target", "with_guidance"]
