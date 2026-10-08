"""Guidance distillation: a student that reads the guidance scale as an input.

Meng et al. 2023 ("On Distillation of Guided Diffusion Models", stage one)
train a student ε_s(z_t, w) to match the teacher's classifier-free guided
prediction at a scale w drawn per example. One student evaluation then
replaces the teacher's two. FLUX.1 [dev] ships such a student, with its
guidance embedded beside the time. No training code is published for it, so
this module follows the paper's equation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.diffusion.presets import Preset
from dew.diffusion.process import DenoisingCondition, Process
from dew.diffusion.schedules import expand
from dew.diffusion.transforms import broadcast_rates
from dew.inputs import InputSpec
from dew.lora import unadapted
from dew.objectives.base import TEACHER, Aux, ProgramModule, Step, Variables

from .objective import DiffusionObjective, _own_loss, teacher_weights


def with_guidance(conditions: dict, scale: jax.Array) -> dict:
    """Return `conditions` with every conditioning record's guidance set to `scale`, one value per row."""
    return {keyword: replace(value, guidance=scale) if isinstance(value, DenoisingCondition) else value
            for keyword, value in conditions.items()}


def guided_target(conditional: jax.Array, unconditional: jax.Array, scale: jax.Array) -> jax.Array:
    """Return the teacher's guided raw output at a per-row scale w.

    The output is unconditional + w (conditional - unconditional).
    """
    return unconditional + expand(scale, conditional) * (conditional - unconditional)


class GuidanceDistillationObjective(DiffusionObjective):
    """Distills a teacher's classifier-free guidance into this model.

    The teacher is the run in the directory `teacher_run`, rebuilt from its
    record, or, without one, the model `teacher` (by default this one without
    its adapter) over this objective's process and inputs. Its whole
    variables tree is held frozen under `TEACHER`: the starting tree's when it
    holds one, as a saved student's does, else the one the teacher run
    published. The teacher reads its own conditioning of the batch, the
    conditioning its run trained on. Each row draws a scale uniformly from
    `scales`. The student reads that scale as its conditioning record's
    `guidance`, and regresses its raw output onto the teacher's guided raw
    output at that scale, both on the same noised sample, under the process's
    weighting. The two must share the process's schedule and prediction and
    the data's latent geometry, and the student's conditioning must have a
    guidance input, as a guidance-embedded Flux's does. Sampling is unguided.
    """

    def __init__(self, model: nn.Module, process: Process | Preset, inputs: InputSpec, *,
                 teacher: nn.Module | None = None, teacher_run: str | None = None,
                 scales: tuple[float, float] = (1.0, 8.0), **kwargs):
        _own_loss("guidance distillation", kwargs)
        super().__init__(model, process, inputs, **kwargs)
        held = teacher_weights(self.variables, teacher_run, whole=True)
        if teacher_run is None:
            built = DiffusionObjective(unadapted(self.model) if teacher is None else teacher, self.process,
                                       self.inputs, autoencoder=self.autoencoder, ema_decay=None)
        elif teacher is not None:
            raise ValueError("the teacher run names its own model; give the run or the model")
        else:
            from .config import DiffusionRunConfig

            built = DiffusionRunConfig.load(teacher_run).build(variables=held)
        if (type(built.process.schedule) is not type(self.process.schedule)
                or type(built.process.prediction) is not type(self.process.prediction)):
            raise ValueError("guidance distillation regresses onto the teacher's raw output, so the "
                             "two share the process's schedule and prediction")
        if built.latent_shape != self.latent_shape:
            raise ValueError(f"the teacher denoises {built.latent_shape} and the student "
                             f"{self.latent_shape}; they share one latent geometry")
        self.teacher = built
        self.teacher_variables = held
        self.scales = scales

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The student, then the teacher's modules, which the step does not train."""
        return (*super().program_key(),
                *(program._replace(trained=False) for program in self.teacher.program_key()))

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        self.model, *teacher = modules
        self.teacher.substitute(teacher)

    def held_variables(self) -> Variables:
        return {**super().held_variables(), TEACHER: self.teacher_variables}

    def loss(self, variables, batch, step: Step):
        encode_key, drop_key, time_key, noise_key, scale_key = jax.random.split(step.key, 5)
        samples = self.clean_samples(variables, batch, encode_key)
        count = samples.shape[0]
        low, high = self.scales
        scale = jax.random.uniform(scale_key, (count,), minval=low, maxval=high)
        schedule = self.process.schedule
        t = schedule.sample_t(time_key, count)
        noise = jax.random.normal(noise_key, samples.shape, dtype=jnp.float32)
        rates = broadcast_rates(schedule, t, samples)
        noisy, c_in, _ = self.process.prediction.forward_diffusion(samples, noise, rates)

        teacher = self.teacher
        frozen = jax.lax.stop_gradient(variables[TEACHER])
        given, blank = teacher._conditions(frozen, batch, drop_key, dropout=False)
        denoise = teacher.process.denoiser(teacher.model, teacher.model_variables(frozen), given, blank)
        conditional, unconditional = denoise.raw_both(noisy, t)
        target = jax.lax.stop_gradient(guided_target(conditional, unconditional, scale))

        conditions, _ = self._conditions(variables, batch, drop_key, dropout=False)
        if not any(isinstance(value, DenoisingCondition) and value.guidance is not None
                   for value in conditions.values()):
            raise ValueError("the student's conditioning carries no guidance input to distill into")
        output = self.model.apply(self.model_variables(variables), noisy * c_in, schedule.model_time(t),
                                  **with_guidance(conditions, scale.astype(jnp.float32)), train=True,
                                  rngs={"dropout": jax.random.fold_in(step.key, 1)})
        assert isinstance(output, jax.Array)
        losses = optax.l2_loss(output, target)
        weighted = losses * expand(self.process.weight(t), losses)
        return self.row_mean(weighted, batch), Aux(metrics={})


__all__ = ["GuidanceDistillationObjective", "guided_target", "with_guidance"]
