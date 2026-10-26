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

from dew.batch import Variables, dicts
from dew.diffusion.presets import Preset
from dew.diffusion.process import DenoisingCondition, Process
from dew.diffusion.schedules import NoiseScheduler, expand
from dew.diffusion.transforms import PredictionTransform, broadcast_rates
from dew.inputs import InputSpec
from dew.lora import unadapted
from dew.nn.kernels.options import Held
from dew.nn.protocols import Lacks
from dew.objectives.base import TEACHER, Aux, ProgramModule, Step, model_rngs
from dew.registry import to_record
from dew.sampling.guidance import guide

from .objective import DiffusionObjective, _own_loss, teacher_weights


def with_guidance(conditions: dict, scale: jax.Array) -> dict:
    """Return `conditions` with every conditioning record's guidance set to `scale`, one value per row."""
    return {keyword: replace(value, guidance=scale) if isinstance(value, DenoisingCondition) else value
            for keyword, value in conditions.items()}


class GuidanceDistillationObjective(DiffusionObjective):
    """Distills a teacher's classifier-free guidance into this model.

    The teacher is `teacher`: an objective, as a run config rebuilds the
    teacher's run from its record (`DiffusionRunConfig.teacher_run`), or a
    model over this objective's process and inputs, by default this one
    without its adapter. Its whole variables tree is held frozen under
    `TEACHER`: the starting tree's when it holds one, as a saved student's
    does, else `teacher_variables`, the tree the teacher's run published,
    else the model the starting tree holds. The teacher reads its own
    conditioning of the batch, the conditioning its run trained on. Each row
    draws a scale uniformly from `scales`. The student reads that scale as its
    conditioning record's `guidance`, and regresses its raw output onto the
    teacher's guided raw output at that scale, both on the same noised sample,
    under the process's weighting. The two must share the process's schedule and prediction and
    the data's latent geometry, and the student's conditioning must have a
    guidance input, as a guidance-embedded Flux's does. Sampling is unguided.
    """

    def __init__(self, model: nn.Module, process: Process | Preset, inputs: InputSpec, *,
                 teacher: nn.Module | DiffusionObjective | None = None,
                 teacher_variables: Variables | None = None, scales: tuple[float, float] = (1.0, 8.0),
                 **kwargs):
        _own_loss("guidance distillation", kwargs)
        super().__init__(model, process, inputs, **kwargs)
        held = teacher_weights(self.variables, teacher_variables, whole=True)
        if isinstance(teacher, DiffusionObjective):
            built = teacher
        else:
            built = DiffusionObjective(unadapted(self.model) if teacher is None else teacher, self.process,
                                       self.inputs, autoencoder=self.autoencoder, ema_decay=None)
        if any(to_record(getattr(built.process, part), kind) != to_record(getattr(self.process, part), kind)
               for part, kind in (("schedule", NoiseScheduler), ("prediction", PredictionTransform))):
            raise ValueError("guidance distillation regresses onto the teacher's raw output, so the "
                             "two share the process's schedule and prediction, configured alike")
        if built.latent_shape != self.latent_shape:
            raise ValueError(f"the teacher denoises {built.latent_shape} and the student "
                             f"{self.latent_shape}; they share one latent geometry")
        self.teacher = built
        self.teacher_variables = dicts(held)
        self.scales = scales

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The student, then the teacher's modules, which the step does not train."""
        return (*super().program_key(),
                *(program._replace(trained=False) for program in self.teacher.program_key()))

    def towers(self) -> tuple[Held, ...]:
        """The student's towers, then the teacher's, whose own encoders condition its targets."""
        return (*super().towers(), *self.teacher.towers())

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        self.model, *teacher = modules
        self.teacher.substitute(teacher)

    def held_variables(self, *, beside: Variables | None = None) -> Variables:
        return {**super().held_variables(beside=beside), TEACHER: self.teacher_variables}

    def loss(self, variables, batch, step: Step):
        # Each example's scale, time and noise are its own (`Step.sample`).
        encode_key, drop_key, _, _, _ = jax.random.split(step.key, 5)
        samples = self.clean_samples(variables, batch, encode_key, step)
        count = samples.shape[0]
        low, high = self.scales
        scale = step.sample("guidance scale", lambda key: jax.random.uniform(key, (), jnp.float32, low, high),
                            rows=count)
        schedule = self.process.schedule
        t = step.sample("times", lambda key: schedule.sample_t(key, 1)[0], rows=count)
        noise = step.sample("noise", jax.random.normal, samples.shape[1:], dtype=jnp.float32, rows=count)
        rates = broadcast_rates(schedule, t, samples)
        noisy, c_in, _ = self.process.prediction.forward_diffusion(samples, noise, rates)

        teacher = self.teacher
        frozen = jax.lax.stop_gradient(variables[TEACHER])
        # The teacher's masked image is the student's: drawn by the same name and places, recorded once,
        # by the student's conditions below.
        unrecorded = replace(step, recorder=None, witness=None)
        given, blank = teacher._conditions(frozen, batch, drop_key, unrecorded)
        # The teacher reads the student's noised sample at the time drawn on the training schedule.
        denoise = teacher.process.denoiser(teacher.model, teacher.model_variables(frozen), given, blank,
                                           schedule=schedule)
        conditional, unconditional = denoise.raw_both(noisy, t)
        target = jax.lax.stop_gradient(guide(conditional, unconditional, scale))

        conditions, _ = self._conditions(variables, batch, drop_key, step)
        if not any(isinstance(value, DenoisingCondition) and value.guidance is not None
                   for value in conditions.values()):
            raise Lacks("guided", "the student's conditioning carries no guidance input to distill into")
        output = self.model.apply(self.model_variables(variables), noisy * c_in, schedule.model_time(t),
                                  **with_guidance(conditions, scale.astype(jnp.float32)), train=step.training,
                                  rngs=model_rngs(jax.random.fold_in(step.key, 1), training=step.training))
        assert isinstance(output, jax.Array)
        losses = optax.l2_loss(output, target)
        weighted = losses * expand(self.process.weight(t), losses)
        return self.row_mean(weighted, batch), Aux(metrics={})


__all__ = ["GuidanceDistillationObjective", "with_guidance"]
