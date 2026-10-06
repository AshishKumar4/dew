"""Guidance distillation: a student that reads the guidance scale as an input.

Meng et al. 2023 ("On Distillation of Guided Diffusion Models", stage one)
train a student ε_s(z_t, w) to match the teacher's classifier-free guided
prediction at a scale w drawn per example. One student evaluation then
replaces the teacher's two. FLUX.1 [dev] ships such a student, with its
guidance embedded beside the time. No training code is published for it, so
this module follows the paper's equation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.diffusion.process import DenoisingCondition, Process
from dew.diffusion.schedules import expand
from dew.diffusion.transforms import broadcast_rates
from dew.inputs import InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.objectives.base import Aux, ProgramModule, Step, Variables
from dew.registry import objectives, trainings

from .objective import TEACHER, DiffusionObjective, Training, _own_loss

if TYPE_CHECKING:
    from .config import DiffusionRunConfig


def with_guidance(conditions: dict, scale: jax.Array) -> dict:
    """Return `conditions` with every conditioning record's guidance set to `scale`, one value per row."""
    return {keyword: replace(value, guidance=scale) if isinstance(value, DenoisingCondition) else value
            for keyword, value in conditions.items()}


def guided_target(conditional: jax.Array, unconditional: jax.Array, scale: jax.Array) -> jax.Array:
    """Return the teacher's guided raw output at a per-row scale w.

    The output is unconditional + w (conditional - unconditional).
    """
    return unconditional + expand(scale, conditional) * (conditional - unconditional)


@trainings("guidance_distillation")
@dataclasses.dataclass(frozen=True)
class GuidanceDistillation(Training):
    """Distillation of a saved run's classifier-free guidance into this run's model.

    The student reads the guidance scale through its conditioning's guidance input
    (`GuidanceDistillationObjective`), so sampling runs one branch. `teacher` is the
    teacher run's directory, which a run must name, and `scales` is the range each
    row's scale is drawn from.
    """

    guided = False

    teacher: str = ""
    scales: tuple[float, float] = (1.0, 8.0)

    def __post_init__(self) -> None:
        low, high = (float(value) for value in self.scales)
        object.__setattr__(self, "scales", (low, high))

    def check(self, run: DiffusionRunConfig) -> None:
        super().check(run)
        if not self.teacher:
            raise ValueError("guidance distillation distills a teacher; name its run directory")

    def objective(self, run: DiffusionRunConfig, model: nn.Module, process: Process, inputs: InputSpec, *,
                  autoencoder: AutoEncoder | None,
                  variables: Variables | None) -> GuidanceDistillationObjective:
        """Return the student objective over the teacher run's objective and its variables.

        The teacher's variables are the copy a saved student tree holds when `variables` is
        given, and otherwise the weights the teacher run published.
        """
        from dew.checkpoints import Checkpoints

        from .config import DiffusionRunConfig

        held = (variables[TEACHER] if variables is not None else Checkpoints(self.teacher).variables(
            ema=None, step=None, mesh=None, layout=None, param_dtype=None))
        teacher = DiffusionRunConfig.load(self.teacher).build(variables=held)
        return GuidanceDistillationObjective(
            model, process, inputs, teacher=teacher, teacher_variables=held, scales=self.scales,
            autoencoder=autoencoder, variables=variables, ema_decay=run.ema_decay, solver=run.solver,
            guidance=None, steps=run.sampling_steps)


@objectives("guidance_distillation")
class GuidanceDistillationObjective(DiffusionObjective):
    """Distills `teacher`'s classifier-free guidance into this model.

    `teacher` is the teacher's objective, and `teacher_variables` is its whole
    variables tree, held frozen under `TEACHER`. The teacher reads its own
    conditioning of the batch, the conditioning its run trained on. Each row
    draws a scale uniformly from `scales`. The student reads that scale as its
    conditioning record's `guidance`, and regresses its raw output onto the
    teacher's guided raw output at that scale, both on the same noised
    sample, under the process's weighting. The two must share the process's
    schedule and prediction and the data's latent geometry, and the student's
    conditioning must have a guidance input, as a guidance-embedded Flux's
    does.
    """

    def __init__(self, model: nn.Module, process: Process, inputs: InputSpec, *,
                 teacher: DiffusionObjective, teacher_variables: Variables,
                 scales: tuple[float, float] = (1.0, 8.0), **kwargs):
        if (type(teacher.process.schedule) is not type(process.schedule)
                or type(teacher.process.prediction) is not type(process.prediction)):
            raise ValueError("guidance distillation regresses onto the teacher's raw output, so the "
                             "two share the process's schedule and prediction")
        _own_loss("guidance distillation", kwargs)
        kwargs.setdefault("guidance", None)
        super().__init__(model, process, inputs, **kwargs)
        if teacher.latent_shape != self.latent_shape:
            raise ValueError(f"the teacher denoises {teacher.latent_shape} and the student "
                             f"{self.latent_shape}; they share one latent geometry")
        self.teacher = teacher
        self.teacher_variables = teacher_variables
        self.scales = scales

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The student, then the teacher's modules, which the step does not train."""
        return (*super().program_key(),
                *(program._replace(trained=False) for program in self.teacher.program_key()))

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        count = len(super().program_key())
        super().substitute(modules[:count])
        self.teacher.substitute(modules[count:])

    def held_variables(self) -> Variables:
        return {**super().held_variables(), TEACHER: self.teacher_variables}

    def init(self, key, variables: Variables | None = None) -> Variables:
        state = dict(super().init(key, variables))
        state.setdefault(TEACHER, self.teacher_variables)
        return state

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
