"""One convention a model is trained and sampled with."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.diffusion.schedules import NoiseScheduler
from dew.diffusion.transforms import (
    PredictionTransform, ScheduleWeighting, Weighting, broadcast_rates,
)
from dew.objectives.base import Variables


@dataclass(frozen=True)
class Process:
    """A schedule, what the model predicts on it, and how the loss is weighted.

    `sampling` is the schedule inference integrates when it is not the training
    one (EDM trains on log-normal sigmas and samples on the Karras grid); None
    means the same schedule.
    """

    schedule: NoiseScheduler
    prediction: PredictionTransform
    weighting: Weighting = ScheduleWeighting()
    sampling: NoiseScheduler | None = None

    @property
    def sampler_schedule(self) -> NoiseScheduler:
        return self.schedule if self.sampling is None else self.sampling

    def weight(self, t) -> jax.Array:
        return self.weighting(self.schedule, self.prediction, t)

    def times(self, steps: int) -> jax.Array:
        """The descending time grid of `steps` points a sampler walks, from
        T to 0. A tabulated schedule cannot take more steps than it has
        entries, so `steps` is capped at T there."""
        schedule = self.sampler_schedule
        if schedule.T > 1:
            steps = min(steps, int(schedule.T))
        return jnp.linspace(schedule.T, 0.0, steps, dtype=jnp.float32)

    def noise(self, key, shape) -> jax.Array:
        """Draw the sampling schedule's Gaussian prior for the requested shape.

        Its default scale is the unit-data marginal at T; a schedule may
        declare a different prior normalization.
        """
        return jax.random.normal(key, shape) * self.sampler_schedule.prior_scale()

    def denoiser(self, model, params, conditions: Mapping[str, Any],
                 unconditional: Mapping[str, Any] | None = None) -> Denoiser:
        """`(x_t, t) -> (x_0, epsilon)` for `model` under `params` with the
        given conditions, on the sampling schedule.

        `unconditional` carries the same keys with the unconditional values
        and is what classifier-free guidance interpolates against.
        """
        return Denoiser(self, model, params, dict(conditions),
                        None if unconditional is None else dict(unconditional))


@dataclass(frozen=True)
class Denoiser:
    """A model, its parameters and its conditions as one denoising function.

    A call is the model's raw output at `(x_t, t)` followed by the process's
    conversion of it into `(x_0, epsilon)`. The two are separate because a
    source's conversion is not always linear in the output: dynamic
    thresholding and sample clipping limit x_0, and a consistency boundary
    reads a function of it, so combining two raw outputs after conversion is
    not combining them before it. Guidance therefore reads `raw_both`,
    combines the raw outputs and converts once, the order a source pipeline
    runs its scheduler in.
    """

    process: Process
    model: nn.Module
    params: Variables
    conditions: dict[str, Any]
    unconditional: dict[str, Any] | None = None

    def _raw(self, x_t, t, conditions) -> jax.Array:
        """The model's own output at `(x_t, t)`, on the input scale and model
        time the process's parameterization asks for."""
        process = self.process
        rates = broadcast_rates(process.sampler_schedule, t, x_t)
        c_in = process.prediction.get_input_scale(rates)
        return self.model.apply(
            self.params, x_t * c_in, process.sampler_schedule.model_time(t), **conditions)

    def convert(self, x_t, t, output) -> tuple[jax.Array, jax.Array]:
        """`(x_0, epsilon)` read out of a raw model output at `(x_t, t)`."""
        process = self.process
        rates = broadcast_rates(process.sampler_schedule, t, x_t)
        preds = process.prediction.pred_transform(x_t, output, rates, t)
        return process.prediction.backward_diffusion(x_t, preds, rates)

    def __call__(self, x_t, t) -> tuple[jax.Array, jax.Array]:
        return self.convert(x_t, t, self._raw(x_t, t, self.conditions))

    def raw_both(self, x_t, t) -> tuple[jax.Array, jax.Array]:
        """The conditional and the unconditional raw outputs, in one model
        call over the doubled batch."""
        if self.unconditional is None:
            raise ValueError("guidance needs the unconditional conditions; pass "
                             "`unconditional` to Process.denoiser")
        batch = x_t.shape[0]
        doubled = jax.tree.map(
            lambda given, null: jnp.concatenate(
                [given, jnp.broadcast_to(null, given.shape)], axis=0),
            self.conditions, self.unconditional)
        output = self._raw(
            jnp.concatenate([x_t, x_t], axis=0), jnp.concatenate([t, t], axis=0), doubled)
        return output[:batch], output[batch:]


__all__ = ["Process", "Denoiser"]
