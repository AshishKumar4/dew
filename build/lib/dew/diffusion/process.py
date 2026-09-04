"""One convention a model is trained and sampled with."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from dew.diffusion.schedules import NoiseScheduler, expand
from dew.diffusion.transforms import (
    PredictionTransform, ScheduleWeighting, Weighting, broadcast_rates,
)


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
        """x_T for `shape`: N(0, alpha_T^2 + sigma_T^2), the marginal at t = T
        of unit-variance data."""
        alpha, sigma = self.sampler_schedule.rates(jnp.asarray(self.sampler_schedule.T))
        return jax.random.normal(key, shape) * jnp.sqrt(alpha ** 2 + sigma ** 2)

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
    """A model, its parameters and its conditions as one denoising function."""

    process: Process
    model: Any
    params: Any
    conditions: dict[str, Any]
    unconditional: dict[str, Any] | None = None

    def _predict(self, x_t, t, conditions) -> tuple[jax.Array, jax.Array]:
        process = self.process
        rates = broadcast_rates(process.sampler_schedule, t, x_t)
        c_in = process.prediction.get_input_scale(rates)
        output = self.model.apply(
            self.params, x_t * c_in, process.sampler_schedule.model_time(t), **conditions)
        preds = process.prediction.pred_transform(x_t, output, rates)
        return process.prediction.backward_diffusion(x_t, preds, rates)

    def __call__(self, x_t, t) -> tuple[jax.Array, jax.Array]:
        return self._predict(x_t, t, self.conditions)

    def both(self, x_t, t) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        """The conditional and the unconditional prediction, in one model call
        over the doubled batch."""
        if self.unconditional is None:
            raise ValueError("guidance needs the unconditional conditions; pass "
                             "`unconditional` to Process.denoiser")
        batch = x_t.shape[0]
        doubled = jax.tree.map(
            lambda given, null: jnp.concatenate(
                [given, jnp.broadcast_to(null, given.shape)], axis=0),
            self.conditions, self.unconditional)
        x_0, eps = self._predict(
            jnp.concatenate([x_t, x_t], axis=0), jnp.concatenate([t, t], axis=0), doubled)
        return (x_0[:batch], eps[:batch]), (x_0[batch:], eps[batch:])


__all__ = ["Process", "Denoiser", "expand"]
