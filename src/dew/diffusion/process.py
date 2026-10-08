"""The diffusion process a model is trained and sampled with, and the denoiser that runs a model on it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
from flax import linen as nn, struct

from dew.diffusion.schedules import NoiseScheduler
from dew.diffusion.transforms import PredictionTransform, ScheduleWeighting, Weighting, broadcast_rates
from dew.objectives.base import Variables


@struct.dataclass
class DenoisingCondition:
    """Text conditioning in the form the published model families read it.

    - `context` holds the token states.
    - `pooled` is a pooled vector.
    - `time_ids` are the size and crop ids that the XL towers add.
    - `guidance` is the distilled guidance value, which a guidance-embedded
      transformer takes as a model input instead of running two guided
      branches.
    - `mask` is the `[B, tokens]` mask of the real token states in a
      right-padded `context`, read by a family that excludes padded keys.
    """

    context: jax.Array
    pooled: jax.Array | None = None
    time_ids: jax.Array | None = None
    guidance: jax.Array | None = None
    mask: jax.Array | None = None

    def aligned(self, given: DenoisingCondition) -> DenoisingCondition:
        """Return this conditioning with `given`'s distilled guidance value.

        The guidance value belongs to the row and not to its caption, so a
        dropped or unconditional caption keeps the guidance scale the
        checkpoint samples at.
        """
        return replace(self, guidance=given.guidance)


# One conditioning keyword's value: the record a text encoder produced, or the
# array a spatial condition adds beside it.
type Conditioning = DenoisingCondition | jax.Array


def aligned_conditions(conditions: Mapping[str, Conditioning],
                       unconditional: Mapping[str, Conditioning]) -> dict[str, Conditioning]:
    """Return `unconditional` with each row's own model inputs taken from `conditions`.

    A keyword must name a conditioning record on both sides or on neither.
    Array keywords, such as the spatial ones an inpainting source adds, pass
    through unchanged. A keyword that is a record on one side only means the
    caller paired two different conditionings, so it raises `ValueError`.
    """
    aligned: dict[str, Conditioning] = {}
    for key, null in unconditional.items():
        given = conditions.get(key)
        if isinstance(null, DenoisingCondition) and isinstance(given, DenoisingCondition):
            aligned[key] = null.aligned(given)
        elif isinstance(null, DenoisingCondition) or isinstance(given, DenoisingCondition):
            raise ValueError(f"conditioning {key!r} is a record on one side only")
        else:
            aligned[key] = null
    return aligned


@dataclass(frozen=True)
class Process:
    """Pairs a noise schedule with what the model predicts on it and how the loss is weighted.

    `sampling` is the schedule that inference integrates when it differs
    from the training schedule. For example, EDM trains on log-normal sigmas
    and samples on the Karras grid. None means sampling uses the training
    schedule.

    `interval` is whether the model predicts over an interval of time and
    not at an instant, as MeanFlow's average velocity and a shortcut model's
    step do. Such a model reads the interval's length in model time as the
    `duration` condition. At each step, `sample` gives the denoiser the
    interval to the next grid point (`Denoiser.spanning`), so a
    single-evaluation solver such as `Euler` covers the whole interval in
    one step. A zero duration gives the instantaneous prediction.
    """

    schedule: NoiseScheduler
    prediction: PredictionTransform
    weighting: Weighting = field(default_factory=ScheduleWeighting)
    sampling: NoiseScheduler | None = None
    interval: bool = False

    @property

    def sampler_schedule(self) -> NoiseScheduler:
        return self.schedule if self.sampling is None else self.sampling

    def weight(self, t) -> jax.Array:
        return self.weighting(self.schedule, self.prediction, t)

    def rates(self, t, *, like: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the sampling schedule's `(alpha, sigma)` at `t`, shaped to broadcast against `like`.

        `like` is a `[B, ...]` state.
        """
        return broadcast_rates(self.sampler_schedule, t, like)

    def times(self, steps: int) -> jax.Array:
        """Return the descending grid of `steps` times, from T to 0, that a solver steps through.

        A tabulated schedule cannot take more steps than it has entries, so
        `steps` is capped at T there.
        """
        schedule = self.sampler_schedule
        if schedule.T > 1:
            steps = min(steps, int(schedule.T))
        return jnp.linspace(schedule.T, 0.0, steps, dtype=jnp.float32)

    def noise(self, key, shape) -> jax.Array:
        """Draw an array of `shape` from the sampling schedule's Gaussian prior.

        By default the prior's standard deviation is that of the marginal at
        T for unit-variance data. A schedule may declare a different scale
        through `prior_scale`.
        """
        return jax.random.normal(key, shape) * self.sampler_schedule.prior_scale()

    def denoiser(self, model, params, conditions: Mapping[str, Conditioning],
                 unconditional: Mapping[str, Conditioning] | None = None) -> Denoiser:
        """Return the denoiser of `model` under `params` with the given conditions.

        The denoiser maps `(x_t, t)` to `(x_0, epsilon)` on the sampling
        schedule. `unconditional` has the same keys with the unconditional
        values, and classifier-free guidance interpolates against it.
        """
        return Denoiser(self, model, params, dict(conditions),
                        None if unconditional is None else dict(unconditional))


@dataclass(frozen=True)
class Denoiser:
    """Denoises with one model, its parameters and its conditions.

    A call computes the model's raw output at `(x_t, t)` and then the
    process's conversion of it into `(x_0, epsilon)`. The two steps are
    separate methods because a conversion is not always linear in the
    output. Dynamic thresholding and sample clipping limit x_0, and a
    consistency boundary computes a function of it, so combining two raw
    outputs after conversion gives a different result from combining them
    before. Guidance therefore calls `raw_both`, combines the raw outputs
    and converts once, in the same order a source pipeline runs its
    scheduler.
    """

    process: Process
    model: nn.Module
    params: Variables
    conditions: dict[str, Conditioning]
    unconditional: dict[str, Conditioning] | None = None

    def _raw(self, x_t, t, conditions) -> jax.Array:
        """The model's own output at `(x_t, t)`, on the input scale and model
        time the process's parameterization asks for."""
        process = self.process
        rates = process.rates(t, like=x_t)
        c_in = process.prediction.get_input_scale(rates)
        output = self.model.apply(
            self.params, x_t * c_in, process.sampler_schedule.model_time(t), **conditions)
        if not isinstance(output, jax.Array):
            raise TypeError("A diffusion model must return one prediction array")
        return output

    def convert(self, x_t, t, output) -> tuple[jax.Array, jax.Array]:
        """Return `(x_0, epsilon)` recovered from a raw model output at `(x_t, t)`."""
        process = self.process
        rates = process.rates(t, like=x_t)
        preds = process.prediction.pred_transform(x_t, output, rates, t)
        return process.prediction.backward_diffusion(x_t, preds, rates)

    def __call__(self, x_t, t) -> tuple[jax.Array, jax.Array]:
        return self.convert(x_t, t, self.raw(x_t, t))

    def spanning(self, t, t_next) -> Denoiser:
        """Return this denoiser over the interval from `t` to `t_next`.

        Every call of the returned denoiser passes the interval's length in
        model time to the model as the `duration` condition.
        """
        schedule = self.process.sampler_schedule
        duration = {"duration": schedule.model_time(t) - schedule.model_time(t_next)}
        return replace(
            self,
            conditions={**self.conditions, **duration},
            unconditional=None if self.unconditional is None else {**self.unconditional, **duration},
        )

    def raw(self, x_t, t) -> jax.Array:
        """Return the model's raw output at `(x_t, t)` under the conditions."""
        return self._raw(x_t, t, self.conditions)

    def raw_both(self, x_t, t) -> tuple[jax.Array, jax.Array]:
        """Return the conditional and the unconditional raw outputs.

        Both come from one model call over the doubled batch. Raises
        `ValueError` when the denoiser has no unconditional conditions.
        """
        if self.unconditional is None:
            raise ValueError("guidance needs the unconditional conditions; pass "
                             "`unconditional` to Process.denoiser")
        batch = x_t.shape[0]
        doubled = jax.tree.map(
            lambda given, null: jnp.concatenate(
                [given, jnp.broadcast_to(null, given.shape)], axis=0),
            self.conditions, aligned_conditions(self.conditions, self.unconditional))
        output = self._raw(
            jnp.concatenate([x_t, x_t], axis=0), jnp.concatenate([t, t], axis=0), doubled)
        return output[:batch], output[batch:]


__all__ = ["Denoiser", "DenoisingCondition", "Process", "aligned_conditions"]
