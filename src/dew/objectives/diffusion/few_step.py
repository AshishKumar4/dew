"""Few-step generators trained from scratch: MeanFlow.

MeanFlow (Geng et al. 2025, "Mean Flows for One-step Generative Modeling")
trains a model of the average velocity u(z_t, r, t) over [r, t] through the
MeanFlow identity u = v - (t - r) du/dt, the total derivative taken along
the flow with one JVP. One step of the average velocity then crosses the
whole interval. The official code is Gsunshine/meanflow's `MeanFlow`, in
JAX, which `tools/meanflow_reference.py` runs.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.diffusion.process import Process, aligned_conditions
from dew.diffusion.schedules import FlowMatchingScheduler, expand
from dew.diffusion.transforms import FlowMatchPredictionTransform, broadcast_rates
from dew.inputs import InputSpec, unit_range
from dew.objectives.base import Aux, Mean, Step
from dew.registry import objectives
from dew.sampling.solvers import Euler

from .objective import DiffusionObjective

Velocity = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
"""An average velocity u(z, t, r) over [r, t]."""


def intervals(t: jax.Array, r: jax.Array, instantaneous: float) -> tuple[jax.Array, jax.Array]:
    """MeanFlow's interval draw from two independent times: the later is t
    and the earlier r, and the first `instantaneous` fraction of the rows
    takes r = t, the plain flow-matching case (`sample_tr`)."""
    t, r = jnp.maximum(t, r), jnp.minimum(t, r)
    count = t.shape[0]
    return t, jnp.where(jnp.arange(count) < int(count * instantaneous), t, r)


def guided_velocity(v, unconditional, conditional, omega, kappa):
    """MeanFlow's training-time guidance (`guidance_fn`): the target
    velocity mixes the sample's v with the model's own unconditional and
    conditional velocities, omega v + (1 - omega - kappa) v_u + kappa v_c;
    kappa 0 is v_u + omega (v - v_u)."""
    return omega * v + (1 - omega - kappa) * unconditional + kappa * conditional


def mean_flow_target(velocity: Velocity, z, t, r, v) -> tuple[jax.Array, jax.Array]:
    """The model's average velocity at (z, t, r) and its regression target
    v - (t - r) du/dt, the derivative along (dz/dt, dt/dt, dr/dt) = (v, 1,
    0) by one JVP, with no gradient through the target."""
    u, derivative = jax.jvp(velocity, (z, t, r), (v, jnp.ones_like(t), jnp.zeros_like(r)))
    target = v - expand(jnp.clip(t - r, 0.0, 1.0), derivative) * derivative
    return u, jax.lax.stop_gradient(target)


def adaptive_loss(u, target, power: float, epsilon: float) -> jax.Array:
    """Each row's squared error summed over its entries, divided by the
    stopped (error + epsilon)^power: MeanFlow's adaptive weighting, whose
    power 0 is the plain error and power 1 a near-unit loss per row."""
    error = jnp.sum(jnp.square(u - target), axis=tuple(range(1, u.ndim)))
    return error / jax.lax.stop_gradient((error + epsilon) ** power)


@objectives("mean_flow")
class MeanFlowObjective(DiffusionObjective):
    """MeanFlow on an interval process (`presets.MeanFlow`).

    `instantaneous` is the fraction of rows trained at r = t (the
    reference's `data_proportion`, 0.75). `omega` and `kappa` are its
    training-time guidance, applied where t lies in `guidance_interval`;
    omega 1 and kappa 0 train without it. `norm_p` and `norm_eps` are the
    adaptive weighting's power and epsilon. The condition is dropped on
    `unconditional_prob` of the rows, whose target is then the unguided v.
    Sampling takes `steps - 1` Euler steps of the average velocity, one by
    default, with no guidance at sampling: it is trained in.
    """

    def __init__(self, model: nn.Module, process: Process, inputs: InputSpec, *,
                 instantaneous: float = 0.75, omega: float = 1.0, kappa: float = 0.0,
                 guidance_interval: tuple[float, float] = (0.0, 1.0), norm_p: float = 1.0,
                 norm_eps: float = 0.01, **kwargs):
        schedule = process.schedule
        if not (process.interval and isinstance(schedule, FlowMatchingScheduler) and schedule.shift == 1.0
                and isinstance(process.prediction, FlowMatchPredictionTransform)):
            raise ValueError("MeanFlow trains an interval model of velocity on the unshifted linear "
                             "path; build the process with presets.MeanFlow")
        kwargs.setdefault("guidance", None)
        kwargs.setdefault("sampler", Euler())
        kwargs.setdefault("steps", 2)
        super().__init__(model, process, inputs, **kwargs)
        self.instantaneous = instantaneous
        self.omega = omega
        self.kappa = kappa
        self.guidance_interval = guidance_interval
        self.norm_p = norm_p
        self.norm_eps = norm_eps

    def loss(self, params, batch, step: Step):
        samples = unit_range(batch[self.inputs.sample.key])
        encode_key, drop_key, time_key, noise_key, dropout_key = jax.random.split(step.key, 5)
        if self.autoencoder is not None:
            samples = self.autoencoder.encode(params["autoencoder"], samples, encode_key)
        count = samples.shape[0]
        schedule = self.process.schedule
        given, unconditional = self._conditions(params, batch, drop_key, dropout=False)
        blank = jax.tree.map(lambda value, null: jnp.broadcast_to(null, value.shape),
                             given, aligned_conditions(given, unconditional))
        later, earlier = jax.random.split(time_key)
        t, r = intervals(schedule.sample_t(later, count), schedule.sample_t(earlier, count),
                         self.instantaneous)
        noise = jax.random.normal(noise_key, samples.shape, dtype=jnp.float32)
        z, _, v = self.process.prediction.forward_diffusion(samples, noise, broadcast_rates(schedule, t, samples))
        variables = self.trainable(params)

        def velocity(conditions, *, train: bool) -> Velocity:
            def average(z, t, r):
                return self.model.apply(variables, z, schedule.model_time(t), **conditions,
                                        duration=schedule.model_time(t) - schedule.model_time(r),
                                        train=train, rngs={"dropout": dropout_key})
            return average

        if self.omega != 1.0 or self.kappa != 0.0:
            start, stop = self.guidance_interval
            inside = (t >= start) & (t <= stop)
            omega = expand(jnp.where(inside, self.omega, 1.0), v)
            kappa = expand(jnp.where(inside, self.kappa, 0.0), v)
            guided = jax.lax.stop_gradient(guided_velocity(
                v, velocity(blank, train=False)(z, t, t), velocity(given, train=False)(z, t, t), omega, kappa))
        else:
            guided = v
        dropped = jax.random.bernoulli(jax.random.fold_in(drop_key, 1), self.unconditional_prob, (count,))
        conditions = jax.tree.map(lambda value, null: jnp.where(expand(dropped, value), null, value),
                                  given, blank)
        guided = jnp.where(expand(dropped, v), v, guided)
        u, target = mean_flow_target(velocity(conditions, train=True), z, t, r, guided)
        losses = adaptive_loss(u, target, self.norm_p, self.norm_eps)
        return Mean(jnp.sum(losses), jnp.asarray(count, jnp.float32)), Aux(metrics={})


__all__ = ["MeanFlowObjective", "adaptive_loss", "guided_velocity", "intervals", "mean_flow_target"]
