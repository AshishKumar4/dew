# Copyright (c) 2025 Jie Liu
# SPDX-License-Identifier: MIT
"""Stochastic rectified-flow transitions and their Gaussian likelihoods.

Flow-GRPO (arXiv:2505.05470v5, equations 8-9), read against
https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
uses diffusion coefficient a * sqrt(t / (1 - t)). At t=1, the reference
replaces the denominator's time with the first interior grid point.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct
from jax.typing import ArrayLike

from dew.diffusion.process import Denoiser, Process
from dew.diffusion.schedules import FlowMatchingScheduler
from dew.diffusion.transforms import FlowMatchPredictionTransform
from dew.registry import samplers

from .guidance import CFG


@struct.dataclass
class GaussianTransition:
    """An isotropic transition with one variance per batch row.

    Sampling and density arithmetic use float32. Densities and KL sum over
    the sample dimensions. A zero variance is a
    deterministic transition: sampling returns its mean and log_prob is NaN,
    since a Dirac measure has no density with respect to Lebesgue measure.
    """

    mean: jax.Array
    variance: jax.Array

    @property
    def stochastic(self) -> jax.Array:
        return self.variance > 0

    def sample(self, key: jax.Array) -> jax.Array:
        mean = jnp.asarray(self.mean, jnp.float32)
        variance = jnp.asarray(self.variance, jnp.float32)
        def draw():
            scale = jnp.sqrt(variance).reshape(
                (mean.shape[0],) + (1,) * (mean.ndim - 1))
            return mean + scale * jax.random.normal(key, mean.shape, dtype=jnp.float32)

        return jax.lax.cond(jnp.all(variance == 0), lambda: mean, draw)

    def log_prob(self, value: ArrayLike) -> jax.Array:
        """Joint log density of an observed next state, one value per row."""
        mean = jnp.asarray(self.mean, jnp.float32)
        variance = jnp.asarray(self.variance, jnp.float32)
        value = jnp.asarray(value, jnp.float32)
        if value.shape != mean.shape:
            raise ValueError("a transition density scores one next state per mean")
        safe_variance = jnp.where(variance > 0, variance, 1)
        error = jnp.square(value - mean).sum(axis=tuple(range(1, value.ndim)))
        dimensions = math.prod(value.shape[1:])
        log_prob = -0.5 * (error / safe_variance
                           + dimensions * jnp.log(2 * math.pi * safe_variance))
        return jnp.where(variance > 0, log_prob, jnp.nan)

    def kl(self, reference_mean: ArrayLike) -> jax.Array:
        """KL to a reference transition with the same policy-independent variance.

        Equal Dirac measures have KL zero; distinct ones have infinite KL.
        """
        mean = jnp.asarray(self.mean, jnp.float32)
        variance = jnp.asarray(self.variance, jnp.float32)
        reference_mean = jnp.asarray(reference_mean, jnp.float32)
        if reference_mean.shape != mean.shape:
            raise ValueError("reference and policy transition means must share one shape")
        error = jnp.square(mean - reference_mean).sum(axis=tuple(range(1, mean.ndim)))
        safe_variance = jnp.where(variance > 0, variance, 1)
        deterministic = jnp.where(error == 0, 0.0, jnp.inf)
        return jnp.where(variance > 0, error / (2 * safe_variance),
                         jnp.where(variance == 0, deterministic, jnp.nan))


def flow_transition(x: ArrayLike, velocity: ArrayLike, t: ArrayLike,
                    t_next: ArrayLike, *, noise_level: float = 0.7) -> GaussianTransition:
    """Euler-Maruyama from physical flow time t to t_next, with 0 <= t_next <= t <= 1.

    x and velocity are [batch, ...]; times are scalars or [batch]. All density
    arithmetic is float32. Variance is sigma(t)^2 * (t - t_next), including
    the elapsed time. Invalid times produce non-finite transitions. At zero
    noise or zero elapsed time the result is deterministic.
    """
    if not math.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and non-negative")
    x = jnp.asarray(x, jnp.float32)
    velocity = jnp.asarray(velocity, jnp.float32)
    if x.ndim < 2 or velocity.shape != x.shape:
        raise ValueError("x and velocity must share a [batch, ...] sample shape")
    t = jnp.broadcast_to(jnp.asarray(t, jnp.float32), (x.shape[0],))
    t_next = jnp.broadcast_to(jnp.asarray(t_next, jnp.float32), t.shape)
    dt = t_next - t
    valid = (t_next >= 0) & (t <= 1) & (dt <= 0)
    denominator = jnp.where(dt == 0, 1, 1 - jnp.where(t == 1, t_next, t))
    # sigma(t)^2 / (2t) has this finite limit at t=0.
    correction = noise_level**2 / (2 * denominator)
    expand = lambda value: value.reshape((x.shape[0],) + (1,) * (x.ndim - 1))
    mean = x * expand(1 + correction * dt) + velocity * expand(
        (1 + correction * (1 - t)) * dt)
    variance = noise_level**2 * t / denominator * -dt
    return GaussianTransition(jnp.where(expand(valid), mean, jnp.nan),
                              jnp.where(valid, variance, jnp.nan))



@samplers("flow_sde")
@dataclass(frozen=True)
class FlowSDE:
    """Flow-GRPO's Euler-Maruyama solver on a rectified-flow Process.

    Process times may be resolution-shifted. The transition integrates in
    the resulting physical noise rate, as the reference scheduler does.
    """

    noise_level: float = 0.7

    def __post_init__(self) -> None:
        if not math.isfinite(self.noise_level) or self.noise_level < 0:
            raise ValueError("noise_level must be finite and non-negative")

    def validate(self, process: Process) -> None:
        schedule = process.sampler_schedule
        if not isinstance(schedule, FlowMatchingScheduler) or not isinstance(
                process.prediction, FlowMatchPredictionTransform):
            raise ValueError("FlowSDE requires a rectified-flow schedule and velocity prediction")
        if not math.isfinite(schedule.shift) or schedule.shift <= 0:
            raise ValueError("a rectified-flow timestep shift must be finite and positive")

    def init(self, x: jax.Array, times: jax.Array, process: Process) -> tuple[()]:
        return ()

    def transition(self, x: jax.Array, t: jax.Array, t_next: jax.Array,
                   denoised: jax.Array, eps: jax.Array, process: Process) -> GaussianTransition:
        self.validate(process)
        _, sigma = process.sampler_schedule.rates(t)
        _, following = process.sampler_schedule.rates(t_next)
        return flow_transition(x, eps - denoised, sigma, following, noise_level=self.noise_level)

    def step(self, x: jax.Array, t: jax.Array, t_next: jax.Array,
             denoised: jax.Array, eps: jax.Array, state: tuple[()], key: jax.Array,
             process: Process,
             denoise: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
             /) -> tuple[jax.Array, tuple[()]]:
        return self.transition(x, t, t_next, denoised, eps, process).sample(key), state


@struct.dataclass
class FlowTrajectory:
    """A reverse trajectory, with batch-major states and joint log densities.

    states is [batch, points, ...], times is [points], and log_probs and
    stochastic are [batch, points - 1]. Deterministic intervals have NaN
    log density and a false stochastic mark. The final state is the sample.
    """

    states: jax.Array
    times: jax.Array
    log_probs: jax.Array
    stochastic: jax.Array

    @property
    def samples(self) -> jax.Array:
        return self.states[:, -1]


def sample_trajectory(denoise: Denoiser, x_T: jax.Array, steps: int, *,
                      solver: FlowSDE = FlowSDE(), guidance: CFG | None = None,
                      key: jax.Array) -> FlowTrajectory:
    """Record FlowSDE transitions over the same time grid and keys as sample.

    steps counts grid points, including both endpoints. A ten-transition
    rollout therefore uses steps=11. Guidance is applied identically before
    constructing each Gaussian. Rectified flow's clean prediction at t=0
    is its state, so the last transition already produces the final sample.
    """
    if steps < 2:
        raise ValueError("a trajectory needs at least two time points")
    process = denoise.process
    solver.validate(process)
    predict = denoise if guidance is None else guidance(denoise)
    times = process.times(steps)
    x_T = jnp.asarray(x_T, jnp.float32)
    batch = x_T.shape[0]

    def body(x, inputs):
        t, t_next, index = inputs
        t, t_next = jnp.full((batch,), t), jnp.full((batch,), t_next)
        denoised, eps = predict(x, t)
        transition = solver.transition(x, t, t_next, denoised, eps, process)
        following = transition.sample(jax.random.fold_in(key, index))
        return following, (following, transition.log_prob(following), transition.stochastic)

    _, (states, log_probs, stochastic) = jax.lax.scan(
        body, x_T, (times[:-1], times[1:], jnp.arange(steps - 1)))
    states = jnp.concatenate((x_T[:, None], jnp.swapaxes(states, 0, 1)), axis=1)
    return FlowTrajectory(states, times, log_probs.T, stochastic.T)


__all__ = ["GaussianTransition", "FlowSDE", "FlowTrajectory", "flow_transition", "sample_trajectory"]

