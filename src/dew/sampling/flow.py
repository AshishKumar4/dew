# Copyright (c) 2025 Jie Liu
# SPDX-License-Identifier: MIT
"""Stochastic rectified-flow transitions and their Gaussian likelihoods.

Flow-GRPO (arXiv:2505.05470v5, equations 8-9) uses the diffusion
coefficient a * sqrt(t / (1 - t)), where a is `FlowSDE.noise_level`. At
t = 1, the reference implementation replaces the time in the denominator
with the first interior grid point. This module reads the equations as
that implementation does:
https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
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
from dew.diffusion.schedules import FlowMatchingScheduler, expand
from dew.diffusion.transforms import FlowMatchPredictionTransform

from .guidance import Guidance, Walk


@struct.dataclass
class GaussianTransition:
    """An isotropic Gaussian transition with one variance per batch row.

    Sampling and density arithmetic use float32, and densities and KL sum
    over the sample dimensions. A zero variance makes the transition
    deterministic. Sampling then returns the mean, and `log_prob` is NaN,
    because a Dirac measure has no density with respect to Lebesgue measure.
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
        """Return the joint log density of an observed next state, one value per row.

        Raises `ValueError` when `value` and the mean differ in shape.
        """
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
        """Return the KL divergence to a reference transition with mean `reference_mean`.

        The reference has the same variance, because the variance does not
        depend on the policy. Equal Dirac measures have KL zero, and
        distinct ones have infinite KL.
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


def flow_transition(x: ArrayLike, velocity: ArrayLike, sigma: ArrayLike,
                    sigma_next: ArrayLike, *, noise_level: float = 0.7) -> GaussianTransition:
    """Euler-Maruyama over the physical noise rate, with 0 <= sigma_next <= sigma <= 1.

    A rectified flow's noise rate is its own physical time, which is what
    `FlowSDE` reads off the schedule and hands over here. x and velocity are
    [batch, ...]; rates are scalars or [batch]. All density arithmetic is
    float32. Variance is sigma^2 times the elapsed rate. Invalid rates
    produce non-finite transitions. At zero noise or zero elapsed rate the
    result is deterministic.
    """
    if not math.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and non-negative")
    x = jnp.asarray(x, jnp.float32)
    velocity = jnp.asarray(velocity, jnp.float32)
    if x.ndim < 2 or velocity.shape != x.shape:
        raise ValueError("x and velocity must share a [batch, ...] sample shape")
    sigma = jnp.broadcast_to(jnp.asarray(sigma, jnp.float32), (x.shape[0],))
    sigma_next = jnp.broadcast_to(jnp.asarray(sigma_next, jnp.float32), sigma.shape)
    dt = sigma_next - sigma
    valid = (sigma_next >= 0) & (sigma <= 1) & (dt <= 0)
    denominator = jnp.where(dt == 0, 1, 1 - jnp.where(sigma == 1, sigma_next, sigma))
    # The diffusion coefficient squared over twice the rate has this finite
    # limit at sigma = 0.
    correction = noise_level**2 / (2 * denominator)
    mean = x * expand(1 + correction * dt, x) + velocity * expand(
        (1 + correction * (1 - sigma)) * dt, x)
    variance = noise_level**2 * sigma / denominator * -dt
    return GaussianTransition(jnp.where(expand(valid, x), mean, jnp.nan),
                              jnp.where(valid, variance, jnp.nan))


@dataclass(frozen=True)
class FlowSDE:
    """Samples a rectified-flow `Process` with Flow-GRPO's Euler-Maruyama solver.

    Process times may be resolution-shifted. The transition integrates in
    the noise rate sigma that results, as the reference scheduler does. A
    process without a rectified-flow schedule and velocity prediction raises
    `ValueError`.
    """

    noise_level: float = 0.7

    def validate(self, process: Process) -> None:
        schedule = process.sampler_schedule
        if not isinstance(schedule, FlowMatchingScheduler) or not isinstance(
                process.prediction, FlowMatchPredictionTransform):
            raise ValueError("FlowSDE requires a rectified-flow schedule and velocity prediction")
        if not math.isfinite(schedule.shift) or schedule.shift <= 0:
            raise ValueError("a rectified-flow timestep shift must be finite and positive")

    def init(self, x: jax.Array, times: jax.Array, process: Process, *, key: jax.Array) -> tuple[()]:
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

    def trajectory(self, denoise: Denoiser, x_T: jax.Array, steps: int, *,
                   guidance: Guidance | None = None, key: int | jax.Array) -> FlowTrajectory:
        """Record this solver's transitions over the same time grid and keys as `sample`.

        `steps` counts grid points, including both endpoints, so a
        ten-transition rollout uses steps=11 and fewer than 2 raises
        `ValueError`. Guidance is applied the same way before each Gaussian
        is built, and each step is guided or not exactly as in `sample`. The
        guidance must keep no state between steps, because each transition
        is later rescored on its own, so APG with momentum raises
        `ValueError`. Rectified flow's clean prediction at t=0 is its state,
        so the last transition already produces the final sample.
        """
        if steps < 2:
            raise ValueError("a trajectory needs at least two time points")
        from dew.nn.inputs import request_key
        key = request_key(key)
        process = denoise.process
        x_T = jnp.asarray(x_T, jnp.float32)
        walk = Walk.over(denoise, guidance, steps - 1)
        if jax.tree.leaves(walk.init(x_T)):
            raise ValueError("a flow trajectory's transitions are rescored one at a time, so its "
                             "guidance carries nothing between steps; APG's momentum does")
        times = process.times(steps)
        batch = x_T.shape[0]

        def body(x, inputs):
            t, t_next, index = inputs
            t, t_next = jnp.full((batch,), t), jnp.full((batch,), t_next)
            denoised, eps = walk.at((), index)(x, t)
            transition = self.transition(x, t, t_next, denoised, eps, process)
            following = transition.sample(jax.random.fold_in(key, index))
            return following, (following, transition.log_prob(following), transition.stochastic)

        _, (states, log_probs, stochastic) = jax.lax.scan(
            body, x_T, (times[:-1], times[1:], jnp.arange(steps - 1)))
        states = jnp.concatenate((x_T[:, None], jnp.swapaxes(states, 0, 1)), axis=1)
        return FlowTrajectory(states, times, log_probs.T, stochastic.T)


@struct.dataclass
class FlowTrajectory:
    """A reverse trajectory, with batch-major states and joint log densities.

    states is [batch, points, ...], times is [points], and log_probs and
    stochastic are [batch, points - 1]. Deterministic intervals have NaN
    log density and a false stochastic mark. The final state is the sample,
    which `samples` returns.
    """

    states: jax.Array
    times: jax.Array
    log_probs: jax.Array
    stochastic: jax.Array

    @property
    def samples(self) -> jax.Array:
        return self.states[:, -1]


__all__ = ["FlowSDE", "FlowTrajectory", "GaussianTransition"]

