"""Probability-flow and ancestral solvers that integrate in sigma."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from dew.diffusion.schedules import expand

from .common import _push, _sigma_integrator


def _euler_step(x, denoised, source, target):
    """One Euler step of the probability-flow ODE, in sigma.

    Returns `(x at the target level, the derivative, the x_0 coefficient,
    the interval)`. A second-order method re-evaluates the derivative at the
    end of this step and needs the coefficient and the interval it was taken
    over.
    """
    (alpha_t, sigma_t), (alpha_s, sigma_s) = source, target
    dt = sigma_s - sigma_t
    x_0_coeff = (alpha_t * sigma_s - alpha_s * sigma_t) / dt
    derivative = (x - x_0_coeff * denoised) / sigma_t
    return x + derivative * dt, derivative, x_0_coeff, dt


def _ancestral(sigma_t, sigma_s):
    """k-diffusion's `get_ancestral_step` at eta 1: `(sigma_down, sigma_up)`,
    the level a deterministic step goes down to and the fresh noise that
    brings the marginal back to sigma_s."""
    sigma_up = (sigma_s ** 2 * (sigma_t ** 2 - sigma_s ** 2) / sigma_t ** 2) ** 0.5
    return (sigma_s ** 2 - sigma_up ** 2) ** 0.5, sigma_up

@dataclass(frozen=True)
class Euler:
    """The DDIM update written as an Euler step of the probability flow ODE.
    On a variance exploding schedule it is dx/dsigma = eps."""

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        stepped, _, _, _ = _euler_step(x, denoised, process.rates(t, like=x), process.rates(t_next, like=x))
        return stepped, state


@dataclass(frozen=True)
class EulerAncestral:
    """Euler with k-diffusion's ancestral noise injection (`get_ancestral_step`, eta 1).

    Each step goes down to sigma_down, then adds fresh noise of standard
    deviation sigma_up to bring the marginal back to sigma_s. It integrates a
    `GeneralizedNoiseScheduler`.
    """

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        _sigma_integrator("EulerAncestral", process)
        _, sigma_t = process.rates(t, like=x)
        _, sigma_s = process.rates(t_next, like=x)
        sigma_down, sigma_up = _ancestral(sigma_t, sigma_s)
        dx = (x - denoised) / sigma_t
        dW = jax.random.normal(key, x.shape) * sigma_up
        return x + dx * (sigma_down - sigma_t) + dW, state


@dataclass(frozen=True)
class Heun:
    """Heun's second-order method: an Euler step, the derivative re-evaluated at its end, and their average.

    This is Algorithm 2 of Karras et al. (2022).

    `s_churn`, `s_tmin`, `s_tmax` and `s_noise` are the algorithm's
    stochasticity. At a sigma in [s_tmin, s_tmax] each step first raises the
    noise level to sigma (1 + gamma), gamma = min(s_churn / N, sqrt(2) - 1)
    for an N-interval walk, by adding fresh noise of standard deviation
    s_noise sqrt(sigma_hat^2 - sigma^2), and then takes the Heun step from
    there. The churn moves sigma directly, so it needs a variance-exploding
    schedule, and a raised level past the schedule's top has no model time, so
    a sampling run that would churn there is refused. With churn, the step
    evaluates the model at the raised level, and the evaluation `sample` made
    at the grid point is unused, so the compiler removes it.

    Diffusers 0.34.0's `HeunDiscreteScheduler` limits the clean prediction of
    both stages under `clip_sample`; that limit belongs to the process's
    conversion, `SourceLimitedPrediction`, so both evaluations here read the
    limited prediction without the solver knowing about it.
    """

    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    s_noise: float = 1.0

    def init(self, x, times, process, *, key):
        if not self.s_churn:
            return ()
        schedule = _sigma_integrator("Heun's churn", process)
        gamma = min(self.s_churn / (times.shape[0] - 1), 2 ** 0.5 - 1)
        with jax.ensure_compile_time_eval():
            sigmas = schedule.sigmas(jnp.asarray(times[:-1], jnp.float32))
            churned = (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax)
            if bool(jnp.any(churned & (sigmas * (1 + gamma) > schedule.sigma_max))):
                raise ValueError(
                    f"the churn raises sigma past the schedule's top, {schedule.sigma_max}, "
                    f"where the model has no time; keep s_tmax below "
                    f"{schedule.sigma_max / (1 + gamma):.4g}")
        return jnp.asarray(gamma, jnp.float32)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        if self.s_churn:
            schedule = _sigma_integrator("Heun's churn", process)
            sigma = schedule.sigmas(t)
            gamma = jnp.where((sigma >= self.s_tmin) & (sigma <= self.s_tmax), state, 0.0)
            raised = sigma * (1 + gamma)
            noise = jax.random.normal(key, x.shape, dtype=jnp.float32)
            # sqrt(raised^2 - sigma^2), in a form that stays zero at gamma 0
            # under a fused multiply-add.
            spread = sigma * jnp.sqrt(gamma * (2 + gamma)) * self.s_noise
            x = x + expand(spread, x) * noise
            t = schedule.t_of_sigma(raised)
            denoised, _ = denoise(x, t)
        source = process.rates(t, like=x)
        target = process.rates(t_next, like=x)
        sigma_s = target[1]
        x_euler, dx_0, x_0_coeff, dt = _euler_step(x, denoised, source, target)

        denoised_next, _ = denoise(x_euler, t_next)
        # When sigma reaches 0 there is no derivative there, so the step is
        # the Euler one.
        safe_sigma_s = jnp.where(sigma_s > 0, sigma_s, 1.0)
        dx_1 = (x_euler - x_0_coeff * denoised_next) / safe_sigma_s
        return jnp.where(sigma_s > 0, x + 0.5 * (dx_0 + dx_1) * dt, x_euler), state


@dataclass(frozen=True)
class RK4:
    """Classical fourth-order Runge-Kutta over dx/dsigma = eps on a variance-exploding schedule.

    The half-step stages evaluate the model at the time the schedule maps
    their sigma back to. This is FlaxDiff's `RK4Sampler`.
    """

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        schedule = _sigma_integrator("RK4", process)
        _, sigma_t = process.rates(t, like=x)
        _, sigma_s = process.rates(t_next, like=x)
        dt = sigma_s - sigma_t

        def derivative(x_at, sigma):
            return denoise(x_at, schedule.t_of_sigma(sigma.reshape(-1)))[1]

        k1 = eps
        k2 = derivative(x + 0.5 * k1 * dt, sigma_t + 0.5 * dt)
        k3 = derivative(x + 0.5 * k2 * dt, sigma_t + 0.5 * dt)
        k4 = derivative(x + k3 * dt, sigma_t + dt)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6, state


@dataclass(frozen=True)
class KDPM2:
    """k-diffusion's DPM-Solver-2 (`sample_dpm_2`), the update of Diffusers
    0.34.0's `KDPM2DiscreteScheduler`, and with `ancestral` its
    `sample_dpm_2_ancestral` and `KDPM2AncestralDiscreteScheduler`.

    It takes an Euler step to the geometric midpoint of sigma_t and the target
    level, evaluates the model there, and steps from x with that midpoint
    derivative. The target is sigma_s, or under `ancestral` the sigma_down of
    k-diffusion's ancestral step with sigma_up of fresh noise added after.
    The midpoint's time comes from the schedule's `t_of_sigma`, so this
    integrates a `GeneralizedNoiseScheduler`.
    """

    ancestral: bool = False

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        schedule = _sigma_integrator("KDPM2", process)
        _, sigma_t = process.rates(t, like=x)
        _, sigma_s = process.rates(t_next, like=x)

        def midpoint(_):
            # The final Euler step has no midpoint evaluation at sigma zero.
            # Keep inactive rows at the source sigma for mixed-time batches.
            positive = sigma_s > 0
            following = jnp.where(positive, sigma_s, sigma_t)
            if self.ancestral:
                target, sigma_up = _ancestral(sigma_t, following)
                sigma_mid = jnp.exp(jnp.log(sigma_t) + 0.5 * (jnp.log(target) - jnp.log(sigma_t)))
            else:
                target, sigma_up = following, 0.0
                sigma_mid = jnp.exp(jnp.log(target) + 0.5 * (jnp.log(sigma_t) - jnp.log(target)))
            dx = (x - denoised) / sigma_t
            x_mid = x + dx * (sigma_mid - sigma_t)
            denoised_mid, _ = denoise(x_mid, schedule.t_of_sigma(sigma_mid.reshape(-1)))
            stepped = x + (x_mid - denoised_mid) / sigma_mid * (target - sigma_t)
            if self.ancestral:
                stepped = stepped + jax.random.normal(key, x.shape) * sigma_up
            return jnp.where(positive, stepped, denoised)

        return lax.cond(jnp.all(sigma_s == 0), lambda _: denoised, midpoint, None), state

@dataclass(frozen=True)
class MultiStepDPM:
    """A third order multistep integrator of dx/dsigma = eps on a variance
    exploding schedule, from finite differences of the last three eps:
    FlaxDiff's `MultiStepDPM`, which despite the name is not DPM-Solver."""

    def init(self, x, times, process, *, key):
        coefficient = jnp.zeros((x.shape[0],) + (1,) * (x.ndim - 1), jnp.float32)
        return (jnp.zeros_like(x), coefficient, jnp.zeros_like(x), coefficient,
                jnp.zeros((), jnp.int32))

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        _sigma_integrator("MultiStepDPM", process)
        _, sigma_t = process.rates(t, like=x)
        _, sigma_s = process.rates(t_next, like=x)
        dt = sigma_s - sigma_t
        last_eps, last_sigma, older_eps, older_sigma, count = state

        def first(_):
            return x + eps * dt

        def second(_):
            dx_2 = (eps - last_eps) / (sigma_t - last_sigma)
            return x + eps * dt + 0.5 * dx_2 * dt ** 2

        def third(_):
            dx_2 = (eps - last_eps) / (sigma_t - last_sigma)
            dx_2_last = (last_eps - older_eps) / (last_sigma - older_sigma)
            dx_3 = (dx_2 - dx_2_last) / (0.5 * ((sigma_t + last_sigma) - (last_sigma + older_sigma)))
            return x + eps * dt + 0.5 * dx_2 * dt ** 2 + dx_3 * dt ** 3 / 6

        next_x = lax.switch(jnp.minimum(count, 2), [first, second, third], None)
        return next_x, (eps, sigma_t, last_eps, last_sigma, count + 1)

def _lagrange_integral(nodes: list, j: int, a, b):
    """The integral from a to b of the Lagrange basis polynomial that is 1 at
    `nodes[j]` and 0 at the other nodes, in closed form."""
    polynomial = [1.0]
    scale = 1.0
    for m, node in enumerate(nodes):
        if m == j:
            continue
        polynomial = [0.0, *polynomial]
        polynomial = [polynomial[i] - node * (polynomial[i + 1] if i + 1 < len(polynomial) else 0.0)
                      for i in range(len(polynomial))]
        scale = scale * (nodes[j] - node)
    integral = sum(coefficient * (b ** (i + 1) - a ** (i + 1)) / (i + 1)
                   for i, coefficient in enumerate(polynomial))
    return integral / scale


@dataclass(frozen=True)
class LMS:
    """Linear multistep over dx/dsigma = (x - x_0) / sigma, k-diffusion's
    `sample_lms` and Diffusers 0.34.0's `LMSDiscreteScheduler`: the last
    `order` derivatives interpolated by the Lagrange polynomial through their
    sigmas and integrated over the step, in closed form where Diffusers
    quadratures; the order grows with the history. Integrates a
    `GeneralizedNoiseScheduler`.
    """

    order: int = 4

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError(f"LMS's order is at least 1, not {self.order}")

    def init(self, x, times, process, *, key):
        sigmas = jnp.ones((self.order, x.shape[0]) + (1,) * (x.ndim - 1), jnp.float32)
        return jnp.zeros((self.order, *x.shape), x.dtype), sigmas, jnp.zeros((), jnp.int32)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        _sigma_integrator("LMS", process)
        _, sigma_t = process.rates(t, like=x)
        _, sigma_s = process.rates(t_next, like=x)
        derivatives, sigmas, count = state
        derivatives = _push(derivatives, (x - denoised) / sigma_t)
        sigmas = _push(sigmas, sigma_t)

        def multistep(k: int):
            nodes = [sigmas[-(j + 1)] for j in range(k)]
            return x + sum(_lagrange_integral(nodes, j, sigma_t, sigma_s) * derivatives[-(j + 1)]
                           for j in range(k))

        branches = [(lambda k: lambda _: multistep(k))(k) for k in range(1, self.order + 1)]
        stepped = lax.switch(jnp.minimum(count, self.order - 1), branches, None)
        return stepped, (derivatives, sigmas, count + 1)
