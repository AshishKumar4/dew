"""One reverse step each, from t to t_next, given the model's denoising at t.

A solver is a value: what it needs between steps travels in its state,
which `init` builds from x_T and `step` threads through `sample`'s scan. The
rates of the sampling schedule come from `process`; a solver that needs
another evaluation of the model (Heun's corrector, RK4's stages) calls
`denoise`. A solver that integrates dx / dsigma = eps says so by refusing a
schedule whose alpha is not one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

import jax
import jax.numpy as jnp
from jax import lax

from dew.diffusion.process import Process
from dew.diffusion.schedules import GeneralizedNoiseScheduler
from dew.diffusion.transforms import broadcast_rates
from dew.registry import samplers

StateT = TypeVar("StateT", covariant=True)


class Solver(Protocol[StateT]):
    """A step of a sampler, and whatever it carries between steps.

    `StateT` is that carried value: nothing for a one-step solver, the
    previous model output for a multi-step one. It is a type parameter rather
    than an attribute so a solver's own state type is checked at its call
    sites.
    """

    def init(self, x) -> StateT: ...

    def step(self, x, t, t_next, denoised, eps, state, key, process,
             denoise) -> tuple[jax.Array, StateT]:
        """`x` at `t_next` from `x` at `t` and the model's `(denoised, eps)` at `t`."""
        ...


def _rates(process: Process, t, t_next, x):
    schedule = process.sampler_schedule
    return broadcast_rates(schedule, t, x), broadcast_rates(schedule, t_next, x)


def _sigma_integrator(name: str, process: Process) -> GeneralizedNoiseScheduler:
    schedule = process.sampler_schedule
    if not isinstance(schedule, GeneralizedNoiseScheduler):
        raise ValueError(
            f"{name} integrates dx/dsigma = eps, which holds only when alpha is 1, "
            f"so it needs a GeneralizedNoiseScheduler and not {type(schedule).__name__}")
    return schedule


@samplers("ddpm")
@dataclass(frozen=True)
class DDPM:
    """Exact ancestral sampler for the reverse diffusion SDE.

    One step draws from the forward posterior q(x_s | x_t, x_0) for
    x_t = alpha_t x_0 + sigma_t eps, written in signal and noise rates so it
    holds for any schedule and any step stride. The
    posterior mean is alpha_s x_0 + alpha_t sigma_s^2 / (alpha_s sigma_t) eps
    and its variance is sigma_s^2 (1 - alpha_t^2 sigma_s^2 / (alpha_s^2 sigma_t^2)).
    """

    State = tuple

    def init(self, x):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        (alpha_t, sigma_t), (alpha_s, sigma_s) = _rates(process, t, t_next, x)
        noise = jax.random.normal(key, x.shape, dtype=jnp.float32)
        eps_coeff = (sigma_s ** 2 * alpha_t) / (sigma_t * alpha_s)
        gamma = sigma_s * jnp.sqrt(
            1 - (alpha_t ** 2 / alpha_s ** 2) * (sigma_s ** 2 / sigma_t ** 2))
        return alpha_s * denoised + eps_coeff * eps + noise * gamma, state


@samplers("ddim")
@dataclass(frozen=True)
class DDIM:
    """DDIM (Song et al. 2021); `eta` is the stochasticity, 0 deterministic and
    1 DDPM-like."""

    eta: float = 0.0
    State = tuple

    def init(self, x):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        (alpha_t, sigma_t), (alpha_s, sigma_s) = _rates(process, t, t_next, x)
        if self.eta > 0:
            # DDIM paper eq. 16: eta=0 is deterministic DDIM, eta=1.0 approaches DDPM.
            # The direction term must shrink to keep the marginal variance right.
            sigma_tilde = self.eta * (sigma_s / sigma_t) * jnp.sqrt(
                jnp.maximum(1 - alpha_t ** 2 / alpha_s ** 2, 0.0))
            noise = jax.random.normal(key, x.shape)
            direction = jnp.sqrt(jnp.maximum(sigma_s ** 2 - sigma_tilde ** 2, 0.0))
            return alpha_s * denoised + direction * eps + sigma_tilde * noise, state
        return alpha_s * denoised + sigma_s * eps, state


@samplers("euler")
@dataclass(frozen=True)
class Euler:
    """The DDIM update written as an Euler step of the probability flow ODE."""

    State = tuple

    def init(self, x):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        (alpha_t, sigma_t), (alpha_s, sigma_s) = _rates(process, t, t_next, x)
        dt = sigma_s - sigma_t
        x_0_coeff = (alpha_t * sigma_s - alpha_s * sigma_t) / dt
        dx = (x - x_0_coeff * denoised) / sigma_t
        return x + dx * dt, state


@samplers("simplified_euler")
@dataclass(frozen=True)
class SimplifiedEuler:
    """Euler for the variance exploding forward process x_{t+1} = x_t + sigma_t eps_t,
    where the derivative is (x - x_0) / sigma. Integrates a
    `GeneralizedNoiseScheduler`."""

    State = tuple

    def init(self, x):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        _sigma_integrator("SimplifiedEuler", process)
        (_, sigma_t), (_, sigma_s) = _rates(process, t, t_next, x)
        dt = sigma_s - sigma_t
        dx = (x - denoised) / sigma_t
        return x + dx * dt, state


@samplers("euler_ancestral")
@dataclass(frozen=True)
class EulerAncestral:
    """Euler with the ancestral noise injection of k-diffusion, on a variance
    exploding schedule."""

    State = tuple

    def init(self, x):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        _sigma_integrator("EulerAncestral", process)
        (alpha_t, sigma_t), (alpha_s, sigma_s) = _rates(process, t, t_next, x)
        sigma_up = (sigma_s ** 2 * (sigma_t ** 2 - sigma_s ** 2) / sigma_t ** 2) ** 0.5
        sigma_down = (sigma_s ** 2 - sigma_up ** 2) ** 0.5
        dt = sigma_down - sigma_t
        x_0_coeff = (alpha_t * sigma_s - alpha_s * sigma_t) / (sigma_s - sigma_t)
        dx = (x - x_0_coeff * denoised) / sigma_t
        dW = jax.random.normal(key, x.shape) * sigma_up
        return x + dx * dt + dW, state


@samplers("heun")
@dataclass(frozen=True)
class Heun:
    """Heun's second order method (Karras et al. 2022, Algorithm 2): an Euler
    step, the derivative re-evaluated at its end, and the average of the two."""

    State = tuple

    def init(self, x):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        (alpha_t, sigma_t), (alpha_s, sigma_s) = _rates(process, t, t_next, x)
        dt = sigma_s - sigma_t
        x_0_coeff = (alpha_t * sigma_s - alpha_s * sigma_t) / dt
        dx_0 = (x - x_0_coeff * denoised) / sigma_t
        x_euler = x + dx_0 * dt

        denoised_next, _ = denoise(x_euler, t_next)
        # When sigma reaches 0 there is no derivative there, so the step stays
        # the Euler one.
        safe_sigma_s = jnp.where(sigma_s > 0, sigma_s, 1.0)
        dx_1 = (x_euler - x_0_coeff * denoised_next) / safe_sigma_s
        return jnp.where(sigma_s > 0, x + 0.5 * (dx_0 + dx_1) * dt, x_euler), state


@samplers("rk4")
@dataclass(frozen=True)
class RK4:
    """Classical Runge-Kutta over dx/dsigma = eps, on a variance exploding
    schedule; the stages at half steps read the model at the time the schedule
    maps that sigma back to."""

    State = tuple

    def init(self, x):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        schedule = _sigma_integrator("RK4", process)
        (_, sigma_t), (_, sigma_s) = _rates(process, t, t_next, x)
        dt = sigma_s - sigma_t

        def derivative(x_at, sigma):
            return denoise(x_at, schedule.t_of_sigma(sigma.reshape(-1)))[1]

        k1 = eps
        k2 = derivative(x + 0.5 * k1 * dt, sigma_t + 0.5 * dt)
        k3 = derivative(x + 0.5 * k2 * dt, sigma_t + 0.5 * dt)
        k4 = derivative(x + k3 * dt, sigma_t + dt)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6, state


@samplers("multistep_dpm")
@dataclass(frozen=True)
class MultiStepDPM:
    """A third order multistep integrator of dx/dsigma = eps on a variance
    exploding schedule, from finite differences of the last three eps."""

    State = tuple

    def init(self, x):
        coefficient = jnp.zeros((x.shape[0],) + (1,) * (x.ndim - 1), jnp.float32)
        return (jnp.zeros_like(x), coefficient, jnp.zeros_like(x), coefficient,
                jnp.zeros((), jnp.int32))

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        _sigma_integrator("MultiStepDPM", process)
        (_, sigma_t), (_, sigma_s) = _rates(process, t, t_next, x)
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


__all__ = ["Solver", "DDPM", "DDIM", "Euler", "SimplifiedEuler", "EulerAncestral",
           "Heun", "RK4", "MultiStepDPM"]
