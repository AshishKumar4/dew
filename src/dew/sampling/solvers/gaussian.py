"""Posterior transfers and consistency updates on signal and noise rates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax

from dew.registry import solvers

from .common import _check_endpoint_domain, _push


@solvers("ddpm")
@dataclass(frozen=True)
class DDPM:
    """Exact ancestral sampler for the reverse diffusion SDE.

    One step draws from the forward posterior q(x_s | x_t, x_0) for
    x_t = alpha_t x_0 + sigma_t eps, written in signal and noise rates so it
    holds for any schedule and any step stride. The
    posterior mean is alpha_s x_0 + alpha_t sigma_s^2 / (alpha_s sigma_t) eps
    and its variance is sigma_s^2 (1 - alpha_t^2 sigma_s^2 / (alpha_s^2 sigma_t^2)).

    `variance` selects which of Diffusers 0.34.0's fixed `DDPMScheduler`
    posterior variances the draw uses. `"small"` is the posterior's own
    variance, written in rates so it is defined on any schedule. `"large"` is
    the forward step's beta, 1 - alpha_t^2 / alpha_s^2, the wider choice from
    `Glide`. That is a variance-preserving quantity and it is zero wherever
    alpha is one, so a variance-exploding grid is refused rather than sampled
    without noise.

    Neither choice adds noise on the step whose own time is the schedule's
    zero, where x_t is already the least noisy state the schedule holds; the
    source skips its draw at that time the same way. Elsewhere, when the
    target's alpha is one on a variance-preserving schedule, the wide noise
    has standard deviation sigma_t, the square root of the source's own
    `current_beta_t`.
    """

    variance: Literal["small", "large"] = "small"

    def __post_init__(self) -> None:
        if self.variance not in ("small", "large"):
            raise ValueError(f"DDPM's fixed variances are 'small' and 'large', not {self.variance}")

    def init(self, x, times, process, *, key):
        if self.variance == "large":
            with jax.ensure_compile_time_eval():
                alpha, _ = process.sampler_schedule.rates(jnp.asarray(times, jnp.float32))
                if bool(jnp.all(alpha == 1)):
                    raise ValueError(
                        "DDPM's wide posterior variance is the variance-preserving forward "
                        "step's beta, which is zero on a schedule whose alpha is one")
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        alpha_t, sigma_t = process.rates(t, like=x)
        alpha_s, sigma_s = process.rates(t_next, like=x)
        noise = jax.random.normal(key, x.shape, dtype=jnp.float32)
        eps_coeff = (sigma_s ** 2 * alpha_t) / (sigma_t * alpha_s)
        if self.variance == "large":
            gamma = jnp.sqrt(1 - alpha_t ** 2 / alpha_s ** 2)
        else:
            gamma = sigma_s * jnp.sqrt(1 - (alpha_t ** 2 / alpha_s ** 2) * (sigma_s ** 2 / sigma_t ** 2))
        gamma = jnp.where(jnp.reshape(jnp.asarray(t, jnp.float32),
                                      (-1,) + (1,) * (x.ndim - 1)) > 0, gamma, 0.0)
        return alpha_s * denoised + eps_coeff * eps + noise * gamma, state


@solvers("ddim")
@dataclass(frozen=True)
class DDIM:
    """The DDIM update, where `eta` sets the stochasticity: 0 is deterministic, 1 is DDPM-like.

    DDIM is from Song et al. (2021). Diffusers 0.34.0's `DDIMScheduler` limits the clean prediction under
    `clip_sample` or `thresholding` and keeps the model's own output as its
    epsilon, so the direction term uses the unlimited one. That pairing comes
    from `SourceLimitedPrediction`, in the process's conversion.
    """

    eta: float = 0.0

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        schedule = process.sampler_schedule
        target = t - schedule.step_interval(t, t_next)
        alpha_t, sigma_t = process.rates(t, like=x)
        alpha_s, sigma_s = process.rates(target, like=x)
        if self.eta > 0:
            # DDIM paper eq. 16: eta=0 is deterministic DDIM, eta=1.0 approaches DDPM.
            # The direction term must shrink to keep the marginal variance right.
            sigma_tilde = self.eta * (sigma_s / sigma_t) * jnp.sqrt(
                jnp.maximum(1 - alpha_t ** 2 / alpha_s ** 2, 0.0))
            noise = jax.random.normal(key, x.shape)
            direction = jnp.sqrt(jnp.maximum(sigma_s ** 2 - sigma_tilde ** 2, 0.0))
            return alpha_s * denoised + direction * eps + sigma_tilde * noise, state
        return alpha_s * denoised + sigma_s * eps, state

def _pndm_step(x, eps, rates_t, rates_s):
    """PNDM's transfer (Liu et al. 2022, eq. 9), DDIM from t to s in the
    rationalized form Diffusers' `_get_prev_sample` evaluates."""
    (alpha_t, sigma_t), (alpha_s, sigma_s) = rates_t, rates_s
    denominator = alpha_t ** 2 * sigma_s + alpha_t * sigma_t * alpha_s
    return (alpha_s / alpha_t) * x - (alpha_s ** 2 - alpha_t ** 2) * eps / denominator


@solvers("pndm")
@dataclass(frozen=True)
class PNDM:
    """PNDM as Diffusers 0.34.0's `PNDMScheduler`: Adams-Bashforth over eps with DDIM as the transfer.

    PNDM is from Liu et al. (2022, arXiv 2202.09778). It is fourth order once
    four outputs are available. The warmup is the paper's,
    three pseudo Runge-Kutta steps of four model evaluations each (the
    stages at the interval's midpoint and end), which seed the history with
    each step's first eps; under `skip_prk_steps`, the PLMS form Stable
    Diffusion runs, the first step is a predictor-corrector pair (an Euler
    step, eps re-read at its end, the step retaken with the mean) and the
    orders grow from there. The schedule sets the transfer stride: ordinary
    native grids use their adjacent interval, while published integer grids
    retain their fixed training stride. Transfers cannot start at alpha = 0.
    """

    skip_prk_steps: bool = False

    def init(self, x, times, process, *, key):
        _check_endpoint_domain(process, times, source=True,
                               reason="PNDM stage differences are divided by the source alpha")
        return jnp.zeros((4, *x.shape), x.dtype), jnp.zeros((), jnp.int32)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        schedule = process.sampler_schedule
        rates_t = process.rates(t, like=x)
        rates_s = process.rates(t_next, like=x)
        interval = schedule.step_interval(t, t_next)
        rates_step = process.rates(t - interval, like=x)
        outputs, count = state
        outputs = _push(outputs, eps)
        e0, e1, e2, e3 = outputs[-1], outputs[-2], outputs[-3], outputs[-4]

        def runge_kutta(_):
            # Stage points: the k2 sample transfers to t - half and the k2/k3
            # evaluations sit at t_next + half; both are the midpoint unless the
            # schedule rounds its interval (Diffusers' integer stride // 2).
            half = schedule.half_interval(t, t_next)
            t_low, t_eval = t - half, t_next + half
            rates_low = process.rates(t_low, like=x)
            rates_eval = process.rates(t_eval, like=x)
            k2 = denoise(_pndm_step(x, eps, rates_t, rates_low), t_eval)[1]
            k3 = denoise(_pndm_step(x, k2, rates_t, rates_eval), t_eval)[1]
            k4 = denoise(_pndm_step(x, k3, rates_t, rates_s), t_next)[1]
            return _pndm_step(x, eps / 6 + k2 / 3 + k3 / 3 + k4 / 6, rates_t, rates_s)

        def predictor_corrector(_):
            eps_next = denoise(_pndm_step(x, eps, rates_t, rates_step), t_next)[1]
            correction_source = process.rates(t_next + interval, like=x)
            return _pndm_step(x, (eps_next + eps) / 2, correction_source, rates_s)

        def adams(k: int):
            if k == 2:
                combined = (3 * e0 - e1) / 2
            elif k == 3:
                combined = (23 * e0 - 16 * e1 + 5 * e2) / 12
            else:
                combined = (1 / 24) * (55 * e0 - 59 * e1 + 37 * e2 - 9 * e3)
            return _pndm_step(x, combined, rates_t, rates_step)

        if self.skip_prk_steps:
            branches = [predictor_corrector] + [(lambda k: lambda _: adams(k))(k) for k in (2, 3, 4)]
            index = jnp.minimum(count, 3)
        else:
            branches = [runge_kutta, lambda _: adams(4)]
            index = jnp.where(count < 3, 0, 1)
        return lax.switch(index, branches, None), (outputs, count + 1)

@solvers("consistency")
@dataclass(frozen=True)
class Consistency:
    """Multistep consistency sampling, the update of Diffusers 0.34.0's `LCMScheduler`.

    This is Algorithm 1 of Song et al. (2023). The clean prediction is noised
    again to the next level with fresh noise, x_s = alpha_s x_0 + sigma_s z,
    and the last step keeps x_0 as it is. A latent consistency
    model's x_0 is the consistency function's output, which
    `ConsistencyBoundary` reads out of the model's prediction.
    """

    def init(self, x, times, process, *, key):
        return jnp.zeros((), jnp.int32), jnp.asarray(times.shape[0] - 1, jnp.int32)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        alpha_s, sigma_s = process.rates(t_next, like=x)
        count, steps = state
        noise = jax.random.normal(key, x.shape, dtype=jnp.float32)
        stepped = jnp.where(count == steps - 1, denoised, alpha_s * denoised + sigma_s * noise)
        return stepped, (count + 1, steps)


@solvers("tcd")
@dataclass(frozen=True)
class TCD:
    """Trajectory consistency sampling, as Diffusers 0.34.0's `TCDScheduler`.

    This is from Zheng et al. (2024, arXiv 2402.19159). The deterministic DDIM
    step lands at (1 - eta) t_next, and the forward process noises it up to
    t_next with fresh noise; this is the paper's gamma-sampling with
    gamma = `eta`. At eta 0 it
    is DDIM; at eta 1 the step goes through the clean end of the schedule.
    A tabulated schedule reads the intermediate time at its truncated
    index, as the reference floors it. The last step of a walk lands on the
    grid's end itself and takes no noise.
    """

    eta: float = 0.3

    def __post_init__(self) -> None:
        if not 0 <= self.eta <= 1:
            raise ValueError(f"TCD's eta is in [0, 1], not {self.eta}")

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        alpha_s, _ = process.rates(t_next, like=x)
        alpha_mid, sigma_mid = process.rates((1 - self.eta) * t_next, like=x)
        noised = alpha_mid * denoised + sigma_mid * eps
        if self.eta == 0:
            return noised, state
        ratio = alpha_s / alpha_mid
        noise = jax.random.normal(key, x.shape, dtype=jnp.float32)
        return ratio * noised + jnp.sqrt(1 - ratio ** 2) * noise, state
