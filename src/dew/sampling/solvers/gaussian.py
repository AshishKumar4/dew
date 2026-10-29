"""Posterior transfers and consistency updates on signal and noise rates."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax

from dew.nn.precision import divided, scaled

from .common import _check_endpoint_domain, _push, _unit_noise


def transition_variance(rates_t, rates_s):
    """The variance of the forward transition q(x_t | x_s) of a Gaussian process
    x_t = alpha_t x_0 + sigma_t eps, for s before t: sigma_t^2 - (alpha_t / alpha_s)^2 sigma_s^2, the
    noise the step from s to t adds. It holds for any schedule, variance preserving, exploding or a
    rectified flow alike, and is the forward step's beta, 1 - alpha_t^2 / alpha_s^2, only where
    alpha^2 + sigma^2 = 1. Taken as the product of the difference and the sum, it is exactly 0 where
    the two levels coincide, as a step of no length adds no noise; a difference of rounded
    squares need not be, once a compiler fuses one square into the subtraction."""
    (alpha_t, sigma_t), (alpha_s, sigma_s) = rates_t, rates_s
    forwarded = alpha_t / alpha_s * sigma_s
    return jnp.maximum((sigma_t - forwarded) * (sigma_t + forwarded), 0.0)


def posterior_std(rates_t, rates_s):
    """The standard deviation of the forward posterior q(x_s | x_t, x_0) for s before t:
    sigma_s sigma_{t|s} / sigma_t, with sigma_{t|s}^2 the transition's variance
    (`transition_variance`), on any Gaussian process."""
    sigma_t, sigma_s = rates_t[1], rates_s[1]
    return sigma_s / sigma_t * jnp.sqrt(transition_variance(rates_t, rates_s))


@struct.dataclass
class DDPM:
    """Exact ancestral sampler for the reverse diffusion SDE.

    One step draws from the forward posterior q(x_s | x_t, x_0) for
    x_t = alpha_t x_0 + sigma_t eps, written in signal and noise rates so it
    holds for any schedule and any step stride. The
    posterior mean is alpha_s x_0 + alpha_t sigma_s^2 / (alpha_s sigma_t) eps
    and its variance is sigma_s^2 (1 - alpha_t^2 sigma_s^2 / (alpha_s^2 sigma_t^2)).

    `variance` selects which of Diffusers 0.34.0's fixed `DDPMScheduler`
    posterior variances the draw uses. `"small"` is the posterior's own
    variance (`posterior_std`). `"large"` is the forward step's own noise
    (`transition_variance`), the wider choice from `Glide`, which Diffusers
    writes as the beta 1 - alpha_t^2 / alpha_s^2 of a variance-preserving
    table. Both are written in rates, so they are defined on any schedule.

    Neither choice adds noise on the step whose own time is the schedule's
    zero, where x_t is already the least noisy state the schedule holds; the
    source skips its draw at that time the same way. Elsewhere, when the
    target's alpha is one on a variance-preserving schedule, the wide noise
    has standard deviation sigma_t, the square root of the source's own
    `current_beta_t`.
    """

    variance: Literal["small", "large"] = struct.field(pytree_node=False, default="small")

    def __post_init__(self) -> None:
        if self.variance not in ("small", "large"):
            raise ValueError(f"DDPM's fixed variances are 'small' and 'large', not {self.variance}")

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        alpha_t, sigma_t = process.rates(t, like=x)
        alpha_s, sigma_s = process.rates(t_next, like=x)
        noise = _unit_noise(key, x)
        eps_coeff = (sigma_s ** 2 * alpha_t) / (sigma_t * alpha_s)
        if self.variance == "large":
            gamma = jnp.sqrt(transition_variance((alpha_t, sigma_t), (alpha_s, sigma_s)))
        else:
            gamma = posterior_std((alpha_t, sigma_t), (alpha_s, sigma_s))
        gamma = jnp.where(jnp.reshape(jnp.asarray(t, jnp.float32),
                                      (-1,) + (1,) * (x.ndim - 1)) > 0, gamma, 0.0)
        return alpha_s * denoised + eps_coeff * eps + noise * gamma, state


@struct.dataclass
class DDIM:
    """The DDIM update, where `eta` sets the stochasticity: 0 is deterministic, 1 is DDPM-like.

    DDIM is from Song et al. (2021). Diffusers 0.34.0's `DDIMScheduler` limits the clean prediction under
    `clip_sample` or `thresholding` and keeps the model's own output as its
    epsilon, so the direction term uses the unlimited one. That pairing comes
    from `SourceLimitedPrediction`, in the process's conversion.
    """

    eta: float = struct.field(pytree_node=False, default=0.0)

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        schedule = process.sampler_schedule
        target = t - schedule.step_interval(t, t_next)
        alpha_t, sigma_t = process.rates(t, like=x)
        alpha_s, sigma_s = process.rates(target, like=x)
        if self.eta > 0:
            # DDIM paper eq. 16: eta times the forward posterior's standard deviation, so eta=1 draws
            # from the DDPM posterior on any process. The direction term shrinks to keep the
            # marginal variance right.
            sigma_tilde = self.eta * posterior_std((alpha_t, sigma_t), (alpha_s, sigma_s))
            noise = _unit_noise(key, x)
            direction = jnp.sqrt(jnp.maximum(sigma_s ** 2 - sigma_tilde ** 2, 0.0))
            return alpha_s * denoised + direction * eps + sigma_tilde * noise, state
        return alpha_s * denoised + sigma_s * eps, state

def _pndm_step(x, eps, variances_t, variances_s):
    """PNDM's transfer (Liu et al. 2022, eq. 9), DDIM from t to s in the
    rationalized form Diffusers' `_get_prev_sample` evaluates, from the
    squared rates (`Process.variances`): the transfer reads their difference,
    which a beta table's cumulative products give more exactly than the
    squares of their rounded roots. The denominator's root of three variances
    is taken as a product of two roots, whose product of three would underflow
    where the signal is small (2^-180 from variances of 2^-100 and 2^-80)."""
    (alpha2_t, sigma2_t), (alpha2_s, sigma2_s) = variances_t, variances_s
    sample_coeff = (alpha2_s / alpha2_t) ** 0.5
    denominator = alpha2_t * sigma2_s ** 0.5 + (alpha2_t * sigma2_t) ** 0.5 * alpha2_s ** 0.5
    return sample_coeff * x - (alpha2_s - alpha2_t) * eps / denominator


@struct.dataclass
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

    skip_prk_steps: bool = struct.field(pytree_node=False, default=False)

    def init(self, x, times, process, *, key):
        _check_endpoint_domain(process, times, source=True,
                               reason="PNDM stage differences are divided by the source alpha")
        # The transfer is DDIM's rationalized for alpha^2 + sigma^2 = 1, Diffusers' beta tables: with
        # alpha 1 (variance exploding) its eps term vanishes and every step returns x unchanged.
        if times.shape[0] >= 2:
            with jax.ensure_compile_time_eval():
                alpha, sigma = process.sampler_schedule.rates(times)
                if not bool(jnp.all(jnp.abs(alpha ** 2 + sigma ** 2 - 1) < 1e-4)):
                    raise ValueError("PNDM's transfer is DDIM's for a variance-preserving process, "
                                     f"not {type(process.sampler_schedule).__name__}'s rates")
        return jnp.zeros((4, *x.shape), x.dtype), jnp.zeros((), jnp.int32)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        schedule = process.sampler_schedule
        rates_t = process.variances(t, like=x)
        rates_s = process.variances(t_next, like=x)
        interval = schedule.step_interval(t, t_next)
        rates_step = process.variances(t - interval, like=x)
        outputs, count = state
        outputs = _push(outputs, eps)
        e0, e1, e2, e3 = outputs[-1], outputs[-2], outputs[-3], outputs[-4]

        def runge_kutta(_):
            # Stage points: the k2 sample transfers to t - half and the k2/k3
            # evaluations sit at t_next + half; both are the midpoint unless the
            # schedule rounds its interval (Diffusers' integer stride // 2).
            half = schedule.half_interval(t, t_next)
            t_low, t_eval = t - half, t_next + half
            rates_low = process.variances(t_low, like=x)
            rates_eval = process.variances(t_eval, like=x)
            k2 = denoise(_pndm_step(x, eps, rates_t, rates_low), t_eval)[1]
            k3 = denoise(_pndm_step(x, k2, rates_t, rates_eval), t_eval)[1]
            k4 = denoise(_pndm_step(x, k3, rates_t, rates_s), t_next)[1]
            # Each stage weighed as Diffusers' step_prk weighs it, a multiplication by 1/6 or 1/3 in
            # fp32 rounded once to the stage's dtype, as torch's opmath does, summed in stage order: no
            # division.
            weighed = scaled(eps, 1 / 6) + scaled(k2, 1 / 3) + scaled(k3, 1 / 3) + scaled(k4, 1 / 6)
            return _pndm_step(x, weighed, rates_t, rates_s)

        def predictor_corrector(_):
            eps_next = denoise(_pndm_step(x, eps, rates_t, rates_step), t_next)[1]
            correction_source = process.variances(t_next + interval, like=x)
            return _pndm_step(x, (eps_next + eps) / 2, correction_source, rates_s)

        def adams(k: int):
            if k == 2:
                combined = (3 * e0 - e1) / 2
            elif k == 3:
                combined = divided(23 * e0 - 16 * e1 + 5 * e2, 12)
            else:
                combined = scaled(55 * e0 - 59 * e1 + 37 * e2 - 9 * e3, 1 / 24)
            return _pndm_step(x, combined, rates_t, rates_step)

        if self.skip_prk_steps:
            branches = [predictor_corrector] + [(lambda k: lambda _: adams(k))(k) for k in (2, 3, 4)]
            index = jnp.minimum(count, 3)
        else:
            branches = [runge_kutta, lambda _: adams(4)]
            index = jnp.where(count < 3, 0, 1)
        return lax.switch(index, branches, None), (outputs, count + 1)

@struct.dataclass
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
        noise = _unit_noise(key, x)
        stepped = jnp.where(count == steps - 1, denoised, alpha_s * denoised + sigma_s * noise)
        return stepped, (count + 1, steps)


@struct.dataclass
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

    eta: float = struct.field(pytree_node=False, default=0.3)

    def __post_init__(self) -> None:
        if not 0 <= self.eta <= 1:
            raise ValueError(f"TCD's eta is in [0, 1], not {self.eta}")

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        alpha_s, sigma_s = process.rates(t_next, like=x)
        alpha_mid, sigma_mid = process.rates((1 - self.eta) * t_next, like=x)
        noised = alpha_mid * denoised + sigma_mid * eps
        if self.eta == 0:
            return noised, state
        ratio = alpha_s / alpha_mid
        noise = _unit_noise(key, x)
        # The forward process's q(x_s | x_mid) (`transition_variance`). The reference's 1 - ratio^2
        # is its variance on a variance-preserving table only: with alpha 1 it adds no noise, and a
        # variance-exploding walk collapsed to zero.
        spread = jnp.sqrt(transition_variance((alpha_s, sigma_s), (alpha_mid, sigma_mid)))
        return ratio * noised + spread * noise, state
