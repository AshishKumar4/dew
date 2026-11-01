"""Probability-flow and ancestral solvers that integrate in sigma."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax

from dew.diffusion.schedules import expand
from dew.diffusion.schedules.common import ancestral_variance
from dew.nn.precision import divided

from .common import _check_gaps, _grid_levels, _history_gaps, _push, _sigma_integrator, _unit_noise


def _euler_step(x, denoised, source, target):
    """One Euler step of the probability-flow ODE, in sigma.

    Returns `(x at the target level, the derivative, the x_0 coefficient,
    the interval)`. A second-order method re-evaluates the derivative at the
    end of this step and needs the coefficient and the interval it was taken
    over. A step between equal levels, which a grid that ends where its
    final sigma repeats takes (`final_sigmas_type="sigma_min"`), has no
    length and leaves x as it is, as the source's `x + derivative * dt` does.
    """
    (alpha_t, sigma_t), (alpha_s, sigma_s) = source, target
    dt = sigma_s - sigma_t
    # Equal levels make the numerator 0 too; any finite coefficient is scaled by the zero interval.
    x_0_coeff = (alpha_t * sigma_s - alpha_s * sigma_t) / jnp.where(dt == 0, 1, dt)
    derivative = divided(x - x_0_coeff * denoised, sigma_t)
    # The source's x + derivative dt, summed as alpha_s x_0 + sigma_s eps, DDIM's form of it: from a
    # large sigma the source's sum cancels two terms of x's size (x is 4e4 at a zero-SNR sigma of
    # 4096) and keeps dt's rounding times the derivative, and near pure noise on a VP schedule the
    # secant coefficient is in the thousands; this form has neither, and a step to sigma 0 lands on
    # the clean prediction itself. A step of no length leaves x exactly as it is.
    stepped = alpha_s * denoised + divided(x - alpha_t * denoised, sigma_t) * sigma_s
    return jnp.where(dt == 0, x, stepped), derivative, x_0_coeff, dt


def _ancestral(sigma_t, sigma_s):
    """k-diffusion's `get_ancestral_step` at eta 1: `(sigma_down, sigma_up)`,
    the level a deterministic step goes down to and the fresh noise that
    brings the marginal back to sigma_s, sigma_up capped at sigma_s as the
    source caps it."""
    sigma_up = jnp.minimum(sigma_s, ancestral_variance(sigma_t, sigma_s) ** 0.5)
    return (sigma_s ** 2 - sigma_up ** 2) ** 0.5, sigma_up

@struct.dataclass
class Euler:
    """The DDIM update written as an Euler step of the probability flow ODE.
    On a variance exploding schedule it is dx/dsigma = eps."""

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        stepped, _, _, _ = _euler_step(x, denoised, process.rates(t, like=x), process.rates(t_next, like=x))
        return stepped, state


@struct.dataclass
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
        dW = _unit_noise(key, x) * sigma_up
        return x + dx * (sigma_down - sigma_t) + dW, state


@struct.dataclass
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

    s_churn: float = struct.field(pytree_node=False, default=0.0)
    s_tmin: float = struct.field(pytree_node=False, default=0.0)
    s_tmax: float = struct.field(pytree_node=False, default=float("inf"))
    s_noise: float = struct.field(pytree_node=False, default=1.0)

    def init(self, x, times, process, *, key):
        if not self.s_churn:
            return ()
        schedule = _sigma_integrator("Heun's churn", process)
        gamma = min(self.s_churn / (times.shape[0] - 1), 2 ** 0.5 - 1)
        with jax.ensure_compile_time_eval():
            sigmas = schedule.sigmas(jnp.asarray(times[:-1], jnp.float32))
            churned = (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax)
            top = schedule.sigma_range[1]
            if bool(jnp.any(churned & (sigmas * (1 + gamma) > top))):
                raise ValueError(
                    f"the churn raises sigma past the schedule's top, {top}, "
                    f"where the model has no time; keep s_tmax below "
                    f"{top / (1 + gamma):.4g}")
        return jnp.asarray(gamma, jnp.float32)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        if self.s_churn:
            schedule = _sigma_integrator("Heun's churn", process)
            sigma = schedule.sigmas(t)
            gamma = jnp.where((sigma >= self.s_tmin) & (sigma <= self.s_tmax), state, 0.0)
            raised = sigma * (1 + gamma)
            noise = _unit_noise(key, x)
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

        def corrected(operand):
            denoised_next, _ = denoise(x_euler, t_next)
            # When sigma reaches 0 there is no derivative there, so a row
            # there takes the Euler step.
            safe_sigma_s = jnp.where(sigma_s > 0, sigma_s, 1.0)
            dx_1 = (x_euler - x_0_coeff * denoised_next) / safe_sigma_s
            return jnp.where(sigma_s > 0, x + 0.5 * (dx_0 + dx_1) * dt, x_euler)

        # The last step of a trajectory to sigma 0 is the Euler one for every
        # row, so its corrector's model call is skipped, as KDPM2's is.
        return lax.cond(jnp.all(sigma_s == 0), lambda _: x_euler, corrected, None), state


@struct.dataclass
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
        return x + divided((k1 + 2 * k2 + 2 * k3 + k4) * dt, 6), state


@struct.dataclass
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

    ancestral: bool = struct.field(pytree_node=False, default=False)

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
                stepped = stepped + _unit_noise(key, x) * sigma_up
            return jnp.where(positive, stepped, denoised)

        return lax.cond(jnp.all(sigma_s == 0), lambda _: denoised, midpoint, None), state

@struct.dataclass
class MultiStepDPM:
    """A third order multistep integrator of dx/dsigma = eps on a variance
    exploding schedule, from finite differences of the last three eps:
    FlaxDiff's `MultiStepDPM`, which despite the name is not DPM-Solver."""

    def init(self, x, times, process, *, key):
        # The order grows by one a step, to three.
        orders = [min(index, 2) + 1 for index in range(times.shape[0] - 1)]
        _check_gaps(_grid_levels(process, times)[1], _history_gaps(orders),
                    reason="its finite differences divide by")
        # The history in the dtype the rates take at the run's times, as each step carries them, so a
        # float64 walk scans in float64 (ledger 2063).
        sigma = process.rates(times[0], like=x)[1]
        coefficient = jnp.zeros((x.shape[0],) + (1,) * (x.ndim - 1), sigma.dtype)
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
            return x + eps * dt + 0.5 * dx_2 * dt ** 2 + divided(dx_3 * dt ** 3, 6)

        next_x = lax.switch(jnp.minimum(count, 2), [first, second, third], None)
        return next_x, (eps, sigma_t, last_eps, last_sigma, count + 1)

def _lagrange_integral(nodes: list, j: int, a, b):
    """The integral from a to b of the Lagrange basis polynomial that is 1 at
    `nodes[j]` and 0 at the other nodes, in closed form on the step itself:
    with x = a + (b - a) s, each factor (x - node_m) / (node_j - node_m) is
    alpha_m + beta_m s, alpha_m = (a - node_m) / (node_j - node_m) and beta_m =
    (b - a) / (node_j - node_m), ratios of the grid's own spacings, so it is
    (b - a) times the integral over s in [0, 1] of their product. No power of
    a sigma is formed (VE sigmas near 2**70 cube past float32's largest), no
    spacing is measured against a wider one (a tiny step beside a distant
    node keeps its coefficient), and an empty step integrates to zero with
    the basis at its end as its derivative."""
    width = b - a
    polynomial = [1.0]
    for m, node in enumerate(nodes):
        if m == j:
            continue
        alpha, beta = (a - node) / (nodes[j] - node), width / (nodes[j] - node)
        polynomial = [(polynomial[i] if i < len(polynomial) else 0.0) * alpha
                      + (polynomial[i - 1] * beta if i > 0 else 0.0) for i in range(len(polynomial) + 1)]
    return width * sum(coefficient / (i + 1) for i, coefficient in enumerate(polynomial))


@struct.dataclass
class LMS:
    """Linear multistep over dx/dsigma = (x - x_0) / sigma, k-diffusion's
    `sample_lms` and Diffusers 0.34.0's `LMSDiscreteScheduler`: the last
    `order` derivatives interpolated by the Lagrange polynomial through their
    sigmas and integrated over the step, in closed form where Diffusers
    quadratures; the order grows with the history. Integrates a
    `GeneralizedNoiseScheduler`.
    """

    order: int = struct.field(pytree_node=False, default=4)

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError(f"LMS's order is at least 1, not {self.order}")

    def init(self, x, times, process, *, key):
        orders = [min(index, self.order - 1) + 1 for index in range(times.shape[0] - 1)]
        _check_gaps(_grid_levels(process, times)[1], _history_gaps(orders),
                    reason="LMS's Lagrange basis divides by")
        # The sigmas in the dtype the rates take at the run's times, as each step pushes them (2063).
        sigma = process.rates(times[0], like=x)[1]
        sigmas = jnp.ones((self.order, x.shape[0]) + (1,) * (x.ndim - 1), sigma.dtype)
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
