"""Diffusion solvers, each taking one reverse step from t to t_next with the model's prediction at t.

A solver is a value. Whatever it needs between steps is kept in its state,
which `init` builds and `step` carries through `sample`'s scan. The signal and
noise rates come from `process`. A solver that needs another model evaluation
(Heun's corrector, RK4's stages, KDPM2's midpoint) calls `denoise`. A solver
that integrates dx / dsigma = eps refuses a schedule whose alpha is not one.

The solvers named after Diffusers 0.34.0 schedulers reproduce their
arithmetic. Before the compiled scan, `init` checks each algorithm's endpoint
domain on the concrete grid; finite endpoint limits are tested, and endpoints
where the update is undefined raise an error instead of substituting another
update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from typing_extensions import TypeVar

from dew.diffusion.process import Process
from dew.diffusion.schedules import GeneralizedNoiseScheduler, expand
from dew.registry import solvers

# tests/test_samplers.py checks the Diffusers-named solvers' trajectories and
# trajectory gradients against fixtures recorded by tools/diffusers_reference.py;
# tools/diffusers_limits_reference.py verifies the finite endpoint limits.

# A solver whose state nobody names is a solver over any state: `StateT` is
# covariant, so `Solver` written bare is the type every concrete solver
# satisfies, and a call site that carries the state names it.
StateT = TypeVar("StateT", covariant=True, default=object)


class Solver(Protocol[StateT]):
    """The interface for one diffusion solver step and the state it keeps between steps.

    `StateT` is that state: nothing for a one-step solver, the previous model
    outputs for a multistep one. It is a type parameter, so each solver's own
    state type is checked at its call sites.
    """

    def init(self, x, times, process, *, key) -> StateT:
        """Return the initial state for sampling from `x` over the concrete grid `times`.

        It checks the grid's endpoint domains at compile time. `key` is the
        root key of the whole sampling run; only `DPMSolverSDE` reads it, for
        its whole-trajectory Brownian tree. Each step draws from the folded key
        passed to it.
        """
        ...

    def step(self, x, t, t_next, denoised, eps, state, key, process,
             denoise, /) -> tuple[jax.Array, StateT]:
        """Return the sample at `t_next` and the new state, from `x` and the model's `(denoised, eps)` at `t`.

        `sample` passes every argument by position, so a solver over another
        algebra can name the pair for what it reads: the discrete solver takes
        log-probabilities where a Gaussian one takes eps.
        """
        ...


def _check_endpoint_domain(process: Process, times: jax.Array, *,
                           source: bool = False, target: bool = False, reason: str) -> None:
    """Reject undefined endpoint integrals while the sampling grid is concrete.

    This runs at initialization, including compile-time initialization in
    sample(), rather than introducing host callbacks into a solver step.
    """
    if times.shape[0] < 2:
        return
    with jax.ensure_compile_time_eval():
        alpha, sigma = process.sampler_schedule.rates(times)
        if source and bool(jnp.any(alpha[:-1] == 0)):
            raise ValueError("alpha=0 source has no finite update: " + reason)
        if target and bool(jnp.any(sigma[1:] == 0)):
            raise ValueError("sigma=0 target has no finite update: " + reason)


def _sigma_integrator(name: str, process: Process) -> GeneralizedNoiseScheduler:
    schedule = process.sampler_schedule
    if not isinstance(schedule, GeneralizedNoiseScheduler):
        raise ValueError(
            f"{name} integrates dx/dsigma = eps, which holds only when alpha is 1, "
            f"so it needs a GeneralizedNoiseScheduler and not {type(schedule).__name__}")
    return schedule


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


@solvers("euler")
@dataclass(frozen=True)
class Euler:
    """The DDIM update written as an Euler step of the probability flow ODE.
    On a variance exploding schedule it is dx/dsigma = eps."""

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        stepped, _, _, _ = _euler_step(x, denoised, process.rates(t, like=x), process.rates(t_next, like=x))
        return stepped, state


@solvers("euler_ancestral")
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


@solvers("heun")
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


@solvers("rk4")
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


@solvers("kdpm2")
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


class _Brownian(NamedTuple):
    """The root interval of a keyed dyadic Brownian bridge, shared by a walk.

    `low` and `high` are the interval the source noise sampler is built over,
    and `key` seeds every node of the bridge. Nothing else is carried: a value
    of the path is a pure function of these three and the point asked for, so
    the state is three scalars whatever the sample's shape.
    """

    key: jax.Array
    low: jax.Array
    high: jax.Array


# A float32 position in [0, 1] carries 24 mantissa bits, so a descent deeper
# than this cannot tell two positions apart and only pretends to refine.
MAX_BROWNIAN_DEPTH = 24


def _brownian_walk(state: _Brownian, point, shape, depth: int) -> jax.Array:
    """W(point) - W(low) of the bridge, by Levy's construction to `depth`.

    W(low) is zero and W(high) is sqrt(high - low) times the root draw; each
    level conditions the midpoint of the half the point falls in, which is the
    Brownian bridge's own N((W(a) + W(b)) / 2, (b - a) / 4). A node's draw comes
    from its level and its dyadic index alone, so deepening the construction
    leaves every coarser node where it was, and the path inside the finest
    cell, of width (high - low) / 2**depth, is read off that cell's two ends.

    The descent reads the binary expansion of the normalized position by
    doubling and subtracting, which is exact in binary floating point, and
    names each node by its integer dyadic index, so no level's bounds are
    accumulated and none collapses however deep the construction runs.

    `point` is a scalar: a source noise sampler is one tree over the whole
    batch tensor, so the interval is the grid's, not a row's.
    """
    span = state.high - state.low
    # A grid whose only interval lands on sigma zero prepares a zero-width
    # interval and never queries it; the walk stays at the path's origin.
    scale = jnp.where(span > 0, span, 1.0)
    position = jnp.clip((point - state.low) / scale, 0.0, 1.0)
    value_low = jnp.zeros(shape, jnp.float32)
    value_high = jnp.sqrt(span) * jax.random.normal(jax.random.fold_in(state.key, 0), shape,
                                                    dtype=jnp.float32)
    index = jnp.zeros((), jnp.int32)
    for level in range(1, depth + 1):
        node = jax.random.fold_in(jax.random.fold_in(state.key, level), 2 * index + 1)
        deviation = 0.5 * jnp.sqrt(span * 2.0 ** (1 - level))
        value_middle = 0.5 * (value_low + value_high) + deviation * jax.random.normal(
            node, shape, dtype=jnp.float32)
        position = position * 2.0
        right = position >= 1.0
        position = position - right.astype(jnp.float32)
        value_low = jnp.where(right, value_middle, value_low)
        value_high = jnp.where(right, value_high, value_middle)
        index = 2 * index + right.astype(jnp.int32)
    return value_low + position * (value_high - value_low)


def _brownian_noise(state: _Brownian, first, second, shape, depth: int) -> jax.Array:
    """The bridge's increment over `[first, second]`, normalized the way
    k-diffusion's `BrownianTreeNoiseSampler` normalizes it: signed with the
    interval's direction and divided by the square root of its width, so the
    result is a standard normal draw carrying the path's correlations."""
    increment = (_brownian_walk(state, second, shape, depth)
                 - _brownian_walk(state, first, shape, depth))
    return increment / jnp.sqrt(jnp.abs(second - first))


def _sde_step(x, denoised, sigma, target):
    """The deterministic part of one `DPMSolverSDEScheduler` step, in the
    exponential form it evaluates: (target / sigma) x - expm1(log target -
    log sigma) x_0, which is the Euler step to `target` regrouped so the
    coefficient stays accurate when the levels are close."""
    return (target / sigma) * x - jnp.expm1(jnp.log(target) - jnp.log(sigma)) * denoised


@solvers("dpmsolver_sde")
@dataclass(frozen=True)
class DPMSolverSDE:
    """Diffusers 0.34.0's `DPMSolverSDEScheduler`, k-diffusion's
    `sample_dpmpp_sde` midpoint solver over a Brownian tree.

    Each interval takes two ancestral first-order steps from its own start:
    one to the geometric midpoint of sigma_t and sigma_s, where the model is
    evaluated, and one to sigma_s with that midpoint's clean prediction. Both
    steps go down to k-diffusion's sigma_down and add sigma_up of noise, and
    both draw that noise from one Brownian path over the trajectory's sigma
    interval: the first over `[sigma_t, sigma_mid]` and the second over
    `[sigma_t, sigma_s]`, so the two are correlated exactly as nested
    increments of one path. The source's sampler transforms sigma with the
    identity even though its own steps integrate -log(sigma), so the interval
    widths are sigma differences.

    `depth` resolves the root interval to `(sigma_max - sigma_min) / 2**depth`:
    on a published VP table's span of about 14.6 the default reaches 8.7e-7,
    at or inside the reference tree's own 1e-6 tolerance, and it is also where
    a float32 position runs out of mantissa, so no deeper descent tells two
    sigmas apart. A zero-sigma target has no ancestral step and lands on the
    clean prediction.

    The root interval is the schedule's positive sigma domain, not the grid's
    extremes, as the source builds its tree over every positive sigma it
    prepared, so a walk over a suffix of the grid keeps the whole grid's path.

    `seed` is the source's `noise_sampler_seed`: the path is then fixed by
    the checkpoint, independent of the walk's key. A Torch seed names no JAX
    stream, so it seeds this bridge rather than reproducing the reference tree.
    """

    depth: int = MAX_BROWNIAN_DEPTH
    seed: int | None = None

    def __post_init__(self) -> None:
        if type(self.depth) is not int or not 1 <= self.depth <= MAX_BROWNIAN_DEPTH:
            raise ValueError(f"the Brownian bridge takes 1 to {MAX_BROWNIAN_DEPTH} levels, "
                             f"the mantissa of a float32 position, not {self.depth}")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError(f"the noise sampler's seed is a nonnegative integer, not {self.seed}")

    def init(self, x, times, process, *, key):
        schedule = _sigma_integrator("DPMSolverSDE", process)
        with jax.ensure_compile_time_eval():
            sigmas = schedule.sigmas(jnp.asarray(times, jnp.float32))
            if times.shape[0] > 1 and bool(jnp.any(sigmas[:-1] <= 0)):
                raise ValueError("sigma=0 source has no finite update: DPMSolverSDE "
                                 "steps down from the interval's own sigma")
        low, high = float(schedule.sigma_min), float(schedule.sigma_max)
        if not 0 < low <= high:
            raise ValueError("the Brownian tree needs a positive sigma domain, and this "
                             f"schedule reports [{low}, {high}]")
        root = key if self.seed is None else jax.random.PRNGKey(self.seed)
        return _Brownian(root, jnp.asarray(low, jnp.float32), jnp.asarray(high, jnp.float32))

    def _noise(self, state, first, second, shape):
        """The path's standard normal draw over `[first, second]`, the one
        place a reference walk couples the source's own tree in."""
        return _brownian_noise(state, first, second, shape, self.depth)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        schedule = _sigma_integrator("DPMSolverSDE", process)
        _, sigma_t = process.rates(t, like=x)
        _, sigma_s = process.rates(t_next, like=x)

        def ancestral(target):
            """The source's `sigma_up`, capped at the target level, and the
            `sigma_down` a step actually reaches."""
            up = jnp.minimum(target, (target ** 2 * (sigma_t ** 2 - target ** 2) / sigma_t ** 2) ** 0.5)
            return (target ** 2 - up ** 2) ** 0.5, up

        def midpoint(_):
            sigma_mid = jnp.exp(0.5 * (jnp.log(sigma_t) + jnp.log(sigma_s)))
            first_down, first_up = ancestral(sigma_mid)
            noise = self._noise(state, jnp.min(sigma_t), jnp.min(sigma_mid), x.shape)
            x_mid = _sde_step(x, denoised, sigma_t, first_down) + noise * first_up
            denoised_mid, _ = denoise(x_mid, schedule.t_of_sigma(sigma_mid.reshape(-1)))
            down, up = ancestral(sigma_s)
            noise = self._noise(state, jnp.min(sigma_t), jnp.min(sigma_s), x.shape)
            return _sde_step(x, denoised_mid, sigma_t, down) + noise * up

        stepped = lax.cond(jnp.all(sigma_s == 0), lambda _: denoised, midpoint, None)
        return jnp.where(sigma_s > 0, stepped, denoised), state


@solvers("multistep_dpm")
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


def _push(buffer, value):
    """`buffer` with its oldest entry dropped and `value` appended, so the
    most recent entry is last."""
    return jnp.concatenate([buffer[1:], value[None]], axis=0)


class Multistep(NamedTuple):
    """What a multistep solver carries: the last `depth` converted model
    outputs with the signal and noise rates they were read at, most recent
    last, the number of steps taken and the number the walk has. Slots not
    yet filled hold zeros at unit rates, which no update reads."""

    outputs: jax.Array
    alphas: jax.Array
    sigmas: jax.Array
    taken: jax.Array
    steps: jax.Array

    @property
    def lambdas(self) -> jax.Array:
        return _half_log_snr(self.alphas, self.sigmas)

    def push(self, output, alpha, sigma) -> Multistep:
        return Multistep(_push(self.outputs, output), _push(self.alphas, alpha),
                         _push(self.sigmas, sigma), self.taken, self.steps)

    def advance(self) -> Multistep:
        return self._replace(taken=self.taken + 1)


def _multistep(x, times, depth: int) -> Multistep:
    rates = jnp.ones((depth, x.shape[0]) + (1,) * (x.ndim - 1), jnp.float32)
    return Multistep(jnp.zeros((depth, *x.shape), x.dtype), rates, rates,
                     jnp.zeros((), jnp.int32), jnp.asarray(times.shape[0] - 1, jnp.int32))


def _half_log_snr(alpha, sigma):
    """lambda = log(alpha) - log(sigma), the variable DPM-Solver integrates in."""
    return jnp.log(alpha) - jnp.log(sigma)


def _lambda_step(alpha_s, sigma_s, alpha_t, sigma_t):
    """Return the target-zero mask and a finite coefficient placeholder at
    either endpoint. Each caller substitutes its own proved endpoint limit;
    higher-order DEIS and corrected UniPC epsilon need more than clean x_0.
    """
    terminal = sigma_t <= 0
    endpoint = terminal | (alpha_s == 0)
    safe_sigma_t = jnp.where(terminal, sigma_s, sigma_t)
    safe_alpha_s = jnp.where(alpha_s == 0, 1.0, alpha_s)
    h = _half_log_snr(alpha_t, safe_sigma_t) - _half_log_snr(safe_alpha_s, sigma_s)
    return terminal, jnp.where(endpoint, 1.0, h)


def _tapered_order(order: int, history: Multistep, lower_order_final: bool,
                   euler_at_final: bool):
    """The order a Diffusers multistep DPM-Solver takes at this step: it grows
    with the history up to `order`; the last step is first order under
    `euler_at_final`, and in a walk under 15 steps `lower_order_final` makes
    the last step first order and caps the one before it at second."""
    taken, steps = history.taken, history.steps
    current = jnp.minimum(order, taken + 1)
    short = jnp.logical_and(lower_order_final, steps < 15)
    final = jnp.logical_and(taken == steps - 1, jnp.logical_or(euler_at_final, short))
    current = jnp.where(final, 1, current)
    penultimate = jnp.logical_and(taken == steps - 2, short)
    return jnp.where(penultimate, jnp.minimum(current, 2), current)


Algorithm = Literal["dpmsolver++", "dpmsolver", "sde-dpmsolver++", "sde-dpmsolver"]


def _dpm_terms(algorithm: str, alpha_s, sigma_s, alpha_t, sigma_t, h):
    """The coefficients of one DPM-Solver algorithm over h, as Diffusers
    writes them: `(a, b, c_midpoint, c_heun, c_2, n)` for
    x_t = a x + b D0 + c D1 + c_2 D2 + n z, with c the midpoint or Heun
    second-order form (the third order takes the Heun one) and z standard
    normal noise the SDE forms add. `dpmsolver++` and `sde-dpmsolver++`
    integrate the clean prediction; the other two integrate eps."""
    alpha_s = jnp.where(alpha_s == 0, 1.0, alpha_s)
    if algorithm == "dpmsolver++":
        phi = jnp.exp(-h) - 1.0
        sample_scale = sigma_t / sigma_s
        # exp(-h) = alpha_s / alpha_t * sigma_t / sigma_s. Use the
        # rates directly rather than round them through logarithms and exp.
        clean_scale = alpha_t - sample_scale * alpha_s
        return (sample_scale, clean_scale, 0.5 * clean_scale,
                alpha_t * (phi / h + 1.0), -alpha_t * ((phi + h) / h ** 2 - 0.5), 0.0)
    if algorithm == "dpmsolver":
        psi = jnp.exp(h) - 1.0
        return (alpha_t / alpha_s, -sigma_t * psi, -0.5 * sigma_t * psi,
                -sigma_t * (psi / h - 1.0), -sigma_t * ((psi - h) / h ** 2 - 0.5), 0.0)
    if algorithm == "sde-dpmsolver++":
        decay = 1.0 - jnp.exp(-2.0 * h)
        return (sigma_t / sigma_s * jnp.exp(-h), alpha_t * decay, 0.5 * alpha_t * decay,
                alpha_t * (decay / (-2.0 * h) + 1.0),
                alpha_t * ((decay - 2.0 * h) / (2.0 * h) ** 2 - 0.5),
                sigma_t * jnp.sqrt(decay))
    psi = jnp.exp(h) - 1.0
    return (alpha_t / alpha_s, -2.0 * sigma_t * psi, -sigma_t * psi,
            -2.0 * sigma_t * (psi / h - 1.0), None, sigma_t * jnp.sqrt(jnp.exp(2 * h) - 1.0))


@solvers("dpmsolver_multistep")
@dataclass(frozen=True)
class DPMSolverMultistep:
    """DPM-Solver and DPM-Solver++ as multistep integrators in lambda = log(alpha) - log(sigma).

    The papers are Lu et al. (2022, arXiv 2206.00927) and arXiv 2211.01095.
    This covers the four algorithms, three orders and two second-order forms
    of Diffusers 0.34.0's `DPMSolverMultistepScheduler`, with its defaults.

    `dpmsolver++` and `sde-dpmsolver++` integrate the clean prediction, the
    other two eps; the `sde-` forms add the noise term of the SDE solver.
    With h = lambda_t - lambda_s0 over the step and D0, D1, D2 the finite
    differences of the last outputs in lambda, the deterministic `dpmsolver++`
    step is

        x_t = (sigma_t / sigma_s0) x - alpha_t (e^-h - 1) D0 + c D1 + c_2 D2

    with each algorithm's coefficients written as Diffusers writes them. The
    first step has no history and is first order, the second at
    most second. `lower_order_final` is Diffusers' rule verbatim, which acts
    only in a walk under 15 steps: first order on the last step and at most
    second on the one before. `euler_at_final` makes the last step first
    order whatever the length. A zero target forces the clean limit for
    deterministic and ++ updates. The non-++ SDE first-order limit also
    retains alpha_t*sigma_s/alpha_s*(noise-eps); it requires a finite source
    alpha and a final order reduction. That endpoint is an equation-limit
    extension: Diffusers refuses a literal zero-terminal non-++ config.
    EDM uses the ++ algorithms over its existing process.

    DPM-Solver++ 2M with no order taper is order=2,
    algorithm="dpmsolver++", solver_type="midpoint", lower_order_final=False,
    euler_at_final=False.
    """
    # `_dpm_terms` holds each algorithm's coefficients.

    order: int = 2
    algorithm: Algorithm = "dpmsolver++"
    solver_type: Literal["midpoint", "heun"] = "midpoint"
    lower_order_final: bool = True
    euler_at_final: bool = False

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError(f"DPM-Solver has orders 1, 2 and 3, not {self.order}")
        if self.algorithm == "sde-dpmsolver" and self.order == 3:
            raise ValueError("sde-dpmsolver has no third-order update; use order 2 or sde-dpmsolver++")

    def init(self, x, times, process, *, key):
        if self.algorithm == "sde-dpmsolver":
            first_at_end = (self.order == 1 or self.euler_at_final
                            or (self.lower_order_final and times.shape[0] - 1 < 15))
            _check_endpoint_domain(
                process, times, source=True, target=not first_at_end,
                reason="sde-dpmsolver requires finite alpha and a lowered order at sigma zero")
        return _multistep(x, times, self.order)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        alpha_s0, sigma_s0 = process.rates(t, like=x)
        alpha_t, sigma_t = process.rates(t_next, like=x)
        history = state.push(denoised if self.algorithm.endswith("++") else eps, alpha_s0, sigma_s0)
        terminal, h = _lambda_step(alpha_s0, sigma_s0, alpha_t, sigma_t)
        a, b, c_midpoint, c_heun, c_2, n = _dpm_terms(
            self.algorithm, alpha_s0, sigma_s0, alpha_t, sigma_t, h)
        lambdas, outputs = history.lambdas, history.outputs
        m0 = outputs[-1]
        base = a * x + b * m0
        target_limit = alpha_t * denoised
        if self.algorithm.startswith("sde"):
            noise = jax.random.normal(key, x.shape, dtype=jnp.float32)
            base = base + n * noise
            source_limit = alpha_t * denoised + sigma_t * noise
            if self.algorithm == "sde-dpmsolver":
                target_limit = target_limit + alpha_t * sigma_s0 / alpha_s0 * (noise - eps)
        else:
            source_limit = alpha_t * denoised + sigma_t * eps
        base = jnp.where(alpha_s0 == 0, source_limit, base)

        def first(_):
            return base

        def second(_):
            r0 = (lambdas[-1] - lambdas[-2]) / h
            d1 = (m0 - outputs[-2]) / r0
            return base + (c_midpoint if self.solver_type == "midpoint" else c_heun) * d1

        def third(_):
            r0 = (lambdas[-1] - lambdas[-2]) / h
            r1 = (lambdas[-2] - lambdas[-3]) / h
            d1_0 = (m0 - outputs[-2]) / r0
            d1_1 = (outputs[-2] - outputs[-3]) / r1
            d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1)
            d2 = (d1_0 - d1_1) / (r0 + r1)
            return base + c_heun * d1 + c_2 * d2

        order = _tapered_order(self.order, history, self.lower_order_final, self.euler_at_final)
        stepped = lax.switch(order - 1, [first, second, third][:self.order], None)
        return jnp.where(terminal, target_limit, stepped), history.advance()


class Singlestep(NamedTuple):
    """`DPMSolverSinglestep`'s state: the output history, the sample its
    current group of steps started from, and the order of every step."""

    history: Multistep
    anchor: jax.Array
    orders: jax.Array


@solvers("dpmsolver_singlestep")
@dataclass(frozen=True)
class DPMSolverSinglestep:
    """Diffusers 0.34.0's grouped DPM-Solver updates from each group's anchor.

    A group of k model evaluations completes one k-th order update. Orders
    repeat [1, 2] or [1, 2, 3]; an incomplete final group uses lower order,
    as set_timesteps does in the reference. lower_order_final also lowers
    the final complete group, and a zero-sigma target forces final order 1.
    Source-domain checks use that effective order list.

    At alpha=0, clean-prediction midpoint and deterministic Heun groups have
    finite limits. Noise-prediction groups above order 1 and third-order
    SDE Heun groups diverge; initialization rejects those source/grid pairs.
    """

    order: int = 2
    algorithm: Literal["dpmsolver++", "dpmsolver", "sde-dpmsolver++"] = "dpmsolver++"
    solver_type: Literal["midpoint", "heun"] = "midpoint"
    lower_order_final: bool = False

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError(f"DPM-Solver has orders 1, 2 and 3, not {self.order}")

    def order_list(self, steps: int) -> list[int]:
        """Reference groups, completing an uneven final group at lower order."""
        if steps == 0:
            return []
        order = self.order
        if self.lower_order_final or steps % order:
            if order == 3:
                if steps % 3 == 0:
                    return [1, 2, 3] * (steps // 3 - 1) + [1, 2] + [1]
                if steps % 3 == 1:
                    return [1, 2, 3] * (steps // 3) + [1]
                return [1, 2, 3] * (steps // 3) + [1, 2]
            if order == 2:
                if steps % 2 == 0:
                    return [1, 2] * (steps // 2 - 1) + [1, 1]
                return [1, 2] * (steps // 2) + [1]
            return [1] * steps
        return list(range(1, order + 1)) * (steps // order)

    def init(self, x, times, process, *, key):
        steps = times.shape[0] - 1
        orders = self.order_list(steps)
        if orders:
            with jax.ensure_compile_time_eval():
                _, last_sigma = process.sampler_schedule.rates(times[-1:])
                if bool(jnp.all(last_sigma == 0)):
                    orders[-1] = 1
        if self.algorithm == "dpmsolver" and len(orders) > 1 and orders[1] > 1:
            _check_endpoint_domain(
                process,
                times,
                source=True,
                reason="noise-prediction singlestep differences diverge at an alpha=0 anchor",
            )
        if (self.algorithm == "sde-dpmsolver++" and self.solver_type == "heun"
                and 3 in orders[:3]):
            _check_endpoint_domain(process, times, source=True,
                                   reason="third-order SDE Heun differences diverge at an alpha=0 anchor")
        return Singlestep(_multistep(x, times, self.order), x, jnp.asarray(orders, jnp.int32))

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        alpha_s0, sigma_s0 = process.rates(t, like=x)
        alpha_t, sigma_t = process.rates(t_next, like=x)
        history = state.history.push(
            denoised if self.algorithm.endswith("++") else eps, alpha_s0, sigma_s0)
        order = jnp.where(jnp.all(sigma_t == 0), 1, state.orders[history.taken])
        anchor = jnp.where(order == 1, x, state.anchor)
        lambdas, outputs = history.lambdas, history.outputs
        m0 = outputs[-1]
        noise = jax.random.normal(key, x.shape, dtype=jnp.float32)

        def update(back: int):
            """The update's coefficients from the group's anchor, `back`
            points behind, and the noise term at its h."""
            alpha_a, sigma_a = history.alphas[-back], history.sigmas[-back]
            _, h = _lambda_step(alpha_a, sigma_a, alpha_t, sigma_t)
            a, b, c_midpoint, c_heun, c_2, n = _dpm_terms(
                self.algorithm, alpha_a, sigma_a, alpha_t, sigma_t, h)
            return h, a * anchor, b, c_midpoint, c_heun, c_2, n * noise

        def source_limit(back: int, clean):
            if self.algorithm.startswith("sde"):
                return alpha_t * clean + sigma_t * noise
            return sigma_t / history.sigmas[-back] * anchor + alpha_t * clean

        def first(_):
            _, ax, b, _, _, _, dz = update(1)
            stepped = ax + b * m0
            if self.algorithm.startswith("sde"):
                stepped = stepped + dz
            return jnp.where(alpha_s0 == 0, source_limit(1, denoised), stepped)

        def second(_):
            h, ax, b, c_midpoint, c_heun, _, dz = update(2)
            m1 = outputs[-2]
            from_zero = history.alphas[-2] == 0
            r0 = jnp.where(from_zero, 1.0, (lambdas[-1] - lambdas[-2]) / h)
            d1 = (m0 - m1) / r0
            stepped = ax + b * m1 + (c_midpoint if self.solver_type == "midpoint" else c_heun) * d1
            if self.algorithm.startswith("sde"):
                stepped = stepped + dz
            weight = 0.5 if self.solver_type == "midpoint" else 1.0
            return jnp.where(from_zero, source_limit(2, m1 + weight * (m0 - m1)), stepped)

        def third(_):
            return _singlestep_third(self, update, source_limit, history, alpha_t, sigma_t)

        stepped = lax.switch(order - 1, [first, second, third][:self.order], None)
        terminal = sigma_t <= 0
        next_x = jnp.where(terminal, alpha_t * denoised, stepped)
        return next_x, Singlestep(history.advance(), anchor, state.orders)


def _singlestep_third(solver: DPMSolverSinglestep, update, source_limit, history: Multistep,
                      alpha_t, sigma_t):
    """The third-order singlestep update from the anchor three points back.

    `update` gives that anchor's coefficients and its noise term, and
    `source_limit` the alpha=0 limit. The Heun `dpmsolver++` form combines
    the two differences before the limit is taken, because their leading
    terms diverge individually and cancel together.
    """
    lambdas, outputs = history.lambdas, history.outputs
    m0, m1, m2 = outputs[-1], outputs[-2], outputs[-3]
    h, ax, b, _, c_heun, c_2, dz = update(3)
    from_zero = history.alphas[-3] == 0
    r0 = jnp.where(from_zero, 1.0, (lambdas[-1] - lambdas[-3]) / h)
    r1 = jnp.where(from_zero, 0.5, (lambdas[-2] - lambdas[-3]) / h)
    d1_0 = (m1 - m2) / r1
    d1_1 = (m0 - m2) / r0
    if solver.solver_type == "midpoint":
        stepped = ax + b * m2 + c_heun * d1_1
    else:
        d1 = (r0 * d1_0 - r1 * d1_1) / (r0 - r1)
        d2 = 2.0 * (d1_1 - d1_0) / (r0 - r1)
        stepped = ax + b * m2 + c_heun * d1 + c_2 * d2
    if solver.algorithm.startswith("sde"):
        stepped = stepped + dz
    endpoint_clean = m0
    if solver.solver_type == "heun" and solver.algorithm == "dpmsolver++":
        # Combine D1 and D2 before taking the infinite-anchor limit.
        # Their individually divergent leading terms cancel.
        gap = _half_log_snr(alpha_t, jnp.where(sigma_t == 0, 1.0, sigma_t)) - lambdas[-1]
        endpoint_clean = m0 + (gap - 1.0) * (m0 - m1) / (lambdas[-1] - lambdas[-2])
    return jnp.where(from_zero, source_limit(3, endpoint_clean), stepped)


def _deis_second(t, b, c):
    """Integral of the log-rho Lagrange basis, including t=0 and a node at infinity."""
    infinite_b, infinite_c = jnp.isinf(b), jnp.isinf(c)
    log_b = jnp.log(jnp.where(infinite_b, 1.0, b))
    log_c = jnp.log(jnp.where(infinite_c, 1.0, c))
    log_t = jnp.log(jnp.where(t == 0, 1.0, t))
    denominator = jnp.where(infinite_b | infinite_c, 1.0, log_b - log_c)
    integral = t * (-log_c + log_t - 1) / denominator
    return jnp.where(infinite_b, 0.0, jnp.where(infinite_c, t, integral))


def _deis_third(t, b, c, d):
    """Integral of the quadratic log-rho basis; an infinite node has zero
    weight, and each other basis loses its corresponding factor."""
    infinite_b, infinite_c, infinite_d = jnp.isinf(b), jnp.isinf(c), jnp.isinf(d)
    lb = jnp.log(jnp.where(infinite_b, 1.0, b))
    lc = jnp.log(jnp.where(infinite_c, 1.0, c))
    ld = jnp.log(jnp.where(infinite_d, 1.0, d))
    lt = jnp.log(jnp.where(t == 0, 1.0, t))
    numerator = t * (lc * (ld - lt + 1) - ld * lt + ld + lt ** 2 - 2 * lt + 2)
    denominator = jnp.where(infinite_b | infinite_c | infinite_d, 1.0, (lb - lc) * (lb - ld))
    integral = numerator / denominator
    integral = jnp.where(infinite_c, _deis_second(t, b, d), integral)
    integral = jnp.where(infinite_d, _deis_second(t, b, c), integral)
    return jnp.where(infinite_b, 0.0, integral)


@solvers("deis")
@dataclass(frozen=True)
class DEIS:
    """DEIS (Zhang and Chen 2023, arXiv 2204.13902) in its log-rho multistep
    form, Diffusers 0.34.0's `DEISMultistepScheduler`: the exponential
    integrator of eps with the polynomial-in-log(rho) interpolation of the
    last outputs, rho = sigma / alpha, integrated in closed form over the
    step. The first order is DPM-Solver's. Orders grow with the history;
    `lower_order_final` is the same short-walk taper as
    `DPMSolverMultistep`'s. At sigma=0 the integrated log-rho basis retains
    its history terms. An alpha=0 source contributes a node at infinite rho;
    that node's weight vanishes in subsequent finite-interval integrals.
    """

    order: int = 2
    lower_order_final: bool = True

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError(f"DEIS has orders 1, 2 and 3, not {self.order}")

    def init(self, x, times, process, *, key):
        return _multistep(x, times, self.order)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        alpha_s0, sigma_s0 = process.rates(t, like=x)
        alpha_t, sigma_t = process.rates(t_next, like=x)
        history = state.push(eps, alpha_s0, sigma_s0)
        terminal, h = _lambda_step(alpha_s0, sigma_s0, alpha_t, sigma_t)
        rho_t = sigma_t / alpha_t
        rhos = history.sigmas / history.alphas
        rho_s0, rho_s1 = rhos[-1], rhos[-2]
        m0, m1 = history.outputs[-1], history.outputs[-2]

        def first(_):
            safe_alpha_s0 = jnp.where(alpha_s0 == 0, 1.0, alpha_s0)
            regular = (alpha_t / safe_alpha_s0) * x - sigma_t * (jnp.exp(h) - 1.0) * m0
            return jnp.where((alpha_s0 == 0) | terminal, alpha_t * denoised + sigma_t * eps, regular)

        def second(_):
            coef1 = _deis_second(rho_t, rho_s0, rho_s1) - _deis_second(rho_s0, rho_s0, rho_s1)
            coef2 = _deis_second(rho_t, rho_s1, rho_s0) - _deis_second(rho_s0, rho_s1, rho_s0)
            return alpha_t * (x / alpha_s0 + coef1 * m0 + coef2 * m1)

        def third(_):
            rho_s2, m2 = rhos[-3], history.outputs[-3]
            coef1 = (_deis_third(rho_t, rho_s0, rho_s1, rho_s2)
                     - _deis_third(rho_s0, rho_s0, rho_s1, rho_s2))
            coef2 = (_deis_third(rho_t, rho_s1, rho_s2, rho_s0)
                     - _deis_third(rho_s0, rho_s1, rho_s2, rho_s0))
            coef3 = (_deis_third(rho_t, rho_s2, rho_s0, rho_s1)
                     - _deis_third(rho_s0, rho_s2, rho_s0, rho_s1))
            return alpha_t * (x / alpha_s0 + coef1 * m0 + coef2 * m1 + coef3 * m2)

        order = _tapered_order(self.order, history, self.lower_order_final, euler_at_final=False)
        stepped = lax.switch(order - 1, [first, second, third][:self.order], None)
        return stepped, history.advance()


class UniPCState(NamedTuple):
    """`UniPC`'s state: the last `order` model outputs, most recent last; the
    sample its last predictor left from; that predictor's order, which its
    corrector takes (0 before the first predictor, when there is nothing to
    correct); the steps taken and the walk's count; and the walk's grid with
    every interval's coefficients (`_unipc_tables`)."""

    outputs: jax.Array
    last_x: jax.Array
    last_order: jax.Array
    taken: jax.Array
    steps: jax.Array
    grid: jax.Array
    corrector: jax.Array
    predictor: jax.Array


def _unipc_weights(rks: list[np.float64], hh: np.float64, B_h: np.float64, order: int,
                   predictor: bool) -> list[np.float64]:
    """UniPC's B(h) weights at `order`: the solution of R rho = b built from
    the relative positions `rks` of the history points (the current point's
    own 1 last). The predictor has no difference at the point it steps to,
    so it drops the last row and column; Diffusers takes 0.5 outright for
    the second-order predictor and the first-order corrector."""
    if order == (2 if predictor else 1):
        return [np.float64(0.5)]
    columns = order - 1 if predictor else order
    nodes = [*rks[:-1], np.float64(1.0)][:columns]
    # Scale the infinite node's Vandermonde column by r**(columns-1).
    # Its lower entries and recovered weight vanish; the leading entry
    # stays finite. This is the limit of the same system, not a lower-order solver.
    inverse = [np.float64(0.0 if np.isinf(rk) else 1.0) for rk in nodes]
    normalized = [np.sign(rk) if np.isinf(rk) else rk for rk in nodes]
    rows, b = [], []
    h_phi_k = np.expm1(hh) / hh - 1
    factorial = 1
    for i in range(1, columns + 1):
        rows.append([rk ** (i - 1) * inv ** (columns - i)
                     for rk, inv in zip(normalized, inverse, strict=True)])
        b.append(h_phi_k * factorial / B_h)
        factorial *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial
    solution = np.linalg.solve(np.asarray(rows, np.float64), np.asarray(b, np.float64))
    return [solution[k] * inverse[k] ** (columns - 1) for k in range(columns)]


def _unipc_tables(alpha: np.ndarray, sigma: np.ndarray, solver: UniPC) -> tuple[np.ndarray, np.ndarray]:
    """Every interval's UniPC update as coefficients, in float64 on the host.

    Each depends only on the grid's rates and the order it is taken at, so
    lambda, h, e^h - 1 and the weight systems are solved once here rather
    than in float32 inside the walk, where Diffusers' scheduler also solves
    them. `corrector[j, p]` corrects the sample at grid point j with order
    p: weights on the sample, the last predictor's sample, m_{j-1}, the
    difference m_j - m_{j-1} and the history's differences
    m_{j-1-k} - m_{j-1}; p = 0 keeps the sample. `predictor[j, q-1]` steps
    from point j to j+1 with order q: weights on the corrected sample, the
    clean prediction, eps and the differences m_{j-k} - m_j. The
    differences stay in the walk, taken before any weight, as the source
    takes them.

    Only the pairs a walk can take are solved: the order a step reaches
    (`reach`, the history and `lower_order_final`'s cap) and below, as a
    walk restarted on the grid takes them, and no correction where
    `disable_corrector` names the step before. Every other pair holds
    zeros, or keeps the sample for a correction.
    """
    order, predict_x0 = solver.order, solver.predict_x0
    # An endpoint's lambda is infinite: log(0) at sigma 0 or alpha 0.
    with np.errstate(divide="ignore"):
        lambdas = np.log(alpha) - np.log(sigma)
    intervals = alpha.shape[0] - 1
    corrector = np.zeros((max(intervals, 0), order + 1, order + 3))
    predictor = np.zeros((max(intervals, 0), order, order + 2))
    corrector[:, :, 0] = 1.0
    here = 1 if predict_x0 else 2

    def reach(j: int) -> int:
        return min(order, j + 1, intervals - j if solver.lower_order_final else order)

    def b_h(hh):
        return hh if solver.solver_type == "bh1" else np.expm1(hh)

    for j in range(intervals):
        for p in range(1, reach(j - 1) + 1 if j > 0 and j - 1 not in solver.disable_corrector else 1):
            row = corrector[j, p]
            row[0] = 0.0
            h = lambdas[j] - lambdas[j - 1]
            hh = -h if predict_x0 else h
            rks = [(lambdas[j - 1 - k] - lambdas[j - 1]) / h for k in range(1, p)] + [np.float64(1.0)]
            B_h = b_h(hh)
            rhos = _unipc_weights(rks, hh, B_h, p, predictor=False)
            head = alpha[j] if predict_x0 else sigma[j]
            row[1] = sigma[j] / sigma[j - 1] if predict_x0 else alpha[j] / alpha[j - 1]
            row[2] = -head * np.expm1(hh)
            row[3] = -head * B_h * rhos[-1]
            for k in range(1, p):
                row[3 + k] = -head * B_h * rhos[k - 1] / rks[k - 1]
        for q in range(1, reach(j) + 1):
            row = predictor[j, q - 1]
            terminal = sigma[j + 1] <= 0
            if terminal:
                # The h -> infinity limit, which each prediction proves apart.
                if predict_x0 or alpha[j] == 0:
                    row[1] = alpha[j + 1]
                else:
                    row[0], row[2] = alpha[j + 1] / alpha[j], -alpha[j + 1] * sigma[j] / alpha[j]
                continue
            h = np.float64(1.0) if alpha[j] == 0 else lambdas[j + 1] - lambdas[j]
            hh = -h if predict_x0 else h
            B_h = b_h(hh)
            head = alpha[j + 1] if predict_x0 else sigma[j + 1]
            if alpha[j] == 0:
                row[1], row[2] = alpha[j + 1], sigma[j + 1]
            else:
                row[0] = sigma[j + 1] / sigma[j] if predict_x0 else alpha[j + 1] / alpha[j]
                row[here] = -head * np.expm1(hh)
            if q > 1:
                rks = [(lambdas[j - k] - lambdas[j]) / h for k in range(1, q)] + [np.float64(1.0)]
                rhos = _unipc_weights(rks, hh, B_h, q, predictor=True)
                for k in range(1, q):
                    row[2 + k] = -head * B_h * rhos[k - 1] / rks[k - 1]
    return corrector, predictor


@solvers("unipc")
@dataclass(frozen=True)
class UniPC:
    """UniPC, a unified predictor and corrector in lambda, as Diffusers 0.34.0's `UniPCMultistepScheduler`.

    UniPC is from Zhao et al. (2023, arXiv 2302.04867). Its weights solve a
    small linear system over the history's positions. Each step first
    corrects the sample the last predictor produced, with
    the model output just read there and that predictor's order, then
    predicts the next sample from the corrected one. `solver_type` picks
    B(h) as h (`bh1`) or e^h - 1 (`bh2`); `predict_x0` integrates the clean
    prediction, otherwise eps. `lower_order_final` caps the order by the
    steps remaining, Diffusers' rule, so the last step is first order;
    `disable_corrector` names the step indices whose predictor's output is
    not corrected. The lowered clean-prediction terminal is alpha_t*x_0;
    epsilon prediction uses the corrected sample with the pre-correction
    epsilon. Initialization rejects an unlowered higher-order zero target.
    At an alpha=0 source, bh1 and epsilon correctors diverge unless the
    first correction is disabled. The finite infinite-node weights use the
    same Vandermonde system with column scaling.

    Every coefficient depends on the grid alone, so `init` solves them in
    float64 for each interval and order, and a step reads
    its interval's row by where `t` sits on that grid and its order from
    the state. A step's `t` is a point of the grid `init` was given.
    """
    # `_unipc_tables` solves the coefficient tables in `init`.

    order: int = 2
    solver_type: Literal["bh1", "bh2"] = "bh2"
    predict_x0: bool = True
    lower_order_final: bool = True
    disable_corrector: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError(f"UniPC's order is at least 1, not {self.order}")

    def init(self, x, times, process, *, key):
        corrects_source = times.shape[0] > 2 and 0 not in self.disable_corrector
        _check_endpoint_domain(
            process, times,
            source=corrects_source and (not self.predict_x0 or self.solver_type == "bh1"),
            target=not self.lower_order_final and self.order > 1 and times.shape[0] > 2,
            reason="UniPC corrector/predictor requires finite log-SNR at this endpoint")
        with jax.ensure_compile_time_eval():
            alpha, sigma = process.sampler_schedule.rates(jnp.asarray(times))
            corrector, predictor = _unipc_tables(np.asarray(alpha, np.float64), np.asarray(sigma, np.float64),
                                                 self)
        return UniPCState(jnp.zeros((self.order, *x.shape), x.dtype), x, jnp.zeros((), jnp.int32),
                          jnp.zeros((), jnp.int32), jnp.asarray(times.shape[0] - 1, jnp.int32),
                          jnp.asarray(times, jnp.float32), jnp.asarray(corrector, jnp.float32),
                          jnp.asarray(predictor, jnp.float32))

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        def weighted(row: jax.Array, terms: list[jax.Array]) -> jax.Array:
            total = row[0] * terms[0]
            for index in range(1, len(terms)):
                total = total + row[index] * terms[index]
            return total

        interval = jnp.argmin(jnp.abs(state.grid - t.reshape(-1)[0]))
        here = denoised if self.predict_x0 else eps
        history = [state.outputs[-k] for k in range(1, self.order + 1)]
        row = state.corrector[interval, state.last_order]
        previous = history[0]
        x = weighted(row, [x, state.last_x, previous, here - previous,
                           *(older - previous for older in history[1:])])
        this_order = (jnp.minimum(self.order, state.steps - state.taken) if self.lower_order_final
                      else self.order)
        this_order = jnp.minimum(this_order, state.taken + 1)
        row = state.predictor[interval, this_order - 1]
        next_x = weighted(row, [x, denoised, eps, *(older - here for older in history[:self.order - 1])])
        return next_x, state._replace(outputs=_push(state.outputs, here), last_x=x, last_order=this_order,
                                      taken=state.taken + 1)


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


@solvers("lms")
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


__all__ = ["DDIM", "DDPM", "DEIS", "KDPM2", "LMS", "PNDM", "RK4", "TCD", "Consistency", "DPMSolverMultistep",
           "DPMSolverSDE", "DPMSolverSinglestep", "Euler", "EulerAncestral", "Heun", "MultiStepDPM", "Solver",
           "UniPC"]
