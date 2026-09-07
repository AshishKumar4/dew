"""One reverse step each, from t to t_next, given the model's denoising at t.

A solver is a value. What it needs between steps travels in its state;
`init` builds it from x_T, a concrete time grid and the process; `step` threads it through
`sample`'s scan. The rates of the sampling schedule come from `process`; a
solver that needs another evaluation of the model (Heun's corrector, RK4's
stages, KDPM2's midpoint) calls `denoise`. A solver that integrates
dx / dsigma = eps says so by refusing a schedule whose alpha is not one.

The solvers named after Diffusers 0.34.0 schedulers reproduce their
arithmetic; `tests/test_samplers.py` holds their trajectories and trajectory
gradients against the fixtures `tools/diffusers_reference.py` records.
Process supplies the time grid. Initializing with a concrete grid and process
checks each algorithm's endpoint domain before the compiled scan. Finite
endpoint limits are verified separately in tools/diffusers_limits_reference.py;
DEIS history, UniPC epsilon correction, and non-++ SDE noise can survive a
zero-sigma target. Undefined limits raise rather than substitute an update.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Literal, NamedTuple, Protocol, TypeVar

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
    previous model outputs for a multi-step one. It is a type parameter, so a
    solver's own state type is checked at its call sites.
    """

    def init(self, x, times, process) -> StateT:
        """Prepare state and check endpoint domains on the concrete time grid.

        sample() materializes this grid at compile time, so validation adds
        no host callbacks to the compiled step.
        """
        ...

    def step(self, x, t, t_next, denoised, eps, state, key, process,
             denoise, /) -> tuple[jax.Array, StateT]:
        """`x` at `t_next` from `x` at `t` and the model's `(denoised, eps)` at
        `t`. `sample` passes every argument by position, so a solver over
        another algebra names the pair for what it reads (the discrete one
        takes log-probabilities where a Gaussian one takes eps)."""
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


def _ancestral(sigma_t, sigma_s):
    """k-diffusion's `get_ancestral_step` at eta 1: `(sigma_down, sigma_up)`,
    the level a deterministic step goes down to and the fresh noise that
    brings the marginal back to sigma_s."""
    sigma_up = (sigma_s ** 2 * (sigma_t ** 2 - sigma_s ** 2) / sigma_t ** 2) ** 0.5
    return (sigma_s ** 2 - sigma_up ** 2) ** 0.5, sigma_up


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

    def init(self, x, times, process):
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

    def init(self, x, times, process):
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
    """The DDIM update written as an Euler step of the probability flow ODE.
    On a variance exploding schedule it is dx/dsigma = eps."""

    def init(self, x, times, process):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        (alpha_t, sigma_t), (alpha_s, sigma_s) = _rates(process, t, t_next, x)
        dt = sigma_s - sigma_t
        x_0_coeff = (alpha_t * sigma_s - alpha_s * sigma_t) / dt
        dx = (x - x_0_coeff * denoised) / sigma_t
        return x + dx * dt, state


@samplers("euler_ancestral")
@dataclass(frozen=True)
class EulerAncestral:
    """Euler with the ancestral noise injection of k-diffusion
    (`get_ancestral_step`, eta 1). The step goes down to sigma_down, and
    sigma_up of fresh noise brings the marginal back to sigma_s. Integrates a
    `GeneralizedNoiseScheduler`."""

    def init(self, x, times, process):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        _sigma_integrator("EulerAncestral", process)
        (_, sigma_t), (_, sigma_s) = _rates(process, t, t_next, x)
        sigma_down, sigma_up = _ancestral(sigma_t, sigma_s)
        dx = (x - denoised) / sigma_t
        dW = jax.random.normal(key, x.shape) * sigma_up
        return x + dx * (sigma_down - sigma_t) + dW, state


@samplers("heun")
@dataclass(frozen=True)
class Heun:
    """Heun's second order method (Karras et al. 2022, Algorithm 2): an Euler
    step, the derivative re-evaluated at its end, and the average of the two."""

    def init(self, x, times, process):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        (alpha_t, sigma_t), (alpha_s, sigma_s) = _rates(process, t, t_next, x)
        dt = sigma_s - sigma_t
        x_0_coeff = (alpha_t * sigma_s - alpha_s * sigma_t) / dt
        dx_0 = (x - x_0_coeff * denoised) / sigma_t
        x_euler = x + dx_0 * dt

        denoised_next, _ = denoise(x_euler, t_next)
        # When sigma reaches 0 there is no derivative there, so the step is
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

    def init(self, x, times, process):
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


@samplers("kdpm2")
@dataclass(frozen=True)
class KDPM2:
    """k-diffusion's DPM-Solver-2 (`sample_dpm_2`), the update of Diffusers
    0.34.0's `KDPM2DiscreteScheduler`, and with `ancestral` its
    `sample_dpm_2_ancestral` and `KDPM2AncestralDiscreteScheduler`.

    An Euler step to the geometric midpoint of sigma_t and the target level,
    the model read there, and the step from x taken with that midpoint
    derivative. The target is sigma_s, or under `ancestral` the sigma_down of
    k-diffusion's ancestral step with sigma_up of fresh noise added after.
    The midpoint's time comes from the schedule's `t_of_sigma`, so this
    integrates a `GeneralizedNoiseScheduler`.
    """

    ancestral: bool = False

    def init(self, x, times, process):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        schedule = _sigma_integrator("KDPM2", process)
        (_, sigma_t), (_, sigma_s) = _rates(process, t, t_next, x)

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


@samplers("multistep_dpm")
@dataclass(frozen=True)
class MultiStepDPM:
    """A third order multistep integrator of dx/dsigma = eps on a variance
    exploding schedule, from finite differences of the last three eps."""

    def init(self, x, times, process):
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
    return Multistep(jnp.zeros((depth,) + x.shape, x.dtype), rates, rates,
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
        return (sigma_t / sigma_s, -alpha_t * phi, -0.5 * alpha_t * phi,
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


@samplers("dpmsolver_multistep")
@dataclass(frozen=True)
class DPMSolverMultistep:
    """DPM-Solver (Lu et al. 2022, arXiv 2206.00927) and DPM-Solver++
    (arXiv 2211.01095) as multistep integrators in lambda = log(alpha) -
    log(sigma): the four algorithms, three orders and two second-order forms
    of Diffusers 0.34.0's `DPMSolverMultistepScheduler`, with its defaults.

    `dpmsolver++` and `sde-dpmsolver++` integrate the clean prediction, the
    other two eps; the `sde-` forms add the noise term of the SDE solver.
    With h = lambda_t - lambda_s0 over the step and D0, D1, D2 the finite
    differences of the last outputs in lambda, the deterministic `dpmsolver++`
    step is

        x_t = (sigma_t / sigma_s0) x - alpha_t (e^-h - 1) D0 + c D1 + c_2 D2

    and `_dpm_terms` holds each algorithm's coefficients as Diffusers writes
    them. The first step has no history and is first order, the second at
    most second. `lower_order_final` is Diffusers' rule verbatim, which acts
    only in a walk under 15 steps: first order on the last step and at most
    second on the one before. `euler_at_final` makes the last step first
    order whatever the length. A zero target forces the clean limit for
    deterministic and ++ updates. The non-++ SDE first-order limit also
    retains alpha_t*sigma_s/alpha_s*(noise-eps); it requires a finite source
    alpha and a final order reduction. That endpoint is an equation-limit
    extension: Diffusers refuses a literal zero-terminal non-++ config.
    EDM uses the ++ algorithms over its existing process.

    The former DPMSolverPP configuration is order=2, algorithm="dpmsolver++",
    solver_type="midpoint", lower_order_final=False, euler_at_final=False.
    """

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

    def init(self, x, times, process):
        if self.algorithm == "sde-dpmsolver":
            first_at_end = (self.order == 1 or self.euler_at_final
                            or (self.lower_order_final and times.shape[0] - 1 < 15))
            _check_endpoint_domain(
                process, times, source=True, target=not first_at_end,
                reason="sde-dpmsolver requires finite alpha and a lowered order at sigma zero")
        return _multistep(x, times, self.order)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        (alpha_s0, sigma_s0), (alpha_t, sigma_t) = _rates(process, t, t_next, x)
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


@samplers("dpmsolver_singlestep")
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

    def init(self, x, times, process):
        steps = times.shape[0] - 1
        orders = self.order_list(steps)
        if orders:
            with jax.ensure_compile_time_eval():
                _, last_sigma = process.sampler_schedule.rates(times[-1:])
                if bool(jnp.all(last_sigma == 0)):
                    orders[-1] = 1
        if self.algorithm == "dpmsolver" and len(orders) > 1 and orders[1] > 1:
            _check_endpoint_domain(process, times, source=True,
                                   reason="noise-prediction singlestep differences diverge at an alpha=0 anchor")
        if (self.algorithm == "sde-dpmsolver++" and self.solver_type == "heun"
                and 3 in orders[:3]):
            _check_endpoint_domain(process, times, source=True,
                                   reason="third-order SDE Heun differences diverge at an alpha=0 anchor")
        return Singlestep(_multistep(x, times, self.order), x, jnp.asarray(orders, jnp.int32))

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        (alpha_s0, sigma_s0), (alpha_t, sigma_t) = _rates(process, t, t_next, x)
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
            h, ax, b, _, c_heun, c_2, dz = update(3)
            m1, m2 = outputs[-2], outputs[-3]
            from_zero = history.alphas[-3] == 0
            r0 = jnp.where(from_zero, 1.0, (lambdas[-1] - lambdas[-3]) / h)
            r1 = jnp.where(from_zero, 0.5, (lambdas[-2] - lambdas[-3]) / h)
            d1_0 = (m1 - m2) / r1
            d1_1 = (m0 - m2) / r0
            if self.solver_type == "midpoint":
                stepped = ax + b * m2 + c_heun * d1_1
            else:
                d1 = (r0 * d1_0 - r1 * d1_1) / (r0 - r1)
                d2 = 2.0 * (d1_1 - d1_0) / (r0 - r1)
                stepped = ax + b * m2 + c_heun * d1 + c_2 * d2
            if self.algorithm.startswith("sde"):
                stepped = stepped + dz
            endpoint_clean = m0
            if self.solver_type == "heun" and self.algorithm == "dpmsolver++":
                # Combine D1 and D2 before taking the infinite-anchor limit.
                # Their individually divergent leading terms cancel.
                gap = _half_log_snr(alpha_t, jnp.where(sigma_t == 0, 1.0, sigma_t)) - lambdas[-1]
                endpoint_clean = m0 + (gap - 1.0) * (m0 - m1) / (lambdas[-1] - lambdas[-2])
            return jnp.where(from_zero, source_limit(3, endpoint_clean), stepped)

        stepped = lax.switch(order - 1, [first, second, third][:self.order], None)
        terminal = sigma_t <= 0
        next_x = jnp.where(terminal, alpha_t * denoised, stepped)
        return next_x, Singlestep(history.advance(), anchor, state.orders)


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


@samplers("deis")
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

    def init(self, x, times, process):
        return _multistep(x, times, self.order)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        (alpha_s0, sigma_s0), (alpha_t, sigma_t) = _rates(process, t, t_next, x)
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

        order = _tapered_order(self.order, history, self.lower_order_final, False)
        stepped = lax.switch(order - 1, [first, second, third][:self.order], None)
        return stepped, history.advance()


class UniPCState(NamedTuple):
    """`UniPC`'s state: the output history, the sample its last predictor
    left from, and that predictor's order, which its corrector takes."""

    history: Multistep
    last_x: jax.Array
    last_order: jax.Array


def _unipc_weights(rks, hh, B_h, order: int, predictor: bool):
    """UniPC's B(h) weights at `order`: the solution of R rho = b built from
    the relative positions `rks` of the history points (the current point's
    own 1 last). The predictor has no difference at the point it steps to,
    so it drops the last row and column; Diffusers takes 0.5 outright for
    the second-order predictor and the first-order corrector."""
    if order == (2 if predictor else 1):
        return [0.5]
    shape = hh.shape
    batch = shape[0]
    flat = lambda value: jnp.reshape(value, (batch,))
    rks = [flat(rk) for rk in rks[:-1]] + [jnp.ones((batch,), hh.dtype)]
    hh, B_h = flat(hh), flat(B_h)
    columns = order - 1 if predictor else order
    nodes = rks[:columns]
    # Scale the infinite node's Vandermonde column by r**(columns-1).
    # Its lower entries and recovered weight vanish; the leading entry
    # stays finite. This is the limit of the same system, not a lower-order solver.
    inverse = [jnp.where(jnp.isinf(rk), 0.0, 1.0) for rk in nodes]
    normalized = [jnp.where(jnp.isinf(rk), jnp.sign(rk), rk) for rk in nodes]
    rows, b = [], []
    h_phi_k = jnp.expm1(hh) / hh - 1
    factorial = 1
    for i in range(1, columns + 1):
        rows.append(jnp.stack([rk ** (i - 1) * inv ** (columns - i)
                               for rk, inv in zip(normalized, inverse)], axis=-1))
        b.append(h_phi_k * factorial / B_h)
        factorial *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial
    solution = jnp.linalg.solve(jnp.stack(rows, axis=-2), jnp.stack(b, axis=-1)[..., None])[..., 0]
    return [jnp.reshape(solution[:, k] * inverse[k] ** (columns - 1), shape)
            for k in range(columns)]


@samplers("unipc")
@dataclass(frozen=True)
class UniPC:
    """UniPC (Zhao et al. 2023, arXiv 2302.04867), Diffusers 0.34.0's
    `UniPCMultistepScheduler`: a unified predictor and corrector in lambda
    whose weights solve a small linear system over the history's positions.
    Each step first corrects the sample the last predictor produced, with
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
    """

    order: int = 2
    solver_type: Literal["bh1", "bh2"] = "bh2"
    predict_x0: bool = True
    lower_order_final: bool = True
    disable_corrector: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError(f"UniPC's order is at least 1, not {self.order}")

    def init(self, x, times, process):
        corrects_source = times.shape[0] > 2 and 0 not in self.disable_corrector
        _check_endpoint_domain(
            process, times,
            source=corrects_source and (not self.predict_x0 or self.solver_type == "bh1"),
            target=not self.lower_order_final and self.order > 1 and times.shape[0] > 2,
            reason="UniPC corrector/predictor requires finite log-SNR at this endpoint")
        return UniPCState(_multistep(x, times, self.order), x, jnp.ones((), jnp.int32))

    def _b_h(self, hh):
        return hh if self.solver_type == "bh1" else jnp.expm1(hh)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        (alpha_here, sigma_here), (alpha_t, sigma_t) = _rates(process, t, t_next, x)
        history = state.history
        m_here = denoised if self.predict_x0 else eps
        taken, steps = history.taken, history.steps
        lambdas = history.lambdas
        lambda_here = _half_log_snr(alpha_here, sigma_here)

        def corrected(p: int):
            """x at this point, corrected with the p-th order UniC from the
            sample the last predictor left."""
            alpha_s0, sigma_s0, m0 = history.alphas[-1], history.sigmas[-1], history.outputs[-1]
            h = lambda_here - lambdas[-1]
            hh = -h if self.predict_x0 else h
            rks = [(lambdas[-(i + 1)] - lambdas[-1]) / h for i in range(1, p)] + [1.0]
            d1s = [(history.outputs[-(i + 1)] - m0) / rks[i - 1] for i in range(1, p)]
            B_h = self._b_h(hh)
            rhos = _unipc_weights(rks, hh, B_h, p, predictor=False)
            if self.predict_x0:
                base = sigma_here / sigma_s0 * state.last_x - alpha_here * jnp.expm1(hh) * m0
                scale = alpha_here
            else:
                base = alpha_here / alpha_s0 * state.last_x - sigma_here * jnp.expm1(hh) * m0
                scale = sigma_here
            residual = sum((rho * d1 for rho, d1 in zip(rhos[:-1], d1s)), rhos[-1] * (m_here - m0))
            return base - scale * B_h * residual

        disabled = reduce(jnp.logical_or, [taken - 1 == index for index in self.disable_corrector],
                          jnp.zeros((), bool))
        use_corrector = jnp.logical_and(taken > 0, jnp.logical_not(disabled))
        branch = jnp.where(use_corrector, state.last_order, 0)
        x = lax.switch(branch, [lambda _: x] + [
            (lambda p: lambda _: corrected(p))(p) for p in range(1, self.order + 1)], None)

        history = history.push(m_here, alpha_here, sigma_here)
        lambdas, outputs = history.lambdas, history.outputs
        terminal, h = _lambda_step(alpha_here, sigma_here, alpha_t, sigma_t)
        hh = -h if self.predict_x0 else h
        B_h = self._b_h(hh)
        if self.predict_x0:
            base = sigma_t / sigma_here * x - alpha_t * jnp.expm1(hh) * m_here
            scale = alpha_t
        else:
            base = alpha_t / jnp.where(alpha_here == 0, 1.0, alpha_here) * x - sigma_t * jnp.expm1(hh) * m_here
            scale = sigma_t
        base = jnp.where(alpha_here == 0, alpha_t * denoised + sigma_t * eps, base)

        def predicted(p: int):
            """The p-th order UniP step to t_next from the corrected x."""
            if p == 1:
                return base
            rks = [(lambdas[-(i + 1)] - lambdas[-1]) / h for i in range(1, p)] + [1.0]
            d1s = [(outputs[-(i + 1)] - m_here) / rks[i - 1] for i in range(1, p)]
            rhos = _unipc_weights(rks, hh, B_h, p, predictor=True)
            return base - scale * B_h * sum(rho * d1 for rho, d1 in zip(rhos, d1s))

        this_order = jnp.minimum(self.order, steps - taken) if self.lower_order_final else self.order
        this_order = jnp.minimum(this_order, taken + 1)
        stepped = lax.switch(this_order - 1, [
            (lambda p: lambda _: predicted(p))(p) for p in range(1, self.order + 1)], None)
        target_limit = (alpha_t * denoised if self.predict_x0
                        else jnp.where(alpha_here == 0, alpha_t * denoised,
                                       alpha_t / jnp.where(alpha_here == 0, 1.0, alpha_here) * (x - sigma_here * eps)))
        next_x = jnp.where(terminal, target_limit, stepped)
        return next_x, UniPCState(history.advance(), x, this_order)


def _pndm_step(x, eps, rates_t, rates_s):
    """PNDM's transfer (Liu et al. 2022, eq. 9), DDIM from t to s in the
    rationalized form Diffusers' `_get_prev_sample` evaluates."""
    (alpha_t, sigma_t), (alpha_s, sigma_s) = rates_t, rates_s
    denominator = alpha_t ** 2 * sigma_s + alpha_t * sigma_t * alpha_s
    return (alpha_s / alpha_t) * x - (alpha_s ** 2 - alpha_t ** 2) * eps / denominator


@samplers("pndm")
@dataclass(frozen=True)
class PNDM:
    """PNDM (Liu et al. 2022, arXiv 2202.09778), Diffusers 0.34.0's
    `PNDMScheduler`: Adams-Bashforth over eps with DDIM as the transfer,
    fourth order once four outputs are in hand. The warmup is the paper's,
    three pseudo Runge-Kutta steps of four model evaluations each (the
    stages at the interval's midpoint and end), which seed the history with
    each step's first eps; under `skip_prk_steps`, the PLMS form Stable
    Diffusion runs, the first step is a predictor-corrector pair (an Euler
    step, eps re-read at its end, the step retaken with the mean) and the
    orders grow from there. Diffusers takes each step's stride from its
    grid; here the stride is `t_next - t`, so any grid works. The transfer
    divides by alpha_t, so the walk cannot start where alpha is 0.
    """

    skip_prk_steps: bool = False

    def init(self, x, times, process):
        _check_endpoint_domain(process, times, source=True,
                               reason="PNDM stage differences are divided by the source alpha")
        return jnp.zeros((4,) + x.shape, x.dtype), jnp.zeros((), jnp.int32)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        schedule = process.sampler_schedule
        rates_t, rates_s = _rates(process, t, t_next, x)
        outputs, count = state
        outputs = _push(outputs, eps)
        e0, e1, e2, e3 = outputs[-1], outputs[-2], outputs[-3], outputs[-4]

        def runge_kutta(_):
            t_mid = 0.5 * (t + t_next)
            rates_mid = broadcast_rates(schedule, t_mid, x)
            k2 = denoise(_pndm_step(x, eps, rates_t, rates_mid), t_mid)[1]
            k3 = denoise(_pndm_step(x, k2, rates_t, rates_mid), t_mid)[1]
            k4 = denoise(_pndm_step(x, k3, rates_t, rates_s), t_next)[1]
            return _pndm_step(x, eps / 6 + k2 / 3 + k3 / 3 + k4 / 6, rates_t, rates_s)

        def predictor_corrector(_):
            eps_next = denoise(_pndm_step(x, eps, rates_t, rates_s), t_next)[1]
            return _pndm_step(x, (eps_next + eps) / 2, rates_t, rates_s)

        def adams(k: int):
            if k == 2:
                combined = (3 * e0 - e1) / 2
            elif k == 3:
                combined = (23 * e0 - 16 * e1 + 5 * e2) / 12
            else:
                combined = (1 / 24) * (55 * e0 - 59 * e1 + 37 * e2 - 9 * e3)
            return _pndm_step(x, combined, rates_t, rates_s)

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
        polynomial = [0.0] + polynomial
        polynomial = [polynomial[i] - node * (polynomial[i + 1] if i + 1 < len(polynomial) else 0.0)
                      for i in range(len(polynomial))]
        scale = scale * (nodes[j] - node)
    integral = sum(coefficient * (b ** (i + 1) - a ** (i + 1)) / (i + 1)
                   for i, coefficient in enumerate(polynomial))
    return integral / scale


@samplers("lms")
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

    def init(self, x, times, process):
        sigmas = jnp.ones((self.order, x.shape[0]) + (1,) * (x.ndim - 1), jnp.float32)
        return jnp.zeros((self.order,) + x.shape, x.dtype), sigmas, jnp.zeros((), jnp.int32)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        _sigma_integrator("LMS", process)
        (_, sigma_t), (_, sigma_s) = _rates(process, t, t_next, x)
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


@samplers("consistency")
@dataclass(frozen=True)
class Consistency:
    """Multistep consistency sampling (Song et al. 2023, Algorithm 1), the
    update of Diffusers 0.34.0's `LCMScheduler`: the clean prediction is
    noised again to the next level with fresh noise, x_s = alpha_s x_0 +
    sigma_s z, and the last step keeps x_0 as it is. A latent consistency
    model's x_0 is the consistency function's output, which
    `ConsistencyBoundary` reads out of the model's prediction.
    """

    def init(self, x, times, process):
        return jnp.zeros((), jnp.int32), jnp.asarray(times.shape[0] - 1, jnp.int32)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        _, (alpha_s, sigma_s) = _rates(process, t, t_next, x)
        count, steps = state
        noise = jax.random.normal(key, x.shape, dtype=jnp.float32)
        stepped = jnp.where(count == steps - 1, denoised, alpha_s * denoised + sigma_s * noise)
        return stepped, (count + 1, steps)


@samplers("tcd")
@dataclass(frozen=True)
class TCD:
    """Trajectory consistency sampling (Zheng et al. 2024, arXiv 2402.19159),
    Diffusers 0.34.0's `TCDScheduler`: the deterministic DDIM step lands at
    (1 - eta) t_next, and the forward process noises it up to t_next with
    fresh noise, the paper's gamma-sampling with gamma = `eta`. At eta 0 it
    is DDIM; at eta 1 the step goes through the clean end of the schedule.
    A tabulated schedule reads the intermediate time at its truncated
    index, as the reference floors it. The last step of a walk lands on the
    grid's end itself and takes no noise.
    """

    eta: float = 0.3

    def __post_init__(self) -> None:
        if not 0 <= self.eta <= 1:
            raise ValueError(f"TCD's eta is in [0, 1], not {self.eta}")

    def init(self, x, times, process):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        _, (alpha_s, sigma_s) = _rates(process, t, t_next, x)
        alpha_mid, sigma_mid = broadcast_rates(process.sampler_schedule, (1 - self.eta) * t_next, x)
        noised = alpha_mid * denoised + sigma_mid * eps
        if self.eta == 0:
            return noised, state
        ratio = alpha_s / alpha_mid
        noise = jax.random.normal(key, x.shape, dtype=jnp.float32)
        return ratio * noised + jnp.sqrt(1 - ratio ** 2) * noise, state


__all__ = ["Solver", "DDPM", "DDIM", "Euler", "EulerAncestral", "Heun", "RK4", "KDPM2",
           "MultiStepDPM", "DPMSolverMultistep", "DPMSolverSinglestep", "DEIS", "UniPC",
           "PNDM", "LMS", "Consistency", "TCD"]
