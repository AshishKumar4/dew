"""Probability-flow and ancestral solvers that integrate in sigma."""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct
from jax import lax

from dew.diffusion.schedules import RectifiedFlow, expand
from dew.diffusion.schedules.common import ancestral_step
from dew.nn.derivatives import straight_through
from dew.nn.kernels.generation import linear_map
from dew.nn.precision import divided, summed_once, two_sum
from dew.nn.programs import on_host
from dew.nn.sharding import summed_like

from .common import _check_gaps, _grid_levels, _history_gaps, _push, _sigma_integrator, _unit_noise


def _flowed_value(x, coefficient, denoised, dt, sigma_t):
    """x + (dt / sigma_t) (x - coefficient denoised), its rounding held (`_flowed`)."""
    # The source's x + derivative dt, its difference taken at half scale where x or x_0 nears the float's
    # range (their difference past it, the step finite as DDIM's form is, `_euler_step`; a normal value halved
    # would flush to zero on a device that flushes subnormals, 2^-126 less -2^-126 then 2^-126, not 0: train
    # 1419's P0), and kept whole: its rounding carried beside it (`two_sum`), so a difference x_0's size
    # cancels to x's small residual keeps it (1 from -1 + 2^-24, half of which rounds to 1, steps to 2^-25 as
    # the DDIM form does; train 1389's P0). Each part is scaled by the step's fraction of its level, dt /
    # sigma_t, rounded once, not divided by the level and multiplied by the interval, two roundings of the
    # part's size: from 47/64 to 47/128 the fraction is -1/2 exactly, and 1 less its x_0 of -1 steps to 0, as
    # DDIM does, not 2^-24 (train 1407's P0); the fraction is at most 1/2 here, so no part passes the float's
    # range.
    # Near the least normal, the other way: scaled up by the mantissa's span, so a difference of two
    # normals, a subnormal, stays normal until the result is formed (2 2^-126 less 2.5 2^-126 halved onto
    # 2 2^-126 is 2.25 2^-126, not 2 2^-126 with the half-step flushed; train 1419's review). The scaled
    # sum is the step's, by a power of two exactly, divided back once.
    float_format = jnp.finfo(x.dtype)
    magnitude = jnp.maximum(jnp.abs(x), jnp.abs(coefficient * denoised))
    up = jnp.asarray(2.0 ** (float_format.nmant + 1), x.dtype)
    scale = jnp.where(magnitude >= 2.0 ** (float_format.maxexp - 3), jnp.asarray(0.5, x.dtype),
                      jnp.where(magnitude < 2.0 ** (float_format.minexp + float_format.nmant + 1), up,
                                jnp.asarray(1.0, x.dtype)))
    high, low = two_sum(scale * x, -(scale * coefficient) * denoised)
    fraction = divided(dt, sigma_t)
    return (scale * x + high * fraction + low * fraction) / scale


def _flowed(x, coefficient, denoised, dt, sigma_t):
    """A near Euler update (a rectified flow's, a variance-exploding walk's high steps), x + (D / sigma_t) dt
    of its operands' difference D = x - c x_0, its value compensated (`_flowed_value`) and its derivatives
    the update's at every order (`straight_through`): a map linear in x and x_0 at the level's c, dt and
    sigma_t (`linear_map`), each of its tangents formed in the source's own order, D over the level and then
    the interval, so no fraction dt / sigma_t is formed nor its tangent (a target level's tangent of 2^115 at
    sigma_t 2^-16 is a fraction's of 2^131, where the update's is 2^115; train 1461's P0), each tangent in
    range wherever the source's is. D is formed whole where it is in range (x_0's tangent of 2^-125 from 1 to
    1/2 steps x by 2^-126, where halving D first flushes each half; train 1452's P0), at half scale taken
    twice where it is not (tangents 2^127 and -2^127), never weighed apart (2 2^127 less 2 2^127; train
    1430's P0); it pulls a cotangent back as each operand's weight, which forms no difference, summed as its
    operand broadcast (`sharding.summed_like`: a replicated prediction's beside a split sample's, train
    1461's P0). Where the operands' tangents' difference itself passes the range (x' = 2^127 under x_0' =
    -2^127), the source's x + ((x - x_0) / sigma_t) dt forms it too, so no order of the update is held finite
    there (row 5426's ruling, tests/contract_support.py's `assert_contract`; train 1452's F3b)."""
    level = (coefficient, dt, sigma_t, lax.stop_gradient(x), lax.stop_gradient(denoised))
    applied = linear_map(_flowed_applied, _flowed_pulled, level, (x, denoised))
    return straight_through(applied, lax.stop_gradient(_flowed_value(x, coefficient, denoised, dt, sigma_t)))


def _flowed_applied(level, moving):
    coefficient, dt, sigma_t, _, _ = level
    x, denoised = moving
    whole = x - coefficient * denoised
    formed = jnp.isfinite(whole)
    held = jnp.where(formed, whole, 0)
    half = 0.5 * jnp.where(formed, 0, x) - (0.5 * coefficient) * jnp.where(formed, 0, denoised)
    halved = divided(half, sigma_t) * dt
    return x + jnp.where(formed, divided(held, sigma_t) * dt, halved + halved)


def _flowed_pulled(level, cotangent):
    coefficient, dt, sigma_t, x, denoised = level
    # The transpose's own order, the interval first (a cotangent of 2^127 over sigma 1/2 passes the range,
    # under the interval -1/8 first it does not; train 1461's review).
    moved = divided(cotangent * dt, sigma_t)
    return summed_like(cotangent + moved, x), summed_like(-(coefficient * moved), denoised)


def _euler_step(x, denoised, source, target, *, flow: bool = False):
    """One Euler step of the probability-flow ODE, in sigma.

    Returns `(x at the target level, the derivative, the x_0 coefficient, the interval)`. A second-order
    method re-evaluates the derivative at the end of this step and needs the coefficient and the interval it
    was taken over; `flow` where the process walks a rectified flow (`RectifiedFlow`). A step between equal
    levels, which a grid that ends where its final sigma repeats takes (`final_sigmas_type="sigma_min"`), has
    no length and leaves x as it is, as the source's `x + derivative * dt` does.
    """
    (alpha_t, sigma_t), (alpha_s, sigma_s) = source, target
    dt = sigma_s - sigma_t
    # A rectified flow's (alpha = 1 - sigma) is one exactly: its secant would divide two rounded
    # cross-products, 1 - 2^-24 from 47/64 to 47/128 (train 1407's P0). Elsewhere equal levels make the
    # numerator 0 too, any finite coefficient scaled by the zero interval, divided as IEEE divides
    # (`divided`): inside a walk's scan XLA takes the interval as a constant and multiplies by its rounded
    # reciprocal.
    x_0_coeff = (jnp.ones_like(dt) if flow
                 else divided(alpha_t * sigma_s - alpha_s * sigma_t, jnp.where(dt == 0, 1, dt)))
    derivative = divided(x - x_0_coeff * denoised, sigma_t)
    # The source's x + derivative dt, summed as alpha_s x_0 + sigma_s eps, DDIM's form of it: from a
    # large sigma the source's sum cancels two terms of x's size (x is 4e4 at a zero-SNR sigma of
    # 4096) and keeps dt's rounding times the derivative, and near pure noise on a VP schedule the
    # secant coefficient is in the thousands; this form has neither, and a step to sigma 0 lands on
    # the clean prediction itself. A step of no length leaves x exactly as it is.
    stepped = alpha_s * denoised + divided(x - alpha_t * denoised, sigma_t) * sigma_s
    # Between near sigmas of a constant signal (2 sigma_s >= sigma_t, a variance-exploding walk's high
    # steps), the source's x + derivative dt cancels nothing, where the form above sums its clean and
    # noise parts, each of x's size, back to x's small step (ledger 2541); a long step from a zero-SNR
    # sigma keeps the form above (ledger 3475).
    near = (alpha_t == alpha_s) & (2 * sigma_s >= sigma_t) & (sigma_s != 0)
    # A rectified flow's levels (alpha = 1 - sigma, a source's FlowMatch schedule, `RectifiedFlow`) hold x
    # as a convex sum of the clean sample and the noise, never larger than they, and its x_0 coefficient is
    # one: between near levels (2 sigma_s >= sigma_t) the source's x + derivative dt cancels nothing, where
    # the form above rounds x through x_0 and back (1.6x Diffusers' own walk's error from float64 over
    # SD3's FlowMatch files, row 13654); a long step down (an exponential grid's last) keeps the form
    # above, which the derivative's rounding at a small sigma the source's sum would carry.
    walks_flow = flow & (2 * sigma_s >= sigma_t)
    moved = _flowed(x, x_0_coeff, denoised, dt, sigma_t)
    stepped = jnp.where(near | walks_flow, moved, stepped)
    return jnp.where(dt == 0, x, stepped), derivative, x_0_coeff, dt


@struct.dataclass
class Euler:
    """The DDIM update written as an Euler step of the probability flow ODE.
    On a variance exploding schedule it is dx/dsigma = eps."""

    def init(self, x, times, process, *, key):
        return ()

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        stepped, _, _, _ = _euler_step(x, denoised, process.rates(t, like=x), process.rates(t_next, like=x),
                                       flow=isinstance(process.sampler_schedule, RectifiedFlow))
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
        sigma_down, sigma_up = ancestral_step(sigma_t, sigma_s)
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
        with on_host():
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
        x_euler, dx_0, x_0_coeff, dt = _euler_step(x, denoised, source, target,
                                                   flow=isinstance(process.sampler_schedule, RectifiedFlow))

        # A row corrects where its step reaches a positive sigma: a step of no length too, which leaves x as
        # it is and reads both slopes for its derivative in its levels (`held` below).
        def corrected(operand):
            denoised_next, _ = denoise(x_euler, t_next)
            # When sigma reaches 0 there is no derivative there, so a row
            # there takes the Euler step.
            safe_sigma_s = jnp.where(sigma_s > 0, sigma_s, 1.0)
            dx_1 = (x_euler - x_0_coeff * denoised_next) / safe_sigma_s
            # The source's mean, summed first; where the step has no length and the sum alone passes the
            # range (two slopes of 2^127, of one sign), the lesser plus half their difference, which is in it,
            # and no common factor XLA folds back into the sum; a sum in range as it is (two least normal
            # slopes, whose halves a device flushes; review).
            summed = 0.5 * (dx_0 + dx_1)
            lesser = jnp.minimum(dx_0, dx_1)
            between = lesser + 0.5 * (jnp.maximum(dx_0, dx_1) - lesser)
            slope = jnp.where((dt == 0) & ~jnp.isfinite(summed), between, summed)
            # A step of no length is x itself, its tangent in the levels its two slopes' mean, each read where
            # it was (a guidance whose state moves between them, APG's momentum, reads two; review): a slope
            # past the range adds none there, where its product with the zero interval was NaN (train 1474's
            # P0: x 0 under a clean prediction of 2^127 at sigma 1/2, a slope of -2^128).
            held = jnp.where(jnp.isfinite(slope) | (dt != 0), slope, 0.0)
            return jnp.where(sigma_s > 0, x + held * dt, x_euler)

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
                target, sigma_up = ancestral_step(sigma_t, following)
                sigma_mid = jnp.sqrt(sigma_t) * jnp.sqrt(target)
            else:
                target, sigma_up = following, 0.0
                sigma_mid = jnp.sqrt(target) * jnp.sqrt(sigma_t)
            # The midpoint at the level its model reads: a level carried through a time comes back to the
            # schedule's rounding of both, several ulps off where the grid is steep, and the source steps
            # at the level it hands its model (row 13657).
            t_mid = schedule.t_of_sigma(sigma_mid.reshape(-1))
            sigma_mid = expand(schedule.sigmas(t_mid), x)
            dx = (x - denoised) / sigma_t
            x_mid = x + dx * (sigma_mid - sigma_t)
            denoised_mid, _ = denoise(x_mid, t_mid)
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
            # `x + eps * dt + 0.5 * dx_2 * dt ** 2 + dx_3 * dt ** 3 / 6` rounded once (`summed_once`), as
            # near its exact value as FlaxDiff's sum with its products contracted into its additions, whatever
            # `divided`'s lowering leaves XLA to fuse around it (row 7979).
            return summed_once((x, divided(dx_3 * dt ** 3, 6)), ((eps, dt), (0.5 * dx_2, dt ** 2)))

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
