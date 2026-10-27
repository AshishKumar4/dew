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

import jax
import jax.numpy as jnp
from flax import struct
from jax.typing import ArrayLike

from dew.diffusion.process import Denoiser, Process
from dew.diffusion.schedules import RectifiedFlow, expand
from dew.nn.kernels.generation import linear_tangent
from dew.nn.precision import at_least_fp32, divided

from .guidance import Guidance, Walk
from .sample import sample
from .solvers.common import _check_endpoint_domain


@struct.dataclass
class GaussianTransition:
    """An isotropic Gaussian transition with one variance per batch row.

    Sampling and density arithmetic use float32, or the mean's and
    variance's own dtype where it is wider, and densities and KL sum over
    the sample dimensions. A zero variance makes the transition
    deterministic. Sampling then returns the mean, and `log_prob` is NaN,
    because a Dirac measure has no density with respect to Lebesgue measure.
    """

    mean: jax.Array
    variance: jax.Array

    @property
    def stochastic(self) -> jax.Array:
        return self.variance > 0

    def _dtype(self) -> jnp.dtype:
        return at_least_fp32(jnp.result_type(self.mean, self.variance))

    def sample(self, key: jax.Array) -> jax.Array:
        """Draw a next state. The normals are float32 at every precision, so
        a wider transition draws the same states, held more exactly."""
        mean = jnp.asarray(self.mean, self._dtype())
        variance = jnp.asarray(self.variance, mean.dtype)
        def draw():
            scale = jnp.sqrt(variance).reshape(
                (mean.shape[0],) + (1,) * (mean.ndim - 1))
            return mean + scale * jax.random.normal(key, mean.shape, dtype=jnp.float32).astype(mean.dtype)

        return jax.lax.cond(jnp.all(variance == 0), lambda: mean, draw)

    def log_prob(self, value: ArrayLike) -> jax.Array:
        """Return the joint log density of an observed next state, one value per row.

        Raises `ValueError` when `value` and the mean differ in shape.
        """
        mean = jnp.asarray(self.mean, self._dtype())
        variance = jnp.asarray(self.variance, mean.dtype)
        value = jnp.asarray(value, mean.dtype)
        if value.shape != mean.shape:
            raise ValueError("a transition density scores one next state per mean")
        safe_variance = jnp.where(variance > 0, variance, 1)
        dimensions = math.prod(value.shape[1:])
        # The form of JAX's isotropic `multivariate_normal.logpdf` (a scalar covariance): log(2 pi) and
        # the log variance summed apart, so a variance near float32's largest does not overflow 2 pi
        # times it. Each row has its own variance, which that function would take one row at a time.
        log_prob = -(_half_mahalanobis(value, mean, safe_variance)
                     + 0.5 * dimensions * (jnp.log(2 * math.pi) + jnp.log(safe_variance)))
        return jnp.where(variance > 0, log_prob, jnp.nan)

    def kl(self, reference_mean: ArrayLike) -> jax.Array:
        """Return the KL divergence to a reference transition with mean `reference_mean`.

        The reference has the same variance, because the variance does not
        depend on the policy. Equal Dirac measures have KL zero, and
        distinct ones have infinite KL.
        """
        mean = jnp.asarray(self.mean, self._dtype())
        variance = jnp.asarray(self.variance, mean.dtype)
        reference_mean = jnp.asarray(reference_mean, mean.dtype)
        if reference_mean.shape != mean.shape:
            raise ValueError("reference and policy transition means must share one shape")
        safe_variance = jnp.where(variance > 0, variance, 1)
        apart = jnp.any(mean != reference_mean, axis=tuple(range(1, mean.ndim)))
        deterministic = jnp.where(apart, jnp.inf, 0.0)
        return jnp.where(variance > 0, _gaussian_kl(mean, reference_mean, safe_variance),
                         jnp.where(variance == 0, deterministic, jnp.nan))


def _exponent(x: jax.Array) -> jax.Array:
    """Each `|x|`'s binary exponent `e`, `2**e <= |x| < 2**(e + 1)` where `x` is a normal number, read
    from its bits: a zero, or a subnormal, which XLA:CPU reads as zero, gives the least normal's less
    one, and an infinity or a NaN the largest's plus one."""
    limits = jnp.finfo(x.dtype)
    unsigned = jnp.uint32 if limits.bits == 32 else jnp.uint64
    field = (jax.lax.bitcast_convert_type(jnp.abs(x), unsigned) >> limits.nmant) & ((1 << limits.nexp) - 1)
    return field.astype(jnp.int32) - (limits.maxexp - 1)


def _scaled(x: jax.Array, power: jax.Array) -> jax.Array:
    """`x * 2**power`, exact where the result is a normal number: the power in parts each a normal
    number's, all of its sign, applied in turn, so each partial product lies between `x` and the
    result and leaves no range the result keeps."""
    limits = jnp.finfo(x.dtype)
    unsigned = jnp.uint32 if limits.bits == 32 else jnp.uint64
    power = jnp.asarray(power, jnp.int32)
    for _ in range(4):
        part = jnp.clip(power, limits.minexp, limits.maxexp - 1)
        biased = (part + (limits.maxexp - 1)).astype(unsigned) << limits.nmant
        x = x * jax.lax.bitcast_convert_type(biased, x.dtype)
        power = power - part
    return x


FLOOR = -4096
"""An exponent below any a float32 or float64 product reaches, for a row with no nonzero term."""


def _split(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Each `x` as its significand, in [1, 2) or zero, and its exponent (`_exponent`), `x = significand *
    2**exponent` for a normal `x`."""
    exponent = _exponent(x)
    return _scaled(x, -exponent), exponent


Split = tuple[jax.Array, jax.Array]
"""A value as a quotient and the power of two it is restored by (`_quotient`)."""


def _shift(top: jax.Array, headroom: int = 0) -> jax.Array:
    """The power of two values whose largest exponent is `top` are taken down by so their arithmetic
    stays in float32's normal range, `headroom` exponents above it kept for growth: down only by what
    overflow requires where they reach float32's largest, so every value a plain computation keeps is
    kept; up to their largest's own exponent where they lie below 2**-63, so what a plain computation
    would flush is kept, and nothing it keeps is lost; and none in between, where the computation is
    the plain one."""
    return jnp.where(top + headroom > 126, top + headroom - 126, jnp.where(top < -63, top, 0))


def _differences(left: jax.Array, right: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`left - right` per coordinate as values and shifts, `(left - right) = values * 2**shifts`: the
    plain subtraction, a coordinate's two operands shifted first as `_shift` asks (`2**127 - -2**127`,
    two neighbouring tiny values), so every coordinate a plain subtraction holds is that subtraction
    and no other coordinate moves."""
    shift = _shift(jnp.maximum(_exponent(left), _exponent(right)))
    return _scaled(left, -shift) - _scaled(right, -shift), shift


def _summed_products(left: jax.Array, left_shift: jax.Array, right: jax.Array,
                     right_shift: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`sum(left * 2**left_shift * right * 2**right_shift)` over each row as a value and a power: each
    product the plain product of the two values (their significands', `_split`, which rounds as it
    does), the row's products shifted together as `_shift` asks for its largest and their sum,
    summed plainly: every term a plain float32 sum keeps, it keeps
    (`[2**120, -2**120, 2**-100]` sums to `2**-100`), and a row no product of which overflows is the
    plain sum. A product below float32's least beside products that reach its largest and cancel
    (`[2**126, -2**126, 2**-146]`) is lost as the plain sum loses it, which flushes it: keeping it
    would take a second sum at its own scale, which the gate's rule, no accuracy below plain
    float32's, does not ask."""
    (left_part, left_power), (right_part, right_power) = _split(left), _split(right)
    products = left_part * right_part
    powers = jnp.where(products != 0, left_power + right_power + left_shift + right_shift, FLOOR)
    count = math.prod(left.shape[1:])
    top = jnp.max(powers, axis=tuple(range(1, left.ndim)))
    excess = _shift(top, 2 + math.ceil(math.log2(max(count, 1))))
    return _scaled(products, powers - expand(excess, powers)).sum(axis=tuple(range(1, left.ndim))), excess


def _quotient(numerator: jax.Array, power: jax.Array, divisor: jax.Array) -> Split:
    """`numerator * 2**power / divisor`, per row, as a quotient and its power: the significands'
    quotient (`_split`), in (1/2, 2), which rounds as the plain division does wherever that is a
    normal number, and the exponents' difference, restored once by whoever reads it (`_scaled`)."""
    (top, top_power), (bottom, bottom_power) = _split(numerator), _split(divisor)
    return top / bottom, top_power + power - bottom_power


def _sum(left: Split, right: Split) -> jax.Array:
    """`left + right`, two values given as quotients and powers (`_quotient`), the plain sum of the two
    restored values, both shifted first as `_shift` asks for the larger and their sum, and restored
    once."""
    (left_value, left_power), (right_value, right_power) = left, right
    top = jnp.maximum(jnp.where(left_value != 0, left_power + _exponent(left_value), FLOOR),
                      jnp.where(right_value != 0, right_power + _exponent(right_value), FLOOR))
    shift = _shift(top, 1)
    return _scaled(_scaled(left_value, left_power - shift) + _scaled(right_value, right_power - shift), shift)


def _divergence_parts(mean: jax.Array, reference: jax.Array, variance: jax.Array):
    """The parts of `|mean - reference|**2 / (2 variance)` (`_divergence`): the differences and their
    shifts (`_differences`), and the KL as a quotient and its power (`_quotient`), the half in its power."""
    difference, shift = _differences(mean, reference)
    error, excess = _summed_products(difference, shift, difference, shift)
    return difference, shift, _quotient(error, excess - 1, variance)


def _divergence(mean: jax.Array, reference: jax.Array, variance: jax.Array) -> jax.Array:
    """`|mean - reference|**2 / (2 variance)`, one row a `variance`: the plain float32 formula wherever it
    stays below float32's largest, bit for bit in a normal result, and elsewhere the same arithmetic
    with its operands taken down by only what keeps it in range and the power restored once at the
    end (`_scaled`): a squared difference past float32's largest over a variance as wide, or a
    variance whose double overflows; and taken up where every value lies below 2**-63, so it keeps
    what the plain formula would flush (`_shift`). It keeps every term the plain formula keeps. Its
    derivatives are `_gaussian_kl`'s."""
    *_, (quotient, power) = _divergence_parts(mean, reference, variance)
    return _scaled(quotient, power)


@jax.custom_jvp
def _directional(mean: jax.Array, reference: jax.Array, variance: jax.Array) -> jax.Array:
    """`_divergence`, whose forward mode is the plain `sum(d * (t_mean - t_reference)) / v - KL * t_v /
    v` formed as the KL is: the tangents' difference (`_differences`), its sum of products against the
    difference (`_summed_products`), each quotient as its significands' (`_quotient`), and the two terms
    joined (`_sum`), each taken down only by what keeps it in range. Its tangent reads the tangents'
    exponents, so it is forward mode alone: reverse mode is `_kl_pullback`'s (`linear_tangent`), and a
    derivative of the reverse mode in forward mode is not defined."""
    return _divergence(mean, reference, variance)


@_directional.defjvp
def _directional_jvp(primals, tangents):
    """`_directional`'s tangent, the plain formula's, each step taken down only where it would overflow."""
    mean_tangent, reference_tangent, variance_tangent = tangents
    difference, shift, (quotient, power) = _divergence_parts(*primals)
    along, excess = _summed_products(difference, shift, *_differences(mean_tangent, reference_tangent))
    stretch, stretch_power = _split(variance_tangent)
    variance_term = _quotient(-(quotient * stretch), power + stretch_power, primals[2])
    moved = _sum(_quotient(along, excess, primals[2]), variance_term)
    return _scaled(quotient, power), moved


def _kl_pullback(saved, operands, cotangent):
    """The KL's reverse mode, the plain formula's: each element's `cotangent * (mean - reference) /
    variance`, the reference's its negation, and the variance's `-cotangent * KL / variance`, each product
    of significands (`_split`) with its exponents added, each quotient as its significands' (`_quotient`),
    and each power restored once (`_scaled`)."""
    del saved
    difference, shift, (quotient, power) = _divergence_parts(*operands)
    variance = operands[2]
    (scale, scale_power), (part, part_power) = _split(cotangent), _split(difference)
    level, spread = _split(variance)
    # The plain order, the product then the division, and the division IEEE's (`divided`), where
    # XLA:CPU takes a division by a broadcast as a product with its reciprocal.
    pulled = _scaled(divided(expand(scale, part) * part, expand(level, part)),
                     expand(scale_power - spread, part) + part_power + shift)
    to_variance = _quotient(-(scale * quotient), scale_power + power, variance)
    return pulled, -pulled, _scaled(*to_variance)


def _measured(mean: jax.Array, reference: jax.Array, variance: jax.Array) -> tuple[jax.Array, tuple]:
    """`_divergence`, with nothing its reverse mode keeps (`linear_tangent`'s `forward`)."""
    return _divergence(mean, reference, variance), ()


@jax.custom_jvp
def _half_mahalanobis(value: jax.Array, mean: jax.Array, variance: jax.Array) -> jax.Array:
    """`|value - mean|**2 / (2 variance)` per row, the density's quadratic term, as the KL forms it
    (`_divergence`): the plain float32 formula bit for bit wherever it stays in range, and finite
    wherever the term is (a residual of 2**64 under a variance of 2**126; a value of 2**127 beside a
    mean of -2**127). Its derivatives are the plain formula's at every order (`_half_mahalanobis_jvp`)."""
    return _divergence(value, mean, variance)


@_half_mahalanobis.defjvp
def _half_mahalanobis_jvp(primals, tangents):
    """The plain formula's tangent, `sum((value - mean) * (t_value - t_mean)) / variance - term * t_variance
    / variance`, formed from the residual and the term as they are held, shifted to range
    (`_divergence_parts`), and restored once: finite wherever the derivative is. It is linear in the
    tangents, so reverse mode transposes it, and its coefficients are functions of the primals, so it
    differentiates again."""
    value, mean, variance = primals
    along_value, along_mean, along_variance = tangents
    difference, shift, (quotient, power) = _divergence_parts(value, mean, variance)
    # The residual over the variance's power of two, in range wherever the derivative is, its products
    # with the tangents summed, then over the variance's significand, one division a row: the plain
    # `sum(residual * tangent) / variance` in range, bit for bit, the powers of two exact. Reverse mode
    # meets the same factors in the other order, the residual's scale applied last.
    level, spread = _split(variance)
    scaled = _scaled(difference, shift - expand(spread, shift))
    moved = (scaled * (along_value - along_mean)).sum(axis=tuple(range(1, value.ndim))) / level
    stretch = _scaled(*_quotient(-quotient, power, variance))
    return _half_mahalanobis(value, mean, variance), moved + stretch * along_variance


_gaussian_kl = linear_tangent(_measured, _kl_pullback, _directional)
"""`_divergence` differentiated through Dew's seam (`dew.nn.kernels.generation.linear_tangent`): its
forward mode `_directional`'s, its reverse mode `_kl_pullback`'s, each forming every difference,
product and quotient from operands scaled to range."""


def flow_transition(x: ArrayLike, velocity: ArrayLike, sigma: ArrayLike,
                    sigma_next: ArrayLike, *, noise_level: float = 0.7) -> GaussianTransition:
    """Euler-Maruyama over the physical noise rate, with 0 <= sigma_next <= sigma <= 1.

    A rectified flow's noise rate is its own physical time, which is what
    `FlowSDE` reads off the schedule and hands over here. x and velocity are
    [batch, ...]; rates are scalars or [batch]. All density arithmetic is
    float32, or x's dtype where it is wider, a sampler's state carrying one
    dtype through its walk whatever the velocity's. Variance is
    sigma^2 times the elapsed rate. Invalid rates produce non-finite
    transitions. At zero noise or zero elapsed rate the result is
    deterministic.
    """
    if not math.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and non-negative")
    dtype = at_least_fp32(jnp.result_type(x))
    x = jnp.asarray(x, dtype)
    velocity = jnp.asarray(velocity, dtype)
    if x.ndim < 2 or velocity.shape != x.shape:
        raise ValueError("x and velocity must share a [batch, ...] sample shape")
    sigma = jnp.broadcast_to(jnp.asarray(sigma, dtype), (x.shape[0],))
    sigma_next = jnp.broadcast_to(jnp.asarray(sigma_next, dtype), sigma.shape)
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


@struct.dataclass
class FlowSDE:
    """Samples a rectified-flow `Process` with Flow-GRPO's Euler-Maruyama solver.

    Process times may be resolution-shifted. The transition integrates in
    the noise rate sigma that results, as the reference scheduler does.
    Predictions are converted to clean samples and noise by the denoiser.
    A process without a rectified-flow schedule raises `ValueError`, and so
    does a grid that evaluates a prediction dividing by the signal rate
    (`PredictionTransform.divides_by_signal`, epsilon's) where that rate is
    0, as at pure noise: a walk from an interior time takes it.
    """

    noise_level: float = struct.field(pytree_node=False, default=0.7)

    def validate(self, process: Process, times: jax.Array | None = None) -> None:
        """Refuse a process this solver cannot walk, and with the concrete grid `times`, a grid
        whose prediction is undefined at a point it is evaluated at: every one but the last."""
        if not isinstance(process.sampler_schedule, RectifiedFlow):
            raise ValueError("FlowSDE requires a rectified-flow schedule")
        if times is not None and process.prediction.divides_by_signal:
            _check_endpoint_domain(process, times, source=True,
                                   reason=f"a {type(process.prediction).__name__} divides by the signal rate")

    def init(self, x: jax.Array, times: jax.Array, process: Process, *,
             key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """No transition yet: a NaN density and no noise, per row."""
        self.validate(process, times)
        return jnp.full((x.shape[0],), jnp.nan, at_least_fp32(x.dtype)), jnp.zeros((x.shape[0],), bool)

    def transition(self, x: jax.Array, t: jax.Array, t_next: jax.Array,
                   denoised: jax.Array, eps: jax.Array, process: Process) -> GaussianTransition:
        self.validate(process)
        _, sigma = process.sampler_schedule.rates(t)
        _, following = process.sampler_schedule.rates(t_next)
        return flow_transition(x, eps - denoised, sigma, following, noise_level=self.noise_level)

    def step(self, x: jax.Array, t: jax.Array, t_next: jax.Array,
             denoised: jax.Array, eps: jax.Array, state: tuple[jax.Array, jax.Array], key: jax.Array,
             process: Process,
             denoise: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
             /) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """Draw the next state; the state carried is the transition's log
        density of that draw and whether the transition is stochastic."""
        transition = self.transition(x, t, t_next, denoised, eps, process)
        following = transition.sample(key)
        return following, (transition.log_prob(following), transition.stochastic)

    def trajectory(self, denoise: Denoiser, x_T: jax.Array, steps: int, *,
                   guidance: Guidance | None = None, key: int | jax.Array) -> FlowTrajectory:
        """`sample` with this solver, recording each transition's state, log
        density and whether it is stochastic (`sample(record=...)`).

        `steps` counts grid points, including both endpoints, so a
        ten-transition rollout uses steps=11 and fewer than 2 raises
        `ValueError`. The guidance must keep no state between steps, because
        each transition is later rescored on its own, so APG with momentum
        raises `ValueError`. Rectified flow's clean prediction at t=0 is its
        state, so the last transition already produces the final sample.
        """
        if steps < 2:
            raise ValueError("a trajectory needs at least two time points")
        x_T = jnp.asarray(x_T)
        x_T = x_T.astype(at_least_fp32(x_T.dtype))
        if jax.tree.leaves(Walk.over(denoise, guidance, steps - 1).init(x_T)):
            raise ValueError("a flow trajectory's transitions are rescored one at a time, so its "
                             "guidance carries nothing between steps; APG's momentum does")
        _, (states, log_probs, stochastic) = sample(
            denoise, x_T, steps, solver=self, guidance=guidance, key=key, final_denoise=False,
            record=lambda following, state, denoised: (following, *state))
        states = jnp.concatenate((x_T[:, None], jnp.swapaxes(states, 0, 1)), axis=1)
        return FlowTrajectory(states, denoise.process.times(steps), log_probs.T, stochastic.T)


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

