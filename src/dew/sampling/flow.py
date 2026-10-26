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
from dew.nn.precision import at_least_fp32

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
        error = jnp.square(value - mean).sum(axis=tuple(range(1, value.ndim)))
        dimensions = math.prod(value.shape[1:])
        # The form of JAX's isotropic `multivariate_normal.logpdf` (a scalar covariance): log(2 pi) and
        # the log variance summed apart, so a variance near float32's largest does not overflow 2 pi
        # times it. Each row has its own variance, which that function would take one row at a time.
        # Halved after the sum, the density itself: a draw's error over its own variance is its
        # chi-squared count, which no transition's draw takes near float32's largest.
        log_prob = -0.5 * (error / safe_variance
                           + dimensions * (jnp.log(2 * math.pi) + jnp.log(safe_variance)))
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


def _rows(x: jax.Array, like: jax.Array) -> jax.Array:
    """A per-row `x` `[B]` shaped to broadcast against `like` `[B, ...]`."""
    return jnp.reshape(x, x.shape + (1,) * (like.ndim - x.ndim))


def _subtracted(left: jax.Array, right: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`left - right` per row as a value and a power, `(left - right) = value * 2**power`: both over
    their rows' largest power of two before the subtraction, so a difference past float32's largest
    (`2**127 - -2**127`) or below its least (two neighbouring tiny means) is formed in range."""
    axes = tuple(range(1, left.ndim))
    power = jnp.maximum(jnp.max(_exponent(left), axis=axes), jnp.max(_exponent(right), axis=axes))
    return _scaled(left, -_rows(power, left)) - _scaled(right, -_rows(power, right)), power


def _contracted(left: jax.Array, right: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`sum(left * right)` over each row as a value and a power: each product formed from the two
    significands (`_split`) with its own exponent, the row's terms aligned to its largest before they
    sum, so a small coordinate's product does not flush beside a large one's and cancelling terms
    cancel before anything is restored."""
    axes = tuple(range(1, left.ndim))
    (left_part, left_power), (right_part, right_power) = _split(left), _split(right)
    products, powers = left_part * right_part, left_power + right_power
    top = jnp.max(jnp.where(products != 0, powers, FLOOR), axis=axes)
    return _scaled(products, powers - _rows(top, powers)).sum(axis=axes), top


def _combined(left: jax.Array, left_power: jax.Array, right: jax.Array, right_power: jax.Array) -> jax.Array:
    """`left * 2**left_power + right * 2**right_power`, aligned to the larger power before the sum and
    restored once, so two terms past float32's range whose sum is in it (`inf - inf`), or two below
    its least whose sum is not, give their sum."""
    left_power = jnp.where(left != 0, left_power, FLOOR)
    right_power = jnp.where(right != 0, right_power, FLOOR)
    power = jnp.maximum(left_power, right_power)
    return _scaled(_scaled(left, left_power - power) + _scaled(right, right_power - power), power)


def _divergence_parts(mean: jax.Array, reference: jax.Array, variance: jax.Array):
    """The scaled parts of `|mean - reference|**2 / (2 variance)` (`_divergence`): the difference and
    its power (`_subtracted`), the variance's significand and exponent (`_split`), and the KL's
    quotient and power, `KL = quotient * 2**power`, the quotient below `4n`."""
    difference, shift = _subtracted(mean, reference)
    level, spread = _split(variance)
    quotient = jnp.square(difference).sum(axis=tuple(range(1, mean.ndim))) / level
    return difference, shift, level, spread, quotient, 2 * shift - spread - 1


def _divergence(mean: jax.Array, reference: jax.Array, variance: jax.Array) -> jax.Array:
    """`|mean - reference|**2 / (2 variance)`, one row a `variance`, every difference, product and
    quotient formed from operands over powers of two and the power restored once at the end
    (`_scaled`), so no step leaves float32's range where the KL keeps it: an error near float32's least
    over a variance as small (a KL of 0.5 where half the error flushed to zero), a squared difference
    past its largest over a variance as wide. A KL within a rounding of float32's least normal may round
    below it, which XLA:CPU flushes. Its derivatives are `_gaussian_kl`'s."""
    *_, quotient, power = _divergence_parts(mean, reference, variance)
    return _scaled(quotient, power)


@jax.custom_jvp
def _directional(mean: jax.Array, reference: jax.Array, variance: jax.Array) -> jax.Array:
    """`_divergence`, whose forward mode forms `sum(d * (t_mean - t_reference)) / v - KL * t_v / v` as
    the KL is formed: the tangents' difference in range (`_subtracted`), its contraction against the
    difference product by product (`_contracted`), the variance's term from the KL's quotient, and the
    two terms joined before their power is restored (`_combined`). Its tangent reads the tangents'
    exponents, so it is forward mode alone: reverse mode is `_kl_pullback`'s (`linear_tangent`), and a
    derivative of the reverse mode in forward mode is not defined."""
    return _divergence(mean, reference, variance)


@_directional.defjvp
def _directional_jvp(primals, tangents):
    """`_directional`'s tangent, every product and quotient of scaled operands, its power restored once."""
    mean_tangent, reference_tangent, variance_tangent = tangents
    difference, shift, level, spread, quotient, power = _divergence_parts(*primals)
    moving, moving_shift = _subtracted(mean_tangent, reference_tangent)
    along, along_power = _contracted(difference, moving)
    stretch, stretch_power = _split(variance_tangent)
    moved = _combined(along / level, along_power + shift + moving_shift - spread,
                      -(quotient * stretch / level), power + stretch_power - spread)
    return _scaled(quotient, power), moved


def _kl_pullback(saved, operands, cotangent):
    """The KL's reverse mode: each element's `cotangent * (mean - reference) / variance` from the
    cotangent's, that element's and the variance's significands with their exponents added, the
    reference's its negation, and the variance's `-cotangent * KL / variance` from the KL's quotient,
    each power restored once (`_scaled`)."""
    del saved
    difference, shift, level, spread, quotient, power = _divergence_parts(*operands)
    (scale, scale_power), (part, part_power) = _split(cotangent), _split(difference)
    pulled = _scaled(_rows(scale / level, part) * part,
                     _rows(scale_power + shift - spread, part) + part_power)
    return pulled, -pulled, -_scaled(scale * quotient / level, scale_power + power - spread)


def _measured(mean: jax.Array, reference: jax.Array, variance: jax.Array) -> tuple[jax.Array, tuple]:
    """`_divergence`, with nothing its reverse mode keeps (`linear_tangent`'s `forward`)."""
    return _divergence(mean, reference, variance), ()


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

