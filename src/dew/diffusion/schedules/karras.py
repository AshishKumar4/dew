"""Karras schedules: the rho sigma grid and the EDM training draw."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.derivatives import straight_through
from dew.nn.precision import two_product, two_sum

from .common import GeneralizedNoiseScheduler, times


def rho_ramp(low_inv, high_inv, ramp, rho):
    """(high_inv + ramp (low_inv - high_inv))^rho: the rho spacing at `ramp` between sigma extremes
    given as their 1/rho powers, on a number or an array of either library, the one interpolation
    the scheduler and the published Karras grids take. `rho` is raised as a float, so an equal rho
    given as an int (a config read back, `7`) walks the grid its float does: an int exponent lowers to
    repeated multiplication, a float to pow, about an ulp apart (ledger 2061)."""
    return (high_inv + ramp * (low_inv - high_inv)) ** float(rho)


def _times(x, y):
    """The product of two unevaluated sums (high, low) as another, to about twice their precision
    (Dekker's product over `two_product`, its split the bits', which no splitter overflows, and
    `two_sum`)."""
    product, lost = two_product(x[0], y[0], bit_split=True)
    return two_sum(product, lost + (x[0] * y[1] + x[1] * y[0]))


def _power(base, exponent: int, times=_times):
    """An unevaluated sum raised to a positive integral `exponent`, by squaring under `times` (`_times`, or
    `_scaled_times` over a sum with its binary exponent apart)."""
    if exponent == 1:
        return base
    halved = _power(times(base, base), exponent // 2, times)
    return times(halved, base) if exponent % 2 else halved


def _normal(pair, exponent):
    """An unevaluated sum times 2^`exponent` with its high part brought into [0.5, 1), exactly, its binary
    exponent carried apart."""
    _, binade = jnp.frexp(pair[0])
    return (jnp.ldexp(pair[0], -binade), jnp.ldexp(pair[1], -binade)), exponent + binade


def _scaled_times(x, y):
    """`_times` of two normal sums with their exponents (`_normal`), brought back into range as it is
    formed: a power of them (`_power`) passes no format's range, whatever its exponent (a root of 2 a few
    floats over its level's at rho 1024 raises past float32's)."""
    return _normal(_times(x[0], y[0]), x[1] + y[1])


def _constant(value: float, like):
    """A float64 `value` as an unevaluated sum of two numbers of `like`'s dtype."""
    high = np.asarray(value, like.dtype)
    return high, np.asarray(value - float(high), like.dtype)


def _rounded_rho_ramp(low_inv: float, high_inv: float, t, rho: int):
    """`rho_ramp` at `ramp = 1 - t` for an integral `rho`, carried to about twice `t`'s precision and
    rounded once: the plain form rounds its base, which rho then multiplies (up to 8 ulps in float32 at the
    paper's rho; rows 13653, 13657). Its derivative is the plain form's, the grid's own, the compensation
    its value's alone (`straight_through`: the compensated value bit for bit, the plain form's tangent)."""
    plain = rho_ramp(low_inv, high_inv, 1 - t, rho)
    return straight_through(plain, jax.lax.stop_gradient(_compensated_ramp(low_inv, high_inv, t, rho)))


def _compensated_ramp(low_inv: float, high_inv: float, t, rho: int):
    """`_rounded_rho_ramp`'s value."""
    ramp = two_sum(jnp.ones_like(t), -t)
    span = _times(ramp, _constant(low_inv - high_inv, t))
    start = _constant(high_inv, t)
    high, error = two_sum(start[0], span[0])
    sigma = _power(two_sum(high, error + span[1] + start[1]), rho)
    return sigma[0] + sigma[1]


def _rounded_rho_time(low_inv: float, high_inv: float, sigma, rho: int, low: float, high: float):
    """The `t` at which `_rounded_rho_ramp` reaches `sigma`, carried to about twice its precision and
    rounded once: sigma's 1/rho power refined by one Newton step on its rho-th power taken in twice the
    precision, then (sigma^(1/rho) - low_inv) / (high_inv - low_inv). The plain inverse rounds the root and
    1 - ramp, a level read back through it several ulps off the one asked for, unlike `sigmas`' (rows
    13653, 13657). A sigma of zero, whose root the Newton step divides by, takes the plain form's time; the
    derivative is the plain inverse's, the compensation its value's alone (`straight_through`). A level at or
    past the grid's ends, `low` and `high` as configured (their roots' powers round off them: sigma_max 2 at
    rho 2 comes back 2 + 2^-51), reads as that end's time exactly, from the level as given, the solve reading
    a level inside the range instead so no arithmetic overflows (a clamped level, rounded, can lie inside a
    narrow range's ends); a range whose ends' roots round to one (1 to 1 + 2^-52 at rho 4) holds no level
    between them, each level one end's time."""
    if low_inv == high_inv:
        return jax.lax.stop_gradient(jnp.where(sigma >= high, jnp.ones_like(sigma), jnp.zeros_like(sigma)))
    inside = jnp.clip(sigma, low, high)
    plain = (inside ** (1 / rho) - low_inv) / (high_inv - low_inv)
    compensated = _compensated_time(low_inv, high_inv, inside, rho, plain)
    # The ends snapped in the value alone: the derivative stays the clipped inverse's there too (train 1461:
    # snapped outside, a level at sigma_min exactly differentiated to zero).
    ended = jnp.where(sigma <= low, jnp.zeros_like(compensated), compensated)
    ended = jnp.where(sigma >= high, jnp.ones_like(ended), ended)
    return straight_through(plain, jax.lax.stop_gradient(ended))


def _compensated_time(low_inv: float, high_inv: float, sigma, rho: int, plain):
    """`_rounded_rho_time`'s value, `plain` at a sigma of zero."""
    positive = sigma > 0
    safe = jnp.where(positive, sigma, jnp.ones_like(sigma))
    root = safe ** (1 / rho)
    # The Newton step as its relative residual times root / rho, the root's power carried with its binary
    # exponent apart (`_scaled_times`) and sigma brought to its scale exactly: no power or derivative is
    # formed outside the range, where a root rounded up raises past it (2^16 for 0x1.fffffcp127 at rho 8;
    # train 1407's P0) or a scaled one underflows (1 at rho 256).
    _, unscaled = jnp.frexp(root)
    raised, exponent = _power(_normal((root, jnp.zeros_like(root)), jnp.zeros_like(unscaled)), rho,
                              _scaled_times)
    left, dropped = two_sum(jnp.ldexp(safe, -exponent), -raised[0])
    correction = root * ((left + (dropped - raised[1])) / raised[0]) / rho
    # The root, its correction and the ends' roots brought to the top end's binade alike, exactly, before
    # the numerator and span are formed: a grid's ends near the least normal leave the root's difference
    # from its low end subnormal, which a device that flushes them reads as zero (2^-127 from 2^-126 to
    # 4 2^-126 at rho 1; train 1419's P0).
    binade = -math.frexp(high_inv)[1]
    low = _constant(math.ldexp(low_inv, binade), sigma)
    numerator, lost = two_sum(jnp.ldexp(root, binade), -low[0])
    # The correction carried into the numerator itself: up to half a float of the root, it is no small part
    # of a narrow grid's span (14 floats of 15 near the format's largest at rho 8), whose quotient's
    # remainder would round it again (review).
    numerator, moved = two_sum(numerator, jnp.ldexp(correction, binade))
    rest = (lost + moved) - low[1]
    span = _constant(math.ldexp(high_inv - low_inv, binade), sigma)
    quotient = numerator / span[0]
    product, error = two_product(quotient, span[0], bit_split=True)
    accurate = quotient + (((numerator - product) - error) + rest - quotient * span[1]) / span[0]
    return jnp.where(positive, accurate, plain)


class KarrasVENoiseScheduler(GeneralizedNoiseScheduler):
    """Places sigmas along t with the rho spacing of the EDM paper.

    The spacing is Eq. 5 of Karras et al. 2022,
    sigma(t) = (sigma_max^(1/rho) + (1 - t) (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho,
    so a uniform grid in t is the paper's sampling grid in sigma.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0,
                 sigma_data: float = 0.5):
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data)
        self.rho = rho
        self.min_inv_rho = sigma_min ** (1 / rho)
        self.max_inv_rho = sigma_max ** (1 / rho)

    def sigmas(self, t):
        t = jnp.clip(times(t) / self.T, 0.0, 1.0)
        if float(self.rho).is_integer() and self.rho > 0:
            # Its ends its configured sigmas, which their roots' powers round off (0.5 at rho 2 is
            # 0.5 + 2^-53 again), so its inverse reads them back as its ends (`t_of_sigma`), and every level
            # within them, so no level next to an end passes it (1.99 to 2 at rho 2: t one ulp short of its
            # top still rounds to 2 + 2^-51); its derivative the grid's own.
            ramp = _rounded_rho_ramp(self.min_inv_rho, self.max_inv_rho, t, int(self.rho))
            within = jnp.clip(ramp, self.sigma_min, self.sigma_max)
            ends = jnp.where(t <= 0, jnp.full_like(ramp, self.sigma_min),
                             jnp.where(t >= 1, jnp.full_like(ramp, self.sigma_max), within))
            return straight_through(ramp, jax.lax.stop_gradient(ends))
        return rho_ramp(self.min_inv_rho, self.max_inv_rho, 1 - t, self.rho)

    def t_of_sigma(self, sigma):
        if float(self.rho).is_integer() and self.rho > 0:
            t = _rounded_rho_time(self.min_inv_rho, self.max_inv_rho, times(sigma), int(self.rho),
                                  self.sigma_min, self.sigma_max)
            return jnp.clip(t, 0.0, 1.0) * self.T
        inv_rho = times(sigma) ** (1 / self.rho)
        ramp = jnp.clip((inv_rho - self.max_inv_rho) / (self.min_inv_rho - self.max_inv_rho),
                        0.0, 1.0)
        return (1 - ramp) * self.T


class EDMNoiseScheduler(GeneralizedNoiseScheduler):
    """Draws training sigmas from exp(N(P_mean, P_std^2)).

    t is the standard normal draw, and sigma(t) = exp(P_mean + P_std t). The
    defaults are EDM2's (Karras et al. 2024), because EDM1's -1.2/1.2
    concentrated too much mass on low noise levels for larger models. Pass
    P_mean=-1.2 and P_std=1.2 explicitly to reproduce an EDM1 run.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0, sigma_data: float = 0.5,
                 P_mean: float = -0.4, P_std: float = 1.0):
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data)
        self.P_mean = P_mean
        self.P_std = P_std

    def sigmas(self, t):
        return jnp.exp(times(t) / self.T * self.P_std + self.P_mean)

    def t_of_sigma(self, sigma):
        return (jnp.log(times(sigma)) - self.P_mean) / self.P_std * self.T

    def sample_t(self, key, n):
        return jax.random.normal(key, (n,), dtype=jnp.float32)
