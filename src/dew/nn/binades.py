"""fp32 arithmetic with each operand's significand and binade apart (`float_bits`), so no intermediate
leaves the normal range where the result does not, and a subnormal operand is its value where a flushing CPU
reads zero: a product over a quotient (`scaled_quotient`), the correctly rounded division's denominator
derivative (`over_denominator`, `precision.correctly_rounded_divide`'s) and a special division's subnormal
stand-in (`stood_in`)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .float_bits import binade_of, float_bits, lowered_rounded, raised_exactly


def stood_in(value: jax.Array, sign: jax.Array, biased: jax.Array, fraction: jax.Array) -> jax.Array:
    """An fp32 `value`, a subnormal one (its biased field 0, its fraction not) as a one of its sign."""
    one = jax.lax.bitcast_convert_type(sign | np.int32(0x3F800000), jnp.float32)
    return jnp.where((biased == 0) & (fraction != 0), one, value)


def scaled_quotient(x: jax.typing.ArrayLike, y: jax.typing.ArrayLike, z: jax.typing.ArrayLike,
                    k: jax.typing.ArrayLike = 0) -> jax.Array:
    """`x * y / z * 2^k` in fp32 for finite `x`, `y` and a finite nonzero `z` (`k` an integer), each
    operand's significand and binade apart (`float_bits`): the significands' product over the third's, in
    (1/2, 4), rounded twice (`correctly_rounded_divide`), then scaled by the binades' sum once
    (`raised_exactly`, `lowered_rounded`), infinite past fp32's largest; a zero `x` or `y` a zero of the
    sign. No intermediate leaves the normal range where the result does not, which a product or quotient
    of the values may (`2^-12 * 2^-9 / 2^-140`'s `2^-12 / 2^-140` overflows), and a subnormal operand is its
    value, where a flushing CPU reads zero. Bit arithmetic: a caller linear in an operand states the map it
    makes (`generation.linear_map`), its derivative not taken here."""
    x, y, z = jnp.broadcast_arrays(*(jax.lax.stop_gradient(jnp.asarray(value, jnp.float32))
                                     for value in (x, y, z)))
    (x_sign, x_biased, x_raw, x_fraction, x_exponent), (y_sign, y_biased, y_raw, y_fraction, y_exponent), (
        z_sign, _, _, z_fraction, z_exponent) = (float_bits(value) for value in (x, y, z))

    def significand(fraction):
        return jax.lax.bitcast_convert_type(fraction | np.int32(0x3F800000), jnp.float32)

    from .precision import correctly_rounded_divide

    quotient = correctly_rounded_divide(significand(x_fraction) * significand(y_fraction),
                                        significand(z_fraction))
    shift = x_exponent + y_exponent - z_exponent + jnp.asarray(k, jnp.int32)
    raised = raised_exactly(quotient, jnp.maximum(shift, 0))
    magnitude = jnp.where(binade_of(quotient) + shift > 127, jnp.float32(jnp.inf),
                          jnp.where(shift >= 0, raised, lowered_rounded(quotient, jnp.maximum(-shift, 0))))
    zero = ((x_biased == 0) & (x_raw == 0)) | ((y_biased == 0) & (y_raw == 0))
    bits = jax.lax.bitcast_convert_type(jnp.where(zero, jnp.float32(0), magnitude), jnp.int32)
    # The sign set in its bit, so a subnormal result keeps it where a negation would flush it.
    return jax.lax.bitcast_convert_type(bits | (x_sign ^ y_sign ^ z_sign), jnp.float32)


def over_denominator(tangent, quotient, denominator):
    """The size of the division's tangent in its denominator, `t n / d^2`, a linear map of `t` that is its
    own transpose (`generation.linear_map`, so reverse mode evaluates it alike): `t ((n / d) / d)` as
    PyTorch's backward divides, each quotient rounded once, where `(n / d) / d` is finite; else
    `scaled_quotient(t, n / d, d)`, the binades apart, as `(n / d) / d` overflows where the product need not
    (2^20 / 2^-60 under a tangent of 2^-60 is 2^80; 2^-149 / 2^-140 under 2^-12, 2^119, and a cotangent
    2^-145 pulled back to 2^-14: the 2026-11-26 audit and its review)."""
    from .kernels.generation import linear_map
    from .precision import correctly_rounded_divide

    def apply(residuals, moving):
        quotient, denominator, twice = residuals
        finite = jnp.isfinite(twice)
        return jnp.where(finite, moving * jnp.where(finite, twice, 0),
                         scaled_quotient(moving, quotient, denominator))

    twice = correctly_rounded_divide(quotient, denominator)
    return linear_map(apply, apply, (quotient, denominator, twice), tangent)
