"""Arithmetic with each operand's significand and binade apart (`float_bits`), so no intermediate leaves
the normal range where the result does not, and a subnormal operand is its value where a flushing CPU reads
zero: a ratio of products (`scaled_ratio`), the correctly rounded division's denominator derivative
(`over_denominator`, `precision.correctly_rounded_divide`'s) and a special division's subnormal stand-in
(`stood_in`)."""
from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from .float_bits import float_bits, lowered_rounded, raised_exactly


def stood_in(value: jax.Array, sign: jax.Array, biased: jax.Array, fraction: jax.Array) -> jax.Array:
    """An fp32 `value`, a subnormal one (its biased field 0, its fraction not) as a one of its sign."""
    one = jax.lax.bitcast_convert_type(sign | np.int32(0x3F800000), jnp.float32)
    return jnp.where((biased == 0) & (fraction != 0), one, value)


def finite_nonzero(value: jax.Array) -> jax.Array:
    """Whether each of an IEEE float's entries is finite and not zero, read from its bits (`float_bits`), so
    a subnormal counts where a flushing CPU's comparisons read it as zero."""
    _, biased, fraction, _, _ = float_bits(value)
    return (biased != (1 << jnp.finfo(value.dtype).nexp) - 1) & ((biased != 0) | (fraction != 0))


def _word(dtype) -> np.dtype:
    """The signed integer of `dtype`'s width, float32's or float64's (`scaled_ratio`'s formats)."""
    if dtype not in (jnp.float32, jnp.float64):
        raise ValueError(f"a ratio with its binades apart is taken in float32 or float64, not {dtype}")
    return np.dtype(f"int{jnp.finfo(dtype).bits}")


@jax.custom_jvp
def scaled_ratio(numerators: tuple[jax.Array, ...], denominators: tuple[jax.Array, ...]) -> jax.Array:
    """`prod(numerators) / prod(denominators)` for finite numerators and finite nonzero denominators of one
    format (float32 or float64), each operand's significand and binade apart (`float_bits`): the
    significands' product over theirs, in `(2^-len(denominators), 2^len(numerators))`, rounded once a
    factor, then scaled by the binades' sum once (float32's from its bits, `raised_exactly` and
    `lowered_rounded`, so a subnormal result rounds once wherever a CPU flushes; float64's by `power2`),
    infinite past the format's largest; a zero numerator a zero of the sign. So `2^-12 * 2^-149 / 2^-280`
    is `2^119` where `2^-149 / 2^-140 / 2^-140` overflows first, and a subnormal operand is its value. Its
    derivative is the ratio's own, each term a ratio of the same kind (`_scaled_ratio_jvp`), at every
    order."""
    from .precision import correctly_rounded_divide, power2

    count = len(numerators)
    dtype = jnp.result_type(*numerators, *denominators)
    word = _word(dtype)
    values = jnp.broadcast_arrays(*(jnp.asarray(value, dtype) for value in (*numerators, *denominators)))
    fields = [float_bits(value) for value in values]
    one = np.asarray(1.0, dtype).view(word)

    def significand(fraction):
        return jax.lax.bitcast_convert_type(fraction | one, dtype)

    top, bottom = (math.prod(significand(field[3]) for field in part)
                   for part in (fields[:count], fields[count:]))
    ratio = correctly_rounded_divide(top, bottom) if dtype == jnp.float32 else top / bottom
    shift = sum(field[4] for field in fields[:count]) - sum(field[4] for field in fields[count:])
    limits = jnp.finfo(dtype)
    binade = float_bits(ratio)[4] + shift
    if dtype == jnp.float32:
        scaled = jnp.where(shift >= 0, raised_exactly(ratio, jnp.maximum(shift, 0)),
                           lowered_rounded(ratio, jnp.maximum(-shift, 0)))
    else:
        least = limits.minexp - limits.nmant - 2
        scaled = power2(ratio, jnp.clip(shift, least - 8, limits.maxexp + 8))
        scaled = jnp.where(binade < least, jnp.zeros((), dtype), scaled)
    magnitude = jnp.where(binade >= limits.maxexp, jnp.asarray(jnp.inf, dtype), scaled)
    zero = functools.reduce(jnp.logical_or, ((field[1] == 0) & (field[2] == 0) for field in fields[:count]))
    bits = jax.lax.bitcast_convert_type(jnp.where(zero, jnp.zeros((), dtype), magnitude), word)
    # The sign set in its bit, so a subnormal result keeps it where a negation would flush it.
    sign = functools.reduce(jnp.bitwise_xor, (field[0] for field in fields))
    return jax.lax.bitcast_convert_type(bits | sign, dtype)


@scaled_ratio.defjvp
def _scaled_ratio_jvp(primals, tangents):
    """`d(P / Q) = sum_i (P / p_i) dp_i / Q - sum_j P dq_j / (Q q_j)`, each term a ratio with its binades
    apart (`scaled_ratio`), so a higher order's is too."""
    (numerators, denominators), (moved_numerators, moved_denominators) = primals, tangents
    terms = [scaled_ratio((*numerators[:index], moved, *numerators[index + 1:]), denominators)
             for index, moved in enumerate(moved_numerators)]
    terms += [-scaled_ratio((*numerators, moved), (*denominators, denominator))
              for moved, denominator in zip(moved_denominators, denominators, strict=True)]
    return scaled_ratio(numerators, denominators), sum(terms)


def over_denominator(tangent: jax.Array, numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """The size of `n / d`'s tangent in its denominator, `t n / d^2`, a linear map of `t` that is its own
    transpose (`generation.linear_map`, so reverse mode evaluates it alike): `t ((n / d) / d)` as PyTorch's
    backward divides, each quotient rounded once, where that is a normal value or an operand is special (a
    zero, an infinity, a NaN: IEEE's result); else `scaled_ratio((t, n), (d, d))`, the binades apart, as
    `(n / d) / d` or its product with `t` leaves the range where the result need not (2^20 / 2^-60 under
    2^-60 is 2^80; 4 / 2^-128 under 2^-149, 2^109; a cotangent 2^-145 of 2^-149 / 2^-140 pulled back to
    2^-14: the 2026-11-26 audit and its review). Its derivative in `n` and `d` is either branch's own."""
    from .kernels.generation import linear_map
    from .precision import correctly_rounded_divide

    def twice(numerator, denominator):
        return correctly_rounded_divide(correctly_rounded_divide(numerator, denominator), denominator)

    def apply(residuals, moving):
        numerator, denominator = residuals
        held = jax.lax.stop_gradient((numerator, denominator, moving))
        ordinary = functools.reduce(jnp.logical_and, map(finite_nonzero, held))
        product = held[2] * twice(*held[:2])
        normal = (jnp.abs(product) >= jnp.finfo(product.dtype).tiny) & (jnp.abs(product) < np.inf)
        torch = ~ordinary | normal
        # Each branch's operands where the other is chosen held at one, so neither's derivative meets the
        # other's infinities.
        kept, other = ((jnp.where(chosen, value, 1) for value in (numerator, denominator))
                       for chosen in (torch, ~torch))
        balanced_numerator, balanced_denominator = other
        return jnp.where(torch, moving * twice(*kept),
                         scaled_ratio((jnp.where(torch, 1, moving), balanced_numerator),
                                      (balanced_denominator, balanced_denominator)))

    return linear_map(apply, apply, (numerator, denominator), tangent)
