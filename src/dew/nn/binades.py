"""Products and quotients kept in range: a value's product with others taken over their significands, each
operand divided by its binade's power of two first, and the binades' sum set back last (`balanced`), so no
intermediate leaves the range where the result does not, each step a product or a correctly rounded division
by an exact power of two or a significand, which JAX differentiates and transposes at every order; the
correctly rounded division's denominator derivative by it (`over_denominator`,
`precision.correctly_rounded_divide`'s); and a special division's subnormal stand-in from its bits
(`stood_in`, `finite_nonzero`)."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .derivatives import contracted, surrogate
from .float_bits import exact_power2, float_bits


def stood_in(value: jax.Array, sign: jax.Array, biased: jax.Array, fraction: jax.Array) -> jax.Array:
    """An fp32 `value`, a subnormal one (its biased field 0, its fraction not) as a one of its sign."""
    one = jax.lax.bitcast_convert_type(sign | np.int32(0x3F800000), jnp.float32)
    return jnp.where((biased == 0) & (fraction != 0), one, value)


def finite_nonzero(value: jax.Array) -> jax.Array:
    """Whether each of an IEEE float's entries is finite and not zero, read from its bits (`float_bits`), so
    a subnormal counts where a flushing CPU's comparisons read it as zero."""
    _, biased, fraction, _, _ = float_bits(value)
    return (biased != (1 << jnp.finfo(value.dtype).nexp) - 1) & ((biased != 0) | (fraction != 0))


def divided_in(numerator: jax.Array, denominator: jax.typing.ArrayLike) -> jax.Array:
    """`numerator / denominator` correctly rounded (`precision.correctly_rounded_divide`) in float32, a
    subnormal operand and result included; IEEE's division in a wider format."""
    from .division import correctly_rounded_divide

    if jnp.result_type(numerator) == jnp.float32:
        return correctly_rounded_divide(numerator, denominator)
    return numerator / denominator


def normal(value: jax.Array) -> jax.Array:
    """Whether each entry is a finite value no smaller than the format's least normal."""
    return jnp.isfinite(value) & (jnp.abs(value) >= jnp.finfo(value.dtype).tiny)


def _binade(value: jax.Array) -> jax.Array:
    """`floor(log2 |value|)` from its bits (`float_bits`, a subnormal's below the least normal), 0 for a zero
    or a special value, held constant."""
    held = jax.lax.stop_gradient(value)
    return jnp.where(finite_nonzero(held), float_bits(held)[4], 0).astype(jnp.int32)


def over_power(value: jax.Array, exponent: jax.Array) -> jax.Array:
    """`value / 2^exponent` for an `exponent` from the least subnormal's binade to the largest's, as two
    correctly rounded divisions by exact normal powers of two, each half the exponent, so a subnormal value
    is raised to its significand and a large one lowered, every step normal and exact."""
    first = exponent // 2
    halves = (exact_power2(first, value.dtype), exact_power2(exponent - first, value.dtype))
    return divided_in(divided_in(value, halves[0]), halves[1])


def _scaled_by(value: jax.Array, exponent: jax.Array) -> jax.Array:
    """`value * 2^exponent` for a `value` within a few binades of one: up by two exact products (infinite
    past the largest), down by a division by an exact power that keeps it normal and a last correctly
    rounded division, which rounds a subnormal result once (zero below the least)."""
    limits = jnp.finfo(value.dtype)
    up = jnp.clip(exponent, 0, 2 * (limits.maxexp - 1))
    raised = value * exact_power2(up // 2, value.dtype) * exact_power2(up - up // 2, value.dtype)
    down = jnp.clip(-exponent, 0, -limits.minexp - 4 + limits.nmant + 8)
    first = jnp.minimum(down, -limits.minexp - 4)
    lowered = divided_in(divided_in(value, exact_power2(first, value.dtype)),
                         exact_power2(down - first, value.dtype))
    return jnp.where(exponent >= 0, raised, lowered)


def balanced(value: jax.Array, multipliers: Sequence[jax.Array],
             divisors: Sequence[jax.Array]) -> jax.Array:
    """`value * prod(multipliers) / prod(divisors)` for finite operands, the divisors nonzero, linear in
    `value`: each operand over its binade's power of two (`over_power`, its significand in [1, 2)), their
    product and quotients taken over the significands, in `[2^-len(divisors), 2^(1 + len(multipliers)))`,
    and the binades' sum set back last (`_scaled_by`), so no intermediate leaves the range where the result
    does not (`2^-12 * 2^-149 / 2^-140 / 2^-140` is `2^119`, where `2^-12 / 2^-140` overflows and
    `2^-12 * 2^-149` flushes). Each step a product or a division by a significand or an exact power, held
    constant: kept in the result's range as a map of `value`, rounded once a step (a product, two quotients
    and the rescale: as many roundings as PyTorch's order, so a result within those ulps of the largest
    finite value may round past it); the other operands held constant, no derivative taken in them (row
    13232). A float64 subnormal is its value only where the backend keeps one."""
    # The other operands held constant: a derivative in them would stage this expression again at every
    # order, and the callers' are taken by PyTorch's order where it is clean (row 13232).
    multipliers, divisors = jax.lax.stop_gradient((tuple(multipliers), tuple(divisors)))
    exponent = _binade(value)
    significands = over_power(value, exponent)
    for multiplier in multipliers:
        binade = _binade(multiplier)
        exponent = exponent + binade
        significands = significands * over_power(multiplier, binade)
    for divisor in divisors:
        binade = _binade(divisor)
        exponent = exponent - binade
        significands = divided_in(significands, over_power(divisor, binade))
    return _scaled_by(significands, exponent)


def _flush_safe(seed: jax.Array, coefficient: jax.Array) -> jax.Array:
    """`seed * coefficient` of held values (no derivative), a subnormal factor (either) raised by `2^64`
    through a correctly rounded division by an exact power (`divided_in`, which reads a subnormal from its
    bits) and the other lowered by as much, so a CPU flushing subnormal operands multiplies normal values:
    the value PyTorch's order computes where subnormals are kept."""
    seed, coefficient = jax.lax.stop_gradient((seed, coefficient))
    small_coefficient = finite_nonzero(coefficient) & ~normal(coefficient)
    small_seed = finite_nonzero(seed) & ~normal(seed) & ~small_coefficient
    raised, lowered = (jnp.where(small_coefficient, coefficient, jnp.where(small_seed, seed, coefficient)),
                       jnp.where(small_coefficient, seed, jnp.where(small_seed, coefficient, seed)))
    shift = jnp.where(small_coefficient | small_seed, 64, 0)
    return (lowered * exact_power2(-shift, lowered.dtype)) * divided_in(raised, exact_power2(-shift,
                                                                                         raised.dtype))


def as_plain(plain: Callable[..., jax.Array]) -> Callable:
    """A surrogate's derivative rule (`derivatives.surrogate`, its `symbolic_zeros`) that differentiates
    `plain` at every order: a range-safe value of `plain`'s arithmetic (a balanced product, a flush-safe one)
    differentiated as that arithmetic, whose derivatives JAX takes where its own intermediates are normal, as
    before the range-safe value replaced it (rows 13260, 13261). An operand no derivative moves adds no term,
    so a constant beside an infinite factor meets no `0 * inf`."""
    from jax.custom_derivatives import SymbolicZero

    def derivative(primals, tangents, value):
        live = tuple(index for index, tangent in enumerate(tangents) if not isinstance(tangent, SymbolicZero))
        if not live:
            return jnp.zeros_like(value)

        def substituted(*moving):
            given = dict(zip(live, moving, strict=True))
            return plain(*(given.get(index, primal) for index, primal in enumerate(primals)))
        return jax.jvp(substituted, tuple(primals[index] for index in live),
                       tuple(tangents[index] for index in live))[1]
    return derivative


def _over_plain(quotient, numerator, denominator, moving):
    """`over_denominator`'s arithmetic as PyTorch's backward writes it, whose derivatives it takes."""
    return moving * divided_in(quotient, denominator)


def _over_exact(quotient, numerator, denominator, moving):
    """`over_denominator`'s value: PyTorch's order wherever its product is a normal value (its quotients
    subnormal or not, each rounded once as PyTorch's are), or the seed is zero beside a finite coefficient,
    valued as where subnormals are kept (`_flush_safe`); the balanced order where that product overflows or
    flushes, a zero seed beside an infinite coefficient among them (zero, not `0 * inf`)."""
    twice = divided_in(quotient, denominator)
    product = _flush_safe(moving, twice)
    special = ~(finite_nonzero(numerator) & finite_nonzero(denominator)) | ~jnp.isfinite(moving)
    torch = special | normal(product) | (~finite_nonzero(moving) & jnp.isfinite(twice))
    kept_numerator, kept_denominator = (jnp.where(torch, 1, value) for value in (numerator, denominator))
    return jnp.where(torch, jnp.where(special, moving * twice, product),
                     balanced(jnp.where(torch, 0, moving), (kept_numerator,),
                              (kept_denominator, kept_denominator)))


_over_valued = contracted("test_the_precision_seams_hold_every_order")(
    surrogate(_over_exact, as_plain(_over_plain), symbolic_zeros=True))
"""`over_denominator`'s map, valued range-safe and differentiated as PyTorch's order at every order
(`as_plain`), held by the division's contract; built once."""


def _over_apply(residuals, moving):
    return _over_valued(*residuals, moving)


def over_denominator(tangent: jax.Array, quotient: jax.Array, numerator: jax.Array,
                     denominator: jax.Array) -> jax.Array:
    """The size of `n / d`'s tangent in its denominator, `t n / d^2`, a linear map of `t` that is its own
    transpose (`generation.linear_map`, so reverse mode evaluates it alike): `t (q / d)` of the division's
    quotient `q` as PyTorch's backward divides, each quotient rounded once, wherever that product is a
    normal value or `t` is zero, or an operand is special (a zero, an infinity, a NaN: IEEE's result);
    else, where PyTorch's product overflows or flushes, `balanced(t, (n,), (d, d))` (2^20 / 2^-60 under
    2^-60 is 2^80; 4 / 2^-128 under 2^-149 is 2^109; 2^-149 / 2^-140 under 2^-12 is 2^119 and pulls 2^-145
    back to 2^-14; 2^-100 / 2^30 under 2^100 is 2^-60; a zero seed is zero beside an infinite `q / d`: the
    2026-11-26 audit and its review). Its contract: every first derivative (the tangent and its pullback)
    finite and exact where PyTorch's is a normal value, as PyTorch's order rounds it, and across fp32's
    range where PyTorch's overflows or flushes; higher orders are JAX's derivatives of PyTorch's order
    everywhere (`as_plain`: `2^-126 / 16` under `2^40`, differentiated in its numerator, is `2^32`, row
    13247; `2^-100 / 2^30` under `2^100`, `2^40`, row 13261), those whose own intermediates leave the range
    row 13232's."""
    from .kernels.generation import linear_map

    return linear_map(_over_apply, _over_apply, (quotient, numerator, denominator), tangent)
