"""Products and quotients kept in range: a value's product with others taken over their significands, each
operand divided by its binade's power of two first, and the binades' sum set back last (`balanced`), so no
intermediate leaves the range where the result does not, each step a product or a correctly rounded division
by an exact power of two or a significand, which JAX differentiates and transposes at every order; the
correctly rounded division's denominator derivative by it (`over_denominator`,
`precision.correctly_rounded_divide`'s); and a special division's subnormal stand-in from its bits
(`stood_in`, `finite_nonzero`)."""
from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

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
    from .precision import correctly_rounded_divide

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


def _pytorch_product(seed: jax.Array, coefficient: jax.Array, special: jax.Array) -> jax.Array:
    """`seed * coefficient` valued as where subnormals are kept (`_flush_safe`; a special lane the plain
    product, IEEE's), its sign and zeros its own, and differentiated as the plain product is
    (`derivatives.surrogate`: `dseed * coefficient + seed * dcoefficient`), so no rescale enters a
    derivative at any order."""
    from .derivatives import surrogate

    def forward(seed, coefficient, special):
        return jnp.where(special, seed * coefficient, _flush_safe(seed, coefficient))

    def derivative(primals, tangents, value):
        # The lane mask, boolean, takes no tangent; a factor no derivative moves adds no term, as JAX's
        # product rule omits it (its constant beside an infinite factor, no `0 * inf`).
        from jax.custom_derivatives import SymbolicZero

        (seed, coefficient, _), (moved_seed, moved_coefficient, _) = primals, tangents
        terms = [moved * held for moved, held in ((moved_seed, coefficient), (seed, moved_coefficient))
                 if not isinstance(moved, SymbolicZero)]
        return sum(terms[1:], terms[0]) if terms else jnp.zeros_like(value)

    return surrogate(forward, derivative, symbolic_zeros=True)(seed, coefficient, special)


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
    range where PyTorch's overflows or flushes; higher orders are JAX's derivatives of PyTorch's order where
    it is taken (`2^-126 / 16` under `2^40`, differentiated in its numerator, is `2^32`: row 13247), and
    none where the balanced one is (`balanced` holds its operands constant), row 13232's."""
    from .kernels.generation import linear_map

    def apply(residuals, moving):
        quotient, numerator, denominator = residuals
        held_quotient, held_numerator, held_denominator, held_moving = jax.lax.stop_gradient(
            (quotient, numerator, denominator, moving))
        twice = divided_in(held_quotient, held_denominator)
        product = _flush_safe(held_moving, twice)
        special = ~(finite_nonzero(held_numerator) & finite_nonzero(held_denominator)) | ~jnp.isfinite(
            held_moving)
        # PyTorch's order wherever its product is a normal value (its quotients subnormal or not, each
        # rounded once as PyTorch's are, and its derivatives JAX's of that order at every order: row 13247),
        # or the seed is zero beside a finite coefficient; the balanced one where that product overflows
        # or flushes, a zero seed beside an infinite coefficient among them (zero, not `0 * inf`).
        torch = special | normal(product) | (~finite_nonzero(held_moving) & jnp.isfinite(twice))
        # Each branch's operands where the other is chosen held at one, so neither's derivative meets the
        # other's infinities.
        kept_quotient, kept_denominator = (jnp.where(torch, value, 1) for value in (quotient, denominator))
        balanced_numerator, balanced_denominator = (jnp.where(torch, 1, value)
                                                    for value in (numerator, denominator))
        # PyTorch's order as its value is where subnormals are kept, its derivatives that order's own
        # (`_pytorch_product`, as before row 13163): no subnormal flushed into the value, no rescale into a
        # derivative.
        kept = _pytorch_product(moving, divided_in(kept_quotient, kept_denominator), special)
        return jnp.where(torch, kept, balanced(jnp.where(torch, 0, moving), (balanced_numerator,),
                                               (balanced_denominator, balanced_denominator)))

    return linear_map(apply, apply, (quotient, numerator, denominator), tangent)
