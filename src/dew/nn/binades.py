"""Products and quotients kept in range: a value's product taken in the first of several orders whose every
intermediate is finite and normal (`first_clean`), so no intermediate leaves the range where the result does
not, each order plain arithmetic JAX differentiates and transposes at every order; the correctly rounded
division's denominator derivative by it (`over_denominator`, `precision.correctly_rounded_divide`'s); and a
special division's subnormal stand-in from its bits (`stood_in`, `finite_nonzero`)."""
from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .float_bits import float_bits


def stood_in(value: jax.Array, sign: jax.Array, biased: jax.Array, fraction: jax.Array) -> jax.Array:
    """An fp32 `value`, a subnormal one (its biased field 0, its fraction not) as a one of its sign."""
    one = jax.lax.bitcast_convert_type(sign | np.int32(0x3F800000), jnp.float32)
    return jnp.where((biased == 0) & (fraction != 0), one, value)


def finite_nonzero(value: jax.Array) -> jax.Array:
    """Whether each of an IEEE float's entries is finite and not zero, read from its bits (`float_bits`), so
    a subnormal counts where a flushing CPU's comparisons read it as zero."""
    _, biased, fraction, _, _ = float_bits(value)
    return (biased != (1 << jnp.finfo(value.dtype).nexp) - 1) & ((biased != 0) | (fraction != 0))


def divided_in(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """`numerator / denominator` correctly rounded (`precision.correctly_rounded_divide`) in float32, a
    subnormal result and operand included; IEEE's division in a wider format."""
    from .precision import correctly_rounded_divide

    if jnp.result_type(numerator, denominator) == jnp.float32:
        return correctly_rounded_divide(numerator, denominator)
    return numerator / denominator


def steps(value: jax.Array, operations: Sequence[tuple[bool, jax.Array]]) -> list[jax.Array]:
    """`value` through `operations` in order, each a division (`divided_in`) or a product by its operand,
    and every intermediate, the last the result."""
    held = []
    for divides, operand in operations:
        value = divided_in(value, operand) if divides else value * operand
        held.append(value)
    return held


def orders(operations: Sequence[tuple[bool, int]]) -> list[Callable]:
    """Each distinct order of `operations` (a division or a product, by the index of its operand), as a
    candidate of `first_clean`: its intermediates over a value and the operands."""
    distinct = dict.fromkeys(itertools.permutations(operations))

    def candidate(order):
        return lambda value, *operands: steps(value, [(divides, operands[index]) for divides, index in order])
    return [candidate(order) for order in distinct]


def first_clean(value: jax.Array, operands: tuple[jax.Array, ...],
                candidates: Sequence[Callable]) -> jax.Array:
    """`value` through the first of `candidates` (each the intermediates it forms over `value` and
    `operands`, its last the result) whose every intermediate is finite and, but the last, normal, or zero
    where `value` is zero, so no intermediate overflows, nor flushes a product to zero, where the result
    need not; the first candidate where none is. Linear in `value` where each candidate is, the choice read
    from its value (a caller transposing it states the map: `generation.linear_map`). Each candidate's
    operands are held at one where another is chosen, so no candidate's derivative meets another's
    infinities, and each is plain arithmetic, its derivatives JAX's at every order."""
    held = jax.lax.stop_gradient((value, *operands))
    tiny = jnp.finfo(jnp.result_type(value)).tiny
    nonzero = finite_nonzero(held[0])
    clean = []
    for candidate in candidates:
        formed = candidate(*held)
        kept = [jnp.isfinite(step) & ((jnp.abs(step) >= tiny) | ~nonzero) for step in formed[:-1]]
        clean.append(functools.reduce(jnp.logical_and, kept, jnp.isfinite(formed[-1])))
    index = jnp.select(clean, list(range(len(candidates))), 0)
    result = value
    for position, candidate in enumerate(candidates):
        chosen = index == position
        formed = candidate(value, *(jnp.where(chosen, operand, 1) for operand in operands))[-1]
        result = jnp.where(chosen, formed, result)
    return result


def over_denominator(tangent: jax.Array, numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """The size of `n / d`'s tangent in its denominator, `t n / d^2`, a linear map of `t` that is its own
    transpose (`generation.linear_map`, so reverse mode evaluates it alike): `t ((n / d) / d)` as PyTorch's
    backward divides, each quotient rounded once, where that is clean (`first_clean`); else `t` times `n`
    and over `d` twice in the first clean order (2^20 / 2^-60 under 2^-60 is 2^80, `(t / d) n`; 4 / 2^-128
    under 2^-149 is 2^109; a zero seed is zero beside an infinite `(n / d) / d`: the 2026-11-26 audit and
    its review); PyTorch's where none is (a special operand: IEEE's result). Each order is plain arithmetic,
    so its derivatives in `n` and `d` are JAX's at every order; a derivative whose terms overflow apart
    where their sum need not is outside it."""
    from .kernels.generation import linear_map

    candidates = [lambda value, numerator, denominator: [
        value * divided_in(divided_in(numerator, denominator), denominator)],
        *orders(((False, 0), (True, 1), (True, 1)))]

    def apply(residuals, moving):
        return first_clean(moving, residuals, candidates)

    return linear_map(apply, apply, (numerator, denominator), tangent)
