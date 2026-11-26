"""Carry the lower bands of a scaled sum through a product of independent sums.

Split each double-float component into integer digits on a common dyadic grid. Each bin is
summed exactly and kept separate; no float rounding of the large bin can erase a small one
before the product's other terms cancel it.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from dew.algebra import scaled
from dew.nn.float_bits import raw_fields
from dew.nn.precision import two_sum


def placed_totals(values, exponent, dtype, *, may_round: bool):
    """Convert signed integer bin totals to double floats, even past one word's exact precision."""
    if not may_round:
        held = scaled.scaled(values.astype(dtype))
        return held.significand, jnp.zeros_like(held.significand), exponent + held.exponent

    # Both halves fit in float32. Round their sum with its exact residual before the multiplier sees it.
    high = jnp.ldexp((values >> 16).astype(dtype), 16)
    low = (values & 65535).astype(dtype)
    rounded, residual = two_sum(high, low)
    held = scaled.scaled(rounded)
    return held.significand, jnp.ldexp(residual, -held.exponent), exponent + held.exponent


def summed(load, chunks: int, *, axis: int, width: int, lower: int, bins: int, dtype, may_round: bool):
    """Sum scaled double-float terms from ``load(chunk)`` into exact dyadic bins along ``axis``.

    Width leaves enough headroom for all the digits of a bin in a signed word and in the
    result dtype. A bin becomes a significand and the power of two of its grid position.
    """
    mantissa = scaled.format_of(dtype).mantissa
    digit_mask = (1 << width) - 1
    parts = (mantissa + 2 * width - 1) // width

    shape = jnp.broadcast_shapes(*(part.shape for term in load(0) for part in term))
    rows = tuple(size for at, size in enumerate(shape) if at != axis)
    initial = jnp.zeros((*rows, bins), jnp.int32)
    steps = jnp.arange(parts, dtype=jnp.int32)

    def of_chunk(index, carry):
        # One digit extractor and scatter for the chunk's terms, rather than an unrolled graph per
        # significand word and digit, which compounds when higher derivatives carry several band sums.
        terms = load(index)
        words = jnp.stack([jnp.broadcast_to(value, shape) for high, low, _ in terms for value in (high, low)],
                          axis=-1)
        exponents = jnp.stack([jnp.broadcast_to(exponent, shape) for _, _, exponent in terms], axis=-1)
        exponents = jnp.repeat(exponents, 2, axis=-1)
        held = scaled.scaled(words)
        _, sign, biased, fraction = raw_fields(held.significand)
        word = (fraction | ((biased != 0).astype(fraction.dtype) << mantissa))[..., None]
        first, offset = jnp.divmod(held.exponent + exponents - mantissa - lower, width)
        shift = steps * width - offset[..., None]
        digit = jnp.where(shift >= 0, word >> jnp.maximum(shift, 0).astype(word.dtype),
                          word << jnp.maximum(-shift, 0).astype(word.dtype)) & digit_mask
        signed = jnp.where(sign[..., None] != 0, -digit.astype(jnp.int32), digit.astype(jnp.int32))
        places = first[..., None] + steps
        flat_rows = math.prod(rows)
        loaded = shape[axis] * 2 * len(terms) * parts
        slots, values = (jnp.moveaxis(part, axis, -3).reshape(flat_rows, loaded) for part in (places, signed))
        scatter = jax.vmap(lambda row, at, value: row.at[at].add(value, mode="drop",
                                                                wrap_negative_indices=False))
        placed = scatter(carry.reshape(flat_rows, bins), slots, values)
        return placed.reshape(*rows, bins)
    sums = of_chunk(0, initial) if chunks == 1 else jax.lax.fori_loop(0, chunks, of_chunk, initial)
    live = sums != 0
    positions = jnp.cumsum(live, axis=-1, dtype=jnp.int32)
    count = positions[..., -1]
    indices = jnp.where(live, positions - 1, bins)
    powers = jnp.arange(bins, dtype=jnp.int32) * width + lower

    def compressed(row, index):
        values = jnp.zeros_like(row).at[index].set(row, mode="drop")
        exponents = jnp.zeros_like(powers).at[index].set(powers, mode="drop")
        return values, exponents

    values, exponent = jax.vmap(compressed)(sums.reshape(-1, bins), indices.reshape(-1, bins))
    totals = placed_totals(values.reshape(sums.shape), exponent.reshape(sums.shape), dtype,
                           may_round=may_round)
    return totals, count
