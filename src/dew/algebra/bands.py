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

    def digits(value, exponent):
        held = scaled.scaled(value)
        _, sign, biased, fraction = raw_fields(held.significand)
        word = fraction | ((biased != 0).astype(fraction.dtype) << mantissa)
        placed = held.exponent + exponent - mantissa
        first, offset = jnp.divmod(placed - lower, width)
        columns = []
        for part in range(parts):
            shift = part * width - offset
            digit = jnp.where(shift >= 0, word >> jnp.maximum(shift, 0).astype(word.dtype),
                              word << jnp.maximum(-shift, 0).astype(word.dtype)) & digit_mask
            columns.append((first + part, jnp.where(sign != 0, -digit.astype(jnp.int32),
                                                    digit.astype(jnp.int32))))
        return columns

    def of_chunk(index, carry):
        for high, low, exponent in load(index):
            for value in (high, low):
                for places, parts in digits(value, exponent):
                    slots, values = (jnp.moveaxis(part, axis, -1) for part in (places, parts))
                    shape = jnp.broadcast_shapes(carry.shape[:-1], slots.shape[:-1])
                    slots = jnp.broadcast_to(slots, (*shape, slots.shape[-1]))
                    values = jnp.broadcast_to(values, (*shape, values.shape[-1]))
                    slots = slots.reshape(math.prod(shape), slots.shape[-1])
                    values = values.reshape(math.prod(shape), values.shape[-1])
                    taken = jnp.broadcast_to(carry, (*shape, bins)).reshape(math.prod(shape), bins)
                    carry = jax.vmap(lambda row, at, digit: row.at[at].add(digit, mode="drop",
                                                                            wrap_negative_indices=False))(
                        taken, slots, values).reshape(*shape, bins)
        return carry

    shape = jnp.broadcast_shapes(*(part.shape for term in load(0) for part in term))
    rows = tuple(size for at, size in enumerate(shape) if at != axis)
    initial = jnp.zeros((*rows, bins), jnp.int32)
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
