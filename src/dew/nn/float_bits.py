"""A float's bits, read and set: its sign, exponent and significand (`float_bits`, `binade_of`), a power of
two set in its exponent bits (`exact_power2`), and fp32 scaled by a power of two from its bits
(`raised_exactly`, `lowered_rounded`), so a process whose float multiply flushes a subnormal (XLA's CPU)
still reads and scales one. Integer arithmetic on a float's bits alone, which `dew.nn.precision`'s
arithmetic (`power2`, `aligned_sum`) and the kernel templates read."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike


def exact_power2(n: jax.typing.ArrayLike, dtype: DTypeLike = jnp.float32) -> jax.Array:
    """`2^n` in `dtype`, exactly, for an integral `n` within its normal exponents, set in its exponent bits at
    the format's own bias (its encoding of 1.0's), and +0 at `minexp - 1` in a format whose biased exponent 0
    holds its zero (IEEE's binary formats, bfloat16, the float8s): the one owner of an exact power of two. No
    `exp2`, which XLA computes as `exp(n ln 2)` and whose products it fuses into one `exp` of their sum. A
    format whose width no integer holds (FP6) takes float32's power, its sentinel its own zero, cast to it,
    exactly. `dtype` is the one JAX makes of it (a Python `float` float32 where x64 is off)."""
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    limits = jnp.finfo(dtype)
    if not hasattr(jnp, f"int{limits.bits}"):
        n = jnp.asarray(n)
        power = jnp.where(n == limits.minexp - 1, np.float32(0), exact_power2(n, jnp.float32))
        return power.astype(dtype)
    word = jnp.dtype(f"int{limits.bits}")
    bias = int(np.asarray(1.0, dtype).view(f"uint{limits.bits}")) >> limits.nmant
    biased = jnp.asarray(n).astype(word) + word.type(bias)
    bits = jax.lax.shift_left(biased, jnp.full_like(biased, limits.nmant))
    return jax.lax.bitcast_convert_type(bits, dtype)


def two_factor_reach(dtype: DTypeLike) -> int:
    """The largest `reach` `precision.power2` takes as two factors in `dtype`, both halves of any `k` within
    it normal powers."""
    limits = jnp.finfo(dtype)
    return 2 * min(-limits.minexp, limits.maxexp - 1)


TWO_FACTOR_REACH = two_factor_reach(jnp.float32)
"""`two_factor_reach` of float32, 252: a caller whose `k` stays within it states it for two factors."""


def float_bits(x: jax.Array) -> tuple[jax.Array, ...]:
    """A float `x` (fp32 or float64) from its bits, each a signed integer of its width: its sign bit, its
    biased exponent, its fraction, its fraction normalized as a normal one's (a subnormal's leading bit moved
    up to the implicit bit's place and dropped) and its binary exponent `floor(log2 |x|)`, a subnormal's
    below the least normal: `(sign, biased, fraction, normalized, exponent)`. Read from its bits, a
    subnormal is kept where a process's float multiply flushes a subnormal operand (XLA's CPU, row 4023):
    the one reading `binade_of`, `raised_exactly` and `aligned_sum` take."""
    limits = jnp.finfo(x.dtype)
    signed = jnp.dtype(f"int{limits.bits}")
    bias, width = limits.maxexp - 1, limits.nmant
    bits = jax.lax.bitcast_convert_type(x, signed)
    biased = (bits >> width) & (2 * limits.maxexp - 1)
    fraction = bits & ((1 << width) - 1)
    sign = bits & np.array(-(1 << (limits.bits - 1)), signed)
    up = jnp.clip(width - (limits.bits - 1 - jax.lax.clz(fraction)), 0, width)
    normalized = jnp.where(biased > 0, fraction, (fraction << up) & ((1 << width) - 1))
    exponent = jnp.where(biased > 0, biased - bias, limits.minexp - up)
    return sign, biased, fraction, normalized, exponent


def binade_of(x: jax.Array) -> jax.Array:
    """`floor(log2(|x|))` of an fp32 `x` from its bits (`float_bits`), int32, a subnormal's by its
    significand's leading bit (its arithmetic on a process that flushes subnormal operands reads none of
    it), -1024 for zero."""
    _, biased, fraction, _, exponent = float_bits(x.astype(jnp.float32))
    return jnp.where((biased == 0) & (fraction == 0), -1024, exponent).astype(jnp.int32)


def raised_exactly(x: jax.Array, k: jax.Array) -> jax.Array:
    """An fp32 `x` times `2^k` (`k` int32, at least 0, broadcast against `x`), exactly, from its bits: a
    subnormal `x` normalized by its significand's leading bit, so a process whose float32 multiply flushes a
    subnormal operand (XLA's CPU, row 4023) still raises it. The result at most fp32's largest; an infinity
    or NaN as it is."""
    sign, biased, fraction, normalized, exponent = float_bits(x.astype(jnp.float32))
    target = exponent + k
    normal = sign | (jnp.clip(target + 127, 1, 254) << 23) | normalized
    # Still below the least normal: the subnormal's significand shifted up by `k`.
    below = sign | (fraction << jnp.clip(k, 0, 31))
    raised = jnp.where(target >= -126, normal, below)
    kept = (biased == 0xFF) | ((biased == 0) & (fraction == 0))
    return jnp.where(kept, x.astype(jnp.float32), jax.lax.bitcast_convert_type(raised, jnp.float32))


def lowered_rounded(x: jax.Array, k: jax.Array) -> jax.Array:
    """`x` (fp32) times `2^-k` (`k` int32, at least 0, broadcast against `x`) rounded once to nearest even
    into fp32 however far below its least normal, in integer arithmetic on its bits: a process whose
    float32 multiply or conversion flushes a subnormal result (XLA's CPU, per primitive, row 4023) still
    returns it, as an IEEE multiply by the power of two would. A finite `x` alone; an infinity or NaN
    returns as it is."""
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    sign = bits & np.uint32(0x80000000)
    biased = ((bits >> 23) & np.uint32(0xFF)).astype(jnp.int32)
    fraction = bits & np.uint32(0x7FFFFF)
    # A subnormal `x` holds its significand at the least normal's exponent, its leading bit none.
    held = jnp.maximum(biased, 1)
    significand = jnp.where(biased > 0, fraction | np.uint32(0x800000), fraction)
    target = held - k
    normal = sign | (jnp.maximum(target, 1).astype(jnp.uint32) << 23) | fraction
    # Below the least normal: the significand shifted right by the binades past it, rounded to nearest even.
    shift = jnp.clip(1 - target, 0, 31).astype(jnp.uint32)
    kept = significand >> shift
    rest = significand & ((np.uint32(1) << shift) - np.uint32(1))
    half = np.uint32(1) << (jnp.maximum(shift, np.uint32(1)) - np.uint32(1))
    even = (kept & np.uint32(1)) == np.uint32(1)
    up = (shift > 0) & ((rest > half) | ((rest == half) & even))
    subnormal = sign | (kept + up.astype(jnp.uint32))
    lowered = jnp.where((target >= 1) & (biased > 0), normal, subnormal)
    rounded = jax.lax.bitcast_convert_type(lowered, jnp.float32)
    return jnp.where(biased == 0xFF, x.astype(jnp.float32), rounded)
