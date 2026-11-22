"""A float's bits, read and set: its raw fields (`raw_fields`), its sign, exponent and significand
(`float_bits`, `binade_of`), a power of two set in its exponent bits (`exact_power2`), fp32 scaled by a power
of two from its bits (`raised_exactly`, `lowered_rounded`) and a float widened exactly (`widened_exactly`),
so a process whose float multiply or conversion flushes a subnormal (XLA's CPU) still reads, scales and
widens one. Integer arithmetic on a float's bits alone, which `dew.nn.precision`'s arithmetic (`power2`,
`aligned_sum`, `float64_sum`) and the kernel templates read."""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jax.typing import DTypeLike


def exact_power2(n: jax.typing.ArrayLike, dtype: DTypeLike = jnp.float32) -> jax.Array:
    """`2^n` in `dtype`, exactly, for an integral `n` within its normal exponents, set in its exponent bits at
    the format's own bias (its encoding of 1.0's), and +0 at `minexp - 1` in a format whose biased exponent 0
    holds its zero (IEEE's binary formats, bfloat16, the float8s); E8M0, whose every biased exponent below its
    NaN's is a power and none a zero, from `minexp` to `maxexp - 1`: the one owner of an exact power of two.
    No `exp2`, which XLA computes as `exp(n ln 2)` and whose products it fuses into one `exp` of their sum. A
    format whose width no integer holds (FP6) takes float32's power, its sentinel its own zero, cast to it,
    exactly. `dtype` is the one JAX makes of it (a Python `float` float32 where x64 is off). Past `maxexp - 1`
    `n` is set in the field as the word's two's complement holds it, which a quantizer's ceiling reads
    (`fake_quant`): IEEE's infinity at `maxexp`, and in float32 a NaN's `maxexp + 1` the sign bit alone,
    `0x80000000`."""
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    limits = jnp.finfo(dtype)
    # Its value, not a narrow type's wrap of a sentinel (an unsigned 2-bit -3 is 1).
    n = jnp.asarray(n).astype(jnp.int32)
    if not hasattr(jnp, f"int{limits.bits}"):
        power = jnp.where(n == limits.minexp - 1, np.float32(0), exact_power2(n, jnp.float32))
        return power.astype(dtype)
    word = jnp.dtype(f"int{limits.bits}")
    bias = int(np.asarray(1.0, dtype).view(f"uint{limits.bits}")) >> limits.nmant
    # Two's complement at the word's width: E8M0's biased exponents past 127 wrap negative, their bits kept.
    biased = n.astype(word) + word.type(bias)
    bits = jax.lax.shift_left(biased, jnp.full_like(biased, limits.nmant))
    return jax.lax.bitcast_convert_type(bits, dtype)


def two_factor_reach(dtype: DTypeLike) -> int:
    """The largest `reach` `precision.power2` takes as two factors in `dtype`, both halves of any `k` within
    it normal powers."""
    limits = jnp.finfo(dtype)
    return 2 * min(-limits.minexp, limits.maxexp - 1)


TWO_FACTOR_REACH = two_factor_reach(jnp.float32)
"""`two_factor_reach` of float32, 252: a caller whose `k` stays within it states it for two factors."""


def raw_fields(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """A float `x`'s bits as the unsigned integer of its width (a format a byte or wider), and its raw fields
    in that type: its sign bit in place (0 in a format with none, E8M0), its biased exponent shifted down
    and its fraction, `(bits, sign, biased, fraction)`. The one reader of a float's fields: a subnormal's
    normalization (`float_bits`) and a result's rounding (`lowered_rounded`, `precision`'s division) are
    their callers'."""
    limits = jnp.finfo(x.dtype)
    word = jnp.dtype(f"uint{limits.bits}")
    bits = jax.lax.bitcast_convert_type(x, word)
    signed = limits.bits - limits.nexp - limits.nmant
    # The exponent's field by its width (`nexp`): a format with no infinity (E4M3) holds finite values in
    # its top binade, past IEEE's `maxexp`.
    return (bits, bits & word.type(signed << (limits.bits - 1)),
            (bits >> word.type(limits.nmant)) & word.type((1 << limits.nexp) - 1),
            bits & word.type((1 << limits.nmant) - 1))


def split_bits(dtype) -> tuple[type, int, int]:
    """A float's significand split for `precision.two_product`'s `bit_split` and `aligned_sum`'s digits: its
    unsigned bits' type, the mask keeping its high part and the low bits masked, `p - floor(p / 2)` of its
    `p`, so its high part holds `floor(p / 2)` bits. An even `p`'s truncated remainder holds the other half
    (fp32's 12, bfloat16's 4); an odd `p`'s would hold one bit more (float64's 27, float16's 6), its low
    parts' product past `p` bits, so the cut rounds to nearest there (`precision`'s `_odd`), its remainder
    signed and at most half the high part's last bit: every product of two parts exact (Dekker's)."""
    limits = jnp.finfo(dtype)
    unsigned = np.dtype(f"uint{limits.bits}").type
    low = (limits.nmant + 2) // 2
    return unsigned, ~((1 << low) - 1) & np.iinfo(unsigned).max, low


def float_bits(x: jax.Array) -> tuple[jax.Array, ...]:
    """A float `x` of IEEE's binary formats (fp16, bfloat16, fp32, float64) from its bits, each a signed
    integer of its width: its sign bit, its biased exponent, its fraction, its fraction normalized as a
    normal one's (a subnormal's leading bit moved up to the implicit bit's place and dropped) and its binary
    exponent `floor(log2 |x|)`, a subnormal's below the least normal: `(sign, biased, fraction, normalized,
    exponent)`. Read from its bits, a subnormal is kept where a process's float multiply flushes a subnormal
    operand (XLA's CPU, row 4023): the one reading `binade_of`, `raised_exactly`, `widened_exactly` and
    `aligned_sum` take."""
    limits = jnp.finfo(x.dtype)
    signed = jnp.dtype(f"int{limits.bits}")
    bias, width = limits.maxexp - 1, limits.nmant
    _, sign, biased, fraction = (jax.lax.bitcast_convert_type(field, signed) for field in raw_fields(x))
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


def rounded_float32(sign: jax.Array, significand: jax.Array, exponent: jax.Array, *,
                    infinite: jax.Array | bool = False, nan: jax.Array | bool = False) -> jax.Array:
    """The float32 nearest `significand * 2^exponent` (`significand` uint32, any width under 2^32; `exponent`
    int32), signed by `sign` (uint32, the sign bit in place), rounded once to nearest even at the result's
    last place: 24 bits below its leading one, or the least subnormal's however far below the least normal,
    past the largest finite an infinity; a zero significand a zero of its sign. In integer arithmetic on the
    bits, so a process whose float32 arithmetic flushes a subnormal (XLA's CPU, row 4023) still returns it.
    A value its caller's operands' bits make non-finite is its: an infinity of `sign` where `infinite`, a
    NaN where `nan` (row 11795: classified on the bits, where a float32 multiply flushed a subnormal operand
    first). The one rounding of an exact value into float32 (`lowered_rounded`'s,
    `dew.quantization.packed_values`'s, row 11807)."""
    length = 32 - jax.lax.clz(significand).astype(jnp.int32)
    last = jnp.maximum(exponent + length - 24, -149)
    drop = last - exponent
    shift = jnp.clip(drop, 1, 31).astype(jnp.uint32)
    kept = jnp.where(drop <= 0, significand << jnp.clip(-drop, 0, 31).astype(jnp.uint32),
                     jnp.where(drop <= 31, significand >> shift, np.uint32(0)))
    rest = significand & ((np.uint32(1) << shift) - np.uint32(1))
    half = np.uint32(1) << (shift - np.uint32(1))
    odd = (kept & np.uint32(1)) == np.uint32(1)
    up = (((drop >= 1) & (drop <= 31) & ((rest > half) | ((rest == half) & odd)))
          | ((drop == 32) & (significand > np.uint32(1 << 31))))
    kept = kept + up.astype(jnp.uint32)
    # `kept` at its last place: a normal's 24 bits (its leading one carried into the exponent where rounding
    # filled it, the largest binade's carry the infinity), a subnormal's fewer at the least normal's
    # exponent, whose biased field the sum's carry sets; a last place past the largest binade's, infinity.
    biased = last + 150
    magnitude = jnp.where(biased >= 255, np.uint32(0x7F800000),
                          (jnp.clip(biased, 1, 254).astype(jnp.uint32) << np.uint32(23)) + kept
                          - np.uint32(1 << 23))
    bits = sign | jnp.where(significand == 0, np.uint32(0), magnitude)
    bits = jnp.where(nan, np.uint32(0x7FC00000), jnp.where(infinite, sign | np.uint32(0x7F800000), bits))
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def lowered_rounded(x: jax.Array, k: jax.Array) -> jax.Array:
    """`x` (fp32) times `2^-k` (`k` int32, at least 0, broadcast against `x`) rounded once to nearest even
    into fp32 however far below its least normal (`rounded_float32`): a process whose float32 multiply or
    conversion flushes a subnormal result (XLA's CPU, per primitive, row 4023) still returns it, as an IEEE
    multiply by the power of two would; an infinity returns as it is, a NaN as a NaN."""
    _, sign, biased, fraction = raw_fields(x.astype(jnp.float32))
    biased = biased.astype(jnp.int32)
    # A subnormal `x` holds its significand at the least normal's exponent, its leading bit none.
    significand = jnp.where(biased > 0, fraction | np.uint32(0x800000), fraction)
    top = biased == 0xFF
    return rounded_float32(sign, significand, jnp.maximum(biased, 1) - 150 - k,
                           infinite=top & (fraction == 0), nan=top & (fraction != 0))


WIDENED = frozenset(map(jnp.dtype, (jnp.float16, jnp.bfloat16, jnp.float32)))
"""The formats `widened_exactly` widens from their bits: IEEE's binary ones a hardware conversion reads as a
float, flushing a subnormal where its process does (XLA:CPU's float32 to float64). An 8-bit or narrower
float converts by integer arithmetic, a fraction's bits moved, exactly as it is."""


def widened_exactly(x: jax.typing.ArrayLike, dtype: DTypeLike) -> jax.Array:
    """`x` in `dtype`, exactly.

    It widens an array of a `WIDENED` format (fp16, bfloat16, float32) into float32 or float64 wider than
    it. A subnormal `x` is set in `dtype`'s bits from `float_bits`' reading of it, so a process whose
    conversion flushes subnormals (XLA's CPU, rows 4023 and 8966) still widens it. A normal `x` is
    converted. Any other `x` (a host scalar, an integer, an 8-bit or sub-byte float, a narrowing) is
    converted as `jnp.asarray` converts it.

    The widening is a linear map (`generation.linear_map`). Its transpose casts the cotangent back to `x`'s
    type (`cast_back`), so a product's linear map that widens its operands transposes it however JAX
    reaches it. Its result is strongly typed, as an explicit conversion's is."""
    if not isinstance(x, jax.Array | np.ndarray) or not _widens(jnp.dtype(x.dtype), jnp.dtype(dtype)):
        return jnp.asarray(x, dtype)
    # The destination as JAX resolves an explicit request (x64, or its explicit dtypes allowed without it).
    resolved = np.dtype(jax.eval_shape(lambda: jnp.zeros((), dtype)).dtype)
    if not _widens(jnp.dtype(x.dtype), resolved):
        return jnp.asarray(x, resolved)
    from dew.nn.kernels.generation import linear_map

    x = jnp.asarray(x)
    source = jax.typeof(x)

    def apply(_, moving):
        # Traced where JAX evaluates the map, which may lie outside its caller's x64 scope (`float64_sum`'s).
        with jax.enable_x64(new_val=True) if resolved == np.dtype(np.float64) else contextlib.nullcontext():
            return _widened_bits(moving, resolved)

    # Its type strong, as an explicit conversion's: a weak one would yield to a strong float32 it meets.
    return linear_map(apply, lambda _, cotangent: cast_back(cotangent, source), (), x,
                      out_aval=source.update(dtype=resolved, weak_type=False), summed_first=True)


def cast_back(cotangent: jax.Array, source: ShapedArray) -> jax.Array:
    """The cotangent of a cast from `source` (an aval) to `cotangent`'s type, as JAX transposes the cast: in
    `source`'s type, its weakness kept, traced in x64 where the cotangent is float64, whatever x64 scope a
    pullback runs in."""
    wide = cotangent.dtype == np.dtype(np.float64)
    with jax.enable_x64(new_val=True) if wide else contextlib.nullcontext():
        # The source's own aval: its placement, manual axes and weakness, which the transpose restores.
        (back,) = jax.linear_transpose(lambda moving: moving.astype(cotangent.dtype), source)(cotangent)
    return back


def _widens(source: np.dtype, target: np.dtype) -> bool:
    """Whether `widened_exactly` sets `source`'s values in `target`'s bits: a `WIDENED` format into float32
    or float64 wider than it."""
    return (source in WIDENED and target in (np.dtype(np.float32), np.dtype(np.float64))
            and np.finfo(target).bits > jnp.finfo(source).bits)


def _widened_bits(x: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """`widened_exactly`'s map: a subnormal `x` placed in `dtype` from `float_bits`' reading, a normal one
    converted."""
    source, target = jnp.finfo(x.dtype), jnp.finfo(dtype)
    word = jnp.dtype(f"uint{target.bits}")
    sign, biased, fraction, normalized, exponent = float_bits(x)
    normal = (((exponent + (1 << (target.nexp - 1)) - 1).astype(word) << word.type(target.nmant))
              | (normalized.astype(word) << word.type(target.nmant - source.nmant)))
    least = int(source.minexp) - int(source.nmant)
    tiny = fraction.astype(word) << word.type(least - (int(target.minexp) - int(target.nmant)))
    held = jnp.where(fraction == 0, word.type(0), jnp.where(exponent >= target.minexp, normal, tiny))
    signed = (sign != 0).astype(word) << word.type(target.bits - 1)
    subnormal = jax.lax.bitcast_convert_type(signed | held, dtype)
    return jnp.where(biased == 0, subnormal, x.astype(dtype))
