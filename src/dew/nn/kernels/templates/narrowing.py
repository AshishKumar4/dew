"""A 16-bit output's single rounding at the attention template's finish (`narrowing_settled`): an fp32
quotient that lands exactly on a 16-bit tie nudged one ulp toward the exact quotient, so its narrowing rounds
once as the exact quotient would (row 9544)."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from dew.nn.derivatives import contracted, surrogate
from dew.nn.float_bits import binade_of, lowered_rounded, raised_exactly
from dew.nn.precision import split_bits, uncontracted


def _narrowing_settled(out: jax.Array, acc: jax.Array, divisor: jax.Array, dtype: DTypeLike) -> jax.Array:
    """`narrowing_settled`'s value."""
    if out.dtype != jnp.float32:
        return out
    above = _remainder_sign(acc, out, divisor)
    div_bits = jax.lax.bitcast_convert_type(divisor, jnp.int32)
    positive = jax.lax.broadcast_in_dim((div_bits > 0) & (div_bits < 0x7F800000), acc.shape,
                                        tuple(range(divisor.ndim)))
    bits = jax.lax.bitcast_convert_type(out, jnp.int32)
    # One fp32 ulp toward the exact quotient: up in magnitude where it lies past the rounded one's side, its
    # side read from its sign bit (a subnormal's comparison with zero reads it as zero where the CPU treats
    # subnormal operands as zero).
    toward = jnp.where((above > 0) == (bits > 0), 1, -1).astype(jnp.int32)
    odd = jax.lax.bitcast_convert_type(bits + toward, jnp.float32)
    nudged = _between(bits, dtype) & (above != 0) & jnp.isfinite(out) & jnp.isfinite(acc) & positive
    return jnp.where(nudged, odd, out)


narrowing_settled = contracted("test_the_narrowing_settle_holds_every_order")(
    surrogate(_narrowing_settled, lambda dtype, primals, tangents, value: tangents[0], nondiff_argnums=(3,)))
"""`out`, the sums `acc` over `divisor` as the finish formed it (fp32; a recovered subnormal's bits,
`attention_algebra.quotient`), rounded to odd where it narrows to the 16-bit `dtype`: where it lands exactly
between two values of `dtype` (`_between`) and is not the exact quotient, one fp32 ulp toward it, so the
narrowing rounds once as the exact quotient would: the exact output 65519.9995 of one kept key of 65504
rounded to fp32's 65520, which tied to inf in fp16 (the base review of 545370f65, F3; row 9544). The side is
the sign of `acc` less `out` times `divisor`, exact (`_remainder_sign`). A finite fp32 `out` over a positive
divisor alone; any other, a held one (its dtype's largest, no tie) and a guarded zero as they are, selected,
no arithmetic on them. Its derivative is `out`'s on every entry, nudged or not (`surrogate`). The bound left
at fp32's resolution, as the fp32 reference's: `divisor` is the one rounded (a dropping call's total times
its multiplier's rest, `dropped_total`), and the sums and exponentials before it round as the reference's
do, so an exact output within their rounding of a boundary may narrow past it."""


def _remainder_sign(acc: jax.Array, quotient: jax.Array, divisor: jax.Array) -> jax.Array:
    """A value of the sign of `acc - quotient * divisor` (fp32, finite) at every 16-bit narrowing tie
    (`_between`), exact with row-hoisted divisor scaling and splitting: `divisor` is scaled once per row into
    `[1/4, 1/2)` and split into two exact 12-bit halves `(d_hi, d_lo)`. Every 16-bit tie `quotient` has at
    most 12 significant bits (9 for bf16, 12 for fp16), so `q_scaled * d_hi` and `q_scaled * d_lo` are exact
    in fp32 with no per-element bit splitting; tiny `|quotient| < 2^-100` is scaled up by `2^52` so every
    intermediate stays normal down to `2^-149`. Flushing paths (`bodies.subnormals`) scale through bit
    arithmetic (`raised_exactly`, `lowered_rounded`) so subnormal accumulators and quotients never flush,
    while non-flushing GPU kernels use two power-of-two multiplies to avoid Triton register spills (row
    11648)."""
    from dew.nn.kernels.templates import bodies

    shift = jnp.clip(binade_of(divisor), -149, 127) + 2
    dims = tuple(range(divisor.ndim))

    def expand(x: jax.Array) -> jax.Array:
        return jax.lax.broadcast_in_dim(x, acc.shape, dims)
    def scale_bits(x: jax.Array, power: jax.Array) -> jax.Array:
        return jnp.where(power >= 0, raised_exactly(x, jnp.maximum(power, 0)),
                         lowered_rounded(x, jnp.maximum(-power, 0)))

    def compute(*, flushing: bool) -> jax.Array:
        if flushing:
            scaled = scale_bits(divisor, -shift)
            bump = jnp.where(binade_of(quotient) < -100, jnp.int32(52), jnp.int32(0))
            q_scaled = raised_exactly(quotient, bump)
            acc_scaled = scale_bits(acc, bump - expand(shift))
        else:
            first = jnp.right_shift(shift + 1, 1)
            second = shift - first
            s1 = jax.lax.bitcast_convert_type(jnp.left_shift(127 - first, 23).astype(jnp.int32), jnp.float32)
            s2 = jax.lax.bitcast_convert_type(jnp.left_shift(127 - second, 23).astype(jnp.int32), jnp.float32)
            scaled = (divisor * s1) * s2
            bump = jnp.where(jnp.abs(quotient) < np.float32(2.0 ** -100),
                             np.float32(2.0 ** 52), np.float32(1.0))
            q_scaled = quotient * bump
            acc_scaled = (acc * (expand(s1) * bump)) * expand(s2)
        (scaled,) = uncontracted(scaled)
        unsigned, mask, _ = split_bits(np.float32)
        d_hi = jax.lax.bitcast_convert_type(
            jax.lax.bitcast_convert_type(scaled, unsigned) & np.array(mask, unsigned), jnp.float32)
        d_lo = scaled - d_hi
        return (acc_scaled - q_scaled * expand(d_hi)) - q_scaled * expand(d_lo)

    return bodies.subnormals(compute, acc.dtype)

def _between(bits: jax.Array, dtype: DTypeLike) -> jax.Array:
    """Whether each fp32 value of `bits` (its int32 bits) lies exactly between two neighbouring values of the
    16-bit `dtype`, the tie its narrowing rounds to even: the significand bits the narrowing drops (16 of
    bf16's, 13 of fp16's normals, more below its least normal, 2^-14) are a one and zeros."""
    found = jnp.finfo(dtype)
    exponent = jnp.right_shift(bits, 23) & 0xFF
    significand = (bits & 0x7FFFFF) | jnp.where(exponent > 0, 0x800000, 0)
    dropped = 23 - (found.nmant)
    if found.minexp > -126:
        # Below the dtype's least normal its narrowing drops a bit more a binade, down to the significand's
        # whole 24 (half the least subnormal, 2^-25 in fp16, its last tie).
        dropped = dropped + jnp.clip(127 + found.minexp - exponent, 0, 24 - dropped)
    half = jnp.left_shift(1, dropped - 1)
    return (significand & (2 * half - 1)) == half
