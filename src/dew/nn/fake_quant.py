"""Fake quantizers of DeepSeek-V4.1's cache (arXiv 2609.19969, section 2.4.4).

Each rounds fp32 values the way the release's tilelang kernels store them
(inference/kernel.py at the revision tools/deepseek_v41_reference.py pins)
and passes the gradient straight through. The rounding is arithmetic on the
fp32 values and bits, never an `astype` round trip through a narrow dtype:
XLA GPU's default xla_allow_excess_precision deletes an f32 -> f8 -> f32
convert pair under jit, and the quantization silently does not happen.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import linen as nn

E2M1 = np.arange(16, dtype=np.uint8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
"""Each E2M1 code's value in code order (ml_dtypes' float4_e2m1fn): 0 to 6
by codes 0-7, then their negatives with the sign in bit 3, -0 included. The
FP4 codecs, these quantizers and the fixture generators all read it."""

E2M1_MIDPOINTS = (E2M1[:7] + E2M1[1:8]) / 2
"""The magnitudes halfway between neighbouring E2M1 values, 0.25 up to 5."""

E2M1_TIES_UP = np.arange(7) % 2 == 1
"""Whether a magnitude exactly on each midpoint rounds up, to even: where the
lower neighbour's mantissa bit, its code's bit 0, is 1 (0.5, 1.5, 3)."""

_E4M3_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
_E2M1_MAX = float(jnp.finfo(jnp.float4_e2m1fn).max)


def _power_of_two_ceil(value):
    """2 ** ceil(log2(value)) off the fp32 bits, as the kernels' fast_round_scale
    computes it (kernel.py:22-37). Unclamped, unlike the host codecs'
    `ceil_to_ue8m0` rule (exponents 1 to 254): only past the normal range do
    they part, a zero giving 0.0 here and 2 ** -126 there, a value above
    2 ** 127 inf here and 2 ** 127 there."""
    bits = jax.lax.bitcast_convert_type(value.astype(jnp.float32), jnp.int32)
    exponent = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).astype(jnp.int32)
    return jax.lax.bitcast_convert_type((exponent + 127) << 23, jnp.float32)


def _round(values, mantissa: int, lowest: int):
    """Round fp32 values to a float format of `mantissa` bits whose smallest
    normal exponent is `lowest`, ties to even, by arithmetic: the quantum is
    the power of two of the value's binade less the mantissa bits, floored
    at the subnormal spacing; dividing by it is exact, and `round` ties to
    the even multiple, the one whose last mantissa bit is 0."""
    bits = jax.lax.bitcast_convert_type(values.astype(jnp.float32), jnp.int32)
    exponent = jnp.maximum(((bits >> 23) & 0xFF) - 127, lowest) - mantissa
    quantum = jax.lax.bitcast_convert_type((exponent + 127) << 23, jnp.float32)
    return jnp.round(values / quantum) * quantum


def _round_e4m3fn(values):
    """Round fp32 values within +-448 to E4M3FN, ties to even (`_round`).
    Not `jax.lax.reduce_precision(x, 4, 3)`, which models IEEE-style e4m3
    with infinities and a largest finite 240, not the FN format whose
    largest finite is 448."""
    return _round(values, 3, -6)


def _e2m1(values):
    """Round fp32 values within +-6 to E2M1, ties to even (`_round`); IEEE-style
    e2m1 would stop at 3 where E2M1 reaches 6, so reduce_precision cannot."""
    return _round(values, 1, 0)


def _e2m1_quotient(values, scale):
    """E2M1 of `values / scale` under an E4M3 scale, ties to even, clamped to
    +-6, without dividing: |values| is compared with each midpoint times the
    scale, a product of at most 7 significant bits and so exact, which
    rounds the exact quotient. For bf16 values that is the rounding of the
    correctly rounded quotient the kernel divides to, since a bf16 value off
    a midpoint times the scale lies further from it than half an fp32 ulp of
    the quotient. A division would not do: XLA divides by a broadcast
    through its reciprocal, and XLA GPU's division is not correctly rounded,
    either of which carries a quotient across a tie (-0.146484375 /
    0.1171875 comes out -1.2500001, rounding to -1.5 where the kernel's
    -1.25 rounds to -1.0)."""
    magnitude = jnp.abs(values)
    index = sum(((magnitude > point * scale) | ((magnitude == point * scale) if up else False))
                .astype(jnp.int32) for point, up in zip(E2M1_MIDPOINTS.tolist(), E2M1_TIES_UP.tolist(),
                                                        strict=True))
    return jnp.copysign(jnp.asarray(E2M1[:8])[index], values)


def straight_through(x, rounded):
    """`rounded` forward, bit for bit in any dtype, and the identity's
    gradient backward. `stop_gradient(x) - x` is +0, and subtracting +0
    leaves every value as it is, a negative zero included, which the
    kernels keep; adding it would turn -0 into +0, and `x + (rounded - x)`
    re-rounds in bf16 wherever the quantizer clamped far from `x`. The
    estimator is Dew's choice; the release is inference code and the paper
    names none."""
    return jax.lax.stop_gradient(rounded.astype(x.dtype)) - (jax.lax.stop_gradient(x) - x)


def fp8_scaled_blocks(x, block: int):
    """`x` in fp32 blocks of `block` channels and each block's power-of-two scale (act_quant with scale_fmt
    ue8m0, kernel.py:40-124): amax floored at 1e-4, the scale the ceiling power of two of amax / 448."""
    blocks = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, block)
    amax = jnp.maximum(jnp.max(jnp.abs(blocks), -1, keepdims=True), 1e-4)
    return blocks, _power_of_two_ceil(amax * jnp.float32(1 / _E4M3_MAX))


def fp8_rounded(x, blocks, scale):
    """`x` as E4M3 of its `blocks` over their `scale` (`fp8_scaled_blocks`), clamped to +-448, times the
    scale."""
    rounded = _round_e4m3fn(jnp.clip(blocks / scale, -_E4M3_MAX, _E4M3_MAX))
    return straight_through(x, (rounded * scale).reshape(x.shape))


def fake_quant_fp8(x, block: int):
    """E4M3 per `block` channels under a power-of-two scale (act_quant with
    scale_fmt ue8m0, kernel.py:40-124): amax floored at 1e-4, scale the
    ceiling power of two of amax / 448, value clamped to +-448 and rounded."""
    return fp8_rounded(x, *fp8_scaled_blocks(x, block))


def fp4_scaled_blocks(x, block: int, e4m3_scale: bool):
    """`x` in fp32 blocks of `block` channels and each block's scale (fp4_act_quant, kernel.py:127-204):
    with `e4m3_scale` the E4M3 of amax / 6 with amax floored at 6 * 2**-9 (the compressed KV), else the
    ceiling power of two of amax / 6 with amax floored at 6 * 2**-126 (the indexer)."""
    blocks = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, block)
    amax = jnp.max(jnp.abs(blocks), -1, keepdims=True)
    if e4m3_scale:
        # the kernel's cast saturates at E4M3's 448 (cvt.rn.satfinite)
        floored = jnp.maximum(amax, _E2M1_MAX * 2 ** -9)
        return blocks, _round_e4m3fn(jnp.minimum(floored / _E2M1_MAX, _E4M3_MAX))
    return blocks, _power_of_two_ceil(jnp.maximum(amax, _E2M1_MAX * 2 ** -126) * jnp.float32(1 / _E2M1_MAX))


def fp4_rounded(x, blocks, scale, e4m3_scale: bool):
    """`x` as E2M1 of its `blocks` over their `scale` (`fp4_scaled_blocks`), clamped to +-6, times the
    scale: an E4M3 scale compared against (`_e2m1_quotient`), a power of two divided by exactly."""
    if e4m3_scale:
        rounded = _e2m1_quotient(blocks, scale)
    else:
        rounded = _e2m1(jnp.clip(blocks / scale, -_E2M1_MAX, _E2M1_MAX))
    return straight_through(x, (rounded * scale).reshape(x.shape))


def fake_quant_fp4(x, block: int, e4m3_scale: bool):
    """E2M1 per `block` channels (fp4_act_quant, kernel.py:127-204): under an
    E4M3 scale amax / 6 with amax floored at 6 * 2**-9 (the compressed KV),
    or under the ceiling power of two of amax / 6 with amax floored at
    6 * 2**-126 (the indexer); the value clamped to +-6 and rounded."""
    return fp4_rounded(x, *fp4_scaled_blocks(x, block, e4m3_scale), e4m3_scale)


class FakeQuantizing(nn.Module):
    """A layer whose activations round through the fake quantizers as its own
    methods, so a caller sees and can stand in for each rounding through
    `nn.intercept_methods`, as `tools/deepseek_v41_numerics.py` does to replay
    a float64 twin's roundings in the given run. Each rounding reads its
    blocks and their scales from a method of their own (`fp8_scaled_blocks`,
    `fp4_scaled_blocks`), so a caller reads what a forward rounded, and how
    near a rounding's boundary it was, through `capture_intermediates`."""

    def fp8_scaled_blocks(self, x, block: int):
        return fp8_scaled_blocks(x, block)

    def fp8(self, x, block: int):
        return fp8_rounded(x, *self.fp8_scaled_blocks(x, block))

    def fp4_scaled_blocks(self, x, block: int, e4m3_scale: bool):
        return fp4_scaled_blocks(x, block, e4m3_scale)

    def fp4(self, x, block: int, e4m3_scale: bool):
        return fp4_rounded(x, *self.fp4_scaled_blocks(x, block, e4m3_scale), e4m3_scale)
