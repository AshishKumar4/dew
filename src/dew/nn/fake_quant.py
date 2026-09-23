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

E4M3_MAX = 448.0
E2M1_MAX = 6.0
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def power_of_two_ceil(value):
    """2 ** ceil(log2(value)) off the fp32 bits, as the kernels' fast_round_scale
    computes it (kernel.py:22-37)."""
    bits = jax.lax.bitcast_convert_type(value.astype(jnp.float32), jnp.int32)
    exponent = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).astype(jnp.int32)
    return jax.lax.bitcast_convert_type((exponent + 127) << 23, jnp.float32)


def round_e4m3fn(values):
    """Round fp32 values within +-448 to E4M3FN, ties to even, by arithmetic.

    Not `astype(float8_e4m3fn)`: XLA GPU's default xla_allow_excess_precision
    deletes an f32 -> f8 -> f32 convert pair under jit, so the rounding would
    silently not happen. Nor `jax.lax.reduce_precision(x, 4, 3)`, which
    models IEEE-style e4m3 with infinities and a largest finite 240, not the
    FN format whose largest finite is 448. The quantum is the power of two
    of the value's binade less three mantissa bits, floored at the subnormal
    spacing 2**-9; dividing by it is exact and `round` ties to even.
    """
    bits = jax.lax.bitcast_convert_type(values.astype(jnp.float32), jnp.int32)
    exponent = jnp.maximum(((bits >> 23) & 0xFF) - 127, -6) - 3
    quantum = jax.lax.bitcast_convert_type((exponent + 127) << 23, jnp.float32)
    return jnp.round(values / quantum) * quantum


def _straight_through(x, rounded):
    """`rounded` forward, bit for bit in any dtype, and the identity's
    gradient backward. `stop_gradient(x) - x` is +0, and subtracting +0
    leaves every value as it is, a negative zero included, which the
    kernels keep; adding it would turn -0 into +0, and `x + (rounded - x)`
    re-rounds in bf16 wherever the quantizer clamped far from `x`. The
    estimator is Dew's choice; the release is inference code and the paper
    names none."""
    return jax.lax.stop_gradient(rounded.astype(x.dtype)) - (jax.lax.stop_gradient(x) - x)


def fake_quant_fp8(x, block: int):
    """E4M3 per `block` channels under a power-of-two scale (act_quant with
    scale_fmt ue8m0, kernel.py:40-124): amax floored at 1e-4, scale the
    ceiling power of two of amax / 448, value clamped to +-448 and rounded."""
    blocks = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, block)
    amax = jnp.maximum(jnp.max(jnp.abs(blocks), -1, keepdims=True), 1e-4)
    scale = power_of_two_ceil(amax * jnp.float32(1 / E4M3_MAX))
    rounded = round_e4m3fn(jnp.clip(blocks / scale, -E4M3_MAX, E4M3_MAX))
    return _straight_through(x, (rounded * scale).reshape(x.shape))


def _e2m1(values):
    """Round values within +-6 to E2M1, ties to the even neighbour."""
    grid = jnp.asarray(_E2M1, jnp.float32)
    magnitude = jnp.abs(values)
    upper = jnp.clip(jnp.searchsorted(grid, magnitude, side='left'), 1, len(_E2M1) - 1)
    lower = upper - 1
    below, above = grid[lower], grid[upper]
    odd = (lower % 2) == 1
    take_above = (above - magnitude < magnitude - below) | (
        (above - magnitude == magnitude - below) & odd)
    return jnp.copysign(jnp.where(take_above, above, below), values)


def fake_quant_fp4(x, block: int, e4m3_scale: bool):
    """E2M1 per `block` channels (fp4_act_quant, kernel.py:127-204): under an
    E4M3 scale amax / 6 with amax floored at 6 * 2**-9 (the compressed KV),
    or under the ceiling power of two of amax / 6 with amax floored at
    6 * 2**-126 (the indexer); the value clamped to +-6 and rounded."""
    blocks = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, block)
    amax = jnp.max(jnp.abs(blocks), -1, keepdims=True)
    if e4m3_scale:
        # the kernel's cast saturates at E4M3's 448 (cvt.rn.satfinite)
        scale = round_e4m3fn(jnp.minimum(jnp.maximum(amax, E2M1_MAX * 2 ** -9) / E2M1_MAX, E4M3_MAX))
    else:
        scale = power_of_two_ceil(
            jnp.maximum(amax, E2M1_MAX * 2 ** -126) * jnp.float32(1 / E2M1_MAX))
    rounded = _e2m1(jnp.clip(blocks / scale, -E2M1_MAX, E2M1_MAX))
    return _straight_through(x, (rounded * scale).reshape(x.shape))
