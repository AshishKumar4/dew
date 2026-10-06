"""Rotary position embeddings: the angles, the rotate-half rotation the HF
decoders use and DeepSeek's interleaved-pair one, Llama 3.1's frequency
ramp, and YaRN's frequency ramp and attention factor.

YaRN's ramp is transformers 5.16.1's `_compute_yarn_parameters` and its
scale `yarn_apply_mscale` (models/deepseek_v3), with `dim` at the rope
width. The inverse-frequency tables are static, built on the host in NumPy
as transformers builds them on the CPU, in the dtype the angles are computed
in (`at_least_fp32` of the rotated activations'). The one transcendental,
`theta ** x`, rounds once from float64 (`_base_powers`), within one ulp of
transformers' float32 `pow`.
"""

import dataclasses
import math

import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from .precision import at_least_fp32


def inverse_frequencies(theta: float, dim: int, pairs: int | None = None, *,
                        dtype: DTypeLike) -> np.ndarray:
    """`1 / theta ** (2i / dim)` for i below `pairs` (all of `dim // 2` by
    default), transformers' `compute_default_rope_parameters` table, built on
    the host in `dtype` (`_base_powers`)."""
    count = dim // 2 if pairs is None else pairs
    return 1.0 / _base_powers(theta, np.arange(0, 2 * count, 2, dtype=dtype) / dim)


def _base_powers(theta: float, exponents: np.ndarray) -> np.ndarray:
    """`theta ** exponents` correctly rounded to the exponents' dtype,
    whatever the backend: computed in float64 and rounded once."""
    return np.power(np.float64(theta), exponents.astype(np.float64)).astype(exponents.dtype)


@dataclasses.dataclass(frozen=True)
class RopeScaling:
    """Scales rotary frequencies by Llama 3.1's ramp, under the reference's names.

    `_compute_llama3_parameters` (modeling_rope_utils.py:580) divides a
    frequency by `factor` when its wavelength exceeds
    original_max_position_embeddings / low_freq_factor, keeps it below
    original_max_position_embeddings / high_freq_factor, and interpolates
    linearly on (original_max_position_embeddings / wavelength -
    low_freq_factor) / (high_freq_factor - low_freq_factor) between. Only
    `rope_type` 'llama3' is this ramp.
    """

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int
    rope_type: str = 'llama3'

    def __post_init__(self):
        if self.rope_type != 'llama3':
            raise ValueError(
                f"rope_scaling applies the llama3 ramp, got rope_type {self.rope_type!r}")
        if self.factor < 1.0:
            raise ValueError(f"rope_scaling factor is at least 1, got {self.factor}")
        if self.high_freq_factor <= self.low_freq_factor:
            raise ValueError(
                f"rope_scaling needs high_freq_factor above low_freq_factor, got "
                f"{self.high_freq_factor} and {self.low_freq_factor}")
        if self.original_max_position_embeddings < 1:
            raise ValueError(
                "rope_scaling original_max_position_embeddings is the pretraining "
                f"context, got {self.original_max_position_embeddings}")

    def apply(self, inv_freq: np.ndarray) -> np.ndarray:
        """Return the scaled inverse frequencies, the reference's arithmetic
        on the host in the table's own dtype."""
        old_context_len = float(self.original_max_position_embeddings)
        wavelen = 2 * math.pi / inv_freq
        divided = np.where(wavelen > old_context_len / self.low_freq_factor,
                           inv_freq / self.factor, inv_freq)
        smooth = ((old_context_len / wavelen - self.low_freq_factor)
                  / (self.high_freq_factor - self.low_freq_factor))
        smoothed = (1 - smooth) * divided / self.factor + smooth * divided
        medium = np.logical_and(wavelen >= old_context_len / self.high_freq_factor,
                                wavelen <= old_context_len / self.low_freq_factor)
        return np.where(medium, smoothed, divided).astype(inv_freq.dtype)


def rotary_freqs(positions, head_dim: int, theta: float, rot_dim: int | None = None,
                 partial_rotary_type: str = 'proportional',
                 rope_scaling: RopeScaling | None = None, *, dtype: DTypeLike):
    """Return cos and sin of the rotary angles at absolute `positions`: [P, pairs].

    `positions` is [P], or [B, P] for a packed batch whose documents restart at
    0. The angles are computed in `dtype` (at least fp32), so a token rotates
    the same in prefill and decode.

    `rot_dim` narrows the rotation to the first rot_dim dimensions, under one
    of two published conventions that rotate different angles:

    - 'proportional' (Gemma 4, `_compute_proportional_rope_parameters`):
      `theta ** (2i / head_dim)` for the rot_dim // 2 rotated pairs, the rest at
      frequency zero; head_dim // 2 wide.
    - 'default' (Qwen3.5, modeling_qwen3_5.py:117-124): a rot_dim-wide rope,
      `theta ** (2i / rot_dim)`, rot_dim // 2 wide; `apply_rotary` passes the
      rest through, the reference's `q_rot, q_pass` split.

    `rope_scaling` (Llama 3.1's ramp) applies before a proportional rope pads
    its zero-frequency tail, as the reference's `dim = head_dim *
    partial_rotary_factor` does.
    """
    if partial_rotary_type not in ('proportional', 'default'):
        raise ValueError(
            "partial_rotary_type names the convention of a partial rotary, "
            f"'proportional' or 'default', got {partial_rotary_type!r}")
    pairs = head_dim // 2 if rot_dim is None else rot_dim // 2
    divisor = head_dim if rot_dim is None or partial_rotary_type == 'proportional' else rot_dim
    inv_freq = inverse_frequencies(theta, divisor, pairs, dtype=dtype)
    if rope_scaling is not None:
        inv_freq = rope_scaling.apply(inv_freq)
    if rot_dim is not None and partial_rotary_type == 'proportional':
        padding = head_dim // 2 - pairs
        inv_freq = np.concatenate([inv_freq, np.zeros((padding,), inv_freq.dtype)])
    angles = jnp.asarray(positions, inv_freq.dtype)[..., None] * inv_freq
    return jnp.cos(angles), jnp.sin(angles)


def apply_rotary(x, freqs_cos, freqs_sin, scale: float | None = None):
    """Rotate [B, S, H, D] heads, rotate-half convention as in the HF decoders.

    The freqs are [S, pairs] for one sequence, or [B, S, pairs] when a packed
    batch restarts positions per document. Freqs narrower than D // 2 rotate
    the first 2 * pairs dimensions and pass the rest through, which is the
    sliced partial rotary of `rotary_freqs(partial_rotary_type='default')`.
    `scale` multiplies the whole head inside the arithmetic, at least fp32,
    so a query's attention scale narrows once, with the product.

    The halves rotate as `x1 cos - x2 sin` and `x2 cos + x1 sin`, which is
    `x cos + rotate_half(x) sin` without the rotated copy. On a TPU v6e the
    copy's slice-and-negate ran as separate passes: without it Qwen3-0.6B's
    training step at 8 x 1024 took 140.6 against 145.6 ms, and Qwen3-1.7B's
    at 4 x 1024 150.0 against 153.8 (docs/performance.md).
    """
    if freqs_cos.ndim == 3:
        cos, sin = freqs_cos[:, :, None, :], freqs_sin[:, :, None, :]
    else:
        cos, sin = freqs_cos[None, :, None, :], freqs_sin[None, :, None, :]
    wide = x.astype(at_least_fp32(x.dtype))
    pairs = cos.shape[-1]
    x1, x2, passed = wide[..., :pairs], wide[..., pairs:2 * pairs], wide[..., 2 * pairs:]
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, passed], axis=-1)
    return (out if scale is None else out * scale).astype(x.dtype)


@dataclasses.dataclass(frozen=True)
class YarnScaling:
    """YaRN rope scaling, the reference's `rope_parameters` fields.

    Both released DeepSeek configs carry this spelling (rope_type yarn,
    factor 40 off 4096 base positions), so the record keeps the reference's
    names and a translation renames nothing. `rope_theta` repeats the
    mixer's own base, and the two must agree, so the scaling is configured
    once (the mscale is applied in the attention as a query pre-scale).
    """

    rope_type: str = 'yarn'
    rope_theta: float = 10000.0
    factor: float = 40.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float | None = None
    mscale_all_dim: float | None = None
    truncate: bool = True
    # An explicit cos/sin amplitude, which the reference applies instead of
    # deriving one; None derives it from factor and the mscales above.
    attention_factor: float | None = None


def yarn_inv_freq(head_dim: int, theta: float, yarn: YarnScaling, *,
                  dtype: DTypeLike) -> np.ndarray:
    """YaRN inverse frequencies over the rope width, `[head_dim // 2]`, in `dtype`.

    Mirrors `modeling_rope_utils._compute_yarn_parameters` with `dim` at the
    head dim, as DeepSeek's configs do by pointing `head_dim` at the rope
    slice. Low dims interpolate towards `1 / (factor * pos_freqs)`,
    high dims keep extrapolating, and the linear ramp between the correction
    bounds blends them.
    """
    dim = head_dim
    pairs = dim // 2
    pos_freqs = _base_powers(theta, np.arange(0, dim, 2, dtype=dtype) / dim)
    inv_extrapolation = 1.0 / pos_freqs
    inv_interpolation = 1.0 / (yarn.factor * pos_freqs)

    def correction_dim(rotations: float) -> float:
        """The dimension seeing `rotations` turns over the original context."""
        return (dim * math.log(yarn.original_max_position_embeddings
                               / (rotations * 2 * math.pi))
                / (2 * math.log(theta)))

    low = correction_dim(yarn.beta_fast)
    high = correction_dim(yarn.beta_slow)
    if yarn.truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    span = high - low
    if span == 0:
        # The reference nudges a degenerate bound to keep the division finite.
        span = 0.001
    ramp = np.clip((np.arange(pairs, dtype=dtype) - low) / span, 0, 1)
    return (inv_interpolation * ramp + inv_extrapolation * (1 - ramp)).astype(dtype)

def yarn_attention_factor(yarn: YarnScaling) -> float:
    """The cos/sin multiplier of `_compute_yarn_parameters`.

    Both released configs set mscale and mscale_all_dim to 1.0, so this is
    1.0 for them; a config that sets them apart rotates at a different
    amplitude.
    """
    if yarn.attention_factor is not None:
        return float(yarn.attention_factor)
    def mscale(scale: float, weight: float) -> float:
        return 1.0 if scale <= 1 else 0.1 * weight * math.log(scale) + 1.0

    if yarn.mscale and yarn.mscale_all_dim:
        return float(mscale(yarn.factor, yarn.mscale)
                     / mscale(yarn.factor, yarn.mscale_all_dim))
    return float(mscale(yarn.factor, 1.0))


def yarn_query_scale(yarn: YarnScaling) -> float:
    """The attention-scale multiplier of `yarn_apply_mscale`, squared.

    The reference folds this into the softmax scale, while dew's kernels
    scale by `1 / sqrt(head_dim)` themselves, so the query carries the
    ratio, the same way `attention_scale` does on the standard mixer. For
    the released configs this is `(0.1 * ln(40) + 1) ** 2`.
    """
    if not yarn.mscale_all_dim or yarn.factor <= 1:
        return 1.0
    mscale = 0.1 * yarn.mscale_all_dim * math.log(yarn.factor) + 1.0
    return mscale * mscale


def yarn_rope_freqs(positions, head_dim: int, theta: float,
                    yarn: YarnScaling | None, *, dtype: DTypeLike):
    """cos/sin over the rope width, plain or YaRN-scaled: `[P, head_dim // 2]`,
    computed in `dtype` as `rotary_freqs`'s are.

    Plain rope is `rotary_freqs`, the one layout every mixer shares. YaRN
    replaces the inverse frequencies with the ramp and scales the resulting
    cos/sin by its attention factor, as the reference's rotary embedding
    does.
    """
    if yarn is None:
        return rotary_freqs(positions, head_dim, theta, dtype=dtype)
    inv_freq = yarn_inv_freq(head_dim, theta, yarn, dtype=dtype)
    angles = jnp.asarray(positions, inv_freq.dtype)[..., None] * inv_freq
    factor = yarn_attention_factor(yarn)
    return jnp.cos(angles) * factor, jnp.sin(angles) * factor


def apply_rotary_interleave(x, freqs_cos, freqs_sin):
    """Rotate `[B, S, H, D]` heads pairwise, DeepSeek's rope convention.

    Pairs `(x0, x1), (x2, x3), ...` each rotate by one frequency
    (`modeling_deepseek_v3.apply_rotary_pos_emb_interleave`): the even and
    odd slices turn against the first half of the cos/sin, and the halves
    stack real over imaginary without interleaving back. Query and key
    take the same layout, so the dot product keeps the complex structure.
    """
    if freqs_cos.ndim == 3:
        cos = freqs_cos[:, :, None, :]
        sin = freqs_sin[:, :, None, :]
    else:
        cos = freqs_cos[None, :, None, :]
        sin = freqs_sin[None, :, None, :]
    wide = x.astype(at_least_fp32(x.dtype))
    even, odd = wide[..., 0::2], wide[..., 1::2]
    out = jnp.concatenate([even * cos - odd * sin, odd * cos + even * sin],
                          axis=-1)
    return out.astype(x.dtype)
