"""Rotary position embeddings: the angles, the rotate-half rotation the HF
decoders use and DeepSeek's interleaved-pair one, Llama 3.1's frequency
ramp, and YaRN's frequency ramp and attention factor.

YaRN scaling, which both released DeepSeek configs ask for, reshapes the
inverse frequencies and multiplies the attention scale; the frequency ramp
is the reference's `_compute_yarn_parameters` and the scale multiplier its
`yarn_apply_mscale` (transformers 5.16.1, models/deepseek_v3), both with
`dim` at the rope width, where DeepSeek points `config.head_dim`.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class RopeScaling:
    """Scales rotary frequencies by Llama 3.1's ramp, under the reference's names.

    `_compute_llama3_parameters` (transformers modeling_rope_utils.py:580)
    divides a frequency by `factor` when its wavelength exceeds
    original_max_position_embeddings / low_freq_factor, and leaves it alone
    below original_max_position_embeddings / high_freq_factor. In between it
    interpolates linearly on
    (original_max_position_embeddings / wavelength - low_freq_factor)
    / (high_freq_factor - low_freq_factor).

    `rope_type` is the record's discriminator and only 'llama3' is this
    ramp; YaRN is a mixer kind's own value.
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

    def apply(self, inv_freq):
        """Return the scaled inverse frequencies, the reference's arithmetic in fp32."""
        old_context_len = float(self.original_max_position_embeddings)
        wavelen = 2 * math.pi / inv_freq
        divided = jnp.where(wavelen > old_context_len / self.low_freq_factor,
                            inv_freq / self.factor, inv_freq)
        smooth = ((old_context_len / wavelen - self.low_freq_factor)
                  / (self.high_freq_factor - self.low_freq_factor))
        smoothed = (1 - smooth) * divided / self.factor + smooth * divided
        medium = jnp.logical_and(wavelen >= old_context_len / self.high_freq_factor,
                                 wavelen <= old_context_len / self.low_freq_factor)
        return jnp.where(medium, smoothed, divided)


def rotary_freqs(positions, head_dim: int, theta: float, rot_dim: int | None = None,
                 partial_rotary_type: str = 'proportional',
                 rope_scaling: RopeScaling | None = None):
    """Return cos and sin of the rotary angles at absolute `positions`: [P, pairs].

    `positions` may be [P] for one sequence, or [B, P] for a packed batch
    whose documents each restart at 0; the angle axes line up with the
    trailing [B, S] either way. The angles are computed in fp32, so a token
    rotates the same in a prefill and in a single decode step.

    `rot_dim` narrows the rotation to the first rot_dim dimensions.
    `partial_rotary_type` names which published convention that is, because
    the two rotate different angles:

    - 'proportional' (Gemma 4, modeling_rope_utils.py
      `_compute_proportional_rope_parameters`): the exponents run over the
      full head_dim, `theta ** (2i / head_dim)` for the rot_dim // 2 rotated
      pairs. The rest keep frequency zero, where the rotation is the
      identity. The output is head_dim // 2 wide.
    - 'default' (Qwen3.5, modeling_qwen3_5.py:117-124
      `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters`): the rope
      is rot_dim-dimensional, `theta ** (2i / rot_dim)`, and the output is
      rot_dim // 2 wide. `apply_rotary` passes the trailing dimensions
      through, the reference's `q_rot, q_pass` split.

    With rot_dim None both are the full rotation and the type is moot.
    `rope_scaling` is Llama 3.1's ramp over the base frequencies. It applies
    before a proportional rope pads its zero-frequency tail, as the
    reference's `dim = head_dim * partial_rotary_factor` does.
    """
    if partial_rotary_type not in ('proportional', 'default'):
        raise ValueError(
            "partial_rotary_type names the convention of a partial rotary, "
            f"'proportional' or 'default', got {partial_rotary_type!r}")
    pairs = head_dim // 2 if rot_dim is None else rot_dim // 2
    divisor = head_dim if rot_dim is None or partial_rotary_type == 'proportional' else rot_dim
    inv_freq = 1.0 / (theta ** (jnp.arange(0, 2 * pairs, 2, dtype=jnp.float32) / divisor))
    if rope_scaling is not None:
        inv_freq = rope_scaling.apply(inv_freq)
    if rot_dim is not None and partial_rotary_type == 'proportional':
        padding = head_dim // 2 - pairs
        inv_freq = jnp.concatenate([inv_freq, jnp.zeros((padding,), jnp.float32)])
    positions = jnp.asarray(positions, jnp.float32)
    if positions.ndim == 1:
        angles = positions[:, None] * inv_freq[None, :]
    else:
        angles = positions[:, :, None] * inv_freq[None, None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rotary(x, freqs_cos, freqs_sin, scale: float | None = None):
    """Rotate [B, S, H, D] heads, rotate-half convention as in the HF decoders.

    The freqs are [S, pairs] for one sequence, or [B, S, pairs] when a packed
    batch restarts positions per document. Freqs narrower than D // 2 rotate
    the first 2 * pairs dimensions and pass the rest through, which is the
    sliced partial rotary of `rotary_freqs(partial_rotary_type='default')`.
    `scale` multiplies the whole head inside the fp32 arithmetic, so a
    query's attention scale narrows once, with the product.
    """
    cos = jnp.concatenate([freqs_cos, freqs_cos], axis=-1)
    sin = jnp.concatenate([freqs_sin, freqs_sin], axis=-1)
    if cos.ndim == 3:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    else:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    fp32 = x.astype(jnp.float32)
    rotated_dims = cos.shape[-1]
    fp32, passed = fp32[..., :rotated_dims], fp32[..., rotated_dims:]
    x1, x2 = jnp.split(fp32, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    out = fp32 * cos + rotated * sin
    if passed.shape[-1]:
        out = jnp.concatenate([out, passed], axis=-1)
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


def yarn_inv_freq(head_dim: int, theta: float, yarn: YarnScaling) -> jax.Array:
    """YaRN inverse frequencies over the rope width: `[head_dim // 2]`.

    Mirrors `modeling_rope_utils._compute_yarn_parameters` with `dim` at the
    head dim, as DeepSeek's configs do by pointing `head_dim` at the rope
    slice. Low dims interpolate towards `1 / (factor * pos_freqs)`,
    high dims keep extrapolating, and the linear ramp between the correction
    bounds blends them.
    """
    dim = head_dim
    pairs = dim // 2
    pos_freqs = theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
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
    ramp = jnp.clip((jnp.arange(pairs, dtype=jnp.float32) - low) / span, 0, 1)
    return inv_interpolation * ramp + inv_extrapolation * (1 - ramp)

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
                    yarn: YarnScaling | None):
    """cos/sin over the rope width, plain or YaRN-scaled: `[P, head_dim // 2]`.

    Plain rope is `rotary_freqs`, the one layout every mixer shares. YaRN
    replaces the inverse frequencies with the ramp and scales the resulting
    cos/sin by its attention factor, as the reference's rotary embedding
    does.
    """
    if yarn is None:
        return rotary_freqs(positions, head_dim, theta)
    inv_freq = yarn_inv_freq(head_dim, theta, yarn)
    positions = jnp.asarray(positions, jnp.float32)
    if positions.ndim == 1:
        angles = positions[:, None] * inv_freq[None, :]
    else:
        angles = positions[:, :, None] * inv_freq[None, None, :]
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
    fp32 = x.astype(jnp.float32)
    even, odd = fp32[..., 0::2], fp32[..., 1::2]
    out = jnp.concatenate([even * cos - odd * sin, odd * cos + even * sin],
                          axis=-1)
    return out.astype(x.dtype)
