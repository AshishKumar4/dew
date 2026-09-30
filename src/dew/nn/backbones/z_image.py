"""Z-Image's single-stream transformer (S3-DiT), as Diffusers 0.40.0's
`ZImageTransformer2DModel` runs it for text-to-image.

The latent is cut into 2x2 patches, one token each, and the prompt states
are the text encoder's own. Each stream is padded to a multiple of 32
tokens: the image's padding holds a learned image pad token at position
(0, 0, 0), the prompt's a learned caption pad token. Two refiner blocks
modulated by the time run over the image tokens, two unmodulated ones over
the prompt, and then the main blocks over both joined, image first. Every
block is pre- and post-normed: RMS norms before and after the attention and
the SwiGLU feed-forward, the residual branches gated by the tanh of the time
modulation. A three-axis rotary table places prompt token i at (1 + i, 0, 0)
and image patch (h, w) at (1 + P, h, w), P being the padded prompt length,
which differs row to row; the source looks each position up in a table it
computed in float64 and rounded to float32, so this does too.

Beyond the padding the source pads each batch to its longest row and masks
those keys. Here the prompt region is the conditioner's budget wide, the
prompt and its padding lead it, and the rest is masked the same way.

The source is called with the time 1 - sigma and its output is the negated
flow; this module takes Dew's model time, sigma times the training count,
and returns the flow Dew's process reads, so the conversion lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm, scaled_dot_product_attention
from dew.nn.backbones.unet_condition import sinusoidal_time
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import logical_axes
from dew.registry import models

from .flux import apply_rotary
from .sd3 import _layer_norm

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition

MULTIPLE = 32
"""Each stream's padded length is a multiple of this."""
FREQUENCIES = 256
"""The time's sinusoid channels, and the widest time embedding the blocks
modulate by: `min(dim, 256)`."""


def rotary_table(dim: int, length: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """One axis of `RopeEmbedder.precompute_freqs_cis`: the cosine and sine
    of each integer position's angles, `[length, dim // 2]`. The angles are
    float64 products rounded to float32, and the source takes their float32
    cosine and sine."""
    frequencies = 1.0 / theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    angles = np.outer(np.arange(length, dtype=np.float64), frequencies).astype(np.float32).astype(np.float64)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def _rotation(positions, axes: Sequence[int], lengths: Sequence[int], theta: float):
    """The rotary cosines and sines for integer `positions` `[B, S, 3]`,
    each channel pair's value repeated for both channels, `[B, S, sum(axes)]`."""
    cosines, sines = [], []
    for index, (dim, length) in enumerate(zip(axes, lengths, strict=True)):
        cos_table, sin_table = rotary_table(dim, length, theta)
        cosines.append(jnp.asarray(cos_table)[positions[..., index]])
        sines.append(jnp.asarray(sin_table)[positions[..., index]])
    return (jnp.repeat(jnp.concatenate(cosines, axis=-1), 2, axis=-1),
            jnp.repeat(jnp.concatenate(sines, axis=-1), 2, axis=-1))


class _Attention(nn.Module):
    """`Attention` under `ZSingleStreamAttnProcessor`: bias-free projections,
    per-head RMS norms on the queries and keys, the rotation, and the keys
    past each row's length masked."""

    heads: int
    head_dim: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.compact
    def __call__(self, x, cos, sin, lengths):
        inner = self.heads * self.head_dim
        split = (x.shape[0], x.shape[1], self.heads, self.head_dim)

        def projection(name):
            return nn.Dense(inner, use_bias=False, dtype=self.dtype, precision=self.precision, name=name)(x).reshape(
                split)

        query = RMSNorm(epsilon=1e-5, dtype=self.dtype, name="norm_q")(projection("to_q"))
        key = RMSNorm(epsilon=1e-5, dtype=self.dtype, name="norm_k")(projection("to_k"))
        rotation = (cos[:, :, None, :], sin[:, :, None, :])
        attended = scaled_dot_product_attention(
            apply_rotary(query, *rotation), apply_rotary(key, *rotation), projection("to_v"),
            implementation=self.attention_impl, precision=self.precision, key_value_seq_lengths=lengths)
        return nn.Dense(self.heads * self.head_dim, use_bias=False, dtype=self.dtype, precision=self.precision,
                        name="to_out_0")(attended.reshape(x.shape[0], x.shape[1], inner))


@logical_axes({("w1",): ("embed", "mlp"), ("w3",): ("embed", "mlp"), ("w2",): ("mlp", "embed")})
class _FeedForward(nn.Module):
    """`FeedForward`: SiLU of `w1` gating `w3`, then `w2`, no biases."""

    features: int
    hidden: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        dense = {"use_bias": False, "dtype": self.dtype, "precision": self.precision}
        gated = nn.silu(nn.Dense(self.hidden, name="w1", **dense)(x)) * nn.Dense(self.hidden, name="w3", **dense)(x)
        return nn.Dense(self.features, name="w2", **dense)(gated)


class ZImageBlock(nn.Module):
    """One `ZImageTransformerBlock`. With `modulation`, the time embedding's
    one map gives the attention's and the feed-forward's input scales (one
    plus) and output gates (tanh)."""

    features: int
    heads: int
    epsilon: float = 1e-5
    modulation: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.compact
    def __call__(self, x, cos, sin, lengths, embedded=None):
        def norm(name):
            return RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name=name)

        attention = _Attention(self.heads, self.features // self.heads, dtype=self.dtype, precision=self.precision,
                               attention_impl=self.attention_impl, name="attention")
        feed_forward = _FeedForward(self.features, int(self.features / 3 * 8), dtype=self.dtype,
                                    precision=self.precision, name="feed_forward")
        if not self.modulation:
            x = x + norm("attention_norm2")(attention(norm("attention_norm1")(x), cos, sin, lengths))
            return x + norm("ffn_norm2")(feed_forward(norm("ffn_norm1")(x)))
        modulated = nn.Dense(4 * self.features, dtype=self.dtype, precision=self.precision,
                             name="modulation")(embedded)[:, None]
        scale, gate, scale_mlp, gate_mlp = jnp.split(modulated, 4, axis=-1)
        attended = attention(norm("attention_norm1")(x) * (1 + scale), cos, sin, lengths)
        x = x + jnp.tanh(gate) * norm("attention_norm2")(attended)
        return x + jnp.tanh(gate_mlp) * norm("ffn_norm2")(feed_forward(norm("ffn_norm1")(x) * (1 + scale_mlp)))


@models("z_image_transformer")
@logical_axes({("x_embedder",): (None, "embed"), ("cap_embedder",): (None, "embed"),
               ("final_linear",): ("embed", None), ("final_modulation",): (None, "embed"),
               ("t_embedder_1",): (None, "mlp"), ("t_embedder_2",): ("mlp", None)})
class ZImageTransformer(nn.Module):
    """Diffusers 0.40.0's `ZImageTransformer2DModel` over Dew's interface.

    `__call__` takes NHWC latents, the model time the schedule supplies (the
    sigma times the training count) and a `DenoisingCondition` whose
    `context` is the prompt states padded on the right to a fixed budget and
    whose `mask` marks the real tokens, and returns the flow.
    """

    in_channels: int = 16
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    norm_eps: float = 1e-5
    cap_feat_dim: int = 2560
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: Sequence[int] = (32, 48, 48)
    axes_lens: Sequence[int] = (1536, 512, 512)
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    def _embedded_time(self, time, dtype):
        """`TimestepEmbedder` of the source's time, (1000 - time) / 1000 times
        `t_scale`: 256 sinusoids, cosines first, 1024 wide inside, and
        `min(dim, 256)` out."""
        width = at_least_fp32(self.dtype)
        source_time = (1000.0 - jnp.asarray(time, width)) / 1000.0 * self.t_scale
        features = sinusoidal_time(source_time, FREQUENCIES, dtype=width).astype(dtype)
        hidden = nn.Dense(1024, dtype=self.dtype, precision=self.precision, name="t_embedder_1")(features)
        return nn.Dense(min(self.dim, FREQUENCIES), dtype=self.dtype, precision=self.precision,
                        name="t_embedder_2")(nn.silu(hidden))

    @nn.compact
    def __call__(self, x, time, conditioning: DenoisingCondition, train: bool = False):
        """`train` is the objective's standard call contract; the published
        transformer holds no dropout, so it changes nothing here."""
        if x.ndim != 4 or x.shape[1] % 2 or x.shape[2] % 2:
            raise ValueError(f"Z-Image takes NHWC latents of even height and width, got shape {x.shape}")
        if conditioning.mask is None:
            raise ValueError("Z-Image reads the prompt's real tokens; its conditioning needs a mask")
        batch, height, width, channels = x.shape
        rows, columns = height // 2, width // 2
        count = rows * columns
        image_span = -(-count // MULTIPLE) * MULTIPLE
        budget = conditioning.context.shape[1]
        if budget % MULTIPLE:
            raise ValueError(f"the prompt budget {budget} is not a multiple of {MULTIPLE}")
        # The prompt and its padding: its real tokens rounded up to 32.
        spans = -(-jnp.sum(conditioning.mask, axis=1, dtype=jnp.int32) // MULTIPLE) * MULTIPLE
        slots = jnp.arange(budget)

        embedded = self._embedded_time(time, x.dtype)
        pad_image = self.param("x_pad_token", nn.initializers.zeros, (1, self.dim))
        pad_caption = self.param("cap_pad_token", nn.initializers.zeros, (1, self.dim))
        patches = x.reshape(batch, rows, 2, columns, 2, channels).transpose(0, 1, 3, 2, 4, 5).reshape(
            batch, count, 4 * channels)
        image = nn.Dense(self.dim, dtype=self.dtype, precision=self.precision, name="x_embedder")(patches)
        image = jnp.concatenate(
            [image, jnp.broadcast_to(pad_image.astype(image.dtype), (batch, image_span - count, self.dim))], axis=1)
        caption = nn.Dense(self.dim, dtype=self.dtype, precision=self.precision, name="cap_embedder")(
            RMSNorm(epsilon=self.norm_eps, dtype=self.dtype, name="cap_norm")(conditioning.context))
        caption = jnp.where(conditioning.mask[..., None], caption, pad_caption.astype(caption.dtype))

        grid = np.indices((rows, columns)).reshape(2, -1).T
        image_positions = jnp.concatenate([
            jnp.concatenate([jnp.broadcast_to(spans[:, None, None] + 1, (batch, count, 1)),
                             jnp.broadcast_to(jnp.asarray(grid)[None], (batch, count, 2))], axis=-1),
            jnp.zeros((batch, image_span - count, 3), jnp.int32)], axis=1)
        # Slots past a row's padded prompt are masked; they sit at the origin.
        leading = jnp.where(slots[None] < spans[:, None], slots[None] + 1, 0)
        caption_positions = jnp.stack([leading, jnp.zeros_like(leading), jnp.zeros_like(leading)], axis=-1)
        rotation = (self.axes_dims, self.axes_lens, self.rope_theta)
        image_cos, image_sin = _rotation(image_positions, *rotation)
        caption_cos, caption_sin = _rotation(caption_positions, *rotation)
        whole_image = jnp.full((batch,), image_span, jnp.int32)
        block = {"epsilon": self.norm_eps, "dtype": self.dtype, "precision": self.precision,
                 "attention_impl": self.attention_impl}

        for index in range(self.n_refiner_layers):
            image = ZImageBlock(self.dim, self.n_heads, name=f"noise_refiner_{index}", **block)(
                image, image_cos, image_sin, whole_image, embedded)
        for index in range(self.n_refiner_layers):
            caption = ZImageBlock(self.dim, self.n_heads, modulation=False, name=f"context_refiner_{index}", **block)(
                caption, caption_cos, caption_sin, spans)
        joined = jnp.concatenate([image, caption], axis=1)
        cos = jnp.concatenate([image_cos, caption_cos], axis=1)
        sin = jnp.concatenate([image_sin, caption_sin], axis=1)
        for index in range(self.n_layers):
            joined = ZImageBlock(self.dim, self.n_heads, name=f"layers_{index}", **block)(
                joined, cos, sin, image_span + spans, embedded)

        scale = 1 + nn.Dense(self.dim, dtype=self.dtype, precision=self.precision,
                             name="final_modulation")(nn.silu(embedded))[:, None]
        out = nn.Dense(4 * channels, dtype=self.dtype, precision=self.precision, name="final_linear")(
            _layer_norm(self.dtype)(joined[:, :count]) * scale)
        out = out.reshape(batch, rows, columns, 2, 2, channels).transpose(0, 1, 3, 2, 4, 5).reshape(x.shape)
        return -out


__all__ = ["ZImageBlock", "ZImageTransformer", "rotary_table"]
