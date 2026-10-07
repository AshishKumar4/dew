"""Z-Image's single-stream transformer (S3-DiT), as Diffusers 0.40.0's
`ZImageTransformer2DModel` runs it for text-to-image.

Each 2x2 latent patch is one token. Each stream is padded to a multiple of 32
tokens with a learned pad token, and the image's pad tokens sit at position
(0, 0, 0). Two time-modulated refiner blocks run over the image and two
unmodulated ones over the prompt, then the main blocks run over both, image
first. Every block RMS-norms before and after the attention and the SwiGLU,
and the tanh of the time modulation gates its residuals. The three-axis
rotary places prompt token i at (1 + i, 0, 0) and patch (h, w) at
(1 + P, h, w), where P is the padded prompt length. The angles are looked up
in a float64 table rounded to float32, as the source does. The source pads a
batch to its longest row and masks; here the prompt region is the
conditioner's budget rounded up to 32, masked the same way. The source takes
time 1 - sigma and returns the negated flow, while this module takes Dew's
model time and returns Dew's flow.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm
from dew.nn.backbones.unet_condition import sinusoidal_time
from dew.nn.precision import at_least_fp32
from dew.nn.scan_orders import patchify, unpatchify
from dew.nn.sharding import logical_axes

from .joint import JointAttention, layer_norm

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition

MULTIPLE = 32
"""Each stream's padded length is a multiple of this."""
FREQUENCIES = 256
"""The time's sinusoid channels, and the widest time embedding the blocks
modulate by: `min(dim, 256)`."""


def rotary_table(dim: int, length: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one axis of `RopeEmbedder.precompute_freqs_cis`: cosines and sines `[length, dim // 2]`.

    There is one angle per channel pair at each integer position. The angles
    are float64 products rounded to float32, and the source takes their
    float32 cosine and sine.
    """
    frequencies = 1.0 / theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    angles = np.outer(np.arange(length, dtype=np.float64), frequencies).astype(np.float32).astype(np.float64)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def _rotation(positions, axes: Sequence[int], lengths: Sequence[int], theta: float, dtype):
    """The rotary cosines and sines for integer `positions` `[B, S, 3]` in
    `dtype`, each channel pair's value repeated for both channels,
    `[B, S, 1, sum(axes)]`."""
    cosines, sines = [], []
    for index, (dim, length) in enumerate(zip(axes, lengths, strict=True)):
        cos_table, sin_table = rotary_table(dim, length, theta)
        cosines.append(jnp.asarray(cos_table, dtype)[positions[..., index]])
        sines.append(jnp.asarray(sin_table, dtype)[positions[..., index]])
    return (jnp.repeat(jnp.concatenate(cosines, axis=-1), 2, axis=-1)[:, :, None],
            jnp.repeat(jnp.concatenate(sines, axis=-1), 2, axis=-1)[:, :, None])


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
        gated = nn.silu(nn.Dense(self.hidden, name="w1", **dense)(x)) * nn.Dense(
            self.hidden, name="w3", **dense
        )(x)
        return nn.Dense(self.features, name="w2", **dense)(gated)


class ZImageBlock(nn.Module):
    """Runs one `ZImageTransformerBlock`.

    With `modulation`, one projection of the time embedding gives the input
    scales (one plus the projection) and the output gates (its tanh) of the
    attention and the feed-forward. Without it, as in the prompt's refiner
    blocks, the block has no time modulation.
    """

    features: int
    heads: int
    epsilon: float = 1e-5
    modulation: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.compact
    def __call__(self, x, rotation, lengths, embedded=None):
        def norm(name):
            return RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name=name)

        # `ZSingleStreamAttnProcessor`: bias-free, its query and key norms at
        # 1e-5 whatever `norm_eps`.
        joint = JointAttention(self.heads, self.features // self.heads, bias=False, epsilon=1e-5,
                               dtype=self.dtype, precision=self.precision, attention_impl=self.attention_impl,
                               name="attention")

        def attention(x):
            return joint(x, rotation=rotation, lengths=lengths)[0]

        feed_forward = _FeedForward(self.features, int(self.features / 3 * 8), dtype=self.dtype,
                                    precision=self.precision, name="feed_forward")
        if not self.modulation:
            x = x + norm("attention_norm2")(attention(norm("attention_norm1")(x)))
            return x + norm("ffn_norm2")(feed_forward(norm("ffn_norm1")(x)))
        modulated = nn.Dense(4 * self.features, dtype=self.dtype, precision=self.precision,
                             name="modulation")(embedded)[:, None]
        scale, gate, scale_mlp, gate_mlp = jnp.split(modulated, 4, axis=-1)
        attended = attention(norm("attention_norm1")(x) * (1 + scale))
        x = x + jnp.tanh(gate) * norm("attention_norm2")(attended)
        return x + jnp.tanh(gate_mlp) * norm("ffn_norm2")(
            feed_forward(norm("ffn_norm1")(x) * (1 + scale_mlp))
        )


@logical_axes({("x_embedder",): (None, "embed"), ("cap_embedder",): (None, "embed"),
               ("final_linear",): ("embed", None), ("final_modulation",): (None, "embed"),
               ("t_embedder_1",): (None, "mlp"), ("t_embedder_2",): ("mlp", None)})
class ZImageTransformer(nn.Module):
    """Runs Diffusers 0.40.0's `ZImageTransformer2DModel` behind Dew's model interface.

    `__call__` takes NHWC latents, the model time the schedule supplies (the
    sigma times the training count) and a `DenoisingCondition`, and returns
    the flow. The condition's `context` is the prompt states, padded on the
    right to a fixed budget, and its `mask` marks the real tokens.
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

    @property
    def text_keyword(self) -> str:
        """Every call takes the caption as `conditioning`, which joins the image's single stream."""
        return "conditioning"

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
        """Return the flow for the latents `x` at `time` under `conditioning`.

        `train` is part of the objective's standard call; the published
        transformer has no dropout, so it changes nothing here. Raises
        `ValueError` for latents of odd height or width, and for a condition
        without a `mask`.
        """
        if x.ndim != 4 or x.shape[1] % 2 or x.shape[2] % 2:
            raise ValueError(f"Z-Image takes NHWC latents of even height and width, got shape {x.shape}")
        if conditioning.mask is None:
            raise ValueError("Z-Image reads the prompt's real tokens; its conditioning needs a mask")
        batch, height, width, channels = x.shape
        rows, columns = height // 2, width // 2
        count = rows * columns
        image_span = -(-count // MULTIPLE) * MULTIPLE
        context, mask = conditioning.context, conditioning.mask
        # The prompt region holds every row's real tokens rounded up to 32.
        extra = -context.shape[1] % MULTIPLE
        context = jnp.pad(context, ((0, 0), (0, extra), (0, 0)))
        mask = jnp.pad(mask, ((0, 0), (0, extra)))
        budget = context.shape[1]
        spans = -(-jnp.sum(mask, axis=1, dtype=jnp.int32) // MULTIPLE) * MULTIPLE
        slots = jnp.arange(budget)

        embedded = self._embedded_time(time, x.dtype)
        pad_image = self.param("x_pad_token", nn.initializers.zeros, (1, self.dim))
        pad_caption = self.param("cap_pad_token", nn.initializers.zeros, (1, self.dim))
        image = nn.Dense(self.dim, dtype=self.dtype, precision=self.precision, name="x_embedder")(
            patchify(x, 2))
        image = jnp.concatenate(
            [image, jnp.broadcast_to(pad_image.astype(image.dtype), (batch, image_span - count, self.dim))],
            axis=1,
        )
        caption = nn.Dense(self.dim, dtype=self.dtype, precision=self.precision, name="cap_embedder")(
            RMSNorm(epsilon=self.norm_eps, dtype=self.dtype, name="cap_norm")(context))
        caption = jnp.where(mask[..., None], caption, pad_caption.astype(caption.dtype))

        grid = np.indices((rows, columns)).reshape(2, -1).T
        image_positions = jnp.concatenate([
            jnp.concatenate([jnp.broadcast_to(spans[:, None, None] + 1, (batch, count, 1)),
                             jnp.broadcast_to(jnp.asarray(grid)[None], (batch, count, 2))], axis=-1),
            jnp.zeros((batch, image_span - count, 3), jnp.int32)], axis=1)
        # Slots past a row's padded prompt are masked; they sit at the origin.
        leading = jnp.where(slots[None] < spans[:, None], slots[None] + 1, 0)
        caption_positions = jnp.stack([leading, jnp.zeros_like(leading), jnp.zeros_like(leading)], axis=-1)
        table = (self.axes_dims, self.axes_lens, self.rope_theta)
        image_rotation = _rotation(image_positions, *table, image.dtype)
        caption_rotation = _rotation(caption_positions, *table, image.dtype)
        whole_image = jnp.full((batch,), image_span, jnp.int32)
        block = {"epsilon": self.norm_eps, "dtype": self.dtype, "precision": self.precision,
                 "attention_impl": self.attention_impl}

        for index in range(self.n_refiner_layers):
            image = ZImageBlock(self.dim, self.n_heads, name=f"noise_refiner_{index}", **block)(
                image, image_rotation, whole_image, embedded)
        for index in range(self.n_refiner_layers):
            caption = ZImageBlock(
                self.dim, self.n_heads, modulation=False, name=f"context_refiner_{index}", **block
            )(caption, caption_rotation, spans)
        joined = jnp.concatenate([image, caption], axis=1)
        rotation = tuple(jnp.concatenate(pair, axis=1)
                         for pair in zip(image_rotation, caption_rotation, strict=True))
        for index in range(self.n_layers):
            joined = ZImageBlock(self.dim, self.n_heads, name=f"layers_{index}", **block)(
                joined, rotation, image_span + spans, embedded)

        scale = 1 + nn.Dense(self.dim, dtype=self.dtype, precision=self.precision,
                             name="final_modulation")(nn.silu(embedded))[:, None]
        out = nn.Dense(4 * channels, dtype=self.dtype, precision=self.precision, name="final_linear")(
            layer_norm(self.dtype)(joined[:, :count]) * scale)
        return -unpatchify(out, 2, *x.shape[1:])


__all__ = ["ZImageBlock", "ZImageTransformer", "rotary_table"]
