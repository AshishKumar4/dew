"""Stable Diffusion 3's own MM-DiT, as the published transformer computes it.

`SimpleMMDiT` is Dew's own dual-stream model, trained from scratch on Dew's
conventions. This module is the other thing: the arithmetic of Diffusers
0.34.0's `SD3Transformer2DModel`, so a published checkpoint's tensors mean
here what they mean there. The differences from the scratch model are not
cosmetic - the modulation channel order, the joint attention's image-then-
context concatenation, the position buffer's centred crop, the separate
timestep and pooled-text embedders that are summed, the last block's
context-only continuous norm, and SD3.5's ninefold modulation with a second
self-attention - so the two stay separate rather than one growing flags.

The interface is Dew's: NHWC noisy latents, a model time, a
`DenoisingCondition` carrying the text tokens and the pooled text vector,
and NHWC velocity out. Position embeddings are a persistent sin/cos buffer
in the source, not a learned parameter, so they ride in the `buffers`
collection: an optimizer never sees them and the checkpoint's own stored
values are what the model reads and what export writes back.
"""

from __future__ import annotations

import math
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm, scaled_dot_product_attention
from dew.nn.backbones.unet_condition import DenoisingCondition, sinusoidal_time
from dew.nn.sharding import logical_axes
from dew.registry import models


def sincos_position(channels: int, grid: int, *, base_size: int,
                    interpolation_scale: float = 1.0):
    """`get_2d_sincos_pos_embed` at the grid the source builds its buffer on.

    The source meshes width first and then reads that first mesh into the
    leading half of the channels, so the leading half carries the column
    coordinate and the trailing half the row, each as sine then cosine over
    frequencies 10000^-(2i/half); both axes are divided by `grid / base_size`
    and the interpolation scale.

    This is only the buffer's initializer. A published checkpoint stores the
    buffer and that stored value is what a load reads; this is here so a
    model built without one starts where the source starts.
    """
    steps = jnp.arange(grid, dtype=jnp.float32) / (grid / base_size) / interpolation_scale
    columns, rows = jnp.meshgrid(steps, steps, indexing="xy")  # width goes first
    half = channels // 2
    omega = 1.0 / 10000 ** (jnp.arange(half // 2, dtype=jnp.float32) / (half / 2.0))

    def axis(values):
        angles = values.reshape(-1)[:, None] * omega[None, :]
        return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=1)

    return jnp.concatenate([axis(columns), axis(rows)], axis=1)[None]


class _Modulation(nn.Module):
    """A source `AdaLayerNormZero`-family projection: SiLU then one linear
    whose output is `pieces` chunks of the width, in the source's order."""

    features: int
    pieces: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, conditioning):
        projected = nn.Dense(self.pieces * self.features, dtype=self.dtype,
                             precision=self.precision, name="linear")(nn.silu(conditioning))
        return jnp.split(projected, self.pieces, axis=-1)


def _modulate(x, shift, scale):
    """The source's `norm(x) * (1 + scale) + shift`, broadcast over tokens."""
    return x * (1 + scale[:, None]) + shift[:, None]


# The source stores one flat [inner, inner] tensor per projection, so these
# kernels keep that shape and shard their embedding side. The suffixes carry
# the attention's own name: the scratch MM-DiT declares three-axis `to_q`
# kernels of its own, and one path suffix holds one set of axes.
@logical_axes({(module, name): axes for module in ("attn", "attn2")
               for name, axes in (
                   *(((name), ("embed", None)) for name in
                     ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")),
                   ("to_out_0", (None, "embed")),
                   ("to_add_out", (None, "embed")))})
class _JointAttention(nn.Module):
    """The source's `Attention` under `JointAttnProcessor2_0`.

    Image and context project through their own weights, the heads
    concatenate with the image sequence first, one softmax attends over the
    joint sequence and the halves split back apart. Without a context the
    same weights run plain self-attention, which is SD3.5's second attention.
    """

    heads: int
    head_dim: int
    qk_norm: str | None = None
    context_out: bool = True
    epsilon: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str | None = "auto"

    def _projection(self, name: str, features: int):
        return nn.Dense(features, dtype=self.dtype, precision=self.precision, name=name)

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim)

    def _norm(self, name: str, x):
        if self.qk_norm is None:
            return x
        if self.qk_norm != "rms_norm":
            raise ValueError(f"Native SD3 implements qk_norm 'rms_norm', not {self.qk_norm!r}")
        return RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name=name)(x)

    @nn.compact
    def __call__(self, image, context=None):
        inner = self.heads * self.head_dim
        query = self._norm("norm_q", self._heads(self._projection("to_q", inner)(image)))
        key = self._norm("norm_k", self._heads(self._projection("to_k", inner)(image)))
        value = self._heads(self._projection("to_v", inner)(image))
        tokens = image.shape[1]
        if context is not None:
            added_query = self._norm(
                "norm_added_q", self._heads(self._projection("add_q_proj", inner)(context)))
            added_key = self._norm(
                "norm_added_k", self._heads(self._projection("add_k_proj", inner)(context)))
            added_value = self._heads(self._projection("add_v_proj", inner)(context))
            query = jnp.concatenate([query, added_query], axis=1)
            key = jnp.concatenate([key, added_key], axis=1)
            value = jnp.concatenate([value, added_value], axis=1)
        attended = scaled_dot_product_attention(
            query, key, value, implementation=self.attention_impl, precision=self.precision)
        attended = attended.reshape(attended.shape[0], attended.shape[1], inner)
        if context is None:
            return self._projection("to_out_0", inner)(attended), None
        image_out = self._projection("to_out_0", inner)(attended[:, :tokens])
        context_out = attended[:, tokens:]
        if self.context_out:
            context_out = self._projection("to_add_out", inner)(context_out)
        else:
            context_out = None
        return image_out, context_out


@logical_axes({("net_0_proj",): ("embed", "mlp"), ("net_2",): ("mlp", "embed")})
class _FeedForward(nn.Module):
    """The source's `FeedForward(activation_fn="gelu-approximate")`: one
    projection into the tanh GELU, then one back, stored as `net.0.proj` and
    `net.2`."""

    features: int
    multiplier: int = 4
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        hidden = nn.Dense(self.features * self.multiplier, dtype=self.dtype,
                          precision=self.precision, name="net_0_proj")(x)
        hidden = nn.gelu(hidden, approximate=True)
        return nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                        name="net_2")(hidden)


def _layer_norm(dtype):
    """The source's `LayerNorm(elementwise_affine=False, eps=1e-6)`."""
    return nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False, dtype=dtype)


class SD3Block(nn.Module):
    """One `JointTransformerBlock`.

    The image stream modulates, attends jointly with the context, gates, then
    modulates and runs its feed-forward. The context stream does the same
    unless it is the last block, where `context_pre_only` gives it a
    continuous scale/shift norm, no output projection and no feed-forward.
    Under `use_dual_attention` the image modulation has nine pieces and a
    second self-attention reads a differently modulated copy of the same
    normalized input.
    """

    features: int
    heads: int
    head_dim: int
    context_pre_only: bool = False
    dual_attention: bool = False
    qk_norm: str | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str | None = "auto"

    @nn.compact
    def __call__(self, image, context, conditioning):
        pieces = 9 if self.dual_attention else 6
        modulation = _Modulation(self.features, pieces, dtype=self.dtype,
                                 precision=self.precision, name="norm1")(conditioning)
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = modulation[:6]
        normalized = _layer_norm(self.dtype)(image)
        image_input = _modulate(normalized, shift, scale)
        # A last block's context is read and dropped, so it takes a shift and a
        # scale; a continuing one also takes the gates its own residual and
        # feed-forward use.
        context_chunks = _Modulation(
            self.features, 2 if self.context_pre_only else 6, dtype=self.dtype,
            precision=self.precision, name="norm1_context")(conditioning)
        # The last block's continuous norm emits its scale before its shift;
        # the zero-initialized one emits shift, scale and then the gates.
        context_scale, context_shift = (context_chunks[:2] if self.context_pre_only
                                        else context_chunks[1::-1])
        context_rest = None if self.context_pre_only else context_chunks[2:]
        context_input = _modulate(_layer_norm(self.dtype)(context), context_shift, context_scale)

        attention = _JointAttention(
            self.heads, self.head_dim, self.qk_norm, not self.context_pre_only, dtype=self.dtype,
            precision=self.precision, attention_impl=self.attention_impl, name="attn")
        image_attended, context_attended = attention(image_input, context_input)
        image = image + gate[:, None] * image_attended
        if self.dual_attention:
            shift2, scale2, gate2 = modulation[6:]
            second = _JointAttention(
                self.heads, self.head_dim, self.qk_norm, False, dtype=self.dtype,
                precision=self.precision, attention_impl=self.attention_impl, name="attn2")
            attended2, _ = second(_modulate(normalized, shift2, scale2))
            image = image + gate2[:, None] * attended2
        image_mlp = _modulate(_layer_norm(self.dtype)(image), shift_mlp, scale_mlp)
        image = image + gate_mlp[:, None] * _FeedForward(
            self.features, dtype=self.dtype, precision=self.precision, name="ff")(image_mlp)
        if context_rest is None:
            return image, None
        # A block that keeps its context is the one whose attention returns it.
        assert context_attended is not None
        context_gate, context_shift_mlp, context_scale_mlp, context_gate_mlp = context_rest
        context = context + context_gate[:, None] * context_attended
        context_mlp = _modulate(_layer_norm(self.dtype)(context), context_shift_mlp, context_scale_mlp)
        context = context + context_gate_mlp[:, None] * _FeedForward(
            self.features, dtype=self.dtype, precision=self.precision, name="ff_context")(context_mlp)
        return image, context


@models("sd3_transformer")
@logical_axes({("context_embedder",): (None, "embed"),
               ("proj_out",): ("embed", None),
               ("timestep_embedder_linear_1",): (None, "embed"),
               ("timestep_embedder_linear_2",): (None, "embed"),
               ("text_embedder_linear_1",): (None, "embed"),
               ("text_embedder_linear_2",): (None, "embed")},
              heuristic=(("pos_embed_proj",),))
class SD3Transformer(nn.Module):
    """Diffusers 0.34.0's `SD3Transformer2DModel` over Dew's interface.

    `__call__` takes NHWC latents, the model time and a `DenoisingCondition`
    whose `context` is the text token states and whose `pooled` is the pooled
    text vector, and returns NHWC velocity. The latent grid may be any even
    rectangle the position buffer covers; the buffer is cropped centred on it,
    the way the source crops.
    """

    patch_size: int = 2
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 18
    heads: int = 18
    head_dim: int = 64
    joint_attention_dim: int = 4096
    caption_projection_dim: int = 1152
    pooled_projection_dim: int = 2048
    sample_size: int = 128
    pos_embed_max_size: int = 96
    dual_attention_layers: Sequence[int] = ()
    qk_norm: str | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str | None = "auto"

    @property
    def features(self) -> int:
        return self.heads * self.head_dim

    def position(self, height: int, width: int):
        """The stored position buffer cropped centred on this patch grid."""
        maximum = self.pos_embed_max_size
        buffer = self.variable(
            "buffers", "pos_embed",
            lambda: sincos_position(self.features, maximum,
                                    base_size=self.sample_size // self.patch_size))
        if height > maximum or width > maximum:
            raise ValueError(f"A {height}x{width} patch grid does not fit the position buffer's "
                             f"{maximum}x{maximum}")
        table = buffer.value.reshape(1, maximum, maximum, self.features)
        top, left = (maximum - height) // 2, (maximum - width) // 2
        cropped = table[:, top:top + height, left:left + width, :]
        return cropped.reshape(1, height * width, self.features)

    @nn.compact
    def __call__(self, x, time, conditioning: DenoisingCondition, train: bool = False):
        """`train` is the objective's standard call contract; the published
        transformer holds no dropout, so it changes nothing here."""
        if x.ndim != 4:
            raise ValueError(f"SD3 takes NHWC latents, got shape {x.shape}")
        patch = self.patch_size
        rows, columns = x.shape[1] // patch, x.shape[2] // patch
        if rows * patch != x.shape[1] or columns * patch != x.shape[2]:
            raise ValueError(f"A {x.shape[1]}x{x.shape[2]} latent is not a whole number of "
                             f"{patch}x{patch} patches")
        if conditioning.pooled is None:
            raise ValueError("SD3 conditioning needs the pooled text vector")
        image = nn.Conv(self.features, (patch, patch), strides=(patch, patch), padding="VALID",
                        dtype=self.dtype, precision=self.precision, name="pos_embed_proj")(x)
        image = image.reshape(image.shape[0], rows * columns, self.features)
        image = image + self.position(rows, columns).astype(image.dtype)

        times = sinusoidal_time(time, 256).astype(conditioning.pooled.dtype)
        timing = nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                          name="timestep_embedder_linear_1")(times)
        timing = nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                          name="timestep_embedder_linear_2")(nn.silu(timing))
        pooled = nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                          name="text_embedder_linear_1")(conditioning.pooled)
        pooled = nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                          name="text_embedder_linear_2")(nn.silu(pooled))
        conditioned = timing + pooled

        context = nn.Dense(self.caption_projection_dim, dtype=self.dtype, precision=self.precision,
                           name="context_embedder")(conditioning.context)
        dual = set(self.dual_attention_layers)
        for index in range(self.num_layers):
            image, context = SD3Block(
                self.features, self.heads, self.head_dim,
                context_pre_only=index == self.num_layers - 1, dual_attention=index in dual,
                qk_norm=self.qk_norm, dtype=self.dtype, precision=self.precision,
                attention_impl=self.attention_impl, name=f"transformer_blocks_{index}")(
                    image, context, conditioned)

        scale, shift = _Modulation(self.features, 2, dtype=self.dtype, precision=self.precision,
                                   name="norm_out")(conditioned)
        image = _modulate(_layer_norm(self.dtype)(image), shift, scale)
        image = nn.Dense(patch * patch * self.out_channels, dtype=self.dtype,
                         precision=self.precision, name="proj_out")(image)
        image = image.reshape(image.shape[0], rows, columns, patch, patch, self.out_channels)
        image = image.transpose(0, 1, 3, 2, 4, 5)
        return image.reshape(image.shape[0], rows * patch, columns * patch, self.out_channels)


__all__ = ["SD3Transformer", "SD3Block", "sincos_position"]
