"""Flux's transformer, as Diffusers 0.34.0's `FluxTransformer2DModel` runs it.

The published model reads a packed latent: 2x2 patches already folded into
the channel axis by its pipeline, one position per patch. Its stack has two
halves. The double-stream blocks carry the image and the text in separate
residual streams with their own modulation and feed-forwards, joined only
inside attention; the single-stream blocks concatenate the two and run one
stream whose attention and feed-forward share a projection. Every attention
call rotates its queries and keys with a three-axis rotary table over the
text and image ids, in the interleaved-real form Flux uses.

`SD3Transformer` and this module share the modulation, the feed-forward and
the layer norm the MM-DiT family uses; what differs is here.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm, scaled_dot_product_attention
from dew.nn.backbones.unet_condition import DenoisingCondition, sinusoidal_time
from dew.nn.sharding import logical_axes
from dew.registry import models

from .sd3 import _FeedForward, _Modulation, _layer_norm, _modulate


def rotary_table(positions: np.ndarray, axes: Sequence[int], *, theta: float = 10000.0,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """`FluxPosEmbed` over ids: one angle per channel pair, per axis.

    The source computes its frequencies and their cosines in float64 from a
    static id grid, so this is the same host computation, and the pair of
    channels that shares an angle is adjacent rather than half a width apart.
    """
    cosines, sines = [], []
    for index, dim in enumerate(axes):
        frequencies = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64)[: dim // 2] / dim))
        angles = np.outer(positions[:, index].astype(np.float64), frequencies)
        cosines.append(np.repeat(np.cos(angles), 2, axis=1))
        sines.append(np.repeat(np.sin(angles), 2, axis=1))
    return (np.concatenate(cosines, axis=1).astype(np.float32),
            np.concatenate(sines, axis=1).astype(np.float32))


def flux_positions(rows: int, columns: int, text: int) -> np.ndarray:
    """The ids a Flux pipeline lays out: the text at the origin, then one
    position per packed patch, indexed by its row and column."""
    positions = np.zeros((text + rows * columns, 3), dtype=np.float32)
    grid = np.indices((rows, columns), dtype=np.float32).reshape(2, -1)
    positions[text:, 1], positions[text:, 2] = grid[0], grid[1]
    return positions


def pack(x: jax.Array) -> jax.Array:
    """`_prepare_latents`' packing: one position per 2x2 patch, whose channels
    run channel-major over the patch's own two rows and columns."""
    rows, columns, channels = x.shape[1] // 2, x.shape[2] // 2, x.shape[3]
    grouped = x.reshape(x.shape[0], rows, 2, columns, 2, channels)
    return grouped.transpose(0, 1, 3, 5, 2, 4).reshape(x.shape[0], rows * columns, channels * 4)


def unpack(x: jax.Array, rows: int, columns: int) -> jax.Array:
    """`_unpack_latents`, back to NHWC."""
    channels = x.shape[-1] // 4
    grouped = x.reshape(x.shape[0], rows, columns, channels, 2, 2)
    return grouped.transpose(0, 1, 4, 2, 5, 3).reshape(x.shape[0], rows * 2, columns * 2, channels)


def apply_rotary(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """The rotation `apply_rotary_emb` applies with `use_real_unbind_dim=-1`:
    adjacent channels are one complex pair, so the rotated copy is
    `(-x1, x0)` within each pair."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    rotated = jnp.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)
    return x * cos + rotated * sin


@logical_axes({("attn", name): ("embed", None) for name in
               ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")}
              | {("attn", "to_out_0"): (None, "embed"),
                 ("attn", "to_add_out"): (None, "embed")})
class _FluxAttention(nn.Module):
    """Flux's `Attention` under `FluxAttnProcessor2_0`.

    With a context it is the double-stream form: both streams project their
    own queries, keys and values, the text's lead the image's in one
    sequence - the opposite order to SD3's - and each stream takes its own
    output projection. Without one it is the single-stream form, which
    projects the already-joined sequence and returns it unprojected for the
    block's own fused output. Queries and keys are RMS normalized per head
    and then rotated, so the rotation is what attention sees.
    """

    heads: int
    head_dim: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str | None = "auto"

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim)

    def _projection(self, name: str, x):
        return self._heads(nn.Dense(self.heads * self.head_dim, dtype=self.dtype,
                                    precision=self.precision, name=name)(x))

    @nn.compact
    def __call__(self, image, context, cos, sin):
        inner = self.heads * self.head_dim
        query = RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm_q")(
            self._projection("to_q", image))
        key = RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm_k")(
            self._projection("to_k", image))
        value = self._projection("to_v", image)
        if context is not None:
            text_query = RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm_added_q")(
                self._projection("add_q_proj", context))
            text_key = RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm_added_k")(
                self._projection("add_k_proj", context))
            query = jnp.concatenate([text_query, query], axis=1)
            key = jnp.concatenate([text_key, key], axis=1)
            value = jnp.concatenate([self._projection("add_v_proj", context), value], axis=1)
        rotation = (cos[None, :, None, :], sin[None, :, None, :])
        query, key = apply_rotary(query, *rotation), apply_rotary(key, *rotation)
        attended = scaled_dot_product_attention(
            query, key, value, implementation=self.attention_impl, precision=self.precision)
        attended = attended.reshape(attended.shape[0], attended.shape[1], inner)
        if context is None:
            return attended, None
        tokens = context.shape[1]
        return (nn.Dense(inner, dtype=self.dtype, precision=self.precision,
                         name="to_out_0")(attended[:, tokens:]),
                nn.Dense(inner, dtype=self.dtype, precision=self.precision,
                         name="to_add_out")(attended[:, :tokens]))


class FluxBlock(nn.Module):
    """One double-stream block: two modulated residual streams, joined in
    attention and separate through their feed-forwards."""

    features: int
    heads: int
    head_dim: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str | None = "auto"

    @nn.compact
    def __call__(self, image, context, conditioning, cos, sin):
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = _Modulation(
            self.features, 6, dtype=self.dtype, precision=self.precision, name="norm1")(
                conditioning)
        (text_shift, text_scale, text_gate, text_shift_mlp, text_scale_mlp,
         text_gate_mlp) = _Modulation(self.features, 6, dtype=self.dtype,
                                      precision=self.precision, name="norm1_context")(conditioning)
        attended, text_attended = _FluxAttention(
            self.heads, self.head_dim, dtype=self.dtype, precision=self.precision,
            attention_impl=self.attention_impl, name="attn")(
                _modulate(_layer_norm(self.dtype)(image), shift, scale),
                _modulate(_layer_norm(self.dtype)(context), text_shift, text_scale), cos, sin)
        # A double-stream block's attention returns both streams.
        assert text_attended is not None
        image = image + gate[:, None] * attended
        image = image + gate_mlp[:, None] * _FeedForward(
            self.features, dtype=self.dtype, precision=self.precision, name="ff")(
                _modulate(_layer_norm(self.dtype)(image), shift_mlp, scale_mlp))
        context = context + text_gate[:, None] * text_attended
        context = context + text_gate_mlp[:, None] * _FeedForward(
            self.features, dtype=self.dtype, precision=self.precision, name="ff_context")(
                _modulate(_layer_norm(self.dtype)(context), text_shift_mlp, text_scale_mlp))
        return image, context


@logical_axes({("proj_mlp",): ("embed", "mlp"), ("proj_fused",): (None, "embed")})
class FluxSingleBlock(nn.Module):
    """One single-stream block: attention and a feed-forward over the joined
    sequence, whose outputs are concatenated and projected together."""

    features: int
    heads: int
    head_dim: int
    mlp_ratio: float = 4.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str | None = "auto"

    @nn.compact
    def __call__(self, x, conditioning, cos, sin):
        shift, scale, gate = _Modulation(self.features, 3, dtype=self.dtype,
                                         precision=self.precision, name="norm")(conditioning)
        normalized = _modulate(_layer_norm(self.dtype)(x), shift, scale)
        hidden = nn.gelu(nn.Dense(int(self.features * self.mlp_ratio), dtype=self.dtype,
                                  precision=self.precision, name="proj_mlp")(normalized),
                         approximate=True)
        attended, _ = _FluxAttention(
            self.heads, self.head_dim, dtype=self.dtype, precision=self.precision,
            attention_impl=self.attention_impl, name="attn")(normalized, None, cos, sin)
        joined = jnp.concatenate([attended, hidden], axis=-1)
        # The source calls this `proj_out` inside its own block; the tree
        # names it apart from the model's own output projection, whose two
        # sides are the other way round.
        return x + gate[:, None] * nn.Dense(self.features, dtype=self.dtype,
                                            precision=self.precision, name="proj_fused")(joined)


@models("flux_transformer")
@logical_axes({("context_embedder",): (None, "embed"), ("x_embedder",): (None, "embed"),
               ("proj_out",): ("embed", None),
               ("timestep_embedder_linear_1",): (None, "embed"),
               ("timestep_embedder_linear_2",): (None, "embed"),
               ("guidance_embedder_linear_1",): (None, "embed"),
               ("guidance_embedder_linear_2",): (None, "embed"),
               ("text_embedder_linear_1",): (None, "embed"),
               ("text_embedder_linear_2",): (None, "embed")})
class FluxTransformer(nn.Module):
    """Diffusers 0.34.0's `FluxTransformer2DModel` over Dew's interface.

    `__call__` takes NHWC latents, the model time the schedule supplies -
    the sigma times the training count, which is the product the source
    reaches by dividing its timestep and multiplying it back - and a
    `DenoisingCondition` whose `context` is the T5
    token states, whose `pooled` is the CLIP pooled vector and whose
    `guidance` is the distilled guidance value a guidance-embedded checkpoint
    reads. The 2x2 packing its pipeline performs is here, so a caller works
    in latents and the position ids follow the latent grid.
    """

    patch_size: int = 1
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 19
    num_single_layers: int = 38
    heads: int = 24
    head_dim: int = 128
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = False
    axes_dims_rope: Sequence[int] = (16, 56, 56)
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str | None = "auto"

    @property
    def features(self) -> int:
        return self.heads * self.head_dim

    def _conditioning(self, time, guidance, pooled):
        """`CombinedTimestepTextProjEmbeddings`, with the guidance embedder a
        distilled checkpoint adds.

        The model time arrives in the schedule's own units, which are the
        sigmas times the training count: the source's pipeline divides those
        by a thousand and its transformer multiplies them back, so the
        product is what both embed. The distilled guidance is a scalar rather
        than a time, and the source scales it by a thousand here.
        """
        def embedder(values, name: str):
            features = sinusoidal_time(values, 256).astype(pooled.dtype)
            hidden = nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                              name=f"{name}_linear_1")(features)
            return nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                            name=f"{name}_linear_2")(nn.silu(hidden))

        conditioning = embedder(time, "timestep_embedder")
        if self.guidance_embeds:
            if guidance is None:
                raise ValueError("This checkpoint embeds its guidance; conditioning needs it")
            conditioning = conditioning + embedder(guidance * 1000.0, "guidance_embedder")
        elif guidance is not None:
            raise ValueError("This checkpoint has no guidance embedder")
        projected = nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                             name="text_embedder_linear_1")(pooled)
        return conditioning + nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                                       name="text_embedder_linear_2")(nn.silu(projected))

    @nn.compact
    def __call__(self, x, time, conditioning: DenoisingCondition, train: bool = False):
        """`train` is the objective's standard call contract; the published
        transformer holds no dropout, so it changes nothing here."""
        if x.ndim != 4:
            raise ValueError(f"Flux takes NHWC latents, got shape {x.shape}")
        rows, columns = x.shape[1] // 2, x.shape[2] // 2
        if rows * 2 != x.shape[1] or columns * 2 != x.shape[2]:
            raise ValueError(f"A {x.shape[1]}x{x.shape[2]} latent does not pack into 2x2 patches")
        if conditioning.pooled is None:
            raise ValueError("Flux conditioning needs the pooled text vector")
        image = nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                         name="x_embedder")(pack(x))
        conditioned = self._conditioning(time, conditioning.guidance, conditioning.pooled)
        context = nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                           name="context_embedder")(conditioning.context)
        cosines, sines = rotary_table(flux_positions(rows, columns, context.shape[1]),
                                      self.axes_dims_rope)
        cos, sin = jnp.asarray(cosines, image.dtype), jnp.asarray(sines, image.dtype)
        for index in range(self.num_layers):
            image, context = FluxBlock(
                self.features, self.heads, self.head_dim, dtype=self.dtype,
                precision=self.precision, attention_impl=self.attention_impl,
                name=f"transformer_blocks_{index}")(image, context, conditioned, cos, sin)
        joined = jnp.concatenate([context, image], axis=1)
        for index in range(self.num_single_layers):
            joined = FluxSingleBlock(
                self.features, self.heads, self.head_dim, dtype=self.dtype,
                precision=self.precision, attention_impl=self.attention_impl,
                name=f"single_transformer_blocks_{index}")(joined, conditioned, cos, sin)
        image = joined[:, context.shape[1]:]
        scale, shift = _Modulation(self.features, 2, dtype=self.dtype, precision=self.precision,
                                   name="norm_out")(conditioned)
        image = _modulate(_layer_norm(self.dtype)(image), shift, scale)
        packed = nn.Dense(self.patch_size ** 2 * self.out_channels, dtype=self.dtype,
                          precision=self.precision, name="proj_out")(image)
        return unpack(packed, rows, columns)


__all__ = ["FluxTransformer", "FluxBlock", "FluxSingleBlock", "rotary_table", "flux_positions",
           "apply_rotary", "pack", "unpack"]
