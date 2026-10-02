"""FLUX.2's transformer, as Diffusers 0.40.0's `Flux2Transformer2DModel` runs it.

Flux's double-stream blocks (image and text meet only in attention) then
single-stream blocks, with these changes: no projection carries a bias and
the feed-forwards are SwiGLU; one modulation per call from the time (and
guidance) embedding, one set each for the double blocks' two streams and
the single blocks; a single block's qkv and feed-forward input are one map
and its attention output and gated state one map back; the rotary table has
four axes (time, row, column, token) of 32 channels at theta 2000, a text
token at (0, 0, 0, i) and an image position at (0, row, column, 0); and the
text is the Mistral-3 encoder's stacked hidden states, with no pooled
vector. The latent is the pipeline's 32 VAE channels folded 2x2 into 128,
which `dew.nn.autoencoders.flux2` produces.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm, scaled_dot_product_attention
from dew.nn.backbones.unet_condition import sinusoidal_time
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import logical_axes
from dew.registry import models

from .joint import (
    DoubleStreamBlock,
    Modulation,
    apply_rotary,
    embedding,
    guided_time,
    layer_norm,
    modulate,
    rotary_table,
)

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition


def flux2_positions(rows: int, columns: int, text: int) -> np.ndarray:
    """The ids `Flux2Pipeline` lays out, text first: token i at (0, 0, 0, i),
    then the latent grid row-major at (0, row, column, 0)."""
    positions = np.zeros((text + rows * columns, 4), dtype=np.float32)
    positions[:text, 3] = np.arange(text)
    grid = np.indices((rows, columns), dtype=np.float32).reshape(2, -1)
    positions[text:, 1], positions[text:, 2] = grid[0], grid[1]
    return positions


@logical_axes({("to_qkv_mlp_proj",): ("embed", None), ("proj_fused",): (None, "embed")})
class Flux2SingleBlock(nn.Module):
    """One `Flux2SingleTransformerBlock`: the joined sequence's queries, keys,
    values and SwiGLU input from one map, per-head RMS norms and the rotary
    on the queries and keys, and the attention output beside the gated
    hidden state through one map back."""

    features: int
    heads: int
    head_dim: int
    hidden: int
    epsilon: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.compact
    def __call__(self, x, mods, rotation):
        shift, scale, gate = mods
        inner = self.heads * self.head_dim
        normalized = modulate(layer_norm(self.dtype, self.epsilon)(x), shift, scale)
        projected = nn.Dense(3 * inner + 2 * self.hidden, use_bias=False, dtype=self.dtype,
                             precision=self.precision, name="to_qkv_mlp_proj")(normalized)
        heads = (x.shape[0], x.shape[1], self.heads, self.head_dim)
        query, key, value = (
            part.reshape(heads) for part in jnp.split(projected[..., : 3 * inner], 3, axis=-1)
        )
        query = RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name="norm_q")(query)
        key = RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name="norm_k")(key)
        attended = scaled_dot_product_attention(
            apply_rotary(query, *rotation), apply_rotary(key, *rotation), value,
            implementation=self.attention_impl, precision=self.precision)
        gated, value_half = jnp.split(projected[..., 3 * inner:], 2, axis=-1)
        joined = jnp.concatenate(
            [attended.reshape(x.shape[0], x.shape[1], inner), nn.silu(gated) * value_half], axis=-1
        )
        # The source's `attn.to_out`, named as Flux's single block names its
        # fused output map.
        return x + gate[:, None] * nn.Dense(self.features, use_bias=False, dtype=self.dtype,
                                            precision=self.precision, name="proj_fused")(joined)


@models("flux2_transformer")
@logical_axes({("context_embedder",): (None, "embed"), ("x_embedder",): (None, "embed"),
               ("proj_out",): ("embed", None),
               ("timestep_embedder_linear_1",): (None, "embed"),
               ("timestep_embedder_linear_2",): (None, "embed"),
               ("guidance_embedder_linear_1",): (None, "embed"),
               ("guidance_embedder_linear_2",): (None, "embed")})
class Flux2Transformer(nn.Module):
    """Diffusers 0.40.0's `Flux2Transformer2DModel` over Dew's interface.

    `__call__` takes the pipeline's NHWC latent (the VAE's channels folded
    2x2), the model time the schedule supplies - the sigma times the training
    count, the product the source reaches by dividing its timestep by a
    thousand and multiplying it back - and a `DenoisingCondition` whose
    `context` is the stacked encoder states and whose `guidance` is the
    distilled guidance a guidance-embedded checkpoint reads.
    """

    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 8
    num_single_layers: int = 48
    heads: int = 48
    head_dim: int = 128
    joint_attention_dim: int = 15360
    timestep_guidance_channels: int = 256
    mlp_ratio: float = 3.0
    axes_dims_rope: Sequence[int] = (32, 32, 32, 32)
    rope_theta: float = 2000.0
    eps: float = 1e-6
    guidance_embeds: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @property
    def features(self) -> int:
        return self.heads * self.head_dim

    @nn.compact
    def __call__(self, x, time, conditioning: DenoisingCondition, train: bool = False):
        """`train` is the objective's standard call contract; the published
        transformer holds no dropout, so it changes nothing here."""
        if x.ndim != 4:
            raise ValueError(f"FLUX.2 takes NHWC latents, got shape {x.shape}")
        batch, rows, columns, _ = x.shape
        dense = {"use_bias": False, "dtype": self.dtype, "precision": self.precision}
        image = nn.Dense(self.features, name="x_embedder", **dense)(x.reshape(batch, rows * columns, -1))
        context = nn.Dense(self.features, name="context_embedder", **dense)(conditioning.context)

        # `Flux2TimestepGuidanceEmbeddings`: the time's sinusoids through two
        # bias-free maps, plus the guidance's where the checkpoint embeds it.
        def embed(values, name: str):
            features = sinusoidal_time(values, self.timestep_guidance_channels,
                                       dtype=at_least_fp32(self.dtype)).astype(image.dtype)
            return embedding(features, self.features, name, bias=False, dtype=self.dtype,
                             precision=self.precision)

        embedded = guided_time(embed, time, conditioning.guidance, guidance_embeds=self.guidance_embeds)

        def modulation(pieces: int, name: str):
            return Modulation(self.features, pieces, bias=False, dtype=self.dtype, precision=self.precision,
                              name=name)(embedded)

        mods = (modulation(6, "double_stream_modulation_img"), modulation(6, "double_stream_modulation_txt"))
        single_mods = modulation(3, "single_stream_modulation")
        tables = rotary_table(flux2_positions(rows, columns, context.shape[1]), self.axes_dims_rope,
                              theta=self.rope_theta)
        rotation = tuple(jnp.asarray(table[None, :, None], image.dtype) for table in tables)
        hidden = int(self.features * self.mlp_ratio)
        block = {"epsilon": self.eps, "dtype": self.dtype, "precision": self.precision,
                 "attention_impl": self.attention_impl}
        for index in range(self.num_layers):
            image, context = DoubleStreamBlock(
                self.features, self.heads, self.head_dim, context_first=True, bias=False, mlp_hidden=hidden,
                shared_modulation=True, name=f"transformer_blocks_{index}", **block)(
                image, context, mods, rotation)
        joined = jnp.concatenate([context, image], axis=1)
        for index in range(self.num_single_layers):
            joined = Flux2SingleBlock(self.features, self.heads, self.head_dim, hidden,
                                      name=f"single_transformer_blocks_{index}", **block)(
                joined, single_mods, rotation)
        image = joined[:, context.shape[1]:]
        scale, shift = modulation(2, "norm_out")
        image = modulate(layer_norm(self.dtype, self.eps)(image), shift, scale)
        out = nn.Dense(self.out_channels, name="proj_out", **dense)(image)
        return out.reshape(batch, rows, columns, self.out_channels)


__all__ = ["Flux2SingleBlock", "Flux2Transformer", "flux2_positions"]
