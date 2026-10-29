"""FLUX.2's transformer, as Diffusers 0.40.0's `Flux2Transformer2DModel` runs it.

Like Flux, it runs double-stream blocks, where the image and the text meet
only in attention, and then single-stream blocks. It differs from Flux in
these ways:

- No projection has a bias, and the feed-forwards are SwiGLU.
- The time (and guidance) embedding gives one modulation per call: one set
  for each of the double blocks' two streams, and one for the single blocks.
- A single block computes its qkv and its feed-forward input with one map,
  and maps its attention output and gated state back with one map.
- The rotary table has four axes (time, row, column, token) of 32 channels
  at theta 2000. A text token sits at (0, 0, 0, i) and an image position at
  (0, row, column, 0).
- The text is the Mistral-3 encoder's stacked hidden states, with no pooled
  vector.

The latent is the pipeline's 32 VAE channels folded 2x2 into 128, which
`dew.nn.autoencoders.flux2` produces.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.activations import gated_product
from dew.nn.attention import AttentionImpl, scaled_dot_product_attention
from dew.nn.blocks import Modulation, sinusoidal_time
from dew.nn.dense import Dense
from dew.nn.flops import matching_lengths, patch_lengths, projection_flops, self_attention_flops
from dew.nn.norms import RMSNorm
from dew.nn.precision import at_least_fp32, float32_dot_general
from dew.nn.rope import axis_tables, rotate
from dew.nn.sharding import logical_axes

from .joint import (
    DoubleStreamBlock,
    TextStream,
    embedding,
    embedding_axes,
    embedding_flops,
    guided_time,
    layer_norm,
    modulate,
)

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition


def flux2_positions(rows: int, columns: int, text: int) -> np.ndarray:
    """Return the position ids `Flux2Pipeline` lays out, text first.

    Text token i sits at (0, 0, 0, i), followed by the latent grid in
    row-major order at (0, row, column, 0).
    """
    positions = np.zeros((text + rows * columns, 4), dtype=np.float32)
    positions[:text, 3] = np.arange(text)
    grid = np.indices((rows, columns), dtype=np.float32).reshape(2, -1)
    positions[text:, 1], positions[text:, 2] = grid[0], grid[1]
    return positions


@logical_axes({("to_qkv_mlp_proj",): ("embed", None), ("proj_fused",): (None, "embed")})
class Flux2SingleBlock(nn.Module):
    """Runs one `Flux2SingleTransformerBlock` over the joined sequence.

    One map gives the queries, keys, values and SwiGLU input. Per-head RMS
    norms and the rotary apply to the queries and keys. One map back takes
    the attention output together with the gated hidden state.
    """

    features: int
    heads: int
    head_dim: int
    hidden: int
    epsilon: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: AttentionImpl = "auto"

    @nn.nowrap
    def projections(self) -> dict[str, tuple[int, int]]:
        inner = self.heads * self.head_dim
        return {'to_qkv_mlp_proj': (self.features, 3 * inner + 2 * self.hidden),
                'proj_fused': (inner + self.hidden, self.features)}

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """The fused QKV/gate/up map and joined output; modulation is shared."""
        return self_attention_flops(lengths, self.projections(), self.heads, self.head_dim)

    @nn.compact
    def __call__(self, x, mods, rotation):
        shift, scale, gate = mods
        inner = self.heads * self.head_dim
        normalized = modulate(layer_norm(self.dtype, self.epsilon)(x), shift, scale)
        maps = self.projections()
        projected = Dense(maps['to_qkv_mlp_proj'][1], use_bias=False, dtype=self.dtype,
                          precision=self.precision, name="to_qkv_mlp_proj",
                          dot_general=float32_dot_general)(normalized)
        heads = (x.shape[0], x.shape[1], self.heads, self.head_dim)
        query, key, value = (
            part.reshape(heads) for part in jnp.split(projected[..., : 3 * inner], 3, axis=-1)
        )
        query = RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name="norm_q")(query)
        key = RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name="norm_k")(key)
        attended = scaled_dot_product_attention(
            rotate(query, *rotation, pairs="adjacent"), rotate(key, *rotation, pairs="adjacent"), value,
            implementation=self.attention_impl, precision=self.precision)
        gated, value_half = jnp.split(projected[..., 3 * inner:], 2, axis=-1)
        joined = jnp.concatenate(
            [attended.reshape(x.shape[0], x.shape[1], inner), gated_product('swiglu')(gated, value_half)],
            axis=-1)
        # The source's `attn.to_out`, named as Flux's single block names its
        # fused output map.
        return x + gate[:, None] * Dense(maps['proj_fused'][1], use_bias=False, dtype=self.dtype,
                                         precision=self.precision, name="proj_fused",
                                         dot_general=float32_dot_general)(joined)


@logical_axes({("context_embedder",): (None, "embed"), ("x_embedder",): (None, "embed"),
               ("proj_out",): ("embed", None), **embedding_axes("timestep_embedder", "guidance_embedder")})
class Flux2Transformer(TextStream, nn.Module):
    """Runs Diffusers 0.40.0's `Flux2Transformer2DModel` behind Dew's model interface.

    `__call__` takes the pipeline's NHWC latent (the VAE's channels folded
    2x2), the model time the schedule supplies, and a `DenoisingCondition`.
    The model time is the sigma times the training count; the source reaches
    the same product by dividing its timestep by a thousand and multiplying
    it back. The condition's `context` is the stacked encoder states, and its
    `guidance` is the distilled guidance that a guidance-embedded checkpoint
    reads.
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
    attention_impl: AttentionImpl = "auto"

    @property
    def modulation_pieces(self) -> dict[str, int]:
        return {'double_stream_modulation_img': 6, 'double_stream_modulation_txt': 6,
                'single_stream_modulation': 3, 'norm_out': 2}

    @nn.nowrap
    def _double(self, **where) -> DoubleStreamBlock:
        """A joint block as the model runs it, named by `where` (or unbound, `parent=None`, where a FLOP
        count reads it): one factory, so the count and the execution build the same block."""
        return DoubleStreamBlock(self.features, self.heads, self.head_dim, context_first=True, bias=False,
                                 mlp_hidden=int(self.features * self.mlp_ratio), swiglu=True,
                                 shared_modulation=True, **self._block_fields(), **where)

    @nn.nowrap
    def _single(self, **where) -> Flux2SingleBlock:
        """A single-stream block as the model runs it (`_double`)."""
        return Flux2SingleBlock(self.features, self.heads, self.head_dim, int(self.features * self.mlp_ratio),
                                **self._block_fields(), **where)

    @nn.nowrap
    def _block_fields(self) -> dict:
        return {"epsilon": self.eps, "dtype": self.dtype, "precision": self.precision,
                "attention_impl": self.attention_impl}

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        return patch_lengths(sample_shape, (1, 1), rows)

    @nn.nowrap
    def multisequence_flops(self, lengths, context_lengths, *, input_channels: int,
                            text_features: int, pooled_features: int | None = None) -> float:
        """Actual input widths and one shared modulation, not one per block."""
        image, context = matching_lengths(lengths, context_lengths)
        rows = np.ones(image.shape, np.int64)
        count = (projection_flops(lengths, input_channels, self.features)
                 + projection_flops(context_lengths, text_features, self.features))
        time = embedding_flops(rows, self.timestep_guidance_channels, self.features)
        count += (1 + self.guidance_embeds) * time
        count += sum(Modulation(self.features, pieces, parent=None).model_flops(rows)
                     for pieces in self.modulation_pieces.values())
        count += self.num_layers * self._double(parent=None).multisequence_flops(lengths, context_lengths)
        joined = (image + context).astype(np.int64)
        count += self.num_single_layers * self._single(parent=None).model_flops(joined)
        return count + projection_flops(lengths, self.features, self.out_channels)

    @nn.compact
    def __call__(self, x, time, conditioning: DenoisingCondition, train: bool = False):
        """Return the flow for the latents `x` at `time` under `conditioning`.

        `train` is part of the objective's standard call; the published
        transformer has no dropout, so it changes nothing here. Raises
        `ValueError` for latents that are not NHWC.
        """
        if x.ndim != 4:
            raise ValueError(f"FLUX.2 takes NHWC latents, got shape {x.shape}")
        batch, rows, columns, _ = x.shape
        dense = {"use_bias": False, "dtype": self.dtype, "precision": self.precision}
        image = Dense(self.features, name="x_embedder", **dense,
                      dot_general=float32_dot_general)(x.reshape(batch, rows * columns, -1))
        context = Dense(self.features, name="context_embedder", **dense,
                        dot_general=float32_dot_general)(conditioning.context)

        # `Flux2TimestepGuidanceEmbeddings`: the time's sinusoids through two
        # bias-free maps, plus the guidance's where the checkpoint embeds it.
        def embed(values, name: str):
            features = sinusoidal_time(values, self.timestep_guidance_channels,
                                       dtype=at_least_fp32(self.dtype)).astype(image.dtype)
            return embedding(self, features, self.features, name, bias=False, dtype=self.dtype,
                             precision=self.precision)

        embedded = guided_time(embed, time, conditioning.guidance, guidance_embeds=self.guidance_embeds)

        def modulation(name: str):
            return Modulation(self.features, self.modulation_pieces[name], bias=False,
                              dtype=self.dtype, precision=self.precision,
                              name=name)(embedded)

        mods = (modulation("double_stream_modulation_img"), modulation("double_stream_modulation_txt"))
        single_mods = modulation("single_stream_modulation")
        tables = axis_tables(flux2_positions(rows, columns, context.shape[1]), self.axes_dims_rope,
                             self.rope_theta, dtype=np.float64)
        rotation = tuple(jnp.asarray(table[None], at_least_fp32(image.dtype)) for table in tables)
        for index in range(self.num_layers):
            image, context = self._double(name=f"transformer_blocks_{index}")(image, context, mods, rotation)
        joined = jnp.concatenate([context, image], axis=1)
        for index in range(self.num_single_layers):
            joined = self._single(name=f"single_transformer_blocks_{index}")(joined, single_mods, rotation)
        image = joined[:, context.shape[1]:]
        scale, shift = modulation("norm_out")
        image = modulate(layer_norm(self.dtype, self.eps)(image), shift, scale)
        out = Dense(self.out_channels, name="proj_out", **dense, dot_general=float32_dot_general)(image)
        return out.reshape(batch, rows, columns, self.out_channels)


__all__ = ["Flux2SingleBlock", "Flux2Transformer", "flux2_positions"]
