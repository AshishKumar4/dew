"""Flux's transformer, as Diffusers 0.34.0's `FluxTransformer2DModel` runs it.

The published model reads a packed latent, in which its pipeline has already
folded 2x2 patches into the channel axis, one position per patch. Its stack
has two halves. The double-stream blocks keep the image and the text in
separate residual streams, each with its own modulation and feed-forward,
and join them only inside attention. The single-stream blocks concatenate
the two and run one stream whose attention and feed-forward share an output
projection. Every attention call rotates its queries and keys with a
three-axis rotary table over the text and image ids, in the interleaved-real
form Flux uses.

The double-stream blocks are `DoubleStreamBlock`s with the text first. This
module holds what only Flux does.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.activations import gelu_tanh
from dew.nn.blocks import Modulation, sinusoidal_time
from dew.nn.dense import Dense
from dew.nn.flops import MissingFlopsCount, matching_lengths, patch_lengths, projection_flops
from dew.nn.precision import at_least_fp32, float32_dot_general
from dew.nn.rope import axis_tables
from dew.nn.scan_orders import pixel_shuffle, pixel_unshuffle
from dew.nn.sharding import logical_axes

from .joint import (
    TIME_FEATURES,
    DoubleStreamBlock,
    JointAttention,
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


def flux_positions(rows: int, columns: int, text: int) -> np.ndarray:
    """Return the position ids a Flux pipeline lays out.

    The text sits at the origin, followed by one position per packed patch,
    indexed by its row and column.
    """
    positions = np.zeros((text + rows * columns, 3), dtype=np.float32)
    grid = np.indices((rows, columns), dtype=np.float32).reshape(2, -1)
    positions[text:, 1], positions[text:, 2] = grid[0], grid[1]
    return positions


@logical_axes({("proj_mlp",): ("embed", "mlp"), ("proj_fused",): (None, "embed")})
class FluxSingleBlock(nn.Module):
    """Runs one single-stream block: attention and a feed-forward over the joined sequence.

    Their outputs are concatenated and projected back together.
    """

    features: int
    heads: int
    head_dim: int
    mlp_ratio: float = 4.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.nowrap
    def projections(self) -> dict[str, tuple[int, int]]:
        hidden, inner = int(self.features * self.mlp_ratio), self.heads * self.head_dim
        return {'proj_mlp': (self.features, hidden), 'proj_fused': (inner + hidden, self.features)}

    @nn.nowrap
    def _modulation(self):
        return Modulation(self.features, 3, dtype=self.dtype, precision=self.precision, name="norm")

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """QKV, the MLP input, the joined output and per-row modulation."""
        length, = matching_lengths(lengths)
        attention = JointAttention(self.heads, self.head_dim, project=False, parent=None)
        return (sum(projection_flops(lengths, *shape) for shape in self.projections().values())
                + self._modulation().model_flops(np.ones(length.shape, np.int64))
                + attention.stream_flops(lengths, None, self.features))

    @nn.compact
    def __call__(self, x, conditioning, rotation):
        shift, scale, gate = self._modulation()(conditioning)
        normalized = modulate(layer_norm(self.dtype)(x), shift, scale)
        maps = self.projections()
        hidden = gelu_tanh(Dense(maps['proj_mlp'][1], dtype=self.dtype,
                               precision=self.precision, name="proj_mlp",
                               dot_general=float32_dot_general)(normalized))
        attended, _ = JointAttention(
            self.heads, self.head_dim, project=False, dtype=self.dtype, precision=self.precision,
            attention_impl=self.attention_impl, name="attn")(normalized, rotation=rotation)
        joined = jnp.concatenate([attended, hidden], axis=-1)
        # The source calls this `proj_out` inside its own block; the tree
        # names it apart from the model's own output projection, whose two
        # sides are the other way round.
        return x + gate[:, None] * Dense(maps['proj_fused'][1], dtype=self.dtype,
                                         precision=self.precision, name="proj_fused",
                                         dot_general=float32_dot_general)(joined)


@logical_axes({("context_embedder",): (None, "embed"), ("x_embedder",): (None, "embed"),
               ("proj_out",): ("embed", None),
               **embedding_axes("timestep_embedder", "guidance_embedder", "text_embedder")})
class FluxTransformer(TextStream, nn.Module):
    """Runs Diffusers 0.34.0's `FluxTransformer2DModel` behind Dew's model interface.

    `__call__` takes NHWC latents, the model time the schedule supplies, and
    a `DenoisingCondition`. The model time is the sigma times the training
    count; the source reaches the same product by dividing its timestep by a
    thousand and multiplying it back. The condition's `context` is the T5
    token states, its `pooled` is the CLIP pooled vector, and its `guidance`
    is the distilled guidance value that a guidance-embedded checkpoint
    reads. The model does the 2x2 packing that the source's pipeline does,
    so a caller works in latents and the position ids follow the latent grid.
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
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        return patch_lengths(sample_shape, (2, 2), rows)

    @nn.nowrap
    def multisequence_flops(self, lengths, context_lengths, *, input_channels: int,
                            text_features: int, pooled_features: int | None = None) -> float:
        """Every packed image/text map, joint block, time/pooled map and output."""
        if pooled_features is None:
            raise MissingFlopsCount("FluxTransformer needs the actual pooled conditioning width")
        image, context = matching_lengths(lengths, context_lengths)
        rows = np.ones(image.shape, np.int64)
        count = (projection_flops(lengths, 4 * input_channels, self.features)
                 + projection_flops(context_lengths, text_features, self.features))
        time = embedding_flops(rows, TIME_FEATURES, self.features)
        count += (1 + self.guidance_embeds) * time
        count += embedding_flops(rows, pooled_features, self.features)
        count += self.num_layers * self._double(parent=None).multisequence_flops(lengths, context_lengths)
        joined = (image + context).astype(np.int64)
        count += self.num_single_layers * self._single(parent=None).model_flops(joined)
        return (count + self._output_modulation().model_flops(rows)
                + projection_flops(lengths, self.features, self.patch_size**2 * self.out_channels))

    def _conditioning(self, time, guidance, pooled):
        """`CombinedTimestepGuidanceTextProjEmbeddings`: the time, the
        distilled guidance where the checkpoint embeds it, and the pooled text.

        The model time arrives in the schedule's own units, which are the
        sigmas times the training count: the source's pipeline divides those
        by a thousand and its transformer multiplies them back, so the
        product is what both embed.
        """
        dense = {"dtype": self.dtype, "precision": self.precision}

        def embed(values, name: str):
            features = sinusoidal_time(values, TIME_FEATURES, dtype=at_least_fp32(self.dtype))
            features = features.astype(pooled.dtype)
            return embedding(self, features, self.features, name, **dense)

        return (guided_time(embed, time, guidance, guidance_embeds=self.guidance_embeds)
                + embedding(self, pooled, self.features, "text_embedder", **dense))

    @nn.nowrap
    def _double(self, **where) -> DoubleStreamBlock:
        """A joint block as the model runs it, named by `where` (or unbound, `parent=None`, where a FLOP
        count reads it): one factory, so the count and the execution build the same block."""
        return DoubleStreamBlock(self.features, self.heads, self.head_dim, context_first=True,
                                 attention_impl=self.attention_impl, dtype=self.dtype,
                                 precision=self.precision, **where)

    @nn.nowrap
    def _single(self, **where) -> FluxSingleBlock:
        """A single-stream block as the model runs it (`_double`)."""
        return FluxSingleBlock(self.features, self.heads, self.head_dim, attention_impl=self.attention_impl,
                               dtype=self.dtype, precision=self.precision, **where)

    @nn.nowrap
    def _output_modulation(self):
        return Modulation(self.features, 2, dtype=self.dtype, precision=self.precision, name="norm_out")

    @nn.compact
    def __call__(self, x, time, conditioning: DenoisingCondition, train: bool = False):
        """Return the flow for the latents `x` at `time` under `conditioning`.

        `train` is part of the objective's standard call; the published
        transformer has no dropout, so it changes nothing here. Raises
        `ValueError` for latents that are not NHWC or do not pack into 2x2
        patches, and for a condition without `pooled`.
        """
        if x.ndim != 4:
            raise ValueError(f"Flux takes NHWC latents, got shape {x.shape}")
        rows, columns = x.shape[1] // 2, x.shape[2] // 2
        if rows * 2 != x.shape[1] or columns * 2 != x.shape[2]:
            raise ValueError(f"A {x.shape[1]}x{x.shape[2]} latent does not pack into 2x2 patches")
        if conditioning.pooled is None:
            raise ValueError("Flux conditioning needs the pooled text vector")
        dense = {"dtype": self.dtype, "precision": self.precision}
        # `_prepare_latents`' packing: one position per 2x2 patch, channel-major.
        packed = pixel_unshuffle(x).reshape(x.shape[0], rows * columns, 4 * x.shape[3])
        image = Dense(self.features, name="x_embedder", **dense, dot_general=float32_dot_general)(packed)
        conditioned = self._conditioning(time, conditioning.guidance, conditioning.pooled)
        context = Dense(self.features, name="context_embedder", **dense,
                        dot_general=float32_dot_general)(conditioning.context)
        # `FluxPosEmbed` computes its angles in float64 from the static ids.
        tables = axis_tables(flux_positions(rows, columns, context.shape[1]), self.axes_dims_rope, 10000.0,
                             dtype=np.float64)
        rotation = tuple(jnp.asarray(table[None], at_least_fp32(image.dtype)) for table in tables)
        for index in range(self.num_layers):
            image, context = self._double(name=f"transformer_blocks_{index}")(image, context, conditioned,
                                                                              rotation)
        joined = jnp.concatenate([context, image], axis=1)
        for index in range(self.num_single_layers):
            joined = self._single(name=f"single_transformer_blocks_{index}")(joined, conditioned, rotation)
        image = joined[:, context.shape[1]:]
        scale, shift = self._output_modulation()(conditioned)
        image = modulate(layer_norm(self.dtype)(image), shift, scale)
        packed = Dense(self.patch_size ** 2 * self.out_channels, name="proj_out", **dense,
                       dot_general=float32_dot_general)(image)
        return pixel_shuffle(packed.reshape(packed.shape[0], rows, columns, -1))


__all__ = ["FluxSingleBlock", "FluxTransformer", "flux_positions"]
