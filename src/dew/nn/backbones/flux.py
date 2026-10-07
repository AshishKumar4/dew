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

from dew.nn.backbones.unet_condition import sinusoidal_time
from dew.nn.precision import at_least_fp32
from dew.nn.scan_orders import pixel_shuffle, pixel_unshuffle
from dew.nn.sharding import logical_axes

from .joint import (
    DoubleStreamBlock,
    JointAttention,
    Modulation,
    embedding,
    guided_time,
    layer_norm,
    modulate,
    rotary_table,
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

    @nn.compact
    def __call__(self, x, conditioning, rotation):
        shift, scale, gate = Modulation(self.features, 3, dtype=self.dtype,
                                        precision=self.precision, name="norm")(conditioning)
        normalized = modulate(layer_norm(self.dtype)(x), shift, scale)
        hidden = nn.gelu(nn.Dense(int(self.features * self.mlp_ratio), dtype=self.dtype,
                                  precision=self.precision, name="proj_mlp")(normalized),
                         approximate=True)
        attended, _ = JointAttention(
            self.heads, self.head_dim, project=False, dtype=self.dtype, precision=self.precision,
            attention_impl=self.attention_impl, name="attn")(normalized, rotation=rotation)
        joined = jnp.concatenate([attended, hidden], axis=-1)
        # The source calls this `proj_out` inside its own block; the tree
        # names it apart from the model's own output projection, whose two
        # sides are the other way round.
        return x + gate[:, None] * nn.Dense(self.features, dtype=self.dtype,
                                            precision=self.precision, name="proj_fused")(joined)


@logical_axes({("context_embedder",): (None, "embed"), ("x_embedder",): (None, "embed"),
               ("proj_out",): ("embed", None),
               ("timestep_embedder_linear_1",): (None, "embed"),
               ("timestep_embedder_linear_2",): (None, "embed"),
               ("guidance_embedder_linear_1",): (None, "embed"),
               ("guidance_embedder_linear_2",): (None, "embed"),
               ("text_embedder_linear_1",): (None, "embed"),
               ("text_embedder_linear_2",): (None, "embed")})
class FluxTransformer(nn.Module):
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

    @property
    def features(self) -> int:
        return self.heads * self.head_dim

    @property
    def text_keyword(self) -> str:
        """Every call takes the text as `conditioning`, run beside the image in the double-stream blocks."""
        return "conditioning"

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
            features = sinusoidal_time(values, 256, dtype=at_least_fp32(self.dtype)).astype(pooled.dtype)
            return embedding(features, self.features, name, **dense)

        return (guided_time(embed, time, guidance, guidance_embeds=self.guidance_embeds)
                + embedding(pooled, self.features, "text_embedder", **dense))

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
        image = nn.Dense(self.features, name="x_embedder", **dense)(packed)
        conditioned = self._conditioning(time, conditioning.guidance, conditioning.pooled)
        context = nn.Dense(self.features, name="context_embedder", **dense)(conditioning.context)
        tables = rotary_table(flux_positions(rows, columns, context.shape[1]), self.axes_dims_rope)
        rotation = tuple(jnp.asarray(table[None, :, None], image.dtype) for table in tables)
        block = {"attention_impl": self.attention_impl, **dense}
        for index in range(self.num_layers):
            image, context = DoubleStreamBlock(
                self.features, self.heads, self.head_dim, context_first=True,
                name=f"transformer_blocks_{index}", **block)(image, context, conditioned, rotation)
        joined = jnp.concatenate([context, image], axis=1)
        for index in range(self.num_single_layers):
            joined = FluxSingleBlock(self.features, self.heads, self.head_dim,
                                     name=f"single_transformer_blocks_{index}", **block)(
                joined, conditioned, rotation)
        image = joined[:, context.shape[1]:]
        scale, shift = Modulation(self.features, 2, name="norm_out", **dense)(conditioned)
        image = modulate(layer_norm(self.dtype)(image), shift, scale)
        packed = nn.Dense(self.patch_size ** 2 * self.out_channels, name="proj_out", **dense)(image)
        return pixel_shuffle(packed.reshape(packed.shape[0], rows, columns, -1))


__all__ = ["FluxSingleBlock", "FluxTransformer", "flux_positions"]
