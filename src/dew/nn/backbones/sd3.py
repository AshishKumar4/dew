"""Stable Diffusion 3's own MM-DiT, as the published transformer computes it.

This module has the arithmetic of Diffusers 0.34.0's `SD3Transformer2DModel`
where it differs from `SimpleMMDiT`: the joint attention with the image
first and no rotary, the centred crop of the position buffer, the summed
timestep and pooled-text embedders, the last block's context-only norm, and
SD3.5's ninefold modulation with a second self-attention. The blocks are
`DoubleStreamBlock`s with the image first. The interface is
Dew's: NHWC latents, a model time and a `DenoisingCondition` with the text
tokens and pooled vector go in, and NHWC velocity comes out. The source's
sin/cos position buffer is stored in the `buffers` collection, so no
optimizer sees it, and the model reads and exports the checkpoint's stored
values.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.blocks import Modulation, sinusoidal_time
from dew.nn.conv import Conv
from dew.nn.dense import Dense
from dew.nn.flops import MissingFlopsCount, matching_lengths, patch_lengths, projection_flops
from dew.nn.precision import at_least_fp32, float32_dot_general
from dew.nn.rope import sincos_grid, sinusoid_frequencies
from dew.nn.scan_orders import unpatchify
from dew.nn.sharding import logical_axes

from .joint import (
    TIME_FEATURES,
    DoubleStreamBlock,
    TextStream,
    embedding,
    embedding_axes,
    embedding_flops,
    layer_norm,
    modulate,
)

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition


def sincos_position(channels: int, grid: int, *, base_size: int):
    """Return `get_2d_sincos_pos_embed` at the grid the source builds its buffer on.

    The leading half of the channels encodes the column and the trailing half
    the row, each as sines then cosines over 10000^-(2i/half), with both axes
    divided by `grid / base_size`. This only initializes the buffer; a loaded
    checkpoint brings its stored buffer. Diffusers builds it as here: the steps in float32, the
    frequencies and the sinusoids in float64, rounded once to float32.
    """
    steps = np.arange(grid, dtype=np.float32) / np.float32(grid / base_size)
    columns, rows = np.meshgrid(steps, steps, indexing="xy")  # width goes first
    # `1 / 10000 ** (i / (half / 2))` over half the channels an axis (an odd half divides by a
    # half-integer, as the reference does).
    frequencies = sinusoid_frequencies(channels // 4, channels / 4, dtype=np.float64)
    return jnp.asarray(sincos_grid(frequencies, columns, rows, dtype=np.float32)[None])


@logical_axes({("context_embedder",): (None, "embed"), ("proj_out",): ("embed", None),
               **embedding_axes("timestep_embedder", "text_embedder")})
class SD3Transformer(TextStream, nn.Module):
    """Runs Diffusers 0.34.0's `SD3Transformer2DModel` behind Dew's model interface.

    `__call__` takes NHWC latents, the model time and a `DenoisingCondition`,
    and returns NHWC velocity. The condition's `context` is the text token
    states and its `pooled` is the pooled text vector. The latent grid may be
    any even rectangle that the position buffer covers; the buffer is cropped
    centred on it, the way the source crops it.
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
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        return patch_lengths(sample_shape, (self.patch_size, self.patch_size), rows)

    @nn.nowrap
    def multisequence_flops(self, lengths, context_lengths, *, input_channels: int,
                            text_features: int, pooled_features: int | None = None) -> float:
        """Patch/text/pooled/time maps and SD3.5's selected dual-attention layers."""
        if pooled_features is None:
            raise MissingFlopsCount("SD3Transformer needs the actual pooled conditioning width")
        image, _ = matching_lengths(lengths, context_lengths)
        rows = np.ones(image.shape, np.int64)
        count = (projection_flops(lengths, self.patch_size**2 * input_channels, self.features)
                 + projection_flops(context_lengths, text_features, self.caption_projection_dim)
                 + embedding_flops(rows, TIME_FEATURES, self.features)
                 + embedding_flops(rows, pooled_features, self.features))
        for index in range(self.num_layers):
            count += self._block(index, parent=None).multisequence_flops(lengths, context_lengths)
        return (count + self._output_modulation().model_flops(rows)
                + projection_flops(lengths, self.features, self.patch_size**2 * self.out_channels))

    def position(self, height: int, width: int):
        """Return the stored position buffer, cropped centred on a `height` by `width` patch grid.

        A grid larger than `pos_embed_max_size` on either side raises
        `ValueError`.
        """
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

    @nn.nowrap
    def _block(self, index: int, **where) -> DoubleStreamBlock:
        """Joint block `index` as the model runs it, named by `where` (or unbound, `parent=None`, where a
        FLOP count reads it): one factory, so the count and the execution build the same block."""
        return DoubleStreamBlock(
            self.features, self.heads, self.head_dim, qk_norm=self.qk_norm is not None,
            context_pre_only=index == self.num_layers - 1, dual_attention=index in self.dual_attention_layers,
            attention_impl=self.attention_impl, dtype=self.dtype, precision=self.precision, **where)

    @nn.nowrap
    def _output_modulation(self):
        return Modulation(self.features, 2, dtype=self.dtype, precision=self.precision, name="norm_out")

    @nn.compact
    def __call__(self, x, time, conditioning: DenoisingCondition, train: bool = False):
        """Return the velocity for the latents `x` at `time` under `conditioning`.

        `train` is part of the objective's standard call; the published
        transformer has no dropout, so it changes nothing here. Raises
        `ValueError` for latents that are not NHWC or not whole patches, for a
        condition without `pooled`, and for a `qk_norm` other than 'rms_norm'
        or None.
        """
        if x.ndim != 4:
            raise ValueError(f"SD3 takes NHWC latents, got shape {x.shape}")
        patch = self.patch_size
        rows, columns = x.shape[1] // patch, x.shape[2] // patch
        if rows * patch != x.shape[1] or columns * patch != x.shape[2]:
            raise ValueError(f"A {x.shape[1]}x{x.shape[2]} latent is not a whole number of "
                             f"{patch}x{patch} patches")
        if conditioning.pooled is None:
            raise ValueError("SD3 conditioning needs the pooled text vector")
        if self.qk_norm not in (None, "rms_norm"):
            raise ValueError(f"Native SD3 implements qk_norm 'rms_norm', not {self.qk_norm!r}")
        dense = {"dtype": self.dtype, "precision": self.precision}
        image = Conv(self.features, (patch, patch), strides=(patch, patch), padding="VALID",
                     name="pos_embed_proj", **dense)(x)
        image = image.reshape(image.shape[0], rows * columns, self.features)
        image = image + self.position(rows, columns).astype(image.dtype)

        times = sinusoidal_time(time, TIME_FEATURES, dtype=at_least_fp32(self.dtype))
        times = times.astype(conditioning.pooled.dtype)
        conditioned = (embedding(self, times, self.features, "timestep_embedder", **dense)
                       + embedding(self, conditioning.pooled, self.features, "text_embedder", **dense))

        context = Dense(self.caption_projection_dim, name="context_embedder", **dense,
                        dot_general=float32_dot_general)(
            conditioning.context)
        for index in range(self.num_layers):
            image, context = self._block(index, name=f"transformer_blocks_{index}")(
                image, context, conditioned)

        scale, shift = self._output_modulation()(conditioned)
        image = modulate(layer_norm(self.dtype)(image), shift, scale)
        image = Dense(patch * patch * self.out_channels, name="proj_out", **dense,
                      dot_general=float32_dot_general)(image)
        return unpatchify(image, patch, rows * patch, columns * patch, self.out_channels)


__all__ = ["SD3Transformer", "sincos_position"]
