"""Qwen-Image 2.1's transformer, as Diffusers' `QwenImage21Transformer2DModel`
runs it at commit 6256aa76.

One residual stream holds the image and then the text. The text is the
vision-language encoder's states passed through a zero-centred RMS norm and
a GELU MLP, and the image is one linear map per latent position. Every block
modulates both with scales and tanh gates from one shared projection of the
time embedding, attends, and runs a SwiGLU. Attention is block-causal: the
text is causal, and the image attends to all the text and to itself. Under
`causal_condition` the text is modulated from time zero, so its activations
do not change from one sampling step to the next. The rotary table's three
axes advance a text token on all three, and place the image at the frame
after the text, on a zero-centred height and width grid. The source starts
the image's frame after the call's longest prompt. Here each row starts
after its own text, which matches the source for a prompt alone or for a
batch of prompts of one length, and keeps each row independent of the rest
of its batch.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm, scaled_dot_product_attention
from dew.nn.blocks import sinusoidal_time
from dew.nn.precision import at_least_fp32
from dew.nn.rope import axis_tables
from dew.nn.sharding import logical_axes

from .decoder_block import GatedMLP
from .joint import JointAttention, embedding, layer_norm

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition


def _image_grid(rows: int, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """The height and width indices of one image block, row-major and
    centred on zero: `-(n - n // 2)` up to `n // 2 - 1` along each side."""
    heights = np.arange(-(rows - rows // 2), rows // 2)
    widths = np.arange(-(columns - columns // 2), columns // 2)
    return np.repeat(heights, columns), np.tile(widths, rows)


def _rotary_positions(lengths: jax.Array, text: int, rows: int, columns: int) -> jax.Array:
    """The rotary ids of every token, image first, `[B, rows * columns + text, 3]`.

    A text token sits at its index on all three axes. The image's frame axis
    is the row's own text length, and its other two are the centred grid.
    The angles are `QwenImage21Rope.rope_params`', theta 10000 in float32
    (`dew.nn.rope.axis_tables`), whose inverse frequencies at these widths
    are the source's float32 `pow` bit for bit.
    """
    heights, widths = _image_grid(rows, columns)
    index = jnp.arange(text)
    batch = lengths.shape[0]
    frame = jnp.concatenate([jnp.broadcast_to(lengths[:, None], (batch, rows * columns)),
                             jnp.broadcast_to(index, (batch, text))], axis=1)
    grid = jnp.concatenate([jnp.asarray(np.stack([heights, widths], axis=-1)),
                            jnp.stack([index, index], axis=-1)])
    return jnp.concatenate([frame[..., None], jnp.broadcast_to(grid, (batch, *grid.shape))], axis=-1)


def _dense(features: int, name: str, dtype, precision) -> nn.Dense:
    return nn.Dense(features, use_bias=False, dtype=dtype, precision=precision, name=name)


def _per_rows(x, values, image: int, apply):
    """`apply(rows, value)` over the image rows with the sampled value and
    over the text rows with the time-zero one, as `_select_modulation_rows`
    picks them. `values` is that pair, each `[B or 1, width]`."""
    still, sampled = values
    return jnp.concatenate([apply(x[:, :image], sampled[:, None]),
                            apply(x[:, image:], still[:, None])], axis=1)


def _scaled(x, scale):
    """The source's `x * (1 + scale)`."""
    return x * (1 + scale)


class _Attention(JointAttention):
    """`QwenImage21Attention` under the source's exact multi-pass prefill:
    image queries attend everything and text queries the text causally, padded
    text keys excluded. With the image first, each query's keys are a prefix,
    passed as lengths that cuDNN skips rather than as a dense mask; attention
    does not depend on key order, so this is the source's arithmetic summed in
    another order.
    """

    image_tokens: int = 0
    """The image's tokens, which lead the sequence."""

    def attend(self, query, key, value, lengths):
        attend = functools.partial(scaled_dot_product_attention,
                                   implementation=self.attention_impl, precision=self.precision)
        image = self.image_tokens
        return jnp.concatenate([
            attend(query[:, :image], key, value, key_value_seq_lengths=image + lengths),
            attend(query[:, image:], key[:, image:], value[:, image:], causal=True,
                   key_value_seq_lengths=lengths)], axis=1)


class _Block(nn.Module):
    """One single-stream block, holding no modulation of its own: the
    model's shared projection hands every block the same scales and gates."""

    features: int
    heads: int
    head_dim: int
    mlp_ratio: int = 3
    eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.compact
    def __call__(self, x, modulation, rotation, lengths, image: int):
        scale, gate, scale_mlp, gate_mlp = modulation
        attended, _ = _Attention(self.heads, self.head_dim, bias=False, epsilon=self.eps, image_tokens=image,
                                 dtype=self.dtype, precision=self.precision,
                                 attention_impl=self.attention_impl, name="attn")(
            _per_rows(layer_norm(self.dtype, self.eps)(x), scale, image, _scaled), rotation=rotation,
            lengths=lengths)
        x = x + _per_rows(attended, gate, image, lambda rows, value: jnp.tanh(value) * rows)
        hidden = GatedMLP(self.features * self.mlp_ratio, self.features, dtype=self.dtype,
                          precision=self.precision, name="img_mlp")(
            _per_rows(layer_norm(self.dtype, self.eps)(x), scale_mlp, image, _scaled))
        x = x + _per_rows(hidden, gate_mlp, image, lambda rows, value: jnp.tanh(value) * rows)
        if x.dtype == jnp.float16:
            x = jnp.clip(x, -65504, 65504)
        return x


@logical_axes({("img_in",): (None, "embed"), ("proj_out",): ("embed", None),
               ("modulation",): (None, "embed"), ("norm_out_linear",): (None, "embed"),
               ("txt_in", "in_layer"): (None, "embed"), ("txt_in", "out_layer"): (None, "embed"),
               ("timestep_embedder_linear_1",): (None, "embed"),
               ("timestep_embedder_linear_2",): (None, "embed")})
class QwenImageTransformer(nn.Module):
    """Runs `QwenImage21Transformer2DModel` behind Dew's model interface.

    `__call__` takes NHWC latents with one token per position, the model time
    the schedule supplies, and a `DenoisingCondition`, and returns the
    prediction at the image's tokens. The model time is the sigma times the
    training count; the source reaches the same product by dividing its
    timestep by a thousand and multiplying it back. The condition's `context`
    is the encoder's text states after the system prompt, right-padded, and
    its `mask` marks the real tokens. Each row's real tokens come first in
    its text, which is the layout that both the rotary positions and the
    attention's key lengths assume.
    """

    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    heads: int = 32
    head_dim: int = 128
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: Sequence[int] = (16, 56, 56)
    eps: float = 1e-6
    causal_condition: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @property
    def features(self) -> int:
        return self.heads * self.head_dim

    @property
    def text_keyword(self) -> str:
        """Every call takes the text as `conditioning`, which shares the image's residual stream."""
        return "conditioning"

    def _time(self, time):
        """`QwenImage21TimestepProjEmbeddings`: cosines first, then two
        bias-free linears with a SiLU between."""
        sinusoids = sinusoidal_time(time, 256, dtype=at_least_fp32(self.dtype))
        return embedding(sinusoids.astype(self.dtype or jnp.float32), self.features, "timestep_embedder",
                         bias=False, dtype=self.dtype, precision=self.precision)

    @nn.compact
    def __call__(self, x, time, conditioning: DenoisingCondition, train: bool = False):
        """Return the prediction for the latents `x` at `time` under `conditioning`.

        `train` is part of the objective's standard call; the published
        transformer has no dropout, so it changes nothing here. Raises
        `ValueError` for latents that are not NHWC.
        """
        if x.ndim != 4:
            raise ValueError(f"Qwen-Image takes NHWC latents, got shape {x.shape}")
        batch, rows, columns, _ = x.shape
        image_tokens = rows * columns
        text = conditioning.context.shape[1]
        valid = (jnp.ones((batch, text), bool) if conditioning.mask is None
                 else jnp.broadcast_to(jnp.asarray(conditioning.mask, bool), (batch, text)))
        lengths = jnp.sum(valid, axis=1, dtype=jnp.int32)
        image = _dense(self.features, "img_in", self.dtype, self.precision)(
            x.reshape(batch, image_tokens, x.shape[-1]))
        context = _TextIn(self.features, self.eps, dtype=self.dtype, precision=self.precision,
                          name="txt_in")(conditioning.context)
        joint = jnp.concatenate([image, context.astype(image.dtype)], axis=1)

        # The source embeds one extra row at time zero, which the text rows
        # read under `causal_condition`.
        wide = at_least_fp32(self.dtype)
        times = jnp.broadcast_to(jnp.asarray(time, wide).reshape(-1), (batch,))
        embedded = self._time(jnp.concatenate([times, jnp.zeros((1,), wide)]))
        embedded, zero = embedded[:batch], embedded[batch:]
        projected = _dense(4 * self.features, "modulation", self.dtype, self.precision)(
            nn.silu(jnp.concatenate([embedded, zero])))
        sampled = jnp.split(projected[:batch], 4, axis=-1)
        still = jnp.split(projected[batch:], 4, axis=-1) if self.causal_condition else sampled
        modulation = tuple(zip(still, sampled, strict=True))

        rotation = axis_tables(_rotary_positions(lengths, text, rows, columns), self.axes_dims_rope, 10000.0,
                               dtype=wide)
        for index in range(self.num_layers):
            joint = _Block(
                self.features, self.heads, self.head_dim, self.mlp_ratio, self.eps,
                dtype=self.dtype, precision=self.precision, attention_impl=self.attention_impl,
                name=f"transformer_blocks_{index}")(joint, modulation, rotation, lengths,
                                                     image_tokens)

        # Only the image's rows leave; the text's final norm and projection
        # are rows the source computes and its pipeline drops.
        image = joint[:, :image_tokens]
        scale = _dense(self.features, "norm_out_linear", self.dtype, self.precision)(
            nn.silu(embedded))
        image = _scaled(layer_norm(self.dtype, self.eps)(image), scale[:, None])
        output = _dense(self.out_channels, "proj_out", self.dtype, self.precision)(image)
        return output.reshape(batch, rows, columns, self.out_channels)


class _TextIn(nn.Module):
    """`QwenImage21TextProjection`: a zero-centred RMS norm, then two
    bias-free linears with a tanh GELU between."""

    features: int
    eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        x = RMSNorm(epsilon=self.eps, scale_offset=True, dtype=self.dtype, name="text_norm")(x)
        x = nn.gelu(_dense(self.features, "in_layer", self.dtype, self.precision)(x), approximate=True)
        return _dense(self.features, "out_layer", self.dtype, self.precision)(x)


__all__ = ["QwenImageTransformer"]
