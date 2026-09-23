"""Qwen-Image 2.1's transformer, as Diffusers' `QwenImage21Transformer2DModel`
runs it at commit 6256aa76.

One residual stream carries the image and the text together, image first.
The text is the vision-language encoder's hidden states, projected through a
zero-centred RMS norm and a GELU MLP; the image is the latent, one token per
latent position, projected by one linear. Every block modulates both with
scales and tanh gates read from one projection of the time embedding that
all blocks share, attends, and runs a SwiGLU feed-forward.

Two things set it apart from the MM-DiT families. Attention is block-causal:
the text attends causally and the image attends to all of the text and to
itself. And under `causal_condition` the text is modulated from time zero
rather than from the sampled time, so its activations do not change across a
walk. The rotary table spans three axes: a text token advances one shared
position on all three, and the image sits at the frame position after the
text on a height and width grid centred on zero.

The source takes the text padded to the longest prompt of the call and
starts the image's frame position after that padding. Here each row starts it
after its own text, which is the source's value for a prompt alone and for a
batch of prompts of one length, and keeps a row's output independent of what
it is batched with.
"""

from __future__ import annotations

import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.diffusion.process import DenoisingCondition
from dew.nn.attention import RMSNorm, scaled_dot_product_attention
from dew.nn.backbones.unet_condition import sinusoidal_time
from dew.nn.sharding import logical_axes
from dew.registry import models

from .causal_transformer import GatedMLP
from .flux import apply_rotary
from .sd3 import _layer_norm


def _image_grid(rows: int, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """The height and width indices of one image block, row-major and
    centred on zero: `-(n - n // 2)` up to `n // 2 - 1` along each side."""
    heights = np.arange(-(rows - rows // 2), rows // 2)
    widths = np.arange(-(columns - columns // 2), columns // 2)
    return np.repeat(heights, columns), np.tile(widths, rows)


def _rotary_angles(lengths: jax.Array, text: int, rows: int, columns: int,
                   axes: Sequence[int]) -> jax.Array:
    """The angle of every channel pair of every token, image first,
    `[B, rows * columns + text, sum(axes) / 2]`.

    A text token sits at its index on all three axes. The image's frame axis
    is the row's own text length, and its other two are the centred grid.
    Each axis turns at `QwenImage21Rope.rope_params`' inverse frequencies,
    theta 10000 in its float32 arithmetic.
    """
    heights, widths = _image_grid(rows, columns)
    index = jnp.arange(text, dtype=jnp.float32)
    batch = lengths.shape[0]
    frame = jnp.concatenate([jnp.broadcast_to(lengths.astype(jnp.float32)[:, None],
                                              (batch, rows * columns)),
                             jnp.broadcast_to(index, (batch, text))], axis=1)
    height = jnp.concatenate([jnp.asarray(heights, jnp.float32), index])
    width = jnp.concatenate([jnp.asarray(widths, jnp.float32), index])
    positions = (frame, jnp.broadcast_to(height, frame.shape), jnp.broadcast_to(width, frame.shape))
    return jnp.concatenate([
        position[..., None] * jnp.asarray(np.float32(1.0) / np.power(
            np.float32(10000), np.arange(0, dim, 2).astype(np.float32) / np.float32(dim)))
        for position, dim in zip(positions, axes, strict=True)], axis=-1)


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


@logical_axes({("attn", name): ("embed", None) for name in ("to_q", "to_k", "to_v")}
              | {("attn", "to_out_0"): (None, "embed")})
class _Attention(nn.Module):
    """`QwenImage21Attention` under the source's exact multi-pass prefill:
    the image queries attend over everything and the text queries causally
    over the text, a padded text key excluded from both. Queries and keys
    are RMS normalized per head, then rotated.

    The sequence runs image first, so the keys a row's image queries read
    are a prefix of it and its text queries' keys a prefix of the text. Both
    calls pass those prefix lengths, which cuDNN applies as its padding mask
    and skips the padded keys by, where a mask would reach it as a dense
    additive bias read once per head. Attention does not depend on the order
    of its keys and every token carries its own rotary position, so this is
    the source's arithmetic, summed in another order.
    """

    heads: int
    head_dim: int
    features: int
    eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    def _heads(self, name: str, x):
        projected = _dense(self.heads * self.head_dim, name, self.dtype, self.precision)(x)
        return projected.reshape(*x.shape[:2], self.heads, self.head_dim)

    @nn.compact
    def __call__(self, x, cos, sin, lengths, image: int):
        query = RMSNorm(epsilon=self.eps, dtype=self.dtype, name="norm_q")(self._heads("to_q", x))
        key = RMSNorm(epsilon=self.eps, dtype=self.dtype, name="norm_k")(self._heads("to_k", x))
        value = self._heads("to_v", x)
        query, key = apply_rotary(query, cos, sin), apply_rotary(key, cos, sin)
        attend = functools.partial(scaled_dot_product_attention,
                                   implementation=self.attention_impl, precision=self.precision)
        attended = jnp.concatenate([
            attend(query[:, :image], key, value, key_value_seq_lengths=image + lengths),
            attend(query[:, image:], key[:, image:], value[:, image:], causal=True,
                   key_value_seq_lengths=lengths)], axis=1)
        return _dense(self.features, "to_out_0", self.dtype, self.precision)(
            attended.reshape(*x.shape[:2], self.heads * self.head_dim))


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
    def __call__(self, x, modulation, cos, sin, lengths, image: int):
        scale, gate, scale_mlp, gate_mlp = modulation
        attended = _Attention(self.heads, self.head_dim, self.features, self.eps,
                              dtype=self.dtype, precision=self.precision,
                              attention_impl=self.attention_impl, name="attn")(
            _per_rows(_layer_norm(self.dtype, self.eps)(x), scale, image, _scaled),
            cos, sin, lengths, image)
        x = x + _per_rows(attended, gate, image, lambda rows, value: jnp.tanh(value) * rows)
        hidden = GatedMLP(self.features * self.mlp_ratio, self.features, dtype=self.dtype,
                          precision=self.precision, name="img_mlp")(
            _per_rows(_layer_norm(self.dtype, self.eps)(x), scale_mlp, image, _scaled))
        x = x + _per_rows(hidden, gate_mlp, image, lambda rows, value: jnp.tanh(value) * rows)
        if x.dtype == jnp.float16:
            x = jnp.clip(x, -65504, 65504)
        return x


@models("qwen_image_transformer")
@logical_axes({("img_in",): (None, "embed"), ("proj_out",): ("embed", None),
               ("modulation",): (None, "embed"), ("norm_out_linear",): (None, "embed"),
               ("txt_in", "in_layer"): (None, "embed"), ("txt_in", "out_layer"): (None, "embed"),
               ("timestep_embedder_linear_1",): (None, "embed"),
               ("timestep_embedder_linear_2",): (None, "embed")})
class QwenImageTransformer(nn.Module):
    """`QwenImage21Transformer2DModel` over Dew's interface.

    `__call__` takes NHWC latents, one token per position, the model time
    the schedule supplies - the sigma times the training count, which is the
    product the source reaches by dividing its timestep by a thousand and
    multiplying it back - and a `DenoisingCondition` whose `context` is the
    encoder's text states after the system prompt, right-padded, with `mask`
    marking the real ones. Each row's real tokens lead its text, the layout
    the rotary positions and the attention's key lengths both read. It
    returns the prediction at the image's tokens.
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

    def _time(self, time):
        """`QwenImage21TimestepProjEmbeddings`: cosines first, then two
        bias-free linears with a SiLU between."""
        hidden = _dense(self.features, "timestep_embedder_linear_1", self.dtype, self.precision)(
            sinusoidal_time(time, 256).astype(self.dtype or jnp.float32))
        return _dense(self.features, "timestep_embedder_linear_2", self.dtype, self.precision)(
            nn.silu(hidden))

    @nn.compact
    def __call__(self, x, time, conditioning: DenoisingCondition, train: bool = False):
        """`train` is the objective's standard call contract; the published
        transformer holds no dropout, so it changes nothing here."""
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
        times = jnp.broadcast_to(jnp.asarray(time, jnp.float32).reshape(-1), (batch,))
        embedded = self._time(jnp.concatenate([times, jnp.zeros((1,), jnp.float32)]))
        embedded, zero = embedded[:batch], embedded[batch:]
        projected = _dense(4 * self.features, "modulation", self.dtype, self.precision)(
            nn.silu(jnp.concatenate([embedded, zero])))
        sampled = jnp.split(projected[:batch], 4, axis=-1)
        still = jnp.split(projected[batch:], 4, axis=-1) if self.causal_condition else sampled
        modulation = tuple(zip(still, sampled, strict=True))

        angles = _rotary_angles(lengths, text, rows, columns, self.axes_dims_rope)
        cos = jnp.repeat(jnp.cos(angles), 2, axis=-1)[:, :, None].astype(joint.dtype)
        sin = jnp.repeat(jnp.sin(angles), 2, axis=-1)[:, :, None].astype(joint.dtype)
        for index in range(self.num_layers):
            joint = _Block(
                self.features, self.heads, self.head_dim, self.mlp_ratio, self.eps,
                dtype=self.dtype, precision=self.precision, attention_impl=self.attention_impl,
                name=f"transformer_blocks_{index}")(joint, modulation, cos, sin, lengths,
                                                     image_tokens)

        # Only the image's rows leave; the text's final norm and projection
        # are rows the source computes and its pipeline drops.
        image = joint[:, :image_tokens]
        scale = _dense(self.features, "norm_out_linear", self.dtype, self.precision)(
            nn.silu(embedded))
        image = _scaled(_layer_norm(self.dtype, self.eps)(image), scale[:, None])
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
