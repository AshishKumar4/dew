"""FLUX.2's transformer, as Diffusers 0.40.0's `Flux2Transformer2DModel` runs it.

FLUX.2 keeps Flux's two halves - double-stream blocks that join the image and
the text only inside attention, then single-stream blocks over the joined
sequence - and changes what fills them:

- no projection carries a bias, and the feed-forwards are SwiGLU: one map
  into twice the hidden width, whose first half gates the second through a
  SiLU, and one map back;
- the modulation is computed once per call from the time (and distilled
  guidance) embedding, one set for the double blocks' image stream, one for
  their text stream and one for the single blocks, and every block reads the
  same set;
- a single block projects its queries, keys, values and the feed-forward's
  input in one map, and its attention output and gated hidden state in one
  map back;
- the rotary table has four axes (time, row, column, token), 32 channels
  each at theta 2000: a text token sits at (0, 0, 0, i) and an image position
  at (0, row, column, 0);
- there is no pooled text vector: the text is a sequence of stacked hidden
  states from the Mistral-3 encoder.

The latent this module reads is the pipeline's: the VAE's 32 channels folded
2x2 into 128, one token per position, which `dew.nn.autoencoders.flux2`
produces, so no packing happens here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm, scaled_dot_product_attention
from dew.nn.backbones.unet_condition import sinusoidal_time
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import logical_axes
from dew.registry import models

from .flux import _FluxAttention, apply_rotary, rotary_table
from .sd3 import _layer_norm, _modulate, _Modulation

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


@logical_axes({("linear_in",): ("embed", "mlp"), ("linear_out",): ("mlp", "embed")})
class _SwiGLU(nn.Module):
    """`Flux2FeedForward`: `linear_in` to twice the hidden width, SiLU of the
    first half times the second, `linear_out` back; no biases."""

    features: int
    hidden: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        gate, value = jnp.split(nn.Dense(2 * self.hidden, use_bias=False, dtype=self.dtype,
                                         precision=self.precision, name="linear_in")(x), 2, axis=-1)
        return nn.Dense(self.features, use_bias=False, dtype=self.dtype, precision=self.precision,
                        name="linear_out")(nn.silu(gate) * value)


class Flux2Block(nn.Module):
    """One `Flux2TransformerBlock`: Flux's double-stream block with bias-free
    projections, SwiGLU feed-forwards, and each stream's six modulation pieces
    handed in."""

    features: int
    heads: int
    head_dim: int
    hidden: int
    epsilon: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.compact
    def __call__(self, image, context, image_mods, text_mods, cos, sin):
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = image_mods
        text_shift, text_scale, text_gate, text_shift_mlp, text_scale_mlp, text_gate_mlp = text_mods
        norm = _layer_norm(self.dtype, self.epsilon)
        attended, text_attended = _FluxAttention(
            self.heads, self.head_dim, bias=False, epsilon=self.epsilon, dtype=self.dtype,
            precision=self.precision, attention_impl=self.attention_impl, name="attn")(
                _modulate(norm(image), shift, scale), _modulate(norm(context), text_shift, text_scale), cos, sin)
        # A double-stream block's attention returns both streams.
        assert text_attended is not None
        image = image + gate[:, None] * attended
        image = image + gate_mlp[:, None] * _SwiGLU(
            self.features, self.hidden, dtype=self.dtype, precision=self.precision, name="ff")(
                _modulate(norm(image), shift_mlp, scale_mlp))
        context = context + text_gate[:, None] * text_attended
        context = context + text_gate_mlp[:, None] * _SwiGLU(
            self.features, self.hidden, dtype=self.dtype, precision=self.precision, name="ff_context")(
                _modulate(norm(context), text_shift_mlp, text_scale_mlp))
        return image, context


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
    def __call__(self, x, mods, cos, sin):
        shift, scale, gate = mods
        inner = self.heads * self.head_dim
        normalized = _modulate(_layer_norm(self.dtype, self.epsilon)(x), shift, scale)
        projected = nn.Dense(3 * inner + 2 * self.hidden, use_bias=False, dtype=self.dtype,
                             precision=self.precision, name="to_qkv_mlp_proj")(normalized)
        heads = (x.shape[0], x.shape[1], self.heads, self.head_dim)
        query, key, value = (part.reshape(heads) for part in jnp.split(projected[..., :3 * inner], 3, axis=-1))
        query = RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name="norm_q")(query)
        key = RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name="norm_k")(key)
        rotation = (cos[None, :, None, :], sin[None, :, None, :])
        attended = scaled_dot_product_attention(
            apply_rotary(query, *rotation), apply_rotary(key, *rotation), value,
            implementation=self.attention_impl, precision=self.precision)
        gated, value_half = jnp.split(projected[..., 3 * inner:], 2, axis=-1)
        joined = jnp.concatenate([attended.reshape(x.shape[0], x.shape[1], inner), nn.silu(gated) * value_half],
                                 axis=-1)
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

    def _embedding(self, time, guidance, dtype):
        """`Flux2TimestepGuidanceEmbeddings`: the time's sinusoids through two
        bias-free maps, plus the guidance's (scaled by a thousand, as the
        source scales it) where the checkpoint embeds it."""
        def embedder(values, name: str):
            features = sinusoidal_time(values, self.timestep_guidance_channels,
                                       dtype=at_least_fp32(self.dtype)).astype(dtype)
            hidden = nn.Dense(self.features, use_bias=False, dtype=self.dtype, precision=self.precision,
                              name=f"{name}_linear_1")(features)
            return nn.Dense(self.features, use_bias=False, dtype=self.dtype, precision=self.precision,
                            name=f"{name}_linear_2")(nn.silu(hidden))

        embedded = embedder(time, "timestep_embedder")
        if self.guidance_embeds:
            if guidance is None:
                raise ValueError("This checkpoint embeds its guidance; conditioning needs it")
            embedded = embedded + embedder(guidance * 1000.0, "guidance_embedder")
        elif guidance is not None:
            raise ValueError("This checkpoint has no guidance embedder")
        return embedded

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
        embedded = self._embedding(time, conditioning.guidance, image.dtype)

        def modulation(pieces: int, name: str):
            return _Modulation(self.features, pieces, bias=False, dtype=self.dtype, precision=self.precision,
                               name=name)(embedded)

        image_mods = modulation(6, "double_stream_modulation_img")
        text_mods = modulation(6, "double_stream_modulation_txt")
        single_mods = modulation(3, "single_stream_modulation")
        cosines, sines = rotary_table(flux2_positions(rows, columns, context.shape[1]), self.axes_dims_rope,
                                      theta=self.rope_theta)
        cos, sin = jnp.asarray(cosines, image.dtype), jnp.asarray(sines, image.dtype)
        hidden = int(self.features * self.mlp_ratio)
        block = {"epsilon": self.eps, "dtype": self.dtype, "precision": self.precision,
                     "attention_impl": self.attention_impl}
        for index in range(self.num_layers):
            image, context = Flux2Block(self.features, self.heads, self.head_dim, hidden,
                                        name=f"transformer_blocks_{index}", **block)(
                image, context, image_mods, text_mods, cos, sin)
        joined = jnp.concatenate([context, image], axis=1)
        for index in range(self.num_single_layers):
            joined = Flux2SingleBlock(self.features, self.heads, self.head_dim, hidden,
                                      name=f"single_transformer_blocks_{index}", **block)(
                joined, single_mods, cos, sin)
        image = joined[:, context.shape[1]:]
        scale, shift = modulation(2, "norm_out")
        image = _modulate(_layer_norm(self.dtype, self.eps)(image), shift, scale)
        out = nn.Dense(self.out_channels, name="proj_out", **dense)(image)
        return out.reshape(batch, rows, columns, self.out_channels)


__all__ = ["Flux2Block", "Flux2SingleBlock", "Flux2Transformer", "flux2_positions"]
