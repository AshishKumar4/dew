"""The blocks the published MM-DiT family shares: SD3, Flux, FLUX.2, Z-Image
and Qwen-Image.

Each is a stack of modulated residual blocks whose attention joins an image
stream and a text stream, as Diffusers 0.34.0 to 0.40.0 run them. What they
share is here: the adaLN modulation, the affine-free layer norm, one joint
attention, the double-stream block SD3, Flux and FLUX.2 run, the feed-forwards
and the embedders. Each family's module keeps what it alone does.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import LayerNorm, RMSNorm, scaled_dot_product_attention
from dew.nn.sharding import logical_axes


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


def apply_rotary(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """The rotation `apply_rotary_emb` applies with `use_real_unbind_dim=-1`:
    adjacent channels are one complex pair, so the rotated copy is
    `(-x1, x0)` within each pair."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    rotated = jnp.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)
    return x * cos + rotated * sin


class Modulation(nn.Module):
    """A source `AdaLayerNormZero`-family projection: SiLU then one linear
    whose output is `pieces` chunks of the width, in the source's order."""

    features: int
    pieces: int
    bias: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, conditioning):
        projected = nn.Dense(self.pieces * self.features, use_bias=self.bias, dtype=self.dtype,
                             precision=self.precision, name="linear")(nn.silu(conditioning))
        return jnp.split(projected, self.pieces, axis=-1)


def modulate(x, shift, scale):
    """The source's `norm(x) * (1 + scale) + shift`, broadcast over tokens."""
    return x * (1 + scale[:, None]) + shift[:, None]


def layer_norm(dtype, epsilon: float = 1e-6):
    """The source's `LayerNorm(elementwise_affine=False)`, at the eps of 1e-6
    SD3 and Flux fix unless a config reads another."""
    return LayerNorm(epsilon=epsilon, use_scale=False, use_bias=False, dtype=dtype)


def embedding(x, features: int, name: str, *, bias: bool = True, dtype: Dtype | None = None,
              precision: PrecisionLike = None):
    """`TimestepEmbedding` and `PixArtAlphaTextProjection`: a linear, SiLU and
    a linear, stored flat as `{name}_linear_1` and `{name}_linear_2` in the
    calling module."""
    hidden = nn.Dense(features, use_bias=bias, dtype=dtype, precision=precision, name=f"{name}_linear_1")(x)
    return nn.Dense(features, use_bias=bias, dtype=dtype, precision=precision,
                    name=f"{name}_linear_2")(nn.silu(hidden))


def guided_time(embed: Callable[[jax.Array, str], jax.Array], time, guidance, *, guidance_embeds: bool):
    """The time's `embed` plus, where the checkpoint embeds its distilled
    guidance, the guidance's, scaled by a thousand as the source scales it.
    A guidance-embedded checkpoint needs the value and any other refuses it."""
    embedded = embed(time, "timestep_embedder")
    if guidance_embeds:
        if guidance is None:
            raise ValueError("This checkpoint embeds its guidance; conditioning needs it")
        return embedded + embed(guidance * 1000.0, "guidance_embedder")
    if guidance is not None:
        raise ValueError("This checkpoint has no guidance embedder")
    return embedded


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
class JointAttention(nn.Module):
    """The source's `Attention` over an image stream and an optional context.

    Each stream projects its own queries, keys and values (the context's are
    the `add_*` projections), per-head RMS norms take the queries and keys,
    the rotation turns them, and one softmax attends over the joined
    sequence, which splits back into each stream's output projection. SD3's
    `JointAttnProcessor2_0` joins the image first, Flux's processor the
    context; without a context it is self-attention over the image. The
    rotation is the cosines and sines of every joined token laid out against
    the heads, `[B or 1, S, 1, D]`, and `lengths` the real keys of each row.
    """

    heads: int
    head_dim: int
    bias: bool = True
    """Whether every projection carries a bias; FLUX.2's, Z-Image's and Qwen-Image's carry none."""
    qk_norm: bool = True
    """SD3 without a `qk_norm` config leaves its queries and keys unnormalized."""
    epsilon: float = 1e-6
    context_first: bool = False
    """Flux's order: the context's tokens lead the joined sequence."""
    project: bool = True
    """Flux's single-stream block projects the attention itself, beside its
    feed-forward, so it takes the output unprojected."""
    context_out: bool = True
    """SD3's last block reads no context back, so it holds no `to_add_out`."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    def _heads(self, name: str, x):
        projected = nn.Dense(self.heads * self.head_dim, use_bias=self.bias, dtype=self.dtype,
                             precision=self.precision, name=name)(x)
        return projected.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim)

    def _norm(self, name: str, x):
        return RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name=name)(x) if self.qk_norm else x

    def attend(self, query, key, value, lengths):
        """One softmax over the joined sequence, each row's keys past `lengths` masked."""
        return scaled_dot_product_attention(query, key, value, implementation=self.attention_impl,
                                            precision=self.precision, key_value_seq_lengths=lengths)

    @nn.compact
    def __call__(self, image, context=None, rotation=None, lengths=None):
        query = self._norm("norm_q", self._heads("to_q", image))
        key = self._norm("norm_k", self._heads("to_k", image))
        value = self._heads("to_v", image)
        if context is not None:
            added = (self._norm("norm_added_q", self._heads("add_q_proj", context)),
                     self._norm("norm_added_k", self._heads("add_k_proj", context)),
                     self._heads("add_v_proj", context))
            streams = (added, (query, key, value)) if self.context_first else ((query, key, value), added)
            query, key, value = (jnp.concatenate(pair, axis=1) for pair in zip(*streams, strict=True))
        if rotation is not None:
            query, key = apply_rotary(query, *rotation), apply_rotary(key, *rotation)
        attended = self.attend(query, key, value, lengths)
        inner = self.heads * self.head_dim
        attended = attended.reshape(attended.shape[0], attended.shape[1], inner)

        def projection(name: str):
            return nn.Dense(inner, use_bias=self.bias, dtype=self.dtype, precision=self.precision, name=name)

        if context is None:
            return (projection("to_out_0")(attended) if self.project else attended), None
        split = context.shape[1] if self.context_first else image.shape[1]
        image_out, context_out = ((attended[:, split:], attended[:, :split]) if self.context_first
                                  else (attended[:, :split], attended[:, split:]))
        return (projection("to_out_0")(image_out),
                projection("to_add_out")(context_out) if self.context_out else None)


@logical_axes({("net_0_proj",): ("embed", "mlp"), ("net_2",): ("mlp", "embed")})
class FeedForward(nn.Module):
    """The source's `FeedForward(activation_fn="gelu-approximate")`: one
    projection to four times the width into the tanh GELU, then one back,
    stored as `net.0.proj` and `net.2`."""

    features: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        hidden = nn.Dense(4 * self.features, dtype=self.dtype,
                          precision=self.precision, name="net_0_proj")(x)
        hidden = nn.gelu(hidden, approximate=True)
        return nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                        name="net_2")(hidden)


@logical_axes({("linear_in",): ("embed", "mlp"), ("linear_out",): ("mlp", "embed")})
class SwiGLU(nn.Module):
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


class DoubleStreamBlock(nn.Module):
    """One double-stream block: SD3's `JointTransformerBlock`, Flux's
    `FluxTransformerBlock` and FLUX.2's `Flux2TransformerBlock`.

    The image and the context each modulate and norm their own residual
    stream, meet in one `JointAttention`, gate its output back, then modulate
    and gate their own feed-forward. A block's modulation is its own
    projection of the conditioning vector (`norm1`, `norm1_context`): six
    pieces per stream, shift, scale and gate before the attention and again
    before the feed-forward.
    """

    features: int
    heads: int
    head_dim: int
    context_first: bool = False
    """Flux's and FLUX.2's: the context leads the joined attention."""
    qk_norm: bool = True
    bias: bool = True
    """FLUX.2's projections carry no bias."""
    epsilon: float = 1e-6
    mlp_hidden: int | None = None
    """FLUX.2's SwiGLU width; None is the four-times GELU feed-forward."""
    shared_modulation: bool = False
    """FLUX.2's: the conditioning is the model's (image, context) pair of six
    pieces each, which every block reads."""
    context_pre_only: bool = False
    """SD3's last block: the context is read and returned as it came, so its
    norm takes a scale and a shift and it holds no output or feed-forward."""
    dual_attention: bool = False
    """SD3.5's: three more image pieces modulate a second self-attention over
    the same normalized input."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    def _modulation(self, conditioning):
        def projected(pieces: int, name: str):
            return Modulation(self.features, pieces, dtype=self.dtype, precision=self.precision,
                              name=name)(conditioning)

        return (projected(9 if self.dual_attention else 6, "norm1"),
                projected(2 if self.context_pre_only else 6, "norm1_context"))

    def _feed_forward(self, name: str):
        if self.mlp_hidden is None:
            return FeedForward(self.features, dtype=self.dtype, precision=self.precision, name=name)
        return SwiGLU(self.features, self.mlp_hidden, dtype=self.dtype, precision=self.precision, name=name)

    def _attention(self, name: str, *, context_out: bool = True):
        return JointAttention(self.heads, self.head_dim, bias=self.bias, qk_norm=self.qk_norm,
                              epsilon=self.epsilon, context_first=self.context_first, context_out=context_out,
                              dtype=self.dtype, precision=self.precision, attention_impl=self.attention_impl,
                              name=name)

    @nn.compact
    def __call__(self, image, context, conditioning, rotation=None):
        image_mods, context_mods = conditioning if self.shared_modulation else self._modulation(conditioning)
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = image_mods[:6]
        # A last block's continuous norm emits its scale before its shift; the
        # zero-initialized one emits shift, scale and then the gates.
        context_scale, context_shift = context_mods[:2] if self.context_pre_only else context_mods[1::-1]
        norm = layer_norm(self.dtype, self.epsilon)
        normalized = norm(image)
        attention = self._attention("attn", context_out=not self.context_pre_only)
        context_input = modulate(norm(context), context_shift, context_scale)
        attended, context_attended = attention(modulate(normalized, shift, scale), context_input, rotation)
        image = image + gate[:, None] * attended
        if self.dual_attention:
            shift2, scale2, gate2 = image_mods[6:]
            attended2, _ = self._attention("attn2")(modulate(normalized, shift2, scale2))
            image = image + gate2[:, None] * attended2
        image = image + gate_mlp[:, None] * self._feed_forward("ff")(
            modulate(norm(image), shift_mlp, scale_mlp))
        if self.context_pre_only:
            return image, context
        # A block that keeps its context is the one whose attention returns it.
        assert context_attended is not None
        context_gate, context_shift_mlp, context_scale_mlp, context_gate_mlp = context_mods[2:]
        context = context + context_gate[:, None] * context_attended
        context = context + context_gate_mlp[:, None] * self._feed_forward("ff_context")(
            modulate(norm(context), context_shift_mlp, context_scale_mlp))
        return image, context


__all__ = ["DoubleStreamBlock", "FeedForward", "JointAttention", "Modulation", "SwiGLU", "apply_rotary",
           "embedding", "guided_time", "layer_norm", "modulate", "rotary_table"]
