"""The blocks shared by the published MM-DiT family: SD3, Flux, FLUX.2, Z-Image and Qwen-Image.

Each of these models is a stack of modulated residual blocks whose attention
joins an image stream and a text stream, as Diffusers 0.34.0 to 0.40.0 run
them. This module holds the shared parts: the adaLN modulation, the layer
norm without affine parameters, one joint attention, the double-stream block
that SD3, Flux and FLUX.2 run, the feed-forwards and the embedders. Each
family's own module keeps what only that family does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import LayerNorm, RMSNorm, scaled_dot_product_attention
from dew.nn.rope import rotate
from dew.nn.sharding import logical_axes


class Modulation(nn.Module):
    """Projects the conditioning as the source's `AdaLayerNormZero` family does: SiLU, then one linear layer.

    The output is `pieces` chunks of the width, in the source's order.
    """

    features: int
    pieces: int
    bias: bool = True
    zero_init: bool = False
    """Whether the projection starts at zero, adaLN-Zero's identity block, which
    Dew's own MM-DiT trains from; the published families load their weights."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, conditioning):
        projected = nn.Dense(self.pieces * self.features, use_bias=self.bias, dtype=self.dtype,
                             precision=self.precision, name="linear",
                             kernel_init=(nn.initializers.zeros if self.zero_init
                                          else nn.initializers.lecun_normal()))(nn.silu(conditioning))
        return jnp.split(projected, self.pieces, axis=-1)


def modulate(x, shift, scale):
    """Return the source's `norm(x) * (1 + scale) + shift` for a normed `x`, broadcast over tokens."""
    return x * (1 + scale[:, None]) + shift[:, None]


def layer_norm(dtype, epsilon: float = 1e-6):
    """Return the source's `LayerNorm(elementwise_affine=False)`.

    Its eps is 1e-6, the value SD3 and Flux fix, unless a config gives another.
    """
    return LayerNorm(epsilon=epsilon, use_scale=False, use_bias=False, dtype=dtype)


def embedding(x, features: int, name: str, *, bias: bool = True, dtype: Dtype | None = None,
              precision: PrecisionLike = None, activation: Callable[[jax.Array], jax.Array] = nn.silu):
    """Apply `TimestepEmbedding` or `PixArtAlphaTextProjection` to `x`.

    Both are a linear layer, an activation and a second linear layer. The
    activation is SiLU, or Wan's tanh GELU. The two layers are stored flat
    as `{name}_linear_1` and `{name}_linear_2` in the calling module.
    """
    hidden = nn.Dense(features, use_bias=bias, dtype=dtype, precision=precision, name=f"{name}_linear_1")(x)
    return nn.Dense(features, use_bias=bias, dtype=dtype, precision=precision,
                    name=f"{name}_linear_2")(activation(hidden))


def guided_time(embed: Callable[[jax.Array, str], jax.Array], time, guidance, *, guidance_embeds: bool):
    """Return the time's `embed`, plus the guidance's when the checkpoint embeds its distilled guidance.

    The guidance is multiplied by a thousand before embedding, as the source
    does. A guidance-embedded checkpoint requires `guidance`, and any other
    checkpoint raises `ValueError` when `guidance` is given.
    """
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
    """Runs the source's `Attention` over an image stream and an optional context.

    Each stream projects its own queries, keys and values; the context uses
    the `add_*` projections. Per-head RMS norms normalize the queries and
    keys, the rotation is applied to them, and one softmax attends over the
    joined sequence. The result is split back into the two streams, and each
    goes through its own output projection. SD3's `JointAttnProcessor2_0`
    puts the image first in the joined sequence, and Flux's processor puts
    the context first. Without a context, it is self-attention over the
    image. The rotation is the cosines and sines of every joined token, one
    angle per channel pair, `[B or 1, S, D // 2]`, and `lengths` is the
    number of real keys in each row.
    """

    heads: int
    head_dim: int
    bias: bool = True
    """Whether every projection has a bias. FLUX.2's, Z-Image's and Qwen-Image's have none."""
    qk_norm: bool = True
    """Whether the queries and keys are RMS-normed. SD3 without a `qk_norm` config leaves them as they are."""
    epsilon: float = 1e-6
    context_first: bool = False
    """Whether the context's tokens come first in the joined sequence, as in Flux."""
    project: bool = True
    """Whether the attention applies its own output projection. Flux's
    single-stream block projects the attention output itself, together with
    its feed-forward, so it takes the output unprojected."""
    context_out: bool = True
    """Whether the context's output is projected and returned. SD3's last block
    reads no context back, so it has no `to_add_out`."""
    rotary_pairs: Literal["half", "adjacent"] = "adjacent"
    """The channels each angle turns (`dew.nn.rope.rotate`): the published
    families' adjacent pairs, or the rotate-half halves of Dew's own MM-DiT."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl
    force_fp32_for_softmax: bool = True

    def _heads(self, name: str, x):
        projected = nn.Dense(self.heads * self.head_dim, use_bias=self.bias, dtype=self.dtype,
                             precision=self.precision, name=name)(x)
        return projected.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim)

    def _norm(self, name: str, x):
        return RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name=name)(x) if self.qk_norm else x

    def attend(self, query, key, value, lengths):
        """Return one softmax attention over the joined sequence, masking each row's keys past `lengths`."""
        return scaled_dot_product_attention(query, key, value, implementation=self.attention_impl,
                                            precision=self.precision, key_value_seq_lengths=lengths,
                                            force_fp32_for_softmax=self.force_fp32_for_softmax)

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
            query, key = (rotate(part, *rotation, pairs=self.rotary_pairs) for part in (query, key))
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
    """Applies the source's `FeedForward(activation_fn="gelu-approximate")`.

    One projection to the `hidden` width feeds the tanh GELU, and a second
    projects back. `hidden` is four times the width unless a config names its
    `inner_dim`, as Wan's `ffn_dim` does. The weights are stored as
    `net.0.proj` and `net.2`.
    """

    features: int
    hidden: int | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        hidden = nn.Dense(self.hidden or 4 * self.features, dtype=self.dtype,
                          precision=self.precision, name="net_0_proj")(x)
        hidden = nn.gelu(hidden, approximate=True)
        return nn.Dense(self.features, dtype=self.dtype, precision=self.precision,
                        name="net_2")(hidden)


@logical_axes({("linear_in",): ("embed", "mlp"), ("linear_out",): ("mlp", "embed")})
class SwiGLU(nn.Module):
    """Applies `Flux2FeedForward`: `linear_in` to twice the hidden width, SiLU of the
    first half times the second, then `linear_out` back. Neither projection has a
    bias."""

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
    """Runs one double-stream block of SD3, Flux, FLUX.2 or Dew's own MM-DiT.

    It matches SD3's `JointTransformerBlock`, Flux's `FluxTransformerBlock`
    and FLUX.2's `Flux2TransformerBlock`. The image and the context each
    modulate and norm their own residual stream. They meet in one
    `JointAttention`, and each adds its gated attention output back to its
    stream. Then each runs its own modulated feed-forward and adds the gated
    output. A block's modulation is its own projection of the conditioning
    vector (`norm1`, `norm1_context`): six pieces per stream, a shift, scale
    and gate before the attention and again before the feed-forward.
    `dropout_rate` drops each projected attention and feed-forward output
    while `train`, before its gate.
    """

    features: int
    heads: int
    head_dim: int
    context_first: bool = False
    """Whether the context comes first in the joined attention, as in Flux and FLUX.2."""
    qk_norm: bool = True
    bias: bool = True
    """Whether the projections have biases. FLUX.2's have none."""
    epsilon: float = 1e-6
    qk_epsilon: float | None = None
    """The query and key norms' epsilon, None for `epsilon`. Dew's MM-DiT norms
    its streams at its stack's epsilon and its queries and keys at 1e-6."""
    mlp_hidden: int | None = None
    """The feed-forward's hidden width, four times the width for None."""
    swiglu: bool = False
    """Whether the feed-forward is FLUX.2's SwiGLU rather than the tanh GELU."""
    shared_modulation: bool = False
    """Whether the conditioning is the model's shared modulation, as in FLUX.2: an
    (image, context) pair of six pieces each, which every block reads."""
    zero_modulation: bool = False
    """Whether each modulation starts at zero (`Modulation.zero_init`)."""
    context_pre_only: bool = False
    """Whether this is SD3's last block, which reads the context and returns it
    unchanged. Its context norm then takes only a scale and a shift, and it has
    no context output projection or feed-forward."""
    dual_attention: bool = False
    """Whether the block runs SD3.5's second self-attention over the same
    normalized input, modulated by three more image pieces."""
    rotary_pairs: Literal["half", "adjacent"] = "adjacent"
    """The channels each angle turns (`JointAttention.rotary_pairs`)."""
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl
    force_fp32_for_softmax: bool = True

    def _modulation(self, conditioning):
        def projected(pieces: int, name: str):
            return Modulation(self.features, pieces, zero_init=self.zero_modulation, dtype=self.dtype,
                              precision=self.precision, name=name)(conditioning)

        return (projected(9 if self.dual_attention else 6, "norm1"),
                projected(2 if self.context_pre_only else 6, "norm1_context"))

    def _feed_forward(self, name: str):
        hidden = self.mlp_hidden or 4 * self.features
        if self.swiglu:
            return SwiGLU(self.features, hidden, dtype=self.dtype, precision=self.precision, name=name)
        return FeedForward(self.features, hidden, dtype=self.dtype, precision=self.precision, name=name)

    def _attention(self, name: str, *, context_out: bool = True):
        return JointAttention(self.heads, self.head_dim, bias=self.bias, qk_norm=self.qk_norm,
                              epsilon=self.epsilon if self.qk_epsilon is None else self.qk_epsilon,
                              context_first=self.context_first, context_out=context_out,
                              rotary_pairs=self.rotary_pairs, dtype=self.dtype, precision=self.precision,
                              attention_impl=self.attention_impl,
                              force_fp32_for_softmax=self.force_fp32_for_softmax, name=name)

    @nn.compact
    def __call__(self, image, context, conditioning, rotation=None, train: bool = False):
        image_mods, context_mods = conditioning if self.shared_modulation else self._modulation(conditioning)
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = image_mods[:6]
        # A last block's continuous norm emits its scale before its shift; the
        # zero-initialized one emits shift, scale and then the gates.
        context_scale, context_shift = context_mods[:2] if self.context_pre_only else context_mods[1::-1]
        norm = layer_norm(self.dtype, self.epsilon)
        dropout = nn.Dropout(self.dropout_rate, deterministic=not train)
        normalized = norm(image)
        attention = self._attention("attn", context_out=not self.context_pre_only)
        context_input = modulate(norm(context), context_shift, context_scale)
        attended, context_attended = attention(modulate(normalized, shift, scale), context_input, rotation)
        image = image + gate[:, None] * dropout(attended)
        if self.dual_attention:
            shift2, scale2, gate2 = image_mods[6:]
            attended2, _ = self._attention("attn2")(modulate(normalized, shift2, scale2))
            image = image + gate2[:, None] * attended2
        image = image + gate_mlp[:, None] * dropout(self._feed_forward("ff")(
            modulate(norm(image), shift_mlp, scale_mlp)))
        if self.context_pre_only:
            return image, context
        # A block that keeps its context is the one whose attention returns it.
        assert context_attended is not None
        context_gate, context_shift_mlp, context_scale_mlp, context_gate_mlp = context_mods[2:]
        context = context + context_gate[:, None] * dropout(context_attended)
        context = context + context_gate_mlp[:, None] * dropout(self._feed_forward("ff_context")(
            modulate(norm(context), context_shift_mlp, context_scale_mlp)))
        return image, context


__all__ = ["DoubleStreamBlock", "FeedForward", "JointAttention", "Modulation", "SwiGLU", "embedding",
           "guided_time", "layer_norm", "modulate"]
