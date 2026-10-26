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
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import (
    LayerNorm,
    RMSNorm,
    projected_heads,
    projected_output,
    scaled_dot_product_attention,
)
from dew.nn.blocks import Modulation
from dew.nn.dropout import Dropout
from dew.nn.flops import MissingFlopsCount, attention_flops, matching_lengths, projection_flops
from dew.nn.mlp import GatedMLP
from dew.nn.rope import rotate
from dew.nn.sharding import logical_axes

TIME_FEATURES = 256


def modulate(x, shift, scale):
    """Return the source's `norm(x) * (1 + scale) + shift` for a normed `x`, broadcast over tokens."""
    return x * (1 + scale[:, None]) + shift[:, None]


def layer_norm(dtype, epsilon: float = 1e-6):
    """Return the source's `LayerNorm(elementwise_affine=False)`.

    Its eps is 1e-6, the value SD3 and Flux fix, unless a config gives another.
    """
    return LayerNorm(epsilon=epsilon, use_scale=False, use_bias=False, dtype=dtype)


def embedding(owner: nn.Module, x, features: int, name: str, *, bias: bool = True, dtype: Dtype | None = None,
              precision: PrecisionLike = None, activation: Callable[[jax.Array], jax.Array] = nn.silu):
    """Apply `TimestepEmbedding` or `PixArtAlphaTextProjection` to `x`.

    Both are a linear layer, an activation and a second linear layer. The
    activation is SiLU, or Wan's tanh GELU. The two layers are stored flat
    as `{name}_linear_1` and `{name}_linear_2` in the calling module.
    """
    mlp = GatedMLP(features, features, in_features=x.shape[-1], activation=activation, use_bias=bias,
                    names=(f"{name}_linear_1", f"{name}_linear_2"), dtype=dtype, precision=precision,
                    name=name)
    nn.share_scope(owner, mlp)
    return mlp(x)


def embedding_axes(*names: str) -> dict[tuple[str, ...], tuple[str | None, ...]]:
    """The logical axes of the two layers `embedding` builds under each of
    `names`: each shards its output, the model's width."""
    return {(f"{name}_linear_{index}",): (None, "embed") for name in names for index in (1, 2)}


class TextStream:
    """What a transformer that runs its text as a second stream beside the
    image declares (SD3, Flux, FLUX.2): its width is its heads' together,
    each row's text attends as the span it fills, every call takes the text
    as `conditioning`, and its FLOPs are counted over both streams
    (`multisequence_flops`), never one."""

    if TYPE_CHECKING:
        heads: int
        head_dim: int

    @property
    def features(self) -> int:
        return self.heads * self.head_dim

    @nn.nowrap
    def condition_lengths(self, lengths: np.ndarray, spans: np.ndarray) -> np.ndarray:
        return spans

    @property
    def text_keyword(self) -> str:
        """Every call takes the text as `conditioning`, run beside the image in the double-stream blocks."""
        return "conditioning"

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount(
            f"{type(self).__name__} needs multisequence_flops(image_lengths, text_lengths)")


def embedding_flops(lengths, input_features: int, features: int) -> float:
    """The two maps that `embedding` builds around its activation."""
    return GatedMLP(features, features, in_features=input_features, activation=nn.silu,
                    parent=None).model_flops(lengths)


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
    angle per channel pair, `[B or 1, S, D // 2]`, or of the image's tokens
    alone, which turn before the join and leave the context as it is (Dew's
    MM-DiT); `lengths` is the number of real keys in each row.
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

    @property
    def inner_features(self) -> int:
        return self.heads * self.head_dim

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("JointAttention needs stream_flops with image/context feature widths")

    @nn.nowrap
    def stream_flops(self, lengths, context_lengths, image_features: int,
                     context_features: int | None = None) -> float:
        """Both streams' Q/K/V maps and joined QK/AV, with optional output maps."""
        joined_streams = context_lengths is not None
        context_lengths = (np.zeros(np.shape(lengths), np.int64)
                           if context_lengths is None else context_lengths)
        image, context = matching_lengths(lengths, context_lengths)
        inner = self.inner_features
        count = (3 * projection_flops(lengths, image_features, inner)
                 + 3 * projection_flops(context_lengths, context_features or 0, inner)
                 + attention_flops((image + context).astype(np.int64), self.heads,
                                   self.head_dim, self.head_dim, causal=False))
        if self.project or joined_streams:
            count += projection_flops(lengths, inner, inner)
            if self.context_out:
                count += projection_flops(context_lengths, inner, inner)
        return count

    def _heads(self, name: str, x, norm: str | None = None):
        return projected_heads(x, name, self.heads, self.head_dim, flat=True, bias=self.bias,
                               dtype=self.dtype, precision=self.precision,
                               norm=RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name=norm)
                               if norm and self.qk_norm else None)

    def _output(self, name: str, attended):
        return projected_output(attended, name, self.inner_features, flat=True, bias=self.bias,
                                dtype=self.dtype, precision=self.precision)

    def attend(self, query, key, value, lengths):
        """Return one softmax attention over the joined sequence, masking each row's keys past `lengths`."""
        return scaled_dot_product_attention(query, key, value, implementation=self.attention_impl,
                                            precision=self.precision, key_value_seq_lengths=lengths,
                                            force_fp32_for_softmax=self.force_fp32_for_softmax)

    @nn.compact
    def __call__(self, image, context=None, rotation=None, lengths=None):
        query = self._heads("to_q", image, "norm_q")
        key = self._heads("to_k", image, "norm_k")
        value = self._heads("to_v", image)
        if context is not None:
            added = (self._heads("add_q_proj", context, "norm_added_q"),
                     self._heads("add_k_proj", context, "norm_added_k"),
                     self._heads("add_v_proj", context))
            if rotation is not None and rotation[0].shape[-2] == image.shape[1]:
                query, key = (rotate(part, *rotation, pairs=self.rotary_pairs) for part in (query, key))
                rotation = None
            streams = (added, (query, key, value)) if self.context_first else ((query, key, value), added)
            query, key, value = (jnp.concatenate(pair, axis=1) for pair in zip(*streams, strict=True))
        if rotation is not None:
            query, key = (rotate(part, *rotation, pairs=self.rotary_pairs) for part in (query, key))
        attended = self.attend(query, key, value, lengths)
        if context is None:
            return (self._output("to_out_0", attended) if self.project
                    else attended.reshape(*attended.shape[:2], self.inner_features)), None
        split = context.shape[1] if self.context_first else image.shape[1]
        image_out, context_out = ((attended[:, split:], attended[:, :split]) if self.context_first
                                  else (attended[:, :split], attended[:, split:]))
        return (self._output("to_out_0", image_out),
                self._output("to_add_out", context_out) if self.context_out else None)


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

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("DoubleStreamBlock requires multisequence_flops(image_lengths, text_lengths)")

    @nn.nowrap
    def multisequence_flops(self, lengths, context_lengths) -> float:
        """The declared attention and MLPs, plus each stream's per-row modulation."""
        image, _ = matching_lengths(lengths, context_lengths)
        attention = self._attention(context_out=not self.context_pre_only)
        count = attention.stream_flops(lengths, context_lengths, self.features, self.features)
        feedforward = self._feed_forward()
        count += feedforward.model_flops(lengths)
        if not self.context_pre_only:
            count += feedforward.model_flops(context_lengths)
        if self.dual_attention:
            count += self._attention().stream_flops(lengths, None, self.features)
        if not self.shared_modulation:
            rows = np.ones(image.shape, np.int64)
            count += sum(Modulation(self.features, pieces, parent=None).model_flops(rows)
                         for pieces in self.modulation_pieces)
        return count

    @property
    def modulation_pieces(self) -> tuple[int, int]:
        return (9 if self.dual_attention else 6, 2 if self.context_pre_only else 6)

    def _modulation(self, conditioning):
        def projected(pieces: int, name: str):
            return Modulation(self.features, pieces, zero_init=self.zero_modulation, dtype=self.dtype,
                              precision=self.precision, name=name)(conditioning)

        image, context = self.modulation_pieces
        return projected(image, "norm1"), projected(context, "norm1_context")

    @nn.nowrap
    def _feed_forward(self, name: str | None = None):
        # FLUX.2's SwiGLU stores its gate and up maps as one bias-free `linear_in`;
        # the others' tanh GELU (`FeedForward(activation_fn="gelu-approximate")`) is biased.
        return GatedMLP(self.mlp_hidden or 4 * self.features, self.features,
                        activation="swiglu" if self.swiglu else "gelu", use_bias=not self.swiglu,
                        packed=self.swiglu, dtype=self.dtype, precision=self.precision, name=name)

    @nn.nowrap
    def _attention(self, name: str | None = None, *, context_out: bool = True):
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
        dropout = Dropout(self.dropout_rate, deterministic=not train)
        normalized = norm(image)
        attention = self._attention("attn", context_out=not self.context_pre_only)
        context_input = modulate(norm(context), context_shift, context_scale)
        attended, context_attended = attention(modulate(normalized, shift, scale), context_input, rotation)
        image = image + gate[:, None] * dropout(attended)
        if self.dual_attention:
            shift2, scale2, gate2 = image_mods[6:]
            attended2, _ = self._attention("attn2")(modulate(normalized, shift2, scale2))
            image = image + gate2[:, None] * dropout(attended2)
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


__all__ = ["DoubleStreamBlock", "JointAttention", "embedding", "guided_time", "layer_norm",
           "modulate"]
