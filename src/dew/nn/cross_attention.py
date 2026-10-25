"""Gated cross-attention from a conditioning sequence into a decoder's layers.

Flamingo's block (Alayrac et al. 2022, section 2.2) runs before each decoder
layer it names. The layer's residual first reads the conditioning sequence
through attention, then runs a feed-forward, and each output joins the
residual through a tanh gate:

    x = x + tanh(alpha_cross_attn) * cross_attn(input_layernorm(x), context)
    x = x + tanh(alpha_dense) * mlp(post_attention_layernorm(x))

Both gates start at zero, so a pretrained decoder with the block attached
computes its own function until the gates train. The pretrained weights stay
frozen while the block trains (`dew.interop.Pretrained.with_cross_attention`).
The reference layer is IDEFICS's `IdeficsGatedCrossAttentionLayer`
(transformers' modeling_idefics.py), and its names are the names used here.

A decoder call carries the sequence in its conditioning
(`dew.nn.inputs.CONTEXT`), with each row's count of real tokens
(`CONTEXT_LENGTHS`). A row with no tokens adds nothing through either
branch, so it computes the decoder's own function. IDEFICS still runs the
feed-forward for a token that attends to no image; here the block is absent
for such a row, as it is for a call with no context at all. A cached call
that carries the context keeps its keys and values in the cache, so later
decode steps attend over them without it.

The block reads and joins whatever residual its decoder carries: the plain
stream (AltUp's active copy among them), mHC's streams, whose mean it reads
and each of which its outputs join, Kimi K3's depth state, whose mixture it
reads and whose partial block its outputs join, and nGPT's unit stream,
which it reads unnormed and which each output joins as a step of its gate
along the sphere, as nGPT's own sublayers do.

A gate multiplies its branch as an outer product, so the gate's gradient,
the sum of the branch times its cotangent over every token, is a dot:
XLA:CPU reduces a broadcast's transpose four times further from float64
than torch's cascade sum over 640 terms, and its dot as close as torch's.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.batch import coordinates

from .activations import GatedActivation
from .attention import projected_heads, projected_output, scaled_dot_product_attention
from .flops import counted_flops, cross_attention_flops, projection_flops
from .inputs import CONTEXT, CONTEXT_LENGTHS, CrossContext
from .precision import L2NORM_EPS, at_least_fp32, float32_dot_general, l2norm, sqrt_or_zero
from .sharding import HEADS, KV_HEADS, constrain

NAME = "cross_attention"
"""The module name of a decoder layer's cross-attention block, in every collection."""

CROSS_KEY = "cross_key"
"""The cache variable of a cached call's context keys, `[rows, context_length, kv_heads, head_dim]`."""
CROSS_VALUE = "cross_value"
"""The cache variable of a cached call's context values, shaped as its keys."""
CROSS_LENGTHS = "cross_lengths"
"""The cache variable of each row's real context tokens, `[rows]`."""
CROSS_CACHE = frozenset({CROSS_KEY, CROSS_VALUE, CROSS_LENGTHS})


def crossing(path) -> bool:
    """Whether a variables path runs through a cross-attention block: the filter that trains the
    blocks alone (`dew.batch.PathFilter`)."""
    return NAME in path


def crosses_alone(trained) -> bool:
    """Whether every leaf of `trained`, the trained part of a tree split from its frozen rest
    (`dew.batch.FROZEN`), is a cross-attention block's (`crossing`)."""
    return all(crossing(tuple(entry.key for entry in path))
               for path, _ in jax.tree_util.tree_leaves_with_path(trained))


type ContextEncoder = Literal["table", "backbone"]
"""How a context of token ids becomes the states a model's cross-attention reads (`encode_context`):
`table`, its tokens through the model's own token table, or `backbone`, the model's own final states
over it in a pass of its own."""


def encode_context(model: nn.Module, variables: Mapping, tokens: jax.Array, lengths: jax.Array, *,
                   encoder: ContextEncoder, train: bool = False,
                   rngs: dict[str, jax.Array] | None = None) -> dict[str, jax.Array]:
    """The conditioning a model's cross-attention layers read (`CrossAttending`) for contexts of
    token ids: `tokens` `[B, C]`, each row's first `lengths` real, as `encoder` says.

    `table` gives each token's row of the model's own table (`token_embeddings`); `backbone` the
    model's final states over each context alone (`HiddenStates`), its real tokens one document, with
    the weights of the model's own pass, an adapter's included. Where the model trains nothing but its
    cross-attention blocks (`crosses_alone`), which a pass without a context runs none of, that pass
    trains nothing and its gradient is not taken. The states keep the dtype they are computed in, and
    the model casts them to its blocks' (`context_dtype`). `variables` are the model's own, whole or
    split from a frozen rest (`dew.objectives.base.FROZEN`). A model without cross-attention layers is
    refused (TypeError), as is a backbone pass over one without final states; another encoder is
    refused (ValueError)."""
    from dew.batch import thaw
    from dew.nn.protocols import CrossAttending, HiddenStates

    if not isinstance(model, CrossAttending) or model.cross_attention is None:
        raise TypeError(f"a context is read by cross-attention layers, which a {type(model).__name__} "
                        "does not have")
    whole = thaw(dict(variables))
    if encoder == "table":
        states = model.apply(whole, tokens, method="token_embeddings")
    elif encoder == "backbone":
        if not isinstance(model, HiddenStates):
            raise TypeError(f"a context's own pass reads final states, which a {type(model).__name__} "
                            "does not give (HiddenStates)")
        real = jnp.arange(tokens.shape[1])[None, :] < lengths[:, None]
        laid = coordinates(segments=real.astype(jnp.int32), max_segments=1, right_padded=True)
        states = model.apply(whole, tokens, train=train, coordinates=laid, method="hidden_states", rngs=rngs)
        if crosses_alone(variables.get("params", {})):
            states = jax.lax.stop_gradient(states)
    else:
        raise ValueError(f"a context is encoded by the model's table or its own pass, not {encoder!r}")
    if not isinstance(states, jax.Array):
        raise TypeError(f"expected an array of context states, got {type(states).__name__}")
    return {CONTEXT: states, CONTEXT_LENGTHS: lengths}


@dataclasses.dataclass(frozen=True)
class CrossAttention:
    """The decoder layers that read a conditioning sequence, and its geometry.

    Each layer in `layer_ids` runs the gated block before its own sublayers.
    `context_length` is the most tokens a row's context holds, which a decode
    cache keeps the keys and values of. `context_features` is the width of
    the context's tokens; None reads the decoder's own width. `intermediate_size`
    is the block's feed-forward width; None takes the width of the layer's own
    dense feed-forward. `activation` is the feed-forward's gated activation,
    IDEFICS's SiLU unless named, whatever the decoder's own feed-forward runs.
    The attention has the decoder's heads, key-value heads and head width, and
    reads the context without positions.
    """

    layer_ids: tuple[int, ...]
    context_length: int
    context_features: int | None = None
    intermediate_size: int | None = None
    activation: GatedActivation = "swiglu"

    def __post_init__(self):
        object.__setattr__(self, "layer_ids", tuple(int(layer) for layer in self.layer_ids))
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError(f"cross-attention names one layer or more, each once, got {self.layer_ids}")
        widths = {"context_length": self.context_length, "context_features": self.context_features,
                  "intermediate_size": self.intermediate_size}
        small = sorted(name for name, width in widths.items() if width is not None and width < 1)
        if small:
            raise ValueError(f"cross-attention's {', '.join(small)} must be positive, got {widths}")


class GatedCrossAttention(nn.Module):
    """One layer's gated block: a residual with its gated attention over a
    `CrossContext` and its gated feed-forward joined in.

    `norm` and `feedforward` are the decoder's factories, which take a name;
    `ngpt` reads the stream unnormed and joins each output as nGPT's step
    (the module docstring), and `norm` is then None. The projections are
    stored flat, `[inputs, heads * head_dim]`, as the decoder's own attention
    stores them. `attention_impl` and `force_fp32_for_softmax` are the
    decoder's.
    """

    norm: Callable[..., nn.Module] | None
    feedforward: Callable[..., nn.Module]
    emb_features: int
    context_features: int
    context_length: int
    heads: int
    kv_heads: int
    head_dim: int
    ngpt: bool = False
    dropout_rate: float = 0.0
    attention_impl: str = "auto"
    force_fp32_for_softmax: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def projections(self) -> dict[str, tuple[int, int]]:
        """The attention's maps as `{name: (inputs, outputs)}`."""
        queries, keys = self.heads * self.head_dim, self.kv_heads * self.head_dim
        return {"q_proj": (self.emb_features, queries), "k_proj": (self.context_features, keys),
                "v_proj": (self.context_features, keys), "o_proj": (queries, self.emb_features)}

    @nn.nowrap
    def cross_flops(self, lengths, context_lengths) -> float:
        """The block's work over rows of `lengths` tokens reading contexts of
        `context_lengths`: queries, output and feed-forward per token, keys and
        values per context token, and each row's query-by-context products."""
        maps = self.projections()
        return (projection_flops(lengths, *maps["q_proj"]) + projection_flops(lengths, *maps["o_proj"])
                + projection_flops(context_lengths, *maps["k_proj"])
                + projection_flops(context_lengths, *maps["v_proj"])
                + cross_attention_flops(lengths, context_lengths, self.heads, self.head_dim, self.head_dim)
                + counted_flops(self.feedforward(parent=None), lengths))

    @nn.compact
    def __call__(self, stream: jax.Array, context: CrossContext | None, *,
                 read: Callable[[jax.Array], jax.Array] | None = None, decode: bool = False,
                 train: bool = False) -> jax.Array:
        """`stream` after the block. `stream` is the residual as the decoder
        carries it, its rows first, and `read` gives the `[B, S, emb_features]`
        states a sublayer reads from it; None reads the stream itself."""
        read = read or (lambda held: held)
        x = read(stream)
        if context is None and self.is_initializing():
            # Without a context the key and value maps never run, and init would declare none of theirs.
            context = CrossContext(jnp.zeros((x.shape[0], 1, self.context_features), x.dtype),
                                   jnp.ones(x.shape[:1], jnp.int32))
        if decode:
            key, value, lengths = self._cached(x, context)
        elif context is None:
            return stream
        else:
            key, value = self._keys_values(context.embeddings)
            lengths = context.lengths
        query = constrain(projected_heads(self._normed("input_layernorm", x), "q_proj", self.heads,
                                          self.head_dim, flat=True, bias=False, dtype=self.dtype,
                                          precision=self.precision), HEADS)
        # A row without context reads its first slot, so its softmax stays finite: one over no keys
        # would be NaN, which the select below drops forward but not from the gradients.
        attended = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax, implementation=self.attention_impl,
            key_value_seq_lengths=jnp.maximum(lengths, 1))
        output = projected_output(constrain(attended, HEADS), "o_proj", self.emb_features, flat=True,
                                  bias=False, dtype=self.dtype, precision=self.precision)
        present = (lengths > 0).reshape((-1,) + (1,) * (stream.ndim - 1))
        dropout = nn.Dropout(rate=self.dropout_rate)

        def joined(stream: jax.Array, name: str, branch: jax.Array) -> jax.Array:
            # The gate in at least float32, and in its own float64 where a run widens it.
            alpha = self.param(name, nn.initializers.zeros, (1,), jnp.float32)
            gate = jnp.tanh(alpha.astype(at_least_fp32(alpha.dtype)))
            branch = dropout(branch, deterministic=not train)
            return jnp.where(present, gated_join(stream, branch, gate, ngpt=self.ngpt,
                                                 precision=self.precision), stream)

        stream = joined(stream, "alpha_cross_attn", output)
        hidden = self.feedforward(name="mlp")(self._normed("post_attention_layernorm", read(stream)))
        return joined(stream, "alpha_dense", hidden)

    def _normed(self, name: str, x: jax.Array) -> jax.Array:
        return x if self.norm is None else self.norm(name=name)(x)

    def _keys_values(self, embeddings: jax.Array) -> tuple[jax.Array, jax.Array]:
        """The context's keys and values, `[B, C, kv_heads, head_dim]` each."""
        if embeddings.shape[-1] != self.context_features:
            raise ValueError(f"this cross-attention reads {self.context_features}-wide context tokens, "
                             f"got {embeddings.shape[-1]}")

        def heads(name: str) -> jax.Array:
            projected = projected_heads(embeddings, name, self.kv_heads, self.head_dim, flat=True,
                                        bias=False, dtype=self.dtype, precision=self.precision)
            return constrain(projected, KV_HEADS)

        return heads("k_proj"), heads("v_proj")

    def _cached(self, x: jax.Array, context: CrossContext | None
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """The keys, values and lengths a cached call reads. A call that carries a context
        computes them from it and, once the cache is allocated, keeps them; a call without
        one reads what the cache keeps. As with the decoder's own cache, the first cached
        call allocates the cache and writes nothing."""
        allocated = self.has_variable("cache", CROSS_KEY)
        dtype = x.dtype if self.dtype is None else self.dtype
        shape = (x.shape[0], self.context_length, self.kv_heads, self.head_dim)
        keys = self.variable("cache", CROSS_KEY, jnp.zeros, shape, dtype)
        values = self.variable("cache", CROSS_VALUE, jnp.zeros, shape, dtype)
        lengths = self.variable("cache", CROSS_LENGTHS, jnp.zeros, shape[:1], jnp.int32)
        if context is None:
            return keys.value, values.value, lengths.value
        width = context.embeddings.shape[1]
        if width > self.context_length:
            raise ValueError(f"a cached call keeps at most {self.context_length} context tokens a row "
                             f"(CrossAttention.context_length), got {width}")
        # The prefill reads the keys and values as the cache keeps them, so its steps read the same.
        key, value = (projected.astype(dtype) for projected in self._keys_values(context.embeddings))
        if allocated:
            padding = ((0, 0), (0, self.context_length - width), (0, 0), (0, 0))
            keys.value = jnp.pad(key, padding)
            values.value = jnp.pad(value, padding)
            lengths.value = context.lengths
        return key, value, context.lengths


def gated_join(stream: jax.Array, branch: jax.Array, gate: jax.Array, *, ngpt: bool = False,
               precision: PrecisionLike = None) -> jax.Array:
    """`stream` with `branch` joined through `gate`, in at least float32 of the three and rounded once
    to the stream's dtype: added, broadcast over any axis the stream has beside the states', or with
    `ngpt` moved by the step of `gate` from the unit stream toward the unit branch, nGPT's step on the
    unit sphere, at the stream's own length (`_moved_on_the_sphere`). The stream is scaled by its
    largest element first, so no square of a finite stream overflows, and a zero gate leaves it bit for
    bit, whatever its length: before a first layer the input embeddings are not unit."""
    wide = stream.astype(at_least_fp32(jnp.result_type(stream, branch, gate)))
    if ngpt:
        # The move scales back by the length of the scaled stream, then by its largest element, so a
        # zero move adds zero.
        peak = jnp.max(jnp.abs(wide), axis=-1, keepdims=True)
        scaled = wide / jnp.where(peak > 0, peak, 1)
        length = sqrt_or_zero(jnp.sum(jnp.square(scaled), axis=-1, keepdims=True))
        start, end = l2norm(scaled), l2norm(branch).astype(wide.dtype)
        move = _moved_on_the_sphere(start, _gated(end - start, gate, precision))
        return (wide + peak * (length * move)).astype(stream.dtype)
    widened = branch.reshape(branch.shape[:2] + (1,) * (stream.ndim - branch.ndim) + branch.shape[2:])
    return (wide + _gated(widened, gate, precision)).astype(stream.dtype)


def _moved_on_the_sphere(start: jax.Array, step: jax.Array) -> jax.Array:
    """`l2norm(start + step) - l2norm(start)` along the last axis, written so a zero step moves by
    exactly zero: with N = |start|^2 + eps and M = N + 2<start, step> + |step|^2 the two points' squared
    lengths (`L2NORM_EPS`), the difference is step / sqrt(M) less start times (M - N) / (sqrt(M)
    sqrt(N) (sqrt(M) + sqrt(N))), and M - N is formed from the step alone."""
    inner = jnp.sum(start * step, axis=-1, keepdims=True)
    grown = 2 * inner + jnp.sum(jnp.square(step), axis=-1, keepdims=True)
    before = jnp.sum(jnp.square(start), axis=-1, keepdims=True) + L2NORM_EPS
    after = before + grown
    root_before, root_after = jnp.sqrt(before), jnp.sqrt(after)
    return step / root_after - start * grown / (root_after * root_before * (root_after + root_before))


def _gated(branch: jax.Array, gate: jax.Array, precision: PrecisionLike) -> jax.Array:
    """`branch` times the one-element `gate`, both in at least float32, as
    their outer product, whose transpose gives the gate a dot (the module
    docstring)."""
    wide = jnp.promote_types(at_least_fp32(branch.dtype), gate.dtype)
    product = float32_dot_general(branch.astype(wide), gate.astype(wide), (((), ()), ((), ())),
                                  precision=precision)
    return product[..., 0]


__all__ = ["CROSS_CACHE", "CROSS_KEY", "CROSS_LENGTHS", "CROSS_VALUE", "NAME", "ContextEncoder",
           "CrossAttention", "GatedCrossAttention", "crossing", "encode_context", "gated_join"]
