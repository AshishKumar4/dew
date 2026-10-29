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
Its attention is the one every cross-attention runs, a diffusion block's and
Wan's alike (`dew.nn.attention_blocks.NormalAttention`), held in the block's own
scope, over the decoder's grouped key-value heads and keeping a cached call's
keys and values; the gates, the feed-forward and the residual join are the
block's.

A decoder call carries the sequence in its conditioning
(`dew.nn.inputs.CONTEXT`), with each row's count of real tokens
(`CONTEXT_LENGTHS`). A row with no tokens adds nothing through either
branch, so it computes the decoder's own function. IDEFICS still runs the
feed-forward for a token that attends to no image; here the block is absent
for such a row, as it is for a call with no context at all. A cached call
that carries the context keeps its keys and values in the cache, so later
decode steps attend over them without it.

That is one form of a decoder's cross-attention (`CrossBlocks`, which a decoder's
`with_cross_attention` takes): `CrossAttention` names it. `SharedCrossAttention`
is another, which adds no attention of its own: each named layer reads the
conditioning sequence through its own attention's projections and norms,
unrotated and uncausal, and joins the read through one tanh gate, zero at the
start, the only weight it adds:

    x = x + tanh(alpha_cross_attn) * self_attn.read(input_layernorm(x), context)

Any configuration that builds a conforming block per layer is a third.

A block reads and joins whatever residual its decoder carries: the plain
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
import functools
from collections.abc import Callable, Hashable, Mapping
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.batch import coordinates
from dew.nn.dropout import Dropout

from .activations import GatedActivation
from .attention import AttentionImpl
from .attention_blocks import CROSS_KEY, CROSS_VALUE, NormalAttention, kept_context
from .documents import CrossDocuments, holds_keys
from .flops import counted_flops
from .inputs import CONTEXT, CONTEXT_LENGTHS, CONTEXT_SEGMENTS, CrossContext
from .precision import L2NORM_EPS, at_least_fp32, float32_dot_general, l2norm, sqrt_or_zero

if TYPE_CHECKING:
    from dew.nn.backbones.causal_transformer import CausalTransformer

NAME = "cross_attention"
"""The module name of a decoder layer's cross-attention block, in every collection."""

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
                   rngs: dict[str, jax.Array] | None = None, segments: jax.Array | None = None,
                   max_segments: int = 1) -> dict[str, jax.Array]:
    """The conditioning a model's cross-attention layers read (`CrossAttending`) for contexts of
    token ids: `tokens` `[B, C]`, each row's first `lengths` real, as `encoder` says, and where a row
    packs the contexts of up to `max_segments` documents, each token's document, `segments` `[B, C]`
    (`CONTEXT_SEGMENTS`), which the backbone pass reads each of alone and the call's own tokens of each
    document read alone.

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
        documents = real.astype(jnp.int32) if segments is None else jnp.where(real, segments, 0)
        laid = coordinates(segments=documents, max_segments=max_segments if segments is not None else 1,
                           right_padded=True)
        states = model.apply(whole, tokens, train=train, coordinates=laid, method="hidden_states", rngs=rngs)
        if crosses_alone(variables.get("params", {})):
            states = jax.lax.stop_gradient(states)
    else:
        raise ValueError(f"a context is encoded by the model's table or its own pass, not {encoder!r}")
    if not isinstance(states, jax.Array):
        raise TypeError(f"expected an array of context states, got {type(states).__name__}")
    return {CONTEXT: states, CONTEXT_LENGTHS: lengths,
            **({} if segments is None else {CONTEXT_SEGMENTS: segments})}


class CrossBlocks(Protocol):
    """A decoder's cross-attention, which `with_cross_attention` takes: the layers that read a
    conditioning sequence, the most tokens a row's context holds, which a decode cache keeps the keys
    and values of, the width of its tokens (None for the decoder's own), and each layer's block, built
    from the decoder. A block runs before its layer's own sublayers, reads the residual and the
    context and joins the residual (`GatedCrossAttention`'s call); one that reads through its layer's
    own attention says so (`reads_layer`) and is handed it (`LayerReads`). `CrossAttention` (IDEFICS's
    gated block) and `SharedCrossAttention` (each layer's own projections behind one gate) are two;
    any conforming configuration is a third."""

    @property
    def layer_ids(self) -> tuple[int, ...]: ...

    @property
    def context_length(self) -> int: ...

    @property
    def context_features(self) -> int | None: ...

    def block(self, decoder: CausalTransformer, index: int) -> Callable[..., nn.Module]:
        """Layer `index`'s block factory, which takes its module `name` (and flax's `parent`),
        refusing a layer the block cannot read the context at."""
        ...

    def layer_key(self, decoder: CausalTransformer, index: int) -> Hashable:
        """What layer `index`'s block depends on beside the decoder's fields and this configuration:
        layers of equal specs and equal keys build one block, which a scanned run of them runs for each
        (`dew.nn.backbones.layer_plan.LayerSpec.cross`)."""
        ...

    def flops(self, decoder: CausalTransformer, index: int, lengths, context_lengths) -> float:
        """Layer `index`'s block's forward work over rows of `lengths` tokens reading contexts of
        `context_lengths` tokens."""
        ...


@runtime_checkable
class AttendsContext(Protocol):
    """A layer's own attention as a read of another sequence (`SharedCrossAttention`): whether it
    projects keys and values of its own to read one with, the key heads' count and the widths of a
    key and a value head, its query heads over the layer's input, its key and value heads over a
    context, its attention of the one over the other through its output projection, and that read's
    work (`dew.nn.mixers.attention.CausalSelfAttention`)."""

    @property
    def reads_context(self) -> bool: ...

    def context_heads(self) -> tuple[int, int, int]: ...

    def context_flops(self, lengths, context_lengths) -> float: ...

    def context_query(self, x: jax.Array) -> tuple[jax.Array, jax.Array | None]: ...

    def context_keys(self, context: jax.Array) -> tuple[jax.Array, jax.Array]: ...

    def attend_context(self, query: jax.Array, gate: jax.Array | None, key: jax.Array,
                       value: jax.Array, *, lengths: jax.Array, queries: jax.Array | None = None,
                       segments: jax.Array | None = None, max_segments: int | None = None) -> jax.Array: ...


class LayerReads(NamedTuple):
    """What a block that reads through its layer (`reads_layer`) is handed: the layer's attention
    input from the states a sublayer reads, normed as its attention reads it, and its attention."""

    normed: Callable[[jax.Array], jax.Array]
    attention: AttendsContext


def named_layers(layer_ids) -> tuple[int, ...]:
    """The layers a cross-attention names as ints, refused unless one or more, each once: every
    `CrossBlocks` configuration's."""
    named = tuple(int(layer) for layer in layer_ids)
    if not named or len(set(named)) != len(named):
        raise ValueError(f"cross-attention names one layer or more, each once, got {named}")
    return named


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
        object.__setattr__(self, "layer_ids", named_layers(self.layer_ids))
        widths = {"context_length": self.context_length, "context_features": self.context_features,
                  "intermediate_size": self.intermediate_size}
        small = sorted(name for name, width in widths.items() if width is not None and width < 1)
        if small:
            raise ValueError(f"cross-attention's {', '.join(small)} must be positive, got {widths}")

    def block(self, decoder: CausalTransformer, index: int) -> Callable[..., GatedCrossAttention]:
        """Layer `index`'s gated block (`CrossBlocks.block`): the decoder's norms, none under nGPT, a
        gated MLP of the block's own activation at its width (the layer's own unless named), with nGPT's
        gains under nGPT, and the decoder's attention heads, kernel, initializers and dtypes. A layer
        with no feed-forward width to take needs `intermediate_size`."""
        from .backbones.decoder_block import decoder_norm
        from .mlp import GatedMLP

        width = self.intermediate_size or decoder.mlp_widths[index]
        if not width:
            raise ValueError(f"layer {index} has no feed-forward whose width the cross-attention block "
                             "takes; name its intermediate_size")
        init_std, output_init_std = decoder.init_stds
        return functools.partial(
            GatedCrossAttention,
            norm=None if decoder.ngpt is not None else decoder_norm(
                decoder.norm_type, epsilon=decoder.norm_eps, bias=decoder.norm_bias,
                scale_offset=decoder.scale_offset, scale_after_cast=decoder.scale_after_cast,
                dtype=decoder.dtype, specs=decoder.norms),
            ngpt=decoder.ngpt is not None,
            feedforward=functools.partial(
                GatedMLP, hidden_features=width, out_features=decoder.emb_features,
                activation=self.activation, ngpt=decoder.ngpt, init_std=init_std,
                output_init_std=output_init_std, dtype=decoder.dtype, precision=decoder.precision),
            emb_features=decoder.emb_features, context_features=decoder.context_width,
            context_length=self.context_length, heads=decoder.num_heads, kv_heads=decoder.kv_heads,
            head_dim=decoder.features_per_head, dropout_rate=decoder.dropout_rate,
            attention_impl=decoder.attention_impl, force_fp32_for_softmax=decoder.force_fp32_for_softmax,
            dtype=decoder.dtype, precision=decoder.precision)

    def layer_key(self, decoder: CausalTransformer, index: int) -> Hashable:
        """The width of layer `index`'s block's feed-forward, all its block takes of the layer."""
        return self.intermediate_size or decoder.mlp_widths[index]

    def flops(self, decoder: CausalTransformer, index: int, lengths, context_lengths) -> float:
        """Layer `index`'s block's work (`GatedCrossAttention.cross_flops`)."""
        return self.block(decoder, index)(parent=None).cross_flops(lengths, context_lengths)


@dataclasses.dataclass(frozen=True)
class SharedCrossAttention:
    """Cross-attention through each named layer's own attention: before its sublayers, the layer in
    `layer_ids` reads the conditioning sequence with its own query, key, value and output projections
    and norms, its logit scale, sinks and softcap, unrotated and uncausal
    (`AttendsContext`), and the read joins the residual through one tanh gate the layer adds, zero
    at the start, so the decoder computes its own function until the gates train. The only weights
    it adds are the gates; the context's tokens are the decoder's width, which the layer's key and
    value projections read. `context_length` is the most tokens a row's context holds, which a
    decode cache keeps the keys and values of. A layer of no attention, or a KV-sharing one, projects
    no keys of its own and is refused."""

    layer_ids: tuple[int, ...]
    context_length: int

    def __post_init__(self):
        object.__setattr__(self, "layer_ids", named_layers(self.layer_ids))
        if self.context_length < 1:
            raise ValueError(f"cross-attention's context_length must be positive, got {self.context_length}")

    @property
    def context_features(self) -> None:
        """None: the context is the decoder's width, as each layer's own projections read it."""
        return None

    def block(self, decoder: CausalTransformer, index: int) -> Callable[..., SharedGatedCrossAttention]:
        """Layer `index`'s gate over its own attention's read (`CrossBlocks.block`), refused where
        the layer's mixer reads no context (`AttendsContext`)."""
        mixer = self._attention(decoder, index)
        return functools.partial(SharedGatedCrossAttention, context_length=self.context_length,
                                 heads=mixer.context_heads(), ngpt=decoder.ngpt is not None,
                                 dropout_rate=decoder.dropout_rate, dtype=decoder.dtype,
                                 precision=decoder.precision)

    def layer_key(self, decoder: CausalTransformer, index: int) -> Hashable:
        """Nothing: a layer's block takes its geometry from the layer's own attention, which the layer's
        spec builds, and reads through the attention the layer hands it."""
        return ()

    def flops(self, decoder: CausalTransformer, index: int, lengths, context_lengths) -> float:
        """Layer `index`'s attention's read of the context (`AttendsContext.context_flops`)."""
        return self._attention(decoder, index).context_flops(lengths, context_lengths)

    @staticmethod
    def _attention(decoder: CausalTransformer, index: int) -> AttendsContext:
        mixer = decoder.mixer_of(index)
        if not isinstance(mixer, AttendsContext) or not mixer.reads_context:
            raise ValueError(f"layer {index}'s mixer ({type(mixer).__name__}) projects no keys and values of "
                             "its own to read a context with")
        return mixer


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

    reads_layer: ClassVar[bool] = False
    """It reads the context with an attention of its own (`CrossBlocks`)."""

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
    attention_impl: AttentionImpl = "auto"
    force_fp32_for_softmax: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def attention(self, **parent) -> NormalAttention:
        """The block's attention over its context: the attention every cross-attention runs
        (`NormalAttention`), its projections flat under the decoder's names, grouped over the
        decoder's key-value heads, keeping a cached call's keys and values. `parent` is flax's: None for
        the unbound description its maps and work are read from."""
        return NormalAttention(self.emb_features, self.heads, self.head_dim, dtype=self.dtype,
                               precision=self.precision, use_bias=False,
                               force_fp32_for_softmax=self.force_fp32_for_softmax,
                               attention_impl=self.attention_impl, flat=True,
                               names=("q_proj", "k_proj", "v_proj", "o_proj"), kv_heads=self.kv_heads,
                               cache_length=self.context_length, **parent)

    @nn.nowrap
    def projections(self) -> dict[str, tuple[int, int]]:
        """The attention's maps as `{name: (inputs, outputs)}`."""
        return self.attention(parent=None).projections(self.context_features)

    @nn.nowrap
    def cross_flops(self, lengths, context_lengths) -> float:
        """The block's work over rows of `lengths` tokens reading contexts of
        `context_lengths`: the attention's (`NormalAttention.cross_flops`) and
        the feed-forward's per token."""
        return (self.attention(parent=None).cross_flops(lengths, context_lengths, self.context_features)
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
        if context is not None and context.embeddings.shape[-1] != self.context_features:
            raise ValueError(f"this cross-attention reads {self.context_features}-wide context tokens, "
                             f"got {context.embeddings.shape[-1]}")
        if decode:
            lengths = _held_lengths(self, x, context)
        elif context is None:
            return stream
        else:
            lengths = context.lengths
        attention = self.attention()
        nn.share_scope(self, attention)
        # A row without context reads its first slot, so its softmax stays finite: one over no keys
        # would be NaN, which the select below drops forward but not from the gradients.
        embeddings = None if context is None else context.embeddings
        # A context of several documents is read by each document's own tokens alone (`CrossContext`).
        documents = ({} if context is None or context.segments is None
                     else {"segment_ids": context.queries, "key_segment_ids": context.segments})
        output = attention(self._normed("input_layernorm", x), embeddings,
                           key_value_seq_lengths=jnp.maximum(lengths, 1), cached=decode,
                           cache_dtype=x.dtype if self.dtype is None else self.dtype, **documents)
        present = _present(stream, context, lengths)
        dropout = Dropout(rate=self.dropout_rate)

        def joined(stream: jax.Array, name: str, branch: jax.Array) -> jax.Array:
            return _joined(self, name, stream, dropout(branch, deterministic=not train), present,
                           ngpt=self.ngpt, precision=self.precision)

        stream = joined(stream, "alpha_cross_attn", output)
        hidden = self.feedforward(name="mlp")(self._normed("post_attention_layernorm", read(stream)))
        return joined(stream, "alpha_dense", hidden)

    def _normed(self, name: str, x: jax.Array) -> jax.Array:
        return x if self.norm is None else self.norm(name=name)(x)


class SharedGatedCrossAttention(nn.Module):
    """One layer's read of a conditioning sequence through its own attention
    (`SharedCrossAttention`): the residual's states normed as the layer's attention reads them, its
    query heads over them and its key and value heads over the context (`AttendsContext`), attended
    and projected by the layer's own weights, the read joined through one tanh gate,
    `alpha_cross_attn`, zero at the start. `heads` is the layer's key heads' count and the widths of
    a key and a value head, which a decode cache keeps the context's keys and values at; a cached call
    keeps them and each row's count, so later steps read them without the context. `ngpt` joins the
    read as nGPT's step along the sphere (`gated_join`)."""

    reads_layer: ClassVar[bool] = True
    """It reads the context through its layer's own attention (`CrossBlocks`, `LayerReads`)."""

    context_length: int
    heads: tuple[int, int, int]
    ngpt: bool = False
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, stream: jax.Array, context: CrossContext | None, *, layer: LayerReads,
                 read: Callable[[jax.Array], jax.Array] | None = None, decode: bool = False,
                 train: bool = False) -> jax.Array:
        """`stream` after the block, as `GatedCrossAttention`'s call, its layer's attention input and
        attention handed it (`layer`)."""
        read = read or (lambda held: held)
        x = read(stream)
        if context is None and self.is_initializing():
            # Without a context the gate never joins a read, and init would declare none.
            context = CrossContext(jnp.zeros((x.shape[0], 1, x.shape[-1]), x.dtype),
                                   jnp.ones(x.shape[:1], jnp.int32))
        if context is not None and context.embeddings.shape[-1] != x.shape[-1]:
            raise ValueError(f"a layer's own projections read {x.shape[-1]}-wide context tokens, the "
                             f"decoder's width, got {context.embeddings.shape[-1]}")
        if decode:
            lengths = _held_lengths(self, x, context)
        elif context is None:
            return stream
        else:
            lengths = context.lengths
        attention = layer.attention
        query, gate = attention.context_query(layer.normed(x))
        embeddings = None if context is None else context.embeddings
        if decode:
            key_heads, key_width, value_width = self.heads
            key, value = kept_context(self, x.shape[0], self.context_length,
                                      ((key_heads, key_width), (key_heads, value_width)),
                                      x.dtype if self.dtype is None else self.dtype, embeddings,
                                      attention.context_keys)
        else:
            assert embeddings is not None
            key, value = attention.context_keys(embeddings)
        documents = ({} if context is None or context.segments is None
                     else {"queries": context.queries, "segments": context.segments})
        branch = attention.attend_context(query, gate, key, value, lengths=lengths, **documents)
        branch = Dropout(rate=self.dropout_rate)(branch, deterministic=not train)
        return _joined(self, "alpha_cross_attn", stream, branch, _present(stream, context, lengths),
                       ngpt=self.ngpt, precision=self.precision)


def _held_lengths(module: nn.Module, x: jax.Array, context: CrossContext | None) -> jax.Array:
    """Each row's real context tokens a cached call of a block reads: a call that carries a context
    keeps them once the cache is allocated, as its attention keeps the keys and values
    (`kept_context`); a call without one reads what the cache keeps."""
    allocated = module.has_variable("cache", CROSS_LENGTHS)
    lengths = module.variable("cache", CROSS_LENGTHS, jnp.zeros, x.shape[:1], jnp.int32)
    if context is None:
        return lengths.value
    if allocated:
        lengths.value = context.lengths
    return context.lengths


def _present(stream: jax.Array, context: CrossContext | None, lengths: jax.Array) -> jax.Array:
    """Where a block's read joins `stream`: each row with context tokens and, where the context holds
    several documents, each token whose document holds some, as its row alone would skip the block."""
    if context is not None and context.segments is not None and context.queries is not None:
        held = holds_keys(CrossDocuments(context.queries, context.segments))
        return held.reshape(held.shape[:2] + (1,) * (stream.ndim - 2))
    return (lengths > 0).reshape((-1,) + (1,) * (stream.ndim - 1))


def _joined(module: nn.Module, name: str, stream: jax.Array, branch: jax.Array, present: jax.Array, *,
            ngpt: bool, precision: PrecisionLike) -> jax.Array:
    """`stream` with `branch` joined through `module`'s tanh gate `name`, a parameter of one element,
    zero at the start, in at least float32 and in its own float64 where a run widens it, where
    `present` (`gated_join`)."""
    alpha = module.param(name, nn.initializers.zeros, (1,), jnp.float32)
    gate = jnp.tanh(alpha.astype(at_least_fp32(alpha.dtype)))
    return jnp.where(present, gated_join(stream, branch, gate, ngpt=ngpt, precision=precision), stream)


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


__all__ = ["CROSS_CACHE", "CROSS_KEY", "CROSS_LENGTHS", "CROSS_VALUE", "NAME", "AttendsContext",
           "ContextEncoder", "CrossAttention", "CrossBlocks", "GatedCrossAttention", "LayerReads",
           "SharedCrossAttention", "SharedGatedCrossAttention", "crossing", "encode_context", "gated_join"]
