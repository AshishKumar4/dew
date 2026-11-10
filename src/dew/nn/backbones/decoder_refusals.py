"""What a causal transformer refuses to run (`CausalTransformer.mixed_admission_refusal` and its kin): a
serving step's mixed call, a generation loop, a prompt split into pieces, and a packed row's cached call,
each read off its layers' mixers and its stack's fields."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mixer_base import MixerBase, MixerContext
    from .causal_transformer import CausalTransformer


def mixed_admission(decoder: CausalTransformer) -> str | None:
    """A cache rebuilt per request, a reader beyond the token, a context
    read beside the tokens, or a layer whose mixer, over the context it is
    built with, would not run the mixed call as the separate calls would
    (`MixerBase.mixed_refusal`)."""
    position = decoder.cache_rebuild_position
    if position is not None:
        return (f'LongRoPE crossing position {position} requires separate admission so each '
                'request keeps its own table and rebuild history')
    if decoder.cross_attention is not None:
        return ("cross-attention reads each row's own context, which the mixed call's one row of "
                "every slot's tokens does not seat")
    if decoder.position_embedding == "learned":
        return ("a learned position embedding's table is read in the decoder's own row layout, not the "
                "mixed row's")
    if decoder.engram is not None:
        return ("an n-gram reads the tokens before each in its row, which the mixed call's row of every "
                "slot's tokens holds apart")
    return _layer(decoder, lambda mixer, context: mixer.mixed_refusal(context))


def generation(decoder: CausalTransformer) -> str | None:
    """Why a generation loop cannot continue the decoder's rows, or None
    (`dew.nn.protocols.Generating`): its sparse layers' mixture, or a mixer it
    names (`declared_mixers`: a layer kind's, a prediction depth's or the
    drafter's), that refuses it (`Mixture.generation_refusal`,
    `MixerBase.generation_refusal`)."""
    refusal = None if decoder.mixture is None else decoder.mixture.generation_refusal()
    for mixer in decoder.declared_mixers:
        refusal = refusal or mixer.generation_refusal()
    return refusal


def split_prefill(decoder: CausalTransformer) -> str | None:
    """A cache rebuilt per request from its whole prompt, or a loop's shared cache that a later
    pass rewrites, whose earlier tokens hold the last pass's keys: a prompt's later piece would
    read those where its whole prefill reads each pass's own. One pass, or a block whose layers
    keep no cache (`MixerBase.keeps_cache`), rewrites nothing."""
    position = decoder.cache_rebuild_position
    if position is not None:
        return (f'LongRoPE crossing position {position} rebuilds a request\'s cache from its whole '
                'prompt; serve it with whole-prompt dense admission')
    loop = decoder.loop
    first, end = (0, decoder.num_layers) if loop is None or loop.layers is None else loop.layers
    rewritten = any((decoder.kind_of(kind).mixer or decoder.layer_mixer).keeps_cache
                    for kind in decoder.per_layer_types[first:end])
    if loop is not None and loop.shared_cache and loop.steps > 1 and rewritten:
        return ("a loop's shared cache holds only the last pass's keys of the tokens before a "
                "call, where the whole prompt's prefill reads each pass's own; keep a cache a "
                "pass (Loop.shared_cache=False, Dory's recurrent_kv_cache_mode 'per_loop') for "
                "chunked prefill or prefix reuse")
    return None


def packed_cache(decoder: CausalTransformer) -> str | None:
    """Why a cached call of several documents a row would read them together, or None: each layer reads its
    own document's keys over its cache (`MixerBase.packed_cache_refusal`, `dew.nn.kv_cache.DOCUMENTS`), a
    call's documents numbered from 1, its first continuing its row's last (`dew.batch.Coordinates.starts`),
    its padding cached never; a step naming none reads every key its row holds. Engram's cached lookback
    reads the n-grams before a token by place, across a document's start."""
    if decoder.engram is not None:
        return "Engram's n-gram lookback, which reads across a cached row's documents"
    return _layer(decoder, lambda mixer, _: mixer.packed_cache_refusal())


def forked(decoder: CausalTransformer, train: bool = False, *, joint: bool = False) -> str | None:
    """Why a call over forks (`dew.batch.Forks`) would not compute each document over its ancestors as its
    own row does, or None (`dew.nn.protocols.ForkedCalls`): a bidirectional decoder, whose shared prefix
    reads each branch's tokens; in training, dropout, which draws each row's
    masks apart where a shared prefix draws them once; engram's n-gram hashes (`Engram.fork_refusal`),
    its mixture's block routing or capacity (`Mixture.fork_refusal`), or a layer whose mixer reads no
    forks (`MixerBase.fork_refusal`). Attention, the recurrent mixers and tokenwise layers read each
    document from its ancestors alone; a learned position embedding reads the places forks give each
    token; cross-attention reads each tree's own context (`Forks.trees`), a context a tree."""
    if not decoder.causal:
        return "a bidirectional decoder reads each token's later tokens, which a fork's branches hold apart"
    # A joint tree is one row, its masks drawn once for it as its own row draws them (`Forks.joint`).
    if train and not joint and (decoder.dropout_rate or decoder.embedding_dropout_rate
                                or decoder.attention_dropout_rate):
        return "dropout draws each row's masks apart, which a shared prefix draws once"
    if decoder.engram is not None:
        return decoder.engram.fork_refusal()
    if decoder.mixture is not None and (refusal := decoder.mixture.fork_refusal()) is not None:
        return refusal
    return _layer(decoder, lambda mixer, _: mixer.fork_refusal())


def _layer(decoder: CausalTransformer, refused: Callable[[MixerBase, MixerContext], str | None]
           ) -> str | None:
    """The first of the decoder's layers whose mixer, over the context it is built with, `refused`
    (`CausalTransformer.layer_mixers`), as "layer i: why", or None."""
    for index, (mixer, context) in enumerate(decoder.layer_mixers()):
        if (refusal := refused(mixer, context)) is not None:
            return f"layer {index}: {refusal}"
    return None
