"""A `CausalTransformer`'s block drafter, DeepSeek's DSpark (`dew.nn.dspark`) or a Markov block drafter
(`dew.nn.block_draft`, Inkling 2's): the drafting stages DSpark builds (`drafting_stages`), the context the
target layers sow, the context's real positions, a block drafted after it, and the call that reaches the
drafter's parameters, which no forward does. The decoder's methods of these names read them here, within
its scope."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from dew.batch import COORDINATES_FIELD, Coordinates, real_slots

from ..block_draft import Draft, draft as markov_draft, tapped
from ..deepseek_v4 import DeepseekV4Mixer
from ..dspark import DSparkStage, draft as dspark_draft
from .decoder_block import BlockWiring, DecoderBlock
from .layer_plan import ResolvedKind

if TYPE_CHECKING:
    from .causal_transformer import CausalTransformer

def drafting_stages(decoder: CausalTransformer, kinds: dict[str, ResolvedKind], mixer_spec, routed,
                    wiring: BlockWiring) -> list[DSparkStage]:
    """DSpark's drafting stages, attending with the V4 attention of their
    layer kind and routing through the trunk's experts."""
    assert decoder.dspark is not None
    layer_type = decoder.dspark.layer_type
    drafting = kinds[layer_type].mixer or mixer_spec
    if not isinstance(drafting, DeepseekV4Mixer):
        raise ValueError(f"DSpark's stages attend with V4 attention, and kind {layer_type!r} "
                         f"builds {type(drafting).__name__}")
    drafter = drafting.drafter(decoder.mixer_context(kinds[layer_type], layer_type, kv_shared=False))
    stages, hc = decoder.dspark.stages, decoder.hyper_connections
    assert hc is not None
    return [
        DSparkStage(
            block=functools.partial(
                DecoderBlock, mixer=drafter,
                feedforward=functools.partial(routed, num_experts=decoder.dspark.experts,
                                              top_k=decoder.dspark.top_k),
                emb_features=decoder.emb_features, norm_eps=decoder.norm_eps,
                scale_offset=decoder.scale_offset, scale_after_cast=decoder.scale_after_cast,
                wiring=wiring, hyper_connections=decoder.hyper_connections,
                dtype=decoder.dtype, precision=decoder.precision),
            emb_features=decoder.emb_features, vocab_size=decoder.vocab_size,
            markov_rank=decoder.dspark.markov_rank,
            first=stage == 0, last=stage == stages - 1, norm_eps=decoder.norm_eps,
            weighted_head=None if hc.single_pass else hc,
            dtype=decoder.dtype, precision=decoder.precision, name=f'dspark_{stage}')
        for stage in range(stages)]


def draft(decoder: CausalTransformer, context, tokens, *, decode: bool = True,
          coordinates: Coordinates | None = None, choose=None) -> Draft | None:
    """Draft a block after each row's last context position.

    `context` `[B, M, targets * D]` comes from `draft_context` (what the
    target layers sow), its `coordinates` mark its real
    positions (`real_slots`: its validity less a padding document, every
    one where None; one document a row, as a draft continues one), and
    `tokens` `[B]` are the
    tokens drawn after it. Returns the block (`Draft`): its ids
    `[B, block + 1]`, logits `[B, block, vocab]` and DSpark's confidence
    `[B, block]` (a Markov block drafter's None). Tokens are chosen by
    `choose(index, logits)`, or greedily when `choose` is None. For a cached
    draft, call `init_draft_cache` first. With `tokens` None the call only
    appends context, and with `context` None it drafts after what the
    cache already holds. A model without a drafter raises `ValueError`.
    """
    valid = context_validity(decoder, context, coordinates)
    if decoder.block_draft is not None:
        return markov_draft(decoder.block_drafter, context, valid, tokens, decoder.anchor_embeddings,
                            decoder.head_scores, decoder.final_logit_softcap, decode=decode, choose=choose)
    if decoder.dspark is None:
        raise ValueError("this model has no block drafter")
    hc = decoder.hyper_connections
    assert hc is not None
    return dspark_draft(decoder.dspark_stages, decoder.dspark, decoder.stream_embeddings,
                        decoder.logits_from_hidden, hc, context, tokens, decode=decode, valid=valid,
                        choose=choose)


def reach(decoder: CausalTransformer, tokens, dtype):
    """Create the block drafter's parameters, which no forward pass reaches.

    It drafts after `tokens` over a zero context as wide as the target's
    taps it reads.
    """
    width = len(tap_layers(decoder)) * decoder.emb_features
    draft(decoder, jnp.zeros((tokens.shape[0], 1, width), dtype), tokens[:, -1], decode=False)


def context_validity(decoder: CausalTransformer, context, coordinates: Coordinates | None
                     ) -> jax.Array | None:
    """The draft context's real positions `[B, M]` (`draft`): its layout's real slots, a padding
    document's left out (`real_slots`), None where every one is; a context of several documents a
    row is refused, since a draft continues one."""
    if coordinates is None or context is None:
        return None
    if coordinates.segments is not None and coordinates.max_segments > 1:
        raise ValueError("a draft continues one document a row, and this context holds several")
    return real_slots({COORDINATES_FIELD: coordinates}, tuple(context.shape[:2]))


def draft_context(decoder: CausalTransformer, prediction_inputs: Mapping) -> jax.Array:
    """Return the block drafter's context `[B, S, targets * D]` from `prediction_inputs`.

    A forward pass sows `prediction_inputs` when the caller makes that
    collection mutable. DSpark's context is each target layer's stream mean,
    concatenated in `target_layers` order; a Markov block drafter's each tapped
    visit's record, in the visits' order, then the final norm's output
    (`MarkovBlockDraft.taps`).
    """
    if decoder.block_draft is not None:
        records = [prediction_inputs[f'layers_{layer}']['visits'][run]
                   for layer, run in tapped(decoder.visits(), decoder.block_draft.visits)]
        return jnp.concatenate([*records, prediction_inputs[FINAL_STATES]], axis=-1)
    assert decoder.dspark is not None
    return jnp.concatenate([prediction_inputs[f'layers_{layer}']['draft_context']
                            for layer in decoder.dspark.target_layers], axis=-1)


FINAL_STATES = 'final_states'
"""The `prediction_inputs` leaf a Markov block drafter's target sows its final norm's output under, its
last tap (`CausalTransformer._exited`)."""


def tap_layers(decoder: CausalTransformer) -> tuple[int, ...]:
    """The target's taps the drafter reads, each a layer's index (DSpark's target layers) or, for a Markov
    block drafter, each tapped visit's layer, its final states standing for its last (-1)."""
    if decoder.block_draft is not None:
        order = decoder.visits()
        return (*(order[visit] for visit in decoder.block_draft.visits), -1)
    assert decoder.dspark is not None
    return decoder.dspark.target_layers
