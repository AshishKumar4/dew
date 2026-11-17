"""A `CausalTransformer`'s multi-token prediction depths: the depths it builds (`depths`), the
mixer kind they mirror (`prediction_kind`), their shifted states and logits over a whole sequence, and
one unshifted step a draft chains. The decoder's methods of these names read them here, within its
scope."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from dew.batch import Coordinates, coordinates as build_coordinates, shifted_span
from dew.nn.inspection import current_positions, watched_positions

from ..inputs import AttentionMetadata, PredictionPhase
from ..mixers.mlp import MLPMixer
from .decoder_block import BlockWiring, MTPBlock
from .embeddings import prepared_embeddings
from .layer_plan import ResolvedKind

if TYPE_CHECKING:
    from .causal_transformer import CausalTransformer

def depths(decoder: CausalTransformer, types: Sequence[str], kinds: dict[str, ResolvedKind], mixer_spec,
           gated_mlp, routed, wiring: BlockWiring) -> list[MTPBlock]:
    """The multi-token prediction depths the decoder builds in its setup."""
    if not decoder.num_nextn_predict_layers:
        return []
    # Prediction depths mirror whole-sequence hidden states (`prediction_kind`); the feed-forward
    # routes like the last layer's (GLM 4.5 ships its depth with the trunk's experts), else dense.
    mtp_type = prediction_kind(decoder, types)
    prediction_mixer = kinds[mtp_type].mixer or mixer_spec
    if decoder.index_share_for_mtp_iteration and not prediction_mixer.shares_prediction_index:
        raise ValueError("index_share_for_mtp_iteration requires a prediction mixer that shares its "
                         "index (`MixerBase.shares_prediction_index`): a k-pool, or a kind composing one")
    mtp_mixer = prediction_mixer.build(decoder.mixer_context(
        kinds[mtp_type], mtp_type, kv_shared=False))
    widths = decoder.mlp_widths
    feeding = decoder.mtp_feedforward_type
    feeder = None if feeding is None else decoder.kind_of(feeding).mixer
    if feeding is not None and not isinstance(feeder, MLPMixer):
        raise ValueError(f"mtp_feedforward_type {feeding!r} builds its feed-forward from an MLPMixer, "
                         f"and the kind's mixer is {type(feeder).__name__}")
    mtp_feedforward = (
        feeder.feedforward(decoder.mixer_context(decoder.kind_of(feeding), feeding, kv_shared=False))
        if isinstance(feeder, MLPMixer) and feeding is not None else
        routed if routed is not None and decoder.num_layers - 1 in decoder.sparse_layers else
        None if widths[-1] == 0 else
        # The last layer's width: the one width of every model with
        # depths, since the widths that vary are Gemma 3n's alone.
        functools.partial(gated_mlp, hidden_features=widths[-1]))
    return [
        MTPBlock(
            mixer=mtp_mixer, feedforward=mtp_feedforward,
            emb_features=decoder.emb_features,
            hyper_connections=decoder.mtp_hyper_connections,
            norm_eps=decoder.norm_eps,
            scale_offset=decoder.scale_offset,
            scale_after_cast=decoder.scale_after_cast,
            wiring=wiring,
            dropout_rate=decoder.dropout_rate,
            remat=decoder.remat,
            dtype=decoder.dtype, precision=decoder.precision, name=f'mtp_{depth}')
        for depth in range(decoder.num_nextn_predict_layers)]


def depth_states(decoder: CausalTransformer, hidden, tokens, train: bool = False, input_embeddings=None,
                 image_groups=None, block_groups=None, rotary_positions=None,
                 coordinates: Coordinates | None = None, conditioning: Mapping[str, jax.Array] | None = None):
    """Return one final-normed state array per shifted prediction depth.

    Each depth combines the preceding hidden state with the next token's embedding, including any media
    replacement, and uses the next token's positions. A pair counts only when both of its positions are
    valid and belong to the same packed document, so padded positions in between never become keys; a
    watched depth's summary reads the positions its caller names (`watched_positions`) carried alike. A
    sequence no longer than the number of depths raises `ValueError`. The depths keep no
    cross-attention: `hidden`, the trunk's states, already read the call's `conditioning`, which they
    take as the trunk does and leave unread.
    """
    del conditioning
    if decoder.mtp and tokens.shape[1] <= len(decoder.mtp):
        raise ValueError("prediction depths need a sequence longer than their depth count")
    documents, given, given_valid = ((None, None, None) if coordinates is None else
                                     (coordinates.segments, coordinates.positions, coordinates.valid))
    embeds = prepared_embeddings(decoder.stream_embeddings(tokens), input_embeddings)
    # A depth restricts its keys only where the caller's validity or a document boundary does. With
    # neither, every shifted pair is real, and no validity says that: an all-true array would make the
    # depth build a mask and drop off the fused kernel.
    valid = None
    if documents is not None or given_valid is not None:
        valid = jnp.ones(tokens.shape, bool) if given_valid is None else given_valid
    states = []
    named = held if (held := current_positions()) is not None and held.shape == hidden.shape[:2] else None
    for depth, block in enumerate(decoder.mtp, start=1):
        within = None if documents is None else documents[:, depth - 1:]
        if valid is not None:
            valid = shifted_span(valid.shape, 1, valid=valid, segments=within)
        named = None if named is None else shifted_span(named.shape, 1, valid=named, segments=within)
        segments, places = (None if field is None else field[:, depth:] for field in (documents, given))
        metadata = AttentionMetadata(
            # Cross-document pairs leave holes even when the token rows were right-padded.
            coordinates=None if coordinates is None else build_coordinates(
                (hidden.shape[0], hidden.shape[1] - 1), segments=segments, valid=valid, positions=places,
                max_segments=coordinates.max_segments),
            image_groups=None if image_groups is None else image_groups[:, depth:],
            block_groups=None if block_groups is None else block_groups[:, depth:],
            rotary_positions=None if rotary_positions is None else rotary_positions[:, depth:])
        with watched_positions(named):
            hidden = block(hidden[:, :-1], embeds[:, depth:], train=train, attention_metadata=metadata)
        states.append(hidden)
    return states


def depth_logits(decoder: CausalTransformer, hidden, tokens, train: bool = False, input_embeddings=None,
                 image_groups=None, block_groups=None, rotary_positions=None,
                 coordinates: Coordinates | None = None, conditioning: Mapping[str, jax.Array] | None = None):
    """Return the shared language head's logits for each prediction depth's hidden states; a
    call's `conditioning` is the trunk's (`mtp_hidden_states`)."""
    return [decoder.logits_from_hidden(state) for state in depth_states(
        decoder, hidden, tokens, train=train, input_embeddings=input_embeddings, image_groups=image_groups,
        block_groups=block_groups, rotary_positions=rotary_positions, coordinates=coordinates,
        conditioning=conditioning)]


def depth_step(decoder: CausalTransformer, hidden, tokens, *, depth: int = 0, input_embeddings=None,
               rotary_positions=None, decode: bool = False, prediction_phase: PredictionPhase = "ordinary",
               coordinates: Coordinates | None = None):
    """Run one unshifted prediction step, optionally appending to the depth's own KV cache.

    Call `init_mtp_cache` before cached steps. `hidden` is the target
    model's preceding state, and `tokens` or `input_embeddings` give the
    candidate next token, as in vLLM's Qwen3_5MultiTokenPredictor. Returns
    the step's logits and hidden state; a chained draft's next step reads
    that state in place of the target's. With
    `index_share_for_mtp_iteration`, a cached "extend" step publishes its
    index selections and a "draft" step reuses them, while "ordinary"
    always recomputes them. Its tokens are laid out by their `coordinates`.
    """
    if prediction_phase not in ("ordinary", "extend", "draft"):
        raise ValueError("prediction_phase must be ordinary, extend or draft")
    if depth < 0 or depth >= len(decoder.mtp):
        raise ValueError("prediction depth is outside the model's configured depths")
    embeds = decoder.stream_embeddings(tokens) if input_embeddings is None else input_embeddings
    state, prediction = decoder.mtp[depth].states(
        hidden, embeds, decode=decode,
        prediction_phase=prediction_phase if decoder.index_share_for_mtp_iteration else "ordinary",
        attention_metadata=AttentionMetadata(coordinates=coordinates, rotary_positions=rotary_positions))
    return decoder.logits_from_hidden(state), prediction


def prediction_kind(decoder: CausalTransformer, types: Sequence[str]) -> str:
    """A prediction depth's mixer kind: `mtp_layer_type`, else full attention, else the first layer's."""
    return decoder.mtp_layer_type or ('full_attention' if 'full_attention' in types else types[0])
