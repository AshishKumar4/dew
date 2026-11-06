"""The per-call data a `CausalTransformer`'s layers read beside the residual (`call_metadata`): its tokens'
layout, its own visibility, its media, routing blocks and cross-attention context."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from dew.batch import Coordinates

from ..inputs import AttentionMetadata, CrossContext, coordinates_of, numbered_blocks
from ..progress import Progress

if TYPE_CHECKING:
    from .causal_transformer import CausalTransformer


def call_metadata(model: CausalTransformer, tokens, decode: bool, image_groups, block_groups,
                  rotary_positions, pairwise_mask, key_positions, key_runs, media_mask, admitted=None,
                  coordinates: Coordinates | None = None,
                  conditioning: Mapping[str, jax.Array] | None = None, *,
                  whole_blocks: bool = False, progress: Progress | None = None,
                  key_valid=None) -> AttentionMetadata | None:
    """What `model`'s layers read beside the residual, None for a call that
    carries none of it: the tokens' layout `hidden_and_mtp_inputs`
    converted (`coordinates`), the token ids a hash router selects by,
    the engram layers' bucket ids, the media mask, which only a model
    with engram or a router with a media bias reads, and the
    cross-attention's context."""
    if conditioning and model.cross_attention is None:
        raise TypeError(f"this decoder has no cross-attention to read the conditioning "
                        f"{sorted(conditioning)}")
    context = CrossContext.from_conditioning(conditioning) if conditioning else None
    if context is not None and context.embeddings.shape[0] != tokens.shape[0]:
        raise ValueError(f"a context row per token row, got {context.embeddings.shape[0]} for "
                         f"{tokens.shape[0]}")
    if context is not None:
        # The context enters in the dtype a server holds it in (`context_dtype`), so a served and
        # a standalone call, and a cached and a full one, project the same values.
        context = dataclasses.replace(context, embeddings=context.embeddings.astype(model.context_dtype))
    # A layer whose mixer reads no pairwise mask or key runs refuses one (`MixerCall`).
    if key_positions is not None and pairwise_mask is None and key_runs is None:
        raise ValueError("attention_key_positions requires attention_pairwise_mask or "
                         "attention_key_runs")
    if pairwise_mask is not None and key_runs is not None:
        raise ValueError("attention_key_runs and attention_pairwise_mask each name a call's whole "
                         "visibility; pass one")
    if key_valid is not None and pairwise_mask is None and key_runs is None:
        raise ValueError("attention_key_valid hides keys from a call's own visibility; it requires "
                         "attention_pairwise_mask or attention_key_runs")
    if model.engram is None and (model.mixture is None or not model.mixture.media_bias):
        media_mask = None
    if decode and coordinates is not None and coordinates.max_segments > 1:
        raise ValueError(
            "packed rows cannot append to a decode cache: it stores no document ids, so a cached "
            "call would attend across documents; prefill each document into a row of its own")
    engram_ids = (None if model.engram is None else model.engram_hashes(tokens, coordinates, decode,
                                                                      media_mask))
    if context is not None and context.segments is not None:
        # A context of several documents is read by the call's tokens of each document alone.
        if coordinates is None or coordinates.segments is None:
            raise ValueError("a context of several documents is read by packed rows, which name "
                             "their tokens' documents (segment_ids or coordinates)")
        context = dataclasses.replace(context, queries=coordinates.segments)
    # A block router's blocks are numbered once for every routed layer.
    block = None if model.mixture is None else model.mixture.routing_block
    blocks = None if block is None else numbered_blocks(
        coordinates_of(None, *tokens.shape[:2]) if coordinates is None else coordinates, tokens.shape[:2],
        block)
    if (coordinates is None and image_groups is None and block_groups is None and rotary_positions is None
            and pairwise_mask is None and key_positions is None and key_runs is None and key_valid is None
            and not model.hash_layers
            and engram_ids is None and media_mask is None and admitted is None and blocks is None
            and context is None and (progress is None or progress.updates is None)):
        return None
    return AttentionMetadata(
        coordinates=coordinates, image_groups=image_groups, block_groups=block_groups,
        rotary_positions=rotary_positions, pairwise_mask=pairwise_mask, key_positions=key_positions,
        key_runs=key_runs, key_valid=key_valid, token_ids=tokens if model.hash_layers else None,
        engram_ids=engram_ids,
        media=media_mask,
        admitted=admitted, blocks=blocks, context=context,
        # One row's progress apiece, as the metadata's other inputs are held by row, which a
        # pipeline's microbatches slice (`run_pipeline`).
        progress=None if progress is None or progress.updates is None else jnp.broadcast_to(
            jnp.asarray(progress.updates, jnp.int32), tokens.shape[:1]),
        routing_block=block,
        whole_blocks=whole_blocks and block_groups is not None)
