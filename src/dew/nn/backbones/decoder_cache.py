"""A decoder's decode cache as its holders state it (`dew.nn.kv_cache.HoldsCache`), laid out as its stack
keeps it, with no forward traced (`CausalTransformer.init_cache`).

Each layer's cache sits under the layer's name; a store banked by run keeps each run's layers' stacked
under the run's name, as a decode over the bank reads them (`StackView`); a loop's passes keep theirs
under the names their views store them by (`looping.pass_entries`); and the decoder's own holders sit
beside them: its cursor of learned positions and its engram's history. The prediction depths' caches and
the block drafter's are apart (`named_caches`), each opened by its own method, as a decode opens them where
it runs them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
from flax import linen as nn

from ..kv_cache import CacheSpec, stacked_spec
from ..rope import LongRopeScaling
from ..sharding import pipeline_stages
from .decoder_stack import StackView, stored_by_stage
from .layer_names import stored_layers
from .looping import pass_entries

if TYPE_CHECKING:
    from .causal_transformer import CausalTransformer


def trunk_cache(model: CausalTransformer, rows: int) -> CacheSpec:
    """`model`'s trunk's decode cache for `rows` rows: its layers', its loop's passes' and its own
    holders' (the module doc)."""
    banked = _banked_runs(model)
    runs = banked or tuple((index, 1) for index in range(model.num_layers))
    if banked is not None:
        # A banked store's layers' specs read each run's first layer where its bank holds it, every layer
        # of a run alike.
        layers = [_banked_layer(model, run).cache_spec(rows)
                  for run, (_, count) in enumerate(banked) for _ in range(count)]
    elif stored_by_stage(model.variables):
        # A stage-local store's layers' specs read each layer where its stage's run holds it, a periodic
        # run's at its position.
        held = _layered(model)
        layers = [model.layers[index].clone(parent=None).bind(
            {collection: tree[f'layers_{index}'] for collection, tree in held.items()
             if f'layers_{index}' in tree}).cache_spec(rows) for index in range(model.num_layers)]
    else:
        layers = [layer.cache_spec(rows) for layer in model.layers]
    entries = [(first, first, count) for first, count in runs]
    entries += [] if model.loop is None else pass_entries(model.loop, model.num_layers, runs)
    names = StackView(tuple((stored, count) for stored, _, count in entries)).bank_names()
    spec: dict = {}
    for name, (_, first, count) in zip(names, entries, strict=True):
        run = layers[first:first + count]
        if run[0]:
            spec[name] = run[0] if count == 1 else stacked_spec(run)
    if model.position_embedding == 'learned':
        spec['decode_cursor'] = model.decode_cursor.cache_spec(rows)
    if model.engram is not None:
        spec['engram_hashes'] = model.engram_hashes.cache_spec(rows)
    return spec


def named_caches(holders: Sequence[nn.Module], rows: int) -> CacheSpec:
    """The decode caches of a decoder's `holders` kept apart from its trunk's, its prediction depths' or its
    block drafter's stages', for `rows` rows, each under its holder's name."""
    return {name: held for holder in holders if (name := holder.name) and (held := holder.cache_spec(rows))}


def _banked_layer(model: CausalTransformer, run: int):
    """The first layer of `model`'s run `run`, bound to its variables as the store's bank holds them, a row
    of it (`CausalTransformer.layer_shapes`): a layer whose own scope the bank leaves empty."""
    first = model.groups[run][0]
    return model.layers[first].clone(parent=None).bind(model.layer_shapes(run))


def _layered(model: CausalTransformer) -> dict:
    """`model`'s stage-local store a layer at a time, each run's rows cut back onto its layers as an export
    reads them (`stored_layers`), for their shapes alone."""
    free = model.clone(parent=None)
    variables = {collection: dict(tree) for collection, tree in model.variables.items()}
    return jax.eval_shape(lambda variables: stored_layers(free, variables), variables)


def _banked_runs(model: CausalTransformer) -> tuple[tuple[int, int], ...] | None:
    """The runs a decode over `model`'s store banked by run keeps its cache in, as it keeps the weights:
    each run's layers one array under the run's name, which the run's scan carries and reads in place,
    where a cache kept a layer at a time is stacked into the run's bank and cut apart again each call (at
    Qwen3-30B-A3B's layer shape, 8 layers of 64 rows of 512 slots on an H100, 371 and 317 us a step,
    OM-1342-cache-base-2474a6e4c); a cache handed in a layer at a time goes back so. None for a store kept
    a layer at a time, a stage-local one, or one a pipeline splits."""
    banked = 'params' in model.banked_collections()
    if not banked or stored_by_stage(model.variables) or pipeline_stages() > 1:
        return None
    return model.groups


def rebuild_position(decoder: CausalTransformer) -> int | None:
    """Where LongRoPE's factors switch from short to long, past which cached keys go stale: the
    original length of every trunk layer's LongRoPE, its kind's or the model's (`kind_of`), None
    where no layer takes LongRoPE. Kinds that switch at different lengths are refused, since a cache
    rebuilds at one."""
    originals = {kind.rope_scaling.original_max_position_embeddings
                 for kind in map(decoder.kind_of, set(decoder.per_layer_types))
                 if isinstance(kind.rope_scaling, LongRopeScaling)}
    if len(originals) > 1:
        raise ValueError(f"layer kinds switch LongRoPE's factors at {sorted(originals)} tokens; a cache "
                         "rebuilds its keys at one")
    return next(iter(originals), None)
