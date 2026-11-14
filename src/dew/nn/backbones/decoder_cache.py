"""A decoder's decode cache as its holders state it (`dew.nn.kv_cache.HoldsCache`), laid out as its stack
keeps it, with no forward traced (`CausalTransformer.init_cache`).

Each layer's cache sits under the layer's name; a store banked by run keeps each run's layers' stacked
under the run's name, as a decode over the bank reads them (`StackView`); a loop's passes keep theirs
under the names their views store them by (`looping.pass_entries`); and the decoder's own holders sit
beside them: its cursor of learned positions and its engram's history. The prediction depths' caches and
the block drafter's are apart, each opened by its own method, as a decode opens them where it runs them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..kv_cache import CacheSpec, stacked_spec
from ..sharding import pipeline_stages
from .decoder_stack import StackView, stored_by_stage
from .looping import pass_entries

if TYPE_CHECKING:
    from .causal_transformer import CausalTransformer


def trunk_cache(model: CausalTransformer, rows: int) -> CacheSpec:
    """`model`'s trunk's decode cache for `rows` rows: its layers', its loop's passes' and its own
    holders' (the module doc)."""
    layers = [layer.cache_spec(rows) for layer in model.layers]
    spec: dict = {}
    runs = _banked_runs(model)
    if runs is None:
        entries = [(index, index) for index in range(model.num_layers)]
        entries += [] if model.loop is None else pass_entries(model.loop, model.num_layers)
        spec.update({f'layers_{stored}': layers[layer] for stored, layer in entries if layers[layer]})
    else:
        for name, (first, count) in zip(StackView(runs).bank_names(), runs, strict=True):
            run = layers[first:first + count]
            if run[0]:
                spec[name] = run[0] if count == 1 else stacked_spec(run)
    if model.position_embedding == 'learned':
        spec['decode_cursor'] = model.decode_cursor.cache_spec(rows)
    if model.engram is not None:
        spec['engram_hashes'] = model.engram_hashes.cache_spec(rows)
    return spec


def depths_cache(model: CausalTransformer, rows: int) -> CacheSpec:
    """`model`'s prediction depths' decode caches for `rows` rows, each under its depth's name."""
    return {name: held for depth in model.mtp if (name := depth.name) and (held := depth.cache_spec(rows))}


def drafter_cache(model: CausalTransformer, rows: int) -> CacheSpec:
    """`model`'s block drafter's window caches for `rows` rows, each under its stage's name."""
    return {name: held for stage in model.dspark_stages
            if (name := stage.name) and (held := stage.cache_spec(rows))}


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
