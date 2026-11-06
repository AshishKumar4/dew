"""The attention template's Mosaic GPU backward (`attention_mosaic.MosaicBackward`'s launches): D about each
query's anchor from a training forward's split sums (`gradient.split_delta`), then a dQ launch and a dK/dV
launch, `dq_finish`'s refinement between them, in either form, or the fast form's one launch, the dK/dV
program adding each tile's dQ (`_launch_backward`). Each launch's programs walk the bundle's streams as the
forward's do (`attention_mosaic_program`), the algebra's steps between its products."""

import dataclasses

import jax.numpy as jnp

from dew.nn.kernels.templates import (
    attention_algebra as algebra,
    attention_gradient as gradient,
    bundles,
    mosaic,
)
from dew.nn.kernels.templates.attention import DQ_GRADS, Dropout, Hooks, Kind, PaddedCall, Param, by_power
from dew.nn.kernels.templates.attention_mosaic_coop import _coop_launch
from dew.nn.kernels.templates.attention_mosaic_dkv import _dkv_launch
from dew.nn.kernels.templates.attention_mosaic_dq import _dq_launch
from dew.nn.kernels.templates.attention_mosaic_fused import _fused_launch
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    _columns,
    _Program,
    _spans_of,
    approximate,
    backward_family,
    score_places,
)
from dew.nn.kernels.templates.attention_mosaic_scores import _Backward


def _launch_backward(query, key, value, dout, peak0, total0, initial_term, maximum, key_index, anchor_value,
                     shift, others, own, others_total, forward_total, start, end, query_ids, key_ids,
                     score_params, *, score_mod, score_kinds, score_dtypes, scale: float, prescale: float,
                     spec: algebra.Forward, accurate: bool, tiles: BackwardTiles, family: mosaic.Family,
                     interpret: bool, drop: Dropout | None = None):
    """`gradients`' launch on this mainloop: D from the forward's split sums, each query's anchor key as the
    first pass chose it. Two launches: the dQ program's sums, their finish (`gradient.dq_finish`) and the
    dK/dV program under its refined D, whose cotangents sum to 0 where the first pass's do not (a small weight
    on a large value, bf16: dK's worst row 2.5e1 at the first pass's D, where check 1 allows 1.55e-2), the
    accurate form's always, and a score hook's parameters', whose gradients need the refined D
    (`gradient.dq_step`). The fast form's one launch (`tiles.fused`, a call whose score hook, if any, takes
    no parameters, an initial state's gradient from the first pass's D): the dK/dV program adding each tile's
    dQ atomically, under the first pass's D and the forward's denominator, the derived default of such a call
    (`MosaicBackward.derived`; in deterministic mode a full or causal one's, its adds in turn: 1.05 to 1.4
    times as fast as two launches, FUSEDFORM-h100; 384 MB of temporaries against 258 at llama-d128 2 x 4096,
    its dQ fp32); two launches, in a fixed order, otherwise. Returns what `gradients` takes: dQ, the softmax's
    rows, each query's anchor key, the dQ parameters' gradients (`DQ_GRADS`; a head's as per-query sums `[1,
    B, H, 1]`, which `gradients` sums over the batch), then dK, dV and the dK/dV parameters' (`DKV_GRADS`).
    The query arrives the call's own: its logits are those of the query taken by `prescale` (`by_power`) at
    `scale`, the scale left (`gradients`), which `spec` takes."""
    spec = dataclasses.replace(spec, scale=scale, prescale=prescale)
    width, value_width = query.shape[-1], value.shape[-1]
    _, state_total = gradient.off_anchor_start(peak0, total0, maximum, value_width, shift)
    cotangent = jnp.swapaxes(dout, 1, 2).astype(jnp.float32)
    values = jnp.swapaxes(anchor_value, 1, 2)
    if drop is not None:
        # A dropping call's D from the forward's dropped split sums too, about its anchor's value as its
        # draw keeps it (`algebra.anchor_multipliers`, the factor the forward's output took it by): no first
        # pass of its own (row 2252, the centre launch's dropped products recomputed every key tile).
        values = values * algebra.anchor_multipliers(drop, key_index)[..., None]
    anchor, delta, chosen = gradient.split_delta(others, own, others_total, forward_total, state_total,
                                                values, cotangent, key_index, shift)
    # The family's swizzled tiles hold whole rows (`Family.column_unit`).
    query, key, value, dout = (_columns(array, family.column_unit(array.dtype))
                               for array in (query, key, value, dout))
    steps = key.shape[1] // tiles.block_k
    program = _Program(family, tiles, steps, min(tiles.stages, steps), spec.keys, spec.form == "causal",
                       spec.queries)
    hooks = Hooks(score_mod, tuple(map(Param, score_params, score_kinds)))
    # The accurate form's exponentials exact: the hardware's ex2.approx flushes a subnormal weight to zero,
    # whose product with a large score cotangent the reference keeps (row 2022, ET-SUBFIX2-h100).
    approx_math = not accurate and approximate(hooks, (tiles.block_q, tiles.block_k), value_width,
                                               differentiated=True)
    walk = _Backward(program, spec, accurate, tiles.consumers, interpret, tiles.fused, score_kinds,
                     approx_math, score_places(hooks), drop is not None)
    padded = PaddedCall(query, key, value, start, end, query_ids, key_ids, (), (), query.shape[1], spec.keys)
    spans = _spans_of(spec, padded, tiles.block_q, tiles.block_k, readers=True)
    # The queries taken by the call's power: the logits' operand of the two launches (`_dkv_step`).
    scaled = by_power(query, prescale)
    # The dK/dV launch on its own layout (`BackwardTiles.dkv`).
    dkv_tiles = tiles.dkv
    dkv_walk = dataclasses.replace(walk, program=dataclasses.replace(
        program, family=backward_family(bundles.named(dkv_tiles.family, query.dtype), dkv_tiles),
        tiles=dkv_tiles),
        consumers=dkv_tiles.consumers)
    if walk.fused:
        if tiles.shared_keys:
            dq, dk, dv = _coop_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                      spans, walk=walk)
        elif tiles.dkv_apart:
            # The fused launch keys first on its own layout (`BackwardTiles.dkv`): its consumers' shares of a
            # slot's queries each a warpgroup MMA's rows, dQ added a tile at a time.
            dq, dk, dv = _dkv_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                     spans, score_params, drop, scaled, walk=dkv_walk)
        else:
            dq, dk, dv = _fused_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                       spans, drop, walk=walk)
        mass = gradient.split_mass(own, others_total, forward_total, shift)
        return (dq[..., :width], delta, anchor, maximum, forward_total, mass, chosen, dk[..., :width],
                dv[..., :value_width])
    # The dQ program reads its queries for their logits alone.
    dq_sum, keyed, rows = _dq_launch(scaled, key, value, dout, maximum, forward_total,
                                     anchor, delta, chosen, spans, score_params, drop, walk=walk)
    count = sum(walk.summed)
    dq_swapped = jnp.swapaxes(dq_sum, 1, 2)
    # The fast form's keyed sums are never read (`dq_finish`): a placeholder there.
    keyed = jnp.zeros_like(dq_swapped) if keyed is None else jnp.swapaxes(keyed, 1, 2)
    sums = gradient.DqSums(dq_swapped, rows[0], rows[1], tuple(rows[4:4 + count]), tuple(rows[4 + count:]),
                          keyed, rows[2], rows[3])
    dq, refined, denominator, mass, dq_grads = gradient.dq_finish(sums, forward_total, state_total, peak0,
                                                                  (total0, initial_term), maximum, anchor,
                                                                  delta, accurate=accurate)
    dk, dv, *key_tile = _dkv_launch(query, key, value, dout, maximum, denominator, anchor, refined, chosen,
                                    spans, score_params, drop, scaled, walk=dkv_walk)
    summed_kinds = [kind for kind in score_kinds if kind in DQ_GRADS]
    head_query = tuple(grad.sum(axis=-1)[None, :, :, None] if kind == Kind.HEAD else grad
                       for grad, kind in zip(dq_grads, summed_kinds, strict=True))
    return (jnp.swapaxes(dq, 1, 2)[..., :width], refined, anchor, maximum, denominator, mass, chosen,
            *head_query, dk[..., :width], dv[..., :value_width], *key_tile)
