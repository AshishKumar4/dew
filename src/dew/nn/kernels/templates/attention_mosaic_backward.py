"""The attention template's Mosaic GPU backward (`attention_mosaic.MosaicBackward`'s launches): D about each
query's anchor from a training forward's split sums (`algebra.split_delta`), then a dQ launch and a dK/dV
launch, `dq_finish`'s refinement between them, in either form, or the fast form's one launch, the dK/dV
program adding each tile's dQ (`_launch_backward`). Each launch's programs walk the bundle's streams as the
forward's do (`attention_mosaic_program`), the algebra's steps between its products."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import attention_algebra as algebra, bundles, mosaic
from dew.nn.kernels.templates.attention import DQ_GRADS, Dropout, Hooks, Kind, PaddedCall, Param, by_power
from dew.nn.kernels.templates.attention_mosaic_coop import _coop_launch
from dew.nn.kernels.templates.attention_mosaic_dkv import _dkv_launch
from dew.nn.kernels.templates.attention_mosaic_dq import _dq_launch
from dew.nn.kernels.templates.attention_mosaic_fused import _fused_launch
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    _columns,
    _coords,
    _Program,
    _rows,
    _Spans,
    _spans_of,
    approximate,
    stated,
    traced_name,
)
from dew.nn.kernels.templates.attention_mosaic_scores import (
    _Backward,
    _query_program,
    _query_tile_loaded,
    _score_params,
)
from dew.nn.precision import running_sum


def _centre_launch(query, key, value, dout, peak, spans: _Spans | None, score_params: tuple, drop: Dropout, *,
                   walk: _Backward) -> algebra.Centre:
    """Dropout's first pass on this mainloop (`algebra.dropout_centre`): each query tile's walk over its keys
    (a consumer's, as the dQ program's), each tile's logits `exp(s - m)` and dropped products
    (`algebra.dropped_products` of `dO Vᵀ` and the call's draw) taken into the query's `Centre`
    (`algebra.centre_step`). Returns the centre and its seen keys `[B, H, Qp]` and each running sum
    `[k, B, H, Qp]` (`precision.running_sum`), which `algebra.centre_finish` takes past the launch."""
    batch, queries, heads, width = query.shape
    kv_heads = key.shape[2]
    program, consumers, kinds = walk.program, walk.consumers, walk.kinds
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_q
    exponent = algebra.centre_scale(walk.spec.keys)
    leaves, treedef = jax.tree.flatten((spans, score_params, drop))
    # The centre, whether a key is seen, and each running sum's parts (`precision.running_sum`).
    parts = running_sum((1,), jnp.float32).shape[0]
    fields = 2 + 2 * parts

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, peak_ref), refs = refs[:5], refs[5:]
        (spans_of, score_refs, drop_of), refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        centre_ref, (q_smem, do_smem, row_smem, q_loaded, do_loaded) = refs[0], refs[1:]
        row, head, views, streams, pair_first, streamed = _query_program(
            program, streams, k_ref, v_ref, spans_of, heads=heads, kv_heads=kv_heads, queries=queries,
            consumers=consumers)

        def consumer(index):
            first = pair_first + index * block_q
            _query_tile_loaded(family, (q_ref, do_ref), (q_smem, do_smem), (q_loaded, do_loaded), (row, head),
                               first, index)
            peak = _rows(family, peak_ref, row, head, first, block_q)
            lo, hi, _ = program.reach(first, spans_of, row)
            shape = (block_q, block_k)
            draw = drop_of.draw(row, head)
            q, do = family.held(q_smem.at[index]), family.held(do_smem.at[index])

            def step(index, slots, state, accumulators):
                k_first = index * block_k
                s = family.product(q, slots["keys"].transpose((1, 0)), shape)
                dp = family.product(do, slots["values"].transpose((1, 0)), shape)
                slots.release("values")
                coords = _coords(family, spans_of, row, first, k_first, shape, draw=draw)
                params = _score_params(family, score_refs, kinds, row, head, first, s, k_first)
                # Every entry's visibility, a tile whole too (`algebra.seen` gives None for one no mask cuts).
                seen = algebra.seen(walk.spec, coords, s)
                seen = jnp.ones(shape, bool) if seen is None else seen
                logits, _ = algebra.tile_logits(walk.spec.score_mod, s, seen, coords.q_pos, coords.k_pos,
                                                params, scale=walk.spec.scale if walk.how.factor is None
                                                else 1.0, slopes=False)
                weights = algebra.below(logits, algebra.expand(peak, logits), walk.how)
                products = algebra.dropped_products(dp, algebra.drawn(coords, coords.q_pos, coords.k_pos))
                centred = algebra.centre_step(state, weights, products, seen, coords.k_pos, exponent)
                return centred, accumulators

            # Every field a row vector of zeros (`algebra.centre_start`), `seen` false, each running sum
            # its leading axis's parts (`precision.added_to`): the layouts hold no unit axis.
            zero = family.zeros((block_q,))
            parts = running_sum((1,), jnp.float32).shape[0]
            start = algebra.Centre(zero, zero > 0, (zero,) * parts, (zero,) * parts)
            centred, _ = family.walk_own(
                streams, views, (lo, hi), streamed, index,
                lambda streams, lo, hi, counted: family.walk(streams, ((step, lo, hi),), start, (),
                                                             walk.interpret, first=counted))
            rows_of = [part.astype(jnp.float32) for part in jax.tree.leaves(centred)]
            family.stored(tuple(family.as_row(part) for part in rows_of),
                          tuple(centre_ref.at[place, row, head, pl.ds(first, block_q)]
                                for place in range(fields)), smem=row_smem.at[index])

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    scoped = [plgpu.SMEM((consumers, block_q, width), query.dtype),
              plgpu.SMEM((consumers, block_q, dout.shape[-1]), dout.dtype),
              plgpu.SMEM((consumers, block_q), jnp.float32),
              family.load_barriers(consumers), family.load_barriers(consumers)]
    rings = family.streams({"keys": ((block_k, width), key.dtype), "values": ((block_k, value.shape[-1]),
                                                                              value.dtype)},
                           program.stages, consumers)

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    out = family.kernel(
        entry, out_type=jax.ShapeDtypeStruct((fields, batch, heads, queries), jnp.float32),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
        interpret=walk.interpret, approx_math=False, name="attention_centre",
        cost=stated(walk.spec, batch, heads, walk.spec.queries or queries, 0, query, key, value, dout)
    )(query, key, value, dout, peak, *leaves)
    return algebra.Centre(out[0], out[1] > 0, out[2:2 + parts], out[2 + parts:])

def _launch_backward(query, key, value, dout, peak0, total0, initial_term, maximum, key_index, anchor_value,
                     shift, others, own, others_total, forward_total, start, end, query_ids, key_ids,
                     score_params, *, score_mod, score_kinds, score_dtypes, scale: float, prescale: float,
                     spec: algebra.Forward, accurate: bool, tiles: BackwardTiles, family: mosaic.Family,
                     interpret: bool, drop: Dropout | None = None):
    """`gradients`' launch on this mainloop: D from the forward's split sums, each query's anchor key as the
    first pass chose it. Two launches: the dQ program's sums, their finish (`algebra.dq_finish`) and the dK/dV
    program under its refined D, whose cotangents sum to 0 where the first pass's do not (a small weight on a
    large value, bf16: dK's worst row 2.5e1 at the first pass's D, where check 1 allows 1.55e-2), the accurate
    form's always, and a score hook's and an initial state's, whose gradients need the refined D
    (`algebra.dq_step`). The fast form's one launch (`tiles.fused`, a hookless call's): the dK/dV program
    adding each tile's dQ atomically, under the first pass's D and the forward's denominator, the derived
    default of a hookless fast-form call outside deterministic mode (`MosaicBackward.derived`: 1.05 to 1.4
    times as fast as two launches, FUSEDFORM-h100; 384 MB of temporaries against 258 at llama-d128 2 x 4096,
    its dQ fp32); two launches, in a fixed order, otherwise. Returns what `gradients` takes: dQ, the softmax's
    rows, each query's anchor key, the dQ parameters' gradients (`DQ_GRADS`; a head's as per-query sums
    `[1, B, H, 1]`, which `gradients` sums over the batch), then dK, dV and the dK/dV parameters'
    (`DKV_GRADS`). The query arrives the call's own: its logits are those of the query taken by `prescale`
    (`by_power`) at `scale`, the scale left (`gradients`), which `spec` takes."""
    spec = dataclasses.replace(spec, scale=scale, prescale=prescale)
    width, value_width = query.shape[-1], value.shape[-1]
    _, state_total = algebra.off_anchor_start(peak0, total0, maximum, value_width, shift)
    cotangent = jnp.swapaxes(dout, 1, 2).astype(jnp.float32)
    anchor, delta, chosen = algebra.split_delta(others, own, others_total, forward_total, state_total,
                                                jnp.swapaxes(anchor_value, 1, 2), cotangent, key_index, shift)
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
                     approx_math)
    padded = PaddedCall(query, key, value, start, end, query_ids, key_ids, (), (), query.shape[1], spec.keys)
    spans = _spans_of(spec, padded, tiles.block_q, tiles.block_k)
    # The queries taken by the call's power: the logits' operand of the two launches (`_dkv_step`).
    scaled = by_power(query, prescale)
    if drop is not None:
        # A dropping call's D its dropped products' own, centred about each query's least one
        # (`algebra.dropout_centre`): no anchor, and no key chosen.
        centre = _centre_launch(scaled, key, value, dout, maximum, spans, score_params,
                                drop, walk=walk)
        anchor = jnp.zeros_like(anchor)
        delta = algebra.centre_finish(centre, state_total, initial_term, algebra.centre_scale(spec.keys))
        chosen = jnp.full_like(chosen, -1)
    # The dK/dV launch on its own layout (`BackwardTiles.dkv`).
    dkv_tiles = tiles.dkv
    dkv_walk = dataclasses.replace(walk, program=dataclasses.replace(
        program, family=bundles.named(dkv_tiles.family, query.dtype), tiles=dkv_tiles),
        consumers=dkv_tiles.consumers)
    if walk.fused:
        if tiles.shared_keys:
            dq, dk, dv = _coop_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                      spans, walk=walk, names=traced_name("attention_dkv_shared"))
        elif tiles.dkv_apart:
            # The fused launch keys first on its own layout (`BackwardTiles.dkv`): its consumers' shares of a
            # slot's queries each a warpgroup MMA's rows, dQ added a tile at a time.
            dq, dk, dv = _dkv_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                     spans, score_params, drop, scaled, walk=dkv_walk)
        else:
            dq, dk, dv = _fused_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                       spans, walk=walk, names=traced_name("attention_dkv_fused"))
        mass = algebra.split_mass(own, others_total, forward_total, shift)
        return (dq[..., :width], delta, anchor, maximum, forward_total, mass, chosen, dk[..., :width],
                dv[..., :value_width])
    # The dQ program reads its queries for their logits alone.
    dq_sum, keyed, rows = _dq_launch(scaled, key, value, dout, maximum, forward_total,
                                     anchor, delta, chosen, spans, score_params, drop, walk=walk)
    count = sum(walk.summed)
    dq_swapped = jnp.swapaxes(dq_sum, 1, 2)
    # The fast form's keyed sums are never read (`dq_finish`): a placeholder there.
    keyed = jnp.zeros_like(dq_swapped) if keyed is None else jnp.swapaxes(keyed, 1, 2)
    sums = algebra.DqSums(dq_swapped, rows[0], rows[1], tuple(rows[4:4 + count]), tuple(rows[4 + count:]),
                          keyed, rows[2], rows[3])
    dq, refined, denominator, mass, dq_grads = algebra.dq_finish(sums, forward_total, state_total, peak0,
                                                                  (total0, initial_term), maximum, anchor,
                                                                  delta, accurate=accurate)
    dk, dv, *key_tile = _dkv_launch(query, key, value, dout, maximum, denominator, anchor, refined, chosen,
                                    spans, score_params, drop, scaled, walk=dkv_walk)
    summed_kinds = [kind for kind in score_kinds if kind in DQ_GRADS]
    head_query = tuple(grad.sum(axis=-1)[None, :, :, None] if kind == Kind.HEAD else grad
                       for grad, kind in zip(dq_grads, summed_kinds, strict=True))
    return (jnp.swapaxes(dq, 1, 2)[..., :width], refined, anchor, maximum, denominator, mass, chosen,
            *head_query, dk[..., :width], dv[..., :value_width], *key_tile)
