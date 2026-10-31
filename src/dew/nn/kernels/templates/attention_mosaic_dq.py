"""The attention template's Mosaic GPU dQ launch (`attention_mosaic_backward`): each consumer's query tile
walked over the key tiles it reads, its dQ, and the row sums `dq_finish` refines between the two launches."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import tuning
from dew.nn.kernels.templates import attention_algebra as algebra, mosaic
from dew.nn.kernels.templates.attention import Dropout
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    _coords,
    _rows,
    _Spans,
    reducing,
    stated,
)
from dew.nn.kernels.templates.attention_mosaic_scores import (
    _Backward,
    _query_program,
    _query_tile_loaded,
    _score_params,
    _scores,
)


def _dq_allocations(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype,
                    stages: int) -> tuple[list, dict]:
    """The dQ program's allocations as `Family.scoped` takes them, in `_dq_launch`'s order: its consumers'
    query and output-cotangent stages, their dQ output stage (fp32, its share of the config's `columns`) and
    their rows' stage, the two loads' barriers, and its key and value streams in `stages` slots
    (`Family.streams`)."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    scoped = [plgpu.SMEM((consumers, block_q, width), dtype),
              plgpu.SMEM((consumers, block_q, value_width), dtype),
              plgpu.SMEM((consumers, block_q, width // config.columns), jnp.float32),
              plgpu.SMEM((consumers, block_q), jnp.float32),
              family.load_barriers(consumers), family.load_barriers(consumers)]
    blocks = {"keys": ((block_k, width), dtype), "values": ((block_k, value_width), dtype)}
    return scoped, family.streams(blocks, stages, consumers)

def _dq_step(walk: _Backward, operands: tuple, stats: tuple, coords_of, params_of, whole, share, index, slots,
             rows, accumulators):
    """A dQ consumer's key tile `index`: its scores and score cotangents from the key and value slots
    (`slots`) and its queries and output cotangents (`operands`) under its rows' statistics (`stats`), the
    tile's coordinates (`coords_of(k_first)`), its hook parameters (`params_of(like, k_first=k_first)`) and
    whether its block sees it whole (`whole(index)`). The row sums are carried (`rows`); dQ's and the keyed
    sums' products are issued into `accumulators` in place (`algebra.dq_issued`)."""
    q, do = operands
    peak, prior, anchor, delta, chosen = stats
    family, tiles, empty = walk.program.family, walk.program.tiles, jnp.zeros((), jnp.float32)
    shape = (tiles.block_q, tiles.block_k)
    k_slot = slots["keys"]
    k_first = index * tiles.block_k
    s = family.product(q, k_slot.transpose((1, 0)), shape)
    dp = family.product(do, slots["values"].transpose((1, 0)), shape)
    slots.release("values")
    params = params_of(s, k_first=k_first)
    residual, mass, grads, weighed, keyed_mass, keyed_shift = rows
    # dQ's and the keyed sums' places stay empty: their products go to the accumulators.
    sums = algebra.DqSums(empty, residual, mass, grads, weighed, empty, keyed_mass, keyed_shift)

    def issued(*, masked: bool):
        # A tile every query sees whole (`masked` False) takes no mask; the row sums and the parts are taken
        # where the mask is read (`algebra.pulled`).
        scored = _scores(walk, coords_of(k_first), params, s, dp, peak, prior, anchor, delta, chosen,
                         masked=masked)
        exponentials, _, cotangents, slopes, seen = scored
        terms, parts, keyed = algebra.dq_issued(sums, exponentials, cotangents, slopes, seen, k_slot.dtype,
                                                prior, scale=walk.grad_scale, kinds=walk.kinds,
                                                summed=walk.summed, accurate=walk.accurate)
        row_sums = (terms.residual, terms.mass, terms.grads, terms.weighed, terms.keyed_mass,
                    terms.keyed_shift)
        return row_sums, parts, keyed

    # One walk, its tiles' masks chosen a tile at a time: a walk of two loops into one warpgroup accumulator,
    # the first empty, faults on an H100 under jax 0.11.2's Mosaic GPU lowering.
    rows, scaled, keyed = lax.cond(whole(index), functools.partial(issued, masked=False),
                                   functools.partial(issued, masked=True))
    dq_acc, *keyed_accs = accumulators
    # dQ's parts formed past the choice, as each product is issued (`algebra.formed_parts`): the choice's
    # branches yield the scaled cotangents alone, not every part.
    parts = algebra.formed_parts(scaled, k_slot.dtype, accurate=walk.accurate)
    # dQ's (and the keyed sums') products against the program's share of the keys' columns (`share`).
    k_columns = k_slot.at[:, pl.ds(*share)]
    dq_acc = family.register_products(dq_acc, parts, k_columns, walk.interpret)
    if keyed is None:
        return rows, (dq_acc,)
    factor, keyed_parts = keyed
    if factor is None:
        keyed_acc = family.register_products(keyed_accs[0], keyed_parts, k_columns, walk.interpret)
    else:
        # The keyed sums rescaled by their row factor, then the tile's products added in place.
        keyed_acc = family.rescaled_products(keyed_accs[0], functools.partial(algebra.rescale, alpha=factor),
                                             keyed_parts, k_columns, walk.interpret)
    return rows, (dq_acc, keyed_acc)

def _dq_launch(query, key, value, dout, peak, prior, anchor, delta, key_index, spans: _Spans | None,
               score_params: tuple, drop: Dropout | None, *, walk: _Backward):
    """`_dq_columns`' launch of each of the config's `columns`, its share of dQ's (and the keyed sums')
    columns, the rows the first's, the sums joined along their widths."""
    columns = walk.program.tiles.columns
    width = query.shape[-1] // columns
    launched = [_dq_columns(query, key, value, dout, peak, prior, anchor, delta, key_index, spans,
                            score_params, drop, walk=walk, share=(column * width, width),
                            name="attention_dq" + ("" if columns == 1 else f"_columns{column}"))
                for column in range(columns)]
    if columns == 1:
        return launched[0]
    dq = jnp.concatenate([out[0] for out in launched], axis=-1)
    keyed_parts = [out[1] for out in launched if out[1] is not None]
    keyed = jnp.concatenate(keyed_parts, axis=-1) if keyed_parts else None
    return dq, keyed, launched[0][2]

@tuning.lowered_once(("walk", "share", "name"))
def _dq_columns(query, key, value, dout, peak, prior, anchor, delta, key_index, spans: _Spans | None,
                score_params: tuple, drop: Dropout | None, *, walk: _Backward, share: tuple[int, int],
                name: str):
    """The dQ program's sums over its key tiles (`algebra.dq_issued`), unfinished: dQ's and, in the accurate
    form, the keys' product with the weights (`DqSums.keyed`), `[B, Qp, H, D]` fp32 each; and the rows
    `[4 + 2n, B, H, Qp]` fp32 (n the dQ parameters, `walk.summed`): the cotangents' and weights' row sums, the
    keyed sums' mass and exponent, then each dQ parameter's gradient and its weights' pullback, which
    `dq_finish` takes past the launch. Each query block of a pair is a consumer's, the key and value tiles
    streamed through a ring each. Its sums are its `share` of their columns (`(first, count)`). Returns dQ,
    the keyed sums (None in the fast form) and the rows."""
    batch, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    program, consumers, kinds = walk.program, walk.consumers, walk.kinds
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_q
    summed = walk.summed
    rows_count = 4 + 2 * sum(summed)
    leaves, treedef = jax.tree.flatten((spans, score_params, drop))

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, peak_ref, prior_ref, anchor_ref, delta_ref, index_ref), refs = (
            refs[:9], refs[9:])
        (spans_of, score_refs, drop_of), refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        dq_ref, refs = refs[0], refs[1:]
        keyed_ref, refs = (refs[0], refs[1:]) if walk.accurate else (None, refs)
        rows_ref, (q_smem, do_smem, out_smem, row_smem, q_loaded, do_loaded) = refs[0], refs[1:]
        row, head, views, streams, pair_first, streamed = _query_program(
            program, streams, k_ref, v_ref, spans_of, heads=heads, kv_heads=kv_heads, queries=queries,
            consumers=consumers)

        def consumer(index):
            first = pair_first + index * block_q
            _query_tile_loaded(family, (q_ref, do_ref), (q_smem, do_smem), (q_loaded, do_loaded), (row, head),
                               first, index)
            stats = tuple(_rows(family, ref, row, head, first, block_q)
                          for ref in (peak_ref, prior_ref, anchor_ref, delta_ref, index_ref))
            lo, hi, whole = program.reach(first, spans_of, row)
            nothing = family.zeros((block_q,))
            operands = family.held(q_smem.at[index]), family.held(do_smem.at[index])
            shape = (block_q, block_k)
            # The tile's places formed a step, not held across the walk beside its sums (`_coords`).
            coords_of = functools.partial(_coords, family, spans_of, row, first, shape=shape,
                                          draw=None if drop_of is None else drop_of.draw(row, head))
            params_of = functools.partial(_score_params, family, score_refs, kinds, row, head, first)
            step = functools.partial(_dq_step, walk, operands, stats, coords_of, params_of, whole, share)
            start = (nothing, nothing, (nothing,) * len(kinds), (nothing,) * len(kinds), nothing,
                     nothing + algebra.NO_SHIFT)
            # The tiles its partner reads around its own (`Family.walk_own`).
            (residual, mass, grads, weighed, keyed_mass, keyed_shift), accumulated = family.walk_own(
                streams, views, (lo, hi), streamed, index,
                lambda streams, lo, hi, counted: family.walk(
                    streams, ((step, lo, hi),), start, ((block_q, share[1]),) * (2 if walk.accurate else 1),
                    walk.interpret, first=counted))
            dq, *keyed = accumulated
            rows = (row, pl.ds(first, block_q), head)
            written = [(dq, dq_ref.at[rows])]
            if keyed_ref is not None:
                written.append((keyed[0], keyed_ref.at[rows]))
            family.stored(*zip(*written, strict=True), smem=out_smem.at[index])
            # Each row in a row's layout whatever its own (a call without a hook's keyed mass is a constant).
            summed_grads = tuple(part for part, on in zip(grads, summed, strict=True) if on)
            summed_weighed = tuple(part for part, on in zip(weighed, summed, strict=True) if on)
            parts = (residual, mass, keyed_mass, keyed_shift, *summed_grads, *summed_weighed)
            family.stored(tuple(family.as_row(part) for part in parts),
                          tuple(rows_ref.at[place, row, head, pl.ds(first, block_q)]
                                for place in range(rows_count)),
                          smem=row_smem.at[index])

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    scoped, rings = _dq_allocations(family, program.tiles, width, value_width, query.dtype, program.stages)

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    sums = jax.ShapeDtypeStruct((batch, queries, heads, share[1]), jnp.float32)
    stats_shape = jax.ShapeDtypeStruct((rows_count, batch, heads, queries), jnp.float32)
    out = family.kernel(
        entry, out_type=(sums, sums, stats_shape) if walk.accurate else (sums, stats_shape),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
        interpret=walk.interpret, approx_math=walk.approx_math,
        name=name,
        reducing_warpgroups=reducing(family, consumers, walk.spec.score_mod),
        # Its share of dQ's product, the model's this launch owns (`stated`): its `S` and `dP` recomputed.
        cost=stated(walk.spec, batch, heads, walk.spec.queries or queries, share[1], query, key, value,
                    dout))(
            query, key, value, dout, peak, prior, anchor, delta, key_index, *leaves)
    dq, keyed, rows = out if walk.accurate else (out[0], None, out[1])
    return dq, keyed, rows
