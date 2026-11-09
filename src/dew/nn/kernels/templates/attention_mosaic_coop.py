"""The attention template's Mosaic GPU fused backward over shared keys (`attention_mosaic_backward`): the fast
form's one launch of a call whose score hook takes no parameters and whose consumers share one key block,
each forming the logits and score cotangents of its share of the keys, the weights and cotangents exchanged
through shared memory, and each holding its share of the head's columns of dK, dV and dQ (FlashAttention
3's sm90 backward at width 256, hopper/flash_bwd_launch_template.h's run_mha_bwd_hdim256: two MMA
warpgroups splitting dK's, dV's and dQ's columns, one stage of dO and dS). fp16's weights and cotangents
exchange as two parts each at the walk's running scale, every share's (`_sixteen_step`)."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from dew.nn.kernels import tuning
from dew.nn.kernels.templates import attention_algebra as algebra, mosaic, swizzling
from dew.nn.kernels.templates.attention_mosaic_fused import staged_parts, walked_sums
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    TileWalk,
    _coords,
    _rows,
    _Spans,
    _streamed,
    _views_of,
    modelled,
    part_count,
    reducing,
    stated,
)
from dew.nn.kernels.templates.attention_mosaic_scores import _Backward, _tile_scores


def _coop_allocations(family: mosaic.Family, config: BackwardTiles, width: int, dtype, stages: int
                      ) -> tuple[list, dict]:
    """The shared-keys launch's allocations as `Family.scoped` takes them, in `_coop_launch`'s order: the
    program's resident keys and values; a tile's weights and score cotangents, every consumer's share of
    the keys whole rows of their swizzle, the widest whose row divides it and the products' contractions
    over the tile's queries and keys (`swizzling.swizzle_for`: 64 bytes at a share of 32), fp16's two parts
    each (`staged_parts`); each consumer's fp32 stage of its adds a chunk of its columns at a time
    (`mosaic.staged_columns`: width 256's two stages of its 64 x 256 slots fit beside it); the two loads'
    barriers and the exchange's (`Family.exchange`); fp16's shares' exponents, a row each of the weights'
    and of the cotangents' (`_sixteen_step`); and its query and output-cotangent streams in `stages` slots
    (`Family.streams`)."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    chunk = mosaic.staged_columns(width // consumers, (jnp.float32,))
    swizzle = swizzling.swizzle_for(dtype, block_k // consumers, block_q, block_k)
    parts = staged_parts(dtype)
    scoped = [family.staged_memory((block_k, width), dtype), family.staged_memory((block_k, width), dtype),
              family.staged_memory((2 * parts, block_q, block_k), dtype, swizzle),
              family.staged_memory((consumers, max(block_q, block_k), chunk), jnp.float32),
              family.load_barriers(2), *family.exchange(consumers)]
    if parts > 1:
        scoped.append(family.staged_memory((2, consumers, block_q), jnp.float32))
    blocks = {"queries": ((block_q, width), dtype), "douts": ((block_q, width), dtype)}
    return scoped, family.streams(blocks, stages, consumers)


def _share_scores(walk: _Backward, share: int, resident: tuple, rows_of, coords_of, whole, index, slots, *,
                  masked: bool | None, whole_operands: bool = False) -> tuple:
    """Consumer `share`'s scores of query tile `index` over its share of the resident keys and values
    (`_tile_scores`, `[bq, bk / consumers]`, `whole_operands` as it takes them), and the shares it reads
    and writes: its keys' `pl.ds` of the block, its columns' of the head, and the tile's first query."""
    k_smem, v_smem = resident
    tiles, consumers = walk.program.tiles, walk.consumers
    keys, columns = tiles.block_k // consumers, k_smem.shape[-1] // consumers
    q_first = index * tiles.block_q
    own = pl.ds(share * keys, keys)
    coords = coords_of(q_first, read=walk.places_read(masked=masked))
    scores = _tile_scores(walk, (slots["queries"], slots["douts"]), (k_smem.at[own], v_smem.at[own]),
                          rows_of(q_first), coords, whole(index), (tiles.block_q, keys), masked=masked,
                          whole_operands=whole_operands)
    return scores, own, pl.ds(share * columns, columns), q_first


def _coop_step(walk: _Backward, share: int, resident: tuple, exchanged, rows_of, coords_of, whole, add_dq,
               exchange: mosaic.Exchange, index, slots, carry, accumulators, *, masked: bool | None = None):
    """Consumer `share`'s query tile `index` over the program's resident keys and values (`resident`):
    its share of the keys' logits and score cotangents (`_scores`, `[bq, bk / consumers]`) from the query
    and output-cotangent slots, under the tile's queries' statistics (`rows_of`) and coordinates
    (`coords_of`), a tile every query sees whole (`whole(index)`) unmasked, or each tile `masked` or not
    where the walk's segment says; its weights and score cotangents staged into the exchanged block
    (`exchanged`) at its share of the keys once it claims the block (`mosaic.Exchange`: every consumer's
    reads of the last tile's done, awaited only past this tile's products and softmax), then, once every
    consumer's are published, its share of the head's columns of dV, dK and dQ over every key of the block
    (`aᵀ @ b`, `a @ b`), dQ added into dQ (`add_dq`), and the block released."""
    k_smem = resident[0]
    family, tiles = walk.program.family, walk.program.tiles
    dk_acc, dv_acc = accumulators
    q_slot, do_slot = slots["queries"], slots["douts"]
    (narrowed, part), own, mine, q_first = _share_scores(walk, share, resident, rows_of, coords_of, whole,
                                                         index, slots, masked=masked)
    weights_ref, cotangents_ref = exchanged.at[0], exchanged.at[1]
    exchange.claim()
    family.staged(narrowed, weights_ref.at[:, own])
    family.staged(part, cotangents_ref.at[:, own])
    exchange.publish()
    dv_acc = family.products(dv_acc, weights_ref.transpose((1, 0)), do_slot.at[:, mine])
    dk_acc = family.products(dk_acc, cotangents_ref.transpose((1, 0)), q_slot.at[:, mine])
    add_dq(q_first, family.product(cotangents_ref, k_smem.at[:, mine], (tiles.block_q, mine.size)))
    exchange.release()
    return carry, (dk_acc, dv_acc)


def _sixteen_step(walk: _Backward, share: int, resident: tuple, exchanged, exponents, rows_of, coords_of,
                  whole, add_dq, exchange: mosaic.Exchange, index, slots, carry, accumulators, *,
                  masked: bool | None = None):
    """`_coop_step`'s for fp16 operands, whose fp32 weights and score cotangents (`_tile_scores`'
    `whole_operands`) pass fp16's range. Every consumer stages its share's exponent of each
    (`algebra.tile_exponent`, a row of it in `exponents`) and publishes it; the walk's running scale of each,
    its last (`carry`, weights' then cotangents') and every share's tile joined, is then every consumer's
    alike; each stages its share of both as two fp16 parts at it (`algebra.parts_at`) and publishes them,
    takes dV's and dK's accumulators to the new scale (`algebra.running_shift`) before their parts'
    products issue into them, and takes the tile's dQ, both cotangent parts over its columns of the keys,
    back by the cotangents' scale (`algebra.running_restored`) before its add."""
    k_smem = resident[0]
    family, tiles, consumers = walk.program.family, walk.program.tiles, walk.consumers
    dk_acc, dv_acc = accumulators
    q_slot, do_slot = slots["queries"], slots["douts"]
    operands, own, mine, q_first = _share_scores(walk, share, resident, rows_of, coords_of, whole, index,
                                                 slots, masked=masked, whole_operands=True)
    exchange.claim()
    for operand, left in enumerate(operands):
        exponent = jnp.full((tiles.block_q,), algebra.tile_exponent(left), jnp.float32)
        family.staged(family.as_row(exponent), exponents.at[operand, share])
    exchange.publish()
    joined = tuple(functools.reduce(jnp.maximum, (jnp.max(family.rows_of(exponents.at[operand, other]))
                                                  for other in range(consumers)), last)
                   for operand, last in enumerate(carry))
    for operand, left in enumerate(operands):
        for offset, part in enumerate(algebra.parts_at(left, joined[operand])):
            family.staged(part, exchanged.at[2 * operand + offset, :, own])
    exchange.publish()
    issued = []
    for operand, (acc, rhs) in enumerate(((dv_acc, do_slot.at[:, mine]), (dk_acc, q_slot.at[:, mine]))):
        shift = algebra.running_shift(carry[operand], joined[operand])
        # The accumulator taken to the new scale (`algebra.scaled_by`, a power of two at most 1).
        acc = family.rescaled_products(acc, lambda value, shift=shift: algebra.scaled_by(value, shift), (),
                                       rhs, walk.interpret)
        for offset in range(2):
            acc = family.products(acc, exchanged.at[2 * operand + offset].transpose((1, 0)), rhs)
        issued.append(acc)
    shape = (tiles.block_q, mine.size)
    dq = (family.product(exchanged.at[2], k_smem.at[:, mine], shape)
          + family.product(exchanged.at[3], k_smem.at[:, mine], shape))
    add_dq(q_first, algebra.running_restored(dq, joined[1]))
    exchange.release()
    return joined, (issued[1], issued[0])


@tuning.lowered_once(("walk",))
def _coop_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None, *,
                 walk: _Backward) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The fast form's one launch over shared keys (`BackwardTiles.shared_keys`): a program's `block_k` keys
    of a key head resident, walked by the query tiles that see them, streamed with their output cotangents
    through one ring its consumers read (`_coop_step`), under the first pass's D and the forward's
    denominator; each consumer's share of the columns of dQ added a tile at a time, and of dK and dV into
    its key head's own (`Family.added`, zeros: a key head's query heads add theirs; dK's and dV's zeros one
    ref of both, `mosaic.stacked_zeros`, row 3180).
    Returns dQ `[B, Qp, H, D]`, dK and dV `[B, Kp, KV, D]`, fp32."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads = key.shape[1], key.shape[2]
    program, consumers = walk.program, walk.consumers
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    columns = width // consumers
    q_steps = queries // block_q
    leaves, treedef = jax.tree.flatten(spans)
    # dK's and dV's zeros one ref of both, the program's views and the host's read of one grouping.
    shapes = [(batch, keys_padded, kv_heads, width)] * 2
    groups = mosaic.stacked_groups(shapes)

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, *stat_refs), refs = refs[:9], refs[9:]
        spans_of, refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        dq_ref, dkv_ref, k_smem, v_smem, exchanged, out_smem, loads, staged_turn, read_turn, *exponents = refs
        dk_ref, dv_ref = mosaic.stacked_views([dkv_ref], groups)
        exchange = mosaic.Exchange(family, staged_turn, read_turn, consumers)
        row, head = family.program_index("batch"), family.program_index("heads")
        key_head = lax.div(head, heads // kv_heads)
        views = _views_of({"queries": q_ref, "douts": do_ref}, row, head)
        streams = family.opened(streams, views)
        k_first = family.program_index("k_tiles") * block_k
        lo, hi, whole = program.readers(k_first, q_steps, spans_of, row)
        streamed = _streamed([(lo, hi)] * consumers)
        run = program.whole_run(k_first, q_steps, spans_of)

        def rows_of(q_first):
            return tuple(_rows(family, ref, row, head, q_first, block_q) for ref in stat_refs)

        def cooperate(share: int, index):
            # Each consumer's columns its own, so each slices its tiles at offsets it holds statically.
            keys_view = (row, pl.ds(k_first, block_k), key_head)
            family.loaded(((k_ref.at[keys_view], k_smem, loads.at[0]),
                           (v_ref.at[keys_view], v_smem, loads.at[1])), issued=share == 0)
            coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first + share * (
                block_k // consumers), shape=(block_q, block_k // consumers), tile_keys=block_k)
            mine = pl.ds(share * columns, columns)
            smem = out_smem.at[share]

            def add_dq(q_first, dq):
                family.added(dq, dq_ref.at[row, pl.ds(q_first, block_q), head, mine], walk.interpret,
                             smem=smem.at[pl.ds(0, block_q)])

            if exponents:
                step = functools.partial(_sixteen_step, walk, share, (k_smem, v_smem), exchanged,
                                         exponents[0], rows_of, coords_of, whole, add_dq, exchange)
            else:
                step = functools.partial(_coop_step, walk, share, (k_smem, v_smem), exchanged, rows_of,
                                         coords_of, whole, add_dq, exchange)

            segments = functools.partial(program.segmented, step, run)

            shapes = ((block_k, columns), (block_k, columns))
            exchange.open()
            dk, dv = walked_sums(walk, streams, views, (lo, hi), streamed, index, segments, shapes,
                                 scaled=bool(exponents))
            for value, ref in ((dk, dk_ref), (dv, dv_ref)):
                family.added(value, ref.at[row, pl.ds(k_first, block_k), key_head, mine], walk.interpret,
                             smem=smem.at[pl.ds(0, block_k)])

        def consumer(index):
            mosaic.per_role(index, consumers, cooperate)

        def producer():
            family.stream(streams, views, *streamed)

        family.scheduled(consumers, consumer, producer)

    scoped, rings = _coop_allocations(family, program.tiles, width, query.dtype,
                                      min(program.tiles.stages, q_steps))

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    launch = family.kernel(
        entry, out_type=(), grid=(heads, keys_padded // block_k, batch),
        grid_names=("heads", "k_tiles", "batch"), consumers=consumers, interpret=walk.interpret,
        approx_math=walk.approx_math, name="attention_dkv_shared",
        # A score hook's warpgroups reserve their reduction scratch, and fp16's running scale's (`reducing`).
        reducing_warpgroups=reducing(family, consumers, walk.reduces, scaled=staged_parts(query.dtype) > 1),
        # The products this launch issues (`stated`): its `S` recomputed, dP's, dV's, and each part of dK's
        # and of dQ's (`part_count`); the model's of them it owns (`modelled`): dQ's, dK's, dV's and dP's.
        cost=stated(walk.spec, batch, heads,
                    width + 2 * value.shape[-1] + 2 * part_count(query.dtype, walk.accurate) * width,
                    query, key, value, dout,
                    tile_walk=TileWalk((queries, keys_padded), (block_q, block_k), keyed=True)),
        model_flops=modelled(walk.spec, batch, heads, walk.spec.queries or queries,
                             2 * width + 2 * value.shape[-1]))
    dq = jax.new_ref(jnp.zeros((batch, queries, heads, width), jnp.float32))
    dkv = mosaic.stacked_zeros(shapes, groups)
    launch(query, key, value, dout, peak, total, anchor, delta, key_index, *leaves, dq, *dkv)
    dk, dv = mosaic.stacked_read(dkv, groups)
    return jax.freeze(dq), dk, dv
