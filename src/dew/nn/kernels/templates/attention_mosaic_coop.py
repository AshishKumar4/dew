"""The attention template's Mosaic GPU fused backward over shared keys (`attention_mosaic_backward`): the fast
form's one launch of a hookless call whose consumers share one key block, each forming the logits and score
cotangents of its share of the keys, the weights and cotangents exchanged through shared memory, and each
holding its share of the head's columns of dK, dV and dQ (FlashAttention 3's sm90 backward at width 256,
hopper/flash_bwd_launch_template.h's run_mha_bwd_hdim256: two MMA warpgroups splitting dK's, dV's and dQ's
columns, one stage of dO and dS)."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import tuning
from dew.nn.kernels.templates import mosaic
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    _coords,
    _rows,
    _Spans,
    _streamed,
    _views_of,
    stated,
)
from dew.nn.kernels.templates.attention_mosaic_scores import _Backward, _tile_scores


def _coop_allocations(family: mosaic.Family, config: BackwardTiles, width: int, dtype, stages: int
                      ) -> tuple[list, dict]:
    """The shared-keys launch's allocations as `Family.scoped` takes them, in `_coop_launch`'s order: the
    program's resident keys and values; a tile's weights and score cotangents, every consumer's share of
    the keys whole rows of their swizzle, the widest whose row divides it and the products' contractions
    over the tile's queries and keys (`mosaic.swizzle_for`: 64 bytes at a share of 32); each consumer's
    fp32 stage of its adds a chunk of its columns at a time (`mosaic.staged_columns`: width 256's two
    stages of its 64 x 256 slots fit beside it); the two loads' barriers and the exchange's
    (`Family.exchange`); and its query and output-cotangent streams in `stages` slots (`Family.streams`)."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    chunk = mosaic.staged_columns(width // consumers, (jnp.float32,))
    swizzle = mosaic.swizzle_for(dtype, block_k // consumers, block_q, block_k)
    scoped = [plgpu.SMEM((block_k, width), dtype), plgpu.SMEM((block_k, width), dtype),
              mosaic.staged((2, block_q, block_k), dtype, swizzle),
              plgpu.SMEM((consumers, max(block_q, block_k), chunk), jnp.float32),
              family.load_barriers(2), *family.exchange(consumers)]
    blocks = {"queries": ((block_q, width), dtype), "douts": ((block_q, width), dtype)}
    return scoped, family.streams(blocks, stages, consumers)


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
    k_smem, v_smem = resident
    family, tiles, consumers = walk.program.family, walk.program.tiles, walk.consumers
    keys, columns = tiles.block_k // consumers, k_smem.shape[-1] // consumers
    dk_acc, dv_acc = accumulators
    q_slot, do_slot = slots["queries"], slots["douts"]
    q_first = index * tiles.block_q
    own = pl.ds(share * keys, keys)
    coords = coords_of(q_first, queries_read=walk.queries_read(masked))
    narrowed, part = _tile_scores(walk, (q_slot, do_slot), (k_smem.at[own], v_smem.at[own]), rows_of(q_first),
                                  coords, whole(index), (tiles.block_q, keys), masked=masked)
    weights_ref, cotangents_ref = exchanged.at[0], exchanged.at[1]
    exchange.claim()
    family.staged(narrowed, weights_ref.at[:, own])
    family.staged(part, cotangents_ref.at[:, own])
    exchange.publish()
    mine = pl.ds(share * columns, columns)
    dv_acc = family.products(dv_acc, weights_ref.transpose((1, 0)), do_slot.at[:, mine], walk.interpret)
    dk_acc = family.products(dk_acc, cotangents_ref.transpose((1, 0)), q_slot.at[:, mine], walk.interpret)
    add_dq(q_first, family.product(cotangents_ref, k_smem.at[:, mine], (tiles.block_q, columns)))
    exchange.release()
    return carry, (dk_acc, dv_acc)


@tuning.lowered_once(("walk",))
def _coop_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None, *,
                 walk: _Backward) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The fast form's one launch over shared keys (`BackwardTiles.shared_keys`): a program's `block_k` keys
    of a key head resident, walked by the query tiles that see them, streamed with their output cotangents
    through one ring its consumers read (`_coop_step`), under the first pass's D and the forward's
    denominator; each consumer's share of the columns of dQ added a tile at a time, and of dK and dV into
    its key head's own (`Family.added`, zeros: a key head's query heads add theirs). Returns dQ
    `[B, Qp, H, D]`, dK and dV `[B, Kp, KV, D]`, fp32."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads = key.shape[1], key.shape[2]
    program, consumers = walk.program, walk.consumers
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    columns = width // consumers
    q_steps = queries // block_q
    leaves, treedef = jax.tree.flatten(spans)

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, *stat_refs), refs = refs[:9], refs[9:]
        spans_of, refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        dq_ref, dk_ref, dv_ref, k_smem, v_smem, exchanged, out_smem, loads, staged_turn, read_turn = refs
        exchange = mosaic.Exchange(family, staged_turn, read_turn, consumers)
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        key_head = lax.div(head, heads // kv_heads)
        views = _views_of({"queries": q_ref, "douts": do_ref}, row, head)
        streams = family.opened(streams, views)
        k_first = lax.axis_index("k_tiles") * block_k
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
                block_k // consumers), shape=(block_q, block_k // consumers))
            mine = pl.ds(share * columns, columns)
            smem = out_smem.at[share]

            def add_dq(q_first, dq):
                family.added(dq, dq_ref.at[row, pl.ds(q_first, block_q), head, mine], walk.interpret,
                             smem=smem.at[pl.ds(0, block_q)])

            step = functools.partial(_coop_step, walk, share, (k_smem, v_smem), exchanged, rows_of, coords_of,
                                     whole, add_dq, exchange)

            segments = functools.partial(program.segmented, step, run)

            shapes = ((block_k, columns), (block_k, columns))
            exchange.open()
            _, (dk, dv) = family.walk_own(
                streams, views, (lo, hi), streamed, index,
                lambda streams, lo, hi, counted: family.walk(streams, segments(lo, hi), (), shapes,
                                                             walk.interpret, first=counted))
            for value, ref in ((dk, dk_ref), (dv, dv_ref)):
                family.added(value, ref.at[row, pl.ds(k_first, block_k), key_head, mine], walk.interpret,
                             smem=smem.at[pl.ds(0, block_k)])

        def consumer(index):
            mosaic.per_role(index, consumers, cooperate)

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    scoped, rings = _coop_allocations(family, program.tiles, width, query.dtype,
                                      min(program.tiles.stages, q_steps))

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    launch = family.kernel(
        entry, out_type=(), grid=(heads, keys_padded // block_k, batch),
        grid_names=("heads", "k_tiles", "batch"), consumers=consumers, interpret=walk.interpret,
        approx_math=walk.approx_math, name="attention_dkv_shared",
        # The model's products this launch owns (`stated`): dQ's, dK's, dV's and dP's; its `S` recomputed.
        cost=stated(walk.spec, batch, heads, walk.spec.queries or queries, 4 * width, query, key, value,
                    dout))
    dq = jax.new_ref(jnp.zeros((batch, queries, heads, width), jnp.float32))
    dk, dv = (jax.new_ref(jnp.zeros((batch, keys_padded, kv_heads, width), jnp.float32)) for _ in range(2))
    launch(query, key, value, dout, peak, total, anchor, delta, key_index, *leaves, dq, dk, dv)
    return jax.freeze(dq), jax.freeze(dk), jax.freeze(dv)
