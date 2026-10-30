"""The attention template's Mosaic GPU fused backward queries first (`attention_mosaic_backward`): the fast
form's one launch of a hookless call, each consumer's keys resident over the query tiles that read them, its
dK and dV, and each tile's dQ added into dQ."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import attention_algebra as algebra, mosaic
from dew.nn.kernels.templates.attention import by_power
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    _coords,
    _rows,
    _Spans,
    _streamed,
    _views_of,
    stated,
)
from dew.nn.kernels.templates.attention_mosaic_scores import _Backward, _dq_added, _scores


def _fused_allocations(family: mosaic.Family, config: BackwardTiles, width: int, dtype, stages: int
                       ) -> tuple[list, dict]:
    """The fused launch's allocations as `Family.scoped` takes them, in `_fused_launch`'s order: its
    consumers' resident keys and values; a tile's weights and score cotangents staged for the transposed
    products' left operands and dQ's; dK's and dV's output stage, which a tile's dQ adds take too (fp32, the
    head widths equal, `MosaicBackward.legal`); the two loads' barriers; and its query and output-cotangent
    streams in `stages` slots (`Family.streams`)."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    scoped = [plgpu.SMEM((consumers, block_k, width), dtype), plgpu.SMEM((consumers, block_k, width), dtype),
              plgpu.SMEM((consumers, 2, block_q, block_k), dtype),
              plgpu.SMEM((consumers, max(block_q, block_k), width), jnp.float32),
              family.load_barriers(consumers), family.load_barriers(consumers)]
    blocks = {"queries": ((block_q, width), dtype), "douts": ((block_q, width), dtype)}
    return scoped, family.streams(blocks, stages, consumers)

def _fused_step(walk: _Backward, resident: tuple, rows_of, staged, coords_of, whole, add_dq, index, slots,
                carry, accumulators, *, masked: bool | None = None):
    """A fused dK/dV consumer's query tile `index`, queries first (`[bq, bk]`, the hookless fast form's
    one launch): its scores and score cotangents from the query and output cotangent slots (`slots`) and its
    resident keys and values (`resident`), under the tile's queries' statistics (`rows_of(first query)`) and
    coordinates (`coords_of(first query)`), a tile every query sees whole (`whole(index)`) unmasked, or
    each tile `masked` or not where the walk's segment says (a run of whole tiles, `whole_run`). Its
    weights and score cotangents staged in `staged`, whose transposes are dV's and dK's left operands
    (`algebra.value_operands`, `cotangent_product`'s `aᵀ @ b`) issued into `accumulators` in place, and the
    cotangents times the resident keys the tile's dQ, added into dQ (`add_dq(first query, dQ)`)."""
    k_smem, v_smem = resident
    family, tiles = walk.program.family, walk.program.tiles
    shape = (tiles.block_q, tiles.block_k)
    dk_acc, dv_acc = accumulators
    q_slot, do_slot = slots["queries"], slots["douts"]
    q_first = index * tiles.block_q
    # The tile's queries' statistics read before its products, which their loads run under (ET-A05AB5/6-h100:
    # read past them, 2.32 ms the launch at Qwen3-8B causal; before them, 2.05 against base's 2.09).
    rows = rows_of(q_first)
    # The logits from the query taken by its power in registers (`_Backward.scaled`), dK from the call's own.
    keys = k_smem.transpose((1, 0))
    s = (family.registers_product(by_power(family.tile_of(q_slot), walk.spec.prescale), keys, walk.interpret)
         if walk.scaled else family.product(q_slot, keys, shape))
    dp = family.product(do_slot, v_smem.transpose((1, 0)), shape)

    def scores(*, masked: bool):
        # A tile every query sees whole (`masked` False) takes no mask (`_dq_step`'s choice).
        exponentials, weights, cotangents, _, _ = _scores(walk, coords_of(q_first), (), s, dp, *rows,
                                                          masked=masked)
        return exponentials, weights, cotangents

    if masked is None:
        exponentials, weights, cotangents = lax.cond(whole(index), functools.partial(scores, masked=False),
                                                     functools.partial(scores, masked=True))
    else:
        exponentials, weights, cotangents = scores(masked=masked)
    ((part, _),) = algebra.cotangent_parts(cotangents * walk.grad_scale, q_slot.dtype, accurate=False)
    narrowed = algebra.value_weights(exponentials, weights, do_slot.dtype)
    family.staged(narrowed, staged.at[0])
    dv_acc = family.products(dv_acc, staged.at[0].transpose((1, 0)), do_slot, walk.interpret)
    cotangents_staged = staged.at[1]
    family.staged(part, cotangents_staged)
    dk_acc = family.products(dk_acc, cotangents_staged.transpose((1, 0)), q_slot, walk.interpret)
    # The tile's dQ, its staged cotangents times the resident keys (`algebra.cotangent_product`'s `a @ b`),
    # added into dQ: the query tile's other key blocks add theirs.
    add_dq(q_first, family.product(cotangents_staged, k_smem, (tiles.block_q, k_smem.shape[-1])))
    return carry, (dk_acc, dv_acc)

def _fused_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None, *,
                  walk: _Backward) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The fast form's one launch of a hookless call (`BackwardTiles.fused`), on the dQ launch's layout:
    each consumer's `block_k` keys of a query head resident, walked by the query tiles that see them,
    streamed with their output cotangents through a ring each, queries first (`_fused_step`), under the
    first pass's D and the forward's denominator, each tile's dQ added into dQ atomically
    (`Family.added`), and a key head's query heads' dK and dV added into its own (FlashAttention 3's grouped
    epilogue, hopper/epilogue_bwd.hpp's CollectiveEpilogueBwdGQA: XLA's sum of them over fp32 copies a query
    head cost 0.169 ms of Qwen3-8B's 3.62 ms causal step, ET-OVERLAP1-h100). Returns dQ `[B, Qp, H, D]`, dK
    and dV `[B, Kp, KV, D]`, fp32."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads = key.shape[1], key.shape[2]
    program, consumers = walk.program, walk.consumers
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_k
    q_steps = queries // block_q
    leaves, treedef = jax.tree.flatten(spans)

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, *stat_refs), refs = refs[:9], refs[9:]
        spans_of, refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        dq_ref, dk_ref, dv_ref, k_smem, v_smem, staged, out_smem, k_loaded, v_loaded = refs
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        key_head = lax.div(head, heads // kv_heads)
        views = _views_of({"queries": q_ref, "douts": do_ref}, row, head)
        streams = family.opened(streams, views)
        pair_first = lax.axis_index("k_tiles") * pair
        streamed = _streamed([program.readers(pair_first + index * block_k, q_steps, spans_of, row)[:2]
                              for index in range(consumers)])

        def rows_of(q_first):
            return tuple(_rows(family, ref, row, head, q_first, block_q) for ref in stat_refs)

        def consumer(index):
            k_first = pair_first + index * block_k
            keys_view = (row, pl.ds(k_first, block_k), key_head)
            family.loaded(((k_ref.at[keys_view], k_smem.at[index], k_loaded.at[index]),
                           (v_ref.at[keys_view], v_smem.at[index], v_loaded.at[index])))
            lo, hi, whole = program.readers(k_first, q_steps, spans_of, row)
            coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first,
                                          shape=(block_q, block_k))
            adds = functools.partial(_dq_added, walk, dq_ref, out_smem, (row, head), index)
            step = functools.partial(_fused_step, walk, (k_smem.at[index], v_smem.at[index]), rows_of,
                                     staged.at[index], coords_of, whole, adds)
            shapes = ((block_k, width), (block_k, width))
            run = program.whole_run(k_first, q_steps, spans_of)

            def segments(lo, hi):
                if run is None:
                    return ((step, lo, hi),)
                # Its whole tiles one segment of their own, unmasked, and the tiles around them masked: no
                # step chooses at run time, whose two branches' registers join (FlashAttention 3's masked
                # and unmasked iterations).
                start, end = (jnp.clip(bound, lo, hi) for bound in run)
                return ((functools.partial(step, masked=True), lo, start),
                        (functools.partial(step, masked=False), start, end),
                        (functools.partial(step, masked=True), end, hi))

            # The tiles its partner reads around its own (`Family.walk_own`); on a layout whose consumers
            # feed their rings, every tile any of the program's keys' readers reads, a tile none of a
            # consumer's own keys sees masked whole, its weights and its dQ zero (FlashAttention 3's two
            # warpgroups a CTA, each its own keys over one query stream).
            _, (dk, dv) = family.walk_own(
                streams, views, (lo, hi), streamed, index,
                lambda streams, lo, hi, counted: family.walk(streams, segments(lo, hi), (), shapes,
                                                             walk.interpret, first=counted))
            smem = out_smem.at[index, pl.ds(0, block_k)]
            for value, ref in ((dk, dk_ref), (dv, dv_ref)):
                family.added(value, ref.at[keys_view], walk.interpret, smem=smem)

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    scoped, rings = _fused_allocations(family, program.tiles, width, query.dtype,
                                       min(program.tiles.stages, q_steps))

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    launch = family.kernel(
        entry, out_type=(), grid=(heads, keys_padded // pair, batch),
        grid_names=("heads", "k_tiles", "batch"), consumers=consumers, interpret=walk.interpret,
        approx_math=walk.approx_math, name="attention_dkv_fused",
        # The model's products this launch owns (`stated`): dQ's, dK's, dV's and dP's; its `S` recomputed.
        cost=stated(walk.spec, batch, heads, walk.spec.queries or queries, 4 * width, query, key, value,
                    dout))
    # dQ, dK and dV added into zeros in place: one copy, or dQ each consumer's own interpreted
    # (`Family.added`), a consumer's keys its own.
    copies = consumers if walk.interpret else 1
    dq = jax.new_ref(jnp.zeros((copies, batch, queries, heads, width), jnp.float32))
    dk, dv = (jax.new_ref(jnp.zeros((batch, keys_padded, kv_heads, width), jnp.float32)) for _ in range(2))
    launch(query, key, value, dout, peak, total, anchor, delta, key_index, *leaves, dq, dk, dv)
    return jax.freeze(dq).sum(axis=0), jax.freeze(dk), jax.freeze(dv)
