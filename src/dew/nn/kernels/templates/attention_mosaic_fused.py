"""The attention template's Mosaic GPU fused backward queries first (`attention_mosaic_backward`): the fast
form's one launch of a hookless call, each consumer's keys resident over the query tiles that read them, its
dK and dV, and each tile's dQ added into dQ."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import mosaic
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    _coords,
    _rows,
    _Spans,
    _streamed,
    _views_of,
    described,
    kernel_of,
    stated,
)
from dew.nn.kernels.templates.attention_mosaic_scores import (
    _Backward,
    _dq_added,
    _launch_extents,
    _tile_scores,
)


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
    k_smem = resident[0]
    family, tiles = walk.program.family, walk.program.tiles
    shape = (tiles.block_q, tiles.block_k)
    dk_acc, dv_acc = accumulators
    q_slot, do_slot = slots["queries"], slots["douts"]
    q_first = index * tiles.block_q
    narrowed, part = _tile_scores(walk, (q_slot, do_slot), resident, rows_of(q_first), coords_of(q_first),
                                  whole(index), shape, masked=masked)
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
    (`Family.added`); where key heads are shared, each query head's dK and dV added into its key head's own
    (FlashAttention 3's grouped epilogue, hopper/epilogue_bwd.hpp's CollectiveEpilogueBwdGQA: Qwen3-8B's step
    1.7% the faster causal, 1.3% full, ET-GQA1-h100, where XLA summed fp32 copies a query head's), else
    stored (their zeros cost an ungrouped D64 step 1.7%). Returns dQ `[B, Qp, H, D]`, dK and dV
    `[B, Kp, KV, D]`, fp32."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads = key.shape[1], key.shape[2]
    leaves, treedef = jax.tree.flatten(spans)
    operands = described((query, key, value, dout))
    launch = kernel_of(_fused_kernel, "attention_dkv_fused", walk, operands, treedef,
                       hooks=(walk.spec.score_mod,))
    gradient = jax.ShapeDtypeStruct((batch, keys_padded, kv_heads, width), jnp.float32)
    grouped = heads != kv_heads
    # dQ added into zeros in place, and grouped dK and dV: one copy, or dQ each consumer's own interpreted
    # (`Family.added`), a consumer's keys its own.
    copies = walk.consumers if walk.interpret else 1
    dq = jax.new_ref(jnp.zeros((copies, batch, queries, heads, width), jnp.float32))
    operands = (query, key, value, dout, peak, total, anchor, delta, key_index, *leaves, dq)
    if not grouped:
        dk, dv = launch(*operands)
        return jax.freeze(dq).sum(axis=0), dk, dv
    dk, dv = (jax.new_ref(jnp.zeros(gradient.shape, jnp.float32)) for _ in range(2))
    launch(*operands, dk, dv)
    return jax.freeze(dq).sum(axis=0), jax.freeze(dk), jax.freeze(dv)


def _fused_kernel(walk: _Backward, operands: tuple, treedef, *, name: str, kernel_name: str | None):
    """`_fused_launch`'s kernel for its static facts (`kernel_of`): the walk, its operands' shapes and dtypes
    and the structure of its spans' arrays."""
    (query, key, value, dout, (batch, queries, heads, width), (keys_padded, kv_heads), program, consumers,
     family, (block_q, block_k)) = _launch_extents(walk, operands)
    pair = consumers * block_k
    q_steps = queries // block_q
    grouped = heads != kv_heads

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, *stat_refs), refs = refs[:9], refs[9:]
        spans_of, refs = treedef.unflatten(refs[:treedef.num_leaves]), refs[treedef.num_leaves:]
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

            segments = functools.partial(program.segmented, step, run)

            # The tiles its partner reads around its own (`Family.walk_own`); on a layout whose consumers
            # feed their rings, every tile any of the program's keys' readers reads, a tile none of a
            # consumer's own keys sees masked whole, its weights and its dQ zero (FlashAttention 3's two
            # warpgroups a CTA, each its own keys over one query stream).
            _, (dk, dv) = family.walk_own(
                streams, views, (lo, hi), streamed, index,
                lambda streams, lo, hi, counted: family.walk(streams, segments(lo, hi), (), shapes,
                                                             walk.interpret, first=counted))
            smem = out_smem.at[index, pl.ds(0, block_k)]
            if grouped:
                for value, ref in ((dk, dk_ref), (dv, dv_ref)):
                    family.added(value, ref.at[keys_view], walk.interpret, smem=smem)
            else:
                family.stored((dk, dv), (dk_ref.at[keys_view], dv_ref.at[keys_view]), smem=smem)

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    scoped, rings = _fused_allocations(family, program.tiles, width, query.dtype,
                                       min(program.tiles.stages, q_steps))

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    gradient = jax.ShapeDtypeStruct((batch, keys_padded, kv_heads, width), jnp.float32)
    return family.kernel(
        entry, out_type=() if grouped else (gradient, gradient), grid=(heads, keys_padded // pair, batch),
        grid_names=("heads", "k_tiles", "batch"), consumers=consumers, interpret=walk.interpret,
        approx_math=walk.approx_math, name=name, kernel_name=kernel_name,
        # The model's products this launch owns (`stated`): dQ's, dK's, dV's and dP's; its `S` recomputed.
        cost=stated(walk.spec, batch, heads, walk.spec.queries or queries, 4 * width, query, key, value,
                    dout))
