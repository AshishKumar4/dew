"""The attention template's Mosaic GPU fused backward queries first (`attention_mosaic_backward`): the fast
form's one launch of a call whose score hook takes no parameters, each consumer's keys resident over the
query tiles that read them, its dK and dV, and each tile's dQ added into dQ."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew._devices import deterministic_ops_requested
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import mosaic, resources
from dew.nn.kernels.templates.attention import Dropout
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    TileWalk,
    _coords,
    _rows,
    _Spans,
    _streamed,
    _views_of,
    launch_scope,
    modelled,
    part_count,
    reducing,
    stated,
)
from dew.nn.kernels.templates.attention_mosaic_scores import _Backward, _dq_added, _tile_scores

STATISTICS = {"peak": jnp.float32, "total": jnp.float32, "anchor": jnp.float32, "delta": jnp.float32,
              "key_index": jnp.int32}
"""A query tile's per-row statistics the fused launch streams beside its queries and output cotangents, in
`_fused_launch`'s operands' order, and their dtypes: the forward's maxima and denominators, D's anchor and
its delta, and the anchor keys."""


def streams_statistics(family: mosaic.Family, config: BackwardTiles, width: int, dtype, stages: int, *,
                       reduced: int) -> bool:
    """Whether the fused launch at `config` streams its query tiles' statistics (`STATISTICS`) beside them:
    where its copies move a vector (`Transport.copies_vectors`) and its block holds their rings beside its
    own allocations (the GPU's `shared_limit`, `launch_scope`); else a step reads them from global memory
    (`_rows`). At width 128 their rings' 2,560 bytes put FlashAttention 3's 64 x 64 launch on two consumers
    past an H100's 232,448, where the derivation then took one consumer (et-bwdderive). `reduced` the
    launch's warpgroups reserving reduction scratch (`reducing`: a score hook's)."""
    if not family.copies_vectors:
        return False
    limit = generation.gpu_target().shared_limit
    allocations = _fused_allocations(family, config, width, dtype, stages, streamed=True, reduced=reduced)
    return limit is None or resources.shared_bytes(
        family, launch_scope(family, allocations, reduced, reduces=False)) <= limit


def stores_typed(family: mosaic.Family, config: BackwardTiles, width: int, dtype, stages: int, *,
                 grouped: bool, reduced: int) -> bool:
    """Whether the fused launch at `config` stores dK and dV in the operands' `dtype` (`_fused_allocations`'
    `typed`): where no key head is shared (not `grouped`, whose adds sum fp32) and its block holds the typed
    stage beside its own allocations (the GPU's `shared_limit`, `launch_scope`: on a layout whose stage is
    held across the walks, beside it, 32 KB at 64 x 64 and width 128 past an H100's 232,448 bytes,
    fa9dbcf122);
    else it stores fp32 and XLA converts them after the launch (11-06: 35 us of a step's glue). `reduced`
    as `streams_statistics` takes it."""
    if grouped:
        return False
    limit = generation.gpu_target().shared_limit
    allocations = _fused_allocations(family, config, width, dtype, stages, typed=True, reduced=reduced)
    return limit is None or resources.shared_bytes(
        family, launch_scope(family, allocations, reduced, reduces=False)) <= limit


def _fused_allocations(family: mosaic.Family, config: BackwardTiles, width: int, dtype, stages: int, *,
                       reduced: int, streamed: bool | None = None, typed: bool = False
                       ) -> tuple[list, dict, list]:
    """The fused launch's allocations as `Family.scoped` takes them, in `_fused_launch`'s order: its
    consumers' resident keys and values; a tile's weights and score cotangents staged for the transposed
    products' left operands and dQ's; dK's and dV's output stage (fp32, the head widths equal,
    `MosaicBackward.legal`), held across the walks where a tile's dQ adds read it
    (`Transport.adds_through_shared`); the two loads' barriers; its query and output-cotangent streams in
    `stages` slots (`Family.streams`), and where it streams them (`streamed`, `streams_statistics` by default,
    its warpgroups reserving `reduced` reduction scratch) its query tiles' per-row statistics (`STATISTICS`) a
    stream each beside them, which a step reads from shared memory (elsewhere from global memory, `_rows`),
    loaded ahead with its tile as FlashAttention 3's producer loads LSE and dPsum
    (hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp's `smem_lse`, `smem_dpsum`), not from global memory once the
    step starts (ET-TPLPROF-h100: the fused launch stalled on long scoreboards 46.2k against FlashAttention
    3's 21.8k); and the output stage scoped after the walks where no add reads it, the walks' slots its memory
    (`Family.scoped`'s `after_walks`: two mma.sync consumers' 64 x 64 launch then fits an A100, FS-4324-TWO2's
    229,424 bytes held whole against its 166,912); where dK and dV leave `typed` (`stores_typed`), in `dtype`
    through a stage of their own after the walks, rounded once there."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    stage = plgpu.SMEM((consumers, max(block_q, block_k), width), jnp.float32)
    held = family.adds_through_shared
    scoped = [plgpu.SMEM((consumers, block_k, width), dtype), plgpu.SMEM((consumers, block_k, width), dtype),
              plgpu.SMEM((consumers, 2, block_q, block_k), dtype), *([stage] if held else []),
              family.load_barriers(consumers), family.load_barriers(consumers)]
    if streamed is None:
        streamed = streams_statistics(family, config, width, dtype, stages, reduced=reduced)
    blocks = {"queries": ((block_q, width), dtype), "douts": ((block_q, width), dtype),
              **({name: ((block_q,), kind) for name, kind in STATISTICS.items()} if streamed else {})}
    out = plgpu.SMEM((consumers, block_k, width), dtype) if typed else stage
    return scoped, family.streams(blocks, stages, consumers), [] if held and not typed else [out]

def _fused_step(walk: _Backward, resident: tuple, rows_of, staged, coords_of, whole, add_dq, index, slots,
                carry, accumulators, *, masked: bool | None = None):
    """A fused dK/dV consumer's query tile `index`, queries first (`[bq, bk]`, the fast form's one
    launch): its scores and score cotangents from the query and output cotangent slots (`slots`) and its
    resident keys and values (`resident`), under the tile's queries' statistics (`rows_of(slots, first
    query)`, their slots' where streamed) and
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
    coords = coords_of(q_first, read=walk.places_read(masked=masked))
    narrowed, part = _tile_scores(walk, (q_slot, do_slot), resident, rows_of(slots, q_first), coords,
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

def _fused_out(family: mosaic.Family, sums: tuple, views: tuple, index, rows: int, stage, *, grouped: bool,
               interpret: bool, ticket: tuple | None = None) -> None:
    """A fused consumer `index`'s dK and dV `sums` out to their `views` through its rows of the output
    `stage`: added into the key heads' sums where query heads share them (`grouped`), in their heads' order
    with a `ticket` (deterministic mode: its key tile's semaphore and its head's turn, `Family.ticketed`),
    else stored in the stage's dtype (`stores_typed`); once the walk's dQ adds have read the stage, whose
    slots they took (`_dq_added`)."""
    if not interpret:
        family.store_wait(0)
    smem = stage.at[index, pl.ds(0, rows)]
    if grouped:
        def added():
            for value, view in zip(sums, views, strict=True):
                family.added(value, view, interpret, smem=smem)

        if ticket is None:
            added()
        else:
            ready, turn = ticket
            family.ticketed(ready, turn, added)
    else:
        family.stored(tuple(value.astype(stage.dtype) for value in sums), views, smem=smem)


def _statistics(family: mosaic.Family, refs: list, row, head, block_q: int, *, streamed: bool
                ) -> tuple[dict, Callable]:
    """A fused program's query tiles' statistics (`STATISTICS`, their `refs` at its batch `row` and `head`):
    their views, streamed beside each tile where the launch streams them (`streams_statistics`), and
    `rows_of(slots, first query)`, a tile's rows from its slots, else from global memory (`_rows`)."""
    if not streamed:
        return {}, lambda slots, first: tuple(_rows(family, ref, row, head, first, block_q) for ref in refs)
    views = {name: (ref.at[row, head], lambda j: (j,)) for name, ref in zip(STATISTICS, refs, strict=True)}
    return views, lambda slots, first: tuple(family.rows_of(slots[name]) for name in STATISTICS)


def _turns(family: mosaic.Family, interpret: bool, dq_tiles: tuple, dkv_tiles: tuple, *, deterministic: bool,
           grouped: bool) -> tuple:
    """Deterministic mode's turns (`Family.ticketed`, `partials_ready`'s semaphores): each query tile's dQ
    adds counted (`dq_tiles`, `[B, H, query tiles]`), in their key tiles' order, and each key tile's grouped
    dK and dV adds (`dkv_tiles`, `[B, KV, key tiles]`), in their query heads'; None for each outside it."""
    if not deterministic:
        return None, None
    return (family.partials_ready(dq_tiles, interpret),
            family.partials_ready(dkv_tiles, interpret) if grouped else None)

@tuning.lowered_once(("walk",))
def _fused_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None,
                  drop: Dropout | None, *, walk: _Backward) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The fast form's one launch of a call whose score hook, if any, takes no parameters
    (`BackwardTiles.fused`), dropping where `drop` draws (under the dropping call's D,
    `gradient.centre_finish`), on the dQ launch's layout: each consumer's `block_k` keys of a query head
    resident, walked by the query tiles that see them, streamed with their output cotangents through a ring
    each, queries first (`_fused_step`), under the first pass's D and the forward's denominator (each query
    tile's streamed beside it, `STATISTICS`), each tile's dQ added into dQ atomically (`Family.added`); where
    key heads are shared, each query head's dK and dV added into its key head's own (FlashAttention 3's
    grouped epilogue, hopper/epilogue_bwd.hpp's CollectiveEpilogueBwdGQA: Qwen3-8B's step 1.7% the faster
    causal, 1.3% full, ET-GQA1-h100, where XLA summed fp32 copies a query head's), into one ref of both
    (`_fused_called`), else stored (their zeros cost an ungrouped D64 step 1.7%). Returns dQ `[B, Qp, H, D]`,
    dK and dV `[B, Kp, KV, D]`: dQ fp32, dK and dV the keys' and values' dtype where it stores them so
    (`stores_typed`), fp32 otherwise."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads = key.shape[1], key.shape[2]
    program, consumers = walk.program, walk.consumers
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_k
    q_steps = queries // block_q
    grouped = heads != kv_heads
    # Deterministic mode orders every add across programs (`Family.ticketed`; `MosaicBackward.legal`).
    deterministic, group = deterministic_ops_requested(), heads // kv_heads
    leaves, treedef = jax.tree.flatten((spans, drop))
    # Grouped dK's and dV's zeros one ref of both, the program's views and the host's read of one grouping.
    shapes = [(batch, keys_padded, kv_heads, width)] * 2
    groups = mosaic.stacked_groups(shapes)

    def body(*refs, streams, after=None):
        (q_ref, k_ref, v_ref, do_ref, *stat_refs), refs = refs[:9], refs[9:]
        (spans_of, drop_of), refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        if grouped:
            dq_ref, dkv_ref, *refs = refs
            dk_ref, dv_ref = mosaic.stacked_views([dkv_ref], groups)
        else:
            dq_ref, dk_ref, dv_ref, *refs = refs
        # The output stage where the walks' adds read it; else the epilogue's own, scoped after the walks.
        k_smem, v_smem, staged, *out_smem, k_loaded, v_loaded = refs
        out_smem = out_smem[0] if out_smem else None
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        key_head = lax.div(head, group)
        statistics, rows_of = _statistics(family, stat_refs, row, head, block_q, streamed=stats_streamed)
        dq_ready, dkv_ready = _turns(family, walk.interpret, (batch, heads, q_steps),
                                     (batch, kv_heads, keys_padded // block_k), deterministic=deterministic,
                                     grouped=grouped)
        views = {**_views_of({"queries": q_ref, "douts": do_ref}, row, head), **statistics}
        streams = family.opened(streams, views)
        pair_first = lax.axis_index("k_tiles") * pair
        streamed = _streamed([program.readers(pair_first + index * block_k, q_steps, spans_of, row)[:2]
                              for index in range(consumers)])

        def consumer(index):
            k_first = pair_first + index * block_k
            keys_view = (row, pl.ds(k_first, block_k), key_head)
            family.loaded(((k_ref.at[keys_view], k_smem.at[index], k_loaded.at[index]),
                           (v_ref.at[keys_view], v_smem.at[index], v_loaded.at[index])))
            lo, hi, whole = program.readers(k_first, q_steps, spans_of, row)
            coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first,
                                          shape=(block_q, block_k),
                                          draw=None if drop_of is None else drop_of.draw(row, head))
            k_tile = lax.div(k_first, block_k)
            adds = functools.partial(_dq_added, walk, dq_ref, out_smem, (row, head), index,
                                     ticket=(dq_ready, k_tile) if deterministic else None)
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
            outs = (dk_ref.at[keys_view], dv_ref.at[keys_view])
            turn = (None if dkv_ready is None else dkv_ready.at[row, key_head, k_tile], lax.rem(head, group))
            finished = functools.partial(_fused_out, family, (dk, dv), outs, index, block_k, grouped=grouped,
                                         interpret=walk.interpret, ticket=turn if deterministic else None)
            if after is None:
                finished(out_smem)
            else:
                after(finished)

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    stages = min(program.tiles.stages, q_steps)
    # A score hook's warpgroups reserve their reduction scratch (`reducing`).
    reduced = reducing(family, consumers, walk.spec.score_mod)
    stats_streamed = streams_statistics(family, program.tiles, width, query.dtype, stages, reduced=reduced)
    typed = stores_typed(family, program.tiles, width, query.dtype, stages, grouped=grouped, reduced=reduced)
    scoped, rings, after = _fused_allocations(family, program.tiles, width, query.dtype, stages,
                                              reduced=reduced, streamed=stats_streamed, typed=typed)

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings, after_walks=tuple(after))

    gradients = tuple(jax.ShapeDtypeStruct((batch, keys_padded, kv_heads, width),
                                           array.dtype if typed else jnp.float32) for array in (key, value))
    launch = family.kernel(
        entry, out_type=() if grouped else gradients, grid=(heads, keys_padded // pair, batch),
        grid_names=("heads", "k_tiles", "batch"), consumers=consumers, interpret=walk.interpret,
        approx_math=walk.approx_math, name="attention_dkv_fused", reducing_warpgroups=reduced,
        # The products this launch issues (`stated`): its `S` recomputed, dP's, dV's, and each part of dK's
        # and of dQ's (`part_count`); the model's of them it owns (`modelled`): dQ's, dK's, dV's and dP's.
        cost=stated(walk.spec, batch, heads,
                    width + 2 * value.shape[-1] + 2 * part_count(query.dtype, walk.accurate) * width,
                    query, key, value, dout,
                    tile_walk=TileWalk((queries, keys_padded), (block_q, block_k), consumers, keyed=True,
                                       common=family.feeds_itself)),
        model_flops=modelled(walk.spec, batch, heads, walk.spec.queries or queries,
                             2 * width + 2 * value.shape[-1]))
    operands = (query, key, value, dout, peak, total, anchor, delta, key_index, *leaves)
    return _fused_called(launch, operands, (shapes, groups), copies=consumers if walk.interpret else 1,
                         grouped=grouped)


def _fused_called(launch: Callable, operands: tuple, stacked: tuple, *, copies: int, grouped: bool
                  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The fused launch called on its `operands` (`_fused_launch`'s): dQ added into zeros in place, and
    grouped dK and dV in one ref of both (`mosaic.stacked_zeros`, `[2, B, Kp, KV, D]`, `stacked` their shapes
    and grouping: where XLA wraps the launch in a dynamic-slice fusion, a step's stacked forward rows sliced
    for this launch's operands, two refs of equal zeros were one buffer, dK and dV each the sum of both, row
    3180, ET-3180-PLAIN-h100): one copy of dQ, or each consumer's own interpreted (`Family.added`,
    `copies`), a consumer's keys its own."""
    query = operands[0]
    batch, queries, heads, width = query.shape
    dq = jax.new_ref(jnp.zeros((copies, batch, queries, heads, width), jnp.float32))
    if not grouped:
        dk, dv = launch(*operands, dq)
        return jax.freeze(dq).sum(axis=0), dk, dv
    dkv = mosaic.stacked_zeros(*stacked)
    launch(*operands, dq, *dkv)
    dk, dv = mosaic.stacked_read(dkv, stacked[1])
    return jax.freeze(dq).sum(axis=0), dk, dv
