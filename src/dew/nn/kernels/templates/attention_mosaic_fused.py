"""The attention template's Mosaic GPU fused backward queries first (`attention_mosaic_backward`): the fast
form's one launch of a call whose score hook takes no parameters, each consumer's keys resident over the
query tiles that read them, its dK and dV, and each tile's dQ added into dQ."""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from dew.nn.kernels import generation, launches, tuning
from dew.nn.kernels.templates import (
    across,
    attention_algebra as algebra,
    mosaic,
    outputs,
    resources,
    stacked,
    stores,
    swizzling,
)
from dew.nn.kernels.templates.attention import Dropout
from dew.nn.kernels.templates.attention_mosaic_program import (
    STATISTICS,
    BackwardTiles,
    TileWalk,
    _coords,
    _rows,
    _Spans,
    _streamed,
    _views_of,
    head_first,
    head_flat,
    head_layout,
    head_slice,
    head_unlayout,
    launch_scope,
    modelled,
    reducing,
    stated,
    streamed_statistics_order,
    transposed_swizzles,
)
from dew.nn.kernels.templates.attention_mosaic_scores import (
    _Backward,
    _dq_added,
    _parts_dq,
    _rescaled_parts,
    _splits_scores,
    _tile_scores,
)


@dataclasses.dataclass(frozen=True)
class FusedStaging:
    """The fused launch's shared-memory staging plan for one target's `shared_limit`: its dQ add stage's
    buffer count, its streamed per-row statistics count, whether dK/dV store typed, and the resulting
    `(scoped, streams, after)` allocations shared by resource planning (`MosaicBackward._scopes`) and
    emission (`_fused_launch`, row 5272)."""

    buffers: int
    streamed: int
    typed: bool
    allocations: tuple[list, dict, list]


def dq_summed(family: mosaic.Family, consumers: int) -> bool:
    """Whether a fused launch's consumers sum each query tile's dQ between them before one add
    (`across.summed_across`): consumers each holding their own keys and walking every streamed step paired
    (`Transport.walks_streamed`), on a family whose adds are element-wise atomics in global memory, no
    stage between them (`Transport.adds_through_shared` false, an A100's `red.global.add.f32`), whose count
    the sum halves: FlashAttention 2's one add a query tile a 128-key block, where an A100's dK/dV launch of
    one consumer spent 38 to 51% of its unissued cycles throttled on the global queue, 90 to 95% of them at
    dQ's adds, 2.05 to 2.12 times FlashAttention 2's (M-NA-9-a100, row 5272)."""
    return consumers > 1 and family.walks_streamed and not family.adds_through_shared


def fused_reduces(family: mosaic.Family, reduced: int) -> bool:
    """Whether a fused backward launch passes `reduces=True` to `Family.kernel`: `False` when the family
    budgets implicit reductions via `warpgroup_reduction_scratch` (so `mma.sync.fed` does not double-charge
    `reduction_scratch` + `warpgroup_reduction_scratch`), and `bool(reduced)` on single-warpgroup `mma.sync`
    whose `warpgroup_reduction_scratch` is zero."""
    return bool(reduced) and not family.warpgroup_reduction_scratch


def _fit_streamed[A: tuple](family: mosaic.Family, limit: int | None, reduced: int, max_streamed: int,
                            allocate: Callable[[int], A]) -> tuple[int, A]:
    """Largest `streamed` count in `max_streamed..0` whose `allocate(streamed)` fits `limit` (row 11996)."""
    reduces = fused_reduces(family, reduced)
    for count in range(max_streamed, 0, -1):
        allocs = allocate(count)
        if limit is None or resources.shared_bytes(
                family, launch_scope(family, allocs, reduced, reduces=reduces)) <= limit:
            return count, allocs
    return 0, allocate(0)


def fused_staging(family: mosaic.Family, config: BackwardTiles, width: int, dtype, stages: int, *,
                  reduced: int, grouped: bool, rounded: bool, anchored: bool = True,
                  target: generation.GpuTarget | None = None) -> FusedStaging:
    """Resolve the fused launch's staging plan once against `target`'s `shared_limit` (defaulting to
    `generation.gpu_target()`): double-buffered chunked dQ add stage when all active streamed statistics
    still fit, else the most streamed statistics that fit at single-buffered staging, and typed dK/dV
    stores when ungrouped, rounded, and fitting within the same limit (rows 5272, 11996)."""
    limit = (generation.gpu_target() if target is None else target).shared_limit
    reduces = fused_reduces(family, reduced)

    def fits(allocs: tuple[list, dict, list]) -> bool:
        return limit is None or resources.shared_bytes(
            family, launch_scope(family, allocs, reduced, reduces=reduces)) <= limit
    active = streamed_statistics_order(anchored=anchored) if family.copies_vectors else ()
    held = family.adds_through_shared
    stage_width = stores.staged_columns(width, (jnp.float32,)) if held and width <= 128 else width
    can_double = held and stage_width < width and max(config.block_q, config.block_k) == config.block_q
    buffers = 1
    allocs: tuple[list, dict, list] | None = None
    streamed = 0
    if can_double:
        trial = _fused_allocations(family, config, width, dtype, stages,
                                   streamed=len(active), typed=False, anchored=anchored, buffers=2)
        if fits(trial):
            buffers, streamed, allocs = 2, len(active), trial
    if allocs is None:
        streamed, allocs = _fit_streamed(
            family, limit, reduced, len(active),
            lambda count: _fused_allocations(family, config, width, dtype, stages,
                                             streamed=count, typed=False, anchored=anchored, buffers=1))
    typed = False
    if not grouped and rounded:
        typed_allocs = _fused_allocations(family, config, width, dtype, stages,
                                          streamed=streamed, typed=True, anchored=anchored, buffers=buffers)
        if fits(typed_allocs):
            typed, allocs = True, typed_allocs
    return FusedStaging(buffers=buffers, streamed=streamed, typed=typed, allocations=allocs)


def streamed_statistics(family: mosaic.Family, config: BackwardTiles, width: int, dtype, stages: int, *,
                        reduced: int, allocations_of: Callable[..., tuple[list, dict]] | None = None,
                        anchored: bool = True, target: generation.GpuTarget | None = None) -> int:
    """How many of its query tiles' statistics (`streamed_statistics_order`, `peak`/`total`/`delta` first) a
    fused or shared-keys launch at `config` streams beside them (`_fit_streamed`, row 11996)."""
    if not family.copies_vectors:
        return 0
    if allocations_of is None:
        return fused_staging(family, config, width, dtype, stages, reduced=reduced,
                             grouped=True, rounded=False, anchored=anchored, target=target).streamed
    limit = (generation.gpu_target() if target is None else target).shared_limit
    active = streamed_statistics_order(anchored=anchored)
    return _fit_streamed(
        family, limit, reduced, len(active),
        lambda count: allocations_of(family, config, width, dtype, stages, streamed=count, reduced=reduced),
    )[0]


def _fused_gradients(family: mosaic.Family, key: jax.Array, value: jax.Array, width: int, *,
                     typed: bool) -> tuple:
    """Fused launch dK/dV `outputs.Output` declarations matching `head_flat`/`head_first` axis order."""
    def grad_out(array: jax.Array):
        dtype = array.dtype if typed else jnp.float32
        b, s, h = (outputs.axis(array, index) for index in range(3))
        if head_flat(family, dtype, width):
            return outputs.output(dtype, b, s, h * width)
        return outputs.output(dtype, *((b, h, s) if head_first(family, dtype, width) else (b, s, h)), width)

    return grad_out(key), grad_out(value)
def walked_sums(walk: _Backward, streams, views, reach: tuple, streamed: tuple, index, segments,
                shapes: tuple, *, scaled: bool, common: bool = False) -> tuple[jax.Array, jax.Array]:
    """A fused consumer `index`'s dK and dV sums, its walk over `reach` among the tiles its program streams
    (`Family.walk_own`, `segments(lo, hi)` its steps, `shapes` its sums; `common` where every consumer's
    segments are the same steps, `_walked_run`'s paired ones, a pipeline a segment); where `scaled`, fp16's
    running scales carried, dV's and dK's (`_sixteen_step`'s and `attention_mosaic_coop`'s), and its sums
    taken back by them (`algebra.running_restored`)."""
    family = walk.program.family
    start = (jnp.float32(-jnp.inf),) * 2 if scaled else ()
    scales, (dk, dv) = family.walk_own(
        streams, views, reach, streamed, index,
        lambda streams, lo, hi, counted: family.walk(streams, segments(lo, hi), start, shapes, walk.interpret,
                                                     first=counted), common=common)
    if scales:
        dv, dk = (algebra.running_restored(sums, scale) for sums, scale in zip((dv, dk), scales, strict=True))
    return dk, dv


def staged_parts(dtype) -> int:
    """The parts each of a fused step's left operands stages: fp16's two at a running scale
    (`algebra.running_parts`), another 16-bit dtype's one."""
    return 2 if jnp.dtype(dtype) == jnp.dtype(jnp.float16) else 1


def _fused_allocations(family: mosaic.Family, config: BackwardTiles, width: int, dtype, stages: int, *,
                       streamed: int = 0, typed: bool = False, anchored: bool = True,
                       buffers: int = 1) -> tuple[list, dict, list]:
    """The fused launch's allocations as `Family.scoped` takes them, in `_fused_launch`'s order: its
    consumers' resident keys and values; a tile's weights and score cotangents staged for the transposed
    products' left operands and dQ's (fp16's two parts each, `_sixteen_step`); dK's and dV's output stage
    (fp32, the head widths equal, `MosaicBackward.legal`), held across the walks where a tile's dQ adds read
    it (`Transport.adds_through_shared`); the two loads' barriers; its query and output-cotangent streams in
    `stages` slots (`Family.streams`), and the first `streamed` of its query tiles' per-row statistics
    (`STATISTICS`, chosen by `fused_staging`) a stream each beside them, which a step reads from shared
    memory (elsewhere from global memory, `_rows`), loaded ahead with its tile as FlashAttention 3's producer
    loads LSE and dPsum (hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp's `smem_lse`, `smem_dpsum`), not from global
    memory once the step starts (ET-TPLPROF-h100: the fused launch stalled on long scoreboards 46.2k against
    FlashAttention 3's 21.8k); and the output stage scoped after the walks where no add reads it, the walks'
    slots its memory (`Family.scoped`'s `after_walks`: two mma.sync consumers' 64 x 64 launch then fits an
    A100, FS-4324-TWO2's 229,424 bytes held whole against its 166,912); where dK and dV leave `typed`
    (`fused_staging`), in `dtype` through a stage of their own after the walks, rounded once there; where the
    consumers sum each tile's dQ before its add (`dq_summed`), the sum's parts and barriers
    (`across.summing`) held across the walks after the load barriers."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    held = family.adds_through_shared
    # Double-buffer held chunked fp32 add stages (`stores.staged_columns`: 2 x 32 cols = 32 KB at d128) when
    # `fused_staging` budgets `buffers == 2`, so `RingFamily.added` overlaps consecutive chunks' TMA reduces
    # via `stores.buffer_waits` without `wait_smem_to_gmem(0)` stalls (row 5272).
    stage_width = stores.staged_columns(width, (jnp.float32,)) if held and width <= 128 else width
    stage_shape = ((consumers, buffers, max(block_q, block_k), stage_width) if buffers > 1
                   else (consumers, max(block_q, block_k), stage_width))
    # The stage at the family's swizzle: at the narrowest one its wgmma accumulator's stores took no select a
    # register pair (8% fewer instructions, 9 registers), but the stores' shared-memory bank conflicts rose
    # (9.5M to 11.5M) and the fused dK/dV ran 23.8% longer on a GH200 (M-NA-32-gh200, row 5272).
    stage = family.staged_memory(stage_shape, jnp.float32)
    kv_swizzles = transposed_swizzles(config, "keys", "values")
    staged_swizzle = swizzling.UNSWIZZLED if family.relay_bytes(block_q, block_k) else None
    scoped = [family.staged_memory((consumers, block_k, width), dtype, kv_swizzles.get("keys")),
              family.staged_memory((consumers, block_k, width), dtype, kv_swizzles.get("values")),
              family.staged_memory((consumers, 2 * staged_parts(dtype), block_q, block_k), dtype,
                                   staged_swizzle),
              *([stage] if held else []),
              family.load_barriers(consumers), family.load_barriers(consumers),
              *(across.summing((block_q, width), consumers).values() if dq_summed(family, consumers) else ())]
    blocks = {"queries": ((block_q, width), dtype), "douts": ((block_q, width), dtype),
              **{name: ((block_q,), STATISTICS[name])
                 for name in streamed_statistics_order(streamed, anchored=anchored)}}
    out = family.staged_memory((consumers, block_k, width), dtype) if typed else stage
    # Every stream a step reads joined under one barrier pair (`rings.joined`, CUTLASS's PipelineTmaAsync):
    # each step's loads awaited once and its slots given back once, where a ring each cost every consumer a
    # wait and an arrival a stream (M-NA-6: 4.4-7x FlashAttention 3's barriers and syncs a CTA, its tensor
    # pipe active 29-33% against 51-68%; row 5272).
    return (scoped, family.streams(blocks, stages, consumers, joined=tuple(blocks)),
            [] if held and not typed else [out])


def dq_and_dkv(walk: _Backward, accumulators: tuple, weights_ref, cotangents_ref,
               q_rhs, k_rhs, do_rhs, *, looped: bool = False) -> tuple[jax.Array, tuple]:
    """Issue dQ, dV, and dK for a bf16/fp32 backward query tile: on overlapping shared-memory issuers
    (wgmma), issue dV and dK inside dQ's product scope and read dQ with their two groups in flight
    (`Issuer.product_with`) so dQ/dV/dK issue back-to-back with no wait bubble and `add_dq` overlaps dV/dK
    (row 5272, 10867)."""
    family, tiles = walk.program.family, walk.program.tiles
    dk_acc, dv_acc = accumulators
    shape = (tiles.block_q, k_rhs.shape[-1])
    issue = family.looped_products if looped else family.products
    if family.products_overlap and not family.operands_in_registers and not walk.interpret:
        def issue_dkv():
            issue(dv_acc, weights_ref.transpose((1, 0)), do_rhs)
            issue(dk_acc, cotangents_ref.transpose((1, 0)), q_rhs)

        _, dq = family.product_with(cotangents_ref, k_rhs, shape, issue_dkv)
    else:
        dq = (None if family.operands_in_registers
              else family.product(cotangents_ref, k_rhs, shape))
        dv_acc = issue(dv_acc, weights_ref.transpose((1, 0)), do_rhs)
        dk_acc = issue(dk_acc, cotangents_ref.transpose((1, 0)), q_rhs)
        if dq is None:
            dq = (family.looped_product(cotangents_ref, k_rhs, shape) if looped
                  else family.product(cotangents_ref, k_rhs, shape))
    return dq, (dk_acc, dv_acc)


def _fused_step(walk: _Backward, resident: tuple, rows_of, staged, coords_of, whole, add_dq, index, slots,
                carry, accumulators, *, masked: bool | None = None):
    """A fused dK/dV consumer's query tile `index`, queries first (`[bq, bk]`, the fast form's one
    launch): its scores and score cotangents from the query and output cotangent slots (`slots`) and its
    resident keys and values (`resident`), under the tile's queries' statistics (`rows_of(slots, first
    query)`, their slots' where streamed) and
    coordinates (`coords_of(first query)`), a tile every query sees whole (`whole(index)`) unmasked, or
    each tile `masked` or not where the walk's segment says (a run of whole tiles, `whole_run`). Its
    weights and score cotangents staged in `staged`, whose transposes are dV's and dK's left operands
    (`algebra.value_weights`, `cotangent_product`'s `aᵀ @ b`) issued into `accumulators` in place, and the
    cotangents times the resident keys the tile's dQ, added into dQ (`add_dq(first query, dQ)`)."""
    k_smem = resident[0]
    family, tiles = walk.program.family, walk.program.tiles
    shape = (tiles.block_q, tiles.block_k)
    dk_acc, dv_acc = accumulators
    q_slot, do_slot = slots["queries"], slots["douts"]
    q_first = index * tiles.block_q
    coords = functools.partial(coords_of, q_first, read=walk.places_read(masked=masked))
    if q_slot.dtype == jnp.float16:
        return _sixteen_step(walk, k_smem, staged, add_dq, slots, carry, accumulators, q_first, _tile_scores(
            walk, (q_slot, do_slot), resident, rows_of(slots, q_first), coords, whole(index), shape,
            masked=masked, whole_operands=True, lean=True))
    cotangents_staged = staged.at[1]
    if masked is not None and _splits_scores(walk):
        _, part = _tile_scores(
            walk, (q_slot, do_slot), resident, rows_of(slots, q_first), coords, whole(index), shape,
            masked=masked,
            on_weights=lambda narrowed: family.staged(narrowed, staged.at[0]),
            after_weights=lambda: family.looped_products(dv_acc, staged.at[0].transpose((1, 0)), do_slot),
            lean=True)
        family.staged(part, cotangents_staged)
        _, dq = family.product_with(
            cotangents_staged, k_smem, (tiles.block_q, k_smem.shape[-1]),
            lambda: family.looped_products(dk_acc, cotangents_staged.transpose((1, 0)), q_slot))
        add_dq(q_first, dq)
        return carry, (dk_acc, dv_acc)
    narrowed, part = _tile_scores(walk, (q_slot, do_slot), resident, rows_of(slots, q_first), coords,
                                  whole(index), shape, masked=masked, lean=True)
    # Stage P and dS under one proxy fence so narrowed and part free before products (`dq_and_dkv`).
    with family.staged_together():
        family.staged(narrowed, staged.at[0])
        family.staged(part, cotangents_staged)
    dq, accumulators = dq_and_dkv(walk, (dk_acc, dv_acc), staged.at[0], cotangents_staged,
                                  q_slot, k_smem, do_slot, looped=True)
    add_dq(q_first, dq)
    return carry, accumulators

def _sixteen_step(walk: _Backward, k_smem, staged, add_dq, slots, carry, accumulators, q_first,
                  operands: tuple[jax.Array, jax.Array]):
    """`_fused_step`'s for fp16 operands, whose fp32 weights and score cotangents (`operands`) pass fp16's
    range: each as two fp16 parts at its walk's running scale (`algebra.running_parts`, the carry's
    exponents, dV's then dK's), dV's and dK's accumulators taken to the new scale before their parts'
    products issue into them, and the tile's dQ, both cotangent parts times the resident keys, taken back by
    its scale (`algebra.running_restored`) before its add."""
    tiles = walk.program.tiles
    q_slot, do_slot = slots["queries"], slots["douts"]
    (dk_acc, dv_acc), exponents, issued = accumulators, [], []
    for (acc, rhs, first), exponent, left in zip(((dv_acc, do_slot, 0), (dk_acc, q_slot, 2)), carry,
                                                 operands, strict=True):
        exponent, shift, parts = algebra.running_parts(left, exponent)
        issued.append(_rescaled_parts(walk, acc, shift, rhs, (staged.at[first], staged.at[first + 1]),
                                      parts))
        exponents.append(exponent)
    dq = _parts_dq(walk, (staged.at[2], staged.at[3]), k_smem, (tiles.block_q, k_smem.shape[-1]),
                   exponents[1])
    add_dq(q_first, dq)
    return tuple(exponents), (issued[1], issued[0])


def _fused_out(family: mosaic.Family, sums: tuple, views: tuple, index, rows: int, stage, *, grouped: bool,
               interpret: bool) -> None:
    """A fused consumer `index`'s dK and dV `sums` out to their `views` through its rows of the output
    `stage`: added into the key heads' sums where query heads share them (`grouped`), else stored in the
    stage's dtype (`stores_typed`); once the walk's dQ adds have read the stage, whose slots they took
    (`_dq_added`)."""
    if not interpret:
        family.store_wait(0)
    if stage.ndim == 4:
        smem = stage.at[index, :, pl.ds(0, rows)] if grouped else stage.at[index, 0, pl.ds(0, rows)]
    else:
        smem = stage.at[index, pl.ds(0, rows)]
    if grouped:
        for value, view in zip(sums, views, strict=True):
            family.added(value, view, interpret, smem=smem)
    else:
        family.stored(tuple(value.astype(stage.dtype) for value in sums), views, smem=smem)


def _statistics(family: mosaic.Family, refs: list, row, head, block_q: int, *, streamed: int,
                anchored: bool = True) -> tuple[dict, Callable]:
    """A program's query tiles' statistics (`STATISTICS`, their `refs` at its batch `row` and `head`):
    the views of the first `streamed` of `_STREAM_ORDER` (omitting dead `chosen` when `not anchored`),
    streamed beside each tile (`streamed_statistics`), and `rows_of(slots, first query)`, a tile's rows in
    `STATISTICS` order, streamed ones from their slots and the rest from global memory (`_rows`)."""
    by_name = dict(zip(STATISTICS, refs, strict=True))
    chosen = streamed_statistics_order(streamed, anchored=anchored)
    views = {name: (by_name[name].at[row, head], lambda j: (j,)) for name in chosen}
    streamed_names = frozenset(chosen)

    def rows_of(slots, first):
        def read(name, ref):
            return (family.rows_of(slots[name]) if name in streamed_names
                    else _rows(family, ref, row, head, first, block_q))
        return tuple(functools.partial(read, name, ref)
                     for name, ref in zip(STATISTICS, refs, strict=True))
    return views, rows_of


@tuning.lowered_once(("walk",))
def _fused_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None,
                  drop: Dropout | None, *, walk: _Backward) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The fast form's one launch of a call whose score hook, if any, takes no parameters
    (`BackwardTiles.fused`), dropping where `drop` draws (under the dropping call's D, the forward's dropped
    split sums', `gradient.split_delta`), on the dQ launch's layout: each consumer's `block_k` keys of a query
    head resident, walked by the query tiles that see them, streamed with their output cotangents through a
    ring each, queries first (`_fused_step`), under the first pass's D and the forward's denominator (each
    query tile's streamed beside it, `STATISTICS`), each tile's dQ added into dQ atomically (`Family.added`);
    where key heads are shared, each query head's dK and dV added into its key head's own (FlashAttention 3's
    grouped epilogue, hopper/epilogue_bwd.hpp's CollectiveEpilogueBwdGQA: Qwen3-8B's step 1.7% the faster
    causal, 1.3% full, ET-GQA1-h100, where XLA summed fp32 copies a query head's), into one ref of both
    (`_fused_called`), else stored (their zeros cost an ungrouped D64 step 1.7%). Where the consumers walk
    paired over element-wise adds (`dq_summed`), each query tile's dQ is the consumers' sum, added once
    (`_dq_summed`). Returns dQ `[B, Qp, H, D]`,
    dK and dV `[B, Kp, KV, D]`: dQ fp32, dK and dV the keys' and values' dtype where it stores them so
    (`stores_typed`), fp32 otherwise."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads = key.shape[1], key.shape[2]
    program, consumers = walk.program, walk.consumers
    family = program.family
    summed = dq_summed(family, consumers)
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_k
    q_steps = queries // block_q
    grouped, group = heads != kv_heads, heads // kv_heads
    leaves, treedef = jax.tree.flatten((spans, drop))
    kv_shape = ((batch, kv_heads, keys_padded, width) if head_first(family, jnp.float32)
                else (batch, keys_padded, kv_heads, width))
    shapes = [kv_shape] * 2
    groups = stacked.groups(shapes)
    def body(*refs, streams, after=None):
        (q_ref, k_ref, v_ref, do_ref, *stat_refs), refs = refs[:9], refs[9:]
        (spans_of, drop_of), refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        if grouped:
            dq_ref, dkv_ref, *refs = refs
            dk_ref, dv_ref = stacked.views([dkv_ref], groups)
        else:
            dq_ref, dk_ref, dv_ref, *refs = refs
        k_smem, v_smem, staged, out_smem, k_loaded, v_loaded, summing = _held_refs(refs, summed=summed)
        row, head = family.program_index("batch"), family.program_index("heads")
        key_head = lax.div(head, group)
        statistics, rows_of = _statistics(family, stat_refs, row, head, block_q, streamed=stats_streamed,
                                          anchored=walk.dropped)
        views = {**_views_of(family, {"queries": q_ref, "douts": do_ref}, row, head, heads=heads),
                 **statistics}
        streams = family.opened(streams, views)
        pair_first = family.program_index("k_tiles") * pair
        streamed = _streamed([program.readers(pair_first + index * block_k, q_steps, spans_of, row)[:2]
                              for index in range(consumers)])

        def consumer(index):
            k_first = pair_first + index * block_k
            k_view = head_slice(family, row, key_head, pl.ds(k_first, block_k), k_ref.dtype,
                                width=k_smem.shape[-1])
            v_view = head_slice(family, row, key_head, pl.ds(k_first, block_k), v_ref.dtype,
                                width=v_smem.shape[-1])
            family.loaded(((k_ref.at[k_view], k_smem.at[index], k_loaded.at[index]),
                           (v_ref.at[v_view], v_smem.at[index], v_loaded.at[index])))
            lo, hi, whole = program.readers(k_first, q_steps, spans_of, row)
            coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first,
                                          shape=(block_q, block_k),
                                          draw=None if drop_of is None else drop_of.draw(row, head))
            adds = _adds_of(walk, dq_ref, out_smem, (row, head), index, summing)
            step = functools.partial(_fused_step, walk, (k_smem.at[index], v_smem.at[index]), rows_of,
                                     staged.at[index], coords_of, whole, adds)
            shapes = ((block_k, width), (block_k, width))
            run = _walked_run(program, consumers, (pair_first, k_first), q_steps, spans_of)
            segments = functools.partial(program.segmented, step, run)

            # The tiles its partner reads around its own (`Family.walk_own`); on a layout whose consumers
            # feed their rings, every tile any of the program's keys' readers reads, a tile none of a
            # consumer's own keys sees masked whole, its weights and its dQ zero (FlashAttention 3's two
            # warpgroups a CTA, each its own keys over one query stream).
            # Paired consumers' segments are their runs intersected (`_walked_run`), the same steps for each.
            dk, dv = walked_sums(walk, streams, views, (lo, hi), streamed, index, segments, shapes,
                                 scaled=staged_parts(query.dtype) > 1, common=summed)
            out_view = head_slice(family, row, key_head, pl.ds(k_first, block_k), dk_ref.dtype, width=width)
            outs = (dk_ref.at[out_view], dv_ref.at[out_view])
            finished = functools.partial(_fused_out, family, (dk, dv), outs, index, block_k, grouped=grouped,
                                         interpret=walk.interpret)
            if after is None:
                finished(out_smem)
            else:
                after(finished)

        def producer():
            family.stream(streams, views, *streamed)

        family.scheduled(consumers, consumer, producer)

    stages = min(program.tiles.stages, q_steps)
    # A score hook's warpgroups' scratch, or fp16's running scale's (`staged_parts`).
    reduced = reducing(family, consumers, walk.reduces, scaled=staged_parts(query.dtype) > 1)
    plan = fused_staging(family, program.tiles, width, query.dtype, stages, reduced=reduced,
                         grouped=grouped, rounded=walk.rounded, anchored=walk.dropped)
    stats_streamed, typed = plan.streamed, plan.typed
    scoped, rings, after = plan.allocations

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings, after_walks=tuple(after))
    launch = family.kernel(
        entry, out_type=() if grouped else _fused_gradients(family, key, value, width, typed=typed),
        grid=(heads, batch, keys_padded // pair),
        grid_names=("heads", "batch", "k_tiles"), consumers=consumers, interpret=walk.interpret,
        approx_math=walk.approx_math, name="attention_dkv_fused", reducing_warpgroups=reduced,
        reduces=fused_reduces(family, reduced), registers=walk.registers(dq=False),
        # (`modelled`): dQ's, dK's, dV's and dP's.
        cost=stated(walk.spec, batch, heads, query, key, value, dout,
                    tile_walk=TileWalk((queries, keys_padded), (block_q, block_k), consumers, keyed=True,
                                       common=family.walks_streamed)),
        model_flops=modelled(walk.spec, batch, heads, walk.spec.queries or queries,
                             2 * width + 2 * value.shape[-1]))
    query, key, value, dout = (head_layout(family, arr) for arr in (query, key, value, dout))
    operands = (query, key, value, dout, peak, total, anchor, delta, key_index, *leaves)
    dq_axes = head_slice(family, batch, heads, queries, jnp.float32)
    dq, dk, dv = _fused_called(launch, operands, (shapes, groups), copies=consumers if walk.interpret else 1,
                               grouped=grouped, dq_shape=(*dq_axes, width))
    return (head_unlayout(family, dq, heads), head_unlayout(family, dk, kv_heads),
            head_unlayout(family, dv, kv_heads))


def _held_refs(refs: list, *, summed: bool) -> tuple:
    """A fused program's allocations (`_fused_allocations`' order): its resident keys and values, its staged
    weights and cotangents, the output stage where the walks' adds read it (else None, the epilogue's own
    scoped after the walks), the two loads' barriers, and the dQ sum's parts and barriers where its
    consumers sum it (`dq_summed`, `across.summing`'s), else None."""
    summing = dict(zip(("parts", "ready", "read"), refs[len(refs) - 3:], strict=True)) if summed else None
    k_smem, v_smem, staged, *out_smem, k_loaded, v_loaded = refs[:len(refs) - 3] if summed else refs
    return k_smem, v_smem, staged, out_smem[0] if out_smem else None, k_loaded, v_loaded, summing


def _adds_of(walk: _Backward, dq_ref, out_smem, place: tuple, index, summing: dict | None) -> Callable:
    """Consumer `index`'s `add_dq(first query, dQ)`: its own tile's add (`_dq_added`), or where its program's
    consumers sum each tile's dQ (`summing`, `dq_summed`) its part of the sum (`_dq_summed`)."""
    if summing is None:
        return functools.partial(_dq_added, walk, dq_ref, out_smem, place, index)
    return functools.partial(_dq_summed, walk, dq_ref, out_smem, place, summing)


def _walked_run(program, consumers: int, firsts: tuple, q_steps: int, spans_of):
    """The run of query tiles a consumer walks unmasked (`whole_run`, its keys from `firsts[1]`): where its
    program's consumers meet at every step to sum its dQ (`dq_summed`), the tiles every consumer's key block
    from `firsts[0]` sees whole (their runs intersected, None where any is), so every consumer walks the same
    segments and meets the other in the same branch of each step (its masked or its whole one): each
    consumer's own run put them in different branches at a step (an A100's two consumers, causal: one
    masked tile before one's run, two before the other's)."""
    family, block_k = program.family, program.tiles.block_k
    pair_first, k_first = firsts
    if not dq_summed(family, consumers):
        return program.whole_run(k_first, q_steps, spans_of)
    runs = [program.whole_run(pair_first + index * block_k, q_steps, spans_of) for index in range(consumers)]
    if any(run is None for run in runs):
        return None
    starts, ends = zip(*runs, strict=True)
    return functools.reduce(jnp.maximum, starts), functools.reduce(jnp.minimum, ends)


def _dq_summed(walk: _Backward, dq_ref, out_smem, place: tuple, summing: dict, q_first, dq) -> None:
    """A query tile's dQ `dq`, a consumer's part, summed across the program's consumers in their order
    (`across.summed_across` over `summing`, its parts and barriers) and the sum added once into dQ at its
    queries from `q_first` by the first consumer (`_dq_added`, its own copy's interpreted), each consumer
    calling it at every streamed step (`dq_summed`)."""
    across.summed_across(dq, summing, lax.axis_index("wg"), walk.consumers,
                         lambda total: _dq_added(walk, dq_ref, out_smem, place, 0, q_first, total))


def _fused_called(launch: Callable, operands: tuple, grads: tuple, *, copies: int, grouped: bool,
                  dq_shape: tuple[int, int, int, int]) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The fused launch called on its `operands` (`_fused_launch`'s): dQ added into zeros in place, and
    grouped dK and dV in one ref of both (`stacked.zeros`, `[2, B, Kp, KV, D]`, `grads` their shapes
    and grouping: where XLA wraps the launch in a dynamic-slice fusion, a step's stacked forward rows sliced
    for this launch's operands, two refs of equal zeros were one buffer, dK and dV each the sum of both, row
    3180, ET-3180-PLAIN-h100): one copy of dQ, or each consumer's own interpreted (`Family.added`,
    `copies`), a consumer's keys its own."""
    dq = jax.new_ref(launches.filled((copies, *dq_shape), jnp.float32, *operands[:4]))
    if not grouped:
        dk, dv = launch(*operands, dq)
        return stacked.added(dq), dk, dv
    shapes, groups = grads
    dkv = stacked.zeros(shapes, groups, *operands[:4])
    launch(*operands, dq, *dkv)
    dk, dv = stacked.read(dkv, groups)
    return stacked.added(dq), dk, dv
