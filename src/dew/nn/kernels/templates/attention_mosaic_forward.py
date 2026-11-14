"""The attention template's Mosaic GPU forward (`attention_mosaic.MosaicForward`'s launches; design §6, one
mainloop per overlap structure): the order of
an instruction family's bundle's steps (`mosaic.Family`: its roles, operand streams, products and layouts)
around the template's algebra (`attention_algebra`), FlashAttention-3's structure on the wgmma family
(`mosaic_sm90.WGMMA`). It holds no instruction, layout or stream barrier of its own: only the overlap
structure, the query block's load and the consumers' schedule.

One program a query head and the family's consumers' query blocks (two warpgroups on wgmma, each a block of
`Tiles.block_q` rows), beside the family's loads streaming the key and value tiles (`Family.stream`,
`Family.walk`), every slot released by every consumer. Two consumers take turns on the tensor cores (a
schedule barrier between them), so one's softmax runs under the other's products. A step issues S = Q·Kᵀ,
releases the key slot, and hands S to the algebra as a tile (`algebra.Keys`) whose values are the value slot
and whose product accumulates the narrowed weights from registers into the running sums in place; while it
runs the consumer waits for its next key tile. A block reads the key tiles its spans reach (a full or causal
call's from its places, a spans call's from its block map, `attention.block_map`) and masks only those it
does not see whole (`algebra.seen`); a consumer keeps releasing the slots its partner alone reads.

The forward over every call form of 16-bit operands under every hook (`Hooks`: the score hook and its
parameters on each tile, the softmax's initial state in the walk's start, the output hook on its output), the
base-2 fold and natural logits, under the default's guards over whole launches (`algebra.walked_launches`: a
program walks its keys once, so the call walks again in the natural form where some row's logits pass the
fold's reach, and its overflowed rows again shifting under headroom). Its legality rule is `refused`."""

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention_algebra as algebra, mosaic, operands, product, rings, swizzling
from dew.nn.kernels.templates.attention import (
    Hooks,
    KeySpans,
    Kind,
    PaddedCall,
    Param,
    Split,
    by_power,
    output_hooked,
    output_split,
    padded_call,
    prescale,
    prologued,
    shared_place,
)
from dew.nn.kernels.templates.attention_mosaic_program import (
    ForwardRegisters,
    Tiles,
    TileWalk,
    _block,
    _columns,
    _coords,
    _Hooked,
    _hooked_of,
    _param,
    _per_span,
    _Program,
    _Spans,
    _spans_of,
    _streamed,
    _views_of,
    approximate,
    bundle_of,
    held_params,
    hook_reduces,
    issued_into,
    issued_pairs,
    modelled,
    param_types,
    query_swizzle,
    reducing,
    refused,
    riding,
    scopes_products,
    score_places,
    score_stream,
    score_streams,
    score_views,
    staged_after,
    stages_after_walks,
    stated,
    transposed_swizzles,
)
from dew.nn.kernels.templates.cost import estimate
from dew.nn.kernels.templates.rings import Ref


def _out_types(batch: int, queries: int, heads: int, value_width: int, out_dtype, *, splits: int, peaks: bool,
               training: bool, finishes: bool, certified: int = 0, flags: bool = False,
               in_place: bool = False) -> tuple:
    """`_launch`'s outputs' types: a peak walk's maxima and anchor keys (`_peak_consumer`); a training walk's
    status, residual
    rows, anchor keys and split sums; else its status and output `[B, S·Qp, H, Dv]` (a split launch's slices
    along its query axis), and its finished rows where it `finishes`. A launch `in_place` into the plain
    launch's outputs (`_Read.into`) writes them in place and declares none of its own: a masked launch's
    programs that walk nothing write nothing, and an output they could leave unwritten (its rows' status,
    which none reads) is uninitialized memory (NSNAN-2)."""
    if in_place:
        return ()
    rows_out = splits * queries
    # Its rows' maxima, which the fold's guard reads; a plain launch that `flags` its rows states their
    # rewalk and their block's beside them (`_Flags`).
    status_type = jax.ShapeDtypeStruct((3 if flags else 1, batch, heads, rows_out), jnp.float32)
    if peaks:
        # Its rows' maxima and anchor keys.
        return (jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32),
                jax.ShapeDtypeStruct((batch, heads, queries), jnp.int32))
    if training:
        return (status_type, jax.ShapeDtypeStruct((5, batch, heads, queries), jnp.float32),
                jax.ShapeDtypeStruct((batch, heads, queries), jnp.int32),
                jax.ShapeDtypeStruct((batch, heads, queries, value_width), jnp.float32))
    out_type = (status_type, jax.ShapeDtypeStruct((batch, rows_out, heads, value_width), out_dtype))
    if finishes:
        out_type += (jax.ShapeDtypeStruct((2, batch, heads, rows_out), jnp.float32),)
    if certified:
        # The certify form's state after each of its `certified` key tiles (`certified_walk`).
        out_type += (jax.ShapeDtypeStruct((2, batch, heads, certified, rows_out), jnp.float32),
                     jax.ShapeDtypeStruct((batch, heads, certified, rows_out, value_width), jnp.float32))
    return out_type


def _stores_through_query(family: mosaic.Family, width: int, value_width: int, dtype, out_dtype) -> bool:
    """Whether an inference launch's output stores through its query's stage: where the two are as wide
    and as narrow, and the query's stage is staged as an output's is (`query_swizzle` None; a register
    family's 8-row tiles 32 bytes wide refuse the store ladder's rungs of fewer rows, FS-4324-DIAG1: "slice
    size (4) ... divisible by the tile size (8)" on an A100)."""
    return (value_width == width and out_dtype in (None, jnp.dtype(dtype)) and query_swizzle(family) is None)


def forward_products(family: mosaic.Family, tiles: Tiles, value_width: int, dtype, *, peaks: bool = False,
                     parts: int = 1, ahead: bool = False, scaled: bool = False, training: bool = False
                     ) -> tuple:
    """A forward program's products as its scope takes its issuer's scratch (`Family.scoped`'s `products`):
    each consumer's logits, a key tile wide, and, but a peak walk's, its weighted values, the values' width,
    whose left operand, the weights a key tile deep in the values' dtype (an fp32 call's bf16 parts), is a
    register value (`Issuer.register_products`); the logits issued a step ahead of their read where `ahead`
    (`logits_ahead`), in the two accumulators of those alone. A `training` walk's split sums take no
    maximum's factor (`algebra.anchored_step`), so they stay where the issuer keeps them across the walk
    (`Issuer.accumulated`'s `sums`: tcgen05's chunks, in tensor memory, no step reading them back), its
    product of the weights added onto them in place; beside them, an accumulator of the values' width a
    product that takes its own and is added into them after it: a `scaled` walk's small weights'
    (`algebra.at_scale`) and an fp16 call's (`product.Scaled`, `operands.Issuer.summed`). An inference
    walk's sums, which each step takes by its maximum's factor, a fresh accumulator a step's product, read
    back onto the sums its registers carry (kept in tensor memory and taken by its factor there, the 4B
    call's inference forward over all 256 value columns priced 156 of 240 registers and spilled 48 bytes
    at 4,096 tokens, et-kept-regs; a column block of the kept sums a half-lane tile is no Mosaic GPU
    layout, et-tmem-b)."""
    logits, ahead_logits = ((), (tiles.block_k,)) if ahead else ((tiles.block_k,), ())
    if peaks:
        return (tiles.consumers, logits, (), ahead_logits)
    lefts = ((tiles.block_k, jnp.bfloat16 if parts > 1 else dtype),)
    # An inference walk's sums kept too where `family`'s issuer takes each step's factor in its own memory
    # (`Issuer.rescales_kept`: tcgen05's rows, a row a lane).
    if not (training or family.rescales_kept):
        return (tiles.consumers, (*logits, value_width), lefts, ahead_logits)
    fresh = (value_width,) if scaled or jnp.dtype(dtype) == jnp.dtype(jnp.float16) else ()
    return (tiles.consumers, (*logits, *fresh), lefts, ahead_logits, (value_width,))


def logits_ahead(family: mosaic.Family, tiles: Tiles, value_width: int, dtype, *, stages: int,
                 peaks: bool = False, parts: int = 1, coded: bool = False, scaled: bool = False,
                 training: bool = False) -> bool:
    """Whether a forward program issues each key tile's logits a step before its softmax reads them, the next
    tile's product running under this one's softmax (FlashAttention 4's S in flight, row 5368: at the 4B
    decision call a tile's wait for its logits, 1.2k to 1.4k cycles, and its softmax, 2.5k, ran in series
    against a 1.2k-cycle tensor floor, nvgpu-lean's model of the B200 forward): where its issuer keeps the
    product in memory of its own (`Issuer.issues_ahead`, tcgen05's chunks), its operands 16-bit and plain
    (no parts, no codes), its rings two steps deep (`mosaic.ahead_stages`), and every consumer's
    accumulators, the logits' second one among them, within the multiprocessor's tensor memory
    (`Issuer.product_columns`, `GpuTarget.tmem_columns`: two consumers' two 64-column logits and 128 value
    columns fill 512)."""
    if not family.issues_ahead or parts > 1 or coded or stages < mosaic.ahead_stages(declared(ahead=True)):
        return False
    columns = family.product_columns(forward_products(family, tiles, value_width, dtype, peaks=peaks,
                                                      ahead=True, scaled=scaled, training=training))
    return columns <= (generation.gpu_target().tmem_columns or 0)


def _forward_allocations(family: mosaic.Family, tiles: Tiles, width: int, value_width: int, dtype,
                         residuals: bool, stages: int, *, peaks: bool = False, out_dtype=None,
                         page: int | None = None, certified: bool = False, codes=None, parts: int = 1,
                         interpret: bool = False, params: tuple = ()) -> tuple[list, dict, list]:
    """The forward program's allocations as `Family.scoped` takes them, in `_launch`'s order, and those it
    scopes after its walks (`after_walks`): its consumers' query stages and their rows' status; each one's
    output stage where the value is not as wide as the query and as narrow, `out_dtype`, the query's by
    default (the output stores through the query's stage where it is), a training forward's its split sums'
    fp32 (`_Residuals`), after the walks where the family scopes them apart (`stages_after_walks`); a
    training forward's or a peak walk's
    anchor keys' stage; the query loads' barriers and
    the consumers' turns' barrier; and its key and value streams in `stages` slots (`Family.streams`), a
    peak walk's keys alone, a paged call's tile of several pages of `page` keys loaded as `paged_reads` reads
    it (`interpret`ed, a gathered tile a copy a row of each of its swizzled tiles' columns, `Ring.parts`:
    the model prices the compiled launch's gather, `MosaicForward.staged_bytes` asking no interpretation); the
    query's stage at its swizzle (`query_swizzle`); a certify form's fp32 stage of a step's sums last
    (`certified_walk`). An fp32 call's `parts` (`MosaicForward.parts`) stage each tile's parts one above the
    other, the query's and each slot's rows `parts` times theirs, a part a copy of its own, its query loads a
    barrier each. The score hook's parameters (`params`, each its `Kind` and dtype) that ride the keys'
    stream take one each (`score_streams`: a tile's every consumer's rows of a key tile, a key vector's
    tile), staged plain, a step's block one copy, read where the keys' pages are not (no table gathers
    them)."""
    consumers, block_q, block_k = tiles.consumers, tiles.block_q, tiles.block_k
    scoped = [family.staged_memory((consumers, parts * block_q, width), dtype, query_swizzle(family)),
              family.staged_memory((consumers, block_q), jnp.float32)]
    # The output's stage (a training walk's split sums', fp32), which no walk reads: after the walks on a
    # family that scopes them apart, in their slots (`stages_after_walks`).
    stage = None
    if residuals:
        # Read out of tensor memory a chunk of their columns at a time where the family keeps its sums there
        # (`scopes_products`), through a stage of one chunk (`mosaic.staged_columns`, CUTLASS's sub-tiled
        # epilogue): the 4B call's training program at width 256 on two consumers stages 231,564 bytes so,
        # of the block's 232,448, where the whole split sums' 131,072 refused it.
        chunk = mosaic.staged_columns(value_width, (jnp.float32,)) if scopes_products(family) else value_width
        stage = family.staged_memory((consumers, block_q, chunk), jnp.float32)
    elif not peaks and not _stores_through_query(family, width, value_width, dtype, out_dtype):
        stage = family.staged_memory((consumers, block_q, value_width), out_dtype or dtype)
    after = [stage] if stage is not None and stages_after_walks(family) else []
    if stage is not None and not after:
        scoped.append(stage)
    if residuals or peaks:
        scoped.append(family.staged_memory((consumers, block_q), jnp.int32))
    if certified:
        scoped.append(family.staged_memory((consumers, block_q, value_width), jnp.float32))
    if codes is not None:
        # Quantized keys' and values' tiles as the products read them, a consumer's own (`Family.converted`).
        scoped.append(family.staged_memory((consumers, block_k, width), dtype,
                                           transposed_swizzles(tiles, "keys").get("keys")))
        scoped.append(family.staged_memory((consumers, block_k, value_width), dtype))
    if relays(family, tiles, peaks=peaks):
        # The weights' way to the values' product where the family's warps split them (`Family.relaying`).
        scoped.append(family.staged_memory((consumers, block_q, block_k), dtype, swizzling.UNSWIZZLED))
    scoped += [family.load_barriers(consumers * parts), family.schedule(consumers)]
    # Its rings stage the keys and values as stored: quantized `codes` where given, an fp32 call's parts
    # one above the other.
    stored = dtype if codes is None else codes
    blocks = {"keys": ((parts * block_k, width), stored)}
    if not peaks:
        blocks["values"] = ((parts * block_k, value_width), stored)
    keyed = tuple(blocks)
    hooked = score_streams(family, params, "keys", consumers * block_q, block_k)
    blocks.update(hooked)
    # The logits' product reads the keys transposed.
    swizzles = {**transposed_swizzles(tiles, "keys"), **dict.fromkeys(hooked, swizzling.UNSWIZZLED)}
    if page is None or page >= tiles.block_k:
        streams = family.streams(blocks, stages, consumers, swizzles=swizzles,
                                 copies=dict.fromkeys(keyed, parts))
    elif not interpret or family.gathers_interpreted:
        streams = family.streams(blocks, stages, consumers, swizzles=swizzles, gathered=keyed)
    else:
        # A copy a row of each swizzled tile's columns (`rings.Ring.parts`), where no lane-level copy runs.
        unit = {name: swizzling.swizzle_columns(stored, swizzles.get(name, swizzling.SWIZZLE_BYTES))
                for name in keyed}
        copies = {name: blocks[name][0][0] * max(1, blocks[name][0][1] // unit[name]) for name in keyed}
        streams = family.streams(blocks, stages, consumers, swizzles=swizzles, copies=copies)
    return scoped, streams, after


def paged_reads(table: jax.Array, page: int, block_k: int) -> tuple[jax.Array, int]:
    """A paged call's page table `[B, pages]` and its keys a page as its launch reads a key tile through them
    (`_paged_views`): where a tile spans several pages, which its family gathers (`gathered.Ring`: the loading
    warpgroup's cp.async copies, one copy a tile whatever the page; `_pages_hold`), a table of one-key pages,
    each slot's row of the pools (`table[b, s // page] * page + s % page`), so any page that divides the tile
    takes it, one key a page (a page-1 pool's own); else the table and page as they are (whole tiles a
    page)."""
    if page >= block_k:
        return table, page
    rows = table[:, :, None] * page + jnp.arange(page, dtype=table.dtype)
    return rows.reshape(table.shape[0], -1), 1


def relays(family: mosaic.Family, tiles: Tiles, *, peaks: bool = False) -> bool:
    """Whether a forward launch's weights `[block_q, block_k]`, the logits' accumulator, reach the values'
    product through shared memory (`Family.relay_bytes`): a decode step's 16-row mma.sync tile, its warps
    along the keys (`attention_mosaic_program.few_rows`); a peak walk takes no values' product."""
    return not peaks and family.relay_bytes(tiles.block_q, tiles.block_k) > 0


def _plain_forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                   masked: bool = False, **walk):
    """`MosaicForward`'s launch without residuals (`_inference_walk`), its launches walked by the fold's guard
    (`algebra.walked_launches`): a `masked` walk's rewalk in place (its launch's programs that walk
    nothing write nothing, so it declares no output of its own), which takes no output hook, whose rows XLA
    forms past the launch from its whole output."""
    if masked and hooks.output_mod is not None:
        raise ValueError("a masked walk rewalks in place, which an output hook formed past its launch does "
                         "not take")
    # A family that takes no masked rewalk in place (`Issuer.rewalks`: tcgen05's rows) rewalks in the guard's
    # branch, its plain launch's maxima the guard's (`algebra.walked_launches`).
    family = bundle_of(walk["tiles"], query.dtype)
    in_place = family.rewalks and (masked or (hooks.output_mod is None and _copy_costs_more(query, value)))
    spec, launch, finished = _inference_walk(query, key, value, spans, hooks, masked=masked, flagged=in_place,
                                             **walk)
    return finished(algebra.walked_launches(spec, launch, in_place=in_place))


def _copy_costs_more(query: jax.Array, value: jax.Array) -> bool:
    """Whether a call's guard rewalks in place (`algebra.walked_launches`' `in_place`, a masked launch every
    call) rather than in a branch, whose common side copies the plain launch's output into its result: where
    that copy, its bytes read and written at the device's bandwidth, costs more than the masked launch and
    its two block-map fusions at the device's launch cost (`GpuTarget.memory_rate`, `launch_seconds`); the
    branch where either fact is unmeasured."""
    target = generation.gpu_target()
    if target.memory_rate is None or target.launch_seconds is None:
        return False
    rows = query.shape[0] * query.shape[1] * query.shape[2]
    copied = 2 * rows * value.shape[-1] * jnp.dtype(query.dtype).itemsize
    return copied / target.memory_rate > 3 * target.launch_seconds


class _Flags(NamedTuple):
    """A masked rewalk's flags (`[B, H, S·Qp]` fp32, nonzero set, a split launch's slice by slice): each row's
    rewalk (`algebra.rewalks`) and its block's of `granule` rows (any of the block's, at each of its rows),
    as a plain launch that flags its rows states them (`_launched`'s `flags`) or its live rows give them in
    XLA (`of`). A program walks where some slice's block among its rows is set, and keeps and guards a row
    set in some slice (`_walked_program`, `_sliced_any`): a row's slices rewalk together."""
    rows: jax.Array
    blocks: jax.Array
    granule: int

    @classmethod
    def of(cls, live: jax.Array, granule: int) -> "_Flags":
        rows = live.astype(jnp.float32)
        blocked = rows.reshape(*rows.shape[:2], -1, granule)
        blocks = jnp.broadcast_to(blocked.max(axis=-1, keepdims=True), blocked.shape)
        return cls(rows, blocks.reshape(rows.shape), granule)

    @classmethod
    def stated(cls, status: jax.Array, granule: int) -> "_Flags":
        return cls(status[1], status[2], granule)


def _inference_walk(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                    scale: float, natural: bool, tiles: Tiles, interpret: bool, rows: bool = False,
                    out_dtype=None, masked: bool = False, splits: int = 1, table: jax.Array | None = None,
                    sources: jax.Array | None = None, subtracted: bool = False,
                    flagged: bool = False, scales: tuple[jax.Array, jax.Array] | None = None,
                    parts: int = 1, scaled: bool = True, counts: bool = True,
                    registered: ForwardRegisters | None = None):
    """The inference walk's launches: its spec and `launch(form, live, into)` as `algebra.walked_launches`
    takes them and `finished`, its outputs from the walk's: `MosaicForward`'s launch without residuals, at
    `tiles`, under the fold's guard over whole launches (`algebra.walked_launches`), a `scaled` walk's rows'
    sums scaled where they could pass fp32's largest (`algebra.SCALED_WALK`: a walk of a cache's, every
    tile, `MosaicForward`); its launches' work stated to Dew's FLOP count where it `counts` (a rare call's
    scaled op's states none, `MosaicForward.scaled`), but for the fold's rewalk's. A launch a block of the
    value columns where its tiles walk a block (`Tiles.columns`): an output hook reads a row's whole output,
    so where there is one the blocks store theirs in fp32 and the hook runs on the rows they make in XLA
    (`template.output_hooked`, the reference's own), rounded once after it.

    With `rows`, the training forward that keeps its output for the backward's D (`takes_output_residuals`):
    the output and the residuals every mainloop's training forward gives, from the plain launch's finished
    rows, no anchor key and the output as the split sums (`output_split`).

    A `flagged` walk (one that rewalks in place, `algebra.walked_launches`' `in_place`) has its plain launch
    state its rows' rewalk flags (`_launch`'s `flags`, `_Flags`), which the masked rewalk reads as they are:
    no array operation between the two launches. An fp32 call's operands enter as their `parts`
    (`MosaicForward.parts`), stacked along the batch axis, each launch reading them so (`_launch`'s
    `parts`). Its launches state their registers by `registered` (`_launch`'s)."""
    family = bundle_of(tiles, query.dtype)
    call, spec, (raw, k, v), (spans_of, hooked), taken, hooks = _prepared(
        query, key, value, spans, hooks, scale=scale, natural=natural, tiles=tiles, family=family,
        table=table, inference=True, subtracted=subtracted, scales=scales, scaled=scaled)
    # A paged call's table, its rows' pages whole blocks of the call's, and its pools' page (`_launch`'s
    # `table`).
    paged = (None if call.table is None or call.page is None
             else paged_reads(call.table, call.page, tiles.block_k))
    def launch(form: algebra.Forward, live: "jax.Array | _Flags | None", into: tuple | None = None
               ) -> "tuple[tuple, jax.Array, None] | tuple[tuple, None, _Flags]":
        q, form = _rewalked(spec, form, raw, taken, hooks)
        # The natural form's rewalk at its own tiles (`Tiles.relaunch`), its spans' block map of them.
        at, at_spans = _form_tiles(spec, call, form, tiles, spans_of)
        # A rewalk's flags: those the plain launch stated of its rows (`flagged`), else its live rows'.
        flags = (None if live is None else live if isinstance(live, _Flags)
                 else _Flags.of(live.reshape(*live.shape[:2], -1), at.block_q))
        # A rewalk into the plain launch's outputs (`algebra.walked_launches`' `in_place`): masked, its rows
        # the live ones, the rest of its outputs left as the plain launch wrote them (`_Read.into`).
        # An in-place rewalk is of live rows (`algebra.walked_launches`).
        in_place = {"into": into, "keeping": flags.rows} if into is not None and flags is not None else {}
        stated = flagged and live is None and into is None
        # The launch on the family arranged for its own tiles (`bundle_of`), as its derivation priced it: a
        # natural rewalk's key tile its own (`Tiles.relaunch`).
        walk = _InferenceWalk(bundle_of(at, query.dtype), spec, call, hooks, hooked, k, v, interpret, rows,
                              out_dtype, masked, splits, paged, sources, parts, counts, registered)
        return _plain_launches(walk, q, form, at, at_spans, flags, in_place, stated=stated, into=into)

    return (spec, launch, functools.partial(_walked_outputs, queries=call.queries, width=value.shape[3],
                                            rows=rows, splits=splits))


class _InferenceWalk(NamedTuple):
    """What an inference walk's launches share (`_inference_walk`'s `launch`): its family and form, the padded
    call and its hooks (`_prepared`), its keys and values, and the walk's options: its output's dtype by name
    (`MosaicForward.out_dtype`), the query's where None."""

    family: mosaic.Family
    spec: algebra.Forward
    call: PaddedCall
    hooks: Hooks
    hooked: _Hooked
    k: jax.Array
    v: jax.Array
    interpret: bool
    rows: bool
    out_dtype: str | None
    masked: bool
    splits: int
    paged: tuple | None
    sources: jax.Array | None
    parts: int = 1
    counts: bool = True
    """Whether its launches state their work to Dew's FLOP count (`_launch_flops`)."""
    registered: ForwardRegisters | None = None
    """The call's facts its launches' registers read (`_launch`'s `registered`)."""


def _plain_launches(walk: _InferenceWalk, q: jax.Array, form: algebra.Forward, at: Tiles, at_spans,
                    flags: "_Flags | None", in_place: dict, *, stated: bool, into: tuple | None
                    ) -> "tuple[tuple, jax.Array, None] | tuple[tuple, None, _Flags]":
    """An inference walk's launch, or a masked rewalk, at tiles `at` in `form` (`_inference_walk`'s
    `launch`): a launch a block of the value columns where its tiles walk a block (`Tiles.columns`), the
    blocks' outputs side by side, their rows' statistics alike; an output hook of a row's whole output in
    XLA past them."""
    family, spec, hooks, hooked, k, v = walk.family, walk.spec, walk.hooks, walk.hooked, walk.k, walk.v
    interpret, rows, out_dtype, splits = walk.interpret, walk.rows, walk.out_dtype, walk.splits
    paged, sources, parts = walk.paged, walk.sources, walk.parts
    walks = None if flags is None else flags
    blocks = _column_blocks(v, hooked, at.columns)
    split = len(blocks) > 1 and hooks.output_mod is not None
    # A scaled launch's keys' magnitudes the call's (`_largest`) but a cache walk's, which its value slots
    # give (a masked or paged walk's, or one over a cache's rows or codes, whose rows its views gather).
    if form.headroom and not walk.masked and paged is None and sources is None and hooked.scales is None:
        blocks = [(first, values, held._replace(magnitudes=_magnitudes(values, held, parts)))
                  for first, values, held in blocks]
    # In place, a block of the columns reads them where they lie and writes its own of the whole
    # output's (`_launch`'s `column_width`).
    launched = [_launch(q, k, v if into is not None and first is not None else values, at_spans,
                        held._replace(output=()) if split else held,
                        spec=form, hooks=_unhooked(hooks) if split else hooks, tiles=at,
                        family=family, training=False, interpret=interpret, parts=parts,
                        counted=walk.counts and form.natural == spec.natural, finishes=rows, columns=first,
                        splits=splits, registered=walk.registered,
                        table=paged, sources=sources,
                        out_dtype=jnp.float32 if split else out_dtype, walks=walks,
                        shared_by=len(blocks),
                        column_width=(values.shape[-1] if into is not None and first is not None
                                      else None), flags=stated and len(blocks) == 1, **in_place)
                for first, values, held in blocks]
    if into is not None:
        assert flags is not None, "an in-place rewalk is of live rows"
        return (), None, flags
    status, _, *finished = launched[0]
    out = _output_whole(jnp.concatenate([block[1] for block in launched], axis=-1), hooks, hooked,
                        out_dtype or q.dtype, split)
    if stated and len(blocks) == 1:
        # Its rows' rewalk flags as its programs stated them (`algebra.walked_launches`' contract).
        return (out, *finished), None, _Flags.stated(status, at.block_q)
    # Its rows' maxima a slice at a time, `[B, H, S, Qp]`, as `algebra.walked_launches` reads a launch's
    # (its last axis a slice's queries, which the guard's own rows and padding are positions along).
    maxima = status[0]
    return (out, *finished), maxima.reshape(*maxima.shape[:2], splits, -1) if splits > 1 else maxima, None


def _walked_outputs(walked: tuple, *, queries: int, width: int, rows: bool, splits: int):
    """The inference walk's outputs from its launches' (`_inference_walk`'s `finished`), unpadded to the
    call's `queries` and value `width`: the output, and with `rows` its finished maxima and denominators, an
    anchor key of none and the output as the split sums. A split launch's are its slices along the query
    axis (`_launch`'s `splits`): `[B, S, Q, H, Dv]`, its rows `[B, H, S, Q]`, each slice's maxima and
    denominators its own, which XLA merges (`attention_decode`)."""
    out, *rows_of = walked
    if splits > 1:
        out = out.reshape(out.shape[0], splits, -1, *out.shape[2:])[:, :, :queries, :, :width]
        maximum, total = (array.reshape(*array.shape[:2], splits, -1)[..., :queries] for array in rows_of[0])
        return out, maximum, total
    out = out[:, :queries, :, :width]
    if not rows:
        return out
    maximum, total = (array[:, :, :queries] for array in rows_of[0])
    return out, maximum, total, jnp.full(maximum.shape, -1, jnp.int32), output_split(out)


def _column_blocks(value: jax.Array, hooked: _Hooked, columns: int | None, *, whole: bool = True
                   ) -> list[tuple[int | None, jax.Array, _Hooked]]:
    """The blocks of `value`'s columns a launch walks `columns` at a time, each its first column, its slice
    and `hooked` with the initial state's sums of its own columns (`_Hooked.initial`, its maximum and
    denominator the rows'): where `whole`, the whole, first None, where a block holds every column
    (`Tiles.columns` None)."""
    if whole and (columns is None or columns >= value.shape[-1]):
        return [(None, value, hooked)]
    width = value.shape[-1] if columns is None else columns

    def held(first: int) -> _Hooked:
        initial = hooked.initial
        return hooked if initial is None else hooked._replace(
            initial=(*initial[:2], initial[2][..., first:first + width]))

    return [(first, value[..., first:first + width], held(first))
            for first in range(0, value.shape[-1], width)]


def _unhooked(hooks: Hooks) -> Hooks:
    """`hooks` without the output hook, which a launch of a block of the value columns leaves to XLA."""
    return dataclasses.replace(hooks, output_mod=None, output_params=())


def _output_whole(out: jax.Array, hooks: Hooks, hooked: _Hooked, dtype, hooked_blocks: bool) -> jax.Array:
    """The blocks' output `out` side by side, the output hook run in XLA over each row's whole output where
    the blocks' launches left it (`hooked_blocks`), in `dtype`."""
    if not hooked_blocks:
        return out
    return jnp.swapaxes(_output_hooked(jnp.swapaxes(out, 1, 2), hooks, hooked), 1, 2).astype(dtype)


def _form_tiles(spec: algebra.Forward, call, form: algebra.Forward, tiles: Tiles, spans_of):
    """The tiles a launch of `form` takes, and its spans' block map: the call's own (`spec`'s form) at
    `tiles`, the natural form's rewalk of a fold's call at `tiles.relaunch`, whose block map counts its own
    key tile (`_spans_of`)."""
    if form.natural == spec.natural:
        return tiles, spans_of
    at = tiles.relaunch
    if (at.block_q, at.block_k, at.consumers) == (tiles.block_q, tiles.block_k, tiles.consumers):
        return at, spans_of
    return at, _spans_of(spec, call, at.block_q, at.block_k)


def _output_hooked(out: jax.Array, hooks: Hooks, hooked: _Hooked) -> jax.Array:
    """The normalized output `out` `[B, H, Qp, Dv]` fp32 under the output hook in XLA, its parameters as the
    launches read them (`hooked`, padded to the call's blocks; `template.output_hooked`)."""
    params = tuple(Param(array, param.kind) for array, param in zip(hooked.output, hooks.output_params,
                                                                     strict=True))
    return output_hooked(out, hooks.output_mod, params)


def _trained_forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                     scale: float, natural: bool, tiles: Tiles, interpret: bool, scaled: bool = False,
                     counts: bool = True, registered: ForwardRegisters | None = None
                     ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    """`MosaicForward`'s training launches at `tiles`, two a form: a walk of the logits alone for each row's
    anchor, its first key at the keys' largest logit (`_peak_consumer`), then the algebra's training step
    at that anchor and maximum (`algebra.anchored_step`) taken in place, the split sums `B` and the rows'
    sums its residuals, with no value row kept or picked; the output is `B` and the anchor's own term over the
    denominator (`algebra.split_output`, `output_of`), its value row gathered in XLA. A `scaled` walk's
    sums are at each row's scale (`algebra.SCALED_WALK`, the op a call some row of which can be scaled
    launches, `MosaicForward.scaled`), the split sums and rows' weights of its column blocks joined at the
    largest of their rows' shifts. The fold's guard composes over whole launches (`algebra.walked_launches`):
    its rewalk is this kernel's natural launch. Its launches state their work to Dew's FLOP count where it
    `counts`, the rewalk's none (`_inference_walk`'s), and their registers by `registered` (`_launch`'s)."""
    family = bundle_of(tiles, query.dtype)
    call, spec, (raw, k, v), (spans_of, hooked), taken, hooks = _prepared(
        query, key, value, spans, hooks, scale=scale, natural=natural, tiles=tiles, family=family,
        scaled=scaled)
    value_width = value.shape[3]

    def launch(form: algebra.Forward, live: jax.Array | None, into=None):
        del live, into
        q, form = _rewalked(spec, form, raw, taken, hooks)
        counted = counts and form.natural == spec.natural
        # The natural form's rewalk at its own tiles (`Tiles.relaunch`), its spans' block map of them.
        at, at_spans = _form_tiles(spec, call, form, tiles, spans_of)
        # The launches on the family arranged for their own tiles (`bundle_of`), as their derivation priced
        # them: a natural rewalk's key tile its own (`Tiles.relaunch`).
        launching = bundle_of(at, query.dtype)
        found = _launch(q, k, v, at_spans, hooked, spec=form, hooks=hooks, tiles=at, family=launching,
                        training=False, interpret=interpret, peaks=True, counted=counted,
                        registered=registered)

        # A training launch a block of the value columns where its tiles walk a block (`Tiles.columns`, row
        # 2526), each block's split sums side by side, their rows alike.
        # A scaled launch's keys' magnitudes the call's (`_largest`).
        launched = [_launch(q, k, values, at_spans,
                            held._replace(magnitudes=_magnitudes(values, held, 1)) if form.headroom else held,
                            spec=form, hooks=hooks, tiles=at, family=launching, training=True,
                            interpret=interpret, anchors=found, counted=counted, columns=first,
                            registered=registered)
                    for first, values, held in _column_blocks(v, hooked, at.columns)]
        status, rows, anchors, _ = launched[0]
        if form.headroom and len(launched) > 1:
            # Each block's rows scale by its own columns' values: its split sums and rows' weights taken to
            # the largest of the blocks' shifts, a power of two on each.
            shift = functools.reduce(jnp.maximum, [block[1][4] for block in launched])
            others = jnp.concatenate([algebra.power2(block[3], (block[1][4] - shift)[..., None],
                                                     reach=2 * 127) for block in launched], axis=-1)
            rows = rows.at[2:4].set(algebra.power2(rows[2:4], (rows[4] - shift)[None], reach=2 * 127))
            rows = rows.at[4].set(shift)
        else:
            others = jnp.concatenate([block[3] for block in launched], axis=-1)
        split = Split(others, rows[2], rows[3], rows[4])
        # Each row's anchor's value row (`[B, H, Qp, Dv]`): its key head's at its key, gathered.
        batch, kv_heads, keys_count = v.shape[0], v.shape[2], v.shape[1]
        kv_of = jnp.arange(q.shape[2]) // (q.shape[2] // kv_heads)
        places = ((jnp.arange(batch)[:, None, None] * kv_heads + kv_of[None, :, None]) * keys_count + anchors)
        value_rows = jnp.take(jnp.swapaxes(v, 1, 2).reshape(-1, v.shape[-1]), places, axis=0)
        # Under dropout the anchor's own weight takes its draw's multiplier, as `algebra.joined` does.
        own = (split.own if hooks.drop is None
               else split.own * algebra.anchor_multipliers(hooks.drop, anchors))
        acc = algebra.split_output(split.others, own, value_rows)
        # The joined sums over the denominator at their rows' scale (`algebra.split_residuals`).
        total = rows[1] if not form.headroom else algebra.power2(rows[1], -split.shift, reach=2 * 127)
        out = algebra.output_of(acc, total, guarded=form.guarded or hooks.initial is not None)
        out = _output_hooked(out, hooks, hooked)
        return (jnp.swapaxes(out, 1, 2).astype(q.dtype), rows[0], rows[1], anchors, split), status[0], None

    out, maximum, total, anchors, split = algebra.walked_launches(spec, launch)
    queries = call.queries
    split = Split(split.others[:, :, :queries, :value_width], split.own[:, :, :queries],
                  split.others_total[:, :, :queries], split.shift[:, :, :queries])
    return (out[:, :queries, :, :value_width], maximum[:, :, :queries], total[:, :, :queries],
            anchors[:, :, :queries], split)


def _whole(array: jax.Array, parts: int) -> jax.Array:
    """An fp32 call's operand `[P·B, ...]` from its bf16 `parts` stacked along the batch axis
    (`MosaicForward.parts`), summed exactly in fp32 (`product.formed`'s parts are its remainders); itself
    where it enters whole."""
    if parts == 1:
        return array
    return jnp.sum(array.astype(jnp.float32).reshape(parts, -1, *array.shape[1:]), axis=0)


def _prepared(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
              scale: float, natural: bool, tiles: Tiles, family: mosaic.Family,
              table: jax.Array | None = None, inference: bool = False, subtracted: bool = False,
              scales: tuple[jax.Array, jax.Array] | None = None, scaled: bool = True):
    """A call as this mainloop takes it at `tiles`: padded to the program's blocks (`padded_call`, `_block`;
    an `inference` walk's queries to its query blocks alone, which every launch of it divides: a decode step's
    four rows padded to 128 took 21 of B128's 795 us at 4096 keys, ET-DECODE7-h100) and its heads to whole
    swizzled rows (`_columns`), with its `algebra.Forward` (a `scaled` walk's rows' sums scaled where its
    values can carry them past fp32's largest, `algebra.SCALED_WALK`; every form's fold under its logits'
    reach, `algebra.walked_launches`) and the arrays its programs read
    beside the operands (`_Spans`, `_Hooked`), and the power of two its plain launch took into its scale (1
    where none, `_rewalked`). A call it takes none of is refused (`refused`, which the op's eligibility reads
    too). A `subtracted` fold's logits are a difference of raw products scaled after it is taken
    (`algebra.Forward.subtracted`): no fold's reach, so no natural rewalk (a decode step's, whose walk the
    memory bounds, row 1955). A quantized call's keys' and values' `scales` (`attention_decode.attend`'s)
    padded to the call's key tiles, its values' headroom its query's dtype's, the dtype it converts them to
    (`_Codes`)."""
    if reason := refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=False, family=family):
        raise NotImplementedError(reason)
    # The query's power of two before its products (`prescale`). A folding call every row of which sees a
    # key ('full', 'causal') takes it into its launch's scale (`Forward.factor`: the products' factor 2^p
    # times the prescaled query's, so every logit bit for bit the prescaled query's where its products stay
    # in fp32's normal range), no pass of XLA's over the queries before it (0.046 ms of Qwen3-8B's 0.66 ms
    # causal forward, ET-GQA1-h100). Its raw products can pass fp32 where the prescaled query's do not: a
    # row of them all past it below zero reads as a -inf maximum, which such a call's rows take as past the
    # fold's reach (`reach_of`), so it walks again natural on the query taken by the power (`_rewalked`).
    power, scale = prescale(query.dtype, query.shape[3], scale)
    folding = algebra.folds(scale, power, natural=natural, score_mod=hooks.score_mod)
    taken = power if folding and spans.form != "spans" else 1.0
    unit = family.column_unit(query.dtype)
    if hooks.query_mod is not None and query.shape[3] % unit:
        # A query prologue over a width its swizzled rows pad (a norm over the zeros past it is another
        # norm): XLA's program before the launch takes it.
        query = prologued(query, hooks.query_mod, hooks.query_params)
        hooks = dataclasses.replace(hooks, query_mod=None, query_params=())
    # A query its consumers take under its prologue enters as the caller's, its power of two taken in the
    # kernel after the prologue (`_prologued`): a norm of the query by the power is no norm's of the query.
    fused = hooks.query_mod is not None
    call = padded_call(query if fused else by_power(query, power / taken), key, value, spans, hooks,
                       _block(tiles), table, by_position=True,
                       query_block=tiles.consumers * tiles.block_q if inference else None)
    spec = algebra.Forward(spans.form, spans.keys, ids=spans.query_ids is not None, scale=scale * taken,
                           score_mod=hooks.score_mod, natural=natural,
                           # A quantized model's codes and values, each at most 127, sum far within fp32.
                           headroom=scaled and hooks.rounds is None and algebra.headroom(
                               query.dtype if scales is not None else value.dtype),
                           initial=hooks.initial is not None,
                           queries=(call.queries if call.query.shape[1] > call.queries or call.per_span > 1
                                    else None),
                           prescale=power / taken, subtracted=subtracted, rounds=hooks.rounds)
    operands = tuple(_columns(array, unit) for array in (call.query, call.key, call.value))
    hooked = _hooked_of(hooks, call, unit)
    if scales is not None:
        # As long as the call's padded keys (a paged call's slots), which every walk's tiles lie within.
        keys = (call.key.shape[1] if call.table is None or call.page is None
                else call.table.shape[1] * call.page)
        keys = max(keys, -(-scales[0].shape[-1] // tiles.block_k) * tiles.block_k)
        hooked = hooked._replace(scales=tuple(
            jnp.pad(array.astype(jnp.float32), ((0, 0), (0, 0), (0, keys - array.shape[-1])))
            for array in scales))
    read = (_spans_of(spec, call, tiles.block_q, tiles.block_k), hooked)
    return call, spec, operands, read, taken, hooks


def _rewalked(spec: algebra.Forward, form: algebra.Forward, query: jax.Array, taken: float, hooks: Hooks
              ) -> tuple[jax.Array, algebra.Forward]:
    """A launch's query operand and form: the fold's natural rewalk (`algebra.unfolded`) of a call whose
    plain launch took its power of two into its scale (`taken`, `_prepared`) the query taken by the power
    (in the kernel, after the prologue, where its consumers take the query's, `_prologued`) and the scale
    left after it, its products finite where the reference's are; every other launch the call's own."""
    if taken == 1.0 or form.natural == spec.natural:
        return query, form
    rewalked = dataclasses.replace(form, scale=form.scale / taken, prescale=taken)
    return (query if hooks.query_mod is not None else by_power(query, taken)), rewalked


class _Read(NamedTuple):
    """What a forward launch's programs read beside its operands (`_launch`), its kernel's refs flattened:
    the spans' block map, the hooks' arrays, a training walk's anchors, a paged call's table, a masked
    launch's walked query blocks, the rows' sources and a column block's index."""
    spans: _Spans | None
    hooked: _Hooked
    anchors: tuple | None
    table: jax.Array | None
    walks: jax.Array | None
    sources: jax.Array | None
    block: jax.Array | None
    into: tuple | None = None
    """A masked launch's outputs held in place (`_launch`'s `into`): the plain launch's, refs it writes its
    rows `kept` into, every other row left as it was."""
    kept: jax.Array | None = None
    """The rows a launch `into` the plain launch's outputs writes (`[B, H, S·Qp]` int32, nonzero at a row it
    keeps): the rows it walks, the plain launch's bits at every other row."""


class _Refs(NamedTuple):
    """A forward program's refs as `_launched` lays them out (`_Layout.unpacked`): its operands, what it
    reads (`_Read`), its outputs, the query's, rows' and output's stages and the rest, the query loads'
    barriers and the consumers' turns' barrier."""
    q: Ref
    k: Ref
    v: Ref
    read: _Read
    written: tuple
    q_smem: Ref
    row_smem: Ref
    out_smem: Ref | None
    stages: tuple
    q_loaded: Ref
    schedule: Ref
    relay: Ref | None = None
    converted: tuple[Ref, Ref] | None = None


class _Layout(NamedTuple):
    """How `_launched` lays its kernel's refs out: its read refs' count and how they make its `_Read` again
    (`unflatten`, its arrays' tree's, `jax.tree.flatten`'s), its outputs' count, the stages' count
    (`_forward_allocations`'), and whether the output stores through the query's stage or one scoped after
    the walks."""
    reads: int
    unflatten: Callable[[Sequence[Ref]], _Read]
    outputs: int
    arrays: int
    reuse: bool
    relays: bool = False
    converts: bool = False
    after: bool = False
    """Whether the output's stage is scoped after the walks (`stages_after_walks`), none of the program's."""

    def unpacked(self, q_ref, k_ref, v_ref, *refs) -> _Refs:
        """A program's refs, its kernel's then its scoped allocations': the weights' relay tile last where
        the launch `relays` them (`relays`), a quantized call's converted keys' and values' tiles before it
        where it `converts` them (`_forward_allocations`' `codes`)."""
        read, refs = refs[:self.reads], refs[self.reads:]
        written, refs = refs[:self.outputs], refs[self.outputs:]
        (q_smem, row_smem, *stages), (q_loaded, schedule) = refs[:self.arrays], refs[self.arrays:]
        relay = stages.pop() if self.relays else None
        converted = (stages[-2], stages[-1]) if self.converts else None
        stages = stages[:-2] if self.converts else stages
        if self.reuse or self.after:
            out_smem = q_smem if self.reuse else None
        else:
            out_smem, stages = stages[0], stages[1:]
        return _Refs(q_ref, k_ref, v_ref, self.unflatten(read), tuple(written), q_smem, row_smem,
                     out_smem, tuple(stages), q_loaded, schedule, relay, converted)

    def read(self, refs: tuple) -> _Read:
        """What a kernel's entry `refs` (its three operands first) read."""
        return self.unflatten(refs[3:3 + self.reads])


class _Block(NamedTuple):
    """A consumer's query block in its launch's outputs: its first query, its output rows `(row, queries,
    head)` and its rows' statistics' places `(row, head, queries)`, a slice's its own along the outputs'
    query axis (`_Program.splits`)."""
    first: jax.Array
    rows: tuple
    state: tuple

    @classmethod
    def of(cls, program: _Program, row, head, first, queries: int, *, fresh: bool = False) -> "_Block":
        at = pl.ds(program.split(fresh=fresh) * queries + first, program.tiles.block_q)
        return cls(first, (row, at, head), (row, head, at))

    @classmethod
    def afresh(cls, program: _Program, row, head, consumer: int, queries: int) -> "_Block":
        """Consumer `consumer`'s block formed where a row output is stored, its query tile's grid coordinate
        read there (`Family.program_index`'s `fresh`): formed before the walk, its first query was held
        across it and spilled (row 5466)."""
        tiles = program.tiles
        pair = tiles.consumers * tiles.block_q
        first = program.query_tile(queries // pair, fresh=True) * pair + consumer * tiles.block_q
        return cls.of(program, row, head, first, queries, fresh=True)

    def status(self, ref) -> tuple | None:
        """Its rows' maxima in a launch's status `ref` (`walked_launches`), and where the launch flags its
        rows their rewalk and their block's beside them (`_Flags`); None where the launch gives none (a launch
        into the plain launch's outputs, `_Read.into`). Formed at the block a store forms (`afresh`): a
        row's store is a contiguous bulk copy addressed by its first element's 64-bit offset, which the
        compiler formed from the block's first query (the walk reads that too) before the walk and held
        across it, 16 bytes spilled of an H100's fp32 forward at width 64 on three consumers (et-zerorow's
        SASS: the offset's STL.64 before the walk, its LDL.64 at the store)."""
        if ref is None:
            return None
        return tuple(ref.at[place, *self.state] for place in range(ref.shape[0]))

    def inference(self, written: tuple, finishes: bool) -> tuple:
        """Its status, output and (where the launch `finishes` them) finished rows' views in an inference
        launch's outputs (`_out_types`)."""
        status_ref, out_ref, *finished = written
        rows = tuple(finished[0].at[place, *self.state] for place in range(2)) if finishes else None
        return self.status(status_ref), out_ref.at[self.rows], rows

    def certified(self, written: tuple) -> tuple:
        """Its views of a certify form's state after each key tile (`_out_types`' last two): the rows'
        maxima's and denominators' `[tiles, block_q]` and the sums' `[tiles, block_q, Dv]`."""
        rows_ref, sums_ref = written[-2:]
        row, head, queries = self.state
        return (tuple(rows_ref.at[place, row, head, :, queries] for place in range(2)),
                sums_ref.at[row, head, :, queries])


def _walked_program(program: _Program, walks, row, head, queries: int, granule: int) -> jax.Array:
    """Whether a masked launch's program walks (`_launch`'s `walks`, `_Flags.blocks`): some slice's block of
    `granule` rows among its own is set, read at one row a block. Else the program does nothing, its rings'
    barriers not even set: the two empty launches of a decode step at 128 rows of 4096 keys took 8.2 and 8.0
    us of its 0.77 ms where each program set its rings up first (ET-DECODE8-h100)."""
    tiles = program.tiles
    pair = tiles.consumers * tiles.block_q
    pair_first = program.query_tile(queries // pair) * pair
    per = walks.shape[-1] // program.splits
    reads = [walks[row, head, split * per + pair_first + offset] != 0
             for split in range(program.splits) for offset in range(0, pair, granule)]
    return functools.reduce(jnp.logical_or, reads)


def _sliced_any(program: _Program, flags, rows: tuple) -> jax.Array:
    """A consumer's rows set in any of their slices' flags (`_Flags.rows`, `[B, H, S·Qp]`, at its batch row,
    head and queries from `first`), in the family's row layout: a row's slices rewalk together."""
    row, head, first = rows
    per = flags.shape[-1] // program.splits
    block_q = program.tiles.block_q
    found = [program.family.rows_of(flags.at[row, head, pl.ds(split * per + first, block_q)]) > 0
             for split in range(program.splits)]
    return functools.reduce(jnp.logical_or, found)


class _Residuals(NamedTuple):
    """A training forward consumer's residual views and the anchor keys' stage: its rows' maxima,
    denominators, anchor's own weight, other keys' weights' sum and their shift (`[bq]` fp32 views, through
    the consumer's row stage), its anchor keys (`[bq]` int32, through `keys_smem`) and its split sums
    (`[bq, Dv]` fp32, through the consumer's output stage, `_forward_allocations`)."""

    rows: tuple
    keys: object
    others: object
    keys_smem: object


CODES = frozenset(map(jnp.dtype, (jnp.int8, jnp.float8_e4m3fn, jnp.float8_e5m2)))
"""The quantized codes a call's keys and values may be stored as (`_codes_of`)."""


def _codes_of(key: jax.Array, scales: tuple | None):
    """A quantized call's keys' and values' storage dtype, the call's with `scales` (read through `_Codes`),
    else None; codes of another dtype refused."""
    if scales is None:
        return None
    if jnp.dtype(key.dtype) not in CODES:
        raise ValueError(f"a quantized call's codes are int8 or an FP8, not {key.dtype}")
    return key.dtype


class _Codes(NamedTuple):
    """A consumer's reads of a quantized call's keys and values (`_Hooked.scales`): each slot's codes
    converted into its own 16-bit tile (`Family.converted`), the values' rows scaled by their keys' scales as
    they are (`row_scale`), and the keys' scales a key tile's columns, which its logits take before their
    scale and hooks (`_scored`), so the fold's reach and the guards read the logits the dequantized keys
    give. The values' tile is one buffer a consumer, written each step: the step's rescale waits on its last
    P·V before the next conversion, so a walk that defers or overlaps P·V takes no codes
    (`MosaicForward._launch_legal`)."""
    keys: Callable
    values: Callable
    key_scales: Callable

    @classmethod
    def of(cls, family: mosaic.Family, tiles: tuple, scales: tuple, row, head, block_k: int,
           interpret: bool) -> "_Codes":
        key_tile, value_tile = tiles
        key_scale, value_scale = scales

        def at(ref, k_first):
            return ref.at[row, head, pl.ds(k_first, block_k)]

        def values(codes, k_first):
            scaled = family.rows_of(at(value_scale, k_first))
            return family.converted(codes, value_tile, interpret, row_scale=scaled)

        return cls(lambda codes: family.converted(codes, key_tile, interpret), values,
                   lambda k_first: family.columns_of(at(key_scale, k_first)))


def _scored(program: _Program, spec: algebra.Forward, q, index, slots, first, spans: _Spans | None,
            hooked: _Hooked, place: tuple, whole, turn, *, hooks: Hooks, drawn: bool, params_held: tuple,
            share: int = 0, held: tuple | None = None, product: Callable[[], jax.Array] | None = None,
            interpret: bool = False, codes: _Codes | None = None, training: bool = False,
            first_at: Callable | None = None) -> tuple:
    """A key tile's first key, coordinates and logits under the score hook (batch row and head `place`, its
    queries from `first`, or from `first_at()` formed at its mask), and their products where the logits are a
    factor of them (`algebra.logits_of`), every walk's (`_consumer`'s, `_peak_consumer`'s), so a peak walk's
    logits are the training walk's to the bit; its places read by the walk's dropout where it draws it
    (`drawn`). A tile every query sees whole takes no mask: one walk, the choice a tile's (a walk of each,
    these rings' waits and arrivals in two loops, faulted on an H100 under jax 0.11.2's lowering). The
    tile's `Q Kᵀ` its own fresh product, or `product()`'s where the walk issues it. A training forward's
    walks' query places (`training`: its peak walk's and its anchored one's), where its hook reads them, are
    in the accumulator's layout (`Family.as_tile`): a hook's term of a slice of them (a row-indexed one's
    `q_pos[:, :1]`) found no layout on mma.sync, whose reductions' inference offers no accumulator layout (row
    5279). The hook's parameters the walk holds (`params_held`, `held_params`) are the tile's as they are; one
    that rides the keys' stream (`score_streams`) is the step's slot's, a tile's from the consumer's rows'
    `share` of it."""
    row, head = place
    family, tiles = program.family, program.tiles
    shape = (tiles.block_q, tiles.block_k)
    if product is not None:
        s = product()
    elif program.ahead:
        # The product it issues ahead, the logits (`declared`).
        s = family.scored(next(product for product in program.products if product.ahead), q, index, slots,
                          shape)
    elif program.parts > 1:
        # An fp32 call's logits its parts' products (`product.terms`), smallest first, into one fresh
        # accumulator in place (`Family.issued`: a term's own accumulator read before the next serializes
        # wgmma's, ptxas C7512), the partner's turn once they are in.
        keys = program.parted(slots["keys"], tiles.block_k)
        s = family.summed(shape, lambda acc: family.issued(acc, q.refs, keys.T.refs, interpret=interpret))
        if turn is not None:
            turn()
    else:
        keys = slots["keys"] if codes is None else codes.keys(slots["keys"])
        s = family.product(q, keys.transpose((1, 0)), shape, turn)
    slots.release("keys")
    k_first = index * tiles.block_k
    if codes is not None:
        # The quantized keys' products their codes', each key's column scaled to its keys' (`_Codes`).
        s = s * algebra.across(codes.key_scales(k_first), s)
    # The tile's places formed a step, not held across the walk beside its sums (`_coords`).
    # Its mask chosen at run time (`whole`): the masked tile's reads.
    read = algebra.places_read(spec, masked=True, hooked=score_places(hooks), dropped=drawn,
                               padded=spec.keys % tiles.block_k != 0)
    # The block's first query formed where the mask reads it (`first_at`): carried across the logits'
    # product, which waits on the step's register-A products to retire (`retires_register_products`), an
    # H100's training forward at 128 x 64 on three consumers spilled its two query places, 16 bytes, and
    # its fp32 forward at width 64 8 (row 5416's retirement; armada 20261111140541-04a31d5a).
    if first_at is not None:
        first = first_at()
    coords = _coords(family, spans, row, first, k_first, shape, read=read,
                     draw=None if hooked.drop is None else hooked.drop.draw(row, head), held=held)
    if training and score_places(hooks)[0]:
        coords = coords._replace(q_pos=family.as_tile(coords.q_pos))
    streamed = riding(family, tuple(param.kind for param in hooks.score_params), "keys")
    laid = zip(hooked.score, hooks.score_params, params_held, strict=True)
    params = tuple(_param(family, ref, param.kind, row, head, first, s, k_first, held=kept,
                          slot=slots[score_stream(index)] if index in streamed else None, at=(share, 0))
                   for index, (ref, param, kept) in enumerate(laid))
    logits, products = algebra.logits_of(spec, s, coords, params, masked=jnp.logical_not(whole(index)))
    return k_first, coords, logits, products


def _initial(family: mosaic.Family, spec: algebra.Forward, hooked: _Hooked, rows: tuple) -> algebra.Partial:
    """The call's initial state at a consumer's `rows` (batch row, head, queries) in the family's layouts
    (`algebra.resumed`)."""
    assert hooked.initial is not None
    m0, total0, acc0 = hooked.initial
    return algebra.resumed(spec, family.rows_of(m0.at[rows]), family.rows_of(total0.at[rows]),
                           family.tile_of(acc0.at[rows]))


def _empty(family: mosaic.Family, shape: tuple[int, int], *, scaled: bool) -> algebra.Partial:
    """`algebra.started`'s empty state of a walk's `[bq, Dv]` sums (`shape`) in the family's layouts, with
    its rows' scale where the walk is `scaled` (`algebra.SCALED_WALK`)."""
    rows = shape[0]
    scale = algebra.Scale(family.zeros((rows,)), family.zeros((rows,))) if scaled else None
    return algebra.Partial(family.zeros(shape), family.full((rows,), algebra.LEAST), family.zeros((rows,)),
                           scale)


def _value_tile(program: _Program, shape: tuple, dtype, turn, slots, converted=None) -> algebra.Operand:
    """A step's value tile of `shape` and `dtype` from its `slots` (`_consumer`'s): its type beside its slot's
    acquisition, which the step defers to the form reading it, the slot taken once the partner's products
    are issued (its turn on the tensor cores is over when this one's come): a quantized call's the tile its
    codes were `converted` into (`_Codes`), an fp32 call's an fp32 tile, its parts (`_Program.parted`)."""
    def slot():
        turn(wait=False)
        value = slots["values"] if converted is None else converted
        turn(arrive=False)
        return program.parted(value, shape[0])

    return algebra.Operand(shape, jnp.float32 if program.parts > 1 else dtype, slot)


def _status_stored(family: mosaic.Family, spec: algebra.Forward, state: algebra.Partial,
                   status_views: tuple | None, row_smem, first) -> None:
    """A consumer's rows' status into `status_views` (`_launch`'s): each row's maximum in the walk's units, a
    row that saw no key -inf (`algebra.reached`'s convention), every other its maximum, `LEAST` included,
    which the fold's guard reads; or, where the launch flags its rows (three views, `_Flags`), beside it each
    one's rewalk and its block's, which the masked rewalk reads as they are (`algebra.rewalks`); nothing where
    the launch gives no status (None, `_Block.status`). `first` is its first row's query."""
    if status_views is None:
        return
    maximum = jnp.where(algebra.no_keys(state.total), -jnp.inf, state.m)
    status = (maximum,)
    if len(status_views) == 3:
        # Its rows' queries where it walks padding (`algebra.padded_queries`), a tile's places reduced along
        # its columns, as `algebra.Coords.query_places` forms them (a row's iota takes no row layout).
        queries = (jnp.max(first + family.iota(state.acc.shape, 0), axis=-1)
                   if algebra.padded_queries(spec) is not None else None)
        live = jnp.where(algebra.rewalks(spec, maximum, queries)[0], 1.0, 0.0)
        status = (maximum, live, jnp.broadcast_to(jnp.max(live), live.shape))
    family.stored(tuple(map(family.as_row, status)), status_views, smem=row_smem)

def _query_of(program: _Program, spec: algebra.Forward, q_view, q_smem, q_loaded, hooks: Hooks,
              hooked: _Hooked, place: tuple, first):
    """A consumer's query block loaded from `q_view` into its stage, taken under the query prologue and then
    by the launch's power of two where the call has one (`_prologued`, `spec`'s `prescale`), and held
    (`Family.held`); or an fp32 call's parts (`MosaicForward.parts`, `q_view` their `rings.Pieces`, whose
    prologue its caller took) one above the other, a copy and a barrier each, read where they lie
    (`_Program.parted`)."""
    family, block_q = program.family, program.tiles.block_q
    if program.parts == 1:
        family.loaded(((q_view, q_smem, q_loaded),))
        if hooks.query_mod is not None:
            _prologued(family, hooks.query_mod, hooks, hooked.query, place, first, q_smem, spec.prescale)
        return family.held(q_smem)
    family.loaded(tuple((view, q_smem.at[pl.ds(part * block_q, block_q)], barrier)
                        for part, (view, barrier) in enumerate(zip(q_view, q_loaded, strict=True))))
    return program.parted(q_smem, block_q)


def _prologued(family: mosaic.Family, mod: Callable, hooks: Hooks, refs: tuple, place: tuple, first, q_smem,
               power: float) -> None:
    """The query block in its stage under the query prologue (`Hooks.query_mod`), once a tile where the
    consumer loads it: read in its family's tile layout (`Family.tile_of`), taken at its rows' indices
    `[rows, 1]` with its parameters, a head's scalar and a width's and a query's as the tile's entries,
    which the hook reads as it reads the reference's `[1, D]` and `[rows, 1]` (`attention.prologued`) where
    it broadcasts them, rounded to its dtype as the reference rounds it, taken by
    the launch's power of two (`attention.by_power`, its caller's `prescale`, exact), written back and
    made visible to the products that read it (`Family.published`). Mosaic GPU lowers a norm's reduction
    over the width and a rotation's halves so (sm90 and sm100, a 64 x 256 bf16 tile, et-prologue-lowers)."""
    row, head = place[:2]
    rows = q_smem.shape[0]
    tile = family.tile_of(q_smem)

    def param(ref, kind: Kind):
        # A width's and a query's parameter as the tile's entries, broadcast in its layout (`algebra.across`,
        # `algebra.expand`): Mosaic GPU's layouts take no singleton axis of a vector read from global memory
        # (an H100's `[1, 128]`: "Tiling rank different than broadcast result rank", et-a6-e).
        if kind is Kind.HEAD:
            return ref[head]
        if kind is Kind.WIDTH:
            return algebra.across(family.columns_of(ref.at[head if ref.shape[0] > 1 else 0]), tile)
        shared_row, shared_head = shared_place(ref.shape, row, head)
        return algebra.expand(family.rows_of(ref.at[shared_row, shared_head, pl.ds(first, rows)]), tile)

    positions = (first + jnp.arange(rows, dtype=jnp.int32))[:, None]
    params = (param(ref, value.kind) for ref, value in zip(refs, hooks.query_params, strict=True))
    taken = mod(tile, positions, *params).astype(q_smem.dtype)
    q_smem[...] = taken if power == 1.0 else taken * jnp.asarray(power, q_smem.dtype)
    family.published()


def _stream_dtype(view: tuple):
    """A stream's elements' dtype from its view (`rings.viewed`'s): its ref's, or its pieces' first's."""
    ref = view[0]
    return (ref[0] if isinstance(ref, rings.Pieces) else ref).dtype


def _dot(family: mosaic.Family, interpret: bool, lhs: jax.Array, rhs) -> jax.Array:
    """A step's fresh product of a register operand `lhs` with a tile `rhs` (`algebra.Keys.dot`): the
    family's, or with an fp32 call's parts their terms' (`_parted_products`)."""
    if isinstance(rhs, operands.Parts):
        return _parted_products(family, None, None, lhs, rhs, interpret, shape=(lhs.shape[0], rhs.shape[1]))
    return family.registers_product(lhs, rhs)

def _consumer(program: _Program, spec: algebra.Forward, consumer, q_view, out_view, stored: Callable,
              q_smem, q_loaded, out_stage: Callable, row_smem, streams, views: dict, turn, streamed: tuple,
              first, spans: _Spans | None, hooked: _Hooked, place: tuple, *, hooks: Hooks, training: bool,
              anchors: tuple | None = None, interpret: bool, certified: tuple | None = None, kept=None,
              relay=None, codes: _Codes | None = None, value_width: int,
              first_at: Callable | None = None):
    """A consumer's query block (batch row, head and key head `place`): its walk of the key tiles it reads
    through the family's streams (`algebra.plain_step` a step) from the hooks' initial state, the score hook
    on each tile, passing those of the program's `streamed` range its partner alone reads; then its output
    under the output hook, `value_width` wide, stored through its stage (`out_stage(store)`: the query's
    where the widths agree, one scoped after the walks where the family scopes them apart), and its rows'
    maxima through `row_smem` to its status (`algebra.walked_launches`' fold guard), each row output into
    views `stored()` forms where it is stored (`_written_views`' at the block formed there). The walk's
    running sums are the family's accumulator, issued into in place, at its rows' scale where they could
    pass fp32's largest (`algebra.SCALED_WALK`). `training`, the walk is the training step's at the
    `anchors` a peak walk found (`algebra.anchored_step`): the accumulator holds the split sums, every
    key's weighted values but the anchor's, which it stores with the rows' sums and status, no output (the
    output's anchor term joins in XLA, `_trained_forward`). With `certified` (its views and an fp32 stage,
    `_Block.certified`), the plain walk stores its state after each key tile (`certified_walk`). With `kept`
    (a launch into the plain launch's outputs, `_Read.kept`), it writes its rows `kept` alone, every other
    row of its views as it was (`_kept_rows`). With `relay` (its own tile, `relays`), its weights reach the
    values' product through it. With `codes` (a quantized call's, `_Codes`), each step converts its slots'
    codes and scales its keys' and values'."""
    row, head, _ = place
    family, tiles = program.family, program.tiles
    if relay is not None:
        # Its weights reach the values' product through its own tile (`relays`).
        family = family.relaying(relay)
    lo, hi, whole = program.reach(first, spans, row)
    q = _query_of(program, spec, q_view, q_smem, q_loaded, hooks, hooked, place, first)
    # The weighted values' running sums live in the family's accumulator (`Family.walk`), from the initial
    # state's where there is one; the walk carries the maxima and denominators, the state's `acc` an empty
    # place the algebra passes through (`algebra.Keys.accumulate`), a constant no loop may carry.
    state_rows = (row, head, pl.ds(first, tiles.block_q))
    start, merged = _started(program, spec, hooked, state_rows, value_width, training=training)
    # Every consumer but the first takes a turn ahead, which the first's closing turn matches: each makes
    # as many arrivals as the others at any count of consumers (three take one fewer than the first and
    # second, and its partners' last turn waited forever: the interpreted hang of row 4840).
    pl.when(consumer > 0)(turn)
    # The queries' intervals and ids, and the score hook's parameters of the query tile, read once, before
    # the walk (`_query_bounds`, `held_params`).
    held = None if spans is None else program.bounds(first, spans, row)
    kinds = tuple(param.kind for param in hooks.score_params)
    params_held = held_params(family, hooked.score, kinds, row, head, first, tiles.block_q)

    def scored(index, slots, product=None):
        return _scored(program, spec, q, index, slots, first, spans, hooked, (row, head), whole, turn,
                       hooks=hooks, drawn=hooked.drop is not None, held=held, params_held=params_held,
                       share=consumer * tiles.block_q, product=product,
                       interpret=interpret, codes=codes, training=training,
                       first_at=(first_at if family.in_flight and family.retires_register_products
                                 else None))

    # The values' tiles as the products read them: a quantized call's in the query's dtype (`_Codes`), an
    # fp32 call's its parts (`_Program.parted`).
    values = functools.partial(_value_tile, program, (tiles.block_k, value_width),
                               q_smem.dtype if codes is not None else _stream_dtype(views["values"]), turn)
    dot = functools.partial(_dot, family, interpret)

    def step(index, slots, carry, accumulators):
        k_first, coords, logits, products = scored(index, slots)
        converted = None if codes is None else codes.values(slots["values"], k_first)
        tile = values(slots, converted)
        # The sums are the accumulator's, which the step takes the tile into (`Keys.accumulate`); the rows'
        # maxima, denominators and a scaled walk's scale the carry (`algebra.exported`).
        state = algebra.imported((accumulators[0], *carry))
        largest = (_largest(program, hooked, (row, place[2]), k_first, slots, converted) if spec.headroom
                   else None)
        accumulate = family.accumulating(mosaic.declared_into(program.products, "sums"), slots,
                                         _accumulated(family, interpret))
        keys = algebra.Keys(coords, logits, tile, dot, accumulate, products, largest)
        state = algebra.plain_step(spec, state, keys)
        if certified is not None:
            _certified_step(family, certified, index, state, row_smem)
        return algebra.exported(state)[1:], (state.acc,)

    def inference(walk):
        # An inference walk through `walk(step, start, shapes, initial=)` (`_walk_of`): its sums the
        # accumulator's.
        rows, (acc,) = walk(step, algebra.exported(start)[1:], ((tiles.block_q, value_width),),
                            initial=(start.acc,))
        return algebra.imported((acc, *rows)), None

    def walked(streams, lo, hi, counted):
        if not training:
            return inference(_walk_of(family, streams, lo, hi, counted, interpret=interpret,
                                      products=program.products))
        assert anchors is not None
        known = algebra.Argmax(family.rows_of(anchors[0].at[state_rows]),
                               family.rows_of(anchors[1].at[state_rows]))
        return _trained(spec, family, start, known, scored, values, (lo, hi), streams, counted,
                        largest=functools.partial(_largest, program, hooked, (row, place[2])),
                        interpret=interpret, placed=program.products)

    # The slots its partner reads around its own, the two turns a step would have taken
    # (`Family.walk_own`).
    state, trained = family.walk_own(streams, views, (lo, hi), streamed, consumer, walked,
                                     between=lambda: (turn(), turn()))
    if merged:
        state = algebra.merge(spec, _initial(family, spec, hooked, state_rows), state)
    keep = None if kept is None else _kept_of(program, kept, rows=(row, head, first))
    if training:
        # The split sums as the walk left them (at its rows' scale, `algebra.split_residuals`).
        def split_sums(stage):
            others = stored().residuals.others
            family.stored((_kept_rows(family, keep, state.acc, others),), (others,), smem=stage)

        out_stage(split_sums)
    pl.when(consumer == 0)(turn)
    _status_stored(family, spec, state, stored().status, row_smem, first)
    _stored_walk(program, spec, state, trained, hooked, (row, head, first), hooks=hooks, training=training,
                 out=(out_view, out_stage), rows=(stored, row_smem), keep=keep)


class _Written(NamedTuple):
    """A consumer's views of its launch's outputs (`_written_views`), None for each it does not write."""
    status: tuple | None
    out: object
    finished: tuple | None
    residuals: _Residuals | None
    certify: tuple | None


def _written_views(family: mosaic.Family, block: _Block, written: tuple, stages: tuple, *, training: bool,
                   finishes: bool, columns_at, certified: bool) -> _Written:
    """A consumer's views of its launch's outputs `written` (`_launched`'s) at `block`, its `stages` its
    scratch: a training launch's status and residuals; an inference launch's status, output (its block of
    the columns, `columns_at`) and finished rows and a certified launch's certificate."""
    if training:
        status_ref, rows_ref, keys_ref, others_ref = written
        residuals = _Residuals(tuple(rows_ref.at[place, *block.state] for place in range(5)),
                               keys_ref.at[block.state], others_ref.at[block.state], stages[0])
        return _Written(block.status(status_ref), None, None, residuals, None)
    status, out_view, finished = block.inference(written, finishes)
    certify = (*block.certified(written), stages[-1]) if certified else None
    if columns_at is not None:
        out_view = out_view.at[:, columns_at]
    return _Written(status, out_view, finished, None, certify)


def _kept_of(program: _Program, kept, *, rows: tuple) -> jax.Array:
    """A consumer's rows a launch into the plain launch's outputs writes (`_Read.kept` at its batch row,
    head and queries from `first`, set in any of the row's slices, `_sliced_any`), in the family's row
    layout."""
    return _sliced_any(program, kept, rows)


def _kept_rows(family: mosaic.Family, keep: jax.Array | None, value: jax.Array, view) -> jax.Array:
    """`value` (a tile, or a vector of rows) at the rows `keep`, `view`'s own elsewhere (a launch into the
    plain launch's outputs, `_Read.into`: every row it does not walk keeps the plain launch's bits), in
    `view`'s dtype; `value` itself without `keep`."""
    if keep is None:
        return value
    tile = value.ndim == 2
    plain = family.tile_of(view) if tile else family.rows_of(view)
    return jnp.where(keep[:, None] if tile else keep, value.astype(plain.dtype), plain)


def _certified_step(family: mosaic.Family, certified: tuple, index, state: algebra.Partial, row_smem
                    ) -> None:
    """The certify form's state after key tile `index` stored to its views (`_Block.certified`): the rows'
    maxima and denominators through `row_smem`, the sums read from the accumulator once its products are
    done (`Issuer.read`) through the certify stage."""
    (maxima, totals), sums, stage = certified
    family.stored((family.as_row(state.m), family.as_row(state.total)),
                  (maxima.at[index], totals.at[index]), smem=row_smem)
    family.stored((family.read(state.acc),), (sums.at[index],), smem=stage)


def certified_walk(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                   scale: float, natural: bool, tiles: Tiles, interpret: bool, subtracted: bool = False,
                   registered: ForwardRegisters | None = None):
    """The forward's certify form (off in production; row 2084's native step witness): its plain launch at
    `tiles` alone, no fold's rewalk, whose walk stores its online-softmax state after each key tile j
    (`algebra.Partial`: the rows' running maxima and denominators in the walk's units, the sums before the
    finish's divide). Returns the launch's output `[B, Q, H, Dv]` (the production launch's to the bit), the
    maxima and denominators `[2, B, H, tiles, Q]` and the sums `[B, H, tiles, Q, Dv]`
    fp32, written in place into refs (`_launch`'s `into`) that hold the empty state where a query block walks
    no key tile (a causal call's past its diagonal, `_Program.reach`): the maximum -inf, the denominator and
    sums zero, so no entry is left unwritten. The logits are in the walk's log2 units (`algebra.exponent`:
    the scale folded, or the raw products scaled after the subtraction where `subtracted`), their
    exponentials `ex2.approx` where the launch approximates (`approximate`). A call of one block of the value
    columns (`Tiles.columns` None), one slice, no output hook. Its launch states its registers by
    `registered` (`_launch`'s)."""
    if tiles.columns is not None or hooks.output_mod is not None:
        raise ValueError("the certify form walks one block of the value columns, no output hook")
    family = bundle_of(tiles, query.dtype)
    call, spec, (raw, k, v), (spans_of, hooked), taken, hooks = _prepared(
        query, key, value, spans, hooks, scale=scale, natural=natural, tiles=tiles, family=family,
        inference=True, subtracted=subtracted)
    q, form = _rewalked(spec, spec, raw, taken, hooks)
    at, at_spans = _form_tiles(spec, call, form, tiles, spans_of)
    batch, padded, heads, _ = q.shape
    value_width, steps = v.shape[-1], k.shape[1] // at.block_k
    held = (jax.new_ref(jnp.zeros((batch, padded, heads, value_width), query.dtype)),
            jax.new_ref(jnp.stack([jnp.full((batch, heads, steps, padded), -jnp.inf, jnp.float32),
                                   jnp.zeros((batch, heads, steps, padded), jnp.float32)])),
            jax.new_ref(jnp.zeros((batch, heads, steps, padded, value_width), jnp.float32)))
    _launch(q, k, v, at_spans, hooked, spec=form, hooks=hooks, tiles=at, family=family, training=False,
            interpret=interpret, certified=True, into=held, registered=registered)
    out, rows, sums = (jax.freeze(ref) for ref in held)
    queries, width = call.queries, value.shape[3]
    return out[:, :queries, :, :width], rows[..., :queries], sums[:, :, :, :queries, :width]


def _stored_walk(program: _Program, spec: algebra.Forward, state: algebra.Partial, trained: tuple | None,
                 hooked: _Hooked, place: tuple, *, hooks: Hooks, training: bool, out: tuple, rows: tuple,
                 keep: jax.Array | None = None) -> None:
    """A consumer's walk's result stored (`_consumer`): its output under the output hook through `out` (its
    view and its stage's `out_stage`), its finished rows through `rows` (the views `stored()` forms where
    each is stored, `_written_views`', its finished rows None where the launch finishes none, and their
    stage); a training walk's (`trained`, its anchor's terms) its residual rows and anchor keys instead, no
    output. `place` its batch row, head and first query; with `keep`, each written
    at those rows alone, every other row of its view as it was (`_kept_rows`)."""
    family = program.family
    row, head, first = place
    (out_view, out_stage), (stored, row_smem) = out, rows
    if not training or trained is None:
        out_params = tuple(_param(family, ref, param.kind, row, head, first, state.acc)
                           for ref, param in zip(hooked.output, hooks.output_params, strict=True))
        if isinstance(state.acc, mosaic.Kept):
            # Sums its issuer keeps past the walk (`Issuer.accumulated`'s `mosaic.Kept`: a plain walk's, its
            # hooks refused and its rows all its own, `MosaicForward._placed_refusal`): the walk's ratio
            # (`finish`'s), each row's factor its store takes in place a block of columns at a time
            # (`Family.stored`).
            maximum, denominator = algebra.finished_rows(spec, state, initial=False)
            kept = state.acc.scaled(algebra.inverse(state.total))
            out_stage(lambda stage: family.stored((kept,), (out_view,), smem=stage))
        else:
            output, maximum, denominator = algebra.finish(spec, state, initial=hooked.initial is not None,
                                                          output_mod=hooks.output_mod,
                                                          output_params=out_params)
            out_stage(lambda stage: family.stored((_kept_rows(family, keep, output, out_view),), (out_view,),
                                                  smem=stage))
        finished = stored().finished
        if finished is not None:
            # Its rows as the backward reads them beside the output (`Output`): the maximum and denominator
            # `finish` leaves.
            family.stored(tuple(_kept_rows(family, keep, family.as_row(part), view)
                                for part, view in zip((maximum, denominator), finished, strict=True)),
                          finished, smem=row_smem)
        return
    maximum, denominator = algebra.finished_rows(spec, state, initial=hooked.initial is not None)
    own, others_total, key = trained
    # The split sums' rows' shift, their scale's exponent, none where the walk holds no scale
    # (`algebra.split_residuals`).
    shift = family.zeros((program.tiles.block_q,)) if state.scale is None else state.scale.exponent
    parts = (maximum, denominator, own, others_total, shift)
    residual_rows = stored().residuals.rows
    family.stored(tuple(_kept_rows(family, keep, family.as_row(part), view)
                        for part, view in zip(parts, residual_rows, strict=True)), residual_rows,
                  smem=row_smem)
    residuals = stored().residuals
    family.stored((_kept_rows(family, keep, family.as_row(key), residuals.keys),), (residuals.keys,),
                  smem=residuals.keys_smem)


def _walk_of(family: mosaic.Family, streams, lo, hi, counted, *, interpret: bool,
             products: Sequence[mosaic.Product] = ()) -> Callable:
    """A consumer's walk of its key tiles `lo` to `hi` over its program's `streams` from the program's
    `counted`-th step (`Family.walk`, which places its declared `products`), as `walk(step, start, shapes,
    initial=)` takes it."""
    def walk(step, start, shapes, initial=None):
        return family.walk(streams, ((step, lo, hi),), start, shapes, interpret, first=counted,
                           initial=initial, products=products)

    return walk


def _started(program: _Program, spec: algebra.Forward, hooked: _Hooked, rows: tuple, value_width: int, *,
             training: bool) -> tuple[algebra.Partial, bool]:
    """A consumer's walk's start at its `rows` (batch row, head, queries): its state and whether the initial
    state merges past the walk instead (`algebra.merge`). A plain inference walk starts empty and merges it:
    read before the walk, the state's rows held live across it beside its products and the walk spilled 296
    bytes at D128's 64 x 128 (the held-term census), where an empty start's are constants."""
    family, block_q = program.family, program.tiles.block_q
    merged = hooked.initial is not None and not training
    start = (_initial(family, spec, hooked, rows) if hooked.initial is not None and not merged
             else _empty(family, (block_q, value_width), scaled=spec.headroom))
    return start, merged


def _parted_products(family: mosaic.Family, acc, scale, weights: jax.Array, values: operands.Parts,
                     interpret: bool, *, shape: tuple[int, int] | None = None):
    """`acc` with an fp32 call's P·V issued into it in place, its factor `scale` first where given, or a
    fresh accumulator of `shape` where `acc` is None: the fp32 `weights`' bf16 parts (`product.bf16_parts`'),
    the largest first, each product with the values' parts (`values`) the pair's orders under their count
    (`product.terms`'), all into the one accumulator (`issued_pairs`). Each part formed as it is taken, from
    the remainder past the last (`product.bf16_parts`; a weight is at most 1, no part of it saturating), so a
    consumer holds that remainder and one part beside the sums."""
    count = len(values.refs)
    taken = set(product.terms(count, count))

    def pairs():
        # Each weight part's pairs as it is formed, the largest part first, as `terms` takes them.
        for i, part in enumerate(product.bf16_parts(weights, count, saturating=0)):
            yield from ((part, values.refs[j]) for j in range(count) if (i, j) in taken)

    if acc is None:
        return issued_pairs(family, None, pairs(), interpret, shape=shape)
    if scale is not None:
        acc = family.rescaled_products(acc, scale, (), None, interpret)
    return issued_pairs(family, acc, pairs(), interpret)


def _accumulated(family: mosaic.Family, interpret: bool) -> Callable:
    """`algebra.Keys.accumulate` on the family's accumulator: the tile's products issued into it in place,
    after its factor where the step gives one (`Family.rescaled_products`). A scaled tile's small weights'
    parts (`algebra.small_parts`), which trail the weights' own, take an accumulator of their own each,
    restored and added in after them (`issued_into`; an fp32 call's of its parts, `_parted_products`), so
    the weights' own products take the unscaled walk's issue."""
    def accumulate(acc, scale, parts, value):
        count = sum(1 for part in parts if _small(part))
        if count:
            acc = accumulate(acc, scale, parts[:-count], value)
            if not isinstance(value, operands.Parts):
                return issued_into(family, acc, parts[-count:], value, interpret)
            for part in parts[-count:]:
                fresh = _parted_products(family, None, None, part.value, value, interpret,
                                         shape=(part.value.shape[0], value.shape[-1]))
                acc = family.added_into(acc, algebra.restored(fresh, part.exponent), interpret)
            return acc
        if isinstance(value, operands.Parts):
            (weights,) = parts
            return _parted_products(family, acc, scale, weights, value, interpret)
        if scale is None:
            return issued_into(family, acc, parts, value, interpret)
        return family.rescaled_products(acc, scale, parts, value, interpret)

    return accumulate


def _small(part) -> bool:
    """Whether a product's part is a scaled walk's small weights' (`algebra.small_parts`): a scaled part whose
    power of two is a static number, where an fp16 part's is traced (`product.Scaled`)."""
    return isinstance(part, product.Scaled) and not isinstance(part.exponent, jax.Array)


def _largest(program: _Program, hooked: _Hooked, place: tuple, k_first, slots, converted=None
             ) -> Callable[[], jax.Array]:
    """A step's keys' largest value magnitudes (`algebra.Keys.largest`), a key tile's columns' layout: from
    the call's (`_Hooked.magnitudes`, at the batch row and key head `place` and the tile's first key
    `k_first`) where it holds them (a scaled op's launch, whose value tiles' transposed load the
    accumulator's layout need not take: a column block 32 wide is 32 of wgmma's 64 rows), else from its
    value slot (`Family.key_magnitudes`, whose transposed load a cache walk's 16-row tiles take at any value
    width): a quantized call's from the tile its codes were converted into (`_Codes`), an fp32 call's from
    its first part (`_Program.parted`), within its rounding of each value, which the scaled walk's binade
    below fp32's largest takes (`algebra.HEADROOM`)."""
    family = program.family
    if hooked.magnitudes is not None:
        row, kv_head = place
        held = hooked.magnitudes
        return lambda: family.columns_of(held.at[row, kv_head, pl.ds(k_first, program.tiles.block_k)])
    tile = slots["values"] if converted is None else converted
    if program.parts > 1:
        tile = tile.at[pl.ds(0, program.tiles.block_k)]
    return functools.partial(family.key_magnitudes, tile)


def _magnitudes(value: jax.Array, hooked: _Hooked, parts: int) -> jax.Array:
    """A launch's keys' value magnitudes (`_Hooked.magnitudes`): each key's largest over `value`'s
    columns `[B, K, Hk, Dv]` (an fp32 call's whole, `_whole`), `[B, Hk, K]` fp32."""
    return jnp.swapaxes(jnp.max(jnp.abs(_whole(value, parts).astype(jnp.float32)), axis=-1), 1, 2)


def _trained(spec: algebra.Forward, family: mosaic.Family, start: algebra.Partial, known: algebra.Argmax,
             scored, values, reach, streams, first, *, largest, interpret: bool,
             placed: Sequence[mosaic.Product] = ()) -> tuple[algebra.Partial, tuple]:
    """A training forward consumer's walk of its key tiles `reach` (`[lo, hi)`, the streams' steps from
    `first`) from `start` at the anchors and keys' largest logits `known` (`algebra.anchored`): each tile
    `algebra.anchored_step`, its split sums the family's accumulator, which takes each tile's product in place
    and no factor but a scaled tile's change of scale. `scored(index, slots)` gives a tile's place,
    coordinates, logits and products, `values(slots)` its value slot as an operand (`algebra.Operand`), and
    on the scaled walk `largest(k_first, slots)` its keys' value magnitudes (`_largest`), its declared
    products `placed` its family's to place (`_Program.products`). Returns the state
    with the split sums as its sums, and what the walk leaves beside them: the anchor's own weight, the
    other keys' weights' sum and the anchor keys, every sum at its rows' scale where the walk is scaled
    (`algebra.SCALED_WALK`)."""
    state, arg, own, others_total = algebra.anchored(spec, start, known)
    held = () if state.scale is None else tuple(state.scale)

    def dot(lhs, rhs):
        return family.registers_product(lhs, rhs)

    def step(index, slots, carry, accumulators):
        total, own, others_total, *scale = carry
        k_first, coords, logits, products = scored(index, slots)
        accumulate = family.accumulating(mosaic.declared_into(placed, "sums"), slots,
                                         _accumulated(family, interpret))
        keys = algebra.Keys(coords, logits, values(slots), dot, accumulate, products,
                            largest(k_first, slots) if spec.headroom else None)
        # The sums are the accumulator's, which the step takes the tile into (`Keys.accumulate`).
        walked = algebra.Partial(accumulators[0], state.m, total, algebra.Scale(*scale) if scale else None)
        walked, _, own, others_total = algebra.anchored_step(spec, (walked, arg, own, others_total), keys)
        kept = () if walked.scale is None else tuple(walked.scale)
        return (walked.total, own, others_total, *kept), (walked.acc,)

    (total, own, others_total, *scale), (acc,) = family.walk(
        streams, ((step, *reach),), (state.total, own, others_total, *held), (state.acc.shape,), interpret,
        first=first, initial=(state.acc,), products=placed)
    return (algebra.Partial(acc, state.m, total, algebra.Scale(*scale) if scale else None),
            (own, others_total, arg.key))


def _peak_consumer(program: _Program, spec: algebra.Forward, consumer, q_view, stored: Callable, q_smem,
                   q_loaded, row_smem, keys_smem, streams, views: dict, turn, streamed: tuple, first,
                   spans: _Spans | None, hooked: _Hooked, place: tuple, *, hooks: Hooks,
                   interpret: bool) -> None:
    """A peak walk's query block (batch row, head and key head `place`): each row's keys' largest logit, in
    the walk's units, and the first key that holds it (`algebra.argmax`, the training walk's anchor,
    `algebra.anchored`), over the key tiles it reads through the family's key stream, the logits under the
    score hook as the training walk forms them (`_scored`), stored through the row stages to the views
    `stored()` forms where each is stored.
    Its turns on the tensor cores and the slots its partner alone reads are the training walk's own
    (`_consumer`)."""
    family, tiles = program.family, program.tiles
    lo, hi, whole = program.reach(first, spans, place[0])
    q = _query_of(program, spec, q_view, q_smem, q_loaded, hooks, hooked, place, first)
    # Every consumer but the first a turn ahead, as `_consumer`'s.
    pl.when(consumer > 0)(turn)
    held = None if spans is None else program.bounds(first, spans, place[0])
    row, head = place[:2]
    kinds = tuple(param.kind for param in hooks.score_params)
    params_held = held_params(family, hooked.score, kinds, row, head, first, tiles.block_q)

    def step(index, slots, arg, accumulators):
        # The peak walk's logits draw no dropout.
        _, coords, logits, _ = _scored(program, spec, q, index, slots, first, spans, hooked, place[:2], whole,
                                       turn, hooks=hooks, drawn=False, held=held, params_held=params_held,
                                       share=consumer * tiles.block_q, interpret=interpret, training=True)
        # The partner's turn, as a training step's value product would take it.
        turn(wait=False)
        turn(arrive=False)
        return algebra.argmax(arg, logits, coords.k_pos), accumulators

    start = algebra.Argmax(family.full((tiles.block_q,), -jnp.inf),
                           family.full((tiles.block_q,), 0, jnp.int32))
    walked, _ = family.walk_own(
        streams, views, (lo, hi), streamed, consumer,
        lambda streams, lo, hi, counted: family.walk(streams, ((step, lo, hi),), start, (), interpret,
                                                     first=counted, products=program.products),
        between=lambda: (turn(), turn()))
    pl.when(consumer == 0)(turn)
    arg = walked
    family.stored((family.as_row(arg.peak),), (stored()[0],), smem=row_smem)
    family.stored((family.as_row(arg.key),), (stored()[1],), smem=keys_smem)


def _key_views(named: dict, row, head, sources, table, *, page: int | None, block_k: int,
               column_block: jax.Array | None) -> dict:
    """The views a launch's rings read its keys and values from (`Family.opened`), at batch row `row` and key
    head `head`: a call's rows of keys at the row its `sources` name (`_views_of`; its own without), a paged
    pool's where its `table` maps them (`_paged_views`); the values' at their block of the columns where the
    launch walks one (`column_block`, the block's index among the columns' blocks), a step's block index
    the block's."""
    if page is None:
        views = _views_of(named, row if sources is None else sources[row], head)
    else:
        views = _paged_views(named, table, row, head, page, block_k)
    if column_block is None or "values" not in views:
        return views
    ref, tile = views["values"]
    block = column_block

    def at_columns(j):
        # A paged tile's pages each at the column block (`rings.Rows`).
        indices = tile(j)
        if isinstance(indices, rings.Rows):
            return dataclasses.replace(indices, rest=(block,))
        return indices[0], block
    return {**views, "values": (ref, at_columns)}


def _paged_views(refs: dict, table, row, head, page: int, block_k: int) -> dict:
    """A paged call's streams' views (`_views_of`): each pool `[KV, P·page, D]`'s `[P·page, D]` at key head
    `head`, step `j` its key tile `j` where row `row`'s table maps it: of a page of several tiles, the
    tile's page `table[row, j // per]` and its place in it `j % per` (`per` tiles a page); of a tile of
    several pages (`count` a tile), each page's rows from its own (`rings.Rows`, a run a page)."""
    per, count = max(1, page // block_k), max(1, block_k // page)

    def tile(j):
        if count == 1:
            return (table[row, lax.div(j, per)] * per + lax.rem(j, per), 0)
        return rings.Rows(table.at[row], j * count, (0,), page)

    return {name: (ref.at[head], tile) for name, ref in refs.items()}


def _launch(query, key, value, spans: _Spans | None, hooked: _Hooked, *,
            spec: algebra.Forward, hooks: Hooks, tiles: Tiles, family: mosaic.Family, training: bool,
            interpret: bool, columns: int | None = None, peaks: bool = False,
            anchors: tuple[jax.Array, jax.Array] | None = None, out_dtype=None, counted: bool = False,
            finishes: bool = False, splits: int = 1,
            table: tuple[jax.Array, int] | None = None, column_width: int | None = None,
            walks: "_Flags | None" = None, sources: jax.Array | None = None, shared_by: int = 1,
            certified: bool = False, into: tuple | None = None,
            keeping: jax.Array | None = None, flags: bool = False, parts: int = 1,
            registered: ForwardRegisters | None = None):
    """The forward's launch in `spec`'s form: its rows' status `[1, B, H, Qp]` fp32, their maxima in the
    walk's units (`algebra.walked_launches`' fold guard), and its output `[B, Qp, H, Dv]` in `out_dtype`
    (the query's by default). In `training`, at the `anchors` a peak walk found, its status and residuals
    (`_Residuals`) and no output: the rows `[5, B, H, Qp]` fp32 (maxima, denominators, the anchor's own
    weight, the other keys' weights' sum and their shift, `algebra.split_residuals`), the anchor keys
    `[B, H, Qp]` int32 and the split sums `[B, H, Qp, Dv]` fp32, in `Split`'s layout, each block's rows one
    contiguous slab. A peak walk (`peaks`) reads the keys alone and gives each row's keys' largest logit
    `[B, H, Qp]` fp32 in the walk's units and its anchor key `[B, H, Qp]` int32 (`_peak_consumer`). With
    `finishes`, the output's launch also gives its rows' finished maxima
    and denominators `[2, B, H, Qp]` fp32 (`Output`, `algebra.finish`).
    Its kernel is built once for every call of its static facts (`_launched`, `tuning.lowered_once`): a
    model's layers alike lower one. With `splits` (an inference launch's, no initial state), each query
    tile's keys are cut into that many slices (`_Program.splits`), each a program of its own, and every
    per-row output above is the slices' own, slice by slice along its query axis (`S·Qp` rows). With
    `table` (a paged call's page table and its pools' keys a page, `forward`'s), `key` and `value` are its
    pools flattened, `[KV, P·size, D]`, and a step's key tile is read where the table maps it; with
    `sources` (`forward`'s) batch row `b`'s keys and values are their row `sources[b]`. With
    `column_width`, it walks `value`'s columns `columns` to `columns + column_width` where they lie, its
    output that block's, the model's `Q K^T` its share among the `shared_by` blocks of an inference call's
    columns (each block's launch recomputes it whole). With `walks` (a masked launch's,
    `algebra.walked_launches`': `[B, Qp // block_q]` int32, nonzero at a query block holding a row it
    walks), a program none of whose query blocks it names loads and writes nothing, its outputs there
    unread. With `certified`, the certify
    form's (`certified_walk`): its plain walk's state after each key tile beside its outputs. With `into`
    (refs holding its outputs but its status, `_Read.into`), it writes its rows `keeping` (every row where
    None) into them in place and gives nothing: a masked rewalk's rows, every other row the plain launch's
    bits, no copy of the outputs and no select over them (`algebra.walked_launches`' `in_place`). Its kernel
    states its consumer's registers at its own tiles as its derivation counts them (`registered`'s,
    `ForwardRegisters.at`; none where None), its name and its register model one launch's facts.
    """
    if splits > 1 and (training or peaks or hooked.initial is not None):
        raise ValueError("a split launch walks an inference call, its initial state merged in XLA")
    # Each launch named apart, so its compile reports and profile rows say which it is; a launch over a block
    # of the value columns by the block's width, every block's one kernel (the name is a static of its
    # lowering, `_launched`: named by its first column, a width 256 compiled four, train 505's review).
    kind = "peaks" if peaks else "training" if training else "forward"
    block_width = column_width or value.shape[-1]
    # A scaled walk's launch (`algebra.SCALED_WALK`) named apart: a call compiles it beside its unscaled form
    # where some row can be scaled (`MosaicForward.scaled`).
    name = (f"attention_{kind}{NATURAL_NAME if spec.natural else ''}{'_scaled' if spec.headroom else ''}"
            f"{'' if columns is None else f'_columns{block_width}'}"
            f"{'_certified' if certified else ''}")
    # The hooks' functions and their arrays' shapes and dtypes static, their arrays the programs' operands.
    # A block of the values' columns its index a scalar the programs read, so every block's launch is one
    # kernel (`_key_views`): a static index compiled a kernel a block, four at width 256 (the forward's
    # resource gate's cases 50 to 150 s on a CPU).
    block = None if column_width is None else jnp.full((1,), (columns or 0) // column_width, jnp.int32)
    target = generation.gpu_target()
    return _launched(query, key, value,
                     _Read(spans, hooked, anchors, None if table is None else table[0],
                           None if walks is None else walks.blocks, sources, block, into, kept=keeping),
                     hooks=jax.tree.map(lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype), hooks),
                     spec=spec, tiles=tiles, family=family, training=training, interpret=interpret,
                     peaks=peaks, out_dtype=None if out_dtype is None else jnp.dtype(out_dtype),
                     counted=counted, finishes=finishes, target=target,
                     name=name if splits == 1 else f"{name}_split", splits=splits,
                     page=None if table is None else table[1],
                     column_width=column_width, shared_by=shared_by,
                     certified=certified, flags=flags,
                     walk_rows=None if walks is None else walks.granule, parts=parts,
                     registers=None if registered is None else registered.at(family, tiles, peaks=peaks,
                                                                              columns=column_width))


def _extents(query, key, value, read: _Read, *, page: int | None, column_width: int | None, parts: int
             ) -> tuple[int, ...]:
    """A forward launch's extents (`_launched`'s): its batch rows (an fp32 call's parts stacked along them,
    `MosaicForward.parts`), queries, heads and head width; its values' columns (a block's where it walks
    one in place, `_launch`'s `column_width`); its keys and key heads (a paged call's its table's slots of
    its pools `[KV, P·size, D]`)."""
    batch, queries, heads, width = query.shape
    value_width = value.shape[-1] if column_width is None else column_width
    keys_count, kv_heads = ((read.table.shape[1] * page, key.shape[0]) if read.table is not None and page
                            else (key.shape[1], key.shape[2]))
    return batch // parts, queries, heads, width, value_width, keys_count, kv_heads

def _query_barriers(barriers, index, parts: int):
    """A consumer `index`'s query load's barrier, or an fp32 call's `parts`' a barrier each
    (`MosaicForward.parts`)."""
    if parts == 1:
        return barriers.at[index]
    return tuple(barriers.at[index * parts + part] for part in range(parts))


def _query_view(q, row, rows, head, *, batch: int, parts: int):
    """A consumer's query block's view (its batch `row`, `rows` and `head`), or an fp32 call's `parts`' views
    from each part's batch rows (`rings.Pieces`, `MosaicForward.parts`)."""
    if parts == 1:
        return q.at[row, rows, head]
    return rings.Pieces(q.at[part * batch + row, rows, head] for part in range(parts))

def _parted_views(views: dict, named: dict, row, kv_head, *, batch: int, parts: int) -> dict:
    """An fp32 call's stream views (`MosaicForward.parts`): each tile its parts' rows, a copy each from its
    part's batch rows (`rings.viewed`'s pieces of refs), at the views' own block indices."""
    return {name: (rings.Pieces(ref.at[part * batch + row, :, kv_head] for part in range(parts)), index)
            for (name, (_, index)), ref in zip(views.items(), named.values(), strict=True)}

def _entry(program: _Program, layout: "_Layout", body: Callable, allocations: tuple, *refs,
           queries: int, walk_rows: int | None) -> None:
    """A forward launch's kernel entry (`_launched`'s): its `body` in its scoped allocations and rings, where
    a masked launch walks it alone (`_walked_program`: else its program loads and writes nothing; its
    `walk_rows` the walks' granule, which a masked launch states)."""
    scoped, rings, after, products = allocations

    def scoped_body():
        program.family.scoped(functools.partial(body, *refs), *scoped, streams=rings, products=products,
                              after_walks=tuple(after))

    walks = layout.read(refs).walks
    if walks is None:
        scoped_body()
    else:
        assert walk_rows is not None, "a masked launch states its walks' granule"
        pl.when(_walked_program(program, walks, program.family.program_index("batch"),
                                program.family.program_index("heads"), queries, walk_rows))(
            scoped_body)


def _program_of(family: mosaic.Family, tiles: Tiles, spec: algebra.Forward, read: _Read, key, steps: int,
                value_width: int, dtype, *, splits: int, peaks: bool, parts: int, training: bool) -> _Program:
    """A launch's program (`_Program`) of `steps` key tiles: no more ring stages than them (a slot past them
    would load past the keys), and its keys read a step ahead where it issues its logits so
    (`logits_ahead`). Such a program's consumers take no turns on the tensor cores (FlashAttention 3's
    ping-pong, one's softmax under the other's products), which would only serialize two consumers each
    running its next logits under its own softmax (nvgpu-lean's model of the 4B call: S ahead with turns
    12% slower than without either, row 5368)."""
    stages = min(tiles.stages, steps)
    ahead = logits_ahead(family, tiles, value_width, dtype, stages=stages, peaks=peaks, parts=parts,
                         coded=_codes_of(key, read.hooked.scales) is not None, scaled=spec.headroom,
                         training=training)
    return _Program(family, tiles, steps, stages, spec.keys, spec.form == "causal", spec.queries, splits,
                    _per_span(spec, read.spans), parts=parts, products=declared(ahead, peaks=peaks))


NATURAL_NAME = "_natural"
"""A natural walk's mark in its launch's kernel's name (`_launched`'s `name`): a fold's rewalk's, a compile's
refusal of which steps the rewalk's tiles alone where it runs apart (`MosaicForward._rewalked_apart`)."""


def declared(ahead: bool, *, peaks: bool = False) -> tuple[mosaic.Product, ...]:
    """A forward walk's products as data (`mosaic.Product`), which its family places: each key tile's
    logits `Q Kᵀ` from the query's stage, fresh each step and issued a step ahead where `ahead`
    (`logits_ahead`), and but on a peak walk the weighted values onto the sums, their left the weights the
    step gives (`Issuer.accumulating`)."""
    logits = mosaic.Product("keys", "logits", left="query", transposed=True, ahead=ahead)
    if peaks:
        return (logits,)
    return logits, mosaic.Product("values", "sums", accumulate=True)


@tuning.lowered_once(("hooks", "spec", "tiles", "family", "training", "interpret", "peaks", "out_dtype",
                      "counted", "finishes", "target", "name", "splits", "page", "column_width", "shared_by",
                      "certified", "flags", "walk_rows", "parts", "registers"))
def _launched(query, key, value, read: _Read, *, hooks: Hooks, spec: algebra.Forward, tiles: Tiles,
              family: mosaic.Family, training: bool, interpret: bool,
              peaks: bool, out_dtype, counted: bool, finishes: bool, target: generation.GpuTarget,
              name: str, splits: int = 1, page: int | None = None,
              column_width: int | None = None, shared_by: int = 1,
              certified: bool = False, flags: bool = False, walk_rows: int | None = None, parts: int = 1,
              registers: int | None = None):
    """`_launch`'s kernel launched on its operands and the arrays its programs read beside them (`read`, the
    hooks' among them, `_Hooked`), its static `hooks` holding their arrays' shapes and dtypes in their place
    (its functions a key's, which `tuning.holdable` reads), lowered once for every call of its static facts
    and its operands' shapes and dtypes (`tuning.lowered_once`); `target` the GPU its config is derived for,
    `name` its kernel's (`mosaic.kernel` reports it as the launch tracing it names it) and `registers` its
    consumer's, which it states with it (`Family.kernel`)."""
    leaves, treedef = jax.tree.flatten(read)
    batch, queries, heads, width, value_width, keys_count, kv_heads = _extents(
        query, key, value, read, page=page, column_width=column_width, parts=parts)
    group, consumers = heads // kv_heads, tiles.consumers
    block_q, block_k = tiles.block_q, tiles.block_k
    spec = dataclasses.replace(spec, approximate=approximate(
        hooks, (block_q, block_k), value_width, differentiated=False, query=(block_q, width, query.dtype)))
    pair = consumers * block_q
    steps = keys_count // block_k
    program = _program_of(family, tiles, spec, read, key, steps, value_width, query.dtype, splits=splits,
                          peaks=peaks, parts=parts, training=training)
    # The output stores through the query's stage where the two are as wide (its rows are read before the
    # store); else it takes a stage of its own beside the query's and the rings (`_forward_allocations`).
    out_dtype = query.dtype if out_dtype is None else jnp.dtype(out_dtype)
    reuse = _stores_through_query(family, width, value_width, query.dtype, out_dtype)
    # A paged call's key tile of several pages loaded as its table reads it (`paged_reads`, `_paged_views`).
    params = param_types(hooks.score_params)
    kinds = tuple(kind for kind, _ in params)
    scoped, rings, after = _forward_allocations(family, tiles, width, value_width, query.dtype, training,
                                         program.stages, peaks=peaks, out_dtype=out_dtype, page=page,
                                         certified=certified, codes=_codes_of(key, read.hooked.scales),
                                         parts=parts, interpret=bool(interpret), params=params)
    out_type = _out_types(batch, queries, heads, value_width, out_dtype, splits=splits,
                          peaks=peaks, training=training, finishes=finishes,
                          certified=steps if certified else 0, flags=flags,
                          in_place=read.into is not None)
    layout = _Layout(treedef.num_leaves, treedef.unflatten, len(out_type), len(scoped) - 2,
                     peaks or (reuse and not training),
                     relays(family, tiles, peaks=peaks), _codes_of(key, read.hooked.scales) is not None,
                     bool(after))

    def body(*refs, streams, after=None):
        refs = layout.unpacked(*refs)
        read = refs.read
        located = functools.partial(_program_views, program, refs, read, peaks=peaks, group=group, page=page,
                                    batch=batch, parts=parts, queries=queries, kinds=kinds)
        # The rings opened once, before the roles' split (a gathered ring's barriers set once).
        streams = family.opened(streams, located()[3])
        turn = functools.partial(family.turn, refs.schedule, 1 if program.ahead else consumers)

        placed = functools.partial(_placed, program, located, read.spans, queries=queries)

        def consumer(index):
            row, head, column, views, pair_first, streamed = placed()
            block = _Block.of(program, row, head, pair_first + index * block_q, queries)
            afresh = functools.partial(_Block.afresh, program, row, head, index, queries)

            stages = (refs.q_smem.at[index], _query_barriers(refs.q_loaded, index, parts))
            # The query block's view, an fp32 call's its parts' (`_query_view`).
            q_view = _query_view(refs.q, row, pl.ds(block.first, block_q), head, batch=batch, parts=parts)
            if peaks:
                _peak_consumer(program, spec, index, q_view,
                               lambda: tuple(ref.at[afresh().state] for ref in refs.written), *stages,
                               refs.row_smem.at[index], refs.stages[0].at[index], streams, views, turn,
                               streamed, block.first, read.spans, read.hooked,
                               (row, head, lax.div(head, group)), hooks=hooks, interpret=interpret)
                return
            written = _written_refs(refs.written, read)
            # A block of the columns writes its own of the output's, the whole call's in place.
            columns_at = (pl.ds(column * value_width, value_width)
                          if column is not None and read.into is not None else None)
            viewed = functools.partial(_written_views, family, written=written,
                                       stages=tuple(stage.at[index] for stage in refs.stages),
                                       training=training, finishes=finishes, columns_at=columns_at,
                                       certified=certified)
            # The output's tile view and a certified walk's, read through the walk, formed with it; its rows'
            # outputs each formed where it is stored (`afresh`).
            ahead = viewed(block)
            out_stage = functools.partial(staged_after, after, None if after else refs.out_smem, index)
            _consumer(program, spec, index, q_view, ahead.out,
                      lambda: viewed(afresh()), *stages, out_stage, refs.row_smem.at[index], streams, views,
                      turn, streamed, block.first, read.spans, read.hooked,
                      (row, head, lax.div(head, group)), hooks=hooks, training=training,
                      anchors=read.anchors, interpret=interpret, certified=ahead.certify,
                      kept=read.kept, first_at=lambda: afresh().first,
                      relay=None if refs.relay is None else refs.relay.at[index],
                      codes=None if refs.converted is None or read.hooked.scales is None else _Codes.of(
                          family, tuple(tile.at[index] for tile in refs.converted), read.hooked.scales,
                          row if read.sources is None else read.sources[row], lax.div(head, group),
                              tiles.block_k, interpret), value_width=value_width)

        def producer():
            *_, views, _, own = placed()
            family.stream(streams, views, *own)

        # Each role forms what it reads (`_placed`), one switch's branch (`exclusive`).
        family.scheduled(consumers, _per_role(family, consumers, consumer), producer, exclusive=True,
                         issuer=_issuer(program, streams, refs, placed))

    products = forward_products(family, tiles, value_width, query.dtype, peaks=peaks, parts=parts,
                                ahead=bool(program.ahead), scaled=spec.headroom, training=training)
    entry = functools.partial(_entry, program, layout, body, (scoped, rings, after, products),
                              queries=queries, walk_rows=walk_rows)
    cost, model = _launch_flops(spec, family, query, key, value, keys_count, value_width, tiles,
                                peaks=peaks, training=training, counted=counted, shared_by=shared_by)
    grid = (heads, queries // pair * splits, batch)
    return family.kernel(entry, out_type=out_type, grid=grid,
                         grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
                         interpret=interpret,
                         approx_math=spec.approximate, name=name, cost=cost, model_flops=model,
                         registers=registers,
                         reducing_warpgroups=reducing(family, consumers,
                                                      hook_reduces(hooks) or hooks.output_mod is not None,
                                                      flagged=flags,
                                                      magnitudes=(spec.headroom and not peaks
                                                                  and read.hooked.magnitudes is None))
                         )(query, key, value, *leaves)


def _per_role(family: mosaic.Family, consumers: int, consumer: Callable) -> Callable:
    """`consumer(index)` as `family`'s roles run it: each consumer built at its static role where its issuer
    scopes its product scratch, a role's own share (`Issuer.product_scratch`, tcgen05's)."""
    if not scopes_products(family):
        return consumer
    return functools.partial(mosaic.per_role, roles=consumers, run=lambda role, index: consumer(index))


def _issuer(program: _Program, streams, refs, placed: Callable) -> Callable | None:
    """The products' role where its family places them on one (`Issuer.places_on_role`): the walk's steps it
    streams (`_placed`), each product issued a step ahead of the consumer that reads it, the query's stage its
    logits' left once its load is in (its one consumer's); None where the consumers issue their own."""
    family = program.family
    if not family.places_on_role:
        return None

    def issuer():
        *_, own = placed()
        family.issuing(streams, program.products, *own, first=own[0], lefts={"query": refs.q_smem.at[0]},
                       ready=_query_barriers(refs.q_loaded, 0, 1))

    return issuer


def _written_refs(written: tuple, read: _Read) -> tuple:
    """The refs a forward program writes its outputs into (`_Block`'s views read them by place): its
    launch's outputs, or the plain launch's held in place where it writes `into` them (its status place None,
    `_Block.status`)."""
    if read.into is not None:
        return (None, *read.into)
    return written


def _placed(program: _Program, located: Callable, spans: _Spans | None, *, queries: int) -> tuple:
    """A role's place: its batch row, head, column block and key views (`located`, `_program_views`), its
    pair's first query and their streamed range, formed in the role that reads them, its grid coordinates
    read there (`Family.program_index`'s `fresh`): read before the roles' split, they were live across the
    producer's register drop (an H100's fp32 decode step's streamed range spilled 16 bytes, et-lowreg-m; a
    B200's forward at width 256, row 5466)."""
    tiles = program.tiles
    pair = tiles.consumers * tiles.block_q
    row, head, column, views = located(fresh=True)
    pair_first = program.query_tile(queries // pair, fresh=True) * pair
    streamed = _streamed([program.reach(pair_first + index * tiles.block_q, spans, row)[:2]
                          for index in range(tiles.consumers)])
    return row, head, column, views, pair_first, streamed


def _program_views(program: _Program, refs, read: _Read, *, peaks: bool, group: int, page: int | None,
                   batch: int, parts: int, queries: int, kinds: tuple, fresh: bool = False) -> tuple:
    """A forward program's place and the key views it streams (`_launched`'s): its batch row, the query head
    and column block of the values it walks (the block `read` names), and its key and value views
    (`_key_views`; a peak walk's keys alone, an fp32 call's its parts', `_parted_views`) beside its score
    hook's parameters' that ride them (`score_views`, of `kinds`: a tile's at the program's query tile); its
    grid coordinates read where they are used where `fresh` (`Family.program_index`)."""
    family, tiles = program.family, program.tiles
    row, head = family.program_index("batch", fresh=fresh), family.program_index("heads", fresh=fresh)
    column = None if read.block is None else family.scalar_of(read.block)
    named = {"keys": refs.k} if peaks else {"keys": refs.k, "values": refs.v}
    views = _key_views(named, row, lax.div(head, group), read.sources, read.table, page=page,
                       block_k=tiles.block_k, column_block=column)
    if parts > 1:
        views = _parted_views(views, named, row, lax.div(head, group), batch=batch, parts=parts)
    if riding(family, kinds, "keys"):
        tile = program.query_tile(queries // (tiles.consumers * tiles.block_q), fresh=fresh)
        views = {**views, **score_views(family, read.hooked.score, kinds, "keys", row, head, tile)}
    return row, head, column, views


def _launch_flops(spec: algebra.Forward, family: mosaic.Family, query, key, value, keys_count: int,
                  value_width: int, tiles: Tiles, *, peaks: bool, training: bool, counted: bool,
                  shared_by: int) -> tuple:
    """A launch's cost and the model's FLOPs it owns (`_launched`'s `cost` and `model_flops`): its walk's
    statement, its work its emission's (`stated`), and the model's products it owns (`modelled`): the
    forward's two, a peak walk's `Q K^T`, a training walk's `P V` beside its `Q K^T` recomputed, a block of
    an inference call's columns its `P V` and its share of `Q K^T` (`shared_by`); a launch not `counted`
    none: a rerun (a guarded launch, the fold's natural walk), whose loop of at most one trip the FLOP count
    bounds by its trip, and a rare call's scaled op (`MosaicForward.scaled`), whose branch it takes where it
    is the larger, either counting work every step that runs at most rarely."""
    batch, queries, heads, width = query.shape
    owned_value = 0 if peaks else value_width
    owned_query = 0 if training and not peaks else width
    cost = (stated(spec, batch, heads, query, key, value,
                   tile_walk=TileWalk((queries, keys_count), (tiles.block_q, tiles.block_k), tiles.consumers,
                                      common=family.feeds_itself)) if counted
            else estimate(0))
    query_share = modelled(spec, batch, heads, spec.queries or queries, owned_query)
    value_share = modelled(spec, batch, heads, spec.queries or queries, owned_value)
    model = (0 if not counted else None if query_share is None or value_share is None
             else query_share // shared_by + value_share)
    return cost, model
