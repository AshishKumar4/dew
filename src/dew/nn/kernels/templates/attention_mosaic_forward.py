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
import math
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    attention_algebra as algebra,
    attention_portable,
    mosaic,
    operands,
    resources,
)
from dew.nn.kernels.templates.attention import (
    Draw,
    Dropout,
    Hooks,
    KeySpans,
    Param,
    Split,
    by_power,
    output_hooked,
    output_split,
    padded_call,
    prescale,
)
from dew.nn.kernels.templates.attention_mosaic_program import (
    Tiles,
    TileWalk,
    _block,
    _columns,
    _coords,
    _Guard,
    _Hooked,
    _hooked_of,
    _param,
    _per_span,
    _products,
    _Program,
    _Spans,
    _spans_of,
    _streamed,
    _views_of,
    approximate,
    bundle_of,
    extents,
    issued_into,
    issued_pairs,
    modelled,
    query_swizzle,
    reducing,
    refused,
    score_places,
    stagings,
    stated,
    transposed_swizzles,
    walked_extents,
)
from dew.nn.kernels.templates.cost import estimate


def _out_types(batch: int, queries: int, heads: int, value_width: int, out_dtype, *, splits: int, peaks: bool,
               training: bool, finishes: bool, certified: int = 0, merges: bool = False,
               flags: bool = False) -> tuple:
    """`_launch`'s outputs' types: a peak walk's maxima and anchor keys; a training walk's status, residual
    rows, anchor keys and split sums; else its status and output `[B, S·Qp, H, Dv]` (a split launch's slices
    along its query axis), and its finished rows where it `finishes`. A split launch that `merges` its slices
    gives their status, its merged output `[B, Qp, H, Dv]` and finished rows `[2, B, H, Qp]`, and the slices'
    states its merge read (`algebra.slice_state`: `[B, S, Qp, H, Dv]` and `[2, B, H, S, Qp]` fp32)."""
    rows_out = splits * queries
    # A plain launch that `flags` its rows states their rewalk and their block's beside their maxima
    # (`_Flags`).
    status_type = jax.ShapeDtypeStruct((3 if flags else 2, batch, heads, rows_out), jnp.float32)
    if merges:
        out_type = (status_type, jax.ShapeDtypeStruct((batch, queries, heads, value_width), out_dtype))
        if finishes:
            out_type += (jax.ShapeDtypeStruct((2, batch, heads, queries), jnp.float32),)
        return (*out_type, jax.ShapeDtypeStruct((batch, splits, queries, heads, value_width), jnp.float32),
                jax.ShapeDtypeStruct((2, batch, heads, splits, queries), jnp.float32))
    if peaks:
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


def _forward_allocations(family: mosaic.Family, tiles: Tiles, width: int, value_width: int, dtype,
                         residuals: bool, stages: int, *, peaks: bool = False, out_dtype=None,
                         pieces: int = 1, certified: bool = False, codes=None, parts: int = 1
                         ) -> tuple[list, dict]:
    """The forward program's allocations as `Family.scoped` takes them, in `_launch`'s order: its consumers'
    query stages and their rows' status; each one's output stage where the value is not as wide as the query
    and as narrow, `out_dtype`, the query's by default (the output stores through the query's stage where it
    is), a training forward's its split sums' fp32 (`_Residuals`); a training forward's or a peak walk's
    anchor keys' stage; the query loads' barriers and
    the consumers' turns' barrier; and its key and value streams in `stages` slots (`Family.streams`), a
    peak walk's keys alone, each slot loaded in `pieces` copies (a paged call's tile of several pages); the
    query's stage at its swizzle (`query_swizzle`); a certify form's fp32 stage of a step's sums last
    (`certified_walk`). An fp32 call's `parts` (`MosaicForward.parts`) stage each tile's parts one above the
    other, the query's and each slot's rows `parts` times theirs, a part a copy of its own, its query loads a
    barrier each."""
    consumers, block_q, block_k = tiles.consumers, tiles.block_q, tiles.block_k
    scoped = [mosaic.staged((consumers, parts * block_q, width), dtype, query_swizzle(family)),
              plgpu.SMEM((consumers, block_q), jnp.float32)]
    if residuals:
        scoped.append(plgpu.SMEM((consumers, block_q, value_width), jnp.float32))
    elif not peaks and not _stores_through_query(family, width, value_width, dtype, out_dtype):
        scoped.append(plgpu.SMEM((consumers, block_q, value_width), out_dtype or dtype))
    if residuals or peaks:
        scoped.append(plgpu.SMEM((consumers, block_q), jnp.int32))
    if certified:
        scoped.append(plgpu.SMEM((consumers, block_q, value_width), jnp.float32))
    if codes is not None:
        # Quantized keys' and values' tiles as the products read them, a consumer's own (`Family.converted`).
        scoped.append(mosaic.staged((consumers, block_k, width), dtype,
                                    transposed_swizzles(tiles, "keys").get("keys")))
        scoped.append(plgpu.SMEM((consumers, block_k, value_width), dtype))
    if relays(family, tiles, peaks=peaks):
        # The weights' way to the values' product where the family's warps split them (`Family.relaying`).
        scoped.append(mosaic.plain((consumers, block_q, block_k), dtype))
    scoped += [family.load_barriers(consumers * parts), family.schedule(consumers)]
    # Its rings stage the keys and values as stored: quantized `codes` where given, an fp32 call's parts
    # one above the other.
    stored = dtype if codes is None else codes
    blocks = {"keys": ((parts * block_k, width), stored)}
    if not peaks:
        blocks["values"] = ((parts * block_k, value_width), stored)
    # The logits' product reads the keys transposed.
    return scoped, family.streams(blocks, stages, consumers, swizzles=transposed_swizzles(tiles, "keys"),
                                  copies=dict.fromkeys(blocks, pieces * parts))


def relays(family: mosaic.Family, tiles: Tiles, *, peaks: bool = False) -> bool:
    """Whether a forward launch's weights `[block_q, block_k]`, the logits' accumulator, reach the values'
    product through shared memory (`Family.relay_bytes`): a decode step's 16-row mma.sync tile, its warps
    along the keys (`attention_mosaic_program.few_rows`); a peak walk takes no values' product."""
    return not peaks and family.relay_bytes(tiles.block_q, tiles.block_k) > 0


def guarded_columns(family: mosaic.Family, value_width: int, dtype, *, parts: int = 1) -> int:
    """The value columns a guarded launch walks at once (`_plain_forward`): a block of the family's swizzled
    columns (`Family.column_unit`); an fp32 call's quarter of one (`MosaicForward.parts`: its weights' parts
    formed beside the guard's two weight tiles and sums, at 64 x 64 on an H100 a whole unit's guarded walk
    spilled 24 bytes and its natural one 484, half a unit's natural one 16, et-fp32regs6), its values' block
    staged at the swizzle that fits it (`mosaic.staged`, unstated)."""
    unit = family.column_unit(dtype)
    return min(value_width, unit if parts == 1 else unit // 4)


GUARD_ROWS = 4
"""The row statistics a guarded walk carries beside a plain one's maximum and denominator (`algebra.Guard`:
each row's shift, reach, unshifted denominator and liveness), each held by the four threads an accumulator
row spans, two rows a thread: a `[block_q, 4]` fp32 tile's words a statistic (`guarded_tiles`)."""


def guarded_family(family: mosaic.Family, rows: int, block: int, columns: int) -> mosaic.Family:
    """A guarded launch's family at its `rows` x `block` share, arranged so its keys' magnitudes
    (`Family.key_magnitudes`: a `[columns, block]` value block read transposed in its layout) take whole row
    atoms (`Issuer.transposed_rows`): at the share's own arrangement where they do, else at the columns'
    rows (an A100's 256-wide softcap forward, 32 columns at a 64 x 32 share, cast to no 64-row atom,
    et-sm80sc5)."""
    arranged = family.specialized(rows, block)
    if columns % arranged.transposed_rows == 0:
        return arranged
    return family.specialized(min(rows, columns), block)


def guarded_tiles(family: mosaic.Family, tiles: Tiles, columns: int, dtype, width: int, *,
                  held_only: bool = True, parts: int = 1) -> Tiles | None:
    """The guarded launch's tiles beside the plain launch's `tiles` (`_plain_forward`): the largest of the
    family's row extents that divides the plain launch's query blocks (the guarded walk is a rare one's of
    its overflowed rows alone, so it takes smaller blocks where the plain launch's hold no guarded state: at
    128 rows on an A100, FS-4324-CENSUS7), the widest of the family's key extents (`walked_extents`) that
    divides the plain key tile, so the plain launch's padded keys are whole tiles of it, and the most of the
    plain launch's consumers, whose live set a consumer's registers hold (`resources.live_registers`): its
    shifted and unshifted sums of `columns` value columns fp32 and its rows' statistics (`GUARD_ROWS`)
    carried, beside a step's logits and weights fp32 and the two products the step forms (`algebra.stepped`),
    their operands' fragments where its family
    reads them from registers (`_products`: the logits' of `width`-deep queries and keys, the values' of its
    weights), its keys staged narrowest where the family's own swizzle's do not fit (`stagings`; ET-CENSUS80:
    the guarded launch at 64 x 64, width 128, spilled 132 bytes on an A100, its IR live set 274, 144 of it
    fragments). Uncounted, the statistics left 64 x 64 on two consumers spilling 20 bytes a thread at heads
    128 and 256 wide (ptxas, sm90a). None where none fits: the plain launch is then illegal at `tiles`
    (`MosaicForward.legal`); unless `held_only`, its least candidate then, which a launch pinned at `tiles`
    walks whatever its legality (a resource witness lowers as pinned). An fp32 call's `parts`
    (`MosaicForward.parts`) form its weights' parts one at a time beside them, its remainder and the part
    (`_parted_products`), and bound its keys' magnitudes from their first parts (et-fp32e: the parts formed
    at once and their magnitudes summed, 64 x 64 at width 64 spilled 1,820 bytes on an H100)."""
    def held(candidate: Tiles) -> bool:
        rows, block = candidate.block_q, candidate.block_k
        # Priced in the layout the guarded launch takes at its share (`guarded_family`).
        arranged = guarded_family(family, rows, block, columns)
        carried = [(rows, columns, jnp.float32)] * 2 + [(rows, 4 * GUARD_ROWS, jnp.float32)]
        logits, sums = [(rows, block, jnp.float32)] * 2, [(rows, columns, jnp.float32)] * 2
        fragments = _products(arranged, dtype, (rows, width, block, logits, False, True),
                              (rows, block, columns, logits[:1] + sums, True, False), narrow=candidate.narrow,
                              lhs_swizzle=query_swizzle(arranged))
        # Its keys' magnitudes read from the value tile (`Family.key_magnitudes`) beside the logits, a phase
        # of their own: ptxas holds the tile's load apart from the step's sums (counted beside them as well,
        # every A100 guarded tile was refused, gate80 46 of 96 against 70; the A100's guarded launch at width
        # 64 still spills 40 bytes counted apart, its decode path the Triton mainloop's until it does not).
        magnitudes = [logits[0], (columns, block, jnp.float32)]
        if parts > 1:
            # Its weights' part and remainder beside them (`_parted_products`), and its keys' magnitudes a
            # vector read beside the tile (`_Guard.magnitudes`), no tile of them.
            logits = [*logits, (rows, block, jnp.float32), (rows, block, dtype)]
            magnitudes = [logits[0]]
        live = resources.live_registers(arranged, carried, logits + sums, magnitudes, *fragments)
        if arranged.operands_in_registers:
            # Its products' fragments loaded beside the step's tiles, which they read (mma.sync: priced
            # apart, the guarded launch at 64 x 64 on two consumers, width 128, held 244 and spilled 208 to
            # 248 bytes on an A100 under every hook, FS-GUARD80-2; beside them it prices 372, so no
            # 64-column tile of its holds it there and the plain launch is illegal at width 128).
            live = max(live, *(resources.live_registers(arranged, carried, logits + sums + phase)
                               for phase in fragments))
        return live <= arranged.registers(candidate.consumers)

    blocks = [block for block in walked_extents(family, dtype, tiles.block_k) if tiles.block_k % block == 0]
    # A plain tile of fewer rows than a role's guards at its own rows (`few_rows`).
    queries = [rows for rows in extents(family, dtype, tiles.block_q) if tiles.block_q % rows == 0] or [
        tiles.block_q]
    # Its key blocks copied by its consumers together where its family's copies split them
    # (`Transport.shares_rows`: two mma.sync consumers' cp.async in whole 64-row parts).
    # An fp32 call's guarded walk on one consumer, the most registers (its parts beside the guard's two
    # weights and products: 64 x 64 on two spilled on an H100, et-fp32j).
    most = 1 if parts > 1 else tiles.consumers
    candidates = (dataclasses.replace(tiles, block_q=rows, block_k=block, consumers=consumers, narrow=narrow)
                  for rows in queries for consumers in range(most, 0, -1) for block in blocks
                  if family.shares_rows(block, consumers) for narrow in stagings(family))
    ranked = list(candidates)
    return next((candidate for candidate in ranked if held(candidate)), None if held_only else ranked[-1])


def _plain_forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                   masked: bool = False, **walk):
    """`MosaicForward`'s launch without residuals (`_inference_walk`), its launches walked by the algebra's
    guards (`algebra.walked_launches`)."""
    in_place = hooks.output_mod is None and (masked or _copy_costs_more(query, value))
    spec, launch, kept, finished = _inference_walk(query, key, value, spans, hooks, masked=masked,
                                                   flagged=in_place, **walk)
    return finished(algebra.walked_launches(spec, launch, kept, masked=masked, in_place=in_place))


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


def _whole_rows(live: jax.Array, splits: int) -> jax.Array:
    """A split launch's row live in every slice where it is in any (`[B, H, S·Qp]`): the guarded launch's
    slices cut its own key tiles, so a row's slices come from one launch, never two."""
    if splits == 1:
        return live
    shaped = live.reshape(*live.shape[:2], splits, -1)
    return jnp.broadcast_to(jnp.any(shaped, axis=2, keepdims=True), shaped.shape).reshape(live.shape)


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

    def whole(self, splits: int) -> "_Flags":
        """A row's flags set where any of its slices' are, its slices one (a merging walk's unsplit
        rewalk)."""
        rows = self.rows.reshape(*self.rows.shape[:2], splits, -1).max(axis=2)
        return _Flags.of(rows > 0, self.granule)


def _inference_walk(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                    scale: float, natural: bool, tiles: Tiles, interpret: bool, rows: bool = False,
                    out_dtype=None, masked: bool = False, splits: int = 1, table: jax.Array | None = None,
                    sources: jax.Array | None = None, subtracted: bool = False, merges: bool = False,
                    flagged: bool = False, scales: tuple[jax.Array, jax.Array] | None = None,
                    parts: int = 1):
    """The inference walk's launches: its spec, `launch(form, live, again)` and `kept` as
    `algebra.walked_launches` takes them and `finished`, its outputs from the walk's: `MosaicForward`'s launch
    without residuals, at `tiles`, under the default's guards over whole launches (`algebra.walked_launches`).
    The guarded launch (`_Guard`) runs a block of the family's swizzled value columns at a time: its walk
    carries its shifted and unshifted sums and forms both tiles' products a step (`algebra.stepped`), four
    `[block_q, Dv]` fp32 tiles beside its logits, which a block of the columns holds within a consumer's
    registers (ledger 1617). Each block's output is exact on its own (its rows' shifts its own columns' room,
    `algebra.room`); the rare launch takes its logits once a block. An output hook reads a row's whole output,
    so where there is one the blocks store theirs in fp32 and the hook runs on the rows they make in XLA
    (`template.output_hooked`, the reference's own), rounded once after it.

    With `rows`, the training forward that keeps its output for the backward's D (`takes_output_residuals`):
    the output and the residuals every mainloop's training forward gives, from the plain launch's finished
    rows, which a row its guards rewalk keeps too (its weighted values overflowed, not its maximum or
    denominator), no anchor key and the output as the split sums (`output_split`).

    A split walk that `merges` its slices merges them in its launches (`_launch`'s `merges`), the initial
    state's among them: its outputs the rows' whole, `[B, Q, H, Dv]` and the rows' finished `[B, H, Q]`, as
    an unsplit walk's, its guards' live rows any of a row's slices'.

    A `flagged` walk (one that rewalks in place, `algebra.walked_launches`' `in_place`) has its plain launch
    state its rows' rewalk flags (`_launch`'s `flags`, `_Flags`), which the masked rewalk reads as they are:
    no array operation between the two launches. An fp32 call's operands enter as their `parts`
    (`MosaicForward.parts`), stacked along the batch axis, each launch reading them so (`_launch`'s
    `parts`)."""
    merges = merges and splits > 1
    family = bundle_of(tiles, query.dtype)
    call, spec, (raw, k, v), (spans_of, hooked), taken = _prepared(
        query, key, value, spans, hooks, scale=scale, natural=natural, tiles=tiles, family=family,
        table=table, inference=True, subtracted=subtracted, scales=scales)
    # The values' dtype as the walk reads them: a quantized call's converted to the query's (`_Codes`).
    computed = query.dtype if scales is not None else value.dtype
    # A paged call's table, its rows' pages whole blocks of the call's, and its pools' page (`_launch`'s
    # `table`).
    paged = None if call.table is None or call.page is None else (call.table, call.page)

    def launch(form: algebra.Forward, live: jax.Array | None, again: bool = False, into: tuple | None = None
               ) -> tuple[tuple, jax.Array, jax.Array]:
        q, form = _rewalked(spec, form, raw, taken)
        # The natural form's rewalk at its own tiles (`Tiles.relaunch`), its spans' block map of them.
        at, at_spans = _form_tiles(spec, call, form, tiles, spans_of)
        # A rewalk's flags: those the plain launch stated of its rows (`flagged`), else its live rows'.
        flags = None if live is None else live if isinstance(live, _Flags) else _Flags.of(live, at.block_q)
        # A rewalk into the plain launch's outputs (`algebra.walked_launches`' `in_place`): masked, its rows
        # the live ones, the rest of its outputs left as the plain launch wrote them (`_Read.into`).
        # An in-place rewalk is of live rows (`algebra.walked_launches`).
        in_place = {"into": into, "keeping": flags.rows} if into is not None and flags is not None else {}
        stated = flagged and live is None and into is None and not again
        walk = _InferenceWalk(family, spec, call, hooks, hooked, k, v, interpret, rows, out_dtype, masked,
                              splits, paged, sources, merges, computed, parts)
        if flags is None or not form.headroom:
            return _plain_launches(walk, q, form, at, at_spans, flags, in_place, stated=stated, again=again,
                                   into=into)
        return _guarded_launches(walk, q, form, at, flags, in_place, into=into)

    return (spec, launch, functools.partial(_kept, splits=splits, merges=merges),
            functools.partial(_walked_outputs, queries=call.queries, width=value.shape[3], rows=rows,
                              splits=splits, merges=merges))


class _InferenceWalk(NamedTuple):
    """What an inference walk's launches share (`_inference_walk`'s `launch`): its family and form, the padded
    call and its hooks (`_prepared`), its keys and values, the walk's options, and the dtype its values are
    read in (a quantized call's converted to its query's, `_Codes`)."""

    family: mosaic.Family
    spec: algebra.Forward
    call: Any
    hooks: Hooks
    hooked: _Hooked
    k: jax.Array
    v: jax.Array
    interpret: bool
    rows: bool
    out_dtype: Any
    masked: bool
    splits: int
    paged: tuple | None
    sources: jax.Array | None
    merges: bool
    computed: Any
    parts: int = 1


def _plain_launches(walk: _InferenceWalk, q: jax.Array, form: algebra.Forward, at: Tiles, at_spans,
                    flags: "_Flags | None", in_place: dict, *, stated: bool, again: bool, into: tuple | None
                    ) -> tuple[tuple, jax.Array | None, jax.Array]:
    """An inference walk's plain launch, or a masked rewalk of sums that cannot overflow, at tiles `at` in
    `form` (`_inference_walk`'s `launch`): a launch a block of the value columns where its tiles walk a
    block (`Tiles.columns`), the blocks' outputs side by side, their rows' statistics alike; an output hook
    of a row's whole output in XLA past them, as the guarded launch's."""
    family, spec, hooks, hooked, k, v = walk.family, walk.spec, walk.hooks, walk.hooked, walk.k, walk.v
    interpret, rows, out_dtype, splits = walk.interpret, walk.rows, walk.out_dtype, walk.splits
    paged, sources, merges, parts = walk.paged, walk.sources, walk.merges, walk.parts
    walks = None if flags is None else flags
    blocks = _column_blocks(v, hooked, at.columns)
    split = len(blocks) > 1 and hooks.output_mod is not None
    # In place, a block of the columns reads them where they lie and writes its own of the whole
    # output's (`_launch`'s `column_width`), as the guarded launch's blocks do.
    launched = [_launch(q, k, v if into is not None and first is not None else values, at_spans,
                        held._replace(output=()) if split else held, None,
                        spec=form, hooks=_unhooked(hooks) if split else hooks, tiles=at,
                        family=family, training=False, interpret=interpret, parts=parts,
                        counted=form.natural == spec.natural and not again, kept=again,
                        finishes=rows, columns=first, splits=splits, table=paged, sources=sources,
                        out_dtype=jnp.float32 if split else out_dtype, walks=walks,
                        shared_by=len(blocks),
                        column_width=(values.shape[-1] if into is not None and first is not None
                                      else None), merges=merges, flags=stated and len(blocks) == 1,
                        **in_place)
                for first, values, held in blocks]
    if into is not None:
        assert flags is not None, "an in-place rewalk is of live rows"
        return (), None, flags.rows
    status, _, *finished = launched[0][:-2] if merges else launched[0]
    out = _output_whole(jnp.concatenate([block[1] for block in launched], axis=-1), hooks, hooked,
                        out_dtype or q.dtype, split)
    if stated and len(blocks) == 1:
        # Its rows' rewalk flags as its programs stated them (`algebra.walked_launches`' contract).
        return (out, *finished), None, _Flags.stated(status, at.block_q)
    overflowed = functools.reduce(jnp.logical_or, [block[0][1] > 0 for block in launched])
    return (out, *finished), status[0], overflowed


def _guarded_launches(walk: _InferenceWalk, q: jax.Array, form: algebra.Forward, at: Tiles,
                      flags: "_Flags", in_place: dict, *, into: tuple | None
                      ) -> tuple[tuple, jax.Array | None, jax.Array]:
    """An inference walk's guarded rewalk of its `flags`' rows in `form` (`_inference_walk`'s `launch`), at
    the guarded tiles beside `at` (`guarded_tiles`), a block of the value columns a launch or all of them in
    one."""
    family, spec, call, hooks, hooked, k, v = (walk.family, walk.spec, walk.call, walk.hooks, walk.hooked,
                                               walk.k, walk.v)
    interpret, rows, out_dtype = walk.interpret, walk.rows, walk.out_dtype
    masked, splits, computed = walk.masked, walk.splits, walk.computed
    paged, sources, merges, parts = walk.paged, walk.sources, walk.merges, walk.parts
    unit = guarded_columns(family, v.shape[-1], computed, parts=parts)
    # The plain launch's legality held a guarded tile at its tiles (`MosaicForward.legal`); a launch
    # pinned past it walks the least.
    guarded = guarded_tiles(family, at, unit, computed, q.shape[-1], held_only=False, parts=parts)
    # An fp32 call's keys' magnitudes over all their columns (`_Guard.magnitudes`), from its values' parts
    # summed exactly, read beside each tile: its blocks' transposed tiles are too few rows to load.
    magnitudes = None
    if parts > 1:
        whole = jnp.sum(v.astype(jnp.float32).reshape(parts, -1, *v.shape[1:]), axis=0)
        magnitudes = jnp.swapaxes(jnp.max(jnp.abs(whole), axis=-1), 1, 2)
    hooked_blocks = hooks.output_mod is not None
    assert guarded is not None, at
    guarded_spans = _spans_of(spec, call, guarded.block_q, guarded.block_k)
    # A merging walk's guarded rewalk walks each of its rows' keys whole, one program a query block (its
    # rows live in every slice where in any, `_whole_rows`), and merges nothing: its tiles and column
    # blocks are not the plain launch's, whose grid its legality holds resident (`MosaicForward.legal`),
    # and it runs where a row's sums overflowed alone.
    rewalk_splits = 1 if merges else splits
    if merges:
        flags = flags.whole(splits)
        in_place = {"into": into, "keeping": flags.rows} if into is not None else {}
    walks = flags if masked or into is not None else None
    # A masked launch's rows its own whole, their finished maxima and denominators beside their output
    # (`kept`): a fold's natural rewalk's are its rows' own, in its units.
    finishes = rows and (masked or into is not None)
    blocks, finished = [], ()
    # The guarded launch reads its live rows and its values' largest magnitudes (`algebra.room`), and
    # the initial state's sums of its own columns (`_column_blocks`), a block a launch at any width.
    # Each block reads its columns of the values where they lie (`_launch`'s `column_width`), no copy of
    # them, and their magnitudes from the tiles it walks (`Family.key_magnitudes`).
    column_blocks = _column_blocks(v, hooked, unit, whole=False)
    # Every block in one launch, a program a block (`_launch`'s `column_blocks`), where the blocks share
    # their state, no initial state's sums of their own columns: a decode step's guarded rewalk at width
    # 128 took two launches of 1.74 us each (ET-DPROF-h100, row 4043).
    together = len(column_blocks) > 1 and hooked.initial is None
    for first, columns, held in column_blocks[:1] if together else column_blocks:
        guard = _Guard(flags.rows, magnitudes)
        launched = _launch(
            q, k, v, guarded_spans, held._replace(output=()), guard, spec=form, hooks=_unhooked(hooks),
            tiles=guarded, training=False, interpret=interpret,
            family=guarded_family(bundle_of(guarded, q.dtype), guarded.block_q, guarded.block_k, unit),
            columns=first, parts=parts,
            splits=rewalk_splits, table=paged, sources=sources,
            out_dtype=jnp.float32 if hooked_blocks else out_dtype,
            finishes=finishes, column_width=columns.shape[-1] if first is not None else None,
            walks=walks, column_blocks=len(column_blocks) if together else 1, **in_place)
        if into is not None:
            # In place, its status alone: its rows are written into the refs.
            continue
        _, block, *block_rows = launched
        blocks.append(block)
        # Every block's rows alike: its logits are the row's whole.
        finished = finished or tuple(block_rows)
    if into is not None:
        return (), None, flags.rows
    out = _output_whole(jnp.concatenate(blocks, axis=-1), hooks, hooked, out_dtype or q.dtype,
                        hooked_blocks)
    # `walked_launches` reads the plain launch's maxima and overflows.
    return (out, *finished), None, flags.rows


def _kept(live: jax.Array, guarded: tuple, plain: tuple, *, splits: int, merges: bool) -> tuple:
    """The inference walk's outputs from its rewalk's and its plain launch's (`_inference_walk`'s `kept`):
    the rewalk's at its `live` rows (its output, and a masked launch's finished rows), the plain launch's
    elsewhere and past them; a merged walk's a row's, live where any of its slices is."""
    live = (jnp.any(live.reshape(*live.shape[:2], splits, -1), axis=2) if merges
            else _whole_rows(live, splits))
    taken = (jnp.where(jnp.swapaxes(live, 1, 2)[..., None], guarded[0], plain[0]),
             *(jnp.where(live, mine, theirs) for mine, theirs in zip(guarded[1:], plain[1:], strict=False)))
    return (*taken, *plain[len(guarded):])


def _walked_outputs(walked: tuple, *, queries: int, width: int, rows: bool, splits: int, merges: bool):
    """The inference walk's outputs from its launches' (`_inference_walk`'s `finished`), unpadded to the
    call's `queries` and value `width`: the output, and with `rows` its finished maxima and denominators, an
    anchor key of none and the output as the split sums. An unmerged split launch's are its slices along the
    query axis (`_launch`'s `splits`): `[B, S, Q, H, Dv]`, its rows `[B, H, S, Q]`, each slice's maxima and
    denominators its own, for a merge (`attention_decode`); a merging one's the rows' whole, theirs alone."""
    out, *rows_of = walked
    if splits > 1 and not merges:
        out = out.reshape(out.shape[0], splits, -1, *out.shape[2:])[:, :, :queries, :, :width]
        maximum, total = (array.reshape(*array.shape[:2], splits, -1)[..., :queries] for array in rows_of[0])
        return out, maximum, total
    out = out[:, :queries, :, :width]
    if not rows:
        return out
    maximum, total = (array[:, :, :queries] for array in rows_of[0])
    if merges:
        return out, maximum, total
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
    if (at.block_k, at.consumers) == (tiles.block_k, tiles.consumers):
        return at, spans_of
    return at, _spans_of(spec, call, at.block_q, at.block_k)


def _output_hooked(out: jax.Array, hooks: Hooks, hooked: _Hooked) -> jax.Array:
    """The normalized output `out` `[B, H, Qp, Dv]` fp32 under the output hook in XLA, its parameters as the
    launches read them (`hooked`, padded to the call's blocks; `template.output_hooked`)."""
    params = tuple(Param(array, param.kind) for array, param in zip(hooked.output, hooks.output_params,
                                                                     strict=True))
    return output_hooked(out, hooks.output_mod, params)


def _trained_forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                     scale: float, natural: bool, tiles: Tiles, interpret: bool
                     ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    """`MosaicForward`'s training launches at `tiles`, two a form: a walk of the logits alone for each row's
    anchor, its first key at the keys' largest logit (`_peak_consumer`), then the algebra's training step
    at that anchor and maximum (`algebra.anchored_step`) taken in place, the split sums `B` and the rows'
    sums its residuals, with no value row kept or picked; the output is `B` and the anchor's own term over the
    denominator
    (`algebra.split_output`, `output_of`), its value row gathered in XLA.

    A program walks its keys once, so the default's guards compose over whole launches
    (`algebra.walked_launches`): the fold's rewalk is this kernel's natural launch, and rows whose weighted
    values overflowed take the plain-JAX mainloop's guarded training walk (its normalized split sums, shift
    -1, `algebra.split_residuals`), which this kernel's single pass does not take."""
    family = bundle_of(tiles, query.dtype)
    call, spec, (raw, k, v), (spans_of, hooked), taken = _prepared(
        query, key, value, spans, hooks, scale=scale, natural=natural, tiles=tiles, family=family)
    value_width = value.shape[3]

    def launch(form: algebra.Forward, live: jax.Array | None, again: bool = False):
        q, form = _rewalked(spec, form, raw, taken)
        if live is None:
            counted = form.natural == spec.natural and not again
            # The natural form's rewalk at its own tiles (`Tiles.relaunch`), its spans' block map of them.
            at, at_spans = _form_tiles(spec, call, form, tiles, spans_of)
            found = _launch(q, k, v, at_spans, hooked, None, spec=form, hooks=hooks, tiles=at,
                            family=family, training=False, interpret=interpret, peaks=True, counted=counted,
                            kept=again)
            # A training launch a block of the value columns where its tiles walk a block (`Tiles.columns`,
            # row 2526), each block's split sums side by side, their rows alike.
            launched = [_launch(q, k, values, at_spans, held, None, spec=form, hooks=hooks, tiles=at,
                                family=family, training=True, interpret=interpret, anchors=found,
                                counted=counted, kept=again, columns=first)
                        for first, values, held in _column_blocks(v, hooked, at.columns)]
            status, rows, anchors, _ = launched[0]
            others = jnp.concatenate([block[3] for block in launched], axis=-1)
            status = status.at[1].set(functools.reduce(jnp.maximum, [block[0][1] for block in launched]))
            split = Split(others, rows[2], rows[3], rows[4])
            # Each row's anchor's value row (`[B, H, Qp, Dv]`): its key head's at its key, gathered.
            batch, kv_heads, keys_count = v.shape[0], v.shape[2], v.shape[1]
            kv_of = jnp.arange(q.shape[2]) // (q.shape[2] // kv_heads)
            places = ((jnp.arange(batch)[:, None, None] * kv_heads + kv_of[None, :, None]) * keys_count
                      + anchors)
            value_rows = jnp.take(jnp.swapaxes(v, 1, 2).reshape(-1, v.shape[-1]), places, axis=0)
            # Under dropout the anchor's own weight takes its draw's multiplier, as `algebra.joined` does.
            own = split.own if hooks.drop is None else split.own * _anchor_multipliers(hooks.drop, anchors)
            acc = algebra.split_output(split.others, own, value_rows)
            out = algebra.output_of(acc, rows[1], guarded=form.guarded or hooks.initial is not None)
            out = _output_hooked(out, hooks, hooked)
            overflowed = (status[1] > 0) | algebra.overflowed(acc)
            return ((jnp.swapaxes(out, 1, 2).astype(q.dtype), rows[0], rows[1], anchors, split), status[0],
                    overflowed)
        # The overflowed rows' guarded training walk, the plain-JAX mainloop's over this call's padding in
        # this launch's form (the rewalk's units kept, `algebra.unfolded`); its maxima and overflows
        # `walked_launches` does not read.
        walked = call if q is raw else dataclasses.replace(call, query=by_power(call.query, taken))
        out, rows = attention_portable.padded_walk(form, walked, hooks, block_q=tiles.block_q,
                                                   block_k=tiles.block_k, residuals=True)
        maximum, total, anchors, split = rows
        widened = (_widened(out, v.shape[-1]), maximum, total, anchors,
                   split._replace(others=_widened(split.others, v.shape[-1])))
        return widened, maximum, jnp.zeros_like(live)

    def kept(live: jax.Array, guarded, plain):
        # The output's query rows lie on its axis 1 (`[B, Qp, H, Dv]`), every residual's on its axis 2
        # (`[B, H, Qp, ...]`).
        out = jnp.where(jnp.swapaxes(live, 1, 2)[..., None], guarded[0], plain[0])

        def rows(new: jax.Array, old: jax.Array) -> jax.Array:
            return jnp.where(live.reshape(live.shape + (1,) * (new.ndim - live.ndim)), new, old)

        return (out, *jax.tree.map(rows, tuple(guarded[1:]), tuple(plain[1:])))

    out, maximum, total, anchors, split = algebra.walked_launches(spec, launch, kept)
    queries = call.queries
    split = Split(split.others[:, :, :queries, :value_width], split.own[:, :, :queries],
                  split.others_total[:, :, :queries], split.shift[:, :, :queries])
    return (out[:, :queries, :, :value_width], maximum[:, :, :queries], total[:, :, :queries],
            anchors[:, :, :queries], split)


def _anchor_multipliers(drop: Dropout, anchors: jax.Array) -> jax.Array:
    """Each row's anchor's dropout multiplier (`Draw`) at its query and its anchor key `anchors`
    `[B, H, Qp]`, every batch row and query head its draw's."""
    batch, heads, queries = anchors.shape
    draw = Draw((drop.seed[0], drop.seed[1]), drop.rows.reshape(batch, 1, 1), drop.heads.reshape(1, heads, 1),
                drop.rate)
    return draw(jnp.arange(queries, dtype=jnp.int32), anchors)


def _widened(array: jax.Array, width: int) -> jax.Array:
    """`array` with its last axis padded to `width`, as this mainloop's launch stages values (`_columns`)."""
    return jnp.pad(array, ((0, 0),) * (array.ndim - 1) + ((0, width - array.shape[-1]),))


def _prepared(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
              scale: float, natural: bool, tiles: Tiles, family: mosaic.Family,
              table: jax.Array | None = None, inference: bool = False, subtracted: bool = False,
              scales: tuple[jax.Array, jax.Array] | None = None):
    """A call as this mainloop takes it at `tiles`: padded to the program's blocks (`padded_call`, `_block`;
    an `inference` walk's queries to its query blocks alone, which every launch of it divides: a decode step's
    four rows padded to 128 took 21 of B128's 795 us at 4096 keys, ET-DECODE7-h100) and its heads to whole
    swizzled rows (`_columns`), with its `algebra.Forward` (every form guarding its sums where its values can
    overflow them, the fold its logits' reach, `algebra.walked_launches`) and the arrays its programs read
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
    call = padded_call(by_power(query, power / taken), key, value, spans, hooks, _block(tiles), table,
                       query_block=tiles.consumers * tiles.block_q if inference else None, by_position=True)
    spec = algebra.Forward(spans.form, spans.keys, ids=spans.query_ids is not None, scale=scale * taken,
                           score_mod=hooks.score_mod, natural=natural,
                           headroom=algebra.headroom(query.dtype if scales is not None else value.dtype),
                           initial=hooks.initial is not None,
                           queries=(call.queries if call.query.shape[1] > call.queries or call.per_span > 1
                                    else None),
                           prescale=power / taken, subtracted=subtracted)
    unit = family.column_unit(query.dtype)
    operands = tuple(_columns(array, unit) for array in (call.query, call.key, call.value))
    hooked = _hooked_of(hooks, call, unit)
    if scales is not None:
        # As long as the call's padded keys (a paged call's slots), which every walk's tiles lie within.
        keys = call.key.shape[1] if call.table is None else call.table.shape[1] * call.page
        keys = max(keys, -(-scales[0].shape[-1] // tiles.block_k) * tiles.block_k)
        hooked = hooked._replace(scales=tuple(
            jnp.pad(array.astype(jnp.float32), ((0, 0), (0, 0), (0, keys - array.shape[-1])))
            for array in scales))
    read = (_spans_of(spec, call, tiles.block_q, tiles.block_k), hooked)
    return call, spec, operands, read, taken


def _rewalked(spec: algebra.Forward, form: algebra.Forward, query: jax.Array, taken: float
              ) -> tuple[jax.Array, algebra.Forward]:
    """A launch's query operand and form: the fold's natural rewalk (`algebra.unfolded`) of a call whose
    plain launch took its power of two into its scale (`taken`, `_prepared`) the query taken by the power
    and the scale left after it, its products finite where the reference's are; every other launch the
    call's own."""
    if taken == 1.0 or form.natural == spec.natural:
        return query, form
    return by_power(query, taken), dataclasses.replace(form, scale=form.scale / taken, prescale=taken)


class _Read(NamedTuple):
    """What a forward launch's programs read beside its operands (`_launch`), its kernel's refs flattened:
    the spans' block map, the hooks' arrays, the guard's live rows, a training walk's anchors, a paged call's
    table, a masked launch's walked query blocks, the rows' sources and a column block's index."""
    spans: _Spans | None
    hooked: _Hooked
    guard: _Guard | None
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
    q: Any
    k: Any
    v: Any
    read: _Read
    written: tuple
    q_smem: Any
    row_smem: Any
    out_smem: Any
    stages: tuple
    q_loaded: Any
    schedule: Any
    relay: Any = None
    converted: tuple | None = None


class _Layout(NamedTuple):
    """How `_launched` lays its kernel's refs out: its read arrays' tree, its outputs' count, the stages'
    count (`_forward_allocations`'), and whether the output stores through the query's stage."""
    treedef: Any
    outputs: int
    arrays: int
    reuse: bool
    relays: bool = False
    converts: bool = False

    def unpacked(self, q_ref, k_ref, v_ref, *refs) -> _Refs:
        """A program's refs, its kernel's then its scoped allocations': the weights' relay tile last where
        the launch `relays` them (`relays`), a quantized call's converted keys' and values' tiles before it
        where it `converts` them (`_forward_allocations`' `codes`)."""
        read, refs = refs[:self.treedef.num_leaves], refs[self.treedef.num_leaves:]
        written, refs = refs[:self.outputs], refs[self.outputs:]
        (q_smem, row_smem, *stages), (q_loaded, schedule) = refs[:self.arrays], refs[self.arrays:]
        relay = stages.pop() if self.relays else None
        converted = (stages[-2], stages[-1]) if self.converts else None
        stages = stages[:-2] if self.converts else stages
        out_smem, stages = (q_smem, stages) if self.reuse else (stages[0], stages[1:])
        return _Refs(q_ref, k_ref, v_ref, self.treedef.unflatten(read), tuple(written), q_smem, row_smem,
                     out_smem, tuple(stages), q_loaded, schedule, relay, converted)

    def read(self, refs: tuple) -> _Read:
        """What a kernel's entry `refs` (its three operands first) read."""
        return self.treedef.unflatten(refs[3:3 + self.treedef.num_leaves])


class _Block(NamedTuple):
    """A consumer's query block in its launch's outputs: its first query, its output rows `(row, queries,
    head)` and its rows' statistics' places `(row, head, queries)`, a slice's its own along the outputs'
    query axis (`_Program.splits`)."""
    first: jax.Array
    rows: tuple
    state: tuple

    @classmethod
    def of(cls, program: _Program, row, head, first, queries: int) -> "_Block":
        at = pl.ds(program.split() * queries + first, program.tiles.block_q)
        return cls(first, (row, at, head), (row, head, at))

    def status(self, ref) -> tuple:
        """Its rows' maxima and overflows in a launch's status `ref` (`walked_launches`), or where the launch
        flags its rows their rewalk and their block's beside their maxima (`_Flags`)."""
        return tuple(ref.at[place, *self.state] for place in range(ref.shape[0]))

    def inference(self, written: tuple, finishes: bool) -> tuple:
        """Its status, output and (where the launch `finishes` them) finished rows' views in an inference
        launch's outputs (`_out_types`)."""
        status_ref, out_ref, *finished = written
        rows = tuple(finished[0].at[place, *self.state] for place in range(2)) if finishes else None
        return self.status(status_ref), out_ref.at[self.rows], rows

    def merging(self, written: tuple, finishes: bool) -> tuple:
        """Its views in a launch that merges its slices (`_out_types`' `merges`): its slice's status, the
        merged output's and (where the launch `finishes` them) finished rows' at its rows unsliced, and every
        slice's state at its rows (`algebra.Partial` of views `[S, bq, Dv]` and `[S, bq]`), which its
        merge's slots are (`Family.fixed_up`)."""
        status_ref, out_ref, *rest = written
        row, sliced, head = self.rows
        at = pl.ds(self.first, sliced.size)
        rows = tuple(rest[0].at[place, row, head, at] for place in range(2)) if finishes else None
        sums, states = rest[-2:]
        slots = algebra.Partial(sums.at[row, :, at, head], states.at[0, row, head, :, at],
                                states.at[1, row, head, :, at])
        return self.status(status_ref), out_ref.at[row, at, head], rows, slots

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
            hooked: _Hooked, place: tuple, whole, turn, *, hooks: Hooks, drawn: bool,
            held: tuple | None = None, product: Callable[[], jax.Array] | None = None,
            interpret: bool = False, codes: _Codes | None = None) -> tuple:
    """A key tile's first key, coordinates and logits under the score hook (batch row and head `place`, its
    queries from `first`), and their products where the logits are a factor of them (`algebra.logits_of`),
    every walk's (`_consumer`'s, `_peak_consumer`'s), so a peak walk's logits are the training walk's to the
    bit; its places read by the walk's dropout where it draws it (`drawn`). A tile every query sees whole
    takes no mask: one walk, the choice a tile's (a walk of each, these rings' waits and arrivals in two
    loops, faulted on an H100 under jax 0.11.2's lowering). The tile's `Q Kᵀ` its own fresh product, or
    `product()`'s where the walk issues it (`_deferred`'s, beside the step before's P·V)."""
    row, head = place
    family, tiles = program.family, program.tiles
    shape = (tiles.block_q, tiles.block_k)
    if product is not None:
        s = product()
    elif program.parts > 1:
        # An fp32 call's logits its parts' products (`product.terms`), smallest first, into one fresh
        # accumulator in place (`Family.issued`: a term's own accumulator read before the next serializes
        # wgmma's, ptxas C7512), the partner's turn once they are in.
        keys = program.parted(slots["keys"], tiles.block_k)
        s = family.summed(shape, lambda acc: family.issued(acc, q.refs, keys.T.refs, interpret=interpret),
                          interpret=interpret)
        if turn is not None:
            turn()
    else:
        keys = slots["keys"] if codes is None else codes.keys(slots["keys"])
        s = family.product(q, keys.transpose((1, 0)), shape, turn, interpret=interpret)
    slots.release("keys")
    k_first = index * tiles.block_k
    if codes is not None:
        # The quantized keys' products their codes', each key's column scaled to its keys' (`_Codes`).
        s = s * algebra.across(codes.key_scales(k_first), s)
    # The tile's places formed a step, not held across the walk beside its sums (`_coords`).
    # Its mask chosen at run time (`whole`): the masked tile's reads.
    read = algebra.places_read(spec, masked=True, hooked=score_places(hooks), dropped=drawn,
                               padded=spec.keys % tiles.block_k != 0)
    coords = _coords(family, spans, row, first, k_first, shape, read=read,
                     draw=None if hooked.drop is None else hooked.drop.draw(row, head), held=held)
    params = tuple(_param(family, ref, param.kind, row, head, first, s, k_first)
                   for ref, param in zip(hooked.score, hooks.score_params, strict=True))
    logits, products = algebra.logits_of(spec, s, coords, params, masked=jnp.logical_not(whole(index)))
    return k_first, coords, logits, products


def _initial(family: mosaic.Family, spec: algebra.Forward, hooked: _Hooked, rows: tuple,
             live: jax.Array | None) -> algebra.Partial:
    """The call's initial state at a consumer's `rows` (batch row, head, queries) in the family's layouts
    (`algebra.resumed`), its `live` rows the guard's."""
    assert hooked.initial is not None
    m0, total0, acc0 = hooked.initial
    return algebra.resumed(spec, family.rows_of(m0.at[rows]), family.rows_of(total0.at[rows]),
                           family.tile_of(acc0.at[rows]), live=live)


def _empty(family: mosaic.Family, shape: tuple[int, int], live: jax.Array | None) -> algebra.Partial:
    """`algebra.started`'s empty state of a walk's `[bq, Dv]` sums (`shape`) in the family's layouts,
    headroomed at the guard's `live` rows."""
    rows = shape[0]
    start = algebra.Partial(family.zeros(shape), family.full((rows,), algebra.LEAST), family.zeros((rows,)))
    if live is None:
        return start
    # No key weighed yet: no shift and no reach, `algebra.headroom_of`'s of an empty state, stated rather than
    # reduced from its zero sums (Mosaic reduces no constant tile: a 16-row mma.sync walk's `[16, 32]`).
    nothing = family.zeros((rows,))
    return start._replace(guard=algebra.Guard(nothing, nothing, start.acc, start.total, live))


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


def _status_stored(family: mosaic.Family, spec: algebra.Forward, state: algebra.Partial, status_views: tuple,
                   row_smem) -> None:
    """A consumer's rows' status into `status_views` (`_launch`'s): each row's maximum in the walk's units, a
    row that saw no key -inf (`algebra.reached`'s convention), every other its maximum, `LEAST` included,
    and 1 where its weighted values overflowed; or, where the launch flags its rows (three views, `_Flags`),
    each one's rewalk and its block's, which the masked rewalk reads as they are (`algebra.rewalks`)."""
    overflowed = jnp.where(algebra.overflowed(state.acc), 1.0, 0.0)
    maximum = jnp.where(algebra.no_keys(state.total), -jnp.inf, state.m)
    status = (maximum, overflowed)
    if len(status_views) == 3:
        live = jnp.where(algebra.rewalks(spec, maximum, overflowed > 0)[0], 1.0, 0.0)
        status = (maximum, live, jnp.broadcast_to(jnp.max(live), live.shape))
    family.stored(tuple(map(family.as_row, status)), status_views, smem=row_smem)

def _merge_of(program: _Program, family: mosaic.Family, hooked: _Hooked, merging: tuple | None,
              state_rows: tuple) -> "_Merge | None":
    """A consumer's merge of its slices (`_Merge`) where its launch merges them (`merging`, its slots and
    semaphores), the initial state's rows read in its `family` at its last slice's; else None."""
    if merging is None:
        return None
    slots, ready = merging
    initial = None if hooked.initial is None else functools.partial(_initial_rows, family, hooked, state_rows)
    return _Merge(slots, ready, program.split(), program.splits, initial)


def _query_of(program: _Program, q_view, q_smem, q_loaded):
    """A consumer's query block loaded from `q_view` into its stage and held (`Family.held`), or an fp32
    call's parts (`MosaicForward.parts`, `q_view` their `mosaic.Pieces`) one above the other, a copy and a
    barrier each, read where they lie (`_Program.parted`)."""
    family, block_q = program.family, program.tiles.block_q
    if program.parts == 1:
        family.loaded(((q_view, q_smem, q_loaded),))
        return family.held(q_smem)
    family.loaded(tuple((view, q_smem.at[pl.ds(part * block_q, block_q)], barrier)
                        for part, (view, barrier) in enumerate(zip(q_view, q_loaded, strict=True))))
    return program.parted(q_smem, block_q)


def _stream_dtype(view: tuple):
    """A stream's elements' dtype from its view (`mosaic.viewed`'s): its ref's, or its pieces' first's."""
    ref = view[0]
    return (ref[0] if isinstance(ref, mosaic.Pieces) else ref).dtype


def _dot(family: mosaic.Family, interpret: bool, lhs: jax.Array, rhs) -> jax.Array:
    """A step's fresh product of a register operand `lhs` with a tile `rhs` (`algebra.Keys.dot`): the
    family's, or with an fp32 call's parts their terms' (`_parted_products`)."""
    if isinstance(rhs, operands.Parts):
        return _parted_products(family, None, None, lhs, rhs, interpret, shape=(lhs.shape[0], rhs.shape[1]))
    return family.registers_product(lhs, rhs, interpret)

def _consumer(program: _Program, spec: algebra.Forward, consumer, q_view, out_view, status_views, q_smem,
              q_loaded, out_smem, row_smem, streams, views: dict, turn, streamed: tuple, first,
              spans: _Spans | None, hooked: _Hooked, guard: _Guard | None, place: tuple, *, hooks: Hooks,
              residuals: _Residuals | None, anchors: tuple | None = None, interpret: bool,
              finished: tuple | None = None, certified: tuple | None = None, kept=None,
              merging: tuple | None = None, relay=None, codes: _Codes | None = None):
    """A consumer's query block (batch row, head and key head `place`): its walk of the key tiles it reads
    through the family's streams (`algebra.plain_step` a step) from the hooks' initial state, the score hook
    on each tile, passing those of the program's `streamed` range its partner alone reads; then its output
    under the output hook stored through `out_smem` (the query's stage where the widths agree), and its rows'
    maxima and whether their weighted values overflowed through `row_smem` to `status_views`
    (`algebra.walked_launches`). The plain walk's running sums are the family's accumulator, issued into in
    place; the guarded walk (`guard`, its rows `live` shifting) carries them, each tile's product fresh
    (`algebra.stepped`). With `residuals` the walk is the training step's at the `anchors` a peak walk found
    (`algebra.anchored_step`): the accumulator holds the split sums, every key's weighted values but the
    anchor's, which it stores with the rows' sums and status, no output (the output's anchor term joins in
    XLA, `_trained_forward`). With `certified` (its views and an fp32 stage, `_Block.certified`), the plain
    walk stores its state after each key tile (`certified_walk`). With `kept` (a launch into the plain
    launch's outputs, `_Read.kept`), it writes its rows `kept` alone, every other row of its views as it was
    (`_kept_rows`). With `merging` (a split launch that merges its slices: their states' slots and its
    semaphores, `_merge`), its output is its slices' merge, the last slice's program's. With `relay` (its
    own tile, `relays`), its weights reach the values' product through it. With `codes` (a quantized call's,
    `_Codes`), each step converts its slots' codes and scales its keys' and values'."""
    row, head, kv_head = place
    family, tiles = program.family, program.tiles
    if relay is not None:
        # Its weights reach the values' product through its own tile (`relays`).
        family = family.relaying(relay)
    lo, hi, whole = program.reach(first, spans, row)
    q = _query_of(program, q_view, q_smem, q_loaded)
    # The output's sums are as wide as its stage (a training walk's its split sums'), the values' width
    # whatever the query's.
    value_width = out_smem.shape[-1]
    # The weighted values' running sums live in the family's accumulator (`Family.walk`), from the initial
    # state's where there is one; the walk carries the maxima and denominators, the state's `acc` an empty
    # place the algebra passes through (`algebra.Keys.accumulate`), a constant no loop may carry.
    state_rows = (row, head, pl.ds(first, tiles.block_q))
    start, live, merged = _started(program, spec, guard, hooked, state_rows, first, value_width,
                                   training=residuals is not None, merges=merging is not None)
    pl.when(consumer == 1)(turn)
    # The queries' intervals and ids read once, before the walk (`_query_bounds`).
    held = None if spans is None else program.bounds(first, spans, row)

    def scored(index, slots, product=None):
        return _scored(program, spec, q, index, slots, first, spans, hooked, (row, head), whole, turn,
                       hooks=hooks, drawn=hooked.drop is not None, held=held, product=product,
                       interpret=interpret, codes=codes)

    # The values' tiles as the products read them: a quantized call's in the query's dtype (`_Codes`), an
    # fp32 call's its parts (`_Program.parted`).
    values = functools.partial(_value_tile, program, (tiles.block_k, value_width),
                               q_smem.dtype if codes is not None else _stream_dtype(views["values"]), turn)
    dot = functools.partial(_dot, family, interpret)

    def step(index, slots, carry, accumulators):
        k_first, coords, logits, products = scored(index, slots)
        converted = None if codes is None else codes.values(slots["values"], k_first)
        tile = values(slots, converted)
        if guard is not None:
            magnitudes = guard.magnitudes
            # Its keys' largest value magnitudes from the value tile it holds (`Family.key_magnitudes`).
            def largest():
                if magnitudes is None:
                    return family.key_magnitudes(slots["values"] if converted is None else converted)
                # An fp32 call's, each key's over all its columns, read beside the tile (`_Guard`).
                k_first = index * tiles.block_k
                return family.columns_of(magnitudes.at[row, kv_head, pl.ds(k_first, tiles.block_k)])

            keys = algebra.Keys(coords, logits, tile, dot, products=products, largest=largest)
            return algebra.plain_step(spec, carry, keys), accumulators
        # The sums are the accumulator's, which the step takes the tile into (`Keys.accumulate`).
        state = algebra.Partial(accumulators[0], *carry)
        keys = algebra.Keys(coords, logits, tile, dot, _accumulated(family, interpret), products)
        state = algebra.plain_step(spec, state, keys)
        if certified is not None:
            _certified_step(family, certified, index, state, row_smem)
        return (state.m, state.total), (state.acc,)

    def inference(walk):
        # An inference walk through `walk(step, start, shapes, initial=)` (`_walk_of`): a guarded walk's whole
        # state its carry, a plain one's sums the accumulator's.
        if guard is not None:
            return walk(step, start, ())[0], None
        rows, (acc,) = walk(step, (start.m, start.total), ((tiles.block_q, value_width),),
                            initial=(start.acc,))
        return algebra.Partial(acc, *rows), None

    deferred = (tiles.deferred and guard is None and residuals is None and certified is None
                and codes is None)

    def walked(streams, lo, hi, counted):
        if deferred:
            return _deferred_walk(spec, family, tiles, q, start, scored, dot, turn, streams, (lo, hi),
                                  counted, value_width=value_width, dtype=views["values"][0].dtype,
                                  interpret=interpret)
        if residuals is None:
            return inference(_walk_of(family, streams, lo, hi, counted, interpret=interpret))
        assert anchors is not None
        known = algebra.Argmax(family.rows_of(anchors[0].at[state_rows]),
                               family.rows_of(anchors[1].at[state_rows]))
        return _trained(spec, family, start, known, scored, values, (lo, hi), streams, counted,
                        interpret=interpret)

    # The slots its partner reads around its own, the turns it would have taken (`Family.walk_own`): a
    # deferred step takes one, a plain step two.
    state, trained = family.walk_own(streams, views, (lo, hi), streamed, consumer, walked,
                                     between=turn if deferred else lambda: (turn(), turn()))
    if merged:
        state = algebra.merge(spec, _initial(family, spec, hooked, state_rows, live), state)
    keep = None if kept is None else _kept_of(program, kept, rows=(row, head, first))
    if residuals is not None:
        # The split sums as the walk left them, shift 0 (`algebra.split_residuals`' plain walk).
        family.stored((_kept_rows(family, keep, state.acc, residuals.others),), (residuals.others,),
                      smem=out_smem)
    pl.when(consumer == 0)(turn)
    _status_stored(family, spec, state, status_views, row_smem)
    _stored_walk(program, spec, state, trained, hooked, (row, head, first), hooks=hooks, residuals=residuals,
                 out=(out_view, out_smem), rows=(finished, row_smem), keep=keep,
                 merge=_merge_of(program, family, hooked, merging, state_rows))


def _written_views(block: _Block, written: tuple, stages: tuple, *, training: bool, finishes: bool,
                   columns_at, certified: bool, merges: tuple | None) -> tuple:
    """A consumer's views of its launch's outputs `written` (`_launched`'s), its `stages` its scratch:
    a training launch's status and residuals; an inference launch's status, output (its block of the
    columns, `columns_at`) and finished rows, a merging launch's merge (`_merging_of`, `merges` its
    semaphores, place and slices) and a certified launch's certificate. None for each it does not write."""
    out_view = finished = residuals = certify = merging = None
    if training:
        status_ref, rows_ref, keys_ref, others_ref = written
        residuals = _Residuals(tuple(rows_ref.at[place, *block.state] for place in range(5)),
                               keys_ref.at[block.state], others_ref.at[block.state], stages[0])
        return block.status(status_ref), out_view, finished, residuals, certify, merging
    if merges is not None:
        status, out_view, finished, merging = _merging_of(block, written, finishes, columns_at, *merges)
    else:
        status, out_view, finished = block.inference(written, finishes)
        if certified:
            certify = (*block.certified(written), stages[-1])
    if columns_at is not None:
        out_view = out_view.at[:, columns_at]
    return status, out_view, finished, residuals, certify, merging


def _merging_of(block: _Block, written: tuple, finishes: bool, columns_at, ready, place: tuple, splits: int
                ) -> tuple:
    """A consumer's views in a launch that merges its slices (`_Block.merging`), a block of the columns'
    slots at `columns_at` where the launch walks one, and its semaphores, a slice each (`_Merge`): at
    `place`, its batch row, the grid's head (a head's block of the columns) and the consumer."""
    status, out_view, finished, slots = block.merging(written, finishes)
    if columns_at is not None:
        slots = slots._replace(acc=slots.acc.at[:, :, columns_at])
    row, axis, index = place
    tile = lax.div(lax.axis_index("q_tiles"), splits)
    return status, out_view, finished, (slots, None if ready is None else ready.at[row, axis, tile, index])


class _Merge(NamedTuple):
    """A split launch's merge of its slices (`_launch`'s `merges`) at a consumer's rows: every slice's
    state's slots (`algebra.Partial` of views, `_Block.merging`), its semaphores a slice (None interpreted,
    `Family.partials_ready`), this program's slice, the slices, and the call's initial state's read (as the
    call gives it, `_initial_rows`; None without one), which the last slice alone makes. Every slice but the
    last leaves its state in its slot and signals; the last merges them in turn and the initial state
    (`algebra.initial_state`), finishes and stores the rows' output (`Family.fixed_up`, the algebra's merge
    its combine): one launch, no merge in XLA after it."""
    slots: algebra.Partial
    ready: Any
    split: jax.Array
    splits: int
    initial: Callable[[], algebra.Partial] | None


def _initial_rows(family: mosaic.Family, hooked: _Hooked, rows: tuple) -> algebra.Partial:
    """The call's initial state at a consumer's `rows` as the call gives it (its maximum natural, its sums
    unnormalized), in the family's layouts."""
    assert hooked.initial is not None
    m0, total0, acc0 = hooked.initial
    return algebra.Partial(family.tile_of(acc0.at[rows]), family.rows_of(m0.at[rows]),
                           family.rows_of(total0.at[rows]))


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
                   scale: float, natural: bool, tiles: Tiles, interpret: bool, subtracted: bool = False):
    """The forward's certify form (off in production; row 2084's native step witness): its plain launch at
    `tiles` alone, no guard's rewalk, whose walk stores its online-softmax state after each key tile j
    (`algebra.Partial`: the rows' running maxima and denominators in the walk's units, the sums before the
    finish's divide). Returns the launch's output `[B, Q, H, Dv]` (the production launch's to the bit, a row
    its guards keep), the maxima and denominators `[2, B, H, tiles, Q]` and the sums `[B, H, tiles, Q, Dv]`
    fp32, a query block's entries past the key tiles it walks unwritten (a causal call's past its diagonal:
    `_Program.reach`). The logits are in the walk's log2 units (`algebra.exponent`: the scale folded, or the
    raw products scaled after the subtraction where `subtracted`), their exponentials `ex2.approx` where the
    launch approximates (`approximate`). A call of one block of the value columns (`Tiles.columns` None), one
    slice, no output hook."""
    if tiles.columns is not None or hooks.output_mod is not None:
        raise ValueError("the certify form walks one block of the value columns, no output hook")
    family = bundle_of(tiles, query.dtype)
    call, spec, (raw, k, v), (spans_of, hooked), taken = _prepared(
        query, key, value, spans, hooks, scale=scale, natural=natural, tiles=tiles, family=family,
        inference=True, subtracted=subtracted)
    q, form = _rewalked(spec, spec, raw, taken)
    at, at_spans = _form_tiles(spec, call, form, tiles, spans_of)
    _, out, rows, sums = _launch(q, k, v, at_spans, hooked, None, spec=form, hooks=hooks, tiles=at,
                                 family=family, training=False, interpret=interpret, certified=True)
    queries, width = call.queries, value.shape[3]
    return out[:, :queries, :, :width], rows[..., :queries], sums[:, :, :, :queries, :width]


def _deferred_walk(spec: algebra.Forward, family: mosaic.Family, tiles: Tiles, q, start: algebra.Partial,
                   scored: Callable, dot: Callable, turn: Callable, streams, bounds: tuple, counted, *,
                   value_width: int, dtype, interpret: bool) -> tuple:
    """A consumer's plain inference walk of the key tiles `bounds` with each tile's P·V issued a step late
    (`Tiles.deferred`, `values_deferred`), from `start`: FlashAttention 3's step, the tile's `Q Kᵀ` issued
    fresh into the walk's own accumulator, then the step before's P·V, its weights carried and its value
    slot held past it (`Family.walk`'s `held_past`); the tensor cores the partner's once both are issued
    (`turn`); the logits read once their product is done (`scored`'s `product`), the P·V running on beside
    that wait; the step before's P·V settled before the step's softmax, whose weights take registers the
    running P·V reads (written under it, ptxas serialized the products, C7513), so the softmax is not yet
    under it. The sums' operations and their order are the plain step's: the step before's product added,
    then the factor. Its first step issues no P·V
    before it; `finish` issues the last one's. Returns the walk's state and no residuals (`_consumer`'s
    `walked`)."""
    lo, hi = bounds

    def deferred_step(index, slots, carry, accumulators, *, issues_before: bool):
        m, total, weights = carry
        acc, logits_acc = accumulators

        def product():
            nonlocal acc, logits_acc
            logits_acc = family.products(logits_acc, q, slots["keys"].transpose((1, 0)), interpret,
                                         accumulate=False)
            if issues_before:
                acc = issued_into(family, acc, (weights,), slots.previous("values"), interpret)
            turn()
            return family.read(logits_acc, pending=1 if issues_before else 0)

        _, coords, logits, products = scored(index, slots, product)
        # The step before's P·V done before the softmax, in either form's order of its arithmetic.
        family.settled()
        taken = {}

        def accumulate(sums, scale, parts, value):
            # The tile's weights taken for the next step's product, the sums rescaled
            # (`Family.rescaled_products` of no parts).
            del value
            (taken["weights"],) = parts
            return family.rescaled_products(sums, scale, (), None, interpret)

        tile = algebra.Operand((tiles.block_k, value_width), dtype, lambda: slots["values"])
        keys = algebra.Keys(coords, logits, tile, dot, accumulate, products)
        state = algebra.plain_step(spec, algebra.Partial(acc, m, total), keys)
        return (state.m, state.total, taken["weights"]), (state.acc, logits_acc)

    def finish(carry, accumulators, kept):
        acc, logits_acc = accumulators
        return carry, (issued_into(family, acc, (carry[2],), kept["values"], interpret), logits_acc)

    segments = ((functools.partial(deferred_step, issues_before=False), lo, lo + 1),
                (functools.partial(deferred_step, issues_before=True), lo + 1, hi))
    none = family.zeros((tiles.block_q, tiles.block_k)).astype(dtype)
    (m, total, _), (acc, _) = family.walk(
        streams, segments, (start.m, start.total, none),
        ((tiles.block_q, value_width), (tiles.block_q, tiles.block_k)), interpret, first=counted,
        initial=(start.acc, None), held_past=("values",), finish=finish)
    return algebra.Partial(acc, m, total), None



def _stored_walk(program: _Program, spec: algebra.Forward, state: algebra.Partial, trained: tuple | None,
                 hooked: _Hooked, place: tuple, *, hooks: Hooks, residuals: _Residuals | None, out: tuple,
                 rows: tuple, keep: jax.Array | None = None, merge: _Merge | None = None) -> None:
    """A consumer's walk's result stored (`_consumer`): its output under the output hook through `out` (its
    view and stage), its finished rows through `rows` (their views, None where the launch finishes none, and
    their stage); a training walk's (`trained`, its anchor's terms) its residual rows and anchor keys
    instead, no output. `place` its batch row, head and first query; with `keep`, each written at those rows
    alone, every other row of its view as it was (`_kept_rows`). With `merge`, the output the slices'
    merge's, which the last slice's program stores (`_Merge`)."""
    family = program.family
    row, head, first = place
    (out_view, out_smem), (finished, row_smem) = out, rows
    if residuals is None or trained is None:
        out_params = tuple(_param(family, ref, param.kind, row, head, first, state.acc)
                           for ref, param in zip(hooked.output, hooks.output_params, strict=True))

        def store(state: algebra.Partial, form: algebra.Forward, *, initial: bool, normalized: bool = False
                  ) -> None:
            output, maximum, denominator = algebra.finish(form, state, initial=initial, normalized=normalized,
                                                          output_mod=hooks.output_mod,
                                                          output_params=out_params)
            family.stored((_kept_rows(family, keep, output, out_view),), (out_view,), smem=out_smem)
            if finished is not None:
                # Its rows as the backward reads them beside the output (`Output`): the maximum and
                # denominator `finish` leaves.
                family.stored(tuple(_kept_rows(family, keep, family.as_row(part), view)
                                    for part, view in zip((maximum, denominator), finished, strict=True)),
                              finished, smem=row_smem)

        if merge is None:
            store(state, spec, initial=hooked.initial is not None)
            return
        # A slice's state finished (a short row's slice may see no key) and folded (`algebra.slice_state`),
        # in the units its finished rows are in (`algebra.finished_form`: a fold's natural ones); each slot a
        # slice from the last, which merges the others' in turn, then the initial state's, and finishes
        # once: no row's output multiplied back by its denominator.
        form = algebra.finished_form(spec)
        own = algebra.slice_state(form, *algebra.finish(spec, state, initial=True))
        slot = merge.splits - 1 - merge.split

        def merged(running: algebra.Partial) -> None:
            if merge.initial is not None:
                running = algebra.finished_merge(form, running, algebra.initial_state(form, merge.initial()))
            store(running, form, initial=True, normalized=True)

        family.fixed_up(own, merge.slots, merge.ready, slot, slot, merge.splits, merged,
                        combine=functools.partial(algebra.finished_merge, form))
        return
    maximum, denominator = algebra.finished_rows(spec, state, initial=hooked.initial is not None)
    own, others_total, key = trained
    parts = (maximum, denominator, own, others_total, family.zeros((program.tiles.block_q,)))
    family.stored(tuple(_kept_rows(family, keep, family.as_row(part), view)
                        for part, view in zip(parts, residuals.rows, strict=True)), residuals.rows,
                  smem=row_smem)
    family.stored((_kept_rows(family, keep, family.as_row(key), residuals.keys),), (residuals.keys,),
                  smem=residuals.keys_smem)


def _walk_of(family: mosaic.Family, streams, lo, hi, counted, *, interpret: bool) -> Callable:
    """A consumer's walk of its key tiles `lo` to `hi` over its program's `streams` from the program's
    `counted`-th step (`Family.walk`), as `walk(step, start, shapes, initial=)` takes it."""
    def walk(step, start, shapes, initial=None):
        return family.walk(streams, ((step, lo, hi),), start, shapes, interpret, first=counted,
                           initial=initial)

    return walk


def _started(program: _Program, spec: algebra.Forward, guard: _Guard | None, hooked: _Hooked, rows: tuple,
             first, value_width: int, *, training: bool, merges: bool = False
             ) -> tuple[algebra.Partial, jax.Array | None, bool]:
    """A consumer's walk's start at its `rows` (batch row, head, queries from `first`): its state, its
    guard's live rows (None unguarded) and whether the initial state merges past the walk instead
    (`algebra.merge`). A plain inference walk starts empty and merges it: read before the walk, the state's
    rows held live across it beside its products and the walk spilled 296 bytes at D128's 64 x 128 (the
    held-term census), where an empty start's are constants."""
    family, block_q = program.family, program.tiles.block_q
    row, head, _ = rows
    live = None
    if guard is not None:
        # A split launch's live rows its slice's (`_Program.splits`), slice by slice along their query axis.
        live = _sliced_any(program, guard.live, (row, head, first))
    # A launch that merges its slices takes the initial state into their merge alone (`_merge`), no slice's
    # walk.
    merged = hooked.initial is not None and guard is None and not training and not merges
    start = (_initial(family, spec, hooked, rows, live) if hooked.initial is not None and not merged
             and not merges else _empty(family, (block_q, value_width), live))
    return start, live, merged


def _parted_products(family: mosaic.Family, acc, scale, weights: jax.Array, values: operands.Parts,
                     interpret: bool, *, shape: tuple[int, int] | None = None):
    """`acc` with an fp32 call's P·V issued into it in place, its factor `scale` first where given, or a
    fresh accumulator of `shape` where `acc` is None: the fp32 `weights`' bf16 parts (`product.bf16_parts`'),
    the largest first, each product with the values' parts (`values`) the pair's orders under their count
    (`product.terms`'), all into the one accumulator (`issued_pairs`). Each part formed from the weights
    alone as it is taken (`_part`), so a consumer holds the weights and one part beside the sums."""
    count = len(values.refs)

    def pairs():
        for i in range(count):
            part = _part(weights, i)
            for j in range(count - i):
                yield part, values.refs[j]

    if acc is None:
        return issued_pairs(family, None, pairs(), interpret, shape=shape)
    if scale is not None:
        acc = family.rescaled_products(acc, scale, (), None, interpret)
    return issued_pairs(family, acc, pairs(), interpret)


def _part(weights: jax.Array, index: int) -> jax.Array:
    """The `index`-th bf16 part of fp32 `weights` (`product.bf16_parts`'), each element's from its own weight:
    its remainder past the parts before it, rounded. A weight is at most 1, no part of it saturating."""
    rest = weights
    for _ in range(index):
        rest = rest - rest.astype(jnp.bfloat16).astype(jnp.float32)
    return rest.astype(jnp.bfloat16)


def _accumulated(family: mosaic.Family, interpret: bool) -> Callable:
    """`algebra.Keys.accumulate` on the family's accumulator: the tile's products issued into it in place,
    after its factor where the step gives one (`Family.rescaled_products`)."""
    def accumulate(acc, scale, parts, value):
        if isinstance(value, operands.Parts):
            (weights,) = parts
            return _parted_products(family, acc, scale, weights, value, interpret)
        if scale is None:
            return issued_into(family, acc, parts, value, interpret)
        return family.rescaled_products(acc, scale, parts, value, interpret)

    return accumulate


def _trained(spec: algebra.Forward, family: mosaic.Family, start: algebra.Partial, known: algebra.Argmax,
             scored, values, reach, streams, first, *, interpret: bool
             ) -> tuple[algebra.Partial, tuple]:
    """A training forward consumer's walk of its key tiles `reach` (`[lo, hi)`, the streams' steps from
    `first`) from `start` at the anchors and keys' largest logits `known` (`algebra.anchored`): each tile
    `algebra.anchored_step`, its split sums the family's accumulator, which takes each tile's product in place
    and no factor. `scored(index, slots)` gives a tile's place, coordinates, logits and products, and
    `values(slots)` its value slot as an operand (`algebra.Operand`). Returns the state with the split sums
    as its sums, and what the walk leaves beside them: the anchor's own weight, the other keys' weights' sum
    and the anchor keys."""
    state, arg, own, others_total = algebra.anchored(spec, start, known)

    def dot(lhs, rhs):
        return family.registers_product(lhs, rhs, interpret)

    def step(index, slots, carry, accumulators):
        total, own, others_total = carry
        _, coords, logits, products = scored(index, slots)
        keys = algebra.Keys(coords, logits, values(slots), dot, _accumulated(family, interpret), products)
        # The sums are the accumulator's, which the step takes the tile into (`Keys.accumulate`).
        walked = algebra.Partial(accumulators[0], state.m, total)
        walked, _, own, others_total = algebra.anchored_step(spec, (walked, arg, own, others_total), keys)
        return (walked.total, own, others_total), (walked.acc,)

    (total, own, others_total), (acc,) = family.walk(
        streams, ((step, *reach),), (state.total, own, others_total), (state.acc.shape,), interpret,
        first=first, initial=(state.acc,))
    return algebra.Partial(acc, state.m, total), (own, others_total, arg.key)


def _peak_consumer(program: _Program, spec: algebra.Forward, consumer, q_view, peak_views: tuple, q_smem,
                   q_loaded, row_smem, keys_smem, streams, views: dict, turn, streamed: tuple, first,
                   spans: _Spans | None, hooked: _Hooked, place: tuple, *, hooks: Hooks,
                   interpret: bool) -> None:
    """A peak walk's query block (batch row and head `place`): each row's keys' largest logit, in the walk's
    units, and the first key that holds it (`algebra.argmax`, the training walk's anchor, `algebra.anchored`),
    over the key tiles it reads through the family's key stream, the logits under the score hook as the
    training walk forms them (`_scored`), stored through the row stages to
    `peak_views`. Its turns on the tensor cores and the slots its partner alone reads are the training walk's
    own (`_consumer`)."""
    family, tiles = program.family, program.tiles
    lo, hi, whole = program.reach(first, spans, place[0])
    family.loaded(((q_view, q_smem, q_loaded),))
    q = family.held(q_smem)
    pl.when(consumer == 1)(turn)
    held = None if spans is None else program.bounds(first, spans, place[0])

    def step(index, slots, arg, accumulators):
        # The peak walk's logits draw no dropout.
        _, coords, logits, _ = _scored(program, spec, q, index, slots, first, spans, hooked, place, whole,
                                       turn, hooks=hooks, drawn=False, held=held, interpret=interpret)
        # The partner's turn, as a training step's value product would take it.
        turn(wait=False)
        turn(arrive=False)
        return algebra.argmax(arg, logits, coords.k_pos), accumulators

    start = algebra.Argmax(family.full((tiles.block_q,), -jnp.inf),
                           family.full((tiles.block_q,), 0, jnp.int32))
    arg, _ = family.walk_own(
        streams, views, (lo, hi), streamed, consumer,
        lambda streams, lo, hi, counted: family.walk(streams, ((step, lo, hi),), start, (), interpret,
                                                     first=counted),
        between=lambda: (turn(), turn()))
    pl.when(consumer == 0)(turn)
    family.stored((family.as_row(arg.peak),), (peak_views[0],), smem=row_smem)
    family.stored((family.as_row(arg.key),), (peak_views[1],), smem=keys_smem)


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
        # A paged tile's pages each at the column block (`mosaic.Rows`).
        indices = tile(j)
        if isinstance(indices, mosaic.Rows):
            return dataclasses.replace(indices, rest=(block,))
        return indices[0], block
    return {**views, "values": (ref, at_columns)}


def _paged_views(refs: dict, table, row, head, page: int, block_k: int) -> dict:
    """A paged call's streams' views (`_views_of`): each pool `[KV, P·page, D]`'s `[P·page, D]` at key head
    `head`, step `j` its key tile `j` where row `row`'s table maps it: of a page of several tiles, the
    tile's page `table[row, j // per]` and its place in it `j % per` (`per` tiles a page); of a tile of
    several pages (`count` a tile), each page's rows from its own (`mosaic.Rows`, a run a page)."""
    per, count = max(1, page // block_k), max(1, block_k // page)

    def tile(j):
        if count == 1:
            return (table[row, lax.div(j, per)] * per + lax.rem(j, per), 0)
        return mosaic.Rows(table.at[row], j * count, (0,), page)

    return {name: (ref.at[head], tile) for name, ref in refs.items()}


def _launch(query, key, value, spans: _Spans | None, hooked: _Hooked, guard: _Guard | None, *,
            spec: algebra.Forward, hooks: Hooks, tiles: Tiles, family: mosaic.Family, training: bool,
            interpret: bool, columns: int | None = None, peaks: bool = False,
            anchors: tuple[jax.Array, jax.Array] | None = None, out_dtype=None, counted: bool = False,
            kept: bool = False, finishes: bool = False, splits: int = 1,
            table: tuple[jax.Array, int] | None = None, column_width: int | None = None,
            walks: "_Flags | None" = None, sources: jax.Array | None = None, shared_by: int = 1,
            column_blocks: int = 1, certified: bool = False, into: tuple | None = None,
            keeping: jax.Array | None = None, merges: bool = False, flags: bool = False, parts: int = 1):
    """The forward's launch in `spec`'s form (`guard`'s rows shifting where given): its rows' status `[2, B,
    H, Qp]` fp32, their maxima in the walk's units and 1 where their weighted values overflowed
    (`algebra.walked_launches`), and its output `[B, Qp, H, Dv]` in `out_dtype` (the query's by default). In
    `training`, at the
    `anchors` a peak walk found, its status and residuals (`_Residuals`) and no output: the rows
    `[5, B, H, Qp]` fp32 (maxima, denominators, the anchor's own weight, the other keys' weights' sum and
    their shift), the anchor keys `[B, H, Qp]` int32 and the split sums `[B, H, Qp, Dv]` fp32, in `Split`'s
    layout, each block's rows one contiguous slab. A peak walk
    (`peaks`) reads the keys alone and gives each row's keys' largest logit `[B, H, Qp]` fp32 in the walk's
    units and its anchor key `[B, H, Qp]` int32 (`_peak_consumer`). With `finishes`, the output's launch also
    gives its rows' finished maxima and denominators `[2, B, H, Qp]` fp32 (`Output`, `algebra.finish`).
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
    unread. With `column_blocks`, a launch walks that many blocks of `column_width` columns from the first,
    a program a block (its heads' grid axis the heads by the blocks), its output every block's columns; its
    per-row outputs each block's alike, as its logits are the row's whole. With `certified`, the certify
    form's (`certified_walk`): its plain walk's state after each key tile beside its outputs. With `into`
    (refs holding the plain launch's outputs but its status, `_Read.into`), it writes its rows `keeping` into
    them in place and gives its status alone: a masked rewalk's rows, every other row the plain launch's bits,
    no copy of the outputs and no select over them (`algebra.walked_launches`' `in_place`). A split launch
    that `merges` its slices gives their merge, the last slice's program's, the initial state's among them
    (`_Merge`), as its output `[B, Qp, H, Dv]` and finished rows `[2, B, H, Qp]`, its status the slices':
    legal where every program of its grid is resident at once (`MosaicForward.legal`), as the merge's waits
    need.
    """
    if splits > 1 and (training or peaks or (hooked.initial is not None and not merges)):
        raise ValueError("a split launch walks an inference call, an initial state only into its merge")
    # Each launch named apart, so its compile reports and profile rows say which it is; a launch over a block
    # of the value columns by the block's width, every block's one kernel (the name is a static of its
    # lowering, `_launched`: named by its first column, a width 256 compiled four, train 505's review).
    kind = "peaks" if peaks else "training" if training else "forward"
    block_width = column_width or value.shape[-1]
    name = (f"attention_{kind}{'_natural' if spec.natural else ''}{'_kept' if kept else ''}"
            f"{'_guarded' if guard is not None else ''}{'' if columns is None else f'_columns{block_width}'}"
            f"{'_certified' if certified else ''}{'_merged' if merges else ''}")
    # The hooks' functions and their arrays' shapes and dtypes static, their arrays the programs' operands.
    # A block of the values' columns its index a scalar the programs read, so every block's launch is one
    # kernel (`_key_views`): a static index compiled a kernel a block, four at width 256 (the forward's
    # resource gate's cases 50 to 150 s on a CPU).
    block = (None if column_width is None or column_blocks > 1
             else jnp.full((1,), (columns or 0) // column_width, jnp.int32))
    target = generation.gpu_target()
    return _launched(query, key, value,
                     _Read(spans, hooked, guard, anchors, None if table is None else table[0],
                           None if walks is None else walks.blocks, sources, block, into, keeping),
                     hooks=jax.tree.map(lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype), hooks),
                     spec=spec, tiles=tiles, family=family, training=training, interpret=interpret,
                     peaks=peaks, out_dtype=None if out_dtype is None else jnp.dtype(out_dtype),
                     counted=counted, finishes=finishes, target=target,
                     name=name if splits == 1 else f"{name}_split", splits=splits,
                     page=None if table is None else table[1],
                     column_width=column_width, shared_by=shared_by, column_blocks=column_blocks,
                     certified=certified, merges=merges and splits > 1, flags=flags,
                     walk_rows=None if walks is None else walks.granule, parts=parts)


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
    from each part's batch rows (`mosaic.Pieces`, `MosaicForward.parts`)."""
    if parts == 1:
        return q.at[row, rows, head]
    return mosaic.Pieces(q.at[part * batch + row, rows, head] for part in range(parts))

def _parted_views(views: dict, named: dict, row, kv_head, *, batch: int, parts: int) -> dict:
    """An fp32 call's stream views (`MosaicForward.parts`): each tile its parts' rows, a copy each from its
    part's batch rows (`mosaic.viewed`'s pieces of refs), at the views' own block indices."""
    return {name: (mosaic.Pieces(ref.at[part * batch + row, :, kv_head] for part in range(parts)), index)
            for (name, (_, index)), ref in zip(views.items(), named.values(), strict=True)}

def _entry(program: _Program, layout: "_Layout", body: Callable, allocations: tuple, *refs,
           column_blocks: int, queries: int, walk_rows: int | None) -> None:
    """A forward launch's kernel entry (`_launched`'s): its `body` in its scoped allocations and rings, where
    a masked launch walks it alone (`_walked_program`: else its program loads and writes nothing; its
    `walk_rows` the walks' granule, which a masked launch states)."""
    scoped, rings = allocations

    def scoped_body():
        program.family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    walks = layout.read(refs).walks
    if walks is None:
        scoped_body()
    else:
        assert walk_rows is not None, "a masked launch states its walks' granule"
        head = lax.div(lax.axis_index("heads"), column_blocks)
        pl.when(_walked_program(program, walks, lax.axis_index("batch"), head, queries, walk_rows))(
            scoped_body)

@tuning.lowered_once(("hooks", "spec", "tiles", "family", "training", "interpret", "peaks", "out_dtype",
                      "counted", "finishes", "target", "name", "splits", "page", "column_width", "shared_by",
                      "column_blocks", "certified", "merges", "flags", "walk_rows", "parts"))
def _launched(query, key, value, read: _Read, *, hooks: Hooks, spec: algebra.Forward, tiles: Tiles,
              family: mosaic.Family, training: bool, interpret: bool,
              peaks: bool, out_dtype, counted: bool, finishes: bool, target: generation.GpuTarget,
              name: str, splits: int = 1, page: int | None = None,
              column_width: int | None = None, shared_by: int = 1, column_blocks: int = 1,
              certified: bool = False, merges: bool = False, flags: bool = False,
              walk_rows: int | None = None, parts: int = 1):
    """`_launch`'s kernel launched on its operands and the arrays its programs read beside them (`read`, the
    hooks' among them, `_Hooked`), its static `hooks` holding their arrays' shapes and dtypes in their place
    (its functions a key's, which `tuning.holdable` reads), lowered once for every call of its static facts
    and its operands' shapes and dtypes (`tuning.lowered_once`); `target` the GPU its config is derived for,
    `name` its kernel's (`mosaic.kernel` reports it as the launch tracing it names it)."""
    leaves, treedef = jax.tree.flatten(read)
    batch, queries, heads, width, value_width, keys_count, kv_heads = _extents(
        query, key, value, read, page=page, column_width=column_width, parts=parts)
    group, consumers = heads // kv_heads, tiles.consumers
    block_q, block_k = tiles.block_q, tiles.block_k
    approx_math = approximate(hooks, (block_q, block_k), value_width, differentiated=False)
    spec = dataclasses.replace(spec, approximate=approx_math)
    pair = consumers * block_q
    steps = keys_count // block_k
    # No more stages than key tiles: a slot past them would load past the keys.
    program = _Program(family, tiles, steps, min(tiles.stages, steps), spec.keys, spec.form == "causal",
                       spec.queries, splits, _per_span(spec, read.spans), parts=parts)
    # The output stores through the query's stage where the two are as wide (its rows are read before the
    # store); else it takes a stage of its own beside the query's and the rings (`_forward_allocations`).
    out_dtype = query.dtype if out_dtype is None else jnp.dtype(out_dtype)
    reuse = _stores_through_query(family, width, value_width, query.dtype, out_dtype)
    # A paged call's key tile of several pages loaded a page at a time (`_paged_views`).
    pieces = block_k // page if page is not None and page < block_k else 1
    scoped, rings = _forward_allocations(family, tiles, width, value_width, query.dtype, training,
                                         program.stages, peaks=peaks, out_dtype=out_dtype, pieces=pieces,
                                         certified=certified, codes=_codes_of(key, read.hooked.scales),
                                         parts=parts)
    out_type = _out_types(batch, queries, heads, value_width * column_blocks, out_dtype, splits=splits,
                          peaks=peaks, training=training, finishes=finishes,
                          certified=steps if certified else 0, merges=merges, flags=flags)
    # A launch into the plain launch's outputs (`_Read.into`) gives its status alone, the rest in place, but
    # a merging launch's slices' states, its own.
    in_place = read.into is not None
    if in_place:
        out_type = (out_type[0], *out_type[-2:]) if merges else out_type[:1]
    layout = _Layout(treedef, len(out_type), len(scoped) - 2, peaks or (reuse and not training),
                     relays(family, tiles, peaks=peaks), _codes_of(key, read.hooked.scales) is not None)

    def body(*refs, streams):
        refs = layout.unpacked(*refs)
        read = refs.read
        row, axis = lax.axis_index("batch"), lax.axis_index("heads")
        # A launch of several column blocks a program a block along its heads' axis.
        if column_blocks > 1:
            head, column = lax.div(axis, column_blocks), lax.rem(axis, column_blocks)
        else:
            head, column = axis, None if read.block is None else read.block[0]
        named = {"keys": refs.k} if peaks else {"keys": refs.k, "values": refs.v}
        views = _key_views(named, row, lax.div(head, group), read.sources, read.table, page=page,
                           block_k=block_k, column_block=column)
        if parts > 1:
            views = _parted_views(views, named, row, lax.div(head, group), batch=batch, parts=parts)
        streams = family.opened(streams, views)
        pair_first = program.query_tile(queries // pair) * pair
        streamed = _streamed([program.reach(pair_first + index * block_q, read.spans, row)[:2]
                              for index in range(consumers)])
        turn = functools.partial(family.turn, refs.schedule, consumers)
        # A merging launch's semaphores, a slice each of a consumer's query block (`_Merge`).
        ready = (family.partials_ready((batch, heads * column_blocks, queries // pair, consumers, splits),
                                       bool(interpret)) if merges else None)

        def consumer(index):
            block = _Block.of(program, row, head, pair_first + index * block_q, queries)
            stages = (refs.q_smem.at[index], _query_barriers(refs.q_loaded, index, parts))
            if peaks:
                _peak_consumer(program, spec, index, refs.q.at[block.rows],
                               tuple(ref.at[block.state] for ref in refs.written), *stages,
                               refs.row_smem.at[index], refs.stages[0].at[index], streams, views, turn,
                               streamed, block.first, read.spans, read.hooked, (row, head), hooks=hooks,
                               interpret=interpret)
                return
            written = (refs.written[0], *(read.into or ()), *refs.written[1:]) if in_place else refs.written
            # A block of the columns writes its own of the output's, the whole call's in place.
            columns_at = (pl.ds(column * value_width, value_width)
                          if column is not None and (column_blocks > 1 or in_place) else None)
            status, out_view, finished, residuals, certify, merging = _written_views(
                block, written, tuple(stage.at[index] for stage in refs.stages), training=training,
                finishes=finishes, columns_at=columns_at, certified=certified,
                merges=(ready, (row, axis, index), splits) if merges else None)
            q_view = _query_view(refs.q, row, pl.ds(block.first, block_q), head, batch=batch, parts=parts)
            _consumer(program, spec, index, q_view, out_view,
                      status, *stages, refs.out_smem.at[index], refs.row_smem.at[index], streams, views, turn,
                      streamed, block.first, read.spans, read.hooked, read.guard,
                      (row, head, lax.div(head, group)), hooks=hooks, residuals=residuals,
                      anchors=read.anchors, interpret=interpret, finished=finished, certified=certify,
                      kept=read.kept, merging=merging,
                      relay=None if refs.relay is None else refs.relay.at[index],
                      codes=None if refs.converted is None or read.hooked.scales is None else _Codes.of(
                          family, tuple(tile.at[index] for tile in refs.converted), read.hooked.scales,
                          row if read.sources is None else read.sources[row], lax.div(head, group),
                          tiles.block_k, interpret))

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    entry = functools.partial(_entry, program, layout, body, (scoped, rings), column_blocks=column_blocks,
                              queries=queries, walk_rows=walk_rows)
    cost, model = _launch_flops(spec, family, query, key, value, keys_count, value_width, tiles,
                                peaks=peaks, training=training, counted=counted, shared_by=shared_by)
    grid = (heads * column_blocks, queries // pair * splits, batch)
    # A merging launch's every program resident at once (`MosaicForward.legal`), its registers capped so
    # that many fit a multiprocessor's (`Family.kernel`'s `resident`).
    resident = (-(-math.prod(grid) // target.multiprocessors) if merges and target.multiprocessors else 1)
    return family.kernel(entry, out_type=out_type, grid=grid,
                         grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
                         interpret=interpret, resident=resident,
                         approx_math=approx_math, name=name, cost=cost, model_flops=model,
                         reducing_warpgroups=reducing(family, consumers, hooks.score_mod, hooks.output_mod,
                                                      guarded=read.guard is not None or flags)
                         )(query, key, value, *leaves)


def _launch_flops(spec: algebra.Forward, family: mosaic.Family, query, key, value, keys_count: int,
                  value_width: int, tiles: Tiles, *, peaks: bool, training: bool, counted: bool,
                  shared_by: int) -> tuple:
    """A launch's cost and the model's FLOPs it owns (`_launched`'s `cost` and `model_flops`): the products it
    issues (`stated`), and the model's of them it owns (`modelled`): the forward's two, a peak walk's
    `Q K^T`, a training walk's `P V` beside its `Q K^T` recomputed, a block of an inference call's columns
    its `P V` and its share of `Q K^T` (`shared_by`); a rerun (a guarded launch, the fold's natural walk)
    none, as the FLOP count bounds a conditional by every branch, which would count the call's work twice
    every step where it reruns at most rarely."""
    batch, queries, heads, width = query.shape
    columns = width if peaks else width + value_width
    owned_value = 0 if peaks else value_width
    owned_query = 0 if training and not peaks else width
    cost = (stated(spec, batch, heads, columns, query, key, value,
                   tile_walk=TileWalk((queries, keys_count), (tiles.block_q, tiles.block_k), tiles.consumers,
                                      common=family.feeds_itself)) if counted
            else estimate(0))
    query_share = modelled(spec, batch, heads, spec.queries or queries, owned_query)
    value_share = modelled(spec, batch, heads, spec.queries or queries, owned_value)
    model = (0 if not counted else None if query_share is None or value_share is None
             else query_share // shared_by + value_share)
    return cost, model
