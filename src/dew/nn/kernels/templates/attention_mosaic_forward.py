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
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import attention_algebra as algebra, attention_portable, mosaic, resources
from dew.nn.kernels.templates.attention import (
    Hooks,
    KeySpans,
    Param,
    Split,
    output_hooked,
    padded_call,
    prescaled,
)
from dew.nn.kernels.templates.attention_mosaic_program import (
    Tiles,
    _block,
    _columns,
    _coords,
    _Guard,
    _Hooked,
    _hooked_of,
    _param,
    _Program,
    _Spans,
    _spans_of,
    _streamed,
    _views_of,
    approximate,
    bundle_of,
    extents,
    reducing,
    refused,
    stated,
)


def _forward_allocations(family: mosaic.Family, tiles: Tiles, width: int, value_width: int, dtype,
                         residuals: bool, stages: int, *, peaks: bool = False, out_dtype=None
                         ) -> tuple[list, dict]:
    """The forward program's allocations as `Family.scoped` takes them, in `_launch`'s order: its consumers'
    query stages and their rows' status; each one's output stage where the value is not as wide as the query
    and as narrow, `out_dtype`, the query's by default (the output stores through the query's stage where it
    is), a training forward's its split sums' fp32 (`_Residuals`); a training forward's or a peak walk's
    anchor keys' stage; the query loads' barriers and
    the consumers' turns' barrier; and its key and value streams in `stages` slots (`Family.streams`), a
    peak walk's keys alone."""
    consumers, block_q, block_k = tiles.consumers, tiles.block_q, tiles.block_k
    scoped = [plgpu.SMEM((consumers, block_q, width), dtype), plgpu.SMEM((consumers, block_q), jnp.float32)]
    if residuals:
        scoped.append(plgpu.SMEM((consumers, block_q, value_width), jnp.float32))
    elif not peaks and (value_width != width or out_dtype not in (None, dtype)):
        scoped.append(plgpu.SMEM((consumers, block_q, value_width), out_dtype or dtype))
    if residuals or peaks:
        scoped.append(plgpu.SMEM((consumers, block_q), jnp.int32))
    scoped += [plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_arrivals=consumers)]
    blocks = {"keys": ((block_k, width), dtype)}
    if not peaks:
        blocks["values"] = ((block_k, value_width), dtype)
    return scoped, family.streams(blocks, stages, consumers)


def guarded_columns(family: mosaic.Family, value_width: int, dtype) -> int:
    """The value columns a guarded launch walks at once (`_plain_forward`): a block of the family's swizzled
    columns (`Family.column_unit`)."""
    return min(value_width, family.column_unit(dtype))


GUARD_ROWS = 4
"""The row statistics a guarded walk carries beside a plain one's maximum and denominator (`algebra.Guard`:
each row's shift, reach, unshifted denominator and liveness), each held by the four threads an accumulator
row spans, two rows a thread: a `[block_q, 4]` fp32 tile's words a statistic (`guarded_tiles`)."""


def guarded_tiles(family: mosaic.Family, tiles: Tiles, columns: int, dtype) -> Tiles | None:
    """The guarded launch's tiles beside the plain launch's `tiles` (`_plain_forward`): its query blocks, the
    widest of the family's key extents (`extents`) that divides the plain key tile, so the plain launch's
    padded keys are whole tiles of it, and the most of the plain launch's consumers, whose live set a
    consumer's registers hold (`resources.live_registers`): its shifted and unshifted sums of `columns` value
    columns fp32 and its rows' statistics (`GUARD_ROWS`) carried, beside a step's logits and weights fp32
    and the two products the step forms (`algebra.stepped`). Uncounted, the statistics left 64 x 64 on two
    consumers spilling 20 bytes a thread at heads 128 and 256 wide (ptxas, sm90a). None where none fits:
    the plain launch is then illegal at `tiles` (`MosaicForward.legal`)."""
    rows = tiles.block_q
    carried = [(rows, columns, jnp.float32)] * 2 + [(rows, 4 * GUARD_ROWS, jnp.float32)]

    def held(block: int, consumers: int) -> bool:
        stepped = [(rows, block, jnp.float32)] * 2 + [(rows, columns, jnp.float32)] * 2
        return resources.live_registers(family, carried, stepped) <= family.registers(consumers)

    blocks = [block for block in extents(family, dtype, tiles.block_k) if tiles.block_k % block == 0]
    candidates = (dataclasses.replace(tiles, block_k=block, consumers=consumers)
                  for consumers in range(tiles.consumers, 0, -1) for block in blocks)
    return next((candidate for candidate in candidates if held(candidate.block_k, candidate.consumers)), None)


def _plain_forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                   scale: float, natural: bool, tiles: Tiles, interpret: bool) -> jax.Array:
    """`MosaicForward`'s launch without residuals, at `tiles`, under the default's guards over whole launches
    (`algebra.walked_launches`). The guarded launch (`_Guard`) runs a block of the family's swizzled value
    columns at a time: its walk carries its shifted and unshifted sums and forms both tiles' products a step
    (`algebra.stepped`), four `[block_q, Dv]` fp32 tiles beside its logits, which a block of the columns
    holds within a consumer's registers (ledger 1617). Each block's output is exact on its own (its rows'
    shifts its own columns' room, `algebra.room`); the rare launch takes its logits once a block. An output
    hook reads a row's whole output, so where there is one the blocks store theirs in fp32 and the hook runs
    on the rows they make in XLA (`template.output_hooked`, the reference's own), rounded once after it."""
    family = bundle_of(tiles, query.dtype)
    call, spec, (q, k, v), (spans_of, hooked) = _prepared(query, key, value, spans, hooks, scale=scale,
                                                          natural=natural, tiles=tiles, family=family)

    def launch(form: algebra.Forward, live: jax.Array | None):
        if live is None:
            status, out = _launch(q, k, v, spans_of, hooked, None, spec=form, hooks=hooks, tiles=tiles,
                                  family=family, training=False, interpret=interpret,
                                  counted=form.natural == spec.natural)
            return out, status[0], status[1] > 0
        unit = guarded_columns(family, v.shape[-1], v.dtype)
        guarded = guarded_tiles(family, tiles, unit, v.dtype)
        hooked_blocks = hooks.output_mod is not None
        # The plain launch's legality held a guarded tile at `tiles` (`MosaicForward.legal`).
        assert guarded is not None, tiles
        # A spans call's block map counts blocks of the launch's own key tile.
        guarded_spans = _spans_of(spec, call, guarded.block_q, guarded.block_k)
        blocks = []
        for first in range(0, v.shape[-1], unit):
            # The guarded launch reads its live rows and its values' largest magnitudes (`algebra.room`), and
            # the initial state's sums of its own columns.
            columns = v[..., first:first + unit]
            guard = _Guard(live.astype(jnp.int32), jnp.swapaxes(algebra.magnitudes(columns), 1, 2))
            initial = hooked.initial
            held = hooked._replace(output=()) if initial is None else hooked._replace(
                output=(), initial=(*initial[:2], initial[2][..., first:first + unit]))
            blocks.append(_launch(q, k, columns, guarded_spans, held, guard, spec=form,
                                  hooks=dataclasses.replace(hooks, output_mod=None, output_params=()),
                                  tiles=guarded, family=family, training=False, interpret=interpret,
                                  columns=first, out_dtype=jnp.float32 if hooked_blocks else None)[1])
        out = jnp.concatenate(blocks, axis=-1)
        if hooked_blocks:
            out = jnp.swapaxes(_output_hooked(jnp.swapaxes(out, 1, 2), hooks, hooked), 1, 2).astype(q.dtype)
        # Its maxima and overflows `walked_launches` reads of the plain launch alone.
        return out, jnp.zeros(live.shape, jnp.float32), jnp.zeros_like(live)

    def kept(live: jax.Array, guarded: jax.Array, plain: jax.Array) -> jax.Array:
        return jnp.where(jnp.swapaxes(live, 1, 2)[..., None], guarded, plain)

    out = algebra.walked_launches(spec, launch, kept)
    return out[:, :call.queries, :, :value.shape[3]]


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
    call, spec, (q, k, v), (spans_of, hooked) = _prepared(query, key, value, spans, hooks, scale=scale,
                                                          natural=natural, tiles=tiles, family=family)
    value_width = value.shape[3]

    def launch(form: algebra.Forward, live: jax.Array | None):
        if live is None:
            counted = form.natural == spec.natural
            found = _launch(q, k, v, spans_of, hooked, None, spec=form, hooks=hooks, tiles=tiles,
                            family=family, training=False, interpret=interpret, peaks=True, counted=counted)
            status, rows, anchors, others = _launch(q, k, v, spans_of, hooked, None, spec=form, hooks=hooks,
                                                    tiles=tiles, family=family, training=True,
                                                    interpret=interpret, anchors=found, counted=counted)
            split = Split(others, rows[2], rows[3], rows[4])
            # Each row's anchor's value row (`[B, H, Qp, Dv]`): its key head's at its key, gathered.
            batch, kv_heads, keys_count = v.shape[0], v.shape[2], v.shape[1]
            kv_of = jnp.arange(q.shape[2]) // (q.shape[2] // kv_heads)
            places = ((jnp.arange(batch)[:, None, None] * kv_heads + kv_of[None, :, None]) * keys_count
                      + anchors)
            value_rows = jnp.take(jnp.swapaxes(v, 1, 2).reshape(-1, v.shape[-1]), places, axis=0)
            acc = algebra.split_output(split.others, split.own, value_rows)
            out = algebra.output_of(acc, rows[1], guarded=form.guarded or hooks.initial is not None)
            out = _output_hooked(out, hooks, hooked)
            overflowed = (status[1] > 0) | algebra.overflowed(acc)
            return ((jnp.swapaxes(out, 1, 2).astype(q.dtype), rows[0], rows[1], anchors, split), status[0],
                    overflowed)
        # The overflowed rows' guarded training walk, the plain-JAX mainloop's over this call's padding in
        # this launch's form (the rewalk's units kept, `algebra.unfolded`); its maxima and overflows
        # `walked_launches` does not read.
        out, rows = attention_portable.padded_walk(form, call, hooks, block_q=tiles.block_q,
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


def _widened(array: jax.Array, width: int) -> jax.Array:
    """`array` with its last axis padded to `width`, as this mainloop's launch stages values (`_columns`)."""
    return jnp.pad(array, ((0, 0),) * (array.ndim - 1) + ((0, width - array.shape[-1]),))


def _prepared(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
              scale: float, natural: bool, tiles: Tiles, family: mosaic.Family):
    """A call as this mainloop takes it at `tiles`: padded to the program's blocks (`padded_call`, `_block`)
    and its heads to whole swizzled rows (`_columns`), with its `algebra.Forward` (every form guarding its
    sums where its values can overflow them, the fold its logits' reach, `algebra.walked_launches`) and the
    arrays its programs read beside the operands (`_Spans`, `_Hooked`). A call it takes none of is refused
    (`refused`, which the op's eligibility reads too)."""
    if reason := refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=False, family=family):
        raise NotImplementedError(reason)
    # The query taken by its scale's power of two before its products (`prescale`).
    query, scale, power = prescaled(query, scale)
    call = padded_call(query, key, value, spans, hooks, _block(tiles))
    spec = algebra.Forward(spans.form, spans.keys, ids=spans.query_ids is not None, scale=scale,
                           score_mod=hooks.score_mod, natural=natural, headroom=algebra.headroom(value.dtype),
                           initial=hooks.initial is not None,
                           queries=call.queries if call.query.shape[1] > call.queries else None,
                           prescale=power)
    unit = family.column_unit(query.dtype)
    operands = tuple(_columns(array, unit) for array in (call.query, call.key, call.value))
    read = (_spans_of(spec, call, tiles.block_q, tiles.block_k), _hooked_of(hooks, call, unit))
    return call, spec, operands, read


class _Residuals(NamedTuple):
    """A training forward consumer's residual views and the anchor keys' stage: its rows' maxima,
    denominators, anchor's own weight, other keys' weights' sum and their shift (`[bq]` fp32 views, through
    the consumer's row stage), its anchor keys (`[bq]` int32, through `keys_smem`) and its split sums
    (`[bq, Dv]` fp32, through the consumer's output stage, `_forward_allocations`)."""

    rows: tuple
    keys: object
    others: object
    keys_smem: object


def _scored(program: _Program, spec: algebra.Forward, q, index, slots, first, spans: _Spans | None,
            hooked: _Hooked, place: tuple, whole, turn, *, hooks: Hooks) -> tuple:
    """A key tile's first key, coordinates and logits under the score hook (batch row and head `place`, its
    queries from `first`), and their products where the logits are a factor of them (`algebra.logits_of`),
    every walk's (`_consumer`'s, `_peak_consumer`'s), so a peak walk's logits are the training walk's to the
    bit. A tile every query sees whole takes no mask: one walk, the choice a tile's (a walk of each, these
    rings' waits and arrivals in two loops, faulted on an H100 under jax 0.11.2's lowering)."""
    row, head = place
    family, tiles = program.family, program.tiles
    shape = (tiles.block_q, tiles.block_k)
    s = family.product(q, slots["keys"].transpose((1, 0)), shape, turn)
    slots.release("keys")
    k_first = index * tiles.block_k
    # The tile's places formed a step, not held across the walk beside its sums (`_coords`).
    coords = _coords(family, spans, row, first, k_first, shape)
    params = tuple(_param(family, ref, param.kind, row, head, first, s, k_first)
                   for ref, param in zip(hooked.score, hooks.score_params, strict=True))
    logits, products = algebra.logits_of(spec, s, coords, params, masked=jnp.logical_not(whole(index)))
    return k_first, coords, logits, products


def _consumer(program: _Program, spec: algebra.Forward, consumer, q_view, out_view, status_views, q_smem,
              q_loaded, out_smem, row_smem, streams, turn, streamed: tuple, first, spans: _Spans | None,
              hooked: _Hooked, guard: _Guard | None, place: tuple, *, hooks: Hooks,
              residuals: _Residuals | None, anchors: tuple | None = None, interpret: bool):
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
    XLA, `_trained_forward`)."""
    row, head, kv_head = place
    family, tiles = program.family, program.tiles
    lo, hi, whole = program.reach(first, spans, row)
    family.loaded(((q_view, q_smem, q_loaded),))
    q = family.held(q_smem)
    # The output's sums are as wide as its stage (a training walk's its split sums'), the values' width
    # whatever the query's.
    value_width = out_smem.shape[-1]
    # The weighted values' running sums live in the family's accumulator (`Family.walk`), from the initial
    # state's where there is one; the walk carries the maxima and denominators, the state's `acc` an empty
    # place the algebra passes through (`algebra.Keys.accumulate`), a constant no loop may carry.
    state_rows = (row, head, pl.ds(first, tiles.block_q))
    live = None if guard is None else family.rows_of(guard.live.at[state_rows]) > 0
    if hooked.initial is not None:
        m0, total0, acc0 = hooked.initial
        start = algebra.resumed(spec, family.rows_of(m0.at[state_rows]),
                                family.rows_of(total0.at[state_rows]), family.tile_of(acc0.at[state_rows]),
                                live=live)
    else:
        # `algebra.started`'s empty state, in the family's layouts.
        start = algebra.Partial(family.zeros((tiles.block_q, value_width)),
                                family.full((tiles.block_q,), -jnp.inf), family.zeros((tiles.block_q,)))
        start = start if live is None else algebra.headroomed(start, live)
    pl.when(consumer == 1)(turn)
    # The slots before this block's first tile its partner reads, the turns it would have taken.
    family.passed(streams, streamed[0], lo, first=streamed[0], between=lambda: (turn(), turn()))

    def scored(index, slots):
        return _scored(program, spec, q, index, slots, first, spans, hooked, (row, head), whole, turn,
                       hooks=hooks)

    def value_of(slots):
        # The value slot, once the partner's products are issued: its turn on the tensor cores is over when
        # this one's come.
        turn(wait=False)
        slot = slots["values"]
        turn(arrive=False)
        return slot

    def dot(lhs, rhs):
        return family.registers_product(lhs, rhs, interpret)

    def step(index, slots, carry, accumulators):
        k_first, coords, logits, products = scored(index, slots)
        load = functools.partial(value_of, slots)
        if guard is not None:
            magnitudes = guard.largest

            def largest():
                return family.columns_of(magnitudes.at[row, kv_head, pl.ds(k_first, tiles.block_k)])

            keys = algebra.Keys(coords, logits, load, dot, products=products, largest=largest)
            return algebra.plain_step(spec, carry, keys), accumulators
        # The sums are the accumulator's, which the step takes the tile into (`Keys.accumulate`).
        state = algebra.Partial(accumulators[0], *carry)
        keys = algebra.Keys(coords, logits, load, dot, _accumulated(family, interpret), products)
        state = algebra.plain_step(spec, state, keys)
        return (state.m, state.total), (state.acc,)

    trained = None
    if residuals is not None:
        assert anchors is not None
        known = algebra.Argmax(family.rows_of(anchors[0].at[state_rows]),
                               family.rows_of(anchors[1].at[state_rows]))
        state, trained = _trained(spec, family, start, known, scored, value_of, (lo, hi), streams,
                                  streamed[0], interpret=interpret)
        # The split sums as the walk left them, shift 0 (`algebra.split_residuals`' plain walk).
        family.stored((state.acc,), (residuals.others,), smem=out_smem)
    elif guard is not None:
        state, _ = family.walk(streams, ((step, lo, hi),), start, (), interpret, first=streamed[0])
    else:
        rows, (acc,) = family.walk(streams, ((step, lo, hi),), (start.m, start.total),
                                   ((tiles.block_q, value_width),), interpret, first=streamed[0],
                                   initial=(start.acc,))
        state = algebra.Partial(acc, *rows)
    # The slots past this block's last tile (a causal block whose diagonal ends first) its partner reads.
    family.passed(streams, hi, streamed[1], first=streamed[0], between=lambda: (turn(), turn()))
    pl.when(consumer == 0)(turn)
    overflowed = jnp.where(algebra.overflowed(state.acc), 1.0, 0.0)
    family.stored((family.as_row(state.m), family.as_row(overflowed)), status_views, smem=row_smem)
    if residuals is None or trained is None:
        out_params = tuple(_param(family, ref, param.kind, row, head, first, state.acc)
                           for ref, param in zip(hooked.output, hooks.output_params, strict=True))
        out, _, _ = algebra.finish(spec, state, initial=hooked.initial is not None,
                                   output_mod=hooks.output_mod, output_params=out_params)
        family.stored((out,), (out_view,), smem=out_smem)
    else:
        maximum, denominator = algebra.finished_rows(spec, state, initial=hooked.initial is not None)
        own, others_total, key = trained
        rows = (maximum, denominator, own, others_total, family.zeros((tiles.block_q,)))
        family.stored(tuple(family.as_row(part) for part in rows), residuals.rows, smem=row_smem)
        family.stored((family.as_row(key),), (residuals.keys,), smem=residuals.keys_smem)


def _accumulated(family: mosaic.Family, interpret: bool) -> Callable:
    """`algebra.Keys.accumulate` on the family's accumulator: the tile's products issued into it in place,
    after its factor where the step gives one (`Family.rescaled_products`)."""
    def accumulate(acc, scale, parts, value):
        if scale is None:
            return family.register_products(acc, parts, value, interpret)
        return family.rescaled_products(acc, scale, parts, value, interpret)

    return accumulate


def _trained(spec: algebra.Forward, family: mosaic.Family, start: algebra.Partial, known: algebra.Argmax,
             scored, value_of, reach, streams, first, *, interpret: bool) -> tuple[algebra.Partial, tuple]:
    """A training forward consumer's walk of its key tiles `reach` (`[lo, hi)`, the streams' steps from
    `first`) from `start` at the anchors and keys' largest logits `known` (`algebra.anchored`): each tile
    `algebra.anchored_step`, its split sums the family's accumulator, which takes each tile's product in place
    and no factor. `scored(index, slots)` gives a tile's place, coordinates, logits and products, and
    `value_of(slots)` its value slot. Returns the state with the split sums as its sums, and what the walk
    leaves beside them: the anchor's own weight, the other keys' weights' sum and the anchor keys."""
    state, arg, own, others_total = algebra.anchored(spec, start, known)

    def dot(lhs, rhs):
        return family.registers_product(lhs, rhs, interpret)

    def step(index, slots, carry, accumulators):
        total, own, others_total = carry
        _, coords, logits, products = scored(index, slots)
        keys = algebra.Keys(coords, logits, functools.partial(value_of, slots), dot,
                            _accumulated(family, interpret), products)
        # The sums are the accumulator's, which the step takes the tile into (`Keys.accumulate`).
        walked = algebra.Partial(accumulators[0], state.m, total)
        walked, _, own, others_total = algebra.anchored_step(spec, (walked, arg, own, others_total), keys)
        return (walked.total, own, others_total), (walked.acc,)

    (total, own, others_total), (acc,) = family.walk(
        streams, ((step, *reach),), (state.total, own, others_total), (state.acc.shape,), interpret,
        first=first, initial=(state.acc,))
    return algebra.Partial(acc, state.m, total), (own, others_total, arg.key)


def _peak_consumer(program: _Program, spec: algebra.Forward, consumer, q_view, peak_views: tuple, q_smem,
                   q_loaded, row_smem, keys_smem, streams, turn, streamed: tuple, first,
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
    family.passed(streams, streamed[0], lo, first=streamed[0], between=lambda: (turn(), turn()))

    def step(index, slots, arg, accumulators):
        _, coords, logits, _ = _scored(program, spec, q, index, slots, first, spans, hooked, place, whole,
                                       turn, hooks=hooks)
        # The partner's turn, as a training step's value product would take it.
        turn(wait=False)
        turn(arrive=False)
        return algebra.argmax(arg, logits, coords.k_pos), accumulators

    start = algebra.Argmax(family.full((tiles.block_q,), -jnp.inf),
                           family.full((tiles.block_q,), 0, jnp.int32))
    arg, _ = family.walk(streams, ((step, lo, hi),), start, (), interpret, first=streamed[0])
    family.passed(streams, hi, streamed[1], first=streamed[0], between=lambda: (turn(), turn()))
    pl.when(consumer == 0)(turn)
    family.stored((family.as_row(arg.peak),), (peak_views[0],), smem=row_smem)
    family.stored((family.as_row(arg.key),), (peak_views[1],), smem=keys_smem)


def _launch(query, key, value, spans: _Spans | None, hooked: _Hooked, guard: _Guard | None, *,
            spec: algebra.Forward, hooks: Hooks, tiles: Tiles, family: mosaic.Family, training: bool,
            interpret: bool, columns: int | None = None, peaks: bool = False,
            anchors: tuple[jax.Array, jax.Array] | None = None, out_dtype=None, counted: bool = False):
    """The forward's launch in `spec`'s form (`guard`'s rows shifting where given): its rows' status `[2, B,
    H, Qp]` fp32, their maxima in the walk's units and 1 where their weighted values overflowed
    (`algebra.walked_launches`), and its output `[B, Qp, H, Dv]` in `out_dtype` (the query's by default). In
    `training`, at the
    `anchors` a peak walk found, its status and residuals (`_Residuals`) and no output: the rows
    `[5, B, H, Qp]` fp32 (maxima, denominators, the anchor's own weight, the other keys' weights' sum and
    their shift), the anchor keys `[B, H, Qp]` int32 and the split sums `[B, H, Qp, Dv]` fp32, in `Split`'s
    layout, each block's rows one contiguous slab. A peak walk
    (`peaks`) reads the keys alone and gives each row's keys' largest logit `[B, H, Qp]` fp32 in the walk's
    units and its anchor key `[B, H, Qp]` int32 (`_peak_consumer`)."""
    batch, queries, heads, width = query.shape
    keys_count, kv_heads = key.shape[1], key.shape[2]
    group, consumers = heads // kv_heads, tiles.consumers
    block_q, block_k = tiles.block_q, tiles.block_k
    pair = consumers * block_q
    steps = keys_count // block_k
    # No more stages than key tiles: a slot past them would load past the keys.
    program = _Program(family, tiles, steps, min(tiles.stages, steps), spec.keys, spec.form == "causal",
                       spec.queries)
    # The output stores through the query's stage where the two are as wide (its rows are read before the
    # store); else it takes a stage of its own beside the query's and the rings (`_forward_allocations`).
    out_dtype = query.dtype if out_dtype is None else jnp.dtype(out_dtype)
    reuse = value.shape[-1] == width and out_dtype == query.dtype
    scoped, rings = _forward_allocations(family, tiles, width, value.shape[-1], query.dtype, training,
                                         program.stages, peaks=peaks, out_dtype=out_dtype)
    arrays = len(scoped) - 2

    def body(q_ref, k_ref, v_ref, *refs, streams):
        read, refs = refs[:len(leaves)], refs[len(leaves):]
        written, refs = refs[:len(out_type)], refs[len(out_type):]
        (q_smem, row_smem, *stages), (q_loaded, schedule) = refs[:arrays], refs[arrays:]
        out_smem, stages = (q_smem, stages) if peaks or (reuse and not training) else (stages[0], stages[1:])
        spans_of, hooked_of, guard_of, anchors_of = treedef.unflatten(read)
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        named = {"keys": k_ref} if peaks else {"keys": k_ref, "values": v_ref}
        views = _views_of(named, row, lax.div(head, group))
        streams = family.opened(streams, views)
        pair_first = program.query_tile(queries // pair) * pair
        streamed = _streamed([program.reach(pair_first + index * block_q, spans_of, row)[:2]
                              for index in range(consumers)])

        def turn(*, arrive: bool = True, wait: bool = True):
            # The consumers' turns on the tensor cores, where a program has two (a schedule barrier).
            if consumers > 1 and arrive:
                plgpu.barrier_arrive(schedule)
            if consumers > 1 and wait:
                plgpu.barrier_wait(schedule)

        def consumer(index):
            first = pair_first + index * block_q
            rows, state_rows = (row, pl.ds(first, block_q), head), (row, head, pl.ds(first, block_q))
            if peaks:
                peak_views = tuple(ref.at[state_rows] for ref in written)
                _peak_consumer(program, spec, index, q_ref.at[rows], peak_views, q_smem.at[index],
                               q_loaded.at[index], row_smem.at[index], stages[0].at[index],
                               streams, turn, streamed, first, spans_of, hooked_of, (row, head), hooks=hooks,
                               interpret=interpret)
                return
            status_ref, *outputs = written
            status = tuple(status_ref.at[place, *state_rows] for place in range(2))
            out_view, residuals = None, None
            if training:
                rows_ref, keys_ref, others_ref = outputs
                residuals = _Residuals(tuple(rows_ref.at[place, *state_rows] for place in range(5)),
                                       keys_ref.at[state_rows], others_ref.at[state_rows],
                                       stages[0].at[index])
            else:
                out_view = outputs[0].at[rows]
            _consumer(program, spec, index, q_ref.at[rows], out_view, status, q_smem.at[index],
                      q_loaded.at[index], out_smem.at[index], row_smem.at[index], streams, turn, streamed,
                      first, spans_of, hooked_of, guard_of, (row, head, lax.div(head, group)), hooks=hooks,
                      residuals=residuals, anchors=anchors_of, interpret=interpret)

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    # The arrays the programs read beside the operands.
    leaves, treedef = jax.tree.flatten((spans, hooked, guard, anchors))

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    status_type = jax.ShapeDtypeStruct((2, batch, heads, queries), jnp.float32)
    if peaks:
        out_type: tuple = (jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32),
                           jax.ShapeDtypeStruct((batch, heads, queries), jnp.int32))
    elif training:
        out_type = (status_type, jax.ShapeDtypeStruct((5, batch, heads, queries), jnp.float32),
                    jax.ShapeDtypeStruct((batch, heads, queries), jnp.int32),
                    jax.ShapeDtypeStruct((batch, heads, queries, value.shape[-1]), jnp.float32))
    else:
        out_type = (status_type, jax.ShapeDtypeStruct((batch, queries, heads, value.shape[-1]), out_dtype))
    approx_math = approximate(hooks, (block_q, block_k), value.shape[-1], differentiated=False)
    # Each launch named apart, so its compile reports and profile rows say which it is.
    kind = "peaks" if peaks else "training" if training else "forward"
    name = (f"attention_{kind}{'_natural' if spec.natural else ''}"
            f"{'_guarded' if guard is not None else ''}{'' if columns is None else f'_columns{columns}'}")
    # The model's products this launch owns (`stated`): the forward's two, a peak walk's `Q K^T` and a
    # training walk's `P V` (its `Q K^T` recomputed); a rerun (a guarded launch, the fold's natural walk)
    # none, as the FLOP count bounds a conditional by every branch, which would count the call twice.
    widths = width if peaks else value.shape[-1] if training else width + value.shape[-1]
    cost = (stated(spec, batch, heads, spec.queries or queries, widths, query, key, value) if counted
            else pl.CostEstimate(flops=0, transcendentals=0, bytes_accessed=0))
    return family.kernel(entry, out_type=out_type, grid=(heads, queries // pair, batch),
                         grid_names=("heads", "q_tiles", "batch"), consumers=consumers, interpret=interpret,
                         approx_math=approx_math, name=name, cost=cost,
                         reducing_warpgroups=reducing(family, consumers, hooks.score_mod, hooks.output_mod))(
                             query, key, value, *leaves)
