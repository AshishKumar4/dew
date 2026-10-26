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
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import attention_algebra as algebra, attention_portable, mosaic
from dew.nn.kernels.templates.attention import Hooks, KeySpans, Split, padded_call
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
    refused,
)


def _forward_allocations(family: mosaic.Family, tiles: Tiles, width: int, value_width: int, dtype,
                         residuals: bool, stages: int) -> tuple[list, dict]:
    """The forward program's allocations as `Family.scoped` takes them, in `_launch`'s order: its consumers'
    query stages and their rows' status; each one's output stage where the value is not as wide as the query
    (the output stores through the query's stage where it is); in a training forward its anchor keys' and
    split sums' stages (`_Residuals`); the query loads' barriers and the consumers' turns' barrier; and its
    key and value streams in `stages` slots (`Family.streams`)."""
    consumers, block_q, block_k = tiles.consumers, tiles.block_q, tiles.block_k
    scoped = [plgpu.SMEM((consumers, block_q, width), dtype), plgpu.SMEM((consumers, block_q), jnp.float32)]
    if value_width != width:
        scoped.append(plgpu.SMEM((consumers, block_q, value_width), dtype))
    if residuals:
        scoped += [plgpu.SMEM((consumers, block_q), jnp.int32),
                   plgpu.SMEM((consumers, block_q, value_width), jnp.float32)]
    scoped += [plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_arrivals=consumers)]
    blocks = {"keys": ((block_k, width), dtype), "values": ((block_k, value_width), dtype)}
    return scoped, family.streams(blocks, stages, consumers)


def guarded_columns(family: mosaic.Family, value_width: int, dtype, *, output_hook: bool) -> int:
    """The value columns a guarded launch walks at once (`_plain_forward`): a block of the family's swizzled
    columns (`Family.column_unit`), the whole width where an output hook reads a row's whole output."""
    return value_width if output_hook else min(value_width, family.column_unit(dtype))


def guarded_tiles(family: mosaic.Family, tiles: Tiles, columns: int, dtype) -> Tiles | None:
    """The guarded launch's tiles beside the plain launch's `tiles` (`_plain_forward`): its consumers and
    query blocks, and the widest of the family's key extents (`extents`) that divides the plain key tile, so
    the plain launch's padded keys are whole tiles of it, whose live set a consumer's registers hold
    (`mosaic.live_registers`): its shifted and unshifted sums of `columns` value columns fp32 carried, beside
    a step's logits and weights fp32 and the two products the step forms (`algebra.stepped`). None where
    none fits: the plain launch is then illegal at `tiles` (`MosaicForward.legal`)."""
    budget = family.registers(tiles.consumers)
    rows = tiles.block_q
    carried = [(rows, columns, jnp.float32)] * 2

    def held(block: int) -> bool:
        stepped = [(rows, block, jnp.float32)] * 2 + [(rows, columns, jnp.float32)] * 2
        return mosaic.live_registers(family, carried, stepped) <= budget

    blocks = [block for block in extents(family, dtype, tiles.block_k) if tiles.block_k % block == 0]
    return next((dataclasses.replace(tiles, block_k=block) for block in blocks if held(block)), None)


def _plain_forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                   scale: float, natural: bool, tiles: Tiles, interpret: bool) -> jax.Array:
    """`MosaicForward`'s launch without residuals, at `tiles`, under the default's guards over whole launches
    (`algebra.walked_launches`). The guarded launch (`_Guard`) runs a block of the family's swizzled value
    columns at a time: its walk carries its shifted and unshifted sums and forms both tiles' products a step
    (`algebra.stepped`), four `[block_q, Dv]` fp32 tiles beside its logits, which a block of the columns
    holds within a consumer's registers (ledger 1617). Each block's output is exact on its own (its rows'
    shifts its own columns' room, `algebra.room`); the rare launch takes its logits once a block."""
    family = bundle_of(tiles, query.dtype)
    call, spec, (q, k, v), (spans_of, hooked) = _prepared(query, key, value, spans, hooks, scale=scale,
                                                          natural=natural, tiles=tiles, family=family)

    def launch(form: algebra.Forward, live: jax.Array | None):
        if live is None:
            out, status = _launch(q, k, v, spans_of, hooked, None, spec=form, hooks=hooks, tiles=tiles,
                                  family=family, training=False, interpret=interpret)
            return out, status[0], status[1] > 0
        # An output hook reads a row's whole output, so its call takes the guarded launch at its full width.
        unit = guarded_columns(family, v.shape[-1], v.dtype, output_hook=hooks.output_mod is not None)
        guarded = guarded_tiles(family, tiles, unit, v.dtype)
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
            held = hooked if initial is None else hooked._replace(
                initial=(*initial[:2], initial[2][..., first:first + unit]))
            blocks.append(_launch(q, k, columns, guarded_spans, held, guard, spec=form, hooks=hooks,
                                  tiles=guarded, family=family, training=False, interpret=interpret,
                                  columns=first)[0])
        # Its maxima and overflows `walked_launches` reads of the plain launch alone.
        return jnp.concatenate(blocks, axis=-1), jnp.zeros(live.shape, jnp.float32), jnp.zeros_like(live)

    def kept(live: jax.Array, guarded: jax.Array, plain: jax.Array) -> jax.Array:
        return jnp.where(jnp.swapaxes(live, 1, 2)[..., None], guarded, plain)

    out = algebra.walked_launches(spec, launch, kept)
    return out[:, :call.queries, :, :value.shape[3]]


def _trained_forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                     scale: float, natural: bool, tiles: Tiles, interpret: bool
                     ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    """`MosaicForward`'s training launch at `tiles`: the algebra's training step taken in place
    (`algebra.split_taken`, `split_rows`). The old anchor's value row, which the split sums take back where
    the anchor moves, is the walk's own: each row's anchor's row kept in registers, picked from a value tile
    by its one-hot product where it moved.

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
            out, status, rows, anchors, others = _launch(q, k, v, spans_of, hooked, None, spec=form,
                                                         hooks=hooks, tiles=tiles, family=family,
                                                         training=True, interpret=interpret)
            split = Split(jnp.swapaxes(others, 1, 2), rows[2], rows[3], rows[4])
            return (out, rows[0], rows[1], anchors, split), status[0], status[1] > 0
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
    call = padded_call(query, key, value, spans, hooks, _block(tiles))
    spec = algebra.Forward(spans.form, spans.keys, ids=spans.query_ids is not None, scale=scale,
                           score_mod=hooks.score_mod, natural=natural, headroom=algebra.headroom(value.dtype),
                           initial=hooks.initial is not None,
                           queries=call.queries if call.query.shape[1] > call.queries else None)
    unit = family.column_unit(query.dtype)
    operands = tuple(_columns(array, unit) for array in (call.query, call.key, call.value))
    read = (_spans_of(spec, call, tiles.block_q, tiles.block_k), _hooked_of(hooks, call, unit))
    return call, spec, operands, read


class _Residuals(NamedTuple):
    """A training forward consumer's residual views and the shared stages it stores them through: its rows'
    maxima, denominators, anchor's own weight, other keys' weights' sum and their shift (`[bq]` fp32 views,
    through the consumer's row stage), its anchor keys (`[bq]` int32, through `keys_smem`) and its split
    sums (`[bq, Dv]` fp32, through `others_smem`)."""

    rows: tuple
    keys: object
    others: object
    keys_smem: object
    others_smem: object


def _consumer(program: _Program, spec: algebra.Forward, consumer, q_view, out_view, status_views, q_smem,
              q_loaded, out_smem, row_smem, streams, turn, streamed: tuple, first, spans: _Spans | None,
              hooked: _Hooked, guard: _Guard | None, place: tuple, *, hooks: Hooks,
              residuals: _Residuals | None, interpret: bool):
    """A consumer's query block (batch row, head and key head `place`): its walk of the key tiles it reads
    through the family's streams (`algebra.plain_step` a step) from the hooks' initial state, the score hook
    on each tile, passing those of the program's `streamed` range its partner alone reads; then its output
    under the output hook stored through `out_smem` (the query's stage where the widths agree), and its rows'
    maxima and whether their weighted values overflowed through `row_smem` to `status_views`
    (`algebra.walked_launches`). The plain walk's running sums are the family's accumulator, issued into in
    place; the guarded walk (`guard`, its rows `live` shifting) carries them, each tile's product fresh
    (`algebra.stepped`). With `residuals` the walk is the training step's (`algebra.training_step`'s
    arithmetic): the accumulator holds the split sums, every key's weighted values but the anchor's, taken
    in place (`algebra.split_taken`), the anchor's value row kept in registers, and its residuals stored."""
    row, head, kv_head = place
    family, tiles = program.family, program.tiles
    shape = (tiles.block_q, tiles.block_k)
    lo, hi, whole = program.reach(first, spans, row)
    family.loaded(((q_view, q_smem, q_loaded),))
    q = family.held(q_smem)
    # The output's sums are as wide as its stage, the values' width whatever the query's.
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
        # A key tile's logits under the score hook, and their products where the logits are a factor of
        # them. A tile every query sees whole takes no mask: one walk, the choice a tile's (a walk of each,
        # these rings' waits and arrivals in two loops, faulted on an H100 under jax 0.11.2's lowering).
        s = family.product(q, slots["keys"].transpose((1, 0)), shape, turn)
        slots.release("keys")
        k_first = index * tiles.block_k
        # The tile's places formed a step, not held across the walk beside its sums (`_coords`).
        coords = _coords(family, spans, row, first, k_first, shape)
        params = tuple(_param(family, ref, param.kind, row, head, first, s, k_first)
                       for ref, param in zip(hooked.score, hooks.score_params, strict=True))
        logits, products = algebra.logits_of(spec, s, coords, params, masked=jnp.logical_not(whole(index)))
        return k_first, coords, logits, products

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
        def accumulate(acc, scale, parts, value, _):
            return family.rescaled_products(acc, lambda acc: scale(acc, None), parts, value, interpret)

        # The sums are the accumulator's, which the step takes the tile into (`Keys.accumulate`).
        state = algebra.Partial(accumulators[0], *carry)
        state = algebra.plain_step(spec, state, algebra.Keys(coords, logits, load, dot, accumulate, products))
        return (state.m, state.total), (state.acc,)

    trained = None
    if residuals is not None:
        state, trained = _trained(spec, family, start, scored, value_of, (lo, hi), streams, streamed[0],
                                  kept_dtype=q_smem.dtype, interpret=interpret)
        # The split sums as the walk left them, shift 0 (`algebra.split_residuals`' plain walk), stored
        # before they join the anchor's term, so no copy of them stays alive beside the output's.
        family.stored((state.acc,), (residuals.others,), smem=residuals.others_smem)
        own, _, _, kept = trained
        state = state._replace(acc=algebra.split_output(state.acc, own, kept))
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
    out_params = tuple(_param(family, ref, param.kind, row, head, first, state.acc)
                       for ref, param in zip(hooked.output, hooks.output_params, strict=True))
    out, maximum, denominator = algebra.finish(spec, state, initial=hooked.initial is not None,
                                               output_mod=hooks.output_mod, output_params=out_params)
    family.stored((out,), (out_view,), smem=out_smem)
    if residuals is not None and trained is not None:
        own, others_total, key, _ = trained
        rows = (maximum, denominator, own, others_total, family.zeros((tiles.block_q,)))
        family.stored(tuple(family.as_row(part) for part in rows), residuals.rows, smem=row_smem)
        family.stored((family.as_row(key),), (residuals.keys,), smem=residuals.keys_smem)


def _trained(spec: algebra.Forward, family: mosaic.Family, start: algebra.Partial, scored, value_of, reach,
             streams, first, *, kept_dtype, interpret: bool) -> tuple[algebra.Partial, tuple]:
    """A training forward consumer's walk of its key tiles `reach` (`[lo, hi)`, the streams' steps from
    `first`) from `start`: `algebra.training_step`, its split sums in the family's accumulators, taken in
    place a block of the family's swizzled value columns at a time (`Keys.accumulate`), and the anchor's
    value row kept in registers, the same blocks of it, picked from a value tile by its one-hot product where
    the anchor moved (`KeptRows`): each block's join and pick take a `[rows, unit]` fp32
    temporary, not the whole width's. `scored(index, slots)` gives a tile's place, coordinates, logits and
    products, and `value_of(slots)` its value slot. Returns the state with the split sums as the state's
    sums, and what the walk leaves beside them: the anchor's own weight, the other keys' weights' sum, the
    anchor keys and the anchor's value row, which the split sums join for the output
    (`algebra.split_output`) once stored as residuals."""
    rows, width = start.acc.shape
    unit = min(width, family.column_unit(kept_dtype))
    blocks = [pl.ds(first_column, unit) for first_column in range(0, width, unit)]

    def dot(lhs, rhs):
        return family.registers_product(lhs, rhs, interpret)

    def step(index, slots, carry, accumulators):
        m, total, arg, own, others_total, row = carry
        _, coords, logits, products = scored(index, slots)

        def accumulate(acc, scale, parts, value, kept):
            return tuple(family.rescaled_products(block, lambda acc, held=held: scale(acc, held), parts,
                                                  value.at[:, columns], interpret)
                         for block, held, columns in zip(acc, kept, blocks, strict=True))

        def picked(onehot, value):
            return tuple(dot(onehot, value.at[:, columns]) for columns in blocks)

        keys = algebra.Keys(coords, logits, functools.partial(value_of, slots), dot, accumulate, products)
        # The sums are the accumulators', which the step takes the tile into (`Keys.accumulate`).
        state = algebra.Partial(tuple(accumulators), m, total)
        (state, arg, own, others_total), kept = algebra.training_step(
            spec, (state, arg, own, others_total), keys, algebra.KeptRows(picked, row))
        return (state.m, state.total, arg, own, others_total, kept.row), state.acc

    nothing = family.zeros((rows,))
    arg = algebra.Argmax(family.full((rows,), -jnp.inf), family.full((rows,), 0, jnp.int32))
    row = tuple(family.zeros((rows, unit)).astype(kept_dtype) for _ in blocks)
    initial = tuple(start.acc[:, columns.start:columns.start + unit] for columns in blocks)
    (m, total, arg, own, others_total, row), accumulators = family.walk(
        streams, ((step, *reach),), (start.m, start.total, arg, nothing, nothing, row),
        tuple((rows, unit) for _ in blocks), interpret, first=first, initial=initial)
    acc = jnp.concatenate(accumulators, axis=1) if len(blocks) > 1 else accumulators[0]
    kept = jnp.concatenate(row, axis=1) if len(blocks) > 1 else row[0]
    return algebra.Partial(acc, m, total), (own, others_total, arg.key, kept)


def _launch(query, key, value, spans: _Spans | None, hooked: _Hooked, guard: _Guard | None, *,
            spec: algebra.Forward, hooks: Hooks, tiles: Tiles, family: mosaic.Family, training: bool,
            interpret: bool, columns: int | None = None):
    """The forward's launch in `spec`'s form (`guard`'s rows shifting where given): its output `[B, Qp, H,
    Dv]` and its rows' status `[2, B, H, Qp]` fp32, their maxima in the walk's units and 1 where their
    weighted values overflowed (`algebra.walked_launches`); in `training` its residuals too (`_Residuals`):
    the rows `[5, B, H, Qp]` fp32 (maxima, denominators, the anchor's own weight, the other keys' weights'
    sum and their shift), the anchor keys `[B, H, Qp]` int32 and the split sums `[B, Qp, H, Dv]` fp32."""
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
    reuse = value.shape[-1] == width
    scoped, rings = _forward_allocations(family, tiles, width, value.shape[-1], query.dtype, training,
                                         program.stages)
    arrays = len(scoped) - 2

    def body(q_ref, k_ref, v_ref, *refs, streams):
        read, refs = refs[:len(leaves)], refs[len(leaves):]
        (out_ref, status_ref), refs = refs[:2], refs[2:]
        trained: tuple = ()
        if training:
            trained, refs = refs[:3], refs[3:]
        (q_smem, row_smem, *stages), (q_loaded, schedule) = refs[:arrays], refs[arrays:]
        out_smem, stages = (q_smem, stages) if reuse else (stages[0], stages[1:])
        spans_of, hooked_of, guard_of = treedef.unflatten(read)
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        views = _views_of({"keys": k_ref, "values": v_ref}, row, lax.div(head, group))
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
            rows = (row, pl.ds(first, block_q), head)
            status = tuple(status_ref.at[place, row, head, pl.ds(first, block_q)] for place in range(2))
            written = None
            if training:
                rows_ref, keys_ref, others_ref = trained
                keys_smem, others_smem = stages
                written = _Residuals(
                    tuple(rows_ref.at[place, row, head, pl.ds(first, block_q)] for place in range(5)),
                    keys_ref.at[row, head, pl.ds(first, block_q)], others_ref.at[rows], keys_smem.at[index],
                    others_smem.at[index])
            _consumer(program, spec, index, q_ref.at[rows], out_ref.at[rows], status, q_smem.at[index],
                      q_loaded.at[index], out_smem.at[index], row_smem.at[index], streams, turn, streamed,
                      first, spans_of, hooked_of, guard_of, (row, head, lax.div(head, group)), hooks=hooks,
                      residuals=written, interpret=interpret)

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    # The arrays the programs read beside the operands.
    leaves, treedef = jax.tree.flatten((spans, hooked, guard))

    def entry(*refs):
        family.scoped(functools.partial(body, *refs), *scoped, streams=rings)

    out_type = (jax.ShapeDtypeStruct((batch, queries, heads, value.shape[-1]), query.dtype),
                jax.ShapeDtypeStruct((2, batch, heads, queries), jnp.float32))
    if training:
        out_type += (jax.ShapeDtypeStruct((5, batch, heads, queries), jnp.float32),
                     jax.ShapeDtypeStruct((batch, heads, queries), jnp.int32),
                     jax.ShapeDtypeStruct((batch, queries, heads, value.shape[-1]), jnp.float32))
    approx_math = approximate(hooks, (block_q, block_k), value.shape[-1], differentiated=False)
    # Each launch named apart, so its compile reports and profile rows say which it is.
    name = (f"attention_{'training' if training else 'forward'}{'_natural' if spec.natural else ''}"
            f"{'_guarded' if guard is not None else ''}{'' if columns is None else f'_columns{columns}'}")
    return family.kernel(entry, out_type=out_type, grid=(heads, queries // pair, batch),
                         grid_names=("heads", "q_tiles", "batch"), consumers=consumers, interpret=interpret,
                         approx_math=approx_math, name=name)(query, key, value, *leaves)
