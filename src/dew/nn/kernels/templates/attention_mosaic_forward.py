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
    BUNDLES,
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


def _plain_forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                   scale: float, natural: bool, tiles: Tiles, interpret: bool) -> jax.Array:
    """`MosaicForward`'s launch without residuals, at `tiles`, under the default's guards over whole launches
    (`algebra.walked_launches`)."""
    family = BUNDLES[tiles.family]
    call, spec, (q, k, v), (spans_of, hooked) = _prepared(query, key, value, spans, hooks, scale=scale,
                                                          natural=natural, tiles=tiles, family=family)

    def launch(form: algebra.Forward, live: jax.Array | None):
        # The guarded launch reads its live rows and its keys' values' largest magnitudes (`algebra.room`).
        guard = None if live is None else _Guard(live.astype(jnp.int32),
                                                 jnp.swapaxes(algebra.magnitudes(v), 1, 2))
        out, status = _launch(q, k, v, spans_of, hooked, guard, spec=form, hooks=hooks, tiles=tiles,
                              family=family, training=False, interpret=interpret)
        return out, status[0], status[1] > 0

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
    family = BUNDLES[tiles.family]
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
    q_pos, columns = first + family.iota(shape, 0), family.iota(shape, 1)
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
        coords = _coords(family, spans, row, first, k_first, shape, q_pos, columns)
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
        held = list(accumulators)

        def accumulate(scale, p, value):
            held[0] = family.rescaled_products(held[0], scale, ((p, None),), value, interpret)

        state = algebra.Partial(jnp.zeros((), jnp.float32), *carry)
        state = algebra.plain_step(spec, state, algebra.Keys(coords, logits, load, dot, accumulate, products))
        return (state.m, state.total), tuple(held)

    trained = None
    if residuals is not None:
        state, trained = _trained(spec, family, start, scored, value_of, (lo, hi), streams, streamed[0],
                                  kept_dtype=q_smem.dtype, interpret=interpret)
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
        # The split sums as the walk left them, shift 0 (`algebra.split_residuals`' plain walk).
        others, own, others_total, key = trained
        family.stored((others,), (residuals.others,), smem=residuals.others_smem)
        rows = (maximum, denominator, own, others_total, family.zeros((tiles.block_q,)))
        family.stored(tuple(family.as_row(part) for part in rows), residuals.rows, smem=row_smem)
        family.stored((family.as_row(key),), (residuals.keys,), smem=residuals.keys_smem)


def _trained(spec: algebra.Forward, family: mosaic.Family, start: algebra.Partial, scored, value_of, reach,
             streams, first, *, kept_dtype, interpret: bool) -> tuple[algebra.Partial, tuple]:
    """A training forward consumer's walk of its key tiles `reach` (`[lo, hi)`, the streams' steps from
    `first`) from `start`: `algebra.training_step`'s arithmetic with the split sums in the family's
    accumulator, taken in place before the tile's other keys' product joins them (`algebra.split_taken`),
    and the old anchor's value row the one kept in registers, picked from a value tile by its one-hot
    product where the anchor moved. `scored(index, slots)` gives a tile's place, coordinates, logits and
    products, and `value_of(slots)` its value slot. Returns the state with the split sums joined
    (`algebra.split_output`, the weighted values the output normalizes) and the residuals the walk leaves:
    the split sums, the anchor's own weight, the other keys' weights' sum and the anchor keys."""
    rows, width = start.acc.shape

    def step(index, slots, carry, accumulators):
        m, total, peak, key, own, others_total, kept = carry
        _, coords, logits, products = scored(index, slots)
        new, alpha = algebra.peak(spec, m, logits, products)
        p = algebra.weights(spec, logits, new)
        value = value_of(slots)
        anchor = algebra.argmax(algebra.Argmax(peak, key), logits, coords.k_pos)
        moved = anchor.peak > peak
        at_anchor = coords.k_pos == algebra.expand(anchor.key, coords.k_pos)
        others, own_tile = algebra.anchor_split(algebra.narrow(p, value.dtype), at_anchor)
        (acc,) = accumulators
        acc = family.rescaled_products(acc, lambda taken: algebra.split_taken(taken, own, alpha, moved, kept),
                                       ((others, None),), value, interpret)
        own, others_total = algebra.split_rows(own, others_total, alpha, moved, others, own_tile)
        # The anchor's value row where it moved to this tile: its one-hot product with the tile's values.
        picked = family.registers_product(at_anchor.astype(value.dtype), value, interpret)
        kept = jnp.where(algebra.expand(moved, kept), picked.astype(kept.dtype), kept)
        total = algebra.denominator_after(total, alpha, p)
        return (new, total, anchor.peak, anchor.key, own, others_total, kept), (acc,)

    nothing = family.zeros((rows,))
    no_key = family.full((rows,), 0, jnp.int32)
    carry = (start.m, start.total, family.full((rows,), -jnp.inf), no_key, nothing, nothing,
             family.zeros((rows, width)).astype(kept_dtype))
    (m, total, _, key, own, others_total, kept), (acc,) = family.walk(
        streams, ((step, *reach),), carry, ((rows, width),), interpret, first=first, initial=(start.acc,))
    return algebra.Partial(algebra.split_output(acc, own, kept), m, total), (acc, own, others_total, key)


def _launch(query, key, value, spans: _Spans | None, hooked: _Hooked, guard: _Guard | None, *,
            spec: algebra.Forward, hooks: Hooks, tiles: Tiles, family: mosaic.Family, training: bool,
            interpret: bool):
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
    return family.kernel(entry, out_type=out_type, grid=(heads, queries // pair, batch),
                         grid_names=("heads", "q_tiles", "batch"), consumers=consumers, interpret=interpret,
                         approx_math=approx_math)(query, key, value, *leaves)
