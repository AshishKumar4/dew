"""The attention template's plain-JAX mainloop (docs/design/kernel-templates.md §9): the algebra of
`attention_algebra` run over the same tiles, in the same order and with the same products as a kernel's
mainloop, in XLA ops, so it runs on any backend and is the boundary test's second arm
(tests/test_attention_algebra.py: the Triton mainloop, interpreted, and this one agree to the bit at the
same tiles). It takes a call at its launches' padded shapes, as `TritonForward` does."""

import functools
from typing import Literal, overload

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates import attention_algebra as algebra
from dew.nn.kernels.templates.attention import (
    ACCURATE,
    DKV_GRADS,
    DQ_GRADS,
    NATURAL,
    Dropout,
    Gradients,
    Hooks,
    KeySpans,
    Kind,
    PaddedCall,
    Split,
    block_map,
    gradients,
    interpreted_dot,
    padded_call,
    prescaled,
    ranges_walk,
    within,
)


def _tile_param(value: jax.Array, kind: Kind, row, head, rows: jax.Array, keys: jax.Array):
    """A score hook parameter's values for a tile of query rows `rows` and keys `keys` (index arrays) of
    batch row `row` and query head `head`, as `kind` lays them out: `_tile_param` in the Triton
    mainloop."""
    match kind:
        case Kind.HEAD:
            return value[head]
        case Kind.QUERY:
            return value[row, head, rows][:, None]
        case Kind.KEY:
            return value[row, head, keys][None, :]
        case Kind.TILE:
            return value[row, head][rows[:, None], keys[None, :]]
        case Kind.OUTPUT:
            raise ValueError("an output hook's parameter has no score tile")


def _output_param(value: jax.Array, kind: Kind, row, head, rows: jax.Array):
    """An output hook parameter's values for query rows `rows` of batch row `row` and query head `head`."""
    match kind:
        case Kind.HEAD:
            return value[head]
        case Kind.QUERY:
            return value[row, head, rows][:, None]
        case Kind.OUTPUT:
            return value[row, rows, head]
        case Kind.KEY | Kind.TILE:
            raise ValueError("an output hook's parameter has no key axis")


def _identity(step: jax.Array) -> jax.Array:
    """A walk's step as its key tile: a range's walk, one step a tile."""
    return step


def _composed(keys_at, tile_of):
    """`keys_at` of each step's key tile (`stepped`)."""
    return lambda step: keys_at(tile_of(step))


def _weighted(carry: algebra.Carry, *, rows) -> jax.Array:
    """A training walk's weighted values, its split sums joined (`algebra.joined`), where they overflow."""
    return algebra.joined(carry, rows).acc


def forward(spec: algebra.Forward, query: jax.Array, key: jax.Array, value: jax.Array, start: jax.Array,
            end: jax.Array, query_ids: jax.Array | None, key_ids: jax.Array | None, score_params,
            score_kinds: tuple[Kind, ...], output_params, output_kinds: tuple[Kind, ...], initial, *,
            block_q: int, block_k: int, output_mod=None, residuals: bool = True,
            drop: Dropout | None = None):
    """`TritonForward._fwd`'s outputs with its residuals from its operands (`query` `[B, Qp, H, Dp]`,
    `key` and `value` `[B, Kp, Hk, ...]`, the spans' rows and ids, the hooks' padded parameters, the
    initial state `(m0, total0, acc0)` or None), one query block and head at a time in sequence, its key
    tiles walked as the Triton mainloop walks them: the output, and with `residuals` its training step's
    (the split sums, `algebra.training_step`) maxima, denominators, anchor keys and split sums, else
    its plain step's output and None, so each rounds as the Triton launch it stands for."""
    batch, queries, heads, _ = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    blocks = queries // block_q
    blocks_map = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)
    ids = None if query_ids is None or key_ids is None else (query_ids, key_ids)
    tiles = -(-spec.keys // block_k)

    def program(index):
        i, rest = index // (batch * heads), index % (batch * heads)
        row, head = rest // heads, rest % heads
        draw = None if drop is None else drop.draw(row, head)
        first_query = i * block_q
        rows = first_query + jnp.arange(block_q)
        q_pos = rows[:, None]
        q = jax.lax.dynamic_slice_in_dim(query[row, :, head], first_query, block_q)
        kv_head = head // group
        spans = (start[row, rows], end[row, rows]) if spec.form == "spans" else None
        row_ids = None if ids is None else (ids[0][row, rows], ids[1][row])

        def keys_at(form: algebra.Forward, j) -> algebra.Keys:
            # A key tile's logits in `form` (the call's, or the fold's rewalk's, `algebra.walked`).
            columns = j * block_k + jnp.arange(block_k)
            coords = algebra.Coords(q_pos, columns[None, :],
                                    None if spans is None else algebra.intervals(*spans),
                                    None if row_ids is None else (row_ids[0], row_ids[1][columns]), draw)
            params = tuple(_tile_param(param, kind, row, head, rows, columns)
                           for param, kind in zip(score_params, score_kinds, strict=True))
            k_tile = jax.lax.dynamic_slice_in_dim(key[row, :, kv_head], j * block_k, block_k)
            v_tile = jax.lax.dynamic_slice_in_dim(value[row, :, kv_head], j * block_k, block_k)
            return algebra.Keys(coords, algebra.modify(form, interpreted_dot(q, k_tile.T), coords, params),
                                lambda: v_tile, interpreted_dot)

        def anchor_rows(keys):
            # Each row's value row at its key (the anchor's).
            return value[row, keys, kv_head]

        resumed = None if initial is None else (initial[0][row, head, rows], initial[1][row, head, rows],
                                                initial[2][row, head, rows])

        tile_of = _identity
        if spec.form == "spans" and blocks_map.ranges is not None:
            # Runs walk their ranges' tiles in turn.
            hi, tile_of = ranges_walk(blocks_map.ranges[row, i])
            lo = 0
        elif spec.form == "spans":
            lo, hi = blocks_map.lo[row, i], blocks_map.hi[row, i]
        elif spec.form == "causal":
            lo, hi = 0, jnp.minimum(-(-(first_query + block_q) // block_k), tiles)
        else:
            lo, hi = 0, tiles
        out_params = tuple(_output_param(param, kind, row, head, rows)
                           for param, kind in zip(output_params, output_kinds, strict=True))

        def walk(form: algebra.Forward, live):
            # The walk in `form`: the plain one where `live` is None, else the guarded one, those rows
            # shifting (`algebra.walked`).
            start_state = algebra.started(form, resumed, block_q, value_width, live)
            tiles_of = _composed(functools.partial(keys_at, form), tile_of)
            if residuals:
                return algebra.training_walk(form, lo, hi, tiles_of, anchor_rows, start_state,
                                             dropping=drop is not None)
            return algebra.plain_walk(form, lo, hi, tiles_of, start_state)

        def finished(form: algebra.Forward, carry):
            # A walk's outputs in `form`: a guarded training walk's live rows take their split sums from a
            # normalized pass at its maximum and denominator, as the Triton mainloop's.
            kept = ()
            state = carry
            if residuals:
                normalized = functools.partial(algebra.normalized_pass, form, carry,
                                               _composed(functools.partial(keys_at, form), tile_of),
                                               anchor_rows, (lo, hi),
                                               None if resumed is None else (resumed[0], resumed[2]))
                state, anchors, sums = algebra.split_written(carry, anchor_rows, normalized)
                kept = (anchors, *sums)
            out, m, denominator = algebra.finish(form, state, initial=initial is not None,
                                                 output_mod=output_mod, output_params=out_params)
            return (out.astype(query.dtype), m, denominator, *kept) if residuals else out.astype(query.dtype)

        # A training walk's weighted values are its split sums joined, where they overflow.
        weighted = functools.partial(_weighted, rows=anchor_rows) if residuals else None
        return algebra.walked(spec, walk, finished, weighted)

    mapped = jax.lax.map(program, jnp.arange(blocks * batch * heads))
    return _unmapped(mapped, (blocks, batch, heads, block_q, value_width), residuals=residuals)


def _unmapped(mapped, shape: tuple[int, ...], *, residuals: bool):
    """The programs' outputs `[blocks * batch * heads, bq, ...]` back to `[B, Qp, H, Dv]`, and with
    `residuals` theirs to `[B, H, Qp, ...]`, as `forward` returns them; `shape` the programs' grid and
    tile, `(blocks, batch, heads, block_q, value_width)`."""
    blocks, batch, heads, block_q, value_width = shape
    queries = blocks * block_q
    out = mapped[0] if residuals else mapped
    out = out.reshape(blocks, batch, heads, block_q, value_width).transpose(1, 0, 3, 2, 4)
    out = out.reshape(batch, queries, heads, value_width)
    if not residuals:
        return out, None
    _, m, denominator, anchors, others, own, others_total, shift = mapped

    def rowwise(x):
        return x.reshape(blocks, batch, heads, block_q).transpose(1, 2, 0, 3).reshape(batch, heads, queries)

    others = others.reshape(blocks, batch, heads, block_q, value_width).transpose(1, 2, 0, 3, 4)
    split = Split(others.reshape(batch, heads, queries, value_width), rowwise(own), rowwise(others_total),
                  rowwise(shift))
    return out, (rowwise(m), rowwise(denominator), rowwise(anchors), split)


@overload
def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
           *, block_q: int, block_k: int, pad: int, scale: float | None = None,
           residuals: Literal[False] = False, natural: bool = NATURAL) -> jax.Array:
    ...


@overload
def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
           *, block_q: int, block_k: int, pad: int, scale: float | None = None, residuals: Literal[True],
           natural: bool = NATURAL) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    ...


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
           *, block_q: int, block_k: int, pad: int, scale: float | None = None, residuals: bool = False,
           natural: bool = NATURAL):
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and `[B, K, Hk, Dv]` and the keys `spans` leaves each query, padded as a backend
    whose largest tile is `pad` pads it (`padded_call`) and walked in `block_q` by `block_k` tiles; with
    `residuals`, the running maxima and denominators `[B, H, Q]`, the anchor keys and the split sums too;
    `natural` keeps the logits natural (`algebra.Forward.base2`)."""
    hooks = Hooks() if hooks is None else hooks
    # The query taken by its scale's power of two before its products (`prescale`).
    query, scale, power = prescaled(query, query.shape[3] ** -0.5 if scale is None else scale)
    call = padded_call(query, key, value, spans, hooks, pad)
    queries, value_width = call.queries, value.shape[-1]
    # Every form guards its sums where its values can overflow them (`algebra.Forward.headroom`), and the
    # fold its logits' reach (`algebra.walked`).
    spec = algebra.Forward(spans.form, spans.keys, call.query_ids is not None, scale, hooks.score_mod,
                           natural, algebra.headroom(value.dtype), hooks.initial is not None,
                           compensated=algebra.compensates(value.dtype), prescale=power)
    if not residuals:
        out, _ = padded_walk(spec, call, hooks, block_q=block_q, block_k=block_k, residuals=False)
        return out[:, :queries, :, :value_width]
    out, (m, denominator, anchors, split) = padded_walk(spec, call, hooks, block_q=block_q, block_k=block_k,
                                                        residuals=True)
    sums = Split(split.others[:, :, :queries, :value_width], split.own[:, :, :queries],
                 split.others_total[:, :, :queries], split.shift[:, :, :queries])
    return (out[:, :queries, :, :value_width], m[:, :, :queries], denominator[:, :, :queries],
            anchors[:, :, :queries], sums)


@overload
def padded_walk(spec: algebra.Forward, call: PaddedCall, hooks: Hooks, *, block_q: int, block_k: int,
                residuals: Literal[False]) -> tuple[jax.Array, None]:
    ...


@overload
def padded_walk(spec: algebra.Forward, call: PaddedCall, hooks: Hooks, *, block_q: int, block_k: int,
                residuals: Literal[True]) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, Split]]:
    ...


def padded_walk(spec: algebra.Forward, call: PaddedCall, hooks: Hooks, *, block_q: int, block_k: int,
                residuals: bool):
    """`attend`'s walk of a padded call (`padded_call`) in `spec`'s form, which a caller holding a form of its
    own walks too (another mainloop's guarded rows, `attention_mosaic`): the output `[B, Qp, H, Dvp]` and,
    with `residuals`, the maxima, denominators, anchor keys and split sums at the padded shapes (else
    None)."""
    initial = None
    if hooks.initial is not None:
        state = hooks.initial
        widened = ((0, 0),) * 3 + ((0, call.value.shape[-1] - state.acc.shape[-1]),)
        initial = call.rows(state.m, -jnp.inf), call.rows(state.total), jnp.pad(call.rows(state.acc), widened)
    return forward(
        spec, call.query, call.key, call.value, call.start, call.end, call.query_ids, call.key_ids,
        call.score_params, tuple(param.kind for param in hooks.score_params), call.output_params,
        tuple(param.kind for param in hooks.output_params), initial, block_q=block_q, block_k=block_k,
        output_mod=hooks.output_mod, residuals=residuals, drop=hooks.drop)


def _seen(spans: tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None], row, rows: jax.Array,
          columns: jax.Array) -> jax.Array:
    """The entries `[len(rows), len(columns)]` of batch row `row` its spans let each query see."""
    start, end, query_ids, key_ids = spans
    ids = (None if query_ids is None or key_ids is None
           else (query_ids[row, rows][:, None], key_ids[row, columns][None, :]))
    return within(columns[None, :], start[row, rows][:, None], end[row, rows][:, None], ids)


def _dq_launch(query, key, value, dout, peak, total, initial_term, maximum, key_index, anchor_value, shift,
               sums, spans, score_params, *, score_mod, score_kinds: tuple[Kind, ...], scale: float,
               dq_scale: float, how: algebra.Recompute, block_q: int, block_k: int, split: bool,
               accurate: bool, drop=None):
    """The Triton mainloop's dQ launch (`attention_triton._dq_launch`), a query block and head at a time:
    each query's denominator and `D - c` from the forward's split `sums` (`algebra.split_delta`), or without
    `split` from a first pass at the forward's maximum (`algebra.off_anchor_delta`) but for the rows whose
    forward walked them normalized (shift -1, `algebra.split_residuals`), which the first pass cannot
    recompute without the forward's denominator; under dropout (`drop`) from its own centred first pass
    (`algebra.dropout_centre`); then dQ and the dQ program's hook gradients under it, `D` refined once from
    its own sums (`algebra.refinement`)."""
    batch, queries, heads, width = query.shape
    # The logits' scale as the forward took it: 1 for raw products, scaled in the weights (`how`).
    factor = scale if how.factor is None else 1.0
    keys, kv_heads = key.shape[1], key.shape[2]
    group = heads // kv_heads
    blocks = queries // block_q
    start, end, query_ids, key_ids = spans
    bounds = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)
    summed = tuple(kind in (Kind.HEAD, Kind.QUERY) for kind in score_kinds)

    def program(index):
        i, rest = index // (batch * heads), index % (batch * heads)
        row, head = rest // heads, rest % heads
        draw = None if drop is None else drop.draw(row, head)
        rows = i * block_q + jnp.arange(block_q)
        q_pos = rows[:, None]
        kv_head = head // group
        q = jax.lax.dynamic_slice_in_dim(query[row, :, head], i * block_q, block_q)
        do = jax.lax.dynamic_slice_in_dim(dout[row, :, head], i * block_q, block_q)
        peak0, total0, mean0 = peak[row, head, rows], total[row, head, rows], initial_term[row, head, rows]
        maxima, chosen = maximum[row, head, rows], key_index[row, head, rows]
        shifts = shift[row, head, rows]
        anchor_rows = jax.lax.dynamic_slice_in_dim(anchor_value[row, :, head], i * block_q, block_q)
        lo, hi, tile_of = bounds.lo[row, i], bounds.hi[row, i], _identity
        if bounds.ranges is not None:
            # Runs walk their ranges' tiles in turn, as the forward does.
            hi, tile_of = ranges_walk(bounds.ranges[row, i])
            lo = 0

        def tile(step):
            j = tile_of(step)
            columns = j * block_k + jnp.arange(block_k)
            params = tuple(_tile_param(param, kind, row, head, rows, columns)
                           for param, kind in zip(score_params, score_kinds, strict=True))
            k_tile = jax.lax.dynamic_slice_in_dim(key[row, :, kv_head], j * block_k, block_k)
            v_tile = jax.lax.dynamic_slice_in_dim(value[row, :, kv_head], j * block_k, block_k)
            return columns, _seen(spans, row, rows, columns), params, k_tile, v_tile

        before, state_total = algebra.off_anchor_start(peak0, total0, maxima, value.shape[-1], shifts)
        others, own, others_total, prior = (part[row, head, rows] for part in sums)
        m = maxima
        if draw is not None:
            # Dropout's D: each dropped product centred about its query's first visible tile's least one.
            def terms(j):
                columns, seen, params, k_tile, v_tile = tile(j)
                logits, _ = algebra.tile_logits(score_mod, interpreted_dot(q, k_tile.T), seen, q_pos,
                                                columns[None, :], params, scale=factor)
                products = algebra.dropped_products(interpreted_dot(do, v_tile.T),
                                                    draw(q_pos, columns[None, :]))
                return algebra.below(logits, m, how), products, seen, columns[None, :]

            anchor, delta, chosen = algebra.dropout_centre(lo, hi, terms, keys, block_q, state_total,
                                                           mean0)
        else:
            # The forward took the first pass (`algebra.split_delta`).
            anchor, delta, chosen = algebra.split_delta(others, own, others_total, prior, state_total,
                                                        anchor_rows, do, chosen, shifts)
            if not split:
                # D without the forward's split sums: the dQ launch's own pass over the keys.
                def row_terms(j, state):
                    columns, seen, params, k_tile, v_tile = tile(j)
                    logits, _ = algebra.tile_logits(score_mod, interpreted_dot(q, k_tile.T), seen, q_pos,
                                                    columns[None, :], params, scale=factor)
                    at_anchor = columns[None, :] == chosen[:, None]
                    exponentials, rest, own, small = algebra.off_anchor_weights(logits, maxima, at_anchor,
                                                                                v_tile.dtype, shifts, how)
                    return algebra.off_anchor(state, exponentials, rest, own, interpreted_dot(rest, v_tile),
                                              shifts, small, interpreted_dot(small, v_tile))

                walked = jax.lax.fori_loop(lo, hi, row_terms, before)
                first = algebra.off_anchor_delta(walked, anchor_rows, do, state_total, mean0,
                                                 key_index[row, head, rows], shifts)
                kept = shifts < 0
                prior = jnp.where(kept, prior, algebra.power2(walked.total, shifts))
                anchor, delta, chosen = (jnp.where(kept, ours, theirs)
                                         for ours, theirs in zip((anchor, delta, chosen), first, strict=True))

        def sweep(anchor, delta, chosen, total):
            # One pass over the keys: each cotangent over `total`, in the reference's order, and taken to
            # the pass's own at the end (`algebra.dq_finish`); the weights' sums unnormalized, as the Triton
            # launch's.
            def step(j, carry):
                columns, seen, params, k_tile, v_tile = tile(j)
                logits, slopes = algebra.tile_logits(score_mod, interpreted_dot(q, k_tile.T), seen, q_pos,
                                                     columns[None, :], params, scale=factor, slopes=True)
                weights = algebra.normalized(logits, m, jnp.ones_like(m), how)[0]
                zt = None if draw is None else draw(q_pos, columns[None, :])
                cotangents = algebra.grad_scores(weights, interpreted_dot(do, v_tile.T), anchor, delta, seen,
                                                 columns[None, :] == chosen[:, None], algebra.inverse(total),
                                                 multiplier=zt)
                return algebra.dq_step(carry, weights, cotangents, slopes, seen, k_tile, total,
                                       scale=dq_scale, kinds=score_kinds, summed=summed, dot=interpreted_dot,
                                       accurate=accurate)

            nothing = jnp.zeros((block_q,), jnp.float32)
            rows_sums = tuple(nothing for _ in score_kinds)
            zeros = jnp.zeros((block_q, width), jnp.float32)
            start = algebra.DqSums(zeros, nothing, nothing, rows_sums, rows_sums, zeros, nothing,
                                   jnp.full((block_q,), algebra.NO_SHIFT, jnp.float32))
            return jax.lax.fori_loop(lo, hi, step, start)

        found = sweep(anchor, delta, chosen, prior)
        dq, delta, denominator, mass, grads = algebra.dq_finish(found, prior, state_total, peak0,
                                                                (total0, mean0), m, anchor, delta,
                                                                accurate=accurate)
        return dq, (delta, anchor, m, denominator, mass), chosen, _written(grads, score_kinds)

    dq, rows_out, chosen, written = jax.lax.map(program, jnp.arange(blocks * batch * heads))
    return _dq_outputs(dq, rows_out, chosen, written, score_kinds, blocks, batch, heads, queries)


def _written(grads, kinds: tuple[Kind, ...]) -> tuple[jax.Array, ...]:
    """A dQ program's hook gradients as it writes them (`DQ_GRADS`): a head's partial sum, a query's own."""
    return tuple(jnp.sum(grad)[None] if kind == Kind.HEAD else grad
                 for grad, kind in zip(grads, kinds, strict=True) if kind in DQ_GRADS)


def _dq_outputs(dq, rows_out, chosen, written, kinds: tuple[Kind, ...], blocks: int, batch: int, heads: int,
                queries: int):
    """The dQ programs' outputs `[blocks * batch * heads, bq, ...]` laid out as the Triton launch's."""
    def queried(x):
        # [blocks * batch * heads, bq, ...] to [B, H, Qp, ...].
        return jnp.moveaxis(x.reshape(blocks, batch, heads, *x.shape[1:]), 0, 2).reshape(
            batch, heads, queries, *x.shape[2:])

    grads, written = [], list(written)
    for kind in kinds:
        if kind not in DQ_GRADS:
            continue
        piece = written.pop(0)
        grads.append(piece.reshape(blocks, batch, heads, 1) if kind == Kind.HEAD else queried(piece))
    return jnp.swapaxes(queried(dq), 1, 2), tuple(queried(x) for x in rows_out), queried(chosen), grads


def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans, score_params, *,
                score_mod, score_kinds: tuple[Kind, ...], scale: float, how: algebra.Recompute, block_q: int,
                block_k: int, accurate: bool, drop=None):
    """The Triton mainloop's dK/dV launch (`attention_triton._dkv_launch`), a key block and query head at a
    time, under the dQ launch's softmax at its tiles: the query head's dK, dV `[B, Kp, H, ...]` and the
    key and tile parameters' gradients (`DKV_GRADS`, `[B, H, Kp]` and `[B, H, Qp, Kp]`)."""

    def transposed_dot(a, b):
        # A query tile's share of the keys' gradients, `aᵀ @ b` (`algebra.cotangent_product`).
        return interpreted_dot(a.T, b)
    batch, queries, heads, width = query.shape
    # The logits' scale as the forward took it: 1 for raw products, scaled in the weights (`how`).
    factor = scale if how.factor is None else 1.0
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    blocks = keys // block_k
    start, end, query_ids, key_ids = spans
    ranges = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)

    def program(index):
        j, rest = index // (batch * heads), index % (batch * heads)
        row, head = rest // heads, rest % heads
        draw = None if drop is None else drop.draw(row, head)
        columns = j * block_k + jnp.arange(block_k)
        kv_head = head // group
        k_tile = jax.lax.dynamic_slice_in_dim(key[row, :, kv_head], j * block_k, block_k)
        v_tile = jax.lax.dynamic_slice_in_dim(value[row, :, kv_head], j * block_k, block_k)

        def step(i, carry):
            dk, dv, tiled, *grads = carry
            rows = i * block_q + jnp.arange(block_q)
            q = jax.lax.dynamic_slice_in_dim(query[row, :, head], i * block_q, block_q)
            do = jax.lax.dynamic_slice_in_dim(dout[row, :, head], i * block_q, block_q)
            seen = _seen(spans, row, rows, columns)
            params = tuple(_tile_param(param, kind, row, head, rows, columns)
                           for param, kind in zip(score_params, score_kinds, strict=True))
            m, denominator = peak[row, head, rows], total[row, head, rows]
            logits, slopes = algebra.tile_logits(score_mod, interpreted_dot(q, k_tile.T), seen, rows[:, None],
                                                 columns[None, :], params, scale=factor, slopes=True)
            exponentials, weights = algebra.normalized(logits, m, denominator, how)
            zt = None if draw is None else draw(rows[:, None], columns[None, :])
            cotangents = algebra.grad_scores(exponentials, interpreted_dot(do, v_tile.T),
                                             anchor[row, head, rows], delta[row, head, rows], seen,
                                             columns[None, :] == key_index[row, head, rows][:, None],
                                             algebra.inverse(denominator), multiplier=zt)
            pulled, pieces = algebra.pulled(cotangents, slopes, seen, score_kinds)
            dv = dv + transposed_dot(*algebra.value_operands(exponentials, weights, do, denominator,
                                                             multiplier=zt))
            dk = dk + algebra.cotangent_product(pulled * scale, q, transposed_dot, accurate=accurate)
            tiled = tuple(jax.lax.dynamic_update_slice_in_dim(entries, piece, i * block_q, axis=0)
                          for entries, piece in zip(tiled, (piece for kind, piece in zip(score_kinds, pieces,
                                                                                       strict=True)
                                                            if kind == Kind.TILE), strict=True))
            return dk, dv, tiled, *(grad + piece if kind == Kind.KEY else grad
                                    for grad, kind, piece in zip(grads, score_kinds, pieces, strict=True))

        sums = [jnp.zeros((block_k,), jnp.float32) if kind == Kind.KEY else jnp.zeros((), jnp.float32)
                for kind in score_kinds]
        zeros = jnp.zeros((block_k, width), jnp.float32), jnp.zeros((block_k, value_width), jnp.float32)
        tiled = tuple(jnp.zeros((queries, block_k), jnp.float32) for kind in score_kinds if kind == Kind.TILE)
        dk, dv, tiled, *grads = jax.lax.fori_loop(ranges.q_lo[row, j], ranges.q_hi[row, j], step,
                                                  (*zeros, tiled, *sums))
        keyed = iter(grad for grad, kind in zip(grads, score_kinds, strict=True) if kind == Kind.KEY)
        columns_of = iter(tiled)
        return dk, dv, tuple(next(keyed) if kind == Kind.KEY else next(columns_of)
                             for kind in score_kinds if kind in DKV_GRADS)

    dk, dv, written = jax.lax.map(program, jnp.arange(blocks * batch * heads))

    def keyed(x):
        # [blocks * batch * heads, bk, ...] to [B, Kp, H, ...].
        return jnp.moveaxis(x.reshape(blocks, batch, heads, *x.shape[1:]), 0, 2).reshape(
            batch, heads, keys, *x.shape[2:]).swapaxes(1, 2)

    def tiles(x):
        # [blocks * batch * heads, Qp, bk] to [B, H, Qp, Kp].
        return jnp.moveaxis(x.reshape(blocks, batch, heads, queries, block_k), 0, 3).reshape(
            batch, heads, queries, keys)

    kinds = [kind for kind in score_kinds if kind in DKV_GRADS]
    return keyed(dk), keyed(dv), tuple(keyed(grad[..., None])[..., 0].swapaxes(1, 2) if kind == Kind.KEY
                                       else tiles(grad) for grad, kind in zip(written, kinds, strict=True))


def _launch(query, key, value, dout, peak, total, initial_term, maximum, key_index, anchor_value, shift,
            others, own, others_total, forward_total, start, end, query_ids, key_ids, score_params, *,
            score_mod, score_kinds: tuple[Kind, ...], score_dtypes: tuple[str, ...], scale: float,
            prescale: float, block_q: int, block_k: int, split: bool, natural: bool, accurate: bool,
            initial: bool, drop=None):
    """The Triton backward's outputs (`TritonBackward`), from this mainloop's two launches, recomputing the
    weights as a forward of the call's form computed them (`algebra.recompute`), the query taken by
    `prescale` (`attention.prescale`), dQ at the call's whole scale."""
    del score_dtypes
    spans = start, end, query_ids, key_ids
    how = algebra.recompute(algebra.Forward("full", 0, ids=False, scale=scale, score_mod=score_mod,
                                            natural=natural, initial=initial, prescale=prescale))
    shared = {"score_mod": score_mod, "score_kinds": score_kinds, "scale": scale, "how": how,
              "block_q": block_q, "block_k": block_k, "accurate": accurate, "drop": drop}
    sums = others, own, others_total, forward_total
    dq, softmax, chosen, dq_grads = _dq_launch(query, key, value, dout, peak, total, initial_term, maximum,
                                               key_index, anchor_value, shift, sums, spans, score_params,
                                               split=split, dq_scale=scale * prescale, **shared)
    delta, anchor, m, denominator, _ = softmax
    dk, dv, key_grads = _dkv_launch(query, key, value, dout, m, denominator, anchor, delta, chosen, spans,
                                    score_params, **shared)
    return (dq, *softmax, chosen, *dq_grads, dk, dv, *key_grads)


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, block_q: int, block_k: int, pad: int,
             residuals: tuple[jax.Array, jax.Array, jax.Array, Split] | None = None, split: bool = True,
             scale: float | None = None, natural: bool = NATURAL, accurate: bool = ACCURATE,
             rounded: bool = True) -> Gradients:
    """The template's backward on this mainloop (`gradients`) at `block_q` by `block_k` tiles, the call
    padded as a backend whose largest tile is `pad` pads it, from the forward's `residuals` (this
    mainloop's `attend`'s where None); with `split`, D from the forward's split sums, as the Triton
    mainloop's `backward`, `natural` keeping the forward's logits natural and `accurate` taking the
    backward's accurate form (`ACCURATE`)."""
    if residuals is None:
        plain = Hooks() if hooks is None else Hooks(hooks.score_mod, hooks.score_params, hooks.initial,
                                                    drop=hooks.drop)
        _, maxima, totals, anchors, sums = attend(query, key, value, spans, plain, block_q=block_q,
                                                  block_k=block_k, pad=pad, scale=scale, residuals=True,
                                                  natural=natural)
        residuals = maxima, totals, anchors, sums
    initial = hooks is not None and hooks.initial is not None
    if hooks is not None and hooks.drop is not None and not split:
        raise ValueError("the first-pass D (split=False) takes no probability dropout; the split sums do")
    launch = functools.partial(_launch, block_q=block_q, block_k=block_k, split=split, natural=natural,
                               accurate=accurate, initial=initial, drop=None if hooks is None else hooks.drop)
    return gradients(launch, query, key, value, spans, hooks, dout, residuals, block=pad, scale=scale,
                     rounded=rounded)
