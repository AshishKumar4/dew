"""The attention template's plain-JAX mainloop (docs/design/kernel-templates.md §9): the algebra of
`attention_algebra` run over the same tiles, in the same order and with the same products as a kernel's
mainloop, in XLA ops, so it runs on any backend and is the boundary test's second arm
(tests/test_attention_algebra.py: the Triton mainloop, interpreted, and this one agree to the bit at the
same tiles). It takes a call at its launches' padded shapes, as `TritonForward` does."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates import attention_algebra as algebra
from dew.nn.kernels.templates.attention import (
    DKV_GRADS,
    DQ_GRADS,
    NATURAL,
    Gradients,
    Hooks,
    KeySpans,
    Kind,
    Split,
    block_map,
    gradients,
    interpreted_dot,
    padded_call,
)


def _normalized_pass(spec: algebra.Forward, state: algebra.Partial, logits_at: Callable, anchor: jax.Array, *,
                     bounds, initial: bool) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A guarded training walk's live rows' split sums, normalized: a pass over its keys (`logits_at` a
    tile's coordinates, logits and values) at the walk's maximum and natural denominator
    (`algebra.normalized_terms`), as the Triton mainloop's."""
    _, maximum, total = algebra.finish(spec, state, initial=initial)

    def step(j, carry):
        coords, logits, v_tile = logits_at(j)
        rest, own_tile = algebra.normalized_terms(logits, maximum, total, coords.k_pos == anchor[:, None],
                                                  v_tile.dtype, algebra.recompute(spec))
        return carry[0] + _dot(rest, v_tile), carry[1] + own_tile, carry[2] + rest.sum(axis=-1)
    nothing = jnp.zeros_like(total)
    return jax.lax.fori_loop(*bounds, step, (jnp.zeros_like(state.acc), nothing, nothing))


def _plain(held: algebra.Room | None, value: jax.Array) -> jax.Array | None:
    """A guarded tile's unshifted weights' product with its values (`algebra.Room.plain`), None unguarded."""
    return None if held is None else _dot(algebra.narrow(held.plain, value.dtype), value)


def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
    """`a @ b`, fp32, the template's CPU product (`interpreted_dot`)."""
    return interpreted_dot(a, b)


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


def forward(spec: algebra.Forward, query: jax.Array, key: jax.Array, value: jax.Array, start: jax.Array,
            end: jax.Array, query_ids: jax.Array | None, key_ids: jax.Array | None, score_params,
            score_kinds: tuple[Kind, ...], output_params, output_kinds: tuple[Kind, ...], initial, *,
            block_q: int, block_k: int, output_mod=None, residuals: bool = True):
    """`TritonForward._fwd`'s outputs with its residuals from its operands (`query` `[B, Qp, H, Dp]`,
    `key` and `value` `[B, Kp, Hk, ...]`, the spans' rows and ids, the hooks' padded parameters, the
    initial state `(m0, total0, acc0)` or None), one query block and head at a time in sequence, its key
    tiles walked as the Triton mainloop walks them: the output, and with `residuals` its training step's
    (the split sums, `algebra.split_accumulate`) maxima, denominators, anchor keys and split sums, else
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
        first_query = i * block_q
        rows = first_query + jnp.arange(block_q)
        q_pos = rows[:, None]
        q = jax.lax.dynamic_slice_in_dim(query[row, :, head], first_query, block_q)
        kv_head = head // group
        narrowing = {}
        if spec.form == "spans":
            narrowing = {"start": start[row, rows], "end": end[row, rows]}
        if ids is not None:
            narrowing["query_ids"] = ids[0][row, rows]

        def logits_at(j):
            columns = j * block_k + jnp.arange(block_k)
            coords = algebra.Coords(q_pos, columns[None, :], **narrowing,
                                    key_ids=None if ids is None else ids[1][row, columns])
            params = tuple(_tile_param(param, kind, row, head, rows, columns)
                           for param, kind in zip(score_params, score_kinds, strict=True))
            k_tile = jax.lax.dynamic_slice_in_dim(key[row, :, kv_head], j * block_k, block_k)
            v_tile = jax.lax.dynamic_slice_in_dim(value[row, :, kv_head], j * block_k, block_k)
            return coords, algebra.modify(spec, _dot(q, k_tile.T), coords, params), v_tile

        def softmax(j, state: algebra.Partial):
            coords, logits, v_tile = logits_at(j)
            m, alpha = algebra.peak(spec, state.m, logits)
            weights = algebra.weights(spec, logits, m)
            weights, factor, held = algebra.tile(state, alpha, weights, v_tile)
            return coords, logits, m, factor, weights, held, v_tile

        def plain(j, state: algebra.Partial) -> algebra.Partial:
            _, _, m, factor, weights, held, v_tile = softmax(j, state)
            product = _dot(algebra.narrow(weights, v_tile.dtype), v_tile)
            return algebra.stepped(state, m, factor, weights, product, held, _plain(held, v_tile))

        def step(j, carry: tuple[algebra.Partial, algebra.Argmax, jax.Array, jax.Array]):
            state, arg, own, others_total = carry
            coords, logits, m, factor, weights, held, v_tile = softmax(j, state)
            # The training forward's split sums (`algebra.split_accumulate`), as the Triton mainloop's.
            anchor = algebra.argmax(arg, logits, coords.k_pos)
            others, own_tile = algebra.anchor_split(algebra.narrow(weights, v_tile.dtype),
                                                    coords.k_pos == anchor.key[:, None])
            acc, own, others_total = algebra.split_accumulate(
                state.acc, own, others_total, factor, anchor.peak > arg.peak, value[row, arg.key, kv_head],
                _dot(others, v_tile), others, own_tile, held)
            stepped = algebra.stepped(state, m, factor, weights, jnp.zeros_like(acc), held,
                                      _plain(held, v_tile))
            return stepped._replace(acc=acc), anchor, own, others_total

        def begun(live) -> algebra.Partial:
            if initial is None:
                empty = algebra.empty(block_q, value_width)
                return empty if live is None else algebra.headroomed(empty, live)
            m0, total0, acc0 = initial
            return algebra.resumed(spec, m0[row, head, rows], total0[row, head, rows], acc0[row, head, rows],
                                   live=live)

        def walk(live=None):
            # The plain walk where `live` is None, else the guarded one, those rows shifting.
            if not residuals:
                return jax.lax.fori_loop(lo, hi, plain, begun(live))
            nothing = jnp.zeros((block_q,), jnp.float32)
            carry = begun(live), algebra.no_argmax(block_q), nothing, nothing
            return jax.lax.fori_loop(lo, hi, step, carry)

        if spec.form == "spans":
            lo, hi = blocks_map.lo[row, i], blocks_map.hi[row, i]
        elif spec.form == "causal":
            lo, hi = 0, jnp.minimum(-(-(first_query + block_q) // block_k), tiles)
        else:
            lo, hi = 0, tiles
        out_params = tuple(_output_param(param, kind, row, head, rows)
                           for param, kind in zip(output_params, output_kinds, strict=True))
        def weighted(carry):
            # A training walk's weighted values are its split sums joined, where they overflow.
            if not residuals:
                return carry.acc
            split, arg, own, _ = carry
            return algebra.split_output(split.acc, own, value[row, arg.key, kv_head])

        walked = algebra.guarded(walk, walk, weighted) if spec.headroom else walk()
        if residuals:
            split, arg, own, others_total = walked
            anchor_value = value[row, arg.key, kv_head]
            state = split._replace(acc=algebra.split_output(split.acc, own, anchor_value))
            terms = None if split.shift is None else _normalized_pass(
                spec, state, logits_at, arg.key, bounds=(lo, hi), initial=initial is not None)
            sums = algebra.split_residuals(split, own, others_total, terms)
            kept = (arg.key, *sums)
        else:
            state, kept = walked, ()
        out, m, denominator = algebra.finish(spec, state, initial=initial is not None, output_mod=output_mod,
                                             output_params=out_params)
        return (out.astype(query.dtype), m, denominator, *kept) if residuals else out.astype(query.dtype)

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


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
           *, block_q: int, block_k: int, pad: int, scale: float | None = None, residuals: bool = False,
           natural: bool = NATURAL):
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and `[B, K, Hk, Dv]` and the keys `spans` leaves each query, padded as a backend
    whose largest tile is `pad` pads it (`padded_call`) and walked in `block_q` by `block_k` tiles; with
    `residuals`, the running maxima and denominators `[B, H, Q]`, the anchor keys and the split sums too;
    `natural` keeps the logits natural (`algebra.Forward.base2`)."""
    hooks = Hooks() if hooks is None else hooks
    call = padded_call(query, key, value, spans, hooks, pad)
    queries, value_width = call.queries, value.shape[-1]
    initial = None
    if hooks.initial is not None:
        state = hooks.initial
        initial = (call.rows(state.m, -jnp.inf), call.rows(state.total),
                   jnp.pad(call.rows(state.acc), ((0, 0),) * 3 + ((0, call.value.shape[-1] - value_width),)))
    # The accurate form alone guards its sums (`algebra.Forward.headroom`); the fold walks as ever.
    resumes = initial is not None
    spec = algebra.Forward(spans.form, spans.keys, call.query_ids is not None,
                           query.shape[3] ** -0.5 if scale is None else scale, hooks.score_mod, natural,
                           natural and algebra.headroom(value.dtype, call.key.shape[1], resumes), resumes)
    out, kept = forward(
        spec, call.query, call.key, call.value, call.start, call.end, call.query_ids, call.key_ids,
        call.score_params, tuple(param.kind for param in hooks.score_params), call.output_params,
        tuple(param.kind for param in hooks.output_params), initial, block_q=block_q, block_k=block_k,
        output_mod=hooks.output_mod, residuals=residuals)
    if kept is None:
        return out[:, :queries, :, :value_width]
    m, denominator, anchors, split = kept
    sums = Split(split.others[:, :, :queries, :value_width], split.own[:, :, :queries],
                 split.others_total[:, :, :queries], split.shift[:, :, :queries])
    return (out[:, :queries, :, :value_width], m[:, :, :queries], denominator[:, :, :queries],
            anchors[:, :, :queries], sums)


def _seen(spans: tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None], row, rows: jax.Array,
          columns: jax.Array) -> jax.Array:
    """The entries `[len(rows), len(columns)]` of batch row `row` its spans let each query see."""
    start, end, query_ids, key_ids = spans
    seen = (columns[None, :] >= start[row, rows][:, None]) & (columns[None, :] < end[row, rows][:, None])
    if query_ids is not None and key_ids is not None:
        seen &= query_ids[row, rows][:, None] == key_ids[row, columns][None, :]
    return seen


def _logits(score_mod, query, key, seen, q_pos, k_pos, params, scale: float, slopes: bool = False):
    """A tile's natural logits under `score_mod`, -inf where unseen, and with `slopes` its entries'
    derivatives: the Triton mainloop's `_tile_logits`."""
    logits, derivatives = algebra.scored(score_mod, _dot(query, key.T) * scale, q_pos, k_pos, params,
                                         slopes=slopes)
    return jnp.where(seen, logits, -jnp.inf), derivatives


def _dq_launch(query, key, value, dout, peak, total, initial_term, maximum, key_index, anchor_value, shift,
               sums, spans, score_params, *, score_mod, score_kinds: tuple[Kind, ...], scale: float,
               how: algebra.Recompute, block_q: int, block_k: int):
    """The Triton mainloop's dQ launch (`attention_triton._dq_launch`), a query block and head at a time:
    the first pass at the forward's maximum to each query's denominator and `D - c`
    (`algebra.off_anchor_delta`), then dQ and the dQ program's hook gradients under it, `D` refined once
    from its own sums (`algebra.refinement`)."""
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
        rows = i * block_q + jnp.arange(block_q)
        q_pos = rows[:, None]
        kv_head = head // group
        q = jax.lax.dynamic_slice_in_dim(query[row, :, head], i * block_q, block_q)
        do = jax.lax.dynamic_slice_in_dim(dout[row, :, head], i * block_q, block_q)
        peak0, total0, term0 = peak[row, head, rows], total[row, head, rows], initial_term[row, head, rows]
        maxima, chosen = maximum[row, head, rows], key_index[row, head, rows]
        shifts = shift[row, head, rows]
        anchor_rows = jax.lax.dynamic_slice_in_dim(anchor_value[row, :, head], i * block_q, block_q)
        lo, hi = bounds.lo[row, i], bounds.hi[row, i]

        def tile(j):
            columns = j * block_k + jnp.arange(block_k)
            params = tuple(_tile_param(param, kind, row, head, rows, columns)
                           for param, kind in zip(score_params, score_kinds, strict=True))
            k_tile = jax.lax.dynamic_slice_in_dim(key[row, :, kv_head], j * block_k, block_k)
            v_tile = jax.lax.dynamic_slice_in_dim(value[row, :, kv_head], j * block_k, block_k)
            return columns, _seen(spans, row, rows, columns), params, k_tile, v_tile

        before, state_total, state_share = algebra.off_anchor_start(peak0, total0, term0, maxima,
                                                                    value.shape[-1], shifts)
        if sums is None:
            # D without the forward's split sums: the dQ launch's own pass over the keys.
            def row_terms(j, state):
                columns, seen, params, k_tile, v_tile = tile(j)
                logits, _ = _logits(score_mod, q, k_tile, seen, q_pos, columns[None, :], params, factor)
                at_anchor = columns[None, :] == chosen[:, None]
                exponentials, rest, own, small = algebra.off_anchor_weights(logits, maxima, at_anchor,
                                                                            v_tile.dtype, shifts, how)
                return algebra.off_anchor(state, exponentials, rest, own, _dot(rest, v_tile), shifts, small,
                                          _dot(small, v_tile))

            walked = jax.lax.fori_loop(lo, hi, row_terms, before)
            prior = algebra.power2(walked.total, shifts)
            anchor, delta, chosen = algebra.off_anchor_delta(walked, anchor_rows, do, state_total,
                                                             state_share, chosen, shifts)
        else:
            # The forward took the first pass (`algebra.split_delta`).
            others, own, others_total, prior = (part[row, head, rows] for part in sums)
            anchor, delta, chosen = algebra.split_delta(others, own, others_total, prior, state_total,
                                                        anchor_rows, do, chosen, shifts)
        m = maxima

        def step(j, carry):
            # Each cotangent over the prior denominator, in the reference's order, and taken to the pass's
            # own at the end (`algebra.dq_finish`); the weights' sums unnormalized, as the Triton
            # mainloop's dQ launch.
            dq, residual, mass, grads, weighed = carry
            columns, seen, params, k_tile, v_tile = tile(j)
            logits, slopes = _logits(score_mod, q, k_tile, seen, q_pos, columns[None, :], params, factor,
                                     slopes=True)
            weights = algebra.normalized(logits, m, jnp.ones_like(m), how)[0]
            cotangents = algebra.grad_scores(weights, _dot(do, v_tile.T), anchor, delta, seen,
                                             columns[None, :] == chosen[:, None], algebra.inverse(prior))
            pulled, pieces = algebra.pulled(cotangents, slopes, seen, score_kinds)
            dq = dq + _dot((pulled * scale).astype(k_tile.dtype), k_tile)
            grads = tuple(grad + piece if add else grad for grad, add, piece in zip(grads, summed, pieces,
                                                                                   strict=True))
            if any(summed):
                _, pieces = algebra.pulled(weights, slopes, seen, score_kinds)
                weighed = tuple(sums + piece if add else sums
                                for sums, add, piece in zip(weighed, summed, pieces, strict=True))
            return dq, residual + cotangents.sum(axis=-1), mass + weights.sum(axis=-1), grads, weighed

        nothing = jnp.zeros((block_q,), jnp.float32)
        rows_sums = tuple(nothing for _ in score_kinds)
        found = algebra.DqSums(*jax.lax.fori_loop(
            lo, hi, step, (jnp.zeros((block_q, width), jnp.float32), nothing, nothing, rows_sums, rows_sums)))
        dq, delta, denominator, mass, grads = algebra.dq_finish(found, prior, state_total, peak0,
                                                                (total0, term0), m, anchor, delta)
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
                block_k: int):
    """The Triton mainloop's dK/dV launch (`attention_triton._dkv_launch`), a key block and query head at a
    time, under the dQ launch's softmax at its tiles: the query head's dK, dV `[B, Kp, H, ...]` and the
    key and tile parameters' gradients (`DKV_GRADS`, `[B, H, Kp]` and `[B, H, Qp, Kp]`)."""
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
            logits, slopes = _logits(score_mod, q, k_tile, seen, rows[:, None], columns[None, :], params,
                                     factor, slopes=True)
            exponentials, weights = algebra.normalized(logits, m, denominator, how)
            cotangents = algebra.grad_scores(exponentials, _dot(do, v_tile.T), anchor[row, head, rows],
                                             delta[row, head, rows], seen,
                                             columns[None, :] == key_index[row, head, rows][:, None],
                                             algebra.inverse(denominator))
            pulled, pieces = algebra.pulled(cotangents, slopes, seen, score_kinds)
            dv = dv + _dot(*algebra.value_operands(exponentials, weights, do, denominator))
            dk = dk + _dot((pulled * scale).astype(q.dtype).T, q)
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
            block_q: int, block_k: int, split: bool, natural: bool, initial: bool):
    """The Triton backward's outputs (`TritonBackward`), from this mainloop's two launches, recomputing the
    weights as a forward of the call's form computed them (`algebra.recompute`)."""
    del score_dtypes
    spans = start, end, query_ids, key_ids
    how = algebra.recompute(algebra.Forward("full", 0, ids=False, scale=scale, score_mod=score_mod,
                                            natural=natural, initial=initial))
    shared = {"score_mod": score_mod, "score_kinds": score_kinds, "scale": scale, "how": how,
              "block_q": block_q, "block_k": block_k}
    sums = (others, own, others_total, forward_total) if split else None
    dq, softmax, chosen, dq_grads = _dq_launch(query, key, value, dout, peak, total, initial_term, maximum,
                                               key_index, anchor_value, shift, sums, spans, score_params,
                                               **shared)
    delta, anchor, m, denominator, _ = softmax
    dk, dv, key_grads = _dkv_launch(query, key, value, dout, m, denominator, anchor, delta, chosen, spans,
                                    score_params, **shared)
    return (dq, *softmax, chosen, *dq_grads, dk, dv, *key_grads)


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, block_q: int, block_k: int, pad: int,
             residuals: tuple[jax.Array, jax.Array, jax.Array, Split] | None = None, split: bool = True,
             scale: float | None = None, natural: bool = NATURAL) -> Gradients:
    """The template's backward on this mainloop (`gradients`) at `block_q` by `block_k` tiles, the call
    padded as a backend whose largest tile is `pad` pads it, from the forward's `residuals` (this
    mainloop's `attend`'s where None); with `split`, D from the forward's split sums, as the Triton
    mainloop's `backward`, `natural` keeping the forward's logits natural."""
    if residuals is None:
        plain = Hooks() if hooks is None else Hooks(hooks.score_mod, hooks.score_params, hooks.initial)
        _, maxima, totals, anchors, sums = attend(query, key, value, spans, plain, block_q=block_q,
                                                  block_k=block_k, pad=pad, scale=scale, residuals=True,
                                                  natural=natural)
        residuals = maxima, totals, anchors, sums
    initial = hooks is not None and hooks.initial is not None
    # A guarded forward's live rows carry normalized split sums (`algebra.split_residuals`), which the first
    # pass cannot recompute without the forward's denominator: D comes from them.
    split = split or (natural and algebra.headroom(value.dtype, key.shape[1], initial))
    launch = functools.partial(_launch, block_q=block_q, block_k=block_k, split=split, natural=natural,
                               initial=initial)
    return gradients(launch, query, key, value, spans, hooks, dout, residuals, block=pad, scale=scale)
