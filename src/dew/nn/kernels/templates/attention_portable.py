"""The attention template's plain-JAX mainloop (docs/design/kernel-templates.md §9): the algebra of
`attention_algebra` run over the same tiles, in the same order and with the same products as a kernel's
mainloop, in XLA ops, so it runs on any backend and is the boundary test's second arm
(tests/test_attention_algebra.py: the Triton mainloop, interpreted, and this one agree to the bit at the
same tiles). It takes a call at its launches' padded shapes, as `TritonForward` does."""

import functools

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates import attention_algebra as algebra
from dew.nn.kernels.templates.attention import (
    DKV_GRADS,
    DQ_GRADS,
    Gradients,
    Hooks,
    KeySpans,
    Kind,
    Split,
    block_map,
    gradients,
    padded_call,
)


def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
    """`a @ b` accumulated in fp32, the product an interpreted kernel takes."""
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


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

        def softmax(j, state: algebra.Partial):
            columns = j * block_k + jnp.arange(block_k)
            coords = algebra.Coords(q_pos, columns[None, :], **narrowing,
                                    key_ids=None if ids is None else ids[1][row, columns])
            params = tuple(_tile_param(param, kind, row, head, rows, columns)
                           for param, kind in zip(score_params, score_kinds, strict=True))
            k_tile = jax.lax.dynamic_slice_in_dim(key[row, :, kv_head], j * block_k, block_k)
            logits = algebra.modify(spec, _dot(q, k_tile.T), coords, params)
            m, alpha = algebra.peak(spec, state.m, logits)
            weights = algebra.weights(spec, logits, m)
            v_tile = jax.lax.dynamic_slice_in_dim(value[row, :, kv_head], j * block_k, block_k)
            return coords, logits, m, alpha, weights, algebra.total(state.total, alpha, weights), v_tile

        def plain(j, state: algebra.Partial) -> algebra.Partial:
            _, _, m, alpha, weights, denominator, v_tile = softmax(j, state)
            acc = algebra.rescale(state.acc, alpha) + _dot(algebra.narrow(weights, v_tile.dtype), v_tile)
            return algebra.Partial(acc, m, denominator)

        def step(j, carry: tuple[algebra.Partial, algebra.Argmax, jax.Array, jax.Array]):
            state, arg, own, others_total = carry
            coords, logits, m, alpha, weights, denominator, v_tile = softmax(j, state)
            # The training forward's split sums (`algebra.split_accumulate`), as the Triton mainloop's.
            anchor = algebra.argmax(arg, logits, coords.k_pos)
            others, own_tile = algebra.anchor_split(algebra.narrow(weights, v_tile.dtype),
                                                    coords.k_pos == anchor.key[:, None])
            acc, own, others_total = algebra.split_accumulate(
                state.acc, own, others_total, alpha, anchor.peak > arg.peak, value[row, arg.key, kv_head],
                _dot(others, v_tile), others, own_tile)
            return algebra.Partial(acc, m, denominator), anchor, own, others_total

        if initial is None:
            state = algebra.empty(block_q, value_width)
        else:
            m0, total0, acc0 = initial
            state = algebra.resumed(spec, m0[row, head, rows], total0[row, head, rows], acc0[row, head, rows])
        if spec.form == "spans":
            lo, hi = blocks_map.lo[row, i], blocks_map.hi[row, i]
        elif spec.form == "causal":
            lo, hi = 0, jnp.minimum(-(-(first_query + block_q) // block_k), tiles)
        else:
            lo, hi = 0, tiles
        out_params = tuple(_output_param(param, kind, row, head, rows)
                           for param, kind in zip(output_params, output_kinds, strict=True))
        if residuals:
            nothing = jnp.zeros((block_q,), jnp.float32)
            carry = state, algebra.no_argmax(block_q), nothing, nothing
            split, arg, own, others_total = jax.lax.fori_loop(lo, hi, step, carry)
            state = split._replace(acc=algebra.split_output(split.acc, own, value[row, arg.key, kv_head]))
            kept = arg.key, split.acc, own, others_total
        else:
            state = jax.lax.fori_loop(lo, hi, plain, state)
            kept = ()
        out, m, denominator = algebra.finish(spec, state, initial=initial is not None, output_mod=output_mod,
                                             output_params=out_params)
        return (out.astype(query.dtype), m, denominator, *kept) if residuals else out.astype(query.dtype)

    mapped = jax.lax.map(program, jnp.arange(blocks * batch * heads))
    out = mapped[0] if residuals else mapped
    # [blocks * batch * heads, bq, ...] back to [B, Qp, H, ...] and [B, H, Qp].
    out = out.reshape(blocks, batch, heads, block_q, value_width).transpose(1, 0, 3, 2, 4)
    out = out.reshape(batch, queries, heads, value_width)

    if not residuals:
        return out, None
    _, m, denominator, anchors, others, own, others_total = mapped

    def rowwise(x):
        return x.reshape(blocks, batch, heads, block_q).transpose(1, 2, 0, 3).reshape(batch, heads, queries)

    others = others.reshape(blocks, batch, heads, block_q, value_width).transpose(1, 2, 0, 3, 4)
    split = Split(others.reshape(batch, heads, queries, value_width), rowwise(own), rowwise(others_total))
    return out, (rowwise(m), rowwise(denominator), rowwise(anchors), split)


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
           *, block_q: int, block_k: int, pad: int, scale: float | None = None, residuals: bool = False):
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and `[B, K, Hk, Dv]` and the keys `spans` leaves each query, padded as a backend
    whose largest tile is `pad` pads it (`padded_call`) and walked in `block_q` by `block_k` tiles; with
    `residuals`, the running maxima and denominators `[B, H, Q]`, the anchor keys and the split sums too."""
    hooks = Hooks() if hooks is None else hooks
    call = padded_call(query, key, value, spans, hooks, pad)
    queries, value_width = call.queries, value.shape[-1]
    initial = None
    if hooks.initial is not None:
        state = hooks.initial
        initial = (call.rows(state.m, -jnp.inf), call.rows(state.total),
                   jnp.pad(call.rows(state.acc), ((0, 0),) * 3 + ((0, call.value.shape[-1] - value_width),)))
    spec = algebra.Forward(spans.form, spans.keys, call.query_ids is not None,
                           query.shape[3] ** -0.5 if scale is None else scale, hooks.score_mod)
    out, kept = forward(
        spec, call.query, call.key, call.value, call.start, call.end, call.query_ids, call.key_ids,
        call.score_params, tuple(param.kind for param in hooks.score_params), call.output_params,
        tuple(param.kind for param in hooks.output_params), initial, block_q=block_q, block_k=block_k,
        output_mod=hooks.output_mod, residuals=residuals)
    if kept is None:
        return out[:, :queries, :, :value_width]
    m, denominator, anchors, split = kept
    sums = Split(split.others[:, :, :queries, :value_width], split.own[:, :, :queries],
                 split.others_total[:, :, :queries])
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


def _dq_launch(query, key, value, dout, peak, total, initial_term, maximum, key_index, anchor_value, sums,
               spans, score_params, *, score_mod, score_kinds: tuple[Kind, ...], scale: float, block_q: int,
               block_k: int):
    """The Triton mainloop's dQ launch (`attention_triton._dq_launch`), a query block and head at a time:
    the first pass at the forward's maximum to each query's denominator and `D - c`
    (`algebra.off_anchor_delta`), then dQ and the dQ program's hook gradients under it, `D` refined once
    from its own sums (`algebra.refinement`)."""
    batch, queries, heads, width = query.shape
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
                                                                    value.shape[-1])
        if sums is None:
            # D without the forward's split sums: the dQ launch's own pass over the keys.
            def row_terms(j, state):
                columns, seen, params, k_tile, v_tile = tile(j)
                logits, _ = _logits(score_mod, q, k_tile, seen, q_pos, columns[None, :], params, scale)
                at_anchor = columns[None, :] == chosen[:, None]
                exponentials, rest, own = algebra.off_anchor_weights(logits, maxima, at_anchor, v_tile.dtype)
                return algebra.off_anchor(state, exponentials, rest, own, _dot(rest, v_tile))

            walked = jax.lax.fori_loop(lo, hi, row_terms, before)
            prior = walked.total
            anchor, delta, chosen = algebra.off_anchor_delta(walked, anchor_rows, do, state_total,
                                                             state_share, chosen)
        else:
            # The forward took the first pass (`algebra.split_delta`).
            others, own, others_total, prior = (part[row, head, rows] for part in sums)
            anchor, delta, chosen = algebra.split_delta(others, own, others_total, prior, state_total,
                                                        anchor_rows, do, chosen)
        m = maxima
        over_prior = algebra.inverse(prior)

        def step(j, carry):
            # Against the unnormalized weights, taken to the pass's own denominator at the end
            # (`algebra.dq_finish`), dQ's operands over the prior one before they narrow, as the Triton
            # mainloop's dQ launch.
            dq, residual, mass, grads, weighed = carry
            columns, seen, params, k_tile, v_tile = tile(j)
            logits, slopes = _logits(score_mod, q, k_tile, seen, q_pos, columns[None, :], params, scale,
                                     slopes=True)
            weights = algebra.normalized(logits, m, jnp.ones_like(m))[0]
            cotangents = algebra.grad_scores(weights, _dot(do, v_tile.T), anchor, delta, seen,
                                             columns[None, :] == chosen[:, None])
            pulled, pieces = algebra.pulled(cotangents, slopes, seen, score_kinds)
            dq = dq + _dot((pulled * scale * over_prior[:, None]).astype(k_tile.dtype), k_tile)
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
                score_mod, score_kinds: tuple[Kind, ...], scale: float, block_q: int, block_k: int):
    """The Triton mainloop's dK/dV launch (`attention_triton._dkv_launch`), a key block and query head at a
    time, under the dQ launch's softmax at its tiles: the query head's dK, dV `[B, Kp, H, ...]` and the
    key and tile parameters' gradients (`DKV_GRADS`, `[B, H, Kp]` and `[B, H, Qp, Kp]`)."""
    batch, queries, heads, width = query.shape
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
                                     scale, slopes=True)
            exponentials, weights = algebra.normalized(logits, m, denominator)
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


def _launch(query, key, value, dout, peak, total, initial_term, maximum, key_index, anchor_value, others, own,
            others_total, forward_total, start, end, query_ids, key_ids, score_params, *, score_mod,
            score_kinds: tuple[Kind, ...], score_dtypes: tuple[str, ...], scale: float, block_q: int,
            block_k: int, split: bool):
    """The Triton backward's outputs (`TritonBackward`), from this mainloop's two launches."""
    del score_dtypes
    spans = start, end, query_ids, key_ids
    shared = {"score_mod": score_mod, "score_kinds": score_kinds, "scale": scale, "block_q": block_q,
              "block_k": block_k}
    sums = (others, own, others_total, forward_total) if split else None
    dq, softmax, chosen, dq_grads = _dq_launch(query, key, value, dout, peak, total, initial_term, maximum,
                                               key_index, anchor_value, sums, spans, score_params, **shared)
    delta, anchor, m, denominator, _ = softmax
    dk, dv, key_grads = _dkv_launch(query, key, value, dout, m, denominator, anchor, delta, chosen, spans,
                                    score_params, **shared)
    return (dq, *softmax, chosen, *dq_grads, dk, dv, *key_grads)


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, block_q: int, block_k: int, pad: int,
             residuals: tuple[jax.Array, jax.Array, jax.Array, Split] | None = None, split: bool = True,
             scale: float | None = None) -> Gradients:
    """The template's backward on this mainloop (`gradients`) at `block_q` by `block_k` tiles, the call
    padded as a backend whose largest tile is `pad` pads it, from the forward's `residuals` (this
    mainloop's `attend`'s where None); with `split`, D from the forward's split sums, as the Triton
    mainloop's `backward`."""
    if residuals is None:
        plain = Hooks() if hooks is None else Hooks(hooks.score_mod, hooks.score_params, hooks.initial)
        _, maxima, totals, anchors, sums = attend(query, key, value, spans, plain, block_q=block_q,
                                                  block_k=block_k, pad=pad, scale=scale, residuals=True)
        residuals = maxima, totals, anchors, sums
    return gradients(functools.partial(_launch, block_q=block_q, block_k=block_k, split=split), query, key,
                     value, spans, hooks, dout, residuals, block=pad, scale=scale)
