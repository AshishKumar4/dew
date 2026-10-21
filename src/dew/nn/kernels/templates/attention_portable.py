"""The attention template's plain-JAX mainloop (docs/design/kernel-templates.md §9): the algebra of
`attention_algebra` run over the same tiles, in the same order and with the same products as a kernel's
mainloop, in XLA ops, so it runs on any backend and is the boundary test's second arm
(tests/test_attention_algebra.py: the Triton mainloop, interpreted, and this one agree to the bit at the
same tiles). It takes a call at its launches' padded shapes, as `TritonForward` does."""

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates import attention_algebra as algebra
from dew.nn.kernels.templates.attention import Hooks, KeySpans, Kind, block_map, padded_call


def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
    """`a @ b` accumulated in fp32, the product an interpreted kernel takes."""
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def _tile_param(value: jax.Array, kind: Kind, row, head, rows: jax.Array, keys: jax.Array | None):
    """A hook parameter's values for a tile of query rows `rows` and keys `keys` (index arrays) of batch
    row `row` and query head `head`, as `kind` lays them out: `_tile_param` in the Triton mainloop."""
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
            return value[row, rows, head]


def forward(spec: algebra.Forward, query, key, value, start, end, query_ids, key_ids, score_params,
            score_kinds: tuple[Kind, ...], output_params, output_kinds: tuple[Kind, ...], initial, *,
            block_q: int, block_k: int, output_mod=None, residuals: bool = False):
    """`TritonForward._fwd`'s outputs from its operands (`query` `[B, Qp, H, Dp]`, `key` and `value`
    `[B, Kp, Hk, ...]`, the spans' rows and ids, the hooks' padded parameters, the initial state
    `(m0, total0, acc0)` or None), one query block and head at a time in sequence, its key tiles walked as
    the Triton mainloop walks them."""
    batch, queries, heads, _ = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    blocks = queries // block_q
    bounds = None
    if spec.form == "spans":
        blocks_map = block_map(KeySpans(start, end, query_ids, key_ids, keys), block_q, block_k)
        bounds = jnp.stack([blocks_map.lo, blocks_map.hi], axis=-1).astype(jnp.int32)
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
        if spec.ids:
            narrowing["query_ids"] = query_ids[row, rows]

        def step(j, state):
            columns = j * block_k + jnp.arange(block_k)
            coords = algebra.Coords(q_pos, columns[None, :], **narrowing,
                                    key_ids=key_ids[row, columns] if spec.ids else None)
            params = tuple(_tile_param(param, kind, row, head, rows, columns)
                           for param, kind in zip(score_params, score_kinds, strict=True))
            k_tile = jax.lax.dynamic_slice_in_dim(key[row, :, kv_head], j * block_k, block_k)
            logits = algebra.modify(spec, _dot(q, k_tile.T), coords, params)
            m, alpha = algebra.peak(spec, state.m, logits)
            weights = algebra.weights(spec, logits, m)
            denominator = algebra.total(state.total, alpha, weights)
            v_tile = jax.lax.dynamic_slice_in_dim(value[row, :, kv_head], j * block_k, block_k)
            acc = algebra.rescale(state.acc, alpha) + _dot(algebra.narrow(weights, v_tile.dtype), v_tile)
            return algebra.Partial(acc, m, denominator)

        if initial is None:
            state = algebra.empty(block_q, value_width)
        else:
            m0, total0, acc0 = initial
            state = algebra.resumed(spec, m0[row, head, rows], total0[row, head, rows], acc0[row, head, rows])
        if spec.form == "spans":
            lo, hi = bounds[row, i, 0], bounds[row, i, 1]
        elif spec.form == "causal":
            lo, hi = 0, jnp.minimum(-(-(first_query + block_q) // block_k), tiles)
        else:
            lo, hi = 0, tiles
        out_params = tuple(_tile_param(param, kind, row, head, rows, None)
                           for param, kind in zip(output_params, output_kinds, strict=True))
        state = jax.lax.fori_loop(lo, hi, step, state)
        out, m, denominator = algebra.finish(spec, state, initial=initial is not None, output_mod=output_mod,
                                             output_params=out_params)
        return out.astype(query.dtype), m, denominator

    out, m, denominator = jax.lax.map(program, jnp.arange(blocks * batch * heads))
    # [blocks * batch * heads, bq, ...] back to [B, Qp, H, ...] and [B, H, Qp].
    out = out.reshape(blocks, batch, heads, block_q, value_width).transpose(1, 0, 3, 2, 4)
    out = out.reshape(batch, queries, heads, value_width)
    if not residuals:
        return out

    def rowwise(x):
        return x.reshape(blocks, batch, heads, block_q).transpose(1, 2, 0, 3).reshape(batch, heads, queries)

    return out, rowwise(m), rowwise(denominator)


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
           *, block_q: int, block_k: int, pad: int, scale: float | None = None, residuals: bool = False):
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and `[B, K, Hk, Dv]` and the keys `spans` leaves each query, padded as a backend
    whose largest tile is `pad` pads it (`padded_call`) and walked in `block_q` by `block_k` tiles; with
    `residuals`, the running maxima and denominators `[B, H, Q]` too."""
    hooks = Hooks() if hooks is None else hooks
    call = padded_call(query, key, value, spans, hooks, pad)
    queries, value_width = call.queries, value.shape[-1]
    initial = None
    if hooks.initial is not None:
        state = hooks.initial
        initial = (call.rows(state.m, -jnp.inf), call.rows(state.total),
                   jnp.pad(call.rows(state.acc), ((0, 0),) * 3 + ((0, call.value.shape[-1] - value_width),)))
    spec = algebra.Forward(spans.form, spans.keys or call.key.shape[1], call.query_ids is not None,
                           query.shape[3] ** -0.5 if scale is None else scale, hooks.score_mod)
    out = forward(spec, call.query, call.key, call.value, call.start, call.end, call.query_ids, call.key_ids,
                  call.score_params, tuple(param.kind for param in hooks.score_params), call.output_params,
                  tuple(param.kind for param in hooks.output_params), initial, block_q=block_q,
                  block_k=block_k, output_mod=hooks.output_mod, residuals=residuals)
    if not residuals:
        return out[:, :queries, :, :value_width]
    out, m, denominator = out
    return out[:, :queries, :, :value_width], m[:, :, :queries], denominator[:, :, :queries]
