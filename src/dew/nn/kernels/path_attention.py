"""PaTH attention (row 1460; fla's `parallel_path_attn`, fla/ops/path_attn): softmax attention whose score
between a query and an earlier key runs through the generalized Householder transforms of the tokens between
them, plus a forget gate's log decay,

    s_ij = scale (q_i^T H_i H_{i-1} ... H_{j+1} k_j + G_i - G_j),   H_t = I - beta_t w_t w_t^T,
    o_i  = sum_{j <= i} softmax_j(s_ij) v_j,

`G` the gate's running sum (fla scales it with the scores). A chunk's transforms compose into one
`I - W^T T (W beta)`, `T = (I + tril(beta W W^T, -1))^-1` a unit lower-triangular solve (the UT form, as
fla's naive reference and the delta rule's chunks take it), so the queries are carried to their chunk's
start, the keys to their chunk's end, and a query chunk walks the key chunks before it from the nearest,
its queries carried through each (`path_attention`); packed documents take the softmax within their own
tokens."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from dew.nn.kernels.deltaformer import causal_masks, masked_softmax
from dew.nn.precision import at_least_fp32


def _headed(t, heads: int, work):
    """`t` `[B, T, H, ...]` in `work`, heads first `[B, H', T, ...]`, its `H` heads repeated to `heads` (a
    key head's tensors for each query head it serves)."""
    t = jnp.moveaxis(jnp.asarray(t, work), 2, 1)
    return jnp.repeat(t, heads // t.shape[1], axis=1)


def _attended(scores, value, gate, scale: float, starts):
    """The output `[B, H, T, Dv]` of `scores` `[B, H, T, T]` (each query against each key, before its scale),
    the gate's log decay `[B, H, T]` (None for none) added as its running sums' difference and scaled with
    them, a causal softmax within each packed document (`starts`)."""
    if gate is not None:
        running = jnp.cumsum(gate, axis=-1)
        scores = scores + running[..., :, None] - running[..., None, :]
    _, causal = causal_masks(scores.shape[-1], starts)
    return jnp.einsum('bhts,bhsd->bhtd', masked_softmax(scores * scale, causal), value)


def path_attention(query, key, value, w, beta, gate=None, *, scale: float | None = None, starts=None,
                   chunk: int = 64):
    """PaTH attention: `query` `[B, T, HQ, D]`, `key` and `w` `[B, T, H, D]`, `value` `[B, T, H, Dv]`,
    `beta` `[B, T, H]`, `gate` `[B, T, HQ]` a token's log decay (None for none), `scale` `D ** -0.5` where
    None, `starts` `[B, T]` each packed document's first token; a key head serves `HQ / H` query heads. Its
    transforms, scores and softmax in `at_least_fp32` of the operands' dtype a `chunk` of tokens at a time,
    the output `[B, T, HQ, Dv]` in it."""
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
    batch, length, heads, width = query.shape
    scale = width ** -0.5 if scale is None else scale
    padded = -(-length // chunk) * chunk
    q, k, v, w = (_headed(t, heads, work) for t in (query, key, value, w))
    b = _headed(beta, heads, work)
    # Padded tokens' transforms the identity (beta zero), their scores masked.
    q, k, w = (jnp.pad(t, ((0, 0), (0, 0), (0, padded - length), (0, 0))) for t in (q, k, w))
    b = jnp.pad(b, ((0, 0), (0, 0), (0, padded - length)))
    chunks = padded // chunk
    q, k, w = (t.reshape(batch, heads, chunks, chunk, width) for t in (q, k, w))
    weighted = w * b.reshape(batch, heads, chunks, chunk)[..., None]
    lower = jnp.tril(jnp.ones((chunk, chunk), bool), -1)
    eye = jnp.eye(chunk, dtype=work)
    # T (W beta) and T (W beta k^T, strictly lower), by one unit lower-triangular solve each chunk.
    system = eye + jnp.where(lower, weighted @ jnp.swapaxes(w, -1, -2), 0.0)
    solved = solve_triangular(system, jnp.concatenate(
        [weighted, jnp.where(lower, weighted @ jnp.swapaxes(k, -1, -2), 0.0)], axis=-1),
        lower=True, unit_diagonal=True)
    t_weighted, t_keys = solved[..., :width], solved[..., width:]
    queries_w = jnp.tril(q @ jnp.swapaxes(w, -1, -2))
    local = jnp.tril(q @ jnp.swapaxes(k, -1, -2)) - queries_w @ t_keys
    carried_q = q - queries_w @ t_weighted
    carried_k = k - jnp.swapaxes(t_keys, -1, -2) @ w
    # Each chunk's transforms `I - W^T T (W beta)`, through which a later chunk's queries pass it.
    passed = jnp.swapaxes(w, -1, -2) @ t_weighted

    later = jnp.arange(chunks)[:, None, None]

    def walked(queries, m):
        # Key chunk `m` against every query chunk's queries carried to it; then the later chunks' queries
        # carried past it.
        block = queries @ jnp.swapaxes(carried_k[:, :, m], -1, -2)[:, :, None]
        return queries - jnp.where(later > m, queries @ passed[:, :, m][:, :, None], 0.0), block

    _, blocks = jax.lax.scan(walked, carried_q, jnp.arange(chunks - 1, -1, -1))
    # blocks[r][..., n, :, :] is query chunk n against key chunk chunks - 1 - r; the causal mask drops n < m.
    blocks = jnp.flip(blocks, axis=0)
    diagonal = jnp.arange(chunks)[:, None] == jnp.arange(chunks)[None, :]
    blocks = jnp.where(diagonal[:, None, None, :, None, None], local[None], blocks)
    scores = jnp.moveaxis(blocks, 0, 4).reshape(batch, heads, chunks, chunk, padded)
    scores = scores.reshape(batch, heads, padded, padded)[:, :, :length, :length]
    out = _attended(scores, v, None if gate is None else jnp.moveaxis(jnp.asarray(gate, work), 2, 1), scale,
                    starts)
    return jnp.moveaxis(out, 1, 2).astype(dtype)


def recurrent_path_attention(query, key, value, w, beta, gate=None, *, scale: float | None = None,
                             starts=None):
    """`path_attention` from its definition a lag at a time: each query carried back through one token's
    transform per step, `r <- r - beta_j w_j (w_j . r)`, its score against key `j` read before it passes
    `H_j`; in at least fp32 (float64 where the operands are); operands and result as `path_attention`'s."""
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
    _, length, heads, width = query.shape
    scale = width ** -0.5 if scale is None else scale
    q, k, v, w = (_headed(t, heads, work) for t in (query, key, value, w))
    b = _headed(beta, heads, work)
    tokens = jnp.arange(length)

    def lag(carried, d):
        # Each query `i` against key `i - d`, then carried past that token's transform.
        j = jnp.clip(tokens - d, 0, length - 1)
        keys, ws, bs = k[:, :, j], w[:, :, j], b[:, :, j]
        score = jnp.sum(carried * keys, axis=-1)
        passed = carried - (bs * jnp.sum(ws * carried, axis=-1))[..., None] * ws
        return passed, score

    _, by_lag = jax.lax.scan(lag, q, tokens)
    lags = jnp.clip(tokens[:, None] - tokens[None, :], 0, length - 1)
    scores = jnp.moveaxis(by_lag, 0, -1)[:, :, tokens[:, None], lags]
    out = _attended(scores, v, None if gate is None else jnp.moveaxis(jnp.asarray(gate, work), 2, 1), scale,
                    starts)
    return jnp.moveaxis(out, 1, 2).astype(dtype)
