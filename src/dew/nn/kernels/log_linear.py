"""Log-linear attention (row 1446; fla's log_linear_attn, its Mamba-2 backbone): a decayed linear attention
whose key at `c` reaches the query at `l` through the level of the Fenwick tree that holds it, each query
weighing its levels apart,

    o_l = scale sum_{c <= l} exp(G_l - G_c) lambda_l[level(l, c)] (q_l . k_c) v_c,

`G` the log decay's running sum, `level(l, c)` zero for `c = l` and else the bit length of `l XOR c` (the
highest bit in which they differ: the bucket `[l` with its low `j` bits cleared, `l` with its low `j - 1`
cleared`)` at level `j`), `lambda` a token's level weights `[levels]`. Grouped keys serve `H / G` value heads
each; packed documents each count positions from their first token and see their own keys alone."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from dew.nn.documents import within_documents
from dew.nn.kernels.deltaformer import causal_masks
from dew.nn.precision import at_least_fp32


def levels(length: int) -> int:
    """The level weights a token of a `length`-token sequence takes, fla's `ceil(log2 T) + 1`."""
    return math.ceil(math.log2(length)) + 1


def _levels(positions):
    """Each query's level of each key `[B, T, T]`, of positions `[B, T]`: zero on the diagonal, else the bit
    length of the positions' exclusive or (keys after the query any, masked)."""
    differing = positions[:, :, None] ^ positions[:, None, :]
    return jnp.where(differing == 0, 0, 32 - jax.lax.clz(differing))


def log_linear_attention(query, key, value, g, level_scales, *, scale: float | None = None, starts=None):
    """`query`, `key` `[B, T, G, N]`, `value` `[B, T, H, P]`, `g` `[B, T, H]` a token's log decay,
    `level_scales` `[B, T, H, levels(T)]` each query's weight of each level, `scale` `N ** -0.5` where None,
    `starts` `[B, T]` each packed document's first token: the output `[B, T, H, P]` in the operands' dtype,
    computed in at least fp32 as a masked attention over the sequence."""
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
    _, length, groups, width = query.shape
    heads = value.shape[2]
    scale = width ** -0.5 if scale is None else scale
    q, k = (jnp.repeat(jnp.asarray(x, work), heads // groups, axis=2) for x in (query, key))
    v, g, weights = (jnp.asarray(x, work) for x in (value, g, level_scales))
    tokens = jnp.arange(length, dtype=jnp.int32)
    first = (jnp.zeros((query.shape[0], length), jnp.int32) if starts is None
             else jax.lax.cummax(jnp.where(jnp.asarray(starts, bool), tokens, 0), axis=1))
    _, causal = causal_masks(length, starts)
    # A key after its query, or in another document, at level zero: masked by its decay.
    level = jnp.where(causal, _levels(tokens - first)[:, None], 0)
    # Each document's own running sum, its differences free of earlier documents' digits.
    begins = jnp.zeros(g.shape[:2], bool) if starts is None else jnp.asarray(starts, bool)
    begins = begins.at[:, 0].set(True)
    running = within_documents(g, begins).transpose(0, 2, 1)
    decay = jnp.exp(jnp.where(causal, running[..., :, None] - running[..., None, :], -jnp.inf))
    weighed = jnp.take_along_axis(weights.transpose(0, 2, 1, 3), jnp.broadcast_to(level, decay.shape),
                                  axis=-1)
    scores = jnp.einsum('blhn,bchn->bhlc', q, k) * decay * weighed
    return (jnp.einsum('bhlc,bchp->blhp', scores, v) * scale).astype(dtype)
