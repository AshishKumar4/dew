"""Log-linear attention (row 1446; fla's log_linear_attn, its Mamba-2 backbone): a decayed linear attention
whose key at `c` reaches the query at `l` through the level of the Fenwick tree that holds it, each query
weighing its levels apart,

    o_l = scale sum_{c <= l} exp(G_l - G_c) lambda_l[level(l, c)] (q_l . k_c) v_c,

`G` the log decay's running sum, `level(l, c)` zero for `c = l` and else the bit length of `l XOR c` (the
highest bit in which they differ: the bucket `[l` with its low `j` bits cleared, `l` with its low `j - 1`
cleared`)` at level `j`), `lambda` a token's level weights `[levels]`. Grouped keys serve `H / G` value heads
each; packed documents each count positions from their first token and see their own keys alone.

An orchestrator over the recurrence (kernel-templates.md, Table 3): level `j` is the decayed linear
recurrence (`recurrence_chunk.chunk`, a decay a head, no delta rule) over the keys in the left half of each
`2^j`-aligned block of a document's positions, its memory reset where each block starts, read by the queries
in the right half at their weight `lambda[j]`; level zero is a token's own key. Each level is one linear pass
over the sequence, `ceil(log2 T) + 1` of them, where a masked attention over every query and key took `T^2`
of each array (a 4096-token call's scores alone 64 MiB in fp32)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.precision import at_least_fp32


def levels(length: int) -> int:
    """The level weights a token of a `length`-token sequence takes, fla's `ceil(log2 T) + 1`."""
    return math.ceil(math.log2(length)) + 1


def log_linear_attention(query, key, value, g, level_scales, *, scale: float | None = None, starts=None,
                         max_segments: int | None = None):
    """`query`, `key` `[B, T, G, N]`, `value` `[B, T, H, P]`, `g` `[B, T, H]` a token's log decay,
    `level_scales` `[B, T, H, levels(T)]` each query's weight of each level, `scale` `N ** -0.5` where None,
    `starts` `[B, T]` each packed document's first token, at most `max_segments` a row (every token where
    None): the output `[B, T, H, P]` in the operands' dtype, computed in at least fp32, a recurrence a level,
    level `j`'s at most `ceil(T / 2^j)` blocks a row past its documents (`chunk`'s `max_segments`, which a
    kernel laying each from a chunk's edge sizes its layout by)."""
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
    rows, length, groups, width = query.shape
    heads = value.shape[2]
    scale = width ** -0.5 if scale is None else scale
    q, k, v, g, weights = (jnp.asarray(x, work) for x in (query, key, value, g, level_scales))
    tokens = jnp.arange(length, dtype=jnp.int32)
    first = (jnp.zeros((rows, length), jnp.int32) if starts is None
             else jax.lax.cummax(jnp.where(jnp.asarray(starts, bool), tokens, 0), axis=1))
    # Each token's position in its document, which its levels' blocks align to.
    positions = tokens - first
    # Level zero: each token's own key.
    grouped = jnp.repeat(q * k, heads // groups, axis=2).sum(-1)
    out = (scale * weights[..., 0] * grouped)[..., None] * v
    hooks = recurrence.Hooks(decay='scalar', delta=False, scale=scale)
    documents = 1 if starts is None else length if max_segments is None else max_segments
    for level in range(1, levels(length)):
        half = 1 << (level - 1)
        # A key in its block's left half; a query in the right half reads them; a block starts afresh.
        left, opened = (positions & half) == 0, positions % (2 * half) == 0
        blocks = min(length, -(-length // (2 * half)) + documents)
        read, _ = chunk(hooks, Chunk(q, k * left[..., None, None], v, log_decay=g, starts=opened),
                        max_segments=blocks, dtype=work)
        out = out + read * jnp.where(left[..., None], 0.0, weights[..., level])[..., None]
    return out.astype(dtype)
