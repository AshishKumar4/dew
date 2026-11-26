"""Log-linear attention (row 1446; fla's log_linear_attn, its Mamba-2 backbone): a decayed linear attention
whose key at `c` reaches the query at `l` through the level of the Fenwick tree that holds it, each query
weighing its levels apart,

    o_l = scale sum_{c <= l} exp(G_l - G_c) lambda_l[level(l, c)] (q_l . k_c) v_c,

`G` the log decay's running sum, `level(l, c)` zero for `c = l` and else the bit length of `l XOR c` (the
highest bit in which they differ: the bucket `[l` with its low `j` bits cleared, `l` with its low `j - 1`
cleared`)` at level `j`), `lambda` a token's level weights `[levels]`. Grouped keys serve `H / G` value heads
each; packed documents each count positions from their first token and see their own keys alone.

An orchestrator over the recurrence and a blocked attention (kernel-templates.md, Table 3 and Appendix B):
the pairs within a `WITHIN`-aligned block of a document's positions (levels zero to `log2 WITHIN`) are one
masked attention over each block, each pair at its query's weight of its level (`_within`); level `j` above
is the decayed linear recurrence (`recurrence_chunk.chunk`, a decay a head, no delta rule) over the keys in
the left half of each `2^j`-aligned block, its memory reset where each block starts, read by the queries in
the right half at their weight `lambda[j]`. Each level above is one linear pass over the sequence, where a
masked attention over every query and key took `T^2` of each array (a 4096-token call's scores alone 64 MiB
in fp32); a level within, as a recurrence, laid each of its blocks as a chunk of its own, level one's
2-token blocks 32 times its tokens in 64-token chunks (row 1446)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra import scan
from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.precision import at_least_fp32

WITHIN = 64
"""The blocks of a document's positions whose pairs one masked attention forms (`_within`), levels zero to
`log2 WITHIN`: a recurrence's chunk (64 tokens), so every level above walks blocks of two chunks or more."""


def levels(length: int) -> int:
    """The level weights a token of a `length`-token sequence takes, fla's `ceil(log2 T) + 1`."""
    return math.ceil(math.log2(length)) + 1


def log_linear_attention(query, key, value, g, level_scales, *, scale: float | None = None, starts=None,
                         max_segments: int | None = None):
    """`query`, `key` `[B, T, G, N]`, `value` `[B, T, H, P]`, `g` `[B, T, H]` a token's log decay,
    `level_scales` `[B, T, H, levels(T)]` each query's weight of each level, `scale` `N ** -0.5` where None,
    `starts` `[B, T]` each packed document's first token, at most `max_segments` a row (every token where
    None): the output `[B, T, H, P]` in the operands' dtype, computed in at least fp32: the levels within a
    `WITHIN` block one masked attention, a recurrence a level above, level `j`'s at most `ceil(T / 2^j)`
    blocks a row past its documents (`chunk`'s `max_segments`, which a kernel laying each from a chunk's edge
    sizes its layout by)."""
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
    rows, length, _, width = query.shape
    scale = width ** -0.5 if scale is None else scale
    q, k, v, g, weights = (jnp.asarray(x, work) for x in (query, key, value, g, level_scales))
    tokens = jnp.arange(length, dtype=jnp.int32)
    first = (jnp.zeros((rows, length), jnp.int32) if starts is None
             else jax.lax.cummax(jnp.where(jnp.asarray(starts, bool), tokens, 0), axis=1))
    # Each token's position in its document, which its levels' blocks align to.
    positions = tokens - first
    documents = 1 if starts is None else length if max_segments is None else max_segments
    out = _within(q, k, v, g, weights, positions, scale=scale, documents=documents, packed=starts is not None)
    hooks = recurrence.Hooks(decay='scalar', delta=False, scale=scale)
    for level in range(WITHIN.bit_length(), levels(length)):
        half = 1 << (level - 1)
        # A key in its block's left half; a query in the right half reads them; a block starts afresh.
        left, opened = (positions & half) == 0, positions % (2 * half) == 0
        blocks = min(length, -(-length // (2 * half)) + documents)
        # The level's recurrence takes the operands as given, its output in at least fp32 (`chunk`'s
        # `dtype`): a kernel takes a 16-bit call's products whole. Widened first, every level was an fp32
        # call, which no GPU's template took (an A100's portable math; a GH200's output and states launches
        # past their registers in every config, M-JS-125b at 3cd06f7f305).
        read, _ = chunk(hooks, Chunk(query, key * left[..., None, None], value, log_decay=g, starts=opened),
                        max_segments=blocks, dtype=work)
        out = out + read * jnp.where(left[..., None], 0.0, weights[..., level])[..., None]
    return out.astype(dtype)


def _levels(count: int) -> np.ndarray:
    """`[L, WITHIN, WITHIN]` fp32: each pair of a block's slots at its level (`_within`'s), one-hot over the
    `L` levels its pairs reach, of a sequence's `count` (`levels`). A query's weights taken at its pairs'
    levels by a product with them (exact: one weight times one, the others times zero), whose gradient is a
    product too: gathered by the levels' indices, the gradient a scatter, which XLA formed as a select and
    reduce over every pair and level, 3.0 ms of a G4's 9.1 ms log-linear Mamba-2 training step against fla's
    2.3 ms whole (M-JS-275)."""
    offsets = np.arange(WITHIN)
    level = np.minimum(np.frexp(np.bitwise_xor.outer(offsets, offsets))[1], count - 1)
    return (level[None] == np.arange(level.max() + 1)[:, None, None]).astype(np.float32)


def _within(q, k, v, g, weights, positions, *, scale: float, documents: int, packed: bool):
    """The pairs within each `WITHIN`-aligned block of a document's `positions` `[B, T]`, levels zero to
    `log2 WITHIN`: each token laid at its block's slot (a row's blocks in order, at most `ceil(T / WITHIN)`
    and its `documents` more), each block's reads the derived scan's (`scan.spread`, its decays
    `scan.between`), each pair's read times its query's weight of the pair's level, its output read back at
    each token's slot; where no document is `packed` the row's tokens are its slots, its blocks the row padded
    to whole blocks (a scatter and a gather of every operand and the output took 1.7 ms of a GH200's 4.7 ms
    forward at Kimi's 64 value heads over 4096 tokens, M-JS-146ll)."""
    rows, length, groups, _ = q.shape
    heads = v.shape[2]
    if packed:
        offset = positions % WITHIN
        count = min(length, -(-length // WITHIN) + documents)
        # A token's slot: its block (a row's blocks in order, one opening at each multiple of WITHIN) and its
        # offset.
        slot = (jnp.cumsum(offset == 0, axis=1) - 1) * WITHIN + offset
        each = jnp.arange(rows)[:, None]

        def laid(x):
            return jnp.zeros((rows, count * WITHIN, *x.shape[2:]), x.dtype).at[each, slot].set(x)

        def unlaid(x):
            return x[each, slot]
    else:
        count = -(-length // WITHIN)

        def laid(x):
            return jnp.pad(x, [(0, 0), (0, count * WITHIN - length), *[(0, 0)] * (x.ndim - 2)])

        def unlaid(x):
            return x[:, :length]

    def blocked(x):
        return laid(x).reshape(rows * count, WITHIN, *x.shape[2:])

    def grouped(x):
        # A value head's operand beside its key head's: `[blocks, WITHIN, G, H / G, ...]`.
        return x.reshape(*x.shape[:2], groups, heads // groups, *x.shape[3:])

    q, k = (blocked(x) for x in (q, k))
    v, g, weights = (grouped(blocked(x)) for x in (v, g, weights))
    # A pair's level: zero for a token's own key, else the bit length of their offsets' XOR (the exponent
    # `frexp` gives, 1 for 1, 0 for 0); past the sequence's levels, a pair of slots no token takes. Each
    # pair's weight its query's at its level, a product with the levels' one-hot masks (`_levels`).
    masks = _levels(weights.shape[-1])

    def head(query, key, value, decay, weight):
        pairs = jnp.einsum('il,lij->ij', weight[:, :len(masks)], masks, precision=jax.lax.Precision.HIGHEST)
        return scan.spread(query, key, value, scan.between(decay[:, None]), weights=pairs)

    # A key head's value heads, its queries and keys unmapped among them, so their scores are formed once a
    # key head (each value head repeating them formed log-linear-mamba2's 64 a block, an fp32 product of 234
    # us in a GH200's forward, M-JS-146ll); then a block's key heads, then every row's blocks.
    value_heads = jax.vmap(head, in_axes=(None, None, 1, 1, 1), out_axes=1)
    out = jax.vmap(jax.vmap(value_heads, in_axes=1, out_axes=1))(q, k, v, g, weights)
    return unlaid((scale * out).reshape(rows, count * WITHIN, heads, -1))
