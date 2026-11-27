"""Log-linear attention (row 1446; fla's log_linear_attn, its Mamba-2 backbone): a decayed linear attention
whose key at `c` reaches the query at `l` through the level of the Fenwick tree that holds it, each query
weighing its levels apart,

    o_l = scale sum_{c <= l} exp(G_l - G_c) lambda_l[level(l, c)] (q_l . k_c) v_c,

`G` the log decay's running sum, `level(l, c)` zero for `c = l` and else the bit length of `l XOR c` (the
highest bit in which they differ: the bucket `[l` with its low `j` bits cleared, `l` with its low `j - 1`
cleared`)` at level `j`), `lambda` a token's level weights `[levels]`. Grouped keys serve `H / G` value heads
each; packed documents each count positions from their first token and see their own keys alone.

An orchestrator over a blocked attention and the levels' memories (kernel-templates.md, Table 3 and Appendix
B): the pairs within a `WITHIN`-aligned block of a document's positions (levels zero to `log2 WITHIN`) are one
masked attention over each block, each pair at its query's weight of its level (`_within`); at level `j`
above, the keys in the left half of each `2^j`-aligned block write a memory that only the queries in its
right half read, at their weight `lambda[j]`. A bf16 call in the fast forms that packs no document takes
fla's chunked arithmetic (`_fla_levels`): a block's memory the left half's keys against its values decayed
to the half's end, one product a block, and its reads the right half's weighed queries against it decayed
from there, one more (`_levels_above`), no pair within a half and no walk of memories, which the recurrence
(`recurrence_chunk.chunk`, a decay a head, no delta rule, its memory reset at each block) spent its scores
and walks on, 896 us of a GH200's 2008 us forward against fla's 712 whole, its documents' layout 221 more
(M-JS-287). Any other call, and packed documents, whose blocks do not align to the row, take the
recurrence a level. Each level above is one linear pass over the sequence, where a masked attention over
every query and key took `T^2` of each array (a 4096-token call's scores alone 64 MiB in fp32); a level
within, as a recurrence, laid each of its blocks as a chunk of its own, level one's 2-token blocks 32 times
its tokens in 64-token chunks (row 1446)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra import scan
from dew.nn.kernels import generation, recurrence, selection
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
    `WITHIN` block one masked attention, the levels above two products a block where a bf16 call in the fast
    forms packs no document (`_fla_levels`, fla's chunked arithmetic), else a recurrence a level (the
    portable recurrence's range, fp32's, fp16's and the accurate forms' among them), level `j`'s at most
    `ceil(T / 2^j)` blocks a row past its documents (`chunk`'s `max_segments`, which a kernel laying each
    from a chunk's edge sizes its layout by)."""
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
    # fla's chunked form scales the within and the levels' reads once, summed (its `b_o * SCALE`): scaled
    # apart, a pair of them cancelling read infinity less infinity.
    fla = starts is None and _fla_levels(dtype)
    out = _within(q, k, v, g, weights, positions, scale=1.0 if fla else scale, documents=documents,
                  packed=starts is not None)
    if fla:
        return (scale * _levels_above(query, key, value, g, weights, within=out)).astype(dtype)
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


def _fla_levels(dtype) -> bool:
    """Whether a call's levels above `WITHIN` take fla's chunked arithmetic (`_levels_above`): a bf16 call in
    the fast forms (`generation.numerics`), fast by default and accurate one option away
    (docs/design/testing.md). Its range is fla's chunked form's and no more (a bf16 query of 2^127 at a
    weight past 1 reads infinity there as in fla's); every other call, fp32's, fp16's (whose range a reader
    or a memory rounded to it leaves) and any in the accurate forms, takes the recurrence a level, the
    portable recurrence's range: a values' decay applied to its memory, not to each value before its key
    (a key of 2^100 under three decays of -40 kept, where `e^-120` on its unit value rounded to zero). A
    call within `generation.chosen` naming the recurrence's kernel (`gated_delta_rule`) takes that kernel a
    level, as an A/B of its kernels asks (tools/benchmark_recurrence.py's `mosaic` and `xla` arms)."""
    return (jnp.dtype(dtype) == jnp.bfloat16 and generation.numerics().forms != 'accurate'
            and selection.forced(generation.Choice.GATED_DELTA_RULE) is None)


def _levels_above(query, key, value, g, weights, *, within):
    """The levels above `WITHIN` of an unpacked bf16 call in the fast forms (`_fla_levels`;
    `log_linear_attention`'s operands, `g` and `weights` in fp32) added to `within` (the levels within a
    chunk's reads, unscaled) a level at a time from the lowest, as fla's `b_o` sums them (a pair cancelling
    within and at level 7 read beside level 8's where the levels summed first overflowed), `[B, T, H, P]`
    fp32 before the scale, as
    fla's chunked form computes them (fla/ops/log_linear_attn/chunk.py at 413ad2c), at its chunks' bounds
    and roundings: each 64-token chunk's memory its keys against its values decayed to the chunk's end and
    rounded to bf16 (`b_v`), one product a chunk, in fp32; a `2^j` block's memory its two halves' joined, the
    left's decayed across the right in fp32 (fla's `kv *= exp(b_g_last)`, `kv += kv`), level by level; and
    at level `j`, each 64-token chunk of a block's right half reading its left half's memory decayed in fp32
    to that chunk's start and rounded once (`kv.to(b_q.dtype)`), each query times its level weight rounded
    (`b_l * b_q`), the read decayed within its chunk after its product, the products the tensor cores' over
    fp32 sums, the scale the caller's at the output (`SCALE`); a row padded to whole top-level blocks with
    tokens no output keeps. Each chunk's decays are its own compensated sums (`_half_decays`), applied as a
    decay a value takes alone (`scan.scaled_exp`), a value through a vanished one zero and a read through
    one selected out. A level-8 write of `2^-60` under a key of `2^100`, decayed by `e^-80` across the next
    chunk, is kept as fla keeps it, where its whole decay on the value before its key rounded to zero."""
    length, groups, heads, columns = query.shape[1], query.shape[2], *value.shape[2:]
    top = levels(length) - 1
    if top < WITHIN.bit_length():
        return within
    padded = -(-length // (1 << top)) * (1 << top)

    def chunked(x):
        x = jnp.pad(x, [(0, 0), (0, padded - length), *[(0, 0)] * (x.ndim - 2)])
        return x.reshape(x.shape[0], padded // WITHIN, WITHIN, *x.shape[2:])

    def by_head(x):
        # A head axis's value heads beside their key head's: `[..., G, E]`.
        return x.reshape(*x.shape[:-1], groups, heads // groups)

    decays = chunked(g)
    to_end, past = _half_decays(decays, to_end=True)
    entered, crossed = _half_decays(decays, to_end=False)
    written = scan.scaled_exp(chunked(value).astype(g.dtype), jnp.where(past, -jnp.inf, to_end)[..., None])
    written = written.astype(value.dtype).reshape(*written.shape[:3], groups, -1, columns)
    memory = jnp.einsum('bmtgk,bmtgep->bmgkep', chunked(key), written, preferred_element_type=jnp.float32)
    # Each chunk's whole decay, -inf through a vanished token, and whether it holds one.
    cut = crossed[:, :, -1]
    chunk_whole = jnp.where(cut, -jnp.inf, entered[:, :, -1])
    whole = chunk_whole
    queries, chunk_weights = chunked(query), chunked(weights)
    out = within
    for level in range(WITHIN.bit_length(), top + 1):
        # The pyramid at the level's half blocks of `chunks` chunks: `memory` `[B, 2 n, G, K, E, P]`, `whole`
        # `[B, 2 n, H]`.
        chunks = 1 << (level - WITHIN.bit_length())

        def halves(x, chunks=chunks):
            return x.reshape(x.shape[0], -1, 2, chunks, *x.shape[2:])

        # Each right-half chunk's decay from the half's start to its own start, and whether a vanished
        # token before it cuts it.
        right, cuts = halves(chunk_whole)[:, :, 1], halves(cut)[:, :, 1]
        # The sums before each chunk, shifted a chunk: no -inf less itself (NaN), no large sum less a term.
        before = jnp.concatenate([jnp.zeros_like(right[:, :, :1]), jnp.cumsum(right, axis=2)[:, :, :-1]], 2)
        severed = jnp.cumsum(cuts, axis=2) - cuts > 0
        local, inside = halves(entered)[:, :, 1], halves(crossed)[:, :, 1]
        # A read through a vanished decay (a segment's reset), within its chunk or before it in its half,
        # discarded: its reader zeroed before its product and its read before its decay and after, as the
        # recurrence selects one out. Masked after the product alone, an overflowed reader's pullback was
        # `0 * inf` (dk, dv at a query of 2^127 weighed 2 severed by -1000), an overflowed read's decay's
        # derivative too (dg at 2^70 by 2^70).
        reset = inside | severed[..., None, :]
        # A chunk whose every read is discarded (a reset before it, or its first token vanished): its held
        # memory zero, its decay's operands too, so no pullback meets an overflowed memory's `0 * inf`.
        emptied = by_head(severed | inside[..., 0, :])[..., None, :, None]
        left = memory[:, 0::2, None]
        # An overflowed memory, reader or read alone is zeroed where its decay keeps no finite fp32 value
        # (`_dropped`), its value there every finite one's; a finite one decays as it is, its derivatives
        # those of its exact product: four live decays of -80 took an overflowed read of 2^70 by 2^70 to
        # NaN and a reader of 2^128 times its emptied memory too (train 1379's review), and zeroing a
        # finite read of 2^127 under four decays of -50 lost its decay's gradient 4.0e-11 (train 1398's).
        spent = emptied | (~jnp.isfinite(left) & by_head(_dropped(before))[..., None, :, None])
        start = by_head(before)[..., None, :, None]
        held = scan.scaled_exp(jnp.where(spent, 0.0, left), jnp.where(spent, 0.0, start))
        held = jnp.where(spent, 0.0, held).astype(value.dtype)
        weight = by_head(halves(chunk_weights[..., level])[:, :, 1])
        readers = halves(queries)[:, :, 1][..., None, :].astype(g.dtype) * weight[..., None]
        readers = readers.astype(query.dtype)
        dropping = _dropped(before)[..., None, :] | _dropped(local)
        unread = by_head(reset)[..., None] | (by_head(dropping)[..., None] & ~jnp.isfinite(readers))
        readers = jnp.where(unread, 0.0, readers)
        read = jnp.einsum('bnrtgek,bnrgkep->bnrtgep', readers, held, preferred_element_type=jnp.float32)
        read = read.reshape(*read.shape[:4], heads, columns)
        discarded = reset[..., None] | (_dropped(local)[..., None] & ~jnp.isfinite(read))
        decayed = scan.scaled_exp(jnp.where(discarded, 0.0, read),
                                  jnp.where(discarded, 0.0, local[..., None]))
        read = jnp.where(discarded, 0.0, decayed)
        laid = jnp.stack([jnp.zeros_like(read), read], axis=2)
        out = out + laid.reshape(laid.shape[0], -1, heads, columns)[:, :length]
        # The next level's half blocks: two of these joined, the left decayed across the right, zero where a
        # vanished token in the right cuts it (an overflowed left memory times its zero decay was NaN, a
        # level-8 read at token 128 of 2^70 by 2^70 cut by -1000 at token 64).
        across = by_head(whole[:, 1::2])[:, :, :, None, :, None]
        gone = jnp.isneginf(across) | (~jnp.isfinite(memory[:, 0::2]) & _dropped(across))
        joined = scan.scaled_exp(jnp.where(gone, 0.0, memory[:, 0::2]), jnp.where(gone, 0.0, across))
        memory, whole = jnp.where(gone, 0.0, joined) + memory[:, 1::2], whole[:, 0::2] + whole[:, 1::2]
    return out


def _dropped(exponent):
    """Where a decay `exponent` (natural units, at most 0) keeps no finite fp32 value, the largest decayed by
    it zero (`scan.scaled_exp`'s factors), -inf (a vanished token) among them: an overflowed read, reader or
    memory through it is zero, as every finite one's value is, where its product with the zero decay was
    NaN; a finite one decays as it is."""
    largest = jnp.full(exponent.shape, jnp.finfo(jnp.float32).max, jnp.float32)
    return jax.lax.stop_gradient(scan.scaled_exp(largest, exponent)) == 0


def _half_decays(exponents, *, to_end: bool):
    """Each token's decay exponent within its half `exponents` `[B, blocks, half, H]` (natural units) and
    whether its span reaches a vanished token (`scan.vanished`): from the half's first token through it, or
    with `to_end` from after it through the half's last; a compensated sum (`scan.compensated`) of the
    half's live exponents, so no span's sum holds a vanished token's (a -2**70 rounds a later -1 away)."""
    live, cut = scan.vanished(exponents)
    if to_end:
        high, low = scan.compensated(live, axis=2, reverse=True)
        # From after each token: the next token's suffix, the last's empty.
        high, low, cut = (jnp.concatenate([x[:, :, 1:], jnp.zeros_like(x[:, :, :1])], axis=2)
                          for x in (high, low, cut))
        crossed = jnp.flip(jnp.cumsum(jnp.flip(cut, 2), axis=2), 2) > 0
    else:
        high, low = scan.compensated(live, axis=2)
        crossed = jnp.cumsum(cut, axis=2) > 0
    return high + low, crossed


def _levels(count: int) -> np.ndarray:
    """`[WITHIN, WITHIN]` int: each pair of a block's slots' level (`_within`'s) of a sequence's `count`
    (`levels`)."""
    offsets = np.arange(WITHIN)
    return np.minimum(np.frexp(np.bitwise_xor.outer(offsets, offsets))[1], count - 1)


def _level_pairs(weight, count: int):
    """A block's pairs' weights `[WITHIN, WITHIN]`, each its query's (`weight` `[WITHIN, count]`) at its level
    (`_levels(count)`): gathered, no arithmetic, so a weight's infinity or NaN reaches its own level's pairs
    alone; a linear map (`generation.linear_map`) whose transpose selects and sums each level's cotangents a
    level at a time, so another level's infinite cotangent is no term of it (as a product with the levels'
    one-hot masks' zeros was, head review of fd3af8c41) and no scatter forms (autodiff's transpose of the
    gather, 3.0 ms of a G4's 9.1 ms log-linear Mamba-2 training step against fla's 2.3 ms whole, M-JS-275),
    every order of it a map of tangents. A level's select of the weights a level at a time took 295.7 us of a
    GH200's log-linear Mamba-2 forward against 47.9 for the masks' product (M-JS-287)."""
    def gathered(levels, moving):
        return jnp.take_along_axis(moving, levels, axis=1)

    def selected(levels, cotangent):
        return jnp.stack([jnp.sum(jnp.where(levels == level, cotangent, 0.0), axis=-1)
                          for level in range(count)], axis=-1)

    return generation.linear_map(gathered, selected, jnp.asarray(_levels(count)), weight)


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
    # pair's weight its query's at its level (`_level_pairs`).
    def head(query, key, value, decay, weight):
        pairs = _level_pairs(weight, weight.shape[-1])
        return scan.spread(query, key, value, scan.between(decay[:, None]), weights=pairs)

    # A key head's value heads, its queries and keys unmapped among them, so their scores are formed once a
    # key head (each value head repeating them formed log-linear-mamba2's 64 a block, an fp32 product of 234
    # us in a GH200's forward, M-JS-146ll); then a block's key heads, then every row's blocks.
    value_heads = jax.vmap(head, in_axes=(None, None, 1, 1, 1), out_axes=1)
    out = jax.vmap(jax.vmap(value_heads, in_axes=1, out_axes=1))(q, k, v, g, weights)
    return unlaid((scale * out).reshape(rows, count * WITHIN, heads, -1))
