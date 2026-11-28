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
    level, as an A/B of its kernels asks (tools/benchmark_recurrence.py's `mosaic` and `xla` arms). The fast
    path's gradient is the exact autodiff of its forward (fla's forward arithmetic, differentiated), held by
    check 1 against float64; fla's backward kernels' own rounding orders (its within's `dV` rounding a score
    to bf16 before its decay, chunk.py:1059-1063 at 413ad2c) are not its contract (Main's ruling, row
    10525)."""
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
    rows, chunks = query.shape[0], padded // WITHIN

    def chunked(x):
        x = jnp.pad(x, [(0, 0), (0, padded - length), *[(0, 0)] * (x.ndim - 2)])
        return x.reshape(rows, chunks, WITHIN, *x.shape[2:])

    def head_major(x):
        # A chunked per-head array `[B, C, t, H, ...]` as `[B, G, C, E, t, ...]`: the products' batch axes
        # (a key head, its chunks, a value head) ahead of their rows, so no level's read leaves its product's
        # own layout (each level's transposes back to the tokens' were most of its training's launches).
        x = x.reshape(rows, chunks, WITHIN, groups, heads // groups, *x.shape[4:])
        return jnp.moveaxis(x, (3, 4), (1, 3))

    decays = chunked(g)
    to_end, past = (head_major(x) for x in _half_decays(decays, to_end=True))
    entered, crossed = (head_major(x) for x in _half_decays(decays, to_end=False))
    keys = jnp.moveaxis(chunked(key), 3, 1)
    queries = jnp.moveaxis(chunked(query), 3, 1)
    # Whether each query holds a NaN, an infinity and a zero `[B, G, C, t]`, readers' NaN-ness from them and
    # a weight: each level's reduction over its value heads' readers cost 2 ms of forward (M-JS-318).
    query_facts = tuple(jnp.any(test(queries), axis=-1) for test in (jnp.isnan, jnp.isinf, lambda q: q == 0))
    memory, poisoned = _chunk_memories(head_major(chunked(value)), keys, to_end, past)
    # Each chunk's whole decay `[B, G, C, E]`, -inf through a vanished token, and whether it holds one.
    cut = crossed[..., -1]
    chunk_whole = jnp.where(cut, -jnp.inf, entered[..., -1])
    whole, whole_cut = chunk_whole, cut
    weights = head_major(chunked(weights))
    # Every read summed into the within's reads, as fla's `b_o` sums them, in the reads' layout
    # `[B, G, C, E, t, P]`, and laid back as the tokens' once.
    out = head_major(chunked(within))
    for level in range(WITHIN.bit_length(), top + 1):
        # The pyramid at the level's half blocks of `span` chunks: `memory` `[B, G, 2 n, E, P, K]`, `whole`
        # `[B, G, 2 n, E]`.
        span = 1 << (level - WITHIN.bit_length())

        def halves(x, span=span):
            return x.reshape(rows, groups, -1, 2, span, *x.shape[3:])

        # Each right-half chunk's decay from the half's start to its own start, and whether a vanished
        # token before it cuts it: `[B, G, n, r, E]`.
        right, cuts = halves(chunk_whole)[:, :, :, 1], halves(cut)[:, :, :, 1]
        # The sums before each chunk, shifted a chunk: no -inf less itself (NaN), no large sum less a term.
        before = jnp.concatenate([jnp.zeros_like(right[:, :, :, :1]),
                                  jnp.cumsum(right, axis=3)[:, :, :, :-1]], 3)
        severed = jnp.cumsum(cuts, axis=3) - cuts > 0
        local, inside = halves(entered)[:, :, :, 1], halves(crossed)[:, :, :, 1]
        prefix_spent, local_spent = _dropped(before), _dropped(local)
        # A read through a vanished decay (a segment's reset), within its chunk or before it in its half,
        # discarded: its reader zeroed before its product and its read before its decay and after, as the
        # recurrence selects one out. Masked after the product alone, an overflowed reader's pullback was
        # `0 * inf` (dk, dv at a query of 2^127 weighed 2 severed by -1000), an overflowed read's decay's
        # derivative too (dg at 2^70 by 2^70).
        reset = inside | severed[..., None]
        # A chunk whose every read is discarded (a reset before it, or its first token vanished): its held
        # memory zero, its decay's operands too, so no pullback meets an overflowed memory's `0 * inf`.
        emptied = (severed | inside[..., 0])[..., None, None]
        _, nan_keyed, inf_keyed = (x[:, :, 0::2, None] for x in poisoned)
        held, held_nan = _held_memory(memory, poisoned, before, emptied, prefix_spent, value.dtype)
        weight = halves(weights[..., level])[:, :, :, 1]
        nan_query, infinite_query, zero_query = (halves(x)[:, :, :, 1][:, :, :, :, None] for x in query_facts)
        reader_nan = (nan_query | jnp.isnan(weight) | (infinite_query & (weight == 0))
                      | (zero_query & jnp.isinf(weight))) & ~reset
        readers = (halves(queries)[:, :, :, 1][:, :, :, :, None].astype(g.dtype) * weight[..., None]).astype(
            query.dtype)
        unread = (reset[..., None] | reader_nan[..., None]
                  | ((prefix_spent[..., None] | local_spent)[..., None] & jnp.isinf(readers)))
        readers = jnp.where(unread, 0.0, readers)
        # `[B, G, n, r, E, t, P]`, the product's own layout.
        read = jnp.einsum('bgnretk,bgnrepk->bgnretp', readers, held, preferred_element_type=jnp.float32)
        keyed = (nan_keyed | (inf_keyed & ~prefix_spent)) & ~emptied[..., 0, 0]
        # A NaN decay within the chunk before the reader, the reader's or the held memory's NaN-ness.
        read_nan = reader_nan[..., None] | held_nan[..., None, :] | jnp.isnan(local)[..., None]
        read = _read_decayed(read, local, local_spent, reset, keyed, read_nan)
        out = out + jnp.stack([jnp.zeros_like(read), read], axis=3).reshape(out.shape)
        memory, poisoned = _joined(memory, poisoned, whole[:, :, 1::2, :, None, None])
        # A span's vanished token dominates a NaN gate in it: summed alone, `NaN + -inf` was NaN, the reset
        # lost and the NaN read past it (train 1423's review).
        whole_cut = whole_cut[:, :, 0::2] | whole_cut[:, :, 1::2]
        whole = jnp.where(whole_cut, -jnp.inf, whole[:, :, 0::2] + whole[:, :, 1::2])
    # Back to the tokens' layout `[B, T, H, P]`, once.
    out = jnp.moveaxis(out, (1, 3), (3, 4)).reshape(rows, padded, heads, columns)
    return out[:, :length]


def _held_memory(memory, poisoned, before, emptied, prefix_spent, dtype):
    """A level's held memories, the left halves' `memory` decayed by `before` to each chunk of the right ones,
    in `dtype`, and their rows' NaN-ness `[..., P]`. A NaN input's NaN-ness is carried apart from the numbers
    (`poisoned`, `_chunk_memories`): the held memory's rows a live NaN input reaches and a NaN decay before
    the chunk, their numbers zero in every product and decay (a pullback's zero cotangent times a NaN operand
    was NaN across a reset: train 1450's review), the read's NaN set once after them. An overflowed memory
    alone (an infinity, or a NaN its products form from finite operands) is zeroed where its decay keeps no
    finite fp32 value (`prefix_spent`, `_dropped`), and an `emptied` chunk's (every read of it discarded); a
    finite one decays as it is (`scan.scaled_exp`, its decay's derivative the cotangent times its decayed
    value, as the plain formula's): four live decays of -80 took an overflowed read of 2^70 by 2^70 to NaN
    (train 1379's review)."""
    left = memory[:, :, 0::2, None]
    rows_nan = poisoned[0][:, :, 0::2, None]
    held_nan = (rows_nan | jnp.isnan(before)[..., None]) & ~emptied[..., 0]
    spent = emptied | held_nan[..., None] | (~jnp.isfinite(left) & prefix_spent[..., None, None])
    held = scan.scaled_exp(jnp.where(spent, 0.0, left), jnp.where(spent, 0.0, before[..., None, None]))
    return jnp.where(spent, 0.0, held).astype(dtype), held_nan


def _read_decayed(read, local, local_spent, reset, keyed, read_nan):
    """The reads `read` decayed by `local` within their chunk: a reset's zero; a keyed (a non-finite key's)
    or NaN read (`read_nan`) no operand of its decay (its finite keys' dot may overflow), its NaN set after;
    an overflowed read zero where its decay keeps no finite fp32 value (`local_spent`), as a reader of 2^128
    times its emptied memory was NaN (train 1379's review)."""
    discarded = (reset[..., None] | keyed[..., None, None] | read_nan
                 | (local_spent[..., None] & ~jnp.isfinite(read)))
    decayed = scan.scaled_exp(jnp.where(discarded, 0.0, read), jnp.where(discarded, 0.0, local[..., None]))
    kept_nan = (((keyed[..., None, None] & ~(local_spent[..., None] & ~read_nan)) | read_nan)
                & ~reset[..., None])
    return jnp.where(kept_nan, jnp.nan, jnp.where(discarded, 0.0, decayed))


def _chunk_memories(values, keys, to_end, past):
    """Each chunk's memory `[B, G, C, E, P, K]` (its product's own layout) of its value heads' `values` `[B,
    G, C, E, t, P]` decayed to its end (`to_end`) and its key heads' `keys` `[B, G, C, t, K]`, shared by the
    value heads, the rows a NaN input reaches through its live pairs `[B, G, C, E, P]` (a NaN value, a NaN
    decay to the chunk's end, a NaN key every row of its value head's: a read sums every key's column, so one
    NaN column is every read's), and whether each value head read a NaN key live, and an infinite one, `[B, G,
    C, E]` each. A value head's pair with a vanished token after it in its chunk (`past`) writes nothing: its
    value zeroed before the product, its decay's operand zero there too (no pullback meets an infinite value's
    `0 * inf`), and a non-finite key kept out of the shared product (zero), a value head reading one live
    reading NaN instead after the product (`_levels_above`: a NaN written before it was the product's saved
    operand, its transpose's zero cotangent through a reset times it NaN in a key's gradient; train 1431's
    review), so no non-finite key across a reset is a term of any head's memory (a NaN key at token 0 cut by a
    vanished token 1 read NaN at token 192, where the recurrence reads zero; train 1416's review) and no value
    head's keys are formed apart (each one's masked for it took a GH200's log-linear bf16 forward from 1.10 to
    3.15 ms, M-JS-317b). A non-finite memory or read the products form from finite operands (an overflowed
    write's `inf - inf`, a zero reader times an overflowed memory) is zeroed where its decay keeps no finite
    value, as an overflowed one is (`_dropped`), and a NaN input's kept through live spans, as the chunked
    reference keeps it (`scan.spread`'s `_kept`; audit 2026-11-27 K02, train 1416's review). A live infinite
    key's head reads NaN where its prefix keeps a finite value, and zero past a spent one as an overflow's
    (train 1431's head review), where main's product read a signed infinity: an already non-finite input's
    output is outside row 5426's envelope (`tests/contract_support.assert_contract`; Main's ruling of
    2026-11-27 on row 13991), so no value head's keys are formed apart for it."""
    wide = values.astype(to_end.dtype)
    live = ~past
    # Whether each token's key holds a non-finite entry, and a NaN, each value head's live `[..., E, t]`.
    broken, nan_keys = (jnp.any(test(keys), axis=-1)[:, :, :, None] & live
                        for test in (lambda key: ~jnp.isfinite(key), jnp.isnan))
    # A live non-finite value or decay a NaN input's: its rows' NaN-ness carried apart and its numbers zero
    # in the product and the decay (a pullback's zero cotangent through a reset times it was NaN).
    unwritten = past[..., None] | ~jnp.isfinite(wide) | ~jnp.isfinite(to_end)[..., None]
    written = jnp.where(unwritten, 0.0, scan.scaled_exp(jnp.where(unwritten, 0.0, wide),
                                                       jnp.where(unwritten, 0.0, to_end[..., None])))
    written = written.astype(values.dtype)
    shared = jnp.where(jnp.isfinite(keys), keys, 0.0)
    memory = jnp.einsum('bgcetp,bgctk->bgcepk', written, shared, preferred_element_type=jnp.float32)
    rows = (jnp.any(~jnp.isfinite(wide) & live[..., None], axis=-2)
            | jnp.any((~jnp.isfinite(to_end) | nan_keys) & live, axis=-1)[..., None])
    return memory, (rows, jnp.any(nan_keys, axis=-1), jnp.any(broken & ~nan_keys, axis=-1))


def _joined(memory, poisoned, across):
    """The pyramid's next half blocks: each two of `memory`'s `[B, G, 2 n, E, P, K]` joined, the left decayed
    across the right (`across`, its whole decay `[B, G, n, E, 1, 1]`), zero where a vanished token in the
    right cuts it (an overflowed left memory times its zero decay was NaN, a level-8 read at token 128 of
    2^70 by 2^70 cut by -1000 at token 64), and an overflowed one where its decay keeps no finite value
    (`_dropped`); and the NaN inputs' rows (`_chunk_memories`) joined alike: the left's through its decay (a
    NaN decay every one of them), none through a reset."""
    rows, nan_keyed, inf_keyed = poisoned
    left_nan = rows[:, :, 0::2, :, :, None] | jnp.isnan(across)
    across_spent = _dropped(across)
    # A NaN input's rows zero in the decay's product, their NaN-ness carried apart (`rows`).
    gone = jnp.isneginf(across) | left_nan | (~jnp.isfinite(memory[:, :, 0::2]) & across_spent)
    joined = scan.scaled_exp(jnp.where(gone, 0.0, memory[:, :, 0::2]), jnp.where(gone, 0.0, across))
    kept, decayed_nan = ~jnp.isneginf(across[..., 0]), jnp.isnan(across[..., 0])
    rows = ((rows[:, :, 0::2] | decayed_nan) & kept) | rows[:, :, 1::2]
    nan_keyed = (nan_keyed[:, :, 0::2] & kept[..., 0]) | nan_keyed[:, :, 1::2]
    # An infinite key's NaN-ness an overflow's: none past a decay that keeps no finite value.
    inf_keyed = (inf_keyed[:, :, 0::2] & kept[..., 0] & ~across_spent[..., 0, 0]) | inf_keyed[:, :, 1::2]
    return jnp.where(gone, 0.0, joined) + memory[:, :, 1::2], (rows, nan_keyed, inf_keyed)


def _dropped(exponent):
    """Where a decay `exponent` (natural units, at most 0) keeps no finite fp32 value, the largest decayed by
    it zero (`scan.scaled_exp`'s factors), -inf (a vanished token) among them: an overflowed read, reader
    or memory through it (an infinity, or a NaN its products form from finite operands) is zero, as every
    finite one's value is, where its product with the zero decay was NaN; a finite one decays as it is, and
    a NaN input's stays NaN."""
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
