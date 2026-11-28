"""Log-linear attention (`dew.nn.kernels.log_linear`, fla's log_linear_attn): its output and gradients on par
with fla's reference, the Fenwick tree's level masks built level by level (fla/ops/log_linear_attn/naive.py),
from float64, one sequence and packed documents (check 1)."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_recurrence_chunk import assert_packed_on_par, hosted, sigmoid

from dew.nn.kernels.log_linear import WITHIN, levels, log_linear_attention

ROWS, LENGTH, GROUPS, HEADS, WIDTH, VALUES = 2, 136, 1, 2, 16, 8
"""Two rows of 136 tokens (nine levels: zero to six within a 64-token block, `log_linear.WITHIN`, seven and
eight a recurrence each), one key group serving two value heads."""
DOCUMENTS = ((0, 13, 50), (0, 70, 99))
"""Each row's packed documents' first tokens: a document of 86 and one of 70 tokens reach level seven."""
SEEDS = {False: 32, True: 32}
"""Each site's declared seeds (one sequence, packed), sized by `seeds_for` on armada."""


def level_mask(level: int, length: int) -> np.ndarray:
    """fla's `construct_level_mask`'s pattern `[T, T]`: the keys a query reaches at `level`."""
    rows, columns = np.indices((length, length))
    if level == 0:
        return rows == columns
    half = 1 << (level - 1)
    base = rows - rows % half
    return (rows % (1 << level) >= half) & (columns + half >= base) & (columns < base)


def naive(query, key, value, g, level_scales, *, starts=None):
    """fla's `naive_log_linear_attn` (scale `N ** -0.5`), each packed document a sequence of its own from
    its first token (`DOCUMENTS`' where `starts` is given)."""
    if starts is not None:
        pieces = []
        for row, begins in enumerate(DOCUMENTS):
            ends = (*begins[1:], LENGTH)
            pieces.append(jnp.concatenate([
                naive(*(x[row:row + 1, begin:end] for x in (query, key, value, g)),
                      level_scales[row:row + 1, begin:end, :, :levels(end - begin)])
                for begin, end in zip(begins, ends, strict=True)], axis=1))
        return jnp.concatenate(pieces)
    every = range(math.ceil(math.log2(query.shape[1])) + 1)
    return (_masked(query, key, value, g, level_scales, every) * WIDTH ** -0.5).astype(value.dtype)


def _masked(query, key, value, g, level_scales, among) -> jax.Array:
    """`naive`'s sum over the levels `among` of one sequence, in at least fp32 and unscaled."""
    length = query.shape[1]
    work = jnp.promote_types(value.dtype, jnp.float32)
    q, k = (jnp.repeat(x.astype(work), value.shape[2] // query.shape[2], axis=2) for x in (query, key))
    running = jnp.cumsum(g.astype(work), axis=1).transpose(0, 2, 1)
    causal = np.tril(np.ones((length, length), bool))
    decay = jnp.exp(jnp.where(causal, running[..., :, None] - running[..., None, :], -jnp.inf))
    weights = level_scales.astype(work).transpose(0, 2, 1, 3)
    mixed = sum(decay * level_mask(level, length) * weights[..., level, None] for level in among)
    scores = jnp.einsum('bhlc,blhn,bchn->bhlc', mixed, q, k)
    return jnp.einsum('bhlc,bchp->blhp', scores, value.astype(work))


def fla_chunked(query, key, value, g, level_scales, *, starts=None):
    """fla's chunked arithmetic for one sequence (fla/ops/log_linear_attn/chunk.py at 413ad2c), a 16-bit
    call's roundings where fla rounds: the levels within a 64-token chunk as `naive`'s; each level above
    read at each 64-token chunk of its blocks' right halves from its keys' memory in fp32, each left-half
    chunk's values decayed to that chunk's end and rounded before its keys (`b_v`), the memory decayed to
    the reading chunk's start and rounded there (`kv.to(b_q.dtype)`), the query times its level weight
    rounded (`b_l * b_q`), the read decayed within its chunk after its product, the scale at the output
    (`SCALE`). An fp32 or a float64 call rounds none."""
    assert starts is None, "one sequence"
    length, chunk = query.shape[1], WITHIN
    work = jnp.promote_types(value.dtype, jnp.float32)

    def rounded(x):
        return x.astype(value.dtype).astype(work) if jnp.dtype(value.dtype).itemsize < 4 else x

    q, k = (jnp.repeat(x.astype(work), value.shape[2] // query.shape[2], axis=2) for x in (query, key))
    v, running, weights = value.astype(work), jnp.cumsum(g.astype(work), axis=1), level_scales.astype(work)
    out = _masked(query, key, value, g, level_scales, range(WITHIN.bit_length()))
    for level in range(WITHIN.bit_length(), levels(length)):
        half = 1 << (level - 1)
        for middle in range(half, length, 2 * half):
            for reading in range(middle, min(middle + half, length), chunk):
                read = slice(reading, min(reading + chunk, middle + half, length))
                memory = 0.0
                for source in range(middle - half, middle, chunk):
                    taken = slice(source, source + chunk)
                    end = running[:, source + chunk - 1]
                    values = rounded(v[:, taken] * jnp.exp(end[:, None] - running[:, taken])[..., None])
                    memory = memory + jnp.exp(running[:, reading - 1] - end)[..., None, None] * jnp.einsum(
                        'bnhk,bnhp->bhkp', k[:, taken], values)
                reader = rounded(q[:, read] * weights[:, read, :, level, None])
                product = jnp.einsum('bnhk,bhkp->bnhp', reader, rounded(memory))
                decayed = jnp.exp(running[:, read] - running[:, reading - 1, None])[..., None]
                out = out.at[:, read].add(product * decayed)
    return (out * WIDTH ** -0.5).astype(value.dtype)


def draw(seed: int, groups: int = GROUPS, heads: int = HEADS, dtype=jnp.float32):
    """Operands from `seed`, `groups` key heads serving `heads` value heads: unit queries and keys and
    values in `dtype`, a log decay `log sigmoid` and level weights about one in fp32, and a cotangent of the
    output in `dtype`."""
    normal = np.random.default_rng([seed, 3]).standard_normal
    q, k = (normal((ROWS, LENGTH, groups, WIDTH)) for _ in range(2))
    q, k = (x / np.linalg.norm(x, axis=-1, keepdims=True) for x in (q, k))
    v = normal((ROWS, LENGTH, heads, VALUES))
    g = np.log(sigmoid(normal((ROWS, LENGTH, heads)) + 2.0))
    weights = 1.0 + 0.3 * normal((ROWS, LENGTH, heads, levels(LENGTH)))
    return (*(hosted(x, dtype) for x in (q, k, v)), *(hosted(x) for x in (g, weights)),
            hosted(normal((ROWS, LENGTH, heads, VALUES)), dtype))


@pytest.mark.parametrize(('packed', 'dtype'), [(False, jnp.float32), (True, jnp.float32),
                                               (False, jnp.bfloat16)], ids=['fp32', 'fp32 packed', 'bf16'])
def test_the_masked_form_is_as_exact_as_flas_level_by_level(packed, dtype):
    """The output and every operand's gradient, the level weights' included, of the masked form (each key's
    level its position's highest bit apart from the query's) on par with fla's reference building each
    level's mask, from float64, one sequence and packed documents counting positions from their own first
    tokens, and one bf16 sequence, its levels above a block fla's chunked arithmetic, held to fla's chunked
    form's roundings at its precision (`fla_chunked`; check 1)."""
    # Packed, each row's documents bound its levels' blocks (`max_segments`).
    dew = functools.partial(log_linear_attention, max_segments=max(map(len, DOCUMENTS))) if packed else \
        log_linear_attention
    reference = naive if dtype == jnp.float32 else fla_chunked
    assert_packed_on_par(dew, reference, functools.partial(draw, dtype=dtype), 'log-linear attention',
                         packed=packed, documents=DOCUMENTS, length=LENGTH, seeds=SEEDS[packed])


def test_key_heads_serve_their_own_value_heads_packed():
    """Two key heads, each serving two value heads, over packed documents: the output and every operand's
    gradient on par with fla's reference (check 1), so each value head reads its own key head's scores
    (the within-block scores formed once a key head, its value heads mapped inside it: head `g H/G + p`)."""
    dew = functools.partial(log_linear_attention, max_segments=max(map(len, DOCUMENTS)))
    assert_packed_on_par(dew, naive, functools.partial(draw, groups=2, heads=4),
                         'log-linear attention, two key heads', packed=True, documents=DOCUMENTS,
                         length=LENGTH, seeds=SEEDS[True])


def _two_tokens(weights, values=(1.0, 1.0)):
    """Two tokens' log-linear attention, one level apart: unit queries and keys, no decay, a unit scale,
    each token's level weights `weights` `[2, 2]` and its value `values`."""
    ones = jnp.ones((1, 2, 1, 1), jnp.float32)
    value = jnp.asarray(values, jnp.float32).reshape(1, 2, 1, 1)
    return functools.partial(log_linear_attention, ones, ones, value, jnp.zeros((1, 2, 1), jnp.float32),
                             scale=1.0), jnp.asarray(weights, jnp.float32).reshape(1, 2, 1, 2)


def test_a_pairs_weight_is_its_own_levels_alone_both_ways():
    """A token's weight at one level reaches that level's pairs alone, forward and back (head review of
    fd3af8c41): an infinite weight at the token's own level leaves its pair with the other token finite,
    its output infinite where a masked product's zero times it read NaN; and an infinite cotangent of its
    own pair leaves the other level's weight's gradient finite, 2 where a product's transpose read NaN."""
    attend, weights = _two_tokens([[1.0, 1.0], [np.inf, 1.0]])
    np.testing.assert_array_equal(np.asarray(attend(weights)).ravel(), [1.0, np.inf])
    attend, weights = _two_tokens([[0.5, 0.5], [0.5, 0.5]], values=(1.0, 2.0 ** 127))
    cotangent = jnp.asarray([0.0, 2.0], jnp.float32).reshape(1, 2, 1, 1)
    grads = jax.grad(lambda w: jnp.sum(attend(w) * cotangent))(weights)
    np.testing.assert_array_equal(np.asarray(grads).reshape(2, 2), [[0.0, 0.0], [np.inf, 2.0]])


@pytest.mark.parametrize("read", ["overflowing", "overflowing at a level of zero weight", "a NaN"])
def test_a_live_span_that_underflows_reads_as_the_recurrence(read):
    """Three tokens of one document within a block under log decays `[0, -80, -80]`, each live, its first
    and last tokens' span `exp(-160)` zero in float32 (`scan.spread`'s pairs): a key's and the last query's
    2^70, whose score overflows before that decay weighs it: every token's read exactly zero and every
    gradient finite, the contract of a score that overflows formed before its decay (rows 5426 and 9547;
    the recurrence a token at a time, which decays before it reads, reads 2^-20 there), where infinity times
    the zero decay was NaN (head review of 512810acf F7), the last token's levels weighed zero too, where
    the weight met the infinity first (head review of bd4031d41 F1); and a NaN in the first key read NaN at
    every token, as the recurrence a token at a time reads it, where a decay rounded to zero took the pair
    for a document's start (train 1079's review F1)."""
    weights = jnp.ones((1, 3, 1, levels(3)), jnp.float32)
    if read == "overflowing at a level of zero weight":
        weights = weights.at[0, 2].set(0.0)
    g = jnp.asarray([[[0.0], [-80.0], [-80.0]]], jnp.float32)
    value = jnp.ones((1, 3, 1, 1), jnp.float32)
    if read == "a NaN":
        query, key = jnp.ones((1, 3, 1, 1), jnp.float32), jnp.asarray([np.nan, 0.0, 0.0]).reshape(1, 3, 1, 1)
        assert np.isnan(np.asarray(log_linear_attention(query, key, value, g, weights, scale=1.0))).all()
        return
    query, key = (jnp.asarray(x, jnp.float32).reshape(1, 3, 1, 1) for x in ([0.0, 0.0, 2.0 ** 70],
                                                                           [2.0 ** 70, 0.0, 0.0]))

    def read(query, key, g):
        return log_linear_attention(query, key, value, g, weights, scale=1.0)

    out, pull = jax.vjp(read, query, key, g)
    np.testing.assert_array_equal(np.asarray(out), np.zeros(out.shape, np.float32))
    assert all(np.isfinite(np.asarray(x)).all() for x in pull(jnp.ones_like(out)))


def test_a_blocks_pairs_keep_a_decay_past_a_vanished_one_and_no_future_overflow():
    """Within a block (`log_linear.WITHIN`), a pair's decay after a vanishing one is its own, as after any
    (a running sum of -2**70 rounds a later -1 away, its pair decayed by one), and an overflowing score of a
    pair whose decay is zero, a future pair's or a past one's through a vanished decay, reaches neither the
    output nor a gradient (an infinity times its zero decay is a NaN)."""
    ones = jnp.ones((1, 2, 1, 1), jnp.float32)
    weights = jnp.ones((1, 2, 1, levels(2)), jnp.float32)
    vanished, kept = (log_linear_attention(ones, ones, ones, jnp.asarray([[[first], [-1.0]]], jnp.float32),
                                           weights, scale=1.0) for first in (-2.0 ** 70, 0.0))
    # Token 1's pair with token 0 decays by token 1's own -1 alone, whatever token 0's decay.
    np.testing.assert_array_equal(np.asarray(vanished), np.asarray(kept))
    big = [([1e20, 0.0], [0.0, 1e20], 0.0), ([0.0, 1e20], [1e20, 0.0], -1000.0)]

    def summed(query, key, g):
        return log_linear_attention(query, key, ones, g, weights, scale=1.0).sum()

    # A future pair's score overflowing, and a past one's through a vanished decay.
    for query, key, decay in big:
        query, key = (jnp.asarray(x, jnp.float32).reshape(1, 2, 1, 1) for x in (query, key))
        g = jnp.asarray([[[0.0], [decay]]], jnp.float32)
        value, grads = jax.value_and_grad(summed, argnums=(0, 1, 2))(query, key, g)
        assert value == 0.0
        assert all(np.isfinite(np.asarray(grad)).all() for grad in grads)


def _across_a_block(query, key, g, *, scale: float = 1.0, weight: float = 1.0, dtype=jnp.bfloat16):
    """65 tokens whose one weighed level is the 128-token blocks' (7, the first above `log_linear.WITHIN`),
    at token 64 alone, the block's right half partial and padded: a unit value at token 0, `query` and `key`
    at tokens 64 and 0, `g` at token 64 and `weight` its level weight there, every other operand zero, the
    query, key and value in `dtype`; the call of its query, key and log decay, and those operands."""
    def at(where, value, kind=dtype):
        return jnp.zeros((1, 65, 1, 1), kind).at[0, where].set(value)

    weights = jnp.zeros((1, 65, 1, levels(65)), jnp.float32).at[0, 64, 0, 7].set(weight)

    def call(q, k, decay):
        return log_linear_attention(q, k, at(0, 1.0), decay, weights, scale=scale)

    return call, (at(64, query), at(0, key), at(64, g, jnp.float32)[..., 0])


def test_a_levels_read_through_a_vanished_decay_reads_zero_its_overflow_selected_out():
    """A bf16 level's read whose decay from its half's first token vanishes (-1000 at token 64) reads zero and
    every gradient stays finite though its product overflows (2^70 by 2^70), as the recurrence selects out a
    read through a vanished decay (`log_linear._levels_above`), where the overflow times the zero decay was
    NaN."""
    call, operands = _across_a_block(2.0 ** 70, 2.0 ** 70, -1000.0)
    out, pull = jax.vjp(call, *operands)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.zeros(out.shape, np.float32))
    assert all(np.isfinite(np.asarray(x)).all() for x in pull(jnp.ones_like(out)))


def test_a_bf16_levels_write_decays_across_a_later_chunk_as_its_memory_as_flas():
    """A bf16 level-8 read (129 tokens, its block's left half two chunks) of a key of 2^100 against a value
    of 2^-60 at token 0, decayed by `e^-80` across the next chunk (-40 at tokens 64 and 65), reads that
    write's `2^40 e^-80`, its chunk's memory formed first and decayed in fp32 as fla's (`kv *=
    exp(b_g_last)`; `log_linear._levels_above`), where the value decayed across the whole half before its
    key rounded to zero."""
    def at(where, value, dtype=jnp.bfloat16):
        return jnp.zeros((1, 129, 1, 1), dtype).at[0, where].set(value)

    weights = jnp.zeros((1, 129, 1, levels(129)), jnp.float32).at[0, 128, 0, 8].set(1.0)
    g = jnp.zeros((1, 129, 1), jnp.float32).at[0, 64:66].set(-40.0)
    out = log_linear_attention(at(128, 1.0), at(0, 2.0 ** 100), at(0, 2.0 ** -60), g, weights, scale=1.0)
    expected = (2.0 ** 40 * jnp.exp(jnp.float32(-80.0))).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(out, np.float32)[0, :, 0, 0],
                                  np.where(np.arange(129) == 128, np.float32(expected), np.float32(0.0)))


def _tokens(length: int, where, values):
    """A bf16 operand `[1, length, 1, 1]`, zero but at the tokens `where`, which take `values`."""
    return jnp.zeros((1, length, 1, 1), jnp.bfloat16).at[0, jnp.asarray(where)].set(
        jnp.asarray(values).reshape(-1, 1, 1))


@pytest.mark.parametrize(("key", "value", "vanished", "gated", "decayed", "nan"), [
    (np.nan, 1.0, [], [], True, True), (np.nan, 1.0, [130], [], True, False),
    (np.nan, 1.0, [1], [], True, False), (np.nan, 1.0, [1], [], False, False),
    (1.0, np.nan, [1], [], False, False), (1.0, np.inf, [], [1], True, True)],
    ids=["a NaN key through a live span", "a NaN key through a vanished token in its span",
         "a NaN key cut by a vanished token after it in its chunk", "the same under no decay",
         "a NaN value cut by a vanished token after it in its chunk",
         "an infinite value under a NaN gate after it in its chunk"])
def test_a_bf16_levels_nan_input_reads_nan_through_a_live_span_alone(key, value, vanished, gated, decayed,
                                                                      nan):
    """A bf16 level-8 read (193 tokens) at token 192 of token 0's key and value (a unit query), its span
    holding four live decays of -80 where `decayed` (tokens 128 to 131; its prefix -320, past every finite
    fp32 value) and a vanished token at each of `vanished`: NaN through a live span, as the chunked
    reference's (`scan.spread`: a NaN read reads NaN), where the levels' mask of an overflowed memory zeroed
    it (audit 2026-11-27 K02), and a NaN gate at each of `gated` alike (an infinite value under it, its write
    a NaN decay's, not an overflow's; a head review); zero through a vanished token, in its span or after
    token 0 in its chunk, as the recurrence selects it out, where a NaN key or value cut from its chunk's
    end times its zero write was NaN (train 1416's review)."""
    weights = jnp.zeros((1, 193, 1, levels(193)), jnp.float32).at[0, 192, 0, 8].set(1.0)
    g = jnp.zeros((1, 193, 1), jnp.float32).at[0, 128:132].set(-80.0 if decayed else 0.0)
    for token in vanished:
        g = g.at[0, token].set(-jnp.inf)
    for token in gated:
        g = g.at[0, token].set(jnp.nan)
    out = log_linear_attention(_tokens(193, [192], [1.0]), _tokens(193, [0], [key]),
                               _tokens(193, [0], [value]), g, weights, scale=1.0)
    found = np.asarray(out, np.float32)[0, 192, 0, 0]
    assert np.isnan(found) if nan else found == 0, found


def test_a_bf16_levels_nan_key_reads_nan_for_its_live_value_heads_alone():
    """Two value heads of one key head, a bf16 level-8 read (193 tokens) at token 192 of token 0's NaN key
    against unit values, four decays of -80 at tokens 128 to 131, the first head's token 1 vanished: zero for
    the first, its reset cutting the key out of its write, and NaN for the second, which reads it live, the
    keys shared by both heads' memories (a value head's keys formed apart for its mask took a GH200's
    forward from 1.10 to 3.15 ms, M-JS-317b)."""
    weights = jnp.zeros((1, 193, 2, levels(193)), jnp.float32).at[0, 192, :, 8].set(1.0)
    g = jnp.zeros((1, 193, 2), jnp.float32).at[0, 128:132].set(-80.0).at[0, 1, 0].set(-jnp.inf)
    values = jnp.zeros((1, 193, 2, 1), jnp.bfloat16).at[0, 0].set(1.0)
    out = log_linear_attention(_tokens(193, [192], [1.0]), _tokens(193, [0], [jnp.nan]), values, g, weights,
                               scale=1.0)
    found = np.asarray(out, np.float32)[0, 192, :, 0]
    assert found[0] == 0 and np.isnan(found[1]), found


def test_a_bf16_levels_nan_weight_reads_nan_under_its_spent_decay():
    """A bf16 level-8 read (200 tokens) at token 196 of token 0's unit key and value by a unit query whose
    level-8 weight is NaN (its own level's pairs alone, the within's finite), five live decays of -80 at
    tokens 192 to 196 in its chunk before it (its local decay -400, past every finite fp32 value, no token
    vanished): NaN, a NaN reader's read kept as a NaN input's, where the levels' mask of a non-finite read
    under a spent decay zeroed it (audit 2026-11-27 K02), its NaN-ness taken from the query's entries and
    its weight, no level reducing its value heads' readers (M-JS-318)."""
    weights = jnp.zeros((1, 200, 1, levels(200)), jnp.float32).at[0, 196, 0, 8].set(jnp.nan)
    g = jnp.zeros((1, 200, 1), jnp.float32).at[0, 192:197].set(-80.0)
    unit = _tokens(200, [0], [1.0])
    out = log_linear_attention(_tokens(200, [196], [1.0]), unit, unit, g, weights, scale=1.0)
    assert np.isnan(np.asarray(out, np.float32)[0, 196, 0, 0]), out


def test_a_bf16_levels_nan_key_leaves_a_reset_reads_key_gradients_finite():
    """A bf16 level-7 read (65 tokens) at token 64, itself vanished, of token 0's NaN key and token 1's unit
    key against a unit value at token 0: its output zero, and its gradient by token 1's key zero, the NaN
    the key's head reads injected after the memory's product, where written into the product's operand the
    reset's zero cotangent met it, NaN (train 1431's review)."""
    weights = jnp.zeros((1, 65, 1, levels(65)), jnp.float32).at[0, 64, 0, 7].set(1.0)
    g = jnp.zeros((1, 65, 1), jnp.float32).at[0, 64].set(-jnp.inf)
    keys = _tokens(65, [0, 1], [jnp.nan, 1.0])

    def read(keys):
        out = log_linear_attention(_tokens(65, [64], [1.0]), keys, _tokens(65, [0], [1.0]), g, weights,
                                   scale=1.0)
        return out[0, 64, 0, 0].astype(jnp.float32)

    found, gradient = jax.value_and_grad(read)(keys)
    assert found == 0, found
    assert np.asarray(gradient, np.float32)[0, 1, 0, 0] == 0, gradient


@pytest.mark.parametrize(("overflowed", "gates"), [(False, (64, 68, -80.0)), (True, (67, 68, -1.0))],
                         ids=["spent decays", "a live decay over an overflowed dot"])
def test_a_bf16_levels_nan_key_leaves_a_later_resets_decay_gradients_finite(overflowed, gates):
    """Level-7 bf16 reads (69 tokens) at tokens 67 and 68 of token 0's NaN key against a unit value, decays
    `gates` (from, to, each) before token 68, vanished: token 68's output zero and its gradient by token 67's
    decay zero (token 68's output depends on none of it), token 67's keyed read no operand of its decay and
    its NaN set after it, where set before it the decay's pullback multiplied token 67's zero cotangent by it,
    NaN (train 1441's review); and with token 1's key and value 2^70 and 1 under token 67's query 2^70, its
    dot overflowed beside the NaN key's (`2^140`) under a live decay, `0 * inf` (a head review)."""
    weights = jnp.zeros((1, 69, 1, levels(69)), jnp.float32).at[0, 67:69, 0, 7].set(1.0)
    big = 2.0 ** 70 if overflowed else 1.0
    queries = _tokens(69, [67, 68], [big, 1.0])
    keys = _tokens(69, [0, 1], [jnp.nan, big]) if overflowed else _tokens(69, [0], [jnp.nan])
    values = _tokens(69, [0, 1], [1.0, 1.0]) if overflowed else _tokens(69, [0], [1.0])

    def read(g):
        out = log_linear_attention(queries, keys, values, g, weights, scale=1.0)
        return out[0, 68, 0, 0].astype(jnp.float32)

    start, stop, decay = gates
    g = jnp.zeros((1, 69, 1), jnp.float32).at[0, start:stop].set(decay).at[0, 68].set(-jnp.inf)
    found, gradient = jax.value_and_grad(read)(g)
    assert found == 0, found
    assert np.asarray(gradient, np.float32)[0, 67, 0] == 0, gradient


def test_a_bf16_levels_nan_weight_leaves_a_later_resets_decay_gradient_finite():
    """Level-7 bf16 reads (69 tokens) at tokens 67 and 68 of token 0's unit key and value, token 67's level-7
    weight NaN and token 68's 1, decays of -80 at tokens 64 to 67, token 68 vanished: token 68's output zero
    and its gradient by token 67's decay zero (token 68's output depends on none of it), token 67's NaN
    reader zero in its product and its read none of its decay's operands, its NaN set once after, where its
    NaN read entered the decay and the pullback's zero cotangent met it, NaN (train 1450's review)."""
    weights = (jnp.zeros((1, 69, 1, levels(69)), jnp.float32).at[0, 67, 0, 7].set(jnp.nan)
               .at[0, 68, 0, 7].set(1.0))
    unit = _tokens(69, [0], [1.0])

    def read(g):
        out = log_linear_attention(_tokens(69, [67, 68], [1.0, 1.0]), unit, unit, g, weights, scale=1.0)
        return out[0, 68, 0, 0].astype(jnp.float32)

    g = jnp.zeros((1, 69, 1), jnp.float32).at[0, 64:68].set(-80.0).at[0, 68].set(-jnp.inf)
    found, gradient = jax.value_and_grad(read)(g)
    assert found == 0, found
    assert np.asarray(gradient, np.float32)[0, 67, 0] == 0, gradient


def test_a_bf16_levels_nan_key_beside_an_overflow_leaves_a_resets_decay_gradient_finite():
    """A bf16 level-8 read (194 tokens) at tokens 192 and 193 by unit queries of token 0's NaN key and unit
    value and token 1's key and value 2^70 (their memory 2^140, past fp32's range), decays of -80 at tokens
    128 to 131, token 193 vanished: token 193's output zero and its gradient by token 131's decay zero, the
    NaN key's rows zero in the held memory's decay and their NaN-ness carried apart, where the overflowed
    memory the NaN exempted from its mask entered the decay and the reset's zero cotangent met it, NaN
    (train 1450's review)."""
    weights = jnp.zeros((1, 194, 1, levels(194)), jnp.float32).at[0, 192:194, 0, 8].set(1.0)
    big = 2.0 ** 70
    keys, values = _tokens(194, [0, 1], [jnp.nan, big]), _tokens(194, [0, 1], [1.0, big])

    def read(g):
        out = log_linear_attention(_tokens(194, [192, 193], [1.0, 1.0]), keys, values, g, weights, scale=1.0)
        return out[0, 193, 0, 0].astype(jnp.float32)

    g = jnp.zeros((1, 194, 1), jnp.float32).at[0, 128:132].set(-80.0).at[0, 193].set(-jnp.inf)
    found, gradient = jax.value_and_grad(read)(g)
    assert found == 0, found
    assert np.asarray(gradient, np.float32)[0, 131, 0] == 0, gradient


def test_a_bf16_levels_vanished_token_dominates_a_nan_gate_in_its_span():
    """A bf16 level-9 read (260 tokens) at token 259 of token 0's unit key and value, a NaN gate at 128 and a
    vanished token at 192 in the span between, decays of -80 at 256 to 259: zero, the vanished token
    selecting the read out whatever the NaN, where the pyramid's whole decays summed `NaN + -inf` to NaN,
    the reset lost and the NaN read at token 259 (train 1423's review)."""
    weights = jnp.zeros((1, 260, 1, levels(260)), jnp.float32).at[0, 259, 0, 9].set(1.0)
    g = (jnp.zeros((1, 260, 1), jnp.float32).at[0, 128].set(jnp.nan).at[0, 192].set(-jnp.inf)
         .at[0, 256:260].set(-80.0))
    unit = _tokens(260, [0], [1.0])
    out = log_linear_attention(_tokens(260, [259], [1.0]), unit, unit, g, weights, scale=1.0)
    assert np.asarray(out, np.float32)[0, 259, 0, 0] == 0, out


def test_a_bf16_levels_overflowed_memory_read_by_a_zero_query_under_a_spent_decay_reads_zero():
    """A bf16 level-7 read (68 tokens) at token 67 by a zero query of token 0's key and value of 2^70 (its
    memory 2^140, past fp32's range), its chunk's decays of -80 at tokens 64 to 67 (its read's -320, past
    every finite value): zero, the zero reader times the overflowed memory's NaN an overflow's, discarded
    under its spent decay as an overflowed read is, where a mask of infinities alone kept it (train 1416's
    review)."""
    weights = jnp.zeros((1, 68, 1, levels(68)), jnp.float32).at[0, 67, 0, 7].set(1.0)
    g = jnp.zeros((1, 68, 1), jnp.float32).at[0, 64:68].set(-80.0)
    large = _tokens(68, [0], [2.0 ** 70])
    out = log_linear_attention(jnp.zeros_like(large), large, large, g, weights, scale=1.0)
    assert np.asarray(out, np.float32)[0, 67, 0, 0] == 0, out


def test_a_bf16_calls_within_and_level_reads_are_summed_before_their_scale_as_flas():
    """A bf16 call whose token 64 reads its own key of 2^100 (level 0) and token 0's of -2^100 (level 7),
    unit values, at a scale of 2^30 reads zero, the within and the levels' reads summed before the scale
    as fla's `b_o * SCALE` (`log_linear.log_linear_attention`), where each scaled apart read infinity less
    infinity."""
    weights = jnp.zeros((1, 65, 1, levels(65)), jnp.float32).at[0, 64, 0, jnp.asarray([0, 7])].set(1.0)
    out = log_linear_attention(_tokens(65, [64], [1.0]), _tokens(65, [0, 64], [-2.0 ** 100, 2.0 ** 100]),
                               _tokens(65, [0, 64], [1.0, 1.0]), jnp.zeros((1, 65, 1), jnp.float32), weights,
                               scale=2.0 ** 30)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.zeros(out.shape, np.float32))


def test_a_bf16_calls_reads_are_summed_from_the_within_up_a_level_at_a_time_as_flas():
    """A bf16 call (193 tokens) whose token 192 reads its own key of -2^127 (level 0), token 128's of 2^127
    (level 7) and token 0's of 2^127 (level 8), unit values, reads 2^127: the within read first, then each
    level's from the lowest, as fla's `b_o` sums them (`log_linear._levels_above`), where the levels
    summed before the within overflowed."""
    weights = jnp.zeros((1, 193, 1, levels(193)), jnp.float32).at[0, 192, 0, jnp.asarray([0, 7, 8])].set(1.0)
    keys = _tokens(193, [0, 128, 192], [2.0 ** 127, 2.0 ** 127, -2.0 ** 127])
    out = log_linear_attention(_tokens(193, [192], [1.0]), keys, _tokens(193, [0, 128, 192], [1.0] * 3),
                               jnp.zeros((1, 193, 1), jnp.float32), weights, scale=1.0)
    np.testing.assert_array_equal(np.asarray(out, np.float32)[0, :, 0, 0],
                                  np.where(np.arange(193) == 192, np.float32(2.0 ** 127), np.float32(0.0)))


@pytest.mark.parametrize(("query", "key", "weight"), [(2.0 ** 70, 2.0 ** 70, 1.0), (2.0 ** 127, 1.0, 2.0)],
                         ids=["an overflowed read", "an overflowed reader"])
def test_a_bf16_levels_read_severed_earlier_in_its_half_reads_zero_its_gradients_finite(query, key, weight):
    """A bf16 level-8 read (193 tokens) at token 192 of token 0's key against a value of 1, its right half's
    first chunk holding a vanished decay (-1000 at token 128), reads zero and every gradient stays finite,
    the read discarded before its product and its decay (`log_linear._levels_above`): masked after them, an
    overflowed read (2^70 by 2^70) times its zero decay's derivative was NaN at token 192's decay, and an
    overflowed reader (2^127 weighed 2) times its zero cotangent NaN at the key's and value's (train reviews
    of bdaf24f63 and 6dfb161ea)."""
    weights = jnp.zeros((1, 193, 1, levels(193)), jnp.float32).at[0, 192, 0, 8].set(weight)
    g = jnp.zeros((1, 193, 1), jnp.float32).at[0, 128].set(-1000.0)

    def call(q, k, v, decay):
        return log_linear_attention(q, k, v, decay, weights, scale=1.0)

    operands = (_tokens(193, [192], [query]), _tokens(193, [0], [key]), _tokens(193, [0], [1.0]), g)
    out, pull = jax.vjp(call, *operands)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.zeros(out.shape, np.float32))
    assert all(np.isfinite(np.asarray(x, np.float32)).all() for x in pull(jnp.ones_like(out)))


def test_a_bf16_levels_memory_cut_across_its_half_joins_as_zero():
    """A bf16 level-8 read (129 tokens) at token 128 of token 0's key and value of 2^70, the next chunk of its
    block's left half holding a vanished decay (-1000 at token 64), reads zero, its forward and gradients
    finite, the left chunk's memory joined across the cut as zero (`log_linear._levels_above`), where its
    overflowed memory times the zero decay was NaN (a train review of 6dfb161ea)."""
    weights = jnp.zeros((1, 129, 1, levels(129)), jnp.float32).at[0, 128, 0, 8].set(1.0)
    g = jnp.zeros((1, 129, 1), jnp.float32).at[0, 64].set(-1000.0)

    def call(k, v, decay):
        return log_linear_attention(_tokens(129, [128], [1.0]), k, v, decay, weights, scale=1.0)

    out, pull = jax.vjp(call, _tokens(129, [0], [2.0 ** 70]), _tokens(129, [0], [2.0 ** 70]), g)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.zeros(out.shape, np.float32))
    assert all(np.isfinite(np.asarray(x, np.float32)).all() for x in pull(jnp.ones_like(out)))


@pytest.mark.parametrize(("length", "level", "decayed", "query", "key", "weight"), [
    (68, 7, (64, 68), 2.0 ** 70, 2.0 ** 70, 1.0), (321, 9, (256, 260), 2.0 ** 127, 1.0, 2.0)],
    ids=["within its chunk", "before its chunk"])
def test_a_bf16_levels_read_whose_decay_keeps_no_finite_value_reads_zero(length, level, decayed, query, key,
                                                                         weight):
    """A bf16 level's read at the last token of token 0's key against a value of 1, through four live
    decays of -80 (none vanished alone) within the reader's chunk or before it in its half, reads zero
    and every gradient stays finite, a decay no finite fp32 survives discarding the read as a vanished one
    does (`log_linear._dropped`): within its chunk an overflowed read (2^70 by 2^70) times `e^-320` was
    NaN, and before it a reader of 2^127 weighed 2, overflowed, times its emptied memory (train 1379's
    review of 515fa30ec, K-F1, and its head review)."""
    weights = jnp.zeros((1, length, 1, levels(length)), jnp.float32).at[0, length - 1, 0, level].set(weight)
    g = jnp.zeros((1, length, 1), jnp.float32).at[0, decayed[0]:decayed[1]].set(-80.0)

    def call(q, k, decay):
        return log_linear_attention(q, k, _tokens(length, [0], [1.0]), decay, weights, scale=1.0)

    out, pull = jax.vjp(call, _tokens(length, [length - 1], [query]), _tokens(length, [0], [key]), g)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.zeros(out.shape, np.float32))
    assert all(np.isfinite(np.asarray(x, np.float32)).all() for x in pull(jnp.ones_like(out)))


def test_a_bf16_levels_infinite_key_under_a_spent_prefix_reads_zero():
    """A bf16 level-9 read (321 tokens) at token 320 by a unit query weighed 1 of token 0's infinite key
    against a value of 1, four live decays of -80 at tokens 256 to 259 (its prefix -320, past every finite
    fp32 value; its own chunk's decay zero): zero, the infinite key's NaN-ness an overflow's, cleared past
    a decay that keeps no finite value as an overflowed memory is (`log_linear._dropped`), where injected
    after the product it read NaN (train 1431's head review); a NaN key there reads NaN."""
    weights = jnp.zeros((1, 321, 1, levels(321)), jnp.float32).at[0, 320, 0, 9].set(1.0)
    g = jnp.zeros((1, 321, 1), jnp.float32).at[0, 256:260].set(-80.0)
    unit = _tokens(321, [0], [1.0])
    for key, expected_nan in ((jnp.inf, False), (jnp.nan, True)):
        out = log_linear_attention(_tokens(321, [320], [1.0]), _tokens(321, [0], [key]), unit, g, weights,
                                   scale=1.0)
        found = np.asarray(out, np.float32)[0, 320, 0, 0]
        assert np.isnan(found) if expected_nan else found == 0, (key, found)


@pytest.mark.parametrize(("query", "key", "cotangent", "decay", "kept"), [
    (1.0, 2.0 ** 127, 2.0 ** 127, -90.0, True), (1.0, 2.0 ** 127, 2.0 ** 127, -150.0, True),
    (1.0, 1.0, 2.0 ** 120, -120.0, False), (1.0, 2.0 ** -120, 2.0 ** 100, -25.0, False),
    (1.0, 2.0 ** 127, 2.0 ** 127, -200.0, False)])
def test_a_bf16_levels_decay_gradient_is_the_cotangent_times_its_decayed_read(query, key, cotangent, decay,
                                                                              kept):
    """A bf16 level-7 read (67 tokens) at token 66 of token 0's key against a value of 1, decayed by `decay`
    over tokens 64 to 66 (a third each, every one live), under `cotangent` there: each of those decays'
    gradients the cotangent times the decayed read (`scan.scaled_exp`'s declared transpose, as the plain
    formula's; Main's ruling on row 13750, 11-28). Where the decay keeps the read normal (`kept`),
    `cotangent * query * key * exp(decay)` in float64's binade: `2^254 e^-90` (2.4e37) and `2^254 e^-150`,
    where autodiff through the decay's factors overflowed (train 1379's review K-F2); where it underflows
    (`1 e^-120` by 2^120, a key of 2^-120 under `e^-25` by 2^100, `2^127 e^-200` by 2^127), zero, as the
    plain formula's, and every output finite."""
    weights = jnp.zeros((1, 67, 1, levels(67)), jnp.float32).at[0, 66, 0, 7].set(1.0)
    g = jnp.zeros((1, 67, 1), jnp.float32).at[0, 64:67].set(decay / 3)

    def call(decays):
        return log_linear_attention(_tokens(67, [66], [query]), _tokens(67, [0], [key]),
                                    _tokens(67, [0], [1.0]), decays, weights, scale=1.0)

    out, pull = jax.vjp(call, g)
    (dg,) = pull(_tokens(67, [66], [cotangent]))
    found = np.asarray(dg, np.float64)[0, 64:67, 0]
    whole = 3 * float(np.float32(decay / 3))
    assert np.isfinite(np.asarray(out, np.float32)).all(), out
    assert np.isfinite(found).all(), dg
    if kept:
        assert (found != 0).all(), dg
        assert (np.frexp(found)[1] == np.frexp(cotangent * query * key * np.exp(whole))[1]).all(), (found, dg)
    else:
        np.testing.assert_array_equal(found, np.zeros(3))


def test_a_bf16_calls_levels_take_the_recurrence_its_kernel_choice_names():
    """An unpacked bf16 call within `generation.chosen` naming the recurrence's kernel (`gated_delta_rule`,
    as tools/benchmark_recurrence.py's `mosaic` and `xla` arms do) runs each level above a block on that
    kernel, its routes the recurrence's, where the fast levels' products took none (a train review of
    bdaf24f63: the A/B's arms were one program)."""
    from dew.nn.kernels import generation

    q, k, v, g, weights, _ = draw(0, dtype=jnp.bfloat16)
    for kernel in ("xla", "mosaic_gpu", None):
        named = {} if kernel is None else {"gated_delta_rule": kernel}
        with generation.assumed("sm90"), generation.chosen(**named), generation.routes() as found:
            jax.eval_shape(log_linear_attention, q, k, v, g, weights)
        taken = {route.forced for route in found if route.choice == generation.Choice.GATED_DELTA_RULE}
        assert taken == (set() if kernel is None else {kernel}), (kernel, taken)


def test_a_bf16_levels_read_weighs_its_query_before_its_product_as_flas():
    """A bf16 level's read of a query of 2^127 at a level weight of 2^-12 against a memory of 2^8 reads
    2^123, its weight on its query before the product and the scale at the output, as fla's chunked form
    (`b_l * b_q`, then `SCALE`; `log_linear._levels_above`), where a weight after the product overflowed."""
    call, operands = _across_a_block(2.0 ** 127, 2.0 ** 8, 0.0, weight=2.0 ** -12)
    np.testing.assert_array_equal(np.asarray(call(*operands), np.float64)[0, :, 0, 0],
                                  np.where(np.arange(65) == 64, 2.0 ** 123, 0.0))


@pytest.mark.parametrize("target", ["sm80", "sm90", "sm100", "sm120"])
def test_packed_levels_train_on_the_template_and_unpacked_ones_take_no_recurrence(target):
    """A log-linear call of Kimi's geometry (64 value heads on one key head of 128, bf16), traced as each
    GPU family's device: over two packed documents each level's recurrence takes the recurrence template
    for its forward and its gradient (`generation.routes`), where a key head's value heads summed in one
    program held sm120's consumer-fed scores' gradient launch past 255 registers and every level trained on
    the portable math, 1091 ms of a step against its forward's 4.6 (row 10522); unpacked, no level takes a
    recurrence (each two products a block, `log_linear._levels_above`) and no route refuses; and unpacked in
    the accurate forms each level takes its recurrence, the portable recurrence's range."""
    from dew.nn.kernels import generation
    from dew.nn.precision import Numerics

    length, heads = 512, 64
    shapes = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in (
        ((1, length, 1, 128), jnp.bfloat16), ((1, length, 1, 128), jnp.bfloat16),
        ((1, length, heads, 64), jnp.bfloat16), ((1, length, heads), jnp.float32),
        ((1, length, heads, levels(length)), jnp.bfloat16))]

    def routed(starts):
        def call(*operands):
            return log_linear_attention(*operands, scale=1.0, starts=starts,
                                        max_segments=None if starts is None else 2)

        with generation.assumed(target), generation.routes() as found:
            jax.eval_shape(lambda *x: jax.vjp(call, *x)[1](call(*x)), *shapes)
        return found

    starts = jnp.zeros((1, length), bool).at[0, jnp.asarray([0, length // 2])].set(True)
    taken = {(route.kernel, None if route.refusal is None else route.refusal.code)
             for route in routed(starts) if route.choice == generation.Choice.GATED_DELTA_RULE}
    assert taken == {("mosaic_gpu", None)}, taken
    unpacked = [route for route in routed(None)
                if route.choice == generation.Choice.GATED_DELTA_RULE or route.refusal is not None]
    assert not unpacked, unpacked
    with generation.run_precision(Numerics(forms='accurate')):
        accurate = {route.choice for route in routed(None)}
    assert generation.Choice.GATED_DELTA_RULE in accurate, accurate


def test_a_level_weights_forward_derivative_is_its_tangents_gathered():
    """The pairs' level weights' forward derivative (`jax.jvp`, `jax.linearize`) of a two-token call whose
    level weights alone move: the call is linear in them, so its tangent is the call at the direction, as
    its select chain's was, where a custom reverse rule alone refused forward mode (head review of
    9d435e066; `log_linear._level_pairs`, a linear map whose transpose selects and sums)."""
    attend, weights = _two_tokens([[0.5, 0.25], [0.75, 1.0]])
    direction = jnp.asarray([[1.0, -2.0], [0.5, 3.0]], jnp.float32).reshape(weights.shape)
    _, tangent = jax.jvp(attend, (weights,), (direction,))
    _, linear = jax.linearize(attend, weights)
    np.testing.assert_array_equal(np.asarray(tangent), np.asarray(attend(direction)))
    np.testing.assert_array_equal(np.asarray(linear(direction)), np.asarray(tangent))
