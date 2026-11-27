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

from dew.nn.kernels.log_linear import levels, log_linear_attention

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
    length = query.shape[1]
    work = jnp.promote_types(value.dtype, jnp.float32)
    q, k = (jnp.repeat(x.astype(work), value.shape[2] // query.shape[2], axis=2) for x in (query, key))
    running = jnp.cumsum(g.astype(work), axis=1).transpose(0, 2, 1)
    causal = np.tril(np.ones((length, length), bool))
    decay = jnp.exp(jnp.where(causal, running[..., :, None] - running[..., None, :], -jnp.inf))
    weights = level_scales.astype(work).transpose(0, 2, 1, 3)
    mixed = sum(decay * level_mask(level, length) * weights[..., level, None]
                for level in range(math.ceil(math.log2(length)) + 1))
    scores = jnp.einsum('bhlc,blhn,bchn->bhlc', mixed, q, k)
    return (jnp.einsum('bhlc,bchp->blhp', scores, value.astype(work)) * WIDTH ** -0.5).astype(value.dtype)


def draw(seed: int, groups: int = GROUPS, heads: int = HEADS):
    """fp32 operands from `seed`, `groups` key heads serving `heads` value heads: unit queries and keys, a
    log decay `log sigmoid`, level weights about one, and a cotangent of the output."""
    normal = np.random.default_rng([seed, 3]).standard_normal
    q, k = (normal((ROWS, LENGTH, groups, WIDTH)) for _ in range(2))
    q, k = (x / np.linalg.norm(x, axis=-1, keepdims=True) for x in (q, k))
    v = normal((ROWS, LENGTH, heads, VALUES))
    g = np.log(sigmoid(normal((ROWS, LENGTH, heads)) + 2.0))
    weights = 1.0 + 0.3 * normal((ROWS, LENGTH, heads, levels(LENGTH)))
    return *(hosted(x) for x in (q, k, v, g, weights)), hosted(normal((ROWS, LENGTH, heads, VALUES)))


@pytest.mark.parametrize('packed', (False, True))
def test_the_masked_form_is_as_exact_as_flas_level_by_level(packed):
    """The output and every operand's gradient, the level weights' included, of the masked form (each key's
    level its position's highest bit apart from the query's) on par with fla's reference building each
    level's mask, from float64, one sequence and packed documents counting positions from their own first
    tokens (check 1)."""
    # Packed, each row's documents bound its levels' blocks (`max_segments`).
    dew = functools.partial(log_linear_attention, max_segments=max(map(len, DOCUMENTS))) if packed else \
        log_linear_attention
    assert_packed_on_par(dew, naive, draw, 'log-linear attention', packed=packed, documents=DOCUMENTS,
                         length=LENGTH, seeds=SEEDS[packed])


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


@pytest.mark.parametrize("target", ["sm80", "sm90", "sm100", "sm120"])
def test_log_linear_levels_train_on_the_template_on_each_gpu_family(target):
    """A log-linear call of Kimi's geometry (64 value heads on one key head of 128, bf16), traced as each
    GPU family's device: each level's recurrence takes the recurrence template for its forward and its
    gradient (`generation.routes`), where a key head's value heads summed in one program held sm120's
    consumer-fed scores' gradient launch past 255 registers and every level trained on the portable math,
    1091 ms of a step against its forward's 4.6 (row 10522)."""
    from dew.nn.kernels import generation

    length, heads = 512, 64
    shapes = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in (
        ((1, length, 1, 128), jnp.bfloat16), ((1, length, 1, 128), jnp.bfloat16),
        ((1, length, heads, 64), jnp.bfloat16), ((1, length, heads), jnp.float32),
        ((1, length, heads, levels(length)), jnp.bfloat16))]

    def call(*operands):
        return log_linear_attention(*operands, scale=1.0)

    with generation.assumed(target), generation.routes() as found:
        jax.eval_shape(lambda *x: jax.vjp(call, *x)[1](call(*x)), *shapes)
    taken = {(route.kernel, None if route.refusal is None else route.refusal.code)
             for route in found if route.choice == generation.Choice.GATED_DELTA_RULE}
    assert taken == {("mosaic_gpu", None)}, taken


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
