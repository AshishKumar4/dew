"""Parallax attention on the template's walk (`attention_parallax`, docs/design/kernel-templates.md row 54):
its output and its five operands' cotangents on par with its reference's (`attention_parallax.reference`)
against float64, over the span forms a call takes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_forms import ROUNDED
from contract_support import normal_draw
from reference_error import FEWEST_SEEDS, assert_as_exact

from dew.nn.kernels.templates import attention as template, attention_parallax

QUERIES, KEYS, HEADS, KV_HEADS, WIDTH, SECOND, VALUES = 48, 48, 4, 2, 16, 8, 16

def _runs() -> jnp.ndarray:
    """Two runs a query `[1, Q, 2, 2]`: the first key (a sink), and the window of 8 keys up to its own past
    it."""
    index = np.arange(QUERIES)
    sink = np.stack([np.zeros_like(index), np.ones_like(index)], axis=-1)
    window = np.stack([np.maximum(index - 7, 1), np.maximum(index + 1, 1)], axis=-1)
    return jnp.asarray(np.stack([sink, window], axis=1)[None], jnp.int32)


SPANS = {
    "full": lambda: template.spans(QUERIES, KEYS, 1),
    "causal": lambda: template.spans(QUERIES, KEYS, 1, causal=True),
    "window": lambda: template.spans(QUERIES, KEYS, 1, causal=True, sliding_window=8),
    "documents": lambda: template.spans(QUERIES, KEYS, 1, causal=True, segment_ids=jnp.repeat(
        jnp.arange(3, dtype=jnp.int32), QUERIES // 3)[None]),
    "runs": lambda: template.spans(QUERIES, KEYS, 1, runs=_runs()),
}


def drawing(dtype):
    """A draw's query, second query, keys, second keys, values and the output's cotangent in `dtype`."""
    def draw(seed):
        shapes = ((1, QUERIES, HEADS, WIDTH), (1, QUERIES, HEADS, SECOND), (1, KEYS, KV_HEADS, WIDTH),
                  (1, KEYS, KV_HEADS, SECOND), (1, KEYS, KV_HEADS, VALUES), (1, QUERIES, HEADS, VALUES))
        return normal_draw([54, seed], *shapes, dtype=dtype)

    return draw


def differentiated(attend, spans):
    """`attend`'s output and its five operands' cotangents under the draw's output cotangent."""
    def run(query, second, key, second_key, value, cotangent):
        out, pull = jax.vjp(lambda *operands: attend(*operands, spans), query, second, key, second_key, value)
        return out, pull(cotangent)

    return run


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("form", sorted(SPANS))
def test_parallax_and_its_derivative_are_on_par_with_its_reference(form, dtype):
    """Parallax over 48 queries of 4 heads by 48 keys of 2 (a key head's two query heads), widths 16, 8 and
    16 (the queries and keys, the second products', the values), its blocks' padding unread: its output and
    its query's, second query's, keys', second keys' and values' cotangents (`attention_parallax.attend`:
    its key tiles' moments merged by Chan's law; the derivative from each row's moments) on par with the
    reference's against float64, whole, causal, in a sliding window of 8 and over three documents."""
    spans = SPANS[form]()
    assert_as_exact(jax.jit(differentiated(attention_parallax.attend, spans)),
                    jax.jit(differentiated(attention_parallax.reference, spans)), drawing(dtype),
                    f"parallax {form} {jnp.dtype(dtype).name}", seeds=FEWEST_SEEDS)


def test_a_query_that_sees_no_key_reads_zero_and_its_cotangents_none():
    """A query whose keys are none (its document's keys all past it: spans of no key), its output exactly 0
    and every cotangent finite, none flowing through it."""
    # Query 0 sees no key; each other its keys up to itself.
    spans = template.KeySpans(jnp.asarray([[1, 0, 0, 0]], jnp.int32), jnp.asarray([[1, 2, 3, 4]], jnp.int32),
                              None, None, 4)
    query, second, key, second_key, value, cotangent = normal_draw(
        [54, 1], (1, 4, 1, 4), (1, 4, 1, 2), (1, 4, 1, 4), (1, 4, 1, 2), (1, 4, 1, 4), (1, 4, 1, 4))
    out, grads = differentiated(attention_parallax.attend, spans)(query, second, key, second_key, value,
                                                                  cotangent.at[:, 1:].set(0))
    np.testing.assert_array_equal(np.asarray(out)[0, 0], 0.0)
    for grad in grads:
        np.testing.assert_array_equal(np.asarray(grad), 0.0)


def _one_key(second: float, value: float, width: int = 4):
    """A query at place 0 of a causal call over two keys (its first alone visible), its second product
    `second` (a second query of `second` against a second key of 1), its visible value `value`."""
    spans = template.spans(2, 2, 1, causal=True)
    query = jnp.zeros((1, 2, 1, width), jnp.float32)
    second_query = jnp.zeros((1, 2, 1, 1), jnp.float32).at[0, 0, 0, 0].set(second)
    key = jnp.zeros((1, 2, 1, width), jnp.float32)
    second_key = jnp.ones((1, 2, 1, 1), jnp.float32)
    values = jnp.full((1, 2, 1, width), value, jnp.float32)
    return query, second_query, key, second_key, values, spans


def test_a_query_of_one_key_reads_its_value_whatever_its_second_product():
    """One visible key of second product 2^20 and value 2^120: Parallax reads the value exactly, its moments
    centred (`attention_parallax.moments`: one key's covariance is 0), where `a (1 + c)` and `b` each
    overflowed to inf and their difference read NaN; and at a second product of 2^24 its value's cotangent
    is the output's, its weight `1 + (c - t)` 1 exactly, where `(1 + c) - t` rounded to 0."""
    *operands, spans = _one_key(2.0 ** 20, 2.0 ** 120)
    out = attention_parallax.attend(*operands, spans)
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0], 2.0 ** 120)
    *operands, spans = _one_key(2.0 ** 24, 1.0)
    _, pull = jax.vjp(lambda value: attention_parallax.attend(*operands[:4], value, spans), operands[4])
    cotangent = jnp.zeros((1, 2, 1, 4), jnp.float32).at[0, 0, 0].set(1.0)
    np.testing.assert_array_equal(np.asarray(pull(cotangent)[0])[0, 0, 0], 1.0)


def test_a_hidden_keys_overflowing_second_product_moves_no_visible_output():
    """A query at place 0 of second query 2^100 over a visible second key of 0 and a hidden one of 2^40 (its
    product past fp32's largest), one visible value of 1: the output 1, the hidden key's product masked
    before it weighs anything (`attention_parallax.moments`), where `0 · inf` read NaN."""
    query, _, key, _, value, spans = _one_key(0.0, 1.0)
    second = jnp.zeros((1, 2, 1, 1), jnp.float32).at[0, 0, 0, 0].set(2.0 ** 100)
    second_key = jnp.zeros((1, 2, 1, 1), jnp.float32).at[0, 1, 0, 0].set(2.0 ** 40)
    out = attention_parallax.attend(query, second, key, second_key, value, spans)
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0], 1.0)


def _two_keys(seconds: tuple[float, float], values: tuple[float, float], width: int = 4):
    """One query over two keys it weighs alike (zero logits), their second products `seconds` (a second
    query of 1 against second keys of `seconds`) and their values `values`."""
    spans = template.spans(1, 2, 1)
    query = jnp.zeros((1, 1, 1, width), jnp.float32)
    second = jnp.ones((1, 1, 1, 1), jnp.float32)
    key = jnp.zeros((1, 2, 1, width), jnp.float32)
    second_key = jnp.asarray(seconds, jnp.float32)[None, :, None, None]
    value = jnp.broadcast_to(jnp.asarray(values, jnp.float32)[None, :, None, None], (1, 2, 1, width))
    return query, second, key, second_key, value, spans


def test_two_keys_second_products_apart_by_less_than_their_size_keep_their_spread():
    """Two keys weighed alike of second products `2^25` and `2^25 + 4` (fp32 apart by 4, their mean `2^25 +
    2` not an fp32) over values 0 and 1: Parallax reads `1/2 (1 + c - t)` of the second, -0.5, and a unit
    cotangent gives the values `1/2 (1 + c - t)`, 1.5 and -0.5, its mean a pair whose sum holds `2^25 + 2`
    (`attention_parallax.Moments.centre`) through the forward and the derivative, where an absolute mean
    rounded to `2^25` and the derivative read 0.5 and -1.5."""
    *operands, spans = _two_keys((2.0 ** 25, 2.0 ** 25 + 4), (0.0, 1.0))
    out, pull = jax.vjp(lambda value: attention_parallax.attend(*operands[:4], value, spans), operands[4])
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0], -0.5)
    (dvalue,) = pull(jnp.ones_like(out))
    np.testing.assert_array_equal(np.asarray(dvalue)[0, :, 0, 0], [1.5, -0.5])


def test_two_keys_second_products_at_fp32s_ends_read_their_finite_output():
    """Two keys weighed alike of second products `2^127` and `-2^127` over values 0 and `2^-120`: the
    second's weight `1/2 (1 + c - t) = 1/2 (1 + 2^127)` times its value, 64, its second products held at
    the row's power (`attention_parallax.second_powers`), where `t - anchor` overflowed to -inf."""
    *operands, spans = _two_keys((2.0 ** 127, -(2.0 ** 127)), (0.0, 2.0 ** -120))
    out = attention_parallax.attend(*operands, spans)
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0], 64.0)


def _equal_keys(seconds: tuple[float, ...], values: tuple[float, ...], width: int = 4):
    """One query of zeros over keys it weighs alike (its logits all 0, its keys the first `len(seconds)`
    unit vectors, so its query's cotangent reads each logit's), their second products `seconds` (a second
    query of 1 against them) and their values `values` in the first channel."""
    count = len(seconds)
    spans = template.spans(1, count, 1)
    query = jnp.zeros((1, 1, 1, width), jnp.float32)
    second = jnp.ones((1, 1, 1, 1), jnp.float32)
    key = jnp.eye(count, width, dtype=jnp.float32)[None, :, None]
    second_key = jnp.asarray(seconds, jnp.float32)[None, :, None, None]
    value = jnp.zeros((1, count, 1, width), jnp.float32).at[0, :, 0, 0].set(jnp.asarray(values))
    return query, second, key, second_key, value, spans


def test_four_keys_second_products_at_fp32s_ends_sum_in_range():
    """Four keys weighed alike of second products `2^127` and three of `-2^127`, the first's value `2^-120`:
    Parallax reads `a - C = -48`, its second products held at the row's power (`attention_parallax.
    second_powers`: `2^-4`) and its covariance's terms at weights summing to at most 1 (`moments`), where
    in halves the three products' sum about the first, `-3 · 2^126`, overflowed."""
    *operands, spans = _equal_keys((2.0 ** 127, -(2.0 ** 127), -(2.0 ** 127), -(2.0 ** 127)),
                                   (2.0 ** -120, 0.0, 0.0, 0.0))
    out = attention_parallax.attend(*operands, spans)
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0, 0], -48.0)


def test_two_keys_second_products_at_fp32s_ends_keep_every_cotangent_finite():
    """Two keys weighed alike of second products `±2^127` over values 0 and `2^-120`, a unit cotangent on
    the output's first channel: every cotangent finite and exact, each term a weight decayed onto its
    moment's product (`attention_parallax.exp_scaled`) and a key's `w (g · C - δ τ)` cancelled before its
    `w δ` meets it (`attention_parallax.derivative`), where `dw - D` read `inf - inf`: the logits'
    cotangents `∓2^-122` (the query's `∓2^-123` at scale 1/2), the second products' `±2^-122` (the second
    query's 64, the second keys' `±2^-122`), the values' `∓2^126`, the keys' 0 (a query of 0)."""
    *operands, spans = _equal_keys((2.0 ** 127, -(2.0 ** 127)), (0.0, 2.0 ** -120))
    out, pull = jax.vjp(lambda *operands: attention_parallax.attend(*operands, spans), *operands)
    dq, dr, dk, dk2, dv = pull(jnp.zeros_like(out).at[0, 0, 0, 0].set(1.0))
    np.testing.assert_array_equal(np.asarray(dq)[0, 0, 0], [-(2.0 ** -123), 2.0 ** -123, 0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(dr)[0, 0, 0], [64.0])
    np.testing.assert_array_equal(np.asarray(dk), 0.0)
    np.testing.assert_array_equal(np.asarray(dk2)[0, :, 0, 0], [2.0 ** -122, -(2.0 ** -122)])
    np.testing.assert_array_equal(np.asarray(dv)[0, :, 0, 0], [-(2.0 ** 126), 2.0 ** 126])


def test_four_keys_values_cotangents_meet_their_weight_before_the_unit():
    """Four keys weighed alike of second products `3 · 2^126` and three of `-3 · 2^126`, the first's value
    `2^-120`, a unit cotangent on the output's first channel: the first value's cotangent `w (1 + c - t) =
    -9·2^123`, its weight met by `t - c` as powers of two before either rounds
    (`attention_parallax.exp_scaled`: `w - w τ`), where `c - t = -9·2^125` overflowed first; the output
    -72."""
    big = 3.0 * 2.0 ** 126
    *operands, spans = _equal_keys((big, -big, -big, -big), (2.0 ** -120, 0.0, 0.0, 0.0))
    out, pull = jax.vjp(lambda value: attention_parallax.attend(*operands[:4], value, spans), operands[4])
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0, 0], -72.0)
    (dvalue,) = pull(jnp.zeros_like(out).at[0, 0, 0, 0].set(1.0))
    np.testing.assert_array_equal(np.asarray(dvalue)[0, 0, 0, 0], -9.0 * 2.0 ** 123)


def test_four_keys_logits_cotangents_meet_their_weight_before_the_unit():
    """Four keys weighed alike of second products `3 · 2^126` and three of `-3 · 2^126`, the first's value
    2, a unit cotangent on the output's first channel: the first key's logit's cotangent `-9·2^123` (the
    query's `-9·2^122` at scale 1/2), its weight met by `δ τ` and `g · C` before either rounds
    (`attention_parallax.derivative`), where `δ τ`, about `-1.5 · 9 · 2^125`, overflowed first."""
    big = 3.0 * 2.0 ** 126
    *operands, spans = _equal_keys((big, -big, -big, -big), (2.0, 0.0, 0.0, 0.0))
    out, pull = jax.vjp(lambda query: attention_parallax.attend(query, *operands[1:], spans), operands[0])
    (dquery,) = pull(jnp.zeros_like(out).at[0, 0, 0, 0].set(1.0))
    np.testing.assert_array_equal(np.asarray(dquery)[0, 0, 0, 0], -9.0 * 2.0 ** 122)


def _float64(spans):
    """Parallax's float64 truth over `spans`, apart from any of Dew's arithmetic: `(query, second, key,
    second_key, value, cotangent)` to the output `[B, Q, H, Dv]`, the values' mean less their centred
    covariance with the second products (where `Σ w (1 + c - t) v` lost its `1` beside a second product
    of `2^140` even in float64), and the values' cotangent `Σ w (1 + c - t) g`."""
    def truth(query, second, key, second_key, value, cotangent):
        group = query.shape[2] // key.shape[2]
        key, second_key, value = (jnp.repeat(x, group, axis=2) for x in (key, second_key, value))
        seen = template.dense(spans)[:, None]
        logits = jnp.where(seen, jnp.einsum("bqhd,bkhd->bhqk", query, key) * query.shape[-1] ** -0.5,
                           -jnp.inf)
        weights = jax.nn.softmax(logits, axis=-1)
        seconds = jnp.where(seen, jnp.einsum("bqhd,bkhd->bhqk", second, second_key), 0.0)
        apart = seconds - jnp.sum(weights * seconds, axis=-1, keepdims=True)
        mean = jnp.einsum("bhqk,bkhd->bhqd", weights, value)
        centred = value[:, None] - jnp.swapaxes(mean, 1, 2)[:, :, None]
        out = jnp.swapaxes(mean, 1, 2) - jnp.einsum("bhqk,bqkhd->bqhd", weights * apart, centred)
        return out, jnp.einsum("bhqk,bqhd->bkhd", weights * (1 - apart), cotangent)

    return truth


def _logits_of(logits, width: int = 4):
    """A query `e0` and keys `2 l e0` at scale `1/2` (width 4): the logits `l`."""
    count = len(logits)
    query = jnp.zeros((1, 1, 1, width), jnp.float32).at[0, 0, 0, 0].set(1.0)
    key = jnp.zeros((1, count, 1, width), jnp.float32).at[0, :, 0, 0].set(2.0 * jnp.asarray(logits))
    return query, key


def _witness(logits, seconds, values, width: int = 4, *, second: float = 1.0, visible: int | None = None):
    """One query over keys of `logits` (`_logits_of`), second products `second · seconds` (a second query of
    `second` against second keys `seconds`) and values `values` in every channel, its first `visible` keys
    visible (every one by default)."""
    query, key = _logits_of(logits, width)
    count = len(logits)
    second = jnp.full((1, 1, 1, 1), second, jnp.float32)
    second_key = jnp.asarray(seconds, jnp.float32)[None, :, None, None]
    value = jnp.broadcast_to(jnp.asarray(values, jnp.float32)[None, :, None, None], (1, count, 1, width))
    spans = (template.spans(1, count, 1) if visible is None else template.KeySpans(
        jnp.zeros((1, 1), jnp.int32), jnp.full((1, 1), visible, jnp.int32), None, None, count))
    return query, second, key, second_key, value, spans


def _second_of(call, second: float, second_key: float):
    """`call`'s operands (`_one_key`'s) with a second query of `second` at its first place and second keys
    of `second_key`."""
    query, _, key, _, value, spans = call
    return (query, jnp.zeros((1, 2, 1, 1), jnp.float32).at[0, 0, 0, 0].set(second), key,
            jnp.full((1, 2, 1, 1), second_key, jnp.float32), value, spans)


WITNESSES = {
    # Train 1407's review of the native Parallax (a4e57bf1f), each P0's own.
    "an overflowing second product's one key": lambda: _second_of(_one_key(0.0, 1.0), 2.0 ** 100, 2.0 ** 40),
    "signed second weights at fp32's ends": lambda: _witness((0.0, 0.0), (2.0 ** 127, -(2.0 ** 127)),
                                                             (8.0, 8.0)),
    "a covariance cancelling past fp32": lambda: _witness((0.0, -1.0), (-(2.0 ** 127), 2.0 ** 127),
                                                          (128.0, 128.0)),
    "a covariance of constant values of 2^100": lambda: _witness((0.0, -1.0), (-(2.0 ** 127), 2.0 ** 127),
                                                                 (2.0 ** 100, 2.0 ** 100)),
    "a softmax tail's second product": lambda: _witness((0.0, -128.0), (0.0, 2.0 ** 126), (0.0, 1.0)),
    "a later tile dominating the first": lambda: _witness(
        (0.0,) * 128 + (256.0, 256.0), (2.0 ** 100,) + (0.0,) * 128 + (4.0,), (0.0,) * 129 + (1.0,)),
    # The tail across tiles (the owner's note): a first tile at logit -128 whose key 0 carries `2^126`, a
    # second at 0.
    "a tile's tail across tiles": lambda: _witness(
        (-128.0,) * 128 + (0.0,) * 128, (2.0 ** 126,) + (0.0,) * 255, (1.0,) + (0.0,) * 255),
    # Train 1407's review of the moments' walk (0a1a346c2): a centre's low word across an empty tile, a
    # value's softmax tail, and values' weights past fp32 under a small cotangent.
    "a centre's low word across an empty tile": lambda: _witness(
        (0.0,) * 129, (2.0 ** 25, 2.0 ** 25 + 4) + (0.0,) * 127, (0.0, 1.0) + (0.0,) * 127, visible=2),
    "a value's softmax tail": lambda: _witness((0.0, -128.0), (0.0, 0.0), (0.0, 2.0 ** 127)),
    "values' weights past fp32 under a small cotangent": lambda: _witness(
        (0.0, 0.0), (0.0, 2.0 ** 40), (0.0, 2.0 ** -100), second=2.0 ** 100),
    # Its review of 68a95bc01: a tile's share past 1 (128 keys at -128) over values of `2^127`.
    "a tile's value tail across tiles": lambda: _witness(
        (-128.0,) * 128 + (0.0,), (0.0,) * 129, (2.0 ** 127,) * 128 + (0.0,)),
    # Its review of aea29e3fb: a cross term decayed by `e^-256` and raised by `2^254` (one live key a tile,
    # the first tile's others weighing nothing), and a tail's values' weight under a cotangent of `2^127`.
    "a cross term decayed past fp32": lambda: _witness(
        (-256.0,) + (-10000.0,) * 127 + (0.0,), (2.0 ** 127,) + (0.0,) * 128, (2.0 ** 127,) + (0.0,) * 128),
    "a tail's values' weight under a large cotangent": lambda: _witness((0.0, -128.0), (0.0, 0.0),
                                                                        (0.0, 0.0)),
    # Its review of 0fafa2dcb: a tail's values' weight past fp32 under a small cotangent.
    "a tail's values' weight past fp32 under a small cotangent": lambda: _witness(
        (0.0, -81.0), (0.0, 2.0 ** 125), (0.0, 0.0), second=2.0 ** 125),
    # Train 1444's review of ef337c837: a tail's second product of `2^200` in the centre, `2^200 e^-128`,
    # the dominant value's cotangent `(1 + c) / (1 + e^-128)`, about 41335.
    "a tail's second product in the centre": lambda: _witness((0.0, -128.0), (0.0, 2.0 ** 100), (0.0, 0.0),
                                                              second=2.0 ** 100),
}
COTANGENTS = {"values' weights past fp32 under a small cotangent": 2.0 ** -100,
              "a tail's values' weight under a large cotangent": 2.0 ** 127,
              "a tail's values' weight past fp32 under a small cotangent": 2.0 ** -100}
"""The output's cotangent of a witness (`WITNESSES`) where it is not 1."""
"""Calls whose output float64 holds and fp32 can: each a finding of train 1407's reviews of the native
Parallax (a4e57bf1f) and of the moments' walk."""
TAILS = frozenset({
    "a tail's values' weight past fp32 under a small cotangent",
    'a cross term decayed past fp32',
    "a softmax tail's second product",
    "a tail's values' weight under a large cotangent",
    "a tile's tail across tiles",
    "a tile's value tail across tiles",
    "a value's softmax tail",
})
"""The witnesses whose truth is a softmax tail's exponential, which no fp32 arithmetic rounds once: held by
check 1 against the dense reference, every other to float64 rounded once (`ROUNDED`) (Main's ruling of
2026-11-27 on Parallax P-II's gate)."""


def _value_cotangent(attend, spans):
    """`attend`'s output and its values' cotangent under the draw's output cotangent."""
    def run(query, second, key, second_key, value, cotangent):
        out, pull = jax.vjp(lambda value: attend(query, second, key, second_key, value, spans), value)
        return out, pull(cotangent)[0]

    return run


@pytest.mark.parametrize("arm", ["walk", "reference"])
@pytest.mark.parametrize("case", sorted(set(WITNESSES) - TAILS))
def test_parallax_reads_float64s_output_at_fp32s_ends(case, arm):
    """Parallax's output and its values' cotangent (`attention_parallax.attend`, its walk and its first
    derivative; and `attention_parallax.reference`, every higher order's) float64's rounded once (check 1,
    `ROUNDED`) at calls of second products and values at fp32's ends (`WITNESSES`, each exact or one
    rounding of float64's): one key whose second product `2^100 · 2^40` passes fp32 (its second query
    taken by its row's power first, `second_powers`), signed second weights of `±2^127`, a covariance
    whose raw sums pass fp32 where it is 0, a tile of logits 256 after one of 0 whose key carries `2^100`
    (Chan's merge, `merged`), a centre `2^25 + 2` held as its pair across an empty tile (`merged`: its
    values' cotangent 1.5 and -0.5, where flattened it read 0.5 and -1.5), and values' weights of `±2^138`
    under a cotangent of `2^-100` (`derivative`: `±2^38`, where the weights alone overflowed)."""
    *operands, spans = WITNESSES[case]()
    cotangent = jnp.full((1, operands[0].shape[1], operands[0].shape[2], operands[4].shape[-1]),
                         COTANGENTS.get(case, 1.0), jnp.float32)
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    assert_as_exact(jax.jit(_value_cotangent(attend, spans)), ROUNDED, lambda seed: (*operands, cotangent),
                    f"parallax {case} ({arm})", seeds=FEWEST_SEEDS, truth_fn=jax.jit(_float64(spans)))


@pytest.mark.parametrize("case", sorted(TAILS))
def test_parallax_keeps_a_softmax_tails_output_on_par_with_its_reference(case):
    """Parallax's output and its values' cotangent (`attention_parallax.attend`) on par with its dense
    reference's (`attention_parallax.reference`) against float64 (check 1) at a softmax tail's corners
    (`TAILS`): a weight of `e^-128` against a second product of `2^126` (`exp_scaled`), a tile at logit -128
    whose `2^126` meets the next tile's share before it rounds, a weight of `e^-128` over a value of `2^127`
    (the mean's raised tier), a tile's share past 1 over values of `2^127`, a cross term decayed by `e^-256`
    and raised by `2^254` (every power with its decay at once), and a tail's values' weight under a cotangent
    of `2^127` (the values' raised tier). The reference's derivative is autodiff of its moments, whose tail
    tiers' powers meet the cotangent after their products: under a cotangent of `2^127` its tail's values'
    cotangent reads 0, as the plain fp32 formula's does, within every order's finite contract
    (`test_parallax_holds_every_order`; Main's ruling of 2026-11-27)."""
    *operands, spans = WITNESSES[case]()
    cotangent = jnp.full((1, operands[0].shape[1], operands[0].shape[2], operands[4].shape[-1]),
                         COTANGENTS.get(case, 1.0), jnp.float32)
    assert_as_exact(jax.jit(_value_cotangent(attention_parallax.attend, spans)),
                    jax.jit(_value_cotangent(attention_parallax.reference, spans)),
                    lambda seed: (*operands, cotangent), f"parallax {case}", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(_float64(spans)))
    # The walk's zeros float64's rounded to fp32 alone, where the reference may share a loss check 1
    # compares against: each leaf a value wherever its truth is one.
    with jax.enable_x64(new_val=True):
        truth = _float64(spans)(*(np.asarray(x, np.float64) for x in (*operands, cotangent)))
    for found, true in zip(_value_cotangent(attention_parallax.attend, spans)(*operands, cotangent), truth,
                           strict=True):
        np.testing.assert_array_equal(np.asarray(found) != 0, np.asarray(true, np.float32) != 0)


def test_a_softmax_tails_second_product_moves_its_tangent():
    """A weight of `e^-128` against a second product of `2^126` over values 0 and 1: the output's tangent
    along the second query (forward mode, `generation.linear_tangent`) on par with its reference's against
    float64 (check 1), `Σ w (a - v) k2`, each `w t` decayed whole (`attention_parallax.exp_scaled`), where the
    weight alone underflowed and it read 0 (train 1407's review)."""
    *operands, spans = _witness((0.0, -128.0), (0.0, 2.0 ** 126), (0.0, 1.0))

    def along(attend):
        def run(query, second, key, second_key, value, direction):
            return jax.jvp(lambda r: attend(query, r, key, second_key, value), (second,), (direction,))[1]

        return run

    def ours(query, r, key, second_key, value):
        return attention_parallax.attend(query, r, key, second_key, value, spans)

    def reference(query, r, key, second_key, value):
        return attention_parallax.reference(query, r, key, second_key, value, spans)

    draw = (*operands, jnp.ones_like(operands[1]))
    assert_as_exact(jax.jit(along(ours)), jax.jit(along(reference)), lambda seed: draw,
                    "parallax's tail tangent", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(along(_centred64(spans))))


def test_values_of_fp32s_largest_under_a_unit_cotangent_keep_every_cotangent_finite():
    """Two keys weighed alike (a query of 0 over unit keys), second products 0, values `2^127` in every
    channel, a unit cotangent: the query's and keys' cotangents exactly 0 and every cotangent finite, the
    cotangent and the values each taken by their power first (`attention_parallax._held`: `δ = 0`), where
    `Σ g v` overflowed and `0 · inf` read NaN (train 1407's review)."""
    *operands, spans = _equal_keys((0.0, 0.0), (2.0 ** 127, 2.0 ** 127))
    value = jnp.full_like(operands[4], 2.0 ** 127)
    out, pull = jax.vjp(lambda *xs: attention_parallax.attend(*xs, spans), *operands[:4], value)
    grads = pull(jnp.ones_like(out))
    np.testing.assert_array_equal(np.asarray(grads[0]), 0.0)
    np.testing.assert_array_equal(np.asarray(grads[2]), 0.0)
    for grad in grads:
        assert np.isfinite(np.asarray(grad)).all()


def test_a_call_of_no_keys_reads_zero_in_every_order():
    """A call over no keys (`KeySpans` of 0): the output 0, every cotangent 0 (the keys' empty), and its
    tangent 0, the walk taking one tile of unseen padding, where a tile of 128 sliced from no keys failed
    to trace and the reference divided by its 0 keys (train 1407's review)."""
    spans = template.KeySpans(jnp.zeros((1, 2), jnp.int32), jnp.zeros((1, 2), jnp.int32), None, None, 0)
    query, second, cotangent = normal_draw([54, 2], (1, 2, 2, 4), (1, 2, 2, 2), (1, 2, 2, 4))
    key, second_key, value = (jnp.zeros((1, 0, 1, width), jnp.float32) for width in (4, 2, 4))
    operands = (query, second, key, second_key, value)
    for attend in (attention_parallax.attend, attention_parallax.reference):
        out, pull = jax.vjp(lambda *xs, attend=attend: attend(*xs, spans), *operands)
        np.testing.assert_array_equal(np.asarray(out), 0.0)
        for grad in pull(cotangent):
            np.testing.assert_array_equal(np.asarray(grad), 0.0)
        _, tangent = jax.jvp(lambda *xs, attend=attend: attend(*xs, spans), operands, operands)
        np.testing.assert_array_equal(np.asarray(tangent), 0.0)


def _centred64(spans):
    """Parallax's output in float64 from its operands, written apart from Dew's (`_float64`'s, its values'
    cotangent aside): what check 1's truth differentiates."""
    def out(query, second, key, second_key, value):
        return _float64(spans)(query, second, key, second_key, value, jnp.zeros_like(query[..., :1]))[0]

    return out


def test_parallax_across_key_tiles_and_their_empty_ones_is_on_par_with_float64():
    """Parallax over 300 queries of 2 heads by 300 keys of 1 in a causal window of 8 (each row's keys
    within a tile or across a tile's edge, every other tile of its row empty), its output and its five
    operands' cotangents on par with its reference's (check 1) against float64's of the plain centred
    formula written apart (`_float64`), so its tiles' merges (`attention_parallax.merged`) and their empty
    parts are measured."""
    length = 300
    spans = template.spans(length, length, 1, causal=True, sliding_window=8)

    def draw(seed):
        return normal_draw([5454, seed], (1, length, 2, 16), (1, length, 2, 8), (1, length, 1, 16),
                           (1, length, 1, 8), (1, length, 1, 16), (1, length, 2, 16))

    assert_as_exact(jax.jit(differentiated(attention_parallax.attend, spans)),
                    jax.jit(differentiated(attention_parallax.reference, spans)), draw,
                    "parallax across key tiles", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(differentiated(lambda *xs: _centred64(spans)(*xs[:5]), spans)))


def _every_cotangent(attend, spans):
    """`attend`'s output and its five operands' cotangents under the draw's output cotangent."""
    def run(query, second, key, second_key, value, cotangent):
        out, pull = jax.vjp(lambda *xs: attend(*xs, spans), query, second, key, second_key, value)
        return out, *pull(cotangent)

    return run


def _plain(query, second, key, second_key, value, spans):
    """Parallax as plain dense fp32 attention computes it (`_centred64`'s formula at its operands' dtype):
    the reference check 1 holds a softmax tail's witness to, which loses it as fp32 attention does."""
    return _centred64(spans)(query, second, key, second_key, value)


def _on_par_with_plain_fp32(operands, spans, cotangent, label: str, reference=_plain) -> None:
    """Parallax's output and five cotangents (`attention_parallax.attend`) on par with a dense fp32
    reference's (`reference`: plain fp32 attention's, `_plain`, by default) against float64 (check 1), and
    nonzero wherever float64's rounded to fp32 is."""
    assert_as_exact(jax.jit(_every_cotangent(attention_parallax.attend, spans)),
                    jax.jit(_every_cotangent(reference, spans)), lambda seed: (*operands, cotangent), label,
                    seeds=FEWEST_SEEDS, truth_fn=jax.jit(_every_cotangent(_plain, spans)))
    with jax.enable_x64(new_val=True):
        truth = _every_cotangent(_plain, spans)(*(np.asarray(x, np.float64) for x in (*operands, cotangent)))
    found = _every_cotangent(attention_parallax.attend, spans)(*operands, cotangent)
    for leaf, true in zip(found, truth, strict=True):
        np.testing.assert_array_equal(np.asarray(leaf) != 0, np.asarray(true, np.float32) != 0)


def test_a_value_tails_logit_cotangents_keep_it():
    """A weight of `e^-128` over a value of `2^125` beside a key of weight 1 and value 0, a unit cotangent:
    every cotangent on par with its dense reference's (`attention_parallax.reference`) against float64
    (check 1, Main's ruling on a tail's gradients), the query's and keys' nonzero, `g · a` summed with each
    `g · v` decayed whole (`attention_parallax.derivative`), where `e^-128` rounded to 0 first and the
    dominant key's logit cotangent, about `-2^127 e^-128`, read 0 (train 1407's review of 68a95bc01, held to
    check 1 by train 1423's)."""
    *operands, spans = _witness((0.0, -128.0), (0.0, 0.0), (0.0, 2.0 ** 125))
    _on_par_with_plain_fp32(operands, spans, jnp.ones((1, 1, 1, 4), jnp.float32),
                            "parallax's value tail's cotangents",
                            reference=attention_parallax.reference)


def test_a_tiles_covariance_past_fp32_comes_back_by_its_share():
    """Three visible keys at places 0, 1 and 128 (the rest hidden), logits `[-128, -128, 0]`, second
    products and values both `[2^127, -2^127, 0]`: the output float64's on par (check 1) and nonzero,
    `-2^255 e^-128 / (1 + 2 e^-128)`, on par with its dense reference's (`attention_parallax.reference`, plain
    fp32 attention reading infinity), the first tile's covariance `2^254` carried with its power through
    the merge (`attention_parallax.Moments.spread_power`), where it landed as infinity first (train 1423's
    review)."""
    logits = (-128.0, -128.0) + (0.0,) * 126 + (0.0,)
    seconds = (2.0 ** 127, -(2.0 ** 127)) + (0.0,) * 127
    query, second, key, second_key, value, _ = _witness(logits, seconds, seconds)
    valid = jnp.zeros((1, 129), bool).at[0, jnp.asarray([0, 1, 128])].set(True)
    spans = template.spans(1, 129, 1, valid=valid)
    # Plain fp32 attention forms `(t - c)(v - a) = 2^254` and reads infinity: the dense reference holds it.
    _on_par_with_plain_fp32((query, second, key, second_key, value[..., :1]), spans,
                            jnp.ones((1, 1, 1, 1), jnp.float32), "parallax's covariance across tiles",
                            reference=attention_parallax.reference)


def test_a_hidden_key_scales_no_visible_second_product():
    """A second query `[2^127, 2^-100]` against visible second keys `(0, 0)` and `(0, 2^100)` and a hidden
    `(2^127, 0)`, equal logits, values `[0, 1]`: second products `[0, 1]`, the output `1/2 - 1/4 = 1/4`,
    its rows' power bounded over the pairs its query sees (`attention_parallax.second_powers`), where the
    hidden key took `2^-132` and erased the visible product (train 1423's review)."""
    query = jnp.zeros((1, 1, 1, 4), jnp.float32).at[0, 0, 0, 0].set(1.0)
    key = jnp.zeros((1, 3, 1, 4), jnp.float32)
    second = jnp.asarray([2.0 ** 127, 2.0 ** -100], jnp.float32)[None, None, None]
    second_key = jnp.asarray([[0.0, 0.0], [0.0, 2.0 ** 100], [2.0 ** 127, 0.0]], jnp.float32)[None, :, None]
    value = jnp.asarray([0.0, 1.0, 0.0], jnp.float32)[None, :, None, None]
    spans = template.spans(1, 3, 1, valid=jnp.asarray([[True, True, False]]))
    out = attention_parallax.attend(query, second, key, second_key, value, spans)
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0, 0], 0.25)


def test_a_cotangents_large_entry_erases_no_live_pair():
    """Equal logits, second products 0, a cotangent `[2^127, 2^-100]` over values `(0, 0)` and `(0, 2^100)`:
    `g · v = [0, 1]`, the logits' cotangents `[-1/4, 1/4]`, so the keys' `[-1/8, 1/8]` at a unit query and
    scale `1/2`, the cotangent's products held at its row's largest pair (`attention_parallax.derivative`),
    where its own largest entry took `2^-100` to `2^-178` and read 0 (train 1423's review)."""
    query = jnp.zeros((1, 1, 1, 4), jnp.float32).at[0, 0, 0, 0].set(1.0)
    key = jnp.zeros((1, 2, 1, 4), jnp.float32)
    second = jnp.zeros((1, 1, 1, 1), jnp.float32)
    second_key = jnp.zeros((1, 2, 1, 1), jnp.float32)
    value = jnp.asarray([[0.0, 0.0], [0.0, 2.0 ** 100]], jnp.float32)[None, :, None]
    spans = template.spans(1, 2, 1)
    _, pull = jax.vjp(lambda key: attention_parallax.attend(query, second, key, second_key, value, spans),
                      key)
    (dkey,) = pull(jnp.asarray([2.0 ** 127, 2.0 ** -100], jnp.float32)[None, None, None])
    np.testing.assert_array_equal(np.asarray(dkey)[0, :, 0, 0], [-0.125, 0.125])


def test_a_tails_second_cotangent_lands_with_its_key():
    """Logits `[0, -128]`, a second query of 0 against second keys `[0, 2^127]`, values `[0, 1]`, a unit
    cotangent: every cotangent on par with its dense reference's (`attention_parallax.reference`) against
    float64 (check 1, Main's ruling on a tail's gradients), the second query's
    `-4 · 2^127 e^-128 / (1 + e^-128)` nonzero, each term landed at its row's power with its key before
    their product (`attention_parallax.derivative`), where `w δ` landed first flushed (train 1423's
    review)."""
    *operands, spans = _witness((0.0, -128.0), (0.0, 2.0 ** 127), (0.0, 1.0), second=0.0)
    _on_par_with_plain_fp32(operands, spans, jnp.ones((1, 1, 1, 4), jnp.float32),
                            "parallax's tail's second cotangent",
                            reference=attention_parallax.reference)


def test_a_querys_cotangent_takes_its_scale_before_its_power():
    """A query of 0 against keys `±2^127` in one channel at scale `1/2`, second products 0, values `[1, -1]`
    under a cotangent of 2: the logits' cotangents `[1, -1]`, the query's `(2^127 + 2^127) / 2 = 2^127`,
    the scale taken while the terms are landed (`attention_parallax.derivative`), where their sum
    overflowed before it (train 1423's review)."""
    query = jnp.zeros((1, 1, 1, 4), jnp.float32)
    key = jnp.zeros((1, 2, 1, 4), jnp.float32).at[0, :, 0, 0].set(jnp.asarray([2.0 ** 127, -(2.0 ** 127)]))
    second = jnp.zeros((1, 1, 1, 1), jnp.float32)
    second_key = jnp.zeros((1, 2, 1, 1), jnp.float32)
    value = jnp.asarray([1.0, -1.0], jnp.float32)[None, :, None, None]
    spans = template.spans(1, 2, 1)
    out, pull = jax.vjp(lambda query: attention_parallax.attend(query, second, key, second_key, value, spans),
                        query)
    (dquery,) = pull(jnp.full_like(out, 2.0))
    np.testing.assert_array_equal(np.asarray(dquery)[0, 0, 0], [2.0 ** 127, 0.0, 0.0, 0.0])


def test_a_logits_cancelling_terms_cancel_before_they_land():
    """Equal logits, second products `[-2^127, 2^127]`, values `[0, 1]`, a cotangent of 16: `g · C` and
    `δ τ` cancel exactly, the logits' cotangents `[-4, 4]`, the query's `[-2, 2]` against unit keys at scale
    `1/2`, the two subtracted at their common power before `δ` meets the difference
    (`attention_parallax.derivative`), where both landed as infinity first and read NaN (train 1423's
    review)."""
    *operands, spans = _equal_keys((-(2.0 ** 127), 2.0 ** 127), (0.0, 1.0))
    out, pull = jax.vjp(lambda query: attention_parallax.attend(query, *operands[1:], spans), operands[0])
    (dquery,) = pull(jnp.full_like(out, 16.0))
    np.testing.assert_array_equal(np.asarray(dquery)[0, 0, 0, :2], [-2.0, 2.0])


def _batches(stream: int, queries: int, values: int):
    """A draw of 8 places: a query, second query, keys and second keys of `queries` batch rows, values and the
    output's cotangent of `values` rows."""
    def draw(seed):
        return normal_draw([54, stream, seed], (queries, 8, 2, 4), (queries, 8, 2, 2), (queries, 8, 1, 4),
                           (queries, 8, 1, 2), (values, 8, 1, 4), (values, 8, 2, 4))

    return draw


@pytest.mark.mesh(devices=2)
def test_parallax_runs_batch_sharded_on_an_explicit_mesh_and_under_a_checked_map():
    """Parallax and its derivative over two batch rows split across two devices: on an Explicit mesh, its
    operands placed by batch, and under a checked `shard_map` over that batch, each on par with the
    unsplit call against float64 (check 1), its scans' carries seeded from its operands (their placement
    and manual variance, `attention_parallax._tiled`), where replicated seeds failed the scan (train 1423's
    review)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    spans = template.spans(8, 8, 2, causal=True)
    whole = jax.jit(_every_cotangent(attention_parallax.attend, spans))
    local = _every_cotangent(attention_parallax.attend, template.spans(8, 8, 1, causal=True))
    mapped = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=P("data"), out_specs=P("data")))

    draw = _batches(23, 2, 2)

    def placed(run):
        def on(*xs):
            return run(*(jax.device_put(x, NamedSharding(mesh, P("data"))) for x in xs))

        return on

    for label, run in (("explicit", whole), ("mapped", mapped)):
        assert_as_exact(placed(run), whole, draw, f"parallax batch-split {label}", seeds=FEWEST_SEEDS,
                        truth_fn=jax.jit(_every_cotangent(_plain, spans)))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_covariance_past_fp32_beside_its_mean_reads_their_difference(arm):
    """Equal logits, second products `[8, 0]`, values `[2^127, 0]`: the mean `2^126` and the covariance
    `2^128`, the output their difference `-3 · 2^126` exactly, the two subtracted at their common power
    (`attention_parallax._output`), where the covariance landed as infinity first (train 1423's review of
    72b1ef355)."""
    *operands, spans = _witness((0.0, 0.0), (8.0, 0.0), (2.0 ** 127, 0.0))
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    out = attend(*operands, spans)
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0], -3.0 * 2.0 ** 126)


def test_a_large_explicit_scale_meets_a_landed_sum_with_its_power():
    """A query of 0 against unit keys at an explicit scale of `2^100`, second products 0, values `[0, 1]`
    under a unit cotangent: the logits' cotangents `[-1/4, 1/4]`, the query's `[-2^98, 2^98]`, the scale's
    power taken with the landing's (`attention_parallax._scaled`), where the landed sum times `2^100`
    overflowed first (train 1423's review of 72b1ef355)."""
    *operands, spans = _equal_keys((0.0, 0.0), (0.0, 1.0))

    def attend(query):
        return attention_parallax.attend(query, *operands[1:], spans, scale=2.0 ** 100)

    out, pull = jax.vjp(attend, operands[0])
    (dquery,) = pull(jnp.ones_like(out))
    np.testing.assert_array_equal(np.asarray(dquery)[0, 0, 0, :2], [-(2.0 ** 98), 2.0 ** 98])


def _tiled_keys(entries: dict[int, tuple[float, float, float]], count: int):
    """One query of 0 over `count` keys of logit 0 (`_logits_of`), the keys at `entries`' places alone
    visible, each `(logit, second key, value)` against a second query of 1, its value one channel."""
    query, key = _logits_of([entries.get(place, (0.0,) * 3)[0] for place in range(count)])
    seconds = [entries.get(place, (0.0,) * 3)[1] for place in range(count)]
    values = [entries.get(place, (0.0,) * 3)[2] for place in range(count)]
    valid = jnp.zeros((1, count), bool).at[0, jnp.asarray(sorted(entries))].set(True)
    return (query, jnp.ones((1, 1, 1, 1), jnp.float32), key,
            jnp.asarray(seconds, jnp.float32)[None, :, None, None],
            jnp.asarray(values, jnp.float32)[None, :, None, None], template.spans(1, count, 1, valid=valid))


def test_a_far_tiles_centre_is_its_own_share_of_its_second_product():
    """Keys 0 and 128 (two tiles), logits `[0, -20]`, second products `[0, 2^100]`, values 0, a unit
    cotangent: every cotangent on par with plain dense fp32 attention's against float64 (check 1), the
    centre `e^-20 2^100 / (1 + e^-20)` merged about the larger share's (`attention_parallax._merged_centre`),
    where `1 - s_A` rounded to 0 read it 0 and the dominant key's values' cotangent 1, not `2.6e21` (train
    1444's review)."""
    *operands, spans = _tiled_keys({0: (0.0, 0.0, 0.0), 128: (-20.0, 2.0 ** 100, 0.0)}, 129)
    _on_par_with_plain_fp32(operands, spans, jnp.ones((1, 1, 1, 1), jnp.float32),
                            "parallax's far tile's centre")


def test_covariances_that_cancel_across_tiles_bound_no_later_merge():
    """Equal logits over three tiles: second products `±2^200` (a second query of `2^100`) over values
    `±2^100` whose two tiles' covariances `±2^300` cancel exactly, and a third tile of second products 0 and
    values 3: the output their mean 1, the cancelled covariance's power cleared
    (`attention_parallax.Moments.spread`), where it held `2^300` and took the mean to 0 (train 1444's
    review)."""
    big, value = 2.0 ** 100, 2.0 ** 100
    entries = {0: (0.0, big, value), 1: (0.0, -big, -value), 128: (0.0, big, -value),
               129: (0.0, -big, value), 256: (0.0, 0.0, 3.0), 257: (0.0, 0.0, 3.0)}
    query, _, key, second_key, values, spans = _tiled_keys(entries, 258)
    second = jnp.full((1, 1, 1, 1), 2.0 ** 100, jnp.float32)
    out = attention_parallax.attend(query, second, key, second_key, values, spans)
    np.testing.assert_array_equal(np.asarray(out), np.ones((1, 1, 1, 1), np.float32))


def test_a_rows_tiny_entry_bounds_its_own_pair():
    """A second query `[2^127, 2^-125]` against second keys `(0, 0)` and `(0, 2^127)`, equal logits, values
    `[0, 1]`: second products `[0, 4]`, the output `1/2 - 1 = -1/2`, each pair bounded by its own entries'
    exponents (`attention_parallax._pair_bound`), where mantissas at the row's power flushed `2^-125`,
    bounded the row by `2^131` and took its query below fp32's normals (train 1444's review)."""
    query = jnp.zeros((1, 1, 1, 4), jnp.float32)
    key = jnp.zeros((1, 2, 1, 4), jnp.float32)
    second = jnp.asarray([2.0 ** 127, 2.0 ** -125], jnp.float32)[None, None, None]
    second_key = jnp.asarray([[0.0, 0.0], [0.0, 2.0 ** 127]], jnp.float32)[None, :, None]
    value = jnp.asarray([0.0, 1.0], jnp.float32)[None, :, None, None]
    out = attention_parallax.attend(query, second, key, second_key, value, template.spans(1, 2, 1))
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0, 0], -0.5)


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_large_value_direction_meets_lifted_terms_at_their_size(arm):
    """Equal logits, second products `[1, -1]`, values 0 along the direction `[2^126, -2^126]`: the
    output's tangent `-2^126`, plain fp32's, each lifted term met at its size before the direction
    (`attention_parallax._shifted`), where terms lifted by `2^60` met it first and overflowed (train
    1444's review)."""
    query, second, key, second_key, value, spans = _witness((0.0, 0.0), (1.0, -1.0), (0.0, 0.0))
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    direction = jnp.broadcast_to(jnp.asarray([2.0 ** 126, -(2.0 ** 126)], jnp.float32)[None, :, None, None],
                                 value.shape)
    _, tangent = jax.jvp(lambda value: attend(query, second, key, second_key, value, spans), (value,),
                         (direction,))
    np.testing.assert_array_equal(np.asarray(tangent), np.full((1, 1, 1, 4), -(2.0 ** 126), np.float32))


def test_a_decay_past_int32_erases_no_dominant_term():
    """Logits `[0, -1488522240]` over values `[1, 2^-100]`, second products 0, a unit cotangent: the far
    weight 0, so the dominant logit's cotangent `g · (v - a)` is 0 and every key's too, its decay's power
    held before it is an integer (`attention_parallax._reach`), where it wrapped past `INT32_MIN`, erased
    `g · a` and read the dominant key's `1/2` (train 1444's review)."""
    query, second, key, second_key, value, spans = _witness((0.0, -1488522240.0), (0.0, 0.0),
                                                             (1.0, 2.0 ** -100))
    out, pull = jax.vjp(lambda key: attention_parallax.attend(query, second, key, second_key, value, spans),
                        key)
    (dkey,) = pull(jnp.ones_like(out))
    np.testing.assert_array_equal(np.asarray(dkey), np.zeros(key.shape, np.float32))


@pytest.mark.mesh(devices=2)
def test_parallax_runs_under_a_checked_map_with_its_values_split_alone():
    """Parallax and its derivative under a checked `shard_map` whose values and cotangent alone are split by
    batch, its queries and keys replicated: on par with each batch's unsplit call against float64 (check
    1), its scans' carries typed as their steps make them (`attention_parallax._tiled`), where carries
    seeded from the query alone failed the scan (train 1444's review)."""
    from jax.sharding import AxisType, PartitionSpec as P

    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Auto,), devices=jax.devices()[:2])
    spans = template.spans(8, 8, 1, causal=True)

    def per_batch(attend):
        def run(query, second, key, second_key, value, cotangent):
            def call(*operands):
                return jnp.concatenate([attend(*operands[:4], operands[4][row:row + 1], spans)
                                        for row in range(2)])

            out, pull = jax.vjp(call, query, second, key, second_key, value)
            return out, *pull(cotangent)

        return run

    def mapped(query, second, key, second_key, value, cotangent):
        local = jax.shard_map(lambda *operands: attention_parallax.attend(*operands, spans), mesh=mesh,
                              in_specs=(P(), P(), P(), P(), P("data")), out_specs=P("data"))
        out, pull = jax.vjp(local, query, second, key, second_key, value)
        return out, *pull(cotangent)

    assert_as_exact(jax.jit(mapped), jax.jit(per_batch(attention_parallax.attend)), _batches(29, 1, 2),
                    "parallax with its values split alone", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(per_batch(_plain)))


def test_a_call_of_no_queries_pulls_back_zeros():
    """No query over one key (`template.spans(0, 1, 1)`): every operand's cotangent zeros of its shape, a
    key's landing over no query `_NONE` (`attention_parallax._landing`), where its maximum over no query
    failed (train 1444's review)."""
    operands = (jnp.zeros((1, 0, 1, 4), jnp.float32), jnp.zeros((1, 0, 1, 1), jnp.float32),
                jnp.ones((1, 1, 1, 4), jnp.float32), jnp.ones((1, 1, 1, 1), jnp.float32),
                jnp.ones((1, 1, 1, 4), jnp.float32))
    spans = template.spans(0, 1, 1)
    out, pull = jax.vjp(lambda *xs: attention_parallax.attend(*xs, spans), *operands)
    for grad, operand in zip(pull(jnp.zeros_like(out)), operands, strict=True):
        np.testing.assert_array_equal(np.asarray(grad), np.zeros(operand.shape, np.float32))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_scale_of_zero_gives_logits_of_zero(arm):
    """A scale of 0 over a query and keys of `2^100` (their product past fp32), values 1: the logits 0 and
    the output 1, never the product times 0 (`inf · 0`, NaN; train 1444's review)."""
    query = jnp.zeros((1, 1, 1, 4), jnp.float32).at[0, 0, 0, 0].set(2.0 ** 100)
    key = jnp.zeros((1, 2, 1, 4), jnp.float32).at[0, :, 0, 0].set(2.0 ** 100)
    second, second_key = jnp.zeros((1, 1, 1, 1), jnp.float32), jnp.zeros((1, 2, 1, 1), jnp.float32)
    value = jnp.ones((1, 2, 1, 4), jnp.float32)
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    out = attend(query, second, key, second_key, value, template.spans(1, 2, 1), scale=0.0)
    np.testing.assert_array_equal(np.asarray(out), np.ones((1, 1, 1, 4), np.float32))


def test_a_scale_of_zero_moves_no_query_or_key():
    """A scale of 0 over a query and keys `±2^127`, values `±2^127` under a cotangent of `2^127`: the query's
    and keys' cotangents 0, the logits' coefficients never formed (`attention_parallax.derivative`), where
    unscaled ones of `2^253` landed, overflowed and met the scale's 0 as NaN (train 1444's review)."""
    query = jnp.zeros((1, 1, 1, 4), jnp.float32).at[0, 0, 0, 0].set(2.0 ** 127)
    key = jnp.zeros((1, 2, 1, 4), jnp.float32).at[0, :, 0, 0].set(jnp.asarray([2.0 ** 127, -(2.0 ** 127)]))
    value = jnp.broadcast_to(jnp.asarray([2.0 ** 127, -(2.0 ** 127)], jnp.float32)[None, :, None, None],
                             (1, 2, 1, 4))
    second, second_key = jnp.zeros((1, 1, 1, 1), jnp.float32), jnp.zeros((1, 2, 1, 1), jnp.float32)
    spans = template.spans(1, 2, 1)
    out, pull = jax.vjp(lambda query, key: attention_parallax.attend(query, second, key, second_key, value,
                                                                     spans, scale=0.0), query, key)
    dquery, dkey = pull(jnp.full_like(out, 2.0 ** 127))
    np.testing.assert_array_equal(np.asarray(dquery), np.zeros(query.shape, np.float32))
    np.testing.assert_array_equal(np.asarray(dkey), np.zeros(key.shape, np.float32))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_each_value_channel_keeps_its_own_covariance(arm):
    """Equal logits, second products `[1, -1]`, values `(2^127, 2^-100)` and `(-2^127, -2^-100)`: the means
    0 and the output `[-2^127, -2^-100]`, each channel's covariance at its own power
    (`attention_parallax.Moments.spread_power`), where the row's one power took the small channel to
    `2^-228` (train 1444's review)."""
    query = jnp.zeros((1, 1, 1, 4), jnp.float32)
    key = jnp.zeros((1, 2, 1, 4), jnp.float32)
    second = jnp.ones((1, 1, 1, 1), jnp.float32)
    second_key = jnp.asarray([1.0, -1.0], jnp.float32)[None, :, None, None]
    value = jnp.asarray([[2.0 ** 127, 2.0 ** -100], [-(2.0 ** 127), -(2.0 ** -100)]],
                        jnp.float32)[None, :, None]
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    out = attend(query, second, key, second_key, value, template.spans(1, 2, 1))
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0], np.asarray([-(2.0 ** 127), -(2.0 ** -100)],
                                                                        np.float32))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_small_value_direction_meets_lifted_terms_at_their_size(arm):
    """Eight equal logits, second products `[2^100, -2^100, 2^70, -2^70, 0, 0, 0, 0]`, values of `2^127` on
    the last four keys, along the direction `[0, 0, 2^-124, -2^-124, 0, 0, 0, 0]`: the output's tangent
    plain fp32's `-2^-56`, the terms taken back toward their size before the direction meets them
    (`attention_parallax._shifted`), where a positive power put back after their products took `2^-165`
    below fp32's least (train 1444's review)."""
    seconds = (2.0 ** 100, -(2.0 ** 100), 2.0 ** 70, -(2.0 ** 70), 0.0, 0.0, 0.0, 0.0)
    query, second, key, second_key, value, spans = _witness((0.0,) * 8, seconds,
                                                             (0.0,) * 4 + (2.0 ** 127,) * 4)
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    direction = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 2.0 ** -124, -(2.0 ** -124), 0.0, 0.0, 0.0, 0.0],
                                             jnp.float32)[None, :, None, None], value.shape)
    _, tangent = jax.jvp(lambda value: attend(query, second, key, second_key, value, spans), (value,),
                         (direction,))
    np.testing.assert_array_equal(np.asarray(tangent), np.full((1, 1, 1, 4), -(2.0 ** -56), np.float32))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_channels_power_is_its_own_before_its_products(arm):
    """Eight equal logits, second products `[2^100, -2^100, 2^70, -2^70, 0, 0, 0, 0]`, a channel of `2^127`
    on the last four keys beside one of `[0, 0, 2^-124, -2^-124, 0, 0, 0, 0]`: the output `[2^126, -2^-56]`,
    each channel taken to its own power before its products (`attention_parallax._channel_powers`), where the
    first channel's power took the second's products to `2^-165` (train 1444's review)."""
    seconds = (2.0 ** 100, -(2.0 ** 100), 2.0 ** 70, -(2.0 ** 70), 0.0, 0.0, 0.0, 0.0)
    query, second, key, second_key, _, spans = _witness((0.0,) * 8, seconds, (0.0,) * 8, width=2)
    value = jnp.stack([jnp.asarray((0.0,) * 4 + (2.0 ** 127,) * 4, jnp.float32),
                       jnp.asarray((0.0, 0.0, 2.0 ** -124, -(2.0 ** -124), 0.0, 0.0, 0.0, 0.0), jnp.float32)],
                      axis=-1)[None, :, None]
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    out = attend(query, second, key, second_key, value, spans)
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0],
                                  np.asarray([2.0 ** 126, -(2.0 ** -56)], np.float32))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_values_tangent_meets_its_mean_before_it_lands(arm):
    """Equal logits, second products `[8, 0]`, values 0 along `[2^127, 0]`: the output's tangent plain fp32's
    `-3 · 2^126` (the first value's coefficient `w (1 - τ)` is `-1.5`), the values' direction met by the one
    contraction `Σ w (1 - τ) dv` (`attention_parallax._dense`), where the covariance's tangent `2^128`
    overflowed before the mean's `2^126` met it (train 1444's review)."""
    query, second, key, second_key, value, spans = _witness((0.0, 0.0), (8.0, 0.0), (0.0, 0.0))
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    direction = jnp.broadcast_to(jnp.asarray([2.0 ** 127, 0.0], jnp.float32)[None, :, None, None],
                                 value.shape)
    _, tangent = jax.jvp(lambda value: attend(query, second, key, second_key, value, spans), (value,),
                         (direction,))
    np.testing.assert_array_equal(np.asarray(tangent), np.full((1, 1, 1, 4), -3.0 * 2.0 ** 126, np.float32))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_hidden_keys_value_bounds_no_channel(arm):
    """One query seeing keys 0 and 1 of three, equal logits, second products `[1, -1, 0]`, values
    `[2^-120, -2^-120, 2^127]`: the output `-2^-120`, the channel's power over the keys some query sees
    (`attention_parallax._channel_powers`), where the hidden key's `2^127` took the seen values to
    `2^-248` (train 1444's review)."""
    query, second, key, second_key, value, _ = _witness((0.0, 0.0, 0.0), (1.0, -1.0, 0.0),
                                                        (2.0 ** -120, -(2.0 ** -120), 2.0 ** 127))
    spans = template.spans(1, 3, 1, valid=jnp.asarray([[True, True, False]]))
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    out = attend(query, second, key, second_key, value, spans)
    np.testing.assert_array_equal(np.asarray(out), np.full((1, 1, 1, 4), -(2.0 ** -120), np.float32))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_second_query_direction_meets_lifted_terms_at_their_size(arm):
    """Equal logits, a second query of 1 over second keys `[1, -1]`, values `[2^127, -2^127]`, along the
    second query's direction 1: the output's tangent plain fp32's `-2^127`, the terms' direction taken toward
    their size before it meets the values (`attention_parallax._shifted`), where a `2^60` direction met
    `2^126` and overflowed (train 1444's review)."""
    query, second, key, second_key, value, spans = _witness((0.0, 0.0), (1.0, -1.0),
                                                            (2.0 ** 127, -(2.0 ** 127)))
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    _, tangent = jax.jvp(lambda second: attend(query, second, key, second_key, value, spans), (second,),
                         (jnp.ones_like(second),))
    np.testing.assert_array_equal(np.asarray(tangent), np.full((1, 1, 1, 4), -(2.0 ** 127), np.float32))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_negligible_keys_value_bounds_no_channel(arm):
    """One query seeing three keys, logits `[0, 0, -1000]`, second products `[1, -1, 0]`, values
    `[2^-120, -2^-120, 2^127]`: the output `-2^-120`, plain fp32's, the third key's covariance term
    (`e^-1000 · 2^127`) past fp32's span below the others' bounding no channel (`attention_parallax.
    _channel_powers`), where its raw value took the live ones to `±2^-248` (train 1444's review)."""
    query, second, key, second_key, value, spans = _witness((0.0, 0.0, -1000.0), (1.0, -1.0, 0.0),
                                                            (2.0 ** -120, -(2.0 ** -120), 2.0 ** 127))
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    out = attend(query, second, key, second_key, value, spans)
    np.testing.assert_array_equal(np.asarray(out), np.full((1, 1, 1, 4), -(2.0 ** -120), np.float32))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_key_at_the_centre_still_moves_its_covariance(arm):
    """Four equal logits, second products `[1, -1, 0, 0]`, values `[1, -1, 4, 0]`, along the third second
    product alone: the tangent plain fp32's `-3/4` (`w (a - v) = (1 - 4) / 4`), the key whose term is 0 at
    the centre still the covariance's (`attention_parallax.moments`), where the channel's significance took
    its value to 0 and the tangent read 0 (train 1444's review)."""
    query, second, key, second_key, value, spans = _witness((0.0,) * 4, (1.0, -1.0, 0.0, 0.0),
                                                            (1.0, -1.0, 4.0, 0.0))
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    direction = jnp.zeros_like(second_key).at[0, 2, 0, 0].set(1.0)
    _, tangent = jax.jvp(lambda second_key: attend(query, second, key, second_key, value, spans),
                         (second_key,), (direction,))
    np.testing.assert_array_equal(np.asarray(tangent), np.full((1, 1, 1, 4), -0.75, np.float32))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_second_product_direction_meets_the_values_as_they_are(arm):
    """Four equal logits, second products `[1, -1, 0, 0]`, values `[2^-8, -2^-8, 2^124, -2^124]`, along
    second products `[0, 0, 1, -1]`: the tangent plain fp32's `-2^123` (`Σ w (a - v) dt`), the values as
    they are (`attention_parallax._dense_tangent`), where the moments' values at their channel's power
    (`2^131`, held at fp32's largest) moved it below `2^120` (train 1444's review)."""
    values = (2.0 ** -8, -(2.0 ** -8), 2.0 ** 124, -(2.0 ** 124))
    query, second, key, second_key, value, spans = _witness((0.0,) * 4, (1.0, -1.0, 0.0, 0.0), values)
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    direction = jnp.zeros_like(second_key).at[0, 2, 0, 0].set(1.0).at[0, 3, 0, 0].set(-1.0)
    _, tangent = jax.jvp(lambda second_key: attend(query, second, key, second_key, value, spans),
                         (second_key,), (direction,))
    np.testing.assert_array_equal(np.asarray(tangent), np.full((1, 1, 1, 4), -(2.0 ** 123), np.float32))


@pytest.mark.parametrize("arm", ["walk", "reference"])
def test_a_second_product_direction_is_centred_before_the_values(arm):
    """Equal logits, second products 0, values `[2^127, 2^126]` along second products `[4, 0]`: the
    tangent plain fp32's `-2^126`, the direction centred (`[-1, 1]` of its weights) before it meets the
    values (`attention_parallax._dense_tangent`), where its weighted sum `2^128` overflowed before the
    mean's `3 · 2^126` met it (train 1444's review)."""
    query, second, key, second_key, value, spans = _witness((0.0, 0.0), (0.0, 0.0), (2.0 ** 127, 2.0 ** 126))
    attend = attention_parallax.attend if arm == "walk" else attention_parallax.reference
    direction = jnp.zeros_like(second_key).at[0, 0, 0, 0].set(4.0)
    _, tangent = jax.jvp(lambda second_key: attend(query, second, key, second_key, value, spans),
                         (second_key,), (direction,))
    np.testing.assert_array_equal(np.asarray(tangent), np.full((1, 1, 1, 4), -(2.0 ** 126), np.float32))
