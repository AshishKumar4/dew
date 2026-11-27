"""A step's score products (`attention_products`, kernel-templates.md §A rows 16a and 25): summed, and
selected by bands of distance, each on par with a reference written over its own products' logits (check 1),
its output and every product's gradients, in the fast and the accurate form, on the template's route and on
the Mosaic mainloop interpreted as an H100's and an A100's."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import traced_attention_terms
from generation_support import INTERPRETS
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_mosaic, attention_products
from dew.nn.kernels.templates.hooks import Hook, Side
from dew.nn.precision import float32_einsum

pytestmark = pytest.mark.kernel

SEEDS = 8
TOKENS, HEADS, KV_HEADS, WIDTH = 160, 2, 1, 64
WINDOW = 37
"""A near band no tile's edge: 64- and 32-key tiles each hold keys of both products along it."""
SCALE = WIDTH ** -0.5
FORMS = {"fast": (False, False), "accurate": (True, True)}
"""Each form's `(natural, accurate)`."""


def _draw(products: int, dtype=jnp.bfloat16, *, queries: int = TOKENS, keys: int = TOKENS):
    """A draw of `products` queries and keys, a value and the output's cotangent, unit Gaussians by seed."""
    count = keys

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 2 * products + 2)
        laid = [jax.random.normal(stream, (1, queries, HEADS, WIDTH), jnp.float32).astype(dtype)
                for stream in streams[:products]]
        keys = [jax.random.normal(stream, (1, count, KV_HEADS, WIDTH), jnp.float32).astype(dtype)
                for stream in streams[products:2 * products]]
        value = jax.random.normal(streams[-2], (1, count, KV_HEADS, WIDTH), jnp.float32).astype(dtype)
        dout = jax.random.normal(streams[-1], (1, queries, HEADS, WIDTH), jnp.float32).astype(dtype)
        return (*laid, *keys, value, dout)

    return draw


def _softmax_reference(logits: jax.Array, value: jax.Array, dtype) -> jax.Array:
    """The template reference's softmax over `logits` `[B, H, Q, K]` (`template.reference`): fp32 (fp64 for
    fp64 inputs), the weights rounded to the value's dtype for their product, normalized, rounded to `dtype`;
    a row that sees no key 0."""
    batch, heads, queries, keys = logits.shape
    kv_heads = value.shape[2]
    m = jnp.max(logits, axis=-1, initial=-jnp.inf)
    safe = jnp.where(jnp.isfinite(m), m, 0)
    p = jnp.exp(logits - safe[..., None])
    total = jnp.sum(p, axis=-1)
    grouped = p.reshape(batch, kv_heads, heads // kv_heads, queries, keys).astype(value.dtype)
    acc = float32_einsum("bhgqk,bkhd->bhgqd", grouped, value, precision=template.REFERENCE_PRECISION,
                         preferred_element_type=logits.dtype).reshape(batch, heads, queries, value.shape[-1])
    out = jnp.where(total[..., None] > 0, acc / jnp.where(total > 0, total, 1)[..., None], 0)
    return jnp.swapaxes(out.astype(dtype), 1, 2)


def _banded_reference(spans: template.KeySpans, bands: tuple[int, ...]):
    """`(*queries, *keys, value)`'s attention whose logits are product `p`'s within band `p` of each query's
    index (`attention_products.Banded`), written over each product's logits apart."""
    def reference(*arguments):
        products = len(bands) + 1
        queries, keys, value = arguments[:products], arguments[products:2 * products], arguments[-1]
        logits = [template.scores(query, key, spans, scale=SCALE)
                  for query, key in zip(queries, keys, strict=True)]
        distance = jnp.abs(jnp.arange(queries[0].shape[1])[:, None] - jnp.arange(spans.keys)[None, :])
        chosen = jnp.searchsorted(jnp.asarray(bands), distance, side="left")
        selected = logits[-1]
        for p in reversed(range(products - 1)):
            selected = jnp.where(chosen == p, logits[p], selected)
        return _softmax_reference(selected, value, queries[0].dtype)

    return reference


def _routed(combine, spans: template.KeySpans, form: str, products: int):
    """`attention_products.attend` on the template's route in `form` over `(*queries, *keys, value)`."""
    natural, accurate = FORMS[form]

    def attended(*arguments):
        queries, keys, value = arguments[:products], arguments[products:2 * products], arguments[-1]
        return attention_products.attend(queries, keys, value, spans, combine, scale=SCALE, natural=natural,
                                         accurate=accurate)

    return attended


@pytest.mark.parametrize("form", FORMS)
def test_summed_products_are_on_par_with_their_logits_summed(form):
    """Two products summed (absorbed MLA's content and rope, kernel-templates.md §A row 25), causal, on the
    template's route: on par with a softmax over each product's logits summed, its output and every
    product's gradients (check 1)."""
    spans = template.spans(TOKENS, TOKENS, 1, causal=True)

    def reference(q1, q2, k1, k2, value):
        logits = (template.scores(q1, k1, spans, scale=SCALE) + template.scores(q2, k2, spans, scale=SCALE))
        return _softmax_reference(logits, value, q1.dtype)

    ours = _routed(attention_products.SUM, spans, form, 2)
    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(reference)), _draw(2),
                    f"summed products {form}", seeds=SEEDS)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize(("case", "bands", "queries"),
                         [("causal", (WINDOW,), TOKENS), ("full", (WINDOW,), TOKENS),
                          ("causal", (11, 53), TOKENS), ("full", (5,), TOKENS + 48)],
                         ids=["causal-two", "full-two", "causal-three", "full-more-queries"])
def test_banded_products_are_on_par_with_their_logits_selected(case, bands, queries, form):
    """Products selected by bands of distance (piecewise attention, §A row 16a), causal and full, two products
    and three, and over more queries than keys (queries past the last key, whose far band holds every key
    past its near one): on the template's route, on par with a softmax over each key's own product's logits,
    its output and every product's gradients (check 1), the bands' edges inside key tiles."""
    spans = template.spans(queries, TOKENS, 1, causal=case == "causal")
    products = len(bands) + 1
    ours = _routed(attention_products.Banded(bands), spans, form, products)
    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(_banded_reference(spans, bands))),
                    _draw(products, queries=queries), f"banded products {case} {bands} {queries} {form}",
                    seeds=SEEDS)


def _banded_terms(spans: template.KeySpans, combine: attention_products.Banded):
    """The float64 terms of a two-product banded call's output and gradients (`attention_oracle`'s
    `attention_terms` of the call laid out by hand from the bands' own predicate, at the products' scale):
    a row that cancels, a query of one key's dQ among them, held to what it sums, as the template's own
    backward checks hold it; each copy's key and value terms its product's, the value's copies' summed in
    quadrature."""
    def terms(q1, q2, k1, k2, value, dout):
        # The call laid out by hand, each copy's keys those its band's predicate holds, and its queries taken
        # so `attention_terms`' scale at the laid width is the products' (`SCALE`), the query's terms back.
        count = spans.keys
        apart = jnp.abs(jnp.arange(q1.shape[1])[:, None] - jnp.arange(count)[None, :])
        near = jnp.searchsorted(jnp.asarray(combine.bands), apart, side="left") == 0
        seen = template.dense(spans)
        keep = jnp.concatenate([seen & near, seen & ~near], axis=-1)[:, None]
        lift = (2 * WIDTH) ** 0.5 * SCALE
        query = jnp.concatenate([q1, q2], axis=-1) * lift
        key = jnp.concatenate([jnp.pad(k1, ((0, 0), (0, 0), (0, 0), (0, WIDTH))),
                               jnp.pad(k2, ((0, 0), (0, 0), (0, 0), (WIDTH, 0)))], axis=1)
        out, dq, dk, dv = traced_attention_terms(query, key, jnp.concatenate([value] * 2, axis=1), dout, keep)
        dq = dq * lift
        return (out, dq[..., :WIDTH], dq[..., WIDTH:], dk[:, :count, :, :WIDTH], dk[:, count:, :, WIDTH:],
                jnp.sqrt(dv[:, :count] ** 2 + dv[:, count:] ** 2))

    return terms


@INTERPRETS
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("device", ["sm90", "sm80"])
def test_banded_products_are_on_par_on_the_mosaic_mainloop_interpreted(device, form):
    """Two products selected at a near band (`WINDOW`) of a causal call on the Mosaic mainloop, its training
    forward and its backward from its residuals interpreted as `device` runs them, the copies' gradients
    summed back through the selection's layout: on par with the softmax over each key's own product's
    logits (check 1), its walks stepping each copy's tiles its band reaches, both at a tile the band's edge
    cuts."""
    spans = template.spans(TOKENS, TOKENS, 1, causal=True)
    combine = attention_products.Banded((WINDOW,))
    natural, accurate = FORMS[form]
    placeholder = jnp.zeros((1, TOKENS, KV_HEADS, WIDTH), jnp.bfloat16)
    _, _, _, chosen, hooks = attention_products.selected(
        (placeholder,) * 2, (placeholder,) * 2, placeholder, spans, combine)

    def layout(q1, q2, k1, k2, value):
        # The keys' and the value's copies wide, their gradients summed unrounded (`attend`'s).
        wide = [k.astype(jnp.float32) for k in (k1, k2)]
        return attention_products.selected((q1, q2), wide, value.astype(jnp.float32), spans, combine)[:3]

    def ours(q1, q2, k1, k2, value, dout):
        (query, wide_key, wide_values), pull = jax.vjp(layout, q1, q2, k1, k2, value)
        key, values = wide_key.astype(value.dtype), wide_values.astype(value.dtype)
        with generation.assumed(device, interpreted=True):
            # The residuals the route keeps for the form (`attention_route._forward`).
            output = template.takes_output_residuals(hooks, query.dtype, accurate=accurate)
            kept = template.Kept.OUTPUT if output else template.Kept.SPLIT
            out, *residuals = attention_mosaic.forward(query, key, values, chosen, hooks, scale=SCALE,
                                                       natural=natural, residuals=kept, interpret=True)
            grads = attention_mosaic.backward(query, key, values, chosen, hooks, dout, residuals=residuals,
                                              scale=SCALE, natural=natural, accurate=accurate, interpret=True,
                                              rounded=False)
        return out, *pull((grads.query.astype(query.dtype), grads.key, grads.value))

    assert_as_exact(jax.jit(ours), jax.jit(with_gradients(_banded_reference(spans, combine.bands))), _draw(2),
                    f"banded products on the Mosaic mainloop {device} {form}", seeds=SEEDS,
                    terms=_banded_terms(spans, combine))


@pytest.mark.parametrize("bands", [(), (8, 8), (16, 4), (-1,)])
def test_a_banded_combine_refuses_bands_whose_products_would_share_keys(bands):
    """A banded combine's bands are distances that increase, so its products' keys are each one product's:
    bands that repeat, fall or are negative (or none) are refused where they are stated."""
    with pytest.raises(ValueError):
        attention_products.Banded(bands)


def test_a_selected_call_refuses_a_score_hook_reading_its_keys_indices():
    """A score hook reading its keys' indices (ALiBi's distance) is refused on a selected call, whose copies'
    keys take indices of their own; one reading the scores alone (a softcap) is taken."""
    spans = template.spans(TOKENS, TOKENS, 1, causal=True)
    combine = attention_products.Banded((WINDOW,))
    operand = jnp.zeros((1, TOKENS, KV_HEADS, WIDTH), jnp.bfloat16)

    def hooked(fn):
        return template.Hooks(score=Hook(fn))

    with pytest.raises(ValueError):
        attention_products.selected((operand,) * 2, (operand,) * 2, operand, spans, combine,
                                    hooked(lambda s, q, k: s - (q - k)))
    # One that may read them, in a branch: refused alike.
    with pytest.raises(ValueError):
        attention_products.selected((operand,) * 2, (operand,) * 2, operand, spans, combine,
                                    hooked(lambda s, q, k: jax.lax.cond(s[0, 0] > 0, lambda k: s + k,
                                                                        lambda k: s - k, k)))
    attention_products.selected((operand,) * 2, (operand,) * 2, operand, spans, combine,
                                hooked(lambda s, q, k: 30.0 * jnp.tanh(s / 30.0)))


def test_a_banded_call_selects_by_a_querys_explicit_place_however_far():
    """A query placed at 100 over two keys, a near band of 100: both keys lie within it (distances 100 and
    99), so both are the near product's, none the far's, whatever the call's extent."""
    spans = template.spans(1, 2, 1)
    operand = jnp.zeros((1, 1, KV_HEADS, WIDTH), jnp.bfloat16)
    keys = jnp.zeros((1, 2, KV_HEADS, WIDTH), jnp.bfloat16)
    laid = attention_products.selected((operand,) * 2, (keys,) * 2, keys, spans,
                                       attention_products.Banded((100,)), places=jnp.asarray([[100]]))[3]
    np.testing.assert_array_equal(np.asarray(template.dense(laid))[0, 0], [True, True, False, False])


def _parallax_reference(spans: template.KeySpans):
    """Parallax's output as its paper states it (arXiv 2605.29157, §A row 54), over the softmax's weights
    `p` and the second product's `t` apart: `O1/d1 (1 + d2/d1) - O2/d1`, `O1 = Σ p v`, `d1 = Σ p`,
    `O2 = Σ (p t) v`, `d2 = Σ p t`, each weight rounded to the value's dtype for its product."""
    def reference(query, second, key, second_key, value):
        logits = template.scores(query, key, spans, scale=SCALE)
        wide = logits.dtype
        m = jnp.max(logits, axis=-1, keepdims=True)
        p = jnp.exp(logits - m)
        seconds = jnp.einsum("bqhd,bkhd->bhqk", second.astype(wide), second_key.astype(wide))
        weighed = p * seconds

        def summed(weights):
            return jnp.einsum("bhqk,bkhd->bhqd", weights.astype(value.dtype).astype(wide), value.astype(wide))

        d1, d2 = p.sum(-1, keepdims=True), weighed.sum(-1, keepdims=True)
        out = summed(p) / d1 * (1 + d2 / d1) - summed(weighed) / d1
        return jnp.swapaxes(out.astype(query.dtype), 1, 2)

    return reference


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("case", ["causal", "full"])
def test_parallax_is_on_par_with_its_two_accumulators(case, form):
    """Parallax (§A row 54): a second, unscaled query's product weighing a second accumulator, on the
    template's route (its kernels' forward and backward, `attention_products.parallax`), causal and full:
    on par with the paper's two accumulators and denominators (check 1), its output and the gradients of the
    query, the second query, the keys and the value."""
    spans = template.spans(TOKENS, TOKENS, 1, causal=case == "causal")
    natural, accurate = FORMS[form]

    def ours(query, second, key, value):
        return attention_products.attend((query, second), (key, key), value, spans,
                                         attention_products.PARALLAX, scale=SCALE, natural=natural,
                                         accurate=accurate)

    def reference(query, second, key, value):
        return _parallax_reference(spans)(query, second, key, key, value)

    def draw(seed: int):
        q1, q2, k1, _, value, dout = _draw(2)(seed)
        return q1, q2 * 0.125, k1, value, dout

    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(reference)), draw,
                    f"parallax {case} {form}", seeds=SEEDS)


def test_a_banded_calls_16_bit_key_bias_sums_its_copies_gradients_before_it_rounds():
    """A fp16 key bias (a KEY side) on a banded call whose near and far products' bias gradients each pass
    fp16's largest and cancel: uniform logits, values 3 and -3, output cotangents 65504 and -65504; the
    bias's gradient finite, its copies' summed wide before their one rounding."""
    spans = template.spans(2, 2, 1)
    combine = attention_products.Banded((0,))
    query = jnp.zeros((1, 2, 1, 8), jnp.float16)
    key = jnp.zeros((1, 2, 1, 8), jnp.float16)
    value = jnp.broadcast_to(jnp.asarray([3.0, -3.0], jnp.float16)[None, :, None, None], (1, 2, 1, 8))
    dout = jnp.broadcast_to(jnp.asarray([65504.0, -65504.0], jnp.float16)[None, :, None, None], (1, 2, 1, 8))

    def attended(bias):
        hooks = template.Hooks(score=Hook(lambda s, q, k, b: s + b, (Side(bias, template.Kind.KEY),)))
        return attention_products.attend((query, query), (key, key), value, spans, combine, hooks, scale=1.0)

    _, pull = jax.vjp(attended, jnp.zeros((1, 1, 2), jnp.float16))
    assert np.isfinite(np.asarray(pull(dout)[0], np.float32)).all()


def test_parallax_holds_a_second_query_past_its_dtypes_range_over_the_scale():
    """Parallax in fp16 at a logit scale of 1/8, uniform logits, a second query of 16384 against second keys
    of -2^-14 and 2^-14 (its products -1 and 1, their mean 0) and values -1 and 1: its weights `a (1 + τ -
    t)` 1 and 0, its output -1 exactly, where the second query over the scale, 131072, passes fp16's
    largest."""
    spans = template.spans(2, 2, 1)
    zeros = jnp.zeros((1, 2, 1, 8), jnp.float16)
    second = zeros.at[..., 0].set(16384.0)
    second_key = zeros.at[0, :, 0, 0].set(jnp.asarray([-(2.0**-14), 2.0**-14]))
    value = jnp.broadcast_to(jnp.asarray([-1.0, 1.0], jnp.float16)[None, :, None, None], (1, 2, 1, 8))
    out = attention_products.parallax(zeros, second, zeros, second_key, value, spans, scale=0.125)
    np.testing.assert_array_equal(np.asarray(out, np.float32), -np.ones((1, 2, 1, 8), np.float32))


def test_parallax_holds_its_correction_at_a_scale_a_16_bit_gradient_would_round_away():
    """Parallax in fp16 at a logit scale of 2^-26, zero logits, a second query of 1 against second keys and
    values of -1 and 1: its correction 1 (the softmax's 0 less it), -1 exactly, where its dQ, the scale
    times it, rounded to fp16 was 0."""
    spans = template.spans(2, 2, 1)
    zeros = jnp.zeros((1, 2, 1, 8), jnp.float16)
    second = zeros.at[..., 0].set(1.0)
    pair = jnp.asarray([-1.0, 1.0], jnp.float16)
    second_key = zeros.at[0, :, 0, 0].set(pair)
    value = jnp.broadcast_to(pair[None, :, None, None], (1, 2, 1, 8))
    out = attention_products.parallax(zeros, second, zeros, second_key, value, spans, scale=2.0**-26)
    np.testing.assert_array_equal(np.asarray(out, np.float32), -np.ones((1, 2, 1, 8), np.float32))


def test_parallax_holds_a_correction_whose_scaled_gradient_passes_fp32s_least():
    """Parallax in fp32 at a logit scale of 2^-120, zero logits, a second query of 1 against second keys of
    -1 and 1 and values of -2^-40 and 2^-40: its output -2^-40 exactly, where the scale times it, 2^-160,
    passes fp32's least subnormal."""
    spans = template.spans(2, 2, 1)
    zeros = jnp.zeros((1, 2, 1, 8), jnp.float32)
    second = zeros.at[..., 0].set(1.0)
    second_key = zeros.at[0, :, 0, 0].set(jnp.asarray([-1.0, 1.0]))
    value = jnp.broadcast_to(jnp.asarray([-(2.0**-40), 2.0**-40])[None, :, None, None], (1, 2, 1, 8))
    out = attention_products.parallax(zeros, second, zeros, second_key, value, spans, scale=2.0**-120)
    np.testing.assert_array_equal(np.asarray(out), np.full((1, 2, 1, 8), -(2.0**-40), np.float32))


def test_a_banded_call_moves_a_narrow_place_past_its_dtypes_range():
    """A query placed at 127 (int8) over 129 keys, a near band of 1: keys 126, 127 and 128 are the near
    product's, where the edge 129 saturated at int8's 127 and left 127 and 128 out."""
    spans = template.spans(1, 129, 1)
    operand = jnp.zeros((1, 1, KV_HEADS, WIDTH), jnp.bfloat16)
    keys = jnp.zeros((1, 129, KV_HEADS, WIDTH), jnp.bfloat16)
    laid = attention_products.selected((operand,) * 2, (keys,) * 2, keys, spans,
                                       attention_products.Banded((1,)),
                                       places=jnp.asarray([[127]], jnp.int8))[3]
    near = np.asarray(template.dense(laid))[0, 0, :129]
    np.testing.assert_array_equal(np.flatnonzero(near), [126, 127, 128])


def test_a_banded_call_keeps_a_querys_place_in_its_own_dtype():
    """A query placed at 2^32 (int64) over two keys, a near band of 0: both keys are the far product's, the
    place's distance from each past int32's range, where an int32 place wrapped to 0 and took key 0 near."""
    spans = template.spans(1, 2, 1)
    operand = jnp.zeros((1, 1, KV_HEADS, WIDTH), jnp.bfloat16)
    keys = jnp.zeros((1, 2, KV_HEADS, WIDTH), jnp.bfloat16)
    with jax.enable_x64(new_val=True):
        laid = attention_products.selected((operand,) * 2, (keys,) * 2, keys, spans,
                                           attention_products.Banded((0,)),
                                           places=jnp.asarray([[2**32]], jnp.int64))[3]
        seen = np.asarray(template.dense(laid))[0, 0]
    np.testing.assert_array_equal(seen, [False, False, True, True])


def test_parallax_holds_its_second_products_mean_where_a_rounded_output_would_move_it():
    """Parallax over uniform logits, values all 1 and fp16 second keys 8192 and 8200 (their mean 8196, which
    fp16 rounds to 8192): its weights `a (1 + τ - t)` sum to 1, so its output is 1, its backward's D the
    second product's mean about an anchor key, not a rounded output's (the fast form's), which moved it by
    4."""
    spans = template.spans(2, 2, 1)
    query = jnp.zeros((1, 2, 1, 8), jnp.float16)
    second = jnp.zeros((1, 2, 1, 8), jnp.float16).at[..., 0].set(1.0)
    second_key = jnp.zeros((1, 2, 1, 8), jnp.float16).at[0, :, 0, 0].set(jnp.asarray([8192.0, 8200.0]))
    value = jnp.ones((1, 2, 1, 8), jnp.float16)
    out = attention_products.parallax(query, second, query, second_key, value, spans, scale=1.0)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.ones((1, 2, 1, 8), np.float32))
