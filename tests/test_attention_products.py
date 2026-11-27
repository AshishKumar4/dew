"""A step's score products (`attention_products`, kernel-templates.md §A rows 16a and 25): summed, and
selected by bands of distance, each on par with a reference written over its own products' logits (check 1),
its output and every product's gradients, in the fast and the accurate form, on the template's route and on
the Mosaic mainloop interpreted as an H100's and an A100's."""

import jax
import jax.numpy as jnp
import pytest
from generation_support import INTERPRETS
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_mosaic, attention_products
from dew.nn.kernels.templates.hooks import Hook
from dew.nn.precision import float32_einsum

pytestmark = pytest.mark.kernel

SEEDS = 8
TOKENS, HEADS, KV_HEADS, WIDTH = 160, 2, 1, 64
WINDOW = 37
"""A near band no tile's edge: 64- and 32-key tiles each hold keys of both products along it."""
SCALE = WIDTH ** -0.5
FORMS = {"fast": (False, False), "accurate": (True, True)}
"""Each form's `(natural, accurate)`."""


def _draw(products: int, dtype=jnp.bfloat16):
    """A draw of `products` queries and keys, a value and the output's cotangent, unit Gaussians by seed."""
    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 2 * products + 2)
        queries = [jax.random.normal(stream, (1, TOKENS, HEADS, WIDTH), jnp.float32).astype(dtype)
                   for stream in streams[:products]]
        keys = [jax.random.normal(stream, (1, TOKENS, KV_HEADS, WIDTH), jnp.float32).astype(dtype)
                for stream in streams[products:2 * products]]
        value = jax.random.normal(streams[-2], (1, TOKENS, KV_HEADS, WIDTH), jnp.float32).astype(dtype)
        dout = jax.random.normal(streams[-1], (1, TOKENS, HEADS, WIDTH), jnp.float32).astype(dtype)
        return (*queries, *keys, value, dout)

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
        distance = jnp.abs(jnp.arange(TOKENS)[:, None] - jnp.arange(spans.keys)[None, :])
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
@pytest.mark.parametrize(("case", "bands"),
                         [("causal", (WINDOW,)), ("full", (WINDOW,)), ("causal", (11, 53))],
                         ids=["causal-two", "full-two", "causal-three"])
def test_banded_products_are_on_par_with_their_logits_selected(case, bands, form):
    """Products selected by bands of distance (piecewise attention, §A row 16a), causal and full, two products
    and three, on the template's route: on par with a softmax over each key's own product's logits, its
    output and every product's gradients (check 1), the bands' edges inside key tiles."""
    spans = template.spans(TOKENS, TOKENS, 1, causal=case == "causal")
    products = len(bands) + 1
    ours = _routed(attention_products.Banded(bands), spans, form, products)
    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(_banded_reference(spans, bands))),
                    _draw(products), f"banded products {case} {bands} {form}", seeds=SEEDS)


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
                    f"banded products on the Mosaic mainloop {device} {form}", seeds=SEEDS)


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
    attention_products.selected((operand,) * 2, (operand,) * 2, operand, spans, combine,
                                hooked(lambda s, q, k: 30.0 * jnp.tanh(s / 30.0)))


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
