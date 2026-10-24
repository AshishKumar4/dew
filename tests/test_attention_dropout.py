"""Attention's probability dropout: the template's weights hook, on every route, and the selector's Route
for a dropping call. A dropping call on the template is on par with the reference path on the same draw,
forward and backward, both measured from float64 (check 1). The draw is one hash of the call's seed and
each entry's global batch row, query head, query and key, so every mainloop and the reference keep the
same entries; the overflow witnesses stay finite where the reference does, and the reference's draw is
unbiased in bf16.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_attention_route import CASES, HEADS, KV_HEADS, WIDTH, drawing, keywords
from test_attention_template import CASES as SPAN_CASES
from test_attention_template import DTYPES, KEYS, QUERIES, ROWS, SEEDS, arguments, dews_mask, geometry
from test_attention_template import drawing as value_drawing

from dew.nn.attention import dropout_draw, reference_attention, scaled_dot_product_attention
from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template
from dew.nn.kernels.templates import attention_portable, attention_route, attention_triton

RATE = 0.25
"""The dropout rate the check-1 tests drop at: a quarter of the weights."""

DROPPED = ("full-grouped", "causal-window", "packed-documents", "sinks-softcap")


def dropping(case: dict, implementation: str):
    """`case`'s call of `scaled_dot_product_attention` on `implementation`, dropping at `RATE` from
    `rng`, as a function of `(query, key, value, sinks, rng)`."""
    def call(query, key, value, sinks, rng):
        return scaled_dot_product_attention(query, key, value, implementation=implementation, sinks=sinks,
                                            dropout_rate=RATE, dropout_rng=rng, deterministic=False,
                                            **keywords(case))

    return call


def pulled(case: dict, implementation: str):
    """`case`'s dropping call on `implementation`: its output, then its gradients of the query, key,
    value and sinks under the cotangent `dout`, the sinks' gradient None where the case has none."""
    call = dropping(case, implementation)

    def run(query, key, value, sinks, dout, rng):
        out, pull = jax.vjp(lambda q, k, v, s: call(q, k, v, s, rng), query, key, value, sinks)
        return (out, *pull(dout))

    return run


def dropping_draw(queries: int, keys: int, dtype, *, sinks: bool):
    """A draw `(query, key, value, sinks, dout, rng)`: the route tests' draw, with the dropout key from
    its seed."""
    base = drawing(queries, keys, dtype, sinks=sinks)

    def draw(seed: int):
        return (*base(seed), jax.random.key(seed))

    return draw


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("name", DROPPED)
def test_a_dropping_template_call_is_on_par_with_the_reference_path_forward_and_backward(name, dtype):
    """The template's output and its gradients of the query, key, value and sinks, on a dropping call,
    on par with the reference path (`implementation='reference'`) on the same draw, both measured from
    float64 (check 1)."""
    case = CASES[name]
    queries, keys = geometry(case)
    assert_as_exact(jax.jit(pulled(case, "template")), jax.jit(pulled(case, "reference")),
                    dropping_draw(queries, keys, dtype, sinks=bool(case.get("sinks"))),
                    f"dropping template {name} {jnp.dtype(dtype).name}", seeds=SEEDS)


def test_a_dropping_call_reports_its_route_and_a_chosen_template_runs_it():
    """A dropping call reports the route it runs (the reference path on a CPU, where the template is
    not preferred), and `generation.chosen(attention='template')` runs it on the template instead."""
    query = jnp.zeros((ROWS, QUERIES, HEADS, WIDTH), jnp.bfloat16)
    key = jnp.zeros((ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
    rng = jax.random.key(0)

    def dropping_call():
        return scaled_dot_product_attention(query, key, key, dropout_rate=RATE, dropout_rng=rng,
                                            deterministic=False)

    with generation.routes() as found:
        dropping_call()
    (route,) = [route for route in found if route.choice == "attention"]
    assert route.kernel == "reference"
    with generation.chosen(attention="template"), generation.routes() as chosen:
        out = dropping_call()
    (named,) = [route for route in chosen if route.choice == "attention"]
    assert named.kernel == "template"
    assert out.shape == (ROWS, QUERIES, HEADS, WIDTH)


def test_the_reference_draw_keeps_a_bf16_call_unbiased():
    """Each of 128 keys of equal weight 1/128 (exact in bf16) keeps with probability 0.9 and scales by
    1/0.9, so each query row's output has mean 1 over the draws. The 16384 rows are independent draws:
    a row's output has standard deviation 0.0295, so the mean of the rows has standard error 0.00023,
    and the bound is 3.5 of those. A multiplier rounded to bf16 (1.109375 for 1/0.9) would bias the
    mean by -0.00156."""
    query = jnp.zeros((1, 16384, 1, 8), jnp.bfloat16)
    key = jnp.zeros((1, 128, 1, 8), jnp.bfloat16)
    value = jnp.ones((1, 128, 1, 8), jnp.bfloat16)
    out = scaled_dot_product_attention(query, key, value, implementation="reference", dropout_rate=0.1,
                                       dropout_rng=jax.random.key(0), deterministic=False)
    assert abs(float(jnp.mean(out.astype(jnp.float32))) - 1) < 0.0008


HOOKED = ("causal-window", "causal-documents-window", "cross-causal-lengths-starts")
"""The span forms the template-level check-1 tests drop over: a window, documents under a window, and
cross documents with lengths and starts."""


def seed_words(seed: int) -> jax.Array:
    """A call's two dropout words for the draw `seed`, the same on every arm."""
    return jax.random.bits(jax.random.key(1000 + seed), (2,), jnp.uint32)


def hooked(words: jax.Array, batch: int, heads: int, rate: float = RATE) -> template.Hooks:
    """The template's hooks dropping at `rate` from `words`, each batch row and query head its own global
    index."""
    dropout = template.Dropout(words, jnp.arange(batch, dtype=jnp.int32), jnp.arange(heads, dtype=jnp.int32),
                               rate)
    return template.Hooks(drop=dropout)


def route_arm(case: dict):
    """`case`'s template route over its spans, dropping from the seed words, as `(query, key, value, words)`."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))

    def call(query, key, value, words):
        return attention_route.attend(query, key, value, spans, hooked(words, ROWS, query.shape[2]))

    return call


def reference_arm(case: dict):
    """`case`'s reference path over the same keys, dropping from the seed words."""
    mask = jnp.asarray(dews_mask(case))[:, None]

    def call(query, key, value, words):
        return reference_attention(query, key, value, mask=mask, dropout_rate=RATE, dropout_seed=words)

    return call


def grads_of(call):
    """`call`'s output and its gradients of the query, key and value, under the cotangent `dout`."""
    def run(query, key, value, words, dout):
        out, pull = jax.vjp(lambda q, k, v: call(q, k, v, words), query, key, value)
        return (out, *pull(dout))

    return run


def route_draw(queries: int, keys: int, dtype):
    """A draw `(query, key, value, words, dout)`: the value draw's inputs, the dropout words of its seed
    and a cotangent."""
    base = value_drawing(queries, keys, dtype)

    def draw(seed: int):
        query, key, value = base(seed)
        dout = jax.random.normal(jax.random.key(2000 + seed), query.shape, dtype)
        return query, key, value, seed_words(seed), dout

    return draw


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("name", HOOKED)
def test_the_template_route_drops_on_par_with_the_reference_path(name, dtype):
    """The route's output and its gradients of the query, key and value, dropping over the spans of each
    form, on par with the reference path on the same draw (`reference_attention`'s dropout from the same
    words), both measured from float64 (check 1)."""
    case = SPAN_CASES[name]
    queries, keys = geometry(case)
    assert_as_exact(jax.jit(grads_of(route_arm(case))), jax.jit(grads_of(reference_arm(case))),
                    route_draw(queries, keys, dtype), f"dropped route {name} {jnp.dtype(dtype).name}",
                    seeds=SEEDS)


def first_seed(rate: float, keys: int, wanted) -> jax.Array:
    """The dropout words of the first draw seed whose keep pattern over one query's `keys` keys satisfies
    `wanted`, a predicate of the boolean row. The draw is the template's (`template.Draw`), vectorized over
    the seeds, so the witness's precondition is a fact of the draw itself."""
    seeds = jnp.arange(1 << 16, dtype=jnp.uint32)[:, None, None]
    draw = template.Draw((seeds, jnp.zeros_like(seeds)), jnp.int32(0), jnp.int32(0), rate)
    kept = np.asarray(draw(jnp.zeros((1, 1), jnp.int32), jnp.arange(keys, dtype=jnp.int32)[None, :]) > 0)
    seed = next(index for index in range(kept.shape[0]) if wanted(kept[index, 0]))
    return jnp.asarray([seed, 0], jnp.uint32)


def probe(words: jax.Array, queries: int, keys: int, batch: int, heads: int, *, block: tuple[int, int]):
    """The keep pattern of a call over uniform weights and identity values: each output entry is its
    entry's draw times `1 / keys`, so its nonzero entries are the kept ones, as the portable mainloop
    computes them at the tiles `block`."""
    query = jnp.zeros((batch, queries, heads, 8), jnp.float32)
    key = jnp.zeros((batch, keys, heads, 8), jnp.float32)
    value = jnp.broadcast_to(jnp.eye(keys, dtype=jnp.float32), (batch, keys, heads, keys))
    spans = template.spans(queries, keys, batch)
    out = attention_portable.attend(query, key, value, spans, hooked(words, batch, heads), block_q=block[0],
                                    block_k=block[1], pad=8)
    return out > 0


@pytest.mark.parametrize("block", [(8, 8), (4, 8), (8, 4)], ids=["8x8", "4x8", "8x4"])
def test_the_portable_mainloop_draws_the_reference_mask_at_every_tiling(block):
    """The portable mainloop's kept entries, at tiles of each shape, are the reference's draw bit for bit
    (`dropout_draw`'s mask over the same coordinates): the draw depends on the global coordinates alone."""
    words = seed_words(3)
    mask = np.asarray(dropout_draw(words, RATE, (2, 4, 8, 16))) > 0
    np.testing.assert_array_equal(np.asarray(probe(words, 8, 16, 2, 4, block=block)),
                                  mask.transpose(0, 2, 1, 3))


def test_the_triton_mainloop_interpreted_draws_the_reference_mask():
    """The Triton mainloop, interpreted, keeps the entries the reference draws: the same identity probe
    through its forward kernel, bit for bit."""
    words = seed_words(3)
    query = jnp.zeros((2, 8, 4, 8), jnp.float32)
    key = jnp.zeros((2, 16, 2, 8), jnp.float32)
    value = jnp.broadcast_to(jnp.eye(16, dtype=jnp.float32), (2, 16, 2, 16))
    spans = template.spans(8, 16, 2)
    out = attention_triton.forward(query, key, value, spans, hooked(words, 2, 4), interpret=True)
    mask = np.asarray(dropout_draw(words, RATE, (2, 4, 8, 16))) > 0
    np.testing.assert_array_equal(np.asarray(out) > 0, mask.transpose(0, 2, 1, 3))


def test_the_draw_under_vmap_is_each_calls_own_draw():
    """Vmapped over seeds, each element keeps the entries its own seed draws (a batched seed is a batched
    draw), as a call of that seed alone does."""
    seeds = jnp.stack([seed_words(index) for index in (1, 2, 3)])
    mapped = jax.vmap(lambda words: probe(words, 8, 16, 2, 4, block=(8, 8)))(seeds)
    for index, words in enumerate(seeds):
        np.testing.assert_array_equal(np.asarray(mapped[index]),
                                      np.asarray(probe(words, 8, 16, 2, 4, block=(8, 8))))


def test_a_kept_anchor_at_twice_the_weight_stays_finite_where_the_reference_does():
    """The fp32 anchor witness: two keys of equal logits, the first (the anchor) kept at z = 2 and the second
    dropped, with values 2^127 and 0. Its own term z * 2^127 is past fp32's largest before its weight of one
    half takes it back, so the output must be the normalized 2^127, finite, as the reference's is."""
    words = first_seed(0.5, 2, lambda kept: bool(kept[0]) and not bool(kept[1]))
    query = jnp.zeros((1, 1, 1, 8), jnp.float32)
    key = jnp.zeros((1, 2, 1, 8), jnp.float32)
    value = jnp.stack([jnp.full((8,), 2.0 ** 127), jnp.zeros((8,))])[None, :, None, :]
    spans = template.spans(1, 2, 1)
    dew = attention_portable.attend(query, key, value, spans, hooked(words, 1, 1, 0.5), block_q=8,
                                    block_k=8, pad=8)
    ref = reference_attention(query, key, value, dropout_rate=0.5, dropout_seed=words)
    assert bool(jnp.all(jnp.isfinite(ref)))
    assert bool(jnp.all(jnp.isfinite(dew)))
    np.testing.assert_allclose(np.asarray(dew, np.float64), 2.0 ** 127, rtol=1e-6)


def test_bf16_values_at_two_to_the_124_under_a_steep_draw_stay_finite_where_the_reference_does():
    """The bf16 witness: values of 2^124 under a rate of 0.95 (z = 20), with at least two of 64 keys kept.
    Each kept weight is 1/64 of 20 * 2^124, and their sum is finite, while their unnormalized products
    overflow fp32 unless the headroom bound counts z."""
    words = first_seed(0.95, 64, lambda kept: int(kept.sum()) >= 2)
    query = jnp.zeros((1, 1, 1, 8), jnp.bfloat16)
    key = jnp.zeros((1, 64, 1, 8), jnp.bfloat16)
    value = jnp.full((1, 64, 1, 8), 2.0 ** 124, jnp.bfloat16)
    spans = template.spans(1, 64, 1)
    dew = attention_portable.attend(query, key, value, spans, hooked(words, 1, 1, 0.95), block_q=8,
                                    block_k=8, pad=8)
    ref = reference_attention(query, key, value, dropout_rate=0.95, dropout_seed=words)
    assert bool(jnp.all(jnp.isfinite(ref)))
    assert bool(jnp.all(jnp.isfinite(dew)))


def test_fp16_values_at_two_to_the_minus_ten_under_a_near_total_draw_stay_finite_where_the_reference_does():
    """The fp16 witness: a rate of 1 - 2^-16, so z = 2^16, with exactly one of two equal-logit keys kept.
    Its unnormalized weight is z, past fp16's largest, where its normalized weight of one half times z is
    not, as the reference's is."""
    rate = 1.0 - 2.0 ** -16
    words = first_seed(rate, 2, lambda kept: int(kept.sum()) == 1)
    query = jnp.zeros((1, 1, 1, 8), jnp.float16)
    key = jnp.zeros((1, 2, 1, 8), jnp.float16)
    value = jnp.full((1, 2, 1, 8), 2.0 ** -10, jnp.float16)
    spans = template.spans(1, 2, 1)
    dew = attention_portable.attend(query, key, value, spans, hooked(words, 1, 1, rate), block_q=8,
                                    block_k=8, pad=8)
    ref = reference_attention(query, key, value, dropout_rate=rate, dropout_seed=words)
    assert bool(jnp.all(jnp.isfinite(ref)))
    assert bool(jnp.all(jnp.isfinite(dew)))
