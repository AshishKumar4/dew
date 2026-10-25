"""Attention's probability dropout: the template's weights hook, on every route, and the selector's Route
for a dropping call. A dropping call on the template is on par with the reference path on the same draw,
forward and backward, both measured from float64 (check 1). The draw is one hash of the call's key and each
entry's global batch row, query head, query and key, so every mainloop and the reference keep the same
entries; the overflow witnesses stay finite where the reference does, and the reference's draw is unbiased
in bf16.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_attention_route import CASES, HEADS, KV_HEADS, WIDTH, drawing, keywords
from test_attention_template import (
    CASES as SPAN_CASES,
    DTYPES,
    KEYS,
    QUERIES,
    ROWS,
    SEEDS,
    SINKS,
    SOFTCAP,
    arguments,
    dews_mask,
    drawing as value_drawing,
    geometry,
)

from dew.nn.attention import attention_case, reference_attention, scaled_dot_product_attention
from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_portable,
    attention_route,
    attention_triton,
    mosaic_sm90,
)
from dew.nn.kernels.templates.attention import dropout_keep, dropout_seed


@pytest.fixture(autouse=True)
def _off_a_gpu(request):
    """Each site runs where the template takes dropout, its plain-JAX mainloop off a GPU
    (`attention_route.drops`; a GPU host's CPU), but the GPU sites, which hold the GPU's refusal."""
    if request.node.get_closest_marker("gpu") is not None:
        yield
        return
    with jax.default_device(jax.devices("cpu")[0]):
        yield


RATE = 0.25
"""The dropout rate the check-1 tests drop at: a quarter of the weights."""

DROPPED = ("full-grouped", "causal-window", "packed-documents", "sinks-softcap")
"""The public calls the check-1 tests drop over: grouped heads, a window, packed documents, and sinks with a
softcap."""

HOOKED = ("causal-window", "causal-documents-window", "cross-causal-lengths-starts")
"""The span forms the template route's check-1 tests drop over: a window, documents under a window, and
cross documents with lengths and starts."""

PUBLIC = ("cross-causal-lengths-starts", "cross-causal-window", "cross-window")
"""The cross-document forms whose dropping calls the public call materializes for the reference path:
causality or a window aligns each document to its last keys, and the key bounds ride beside them."""


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
        return (*base(seed), jax.random.PRNGKey(seed))

    return draw


def keyed_draw(queries: int, keys: int, dtype):
    """A draw `(query, key, value, rng, dout)`: the value draw's inputs, a dropout key of its seed and a
    cotangent."""
    base = value_drawing(queries, keys, dtype)

    def draw(seed: int):
        query, key, value = base(seed)
        dout = jax.random.normal(jax.random.PRNGKey(2000 + seed), query.shape, dtype)
        return query, key, value, jax.random.PRNGKey(1000 + seed), dout

    return draw


def grads_of(call):
    """`call`'s output and its gradients of the query, key and value, under the cotangent `dout`."""
    def run(query, key, value, rng, dout):
        out, pull = jax.vjp(lambda q, k, v: call(q, k, v, rng), query, key, value)
        return (out, *pull(dout))

    return run


def dropout_of(words: jax.Array, batch: int, heads: int, rate: float = RATE) -> template.Dropout:
    """The dropout hook of the seed `words`, each batch row and query head its own global index."""
    return template.Dropout(words, jnp.arange(batch, dtype=jnp.int32), jnp.arange(heads, dtype=jnp.int32),
                            rate)


def hooked(words: jax.Array, batch: int, heads: int, rate: float = RATE) -> template.Hooks:
    """The template's hooks with only the dropout hook of `dropout_of`."""
    return template.Hooks(drop=dropout_of(words, batch, heads, rate))


def route_arm(case: dict):
    """`case`'s template route over its spans, dropping from the call's key, as `(query, key, value, rng)`."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))

    def call(query, key, value, rng):
        return attention_route.attend(query, key, value, spans, hooked(dropout_seed(rng), ROWS,
                                                                       query.shape[2]))

    return call


def reference_arm(case: dict, names: frozenset = frozenset()):
    """`case`'s reference path over its dense mask `dews_mask(case)`, dropping at `RATE` from the call's
    key, with the score hooks `names` (`sinks`, `softcap`); a query that sees no key reads zero, as every
    attention call reads (`attention_kernel`)."""
    mask = np.asarray(dews_mask(case))
    live = jnp.asarray(mask.any(axis=-1))[:, :, None, None]
    sinks = SINKS if "sinks" in names else None
    softcap = SOFTCAP if "softcap" in names else None

    def call(query, key, value, rng):
        out = reference_attention(query, key, value, mask=jnp.asarray(mask)[:, None], sinks=sinks,
                                  softcap=softcap, dropout_rate=RATE, dropout_seed=dropout_seed(rng))
        return jnp.where(live, out, 0)

    return call


def public_keywords(case: dict) -> dict:
    """`case`'s keywords of `scaled_dot_product_attention` for its documents, key bounds and block and image
    groups."""
    spans = arguments(case)
    return {"causal": spans.get("causal", False), "sliding_window": spans.get("sliding_window"),
            "segment_ids": spans.get("segment_ids"), "key_value_seq_lengths": spans.get("lengths"),
            "key_value_seq_starts": spans.get("starts"), "block_groups": spans.get("block_groups"),
            "image_groups": spans.get("image_groups")}


def public_arm(case: dict, implementation: str):
    """`case`'s dropping public call on `implementation`, as `(query, key, value, rng)`."""
    options = public_keywords(case)

    def call(query, key, value, rng):
        return scaled_dot_product_attention(query, key, value, implementation=implementation,
                                            dropout_rate=RATE, dropout_rng=rng, deterministic=False,
                                            **options)

    return call


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
    rng = jax.random.PRNGKey(0)

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


def test_a_dropping_decode_step_takes_the_reference_path_and_never_the_decode_kernel():
    """A dropping call shaped as a decode step (one query over key lengths) is not the decode kernel's: it
    reads no draw, so the selector runs it on the reference path, which reads the lengths and drops."""
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    query = jax.random.normal(keys[0], (ROWS, 1, HEADS, WIDTH), jnp.bfloat16)
    key = jax.random.normal(keys[1], (ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
    value = jax.random.normal(keys[2], (ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
    lengths = jnp.full((ROWS,), KEYS - 3, jnp.int32)
    with generation.routes() as found:
        out = scaled_dot_product_attention(query, key, value, key_value_seq_lengths=lengths,
                                           dropout_rate=RATE, dropout_rng=jax.random.PRNGKey(0),
                                           deterministic=False)
    (route,) = [route for route in found if route.choice == "attention"]
    assert route.kernel == "reference"
    assert out.shape == (ROWS, 1, HEADS, WIDTH)


def test_a_dropping_call_at_eval_takes_no_dropout():
    """A call that is not training (`deterministic=True`) takes no dropout whatever its rate, on every
    route: each route's output is its own undropped output, bit for bit."""
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    query = jax.random.normal(keys[0], (ROWS, QUERIES, HEADS, WIDTH), jnp.bfloat16)
    key = jax.random.normal(keys[1], (ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
    value = jax.random.normal(keys[2], (ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
    for implementation in ("reference", "template"):
        plain = scaled_dot_product_attention(query, key, value, implementation=implementation)
        out = scaled_dot_product_attention(query, key, value, implementation=implementation,
                                           dropout_rate=RATE, dropout_rng=jax.random.PRNGKey(0),
                                           deterministic=True)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(plain))


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("name", HOOKED)
def test_the_template_route_drops_on_par_with_the_reference_path(name, dtype):
    """The route's output and its gradients of the query, key and value, dropping over the spans of each
    form, on par with the reference path on the same draw (`reference_attention`'s dropout from the same
    key), both measured from float64 (check 1)."""
    case = SPAN_CASES[name]
    queries, keys = geometry(case)
    assert_as_exact(jax.jit(grads_of(route_arm(case))), jax.jit(grads_of(reference_arm(case))),
                    keyed_draw(queries, keys, dtype), f"dropped route {name} {jnp.dtype(dtype).name}",
                    seeds=SEEDS)


@pytest.mark.parametrize("implementation", ["reference", "template"])
@pytest.mark.parametrize("name", PUBLIC)
def test_a_dropping_call_materializes_its_documents_and_key_bounds_as_the_mask_they_mean(name,
                                                                                         implementation):
    """A dropping public call on `implementation`, with cross documents, causality or a window, and key
    bounds, is on par with the reference path over the dense mask those documents and bounds mean, on the
    same draw, forward and backward, both measured from float64 (check 1)."""
    case = SPAN_CASES[name]
    queries, keys = geometry(case)
    assert_as_exact(jax.jit(grads_of(public_arm(case, implementation))),
                    jax.jit(grads_of(reference_arm(case))), keyed_draw(queries, keys, jnp.float32),
                    f"dropping public {implementation} {name}", seeds=SEEDS)


STRUCTURED = ("block-causal",)
"""Call forms the template reads as spans and every other path as the mask they mean: block groups."""


@pytest.mark.parametrize("name", STRUCTURED)
def test_a_dropping_named_template_call_reads_its_structured_visibility(name):
    """A dropping public call naming the template, over block groups, runs on the template's spans, on par
    with the reference path over the dense mask they mean, on the same draw, forward and backward (check 1):
    the call keeps its structure until the route is chosen."""
    case = SPAN_CASES[name]
    queries, keys = geometry(case)
    assert_as_exact(jax.jit(grads_of(public_arm(case, "template"))), jax.jit(grads_of(reference_arm(case))),
                    keyed_draw(queries, keys, jnp.float32), f"dropping structured template {name}",
                    seeds=SEEDS)


def first_seed(rate: float, keys: int, wanted) -> jax.Array:
    """The dropout words of the first draw seed whose keep pattern over one query's `keys` keys satisfies
    `wanted`, a predicate of the boolean row. The draw is the template's (`template.Draw`), vectorized over
    the seeds, so the witness's precondition is a fact of the draw itself."""
    seeds = jnp.arange(1 << 16, dtype=jnp.uint32)[:, None, None]
    draw = template.Draw((seeds, jnp.zeros_like(seeds)), jnp.int32(0), jnp.int32(0), rate)
    kept = np.asarray(draw.keep(jnp.zeros((1, 1), jnp.int32), jnp.arange(keys, dtype=jnp.int32)[None, :]))
    seed = next(index for index in range(kept.shape[0]) if wanted(kept[index, 0]))
    return jnp.asarray([seed, 0], jnp.uint32)


def probe(words: jax.Array, queries: int, keys: int, batch: int, heads: int, *, block: tuple[int, int]):
    """The keep pattern of a call over uniform weights and identity values: each output entry is its
    entry's draw times `1 / keys`, so its nonzero entries are the kept ones, as the portable mainloop
    computes them at the tiles `block`."""
    query = jnp.zeros((batch, queries, heads, 8), jnp.float32)
    key = jnp.zeros((batch, keys, heads, 8), jnp.float32)
    value = jnp.broadcast_to(jnp.eye(keys, dtype=jnp.float32)[None, :, None, :], (batch, keys, heads, keys))
    spans = template.spans(queries, keys, batch)
    out = attention_portable.attend(query, key, value, spans, hooked(words, batch, heads), block_q=block[0],
                                    block_k=block[1], pad=8)
    return out > 0


def seed_words(seed: int) -> jax.Array:
    """A call's two dropout words for the draw `seed`, the same on every arm."""
    return jax.random.bits(jax.random.PRNGKey(1000 + seed), (2,), jnp.uint32)


@pytest.mark.parametrize("block", [(8, 8), (4, 8), (8, 4)], ids=["8x8", "4x8", "8x4"])
def test_the_portable_mainloop_draws_the_reference_mask_at_every_tiling(block):
    """The portable mainloop's kept entries, at tiles of each shape, are the reference's draw bit for bit
    (`dropout_keep`'s mask over the same coordinates): the draw depends on the global coordinates alone."""
    words = seed_words(3)
    mask = np.asarray(dropout_keep(words, RATE, (2, 4, 8, 16)))
    np.testing.assert_array_equal(np.asarray(probe(words, 8, 16, 2, 4, block=block)),
                                  mask.transpose(0, 2, 1, 3))


def test_the_draw_under_vmap_is_each_calls_own_draw():
    """Vmapped over seeds, each element keeps the entries its own seed draws (a batched seed is a batched
    draw), as a call of that seed alone does. The probe's fp32 products are promoted (`Precision.HIGHEST`),
    and `interpreted_dot` issues those in float64 off a GPU only: on an A100 a vmapped call of them fails
    (receipt DROPOUT-a100: `GEMM is not supported by cublasLt`), so the comparison runs on the CPU, where
    the arm issues its products."""
    with jax.default_device(jax.devices("cpu")[0]):
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
    # Equal logits give each weight one half, z = 2 keeps the anchor's at one, and every step is a power of
    # two, so the output is 2^127 exactly.
    np.testing.assert_array_equal(np.asarray(dew), np.full(dew.shape, 2.0 ** 127, np.float32))


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


def identical_keys(seed: int, queries: int, keys: int, dtype):
    """A query, a cotangent, and a key and value repeating one row across the keys: every query's keys are
    identical, so its score cotangents must sum to zero over them."""
    streams = jax.random.split(jax.random.PRNGKey(seed), 3)
    row = jax.random.normal(streams[1], (ROWS, 1, KV_HEADS, WIDTH), jnp.float32).astype(dtype)
    query = jax.random.normal(streams[0], (ROWS, queries, HEADS, WIDTH), jnp.float32).astype(dtype)
    dout = jax.random.normal(streams[2], (ROWS, queries, HEADS, WIDTH), jnp.float32).astype(dtype)
    key = jnp.broadcast_to(row, (ROWS, keys, KV_HEADS, WIDTH))
    return query, key, key, dout


def test_identical_keys_give_an_exactly_zero_query_gradient_on_the_portable_mainloop():
    """Two identical keys and values, both kept at rate 1/4, in bf16. Each query's score cotangents sum to
    zero over its keys, and identical keys multiply each by the same key, so dQ is exactly zero, as the
    reference's is. A first D from the forward's narrowed dropped weights leaves dQ off zero (-3.26e-4)."""
    words = first_seed(RATE, 2, lambda kept: bool(kept.all()))
    query, key, value, dout = identical_keys(0, 3, 2, jnp.bfloat16)
    grads = attention_portable.backward(query, key, value, template.spans(3, 2, ROWS),
                                        hooked(words, ROWS, HEADS), dout, block_q=8, block_k=8,
                                        pad=attention_triton.BLOCK)
    np.testing.assert_array_equal(np.asarray(grads.query, np.float32), np.zeros(query.shape, np.float32))


def test_a_kept_bf16_weight_is_scaled_in_fp32_and_rounded_once():
    """Ledger 1533: three equal weights of 1/3 at rate 0.1, in bf16. A kept weight is 1/3 times 1/0.9 in fp32,
    rounded once to bf16: 0.37109375 (190/512). Rounding the multiplier to bf16 first (1.109375) gives
    0.369140625 (189/512)."""
    rate = 0.1
    words = first_seed(rate, 3, lambda kept: bool(kept.any()))
    query = jnp.zeros((1, 1, 1, 4), jnp.bfloat16)
    key = jnp.zeros((1, 3, 1, 4), jnp.bfloat16)
    value = jnp.broadcast_to(jnp.eye(3, dtype=jnp.bfloat16)[None, :, None, :], (1, 3, 1, 3))
    out = reference_attention(query, key, value, dropout_rate=rate, dropout_seed=words)
    kept = np.asarray(dropout_keep(words, rate, (1, 1, 1, 3)))[0, 0, 0]
    row = np.asarray(out, np.float32)[0, 0, 0]
    np.testing.assert_array_equal(row[kept], np.full(int(kept.sum()), 0.37109375, np.float32))
    np.testing.assert_array_equal(row[~kept], np.zeros(int((~kept).sum()), np.float32))


def test_the_template_reference_narrows_the_normalized_dropped_coefficient():
    """The fp16 witness on the template's reference: rate 1 - 2^-16 (z = 2^16), one of two equal-logit keys
    kept. Narrowed unnormalized, its weight z is past fp16's largest and the output is infinite. Normalized
    first, the weight is z / 2 = 2^15 and the output is 2^5 = 32, exactly."""
    rate = 1.0 - 2.0 ** -16
    words = first_seed(rate, 2, lambda kept: int(kept.sum()) == 1)
    query = jnp.zeros((1, 1, 1, 8), jnp.float16)
    key = jnp.zeros((1, 2, 1, 8), jnp.float16)
    value = jnp.full((1, 2, 1, 8), 2.0 ** -10, jnp.float16)
    out = template.reference(query, key, value, template.spans(1, 2, 1), hooked(words, 1, 1, rate))
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.full((1, 1, 1, 8), 32.0, np.float32))


def test_a_dropping_call_under_x64_on_fp32_inputs_keeps_fp32_gradients():
    """The dropout multiplier stays fp32 under x64 (`Draw`): a float64 multiplier changes the backward's loop
    carries, so the gradients would not stay fp32 (review)."""
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    with jax.enable_x64(new_val=True):
        streams = jax.random.split(jax.random.PRNGKey(0), 3)
        query = jax.random.normal(streams[0], (ROWS, QUERIES, HEADS, WIDTH), jnp.float32)
        key = jax.random.normal(streams[1], (ROWS, KEYS, KV_HEADS, WIDTH), jnp.float32)
        value = jax.random.normal(streams[2], (ROWS, KEYS, KV_HEADS, WIDTH), jnp.float32)
        hooks = hooked(seed_words(4), ROWS, HEADS)
        out, pull = jax.vjp(lambda q, k, v: attention_route.attend(q, k, v, spans, hooks), query, key, value)
        grads = pull(jnp.ones_like(out))
    for array in (out, *grads):
        assert array.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(array)))


@pytest.mark.gpu(reason="the route's GPU mainloops refuse dropout where calls run on an sm80 GPU or later")
def test_a_dropping_call_on_a_gpu_takes_the_template_only_where_its_mainloop_draws():
    """On a GPU the route's mainloops of sm80 and later, Triton's and Mosaic's, take no dropout yet, and
    before sm80 the plain-JAX mainloop runs every call and draws it (`attention_route.drops`). The selector
    refuses the template for a dropping call, naming row 1521, and a direct route call refuses rather than
    drop nothing, exactly where the route's mainloops take none; elsewhere the route drops."""
    from dew.nn.attention_kernels import ATTENTION, kernel_refusals
    from dew.nn.kernels.templates import attention_route

    query, key, value = (jnp.ones((ROWS, 8, HEADS, WIDTH), jnp.bfloat16) for _ in range(3))
    case = attention_case(query, key, dropout=True)
    drops = attention_route.drops()
    assert drops == (not generation.bf16_dot_runs())
    refused = any(refusal.row == "1521" for refusal in kernel_refusals(ATTENTION, "template", case))
    assert refused == (not drops)
    call = functools.partial(attention_route.attend, query, key, value, template.spans(8, 8, ROWS),
                             template.Hooks(drop=dropout_of(seed_words(0), ROWS, HEADS)))
    if drops:
        assert bool(jnp.all(jnp.isfinite(call())))
    else:
        with pytest.raises(NotImplementedError):
            call()


def test_the_gpu_mainloops_refuse_a_dropping_call_at_every_entry():
    """Mosaic's and the frozen Triton mainloop's launches take no probability dropout yet: each entry point,
    the forward and the backward called directly, refuses a dropping call before any launch rather than
    run it undropped, and so does each legality rule (ledger row 1521)."""
    query, key, value = (jnp.ones((ROWS, 8, HEADS, WIDTH), jnp.bfloat16) for _ in range(3))
    spans = template.spans(8, 8, ROWS)
    hooks = hooked(seed_words(0), ROWS, HEADS)
    stats = jnp.zeros((ROWS, HEADS, 8), jnp.float32)
    residuals = (stats, stats, jnp.zeros((ROWS, HEADS, 8), jnp.int32),
                 template.Split(jnp.zeros((ROWS, HEADS, 8, WIDTH), jnp.float32), stats, stats, stats))
    for backward in (False, True):
        assert attention_mosaic.refused(jnp.bfloat16, WIDTH, WIDTH, hooks, backward=backward,
                                        family=mosaic_sm90.WGMMA) is not None
    assert attention_triton.refused(spans, hooks) is not None
    with generation.assumed("sm90"):
        with pytest.raises(NotImplementedError):
            attention_mosaic.forward(query, key, value, spans, hooks, interpret=True)
        with pytest.raises(NotImplementedError):
            attention_mosaic.backward(query, key, value, spans, hooks, query, residuals=residuals,
                                      interpret=True)
    with pytest.raises(NotImplementedError):
        attention_triton.forward(query, key, value, spans, hooks, interpret=True)
    with pytest.raises(NotImplementedError):
        attention_triton.backward(query, key, value, spans, hooks, query, residuals=residuals, interpret=True)


def three_identical_keys():
    """One zero query of width 16 over three identical keys (first coordinate 1), values (0.75, 3u, 0, ...)
    with u = 2^-23, and dO = (1, 1, 0, ...), in bf16: at a rate of 1/2 each kept product is 2 (0.75 + 3u)."""
    u = 2.0 ** -23
    row = jnp.zeros((16,), jnp.bfloat16).at[:2].set(jnp.array([0.75, 3 * u], jnp.bfloat16))
    query = jnp.zeros((1, 1, 1, 16), jnp.bfloat16)
    key = jnp.zeros((1, 3, 1, 16), jnp.bfloat16).at[..., 0].set(1)
    value = jnp.broadcast_to(row, (1, 3, 1, 16))
    dout = jnp.zeros((1, 1, 1, 16), jnp.bfloat16).at[..., :2].set(1)
    return query, key, value, dout


def overflowing_keys():
    """One zero query of width 16 over three identical keys (first coordinate 1), values whose first
    coordinate is 0, 2^126 and 2^126, and dO = (1, 0, ...), in bf16. At a rate of 1/2 the dropped products
    are 0, 2^127 and 2^127, whose raw sum overflows fp32."""
    query = jnp.zeros((1, 1, 1, 16), jnp.bfloat16)
    key = jnp.zeros((1, 3, 1, 16), jnp.bfloat16).at[..., 0].set(1)
    value = jnp.zeros((1, 3, 1, 16), jnp.bfloat16).at[0, 1:, 0, 0].set(2.0 ** 126)
    dout = jnp.zeros((1, 1, 1, 16), jnp.bfloat16).at[..., 0].set(1)
    return query, key, value, dout


def reference_query_gradient(query, key, value, dout, words, rate: float):
    """The reference path's gradient of the query under `dout`, dropping at `rate` from `words`."""
    _, pull = jax.vjp(
        lambda q: reference_attention(q, key, value, dropout_rate=rate, dropout_seed=words), query)
    return pull(dout)[0]


def test_three_identical_keys_give_an_exactly_zero_query_gradient_on_the_portable_mainloop():
    """Three identical keys, all kept at rate 1/2, in bf16: each product is the same, so their query's score
    cotangents sum to zero and dQ is exactly zero, as the reference's is. The first pass's centre divided
    two rounded sums, 3 (1.5 + 6u) rounding to 4.5 + 16u, and the ratio 1.5 + 5u left dQ near 2^-25."""
    words = first_seed(0.5, 3, lambda kept: bool(kept.all()))
    query, key, value, dout = three_identical_keys()
    dq = reference_query_gradient(query, key, value, dout, words, 0.5)
    np.testing.assert_array_equal(np.asarray(dq, np.float32), np.zeros(query.shape, np.float32))
    grads = attention_portable.backward(query, key, value, template.spans(1, 3, 1), hooked(words, 1, 1, 0.5),
                                        dout, block_q=8, block_k=8, pad=attention_triton.BLOCK)
    np.testing.assert_array_equal(np.asarray(grads.query, np.float32), np.zeros(query.shape, np.float32))


def test_finite_products_whose_raw_sum_overflows_keep_a_finite_query_gradient_on_the_portable_mainloop():
    """Three keys whose dropped products are 0, 2^127 and 2^127: their raw sum, 2^128, is past fp32's
    largest, where the reference's weighted mean is 2^128 / 3. The query gradient must be finite, too."""
    words = first_seed(0.5, 3, lambda kept: bool(kept.all()))
    query, key, value, dout = overflowing_keys()
    dq = reference_query_gradient(query, key, value, dout, words, 0.5)
    assert bool(np.all(np.isfinite(np.asarray(dq, np.float32))))
    grads = attention_portable.backward(query, key, value, template.spans(1, 3, 1), hooked(words, 1, 1, 0.5),
                                        dout, block_q=8, block_k=8, pad=attention_triton.BLOCK)
    assert bool(np.all(np.isfinite(np.asarray(grads.query, np.float32))))


@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
def test_a_resumed_state_of_a_large_denominator_keeps_a_finite_dropping_backward(accurate):
    """One zero fp32 query over one kept key, its value's first coordinate 2^100 and dO's 1, resumed from a
    state of maximum 0, denominator 2^100 and zero sums, at rate 1/2: the dropped product is 2^101 and the
    output finite, so the reference's gradients are finite (the query's and key's zero). The state's share of
    dropout's centred D, its denominator times its centred mean, must not overflow before its division by
    the total."""
    words = first_seed(0.5, 1, lambda kept: bool(kept[0]))
    query = jnp.zeros((1, 1, 1, 8), jnp.float32)
    key = jnp.zeros((1, 1, 1, 8), jnp.float32)
    value = jnp.zeros((1, 1, 1, 8), jnp.float32).at[..., 0].set(2.0 ** 100)
    dout = jnp.zeros((1, 1, 1, 8), jnp.float32).at[..., 0].set(1)
    state = template.Initial(jnp.zeros((1, 1, 1)), jnp.full((1, 1, 1), 2.0 ** 100), jnp.zeros((1, 1, 1, 8)))
    spans = template.spans(1, 1, 1)
    hooks = template.Hooks(initial=state, drop=dropout_of(words, 1, 1, 0.5))

    def reference(query, key, value):
        return template.reference(query, key, value, spans, hooks)

    _, pull = jax.vjp(reference, query, key, value)
    expected = pull(dout)
    np.testing.assert_array_equal(np.asarray(expected[0]), 0)
    np.testing.assert_array_equal(np.asarray(expected[1]), 0)
    assert bool(np.all(np.isfinite(np.asarray(expected[2]))))
    grads = attention_portable.backward(query, key, value, spans, hooks, dout, block_q=8, block_k=8,
                                        pad=attention_triton.BLOCK, accurate=accurate)
    # The query's and key's gradients are their score cotangents times a zero key and query: exactly zero
    # where those are finite.
    np.testing.assert_array_equal(np.asarray(grads.query), 0)
    np.testing.assert_array_equal(np.asarray(grads.key), 0)
    assert bool(np.all(np.isfinite(np.asarray(grads.value))))


def test_the_reference_scales_a_dropped_weight_at_float64_precision():
    """One key, kept at a rate of 1/4, in float64: its output is 1 / (1 - 1/4) = 4/3 exactly. The
    reference rounded the multiplier to fp32 first (1.3333333730697632) and widened it into the truth."""
    with jax.enable_x64(new_val=True):
        words = first_seed(RATE, 1, lambda kept: bool(kept[0]))
        query = jnp.zeros((1, 1, 1, 8), jnp.float64)
        key = jnp.zeros((1, 1, 1, 8), jnp.float64)
        value = jnp.ones((1, 1, 1, 8), jnp.float64)
        out = reference_attention(query, key, value, dropout_rate=RATE, dropout_seed=words)
        np.testing.assert_array_equal(np.asarray(out), np.full(out.shape, 4 / 3))


def test_the_template_reference_scales_a_dropped_weight_at_float64_precision():
    """The one-key witness of the reference, on the template's reference in float64: 4/3 exactly."""
    with jax.enable_x64(new_val=True):
        words = first_seed(RATE, 1, lambda kept: bool(kept[0]))
        query = jnp.zeros((1, 1, 1, 8), jnp.float64)
        key = jnp.zeros((1, 1, 1, 8), jnp.float64)
        value = jnp.ones((1, 1, 1, 8), jnp.float64)
        out = template.reference(query, key, value, template.spans(1, 1, 1), hooked(words, 1, 1))
        np.testing.assert_array_equal(np.asarray(out), np.full(np.asarray(out).shape, 4 / 3))
