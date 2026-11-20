"""Attention's probability dropout: the template's weights hook, on every route, and the selector's Route
for a dropping call. A dropping call on the template is on par with the reference path on the same draw,
forward and backward, both measured from float64 (check 1). The draw is one hash of the call's key and each
entry's global batch row, query head, query and key, so every mainloop and the reference keep the same
entries; the overflow witnesses stay finite where the reference does, and the reference's draw is unbiased
in bf16.
"""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import interprets
from reference_error import FEWEST_SEEDS, assert_as_exact
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
    case_keep,
    drawing as value_drawing,
    geometry,
)

from dew.nn.attention import reference_attention, scaled_dot_product_attention
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
    """`case`'s reference path over its dense mask `case_keep(case)`, dropping at `RATE` from the call's
    key, with the score hooks `names` (`sinks`, `softcap`); a query that sees no key reads zero, as every
    attention call reads (`attention_kernel`)."""
    mask = np.asarray(case_keep(case))
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


def test_a_dropping_call_runs_and_a_chosen_template_runs_it():
    """A dropping call runs, and `generation.chosen(attention='template')` runs it on the template, which
    takes it (bible §6: every kernel honours `generation.chosen()`)."""
    query = jnp.zeros((ROWS, QUERIES, HEADS, WIDTH), jnp.bfloat16)
    key = jnp.zeros((ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
    rng = jax.random.PRNGKey(0)

    def dropping_call():
        return scaled_dot_product_attention(query, key, key, dropout_rate=RATE, dropout_rng=rng,
                                            deterministic=False)

    assert dropping_call().shape == (ROWS, QUERIES, HEADS, WIDTH)
    with generation.chosen(attention="template"), generation.routes() as chosen:
        out = dropping_call()
    # The chosen kernel's contract: the call reports, and so runs, the kernel it was given.
    (named,) = [route for route in chosen if route.choice == "attention"]
    assert named.kernel == "template"
    assert out.shape == (ROWS, QUERIES, HEADS, WIDTH)


def test_a_dropping_decode_step_drops_as_the_reference_path_does():
    """A dropping call shaped as a decode step (one query over key lengths), which the decode kernel, reading
    no draw, does not take: 'auto' reads the lengths and drops as the reference path does over drawn calls
    (check 1), whatever kernel runs it."""
    lengths = jnp.full((ROWS,), KEYS - 3, jnp.int32)

    def draw(seed):
        keys = jax.random.split(jax.random.PRNGKey(seed), 3)
        return (jax.random.normal(keys[0], (ROWS, 1, HEADS, WIDTH), jnp.bfloat16),
                jax.random.normal(keys[1], (ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16),
                jax.random.normal(keys[2], (ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16))

    def step(implementation):
        return jax.jit(lambda q, k, v: scaled_dot_product_attention(
            q, k, v, key_value_seq_lengths=lengths, dropout_rate=RATE, dropout_rng=jax.random.PRNGKey(0),
            deterministic=False, implementation=implementation))

    assert_as_exact(step("auto"), step("reference"), draw, "dropping decode step", seeds=FEWEST_SEEDS)


def test_a_dropping_call_at_eval_takes_no_dropout():
    """A call that is not training (`deterministic=True`) takes no dropout whatever its rate, on every
    route: each route's output is its own undropped output, both in one program, bit for bit (ledger
    3178)."""
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    query = jax.random.normal(keys[0], (ROWS, QUERIES, HEADS, WIDTH), jnp.bfloat16)
    key = jax.random.normal(keys[1], (ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
    value = jax.random.normal(keys[2], (ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
    for implementation in ("reference", "template"):
        out, plain = jax.jit(lambda query, key, value, implementation=implementation: (
            scaled_dot_product_attention(query, key, value, implementation=implementation,
                                         dropout_rate=RATE, dropout_rng=jax.random.PRNGKey(0),
                                         deterministic=True),
            scaled_dot_product_attention(query, key, value, implementation=implementation)))(
                query, key, value)
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
    out = attention_portable.xla_mainloop(attention_portable.Tiles(block[0], block[1], 8), query, key, value,
                                          spans, hooked(words, batch, heads))
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
    dew = attention_portable.xla_mainloop(attention_portable.Tiles(8, 8, 8), query, key, value, spans,
                                          hooked(words, 1, 1, 0.5))
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
    dew = attention_portable.xla_mainloop(attention_portable.Tiles(8, 8, 8), query, key, value, spans,
                                          hooked(words, 1, 1, 0.95))
    ref = reference_attention(query, key, value, dropout_rate=0.95, dropout_seed=words)
    assert bool(jnp.all(jnp.isfinite(ref)))
    assert bool(jnp.all(jnp.isfinite(dew)))


@pytest.mark.parametrize("forms", ["fast", "accurate"])
def test_a_near_total_fp16_draw_keeps_its_one_kept_weight_finite_on_every_path(forms):
    """The fp16 witness: a rate of 1 - 2^-16 (a kept weight's multiplier z = 2^16), exactly one of two
    equal-logit keys kept, values 2^-10. Its unnormalized weight times z is past fp16's largest, its
    normalized weight of one half times z is not: the template's reference, which narrows the normalized
    coefficient, gives 2^5 = 32 exactly; the portable mainloop in either form and the public reference path
    give finite outputs, as the reference does."""
    rate = 1.0 - 2.0 ** -16
    words = first_seed(rate, 2, lambda kept: int(kept.sum()) == 1)
    query = jnp.zeros((1, 1, 1, 8), jnp.float16)
    key = jnp.zeros((1, 2, 1, 8), jnp.float16)
    value = jnp.full((1, 2, 1, 8), 2.0 ** -10, jnp.float16)
    spans, hooks = template.spans(1, 2, 1), hooked(words, 1, 1, rate)
    narrowed = template.reference(query, key, value, spans, hooks)
    np.testing.assert_array_equal(np.asarray(narrowed, np.float32), np.full((1, 1, 1, 8), 32.0, np.float32))
    with generation.run_precision(dataclasses.replace(generation.numerics(), forms=forms)):
        walked = attention_portable.xla_mainloop(attention_portable.Tiles(8, 8, 8), query, key, value, spans,
                                                 hooks)
    public = reference_attention(query, key, value, dropout_rate=rate, dropout_seed=words)
    assert bool(jnp.all(jnp.isfinite(walked))) and bool(jnp.all(jnp.isfinite(public)))


def test_the_mosaic_dropping_walk_refuses_fp16_weights_past_fp16s_largest():
    """At the fp16 witness's rate, 1 - 2^-16, a kept weight's multiplier, 2^16, passes fp16's largest, which
    the Mosaic dropping backward's dV sums in place take no scaled part of (`attention.drops_in_place`): its
    legality rule refuses an fp16 call there, forward and backward, and takes a bf16 one and an fp16 one at an
    ordinary rate (the plain-JAX mainloop takes the witness, above); it refuses an fp16 call from a state
    whose multiplier is no power of two and takes a bf16 one and an fp16 one at a rate of 1/2."""
    near_total = hooked(seed_words(0), ROWS, HEADS, 1.0 - 2.0 ** -16)

    def refusal(dtype, hooks, backward):
        return attention_mosaic.refused(dtype, WIDTH, WIDTH, hooks, backward=backward,
                                        family=mosaic_sm90.WGMMA)

    # A state beside an fp16 walk, which has no headroom, holds the multiplier itself (`sums_units`), each
    # tile's products taken down to it, which sums accumulated in place do not take.
    state = template.Initial(jnp.zeros((ROWS, HEADS, 1)), jnp.ones((ROWS, HEADS, 1)),
                             jnp.zeros((ROWS, HEADS, 1, WIDTH)))
    resumed = dataclasses.replace(hooked(seed_words(0), ROWS, HEADS), initial=state)
    halved = dataclasses.replace(hooked(seed_words(0), ROWS, HEADS, 0.5), initial=state)
    for backward in (False, True):
        assert refusal(jnp.float16, near_total, backward).code == "dropout_rate"
        assert refusal(jnp.bfloat16, near_total, backward) is None
        assert refusal(jnp.float16, hooked(seed_words(0), ROWS, HEADS), backward) is None
        assert refusal(jnp.float16, resumed, backward).code == "dropout_rate"
        assert refusal(jnp.bfloat16, resumed, backward) is None
        assert refusal(jnp.float16, halved, backward) is None


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
    grads = attention_portable.backward(attention_portable.Tiles(8, 8, attention_triton.BLOCK), query, key,
                                        value, template.spans(3, 2, ROWS), hooked(words, ROWS, HEADS), dout)
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


def a_dropping_route_call_on_a_gpu_runs_finite():
    """On a GPU, a direct route call that drops runs, finite, on a mainloop that draws it (the Pallas-Triton
    mainloop's legality rule refuses it, `attention_triton.refused`, ledger row 1521)."""
    from dew.nn.kernels.templates import attention_route

    query, key, value = (jnp.ones((ROWS, 8, HEADS, WIDTH), jnp.bfloat16) for _ in range(3))
    call = functools.partial(attention_route.attend, query, key, value, template.spans(8, 8, ROWS),
                             template.Hooks(drop=dropout_of(seed_words(0), ROWS, HEADS)))
    assert bool(jnp.all(jnp.isfinite(call())))


def test_the_triton_mainloop_refuses_a_dropping_call_at_every_entry():
    """The Pallas-Triton mainloop takes no probability dropout: each entry point, the forward and the backward
    called directly, refuses such a call before any launch rather than run it undropped, and so does its
    legality rule (ledger row 1521)."""
    query, key, value = (jnp.ones((ROWS, 8, HEADS, WIDTH), jnp.bfloat16) for _ in range(3))
    spans = template.spans(8, 8, ROWS)
    hooks = hooked(seed_words(0), ROWS, HEADS)
    stats = jnp.zeros((ROWS, HEADS, 8), jnp.float32)
    residuals = (stats, stats, jnp.zeros((ROWS, HEADS, 8), jnp.int32),
                 template.Split(jnp.zeros((ROWS, HEADS, 8, WIDTH), jnp.float32), stats, stats, stats))
    assert attention_triton.refused(spans, hooks) is not None
    with pytest.raises(NotImplementedError):
        attention_triton.forward(query, key, value, spans, hooks, interpret=True)
    with pytest.raises(NotImplementedError):
        attention_triton.backward(query, key, value, spans, hooks, query, residuals=residuals, interpret=True)


@pytest.mark.parametrize("sliced", ["queries-and-keys", "keys"])
@pytest.mark.parametrize("training", [template.Kept.NONE, template.Kept.SPLIT], ids=["inference", "training"])
def test_the_mosaic_walk_drops_a_slice_at_its_global_places(training, sliced):
    """A Mosaic call drawing at a sequence exchange slice's global places (`template.DropPlaces`: its keys the
    last 32 of 64, and its queries the second eight of sixteen, or its own where it holds them whole, a
    gathered-keys exchange's), interpreted for an H100, drops exactly the entries `dropout_keep` keeps there
    at a rate of 1/2, its launches' tiles past the call's eight queries reading padded places. Its logits all
    zero and its values the identity, each output entry is its query's and key's multiplier over the keys'
    count, 2 / 32 kept and 0 dropped, exact in bf16; under training the first key is each row's anchor,
    whose multiplier the walk takes after it (`attention.anchor_multipliers`), and the backward under a
    cotangent of the identity gives each value's gradient the same entries transposed (the dK/dV launch's
    own draw)."""
    queries, keys, rate = 8, WIDTH, 0.5
    rows, heads = jnp.arange(ROWS, dtype=jnp.int32), jnp.arange(HEADS, dtype=jnp.int32)
    sliced_queries = None if sliced == "keys" else jnp.arange(8, 16, dtype=jnp.int32)
    places = template.DropPlaces(rows, heads, sliced_queries, jnp.arange(32, 32 + keys, dtype=jnp.int32))
    words = seed_words(0)
    hooks = template.Hooks(drop=template.Dropout(words, rows, heads, rate, places.queries, places.keys))
    spans = template.spans(queries, keys, ROWS)
    query = jnp.zeros((ROWS, queries, HEADS, WIDTH), jnp.bfloat16)
    key = jnp.zeros((ROWS, keys, HEADS, WIDTH), jnp.bfloat16)
    value = jnp.broadcast_to(jnp.eye(keys, WIDTH, dtype=jnp.bfloat16)[None, :, None], key.shape)
    kept = np.asarray(dropout_keep(words, rate, (ROWS, HEADS, queries, keys), places))
    expected = np.where(kept, 1 / (1 - rate) / keys, 0.0).astype(np.float32)
    dout = jnp.broadcast_to(jnp.eye(queries, WIDTH, dtype=jnp.bfloat16)[None, :, None], query.shape)
    with generation.assumed("sm90", interpreted=True):
        found = attention_mosaic.forward(query, key, value, spans, hooks, residuals=training, interpret=True)
        trained = training != template.Kept.NONE
        out = found[0] if trained else found
        grads = (attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=found[1:],
                                           interpret=True) if trained else None)
    np.testing.assert_array_equal(np.asarray(out, np.float32).transpose(0, 2, 1, 3), expected)
    if grads is not None:
        value_grads = np.asarray(grads.value, np.float32)[..., :queries]
        np.testing.assert_array_equal(value_grads.transpose(0, 2, 3, 1), expected)


def _mosaic_gradients(spans, hooks):
    """A training Mosaic call's gradients interpreted for an H100 (`attention_mosaic.backward` from its
    split forward's residuals, `D` from them, `attention_gradient.split_delta`) and the reference's
    (`template.reference` under `jax.vjp`), of `(query, key, value, dout)` at scale 1, for check 1."""
    def ours(query, key, value, dout):
        with generation.assumed("sm90", interpreted=True):
            _, *residuals = attention_mosaic.forward(query, key, value, spans, hooks, scale=1.0,
                                                     residuals=template.Kept.SPLIT, interpret=True)
            grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                              scale=1.0, interpret=True)
        return grads.query, grads.key, grads.value

    def reference(query, key, value, dout):
        _, pull = jax.vjp(lambda *qkv: template.reference(*qkv, spans, hooks, scale=1.0), query, key, value)
        return pull(dout)

    return ours, reference


def test_a_mosaic_dropping_backwards_tiny_split_mean_keeps_its_query_gradient():
    """A bf16 Mosaic call interpreted for an H100, with logits 0 and -22, values 0 and 2^-126, dO 2^73,
    and dropout at rate 1 - 2^-53 keeping only the second key. Its split mean is below fp32's least
    subnormal once the power of 2^53 comes out, so D contracts it with dO in the power its sums hold
    (`attention_gradient.split_delta`): dQ's second coordinate is about -2.79e-10, where it was 0
    (the base review of eb3ab2670, F1). The -22 logit is a sum of powers of two so the gradients'
    products do not add a second rounding to the fast form's bf16 score cotangents."""
    rate = 1.0 - 2.0 ** -53
    words = jnp.asarray([3181759654, 0], jnp.uint32)
    hooks, spans = hooked(words, 1, 1, rate), template.spans(1, 2, 1)
    query = jnp.zeros((1, 1, 1, WIDTH), jnp.bfloat16).at[..., 0].set(1).at[..., 2:4].set(1)
    key = jnp.zeros((1, 2, 1, WIDTH), jnp.bfloat16).at[0, 0, 0, 1].set(1).at[0, 1, 0, 0].set(-16)
    key = key.at[0, 1, 0, 2].set(-4).at[0, 1, 0, 3].set(-2)
    value = jnp.zeros((1, 2, 1, WIDTH), jnp.bfloat16).at[0, 1, 0, 0].set(2.0 ** -126)
    dout = jnp.zeros((1, 1, 1, WIDTH), jnp.bfloat16).at[..., 0].set(2.0 ** 73)
    ours, reference = _mosaic_gradients(spans, hooks)
    assert_as_exact(ours, reference, lambda seed: (query, key, value, dout),
                    "a dropping split mean below fp32's least subnormal", seeds=FEWEST_SEEDS)


def test_a_mosaic_dropping_backwards_kept_anchor_keeps_its_tiny_product():
    """A bf16 Mosaic call interpreted for an H100 at logits -0.5 and 0, dropout at rate 1 - 2^-53 keeping
    only the anchor, whose value is 2^-126 under a dO of 2^-40. Its product 2^-166 needs the power of 2^53
    before the contraction (`attention_gradient.split_delta`), or dQ's second coordinate is -6.86e-36
    instead of +1.13e-35 (the base review of 6a57b66de, F1). In the score cotangents the anchor's
    self-difference is exactly zero even though the tile's unscaled product rounds to zero, or the
    anchor's key gradient has the wrong sign. Both gradients are on par with the reference (check 1)."""
    words = jnp.asarray([3181759654, 0], jnp.uint32)
    hooks, spans = hooked(words, 1, 1, 1.0 - 2.0 ** -53), template.spans(1, 2, 1)
    query = jnp.zeros((1, 1, 1, WIDTH), jnp.bfloat16).at[..., 1].set(1)
    key = jnp.zeros((1, 2, 1, WIDTH), jnp.bfloat16).at[0, 0, 0, 1].set(-0.5)
    value = jnp.zeros((1, 2, 1, WIDTH), jnp.bfloat16).at[0, 1, 0, 0].set(2.0 ** -126)
    dout = jnp.zeros((1, 1, 1, WIDTH), jnp.bfloat16).at[..., 0].set(2.0 ** -40)
    ours, reference = _mosaic_gradients(spans, hooks)
    assert_as_exact(ours, reference, lambda seed: (query, key, value, dout),
                    "a dropping kept anchor's tiny product", seeds=FEWEST_SEEDS)


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
    grads = attention_portable.backward(attention_portable.Tiles(8, 8, attention_triton.BLOCK), query, key,
                                        value, template.spans(1, 3, 1), hooked(words, 1, 1, 0.5), dout)
    np.testing.assert_array_equal(np.asarray(grads.query, np.float32), np.zeros(query.shape, np.float32))


def test_finite_products_whose_raw_sum_overflows_keep_a_finite_query_gradient_on_the_portable_mainloop():
    """Three keys whose dropped products are 0, 2^127 and 2^127: their raw sum, 2^128, is past fp32's
    largest, where the reference's weighted mean is 2^128 / 3. The query gradient must be finite, too."""
    words = first_seed(0.5, 3, lambda kept: bool(kept.all()))
    query, key, value, dout = overflowing_keys()
    dq = reference_query_gradient(query, key, value, dout, words, 0.5)
    assert bool(np.all(np.isfinite(np.asarray(dq, np.float32))))
    grads = attention_portable.backward(attention_portable.Tiles(8, 8, attention_triton.BLOCK), query, key,
                                        value, template.spans(1, 3, 1), hooked(words, 1, 1, 0.5), dout)
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
    grads = attention_portable.backward(attention_portable.Tiles(8, 8, attention_triton.BLOCK), query, key,
                                        value, spans, hooks, dout, accurate=accurate)
    # The query's and key's gradients are their score cotangents times a zero key and query: exactly zero
    # where those are finite.
    np.testing.assert_array_equal(np.asarray(grads.query), 0)
    np.testing.assert_array_equal(np.asarray(grads.key), 0)
    assert bool(np.all(np.isfinite(np.asarray(grads.value))))


WALKS = ("portable", *(("mosaic",) if interprets() else ()))
"""The forwards a dropping edge's test walks: the plain-JAX mainloop's at tiles of 8, and in a CPU process the
Mosaic forward's interpreted as an H100 runs it, its split join's where it trains."""


def _walked(spans, hooks, training: bool, mainloop: str = "portable", **options) -> Callable:
    """`mainloop`'s call over `spans` under `hooks` (`WALKS`; its training walk's where `training`), its
    output alone, a function of `(query, key, value)` for check 1."""
    def walked(query, key, value):
        if mainloop == "portable":
            out = attention_portable.xla_mainloop(attention_portable.Tiles(8, 8, 8), query, key, value, spans,
                                                  hooks, residuals=training, **options)
        else:
            kept = template.Kept.SPLIT if training else template.Kept.NONE
            with generation.assumed("sm90"):
                out = attention_mosaic.forward(query, key, value, spans, hooks, residuals=kept,
                                               interpret=True, **options)
        return out[0] if training else out

    return walked


@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize(("rate", "logit", "seed"),
                         [(1.0 - 2.0 ** -14, -18.0, None), (127 / 256, -17.34375, None),
                          (2 / 3, -18.25, None), (1.0 - 2.0 ** -17, -28.0, 355564)],
                         ids=["near-total", "below-two", "below-four", "past-fp16s-binade"])
def test_an_fp16_draw_keeps_a_small_kept_weight_the_reference_keeps(rate, logit, seed, training):
    """Two keys at logits 0 and `logit`, the first dropped and the second kept, values 0 and 2^15, in fp16:
    the kept weight is below fp16's least subnormal (or rounds to 0 at it) where its weight times the draw's
    multiplier z does not, so the output is z e^logit 2^15 / (1 + e^logit), as the reference's narrowed
    coefficient gives it, on par with the reference (check 1) on the inference walk and the training walk
    alike. Narrowed without z, e^-18 at z = 2^14 flushed to 0 (the base review of 56a716c95, F1); taken by
    the power of two below z, e^-17.34 at z = 256/129 and e^-18.25 at z = 3 did too (the base reviews of
    4e2eb287e and b872d1849, F1), and e^-28 at z = 2^17 taken by fp16's 2^15 alone (of a648b3279, F3). A
    rate whose kept key no first `2^16` draws keep takes its `seed`."""
    words = (first_seed(rate, 2, lambda kept: not bool(kept[0]) and bool(kept[1])) if seed is None
             else jnp.asarray([seed, 0], jnp.uint32))
    query = jnp.zeros((1, 1, 1, 8), jnp.float16).at[..., 0].set(1)
    key = jnp.zeros((1, 2, 1, 8), jnp.float16).at[0, 1, 0, 0].set(logit)
    value = jnp.zeros((1, 2, 1, 8), jnp.float16).at[0, 1, 0, 0].set(2.0 ** 15)
    spans, hooks = template.spans(1, 2, 1), hooked(words, 1, 1, rate)

    walked = _walked(spans, hooks, training, scale=1.0)
    assert_as_exact(walked, lambda query, key, value: template.reference(query, key, value, spans, hooks,
                                                                         scale=1.0),
                    lambda seed: (query, key, value), "a small kept fp16 weight", seeds=FEWEST_SEEDS)


@pytest.mark.parametrize("mainloop", WALKS)
@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize(("weight", "multiplier"), [(0.4999, 3.00085), (0.5002, 2.9998)],
                         ids=["rounded-at-its-reach", "rounded-with-its-multiplier"])
def test_an_fp16_kept_weight_near_fp16s_largest_keeps_its_output_finite(weight, multiplier, training,
                                                                        mainloop):
    """Two fp16 keys at logits 0 and log(`weight`), the first dropped and the second kept at the multiplier
    z, values 0 and fp16's largest: the output z w 65504 / (1 + w), 65513.82 and 65517.09, is below the
    65520 that rounds to inf, finite and on par with the reference (check 1) on both walks of each mainloop
    (`WALKS`), where one fp16
    rounding of the kept weight put it past: at the multiplier's reach of 4, 4w to 2 (the base review of
    70c1aa096, F1); with the multiplier, z w to 1.500977 (row 9505). The logit is two fp16 entries of the
    key, their sum in fp32 log(`weight`)."""
    rate = 1.0 - 1.0 / multiplier
    words = first_seed(rate, 2, lambda kept: not bool(kept[0]) and bool(kept[1]))
    logit = np.float32(np.log(weight))
    high = np.float16(logit)
    query = jnp.zeros((1, 1, 1, 8), jnp.float16).at[..., :2].set(1)
    key = (jnp.zeros((1, 2, 1, 8), jnp.float16).at[0, 1, 0, 0].set(high)
           .at[0, 1, 0, 1].set(np.float16(logit - np.float32(high))))
    value = jnp.zeros((1, 2, 1, 8), jnp.float16).at[0, 1].set(jnp.finfo(jnp.float16).max)
    spans, hooks = template.spans(1, 2, 1), hooked(words, 1, 1, rate)

    walked = _walked(spans, hooks, training, mainloop, scale=1.0)
    assert_as_exact(walked, lambda query, key, value: template.reference(query, key, value, spans, hooks,
                                                                         scale=1.0),
                    lambda seed: (query, key, value), "an fp16 kept weight near fp16's largest",
                    seeds=FEWEST_SEEDS)


@pytest.mark.parametrize("mainloop", WALKS)
@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
def test_an_fp16_output_whose_fp32_quotient_is_its_overflow_midpoint_narrows_below_it(training, mainloop):
    """One fp16 key at logit 0 and value 65504, kept (`first_seed`) at a rate of 1/4095 - 2^-27: the output,
    65504 times the multiplier, 65519.9995, is below fp16's overflow midpoint, 65520, so it narrows to
    65504 on both walks of each mainloop (`WALKS`: the portable finish's, the Mosaic forward's register
    epilogue and its split join's), on par with the reference (check 1), where its fp32 quotient rounded
    onto 65520 and tied to inf (the base review of 545370f65, F3; row 9544; of dc97cee6a, F6: a helper's
    test alone left the routes unwired)."""
    rate = 1.0 / 4095 - 2.0 ** -27
    query, key = jnp.zeros((1, 1, 1, 8), jnp.float16), jnp.zeros((1, 1, 1, 8), jnp.float16)
    value = jnp.full((1, 1, 1, 8), jnp.finfo(jnp.float16).max, jnp.float16)
    words = first_seed(rate, 1, lambda kept: bool(kept[0]))
    spans, hooks = template.spans(1, 1, 1), hooked(words, 1, 1, rate)
    assert_as_exact(_walked(spans, hooks, training, mainloop),
                    lambda query, key, value: template.reference(query, key, value, spans, hooks),
                    lambda seed: (query, key, value), "an fp16 output at its fp32 overflow midpoint",
                    seeds=FEWEST_SEEDS)


def _anchor_below_least_subnormal():
    """`test_a_kept_anchor_below_fp16s_least_subnormal_keeps_its_term`'s call: its spans, hooks and operands
    `(query, key, value)`."""
    rate = 1.0 - 2.0 ** -14
    words = first_seed(rate, 2, lambda kept: not bool(kept[0]) and bool(kept[1]))
    query = jnp.zeros((1, 1, 1, 8), jnp.float16).at[..., 0].set(1)
    key = jnp.zeros((1, 2, 1, 8), jnp.float16).at[0, 0, 0, 0].set(-1)
    value = jnp.zeros((1, 2, 1, 8), jnp.float16).at[0, 1].set(2.0 ** 15)
    spans = template.spans(1, 2, 1)
    hooks = template.Hooks(initial=template.sinks(jnp.asarray([18.0], jnp.float16), 1, 1, 8),
                           drop=dropout_of(words, 1, 1, rate))
    return spans, hooks, (query, key, value)


@pytest.mark.parametrize("mainloop", WALKS)
@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
def test_a_kept_anchor_below_fp16s_least_subnormal_keeps_its_term(training, mainloop):
    """One fp16 query over a sink at 18 and two keys at logits -1 and 0, values 0 and 2^15, at a rate of
    1 - 2^-14 under a draw dropping the first key and keeping the second, the row's anchor: its weight e^-18
    is below fp16's least subnormal, its term times the multiplier 2^14 is not, so the output is about
    8.18, on par with the reference (check 1) on both walks of each mainloop (`WALKS`), where the training
    walk's split sums rounded the anchor's own weight to fp16 before the multiplier took it, 0 (the base
    review of 70c1aa096, F2; row 9506)."""
    spans, hooks, operands = _anchor_below_least_subnormal()
    walked = _walked(spans, hooks, training, mainloop, scale=1.0)
    assert_as_exact(walked, lambda query, key, value: template.reference(query, key, value, spans, hooks,
                                                                         scale=1.0),
                    lambda seed: operands, "a kept anchor below fp16's least subnormal", seeds=FEWEST_SEEDS)


def test_an_fp16_reference_dropping_products_mixed_derivatives_are_its_products():
    """The reference's fp16 dropping product of narrowed weights (`attention.reference`'s wide cotangent) is
    bilinear at every order: one fp16 query over keys at logits 0 and 1, values 0 and v, at a rate of 1/2
    under a draw dropping the first key and keeping the second, whose output is 2 sigmoid(q) v where q is
    the query's logit difference: its mixed second derivative as to the query and the kept value, each order
    of the two, on par with float64's (check 1), where each operand's derivative path stopped the other's
    and took it to zero (the head review of af043f9b1, F1)."""
    rate = 0.5
    words = first_seed(rate, 2, lambda kept: not bool(kept[0]) and bool(kept[1]))
    spans, hooks = template.spans(1, 2, 1), hooked(words, 1, 1, rate)
    key = jnp.zeros((1, 2, 1, 8), jnp.float16).at[0, 1, 0, 0].set(1)

    def output(q, v):
        query = jnp.zeros((1, 1, 1, 8), jnp.float16).at[0, 0, 0, 0].set(q.astype(jnp.float16))
        value = jnp.zeros((1, 2, 1, 8), jnp.float16).at[0, 1, 0, 0].set(v.astype(jnp.float16))
        return template.reference(query, key, value, spans, hooks, scale=1.0)[0, 0, 0, 0].astype(jnp.float32)

    def mixed(q, v):
        by_value_of_query = jax.grad(jax.grad(output, argnums=0), argnums=1)(q, v)
        by_query_of_value = jax.grad(jax.grad(output, argnums=1), argnums=0)(q, v)
        return jnp.stack([by_value_of_query, by_query_of_value])

    def truth(q, v):
        sigmoid = 1 / (1 + jnp.exp(-q))
        return jnp.stack([2 * sigmoid * (1 - sigmoid)] * 2) * jnp.ones_like(v)

    assert_as_exact(mixed, truth, lambda seed: (jnp.float32(0.0), jnp.float32(1.0)),
                    "an fp16 dropping product's mixed derivatives", seeds=FEWEST_SEEDS, truth_fn=truth)


def test_a_kept_anchor_below_fp16s_least_subnormal_keeps_its_gradients():
    """`test_a_kept_anchor_below_fp16s_least_subnormal_keeps_its_term`'s call through the Mosaic backward from
    its forward's split sums (`attention_mosaic_backward`'s D, which reads the anchor's own weight they
    keep), interpreted for an H100, under a cotangent of ones on every output column: its gradients on par
    with the reference's (check 1), where its first pass rounded the anchor's own weight to fp16 as the
    forward's split sums did (the head reviews of c55dff91f, F5, F7, and of 12e76ca7e: the portable
    backward's centred pass reads no split sum; row 9506)."""
    spans, hooks, operands = _anchor_below_least_subnormal()
    ours, reference = _mosaic_gradients(spans, hooks)
    dout = jnp.ones((1, 1, 1, 8), jnp.float16)
    assert_as_exact(ours, reference, lambda seed: (*operands, dout),
                    "a kept anchor below fp16's least subnormal's gradients", seeds=FEWEST_SEEDS)


@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("bits", [32, 48, 53])
def test_a_resumed_states_small_sums_are_kept_under_a_near_total_draw(bits, dtype, training):
    """An fp32 state of maximum 0, denominator 1 and sums 2^-120 resumed over one key of logit 0 and value 0
    of `dtype` at a rate of `1 - 2^-bits`, under an output hook taking it by 2^125, whatever the draw: the
    output is the state's sums over the joint denominator 2 by it, exactly 16, where the state's sums taken
    down into the walk's flushed to 0: by the whole multiplier at 2^-32 (the base review of 56a716c95, F2),
    by its rest past a power capped at 2^32 at 2^-53 and past fp16's 2^15 at 2^-48 (of 4e2eb287e and
    b872d1849, F2)."""
    rate = 1.0 - 2.0 ** -bits
    query, key, value = (jnp.zeros((1, 1, 1, 8), dtype) for _ in range(3))
    state = template.Initial(jnp.zeros((1, 1, 1)), jnp.ones((1, 1, 1)), jnp.full((1, 1, 1, 8), 2.0 ** -120))
    hooks = template.Hooks(initial=state, output_mod=lambda out: out * 2.0 ** 125,
                           drop=dropout_of(seed_words(0), 1, 1, rate))
    walked = attention_portable.xla_mainloop(attention_portable.Tiles(8, 8, 8), query, key, value,
                                             template.spans(1, 1, 1), hooks, residuals=training)
    out = walked[0] if training else walked
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.full(out.shape, 16.0, np.float32))


@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
def test_a_kept_fp32_key_near_fp32s_largest_keeps_its_finite_output(training):
    """One fp32 query over one kept key of value 2^127 at a rate of 1/4: the output is 2^127 4/3, finite,
    on par with the reference (check 1), where the quotient taken before the multiplier's rest was 2^128
    (the base review of a648b3279, F1)."""
    words = first_seed(0.25, 1, lambda kept: bool(kept[0]))
    query, key = (jnp.zeros((1, 1, 1, 8), jnp.float32) for _ in range(2))
    value = jnp.full((1, 1, 1, 8), 2.0 ** 127, jnp.float32)
    spans, hooks = template.spans(1, 1, 1), hooked(words, 1, 1, 0.25)

    walked = _walked(spans, hooks, training)
    assert_as_exact(walked, lambda query, key, value: template.reference(query, key, value, spans, hooks),
                    lambda seed: (query, key, value), "a kept key near fp32's largest", seeds=FEWEST_SEEDS)


@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
def test_a_large_state_beside_an_fp16_walk_keeps_its_finite_output(training):
    """An fp32 state of maximum 0, denominator 2^120 and sums 3 2^126 resumed over one fp16 key of logit 0
    and value 0 at a rate of 1/4, whatever the draw: the output is the state's sums over the joint
    denominator, 192, where the state's sums taken up into the walk's units by 1.5 overflowed (the base
    review of a648b3279, F2)."""
    query, key, value = (jnp.zeros((1, 1, 1, 8), jnp.float16) for _ in range(3))
    state = template.Initial(jnp.zeros((1, 1, 1)), jnp.full((1, 1, 1), 2.0 ** 120),
                             jnp.full((1, 1, 1, 8), 3 * 2.0 ** 126))
    hooks = template.Hooks(initial=state, drop=dropout_of(seed_words(0), 1, 1, 0.25))
    walked = attention_portable.xla_mainloop(attention_portable.Tiles(8, 8, 8), query, key, value,
                                             template.spans(1, 1, 1), hooks, residuals=training)
    out = walked[0] if training else walked
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.full(out.shape, 192.0, np.float32))


def _portable_gradients(spans, hooks, scale: float = 1.0):
    """The portable backward's gradients of a call (`attention_portable.backward`) and the reference's
    (`template.reference` under `jax.vjp`), of `(query, key, value, dout)`, for check 1."""
    def ours(query, key, value, dout):
        tiles = attention_portable.Tiles(8, 8, attention_triton.BLOCK)
        grads = attention_portable.backward(tiles, query, key, value, spans, hooks, dout, scale=scale)
        return grads.query, grads.key, grads.value

    def reference(query, key, value, dout):
        _, pull = jax.vjp(lambda *qkv: template.reference(*qkv, spans, hooks, scale=scale), query, key, value)
        return pull(dout)

    return ours, reference


def test_a_dropping_backward_of_opposite_products_near_fp32s_largest_is_finite():
    """One fp32 query over two kept keys at logits 0 and -1.25, rate 1/2, their dropped products 5 2^125 and
    -5 2^125: each centred about the other passes fp32's largest, where dQ is about 9.2e37. Held at a quarter
    and taken back after their weight (`attention_backward_algebra.softmax_cotangents`), the gradients are on
    par with the reference's (check 1), where they were not finite (the base review of e2b671ce9, F1)."""
    words = first_seed(0.5, 2, lambda kept: bool(kept[0]) and bool(kept[1]))
    query = jnp.zeros((1, 1, 1, 8), jnp.float32).at[..., 0].set(1)
    key = jnp.zeros((1, 2, 1, 8), jnp.float32).at[0, 1, 0, 0].set(-1.25)
    value = jnp.zeros((1, 2, 1, 8), jnp.float32).at[0, 0, 0, 1].set(5 * 2.0 ** 124).at[0, 1, 0, 1].set(
        -5 * 2.0 ** 124)
    dout = jnp.zeros((1, 1, 1, 8), jnp.float32).at[..., 1].set(1)
    ours, reference = _portable_gradients(template.spans(1, 2, 1), hooked(words, 1, 1, 0.5))
    assert_as_exact(ours, reference, lambda seed: (query, key, value, dout),
                    "dropped products about one another near fp32's largest", seeds=FEWEST_SEEDS)


def test_a_dropping_backwards_split_mean_past_fp32s_largest_before_its_power_is_finite():
    """A bf16 row of four keys at equal logits, rate 1/4, three kept, values 3 2^126, dO 2^-125: the forward's
    split sums at the multiplier's reach of 2, over their total of 4, pass fp32's largest before the reach
    comes out, where D is 6. Taken out in one division with the total (`attention_gradient.split_delta`),
    the gradients of a Mosaic call interpreted for an H100, whose backward takes D from them, are on par
    with the reference's (check 1), where D was not finite (the base review of e2b671ce9, F2; the portable
    backward takes its own pass, of 6a57b66de, F2)."""
    words = first_seed(0.25, 4, lambda kept: int(np.sum(kept)) == 3)
    query = jnp.zeros((1, 1, 1, WIDTH), jnp.bfloat16).at[..., 1].set(1)
    key = jnp.zeros((1, 4, 1, WIDTH), jnp.bfloat16)
    value = jnp.zeros((1, 4, 1, WIDTH), jnp.bfloat16).at[:, :, :, 0].set(3 * 2.0 ** 126)
    dout = jnp.zeros((1, 1, 1, WIDTH), jnp.bfloat16).at[..., 0].set(2.0 ** -125)
    ours, reference = _mosaic_gradients(template.spans(1, 4, 1), hooked(words, 1, 1, 0.25))
    assert_as_exact(ours, reference, lambda seed: (query, key, value, dout),
                    "a dropping split mean past fp32's largest", seeds=FEWEST_SEEDS)


def test_a_dropping_backward_centres_its_d_at_the_rows_heaviest_key():
    """One fp32 query e0 over 16 keys in tiles of 8 at scale 1: key 0 at logit -64 with value 2^64 e0, key 1
    at -65, key 8 at 0 with value e0, the rest at -128 and value 0, dO e0, at a rate of 1/4 keeping keys 0
    and 8. Key 8 holds the row's weight, so D is about its product, 4/3: centred at the first tile's
    heaviest product, 2.46e19, D cancelled to 0 and the query's gradient was about 4/3 where it is -3.94e-9
    (the base review of 4e2eb287e, F4). The portable backward's dQ on par with the reference's (check 1)."""
    rate = 0.25
    words = first_seed(rate, 16, lambda kept: bool(kept[0]) and bool(kept[8]))
    query = jnp.zeros((1, 1, 1, 8), jnp.float32).at[..., 0].set(1)
    key = (jnp.zeros((1, 16, 1, 8), jnp.float32).at[0, :, 0, 0].set(-128).at[0, 0, 0, 0].set(-64)
           .at[0, 1, 0, 0].set(-65).at[0, 8, 0, 0].set(0).at[0, 8, 0, 1].set(1))
    value = jnp.zeros((1, 16, 1, 8), jnp.float32).at[0, 0, 0, 0].set(2.0 ** 64).at[0, 8, 0, 0].set(1)
    dout = jnp.zeros((1, 1, 1, 8), jnp.float32).at[..., 0].set(1)
    spans, hooks = template.spans(1, 16, 1), hooked(words, 1, 1, rate)

    ours, reference = _portable_gradients(spans, hooks)
    assert_as_exact(ours, reference, lambda seed: (query, key, value, dout),
                    "a dropping query gradient about its heaviest key", seeds=FEWEST_SEEDS)


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
