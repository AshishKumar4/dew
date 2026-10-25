"""Models' own dtype and attention fields, and the arithmetic their kernels honor."""

import contextlib
import functools
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_fp32_reduction_bound, equations

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.diffusion_gemma import soft_embeddings
from dew.nn.dsa_kpool import KPoolSparseAttention
from dew.nn.inputs import AttentionMetadata, MixerCall, coordinates
from dew.nn.local_attention import local_attention
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import AttentionMixer
from dew.nn.mla import MultiHeadLatentAttention
from dew.nn.mlp import GatedMLP
from dew.nn.precision import fp32_result_dot_general

BF16_QKV = (1, 4, 2, 8)  # [B, S, H, D]


@pytest.mark.parametrize("arguments", [
    {"precision": jax.lax.Precision.HIGHEST},
    {"precision": jax.lax.DotAlgorithmPreset.F32_F32_F32},
    {"force_fp32_for_softmax": False},
    {"dtype": jnp.float32},
])
def test_auto_takes_the_reference_path_for_arithmetic_only_it_performs(arguments):
    """'auto' resolves a call that asks for a matmul precision, a dot
    algorithm among them, a softmax dtype or a compute dtype no fused kernel
    honours to the reference path, and computes exactly what naming that path
    computes."""
    query, key, value = jax.random.normal(jax.random.key(1), (3, *BF16_QKV), jnp.bfloat16)
    auto = scaled_dot_product_attention(query, key, value, implementation="auto", **arguments)
    reference = scaled_dot_product_attention(query, key, value, implementation="reference",
                                             **arguments)
    assert jnp.array_equal(auto, reference)


PACKED = jnp.asarray([[1] * 6 + [2] * 7 + [0] * 3, [1] * 16])
LAYERS = {
    "window": lambda **policy: CausalTransformer(
        vocab_size=64, num_layers=2, emb_features=32, num_heads=4, num_kv_heads=2,
        max_seq_len=32, layer_types=("sliding", "full_attention"),
        kinds={"sliding": {"window": 4}}, **policy),
    "packed": lambda **policy: CausalTransformer(
        vocab_size=64, num_layers=1, emb_features=32, num_heads=4, num_kv_heads=2,
        max_seq_len=32, **policy),
    "llama4_chunk": lambda **policy: AttentionMixer(rotary_pairs="adjacent", qk_norm_weight=False).build(
        MixerContext(emb_features=32, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=32,
                     attention_chunk=4, qk_norm=True, **policy))(),
    "llama4_global": lambda **policy: AttentionMixer(nope=True, temperature_tuning=(8192.0, 0.1)).build(
        MixerContext(emb_features=32, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=32, qk_norm=False,
                     **policy))(),
    "mla": lambda **policy: MultiHeadLatentAttention(
        emb_features=32, num_heads=4, max_seq_len=32, q_lora_rank=16, kv_lora_rank=8,
        qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, **policy),
    "kpool": lambda **policy: KPoolSparseAttention(
        emb_features=32, num_heads=4, max_seq_len=32, q_lora_rank=8, kv_lora_rank=8,
        qk_nope_head_dim=8, qk_rope_head_dim=0, v_head_dim=8, index_n_heads=2, index_head_dim=8,
        index_topk=4, index_kpool=2, **policy),
}


@pytest.mark.parametrize("arguments", [
    {"precision": "highest"},
    {"force_fp32_for_softmax": False},
])
@pytest.mark.parametrize("layer", sorted(LAYERS))
def test_auto_under_a_window_or_packing_takes_the_reference_path(layer, arguments):
    """A window, a chunk, packed documents or a sparse selection build a
    mask, which a fused kernel takes; the call still resolves to the
    reference path first when only that path computes what it asks for,
    and matches naming that path."""
    if layer == "mla" and "precision" not in arguments:
        pytest.skip("latent attention always runs its softmax in fp32")
    if layer in ("window", "packed"):
        x = jax.random.randint(jax.random.key(0), (2, 16), 0, 64)
    else:
        x = jax.random.normal(jax.random.key(0), (2, 16, 32))
    call = {} if layer == "window" else {
        "segment_ids": PACKED} if layer == "packed" else {
        "call": MixerCall(AttentionMetadata(coordinates=coordinates(segments=PACKED)))}
    if layer == "kpool":
        call = {"call": MixerCall(AttentionMetadata(coordinates=coordinates(valid=PACKED != 0)))}
    variables = LAYERS[layer](**arguments).init(jax.random.key(1), x, **call)
    auto = LAYERS[layer](attention_impl="auto", **arguments).apply(variables, x, **call)
    reference = LAYERS[layer](attention_impl="reference", **arguments).apply(variables, x, **call)
    assert jnp.array_equal(auto, reference)


@pytest.mark.parametrize("span", [{"window": 4}, {"chunk": 4}])
def test_local_attention_resolves_auto_before_the_mask(span):
    query, key, value = jax.random.normal(jax.random.key(1), (3, 2, 64, 4, 8))
    packed = {"segment_ids": jnp.ones((2, 64), jnp.int32), "precision": "highest"}
    auto = local_attention(query, key, value, **span, **packed, implementation="auto")
    reference = local_attention(query, key, value, **span, **packed, implementation="reference")
    assert jnp.array_equal(auto, reference)


@pytest.mark.parametrize("precision", [None, jax.lax.Precision.DEFAULT, "default"])
def test_fused_attention_default_precision_matches_the_attention_equation(precision):
    query, key, value = jax.random.normal(
        jax.random.key(0), (3, *BF16_QKV), dtype=jnp.float32)
    scores = jnp.einsum("bqhd,bkhd->bhqk", query, key) / jnp.sqrt(query.shape[-1])
    expected = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(scores, axis=-1), value)
    actual = scaled_dot_product_attention(
        query, key, value, precision=precision, implementation="xla")
    assert jnp.max(jnp.abs(actual - expected)) < 1e-5


@pytest.mark.parametrize("helper", ["rounded_operand", "rounded_to"])
def test_a_rounding_survives_the_reduction_it_feeds(helper):
    """XLA:CPU's YNNPACK reduce fusion sums the unrounded values of a bare
    `astype` round trip, even with excess precision off (openxla/xla#49978);
    the helpers' optimization barrier keeps the rounding a sum and a mean
    of squares read, as a norm's statistics do."""
    from dew.nn import precision
    x = np.random.default_rng(0).normal(size=(8, 4096)).astype(np.float32) * 3
    rounded = x.astype(jnp.bfloat16).astype(np.float64)
    held = jax.jit(lambda v: getattr(precision, helper)(v, jnp.bfloat16).astype(jnp.float32))
    sums = np.asarray(jax.jit(lambda v: jnp.sum(held(v), axis=-1))(x), np.float64)
    squares = np.asarray(jax.jit(lambda v: jnp.mean(jnp.square(held(v)), axis=-1))(x), np.float64)
    assert_fp32_reduction_bound(sums, rounded.sum(-1), np.abs(rounded).sum(-1), x.shape[-1])
    mean_square = np.mean(rounded ** 2, -1)
    assert_fp32_reduction_bound(squares, mean_square, mean_square, x.shape[-1] + 2)
    # The bound is a worst case (4.8 on these sums), wider than what dropping
    # the rounding moves them (at most 0.96), so each sum must also be nearer
    # the rounded values' than the unrounded ones'.
    assert np.all(np.abs(sums - rounded.sum(-1)) < np.abs(sums - x.astype(np.float64).sum(-1)))


@pytest.mark.parametrize("configured, default", [("F32_F32_F32", False), ("BF16_BF16_F32", False),
                                                  ("TF32_TF32_F32_X3", False), ("tensorfloat32", False),
                                                  ("highest", False), ("bfloat16", True), ("default", True)])
def test_a_configured_dot_algorithm_reads_as_the_precision_it_asks(configured, default):
    """`jax_default_matmul_precision` takes a dot algorithm preset's name as well as a precision's, and
    a product without a precision of its own asks for the default only where the configured one is
    the default."""
    from dew.nn import precision
    with jax.default_matmul_precision(configured):
        assert precision.asks_default_precision(None, configured=True) is default


def test_a_dot_algorithm_reads_as_its_own_precision_past_the_default():
    """A `jax.lax.DotAlgorithm`, which JAX's precision alias takes beside its presets, asks for more
    than the default precision, by its own name, apart from another algorithm's, so attention takes it
    on its reference path. (CPU's dot refuses a raw algorithm, so only its routing is asked here.)"""
    from dew.nn import precision
    wide = jax.lax.DotAlgorithm(jnp.float32, jnp.float32, jnp.float32)
    narrow = jax.lax.DotAlgorithm(jnp.bfloat16, jnp.bfloat16, jnp.float32)
    assert not precision.asks_default_precision(wide)
    assert len(precision.precision_names(wide)) == 1
    assert precision.precision_names(wide) != precision.precision_names(narrow)
    # Attention takes it, as it does a preset, on the path that computes it (the reference one).
    from dew.nn.attention import reference_only
    assert reference_only(jnp.zeros(BF16_QKV), dtype=None, precision=narrow, force_fp32_for_softmax=True)


def test_a_compensated_running_sum_holds_its_value_rounded_in_its_high_half():
    """Chained `compensated_add`s keep the pair renormalised: after -1 and 63 additions of -2^-25,
    each a quarter of a float32 step of the running sum and so lost from a plain one, the high half
    is the exact sum rounded once and the low half what that rounding dropped."""
    from dew.nn import precision
    tokens = np.full(64, -2.0 ** -25, np.float32)
    tokens[0] = -1
    zero = jnp.zeros((), jnp.float32)

    def running(values):
        return jax.lax.fori_loop(0, 64, lambda i, total: precision.compensated_add(total, (values[i], zero)),
                                 (zero, zero))

    high, low = (np.asarray(x) for x in jax.jit(running)(tokens))
    exact = -1 - 63 * 2.0 ** -25
    assert np.cumsum(tokens, dtype=np.float32)[-1] == -1
    assert high == np.float32(exact)
    assert np.float64(high) + np.float64(low) == exact


@pytest.mark.parametrize("bit_split,subtract,barrier", [(False, False, False), (True, False, True),
                                                       (True, True, False)],
                         ids=["projection", "norm", "division"])
@pytest.mark.parametrize("compiled", [False, True])
def test_a_two_float_product_keeps_the_exact_float64_residual(bit_split, subtract, barrier, compiled):
    """Split boundaries, product ties, both signs and 80 binades, with normal residuals.
    Float64 represents each fp32 product exactly; neither half admits any error."""
    from dew.nn.precision import two_product

    mantissas = np.array([0, 1, 0xfff, 0x1000, 0x1001, 0x3fffff, 0x400000, 0x7ffffe, 0x7fffff], np.uint32)
    significands = (np.uint32(127 << 23) | mantissas).view(np.float32)
    values = np.ldexp(significands[:, None], [-40, 0, 40]).reshape(-1)
    rng = np.random.default_rng(411)
    random = np.ldexp(rng.uniform(1, 2, 128).astype(np.float32), rng.integers(-40, 41, 128))
    values = np.concatenate([values, -values, random, -random, np.array([0.0, -0.0], np.float32)])
    # Equal-shaped operands isolate the scalar law from XLA's broadcast fusion;
    # the quantizer's composed division below exercises its broadcast arithmetic.
    left, right = (np.array(x) for x in np.broadcast_arrays(values[:, None], values[None, :]))

    def product(a, b):
        return two_product(a, b, bit_split=bit_split, subtract=subtract,
                           rounding=jax.lax.optimization_barrier if barrier else None)

    operands = jnp.asarray(left), jnp.asarray(right)
    high, low = (np.asarray(x) for x in (jax.jit(product) if compiled else product)(*operands))
    truth = left.astype(np.float64) * right.astype(np.float64)
    np.testing.assert_array_equal(high.view(np.uint32), truth.astype(np.float32).view(np.uint32))
    np.testing.assert_array_equal(low.astype(np.float64), truth - high.astype(np.float64))


def test_a_two_float_product_preserves_float64_projection_arithmetic():
    from dew.nn.precision import two_product

    mantissas = np.array([0, 1, (1 << 27) - 1, 1 << 27, (1 << 27) + 1, (1 << 52) - 1], np.uint64)
    significands = (np.uint64(1023 << 52) | mantissas).view(np.float64)
    values = np.ldexp(significands[:, None], [-400, 0, 400]).reshape(-1)
    left = np.concatenate([values, -values])
    right = np.roll(left[::-1], 7)
    nearest = left * right
    residual = np.array([float(Fraction(a) * Fraction(b) - Fraction(high))
                         for a, b, high in zip(left, right, nearest, strict=True)])
    with jax.enable_x64(new_val=True):
        high, low = (np.asarray(x) for x in jax.jit(two_product)(left, right))
    np.testing.assert_array_equal(high.view(np.uint64), nearest.view(np.uint64))
    np.testing.assert_array_equal(low, residual)


@pytest.mark.parametrize("barrier,subtract", [(True, False), (False, True)])
def test_a_bit_split_keeps_large_factors_finite(barrier, subtract):
    from dew.nn.precision import two_product

    left = np.array([1.0000001, -1.9999999, 1.5, -1.0002443], np.float32) * np.float32(2.0 ** 120)
    right = np.array([1.9999999, 1.0000001, -1.3333334, -1.0002443], np.float32) * np.float32(2.0 ** -100)
    high, low = jax.jit(lambda a, b: two_product(
        a, b, bit_split=True, subtract=subtract,
        rounding=jax.lax.optimization_barrier if barrier else None))(left, right)
    truth = left.astype(np.float64) * right.astype(np.float64)
    np.testing.assert_array_equal(high, truth.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(low, np.float64), truth - np.asarray(high, np.float64))


def test_bf16_logits_round_as_rounded_to_and_are_held_as_bf16_under_cuda():
    """The head's logits take `rounded_to`'s values and cotangents, bitwise;
    under CUDA the program holds them as bf16 instead of rounding fp32 in a
    reduce-precision of its own, which a serving draw's readers made a full
    fp32 pass (docs/performance.md)."""
    from dew.nn.precision import bf16_logits, rounded_to
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(16, 512)).astype(np.float32) * 8)
    weights = jnp.asarray(rng.normal(size=(16, 512)).astype(np.float32))

    def rounded(v):
        return rounded_to(v, jnp.bfloat16)

    def loss(head):
        return lambda v: jnp.sum(jax.nn.log_softmax(head(v * 1.5)) * weights)

    held = jax.jit(bf16_logits)(x)
    assert jnp.array_equal(held, jax.jit(rounded)(x)) and not jnp.array_equal(held, x)
    assert jnp.array_equal(jax.jit(jax.grad(loss(bf16_logits)))(x), jax.jit(jax.grad(loss(rounded)))(x))
    if jax.default_backend() == "gpu":
        text = jax.jit(jax.value_and_grad(loss(bf16_logits))).lower(x).compile().as_text()
        assert "reduce-precision" not in text and "bf16[16,512]" in text


@pytest.mark.parametrize("rows, columns, features", [(1, 37, 16), (1, 2048, 64), (6, 1, 8), (6, 3, 8),
                                                     (8, 1, 16), (12, 40, 16)])
def test_a_bf16_product_is_its_rounded_operands_exact_product_at_every_shape(rows, columns, features):
    """`bf16_product` and its backward are the exact products of the operands rounded to bf16, the
    result rounded to bf16 once, at a vector side (one row, one column) and under XLA's GEMM rewrite
    threshold as at any shape, over sixteen draws. Eighths sum exactly in fp32 in any order, so the
    logits, and the gradients under eighths' cotangent, are float64's to the bit, and the logits' many
    ties show round-to-nearest-even. On an A100 XLA's BF16_BF16_F32 algorithm, which the product
    named, missed both at these shapes (node receipts SF-a100-tie-probe3)."""
    import ml_dtypes

    from dew.nn.precision import bf16_product

    for seed in range(16):
        rng = np.random.default_rng([rows, columns, seed])
        states, table, weights = (np.round(rng.standard_normal(shape) * 8).astype(np.float32) / 8
                                  for shape in ((rows, features), (columns, features), (rows, columns)))
        exact = states.astype(np.float64) @ table.astype(np.float64).T
        logits, pullback = jax.vjp(lambda states, table: bf16_product("td,vd->tv", states, table),
                                   states, table)
        np.testing.assert_array_equal(
            np.asarray(logits), exact.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float32))
        d_states, d_table = pullback(jnp.asarray(weights))
        wide = weights.astype(np.float64)
        np.testing.assert_array_equal(np.asarray(d_states), (wide @ table).astype(np.float32))
        np.testing.assert_array_equal(np.asarray(d_table), (wide.T @ states).astype(np.float32))


BF16_PRODUCTS = [
    ("bxd,byd->bxy", ((2, 1, 16), (2, 37, 16))),
    ("tv,vd->td", ((1, 12), (12, 8))),
    ("tv,td->vd", ((3, 5), (3, 8))),
    ("...d,vd->...v", ((2, 3, 8), (6, 8))),
    ((((3, 2), (1, 3)), ((0,), (0,))), ((2, 3, 4, 5), (2, 5, 6, 4))),
    ("bxd,byd->bxy", ((1, 3, 16), (3, 5, 16))),
]
"""Products of the dot family at the BF16_BF16_F32 algorithm: an einsum's subscripts or a
dot_general's dimension numbers, and the operands' shapes. A decode's one-query product, a one-token
tile's state gradient, a head's gradient, a broadcast head, a batched product contracting two axes
paired out of order, and a batch of one broadcast against three, as einsum broadcasts it."""


@pytest.mark.parametrize("form, shapes", BF16_PRODUCTS,
                         ids=["decode", "token", "head", "ellipsis", "batched", "broadcast"])
@pytest.mark.parametrize("scope", [None, "F16_F16_F32"])
def test_the_dot_family_at_the_bf16_algorithm_multiplies_bf16_operands(form, shapes, scope):
    """`float32_einsum` and `float32_dot_general` at XLA's BF16_BF16_F32 algorithm form the products
    of their operands rounded to bf16 (to nearest even), summed in fp32 to an fp32 result, and
    backward the same of the cotangent rounded to bf16 (`bf16_dot_general`), at any shape and in a
    scope whose default precision is an F16 preset, which would narrow a bf16 operand of 2^20 to
    fp16's infinity. The operands are eighths, scaled by 2^20 and 2^-20, each moved by less than half a
    bf16 ulp, so their rounding is the eighths and every sum is exact: float64's to the bit."""
    from dew.nn.precision import float32_dot_general, float32_einsum

    algorithm = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    rng = np.random.default_rng(len(shapes[0]) * 31 + shapes[1][-1])
    exact = [np.round(rng.standard_normal(shape) * 8) / 8 * scale
             for shape, scale in zip(shapes, (2.**20, 2.**-20), strict=True)]
    moved = [(value * (1 + 2.**-10 * rng.uniform(-1, 1, value.shape))).astype(np.float32) for value in exact]

    def product(lhs, rhs, *, wide=False):
        if isinstance(form, str):
            if wide:
                return jnp.einsum(form, lhs, rhs, precision=jax.lax.Precision.HIGHEST)
            return float32_einsum(form, lhs, rhs, precision=algorithm, preferred_element_type=jnp.float32)
        if wide:
            return jax.lax.dot_general(lhs, rhs, form, precision=jax.lax.Precision.HIGHEST)
        return float32_dot_general(lhs, rhs, form, precision=algorithm, preferred_element_type=jnp.float32)

    with jax.enable_x64(new_val=True):
        want, pullback = jax.vjp(lambda lhs, rhs: product(lhs, rhs, wide=True), *exact)
        cotangent = np.round(rng.standard_normal(want.shape) * 8) / 8
        wants = pullback(jnp.asarray(cotangent))
    with contextlib.nullcontext() if scope is None else jax.default_matmul_precision(scope):
        got, pullback = jax.vjp(product, *moved)
        grads = pullback(jnp.asarray(cotangent, jnp.float32))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want, np.float32))
    for grad, wanted in zip(grads, wants, strict=True):
        np.testing.assert_array_equal(np.asarray(grad), np.asarray(wanted, np.float32))


@pytest.mark.parametrize(("subscripts", "shapes"), [
    ("ij,j", ((5, 7), (7,))),
    ("oh,bhwc,pw->bopc", ((3, 4), (2, 4, 5, 6), (2, 5))),
    ("...ij,...jk", ((2, 3, 4), (2, 4, 5))),
    ("ii,ij->ij", ((4, 4), (4, 3))),
    ("ij,jk,kl,lm->im", ((2, 3), (3, 4), (4, 5), (5, 2))),
], ids=["implicit", "three", "ellipsis", "diagonal", "chain"])
def test_any_einsum_at_the_bf16_algorithm_multiplies_bf16_operands(subscripts, shapes):
    """`float32_einsum` at XLA's BF16_BF16_F32 algorithm takes every expression einsum does (an
    implicit result, three or more operands, an ellipsis, a repeated index), each product its
    operands rounded to bf16 and summed in fp32, forward and backward.

    Equal to float64's einsum by construction: the operands and the cotangent are integers in
    [-2, 2], and every contraction here sums at most 20 terms, so each value a product reads (an
    operand, an intermediate of the contraction path, a cotangent flowing back through it) is an
    integer of magnitude at most 2 * 20 * 2 * 2 = 160 < 256, which bf16 holds exactly, and each
    product and sum is an integer fp32 holds exactly. Where a value a product reads were not exact
    in bf16, the algorithm's rounding of it would be its defined result, which this check does not
    reach."""
    from dew.nn.precision import float32_einsum

    rng = np.random.default_rng(len(subscripts))
    exact = [rng.integers(-2, 3, shape).astype(np.float64) for shape in shapes]

    def product(*operands, wide=False):
        if wide:
            return jnp.einsum(subscripts, *operands, precision=jax.lax.Precision.HIGHEST)
        return float32_einsum(subscripts, *operands, precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
                              preferred_element_type=jnp.float32)

    with jax.enable_x64(new_val=True):
        want, pullback = jax.vjp(lambda *operands: product(*operands, wide=True), *exact)
        cotangent = rng.integers(-2, 3, want.shape).astype(np.float64)
        wants = pullback(jnp.asarray(cotangent))
        assert max(float(np.abs(np.asarray(value)).max()) for value in (want, *wants)) < 2.**8
    got, pullback = jax.vjp(product, *(value.astype(np.float32) for value in exact))
    grads = pullback(jnp.asarray(cotangent, jnp.float32))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want, np.float32))
    for grad, wanted in zip(grads, wants, strict=True):
        np.testing.assert_array_equal(np.asarray(grad), np.asarray(wanted, np.float32))


@pytest.mark.gpu(reason="the GPU's fp32 division is what `import dew` sets; a CPU's rounds to the nearest")
def test_the_gpu_divides_fp32_to_the_nearest_float():
    """`import dew` has XLA:GPU divide fp32 to the nearest float
    (`dew.devices.divide_exactly`): over 2^20 quotients of operands
    from 1e-4 to 10 apart, each is the float64 quotient rounded to fp32, as a
    CPU's and the references' are. XLA's own `div.full` left 28% of them off
    by up to 2 ulps on an H100."""
    rng = np.random.default_rng(0)
    numerator, denominator = (rng.uniform(1e-4, 10, 1 << 20).astype(np.float32) for _ in range(2))
    quotient = np.asarray(jax.jit(jnp.divide)(numerator, denominator))
    nearest = (numerator.astype(np.float64) / denominator.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(quotient, nearest)


def test_quantizer_division_rounds_midpoint_neighbours_as_ieee_division():
    from dew.nn.precision import correctly_rounded_divide

    rng = np.random.default_rng(411)
    denominator = rng.uniform(1e-4, 10, (1, 16384)).astype(np.float32)
    quotient = rng.uniform(-16, 16, denominator.shape).astype(np.float32)
    midpoint = (quotient.astype(np.float64)
                + np.nextafter(quotient, np.float32(np.inf)).astype(np.float64)) * 0.5
    numerator = (midpoint * denominator.astype(np.float64)).astype(np.float32)
    numerator = np.concatenate([np.nextafter(numerator, np.float32(-np.inf)), numerator,
                                np.nextafter(numerator, np.float32(np.inf))])
    expected = (numerator.astype(np.float64) / denominator.astype(np.float64)).astype(np.float32)
    actual = np.asarray(jax.jit(correctly_rounded_divide)(numerator, denominator))
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def dot_precisions(function, *arguments) -> list:
    """The precision of every dot_general `function` traces over `arguments` under JAX's own default
    precision (the test lane sets 'highest' for every product that states none), with the dtype its
    operands promote to, sub-jaxprs included, as a GPU runs them: a CPU's float64 sums
    (`dew.nn.precision.accumulated`) are tests/test_cpu_accumulation.py's."""
    with jax.default_matmul_precision(None):
        graph = jax.make_jaxpr(function)(*arguments)
    return [(equation.params["precision"],
             jnp.result_type(*(variable.aval.dtype for variable in equation.invars)))
            for equation in equations(graph, platform="cuda") if equation.primitive.name == "dot_general"]


HIGHEST = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)


def assert_float32_products(precisions) -> None:
    """No product of float32 operands at JAX's default precision, which is TF32 on a GPU, and none
    of narrower ones at HIGHEST. A product stating its own dot algorithm (a bf16 head's) is its own."""
    assert precisions
    off = [(precision, dtype) for precision, dtype in precisions
           if (precision is None and dtype == jnp.float32) or (precision == HIGHEST and dtype != jnp.float32)]
    assert not off, off


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("stored", ["float32", "bfloat16", "bfloat16 kernels"])
def test_a_model_with_no_dtype_multiplies_each_product_as_its_operands_compute(stored, tied):
    """A model with no dtype computes each product in the dtype its operands promote to, as flax's
    layers do, so float32 products multiply at HIGHEST and bf16 ones at their own precision, product
    by product: over float32 parameters, bf16 ones, and bf16 kernels beside float32 norm scales,
    its vocabulary head tied to its embedding or its own. Initializing it multiplies as applying it
    does. A precision resolved once for the whole model from its parameters pinned HIGHEST on bf16
    products and missed float32 ones."""
    model = LAYERS["packed"](tie_embeddings=tied)
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % 64
    variables = model.init(jax.random.key(0), tokens)
    if stored != "float32":
        kept = stored == "bfloat16 kernels"
        variables = jax.tree_util.tree_map_with_path(
            lambda path, leaf: leaf if kept and "kernel" not in jax.tree_util.keystr(path)
            else leaf.astype(jnp.bfloat16), variables)
    assert_float32_products(dot_precisions(model.apply, variables, tokens))
    assert_float32_products(dot_precisions(lambda key: model.init(key, tokens), jax.random.key(0)))


def test_a_layer_over_bf16_parameters_fed_float32_multiplies_float32_products():
    """flax promotes a layer with no dtype over bf16 parameters and a float32 input to float32, so
    its products multiply at HIGHEST, though its parameters are bf16."""
    layer = GatedMLP(hidden_features=24, out_features=16)
    x = jax.random.normal(jax.random.key(0), (2, 16), jnp.float32)
    variables = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), layer.init(jax.random.key(1), x))
    precisions = dot_precisions(layer.apply, variables, x)
    assert precisions and all(dtype == jnp.float32 for _, dtype in precisions)
    assert_float32_products(precisions)


def test_soft_embeddings_over_a_bf16_table_multiply_float32_products():
    """Soft embeddings mix a bf16-stored table by float32 probabilities, a product jax computes in
    float32, so it multiplies at HIGHEST as a float32-stored table's does."""
    logits = jax.random.normal(jax.random.key(0), (2, 3, 32), jnp.float32)
    for stored in (jnp.float32, jnp.bfloat16):
        table = jax.random.normal(jax.random.key(1), (32, 16), jnp.float32).astype(stored)
        precisions = dot_precisions(lambda logits, table: soft_embeddings(logits, table, 1.0), logits, table)
        assert [precision for precision, _ in precisions] == [HIGHEST]


def test_a_projection_factory_keeps_the_contraction_its_caller_gives():
    """A layer built from a caller's projection factory multiplies as that factory says: a bf16 MLP
    whose projections return float32 results (`fp32_result_dot_general`) returns float32."""
    x = jax.random.normal(jax.random.key(0), (2, 16), jnp.bfloat16)
    layer = GatedMLP(hidden_features=24, out_features=16, dtype=jnp.bfloat16,
                     linear=functools.partial(nn.Dense, dot_general=fp32_result_dot_general()))
    assert layer.apply(layer.init(jax.random.key(1), x), x).dtype == jnp.float32


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("precision", [None, jax.lax.Precision.HIGHEST])
def test_xla_attention_multiplies_float32_products_and_a_stated_precision(dtype, precision):
    """'xla' attention over float32 queries multiplies float32 products where the caller states no
    precision, and every product at HIGHEST where it states that, on whichever path computes it;
    bf16 queries with none stated keep their own."""
    query, key, value = (part.astype(dtype) for part in
                         jax.random.normal(jax.random.key(1), (3, *BF16_QKV), jnp.float32))
    precisions = dot_precisions(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, implementation="xla", precision=precision), query, key, value)
    if precision is None:
        assert_float32_products(precisions)
    else:
        assert precisions and all(stated == HIGHEST for stated, _ in precisions)


@pytest.mark.parametrize("tied", [True, False])
def test_a_tiled_head_takes_each_tile_as_the_whole_head_takes_its_logits(tied):
    """A chunked head multiplies each vocabulary tile by `bf16_product` over the tile's rows
    (`dew.objectives.lm.chunked`): of a bf16 model those are its forward's logits' columns, bitwise,
    so the chunked head predicts those logits' argmax wherever a tile cuts; its scores are held to
    those logits' cross entropy by check 1 (tests/test_chunked_cross_entropy.py)."""
    from dew.nn.precision import bf16_product
    from dew.objectives.lm.chunked import chunked_cross_entropy

    model = CausalTransformer(vocab_size=97, emb_features=32, num_layers=2, num_heads=4, mlp_features=64,
                              max_seq_len=16, tie_embeddings=tied, dtype=jnp.bfloat16)
    ids = jax.random.randint(jax.random.key(0), (2, 12), 0, 97)
    targets = jax.random.randint(jax.random.key(1), (2, 12), 0, 97)
    variables = model.init(jax.random.key(0), ids)
    logits = np.asarray(model.apply(variables, ids)).reshape(-1, 97)
    states = model.apply(variables, ids, method="hidden_states")
    table = model.apply(variables, method="output_table")
    rows = table.matrix if table.vocab_major else table.matrix.T
    for first in range(0, 97, 25):
        tile = np.asarray(bf16_product("td,vd->tv", states.reshape(-1, 32), rows[first:first + 25]))
        np.testing.assert_array_equal(tile, logits[:, first:first + 25])
    for chunks in (1, 4):
        _, predicted, _ = chunked_cross_entropy(states, table, targets, chunks)
        np.testing.assert_array_equal(np.asarray(predicted).reshape(-1), logits.argmax(-1))


@pytest.mark.parametrize("axis", [None, 1, (0, 1), -1])
def test_a_weighted_total_rounds_once_and_keeps_excluded_values_out(axis):
    """`weighted_total` sums its weighted values and their weights each rounded once, as float64's
    sum rounded to float32: a row of one large value and 4095 ones, whose running float32 sum drops
    every one, totals as the exact sum does, whatever the axes; a NaN or infinity under weight zero
    reaches neither the total nor the mass."""
    from dew.nn.precision import weighted_total

    rows = np.ones((3, 4096), np.float32)
    rows[:, 0] = 2.0 ** 25
    weights = np.ones_like(rows)
    weights[:, -2:] = 0
    rows[:, -1], rows[:, -2] = np.nan, np.inf
    total, mass = jax.jit(lambda x, w: weighted_total(x, w, axis))(rows, weights)
    reduced = tuple(range(rows.ndim)) if axis is None else axis
    kept = np.where(weights != 0, rows, 0).astype(np.float64) * weights
    np.testing.assert_array_equal(np.asarray(total), np.sum(kept, axis=reduced).astype(np.float32))
    exact_mass = np.sum(weights.astype(np.float64), axis=reduced).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(mass), exact_mass)
