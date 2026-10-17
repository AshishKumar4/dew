"""Models' own dtype and attention fields, and the arithmetic their kernels honor."""

from fractions import Fraction
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_fp32_reduction_bound

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.dsa_kpool import KPoolSparseAttention
from dew.nn.inputs import AttentionMetadata, MixerCall, coordinates
from dew.nn.llama4 import Llama4Mixer
from dew.nn.local_attention import local_attention
from dew.nn.mixer_base import MixerContext
from dew.nn.mla import MultiHeadLatentAttention

import_module("dew.nn.multimodal")  # registers the fixture kind


BF16_QKV = (1, 4, 2, 8)  # [B, S, H, D]


@pytest.mark.parametrize("arguments", [
    {"precision": jax.lax.Precision.HIGHEST},
    {"force_fp32_for_softmax": False},
    {"dtype": jnp.float32},
])
def test_auto_takes_the_reference_path_for_arithmetic_only_it_performs(arguments):
    """'auto' resolves a call that asks for a matmul precision, a softmax
    dtype or a compute dtype no fused kernel honours to the reference path,
    and computes exactly what naming that path computes."""
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
    "llama4_chunk": lambda **policy: Llama4Mixer().build(MixerContext(
        emb_features=32, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=32, attention_chunk=4,
        qk_norm=False, **policy))(),
    "llama4_global": lambda **policy: Llama4Mixer(use_rope=False).build(MixerContext(
        emb_features=32, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=32, qk_norm=False,
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


@pytest.mark.gpu(reason="the GPU's fp32 division is what `import dew` sets; a CPU's rounds to the nearest")
def test_the_gpu_divides_fp32_to_the_nearest_float():
    """`import dew` has XLA:GPU divide fp32 to the nearest float
    (`dew.telemetry.devices.divide_exactly`): over 2^20 quotients of operands
    from 1e-4 to 10 apart, each is the float64 quotient rounded to fp32, as a
    CPU's and the references' are. XLA's own `div.full` left 28% of them off
    by up to 2 ulps on an H100."""
    rng = np.random.default_rng(0)
    numerator, denominator = (rng.uniform(1e-4, 10, 1 << 20).astype(np.float32) for _ in range(2))
    quotient = np.asarray(jax.jit(jnp.divide)(numerator, denominator))
    nearest = (numerator.astype(np.float64) / denominator.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(quotient, nearest)


def test_quantizer_division_rounds_midpoint_neighbours_as_ieee_division():
    from dew.nn.fake_quant import correctly_rounded_divide

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
