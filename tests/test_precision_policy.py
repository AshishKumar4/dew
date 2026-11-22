"""Models' own dtype and attention fields, and the arithmetic their kernels honor."""

import contextlib
import functools
from fractions import Fraction
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from low_precision_support import edge_values
from reference_error import assert_as_exact, assert_fp32_reduction_bound, dot_precisions

from dew.nn.aligned import aligned_sum
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.dense import Dense
from dew.nn.diffusion_gemma import soft_embeddings
from dew.nn.dsa_kpool import KPoolSparseAttention
from dew.nn.float_bits import (
    TWO_FACTOR_REACH,
    binade_of,
    float_bits,
    lowered_rounded,
    raised_exactly,
    raw_fields,
    widened_exactly,
)
from dew.nn.kernels import generation
from dew.nn.local_attention import local_attention
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import AttentionMixer
from dew.nn.mla import MultiHeadLatentAttention
from dew.nn.mlp import GatedMLP
from dew.nn.precision import (
    ACCURATE,
    Numerics,
    _rounded_significand,
    compensated_add,
    compensated_difference,
    exact_products,
    float64_sum,
    fp32_result_dot_general,
    two_product,
    two_sum,
)
from dew.telemetry.instrumentation import hlo_instructions

BF16_QKV = (1, 4, 2, 8)  # [B, S, H, D]
SEEDS = {"auto": 8, "local auto": 8, "default": 8}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (paired, on armada): the most any
of a site's cases took."""


def reference_attention(q, k, v):
    """The reference path with no precision of its own: run on the draw widened, check 1's truth."""
    return scaled_dot_product_attention(q, k, v, implementation="reference")


def qkv_draws(dtype):
    """Seeded draws of a query, a key and a value of `BF16_QKV`'s shape in `dtype`."""
    def draw(seed):
        return tuple(jax.random.normal(jax.random.fold_in(jax.random.key(1), seed), (3, *BF16_QKV), dtype))
    return draw


@pytest.mark.parametrize("arguments", [
    {"precision": jax.lax.Precision.HIGHEST},
    {"precision": jax.lax.DotAlgorithmPreset.F32_F32_F32},
    {"force_fp32_for_softmax": False},
    {"dtype": jnp.float32},
])
def test_auto_computes_arithmetic_only_the_reference_path_performs(arguments):
    """'auto' given a matmul precision, a dot algorithm among them, a softmax dtype or a compute dtype no
    fused kernel honours computes on par with naming the reference path over the seeds (check 1), the
    reference path run in float64 its truth."""
    draw = qkv_draws(jnp.bfloat16)

    def call(implementation):
        return lambda q, k, v: scaled_dot_product_attention(q, k, v, implementation=implementation,
                                                            **arguments)

    assert_as_exact(jax.jit(call("auto")), jax.jit(call("reference")), draw, f"auto {arguments}",
                    seeds=SEEDS["auto"], truth_fn=reference_attention)


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


@pytest.mark.parametrize("implementation, target", [("xla", "sm90"), ("template", None), ("tpu", None)])
def test_an_fp32_call_stating_highest_multiplies_every_product_at_highest(implementation, target):
    """An fp32 call stating precision='highest' in a run at the default precision multiplies every fp32
    product of its gradient, forward and backward, at HIGHEST on each fused route that takes it
    (`precision.at_precision`, which holds the call's precision over the backward a custom derivative traces
    after the call): jax.nn's xla attention, fp32's declared cell on an H100 and an A100, traced as an H100
    runs it; the template, whose portable mainloop (an A100's, a call Mosaic and Triton decline) multiplies at
    the configuration its products read (`templates.attention.interpreted_dot`); and splash's route, its
    Pallas kernels interpreted off a TPU. Never at the run's default."""
    def loss(q, k, v):
        return jnp.sum(scaled_dot_product_attention(q, k, v, implementation=implementation,
                                                    precision="highest", causal=True) ** 2)

    draw = jax.random.normal(jax.random.key(1), (3, 1, 128, 2, 64), jnp.float32)
    with generation.run_precision(Numerics("32-true")), generation.assumed(target):
        precisions = dot_precisions(jax.grad(loss, argnums=(0, 1, 2)), *draw)
    narrowed = [stated for stated, dtype, _ in precisions if dtype == jnp.float32 and stated != HIGHEST]
    assert precisions and not narrowed, narrowed


@pytest.mark.parametrize("operand, stated, run, cublas", [
    (jnp.float32, jax.lax.Precision.HIGHEST, None, True), (jnp.float32, None, "highest", True),
    (jnp.float32, None, None, False), (jnp.bfloat16, jax.lax.Precision.HIGHEST, None, False),
    (jnp.int8, None, None, True)],
    ids=["an operation's fp32 at highest", "the run's fp32 at highest", "fp32 at the default",
         "bf16 at highest", "int8"])
def test_a_program_holding_an_ieee_fp32_dot_compiles_its_gemm_on_cublas(operand, stated, run, cublas,
                                                                        monkeypatch):
    """A program holding a dot XLA's Triton GEMM computes wrong on an A100 compiles it on cuBLAS
    (`report.targeted` reads the program's own dots and its device's target, `report.triton_wrong`): fp32
    operands at IEEE fp32 products, whose cuBLAS sum is nearer float64 (rows 5026/5028/5029: a Wan walk
    1.59 times the reference's distance where 1.5 holds), one an operation states 'highest' in a run at
    the default as one the run states; and an int8 product, which Triton computes as two or four times
    it (openxla/xla#51016), whoever made it. At the default precision, or of bf16 operands, the dot stays
    XLA's own, a Triton GEMM. Compiled through Dew's one compile for a compile-only
    A100 (`report.xla_compiled`), its instructions read as XLA's schema states them."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.kernels.templates.report import xla_compiled
    from dew.telemetry.instrumentation import custom_call_kind, hlo_instructions

    monkeypatch.setenv("XLA_FLAGS", "")
    mesh = Mesh(np.array(compile_only_gpus("sm80", 1, 80 * 2**30)), ("data",))
    placed = NamedSharding(mesh, PartitionSpec())

    def product(x, w):
        return jnp.dot(x, w, precision=stated,
                       preferred_element_type=jnp.int32 if operand == jnp.int8 else None)

    shapes = [jax.ShapeDtypeStruct(shape, operand, sharding=placed) for shape in ((256, 512), (512, 384))]
    with jax.default_matmul_precision(run):
        lowered = jax.jit(product).trace(*shapes).lower()
    instructions = hlo_instructions(xla_compiled(lowered, None)).values()
    # No dot left to XLA's own fusions (a Triton GEMM), the product a GEMM library call, whichever of
    # cuBLAS's the device's compile takes (`custom_call_kind`).
    own = [each.op for each in instructions if each.op == "dot" or "dot" in each.fused]
    library = [each.target for each in instructions if custom_call_kind(each.target) == "gemm"]
    assert (bool(library) and not own) == cublas, (library, own)


@pytest.mark.parametrize("scope", ["plain", "forward mode"])
def test_an_fp32_call_stating_highest_takes_its_forward_derivative_on_jax_nn_s_xla_attention(scope):
    """jax.nn's xla attention is JAX's own products, which forward mode differentiates as written: an fp32
    call stating 'highest', traced as an H100 runs it, takes its tangent, with `forward_mode_attention` or
    without, every fp32 product at HIGHEST (the forward's scope, `configured_precision`, each product keeping
    the precision it was traced at)."""
    from dew.nn.attention import forward_mode_attention

    def call(q, k, v):
        return scaled_dot_product_attention(q, k, v, implementation="xla", precision="highest", causal=True)

    draw = jax.random.normal(jax.random.key(1), (3, 1, 128, 2, 64), jnp.float32)
    held = forward_mode_attention() if scope == "forward mode" else contextlib.nullcontext()
    with generation.run_precision(Numerics("32-true")), generation.assumed("sm90"), held:
        precisions = dot_precisions(lambda q, k, v: jax.jvp(call, (q, k, v), (q, k, v)), *draw)
    narrowed = [stated for stated, dtype, _ in precisions if dtype == jnp.float32 and stated != HIGHEST]
    assert precisions and not narrowed, narrowed


@pytest.mark.parametrize("span", [{"window": 4}, {"chunk": 4}])
def test_local_attention_under_auto_computes_the_precision_it_is_asked_for(span):
    """A window or a chunk over packed documents at the highest matmul precision: 'auto' on par with
    naming the reference path over the seeds (check 1), the reference path run in float64 its truth."""
    packed = {"segment_ids": jnp.ones((2, 64), jnp.int32)}

    def draw(seed):
        return tuple(jax.random.normal(jax.random.fold_in(jax.random.key(1), seed), (3, 2, 64, 4, 8)))

    def call(implementation, **precision):
        return lambda q, k, v: local_attention(q, k, v, **span, **packed, **precision,
                                               implementation=implementation)

    auto, reference = (jax.jit(call(name, precision="highest")) for name in ("auto", "reference"))
    assert_as_exact(auto, reference, draw, f"auto {span}", seeds=SEEDS["local auto"],
                    truth_fn=call("reference"))


def test_fused_attention_default_precision_matches_the_attention_equation():
    """XLA's attention at the default precision on par with the attention equation, the reference path run
    in float64 its truth (check 1); `Precision.DEFAULT` and "default" spell that default alike, each the same
    program as no precision given, bit for bit on one draw."""
    draw = qkv_draws(jnp.float32)

    def attended(precision):
        return jax.jit(lambda q, k, v: scaled_dot_product_attention(q, k, v, precision=precision,
                                                                    implementation="xla"))

    assert_as_exact(attended(None), jax.jit(reference_attention), draw, "xla at the default precision",
                    seeds=SEEDS["default"], truth_fn=reference_attention)
    operands = draw(0)
    for alias in (jax.lax.Precision.DEFAULT, "default"):
        np.testing.assert_array_equal(np.asarray(attended(alias)(*operands)),
                                      np.asarray(attended(None)(*operands)))


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


ERROR_FREE = {
    "two_sum": two_sum,
    "two_product": two_product,
    "two_product bit-split": functools.partial(two_product, bit_split=True, subtract=True),
    "compensated_add": lambda a, b: compensated_add((a, b), (b, a)),
    "compensated_difference": lambda a, b: compensated_difference((a, b), (b, a)),
    "rounded significand": _rounded_significand,
    "two_sum by name": lambda a, b: two_sum(a, right=b),
    "two_sum by names": lambda a, b: two_sum(left=a, right=b),
}
"""Each error-free transform of `dew.nn.precision` over two arrays, the correctly rounded division's
significand among them, and one given its operands by name."""


@pytest.mark.parametrize("name", ERROR_FREE)
def test_an_error_free_transform_is_one_protected_span_its_derivative_keeps(name):
    """An error-free transform traces as one marked span of kind `Protected` (`dew.nn.derivatives.marked`)
    holding the whole of it, every array it is given by position or by name an operand of the span (a
    compensated sum's pairs repeat them) and none other, and
    its derivative's program keeps the mark, so the algebra pass reads what it may never reassociate as an
    object, not a name scope (ledger 3640). A composite (a compensated sum, the significand's rounding)
    is its own span, not only the spans of the transforms it calls."""
    from reference_error import equations

    from dew.nn.derivatives import Protected, marked_call_p

    transform = ERROR_FREE[name]
    a, b = (jnp.asarray(np.random.default_rng(seed).uniform(0.5, 2.0, 8), jnp.float32) for seed in (1, 2))
    program = jax.make_jaxpr(transform)(a, b).jaxpr
    (span,) = program.eqns
    assert (span.primitive, span.params["kind"]) == (marked_call_p, Protected())
    assert set(span.invars) == set(program.invars)

    def summed(a, b):
        return sum(jnp.sum(leaf) for leaf in jax.tree.leaves(transform(a, b)))

    derivative = jax.make_jaxpr(jax.grad(summed, argnums=(0, 1)))(a, b)
    assert Protected() in [equation.params["kind"] for equation in equations(derivative)
                           if equation.primitive is marked_call_p]


def test_a_transform_given_its_operands_by_name_computes_and_differentiates_as_by_position():
    """An operand passed by name is the span's like one passed by position (`_protected` binds the
    transform's signature), so a compiled program and its derivative take it as an input of the span, not
    a constant of its body, and compute what the positional call does."""
    a, b = (jnp.asarray(np.random.default_rng(seed).uniform(0.5, 2.0, 8), jnp.float32) for seed in (4, 5))
    for call in (lambda a, b: two_sum(a, right=b), lambda a, b: two_sum(left=a, right=b)):
        for left, right in zip(jax.jit(call)(a, b), jax.jit(two_sum)(a, b), strict=True):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        by_name = jax.jit(jax.grad(lambda a, b, call=call: jnp.sum(call(a, b)[0]), argnums=(0, 1)))(a, b)
        by_position = jax.jit(jax.grad(lambda a, b: jnp.sum(two_sum(a, b)[0]), argnums=(0, 1)))(a, b)
        for left, right in zip(by_name, by_position, strict=True):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_a_constant_factor_splits_on_the_host_outside_its_protected_span():
    """A factor known when the product traces (a kernel's static one) splits where it is known: the
    protected span takes the traced operand alone, the constant's halves bound into it, and the product
    and residual are the float64 product's, as an unmarked split's are."""
    from dew.nn.derivatives import marked_call_p

    left = jnp.asarray(np.random.default_rng(3).uniform(-4.0, 4.0, 8), jnp.float32)
    factor = np.float32(1.4426950408889634)
    program = jax.make_jaxpr(lambda a: two_product(a, factor, bit_split=True))(left)
    (span,) = [equation for equation in program.jaxpr.eqns if equation.primitive is marked_call_p]
    assert len(span.invars) == 1
    high, low = jax.jit(lambda a: two_product(a, factor, bit_split=True))(left)
    truth = np.asarray(left, np.float64) * np.float64(factor)
    np.testing.assert_array_equal(np.asarray(high), truth.astype(np.float32))
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
        compiled = jax.jit(jax.value_and_grad(loss(bf16_logits))).lower(x).compile()
        program = hlo_instructions(compiled.as_text())
        assert not [name for name, instruction in program.items()
                    if "reduce-precision" in {instruction.op, *instruction.fused}]
        results = {(instruction.element, instruction.dims) for instruction in program.values()}
        assert ("bf16", (16, 512)) in results


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


def test_an_einsum_at_the_bf16_algorithm_sums_without_a_product_in_fp32():
    """An einsum that forms no product, a sum, sums in fp32 at the bf16 algorithm, as einsum's own
    reduction does at the preferred type: 257 bf16 ones sum to 257, which bf16 cannot hold. (A sum
    that feeds a product is the product's operand, which the algorithm rounds to bf16.)"""
    from dew.nn.precision import float32_einsum

    summed = float32_einsum("i->", jnp.ones((257,), jnp.bfloat16),
                            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
                            preferred_element_type=jnp.float32)
    np.testing.assert_array_equal(np.asarray(summed), np.asarray(257.0, np.float32))


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


def the_gpu_divides_fp32_to_the_nearest_float():
    """`import dew` has XLA:GPU divide fp32 to the nearest float
    (`dew._devices.divide_exactly`): over 2^20 quotients of operands
    from 1e-4 to 10 apart, each is the float64 quotient rounded to fp32, as a
    CPU's and the references' are. XLA's own `div.full` left 28% of them off
    by up to 2 ulps on an H100."""
    from reference_error import ahead_arm

    rng = np.random.default_rng(0)
    numerator, denominator = (rng.uniform(1e-4, 10, 1 << 20).astype(np.float32) for _ in range(2))
    # A program with no check-1 site, prepared as an arm is in a GPU tier part (`ahead_arm`).
    quotient = np.asarray(ahead_arm(jax.jit(jnp.divide), "the GPU's fp32 division")(numerator, denominator))
    nearest = (numerator.astype(np.float64) / denominator.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(quotient, nearest)


def division_edges() -> tuple[np.ndarray, np.ndarray]:
    """Float32 numerators and denominators at division's edges: normal operands of every exponent, whose
    quotients overflow, fall to the subnormal grid or below it; powers of two; the largest and least normal
    floats; quotients on a midpoint of the normal grid and of the subnormal grid and their neighbours; and
    zero, subnormal, infinite and NaN operands."""
    rng = np.random.default_rng(6312)
    width = 1 << 16

    def normal(count, low=-126, high=127):
        signs = np.where(rng.random(count) < 0.5, -1, 1)
        scaled = np.ldexp(rng.uniform(1, 2, count), rng.integers(low, high + 1, count))
        return (signs * scaled).astype(np.float32)

    pairs = [(normal(width), normal(width))]
    # Quotients near the bottom of the range: a small numerator over a large denominator.
    pairs.append((normal(width, -126, -100), normal(width, 0, 30)))
    powers = np.ldexp(np.float32(1), rng.integers(-126, 128, (2, width))).astype(np.float32)
    pairs.append((powers[0], powers[1]))
    pairs.append((powers[0], normal(width)))
    extreme = np.array([np.finfo(np.float32).max, np.finfo(np.float32).tiny], np.float32)
    pairs.append((np.repeat(extreme, width // 2), normal(width)))
    pairs.append((normal(width), np.repeat(extreme, width // 2)))
    # Normal quotients on a midpoint and beside it.
    denominator = rng.uniform(1, 2, width // 4).astype(np.float32)
    quotient = rng.uniform(-16, 16, denominator.shape).astype(np.float32)
    midpoint = (quotient.astype(np.float64)
                + np.nextafter(quotient, np.float32(np.inf)).astype(np.float64)) * 0.5
    numerator = (midpoint * denominator.astype(np.float64)).astype(np.float32)
    pairs.extend((numerator if step is None else np.nextafter(numerator, np.float32(step)), denominator)
                 for step in (-np.inf, None, np.inf))
    # Subnormal quotients on a midpoint of the subnormal grid, exactly: an odd multiple of 2^-150 times a
    # denominator of three times a power of two, a normal numerator, and beside it.
    odd = 2 * rng.integers(1 << 18, 1 << 19, width // 4) + 1
    exponent = rng.integers(155, 175, width // 4)
    small = np.ldexp(3.0, exponent - 150).astype(np.float32)
    exact = np.ldexp(odd * 3.0, exponent - 300).astype(np.float32)
    pairs.extend((exact if step is None else np.nextafter(exact, np.float32(step)), small)
                 for step in (-np.inf, None, np.inf))
    specials = np.array([0.0, -0.0, np.float32(1e-40), -np.float32(1e-40), np.inf, -np.inf, np.nan, 1.5,
                         -3.0], np.float32)
    grid = np.meshgrid(specials, specials)
    pairs.append((grid[0].ravel(), grid[1].ravel()))
    return (np.concatenate([numerator for numerator, _ in pairs]),
            np.concatenate([denominator for _, denominator in pairs]))


EDGES = "a correctly rounded division's edges"
"""The label of `divides_as_ieee_division_at_its_edges`'s arm over the edges, before its divider's."""


def divides_as_ieee_division_at_its_edges(divider: str = ""):
    """`correctly_rounded_divide` over `division_edges`, compiled for this backend: each quotient of normal
    operands IEEE division's, bit for bit (NumPy's float64 quotient rounded once to float32, which rounds as
    once: 53 >= 2 x 24 + 2), its overflows infinite and its subnormal quotients on their grid, ties to even,
    over divisors given as arrays and as constants (which the backend's division approximates); and each
    quotient of a zero, subnormal, infinite or NaN operand the backend's own division. Its arms' labels end in
    `divider`, a process's divider of its own (`APPROXIMATELY`), so one test's store holds both processes'."""
    from reference_error import ahead_arm

    from dew.nn.precision import correctly_rounded_divide

    numerator, denominator = division_edges()
    normal = (np.abs(numerator) >= np.finfo(np.float32).tiny) & np.isfinite(numerator) & (
        np.abs(denominator) >= np.finfo(np.float32).tiny) & np.isfinite(denominator)
    # Programs with no check-1 site, each prepared as an arm is in a GPU tier part (`ahead_arm`), every one
    # run before any is judged: a prepare's arm gives zeros.
    actual = np.asarray(ahead_arm(jax.jit(correctly_rounded_divide), EDGES + divider)(numerator, denominator))
    past = f"the plain division of the edges past the normals{divider}"
    plain = np.asarray(ahead_arm(jax.jit(jnp.divide), past)(numerator[~normal], denominator[~normal]))
    # A constant divisor, which XLA takes as a product by its rounded reciprocal (on a CPU and a GPU alike),
    # so the first quotient is off and the correction and the last move decide: the edges' normal numerators
    # over divisors just below a power of two, odd ones and the ones the callers divide by.
    held = numerator[(np.abs(numerator) >= np.finfo(np.float32).tiny) & np.isfinite(numerator)]
    divisors = [np.array(bits, np.uint32).view(np.float32) for bits in (
        0x3FFFFFFF, 0x3F7FFFFF, 0x40400000, 0x42FE0000, 0x43E00000, 0x3F400000, 0x7F000000, 0x00800000)]
    quotients = [np.asarray(ahead_arm(jax.jit(lambda values, divisor=divisor: correctly_rounded_divide(
        values, divisor)), f"a correctly rounded division by {hex(divisor.view(np.uint32))}{divider}")(held))
        for divisor in divisors]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        expected = (numerator.astype(np.float64) / denominator.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(actual[normal].view(np.uint32), expected[normal].view(np.uint32))
    np.testing.assert_array_equal(actual[~normal], plain)
    for divisor, quotient in zip(divisors, quotients, strict=True):
        with np.errstate(over="ignore", under="ignore"):
            wanted = (held.astype(np.float64) / np.float64(divisor)).astype(np.float32)
        np.testing.assert_array_equal(quotient.view(np.uint32), wanted.view(np.uint32),
                                      err_msg=hex(divisor.view(np.uint32)))


APPROXIMATE = "--xla_backend_extra_options=-nvptx-prec-divf32=1"
"""XLA:GPU's own fp32 division, approximate (`div.full`), which a run naming its backend options keeps
(`dew._devices.divide_exactly`)."""


APPROXIMATELY = ", on an approximate divider"
"""The suffix of the arms' labels of `divides_as_ieee_division_on_an_approximate_divider`'s process."""


def divides_as_ieee_division_on_an_approximate_divider():
    """`divides_as_ieee_division_at_its_edges` in a process whose GPU divides fp32 approximately
    (`APPROXIMATE`), so each first quotient is off and the correction and its last move decide every
    result: there its plain division is off IEEE's somewhere, which the check would otherwise not show. The
    process opens its test's store (`truths_ahead.for_a_child`, `approximately_divided`), so a GPU tier
    part's prepare prepares its arms and its run compiles none on its GPU (row 10748: it was passed by in the
    prepare, every arm compiled on the held GPU unprepared and unrecorded)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    from truths_ahead import for_a_child

    flags = " ".join(filter(None, (os.environ.get("XLA_FLAGS"), APPROXIMATE)))
    program = ("import sys; sys.path.insert(0, sys.argv[1])\n"
               "import test_precision_policy\n"
               "test_precision_policy.approximately_divided()\n")
    subprocess.run([sys.executable, "-c", program, str(Path(__file__).parent)],
                   env={**os.environ, **for_a_child(), "XLA_FLAGS": flags}, check=True)


def approximately_divided():
    """`divides_as_ieee_division_on_an_approximate_divider`'s own process: its plain division off IEEE's
    somewhere, and the edges held, in its test's store (`truths_ahead.of_the_parent`), each program an arm
    (`ahead_arm`) judged after every one ran; in a prepare, a failure checking its arms' zeros passes it
    (`CompiledTruths.judged_its_zeros`), its run judging it."""
    from reference_error import ahead_arm
    from truths_ahead import of_the_parent

    rng = np.random.default_rng(0)
    numerator, denominator = (rng.uniform(1e-4, 10, 1 << 20).astype(np.float32) for _ in range(2))
    with of_the_parent() as store:
        try:
            plain = np.asarray(ahead_arm(jax.jit(jnp.divide), f"the GPU's fp32 division{APPROXIMATELY}")(
                numerator, denominator))
            divides_as_ieee_division_at_its_edges(APPROXIMATELY)
            nearest = (numerator.astype(np.float64) / denominator.astype(np.float64)).astype(np.float32)
            assert (plain != nearest).any(), "this process divides as IEEE does"
        except Exception as error:
            if store is None or not store.judged_its_zeros(error):
                raise


def test_a_correctly_rounded_division_is_ieee_divisions_at_its_edges():
    """On this CPU (`divides_as_ieee_division_at_its_edges`); the GPU tiers hold it there."""
    divides_as_ieee_division_at_its_edges()


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


def test_a_correctly_rounded_divisions_derivatives_divide_as_ieee_division():
    """The derivatives of a correctly rounded division divide correctly rounded too, as PyTorch's backward
    divides: a cotangent or a tangent over a constant divisor is its IEEE quotient, where XLA's own division
    by a constant multiplies by the constant's rounded reciprocal (an ulp off on a third of entries, row
    5277), and the gradient in the denominator is `-g * ((n / d) / d)`, each quotient rounded once."""
    from dew.nn.precision import divided

    rng = np.random.default_rng(5277)
    numerator, weights = (rng.uniform(-8, 8, (4096,)).astype(np.float32) for _ in range(2))
    denominator = rng.uniform(0.5, 8, (4096,)).astype(np.float32)
    six = np.float32(6)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(divided(x, 6.0) * weights)))(numerator)
    tangent = jax.jit(lambda x, direction: jax.jvp(lambda x: divided(x, 6.0), (x,), (direction,))[1])(
        numerator, weights)
    np.testing.assert_array_equal(np.asarray(gradient), weights / six)
    np.testing.assert_array_equal(np.asarray(tangent), weights / six)
    pulled = jax.jit(jax.grad(lambda d: jnp.sum(divided(numerator, d) * weights)))(denominator)
    np.testing.assert_array_equal(np.asarray(pulled), -(weights * ((numerator / denominator) / denominator)))
    batched = jax.jit(jax.vmap(jax.grad(lambda x, w: jnp.sum(divided(x, 6.0) * w))))(
        numerator.reshape(64, 64), weights.reshape(64, 64))
    np.testing.assert_array_equal(np.asarray(batched).reshape(-1), weights / six)
    # A divisor batched beside a numerator that is not: the division broadcasts the one over the other, and
    # the numerator's gradient is the sum of the cotangent's quotients over the divisors (two, one rounding).
    first, second = denominator[:2]
    over_both = jax.jit(jax.grad(lambda x: jnp.sum(jax.vmap(lambda d: divided(x, d))(denominator[:2]))))(six)
    np.testing.assert_array_equal(np.asarray(over_both), np.float32(1) / first + np.float32(1) / second)


@pytest.mark.mesh(devices=2)
def test_a_correctly_rounded_division_keeps_an_explicit_axis_through_its_gradient():
    """A correctly rounded division of rows a mesh's Explicit data axis splits is typed as JAX's own division
    is, its result and its gradient split along that axis as its numerator is: the gradient over a constant
    divisor is the cotangent's IEEE quotient, placed as the numerator. Typed without its operands' sharding,
    the division's transpose was refused its cotangent (a JEPA step on an Explicit data axis)."""
    from dew.nn.precision import divided

    rng = np.random.default_rng(5277)
    numerator, weights = (rng.uniform(-8, 8, (8, 16)).astype(np.float32) for _ in range(2))
    mesh = jax.make_mesh((2,), ("data",), (jax.sharding.AxisType.Explicit,), devices=jax.devices()[:2])
    with jax.set_mesh(mesh):
        rows = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
        split, weighed = (jax.device_put(value, rows) for value in (numerator, weights))
        gradient = jax.jit(jax.grad(lambda x, w: jnp.sum(divided(x, 6.0) * w)))(split, weighed)
    np.testing.assert_array_equal(np.asarray(gradient), weights / np.float32(6))
    assert gradient.sharding.is_equivalent_to(rows, gradient.ndim)


HIGHEST = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)


def assert_configured_products(precisions, vocab: int | None = None) -> None:
    """Every product at JAX's configured precision (`jax_default_matmul_precision`, ledger 2500), none
    pinned at HIGHEST, which JAX's default runs on a GPU in TF32 at 2.8 times the speed, but a float32
    vocabulary head's, over an operand `vocab` wide, which states IEEE float32
    (`dew.nn.precision.exact_float32`). A product stating its own dot algorithm (a bf16 head's) is its
    own."""
    assert precisions
    off = [(precision, dtype, shapes) for precision, dtype, shapes in precisions
           if (precision == HIGHEST) != (dtype == jnp.float32 and any(vocab in shape for shape in shapes))]
    assert not off, off


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("stored", ["float32", "bfloat16", "bfloat16 kernels"])
def test_a_model_with_no_dtype_multiplies_each_product_as_its_operands_compute(stored, tied):
    """A model with no dtype or precision computes each product in the dtype its operands promote to,
    as flax's layers do, at JAX's configured precision, product by product: over float32 parameters,
    bf16 ones, and bf16 kernels beside float32 norm scales, its vocabulary head tied to its embedding
    or its own. Initializing it multiplies as applying it does. float32 products were pinned at
    HIGHEST, over JAX's config."""
    model = LAYERS["packed"](tie_embeddings=tied)
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % 64
    variables = model.init(jax.random.key(0), tokens)
    if stored != "float32":
        kept = stored == "bfloat16 kernels"
        variables = jax.tree_util.tree_map_with_path(
            lambda path, leaf: leaf if kept and "kernel" not in jax.tree_util.keystr(path)
            else leaf.astype(jnp.bfloat16), variables)
    assert_configured_products(dot_precisions(model.apply, variables, tokens), vocab=64)
    assert_configured_products(dot_precisions(lambda key: model.init(key, tokens), jax.random.key(0)),
                               vocab=64)


NUMERICS = {"default": Numerics(), "accurate": ACCURATE}


@pytest.mark.parametrize("policy", sorted(NUMERICS))
def test_a_runs_precision_policy_sets_every_product_that_states_none(policy):
    """Under a run's ACCURATE numerics (`generation.run_precision`, ledger 2500) every product of a float32
    decoder multiplies at HIGHEST, IEEE float32, and the attention template takes its accurate form;
    under the default each takes JAX's default but the vocabulary head's, which states IEEE float32
    (`dew.nn.precision.exact_float32`)."""
    model = LAYERS["packed"](dtype=jnp.float32)
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % 64
    variables = model.init(jax.random.key(0), tokens)

    def traced(variables, tokens):
        with generation.run_precision(NUMERICS[policy]):
            return model.apply(variables, tokens)

    precisions = dot_precisions(traced, variables, tokens)
    with generation.run_precision(NUMERICS[policy]):
        form = generation.numerics().forms
    if policy == "accurate":
        assert precisions and all(precision == HIGHEST for precision, _, _ in precisions), precisions
        assert form == "accurate"
    else:
        assert_configured_products(precisions, vocab=64)
        assert form == "fast"


def test_a_bf16_head_under_a_runs_highest_precision_leaves_the_bf16_algorithm():
    """A bf16 decoder's vocabulary head takes the bf16 dot algorithm at the default precision; under a
    run's ACCURATE numerics, which configure "highest", every product of its forward and of its chunked
    head multiplies at HIGHEST, none at the bf16 algorithm (`operand_precision`, review of ledger 2500)."""
    from dew.objectives.lm.chunked import chunked_cross_entropy

    model = LAYERS["packed"](dtype=jnp.bfloat16)
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % 64
    variables = model.init(jax.random.key(0), tokens)

    def traced(variables, tokens):
        with generation.run_precision(ACCURATE):
            states = model.apply(variables, tokens, method="hidden_states")
            table = model.apply(variables, method="output_table")
            return model.apply(variables, tokens), chunked_cross_entropy(states, table, tokens, 2)

    precisions = dot_precisions(traced, variables, tokens)
    assert precisions and all(precision == HIGHEST for precision, _, _ in precisions), precisions


def test_a_bf16_head_under_a_configured_dot_algorithm_keeps_the_bf16_algorithm():
    """A configured dot algorithm preset states how dots that state no precision sum (check 1's arms
    configure F32_F32_F32), no level a run picks: a bf16 product under it takes the bf16 algorithm, whose
    sums are fp32 already, as under the default precision; a run's "highest" level leaves it
    (`operand_precision`, `rounds_to_bf16`)."""
    from dew.nn.precision import rounds_to_bf16

    def rounds(configured):
        with jax.default_matmul_precision(configured):
            return rounds_to_bf16(jnp.bfloat16)

    assert rounds("F32_F32_F32") == rounds(None)
    assert not rounds("highest")


def test_a_layer_over_bf16_parameters_fed_float32_multiplies_at_jax_s_configured_precision():
    """flax promotes a layer with no dtype over bf16 parameters and a float32 input to float32, and
    its products multiply at JAX's configured precision, as a float32 layer's do."""
    layer = GatedMLP(hidden_features=24, out_features=16)
    x = jax.random.normal(jax.random.key(0), (2, 16), jnp.float32)
    variables = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), layer.init(jax.random.key(1), x))
    precisions = dot_precisions(layer.apply, variables, x)
    assert precisions and all(dtype == jnp.float32 for _, dtype, _ in precisions)
    assert_configured_products(precisions)


def test_soft_embeddings_over_a_bf16_table_multiply_at_jax_s_configured_precision():
    """Soft embeddings mix a bf16-stored table by float32 probabilities, a product jax computes in
    float32, at JAX's configured precision, as a float32-stored table's is."""
    logits = jax.random.normal(jax.random.key(0), (2, 3, 32), jnp.float32)
    for stored in (jnp.float32, jnp.bfloat16):
        table = jax.random.normal(jax.random.key(1), (32, 16), jnp.float32).astype(stored)
        precisions = dot_precisions(lambda logits, table: soft_embeddings(logits, table, 1.0), logits, table)
        assert [precision for precision, _, _ in precisions] == [None]


def test_a_projection_factory_keeps_the_contraction_its_caller_gives():
    """A layer built from a caller's projection factory multiplies as that factory says: a bf16 MLP
    whose projections return float32 results (`fp32_result_dot_general`) returns float32."""
    x = jax.random.normal(jax.random.key(0), (2, 16), jnp.bfloat16)
    layer = GatedMLP(hidden_features=24, out_features=16, dtype=jnp.bfloat16,
                     linear=functools.partial(Dense, dot_general=fp32_result_dot_general()))
    assert layer.apply(layer.init(jax.random.key(1), x), x).dtype == jnp.float32


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("precision", [None, jax.lax.Precision.HIGHEST])
def test_xla_attention_multiplies_at_jax_s_configured_precision_or_a_stated_one(dtype, precision):
    """'xla' attention multiplies at JAX's configured precision where the caller states none, float32
    queries as bf16 ones, and every product at HIGHEST where it states that, on whichever path
    computes it: at HIGHEST, or at the dot algorithm of its 16-bit operands' own exact products with fp32
    sums (jax.nn's BF16_BF16_F32 for a bf16 query's logits), which computes HIGHEST for them
    (`exact_products`)."""
    query, key, value = (part.astype(dtype) for part in
                         jax.random.normal(jax.random.key(1), (3, *BF16_QKV), jnp.float32))
    precisions = dot_precisions(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, implementation="xla", precision=precision), query, key, value)
    if precision is None:
        assert_configured_products(precisions)
    else:
        assert precisions and all(stated == HIGHEST or (isinstance(stated, jax.lax.DotAlgorithmPreset)
                                                        and exact_products(stated, operands))
                                  for stated, operands, _ in precisions)


@pytest.mark.parametrize("tied", [True, False])
def test_a_tiled_head_takes_each_tile_as_the_whole_head_takes_its_logits(tied):
    """A chunked head multiplies each vocabulary tile by `bf16_product` over the tile's rows
    (`dew.objectives.lm.chunked`): of a bf16 model those are its forward's logits' columns, bitwise,
    so the chunked head predicts those logits' argmax wherever a tile cuts; its scores are held to
    those logits' cross entropy by check 1 (tests/test_chunked_cross_entropy.py). Under JAX's default
    precision, as a run's default numerics multiply: a run's "highest" leaves the bf16 algorithm, the
    head's and each tile's alike (`operand_precision`)."""
    from dew.nn.precision import bf16_product
    from dew.objectives.lm.chunked import chunked_cross_entropy

    model = CausalTransformer(vocab_size=97, emb_features=32, num_layers=2, num_heads=4, mlp_features=64,
                              max_seq_len=16, tie_embeddings=tied, dtype=jnp.bfloat16)
    with jax.default_matmul_precision(None):
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


@pytest.mark.parametrize("name", ["softplus", "silu", "exp", "exp2"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=str)
@pytest.mark.parametrize("span", [(-88.0, -20.0), (-20.0, -1e-3), (-1e-3, 1e-3), (1e-3, 20.0), (20.0, 88.0)],
                         ids=str)
def test_dews_own_transcendentals_are_as_exact_as_jaxs_across_their_range(name, dtype, span):
    """`dew.nn.precision.softplus` (its log1p by its own series), `silu` (`x / (1 + e^-x)`, no logistic) and
    `accurate_exp` as a TPU takes it (`e^x` and `2^x` by its own sum, traced for an assumed v5e), each past
    the backend's own (row 7062), and their
    derivatives on par with JAX's from float64 on this backend over each span of their range, large negative
    to large positive and near zero, their input in fp32 and bf16 (check 1)."""
    from contract_support import on_a_tpu

    from dew.nn import activations, precision

    def draw(seed):
        x = np.random.default_rng(seed).uniform(*span, 4096).astype(np.float32)
        return (jnp.asarray(x, dtype),)

    ours, theirs = {"softplus": (precision.softplus, jax.nn.softplus),
                    "silu": (activations.silu, jax.nn.silu),
                    "exp": (on_a_tpu(precision.accurate_exp), jnp.exp),
                    "exp2": (on_a_tpu(functools.partial(precision.accurate_exp, base2=True)), jnp.exp2)}[name]
    for part, mine, jaxs in (("value", ours, theirs),
                             ("slope", jax.vmap(jax.grad(ours)), jax.vmap(jax.grad(theirs)))):
        assert_as_exact(jax.jit(mine), jax.jit(jaxs), draw, f"{name}'s {part} {span} {dtype}", seeds=8)


@pytest.mark.tpu(reason="a TPU flushes subnormal floats, which a CPU keeps")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=str)
def test_softplus_keeps_its_tail_on_a_tpu(dtype):
    """Dew's softplus near x = -87 on a TPU, which flushes subnormals: `y = e^x` is a normal float there and
    half of it is not, so a series whose leading term doubled `y / (2 + y)` was zero where float64's softplus
    is a normal float (5895 of 65536 fp32 points over [-87.3, -80] on a v5e, Main's tail review); its
    leading term `y / (1 + y / 2)` is never zero there, and the softplus is on par with JAX's from float64
    (check 1)."""
    from dew.nn import precision

    def draw(seed):
        x = np.random.default_rng(seed).uniform(-87.3, -80.0, 4096).astype(np.float32)
        return (jnp.asarray(x, dtype),)

    (x,) = draw(0)
    normal = np.log1p(np.exp(np.asarray(x, np.float64))) >= float(jnp.finfo(dtype).tiny)
    assert not np.any(np.asarray(jax.jit(precision.softplus)(x))[normal] == 0)
    assert_as_exact(jax.jit(precision.softplus), jax.jit(jax.nn.softplus), draw, f"softplus's tail {dtype}",
                    seeds=8)


def test_a_routes_declaration_is_the_runs_configured_precision_where_the_call_states_none():
    """A run's configured product precision (`Numerics.matmul`, `run_precision`) is each route's declaration
    where a call states none (`dew.nn.precision.effective_precision`): attention hands a call to the
    reference path, which honours it, where whole products of its dtype do not compute the run's
    (`exact_products`: an algorithm narrowing them; a 16-bit or fp32 query at 'highest' keeps the fused
    routes, which multiply whole); a native recurrence keeps an
    fp32 product whole at 'highest' whatever the forms (`rounds_products`), the ports, whose fp32 products
    are TF32 where Triton runs them, take no fp32 call there (off a GPU they interpret, exact under any
    declaration), and the template takes none under an algorithm that narrows its
    operands, each handing the call to XLA, which computes the declaration; a dilated depthwise
    convolution's exact forms take a declaration of exact products alone, not one narrowing the operands;
    and Muon's iteration multiplies in its fast forms in bfloat16 for a 16-bit compute dtype
    (`iteration_dtype`: bf16-mixed's measured default, and 16-mixed's, whose float16 would divide a zero
    update by a zero norm) and in the compute dtype otherwise, float32 under 32-true and float64 under a
    float64 run."""
    from dew.newton_schulz import ns_operands
    from dew.nn import precision
    from dew.nn.attention import reference_only
    from dew.nn.kernels import recurrence_chunk
    from dew.nn.precision import Dtypes, Numerics

    def admits(supports, dtype) -> bool:
        requirement = next(one for one in supports if one.code == "precision")
        return requirement.analysis(SimpleNamespace(dtype=np.dtype(dtype))) is None

    query = jnp.ones((1, 4, 2, 8), jnp.float32)
    ports = (recurrence_chunk.PORT_SUPPORTS, recurrence_chunk.SSD_SUPPORTS)
    # The suite's lanes configure 'highest' (lane_environment); the run here declares the default.
    with generation.run_precision(Numerics("32-true")), jax.default_matmul_precision("default"):
        assert not reference_only(query, None, None, force_fp32_for_softmax=True)
        assert precision.rounds_products() and recurrence_chunk.fast_rounding(jnp.float32)
        assert all(admits(supports, np.float32) for supports in (*ports, recurrence_chunk.TEMPLATE_SUPPORTS))
    with generation.run_precision(Numerics("32-true", matmul="highest")):
        assert not reference_only(query, None, None, force_fp32_for_softmax=True)
        assert not reference_only(query.astype(jnp.bfloat16), None, None, force_fp32_for_softmax=True)
        assert precision.effective_precision(None) == "highest"
        assert not precision.rounds_products() and not recurrence_chunk.fast_rounding(jnp.float32)
        # A 16-bit call's rounded numbers are its intermediates', mamba_ssm's and fla's, its products whole
        # at any precision: the run's fast forms take them at 'highest' too, and its accurate ones never.
        assert recurrence_chunk.fast_rounding(jnp.bfloat16)
        assert recurrence_chunk.chunk_rounding(delta=True, dtype=jnp.bfloat16)
    with generation.run_precision(Numerics("32-true", forms="accurate", matmul="default")):
        assert not recurrence_chunk.fast_rounding(jnp.bfloat16)
        # Where Triton runs the ports, their fp32 products are TF32; off a GPU they interpret, exact under
        # any declaration (`recurrence_chunk._port_precision`).
        with generation.assumed("sm90"):
            assert not any(admits(supports, np.float32) for supports in ports)
            assert all(admits(supports, jnp.bfloat16) for supports in ports)
        assert generation.triton_runs() or all(admits(supports, np.float32) for supports in ports)
        assert admits(recurrence_chunk.TEMPLATE_SUPPORTS, np.float32)
    with generation.run_precision(Numerics("32-true", matmul="TF32_TF32_F32")):
        assert reference_only(query, None, None, force_fp32_for_softmax=True)
        assert not admits(recurrence_chunk.TEMPLATE_SUPPORTS, np.float32)
    # The TPU scan and the selective template widen a 16-bit call's operands and multiply them in fp32,
    # which bf16 parts narrow.
    widening = (recurrence_chunk.TPU_SCAN_SUPPORTS, recurrence_chunk.SELECTIVE_SUPPORTS)
    with generation.run_precision(Numerics("32-true", matmul="BF16_BF16_F32")):
        assert not any(admits(supports, jnp.bfloat16) for supports in widening)
    with generation.run_precision(Numerics("32-true", matmul="highest")):
        assert all(admits(supports, jnp.bfloat16) for supports in widening)
    tf32 = jax.lax.DotAlgorithmPreset.TF32_TF32_F32
    assert precision.exact_products(None, jnp.float32)
    assert precision.exact_products(jax.lax.Precision.HIGHEST, jnp.float32)
    assert precision.exact_products(jax.lax.DotAlgorithmPreset.BF16_BF16_F32, jnp.bfloat16)
    assert not precision.exact_products(tf32, jnp.float32)
    assert not precision.exact_products(jax.lax.DotAlgorithmPreset.BF16_BF16_F32, jnp.float32)
    for numerics, dtype in ((Numerics(), jnp.bfloat16), (Numerics("16-mixed"), jnp.bfloat16),
                            (Numerics("32-true"), jnp.float32),
                            (Numerics(forms="accurate"), jnp.float32),
                            (Numerics(Dtypes(compute="float64")), jnp.float64)):
        with generation.run_precision(numerics):
            assert ns_operands(jnp.float32) == jnp.dtype(dtype), numerics



@pytest.mark.kernel
def test_an_fp32_delta_rules_route_computes_the_runs_declared_products():
    """An fp32 gated delta rule's forward, routed for an H100 under the run's fast forms at each declared
    product precision, takes a route that computes the declaration: at the default, any (the port's TF32
    products, the template's two bf16 parts an operand: `fast_rounding`); at 'highest' and F32_F32_F32 no
    route that narrows them (the port refuses, `_port_precision`), the template's launches multiplying
    whole (its three parts an operand, twice the default's two parts' products, by the FLOPs each launch
    states) where it takes the call; and under a TF32 algorithm XLA alone, which computes it (the template's
    and the port's 'precision' requirements)."""
    from dew.nn.kernels import recurrence
    from dew.nn.kernels.lowered import kernel_calls
    from dew.nn.kernels.recurrence_chunk import Chunk, chunk
    from dew.nn.kernels.templates import product
    from dew.nn.kernels.templates.recurrence import mainloop

    rows, length, heads, width = 1, 128, 2, 64
    operands = [jax.ShapeDtypeStruct((rows, length, heads, width), jnp.float32)] * 3 + [
        jax.ShapeDtypeStruct((rows, length, heads), jnp.float32)] * 2
    hooks = recurrence.Hooks(decay='scalar', scale=width ** -0.5, write_bound=1.0)

    def forward(query, key, value, log_decay, beta):
        return chunk(hooks, Chunk(query, key, value, log_decay=log_decay, beta=beta))[0]

    def routed(matmul: str) -> tuple[set[str], list[float]]:
        mesh = generation.gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
        with (generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh),
              generation.routes() as found, generation.run_precision(Numerics("32-true")),
              jax.default_matmul_precision(matmul)):
            lowered = jax.jit(forward).trace(*operands).lower(lowering_platforms=("cuda",))
        kernels = {route.kernel for route in found if route.choice == "gated_delta_rule"}
        launches = [call.flops or 0.0 for call in kernel_calls(lowered) if call.kind == "mosaic_gpu"]
        return kernels, sorted(launches)

    fast, fast_flops = routed("default")
    whole_terms = len(product.terms(product.EXACT_PARTS, product.EXACT_PARTS))
    rounded_terms = len(product.terms(mainloop.FAST_PARTS, mainloop.FAST_PARTS))
    for matmul in ("highest", "F32_F32_F32"):
        kernels, flops = routed(matmul)
        assert "fla" not in kernels, matmul
        if kernels == {"mosaic_gpu"} and fast == {"mosaic_gpu"}:
            assert flops == [held * whole_terms / rounded_terms for held in fast_flops], matmul
    assert routed("TF32_TF32_F32") == ({"xla"}, [])


def test_a_float64_value_asked_for_without_x64_fails_rather_than_computing_in_float32():
    """A float64 value asked for where x64 is off fails (the suite's filter on JAX's truncation), where JAX
    alone would hand back float32 with a warning: no float64 truth or twin computes in float32 unseen.
    Under x64 the same request is float64."""
    with jax.enable_x64(new_val=False), pytest.raises(UserWarning):
        jnp.ones(3, jnp.float64)
    with jax.enable_x64(new_val=True):
        assert jnp.ones(3, jnp.float64).dtype == jnp.float64


@pytest.mark.parametrize(("source", "target"), [
    (np.float32, np.float64), (jnp.bfloat16, np.float32), (np.float16, np.float32), (np.float16, np.float64),
    (jnp.float8_e4m3fn, np.float32), (jnp.float8_e4m3fnuz, np.float32), (np.float32, np.float32)],
    ids=["fp32-float64", "bf16-fp32", "fp16-fp32", "fp16-float64", "e4m3-fp32", "e4m3fnuz-fp32", "fp32-fp32"])
def test_a_float_widens_exactly_from_its_bits(source, target):
    """`widened_exactly` of a format's edge values of both signs (its subnormals of every leading bit's place
    among them) into a format at least as wide, compiled: NumPy's exact conversion bit for bit, a negative
    zero's sign kept, where XLA:CPU's cast flushes a subnormal (row 8966); an 8-bit float's as its
    conversion gives it, E4M3FNUZ's 0x80 a NaN, not -0 (base review ed11c8e4e F4)."""
    with jax.enable_x64(new_val=target == np.float64):
        values = edge_values(source)
        found = np.asarray(jax.jit(lambda x: widened_exactly(x, target))(jnp.asarray(values)))
    assert found.dtype == target
    expected = values.astype(target)
    word = f"uint{np.finfo(target).bits}"
    np.testing.assert_array_equal(np.isnan(found), np.isnan(expected))
    held = ~np.isnan(expected)
    np.testing.assert_array_equal(found[held].view(word), expected[held].view(word))


@pytest.mark.parametrize("source", [jnp.float4_e2m1fn, jnp.float8_e4m3fnuz, jnp.float8_e5m2fnuz])
def test_a_narrow_float_widens_as_it_converts(source):
    """`widened_exactly` of every code of a narrow float into float32, compiled: its conversion, exact, where
    its bits were read as an IEEE format's: a sub-byte float's (FP4) asked for an unsigned integer of its
    width (base review ed11c8e4e F3), and E4M3FNUZ's and E5M2FNUZ's 0x80 widened to -0, not their NaN (F4).
    FP6's conversion XLA:CPU compiles none of (an invalid i6 bitcast), so its codes are no cell here."""
    values = np.arange(1 << jnp.finfo(source).bits, dtype=np.uint8).view(source)
    found = np.asarray(jax.jit(lambda x: widened_exactly(x, np.float32))(jnp.asarray(values)))
    expected = values.astype(np.float32)
    np.testing.assert_array_equal(np.isnan(found), np.isnan(expected))
    np.testing.assert_array_equal(found[~np.isnan(expected)], expected[~np.isnan(expected)])


@pytest.mark.parametrize(("lanes", "gradient"), [((2.0 ** 140, -(2.0 ** 140)), 0.0),
                                                ((2.0 ** 140, -(2.0 ** 140) + 2.0 ** 100), 2.0 ** 100)],
                         ids=["cancelling", "past float32 alone"])
def test_an_aligned_sums_shared_float32_factor_takes_its_gradient_summed_wide(lanes, gradient):
    """The gradient in a float32 `a` shared by a batch of aligned sums `aligned_sum([(a, b)])` (`jax.vmap`
    over float64 `b` of `2^140` and `-2^140`, or `-2^140 + 2^100`), with x64 on, summed over the batch: 0 and
    `2^100`, each cotangent in float64 through the batch's sum and cast once, where each was cast to float32
    first, infinite, and their sum NaN (base reviews 279405f7f F1, 7eccb9f88 F4)."""
    with jax.enable_x64(new_val=True):
        b = jnp.asarray(lanes, jnp.float64)
        found = jax.grad(lambda a: jnp.sum(jax.vmap(lambda b: aligned_sum([(a, b)]))(b)))(jnp.float32(1.0))
    assert found.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(found), np.float32(gradient))


def test_an_explicit_float64_widens_as_jax_resolves_it():
    """`widened_exactly` of a float64 `2^140` to float64 with x64 off and JAX's explicit 64-bit dtypes allowed
    (`jax_explicit_x64_dtypes`): the float64 itself, the destination as JAX resolves the request, where
    read as x64's default policy it narrowed to float32's infinity (base review 7eccb9f88 F1)."""
    held = jax.config.jax_explicit_x64_dtypes
    jax.config.update("jax_explicit_x64_dtypes", "allow")
    try:
        x = jnp.asarray(2.0 ** 140, dtype=np.float64)
        found = widened_exactly(x, np.float64)
    finally:
        jax.config.update("jax_explicit_x64_dtypes", held)
    assert found.dtype == np.float64 and float(found) == 2.0 ** 140


def test_a_wide_cpu_reduction_widens_its_subnormal_operands_exactly(monkeypatch):
    """`widened_reduction`'s sum, under Dew's wide CPU arithmetic, of float32's two largest subnormals (built
    from their bits): their sum `2^-125 - 2^-148`, a normal float32, each widened from its bits where
    XLA:CPU's cast flushed them and the sum was 0 (base review 7eccb9f88 F2)."""
    from dew.nn.precision import WIDE_CPU_ARITHMETIC, widened_reduction

    monkeypatch.setenv(WIDE_CPU_ARITHMETIC, "1")
    largest = jnp.asarray(np.full((2,), 0x7FFFFF, np.uint32).view(np.float32))
    with jax.default_device(jax.devices("cpu")[0]):
        (found,) = jax.jit(lambda x: widened_reduction(lambda x: (jnp.sum(x),), x))(largest)
    assert found.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(found), np.float32(2.0 ** -125 - 2.0 ** -148))


def test_an_exact_widenings_transpose_keeps_its_sources_type_outside_x64():
    """The transpose of a batch's exact widening (`jax.vmap` of `widened_exactly` of a weak float32 to
    float64, traced with x64 on), its cotangents `2^140` and `-2^140 + 2^100` pulled back with x64 off:
    the weak float32 `2^100`, the batch summed in float64 and cast once, where summed as x64's default
    narrowed them first, `inf - inf`, NaN, and the cast's strong result dropped the source's weakness (base
    review of 961433795 F3, F6)."""
    # A weak float32: a Python float's array with x64 off.
    weak = jnp.asarray(1.0)
    assert weak.dtype == jnp.float32 and jax.typeof(weak).weak_type
    with jax.enable_x64(new_val=True):
        transposed = jax.linear_transpose(
            lambda x: jax.vmap(lambda _: widened_exactly(x, jnp.float64))(jnp.zeros(2)), weak)
        cotangent = jnp.asarray([2.0 ** 140, -(2.0 ** 140) + 2.0 ** 100], jnp.float64)
    (pulled,) = transposed(cotangent)
    assert pulled.dtype == jnp.float32 and float(pulled) == 2.0 ** 100
    assert jax.typeof(pulled).weak_type


@pytest.mark.mesh(devices=2)
def test_an_exact_widenings_transpose_keeps_its_sources_placement():
    """The gradient through `widened_exactly` of a float32 split over its rows by an Explicit axis of two
    devices, widened to float64 under x64: the float32 ones, placed as the source is, where the transpose's
    source rebuilt from its shape and dtype alone lost the placement (review of cbb640145 F1)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    mesh = jax.make_mesh((2,), ("rows",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    split = NamedSharding(mesh, PartitionSpec("rows", None))
    with jax.set_mesh(mesh), jax.enable_x64(new_val=True):
        x = jax.device_put(np.ones((4, 3), np.float32), split)
        gradient = jax.jit(jax.grad(lambda x: jnp.sum(widened_exactly(x, jnp.float64))))(x)
    assert gradient.dtype == jnp.float32 and jax.typeof(gradient).sharding.spec == PartitionSpec("rows", None)
    np.testing.assert_array_equal(np.asarray(gradient), np.ones((4, 3), np.float32))


def test_an_exact_power_of_two_past_float32s_range_is_its_field():
    """`exact_power2` in float32 at `minexp - 1`, `maxexp` and `maxexp + 1`: +0, infinity and the sign bit
    alone, the field as the word holds it, which a quantizer's ceiling of a zero, an infinity and a NaN
    reads (`fake_quant._power_of_two_ceil`: 0.0, inf, -0.0; review of e71e8066f F7)."""
    from dew.nn import fake_quant
    from dew.nn.float_bits import exact_power2

    found = np.asarray(exact_power2(jnp.asarray([-127, 128, 129], jnp.int32), jnp.float32))
    np.testing.assert_array_equal(found.view(np.uint32), np.uint32([0, 0x7F800000, 0x80000000]))
    ceiling = np.asarray(fake_quant._power_of_two_ceil(jnp.asarray([0.0, np.inf, np.nan], jnp.float32)))
    np.testing.assert_array_equal(ceiling.view(np.uint32), np.uint32([0, 0x7F800000, 0x80000000]))


def test_a_float64_sum_widens_its_subnormal_operands_exactly():
    """`float64_sum`, Dew's wide CPU arithmetic's sum, of fp32 operands below the least normal: `2^-149 2^127`
    and `(2^-126 - 2^-149) 2^127` taken in float64 and rounded once to fp32, `2^-22` and `2 - 2^-22`, where
    widened by a cast XLA:CPU flushed them and each product was 0 (rows 8966, 9780)."""
    lhs = jnp.asarray(np.asarray([[1], [0x7FFFFF]], np.uint32).view(np.float32))
    found = float64_sum(jnp.matmul, lhs, jnp.asarray([[2.0 ** 127]], jnp.float32), dtype=jnp.float32)
    assert found.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(found), np.float32([[2.0 ** -22], [2.0 - 2.0 ** -22]]))


def test_a_host_scalar_widens_at_its_destination():
    """`widened_exactly` of a Python integer past int32 (`2^40`) to float32 with x64 off is float32's
    `2^40`, converted at its destination, where staging it as int32 first raised (base review 279405f7f
    F2)."""
    found = widened_exactly(2 ** 40, jnp.float32)
    assert found.dtype == jnp.float32 and float(found) == 2.0 ** 40


def test_an_exact_widening_transposes_as_a_cast():
    """`jax.linear_transpose` of `widened_exactly` from bfloat16 to float32 at a float32 cotangent of 1:
    bfloat16's 1, the cotangent cast back, where the widening's select on its operand's bits took no
    transpose (base review 20ad463a7 F1)."""
    transposed = jax.linear_transpose(lambda x: widened_exactly(x, jnp.float32), jnp.bfloat16(1))
    (pulled,) = transposed(jnp.float32(1))
    assert pulled.dtype == jnp.bfloat16 and float(pulled) == 1.0


def test_an_exact_widening_is_strongly_typed():
    """`widened_exactly` of a weakly typed float32 `2^-25` to float64, added to a strong float32 1 inside
    jit, with x64 on: float64's `1 + 2^-25`, the widening's type strong as an explicit conversion's, where a
    weak float64 yielded to the strong float32 and the sum rounded to 1 (base review 7fa8fe6c2 F2)."""
    weak = jnp.asarray(2.0 ** -25)
    assert weak.weak_type and weak.dtype == jnp.float32
    with jax.enable_x64(new_val=True):
        found = jax.jit(lambda w: widened_exactly(w, jnp.float64) + jnp.float32(1.0))(weak)
    assert found.dtype == jnp.float64 and float(found) == 1.0 + 2.0 ** -25


def test_a_wide_products_cotangent_widens_exactly(monkeypatch):
    """The gradient of a wide CPU product (`accumulated`, Dew's wide arithmetic) of float32 `[[1]]` and
    `[[2^127]]` at the subnormal cotangent `2^-149` (built from its bits): `2^-22`, the cotangent widened
    from its bits into the float64 sum, where XLA:CPU's cast flushed it and the gradient was 0 (base review
    7fa8fe6c2 F3)."""
    from dew.nn.precision import WIDE_CPU_ARITHMETIC, accumulated

    monkeypatch.setenv(WIDE_CPU_ARITHMETIC, "1")
    rhs = jnp.asarray([[2.0 ** 127]], jnp.float32)
    _, pullback = jax.vjp(lambda lhs: accumulated(jnp.matmul, lhs, rhs), jnp.ones((1, 1), jnp.float32))
    (gradient,) = pullback(jnp.asarray(np.asarray([[1]], np.uint32).view(np.float32)))
    np.testing.assert_array_equal(np.asarray(gradient), np.float32([[2.0 ** -22]]))


def test_a_compensated_sum_over_axes_of_one_element_is_its_values():
    """`compensated_sum` over an axis of one element, alone or beside a longer one, and the compensated pair
    reduction under it (`summed_pairs`): the values themselves, both limbs of a pair as they are, with no
    reduction of pairs left in the program, and beside a longer axis the same sum as the reduction over that
    axis alone, bitwise, where XLA:GPU's simplifier rewrote a pair reduction over an axis of one element into
    one array's bitcast and refused its program (row 10674; the GPU's own run in test_gpu_models'
    users-process cells). An axis out of range or a pair of two shapes is refused."""
    from reference_error import equations

    from dew.nn.precision import compensated_sum, summed_pairs

    rng = np.random.default_rng(10674)
    x = jnp.asarray(rng.standard_normal((32, 1, 5)), jnp.float32)
    # Each low limb under half its high one's unit, so the pair is renormalized as it is.
    low = x * jnp.asarray(rng.uniform(-1.0, 1.0, (32, 1, 5)) * 2.0 ** -30, jnp.float32)
    found = jax.jit(lambda x: compensated_sum(x, (1,)))(x)
    np.testing.assert_array_equal(np.asarray(found), np.asarray(x[:, 0]))
    alone = jax.jit(lambda x: compensated_sum(x, (1,)))(x[:, 0])
    beside = jax.jit(lambda x: compensated_sum(x, (1, 2)))(x)
    np.testing.assert_array_equal(np.asarray(beside), np.asarray(alone))
    hi, lo = jax.jit(lambda x, low: summed_pairs((x, low), (1,)))(x, low)
    np.testing.assert_array_equal(np.asarray(hi), np.asarray(x[:, 0]))
    np.testing.assert_array_equal(np.asarray(lo), np.asarray(low[:, 0]))
    traced = jax.make_jaxpr(lambda x, low: summed_pairs((x, low), (1,)))(x, low)
    assert not [eqn for eqn in equations(traced) if eqn.primitive is jax.lax.reduce_p]
    for axes, pair in (((3,), (x, low)), ((1,), (x, low[:1]))):
        with pytest.raises(ValueError):
            summed_pairs(pair, axes)


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_a_gated_norms_gate_gradient_is_a_number_where_its_exp_overflows(activation):
    """A gated norm's gate on JAX's derivative (the portable path, `norms.gated_rms_normalized`) at gates of
    -100 and -1000, where `e^-gate` overflows fp32: every gradient a number, as the one logistic's declared
    tangent gives it (`activations.GATE_ACTIVATIONS`), where JAX's derivative of the logistic's arithmetic
    made the gate's zero times infinity. Its values are the gated norm's own check 1's
    (tests/test_norm_swiglu.py)."""
    from dew.nn.norms import gated_rms_normalized

    x = jnp.asarray(np.random.default_rng(7609).standard_normal((2, 8)), jnp.float32)
    gate = jnp.asarray([[-100.0, -1000.0, -3.0, 0.5, 17.0, -100.0, 2.0, -50.0]] * 2, jnp.float32)
    scale = jnp.ones((8,), jnp.float32)

    def loss(x, gate):
        return jnp.sum(gated_rms_normalized(x, gate, scale, 1e-6, x.dtype, activation, implementation="xla"))

    grads = jax.grad(loss, argnums=(0, 1))(x, gate)
    assert all(np.all(np.isfinite(np.asarray(grad))) for grad in grads)


@pytest.mark.parametrize("gate", [17.0, 20.0, -17.0])
def test_a_kernels_logistic_slope_is_the_declared_tangents_past_its_value_rounding_to_one(gate):
    """A kernel's slope of the one logistic (`precision.activation_and_slope`, the recurrence conv's
    backward) is the declared tangent's, `s(x) s(-x)` (`activations.logistic`'s, as JAX differentiates it),
    bit for bit, and a number past zero at an fp32 gate of 17, where `s(x)` rounds to one and `s (1 - s)`
    made the slope zero; silu's is `s + g s(x) s(-x)`, the product rule over it. Their values against
    float64 are the logistic's contract's (`test_silu_holds_every_order`)."""
    from dew.nn.activations import logistic
    from dew.nn.precision import activation_and_slope, logistic_value

    value = jnp.asarray([gate], jnp.float32)
    declared = jax.grad(lambda g: logistic(g)[0])(value)
    _, slope = activation_and_slope(value, "sigmoid")
    np.testing.assert_array_equal(np.asarray(slope), np.asarray(declared))
    product = logistic_value(value) * logistic_value(-value)
    np.testing.assert_array_equal(np.asarray(slope), np.asarray(product))
    assert float(slope[0]) > 0
    _, silu_slope = activation_and_slope(value, "silu")
    np.testing.assert_array_equal(np.asarray(silu_slope), np.asarray(logistic_value(value) + value * slope))


def _ones(digits: int, exponent: int) -> float:
    """A float of `digits` significant bits all ones, times `2^exponent`."""
    return (2 ** digits - 1) / 2 ** (digits - 1) * 2.0 ** exponent


@pytest.mark.parametrize(("dtype", "left", "right"), [
    (jnp.float32, _ones(24, 60), _ones(24, 59) - 2.0 ** (59 - 22)),
    (jnp.float64, _ones(53, 500), _ones(53, 499) - 2.0 ** (499 - 51)),
    (jnp.float64, _ones(53, 1023), _ones(53, -1) - 2.0 ** (-1 - 51)),
    (jnp.float16, _ones(11, 15), _ones(11, -1) - 2.0 ** (-1 - 9)),
    (jnp.bfloat16, _ones(8, 60), _ones(8, 59) - 2.0 ** (59 - 6)),
    (jnp.float64, _ones(53, 511), _ones(53, 511)),
    (jnp.float16, _ones(11, 7), _ones(11, 7)),
], ids=["fp32 edge", "float64 edge", "float64 top", "float16 top", "bfloat16 edge", "float64 top product",
        "float16 top product"])
def test_an_exact_product_at_a_dtypes_edge_keeps_its_residual(dtype, left, right):
    """`precision.two_product`'s range-safe split (`bit_split`) at the top of each dtype's range, its
    operands' significands all ones (and two less): their product and its residual sum to the exact
    rational product, a product in the top binade's too.
    A Veltkamp split multiplies by `2^(p/2) + 1` first and overflows to NaN (past 2^115 in fp32, 2^996 in
    float64; base reviews 9bc4326dd F1, 1fe546a10 F1); an odd significand's truncated cut left its low parts'
    product past `p` bits, rounded (float64's 27 and 26 bits; the precision owner's review); the top binade's
    rounded cut carried past the largest float, and two cuts rounded up a product in it ((2 - 2^-52)
    2^511 squared, 255.875 squared in float16: train 1018's review F1)."""
    from dew.nn.precision import two_product

    with jax.enable_x64(new_val=dtype == jnp.float64):
        a, b = jnp.asarray(left, dtype), jnp.asarray(right, dtype)
        product, residual = two_product(a, b, bit_split=True)
        exact = Fraction(float(a)) * Fraction(float(b))
        assert Fraction(float(product)) + Fraction(float(residual)) == exact


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.bfloat16, jnp.float16, jnp.float8_e4m3fn,
                                   jnp.float8_e5m2, jnp.float6_e2m3fn, jnp.float6_e3m2fn, jnp.float4_e2m1fn])
def test_an_exact_power_of_two_is_set_in_its_formats_exponent_bits(dtype):
    """`dew.nn.float_bits.exact_power2` is `2^n` in its format, bit for bit, at every normal exponent from the
    least to the largest, and +0 one below them (a zero a caller encodes so), compiled, a sub-byte format's
    too (FP6, FP4: base review d47c0d69c F1); an exponent of a narrow integer type is its value, where an
    unsigned 2-bit 1 read as the E3M2 sentinel -3's wrap and gave +0 (base review 1e030145c F1)."""
    from dew.nn.float_bits import exact_power2

    # Each exponent type's values: int32's every one here, int4's and an unsigned 2-bit's narrow.
    held = {jnp.int32: (-(2 ** 31), 2 ** 31 - 1), jnp.int4: (-8, 7), jnp.uint2: (0, 3)}
    with jax.enable_x64(new_val=dtype == jnp.float64):
        limits = jnp.finfo(dtype)
        exponents = np.arange(limits.minexp - 1, limits.maxexp)
        taken = {exponent: (low <= exponents) & (exponents <= high) for exponent, (low, high) in held.items()}
        found = {exponent: np.asarray(jax.jit(lambda n: exact_power2(n, dtype))(
            jnp.asarray(exponents[within], exponent))) for exponent, within in taken.items()}
    expected = np.asarray([0.0, *(2.0 ** int(n) for n in exponents[1:])], np.float64).astype(dtype)
    # A sub-byte format's value is held in a byte (ml_dtypes'): its storage's bits.
    stored = f"uint{8 * np.dtype(dtype).itemsize}"
    for exponent, within in taken.items():
        assert found[exponent].dtype == dtype, exponent
        np.testing.assert_array_equal(found[exponent].view(stored), expected[within].view(stored),
                                      err_msg=str(exponent))


def test_an_exact_power_of_two_in_e8m0_is_every_biased_exponent():
    """`exact_power2` in E8M0 (an MX block's scale) is `2^n` at every exponent from its least, `2^-127`, its
    biased exponent 0 a power and no zero, to its largest, `2^127`, compiled, bit for bit the byte MXFP8's
    encode wrote as `n + 127` (row 8832)."""
    from dew.nn.float_bits import exact_power2

    exponents = np.arange(-127, 128)
    found = np.asarray(jax.jit(lambda n: exact_power2(n, jnp.float8_e8m0fnu))(
        jnp.asarray(exponents, jnp.int32)))
    assert found.dtype == jnp.float8_e8m0fnu
    np.testing.assert_array_equal(found.view(np.uint8), (exponents + 127).astype(np.uint8))
    np.testing.assert_array_equal(found.astype(np.float64), np.exp2(exponents.astype(np.float64)))


def power2_rounds_a_subnormal_result_once():
    """`dew.nn.precision.power2` by `2^o 2^(k-2o) 2^o` rounds a result below the least normal once, at its
    last factor, on a device that keeps subnormals (a GPU's XLA, whose fp32 arithmetic keeps them where
    XLA:CPU's flushes them): `(0.5 + 2^-24) 2^-149` is `2^-149`, where a whole factor first rounded the
    partial product to `2^-127` and the remainder's then rounded its tie to zero (base review 54bb1f61a F2).
    Its pullback applies the factors in reverse and rounds once too, a broadcast operand's cotangent summed
    after each term is scaled: the cotangent `0.5 + 2^-24` to `1` by `2^-149` is `2^-149`, and to a scalar by
    two such terms `2^-148`, where the remainder applied last rounded it to zero (base review 44e282c4f F1).
    A GPU numerics cell (tests/test_gpu_numerics.py), one program prepared as an arm is in a GPU tier part
    (`ahead_arm`, base review 85b5535a6 F4)."""
    from reference_error import ahead_arm

    from dew.nn.precision import power2

    # Each exponent given (traced) and fixed (a constant the compiler folds its factors of): a scalar and
    # two broadcast against the operand (base review 3e560b112, row 9469).
    fixed = (np.int32(-149), np.full((2,), -149, np.int32))

    def scaled(x, one, cotangent, exponents):
        exponents = (exponents[0], exponents, *fixed)
        pulled = [jax.vjp(lambda y, k=k: power2(y, k), one)[1](jnp.broadcast_to(cotangent, np.shape(k)))[0]
                  for k in exponents]
        return power2(x, exponents[0]), power2(x, fixed[0]), *pulled

    tiny = np.float32(0.5 + 2.0 ** -24)
    found = ahead_arm(jax.jit(scaled), "a power of two's scaling and its pullback")(
        tiny, np.float32(1.0), tiny, np.full((2,), -149, np.int32))
    # Each one correctly rounded answer: the least subnormal's bits, twice it summed.
    for value, bits in zip(found, (1, 1, 1, 2, 1, 2), strict=True):
        value = np.asarray(value)
        assert value.dtype == np.float32 and value.view(np.uint32) == bits, value.view(np.uint32)


def test_a_python_integer_exponent_past_float32_saturates():
    """`dew.nn.precision.power2` holds a Python integer exponent within its reach before converting it:
    `2^(±10^400)` of `1` is infinite and zero, where `10^400` raised in its host conversion to float32 (base
    review 44e282c4f F2)."""
    from dew.nn.precision import power2

    found = [np.asarray(power2(np.float32(1.0), sign * 10 ** 400)) for sign in (1, -1)]
    assert [value.dtype for value in found] == [np.float32] * 2
    assert np.isposinf(found[0]) and found[1] == 0, found


def test_an_exact_power_of_two_in_a_python_float_is_jaxs_float():
    """`exact_power2` of a Python `float` destination is JAX's float (float32 where x64 is off), its layout
    that dtype's: `2^0` is float32's 1.0, the sentinel below the normals (`2^-127`) its zero (base review
    43f53b12f F1)."""
    from dew.nn.float_bits import exact_power2

    found = np.asarray(jax.jit(lambda n: exact_power2(n, float))(jnp.asarray([0, 3, -126, -127], jnp.int32)))
    assert found.dtype == np.float32
    np.testing.assert_array_equal(found, np.asarray([1.0, 8.0, 2.0 ** -126, 0.0], np.float32))


def test_a_stated_reach_past_two_normal_halves_scales_a_large_operand_exactly():
    """`dew.nn.precision.power2` at a reach past `TWO_FACTOR_REACH` (254, float32's twice its largest
    exponent) takes `2^127` and `1.5 2^127` by `2^-253` to the normal `2^-126` and `1.5 2^-126` exactly,
    compiled: two halves there hold a -127, no normal power, whose encoding is +0, and gave zero."""
    from dew.nn.precision import power2

    found = np.asarray(jax.jit(lambda x, k: power2(x, k, reach=2 * 127))(
        jnp.asarray([2.0 ** 127, 1.5 * 2.0 ** 127, 2.0 ** 127], jnp.float32), jnp.asarray([-253, -253, 254])))
    np.testing.assert_array_equal(found, np.asarray([2.0 ** -126, 1.5 * 2.0 ** -126, np.inf], np.float32))


@pytest.mark.parametrize(("dtype", "reach", "exponent"), [
    (jnp.float32, None, jnp.int32), (jnp.float32, TWO_FACTOR_REACH, jnp.int32),
    (jnp.float32, None, jnp.int8), (jnp.float32, TWO_FACTOR_REACH, jnp.int8), (jnp.float32, None, jnp.uint8),
    (jnp.float32, None, jnp.float32), (jnp.float64, None, jnp.int32), (jnp.bfloat16, None, jnp.int32),
    (jnp.float8_e4m3fn, None, jnp.int32), (jnp.float8_e5m2, TWO_FACTOR_REACH, jnp.int32)])
def test_a_power_of_two_scales_exactly_under_jit_at_every_exponent(dtype, reach, exponent):
    """`dew.nn.precision.power2` (the one owner of scaling by a power of two: a kernel template's, a scaled
    value's, an aligned sum's) scales by powers of two exactly wherever the operand and the result are
    normal values of its format (`x`'s, at least float32), compiled (where XLA computes `exp2` as an inexact
    `exp` and fuses a product of them into one `exp`, past the format), at every integral exponent its type
    holds past the format's range both ways against exact rational scaling, equal parts by default and a
    stated reach's two halves within it, operands at both ends of the normal range among them, an 8-bit
    float's in float32 (`at_least_fp32`, which no implicit promotion reaches; base review a7c5428ef F1); zero
    stays zero, and a result past the format is infinite."""
    from dew.nn.precision import at_least_fp32, power2

    with jax.enable_x64(new_val=dtype == jnp.float64):
        wide = at_least_fp32(dtype)
        limits = np.finfo(wide)
        span = int(limits.maxexp - limits.minexp + limits.nmant) + 8 if reach is None else reach
        held = np.iinfo(exponent) if jnp.issubdtype(exponent, jnp.integer) else np.finfo(exponent)
        exponents = np.arange(max(-span, int(held.min)), min(span, int(held.max)) + 1)
        operands = [0.0, 0.5, 0.75, 1.0, -0.6, 1.0 + 2.0 ** -7, float(jnp.finfo(dtype).tiny),
                    -1.5 * 2.0 ** (jnp.finfo(dtype).maxexp - 2)]
        found = np.asarray(jax.jit(lambda x, k: power2(x[:, None], k[None, :], reach))(
            jnp.asarray(operands, dtype), jnp.asarray(exponents, exponent)))
    assert found.dtype == wide
    taken = np.asarray(np.asarray(operands, np.float64).astype(dtype), np.float64)
    exact = [[Fraction(float(x)) * Fraction(2) ** int(k) for k in exponents] for x in taken]
    normal = np.asarray([[value == 0 or limits.tiny <= abs(value) <= limits.max for value in row]
                         for row in exact])
    expected = np.asarray([[float(value) if normal[i, j] else 0.0 for j, value in enumerate(row)]
                           for i, row in enumerate(exact)], wide)
    np.testing.assert_array_equal(found[normal], expected[normal])
    np.testing.assert_array_equal(found[0], 0.0)
    past = np.asarray([[abs(value) > limits.max for value in row] for row in exact])
    assert np.all(np.isinf(found[past]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, jnp.bfloat16, jnp.float16, jnp.float8_e4m3fn,
                                   jnp.float8_e5m2])
def test_a_floats_raw_fields_are_its_bits(dtype):
    """`raw_fields` of a format's edge values of both signs, compiled: its bits as the unsigned integer of its
    width, its sign bit in place, its biased exponent and its fraction, each NumPy's reading of the same
    bits, and the three fields recompose the bits."""
    with jax.enable_x64(new_val=dtype == np.float64):
        values = edge_values(dtype)
        bits, sign, biased, fraction = (np.asarray(field)
                                        for field in jax.jit(raw_fields)(jnp.asarray(values)))
    limits = jnp.finfo(dtype)
    word = np.dtype(f"uint{limits.bits}")
    expected = values.view(word)
    assert {field.dtype for field in (bits, sign, biased, fraction)} == {word}
    np.testing.assert_array_equal(bits, expected)
    np.testing.assert_array_equal(sign, expected & word.type(1 << (limits.bits - 1)))
    # The exponent field's own width (`nexp`): E4M3's top binade holds finite values.
    np.testing.assert_array_equal(biased, (expected >> limits.nmant) & word.type((1 << limits.nexp) - 1))
    np.testing.assert_array_equal(sign | (biased << limits.nmant) | fraction, expected)


def test_e8m0s_raw_fields_hold_no_sign():
    """`raw_fields` of E8M0 (an MX block's scale, unsigned, no fraction): no sign bit, every bit its biased
    exponent, where the sign's mask read its top exponent bit (base review ed11c8e4e F3)."""
    codes = np.arange(255, dtype=np.uint8)
    bits, sign, biased, fraction = (np.asarray(field) for field in jax.jit(raw_fields)(
        jnp.asarray(codes.view(jnp.float8_e8m0fnu))))
    np.testing.assert_array_equal(bits, codes)
    np.testing.assert_array_equal(sign, 0)
    np.testing.assert_array_equal(biased, codes)
    np.testing.assert_array_equal(fraction, 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["fp32", "float64"])
def test_a_floats_bits_read_as_its_exponent_and_significand(dtype):
    """`float_bits` of normal, subnormal and zero values of both signs: each nonzero value's exponent
    `floor(log2 |x|)` (numpy's `frexp`, less one) and its normalized fraction with the implicit bit the
    value's significand, the value exactly their product, a subnormal's read from its bits."""
    with jax.enable_x64(new_val=dtype == np.float64):
        values = edge_values(dtype)
        _, _, _, normalized, exponent = (np.asarray(part) for part in float_bits(jnp.asarray(values)))
    nonzero = values != 0
    expected = np.frexp(values.astype(np.float64))[1] - 1
    np.testing.assert_array_equal(exponent[nonzero], expected[nonzero])
    nmant = np.finfo(dtype).nmant
    significand = (normalized.astype(np.float64) + 2.0 ** nmant) / 2.0 ** nmant
    rebuilt = np.copysign(np.ldexp(significand, exponent.astype(np.int64)), values.astype(np.float64))
    np.testing.assert_array_equal(rebuilt[nonzero], values.astype(np.float64)[nonzero])


TWO_PI_SPLIT = (np.float32(6.2831855), np.float32(-1.7484555e-07))
"""2π as a float32 and the float32 nearest what it dropped: a constant a broadcast multiply reads."""


def _compensated_shapes():
    """The shapes an error-free transform's operand is made in under one jit, each a function of drawn
    `(x, y, z)` float32 rows: a two_product of a broadcast constant fed into a compensated sum (an angle less
    its whole turns, Mamba-3's, the shape that split one value into its exact and rounded readings), a
    two_product into a two_sum, a plain multiply by a broadcast constant into a two_sum, and one into a
    compensated sum's pair."""
    from dew.nn.precision import compensated_add, two_product, two_sum

    def turned(x, y, z):
        del y
        # The whole turns of the angle's float32 product, a stated rounding at every width (the exact
        # product of two float32 values rounded once is float32's own), so the float64 truth reduces by the
        # same ones.
        turns = jnp.floor((x * np.float32(1 / (2 * np.pi))).astype(jnp.float32).astype(x.dtype))
        whole, dropped = two_product(-turns, jnp.full_like(turns, TWO_PI_SPLIT[0]))
        return compensated_add((x, z * np.float32(2 ** -30)), (whole, dropped - turns * TWO_PI_SPLIT[1]))

    def product_into_sum(x, y, z):
        return two_sum(two_product(x, y)[0], z)

    def scaled_into_sum(x, y, z):
        del y
        return two_sum(x * jnp.full_like(x, TWO_PI_SPLIT[0]), z)

    def scaled_into_pair(x, y, z):
        return compensated_add((x * jnp.full_like(x, TWO_PI_SPLIT[0]), jnp.zeros_like(x)), (z, y * z))

    def multiplied_into_product(x, y, z):
        return two_product(x * y, z, bit_split=True)

    return {"turned": turned, "product into sum": product_into_sum, "scaled into sum": scaled_into_sum,
            "scaled into pair": scaled_into_pair, "multiplied into product": multiplied_into_product}


COMPENSATED_SHAPES = tuple(_compensated_shapes())
"""Each of `_compensated_shapes`' names, the cases every platform's cell runs."""


def compensated_primitives_hold_under_jit(shape: str) -> None:
    """`_compensated_shapes`' `shape` jitted, its pair's high and low parts each on par with its eager run's
    from float64 (check 1) on the process's default device: eagerly each operation is its own program,
    rounded on its own, the arithmetic an error-free transform is stated in; under jit a compiler that
    contracted a multiply into the sum reading it (an FMA, XLA's CPU and GPU) read the exact product where
    the transform's later steps read its rounded one, and the compensated sum lost what it carried (a turn's
    high part thousands of times its eager error on a CPU, 1.9e-6 off in Mamba-3). A contraction that only
    rounds a low part's own arithmetic once keeps the pair as exact as eager. The precision P0 of
    2026-11-18."""
    from arm_forms import as_given

    function = _compensated_shapes()[shape]

    def draw(seed):
        rng = np.random.default_rng([2026_1118, seed])
        return (jnp.asarray(rng.uniform(-3e4, 3e4, 4096), jnp.float32),
                jnp.asarray(rng.standard_normal(4096), jnp.float32),
                jnp.asarray(rng.standard_normal(4096) * 1e-3, jnp.float32))

    # The reference run as given, op by op (`as_given`); the gate compiles Dew's as its users do.
    assert_as_exact(function, as_given(function), draw, f"a compensated {shape} under jit", seeds=8)


@pytest.mark.tpu(reason="a TPU's XLA programs, which keep the error-free transforms' barrier")
@pytest.mark.parametrize("shape", COMPENSATED_SHAPES)
def test_a_compensated_primitive_under_jit_on_a_tpu_is_its_eager_arithmetic(shape):
    """`compensated_primitives_hold_under_jit` on a TPU, its XLA programs' (a compiled Pallas TPU kernel's
    body takes the barrier once the fork's post111 lowers it there, row 7954)."""
    compensated_primitives_hold_under_jit(shape)


PATHS = ("marked", "barriered")
"""The error-free transforms' two paths (row 8378): their arithmetic marked to round alone where the jaxlib
marks (`dew.nn.precision.rounding_alone`), or their operands behind a barrier (`dew._cache.BARRIERED`, which
the suite enters to run that path on a lane whose jaxlib marks)."""


def on_path(path: str):
    """The block run on `path` (`PATHS`): the barrier path entered, the marked one as the process runs."""
    from dew import _cache

    return _cache.BARRIERED(new_value=True) if path == "barriered" else contextlib.nullcontext()


@pytest.mark.parametrize("path", PATHS)
def test_a_sum_of_products_rounded_once_cancels_a_product_made_in_its_program(path):
    """`summed_once` of `-q` and `1 * q`, `q` a square made in the same jitted program (`t = 1 + 2^-12`, whose
    square's rounding drops `2^-24`), is 0 exactly: its products' splits read each operand as one value,
    where a compiler that contracted the square into a split's subtraction read the exact square there and
    the rounded one in the product, a spurious low part of `2^-24` the sum carried (row 7953). On each path
    (`PATHS`): the caller's square, unmarked, is read by the transform's marked arithmetic, which rounds alone
    (row 8378)."""
    from dew.nn.precision import summed_once

    @jax.jit
    def cancelled(t):
        square = t * t
        return summed_once((-square,), ((jnp.ones_like(square), square),))

    t = jnp.full((4096,), 1 + 2.0 ** -12, jnp.float32)
    with on_path(path):
        np.testing.assert_array_equal(np.asarray(cancelled(t)), np.zeros(4096, np.float32))


def test_a_sum_of_products_rounded_once_reaching_an_infinity_is_the_plain_sums():
    """`summed_once` whose sum overflows float32, by its values or by a product, is the plain sum's
    infinity, as `jnp.sum`'s is, and a finite sum beside it is its own: a residual past an infinity is
    inf - inf, which added back would make it NaN (row 8453)."""
    from dew.nn.precision import summed_once

    largest = jnp.finfo(jnp.float32).max
    values = jnp.asarray([largest, -largest, 1.0], jnp.float32)
    summed = jax.jit(lambda values: summed_once((values, values)))(values)
    np.testing.assert_array_equal(np.asarray(summed), np.asarray([np.inf, -np.inf, 2.0], np.float32))
    scaled = jax.jit(lambda values: summed_once((values,), ((values, jnp.full_like(values, 2.0)),)))(values)
    np.testing.assert_array_equal(np.asarray(scaled), np.asarray([np.inf, -np.inf, 3.0], np.float32))


def test_a_sum_of_products_rounded_once_keeps_its_residuals_apart_from_its_running_value():
    """`summed_once` carries each rounding's residual apart and adds them back at the end (Dot2): of
    `M, 2^102, 2^102, -M` (`M` float32's largest) it is `2^103`, where residuals renormalized into the
    running value as they come rounded it past `M` to an infinity, and of `-2^-24 + 1 * 2^-49 + t * t - fl(t
    t)` (`t = 1 + 2^-12`) it is `2^-49`, where the residual a later one cancels was rounded away first (row
    8453's review)."""
    from dew.nn.precision import summed_once

    largest, big = np.finfo(np.float32).max, np.float32(2.0 ** 102)
    overflowing = jax.jit(lambda values: summed_once(tuple(values)))(
        jnp.asarray([largest, big, big, -largest], jnp.float32))
    assert float(overflowing) == 2.0 ** 103
    t = np.float32(1 + 2.0 ** -12)

    @jax.jit
    def cancelled(t):
        return summed_once((jnp.float32(-(2.0 ** -24)), -(t * t)), ((jnp.float32(1), jnp.float32(2.0 ** -49)),
                                                                     (t, t)))

    assert float(cancelled(jnp.asarray(t))) == 2.0 ** -49


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("shape", COMPENSATED_SHAPES)
def test_a_compensated_primitive_under_jit_is_its_eager_arithmetic(shape, path):
    """`compensated_primitives_hold_under_jit` on the CPU (a GPU's in tests/test_gpu_numerics.py), on each
    path (`PATHS`)."""
    with on_path(path):
        compensated_primitives_hold_under_jit(shape)


def test_a_raise_and_lowering_by_powers_of_two_are_ieee_from_the_bits_on_a_flushing_cpu():
    """On XLA's CPU, whose fp32 multiply flushes a subnormal operand and result (row 4023):
    `dew.nn.float_bits.raised_exactly` is `x * 2^k` and `lowered_rounded` `x * 2^-k` rounded once to nearest
    even, bitwise numpy's IEEE `ldexp`, across every binade, subnormals both ways, at shifts 0 to 300, zero
    kept zero; and `binade_of` is `floor(log2|x|)`, -1024 at zero, int32 (row 6302's finish,
    `algebra.quotient`, and an initial state's gradient)."""
    values = edge_values(np.float32)
    # The largest int32 shift among them: lowered by it every value is a zero, no exponent's sum wrapping (row
    # 11807's review: a wrapped sum made it the least normal).
    shifts = np.asarray([0, 1, 2, 23, 24, 25, 63, 64, 126, 127, 149, 150, 151, 252, 300, 2**31 - 1], np.int32)
    on_cpu = jax.devices("cpu")[0]
    x, k = (jax.device_put(each, on_cpu) for each in (values[:, None], shifts[None, :]))
    lowered = np.asarray(jax.jit(lowered_rounded)(x, k))
    rounded = np.ldexp(values[:, None], -shifts[None, :]).astype(np.float32)
    np.testing.assert_array_equal(lowered.view(np.uint32), rounded.view(np.uint32))
    raised = np.asarray(jax.jit(raised_exactly)(x, k[:, :-1]))
    assert raised.dtype == np.float32
    exact = np.ldexp(values[:, None].astype(np.float64), shifts[None, :-1])
    held = np.abs(exact) <= np.finfo(np.float32).max
    expected = exact[held].astype(np.float32)
    np.testing.assert_array_equal(raised[held].view(np.uint32), expected.view(np.uint32))
    binades = np.asarray(jax.jit(binade_of)(jax.device_put(values, on_cpu)))
    assert binades.dtype == np.int32
    nonzero = values != 0
    np.testing.assert_array_equal(binades[nonzero],
                                  np.floor(np.log2(np.abs(values[nonzero].astype(np.float64)))).astype(np.int32))
    np.testing.assert_array_equal(binades[~nonzero], -1024)
