"""Regions fused through `fused_gemm.fused` (the fusion pass's entry): a region's product declared as the dot
it stands for (`fused_gemm.product`), its consumers reading it as that dot rounds it, and
`fused_gemm.analysis`' report of what fusing a region decides, raises and costs, from its shapes alone."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from reference_error import assert_as_exact
from test_algebra import traces_cheaper

from dew import algebra
from dew.nn.kernels import fused_gemm, generation, selection


@INTERPRETS
@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_a_fused_regions_consumer_reads_its_product_as_the_dot_rounds_it(generation_name):
    """A region fused through `fused_gemm.fused`, its dot written as `fused_gemm.product` declaring a bf16
    result: its consumer, run in the epilogue, reads the accumulator rounded once to bf16, the dot's own
    rounding, before its first operation, so the region's value is the program's bitwise. Over integers
    whose sums pass bf16's 256 (the rounding matters; every product and fp32 sum exact): a bias added to the
    bf16 product, as XLA's program adds it; and `fused_gemm.analysis` reports the epilogue fused, run on
    nothing."""
    sizes = jnp.asarray([96, 40], jnp.int32)
    keys = jax.random.split(jax.random.key(7), 3)
    x = jax.random.randint(keys[0], (136, 128), -7, 8).astype(jnp.bfloat16)
    w = jax.random.randint(keys[1], (2, 128, 256), -7, 8).astype(jnp.bfloat16)
    bias = jax.random.randint(keys[2], (256,), -3, 4).astype(jnp.bfloat16)

    def region(x, w, bias):
        return fused_gemm.product(x, w, sizes, out_dtype=jnp.bfloat16, interpret_on_cpu=True) + bias

    def program(x, w, bias):
        return jax.lax.ragged_dot(x, w, sizes, preferred_element_type=jnp.float32).astype(jnp.bfloat16) + bias

    with generation.assumed(generation_name):
        found = fused_gemm.analysis(region, x, w, bias)
        assert found.epilogue is not None and found.epilogue.kernel == "fused", found
        got = jax.jit(functools.partial(fused_gemm.fused, region))(x, w, bias)
    expected = np.asarray(program(x, w, bias), np.float32)
    np.testing.assert_array_equal(np.asarray(got, np.float32), expected)
    # The draw tells the two roundings apart: the fp32 sum plus the bias, rounded once, differs.
    once = jax.lax.ragged_dot(x, w, sizes, preferred_element_type=jnp.float32) + bias.astype(jnp.float32)
    assert np.any(np.asarray(once.astype(jnp.bfloat16), np.float32) != expected)


@INTERPRETS
def test_a_fused_regions_consumer_is_rewritten_by_the_algebra_pass_before_its_epilogue_tiles_it():
    """A region fused through `fused_gemm.fused` whose consumer two capped column scales of the product
    summed, which the epilogue takes tile by tile: the region traces the consumer's cheaper form, the one
    program its analysis tiles and its epilogue runs (algebra.md §2), and computes the region's values as
    the unfused program does (check 1); with the pass off, the consumer as written."""
    sizes = jnp.asarray([96, 40], jnp.int32)

    def region(x, w, a, b):
        capped = jnp.tanh(fused_gemm.product(x, w, sizes, out_dtype=jnp.float32, interpret_on_cpu=True))
        return capped * jnp.tanh(a) + capped * jnp.tanh(b)

    def program(x, w, a, b):
        capped = jnp.tanh(jax.lax.ragged_dot(x, w, sizes))
        return capped * jnp.tanh(a) + capped * jnp.tanh(b)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.standard_normal((136, 128)) / 8, jnp.float32),
                jnp.asarray(rng.standard_normal((2, 128, 256)) / 8, jnp.float32),
                *(jnp.asarray(rng.uniform(0.5, 2.0, 256), jnp.float32) for _ in range(2)))

    fused = functools.partial(fused_gemm.fused, region)
    with generation.assumed("sm90"):
        found = fused_gemm.analysis(region, *draw(0))
        assert found.epilogue is not None and found.epilogue.kernel == "fused", found
        assert traces_cheaper(fused, *draw(0))
        with algebra.passes(on=False):
            assert not traces_cheaper(fused, *draw(0))
        assert_as_exact(jax.jit(fused), jax.jit(program), draw, "a fused region's rewritten consumer",
                        seeds=8)


def test_a_regions_analysis_prices_it_fused_and_unfused_on_one_model():
    """`fused_gemm.analysis` prices a region both ways from one model (`fused_gemm.priced`, the shared
    `templates.cost`): a bf16 product of 8192 x 4096 x 4096 with a bias and a SiLU after it costs less fused
    than its launch writing the product and XLA's fusion reading it back, by at least that round trip's
    bytes at the H100's device memory rate; a region with no consumer is not priced. Shapes only."""
    sizes = jax.ShapeDtypeStruct((4,), jnp.int32)
    x = jax.ShapeDtypeStruct((8192, 4096), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((4, 4096, 4096), jnp.bfloat16)
    bias = jax.ShapeDtypeStruct((4096,), jnp.float32)

    def region(x, w, sizes, bias):
        product = fused_gemm.product(x, w, sizes, out_dtype=jnp.bfloat16)
        return jax.nn.silu(product.astype(jnp.float32) + bias).astype(jnp.bfloat16)

    with generation.assumed("sm90"):
        found = fused_gemm.analysis(region, x, w, sizes, bias)
        plain = fused_gemm.analysis(lambda x, w, sizes: fused_gemm.product(x, w, sizes), x, w, sizes)
        rate = generation.gpu_target().memory_rate
    assert found.fused_seconds is not None and found.unfused_seconds is not None, found
    assert found.unfused_seconds - found.fused_seconds >= 2 * 8192 * 4096 * 2 / rate, found
    assert plain.fused_seconds is None and plain.unfused_seconds is None


def test_an_integer_regions_product_is_xlas_own_dot_with_nothing_to_analyse():
    """`fused_gemm.product` of int8 operands (a quantized dot's, int32 out) is XLA's own dot, nothing fused:
    `fused_gemm.analysis` of its region reports no stage and no price, and `fused` gives XLA's value."""
    keys = jax.random.split(jax.random.key(3))
    x = jax.random.randint(keys[0], (64, 128), -8, 8).astype(jnp.int8)
    w = jax.random.randint(keys[1], (128, 96), -8, 8).astype(jnp.int8)

    def region(x, w):
        return fused_gemm.product(x, w, out_dtype=jnp.int32) * 2

    with generation.assumed("sm90"):
        found = fused_gemm.analysis(region, x, w)
        got = fused_gemm.fused(region, x, w)
    assert found.epilogue is None and found.launch is None and found.fused_seconds is None, found
    expected = jax.lax.dot_general(x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32) * 2
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@INTERPRETS
def test_a_regions_analysis_reports_what_fusing_it_would_raise():
    """Where the call names a kernel that refuses the region (the template, which multiplies its operands
    whole, named under `generation.chosen` for a declared dot algorithm that narrows fp32 operands to
    TF32's), `fused_gemm.analysis` reports the refusal `fused` raises, so a pass never takes a region whose
    fusion fails."""
    sizes = jnp.asarray([100, 156], jnp.int32)
    x, w = jnp.ones((256, 512), jnp.float32), jnp.ones((2, 512, 256), jnp.float32)
    bias = jnp.ones((256,), jnp.float32)

    def region(x, w, sizes, bias):
        return fused_gemm.product(x, w, sizes, out_dtype=jnp.float32, interpret_on_cpu=True,
                                  precision=jax.lax.DotAlgorithmPreset.TF32_TF32_F32) + bias

    with generation.assumed("sm80"), generation.chosen(grouped_matmul="mosaic"):
        found = fused_gemm.analysis(region, x, w, sizes, bias)
        assert found.refused is not None and found.launch is None, found
        with pytest.raises(selection.KernelRefused):
            jax.jit(functools.partial(fused_gemm.fused, region))(x, w, sizes, bias)


@INTERPRETS
@pytest.mark.parametrize("generation_name", ["sm90"])
def test_a_fused_prologue_never_reads_the_depths_padding(generation_name):
    """A region whose product reads a per-row scale divided by its rows (`fusible_gemm(s / x, w)`, the
    division fused into the template's prologue) at a depth of 100, which the launch pads to whole tiles with
    zeros: the prologue's result is zero past the call's depth (`mosaic_gpu.depth_masked`), so the region is
    the program's, not NaN (a scale over the padding's zeros, infinities times the matrix's zero rows, where
    the depth was padded before the prologue ran)."""
    sizes = jnp.asarray([40, 24], jnp.int32)
    keys = jax.random.split(jax.random.key(96), 2)
    magnitude = jax.random.randint(keys[0], (64, 100), 1, 4)
    x = jnp.where(jax.random.bernoulli(keys[1], shape=(64, 100)), magnitude, -magnitude).astype(jnp.bfloat16)
    w = jax.random.randint(keys[1], (2, 100, 128), -3, 4).astype(jnp.bfloat16)
    s = jnp.full((64, 1), 6.0, jnp.bfloat16)

    def region(x, w, s):
        return fused_gemm.fusible_gemm(s / x, w, sizes, interpret_on_cpu=True)

    def program(x, w, s):
        return jax.lax.ragged_dot(s / x, w, sizes, preferred_element_type=jnp.float32)

    with generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"):
        got = jax.jit(functools.partial(fused_gemm.fused, region))(x, w, s)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(program(x, w, s)))
