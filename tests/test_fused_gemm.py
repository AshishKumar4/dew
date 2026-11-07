"""Regions fused through `fused_gemm.fused` (the fusion pass's entry): a region's product declared as the dot
it stands for (`fused_gemm.product`), its consumers reading it as that dot rounds it, and
`fused_gemm.analysis`' report of what fusing a region decides, raises and costs, from its shapes alone."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS

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
    """Where the call names a kernel that refuses the region (the template on mma.sync, whose model counts no
    side operand, under `generation.chosen`), `fused_gemm.analysis` reports the refusal `fused` raises, so a
    pass never takes a region whose fusion fails."""
    sizes = jnp.asarray([100, 156], jnp.int32)
    x, w = jnp.ones((256, 512), jnp.bfloat16), jnp.ones((2, 512, 256), jnp.bfloat16)
    bias = jnp.ones((256,), jnp.float32)

    def region(x, w, sizes, bias):
        return fused_gemm.product(x, w, sizes, out_dtype=jnp.float32, interpret_on_cpu=True) + bias

    with generation.assumed("sm80"), generation.chosen(grouped_matmul="mosaic_gpu"):
        found = fused_gemm.analysis(region, x, w, sizes, bias)
        assert found.refused is not None and found.launch is None, found
        with pytest.raises(selection.KernelRefused):
            jax.jit(functools.partial(fused_gemm.fused, region))(x, w, sizes, bias)
