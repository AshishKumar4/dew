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
    bytes at the H100's device memory rate; a region with no consumer and no producer is not priced. Shapes
    only."""
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


def test_a_normed_projections_analysis_prices_its_prologue_against_forming_the_rows_whole():
    """`fused_gemm.analysis` of an RMS norm feeding a bf16 product of 8192 x 4096 x 4096 (a compile-only
    H100): its prologue fused (the rows' inverse RMS computed whole beside, the normalization tile by tile
    in the kernel), it costs less than XLA forming the normalized rows whole and the launch reading them
    back, by at least the rows' bytes once at the device memory rate: the normalized rows are never written
    or read, where the statistics' pass reads the rows once more. Shapes only."""
    x = jax.ShapeDtypeStruct((8192, 4096), jnp.bfloat16)
    scale = jax.ShapeDtypeStruct((4096,), jnp.float32)
    w = jax.ShapeDtypeStruct((4096, 4096), jnp.bfloat16)

    def region(x, scale, w):
        wide = x.astype(jnp.float32)
        inverse = jax.lax.rsqrt(jnp.mean(wide * wide, axis=-1, keepdims=True) + 1e-6)
        return fused_gemm.product((wide * inverse * scale).astype(jnp.bfloat16), w, out_dtype=jnp.bfloat16)

    with generation.assumed("sm90"):
        found = fused_gemm.analysis(region, x, scale, w)
        rate = generation.gpu_target().memory_rate
    assert found.prologue is not None and found.prologue.kernel == "fused", found
    assert found.fused_seconds is not None and found.unfused_seconds is not None, found
    assert found.unfused_seconds - found.fused_seconds >= 8192 * 4096 * 2 / rate, found


def test_a_regions_price_reads_its_declared_precision_and_every_bank_of_its_matrix():
    """`fused_gemm.priced` reads what the region's product declares (a compile-only H100): a float32 product
    with a bias declared at a three-pass algorithm (`BF16_BF16_F32_X3`) costs three passes of its one-pass
    preset's (`algebra.cost.product_work`), where the ambient default priced both alike; and a grouped
    product of few rows over four banks of its matrix reads every bank, at least their bytes at the device
    memory rate, where one bank was priced. Shapes only."""
    x32 = jax.ShapeDtypeStruct((8192, 4096), jnp.float32)
    w32 = jax.ShapeDtypeStruct((4096, 4096), jnp.float32)
    bias = jax.ShapeDtypeStruct((4096,), jnp.float32)
    sizes = jax.ShapeDtypeStruct((4,), jnp.int32)
    x = jax.ShapeDtypeStruct((64, 4096), jnp.bfloat16)
    banks = jax.ShapeDtypeStruct((4, 4096, 4096), jnp.bfloat16)

    def biased(precision):
        return lambda x, w, bias: fused_gemm.product(x, w, precision=precision, out_dtype=jnp.float32) + bias

    def grouped(x, w, sizes, bias):
        return fused_gemm.product(x, w, sizes, out_dtype=jnp.float32) + bias

    with generation.assumed("sm90"):
        thrice = fused_gemm.analysis(biased(jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3), x32, w32, bias)
        once = fused_gemm.analysis(biased(jax.lax.DotAlgorithmPreset.BF16_BF16_F32), x32, w32, bias)
        found = fused_gemm.analysis(grouped, x, banks, sizes, bias)
        rate = generation.gpu_target().memory_rate
    assert thrice.fused_seconds > 2 * once.fused_seconds, (thrice, once)
    assert found.fused_seconds >= 4 * 4096 * 4096 * 2 / rate, found


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


def _products(jaxpr) -> int:
    """How many products `jaxpr` and the jaxprs it holds compute: dots and kernel launches."""
    products = {"dot_general", "ragged_dot_general", "pallas_call"}
    return sum(eqn.primitive.name in products for eqn in jaxpr.eqns) + sum(
        _products(inner) for eqn in jaxpr.eqns for inner in jax.extend.core.jaxprs_in_params(eqn.params))


def test_a_named_gate_projection_is_its_launchs_residual_and_its_backward_recomputes_no_product():
    """A SwiGLU on a gate projection the program names `gate_proj` (`checkpoint_name`), fused through
    `fused_gemm.fused` on an H100 (a CPU's interpretation): under a remat policy saving that name, the
    launch's output under it is the derivative's residual (`gemm`'s `named`), so the reverse pass reads
    the gate from it and computes one product fewer than the same region unnamed, whose backward recomputes
    the gate's accumulator (row 7131); its gradients are the unnamed region's, bitwise (the epilogue reads
    its accumulator through its rounding to bf16 alone, the residual that rounding)."""
    keys = jax.random.split(jax.random.key(11), 3)
    x = jax.random.randint(keys[0], (64, 128), -3, 4).astype(jnp.bfloat16)
    wg = jax.random.randint(keys[1], (128, 64), -3, 4).astype(jnp.bfloat16)
    up = jax.random.randint(keys[2], (64, 64), -3, 4).astype(jnp.bfloat16)

    def region(name):
        def swiglu(x, wg, up):
            gate = fused_gemm.product(x, wg, out_dtype=jnp.bfloat16, interpret_on_cpu=True)
            gate = jax.ad_checkpoint.checkpoint_name(gate, name) if name else gate
            return (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(jnp.bfloat16)
        return functools.partial(fused_gemm.fused, swiglu)

    policy = jax.checkpoint_policies.save_only_these_names("gate_proj")

    def gradient(name):
        def loss(x, wg, up):
            return jnp.sum(region(name)(x, wg, up).astype(jnp.float32))
        return jax.grad(jax.checkpoint(loss, policy=policy), argnums=(0, 1))

    with generation.assumed("sm90"):
        named, unnamed = (_products(jax.make_jaxpr(gradient(name))(x, wg, up).jaxpr)
                          for name in ("gate_proj", None))
        got, expected = (gradient(name)(x, wg, up) for name in ("gate_proj", None))
    assert named < unnamed, (named, unnamed)
    for g, e in zip(got, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(g, np.float32), np.asarray(e, np.float32))


@pytest.mark.parametrize("gpu", ["sm80", "sm90"])
@pytest.mark.parametrize(("hidden", "width", "rows", "taken"),
                         [(1024, 4096 + 512 + 512, 12288, ("sm80", "sm90")),
                          (1024, 2 * 3584, 12288, ("sm80", "sm90")),
                          (1024, 2048, 12288, ()), (2560, 9216, 24576, ("sm90",)),
                          (2560, 2 * 9216, 4096, ("sm90",)), (4096, 2 * 14336, 12288, ("sm90",))],
                         ids=["q/k/v", "gate/up", "unmeasured", "4B up", "4B gate/up", "8B gate/up"])
def test_a_normed_projection_takes_the_template_where_a_receipt_timed_it(hidden, width, rows, taken, gpu):
    """An RMS norm feeding a bf16 projection of 4 x 3072 rows of 1024 on an A100 or an H100 (its program
    through the algebra's pass, which recognizes the norm as the product's prologue): at the widths M-EM-65
    timed (Qwen3.5-0.8B's joined q/k/v and gate/up, `grouped_matmul.NORMED_08B`) the product's launch is the
    template's and the norm its prologue, as the receipt measured faster than the norm kernel then XLA's
    dot, and on an H100 at the 4B's and the 8B's normed widths M-EM-67 timed (`NORMED_4B_8B`: an A100's lost
    there); a formed projection at a matrix another receipt timed (the 4B's up, M-EM-36 on wgmma) its
    cell's, as the projection unnormed (train 977 F1); at a width none timed, XLA's dot as every dense
    cell's of their families (DENSE1)."""
    from dew import algebra

    def normed(x, scale, w):
        wide = x.astype(jnp.float32)
        inverse = jax.lax.rsqrt(jnp.mean(wide * wide, axis=-1, keepdims=True) + 1e-6)
        return jnp.dot((wide * inverse * scale).astype(x.dtype), w,
                       preferred_element_type=jnp.float32).astype(x.dtype)

    x = jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16)
    scale = jax.ShapeDtypeStruct((hidden,), jnp.float32)
    w = jax.ShapeDtypeStruct((hidden, width), jnp.bfloat16)
    with generation.assumed(gpu), generation.routes() as routes:
        jax.jit(algebra.rewritten(normed, kernels=True)).trace(x, scale, w)
    found = {(route.choice, route.kernel) for route in routes}
    assert (("grouped_matmul", "mosaic") in found) == (gpu in taken), found

