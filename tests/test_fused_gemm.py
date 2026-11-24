"""Regions fused through `fused_gemm.fused` (the fusion pass's entry): a region's product declared as the dot
it stands for (`fused_gemm.product`), its consumers reading it as that dot rounds it, and
`fused_gemm.analysis`' report of what fusing a region decides, raises and costs, from its shapes alone."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from reference_error import assert_as_exact, seeds_for
from test_algebra import traces_cheaper
from test_algebra_program import _scaled_twice

import dew
from dew import algebra
from dew.nn import programs
from dew.nn.kernels import fused_gemm, gemm, generation, selection


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


def _checked_draws(seed: int, wide: int, narrow: int) -> tuple:
    """Two devices' rows (2 x 64 x 128) and biases (2 x 256), a feature scale (128) and a matrix (128 x
    256), integers in bf16: the rows and the matrix within `wide` of zero, the scale and the bias within
    `narrow`."""
    keys = jax.random.split(jax.random.key(seed), 4)
    x = jax.random.randint(keys[0], (2, 64, 128), -wide, wide + 1).astype(jnp.bfloat16)
    scale = jax.random.randint(keys[1], (128,), -narrow, narrow + 1).astype(jnp.bfloat16)
    w = jax.random.randint(keys[2], (128, 256), -wide, wide + 1).astype(jnp.bfloat16)
    bias = jax.random.randint(keys[3], (2, 256), -narrow, narrow + 1).astype(jnp.bfloat16)
    return x, scale, w, bias


def _checked_map(target: str, *, fused: bool = True):
    """A region fused through `fused_gemm.fused` (the rows scaled before the product, its producer, and a bias
    added after, its epilogue) in a shard_map over two CPU devices that checks the axes its values vary over
    (`check_vma`): each device's rows and bias its own, the scale and the matrix the same on every device;
    the product interpreted where `target` is a GPU's. With `fused` False, the region written plainly in the
    same map: its unfused program, the law the fused one is held to."""
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    data, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def region(x, scale, w, bias):
        if not fused:
            return jnp.dot(x * scale, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16) + bias
        return fused_gemm.product(x * scale, w, out_dtype=jnp.bfloat16,
                                  interpret_on_cpu=target != "cpu") + bias

    def each_device(x, scale, w, bias):
        apply = functools.partial(fused_gemm.fused, region) if fused else region
        return apply(x[0], scale, w, bias[0])[None]

    return jax.shard_map(each_device, mesh=mesh, in_specs=(data, whole, whole, data), out_specs=data,
                         check_vma=True)


@INTERPRETS
@pytest.mark.parametrize("target", ["sm90", "cpu"])
def test_a_fused_region_in_a_checked_manual_map_reads_values_each_device_holds_its_own_of(target):
    """A region fused through `fused_gemm.fused` in a shard_map over two CPU devices that checks the axes its
    values vary over (`check_vma`): each device's rows and its own bias (both varying over the map's axis), a
    feature scale and the matrix the same on every device, the rows scaled before the product (its producer,
    the template's prologue) and the bias added after (its epilogue). On an assumed, interpreted H100 the
    template's launch, named, takes both, its product typed as its operands vary (row 9808); on the CPU, XLA's
    products run them whole, checked. Each device's region computes the scaled rows' dot rounded to bf16 plus
    its bias, bitwise, over integers whose sums pass bf16's 256."""
    x, scale, w, bias = _checked_draws(9808, 7, 3)
    mapped = jax.jit(_checked_map(target))
    if target == "cpu":
        got = mapped(x, scale, w, bias)
    else:
        with generation.assumed(target), generation.chosen(grouped_matmul="mosaic"), \
                generation.routes() as routes:
            got = mapped(x, scale, w, bias)
        assert ("grouped_matmul", "mosaic") in {(route.choice, route.kernel) for route in routes}, routes
    expected = jnp.einsum("dri,io->dro", x * scale, w,
                          preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(got, np.float32),
                                  np.asarray(expected + bias[:, None, :], np.float32))


@INTERPRETS
@pytest.mark.parametrize("target", ["sm90", "cpu"])
def test_a_replicated_operands_gradient_in_a_checked_manual_map_is_summed_over_its_devices(target):
    """The gradient of a region fused through `fused_gemm.fused` in a shard_map that checks the axes its
    values vary over (`check_vma`), with respect to the feature scale and the matrix every device holds the
    same of: each device's rows and bias its own, so each device's partial gradient of the replicated values
    varies over the map's axis, and the gradient of the replicated value is their sum over the devices, typed
    the same on every device. On an assumed, interpreted H100 the template's launch takes the product, its
    kernel gradient typed as its operands vary (row 9808), which reverse mode then sums over the devices (row
    10526: `ad.ct_check` refused a varying cotangent for the unvarying matrix); on the CPU, XLA's products,
    checked. The gradients equal the unfused program's in the same map, bitwise: each device's partial rounded
    to the operand's dtype and summed over the devices, as JAX's own transposes sum them."""
    x, scale, w, bias = _checked_draws(10526, 3, 2)
    mapped = _checked_map(target)

    def loss(scale, w):
        return jnp.sum(mapped(x, scale, w, bias).astype(jnp.float32))

    gradient = jax.jit(jax.grad(loss, argnums=(0, 1)))
    if target == "cpu":
        got = gradient(scale, w)
    else:
        with generation.assumed(target), generation.chosen(grouped_matmul="mosaic"), \
                generation.routes() as routes:
            got = gradient(scale, w)
        assert ("grouped_matmul", "mosaic") in {(route.choice, route.kernel) for route in routes}, routes

    unfused_map = _checked_map(target, fused=False)

    def unfused_loss(scale, w):
        return jnp.sum(unfused_map(x, scale, w, bias).astype(jnp.float32))

    expected = jax.jit(jax.grad(unfused_loss, argnums=(0, 1)))(scale, w)
    for each, want in zip(got, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(each, np.float32), np.asarray(want, np.float32))


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


@INTERPRETS
def test_an_explicit_epilogue_is_rewritten_by_the_algebra_pass_as_an_inferred_one_is():
    """A launch given its epilogue directly (`gemm.gemm`'s `epilogue`) runs it as the algebra pass rewrites
    it, as a fused region's inferred consumer is (the test above's two capped column scales): one
    preparation for every hook a launch takes, explicit or inferred (`tuning.KernelOp.prepared`; spec
    alignment 2026-11-19 PM, ALGEBRA-6); with the pass off, as written."""
    sizes = jnp.asarray([96, 40], jnp.int32)

    def explicit(x, w, a, b):
        # The consumer the pass rewrites (test_algebra_program's score hook), on the product's tile.
        return gemm.gemm(x, w, sizes, epilogue=lambda acc, a, b: _scaled_twice(acc, None, None, a, b),
                         epilogue_operands=((a, ("columns",)), (b, ("columns",))), out_dtype=jnp.float32,
                         interpret_on_cpu=True)

    rng = np.random.default_rng(0)
    operands = (jnp.asarray(rng.standard_normal((136, 128)) / 8, jnp.float32),
                jnp.asarray(rng.standard_normal((2, 128, 256)) / 8, jnp.float32),
                *(jnp.asarray(rng.uniform(0.5, 2.0, 256), jnp.float32) for _ in range(2)))
    with generation.assumed("sm90"):
        assert traces_cheaper(explicit, *operands)
        with algebra.passes(on=False):
            assert not traces_cheaper(explicit, *operands)


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


def test_a_scattered_regions_fused_price_counts_its_sides_permutation():
    """A scattered product's fused arm permutes its consumer's row-indexed residual to the product's rows
    before its launch, an XLA pass reading and writing it and reading the scatter table: priced so, its fused
    arm costs at least those bytes at the H100's device memory rate more than the same region gathered, its
    residual read where it lies; its unfused arm the same either way. Shapes only (rows 12399, 12419)."""
    sizes = jax.ShapeDtypeStruct((4,), jnp.int32)
    x = jax.ShapeDtypeStruct((8192, 4096), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((4, 4096, 4096), jnp.bfloat16)
    rows = jax.ShapeDtypeStruct((8192,), jnp.int32)
    residual = jax.ShapeDtypeStruct((8192, 4096), jnp.float32)

    def routed(table):
        def region(x, w, sizes, rows, residual):
            product = fused_gemm.product(x, w, sizes, out_dtype=jnp.float32, **{table: rows})
            return (product + residual).astype(jnp.bfloat16)
        return region

    with generation.assumed("sm90"):
        scattered, gathered = (fused_gemm.analysis(routed(table), x, w, sizes, rows, residual)
                               for table in ("scatter", "gather"))
        rate = generation.gpu_target().memory_rate
    assert scattered.unfused_seconds == gathered.unfused_seconds, (scattered, gathered)
    # The residual read and written, and the scatter table read, once each.
    permuted = (2 * 8192 * 4096 * 4 + 8192 * 4) / rate
    assert scattered.fused_seconds - gathered.fused_seconds >= permuted, (scattered, gathered)


def _chained(function, wide):
    """`function` applied eight times over `wide`, in float32."""
    for _ in range(8):
        wide = function(wide)
    return wide


def _logs_and_exp2s(stage: str, target: str) -> tuple:
    """`fused_gemm.analysis` of a bf16 product of 8192 x 256 x 4096 with eight chained float32 logs after it
    (`stage` 'epilogue'), or of 8192 x 4096 x 256 with them before it on its rows ('prologue'), launched by
    the Mosaic template on a compile-only `target`; and of the same with exp2s."""
    def region(function, x, w):
        if stage == "prologue":
            x = _chained(function, x.astype(jnp.float32)).astype(jnp.bfloat16)
        product = fused_gemm.product(x, w, out_dtype=jnp.bfloat16)
        return product if stage == "prologue" else _chained(function, product.astype(jnp.float32))

    depth, width = (4096, 256) if stage == "prologue" else (256, 4096)
    x = jax.ShapeDtypeStruct((8192, depth), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((depth, width), jnp.bfloat16)
    with generation.assumed(target), generation.chosen(grouped_matmul="mosaic"):
        return tuple(fused_gemm.analysis(functools.partial(region, function), x, w)
                     for function in (jnp.log, jnp.exp2))


@pytest.mark.parametrize("stage", ["epilogue", "prologue"])
def test_a_fused_launchs_special_functions_are_priced_in_the_forms_it_compiles(stage):
    """A product's epilogue or prologue of eight chained float32 logs, its launch the Mosaic template's on a
    compile-only H100 (`_logs_and_exp2s`): fused, the logs cost its launch more than as many exp2s, as it
    compiles them without `approx_math` (`emission.ACCURATE_FORMS`: a log a polynomial of FP32 instructions,
    an exp2 one MUFU), where each was priced one MUFU an element and the two alike (row 10824). Shapes
    only."""
    logs, exp2s = _logs_and_exp2s(stage, "sm90")
    assert exp2s.fused_seconds is not None, exp2s
    assert logs.fused_seconds > exp2s.fused_seconds, (logs, exp2s)


def test_a_tpu_fused_launchs_special_functions_are_priced_as_its_kernel_counts_them():
    """A product's prologue of eight chained float32 logs on a v5e (`_logs_and_exp2s`), whose Mosaic TPU
    kernel lowers its special functions one way whatever `approx_math`: fused, the logs cost its launch as
    many exp2s do, as its own estimate counts them (`emission.launch_work`), where an H100's forms priced its
    logs (train 1165's F1). (The epilogue's product the v5e's launch refuses: its MXUs' sums within a step
    no measurement states IEEE.) Shapes only."""
    logs, exp2s = _logs_and_exp2s("prologue", "v5e")
    assert exp2s.fused_seconds is not None, exp2s
    assert logs.fused_seconds == exp2s.fused_seconds, (logs, exp2s)


@pytest.mark.parametrize("operation", [jax.lax.erf, jnp.expm1, jnp.isfinite], ids=lambda f: f.__name__)
def test_a_projection_followed_by_an_elementwise_function_traces_through_dew_jit(operation):
    """A bf16 projection followed by an erf, an expm1 or a finiteness test, traced through `dew.jit` for an
    H100 (whose analysis prices each region, `fused_gemm.analysis`): it traces, each counted as the CUDA
    cores' work (`emission`'s rule), where a counter that priced none of the three refused the program (base
    review 92f34bc8d F1)."""
    x = jax.ShapeDtypeStruct((256, 512), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((512, 256), jnp.bfloat16)
    with generation.assumed("sm90"):
        traced = dew.jit(lambda x, w: operation(x @ w)).trace(x, w)
    assert traced.out_info.shape == (256, 256)


def test_a_regions_analysis_reads_a_divisibility_hint_as_its_value():
    """A projection's product rounded to int32 under a divisibility hint (`pl.multiple_of`), analysed for an
    H100: the hint is its operand's value and no work, so the region analyses (`fused_gemm.analysis`), where
    the counter priced no rule for it and refused the region (base review 05bd9f77b F1)."""
    from jax.experimental import pallas as pl

    x = jax.ShapeDtypeStruct((256, 512), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((512, 256), jnp.bfloat16)
    with generation.assumed("sm90"):
        fused_gemm.analysis(lambda x, w: pl.multiple_of(fused_gemm.product(x, w).astype(jnp.int32), 1), x, w)


_NORMED = (jax.ShapeDtypeStruct((8192, 4096), jnp.bfloat16), jax.ShapeDtypeStruct((4096,), jnp.float32),
           jax.ShapeDtypeStruct((4096, 4096), jnp.bfloat16))
"""An RMS norm's rows, its scale and a projection's weights, 8192 x 4096 x 4096, as shapes."""


def _normed_product(x, scale, w):
    """`x`'s rows RMS-normed and scaled, then projected by `w` (`fused_gemm.product`), in bf16."""
    wide = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.mean(wide * wide, axis=-1, keepdims=True) + 1e-6)
    return fused_gemm.product((wide * inverse * scale).astype(jnp.bfloat16), w, out_dtype=jnp.bfloat16)

def test_a_normed_projections_analysis_prices_its_prologue_against_forming_the_rows_whole():
    """`fused_gemm.analysis` of an RMS norm feeding a bf16 product of 8192 x 4096 x 4096 (a compile-only
    H100): its prologue fused (the rows' inverse RMS computed whole beside, the normalization tile by tile
    in the kernel), it costs less than XLA forming the normalized rows whole and the launch reading them
    back, by at least the rows' bytes once at the device memory rate: the normalized rows are never written
    or read, where the statistics' pass reads the rows once more. Shapes only."""
    with generation.assumed("sm90"):
        found = fused_gemm.analysis(_normed_product, *_NORMED)
        rate = generation.gpu_target().memory_rate
    assert found.prologue is not None and found.prologue.kernel == "fused", found
    assert found.fused_seconds is not None and found.unfused_seconds is not None, found
    assert found.unfused_seconds - found.fused_seconds >= 8192 * 4096 * 2 / rate, found


def test_a_normed_projection_read_transposed_is_priced_as_its_product_alone():
    """An RMS norm feeding a bf16 product of 8192 x 4096 x 4096 (a compile-only H100), its product read
    transposed: the launch takes the prologue and not the consumer (the product read at other than its
    tile), so `fused_gemm.analysis` prices the region as the normed product alone, the transpose XLA's
    after it either way. Shapes only."""
    with generation.assumed("sm90"):
        alone = fused_gemm.analysis(_normed_product, *_NORMED)
        transposed = fused_gemm.analysis(lambda *args: _normed_product(*args).T, *_NORMED)
    assert transposed.epilogue is not None and transposed.epilogue.kernel != "fused", transposed
    assert transposed.prologue is not None and transposed.prologue.kernel == "fused", transposed
    assert (transposed.fused_seconds, transposed.unfused_seconds) == (
        alone.fused_seconds, alone.unfused_seconds)
    assert alone.fused_seconds is not None, alone

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


def test_a_fused_regions_launch_is_launched_once():
    """A biased bf16 product's fused arm (`fused_gemm.priced`: the product and its epilogue one launch, no
    prologue) pays its target's fixed launch once: priced on an H100 that launches in no time and on one
    that launches in a second, its seconds differ by that second alone, where the product's kernel was
    charged a launch of its own beside its launch's (row 8814's review). Shapes only."""
    import dataclasses

    x, w = (jax.ShapeDtypeStruct(shape, jnp.bfloat16) for shape in ((8192, 4096), (4096, 4096)))
    bias = jax.ShapeDtypeStruct((4096,), jnp.float32)
    found, h100 = {}, generation.gpu_target("sm90")
    for launch in (0.0, 1.0):
        measured = {kind: dataclasses.replace(part, launch=launch) for kind, part in h100.measured.items()}
        with generation.assumed(dataclasses.replace(h100, measured=measured)):
            found[launch] = fused_gemm.analysis(
                lambda x, w, bias: fused_gemm.product(x, w, out_dtype=jnp.float32) + bias, x, w, bias)
    assert found[0.0].fused_seconds is not None, found[0.0]
    assert found[1.0].fused_seconds == found[0.0].fused_seconds + 1.0, found


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


@pytest.mark.parametrize("row_value", [lambda x: jnp.prod(x, axis=1, keepdims=True),
                                       lambda x: jnp.argmax(x, axis=1, keepdims=True).astype(x.dtype)],
                         ids=["product", "argmax"])
def test_a_producer_whose_row_value_is_any_reduction_fuses_through_the_public_entry(row_value):
    """A producer scaling each row by a value XLA computes whole beside the prologue (its elements' product,
    the index of its largest) is priced and fused through `fused_gemm.fused` (a CPU's interpretation): XLA's
    pass for the row value priced by its bytes, the stage the kernel runs by the kernels' counter, where a
    product's and an index's reductions were refused unpriced (base reviews of 045f07e31 and df983f703,
    F1); the region's values the program's as written (check 1)."""
    def region(x, w):
        return fused_gemm.product(x * row_value(x), w, out_dtype=jnp.float32, interpret_on_cpu=True)

    def program(x, w):
        return jnp.dot(x * row_value(x), w)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(1.0 + rng.standard_normal((64, 128)) / 64, jnp.float32),
                jnp.asarray(rng.standard_normal((128, 64)) / 8, jnp.float32))

    fused = functools.partial(fused_gemm.fused, region)
    with generation.assumed("sm90"):
        found = fused_gemm.analysis(region, *draw(0))
        assert found.prologue is not None, found
        assert_as_exact(jax.jit(fused), jax.jit(program), draw, "a product reduction's producer, fused",
                        seeds=8)


def test_a_prologues_sides_are_priced_by_their_arithmetic_as_xla_runs_them():
    """Two producers scaling each row of an 8192 x 4096 bf16 operand by a row value XLA computes beside the
    prologue, a sum of the row and a sum of its exponentials, feeding one 4096-wide product on an H100: the
    exponential's side prices above the sum's by its 33.5M exponentials, as the algebra prices XLA's
    program (`algebra.program.priced`), where bytes alone priced them alike (row 7260); the fused price of
    each its unfused one's less its forming, the stage priced once."""
    x = jax.ShapeDtypeStruct((8192, 4096), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((4096, 4096), jnp.bfloat16)

    def region(side):
        def fn(x, w):
            return fused_gemm.product(x * side(x), w, out_dtype=jnp.bfloat16)
        return fn

    with generation.assumed("sm90"):
        summed = fused_gemm.analysis(region(lambda x: jnp.sum(x, axis=1, keepdims=True)), x, w)
        exponential = fused_gemm.analysis(region(lambda x: jnp.sum(jnp.exp(x), axis=1, keepdims=True)), x, w)
    for found in (summed, exponential):
        assert found.prologue is not None and found.prologue.kernel == "fused", found.prologue
    assert exponential.fused_seconds > summed.fused_seconds, (exponential, summed)


def test_an_elementwise_producer_formed_whole_prices_xlas_launch_of_it():
    """`fused_gemm.analysis` on an H100 of a small product (128 x 128 x 128, a bias its consumer) whose
    operand an elementwise producer forms (`x * scale`), against the same region given its operand: its
    unfused price is XLA's launch forming the operand whole beside it, at least the part's fixed latency
    more (row 7284: an elementwise program priced no launch), its bytes a few hundred nanoseconds at most."""
    x, w = (jax.ShapeDtypeStruct((128, 128), jnp.bfloat16) for _ in range(2))
    scale, bias = (jax.ShapeDtypeStruct((128,), jnp.bfloat16) for _ in range(2))

    def scaled(x, scale, w, bias):
        return fused_gemm.product(x * scale, w, out_dtype=jnp.bfloat16) + bias

    def given(x, w, bias):
        return fused_gemm.product(x, w, out_dtype=jnp.bfloat16) + bias

    with generation.assumed("sm90"):
        target = generation.gpu_target()
        formed = fused_gemm.analysis(scaled, x, scale, w, bias)
        bare = fused_gemm.analysis(given, x, w, bias)
    assert formed.unfused_seconds - bare.unfused_seconds >= target.fixed_seconds, (formed, bare)


def test_a_prologues_side_program_prices_what_its_sides_read():
    """`fused_gemm.analysis` on an H100 of an 8192 x 4096 x 4096 bf16 product whose producer scales each row
    by its sum, its stage `x` or an exponential read through a bitcast (an equation the algebra reads as
    opaque, whose inputs a program's roots price): the sides XLA computes beside the launch are the row sums
    alike, so the fused prices agree (the stage's work hidden under the launch's products), where the side
    program priced the stage's dead equations (row 7285)."""
    x = jax.ShapeDtypeStruct((8192, 4096), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((4096, 4096), jnp.bfloat16)

    def region(stage):
        def fn(x, w):
            scaled = stage(x) * jnp.sum(x, axis=1, keepdims=True)
            return fused_gemm.product(scaled, w, out_dtype=jnp.bfloat16)
        return fn

    def bitcast(x):  # an exponential through a bitcast, which the algebra reads as an opaque equation
        return jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(jnp.exp(x), jnp.int16), jnp.bfloat16)

    with generation.assumed("sm90"):
        plain = fused_gemm.analysis(region(lambda x: x), x, w)
        exponential = fused_gemm.analysis(region(bitcast), x, w)
    for found in (plain, exponential):
        assert found.prologue is not None and found.prologue.kernel == "fused", found.prologue
    assert exponential.fused_seconds == plain.fused_seconds, (exponential, plain)


def _products(jaxpr) -> int:
    """How many products `jaxpr` and the programs within it compute (`programs.equations`): dots and kernel
    launches."""
    products = {"dot_general", "ragged_dot_general", "pallas_call"}
    return sum(eqn.primitive.name in products for eqn in programs.equations(jaxpr))


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
    timed (Qwen3.5-0.8B's joined q/k/v and gate/up, `grouped_cells.NORMED_08B`) the product's launch is the
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


def test_fused_product_fuses_its_epilogue_with_gather_and_scatter():
    """`fused_gemm.product(..., gather=..., scatter=...)` over a token-order RMSNorm producer (top-k routes,
    `M_routed != M_tok`) runs the producer in XLA in its tokens' order, so its derivative sums each token's
    routes first, and fuses a post-scatter bias+residual consumer into the epilogue, holding check 1
    forward and VJP against float64 truth (rows 5391, 12306)."""
    # 512 routed rows: the bias's gradient, a sum over them of bf16 cotangents, rounds in fp32 on most draws,
    # so its error from float64 resolves (at 64 rows it was exact on most, its ratio inconclusive).
    tokens_n, rows, depth, width, groups = 256, 512, 64, 64, 4
    rng = np.random.default_rng(53910)
    sizes = jnp.asarray([128, 0, 192, 192], jnp.int32)
    gather = jnp.asarray(rng.integers(0, tokens_n, size=(rows,)), jnp.int32)
    scatter = jnp.asarray(rng.permutation(rows), jnp.int32)

    with pytest.raises(ValueError):
        fused_gemm.product(jnp.zeros((tokens_n, depth), jnp.bfloat16),
                           jnp.zeros((depth, width), jnp.bfloat16), gather=gather)

    def draw(seed):
        r = np.random.default_rng(53911 + seed)
        return (jnp.asarray(r.normal(size=(tokens_n, depth)) / 4, jnp.bfloat16),
                jnp.asarray(r.normal(size=(depth,)) / 4 + 1.0, jnp.float32),
                jnp.asarray(r.normal(size=(groups, depth, width)) / 8, jnp.bfloat16),
                jnp.asarray(r.normal(size=(width,)) / 8, jnp.float32),
                jnp.asarray(r.normal(size=(rows, width)) / 8, jnp.bfloat16),
                jnp.asarray(r.normal(size=(rows, width)) / 8, jnp.bfloat16))

    def normed(x, scale, *, wide_dtype=jnp.float32):
        w = x.astype(wide_dtype)
        inv = jax.lax.rsqrt(jnp.mean(w * w, axis=-1, keepdims=True) + 1e-5)
        out_d = x.dtype if wide_dtype == jnp.float32 else wide_dtype
        return (w * inv * scale.astype(wide_dtype)).astype(out_d)

    def region(x, scale, weight, bias, residual):
        y = fused_gemm.product(normed(x, scale), weight, sizes, out_dtype=jnp.float32,
                               gather=gather, scatter=scatter, interpret_on_cpu=True)
        return (y + bias + residual.astype(jnp.float32)).astype(jnp.bfloat16)

    def ref_fwd(x, scale, weight, bias, residual, *, dtype=jnp.float32):
        out_d = jnp.bfloat16 if dtype == jnp.float32 else dtype
        y = jax.lax.ragged_dot(normed(x, scale, wide_dtype=dtype)[gather], weight, sizes,
                               preferred_element_type=dtype)
        y = jnp.empty_like(y).at[scatter].set(y)
        return (y + bias.astype(dtype) + residual.astype(dtype)).astype(out_d)

    def truth(x, scale, weight, bias, residual, dy):
        out, pull = jax.vjp(lambda *a: ref_fwd(*a, dtype=jnp.float64),
                            *(a.astype(jnp.float64) for a in (x, scale, weight, bias, residual)))
        return out, pull(dy.astype(jnp.float64))

    def plain_vjp(x, scale, weight, bias, residual, dy):
        out, pull = jax.vjp(ref_fwd, x, scale, weight, bias, residual)
        return out, pull(dy)

    def dew_vjp(x, scale, weight, bias, residual, dy):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic", gemm_prologue="xla",
                                                           gemm_epilogue="fused"):
            out, pull = jax.vjp(lambda *a: fused_gemm.fused(region, *a), x, scale, weight, bias, residual)
            return out, pull(dy)

    dew_fn, ref_fn = jax.jit(dew_vjp), jax.jit(plain_vjp)
    label = "fused_gemm gather+scatter+prologue+epilogue"
    assert_as_exact(dew_fn, ref_fn, draw, label, truth_fn=truth,
                    seeds=seeds_for(ref_fn, draw, label, truth_fn=truth, dew_fn=dew_fn))


def test_a_gathered_products_producer_pulls_back_each_tokens_routes_summed():
    """A token routed twice through a unit weight from a producer `x * 4`, its routes' cotangents 2**126 and
    -2**126: their sum, 0, meets the producer's derivative, as the program written does, not each route's
    4 * 2**126 (inf - inf) (row 5391)."""
    gather, sizes = jnp.asarray([0, 0], jnp.int32), jnp.asarray([2], jnp.int32)
    weight = jnp.eye(8, dtype=jnp.bfloat16)[None]

    def region(x, w):
        return fused_gemm.product(x * 4, w, sizes, out_dtype=jnp.float32, gather=gather,
                                  interpret_on_cpu=True)

    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        _, pull = jax.vjp(lambda x: fused_gemm.fused(region, x, weight), jnp.ones((1, 8), jnp.bfloat16))
        (d_x,) = pull(jnp.zeros((2, 8), jnp.float32).at[0, 0].set(2.0 ** 126).at[1, 0].set(-(2.0 ** 126)))
    np.testing.assert_array_equal(np.asarray(d_x, np.float32), np.zeros((1, 8), np.float32))


@pytest.mark.mesh(devices=2)
def test_a_gathered_products_consumer_reads_it_as_varying_as_its_routing_table():
    """A fused product in a checked `shard_map` over two CPU devices, its tokens and weight every device's
    and its routing table and residual each device's own: its consumer reads the product varying as the
    table does, adding the residual as the unfused program does (row 9808)."""
    tokens = jnp.asarray(np.random.default_rng(9808).integers(-3, 4, (4, 8)), jnp.float32)
    weight = jnp.eye(8, dtype=jnp.float32)[None]
    tables = jnp.asarray([0, 1, 2, 3, 3, 2, 1, 0], jnp.int32)
    residual = jnp.arange(64, dtype=jnp.float32).reshape(8, 8)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    data, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()
    sizes = jnp.asarray([4], jnp.int32)

    def each_device(x, w, table, side, *, fused):
        def region(x, w, side):
            y = (fused_gemm.product(x, w, sizes, out_dtype=jnp.float32, gather=table, interpret_on_cpu=True)
                 if fused else jax.lax.ragged_dot(x[table], w, sizes))
            return y + side
        return fused_gemm.fused(region, x, w, side) if fused else region(x, w, side)

    def mapped(*, fused):
        return jax.jit(jax.shard_map(functools.partial(each_device, fused=fused), mesh=mesh,
                                     in_specs=(whole, whole, data, data), out_specs=data, check_vma=True))(
            tokens, weight, tables, residual)

    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        np.testing.assert_array_equal(np.asarray(mapped(fused=True)), np.asarray(mapped(fused=False)))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize(("routed", "consumer"), [("gather", "negated"), ("scatter", "negated"),
                                                  ("scatter", "squared")])
def test_a_routed_products_reverse_pass_reads_it_as_varying_as_its_routing_table(routed, consumer):
    """A fused product's VJP in a checked `shard_map`, its tokens, weight and sizes every device's and its
    routing table (a gather, or a scatter's permutation) each device's own, under a consumer of no sides
    (`-y`, or `y * y`, whose derivative reads the product again): its reverse pass types the product
    varying as its table does, as its forward did, its tokens' gradient the unfused program's (row 9808)."""
    tokens = jnp.asarray(np.random.default_rng(98080).integers(-3, 4, (8, 8)), jnp.bfloat16)
    weight = jnp.eye(8, dtype=jnp.bfloat16)[None]
    tables = jnp.asarray([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0], jnp.int32)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    data, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()
    sizes = jnp.asarray([8], jnp.int32)

    def each_device(x, w, table, *, fused):
        tables = {routed: table}

        def region(x, w):
            if fused:
                y = fused_gemm.product(x, w, sizes, out_dtype=jnp.float32, interpret_on_cpu=True, **tables)
            else:
                rows = x[table] if routed == "gather" else x
                y = jax.lax.ragged_dot(rows, w, sizes, preferred_element_type=jnp.float32)
                y = y if routed == "gather" else jnp.empty_like(y).at[table].set(y)
            return -y if consumer == "negated" else y * y
        program = functools.partial(fused_gemm.fused, region) if fused else region
        return jax.grad(lambda x: jnp.sum(program(x, w) * jnp.arange(8, dtype=jnp.float32)))(x)

    def mapped(*, fused):
        return jax.jit(jax.shard_map(functools.partial(each_device, fused=fused), mesh=mesh,
                                     in_specs=(whole, whole, data), out_specs=data, check_vma=True))(
            tokens, weight, tables)

    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        np.testing.assert_array_equal(np.asarray(mapped(fused=True), np.float32),
                                      np.asarray(mapped(fused=False), np.float32))


@pytest.mark.mesh(devices=2)
def test_a_scattered_products_consumer_reads_an_explicitly_placed_side_in_its_rows_order():
    """A fused scattered product under an Explicit mesh, its tokens, weight and permutation whole and its
    residual's rows split over two CPU devices: its consumer reads the residual permuted to the product's
    rows, placed as the residual is, its sum named (`checkpoint_name`) and read back in the product's rows'
    order, and its value and gradients are the product's unfused (`gemm.gemm`, then the add), forward and
    backward (rows 5391, 12401)."""
    from jax.ad_checkpoint import checkpoint_name
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    tokens = jnp.asarray(np.random.default_rng(5391).integers(-3, 4, (8, 8)), jnp.bfloat16)
    weight = jnp.eye(8, dtype=jnp.bfloat16)[None]
    scatter = jnp.asarray([7, 6, 5, 4, 3, 2, 1, 0], jnp.int32)
    sizes = jnp.asarray([8], jnp.int32)

    def region(x, w, side, *, fused):
        call = fused_gemm.product if fused else gemm.gemm
        y = call(x, w, sizes, out_dtype=jnp.float32, scatter=scatter, interpret_on_cpu=True)
        # Named, a residual of the derivative under its name (`gemm`'s `named`), read back in row order.
        return checkpoint_name(y + side, "saved") * side

    def both(side, *, fused):
        program = (functools.partial(fused_gemm.fused, functools.partial(region, fused=True)) if fused
                   else functools.partial(region, fused=False))
        out, pull = jax.vjp(program, tokens, weight, side)
        return out, pull(jnp.ones_like(out))

    with jax.set_mesh(mesh), generation.assumed("sm90"):
        side = jax.device_put(jnp.arange(64, dtype=jnp.float32).reshape(8, 8),
                              NamedSharding(mesh, PartitionSpec("data")))
        got = jax.jit(functools.partial(both, fused=True))(side)
        want = jax.jit(functools.partial(both, fused=False))(side)
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(np.asarray(g, np.float32), np.asarray(w, np.float32))
