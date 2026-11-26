"""The MXFP8 operand form (`formats.MxFp8`: E4M3 codes, an E8M0 power-of-two scale every 32 depths, OCP MX)
and its grouped product's portable arm (`gemm.mxfp8_product`, row 4519), on the CPU."""

import contextlib

import jax
import jax.numpy as jnp
import mx_matrix
import numpy as np
import pytest
from blockwise_matrix import step_operands
from low_precision_support import spread
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import formats, gemm, generation, projections
from dew.nn.kernels.templates import product

SEEDS = 8


POLICIES = ("wide", "fp32 sums")
"""The two sums an MXFP8 portable product takes on a CPU (`xla_products.exponent_scaled`): float64 under
Dew's tests' wide arithmetic, and the compensated exponent-held fp32 sum where a scope states `FP32_SUMS`."""


def _policy(name: str):
    """The precision scope of the sum `name` (`POLICIES`)."""
    from dew.nn.precision import FP32_SUMS

    return contextlib.nullcontext() if name == "wide" else jax.default_matmul_precision(FP32_SUMS)


def test_an_mxfp8_operand_holds_each_blocks_largest_code_in_its_scales_binade():
    """Each 32-depth block's scale puts its largest magnitude in [256, 512) (OCP MX's rule), so its code,
    rounded to E4M3 and saturating at 448, lies in [256, 448]; a block at float32's top binade decodes
    finite; and its values are an MXFP8 operand's already, quantizing them again giving them back."""
    operand = formats.MxFp8.quantized(spread(0, (64, 256), product.MXFP8.depth))
    per = product.MXFP8.depth
    largest = jnp.max(jnp.abs(operand.codes.astype(jnp.float32)).reshape(64, -1, per), axis=-1)
    assert bool(jnp.all((largest >= 256) & (largest <= 448))), largest
    top = jnp.full((1, product.MXFP8.depth), jnp.finfo(jnp.float32).max * 0.99, jnp.float32)
    assert bool(jnp.all(jnp.isfinite(formats.MxFp8.quantized(top).decoded())))
    again = formats.MxFp8.quantized(operand.decoded())
    np.testing.assert_array_equal(np.asarray(again.decoded()), np.asarray(operand.decoded()))


def test_an_mxfp8_scale_reads_the_exponent_of_a_magnitude_just_under_a_power_of_two():
    """A block whose largest magnitude is one float32 ulp under 2^100 takes OCP MX's scale for it, 2 to its
    exponent (99) less E4M3's largest's (8), its code saturating at 448, where a float32 `log2` rounds the
    magnitude's logarithm up onto 100 (2^92, code 256); a float64 run of the same values quantizes to the
    same operand."""
    under = np.nextafter(np.float32(2.0 ** 100), np.float32(0))
    block = jnp.full((1, product.MXFP8.depth), under, jnp.float32)
    operand = formats.MxFp8.quantized(block)
    assert float(operand.scales[0, 0].astype(jnp.float32)) == 2.0 ** 91
    assert float(operand.codes[0, 0].astype(jnp.float32)) == 448.0
    with jax.enable_x64(new_val=True):
        wide = formats.MxFp8.quantized(jnp.asarray(block, jnp.float64))
    for field in ("codes", "scales"):
        np.testing.assert_array_equal(np.asarray(getattr(wide, field)).view(np.uint8),
                                      np.asarray(getattr(operand, field)).view(np.uint8))


def test_an_mxfp8_scale_byte_0xff_decodes_its_block_nan():
    """An MX operand whose first block's E8M0 scale byte is 0xFF (OCP MX 5.4.1: E8M0's NaN) decodes that
    block NaN and every other block as its codes times its scale, where 0xFF read as the exponent 128 gave
    infinities and zeros (spec alignment SA20261119pm F52)."""
    operand = formats.MxFp8.quantized(jnp.ones((2, 2 * product.MXFP8.depth), jnp.float32))
    held = np.asarray(operand.scales).view(np.uint8).copy()
    held[0, 0] = 0xFF
    nan = formats.MxFp8(operand.codes, jnp.asarray(held).view(operand.scales.dtype)).decoded()
    per = product.MXFP8.depth
    assert bool(jnp.all(jnp.isnan(nan[0, :per]))) and not bool(jnp.any(jnp.isnan(nan[0, per:])))
    np.testing.assert_array_equal(np.asarray(nan[1]), np.asarray(operand.decoded()[1]))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.bfloat16, jnp.float16])
def test_mxfp8_decodes_the_destination_bits_at_both_binade_edges(dtype):
    numbers = [0.0, -0.0, 0.5, -0.5, 1, -1, 2, -2, 448, -448, np.nan]
    codes = np.tile(np.asarray(numbers + [0] * (32 - len(numbers)), jnp.float8_e4m3fn), (4, 1))
    stored = np.asarray([0, 1, 254, 255], np.uint8).reshape(4, 1).view(jnp.float8_e8m0fnu)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected = (codes.astype(np.float64) * stored.astype(np.float64)).astype(dtype)
    with jax.enable_x64(dtype == jnp.float64):
        got = np.asarray(jax.jit(lambda x, e: formats.MxFp8(x, e).decoded(dtype))(codes, stored))
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    word = np.dtype(f"uint{got.dtype.itemsize * 8}")
    numbers = ~np.isnan(expected)
    np.testing.assert_array_equal(got.view(word)[numbers], expected.view(word)[numbers])


GROUP_BLOCKS = (1, 0, 2, 1)
"""Each group's 128-row blocks: a one-block group, an empty one, two blocks, one."""


@pytest.mark.parametrize("kernel", ["xla", "mosaic"])
def test_an_mxfp8_grouped_product_is_its_decoded_values_product(kernel):
    """Each group's rows times its matrix, both MXFP8, through `kernel` (the template on Blackwell's
    block-scaled tensor cores interpreted as a B200's, `mosaic_gpu.gmm`'s `mx_scales`, or the decoded operands
    in XLA), on par with the decoded values' grouped product in fp32 (check 1), from float64; an empty group's
    rows contribute nothing."""
    blocks = jnp.asarray(GROUP_BLOCKS, jnp.int32)
    sizes = 128 * blocks
    rows, depth, width = 128 * sum(GROUP_BLOCKS), 256, 128

    def draw(seed):
        return (spread(seed, (rows, depth), product.MXFP8.depth),
                spread(seed + 1, (len(GROUP_BLOCKS), width, depth), product.MXFP8.depth))

    def dew(x, w):
        with generation.chosen(mxfp8=kernel):
            return gemm.mxfp8_product(formats.MxFp8.quantized(x), formats.MxFp8.quantized(w), blocks,
                                      out_dtype=jnp.float32, interpret_on_cpu=kernel == "mosaic")

    with generation.assumed("sm100"), generation.routes() as routes:
        assert_as_exact(jax.jit(dew), jax.jit(mx_matrix.values_product(mx_matrix.mx_values, sizes)), draw,
                        f"an mxfp8 grouped product, {kernel}", seeds=SEEDS,
                        truth_fn=mx_matrix.values_product(mx_matrix.mx_values, sizes, jnp.float64))
    assert ("mxfp8", kernel) in {(route.choice, route.kernel) for route in routes}


def test_an_mxfp8_projections_training_step_is_on_par():
    """One training step through the MXFP8 recipe (`projections.mxfp8_projection`: its output and both
    gradients), the decoded arms in XLA, on par with the recipe in float64 (tests/mx_matrix.py's
    `recipe_truth`) over the draws (check 1), against the same step's float32 sums of the decoded operands."""
    blocks = jnp.asarray([1, 0, 2], jnp.int32)

    def projection(x, w):
        with generation.chosen(mxfp8="xla"):
            return projections.mxfp8_projection(x, w, blocks, out_dtype=jnp.float32)

    plain = mx_matrix.recipe_truth(blocks, jnp.float32)
    assert_as_exact(jax.jit(with_gradients(projection)), jax.jit(with_gradients(plain)), step_operands,
                    "mxfp8 training step", seeds=SEEDS,
                    truth_fn=with_gradients(mx_matrix.recipe_truth(blocks)))


def test_mxfp8_named_mosaic_uninterpreted_dispatches_portable_on_cpu():
    """In a CPU process assuming a B200 with `mxfp8="mosaic"` named and `interpret_on_cpu=False`,
    `mxfp8_product` dispatches through `_dispatched` to its portable arm on CPU instead of lowering Mosaic
    uninterpreted (row 8646)."""
    from dew.nn.kernels import formats, gemm, generation

    rng = np.random.default_rng(23)
    lhs = formats.MxFp8(jnp.asarray(rng.normal(size=(128, 128)) / 4, jnp.float8_e4m3fn),
                        jnp.ones((128, 4), jnp.float8_e8m0fnu))
    matrices = formats.MxFp8(jnp.asarray(rng.normal(size=(1, 128, 128)) / 4, jnp.float8_e4m3fn),
                             jnp.ones((1, 128, 4), jnp.float8_e8m0fnu))
    blocks = jnp.asarray([1], jnp.int32)
    with generation.assumed("sm100"):
        with generation.chosen(mxfp8="mosaic"):
            got = gemm.mxfp8_product(lhs, matrices, blocks, out_dtype=jnp.float32, interpret_on_cpu=False)
        with generation.chosen(mxfp8="xla"):
            ref = gemm.mxfp8_product(lhs, matrices, blocks, out_dtype=jnp.float32, interpret_on_cpu=False)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_a_block_holding_an_infinity_takes_the_nan_scale_quantized_or_in_the_epilogue():
    """A 32-column block of a product whose fp32 sum overflows in half its columns takes the NaN scale
    (`0xFF`), its finite values coded at a unit scale, and its row's finite block its own scale, alike
    whether `formats.MxFp8.quantized` quantizes the product or the GEMM's quantizing epilogue does
    (`gemm.gemm_quantized(..., MxFp8)`), bit for bit (row 13164: the epilogue's 247 against the
    quantizer's 118)."""
    x = jnp.zeros((16, 64), jnp.bfloat16).at[:, 1].set(1).at[0, 0].set(2.0 ** 127)
    w = jnp.zeros((64, 64), jnp.bfloat16).at[1, :].set(1).at[0, :16].set(4)
    product = jnp.dot(x, w, preferred_element_type=jnp.float32)
    fused = gemm.gemm_quantized(x, w, minifloat=formats.MxFp8)
    assert isinstance(fused, formats.MxFp8)
    quantized = formats.MxFp8.quantized(product)
    def raw(array):
        return np.asarray(jax.lax.bitcast_convert_type(array, jnp.uint8))

    scales = (raw(quantized.scales), raw(fused.scales))
    codes = (raw(quantized.codes), raw(fused.codes))
    np.testing.assert_array_equal(scales[0], scales[1])
    np.testing.assert_array_equal(codes[0], codes[1])
    unit = np.asarray(jax.lax.bitcast_convert_type(jnp.ones((), jnp.float8_e4m3fn), jnp.uint8))
    assert scales[0][0, 0] == 0xFF and scales[0][0, 1] == 127 - 8 and (scales[0][1:] == 127 - 8).all()
    # The overflowing block's finite values (its columns 16 to 31, each one) coded at the unit scale.
    np.testing.assert_array_equal(codes[0][0, 16:32], np.full(16, unit))
    assert bool(jnp.all(jnp.isnan(fused.decoded()[0, :32])))
    np.testing.assert_array_equal(np.asarray(fused.decoded()[0, 32:]), np.ones(32, np.float32))


@pytest.mark.parametrize("policy", POLICIES)
def test_an_mxfp8_products_least_scale_meets_its_largest_in_the_exponents(policy):
    """An MXFP8 product of a code of one at the least scale (`0x00`, 2^-127, a float32 subnormal) and a code
    of one at the largest (`0xFE`, 2^127) is one a depth, summed: exactly 32 over a block, on a CPU, whose
    XLA flushes a subnormal operand to zero (row 13164: the decoded operands' product read 0)."""
    scale = formats.MxFp8(jnp.ones((128, 32), jnp.float8_e4m3fn),
                          jax.lax.bitcast_convert_type(jnp.zeros((128, 1), jnp.uint8), jnp.float8_e8m0fnu))
    large = formats.MxFp8(jnp.ones((1, 128, 32), jnp.float8_e4m3fn),
                          jax.lax.bitcast_convert_type(jnp.full((1, 128, 1), 0xFE, jnp.uint8),
                                                       jnp.float8_e8m0fnu))
    with generation.chosen(mxfp8="xla"), _policy(policy):
        out = jax.jit(lambda a, b: gemm.mxfp8_product(a, b, jnp.asarray([1], jnp.int32),
                                                      out_dtype=jnp.float32))(scale, large)
    np.testing.assert_array_equal(np.asarray(out), np.full((128, 128), 32.0, np.float32))


def test_a_blockwise_products_rows_past_its_groups_are_zero_whatever_their_scales():
    """The blockwise FP8 product's portable sum (`xla_products.step_scaled`): rows past the groups are zero
    even where their row scales are infinite, a padded row's (row 13164: masked before its scaling, its zero
    times an infinite scale read NaN)."""
    from dew.nn.kernels import xla_products

    tokens = jnp.ones((256, 128), jnp.bfloat16)
    kernel = jnp.ones((1, 128, 128), jnp.bfloat16)
    row_scales = jnp.ones((1, 256), jnp.float32).at[0, 128:].set(jnp.inf)
    column_scales = jnp.ones((1, 1, 1), jnp.float32)
    out = jax.jit(xla_products.step_scaled)(tokens, kernel, jnp.asarray([128], jnp.int32), row_scales,
                                           column_scales)
    np.testing.assert_array_equal(np.asarray(out[128:]), np.zeros((128, 128), np.float32))
    np.testing.assert_array_equal(np.asarray(out[:128]), np.full((128, 128), 128.0, np.float32))


@pytest.mark.parametrize("policy", POLICIES)
def test_an_mxfp8_products_partials_below_fp32s_normals_sum_to_its_normal_result(policy):
    """Two 32-deep blocks of codes of one, each operand's scale 2^-66 (`0x3D`): each block's partial is
    2^-127, below fp32's normals, and their sum 2^-126 is normal, exactly so under either sum: float64's,
    and fp32's held at its own exponent, neither partial a flushed subnormal (row 13164's review)."""
    tiny = jax.lax.bitcast_convert_type(jnp.full((128, 2), 0x3D, jnp.uint8), jnp.float8_e8m0fnu)
    rows = formats.MxFp8(jnp.ones((128, 64), jnp.float8_e4m3fn), tiny)
    matrices = formats.MxFp8(jnp.ones((1, 128, 64), jnp.float8_e4m3fn), tiny[None])
    with generation.chosen(mxfp8="xla"), _policy(policy):
        out = jax.jit(lambda a, b: gemm.mxfp8_product(a, b, jnp.asarray([1], jnp.int32),
                                                      out_dtype=jnp.float32))(rows, matrices)
    np.testing.assert_array_equal(np.asarray(out), np.full((128, 128), 2.0 ** -126, np.float32))


def test_an_mxfp8_product_of_no_groups_is_zero():
    """An MXFP8 product of a padded row buffer (128 rows, 64 deep) over no group, its group table empty: no
    row belongs to a group, so it is its zeros (row 13164's review: the block walk's broadcast of the rows'
    exponents against no group's failed to trace)."""
    unit = jax.lax.bitcast_convert_type(jnp.full((128, 2), 127, jnp.uint8), jnp.float8_e8m0fnu)
    rows = formats.MxFp8(jnp.ones((128, 64), jnp.float8_e4m3fn), unit)
    matrices = formats.MxFp8(jnp.ones((0, 64, 64), jnp.float8_e4m3fn), unit[:0, None].reshape(0, 64, 2))
    with generation.chosen(mxfp8="xla"):
        out = jax.jit(lambda a, b: gemm.mxfp8_product(a, b, jnp.zeros((0,), jnp.int32),
                                                      out_dtype=jnp.float32))(rows, matrices)
    np.testing.assert_array_equal(np.asarray(out), np.zeros((128, 64), np.float32))


def test_an_mxfp8_product_of_no_depth_is_zero():
    """An MXFP8 product over no depth (no 32-deep block) is its zeros, as the empty sum is."""
    no_scales = jnp.zeros((128, 0), jnp.float8_e8m0fnu)
    rows = formats.MxFp8(jnp.zeros((128, 0), jnp.float8_e4m3fn), no_scales)
    matrices = formats.MxFp8(jnp.zeros((1, 64, 0), jnp.float8_e4m3fn), no_scales[:64][None])
    with generation.chosen(mxfp8="xla"):
        out = jax.jit(lambda a, b: gemm.mxfp8_product(a, b, jnp.asarray([1], jnp.int32),
                                                      out_dtype=jnp.float32))(rows, matrices)
    np.testing.assert_array_equal(np.asarray(out), np.zeros((128, 64), np.float32))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("split_rows", [False, True], ids=["columns", "rows and columns"])
def test_an_mxfp8_product_keeps_its_operands_explicit_placement(split_rows):
    """An MXFP8 product under an Explicit mesh, its matrices' columns split over two CPU devices and its rows
    whole or split over the same axis (each operand's scales beside its codes): its output placed as a
    ragged product places it (the columns' axis, or the rows' where one axis splits both), its value the
    unsplit product's (row 13164's review: the block walk's sum placed as its partials, the matrices' side
    of each block's exponents and NaN mask placed with it)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    x = formats.MxFp8.quantized(spread(0, (128, 64), product.MXFP8.depth))
    w = formats.MxFp8.quantized(spread(1, (1, 128, 64), product.MXFP8.depth))
    blocks = jnp.asarray([1], jnp.int32)

    def run(a, b):
        return gemm.mxfp8_product(a, b, blocks, out_dtype=jnp.float32)

    with generation.chosen(mxfp8="xla"):
        whole = jax.jit(run)(x, w)
        with jax.set_mesh(mesh):
            columns = NamedSharding(mesh, PartitionSpec(None, "data"))
            split = jax.tree.map(lambda leaf: jax.device_put(leaf, columns), w)
            rows = NamedSharding(mesh, PartitionSpec("data" if split_rows else None))
            out = jax.jit(run)(jax.tree.map(lambda leaf: jax.device_put(leaf, rows), x), split)
    assert out.sharding.spec[0 if split_rows else 1] == "data", out.sharding
    np.testing.assert_array_equal(np.asarray(out), np.asarray(whole))


def test_a_nan_block_takes_the_nan_scale_and_poisons_only_the_sums_it_reaches():
    """A block holding a NaN takes the NaN scale (`0xFF`, `formats.MxFp8.quantized`), the row's other block
    its own; in a product, a NaN-scaled block of the rows makes that row's sums NaN and one of a matrix's
    columns that column's, every other sum finite and the rows past the groups zero (row 13164)."""
    x = jnp.ones((2, 64), jnp.float32).at[0, 5].set(jnp.nan)
    held = np.asarray(jax.lax.bitcast_convert_type(formats.MxFp8.quantized(x).scales, jnp.uint8))
    assert held[0, 0] == 0xFF and held[0, 1] != 0xFF and (held[1] != 0xFF).all()
    nan_byte = jnp.uint8(0xFF)
    one = jnp.uint8(127)
    row_scales = jnp.full((256, 2), one).at[3, 1].set(nan_byte)
    column_scales = jnp.full((1, 128, 2), one).at[0, 7, 0].set(nan_byte)
    rows = formats.MxFp8(jnp.ones((256, 64), jnp.float8_e4m3fn),
                         jax.lax.bitcast_convert_type(row_scales, jnp.float8_e8m0fnu))
    matrices = formats.MxFp8(jnp.ones((1, 128, 64), jnp.float8_e4m3fn),
                             jax.lax.bitcast_convert_type(column_scales, jnp.float8_e8m0fnu))
    with generation.chosen(mxfp8="xla"):
        out = np.asarray(jax.jit(lambda a, b: gemm.mxfp8_product(a, b, jnp.asarray([1], jnp.int32),
                                                                 out_dtype=jnp.float32))(rows, matrices))
    nan = np.isnan(out)
    assert nan[3].all() and nan[:128, 7].all() and nan.sum() == 128 + 128 - 1
    np.testing.assert_array_equal(out[128:], np.zeros((128, 128), np.float32))
    assert (out[:128][~nan[:128]] == 64.0).all()


def _blocks_product(signs: tuple[int, ...], row_exponents: tuple[int, ...],
                    column_exponents: tuple[int, ...], policy: str) -> np.ndarray:
    """An MXFP8 product of one 128-row group over `len(signs)` 32-deep blocks, every code one (its row's sign
    `signs[b]` in block `b`), the rows' and the matrix's E8M0 exponents per block, by XLA (each block's
    partial 32 times 2 to its two exponents), its sum `policy`'s (`POLICIES`)."""
    def scales(exponents, shape):
        held = jnp.broadcast_to(jnp.asarray([e + 127 for e in exponents], jnp.uint8), shape)
        return jax.lax.bitcast_convert_type(held, jnp.float8_e8m0fnu)

    blocks = len(signs)
    codes = jnp.repeat(jnp.asarray(signs, jnp.float32), 32)[None].repeat(128, 0).astype(jnp.float8_e4m3fn)
    rows = formats.MxFp8(codes, scales(row_exponents, (128, blocks)))
    matrices = formats.MxFp8(jnp.ones((1, 128, 32 * blocks), jnp.float8_e4m3fn),
                             scales(column_exponents, (1, 128, blocks)))
    with generation.chosen(mxfp8="xla"), _policy(policy):
        return np.asarray(jax.jit(lambda a, b: gemm.mxfp8_product(a, b, jnp.asarray([1], jnp.int32),
                                                                  out_dtype=jnp.float32))(rows, matrices))


@pytest.mark.parametrize("policy", POLICIES)
def test_an_mxfp8_sum_that_cancels_keeps_its_small_remainder(policy):
    """Three blocks' partials 32 x 2^120, -32 x 2^120 and 32 x 2^-32: the first two cancel exactly and the
    sum is the third's, 2^-27, kept where the running sum's exponent had been 2^125's (row 13164's
    review: an exponent left at the cancelled blocks' brought the last to 2^-153 and lost it)."""
    out = _blocks_product((1, -1, 1), (60, 60, -16), (60, 60, -16), policy)
    np.testing.assert_array_equal(out, np.full((128, 128), 2.0 ** -27, np.float32))


@pytest.mark.parametrize("policy", POLICIES)
def test_an_mxfp8_sum_keeps_what_each_addition_rounds_away(policy):
    """Three blocks' partials 2^24, 1 and -2^24: fp32 adds the one to 2^24 and rounds it away, the sum
    compensated keeps it, so the product is 1 as its depth contracted whole is (testing.md's split
    products; row 13164's review)."""
    out = _blocks_product((1, 1, -1), (10, -3, 10), (9, -2, 9), policy)
    np.testing.assert_array_equal(out, np.ones((128, 128), np.float32))


def test_a_nan_in_a_quantizing_epilogues_product_takes_the_nan_scale():
    """A NaN in a GEMM's rows reaches every column of its row's product, whose blocks the quantizing epilogue
    (`gemm.gemm_quantized(..., MxFp8)`) scales NaN (`0xFF`), every other row's block its own scale, as the
    quantizer scales the product (row 13164)."""
    x = jnp.ones((16, 64), jnp.bfloat16).at[2, 7].set(jnp.nan)
    w = jnp.eye(64, dtype=jnp.bfloat16)
    fused = gemm.gemm_quantized(x, w, minifloat=formats.MxFp8)
    assert isinstance(fused, formats.MxFp8)
    held = np.asarray(jax.lax.bitcast_convert_type(fused.scales, jnp.uint8))
    assert (held[2] == 0xFF).all() and (np.delete(held, 2, axis=0) != 0xFF).all()
    expected = formats.MxFp8.quantized(jnp.dot(x, w, preferred_element_type=jnp.float32)).scales
    np.testing.assert_array_equal(held, np.asarray(jax.lax.bitcast_convert_type(expected, jnp.uint8)))


def test_a_float64_truths_block_past_float32s_range_keeps_its_scale():
    """A float64 truth's block whose largest magnitude is 2^128, past float32's range, takes the scale 2^120
    and the code 256 from its own exponent field, decoding to 2^128 again, not the NaN scale a float32 cast
    of its magnitude would read (row 13249's review)."""
    with jax.enable_x64(new_val=True):
        block = jnp.zeros((1, 32), jnp.float64).at[0, 0].set(2.0 ** 128)
        operand = formats.MxFp8.quantized(block)
        held = np.asarray(jax.lax.bitcast_convert_type(operand.scales, jnp.uint8))
        assert held[0, 0] == 127 + 120
        assert float(operand.decoded(jnp.float64)[0, 0]) == 2.0 ** 128


def test_an_mxfp8_sums_two_policies_differ_where_fp32_rounds_a_blocks_low_term_away():
    """A block whose partial is 2^16 + 2^-18 (codes 256 and 2^-9 against 256 and 2^-9) and one of -2^16: the
    wide sum keeps the first block's low term, 2^-18, and fp32's sum loses it in the block's own fp32 dot,
    so the two policies dispatch apart (`xla_products.exponent_scaled`; row 13164's review)."""
    codes = np.zeros((128, 64), np.float32)
    codes[:, 0], codes[:, 1], codes[:, 32] = 256, 2.0 ** -9, -256
    columns = np.zeros((1, 128, 64), np.float32)
    columns[..., 0], columns[..., 1], columns[..., 32] = 256, 2.0 ** -9, 256
    unit = jax.lax.bitcast_convert_type(jnp.full((128, 2), 127, jnp.uint8), jnp.float8_e8m0fnu)
    rows = formats.MxFp8(jnp.asarray(codes, jnp.float8_e4m3fn), unit)
    matrices = formats.MxFp8(jnp.asarray(columns, jnp.float8_e4m3fn), unit[None])
    found = {}
    for policy in POLICIES:
        with generation.chosen(mxfp8="xla"), _policy(policy):
            found[policy] = np.asarray(jax.jit(lambda a, b: gemm.mxfp8_product(
                a, b, jnp.asarray([1], jnp.int32), out_dtype=jnp.float32))(rows, matrices))
    np.testing.assert_array_equal(found["wide"], np.full((128, 128), 2.0 ** -18, np.float32))
    np.testing.assert_array_equal(found["fp32 sums"], np.zeros((128, 128), np.float32))
