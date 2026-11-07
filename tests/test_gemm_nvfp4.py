"""The GEMM template's product of an NVFP4 matrix (`gemm.nvfp4_product`, row 3621): its weights' E2M1 codes
and E4M3 block scales decoded exactly (`NvFp4.decoded`, its codes in the staged order a stage decodes), the
deferred factor applied to the fp32 accumulator before the one rounding."""

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from jax.sharding import Mesh

from dew.algebra import bits
from dew.nn.kernels import gemm, generation, grouped_matmul
from dew.nn.kernels.templates.gemm import mosaic_gpu

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
"""An E2M1 code's magnitude by its low three bits; the fourth is its sign."""


def nvfp4(seed: int, depth: int, width: int, *, scale: float = 0.25) -> tuple[gemm.NvFp4, np.ndarray]:
    """An NVFP4 matrix of random codes under power-of-two block scales (`gemm.NvFp4.of`, each value exact in
    float64), and its values: code times block scale times the tensor's `scale`."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(depth, width))
    magnitudes = np.where(codes >= 8, -1.0, 1.0) * np.asarray(E2M1)[codes & 7]
    block = np.exp2(rng.integers(-3, 4, size=(depth // 16, width))).astype(np.float32)
    matrix = gemm.NvFp4.of(jnp.asarray(magnitudes, jnp.float32), jnp.asarray(block, jnp.float8_e4m3fn), scale)
    return matrix, magnitudes * np.repeat(block.astype(np.float64), 16, axis=0) * scale


def test_an_nvfp4_matrix_holds_its_codes_and_the_staged_order_reads_back():
    """A matrix quantized from fp32 (`gemm.NvFp4.quantized`) holds `bits.nvfp4_quantized`'s codes and scales:
    its codes read back in the depths' own order (`NvFp4.codes`) are the quantizer's natural words' codes,
    its scales the quantizer's; and its words are K-major, in the staged order (`gemm.staged_rows`, within
    each 128 depths word w holding depths w + 16j), not the natural one's."""
    x = jnp.asarray(np.random.default_rng(5).standard_normal((256, 64)), jnp.float32)
    matrix = gemm.NvFp4.quantized(x)
    words, scales, scale = bits.nvfp4_quantized(x)
    natural = bits.nvfp4_decoded(words, jnp.full(scales.shape, bits.NVFP4_OWED, jnp.float8_e4m3fn))
    np.testing.assert_array_equal(np.asarray(matrix.codes()), np.asarray(natural, np.float32))
    np.testing.assert_array_equal(np.asarray(matrix.scales.T, np.float32), np.asarray(scales, np.float32))
    assert float(matrix.scale) == float(scale)
    assert matrix.words.shape == (64, 256 // 8)
    assert not np.array_equal(np.asarray(matrix.words.T), np.asarray(words))
    order = gemm.staged_rows(256)
    assert sorted(order) == list(range(256)) and order[8 * 3 + 5] == 16 * 5 + 3 and order[128] == 128


def test_an_nvfp4_product_is_its_decoded_products_times_the_deferred_factor():
    """XLA's route (a CPU's): the rows rounded to bfloat16 times the decoded matrix, summed in fp32, times
    `bits.NVFP4_OWED` and the tensor's scale, then rounded once: bitwise the program written out, and, its
    operands small integers and powers of two, exactly the float64 product of the matrix's values."""
    matrix, values = nvfp4(0, 256, 192)
    rows = jnp.asarray(np.random.default_rng(1).integers(-3, 4, size=(64, 256)), jnp.bfloat16)
    got = jax.jit(lambda x, m: gemm.nvfp4_product(x, m, out_dtype=jnp.float32))(rows, matrix)
    written = jnp.dot(rows, matrix.decoded(), preferred_element_type=jnp.float32) * matrix.factor
    np.testing.assert_array_equal(np.asarray(got), np.asarray(written))
    np.testing.assert_array_equal(np.asarray(got, np.float64), np.asarray(rows, np.float64) @ values)


@INTERPRETS
@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_an_nvfp4_product_through_the_template_is_exact(generation_name):
    """The template's launch interpreted on each family, the template named for the product: the same exact
    product; and the rows' gradient through the decoded matrix, the weights frozen, on the route its selector
    picks (the cotangent fp32 beside a bfloat16 matrix)."""
    matrix, values = nvfp4(2, 256, 256)
    rows = jnp.asarray(np.random.default_rng(3).integers(-3, 4, size=(128, 256)), jnp.bfloat16)
    cotangent = jnp.asarray(np.random.default_rng(4).integers(-2, 3, size=(128, 256)), jnp.float32)

    def product(x):
        return gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32, interpret_on_cpu=True)

    with generation.assumed(generation_name):
        with generation.chosen(grouped_matmul="mosaic_gpu"), generation.routes() as routes:
            got = jax.jit(product)(rows)
        _, pull = jax.vjp(product, rows)
        (gradient,) = pull(cotangent)
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic_gpu"}
    np.testing.assert_array_equal(np.asarray(got, np.float64), np.asarray(rows, np.float64) @ values)
    # The rows' gradient is exact in fp32 (small integers times powers of two) and rounds once to bfloat16.
    expected = jnp.asarray(np.asarray(cotangent, np.float64) @ values.T, jnp.float32).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(gradient, np.float32), np.asarray(expected, np.float32))


@INTERPRETS
@pytest.mark.parametrize("family", ["mma.sync", "wgmma"])
def test_the_mainloop_decodes_an_nvfp4_stage_in_place(family):
    """`mosaic_gpu.gmm` of an NVFP4 matrix's words and scales (`scales`, row 3621's step hook) decodes each
    256-deep stage into bfloat16 in the consumer's own buffer before its products: the rows times the
    decoded values, bitwise `NvFp4.decoded`'s product (small integer rows, every product and sum exact),
    two column tiles and two steps, interpreted on each family; a 128-deep step is refused."""
    matrix, _ = nvfp4(6, 512, 128)
    rows = jnp.asarray(np.random.default_rng(7).integers(-3, 4, size=(136, 512)), jnp.bfloat16)
    expected = np.asarray(rows, np.float64) @ np.asarray(matrix.decoded(), np.float64)
    tiles = mosaic_gpu.Tiles(64, 64, 256, 2, 1, family=family)

    def packed(step_tiles):
        return mosaic_gpu.gmm(rows, matrix.words[None], jnp.asarray([136], jnp.int32), tiles=step_tiles,
                              transpose_rhs=True, out_dtype=jnp.float32, multiprocessors=4,
                              interpret=grouped_matmul.INTERPRETED, dense=True, scales=matrix.scales[None])

    with generation.assumed("sm90" if family == "wgmma" else "sm80"):
        np.testing.assert_array_equal(np.asarray(packed(tiles), np.float64), expected)
        with pytest.raises(ValueError):
            packed(dataclasses.replace(tiles, block_k=128))


@pytest.mark.parametrize("family", ["mma.sync", "wgmma"])
def test_the_mainloops_nvfp4_stage_decode_lowers_for_cuda(family):
    """The packed matrix's kernel (`mosaic_gpu.gmm` with `scales`) lowers for CUDA on the CPU on each
    family: its words' and scales' streams, the stage decode in the warpgroup MMA's fragment layout and its
    buffers."""
    words = jax.ShapeDtypeStruct((1, 128, 512 // 8), jnp.uint32)
    scales = jax.ShapeDtypeStruct((1, 128, 512 // 16), jnp.float8_e4m3fn)
    rows = jax.ShapeDtypeStruct((256, 512), jnp.bfloat16)
    name = "sm90" if family == "wgmma" else "sm80"
    tiles = mosaic_gpu.Tiles(64, 64, 256, 2, 1, family=family)

    def packed(rows, words, scales):
        return mosaic_gpu.gmm(rows, words, jnp.asarray([256], jnp.int32), tiles=tiles, transpose_rhs=True,
                              out_dtype=jnp.float32, multiprocessors=4, dense=True, scales=scales)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), name)
    with generation.assumed(name), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(packed).trace(rows, words, scales).lower(lowering_platforms=("cuda",))


@INTERPRETS
@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_an_nvfp4_products_packed_kernel_is_its_decoded_product(generation_name):
    """`gemm.nvfp4_product` with its packed kernel named (`generation.chosen(nvfp4='packed')`: the template's
    mainloop decoding each stage in place, row 3621) gives the decoded product's value and rows' gradient
    bitwise, its route reporting the packed kernel: small integer rows, every product and sum exact in
    fp32."""
    matrix, values = nvfp4(8, 512, 128)
    rows = jnp.asarray(np.random.default_rng(9).integers(-3, 4, size=(96, 512)), jnp.float32)
    cotangent = jnp.asarray(np.random.default_rng(10).integers(-2, 3, size=(96, 128)), jnp.float32)

    def product(x, kernel):
        with generation.chosen(nvfp4=kernel):
            return gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32, interpret_on_cpu=True)

    with generation.assumed(generation_name):
        with generation.routes() as routes:
            got, pull = jax.vjp(lambda x: product(x, "packed"), rows)
        assert {route.kernel for route in routes if route.choice == "nvfp4"} == {"packed"}, routes
        want, pull_xla = jax.vjp(lambda x: product(x, "xla"), rows)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        np.testing.assert_array_equal(np.asarray(pull(cotangent)[0]), np.asarray(pull_xla(cotangent)[0]))
    np.testing.assert_array_equal(np.asarray(got, np.float64), np.asarray(rows, np.float64) @ values)


def nvfp4_bank(seed: int, groups: int, depth: int, width: int) -> tuple[gemm.NvFp4, np.ndarray]:
    """A bank of `groups` NVFP4 matrices (`nvfp4`'s each, its own tensor scale, row 4516) as one
    `gemm.NvFp4` (`[groups, depth, width]`), and its values."""
    matrices = [nvfp4(seed + group, depth, width, scale=2.0 ** -group) for group in range(groups)]
    codes = jnp.stack([matrix.codes() for matrix, _ in matrices])
    scales = jnp.stack([matrix.scales.T for matrix, _ in matrices])
    bank = gemm.NvFp4.of(codes, scales, jnp.asarray([2.0 ** -group for group in range(groups)], jnp.float32))
    return bank, np.stack([values for _, values in matrices])


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_an_nvfp4_banks_packed_grouped_product_is_its_decoded_banks_product(generation_name):
    """`gemm.nvfp4_product` of a bank of four experts' NVFP4 matrices (row 4516: each its own tensor scale)
    over a ragged group table (one group empty, rows past the groups), its packed kernel named (each tile
    decoding its group's expert's stages in place) against the decoded bank's grouped product (XLA's arm),
    bitwise, and both exactly the float64 product of the values: small integer rows, every product and sum
    exact in fp32; the bank takes no gradient."""
    bank, values = nvfp4_bank(11, 4, 512, 128)
    sizes = np.asarray([40, 0, 72, 24])
    rows = jnp.asarray(np.random.default_rng(12).integers(-3, 4, size=(140, 512)), jnp.float32)

    def product(x, kernel):
        with generation.chosen(nvfp4=kernel):
            return gemm.nvfp4_product(x, bank, jnp.asarray(sizes, jnp.int32), out_dtype=jnp.float32,
                                      interpret_on_cpu=True)

    with generation.assumed(generation_name):
        with generation.routes() as routes:
            got = product(rows, "packed")
        assert {route.kernel for route in routes if route.choice == "nvfp4"} == {"packed"}, routes
        want = product(rows, "xla")
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    starts = np.cumsum([0, *sizes])
    truth = np.zeros((140, 128))
    for group, (first, last) in enumerate(itertools.pairwise(starts)):
        truth[first:last] = np.asarray(rows[first:last], np.float64) @ values[group]
    np.testing.assert_array_equal(np.asarray(got, np.float64), truth)


def test_an_nvfp4_bank_packs_its_nibbles_on_the_host_as_its_values():
    """A bank's codes given as uint8 nibbles in NumPy (row 4516: a checkpoint's E2M1 codes, never floats)
    pack on the host, `gemm.NvFp4.of` returning NumPy words, scales and scale, bitwise the words of the
    same codes given as their E2M1 values on the device."""
    rng = np.random.default_rng(13)
    nibbles = rng.integers(0, 16, size=(2, 256, 64)).astype(np.uint8)
    scales = np.exp2(rng.integers(-3, 4, size=(2, 16, 64))).astype(jnp.float8_e4m3fn)
    scale = np.asarray([0.5, 0.25], np.float32)
    host = gemm.NvFp4.of(nibbles, scales, scale)
    assert all(isinstance(field, np.ndarray) for field in (host.words, host.scales, host.scale))
    values = np.where(nibbles >= 8, -1.0, 1.0) * np.asarray(E2M1)[nibbles & 7]
    device = gemm.NvFp4.of(jnp.asarray(values, jnp.float32), jnp.asarray(scales), jnp.asarray(scale))
    np.testing.assert_array_equal(host.words, np.asarray(device.words))


@pytest.mark.parametrize("kernel", ["packed", "xla"])
def test_gemm_takes_an_nvfp4_bank_as_its_matrix(kernel):
    """`gemm.gemm` of a bank of NVFP4 experts (row 4516: moe's projection seam, no branch of its own), its
    bias epilogue on the fp32 sums before the one rounding, its rows gathered from the tokens and scattered
    back by a permutation: bitwise the decoded bank's grouped product of the gathered rows plus the bias,
    the rows put back by the permutation (small integers, every sum exact), through `kernel`; and refused
    with a prologue."""
    bank, values = nvfp4_bank(14, 4, 512, 128)
    rng = np.random.default_rng(15)
    tokens = jnp.asarray(rng.integers(-3, 4, size=(60, 512)), jnp.float32)
    table = jnp.asarray(rng.integers(0, 60, 136), jnp.int32)
    slots = jnp.asarray(rng.permutation(136), jnp.int32)
    sizes = np.asarray([40, 0, 72, 24])
    bias = jnp.asarray(rng.integers(-2, 3, size=(4, 128)), jnp.float32)

    def plus(acc, b):
        return acc + b

    with generation.assumed("sm90"), generation.chosen(nvfp4=kernel):
        got = gemm.gemm(tokens, bank, jnp.asarray(sizes, jnp.int32), out_dtype=jnp.float32, gather=table,
                        scatter=slots, epilogue=plus, epilogue_operands=((bias, ("groups", "columns")),),
                        interpret_on_cpu=True)
        with pytest.raises(ValueError):
            gemm.gemm(tokens, bank, jnp.asarray(sizes, jnp.int32), out_dtype=jnp.float32,
                      prologue=lambda x: x)
    rows = np.asarray(tokens, np.float64)[np.asarray(table)]
    starts = np.cumsum([0, *sizes])
    truth = np.zeros((136, 128))
    for group, (first, last) in enumerate(itertools.pairwise(starts)):
        truth[first:last] = rows[first:last] @ values[group] + np.asarray(bias[group], np.float64)
    placed = np.zeros_like(truth)
    placed[np.asarray(slots)] = truth
    np.testing.assert_array_equal(np.asarray(got, np.float64), placed)


def test_an_nvfp4_products_input_gradient_keeps_its_cotangents_precision():
    """The packed NVFP4 product's input gradient (audit A2) multiplies the cotangent at its own precision:
    a cotangent row of 1 and -511/512 against a matrix of equal values sums to the 1/512 their difference
    is, where the cotangent narrowed to bfloat16 first (-511/512 rounding to -1) summed to 0."""
    matrix = gemm.NvFp4.quantized(jnp.ones((512, 128), jnp.float32))
    rows = jnp.zeros((8, 512), jnp.float32)
    cotangent = jnp.zeros((8, 128), jnp.float32).at[:, 0].set(1.0).at[:, 1].set(-511 / 512)
    with generation.assumed("sm90"), generation.chosen(nvfp4="packed"), generation.routes() as routes:
        _, pullback = jax.vjp(lambda x: gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32,
                                                           interpret_on_cpu=True), rows)
        (d_rows,) = pullback(cotangent)
    assert {route.kernel for route in routes if route.choice == "nvfp4"} == {"packed"}, routes
    np.testing.assert_array_equal(np.asarray(d_rows), np.full((8, 512), 1 / 512, np.float32))
