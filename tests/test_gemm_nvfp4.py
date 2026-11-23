"""The GEMM template's product of an NVFP4 matrix (`gemm.nvfp4_product`, row 3621): its weights' E2M1 codes
and E4M3 block scales decoded exactly (`NvFp4.decoded`, its codes in the staged order a stage decodes), the
deferred factor applied to the fp32 accumulator before the one rounding."""

import dataclasses
import faulthandler
import functools
import itertools

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from generation_support import INTERPRETS, lowers
from jax.sharding import Mesh
from low_precision_support import nvfp4
from reference_error import assert_as_exact

from dew.algebra import bits
from dew.nn import fake_quant
from dew.nn.kernels import formats, gemm, generation, grouped_launch, grouped_matmul, tuning
from dew.nn.kernels.templates.gemm import mosaic_gpu, tiling


def test_an_nvfp4_matrix_holds_its_codes_and_the_staged_order_reads_back():
    """A matrix quantized from fp32 (`formats.NvFp4.quantized`) holds `bits.nvfp4_quantized`'s codes and
    scales: its codes read back in the depths' own order (`NvFp4.codes`) are the quantizer's natural words'
    codes, its scales the quantizer's; and its words are K-major, in the staged order (`formats.staged_rows`,
    within each 128 depths word w holding depths w + 16j), not the natural one's."""
    x = jnp.asarray(np.random.default_rng(5).standard_normal((256, 64)), jnp.float32)
    matrix = formats.NvFp4.quantized(x)
    words, scales, scale = bits.nvfp4_quantized(x)
    natural = bits.nvfp4_decoded(words, jnp.full(scales.shape, bits.NVFP4_OWED, jnp.float8_e4m3fn))
    np.testing.assert_array_equal(np.asarray(matrix.codes()), np.asarray(natural, np.float32))
    np.testing.assert_array_equal(np.asarray(matrix.scales.T, np.float32), np.asarray(scales, np.float32))
    assert float(matrix.scale) == float(scale)
    assert matrix.words.shape == (64, 256 // 8)
    assert not np.array_equal(np.asarray(matrix.words.T), np.asarray(words))
    order = formats.staged_rows(256)
    assert sorted(order) == list(range(256)) and order[8 * 3 + 5] == 16 * 5 + 3 and order[128] == 128


def test_an_nvfp4_product_is_its_decoded_products_times_the_deferred_factor():
    """XLA's route (a CPU's): the rows rounded to bfloat16 times the decoded matrix, summed in fp32, times
    `bits.NVFP4_OWED` and the tensor's scale, then rounded once: bitwise the program written out, and, its
    operands small integers and powers of two, exactly the float64 product of the matrix's values."""
    matrix, values = nvfp4(0, 256, 192, scale=0.25)
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
    matrix, values = nvfp4(2, 256, 256, scale=0.25)
    rows = jnp.asarray(np.random.default_rng(3).integers(-3, 4, size=(128, 256)), jnp.bfloat16)
    cotangent = jnp.asarray(np.random.default_rng(4).integers(-2, 3, size=(128, 256)), jnp.float32)

    def product(x):
        return gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32, interpret_on_cpu=True)

    with generation.assumed(generation_name):
        with generation.chosen(grouped_matmul="mosaic"), generation.routes() as routes:
            got = jax.jit(product)(rows)
        _, pull = jax.vjp(product, rows)
        (gradient,) = pull(cotangent)
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}
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
    matrix, _ = nvfp4(6, 512, 128, scale=0.25)
    rows = jnp.asarray(np.random.default_rng(7).integers(-3, 4, size=(136, 512)), jnp.bfloat16)
    expected = np.asarray(rows, np.float64) @ np.asarray(matrix.decoded(), np.float64)
    tiles = tiling.Tiles(64, 64, 256, 2, 1, family=family)

    def packed(step_tiles):
        return mosaic_gpu.gmm(rows, matrix.words[None], jnp.asarray([136], jnp.int32), tiles=step_tiles,
                              transpose_rhs=True, out_dtype=jnp.float32, multiprocessors=4,
                              interpret=grouped_launch.INTERPRETED, dense=True, scales=matrix.scales[None])

    with generation.assumed("sm90" if family == "wgmma" else "sm80"):
        np.testing.assert_array_equal(np.asarray(packed(tiles), np.float64), expected)
        with pytest.raises(ValueError):
            packed(dataclasses.replace(tiles, block_k=128))


@INTERPRETS
def test_the_mainloop_decodes_an_nvfp4_matrixs_rows_into_its_products_left_operand():
    """`mosaic_gpu.gmm` of an NVFP4 matrix as its rows (`packed_rows`: a decode-sized product swapped, row
    5387) decodes each 256-deep stage of the matrix in registers into the warpgroup MMA's left operand and
    multiplies the call's rows read transposed: the decoded matrix's transpose times the rows', bitwise
    (small integer rows, every product and sum exact), two consumers' row tiles and two steps, interpreted
    on warpgroup MMA; mma.sync, whose register operand's fragment is not the decode's, is refused."""
    matrix, _ = nvfp4(11, 512, 256, scale=0.25)
    rows = jnp.asarray(np.random.default_rng(12).integers(-3, 4, size=(16, 512)), jnp.bfloat16)
    expected = np.asarray(matrix.decoded(), np.float64).T @ np.asarray(rows, np.float64).T

    def swapped(family):
        return mosaic_gpu.gmm(matrix.words, rows[None], jnp.asarray([256], jnp.int32),
                              tiles=tiling.Tiles(128, 16, 256, 2, 1, warpgroups=2, family=family),
                              transpose_rhs=True, out_dtype=jnp.float32, multiprocessors=4,
                              interpret=grouped_launch.INTERPRETED, dense=True, scales=matrix.scales,
                              packed_rows=True)

    with generation.assumed("sm90"):
        np.testing.assert_array_equal(np.asarray(swapped("wgmma"), np.float64), expected)
    with generation.assumed("sm80"), pytest.raises(ValueError):
        swapped("mma.sync")


@pytest.mark.parametrize("family", ["mma.sync", "wgmma"])
def test_the_mainloops_nvfp4_stage_decode_lowers_for_cuda(family):
    """The packed matrix's kernel (`mosaic_gpu.gmm` with `scales`) lowers for CUDA on the CPU on each
    family: its words' and scales' streams, the stage decode in the warpgroup MMA's fragment layout and its
    buffers."""
    words = jax.ShapeDtypeStruct((1, 128, 512 // 8), jnp.uint32)
    scales = jax.ShapeDtypeStruct((1, 128, 512 // 16), jnp.float8_e4m3fn)
    rows = jax.ShapeDtypeStruct((256, 512), jnp.bfloat16)
    name = "sm90" if family == "wgmma" else "sm80"
    tiles = tiling.Tiles(64, 64, 256, 2, 1, family=family)

    def packed(rows, words, scales):
        return mosaic_gpu.gmm(rows, words, jnp.asarray([256], jnp.int32), tiles=tiles, transpose_rhs=True,
                              out_dtype=jnp.float32, multiprocessors=4, dense=True, scales=scales)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), name)
    with generation.assumed(name), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(packed).trace(rows, words, scales).lower(lowering_platforms=("cuda",))


def test_the_mainloops_nvfp4_rows_decode_lowers_for_hopper():
    """The swapped packed kernel (`mosaic_gpu.gmm` with `packed_rows`) lowers for CUDA on the CPU on warpgroup
    MMA: its rows' words and scales streamed at the row tile, each stage decoded in registers in the
    warpgroup MMA's fragment layout and taken as the products' register operand."""
    words = jax.ShapeDtypeStruct((256, 512 // 8), jnp.uint32)
    scales = jax.ShapeDtypeStruct((256, 512 // 16), jnp.float8_e4m3fn)
    rows = jax.ShapeDtypeStruct((1, 16, 512), jnp.bfloat16)
    tiles = tiling.Tiles(128, 16, 256, 2, 1, warpgroups=2, family="wgmma")

    def swapped(words, rows, scales):
        return mosaic_gpu.gmm(words, rows, jnp.asarray([256], jnp.int32), tiles=tiles, transpose_rhs=True,
                              out_dtype=jnp.float32, multiprocessors=4, dense=True, scales=scales,
                              packed_rows=True)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(swapped).trace(words, rows, scales).lower(lowering_platforms=("cuda",))


@INTERPRETS
@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_an_nvfp4_products_packed_kernel_is_its_decoded_product(generation_name):
    """`gemm.nvfp4_product` with its packed kernel named (`generation.chosen(nvfp4='packed')`: the template's
    mainloop decoding each stage in place, row 3621) gives the decoded product's value and rows' gradient
    bitwise, its route reporting the packed kernel: small integer rows, every product and sum exact in
    fp32."""
    matrix, values = nvfp4(8, 512, 128, scale=0.25)
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


def test_an_nvfp4_products_packed_kernel_on_a_cpu_uninterpreted_is_its_decoded_product():
    """`gemm.nvfp4_product` with its packed kernel named for an assumed H100 and not interpreted, in a CPU
    process: its launch's branch is CUDA's alone (`gemm._dispatched`), the CPU's the decoded product, exact
    here (small integer rows), where the launch lowered for the CPU raised "Only interpret mode is supported
    on CPU backend" (row 8492: a packed product's per-shard cell)."""
    matrix, values = nvfp4(8, 512, 128, scale=0.25)
    rows = jnp.asarray(np.random.default_rng(11).integers(-3, 4, size=(96, 512)), jnp.float32)
    with generation.assumed("sm90"), generation.chosen(nvfp4="packed"), generation.routes() as routes:
        got = jax.jit(lambda x: gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32))(rows)
    assert {route.kernel for route in routes if route.choice == "nvfp4"} == {"packed"}, routes
    np.testing.assert_array_equal(np.asarray(got, np.float64), np.asarray(rows, np.float64) @ values)


@INTERPRETS
def test_a_decode_sized_nvfp4_product_swapped_is_its_decoded_product():
    """`gemm.nvfp4_product` at a decode step's 13 rows on an H100, its packed kernel named, derives a tile
    the launch holds legal, and at a swapped tile (pinned: the matrix decoded in registers into the
    warpgroup MMA's left operand, the rows padded to the tile's 64 columns and read transposed, row 5387)
    gives the decoded product's value bitwise over four stages: small integer rows, every product and sum
    exact in fp32."""
    matrix, values = nvfp4(13, 1024, 256, scale=0.25)
    rows = jnp.asarray(np.random.default_rng(14).integers(-3, 4, size=(13, 1024)), jnp.float32)
    swapped = tiling.Tiles(128, 64, 256, 3, 1, warpgroups=2, family="wgmma.fed", swapped=True)
    launch = grouped_matmul.MosaicPackedGmm()

    def product(x, kernel):
        with generation.chosen(nvfp4=kernel):
            return gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32, interpret_on_cpu=True)

    with generation.assumed("sm90"):
        with tuning.seen() as launches:
            product(rows, "packed")
        assert all(launch.legal(ba, config, generation.gpu_target())
                   for (op, _), (config, ba) in zip(launches, launches.taken, strict=True)
                   if isinstance(op, grouped_matmul.MosaicPackedGmm)), launches.taken
        with tuning.pinned({grouped_matmul.MosaicPackedGmm: swapped}):
            got = product(rows, "packed")
        want = product(rows, "xla")
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(np.asarray(got, np.float64), np.asarray(rows, np.float64) @ values)
    np.testing.assert_array_equal(np.asarray(got, np.float64), np.asarray(rows, np.float64) @ values)


def nvfp4_bank(seed: int, groups: int, depth: int, width: int) -> tuple[formats.NvFp4, np.ndarray]:
    """A bank of `groups` NVFP4 matrices (`nvfp4`'s each, its own tensor scale, row 4516) as one
    `formats.NvFp4` (`[groups, depth, width]`), and its values."""
    matrices = [nvfp4(seed + group, depth, width, scale=2.0 ** -group) for group in range(groups)]
    codes = jnp.stack([matrix.codes() for matrix, _ in matrices])
    scales = jnp.stack([matrix.scales.T for matrix, _ in matrices])
    bank = formats.NvFp4.of(codes, scales,
                            jnp.asarray([2.0 ** -group for group in range(groups)], jnp.float32))
    return bank, np.stack([values for _, values in matrices])


@pytest.mark.parametrize("sizes", [(40, 0, 72, 24), (0, 0, 0, 0)], ids=["ragged", "empty"])
@pytest.mark.parametrize("kernel", ["packed", "xla"])
@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_an_nvfp4_banks_grouped_product_is_its_values_product(generation_name, kernel, sizes):
    """`gemm.nvfp4_product` of a bank of four experts' NVFP4 matrices (row 4516: each its own tensor scale)
    over a ragged group table (one group empty, rows past the groups), or a call of no rows, through
    `kernel`: the packed kernel, each tile decoding its group's expert's stages in place (interpreted), or
    the bank decoded once a call;
    exactly the float64 product of the values, small integer rows making every product and sum exact in
    fp32; the bank takes no gradient. A hang dumps every thread's stack (the packed walk was suspected of
    one on an empty group; NSPROBE-PK1 found it walks)."""
    faulthandler.dump_traceback_later(240, exit=True)
    bank, values = nvfp4_bank(11, 4, 512, 128)
    sizes = np.asarray(sizes)
    count = 140 if sizes.sum() else 0
    rows = jnp.asarray(np.random.default_rng(12).integers(-3, 4, size=(count, 512)), jnp.float32)
    with generation.assumed(generation_name), generation.chosen(nvfp4=kernel), generation.routes() as routes:
        got = gemm.nvfp4_product(rows, bank, jnp.asarray(sizes, jnp.int32), out_dtype=jnp.float32,
                                 interpret_on_cpu=True)
    faulthandler.cancel_dump_traceback_later()
    assert {route.kernel for route in routes if route.choice == "nvfp4"} == {kernel}, routes
    starts = np.cumsum([0, *sizes])
    truth = np.zeros((count, 128))
    for group, (first, last) in enumerate(itertools.pairwise(starts)):
        truth[first:last] = np.asarray(rows[first:last], np.float64) @ values[group]
    np.testing.assert_array_equal(np.asarray(got, np.float64), truth)


@pytest.mark.parametrize("gathered", [False, True], ids=["rows", "gathered-rows"])
@pytest.mark.parametrize("kernel", ["packed", "xla"])
def test_a_stack_of_nvfp4_banks_is_read_at_its_layer_as_that_layers_bank(kernel, gathered):
    """`gemm.nvfp4_product` of a stack of three layers' banks (`[layers, groups, ...]` fields, a scanned
    packed load's, row 4861) at a traced layer (`layer`) through `kernel`: the packed kernel reading the
    layer's experts where they lie in the stack (`mosaic_gpu.gmm`'s `first`, interpreted as an H100's), or
    the layer's bank decoded; bit for bit that layer's bank's own product, its rows' gradient too, at each
    layer, over a ragged group table; `gathered`, its rows a routing's gather of the tokens read where they
    lie (a scanned MoE layer's, row 5391), as that bank's product of the same gather."""
    banks = [nvfp4_bank(20 + 4 * layer, 4, 256, 128)[0] for layer in range(3)]
    stack = jax.tree.map(lambda *fields: jnp.stack(fields), *banks)
    sizes = jnp.asarray([40, 0, 56, 24], jnp.int32)
    rows = jnp.asarray(np.random.default_rng(21).integers(-3, 4, size=(128, 256)), jnp.float32)
    cotangent = jnp.asarray(np.random.default_rng(22).integers(-3, 4, size=(128, 128)), jnp.float32)
    table = jnp.asarray(np.random.default_rng(23).integers(0, 100, 128), jnp.int32) if gathered else None

    def product(rows, matrix, layer=None):
        return gemm.nvfp4_product(rows, matrix, sizes, table, layer=layer, out_dtype=jnp.float32,
                                  interpret_on_cpu=True)

    with generation.assumed("sm90"), generation.chosen(nvfp4=kernel), generation.routes() as routes:
        for layer, bank in enumerate(banks):
            got, pull = jax.vjp(lambda x, layer=layer: product(x, stack, jnp.asarray(layer)), rows)
            want, pull_bank = jax.vjp(lambda x, bank=bank: product(x, bank), rows)
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes(), layer
            assert np.asarray(pull(cotangent)[0]).tobytes() == np.asarray(pull_bank(cotangent)[0]).tobytes()
    assert {route.kernel for route in routes if route.choice == "nvfp4"} == {kernel}, routes


def test_an_nvfp4_bank_packs_its_nibbles_on_the_host_as_its_values():
    """A bank's codes given as uint8 nibbles in NumPy (row 4516: a checkpoint's E2M1 codes, never floats)
    pack on the host, `formats.NvFp4.of` returning NumPy words, scales and scale, bitwise the words of the
    same codes given as their E2M1 values on the device."""
    rng = np.random.default_rng(13)
    nibbles = rng.integers(0, 16, size=(2, 256, 64)).astype(np.uint8)
    scales = np.exp2(rng.integers(-3, 4, size=(2, 16, 64))).astype(jnp.float8_e4m3fn)
    scale = np.asarray([0.5, 0.25], np.float32)
    host = formats.NvFp4.of(nibbles, scales, scale)
    assert all(isinstance(field, np.ndarray) for field in (host.words, host.scales, host.scale))
    values = fake_quant.E2M1[nibbles]
    device = formats.NvFp4.of(jnp.asarray(values, jnp.float32), jnp.asarray(scales), jnp.asarray(scale))
    np.testing.assert_array_equal(host.words, np.asarray(device.words))


@pytest.mark.parametrize("kernel", ["packed", "xla"])
def test_gemm_takes_an_nvfp4_bank_as_its_matrix(kernel):
    """`gemm.gemm` of a bank of NVFP4 experts (row 4516: moe's projection seam, no branch of its own), its
    bias epilogue on the fp32 sums before the one rounding, its rows gathered from the tokens and scattered
    back by a permutation: bitwise the decoded bank's grouped product of the gathered rows plus the bias,
    the rows put back by the permutation (small integers, every sum exact), through `kernel` (the packed
    kernel's producer gathering the rows, interpreted); and refused with a prologue. A hang dumps every
    thread's stack."""
    faulthandler.dump_traceback_later(240, exit=True)
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
    faulthandler.cancel_dump_traceback_later()
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
    matrix = formats.NvFp4.quantized(jnp.ones((512, 128), jnp.float32))
    rows = jnp.zeros((8, 512), jnp.float32)
    cotangent = jnp.zeros((8, 128), jnp.float32).at[:, 0].set(1.0).at[:, 1].set(-511 / 512)
    with generation.assumed("sm90"), generation.chosen(nvfp4="packed"), generation.routes() as routes:
        _, pullback = jax.vjp(lambda x: gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32,
                                                           interpret_on_cpu=True), rows)
        (d_rows,) = pullback(cotangent)
    assert {route.kernel for route in routes if route.choice == "nvfp4"} == {"packed"}, routes
    np.testing.assert_array_equal(np.asarray(d_rows), np.full((8, 512), 1 / 512, np.float32))


def test_a_bank_carries_its_forms_version_and_a_load_refuses_another(tmp_path):
    """`formats.NvFp4` stores the version of the form its words are packed by (`formats.FORM`, the encoder's
    digest, `bits.BitForm.version`; algebra.md 7.4, row 6101), a stacked bank's each layer, and a checkpoint
    restore of words another encoder packed is refused (`NvFp4.checked`): written and read back with its
    version, a bank restores; the same bank stored under another version is refused by name."""
    from dew.checkpoints import Checkpoints

    bank, _ = nvfp4_bank(3, 2, 256, 128)
    assert bytes(np.asarray(bank.form)).hex() == formats.FORM.tobytes().hex()
    stacked = jax.tree.map(lambda *layers: jnp.stack(layers), bank, bank)
    assert bytes(np.asarray(stacked.layer(jnp.int32(1)).form)).hex() == formats.FORM.tobytes().hex()
    reordered = formats.form_version(bits.NVFP4_BFLOAT16_FORM.version,
                                     formats.staged_rows(formats.NVFP4_STAGE)[::-1])
    assert not np.array_equal(reordered, formats.FORM)
    with pytest.raises(ValueError):
        dataclasses.replace(bank, form=reordered).checked()
    other = dataclasses.replace(bank, form=np.frombuffer(bytes.fromhex("0" * 16), np.uint8).copy())
    with pytest.raises(ValueError):
        other.checked()
    def template(of):
        where = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=where), of)

    for name, written, refused in (("same", bank, False), ("other", other, True)):
        checkpoints = Checkpoints(str(tmp_path / name))
        checkpoints.save(1, {"bank": written})
        checkpoints.wait()
        if refused:
            with pytest.raises(ValueError):
                Checkpoints(str(tmp_path / name)).restore(template({"bank": written}))
        else:
            restored = Checkpoints(str(tmp_path / name)).restore(template({"bank": written}))[0]["bank"]
            np.testing.assert_array_equal(np.asarray(restored.words), np.asarray(written.words))
    if len(jax.devices()) >= 2:
        # Row 8958: a two-layer bank sharded by layer across two devices with a foreign digest on shard 1
        # must be refused both locally (`checked()`) and through collective checkpoint restore.
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("layers",))
        by_layer = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("layers"))
        mixed = jax.tree.map(lambda *layers: jax.device_put(jnp.stack(layers), by_layer), bank, other)
        with pytest.raises(ValueError):
            mixed.checked()
        sharded_ckpt = Checkpoints(str(tmp_path / "sharded_mixed"))
        sharded_ckpt.save(1, {"bank": mixed})
        sharded_ckpt.wait()
        sharded_template = jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=by_layer), {"bank": mixed})
        with pytest.raises(ValueError):
            Checkpoints(str(tmp_path / "sharded_mixed")).restore(sharded_template)


@pytest.mark.parametrize("group_sizes", [None, (8, 8)], ids=["matrix", "expert bank"])
def test_an_nvfp4_product_over_operands_a_mesh_splits_decodes_its_matrix(group_sizes):
    """An NVFP4 product at the H100's 16-row decode cell (`gemm.NVFP4_CELLS`), a matrix's or an expert
    bank's, takes its packed launch over whole operands and decodes its matrix where a mesh XLA places
    splits them (an automatic axis of two, `sharding.automatic_axes`), as every GEMM selector's launch reads
    one device's operands whole (row 11642: a Linear forced XLA by hand beside it, and an expert bank's
    product took the packed launch under such a mesh)."""
    from jax.sharding import AbstractMesh, AxisType

    if group_sizes is None:
        matrix, sizes = nvfp4(0, 512, 128, scale=0.25)[0], None
    else:
        rng = np.random.default_rng(1)
        codes = rng.integers(0, 16, size=(len(group_sizes), 512, 128)).astype(np.uint8)
        powers = rng.integers(-3, 4, size=(len(group_sizes), 32, 128))
        scales = np.exp2(powers).astype(ml_dtypes.float8_e4m3fn)
        matrix = formats.NvFp4.of(codes, scales, np.full(len(group_sizes), 0.25, np.float32))
        sizes = jnp.asarray(group_sizes, jnp.int32)
    rows = jax.ShapeDtypeStruct((16, 512), jnp.bfloat16)

    def taken(mesh):
        with (generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh),
              generation.routes() as routes):
            jax.eval_shape(lambda x: gemm.nvfp4_product(x, matrix, sizes, out_dtype=jnp.float32), rows)
        return {route.kernel for route in routes if route.choice == "nvfp4"}

    whole = AbstractMesh((1,), ("data",), axis_types=(AxisType.Auto,))
    split = AbstractMesh((2,), ("data",), axis_types=(AxisType.Auto,))
    assert taken(whole) == {"packed"}
    assert taken(split) == {"xla"}


@INTERPRETS
@pytest.mark.parametrize(("minifloat", "out_dtype"),
                         [(bits.E4M3, jnp.float8_e4m3fn), (bits.E5M2, jnp.float8_e5m2)], ids=["e4m3", "e5m2"])
def test_gemm_quantize_epilogue_matches_bits_encoded(minifloat, out_dtype):
    """`gemm.gemm_quantized` encodes the GEMM accumulator via `bits.encode_form` bit-exact with
    `bits.encoded` across E4M3 and E5M2 (rows 8547, 12037)."""
    rng = np.random.default_rng(17)
    lhs = jnp.asarray(rng.integers(-3, 4, size=(128, 128)), jnp.bfloat16)
    rhs = jnp.asarray(rng.integers(-3, 4, size=(128, 128)), jnp.bfloat16)
    acc = jnp.dot(lhs, rhs, preferred_element_type=jnp.float32)
    words = jax.lax.bitcast_convert_type(acc, jnp.uint32)
    expected_u8 = bits.encoded(words, minifloat).astype(jnp.uint8)
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        got = gemm.gemm_quantized(lhs, rhs, minifloat=minifloat, interpret_on_cpu=True)
    assert isinstance(got, jax.Array) and got.dtype == out_dtype
    got_u8 = jax.lax.bitcast_convert_type(got, jnp.uint8)
    np.testing.assert_array_equal(np.asarray(got_u8), np.asarray(expected_u8))


@pytest.mark.parametrize("gpu", [pytest.param("sm80", marks=lowers("sm80")),
                                 pytest.param("sm90", marks=lowers("sm90"))])
def test_gemm_quantize_epilogue_lowers_inline_ptx(gpu):
    """`gemm.gemm_quantized` lowers FP8 and MXFP8 epilogues with `bits.encode_form`'s inline PTX on sm80
    and sm90 (rows 8547, 12037)."""
    x = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), gpu)
    with (generation.assumed(gpu), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(grouped_matmul="mosaic")):
        jax.jit(lambda a, b: gemm.gemm_quantized(a, b, minifloat=bits.E4M3)).trace(x, w).lower(
            lowering_platforms=("cuda",))
        jax.jit(lambda a, b: gemm.gemm_quantized(a, b, minifloat=formats.MxFp8)).trace(x, w).lower(
            lowering_platforms=("cuda",))


def test_packed_rows_supported_reads_bundle_fragment_name():
    """`packed.rows_supported` checks `family.name` prefix in `bits.NVFP4_BFLOAT16.fragments` and
    `hooks._epilogue_layout` reads the issuing GPU bundle's `family.layout` directly (row 12038)."""
    from dew.nn.kernels.templates import bundles
    from dew.nn.kernels.templates.gemm import hooks, packed

    sm90_bundles = bundles.groups("sm90", jnp.bfloat16, stated=True)[0]
    sm80_bundle = bundles.of("sm80", jnp.bfloat16)
    assert all(packed.rows_supported(family) for family in sm90_bundles)
    assert not packed.rows_supported(sm80_bundle)
    assert hooks._epilogue_layout(jnp.bfloat16, generation.gpu_target("sm90")) == sm90_bundles[0].layout
    assert hooks._epilogue_layout(jnp.bfloat16, generation.gpu_target("sm80")) == sm80_bundle.layout
    assert hooks._epilogue_layout(jnp.bfloat16, generation.gpu_target("v5e")) is None

def test_fp8_and_e8m0_bit_forms_are_exact():
    """`E4M3_BFLOAT16_FORM`, `E5M2_BFLOAT16_FORM`, and `E8M0_SCALE_FORM` are proven bit-exact on every code
    against `ml_dtypes` / IEEE values, and `MxFp8.decoded` handles subnormal `0x00` and `0xFF` NaN across
    fp32 and fp64 (rows 3733, 8548, 9040)."""
    bits.proven(bits.E4M3_BFLOAT16, bits.E4M3_BFLOAT16_FORM)
    bits.proven(bits.E5M2_BFLOAT16, bits.E5M2_BFLOAT16_FORM)
    for minifloat, fp8_dtype in ((bits.E4M3, jnp.float8_e4m3fn), (bits.E5M2, jnp.float8_e5m2)):
        codes = np.arange(256, dtype=np.uint32).reshape(64, 4)
        packed_words = bits.fp8_packed(jnp.asarray(codes), minifloat)
        got = np.asarray(bits.fp8_codes(packed_words, minifloat, jnp.float32)).reshape(256)
        expected = np.asarray(
            jax.lax.bitcast_convert_type(jnp.arange(256, dtype=jnp.uint8), fp8_dtype).astype(jnp.float32))
        nans = np.isnan(expected)
        assert np.array_equal(np.isnan(got), nans)
        np.testing.assert_array_equal(got[~nans], expected[~nans])

    # Exhaustive check of branchless E8M0_SCALE_FORM across all 256 bytes and high-byte noise.
    all_bytes = np.arange(256, dtype=np.uint32)
    exp_normal = np.ldexp(np.ones(254, dtype=np.float32), np.arange(1, 255) - 127)
    for high in (0, 0xAB00, 0xFFFF00, 0x12345600):
        val_np = np.asarray(bits.e8m0_decoded(jnp.asarray(all_bytes | high, jnp.uint32)))
        assert val_np[0] == np.float32(2.0 ** -127)
        assert np.isnan(val_np[255])
        np.testing.assert_array_equal(val_np[1:255], exp_normal)
    # MxFp8.decoded preserves 0x00 (2^-127) without FTZ flush on normal outputs (4 * 2^-127 = 2^-125 in fp32
    # and fp64; 1 * 2^-127 in fp64) and 0xFF (NaN).
    raw_codes = jnp.full((1, 64), 4.0, jnp.float8_e4m3fn)
    e8m0_scales = jnp.asarray([[0, 255]], jnp.uint8)
    mx = formats.MxFp8(raw_codes, e8m0_scales)
    dec32 = np.asarray(mx.decoded(jnp.float32))
    np.testing.assert_array_equal(dec32[0, :32], np.full((32,), np.float32(4.0 * 2.0 ** -127)))
    assert np.all(np.isnan(dec32[0, 32:]))
    with jax.enable_x64():
        dec64 = np.asarray(mx.decoded(jnp.float64))
        np.testing.assert_array_equal(dec64[0, :32], np.full((32,), np.float64(4.0 * 2.0 ** -127)))
        assert np.all(np.isnan(dec64[0, 32:]))
        mx_one = formats.MxFp8(jnp.ones((1, 64), jnp.float8_e4m3fn), e8m0_scales)
        np.testing.assert_array_equal(np.asarray(mx_one.decoded(jnp.float64))[0, :32],
                                      np.full((32,), np.float64(2.0 ** -127)))
    # Fp8Weights.checked() accepts valid form and rejects tampered form version.
    sample = formats.Fp8Weights.of(np.zeros((64, 64), np.float32), fmt="e4m3")
    assert sample.checked() is sample
    bad = dataclasses.replace(sample, form=np.zeros((8,), np.uint8))
    with pytest.raises(ValueError):
        bad.checked()

@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_fp8_weight_product_holds_check_1_and_compiles_on_sm80(fmt: str):
    """`gemm.fp8_weight_product` (inline mainloop decode via `E4M3_BFLOAT16_FORM`/`E5M2_BFLOAT16_FORM`)
    holds check 1 on both the default XLA route and interpreted Mosaic route with non-unit scalar,
    per-column, and grouped scales, refuses invalid scale shapes, and compiles spill-free on sm80
    (rows 3733, 8548)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels
    from low_precision_support import decoded_weight_dot

    def draw(seed):
        rng = np.random.default_rng(8548 + seed)
        x = jnp.asarray(rng.standard_normal((128, 64)) / np.sqrt(64), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((64, 64)) * 0.5, jnp.float32)
        scale = jnp.asarray(rng.uniform(0.5, 2.5, size=(64,)), jnp.float32)
        return x, formats.Fp8Weights.of(w, fmt=fmt, scale=scale)

    for interpret in (False, True):
        def dew_fn(x, weights, *, interpret=interpret):
            return gemm.fp8_weight_product(x, weights, out_dtype=jnp.float32, interpret_on_cpu=interpret)

        assert_as_exact(jax.jit(dew_fn), jax.jit(decoded_weight_dot), draw,
                        f"fp8_weight_product {fmt} interp={interpret}",
                        seeds=8, truth_fn=functools.partial(decoded_weight_dot, dtype=jnp.float64))
    # Exact single-scale check on default XLA and interpreted Mosaic routes (scale = 2.0 must apply once).
    ones = jnp.ones((64, 64), jnp.bfloat16)
    scaled_w = formats.Fp8Weights.of(jnp.full((64, 64), 2.0, jnp.float32), fmt=fmt, scale=2.0)
    for interpret in (False, True):
        got = gemm.fp8_weight_product(ones, scaled_w, out_dtype=jnp.float32, interpret_on_cpu=interpret)
        np.testing.assert_array_equal(np.asarray(got), np.full((64, 64), 128.0, np.float32))
    # Grouped fp8_weight_product across scalar, per-column, per-group, and per-group-per-column scales.
    rng = np.random.default_rng(8549)
    x_g = jnp.asarray(rng.integers(-2, 3, size=(64, 64)), jnp.bfloat16)
    w_g = jnp.asarray(rng.integers(-2, 3, size=(2, 64, 64)), jnp.float32)
    sizes = jnp.asarray([40, 24], jnp.int32)
    scale_cases = [
        jnp.asarray(2.0, jnp.float32),
        jnp.asarray(rng.choice([0.5, 1.0, 2.0], size=(64,)), jnp.float32),
        jnp.asarray([[[0.5]], [[2.0]]], jnp.float32),
        jnp.asarray(rng.choice([0.5, 1.0, 2.0], size=(2, 1, 64)), jnp.float32),
    ]
    for scale in scale_cases:
        bank = formats.Fp8Weights.of(w_g, fmt=fmt, scale=scale)
        ref = jax.lax.ragged_dot(x_g.astype(jnp.float32), bank.decoded(jnp.float32), sizes,
                                 preferred_element_type=jnp.float32)
        for interpret in (False, True):
            got = gemm.fp8_weight_product(x_g, bank, sizes, out_dtype=jnp.float32,
                                          interpret_on_cpu=interpret)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    with pytest.raises(ValueError):
        formats.Fp8Weights.of(w_g, fmt=fmt, scale=jnp.ones((2,), jnp.float32))
    with pytest.raises(ValueError):
        formats.Fp8Weights.of(w_g, fmt=fmt, scale=jnp.ones((2, 64), jnp.float32))

    if generation.compile_only_gpus("sm80"):
        x_s = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16)
        w_sample = formats.Fp8Weights.of(np.zeros((128, 128), np.float32), fmt=fmt)
        w_spec = jax.tree.map(lambda a: jax.ShapeDtypeStruct(np.shape(a), np.asarray(a).dtype), w_sample)

        def compiled_fp8(a, w):
            with generation.chosen(grouped_matmul="mosaic"):
                return gemm.fp8_weight_product(a, w, out_dtype=jnp.bfloat16)

        with generation.assumed("sm80"), generation.chosen(grouped_matmul="mosaic"):
            records = compile_mosaic_kernels(
                [KernelCase(f"fp8_weight_{fmt}_sm80", lambda: (compiled_fp8, (x_s, w_spec)))], "sm80")
        for rec in records:
            assert not rec.get("error"), rec
            assert rec["kernels"], rec
            assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_stage_decoded_fp8_preserves_nonfinite_codewords(fmt: str):
    """`packed.stage_decoded_fp8` shares `bits.fp8_restored` with `bits.fp8_codes`, preserving E4M3 NaN
    (`0x7F`/`0xFF`) and E5M2 +/-inf/NaN (`0x7C`/`0xFC`/`0x7D`..`0x7F`) bitwise identically (row 8548)."""
    from dew.nn.kernels.templates.gemm import packed

    minifloat = packed.fp8_minifloat(fmt)
    codes_1d = np.arange(256, dtype=np.uint32)
    codes = np.broadcast_to(codes_1d[:64, None], (64, 64)).copy()
    codes[0, 0] = 0x7F
    codes[1, 0] = 0xFF
    if fmt == "e5m2":
        codes[2, 0] = 0x7C
        codes[3, 0] = 0xFC
        codes[4, 0] = 0x7D
        codes[5, 0] = 0xFD
    staged_codes = codes[formats.fp8_staged_rows(64), :]
    packed_words = bits.fp8_packed(jnp.asarray(staged_codes.T, jnp.uint32), minifloat)

    from low_precision_support import StageRef

    out = StageRef(np.zeros((64, 64), dtype=jnp.bfloat16))
    packed.stage_decoded_fp8([StageRef(packed_words)], out, fmt, interpret=True)
    inv = np.argsort(formats.fp8_staged_rows(64))
    expected = bits.fp8_codes(packed_words.T, minifloat, jnp.bfloat16)[inv].T
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(out.arr, jnp.bfloat16), jnp.uint16)),
        np.asarray(jax.lax.bitcast_convert_type(expected, jnp.uint16)),
    )
    unpermuted = np.asarray(out.arr, np.float32)
    assert np.isnan(unpermuted[0, 0]) and np.isnan(unpermuted[0, 1])
    if fmt == "e5m2":
        assert np.isposinf(unpermuted[0, 2]) and np.isneginf(unpermuted[0, 3])
        assert np.isnan(unpermuted[0, 4]) and np.isnan(unpermuted[0, 5])
    # Feed pre-encoded NaN and +-Inf through the public gemm.fp8_weight_product boundary (row 12100).
    col_codes = np.zeros((64, 64), dtype=np.uint32)
    col_codes[0, 0], col_codes[0, 1] = 0x7F, 0xFF
    if fmt == "e5m2":
        col_codes[0, 2], col_codes[0, 3] = 0x7C, 0xFC
        col_codes[0, 4], col_codes[0, 5] = 0x7D, 0xFD
    col_words = bits.fp8_packed(jnp.asarray(col_codes[formats.fp8_staged_rows(64), :].T, jnp.uint32),
                                minifloat)
    bank = formats.Fp8Weights(col_words, jnp.ones((), jnp.float32), formats.fp8_form_version(fmt), fmt=fmt)
    sel = jnp.zeros((64, 64), jnp.bfloat16).at[:, 0].set(1.0)
    for interpret in (False, True):
        got = np.asarray(gemm.fp8_weight_product(sel, bank, out_dtype=jnp.float32,
                                                 interpret_on_cpu=interpret))
        assert np.isnan(got[0, 0]) and np.isnan(got[0, 1])
        if fmt == "e5m2":
            assert np.isposinf(got[0, 2]) and np.isneginf(got[0, 3])
            assert np.isnan(got[0, 4]) and np.isnan(got[0, 5])


@pytest.mark.parametrize("gpu", [pytest.param("sm80", marks=lowers("sm80")),
                                 pytest.param("sm90", marks=lowers("sm90"))])
def test_gemm_quantized_nvfp4_block_matches_bits_nvfp4_coded_and_lowers(gpu: str):
    """`gemm.gemm_quantized` emits `formats.NvFp4Block` and `formats.MxFp8` bitwise matching
    `formats.NvFp4Block.quantized` and `formats.MxFp8.quantized`, and lowers through Mosaic GPU on sm80/sm90
    (rows 8547, 12037)."""
    rng = np.random.default_rng(8547)
    x = jnp.asarray(rng.standard_normal((64, 64)) / np.sqrt(64), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((64, 64)) / np.sqrt(64), jnp.bfloat16)
    acc = jax.lax.dot(x, w, preferred_element_type=jnp.float32)
    ref_nv = formats.NvFp4Block.quantized(acc)
    got_nv = jax.jit(lambda a, b, s: gemm.gemm_quantized(
        a, b, minifloat=formats.NvFp4Block, tensor_scale=s))(x, w, ref_nv.scale)
    assert isinstance(got_nv, formats.NvFp4Block)
    np.testing.assert_array_equal(np.asarray(got_nv.codes), np.asarray(ref_nv.codes))
    np.testing.assert_array_equal(np.asarray(got_nv.scales), np.asarray(ref_nv.scales))
    np.testing.assert_array_equal(np.asarray(got_nv.scale), np.asarray(ref_nv.scale))
    np.testing.assert_array_equal(np.asarray(got_nv.decoded()), np.asarray(ref_nv.decoded()))

    ref_mx = formats.MxFp8.quantized(acc)
    got_mx = jax.jit(lambda a, b: gemm.gemm_quantized(a, b, minifloat=formats.MxFp8))(x, w)
    assert isinstance(got_mx, formats.MxFp8)
    np.testing.assert_array_equal(np.asarray(got_mx.codes), np.asarray(ref_mx.codes))
    np.testing.assert_array_equal(np.asarray(got_mx.scales), np.asarray(ref_mx.scales))
    np.testing.assert_array_equal(np.asarray(got_mx.decoded()), np.asarray(ref_mx.decoded()))
    with pytest.raises(ValueError):
        gemm.gemm_quantized(x, jnp.ones((64, 33), jnp.bfloat16), minifloat=formats.MxFp8)
    with pytest.raises(ValueError):
        gemm.gemm_quantized(x, jnp.ones((64, 24), jnp.bfloat16), minifloat=formats.NvFp4Block)
    x_s = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    w_s = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    s_s = jax.ShapeDtypeStruct((), jnp.float32)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), gpu)
    with (generation.assumed(gpu), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(grouped_matmul="mosaic")):
        jax.jit(lambda a, b, s: gemm.gemm_quantized(a, b, minifloat=bits.E2M1, tensor_scale=s)).trace(
            x_s, w_s, s_s).lower(lowering_platforms=("cuda",))
