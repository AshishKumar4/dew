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

from dew.algebra import bits
from dew.nn import fake_quant
from dew.nn.kernels import formats, gemm, generation, grouped_launch, grouped_quantized, tuning
from dew.nn.kernels.templates.gemm import mosaic_gpu, quantized, tiling
from dew.nn.kernels.templates.hooks import Hook, Side


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
    owed = jnp.float32(bits.NVFP4_OWED) * jnp.asarray(matrix.scale, jnp.float32)
    written = jnp.dot(rows, matrix.decoded(), preferred_element_type=jnp.float32) * owed
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


@pytest.mark.parametrize("family, gathered", [("mma.sync", False), ("wgmma", False), ("wgmma", True)],
                         ids=["mma.sync", "wgmma", "wgmma gathered"])
def test_the_mainloops_nvfp4_stage_decode_lowers_for_cuda(family, gathered):
    """The packed matrix's kernel (`mosaic_gpu.gmm` with `scales`) lowers for CUDA on the CPU on each
    family: its words' and scales' streams, the stage decode in the warpgroup MMA's fragment layout and its
    buffers; and on Hopper with its rows gathered by a table (a mixture of experts' tokens, the producer's
    128-byte-swizzled ring), whose decoded buffer the products read beside the rows under one transform
    (row 14111: its 16-column stores held it to a narrower swizzle, which layout inference could not
    assign beside the rows', on a GH200 and in this lowering alike)."""
    words = jax.ShapeDtypeStruct((1, 128, 512 // 8), jnp.uint32)
    scales = jax.ShapeDtypeStruct((1, 128, 512 // 16), jnp.float8_e4m3fn)
    rows = jax.ShapeDtypeStruct((60 if gathered else 256, 512), jnp.bfloat16)
    table = jax.ShapeDtypeStruct((256,), jnp.int32)
    name = "sm90" if family == "wgmma" else "sm80"
    tiles = tiling.Tiles(64, 64, 256, 2 if not gathered else 3, 1, family=family)

    def packed(rows, words, scales, table):
        return mosaic_gpu.gmm(rows, words, jnp.asarray([256], jnp.int32), tiles=tiles, transpose_rhs=True,
                              out_dtype=jnp.float32, multiprocessors=4, dense=not gathered, scales=scales,
                              gather=table if gathered else None)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), name)
    with generation.assumed(name), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(packed).trace(rows, words, scales, table).lower(lowering_platforms=("cuda",))


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
    launch = grouped_quantized.MosaicPackedGmm()

    def product(x, kernel):
        with generation.chosen(nvfp4=kernel):
            return gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32, interpret_on_cpu=True)

    with generation.assumed("sm90"):
        with tuning.seen() as launches:
            product(rows, "packed")
        assert all(launch.legal(observed.arguments, observed.config, generation.gpu_target())
                   for observed in launches
                   if isinstance(observed.op, grouped_quantized.MosaicPackedGmm)), launches
        with tuning.pinned({grouped_quantized.MosaicPackedGmm: swapped}):
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


def a_gathered_packed_nvfp4_bank_is_its_values_product():
    """On a Hopper, a bank of four NVFP4 experts named packed (the mainloop decoding each stage in place) over
    rows its producer gathers by a table (`gemm.gemm`'s `gather`: a mixture of experts' tokens read where
    they lie), natively: bitwise its values' product over the gathered rows, XLA's decoded route's and the
    float64 truth's, small integers making every sum exact (row 14111: the launch failed Mosaic's layout
    inference on a GH200). Run by tests/test_gpu_gemm.py's cells."""
    bank, values = nvfp4_bank(14, 4, 512, 128)
    rng = np.random.default_rng(15)
    tokens = jnp.asarray(rng.integers(-3, 4, size=(60, 512)), jnp.bfloat16)
    table = jnp.asarray(rng.integers(0, 60, 136), jnp.int32)
    sizes = np.asarray([40, 0, 72, 24])

    def product(route):
        with generation.chosen(nvfp4=route):
            return np.asarray(jax.jit(lambda x: gemm.gemm(x, bank, jnp.asarray(sizes, jnp.int32),
                                                          out_dtype=jnp.float32, gather=table))(tokens))

    rows = np.asarray(tokens, np.float64)[np.asarray(table)]
    starts = np.cumsum([0, *sizes])
    truth = np.zeros((136, 128))
    for group, (first, last) in enumerate(itertools.pairwise(starts)):
        truth[first:last] = rows[first:last] @ values[group]
    packed, decoded = product("packed"), product("xla")
    np.testing.assert_array_equal(decoded, truth.astype(np.float32))
    np.testing.assert_array_equal(packed, decoded)


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


@pytest.mark.parametrize("grouped", [False, True], ids=["matrix", "bank"])
@pytest.mark.parametrize("kernel", ["packed", "xla"])
def test_nvfp4_members_side_by_side_are_their_products_apart(kernel, grouped):
    """Two NVFP4 members (a gate and an up projection, 128 columns each) of tensor scales of their own, side
    by side as one matrix whose tensor scale is one a column (`NvFp4.columned`, row 13743: a ModelOpt
    source's fused experts), through `kernel` on an H100 (interpreted) and as a bank of three experts each
    its own pair: `gemm.nvfp4_product` of it is exactly the float64 product of each member's values times
    its own scale, side by side (small integer rows, dyadic values: every product and sum exact in fp32),
    the decoded arm's inner product XLA's; the matrix checks its scale one a matrix or one a column
    (`NvFp4.checked`)."""
    experts = 3 if grouped else 1
    drawn = [nvfp4_bank(21 + member, experts, 512, 128) for member in range(2)]
    # Each member's own tensor scales, five binades apart from the other's.
    members = [dataclasses.replace(bank, scale=jnp.asarray(bank.scale) * 2.0 ** (5 * member))
               for member, (bank, _) in enumerate(drawn)]
    values = np.concatenate([held * 2.0 ** (5 * member) for member, (_, held) in enumerate(drawn)], axis=-1)
    fused = formats.NvFp4(jnp.concatenate([bank.words for bank in members], axis=-2),
                          jnp.concatenate([bank.scales for bank in members], axis=-2),
                          jnp.concatenate([jnp.repeat(jnp.asarray(bank.scale)[..., None], 128, axis=-1)
                                           for bank in members], axis=-1))
    assert fused.columned and fused.checked() is fused
    if not grouped:
        fused, values = jax.tree.map(lambda leaf: leaf[0], fused), values[0]
    counts = (40, 0, 56) if grouped else (96,)
    sizes = jnp.asarray(counts, jnp.int32) if grouped else None
    rows = jnp.asarray(np.random.default_rng(13743).integers(-3, 4, size=(96, 512)), jnp.float32)
    inner = {} if kernel == "packed" else {"grouped_matmul": "xla"}
    with generation.assumed("sm90"), generation.chosen(nvfp4=kernel, **inner):
        got = np.asarray(gemm.nvfp4_product(rows, fused, sizes, out_dtype=jnp.float32, interpret_on_cpu=True))
    truth = np.zeros((96, 256))
    for group, (first, last) in enumerate(itertools.pairwise(np.cumsum([0, *counts]))):
        truth[first:last] = np.asarray(rows[first:last], np.float64) @ (values[group] if grouped else values)
    np.testing.assert_array_equal(np.asarray(got, np.float64), truth)
    with pytest.raises(ValueError):
        dataclasses.replace(fused, scale=jnp.ones((7,), jnp.float32)).checked()


def test_a_columned_nvfp4_linear_binds_as_a_dense_kernel():
    """A Linear's NVFP4 kernel whose tensor scale is one a column (`NvFp4.columned`, row 13743) binds by its
    matrix's rank as a dense kernel (`quantized_loads.with_banks`: transposed from the source's `[out, in]`),
    not as a bank by its scale's (train 1390's review); a packed matrix of another rank is refused by name."""
    from dew.interop.quantized_loads import with_banks
    from dew.interop.streaming import WeightLayout
    from dew.quantization import QUANTIZATION

    bank, _ = nvfp4_bank(31, 1, 512, 128)
    dense = jax.tree.map(lambda leaf: leaf[0], dataclasses.replace(
        bank, scale=jnp.repeat(jnp.asarray(bank.scale)[:, None], 128, axis=-1)))
    assert dense.columned and dense.shape == (512, 128)
    layout = WeightLayout("proj.weight", (("params", "proj", "kernel"),), (128, 512), transpose=(1, 0))
    variables = {"params": {"proj": {"kernel": jnp.zeros((512, 128), jnp.bfloat16)}}}
    bound = with_banks(variables, (layout,), {"proj.weight": dense})
    assert bound[QUANTIZATION]["proj"]["kernel"] is dense and "proj" not in bound["params"]
    # A stack of banks (rank 4) under an identity layout of its shape, which binds a bank as it stands: the
    # rank alone refuses it.
    stacked = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf]), bank)
    whole = WeightLayout("experts.weight", (("params", "experts"),), stacked.shape)
    with pytest.raises(ValueError):
        with_banks({"params": {"experts": jnp.zeros(stacked.shape, jnp.bfloat16)}}, (whole,),
                   {"experts.weight": stacked})


def _cancelling_columns(depth: int, width: int) -> formats.NvFp4:
    """An NVFP4 matrix of codes +1, -1, +1 at depth 0 of its first three columns, unit block scales, and
    tensor scales one a column (`NvFp4.columned`): 4, 8 and 1 there, 1 elsewhere."""
    codes = np.zeros((depth, width), np.float32)
    codes[0, :3] = (1.0, -1.0, 1.0)
    scale = np.ones((width,), np.float32)
    scale[:3] = (4.0, 8.0, 1.0)
    return formats.NvFp4.of(jnp.asarray(codes), jnp.ones((depth // 16, width), jnp.float8_e4m3fn),
                            jnp.asarray(scale))


@pytest.mark.parametrize("gathered", [False, True], ids=["dense", "gathered-bank"])
@pytest.mark.parametrize("kernel", ["packed", "xla"])
def test_an_nvfp4_rows_gradient_cancels_across_column_scales(kernel, gathered):
    """Codes +1, -1, +1 at one depth of three columns whose tensor scales are 4, 8 and 1 (`NvFp4.columned`),
    unit block scales, and the output's cotangents 2^126, 2^125 and 1 there, through `kernel` on an H100
    (interpreted; the decoded arm's inner product XLA's), dense and as a bank of one expert read by duplicate
    gather routes: the rows' gradient at that depth is 2^128 - 2^128 + 1 = 1 a route (a token's its routes'
    sum), each product scaled apart overflowing (train 1390's review of row 13743: the column factor's
    transpose bypassed `factored`'s range-safe one)."""
    depth, width = 256, 128
    matrix = _cancelling_columns(depth, width)
    routes = 8
    cotangent = np.zeros((routes, width), np.float32)
    cotangent[:, :3] = (2.0 ** 126, 2.0 ** 125, 1.0)
    if gathered:
        matrix = jax.tree.map(lambda leaf: leaf[None], matrix)
        tokens, gather = 3, jnp.asarray([0, 0, 2, 2, 2, 0, 1, 2], jnp.int32)
        sizes = jnp.asarray([routes], jnp.int32)
        expected = np.bincount(np.asarray(gather), minlength=tokens).astype(np.float32)
    else:
        tokens, gather, sizes = routes, None, None
        expected = np.ones((routes,), np.float32)
    rows = jnp.ones((tokens, depth), jnp.float32)
    inner = {} if kernel == "packed" else {"grouped_matmul": "xla"}
    with generation.assumed("sm90"), generation.chosen(nvfp4=kernel, **inner):
        _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, matrix, sizes, gather, out_dtype=jnp.float32,
                                                          interpret_on_cpu=True), rows)
        (grad,) = pull(jnp.asarray(cotangent))
    grad = np.asarray(grad)
    np.testing.assert_array_equal(grad[:, 0], expected)
    np.testing.assert_array_equal(grad[:, 1:], 0.0)


@pytest.mark.parametrize("columned", [False, True], ids=["scale a matrix", "scale a column"])
def test_an_nvfp4_bank_of_no_experts_multiplies_no_rows(columned):
    """A bank of no experts (`[0, depth, width]`), its tensor scales `[0]` or one a column `[0, width]`
    (`NvFp4.columned`): its product of no rows is `[0, width]` and pulls back an empty gradient, each row
    taking its group's scale where there is none (`factored`'s `groups`; train 1403's review: the factor
    reshaped to `[0, -1]` raised)."""
    scale = jnp.zeros((0, 128) if columned else (0,), jnp.float32)
    bank = formats.NvFp4(jnp.zeros((0, 128, 64), jnp.uint32), jnp.zeros((0, 128, 32), jnp.float8_e4m3fn),
                         scale)
    assert bank.checked().columned == columned
    rows = jnp.zeros((0, 512), jnp.float32)
    with generation.chosen(nvfp4="xla"):
        out, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, bank, jnp.zeros((0,), jnp.int32),
                                                            out_dtype=jnp.float32), rows)
        (grad,) = pull(jnp.zeros((0, 128), jnp.float32))
    assert out.shape == (0, 128) and grad.shape == (0, 512)


@pytest.mark.parametrize("grouped", [False, True], ids=["matrix", "bank"])
@pytest.mark.parametrize("kernel", ["packed", "xla"])
def test_an_nvfp4_column_scale_of_2_122_keeps_its_product_and_gradient_finite(kernel, grouped):
    """Columns whose tensor scales are 2^122 beside columns of 2^100 (`NvFp4.columned`, row 13743), every
    code 0.5 under block scales of 0.25, so each value is 2^119 or 2^97 (`quantization.packed_values`),
    through `kernel` on an H100 (interpreted; the decoded arm's inner product XLA's), as one matrix and as
    a bank of two experts whose large columns differ: the product of small integer rows and their gradient
    under a small integer cotangent are the float64 ones exactly, finite, where the decode's owed 2^8 folded
    into the scale (2^130) was infinite (train 1403's review)."""
    from dew.quantization import packed_values

    experts, depth, width = (2 if grouped else 1), 256, 64
    large = (np.arange(width)[None] + np.arange(experts)[:, None]) % 2 == 0
    scale = np.where(large, 2.0 ** 122, 2.0 ** 100).astype(np.float32)
    matrix = formats.NvFp4.of(jnp.full((experts, depth, width), 0.5, jnp.float32),
                              jnp.full((experts, depth // 16, width), 0.25, jnp.float8_e4m3fn),
                              jnp.asarray(scale))
    if not grouped:
        matrix = jax.tree.map(lambda leaf: leaf[0], matrix)
    values = np.asarray(packed_values(matrix), np.float64).reshape(experts, depth, width)
    assert set(np.unique(values)) == {2.0 ** 97, 2.0 ** 119}
    counts = (40, 24) if grouped else (64,)
    sizes = jnp.asarray(counts, jnp.int32) if grouped else None
    rng = np.random.default_rng(14032)
    rows = jnp.asarray(rng.integers(-1, 2, size=(64, depth)), jnp.float32)
    cotangent = jnp.asarray(rng.integers(-1, 2, size=(64, width)), jnp.float32)
    inner = {} if kernel == "packed" else {"grouped_matmul": "xla"}
    with generation.assumed("sm90"), generation.chosen(nvfp4=kernel, **inner):
        out, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, matrix, sizes, out_dtype=jnp.float32,
                                                            interpret_on_cpu=True), rows)
        (grad,) = pull(cotangent)
    product, gradient = np.zeros((64, width)), np.zeros((64, depth))
    for group, (first, last) in enumerate(itertools.pairwise(np.cumsum([0, *counts]))):
        product[first:last] = np.asarray(rows[first:last], np.float64) @ values[group]
        gradient[first:last] = np.asarray(cotangent[first:last], np.float64) @ values[group].T
    np.testing.assert_array_equal(np.asarray(out, np.float64), product)
    np.testing.assert_array_equal(np.asarray(grad), gradient.astype(np.float32))
    assert np.all(np.isfinite(product)) and np.all(np.isfinite(gradient.astype(np.float32)))


@pytest.mark.parametrize("kernel", ["packed", "xla"])
def test_an_nvfp4_rows_gradient_keeps_a_cancelling_pairs_rounding_residual(kernel):
    """Codes +1 and -1 at depth 0 of two columns whose tensor scales are 4 (1 - 2^-23) and 4, unit block
    scales, under cotangents 2^126 (1 + 2^-23) and 2^126 there, through `kernel` on an H100 (interpreted;
    the decoded arm's inner product XLA's): each scaled cotangent overflows, so the exact transpose sums
    them, and the depth's gradient is their exact difference, 2^128 ((1 - 2^-46) - 1) = -2^82, bit for bit
    (train 1426's review: their significands' product rounded in fp32, both read 2^129, summed 0)."""
    depth, width = 256, 128
    codes = np.zeros((depth, width), np.float32)
    codes[0, :2] = (1.0, -1.0)
    scale = np.ones((width,), np.float32)
    scale[:2] = (4 * (1 - 2.0 ** -23), 4.0)
    matrix = formats.NvFp4.of(jnp.asarray(codes), jnp.ones((depth // 16, width), jnp.float8_e4m3fn),
                              jnp.asarray(scale))
    cotangent = np.zeros((8, width), np.float32)
    cotangent[:, :2] = (2.0 ** 126 * (1 + 2.0 ** -23), 2.0 ** 126)
    inner = {} if kernel == "packed" else {"grouped_matmul": "xla"}
    with generation.assumed("sm90"), generation.chosen(nvfp4=kernel, **inner):
        _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, matrix, out_dtype=jnp.float32,
                                                          interpret_on_cpu=True), jnp.zeros((8, depth)))
        (grad,) = pull(jnp.asarray(cotangent))
    want = np.zeros((8, depth), np.float32)
    want[:, 0] = -(2.0 ** 82)
    np.testing.assert_array_equal(np.asarray(grad).view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("left_dtype", [jnp.float32, jnp.bfloat16])
def test_an_owed_product_is_its_exact_value_rounded_once_into_float32(left_dtype):
    """`float_bits.owed_product` of operands drawn across their whole range (a normal's binades and a
    subnormal's, both signs) times powers of two owed either way, jitted on this CPU, whose float32
    arithmetic flushes subnormals: each the float64 product, exact, rounded once into float32 by numpy, bit
    for bit, a subnormal or an overflow's infinity included (the one rule GEMM's factored products and a
    stored decode's values share, train 1437's review)."""
    from dew.nn.float_bits import owed_product

    rng = np.random.default_rng(14372)
    bits = 16 if left_dtype == jnp.bfloat16 else 32
    words = rng.integers(0, 2 ** bits, size=4096, dtype=np.uint64).astype(f"uint{bits}")
    left = jax.lax.bitcast_convert_type(jnp.asarray(words), left_dtype)
    right = rng.integers(0, 2 ** 32, size=4096, dtype=np.uint64).astype(np.uint32).view(np.float32)
    owed = rng.integers(-40, 41, size=4096).astype(np.int32)
    finite = np.isfinite(np.asarray(left, np.float64)) & np.isfinite(right.astype(np.float64))
    got = np.asarray(jax.jit(owed_product)(left, jnp.asarray(right), jnp.asarray(owed)))
    with np.errstate(over="ignore"):
        want = (np.asarray(left, np.float64) * right.astype(np.float64) * np.exp2(owed)).astype(np.float32)
    np.testing.assert_array_equal(got[finite].view(np.uint32), want[finite].view(np.uint32))


def test_a_scaled_products_subnormal_operand_keeps_its_products_every_bit():
    """`factored.scaled_exactly` of the least subnormal, 2^-149, times (1 + 2^-23) 2^126, on this CPU whose
    float32 arithmetic flushes subnormals: 2^-23 + 2^-46 exactly, its last bit kept (train 1437's review:
    the subnormal's short significand, multiplied unnormalized, lost it to the product's cut), and an owed
    exponent past every place holds at its edge, not wrapping: 1 times 2^28 lowered by 2^31 - 1 is zero."""
    from dew.nn.kernels.templates.gemm.factored import lowered_product, scaled_exactly

    least = np.uint32(1).view(np.float32)
    got = jax.jit(scaled_exactly)(jnp.float32(least), jnp.float32((1 + 2.0 ** -23) * 2.0 ** 126))
    assert np.asarray(got).view(np.uint32) == np.float32(2.0 ** -23 + 2.0 ** -46).view(np.uint32)
    lowered = jax.jit(lowered_product)(jnp.float32(1.0), jnp.float32(2.0 ** 28), jnp.int32(2 ** 31 - 1))
    assert np.asarray(lowered).view(np.uint32) == 0


def test_an_owed_products_reverse_mode_in_its_factor_is_the_product_owed():
    """`factored.owed_exactly(1, scale, 8)`'s gradient in its scale is the owed power of two, 256, its
    derivative in the factor a linear map of its own (train 1426's review: it ran through the value's bit
    reading, which reverse mode cannot transpose)."""
    from dew.nn.kernels.templates.gemm.factored import owed_exactly

    grad = jax.grad(lambda scale: owed_exactly(jnp.float32(1.0), scale, 8))(jnp.float32(0.5))
    assert float(grad) == 256.0


def test_an_owed_power_past_float32s_is_owed_exactly_natively_and_in_software():
    """An owed power of two past fp32's largest is owed exactly by `factored.owed_native` (the fold a
    platform keeping subnormals takes, CUDA's) and `owed_exactly` (this CPU's, in integer arithmetic)
    alike: 2^-100 times 1 owed 2^128 is 2^28, 2^-120 times 2^-20 owed 2^160 is 2^20, 1.5 2^-126 times
    2^-10 owed 2^130 is 1.5 2^-6, where the native fold's float32 power of 2^128 was infinite (train
    1450's review)."""
    from dew.nn.kernels.templates.gemm.factored import owed_exactly, owed_native

    cases = ((2.0 ** -100, 1.0, 128, 2.0 ** 28), (2.0 ** -120, 2.0 ** -20, 160, 2.0 ** 20),
             (1.5 * 2.0 ** -126, 2.0 ** -10, 130, 1.5 * 2.0 ** -6))
    for product, factor, raised, want in cases:
        for owed in (functools.partial(owed_native, raised=raised),
                     functools.partial(owed_exactly, raised=raised)):
            got = jax.jit(owed)(jnp.float32(product), jnp.float32(factor))
            assert np.asarray(got).view(np.uint32) == np.float32(want).view(np.uint32), (raised, got)


def an_owed_products_native_fold_rounds_once_below_the_normals():
    """On a GPU, whose float32 keeps subnormals (`factored.flushing`'s native platform),
    `factored.owed_native` and `owed_exactly` owe 2^127 on a subnormal product past a factor whose folded
    multiplier overflows, rounding once: 2^-149 times 2.5 owed 2^127 is 5 2^-23, the product raised by the
    power before the factor's one rounding, where its product with the factor first, rounded on the
    subnormal grid, gave 2^-21 (train 1450's review). Run by tests/test_gpu_gemm.py's cells: this CPU
    flushes the subnormal operand."""
    from dew.nn.kernels.templates.gemm.factored import owed_exactly, owed_native

    least = np.uint32(1).view(np.float32)
    for owed in (functools.partial(owed_native, raised=127), functools.partial(owed_exactly, raised=127)):
        got = jax.jit(owed)(jnp.float32(least), jnp.float32(2.5))
        assert np.asarray(got).view(np.uint32) == np.float32(5 * 2.0 ** -23).view(np.uint32), got


@pytest.mark.mesh(devices=2)
def test_an_owed_products_scale_gradient_over_a_column_split_product_runs_outside_its_mesh():
    """`factored.owed_exactly`'s sum over a product of ones an Explicit axis splits by its columns, times
    one replicated scale owed 2^8, differentiated in the scale and jitted outside the mesh's context: the
    scale's gradient is 256 times the product's 2 x 128 entries, its transpose placed as the scale's
    direction by its own sharding (train 1426's review: resharded by a bare spec, which needs the
    context)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    from dew.nn.kernels.templates.gemm.factored import owed_exactly

    mesh = jax.make_mesh((2,), ("tensor",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    columns = NamedSharding(mesh, PartitionSpec(None, "tensor"))
    product = jax.device_put(jnp.ones((2, 128), jnp.float32), columns)
    scale = jax.device_put(jnp.float32(0.5), NamedSharding(mesh, PartitionSpec()))
    grad = jax.jit(jax.grad(lambda scale, product: jnp.sum(owed_exactly(product, scale, 8))))(scale, product)
    assert float(grad) == 256.0 * 2 * 128


@pytest.mark.parametrize(("scale", "code", "block", "value"), [
    (2.0 ** -126, 0.5, 0.25, 2.0 ** -129),
    (2.0 ** -140, 4.0, 256.0, 2.0 ** -122),
], ids=["subnormal product", "subnormal scale"])
@pytest.mark.parametrize("kernel", ["packed", "xla"])
def test_an_nvfp4_products_owed_power_of_two_keeps_its_subnormals(kernel, scale, code, block, value):
    """An NVFP4 product whose tensor scale and the decode's owed 2^8 (`bits.NVFP4_OWED`) make a subnormal
    output (a scale of 2^-126 under one code of 0.5 at block scale 0.25 a column: 2^-129), or whose
    subnormal scale (2^-140) makes a normal one (codes of 4 at block scale 256 over 256 depths: 2^-122),
    through `kernel` on an H100 (interpreted; the decoded arm's inner product XLA's): each output is that
    value, bit for bit, on this CPU whose float32 arithmetic flushes subnormals, the owed power of two the
    product's one rounding's (`factored.owed_exactly`; train 1403's review: a runtime multiply by one
    flushed the first, the scale raised by 2^8 the second)."""
    depth, width = 256, 64
    codes = np.zeros((depth, width), np.float32)
    codes[: 1 if code == 0.5 else depth] = code
    matrix = formats.NvFp4.of(jnp.asarray(codes), jnp.full((depth // 16, width), block, jnp.float8_e4m3fn),
                              jnp.float32(scale))
    rows = jnp.ones((8, depth), jnp.float32)
    inner = {} if kernel == "packed" else {"grouped_matmul": "xla"}
    with generation.assumed("sm90"), generation.chosen(nvfp4=kernel, **inner):
        got = gemm.nvfp4_product(rows, matrix, out_dtype=jnp.float32, interpret_on_cpu=True)
    np.testing.assert_array_equal(np.asarray(got).view(np.uint32),
                                  np.full((8, width), value, np.float32).view(np.uint32))


@pytest.mark.parametrize(("case", "code", "block", "scale", "cotangent", "want"), [
    ("tiny routes", (4.0,) * 128, 256.0, 2.0 ** -142, (1.0,) * 128, 2.0 ** -125),
    ("a cancelling pair's residual", (0.5, -0.5), 2.0 ** -9, 1.0, (2.0 ** -116 + 2.0 ** -131, 2.0 ** -116),
     2.0 ** -141),
    ("a product rounding to zero", (4.0,), 448.0, 2.0 ** -50, (2.0 ** -108,), 2.0 ** -147),
], ids=lambda value: value if isinstance(value, str) else None)
@pytest.mark.parametrize("kernel", ["packed", "xla"])
def test_an_nvfp4_rows_gradient_keeps_tiny_scaled_products(kernel, case, code, block, scale, cotangent, want):
    """An NVFP4 matrix whose depth 0 holds `code` at block scale `block` in its first columns, every column's
    tensor scale `scale` (`NvFp4.columned`), its rows' gradient at depth 0 under output cotangents
    `cotangent` there, through `kernel` on an H100 (interpreted; the decoded arm's inner product XLA's),
    jitted on this CPU, which flushes subnormal float32 arithmetic: the exact transpose's `want`, bit for
    bit, where the plain one lost it below its range (train 1409's review): 128 routes' 2^-134, each below
    its bf16 parts, sum to the normal 2^-125; a cancelling pair's residual 2^-123 times 2^-18 is the
    subnormal 2^-141; and 2^-108 times 2^-50 owed 2^8, 2^-150, rounds to zero alone yet times 7 is
    2^-147."""
    del case
    depth, width = 256, 128
    codes = np.zeros((depth, width), np.float32)
    codes[0, :len(code)] = code
    matrix = formats.NvFp4.of(jnp.asarray(codes), jnp.full((depth // 16, width), block, jnp.float8_e4m3fn),
                              jnp.full((width,), scale, jnp.float32))
    dy = np.zeros((8, width), np.float32)
    dy[:, :len(cotangent)] = cotangent
    inner = {} if kernel == "packed" else {"grouped_matmul": "xla"}
    with generation.assumed("sm90"), generation.chosen(nvfp4=kernel, **inner):
        pull = jax.jit(lambda dy: jax.vjp(lambda rows: gemm.nvfp4_product(
            rows, matrix, out_dtype=jnp.float32, interpret_on_cpu=True), jnp.zeros((8, depth)))[1](dy)[0])
        grad = pull(jnp.asarray(dy))
    expected = np.zeros((8, depth), np.float32)
    expected[:, 0] = want
    np.testing.assert_array_equal(np.asarray(grad).view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("kernel", ["packed", "xla"])
def test_an_nvfp4_pullbacks_derivative_keeps_a_cancelling_directions_remainder(kernel):
    """Codes +1, -1, +1 at one depth of three columns whose tensor scales are 4, 8 and 1, unit block scales,
    through `kernel` on an H100 (interpreted): the rows' pullback's derivative at a zero cotangent along
    2^126, 2^125 and 1 is the range-safe transpose of that direction, 2^128 - 2^128 + 1 = 1 at the depth
    (train 1409's review: the derivative's terms pulled back plainly, two infinities, read NaN), and reverse
    mode over the pullback reads the product's own transpose, each column's value (4, -8, 1)."""
    depth, width = 256, 128
    matrix = _cancelling_columns(depth, width)
    direction = jnp.zeros((8, width), jnp.float32).at[:, :3].set(jnp.asarray([2.0 ** 126, 2.0 ** 125, 1.0]))
    inner = {} if kernel == "packed" else {"grouped_matmul": "xla"}
    with generation.assumed("sm90"), generation.chosen(nvfp4=kernel, **inner):
        def pullback(cotangent, matrix):
            _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, matrix, out_dtype=jnp.float32,
                                                              interpret_on_cpu=True), jnp.zeros((8, depth)))
            return pull(cotangent)[0]

        # Jitted, the matrix an argument: the derivative's map reads it as its residual.
        _, moved = jax.jit(lambda matrix, direction: jax.jvp(
            lambda cotangent: pullback(cotangent, matrix), (jnp.zeros_like(direction),), (direction,)))(
            matrix, direction)
        # Reverse mode over the pullback: its transpose is the product, each column its value at depth 0.
        across = jax.grad(lambda cotangent: jnp.sum(pullback(cotangent, matrix)))(jnp.zeros_like(direction))
    want = np.zeros((8, depth), np.float32)
    want[:, 0] = 1.0
    np.testing.assert_array_equal(np.asarray(moved), want)
    values = np.zeros((8, width), np.float32)
    values[:, :3] = (4.0, -8.0, 1.0)
    np.testing.assert_array_equal(np.asarray(across), values)


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

    def pulled(rows, matrix, layer=None):
        out, pull = jax.vjp(lambda x: product(x, matrix, layer), rows)
        return out, pull(cotangent)[0]

    with generation.assumed("sm90"), generation.chosen(nvfp4=kernel), generation.routes() as routes:
        # One program for the stack at a traced layer and one for a bank, each run at every layer: three eager
        # traces of the interpreted kernel a layer took 160 s, past a group's 120 s target (row 12471's file).
        stacked = jax.jit(lambda layer: pulled(rows, stack, layer))
        alone = jax.jit(lambda bank: pulled(rows, bank))
        for layer, bank in enumerate(banks):
            got, want = stacked(jnp.asarray(layer)), alone(bank)
            for got_leaf, want_leaf in zip(got, want, strict=True):
                assert np.asarray(got_leaf).tobytes() == np.asarray(want_leaf).tobytes(), layer
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
                        scatter=slots, epilogue=Hook(plus, (Side(bias, ("groups", "columns")),)),
                        interpret_on_cpu=True)
        with pytest.raises(ValueError):
            gemm.gemm(tokens, bank, jnp.asarray(sizes, jnp.int32), out_dtype=jnp.float32,
                      prologue=Hook(lambda x: x))
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _EarlierBank:
    """An NVFP4 bank as an earlier Dew registered it (`formats.NvFp4` before row 13700): its digest `form` a
    leaf after its words, scales and scale, the layout its checkpoints hold."""

    words: jax.Array
    scales: jax.Array
    scale: jax.Array
    form: jax.Array


def test_a_bank_carries_its_forms_version_and_a_load_refuses_another(tmp_path):
    """`formats.NvFp4` states the version of the form its words are packed by (`formats.FORM`, the encoder's
    digest, `bits.BitForm.version`; algebra.md 7.4, row 6101), a stacked bank's one, and a checkpoint holds
    it beside its words (`formats.stored`); a restore of words another encoder packed is refused
    (`formats.restored`): written and read back with its version, a bank restores into a template of its
    shapes alone; the same bank stored under another version is refused by name, read into this Dew's
    template."""
    from dew.checkpoints import Checkpoints

    bank, _ = nvfp4_bank(3, 2, 256, 128)
    assert bank.form == formats.FORM
    stacked = jax.tree.map(lambda *layers: jnp.stack(layers), bank, bank)
    assert stacked.layer(jnp.int32(1)).form == formats.FORM
    reordered = formats.form_version(bits.NVFP4_BFLOAT16_FORM.version,
                                     formats.staged_rows(formats.NVFP4_STAGE)[::-1])
    assert reordered != formats.FORM
    with pytest.raises(ValueError):
        dataclasses.replace(bank, form=reordered).checked()
    other = dataclasses.replace(bank, form=bytes(8))
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
                Checkpoints(str(tmp_path / name)).restore(template({"bank": bank}))
        else:
            restored = Checkpoints(str(tmp_path / name)).restore(template({"bank": bank}))[0]["bank"]
            np.testing.assert_array_equal(np.asarray(restored.words), np.asarray(written.words))
    # A checkpoint an earlier Dew wrote, its digest a leaf of its bank (`_EarlierBank`, the layout it
    # registered), a stacked bank's a row a layer, layer-sharded across two devices where there are two:
    # restored into this Dew's stacked bank where every row is this Dew's form, refused where its second
    # layer's row is another encoder's (row 8958's collective check of the rows each process holds).
    digest = np.frombuffer(formats.FORM, np.uint8)
    devices = jax.devices()[:2] if len(jax.devices()) >= 2 else jax.devices()[:1]
    by_layer = jax.sharding.NamedSharding(jax.sharding.Mesh(np.array(devices), ("layers",)),
                                          jax.sharding.PartitionSpec("layers"))
    for name, second, refused in (("earlier", digest, False), ("earlier-mixed", np.zeros(8, np.uint8), True)):
        earlier = _EarlierBank(*(jax.device_put(np.asarray(field), by_layer)
                                 for field in (stacked.words, stacked.scales, stacked.scale)),
                               jax.device_put(np.stack([digest, second]), by_layer))
        checkpoints = Checkpoints(str(tmp_path / name))
        checkpoints.save(1, {"bank": earlier})
        checkpoints.wait()
        wanted = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=by_layer),
                              {"bank": stacked})
        if refused:
            with pytest.raises(ValueError):
                Checkpoints(str(tmp_path / name)).restore(wanted)
        else:
            restored = Checkpoints(str(tmp_path / name)).restore(wanted)[0]["bank"]
            assert isinstance(restored, formats.NvFp4) and restored.form == formats.FORM
            np.testing.assert_array_equal(np.asarray(restored.scales), np.asarray(stacked.scales))


@pytest.mark.parametrize("group_sizes", [None, (8, 8)], ids=["matrix", "expert bank"])
def test_an_nvfp4_product_over_operands_a_mesh_splits_decodes_its_matrix(group_sizes):
    """An NVFP4 product at the H100's 16-row decode cell (`grouped_cells.NVFP4_CELLS`), a matrix's or an
    expert bank's, takes its packed launch over whole operands and decodes its matrix where a mesh XLA places
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


def the_quantize_epilogue_encodes_as_bits_encoded(minifloat: bits.Minifloat, interpret_on_cpu: bool = False
                                                   ) -> None:
    """`quantized.gemm_quantized` through the template encodes the GEMM accumulator via `bits.encode_form`
    bit-exact with `bits.encoded`, dividing by `tensor_scale` on encode and returning `(codes, scale)`
    consistent with `NvFp4Block` (rows 8547, 12037, 12160): interpreted, the form's evaluation
    (`BitForm.traced`); compiled on a GPU, its PTX inline in the epilogue (`inline_forms.registers`, row
    14105)."""
    rng = np.random.default_rng(17)
    lhs = jnp.asarray(rng.integers(-3, 4, size=(128, 128)), jnp.bfloat16)
    rhs = jnp.asarray(rng.integers(-3, 4, size=(128, 128)), jnp.bfloat16)
    acc = jnp.dot(lhs, rhs, preferred_element_type=jnp.float32)
    words = jax.lax.bitcast_convert_type(acc, jnp.uint32)
    expected_u8 = bits.encoded(words, minifloat).astype(jnp.uint8)
    with generation.chosen(grouped_matmul="mosaic"), generation.routes() as routes:
        got, scale = quantized.gemm_quantized(lhs, rhs, minifloat=minifloat,
                                              interpret_on_cpu=interpret_on_cpu)
        got_s, scale_s = quantized.gemm_quantized(lhs, rhs, minifloat=minifloat, tensor_scale=2.0,
                                                  interpret_on_cpu=interpret_on_cpu)
    assert ("grouped_matmul", "mosaic") in {(route.choice, route.kernel) for route in routes}
    out_dtype = quantized.minifloat_dtype(minifloat)
    assert isinstance(got, jax.Array) and got.dtype == out_dtype and float(scale) == 1.0
    got_u8 = jax.lax.bitcast_convert_type(got, jnp.uint8)
    np.testing.assert_array_equal(np.asarray(got_u8), np.asarray(expected_u8))
    assert isinstance(got_s, jax.Array) and got_s.dtype == out_dtype and float(scale_s) == 2.0
    scaled_u8 = bits.encoded(jax.lax.bitcast_convert_type(acc / jnp.float32(2.0), jnp.uint32),
                             minifloat).astype(jnp.uint8)
    np.testing.assert_array_equal(np.asarray(jax.lax.bitcast_convert_type(got_s, jnp.uint8)),
                                  np.asarray(scaled_u8))


@INTERPRETS
@pytest.mark.parametrize("minifloat", [bits.E4M3, bits.E5M2], ids=["e4m3", "e5m2"])
def test_gemm_quantize_epilogue_matches_bits_encoded(minifloat):
    """The quantize epilogue's encode, interpreted on an H100's template, is `bits.encoded`
    (`the_quantize_epilogue_encodes_as_bits_encoded`; its compiled PTX is tests/test_gpu_gemm.py's)."""
    with generation.assumed("sm90"):
        the_quantize_epilogue_encodes_as_bits_encoded(minifloat, interpret_on_cpu=True)


@pytest.mark.parametrize("gpu", [pytest.param("sm80", marks=lowers("sm80")),
                                 pytest.param("sm90", marks=lowers("sm90"))])
def test_gemm_quantize_epilogue_lowers_inline_ptx(gpu):
    """`quantized.gemm_quantized` lowers FP8 and MXFP8 epilogues with `bits.encode_form`'s inline PTX on sm80
    and sm90 (rows 8547, 12037)."""
    x = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), gpu)
    with (generation.assumed(gpu), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(grouped_matmul="mosaic")):
        jax.jit(lambda a, b: quantized.gemm_quantized(a, b, minifloat=bits.E4M3)).trace(x, w).lower(
            lowering_platforms=("cuda",))
        jax.jit(lambda a, b: quantized.gemm_quantized(a, b, minifloat=formats.MxFp8)).trace(x, w).lower(
            lowering_platforms=("cuda",))


def _columned_case(rows: int, *, grouped: bool):
    """Rows of a call and an NVFP4 matrix (or a bank of two experts with its group sizes) whose tensor scale
    is one a column (`NvFp4.columned`)."""
    rng = np.random.default_rng(13743)
    experts = (2,) if grouped else ()
    matrix = formats.NvFp4.of(jnp.asarray(rng.integers(-2, 3, (*experts, 512, 128)), jnp.float32),
                              jnp.ones((*experts, 32, 128), jnp.float8_e4m3fn),
                              jnp.asarray(np.exp2(rng.integers(-4, 4, (*experts, 128))), jnp.float32))
    sizes = jnp.asarray([rows // 2, rows - rows // 2], jnp.int32) if grouped else None
    return jax.ShapeDtypeStruct((rows, 512), jnp.bfloat16), matrix, sizes


def a_packed_nvfp4_call_placed_on_its_hosts_cpu_keeps_a_subnormal_scale():
    """On a GPU process, a packed NVFP4 prefill (named packed, its tensor scale one a column) whose operands
    are committed to the host's CPU, interpretation asked: it runs on the CPU as the plain launch
    interpreted, `factored`'s exact finish after it, never the GPU's hooked launch interpreted there, whose
    epilogue's product by 2^8 flushed a column scale of 2^-127 to zero (the review of row 13743's restack):
    two experts of 256-deep ones over 256 rows of ones, each value 2^-119. Run by tests/test_gpu_gemm.py's
    cells: a CPU process plans no hooked launch."""
    cpu = jax.devices("cpu")[0]
    bank = formats.NvFp4.of(jnp.ones((2, 256, 128), jnp.float32), jnp.ones((2, 16, 128), jnp.float8_e4m3fn),
                            jnp.full((2, 128), 2.0 ** -127, jnp.float32))
    rows, sizes = jnp.ones((256, 256), jnp.bfloat16), jnp.asarray([128, 128], jnp.int32)
    rows, bank, sizes = jax.device_put((rows, bank, sizes), cpu)
    # The interpreter's GPU stated, as every interpreted call states it; the process's default stays its GPU.
    with generation.assumed("sm90"), generation.chosen(nvfp4="packed"):
        out = gemm.nvfp4_product(rows, bank, sizes, out_dtype=jnp.float32, interpret_on_cpu=True)
    np.testing.assert_array_equal(np.asarray(out), np.full((256, 128), 2.0 ** -119, np.float32))


def a_packed_nvfp4_prefills_finish_is_its_products_finished_apart():
    """On a GPU, a packed NVFP4 prefill (named packed, 256 rows of two experts' groups) whose tensor scale is
    one a column, the columns' scales 1, 2^-127 (subnormal, the 2^8 it owes taking it to a normal 2^-119)
    and 2^121 (finite, its product by 2^8 past float32's largest) by turns and each expert's its own: its
    launch's finish (`gemm._packed_finish`, the scale and the owed 2^8 in the epilogue) is bitwise its
    products finished apart (XLA's decoded product, `factored`'s one rounding after it) and the float64
    truth, over small integer codes and rows (train 1474's review: the finish was lowered and run on the
    host, never natively against the finish apart). Run by tests/test_gpu_gemm.py's cells."""
    rng = np.random.default_rng(13743)
    # Codes of E2M1's half steps and rows of units: every sum small, so 2^121 times it stays finite.
    codes = rng.integers(-2, 3, (2, 256, 128)).astype(np.float32) / 2
    scales = np.tile(np.asarray([1.0, 2.0 ** -127, 2.0 ** 121], np.float32), 43)[:128]
    scale = np.stack([scales, np.roll(scales, 1)])
    bank = formats.NvFp4.of(jnp.asarray(codes), jnp.ones((2, 16, 128), jnp.float8_e4m3fn), jnp.asarray(scale))
    rows = rng.integers(-1, 2, (256, 256)).astype(np.float32)
    sizes = jnp.asarray([128, 128], jnp.int32)

    def product(route):
        with generation.chosen(nvfp4=route), tuning.seen() as launched:
            out = np.asarray(jax.jit(lambda x: gemm.nvfp4_product(x, bank, sizes, out_dtype=jnp.float32))(
                jnp.asarray(rows, jnp.bfloat16)))
        return out, _packed_epilogues(launched)

    truth = np.concatenate([rows[group * 128:(group + 1) * 128].astype(np.float64) @ codes[group]
                            * scale[group].astype(np.float64) for group in range(2)]).astype(np.float32)
    (finished, hooked), (apart, _) = product("packed"), product("xla")
    # The packed call's launch took the finish in its epilogue (its review: else the finish was XLA's).
    assert hooked and all(epilogue is not None for epilogue in hooked), hooked
    assert np.isfinite(truth).all()
    np.testing.assert_array_equal(apart.view(np.uint32), truth.view(np.uint32))
    np.testing.assert_array_equal(finished.view(np.uint32), apart.view(np.uint32))


def _packed_epilogues(launched) -> list:
    """The epilogue of each packed NVFP4 launch a trace took (`tuning.seen`)."""
    return [observed.keyed.arguments["hooks"].epilogue for observed in launched
            if isinstance(observed.op, grouped_quantized.MosaicPackedGmm)]


@pytest.mark.parametrize("grouped", [False, True], ids=["matrix", "bank"])
def test_a_packed_nvfp4_call_finishes_its_scale_in_its_launch_where_it_varies_by_group(grouped):
    """An NVFP4 product named packed on an H100 whose tensor scale is one a column, as a bank of two experts
    and as a matrix: a bank's prefill launch takes the scale and the decode's owed 2^8 in its epilogue
    (`gemm._packed_finish`, row 13743), no row's expert's scales gathered after it; a matrix's takes none,
    its one scale XLA's in the output's conversion, which the epilogue's finish only slowed (M-NG-68t4,
    GH200); a decode's 16 rows (swapped or not) take none, as its receipts timed it; interpreted on this CPU
    (whose float32 flushes the subnormals `factored`'s own rounding keeps) no launch takes it either."""

    def epilogues(rows, *, interpret):
        spec, matrix, sizes = _columned_case(rows, grouped=grouped)
        with generation.assumed("sm90"), generation.chosen(nvfp4="packed"), tuning.seen() as launched:
            jax.eval_shape(lambda x: gemm.nvfp4_product(x, matrix, sizes, out_dtype=jnp.float32,
                                                        interpret_on_cpu=interpret), spec)
        found = _packed_epilogues(launched)
        assert found, rows
        return found

    assert all((epilogue is not None) == grouped for epilogue in epilogues(256, interpret=False))
    assert all(epilogue is None for epilogue in epilogues(256, interpret=True))
    # A decode's rows (the receipts' decode cells, `NVFP4_DECODE_ROWS`) keep the factor after the launch.
    assert all(epilogue is None for epilogue in epilogues(16, interpret=False))


@pytest.mark.parametrize("interpret", [False, True], ids=["native", "interpretation asked on a GPU"])
def test_a_packed_nvfp4_pin_its_finish_cannot_take_keeps_the_plain_launch(interpret):
    """A prefill's packed NVFP4 product on an H100's abstract mesh (its launch native, whatever it asks of a
    CPU), its finished launch's tile refused by a compile (`report.refuse_launch`, as a spill refuses it)
    while another finished tile still derives, and that tile pinned, which the plain launch takes: the call
    keeps the plain launch at the pin, the factor XLA's after it, where the finish's refused pin would have
    raised (`gemm._packed_finish`: the optional finish bound as its native launch is, row 13743's review)."""
    from dew.nn.kernels.templates import report

    spec, matrix, sizes = _columned_case(256, grouped=True)
    launch = grouped_quantized.MosaicPackedGmm()
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")

    def packed(pins):
        with generation.chosen(nvfp4="packed"), tuning.pinned(pins), tuning.seen() as launched:
            jax.jit(lambda x: gemm.nvfp4_product(x, matrix, sizes, out_dtype=jnp.float32,
                                                 interpret_on_cpu=interpret)).trace(spec)
        return [(observed.config, observed.arguments) for observed in launched.held()
                if isinstance(observed.arguments.op, grouped_quantized.MosaicPackedGmm)]

    with report.refusals(), generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        target = generation.gpu_target()
        [(config, hooked)] = [(config, ba) for config, ba in packed({})
                              if ba.arguments["hooks"].epilogue is not None]
        assert not hooked.arguments["interpret"]
        plain = launch.bind(**{name: value for name, value in hooked.arguments.items()
                               if name not in ("hooks", "epilogue_operands")})
        assert launch.pinnable(plain, launch.completed(plain, config))
        report.refuse_launch(launch, hooked, launch.completed(hooked, config), {launch.KERNEL})
        assert launch.derived(hooked, target) not in (None, config)
        after = packed({grouped_quantized.MosaicPackedGmm: config})
    assert after and all(ba.arguments["hooks"].epilogue is None for _, ba in after), after
    assert all(taken == launch.completed(plain, config) for taken, _ in after), after


@pytest.mark.parametrize("gpu", [pytest.param("sm80", marks=lowers("sm80")),
                                 pytest.param("sm90", marks=lowers("sm90"))])
def test_a_packed_nvfp4_prefill_finishing_its_scale_lowers_for_cuda(gpu):
    """A prefill-sized NVFP4 product named packed whose tensor scale is one a column of each of a bank's
    two experts lowers for CUDA on an A100 and an H100 with its launch's epilogue taking the scale, a side
    each tile reads where it lands (row 13743)."""
    spec, matrix, sizes = _columned_case(256, grouped=True)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), gpu)
    with (generation.assumed(gpu), jax.sharding.use_abstract_mesh(mesh), generation.chosen(nvfp4="packed"),
          tuning.seen() as launched):
        traced = jax.jit(lambda x: gemm.nvfp4_product(x, matrix, sizes, out_dtype=jnp.float32)).trace(spec)
        finished = _packed_epilogues(launched)
        assert finished and all(epilogue is not None for epilogue in finished), finished
        traced.lower(lowering_platforms=("cuda",))


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

def test_e8m0_bit_form_is_exact():
    """`E8M0_SCALE_FORM` is exact on every byte under high-byte noise, and `MxFp8.decoded` handles subnormal
    `0x00` and `0xFF` NaN across fp32 and fp64 (row 9040)."""
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

@pytest.mark.parametrize("gpu", [pytest.param("sm80", marks=lowers("sm80")),
                                 pytest.param("sm90", marks=lowers("sm90"))])
def test_gemm_quantized_nvfp4_block_matches_bits_nvfp4_coded_and_lowers(gpu: str):
    """`quantized.gemm_quantized` emits `formats.NvFp4Block` and `formats.MxFp8` bitwise matching
    `formats.NvFp4Block.quantized` and `formats.MxFp8.quantized`, and lowers through Mosaic GPU on sm80/sm90
    (rows 8547, 12037)."""
    rng = np.random.default_rng(8547)
    x = jnp.asarray(rng.standard_normal((64, 64)) / np.sqrt(64), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((64, 64)) / np.sqrt(64), jnp.bfloat16)
    acc = jax.lax.dot(x, w, preferred_element_type=jnp.float32)
    ref_nv = formats.NvFp4Block.quantized(acc)
    got_nv = jax.jit(lambda a, b, s: quantized.gemm_quantized(
        a, b, minifloat=formats.NvFp4Block, tensor_scale=s))(x, w, ref_nv.scale)
    assert isinstance(got_nv, formats.NvFp4Block)
    np.testing.assert_array_equal(np.asarray(got_nv.codes), np.asarray(ref_nv.codes))
    np.testing.assert_array_equal(np.asarray(got_nv.scales), np.asarray(ref_nv.scales))
    np.testing.assert_array_equal(np.asarray(got_nv.scale), np.asarray(ref_nv.scale))
    np.testing.assert_array_equal(np.asarray(got_nv.decoded()), np.asarray(ref_nv.decoded()))

    ref_mx = formats.MxFp8.quantized(acc)
    got_mx = jax.jit(lambda a, b: quantized.gemm_quantized(a, b, minifloat=formats.MxFp8))(x, w)
    assert isinstance(got_mx, formats.MxFp8)
    np.testing.assert_array_equal(np.asarray(got_mx.codes), np.asarray(ref_mx.codes))
    np.testing.assert_array_equal(np.asarray(got_mx.scales), np.asarray(ref_mx.scales))
    np.testing.assert_array_equal(np.asarray(got_mx.decoded()), np.asarray(ref_mx.decoded()))
    with pytest.raises(ValueError):
        quantized.gemm_quantized(x, jnp.ones((64, 33), jnp.bfloat16), minifloat=formats.MxFp8)
    with pytest.raises(ValueError):
        quantized.gemm_quantized(x, jnp.ones((64, 24), jnp.bfloat16), minifloat=formats.NvFp4Block)
    x_s = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    w_s = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    s_s = jax.ShapeDtypeStruct((), jnp.float32)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), gpu)
    with (generation.assumed(gpu), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(grouped_matmul="mosaic")):
        jax.jit(lambda a, b, s: quantized.gemm_quantized(a, b, minifloat=bits.E2M1, tensor_scale=s)).trace(
            x_s, w_s, s_s).lower(lowering_platforms=("cuda",))


def test_a_block_whose_ratio_is_subnormal_takes_the_least_scale_in_both_routes():
    """A block of values at fp32's least normal under a tensor scale of 1, its ratio over 6 below the
    normals, takes e4m3's least subnormal scale (0x01), the least at or above its ratio, its codes zero, in
    `formats.NvFp4Block.quantized` and the GEMM epilogue (`quantized.gemm_quantized`) alike on this CPU, whose
    flushing comparison read the ratio as zero and gave the block a scale of zero (row 14083's review:
    `bits.scales_rounded_up` compares words)."""
    x = jnp.eye(64, dtype=jnp.bfloat16)
    w = np.ones((64, 64), np.float32)
    w[0, :16] = 2.0 ** -126
    w = jnp.asarray(w, jnp.bfloat16)
    one = jnp.float32(1)
    ref = formats.NvFp4Block.quantized(jax.lax.dot(x, w, preferred_element_type=jnp.float32), one)
    got = jax.jit(lambda a, b, s: quantized.gemm_quantized(a, b, minifloat=formats.NvFp4Block,
                                                           tensor_scale=s))(x, w, one)
    for block in (ref, got):
        assert np.asarray(block.scales).view(np.uint8)[0, 0] == 0x01
        assert not np.asarray(block.codes).view(np.uint8)[0, :16].any()
    for found, expected in ((got.scales, ref.scales), (got.codes, ref.codes)):
        np.testing.assert_array_equal(np.asarray(found).view(np.uint8), np.asarray(expected).view(np.uint8))


def test_every_route_codes_a_blocks_scale_alike_at_its_edges():
    """Each route quantizing an NVFP4 operand codes the same block scales on this CPU, whose float32
    arithmetic flushes subnormals (row 14083's review): a block whose largest accumulator is subnormal
    (2^-140) at a tensor scale of 1, through the GEMM epilogue outside a kernel and `NvFp4Block.quantized`,
    both its least subnormal scale 0x01, and at a negative, negative-zero or NaN scale every block's scale
    e4m3's NaN, 0x7F (`bits.scales_rounded_up`); at float32 width and a float64
    truth's, a block of fp32's least normal at a scale of 3 (0x01) and one of 2^-124 at a subnormal scale
    of 2^-127 (8 over 6, 0x3B: 1.375), a zero scale's NaN lanes coded alike at both widths and in the
    epilogue, and the tensor scale's derivative in the block's largest, 1/2688."""
    acc = np.zeros((4, 32), np.float32)
    acc[0, 0] = 2.0 ** -140
    acc[1:, :] = 1.0

    def epilogue(acc, scale):
        codes, scales = quantized._nvfp4_quantize_epilogue(acc, jnp.broadcast_to(scale, (acc.shape[0],)))
        return formats.NvFp4Block.encoded(codes, scales[..., ::16], scale)

    def coded(acc, scale):
        return formats.NvFp4Block.quantized(acc, scale)

    for scale in (np.float32(1), np.float32(-1), np.float32(-0.0), np.float32(np.nan)):
        found = [jax.jit(route)(acc, jnp.float32(scale)) for route in (epilogue, coded)]
        for operand in found:
            scales, codes = (np.asarray(each).view(np.uint8) for each in (operand.scales, operand.codes))
            if scale > 0:
                assert scales[0, 0] == 0x01, scales
            else:
                # A negative, negative-zero or NaN scale codes every block's scale NaN, its codes zero.
                assert (scales == bits.E4M3.nan).all() and not codes.any(), (scale, scales, codes)
        for each in ("scales", "codes"):
            np.testing.assert_array_equal(*(np.asarray(getattr(operand, each)).view(np.uint8)
                                            for operand in found))
    with jax.enable_x64(new_val=True):
        # A zero scale: a block of a value beside zeros at 448, its zeros' quotients NaN, one code at either
        # width and in the epilogue, whatever sign the platform's 0/0 takes.
        found = [jax.jit(route)(acc.astype(dtype), jnp.float32(0)) for route, dtype in
                 ((epilogue, np.float32), (coded, np.float32), (coded, np.float64))]
        for each in ("scales", "codes"):
            first, *others = (np.asarray(getattr(operand, each)).view(np.uint8) for operand in found)
            for other in others:
                np.testing.assert_array_equal(other, first)
        for value, scale, code in ((2.0 ** -126, 3.0, 0x01), (2.0 ** -124, 2.0 ** -127, 0x3B)):
            for dtype in (jnp.float32, jnp.float64):
                block = jnp.full((4, 32), value, dtype)
                assert block.dtype == dtype
                operand = jax.jit(coded)(block, jnp.float32(scale))
                assert (np.asarray(operand.scales).view(np.uint8) == code).all(), (value, dtype)
        largest = np.zeros((16, 2), np.float64)
        largest[3, 1] = 2688.0
        for dtype in (jnp.float32, jnp.float64):
            moved = np.zeros_like(largest)
            moved[3, 1] = 1.0
            _, slope = jax.jvp(lambda x: bits.nvfp4_coded(x)[2], (jnp.asarray(largest, dtype),),
                               (jnp.asarray(moved, dtype),))
            assert np.asarray(slope, np.float32) == np.float32(1 / 2688), (dtype, slope)


def test_ternary_weights_checkpoint_restore_verifies_form_and_scale(tmp_path):
    """Checkpoint restore (`formats.restored`) restores a matching `TernaryWeights` bank, its words and scale,
    beside a concrete bank the checkpoint holds none of, kept as supplied; and refuses one stored under
    another form version, read into this Dew's template (the digest it read refused, not its template's), one
    of an invalid scale shape (row 12161), and a supplied concrete bank of another form or an invalid scale
    shape, which the restore checks as it checks a read one (train 1390's review)."""
    from dew.checkpoints import Checkpoints

    where = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def template(of):
        return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=where), of)

    good = formats.TernaryWeights.of(np.ones((256, 64), np.float32), scale=2.0)
    ok_ckpt = Checkpoints(str(tmp_path / "ok"))
    ok_ckpt.save(1, {"w": good})
    ok_ckpt.wait()
    # Beside it, a concrete bank the checkpoint holds none of: kept as supplied, as any concrete leaf is.
    supplied = formats.TernaryWeights.of(np.full((256, 64), -1.0, np.float32), scale=0.5)
    held = Checkpoints(str(tmp_path / "ok")).restore({**template({"w": good}), "supplied": supplied})[0]
    restored = held["w"]
    np.testing.assert_array_equal(np.asarray(restored.words), np.asarray(good.words))
    np.testing.assert_array_equal(np.asarray(restored.scale), np.asarray(good.scale))
    assert isinstance(held["supplied"], formats.TernaryWeights), held["supplied"]
    np.testing.assert_array_equal(np.asarray(held["supplied"].words), np.asarray(supplied.words))
    bad_form, bad_scale = (dataclasses.replace(good, form=bytes(8)),
                           dataclasses.replace(good, scale=jnp.ones((3,), jnp.float32)))
    for foreign in (bad_form, bad_scale):
        with pytest.raises(ValueError):
            Checkpoints(str(tmp_path / "ok")).restore({**template({"w": good}), "supplied": foreign})
    for tag, bad, into in (("form", bad_form, good), ("scale", bad_scale, bad_scale)):
        bad_ckpt = Checkpoints(str(tmp_path / tag))
        bad_ckpt.save(1, {"w": bad})
        bad_ckpt.wait()
        with pytest.raises(ValueError):
            Checkpoints(str(tmp_path / tag)).restore(template({"w": into}))


@INTERPRETS
def test_an_nvfp4_products_rescue_leaves_a_nan_block_scale_to_its_own_coordinates():
    """Codes +1, -1 and +1 at input coordinate 0 beside a NaN block scale at coordinates 16 to 31 of
    another column, a factor of 4, under cotangents 2**126, 2**126 and 1: coordinate 0's gradient is 4 and
    the NaN block's coordinates are IEEE's NaN, every other zero (rows 1464, 12403)."""
    codes = jnp.zeros((512, 128), jnp.float32).at[0, :3].set(jnp.asarray([1.0, -1.0, 1.0]))
    scales = jnp.ones((32, 128), jnp.float8_e4m3fn).at[1, 3].set(jnp.nan)
    matrix = formats.NvFp4.of(codes, scales, jnp.float32(4.0))
    _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, matrix, out_dtype=jnp.float32),
                      jnp.zeros((4, 512), jnp.float32))
    (d_x,) = pull(jnp.zeros((4, 128), jnp.float32).at[0, :3].set(
        jnp.asarray([2.0 ** 126, 2.0 ** 126, 1.0], jnp.float32)))
    got = np.array(d_x)
    assert np.isnan(got[:, 16:32]).all(), got[:, 16:32]
    want = np.zeros((4, 512), np.float32)
    want[0, 0] = 4.0
    got[:, 16:32] = 0.0
    np.testing.assert_array_equal(got, want)


def test_the_rescues_routes_add_into_their_tokens_alike_at_any_fan_in():
    """Carried limbs of seven routes into three tokens (`factored._routed`), one route a carry or all at
    once: the same carried digits, the tokens' exact sums (rows 1464, 12403)."""
    from dew.nn.kernels.templates.gemm import factored

    rng = np.random.default_rng(12403)
    digits = rng.integers(0, 2 ** factored.FACTORED_SLICE, size=(5, 7, 4)).astype(np.int32)
    digits[0] = rng.integers(-3, 4, size=(7, 4))
    gather = jnp.asarray([0, 2, 0, 1, 2, 2, 0], jnp.int32)
    tokens = jnp.zeros((5, 3, 4), jnp.int32)
    one = np.asarray(factored._routed(tokens, gather, jnp.asarray(digits), fan_in=1))
    whole = np.asarray(factored._routed(tokens, gather, jnp.asarray(digits)))
    np.testing.assert_array_equal(one, whole)
    weights = 2 ** (factored.FACTORED_SLICE * np.arange(4, -1, -1, dtype=np.int64))
    value = np.tensordot(weights, digits.astype(np.int64), axes=1)
    want = np.zeros((3, 4), np.int64)
    np.add.at(want, np.asarray(gather), value)
    np.testing.assert_array_equal(np.tensordot(weights, one.astype(np.int64), axes=1), want)
    assert (one[1:] >= 0).all() and (one[1:] < 2 ** factored.FACTORED_SLICE).all()


def test_the_rescues_class_presence_never_wraps():
    """A non-finite class's presence at a coordinate (`factored._present`) stays 1 over 256 chunks each
    counting 2**24 - 1 of its products, past int32's largest as a sum, where summed counts would wrap
    negative and read as none (NaN returned as 0); a chunk counting none leaves it 0 (rows 12403,
    12621)."""
    from dew.nn.kernels.templates.gemm import factored

    held = jnp.zeros((), jnp.int32)
    for _ in range(256):
        held = factored._present(held, jnp.float32(2 ** 24 - 1))
    assert int(held) == 1
    assert int(factored._present(jnp.zeros((), jnp.int32), jnp.float32(0.0))) == 0


def test_the_rescues_rounding_reads_a_leading_limb_past_one_digit():
    """A sum whose first limb (`factored._rounded`'s, of any int32 size) is 128, past one digit: 2**169 at a
    unit of 2**162 is +inf (-inf negated), not its wrapped bits; 128 + 2**-6 at a unit of one is exact, and
    2**30 at a unit of 2**-40 is 2**-10 (rows 1464, 12403)."""
    from dew.nn.kernels.templates.gemm import factored

    def rounded(digits, unit):
        limbs = jnp.zeros((8, 1, 1), jnp.int32).at[:len(digits), 0, 0].set(jnp.asarray(digits, jnp.int32))
        return float(factored._rounded(limbs, jnp.full((1, 1), unit, jnp.int32))[0, 0])

    assert np.isposinf(rounded([128], 162)) and np.isneginf(rounded([-128], 162))
    assert rounded([128, 1], 0) == 128.0 + 2.0 ** -6
    assert rounded([2 ** 30], -40) == 2.0 ** -10


def _cancelling():
    """Codes +1 and -1 at input 0 (the cancellation the rescue tests share), unit block scales, and one
    token at input 0."""
    return (jnp.zeros((512, 128), jnp.float32).at[0, 0].set(1.0).at[0, 1].set(-1.0),
            jnp.ones((32, 128), jnp.float8_e4m3fn), jnp.zeros((1, 512), jnp.float32).at[0, 0].set(1.0))


def _cancelling_bank():
    """Two experts: the first's codes cancelling (`_cancelling`) at a tensor scale of 2**110, the second's +1
    code at column 2 at 2**-8."""
    codes, scales, _ = _cancelling()
    second = jnp.zeros((512, 128), jnp.float32).at[0, 2].set(1.0)
    return formats.NvFp4.of(jnp.stack([codes, second]), jnp.stack([scales, scales]),
                            jnp.asarray([2.0 ** 110, 2.0 ** -8], jnp.float32))


@INTERPRETS
def test_an_nvfp4_products_input_gradient_cancels_past_fp32s_largest():
    """An NVFP4 product's input gradient where each cotangent times the matrix's factor (2**8 owed times a
    tensor scale of 2**110) passes fp32's largest, against codes +1 and -1 that cancel: 0, not inf - inf;
    a cotangent of 3 * 2**126 at a factor of 3 (a tensor scale of 3/256) against a +1 code of block scale
    one, 2**-8 decoded: 9 * 2**118, its product with the factor's significand moved, not overflowed (row
    1464's transpose, `factored`)."""
    codes, scales, _ = _cancelling()
    x = jnp.zeros((4, 512), jnp.float32).at[:, 0].set(1.0)
    matrix = formats.NvFp4.of(codes, scales, jnp.float32(2.0 ** 110))
    _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, matrix, out_dtype=jnp.float32), x)
    (d_x,) = pull(jnp.zeros((4, 128), jnp.float32).at[:2, :2].set(2.0 ** 20))
    np.testing.assert_array_equal(np.asarray(d_x), np.zeros((4, 512), np.float32))

    single = formats.NvFp4.of(jnp.zeros((512, 128), jnp.float32).at[0, 0].set(1.0), scales,
                              jnp.float32(3.0 / 256.0))
    _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, single, out_dtype=jnp.float32), x)
    (d_x,) = pull(jnp.zeros((4, 128), jnp.float32).at[0, 0].set(3.0 * 2.0 ** 126))
    want = np.zeros((4, 512), np.float32)
    want[0, 0] = 9.0 * 2.0 ** 118
    np.testing.assert_array_equal(np.asarray(d_x), want)


@INTERPRETS
def test_an_nvfp4_products_gathered_gradient_sums_its_routes_before_it_rounds():
    """Gathered, one token routed to two experts, the first's cancelling (cotangents 2**100 at a tensor
    scale of 2**110) beside the second's +1 code under a cotangent of 2**-30 at a tensor scale of 2**-8: the
    token's gradient is that second route's 2**-38; two routes' cotangents 256 and -255.5 through a 2**-8
    value sum to 2**-9 before they round, alone or beside a cancelling route, packed or decoded (row 1464's
    transpose, `factored`)."""
    (codes, scales, token), bank = _cancelling(), _cancelling_bank()
    second = jnp.zeros((512, 128), jnp.float32).at[0, 2].set(1.0)
    _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, bank, jnp.asarray([1, 1], jnp.int32),
                                                      jnp.asarray([0, 0], jnp.int32), out_dtype=jnp.float32),
                      token)
    (d_token,) = pull(jnp.zeros((2, 128), jnp.float32).at[0, :2].set(2.0 ** 100).at[1, 2].set(2.0 ** -30))
    want = np.zeros((1, 512), np.float32)
    want[0, 0] = 2.0 ** -38
    np.testing.assert_array_equal(np.asarray(d_token), want)

    # Routed twice through the packed kernel to one +1 code of unit block scale (2**-8 decoded, a tensor
    # scale of 2**-8, so a factor of one), its routes' cotangents 256 and -255.5 sum to 0.5 in fp32 before
    # rounding, as the packed kernel's pullback sums them: 2**-9, not each route rounded to bf16's 1 and -1.
    one = formats.NvFp4.of(jnp.zeros((1, 512, 128), jnp.float32).at[0, 0, 0].set(1.0), scales[None],
                           jnp.asarray([2.0 ** -8], jnp.float32))
    twice, both = jnp.asarray([2], jnp.int32), jnp.asarray([0, 0], jnp.int32)
    with generation.assumed("sm90"), generation.chosen(nvfp4="packed"):
        _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, one, twice, both, out_dtype=jnp.float32,
                                                          interpret_on_cpu=True), token)
        (d_token,) = pull(jnp.zeros((2, 128), jnp.float32).at[0, 0].set(256.0).at[1, 0].set(-255.5))
    want = np.zeros((1, 512), np.float32)
    want[0, 0] = 2.0 ** -9
    np.testing.assert_array_equal(np.asarray(d_token), want)

    # Beside a route whose products cancel past fp32's largest (the plain transpose not finite there), the
    # same token's routes of 256 and -255.5 through a 2**-8 value still sum wide: 2**-9, packed or decoded.
    mixed = formats.NvFp4.of(jnp.stack([codes, second]), jnp.stack([scales, scales]),
                             jnp.asarray([2.0 ** 110, 2.0 ** -8], jnp.float32))
    thrice = jnp.asarray([0, 0, 0], jnp.int32)
    for kernel in ("packed", "xla"):
        with generation.assumed("sm90"), generation.chosen(nvfp4=kernel):
            _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(
                rows, mixed, jnp.asarray([1, 2], jnp.int32), thrice, out_dtype=jnp.float32,
                interpret_on_cpu=True), token)
            (d_token,) = pull(jnp.zeros((3, 128), jnp.float32).at[0, :2].set(2.0 ** 100)
                              .at[1, 2].set(256.0).at[2, 2].set(-255.5))
        want = np.zeros((1, 512), np.float32)
        want[0, 0] = 2.0 ** -9
        np.testing.assert_array_equal(np.asarray(d_token), want, err_msg=kernel)



@INTERPRETS
def test_an_nvfp4_products_gathered_gradient_keeps_far_routes_apart_and_routes_none_to_zeros():
    """Two routes of one token far apart (one cancelling at 2**240, the other's products peaking at 2**97)
    each summed exactly; one route cancelling at 2**252 beside a third product 2**-126 times its factor keeps
    the third's 2**-8; and a gather routing no token pulls back zeros (row 1464's transpose, `factored`)."""
    (_, scales, token), bank = _cancelling(), _cancelling_bank()
    # Two routes of one token far apart: one cancelling at 2**240 on input 0, the other's products peaking
    # at 2**97 on input 1 and giving input 0 2**-28, each summed exactly (`aligned_segment_sum`).
    far = (jnp.zeros((2, 512, 128), jnp.float32).at[0, 0, 0].set(1.0).at[0, 0, 1].set(-1.0)
           .at[1, 1, 2].set(1.0).at[1, 0, 3].set(1.0))
    wide_bank = formats.NvFp4.of(far, jnp.stack([scales, scales]), jnp.asarray([2.0 ** 112, 2.0 ** -8],
                                                                                jnp.float32))
    pair, two = jnp.asarray([1, 1], jnp.int32), jnp.asarray([0, 0], jnp.int32)
    tokens2 = jnp.zeros((1, 512), jnp.float32).at[0, :2].set(1.0)
    for kernel in ("packed", "xla"):
        with generation.assumed("sm90"), generation.chosen(nvfp4=kernel):
            routed = functools.partial(gemm.nvfp4_product, matrix=wide_bank, group_sizes=pair, gather=two,
                                       out_dtype=jnp.float32, interpret_on_cpu=True)
            _, pull = jax.vjp(routed, tokens2)
            (d_token,) = pull(jnp.zeros((2, 128), jnp.float32).at[0, :2].set(2.0 ** 120)
                              .at[1, 2].set(2.0 ** 97).at[1, 3].set(2.0 ** -20))
        want = np.zeros((1, 512), np.float32)
        want[0, 0], want[0, 1] = 2.0 ** -28, 2.0 ** 89
        np.testing.assert_array_equal(np.asarray(d_token), want, err_msg=kernel)

    # One route's products cancelling at 2**252 beside a third, 2**-126 times its factor, on the same input:
    # each band of the route pulled back apart, the third keeps its 2**-8.
    three = jnp.zeros((1, 512, 128), jnp.float32).at[0, 0, :3].set(jnp.asarray([1.0, -1.0, 1.0]))
    lone = formats.NvFp4.of(three, scales[None], jnp.asarray([2.0 ** 118], jnp.float32))
    with generation.assumed("sm90"), generation.chosen(nvfp4="xla"):
        _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, lone, jnp.asarray([1], jnp.int32),
                                                          jnp.asarray([0], jnp.int32), out_dtype=jnp.float32),
                          token)
        cotangent = jnp.zeros((1, 128), jnp.float32).at[0, :2].set(2.0 ** 126).at[0, 2].set(2.0 ** -126)
        (d_token,) = pull(cotangent)
    want = np.zeros((1, 512), np.float32)
    want[0, 0] = 2.0 ** -8
    np.testing.assert_array_equal(np.asarray(d_token), want)

    # A gather routing no token pulls back zeros.
    none, nowhere = jnp.asarray([0, 0], jnp.int32), jnp.zeros((0,), jnp.int32)
    routed = functools.partial(gemm.nvfp4_product, matrix=bank, group_sizes=none, gather=nowhere,
                               out_dtype=jnp.float32)
    _, pull = jax.vjp(routed, token)
    (d_none,) = pull(jnp.zeros((0, 128), jnp.float32))
    np.testing.assert_array_equal(np.asarray(d_none), np.zeros((1, 512), np.float32))


@pytest.mark.parametrize("depth, devices", [(768, 3), (256, 4)],
                         ids=["whole groups a device", "groups cut by the devices"])
@pytest.mark.mesh(devices=4)
def test_a_depth_split_nvfp4_projection_decodes_and_pulls_back_on_an_explicit_mesh(depth, devices):
    """A dense NVFP4 matrix whose depth an Explicit 'tensor' axis splits (its words' and scales' last axis:
    768 deep three ways, whole 128-deep groups a device; 256 deep four ways, each device half a group),
    jitted: its decode is the codes' values placed by the same depth, its values (`packed_values`) the
    unsplit matrix's, and its projection of rows split by that depth, forward and the rows' gradient, the
    product written plainly over its values, exactly over small integers. Each group's staged order is
    undone where it lies, the devices that cut a group gathered off its words while it is (row 14097: a
    gather across the split depth had no placement; train 1466's review: the four-way split met a reshape
    of two groups, which JAX refused)."""
    from jax.sharding import AxisType, PartitionSpec

    from dew.nn.kernels.templates import product
    from dew.quantization import packed_values

    rng = np.random.default_rng(14097)
    codes = jnp.asarray(rng.integers(-2, 3, (depth, 128)), jnp.float32)
    blocks = depth // product.NVFP4.depth
    matrix = formats.NvFp4.of(codes, jnp.ones((blocks, 128), jnp.float8_e4m3fn), jnp.float32(2.0 ** -8))
    values = np.asarray(packed_values(matrix), np.float64)
    rows = rng.integers(-3, 4, (8, depth)).astype(np.float32)
    cotangent = rng.integers(-3, 4, (8, 128)).astype(np.float32)
    mesh = jax.make_mesh((devices,), ("tensor",), axis_types=(AxisType.Explicit,),
                         devices=jax.devices()[:devices])

    @jax.jit
    def run(x, matrix, cotangent):
        out, pull = jax.vjp(lambda x: gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32), x)
        return (matrix.decoded(), jax.reshard(packed_values(matrix), PartitionSpec()),
                jax.reshard(out, PartitionSpec()), jax.reshard(pull(cotangent)[0], PartitionSpec()))

    with jax.set_mesh(mesh):
        deep = PartitionSpec(None, "tensor")
        placed = formats.NvFp4(jax.device_put(matrix.words, deep), jax.device_put(matrix.scales, deep),
                               jax.device_put(matrix.scale, PartitionSpec()))
        decoded, found, out, grad = run(jax.device_put(rows, deep), placed,
                                        jax.device_put(cotangent, PartitionSpec()))
    assert jax.typeof(decoded).sharding.spec[0] == "tensor"
    np.testing.assert_array_equal(np.asarray(decoded, np.float64), np.asarray(matrix.decoded(), np.float64))
    np.testing.assert_array_equal(np.asarray(found, np.float64), values)
    np.testing.assert_array_equal(np.asarray(out, np.float64), rows.astype(np.float64) @ values)
    np.testing.assert_array_equal(np.asarray(grad, np.float64), cotangent.astype(np.float64) @ values.T)


def test_a_staged_order_no_transposition_states_is_refused_its_portable_decode():
    """The portable decode undoes a staged order by transposing each group's words and codes
    (`formats._unstaging`), so a declaration that no such transposition states is refused at the depth it
    decodes, not decoded wrong: NVFP4's own order with words 0 and 1 of code 0 exchanged in the third group
    alone, which the first two groups' strides cannot show (the review of row 14097's respin)."""
    from dew.nn.kernels.templates.gemm import packed

    def exchanged(group, word, code):
        depth = packed.nvfp4_staged_depth(group, word, code)
        moved = (group == 2) & (code == 0) & (word < 2)
        return np.where(moved, depth + 1 - 2 * word, depth)

    with pytest.raises(ValueError):
        formats._unstaging(768, packed.NVFP4_GROUP, bits.E2M1.width, exchanged, "NVFP4")


def _small_integer_projection(seed: int) -> tuple:
    """A `[512, 128]` NVFP4 matrix of small integer codes (unit block scales, a tensor scale of 2^-8, its
    values the codes), its values in float64 (`packed_values`), eight rows and their output's cotangent,
    small integers each: every product and sum of them exact."""
    from dew.quantization import packed_values

    rng = np.random.default_rng(seed)
    codes = jnp.asarray(rng.integers(-2, 3, (512, 128)), jnp.float32)
    matrix = formats.NvFp4.of(codes, jnp.ones((32, 128), jnp.float8_e4m3fn), jnp.float32(2.0 ** -8))
    rows = rng.integers(-3, 4, (8, 512)).astype(np.float32)
    cotangent = rng.integers(-3, 4, (8, 128)).astype(np.float32)
    return matrix, np.asarray(packed_values(matrix), np.float64), rows, cotangent


@pytest.mark.mesh(devices=2)
def test_a_column_split_nvfp4_projections_input_gradient_runs_on_an_explicit_mesh():
    """A dense NVFP4 projection whose matrix an Explicit 'tensor' axis splits by its columns (its words' and
    scales' leading axis), over replicated rows, jitted: its forward and its rows' gradient, the cotangent
    split by the same columns, are the product written plainly over its values (`packed_values`), exactly over
    small integers. The input gradient contracts the columns both its operands split, which JAX asks to be
    told where it lands (`xla_products.xla_dot` through `sharding.contracted_alike`, row 12225; train 1403's
    review: the placed call's XLA product raised)."""
    from jax.sharding import AxisType, PartitionSpec

    matrix, values, rows, cotangent = _small_integer_projection(14031)
    mesh = jax.make_mesh((2,), ("tensor",), axis_types=(AxisType.Explicit,))

    @jax.jit
    def pulled(x, matrix, cotangent):
        out, pull = jax.vjp(lambda x: gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32), x)
        return jax.reshard(out, PartitionSpec()), pull(cotangent)[0]

    with jax.set_mesh(mesh):
        split = PartitionSpec("tensor", None)
        placed = formats.NvFp4(jax.device_put(matrix.words, split), jax.device_put(matrix.scales, split),
                               jax.device_put(matrix.scale, PartitionSpec()))
        out, grad = pulled(jax.device_put(rows, PartitionSpec()), placed,
                           jax.device_put(cotangent, PartitionSpec(None, "tensor")))
    np.testing.assert_array_equal(np.asarray(out, np.float64), rows.astype(np.float64) @ values)
    np.testing.assert_array_equal(np.asarray(grad, np.float64), cotangent.astype(np.float64) @ values.T)


@pytest.mark.mesh(devices=3)
def test_a_factored_product_split_three_ways_by_its_depth_runs_on_an_explicit_mesh():
    """`factored`'s product of rows by a bfloat16 matrix 768 deep, an Explicit 'tensor' axis splitting the
    matrix's depth and the rows' three ways, jitted: its forward and its rows' gradient are the product
    written plainly, exactly over small integers. The exact rescue its pullback traces beside the plain one
    chunks the depth whole on every device (`depth_chunk`) and places each class's counts as the classes
    it updates (train 1455's review: 64 of a depth split three ways sliced no device's share whole, and
    the counts' split met whole classes; each trace raised)."""
    from jax.sharding import AxisType, PartitionSpec

    from dew.nn.kernels.templates.gemm.factored import factored

    rng = np.random.default_rng(14552)
    matrix = rng.integers(-2, 3, (768, 128)).astype(np.float32)
    rows = rng.integers(-3, 4, (8, 768)).astype(np.float32)
    cotangent = rng.integers(-3, 4, (8, 128)).astype(np.float32)
    mesh = jax.make_mesh((3,), ("tensor",), axis_types=(AxisType.Explicit,))

    def product(x, matrix):
        return factored(functools.partial(gemm.decoded_product, interpret_on_cpu=False), (matrix, None, None),
                        jnp.float32(1), x, wide=gemm.interpreted_wide(interpret_on_cpu=False),
                        wide_operands=(matrix, None))

    @jax.jit
    def pulled(x, matrix, cotangent):
        out, pull = jax.vjp(lambda x: product(x, matrix), x)
        return jax.reshard(out, PartitionSpec()), jax.reshard(pull(cotangent)[0], PartitionSpec())

    with jax.set_mesh(mesh):
        deep = jax.device_put(jnp.asarray(matrix, jnp.bfloat16), PartitionSpec("tensor", None))
        split_rows = jax.device_put(jnp.asarray(rows, jnp.bfloat16), PartitionSpec(None, "tensor"))
        out, grad = pulled(split_rows, deep, jax.device_put(cotangent, PartitionSpec()))
    np.testing.assert_array_equal(np.asarray(out, np.float64), rows.astype(np.float64) @ matrix)
    np.testing.assert_array_equal(np.asarray(grad, np.float64), cotangent.astype(np.float64) @ matrix.T)


@pytest.mark.mesh(devices=2)
def test_a_column_split_nvfp4_banks_column_scales_run_on_an_explicit_mesh():
    """A bank of two NVFP4 experts whose tensor scales are one a column (`NvFp4.columned`, row 13743: a
    ModelOpt source's fused gate and up), an Explicit 'tensor' axis splitting every field by its columns
    (the bank's own layout, `NvFp4.leaf_axes`), over replicated sorted bfloat16 rows and group sizes,
    jitted: its forward, exactly, and its rows' gradient, the cotangent split by the same columns, rounded
    once to the rows' bfloat16, are the product written plainly over each expert's values
    (`packed_values`) in float64, over small integers; each row reads its group's column scales where the
    product and its pullback expand them (`factored`'s `groups`)."""
    from jax.sharding import AxisType, PartitionSpec

    from dew.quantization import packed_values

    rng = np.random.default_rng(14033)
    codes = jnp.asarray(rng.integers(-2, 3, (2, 512, 128)), jnp.float32)
    scale = jnp.asarray(np.exp2(rng.integers(-10, -6, (2, 128))), jnp.float32)
    bank = formats.NvFp4.of(codes, jnp.ones((2, 32, 128), jnp.float8_e4m3fn), scale)
    values = np.asarray(packed_values(bank), np.float64)
    counts = (40, 24)
    sizes = jnp.asarray(counts, jnp.int32)
    rows = rng.integers(-3, 4, (64, 512)).astype(np.float32)
    cotangent = rng.integers(-3, 4, (64, 128)).astype(np.float32)
    mesh = jax.make_mesh((2,), ("tensor",), axis_types=(AxisType.Explicit,))
    narrow = jnp.asarray(rows, jnp.bfloat16)

    @jax.jit
    def pulled(x, bank, sizes, cotangent):
        out, pull = jax.vjp(lambda x: gemm.nvfp4_product(x, bank, sizes, out_dtype=jnp.float32), x)
        return jax.reshard(out, PartitionSpec()), pull(cotangent)[0]

    with jax.set_mesh(mesh):
        split = PartitionSpec(None, "tensor")
        placed = formats.NvFp4(*(jax.device_put(field, split)
                                 for field in (bank.words, bank.scales, bank.scale)))
        whole = PartitionSpec()
        out, grad = pulled(jax.device_put(narrow, whole), placed, jax.device_put(sizes, whole),
                           jax.device_put(cotangent, split))
    product, gradient = np.zeros((64, 128)), np.zeros((64, 512))
    for group, (first, last) in enumerate(itertools.pairwise(np.cumsum([0, *counts]))):
        product[first:last] = rows[first:last].astype(np.float64) @ values[group]
        gradient[first:last] = cotangent[first:last].astype(np.float64) @ values[group].T
    np.testing.assert_array_equal(np.asarray(out, np.float64), product)
    rounded = jnp.asarray(gradient, jnp.float32).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(grad, np.float32), np.asarray(rounded, np.float32))


@pytest.mark.mesh(devices=2)
def test_an_expert_split_nvfp4_banks_column_scales_run_on_an_explicit_mesh():
    """A bank of two NVFP4 experts whose tensor scales are one a column (`NvFp4.columned`), an Explicit
    'expert' axis splitting every field by its experts (`P('expert', ...)`), over replicated rows one an
    expert, jitted: its forward and its rows' gradient are the product written plainly over each expert's
    values (`packed_values`) in float64, exactly over small integers, the rows' gradient rounded once to
    their bfloat16; each row reads its expert's column scales where an Explicit axis splits the experts
    (`axes.at_rows`, train 1409's review: the repeat along a split axis raised)."""
    from jax.sharding import AxisType, PartitionSpec

    from dew.quantization import packed_values

    rng = np.random.default_rng(14091)
    codes = jnp.asarray(rng.integers(-2, 3, (2, 256, 128)), jnp.float32)
    scale = jnp.asarray(np.exp2(rng.integers(-10, -6, (2, 128))), jnp.float32)
    bank = formats.NvFp4.of(codes, jnp.ones((2, 16, 128), jnp.float8_e4m3fn), scale)
    values = np.asarray(packed_values(bank), np.float64)
    rows = rng.integers(-3, 4, (2, 256)).astype(np.float32)
    cotangent = rng.integers(-3, 4, (2, 128)).astype(np.float32)
    mesh = jax.make_mesh((2,), ("expert",), axis_types=(AxisType.Explicit,))

    @jax.jit
    def pulled(x, bank, cotangent):
        out, pull = jax.vjp(lambda x: gemm.nvfp4_product(x, bank, jnp.asarray([1, 1], jnp.int32),
                                                         out_dtype=jnp.float32), x)
        return jax.reshard(out, PartitionSpec()), pull(cotangent)[0]

    with jax.set_mesh(mesh):
        experts = PartitionSpec("expert")
        placed = formats.NvFp4(*(jax.device_put(field, experts)
                                 for field in (bank.words, bank.scales, bank.scale)))
        whole = PartitionSpec()
        out, grad = pulled(jax.device_put(jnp.asarray(rows, jnp.bfloat16), whole), placed,
                           jax.device_put(cotangent, whole))
    product = np.stack([rows[expert].astype(np.float64) @ values[expert] for expert in range(2)])
    gradient = np.stack([cotangent[expert].astype(np.float64) @ values[expert].T for expert in range(2)])
    np.testing.assert_array_equal(np.asarray(out, np.float64), product)
    rounded = jnp.asarray(gradient, jnp.float32).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(grad, np.float32), np.asarray(rounded, np.float32))


def _read_at_rows(side, sizes, cotangent) -> tuple:
    """A group side read at the cotangent's rows (`axes.at_rows`), its sum against the cotangent beside the
    rows: a value and its gradient's witness."""
    from dew.nn.kernels.templates import axes

    rows = axes.at_rows(side, (axes.GROUPS, "columns"), sizes, cotangent.shape[0])
    return jnp.sum(rows * cotangent), rows


@pytest.mark.mesh(devices=2)
def test_a_replicated_group_side_read_at_rows_its_split_group_sizes_count_is_its_repeat():
    """`axes.at_rows` of a replicated `[4, 128]` group side over group sizes of one row each that an
    Explicit axis splits over two devices, jitted outside the mesh's context: each row its group's value
    and its gradient each group's row's cotangent, as the unsplit repeat reads them, exactly over small
    integers (train 1437's review: the split sizes repeated the side by each device's half, which raised)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    rng = np.random.default_rng(14371)
    side = jnp.asarray(rng.integers(-3, 4, (4, 128)), jnp.float32)
    cotangent = jnp.asarray(rng.integers(-3, 4, (4, 128)), jnp.float32)

    def read(side, sizes):
        return _read_at_rows(side, sizes, cotangent)

    sizes = jnp.ones((4,), jnp.int32)
    mesh = jax.make_mesh((2,), ("expert",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    split = jax.device_put(sizes, NamedSharding(mesh, PartitionSpec("expert")))
    whole = jax.device_put(side, NamedSharding(mesh, PartitionSpec()))
    (_, rows), grad = jax.jit(jax.value_and_grad(read, has_aux=True))(whole, split)
    (_, plain_rows), plain_grad = jax.value_and_grad(read, has_aux=True)(side, sizes)
    np.testing.assert_array_equal(np.asarray(rows), np.asarray(plain_rows))
    np.testing.assert_array_equal(np.asarray(grad), np.asarray(plain_grad))


@pytest.mark.mesh(devices=4)
def test_a_group_side_read_at_its_rows_across_an_explicit_groups_split_is_its_repeat():
    """`axes.at_rows` of a `[2, 128]` group side whose groups one Explicit axis splits and whose columns
    another does, over groups of 3 and 5 rows and 2 rows past them, jitted: each row its group's value (a
    row past the groups the last's), as the unsplit repeat reads them, placed whole along the rows with its
    columns' split kept, and its gradient under each row's cotangent each group's rows summed, as the
    repeat's transpose sums them, placed as the side is, exactly over small integers (train 1409's review:
    the repeat along a split axis raised)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    rng = np.random.default_rng(14092)
    side = jnp.asarray(rng.integers(-3, 4, (2, 128)), jnp.float32)
    cotangent = jnp.asarray(rng.integers(-3, 4, (10, 128)), jnp.float32)
    sizes = jnp.asarray([3, 5], jnp.int32)

    def read(side):
        return _read_at_rows(side, sizes, cotangent)

    mesh = jax.make_mesh((2, 2), ("expert", "tensor"), axis_types=(AxisType.Explicit,) * 2,
                         devices=jax.devices()[:4])
    split = jax.device_put(side, NamedSharding(mesh, PartitionSpec("expert", "tensor")))
    (_, rows), grad = jax.jit(jax.value_and_grad(read, has_aux=True))(split)
    (_, plain_rows), plain_grad = jax.value_and_grad(read, has_aux=True)(side)
    assert rows.sharding.spec == PartitionSpec(None, "tensor"), rows.sharding
    assert grad.sharding.spec == PartitionSpec("expert", "tensor"), grad.sharding
    np.testing.assert_array_equal(np.asarray(rows), np.asarray(plain_rows))
    np.testing.assert_array_equal(np.asarray(grad), np.asarray(plain_grad))


@pytest.mark.mesh(devices=2)
def test_an_nvfp4_pullback_over_a_matrix_split_by_its_columns_alone_runs_outside_its_mesh():
    """A dense NVFP4 projection whose matrix alone an Explicit 'tensor' axis splits by its columns, its rows,
    tensor scale and cotangent unplaced, jitted outside the mesh's context (the matrix's type carrying it):
    its rows' gradient is the product written plainly over its values (`packed_values`), exactly over small
    integers, the exact rescue (traced in every pullback) placing its sums on the matrix's mesh (train
    1426's review: meshless sums beside a GEMM on the matrix's)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    matrix, values, rows, cotangent = _small_integer_projection(14263)
    mesh = jax.make_mesh((2,), ("tensor",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    split = NamedSharding(mesh, PartitionSpec("tensor"))
    placed = formats.NvFp4(jax.device_put(matrix.words, split), jax.device_put(matrix.scales, split),
                           matrix.scale)

    def pulled(x, matrix, cotangent):
        _, pull = jax.vjp(lambda x: gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32), x)
        return pull(cotangent)[0]

    grad = jax.jit(pulled)(rows, placed, cotangent)
    np.testing.assert_array_equal(np.asarray(grad, np.float64), cotangent.astype(np.float64) @ values.T)


@pytest.mark.mesh(devices=2)
def test_an_nvfp4_products_gathered_gradient_in_a_checked_map_is_the_tokens_sum():
    """An NVFP4 bank's gathered product in a `shard_map` over two CPU devices that checks the axes its values
    vary over, the tokens and bank every device's and each device its own routing table: the tokens'
    gradient is each device's routes summed into their tokens and over the devices, as the product written
    plainly over the decoded bank gives it, exactly, over small integers (row 1464)."""
    rng = np.random.default_rng(14641)
    codes = jnp.asarray(rng.integers(-2, 3, (2, 512, 128)), jnp.float32)
    bank = formats.NvFp4.of(codes, jnp.ones((2, 32, 128), jnp.float8_e4m3fn),
                            jnp.asarray([2.0 ** -8, 2.0 ** -7], jnp.float32))
    decoded = jnp.asarray(bank.decoded(), jnp.float32)
    tokens = jnp.asarray(rng.integers(-3, 4, (4, 512)), jnp.float32)
    tables = jnp.asarray([0, 1, 2, 3, 3, 2, 1, 0], jnp.int32)
    sizes = jnp.asarray([2, 2], jnp.int32)
    weights = jnp.asarray(rng.integers(-2, 3, (8, 128)), jnp.float32)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    data, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def packed(x, table, w):
        return jnp.sum(gemm.nvfp4_product(x, bank, sizes, table, out_dtype=jnp.float32) * w)

    def plain(x, table, w):
        experts = jnp.repeat(jnp.arange(2), sizes, total_repeat_length=table.shape[0])
        rows = x[table].astype(jnp.bfloat16).astype(jnp.float32)
        factor = jnp.float32(bits.NVFP4_OWED) * bank.scale[experts][:, None]
        return jnp.sum(jnp.einsum('rd,rdw->rw', rows, decoded[experts]) * factor * w)

    def gradient(product):
        def each_device(x, table, w):
            return jax.grad(product)(x, table, w)

        return jax.jit(jax.shard_map(each_device, mesh=mesh, in_specs=(whole, data, data), out_specs=whole,
                                     check_vma=True))(tokens, tables, weights)

    np.testing.assert_array_equal(np.asarray(gradient(packed)), np.asarray(gradient(plain)))


@pytest.mark.mesh(devices=2)
def test_an_nvfp4_products_gathered_rescue_sums_devices_routes_before_it_moves_them_back():
    """One token every device holds, each of two CPU devices routing it to its own expert in a checked
    `shard_map`, +1 on one and -1 on the other at a tensor scale of 2**110 under a cotangent of 2**30: each
    device's partial, +-2**140, is past fp32's largest, and their sum is 0, not inf - inf, summed while
    still moved by the token's shared power; beside them, the first device's second route, a +1 at a
    tensor scale of 2**-8 under a cotangent of 2**-30, keeps its 2**-38 unmoved (row 1464)."""
    codes = (jnp.zeros((3, 512, 128), jnp.float32).at[0, 0, 0].set(1.0).at[1, 0, 0].set(-1.0)
             .at[2, 0, 2].set(1.0))
    bank = formats.NvFp4.of(codes, jnp.ones((3, 32, 128), jnp.float8_e4m3fn),
                            jnp.asarray([2.0 ** 110, 2.0 ** 110, 2.0 ** -8], jnp.float32))
    token = jnp.zeros((1, 512), jnp.float32).at[0, 0].set(1.0)
    tables = jnp.zeros((4,), jnp.int32)
    sizes = jnp.asarray([1, 0, 1, 0, 1, 1], jnp.int32)
    cotangents = jnp.zeros((4, 128), jnp.float32).at[0, 0].set(2.0 ** 30).at[1, 2].set(2.0 ** -30)
    cotangents = cotangents.at[2, 0].set(2.0 ** 30)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    data, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def each_device(x, table, size, w):
        return jax.grad(lambda x: jnp.sum(gemm.nvfp4_product(x, bank, size, table, out_dtype=jnp.float32)
                                          * w))(x)

    d_token = jax.jit(jax.shard_map(each_device, mesh=mesh, in_specs=(whole, data, data, data),
                                    out_specs=whole, check_vma=True))(token, tables, sizes, cotangents)
    want = np.zeros((1, 512), np.float32)
    want[0, 0] = 2.0 ** -38
    np.testing.assert_array_equal(np.asarray(d_token), want)


def test_an_nvfp4_products_dense_rescue_keeps_each_product_through_its_band():
    """An NVFP4 product's input gradient past fp32's largest, one row: input 0's codes +1, -1, +1 and input
    1's +1 under four cotangents, each product kept through its band (`factored`'s), whatever lies beside it:
    2**101 of 2**100, 2**100, 2**79 at a factor of 2**30; 2**-8 of 2**-126 beside 2**126s at a factor of
    2**126, input 1's 2**244 past fp32's largest; 2**9 of a 2**-99 through an entry of 2**-18 (a code of 0.5
    at a block scale of 2**-9) beside cancelling 2**126s; and 2**33 of 2**-85 beside cancelling 2**-126s
    (row 1464)."""
    scales = jnp.ones((32, 128), jnp.float8_e4m3fn)
    row = jnp.zeros((1, 512), jnp.float32).at[0, :2].set(1.0)
    spread = jnp.zeros((512, 128), jnp.float32).at[0, :3].set(jnp.asarray([1.0, -1.0, 1.0])).at[1, 3].set(1.0)
    fine = scales.at[0, 2].set(jnp.float8_e4m3fn(2.0 ** -9))

    def pulled(tensor, cotangent, codes=spread, blocks=scales):
        matrix = formats.NvFp4.of(codes, blocks, jnp.float32(tensor))
        _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, matrix, out_dtype=jnp.float32), row)
        (d_row,) = pull(jnp.zeros((1, 128), jnp.float32).at[0, :4].set(jnp.asarray(cotangent, jnp.float32)))
        return np.asarray(d_row)[0, :2]

    np.testing.assert_array_equal(pulled(2.0 ** 22, (2.0 ** 100, 2.0 ** 100, 2.0 ** 79, 0.0)),
                                  np.float32([2.0 ** 101, 0.0]))
    np.testing.assert_array_equal(pulled(2.0 ** 118, (2.0 ** 100, 2.0 ** 100, 2.0 ** -126, 2.0 ** 126)),
                                  np.float32([2.0 ** -8, np.inf]))
    halves = spread.at[0, 2].set(0.5)
    np.testing.assert_array_equal(pulled(2.0 ** 118, (2.0 ** 126, 2.0 ** 126, 2.0 ** -99, 0.0), halves, fine),
                                  np.float32([2.0 ** 9, 0.0]))
    plus = spread.at[0, :3].set(jnp.asarray([1.0, 1.0, -1.0]))
    cotangent = (2.0 ** -85, 2.0 ** -126, 2.0 ** -126, 2.0 ** 126)
    np.testing.assert_array_equal(pulled(2.0 ** 118, cotangent, plus), np.float32([2.0 ** 33, np.inf]))


def test_an_nvfp4_products_rescue_keeps_ieee_and_subnormal_operands():
    """Beside a product past fp32's largest (which takes the rescue): an infinite cotangent's entries are
    IEEE's, as float64's transpose gives them (inf through a +1 code, NaN through every 0 one); and a
    subnormal cotangent's product, 2**-149 times a factor of 2**126 through a 2**-8 entry, is 2**-31, read
    from its bits where a process flushes subnormal operands (row 1464)."""
    scales = jnp.ones((32, 128), jnp.float8_e4m3fn)
    codes = jnp.zeros((512, 128), jnp.float32).at[0, 0].set(1.0).at[1, 1].set(1.0).at[2, 2].set(1.0)
    matrix = formats.NvFp4.of(codes, scales, jnp.float32(2.0 ** 118))
    row = jnp.zeros((1, 512), jnp.float32).at[0, :3].set(1.0)
    _, pull = jax.vjp(lambda rows: gemm.nvfp4_product(rows, matrix, out_dtype=jnp.float32), row)
    infinite = jnp.zeros((1, 128), jnp.float32).at[0, 0].set(2.0 ** 126).at[0, 1].set(jnp.inf)
    (d_row,) = pull(infinite)
    with np.errstate(invalid="ignore", over="ignore"):
        products = np.asarray(infinite, np.float64) * 2.0 ** 126
        truth = (products @ (np.asarray(codes, np.float64).T * 2.0 ** -8)).astype(np.float32)
    np.testing.assert_array_equal(np.isnan(np.asarray(d_row)), np.isnan(truth))
    np.testing.assert_array_equal(np.isinf(np.asarray(d_row)), np.isinf(truth))
    subnormal = jnp.zeros((1, 128), jnp.float32).at[0, 0].set(2.0 ** 126)
    subnormal = subnormal.at[0, 2].set(jnp.float32(2.0 ** -149))
    (d_row,) = pull(subnormal)
    want = np.zeros((512,), np.float32)
    want[0], want[2] = np.inf, 2.0 ** -31
    np.testing.assert_array_equal(np.asarray(d_row)[0], want)


def test_an_nvfp4_products_gathered_rescue_sums_across_its_bands_seam_exactly():
    """One token routed to four experts at a factor of 2**126 each: a route's product of 2**165 just above a
    band's seam (2**183 through 2**-18), another's of -2**165 just below it (2**182 through -2**-17), a
    third's 2**125 below it too, and a fourth's 2**244 on input 1, past fp32's largest: input 0's gradient
    is 2**125, every band's routes summed into the token exactly before one rounding, not each band rounded
    apart (row 1464)."""
    codes = jnp.zeros((4, 512, 128), jnp.float32)
    codes = codes.at[0, 0, 0].set(0.5).at[1, 0, 0].set(-1.0).at[2, 0, 0].set(0.5).at[3, 1, 1].set(1.0)
    scales = jnp.ones((4, 32, 128), jnp.float8_e4m3fn).at[:3, 0, 0].set(jnp.float8_e4m3fn(2.0 ** -9))
    bank = formats.NvFp4.of(codes, scales, jnp.full((4,), 2.0 ** 118, jnp.float32))
    token = jnp.zeros((1, 512), jnp.float32).at[0, :2].set(1.0)
    routed = functools.partial(gemm.nvfp4_product, matrix=bank, group_sizes=jnp.ones((4,), jnp.int32),
                               gather=jnp.zeros((4,), jnp.int32), out_dtype=jnp.float32)
    _, pull = jax.vjp(routed, token)
    cotangent = jnp.zeros((4, 128), jnp.float32).at[0, 0].set(2.0 ** 57).at[1, 0].set(2.0 ** 56)
    (d_token,) = pull(cotangent.at[2, 0].set(2.0 ** 17).at[3, 1].set(2.0 ** 126))
    np.testing.assert_array_equal(np.asarray(d_token)[0, :3], np.float32([2.0 ** 125, np.inf, 0.0]))


def test_an_nvfp4_operand_at_a_given_tensor_scale_is_the_rule_at_that_scale():
    """`NvFp4Block.quantized(x, tensor_scale)`, a delayed-scaling recipe's operand (row 12721): at `x`'s own
    tensor scale it is the current rule's operand bit for bit; at four times it, the operand of `x / 4` at
    `x`'s scale (an exact power of two through every quotient), its tensor scale the given one; at a scale
    below `x`'s own, its block scales saturate at 448 and its codes at 6, as delayed scaling clips; and a
    scale an array of more than one is refused."""
    from dew.nn.kernels.projections import delayed_tensor_scale

    x = jnp.asarray(np.random.default_rng(12721).normal(size=(64, 128)), jnp.float32)
    own = formats.NvFp4Block.quantized(x)
    at_own = formats.NvFp4Block.quantized(x, own.scale)
    quartered = formats.NvFp4Block.quantized(x / 4, own.scale)
    at_four = formats.NvFp4Block.quantized(x, 4 * own.scale)
    for got, want in ((at_own, own), (at_four, quartered)):
        np.testing.assert_array_equal(np.asarray(got.codes).view(np.uint8),
                                      np.asarray(want.codes).view(np.uint8))
        np.testing.assert_array_equal(np.asarray(got.scales).view(np.uint8),
                                      np.asarray(want.scales).view(np.uint8))
    assert float(at_four.scale) == 4 * float(own.scale)
    clipped = formats.NvFp4Block.quantized(x, own.scale / 64)
    assert float(np.max(np.asarray(clipped.scales, np.float32))) == 448.0
    assert float(np.max(np.abs(np.asarray(clipped.codes, np.float32)))) == 6.0
    history = jnp.asarray([0.5, float(jnp.max(jnp.abs(x))), 2.0], jnp.float32)
    assert float(delayed_tensor_scale(history)) == float(bits.nvfp4_tensor_scale(jnp.max(history)))
    with pytest.raises(ValueError):  # one scale for the tensor, never one a row
        formats.NvFp4Block.quantized(x, jnp.full((64,), own.scale))


def test_an_nvfp4_projection_at_its_rows_own_tensor_scale_is_the_current_recipes_bitwise():
    """`projections.nvfp4_projection` given its rows' own tensor scale (the delayed recipe's at a history
    of this step alone, row 12721): its value and both gradients bit for bit the current recipe's."""
    from dew.nn.kernels import projections

    rng = np.random.default_rng(12722)
    x = jnp.asarray(rng.normal(size=(128, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(1, 128, 64)) / 8, jnp.bfloat16)
    blocks = jnp.asarray([1], jnp.int32)
    scale = bits.nvfp4_tensor_scale(projections.nvfp4_amax(x))

    def run(given):
        def projected(x, w):
            out = projections.nvfp4_projection(x, w, blocks, out_dtype=jnp.bfloat16, tensor_scale=given)
            return jnp.sum(out.astype(jnp.float32)), out
        (_, out), grads = jax.value_and_grad(projected, argnums=(0, 1), has_aux=True)(x, w)
        return out, *grads

    for got, want in zip(run(scale), run(None), strict=True):
        np.testing.assert_array_equal(np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16))


def test_a_delayed_nvfp4_recipe_takes_its_rows_tensor_scale_and_only_it_does():
    """A run's NVFP4 recipe under delayed tensor scaling (`LowPrecision.tensor_scaling`, opt-in, row 12721):
    its recipe projection requires the rows' tensor scale and is `nvfp4_projection` at it; the current
    recipe refuses one; a layer's product (`recipe_general`, a `Dense`'s) and its `dot_general` take the
    scale they are given; and delayed scaling is NVFP4's alone."""
    from dew.nn.kernels import projections
    from dew.nn.precision import DTYPE_PRESETS, NVFP4, LowPrecision, Numerics, projection_role

    with pytest.raises(ValueError):
        LowPrecision(recipe="mxfp8", tensor_scaling="delayed")
    rng = np.random.default_rng(12723)
    x = jnp.asarray(rng.normal(size=(100, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(128, 64)) / 8, jnp.bfloat16)
    scale = jnp.float32(0.25)
    delayed = Numerics(DTYPE_PRESETS["bf16-mixed"],
                       low_precision=LowPrecision(recipe="nvfp4", tensor_scaling="delayed"))
    current = NVFP4
    with generation.run_precision(delayed), projection_role("mlp"):
        with pytest.raises(ValueError):
            projections.recipe_projection(x, w, out_dtype=jnp.bfloat16)
        got = projections.recipe_projection(x, w, out_dtype=jnp.bfloat16, tensor_scale=scale)
    laid = jnp.pad(x, ((0, 28), (0, 0)))
    want = projections.nvfp4_projection(laid, w[None], jnp.asarray([1], jnp.int32), out_dtype=jnp.bfloat16,
                                        tensor_scale=scale)[:100]
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16))
    with generation.run_precision(current), projection_role("mlp"), pytest.raises(ValueError):
        projections.recipe_projection(x, w, out_dtype=jnp.bfloat16, tensor_scale=scale)
    numbers = (((1,), (0,)), ((), ()))
    with generation.run_precision(delayed), projection_role("mlp"):
        general = projections.recipe_general(x, w, numbers, jnp.bfloat16, tensor_scale=scale)
        layered = projections.dot_general(jax.lax.dot_general)(x, w, numbers,
                                                                 preferred_element_type=jnp.bfloat16,
                                                                 tensor_scale=scale)
    for got in (general, layered):
        np.testing.assert_array_equal(np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16))


def test_a_delayed_layers_product_takes_each_call_its_own_tensor_scale():
    """A layer's product under delayed NVFP4 scaling (`recipe_general`, a `Dense`'s, and its `dot_general`)
    takes its rows' tensor scale as an operand (row 12721): one compiled program called at two scales is the
    projection at each, and under `jax.checkpoint` in a `jax.lax.scan` over the scales each step's value is
    the projection's at its scale and the gradients the sum of theirs."""
    from dew.nn.kernels import projections
    from dew.nn.precision import DTYPE_PRESETS, LowPrecision, Numerics, projection_role

    rng = np.random.default_rng(12725)
    x = jnp.asarray(rng.normal(size=(128, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(128, 64)) / 8, jnp.bfloat16)
    numbers = (((1,), (0,)), ((), ()))
    scales = jnp.asarray([0.25, 2.0 ** -12], jnp.float32)
    blocks = jnp.asarray([1], jnp.int32)

    def summed(product):
        def total(x, w):
            out = product(x, w)
            return jnp.sum(out.astype(jnp.float32)), out
        return total

    def apart(scale):
        projected = summed(lambda x, w: projections.nvfp4_projection(
            x, w[None], blocks, out_dtype=jnp.bfloat16, tensor_scale=scale))
        (_, out), grads = jax.value_and_grad(projected, argnums=(0, 1), has_aux=True)(x, w)
        return out, grads

    def same(got, want):
        np.testing.assert_array_equal(np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16))

    wants = [apart(scale) for scale in scales]
    # The two scales quantize the rows apart: a program reusing the first scale's would differ.
    assert not np.array_equal(np.asarray(wants[0][0]), np.asarray(wants[1][0]))
    delayed = Numerics(DTYPE_PRESETS["bf16-mixed"],
                       low_precision=LowPrecision(recipe="nvfp4", tensor_scaling="delayed"))
    with generation.run_precision(delayed), projection_role("mlp"):
        general = jax.jit(lambda x, w, s: projections.recipe_general(x, w, numbers, jnp.bfloat16,
                                                                     tensor_scale=s))
        layered = jax.jit(lambda x, w, s: projections.dot_general(jax.lax.dot_general)(
            x, w, numbers, preferred_element_type=jnp.bfloat16, tensor_scale=s))
        for scale, (want, _) in zip(scales, wants, strict=True):
            same(general(x, w, scale), want)
            same(layered(x, w, scale), want)

        def stepped(x, w):
            def step(total, scale):
                product = summed(lambda x, w: projections.recipe_general(x, w, numbers, jnp.bfloat16,
                                                                         tensor_scale=scale))
                each, out = jax.checkpoint(product)(x, w)
                return total + each, out
            return jax.lax.scan(step, jnp.float32(0), scales)

        (_, outs), grads = jax.jit(jax.value_and_grad(stepped, argnums=(0, 1), has_aux=True))(x, w)
    for step, (want, _) in enumerate(wants):
        same(outs[step], want)
    for got, first, second in zip(grads, wants[0][1], wants[1][1], strict=True):
        same(got, first + second)


def test_a_delayed_scale_is_its_records_largest_finite_magnitude():
    """`nvfp4_amax` and `delayed_tensor_scale` (row 12721), compiled, reduce finite values alone from zero:
    an empty or all-zero history's scale is 1, a NaN or infinite record leaves the finite ones' scale, an
    empty or all-nonfinite tensor records 0, and a subnormal largest is the rule's own as NaN beside it
    leaves it."""
    from dew.nn.kernels import projections

    amax, scale = jax.jit(projections.nvfp4_amax), jax.jit(projections.delayed_tensor_scale)
    rule = jax.jit(bits.nvfp4_tensor_scale)
    for history, want in (((), 1.0), ((0.0, 0.0), 1.0), ((np.nan, 3.0), rule(jnp.float32(3.0))),
                          ((np.inf, 3.0, 0.5), rule(jnp.float32(3.0))), ((np.nan, -np.inf), 1.0)):
        assert float(scale(jnp.asarray(history, jnp.float32))) == float(want)
    assert float(amax(jnp.asarray([[np.nan, -5.0], [np.inf, 2.0]], jnp.float32))) == 5.0
    for empty in (jnp.zeros((0, 4), jnp.bfloat16), jnp.full((2, 2), np.nan, jnp.float32),
                  jnp.full((3,), -np.inf, jnp.float32)):
        assert float(amax(empty)) == 0.0
    subnormal = np.float32(2.0 ** -140)
    alone, besides = (scale(jnp.asarray(history, jnp.float32))
                      for history in ((subnormal,), (subnormal, np.nan)))
    assert float(alone) == float(besides) and np.isfinite(float(alone)) and float(alone) > 0
    assert float(amax(jnp.asarray([subnormal, np.nan, np.inf]))) == float(amax(jnp.asarray([subnormal])))


@pytest.mark.parametrize("history", [0, 1.5, float("nan"), float("inf"), True])
def test_an_amax_history_is_a_whole_number_of_steps(history):
    """`LowPrecision.amax_history` (row 12721) refuses a history of no steps, a fraction, NaN, infinity and
    a bool."""
    from dew.nn.precision import LowPrecision

    with pytest.raises(ValueError):
        LowPrecision(recipe="nvfp4", tensor_scaling="delayed", amax_history=history)


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_a_grouped_nvfp4_product_of_no_depth_is_zero(generation_name):
    """A grouped NVFP4 product of rows of no depth and a bank of one empty matrix `[1, 0, 64]` is zeros,
    its launch priced by its format's bytes a value (a bank of no values, row 5387's review: a byte count
    divided by its size raised before its route was chosen)."""
    matrix = formats.NvFp4.of(np.zeros((1, 0, 64), np.float32), np.zeros((1, 0, 64), ml_dtypes.float8_e4m3fn),
                              np.ones((1,), np.float32))
    with generation.assumed(generation_name):
        out = gemm.nvfp4_product(jnp.zeros((8, 0), jnp.float32), matrix, jnp.asarray([8], jnp.int32),
                                 out_dtype=jnp.float32, interpret_on_cpu=True)
    np.testing.assert_array_equal(np.asarray(out), np.zeros((8, 64), np.float32))


def test_a_packed_steps_price_counts_its_decodes_integer_pipes():
    """A step whose NVFP4 operand its consumers decode (`pricing.step_seconds`, the form's work in its
    `Decoded`, row 14106): on an H100's narrowest legal packed tile (two consumers decoding a 256-deep stage
    of 128 rows for 8 tokens, each weight's half byte and its block's scale stored) the decode ties the
    shared bytes, so its price stands, and a decode of four times its work outlasts them, the step priced
    at it; and two operands' decodes on one pipe add, not overlap, each pipe's work summed before the
    busiest is read (its review: a 64x64x256 step's two decodes 384 clocks, not 192)."""
    from dew.nn.kernels.templates import cost
    from dew.nn.kernels.templates.gemm import packed, pricing

    h100, work = generation.gpu_target("sm90"), packed.nvfp4_decode_work()
    stored = 0.5 + 1 / 16
    narrow = tiling.Tiles(128, 8, 256, 2, 1, warpgroups=2, family="wgmma")

    def priced(form):
        return pricing.step_seconds(narrow, h100, (stored, 2), (form, None), issued=jnp.bfloat16)

    assert priced(pricing.Decoded(0, 1, work)) == priced(pricing.Decoded(0, 1))
    assert priced(pricing.Decoded(0, 1, cost.repeated(work, 4))) > priced(pricing.Decoded(0, 1))
    rates, square = cost.rates(h100, device=False), tiling.Tiles(64, 64, 256, 2, 1, family="wgmma")
    one = pricing.Decoded(2, 1, work)
    # The square tile's two operands alike, each decode's seconds the other's; doubled exactly.
    left, right = (pricing.decode_seconds(square, forms, rates) for forms in ((one, None), (None, one)))
    assert left == right and pricing.decode_seconds(square, (one, one), rates) == 2 * left
