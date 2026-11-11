"""The GEMM template's gathered rows (row 1376): a grouped product whose rows are read from the tokens where a
table says, a mixture of experts' sort fused into the rows' loads (`mosaic_gpu.gmm`'s `gather`), and a kernel
gradient's operands read so (`mosaic_gpu.tgmm`'s)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, equations

from dew.nn.kernels import formats, gemm, generation, grouped_matmul, tuning
from dew.nn.kernels.templates.gemm import mosaic_gpu

SEEDS = 8
"""The gathered product's check-1 seeds, the template's grouped products' count."""


def test_a_gathered_grouped_product_reads_each_row_where_its_table_says():
    """`a_gathered_product_is_on_par` on an H100, interpreted (each row a copy at its slot's swizzled row,
    `Ring.parts`), plain and paired; a gathered product on mma.sync, which has no producer warpgroup, is
    refused (`Family.gathers_rows`)."""
    with generation.assumed("sm90"):
        a_gathered_product_is_on_par(interpret=True)
        a_gathered_product_is_on_par(interpret=True, paired=True, scattered=True)
        a_deep_gathered_step_reads_its_rows_as_their_copy_would(grouped_matmul.INTERPRETED)
    tokens, table = jnp.zeros((200, 128), jnp.bfloat16), jnp.zeros((256,), jnp.int32)
    with generation.assumed("sm80"), pytest.raises(ValueError):
        mosaic_gpu.gmm(tokens, jnp.zeros((4, 128, 256), jnp.bfloat16),
                       jnp.asarray([70, 0, 129, 57], jnp.int32),
                       tiles=mosaic_gpu.Tiles(128, 128, 32, 3, 1, family="mma.sync"), transpose_rhs=False,
                       out_dtype=jnp.float32, multiprocessors=4, interpret=grouped_matmul.INTERPRETED,
                       gather=table)


def a_deep_gathered_step_reads_its_rows_as_their_copy_would(interpret):
    """A gathered product's 256-deep step, four swizzled tiles a row (its rows' copies, `rings.piece_copies`,
    the load barrier's arrivals: counted apart, the packed NVFP4 bank's gathered walk hung): bitwise the same
    tile's product of the rows gathered first, its groups ragged, one empty, rows past them."""
    tiles = mosaic_gpu.Tiles(128, 128, 256, 2, 1, warpgroups=2, family='wgmma')
    sizes = jnp.asarray([70, 0, 129, 51], jnp.int32)
    rng = np.random.default_rng(7)
    tokens = jnp.asarray(rng.integers(-3, 4, size=(200, 512)), jnp.bfloat16)
    table = rng.permutation(np.concatenate([np.arange(150), rng.integers(0, 200, 106)]))
    table = jnp.asarray(table, jnp.int32)
    matrices = jnp.asarray(rng.integers(-3, 4, size=(4, 512, 256)), jnp.bfloat16)

    def product(rows, gather):
        return mosaic_gpu.gmm(rows, matrices, sizes, tiles=tiles, transpose_rhs=False, out_dtype=jnp.float32,
                              multiprocessors=4, interpret=interpret, gather=gather)

    np.testing.assert_array_equal(np.asarray(product(tokens, table)),
                                  np.asarray(product(tokens[table], None)))


def a_gathered_product_is_on_par(interpret: bool = False, *, paired: bool = False, scattered: bool = False):
    """A grouped product whose rows are gathered from the tokens by a table (`gmm`'s `gather`: a MoE's sort
    fused into the rows' loads, row 1376), its groups ragged, one empty, rows past them, a token read by
    several slots and some by none, through the template named at a pinned cooperative tile (`gemm.gemm`,
    `grouped_matmul` chosen `mosaic`, `tuning.pinned`: the product the call's precision declares, its sums
    promoted where it asks for them), on par with the arm Dew's selector runs for the same call where the
    template is not taken (check 1: `grouped_matmul` chosen `xla`, the gathered rows' ragged dot), from
    float64: on this GPU (test_gpu_gemm's cell, the producer's cp.async gather), or interpreted (`interpret`);
    `paired`, a second matrix's product beside it; `scattered`, each row stored at its entry of a permutation
    (`gmm`'s `scatter`: the MoE's un-permute fused into the stores). Called through `mosaic_gpu.gmm` itself,
    the tile summed its products in the tensor cores' running sum whatever the precision asked: 1.658x the
    reference at the suite's highest precision on a GH200 (M-EM-19), as summed through gemm.gemm its error is
    the reference's."""
    tiles = mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')
    sizes, depth = jnp.asarray([70, 0, 129, 51], jnp.int32), 128

    def draw(seed):
        rng = np.random.default_rng(seed)
        tokens = jnp.asarray(rng.normal(size=(200, depth)), jnp.bfloat16)
        table = jnp.asarray(rng.permutation(np.concatenate([np.arange(150), rng.integers(0, 200, 106)])),
                            jnp.int32)
        matrices = jnp.asarray(rng.normal(size=(2, 4, depth, 256)) / np.sqrt(depth), jnp.bfloat16)
        slots = jnp.asarray(rng.permutation(256), jnp.int32)
        return tokens, table, matrices, slots

    def through(kernel):
        def run(tokens, table, matrices, slots):
            # The call through Dew's selector (`grouped_matmul` chosen `kernel`): the template's at the pinned
            # tile, or XLA's ragged dot of the gathered rows as they are (bf16 products summed into fp32), its
            # rows stored at their slots; the float64 truth XLA's over the gate's widened draws.
            wide = jnp.promote_types(tokens.dtype, jnp.float32)
            with generation.chosen(grouped_matmul=kernel), tuning.pinned({grouped_matmul.MosaicGmm: tiles}):
                outs = gemm.gemm(tokens, matrices[0], sizes, out_dtype=wide,
                                 out_dtypes=(wide,) * 2 if paired else None,
                                 paired=matrices[1] if paired else None, gather=table,
                                 scatter=slots if scattered else None, interpret_on_cpu=interpret)
            return jnp.stack(outs) if paired else outs
        return run

    with tuning.seen() as launches:
        assert_as_exact(jax.jit(through("mosaic")), jax.jit(through("xla")), draw,
                        "a gathered grouped product", seeds=SEEDS, truth_fn=through("xla"))
    taken = [config for (op, _), (config, _) in zip(launches, launches.taken, strict=True)
             if type(op) is grouped_matmul.MosaicGmm]
    assert taken and all(config == tiles for config in taken), taken


@pytest.mark.parametrize("paired", [False, True])
def test_a_gathered_product_lowers_for_cuda(paired):
    """A gathered grouped product on a compile-only H100 lowers through Mosaic GPU: its producer's
    cp.async gather and its load barriers' lane-level arrivals and parity waits (`gathered.Ring`), plain and
    paired, the paired one's rows stored at a permutation's entries (`gathered.scattered_rows`)."""
    tiles = mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')
    sizes = jnp.asarray([700, 0, 1290, 58], jnp.int32)
    product = functools.partial(mosaic_gpu.gmm, tiles=tiles, transpose_rhs=False, out_dtype=jnp.bfloat16,
                                out_dtypes=(jnp.bfloat16,) * 2 if paired else None, multiprocessors=132,
                                interpret=False)

    def gathered(tokens, matrices, table):
        return product(tokens, matrices[0], sizes, gather=table, paired=matrices[1] if paired else None,
                       scatter=table if paired else None)

    with generation.assumed("sm90"):
        jax.jit(gathered).trace(jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                                jax.ShapeDtypeStruct((2, 4, 512, 256), jnp.bfloat16),
                                jax.ShapeDtypeStruct((2048,), jnp.int32)).lower(lowering_platforms=("cuda",))


@pytest.mark.parametrize("splits", [1, 2])
def test_a_gathered_kernel_gradient_reads_each_operands_rows_where_its_table_says(splits):
    """`a_gathered_kernel_gradient_is_its_gathered_rows_gradient` on an H100, interpreted; a gathered operand
    beside a prologue on the rows, and on mma.sync (no producer warpgroup), refused."""
    with generation.assumed("sm90"):
        tokens, cotangent, table = a_gathered_kernel_gradient_is_its_gathered_rows_gradient(
            grouped_matmul.INTERPRETED, splits)
        with pytest.raises(ValueError):
            mosaic_gpu.tgmm(tokens, cotangent, jnp.asarray([70, 0, 129, 51], jnp.int32),
                            tiles=mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma'),
                            out_dtype=jnp.float32, multiprocessors=4, interpret=grouped_matmul.INTERPRETED,
                            gather=(table, None), prologue=lambda rows: rows)
    with generation.assumed("sm80"), pytest.raises(ValueError):
        mosaic_gpu.tgmm(tokens, cotangent, jnp.asarray([70, 0, 129, 51], jnp.int32),
                        tiles=mosaic_gpu.Tiles(128, 128, 32, 3, 1, family="mma.sync"), out_dtype=jnp.float32,
                        multiprocessors=4, interpret=grouped_matmul.INTERPRETED, gather=(table, None))


def a_gathered_kernel_gradient_is_its_gathered_rows_gradient(interpret=False, splits: int = 1) -> tuple:
    """A kernel gradient's gathered operands (`mosaic_gpu.tgmm`'s `gather`, row 1376: the tokens a mixture of
    experts' slots read, and a scattered product's cotangent at its slots), bitwise the same tile's gradient
    of the rows gathered first, the tokens' rows alone and both operands', its groups ragged, one empty, rows
    past them, its depth split or not, at 257 rows (a table padded to whole steps, an ungathered operand to
    8 rows). Integers, every sum exact. Its tokens, cotangent and table."""
    tiles = mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma', splits=splits)
    sizes = jnp.asarray([70, 0, 129, 51], jnp.int32)
    rng = np.random.default_rng(11)
    tokens = jnp.asarray(rng.integers(-3, 4, size=(200, 256)), jnp.bfloat16)
    table = rng.permutation(np.concatenate([np.arange(150), rng.integers(0, 200, 107)]))
    table = jnp.asarray(table, jnp.int32)
    cotangent = jnp.asarray(rng.integers(-3, 4, size=(257, 128)), jnp.bfloat16)
    slots = jnp.asarray(rng.permutation(257), jnp.int32)
    gradient = jax.jit(functools.partial(mosaic_gpu.tgmm, group_sizes=sizes, tiles=tiles,
                                         out_dtype=jnp.float32, multiprocessors=4 if interpret else 132,
                                         interpret=interpret))
    for tables, rows, columns in (((table, None), tokens[table], cotangent),
                                  ((table, slots), tokens[table], cotangent[slots])):
        np.testing.assert_array_equal(np.asarray(gradient(tokens, cotangent, gather=tables)),
                                      np.asarray(gradient(rows, columns)))
    return tokens, cotangent, table


def test_a_gathered_kernel_gradients_key_meets_its_table_and_rows_at_one_count():
    """A gathered kernel gradient's key (`_GroupedLaunch.keyed`, the call a tuning run materializes and
    times) buckets its table as it buckets its ungathered operand's rows, 500 to 512 each, so the launch it
    measures takes its own operands; the gathered tokens keep a count of their own."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    with generation.assumed("sm90"):
        launch = grouped_matmul.MosaicTgmm()
        table = jax.ShapeDtypeStruct((500,), jnp.int32)
        ba = launch.bind(bf16((100, 256)), bf16((500, 128)), jax.ShapeDtypeStruct((4,), jnp.int32), (),
                         (table, None), out_dtype="float32", interpret=False)
        keyed = launch.keyed(ba).arguments
    assert keyed["gather"][0].shape == (512,) and keyed["gather"][1] is None
    assert keyed["cotangent"].shape[0] == 512 and keyed["tokens"].shape[0] == 128


def test_a_gathered_kernel_gradient_lowers_for_cuda():
    """A kernel gradient of gathered tokens and a scattered product's cotangent on a compile-only H100
    lowers through Mosaic GPU: each operand's ring its producer's cp.async gather (`gathered.Ring`, a step's
    rows by the gradient's depth or width)."""
    tiles = mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')
    sizes = jnp.asarray([700, 0, 1290, 58], jnp.int32)

    def gathered(tokens, cotangent, table, slots):
        return mosaic_gpu.tgmm(tokens, cotangent, sizes, tiles=tiles, out_dtype=jnp.float32,
                               multiprocessors=132, interpret=False, gather=(table, slots))

    with generation.assumed("sm90"):
        jax.jit(gathered).trace(jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                                jax.ShapeDtypeStruct((2048, 256), jnp.bfloat16),
                                *(jax.ShapeDtypeStruct((2048,), jnp.int32),) * 2).lower(
                                    lowering_platforms=("cuda",))


def test_a_blackwells_gathered_product_reads_its_rows_by_its_load_warps_gathers():
    """`a_blackwells_gathered_product_is_its_gathered_rows_product` on an sm100, interpreted (each row a copy
    where the GPU's load warp gathers four)."""
    with generation.assumed("sm100"):
        a_blackwells_gathered_product_is_its_gathered_rows_product(interpret_on_cpu=True)


def a_blackwells_gathered_product_is_its_gathered_rows_product(interpret_on_cpu: bool = False):
    """A gathered, scattered grouped product on a Blackwell GPU (row 1376): tcgen05's bundle takes it (its
    load warp's TMA four-row gathers, `gathered.Committed`), its value bitwise the product of the rows
    gathered first and put at their slots after; and its kernel gradient's tables on the template too (row
    4852: mma.sync's consumer-fed rings, their loading consumer gathering each step's rows, `gathered.Ring`),
    bitwise the gradient of the rows gathered first. Integers, every sum exact."""

    rng = np.random.default_rng(13)
    tokens = jnp.asarray(rng.integers(-1, 2, size=(200, 256)), jnp.bfloat16)
    matrices = jnp.asarray(rng.integers(-1, 2, size=(3, 256, 256)), jnp.bfloat16)
    table = jnp.asarray(rng.integers(0, 200, 384), jnp.int32)
    slots = jnp.asarray(rng.permutation(384), jnp.int32)
    cotangent = jnp.asarray(rng.integers(-1, 2, size=(384, 256)), jnp.bfloat16)
    sizes = jnp.asarray([128, 0, 200], jnp.int32)
    options = {"out_dtype": jnp.bfloat16, "precision": "default", "interpret_on_cpu": interpret_on_cpu}
    # The template's launch named, its product at tcgen05's tile (`tuning.pinned`, a pin its legality checks).
    tile = mosaic_gpu.Tiles(128, 128, 64, 4, 1, family='tcgen05')
    with generation.chosen(grouped_matmul="mosaic"), tuning.pinned({grouped_matmul.MosaicGmm: tile}):
        got = jax.jit(lambda: gemm.gemm(tokens, matrices, sizes, gather=table, scatter=slots, **options))()
        apart = jax.jit(lambda: gemm.gemm(tokens[table], matrices, sizes, **options))()
    np.testing.assert_array_equal(np.asarray(got), np.asarray(jnp.zeros_like(apart).at[slots].set(apart)))
    gradient = {**options, "out_dtype": jnp.float32}
    with generation.chosen(grouped_matmul="mosaic"):
        got = jax.jit(lambda: gemm.gemm_t(tokens, cotangent, sizes, gather=(table, slots), **gradient))()
    expected = jax.jit(lambda: gemm.gemm_t(tokens[table], cotangent[slots], sizes, **gradient))()
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_a_blackwells_gathered_product_lowers_for_cuda():
    """A gathered grouped product on a compile-only B200 lowers through Mosaic GPU on tcgen05's bundle: its
    load warp's TMA four-row gathers into the swizzled ring (`gathered.tma_rows`), each slot released by its
    products' commit; and its kernel gradient's tables at the tile its launch derives for a wide and a
    32-column cotangent, one its launch takes (`legal`), on whichever family derives it."""
    tiles = mosaic_gpu.Tiles(128, 128, 64, 4, 1, family='tcgen05')
    sizes = jnp.asarray([700, 0, 1290, 58], jnp.int32)

    def gathered(tokens, matrices, table):
        return mosaic_gpu.gmm(tokens, matrices, sizes, tiles=tiles, transpose_rhs=False,
                              out_dtype=jnp.bfloat16, multiprocessors=148, interpret=False, gather=table)

    def gradient(tokens, cotangent, table, slots):
        bound = launch.bind(*(jax.ShapeDtypeStruct(each.shape, each.dtype) for each in (tokens, cotangent)),
                            jax.ShapeDtypeStruct(sizes.shape, sizes.dtype), (), (table, slots),
                            out_dtype="float32", interpret=False)
        tile = launch.derived(bound, generation.gpu_target())
        assert tile is not None and launch.legal(bound, tile, generation.gpu_target()), tile
        return mosaic_gpu.tgmm(tokens, cotangent, sizes, tiles=tile, out_dtype=jnp.float32,
                               multiprocessors=148, interpret=False, gather=(table, slots))

    launch = grouped_matmul.MosaicTgmm()

    with generation.assumed("sm100"):
        jax.jit(gathered).trace(jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                                jax.ShapeDtypeStruct((4, 512, 256), jnp.bfloat16),
                                jax.ShapeDtypeStruct((2048,), jnp.int32)).lower(lowering_platforms=("cuda",))
        # A narrow cotangent's stage too, its tile the launch's own, staged at the swizzle its row holds
        # (`gathered.swizzled`).
        for width in (256, 32):
            operands = (jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                        jax.ShapeDtypeStruct((2048, width), jnp.bfloat16),
                        *(jax.ShapeDtypeStruct((2048,), jnp.int32),) * 2)
            jax.jit(gradient).trace(*operands).lower(lowering_platforms=("cuda",))


@pytest.mark.parametrize("format_", ["mxfp8", "nvfp4"])
def test_a_blackwells_gathered_block_scaled_product_gathers_its_scales_by_its_table(format_):
    """A gathered block-scaled product on an sm100, interpreted (row 4851: `mxfp8_product`'s and
    `nvfp4_block_product`'s `gather`): its codes read where they lie by tcgen05's gather and its rows' block
    scales by the same table, bitwise the product of the operand gathered first (codes and scales), its
    groups whole 128-row blocks, one empty."""

    choice, form, product = {"mxfp8": ("mxfp8", formats.MxFp8, gemm.mxfp8_product),
                             "nvfp4": ("nvfp4_block", formats.NvFp4Block, gemm.nvfp4_block_product)}[format_]
    rng = np.random.default_rng(17)
    tokens = form.quantized(jnp.asarray(rng.normal(size=(300, 256)), jnp.float32))
    matrices = form.quantized(jnp.asarray(rng.normal(size=(4, 128, 256)), jnp.float32))
    blocks = jnp.asarray([1, 0, 2, 1], jnp.int32)
    table = jnp.asarray(rng.integers(0, 300, 512), jnp.int32)
    gathered = dataclasses.replace(tokens, codes=tokens.codes[table], scales=tokens.scales[table])
    named = generation.chosen(**{choice: "mosaic"})
    with generation.assumed("sm100"), named, generation.routes() as routes:
        got = jax.jit(lambda: product(tokens, matrices, blocks, out_dtype=jnp.float32, interpret_on_cpu=True,
                                      gather=table))()
        expected = jax.jit(lambda: product(gathered, matrices, blocks, out_dtype=jnp.float32,
                                           interpret_on_cpu=True))()
    assert {route.kernel for route in routes if route.choice == choice} == {"mosaic"}, routes
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_a_gathered_products_table_covers_every_tile_its_schedule_walks():
    """A gathered product's table, on a family whose loads past the rows are bounded (wgmma's TMA, compiled),
    still extends to every row its schedule's tiles walk (`mosaic_gpu.row_extent`): the producer's threads
    read a tile's entries by plain loads, which no TMA bounds; read past it, a GH200 faulted
    (CUDA_ERROR_ILLEGAL_ADDRESS, FS-4346-base-sm90)."""
    tiles = mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')
    table = jnp.zeros((256,), jnp.int32)
    tokens = jnp.zeros((200, 128), jnp.bfloat16)
    _, gathered, _, m, extent = mosaic_gpu._walked_rows(tokens, table, None, tiles, transpose_lhs=False,
                                                        dense=False, bounded=True)
    assert gathered.shape[0] == extent == mosaic_gpu.row_extent(m, tiles.block_m) > m


def test_a_grouped_products_depth_split_sums_as_its_walk_apart():
    """A grouped product's depth split (row 3622's lever: a decode step's experts' one wave of tiles, each
    run a share of the depth, the fp32 partials summed): over small integers, bitwise the unsplit product,
    its groups ragged with one empty and rows past them zero, interpreted as an H100's."""
    rng = np.random.default_rng(31)
    tokens = jnp.asarray(rng.integers(-3, 4, size=(40, 512)), jnp.bfloat16)
    matrices = jnp.asarray(rng.integers(-3, 4, size=(4, 512, 256)), jnp.bfloat16)
    sizes = jnp.asarray([10, 0, 17, 9], jnp.int32)
    tiles = mosaic_gpu.Tiles(64, 128, 64, 2, 1, warpgroups=1, family='wgmma')

    def product(splits):
        return mosaic_gpu.gmm(tokens, matrices, sizes, tiles=dataclasses.replace(tiles, splits=splits),
                              transpose_rhs=False, out_dtype=jnp.float32, multiprocessors=4,
                              interpret=grouped_matmul.INTERPRETED)

    with generation.assumed("sm90"):
        np.testing.assert_array_equal(np.asarray(product(4)), np.asarray(product(1)))


def test_a_hoppers_moe_decode_splits_its_experts_depth():
    """A MoE decode step on an H100 (Qwen3-30B-A3B's down projection: 768 deep, 2048 wide, eight of 128
    experts active a token, one token's slots, row 3622): its eight experts' 64 tiles leave half the wave's
    multiprocessors idle, so the derived tile splits its depth over them (`MosaicGmm._decode_split`),
    legally; a prefill's rows keep it unsplit."""
    launch = grouped_matmul.MosaicGmm()

    def derived(rows):
        bound = launch.bind(jax.ShapeDtypeStruct((rows, 768), jnp.bfloat16),
                            jax.ShapeDtypeStruct((128, 768, 2048), jnp.bfloat16),
                            jax.ShapeDtypeStruct((128,), jnp.int32), out_dtype="bfloat16", trans_rhs=False,
                            interpret=False)
        target = generation.gpu_target()
        tile = launch.derived(bound, target)
        assert tile is not None and launch.legal(bound, tile, target), tile
        return tile

    with generation.assumed("sm90"):
        assert derived(8).splits > 1 and derived(8192).splits == 1


def test_a_kernel_gradient_no_launch_reads_through_its_tables_runs_a_launch_on_its_gathered_rows():
    """fp32 rows and cotangent read through tables (an fp32 mixture of experts' gathered tokens and scattered
    slots), which no launch reads where they lie (Mosaic GPU's gathering producer takes 16-bit rows): on an
    H100 the kernel gradient gathers them first and a grouped launch takes the product, as the forward's
    does, never XLA's ragged dot over every group's matrix, whose footprint an MoE step's derived plan bounds
    from below alone (row 5213); interpreted, bitwise the kernel gradient of the gathered rows."""
    rng = np.random.default_rng(5)
    tokens = jnp.asarray(rng.standard_normal((200, 128)), jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal((256, 64)), jnp.float32)
    table = jnp.asarray(rng.integers(0, 200, 256), jnp.int32)
    slots = jnp.asarray(rng.permutation(256), jnp.int32)
    sizes = jnp.asarray([100, 0, 156, 0], jnp.int32)
    gradient = {"out_dtype": jnp.float32, "precision": "default", "interpret_on_cpu": True}
    with generation.assumed("sm90"):
        def tabled():
            return gemm.gemm_t(tokens, cotangent, sizes, gather=(table, slots), **gradient)

        graph = jax.make_jaxpr(tabled)()
        got = jax.jit(tabled)()
        expected = jax.jit(lambda: gemm.gemm_t(tokens[table], cotangent[slots], sizes, **gradient))()
    names = {equation.primitive.name for equation in equations(graph, platform="cuda")}
    assert "ragged_dot_general" not in names, names
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
