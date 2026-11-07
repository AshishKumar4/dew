"""The GEMM template's gathered rows (row 1376): a grouped product whose rows are read from the tokens where a
table says, a mixture of experts' sort fused into the rows' loads (`mosaic_gpu.gmm`'s `gather`)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.templates.gemm import mosaic_gpu

SEEDS = 8
"""The gathered product's check-1 seeds, the template's grouped products' count."""


def test_a_gathered_grouped_product_reads_each_row_where_its_table_says():
    """`a_gathered_product_is_on_par` on an H100, interpreted (each row a copy at its slot's swizzled row,
    `Ring.parts`), plain and paired; a gathered product on mma.sync, which has no producer warpgroup, is
    refused (`Family.gathers_rows`)."""
    with generation.assumed("sm90"):
        a_gathered_product_is_on_par(grouped_matmul.INTERPRETED)
        a_gathered_product_is_on_par(grouped_matmul.INTERPRETED, paired=True, scattered=True)
    tokens, table = jnp.zeros((200, 128), jnp.bfloat16), jnp.zeros((256,), jnp.int32)
    with generation.assumed("sm80"), pytest.raises(ValueError):
        mosaic_gpu.gmm(tokens, jnp.zeros((4, 128, 256), jnp.bfloat16),
                       jnp.asarray([70, 0, 129, 57], jnp.int32),
                       tiles=mosaic_gpu.Tiles(128, 128, 32, 3, 1, family="mma.sync"), transpose_rhs=False,
                       out_dtype=jnp.float32, multiprocessors=4, interpret=grouped_matmul.INTERPRETED,
                       gather=table)


def a_gathered_product_is_on_par(interpret=False, *, paired: bool = False, scattered: bool = False):
    """A grouped product whose rows are gathered from the tokens by a table (`gmm`'s `gather`: a MoE's sort
    fused into the rows' loads, row 1376), its groups ragged, one empty, rows past them, a token read by
    several slots and some by none, on par with XLA's ragged dot of the gathered rows (check 1), from
    float64: on this GPU (test_gpu_gemm's cell, the producer's cp.async gather), or `interpret`ed; `paired`,
    a second matrix's product beside it; `scattered`, each row stored at its entry of a permutation (`gmm`'s
    `scatter`: the MoE's un-permute fused into the stores)."""
    tiles = mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')
    sizes = jnp.asarray([70, 0, 129, 51], jnp.int32)

    def draw(seed):
        rng = np.random.default_rng(seed)
        tokens = jnp.asarray(rng.normal(size=(200, 128)), jnp.bfloat16)
        table = jnp.asarray(rng.permutation(np.concatenate([np.arange(150), rng.integers(0, 200, 106)])),
                            jnp.int32)
        matrices = jnp.asarray(rng.normal(size=(2, 4, 128, 256)) / np.sqrt(128), jnp.bfloat16)
        slots = jnp.asarray(rng.permutation(256), jnp.int32)
        return tokens, table, matrices, slots

    def dew(tokens, table, matrices, slots):
        outs = mosaic_gpu.gmm(tokens, matrices[0], sizes, tiles=tiles, transpose_rhs=False,
                              out_dtype=jnp.float32, out_dtypes=(jnp.float32,) * 2 if paired else None,
                              multiprocessors=4, interpret=interpret, gather=table,
                              paired=matrices[1] if paired else None, scatter=slots if scattered else None)
        return jnp.stack(outs) if paired else outs

    def plain(tokens, table, matrices, slots, wide=jnp.float32):
        rows = tokens[table].astype(wide)
        outs = [jax.lax.ragged_dot(rows, matrix.astype(wide), sizes, preferred_element_type=wide)
                for matrix in matrices[:2 if paired else 1]]
        outs = [jnp.zeros_like(out).at[slots].set(out) if scattered else out for out in outs]
        return jnp.stack(outs) if paired else outs[0]

    assert_as_exact(jax.jit(dew), jax.jit(plain), draw, "a gathered grouped product", seeds=SEEDS,
                    truth_fn=functools.partial(plain, wide=jnp.float64))


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
