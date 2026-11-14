"""The GEMM template's kernel gradient (`mosaic_gpu.tgmm`, dW = x^T dy) on each instruction family,
interpreted on the CPU: a dense call's stages read as loaded, and an exactly summed gradient's depth split."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS

from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.templates import mosaic_sm90
from dew.nn.kernels.templates.gemm import mosaic_gpu

pytestmark = pytest.mark.kernel

DENSE_GRADIENT_TILES = {
    "wgmma, four stages": mosaic_gpu.Tiles(128, 128, 64, 4, 1, warpgroups=2, family='wgmma'),
    "wgmma, three stages": mosaic_gpu.Tiles(128, 128, 64, 3, 1, warpgroups=2, family='wgmma'),
    "wgmma fed": mosaic_gpu.Tiles(128, 128, 64, 3, 1, warpgroups=2, family=mosaic_sm90.FED.name),
    "mma.sync": mosaic_gpu.Tiles(128, 64, 32, 3, 1, family='mma.sync'),
    "tcgen05": mosaic_gpu.Tiles(128, 128, 64, 4, 1, family='tcgen05'),
}
"""A dense kernel gradient's tiles on each family: wgmma's rings of four and three stages, walked a rotation a
trip into bf16 and a step a trip exactly summed; its consumers feeding their own rings; mma.sync's, whose
128-deep windows a branch closes; tcgen05's, whose persist sums the windows itself."""

GENERATION_OF = {"wgmma": "sm90", "wgmma.fed": "sm90", "mma.sync": "sm80", "tcgen05": "sm100"}


@INTERPRETS
@pytest.mark.parametrize("out_dtype", [jnp.bfloat16, jnp.float32], ids=["bf16", "fp32, exact sums"])
@pytest.mark.parametrize(("rows", "depth"), [(200, 256), (520, 192)])
@pytest.mark.parametrize("name", list(DENSE_GRADIENT_TILES))
def test_a_dense_kernel_gradient_reads_its_stages_as_loaded(name, rows, depth, out_dtype):
    """A dense kernel gradient (`mosaic_gpu.tgmm`'s `dense`, no prologue) reads each step's stages as they
    were loaded, no consumer or role zeroing a neighbour's rows (`mosaic_gpu.rows_unstaged`): over integers
    from -1 to 1, every product and sum exact, bitwise numpy's on each family's tile, its rows a whole
    number of steps, rows no step divides (the rows past them zeros, read padded where an interpreted load
    past an operand clamps), and a depth no tile's rows divide (its last row tile partial on a TMA family,
    `mosaic_gpu.ragged_rows`, padded elsewhere); summed by the tensor cores into bf16 and exactly into fp32,
    each window's partial added into the running sum once."""
    tiles = DENSE_GRADIENT_TILES[name]
    width = 256
    keys = jax.random.split(jax.random.key(7), 2)
    x = jax.random.randint(keys[0], (rows, depth), -1, 2).astype(jnp.bfloat16)
    dy = jax.random.randint(keys[1], (rows, width), -1, 2).astype(jnp.bfloat16)
    expected = np.asarray(x, np.float32).T @ np.asarray(dy, np.float32)
    with generation.assumed(GENERATION_OF[tiles.family]):
        got = mosaic_gpu.tgmm(x, dy, jnp.asarray([rows], jnp.int32), tiles=tiles, out_dtype=out_dtype,
                              multiprocessors=8, interpret=grouped_matmul.INTERPRETED, dense=True,
                              exact_sums=bool(jnp.dtype(out_dtype) == jnp.float32))
    np.testing.assert_array_equal(np.asarray(got[0], np.float32), expected)


@INTERPRETS
@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_an_exact_kernel_gradients_depth_split_sums_its_partials_once(generation_name):
    """An exactly summed kernel gradient (fp32 out of 16-bit products, `templates.product.promoted`) whose
    contraction splits (`Tiles.splits`) at the tile its call derives: a split its launch takes
    (`MosaicTgmm.legal`), each run summed exactly and the runs' fp32 partials added by IEEE adds, over small
    integers every product and sum exact, bitwise the unsplit launch's at two and three runs (the last
    shorter), a dense call's and a grouped one's (groups of rows no step divides, one empty)."""
    rows, depth, width = 456, 128, 256
    keys = jax.random.split(jax.random.key(8), 2)
    x = jax.random.randint(keys[0], (rows, depth), -3, 4).astype(jnp.bfloat16)
    dy = jax.random.randint(keys[1], (rows, width), -3, 4).astype(jnp.bfloat16)
    gradient = grouped_matmul.MosaicTgmm()
    spec = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    for sizes, dense in (([rows], True), ([200, 0, 256], False)):
        group_sizes = jnp.asarray(sizes, jnp.int32)
        with generation.assumed(generation_name):
            target = generation.gpu_target()
            call = gradient.bind(spec((rows, depth)), spec((rows, width)),
                                 jax.ShapeDtypeStruct(group_sizes.shape, jnp.int32), interpret=False,
                                 out_dtype="float32", dense=dense, exact_sums=True)
            tile = gradient.derived(call, target)

            def dew(splits, group_sizes=group_sizes, dense=dense, tile=tile):
                return mosaic_gpu.tgmm(x, dy, group_sizes, tiles=dataclasses.replace(tile, splits=splits),
                                       out_dtype=jnp.float32, multiprocessors=8,
                                       interpret=grouped_matmul.INTERPRETED, dense=dense, exact_sums=True)

            whole = np.asarray(dew(1))
            for splits in (2, 3):
                assert gradient.legal(call, dataclasses.replace(tile, splits=splits), target), (tile, splits)
                np.testing.assert_array_equal(np.asarray(dew(splits)), whole, err_msg=f"{sizes} at {splits}")


@INTERPRETS
def test_a_split_kernel_gradient_reads_each_groups_side_operand_at_its_own_group():
    """A kernel gradient split in two (`Tiles.splits`), each group's rows two runs of their own groups
    (`mosaic_gpu._sliced`), whose prologue scales each group's rows by its group's own vector (`('groups',
    'depth')`, an expert's input scale): each run reads its own group's vector (it read the run's index
    modulo the groups, so a group's second run took the next group's), and over ones the gradient is each
    group's rows times its scale, 128 and 256, as unsplit."""
    rows, depth, width = 256, 64, 128
    x, dy = jnp.ones((rows, depth), jnp.bfloat16), jnp.ones((rows, width), jnp.bfloat16)
    scales = jnp.asarray([[1.0] * depth, [2.0] * depth], jnp.float32)
    tiles = mosaic_gpu.Tiles(64, 128, 64, 2, 1, family='wgmma')

    def scaled(stage, scale):
        return (stage * scale).astype(stage.dtype)

    def dew(splits):
        return mosaic_gpu.tgmm(x, dy, jnp.asarray([128, 128], jnp.int32),
                               tiles=dataclasses.replace(tiles, splits=splits), out_dtype=jnp.float32,
                               multiprocessors=8, interpret=grouped_matmul.INTERPRETED, prologue=scaled,
                               prologue_operands=(scales,), prologue_axes=(("groups", "depth"),),
                               exact_sums=True)

    expected = np.broadcast_to(np.asarray([128.0, 256.0])[:, None, None], (2, depth, width))
    with generation.assumed("sm90"):
        for splits in (1, 2):
            np.testing.assert_array_equal(np.asarray(dew(splits)), expected, err_msg=f"at {splits} runs")
