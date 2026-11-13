"""The GEMM template on widths and depths no tile divides: on a TMA family, a product whose operands' rows
are whole 16 bytes runs its last step and column tile partial (`mosaic_gpu.partial_runs`, row 5299), its
loads past the call zero-filled and its stores past it dropped; elsewhere the call runs padded with zeros to
whole tiles of its family and its outputs sliced back (`grouped_matmul._Mosaic._aligned`), as Triton's
masked tiles ran it."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS

from dew.nn.kernels import gemm, generation, grouped_matmul
from dew.nn.kernels.templates.gemm import mosaic_gpu, pricing

WIDTHS = [(96, 100), (200, 72)]
"""Depths and widths no tile of any family divides: on a TMA family 96 x 100's depth runs partial and its
width padded (a 100-wide bf16 row is 200 bytes), 200 x 72's both partial (rows of 400 and 144 bytes)."""


@pytest.mark.parametrize("generation_name", ["sm90", "sm100"])
def test_a_tma_family_runs_a_partial_step_and_column_tile_where_its_rows_are_whole_16_bytes(generation_name):
    """A grouped bf16 product compiled for a TMA family runs a depth and a width no tile divides as they are
    (`grouped_matmul._Mosaic._aligned` copies nothing) where its operands' and outputs' rows along them are
    whole 16 bytes (1000 x 1400: 2000 and 2800 bytes) and the tile it derives on the padded call runs them
    partial unchanged: a matrix lying `[width, depth]` its width where the tile is no wider than it, its
    depth where the tile's step is one swizzled tile; at that same tile, so no narrower tile pays for the
    copy saved (200 x 136's padded 192-wide tile pads its width). Its width is
    padded where its rows are not whole 16 bytes (100 wide: 200-byte rows), its depth where its rows are
    gathered (a producer's cp.async gather is unbounded), and both on a family whose loads are not bounded
    (an A100's cp.async); and its launch, with a prologue over the depth and an epilogue over the columns,
    lowers for CUDA at its derived tile."""
    launch = grouped_matmul.MosaicGmm()
    sizes = jax.ShapeDtypeStruct((4,), jnp.int32)
    step = mosaic_gpu.partial_step(jnp.bfloat16)

    def ran(depth: int, width: int, *, name=generation_name, gather=None) -> tuple:
        bf16 = jax.ShapeDtypeStruct
        with generation.assumed(name):
            ba = launch.bind(bf16((4096, depth), jnp.bfloat16), bf16((4, width, depth), jnp.bfloat16), sizes,
                             *(() if gather is None else ((), (), None, None, gather)),
                             out_dtype="float32", trans_rhs=True, interpret=False)
            target = generation.gpu_target()
            width_ran, depth_ran = launch._aligned(ba).arguments['kernel'].shape[1:]
            padded = launch.derived(launch._aligned(ba, whole_tiles=True), target)
            return (depth_ran, width_ran), launch.derived(ba, target), padded

    for depth, width in ((200, 136), (1000, 1400)):
        (depth_ran, width_ran), tile, padded = ran(depth, width)
        assert tile == padded and tile is not None
        family = pricing.family_of(tile, jnp.bfloat16)
        # A pair's tile (a B200's, `paired_first`) is padded yet: each CTA's copy of its half of the pair's
        # block is not partitioned under a partial extent until the fork's partitioned untiling (row 5727).
        alone = not family.pairs_product
        assert (width_ran == width) is (alone and pricing.tile_columns(tile, family) <= width)
        assert (depth_ran == depth) is (alone and tile.block_k == step)
    assert ran(1000, 1400)[0] == ((1000, 1400) if generation_name == "sm90" else (1024, 1408))
    assert ran(200, 100)[0][1] == 128
    assert ran(200, 136, gather=jax.ShapeDtypeStruct((4096,), jnp.int32))[0][0] > 200
    assert ran(200, 136, name="sm80")[0] != (200, 136)

    def normed(x, weight):
        return (x.astype(jnp.float32) * weight).astype(x.dtype)

    def biased(acc, bias):
        return acc + bias

    def product(x, w, weight, bias):
        return gemm.gemm(x, w, jnp.asarray([1024] * 4, jnp.int32), transpose_rhs=True, out_dtype=jnp.float32,
                         prologue=normed, prologue_operands=((weight, ("depth",)),),
                         epilogue=biased, epilogue_operands=((bias, ("columns",)),))

    specs = (jax.ShapeDtypeStruct((4096, 200), jnp.bfloat16),
             jax.ShapeDtypeStruct((4, 136, 200), jnp.bfloat16),
             jax.ShapeDtypeStruct((200,), jnp.float32), jax.ShapeDtypeStruct((136,), jnp.float32))
    with (generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"),
          generation.routes() as routes):
        jax.jit(product).trace(*specs).lower(lowering_platforms=("cuda",))
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}


@INTERPRETS
@pytest.mark.parametrize("generation_name", ["sm80", "sm90", "v5e"])
@pytest.mark.parametrize(("depth", "width"), WIDTHS, ids=["96x100", "200x72"])
def test_a_product_of_unaligned_widths_runs_on_the_template(generation_name, depth, width):
    """`unaligned_widths_run_on_the_template`, interpreted on the CPU as each family."""
    with generation.assumed(generation_name):
        unaligned_widths_run_on_the_template(depth, width, interpret=True)


def unaligned_widths_run_on_the_template(depth: int, width: int, *, interpret: bool) -> None:
    """A grouped forward, its input gradient and its kernel gradient, and a dense forward, at a depth and a
    width no tile of the family divides, the template's launch named (`'mosaic'`): over integers from -1 to
    1, every product and sum exact, bitwise numpy's, groups of rows no tile divides and one empty; on a GPU
    itself too (tests/test_gpu_gemm.py's cell), where a TMA family runs a product's last step and column
    tile partial (`mosaic_gpu.partial_runs`), its loads past the call zero-filled, interpreted padded to
    whole tiles (`grouped_matmul._Mosaic._aligned`), as the kernel gradient is everywhere."""
    sizes = [70, 0, 129, 57]
    starts = np.cumsum([0, *sizes])
    keys = jax.random.split(jax.random.key(4), 3)

    def ints(key, shape):
        return jax.random.randint(key, shape, -1, 2).astype(jnp.bfloat16)

    x, w, dy = ints(keys[0], (sum(sizes), depth)), ints(keys[1], (len(sizes), depth, width)), ints(
        keys[2], (sum(sizes), width))
    group_sizes = jnp.asarray(sizes, jnp.int32)
    options = {"out_dtype": jnp.float32, "precision": "default", "interpret_on_cpu": interpret}

    def grouped(lhs, matrices):
        return np.concatenate([np.asarray(lhs[a:b], np.float32) @ np.asarray(matrices[g], np.float32)
                               for g, (a, b) in enumerate(itertools.pairwise(starts))])

    calls = {
        "grouped forward": (lambda: gemm.gemm(x, w, group_sizes, **options), grouped(x, w)),
        "grouped input gradient": (lambda: gemm.gemm(dy, w, group_sizes, transpose_rhs=True, **options),
                                   grouped(dy, jnp.swapaxes(w, 1, 2))),
        "grouped kernel gradient": (lambda: gemm.gemm_t(x, dy, group_sizes, **options), np.stack(
            [np.asarray(x[a:b], np.float32).T @ np.asarray(dy[a:b], np.float32)
             for a, b in itertools.pairwise(starts)])),
        "dense forward": (lambda: gemm.gemm(x, w[0], None, **options),
                          np.asarray(x, np.float32) @ np.asarray(w[0], np.float32)),
    }
    with generation.chosen(grouped_matmul="mosaic"):
        for name, (call, expected) in calls.items():
            with generation.routes() as routes:
                got = np.asarray(jax.jit(call)(), np.float32)
            assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}, name
            np.testing.assert_array_equal(got, expected, err_msg=name)


def test_a_dense_products_partial_depth_lowers_with_its_matrix_prefetched():
    """A dense product whose depth no step divides (16 rows of 96 by 64 columns, a 64-deep step: its last
    step partial, `mosaic_gpu.partial_runs`) lowers for an H100 on the template, its matrix's whole steps
    prefetched into L2 before its launch's wait (`_prefetched`) and its partial last step read bounded: a
    prefetch of every step's block read past the matrix's depth (row 3601's prefetch, which the compile-ahead
    step's 96-wide projections reached once the template became Hopper's dense default)."""
    from kernel_compile import lowered_for

    from dew.nn.kernels.lowered import kernel_calls

    def product(x, w):
        return gemm.gemm(x, w, None, out_dtype=jnp.float32)

    with generation.chosen(grouped_matmul="mosaic"):
        lowered = lowered_for(product, (jax.ShapeDtypeStruct((16, 96), jnp.bfloat16),
                                        jax.ShapeDtypeStruct((96, 64), jnp.bfloat16)), "sm90")
    assert [call.kind for call in kernel_calls(lowered)] == ["mosaic_gpu"]
