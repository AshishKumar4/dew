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
            width_ran, depth_ran = launch._aligned(ba, target).arguments['kernel'].shape[1:]
            padded = launch.derived(launch._aligned(ba, target, whole_tiles=True), target)
            return (depth_ran, width_ran), launch.derived(ba, target), padded

    for depth, width in ((200, 136), (1000, 1400)):
        (depth_ran, width_ran), tile, padded = ran(depth, width)
        assert tile == padded and tile is not None
        family = pricing.family_of(tile, jnp.bfloat16)
        assert (width_ran == width) is (pricing.tile_columns(tile, family) <= width)
        assert (depth_ran == depth) is (tile.block_k == step)
    assert ran(1000, 1400)[0] == (1000, 1400)
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
def test_a_product_of_unaligned_widths_runs_on_the_template(generation_name):
    """`unaligned_widths_run_on_the_template`, interpreted on the CPU as each family."""
    with generation.assumed(generation_name):
        unaligned_widths_run_on_the_template(WIDTHS, interpret=True)


def unaligned_widths_run_on_the_template(widths: list, *, interpret: bool) -> None:
    """At each of `widths`' depth and width, which no tile of the family divides, a grouped forward, its input
    gradient and its kernel gradient, and a dense forward, all one program (traced and compiled once, bible
    §8, in a tier's prepare: `ahead_arm`), the template's launch named (`'mosaic'`): over integers from -1 to
    1, every product and sum exact, bitwise numpy's, groups of rows no tile divides and one empty; on a GPU
    itself too (tests/test_gpu_gemm.py's cell), where a TMA family runs a product's last step and column
    tile partial (`mosaic_gpu.partial_runs`), its loads past the call zero-filled, interpreted padded to
    whole tiles (`grouped_matmul._Mosaic._aligned`), as the kernel gradient is everywhere."""
    from reference_error import ahead_arm

    calls, inputs = {}, []
    for depth, width in widths:
        each, operands = _unaligned_calls(depth, width, interpret=interpret)
        calls.update({name: (call, len(inputs), expected) for name, (call, expected) in each.items()})
        inputs.append(operands)

    def products(*flat):
        return tuple(call(*flat[3 * at:3 * at + 3]) for call, at, _ in calls.values())

    label = "unaligned widths " + ", ".join(f"{depth}x{width}" for depth, width in widths)
    with generation.chosen(grouped_matmul="mosaic"), generation.routes() as routes:
        got = ahead_arm(jax.jit(products), label)(*(array for operands in inputs for array in operands))
    launched = [route.kernel for route in routes if route.choice == "grouped_matmul"]
    assert len(launched) >= len(calls) and set(launched) == {"mosaic"}, launched
    for (name, (_, _, expected)), value in zip(calls.items(), got, strict=True):
        np.testing.assert_array_equal(np.asarray(value, np.float32), expected, err_msg=name)


def _unaligned_calls(depth: int, width: int, *, interpret: bool) -> tuple[dict, tuple]:
    """`unaligned_widths_run_on_the_template`'s calls at a depth and a width, each a function of its
    operands with numpy's product, and those operands: NumPy's, as a site's draws are (a GPU tier's prepare
    computes on no device)."""
    sizes = [70, 0, 129, 57]
    starts = np.cumsum([0, *sizes])
    rng = np.random.default_rng(4)
    x, w, dy = (rng.integers(-1, 2, shape).astype(jnp.bfloat16)
                for shape in ((sum(sizes), depth), (len(sizes), depth, width), (sum(sizes), width)))
    group_sizes = np.asarray(sizes, np.int32)
    options = {"out_dtype": jnp.float32, "precision": "default", "interpret_on_cpu": interpret}

    def grouped(lhs, matrices):
        return np.concatenate([np.asarray(lhs[a:b], np.float32) @ np.asarray(matrices[g], np.float32)
                               for g, (a, b) in enumerate(itertools.pairwise(starts))])

    shape = f"{depth}x{width}"
    return {
        f"{shape} grouped forward": (lambda x, w, dy: gemm.gemm(x, w, group_sizes, **options), grouped(x, w)),
        f"{shape} grouped input gradient": (
            lambda x, w, dy: gemm.gemm(dy, w, group_sizes, transpose_rhs=True, **options),
            grouped(dy, np.swapaxes(w, 1, 2))),
        f"{shape} grouped kernel gradient": (lambda x, w, dy: gemm.gemm_t(x, dy, group_sizes, **options),
                                             np.stack([np.asarray(x[a:b], np.float32).T
                                                       @ np.asarray(dy[a:b], np.float32)
                                                       for a, b in itertools.pairwise(starts)])),
        f"{shape} dense forward": (lambda x, w, dy: gemm.gemm(x, w[0], None, **options),
                                   np.asarray(x, np.float32) @ np.asarray(w[0], np.float32)),
    }, (x, w, dy)


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
