"""The grouped GEMM template's kernels and their choice: Dew's Mosaic GPU grouped matmuls for Hopper
(`dew.nn.kernels.templates.gemm.mosaic_gpu`), interpreted on the CPU, against float64 sums of the same
rounded operands; the MoE dispatcher's routes to each implementation; the products' dtypes as the
kernels write them; and the cost model (`dew.nn.kernels.templates.cost`) on analytic launches."""

import dataclasses
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS, generations, lowers
from jax.sharding import Mesh
from reference_error import assert_as_exact_as_the_reference, equations

from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.templates import cost
from dew.nn.kernels.templates.gemm import mosaic_gpu
from dew.nn.moe import expert_projection
from tools import grouped_oracle as oracle

pytestmark = pytest.mark.kernel

SIZES = {"ragged": [0, 70, 1, 100, 0], "one-group": [200], "all-empty": [0, 0, 0], "aligned": [64, 64, 64, 8],
         "unrouted-tail": [3, 0, 120]}
ROWS, DEPTH, WIDTH = 200, 128, 256


def rounded(values, dtype) -> np.ndarray:
    return np.asarray(jnp.asarray(values, dtype), np.float64)


def operands(sizes, dtype, seed=0):
    """Rows, per-group matrices and a cotangent, rounded to `dtype`, and the group of each row (-1 past the
    routed ones)."""
    rng = np.random.default_rng(seed)
    x = rounded(rng.normal(size=(ROWS, DEPTH)), dtype)
    w = rounded(rng.normal(size=(len(sizes), DEPTH, WIDTH)) / np.sqrt(DEPTH), dtype)
    dy = rounded(rng.normal(size=(ROWS, WIDTH)), dtype)
    group = np.full(ROWS, -1)
    group[:sum(sizes)] = np.repeat(np.arange(len(sizes)), sizes)
    return x, w, dy, group


def xla_products(x, w, dy, sizes, dtype, kernel_dtype):
    """The portable projection's (`dew.nn.moe.expert_projection` on XLA) forward and gradients over the
    same operands, `w` the master in `kernel_dtype`: each product summed in fp32 and rounded once."""
    group_sizes = jnp.asarray(sizes, jnp.int32)
    output, pullback = jax.vjp(lambda x, w: expert_projection(x, w, group_sizes, dtype, "xla", None),
                               jnp.asarray(x, dtype), jnp.asarray(w, kernel_dtype))
    return (output, *pullback(jnp.asarray(dy, dtype)))


@INTERPRETS
@pytest.mark.parametrize("sizes", list(SIZES), ids=list(SIZES))
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_the_hopper_grouped_matmuls_sum_each_groups_rows_as_exactly_as_xla(sizes, dtype):
    """Both orientations of the forward, and the kernel gradient, interpreted on a CPU, at row blocks
    groups share and blocks of their own: every product as exact as XLA's ragged dot from float64 sums of
    the same rounded operands, unrouted rows and empty groups exactly zero."""
    sizes = SIZES[sizes]
    x, w, dy, group = operands(sizes, dtype)
    truth = oracle.products(x, w, dy, sizes)
    reference = xla_products(x, w, dy, sizes, dtype, jnp.float32)
    arrays = [jnp.asarray(value, dtype) for value in (x, w, dy)]
    group_sizes = jnp.asarray(sizes, jnp.int32)
    # One warpgroup a tile at two row blocks, and two (CUTLASS's cooperative schedule; a pair of
    # multiprocessors the interpreter does not model, the GPU's tile gate runs).
    tiled = [mosaic_gpu.Tiles(64, 64, 64, 2, 1), mosaic_gpu.Tiles(128, 64, 64, 2, 1),
             mosaic_gpu.Tiles(64, 128, 64, 2, 1, warpgroups=2)]
    for tiles, transpose in itertools.product(tiled, (False, True)):
        matrix = jnp.swapaxes(arrays[1], 1, 2) if transpose else arrays[1]
        got = mosaic_gpu.gmm(arrays[0], matrix, group_sizes, tiles=tiles, transpose_rhs=transpose,
                             out_dtype=dtype, multiprocessors=1, interpret=grouped_matmul.INTERPRETED)
        check(got, reference[0], truth[0], group, f"forward {tiles} transpose={transpose}")
    for tiles in (mosaic_gpu.Tiles(64, 64, 64, 2, 1), mosaic_gpu.Tiles(64, 64, 128, 2, 1),
                  mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2)):
        got = mosaic_gpu.tgmm(arrays[0], arrays[2], group_sizes, tiles=tiles, out_dtype=jnp.float32,
                              multiprocessors=1, interpret=grouped_matmul.INTERPRETED)
        check(got, reference[2], truth[2], None, f"kernel gradient {tiles}")


def check(got, reference, truth, group, label):
    got = np.asarray(got, np.float64)
    if group is not None:
        assert not np.any(got[group < 0]), f"{label}: an unrouted row is not zero"
    if np.any(truth):
        assert_as_exact_as_the_reference(got, np.asarray(reference, np.float64), truth, label)
    else:
        assert not np.any(got), f"{label}: every group is empty and the product is not zero"


@INTERPRETS
def test_the_kernel_gradient_keeps_a_neighbours_non_finite_rows_out():
    """The rows of the next group and the unrouted tail hold NaN and inf; each group's kernel gradient
    sums its own rows alone, every one finite and as exact as XLA's over clean rows."""
    sizes = [37, 100, 0, 40]
    x, w, dy, group = operands(sizes, jnp.bfloat16, seed=3)
    truth = oracle.kernel_gradient(x, dy, sizes)
    reference = xla_products(x, w, dy, sizes, jnp.bfloat16, jnp.float32)[2]
    dirty_x, dirty_dy = x.copy(), dy.copy()
    dirty_x[group < 0], dirty_dy[group < 0] = np.inf, np.nan
    tiles = mosaic_gpu.Tiles(64, 64, 64, 2, 1)
    interpret = grouped_matmul.INTERPRETED
    got = mosaic_gpu.tgmm(jnp.asarray(dirty_x, jnp.bfloat16), jnp.asarray(dirty_dy, jnp.bfloat16),
                          jnp.asarray(sizes, jnp.int32), tiles=tiles, out_dtype=jnp.float32,
                          multiprocessors=1, interpret=interpret)
    assert np.all(np.isfinite(np.asarray(got)))
    assert_as_exact_as_the_reference(np.asarray(got, np.float64), np.asarray(reference, np.float64), truth,
                                      "kernel gradient")


def test_the_portable_branch_writes_each_gradient_from_the_rounded_operands():
    """The template's portable branch (a CPU's, uninterpreted) differentiates the operands as the
    projection rounded them: an fp16 master's largest value rounds to bf16's 65536, which fp16 does not
    hold, and the input gradient through it is finite, each gradient rounded once to its operand's
    dtype."""
    x = jnp.ones((1, 1), jnp.float32)
    kernel = jnp.full((1, 1, 1), 65504, jnp.float16)
    _, pullback = jax.vjp(lambda x, kernel: expert_projection(
        x, kernel, jnp.ones((1,), jnp.int32), jnp.bfloat16, "auto", None), x, kernel)
    d_x, d_kernel = pullback(jnp.full((1, 1), 2.0 ** -16, jnp.bfloat16))
    np.testing.assert_array_equal(np.asarray(d_x), np.ones((1, 1), np.float32))
    np.testing.assert_array_equal(np.asarray(d_kernel), np.full((1, 1, 1), 2.0 ** -16, np.float16))

@INTERPRETS
@pytest.mark.parametrize("generation_name,implementation", [("sm80", "triton"), ("sm90", "mosaic_gpu")])
def test_an_expert_projection_runs_the_implementation_chosen_for_it_on_the_cpu(generation_name: str,
                                                                               implementation: str):
    """'pallas' interprets the kernels on a CPU as the named generation runs them, each launch the
    implementation `generation.chosen` names: its forward and both gradients as exact as XLA's, the input
    gradient in the input's dtype and the kernel gradient in the master's, each launch's route the one
    named."""
    sizes = [0, 70, 1, 100, 0]
    x, w, dy, _ = operands(sizes, jnp.bfloat16, seed=5)
    forward, input_gradient, kernel_gradient = oracle.products(x, w, dy, sizes)
    reference = xla_products(x, w, dy, sizes, jnp.bfloat16, jnp.float32)
    group_sizes = jnp.asarray(sizes, jnp.int32)

    def products(x, w):
        y, pull = jax.vjp(lambda x, w: expert_projection(x, w, group_sizes, jnp.bfloat16, "pallas", None),
                          x, w)
        return (y, *pull(jnp.asarray(dy, jnp.bfloat16)))

    with (generation.assumed(generation_name), generation.chosen(grouped_matmul=implementation),
          generation.routes() as routes):
        operands_ = (jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.float32))
        # The CPU runs the interpreted kernels (each launch's route says so), not XLA's ragged dot compared
        # with itself. Mosaic's interpreter's ordered callbacks run eagerly: a jitted `linear_call` cannot
        # carry them.
        y, dx, dw = (products if implementation == "mosaic_gpu" else jax.jit(products))(*operands_)
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {implementation}
    assert (y.dtype, dx.dtype, dw.dtype) == (jnp.bfloat16, jnp.bfloat16, jnp.float32)
    for label, got, theirs, truth in zip(("y", "dx", "dw"), (y, dx, dw), reference,
                                         (forward, input_gradient, kernel_gradient), strict=True):
        assert_as_exact_as_the_reference(np.asarray(got, np.float64), np.asarray(theirs, np.float64), truth,
                                          label)



@INTERPRETS
@pytest.mark.parametrize("generation_name,implementation", [("sm80", "triton"), ("sm90", "mosaic_gpu")])
@pytest.mark.parametrize("master", [jnp.bfloat16, jnp.float32], ids=["bf16", "fp32"])
def test_a_banked_projection_is_its_layers_plain_projection(generation_name: str, implementation: str,
                                                           master):
    """An expert projection reading its matrices from a bank's layer (`layer`) is the plain projection of
    that layer, bit for bit: its output and input gradient, and its kernel gradient at the layer, zero at
    every other (the plain projection held to check 1 above). A bf16 bank is read in place by the kernels,
    an fp32 master's layer cast first."""
    sizes = [0, 70, 1, 100, 0]
    x, w, dy, _ = operands(sizes, jnp.bfloat16, seed=5)
    rng = np.random.default_rng(6)
    bank = jnp.asarray(rng.normal(size=(3, *w.shape)), master).at[1].set(jnp.asarray(w, master))
    group_sizes = jnp.asarray(sizes, jnp.int32)
    x, dy = jnp.asarray(x, jnp.bfloat16), jnp.asarray(dy, jnp.bfloat16)

    def products(x, w, **layer):
        y, pull = jax.vjp(lambda x, w: expert_projection(x, w, group_sizes, jnp.bfloat16, "pallas", None,
                                                         **layer), x, w)
        return (y, *pull(dy))

    # Mosaic's interpreter's ordered callbacks run eagerly: a jitted `linear_call` cannot carry them.
    staged = (lambda f: f) if implementation == "mosaic_gpu" else jax.jit
    with (generation.assumed(generation_name), generation.chosen(grouped_matmul=implementation),
          generation.routes() as routes):
        banked = staged(functools.partial(products, layer=jnp.int32(1)))(x, bank)
        plain = staged(products)(x, bank[1])
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {implementation}
    for got, want in zip(banked[:2], plain[:2], strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(np.asarray(banked[2][1]), np.asarray(plain[2]))
    np.testing.assert_array_equal(np.asarray(banked[2][::2]), np.zeros((2, *w.shape)))

@lowers("sm90")
@pytest.mark.parametrize("launch,trans_rhs", [(grouped_matmul.MosaicGmm(), False),
                                              (grouped_matmul.MosaicGmm(), True),
                                              (grouped_matmul.MosaicTgmm(), None)],
                         ids=["MosaicGmm", "MosaicGmm-transposed", "MosaicTgmm"])
def test_every_tile_of_the_hopper_grouped_matmuls_lowers_for_an_h100(launch, trans_rhs):
    """The tiles the Mosaic GPU launches may pick at an expert layer's shapes lower for an H100 on the
    CPU, as a GPU session's compile would, over rows no multiple of 8: every block shape and every
    schedule among those that fit an H100's block (`Template.fitting`). The interpreter takes kernels
    Mosaic GPU's layout inference and TMA descriptors refuse; the GPU's tile gate runs each one
    (test_moe_precision)."""
    # Rows not a multiple of 8, which a TMA descriptor tiles by.
    rows, depth, width, groups = 2044, 768, 1536, 8
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sizes, matrices = jax.ShapeDtypeStruct((groups,), jnp.int32), bf16((groups, depth, width))
    if isinstance(launch, grouped_matmul.MosaicTgmm):
        args, options = (bf16((rows, depth)), bf16((rows, width)), sizes), {"out_dtype": "float32"}
    elif trans_rhs:
        args, options = (bf16((rows, width)), matrices, sizes), {"trans_rhs": True}
    else:
        args, options = (bf16((rows, depth)), matrices, sizes), {"trans_rhs": False}
    options = {"out_dtype": "bfloat16", **options, "interpret": False}
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        fits = launch.fitting(launch.bind(*args, **options), generation.gpu_target())
        # A fitting tile of every block shape and of every schedule: what the lowering depends on
        # (shared memory, the snake's index arithmetic), each value once.
        covering = set({(c.block_m, c.block_n): c for c in reversed(fits)}.values())
        covering |= set({(c.stages, c.grid_block_n): c for c in reversed(fits)}.values())
        covering |= set({(c.warpgroups, c.cluster, c.block_n): c for c in reversed(fits)}.values())
        assert {(c.block_m, c.block_n) for c in covering} == {(c.block_m, c.block_n) for c in fits}
        assert {(c.stages, c.grid_block_n) for c in covering} == {(c.stages, c.grid_block_n) for c in fits}
        assert {(c.warpgroups, c.cluster) for c in covering} == {(c.warpgroups, c.cluster) for c in fits}
        for config in sorted(covering, key=repr):
            jax.jit(functools.partial(launch.replace(config=config), **options)).trace(*args).lower(
                lowering_platforms=("cuda",))


def test_the_kernels_forward_mode_keeps_the_projections_law():
    """An expert projection's tangent through the template is the portable projection's, bit for bit:
    `dx @ Q(kernel) + Q(x) @ dkernel` with the tangents unrounded, where a cast's own tangent would round
    an fp32 input's to bf16 first (an fp32 1 + 2^-8 times 1.5 is 1.5078125, rounded first 1.5)."""
    sizes = jnp.asarray([1, 2], jnp.int32)
    x, w = jnp.zeros((3, 2), jnp.float32), jnp.full((2, 2, 2), 1.5, jnp.float32)
    dx, dw = jnp.full((3, 2), 1 + 2.0 ** -8, jnp.float32), jnp.zeros((2, 2, 2), jnp.float32)
    tangents = {implementation: jax.jvp(lambda x, w, implementation=implementation: expert_projection(
        x, w, sizes, jnp.bfloat16, implementation, None), (x, w), (dx, dw))[1]
        for implementation in ("auto", "xla")}
    np.testing.assert_array_equal(np.asarray(tangents["auto"], np.float32),
                                  np.asarray(tangents["xla"], np.float32))
    assert np.asarray(tangents["xla"], np.float32)[0, 0] == np.float32(1.5078125).astype(jnp.bfloat16)


def test_a_named_kernel_refuses_where_the_kernels_do_not_run():
    """A kernel `generation.chosen` names for a call no kernel runs (a CPU's without 'pallas', which alone
    interprets them) raises with the reason, rather than running XLA."""
    x, w, sizes = jnp.ones((8, 16), jnp.bfloat16), jnp.ones((2, 16, 16), jnp.float32), jnp.asarray([4, 4])
    with generation.chosen(grouped_matmul="triton"), pytest.raises(ValueError, match="GPU of sm80"):
        expert_projection(x, w, sizes, jnp.bfloat16, "auto", None)

def test_a_forced_implementation_runs_or_says_why_it_cannot():
    """'xla' named runs XLA's ragged dot whatever the call asked; a kernel named for a product the
    kernels do not compute (fp32 above the default precision) raises with the reason rather than
    running XLA."""
    x, w, sizes = jnp.ones((8, 16), jnp.bfloat16), jnp.ones((2, 16, 16), jnp.float32), jnp.asarray([4, 4])
    with generation.chosen(grouped_matmul="xla"), generation.routes() as taken:
        expert_projection(x, w, sizes, jnp.bfloat16, "pallas", None)
    assert {route.kernel for route in taken if route.choice == "grouped_matmul"} == {"xla"}
    with generation.chosen(grouped_matmul="triton"), pytest.raises(ValueError):
        expert_projection(x.astype(jnp.float32), w, sizes, jnp.float32, "pallas", "highest")


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_the_kernels_write_each_product_in_its_own_dtype(generation_name):
    """A bf16 projection over an fp32 master, through the implementations each generation picks: its
    launches write the forward and the input gradient in bf16 and the kernel gradient in fp32, with no
    fp32 `[rows, width]` product for a conversion pass after."""
    x = jax.ShapeDtypeStruct((ROWS, DEPTH), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((4, DEPTH, WIDTH), jnp.float32)
    sizes = jax.ShapeDtypeStruct((4,), jnp.int32)

    def loss(x, w, sizes):
        return jnp.sum(expert_projection(x, w, sizes, jnp.bfloat16, "pallas", None))

    with generation.assumed(generation_name):
        jaxpr = jax.make_jaxpr(jax.grad(loss, (0, 1)))(x, w, sizes)
    # Each launch's own outputs: Triton's are a pallas_call's, Mosaic GPU's a custom_vmap_call's
    # (`plgpu.kernel` wraps its call in `custom_vmap`).
    written = {(var.aval.shape, var.aval.dtype) for equation in equations(jaxpr)
               if equation.primitive.name in ("pallas_call", "custom_vmap_call") for var in equation.outvars}
    assert ((ROWS, WIDTH), jnp.dtype(jnp.float32)) not in written
    assert {((ROWS, WIDTH), jnp.dtype(jnp.bfloat16)), ((ROWS, DEPTH), jnp.dtype(jnp.bfloat16)),
            ((4, DEPTH, WIDTH), jnp.dtype(jnp.float32))} <= written


def test_the_cost_model_charges_each_multiprocessor_its_programs_in_turn():
    """A multiprocessor does its programs' work one after another: a launch of one program per
    multiprocessor takes one program's time, one more program a second, and the device bytes bound it
    from below."""
    target = generation.TARGETS["sm90"]
    program = 4096 * 1000 / (target.tensor_rate("16bit") * target.clock_hz)

    def seconds(programs, device_bytes=0.0):
        work = cost.Cost(programs=programs, device_bytes=device_bytes, program_flops=4096 * 1000)
        return cost.seconds(work, target)

    sms = target.multiprocessors
    # Each side is the same arithmetic in the same order, so the times are equal exactly.
    assert seconds(0) == 0
    assert seconds(sms) == program
    assert seconds(sms + 1) == 2 * program
    assert seconds(1, device_bytes=target.memory_bandwidth) == 1.0


def test_a_mig_slice_keeps_its_generations_tensor_rate():
    """A SKU's catalogued peak lowers its generation's rate where its multiprocessors and clock would
    exceed it (a GeForce part's half-rate accumulation) and never raises it: an A100, whose catalogue
    peak its 108 multiprocessors and clock reach, and a slice of it reporting 14, keep 2048 FLOPs a
    clock each; a part whose peak is half its multiprocessors at the generation's rate (powers of two,
    so the halving is exact) takes half."""
    a100 = generation.TARGETS["sm80"]
    assert a100.tensor_rate("16bit") == 2048
    assert dataclasses.replace(a100, multiprocessors=14).tensor_rate("16bit") == 2048
    peaks = {"a half-rate part": {"bf16": 512 * 64 * 2.0 ** 31}}
    half = dataclasses.replace(generation.TARGETS["sm89"], device_kind="a half-rate part", multiprocessors=64,
                               clock_hz=2.0 ** 31, dense_peaks=peaks)
    assert half.tensor_rate("16bit") == 512


def test_an_unseen_gpu_runs_xla_s_grouped_matmul():
    """A GPU generation that states no tensor rate (an sm103 the table does not name) runs an expert
    projection on XLA's ragged dot, its route saying why, with no estimate asked of the kernels."""
    x = jnp.ones((32, 64), jnp.bfloat16)
    w = jnp.ones((4, 64, 128), jnp.float32)
    sizes = jnp.asarray([8, 8, 8, 8], jnp.int32)
    with generation.assumed("sm103"), generation.routes() as taken:
        jax.jit(lambda x, w: expert_projection(x, w, sizes, jnp.bfloat16, "auto", None)).trace(x, w)
    (route,) = {(route.kernel, route.refusal.detail if route.refusal else None) for route in taken
                if route.choice == "grouped_matmul"}
    assert route[0] == "xla" and "tensor rate" in route[1]


def test_a_block_aligned_launch_counts_the_tiles_its_groups_take():
    """A block-aligned launch runs each row block once and once more for each group boundary inside a
    block: one group takes exactly its blocks, many groups at most one more each after the first, and
    one row a single tile however many groups there are."""
    assert grouped_matmul._block_tiles(16384, 1, 128) == 128
    assert grouped_matmul._block_tiles(16384, 128, 128) == 128 + 127
    assert grouped_matmul._block_tiles(1, 256, 128) == 1
    assert grouped_matmul._block_tiles(100, 8, 64) == 2 + 7


def test_the_cost_model_names_a_fact_its_target_lacks():
    """A generation that states no tensor rate gives no estimate, naming the fact; the grouped kernels are
    then not eligible there, so XLA's ragged dot runs without one."""
    target = generation.GpuTarget("sm100", "an unseen GPU", 227 * 1024, multiprocessors=148, clock_hz=1.9e9,
                                  memory_bandwidth=8e12)
    with pytest.raises(ValueError):
        cost.seconds(cost.Cost(programs=1, device_bytes=0, program_flops=1.0), target)
    ba = grouped_matmul.Gmm().bind(jax.ShapeDtypeStruct((64, 128), jnp.bfloat16),
                                   jax.ShapeDtypeStruct((2, 128, 128), jnp.bfloat16),
                                   jax.ShapeDtypeStruct((2,), jnp.int32), out_dtype="bfloat16",
                                   trans_rhs=False, interpret=False)
    assert not grouped_matmul.Gmm().eligible(ba, target)


def halved(acc, bias):
    return (acc + bias[None, :]).astype(jnp.bfloat16)


def doubled_halves(acc, bias):
    return (acc * 2).astype(jnp.bfloat16), (acc + bias[None, :]).astype(jnp.bfloat16)


def twice(acc, bias):
    del bias
    return (acc * 2).astype(jnp.bfloat16)


def test_a_cooperative_tile_stores_each_of_several_outputs():
    """A cooperative tile's consumers stage every output at once: a two-output epilogue's outputs, each its
    own staging buffer, are bit for bit those of the same tile asked for each output alone."""
    x, w, _, _ = operands(SIZES["ragged"], jnp.bfloat16)
    x, w = jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.bfloat16)
    bias = jnp.asarray(np.linspace(-1, 1, WIDTH), jnp.float32)
    sizes = jnp.asarray(SIZES["ragged"], jnp.int32)
    tiles = mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2)

    def run(epilogue, out_dtypes=None):
        return mosaic_gpu.gmm(x, w, sizes, tiles=tiles, transpose_rhs=False, out_dtype=jnp.bfloat16,
                              multiprocessors=1, interpret=grouped_matmul.INTERPRETED, epilogue=epilogue,
                              epilogue_operands=(bias,), epilogue_axes=(("columns",),), out_dtypes=out_dtypes)

    first, second = run(doubled_halves, (jnp.bfloat16, jnp.bfloat16))
    np.testing.assert_array_equal(np.asarray(first, np.float32), np.asarray(run(twice), np.float32))
    np.testing.assert_array_equal(np.asarray(second, np.float32), np.asarray(run(halved), np.float32))
