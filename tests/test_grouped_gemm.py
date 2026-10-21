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
from jax.sharding import Mesh
from reference_error import assert_as_exact_as_the_reference, equations

from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.templates import cost
from dew.nn.kernels.templates.gemm import mosaic_gpu
from dew.nn.moe import expert_projection, grouped_matmul_kernel

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


def float64_products(x, w, dy, group):
    """The forward, input gradient and kernel gradient in float64: zero on the unrouted rows and empty
    groups."""
    forward, input_gradient = np.zeros((ROWS, WIDTH)), np.zeros((ROWS, DEPTH))
    kernel_gradient = np.zeros(w.shape)
    for expert in range(w.shape[0]):
        rows = group == expert
        forward[rows] = x[rows] @ w[expert]
        input_gradient[rows] = dy[rows] @ w[expert].T
        kernel_gradient[expert] = x[rows].T @ dy[rows]
    return forward, input_gradient, kernel_gradient


def xla_products(x, w, dy, sizes, dtype, kernel_dtype):
    """XLA's ragged dot over the same operands, each product summed in fp32 and rounded once."""
    sizes = jnp.asarray(sizes, jnp.int32)
    x, w, dy = (jnp.asarray(value, dtype) for value in (x, w, dy))
    forward = grouped_matmul.xla_ragged_dot(x, w, sizes, preferred_element_type=jnp.float32)
    input_gradient = grouped_matmul.xla_ragged_dot(dy, jnp.swapaxes(w, 1, 2), sizes,
                                                   preferred_element_type=jnp.float32)
    _, pullback = jax.vjp(
        lambda w: grouped_matmul.xla_ragged_dot(x, w, sizes, preferred_element_type=jnp.float32),
        w.astype(jnp.float32))
    kernel_gradient = pullback(dy.astype(jnp.float32))[0]
    return forward.astype(dtype), input_gradient.astype(dtype), kernel_gradient.astype(kernel_dtype)


@pytest.mark.parametrize("sizes", list(SIZES), ids=list(SIZES))
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_the_hopper_grouped_matmuls_sum_each_groups_rows_as_exactly_as_xla(sizes, dtype):
    """Both orientations of the forward, and the kernel gradient, interpreted on a CPU, at row blocks
    groups share and blocks of their own: every product as exact as XLA's ragged dot from float64 sums of
    the same rounded operands, unrouted rows and empty groups exactly zero."""
    sizes = SIZES[sizes]
    x, w, dy, group = operands(sizes, dtype)
    truth = float64_products(x, w, dy, group)
    reference = xla_products(x, w, dy, sizes, dtype, jnp.float32)
    arrays = [jnp.asarray(value, dtype) for value in (x, w, dy)]
    group_sizes = jnp.asarray(sizes, jnp.int32)
    for block_m, transpose in itertools.product((64, 128), (False, True)):
        tiles = mosaic_gpu.Tiles(block_m, 64, 64, 2, 1)
        matrix = jnp.swapaxes(arrays[1], 1, 2) if transpose else arrays[1]
        got = mosaic_gpu.gmm(arrays[0], matrix, group_sizes, tiles=tiles, transpose_rhs=transpose,
                             out_dtype=dtype, multiprocessors=1, interpret=grouped_matmul.INTERPRETED)
        check(got, reference[0], truth[0], group, f"forward block_m={block_m} transpose={transpose}")
    for block_k in (64, 128):
        tiles = mosaic_gpu.Tiles(64, 64, block_k, 2, 1)
        got = mosaic_gpu.tgmm(arrays[0], arrays[2], group_sizes, tiles=tiles, out_dtype=jnp.float32,
                              multiprocessors=1, interpret=grouped_matmul.INTERPRETED)
        check(got, reference[2], truth[2], None, f"kernel gradient block_k={block_k}")


def check(got, reference, truth, group, label):
    got = np.asarray(got, np.float64)
    if group is not None:
        assert not np.any(got[group < 0]), f"{label}: an unrouted row is not zero"
    if np.any(truth):
        assert_as_exact_as_the_reference(got, np.asarray(reference, np.float64), truth, label)
    else:
        assert not np.any(got), f"{label}: every group is empty and the product is not zero"


def test_the_kernel_gradient_keeps_a_neighbours_non_finite_rows_out():
    """The rows of the next group and the unrouted tail hold NaN and inf; each group's kernel gradient
    sums its own rows alone, every one finite and as exact as XLA's over clean rows."""
    sizes = [37, 100, 0, 40]
    x, w, dy, group = operands(sizes, jnp.bfloat16, seed=3)
    truth = float64_products(x, w, dy, group)[2]
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


@pytest.mark.parametrize("generation_name,implementation", [("sm80", "triton"), ("sm90", "mosaic_gpu")])
def test_an_expert_projection_runs_the_implementation_chosen_for_it_on_the_cpu(generation_name: str,
                                                                               implementation: str):
    """'pallas' interprets the kernels on a CPU as the named generation runs them, each launch the
    implementation `generation.chosen` names: its forward and both gradients as exact as XLA's, the input
    gradient in the input's dtype and the kernel gradient in the master's, each launch's route the one
    named."""
    sizes = [0, 70, 1, 100, 0]
    x, w, dy, group = operands(sizes, jnp.bfloat16, seed=5)
    forward, input_gradient, kernel_gradient = float64_products(x, w, dy, group)
    reference = xla_products(x, w, dy, sizes, jnp.bfloat16, jnp.float32)
    group_sizes = jnp.asarray(sizes, jnp.int32)

    def products(x, w):
        y, pull = jax.vjp(lambda x, w: expert_projection(x, w, group_sizes, jnp.bfloat16, "pallas", None),
                          x, w)
        return (y, *pull(jnp.asarray(dy, jnp.bfloat16)))

    with (generation.assumed(generation_name), generation.chosen(grouped_matmul=implementation),
          generation.routes() as routes):
        y, dx, dw = jax.jit(products)(jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.float32))
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {implementation}
    assert (y.dtype, dx.dtype, dw.dtype) == (jnp.bfloat16, jnp.bfloat16, jnp.float32)
    for label, got, theirs, truth in zip(("y", "dx", "dw"), (y, dx, dw), reference,
                                         (forward, input_gradient, kernel_gradient), strict=True):
        assert_as_exact_as_the_reference(np.asarray(got, np.float64), np.asarray(theirs, np.float64), truth,
                                          label)


@pytest.mark.parametrize("launch", [grouped_matmul.MosaicGmm(), grouped_matmul.MosaicGmmTransposed(),
                                    grouped_matmul.MosaicTgmm()], ids=lambda launch: type(launch).__name__)
def test_every_tile_of_the_hopper_grouped_matmuls_lowers_for_an_h100(launch):
    """The tiles the Mosaic GPU launches may pick at an expert layer's shapes lower for an H100 on the
    CPU, as a GPU session's compile would: every block shape and every schedule among those that fit an
    H100's block (`Template.fitting`). The interpreter takes kernels Mosaic GPU's layout inference
    refuses; the GPU's tile gate runs each one (test_moe_precision)."""
    rows, depth, width, groups = 2048, 768, 1536, 8
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sizes, matrices = jax.ShapeDtypeStruct((groups,), jnp.int32), bf16((groups, depth, width))
    if isinstance(launch, grouped_matmul.MosaicTgmm):
        args, options = (bf16((rows, depth)), bf16((rows, width)), sizes), {"out_dtype": "float32"}
    elif isinstance(launch, grouped_matmul.MosaicGmmTransposed):
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
        assert {(c.block_m, c.block_n) for c in covering} == {(c.block_m, c.block_n) for c in fits}
        assert {(c.stages, c.grid_block_n) for c in covering} == {(c.stages, c.grid_block_n) for c in fits}
        for config in sorted(covering, key=repr):
            jax.jit(functools.partial(launch.replace(config=config), **options)).trace(*args).lower(
                lowering_platforms=("cuda",))


def test_a_forced_implementation_runs_or_says_why_it_cannot():
    """'xla' named runs XLA's ragged dot whatever the call asked; a kernel named for a product the
    kernels do not compute (fp32 above the default precision) raises with the reason rather than
    running XLA."""
    with generation.chosen(grouped_matmul="xla"):
        assert grouped_matmul_kernel("pallas", jnp.bfloat16, (jnp.bfloat16, jnp.float32), None) == "xla"
    with generation.chosen(grouped_matmul="triton"), pytest.raises(ValueError):
        grouped_matmul_kernel("pallas", jnp.float32, (jnp.float32, jnp.float32), "highest")


def test_the_kernels_write_each_product_in_its_own_dtype():
    """A bf16 projection over an fp32 master: the forward launch writes bf16 and the kernel gradient's
    fp32, with no fp32 `[rows, width]` product left for a conversion pass after the launch."""
    x = jax.ShapeDtypeStruct((ROWS, DEPTH), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((4, DEPTH, WIDTH), jnp.float32)
    sizes = jax.ShapeDtypeStruct((4,), jnp.int32)

    def loss(x, w, sizes):
        return expert_projection(x, w, sizes, jnp.bfloat16, "pallas", None).astype(jnp.float32).sum()

    jaxpr = jax.make_jaxpr(jax.grad(loss, (0, 1)))(x, w, sizes)
    written = [(var.aval.shape, var.aval.dtype) for equation in equations(jaxpr)
               if equation.primitive.name == "pallas_call" for var in equation.outvars]
    assert ((ROWS, WIDTH), jnp.dtype(jnp.bfloat16)) in written
    assert ((ROWS, WIDTH), jnp.dtype(jnp.float32)) not in written
    assert ((4, DEPTH, WIDTH), jnp.dtype(jnp.float32)) in written


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
    assert seconds(0) == 0
    assert seconds(sms) == pytest.approx(program)
    assert seconds(sms + 1) == pytest.approx(2 * program)
    assert seconds(1, device_bytes=target.memory_bandwidth) == pytest.approx(1.0)


def test_a_mig_slice_keeps_its_generations_tensor_rate():
    """A SKU's catalogued peak lowers its generation's rate where its multiprocessors and clock would
    exceed it (a GeForce part's half-rate accumulation) and never raises it: an A100 slice reporting 14
    of its 108 multiprocessors keeps 2048 FLOPs a clock each."""
    a100 = generation.TARGETS["sm80"]
    assert a100.tensor_rate("16bit") == pytest.approx(2048, rel=1e-3)
    assert dataclasses.replace(a100, multiprocessors=14).tensor_rate("16bit") == 2048
    rtx4080 = dataclasses.replace(generation.TARGETS["sm89"], device_kind="NVIDIA GeForce RTX 4080",
                                  multiprocessors=76, clock_hz=2.505e9)
    assert rtx4080.tensor_rate("16bit") == pytest.approx(512, rel=1e-2)


def test_an_unseen_gpu_runs_xla_s_grouped_matmul():
    """A GPU generation that states no tensor rate (an sm100 the table does not name) runs an expert
    projection on XLA's ragged dot, its route saying why, with no estimate asked of the kernels."""
    x = jnp.ones((32, 64), jnp.bfloat16)
    w = jnp.ones((4, 64, 128), jnp.float32)
    sizes = jnp.asarray([8, 8, 8, 8], jnp.int32)
    with generation.assumed("sm100"), generation.routes() as taken:
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
