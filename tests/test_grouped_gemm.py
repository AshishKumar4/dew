"""The grouped GEMM template's kernels and their choice: Dew's Mosaic GPU grouped matmuls for Hopper
(`dew.nn.kernels.templates.gemm.mosaic_gpu`), interpreted on the CPU, against float64 sums of the same
rounded operands; the MoE dispatcher's routes to each implementation; the products' dtypes as the
kernels write them; and the cost model (`dew.nn.kernels.templates.cost`) on analytic launches."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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
    """Each warpgroup split and both orientations of the forward, and the kernel gradient, interpreted on
    a CPU: every product as exact as XLA's ragged dot from float64 sums of the same rounded operands,
    unrouted rows and empty groups exactly zero."""
    sizes = SIZES[sizes]
    x, w, dy, group = operands(sizes, dtype)
    truth = float64_products(x, w, dy, group)
    reference = xla_products(x, w, dy, sizes, dtype, jnp.float32)
    interpret = grouped_matmul.INTERPRETED
    arrays = [jnp.asarray(value, dtype) for value in (x, w, dy)]
    group_sizes = jnp.asarray(sizes, jnp.int32)
    for split, transpose in itertools.product("mn", (False, True)):
        tiles = mosaic_gpu.Tiles(tile_m=64, tile_n=64, tile_k=64, stages=2, split=split, width=1)
        matrix = jnp.swapaxes(arrays[1], 1, 2) if transpose else arrays[1]
        got = mosaic_gpu.gmm(arrays[0], matrix, group_sizes, tiles=tiles, transpose_rhs=transpose,
                             out_dtype=dtype, multiprocessors=1, interpret=interpret)
        check(got, reference[0], truth[0], group, f"forward split={split} transpose={transpose}")
    tiles = mosaic_gpu.Tiles(tile_m=64, tile_n=64, tile_k=64, stages=2, split="n", width=1)
    got = mosaic_gpu.tgmm(arrays[0], arrays[2], group_sizes, tiles=tiles, out_dtype=jnp.float32,
                          multiprocessors=1, interpret=interpret)
    check(got, reference[2], truth[2], None, "kernel gradient")


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
    tiles = mosaic_gpu.Tiles(tile_m=64, tile_n=64, tile_k=64, stages=2, split="n", width=1)
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


def test_a_forced_implementation_runs_or_says_why_it_cannot():
    """'xla' named runs XLA's ragged dot whatever the call asked; a kernel named for a product the
    kernels do not compute (fp32 above the default precision) raises with the reason rather than
    running XLA."""
    with generation.chosen(grouped_matmul="xla"):
        assert grouped_matmul_kernel("pallas", jnp.bfloat16, (jnp.bfloat16, jnp.float32), None) == "xla"
    with generation.chosen(grouped_matmul="triton"), pytest.raises(ValueError, match="TF32"):
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


def test_the_cost_model_charges_the_programs_each_wave_has():
    """Full waves charge every resident program; a last wave short of every multiprocessor only the
    programs it has, so room for a second program costs nothing where the launch has none."""
    target = generation.TARGETS["sm90"]
    program = 4096 * 1000 / (target.tensor_rate("16bit") * target.clock_hz)

    def seconds(programs, resident):
        return cost.seconds(cost.Cost(programs=programs, device_bytes=0, program_flops=4096 * 1000,
                                      resident=resident), target)

    sms = target.multiprocessors
    assert seconds(0, 1) == 0
    assert seconds(sms, 1) == pytest.approx(program)
    assert seconds(sms, 2) == pytest.approx(program)
    assert seconds(sms + 1, 1) == pytest.approx(2 * program)
    assert seconds(2 * sms, 2) == pytest.approx(2 * program)
    assert seconds(2 * sms + 1, 2) == pytest.approx(3 * program)


def test_the_cost_model_names_a_fact_its_target_lacks():
    """A generation that states no tensor rate gives no estimate, naming the fact; the grouped kernels are
    then not eligible there, so XLA's ragged dot runs without one."""
    target = generation.GpuTarget("sm100", "an unseen GPU", 227 * 1024, multiprocessors=148, clock_hz=1.9e9,
                                  memory_bandwidth=8e12)
    with pytest.raises(ValueError, match="tensor_rates"):
        cost.seconds(cost.Cost(programs=1, device_bytes=0, program_flops=1.0), target)
    ba = grouped_matmul.Gmm().bind(jax.ShapeDtypeStruct((64, 128), jnp.bfloat16),
                                   jax.ShapeDtypeStruct((2, 128, 128), jnp.bfloat16),
                                   jax.ShapeDtypeStruct((2,), jnp.int32), out_dtype="bfloat16",
                                   trans_rhs=False, interpret=False)
    assert not grouped_matmul.Gmm().eligible(ba, target)
