"""The grouped GEMM template's kernels and their choice: Dew's Mosaic GPU grouped matmuls for Hopper
(`dew.nn.kernels.templates.gemm.mosaic_gpu`), interpreted on the CPU, each output element held to its
exact products (check 2's native GEMM contract, tests/native_certificates.py); the MoE dispatcher's
implementations; and the cost model (`dew.nn.kernels.templates.cost`) on analytic launches."""

import dataclasses
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from certified import lane
from generation_support import INTERPRETS, interpreted_projection, lowers, runs
from jax.sharding import Mesh
from native_certificates import (
    forward_products,
    input_gradient_products,
    kernel_gradient_products,
    native_dot_holds,
)
from op_certificates import BFLOAT16, FLOAT16, FLOAT32
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import gemm, generation, grouped_matmul, tuning
from dew.nn.kernels.templates.gemm import mosaic_gpu
from dew.nn.moe import expert_projection

pytestmark = pytest.mark.kernel

SIZES = {"ragged": [0, 70, 1, 100, 0], "one-group": [200], "all-empty": [0, 0, 0], "aligned": [64, 64, 64, 8],
         "unrouted-tail": [3, 0, 120], "tile-boundaries": [100, 0, 128, 129, 3, 40]}
ROWS, DEPTH, WIDTH = 200, 128, 256
"""The rows a draw holds at least (`operands`): its groups' rows where they are more (`tile-boundaries`: 400,
groups that start inside a 128-row tile, end past one, and an empty one between)."""
SEEDS = 8
"""Each site's declared draws (`reference_error.seeds_for` of the plain program on them)."""


def rounded(values, dtype) -> np.ndarray:
    return np.asarray(jnp.asarray(values, dtype), np.float64)


def operands(sizes, dtype, seed=0):
    """Rows (`ROWS`, or the groups' where they are more), per-group matrices and a cotangent, rounded to
    `dtype`, and the group of each row (-1 past the routed ones)."""
    rng = np.random.default_rng(seed)
    rows = max(ROWS, sum(sizes))
    x = rounded(rng.normal(size=(rows, DEPTH)), dtype)
    w = rounded(rng.normal(size=(len(sizes), DEPTH, WIDTH)) / np.sqrt(DEPTH), dtype)
    dy = rounded(rng.normal(size=(rows, WIDTH)), dtype)
    group = np.full(rows, -1)
    group[:sum(sizes)] = np.repeat(np.arange(len(sizes)), sizes)
    return x, w, dy, group


@INTERPRETS
@pytest.mark.parametrize("sizes", list(SIZES), ids=list(SIZES))
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_the_hopper_grouped_matmuls_sum_each_groups_rows_as_exactly_as_xla(sizes, dtype):
    """Both orientations of the forward, and the kernel gradient, interpreted on a CPU, at row blocks
    groups share and blocks of their own: every element a value its exact products admit summed in fp32
    and rounded once (check 2's native GEMM contract, per element, so one draw holds every element),
    unrouted rows and empty groups exactly zero."""
    sizes = SIZES[sizes]
    x, w, dy, group = operands(sizes, dtype)
    fmt = FORMAT[dtype]
    forward = forward_products(x[:sum(sizes)], w, sizes, fmt)
    kernel_gradient = kernel_gradient_products(x, dy, sizes, fmt)
    arrays = [jnp.asarray(value, dtype) for value in (x, w, dy)]
    group_sizes = jnp.asarray(sizes, jnp.int32)
    # One warpgroup a tile at two row blocks, and two (CUTLASS's cooperative schedule; a pair of
    # multiprocessors the interpreter does not model, the GPU's tile gate runs).
    tiled = [mosaic_gpu.Tiles(64, 64, 64, 2, 1, family='wgmma'),
             mosaic_gpu.Tiles(128, 64, 64, 2, 1, family='wgmma'),
             mosaic_gpu.Tiles(128, 64, 64, 2, 1, warpgroups=2, family='wgmma'),
             mosaic_gpu.Tiles(128, 256, 64, 2, 1, warpgroups=2, family='wgmma')]
    for tiles, transpose in itertools.product(tiled, (False, True)):
        matrix = jnp.swapaxes(arrays[1], 1, 2) if transpose else arrays[1]
        got = mosaic_gpu.gmm(arrays[0], matrix, group_sizes, tiles=tiles, transpose_rhs=transpose,
                             out_dtype=dtype, multiprocessors=1, interpret=grouped_matmul.INTERPRETED)
        routed(got, group, forward, fmt, f"forward {tiles} transpose={transpose}")
    for tiles in (mosaic_gpu.Tiles(64, 64, 64, 2, 1, family='wgmma'),
                  mosaic_gpu.Tiles(64, 64, 128, 2, 1, family='wgmma'),
                  mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma'),
                  mosaic_gpu.Tiles(128, 256, 64, 2, 1, warpgroups=2, family='wgmma')):
        got = mosaic_gpu.tgmm(arrays[0], arrays[2], group_sizes, tiles=tiles, out_dtype=jnp.float32,
                              multiprocessors=1, interpret=grouped_matmul.INTERPRETED)
        for index, products in enumerate(kernel_gradient):
            held(got[index], products, FLOAT32, f"kernel gradient {tiles} of group {index}")


FORMAT = {jnp.bfloat16: BFLOAT16, jnp.float16: FLOAT16, jnp.float32: FLOAT32}


def held(got, products, out, label):
    """Every element of `got` a value its exact `products` admit, in `out` (`native_dot_holds`), at the
    lane's flushing (an interpreted kernel's sums are XLA's on the CPU)."""
    found = native_dot_holds(np.asarray(got, np.float64), products, out, lane().flushes)
    assert np.all(found), f"{label}: {np.size(found) - np.count_nonzero(found)} elements past their products"


def routed(got, group, products, out, label):
    """A grouped product's routed rows held to their exact `products` (`held`), its unrouted rows zero."""
    got = np.asarray(got, np.float64)
    assert not np.any(got[group < 0]), f"{label}: an unrouted row is not zero"
    held(got[group >= 0], products, out, label)


@INTERPRETS
@pytest.mark.parametrize("tiles", [mosaic_gpu.Tiles(64, 64, 64, 2, 1, family='wgmma'),
                                   mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')],
                         ids=["one warpgroup", "cooperative"])
def test_the_kernel_gradient_keeps_a_neighbours_non_finite_rows_out(tiles):
    """For each group, every other row (its neighbours' and the unrouted tail) holds inf in the rows and
    NaN in the cotangent: the group's kernel gradient sums its own rows alone, each element a value its
    clean rows' exact products admit, one warpgroup a tile or two zeroing the stage both read (boundaries
    fall in the first and the last row blocks)."""
    sizes = [37, 100, 0, 40]
    x, _, dy, group = operands(sizes, jnp.bfloat16, seed=3)
    kernel_gradient = kernel_gradient_products(x, dy, sizes)
    for kept, size in enumerate(sizes):
        if not size:
            continue
        dirty_x, dirty_dy = x.copy(), dy.copy()
        dirty_x[group != kept], dirty_dy[group != kept] = np.inf, np.nan
        got = mosaic_gpu.tgmm(jnp.asarray(dirty_x, jnp.bfloat16), jnp.asarray(dirty_dy, jnp.bfloat16),
                              jnp.asarray(sizes, jnp.int32), tiles=tiles, out_dtype=jnp.float32,
                              multiprocessors=1, interpret=grouped_matmul.INTERPRETED)[kept]
        held(got, kernel_gradient[kept], FLOAT32, f"kernel gradient of group {kept}")


@lowers("sm90")
@pytest.mark.parametrize("launch,trans_rhs", [(grouped_matmul.MosaicGmm(), False),
                                              (grouped_matmul.MosaicGmm(), True),
                                              (grouped_matmul.MosaicTgmm(), None)],
                         ids=["MosaicGmm", "MosaicGmm-transposed", "MosaicTgmm"])
def test_the_hopper_grouped_matmuls_lower_for_an_h100_at_their_tile_and_a_narrow_pin(launch, trans_rhs):
    """The tile each Mosaic GPU launch derives at an expert layer's shapes (`Template.derived`), and the
    narrowest cooperative tile a pin may name there, of another block shape and schedule, lower for an
    H100 on the CPU, as a GPU session's compile would, over rows no multiple of 8; the forward's on a pair
    of multiprocessors and on one (its matrix's blocks multicast between the pair). The interpreter takes
    kernels Mosaic GPU's layout inference and TMA descriptors refuse; the GPU's tile gate runs each one
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
        ba, target = launch.bind(*args, **options), generation.gpu_target()
        derived = launch.derived(ba, target)
        narrow = mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')
        assert derived is not None and launch.legal(ba, narrow, target)
        # The forward on a pair of multiprocessors as well (its matrix's blocks multicast), whatever the
        # derivation picks (row 4491).
        pairs = 1 if isinstance(launch, grouped_matmul.MosaicTgmm) else 2
        paired = dataclasses.replace(narrow, cluster=pairs)
        assert launch.legal(ba, paired, target)
        for config in {derived, narrow, paired}:
            jax.jit(functools.partial(launch.replace(config=config), **options)).trace(*args).lower(
                lowering_platforms=("cuda",))


def test_the_portable_branch_writes_each_gradient_from_the_rounded_operands():
    """The projection's XLA route (a CPU's, which runs no kernel) differentiates the operands as it rounded
    them: an fp16 master's largest value rounds to bf16's 65536, which fp16 does not hold, and the input
    gradient through it is finite, each gradient rounded once to its operand's dtype."""
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
    """The projection's kernels, interpreted on a CPU as the named generation runs them
    (`interpreted_projection`), each launch the
    implementation `generation.chosen` names: its forward and both gradients each element a value its
    exact products admit (check 2's native GEMM contract), the input gradient in the input's dtype and the
    kernel gradient in the master's."""
    sizes = [0, 70, 1, 100, 0]
    x, w, dy, group = operands(sizes, jnp.bfloat16, seed=5)
    group_sizes = jnp.asarray(sizes, jnp.int32)

    def products(x, w):
        y, pull = jax.vjp(lambda x, w: interpreted_projection(x, w, group_sizes, jnp.bfloat16), x, w)
        return (y, *pull(jnp.asarray(dy, jnp.bfloat16)))

    with generation.assumed(generation_name), generation.chosen(grouped_matmul=implementation):
        operands_ = (jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.float32))
        # Mosaic's interpreter's ordered callbacks run eagerly: a jitted `linear_call` cannot carry them.
        y, dx, dw = (products if implementation == "mosaic_gpu" else jax.jit(products))(*operands_)
    assert (y.dtype, dx.dtype, dw.dtype) == (jnp.bfloat16, jnp.bfloat16, jnp.float32)
    routed(y, group, forward_products(x[:sum(sizes)], w, sizes), BFLOAT16, "y")
    routed(dx, group, input_gradient_products(dy[:sum(sizes)], w, sizes), BFLOAT16, "dx")
    for index, held_products in enumerate(kernel_gradient_products(x, dy, sizes)):
        held(dw[index], held_products, FLOAT32, f"dw of group {index}")


def cutlass_runs_the_projection_as_exactly_as_xla_where_chosen(sizes):
    """Where the installed dew-cutlass holds this GPU's kernels, the projection's forward and both
    gradients through CUTLASS's launches, chosen by name, are as exact as XLA's ragged dot from float64,
    unrouted rows and empty groups included."""
    # The wheel is optional: a process without it, or whose wheel holds no kernel for its GPU, runs the
    # template's launches, which the suite checks apart.
    here = generation.gpu_target().generation
    if here not in grouped_matmul.cutlass_architectures():
        pytest.skip(f"no installed dew-cutlass holds kernels for {here}")
    group_sizes = jnp.asarray(SIZES[sizes], jnp.int32)

    def draw(seed):
        x, w, dy, _ = operands(SIZES[sizes], jnp.bfloat16, seed)
        return tuple(jnp.asarray(value, jnp.bfloat16) for value in (x, w, dy))

    def projection(implementation, dtype=jnp.bfloat16):
        return lambda x, w: expert_projection(x, w, group_sizes, dtype, implementation, None)

    def through_cutlass(x, w, dy):
        with generation.chosen(grouped_matmul="cutlass"):
            return with_gradients(projection("pallas"))(x, w, dy)

    assert_as_exact(jax.jit(through_cutlass), jax.jit(with_gradients(projection("xla"))), draw,
                    f"cutlass {sizes}", seeds=SEEDS, truth_fn=with_gradients(projection("xla", jnp.float64)))


def the_template_sums_16_bit_products_into_fp32_as_exactly_as_xla(depth):
    """On this GPU the template's grouped product of bfloat16 rows and matrices into an fp32 output (its
    route named, `mosaic_gpu`), forward and both gradients, is as exact as XLA's ragged dot from float64 at a
    `depth`-deep contraction: its sums promoted a step at a time (`templates.product.promoted`), where the
    tensor cores' in-order fp32 accumulation was 1.661x XLA's error at depth 128 and 3.88x at 512 on an A100
    (M-NS-1, NS4331-VALUE4-a100; row 4331)."""
    group_sizes = jnp.asarray([70, 66], jnp.int32)

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 3)
        return (jax.random.normal(keys[0], (136, depth)).astype(jnp.bfloat16),
                (jax.random.normal(keys[1], (2, depth, 256)) / np.sqrt(depth)).astype(jnp.bfloat16),
                jax.random.normal(keys[2], (136, 256)).astype(jnp.float32))

    def product(dtype):
        return lambda x, w: gemm.gemm(x.astype(dtype), w.astype(dtype), group_sizes, out_dtype=jnp.float32)

    def through_template(x, w, dy):
        with generation.chosen(grouped_matmul="mosaic_gpu"):
            return with_gradients(product(jnp.bfloat16))(x, w, dy)

    def through_xla(x, w, dy):
        with generation.chosen(grouped_matmul="xla"):
            return with_gradients(product(jnp.bfloat16))(x, w, dy)

    assert_as_exact(jax.jit(through_template), jax.jit(through_xla), draw,
                    f"16-bit products into fp32 {depth}", seeds=SEEDS,
                    truth_fn=with_gradients(product(jnp.float64)))
def the_template_takes_a_gpt_oss_width_as_exactly_as_xla():
    """On this GPU the template named (`mosaic_gpu`) for a projection 2880 wide, gpt-oss-120b's (every tile
    width dividing it an odd multiple of 64), its groups' rows odd, so each group's last tile stores a
    one-row rung (an A100's in a row's layout, `mosaic.copied`: row 4474): its forward and both gradients
    are as exact as XLA's ragged dot, from float64."""
    group_sizes = jnp.asarray([67, 69], jnp.int32)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rounded(values, jnp.bfloat16), jnp.bfloat16)
                     for values in (rng.normal(size=(136, 256)), rng.normal(size=(2, 256, 2880)) / 16,
                                    rng.normal(size=(136, 2880))))

    def projection(implementation, dtype=jnp.bfloat16):
        return with_gradients(lambda x, w: expert_projection(x, w, group_sizes, dtype, implementation, None))

    def through_template(x, w, dy):
        with generation.chosen(grouped_matmul="mosaic_gpu"):
            return projection("pallas")(x, w, dy)

    assert_as_exact(jax.jit(through_template), jax.jit(projection("xla")), draw, "template gpt-oss width",
                    seeds=SEEDS, truth_fn=projection("xla", jnp.float64))


def cutlass_runs_a_mapped_projection_as_exactly_as_xla_where_chosen(shared):
    """Where the installed dew-cutlass holds this GPU's kernels, a projection mapped over a batch of two
    (`jax.vmap`: each its rows and cotangent, and its matrices unless `shared`), its forward and both
    gradients through CUTLASS's launches chosen by name, each map one launch in turn (their FFI calls'
    sequential batching, row 4389), are as exact as XLA's ragged dot mapped, from float64."""
    here = generation.gpu_target().generation
    if here not in grouped_matmul.cutlass_architectures():
        pytest.skip(f"no installed dew-cutlass holds kernels for {here}")
    group_sizes = jnp.asarray(SIZES["ragged"], jnp.int32)

    def draw(seed):
        first, second = (operands(SIZES["ragged"], jnp.bfloat16, seed + offset)[:3]
                         for offset in (0, 1 << 16))
        x, w, dy = (jnp.stack([jnp.asarray(a, jnp.bfloat16), jnp.asarray(b, jnp.bfloat16)])
                    for a, b in zip(first, second, strict=True))
        return x, w[0] if shared else w, dy

    def mapped(implementation, dtype=jnp.bfloat16):
        one = with_gradients(lambda x, w: expert_projection(x, w, group_sizes, dtype, implementation, None))
        return jax.vmap(one, in_axes=(0, None if shared else 0, 0))

    def through_cutlass(x, w, dy):
        with generation.chosen(grouped_matmul="cutlass"):
            return mapped("pallas")(x, w, dy)

    assert_as_exact(jax.jit(through_cutlass), jax.jit(mapped("xla")), draw,
                    f"cutlass mapped{' shared' if shared else ''}", seeds=SEEDS,
                    truth_fn=mapped("xla", jnp.float64))


@pytest.mark.parametrize("generation_name", ["sm90", "sm100"])
def test_tritons_tile_counts_every_stage_where_its_mma_reads_shared_memory_asynchronously(generation_name):
    """Pallas Triton's pipeline holds all its stages where the MMA reads shared memory asynchronously
    (Hopper's wgmma, Blackwell's tcgen05), so the dense fp32 product of the exact-sums cell (64 rows, depth
    4096, 128 columns, at the highest precision) derives a tile whose `stages` steps fit the device's shared
    memory: a B200 refused 262,160 bytes for the tile counted at `stages - 1` (row 4449)."""
    launch = grouped_matmul.Gmm()
    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    with generation.assumed(generation_name):
        target = generation.gpu_target()
        ba = launch.bind(f32((64, 4096)), f32((1, 4096, 128)), jax.ShapeDtypeStruct((1,), jnp.int32),
                         out_dtype="float32", trans_rhs=False, interpret=False, dense=True, exact_sums=True)
        tile = launch.derived(ba, target)
        assert tile is not None
        step = launch.loaded(tile) * 4
        assert launch.staged_bytes(ba, tile, target) >= tile.stages * step
        assert tile.stages * step <= generation.shared_budget(target)


@pytest.mark.parametrize("product", ["forward", "input gradient", "kernel gradient"])
def test_an_amperes_gpt_oss_projection_routes_to_a_kernel_that_lowers(product):
    """gpt-oss-120b's expert projections (2880 wide and deep) on an A100, the selector unnamed: with the input
    gradient's declared Triton cell gone (505803b8e2) each product routes to a kernel that takes it and
    lowers for CUDA, never a refusal; on an A100 itself the template refused the forward it was named for
    (NS2295-CELLS2-a100, row 4474), where the unnamed call takes the next kernel."""
    sizes = jax.ShapeDtypeStruct((4,), jnp.int32)
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    x, w, dy = bf16((512, 2880)), bf16((4, 2880, 2880)), bf16((512, 2880))
    calls = {"forward": lambda x, w, dy, sizes: gemm.gemm(x, w, sizes, out_dtype=jnp.bfloat16),
             "input gradient": lambda x, w, dy, sizes: gemm.gemm(dy, w, sizes, transpose_rhs=True,
                                                                 out_dtype=jnp.bfloat16),
             "kernel gradient": lambda x, w, dy, sizes: gemm.gemm_t(x, dy, sizes, out_dtype=jnp.bfloat16)}
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
    with generation.assumed("sm80"), jax.sharding.use_abstract_mesh(mesh), generation.routes() as routes:
        jax.jit(calls[product]).trace(x, w, dy, sizes).lower(lowering_platforms=("cuda",))
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"}, routes


def ragged_draw(seed):
    """The ragged sizes' rows, matrices and cotangent in bf16, at a seed (`operands`)."""
    x, w, dy, _ = operands(SIZES["ragged"], jnp.bfloat16, seed)
    return tuple(jnp.asarray(value, jnp.bfloat16) for value in (x, w, dy))


UNSEEN = {
    "ampere-class": generation.GpuTarget("sm80", "Unmeasured Ampere-class GPU", shared_limit=99 * 1024,
                                         multiprocessors=46, clock_hz=1.4e9, memory_bandwidth=0.45e12),
    "hopper-class": generation.GpuTarget("sm90", "Unmeasured Hopper-class GPU", shared_limit=180 * 1024,
                                         multiprocessors=78, clock_hz=1.6e9, memory_bandwidth=2.4e12),
}
"""GPUs no row measured, one of each family: an Ampere-class part with fewer multiprocessors and the
smaller block of an sm86 than an A100's, and a Hopper-class one with fewer multiprocessors, a smaller
block and another L2 bandwidth than an H100's."""


def test_a_program_traced_for_one_gpu_traces_again_for_another_of_its_family():
    """Two GPUs of one family whose facts differ key a jitted program's traces apart
    (`generation.assumed`), so neither's launches are sized and picked by the other's facts; the same
    facts, stated again, share one trace."""
    traced = []

    @jax.jit
    def program(x):
        traced.append(generation.gpu_target())
        return x + 1

    measured = dataclasses.replace(UNSEEN["hopper-class"], memory_bandwidth=3.0e12)
    restated = dataclasses.replace(measured)
    targets = (UNSEEN["hopper-class"], generation.TARGETS["sm90"], measured, restated)
    for target in targets:
        with generation.assumed(target):
            program(jnp.zeros(2))
    assert traced == list(targets[:3])


@INTERPRETS
@pytest.mark.parametrize("unseen", UNSEEN)
def test_a_gpu_no_one_measured_runs_the_tiles_its_facts_give_exactly(unseen):
    """On a GPU no row measured, each Mosaic GPU launch's tile is its family's, derived from the
    family's facts, the GPU's own and the call's extents, and fits the GPU's block, and the call takes
    it. Named there, the projection's forward and
    its gradients through them (the matrix frozen where the family runs no kernel gradient) are as exact
    as XLA's ragged dot from float64."""
    target = UNSEEN[unseen]
    family = generation.family(target.generation)

    def products(rows, depth, width, groups):
        # The forward, and its input and kernel gradients as the reverse pass launches them, each alone.
        sizes = jnp.full((groups,), rows // groups, jnp.int32)
        forward = functools.partial(gemm.gemm, group_sizes=sizes, out_dtype=jnp.bfloat16,
                                    precision="default", interpret_on_cpu=True)
        bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
        x, w, dy = bf16((rows, depth)), bf16((groups, depth, width)), bf16((rows, width))
        named = [(forward, (x, w)), (functools.partial(forward, transpose_rhs=True), (dy, w))]
        if target.wgmma:
            named.append((functools.partial(gemm.gemm_t, group_sizes=sizes, out_dtype=jnp.bfloat16,
                                            interpret_on_cpu=True), (x, dy)))
        return named

    # At a Qwen3 MoE projection's extents, traced (never run): each launch takes the tile it derives on
    # the GPU's facts.
    budget = generation.shared_budget(target)
    with (generation.assumed(target), generation.chosen(grouped_matmul="mosaic_gpu"),
          tuning.seen() as launches):
        for product, shapes in products(65536, 2048, 1536, 128):
            jax.jit(product).trace(*shapes)
        for (launch, _), (tile, ba) in zip(launches, launches.taken, strict=True):
            if not isinstance(launch, grouped_matmul._Mosaic):
                continue
            assert tile.family == family and tile in launch.fitting(ba, target)
            assert launch.staged_bytes(ba, tile, target) <= budget
            assert tile == launch.derived(ba, target)

    sizes = SIZES["ragged"]
    group_sizes = jnp.asarray(sizes, jnp.int32)


    def held(w):
        # Where the family runs no kernel gradient, the matrix is frozen: the template forms no gradient
        # of it (a frozen operand's cotangent is never formed).
        return w if target.wgmma else jax.lax.stop_gradient(w)

    def projection(dtype=jnp.bfloat16):
        return lambda x, w: expert_projection(x, held(w), group_sizes, dtype, "xla", None)

    def through_mosaic(x, w, dy):
        with generation.assumed(target), generation.chosen(grouped_matmul="mosaic_gpu"):
            return with_gradients(lambda x, w: gemm.gemm(x, held(w), group_sizes, out_dtype=jnp.bfloat16,
                                                             precision="default",
                                                             interpret_on_cpu=True))(x, w, dy)

    # Eagerly: the Mosaic interpreter's ordered callbacks are not a jitted `linear_call`'s to carry.
    with tuning.seen() as launches:
        assert_as_exact(through_mosaic, jax.jit(with_gradients(projection())), ragged_draw,
                        f"unseen {unseen}", seeds=SEEDS, truth_fn=with_gradients(projection(jnp.float64)))
    mosaic = [(launch, ba, tile) for (launch, _), (tile, ba) in zip(launches, launches.taken, strict=True)
              if isinstance(launch, grouped_matmul._Mosaic)]
    with generation.assumed(target):
        assert all(tile == launch.derived(ba, target) for launch, ba, tile in mosaic)


@INTERPRETS
def test_a_frozen_matrix_forms_no_kernel_gradient():
    """A projection whose matrix is frozen (no reverse pass reaches it): its input gradient through the Mosaic
    launches is as exact as XLA's from float64."""
    sizes = SIZES["ragged"]
    group_sizes = jnp.asarray(sizes, jnp.int32)


    def frozen(product):
        return lambda x, w: product(x, jax.lax.stop_gradient(w))

    def projection(dtype=jnp.bfloat16):
        return frozen(lambda x, w: expert_projection(x, w, group_sizes, dtype, "xla", None))

    def through_mosaic(x, w, dy):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"):
            return with_gradients(frozen(lambda x, w: gemm.gemm(x, w, group_sizes, out_dtype=jnp.bfloat16,
                                                                    interpret_on_cpu=True)))(x, w, dy)

    # Eagerly: the Mosaic interpreter's ordered callbacks are not a jitted `linear_call`'s to carry.
    assert_as_exact(through_mosaic, jax.jit(with_gradients(projection())), ragged_draw, "frozen matrix",
                    seeds=SEEDS, truth_fn=with_gradients(projection(jnp.float64)))


@runs("sm80")
def test_the_kernels_forward_mode_keeps_the_projections_law():
    """An expert projection's tangent through the template (its launches interpreted, as an A100 runs
    them), and the XLA projection's, each keep the projection's law `dx @ Q(kernel) + Q(x) @ dkernel` with
    the tangents unrounded, where a cast's own tangent would round an fp32 input's to bf16 first: two
    products of an fp32 1 + 2^-8 and 1.5 sum to 3.01171875, which bf16 rounds to 3.015625 (rounded first,
    3), every value exact."""
    sizes = jnp.asarray([1, 2], jnp.int32)
    x, w = jnp.zeros((3, 2), jnp.float32), jnp.full((2, 2, 2), 1.5, jnp.float32)
    dx, dw = jnp.full((3, 2), 1 + 2.0 ** -8, jnp.float32), jnp.full((2, 2, 2), 2.0 ** -9, jnp.float32)

    def tangent(implementation):
        return jax.jit(lambda x, w, dx, dw: jax.jvp(lambda x, w: expert_projection(
            x, w, sizes, jnp.bfloat16, implementation, None), (x, w), (dx, dw))[1])(x, w, dx, dw)

    with generation.assumed("sm80"):
        kernels = tangent("pallas")
    for found in (kernels, tangent("xla")):
        np.testing.assert_array_equal(np.asarray(found, np.float32), np.full((3, 2), 3.015625, np.float32))


@runs("sm80")
def test_a_call_of_no_rows_gives_its_empty_value():
    """A projection of no rows on an A100 gives its empty value."""
    x, w = jnp.ones((0, 16), jnp.bfloat16), jnp.ones((2, 16, 16), jnp.float32)
    sizes = jnp.zeros((2,), jnp.int32)
    with generation.assumed("sm80"):
        y = expert_projection(x, w, sizes, jnp.bfloat16, "pallas", None)
    assert y.shape == (0, 16)

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


@pytest.mark.parametrize("block_m,block_n", [(64, 128), (128, 512)])
def test_a_cooperative_tile_refuses_what_its_consumers_cannot_compute(block_m, block_n):
    """A cooperative tile's two consumers each take half its rows in warpgroup MMAs (a multiple of 64
    rows by at most 256 columns): a tile whose halves are not that refuses."""
    with pytest.raises(ValueError):
        mosaic_gpu.Tiles(block_m, block_n, 64, 2, 1, warpgroups=2, family='wgmma')


@INTERPRETS
def test_a_tile_of_192_columns_stages_every_column_it_writes():
    """A consumer stages its bf16 output in chunks of at most 128 columns that divide its tile: a 192-column
    tile, one warpgroup or two, writes all of its columns, each a value its exact products admit."""
    sizes = SIZES["ragged"]
    x, w, _, group = operands(sizes, jnp.bfloat16)
    wide = np.tile(w, (1, 1, 2))[..., :192]
    forward = forward_products(x[:sum(sizes)], wide, sizes)
    for warpgroups in (1, 2):
        tiles = mosaic_gpu.Tiles(128, 192, 64, 2, 1, warpgroups=warpgroups, family='wgmma')
        got = mosaic_gpu.gmm(jnp.asarray(x, jnp.bfloat16), jnp.asarray(wide, jnp.bfloat16),
                             jnp.asarray(sizes, jnp.int32), tiles=tiles, transpose_rhs=False,
                             out_dtype=jnp.bfloat16, multiprocessors=1, interpret=grouped_matmul.INTERPRETED)
        routed(got, group, forward, BFLOAT16, f"forward {tiles}")


AMPERE_CLASS = jax.default_backend() == "cpu" or (
    jax.default_backend() == "gpu" and not generation.gpu_target().wgmma)
"""Whether this process runs mma.sync's kernels: a CPU interprets them as an A100's; a GPU without warpgroup
MMA (an A100, an A10, an L4) runs them on its own facts, where one with warpgroup MMA lowers them for its
own device, whose pipelines its Mosaic sizes past its block (SM80-h100)."""


def ampere(generation_name: str = "sm80"):
    """The assumed device an mma.sync test runs on: the process's own Ampere-class GPU, else
    `generation_name`'s, which a CPU interprets."""
    return generation.assumed(None if jax.default_backend() == "gpu" else generation_name)


@pytest.mark.skipif(not AMPERE_CLASS, reason="no mma.sync kernel this GPU and JAX lower")
@pytest.mark.parametrize("out_dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("sizes", [name for name in SIZES if any(SIZES[name])])
def test_an_ampere_gpus_mosaic_and_triton_launches_are_each_as_exact_as_xla(sizes, out_dtype):
    """Without warpgroup MMA, the Mosaic GPU forward's mma.sync tile (`mosaic_gpu.derived`) and Triton's
    launch, each named: every routed element a value its exact products admit, rounded once to its output
    dtype, unrouted rows zero."""
    x, w, _, group = operands(SIZES[sizes], jnp.bfloat16)
    products = forward_products(x[:sum(SIZES[sizes])], w, SIZES[sizes])
    group_sizes = jnp.asarray(SIZES[sizes], jnp.int32)
    lhs, rhs = jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.bfloat16)

    def forward(x, w):
        # The default precision: the suite's highest asks each step's sum apart, which mma.sync's tiles do
        # not add.
        return gemm.gemm(x, w, group_sizes, out_dtype=out_dtype, precision="default",
                             interpret_on_cpu=True)

    for named in ("triton", "mosaic_gpu"):
        with ampere(), generation.chosen(grouped_matmul=named):
            got = jax.jit(forward)(lhs, rhs)
        assert got.dtype == out_dtype
        routed(got, group, products, FORMAT[out_dtype],
               f"sm80 forward through {named} to {jnp.dtype(out_dtype)}")


@pytest.mark.skipif(not AMPERE_CLASS, reason="no mma.sync kernel this GPU and JAX lower")
def test_an_mma_sync_forward_of_no_depth_is_zero():
    """A product of no depth through mma.sync's tile, named on an Ampere GPU, is zero: it sums nothing,
    and no launch runs."""
    lhs, rhs = jnp.zeros((128, 0), jnp.bfloat16), jnp.zeros((1, 0, 256), jnp.bfloat16)
    with ampere(), generation.chosen(grouped_matmul="mosaic_gpu"):
        got = jax.jit(lambda x, w: gemm.gemm(x, w, jnp.asarray([128], jnp.int32), out_dtype=jnp.float32,
                                                 precision="default", interpret_on_cpu=True))(lhs, rhs)
    assert not np.any(np.asarray(got))


@pytest.mark.parametrize("sizes", [(3, 2), (2, 2)])
def test_a_grouped_epilogue_of_no_depth_reads_each_rows_group(sizes):
    """A grouped product of no depth runs no launch, its zeros through the epilogue on whole arrays: its side
    operands by group read at each row as XLA's program reads them (`axes.at_rows`, row 4489), each row its
    group's bias, a row past the groups' the last's; a side operand by column as it is."""
    bias = jnp.asarray(np.arange(2 * 128).reshape(2, 128), jnp.float32)
    column = jnp.asarray(np.linspace(-1, 1, 128), jnp.float32)
    def added(acc, by_group, by_column):
        return acc + by_group + by_column

    tiles = mosaic_gpu.Tiles(128, 128, 64, 2, 1, family="wgmma")
    got = mosaic_gpu.gmm(jnp.zeros((5, 0), jnp.bfloat16), jnp.zeros((2, 0, 128), jnp.bfloat16),
                         jnp.asarray(sizes, jnp.int32), tiles=tiles, transpose_rhs=False,
                         out_dtype=jnp.float32, multiprocessors=1, epilogue=added,
                         epilogue_operands=(bias, column),
                         epilogue_axes=(("groups", "columns"), ("columns",)))
    # Each row's group, the rows past the groups' the last group's.
    groups = np.concatenate([np.repeat(np.arange(2), sizes), np.full(5 - sum(sizes), 1)])
    np.testing.assert_array_equal(np.asarray(got), np.asarray(bias)[groups] + np.asarray(column)[None])


def halved(acc, bias):
    return (acc + bias[None, :]).astype(jnp.bfloat16)


def doubled_halves(acc, bias):
    return (acc * 2).astype(jnp.bfloat16), (acc + bias[None, :]).astype(jnp.bfloat16)


def twice(acc, bias):
    del bias
    return (acc * 2).astype(jnp.bfloat16)


def test_a_cooperative_tile_stores_each_of_several_outputs():
    """A cooperative tile's consumers stage every output at once: a two-output epilogue's outputs, each its
    own staging buffer, on par with the plain program's two over the draws (check 1), truth the product
    and its epilogue in float64 (eagerly: the interpreter's ordered callbacks)."""
    bias = jnp.asarray(np.linspace(-1, 1, WIDTH), jnp.float32)
    sizes = jnp.asarray(SIZES["ragged"], jnp.int32)
    tiles = mosaic_gpu.Tiles(128, 256, 64, 2, 1, warpgroups=2, family='wgmma')

    def draw(seed):
        x, w, _, _ = operands(SIZES["ragged"], jnp.bfloat16, seed)
        return jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.bfloat16)

    def run(x, w):
        return mosaic_gpu.gmm(x, w, sizes, tiles=tiles, transpose_rhs=False, out_dtype=jnp.bfloat16,
                              multiprocessors=1, interpret=grouped_matmul.INTERPRETED,
                              epilogue=doubled_halves, epilogue_operands=(bias,),
                              epilogue_axes=(("columns",),), out_dtypes=(jnp.bfloat16, jnp.bfloat16))

    def plain(x, w):
        acc = jax.lax.ragged_dot(x, w, sizes, preferred_element_type=jnp.float32)
        return twice(acc, bias), halved(acc, bias)

    def truth(x, w):
        acc = jax.lax.ragged_dot(x, w, sizes, preferred_element_type=x.dtype)
        return acc * 2, acc + bias[None, :]

    assert_as_exact(run, jax.jit(plain), draw, "a cooperative tile's two outputs", seeds=SEEDS,
                    truth_fn=truth)


@pytest.mark.parametrize("sizes, row_tile", [([120, 8], 64), ([3, 200, 1, 0, 57], 128), ([0, 0, 13], 256),
                                             ([64, 64], 64), ([7] * 9, 128)])
def test_no_row_tile_reads_past_its_operands_allocation(sizes, row_tile):
    """Every row tile `gmm`'s schedule takes (`mosaic_gpu.tile_groups`, a group's tiles from its first row)
    reads within the rows its operands are allocated to (`mosaic_gpu.row_extent`): its first row plus the
    tile's rows, unpredicated on an A100, stay inside the extent (row 1866 K3: groups [120, 8] at 64-row
    tiles read rows 120 to 183 of a 128-row lhs)."""
    rows = -(-sum(sizes) // mosaic_gpu.TMA_ROWS) * mosaic_gpu.TMA_ROWS
    lengths = jnp.asarray([*sizes, rows - sum(sizes)], jnp.int32)
    table = np.asarray(mosaic_gpu.tile_groups(lengths, row_tile, -(-rows // row_tile) + len(sizes)))
    used = table[:, 3] > table[:, 2]
    reads = int((table[used, 1] + row_tile).max())
    assert reads <= mosaic_gpu.row_extent(rows, row_tile)
    if sizes == [120, 8]:
        assert reads > -(-rows // row_tile) * row_tile  # past the row-tile padding that held before
