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
from generation_support import INTERPRETS, generations, interpreted_projection, lowers, runs
from jax.sharding import Mesh
from reference_error import assert_as_exact, assert_as_exact_as_the_reference, equations, with_gradients

from dew.nn.kernels import generation, grouped_matmul, tuning
from dew.nn.kernels.templates.gemm import mosaic_gpu, template
from dew.nn.moe import expert_projection
from tools import grouped_oracle as oracle

pytestmark = pytest.mark.kernel

SIZES = {"ragged": [0, 70, 1, 100, 0], "one-group": [200], "all-empty": [0, 0, 0], "aligned": [64, 64, 64, 8],
         "unrouted-tail": [3, 0, 120]}
ROWS, DEPTH, WIDTH = 200, 128, 256
SEEDS = 8
"""Each site's declared draws (`reference_error.seeds_for` of the plain program on them)."""


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
             mosaic_gpu.Tiles(128, 64, 64, 2, 1, warpgroups=2),
             mosaic_gpu.Tiles(128, 256, 64, 2, 1, warpgroups=2)]
    for tiles, transpose in itertools.product(tiled, (False, True)):
        matrix = jnp.swapaxes(arrays[1], 1, 2) if transpose else arrays[1]
        got = mosaic_gpu.gmm(arrays[0], matrix, group_sizes, tiles=tiles, transpose_rhs=transpose,
                             out_dtype=dtype, multiprocessors=1, interpret=grouped_matmul.INTERPRETED)
        check(got, reference[0], truth[0], group, f"forward {tiles} transpose={transpose}")
    for tiles in (mosaic_gpu.Tiles(64, 64, 64, 2, 1), mosaic_gpu.Tiles(64, 64, 128, 2, 1),
                  mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2),
                  mosaic_gpu.Tiles(128, 256, 64, 2, 1, warpgroups=2)):
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
@pytest.mark.parametrize("tiles", [mosaic_gpu.Tiles(64, 64, 64, 2, 1),
                                   mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2)],
                         ids=["one warpgroup", "cooperative"])
def test_the_kernel_gradient_keeps_a_neighbours_non_finite_rows_out(tiles):
    """For each group, every other row (its neighbours' and the unrouted tail) holds inf in the rows and
    NaN in the cotangent: the group's kernel gradient sums its own rows alone, finite and as exact as
    XLA's over clean rows, one warpgroup a tile or two zeroing the stage both read (boundaries fall in
    the first and the last row blocks)."""
    sizes = [37, 100, 0, 40]
    x, w, dy, group = operands(sizes, jnp.bfloat16, seed=3)
    truth = oracle.kernel_gradient(x, dy, sizes)
    reference = xla_products(x, w, dy, sizes, jnp.bfloat16, jnp.float32)[2]
    for kept, size in enumerate(sizes):
        if not size:
            continue
        dirty_x, dirty_dy = x.copy(), dy.copy()
        dirty_x[group != kept], dirty_dy[group != kept] = np.inf, np.nan
        got = mosaic_gpu.tgmm(jnp.asarray(dirty_x, jnp.bfloat16), jnp.asarray(dirty_dy, jnp.bfloat16),
                              jnp.asarray(sizes, jnp.int32), tiles=tiles, out_dtype=jnp.float32,
                              multiprocessors=1, interpret=grouped_matmul.INTERPRETED)[kept]
        assert np.all(np.isfinite(np.asarray(got))), f"group {kept}: a neighbour's row reached it"
        assert_as_exact_as_the_reference(np.asarray(got, np.float64), np.asarray(reference[kept], np.float64),
                                          truth[kept], f"kernel gradient of group {kept}")


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
        derived, narrow = launch.derived(ba, target), mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2)
        assert launch.legal(ba, narrow, target) and derived != narrow
        # The forward's rows, 256 a group, span two of its row tiles: a pair (`mosaic_gpu.derived`).
        pairs = 1 if isinstance(launch, grouped_matmul.MosaicTgmm) else 2
        assert derived is not None and derived.cluster == pairs
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
    implementation `generation.chosen` names: its forward and both gradients as exact as XLA's, the input
    gradient in the input's dtype and the kernel gradient in the master's, each launch's route the one
    named."""
    sizes = [0, 70, 1, 100, 0]
    x, w, dy, _ = operands(sizes, jnp.bfloat16, seed=5)
    forward, input_gradient, kernel_gradient = oracle.products(x, w, dy, sizes)
    reference = xla_products(x, w, dy, sizes, jnp.bfloat16, jnp.float32)
    group_sizes = jnp.asarray(sizes, jnp.int32)

    def products(x, w):
        y, pull = jax.vjp(lambda x, w: interpreted_projection(x, w, group_sizes, jnp.bfloat16), x, w)
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


@pytest.mark.gpu(reason="needs a GPU the installed dew-cutlass holds kernels for")
@pytest.mark.parametrize("sizes", [name for name in SIZES if any(SIZES[name])])
def test_cutlass_runs_the_projection_as_exactly_as_xla_where_chosen(sizes):
    """Where the installed dew-cutlass holds this GPU's kernels, the projection's forward and both
    gradients through CUTLASS's launches, chosen by name, are as exact as XLA's ragged dot from float64,
    unrouted rows and empty groups included, and the CUDA branch's launches are CUTLASS's."""
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

    with tuning.seen() as launches:
        assert_as_exact(jax.jit(through_cutlass), jax.jit(with_gradients(projection("xla"))), draw,
                        f"cutlass {sizes}", seeds=SEEDS,
                        truth_fn=with_gradients(projection("xla", jnp.float64)))
    # The CUDA branch's launches; the CPU branch beside it interprets Triton's.
    ran = {type(launch).__name__ for launch, ba in launches if not ba.arguments["interpret"]}
    assert ran == {"CutlassGmm", "CutlassTgmm"}


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
    # mma.sync's tiles run the product and the input gradient; the kernel gradient is warpgroup MMA's.
    launched = {"MosaicGmm", "MosaicTgmm"} if target.wgmma else {"MosaicGmm"}

    def products(rows, depth, width, groups):
        # The forward, and its input and kernel gradients as the reverse pass launches them, each alone.
        sizes = jnp.full((groups,), rows // groups, jnp.int32)
        forward = functools.partial(template.gemm, group_sizes=sizes, out_dtype=jnp.bfloat16,
                                    precision="default", interpret_on_cpu=True)
        bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
        x, w, dy = bf16((rows, depth)), bf16((groups, depth, width)), bf16((rows, width))
        named = [(forward, (x, w)), (functools.partial(forward, transpose_rhs=True), (dy, w))]
        if target.wgmma:
            named.append((functools.partial(template.gemm_t, group_sizes=sizes, out_dtype=jnp.bfloat16,
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
            return with_gradients(lambda x, w: template.gemm(x, held(w), group_sizes, out_dtype=jnp.bfloat16,
                                                             precision="default",
                                                             interpret_on_cpu=True))(x, w, dy)

    # Eagerly: the Mosaic interpreter's ordered callbacks are not a jitted `linear_call`'s to carry.
    with tuning.seen() as launches:
        assert_as_exact(through_mosaic, jax.jit(with_gradients(projection())), ragged_draw,
                        f"unseen {unseen}", seeds=SEEDS, truth_fn=with_gradients(projection(jnp.float64)))
    mosaic = [(launch, ba, tile) for (launch, _), (tile, ba) in zip(launches, launches.taken, strict=True)
              if isinstance(launch, grouped_matmul._Mosaic)]
    assert {type(launch).__name__ for launch, _, _ in mosaic} == launched
    with generation.assumed(target):
        assert all(tile == launch.derived(ba, target) for launch, ba, tile in mosaic)


@INTERPRETS
def test_a_frozen_matrix_forms_no_kernel_gradient():
    """A projection whose matrix is frozen (no reverse pass reaches it) forms no kernel gradient: no
    launch computes one, and its input gradient through the Mosaic launches is as exact as XLA's from
    float64."""
    sizes = SIZES["ragged"]
    group_sizes = jnp.asarray(sizes, jnp.int32)


    def frozen(product):
        return lambda x, w: product(x, jax.lax.stop_gradient(w))

    def projection(dtype=jnp.bfloat16):
        return frozen(lambda x, w: expert_projection(x, w, group_sizes, dtype, "xla", None))

    def through_mosaic(x, w, dy):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"):
            return with_gradients(frozen(lambda x, w: template.gemm(x, w, group_sizes, out_dtype=jnp.bfloat16,
                                                                    interpret_on_cpu=True)))(x, w, dy)

    # Eagerly: the Mosaic interpreter's ordered callbacks are not a jitted `linear_call`'s to carry.
    with tuning.seen() as launches:
        assert_as_exact(through_mosaic, jax.jit(with_gradients(projection())), ragged_draw, "frozen matrix",
                        seeds=SEEDS, truth_fn=with_gradients(projection(jnp.float64)))
    assert {type(launch).__name__ for launch, _ in launches} == {"MosaicGmm"}


@runs("sm80")
def test_the_kernels_forward_mode_keeps_the_projections_law():
    """An expert projection's tangent through the template (its launches interpreted, as an A100 runs
    them) is the XLA projection's, bit for bit: `dx @ Q(kernel) + Q(x) @ dkernel` with the tangents
    unrounded, where a cast's own tangent would round an fp32 input's to bf16 first (two products of an
    fp32 1 + 2^-8 and 1.5 sum to 3.015625, rounded first 3)."""
    sizes = jnp.asarray([1, 2], jnp.int32)
    x, w = jnp.zeros((3, 2), jnp.float32), jnp.full((2, 2, 2), 1.5, jnp.float32)
    dx, dw = jnp.full((3, 2), 1 + 2.0 ** -8, jnp.float32), jnp.full((2, 2, 2), 2.0 ** -9, jnp.float32)

    def tangent(implementation):
        return jax.jit(lambda x, w, dx, dw: jax.jvp(lambda x, w: expert_projection(
            x, w, sizes, jnp.bfloat16, implementation, None), (x, w), (dx, dw))[1])(x, w, dx, dw)

    with generation.assumed("sm80"), generation.routes() as taken:
        kernels = tangent("pallas")
    assert {route.kernel for route in taken if route.choice == "grouped_matmul"} == {"triton"}
    np.testing.assert_array_equal(np.asarray(kernels, np.float32), np.asarray(tangent("xla"), np.float32))


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="a GPU of sm80 or later runs the named kernel")
def test_a_named_kernel_refuses_where_the_kernels_do_not_run():
    """A kernel `generation.chosen` names for a call no kernel runs (a CPU's, whose kernels run nowhere
    but interpreted as a GPU it states) raises with the reason, rather than running XLA."""
    x, w, sizes = jnp.ones((8, 16), jnp.bfloat16), jnp.ones((2, 16, 16), jnp.float32), jnp.asarray([4, 4])
    with generation.chosen(grouped_matmul="triton"), pytest.raises(ValueError):
        expert_projection(x, w, sizes, jnp.bfloat16, "auto", None)


@runs("sm80")
def test_a_call_of_no_rows_runs_xla_or_refuses_a_named_kernel():
    """A projection of no rows launches nothing: XLA gives its empty value, the route saying why, and a
    kernel `generation.chosen` names for it raises rather than running XLA."""
    x, w = jnp.ones((0, 16), jnp.bfloat16), jnp.ones((2, 16, 16), jnp.float32)
    sizes = jnp.zeros((2,), jnp.int32)
    with generation.assumed("sm80"), generation.routes() as taken:
        y = expert_projection(x, w, sizes, jnp.bfloat16, "pallas", None)
    assert y.shape == (0, 16)
    assert {(route.kernel, route.refusal.detail if route.refusal else None) for route in taken
            if route.choice == "grouped_matmul"} == {("xla", "the call has no rows")}
    with generation.assumed("sm80"), generation.chosen(grouped_matmul="triton"), pytest.raises(ValueError):
        expert_projection(x, w, sizes, jnp.bfloat16, "pallas", None)

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


def test_a_gpu_of_no_family_runs_xla_s_grouped_matmul():
    """A GPU outside every instruction family Dew knows (sm70, Volta) runs an expert projection on XLA's
    ragged dot, its route saying why, with no estimate asked of the kernels."""
    x = jnp.ones((32, 64), jnp.bfloat16)
    w = jnp.ones((4, 64, 128), jnp.float32)
    sizes = jnp.asarray([8, 8, 8, 8], jnp.int32)
    with generation.assumed("sm70"), generation.routes() as taken:
        jax.jit(lambda x, w: expert_projection(x, w, sizes, jnp.bfloat16, "auto", None)).trace(x, w)
    (route,) = {(route.kernel, route.refusal.detail if route.refusal else None) for route in taken
                if route.choice == "grouped_matmul"}
    assert route[0] == "xla" and route[1]


@pytest.mark.parametrize("block_m,block_n", [(64, 128), (128, 512)])
def test_a_cooperative_tile_refuses_what_its_consumers_cannot_compute(block_m, block_n):
    """A cooperative tile's two consumers each take half its rows in warpgroup MMAs (a multiple of 64
    rows by at most 256 columns): a tile whose halves are not that refuses."""
    with pytest.raises(ValueError):
        mosaic_gpu.Tiles(block_m, block_n, 64, 2, 1, warpgroups=2)


@INTERPRETS
def test_a_tile_of_192_columns_stages_every_column_it_writes():
    """A consumer stages its bf16 output in chunks of at most 128 columns that divide its tile: a 192-column
    tile, one warpgroup or two, writes all of its columns, as exact as XLA's ragged dot from float64."""
    sizes = SIZES["ragged"]
    x, w, dy, group = operands(sizes, jnp.bfloat16)
    wide = np.tile(w, (1, 1, 2))[..., :192]
    truth = oracle.products(x, wide, np.tile(dy, (1, 2))[:, :192], sizes)
    reference = xla_products(x, wide, np.tile(dy, (1, 2))[:, :192], sizes, jnp.bfloat16, jnp.float32)
    for warpgroups in (1, 2):
        tiles = mosaic_gpu.Tiles(128, 192, 64, 2, 1, warpgroups=warpgroups)
        got = mosaic_gpu.gmm(jnp.asarray(x, jnp.bfloat16), jnp.asarray(wide, jnp.bfloat16),
                             jnp.asarray(sizes, jnp.int32), tiles=tiles, transpose_rhs=False,
                             out_dtype=jnp.bfloat16, multiprocessors=1, interpret=grouped_matmul.INTERPRETED)
        check(got, reference[0], truth[0], group, f"forward {tiles}")


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
def test_an_ampere_gpu_runs_mosaics_launch_unnamed_and_tritons_by_name(sizes, out_dtype):
    """Without warpgroup MMA, the Mosaic GPU forward's mma.sync tile (`mosaic_gpu.derived`) runs the grouped
    forward unnamed, a cell no old route measured faster on (`grouped_matmul.DECLARED`, CONSUMERS3), and
    Triton's launch where the call names it; named, each is as exact as XLA's ragged dot rounded to its
    output dtype from float64, unrouted rows and empty groups included."""
    x, w, dy, group = operands(SIZES[sizes], jnp.bfloat16)
    truth = oracle.products(x, w, dy, SIZES[sizes])
    group_sizes = jnp.asarray(SIZES[sizes], jnp.int32)
    lhs, rhs = jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.bfloat16)
    reference = grouped_matmul.xla_ragged_dot(lhs, rhs, group_sizes,
                                              preferred_element_type=jnp.float32).astype(out_dtype)

    def forward(x, w):
        # The default precision: the suite's highest asks each step's sum apart, which mma.sync's tiles do
        # not add.
        return template.gemm(x, w, group_sizes, out_dtype=out_dtype, precision="default",
                             interpret_on_cpu=True)

    def running(**chosen):
        # The launches of the branch the call runs; a GPU process's CPU branch beside it interprets Triton's.
        with ampere(), generation.chosen(**chosen), tuning.seen() as launches:
            got = jax.jit(forward)(lhs, rhs)
        ran = [(launch, ba) for (launch, _), (_, ba) in zip(launches, launches.taken, strict=True)
               if ba.arguments["interpret"] == (jax.default_backend() == "cpu")]
        return got, ran

    for named, kernel in (("triton", "Gmm"), ("mosaic_gpu", "MosaicGmm")):
        got, ran = running(grouped_matmul=named)
        assert {type(launch).__name__ for launch, _ in ran} == {kernel} and got.dtype == out_dtype
        check(got, reference, truth[0], group, f"sm80 forward through {kernel} to {jnp.dtype(out_dtype)}")
    _, ran = running()
    assert {type(launch).__name__ for launch, _ in ran} == {"MosaicGmm"}


@pytest.mark.skipif(not AMPERE_CLASS, reason="no mma.sync kernel this GPU and JAX lower")
def test_an_mma_sync_forward_of_no_depth_is_zero():
    """A product of no depth through mma.sync's tile, named on an Ampere GPU, is zero: it sums nothing,
    and no launch runs."""
    lhs, rhs = jnp.zeros((128, 0), jnp.bfloat16), jnp.zeros((1, 0, 256), jnp.bfloat16)
    with ampere(), generation.chosen(grouped_matmul="mosaic_gpu"):
        got = jax.jit(lambda x, w: template.gemm(x, w, jnp.asarray([128], jnp.int32), out_dtype=jnp.float32,
                                                 precision="default", interpret_on_cpu=True))(lhs, rhs)
    assert not np.any(np.asarray(got))


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
    tiles = mosaic_gpu.Tiles(128, 256, 64, 2, 1, warpgroups=2)

    def run(epilogue, out_dtypes=None):
        return mosaic_gpu.gmm(x, w, sizes, tiles=tiles, transpose_rhs=False, out_dtype=jnp.bfloat16,
                              multiprocessors=1, interpret=grouped_matmul.INTERPRETED, epilogue=epilogue,
                              epilogue_operands=(bias,), epilogue_axes=(("columns",),), out_dtypes=out_dtypes)

    first, second = run(doubled_halves, (jnp.bfloat16, jnp.bfloat16))
    np.testing.assert_array_equal(np.asarray(first, np.float32), np.asarray(run(twice), np.float32))
    np.testing.assert_array_equal(np.asarray(second, np.float32), np.asarray(run(halved), np.float32))


def test_each_groups_row_tiles_start_at_its_first_row():
    """A group's row tiles start at its first row, rounded down to the 8 a TMA descriptor tiles rows by,
    whole tiles after it, as CUTLASS's grouped GEMM tiles each group: a group within a tile's rows from
    there takes one tile, reading its matrix once, an empty group none, and the tiles fit the launch's
    bound (`mosaic_gpu.tile_groups`)."""
    lengths = jnp.asarray([100, 0, 128, 129, 3, 40], jnp.int32)
    bound = -(-int(lengths.sum()) // 128) + len(lengths) - 1
    table = np.asarray(mosaic_gpu.tile_groups(lengths, 128, bound))
    assert table.tolist() == [[0, 0, 0, 100], [2, 96, 100, 228], [2, 224, 100, 228], [3, 224, 228, 357],
                              [3, 352, 228, 357], [4, 352, 357, 360], [5, 360, 360, 400], [0, 0, 0, 0],
                              [0, 0, 0, 0]]
