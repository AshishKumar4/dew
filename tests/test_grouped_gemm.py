"""The grouped GEMM template's kernels and their choice: Dew's Mosaic GPU grouped matmuls for Hopper
(`dew.nn.kernels.templates.gemm.mosaic_gpu`), interpreted on the CPU, each output element held to its
exact products (check 2's native GEMM contract, tests/native_certificates.py); the MoE dispatcher's
implementations; and the cost model (`dew.nn.kernels.templates.cost`) on analytic launches."""

import dataclasses
import functools
import importlib
import itertools
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from certified import lane
from generation_support import INTERPRETS, here, interpreted_projection, lowers, runs
from jax.sharding import Mesh
from native_certificates import (
    forward_products,
    input_gradient_products,
    kernel_gradient_products,
    native_dot_holds,
)
from op_certificates import BFLOAT16, FLOAT16, FLOAT32
from reference_error import assert_as_exact, equations, with_gradients

from dew.nn.kernels import (
    gemm,
    generation,
    grouped_cells,
    grouped_cutlass,
    grouped_launch,
    grouped_matmul,
    grouped_triton,
    selection,
    tuning,
)
from dew.nn.kernels.templates.gemm import mosaic_gpu, pricing, row_tiles, tiling
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
    """`values` rounded to `dtype` and held in float64, on the host (a GPU tier's prepare computes on no
    device)."""
    return np.asarray(values).astype(dtype).astype(np.float64)


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
    tiled = [tiling.Tiles(64, 64, 64, 2, 1, family='wgmma'),
             tiling.Tiles(128, 64, 64, 2, 1, family='wgmma'),
             tiling.Tiles(128, 64, 64, 2, 1, warpgroups=2, family='wgmma'),
             tiling.Tiles(128, 256, 64, 2, 1, warpgroups=2, family='wgmma')]
    for tiles, transpose in itertools.product(tiled, (False, True)):
        matrix = jnp.swapaxes(arrays[1], 1, 2) if transpose else arrays[1]
        got = mosaic_gpu.gmm(arrays[0], matrix, group_sizes, tiles=tiles, transpose_rhs=transpose,
                             out_dtype=dtype, multiprocessors=1, interpret=grouped_launch.INTERPRETED)
        routed(got, group, forward, fmt, f"forward {tiles} transpose={transpose}")
    for tiles in (tiling.Tiles(64, 64, 64, 2, 1, family='wgmma'),
                  tiling.Tiles(64, 64, 128, 2, 1, family='wgmma'),
                  tiling.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma'),
                  tiling.Tiles(128, 256, 64, 2, 1, warpgroups=2, family='wgmma')):
        got = mosaic_gpu.tgmm(arrays[0], arrays[2], group_sizes, tiles=tiles, out_dtype=jnp.float32,
                              multiprocessors=1, interpret=grouped_launch.INTERPRETED)
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
@pytest.mark.parametrize("tiles", [tiling.Tiles(64, 64, 64, 2, 1, family='wgmma'),
                                   tiling.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')],
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
                              multiprocessors=1, interpret=grouped_launch.INTERPRETED)[kept]
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
        narrow = tiling.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')
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
@pytest.mark.parametrize("generation_name,implementation", [("sm80", "triton"), ("sm90", "mosaic")])
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
        y, dx, dw = (products if implementation == "mosaic" else jax.jit(products))(*operands_)
    assert (y.dtype, dx.dtype, dw.dtype) == (jnp.bfloat16, jnp.bfloat16, jnp.float32)
    routed(y, group, forward_products(x[:sum(sizes)], w, sizes), BFLOAT16, "y")
    routed(dx, group, input_gradient_products(dy[:sum(sizes)], w, sizes), BFLOAT16, "dx")
    for index, held_products in enumerate(kernel_gradient_products(x, dy, sizes)):
        held(dw[index], held_products, FLOAT32, f"dw of group {index}")


def test_a_16_bit_products_stated_dot_algorithm_reaches_xla():
    """A dense float16 product through Dew's selector's portable arm (`gemm.gemm`, `grouped_matmul` chosen
    `xla`, on this CPU) stating an explicit dot algorithm multiplies by it, as XLA's own dot does, where a
    precision level on 16-bit operands is the default product's: one XLA:CPU lacks (`F16_F16_F16`, its
    float16 accumulator) is refused as XLA's own dot refuses it, not run as the default product (audit
    K2-7), and one it has (`BF16_BF16_F32` on bfloat16 operands) is XLA's own dot with it."""
    preset = jax.lax.DotAlgorithmPreset

    def through_dew(x, w, algorithm, out_dtype):
        with generation.chosen(grouped_matmul="xla"):
            return gemm.gemm(x, w, out_dtype=out_dtype, precision=algorithm)

    def xla_own(x, w, algorithm, out_dtype):
        return jax.lax.dot_general(x, w, (((1,), (0,)), ((), ())), precision=algorithm,
                                   preferred_element_type=out_dtype)

    x, w = jnp.asarray([[2048.0] + [1.0] * 6 + [-2048.0]], jnp.float16), jnp.ones((8, 1), jnp.float16)
    refusals = []
    for arm in (xla_own, through_dew):
        with pytest.raises(jax.errors.JaxRuntimeError) as refused:
            jax.block_until_ready(jax.jit(arm, static_argnums=(2, 3))(x, w, preset.F16_F16_F16, jnp.float16))
        refusals.append(str(refused.value).splitlines()[0])
    assert refusals[0] == refusals[1]
    rng = np.random.default_rng(0)
    xb, wb = (jnp.asarray(rng.normal(size=shape), jnp.bfloat16) for shape in ((4, 64), (64, 8)))
    np.testing.assert_array_equal(
        np.asarray(jax.jit(through_dew, static_argnums=(2, 3))(xb, wb, preset.BF16_BF16_F32, jnp.float32)),
        np.asarray(jax.jit(xla_own, static_argnums=(2, 3))(xb, wb, preset.BF16_BF16_F32, jnp.float32)))


def test_cutlass_eligibility_reads_the_dew_cutlass_build_installed_now(tmp_path, monkeypatch):
    """Which generations' kernels the installed dew-cutlass holds is read where CUTLASS's eligibility asks
    (`generation.cutlass_built`), from the build installed then: a process that installs the wheel, or
    another build of it, routes as that build allows (audit P9), and the program its trace lowers states
    the route it took. Two builds, one holding Hopper's kernels and one Ampere's, stand in for the wheel
    ahead of any installed one."""
    path = list(sys.path)
    # The process's own dew_cutlass, or none, back in sys.modules after the test: each stand-in build is
    # imported in its place, never left there (a stand-in left stated no TARGETS, and every later test of
    # the process that counted a program's kernel calls failed, train 991's red).
    monkeypatch.setitem(sys.modules, "dew_cutlass", sys.modules.get("dew_cutlass"))

    def installed(minimum: tuple[int, int], built: str) -> dict:
        # Each build its own directory ahead of the installed packages, so no build's bytecode stands in
        # for another's.
        wheel = tmp_path / built
        wheel.mkdir()
        dispatch = f"return {built!r} if (major, minor) >= {minimum} else None"
        (wheel / "dew_cutlass.py").write_text(f"def architecture(major, minor):\n    {dispatch}\n")
        monkeypatch.setattr(sys, "path", [str(wheel), *path])
        del sys.modules["dew_cutlass"]
        importlib.invalidate_caches()
        return generation.cutlass_built()

    hopper = installed((9, 0), "sm90")
    assert hopper["sm90"] == "sm90" and hopper["sm80"] is None
    assert grouped_cutlass.cutlass_built_for("sm90") == "sm90"
    assert grouped_cutlass.cutlass_built_for("sm80") is None
    ampere = installed((8, 0), "sm80")
    assert ampere["sm80"] == "sm80" and grouped_cutlass.cutlass_built_for("sm80") == "sm80"


def cutlass_runs_the_projection_as_exactly_as_xla_where_chosen(sizes, stacked=False, prologued=False):
    """Where the installed dew-cutlass holds this GPU's kernels, the projection's forward and both
    gradients through CUTLASS's launches, chosen by name and the forward's kernel pinned, are as exact as
    XLA's ragged dot from float64, unrouted rows and empty groups included. `stacked`, its matrices the
    second layer of a stack the call reads from its layer's first (a scanned stack's, `gemm._grouped`),
    which a pinned CUTLASS launch takes as a slice of the stack (row 4074's review). `prologued`, its rows
    scaled by a side of one value a row, `[rows, 1]` (an input-scaled MoE's routing weights), which a
    CUTLASS launch forms in XLA before its call (`grouped_cutlass._formed`), its gradient compared too
    (#3985's review)."""
    # The wheel is optional: a process without it, or whose wheel holds no kernel for its GPU, runs the
    # template's launches, which the suite checks apart.
    here = generation.gpu_target().generation
    if grouped_cutlass.cutlass_built_for(here) is None:
        pytest.skip(f"no installed dew-cutlass holds kernels for {here}")
    group_sizes = np.asarray(SIZES[sizes], np.int32)

    def draw(seed):
        x, w, dy, _ = operands(SIZES[sizes], jnp.bfloat16, seed)
        weights = np.random.default_rng(seed).uniform(0.5, 1.5, size=(x.shape[0], 1)).astype(np.float32)
        return (*(np.asarray(value, jnp.bfloat16) for value in (x, w)), weights, np.asarray(dy, jnp.bfloat16))

    def scaled(x, w, weights, dtype=jnp.bfloat16):
        return gemm.gemm(x.astype(dtype), w.astype(dtype), jnp.asarray(group_sizes, jnp.int32),
                         prologue=lambda rows, s: rows * s.astype(rows.dtype),
                         prologue_operands=((weights.astype(dtype), ("rows", None)),), out_dtype=dtype)

    def projection(implementation, dtype=jnp.bfloat16):
        return lambda x, w: expert_projection(x, w, jnp.asarray(group_sizes, jnp.int32), dtype,
                                              implementation, None)
    # Its build's kernel pinned, as a tuned pick names it: the pin taken as legal for the wheel's one kernel
    # (`_Cutlass.legal`), where a template's shared-memory model, which CUTLASS has none of, raised.
    built = grouped_cutlass.cutlass_built_for(here)
    pinned = {grouped_cutlass.CutlassGmm: grouped_cutlass.cutlass_tile(built, jnp.bfloat16)}

    def layer_of_a_stack(x, stack):
        return gemm.gemm(x, stack, jnp.asarray(group_sizes, jnp.int32), layer=jnp.int32(1),
                         out_dtype=jnp.bfloat16)

    def through_cutlass(x, w, weights, dy):
        with generation.chosen(grouped_matmul="cutlass"), tuning.pinned(pinned):
            if prologued:
                return with_gradients(scaled)(x, w, weights, dy)
            if not stacked:
                return with_gradients(projection("pallas"))(x, w, dy)
            # The whole stack differentiated: its other layer's gradient zero, its layer's the projection's.
            out, d_x, d_stack = with_gradients(layer_of_a_stack)(x, jnp.stack([jnp.ones_like(w), w]), dy)
            return out, d_x, d_stack[1], d_stack[0]

    def plain(x, w, weights, dy, dtype=jnp.bfloat16):
        if prologued:
            with generation.chosen(grouped_matmul="xla"):
                return with_gradients(functools.partial(scaled, dtype=dtype))(x, w, weights, dy)
        out, d_x, d_w = with_gradients(projection("xla", dtype))(x, w, dy)
        return (out, d_x, d_w, *((jnp.zeros_like(d_w),) if stacked else ()))

    assert_as_exact(jax.jit(through_cutlass), jax.jit(plain), draw,
                    f"cutlass {sizes}{' stacked' if stacked else ''}{' prologued' if prologued else ''}",
                    seeds=SEEDS,
                    truth_fn=functools.partial(plain, dtype=jnp.float64))


def the_template_sums_16_bit_products_into_fp32_as_exactly_as_xla(depth):
    """On this GPU the template's grouped product of bfloat16 rows and matrices into an fp32 output (its
    route named, `mosaic_gpu`), forward and both gradients, is as exact as XLA's ragged dot from float64 at a
    `depth`-deep contraction: at the default precision its sums the tensor cores' own, as XLA's (cuBLAS's)
    are, which held a dense kernel gradient within 0.98-1.00 of cuBLAS's error from float64 on an A100 and an
    H100 at depths of 128 to 2048 (M-EM-3; row 4331's 1.661x at depth 128 did not reproduce)."""
    group_sizes = np.asarray([70, 66], np.int32)

    def draw(seed):
        # NumPy's, as a site's draws are: a GPU tier's prepare computes on no device.
        rng = np.random.default_rng(seed)
        return (rng.normal(size=(136, depth)).astype(jnp.bfloat16),
                (rng.normal(size=(2, depth, 256)) / np.sqrt(depth)).astype(jnp.bfloat16),
                rng.normal(size=(136, 256)).astype(np.float32))

    def product(dtype):
        # fp32 out of 16-bit products, float64 out of the truth's, whose cotangent is float64.
        out = jnp.promote_types(dtype, jnp.float32)
        return lambda x, w: gemm.gemm(x.astype(dtype), w.astype(dtype), group_sizes, out_dtype=out)

    def through_template(x, w, dy):
        with generation.chosen(grouped_matmul="mosaic"):
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
    one-row rung (an A100's in a row's layout, `swizzling.copied`: row 4474): its forward and both gradients
    are as exact as XLA's ragged dot, from float64."""
    group_sizes = np.asarray([67, 69], np.int32)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(np.asarray(values).astype(jnp.bfloat16)
                     for values in (rng.normal(size=(136, 256)), rng.normal(size=(2, 256, 2880)) / 16,
                                    rng.normal(size=(136, 2880))))

    def projection(implementation, dtype=jnp.bfloat16):
        return with_gradients(lambda x, w: expert_projection(x, w, group_sizes, dtype, implementation, None))

    def through_template(x, w, dy):
        with generation.chosen(grouped_matmul="mosaic"):
            return projection("pallas")(x, w, dy)

    assert_as_exact(jax.jit(through_template), jax.jit(projection("xla")), draw, "template gpt-oss width",
                    seeds=SEEDS, truth_fn=projection("xla", jnp.float64))


def cutlass_runs_a_mapped_projection_as_exactly_as_xla_where_chosen(shared):
    """Where the installed dew-cutlass holds this GPU's kernels, a projection mapped over a batch of two
    (`jax.vmap`: each its rows and cotangent, and its matrices unless `shared`), its forward and both
    gradients through CUTLASS's launches chosen by name, each map one launch in turn (their FFI calls'
    sequential batching, row 4389), are as exact as XLA's ragged dot mapped, from float64."""
    here = generation.gpu_target().generation
    if grouped_cutlass.cutlass_built_for(here) is None:
        pytest.skip(f"no installed dew-cutlass holds kernels for {here}")
    group_sizes = np.asarray(SIZES["ragged"], np.int32)

    def draw(seed):
        first, second = (operands(SIZES["ragged"], jnp.bfloat16, seed + offset)[:3]
                         for offset in (0, 1 << 16))
        x, w, dy = (np.stack([np.asarray(a, jnp.bfloat16), np.asarray(b, jnp.bfloat16)])
                    for a, b in zip(first, second, strict=True))
        return x, w[0] if shared else w, dy

    def mapped(implementation, dtype=jnp.bfloat16):
        one = with_gradients(lambda x, w: expert_projection(x, w, jnp.asarray(group_sizes, jnp.int32), dtype,
                                                            implementation, None))
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
    launch = grouped_triton.Gmm()
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



def test_a_dense_kernel_gradient_xla_takes_is_its_own_dot():
    """A dense kernel gradient the selector gives XLA (an A100's declared cell, `grouped_matmul.DECLARED`,
    which DENSE1 timed as XLA's dot) runs XLA's own dot over its rows, as a dense forward does (`xla_dot`): no
    ragged dot of one group, which took an A100's gate_up weight gradient 470 ms against the template's 11.4
    (W121-a100), and its rows read as they are: a dense call routes every row, and a mask of them was a select
    XLA kept, up to 23% of the A100's XLA kernel gradient (M-NG-45-a100's 4B key)."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    with generation.assumed("sm80"), generation.routes() as routes:
        graph = jax.make_jaxpr(lambda x, dy: gemm.gemm_t(x, dy, None, out_dtype=jnp.bfloat16))(
            bf16((512, 256)), bf16((512, 384)))
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"xla"}, routes
    names = {equation.primitive.name for equation in equations(graph, platform="cuda")}
    assert "dot_general" in names and "ragged_dot_general" not in names, names
    masks = [equation for equation in equations(graph, platform="cuda")
             if equation.primitive.name == "select_n" and equation.outvars[0].aval.shape == (512, 256)]
    assert not masks, "a dense call's rows are selected whole before its product"
    assert graph.out_avals[0].shape == (1, 256, 384)


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


def test_a_hoppers_gpt_oss_kernel_gradient_takes_a_cooperative_tile_its_depth_leaves_partial():
    """gpt-oss's expert kernel gradient (2880 deep and wide: 2880 divides by no cooperative wgmma tile's 128
    rows) on an H100 derives a tile the launch takes (`legal`, `candidate`), and a two-consumer 128-row tile,
    its last row tile half past the depth (`mosaic_gpu.ragged_rows`; row 2295: one consumer's 64 x 192 tile
    ran 1.79x Triton's time), is one it takes and lowers for CUDA; interpreted at a smaller depth no 128 rows
    divide (320), its value bitwise the gradient at a tile that divides it, each partial share's rows past the
    depth stored nowhere. A depth a cooperative tile divides keeps whole tiles. On an A100, a 192-row
    mma.sync tile's shares at a 264-deep depth (72 rows past the first share) give the exact gradient."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((128,), jnp.int32)
    launch = grouped_matmul.MosaicTgmm()
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        bound = launch.bind(bf16((32768, 2880)), bf16((32768, 2880)), sizes, (), out_dtype="float32",
                            interpret=False)
        tile = launch.derived(bound, target)
        assert tile is not None and launch.legal(bound, tile, target)
        assert launch.candidate(bound, tile, target)
        cooperative = tiling.Tiles(128, 192, 64, 2, 1, warpgroups=2, family="wgmma")
        assert 2880 % cooperative.block_m and launch.legal(bound, cooperative, target) and launch.candidate(
            bound, cooperative, target)
        lowered = functools.partial(mosaic_gpu.tgmm, tiles=cooperative, out_dtype=jnp.float32,
                                    multiprocessors=132, interpret=False)
        jax.jit(lowered).trace(bf16((4096, 2880)), bf16((4096, 2880)), jax.ShapeDtypeStruct((16,), jnp.int32)
                               ).lower(lowering_platforms=("cuda",))
        whole = launch.derived(launch.bind(bf16((32768, 2048)), bf16((32768, 2880)), sizes, (),
                                           out_dtype="float32", interpret=False), target)
        assert whole is not None and 2048 % whole.block_m == 0, whole
        rng = np.random.default_rng(37)
        tokens = jnp.asarray(rng.integers(-3, 4, size=(300, 320)), jnp.bfloat16)
        cotangent = jnp.asarray(rng.integers(-3, 4, size=(300, 128)), jnp.bfloat16)
        groups = jnp.asarray([100, 0, 150, 30], jnp.int32)

        def gradient(block_m, warpgroups):
            tiles = tiling.Tiles(block_m, 128, 64, 2, 1, warpgroups=warpgroups, family="wgmma")
            return np.asarray(mosaic_gpu.tgmm(tokens, cotangent, groups, tiles=tiles, out_dtype=jnp.float32,
                                              multiprocessors=4, interpret=grouped_launch.INTERPRETED))

        np.testing.assert_array_equal(gradient(128, 2), gradient(64, 1))
    with generation.assumed("sm80"):
        tokens = jnp.asarray(rng.integers(-3, 4, size=(300, 264)), jnp.bfloat16)
        cotangent = jnp.asarray(rng.integers(-3, 4, size=(300, 64)), jnp.bfloat16)
        narrow = tiling.Tiles(192, 64, 32, 2, 1, family="mma.sync")
        got = mosaic_gpu.tgmm(tokens, cotangent, groups, tiles=narrow, out_dtype=jnp.float32,
                              multiprocessors=4, interpret=grouped_launch.INTERPRETED)
    # Every product and sum exact (small integers): each group's rows' tokens times their cotangent.
    ends = np.cumsum(np.asarray(groups))
    exact = np.stack([np.asarray(tokens, np.float64)[end - size:end].T @ np.asarray(cotangent, np.float64)[
        end - size:end] for end, size in zip(ends, np.asarray(groups), strict=True)])
    np.testing.assert_array_equal(np.asarray(got), exact)


def test_a_paired_products_input_gradient_adds_its_parts_on_the_launch_its_plain_product_takes():
    """A paired product's input gradient (row 2164) adds its first product into the second's epilogue
    (`gemm._added`) where that hook keeps the second on the launch its plain product takes (`_keeps_launch`:
    an H100's warpgroup MMA and an A100's mma.sync both count the hook); the same sum rounded once:
    interpreted on an H100 and an A100 through the template, the paired projection's input gradient bitwise
    the two products' fp32 sum rounded to the rows' dtype."""
    rng = np.random.default_rng(41)
    tokens = jnp.asarray(rng.integers(-3, 4, size=(200, 128)), jnp.bfloat16)
    first, second = (jnp.asarray(rng.integers(-3, 4, size=(3, 128, 128)), jnp.bfloat16) for _ in range(2))
    cotangents = tuple(jnp.asarray(rng.integers(-3, 4, size=(200, 128)), jnp.bfloat16) for _ in range(2))
    sizes = jnp.asarray([70, 0, 120], jnp.int32)
    prior = jnp.zeros((200, 128), jnp.float32)
    options = {"interpret_on_cpu": True, "precision": None, "transpose_rhs": True, "dense": False,
               "first": None, "gather": None}
    def paired(tokens):
        return gemm.gemm(tokens, first, sizes, out_dtype=jnp.bfloat16, out_dtypes=(jnp.bfloat16,) * 2,
                         paired=second, interpret_on_cpu=True)

    wide = sum(jax.lax.ragged_dot(d.astype(jnp.float32), jnp.swapaxes(m, 1, 2).astype(jnp.float32), sizes)
               for d, m in zip(cotangents, (first, second), strict=True))
    for generation_name in ("sm90", "sm80"):
        with generation.assumed(generation_name):
            assert gemm._keeps_launch(cotangents[1], second, sizes, gemm._added, (prior,), (jnp.bfloat16,),
                                      (jnp.bfloat16,), options), generation_name
        # Interpreted on the host's CPU, whatever GPU it has (row 7130).
        with (generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"),
              jax.default_device(jax.devices("cpu")[0])):
            [d_tokens] = jax.vjp(paired, tokens)[1](cotangents)
        np.testing.assert_array_equal(np.asarray(d_tokens), np.asarray(wide.astype(jnp.bfloat16)),
                                      err_msg=generation_name)


def ragged_draw(seed):
    """The ragged sizes' rows, matrices and cotangent in bf16, at a seed (`operands`)."""
    x, w, dy, _ = operands(SIZES["ragged"], jnp.bfloat16, seed)
    return tuple(jnp.asarray(value, jnp.bfloat16) for value in (x, w, dy))


UNSEEN = {
    "ampere-class": generation.GpuTarget("sm80", "Unmeasured Ampere-class GPU", shared_limit=99 * 1024,
                                         multiprocessors=46, clock_hz=1.4e9, memory_bandwidth=0.45e12,
                                         host_links={}),
    "hopper-class": generation.GpuTarget("sm90", "Unmeasured Hopper-class GPU", shared_limit=180 * 1024,
                                         multiprocessors=78, clock_hz=1.6e9, memory_bandwidth=2.4e12,
                                         host_links={}),
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
    with (generation.assumed(target), generation.chosen(grouped_matmul="mosaic"),
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
        with generation.assumed(target), generation.chosen(grouped_matmul="mosaic"):
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
        with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
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


@INTERPRETS
def test_the_template_takes_an_fp32_cotangent_at_the_highest_precision_as_its_bf16_parts():
    """A grouped product of bf16 rows and matrices into fp32 with the template named (`mosaic_gpu`),
    interpreted as an H100's at the highest precision: its input gradient multiplies the fp32 cotangent by the
    bf16 matrix and its kernel gradient the rows by it, each an fp32 product the tensor cores multiply only in
    TF32, so each runs as the cotangent's bf16 parts (`gemm._in_parts`), each part's product exact, on par
    with XLA's whole fp32 product from float64 (check 1), where the template refused them (row 5405)."""
    group_sizes = jnp.asarray([70, 66], jnp.int32)

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 3)
        return (jax.random.normal(keys[0], (136, 128)).astype(jnp.bfloat16),
                (jax.random.normal(keys[1], (2, 128, 256)) / np.sqrt(128)).astype(jnp.bfloat16),
                jax.random.normal(keys[2], (136, 256)).astype(jnp.float32))

    def product(dtype, **options):
        out = jnp.promote_types(dtype, jnp.float32)
        return lambda x, w: gemm.gemm(x.astype(dtype), w.astype(dtype), group_sizes, out_dtype=out, **options)

    def step(kernel):
        def run(x, w, dy):
            with (generation.assumed("sm90"), generation.chosen(grouped_matmul=kernel),
                  jax.default_matmul_precision("highest")):
                return with_gradients(product(jnp.bfloat16, interpret_on_cpu=True))(x, w, dy)
        return run

    assert_as_exact(jax.jit(step("mosaic")), jax.jit(step("xla")), draw, "an fp32 cotangent's bf16 parts",
                    seeds=8, truth_fn=with_gradients(product(jnp.float64)))


def _highest_on_h100(kernel: str, rows, matrices, group_sizes):
    """A grouped product's fp32 output at the highest precision through `kernel`, interpreted as an H100's."""
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul=kernel),
          jax.default_matmul_precision("highest")):
        return np.asarray(jax.jit(lambda x, w: gemm.gemm(x, w, group_sizes, out_dtype=jnp.float32,
                                                         interpret_on_cpu=True))(rows, matrices))


@INTERPRETS
def test_an_fp32_products_parts_keep_a_nonfinite_operand_whole():
    """An fp32 grouped product at the highest precision with the template named, interpreted as an H100's,
    runs as its operands' bf16 parts (`gemm._in_parts`): a row holding an infinity gives that row's
    infinities, and an infinity times a zero column NaN, as XLA's whole fp32 product gives them, where the
    parts saturated an infinity to finite values."""
    group_sizes = jnp.asarray([64, 64], jnp.int32)
    rows = jnp.ones((128, 128), jnp.float32).at[3, 5].set(jnp.inf)
    matrices = jnp.full((2, 128, 128), 2.0 ** -100, jnp.float32).at[0, 5, 7].set(0.0)

    parted, whole = (_highest_on_h100(kernel, rows, matrices, group_sizes) for kernel in ("mosaic", "xla"))
    np.testing.assert_array_equal(np.isinf(parted), np.isinf(whole))
    np.testing.assert_array_equal(np.isnan(parted), np.isnan(whole))
    assert np.isinf(parted[3]).sum() == 127 and np.isnan(parted[3, 7])


@INTERPRETS
def test_bf16_rows_multiply_an_fp16_matrix_rounded_to_bf16_at_the_highest_precision():
    """A grouped product of bf16 rows and an fp16 matrix at the highest precision, the template named,
    interpreted as an H100's: the matrix is rounded to the rows' bf16 as the product multiplies them (1 +
    2^-10 to 1), as XLA's route rounds it, where splitting the fp16 matrix into parts kept its 2^-10."""
    group_sizes = jnp.asarray([64, 64], jnp.int32)
    rows = jnp.ones((128, 128), jnp.bfloat16)
    matrices = jnp.full((2, 128, 128), 1 + 2.0 ** -10, jnp.float16)

    for kernel in ("mosaic", "xla"):
        np.testing.assert_array_equal(_highest_on_h100(kernel, rows, matrices, group_sizes),
                                      np.full((128, 128), 128.0, np.float32))


@INTERPRETS
def test_a_mixed_kernel_gradients_prologue_reads_and_rounds_the_rows_in_their_dtype():
    """A kernel gradient of bf16 rows and fp32 cotangent rows (`gemm.gemm_t`, multiplied in fp32 whole at the
    highest precision, so on the template as the cotangent's bf16 parts) whose prologue adds 2^-8 to the
    rows, the template named and interpreted as an H100's: the prologue reads the rows in their bf16 and
    rounds its result there (1 + 2^-8 to 1), as the portable program does, so the gradient is the rows' 128
    per entry, where the widened rows kept 1 + 2^-8 (128.5)."""
    rows = jnp.ones((128, 128), jnp.bfloat16)
    cotangent = jnp.ones((128, 128), jnp.float32)

    def gradient(kernel):
        with (generation.assumed("sm90"), generation.chosen(grouped_matmul=kernel),
              jax.default_matmul_precision("highest")):
            return np.asarray(jax.jit(lambda x, dy: gemm.gemm_t(
                x, dy, None, prologue=lambda r: r + 2.0 ** -8, out_dtype=jnp.float32,
                interpret_on_cpu=True))(rows, cotangent))

    np.testing.assert_array_equal(gradient("mosaic"), np.full((1, 128, 128), 128.0, np.float32))
    np.testing.assert_array_equal(gradient("xla"), np.full((1, 128, 128), 128.0, np.float32))


@INTERPRETS
@pytest.mark.parametrize("kernel", ["triton", "mosaic"])
def test_a_row_scales_gradient_reads_the_accumulator_summed_at_its_own_width(kernel):
    """A grouped product with a row-scale epilogue (an fp32 scale a row), interpreted as an A100's at the
    highest precision with `kernel` named: the scale's gradient sums the recomputed fp32 accumulator times
    the cotangent, and that accumulator is summed as an fp32 output's (`product.promoted`) whatever the
    product's own output, so a float16 output's scale gradient is the float32 output's, bit for bit, under a
    cotangent both hold exactly: one recomputation program for both. An fp16 output's recomputation summed as
    its own output, unpromoted, put an A100's scale gradient at 1.37 times XLA's error (row 5406, M-EM-14)."""
    group_sizes = jnp.asarray([100, 0, 156], jnp.int32)
    rng = np.random.default_rng(5406)
    x = jnp.asarray(rng.standard_normal((256, 512)), jnp.float16)
    w = jnp.asarray(rng.standard_normal((3, 512, 128)) / np.sqrt(512), jnp.float16)
    scale = jnp.asarray(rng.uniform(0.5, 2.0, size=(256,)), jnp.float32)
    cotangent = jnp.asarray(rng.integers(-4, 5, size=(256, 128)), jnp.float32)

    def scale_gradient(out_dtype):
        def product(scale):
            return gemm.gemm(x, w, group_sizes, epilogue=lambda acc, s: acc * s[:, None],
                             epilogue_operands=((scale, ("rows",)),), out_dtype=out_dtype,
                             interpret_on_cpu=True)

        with generation.assumed("sm80"), generation.chosen(grouped_matmul=kernel):
            _, pull = jax.vjp(product, scale)
            return np.asarray(jax.jit(pull)(cotangent.astype(out_dtype))[0])

    np.testing.assert_array_equal(scale_gradient(jnp.float16), scale_gradient(jnp.float32))


def test_a_kernel_gradient_of_no_rows_on_a_hopper_is_its_groups_zeros():
    """A kernel gradient contracting no rows on an H100 (a step whose experts received no tokens): its
    selection derives no split of an empty contraction (`pricing.split_count`, which took the least of
    no counts) and the gradient is its group's zero block."""
    lhs, rhs = jnp.ones((0, 128), jnp.bfloat16), jnp.ones((0, 128), jnp.bfloat16)
    with generation.assumed("sm90"):
        gradient = jax.jit(lambda lhs, rhs: gemm.gemm_t(lhs, rhs, jnp.zeros((1,), jnp.int32),
                                                        out_dtype=jnp.bfloat16))(lhs, rhs)
    np.testing.assert_array_equal(np.asarray(gradient, np.float32), np.zeros((1, 128, 128), np.float32))


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
        tiling.Tiles(block_m, block_n, 64, 2, 1, warpgroups=2, family='wgmma')


@INTERPRETS
def test_a_tile_of_192_columns_stages_every_column_it_writes():
    """A consumer stages its bf16 output in chunks of at most 128 columns that divide its tile: a 192-column
    tile, one warpgroup or two, writes all of its columns, each a value its exact products admit."""
    sizes = SIZES["ragged"]
    x, w, _, group = operands(sizes, jnp.bfloat16)
    wide = np.tile(w, (1, 1, 2))[..., :192]
    forward = forward_products(x[:sum(sizes)], wide, sizes)
    for warpgroups in (1, 2):
        tiles = tiling.Tiles(128, 192, 64, 2, 1, warpgroups=warpgroups, family='wgmma')
        got = mosaic_gpu.gmm(jnp.asarray(x, jnp.bfloat16), jnp.asarray(wide, jnp.bfloat16),
                             jnp.asarray(sizes, jnp.int32), tiles=tiles, transpose_rhs=False,
                             out_dtype=jnp.bfloat16, multiprocessors=1, interpret=grouped_launch.INTERPRETED)
        routed(got, group, forward, BFLOAT16, f"forward {tiles}")


HERE = here()
AMPERE_CLASS = HERE == "cpu" or (HERE.startswith("sm") and not generation.gpu_target(HERE).wgmma)
"""Whether this process runs mma.sync's kernels: a CPU interprets them as an A100's; a GPU without warpgroup
MMA (an A100, an A10, an L4) runs them on its own facts, where one with warpgroup MMA lowers them for its
own device, whose pipelines its Mosaic sizes past its block (SM80-h100)."""


def ampere(generation_name: str = "sm80"):
    """The assumed device an mma.sync test runs on: the process's own Ampere-class GPU, else
    `generation_name`'s, which a CPU interprets."""
    return generation.assumed(generation_name if HERE == "cpu" else None)


@pytest.mark.skipif(not AMPERE_CLASS, reason="no mma.sync kernel this GPU and JAX lower")
@pytest.mark.parametrize("out_dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("sizes", [name for name in SIZES if any(SIZES[name])])
def test_an_ampere_gpus_mosaic_and_triton_launches_are_each_as_exact_as_xla(sizes, out_dtype):
    """Without warpgroup MMA, the Mosaic GPU forward's mma.sync tile (`derivation.derived`) and Triton's
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

    for named in ("triton", "mosaic"):
        with ampere(), generation.chosen(grouped_matmul=named):
            got = jax.jit(forward)(lhs, rhs)
        assert got.dtype == out_dtype
        routed(got, group, products, FORMAT[out_dtype],
               f"sm80 forward through {named} to {jnp.dtype(out_dtype)}")


@pytest.mark.skipif(not AMPERE_CLASS, reason="no mma.sync kernel this GPU and JAX lower")
def test_an_mma_sync_forward_of_no_depth_is_zero():
    """A product of no depth on an Ampere GPU is zero: it sums nothing, no launch taking a call with no
    extent on an axis (`SUPPORTS`' extents), XLA's program its sum of nothing."""
    lhs, rhs = jnp.zeros((128, 0), jnp.bfloat16), jnp.zeros((1, 0, 256), jnp.bfloat16)
    with ampere(), generation.routes() as routes:
        got = jax.jit(lambda x, w: gemm.gemm(x, w, jnp.asarray([128], jnp.int32), out_dtype=jnp.float32,
                                                 precision="default", interpret_on_cpu=True))(lhs, rhs)
    assert got.shape == (128, 256) and not np.any(np.asarray(got))
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"xla"}, routes


@pytest.mark.parametrize("sizes", [(3, 2), (2, 2)])
def test_a_grouped_epilogue_of_no_depth_reads_each_rows_group(sizes):
    """A grouped product of no depth runs no launch, its zeros through the epilogue on whole arrays: its side
    operands by group read at each row as XLA's program reads them (`axes.at_rows`, row 4489), each row its
    group's bias, a row past the groups' the last's; a side operand by column as it is."""
    bias = jnp.asarray(np.arange(2 * 128).reshape(2, 128), jnp.float32)
    column = jnp.asarray(np.linspace(-1, 1, 128), jnp.float32)
    def added(acc, by_group, by_column):
        return acc + by_group + by_column

    tiles = tiling.Tiles(128, 128, 64, 2, 1, family="wgmma")
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
    tiles = tiling.Tiles(128, 256, 64, 2, 1, warpgroups=2, family='wgmma')

    def draw(seed):
        x, w, _, _ = operands(SIZES["ragged"], jnp.bfloat16, seed)
        return jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.bfloat16)

    def run(x, w):
        return mosaic_gpu.gmm(x, w, sizes, tiles=tiles, transpose_rhs=False, out_dtype=jnp.bfloat16,
                              multiprocessors=1, interpret=grouped_launch.INTERPRETED,
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
    """Every row tile `gmm`'s schedule takes (`row_tiles.tile_groups`, a group's tiles from its first row)
    reads within the rows its operands are allocated to (`row_tiles.row_extent`): its first row plus the
    tile's rows, unpredicated on an A100, stay inside the extent (row 1866 K3: groups [120, 8] at 64-row
    tiles read rows 120 to 183 of a 128-row lhs)."""
    rows = -(-sum(sizes) // row_tiles.TMA_ROWS) * row_tiles.TMA_ROWS
    lengths = jnp.asarray([*sizes, rows - sum(sizes)], jnp.int32)
    table = row_tiles.tile_groups(lengths, row_tile, -(-rows // row_tile) + len(sizes))
    used = np.asarray(table.group_end > table.group_start)
    reads = int((np.asarray(table.block_start)[used] + row_tile).max())
    assert reads <= row_tiles.row_extent(rows, row_tile)
    if sizes == [120, 8]:
        assert reads > -(-rows // row_tile) * row_tile  # past the row-tile padding that held before


@pytest.mark.parametrize("groups", [1, 4])
def test_the_tuners_group_sizes_route_every_row(groups):
    """The tuner's draw of a launch's group sizes (`_GroupedLaunch.operands`) routes every row: a grouped
    call's first group empty, as a step routes them, and a dense call's one group whole, which emptying left
    sized from a zero sum (W28-a100-2's NaN sizes)."""
    specs = [jax.ShapeDtypeStruct((300, 64), jnp.bfloat16),
             jax.ShapeDtypeStruct((groups, 64, 64), jnp.bfloat16), jax.ShapeDtypeStruct((groups,), jnp.int32)]
    sizes = np.asarray(grouped_matmul.MosaicGmm().operands(specs, 0)[2])
    assert sizes.sum() == 300 and (sizes >= 0).all() and (groups == 1 or sizes[0] == 0), sizes


@pytest.mark.mesh(devices=8)
def test_a_ragged_products_portable_route_places_its_output_on_an_explicit_mesh():
    """XLA's ragged products on a CPU (`gemm.gemm` and its reverse pass, `gemm.gemm_t`, the portable route a
    fused producer's analysis replays) of rows an Explicit data axis splits, as a device step's are
    (test_device_image_augmentation): JAX infers no placement for a ragged dot, so each states its own
    (`xla_products.ragged_placement`): the forward's and the input gradient's rows split as the tokens'
    are, the kernel gradient's sum over the split rows whole; a replicated operand on that mesh, placed too.
    Integers, so each value is the unplaced program's bit for bit; and of no rows, each the empty product,
    the kernel gradient each group's zeros."""
    from dew.distributed.placement import MeshSpec

    rng = np.random.default_rng(23)
    tokens = jnp.asarray(rng.integers(-3, 4, size=(64, 32)), jnp.float32)
    kernel = jnp.asarray(rng.integers(-3, 4, size=(2, 32, 16)), jnp.float32)
    cotangent = jnp.asarray(rng.integers(-3, 4, size=(64, 16)), jnp.float32)

    def products(tokens, cotangent, sizes):
        forward, pull = jax.vjp(lambda t, k: gemm.gemm(t, k, sizes, out_dtype=jnp.float32), tokens, kernel)
        return (forward, *pull(cotangent), gemm.gemm_t(tokens, cotangent, sizes, out_dtype=jnp.float32))

    sizes = jnp.asarray([40, 24], jnp.int32)
    expected = jax.jit(products)(tokens, cotangent, sizes)
    mesh = MeshSpec(explicit=("data",)).build(jax.devices()[:8])
    with jax.set_mesh(mesh):
        rows = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
        split = jax.jit(products)(jax.device_put(tokens, rows), jax.device_put(cotangent, rows), sizes)
        forward, input_gradient, kernel_gradient, gradient = split
        assert jax.typeof(forward).sharding.spec[0] == "data"
        assert jax.typeof(input_gradient).sharding.spec[0] == "data"
        for whole_sum in (kernel_gradient, gradient):
            assert all(entry is None for entry in jax.typeof(whole_sum).sharding.spec)
        whole = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        replicated = jax.jit(products)(jax.device_put(tokens, whole), jax.device_put(cotangent, whole), sizes)
    for found in (split, replicated):
        for got, want in zip(found, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    empty = jax.jit(products)(tokens[:0], cotangent[:0], jnp.zeros((2,), jnp.int32))
    for got, want in zip(empty, (cotangent[:0], tokens[:0], jnp.zeros_like(kernel), jnp.zeros_like(kernel)),
                         strict=True):
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("layered", [False, True], ids=["matrices", "stack-layer"])
@pytest.mark.parametrize("scattered", [False, True])
def test_a_gathered_projection_is_its_gathered_rows_projection_in_every_order(scattered, layered):
    """`gemm.gemm`'s `gather` (row 1376) on a CPU (XLA's program): the projection of `tokens[gather]`, its
    value, its tangent and its reverse pass (each token's cotangent the sum of the slots reading it, a token
    no slot reads zero) bitwise the projection of the rows gathered first, plain and paired; `scattered`,
    each row written at its entry of a permutation (`scatter`), bitwise those rows put there after, its
    cotangents read back through it; `layered`, the matrices a stack's layer (a scanned MoE layer's gate and
    up over its gathered tokens, row 5391), bitwise that layer's matrices'; a gathered dense call, and a
    scattered dense one, refused."""
    rng = np.random.default_rng(3)
    tokens = jnp.asarray(rng.normal(size=(40, 64)), jnp.float32)
    matrices = jnp.asarray(rng.normal(size=(2, *((3,) if layered else ()), 3, 64, 32)) / 8, jnp.float32)
    layer = jnp.asarray(1, jnp.int32) if layered else None
    a_gathered_projection_is_its_gathered_rows_projection(tokens, matrices, rng, scattered=scattered,
                                                          layer=layer)
    table = jnp.asarray(rng.integers(0, 30, 72), jnp.int32)
    with pytest.raises(ValueError):
        gemm.gemm(tokens, matrices.reshape(-1, 64, 32)[0], None, out_dtype=jnp.float32, gather=table)
    with pytest.raises(ValueError):
        gemm.gemm(tokens, matrices.reshape(-1, 64, 32)[0], None, out_dtype=jnp.float32,
                  scatter=jnp.arange(40, dtype=jnp.int32))


@pytest.mark.parametrize(("name", "layered"), [("sm90", False), ("sm90", True), ("sm100", True)],
                         ids=["h100", "h100-stack-layer", "b200-stack-layer"])
@pytest.mark.parametrize("scattered", [False, True])
def test_a_gathered_projections_reverse_pass_reads_its_rows_where_they_lie(scattered, name, layered):
    """`a_gathered_projection_is_its_gathered_rows_projection` through the template, interpreted (row 1376's
    reverse pass): on an H100 the kernel gradient reads the tokens through the gather table and a scattered
    product's cotangent through its slots (`gemm_t`'s `gather`), and the input gradient reads that cotangent
    through the slots (`gemm`'s `gather`), none of them gathered first; `layered`, the matrices a stack's
    layer read in place (row 5391), the pair one launch on an H100 and its two products apart on a B200
    (tcgen05's bundles take no pair, `Family.takes_pairs`: `gemm._apart`, each product at the layer). bf16
    throughout, so each gradient is the template's own product (an fp32 cotangent's would be XLA's);
    integer values, so every sum is exact whatever tile each launch derives, and each side's one rounding
    the same."""
    rng = np.random.default_rng(5)
    # 64 deep and wide, one step and one column tile a launch: every path the cell names at a quarter of
    # the interpreter's work at 128 (row 8942: 130 s past its group's 120).
    tokens = jnp.asarray(rng.integers(-3, 4, size=(40, 64)), jnp.bfloat16)
    stacked = (3,) if layered else ()
    matrices = jnp.asarray(rng.integers(-3, 4, size=(2, *stacked, 3, 64, 64)), jnp.bfloat16)
    layer = jnp.asarray(2, jnp.int32) if layered else None
    # Interpreted on the host's CPU, whatever GPU it has: a B200's kernels compiled for a GH200 abort its
    # process (row 7130).
    with (generation.assumed(name), generation.chosen(grouped_matmul="mosaic"),
          jax.default_device(jax.devices("cpu")[0])):
        a_gathered_projection_is_its_gathered_rows_projection(tokens, matrices, rng, scattered=scattered,
                                                              interpret_on_cpu=True, layer=layer)


def a_gathered_projection_is_its_gathered_rows_projection(tokens, matrices, rng, *, scattered: bool,
                                                          interpret_on_cpu: bool = False, layer=None):
    """The projection of `tokens[gather]` (72 slots of 40 tokens, three groups, one empty, rows past them),
    its value, its tangent and its reverse pass (each token's cotangent the sum of the slots reading it, a
    token no slot reads zero), bitwise the projection of the rows gathered first, paired; `scattered`, each
    row written at its entry of a permutation, bitwise those rows put there after, its cotangents read
    back through it; with `layer`, `matrices` two stacks read at that layer (`gemm`'s `layer`), bitwise the
    layer's matrices' projection, each stack's gradient that layer's, every other zero; and an unpaired one
    whose epilogue reads its accumulator through its rounding alone, the rounding a named output (`gemm`'s
    `named`, row 7131's residual), its reverse pass bitwise the gathered rows' (the saved output read back
    where its rows landed: the scattered product's P0, [[4], [2]] for [[2], [4]]). Its outputs, tangents and
    cotangents are in the tokens' dtype, the cotangents integers."""
    table = jnp.asarray(rng.integers(0, 30, 72), jnp.int32)
    slots = jnp.asarray(rng.permutation(72), jnp.int32) if scattered else None
    sizes = jnp.asarray([20, 0, 44], jnp.int32)
    options = {"out_dtype": tokens.dtype, "out_dtypes": (tokens.dtype,) * 2,
               "interpret_on_cpu": interpret_on_cpu}

    def fused(tokens, first, second):
        return gemm.gemm(tokens, first, sizes, paired=second, gather=table, scatter=slots, layer=layer,
                         **options)

    def apart(tokens, first, second):
        first, second = (first, second) if layer is None else (first[layer], second[layer])
        outs = gemm.gemm(tokens[table], first, sizes, paired=second, **options)
        return outs if slots is None else tuple(jnp.zeros_like(out).at[slots].set(out) for out in outs)

    operands = (tokens, matrices[0], matrices[1])
    tangents = tuple(jnp.asarray(rng.integers(-2, 3, size=operand.shape), operand.dtype)
                     for operand in operands)
    cotangent = tuple(jnp.asarray(rng.integers(-2, 3, size=(72, matrices.shape[-1])), tokens.dtype)
                      for _ in range(2))
    for check in (lambda f: f(*operands), lambda f: jax.jvp(f, operands, tangents),
                  lambda f: jax.vjp(f, *operands)[1](cotangent)):
        jax.tree.map(np.testing.assert_array_equal, check(fused), check(apart))
    unread = np.setdiff1d(np.arange(tokens.shape[0]), np.asarray(table))
    assert np.all(np.asarray(jax.vjp(fused, *operands)[1](cotangent)[0])[unread] == 0)

    def squared(acc):  # the accumulator read through its rounding alone, the named second output
        rounded = acc.astype(tokens.dtype)
        return rounded * rounded, rounded

    named = {**options, "epilogue": squared, "named": (None, "rounded")}

    def named_fused(tokens, first):
        return gemm.gemm(tokens, first, sizes, gather=table, scatter=slots, layer=layer, **named)

    def named_apart(tokens, first):
        outs = gemm.gemm(tokens[table], first if layer is None else first[layer], sizes, **named)
        return outs if slots is None else tuple(jnp.zeros_like(out).at[slots].set(out) for out in outs)

    jax.tree.map(np.testing.assert_array_equal, jax.vjp(named_fused, *operands[:2])[1](cotangent),
                 jax.vjp(named_apart, *operands[:2])[1](cotangent))


@runs("sm100")
@pytest.mark.parametrize("out", ["bfloat16", "float32"])
@pytest.mark.parametrize("sizes", ["ragged", "one-group", "dense"])
def test_a_blackwells_projection_and_its_gradients_hold_check_1(sizes, out):
    """A Blackwell's projection of bf16 operands (grouped, or a dense layer's where no groups are given) into
    `out`, its forward and both gradients on the launches a B200 derives for them, interpreted: an fp32
    output asked at the highest precision summed a window at a time (`product.promoted`; tcgen05's own
    windows, `Tcgen05Rings.persist`'s `summed`, row 5209), each as exact as XLA's product from float64,
    unrouted rows and empty groups included."""
    group_sizes = None if sizes == "dense" else jnp.asarray(SIZES[sizes], jnp.int32)
    rows = 200 if sizes == "dense" else sum(SIZES[sizes]) + 9
    dtype = jnp.dtype(out)

    def draw(seed):
        rng = np.random.default_rng(seed)
        shape = (256, 128) if group_sizes is None else (group_sizes.shape[0], 256, 128)
        return (jnp.asarray(rng.normal(size=(rows, 256)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=shape) / 16, jnp.bfloat16),
                jnp.asarray(rng.normal(size=(rows, 128)), dtype))

    def on_xla(wide=None):
        def product(x, w):
            x, w = (x, w) if wide is None else (x.astype(wide), w.astype(wide))
            into = dtype if wide is None else wide
            if group_sizes is None:
                return jnp.matmul(x, w, preferred_element_type=into)
            return jax.lax.ragged_dot(x, w, group_sizes, preferred_element_type=into)
        return product

    def on_the_template(x, w, dy):
        precision = "highest" if dtype == jnp.float32 else None
        return with_gradients(lambda x, w: gemm.gemm(x, w, group_sizes, compute=jnp.bfloat16, out_dtype=dtype,
                                                     precision=precision, interpret_on_cpu=True))(x, w, dy)

    with generation.assumed("sm100"):
        # Mosaic's interpreter's ordered callbacks run eagerly: a jitted `linear_call` cannot carry them.
        assert_as_exact(on_the_template, jax.jit(with_gradients(on_xla())), draw, f"sm100 {sizes} into {out}",
                        seeds=SEEDS, truth_fn=with_gradients(on_xla(jnp.float64)))


def _dense_case(generation_name: str, made: grouped_cells.Made, matrix: tuple[int, int], rows: int,
                dtype=jnp.bfloat16, stored=None) -> grouped_matmul.GroupedCase:
    """The `GroupedCase` of a dense product of `dtype` operands on `generation_name` making `made` of `matrix`
    `[in, out]` over `rows` tokens into `stored` (the receipts' own output dtype, `grouped_cells.STORED`,
    where None)."""
    from dew.nn.kernels.templates.product import declared

    width_in, width_out = matrix
    stored = jnp.dtype(grouped_cells.STORED[made] if stored is None else stored)
    shaped = functools.partial(jax.ShapeDtypeStruct, dtype=dtype)
    sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
    product = declared(dtype, None, outputs=(stored,))
    with generation.assumed(generation_name):
        if made == grouped_cells.Made.KERNEL_GRADIENT:
            call = grouped_matmul.tgmm_call(shaped((rows, width_in)), shaped((rows, width_out)), sizes,
                                            out_dtype=stored, interpret=False, dense=True)
        else:
            transposed = made == grouped_cells.Made.INPUT_GRADIENT
            tokens = shaped((rows, width_out if transposed else width_in))
            call = grouped_matmul.gmm_call(tokens, shaped((1, width_in, width_out)), sizes, out_dtype=stored,
                                           trans_rhs=transposed, interpret=False, dense=True)
        return grouped_matmul.grouped_case(*call, product=product, whole=True)


def _route(case: grouped_matmul.GroupedCase) -> str:
    """The kernel the grouped selector runs `case` on, as its target's process would."""
    with generation.assumed(case.call.target.generation):
        return selection.select(grouped_matmul.GROUPED, case).kernel


def test_a_timed_product_is_recognized_over_its_receipts_rows_and_arithmetic_alone():
    """A receipt is evidence for the products it timed alone (`grouped_cells.Timed`, row 6794): each product a
    receipt timed (`grouped_cells.timings`) is recognized as its matrix over that receipt's rows
    (`Kind.span`) in the arithmetic and output dtype it timed, and not past those rows (a fine-tune's 2,048
    took the 4B key's cell and its launches made the step's products 1.75 ms slower) nor in another
    arithmetic (a float32 call's TF32 kernel) or output dtype (another staging and memory)."""
    for timing in grouped_cells.timings():
        if timing.formed:
            continue
        rows = timing.timed.fewest
        assert _dense_case(timing.generation, timing.made, timing.matrix, rows).kind.span == timing.timed.span
        others = grouped_cells.MEASURED[timing.generation, timing.made, False]
        stored = jnp.dtype(grouped_cells.STORED[timing.made]).name
        if not any(timed.holds(timing.matrix, rows - 1, timing.timed.dtype, stored) for timed in others):
            assert _dense_case(timing.generation, timing.made, timing.matrix, rows - 1).kind.matrix is None
        wide = _dense_case(timing.generation, timing.made, timing.matrix, rows, dtype=jnp.float32,
                           stored=jnp.float32)
        assert wide.kind.matrix is None, timing
        if timing.timed.stored is not None:
            other = "bfloat16" if timing.timed.stored == "float32" else "float32"
            assert _dense_case(timing.generation, timing.made, timing.matrix, rows,
                               stored=other).kind.matrix is None, timing


@pytest.mark.parametrize("board", grouped_cells.SCOREBOARDS, ids=lambda board: board.evidence.split()[0])
def test_a_scoreboards_products_run_the_template_where_at_parity_and_its_kind_elsewhere(board):
    """A dense kind runs the template whole only where every product its receipt timed is at parity in time
    and temporaries (kernel-templates.md §1; bible §6, 2026-11-24); a kind with a product lagging keeps XLA's
    dot, its products at parity the template's over their timed rows, its lagging ones and every product it
    did not time XLA's (`grouped_cells.Scoreboard`)."""
    untimed = (3 * 1024 + 64, 8 * 1024 + 64)
    for made, projections in board.times.items():
        kind = board.winner(made)
        assert _route(_dense_case(board.generation, made, untimed, board.rows)) == kind, made
        for name in projections:
            route = _route(_dense_case(board.generation, made, grouped_cells.PROJECTIONS[name], board.rows))
            expected = "mosaic" if kind == "mosaic" or board.at_parity(made, name) else "xla"
            assert route == expected, (made, name)


def test_a_generation_no_receipt_timed_runs_the_template():
    """A generation no receipt timed (a B300) runs the template at every dense kind, whatever its
    instruction family's other generations measured (the B200's lagging 4B cells are the B200's)."""
    for made in grouped_cells.Made:
        if made != grouped_cells.Made.EXACT_KERNEL_GRADIENT:
            assert _route(_dense_case("sm103", made, (2560, 9216), 24576)) == "mosaic", made


@pytest.mark.parametrize("gpu", [pytest.param(name, marks=runs(name)) for name in ("sm80", "sm90")])
def test_a_ragged_grouped_forward_at_its_derived_tile_is_its_groups_products(gpu):
    """A grouped forward of ragged groups (an empty one, one of a single row, the rest uneven), its tile
    derived where its rows weigh its groups' padding (`derivation._padded_side`), interpreted as an A100 and
    an H100 run it: each row its group's matrix's product, exactly (small integers), the derived tile one
    the launch holds legal."""
    rng = np.random.default_rng(23)
    sizes = np.asarray([0, 1, 37, 130, 64, 3, 90, 75], np.int32)
    rows = jnp.asarray(rng.integers(-3, 4, size=(int(sizes.sum()), 128)), jnp.bfloat16)
    matrices = jnp.asarray(rng.integers(-3, 4, size=(8, 128, 256)), jnp.bfloat16)
    launch = grouped_matmul.MosaicGmm()
    arguments = launch.bind(rows, matrices, jnp.asarray(sizes), out_dtype="float32", trans_rhs=False,
                            interpret=True)
    with generation.assumed(gpu), generation.chosen(grouped_matmul="mosaic"):
        target = generation.gpu_target()
        tile = launch.derived(arguments, target)
        got = gemm.gemm(rows, matrices, jnp.asarray(sizes), out_dtype=jnp.float32, interpret_on_cpu=True)
    assert launch.legal(arguments, tile, target), tile
    group_of = np.repeat(np.arange(8), sizes)
    exact = np.einsum("rk,rkn->rn", np.asarray(rows, np.float64), np.asarray(matrices, np.float64)[group_of])
    np.testing.assert_array_equal(np.asarray(got, np.float64), exact)


def test_mosaic_completed_starts_unstarted_gathered_b200_producer():
    """Pinning an unstarted gathered B200 tile completes it to a legal producer share and lowers the same
    producer allocation in `launch_of` (rows 7083, 11897)."""
    from dew.nn.kernels.templates.gemm import tiling

    launch = grouped_matmul.MosaicGmm()
    ba = launch.bind(jax.ShapeDtypeStruct((4096, 1024), jnp.bfloat16),
                     jax.ShapeDtypeStruct((4, 1024, 1024), jnp.bfloat16),
                     jax.ShapeDtypeStruct((4,), jnp.int32), (), (), None, None,
                     jax.ShapeDtypeStruct((4096,), jnp.int32),
                     out_dtype="bfloat16", trans_rhs=False, interpret=False)
    unstarted = tiling.Tiles(128, 128, 64, 4, 1, warpgroups=1, family="tcgen05")
    with generation.assumed("sm100"), tuning.pinned({grouped_matmul.MosaicGmm: unstarted}):
        sm100 = generation.gpu_target()
        pinned = launch.pinned(ba)
        planned = launch.launch_of(ba, pinned, sm100)
    assert pinned is not None and pinned.producers.of(launch.KERNEL) is not None
    assert launch.legal(ba, pinned, sm100)
    assert planned.producer == pinned.producers.of(launch.KERNEL)


def test_mosaic_tgmm_split_pricing_preserves_final_output_dtype():
    """`MosaicTgmm._contraction` and `_walk` price split walks using each bound call's final `out_dtype`
    rather than fp32 partials' width (rows 8296, 11888, 11947)."""
    launch = grouped_matmul.MosaicTgmm()
    target = generation.gpu_target("sm90")
    ba_bf16 = launch.bind(jax.ShapeDtypeStruct((8192, 128), jnp.bfloat16),
                          jax.ShapeDtypeStruct((8192, 128), jnp.bfloat16),
                          jax.ShapeDtypeStruct((1,), jnp.int32),
                          out_dtype="bfloat16", interpret=False, dense=True)
    ba_fp32 = launch.bind(jax.ShapeDtypeStruct((8192, 128), jnp.bfloat16),
                          jax.ShapeDtypeStruct((8192, 128), jnp.bfloat16),
                          jax.ShapeDtypeStruct((1,), jnp.int32),
                          out_dtype="float32", interpret=False, dense=True)
    split_tile = tiling.Tiles(64, 128, 64, 3, 1, warpgroups=1, family="wgmma", splits=2)
    with generation.assumed("sm90"):
        assert launch.legal(ba_bf16, split_tile, target) and launch.legal(ba_fp32, split_tile, target)
        assert launch._contraction(ba_bf16, split_tile, target)["stored"] == 2
        assert launch._contraction(ba_fp32, split_tile, target)["stored"] == 4
        w_bf16 = launch._walk(ba_bf16, split_tile, target)
        w_fp32 = launch._walk(ba_fp32, split_tile, target)
        w_partials = launch._walk(launch._partials(ba_bf16), split_tile, target)
        assert w_bf16 is not None and w_fp32 is not None and w_partials is not None
        assert w_bf16.stored < w_fp32.stored and w_bf16.stored < w_partials.stored
        assert w_bf16.seconds(split_tile.splits) < w_fp32.seconds(split_tile.splits)


def test_mosaic_planning_uses_explicit_target_not_ambient_gpu():
    """Passing an explicit `target` to `MosaicGmm.derived`, `legal`, `takes`, `launch_of` and `staged_bytes`
    plans for that target, including its recorded refusals, rather than the ambient `generation.gpu_target()`
    (row 10986)."""
    from dew.nn.kernels.templates import report

    launch = grouped_matmul.MosaicGmm()
    ba = launch.bind(jax.ShapeDtypeStruct((4096, 1024), jnp.bfloat16),
                     jax.ShapeDtypeStruct((4, 1024, 1024), jnp.bfloat16),
                     jax.ShapeDtypeStruct((4,), jnp.int32),
                     out_dtype="bfloat16", trans_rhs=False, interpret=False)
    sm86 = generation.gpu_target("sm86")
    with generation.assumed("sm86"):
        expected_tile = launch.derived(ba, sm86)
        assert expected_tile is not None
        expected_staged = launch.staged_bytes(ba, expected_tile, sm86)
        expected_launch = launch.launch_of(ba, expected_tile, sm86)
        sm86_identity = report._identity(launch, ba, sm86)
    with generation.assumed("sm80"):
        got_tile = launch.derived(ba, sm86)
        assert got_tile == expected_tile
        assert launch.legal(ba, got_tile, sm86) and launch.takes(ba, got_tile, sm86)
        assert launch.staged_bytes(ba, got_tile, sm86) == expected_staged
        assert launch.launch_of(ba, got_tile, sm86).most == expected_launch.most
        with report.refusals():
            report.refuse([(sm86_identity, expected_tile, (launch.KERNEL,))])
            assert not launch.legal(ba, expected_tile, sm86)
            assert not launch.takes(ba, expected_tile, sm86)


@lowers("sm90")
def test_gemm_register_model_fits_sm90_epilogue_and_tgmm_census():
    """The GEMM register model holds legal multi-output epilogues and `dew_tgmm` spill-free under sm90 ptxas
    while refusing wide multi-output tiles past the single-warpgroup budget, and scopes wgmma `tgmm` scalar
    addressing away from tcgen05 (rows 8723, 11946, 11947)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    gmm, tgmm = grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm()
    sm90, sm100 = generation.gpu_target("sm90"), generation.gpu_target("sm100")
    x = jax.ShapeDtypeStruct((4096, 4096), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((1, 4096, 14336), jnp.bfloat16)
    dy = jax.ShapeDtypeStruct((4096, 14336), jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
    with generation.assumed("sm90"):
        two_out_hooks = grouped_matmul.Hooks(
            epilogue=lambda acc: (acc.astype(jnp.bfloat16), (acc + 1.0).astype(jnp.bfloat16)),
            out_dtypes=("bfloat16", "bfloat16"))
        ba_two = gmm.bind(x, w, sizes, out_dtype="bfloat16", trans_rhs=False, interpret=False,
                          hooks=two_out_hooks)
        wide_last = grouped_matmul.Hooks(
            epilogue=lambda acc: (acc.astype(jnp.bfloat16), acc + 1.0),
            out_dtypes=("bfloat16", "float32"))
        ba_wide_last = gmm.bind(x, w, sizes, out_dtype="bfloat16", trans_rhs=False, interpret=False,
                                hooks=wide_last)
        ba_dw = tgmm.bind(x, dy, sizes, out_dtype="float32", interpret=False, dense=False)
        t256 = tiling.Tiles(64, 256, 64, 3, 56, warpgroups=1, family="wgmma")
        # 128-col mixed (bf16, fp32) exercises per-output staging widths: 64-col bf16 and 32-col fp32.
        t128 = tiling.Tiles(64, 128, 64, 3, 56, warpgroups=1, family="wgmma")
        assert gmm.legal(ba_two, t256, sm90)
        assert not gmm.legal(ba_wide_last, t256, sm90)
        assert gmm.legal(ba_wide_last, t128, sm90)
        assert tgmm.legal(ba_dw, t256, sm90)
        cases = [
            KernelCase("two_out_64x256", lambda: gmm.lowering(ba_two, t256)),
            KernelCase("mixed_64x128", lambda: gmm.lowering(ba_wide_last, t128)),
            KernelCase("tgmm_64x256", lambda: tgmm.lowering(ba_dw, t256)),
        ]
        records = compile_mosaic_kernels(cases, "sm90")
    for rec in records:
        assert not rec.get("error"), rec
        assert rec["kernels"], rec
        assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec
    with generation.assumed("sm100"):
        t128_b200 = tiling.Tiles(128, 128, 64, 3, 1, warpgroups=1, family="tcgen05")
        ba_dw_b200 = tgmm.bind(x, dy, sizes, out_dtype="float32", interpret=False, dense=True)
        assert tgmm.legal(ba_dw_b200, t128_b200, sm100)


@pytest.mark.parametrize("tpu_name", ["v5e", "v6e"])
def test_tpu_gemm_derives_legal_vmem_bounded_tiles_from_step_seconds(tpu_name):
    """On TPU v5e and v6e (`mxu`), `MosaicGmm` and `MosaicTgmm` derive legal VMEM-bounded tiles through the
    shared `pricing.step_seconds` cost model (rows 10152, 11944)."""
    from dew.nn.kernels.templates.gemm import derivation

    target = generation.gpu_target(tpu_name)
    gmm, tgmm = grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm()
    sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
    with generation.assumed(tpu_name):
        for depth, width in ((4096, 4096), (4096, 28672), (5120, 6144), (5120, 7168)):
            x = jax.ShapeDtypeStruct((8192, depth), jnp.bfloat16)
            w = jax.ShapeDtypeStruct((1, depth, width), jnp.bfloat16)
            w_t = jax.ShapeDtypeStruct((1, depth, width), jnp.bfloat16)
            dy = jax.ShapeDtypeStruct((8192, width), jnp.bfloat16)
            for ba_gmm in (
                gmm.bind(x, w, sizes, out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=True),
                gmm.bind(dy, w_t, sizes, out_dtype="bfloat16", trans_rhs=True, interpret=False, dense=True),
            ):
                tile = gmm.derived(ba_gmm, target)
                assert tile is not None and gmm.legal(ba_gmm, tile, target)
                assert gmm.staged_bytes(ba_gmm, tile, target) <= target.scoped_vmem
            ba_dw = tgmm.bind(x, dy, sizes, out_dtype="float32", interpret=False, dense=True)
            tile_dw = tgmm.derived(ba_dw, target)
            assert tile_dw is not None and tgmm.legal(ba_dw, tile_dw, target)
            assert tgmm.staged_bytes(ba_dw, tile_dw, target) <= target.scoped_vmem
        # Verify unified `pricing.step_seconds` prices deeper K cheaper than collapsed 128-K on TPU.
        t_deep = tiling.Tiles(512, 1024, 1024, 2, 1, warpgroups=1, family="mxu")
        t_shallow = tiling.Tiles(512, 1024, 128, 2, 1, warpgroups=1, family="mxu")
        fam = pricing.family_of(t_deep, jnp.bfloat16)
        assert (derivation._tile_cost(t_deep, family=fam, target=target, dtype=jnp.bfloat16,
                                      rows=8192, width=4096, padded=False)
                < derivation._tile_cost(t_shallow, family=fam, target=target, dtype=jnp.bfloat16,
                                        rows=8192, width=4096, padded=False))


@INTERPRETS
def test_ampere_multi_wave_dense_dx_unifies_planner_and_kernel_and_holds_check_1():
    """On Ampere (`mma.sync`), `pricing.dense_tail` prices multi-wave persistent L2 drift once so planner
    (`_merges`, `_balances`) and kernel (`_dense_tail`) agree by construction, a tail's partials within the
    reference's workspace (row 12850), and a multi-wave dense input gradient holds check 1 via
    `assert_as_exact` (rows 5363, 11945, 11947)."""

    sm80 = generation.gpu_target("sm80")
    gmm, tgmm = grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm()
    sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
    with generation.assumed("sm80"):
        dy_deep = jax.ShapeDtypeStruct((8192, 28672), jnp.bfloat16)
        w_deep = jax.ShapeDtypeStruct((1, 4096, 28672), jnp.bfloat16)
        ba_deep = gmm.bind(dy_deep, w_deep, sizes, out_dtype="bfloat16", trans_rhs=True,
                           interpret=False, dense=True)
        tile_64x256 = tiling.Tiles(64, 256, 32, 4, 16, warpgroups=1, family="mma.sync", programs=2)
        tail_deep = mosaic_gpu._dense_tail(tile_64x256, jnp.bfloat16, (8192, 4096, 28672), (2, 2), 216,
                                           balanced=True)
        assert tail_deep is None
        assert not gmm.balanced(ba_deep, tile_64x256, sm80)
        # A whole A100's 216 programs' partials (14 MiB) exceed the 4 MiB cuBLAS's workspace is, so no
        # balanced tail there (row 12850); on 40 programs (2.6 MiB), a single-wave tail (64 tiles: waves=1,
        # rem=24, 0 drift): planner produces one Tail plan passed straight to emitter via `_stated` without
        # re-pricing.
        x_1w = jax.ShapeDtypeStruct((4096, 2048), jnp.bfloat16)
        dy_1w = jax.ShapeDtypeStruct((4096, 1024), jnp.bfloat16)
        ba_1w = tgmm.bind(x_1w, dy_1w, sizes, out_dtype="float32", interpret=False, dense=True)
        tile_1w = tiling.Tiles(128, 64, 32, 4, 8, warpgroups=1, family="mma.sync", programs=2)
        assert tgmm._stated(ba_1w, tile_1w, sm80)["balanced"] is False
        few = dataclasses.replace(sm80, multiprocessors=20)
        x_few = jax.ShapeDtypeStruct((4096, 512), jnp.bfloat16)
        ba_few = tgmm.bind(x_few, dy_1w, sizes, out_dtype="float32", interpret=False, dense=True)
        stated_few = tgmm._stated(ba_few, tile_1w, few)
        tail_few = mosaic_gpu._dense_tail(tile_1w, jnp.bfloat16, (512, 1024, 4096), (2, 2), 40,
                                          balanced=stated_few["balanced"])
        assert tail_few is stated_few["balanced"] and tail_few is not None
        assert tgmm.balanced(ba_few, tile_1w, few)
    # Multi-wave nonzero-tail check-1 witness: 320x128 -> grid=(5, 2)=10 tiles on 4 programs -> waves=2,
    # rem=2 (nonzero tail!). At l2=64KB and memory_rate=120GB/s, persistent L2 drift flips the tail decision
    # from balanced (no drift) to non-persistent (with drift).
    tile_mw = tiling.Tiles(64, 64, 32, 2, 2, warpgroups=1, family="mma.sync", programs=1)
    small_sm80 = dataclasses.replace(sm80, multiprocessors=4, l2_bytes=64 * 1024, measured={},
                                     memory_bandwidth=120e9)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(320, 128)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(128, 128)) / np.sqrt(128), jnp.bfloat16))

    def native(dy, w):
        return mosaic_gpu.gmm(dy, w[None], jnp.asarray([320], jnp.int32), tiles=tile_mw,
                              transpose_rhs=True, out_dtype=jnp.bfloat16, multiprocessors=4,
                              interpret=grouped_launch.INTERPRETED, dense=True)

    def plain(dy, w, wide=jnp.float32, out=jnp.bfloat16):
        return jnp.dot(dy.astype(wide), w.T.astype(wide), preferred_element_type=wide).astype(out)

    with generation.assumed(small_sm80):
        from dew.nn.kernels.templates import tail as tail_mod

        step_s = pricing.step_seconds(tile_mw, small_sm80, (2, 2))
        assert step_s is not None
        traffic = tail_mod.Traffic((5, 2), 2, 64 * 32 * 2, 64 * 32 * 2, step_s, 4, 120e9, 64 * 1024)
        # Unscoped, tail.balanced accepts the 2-tile tail; scoped (mma.sync's walk launched persistent), the
        # whole waves' drift refuses it, as dense_tail does.
        assert tail_mod.balanced(10, 4, 4, traffic, handoff_steps=0.0, scoped=False, partial_bytes=0)
        assert tail_mod.balanced(10, 4, 4, traffic, handoff_steps=0.0, scoped=True, partial_bytes=0) is None
        assert pricing.dense_tail(tile_mw, jnp.bfloat16, (320, 128, 128), (2, 2), small_sm80, 4) is None
        assert_as_exact(native, jax.jit(plain), draw, "sm80 multi-wave nonzero-tail dense dX", seeds=SEEDS,
                        truth_fn=functools.partial(plain, wide=jnp.float64, out=jnp.float64))


@lowers("sm80")
def test_sm80_prefetch_registers_and_tgmm_prefetch_predicate():
    """On sm80 (`mma.sync`), `pricing.prefetch_registers` rejects spilling `64x320` forward tiles while
    holding `128x128` forward and prefetching unstaged dense `tgmm` spill-free under ptxas (rows 5271,
    11947)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels


    sm80 = generation.gpu_target("sm80")
    gmm, tgmm = grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm()
    sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
    with generation.assumed("sm80"):
        x = jax.ShapeDtypeStruct((8192, 4096), jnp.bfloat16)
        w = jax.ShapeDtypeStruct((1, 4096, 14336), jnp.bfloat16)
        dy = jax.ShapeDtypeStruct((8192, 14336), jnp.bfloat16)
        ba_fwd = gmm.bind(x, w, sizes, out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=True)
        ba_fwd_interp = gmm.bind(x, w, sizes, out_dtype="bfloat16", trans_rhs=False, interpret=True,
                                 dense=True)
        ba_dw = tgmm.bind(x, dy, sizes, out_dtype="float32", interpret=False, dense=True)
        ba_dw_interp = tgmm.bind(x, dy, sizes, out_dtype="float32", interpret=True, dense=True)
        t320 = tiling.Tiles(64, 320, 32, 5, 8, warpgroups=1, family="mma.sync")
        t128 = tiling.Tiles(128, 128, 32, 5, 8, warpgroups=1, family="mma.sync", programs=2)
        t128_k64 = tiling.Tiles(128, 128, 64, 3, 8, warpgroups=1, family="mma.sync")
        assert not gmm.legal(ba_fwd, t320, sm80)
        assert gmm.legal(ba_fwd, t128, sm80)
        fam = pricing.family_of(t128, jnp.bfloat16)
        assert pricing.prefetch_registers(fam, t128, ba=ba_fwd) == 16
        assert gmm.registers(ba_fwd, t128, sm80) == gmm.registers(ba_fwd_interp, t128, sm80) + 16
        # TGMM skips fragment prefetch registers entirely at the call site.
        assert tgmm.registers(ba_dw, t128, sm80) == tgmm.registers(ba_dw_interp, t128, sm80)
        cases = [
            KernelCase("sm80_fwd_128x128x32", lambda: gmm.lowering(ba_fwd, t128)),
            KernelCase("sm80_dw_128x128x32", lambda: tgmm.lowering(ba_dw, t128)),
            KernelCase("sm80_dw_128x128x64", lambda: tgmm.lowering(ba_dw, t128_k64)),
        ]
        records = compile_mosaic_kernels(cases, "sm80")
    for rec in records:
        assert not rec.get("error"), rec
        assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


@INTERPRETS
@pytest.mark.parametrize(("rows", "depth", "mp"), [(600, 256, 4), (900, 128, 6)],
                         ids=["ragged-pairs2", "few-iters-pairs3"])
def test_hopper_cluster2_balanced_tail_holds_check_1(rows, depth, mp):
    """Hopper `cluster=2` dense forward (`gmm`) and kernel gradient (`tgmm`) with balanced tail across
    cluster pairs (`pairs = mp // 2`, leading `pair` axis on `workspace`/`ready`), including ragged
    `m % (2*bm) != 0` and `iterations < pairs`, hold check 1 via `assert_as_exact` (row 10151)."""

    tile = tiling.Tiles(128, 256, 64, 2, 1, warpgroups=2, cluster=2, family="wgmma")
    target = dataclasses.replace(generation.gpu_target("sm90"), multiprocessors=mp, measured={})
    assert pricing.dense_tail(tile, jnp.bfloat16, (rows, 256, depth), (2, 2), target, mp) is not None

    def draw_fwd(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(depth, 256)) / np.sqrt(depth), jnp.bfloat16))

    def fwd_native(x, w):
        return mosaic_gpu.gmm(x, w[None], jnp.asarray([rows], jnp.int32), tiles=tile, transpose_rhs=False,
                              out_dtype=jnp.bfloat16, multiprocessors=mp,
                              interpret=grouped_launch.INTERPRETED, dense=True)

    def fwd_plain(x, w, wide=jnp.float32, out=jnp.bfloat16):
        return jnp.dot(x.astype(wide), w.astype(wide), preferred_element_type=wide).astype(out)

    with generation.assumed(target):
        assert_as_exact(fwd_native, jax.jit(fwd_plain), draw_fwd, f"cluster2 tail fwd {rows}x{depth} mp{mp}",
                        seeds=SEEDS, truth_fn=functools.partial(fwd_plain, wide=jnp.float64, out=jnp.float64))

    if mp == 4:
        def draw_dw(seed):
            rng = np.random.default_rng(seed)
            return (jnp.asarray(rng.normal(size=(depth, 768)), jnp.bfloat16),
                    jnp.asarray(rng.normal(size=(depth, 256)) / np.sqrt(depth), jnp.bfloat16))

        def dw_native(x, dy):
            return mosaic_gpu.tgmm(x, dy, jnp.asarray([depth], jnp.int32), tiles=tile,
                                   out_dtype=jnp.float32, multiprocessors=mp,
                                   interpret=grouped_launch.INTERPRETED, dense=True)[0]

        def dw_plain(x, dy, wide=jnp.float32, out=jnp.float32):
            return jnp.dot(x.T.astype(wide), dy.astype(wide), preferred_element_type=wide).astype(out)

        with generation.assumed(target):
            assert pricing.dense_tail(tile, jnp.bfloat16, (768, 256, depth), (2, 2), target, mp) is not None
            assert_as_exact(dw_native, jax.jit(dw_plain), draw_dw, f"cluster2 tail dw 768x256x{depth}",
                            seeds=SEEDS,
                            truth_fn=functools.partial(dw_plain, wide=jnp.float64, out=jnp.float64))


@pytest.mark.parametrize("swapped", [False, True], ids=["rows", "swapped"])
def test_a_splits_partials_are_priced_as_its_launch_declares_them(swapped, monkeypatch):
    """A split's fp32 partials as its legality and its price count them (`MosaicGmm._slice_bytes`, row
    12850) are the ones its launch declares (`mosaic_gpu._declared`), its width in whole column units and,
    swapped, its rows padded to whole column tiles: an A100's 264 x 1985 product (its width launched 2016
    wide) and a 16-row decode by 20,480 columns swapped."""
    from dew.nn.kernels.templates.gemm import mosaic_gpu

    launch = grouped_matmul.MosaicGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    one = jax.ShapeDtypeStruct((1,), jnp.int32)
    rows, depth, width = (16, 4096, 20480) if swapped else (264, 1024, 1985)
    declared = []
    real = mosaic_gpu._declared

    def recording(*args, **kwargs):
        found = real(*args, **kwargs)
        declared.extend(each for each in found if each.dtype == jnp.float32)
        return found

    monkeypatch.setattr(mosaic_gpu, "_declared", recording)
    with generation.assumed("sm80"):
        target = generation.gpu_target()
        ba = launch.bind(bf16((rows, depth)), bf16((1, depth, width)), one, out_dtype="bfloat16",
                         trans_rhs=False, interpret=False, dense=True)
        if swapped:
            tile = dataclasses.replace(launch._derived(launch._swapped(ba, target), target), swapped=True)
        else:
            tile = launch.derived(ba, target)
        tile = dataclasses.replace(tile, splits=2)
        fn, args = launch.lowering(ba, tile)
        jax.eval_shape(fn, *args)
        assert len(declared) == 1, declared
        assert 2 * launch._slice_bytes(ba, tile, target) == math.prod(declared[0].shape) * 4, declared


def _traced_on(target, fn, args):
    """`fn` of `args` traced on `target` assumed, whatever generation its compile names."""
    def traced(*arrays):
        with generation.assumed(target):
            return fn(*arrays)

    return traced, args


@lowers("sm90")
def test_hopper_cluster2_balanced_tail_lowers_spill_free():
    """Hopper `128x256x64 cluster=2` dense `gmm` and `tgmm` with balanced tail compile under sm90 ptxas
    with zero spills (row 10151), on an H100 PCIe's 114 multiprocessors, whose 57 pairs' partials (30 MiB)
    are within cuBLAS's 32 MiB workspace; a GH200's 66 pairs' (33 MiB) are not, so it takes none (row
    12850)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    gmm, tgmm = grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm()
    sm90 = generation.gpu_target("sm90")
    sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
    with generation.assumed("sm90"):
        # Qwen3.5-4B q dX outlier shape (row 10151): 2048x2560x8192 (fwd) and 2560x2048x8192 (dw) at
        # 128x256x64 cluster=2 -> grid 8x10=80 super-tiles across 66 cluster pairs -> waves=1, tail=14,
        # steps=128 (nonzero balanced tail beats walking even after fixed_up partial write+read traffic).
        # By contrast, deep multi-wave 4B up-projection (16384x9728x2560, 16384x2560x9728) declines balanced
        # tail once fixed_up partial traffic exceeds the wave saving.
        from dew.nn.kernels.templates.gemm import pricing

        tile_c2 = tiling.Tiles(128, 256, 64, 3, 4, warpgroups=2, cluster=2, family="wgmma")
        assert pricing.dense_tail(tile_c2, jnp.bfloat16, (2048, 2560, 8192), (2, 2), sm90, 132) is None
        sm90 = dataclasses.replace(sm90, multiprocessors=114)
        assert pricing.dense_tail(tile_c2, jnp.bfloat16, (16384, 9728, 2560), (2, 2), sm90, 114) is None
        assert pricing.dense_tail(tile_c2, jnp.bfloat16, (16384, 2560, 9728), (2, 2), sm90, 114) is None
        assert pricing.dense_tail(tile_c2, jnp.bfloat16, (2048, 2560, 8192), (2, 2), sm90, 114) is not None
        x_fwd = jax.ShapeDtypeStruct((2048, 8192), jnp.bfloat16)
        w_fwd = jax.ShapeDtypeStruct((1, 8192, 2560), jnp.bfloat16)
        x_dw = jax.ShapeDtypeStruct((8192, 2560), jnp.bfloat16)
        dy_dw = jax.ShapeDtypeStruct((8192, 2048), jnp.bfloat16)
        ba_fwd = gmm.bind(x_fwd, w_fwd, sizes, out_dtype="bfloat16", trans_rhs=False, interpret=False,
                          dense=True)
        ba_dw = tgmm.bind(x_dw, dy_dw, sizes, out_dtype="float32", interpret=False, dense=True)
        tile_c2_split = dataclasses.replace(tile_c2, splits=2)
        assert gmm.legal(ba_fwd, tile_c2, sm90) and gmm.balanced(ba_fwd, tile_c2, sm90)
        # Split two ways its 2048 rows' fp32 partials (40 MiB) are past the workspace, pinned or derived
        # (row 12850); 1024 rows' (20 MiB) within it, and a split tile takes no tail.
        assert not gmm.legal(ba_fwd, tile_c2_split, sm90)
        ba_half = gmm.bind(jax.ShapeDtypeStruct((1024, 8192), jnp.bfloat16), w_fwd, sizes,
                           out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=True)
        assert gmm.legal(ba_half, tile_c2_split, sm90) and not gmm.balanced(ba_half, tile_c2_split, sm90)
        assert gmm._stated(ba_half, tile_c2_split, sm90)["balanced"] is False
        assert tgmm.legal(ba_dw, tile_c2, sm90) and tgmm.balanced(ba_dw, tile_c2, sm90)
        cases = [
            KernelCase("sm90_cluster2_tail_fwd", lambda: _traced_on(sm90, *gmm.lowering(ba_fwd, tile_c2))),
            KernelCase("sm90_cluster2_split2_fwd",
                       lambda: _traced_on(sm90, *gmm.lowering(ba_half, tile_c2_split))),
            KernelCase("sm90_cluster2_tail_dw", lambda: _traced_on(sm90, *tgmm.lowering(ba_dw, tile_c2))),
        ]
        records = compile_mosaic_kernels(cases, "sm90")
    for rec in records:
        assert not rec.get("error"), rec
        assert rec["kernels"], rec
        assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


def test_explicit_target_derivation_sm120_and_refusal_under_mismatched_ambient_gpu():
    """Deriving and pricing `MosaicGmm` for an explicit `target` reads that target's bundle (`MmaFed` 60
    held regs on sm120 vs 65 on sm100) and explicit-target refusals regardless of the ambient GPU, with no
    `_with_target` override (rows 12011, 11889)."""
    from dew.nn.kernels.templates import report

    gmm = grouped_matmul.MosaicGmm()
    assert not hasattr(gmm, "_with_target")
    sm120 = generation.gpu_target("sm120")
    sm86 = generation.gpu_target("sm86")
    x = jax.ShapeDtypeStruct((4096, 4096), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((1, 4096, 4096), jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
    ba = gmm.bind(x, w, sizes, out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=True)

    with generation.assumed("sm120"):
        expected_tile = gmm.derived(ba, sm120)
        assert expected_tile is not None
        expected_regs = gmm.registers(ba, expected_tile, sm120)

    for ambient in ("sm90", "sm100"):
        with generation.assumed(ambient):
            got_tile = gmm.derived(ba, sm120)
            assert got_tile == expected_tile
            assert gmm.registers(ba, expected_tile, sm120) == expected_regs

    # Explicit-target refusal recorded under sm86 is honored when deriving for sm86 under ambient sm90.
    with report.refusals():
        with generation.assumed("sm86"):
            first = gmm.derived(ba, sm86)
            assert first is not None
            sm86_id = report._identity(gmm, ba, sm86)
        report.refuse([(sm86_id, first, frozenset({gmm.KERNEL}))])
        with generation.assumed("sm90"):
            stepped = gmm.derived(ba, sm86)
            assert stepped != first


def test_stream_rms_normalized_fuses_through_fused_gemm_prologue():
    """Residual-stream RMSNorm (`norms.stream_rms_normalized`) feeding `fused_gemm.product` splits its
    stream sum + statistics into the XLA pre-pass and fuses the scaling into the GEMM prologue through
    `fused_gemm.analysis` while holding Check 1 on value and VJP (rows 6104, 12137)."""
    from jax.experimental.pallas import fuser

    from dew.nn.kernels import fused_gemm
    from dew.nn.norms import stream_rms_normalized

    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)

    def body(x, branch, scale, w):
        _, normed = stream_rms_normalized(x, branch, scale, 1e-6, jnp.bfloat16, scale_offset=False,
                                          scale_after_cast=False, fp32_statistics=True,
                                          implementation="xla")
        return fused_gemm.product(normed, w, out_dtype=jnp.bfloat16)

    with generation.assumed("sm90"):
        found = fused_gemm.analysis(body, bf16((8192, 4096)), bf16((8192, 4096)), f32((4096,)),
                                    bf16((4096, 4096)))
    assert found.prologue is not None and found.prologue.kernel == "fused", found

    def draw(seed):
        rng = np.random.default_rng(12137 + seed)
        return (
            jnp.asarray(rng.standard_normal((64, 64)) / 8.0, jnp.bfloat16),
            jnp.asarray(rng.standard_normal((64, 64)) / 8.0, jnp.bfloat16),
            jnp.asarray(rng.uniform(0.8, 1.2, size=(64,)), jnp.float32),
            jnp.asarray(rng.standard_normal((64, 64)) / 8.0, jnp.bfloat16),
            jnp.asarray(rng.standard_normal((64, 64)), jnp.bfloat16),
        )

    def run_fused(x, branch, scale, w, dy):
        with generation.assumed("sm90"), generation.chosen(gemm_prologue="fused"):
            y, pull = jax.vjp(fuser.fuse(body), x, branch, scale, w)
            return y, pull(dy)

    def run_ref(x, branch, scale, w, dy, *, dtype=jnp.bfloat16, wide=jnp.float32):
        def fn(a, b, s, m):
            _, normed = stream_rms_normalized(a, b, s, 1e-6, dtype, scale_offset=False,
                                              scale_after_cast=False, fp32_statistics=True,
                                              implementation="xla")
            return jax.lax.dot(normed, m, preferred_element_type=wide).astype(dtype)

        y, pull = jax.vjp(fn, x.astype(dtype), branch.astype(dtype), scale.astype(wide), w.astype(dtype))
        return y, pull(dy.astype(dtype))

    assert_as_exact(jax.jit(run_fused), jax.jit(run_ref), draw, "stream_rms_normalized fused_gemm",
                    seeds=SEEDS, truth_fn=functools.partial(run_ref, dtype=jnp.float64, wide=jnp.float64))


def test_derivation_honors_carved_out_resource_budgets():
    """`derivation.derived` honors a tightened `GpuTarget.shared_limit` below the device whole budget
    (rows 1762, 12139)."""
    from dew.nn.kernels.templates import bundles, resources
    from dew.nn.kernels.templates.gemm import derivation

    sm90 = generation.gpu_target("sm90")
    gmm = grouped_matmul.MosaicGmm()
    x = jax.ShapeDtypeStruct((2048, 2048), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((1, 2048, 2048), jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
    ba = gmm.bind(x, w, sizes, out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=True)
    layouts = bundles.groups("sm90", jnp.bfloat16, stated=True)[0]
    carved_smem = 96 * 1024
    carved = dataclasses.replace(sm90, shared_limit=carved_smem)
    tile = derivation.derived(
        layouts, rows=2048, depth=2048, width=2048, dtype=jnp.bfloat16,
        multiprocessors=carved.multiprocessors or 132, target=carved,
        candidate=lambda t: gmm.candidate(ba, t, carved),
        launch_of=lambda t: gmm.launch_of(ba, t, carved),
        grouped=False, dense=True,
    )
    assert tile is not None
    assert gmm.staged_bytes(ba, tile, carved) <= resources.program_shared(carved_smem, tile.programs)


def _add_tangent_rule(a, b):
    import operator

    del a, b
    return operator.add


def test_prologued_custom_jvp_is_picklable():
    """`gemm._prologued` uses a top-level picklable `_ProloguedTangent` (pickle round-trip + JVP check,
    row 12142)."""
    import operator
    import pickle

    tangent_rule = gemm._ProloguedTangent(operator.add, _add_tangent_rule)
    restored = pickle.loads(pickle.dumps(tangent_rule))
    assert restored == tangent_rule
    wrapped = gemm._prologued(operator.add, _add_tangent_rule)
    x = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], jnp.float32)
    y = jnp.asarray([[0.5, 1.5], [2.5, 3.5]], jnp.float32)
    dx = jnp.ones_like(x)
    dy = jnp.full_like(y, 2.0)
    primal_out, tangent_out = jax.jvp(wrapped, (x, y), (dx, dy))
    np.testing.assert_array_equal(np.asarray(primal_out), np.asarray(x + y))
    np.testing.assert_array_equal(np.asarray(tangent_out), np.asarray(dx + dy))
    primal_r, tangent_r = restored((x, y), (dx, dy))
    np.testing.assert_array_equal(np.asarray(primal_r), np.asarray(x + y))
    np.testing.assert_array_equal(np.asarray(tangent_r), np.asarray(dx + dy))


def test_gemm_named_compute_beside_prologue_and_applied_holds_check_1():
    """`gemm.gemm` accepts a named `compute` beside `prologue`, `applied`, and `scatter` (row 1958, unblocking
    MoE `ExpertMLP` gated down projections over fp32 master weights): value and VJP (including full fp32 `dW`
    precision) hold check 1 against float64 truth."""
    from dew.nn import activations
    from dew.nn.kernels import xla_products

    hook = activations.gated_hook("swiglu", None)
    group_sizes = jnp.asarray([40, 0, 24], jnp.int32)
    scatter = jnp.asarray(np.random.default_rng(19580).permutation(64), jnp.int32)

    def draw(seed):
        rng = np.random.default_rng(1958 + seed)
        gate = jnp.asarray(rng.standard_normal((64, 32)) / np.sqrt(32), jnp.bfloat16)
        up = jnp.asarray(rng.standard_normal((64, 32)) / np.sqrt(32), jnp.bfloat16)
        down = jnp.asarray(rng.standard_normal((3, 32, 32)) / np.sqrt(32), jnp.float32)
        cot = jnp.asarray(rng.standard_normal((64, 32)), jnp.bfloat16)
        return gate, up, down, cot

    def dew_fn(gate, up, down, cot):
        def fwd(g, u, w):
            hidden = hook(g, u).astype(jnp.bfloat16)
            return gemm.gemm(g, w, group_sizes, prologue=hook, prologue_operands=((u, ("rows", "depth")),),
                             applied=hidden, scatter=scatter, compute=jnp.bfloat16, out_dtype=jnp.bfloat16)

        out, pull = jax.vjp(fwd, gate, up, down)
        return out, pull(cot)

    def ref_fn(gate, up, down, cot, *, dtype=jnp.float32):
        def fwd(g, u, w):
            act = hook(g.astype(dtype), u.astype(dtype))
            if dtype == jnp.float32:
                act = act.astype(jnp.bfloat16)
            slots = xla_products.portable_projection(
                act, w, group_sizes, jnp.bfloat16 if dtype == jnp.float32 else dtype,
                None, jnp.bfloat16 if dtype == jnp.float32 else dtype)
            return jnp.zeros_like(slots).at[scatter].set(slots)

        if dtype == jnp.float64:
            out, pull = jax.vjp(fwd, gate.astype(dtype), up.astype(dtype), down.astype(dtype))
            return out, pull(cot.astype(dtype))
        out, pull = jax.vjp(fwd, gate, up, down)
        return out, pull(cot)

    assert_as_exact(jax.jit(dew_fn), jax.jit(ref_fn), draw, "gemm compute beside prologue+applied+scatter",
                    seeds=8, truth_fn=functools.partial(ref_fn, dtype=jnp.float64))
    g0, u0, d0, c0 = draw(0)
    _, (_, _, dw) = jax.jit(dew_fn)(g0, u0, d0, c0)
    assert dw.dtype == jnp.float32


def test_gemm_named_compute_prologue_preserves_compute_precision_in_vjp():
    """`gemm.gemm` with `compute` + `prologue` keeps the operand cotangent in `compute` precision before
    pulling back through the prologue VJP for both wider (`bf16 -> fp32`) and narrower (`fp32 -> bf16`)
    compute dtypes (row 1958)."""
    from dew.nn.precision import rounded_operand

    def scale_pro(x, s):
        return x * s[None, :]

    def scaled_gemm(a, s, b, *, compute_dtype):
        return gemm.gemm(a, b, prologue=scale_pro, prologue_operands=((s, ("depth",)),),
                         compute=compute_dtype, out_dtype=compute_dtype)

    # Exact wider-compute witness: bf16 lhs, fp32 rhs = 1 + 2^-9, compute=out_dtype=fp32.
    # Rounding d_operand to bf16 before prologue VJP would collapse d_scale to 1.0 instead of 1 + 2^-9.
    lhs = jnp.ones((4, 1), jnp.bfloat16)
    rhs = jnp.full((1, 1), 1.0 + 2.0 ** -9, jnp.float32)
    scale = jnp.ones((1,), jnp.float32)
    cot = jnp.ones((4, 1), jnp.float32)
    _, pull = jax.vjp(functools.partial(scaled_gemm, compute_dtype=jnp.float32), lhs, scale, rhs)
    d_lhs, d_scale, _ = pull(cot)
    assert d_lhs.dtype == jnp.bfloat16
    assert d_scale.dtype == jnp.float32
    expected = np.asarray([4.0 * (1.0 + 2.0 ** -9)], np.float32)
    np.testing.assert_array_equal(np.asarray(d_scale), expected)

    # Narrower compute (`fp32` lhs/rhs, `compute=out_dtype=bf16`) matches portable XLA forward and VJP.
    rng = np.random.default_rng(19581)
    x_f32 = jnp.asarray(rng.standard_normal((16, 32)), jnp.float32)
    s_f32 = jnp.asarray(rng.uniform(0.5, 1.5, (32,)), jnp.float32)
    w_f32 = jnp.asarray(rng.standard_normal((32, 16)), jnp.float32)
    dy_bf16 = jnp.asarray(rng.standard_normal((16, 16)), jnp.bfloat16)

    def narrow_ref(a, s, b):
        y = scale_pro(a.astype(jnp.bfloat16), s).astype(jnp.bfloat16)
        return jax.lax.dot(y, rounded_operand(b, jnp.bfloat16),
                           preferred_element_type=jnp.float32).astype(jnp.bfloat16)

    out_d, pull_d = jax.vjp(functools.partial(scaled_gemm, compute_dtype=jnp.bfloat16), x_f32, s_f32, w_f32)
    out_r, pull_r = jax.vjp(narrow_ref, x_f32, s_f32, w_f32)
    np.testing.assert_array_equal(np.asarray(out_d), np.asarray(out_r))
    for gd, gr in zip(pull_d(dy_bf16), pull_r(dy_bf16), strict=True):
        assert gd.dtype == gr.dtype
        np.testing.assert_array_equal(np.asarray(gd), np.asarray(gr))


def test_vmap_folds_batch_dimensions_into_single_gemm_launch():
    """`jax.vmap` of `gemm.gemm` folds `[B, M, K] @ [K, N]` into one `[B*M, K] @ [K, N]` dense launch and
    `[B, M, K] @ [B, K, N]` into one grouped launch with `group_sizes=[M]*B`; forward, `vmap(grad)`,
    `grad(vmap)`, and nested `vmap(vmap)` hold check 1 on interpreted Mosaic GPU (row 6104)."""
    batch, rows, depth, width = 3, 64, 64, 64

    def draw_shared(seed):
        rng = np.random.default_rng(6104 + seed)
        xs = jnp.asarray(rng.standard_normal((batch, rows, depth)) / np.sqrt(depth), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((depth, width)) / np.sqrt(depth), jnp.bfloat16)
        cot = jnp.asarray(rng.standard_normal((batch, rows, width)), jnp.bfloat16)
        return xs, w, cot

    def draw_batched_w(seed):
        rng = np.random.default_rng(61040 + seed)
        xs = jnp.asarray(rng.standard_normal((batch, rows, depth)) / np.sqrt(depth), jnp.bfloat16)
        ws = jnp.asarray(rng.standard_normal((batch, depth, width)) / np.sqrt(depth), jnp.bfloat16)
        cot = jnp.asarray(rng.standard_normal((batch, rows, width)), jnp.bfloat16)
        return xs, ws, cot

    # 1. Verify single folded launch of B*M rows (shared W) and B groups (batched W) under assumed sm90.
    xs0, w0, _ = draw_shared(0)
    xs1, ws1, _ = draw_batched_w(0)
    from kernel_outputs import launched_outputs

    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        found_shared = launched_outputs(
            jax.vmap(lambda a: gemm.gemm(a, w0, out_dtype=jnp.bfloat16, interpret_on_cpu=True)), xs0)
        assert len(found_shared) == 1
        assert found_shared[0][1][0].shape == (batch * rows, width)

        found_grouped = launched_outputs(
            jax.vmap(lambda a, b: gemm.gemm(a, b, out_dtype=jnp.bfloat16, interpret_on_cpu=True)), xs1, ws1)
        assert len(found_grouped) == 1
        assert found_grouped[0][1][0].shape == (batch * rows, width)

        # Nested vmap(vmap) folds [2, 3, 64, 64] -> [384, 64] in one launch.
        xs2 = jnp.stack([xs0, xs0 * jnp.bfloat16(0.5)])
        found_nested = launched_outputs(
            jax.vmap(jax.vmap(lambda a: gemm.gemm(a, w0, out_dtype=jnp.bfloat16, interpret_on_cpu=True))),
            xs2)
        assert len(found_nested) == 1
        assert found_nested[0][1][0].shape == (2 * batch * rows, width)
    # 2. Check 1 on vmap-of-grad and grad-of-vmap for shared W and batched W.
    def vmap_of_grad_shared(xs, w, cot):
        def one_grad(a, b, c):
            return jax.grad(lambda x, y: jnp.sum(
                gemm.gemm(x, y, out_dtype=jnp.bfloat16, interpret_on_cpu=True).astype(jnp.float32)
                * c.astype(jnp.float32)), argnums=(0, 1))(a, b)
        with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
            return jax.vmap(one_grad, in_axes=(0, None, 0))(xs, w, cot)

    def vmap_of_grad_shared_ref(xs, w, cot, *, dtype=jnp.float32):
        def one_grad(a, b, c):
            return jax.grad(lambda x, y: jnp.sum(
                jax.lax.dot(x, y, preferred_element_type=dtype).astype(
                    jnp.bfloat16 if dtype == jnp.float32 else dtype).astype(dtype) * c.astype(dtype)),
                argnums=(0, 1))(a, b)
        if dtype == jnp.float64:
            return jax.vmap(one_grad, in_axes=(0, None, 0))(
                xs.astype(dtype), w.astype(dtype), cot.astype(dtype))
        return jax.vmap(one_grad, in_axes=(0, None, 0))(xs, w, cot)

    assert_as_exact(vmap_of_grad_shared, jax.jit(vmap_of_grad_shared_ref), draw_shared,
                    "vmap(grad(gemm)) shared W", seeds=8,
                    truth_fn=functools.partial(vmap_of_grad_shared_ref, dtype=jnp.float64))

    def grad_of_vmap_batched(xs, ws, cot):
        def loss(a, b):
            y = jax.vmap(lambda x, m: gemm.gemm(x, m, out_dtype=jnp.bfloat16,
                                                interpret_on_cpu=True))(a, b)
            return jnp.sum(y.astype(jnp.float32) * cot.astype(jnp.float32))
        with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
            return jax.grad(loss, argnums=(0, 1))(xs, ws)

    def grad_of_vmap_batched_ref(xs, ws, cot, *, dtype=jnp.float32):
        def loss(a, b):
            y = jax.vmap(lambda x, m: jax.lax.dot(x, m, preferred_element_type=dtype).astype(
                jnp.bfloat16 if dtype == jnp.float32 else dtype))(a, b)
            return jnp.sum(y.astype(dtype) * cot.astype(dtype))
        if dtype == jnp.float64:
            return jax.grad(loss, argnums=(0, 1))(xs.astype(dtype), ws.astype(dtype))
        return jax.grad(loss, argnums=(0, 1))(xs, ws)

    assert_as_exact(grad_of_vmap_batched, jax.jit(grad_of_vmap_batched_ref), draw_batched_w,
                    "grad(vmap(gemm)) batched W", seeds=8,
                    truth_fn=functools.partial(grad_of_vmap_batched_ref, dtype=jnp.float64))


def test_vmap_folds_singleton_group_sides_and_preserves_named_dispatch():
    """Vmapped dense `gemm` and `gemm_t` fold both unmapped `(1, N)` and mapped `(B, 1, N)` singleton
    `('groups', ...)` sides cleanly without crashing, and unfoldable vmapped calls preserve named dispatch
    (`cutlass`/`xla`) while raising `selection.KernelRefused` for forced Pallas kernels
    (row 6104)."""
    from dew.nn.kernels import selection

    rng = np.random.default_rng(61042)
    b, m, k, n = 3, 64, 64, 64
    xs = jnp.asarray(rng.standard_normal((b, m, k)) / np.sqrt(k), jnp.bfloat16)
    ws = jnp.asarray(rng.standard_normal((b, k, n)) / np.sqrt(k), jnp.bfloat16)
    shared_bias = jnp.asarray(rng.standard_normal((1, n)), jnp.float32)
    mapped_bias = jnp.asarray(rng.standard_normal((b, 1, n)), jnp.float32)
    shared_scale = jnp.asarray(rng.uniform(0.5, 1.5, (1, k)), jnp.bfloat16)
    mapped_scale = jnp.asarray(rng.uniform(0.5, 1.5, (b, 1, k)), jnp.bfloat16)
    dys = jnp.asarray(rng.standard_normal((b, m, n)) / np.sqrt(n), jnp.bfloat16)

    def add_bias(acc, bias):
        return acc + bias

    def mul_scale(rows, scale):
        return rows * scale

    # Unmapped (1, N) and mapped (B, 1, N) ('groups', 'columns') epilogue under batched rhs.
    y_shared = jax.vmap(
        lambda x, w: gemm.gemm(x, w, epilogue=add_bias,
                               epilogue_operands=((shared_bias, ("groups", "columns")),),
                               out_dtype=jnp.bfloat16),
        in_axes=(0, 0))(xs, ws)
    y_mapped = jax.vmap(
        lambda x, w, bias: gemm.gemm(x, w, epilogue=add_bias,
                                     epilogue_operands=((bias, ("groups", "columns")),),
                                     out_dtype=jnp.bfloat16),
        in_axes=(0, 0, 0))(xs, ws, mapped_bias)
    ref_shared = jnp.stack([
        gemm.gemm(xs[i], ws[i], epilogue=add_bias, epilogue_operands=((shared_bias, ("groups", "columns")),),
                  out_dtype=jnp.bfloat16) for i in range(b)])
    ref_mapped = jnp.stack([
        gemm.gemm(xs[i], ws[i], epilogue=add_bias,
                  epilogue_operands=((mapped_bias[i], ("groups", "columns")),),
                  out_dtype=jnp.bfloat16) for i in range(b)])
    np.testing.assert_array_equal(np.asarray(y_shared), np.asarray(ref_shared))
    np.testing.assert_array_equal(np.asarray(y_mapped), np.asarray(ref_mapped))

    # Unmapped (1, K) and mapped (B, 1, K) ('groups', 'depth') prologue in vmapped dense gemm_t.
    dw_shared = jax.vmap(
        lambda x, dy: gemm.gemm_t(x, dy, None, prologue=mul_scale,
                                  prologue_operands=((shared_scale, ("groups", "depth")),),
                                  out_dtype=jnp.float32),
        in_axes=(0, 0))(xs, dys)
    dw_mapped = jax.vmap(
        lambda x, dy, sc: gemm.gemm_t(x, dy, None, prologue=mul_scale,
                                      prologue_operands=((sc, ("groups", "depth")),),
                                      out_dtype=jnp.float32),
        in_axes=(0, 0, 0))(xs, dys, mapped_scale)
    ref_dw_shared = jnp.stack([
        gemm.gemm_t(xs[i], dys[i], None, prologue=mul_scale,
                    prologue_operands=((shared_scale, ("groups", "depth")),), out_dtype=jnp.float32)
        for i in range(b)])
    ref_dw_mapped = jnp.stack([
        gemm.gemm_t(xs[i], dys[i], None, prologue=mul_scale,
                    prologue_operands=((mapped_scale[i], ("groups", "depth")),), out_dtype=jnp.float32)
        for i in range(b)])
    np.testing.assert_array_equal(np.asarray(dw_shared), np.asarray(ref_dw_shared))
    np.testing.assert_array_equal(np.asarray(dw_mapped), np.asarray(ref_dw_mapped))

    # Unfoldable grouped vmap: forced mosaic raises KernelRefused; xla/cutlass route via selection.
    gs = jnp.asarray([32, 32], jnp.int32)
    gw = jnp.asarray(rng.standard_normal((b, 2, k, n)) / np.sqrt(k), jnp.bfloat16)
    with generation.assumed("sm90"):
        with generation.chosen(grouped_matmul="mosaic"), pytest.raises(selection.KernelRefused):
            jax.vmap(lambda x, w: gemm.gemm(x, w, gs, out_dtype=jnp.bfloat16, interpret_on_cpu=True))(xs, gw)
        with generation.chosen(grouped_matmul="xla"), generation.routes() as seen:
            jax.jit(jax.vmap(lambda x, w: gemm.gemm(x, w, gs, out_dtype=jnp.bfloat16,
                                                    interpret_on_cpu=True)))(xs, gw)
        assert any(r.kernel == "xla" for r in seen), seen
        if grouped_cutlass.cutlass_built_for("sm90") is None:
            with generation.chosen(grouped_matmul="cutlass"), pytest.raises(selection.KernelRefused):
                jax.jit(jax.vmap(lambda x, w: gemm.gemm(x, w, gs, out_dtype=jnp.bfloat16,
                                                        interpret_on_cpu=True)))(xs, gw)


@lowers("sm90")
def test_vmap_unfolded_named_triton_lowers_for_cuda():
    """An unfoldable 2-batch grouped `gemm` / `gemm_t` lowers natively through forced `triton`
    (`interpret_on_cpu=False`) on CUDA while refusing non-batchable forced `mosaic` or CPU-interpreted
    `triton` with `selection.KernelRefused` (row 6104)."""
    from kernel_compile import lowered_for

    from dew.nn.kernels import selection

    b, m, k, n = 2, 64, 64, 64
    xs = jax.ShapeDtypeStruct((b, m, k), jnp.bfloat16)
    gw = jax.ShapeDtypeStruct((b, 2, k, n), jnp.bfloat16)
    dys = jax.ShapeDtypeStruct((b, m, n), jnp.bfloat16)
    gs = jax.ShapeDtypeStruct((2,), jnp.int32)

    fwd = jax.vmap(lambda x, w, s: gemm.gemm(x, w, s, out_dtype=jnp.bfloat16), in_axes=(0, 0, None))
    dw = jax.vmap(lambda x, dy, s: gemm.gemm_t(x, dy, s, out_dtype=jnp.float32), in_axes=(0, 0, None))
    with generation.chosen(grouped_matmul="triton"), generation.routes() as seen_fwd:
        lowered_for(fwd, (xs, gw, gs), "sm90")
    assert any(r.kernel == "triton" for r in seen_fwd), seen_fwd
    with generation.chosen(grouped_matmul="triton"), generation.routes() as seen_dw:
        lowered_for(dw, (xs, dys, gs), "sm90")
    assert any(r.kernel == "triton" for r in seen_dw), seen_dw
    with generation.chosen(grouped_matmul="mosaic"), pytest.raises(selection.KernelRefused):
        lowered_for(fwd, (xs, gw, gs), "sm90")
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="triton"),
          pytest.raises(selection.KernelRefused)):
        jax.eval_shape(
            jax.vmap(lambda x, w, s: gemm.gemm(x, w, s, out_dtype=jnp.bfloat16, interpret_on_cpu=True),
                     in_axes=(0, 0, None)),
            xs, gw, gs)


def test_dense_gemm_takes_split_and_unsplit_pins_and_holds_check_1():
    """A deep dense GEMM's tile on sm90, and its transposed input gradient's on sm80, run split in two or
    unsplit alike (both legal), and pinned split or unsplit an sm90 product holds check 1 against float64
    truth (row 10151)."""
    from dew.nn.kernels import tuning

    launch = grouped_matmul.MosaicGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    one = jax.ShapeDtypeStruct((1,), jnp.int32)
    for name, transposed, matrix in (("sm90", False, (1, 16384, 1024)), ("sm80", True, (1, 1024, 16384))):
        with generation.assumed(name):
            target = generation.gpu_target()
            ba = launch.bind(bf16((512, 16384)), bf16(matrix), one, out_dtype="bfloat16",
                             trans_rhs=transposed, interpret=False, dense=True)
            tile = launch.derived(ba, target)
            assert tile is not None, name
            for splits in (1, 2):
                assert launch.legal(ba, dataclasses.replace(tile, splits=splits), target), (name, splits)

    # Exercise explicit unsplit and split pins numerically against float64 truth.
    def draw(seed):
        rng = np.random.default_rng(10151 + seed)
        return (jnp.asarray(rng.standard_normal((128, 256)) / 16.0, jnp.bfloat16),
                jnp.asarray(rng.standard_normal((256, 128)) / 16.0, jnp.bfloat16))

    for splits in (1, 2):
        pin = tiling.Tiles(128, 128, 64, 2, 1, warpgroups=2, family="wgmma", splits=splits)

        def dew_fn(x, w, *, pinned=pin):
            with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"),
                  tuning.pinned({grouped_matmul.MosaicGmm: pinned})):
                return gemm.gemm(x, w, out_dtype=jnp.bfloat16, interpret_on_cpu=True)

        assert_as_exact(
            jax.jit(dew_fn),
            jax.jit(lambda x, w: jax.lax.dot(x, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)),
            draw, f"dense explicit splits={splits}", seeds=8,
            truth_fn=lambda x, w: jax.lax.dot(x.astype(jnp.float64), w.astype(jnp.float64),
                                              preferred_element_type=jnp.float64))


def test_sm80_transposed_dx_and_tgmm_skip_regressing_fragment_prefetch():
    """Ampere `CpAsync` skips cross-step fragment prefetch on transposed `dX` (`M-FS-53-a100`, `M-NG-2-a100`:
    gate_up dX +32-34% slower) and `tgmm` (`M-NG-45-a100`: dW geomean 1.019x, 4B key +11.6%), keeping forward
    untransposed prefetch active with its +16/+32 register cost (row 5271)."""
    from dew.nn.kernels.templates.gemm import mosaic_gpu

    launch = grouped_matmul.MosaicGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    one = jax.ShapeDtypeStruct((1,), jnp.int32)
    with generation.assumed("sm80"):
        target = generation.gpu_target()
        tile = tiling.Tiles(128, 128, 32, 3, 1, warpgroups=1, family="mma.sync")
        fam = pricing.family_of(tile, jnp.bfloat16, target)
        assert mosaic_gpu.prefetches(fam, matrices=1, prologue=None, scaled=False, chained=False,
                                     interpret=False, transposed=(False, False), summed=False)
        assert not mosaic_gpu.prefetches(fam, matrices=1, prologue=None, scaled=False, chained=False,
                                         interpret=False, transposed=(False, True), summed=False)
        fwd_ba = launch.bind(bf16((1024, 1024)), bf16((1, 1024, 1024)), one, out_dtype="bfloat16",
                             trans_rhs=False, interpret=False, dense=True)
        dx_ba = launch.bind(bf16((1024, 1024)), bf16((1, 1024, 1024)), one, out_dtype="bfloat16",
                            trans_rhs=True, interpret=False, dense=True)
        fwd_regs = pricing.prefetch_registers(fam, tile, ba=fwd_ba)
        dx_regs = pricing.prefetch_registers(fam, tile, ba=dx_ba)
        assert fwd_regs == 16 and dx_regs == 0, (fwd_regs, dx_regs)


def test_explicit_target_b200_gathered_settles_under_ambient_sm80():
    """Deriving and settling a B200 (`sm100`) gathered tile under ambient `sm80` binds `sm100`'s
    register/raise facts via `generation.as_target(target)` instead of crashing on Ampere's `None` raises
    (row 12098)."""
    from dew.nn.kernels.templates import resources
    from dew.nn.kernels.templates.gemm import producers

    launch = grouped_matmul.MosaicGmm()
    sm100 = generation.gpu_target("sm100")
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((2,), jnp.int32)
    gather = jax.ShapeDtypeStruct((256,), jnp.int32)
    ba = launch.bind(bf16((256, 256)), bf16((2, 256, 256)), sizes, (), (), None, None, gather,
                     out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=False)
    # Cold cache under ambient sm80 first so `_DERIVED` cannot mask derivation or producer settlement.
    with generation.assumed("sm80"):
        got = launch.derived(ba, sm100)
        assert got is not None and got.producers.of(launch.KERNEL) is not None
        assert launch.legal(ba, got, sm100)
        unsettled = dataclasses.replace(got, producers=resources.Producers())
        assert not launch._refused(ba, unsettled, sm100)
        assert producers.settled(launch, ba, unsettled, sm100) == got
    with generation.assumed("sm100"):
        want = producers.settled(launch, ba, unsettled, sm100)
        assert launch.legal(ba, got, sm100)
    assert got == want


def test_a100_deep_k_dense_dx_derives_l2_drift_split_and_compiles():
    """On `walks_scoped` A100 (`sm80`, L2=40MB), `SplitWalk` prices a deep-K dense wave past the L2 drift
    horizon (`8192x51200->5120`, `M-FS-6`, row 5363) faster split than unsplit while shallower shapes and
    Hopper `sm90` price unsplit least; derived tiles stay legal, a split whose partials pass the reference's
    workspace is none (row 12850), and a pinned `splits=2` `sm80` tile within it compiles spill-free."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels


    launch = grouped_matmul.MosaicGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    one = jax.ShapeDtypeStruct((1,), jnp.int32)

    def walk_of(tile, target, depth_k: int, width_n: int):
        unsplit = dataclasses.replace(tile, splits=1)
        work = pricing.decode_contraction(
            unsplit, target, rows=8192, groups=1, depth=depth_k, width=width_n, columns=None, held=8192,
            operands=((2, 2), (None, None)), stored=2.0, multiprocessors=target.multiprocessors)
        return pricing.split_walk(unsplit, **work)

    with generation.assumed("sm80"):
        target = generation.gpu_target()
        deep_ba = launch.bind(bf16((8192, 51200)), bf16((1, 5120, 51200)), one, out_dtype="bfloat16",
                              trans_rhs=True, interpret=False, dense=True)
        deep_tile = launch.derived(deep_ba, target)
        assert deep_tile is not None and launch.legal(deep_ba, deep_tile, target), deep_tile
        deep_walk = walk_of(deep_tile, target, 51200, 5120)
        assert deep_walk is not None and deep_walk.seconds(2) < deep_walk.seconds(1), deep_walk
        for depth_k, width_n in ((4096, 4096), (24576, 4096)):
            ba = launch.bind(bf16((8192, depth_k)), bf16((1, width_n, depth_k)), one, out_dtype="bfloat16",
                             trans_rhs=True, interpret=False, dense=True)
            tile = launch.derived(ba, target)
            assert tile is not None and launch.legal(ba, tile, target), tile
            walk = walk_of(tile, target, depth_k, width_n)
            assert walk is not None and walk.seconds(1) <= walk.seconds(2), (depth_k, walk)
        # Its 8192 rows' fp32 partials split two ways (320 MiB) are past the 4 MiB workspace cuBLAS holds
        # (row 12850), so the drift's split is no tile's, pinned or derived; 64 rows' (2.6 MiB) compile split.
        pinned = dataclasses.replace(deep_tile, splits=2)
        assert not launch.legal(deep_ba, pinned, target) and launch.derived(deep_ba, target).splits == 1
        few_ba = launch.bind(bf16((64, 51200)), bf16((1, 5120, 51200)), one, out_dtype="bfloat16",
                             trans_rhs=True, interpret=False, dense=True)
        assert launch.legal(few_ba, pinned, target), pinned
        if generation.compile_only_gpus("sm80"):
            records = compile_mosaic_kernels(
                [KernelCase("sm80_deep_k_split_dx", lambda: launch.lowering(few_ba, pinned))], "sm80")
            for rec in records:
                assert not rec.get("error"), rec
                assert rec["kernels"], rec
                assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        h_ba = launch.bind(bf16((8192, 51200)), bf16((1, 5120, 51200)), one, out_dtype="bfloat16",
                           trans_rhs=True, interpret=False, dense=True)
        h_tile = launch.derived(h_ba, target)
        assert h_tile is not None and launch.legal(h_ba, h_tile, target), h_tile
        h_walk = walk_of(h_tile, target, 51200, 5120)
        assert h_walk is not None and h_walk.seconds(1) <= h_walk.seconds(2), h_walk


def test_sm100_pair_tiles_refuse_depth_splits_in_legal_and_derivation():
    """Blackwell `tcgen05.pair` refuses `splits > 1` (`mosaic_gpu.pair_refuses`) in both `MosaicGmm.legal`
    and `MosaicTgmm.legal` even though `_ran` validates the unsplit partial tile, and underfilled dense
    `sm100` calls keep `splits == 1` on `tcgen05.pair`."""
    gmm, tgmm = grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    one = jax.ShapeDtypeStruct((1,), jnp.int32)
    with generation.assumed("sm100"):
        sm100 = generation.gpu_target()
        fwd_ba = gmm.bind(bf16((512, 16384)), bf16((1, 16384, 1024)), one, out_dtype="bfloat16",
                          trans_rhs=False, interpret=False, dense=True)
        tile = gmm.derived(fwd_ba, sm100)
        assert tile is not None and gmm.legal(fwd_ba, tile, sm100), tile
        pair_tile = tiling.Tiles(128, 128, 64, 6, 4, warpgroups=1, cluster=2, family="tcgen05.pair")
        assert not gmm.legal(fwd_ba, dataclasses.replace(pair_tile, splits=3), sm100)
        if tile.family == "tcgen05.pair":
            assert tile.splits == 1, tile
        dw_ba = tgmm.bind(bf16((8192, 1024)), bf16((8192, 1024)), one, out_dtype="float32",
                          interpret=False, dense=True)
        assert not tgmm.legal(dw_ba, dataclasses.replace(pair_tile, splits=2), sm100)
        # Power-of-two tcgen05.alloc rounding: 128 acc cols + 1 block-scale ring (128 < tensor < 170)
        # allocates 256 cols/program, so 2 programs (512 <= 512) fit while 3 programs (768 > 512, vs
        # unrounded 3 * tensor <= 512) refuse on real sm100 (row 12302).
        from dew.nn.kernels.templates.gemm import pricing

        def scales() -> dict[str, tuple[tuple[int, ...], object]]:
            return {"s": ((1, 1, 32, 16), jnp.uint8)}

        t64 = tiling.Tiles(128, 64, 64, 2, 1, warpgroups=1, cluster=1, family="tcgen05", programs=2)
        fam = pricing.family_of(t64, jnp.bfloat16, sm100)
        tensor = fam.tensor_columns(64, 1) + fam.stream_tensor_columns(scales(), 2)
        assert tensor > 128 and tensor * 3 <= (sm100.tmem_columns or 0), tensor
        assert pricing.granular_tile(fam, t64, jnp.bfloat16, target=sm100, stream_blocks=scales)
        assert not pricing.granular_tile(
            fam, dataclasses.replace(t64, programs=3), jnp.bfloat16, target=sm100, stream_blocks=scales)


def test_the_planner_prices_mma_syncs_row_bound_where_its_grouped_kernel_issues_it():
    """The planner's register estimate of a one-consumer grouped mma.sync launch, whose products take a row
    bound, is a dense one's at the same tile and rows and its issuer's registers for the bound
    (`MmaIssuer.bound_registers`) besides: the dense call's products take none (row 12304)."""
    gmm, tile = grouped_matmul.MosaicGmm(), tiling.Tiles(128, 128, 32, 3, 1, warpgroups=1, family="mma.sync")
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    calls = {grouped: gmm.bind(bf16((512, 256)), bf16((4 if grouped else 1, 256, 256)),
                               jax.ShapeDtypeStruct((4 if grouped else 1,), jnp.int32), out_dtype="bfloat16",
                               trans_rhs=False, interpret=False, dense=not grouped)
             for grouped in (True, False)}
    with generation.assumed("sm80"):
        target = generation.gpu_target()
        estimated = {grouped: gmm._registers(ba, tile, merging=False, target=target)
                     for grouped, ba in calls.items()}
        bound = pricing.family_of(tile, jnp.bfloat16, target).bound_registers
    assert bound > 0 and estimated[True] - estimated[False] == bound, (bound, estimated)


@INTERPRETS
def test_mma_sync_grouped_gemm_suppresses_ragged_rows_via_bounded():
    """One-consumer `mma.sync` grouped tiles (`tiles.sharing == 1`) bind `group.actual_end - row_start` onto
    `family.bounded` (`_row_bound`), emitting runtime `int32` row-bound operands on every native `mma` (none
    on dense), holding check 1 on `[64, 64]` and unaligned+zero-row `[48, 0, 35, 45]` groupings, and
    compiling spill-free on `sm80` (rows 5623, 7091)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    tile_128 = tiling.Tiles(128, 128, 32, 3, 1, warpgroups=1, family="mma.sync")
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    with (generation.assumed("sm80"), generation.chosen(grouped_matmul="mosaic"),
          tuning.pinned({grouped_matmul.MosaicGmm: tile_128})):
        grouped_graph = jax.make_jaxpr(
            lambda x, w, s: gemm.gemm(x, w, s, out_dtype=jnp.bfloat16)
        )(bf16((192, 64)), bf16((3, 64, 128)), jax.ShapeDtypeStruct((3,), jnp.int32))
        dense_graph = jax.make_jaxpr(
            lambda x, w: gemm.gemm(x, w, None, out_dtype=jnp.bfloat16)
        )(bf16((128, 64)), bf16((64, 128)))
    grouped_mmas = [eq for eq in equations(grouped_graph, platform="cuda") if eq.primitive.name == "mma"]
    dense_mmas = [eq for eq in equations(dense_graph, platform="cuda") if eq.primitive.name == "mma"]
    assert grouped_mmas and all(
        len(eq.invars) == 4 and eq.invars[3].aval.dtype == jnp.int32 and eq.invars[3].aval.shape == ()
        for eq in grouped_mmas), grouped_mmas
    assert dense_mmas and all(len(eq.invars) == 3 for eq in dense_mmas), dense_mmas

    # Verify the typed mma_p row bound derives from group metadata (`actual_end - row_start`, clipped), not a
    # constant (rows 5623, 7091).
    from jax.extend import core as jcore

    bound_ops: list[set[str]] = []

    def _trace_mma_bounds(jaxpr, outer_defs: dict, outer_bind: dict):
        defs = {**outer_defs}
        platform_idx: dict = {}
        for eqn in jaxpr.eqns:
            for outv in eqn.outvars:
                defs[outv] = eqn
            if eqn.primitive.name == "platform_index":
                platforms = eqn.params.get("platforms", ())
                if "cuda" in platforms:
                    platform_idx[eqn.outvars[0]] = platforms.index("cuda")
                continue
            if eqn.primitive.name == "mma" and len(eqn.invars) == 4:
                seen_vars, stack, ops = set(), [eqn.invars[3]], set()
                while stack:
                    var = stack.pop()
                    if not isinstance(var, jcore.Var) or var in seen_vars:
                        continue
                    seen_vars.add(var)
                    if var in outer_bind:
                        stack.append(outer_bind[var])
                    if (producer := defs.get(var)) is not None:
                        ops.add(producer.primitive.name)
                        stack.extend(producer.invars)
                bound_ops.append(ops)
            if (eqn.primitive.name == "cond" and eqn.invars
                    and isinstance(eqn.invars[0], jcore.Var) and eqn.invars[0] in platform_idx):
                sub = eqn.params["branches"][platform_idx[eqn.invars[0]]]
                inner = sub.jaxpr if hasattr(sub, "jaxpr") else sub
                bind = {**outer_bind, **dict(zip(inner.invars, eqn.invars[1:], strict=False))}
                _trace_mma_bounds(inner, defs, bind)
                continue
            for value in eqn.params.values():
                subs = value if isinstance(value, (tuple, list)) else (value,)
                for sub in subs:
                    inner = sub.jaxpr if hasattr(sub, "jaxpr") else sub
                    if not hasattr(inner, "eqns"):
                        continue
                    bind = {**outer_bind}
                    for iv, ov in zip(inner.constvars, eqn.invars, strict=False):
                        bind[iv] = ov
                    for iv, ov in zip(inner.invars, eqn.invars[len(inner.constvars):], strict=False):
                        bind[iv] = ov
                    _trace_mma_bounds(inner, defs, bind)

    _trace_mma_bounds(grouped_graph.jaxpr, {}, {})
    assert bound_ops and all(
        "sub" in ops and bool({"max", "min", "clamp"} & ops)
        and bool({"get", "load", "gather", "dynamic_slice", "ref_get"} & ops)
        for ops in bound_ops), bound_ops
    def dew_fn(x, w, sizes):
        with (generation.assumed("sm80"), generation.chosen(grouped_matmul="mosaic"),
              tuning.pinned({grouped_matmul.MosaicGmm: tile_128})):
            return gemm.gemm(x, w, sizes, out_dtype=jnp.bfloat16, interpret_on_cpu=True)

    def ref_fn(x, w, sizes, *, dtype=jnp.float32):
        return jax.lax.ragged_dot(x.astype(dtype), w.astype(dtype), sizes,
                                  preferred_element_type=dtype).astype(
            jnp.float64 if dtype == jnp.float64 else jnp.bfloat16)

    for tag, counts in (("[64, 64]", (64, 64)), ("[48, 0, 35, 45]", (48, 0, 35, 45))):
        def draw(seed, counts=counts):
            rng = np.random.default_rng(7091 + len(counts) * 100 + seed)
            x = jnp.asarray(rng.standard_normal((128, 64)) / np.sqrt(64), jnp.bfloat16)
            w = jnp.asarray(rng.standard_normal((len(counts), 64, 128)) / np.sqrt(64), jnp.bfloat16)
            return x, w, jnp.asarray(counts, jnp.int32)

        assert_as_exact(jax.jit(dew_fn), jax.jit(ref_fn), draw, f"sm80 bounded {tag}", seeds=8,
                        truth_fn=functools.partial(ref_fn, dtype=jnp.float64))

    if generation.compile_only_gpus("sm80"):
        s_spec = jax.ShapeDtypeStruct((3,), jnp.int32)

        def compiled(x, w, s):
            with generation.chosen(grouped_matmul="mosaic"), tuning.pinned(
                    {grouped_matmul.MosaicGmm: tile_128}):
                return gemm.gemm(x, w, s, out_dtype=jnp.bfloat16)

        with generation.assumed("sm80"):
            records = compile_mosaic_kernels(
                [KernelCase("sm80_row_idle_grouped",
                            lambda: (compiled, (bf16((192, 64)), bf16((3, 64, 128)), s_spec)))], "sm80")
        for rec in records:
            assert not rec.get("error"), rec
            assert rec["kernels"], rec
            assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec
