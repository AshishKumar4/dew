"""The GEMM template's shared mainloop (`mosaic_gpu.gmm`) on Blackwell's tcgen05 (`mosaic_sm100.RINGS`, the
fold of the old Blackwell GEMM mainloop into it, template push piece 7, and its two-CTA pair,
`mosaic_sm100.PAIR`): interpreted as an sm100 on a CPU, compiled for a compile-only B200, and run on a B200 or
a B300 itself (tests/test_gpu_gemm.py's cells)."""

import dataclasses
import functools

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from reference_error import assert_as_exact

from dew.nn.kernels import gemm, generation, grouped_matmul, tuning
from dew.nn.kernels.templates.gemm import mosaic_gpu

TILE = mosaic_gpu.Tiles(128, 128, 64, 4, 1, family='tcgen05')
"""A legal tcgen05 tile of every product `_on_tcgen05` pins (`tuning.pinned`): a CTA's 128 rows by 128
columns, four 64-deep stages."""
TILES = (mosaic_gpu.Tiles(128, 128, 64, 2, 1, family='tcgen05'),
         mosaic_gpu.Tiles(128, 256, 64, 3, 2, family='tcgen05'))
"""A tile of each accumulator width, its accumulators within the 512 columns of tensor memory, its ring within
a B200's shared memory beside two outputs' staging (four stages asked 245,900 of its 232,448 bytes,
gate-2738-gemm-b200, row 4449)."""
DEEP = (mosaic_gpu.Tiles(128, 128, 64, 6, 1, family='tcgen05'),
        mosaic_gpu.Tiles(128, 64, 64, 8, 1, family='tcgen05'))
"""The compile-ahead step's two spilling pins (row 5358): the deepest ring a 128-column tile's shared memory
holds, which spilled 56 bytes at 248 registers while its epilogue read every tensor-memory chunk at once, and
an eight-slot ring, whose kernel gradient spilled the predicate file (P2R, STL) holding a predicate a slot
past a thread's seven (`test_an_eight_slot_kernel_gradient_compiles_within_its_predicates`)."""
PAIRED = mosaic_gpu.Tiles(128, 128, 64, 4, 1, cluster=2, family='tcgen05.pair')
"""A pair of CTAs' tile (`mosaic_sm100.PAIR`: 256 x 256, each CTA's accumulators its 128 rows by the pair's
256 columns). Mosaic GPU's interpreter models no collective tcgen05 product, so it is compiled here and run
on a B200 (and in nvgpu-lean's simulator) alone."""


def _on_tcgen05(*, interpret: bool) -> None:
    """The shared GEMM mainloop (`mosaic_gpu.gmm`) on tcgen05's bundle (`mosaic_sm100.RINGS`), the template's
    launch named and tcgen05's tile pinned (`TILE`), over integers from -1 to 1 at a depth of 256 (every sum
    at most 256, which a bf16 output holds exactly): a grouped forward, a grouped input gradient (its matrix
    read transposed) and a dense forward, each exactly numpy's; a grouped kernel gradient, its stages
    rewritten by a role of their own, and a dense one into bf16 and into fp32, its stages read as loaded (row
    5209), each exactly numpy's too. Interpreted as an sm100 on a CPU, or run on the Blackwell GPU itself."""
    sizes = jnp.asarray([128, 200, 56], jnp.int32)
    starts = np.cumsum([0, *np.asarray(sizes)])

    def ints(key, shape):
        return jax.random.randint(jax.random.key(key), shape, -1, 2).astype(jnp.bfloat16)

    x, w, dy = ints(0, (384, 256)), ints(1, (3, 256, 256)), ints(2, (384, 256))

    def grouped(lhs, matrices):
        return np.concatenate([np.asarray(lhs[starts[g]:starts[g + 1]], np.float32)
                               @ np.asarray(matrices[g], np.float32) for g in range(3)])

    def summed(lhs, rhs, sizes, **options):
        return gemm.gemm(lhs, rhs, sizes, out_dtype=jnp.bfloat16, precision="default",
                         interpret_on_cpu=interpret, **options)

    calls = {
        "grouped forward": (lambda: summed(x, w, sizes), grouped(x, w)),
        "grouped input gradient": (lambda: summed(dy, w, sizes, transpose_rhs=True),
                                   grouped(dy, jnp.swapaxes(w, 1, 2))),
        "dense forward": (lambda: summed(x, w[0], None),
                          np.asarray(x, np.float32) @ np.asarray(w[0], np.float32)),
    }
    # tcgen05's tile named for every launch (`tuning.pinned`, a pin its legality checks), the template's
    # launch named for the product.
    pinned = {grouped_matmul.MosaicGmm: TILE, grouped_matmul.MosaicTgmm: TILE}
    with generation.chosen(grouped_matmul="mosaic"), tuning.pinned(pinned):
        for name, (call, expected) in calls.items():
            np.testing.assert_array_equal(np.asarray(jax.jit(call)(), np.float32), expected, err_msg=name)
        got = np.asarray(jax.jit(lambda: gemm.gemm_t(x, dy, sizes, out_dtype=jnp.float32, precision="default",
                                                     interpret_on_cpu=interpret))())
        expected = np.stack([np.asarray(x[starts[g]:starts[g + 1]], np.float32).T
                             @ np.asarray(dy[starts[g]:starts[g + 1]], np.float32) for g in range(3)])
        np.testing.assert_array_equal(got, expected, err_msg="kernel gradient")
        # 256 rows: every sum at most 256, which a bf16 output holds exactly.
        rows = 256
        for out_dtype in (jnp.bfloat16, jnp.float32):
            got = np.asarray(jax.jit(lambda out_dtype=out_dtype: gemm.gemm_t(
                x[:rows], dy[:rows], None, out_dtype=out_dtype, precision="default",
                interpret_on_cpu=interpret)[0])(), np.float32)
            expected = np.asarray(x[:rows], np.float32).T @ np.asarray(dy[:rows], np.float32)
            np.testing.assert_array_equal(got, expected, err_msg=f"dense kernel gradient into {out_dtype}")


@INTERPRETS
def test_blackwells_products_run_the_shared_mainloop_on_tcgen05():
    """On an sm100 GPU the template's products run the one GEMM mainloop on tcgen05's bundle, its
    accumulators in tensor memory, read out a chunk of columns at a time (`_on_tcgen05`, interpreted)."""
    with generation.assumed("sm100"):
        _on_tcgen05(interpret=True)


def the_shared_mainloop_runs_exactly_on_blackwell():
    """`_on_tcgen05` on a B200 or a B300 itself (tests/test_gpu_gemm.py's cell)."""
    _on_tcgen05(interpret=False)


@INTERPRETS
@pytest.mark.parametrize("depth", [576, 1088], ids=["nine steps", "seventeen steps"])
def test_a_blackwell_tiles_sum_holds_every_step_however_its_rotations_fall(depth):
    """A dense product on tcgen05's bundle at a depth whose steps (64 deep, a ring of four stages) no tile's
    rotations divide: its walk takes a rotation of the ring a trip (`mosaic.stepping`), so a tile past the
    first starts within a rotation, and each step's products add onto the tile's sum but at its first step
    (the tensor-memory accumulator's `accumulate`), over integers from -1 to 1 bitwise numpy's at every
    tile; interpreted as an sm100."""
    tiles = mosaic_gpu.Tiles(128, 128, 64, 4, 1, family='tcgen05')
    keys = jax.random.split(jax.random.key(3), 2)
    x = jax.random.randint(keys[0], (256, depth), -1, 2).astype(jnp.bfloat16)
    w = jax.random.randint(keys[1], (depth, 512), -1, 2).astype(jnp.bfloat16)
    with generation.assumed("sm100"):
        got = mosaic_gpu.gmm(x, w[None], jnp.asarray([256], jnp.int32), tiles=tiles, transpose_rhs=False,
                             out_dtype=jnp.float32, multiprocessors=4, interpret=grouped_matmul.INTERPRETED,
                             dense=True)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(x, np.float32) @ np.asarray(w, np.float32))


def biased(acc, bias):
    return acc + bias[None, :]


def scaled_rows(acc, scale):
    return acc * scale[:, None]


def doubled(acc):
    return acc, acc * 2


EPILOGUES = {
    "plain": ((), None, (jnp.bfloat16,)),
    "transposed": ((), None, (jnp.bfloat16,)),
    "bias": ((("bias", ("columns",)),), biased, (jnp.bfloat16,)),
    "row scale": ((("scale", ("rows",)),), scaled_rows, (jnp.float32,)),
    "two outputs": ((), doubled, (jnp.bfloat16, jnp.float32)),
}
"""Each case's epilogue side operands (drawn by name, with their launch axes), its epilogue and its output
dtypes: a column bias, a per-row scale and a mixed-dtype pair of outputs, which the fold's old Blackwell
mainloop's tests held."""
SIZES = (0, 70, 1, 129, 100, 0)
ROWS, DEPTH, WIDTH = 307, 128, 512
DRAWN = ("x", "w", "bias", "scale")
SEEDS = 8
"""Each site's declared draws (`reference_error.seeds_for` of the plain program on them)."""


def _draw(seed):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.bfloat16),
            jnp.asarray(rng.normal(size=(len(SIZES), DEPTH, WIDTH)) / np.sqrt(DEPTH), jnp.bfloat16),
            jnp.asarray(rng.normal(size=(WIDTH,)), jnp.float32),
            jnp.asarray(rng.uniform(0.5, 2.0, size=(ROWS,)), jnp.float32))


def _epilogued(case: str, tiles: mosaic_gpu.Tiles, *, interpret: bool):
    """The shared mainloop at `tiles` running `case`'s epilogue (`EPILOGUES`), over every multiprocessor of
    the GPU (one, interpreted)."""
    sides, epilogue, dtypes = EPILOGUES[case]
    programs = tiles.cluster if interpret else generation.gpu_target().multiprocessors

    def run(*inputs):
        drawn = dict(zip(DRAWN, inputs, strict=True))
        w = jnp.swapaxes(drawn["w"], 1, 2) if case == "transposed" else drawn["w"]
        outs = mosaic_gpu.gmm(drawn["x"], w, jnp.asarray(SIZES, jnp.int32), tiles=tiles,
                              transpose_rhs=case == "transposed", out_dtype=dtypes[0],
                              multiprocessors=programs,
                              interpret=grouped_matmul.INTERPRETED if interpret else False, epilogue=epilogue,
                              epilogue_operands=tuple(drawn[name] for name, _ in sides),
                              epilogue_axes=tuple(axes for _, axes in sides),
                              out_dtypes=dtypes if len(dtypes) > 1 else None)
        return outs if isinstance(outs, tuple) else (outs,)
    return run


def _plain(case: str, *, truth: bool = False):
    """The program the mainloop stands for: each group's rows times its matrix summed wide, the rows past the
    routed ones zero, the epilogue, each output rounded once (in float64 as truth)."""
    sides, epilogue, dtypes = EPILOGUES[case]

    def run(*inputs):
        drawn = dict(zip(DRAWN, inputs, strict=True))
        wide = jnp.promote_types(drawn["x"].dtype, jnp.float32)
        acc = jax.lax.ragged_dot(drawn["x"], drawn["w"].astype(drawn["x"].dtype), jnp.asarray(SIZES),
                                 preferred_element_type=wide)
        acc = jnp.where((jnp.arange(ROWS) < sum(SIZES))[:, None], acc, 0)
        if epilogue is not None:
            acc = epilogue(acc, *(drawn[name] for name, _ in sides))
        outs = acc if isinstance(acc, tuple) else (acc,)
        return tuple(out.astype(jnp.float64 if truth else dtype)
                     for out, dtype in zip(outs, dtypes, strict=True))
    return run


@INTERPRETS
@pytest.mark.parametrize("tiles", TILES, ids=lambda tiles: f"{tiles.block_n}x{tiles.stages}")
@pytest.mark.parametrize("case", list(EPILOGUES))
def test_tcgen05s_epilogues_hold_check_1(case, tiles):
    """Each epilogue case through the shared mainloop on tcgen05's bundle at each accumulator width,
    interpreted as an sm100: as exact as the plain program from float64 truth, the empty groups, the group of
    one row and the rows past the routed ones included."""
    with generation.assumed("sm100"):
        assert_as_exact(_epilogued(case, tiles, interpret=True), jax.jit(_plain(case)), _draw,
                        f"{case} at {tiles}", seeds=SEEDS, truth_fn=_plain(case, truth=True))


def tcgen05s_epilogues_hold_check_1_on_blackwell(case, tiles):
    """Each epilogue case compiled and run on a B200 or a B300 at `tiles`, a CTA's or a pair's (`PAIRED`),
    over every multiprocessor, where the interpreter's synchronization is the hardware's: as exact as the
    plain program from float64 truth (tests/test_gpu_gemm.py's cells)."""
    assert_as_exact(jax.jit(_epilogued(case, tiles, interpret=False)), jax.jit(_plain(case)), _draw,
                    f"{case} at {tiles} on Blackwell", seeds=SEEDS, truth_fn=_plain(case, truth=True))


def test_tcgen05s_mainloop_compiles_for_a_b200_within_its_resources():
    """The shared mainloop at each tcgen05 tile, its deepest rings' (`DEEP`) and a pair's (`PAIRED`, each
    CTA's accumulators 256 columns), with a bias epilogue, compiled by Mosaic for a compile-only B200 on this
    CPU through ptxas (tests/kernel_compile.py): within the multiprocessor's shared memory and its 512
    columns of tensor memory, no register spilled, its epilogue reading each tensor-memory chunk as it takes
    it (read all at once, the 256-column accumulators spilled 24 and 36 bytes and the deep ring 56, row
    5358)."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    def case(tiles):
        def call():
            def product(lhs, rhs, sizes, bias):
                return mosaic_gpu.gmm(lhs, rhs, sizes, tiles=tiles, transpose_rhs=False,
                                      out_dtype=jnp.bfloat16,
                                      multiprocessors=generation.gpu_target("sm100").multiprocessors,
                                      epilogue=biased, epilogue_operands=(bias,),
                                      epilogue_axes=(("columns",),))

            return product, (jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                             jax.ShapeDtypeStruct((4, 512, 1024), jnp.bfloat16),
                             jax.ShapeDtypeStruct((4,), jnp.int32),
                             jax.ShapeDtypeStruct((1024,), jnp.float32))

        return support.KernelCase(f"tcgen05 gmm {tiles}", call)

    found = compile_mosaic_kernels([case(tiles) for tiles in (*TILES, *DEEP, PAIRED)], "sm100")
    assert len(found) == len(TILES) + len(DEEP) + 1 and all(record.get("kernels") for record in found), found
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


def test_an_eight_slot_kernel_gradient_compiles_within_its_predicates():
    """The dense kernel gradient (`mosaic_gpu.tgmm`) on an eight-slot tcgen05 ring (`DEEP`'s second), into
    fp32 and bf16, compiled by Mosaic for a compile-only B200 through ptxas: no spill, where its ring's
    per-slot tests held one predicate more than a thread's seven (P2R and STL of the predicate file at every
    raise to 200, row 5358)."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    def case(out_dtype):
        def call():
            def product(lhs, rhs, sizes):
                return mosaic_gpu.tgmm(lhs, rhs, sizes, tiles=DEEP[1], out_dtype=out_dtype,
                                       multiprocessors=generation.gpu_target("sm100").multiprocessors,
                                       dense=True)

            return product, (jax.ShapeDtypeStruct((28672, 1024), jnp.bfloat16),
                             jax.ShapeDtypeStruct((28672, 1024), jnp.bfloat16),
                             jax.ShapeDtypeStruct((1,), jnp.int32))

        return support.KernelCase(f"tcgen05 tgmm {DEEP[1]} {jnp.dtype(out_dtype).name}", call)

    found = compile_mosaic_kernels([case(jnp.float32), case(jnp.bfloat16)], "sm100")
    assert len(found) == 2 and all(record.get("kernels") for record in found), found
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


def test_a_pairs_tile_is_twice_a_ctas_and_takes_only_what_its_views_read():
    """A pair's tile (`PAIRED`, `mosaic.Transport.pairs_product`): its accumulators hold both CTAs' columns
    (`mosaic_gpu.tile_columns`); a tile of 256 columns a CTA, past its half of one product, or of one CTA is
    none of the pair's (`mosaic_gpu.pair_takes`); and its product takes no prologue (its step views are its
    rows' and matrix's alone)."""
    family = mosaic_gpu.family_of(PAIRED, jnp.bfloat16)
    assert mosaic_gpu.tile_columns(PAIRED, family) == 2 * PAIRED.block_n
    assert mosaic_gpu.pair_takes(PAIRED, family)
    for refused in (dataclasses.replace(PAIRED, block_n=256), dataclasses.replace(PAIRED, cluster=1)):
        assert not mosaic_gpu.pair_takes(refused, family), refused
    x, w = jnp.ones((256, 128), jnp.bfloat16), jnp.ones((1, 128, 512), jnp.bfloat16)
    with generation.assumed("sm100"), pytest.raises(ValueError):
        mosaic_gpu.gmm(x, w, jnp.asarray([256], jnp.int32), tiles=PAIRED, transpose_rhs=False,
                       out_dtype=jnp.bfloat16, multiprocessors=2, dense=True, prologue=lambda rows: rows)


def test_a_pairs_tile_is_legal_for_a_dense_product_on_a_b200():
    """A pair's tile is a dense product's legal tile on a B200, its streams planned as its kernel loads them
    (`mosaic_gpu.multicast`: each CTA its half of the pair's block, no multicast ring), at Qwen3.5-4B's up
    projection over a step's 24,576 tokens; a pair's tile was never legal while the planner multicast its
    matrix, which the pair's streams refuse."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    launch = grouped_matmul.MosaicGmm()
    with generation.assumed("sm100"):
        target = generation.gpu_target()
        ba = launch.bind(bf16((24576, 2560)), bf16((1, 2560, 9216)), jax.ShapeDtypeStruct((1,), jnp.int32),
                         out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=True)
        assert launch.legal(ba, dataclasses.replace(PAIRED, stages=5), target)


def test_a_b200s_products_lower_at_the_tiles_their_launch_derives():
    """On a B200, where the derivation tries a pair's bundle first (`mosaic_gpu.paired_first`), a dense
    product over a step's rows and a routed one, its outputs scattered to their slots (which a pair's
    bundle does not take, `Transport.gathers_rows`), each lower through Mosaic for sm100 at the tile their
    launch derives, which it holds legal."""
    from kernel_compile import lowered_for

    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((4,), jnp.int32)
    slots = jax.ShapeDtypeStruct((4096,), jnp.int32)
    calls = {"dense": (lambda x, w: gemm.gemm(x, w, None, out_dtype=jnp.bfloat16),
                       (bf16((4096, 1024)), bf16((1024, 2048)))),
             "scattered": (lambda x, w, sizes, slots: gemm.gemm(x, w, sizes, scatter=slots,
                                                                out_dtype=jnp.bfloat16),
                           (bf16((4096, 1024)), bf16((4, 1024, 2048)), sizes, slots))}
    for name, (call, specs) in calls.items():
        with generation.assumed("sm100"), generation.chosen(grouped_matmul="mosaic"), tuning.seen() as seen:
            lowered_for(call, specs, "sm100")
        taken = [(op, config, ba) for (op, _), (config, ba) in zip(seen, seen.taken, strict=True)
                 if isinstance(op, grouped_matmul.MosaicGmm)]
        assert taken, name
        with generation.assumed("sm100"):
            assert all(op.legal(ba, config, generation.gpu_target()) for op, config, ba in taken), name


def test_an_exactly_summed_tile_holds_the_registers_its_promotes_hold():
    """A B200's exactly summed grouped product (bf16 operands, 128 rows a group, depth and width 128) holds
    its 128 x 128 tile legal: its windows' partials and running sums stay in tensor memory, promoted a chunk
    at a time (`Tcgen05Rings._promoted`), whose two chunks the register model charges, not two register
    accumulators (B6, where 272 registers charged past the 248 budget refused the tile)."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    launch = grouped_matmul.MosaicGmm()
    with generation.assumed("sm100"):
        target = generation.gpu_target()
        ba = launch.bind(bf16((512, 128)), bf16((4, 128, 128)), jax.ShapeDtypeStruct((4,), jnp.int32),
                         out_dtype="float32", trans_rhs=False, interpret=False, exact_sums=True)
        assert launch.legal(ba, mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=1, family="tcgen05"), target)
