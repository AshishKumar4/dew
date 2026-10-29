"""The GEMM template's Blackwell mainloop (`dew.nn.kernels.templates.gemm.mosaic_blackwell`), first rung:
the grouped forward and input gradient with the epilogue, interpreted on the CPU by Mosaic GPU's own
interpreter, which models tensor memory and tcgen05's products, held to check 1 against the plain program
from float64 truth (empty groups, rows past the routed ones, two outputs); and compiled for a B200 on
this CPU within its shared and tensor memory, no register spilled. The interpreter is coverage, not proof
of the compiled kernel's synchronization; a B200's or a B300's run is."""

import dataclasses

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from reference_error import assert_as_exact

from dew.nn.kernels import generation
from dew.nn.kernels.grouped_matmul import INTERPRETED
from dew.nn.kernels.templates.gemm import mosaic_blackwell

pytestmark = pytest.mark.kernel

SIZES = (0, 70, 1, 129, 100, 0)
ROWS, DEPTH, WIDTH = 307, 128, 256
SEEDS = 8
"""Each site's declared draws (`reference_error.seeds_for` of the plain program on them)."""
TILES = (mosaic_blackwell.Tiles(128, 128, 64, 2, 1), mosaic_blackwell.Tiles(128, 256, 64, 4, 2))
"""A tile of each accumulator width, its two accumulators within the 512 columns of tensor memory."""


def biased(acc, bias):
    return acc + bias[None, :]


def scaled_rows(acc, scale):
    return acc * scale[:, None]


def doubled(acc):
    return acc, acc * 2


CASES = {
    "plain": ((), None, (jnp.bfloat16,)),
    "transposed": ((), None, (jnp.bfloat16,)),
    "bias": ((("bias", ("columns",)),), biased, (jnp.bfloat16,)),
    "row scale": ((("scale", ("rows",)),), scaled_rows, (jnp.float32,)),
    "two outputs": ((), doubled, (jnp.bfloat16, jnp.float32)),
}
"""Each case's epilogue side operands (drawn by name, with their launch axes), its epilogue and its
output dtypes."""
DRAWN = ("x", "w", "bias", "scale")


def _draw(seed):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.bfloat16),
            jnp.asarray(rng.normal(size=(len(SIZES), DEPTH, WIDTH)) / np.sqrt(DEPTH), jnp.bfloat16),
            jnp.asarray(rng.normal(size=(WIDTH,)), jnp.float32),
            jnp.asarray(rng.uniform(0.5, 2.0, size=(ROWS,)), jnp.float32))


def _mainloop(case: str, tiles, *, interpret=INTERPRETED, target=None):
    target = target or dataclasses.replace(generation.gpu_target("sm100"), multiprocessors=1)
    sides, epilogue, dtypes = CASES[case]

    def run(*inputs):
        drawn = dict(zip(DRAWN, inputs, strict=True))
        w = jnp.swapaxes(drawn["w"], 1, 2) if case == "transposed" else drawn["w"]
        outs = mosaic_blackwell.gmm(drawn["x"], w, jnp.asarray(SIZES), tiles=tiles,
                                    transpose_rhs=case == "transposed", out_dtype=dtypes[0],
                                    target=target, interpret=interpret, epilogue=epilogue,
                                    epilogue_operands=tuple(drawn[name] for name, _ in sides),
                                    epilogue_axes=tuple(axes for _, axes in sides),
                                    out_dtypes=dtypes if len(dtypes) > 1 else None)
        return outs if isinstance(outs, tuple) else (outs,)
    return run


def _plain(case: str, *, truth: bool = False):
    """The program the mainloop stands for: each group's rows times its matrix summed wide, the rows past
    the routed ones zero, the epilogue, each output rounded once (in float64 as truth)."""
    sides, epilogue, dtypes = CASES[case]

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
@pytest.mark.parametrize("case", list(CASES))
def test_the_blackwell_gmm_holds_check_1(case, tiles):
    """Each case through the mainloop, interpreted: as exact as the plain program from float64 truth, the
    empty groups, the group of one row and the rows past the routed ones included."""
    assert_as_exact(_mainloop(case, tiles), jax.jit(_plain(case)), _draw, f"{case} at {tiles}", seeds=SEEDS,
                    truth_fn=_plain(case, truth=True))


ON_BLACKWELL = pytest.mark.skipif(generation.device_generation() not in ("sm100", "sm103"),
                                  reason="runs the mainloop compiled, on a B200 or a B300")


@ON_BLACKWELL
@pytest.mark.gpu(reason="runs the Blackwell mainloop compiled, on a B200 or a B300")
@pytest.mark.parametrize("tiles", TILES, ids=lambda tiles: f"{tiles.block_n}x{tiles.stages}")
@pytest.mark.parametrize("case", list(CASES))
def test_the_blackwell_gmm_holds_check_1_on_blackwell(case, tiles):
    """Each case compiled and run on a B200 or a B300, its tiles over every multiprocessor, where the
    interpreter's synchronization is the hardware's: as exact as the plain program from float64 truth."""
    run = _mainloop(case, tiles, interpret=False, target=generation.gpu_target())
    assert_as_exact(jax.jit(run), jax.jit(_plain(case)), _draw, f"{case} at {tiles} on Blackwell",
                    seeds=SEEDS, truth_fn=_plain(case, truth=True))


def test_the_blackwell_gmm_compiles_for_a_b200_within_its_resources():
    """The mainloop at each tile, with an epilogue, compiled by Mosaic for a compile-only B200 on this CPU
    through ptxas (tests/kernel_compile.py): within the multiprocessor's shared memory and its 512 columns
    of tensor memory, no register spilled."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    def case(tiles):
        def call():
            def gmm(lhs, rhs, sizes, bias):
                return mosaic_blackwell.gmm(lhs, rhs, sizes, tiles=tiles, transpose_rhs=False,
                                            out_dtype=jnp.bfloat16, target=generation.gpu_target("sm100"),
                                            epilogue=biased, epilogue_operands=(bias,),
                                            epilogue_axes=(("columns",),))

            return gmm, (jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                         jax.ShapeDtypeStruct((4, 512, 1024), jnp.bfloat16),
                         jax.ShapeDtypeStruct((4,), jnp.int32), jax.ShapeDtypeStruct((1024,), jnp.float32))

        return support.KernelCase(f"blackwell gmm {tiles}", call)

    found = compile_mosaic_kernels([case(tiles) for tiles in TILES], "sm100")
    assert len(found) == len(TILES) and all(record.get("kernels") for record in found), found
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
