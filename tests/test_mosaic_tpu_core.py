"""A TPU core's kernel toolkit (`dew.nn.kernels.templates.mosaic_tpu`): its reads through HBM, its programs'
adds into one ref in turn, its plans within its scoped VMEM and the GEMM template's kernel bodies on its
bundle, interpreted on the host's CPU and lowered for a TPU v5e on any host."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import on_a_v5e, on_the_host
from jax.experimental import pallas as pl

from dew.nn.kernels import generation
from dew.nn.kernels.templates import resources


def _read_through(x, rows, index, *, interpret):
    """`x * index[0] + rows[:, None]` on a TPU core, each operand left in HBM and read through its family
    (`tile_of`, `scalar_of`, `rows_of`), the result stored by a copy out of VMEM."""
    from jax.experimental.pallas import tpu as pltpu

    from dew.nn.kernels.templates import mosaic_tpu
    family = mosaic_tpu.MXU

    def body(x_hbm, rows_hbm, index_hbm, out_hbm):
        value = (family.tile_of(x_hbm) * family.scalar_of(index_hbm).astype(jnp.float32)
                 + family.rows_of(rows_hbm)[:, None])

        def write(held):
            held[...] = value
            pltpu.sync_copy(held, out_hbm)

        pl.run_scoped(write, pltpu.VMEM(x.shape, jnp.float32))

    launch = family.kernel(body, out_type=jax.ShapeDtypeStruct(x.shape, jnp.float32), grid=(1,),
                           grid_names=("programs",), consumers=1, interpret=interpret, name="reads")
    return launch(x, rows, index)


def test_a_tpu_core_reads_its_hbm_operands_through_its_family():
    """A TPU core's tile, row vector and scalar read from operands left in HBM (`VmemRings.tile_of`,
    `rows_of`, `scalar_of`, each through a copy): interpreted on any host, exactly `x * index[0] + rows`, on
    the host's CPU (`on_the_host`: its interpreter's callbacks make no program a tier's prepare stores);
    lowered for a TPU v5e."""
    x = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)
    rows, index = np.arange(8, dtype=np.float32) * 1000, np.asarray([3], np.int32)
    with on_the_host():
        got = _read_through(x, rows, index, interpret=True)
    np.testing.assert_array_equal(np.asarray(got), x * 3 + rows[:, None])
    shapes = [jax.ShapeDtypeStruct(each.shape, each.dtype) for each in (x, rows, index)]
    with on_a_v5e():
        jax.jit(functools.partial(_read_through, interpret=False)).trace(*shapes).lower(
            lowering_platforms=("tpu",))


def _added_in_place(x, *, interpret):
    """Three programs of a TPU core each adding `x * (its index + 1)` into one in-place ref's rows
    (`VmemRings.added`, the ref a `jax.new_ref` the launch takes as an operand), beside an output each
    program writes its index into: the ref after the launch, and the output."""
    from jax.experimental.pallas import tpu as pltpu

    from dew.nn.kernels.templates import mosaic_tpu
    family = mosaic_tpu.MXU
    summed = jax.new_ref(jnp.zeros(x.shape, jnp.float32))

    def body(x_hbm, summed_hbm, out_hbm):
        index = family.program_index("programs")
        family.added(family.tile_of(x_hbm) * (index + 1).astype(jnp.float32), summed_hbm, interpret,
                     smem=None)

        def write(held):
            held[...] = jnp.full(held.shape, index, jnp.int32)
            pltpu.sync_copy(held, out_hbm.at[pl.ds(index, 1)])

        pl.run_scoped(write, pltpu.VMEM((1, 128), jnp.int32))

    launch = family.kernel(body, out_type=jax.ShapeDtypeStruct((3, 128), jnp.int32), grid=(3,),
                           grid_names=("programs",), consumers=1, interpret=interpret, name="added")
    out = launch(x, summed)
    return summed[...], out


def test_a_tpu_cores_programs_add_into_an_in_place_ref_in_turn():
    """Three programs of a TPU core adding into one ref the launch takes in place (a launch's dQ, its grouped
    dK and dV), each read-modify-write through VMEM in turn: interpreted on any host, the ref holds the three
    adds' sum exactly, and the launch's own output each program's row, on the host's CPU (`on_the_host`);
    lowered for a TPU v5e."""
    x = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)
    with on_the_host():
        summed, out = jax.jit(functools.partial(_added_in_place, interpret=True))(x)
    np.testing.assert_array_equal(np.asarray(summed), x * 6)
    np.testing.assert_array_equal(np.asarray(out), np.repeat(np.arange(3, dtype=np.int32)[:, None], 128, 1))
    with on_a_v5e():
        jax.jit(functools.partial(_added_in_place, interpret=False)).trace(
            jax.ShapeDtypeStruct(x.shape, x.dtype)).lower(lowering_platforms=("tpu",))


def _gemm_on_a_tpu_core():
    """The GEMM template's kernel bodies on a TPU core's bundle (`mosaic_tpu.MXU`), each `(call, operands,
    expected)` at small integers, so each product is exact in fp32: a dense forward with a bias epilogue,
    the grouped forward (three groups, one empty, rows past the routed ones zero) with its matrices stored
    as they come and transposed, and the grouped kernel gradient."""
    from dew.nn.kernels.templates.gemm import mosaic_gpu, tiling

    rows, depth, width = 384, 512, 256
    tiles = tiling.Tiles(128, 128, 128, 2, 1, family="mxu")
    rng = np.random.default_rng(4)
    x, w, dy = (rng.integers(-3, 4, shape).astype(jnp.bfloat16)
                for shape in ((rows, depth), (3, depth, width), (rows, width)))
    bias = rng.integers(-3, 4, (width,)).astype(np.float32)
    sizes = np.asarray([100, 0, 250], np.int32)
    xf, wf, dyf = (np.asarray(each, np.float32) for each in (x, w, dy))
    unrouted = np.zeros((rows - 350, width), np.float32)
    routed = np.concatenate([xf[:100] @ wf[0], xf[100:350] @ wf[2], unrouted])
    grouped = functools.partial(mosaic_gpu.gmm, tiles=tiles, out_dtype=jnp.float32, multiprocessors=2)
    return {
        "dense bias": (functools.partial(mosaic_gpu.gmm, tiles=tiles, transpose_rhs=False,
                                         out_dtype=jnp.float32, multiprocessors=1, dense=True,
                                         epilogue=lambda acc, b: acc + b[None, :], epilogue_operands=(bias,),
                                         epilogue_axes=(("columns",),)),
                       (x, w[:1], None), xf @ wf[0] + np.asarray(bias)[None, :]),
        "grouped": (functools.partial(grouped, transpose_rhs=False), (x, w, sizes), routed),
        "grouped transposed": (functools.partial(grouped, transpose_rhs=True),
                               (x, np.swapaxes(w, 1, 2), sizes), routed),
        "kernel gradient": (functools.partial(mosaic_gpu.tgmm, tiles=tiles, out_dtype=jnp.float32,
                                              multiprocessors=2), (x, dy, sizes),
                            np.stack([xf[:100].T @ dyf[:100], np.zeros((depth, width), np.float32),
                                      xf[100:350].T @ dyf[100:350]])),
    }


def test_a_tpu_launch_plans_one_program_within_its_cores_scoped_vmem():
    """`resources.plan` on a TPU v5e for a launch on its core's bundle (`mosaic_tpu.MXU`, which budgets no
    registers): one program at a ring its scoped VMEM holds, its DMA semaphores and its turns' schedule
    (`schedule`, a role's) no VMEM byte of it (a scope holding them plans as one without them); one whose
    floor of two stages the scoped VMEM does not hold plans nothing."""
    from jax.experimental.pallas import tpu as pltpu

    from dew.nn.kernels.templates import mosaic_tpu

    family = mosaic_tpu.MXU

    def launch(stage_bytes: int, *, signalled: bool) -> resources.Launch:
        def scope(stages: int) -> resources.Scope:
            semaphores = (pltpu.SemaphoreType.DMA((stages,)), family.schedule(1)) if signalled else ()
            return resources.Scope(program=(pltpu.VMEM((stages * stage_bytes,), jnp.uint8), *semaphores))

        return resources.Launch(scope, consumers=1, registers=0)

    with generation.assumed("v5e"):
        target = generation.gpu_target()
        plain, signalled = (resources.plan([launch(2 ** 20, signalled=held)], family, target)
                            for held in (False, True))
        assert plain is not None and plain == signalled
        (found,) = plain
        assert found.programs == 1 and found.stages >= 2 and found.shared <= target.scoped_vmem
        assert resources.plan([launch(target.scoped_vmem, signalled=True)], family, target) is None


@pytest.mark.parametrize("case", ["dense bias", "grouped", "grouped transposed", "kernel gradient"])
def test_the_gemms_kernel_bodies_run_on_a_tpu_core(case):
    """Interpreted on any host, each of the GEMM template's kernel bodies on a TPU core's bundle gives its
    product exactly, and it lowers for a TPU v5e, its devices abstract. Interpreted on the host's CPU
    (`on_the_host`), its operands NumPy's."""
    call, operands, expected = _gemm_on_a_tpu_core()[case]
    with on_the_host():
        np.testing.assert_array_equal(np.asarray(call(*operands, interpret=True)), expected)
    abstract = [None if each is None else jax.ShapeDtypeStruct(each.shape, each.dtype) for each in operands]
    with on_a_v5e():
        jax.jit(call).trace(*abstract).lower(lowering_platforms=("tpu",))
