"""The compile gate's reading of ptxas's notes: a kernel whose assembly ptxas notes a performance loss of
fails it."""
import jax
import jax.numpy as jnp
import kernel_compile as kc
import pytest
from fla_gdn_support import KernelCase
from generation_support import lowers
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import report

SERIALIZING = ("ptxas info    : (C7512) Potential Performance Loss: wgmma.mma_async instructions are "
               "serialized due to insufficient register resources for the function "
               "'recurrence_output_mosaic_gpu_kernel'\n"
               "ptxas info    : Compiling entry function 'recurrence_output_mosaic_gpu_kernel' for 'sm_90a'\n"
               "ptxas info    : Used 152 registers, used 1 barriers, 0 bytes smem\n"
               "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads")
"""ptxas's report of the recurrence output kernel at a consumer cap of 152 (FS-LT-OUTPUT-CAPS1): spill-free,
its warpgroup MMAs serialized."""


IGNORING = ("ptxas info    : (C7507) Potential Performance Loss: 'setmaxnreg' ignored to maintain minimum "
            "register requirements\n"
            "ptxas info    : Compiling entry function 'gemm_mosaic_gpu_kernel' for 'sm_90a'\n"
            "ptxas info    : Used 168 registers, used 1 barriers, 0 bytes smem\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads")
"""ptxas's report of a warp-specialized kernel whose roles' paths rejoin after their setmaxnreg: the kernel
challenge's C7507 (notes/ptxas-re.md §4, 'setmaxnreg' ignored), its threads left at the entry's registers."""


@pytest.mark.parametrize(("notes", "registers", "code", "named"),
                         [(SERIALIZING, 152, "C7512", "serialize"), (IGNORING, 168, "C7507", "setmaxnreg")])
def test_a_kernel_ptxas_notes_a_performance_loss_of_fails_the_gate(notes, registers, code, named):
    """A Mosaic GPU kernel ptxas notes a performance loss of fails the compile gate (`mosaic_failures`), its
    registers and spills clean: one whose warpgroup MMAs it serializes (C7510 to C7515: each HGMMA waited, as
    a spill-free 152-register recurrence output was) runs its MMAs one at a time (row 3615; the kernel
    challenge's C7511 at 112 registers, C7512 at 96), and one whose setmaxnreg it ignores (C7507: code after
    the warp roles' split reachable from both budgets, every raise and decrease dropped) keeps its consumers
    at the entry's registers, never the ones its tile was sized for (row 4071)."""
    read = kc.ptxas_read(notes)
    assert (read.registers, read.spilled, read.losses) == (registers, 0, (code,))
    (failure,) = kc.mosaic_failures([_assembled_record("recurrence_output_mosaic_gpu_kernel", read)])
    assert named in failure


def _assembled_record(kernel: str, read) -> dict:
    """A compile record of one spill-free H100 kernel as ptxas `read` it."""
    return {"name": kernel, "generation": "sm90", "limit": 232448, "tmem_limit": 0,
            "kernels": [{"kernel": kernel, "shared": 1024, "tmem": 0, "registers": read.registers,
                         "spilled": read.spilled, "losses": read.losses, "unread": []}]}


@pytest.fixture
def judging():
    """judged_builds' cache empty around a cell that fakes its host: no later cell reads the fake's answer."""
    kc.judged_builds.cache_clear()
    yield
    kc.judged_builds.cache_clear()


@pytest.mark.parametrize(("gpus", "installed", "generation", "judged"), [
    (None, ["cuda12"], "sm90", ("cuda13", "cuda12")),
    (None, ["cuda12"], "sm100", ("cuda13",)),
    (("580.65", (8, 0)), ["cuda12"], "sm80", ("cuda12",)),
    (("580.65", (8, 0)), ["cuda13"], "sm90", ("cuda13",)),
    (("580.65", (8, 0)), [], "sm90", ("cuda13", "cuda12")),
])
def test_the_gate_judges_every_build_on_a_host_with_no_gpu_and_a_gpu_environments_own_on_one(
        monkeypatch, judging, gpus, installed, generation, judged):
    """The kernel-compile environment, on a host with no GPU, judges a generation's kernels under every CUDA
    build its GPUs run, whatever plugin it holds, as does an environment with none on any host; a GPU node's
    environment under the build its CUDA plugin is alone, as it holds no other build's ptxas (row 8769: an
    A100's cuda12 environment asked CUDA 13's)."""
    monkeypatch.setattr(kc, "nvidia_gpus", lambda: gpus)
    monkeypatch.setattr(kc, "cuda_builds", lambda: installed)
    assert kc.judged_builds(generation) == judged


def test_a_gpu_environment_none_of_whose_builds_a_generation_runs_is_refused(monkeypatch, judging):
    """A generation whose GPUs run no build this GPU environment installs (a B200's cuda13 on a cuda12
    node) is refused, never judged under no assembler."""
    monkeypatch.setattr(kc, "nvidia_gpus", lambda: ("580.65", (8, 0)))
    monkeypatch.setattr(kc, "cuda_builds", lambda: ["cuda12"])
    with pytest.raises(RuntimeError):
        kc.judged_builds("sm100")
    # The same environment judges a generation its build runs: the refusal is the generation's alone.
    assert kc.judged_builds("sm80") == ("cuda12",)


def _register_bound_product(depth: int):
    """A warpgroup's 64 x 128 product of a depth-`depth` operand it holds in registers by one staged in
    shared memory, five blocks a multiprocessor (`min_blocks_per_sm`, a launch's register cap of 96), which
    leaves its MMA pipeline short of registers: ptxas serializes it (FS-6456-SCAN5: C7511 spill-free at
    depth 128, C7512 at 64)."""
    from jax._src.pallas.mosaic_gpu import pallas_call as mosaic_pallas_call
    from jax.experimental import pallas as pl

    swizzled = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))

    def kernel(a_ref, b_ref, o_ref):
        def scope(acc_ref):
            plgpu.wgmma(acc_ref, a_ref[...], b_ref)
            return acc_ref[...]

        o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 128), jnp.float32))

    return mosaic_pallas_call.pallas_call(
        kernel,
        in_specs=[plgpu.BlockSpec((64, depth), lambda: (0, 0), transforms=swizzled),
                  plgpu.BlockSpec((depth, 128), lambda: (0, 0), transforms=swizzled)],
        out_specs=plgpu.BlockSpec((64, 128), lambda: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((64, 128), jnp.float32),
        compiler_params=plgpu.CompilerParams(min_blocks_per_sm=5))


@lowers("sm90")
@pytest.mark.parametrize("depth", [128, 64])
def test_a_kernel_ptxas_serializes_carries_ptxas_own_notes_in_its_binary(depth):
    """A warpgroup product ptxas serializes, compiled by Dew's JAX build for a compile-only H100: the notes
    its binary carries (`report.Binary.notes`, the native compile's own) are ptxas's of its PTX, the gate's
    read of it, a serialization among them, so a run's certification refuses it on its binary's word (row
    6456: the notes once read none, a serialized kernel passing `Binary.refused`)."""
    shapes = (jax.ShapeDtypeStruct((64, depth), jnp.float16), jax.ShapeDtypeStruct((depth, 128), jnp.float16))
    (record,) = kc.compile_mosaic_kernels(
        [KernelCase("register_bound", lambda: (_register_bound_product(depth), shapes))], "sm90")
    (kernel,) = record["kernels"]
    assert set(kernel["losses"]) & report.SERIALIZED, kernel
    assert tuple(kernel["binary_losses"]) == tuple(kernel["losses"]), kernel

