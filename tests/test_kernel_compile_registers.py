"""The compile gate's reading of ptxas's notes: a kernel whose assembly ptxas notes a performance loss of
fails it."""
import kernel_compile as kc
import pytest

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


@pytest.mark.parametrize(("gpus", "installed", "generation", "judged"), [
    (None, ["cuda12"], "sm90", ("cuda13", "cuda12")),
    (None, ["cuda12"], "sm100", ("cuda13",)),
    (("580.65", (8, 0)), ["cuda12"], "sm80", ("cuda12",)),
    (("580.65", (8, 0)), ["cuda13"], "sm90", ("cuda13",)),
])
def test_the_gate_judges_every_build_on_a_host_with_no_gpu_and_a_gpu_environments_own_on_one(
        monkeypatch, gpus, installed, generation, judged):
    """The kernel-compile environment, on a host with no GPU, judges a generation's kernels under every CUDA
    build its GPUs run, whatever plugin it holds; a GPU node's environment under the build its CUDA plugin is
    alone, as it holds no other build's ptxas (row 8769: an A100's cuda12 environment asked CUDA 13's)."""
    monkeypatch.setattr(kc, "nvidia_gpus", lambda: gpus)
    monkeypatch.setattr(kc, "cuda_builds", lambda: installed)
    kc.judged_builds.cache_clear()
    assert kc.judged_builds(generation) == judged


def test_a_gpu_environment_none_of_whose_builds_a_generation_runs_is_refused(monkeypatch):
    """A generation whose GPUs run no build this GPU environment installs (a B200's cuda13 on a cuda12
    node) is refused, never judged under no assembler."""
    monkeypatch.setattr(kc, "nvidia_gpus", lambda: ("580.65", (8, 0)))
    monkeypatch.setattr(kc, "cuda_builds", lambda: ["cuda12"])
    kc.judged_builds.cache_clear()
    with pytest.raises(RuntimeError, match="sm100's GPUs run JAX's cuda13 build"):
        kc.judged_builds("sm100")
