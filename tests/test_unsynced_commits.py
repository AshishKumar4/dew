"""A tensor-core commit at a barrier its thread waited at, with no warpgroup sync between
(`lowered.unsynced_commits`, row 5753): the race row 5671 wedged a B200 by, read from a kernel's typed
module as a compile-only sm100 lowers it (tests/kernel_compile.py's gate)."""
import jax
import jax.numpy as jnp
import pytest
from generation_support import COMPILES_ON_CPU
from jax._src.lib.mlir.dialects import nvvm
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir
from kernel_compile import compile_mosaic_kernels, failures, lowered_for

from dew.nn.kernels.lowered import kernel_calls, unsynced_commits

pytestmark = pytest.mark.kernel

OPERAND = jax.ShapeDtypeStruct((128,), jnp.float32)


def _warp_barrier(ctx) -> None:
    """One warp's barrier (32 threads on barrier 1), inlined: no sync of a warpgroup's other warps."""
    del ctx
    i32 = ir.IntegerType.get_signless(32)
    nvvm.barrier(barrier_id=mgpu_utils.c(1, i32), number_of_threads=mgpu_utils.c(mgpu_utils.WARP_SIZE, i32))


BETWEEN = {"nothing": lambda: None, "a warpgroup's sync": plgpu.commit_smem,
           "a warp's barrier": lambda: plgpu.inline_mgpu()(_warp_barrier)()}
"""What a kernel runs between its wait and its next commit (`_committed_after_its_wait`)."""


def _committed_after_its_wait(between: str, slot: int = 0):
    """A kernel whose warpgroup commits at the first of its two tensor-core barriers, waits at it, runs
    `between` (`BETWEEN`) and commits at barrier `slot`."""
    def body(x_ref, o_ref, barriers):
        plgpu.tcgen05_commit_arrive(barriers.at[0])
        plgpu.barrier_wait(barriers.at[0])
        BETWEEN[between]()
        plgpu.tcgen05_commit_arrive(barriers.at[slot])
        plgpu.barrier_wait(barriers.at[slot])
        o_ref[...] = x_ref[...]

    return plgpu.kernel(body, out_type=OPERAND, scratch_types=[plgpu.Barrier(num_barriers=2,
                                                                             orders_tensor_core=True)])


def _found(kernel) -> list[str]:
    lowered = lowered_for(kernel, (OPERAND,), "sm100")
    (call,) = [call for call in kernel_calls(lowered) if call.kind == "mosaic_gpu"]
    return unsynced_commits(call.ir)


@COMPILES_ON_CPU
@pytest.mark.parametrize("between", list(BETWEEN))
def test_a_commit_after_its_wait_is_found_unless_a_warpgroup_sync_lies_between(between):
    """A commit at the barrier its warpgroup waited at is found with nothing between or only a warp's own
    barrier, which syncs none of the warpgroup's other warps, and not past the warpgroup's sync (row 12144:
    any `nvvm.barrier` read as a sync)."""
    found = _found(_committed_after_its_wait(between))
    assert (found == []) == (between == "a warpgroup's sync"), found


@COMPILES_ON_CPU
@pytest.mark.parametrize("slot", [0, 1])
def test_a_commit_at_another_slot_of_the_waited_array_is_not_found(slot):
    """A commit after a wait at one slot of a barrier array is found at that slot and not at another a
    constant names: the array's slots are barriers apart (row 12144: every slot read as the array)."""
    assert (_found(_committed_after_its_wait("nothing", slot)) == []) == (slot == 1)


@COMPILES_ON_CPU
def test_the_compile_gate_fails_a_case_whose_commit_passes_a_wait():
    """The shared Mosaic GPU compile gate (`compile_mosaic_kernels`, every case's `failures`) fails a case
    whose commit passes another warp's wait, before it reads its kernels, where only the template gates' own
    collector read it (row 12144: the SYRK gate's cases)."""
    from fla_gdn_support import KernelCase

    case = KernelCase("unsynced", lambda: (_committed_after_its_wait("nothing"), (OPERAND,)))
    assert failures(compile_mosaic_kernels([case], "sm100"))
