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


def _turned(times: int, turn) -> None:
    """`turn()` in a loop of `times` turns (`jax.lax.fori_loop`: an `scf.for` of constant bounds)."""
    jax.lax.fori_loop(0, times, lambda index, carry: (turn(), carry)[1], 0)


BETWEEN = {"nothing": lambda: None, "a warpgroup's sync": plgpu.commit_smem,
           "a warp's barrier": lambda: plgpu.inline_mgpu()(_warp_barrier)(),
           "a one-turn loop's sync": lambda: _turned(1, plgpu.commit_smem),
           "a no-turn loop's sync": lambda: _turned(0, plgpu.commit_smem),
           "a while loop's sync": lambda: jax.lax.while_loop(
               lambda turn: turn < 1, lambda turn: (plgpu.commit_smem(), turn + 1)[1], 0)}
"""What a kernel runs between its wait and its next commit (`_committed_after_its_wait`)."""

SYNCED = ("a warpgroup's sync", "a one-turn loop's sync")
"""What syncs every warp of the warpgroup between (`BETWEEN`): a loop's sync where it turns at least once."""


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
    """A commit at the barrier its warpgroup waited at is found with nothing between, only a warp's own
    barrier, which syncs none of the warpgroup's other warps, or a sync in a loop that turns no times or in a
    while loop's body, which its condition may leave unturned, and not past the warpgroup's sync, in a loop
    that turns once among them (row 12144: any `nvvm.barrier` read as a sync; its review: a loop of constant
    bounds read as one that may not turn)."""
    found = _found(_committed_after_its_wait(between))
    assert (found == []) == (between in SYNCED), found


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


def _through_an_edge(module: bytes) -> bytes:
    """`module` with its second commit's barrier passed to it along a branch's edge (`cf.br`): its kernel's
    block split before that commit's guard, the barrier the new block's argument."""
    import io
    import re

    import jax.experimental.mosaic.gpu as mgpu
    from jax.experimental.mosaic.gpu import core as mgpu_core

    typed = mgpu.deserialized(module)
    with typed.context, ir.Location.unknown():
        lines = str(typed).splitlines()
        at = [index for index, line in enumerate(lines) if "tcgen05.commit" in line][1]
        (barrier,) = re.findall(r"\((%[\w.#]+)\)", lines[at])
        guard = max(index for index in range(at) if lines[index].lstrip().startswith("scf.if"))
        lines[at] = lines[at].replace(f"({barrier})", "(%passed)")
        lines[guard:guard] = [f"cf.br ^passing({barrier} : !llvm.ptr<3>)", "^passing(%passed: !llvm.ptr<3>):"]
        edited = mgpu_core._run_serde_pass(ir.Module.parse("\n".join(lines)), serialize=True)
        written = io.BytesIO()
        edited.operation.write_bytecode(written)
    return written.getvalue()


@COMPILES_ON_CPU
@pytest.mark.parametrize("between", ["nothing", "a warpgroup's sync"])
def test_a_barrier_passed_along_a_branchs_edge_is_the_barrier_passed(between):
    """A commit whose barrier a branch passes its block (`cf.br`'s operand, the block's argument) commits at
    the barrier passed: found after a wait at it with nothing between, not past the warpgroup's sync (row
    12144's review: a block's argument read as a barrier of its own)."""
    lowered = lowered_for(_committed_after_its_wait(between), (OPERAND,), "sm100")
    (call,) = [call for call in kernel_calls(lowered) if call.kind == "mosaic_gpu"]
    assert (unsynced_commits(_through_an_edge(call.ir)) == []) == (between in SYNCED)
