"""A tensor-core commit at a barrier its thread waited at, with no warpgroup sync between
(`lowered.unsynced_commits`, row 5753): the race row 5671 wedged a B200 by, read from a kernel's typed
module as a compile-only sm100 lowers it (tests/kernel_compile.py's gate)."""
import jax
import jax.numpy as jnp
import pytest
from generation_support import COMPILES_ON_CPU
from jax.experimental.pallas import mosaic_gpu as plgpu
from kernel_compile import lowered_for

from dew.nn.kernels.lowered import kernel_calls, unsynced_commits

pytestmark = pytest.mark.kernel


def _commits_after_its_wait(synced: bool):
    """A kernel whose warpgroup commits at its tensor-core barrier, waits at it and commits at it again, with
    a warpgroup sync between the wait and the second commit (`commit_smem`'s) where `synced`."""
    def body(x_ref, o_ref, barrier):
        plgpu.tcgen05_commit_arrive(barrier)
        plgpu.barrier_wait(barrier)
        if synced:
            plgpu.commit_smem()
        plgpu.tcgen05_commit_arrive(barrier)
        plgpu.barrier_wait(barrier)
        o_ref[...] = x_ref[...]

    return plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
                        scratch_types=[plgpu.Barrier(orders_tensor_core=True)])


@COMPILES_ON_CPU
@pytest.mark.parametrize("synced", [False, True])
def test_a_commit_after_its_wait_is_found_unless_a_warpgroup_sync_lies_between(synced):
    operand = jax.ShapeDtypeStruct((128,), jnp.float32)
    lowered = lowered_for(_commits_after_its_wait(synced), (operand,), "sm100")
    (call,) = [call for call in kernel_calls(lowered) if call.kind == "mosaic_gpu"]
    found = unsynced_commits(call.ir)
    assert (found == []) == synced, found
