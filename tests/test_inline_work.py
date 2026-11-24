"""An inlined Mosaic GPU body's work as a kernel's emission counts it: read from the code the body emits
(`dew.nn.kernels.templates.inline_work`), each op over a warpgroup's lanes (row 12466)."""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import arith, memref, scf

from dew.nn.kernels.templates import emission
from dew.nn.kernels.templates.cost import Work

_ROW = jax.ShapeDtypeStruct((128,), jnp.float32)


def _counted(body, *scalars: int) -> Work:
    """`body` inlined over a shared fp32 tile of 128 words (and `scalars`, int32 operands the kernel passes as
    literals), in a kernel of one warpgroup: the work its emission counts."""
    def kernel(out_gmem, smem):
        del out_gmem
        scalar = plgpu.Layout.WG_SPLAT(())
        plgpu.inline_mgpu(arg_types=(plgpu.RefType(), *(scalar,) * len(scalars)))(body)(
            smem, *(jnp.asarray(value, jnp.int32) for value in scalars))

    launched = plgpu.kernel(kernel, out_type=_ROW, num_threads=1, thread_name="wg",
                            scratch_types=[plgpu.SMEM((128,), jnp.float32)])
    return emission.emitted(jax.make_jaxpr(launched)().jaxpr, 1, "wg").roles[0].fixed


def _scaled(ref) -> None:
    """Each thread's word of `ref` doubled in place: a load, a multiply and a store a lane."""
    index = ir.IndexType.get()
    thread = arith.index_cast(index, mgpu_utils.thread_idx())
    word = memref.load(ref, [thread])
    memref.store(arith.mulf(word, arith.constant(ir.F32Type.get(), 2.0)), ref, [thread])


def _thread(bound: int):
    return arith.cmpi(arith.CmpIPredicate.ult, mgpu_utils.thread_idx(),
                      arith.constant(ir.IntegerType.get_signless(32), bound))


def test_a_bodys_lane_work_counts_over_the_warpgroup():
    """A body each lane of which loads, scales and stores its word issues a multiply a lane and moves two
    words a lane through shared memory."""
    def body(ctx, ref):
        del ctx
        _scaled(ref)

    assert _counted(body) == Work(operations=128, shared_bytes=2 * 4 * 128)


def test_a_branch_a_lanes_place_decides_issues_for_its_warps_and_moves_its_lanes_bytes():
    """A branch the first 16 lanes take issues for their one warp, all 32 of its lanes, and moves the 16
    lanes' words alone."""
    def body(ctx, ref):
        del ctx
        with mgpu_utils.when(_thread(16)):
            _scaled(ref)

    assert _counted(body) == Work(operations=32, shared_bytes=2 * 4 * 16)


def test_a_static_loop_counts_its_trips_and_a_concrete_operand_its_lanes():
    """A loop of four static trips counts its body four times; a branch on an operand the kernel passes
    concretely (a bound of 64 lanes) takes its lanes, two warps'."""
    def looped(ctx, ref):
        del ctx
        index = ir.IndexType.get()
        lower, upper, step = (arith.constant(index, value) for value in (0, 4, 1))
        for_op = scf.ForOp(lower, upper, step)
        with ir.InsertionPoint(for_op.body):
            _scaled(ref)
            scf.YieldOp([])

    assert _counted(looped) == Work(operations=4 * 128, shared_bytes=4 * 2 * 4 * 128)

    def bounded(ctx, ref, bound):
        del ctx
        taken = arith.cmpi(arith.CmpIPredicate.ult, mgpu_utils.thread_idx(), bound.registers.item())
        with mgpu_utils.when(taken):
            _scaled(ref)

    assert _counted(bounded, 64) == Work(operations=64, shared_bytes=2 * 4 * 64)


def test_inline_ptx_counts_by_its_instructions_and_an_unknown_one_is_unpriced():
    """Inline PTX counts an instruction at a time: a packed bf16 multiply-add two operations a lane; an
    instruction no rule names is unpriced, never guessed."""
    def packed(ctx, ref):
        del ctx
        i32 = ir.IntegerType.get_signless(32)
        one = arith.constant(i32, 0x3F803F80)
        mgpu_utils.inline_ptx("fma.rn.bf16x2 $0, $1, $2, $3;", one, one, one, result_types=i32)

    assert _counted(packed) == Work(operations=2 * 128)

    def unknown(ctx, ref):
        del ctx
        i32 = ir.IntegerType.get_signless(32)
        one = arith.constant(i32, 1)
        mgpu_utils.inline_ptx("dp4a.u32.u32 $0, $1, $2, $3;", one, one, one, result_types=i32)

    with pytest.raises(emission.Unpriced):
        _counted(unknown)
