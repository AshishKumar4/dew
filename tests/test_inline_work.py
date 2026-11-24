"""An inlined Mosaic GPU body's work as a kernel's emission counts it: read from the code the body emits
(`dew.nn.kernels.templates.inline_work`), each op over a warpgroup's lanes (row 12466)."""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl
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


def _ptx(instruction: str):
    """A body inlining `instruction`, of one 32-bit result and three operands."""
    def body(ctx, ref):
        del ctx, ref
        i32 = ir.IntegerType.get_signless(32)
        word = arith.constant(i32, 0x3F803F80)
        mgpu_utils.inline_ptx(f"{instruction} $0, $1, $2, $3;", word, word, word, result_types=i32)
    return body


def test_inline_ptx_counts_by_its_instructions_and_an_unknown_one_is_unpriced():
    """Inline PTX counts an instruction at a time: a packed bf16 multiply-add two operations a lane; an
    instruction no rule names is unpriced, never guessed."""
    assert _counted(_ptx("fma.rn.bf16x2")) == Work(operations=2 * 128)
    with pytest.raises(emission.Unpriced):
        _counted(_ptx("dp4a.u32.u32"))


def test_a_warps_tensor_core_product_counts_its_operations_in_its_mode():
    """A TF32 matrix product a warp issues in inline PTX (mma.sync's m16n8k8, the recurrence solve's) counts
    2 M N K operations each warp, on the TF32 tensor cores."""
    assert _counted(_ptx("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32")) == Work(
        tf32_flops=2 * 16 * 8 * 8 * (128 // 32))


def test_a_store_its_operand_guards_counts_the_lanes_that_take_it():
    """A store guarded by an operand (`inline_ptx`'s predicate: the first lane's) moves that lane's word
    alone; one guarded by a predicate the block sets itself of a constant moves what that predicate's lanes
    move (none), and one of the data is unpriced, its lanes unknown."""
    def elected(ctx, ref):
        del ctx, ref
        i32 = ir.IntegerType.get_signless(32)
        word = arith.constant(i32, 1)
        mgpu_utils.inline_ptx("st.shared.u32 [$0], $1;", word, word, predicate=_thread(1))

    assert _counted(elected) == Work(shared_bytes=4)

    def guarded(held):
        def body(ctx, ref):
            del ctx
            i32 = ir.IntegerType.get_signless(32)
            word = held(ref)
            mgpu_utils.inline_ptx("{ .reg .pred p; setp.eq.u32 p, $1, 0; @p st.shared.u32 [$0], $1; }",
                                  arith.constant(i32, 1), word, has_side_effects=True)
        return body

    # A predicate the block sets of a constant: no lane holds it, so the store moves nothing.
    i32 = ir.IntegerType.get_signless
    assert _counted(guarded(lambda ref: arith.constant(i32(32), 1))) == Work()

    def loaded(ref):
        # A word of the data: which lanes hold the predicate set of it is unknown.
        value = memref.load(ref, [arith.constant(ir.IndexType.get(), 0)])
        return arith.bitcast(i32(32), value)

    with pytest.raises(emission.Unpriced):
        _counted(guarded(loaded))


def test_a_copy_through_the_launch_context_moves_its_shared_tile_once():
    """A copy a body asks its launch context for (`ctx.async_copy`, a gathered ring's TMA rows) moves its
    shared tile's bytes between device and shared memory once, whatever lanes issue it."""
    def copied(ctx, ref):
        ctx.async_copy(src_ref=ref, dst_ref=ref)

    assert _counted(copied) == Work(device_bytes=4 * 128, shared_bytes=4 * 128)


def test_a_tickets_elected_and_last_lanes_move_their_words_alone():
    """A ticket's PTX (`across.arrived`'s) guards its atomic by the lane it elects itself (`setp` on an
    operand whose lanes are known: the first thread) and its reset by the last arriver among it: the atomic
    moves one lane's word, the reset at most that lane's, where a block's own predicate once refused it."""
    def ticket(ctx, ref):
        del ctx, ref
        i32 = ir.IntegerType.get_signless(32)
        thread = arith.remui(mgpu_utils.thread_idx(), arith.constant(i32, 128))
        count = arith.constant(i32, 1)
        mgpu_utils.inline_ptx("""{
            .reg .pred elected, last;
            .reg .b32 before;
            setp.eq.u32 elected, $1, 0;
            @elected atom.acq_rel.gpu.global.add.u32 before, [$2], 1;
            setp.eq.and.u32 last, before, $3, elected;
            @last st.relaxed.gpu.global.u32 [$2], 0;
            selp.u32 $0, 1, 0, last;
            }""", thread, count, count, result_types=i32, has_side_effects=True)

    assert _counted(ticket) == Work(device_bytes=8 + 4)


def test_a_launch_contexts_copy_counts_under_its_branch_and_loop():
    """A copy a body asks its launch context for inside a branch no lane takes, or under an issuer no lane
    holds, moves nothing, and inside a static loop of two trips moves its tile twice: it is counted where
    the body asks for it, under its issuer's guard."""
    def untaken(ctx, ref):
        with mgpu_utils.when(_thread(0)):
            ctx.async_copy(src_ref=ref, dst_ref=ref)

    assert _counted(untaken) == Work()

    def unissued(ctx, ref):
        # Its issuer a predicate no lane holds.
        ctx.async_copy(src_ref=ref, dst_ref=ref, predicate=_thread(0))

    assert _counted(unissued) == Work()

    def looped(ctx, ref):
        index = ir.IndexType.get()
        lower, upper, step = (arith.constant(index, value) for value in (0, 2, 1))
        for_op = scf.ForOp(lower, upper, step)
        with ir.InsertionPoint(for_op.body):
            ctx.async_copy(src_ref=ref, dst_ref=ref)
            scf.YieldOp([])

    assert _counted(looped) == Work(device_bytes=2 * 4 * 128, shared_bytes=2 * 4 * 128)


@pytest.mark.parametrize("form", ["encode-e4m3", "decode-nvfp4"])
def test_a_forms_inline_block_compiles_for_an_h100(form):
    """A bit form inlined as one PTX block a word (`inline_forms.registers`) compiles for an H100 on any
    host, its one output register the block's value (an encode's, a quantized GEMM epilogue's: LLVM refuses
    a struct of one output) and several a struct of them (a decode's)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.algebra import bits
    from dew.nn.kernels.templates import inline_forms

    chosen = bits.encode_form(bits.E4M3) if form == "encode-e4m3" else bits.NVFP4_BFLOAT16_FORM
    shape = (64, 16)

    def body(words_ref, *out_refs):
        decoded = inline_forms.registers(chosen, plgpu.layout_cast(words_ref[...], plgpu.Layout.WGMMA),
                                         plgpu.Layout.WGMMA)
        for at, out_ref in enumerate(out_refs):
            out_ref[...] = decoded[at]

    launch = plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct(shape, jnp.uint32),) * len(chosen.outputs),
                          num_threads=1, thread_name="wg")
    case = KernelCase(form, lambda: (launch, (jax.ShapeDtypeStruct(shape, jnp.uint32),)))
    assert not (failed := mosaic_failures(compile_mosaic_kernels([case], "sm90"))), failed


def test_a_store_a_branch_holds_under_a_guard_of_the_data_is_refused():
    """A store some lanes reach (a branch's first 16) under a guard the block sets of the data is refused,
    its lanes unknown among the active ones (a partial active mask once passed it as bounded)."""
    def body(ctx, ref):
        del ctx
        i32 = ir.IntegerType.get_signless(32)
        with mgpu_utils.when(_thread(16)):
            word = arith.bitcast(i32, memref.load(ref, [arith.constant(ir.IndexType.get(), 0)]))
            mgpu_utils.inline_ptx("{ .reg .pred p; setp.eq.u32 p, $1, 010; @p st.shared.u32 [$0], $1; }",
                                  arith.constant(i32, 1), word, has_side_effects=True)

    with pytest.raises(emission.Unpriced):
        _counted(body)


def test_a_comparison_reads_its_sides_at_their_width_and_signedness():
    """A thread's index less one compared unsigned with 32 bits' largest (PTX's octal `037777777777`, and
    arith's `ult` against -1) holds on the first lane alone, its -1 being that largest: a store it guards
    moves one word, and a branch on it runs one lane's."""
    def octal(ctx, ref):
        del ctx, ref
        i32 = ir.IntegerType.get_signless(32)
        below = arith.subi(arith.remui(mgpu_utils.thread_idx(), arith.constant(i32, 128)),
                           arith.constant(i32, 1))
        mgpu_utils.inline_ptx("{ .reg .pred p; setp.eq.u32 p, $1, 037777777777; @p st.shared.u32 [$0], $1; }",
                              arith.constant(i32, 1), below, has_side_effects=True)

    assert _counted(octal) == Work(shared_bytes=4)

    def unsigned(ctx, ref):
        del ctx
        i32 = ir.IntegerType.get_signless(32)
        below = arith.subi(arith.remui(mgpu_utils.thread_idx(), arith.constant(i32, 128)),
                           arith.constant(i32, 1))
        with mgpu_utils.when(arith.cmpi(arith.CmpIPredicate.uge, below, arith.constant(i32, 127))):
            _scaled(ref)

    assert _counted(unsigned) == Work(operations=32, shared_bytes=2 * 4)


def test_a_body_one_warps_role_runs_counts_that_warps_lanes_alone():
    """A body the second warp's role runs (`plgpu.warp_map`) issues its multiply over that warp's 32 lanes
    and moves their 32 words through shared memory, not a warpgroup's 128."""
    def body(ctx, ref):
        del ctx
        _scaled(ref)

    def kernel(out_gmem, smem):
        del out_gmem

        @plgpu.warp_map
        def _(warp):
            pl.when(warp == 1)(lambda: plgpu.inline_mgpu(arg_types=(plgpu.RefType(),))(body)(smem))

    launched = plgpu.kernel(kernel, out_type=_ROW, num_threads=1, thread_name="wg",
                            scratch_types=[plgpu.SMEM((128,), jnp.float32)])
    counted = emission.emitted(jax.make_jaxpr(launched)().jaxpr, 1, "wg").roles[0].fixed
    assert (counted.operations, counted.shared_bytes) == (32, 2 * 4 * 32)
