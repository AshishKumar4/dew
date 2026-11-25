"""An inlined Mosaic GPU body's work as a kernel's emission counts it: read from the code the body emits
(`dew.nn.kernels.templates.inline_work`), each op over a warpgroup's lanes (row 12466)."""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import arith, math, memref, scf

from dew.nn.kernels.templates import cost, emission
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


def _uniform_branch(*, multiplied: bool, exponentiated: bool):
    """A body branching on a word every lane loads at one place, data its lanes hold alike: its then arm a
    multiply a lane where `multiplied` (the fp32 lanes'), its else arm an exponential a lane where
    `exponentiated` (the special-function units')."""
    def body(ctx, ref):
        del ctx
        first = memref.load(ref, [arith.constant(ir.IndexType.get(), 0)])
        positive = arith.cmpf(arith.CmpFPredicate.OGT, first, arith.constant(ir.F32Type.get(), 0.0))
        branch = scf.IfOp(positive, [], has_else=True)
        with ir.InsertionPoint(branch.then_block):
            if multiplied:
                arith.mulf(first, first)
            scf.YieldOp([])
        with ir.InsertionPoint(branch.else_block):
            if exponentiated:
                math.exp2(first)
            scf.YieldOp([])

    return body


def test_a_uniform_branch_issues_its_costliest_arms_slots():
    """A branch data every lane holds alike decides runs one arm, so its issue slots are the costlier arm's,
    a body with that arm alone: its arms' most taken unit by unit (one arm's fp32 multiplies beside the
    other's exponentials) counted a mix neither issues (the review of lane/toolkit 4667ed46469)."""
    both = cost.issued(_counted(_uniform_branch(multiplied=True, exponentiated=True)))
    alone = max(cost.issued(_counted(_uniform_branch(multiplied=True, exponentiated=False))),
                cost.issued(_counted(_uniform_branch(multiplied=False, exponentiated=True))))
    assert both == alone, (both, alone)


def _guarded_on(predicate, operation, divisor: int, wanted: int):
    """A body scaling each word whose lane's `(tid - 64) operation divisor` compares to `wanted` by
    `predicate`."""
    def body(ctx, ref):
        del ctx
        i32 = ir.IntegerType.get_signless(32)
        shifted = arith.subi(mgpu_utils.thread_idx(), arith.constant(i32, 64))
        taken = arith.cmpi(predicate, operation(shifted, arith.constant(i32, divisor)),
                           arith.constant(i32, wanted))
        with mgpu_utils.when(taken):
            _scaled(ref)

    return body


def test_a_guard_reads_a_signed_quotient_truncated_and_a_logical_shift_unsigned():
    """A lane's integers as LLVM computes them (the review of train 1261, F7): `(tid - 64) divsi 32` is 0 on
    lanes 33 to 95, truncated toward zero (floored, 64 to 95), two warps issuing for 63 lanes; `(tid - 64)
    shrui 31` is 1 on lanes 0 to 63, their negative words read unsigned (shifted signed, -1)."""
    assert _counted(_guarded_on(arith.CmpIPredicate.eq, arith.divsi, 32, 0)) == Work(
        operations=64, shared_bytes=2 * 4 * 63)
    assert _counted(_guarded_on(arith.CmpIPredicate.eq, arith.shrui, 31, 1)) == Work(
        operations=64, shared_bytes=2 * 4 * 64)


def test_adds_under_a_predicate_and_its_negation_each_issue_on_their_warps():
    """`@p add.u32` and `@!p add.u32` into one register, `p` the first warp's: each issues on its own warps
    (one, then three), neither fused into the other as an unpredicated IADD3's adds are (the review of
    lane/toolkit a1b861d7f2d, F1: the second dropped, 32 lanes' issues counted of 128), beside the setp's
    and the immediate's move's warpgroup-wide issues."""
    def body(ctx, ref):
        del ctx, ref
        i32 = ir.IntegerType.get_signless(32)
        thread = arith.remui(mgpu_utils.thread_idx(), arith.constant(i32, 128))
        mgpu_utils.inline_ptx("""{
            .reg .pred p;
            .reg .b32 r;
            setp.lt.u32 p, $1, 32;
            mov.b32 r, 0;
            @p add.u32 r, r, 1;
            @!p add.u32 r, r, 2;
            mov.b32 $0, r;
            }""", thread, result_types=i32, has_side_effects=True)

    assert _counted(body) == Work(integer_operations=128 + 128 + 32 + 96)


def test_a_narrow_signed_quotient_reads_its_operands_at_their_width():
    """`(tid - 65)` truncated to i8 (its bits held unsigned, -65 as 191) divided signed by 2 is -32 on the
    lanes whose value is -65 or -64, the first two (the review of lane/toolkit a1b861d7f2d, F7: 191 / 2, 95,
    on none)."""
    def body(ctx, ref):
        del ctx
        i8, i32 = ir.IntegerType.get_signless(8), ir.IntegerType.get_signless(32)
        narrow = arith.trunci(i8, arith.subi(mgpu_utils.thread_idx(), arith.constant(i32, 65)))
        halved = arith.divsi(narrow, arith.constant(i8, 2))
        with mgpu_utils.when(arith.cmpi(arith.CmpIPredicate.eq, halved, arith.constant(i8, -32))):
            _scaled(ref)

    assert _counted(body) == Work(operations=32, shared_bytes=2 * 4 * 2)


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


def test_a_packed_conversion_counts_one_conversion_a_lane():
    """A conversion of a packed pair (`cvt.rn.satfinite.bf16x2.f32`, `inline_forms.satfinite_bf16`'s) is one
    F2FP a lane, as an element's conversion is: two a lane priced a pair's narrowing at twice its issues."""
    assert _counted(_ptx("cvt.rn.satfinite.bf16x2.f32")) == Work(conversions=128)


def test_a_warps_tensor_core_product_counts_its_operations_in_its_mode():
    """A TF32 matrix product a warp issues in inline PTX (mma.sync's m16n8k8, the recurrence solve's) counts
    2 M N K operations each warp, on the TF32 tensor cores."""
    assert _counted(_ptx("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32")) == Work(
        tf32_flops=2 * 16 * 8 * 8 * (128 // 32))


def test_a_store_its_operand_guards_counts_the_lanes_that_take_it():
    """A store guarded by an operand (`inline_ptx`'s predicate: the first lane's) moves that lane's word
    alone; one guarded by a predicate the block sets itself of a constant moves what that predicate's lanes
    move (none), its setp an ISETP on every lane it issues on (the review of train 1261, F1), and one of the
    data is unpriced, its lanes unknown."""
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

    # A predicate the block sets of a constant: no lane holds it, so the store moves nothing; the setp issues.
    i32 = ir.IntegerType.get_signless
    assert _counted(guarded(lambda ref: arith.constant(i32(32), 1))) == Work(integer_operations=128)

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
    moves one lane's word, the reset at most that lane's, where a block's own predicate once refused it; its
    two setps and its selp issue on the warpgroup's every lane (the review of train 1261, F1)."""
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

    assert _counted(ticket) == Work(device_bytes=8 + 4, integer_operations=3 * 128)


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


@pytest.mark.parametrize("generation", ["sm80", "sm90", "sm100", "sm120"])
def test_a_saturating_narrowing_compiles_and_counts_a_conversion_a_pair(generation):
    """An fp32 tile narrowed to bf16 saturating (`inline_forms.satfinite_bf16`, PTX's
    `cvt.rn.satfinite.bf16x2.f32` a register pair: a staged operand's parts, `Family.formed`) in the
    family's register layout compiles for each GPU on any host, and its kernel counts one conversion a pair
    and no clamp, where the clip it stands for issued two an element."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn import programs
    from dew.nn.kernels.templates import bundles, inline_forms
    from dew.nn.kernels.templates.cost import Stated

    shape = (64, 64)
    layout = bundles.of(generation, jnp.bfloat16).layout

    def body(x_ref, out_ref, narrowed_ref):
        # Through shared memory, as a staged operand lands: a thread's bf16 pair is half of a 32-byte sector.
        narrowed_ref[...] = inline_forms.satfinite_bf16(plgpu.layout_cast(x_ref[...], layout), layout)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(narrowed_ref, out_ref)
        plgpu.wait_smem_to_gmem(0)

    launch = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct(shape, jnp.bfloat16), num_threads=1,
                          thread_name="wg", scratch_types=[plgpu.SMEM(shape, jnp.bfloat16)])
    given = jax.ShapeDtypeStruct(shape, jnp.float32)
    case = KernelCase("narrowed", lambda: (launch, (given,)))
    assert not (failed := mosaic_failures(compile_mosaic_kernels([case], generation))), failed
    traced = jax.make_jaxpr(launch)(jnp.zeros(shape, jnp.float32)).jaxpr
    (kernel,) = [equation for equation in programs.equations(traced)
                 if equation.primitive in programs.NATIVE_KERNELS]
    counted = emission.launch_work(kernel, Stated(steps=0, bytes_accessed=0))
    assert (counted.conversions, counted.operations) == (shape[0] * shape[1] / 2, 0), counted


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
    moves one word (its setp an ISETP a lane, the review of train 1261, F1), and a branch on it runs one
    lane's."""
    def octal(ctx, ref):
        del ctx, ref
        i32 = ir.IntegerType.get_signless(32)
        below = arith.subi(arith.remui(mgpu_utils.thread_idx(), arith.constant(i32, 128)),
                           arith.constant(i32, 1))
        mgpu_utils.inline_ptx("{ .reg .pred p; setp.eq.u32 p, $1, 037777777777; @p st.shared.u32 [$0], $1; }",
                              arith.constant(i32, 1), below, has_side_effects=True)

    assert _counted(octal) == Work(shared_bytes=4, integer_operations=128)

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


def test_a_short_loops_trips_each_take_their_own_lanes():
    """A loop of four trips each storing the word of the lane its trip names (`thread == trip`) moves four
    words: each trip walked at its own induction value (an own band's row a trip selects), where a trip
    walked once with its induction unknown stored every lane's word each trip."""
    def body(ctx, ref):
        del ctx
        index, i32 = ir.IndexType.get(), ir.IntegerType.get_signless(32)
        lower, upper, step = (arith.constant(index, value) for value in (0, 4, 1))
        for_op = scf.ForOp(lower, upper, step)
        with ir.InsertionPoint(for_op.body):
            trip = arith.index_cast(i32, for_op.induction_variable)
            with mgpu_utils.when(arith.cmpi(arith.CmpIPredicate.eq, mgpu_utils.thread_idx(), trip)):
                memref.store(arith.constant(ir.F32Type.get(), 1.0), ref,
                             [arith.index_cast(index, mgpu_utils.thread_idx())])
            scf.YieldOp([])

    assert _counted(body).shared_bytes == 4 * 4


def test_an_address_through_chained_element_pointers_keeps_each_index_lanes():
    """A word each lane addresses through two element pointers, its own word's then a uniform index past it
    (`llvm.getelementptr` of a lane's pointer): every lane's address its own, 128 words moved, where the
    second index's lanes alone (one address) moved one."""
    def body(ctx, ref, uniform):
        del ctx
        from jax._src.lib.mlir.dialects import llvm

        f32 = ir.F32Type.get()
        lane = mgpu_utils.getelementptr(mgpu_utils.memref_ptr(ref), [mgpu_utils.thread_idx()], f32)
        llvm.load(f32, mgpu_utils.getelementptr(lane, [uniform.registers.item()], f32))

    assert _counted(body, 0).shared_bytes == 4 * 128


def test_element_pointers_of_mixed_widths_cancel_by_their_bytes_and_a_narrow_index_reads_signed():
    """Two element pointers of a lane, its word's (4 bytes) then minus four times its index in bytes (an i16
    index of -4 times the lane, `arith.trunci` of it, over bytes): every lane's address the same, one word
    moved, as the bytes cancel; read unscaled, or unsigned (-4 as 65532), the lanes stay apart."""
    def body(ctx, ref):
        del ctx
        from jax._src.lib.mlir.dialects import llvm

        f32, i8, i16 = (ir.F32Type.get(), ir.IntegerType.get_signless(8), ir.IntegerType.get_signless(16))
        thread = mgpu_utils.thread_idx()
        word = mgpu_utils.getelementptr(mgpu_utils.memref_ptr(ref), [thread], f32)
        back = arith.trunci(i16, arith.muli(thread, arith.constant(ir.IntegerType.get_signless(32), -4)))
        llvm.load(f32, mgpu_utils.getelementptr(word, [back], i8))

    assert _counted(body).shared_bytes == 4


def test_an_elements_stride_is_stated_only_where_its_layout_is_its_payload():
    """An element pointer's stride is its element's bytes only where LLVM lays it out so (a scalar of 8 to 64
    bits, a vector of a power of two of them); an i24 (padded to four bytes), i1s (packed to bits) and a
    vector of three (padded to four) state none, their lanes left unknown."""
    from dew.nn.kernels.templates import inline_work

    with ir.Context(), ir.Location.unknown():
        i32, f16 = ir.IntegerType.get_signless(32), ir.F16Type.get()
        assert inline_work._stride(i32) == 4 and inline_work._stride(ir.VectorType.get([4], f16)) == 8
        bits = ir.VectorType.get([8], ir.IntegerType.get_signless(1))
        for unknown in (ir.IntegerType.get_signless(24), bits, ir.VectorType.get([3], i32)):
            assert inline_work._stride(unknown) is None
