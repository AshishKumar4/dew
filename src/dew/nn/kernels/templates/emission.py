"""A launch's work by emission (row 5895): its kernel body's jaxpr, each equation priced in Dew's units
(`cost.Work`) as each of a program's warpgroups runs it, its roles' branches resolved by its warpgroup's index
(`handoffs.Schedule.run`'s conditionals and switch on `lax.axis_index`). A loop of trips its jaxpr states
(`scan`) counts each trip as it runs, where it stands. The outermost loop whose trips its data decides
(`while`) and that runs its walk's steps (`derivatives.WalkStep` marks, `mosaic.stepped`) counts one step of
its walk (`Role.step`), a trip's work over the steps the trip runs, which the launch's statement of its walk's
steps multiplies; any other such loop its body once. A tensor-core product counts its logical operands' shapes
(its refs' transforms applied), by its operands' precision; an exponential, a conversion and each other
operation of the CUDA cores its output's elements; a copy between memories its bytes, in device memory and in
shared memory. A primitive no rule here prices raises (`Unpriced`): the count states every instruction a
kernel emits."""

import collections
import dataclasses
import functools
import math
import weakref
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import ad_util, core as jax_core, pjit
from jax._src.lax import control_flow
from jax._src.pallas import mpmd, pallas_call, primitives as pallas
from jax._src.pallas.mosaic import primitives as tpu
from jax._src.pallas.mosaic_gpu import core as gpu_core, primitives as gpu
from jax._src.pallas.triton import core as triton_core
from jax._src.state import primitives as state, types as state_types
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import fragmented_array as fragments
from jax.experimental.mosaic.gpu.mma import MMALayouts

from dew.nn import derivatives, division, programs
from dew.nn.kernels import generation
from dew.nn.kernels.templates import inline_work, resources
from dew.nn.kernels.templates.cost import (
    ISSUED_UNITS,
    Chain,
    Stated,
    Work,
    followed,
    issued as issue_slots,
    launch_seconds,
    most,
    multiplied,
    repeated,
    stored,
)
from dew.nn.kernels.templates.product import product_mode


class Unpriced(ValueError):
    """A primitive in a kernel's body no rule of `emitted` prices."""


@dataclasses.dataclass(frozen=True)
class Emitted:
    """A program's work as its kernel body emits it (`emitted`), each of its warpgroups' apart (`roles`, by
    index): outside its walks (`fixed`), and one step of a walk (`step`)."""

    roles: tuple["Role", ...]

    @property
    def fixed(self) -> Work:
        """A program's work outside its walks, every warpgroup's."""
        return sum((role.fixed for role in self.roles), Work())

    def issued(self, unit: Callable[[Work], float], steps: float, programs: int) -> float:
        """A launch's whole work in `unit`: a step of its walk (the most a walking warpgroup's takes) its
        `steps` times, the steps every warpgroup walks (`cost.Stated.steps`), and each of its `programs`'
        work outside its walks."""
        stepping = [unit(_mean(role.walks)) for role in self.roles if role.walks]
        return max(stepping, default=0) * steps + unit(self.fixed) * programs

    def paths(self, steps: float) -> tuple[Chain, ...]:
        """A program's dependent paths: each warpgroup's work outside its walks followed by a step of its walk
        `steps` times (`cost.followed`), the warpgroups side by side (`Work.__add__`)."""
        def walked(role: Role) -> Work:
            step = tuple(chain.times(steps) for chain in _mean(role.walks).chains) if role.walks else ()
            return Work(chains=followed(role.fixed.chains, step))

        return sum(map(walked, self.roles), Work()).chains


@dataclasses.dataclass(frozen=True)
class Role:
    """One warpgroup's work as a kernel body emits it: outside its walks (`fixed`: a loop of no walk step's
    trip counted once), and one step of each walk's loop (`walks`, in the order it emits them: a walk's
    segments each a loop of its own)."""

    fixed: Work
    walks: tuple[Work, ...]

    def step(self, unit: Callable[[Work], float]) -> float | None:
        """One step of the role's walk in `unit` (a `Work` field's reading): the mean of its loops' steps
        (`walks`), none where it walks nothing. It is the walk's step wherever its loops step alike in it (a
        rotation's, `mosaic.stepping`; its products and exponentials over a causal walk's segments), and
        where a walk's parts its loops split it into take together as many of its steps' work as they mark
        (a pipelined walk's opening, steady and closing loops: the opening issuing a step ahead, the closing
        none). The trips of each loop would be a statement the count does not have: a unit a causal walk's
        masked segment takes more of (the CUDA cores' operations) is its segments' mean. A loop's guards on
        its steps' place are read at their trips (`_Place`)."""
        return unit(_mean(self.walks)) if self.walks else None


def emitted(body: jax_core.Jaxpr, warpgroups: int, axis: str | None, *, device: Sequence[int] = (),
            forms: Mapping | None = None) -> Emitted:
    """`body`, a kernel's jaxpr (its `mpmd_map`'s or its `pallas_call`'s), as each of a program's
    `warpgroups` warpgroups along `axis` runs it (`Emitted`; a program of one, no axis, a TPU's or an
    mma.sync family's). Its operands at the indices `device` are refs in device memory, each read and write
    of them its bytes' device traffic (a Triton launch's blocks, which its loads and stores move: the decode
    merge's, `merged._channel_block`). Its special functions in `forms` where given (a Mosaic GPU kernel's,
    `forms_of`), else one MUFU an element each (`operation_work`). A wait on a barrier no landing of its own
    answers waits for what another warpgroup or an earlier trip of a loop lands on it (a producer's copies, a
    pipelined walk's next step's, `_Arrivals`), its roles counted again where a first count found any (row
    4123)."""
    values = [_DEVICE if at in device else _barrier(var) for at, var in enumerate(body.invars)]
    arrivals = _Arrivals()
    laid = layouts(body) if forms in _KERNEL_FORMS.values() else {}

    def counted() -> list[Role]:
        roles = []
        for index in range(warpgroups):
            counting = _Counting(axis, index, forms=forms, arrivals=arrivals, laid=laid)
            counting.run(body, values)
            roles.append(Role(counting.held(), tuple(counting.walks)))
        return roles

    roles = counted()
    return Emitted(tuple(counted() if arrivals.collected() else roles))


_DEVICE = object()
"""A ref in device memory, as `emitted`'s walk holds it (`_Counting.equation` prices its reads and writes)."""


@dataclasses.dataclass(frozen=True)
class _Barrier:
    """A barrier array a kernel allocates (an operand of its own or its scope's, `run_scoped`), as every
    warpgroup's count holds it: one allocation however each role's branch names it, so what one warpgroup
    lands on it another's wait reads (`_Arrivals`)."""

    allocated: jax_core.Var


def _barrier(var: jax_core.Var) -> _Barrier | None:
    """`var` as a count holds it where it is a barrier array (`_Barrier`: a ref of a barrier type), None for
    any other operand."""
    aval = var.aval
    inner = aval.inner_aval if isinstance(aval, state_types.AbstractRef) else None
    if isinstance(inner, jax_core.ShapedArray) and isinstance(
            inner.dtype, gpu_core.BarrierType | gpu_core.ClusterBarrierType):
        return _Barrier(var)
    return None


class _Arrivals:
    """What a loop's trips land on each barrier a kernel allocates (`_Barrier`) that their count never waits
    on: a producer warpgroup's copies, a pipelined walk's next step's, collected as a count of the kernel's
    roles runs (`left`); and read as the next count prices a wait no landing of its own answers
    (`arriving`): its paths, each one no other outlasts (`Work.__add__`). What lands outside a loop's trip
    arrives for none, its wait a round trip as before: a program's one load onto a barrier of one slot held a
    walk's whole path to one stage, and a scan's trips, counted in one count, chained their copies into one
    arrival (M-FS-65, this head's first form re-read on its own code: a gated delta net's states launch
    priced at 1.34 of its time, KDA's weighted at 3.69)."""

    def __init__(self):
        self.landed: dict[_Barrier, tuple[Chain, ...]] = {}
        self.arrived: dict[_Barrier, tuple[Chain, ...]] = {}

    def left(self, landing: Mapping) -> None:
        """Collect what a loop's trip left unwaited (`landing`)."""
        for barrier, chain in landing.items():
            if isinstance(barrier, _Barrier):
                held = Work(chains=self.landed.get(barrier, ()))
                self.landed[barrier] = (held + Work(chains=(chain,))).chains

    def collected(self) -> bool:
        """Take what the count left as what arrives on each barrier; whether anything does."""
        self.arrived = dict(self.landed)
        return bool(self.arrived)

    def arriving(self, barrier) -> tuple[Chain, ...]:
        """The paths that arrive on `barrier`, none where the count took none."""
        return self.arrived.get(barrier, ())


_OPERATIONS = frozenset({
    lax.abs_p, lax.add_p, lax.and_p, lax.ceil_p, lax.clamp_p, lax.div_p, lax.eq_p, lax.floor_p, lax.ge_p,
    lax.gt_p, lax.le_p, lax.lt_p, lax.max_p, lax.min_p, lax.mul_p, lax.ne_p, lax.neg_p, lax.not_p, lax.or_p,
    lax.rem_p, lax.select_n_p, lax.shift_left_p, lax.shift_right_logical_p, lax.shift_right_arithmetic_p,
    lax.sign_p, lax.sub_p, lax.xor_p, lax.integer_pow_p, lax.square_p, lax.round_p,
    lax.one_minus_square_p, ad_util.add_jaxvals_p, lax.clz_p, lax.is_finite_p, lax.erf_p, lax.expm1_p,
    lax.nextafter_p, division.correctly_rounded_divide_p})
"""Elementwise operations of the CUDA cores, one an output element (a cotangent's sum, `add_any`, among
them; a leading-zero count, `clz`, an integer instruction the finish's exponent reads a subnormal's by,
`dew.nn.float_bits.binade_of`). A finiteness test is a magnitude's comparison (XLA's elemental emitter: fabs
and an fcmp), an erf XLA's own rational polynomial (xla/codegen/intrinsic/erf.cc: FMAs and a divide), and an
expm1 CUDA's expm1f, whose fp32 SASS issues FMAs alone, no MUFU (sm80, probe FS-LIBM2: ten FFMAs and three
FADDs, where an expf issues a MUFU.EX2); a float's neighbour (`nextafter`) an integer step of its bits; a
correctly rounded fp32 division (`precision.correctly_rounded_divide`) PTX's `div.rn.f32`, a division's."""

_REDUCTIONS = frozenset({lax.reduce_max_p, lax.reduce_min_p, lax.reduce_sum_p, lax.reduce_prod_p,
                         lax.reduce_and_p, lax.reduce_or_p, lax.reduce_xor_p})
"""Reductions of one operation an input element: an extreme, a sum or a product, and the bitwise and, or and
xor (a mask's all, any and parity), so a producer's row value of any of them is priced (a fused region's
prologue sides, `fused_gemm._split`). An extreme's index (argmax, argmin) carries its index beside each
step's value, a compare and two selects as lowered: no rule here, its count its lowering's when a kernel
reaches one."""

APPROXIMATED = frozenset({lax.exp_p, lax.exp2_p, lax.log_p, lax.log2_p, lax.log1p_p, lax.tanh_p,
                          lax.logistic_p, lax.sin_p, lax.cos_p, lax.sqrt_p, lax.rsqrt_p, lax.pow_p})
"""The operations Mosaic GPU computes in the hardware's approximate forms under `approx_math`
(`mosaic.compiler_params`; its lowering's exp, exp2, log, log2, tanh, sin, cos, sqrt and rsqrt; jax's
logistic, log1p and pow lower through them)."""

SPECIAL_FUNCTIONS = APPROXIMATED
"""The special-function units' operations, one an output element, the one rule Dew counts them by: those
Mosaic GPU lowers to the hardware's approximate forms, or through them (`APPROXIMATED`), square roots among
them. An fp32 sqrt and rsqrt issue one MUFU each an element on sm80, sm90 and sm100 alike, beside two FMULs
and an FSETP scaling a subnormal: Mosaic GPU's `sqrt.approx.f32` a MUFU.SQRT and `rsqrt.approx.f32` a
MUFU.RSQ, and XLA's `sqrt.rn.f32` a MUFU.RSQ refined on the FMA pipe (ptxas 13.0's SASS, probes FS-SQRT2 and
FS-SQRT4, 2026-11-16), as an exp's MUFU.EX2 is one beside its scaling FMUL."""

MOSAIC_DIVISION = Work(exponentials=1, operations=7)
"""An fp32 division in a Mosaic GPU kernel, an element: `div.full.f32`, with or without `approx_math`, its
MUFU.RCP beside five FMULs and two FSETPs (sm90's SASS, FS-DIV-SASS1 and FS-DIV-SASS5)."""

POLYNOMIAL_FORMS: Mapping = {
    lax.erf_p: Work(exponentials=1, operations=21, integer_operations=2),
    lax.atan2_p: MOSAIC_DIVISION + Work(exponentials=1, operations=30, integer_operations=3)}
"""The polynomials a Mosaic GPU kernel issues an element for an fp32 erf and atan2 (fp32 alone: an fp64 one
lowers to libdevice's `__nv_erf`, `__nv_atan2`) whatever `approx_math`
(math.erf's and math.atan2's lowering, from the PTX of a warpgroup kernel of one fp32 tile on sm90,
FS-ERF-FORMS1): an erf one ex2.approx beside 21 fp32 instructions (seven FMAs, nine selects, a multiply, a
subtraction, an absolute value, a negation and a compare) and two integer ones; an atan2 of two operands a
reciprocal's MUFU and an fp32 division (`MOSAIC_DIVISION`) beside 30 fp32 instructions and three integer
ones. Both forms tables of a kernel hold them (row 12426)."""

APPROXIMATE_FORMS: Mapping = {
    lax.exp_p: Work(exponentials=1, operations=1), lax.exp2_p: Work(exponentials=1),
    lax.log_p: Work(exponentials=1, operations=1), lax.log2_p: Work(exponentials=1),
    lax.tanh_p: Work(exponentials=1), lax.sin_p: Work(exponentials=1), lax.cos_p: Work(exponentials=1),
    lax.sqrt_p: Work(exponentials=1, operations=3), lax.rsqrt_p: Work(exponentials=1, operations=3),
    lax.logistic_p: Work(exponentials=2, operations=3), lax.div_p: MOSAIC_DIVISION, **POLYNOMIAL_FORMS}
"""Each special function's issue an element in a Mosaic GPU kernel compiled with `approx_math` (attention's):
the hardware's approximate form, one MUFU (an exp's and a log's beside the multiply scaling it to base 2; a
logistic's two, its exp and its reciprocal, beside a negation, a multiply and an add), counted from the PTX
of a warpgroup kernel of one fp32 tile on sm90 and sm100 alike (FS-APPROX-FORMS3, row 10824); a square root's
and its reciprocal's MUFU beside two FMULs and an FSETP scaling a subnormal, in their SASS (FS-SQRT2,
FS-SQRT4); an fp32 division its one form (`MOSAIC_DIVISION`). log1p and pow have none: Mosaic GPU lowers
neither."""

ACCURATE_FORMS: Mapping = {
    lax.exp_p: Work(exponentials=1, operations=7, conversions=1, integer_operations=1),
    lax.exp2_p: Work(exponentials=1),
    lax.log_p: Work(operations=21, conversions=1, integer_operations=4),
    lax.log2_p: Work(operations=23, conversions=1, integer_operations=4),
    lax.tanh_p: Work(exponentials=2, operations=13, integer_operations=3),
    lax.logistic_p: Work(exponentials=2, operations=9, conversions=1, integer_operations=1),
    lax.sqrt_p: Work(exponentials=1, operations=3), lax.rsqrt_p: Work(exponentials=1, operations=3),
    lax.div_p: MOSAIC_DIVISION, **POLYNOMIAL_FORMS}
"""Each special function's issue an element in a Mosaic GPU kernel compiled without `approx_math` (every
GEMM launch's), from the same PTX (FS-APPROX-FORMS3, sm90 and sm100): an exp the MUFU beside seven fp32
instructions (its range reduction's four FMAs, an add, a negation and a multiply), a saturating convert and a
shift; a logistic the exp's beside an add, a negation and its reciprocal's MUFU; a tanh two MUFU (an exp and
a reciprocal) beside thirteen fp32 instructions and three integer ones; a log and a log2 no MUFU at all, a
polynomial of 21 and 23 fp32 instructions (its FMAs, selects and compares) beside four integer ones and a
convert; exp2, sqrt and rsqrt their approximate forms. A sin's and a cos's accurate form branches to a slow
reduction for large arguments, which a static count misstates: unpriced, with log1p and pow, which Mosaic
GPU lowers in neither form. An fp32 division its one form (`MOSAIC_DIVISION`)."""

_F32, _BF16, _F16, _E4M3, _S32 = (np.dtype(jnp.float32), np.dtype(jnp.bfloat16), np.dtype(jnp.float16),
                                   np.dtype(jnp.float8_e4m3fn), np.dtype(jnp.int32))
CONVERSION_FORMS: Mapping = {
    (_F32, _BF16): Work(conversions=0.5), (_F32, _F16): Work(conversions=0.5),
    (_F32, _E4M3): Work(conversions=0.5),
    (_BF16, _F32): Work(integer_operations=1), (_F16, _F32): Work(operations=1),
    (_E4M3, _F32): Work(conversions=0.5, operations=1), (_F32, _S32): Work(conversions=1, operations=2),
    (_S32, _F32): Work(conversions=1), (_BF16, _F16): Work(integer_operations=1, conversions=0.5)}
"""An element type conversion's issue an element in a Mosaic GPU kernel, by its source's and its target's
exact dtypes, from sm90's SASS of a kernel of one
(FS-CONV-SASS1): a narrowing to a 16- or 8-bit float one F2FP a packed pair (`cvt.rn.bf16x2.f32`,
`cvt.rn.f16x2.f32`, `cvt.rn.satfinite.e4m3x2.f32`); a bf16 widened by its bits, a PRMT and an IMAD.U32 a
pair; an fp16 by an HADD2.F32 an element; an e4m3 unpacked to fp16 pairs (F2FP.UNPACK_B) and widened alike;
an fp32 to an int32 an F2I beside its saturating clamp's two FMNMX; an int32 to an fp32 an I2FP; a bf16 to
an fp16 through fp32. One to its own dtype nothing, anywhere (FS-CONV-SASS2); any other pair a
conversion an element, unmeasured."""

_KERNEL_FORMS: Mapping = {"approximate": APPROXIMATE_FORMS, "accurate": ACCURATE_FORMS}
"""A Mosaic GPU kernel's two forms tables by name, which a kernel's forms are by equality (a copy of one
alike), so a price cached by name (`_software_division`) reads the same table."""

IEEE_DIVISION = Work(exponentials=1, operations=6)
"""An fp32 division as PTX's `div.rn.f32` lowers on sm90, an element: on its common path a MUFU.RCP beside
five FFMAs and an FCHK, its slow path, a subnormal's or an extreme's, a call a count takes as not run (the
SASS of a hand-written kernel of one, FS-DIV-SASS5). XLA's own fp32 division is it (M-FS-77: plain LLVM
`fdiv`, PTX `div.rn.f32`)."""

XLA_FORMS: Mapping = {
    lax.exp_p: Work(exponentials=1, operations=6, integer_operations=1),
    lax.exp2_p: Work(exponentials=1, operations=7, integer_operations=1),
    lax.log_p: Work(operations=19, conversions=1, integer_operations=4),
    lax.log2_p: Work(operations=20, conversions=1, integer_operations=4),
    lax.log1p_p: Work(operations=19, conversions=1, integer_operations=6),
    lax.sqrt_p: Work(exponentials=1, operations=3), lax.rsqrt_p: Work(exponentials=1, operations=3)}
"""Each fp32 special function's issue an element in XLA's own fusions (a program's, `forms` None), from their
SASS for an H100 and an A100 alike (ptxas 13.0 of the PTX a compile-only client dumps, M-FS-78): an exp and
an exp2 libdevice's MUFU.EX2 beside its range reduction's six or seven fp32 instructions and a shift; a log,
a log2 and a log1p no MUFU at all, a polynomial of 19 or 20 fp32 instructions (its FFMAs, multiplies, selects
and compares) beside an I2F and four or six integer ones; a sqrt and an rsqrt a MUFU.RSQ beside two FMULs
and an FSETP. A tanh, a logistic, a sin, a cos and a pow branch to slow paths or call a division's, which a
static count misstates: one exponential an element, as before any was measured."""

TRITON_FORMS: Mapping = {**{function: Work(exponentials=1) for function in APPROXIMATED},
                         division.correctly_rounded_divide_p: IEEE_DIVISION}
"""A Pallas Triton kernel's forms (`forms_of`): each special function one exponential an element, as a
kernel of no measured forms; a correctly rounded fp32 division its native `div.rn.f32`
(`division.triton_lowerings`, `IEEE_DIVISION`; row 12886)."""


_FREE = frozenset({
    lax.axis_index_p, gpu.barrier_arrive_p, gpu.barrier_wait_p, lax.bitcast_convert_type_p,
    lax.broadcast_in_dim_p, gpu.commit_smem_p, gpu.commit_tmem_p, gpu.fresh_axis_index_p,
    gpu.griddepcontrol_launch_dependents_p, gpu.griddepcontrol_wait_p, lax.iota_p, gpu_core.layout_cast_p,
    lax.optimization_barrier_p, gpu.set_max_registers_p, lax.slice_p, lax.squeeze_p, lax.reshape_p,
    lax.transpose_p, gpu.tcgen05_commit_arrive_p, gpu.wgmma_wait_p, gpu.wait_smem_to_gmem_p,
    gpu.wait_load_tmem_p, gpu.wgmma_accumulator_deref_p, gpu.wgmma_accumulator_store_p, gpu.load_p,
    gpu.store_p, state.swap_p, gpu.async_load_tmem_p, gpu.async_store_tmem_p, state.get_p,
    pallas.program_id_p, tpu.dma_wait_p, gpu.wait_gmem_to_smem_p, pjit.reshard_p, lax.gather_p,
    lax.dynamic_slice_p, lax.dynamic_update_slice_p, lax.concatenate_p, lax.pad_p, lax.rev_p, lax.split_p,
    lax.stack_p, lax.tile_p, lax.unstack_p, gpu.commit_group_p, ad_util.stop_gradient_p, jax_core.pvary_p,
    jax_core.reduced_vary_cast_p, pallas.get_global_p, gpu.async_prefetch_p})
"""What moves no data between memories and computes nothing a unit's rate prices: synchronization, layouts,
a value's type cast along manual mesh axes (to vary, or from reduced: no value changes), a global
allocation's ref (`pl.get_global`: its semaphores' and flags' uses count where they move data), a prefetch
into the L2 (the copies after it read the bytes it warms and count them, the device traffic it moves early
the launch's statement's, `Stated`),
indices, a value's elements rearranged among a role's registers (a gather's, a slice's, a concatenation's;
every layout move the Pallas fuser tiles: a dynamic slice and its update, whose index arithmetic is scalar,
which nothing counts, a split, a stack, a tile and an unstack, so a fused prologue's stage holds no
operation the counter cannot price, `fused_gemm._split`), its reads and writes between its registers and
its scoped refs, which its products' operands and its copies count, and a stopped gradient, its operand."""

_PRODUCTS = frozenset({gpu.wgmma_ref_p, gpu.tcgen05_mma_p})
"""Tensor-core products whose operands' shapes their transforms state (`_product`)."""

_COPIES = frozenset({gpu.copy_gmem_to_smem_p, gpu.copy_smem_to_gmem_p})
"""Copies between device and shared memory (`_copied`)."""

_TO_TENSOR_MEMORY = frozenset({gpu.async_copy_smem_to_tmem_p, gpu.async_copy_scales_to_tmem_p})
"""Copies from shared into tensor memory, a block-scaled product's scales among them
(`_smem_to_tmem_bytes`)."""

def _surely(value) -> bool:
    """Whether a predicate the count holds as `value` is known true here (`programs.integral_scalar`)."""
    return programs.integral_scalar(value) and bool(value)


def _runs(start, stop) -> bool:
    """Whether a range from `start` to `stop`, as the count holds them, takes a trip: both known, apart."""
    return programs.integral_scalar(start) and programs.integral_scalar(stop) and int(stop) > int(start)


def _unread(work: Work) -> Work:
    """`work` with no copies' reads counted (`Work.device_reads`): a trip's that no bound every trip makes
    holds (`_Counting.loop`)."""
    return dataclasses.replace(work, device_reads=0.0)


def _folded(equation, operands: list) -> list:
    """`equation` over its concrete integer and boolean scalar operands, each admitted at its atom's dtype
    (`programs.admitted`), as its primitive computes it on the GPU (`programs.scalar_folded`): a width's wrap,
    a quotient's truncation and a remainder's sign the program's own; unknown where it computes none, a
    quotient or remainder by zero or of the least integer by -1 among them, which the GPU leaves undefined
    (LLVM's `sdiv` and `srem`, which Mosaic GPU's integer division lowers to), and where Mosaic GPU types it
    otherwise than XLA (`_typed_alike`)."""
    if not _typed_alike(equation):
        return [None]
    if equation.primitive in (lax.div_p, lax.rem_p) and _undefined(*operands):
        return [None]
    if equation.primitive in _SHIFTS and not 0 <= int(operands[1]) < jnp.iinfo(operands[0].dtype).bits:
        # Poison where Mosaic GPU computes it (MLIR's `shli`, `shrsi`, `shrui`), XLA's saturating fold apart.
        return [None]
    (out,) = equation.outvars
    return [programs.scalar_folded(equation.primitive, operands, equation.params, out.aval.dtype)]


_SHIFTS = frozenset({lax.shift_left_p, lax.shift_right_logical_p, lax.shift_right_arithmetic_p})
"""The shifts, which Mosaic GPU lowers to MLIR's own (`shli`, `shrsi`, `shrui`): poison by an amount past the
operand's width (base review bb25f261d F1)."""


def _typed_alike(equation) -> bool:
    """Whether `equation`'s operands hold one type and it states no output dtype apart from it: where Mosaic
    GPU's lowerings compute it as XLA does. Mosaic GPU types any other by rules of its own (Lane at its first
    operand's type whatever dtype it states, a weak literal materialized at its partner's), which the fold
    declines rather than restates (base reviews 117cc9088 F2 and 7e470364e F1-F3)."""
    types = {np.dtype(atom.aval.dtype) for atom in equation.invars}
    stated = equation.params.get("out_dtype")
    return len(types) <= 1 and (stated is None or {np.dtype(stated)} == types)


def _undefined(dividend: np.generic | np.ndarray, divisor: np.generic | np.ndarray) -> bool:
    """Whether an integer quotient or remainder is undefined where a Mosaic GPU kernel computes it, its
    division lowered to MLIR's `divsi` and `remsi` (LLVM's `sdiv` and `srem`) or `divui` and `remui`: by
    zero, or a signed least integer by -1. The host's XLA guards both and defines them
    (`programs.scalar_folded` folds them its way), so a kernel's fold refuses them here."""
    if divisor == 0:
        return True
    return divisor == -1 and dividend == jnp.iinfo(dividend.dtype).min


_TRANSFORMS = ("acc_transforms_tree", "a_transforms_tree", "b_transforms_tree", "barrier_transforms_tree",
               "a_scale_transforms_tree", "b_scale_transforms_tree", "a_sparse_metadata_transforms_tree")
"""A product's operands' transforms, in the order its trailing operands hold their leaves."""

_EIGHT_BIT = frozenset({jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2)})


def _elements(aval) -> int:
    return math.prod(getattr(aval, "shape", ()))


def _bytes(aval) -> int:
    return _elements(aval) * jnp.dtype(aval.dtype).itemsize


def _tensor(dtype, flops: float) -> Work:
    """A product's `flops` in the tensor cores' mode its operands' `dtype` takes."""
    dtype = jnp.dtype(dtype)
    if dtype in _EIGHT_BIT:
        return Work(fp8_flops=flops)
    if dtype == jnp.float32:
        return Work(tf32_flops=flops)
    return Work(flops=flops)


def _unreffed(aval) -> jax_core.ShapedArray:
    """A ref's value's aval (`AbstractRef.inner_aval`), a value's its own: an array's, shaped."""
    found = aval.inner_aval if isinstance(aval, state_types.AbstractRef) else aval
    if not isinstance(found, jax_core.ShapedArray):
        raise Unpriced(f"a product or copy of no array: {found}")
    return found


def _transformed(aval, tree, leaves: Sequence) -> jax_core.ShapedArray:
    """A ref's aval through its transforms (`tree` of `leaves`), applied in order by JAX's own walk of a
    transform sequence (`state_types.transform_type`, Mosaic GPU's)."""
    shaped = _unreffed(aval)
    if tree is None:
        return shaped
    return _unreffed(state_types.transform_type(jax.tree.unflatten(tree, leaves), shaped))


def _logical(aval, tree, leaves: Sequence) -> tuple[int, ...]:
    """A ref's shape as a product reads it: its aval through its transforms (`_transformed`)."""
    return tuple(_transformed(aval, tree, leaves).shape)


def _product(equation) -> Work:
    """A tensor-core product's work, `2 m n k` in its operands' mode (`_shape`)."""
    m, n, k = _shape(equation)
    return _tensor(equation.invars[1].aval.dtype, 2 * m * n * k)


def _columns(equation) -> float:
    """A product's columns as a waited one's latency counts them (`cost.Chain.columns`): its `n` columns, each
    a step of 128 bytes of its depth (a 16-bit operand 64 deep)."""
    _, n, k = _shape(equation)
    return n * k * jnp.dtype(equation.invars[1].aval.dtype).itemsize / 128


def _shape(equation) -> tuple[int, int, int]:
    """A tensor-core product's `m, n, k`: its accumulator `[m, n]` and its left operand `[m, k]` as their
    transforms read them (the trailing `*_transforms_tree` leaves of its operands)."""
    avals = [variable.aval for variable in equation.invars]
    names = sorted(name for name in equation.params if name.endswith("_transforms_tree"))
    trees = {name: equation.params[name] for name in names}
    held = sum(tree.num_leaves for tree in trees.values() if tree is not None)
    leaves = avals[len(avals) - held:]
    split, at = {}, 0
    order = [name for name in _TRANSFORMS if name in trees]
    if set(order) != set(names):
        raise Unpriced(f"{equation.primitive} names transforms {names} no rule reads")
    for name in order:
        count = 0 if trees[name] is None else trees[name].num_leaves
        split[name], at = leaves[at:at + count], at + count
    m, n = _logical(avals[0], trees.get("acc_transforms_tree"), split.get("acc_transforms_tree", ()))[-2:]
    k = _logical(avals[1], trees.get("a_transforms_tree"), split.get("a_transforms_tree", ()))[-1]
    return m, n, k


def _ring_depth(barrier: jax_core.Var) -> int:
    """The stages of the ring a copy lands in: the barriers of the array it arrives on (`barrier`, one a slot
    of a ring's), whose S - 1 slots ahead hide what of its copy a step outlasts; one where it arrives on a
    barrier of its own. Priced with none, every waited copy a ring hid read as waited whole (row 4123)."""
    return max(1, math.prod(getattr(barrier.aval, "shape", ()) or (1,)))


def _slot(leaves: Sequence) -> tuple:
    """The slot of a barrier array a copy lands on or a wait reads, as its transforms' leaves name it (each
    primitive's own, its predicate or phase none of them): each a literal's value (a captured 0-d array's as
    the number it holds) or the variable it reads, so a copy and a wait on one slot name it alike, a step's
    own slot (`i % S`) and the one it prefetches (`(i + 1) % S`) apart, each computed once."""
    return tuple(_literal(leaf.val) if isinstance(leaf, jax_core.Literal) else leaf for leaf in leaves)


def _literal(value) -> Hashable:
    """A literal index as a slot names it: the number a 0-d one holds (a captured NumPy scalar array, which
    hashes as none), an array's numbers in order."""
    held = np.asarray(value)
    return held.item() if held.ndim == 0 else tuple(held.ravel().tolist())


def _copied(equation) -> int:
    """A copy's bytes between memories: its shared-memory side's, as its transforms slice it (a ring's slot,
    a tile's rows), the operands after its source and destination its barrier (`has_barrier`), then its
    source's and its destination's transforms' leaves."""
    params, avals = equation.params, [variable.aval for variable in equation.invars]
    at = 2 + bool(params.get("has_barrier"))
    sources = params["src_transforms_treedef"].num_leaves
    leaves = {"src": avals[at:at + sources],
              "dst": avals[at + sources:at + sources + params["dst_transforms_treedef"].num_leaves]}
    side = "dst" if equation.primitive is gpu.copy_gmem_to_smem_p else "src"
    shaped = _transformed(avals[0 if side == "src" else 1], params[f"{side}_transforms_treedef"],
                          leaves[side])
    return resources.stored_bytes(shaped.shape, shaped.dtype)


def _smem_to_tmem_bytes(equation) -> int:
    """A copy into tensor memory's shared-memory read bytes (`_TO_TENSOR_MEMORY`): its SMEM source's, as its
    transforms slice it (`smem_tree` leaves after `(smem_ref, tmem_ref)`, rows 9106, 9476)."""
    params, avals = equation.params, [variable.aval for variable in equation.invars]
    smem_tree = params["smem_tree"]
    shaped = _transformed(avals[0], smem_tree, avals[2:2 + smem_tree.num_leaves])
    return resources.stored_bytes(shaped.shape, shaped.dtype)


@dataclasses.dataclass
class _Loop:
    """A loop being counted (`_Counting.loop`): its body's work a trip, and the walk steps a trip runs (its
    marks, `derivatives.WalkStep`, along the branches its warpgroup takes)."""

    work: Work = dataclasses.field(default_factory=Work)
    marks: float = 0


@dataclasses.dataclass(frozen=True)
class _Same:
    """A value the count holds by the atom that gives it where its value is unknown (a loop's range's bound
    read inside its body), so a guard comparing a step's place with it is read (`_Place`)."""

    atom: object


type _Bound = int | np.integer | _Same
"""A range's bound as the count holds it: its concrete value, else the atom that gives it (`_Same`)."""


@dataclasses.dataclass(frozen=True)
class _Place:
    """A data-decided loop's trip counter (a `fori_loop`'s) as one of its trips holds it, `plus` past it: its
    first trip (`at` "first"), its last ("last") or one between ("steady"), at its `parity`; `start` and
    `stop` its range's bounds as the count holds them. A step's guard on its place (`index == lo`,
    `index + 1 < hi`: a walk's first step opening its own slots, every step but the last reading the next's,
    `rings.Slots`) and on its parity (`index % 2`: the accumulator of two a step issues ahead into) read
    concretely at each."""

    at: str
    start: _Bound
    stop: _Bound
    parity: int
    plus: int = 0
    consulted: set[str] = dataclasses.field(default_factory=set, compare=False, repr=False)
    """What a trip's count read of it: "at" where a guard read which trip it is, "parity" where its
    remainder by two; every place of one trip (`plus` past it) records into one."""

    def compared(self, primitive, other) -> bool | int | None:
        """`self <primitive> other` where the place decides it: equal to its range's start at its first
        trip alone, one past it short of its range's stop at every trip but its last, its remainder by two
        its parity; None otherwise."""
        if primitive is lax.eq_p and self.plus == 0 and other == self.start:
            self.consulted.add("at")
            return self.at == "first"
        if primitive is lax.lt_p and self.plus == 1 and other == self.stop:
            self.consulted.add("at")
            return self.at != "last"
        if primitive is lax.rem_p and other == 2:
            self.consulted.add("parity")
            return (self.parity + self.plus) % 2
        return None


_PLACES = ("steady", "first", "last")
"""The trips a data-decided loop's guards on its place tell apart (`_Place`), its steady trip first, each
counted at either parity and their mean taken."""


class _Counting:
    """A walk of a jaxpr as warpgroup `index` along `axis` runs it: its work outside its walks (`fixed`)
    and a step of each walk (`walks`), each scalar its index's arithmetic decides held concretely; within
    `loops` loops where it starts (a conditional's branch, `cond`), and as warp `warp` (its axis and index)
    of its warpgroup's warps where it is one's (`warps`)."""

    def __init__(self, axis: str | None, index: int, loops: int = 0, warp: tuple | None = None,
                 forms: Mapping | None = None, arrivals: _Arrivals | None = None, lanes: int | None = None,
                 hoisted: dict[tuple, Work] | None = None, sites: Sequence = (),
                 expansions: dict[int, tuple] | None = None, laid: Mapping | None = None,
                 guaranteed: bool = True):
        self.axis, self.index, self.warp, self.forms = axis, index, warp, forms
        # Whether every path its data may take runs it: none within an arm of a conditional its data decides,
        # whose copies' reads count none (`Work.device_reads`, a bound on a launch's rereads).
        self.guaranteed = guaranteed
        # Static work inside a loop, by its site: computed once outside every loop, as the compiler hoists
        # it, whichever trip or arm of whichever trial reaches it (`hoist`, row 12767), so every count within
        # that loop shares it; a conditional's arm outside every loop hoists into its own, its arm's (the
        # review of lane/toolkit 55ba7911d). A site is its call path, the equations being counted within
        # (`sites`) and its own, so a cached body's two calls are two sites.
        self.hoisted = {} if hoisted is None else hoisted
        self.sites = list(sites)
        # Each expanded equation's expansion as the count first traced it (`expanded`), the equation held so
        # its identity names it for the count's length, so its equations' sites hold across visits.
        self.expansions = {} if expansions is None else expansions
        # Each value's register layout its provenance names (`layouts`, a Mosaic GPU kernel's), read where
        # an op's instructions are its layout's (a reduction's butterfly, `_butterfly`).
        self.laid = {} if laid is None else laid
        # The warp of its warpgroup whose 32 lanes run its body where a warp mesh's role runs it
        # (`plgpu.warp_map`), None where its warpgroup's 128 do.
        self.lanes = lanes
        # Whether its target's assembler fuses a multiply into the add that alone reads it (`fused`): a
        # CUDA multiprocessor's ptxas; a TPU's VPU runs them as two operations.
        self.fuses = generation.gpu_target().architecture.multiprocessor is not None
        self.arrivals = _Arrivals() if arrivals is None else arrivals
        self.fixed = Work()
        self.walks: list[Work] = []
        self.loops = [_Loop() for _ in range(loops)]
        # The products it issued and has not waited on (each one's columns, `_columns`), and what lands on
        # each barrier it will wait on: products committed to it, copies arriving on it (`waited`), by the
        # barrier's allocation where the count holds it (`_Barrier`).
        # The variables the program's constants alone decide (`run`, `_static`), and the provenance of the
        # last body's outputs it ran.
        self.static: set[jax_core.Var] = set()
        self.returned: list[bool] | None = None
        self.issued: list[float] = []
        self.landing: dict[_Barrier | jax_core.Var, Chain] = {}
        # The slots of each barrier its copies landed on, as their transforms name them (`_slot`): a wait on
        # one of them waits for a copy it issued at once, which no stage of the ring hides (row 12369).
        self.slots: dict[_Barrier | jax_core.Var, set[tuple]] = {}

    def add(self, work: Work) -> None:
        # The warpgroup's work in its order: its paths followed one by the next (`_then`).
        if self.loops:
            self.loops[-1].work = _then(self.loops[-1].work, work)
        else:
            self.fixed = _then(self.fixed, work)

    def waited(self, equation, values: list) -> None:
        """A wait the warpgroup's path runs through: products drained (`wgmma_wait`, the oldest past the `n`
        it lets fly), a barrier's phase (its committed products, its copies; else what another warpgroup or
        an earlier trip lands on it, `_Arrivals`; else another warpgroup's arrival, a round trip); what lands
        on a barrier recorded as it is committed or copied to it, by its allocation (`_Barrier`, `values`'s).
        A product waited a trip after it issued (a pipelined walk's) waits no path here: its trip's work hid
        it."""

        def barrier(at: int):
            return values[at] if isinstance(values[at], _Barrier) else equation.invars[at]

        primitive = equation.primitive
        if primitive in _PRODUCTS:
            self.issued.append(_columns(equation))
        elif primitive is gpu.wgmma_wait_p:
            drained = self.issued[:max(0, len(self.issued) - equation.params["n"])]
            self.issued = self.issued[len(drained):]
            if drained:
                self.add(Work(chains=(Chain(products=len(drained), columns=sum(drained)),)))
        elif primitive is gpu.tcgen05_commit_arrive_p:
            committed = Chain(products=len(self.issued), columns=sum(self.issued))
            self.landing[barrier(0)] = self.landing.get(barrier(0), Chain()).then(committed)
            self.issued = []
        elif primitive is gpu.copy_gmem_to_smem_p and equation.params.get("has_barrier"):
            copied = Chain(copies=1, copy_bytes=_copied(equation), stages=_ring_depth(equation.invars[2]))
            self.landing[barrier(2)] = self.landing.get(barrier(2), Chain()).then(copied)
            params = equation.params
            start = (3 + params["src_transforms_treedef"].num_leaves
                     + params["dst_transforms_treedef"].num_leaves)
            named = _slot(equation.invars[start:start + params["barrier_transforms_treedef"].num_leaves])
            self.slots.setdefault(barrier(2), set()).add(named)
        elif primitive is gpu.barrier_wait_p:
            landed = self.landing.pop(barrier(0), None)
            waited = _slot(equation.invars[1:1 + equation.params["transforms_treedef"].num_leaves])
            if landed is not None and waited in self.slots.pop(barrier(0), set()):
                # Waited on the slot it was just copied into: no turn of the ring between, nothing hidden.
                landed = dataclasses.replace(landed, stages=1)
            arrived = () if landed is not None else self.arrivals.arriving(barrier(0))
            self.add(Work(chains=(landed,) if landed is not None else arrived or (Chain(barriers=1),)))

    def run(self, jaxpr: jax_core.Jaxpr, values: list, constants: Sequence = (),
            static: Sequence[bool] = (), constant_static: Sequence[bool] = ()) -> list:
        """Count `jaxpr` given its inputs' concrete `values` (None where unknown) and which of them are
        `static` (none where not given), and its constants' (`constvars`) where its caller binds them (a
        scoped body's, `run_scoped`'s operands, `constant_static` theirs); its outputs' concrete values, and
        which of them are static (`returned`). A value is static where the program's own constants alone
        decide it (`_static`), whatever the count holds of it: a trip's place the count holds concretely is
        none (`_Place`)."""
        held = dict(zip(jaxpr.invars, values, strict=True))
        if constants:
            held.update(zip(jaxpr.constvars, constants, strict=True))
        # Each variable's provenance rewritten as it is bound, so a jaxpr run again (a call's, shared by its
        # callers) holds none of an earlier run's.
        for variables, flags in ((jaxpr.invars, static), (jaxpr.constvars, constant_static)):
            for at, variable in enumerate(variables):
                (self.static.add if at < len(flags) and flags[at] else self.static.discard)(variable)

        def read(atom):
            if isinstance(atom, jax_core.Literal):
                # A literal's value is a host array (`literals.TypedNdArray`), a scalar's of no dimension,
                # held at its own dtype as the program computes with it.
                return np.asarray(atom.val)[()] if np.ndim(atom.val) == 0 else atom.val
            return held.get(atom)

        # How often a later equation or the jaxpr's outputs read each variable: a swap's old value is live
        # where read, and a multiply an add alone reads fuses with it (`fused`).
        live = collections.Counter(atom for atoms in (*(equation.invars for equation in jaxpr.eqns),
                                                      jaxpr.outvars)
                                   for atom in atoms if isinstance(atom, jax_core.Var))
        products: set = set()
        # Each reduction's outputs by what it computes (`_computes`): one repeating it computes nothing, as
        # the lowering's CSE merges it (Mosaic GPU's `canonicalize,cse`).
        computed: dict = {}
        for equation in jaxpr.eqns:
            computes = _computes(equation)
            if computes is not None and (first := computed.get(computes)) is not None:
                held.update(zip(equation.outvars, (held.get(variable) for variable in first), strict=True))
                for variable, original in zip(equation.outvars, first, strict=True):
                    (self.static.add if original in self.static else self.static.discard)(variable)
                continue
            if computes is not None:
                computed[computes] = equation.outvars
            values = [read(atom) for atom in equation.invars]
            given = [isinstance(atom, jax_core.Literal) or atom in self.static for atom in equation.invars]
            self.returned = None
            self.sites.append(id(equation))
            try:
                outputs = self.moved(equation, values, live) or self.equation(equation, values, given)
            finally:
                self.sites.pop()
            held.update(zip(equation.outvars, outputs, strict=True))
            # A body it ran here states its outputs' provenance (a call's, a scope's, a branch's); else the
            # operation's own (`_static`).
            fixed = (self.returned if self.returned is not None and len(self.returned) == len(outputs)
                     else [_static(equation, given)] * len(outputs))
            for variable, known in zip(equation.outvars, fixed, strict=True):
                (self.static.add if known else self.static.discard)(variable)
            self.fused(equation, outputs, live, products)
        self.returned = [isinstance(atom, jax_core.Literal) or atom in self.static for atom in jaxpr.outvars]
        return [read(atom) for atom in jaxpr.outvars]

    def fused(self, equation, outputs: list, live: collections.Counter, products: set) -> None:
        """A floating multiply the count issued (its value unknown, not of the program's constants) whose
        output one equation alone reads (`live`) is kept in `products`; an add or a subtract the count issued
        reading one of its own shape, an element each, fuses with it into one instruction an element (the
        assembler's FFMA, ptxas's --fmad, its default): its operations, just counted, become adds fused
        (`Work.fused_multiply_adds`) in the same count, however many of its operands are products. A scalar
        product added to a row's every element is one multiply for many adds, none fused (the review of
        lane/toolkit 5cbbe66c586). A target whose assembler fuses none (a TPU's, `fuses`) counts both."""
        primitive = equation.primitive
        if not self.fuses or primitive not in (lax.mul_p, lax.add_p, lax.sub_p):
            return
        (output,) = equation.outvars
        if (not jnp.issubdtype(output.aval.dtype, jnp.floating) or outputs[0] is not None
                or output in self.static):
            return
        if primitive is lax.mul_p:
            if live[output] == 1:
                products.add(output)
        elif any(isinstance(atom, jax_core.Var) and atom in products
                 and getattr(atom.aval, "shape", ()) == getattr(output.aval, "shape", ())
                 for atom in equation.invars):
            added = _elements(output.aval)
            self.add(Work(operations=-added, fused_multiply_adds=added))

    def computed(self, equation) -> Work:
        """An arithmetic equation's work (`operation_work`), a reduction's beside it its exchange across the
        lanes its operand's layout spreads it over where its provenance names that layout (`_butterfly`)."""
        layout = self.laid.get(equation.invars[0]) if equation.primitive in _REDUCTIONS else None
        if isinstance(layout, fragments.TiledLayout):
            threads = inline_work.WARP if self.lanes is not None else resources.THREADS
            reduced = _laid_reduction(layout, equation.invars[0].aval, equation.params["axes"], threads,
                                      equation.primitive, generation.lowering_capability())
            if reduced is not None:
                return reduced
        return operation_work(equation.primitive, [variable.aval for variable in equation.invars],
                              [variable.aval for variable in equation.outvars], equation.params,
                              forms=self.forms)

    def moved(self, equation, values: list, live: collections.Counter) -> list | None:
        """A read or write of a ref in device memory (`emitted`'s `device`) counted as the bytes it moves,
        its outputs unknown; None for any other equation. A read and a swap read the old value where it is
        `live` (a later equation or the jaxpr's outputs read it: a swap a load and a store, as Triton lowers
        it), a swap a store alone where it is dropped."""
        primitive = equation.primitive
        if primitive not in (state.get_p, state.swap_p) or not values or values[0] is not _DEVICE:
            return None
        old = equation.outvars[0]
        read = _bytes(old.aval) if old in live else 0
        written = _bytes(equation.invars[1].aval) if primitive is state.swap_p else 0
        self.add(Work(device_bytes=read + written))
        return [None] * len(equation.outvars)

    def equation(self, equation, values: list, given: list[bool]) -> list:
        primitive, params = equation.primitive, equation.params
        unknown = [None] * len(equation.outvars)
        self.waited(equation, values)
        if primitive is lax.axis_index_p:
            return [self.place(params["axis_name"])]
        if (primitive in programs.SCALAR_FOLDED and values and (operands := programs.admitted(
                values, [atom.aval.dtype for atom in equation.invars])) is not None):
            if primitive is lax.convert_element_type_p:
                self.add(operation_work(primitive, [atom.aval for atom in equation.invars],
                                        [atom.aval for atom in equation.outvars], params, forms=self.forms))
            return _folded(equation, operands)
        if (placed := _placed(primitive, values)) is not None:
            return [placed]
        if primitive is control_flow.platform_index_p:
            # The platform a program lowers for, its compile's choice: the priced target's, so a platform's
            # branch (`lax.platform_dependent`) counts as its own runs.
            return [_platform_branch(params["platforms"])]
        if primitive is pallas.multiple_of_p:
            # A divisibility hint: its operand's value, nothing computed.
            return [values[0]]
        if (primitive is lax.select_n_p and programs.integral_scalar(values[0])
                and all(value is not None for value in values[1:])):
            return [values[1 + int(values[0])]]
        if _static(equation, given):
            # Of the program's constants alone: hoisted out of every loop it lies in, computed once (`hoist`),
            # else where its path runs it. A call of them runs on, its body's folds the values a later guard
            # reads.
            if primitive in _ARITHMETIC:
                self.hoist(operation_work(primitive, [variable.aval for variable in equation.invars],
                                          [variable.aval for variable in equation.outvars], params,
                                          forms=self.forms))
            return unknown
        if primitive in _ARITHMETIC:
            self.add(self.computed(equation))
        elif primitive in _PRODUCTS:
            self.add(_product(equation))
        elif primitive is gpu.mma_p:
            acc, lhs = equation.invars[0].aval, equation.invars[1].aval
            self.add(_tensor(lhs.dtype, 2 * acc.shape[0] * acc.shape[1] * lhs.shape[1]))
        elif primitive in _COPIES:
            copied = _copied(equation)
            # A copy its own predicate guards reads where the predicate holds: surely, where it is known true.
            surely = not params.get("has_user_predicate") or _surely(values[-1])
            read = copied if primitive is gpu.copy_gmem_to_smem_p and self.guaranteed and surely else 0
            self.add(Work(device_bytes=copied, shared_bytes=copied, device_reads=read))
        elif primitive in _TO_TENSOR_MEMORY:
            self.add(Work(shared_bytes=_smem_to_tmem_bytes(equation)))
        elif primitive is tpu.dma_start_p:
            # A TPU core's copy between HBM and VMEM: its destination's bytes as its transforms read it.
            copied = _bytes(params["tree"].unflatten([variable.aval for variable in equation.invars])[1])
            self.add(Work(device_bytes=copied, shared_bytes=copied))
        elif primitive is gpu.atomic_store_p:
            # An atomic update (an add into a sum other programs or roles add to): its value's bytes, in
            # device memory each lane's reduced at L2 an element at a time (`red.global`); in shared memory
            # (`red.shared::cta`), shared bytes alone.
            ref, _, value = params["args_tree"].unflatten([variable.aval for variable in equation.invars])
            moved = _bytes(value)
            if isinstance(ref, state_types.AbstractRef) and ref.memory_space == gpu_core.SMEM:
                self.add(Work(shared_bytes=moved))
            else:
                self.add(Work(device_bytes=moved, global_reductions=moved))
        elif primitive is lax.dot_general_p:
            # A TPU core's product on its MXU: `2 m n k`, its output's elements over its contracted ones,
            # one pass in its operands' mode. Not `product_mode`'s passes: Mosaic TPU lowers a body's product
            # at the default precision with no contract precision, its own default, HIGH as `bf16x3` and
            # HIGHEST as `fp32` (jax/_src/pallas/mosaic/lowering.py, `_dot_general_lowering_rule`, at the
            # pin), and the passes its default and its fp32 take are unmeasured, where XLA states its own
            # (`TPU_PASSES`).
            (contracted, _), _ = params["dimension_numbers"]
            lhs = equation.invars[0].aval
            self.add(_tensor(lhs.dtype, 2 * _elements(equation.outvars[0].aval)
                             * math.prod(lhs.shape[dimension] for dimension in contracted)))
        elif primitive is gpu.inline_mgpu_p:
            self.add(_inlined(params, equation.invars, values, self.lanes))
        elif primitive is lax.cond_p:
            return self.cond(params["branches"], values, given)
        elif (region := _REGIONS.get(primitive)) is not None:
            return region(self, equation, values, given) or unknown
        elif primitive is derivatives.marked_call_p and isinstance(params["kind"], _MARKS):
            return self.marked(params, values, given)
        elif (expanded := self.expanded(equation)) is not None:
            # The expansion's constants are its trace's own.
            inner, constants, taken = expanded
            return self.run(inner, [values[at] for at in taken], constants, [given[at] for at in taken],
                            [True] * len(constants))
        elif primitive not in _FREE:
            raise Unpriced(f"no rule prices {primitive} in a kernel's body")
        return unknown

    def marked(self, params, values: list, given: list[bool]) -> list:
        """A marked span a count reads by its kind: a rare span's verdict as its common value, False
        (`derivatives.Rare`), a walk's step counted by its loop and run (`derivatives.WalkStep`)."""
        if isinstance(params["kind"], derivatives.Rare):
            return [np.False_]
        if self.loops:
            self.loops[-1].marks += 1
        return self.run(params["call_jaxpr"], values, static=given)

    def place(self, name) -> int | None:
        """The concrete index along axis `name`: its warp's among its warpgroup's warps (`warp`), its
        warpgroup's along `axis`; None along any other."""
        found = dict([(self.axis, self.index), *([self.warp] if self.warp is not None else [])])
        return next((index for axis, index in found.items() if name in (axis, (axis,))), None)

    def scoped(self, equation, values: list, given: list[bool]) -> list:
        """A scope (`run_scoped`): its body's leading inputs its operands, the rest the refs it allocates."""
        body = equation.params["jaxpr"]
        allocated = body.invars[len(values):]
        return self.run(body, [*values, *map(_barrier, allocated)], static=given)

    def loop(self, equation, values: list, given: list[bool]) -> None:
        """A loop whose trips its data decides (a while's): the outermost such loop that runs walk steps
        (`derivatives.WalkStep` marks along the branches its warpgroup takes) a walk, its steady trip's work
        over its marks one step of it (`walks`), which the launch's statement of its steps multiplies; any
        other its trip counted once where it stands, its marks its enclosing loop's. A walk over a range (a
        `fori_loop`'s, `_ranged`) whose steps guard on their place (`_Place`) is counted at its steady trip,
        its first and last trips' difference from it once where it stands."""
        params = equation.params
        held = [value if value is not None else _Same(atom) for atom, value in zip(equation.invars, values,
                                                                                   strict=True)]
        body, consts = params["body_jaxpr"], held[params["cond_nconsts"]:][:params["body_nconsts"]]
        carry = len(body.invars) - len(consts)
        # Its constants keep their provenance in every trip; a carried value is none past the first.
        static = [*given[params["cond_nconsts"]:][:params["body_nconsts"]], *([False] * carry)]
        ranged = _ranged(params, held)
        consulted: set[str] = set()

        def trip(at: str, parity: int) -> "_Counting":
            # A carried value is unknown past the first trip, but a range's counter and stop (`_ranged`).
            inputs = [*consts, *([None] * carry)]
            if ranged is not None:
                counter, start, stop, bound = ranged
                inputs[len(consts) + counter] = _Place(at, start, stop, parity, consulted=consulted)
                inputs[len(consts) + bound] = stop
            trial = _Counting(self.axis, self.index, loops=len(self.loops) + 1, warp=self.warp,
                              forms=self.forms, arrivals=self.arrivals, lanes=self.lanes,
                              hoisted=self.hoisted, sites=self.sites, expansions=self.expansions,
                              laid=self.laid, guaranteed=self.guaranteed)
            trial.run(body, inputs, static=static)
            self.arrivals.left(trial.landing)
            return trial

        # A trip whose count read nothing of its place (`_Place.consulted`) is every trip: its count holds the
        # same values and shared state (layouts fixed before the count, a landing kept once), so one at
        # another place or parity retraces it to the bit; where it read its parity alone, so does one at the
        # other trips' places.
        steady = trip("steady", 0)
        odd = trip("steady", 1) if "parity" in consulted else steady
        # Its reads a bound every run makes (`Work.device_reads`): a trip that read its place or parity reads
        # as they decide, one trip may be every place at once, and a loop no walk marks counts its one trip
        # where it stands, which may never run where its data decides its trips: such a loop's count none.
        marked = steady.loops[-1].marks > 0
        unread = bool(consulted) or not (marked or (ranged is not None and _runs(*ranged[1:3])))
        self.fixed = self.fixed + steady.fixed
        self.walks.extend(_unread(walk) if unread else walk for walk in steady.walks)
        work = _mean([steady.loops[-1].work, odd.loops[-1].work])
        found = _Loop(_unread(work) if unread else work, steady.loops[-1].marks)
        self.ended(found, walks=True)
        if ranged is not None and found.marks and not self.loops:
            # The first and last trips each take their own work in place of a steady one's.
            for at in _PLACES[1:]:
                taken = (_mean([trip(at, parity).loops[-1].work for parity in (0, 1)]) if "at" in consulted
                         else found.work)
                self.fixed = self.fixed + (_unread(taken) if unread else taken) + repeated(found.work, -1)

    def warps(self, equation, values: list, given: list[bool]) -> None:
        """A warpgroup's warps each running its role (`plgpu.warp_map`'s `mpmd_map`: tcgen05's load and
        product warps): each warp's body counted with its warp's index concrete, so its role's branch is the
        one it takes; the warpgroup's work theirs together, a step of its walk each warp's step summed, and
        the steps its warps take together its enclosing loop's (the most any warp marks). The barriers its
        body takes are allocations every warp's count holds alike (`_Barrier`)."""
        del values, given
        (mesh,), (inner,) = equation.params["meshes"], equation.params["jaxprs"]
        ((name, count),) = mesh.shape.items()
        steps, marks = [], 0.0
        barriers = [_barrier(var) for var in inner.invars]
        for warp in range(count):
            # Each warp's static work its own (its instructions), its sites marked by its warp.
            counted = _Counting(self.axis, self.index, loops=len(self.loops), warp=(name, warp),
                                forms=self.forms, arrivals=self.arrivals,
                                lanes=warp if isinstance(mesh, gpu_core.WarpMesh) else self.lanes,
                                hoisted=self.hoisted, sites=[*self.sites, (name, warp)],
                                expansions=self.expansions, laid=self.laid, guaranteed=self.guaranteed)
            counted.run(inner, barriers)
            if self.loops:
                self.arrivals.left(counted.landing)
            self.fixed = self.fixed + counted.fixed
            if counted.walks:
                steps.append(_mean(counted.walks))
            if self.loops:
                self.loops[-1].work = self.loops[-1].work + counted.loops[-1].work
                marks = max(marks, counted.loops[-1].marks)
        if steps:
            self.walks.append(sum(steps, Work()))
        if self.loops:
            self.loops[-1].marks += marks

    def scanned(self, equation, values: list, given: list[bool]) -> list:
        """A loop of the trips its jaxpr states (a scan's), each trip counted as it runs, its carry's
        concrete values carried to the next (a walk's step index, so a step's guard on it takes its
        branch): its work where it stands, its marks its enclosing loop's; its outputs' concrete carry."""
        params = equation.params
        body, length = params["jaxpr"], params["length"]
        consts, carried, xs = map(list, params["ft_in"].update(values).unpack())
        # Its constants keep their provenance; its carry, which the count holds concretely trip by trip, has
        # none.
        static = [*given[:len(consts)], *([False] * (len(carried) + len(xs)))]
        self.loops.append(_Loop())
        for _ in range(length):
            carried = self.run(body, [*consts, *carried, *([None] * len(xs))], static=static)[:len(carried)]
        self.ended(self.loops.pop(), walks=False)
        return [*carried, *([None] * (len(body.outvars) - len(carried)))]

    def ended(self, found: _Loop, *, walks: bool) -> None:
        """A loop counted (`found`): its work and marks its enclosing loop's, else a walk where it `walks`
        steps, else the program's own work."""
        if self.loops:
            self.loops[-1].work = self.loops[-1].work + found.work
            self.loops[-1].marks += found.marks
        elif walks and found.marks:
            self.walks.append(repeated(found.work, 1 / found.marks))
        else:
            self.fixed = self.fixed + found.work

    def held(self) -> Work:
        """Its work outside its walks and what it hoisted out of its loops (`hoisted`), together."""
        total = self.fixed
        for work in self.hoisted.values():
            total = total + work
        return total

    def expanded(self, equation) -> tuple[jax_core.Jaxpr, Sequence, Sequence[int]] | None:
        """`equation`'s expansion (`_expanded`), traced once for the count: a loop's trips and trials visit
        one, so its equations' sites (`hoist`) are one across them (the review of lane/toolkit 55ba7911d)."""
        if id(equation) not in self.expansions:
            self.expansions[id(equation)] = (equation, _expanded(equation))
        return self.expansions[id(equation)][1]

    def hoist(self, work: Work) -> None:
        """Static work (of the program's constants alone) at the equation being counted: inside a loop, its
        site's once (`hoisted`, the compiler's hoisting out of every loop, whatever trip or arm reaches it
        and however often); outside every loop, its path's own, beside the rest of its work (under a
        conditional's arms' envelope, a call's at each call)."""
        if not self.loops:
            self.fixed = _then(self.fixed, work)
        else:
            self.hoisted.setdefault(tuple(self.sites), work)

    def cond(self, branches, values: list, given: list[bool]) -> list:
        """A conditional: its branch where its index is concrete (a role's, by its warpgroup), else every
        branch's envelope, unit by unit (the most of each any branch issues, as the custom call's FLOP count
        takes a conditional), its steps' marks likewise; the walks its branches hold one walk, a step of it
        the most of each unit any branch's walks take a step (their mean, `Role.step`)."""
        index, operands = values[0], values[1:]
        if programs.integral_scalar(index):
            taken = self.run(branches[int(index)], operands, static=given[1:])
            if not given[0]:
                # Its selector a value of a trip or a thread: which branch runs varies, so its outputs do.
                self.returned = None
            return taken
        trials = []
        for branch in branches:
            # Inside a loop its arms' hoisted work leaves the loop with every trip's (shared); outside every
            # loop an arm's stays in the arm (its own), under the arms' envelope with the arm's other work.
            trial = _Counting(self.axis, self.index, loops=len(self.loops), warp=self.warp,
                              forms=self.forms, arrivals=self.arrivals, lanes=self.lanes,
                              hoisted=self.hoisted if self.loops else {}, sites=self.sites,
                              expansions=self.expansions, laid=self.laid, guaranteed=False)
            trial.run(branch, operands, static=given[1:])
            if self.loops:
                self.arrivals.left(trial.landing)
            trials.append(trial)
        self.fixed = self.fixed + most([trial.fixed if self.loops else trial.held() for trial in trials])
        if any(trial.walks for trial in trials):
            self.walks.append(most([_mean(trial.walks) for trial in trials if trial.walks]))
        if self.loops:
            self.loops[-1].work = self.loops[-1].work + most([trial.loops[-1].work for trial in trials])
            self.loops[-1].marks += max(trial.loops[-1].marks for trial in trials)
        return [None] * len(branches[0].outvars)


def _inlined(params: Mapping, invars: Sequence, values: Sequence, warp: int | None) -> Work:
    """An inlined Mosaic GPU body's work: what the code it emits issues (`inline_work.body_work`), its
    arguments its operands' leading ones, their refs' transforms' leaves after them, its scalars' values
    where the count holds them."""
    leading = params["pytree_args"].num_leaves
    # Each ref argument viewed as its body takes it: its view's transforms (a slot's index, a tile's
    # untiling) applied, as the lowering applies them (`_inline_mgpu_flat_transformed_args`).
    viewed = params["pytree_ref_transforms"].unflatten([variable.aval for variable in invars[leading:]])
    avals = [functools.reduce(lambda aval, transform: transform.transform_type(aval), transforms or (),
                              variable.aval)
             for variable, transforms in zip(invars[:leading], viewed, strict=True)]
    try:
        return inline_work.body_work(params["mgpu_fn"], avals, params["flat_arg_types"], values[:leading],
                                     warp)
    except inline_work.Uncounted as error:
        raise Unpriced(f"an inlined Mosaic GPU body ({params['mgpu_fn']}) emits what no rule counts: "
                       f"{error}") from error


def _ranged(params, values: list) -> tuple[int, _Bound, _Bound, int] | None:
    """A while loop over a range (`fori_loop`'s: its condition `counter < stop` of two carried values, the
    stop carried through unchanged): its counter's place in its carry, its range's start and stop as the count
    holds them (`_Bound`), and its stop's place; None for any other loop."""
    condition = params["cond_jaxpr"]
    if len(condition.eqns) != 1 or condition.eqns[0].primitive is not lax.lt_p:
        return None
    carry = condition.invars[params["cond_nconsts"]:]
    counter, stop = condition.eqns[0].invars
    if counter not in carry or stop not in carry:
        return None
    carried = values[params["cond_nconsts"] + params["body_nconsts"]:]
    return carry.index(counter), carried[carry.index(counter)], carried[carry.index(stop)], carry.index(stop)


def _placed(primitive, values: list):
    """A step's place read where `primitive` reads it (`_Place`): one past it (`counter + 1`), or a guard
    its place decides; None where it reads none."""
    if len(values) != 2 or not any(isinstance(value, _Place) for value in values):
        return None
    place, other = values if isinstance(values[0], _Place) else values[::-1]
    if not (programs.integral_scalar(other) or isinstance(other, _Same)):
        return None
    if primitive is lax.add_p and other == 1:
        return dataclasses.replace(place, plus=place.plus + 1)
    return place.compared(primitive, other) if place is values[0] else None


_MARKS = (derivatives.Rare, derivatives.WalkStep)
"""The marked spans a count reads by their kind (`_Counting.marked`)."""

_REGIONS = {pallas.run_scoped_p: _Counting.scoped, lax.while_p: _Counting.loop, lax.scan_p: _Counting.scanned,
            mpmd.mpmd_map_p: _Counting.warps}
"""The regions a kernel's body counts within it, each by its method: its outputs' concrete values, or none
where it leaves them unknown."""


def _expanded(equation) -> tuple[jax_core.Jaxpr, Sequence, Sequence[int]] | None:
    """The program an equation runs as it lowers, where it is one of JAX's primitives' own, its constants,
    and the equation's operands it takes, in order: a call's (`programs.called`); a high-level operation's
    expansion as JAX lowers it (`programs.high_expansion`: a declared tangent's, a contraction whose
    transpose it states), its literal operands its own values; the construct's own operation's value (the
    algebra's law, `scaled.primitive`, as a kernel's lowering runs it: the decode merge's, `merged`). None
    for any other."""
    # The law's module read here: `dew.algebra` reads this module's counts as it is imported (`cost`).
    from dew.algebra import scaled

    every = range(len(equation.invars))
    if (inner := programs.called(equation)) is not None:
        return inner, (), every
    if (expansion := programs.high_expansion(equation)) is not None:
        return expansion, expansion.consts, [at for at in every
                                             if not isinstance(equation.invars[at], jax_core.Literal)]
    if (made := scaled.MADE.get(equation.primitive)) is None:
        return None
    traced = jax.make_jaxpr(functools.partial(made[0], **equation.params))(*(v.aval for v in equation.invars))
    return traced.jaxpr, traced.consts, every


def _platform_branch(platforms) -> int:
    """The branch of a `lax.platform_dependent` whose platforms (`platform_index`'s `platforms`, a branch's
    names, or None for its default) hold the platform the call lowers for (`generation.lowering_platform`),
    as JAX keeps it (`programs.platform_branch`); unpriced where its facts do not name one."""
    if (name := generation.lowering_platform()) is None:
        raise Unpriced(f"a platform choice among {platforms} on a GPU whose platform its facts do not name")
    if not any(names is None or name in names for names in platforms):
        raise Unpriced(f"no branch of a platform choice among {platforms} runs on {name}")
    return programs.platform_branch(platforms, name)


@functools.lru_cache(maxsize=64)
def _software_division(operands: tuple, target: generation.GpuTarget, forms: str | None) -> Work:
    """A correctly rounded division's software form (`division.software_divide`, XLA's and Mosaic
    GPU's lowering) over operands of `(shape, dtype)` `operands`, traced for `target` (its platform's
    arithmetic chosen as its program's is) and counted as a kernel's body is, through its protected and
    marked calls, with a Mosaic GPU kernel's `forms` (by name) or none (row 12886); once an operand set."""
    with generation.assumed(target):
        traced = jax.make_jaxpr(division.software_divide)(
            *(jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in operands))
        counted = emitted(traced.jaxpr, 1, None, forms=_KERNEL_FORMS.get(forms or "")).roles[0]
    return counted.fixed + sum(counted.walks, Work())


def layouts(jaxpr) -> dict:
    """Each array value of a Mosaic GPU kernel's body (`jaxpr`, its nested bodies' too) whose register layout
    its provenance names, as Mosaic GPU's layout inference assigns it (row 12889): a warpgroup product's
    accumulator (WGMMA's, `wgmma_accumulator_deref`; mma.sync's, `mma`'s by its split), a stated layout
    (`layout_cast`), an elementwise op's operands' one layout, a reduction's its operand's reduced
    (`TiledLayout.reduce`), through scoped bodies, calls, a conditional's agreeing arms, a loop's carries
    its trips keep and any call's program (`programs.called`). A value of any other provenance
    is not among them: its layout is inference's choice from its uses, which no rule here reads."""
    found: dict = {}
    _laid_out(jaxpr, [None] * len(jaxpr.invars), found)
    return found



def _laid_out(jaxpr, given: Sequence, found: dict) -> list:
    """The layouts of `jaxpr`'s outputs, its inputs' `given` (its constants' none), each value's noted in
    `found` (a value two visits lay out apart, a body's two calls', none)."""
    known: dict = {}

    def note(var, layout) -> None:
        # Each visit's layout of `var` joins its others': one visit's unknown or another layout, none.
        if isinstance(var, jax_core.DropVar):
            return
        found[var] = layout if found.get(var, layout) == layout else None
        if layout is not None:
            known[var] = layout

    def of(atom):
        return known.get(atom) if isinstance(atom, jax_core.Var) else None

    for var, layout in [*zip(jaxpr.invars, given, strict=True), *((var, None) for var in jaxpr.constvars)]:
        note(var, layout)
    for equation in jaxpr.eqns:
        laid = _equation_layouts(equation, [of(atom) for atom in equation.invars], found)
        for var, layout in zip(equation.outvars, laid, strict=True):
            note(var, layout)
    return [of(atom) for atom in jaxpr.outvars]


def _equation_layouts(equation, given: list, found: dict) -> list:
    """`equation`'s outputs' layouts (`_laid_out`'s rules), its operands' `given`."""
    primitive, params, outs = equation.primitive, equation.params, equation.outvars
    unknown = [None] * len(outs)
    if primitive is gpu_core.layout_cast_p:
        # As its lowering reads it (`new_layout.to_mgpu()`).
        return [params["new_layout"].to_mgpu()]
    if primitive is gpu.wgmma_accumulator_deref_p:
        return [fragments.WGMMA_LAYOUT]
    if primitive is gpu.mma_p:
        m, n = outs[0].aval.shape
        held = params.get("m_warps")
        split = (MMALayouts(outs[0].aval.dtype, m_warps=held) if held is not None
                 else MMALayouts.for_shape(outs[0].aval.dtype, m, n))
        return [split.acc]
    if primitive in _REDUCTIONS:
        layout, shape = given[0], tuple(equation.invars[0].aval.shape)
        if not isinstance(layout, fragments.TiledLayout):
            return unknown
        untiled = len(shape) - len(layout.base_tile_shape)
        axes = tuple(params["axes"])
        if any(axis < untiled for axis in axes) or len(axes) == len(shape):
            # A reduction to a scalar holds no tiled layout.
            return unknown
        return [layout.reduce(tuple(axis - untiled for axis in axes))]
    if primitive in _OPERATIONS or primitive in SPECIAL_FUNCTIONS or primitive in (
            lax.convert_element_type_p, lax.select_n_p, lax.integer_pow_p):
        shape = tuple(getattr(outs[0].aval, "shape", ()))
        held = {layout for layout, atom in zip(given, equation.invars, strict=True)
                if layout is not None and tuple(getattr(atom.aval, "shape", ())) == shape}
        return [next(iter(held))] if len(held) == 1 else unknown
    if primitive is pallas.run_scoped_p:
        # Its body's leading inputs its operands (`pallas.run_scoped`), the rest the refs it allocates.
        body = params["jaxpr"]
        return _laid_out(body, [*given, *[None] * (len(body.invars) - len(given))], found)
    if primitive is mpmd.mpmd_map_p:
        # A warp mesh's or a kernel's regions, each its own body over its refs.
        for region in params["jaxprs"]:
            _laid_out(region, [None] * len(region.invars), found)
        return unknown
    if primitive is lax.cond_p:
        arms = [_laid_out(branch.jaxpr, given[1:], found) for branch in params["branches"]]
        return [first if all(arm[index] == first for arm in arms) else None
                for index, first in enumerate(arms[0])]
    if primitive is lax.while_p:
        consts = params["cond_nconsts"] + params["body_nconsts"]
        carried = list(given[consts:])
        body_consts = given[params["cond_nconsts"]:consts]
        # Each pass keeps a carry's layout its trip keeps (a carry only loses one), so passes stop by the
        # carries' count; each pass's facts join the others' (`_laid_out`), so the fixed point's stand.
        while True:
            out = _laid_out(params["body_jaxpr"].jaxpr, [*body_consts, *carried], found)
            kept = [before if before == after else None for before, after in zip(carried, out, strict=True)]
            if kept == carried:
                return carried
            carried = kept
    if primitive is lax.scan_p:
        consts, carried, xs = map(list, params["ft_in"].update(given).unpack())
        while True:
            out = _laid_out(params["jaxpr"].jaxpr, [*consts, *carried, *([None] * len(xs))], found)
            kept = [before if before == after else None
                    for before, after in zip(carried, out[:len(carried)], strict=True)]
            if kept == carried:
                return carried + [None] * (len(outs) - len(carried))
            carried = kept
    if (inner := programs.called(equation)) is not None:
        # A call (a jit's, a custom rule's, a custom batching's): its program on its operands one for one.
        return _laid_out(inner, given, found)
    return unknown


_REDUCTION_PARAMS = ("axes", "accumulator_ilp", "out_sharding")
"""The parameters a reduction binds (its lowering's: `accumulator_ilp` its grouping), its merge key's."""


def _computes(equation) -> tuple | None:
    """What a reduction of a variable computes, a key two reductions of one body computing the same share:
    its primitive, its operand (the variable itself) and each parameter it binds; None for any other
    equation, or a reduction binding one beyond `_REDUCTION_PARAMS`. Only reductions merge here: a repeated
    product's readers decide its fusion (`_Counting.fused`), which a merge would have to rewrite."""
    if (equation.primitive not in _REDUCTIONS or not isinstance(equation.invars[0], jax_core.Var)
            or not set(equation.params) <= set(_REDUCTION_PARAMS)):
        return None
    return (equation.primitive, equation.invars[0],
            *(equation.params.get(name) for name in _REDUCTION_PARAMS))


def _redux(primitive, dtype, capability: tuple[int, int] | None) -> bool:
    """Whether Mosaic GPU lowers a reduction across all of a warp's lanes by `redux.sync`, no shuffle
    (`FragmentedArray.reduce`'s `redux_op`): a 32-bit integer's sum, maximum or minimum, and an fp32 maximum
    or minimum where the kernel lowers for compute `capability` 10.x (a datacenter Blackwell's, which alone
    has the float form, `generation.lowering_capability`; a Hopper's shuffles five rounds)."""
    if np.dtype(dtype).itemsize == 4 and np.issubdtype(dtype, np.integer):
        return primitive in (lax.reduce_sum_p, lax.reduce_max_p, lax.reduce_min_p)
    blackwell = capability is not None and capability[0] == 10
    return blackwell and np.dtype(dtype) == np.float32 and primitive in (lax.reduce_max_p, lax.reduce_min_p)


def _laid_reduction(layout: fragments.TiledLayout, value, axes: Sequence[int], threads: int,
                    primitive, capability: tuple[int, int] | None) -> Work | None:
    """A reduction of `value`, laid out as `layout`, along `axes`, as Mosaic GPU lowers it
    (`FragmentedArray.reduce`): each thread combines the values its registers hold of each output it keeps
    (a replicated layout's copies each thread's own), then
    exchanges them across the lanes the layout spreads `axes` over, log2 of them rounds of a shuffle of each
    32-bit word and a combine of each value, on each of the role's `threads`; None where the layout states
    no such lowering here (a scalar's, an axis no tile holds, an exchange across warps, one across all of a
    warp's lanes `redux.sync` may take, `_redux`). The attention
    forward's walk issues 12 SHFL a thread a step for its three row reductions of a WGMMA (sm90) or mma.sync
    (sm80) tile, two rows a thread and two rounds each (FS-REDSHFL4)."""
    shape = tuple(value.shape)
    untiled = len(shape) - len(layout.base_tile_shape)
    kept = tuple(size for axis, size in enumerate(shape) if axis not in axes)
    if not kept or any(axis < untiled for axis in axes):
        return None
    reduced = layout.reduce(tuple(axis - untiled for axis in axes))

    def spread(dims) -> int:
        return math.prod(dim.times for dim in dims if isinstance(dim, fragments.Replicated))

    if spread(reduced.warp_dims) > spread(layout.warp_dims):
        return None
    # As the lowering decides: every lane dimension one the reduction collapses (`all_lanes`).
    every_lane = all(isinstance(target, fragments.Replicated) and not isinstance(source, fragments.Replicated)
                     for source, target in zip(layout.lane_dims, reduced.lane_dims, strict=True))
    if every_lane and _redux(primitive, value.dtype, capability):
        return None
    rounds = int(math.log2(spread(reduced.lane_dims) // spread(layout.lane_dims)))
    # A thread's values in and out as its registers hold them (a replicated layout's every copy's).
    given = math.prod(layout.registers_shape(shape)) * layout.vector_length
    kept_values = math.prod(reduced.registers_shape(kept)) * reduced.vector_length
    words = math.prod(reduced.registers_shape(kept)) * math.ceil(
        reduced.vector_length * np.dtype(value.dtype).itemsize / 4)
    return Work(operations=(given - kept_values + rounds * kept_values) * threads,
                shuffles=rounds * words * threads)


def _fp32_on_cuda(outputs: Sequence, target: generation.GpuTarget | None, fp32: bool | None = None) -> bool:
    """Whether an operation of results `outputs` is an fp32 one on a CUDA multiprocessor (`target`, else the
    process's own), whose forms Dew measured (a division's, XLA's special functions'): `_fp32`, on a CUDA
    target."""
    held = target or generation.gpu_target()
    return held.architecture.multiprocessor is not None and _fp32(outputs, fp32)


def _fp32(outputs: Sequence, fp32: bool | None = None) -> bool:
    """Whether an operation of results `outputs` is an fp32 one: by `fp32` where a caller states it (the
    algebra's evidence, whose terms hold no dtype), else by its results' dtypes; fp64's and narrower floats'
    unmeasured. A kernel's own forms are a CUDA kernel's whatever process counts them."""
    return all(aval.dtype == jnp.float32 for aval in outputs) if fp32 is None else fp32



def _operated(primitive, inputs: Sequence, outputs: Sequence) -> Work:
    """An elementwise operation's work, one an output element, a clamp's two (a maximum and a minimum: no
    instruction bounds a value on both sides, and every lowering, XLA's, Mosaic GPU's and Triton's, issues
    both): on the integer units where its operands are integers or predicates (logic, shifts, integer
    arithmetic and comparisons: the ALU's LOP3, SHF, IADD3, ISETP and PLOP3, `Work.integer_operations`; a
    multiply the FMA pipe's IMAD, `Work.integer_multiply_adds`, as a bit form's are, `bits.BitForm.work`),
    else on the fp32 lanes. Its last operand's type is its operands' (a select's arms', its predicate
    first)."""
    count = sum(_elements(aval) for aval in outputs) * (2 if primitive is lax.clamp_p else 1)
    dtype = jnp.dtype(inputs[-1].dtype)
    if not (jnp.issubdtype(dtype, jnp.integer) or dtype == jnp.bool_):
        return Work(operations=count)
    if primitive is lax.mul_p:
        return Work(integer_multiply_adds=count)
    return Work(integer_operations=count)


_ARITHMETIC = _OPERATIONS | _REDUCTIONS | SPECIAL_FUNCTIONS | {lax.convert_element_type_p, lax.atan2_p}
"""The primitives a kernel body's count prices by `operation_work`: its elementwise arithmetic, a product
apart (`_Counting.equation`'s, a TPU core's)."""

_PURE = _ARITHMETIC | frozenset({lax.broadcast_in_dim_p, lax.reshape_p, lax.squeeze_p, lax.transpose_p,
                                 lax.bitcast_convert_type_p, gpu_core.layout_cast_p, lax.slice_p,
                                 lax.concatenate_p, lax.rev_p, pallas.multiple_of_p})
"""What computes its outputs of its operands' values alone, nothing read of memory nor of a thread's place:
elementwise arithmetic and a value's elements rearranged, which of static operands is static (`_static`)."""


def _static(equation, given: Sequence[bool]) -> bool:
    """Whether `equation`'s outputs are of the program's constants alone: an iota's (a tile's row and column
    indices), and what a pure operation (`_PURE`) computes of static operands (`given`: literals, and values
    so computed), the masks a tile's indices make among them; a call's, a scope's or a branch's outputs are
    as the body the count runs states them (`_Counting.run`). The GPU's
    compile folds such a value or computes it once outside every loop, so it issues nothing a step: the
    recurrence solve's masks, priced at 6.1M integer operations a GH200 program (FS-4123-INT3, its launch
    integer-bound), compile to about a thousand integer instructions a thread a walk's trip, its moves and
    conversions among them (FS-SOLVE-SASS2), their floor division and its truncating form to one binary
    (M-JS-258, row 4123). A value the count holds concretely of a thread's place or a trip's (a warpgroup's
    index, a step's parity, a scan's carry) is none of them."""
    if equation.primitive in (lax.iota_p, control_flow.platform_index_p):
        # An iota's, and the platform its program's compile chooses for (`_platform_branch`).
        return True
    return equation.primitive in _PURE and bool(given) and all(given)


def operation_work(primitive, inputs: Sequence, outputs: Sequence, params: Mapping,
                   target: generation.GpuTarget | None = None, *, forms: Mapping | None = None,
                   fp32: bool | None = None) -> Work:
    """What one equation of `primitive` computes over operands and results of the avals `inputs` and
    `outputs` (each its shape and dtype), with `params`, on `target` (the process's own, or the one it
    assumes, where None: `generation.gpu_target`): the one rule the counter prices a kernel body's
    elementwise arithmetic by (`emitted`) and the algebra a term (`dew.algebra.cost.work`). An elementwise
    operation's work (`_operated`); a reduction's one operation an input element; a special function's
    exponentials and a conversion's conversions, one an output element; a product's `2 m n k`, its output's
    elements over its contracted ones, in the mode and passes XLA multiplies it in on `target`
    (`product.product_mode`: on a TPU, an fp32 product's bfloat16 passes); what moves and computes nothing
    nothing (`_FREE`: a layout change, a barrier); any other primitive's output elements unpriced
    (`Work.unpriced_operations`). With `forms` (a kernel's: `APPROXIMATE_FORMS`, `ACCURATE_FORMS`), a special
    function issues its form an output element, one with none unpriced; without, an fp32 one on a CUDA
    target XLA's own (`XLA_FORMS`, M-FS-78), any other one exponential an element. An fp32 division on a
    CUDA target issues its kernel's form (`MOSAIC_DIVISION`), XLA's own `div.rn.f32`'s (`IEEE_DIVISION`,
    M-FS-77); other widths' and a TPU's, unmeasured, an operation an element. A correctly rounded one issues
    its native form where a kernel's forms state one (a Triton kernel's, `TRITON_FORMS`), else the software
    division it lowers to (XLA's and Mosaic GPU's, `division.software_divide`), counted as a
    kernel's body is, through its calls (row 12886). An integer power issues its repeated squaring's
    multiplies, a negative one beside the division of one by them (row 12948). A kernel's fp32 erf and atan2
    issue the polynomials its forms state (`POLYNOMIAL_FORMS`); another's erf one operation and atan2
    unpriced, unmeasured (an fp64 one lowers to libdevice's own). Whether an operation is fp32 its results'
    dtypes say, or `fp32` where a caller states it (the algebra's terms hold no dtype)."""
    if primitive is division.correctly_rounded_divide_p:
        if (native := (forms or {}).get(primitive)) is not None:
            return repeated(native, sum(_elements(aval) for aval in outputs))
        kind = next((name for name, table in _KERNEL_FORMS.items() if forms == table), None)
        return _software_division(tuple((tuple(aval.shape), np.dtype(aval.dtype)) for aval in inputs),
                                  target or generation.gpu_target(), kind)
    if primitive is lax.div_p and _fp32_on_cuda(outputs, target, fp32) and (
            (form := IEEE_DIVISION if forms is None else forms.get(primitive)) is not None):
        return repeated(form, sum(_elements(aval) for aval in outputs))
    if primitive is lax.integer_pow_p:
        # Repeated squaring's multiplies, as jax lowers it, and a negative power's reciprocal of them.
        power = abs(params["y"])
        squared = _operated(lax.mul_p, inputs, outputs)
        squarings = repeated(squared, max(power.bit_length() + power.bit_count() - 2, 0))
        return squarings if params["y"] >= 0 else squarings + operation_work(
            lax.div_p, [*inputs, *inputs], outputs, {}, target, forms=forms, fp32=fp32)
    if (primitive in POLYNOMIAL_FORMS and (form := (forms or {}).get(primitive)) is not None
            and _fp32(outputs, fp32)):
        return repeated(form, sum(_elements(aval) for aval in outputs))
    if primitive in _OPERATIONS:
        return _operated(primitive, inputs, outputs)
    if primitive in _REDUCTIONS:
        return Work(operations=_elements(inputs[0]))
    if primitive in SPECIAL_FUNCTIONS:
        elements = sum(_elements(aval) for aval in outputs)
        if forms is None:
            measured = XLA_FORMS.get(primitive) if _fp32_on_cuda(outputs, target, fp32) else None
            return Work(exponentials=elements) if measured is None else repeated(measured, elements)
        form = forms.get(primitive)
        return Work(unpriced_operations=elements) if form is None else repeated(form, elements)
    if primitive is lax.convert_element_type_p:
        elements = sum(_elements(aval) for aval in outputs)
        # Its target the dtype it names (`new_dtype`), which the algebra's terms, every value priced at fp32
        # (`cost.VALUE`), state where their avals do not.
        source, made = np.dtype(inputs[0].dtype), np.dtype(params.get("new_dtype") or outputs[0].dtype)
        if source == made:
            # Its own dtype (a weak type made strong): no instruction (FS-CONV-SASS2's fp32 to fp32).
            return Work()
        if forms in _KERNEL_FORMS.values():
            form = CONVERSION_FORMS.get((source, made))
            if form is not None:
                return repeated(form, elements)
        return Work(conversions=elements)
    if primitive is lax.dot_general_p:
        (contracted, _), _ = params["dimension_numbers"]
        lhs, rhs = inputs[:2]
        field, passes = product_mode(params.get("precision"), lhs.dtype, rhs.dtype,
                                     target or generation.gpu_target())
        flops = 2 * _elements(outputs[0]) * math.prod(lhs.shape[dimension] for dimension in contracted)
        return multiplied(field, flops * passes)
    if primitive in _FREE:
        return Work()
    return Work(unpriced_operations=sum(_elements(aval) for aval in outputs))


def _then(first: Work, then: Work) -> Work:
    """`first`'s work followed by `then`'s on one warpgroup: their units' sum, their paths followed
    (`cost.followed`)."""
    return dataclasses.replace(first + then, chains=followed(first.chains, then.chains))


def _mean(walks: Sequence[Work]) -> Work:
    """A step of `walks`, a role's loops: their steps' mean (`Role.step`), their paths followed one by the
    next through the ring they share (`cost.followed`), each loop's weight kept, before the mean."""
    paths = functools.reduce(followed, (walk.chains for walk in walks))
    return repeated(dataclasses.replace(sum(walks, Work()), chains=paths), 1 / len(walks))


def built_with(cost: pl.CostEstimate | Stated | None) -> pl.CostEstimate | None:
    """The estimate a launch of `cost` is built with: its own, or none where its emission counts it
    (`costed`)."""
    return None if isinstance(cost, Stated) else cost


def costed(launch: Callable, cost: pl.CostEstimate | Stated | None) -> Callable:
    """`launch` (a family's, `mosaic.Family.kernel`, built `built_with` its `cost`), its kernel's cost (its
    `pallas_call`'s `cost_estimate`, which XLA and Dew's FLOP count read) counted from its emission where
    the launch states its walk (`Stated`): traced once, its kernel's estimate filled (`_filled`), then bound
    as traced. Any other cost is the launch's own."""
    if not isinstance(cost, Stated):
        return launch

    @functools.wraps(launch)
    def counted(*operands):
        traced, shape = jax.make_jaxpr(launch, return_shape=True)(*operands)
        filled = _filled(traced, cost)
        return jax.tree.unflatten(jax.tree.structure(shape),
                                  jax_core.eval_jaxpr(filled, filled.consts, *jax.tree.leaves(operands)))

    return counted


def _filled(jaxpr: jax_core.Jaxpr, stated: Stated) -> jax_core.Jaxpr:
    """`jaxpr` with its kernel's estimate counted (`_estimate`), the calls holding it (a custom batching
    rule's, `plgpu.kernel`'s) rewritten around it."""
    def rewritten(equation):
        if equation.primitive in (mpmd.mpmd_map_p, pallas_call.pallas_call_p):
            launched = _LAUNCHED[_body(equation)] = _launched(equation, stated)
            work = launched.work
            return equation.replace(params={**equation.params, "cost_estimate": _estimate(work)})
        return equation.replace(params={name: _filled(value, stated) if isinstance(value, jax_core.Jaxpr)
                                        else value for name, value in equation.params.items()})

    return jaxpr.replace(eqns=[rewritten(equation) for equation in jaxpr.eqns])


def launch_programs(kernel) -> int:
    """The programs launched by `kernel` (`mpmd_map` or `pallas_call`), its grid's product."""
    if kernel.primitive is mpmd.mpmd_map_p:
        (mesh,) = kernel.params["meshes"]
        return math.prod(mesh.grid)
    return math.prod(kernel.params["grid_mapping"].grid)


def launch_work(kernel, stated: Stated) -> Work:
    """A kernel equation's work from its body's emission, a launch's: each unit's, a step's `stated.steps`
    times and the rest each program's (`Emitted.issued`), the bytes it states its device's and the bytes its
    copies read past its buffers' whole its L2's (`Work.l2_bytes`), its span the
    steps its longest program walks (`Stated.longest`, else their mean), and one kernel. Its cost
    estimate's (`_estimate`) and a step's price (`work_of`, `attention_decode.priced_step`) alike."""
    return _launched(kernel, stated)[0]


def _launched(kernel, stated: Stated) -> "Launched":
    """`launch_work`'s work, and the warps a program's issue spreads over (`warps_of`)."""
    forms = forms_of(kernel.params["compiler_params"])
    programs = launch_programs(kernel)
    if kernel.primitive is mpmd.mpmd_map_p:
        (mesh,) = kernel.params["meshes"]
        counted = emitted(_body(kernel), mesh.num_threads or 1, mesh.thread_name, forms=forms)
    else:
        counted = emitted(_body(kernel), 1, None, forms=forms)

    def issued(field: str) -> float:
        return counted.issued(lambda work: getattr(work, field), stated.steps, programs)

    # The steps on its critical path: its longest program's walk where it states one, else their mean.
    walked = stated.steps / programs if stated.longest is None else stated.longest

    # Its compulsory bytes the ones it states; its copies' reads past every buffer it holds, each whole once,
    # read again from the L2 (row 14180: a tile's repeated reads of an operand other programs read). Past
    # them all, so at most its rereads: a first read a statement omits is no reread (an operand's, an
    # output's read back), nor a store or a reduction; a buffer read in part or written alone only hides
    # another's.
    footprint = stored(*(var.aval for var in (*kernel.invars, *kernel.outvars)))
    reads = counted.issued(lambda work: work.device_reads, stated.steps, programs)
    work = Work(**{unit: issued(unit) for unit in ISSUED_UNITS if unit != "issue_excess"},
                device_bytes=stated.bytes_accessed, l2_bytes=max(0.0, reads - footprint),
                span=walked, kernels=1, chains=counted.paths(walked))
    # Its walking roles' issue slots, the most one of them issues a step (`Work.issue_excess`).
    issuing = counted.issued(issue_slots, stated.steps, programs)
    # A Mosaic GPU program's warps: a core map's (`plgpu.kernel`) or a call compiled with Mosaic GPU's
    # parameters (`forms_of`); another backend's unknown.
    mosaic = kernel.primitive is mpmd.mpmd_map_p or forms in _KERNEL_FORMS.values()
    return Launched(dataclasses.replace(work, issue_excess=max(0.0, issue_slots(work) - issuing)),
                    _warps(counted) if mosaic else 0.0)


def _warps(counted: Emitted) -> float:
    """The warps a Mosaic GPU program's issue spreads over, from the population its issue slots count
    (`Emitted.issued`: the busiest walking warpgroup's step, the steps they walk together): each warpgroup
    whose step is of the busiest's scale (at least half its issue slots) its four, weighted by its share of
    the busiest's, a step of its walk's where any walks, else outside it. A warpgroup issuing a sliver of
    them (a producer's copies) hides no interval the slots count (the review of lane/toolkit 61dd3c7d2)."""
    walking = [role for role in counted.roles if role.walks]
    slots = ([issue_slots(_mean(role.walks)) for role in walking] if walking
             else [issue_slots(role.fixed) for role in counted.roles])
    busiest = max(slots, default=0.0)
    if busiest <= 0:
        return 0.0
    return 4 * sum(each for each in slots if 2 * each >= busiest) / busiest


def _estimate(work: Work) -> pl.CostEstimate:
    """A kernel's cost estimate from its launch's work (`launch_work`): its tensor work (every precision's
    FLOPs), its exponentials and the bytes it states."""
    return pl.CostEstimate(flops=int(work.flops + work.tf32_flops + work.fp8_flops),
                           transcendentals=int(work.exponentials), bytes_accessed=int(work.device_bytes))


class Launched(NamedTuple):
    """A costed launch as its emission counted it once: its work (`launch_work`) and the warps its programs
    issue over (`_warps`)."""

    work: Work
    warps: float


_LAUNCHED: "weakref.WeakKeyDictionary[jax_core.Jaxpr, Launched]" = weakref.WeakKeyDictionary()
"""Each costed launch's count (`Launched`) by its kernel's body, as `_filled` counted it: a program a
launch's lowering cache holds (`tuning.lowered_once`) answers with the body it traced once."""


def forms_of(compiler_params) -> Mapping | None:
    """The special functions' forms a kernel compiled with `compiler_params` lowers them to (a kernel
    equation's own, `launch_work`; a launch's family's, `Family.compiled_with`): a Mosaic GPU kernel's
    approximate forms where they ask `approx_math`, its accurate ones elsewhere; a Pallas Triton kernel's its
    own (`TRITON_FORMS`), read from the parameters it binds (a `CompilerParams` instance or a mapping of
    one); None for another backend's (a Mosaic TPU kernel), counted by `operation_work`'s one rule."""
    held = compiler_params
    if isinstance(compiler_params, Mapping):
        held = compiler_params.get("mosaic_gpu") or compiler_params.get("triton")
    if isinstance(held, triton_core.CompilerParams):
        return TRITON_FORMS
    if not isinstance(held, gpu_core.CompilerParams):
        return None
    return APPROXIMATE_FORMS if held.approx_math else ACCURATE_FORMS


def _body(kernel) -> jax_core.Jaxpr:
    """A kernel equation's body: its `mpmd_map`'s one jaxpr or its `pallas_call`'s."""
    return kernel.params["jaxprs"][0] if kernel.primitive is mpmd.mpmd_map_p else kernel.params["jaxpr"]


def warps_of(kernel) -> float | None:
    """The warps a kernel equation's programs issue over as its costed launch counted them (`_warps`: a
    Mosaic GPU program's; 0 for another backend's), None where no costed launch counted it."""
    launched = _LAUNCHED.get(_body(kernel))
    return None if launched is None else launched.warps


def resident_of(kernel) -> int:
    """The programs a multiprocessor holds of a kernel equation's launch as its plan compiled it: its Mosaic
    GPU parameters' `min_blocks_per_sm` (`mosaic.compiler_params`' `resident`), one where it states none."""
    params = kernel.params.get("compiler_params")
    held = params.get("mosaic_gpu") if isinstance(params, Mapping) else params
    if not isinstance(held, gpu_core.CompilerParams):
        return 1
    return max(1, held.min_blocks_per_sm)


def launched_kernels(jaxpr: jax_core.Jaxpr) -> list[tuple[jax_core.JaxprEqn, Launched]] | None:
    """A traced program's native launches in sequence, through the calls it makes (`_expanded`), each with
    its costed launch (`Launched`: its work and warps as its emission counted them): the one discovery a
    planner prices a program's launches by (`launches_seconds`, attention's `price_config`s). None where any
    launch's work its emission does not count, or a launch lies under a loop or a branch, whose trips or
    taken arm no sequence states (the review of lane/toolkit 61dd3c7d2): its price unknown, never 0."""
    found = []
    for equation in jaxpr.eqns:
        if equation.primitive in programs.NATIVE_KERNELS:
            if (launched := _LAUNCHED.get(_body(equation))) is None:
                return None
            found.append((equation, launched))
        elif (expanded := _expanded(equation)) is not None:
            if (called := launched_kernels(expanded[0])) is None:
                return None
            found += called
        elif any(inner.primitive in programs.NATIVE_KERNELS for inner in programs.equations(jaxpr.replace(
                eqns=[equation]))):
            return None
    return found


def launches_seconds(jaxpr: jax_core.Jaxpr, target: generation.GpuTarget, *,
                     resident: Callable[[jax_core.JaxprEqn], int] = resident_of) -> float | None:
    """A traced program's native launches (`launched_kernels`) priced on `target`, in sequence: each
    kernel's work as its costed launch counted it, over its programs (`launch_programs`) at the residency its
    plan gives it (`resident`; its compiled plan's by default, `resident_of`) and the warps it issues over,
    through `cost.launch_seconds`; None where `launched_kernels` is. What XLA runs beside them is the
    algebra's to price (`dew.algebra.program.priced`)."""
    if (found := launched_kernels(jaxpr)) is None:
        return None
    return sum(launch_seconds(launched.work, launch_programs(equation), target, resident=resident(equation),
                              warps=launched.warps) for equation, launched in found)


def work_of(kernel) -> Work | None:
    """A kernel equation's work as its costed launch counted it (`launch_work`), None where no costed launch
    counted it (a launch whose cost its emission does not count, `costed`)."""
    launched = _LAUNCHED.get(_body(kernel))
    return None if launched is None else launched.work
