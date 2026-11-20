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

import dataclasses
import functools
import math
import weakref
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import ad_util, core as jax_core, hijax, pjit
from jax._src.pallas import mpmd, pallas_call, primitives as pallas
from jax._src.pallas.mosaic import primitives as tpu
from jax._src.pallas.mosaic_gpu import core as gpu_core, primitives as gpu
from jax._src.state import primitives as state, types as state_types
from jax.experimental import pallas as pl

from dew.nn import derivatives, precision, programs
from dew.nn.kernels.templates import resources
from dew.nn.kernels.templates.cost import Stated, Work


class Unpriced(ValueError):
    """A primitive in a kernel's body no rule of `emitted` prices."""


_STATED: "weakref.WeakKeyDictionary[Callable, Callable]" = weakref.WeakKeyDictionary()
"""What each opaque Mosaic GPU body a kernel inlines does (`stating`), by the body."""


def stating[F: Callable](work: Callable[..., Work]) -> Callable[[F], F]:
    """A decorator for an `inline_mgpu` body (`plgpu.inline_mgpu`'s function), stating what it does as
    `work(*avals)` of its arguments' avals (a ref's own, before any transform it is passed through), which
    the count reads where the kernel inlines it: an inlined
    body is opaque to it, so its emitter, which knows it, states it (a ticket's atomic and barrier, no
    product; a gathered copy's bytes)."""
    def stated(fn: F) -> F:
        _STATED[fn] = work
        return fn
    return stated


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


def emitted(body: jax_core.Jaxpr, warpgroups: int, axis: str | None, *, device: Sequence[int] = ()
            ) -> Emitted:
    """`body`, a kernel's jaxpr (its `mpmd_map`'s or its `pallas_call`'s), as each of a program's
    `warpgroups` warpgroups along `axis` runs it (`Emitted`; a program of one, no axis, a TPU's or an
    mma.sync family's). Its operands at the indices `device` are refs in device memory, each read and write
    of them its bytes' device traffic (a Triton launch's blocks, which its loads and stores move: the decode
    merge's, `merged._channel_block`)."""
    roles = []
    for index in range(warpgroups):
        counted = _Counting(axis, index)
        counted.run(body, [_DEVICE if at in device else None for at in range(len(body.invars))])
        roles.append(Role(counted.fixed, tuple(counted.walks)))
    return Emitted(tuple(roles))


_DEVICE = object()
"""A ref in device memory, as `emitted`'s walk holds it (`_Counting.equation` prices its reads and writes)."""


_OPERATIONS = frozenset({
    lax.abs_p, lax.add_p, lax.and_p, lax.ceil_p, lax.clamp_p, lax.div_p, lax.eq_p, lax.floor_p, lax.ge_p,
    lax.gt_p, lax.le_p, lax.lt_p, lax.max_p, lax.min_p, lax.mul_p, lax.ne_p, lax.neg_p, lax.not_p, lax.or_p,
    lax.rem_p, lax.select_n_p, lax.shift_left_p, lax.shift_right_logical_p, lax.shift_right_arithmetic_p,
    lax.sign_p, lax.sub_p, lax.xor_p, lax.integer_pow_p, lax.square_p, lax.round_p,
    lax.one_minus_square_p, ad_util.add_jaxvals_p, lax.clz_p, lax.is_finite_p, lax.erf_p, lax.expm1_p,
    lax.nextafter_p, precision.correctly_rounded_divide_p})
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
    lax.stack_p, lax.tile_p, lax.unstack_p, gpu.commit_group_p, ad_util.stop_gradient_p})
"""What moves no data between memories and computes nothing a unit's rate prices: synchronization, layouts,
indices, a value's elements rearranged among a role's registers (a gather's, a slice's, a concatenation's;
every layout move the Pallas fuser tiles: a dynamic slice and its update, whose index arithmetic is scalar,
which nothing counts, a split, a stack, a tile and an unstack, so a fused prologue's stage holds no
operation the counter cannot price, `fused_gemm._split`), its reads and writes between its registers and
its scoped refs, which its products' operands and its copies count, and a stopped gradient, its operand."""

_PRODUCTS = frozenset({gpu.wgmma_ref_p, gpu.tcgen05_mma_p})
"""Tensor-core products whose operands' shapes their transforms state (`_product`)."""

_COPIES = frozenset({gpu.copy_gmem_to_smem_p, gpu.copy_smem_to_gmem_p})
"""Copies between device and shared memory (`_copied`)."""

def _folded(equation, values: list) -> list:
    """`equation` over its concrete integer and boolean scalar operands, each at its atom's dtype, as its
    primitive computes it on the GPU (`programs.scalar_folded`): a width's wrap, a quotient's truncation and a
    remainder's sign the program's own; unknown where it computes none, a quotient or remainder by zero or of
    the least integer by -1 among them, which the GPU leaves undefined (LLVM's `sdiv` and `srem`, which Mosaic
    GPU's integer division lowers to), and where Mosaic GPU types it otherwise than XLA (`_typed_alike`)."""
    if not _typed_alike(equation):
        return [None]
    operands = [np.asarray(value, dtype=atom.aval.dtype)[()]
                for value, atom in zip(values, equation.invars, strict=True)]
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


def repeated(work: Work, count: float) -> Work:
    """`work` repeated `count` times in a chain: its every unit's, and its span."""
    return Work(**{field.name: getattr(work, field.name) * count for field in dataclasses.fields(Work)
                   if field.name != "kernels"}, kernels=work.kernels)


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
    """A ref's aval through its transforms (`tree` of `leaves`)."""
    shaped = _unreffed(aval)
    if tree is not None:
        for transform in jax.tree.unflatten(tree, leaves):
            shaped = _unreffed(transform.transform_type(shaped))
    return shaped


def _logical(aval, tree, leaves: Sequence) -> tuple[int, ...]:
    """A ref's shape as a product reads it: its aval through its transforms (`_transformed`)."""
    return tuple(_transformed(aval, tree, leaves).shape)


def _product(equation) -> Work:
    """A tensor-core product's work, `2 m n k` in its operands' mode: its accumulator `[m, n]` and its left
    operand `[m, k]` as their transforms read them (the trailing `*_transforms_tree` leaves of its
    operands)."""
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
    return _tensor(avals[1].dtype, 2 * m * n * k)


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
    """An `async_copy_smem_to_tmem` copy's shared-memory read bytes: its SMEM source's, as its transforms
    slice it (`smem_tree` leaves after `(smem_ref, tmem_ref)`, rows 9106, 9476)."""
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

    def compared(self, primitive, other) -> bool | int | None:
        """`self <primitive> other` where the place decides it: equal to its range's start at its first
        trip alone, one past it short of its range's stop at every trip but its last, its remainder by two
        its parity; None otherwise."""
        if primitive is lax.eq_p and self.plus == 0 and other == self.start:
            return self.at == "first"
        if primitive is lax.lt_p and self.plus == 1 and other == self.stop:
            return self.at != "last"
        if primitive is lax.rem_p and other == 2:
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

    def __init__(self, axis: str | None, index: int, loops: int = 0, warp: tuple | None = None):
        self.axis, self.index, self.warp = axis, index, warp
        self.fixed = Work()
        self.walks: list[Work] = []
        self.loops = [_Loop() for _ in range(loops)]

    def add(self, work: Work) -> None:
        if self.loops:
            self.loops[-1].work = self.loops[-1].work + work
        else:
            self.fixed = self.fixed + work

    def run(self, jaxpr: jax_core.Jaxpr, values: list, constants: Sequence = ()) -> list:
        """Count `jaxpr` given its inputs' concrete `values` (None where unknown), and its constants'
        (`constvars`) where its caller binds them (a scoped body's, `run_scoped`'s operands); its outputs'
        concrete values."""
        held = dict(zip(jaxpr.invars, values, strict=True))
        if constants:
            held.update(zip(jaxpr.constvars, constants, strict=True))

        def read(atom):
            if isinstance(atom, jax_core.Literal):
                # A literal's value is a host array (`literals.TypedNdArray`), a scalar's of no dimension,
                # held at its own dtype as the program computes with it.
                return np.asarray(atom.val)[()] if np.ndim(atom.val) == 0 else atom.val
            return held.get(atom)

        # The variables a later equation or the jaxpr's outputs read: a swap's old value is live where read.
        live = {atom for atoms in (*(equation.invars for equation in jaxpr.eqns), jaxpr.outvars)
                for atom in atoms if isinstance(atom, jax_core.Var)}
        for equation in jaxpr.eqns:
            values = [read(atom) for atom in equation.invars]
            outputs = self.moved(equation, values, live) or self.equation(equation, values)
            held.update(zip(equation.outvars, outputs, strict=True))
        return [read(atom) for atom in jaxpr.outvars]

    def moved(self, equation, values: list, live: set) -> list | None:
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

    def equation(self, equation, values: list) -> list:
        primitive, params = equation.primitive, equation.params
        unknown = [None] * len(equation.outvars)
        if primitive is lax.axis_index_p:
            return [self.place(params["axis_name"])]
        if (primitive in programs.SCALAR_FOLDED and values
                and all(_scalar(value) for value in values)):
            if primitive is lax.convert_element_type_p:
                self.add(Work(conversions=1))
            return _folded(equation, values)
        if (placed := _placed(primitive, values)) is not None:
            return [placed]
        if primitive is pallas.multiple_of_p:
            # A divisibility hint: its operand's value, nothing computed.
            return [values[0]]
        if (primitive is lax.select_n_p and _scalar(values[0])
                and all(value is not None for value in values[1:])):
            return [values[1 + int(values[0])]]
        if primitive in _OPERATIONS:
            self.add(Work(operations=sum(_elements(v.aval) for v in equation.outvars)))
        elif primitive in _REDUCTIONS:
            self.add(Work(operations=_elements(equation.invars[0].aval)))
        elif primitive in SPECIAL_FUNCTIONS:
            self.add(Work(exponentials=sum(_elements(v.aval) for v in equation.outvars)))
        elif primitive is lax.convert_element_type_p:
            self.add(Work(conversions=sum(_elements(v.aval) for v in equation.outvars)))
        elif primitive in _PRODUCTS:
            self.add(_product(equation))
        elif primitive is gpu.mma_p:
            acc, lhs = equation.invars[0].aval, equation.invars[1].aval
            self.add(_tensor(lhs.dtype, 2 * acc.shape[0] * acc.shape[1] * lhs.shape[1]))
        elif primitive in _COPIES:
            copied = _copied(equation)
            self.add(Work(device_bytes=copied, shared_bytes=copied))
        elif primitive is gpu.async_copy_smem_to_tmem_p:
            self.add(Work(shared_bytes=_smem_to_tmem_bytes(equation)))
        elif primitive is tpu.dma_start_p:
            # A TPU core's copy between HBM and VMEM: its destination's bytes as its transforms read it.
            copied = _bytes(params["tree"].unflatten([variable.aval for variable in equation.invars])[1])
            self.add(Work(device_bytes=copied, shared_bytes=copied))
        elif primitive is gpu.atomic_store_p:
            # An atomic update of device memory (an add into a sum other programs add to): its value's bytes.
            value = params["args_tree"].unflatten([variable.aval for variable in equation.invars])[2]
            self.add(Work(device_bytes=_bytes(value)))
        elif primitive is lax.dot_general_p:
            # A TPU core's product on its MXU: `2 m n k`, its output's elements over its contracted ones.
            (contracted, _), _ = params["dimension_numbers"]
            lhs = equation.invars[0].aval
            self.add(_tensor(lhs.dtype, 2 * _elements(equation.outvars[0].aval)
                             * math.prod(lhs.shape[dimension] for dimension in contracted)))
        elif primitive is gpu.inline_mgpu_p:
            work = _STATED.get(params["mgpu_fn"])
            if work is None:
                raise Unpriced(f"an inlined Mosaic GPU body states no work ({params['mgpu_fn']}): its "
                               "emitter's `emission.stating`")
            # Its arguments lead its operands, their refs' transforms' leaves after them.
            arguments = equation.invars[:params["pytree_args"].num_leaves]
            self.add(work(*(variable.aval for variable in arguments)))
        elif primitive is lax.cond_p:
            return self.cond(params["branches"], values)
        elif (region := _REGIONS.get(primitive)) is not None:
            return region(self, equation, values) or unknown
        elif primitive is derivatives.marked_call_p and isinstance(params["kind"], _MARKS):
            return self.marked(params, values)
        elif (expanded := _expanded(equation)) is not None:
            return self.run(expanded[0], values, expanded[1])
        elif primitive not in _FREE:
            raise Unpriced(f"no rule prices {primitive} in a kernel's body")
        return unknown

    def marked(self, params, values: list) -> list:
        """A marked span a count reads by its kind: a rare span's verdict as its common value, False
        (`derivatives.Rare`), a walk's step counted by its loop and run (`derivatives.WalkStep`)."""
        if isinstance(params["kind"], derivatives.Rare):
            return [np.False_]
        if self.loops:
            self.loops[-1].marks += 1
        return self.run(params["call_jaxpr"], values)

    def place(self, name) -> int | None:
        """The concrete index along axis `name`: its warp's among its warpgroup's warps (`warp`), its
        warpgroup's along `axis`; None along any other."""
        found = dict([(self.axis, self.index), *([self.warp] if self.warp is not None else [])])
        return next((index for axis, index in found.items() if name in (axis, (axis,))), None)

    def scoped(self, equation, values: list) -> list:
        """A scope (`run_scoped`): its operands its body's constants, its body's inputs the refs it
        allocates."""
        body = equation.params["jaxpr"]
        return self.run(body, [None] * len(body.invars), values)

    def loop(self, equation, values: list) -> None:
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
        ranged = _ranged(params, held)

        def trip(at: str, parity: int) -> "_Counting":
            # A carried value is unknown past the first trip, but a range's counter and stop (`_ranged`).
            inputs = [*consts, *([None] * carry)]
            if ranged is not None:
                counter, start, stop, bound = ranged
                inputs[len(consts) + counter] = _Place(at, start, stop, parity)
                inputs[len(consts) + bound] = stop
            trial = _Counting(self.axis, self.index, loops=len(self.loops) + 1, warp=self.warp)
            trial.run(body, inputs)
            return trial

        steady, odd = trip("steady", 0), trip("steady", 1)
        self.fixed = self.fixed + steady.fixed
        self.walks.extend(steady.walks)
        found = _Loop(_mean([steady.loops[-1].work, odd.loops[-1].work]), steady.loops[-1].marks)
        self.ended(found, walks=True)
        if ranged is not None and found.marks and not self.loops:
            # The first and last trips each take their own work in place of a steady one's.
            for at in _PLACES[1:]:
                taken = _mean([trip(at, parity).loops[-1].work for parity in (0, 1)])
                self.fixed = self.fixed + taken + repeated(found.work, -1)

    def warps(self, equation, values: list) -> None:
        """A warpgroup's warps each running its role (`plgpu.warp_map`'s `mpmd_map`: tcgen05's load and
        product warps): each warp's body counted with its warp's index concrete, so its role's branch is the
        one it takes; the warpgroup's work theirs together, a step of its walk each warp's step summed, and
        the steps its warps take together its enclosing loop's (the most any warp marks)."""
        del values
        (mesh,), (inner,) = equation.params["meshes"], equation.params["jaxprs"]
        ((name, count),) = mesh.shape.items()
        steps, marks = [], 0.0
        for warp in range(count):
            counted = _Counting(self.axis, self.index, loops=len(self.loops), warp=(name, warp))
            counted.run(inner, [None] * len(inner.invars))
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

    def scanned(self, equation, values: list) -> list:
        """A loop of the trips its jaxpr states (a scan's), each trip counted as it runs, its carry's
        concrete values carried to the next (a walk's step index, so a step's guard on it takes its
        branch): its work where it stands, its marks its enclosing loop's; its outputs' concrete carry."""
        params = equation.params
        body, length = params["jaxpr"], params["length"]
        consts, carried, xs = map(list, params["ft_in"].update(values).unpack())
        self.loops.append(_Loop())
        for _ in range(length):
            carried = self.run(body, [*consts, *carried, *([None] * len(xs))])[:len(carried)]
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

    def cond(self, branches, values: list) -> list:
        """A conditional: its branch where its index is concrete (a role's, by its warpgroup), else every
        branch's envelope, unit by unit (the most of each any branch issues, as the custom call's FLOP count
        takes a conditional), its steps' marks likewise; the walks its branches hold one walk, a step of it
        the most of each unit any branch's walks take a step (their mean, `Role.step`)."""
        index, operands = values[0], values[1:]
        if _scalar(index):
            return self.run(branches[int(index)], operands)
        trials = []
        for branch in branches:
            trial = _Counting(self.axis, self.index, loops=len(self.loops), warp=self.warp)
            trial.run(branch, operands)
            trials.append(trial)
        self.fixed = self.fixed + _envelope(trial.fixed for trial in trials)
        if any(trial.walks for trial in trials):
            self.walks.append(_envelope(_mean(trial.walks) for trial in trials if trial.walks))
        if self.loops:
            self.loops[-1].work = self.loops[-1].work + _envelope(trial.loops[-1].work for trial in trials)
            self.loops[-1].marks += max(trial.loops[-1].marks for trial in trials)
        return [None] * len(branches[0].outvars)


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
    if not (_scalar(other) or isinstance(other, _Same)):
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


def _expanded(equation) -> tuple[jax_core.Jaxpr, Sequence] | None:
    """The program an equation runs as it lowers, where it is one of JAX's primitives' own, and its
    constants: a call's (`programs.called`); a high-level operation's expansion (`HiPrim`'s `to_lojax`: a
    declared tangent's, a contraction whose transpose it states); the construct's own operation's value (the
    algebra's law, `scaled.primitive`, as a kernel's lowering runs it: the decode merge's, `merged`). None
    for any other."""
    # The law's module read here: `dew.algebra` reads this module's counts as it is imported (`cost`).
    from dew.algebra import scaled

    primitive, params = equation.primitive, equation.params
    if (inner := programs.called(equation)) is not None:
        return inner, ()
    if primitive is hijax.call_hi_primitive_p:
        expansion = functools.partial(primitive.to_lojax, **params)
    elif (made := scaled.MADE.get(primitive)) is not None:
        expansion = functools.partial(made[0], **params)
    else:
        return None
    traced = jax.make_jaxpr(expansion)(*(v.aval for v in equation.invars))
    return traced.jaxpr, traced.consts


def _mean(walks: Sequence[Work]) -> Work:
    """A step of `walks`, a role's loops: their steps' mean (`Role.step`)."""
    return repeated(sum(walks, Work()), 1 / len(walks))


def _envelope(works) -> Work:
    """The most of each unit any of `works` holds."""
    works = list(works)
    return Work(**{field.name: max(getattr(work, field.name) for work in works)
                   for field in dataclasses.fields(Work)})


def _scalar(value) -> bool:
    """Whether `value` is a concrete integer or boolean scalar, what a warpgroup's index arithmetic holds:
    a Python integer or boolean, or a NumPy scalar of an integer or boolean dtype (JAX's 4-bit integers among
    them). A float is none: its predicate's branches are unknown, each counted
    (`programs.scalar_folded`)."""
    return isinstance(value, int) or (isinstance(value, np.generic) and any(
        jnp.issubdtype(value.dtype, kind) for kind in (jnp.integer, jnp.bool_)))


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
            work = _WORKS[_body(equation)] = launch_work(equation, stated)
            return equation.replace(params={**equation.params, "cost_estimate": _estimate(work)})
        return equation.replace(params={name: _filled(value, stated) if isinstance(value, jax_core.Jaxpr)
                                        else value for name, value in equation.params.items()})

    return jaxpr.replace(eqns=[rewritten(equation) for equation in jaxpr.eqns])


def launch_work(kernel, stated: Stated) -> Work:
    """A kernel equation's work from its body's emission, a launch's: each unit's, a step's `stated.steps`
    times and the rest each program's (`Emitted.issued`), the bytes it states its device's, its span the
    steps one of its programs walks, and one kernel. Its cost estimate's (`_estimate`) and a step's price
    (`work_of`, `attention_decode.priced_step`) alike."""
    if kernel.primitive is mpmd.mpmd_map_p:
        (mesh,) = kernel.params["meshes"]
        counted = emitted(_body(kernel), mesh.num_threads or 1, mesh.thread_name)
        programs = math.prod(mesh.grid)
    else:
        counted, programs = emitted(_body(kernel), 1, None), math.prod(kernel.params["grid_mapping"].grid)

    def issued(field: str) -> float:
        return counted.issued(lambda work: getattr(work, field), stated.steps, programs)

    return Work(flops=issued("flops"), operations=issued("operations"), exponentials=issued("exponentials"),
                conversions=issued("conversions"), device_bytes=stated.bytes_accessed,
                shared_bytes=issued("shared_bytes"), tf32_flops=issued("tf32_flops"),
                span=stated.steps / programs, fp8_flops=issued("fp8_flops"), kernels=1)


def _estimate(work: Work) -> pl.CostEstimate:
    """A kernel's cost estimate from its launch's work (`launch_work`): its tensor work (every precision's
    FLOPs), its exponentials and the bytes it states."""
    return pl.CostEstimate(flops=int(work.flops + work.tf32_flops + work.fp8_flops),
                           transcendentals=int(work.exponentials), bytes_accessed=int(work.device_bytes))


_WORKS: "weakref.WeakKeyDictionary[jax_core.Jaxpr, Work]" = weakref.WeakKeyDictionary()
"""Each costed launch's work (`launch_work`) by its kernel's body, as `_filled` counted it: a program a
launch's lowering cache holds (`tuning.lowered_once`) answers with the body it traced once."""


def _body(kernel) -> jax_core.Jaxpr:
    """A kernel equation's body: its `mpmd_map`'s one jaxpr or its `pallas_call`'s."""
    return kernel.params["jaxprs"][0] if kernel.primitive is mpmd.mpmd_map_p else kernel.params["jaxpr"]


def work_of(kernel) -> Work | None:
    """A kernel equation's work as its costed launch counted it (`launch_work`), None where no costed launch
    counted it (a launch whose cost its emission does not count, `costed`)."""
    return _WORKS.get(_body(kernel))
