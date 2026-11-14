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
import operator
import weakref
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import ad_util, core as jax_core, pjit
from jax._src.pallas import mpmd, pallas_call, primitives as pallas
from jax._src.pallas.mosaic import primitives as tpu
from jax._src.pallas.mosaic_gpu import core as gpu_core, primitives as gpu
from jax._src.state import primitives as state, types as state_types
from jax.experimental import pallas as pl

from dew.nn import derivatives, programs
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


def emitted(body: jax_core.Jaxpr, warpgroups: int, axis: str | None) -> Emitted:
    """`body`, a kernel's jaxpr (its `mpmd_map`'s or its `pallas_call`'s), as each of a program's
    `warpgroups` warpgroups along `axis` runs it (`Emitted`; a program of one, no axis, a TPU's or an
    mma.sync family's)."""
    roles = []
    for index in range(warpgroups):
        counted = _Counting(axis, index)
        counted.run(body, [None] * len(body.invars))
        roles.append(Role(counted.fixed, tuple(counted.walks)))
    return Emitted(tuple(roles))


_OPERATIONS = frozenset({
    lax.abs_p, lax.add_p, lax.and_p, lax.ceil_p, lax.clamp_p, lax.div_p, lax.eq_p, lax.floor_p, lax.ge_p,
    lax.gt_p, lax.le_p, lax.lt_p, lax.max_p, lax.min_p, lax.mul_p, lax.ne_p, lax.neg_p, lax.not_p, lax.or_p,
    lax.rem_p, lax.select_n_p, lax.shift_left_p, lax.shift_right_logical_p, lax.shift_right_arithmetic_p,
    lax.sign_p, lax.sub_p, lax.xor_p, lax.integer_pow_p, lax.sqrt_p, lax.rsqrt_p, lax.square_p, lax.round_p,
    lax.one_minus_square_p, ad_util.add_jaxvals_p})
"""Elementwise operations of the CUDA cores, one an output element (a cotangent's sum, `add_any`, among
them)."""

_REDUCTIONS = frozenset({lax.reduce_max_p, lax.reduce_min_p, lax.reduce_sum_p})
"""Reductions, one operation an input element."""

_EXPONENTIALS = frozenset({lax.exp_p, lax.exp2_p, lax.log_p, lax.log2_p, lax.log1p_p, lax.tanh_p,
                           lax.logistic_p, lax.sin_p, lax.cos_p, lax.pow_p})
"""The special-function units' operations, one an output element: those Mosaic GPU lowers to the hardware's
approximate forms, or through them (`attention_mosaic_program.APPROXIMATED`'s, but its square roots, which
the CUDA cores' operations count)."""

_FREE = frozenset({
    lax.axis_index_p, gpu.barrier_arrive_p, gpu.barrier_wait_p, lax.bitcast_convert_type_p,
    lax.broadcast_in_dim_p, gpu.commit_smem_p, gpu.commit_tmem_p, gpu.fresh_axis_index_p,
    gpu.griddepcontrol_launch_dependents_p, gpu.griddepcontrol_wait_p, lax.iota_p, gpu_core.layout_cast_p,
    lax.optimization_barrier_p, gpu.set_max_registers_p, lax.slice_p, lax.squeeze_p, lax.reshape_p,
    lax.transpose_p, gpu.tcgen05_commit_arrive_p, gpu.wgmma_wait_p, gpu.wait_smem_to_gmem_p,
    gpu.wait_load_tmem_p, gpu.wgmma_accumulator_deref_p, gpu.wgmma_accumulator_store_p, gpu.load_p,
    gpu.store_p, state.swap_p, gpu.async_load_tmem_p, gpu.async_store_tmem_p, state.get_p,
    pallas.program_id_p, tpu.dma_wait_p, gpu.wait_gmem_to_smem_p, pjit.reshard_p, lax.gather_p,
    lax.dynamic_slice_p, lax.dynamic_update_slice_p, lax.concatenate_p, lax.pad_p, lax.rev_p})
"""What moves no data between memories and computes nothing a unit's rate prices: synchronization, layouts,
indices, a value's elements rearranged among a role's registers (a gather's, a slice's, a concatenation's),
and its reads and writes between its registers and its scoped refs, which its products' operands and its
copies count."""

_PRODUCTS = frozenset({gpu.wgmma_ref_p, gpu.tcgen05_mma_p})
"""Tensor-core products whose operands' shapes their transforms state (`_product`)."""

_COPIES = frozenset({gpu.copy_gmem_to_smem_p, gpu.copy_smem_to_gmem_p})
"""Copies between device and shared memory (`_copied`)."""

_CONCRETE: dict = {
    lax.add_p: operator.add, lax.sub_p: operator.sub, lax.mul_p: operator.mul, lax.max_p: max, lax.min_p: min,
    lax.lt_p: operator.lt, lax.le_p: operator.le, lax.gt_p: operator.gt, lax.ge_p: operator.ge,
    lax.eq_p: operator.eq, lax.ne_p: operator.ne, lax.and_p: operator.and_, lax.or_p: operator.or_,
    lax.not_p: operator.not_, lax.neg_p: operator.neg,
    lax.div_p: lambda a, b: a // b if isinstance(a, int) and isinstance(b, int) else a / b,
    lax.rem_p: lambda a, b: a % b}
"""The scalar arithmetic a warpgroup's index flows through to its roles' predicates and switch, evaluated."""

_TRANSFORMS = ("acc_transforms_tree", "a_transforms_tree", "b_transforms_tree", "barrier_transforms_tree",
               "a_scale_transforms_tree", "b_scale_transforms_tree", "a_sparse_metadata_transforms_tree")
"""A product's operands' transforms, in the order its trailing operands hold their leaves."""

_EIGHT_BIT = frozenset({jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2)})


def _elements(aval) -> int:
    return math.prod(getattr(aval, "shape", ()))


def _bytes(aval) -> int:
    return _elements(aval) * jnp.dtype(aval.dtype).itemsize


def _times(work: Work, count: float) -> Work:
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


def _logical(aval, tree, leaves: Sequence) -> tuple[int, ...]:
    """A ref's shape as a product reads it: its aval through its transforms (`tree` of `leaves`)."""
    shaped = _unreffed(aval)
    if tree is not None:
        for transform in jax.tree.unflatten(tree, leaves):
            shaped = _unreffed(transform.transform_type(shaped))
    return tuple(shaped.shape)


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
    shaped = _unreffed(avals[0 if side == "src" else 1])
    for transform in jax.tree.unflatten(params[f"{side}_transforms_treedef"], leaves[side]):
        shaped = _unreffed(transform.transform_type(shaped))
    return math.prod(shaped.shape) * jnp.dtype(shaped.dtype).itemsize


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
    `mosaic.Slots`) and on its parity (`index % 2`: the accumulator of two a step issues ahead into) read
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
                # A literal's value is a host array (`literals.TypedNdArray`), a scalar's of no dimension.
                return np.asarray(atom.val).item() if np.ndim(atom.val) == 0 else atom.val
            return held.get(atom)

        for equation in jaxpr.eqns:
            outputs = self.equation(equation, [read(atom) for atom in equation.invars])
            held.update(zip(equation.outvars, outputs, strict=True))
        return [read(atom) for atom in jaxpr.outvars]

    def equation(self, equation, values: list) -> list:
        primitive, params = equation.primitive, equation.params
        unknown = [None] * len(equation.outvars)
        if primitive is lax.axis_index_p:
            return [self.place(params["axis_name"])]
        if primitive in _CONCRETE and values and all(_scalar(value) for value in values):
            return [_CONCRETE[primitive](*values)]
        if (placed := _placed(primitive, values)) is not None:
            return [placed]
        if primitive is lax.convert_element_type_p and _scalar(values[0]):
            self.add(Work(conversions=1))
            return [values[0]]
        if (primitive is lax.select_n_p and _scalar(values[0])
                and all(value is not None for value in values[1:])):
            return [values[1 + int(values[0])]]
        if primitive in _OPERATIONS:
            self.add(Work(operations=sum(_elements(v.aval) for v in equation.outvars)))
        elif primitive in _REDUCTIONS:
            self.add(Work(operations=_elements(equation.invars[0].aval)))
        elif primitive in _EXPONENTIALS:
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
        elif primitive is derivatives.marked_call_p and isinstance(params["kind"], derivatives.WalkStep):
            if self.loops:
                self.loops[-1].marks += 1
            return self.run(params["call_jaxpr"], values)
        elif (inner := programs.called(equation)) is not None:
            return self.run(inner, values)
        elif primitive not in _FREE:
            raise Unpriced(f"no rule prices {primitive} in a kernel's body")
        return unknown

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
                self.fixed = self.fixed + taken + _times(found.work, -1)

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
            self.walks.append(_times(found.work, 1 / found.marks))
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


_REGIONS = {pallas.run_scoped_p: _Counting.scoped, lax.while_p: _Counting.loop, lax.scan_p: _Counting.scanned,
            mpmd.mpmd_map_p: _Counting.warps}
"""The regions a kernel's body counts within it, each by its method: its outputs' concrete values, or none
where it leaves them unknown."""


def _mean(walks: Sequence[Work]) -> Work:
    """A step of `walks`, a role's loops: their steps' mean (`Role.step`)."""
    return _times(sum(walks, Work()), 1 / len(walks))


def _envelope(works) -> Work:
    """The most of each unit any of `works` holds."""
    works = list(works)
    return Work(**{field.name: max(getattr(work, field.name) for work in works)
                   for field in dataclasses.fields(Work)})


def _scalar(value) -> bool:
    return isinstance(value, int | bool | float | np.integer | np.bool_ | np.floating)


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
            return equation.replace(params={**equation.params, "cost_estimate": _estimate(equation, stated)})
        return equation.replace(params={name: _filled(value, stated) if isinstance(value, jax_core.Jaxpr)
                                        else value for name, value in equation.params.items()})

    return jaxpr.replace(eqns=[rewritten(equation) for equation in jaxpr.eqns])


def _estimate(kernel, stated: Stated) -> pl.CostEstimate:
    """A kernel equation's cost from its body's emission: its tensor work (every precision's FLOPs) and its
    exponentials, a step's `stated.steps` times and the rest each program's (`Emitted.issued`), and the
    bytes it states."""
    if kernel.primitive is mpmd.mpmd_map_p:
        (mesh,), (body,) = kernel.params["meshes"], kernel.params["jaxprs"]
        counted, programs = emitted(body, mesh.num_threads or 1, mesh.thread_name), math.prod(mesh.grid)
    else:
        body, programs = kernel.params["jaxpr"], math.prod(kernel.params["grid_mapping"].grid)
        counted = emitted(body, 1, None)
    return pl.CostEstimate(
        flops=int(counted.issued(lambda work: work.flops + work.tf32_flops + work.fp8_flops, stated.steps,
                                 programs)),
        transcendentals=int(counted.issued(lambda work: work.exponentials, stated.steps, programs)),
        bytes_accessed=stated.bytes_accessed)
