"""A launch's work by emission (row 5895): its kernel body's jaxpr, each equation priced in Dew's units
(`cost.Work`) as each of a program's warpgroups runs it, its roles' branches resolved by its warpgroup's index
(`handoffs.Schedule.run`'s conditionals and switch on `lax.axis_index`). A loop of trips its jaxpr states
(`scan`) counts its body that many times, one whose trips its data decides (`while`) once; the outermost loop
whose trips run its walk's steps (`derivatives.WalkStep` marks, `mosaic.stepped`) counts one step of its walk
(`Role.step`), a trip's work over the steps the trip runs, which the launch's statement of its walk's steps
multiplies. A tensor-core product counts its logical operands' shapes
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

    def issued(self, unit: Callable[[Work], float], steps: int, programs: int) -> float:
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
        where a segment's peeled first and last steps take together two of its loop's (a walk reading
        ahead, `RingFamily._walked`'s `peeled`: its first issues two products, its last none, a segment of
        any length its loop's one a step). The trips of each loop would be a statement the count does not
        have: a unit a causal walk's masked segment takes more of (the CUDA cores' operations) is its
        segments' mean."""
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
    pallas.program_id_p, tpu.dma_wait_p, gpu.wait_gmem_to_smem_p, pjit.reshard_p})
"""What moves no data between memories and computes nothing a unit's rate prices: synchronization, layouts,
indices, and a value's reads and writes between a role's registers and its scoped refs, which its products'
operands and its copies count."""

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


class _Counting:
    """A walk of a jaxpr as warpgroup `index` along `axis` runs it: its work outside its walks (`fixed`)
    and a step of each walk (`walks`), each scalar its index's arithmetic decides held concretely; within
    `loops` loops where it starts (a conditional's branch, `cond`)."""

    def __init__(self, axis: str | None, index: int, loops: int = 0):
        self.axis, self.index = axis, index
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
            return [self.index if params["axis_name"] in (self.axis, (self.axis,)) else None]
        if primitive in _CONCRETE and values and all(_scalar(value) for value in values):
            return [_CONCRETE[primitive](*values)]
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
        elif primitive is lax.while_p:
            self.loop(params["body_jaxpr"], 1)
        elif primitive is lax.scan_p:
            self.loop(params["jaxpr"], params["length"])
        elif primitive is pallas.run_scoped_p:
            # Its operands are its body's constants; its body's inputs are the refs it allocates.
            return self.run(params["jaxpr"], [None] * len(params["jaxpr"].invars), values)
        elif primitive is derivatives.marked_call_p and isinstance(params["kind"], derivatives.WalkStep):
            if self.loops:
                self.loops[-1].marks += 1
            return self.run(params["call_jaxpr"], values)
        elif (inner := programs.called(equation)) is not None:
            return self.run(inner, values)
        elif primitive not in _FREE:
            raise Unpriced(f"no rule prices {primitive} in a kernel's body")
        return unknown

    def loop(self, body: jax_core.Jaxpr, trips: int) -> None:
        """A loop's body, `trips` trips where its trips are static (a scan's; a while's data decides its own,
        counted one): the outermost loop whose trips run walk steps (`derivatives.WalkStep` marks along the
        branches its warpgroup takes) a walk, its trip's work over its marks one step of it (`walks`); any
        other loop's work its trips' where it stands, and its marks its enclosing loop's."""
        self.loops.append(_Loop())
        self.run(body, [None] * len(body.invars))
        found = self.loops.pop()
        if self.loops:
            self.loops[-1].work = self.loops[-1].work + _times(found.work, trips)
            self.loops[-1].marks += found.marks * trips
        elif found.marks:
            self.walks.append(_times(found.work, 1 / found.marks))
        else:
            self.fixed = self.fixed + _times(found.work, trips)

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
            trial = _Counting(self.axis, self.index, loops=len(self.loops))
            trial.run(branch, operands)
            trials.append(trial)
        self.fixed = self.fixed + _envelope(trial.fixed for trial in trials)
        if any(trial.walks for trial in trials):
            self.walks.append(_envelope(_mean(trial.walks) for trial in trials if trial.walks))
        if self.loops:
            self.loops[-1].work = self.loops[-1].work + _envelope(trial.loops[-1].work for trial in trials)
            self.loops[-1].marks += max(trial.loops[-1].marks for trial in trials)
        return [None] * len(branches[0].outvars)


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
