"""The narrow copies of the parameters the forward reads.

A model that computes in bf16 over fp32 parameters casts each weight to
bf16 in the forward, every step: on CUDA a kernel of its own per weight
that reads four bytes and writes two, and in the backward a second that
widens the weight's bf16 gradient to fp32 before the optimizer reads it.
Instead the update writes each such weight's bf16 copy (`TrainState.compute`)
from the new fp32 value, the forward reads the copy, and the gradient
reaches the update in bf16, which widens it as it reads it. The copy is the
cast the forward made, so the forward and the gradient are bitwise the
same; the update's fused arithmetic rounds otherwise (an fp32 multiply-add
contracted or not), every accumulation still fp32.

Which weights: the parameters whose one use in the loss is that cast
(`narrowed_paths`). A weight used twice, as a tied embedding's table is,
sums its cotangents where it is used; read through a bf16 copy those sums
would be bf16 where they are fp32 now, so such a weight keeps its fp32
read. A call the loss passes the weight into unchanged (`pjit`, a
rematerialized region, a scan over stacked layers) is followed, and so is
the stack a scanned run builds of its layers' weights, whose cast is the
stack of their casts; any other use, a custom VJP's included, keeps it.

A step that pools several microbatches into one update, by accumulation or
a batch read in parts, reads one set of copies through its window and
rewrites them as it commits. Each microbatch's gradient is widened as it is
pooled, so the window sums in fp32 as it does without copies.

Where: on every GPU, where XLA runs each such cast as a kernel of its own
apart from the matmul; a generation's row names `KERNELS['narrow_copies']`
'off' only where a measurement says the copies cost more than they save
(docs/performance.md). A TPU fuses the cast into the matmul, so the
copy there only adds writes. The gradients are the cast's on split meshes
as partitioned for the CPU (tests/test_narrow.py); a multi-GPU run over
NCCL has not been measured, so its first should compare a few steps
against copies off.
"""
import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend import core

from dew.batch import FROZEN, Variables, merge, select

_PASSED_THROUGH = ("jit", "pjit", "closed_call", "core_call", "remat2", "checkpoint")


def _inner(eqn, index: int):
    """The callee jaxpr of `eqn` and the variable it binds the `index`th
    operand to, where the callee sees each iteration's own slice or the
    operand itself; else None."""
    name = eqn.primitive.name
    inner = eqn.params.get("jaxpr", eqn.params.get("call_jaxpr"))
    if inner is None or name not in (*_PASSED_THROUGH, "scan") or index >= len(inner.invars):
        return None
    # A scan's constants and carries are read by every iteration, whose
    # cotangents it sums in the operand's dtype; only its stacked operands
    # are sliced, one slice an iteration, the leading axis gone.
    if name == "scan":
        outer, bound = eqn.invars[index].aval, inner.invars[index].aval
        if not (isinstance(outer, ShapedArray) and isinstance(bound, ShapedArray)
                and outer.shape == (eqn.params["length"], *bound.shape)):
            return None
    return inner, inner.invars[index]


_STACKING = ("stack", "reshape", "expand_dims", "squeeze", "broadcast_in_dim", "concatenate")
"""The ops a stack of layers builds its operand with, each element of the result one of its
operands': a cast of the result is the stack of its operands' casts, and its cotangent's slices are
theirs."""


def _uses(var, jaxpr) -> list[tuple]:
    return [(eqn, index) for eqn in jaxpr.eqns for index, operand in enumerate(eqn.invars) if operand is var]


def _read_once(var, jaxpr) -> bool:
    """Whether `var` is read by one equation of `jaxpr` and returned by none."""
    return len(_uses(var, jaxpr)) == 1 and var not in jaxpr.outvars


def _cast_once(var, jaxpr, weights: frozenset) -> jnp.dtype | None:
    """The narrower float dtype `var`'s one use in `jaxpr` casts it to, or None: through a call
    that passes it on (`_inner`), and through the stack a scanned run builds of its layers'
    weights (`_STACKING`), whose stacked operand the scan slices and casts. A stack of several is
    followed only where every operand is a weight (`weights`, the variables in `jaxpr` the
    parameters are bound to) of the one dtype read nowhere else, each copied as the others are: the
    stack takes one dtype, which a copy of some and not others would split."""
    if not _read_once(var, jaxpr):
        return None
    eqn, index = _uses(var, jaxpr)[0]
    if eqn.primitive.name == "convert_element_type":
        dtype = jnp.dtype(eqn.params["new_dtype"])
        narrower = jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize < var.aval.dtype.itemsize
        return dtype if narrower else None
    if eqn.primitive.name in _STACKING and len(eqn.outvars) == 1:
        stacked = eqn.outvars[0]
        operands = [operand for operand in eqn.invars if isinstance(operand, core.Var)]
        sized = [operand.aval for operand in operands if isinstance(operand.aval, ShapedArray)]
        if not isinstance(stacked.aval, ShapedArray) or len(sized) != len(operands):
            return None
        if len(operands) > 1 and not all(operand in weights and aval.dtype == stacked.aval.dtype
                                         and _read_once(operand, jaxpr)
                                         for operand, aval in zip(operands, sized, strict=True)):
            return None
        # A broadcast that repeats elements would read one weight in many places, whose
        # cotangents it sums in the weight's dtype.
        if stacked.aval.size != sum(aval.size for aval in sized):
            return None
        return _cast_once(stacked, jaxpr, weights | {stacked})
    called = _inner(eqn, index)
    if called is None:
        return None
    # The weights the callee binds: each of the call's operands that is a weight read by the call
    # alone, where it is passed on unchanged.
    inner = frozenset(bound[1] for position, operand in enumerate(eqn.invars)
                      if isinstance(operand, core.Var) and operand in weights and _read_once(operand, jaxpr)
                      and (bound := _inner(eqn, position)) is not None)
    return _cast_once(called[1], called[0], inner)


def narrowed_paths(loss, params: Variables, *rest) -> dict[tuple[str, ...], jnp.dtype]:
    """The paths under `params` (relative to it) whose one use in
    `loss(params, *rest)` is a cast to a narrower float dtype, with that
    dtype. The arguments may be shapes alone."""
    closed = jax.make_jaxpr(loss)(params, *rest)
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    weights = frozenset(closed.jaxpr.invars[:len(leaves)])
    paths = {}
    for (path, leaf), var in zip(leaves, closed.jaxpr.invars[:len(leaves)], strict=True):
        if isinstance(var, core.Var) and jnp.issubdtype(leaf.dtype, jnp.floating):
            dtype = _cast_once(var, closed.jaxpr, weights)
            if dtype is not None:
                paths[tuple(key.key for key in path)] = dtype
    return paths


def narrowed(variables: Variables, paths: dict[tuple[str, ...], jnp.dtype]) -> Variables:
    """The `params` leaves at `paths` cast to their dtypes, nested as in
    `variables` (`compute`'s tree)."""
    chosen = select(variables, lambda path: path[0] == "params" and path[1:] in paths)
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf.astype(paths[tuple(key.key for key in path)[1:]]), chosen)


def forward_variables(variables: Variables, compute: Variables | None) -> Variables:
    """The variables the forward reads: the narrow copies in place of their parameters."""
    return variables if compute is None else merge(variables, compute)


def narrow_copies(objective, state, batch, model_batch, *, composed: bool) -> dict:
    """The parameters a step reads through narrow copies, by path under `params` (`narrowed_paths`):
    on every GPU but one whose row measured them off, for a step the trainer composes (`composed`),
    without a loss scale, whose state holds no frozen collection. `model_batch` is the trainer's
    batch as the model reads it."""
    from dew.nn.kernels.generation import backend, measured_kernel
    from dew.objectives.base import Step
    from dew.optim import clips_qk
    from dew.training.transaction import with_ema

    if (measured_kernel('narrow_copies', 'on' if backend() == 'gpu' else 'off') == 'off'
            or not composed or FROZEN in state.variables):
        return {}
    variables = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)

    def loss(params, rest, batch, microstep, key, step, ema):
        # As the trainer's loss shape traces it, the averaged weights the step hands the objective
        # included: a parameter read there is used twice.
        variables = {**rest, "params": params}
        return objective._loss(variables, model_batch(batch),
                               Step(microstep, jax.random.fold_in(key, step), with_ema(variables, ema),
                                    qk_stats=clips_qk(state.opt_state)))

    rest = {name: tree for name, tree in variables.items() if name != "params"}
    return narrowed_paths(loss, variables["params"], rest, batch, state.microstep, state.key, state.step,
                          state.ema)
