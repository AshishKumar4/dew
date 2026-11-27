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
apart from the matmul; `chosen(narrow_copies='off')` turns them off for a
run that measures them costing more than they save (docs/performance.md).
A TPU fuses the cast into the matmul, so the copy there only adds writes.
The gradients are the cast's on split meshes as partitioned for the CPU
(tests/test_narrow.py); a multi-GPU run over NCCL has not been measured, so
its first should compare a few steps against copies off.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax._src.interpreters import partial_eval
from jax.core import ShapedArray
from jax.extend import core

from dew.batch import FROZEN, Variables, merge, select, under_roots
from dew.nn.kernels.options import OuterProgram, owned_program
from dew.nn.programs import branches, called, own_derivative, scan_body
from dew.nn.protocols import COPIES_SCOPE


def _inner(eqn, index: int):
    """The callee jaxpr of `eqn` and the variable it binds the `index`th
    operand to, where the callee sees each iteration's own slice or the
    operand itself; else None. A narrowed operand reaches a call's program as it is (`called`) through every
    call but a custom derivative's (`own_derivative`), whose derivative rule, not its primal's program, makes
    its operand's cotangent, so a cast folded into its primal leaves its cotangent the rule's."""
    scanned = scan_body(eqn) if eqn.primitive is jax.lax.scan_p else None
    inner = scanned if scanned is not None or own_derivative(eqn.primitive) else called(eqn)
    if inner is None or index >= len(inner.invars):
        return None
    # A scan's constants and carries are read by every iteration, whose
    # cotangents it sums in the operand's dtype; only its stacked operands
    # are sliced, one slice an iteration, the leading axis gone. A scan of one
    # iteration (a loop's one later pass, scanned where a call's count runs it,
    # row 11951's design) reads its constants and carries once.
    if scanned is not None:
        outer, bound = eqn.invars[index].aval, inner.invars[index].aval
        if not (isinstance(outer, ShapedArray) and isinstance(bound, ShapedArray)):
            return None
        once = eqn.params["length"] == 1 and outer.shape == bound.shape
        if not once and outer.shape != (eqn.params["length"], *bound.shape):
            return None
    return inner, inner.invars[index]


_STACKING = (jax.lax.stack_p, jax.lax.reshape_p, jax.lax.squeeze_p, jax.lax.broadcast_in_dim_p,
             jax.lax.concatenate_p)
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
    if eqn.primitive is jax.lax.convert_element_type_p:
        dtype = jnp.dtype(eqn.params["new_dtype"])
        narrower = jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize < var.aval.dtype.itemsize
        return dtype if narrower else None
    if eqn.primitive in _STACKING and len(eqn.outvars) == 1:
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
    if eqn.primitive is jax.lax.cond_p and index > 0:
        # A conditional's operand reaches each branch as it is, and runs where its branch is taken (a loop's
        # pass a call's count runs, `loop_passes`): cast alike by every branch that reads it, it is that
        # cast, and its cotangent the taken branch's (row 11951's design: one program at a loop's cap).
        found = set()
        for branch in branches(eqn):
            bound = branch.jaxpr.invars[index - 1]
            if _uses(bound, branch.jaxpr) or bound in branch.jaxpr.outvars:
                found.add(_cast_once(bound, branch.jaxpr, frozenset()))
        return found.pop() if len(found) == 1 and None not in found else None
    callee = _inner(eqn, index)
    if callee is None:
        return None
    # The weights the callee binds: each of the call's operands that is a weight read by the call
    # alone, where it is passed on unchanged.
    inner = frozenset(bound[1] for position, operand in enumerate(eqn.invars)
                      if isinstance(operand, core.Var) and operand in weights and _read_once(operand, jaxpr)
                      and (bound := _inner(eqn, position)) is not None)
    return _cast_once(callee[1], callee[0], inner)


def narrowed_paths(loss, params: Variables, *rest) -> dict[tuple[str, ...], jnp.dtype]:
    """The paths under `params` (relative to it) whose one use in
    `loss(params, *rest)` is a cast to a narrower float dtype, with that
    dtype. The arguments may be shapes alone."""
    return program_reads(jax.make_jaxpr(loss)(params, *rest), params).casts


class Reads(NamedTuple):
    """How a program reads the leaves of a tree it takes first (`program_reads`), each by its path: those
    whose one use is a cast to a narrower float dtype, with that dtype (`casts`), and every one it reads at
    all (`read`), in its live equations alone."""

    casts: dict[tuple[str, ...], jnp.dtype]
    read: frozenset[tuple[str, ...]]


def program_reads(closed: core.Jaxpr, tree: Variables, within: tuple[str, ...] = ()) -> Reads:
    """How `closed`, a program whose first inputs are `tree`'s leaves, reads those under `within` (relative
    to it), its other inputs read as it reads them (`Reads`). Uses are counted in its live equations alone
    (`partial_eval.dce_jaxpr`): a read whose value nothing takes (a selector's probe that declined its cast)
    is none, the program as it runs reading the weight once, and a leaf it reads nowhere is no part of what
    it computes."""
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    # The program its uses are counted in, its dead equations dropped; JAX's dead-code elimination keeps an
    # effectful equation whatever reads it. Only the counting reads it: the step runs `closed` as traced.
    live, _ = partial_eval.dce_jaxpr(closed.jaxpr, used_outputs=[True] * len(closed.jaxpr.outvars),
                                     instantiate=True)
    # Every leaf bound to its input in order, those under `within` kept where a mapping reaches them, as a
    # copy's overlay does (`narrowed`, `batch.select`): a leaf of another collection, or one reached through
    # an attribute or an index, is read as the program reads it and named nowhere (row 11730: a packed
    # checkpoint's `quantization`, NvFp4's attributes, raised before a step).
    held = [(tuple(key.key for key in path)[len(within):], leaf, var)
            for (path, leaf), var in zip(leaves, live.invars[:len(leaves)], strict=True)
            if under_roots(path, (within,))
            and all(isinstance(key, jax.tree_util.DictKey) for key in path)]
    weights = frozenset(var for _, _, var in held)
    casts = {}
    for path, leaf, var in held:
        if isinstance(var, core.Var) and jnp.issubdtype(leaf.dtype, jnp.floating):
            dtype = _cast_once(var, live, weights)
            if dtype is not None:
                casts[path] = dtype
    # Every variable the live equations read, in one pass over them.
    used = {id(operand) for eqn in live.eqns for operand in eqn.invars}
    returned = {id(var) for var in live.outvars}
    return Reads(casts, frozenset(path for path, _, var in held if id(var) in used | returned))


def narrowed(variables: Variables, paths: dict[tuple[str, ...], jnp.dtype]) -> Variables:
    """The `params` leaves at `paths` cast to their dtypes, nested as in
    `variables` (`compute`'s tree)."""
    chosen = select(variables, lambda path: path[0] == "params" and path[1:] in paths)
    with jax.named_scope(COPIES_SCOPE):
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: leaf.astype(paths[tuple(key.key for key in path)[1:]]), chosen)


def forward_variables(variables: Variables, compute: Variables | None) -> Variables:
    """The variables the forward reads: the narrow copies in place of their parameters."""
    return variables if compute is None else merge(variables, compute)


def narrow_copies(state) -> bool:
    """Whether a step the trainer composes over `state` reads narrow copies, of the parameters whose one use
    is a narrowing cast (`Reads.casts`): on every GPU but one whose row measured them off, for a state that
    holds no frozen collection. They are read off the loss its first attempt reads as the trainer traced it
    for its shapes (`state_reads`; `dew.training.transaction.traced_loss`: its update a window of
    microbatches, each step read in its split, its committed updates among its inputs, the state's
    variables its first inputs), in place of a trace of its own, so a parameter the step computes on before
    its cast (a DEX lambda's anneal) is not copied: a trace of a Qwen3.5-4B step's loss is seconds of its
    compile (row 4260)."""
    from dew.nn.kernels.generation import Choice, backend, chosen_kernel

    return (chosen_kernel(Choice.NARROW_COPIES, 'on' if backend() == 'gpu' else 'off') == 'on'
            and FROZEN not in state.variables)


def state_reads(state, traced: core.Jaxpr) -> Reads:
    """How `traced`, a step's loss whose first inputs are `state`'s variables, reads its parameters
    (`program_reads` under `params`): once, for its narrow copies and for those its window holds."""
    variables = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)
    return program_reads(traced, variables, ("params",))


def held_copies(compute: Variables | None) -> dict[tuple[str, ...], jnp.dtype]:
    """The narrow copies a state's `compute` holds (`narrowed`'s tree), by path under `params`, each its
    dtype: the plan of the program that made them, none for a state that holds none."""
    if compute is None:
        return {}
    return {tuple(key.key for key in path)[1:]: jnp.dtype(leaf.dtype)
            for path, leaf in jax.tree_util.tree_flatten_with_path(compute)[0]}


def holds_copies(compute: Variables | None, copies: dict[tuple[str, ...], jnp.dtype]) -> bool:
    """Whether a state's `compute` holds the narrow copies `copies` plans (`held_copies`): a step reads the
    state with its own, made where it does not (`dew.training.compiler.StepCompiler.runner`), and a plan's
    fit counts them built where it does not (`dew.training._candidates`)."""
    return held_copies(compute) == {path: jnp.dtype(dtype) for path, dtype in copies.items()}


def narrowing(copies: dict[tuple[str, ...], jnp.dtype], out_shardings) -> OuterProgram:
    """The program a step's state runs to make its narrowed copies of its weights (`narrowed`), placed by
    `out_shardings`, compiled through Dew's compile point: the one program a run's first step runs and a
    run compiled ahead compiles for it (`dew.training.ahead.started`)."""
    return owned_program(lambda variables: narrowed(variables, copies), out_shardings=out_shardings)
