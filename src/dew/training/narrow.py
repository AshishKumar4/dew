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
rematerialized region, a scan over stacked layers) is followed; any other
use, a custom VJP's included, keeps it.

Where: `KERNELS['narrow_copies']`, the CUDA generations it was measured to
pay on (docs/performance.md). A TPU fuses the cast into the matmul, so the
copy there only adds writes. The gradients are the cast's on split meshes
as partitioned for the CPU (tests/test_narrow.py); a multi-GPU run over
NCCL has not been measured, so its first should compare a few steps
against copies off.
"""
import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend import core

from dew.objectives.base import Variables, merge, select

_PASSED_THROUGH = ("jit", "pjit", "closed_call", "core_call", "remat2", "checkpoint")


def _inner(eqn, index: int):
    """The callee jaxpr of `eqn` and the variable it binds the `index`th
    operand to, where the callee sees each iteration's own slice or the
    operand itself; else None."""
    name = eqn.primitive.name
    inner = eqn.params.get("jaxpr", eqn.params.get("call_jaxpr"))
    if isinstance(inner, core.ClosedJaxpr):
        inner = inner.jaxpr
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


def _cast_once(var, jaxpr) -> jnp.dtype | None:
    """The narrower float dtype `var`'s one use in `jaxpr` casts it to, or None."""
    uses = [(eqn, index) for eqn in jaxpr.eqns for index, operand in enumerate(eqn.invars) if operand is var]
    if len(uses) != 1 or var in jaxpr.outvars:
        return None
    eqn, index = uses[0]
    if eqn.primitive.name == "convert_element_type":
        dtype = jnp.dtype(eqn.params["new_dtype"])
        narrower = jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize < var.aval.dtype.itemsize
        return dtype if narrower else None
    called = _inner(eqn, index)
    return None if called is None else _cast_once(called[1], called[0])


def narrowed_paths(loss, params: Variables, *rest) -> dict[tuple[str, ...], jnp.dtype]:
    """The paths under `params` (relative to it) whose one use in
    `loss(params, *rest)` is a cast to a narrower float dtype, with that
    dtype. The arguments may be shapes alone."""
    closed = jax.make_jaxpr(loss)(params, *rest)
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    paths = {}
    for (path, leaf), var in zip(leaves, closed.jaxpr.invars[:len(leaves)], strict=True):
        if isinstance(var, core.Var) and jnp.issubdtype(leaf.dtype, jnp.floating):
            dtype = _cast_once(var, closed.jaxpr)
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
