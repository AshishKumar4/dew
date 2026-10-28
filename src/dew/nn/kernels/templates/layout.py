"""A tile's layout at the boundary between a template's own functions and the orientation a mainloop holds its
tiles in: a function written over `[rows, columns]` tiles run on tiles held the other way round, its program
taken axis for axis (`transposed`), so a score hook keeps its canonical query and key axes whichever way a
program's products lay a tile out (the attention dK/dV program's keys-first `[bk, bq]`)."""

from collections.abc import Callable
from typing import Any

import jax
import jax.extend.core as jax_core


class Untransposable(NotImplementedError):
    """A function whose program `transposed` cannot take axis for axis: an operation it has no rule for, or a
    custom derivative over an axis."""


def _axis(axis: int, rank: int) -> int:
    """`axis` of a value of `rank` held transposed: a tile's two axes swap, a vector's and a scalar's stay."""
    return 1 - axis if rank == 2 else axis


def _swapped(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(shape)) if len(shape) == 2 else shape


def _rank(var) -> int:
    return len(getattr(var.aval, "shape", ()))


def _broadcast(eqn, operand, **params):
    # The operand's own axes (a tile's held transposed) to the output's, transposed.
    rank, out_rank = _rank(eqn.invars[0]), _rank(eqn.outvars[0])
    if out_rank != 2:
        raise Untransposable(f"a broadcast to rank {out_rank}")
    dims = params["broadcast_dimensions"]
    held = [_axis(dims[_axis(axis, rank)], out_rank) for axis in range(rank)]
    return jax.lax.broadcast_in_dim(operand, _swapped(params["shape"]), tuple(held))


def _reshape(eqn, operand, **params):
    # Only between a tile with one side of length one and a vector or scalar of its entries, which keep their
    # order whichever way the tile is held.
    shapes = (eqn.invars[0].aval.shape, tuple(params["new_sizes"]))
    if any(len(shape) == 2 and min(shape) != 1 for shape in shapes):
        raise Untransposable(f"a reshape between {shapes[0]} and {shapes[1]}")
    return jax.lax.reshape(operand, _swapped(shapes[1]))


def _reduced(name: str) -> Callable:
    def rule(eqn, operand, **params):
        rank = _rank(eqn.invars[0])
        axes = tuple(_axis(axis, rank) for axis in params["axes"])
        return eqn.primitive.bind(operand, **{**params, "axes": axes})
    rule.__name__ = name
    return rule


def _along(key: str) -> Callable:
    # An operation along one axis (`key`) of a tile, its shape its operand's.
    def rule(eqn, *operands, **params):
        return eqn.primitive.bind(*operands, **{**params, key: _axis(params[key], _rank(eqn.invars[0]))})
    return rule


def _reversed(*keys: str) -> Callable:
    # An operation whose `keys` hold one entry an axis (a slice's bounds, a pad's widths).
    def rule(eqn, *operands, **params):
        if _rank(eqn.invars[0]) != 2:
            return eqn.primitive.bind(*operands, **params)
        return eqn.primitive.bind(*operands, **{**params, **{key: _swapped(tuple(params[key])) for key in keys
                                                             if params[key] is not None}})
    return rule


def _squeeze(eqn, operand, **params):
    rank = _rank(eqn.invars[0])
    return jax.lax.squeeze(operand, tuple(_axis(axis, rank) for axis in params["dimensions"]))


def _iota(eqn, **params):
    shape = params["shape"]
    return jax.lax.broadcasted_iota(params["dtype"], _swapped(shape), _axis(params["dimension"], len(shape)))


def _dynamic_slice(eqn, operand, *indices, **params):
    if len(indices) != 2:
        return eqn.primitive.bind(operand, *indices, **params)
    return jax.lax.dynamic_slice(operand, indices[::-1], _swapped(tuple(params["slice_sizes"])))


def _transpose(eqn, operand, **params):
    # A tile's transpose held transposed is the tile: the same transpose of the held one.
    return eqn.primitive.bind(operand, **params)


def _call(eqn, *operands, **params):
    # A nested program (`jit`, a call), taken axis for axis.
    closed = params["jaxpr"]
    return _run(closed.jaxpr, closed.consts, operands)


def _custom(eqn, *operands, **params):
    # A custom derivative keeps its rules only over a program of no axis, one entry at a time: bound on the
    # held tiles' entries at the shape its rules were traced at (a reshape, which keeps each entry's place
    # in their order, its outputs' as their inputs'), then given back their held shape.
    if not _axis_free(params["call_jaxpr"].jaxpr):
        raise Untransposable(f"a {eqn.primitive.name} over the tile's axes")
    traced = [var.aval.shape for var in eqn.invars]
    out = _bound(eqn, [jax.lax.reshape(value, shape) if jax.numpy.ndim(value) == 2 else value
                       for value, shape in zip(operands, traced, strict=True)], params)
    return [jax.lax.reshape(value, _swapped(value.shape)) if value.ndim == 2 else value for value in out]


RULES: dict[str, Callable] = {
    "broadcast_in_dim": _broadcast, "reshape": _reshape, "squeeze": _squeeze, "iota": _iota,
    "reduce_sum": _reduced("reduce_sum"), "reduce_max": _reduced("reduce_max"),
    "reduce_min": _reduced("reduce_min"), "reduce_prod": _reduced("reduce_prod"),
    "reduce_and": _reduced("reduce_and"), "reduce_or": _reduced("reduce_or"),
    "argmax": _reduced("argmax"), "argmin": _reduced("argmin"),
    "cumsum": _along("axis"), "cumprod": _along("axis"), "cummax": _along("axis"), "cummin": _along("axis"),
    "cumlogsumexp": _along("axis"), "concatenate": _along("dimension"),
    "slice": _reversed("start_indices", "limit_indices", "strides"), "pad": _reversed("padding_config"),
    "dynamic_slice": _dynamic_slice, "transpose": _transpose,
    "jit": _call, "pjit": _call, "closed_call": _call, "custom_jvp_call": _custom, "custom_vjp_call": _custom,
}
"""Each operation `transposed` takes that names an axis, by its primitive's name: its rule binds it on the
held operands with each axis it names swapped. Any other operation is one entry at a time (`_axis_free`)."""

CALLS = frozenset({"jit", "pjit", "closed_call", "custom_jvp_call", "custom_vjp_call"})
"""The operations of `RULES` that call a nested program, which `_axis_free` looks into."""

AXIS_PARAMS = frozenset({"axes", "axis", "dimension", "dimensions", "broadcast_dimensions", "shape",
                         "new_sizes", "start_indices", "limit_indices", "strides", "permutation",
                         "slice_sizes", "dimension_numbers", "padding_config", "sizes"})
"""The parameters an operation names an axis or a shape by: one holding any has a rule or is refused."""


def _nested(params) -> list:
    """The programs `params` holds, open or closed."""
    found = []
    for value in params.values():
        for program in value if isinstance(value, tuple) else (value,):
            if isinstance(program, jax_core.Jaxpr):
                found.append(program)
            elif isinstance(program, jax_core.ClosedJaxpr):
                found.append(program.jaxpr)
    return found


def _axis_free(jaxpr: jax_core.Jaxpr) -> bool:
    """Whether `jaxpr` acts one entry at a time: every value a scalar or of one shape, no operation naming an
    axis, and so every program it holds."""
    shapes = {getattr(var.aval, "shape", ())
              for eqn in jaxpr.eqns for var in (*eqn.invars, *eqn.outvars)} - {()}
    return (len(shapes) <= 1 and all(eqn.primitive.name not in RULES or eqn.primitive.name in CALLS
                                     for eqn in jaxpr.eqns)
            and not any(AXIS_PARAMS & eqn.params.keys() for eqn in jaxpr.eqns)
            and all(_axis_free(inner) for eqn in jaxpr.eqns for inner in _nested(eqn.params)))


def _bound(eqn, operands, params):
    """`eqn`'s operation bound on `operands` as `jax.core.eval_jaxpr` binds it (its programs as functions)."""
    return eqn.primitive.bind(*operands, **eqn.primitive.get_bind_params(params))


def _run(jaxpr: jax_core.Jaxpr, consts, operands) -> list:
    """`jaxpr`'s outputs, each tile held transposed, from its `operands` held so (`transposed`)."""
    if any(len(getattr(value, "shape", ())) == 2 for value in consts):
        raise Untransposable("a constant tile, which no axis of the call's tiles names")
    values: dict[Any, Any] = dict(zip((*jaxpr.constvars, *jaxpr.invars), (*consts, *operands), strict=True))

    def read(var):
        return var.val if isinstance(var, jax_core.Literal) else values[var]

    for eqn in jaxpr.eqns:
        operands_of = [read(var) for var in eqn.invars]
        rule = RULES.get(eqn.primitive.name)
        if rule is None:
            if AXIS_PARAMS & eqn.params.keys() or _nested(eqn.params):
                raise Untransposable(f"{eqn.primitive.name}, which names an axis")
            out = _bound(eqn, operands_of, eqn.params)
        else:
            out = rule(eqn, *operands_of, **eqn.params)
        values.update(zip(eqn.outvars, out if eqn.primitive.multiple_results else [out], strict=True))
    return [read(var) for var in jaxpr.outvars]


def transposed(fn: Callable) -> Callable:
    """`fn`, a function of `[rows, columns]` tiles (and scalars) as their own layout's callers write it, as
    one of the same tiles held transposed, `[columns, rows]`: `transposed(fn)(*(x.T for x in tiles))` is
    `fn(*tiles).T`, each of its tile outputs held transposed too. Its program is traced at the tiles' own
    shapes, so a choice it makes by their shape is the one it makes there, and run operation by operation,
    each axis an operation names swapped (`RULES`), a vector's and a scalar's unchanged; it differentiates
    as `fn` does, a custom derivative's rule kept where its program acts one entry at a time. An operation it
    has no rule for raises `Untransposable` as it is traced."""
    def held(*operands):
        flat, tree = jax.tree.flatten(operands)
        canonical = [jax.ShapeDtypeStruct(_swapped(jax.numpy.shape(value)), jax.numpy.result_type(value))
                     for value in flat]
        closed, out_tree = jax.make_jaxpr(lambda *values: fn(*jax.tree.unflatten(tree, values)),
                                          return_shape=True)(*canonical)
        outs = _run(closed.jaxpr, closed.consts, flat)
        return jax.tree.unflatten(jax.tree.structure(out_tree), outs)

    return held
