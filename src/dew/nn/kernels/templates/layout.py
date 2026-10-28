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
    # The primitive's own indices, already normalized by the call that formed it: reversed, not again.
    if len(indices) != 2:
        return eqn.primitive.bind(operand, *indices, **params)
    sizes = _swapped(tuple(params["slice_sizes"]))
    return eqn.primitive.bind(operand, *indices[::-1], **{**params, "slice_sizes": sizes})


def _transpose(eqn, operand, **params):
    # A tile's transpose held transposed is the tile: the same transpose of the held one.
    return eqn.primitive.bind(operand, **params)


def _call(eqn, *operands, **params):
    # A nested program (`jit`, a call), taken axis for axis.
    closed = params["jaxpr"]
    return _run(closed.jaxpr, closed.consts, operands)


def _custom(eqn, *operands, **params):
    # A custom derivative, its rule unread but traced: bound on the held tiles' entries at the shape it was
    # traced at (a reshape keeps each entry's place in their order, its outputs' as its inputs'), where its
    # program and its derivative's act one entry at a time (`_axis_free`), so any order of the entries is
    # theirs.
    traced = [var.aval for var in eqn.invars]

    def call(*values):
        return eqn.primitive.bind(*values, **eqn.primitive.get_bind_params(params))

    floating = [index for index, aval in enumerate(traced)
                if jax.numpy.issubdtype(aval.dtype, jax.numpy.inexact)]

    def pulled(*values):
        outs, pull = jax.vjp(lambda *chosen: call(*[chosen[floating.index(index)] if index in floating
                                                     else value for index, value in enumerate(values)]),
                             *(values[index] for index in floating))
        return pull(jax.tree.map(jax.numpy.ones_like, outs))

    shapes = {aval.shape for aval in traced} - {()}
    programs = [params["call_jaxpr"].jaxpr, jax.make_jaxpr(pulled)(*traced).jaxpr]
    if len(shapes) > 1 or not all(_axis_free(program, *shapes) for program in programs):
        raise Untransposable(f"a {eqn.primitive.name} whose program or derivative reads an axis")
    out = call(*[jax.lax.reshape(value, aval.shape) if aval.ndim == 2 else value
                 for value, aval in zip(operands, traced, strict=True)])
    return [jax.lax.reshape(value, _swapped(value.shape)) if value.ndim == 2 else value for value in out]


def _axis_free(jaxpr: jax_core.Jaxpr, shape: tuple[int, ...] = ()) -> bool:
    """Whether `jaxpr` acts one entry at a time: every value a scalar or of `shape`, every operation one of
    `ENTRYWISE`, a scalar's broadcast or a call of a program so."""
    for eqn in jaxpr.eqns:
        if any(getattr(var.aval, "shape", ()) not in ((), shape) for var in (*eqn.invars, *eqn.outvars)):
            return False
        nested = list(jax_core.jaxprs_in_params(eqn.params))
        name = eqn.primitive.name
        if nested:
            # A call, or the custom derivative's own primal (its pullback's trace evaluates it).
            if name not in (*CALLS, "custom_jvp_call", "custom_vjp_call") or not all(
                    _axis_free(inner, shape) for inner in nested):
                return False
        elif name not in ENTRYWISE and not (name == "broadcast_in_dim" and eqn.invars[0].aval.shape == ()):
            return False
    return True


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
held operands with each axis it names swapped."""

ENTRYWISE = frozenset({
    "abs", "acos", "acosh", "add", "and", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "ceil", "clamp",
    "convert_element_type", "copy", "copy_p", "cos", "cosh", "div", "eq", "erf", "erf_inv", "erfc", "exp",
    "exp2", "expm1", "floor", "ge", "gt", "integer_pow", "is_finite", "le", "lgamma", "digamma", "log",
    "log1p", "logistic", "lt", "max", "min", "mul", "ne", "neg", "nextafter", "not", "or", "pow",
    "population_count", "rem", "round", "rsqrt", "select_n", "shift_left", "shift_right_arithmetic",
    "shift_right_logical", "sign", "sin", "sinh", "sqrt", "square", "stop_gradient", "sub", "tan", "tanh",
    "xor",
})
"""The operations of one entry at a time, whose operands are each a scalar or a value of their output's
shape: bound on held tiles unchanged. An operation neither these nor `RULES` take is refused."""

CALLS = frozenset({"jit", "pjit", "closed_call"})
"""The operations that call a nested program as it is (`_axis_free` looks into them)."""


def _run(jaxpr: jax_core.Jaxpr, consts, operands) -> list:
    """`jaxpr`'s outputs, each tile held transposed, from its `operands` held so (`transposed`)."""
    if any(len(getattr(value, "shape", ())) == 2 for value in consts):
        raise Untransposable("a constant tile, which no axis of the call's tiles names")
    values: dict[Any, Any] = dict(zip((*jaxpr.constvars, *jaxpr.invars), (*consts, *operands), strict=True))

    def read(var):
        if not isinstance(var, jax_core.Literal):
            return values[var]
        if len(getattr(var.aval, "shape", ())) == 2:
            raise Untransposable("a constant tile, which no axis of the call's tiles names")
        return var.val

    for eqn in jaxpr.eqns:
        if any(_rank(var) > 2 for var in (*eqn.invars, *eqn.outvars)):
            raise Untransposable(f"{eqn.primitive.name} of a value past a tile's two axes")
        operands_of = [read(var) for var in eqn.invars]
        name = eqn.primitive.name
        if name in RULES:
            out = RULES[name](eqn, *operands_of, **eqn.params)
        elif name in ENTRYWISE:
            out = eqn.primitive.bind(*operands_of, **eqn.primitive.get_bind_params(eqn.params))
        else:
            raise Untransposable(f"{name}, which no layout rule takes")
        values.update(zip(eqn.outvars, out if eqn.primitive.multiple_results else [out], strict=True))
    return [read(var) for var in jaxpr.outvars]


def transposed(fn: Callable) -> Callable:
    """`fn`, a function of `[rows, columns]` tiles (and scalars) as their own layout's callers write it, as
    one of the same tiles held transposed, `[columns, rows]`: `transposed(fn)(*(x.T for x in tiles))` is
    `fn(*tiles).T`, each of its tile outputs held transposed too. Its program is traced at the tiles' own
    shapes, so a choice it makes by their shape is the one it makes there, and run operation by operation,
    each axis an operation names swapped (`RULES`), one entry at a time unchanged (`ENTRYWISE`); it
    differentiates as `fn` does, a custom derivative's rule kept where it and its program act one entry at
    a time. Any other operation, a value past two axes or a constant tile raises `Untransposable` as it is
    traced."""
    def held(*operands):
        flat, tree = jax.tree.flatten(operands)
        # Each operand's type as the caller holds it (a Python scalar's weak), its tile's sides swapped back.
        canonical = [jax.ShapeDtypeStruct(_swapped(aval.shape), aval.dtype, weak_type=aval.weak_type)
                     for aval in map(jax.typeof, flat)]
        closed, out_tree = jax.make_jaxpr(lambda *values: fn(*jax.tree.unflatten(tree, values)),
                                          return_shape=True)(*canonical)
        outs = _run(closed.jaxpr, closed.consts, flat)
        return jax.tree.unflatten(jax.tree.structure(out_tree), outs)

    return held
