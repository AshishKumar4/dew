"""A tile's layout at the boundary between a template's own functions and the orientation a mainloop holds its
tiles in: a function written over `[rows, columns]` tiles run on tiles held the other way round, its program
taken axis for axis (`transposed`), so a score hook keeps its canonical query and key axes whichever way a
program's products lay a tile out (the attention dK/dV program's keys-first `[bk, bq]`)."""

from collections.abc import Callable
from typing import Any

import jax
import jax.extend.core as jax_core

from dew.nn import programs
from dew.nn.derivatives import KernelGlue, contracted, declared_derivative, declared_tangent
from dew.nn.programs import called, cumulative, equations, own_derivative, purely_called, reduction


class Untransposable(NotImplementedError):
    """A function whose program `transposed` cannot take axis for axis: an operation it has no rule for, or a
    custom derivative over an axis."""


def _axis(axis: int, rank: int) -> int:
    """`axis` of a value of `rank` held transposed: a tile's two axes swap, a vector's and a scalar's stay."""
    return 1 - axis if rank == 2 else axis


def _swapped(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(shape)) if len(shape) == 2 else shape


RULES: dict[jax_core.Primitive, Callable] = {}
"""Each operation `transposed` takes that names an axis, by its primitive object: its rule binds it on the
held operands with each axis it names swapped. A reduction and a cumulative one take their classes' rules
(`dew.nn.programs.reduction`, `cumulative`), a call that only runs its program is taken axis for axis, a
custom derivative's by its program and its rule; an entrywise operation (`entrywise`) is bound on held
tiles unchanged; one none of these take is refused."""


def _rule(*operations: jax_core.Primitive) -> Callable[[Callable], Callable]:
    """Registers a layout rule for `operations`."""
    def registered(rule: Callable) -> Callable:
        RULES.update(dict.fromkeys(operations, rule))
        return rule
    return registered


def _rank(var) -> int:
    return len(getattr(var.aval, "shape", ()))


@_rule(jax.lax.broadcast_in_dim_p)
def _broadcast(eqn, operand, **params):
    # The operand's own axes (a tile's held transposed) to the output's, transposed.
    rank, out_rank = _rank(eqn.invars[0]), _rank(eqn.outvars[0])
    if out_rank != 2:
        raise Untransposable(f"a broadcast to rank {out_rank}")
    dims = params["broadcast_dimensions"]
    held = [_axis(dims[_axis(axis, rank)], out_rank) for axis in range(rank)]
    return jax.lax.broadcast_in_dim(operand, _swapped(params["shape"]), tuple(held))


@_rule(jax.lax.reshape_p)
def _reshape(eqn, operand, **params):
    # Only between a tile with one side of length one and a vector or scalar of its entries, which keep their
    # order whichever way the tile is held.
    shapes = (eqn.invars[0].aval.shape, tuple(params["new_sizes"]))
    if any(len(shape) == 2 and min(shape) != 1 for shape in shapes):
        raise Untransposable(f"a reshape between {shapes[0]} and {shapes[1]}")
    return jax.lax.reshape(operand, _swapped(shapes[1]))


def _reduced(eqn, operand, **params):
    # A reduction (`dew.nn.programs.reduction`) over the axes it names.
    rank = _rank(eqn.invars[0])
    axes = tuple(_axis(axis, rank) for axis in params["axes"])
    return eqn.primitive.bind(operand, **{**params, "axes": axes})


def _along(key: str) -> Callable:
    # An operation along one axis (`key`) of a tile, its shape its operand's.
    def rule(eqn, *operands, **params):
        return eqn.primitive.bind(*operands, **{**params, key: _axis(params[key], _rank(eqn.invars[0]))})
    return rule


_cumulated = _along("axis")
RULES[jax.lax.concatenate_p] = _along("dimension")


def _reversed(*keys: str) -> Callable:
    # An operation whose `keys` hold one entry an axis (a slice's bounds, a pad's widths).
    def rule(eqn, *operands, **params):
        if _rank(eqn.invars[0]) != 2:
            return eqn.primitive.bind(*operands, **params)
        return eqn.primitive.bind(*operands, **{**params, **{key: _swapped(tuple(params[key])) for key in keys
                                                             if params[key] is not None}})
    return rule


RULES[jax.lax.slice_p] = _reversed("start_indices", "limit_indices", "strides")
RULES[jax.lax.pad_p] = _reversed("padding_config")


@_rule(jax.lax.squeeze_p)
def _squeeze(eqn, operand, **params):
    rank = _rank(eqn.invars[0])
    return jax.lax.squeeze(operand, tuple(_axis(axis, rank) for axis in params["dimensions"]))


@_rule(jax.lax.iota_p)
def _iota(eqn, **params):
    shape = params["shape"]
    return jax.lax.broadcasted_iota(params["dtype"], _swapped(shape), _axis(params["dimension"], len(shape)))


@_rule(jax.lax.dynamic_slice_p)
def _dynamic_slice(eqn, operand, *indices, **params):
    # The primitive's own indices, already normalized by the call that formed it: reversed, not again.
    if len(indices) != 2:
        return eqn.primitive.bind(operand, *indices, **params)
    sizes = _swapped(tuple(params["slice_sizes"]))
    return eqn.primitive.bind(operand, *indices[::-1], **{**params, "slice_sizes": sizes})


@_rule(jax.lax.transpose_p)
def _transpose(eqn, operand, **params):
    # A tile's transpose held transposed is the tile: the same transpose of the held one.
    return eqn.primitive.bind(operand, **params)


def _program(eqn, read=called) -> jax_core.Jaxpr:
    """The program a call `eqn` runs on its operands one for one (`read`: `called`, or `purely_called` for a
    call that does nothing else), refused where it runs none so."""
    program = read(eqn)
    if program is None:
        raise Untransposable(f"a {eqn.primitive.name} whose program is not run on its operands one for one")
    return program


def _call(eqn, *operands, **params):
    # A nested program (`jit`, a call that only runs it), taken axis for axis.
    closed = _program(eqn, purely_called)
    return _run(closed.jaxpr, closed.consts, operands)


def _custom(eqn, *operands, **params):
    # A custom derivative whose program and rule act one entry at a time (`_axis_free`): its program and its
    # rule, both traced at its own shapes, run entry for entry on the held tiles (`_run`), the rule kept the
    # call's own kind, so no tile is reshaped (Mosaic GPU lowers none): a custom JVP's tangent program, which
    # reverse mode transposes and a higher order differentiates (`_held_jvp`), a custom VJP's pullback.
    traced = [var.aval for var in eqn.invars]
    cotangents = [var.aval for var in eqn.outvars]

    def call(*values):
        return eqn.primitive.bind(*values, **eqn.primitive.get_bind_params(params))

    floating = [index for index, aval in enumerate(traced)
                if jax.numpy.issubdtype(aval.dtype, jax.numpy.inexact)]
    primal = _program(eqn)
    if eqn.primitive is jax_core.primitives.custom_jvp_call_p:
        return _held_jvp(primal, call, traced, floating)(*operands)

    def pulled(*values_and_cotangents):
        values, given = values_and_cotangents[:len(traced)], values_and_cotangents[len(traced):]
        _, pull = jax.vjp(lambda *chosen: call(*[chosen[floating.index(index)] if index in floating
                                                  else value for index, value in enumerate(values)]),
                          *(values[index] for index in floating))
        return pull(list(given))

    shapes = {aval.shape for aval in (*traced, *cotangents)} - {()}
    pullback = jax.make_jaxpr(pulled)(*traced, *cotangents)
    if len(shapes) > 1 or not all(_axis_free(program.jaxpr, *shapes) and not _tiles_in(program)
                                  for program in (primal, pullback)):
        raise Untransposable(f"a {eqn.primitive.name} whose program or derivative reads an axis or a "
                             "constant tile")

    def holding(*values):
        return _run(primal.jaxpr, primal.consts, values)

    def held_pullback(values, given):
        grads = iter(_run(pullback.jaxpr, pullback.consts, [*values, *given]))
        return tuple(next(grads) if index in floating else None for index in range(len(values)))

    held = contracted("test_a_transposed_hooks_custom_vjp_holds_every_reverse_order")(
        declared_derivative(KernelGlue(), holding, lambda *values: (held(*values), values), held_pullback))
    return held(*operands)


def _held_jvp(primal, call, traced: list, floating: list[int]) -> Callable:
    """A custom JVP call (`call`, at `traced` avals, its inexact operands at `floating`) held entry for entry:
    its program and its JVP's, both traced at its own shapes, run on the held tiles (`_run`); the JVP is the
    call's own rule, so forward mode, its transpose and every higher order meet that rule as on the call."""
    def tangent(*values_and_tangents):
        values, tangents = values_and_tangents[:len(traced)], values_and_tangents[len(traced):]

        def chosen(*picked):
            return call(*[picked[floating.index(index)] if index in floating else value
                          for index, value in enumerate(values)])

        picked = tuple(values[index] for index in floating)
        outputs, output_tangents = jax.jvp(chosen, picked, tuple(tangents))
        return [*outputs, *output_tangents]

    tangents = jax.make_jaxpr(tangent)(*traced, *(traced[index] for index in floating))
    shapes = {aval.shape for aval in traced} - {()}
    if len(shapes) > 1 or not all(_axis_free(program.jaxpr, *shapes) and not _tiles_in(program)
                                  for program in (primal, tangents)):
        raise Untransposable("a custom_jvp_call whose program or derivative reads an axis or a constant tile")

    def holding(*values):
        return _run(primal.jaxpr, primal.consts, values)

    def held_tangent(values, directions):
        both = _run(tangents.jaxpr, tangents.consts, [*values, *(directions[index] for index in floating)])
        return both[:len(both) // 2], both[len(both) // 2:]

    return contracted("test_a_transposed_hooks_custom_derivative_holds_every_order")(
        declared_tangent(KernelGlue(), holding, held_tangent))


def _tiles_in(closed) -> bool:
    """Whether a closed program holds a constant tile, among its constants or its operations' literals, its
    nested programs' too (`dew.nn.programs.equations`)."""
    return (any(len(getattr(value, "shape", ())) == 2 for value in closed.consts)
            or any(isinstance(var, jax_core.Literal) and _rank(var) == 2
                   for eqn in equations(closed.jaxpr) for var in eqn.invars))


def entrywise(eqn) -> bool:
    """Whether `eqn` acts one entry at a time: an entrywise operation (`dew.nn.programs.entrywise`), or a
    scalar's broadcast. The one test of an equation's kind its readers share, each with its own of the shapes
    it reads (`_axis_free`, the Mosaic program's `_entrywise`, `dew.training.stages`)."""
    return programs.entrywise(eqn.primitive) or (
        eqn.primitive is jax.lax.broadcast_in_dim_p and _rank(eqn.invars[0]) == 0)


def _axis_free(jaxpr: jax_core.Jaxpr, shape: tuple[int, ...] = ()) -> bool:
    """Whether `jaxpr` acts one entry at a time: every value a scalar or of `shape`, every operation
    entrywise (`entrywise`), a scalar's broadcast or a call of a program so."""
    for eqn in jaxpr.eqns:
        if any(getattr(var.aval, "shape", ()) not in ((), shape) for var in (*eqn.invars, *eqn.outvars)):
            return False
        nested = list(jax_core.jaxprs_in_params(eqn.params))
        if nested:
            # A call, or the custom derivative's own primal (its pullback's trace evaluates it).
            if not eqn.primitive.inlined_call or not all(
                    _axis_free(inner, shape) for inner in nested):
                return False
        elif not entrywise(eqn):
            return False
    return True


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
        primitive = eqn.primitive
        rule = RULES.get(primitive) or (
            _reduced if reduction(primitive) else _cumulated if cumulative(primitive)
            else _custom if own_derivative(primitive) else _call if purely_called(eqn) is not None else None)
        if rule is not None:
            out = rule(eqn, *operands_of, **eqn.params)
        elif entrywise(eqn):
            out = eqn.primitive.bind(*operands_of, **eqn.primitive.get_bind_params(eqn.params))
        else:
            raise Untransposable(f"{eqn.primitive.name}, which no layout rule takes")
        values.update(zip(eqn.outvars, out if eqn.primitive.multiple_results else [out], strict=True))
    return [read(var) for var in jaxpr.outvars]


def transposed(fn: Callable) -> Callable:
    """`fn`, a function of `[rows, columns]` tiles (and scalars) as their own layout's callers write it, as
    one of the same tiles held transposed, `[columns, rows]`: `transposed(fn)(*(x.T for x in tiles))` is
    `fn(*tiles).T`, each of its tile outputs held transposed too. Its program is traced at the tiles' own
    shapes, so a choice it makes by their shape is the one it makes there, and run operation by operation,
    each axis an operation names swapped (`RULES`), one entry at a time unchanged (`entrywise`); it
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
