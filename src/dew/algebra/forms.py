"""What every derivation shares (`derive`, `scan`): the forms a declaration derives (`Derived`), those forms
with their math rewritten by the engine (`rewritten`), its refusal (`Refused`), and the reading of a
declaration's jaxpr: what a value reads, an equation's evaluation, a value's array type."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple, Protocol

import jax
from jax.core import ShapedArray
from jax.extend import core as jcore

from dew.algebra import program


class Derived(NamedTuple):
    """A declaration's forms (`derive`): `init(shapes)` the empty state for operands of `shapes` (by name,
    their walked axis a tile's), `step(state, tiles)` the state after one tile of each walked operand (and
    the whole of each other), `merge(a, b)` two states of disjoint walks joined, `finish(state, operands)`
    the declaration's outputs that the walk reduces, `mapped(tiles)` its outputs along the walked axis, a
    tile's own (each reads no walked reduction: a backward's dQ, walking queries), each form None in the
    other's outputs' places; a scan's `mapped(tiles, state)` a chunk's outputs from the state entering it
    (`scan.derive_scan`); `report` what the derivation decoupled and its rounding class. A scan's
    `backward(tiles, state, d_state, d_outputs)` is a chunk's adjoint: the cotangent of the state entering it
    and of each of its operands, from the state entering it and the cotangents of the state leaving it and of
    its outputs (a reduction's backward is a declaration of its own, None here)."""

    init: Callable
    step: Callable
    merge: Callable
    finish: Callable
    mapped: Callable
    report: Mapping
    backward: Callable | None = None


class Program(Protocol):
    """A traced program: its jaxpr and the constants its constant variables take (a closed jaxpr,
    `jax.make_jaxpr`'s)."""

    @property
    def jaxpr(self) -> jcore.Jaxpr: ...

    @property
    def consts(self) -> Sequence: ...


class Refused(ValueError):
    """A declaration the derivation cannot express as monoids along its walked axis, and why."""


def read_by(jaxpr: jcore.Jaxpr, sinks: set) -> set:
    """The values `sinks` read, themselves included."""
    read = set(sinks)
    for equation in reversed(jaxpr.eqns):
        if any(var in read for var in equation.outvars):
            read.update(var for var in equation.invars if isinstance(var, jcore.Var))
    return read


def evaluate(closed, env: dict, needed: set) -> None:
    """Each equation an output of which is `needed` and not yet in `env`, evaluated into it in order."""
    def read(atom):
        return atom.val if isinstance(atom, jcore.Literal) else env[atom]

    for equation in closed.jaxpr.eqns:
        done = all(out in env for out in equation.outvars)
        if done or not any(out in needed for out in equation.outvars):
            continue
        outs = equation.primitive.bind(*map(read, equation.invars), **equation.params)
        outs = outs if equation.primitive.multiple_results else [outs]
        env.update(zip(equation.outvars, outs, strict=True))


def environment(closed, values, known: dict) -> dict:
    """The values a declaration's equations read: `known` ones (reductions' running values), its inputs and
    its constants."""
    return {**dict(zip(closed.jaxpr.invars, values, strict=True)),
            **dict(zip(closed.jaxpr.constvars, closed.consts, strict=True)), **known}


def aval(atom) -> ShapedArray:
    """An atom's array type: every value of the language's equations has one."""
    if not isinstance(atom.aval, ShapedArray):
        raise Refused(f"a value of no array type ({atom.aval}) in the declaration")
    return atom.aval


def rewritten(derived: Derived, *, accurate: bool, saturation: program.Saturation) -> Derived:
    """`derived`'s forms, each traced where it is called and its math rewritten by the engine before it is
    evaluated (`program.rewrite`: every window a rule reads in its cheapest form under the program's cost,
    each rewrite in the kernel report): the derivation's forms joined to the e-graph's rules, `accurate`
    admitting exact rewrites only. The empty state (`init`) takes shapes, not values: as it is. A rewritten
    form computes its form's values (check 1 holds it, tests/test_algebra_join.py); it is cheaper where a
    rule found a cheaper form, the form as derived where none did."""

    def through(form: Callable | None) -> Callable | None:
        if form is None:
            return None

        def call(*arguments):
            flat, tree = jax.tree.flatten(arguments)
            closed, shapes = jax.make_jaxpr(lambda *values: form(*jax.tree.unflatten(tree, list(values))),
                                            return_shape=True)(*flat)
            outputs = program.rewrite(closed, flat, accurate=accurate, saturation=saturation)
            return jax.tree.unflatten(jax.tree.structure(shapes), outputs)

        return call

    return derived._replace(step=through(derived.step), merge=through(derived.merge),
                            finish=through(derived.finish), mapped=through(derived.mapped),
                            backward=through(derived.backward))
