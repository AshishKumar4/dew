"""What every derivation shares (`derive`, `scan`, `carry`): the forms a declaration derives (`Derived`),
those forms with their math rewritten by the engine (`rewritten`), its refusal (`Refused`), the reading of a
declaration's jaxpr (what a value reads, an equation's evaluation, a value's array type), a declaration
traced with Dew's products as one each (`inlined`), an extracted form's value (`evaluated`, `evaluator`), and
the one switch the templates take their derived forms behind (`derived`)."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import NamedTuple, Protocol

import jax
import numpy as np
from egglog import get_callable_args
from jax.core import ShapedArray
from jax.extend import core as jcore

from dew.algebra import program, terms
from dew.cache import DERIVED
from dew.nn.precision import float32_dot_general


class Accumulated(NamedTuple):
    """A reduction's step a mainloop's accumulator takes (`Derived.step`'s `stepped`): its state rescaled by
    the row `factor` plus the tile's `weights @ values`, each value as the step evaluates it."""

    factor: jax.Array
    weights: jax.Array
    values: jax.Array


class Added(NamedTuple):
    """A reduction's step a mainloop's own addition takes (`Derived.step`'s `stepped`): its state rescaled by
    the row `factor` plus the tile's `sum` of its `terms`."""

    factor: jax.Array
    sum: jax.Array
    terms: jax.Array


class Merged(NamedTuple):
    """A rescaled sum's merge a mainloop's own merge takes (`Derived.merge`'s `merged`): the factors the first
    state and the second take to their joined anchor, as the merge evaluates them."""

    first: jax.Array
    second: jax.Array


class Derived(NamedTuple):
    """A declaration's forms (`derive`): `init(shapes)` the empty state for operands of `shapes` (by name,
    their walked axis a tile's), `step(state, tiles)` the state after one tile of each walked operand (and the
    whole of each other), `merge(a, b, operands)` two states of disjoint walks joined (its unwalked operands
    whole beside them), `finish(state, operands)` the declaration's outputs that the walk reduces,
    `mapped(tiles)` its outputs along the walked axis, a tile's own (each reads no walked reduction: a
    backward's dQ, walking queries), each form None in the other's outputs' places; `report` what the
    derivation decoupled and its rounding class; `prepare(tiles)` a tile's values that read no state, which a
    walk computes for every tile at once (a reduction's tiles as they are). A scan's (`scan.derive_scan`)
    `step(state, prepared)` and `mapped(prepared, state)` read a chunk's prepared values and the state alone
    (the carry, and a chunk's outputs from the state entering it); its `backward(tiles, state, d_state,
    d_outputs)` is a chunk's adjoint: the cotangent of the state entering it and of each of its operands, from
    the state entering it and the cotangents of the state leaving it and of its outputs (a reduction's
    backward is a declaration of its own, None here). A reduction's `slots(shapes)` names each carried state's
    role (`max`, `sum`, `contraction`: the reduction that made it) and shape in the state's order, so a
    template holds each where its mainloop holds that value; its `step(state, tiles, product=, stepped=,
    outputs=)` takes a mainloop's own matrix product (`evaluator`) and its own step (`stepped(states,
    parts)`, called once a step with each carried state's parts in the slots' order: `Accumulated` for a sum
    a rule stepped as `state * factor + weights @ values`, `Added` for one stepped as `state * factor +
    sum(terms)`, None for any other; it returns each new state it takes, None for one it leaves to the
    forms: a mainloop's accumulator, a compensated walk's running sums, a guard's shifted ones), the factor a
    row's, the parts the rules state (`derive.rescaled_by`, `summand`), each evaluated once), and with
    `outputs` (places) returns those of the tile's outputs along the walked axis beside the state, by the
    same evaluation; its `merge(a, b, operands=None, shapes=, merged=)` (a mainloop's own merge,
    `merged(firsts, seconds, parts)`, called once with each state's `Merged` factors or None, as `stepped`)
    and `finish(state, operands=None, shapes=)` read the state (and the unwalked operands given) by the forms
    of the walk whose tiles were of `shapes` (`init`'s), named, as a state's values alone name no walk (a
    bfloat16 walk's float32 sums are a float32 walk's too)."""

    init: Callable
    step: Callable
    merge: Callable
    finish: Callable
    mapped: Callable
    report: Mapping
    backward: Callable | None = None
    prepare: Callable = lambda tiles: tiles
    slots: Callable | None = None


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
        env.update(zip(equation.outvars, bound(equation, map(read, equation.invars)), strict=True))


def bound(equation: jcore.JaxprEqn, inputs) -> list:
    """`equation`'s outputs over `inputs`, bound as `jax.core.eval_jaxpr` binds it (a custom derivative's
    parameters as its primitive takes them, `get_bind_params`)."""
    outs = equation.primitive.bind(*inputs, **equation.primitive.get_bind_params(equation.params))
    return outs if equation.primitive.multiple_results else [outs]


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
                            backward=through(derived.backward), prepare=through(derived.prepare))


def inlined(function: Callable) -> Callable:
    """`function` with each choice by the platform a program lowers for (`lax.platform_dependent`, a `cond` on
    a `platform_index`: Dew's products under a wide arithmetic) traced as its default branch, and each call
    of a jitted function traced inline (Dew's product under a wide CPU arithmetic is a jitted choice), so its
    product enters the e-graph as one whatever the platform; a form evaluates products as Dew's
    (`evaluated`)."""
    def run(*values):
        closed = jax.make_jaxpr(function)(*values)
        outputs = evaluated_inline(closed.jaxpr, closed.consts, values)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    return run


def evaluated_inline(jaxpr: jcore.Jaxpr, consts, values) -> list:
    """`jaxpr`'s outputs over `values`, each platform choice its default branch's (`inlined`)."""
    env: dict = {}

    def read(atom):
        return atom.val if isinstance(atom, jcore.Literal) else env[atom]

    env.update(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, values, strict=True))
    producers = {out: equation for equation in jaxpr.eqns for out in equation.outvars}
    for equation in jaxpr.eqns:
        branches = equation.params.get("branches") if equation.primitive.name == "cond" else None
        index = equation.invars[0] if equation.invars else None
        chooser = producers.get(index) if branches and isinstance(index, jcore.Var) else None
        if branches and chooser is not None and chooser.primitive.name == "platform_index":
            default = branches[-1]
            operands = [read(atom) for atom in equation.invars[1:]]
            outs = evaluated_inline(default.jaxpr, default.consts, operands)
        elif equation.primitive.name in ("jit", "pjit", "closed_call"):
            called = equation.params.get("jaxpr") or equation.params["call_jaxpr"]
            outs = evaluated_inline(called.jaxpr, called.consts, [read(atom) for atom in equation.invars])
        else:
            outs = bound(equation, map(read, equation.invars))
        env.update(zip(equation.outvars, outs, strict=True))
    return [read(atom) for atom in jaxpr.outvars]


@contextlib.contextmanager
def derived(on: bool = True) -> Iterator[None]:
    """Within the block, the templates take their algebra's derived forms (`derive`, `scan`) where they hold
    them in place of their hand-written ones (`on`), or the hand-written ones: a context of JAX's trace
    (`dew.cache.DERIVED`), so a program held across the switch traces afresh. The one switch the derived and
    hand-written forms swap behind, which check 1 drives both ways (docs/design/algebra.md §6.9)."""
    with DERIVED(on):
        yield


def derived_on() -> bool:
    """Whether the templates take their derived forms here (`derived`)."""
    return bool(DERIVED.value)


def evaluated(region: terms.Region, form: terms.Term, leaves: Sequence,
              known: Mapping[Hashable, jax.Array]) -> jax.Array:
    """A form's value (`evaluator`'s)."""
    return evaluator(region, leaves, known)(form)


PLAIN_PRODUCT = (((1,), (0,)), ((), ()))
"""A matrix product's dimension numbers, `a @ b`: the left's last axis with the right's first, no batch."""


class Operand(NamedTuple):
    """A leaf's value given by its type and the call that acquires it (`load`), which an evaluation calls
    where a form first reads the leaf (`evaluator`): a mainloop's operand whose acquisition waits on its
    pipeline (a Mosaic step's value slot, its partner's turn) is acquired after what the form computes
    before it (a step's weights)."""

    shape: tuple[int, ...]
    dtype: object
    load: Callable[[], jax.Array]


def _scaled(operand: terms.Term, value: Callable[[terms.Term], jax.Array]) -> tuple:
    """An exponential's operand as a declaration's exponential takes it (`derive.Reduction.exponential`): a
    product of a value and a literal as the value and the literal's scale, so the exponential carries the
    scale's product itself (raw logits' `exp((s - m) c)`, its rounding compensated); else the value alone."""
    if (args := get_callable_args(operand, terms.Term.binary)) is not None and terms.text(args[1]) == "mul":
        for scaled, literal in ((args[0], args[2]), (args[2], args[0])):
            if get_callable_args(literal, terms.Term.literal) is not None:
                return value(scaled), float(np.asarray(value(literal)))
    return (value(operand),)


def evaluator(region: terms.Region[terms.Term], leaves: Sequence, known: Mapping[Hashable, jax.Array], *,
              product: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
              chosen: Mapping[Hashable, terms.Term] | None = None,
              exponential: tuple[str, Callable[..., jax.Array]] | None = None,
              rescale: Callable[[jax.Array, jax.Array], jax.Array] | None = None
              ) -> Callable[[terms.Term], jax.Array]:
    """The values of a region's forms, each term once however many forms read it: its leaves the given
    values (`leaves`, by index; None for one no form reads) and the outputs of the region's equations
    outside the language over them, their inputs in the forms `chosen` names for them (by `terms.key`)
    where given, each term `known` names (by `terms.key`) its value there, each matrix product
    `product(a, b)` where a caller holds its own (a mainloop's issuer) and every other product Dew's
    (`float32_dot_general`), each exponential a declaration implements by its own (`exponential`: the
    language's name of its base and its function, `exponential(x, scale=1.0)`, `derive.Reduction.exponential`;
    a product with a literal its `scale`, `_scaled`) and each rescale of two
    anchors by the declaration's (`rescale`, `terms.RESCALE`), every other term the engine's
    (`terms.value_of`)."""
    values = dict(known)
    outside = {leaf: opaque for opaque in region.opaque for leaf in opaque.outputs}
    given = dict(enumerate(leaves))

    def leaf(index: int):
        if given.get(index) is None:
            opaque = outside[index]
            picked = chosen or {}
            inputs = [literal if term is None else value(picked.get(terms.key(term), term))
                      for term, literal in zip(opaque.inputs, opaque.literals, strict=True)]
            given.update(zip(opaque.outputs, bound(opaque.equation, inputs), strict=True))
        if isinstance(given[index], Operand):
            given[index] = given[index].load()
        return given[index]

    def value(term: terms.Term) -> jax.Array:
        key = terms.key(term)
        if key in values:
            return values[key]
        if (args := get_callable_args(term, terms.Term.leaf)) is not None:
            made = leaf(terms.integer(args[0]))
        elif (args := get_callable_args(term, terms.Term.dot)) is not None:
            left, right, index = args
            _, params = region.params[terms.integer(index)]
            numbers = jax.tree.map(int, params["dimension_numbers"])
            if product is not None and numbers == PLAIN_PRODUCT:
                made = product(value(left), value(right))
            else:
                made = float32_dot_general(value(left), value(right), params["dimension_numbers"],
                                           precision=params.get("precision"),
                                           preferred_element_type=params.get("preferred_element_type"))
        elif (exponential is not None and (args := get_callable_args(term, terms.Term.unary)) is not None
              and terms.text(args[1]) == exponential[0]):
            made = exponential[1](*_scaled(args[0], value))
        elif (rescale is not None and (args := get_callable_args(term, terms.Term.binary)) is not None
              and terms.text(args[1]) == terms.RESCALE):
            made = rescale(value(args[0]), value(args[2]))
        else:
            # Every leaf is read above.
            made = terms.value_of(term, value, region, ())
        values[key] = made
        return made

    return value
