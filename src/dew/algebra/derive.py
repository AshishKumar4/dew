"""Template forms derived from a declaration through the e-graph (docs/design/algebra.md §6.9): a reduction's
math stated on whole operands becomes the state a kernel carries along its walked axis, that state's empty
value, one tile's step into it, the merge of two states (a split walk's, the work scheduler's fix-up), the
outputs a finished state gives and the outputs along the walked axis a tile gives alone.

The declaration, traced at one tile, enters the e-graph as terms (`terms.region`). Facts the rules read are
saturated first (`WALKING`): each value's walked axis (from each operand's, through the layouts, reductions
and products that move it), which values read no walked reduction (`local`), which reductions sum or take
the maximum along the walked axis (`summed`, `peaked`), which values are functions of walked maxima alone
(`anchor`), and which scale as `exp(-k)` with such a value `k` (`scales`: `exp(x - k)` and what multiplies,
selects or sums it). The derivation's rules (`DECOUPLED`) then say what a step, a merge, a finish and a
carried state read of each value (`Term.stepped`, `merged`, `finished`, `held`): each distributes over the
language, and at a walked reduction a monoid's identity holds (the maximum of the carried and the tile's; a
sum's carried plus the tile's), a sum that scales with an anchor rescaled by `exp(k_carried - k_new)` first
(Neptune's correction terms: online softmax and its split-walk merge, derived rather than written). A
carried state is `Term.state` (its reduction's index, the side of a merge). The engine's rule sets the
numerics admit (`rules.admitted`) saturate in the same e-graph, and extraction under the program's cost
(`cost.model`) picks each form from both; the derivation's own constructors cost without bound, so a value no
rule rewrites (a sum that reads a maximum otherwise, an output along the walked axis that reads a reduction)
leaves its form unpriced, and the declaration is refused (`Refused`). Dew's products in a declaration
(`float32_einsum`, under a wide arithmetic a choice by platform) enter as their default branch's product and
are evaluated as Dew's (`float32_dot_general`). The decoupling reorders rounding.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Mapping
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from egglog import (
    DagCostModel,
    EGraph,
    Schedule,
    String,
    Unit,
    Vec,
    eq,
    function,
    get_callable_args,
    get_literal_value,
    i64,
    rewrite,
    rule,
    ruleset,
    seq,
    set_,
    var,
    vars_,
)
from egglog.bindings import EggSmolError

from dew.algebra import cost, program, rules, terms
from dew.algebra.cost import Work
from dew.algebra.forms import Derived, Refused, evaluated, evaluator, inlined
from dew.algebra.terms import Term
from dew.nn.kernels.templates import report as kernel_report

Axes = tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Reduction:
    """A walk whose state is monoids' (attention): `function(**operands)` on whole operands in plain JAX,
    each operand's axes by name (`operands`), the axis the kernel walks in tiles (`walked`), and the
    exponent's base the template takes (`exponent`: 2 where it folds `log2(e)` into its logits, `"e"`
    otherwise), which the derivation keeps."""

    function: Callable
    operands: Mapping[str, Axes]
    walked: str
    exponent: Literal[2, "e"]


# The facts the derivation reads (§6.3), each registered from a parameter's own values or saturated by
# `WALKING`.


@function(merge=lambda old, new: old)
def walked(term: Term) -> i64: ...
"""The position of the walked axis in a value."""


@function
def keeps(params: i64) -> Unit: ...
"""An elementwise equation's parameters: the walked axis where its operand's is."""


@function
def lays(params: i64, axis: i64, to: i64) -> Unit: ...
"""A layout change moving its operand's `axis` to `to`."""


@function
def reduces(params: i64, axis: i64, to: i64) -> Unit: ...
"""A reduction over axes other than `axis`, which it moves to `to`."""


@function
def left_of(params: i64, axis: i64, to: i64) -> Unit: ...
"""A product keeping its left operand's `axis` at `to`."""


@function
def right_of(params: i64, axis: i64, to: i64) -> Unit: ...
"""A product keeping its right operand's `axis` at `to`."""


@function
def contracts_left(params: i64, axis: i64) -> Unit: ...
"""A product contracting its left operand's `axis`."""


@function
def contracts_right(params: i64, axis: i64) -> Unit: ...
"""A product contracting its right operand's `axis`."""


@function
def local(term: Term) -> Unit: ...
"""A value that reads no walked reduction: a tile's own."""


@function
def unwalked(term: Term) -> Unit: ...
"""A value with no walked axis that reads no walked reduction."""


@function
def summed(term: Term) -> Unit: ...
"""A sum along the walked axis (a reduction, or a product contracting it) of local values."""


@function
def peaked(term: Term) -> Unit: ...
"""A maximum along the walked axis of a local value."""


@function
def anchor(term: Term) -> Unit: ...
"""A value of walked maxima and local values alone (`k`, which a decoupled sum subtracts in its exponent)."""


@function
def scales(term: Term, by: Term, name: String, exponent: i64, difference: i64) -> Unit: ...
"""A value that scales as `exp(-by)` (`name` the exponential, `exponent` and `difference` its and its
subtraction's parameters): `exp(x - by)` with `x` local, and what multiplies, selects or sums it with local
values."""


@function
def zero(term: Term) -> Unit: ...
"""A value of zeros (a literal, or a layout of one)."""


@function(merge=lambda old, new: old)
def lifts(given: Vec[i64], to: Vec[i64]) -> i64: ...
"""The parameters of a broadcast of a value of shape `given` to `to`, its axes the leading ones."""


@function(merge=lambda old, new: old)
def retype(slot: i64) -> i64: ...
"""The parameters of a conversion to a walked reduction's dtype, by its index (`slot`): a sum's correction,
of its anchor's dtype, converted to its accumulator's (a narrow exponent summed in float32, as JAX sums a
narrow value) before it scales the carried sum."""


@function(merge=lambda old, new: old)
def least() -> i64: ...
"""The literal of the walked maxima's dtype's least finite value (`Region.literals`): a correction's shift
`max(k_now, least)` keeps `exp(k_carried - k_now)` zero, not `exp(-inf + inf)`, where nothing is seen yet."""


@function(merge=lambda old, new: old)
def slot(term: Term) -> i64: ...
"""A walked reduction's index among the walk's, its carried state's (`Term.state`)."""


def _walking():
    """The facts' rules (the module's account)."""
    x, y, z, k, t, base = vars_("x y z k t base", Term)
    name, layout = vars_("name layout", String)
    p, q, a, b, e, s = vars_("p q a b e s", i64)
    axes = var("axes", Vec[i64])
    walking = []
    for constructor in ("binary", "compare"):
        def made(left, right, params, constructor=constructor):
            return getattr(left, constructor)(name, right, params)
        walking += [
            rule(eq(t).to(made(x, y, p)), eq(walked(x)).to(a), keeps(p)).then(set_(walked(t)).to(a)),
            rule(eq(t).to(made(x, y, p)), eq(walked(y)).to(a), keeps(p)).then(set_(walked(t)).to(a)),
            rule(eq(t).to(made(x, y, p)), local(x), local(y)).then(local(t)),
            rule(eq(t).to(made(x, y, p)), unwalked(x), unwalked(y)).then(unwalked(t)),
            rule(eq(t).to(made(x, y, p)), anchor(x), anchor(y)).then(anchor(t)),
            rule(eq(t).to(made(x, y, p)), anchor(x), local(y)).then(anchor(t)),
            rule(eq(t).to(made(x, y, p)), local(x), anchor(y)).then(anchor(t)),
        ]
    for one in ("unary", "convert", "shaped"):
        def single(operand, params, one=one):
            return operand.unary(name, params) if one == "unary" else (
                operand.convert(params) if one == "convert" else operand.shaped(name, params))
        walking += [
            rule(eq(t).to(single(x, p)), local(x)).then(local(t)),
            rule(eq(t).to(single(x, p)), unwalked(x)).then(unwalked(t)),
            rule(eq(t).to(single(x, p)), anchor(x)).then(anchor(t)),
        ]
    walking += [
        rule(eq(t).to(x.unary(name, p)), eq(walked(x)).to(a)).then(set_(walked(t)).to(a)),
        rule(eq(t).to(x.convert(p)), eq(walked(x)).to(a)).then(set_(walked(t)).to(a)),
        rule(eq(t).to(x.shaped(name, p)), eq(walked(x)).to(a), lays(p, a, b)).then(set_(walked(t)).to(b)),
        rule(eq(t).to(x.select(y, z)), eq(walked(x)).to(a)).then(set_(walked(t)).to(a)),
        rule(eq(t).to(x.select(y, z)), eq(walked(y)).to(a)).then(set_(walked(t)).to(a)),
        rule(eq(t).to(x.select(y, z)), eq(walked(z)).to(a)).then(set_(walked(t)).to(a)),
        rule(eq(t).to(x.select(y, z)), local(x), local(y), local(z)).then(local(t)),
        rule(eq(t).to(x.select(y, z)), unwalked(x), unwalked(y), unwalked(z)).then(unwalked(t)),
        rule(eq(t).to(x.select(y, z)), anchor(x), anchor(y), anchor(z)).then(anchor(t)),
        rule(eq(t).to(x.select(y, z)), local(x), local(y), anchor(z)).then(anchor(t)),
        rule(eq(t).to(x.select(y, z)), anchor(x), local(y), anchor(z)).then(anchor(t)),
        rule(eq(t).to(x.select(y, z)), anchor(x), anchor(y), local(z)).then(anchor(t)),
        rule(eq(t).to(x.select(y, z)), anchor(x), local(y), local(z)).then(anchor(t)),
        # Reductions: over other axes the walked axis moves; over the walked axis alone a walked one.
        rule(eq(t).to(x.reduce(name, axes, p)), eq(walked(x)).to(a), reduces(p, a, b)).then(
            set_(walked(t)).to(b)),
        rule(eq(t).to(x.reduce(name, axes, p)), eq(walked(x)).to(a), reduces(p, a, b), local(x)).then(
            local(t)),
        rule(eq(t).to(x.reduce(name, axes, p)), unwalked(x)).then(unwalked(t), local(t)),
        rule(eq(t).to(x.reduce(String("sum"), axes, p)), eq(walked(x)).to(a), eq(axes).to(Vec[i64](a)),
             local(x)).then(summed(t)),
        rule(eq(t).to(x.reduce(String("max"), axes, p)), eq(walked(x)).to(a), eq(axes).to(Vec[i64](a)),
             local(x)).then(peaked(t), anchor(t)),
        # Products: a kept axis moves to its place in the product; a contracted walked axis is a walked sum.
        rule(eq(t).to(x.dot(y, p)), eq(walked(x)).to(a), left_of(p, a, b)).then(set_(walked(t)).to(b)),
        rule(eq(t).to(x.dot(y, p)), eq(walked(y)).to(a), right_of(p, a, b)).then(set_(walked(t)).to(b)),
        rule(eq(t).to(x.dot(y, p)), unwalked(x), unwalked(y)).then(unwalked(t), local(t)),
        rule(eq(t).to(x.dot(y, p)), eq(walked(x)).to(a), left_of(p, a, b), local(x), unwalked(y)).then(
            local(t)),
        rule(eq(t).to(x.dot(y, p)), eq(walked(y)).to(a), right_of(p, a, b), local(y), unwalked(x)).then(
            local(t)),
        rule(eq(t).to(x.dot(y, p)), eq(walked(x)).to(a), left_of(p, a, b), eq(walked(y)).to(q),
             right_of(p, q, b), local(x), local(y)).then(local(t)),
        rule(eq(t).to(x.dot(y, p)), eq(walked(x)).to(a), contracts_left(p, a), local(x), local(y)).then(
            summed(t)),
        rule(eq(t).to(x.dot(y, p)), eq(walked(y)).to(a), contracts_right(p, a), local(x), local(y)).then(
            summed(t)),
        # Zeros, through their layouts and casts.
        rule(eq(t).to(x.shaped(name, p)), zero(x)).then(zero(t)),
        rule(eq(t).to(x.convert(p)), zero(x)).then(zero(t)),
    ]
    # What scales as exp(-k): an exponential of a local value less an anchor (or its broadcast), and what
    # multiplies, divides, selects against zeros, sums, casts or lays it out with local values.
    walking += [
        rule(eq(t).to(x.binary(String("sub"), base, s).unary(name, e)), local(x),
             eq(base).to(k.shaped(String("broadcast_in_dim"), q)), anchor(k),
             eq(name).to(String("exp"))).then(scales(t, k, name, e, s)),
        rule(eq(t).to(x.binary(String("sub"), base, s).unary(name, e)), local(x),
             eq(base).to(k.shaped(String("broadcast_in_dim"), q)), anchor(k),
             eq(name).to(String("exp2"))).then(scales(t, k, name, e, s)),
    ]
    for side in (0, 1):
        def product(scaled, other, params, side=side):
            return scaled.binary(String("mul"), other, params) if side == 0 else other.binary(
                String("mul"), scaled, params)
        walking.append(rule(eq(t).to(product(x, y, p)), scales(x, k, name, e, s), local(y)).then(
            scales(t, k, name, e, s)))
        walking.append(rule(eq(t).to(x.dot(y, p) if side == 0 else y.dot(x, p)), scales(x, k, name, e, s),
                            local(y)).then(scales(t, k, name, e, s)))
    walking += [
        rule(eq(t).to(x.binary(String("div"), y, p)), scales(x, k, name, e, s), local(y)).then(
            scales(t, k, name, e, s)),
        rule(eq(t).to(x.binary(String("add"), y, p)), scales(x, k, name, e, s),
             scales(y, k, name, e, s)).then(scales(t, k, name, e, s)),
        rule(eq(t).to(z.select(y, x)), scales(x, k, name, e, s), zero(y), local(z)).then(
            scales(t, k, name, e, s)),
        rule(eq(t).to(z.select(x, y)), scales(x, k, name, e, s), zero(y), local(z)).then(
            scales(t, k, name, e, s)),
        rule(eq(t).to(x.convert(p)), scales(x, k, name, e, s)).then(scales(t, k, name, e, s)),
        rule(eq(t).to(x.shaped(layout, p)), scales(x, k, name, e, s)).then(scales(t, k, name, e, s)),
        rule(eq(t).to(x.reduce(String("sum"), axes, p)), scales(x, k, name, e, s), eq(walked(x)).to(a),
             reduces(p, a, b)).then(scales(t, k, name, e, s)),
        rule(eq(t).to(x.reduce(String("sum"), axes, p)), eq(walked(x)).to(a), eq(axes).to(Vec[i64](a)),
             scales(x, k, name, e, s)).then(summed(t)),
        rule(eq(t).to(x.dot(y, p)), eq(walked(x)).to(a), contracts_left(p, a), scales(x, k, name, e, s),
             local(y)).then(summed(t)),
        rule(eq(t).to(x.dot(y, p)), eq(walked(y)).to(a), contracts_right(p, a), scales(y, k, name, e, s),
             local(x)).then(summed(t)),
    ]
    return ruleset(*walking, name="walking")


def _shapes():
    """The shapes of the values the derivation's rules make: a carried state its reduction's, a value a step,
    merge, finish or carried state reads its own, a select its operands', a lifted correction its broadcast's
    target."""
    x, y, z = vars_("x y z", Term)
    side = var("side", i64)
    extent = var("extent", Vec[i64])
    given, to = vars_("given to", Vec[i64])
    shape = rules.shape
    return ruleset(
        rule(eq(extent).to(shape(x)), eq(y).to(x.held(side))).then(set_(shape(y)).to(extent)),
        rule(eq(extent).to(shape(x)), eq(y).to(x.stepped())).then(set_(shape(y)).to(extent)),
        rule(eq(extent).to(shape(x)), eq(y).to(x.merged())).then(set_(shape(y)).to(extent)),
        rule(eq(extent).to(shape(x)), eq(y).to(x.finished())).then(set_(shape(y)).to(extent)),
        rule(eq(extent).to(shape(y)), eq(z).to(x.select(y, z))).then(set_(shape(x.select(y, z))).to(extent)),
        rule(eq(y).to(x.shaped(String("broadcast_in_dim"), lifts(given, to)))).then(set_(shape(y)).to(to)),
        name="derived shapes")


WALKING = _walking()
SHAPES = _shapes()


def _decoupled():
    """The derivation's rules (the module's account): each reading distributes over the language, and a walked
    reduction's step, merge and finish are its monoid's, a sum that scales with an anchor rescaled first."""
    x, y, z, k, t = vars_("x y z k t", Term)
    name = var("name", String)
    p, e, s, side, a, b, lift, added, multiplied, largest, i, floor, typed = vars_(
        "p e s side a b lift added multiplied largest i floor typed", i64)
    axes = var("axes", Vec[i64])
    given, to = vars_("given to", Vec[i64])
    stock = rules.stock
    shape = rules.shape
    readings = [lambda term: term.stepped(), lambda term: term.merged(), lambda term: term.finished(),
                lambda term: term.held(side)]
    derived = []
    for read in readings:
        derived += [
            rewrite(read(Term.leaf(p))).to(Term.leaf(p)),
            rewrite(read(Term.literal(p))).to(Term.literal(p)),
            rewrite(read(x.binary(name, y, p))).to(read(x).binary(name, read(y), p)),
            rewrite(read(x.compare(name, y, p))).to(read(x).compare(name, read(y), p)),
            rewrite(read(x.unary(name, p))).to(read(x).unary(name, p)),
            rewrite(read(x.convert(p))).to(read(x).convert(p)),
            rewrite(read(x.shaped(name, p))).to(read(x).shaped(name, p)),
            rewrite(read(x.select(y, z))).to(read(x).select(read(y), read(z))),
            # A reduction over other axes, or of no walked value, and a product contracting no walked axis.
            rewrite(read(x.reduce(name, axes, p))).to(read(x).reduce(name, axes, p), eq(walked(x)).to(a),
                                                      reduces(p, a, b)),
            rewrite(read(x.reduce(name, axes, p))).to(read(x).reduce(name, axes, p), unwalked(x)),
            rewrite(read(x.dot(y, p))).to(read(x).dot(read(y), p), unwalked(x), unwalked(y)),
            rewrite(read(x.dot(y, p))).to(read(x).dot(read(y), p), eq(walked(x)).to(a), left_of(p, a, b),
                                          unwalked(y)),
            rewrite(read(x.dot(y, p))).to(read(x).dot(read(y), p), eq(walked(y)).to(a), right_of(p, a, b),
                                          unwalked(x)),
            rewrite(read(x.dot(y, p))).to(read(x).dot(read(y), p), eq(walked(x)).to(a), left_of(p, a, b),
                                          eq(walked(y)).to(e), right_of(p, e, b)),
        ]

    slotted = eq(i).to(slot(t))
    for combined, which, index in (("max", peaked, largest), ("add", summed, added)):
        stocked = eq(index).to(stock(combined))
        derived += [
            rewrite(t.stepped()).to(Term.state(i, 0).binary(combined, t, index), which(t), local_of(t),
                                    stocked, slotted),
            rewrite(t.merged()).to(Term.state(i, 0).binary(combined, Term.state(i, 1), index), which(t),
                                   local_of(t), stocked, slotted),
            rewrite(t.finished()).to(Term.state(i, 0), which(t), slotted),
        ]
    derived.append(rewrite(t.held(side)).to(Term.state(i, side), peaked(t), slotted))

    # A sum along the walked axis of values that scale with an anchor `k`: its carried value rescaled by
    # exp(k carried - k now), lifted to its shape, beside the tile's sum at the anchor now; the anchor now
    # at least the least finite value, so a state that has seen nothing (both -inf) rescales by 0.
    def rescaled(carried_side, now):
        shifted = now.binary(String("max"), Term.literal(floor), largest)
        correction = k.held(carried_side).binary(String("sub"), shifted, s).unary(name, e).convert(typed)
        lifted = correction.shaped(String("broadcast_in_dim"), lift)
        return Term.state(i, carried_side).binary(String("mul"), lifted, multiplied)

    lifted = (eq(given).to(shape(k)), eq(to).to(shape(t)), eq(lift).to(lifts(given, to)),
              eq(added).to(stock("add")), eq(multiplied).to(stock("mul")), eq(largest).to(stock("max")),
              eq(floor).to(least()), eq(typed).to(retype(i)), slotted)
    for summed_term, tile in ((x.reduce(String("sum"), axes, p), x.stepped().reduce(String("sum"), axes, p)),
                              (x.dot(y, p), x.stepped().dot(y.stepped(), p)),
                              (y.dot(x, p), y.stepped().dot(x.stepped(), p))):
        conditions = (eq(t).to(summed_term), summed(t), scales(x, k, name, e, s), *lifted)
        derived += [
            rewrite(t.stepped()).to(rescaled(0, k.stepped()).binary(String("add"), tile, added), *conditions),
            rewrite(t.merged()).to(rescaled(0, k.merged()).binary(String("add"), rescaled(1, k.merged()),
                                                                  added), *conditions),
            rewrite(t.finished()).to(Term.state(i, 0), *conditions),
        ]
    return ruleset(*derived, name="decoupled")


@function
def local_of(term: Term) -> Unit: ...
"""A walked reduction whose operands are local (its monoid alone, no rescaling)."""


def _plain():
    x, y, t = vars_("x y t", Term)
    name = var("name", String)
    p, a = vars_("p a", i64)
    axes = var("axes", Vec[i64])
    return ruleset(
        rule(eq(t).to(x.reduce(name, axes, p)), local(x), eq(walked(x)).to(a), eq(axes).to(Vec[i64](a))).then(
            local_of(t)),
        rule(eq(t).to(x.dot(y, p)), local(x), local(y), summed(t)).then(local_of(t)),
        name="plain walked")


DECOUPLED = rules.Rules("decoupled", "reorders", _decoupled(),
                        receipt="docs/design/algebra.md §6.9; tests/test_algebra_derive.py")
"""The derivation's rules, run beside the engine's sets (`rules.admitted`) and never among them."""
PLAIN = _plain()


UNBOUNDED = Work(operations=math.inf)
"""A reading's work (`_priced`): a form that keeps one is no form."""
_READINGS = (Term.stepped, Term.merged, Term.finished, Term.held)


def _priced(region: terms.Region, schedule) -> DagCostModel:
    """The template's tile walk's price of each node (`cost.model` under `cost.walk`: a form the template's
    algebra takes runs between its mainloop's products, not as XLA fuses it), each of the derivation's
    readings unbounded."""
    timed = cost.model(region, schedule)

    def marginal(graph: EGraph, node) -> cost.Seconds:
        if any(get_callable_args(node, reading) is not None for reading in _READINGS):
            return cost.Seconds(UNBOUNDED, schedule)
        return timed.marginal_cost(graph, node)

    return DagCostModel(marginal_cost=marginal, identity=cost.Seconds(Work(), schedule))


@dataclasses.dataclass(frozen=True)
class _Forms:
    """A declaration's derived forms at one tile: its region, its walked reductions in order, each one's step
    and merge, each output's finish (None for one along the walked axis) and whether it is mapped (a tile's
    alone), the leaves' values given (the tile's operands), the region's leaves the walked operands occupy,
    and the report of the rewrite that chose the forms."""

    region: terms.Region
    reductions: tuple[Term, ...]
    steps: tuple[Term, ...]
    merges: tuple[Term, ...]
    finishes: tuple[Term | None, ...]
    mapped: tuple[bool, ...]
    outputs: tuple[Term | None, ...]
    walked_leaves: frozenset[int]
    consts: tuple
    report: program.AlgebraReport


def derive(declaration: Reduction, *, accurate: bool, saturation: program.Saturation) -> Derived:
    """`declaration`'s monoid forms along its walked axis (the module's account), saturated beside the
    engine's sets `accurate` admits under `saturation`'s limits, each tile shape's rewrite in the kernel
    report (`report.rewrites`); refused (`Refused`) where a form keeps a reading no rule rewrites."""
    names = tuple(declaration.operands)
    walked_axes = {name: axes.index(declaration.walked) for name, axes in declaration.operands.items()
                   if declaration.walked in axes}
    cache: dict = {}

    def forms(shapes: Mapping) -> _Forms:
        key = tuple((name, tuple(shapes[name][0]), str(shapes[name][1])) for name in names)
        if key not in cache:
            cache[key] = _derived(declaration, names, walked_axes, shapes, accurate, saturation)
            kernel_report.record_rewrite(cache[key].report)
        return cache[key]

    def shaped(values):
        return {name: (value.shape, value.dtype) for name, value in values.items()}

    def init(shapes):
        made = forms(shapes)
        avals = [jax.ShapeDtypeStruct(*shapes[name]) for name in names]
        typed = jax.eval_shape(lambda *values: tuple(evaluated(made.region, reduction,
                                                               [*made.consts, *values], {})
                                                     for reduction in made.reductions), *avals)
        return tuple(jnp.full(aval.shape, _identity(reduction, aval.dtype), aval.dtype)
                     for reduction, aval in zip(made.reductions, typed, strict=True))

    def carried(made: _Forms, *states) -> dict:
        return {str(Term.state(i64(index), i64(side))): state[index] for side, state in enumerate(states)
                for index in range(len(made.reductions))}

    def held(state) -> _Forms:
        # The forms a state's walk took: the merges and finishes read the state alone, so its shapes name
        # them whatever tile the walk took.
        shapes = tuple(tuple(value.shape) for value in state)
        for made in reversed(cache.values()):
            if tuple(made.region.shape(reduction) for reduction in made.reductions) == shapes:
                return made
        raise Refused(f"a state of shapes {shapes} no walk formed (`init`, `step`)")

    def step(state, tiles, *, product=None, accumulate=None):
        made = forms(shaped(tiles))
        value = evaluator(made.region, [*made.consts, *(tiles[name] for name in names)],
                          carried(made, state), product=product)
        stepped = []
        for index, form in enumerate(made.steps):
            if accumulate is not None and (parts := _accumulated(form, index)) is not None:
                factor, weights, values = parts
                stepped.append(accumulate(state[index], value(factor), value(weights), value(values)))
            else:
                stepped.append(value(form))
        return tuple(stepped)

    def merge(a, b, operands=None):
        made = held(a)
        # A merge reads no walked operand (`_derived`), its unwalked ones whole.
        leaves = [*made.consts, *(None if operands is None or name in walked_axes else operands[name]
                                  for name in names)]
        value = evaluator(made.region, leaves, carried(made, a, b))
        return tuple(value(form) for form in made.merges)

    def finish(state, operands=None):
        made = held(state)
        leaves = [*made.consts, *(None if operands is None else operands[name] for name in names)]
        value = evaluator(made.region, leaves, carried(made, state))
        # A literal output is its literal; one along the walked axis, `mapped`'s.
        outputs = [jnp.asarray(literal) if term is None else None if form is None else value(form)
                   for term, literal, form in zip(made.region.outputs, made.region.literal_outputs,
                                                  made.finishes, strict=True)]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def mapped(tiles):
        made = forms(shaped(tiles))
        value = evaluator(made.region, [*made.consts, *(tiles[name] for name in names)], {})
        outputs = [value(form) if along and form is not None else None
                   for form, along in zip(made.outputs, made.mapped, strict=True)]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def slots(shapes) -> tuple[tuple[str, tuple[int, ...]], ...]:
        made = forms(shapes)
        return tuple((_monoid(reduction), made.region.shape(reduction)) for reduction in made.reductions)

    return Derived(init, step, merge, finish, mapped,
                   {"rounding": "reorders", "walked": declaration.walked, "exponent": declaration.exponent,
                    "rules": ("walking", "decoupled")}, slots=slots)


def _monoid(reduction: Term) -> str:
    """A walked reduction's monoid: a maximum's `max`; a sum's, or a product's contracting the walked axis,
    `sum`."""
    args = get_callable_args(reduction, Term.reduce)
    return "max" if args is not None and get_literal_value(args[1]) == "max" else "sum"


def _accumulated(form: Term, index: int) -> tuple[Term, Term, Term] | None:
    """A sum's step a mainloop takes into its own accumulator (`Derived.step`'s `accumulate`): `form`, the
    `index`th state's, as `state * lift(factor) + weights @ values`, its row factor, weights and values; None
    for a step of another form."""
    args = get_callable_args(form, Term.binary)
    if args is None or get_literal_value(args[1]) != "add":
        return None
    for scaled, added in ((args[0], args[2]), (args[2], args[0])):
        product = get_callable_args(added, Term.dot)
        times = get_callable_args(scaled, Term.binary)
        if product is None or times is None or get_literal_value(times[1]) != "mul":
            continue
        for state, lifted in ((times[0], times[2]), (times[2], times[0])):
            at = get_callable_args(state, Term.state)
            lift = get_callable_args(lifted, Term.shaped)
            slot = None if at is None else (int(get_literal_value(at[0])), int(get_literal_value(at[1])))
            if slot == (index, 0) and lift is not None and get_literal_value(lift[1]) == "broadcast_in_dim":
                return lift[0], product[0], product[1]
    return None


def _derived(declaration: Reduction, names: tuple[str, ...], walked_axes: Mapping[str, int],
             shapes: Mapping, accurate: bool, saturation: program.Saturation) -> _Forms:
    """The forms at one tile's `shapes` (the module's account)."""
    began = time.perf_counter()
    def named(*values):
        return declaration.function(**dict(zip(names, values, strict=True)))

    avals = [jax.ShapeDtypeStruct(*shapes[name]) for name in names]
    with jax.disable_jit():
        closed = jax.make_jaxpr(inlined(named))(*avals)
    region = terms.region(closed)
    graph = EGraph()
    _register(graph, region, closed, names, walked_axes)
    graph.run(WALKING.saturate())
    for term in _opaque_locals(graph, region):
        graph.register(local(term))
    graph.run(WALKING.saturate())
    graph.run(PLAIN.saturate())
    found = [term for term, _ in region.shapes if _holds(graph, summed(term)) or _holds(graph, peaked(term))]
    reductions = tuple({str(term): term for term in found}.values())
    for index, reduction in enumerate(reductions):
        graph.register(set_(slot(reduction)).to(i64(index)))
        region.params.append((jax.lax.convert_element_type_p, {
            "new_dtype": region.dtypes[str(reduction)], "weak_type": False, "sharding": None}))
        graph.register(set_(retype(i64(index))).to(i64(len(region.params) - 1)))
        for side in (0, 1):
            graph.register(set_(rules.shape(Term.state(i64(index), i64(side)))).to(
                Vec[i64](*(i64(size) for size in region.shape(reduction)))))
    _lift(graph, region, reductions)
    outputs = list(region.outputs)
    along = [term is not None and _walked(graph, term) for term in outputs]
    for term, is_along in zip(outputs, along, strict=True):
        if is_along and not _holds(graph, local(term)):
            raise Refused("an output along the walked axis reads a walked reduction: no tile holds it")
    roots = ([reduction.stepped() for reduction in reductions]
             + [reduction.merged() for reduction in reductions]
             + [term.finished() for term, is_along in zip(outputs, along, strict=True)
                if term is not None and not is_along])
    declared = Term.bundle(Vec[Term](*region.roots()))
    bundle = Term.bundle(Vec[Term](*roots) if roots else Vec[Term].empty())
    graph.register(declared, bundle)
    schedule = cost.walk(cost.target())
    _, before = graph.extract(declared, include_cost=True, cost_model=cost.model(region, schedule),
                              extractor="greedy-dag")
    facts = seq(rules.FACTS.saturate(), SHAPES.saturate(), WALKING.saturate())
    _decoupled_through(graph, region, facts)
    saturated = program.saturate(graph, region, (*rules.admitted(accurate), DECOUPLED),
                                 saturation, facts)
    chosen, after = program.extracted(graph, bundle, _priced(region, schedule))
    if not math.isfinite(after.value):
        unwritten = next(root for root, form in zip(roots, chosen, strict=True) if not _written(form))
        raise Refused(f"no rule writes {unwritten} in the language: a walked reduction read other than by a "
                      "monoid or a decoupled exponent")
    report = program.AlgebraReport(region.equations, tuple(saturated.fired), saturated.fired,
                                   saturated.limits, saturated.iterations, program.node_count(graph),
                                   saturated.limited, before.value, after.value, time.perf_counter() - began)
    count = len(reductions)
    steps, merges, finished = chosen[:count], chosen[count:2 * count], iter(chosen[2 * count:])
    finishes = tuple(None if term is None or is_along else next(finished)
                     for term, is_along in zip(outputs, along, strict=True))
    first = len(closed.consts)
    walked_leaves = frozenset(first + index for index, name in enumerate(names) if name in walked_axes)
    for form in finishes:
        if form is not None and _reads_leaves(form) & walked_leaves:
            raise Refused("an output reads a walked operand outside every walked reduction")
    for form in merges:
        if _reads_leaves(form) & walked_leaves:
            raise Refused("a merge reads a walked operand: no tile is there to read")
    return _Forms(region, reductions, tuple(steps), tuple(merges), finishes, tuple(along), tuple(outputs),
                  walked_leaves, tuple(closed.consts), report)


def _decoupled_through(graph: EGraph, region: terms.Region, facts: Schedule) -> None:
    """The derivation's readings run through `region`'s terms alone first (`DECOUPLED`, its facts saturated
    after each pass): each reading meets the language at the region's depth, before the engine's sets grow
    the graph toward their limits, so a limit the engine reaches never stops a form."""
    for _ in range(4 * max(region.depth, 1) + 8):
        graph.run(facts)
        if not graph.run(DECOUPLED.ruleset).updated:
            break
    graph.run(facts)


def _register(graph: EGraph, region: terms.Region, closed, names, walked_axes) -> None:
    """The facts the rules read that each parameter's own values state (the engine's, `program.stated`, and
    the derivation's own), and each input's walked axis."""
    program.stated(graph, region)
    elementwise = terms.BINARY | terms.COMPARISONS | terms.UNARY | {"convert_element_type"}
    _least(graph, region, closed)
    for index, (primitive, params) in enumerate(region.params):
        name = primitive.name
        if name in elementwise:
            graph.register(keeps(i64(index)))
        elif name == "broadcast_in_dim":
            for axis, to in enumerate(params["broadcast_dimensions"]):
                graph.register(lays(i64(index), i64(axis), i64(int(to))))
        elif name == "transpose":
            permutation = list(params["permutation"])
            for axis in range(len(permutation)):
                graph.register(lays(i64(index), i64(axis), i64(permutation.index(axis))))
        elif name in ("squeeze", "expand_dims", "reshape"):
            for axis, to in _moved(primitive, params, region, index):
                graph.register(lays(i64(index), i64(axis), i64(to)))
        elif name in terms.REDUCTIONS:
            reduced = {int(axis) for axis in params["axes"]}
            for axis in range(_rank_before(region, index, reduced)):
                if axis not in reduced:
                    graph.register(reduces(i64(index), i64(axis), i64(axis - sum(r < axis for r in reduced))))
        elif name == "dot_general":
            (left_contracted, right_contracted), (left_batch, right_batch) = params["dimension_numbers"]
            left_rank, right_rank = _dot_ranks(region, index)
            left_free = [axis for axis in range(left_rank) if axis not in left_contracted and
                         axis not in left_batch]
            right_free = [axis for axis in range(right_rank) if axis not in right_contracted and
                          axis not in right_batch]
            for position, (left, right) in enumerate(zip(left_batch, right_batch, strict=True)):
                graph.register(left_of(i64(index), i64(int(left)), i64(position)),
                               right_of(i64(index), i64(int(right)), i64(position)))
            for position, axis in enumerate(left_free):
                graph.register(left_of(i64(index), i64(axis), i64(len(left_batch) + position)))
            for position, axis in enumerate(right_free):
                place = len(left_batch) + len(left_free) + position
                graph.register(right_of(i64(index), i64(axis), i64(place)))
            for axis in left_contracted:
                graph.register(contracts_left(i64(index), i64(int(axis))))
            for axis in right_contracted:
                graph.register(contracts_right(i64(index), i64(int(axis))))
    for index in range(len(closed.jaxpr.constvars)):
        graph.register(local(Term.leaf(i64(index))), unwalked(Term.leaf(i64(index))))
    first = len(closed.jaxpr.constvars)
    for index, input_name in enumerate(names):
        leaf = Term.leaf(i64(first + index))
        graph.register(local(leaf))
        if input_name in walked_axes:
            graph.register(set_(walked(leaf)).to(i64(walked_axes[input_name])))
        else:
            graph.register(unwalked(leaf))
    for index, value in enumerate(region.literals):
        literal = Term.literal(i64(index))
        graph.register(local(literal), unwalked(literal))
        if np.all(np.asarray(value) == 0):
            graph.register(zero(literal))


def _least(graph: EGraph, region: terms.Region, closed) -> None:
    """The walked maxima's least finite value, a literal of their dtype (`least`), refused where the maxima
    are of more than one dtype: no one shift is each one's."""
    peaks = {jnp.dtype(equation.outvars[0].aval.dtype) for equation in closed.jaxpr.eqns
             if equation.primitive.name == "reduce_max"}
    if len(peaks) > 1:
        raise Refused(f"maxima of more than one dtype ({sorted(map(str, peaks))}): no one shift corrects "
                      "them")
    if peaks and jnp.issubdtype(dtype := peaks.pop(), jnp.floating):
        region.literals.append(np.asarray(jnp.finfo(dtype).min, dtype))
        literal = Term.literal(i64(len(region.literals) - 1))
        region.shaped(literal, ())
        graph.register(set_(least()).to(i64(len(region.literals) - 1)),
                       set_(rules.shape(literal)).to(Vec[i64]()))


def _moved(primitive, params, region: terms.Region, index: int) -> list[tuple[int, int]]:
    """A squeeze's, an expand_dims' or a reshape's moves of its operand's axes, where each axis of extent
    other than one keeps its order: each such axis's place before and after."""
    before, after = _operand_and_result_shapes(region, index)
    if before is None or after is None:
        return []
    kept_before = [axis for axis, extent in enumerate(before) if extent != 1]
    kept_after = [axis for axis, extent in enumerate(after) if extent != 1]
    if [before[axis] for axis in kept_before] != [after[axis] for axis in kept_after]:
        return []
    return list(zip(kept_before, kept_after, strict=True))


def _operand_and_result_shapes(region: terms.Region, index: int):
    """The shapes of the operand and the result of a single-operand term of parameters `index`."""
    for term, extent in region.shapes:
        args = get_callable_args(term, Term.shaped) or get_callable_args(term, Term.reduce)
        if args is not None and int(get_literal_value(args[-1])) == index:
            return region.extents.get(str(args[0])), extent
    return None, None


def _rank_before(region: terms.Region, index: int, reduced: set[int]) -> int:
    before, after = _operand_and_result_shapes(region, index)
    return len(before) if before is not None else (len(after) + len(reduced) if after is not None else 0)


def _dot_ranks(region: terms.Region, index: int) -> tuple[int, int]:
    for term, _ in region.shapes:
        args = get_callable_args(term, Term.dot)
        if args is not None and int(get_literal_value(args[-1])) == index:
            return len(region.shape(args[0])), len(region.shape(args[1]))
    return 0, 0


def _opaque_locals(graph: EGraph, region: terms.Region) -> list[Term]:
    """The outputs of equations outside the language (`terms.Opaque`) over local, unwalked inputs alone: local
    themselves; an equation outside the language over a walked value is refused, its axis unknown."""
    found = []
    for opaque in region.opaque:
        inputs = [term for term in opaque.inputs if term is not None]
        if any(_walked(graph, term) for term in inputs):
            raise Refused(f"{opaque.equation.primitive.name}: an equation outside the language over a walked "
                          "value")
        if all(_holds(graph, local(term)) and _holds(graph, unwalked(term)) for term in inputs):
            for leaf in opaque.outputs:
                found.append(Term.leaf(i64(leaf)))
                graph.register(unwalked(Term.leaf(i64(leaf))))
    return found


def _lift(graph: EGraph, region: terms.Region, reductions) -> None:
    """Each broadcast a rescaled sum needs, an anchor's shape to the sum's (`lifts`), its parameters added to
    the region's table: the anchor's axes its leading ones."""
    anchors = {extent for term, extent in region.shapes if _holds(graph, anchor(term))}
    lifted = set()
    for reduction in reductions:
        to = region.extents.get(str(reduction))
        if to is None:
            continue
        for given in anchors:
            if tuple(to[:len(given)]) != tuple(given) or (given, tuple(to)) in lifted:
                continue
            lifted.add((given, tuple(to)))
            params = {"shape": tuple(to), "broadcast_dimensions": tuple(range(len(given))), "sharding": None}
            region.params.append((jax.lax.broadcast_in_dim_p, params))
            graph.register(set_(lifts(Vec[i64](*(i64(size) for size in given)),
                                      Vec[i64](*(i64(size) for size in to)))).to(i64(len(region.params) - 1)))


def _walked(graph: EGraph, term: Term) -> bool:
    """Whether a value has a walked axis (`walked`)."""
    return graph.lookup_function_value(walked(term)) is not None


def _holds(graph: EGraph, fact) -> bool:
    try:
        graph.check(fact)
    except EggSmolError:
        return False
    return True


def _identity(reduction: Term, dtype) -> float:
    """A walked reduction's empty value, its monoid's identity: a sum's zero; a maximum's -inf (its dtype's
    least value where it has no infinity). A correction never reads it unshifted (`least`)."""
    if _monoid(reduction) != "max":
        return 0.0
    return -math.inf if jnp.issubdtype(dtype, jnp.floating) else float(jnp.iinfo(dtype).min)


def _reads_leaves(form: Term) -> set[int]:
    """The region's leaves `form` reads."""
    return {int(get_literal_value(args[0])) for term in terms.subterms(form)
            if (args := get_callable_args(term, Term.leaf)) is not None}


def _written(form: Term) -> bool:
    """Whether `form` keeps none of the derivation's readings: a form in the language."""
    return not any(get_callable_args(term, reading) is not None for term in terms.subterms(form)
                   for reading in _READINGS)
