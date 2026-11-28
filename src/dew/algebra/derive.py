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

import collections
import dataclasses
import functools
import itertools
import logging
import math
import numbers
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from egglog import (
    DagCostModel,
    EGraph,
    Ruleset,
    Schedule,
    String,
    Unit,
    Vec,
    eq,
    function,
    i64,
    rewrite,
    rule,
    ruleset,
    seq,
    set_,
    union,
    var,
    vars_,
)
from egglog.bindings import EggSmolError
from jax.extend import core as jcore

from dew import _cache
from dew._cache import ALGEBRA, unintercepted
from dew._facts import Facts, Held, recording, replay
from dew.algebra import cost, derive_moments, extract, program, rules, stored, terms
from dew.algebra.cost import Work
from dew.algebra.derive_moments import (
    MOMENT_SSA,
    anchor_slot,
    anchored_first,
    boolean,
    filled,
    hi_slot,
    identity_broadcast,
    inserts_axis,
    is_square,
    moment_anchor,
    moment_hi,
    moment_scale,
    nonnegative,
    plain_local,
    positive_guard,
    scale_slot,
    scaled_second,
    unit_array,
    unreduced,
)
from dew.algebra.forms import Accumulated, Added, Derived, Merged, Refused, evaluated, evaluator, inlined
from dew.algebra.terms import Term
from dew.nn.kernels import generation
from dew.nn.kernels.templates import report as kernel_report

Axes = tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Reduction:
    """A walk whose state is monoids' (attention): `function(**operands)` on whole operands in plain JAX,
    each operand's axes by name (`operands`), the axis the kernel walks in tiles (`walked`), and the
    exponent's base the template takes (`exponent`: 2 where it folds `log2(e)` into its logits, `"e"`
    otherwise), which the derivation keeps; `ranges` bounds operands' magnitudes by name, the range facts a
    reordering's new intermediates are held under (docs/design/algebra.md §6.3), an operand it names none
    of unbounded."""

    function: Callable
    operands: Mapping[str, Axes]
    walked: str
    exponent: Literal[2, "e"] = "e"
    ranges: Mapping[str, float | tuple[float, float]] = dataclasses.field(default_factory=dict)
    exact: bool = False
    """Whether its forms are extracted exactly (`extract.cheapest`): a declaration whose greedy extraction the
    test suite's oracle finds dearer than the optimum (`dew._cache.Passes.checked`)."""
    exponential: Callable[..., jax.Array] | None = None
    """The implementation of its exponential (`exponent`'s base), `exponential(x, scale=1.0)`, `exp(x scale)`
    in the reals, which its walk's every exponential of that base evaluates by, an exponential of a
    difference times a literal taking the literal as its `scale` (raw logits', `forms._scaled`): a numerics
    choice of the declaration's (attention's accurate form's `natural_exp`, as exact as the reference's on
    every backend), equal to the plain one in the reals; None for JAX's."""
    rescale: Callable[[jax.Array, jax.Array], jax.Array] | None = None
    """The factor a carried state takes from its anchor `carried` to the walk's new one `now`, the walk's
    weights' exponential of their difference: `exp(carried - now)` in the reals, or `exp((carried - now) c)`
    where the weights are `exp((x - b) c)` (raw logits, `c` the declaration's literal, which the rescale
    carries itself: only a declaration that states its rescale walks scaled weights, `WALKING_SCALED`),
    which the walk's rescale evaluates by where given (attention's accurate form's `carried`, its exponent's
    rounding compensated); None for the plain exponential of the difference."""


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
def scales(term: Term, by: Term, name: String, exponent: i64, difference: i64, at: i64) -> Unit: ...
"""A value that scales as `exp(-by)` (`name` the exponential, `exponent` and `difference` its and its
subtraction's parameters): `exp(x - by)` with `x` local, and what multiplies, selects or sums it with local
values; `at` the value's axis along which the anchor's own lies (-1 for a scalar anchor), as its broadcast
placed it and each equation since moved it."""


@function(merge=lambda old, new: old)
def anchored(term: Term) -> i64: ...
"""The axis of a walked sum (`summed`) along which its anchor's lies (-1 for a scalar anchor): where its
correction broadcasts (`lifts`)."""


@function
def zero(term: Term) -> Unit: ...
"""A value of zeros (a literal, or a layout of one)."""


@function(merge=lambda old, new: old)
def lifts(given: Vec[i64], to: Vec[i64], at: i64) -> i64: ...
"""The parameters of a broadcast of an anchor of shape `given` to `to`, its axis at `to`'s `at` (none for a
scalar anchor's, -1)."""


@function(merge=lambda old, new: old)
def retype(slot: i64) -> i64: ...
"""The parameters of a conversion to a walked reduction's dtype, by its index (`slot`): a sum's correction,
of its anchor's dtype, converted to its accumulator's (a narrow exponent summed in float32, as JAX sums a
narrow value) before it scales the carried sum."""


@function(merge=lambda old, new: old)
def least(anchor: Term) -> i64: ...
"""The literal of an anchor's dtype's least finite value (`Region.literals`), each anchor its own dtype's: a
correction's shift `max(k_now, least)` keeps `exp(k_carried - k_now)` zero, not `exp(-inf + inf)`, where
nothing is seen yet."""


@function(merge=lambda old, new: old)
def rescaled_by(step: Term) -> Term: ...
"""The row factor a sum's step rescales its carried state by (`state * lift(factor) + weights @ values`), as
the rule that wrote the step states it: a mainloop takes such a step into its own accumulator."""


@function(merge=lambda old, new: old)
def weighed(step: Term) -> Term: ...
"""The weights of the tile's product a sum's step adds (`rescaled_by`'s)."""


@function(merge=lambda old, new: old)
def weighted(step: Term) -> Term: ...
"""The values of the tile's product a sum's step adds (`rescaled_by`'s)."""


@function(merge=lambda old, new: old)
def summand(step: Term) -> Term: ...
"""The tile's sum a reduction's step adds to its rescaled state (`state * lift(factor) + sum`, `rescaled_by`'s
factor): a mainloop's own addition takes it, as its accumulator takes a product's parts."""


@function(merge=lambda old, new: old)
def merge_factors(merge: Term) -> Term: ...
"""The factor a merge of two rescaled sums takes its first state by (its maximum to the joined one), as the
rule writes the merge: a mainloop's own merge takes it (`Merged`)."""


@function(merge=lambda old, new: old)
def merge_second(merge: Term) -> Term: ...
"""The factor the same merge takes its second state by."""


@function(merge=lambda old, new: old)
def summed_terms(step: Term) -> Term: ...
"""The tile's terms `summand` sums, before their reduction: a mainloop's addition that takes them apart (a
guard's shift on each) reads them."""


@function(merge=lambda old, new: old)
def slot(term: Term) -> i64: ...
"""A walked reduction's index among the walk's, its carried state's (`Term.state`)."""


@function(merge=lambda old, new: old)
def at_least(term: Term) -> i64: ...
"""A floor `term` is at least (`floor_of`'s class): a floored maximum's carried state (`_floors`), and a
maximum of a value at least it."""


@function(merge=lambda old, new: old)
def floor_of(literal: Term) -> i64: ...
"""The class of a literal's value among the walk's floors (`_floors`): literals of one value one class."""


_log = logging.getLogger(__name__)

EXPONENTIALS = {"e": "exp", 2: "exp2"}
"""The language's exponential of each base a declaration states (`Reduction.exponent`): the exponentials a
walk's weights scale by (`_exponentials`, `scales`), whose parameters a declaration's own rescale reads."""


def _exponentials(scaled: bool = False) -> list:
    """The rules that find what scales as `exp(-k)` of an anchor `k` (`scales`), from its exponential; and
    `scaled`, as `exp(-c k)`, an exponential of a difference times a positive literal `c` (raw logits, the
    scale after the subtraction), which a declaration's own rescale alone carries (`Reduction.rescale`)."""
    x, y, z, k, t, base, c_ = vars_("x y z k t base c_", Term)
    name = var("name", String)
    p, q, c, d, e, s, f = vars_("p q c d e s f", i64)
    axes = var("axes", Vec[i64])
    found = []
    # An exponential of a local value less an anchor's broadcast, with the axis along which the anchor's own
    # lies (`at`, -1 for a scalar anchor's), as its broadcast placed it. The same where the largest value's
    # exponent is held to 0 by a select, `exp(where(x < b, x - b, 0))`, `b` the broadcast of a bound of every
    # `x` (so the two are equal in the reals): attention's accurate form's weights over scaled logits, its
    # largest weighing exactly 1 against a compiler's contraction.
    for power in EXPONENTIALS.values():
        difference = x.binary(String("sub"), base, s)
        held = x.compare(String("lt"), base, d).select(y, difference)
        broadcast = (local(x), eq(base).to(k.shaped(String("broadcast_in_dim"), q)), anchor(k),
                     eq(name).to(String(power)))
        # `b` at least every `x` (its own maximum, or that maximum floored, `max(max(x), z)`), so that where
        # `x < b` fails the difference is 0 too.
        bounds = (eq(k).to(x.reduce(String("max"), axes, p)),
                  eq(k).to(x.reduce(String("max"), axes, p).binary(String("max"), z, f)),
                  eq(k).to(z.binary(String("max"), x.reduce(String("max"), axes, p), f)))
        forms = ((difference.unary(name, e), ()),
                 *((held.unary(name, e), (zero(y), bound)) for bound in bounds))
        if scaled:
            forms = tuple((product.unary(name, e), (rules.positive_scale(c_), eq(c_).to(Term.literal(f))))
                          for product in (difference.binary(String("mul"), c_, d),
                                          c_.binary(String("mul"), difference, d)))
        for exponential, conditions in forms:
            found += [
                rule(eq(t).to(exponential), *broadcast, *conditions, lays(q, i64(0), c)).then(
                    scales(t, k, name, e, s, c)),
                rule(eq(t).to(exponential), *broadcast, *conditions, eq(rules.shape(k)).to(Vec[i64]())).then(
                    scales(t, k, name, e, s, i64(-1))),
            ]
    return found


def _ternary_walking(ternary, t: Term, x: Term, y: Term, z: Term, a) -> list:
    """`_walking`'s rules of a term of three operands made by `ternary` (`Term.select`, `Term.clamp`)."""
    made = eq(t).to(ternary(x, y, z))
    return [
        *(rule(made, eq(walked(held)).to(a)).then(set_(walked(t)).to(a)) for held in (x, y, z)),
        rule(made, local(x), local(y), local(z)).then(local(t)),
        rule(made, unwalked(x), unwalked(y), unwalked(z)).then(unwalked(t)),
        *(rule(made, *(kind(held) for kind, held in zip(kinds, (x, y, z), strict=True))).then(anchor(t))
          for kinds in itertools.product((local, anchor), repeat=3) if anchor in kinds),
    ]


def _walking(scaled: bool = False):
    """The facts' rules (the module's account); `scaled`, with an exponential of a difference times a positive
    literal among what scales (`_exponentials`)."""
    x, y, z, k, t = vars_("x y z k t", Term)
    name, layout = vars_("name layout", String)
    p, q, a, b, c, d, e, s = vars_("p q a b c d e s", i64)
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
        # A selection and a clamp, each of three operands: walked where one is, local or unwalked where all
        # are, an anchor where one is and every other is local or an anchor.
        *(each for ternary in (Term.select, Term.clamp) for each in _ternary_walking(ternary, t, x, y, z, a)),
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
    # What scales as exp(-k) (`_exponentials`), and what multiplies, divides, selects against zeros, sums,
    # casts or lays it out with local values, each equation moving the anchor's axis.
    walking += _exponentials() + (_exponentials(scaled=True) if scaled else [])
    for side in (0, 1):
        def product(scaled, other, params, side=side):
            return scaled.binary(String("mul"), other, params) if side == 0 else other.binary(
                String("mul"), scaled, params)
        moved = left_of(p, c, d) if side == 0 else right_of(p, c, d)
        walking.append(rule(eq(t).to(product(x, y, p)), scales(x, k, name, e, s, c), local(y)).then(
            scales(t, k, name, e, s, c)))
        walking.append(rule(eq(t).to(x.dot(y, p) if side == 0 else y.dot(x, p)), scales(x, k, name, e, s, c),
                            local(y), moved).then(scales(t, k, name, e, s, d)))
    walking += [
        rule(eq(t).to(x.binary(String("div"), y, p)), scales(x, k, name, e, s, c), local(y)).then(
            scales(t, k, name, e, s, c)),
        rule(eq(t).to(x.binary(String("add"), y, p)), scales(x, k, name, e, s, c),
             scales(y, k, name, e, s, c)).then(scales(t, k, name, e, s, c)),
        rule(eq(t).to(z.select(y, x)), scales(x, k, name, e, s, c), zero(y), local(z)).then(
            scales(t, k, name, e, s, c)),
        rule(eq(t).to(z.select(x, y)), scales(x, k, name, e, s, c), zero(y), local(z)).then(
            scales(t, k, name, e, s, c)),
        rule(eq(t).to(x.convert(p)), scales(x, k, name, e, s, c)).then(scales(t, k, name, e, s, c)),
        rule(eq(t).to(x.shaped(layout, p)), scales(x, k, name, e, s, c), lays(p, c, d)).then(
            scales(t, k, name, e, s, d)),
        rule(eq(t).to(x.reduce(String("sum"), axes, p)), scales(x, k, name, e, s, c), eq(walked(x)).to(a),
             reduces(p, a, b), reduces(p, c, d)).then(scales(t, k, name, e, s, d)),
        rule(eq(t).to(x.reduce(String("sum"), axes, p)), eq(walked(x)).to(a), eq(axes).to(Vec[i64](a)),
             scales(x, k, name, e, s, c), reduces(p, c, d)).then(summed(t), set_(anchored(t)).to(d)),
        rule(eq(t).to(x.dot(y, p)), eq(walked(x)).to(a), contracts_left(p, a), scales(x, k, name, e, s, c),
             local(y), left_of(p, c, d)).then(summed(t), set_(anchored(t)).to(d)),
        rule(eq(t).to(x.dot(y, p)), eq(walked(y)).to(a), contracts_right(p, a), scales(y, k, name, e, s, c),
             local(x), right_of(p, c, d)).then(summed(t), set_(anchored(t)).to(d)),
        *derive_moments.moment_walking(local, summed, walked, zero),
    ]
    return ruleset(*walking, name="walking scaled" if scaled else "walking")


def _shapes():
    """The shapes of the values the derivation's rules make: a carried state its reduction's, a value a step,
    merge, finish or carried state reads its own, a comparison its operands', a select its operands', a clamp
    its operand's, a lifted correction or broadcast its target."""
    x, y, z, w = vars_("x y z w", Term)
    name = var("name", String)
    side, at, p = vars_("side at p", i64)
    extent = var("extent", Vec[i64])
    given, to = vars_("given to", Vec[i64])
    shape = rules.shape
    return ruleset(
        rule(eq(extent).to(shape(x)), eq(y).to(x.held(side))).then(set_(shape(y)).to(extent)),
        rule(eq(extent).to(shape(x)), eq(y).to(x.stepped())).then(set_(shape(y)).to(extent)),
        rule(eq(extent).to(shape(x)), eq(y).to(x.merged())).then(set_(shape(y)).to(extent)),
        rule(eq(extent).to(shape(x)), eq(y).to(x.finished())).then(set_(shape(y)).to(extent)),
        rule(eq(extent).to(shape(x)), eq(y).to(moment_anchor(x))).then(set_(shape(y)).to(extent)),
        rule(eq(extent).to(shape(x)), eq(y).to(moment_hi(x))).then(set_(shape(y)).to(extent)),
        rule(eq(extent).to(shape(x)), eq(y).to(moment_scale(x))).then(set_(shape(y)).to(extent)),
        rule(eq(w).to(x.compare(name, y, p)), eq(extent).to(shape(x)), eq(shape(y)).to(extent)).then(
            set_(shape(w)).to(extent)),
        rule(eq(extent).to(shape(y)), eq(w).to(x.select(y, z))).then(set_(shape(w)).to(extent)),
        rule(eq(extent).to(shape(y)), eq(w).to(x.clamp(y, z))).then(set_(shape(w)).to(extent)),
        rule(eq(y).to(x.shaped(String("broadcast_in_dim"), lifts(given, to, at)))).then(
            set_(shape(y)).to(to)),
        rule(eq(y).to(x.shaped(String("broadcast_in_dim"), filled(to)))).then(
            set_(shape(y)).to(to)),
        rule(eq(y).to(x.shaped(String("broadcast_in_dim"), unreduced(to, at)))).then(
            set_(shape(y)).to(to)),
        name="derived shapes")


def _floored():
    """The floors' rules (the module's account): a maximum is at least what either side is, and a maximum
    of a value at least a floor with that floor's literal is the value."""
    x, y, t = vars_("x y t", Term)
    c, p = vars_("c p", i64)
    return ruleset(
        rule(eq(t).to(x.binary(String("max"), y, p)), eq(at_least(x)).to(c)).then(set_(at_least(t)).to(c)),
        rule(eq(t).to(y.binary(String("max"), x, p)), eq(at_least(x)).to(c)).then(set_(at_least(t)).to(c)),
        rule(eq(t).to(x.binary(String("max"), y, p)), eq(at_least(x)).to(c), eq(floor_of(y)).to(c)).then(
            union(t).with_(x)),
        rule(eq(t).to(y.binary(String("max"), x, p)), eq(at_least(x)).to(c), eq(floor_of(y)).to(c)).then(
            union(t).with_(x)),
        name="floored")


FLOORED = _floored()
"""A floored maximum's readings: its floor taken once, at its identity (`_floors`)."""

WALKING = _walking()
WALKING_SCALED = _walking(scaled=True)
"""`WALKING` for a declaration that states its own rescale (`Reduction.rescale`), which carries a scaled
exponential's factor."""
SHAPES = _shapes()


def _parts_of(value: Callable, accumulated, added) -> Accumulated | Added | None:
    """A reduction's step's parts as a mainloop's own step takes them (`Derived.step`'s `stepped`), each
    evaluated by `value`: a product's (`Accumulated`), a sum's (`Added`), or None."""
    if accumulated is not None:
        factor, weights, values = accumulated
        return Accumulated(None if factor is None else value(factor), value(weights), value(values))
    if added is not None:
        factor, summed, terms_of = added
        return Added(value(factor), value(summed), value(terms_of))
    return None


def _decoupled(rescaling: bool = False):
    """The derivation's rules (the module's account): each reading distributes over the language, and a walked
    reduction's step, merge and finish are its monoid's, a sum that scales with an anchor rescaled first, by
    the plain exponential of the anchors' difference, or `rescaling` by the declaration's own rescale
    (`Reduction.rescale`, the language's binary `rescale` of the two anchors, which no rule rewrites)."""
    x, y, z, k, t = vars_("x y z k t", Term)
    name = var("name", String)
    p, e, s, side, a, b, at, along, lift, added, multiplied, largest, i, floor, typed = vars_(
        "p e s side a b at along lift added multiplied largest i floor typed", i64)
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
            rewrite(read(x.clamp(y, z))).to(read(x).clamp(read(y), read(z))),
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
    # A walked sum of local products states its weights and values as it writes them (no factor: it takes no
    # rescale), which a mainloop's own accumulator takes (`Accumulated`).
    stepped, merged = vars_("stepped merged", Term)
    derived.append(rule(eq(stepped).to(t.stepped()), eq(t).to(x.dot(y, p)), summed(t), plain_local(t),
                        slotted).then(set_(weighed(stepped)).to(x.stepped()),
                                      set_(weighted(stepped)).to(y.stepped())))
    for combined, which, index in (("max", peaked, largest), ("add", summed, added)):
        stocked = eq(index).to(stock(combined))
        derived += [
            rewrite(t.stepped()).to(Term.state(i, 0).binary(combined, t, index), which(t), plain_local(t),
                                    stocked, slotted),
            rewrite(t.merged()).to(Term.state(i, 0).binary(combined, Term.state(i, 1), index), which(t),
                                   plain_local(t), stocked, slotted),
        ]
    derived += [
        rewrite(t.finished()).to(Term.state(i, 0), peaked(t), slotted),
        rewrite(t.finished()).to(Term.state(i, 0), summed(t), plain_local(t), slotted),
        rewrite(t.held(side)).to(Term.state(i, side), peaked(t), slotted),
        rewrite(t.held(side)).to(Term.state(i, side), summed(t), plain_local(t), slotted),
    ]
    derived += derive_moments.moment_decoupled(slot)

    # A sum along the walked axis of values that scale with an anchor `k`: its carried value rescaled by
    # exp(k carried - k now), lifted to its shape, beside the tile's sum at the anchor now; the anchor now
    # at least the least finite value, so a state that has seen nothing (both -inf) rescales by 0.
    def correction(carried_side, now):
        shifted = now.binary(String("max"), Term.literal(floor), largest)
        if rescaling:
            return k.held(carried_side).binary(String(terms.RESCALE), shifted, e).convert(typed)
        return k.held(carried_side).binary(String("sub"), shifted, s).unary(name, e).convert(typed)

    def rescaled(carried_side, now):
        lifted = correction(carried_side, now).shaped(String("broadcast_in_dim"), lift)
        return Term.state(i, carried_side).binary(String("mul"), lifted, multiplied)

    lifted = (eq(given).to(shape(k)), eq(to).to(shape(t)), eq(along).to(anchored(t)),
              eq(lift).to(lifts(given, to, along)),
              eq(added).to(stock("add")), eq(multiplied).to(stock("mul")), eq(largest).to(stock("max")),
              eq(floor).to(least(k)), eq(typed).to(retype(i)), slotted)
    for summed_term, tile in ((x.reduce(String("sum"), axes, p), x.stepped().reduce(String("sum"), axes, p)),
                              (x.dot(y, p), x.stepped().dot(y.stepped(), p)),
                              (y.dot(x, p), y.stepped().dot(x.stepped(), p))):
        conditions = (eq(t).to(summed_term), summed(t), scales(x, k, name, e, s, at), *lifted)
        # A step states its parts as it writes them, whatever form extraction then picks of the step: the
        # factor (`rescaled_by`) and a product's weights and values a mainloop's own accumulator takes, or a
        # reduction's tile sum its own addition takes (`summand`).
        product = terms.args_of(tile, Term.dot)
        parts = (set_(rescaled_by(stepped)).to(correction(0, k.stepped())),
                 *((set_(summand(stepped)).to(tile), set_(summed_terms(stepped)).to(x.stepped()))
                   if product is None else
                   (set_(weighed(stepped)).to(product[0]), set_(weighted(stepped)).to(product[1]))))
        derived += [
            rule(eq(stepped).to(t.stepped()), *conditions).then(
                union(stepped).with_(rescaled(0, k.stepped()).binary(String("add"), tile, added)), *parts),
            rule(eq(merged).to(t.merged()), *conditions).then(
                union(merged).with_(rescaled(0, k.merged()).binary(String("add"), rescaled(1, k.merged()),
                                                                   added)),
                set_(merge_factors(merged)).to(correction(0, k.merged())),
                set_(merge_second(merged)).to(correction(1, k.merged()))),
            rewrite(t.finished()).to(Term.state(i, 0), *conditions),
        ]
    return ruleset(*derived, name="decoupled rescaled" if rescaling else "decoupled")


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
                        receipt="docs/design/algebra.md §6.9; tests/test_algebra_derive.py; measured M-IE-1 "
                                "(the derived attention forward against the hand forms it replaced, GH200: "
                                "at parity, 0.986x to 1.008x, and bitwise)")
"""The derivation's rules, run beside the engine's sets (`rules.admitted`) and never among them."""
DECOUPLED_RESCALED = rules.Rules("decoupled rescaled", "reorders", _decoupled(rescaling=True),
                                 receipt="docs/design/algebra.md §6.9; tests/test_algebra_derive.py; "
                                         "measured M-ET-80 (the accurate form's derived training step on the "
                                         "route, GH200) and M-ET-76 (its kernels at the stability entries, "
                                         "A100 and GH200)")
"""`DECOUPLED` for a declaration that states its own rescale (`Reduction.rescale`)."""
PLAIN = _plain()


_READINGS = frozenset(map(terms.ref, (Term.stepped, Term.merged, Term.finished, Term.held, moment_anchor,
                                       moment_hi, moment_scale, *MOMENT_SSA)))
"""The derivation's readings by constructor (`terms.made_by`): a form that keeps one is no form."""


def _priced(region: terms.Region, schedule) -> DagCostModel[cost.Seconds]:
    """The template's tile walk's price of each node (`cost.model` under `cost.walk`: a form the template's
    algebra takes runs between its mainloop's products, not as XLA fuses it), each of the derivation's
    readings infinite in every schedule (`cost.Seconds.extra`: a form that keeps one is no form, whatever
    rates the target states)."""
    timed = cost.model(region, schedule)

    def marginal(graph: EGraph, node) -> cost.Seconds:
        if terms.made_by(node) in _READINGS:
            return cost.Seconds(Work(), schedule, math.inf)
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
    accumulated: tuple[tuple[Term | None, Term, Term] | None, ...] = ()
    """Each reduction's step as a mainloop's accumulator takes it (`rescaled_by`): its row factor (None for a
    sum that takes no rescale), the tile product's weights and values, extracted with the forms; None for a
    step of no such rule."""
    added: tuple[tuple[Term, Term, Term] | None, ...] = ()
    """Each reduction's step as a mainloop's own addition takes it (`summand`): its row factor, the tile's sum
    and the terms it sums (`summed_terms`), extracted with the forms; None for a step of no such rule."""
    merged: tuple[tuple[Term, Term] | None, ...] = ()
    """Each reduction's merge as a mainloop's own merge takes it (`merge_factors`): the factors its first and
    second states take, extracted with the forms; None for a merge of no such rule."""
    floors: tuple[float | None, ...] = ()
    """Each reduction's floor, the identity it is carried from where the declaration reads it only floored
    (`_floors`); None for every other."""
    anchors: tuple[Term, ...] = ()
    """First-moment reductions carrying a companion base-mean anchor slot (`moment_anchor`)."""
    scales: tuple[Term, ...] = ()
    """Central second-moment reductions carrying a companion zero-mass scale slot (`moment_scale`)."""


def context() -> tuple:
    """What a derivation reads where it runs beside its declaration and shapes, the key of every derivation a
    declaration holds (`derive`'s forms, beside the rule sets its numerics admit, `rules.admitted_names`;
    `scan.derive_scan`'s carry splits): forms traced under one abstract
    mesh hold its avals in their literals (a sharded step's bound under a local one's mesh refused to lower),
    and are priced for one target's pricing facts (`cost.pricing_key`), under one run's numerics and
    precision, with 64-bit values on or off, and under each of JAX's promotion modes (rank and dtype): what a
    declaration's trace may read of JAX's numerics, not the device a trace is assumed for (its capacity
    priced apart, `cost.pricing_key`).
    The extractor oracle may be on or off (`Passes.checked`), and the pass may leave out a registered set
    of rules (`Passes.without`, a receipt's arm, which shares no derivation or recognition with another arm).
    Identity-keyed collections (`dew._cache.KERNEL_ROUTES`, `KERNEL_SEEN`) are not part of this key:
    they are new in each lowering, which would derive again and grow the cache without end."""
    return (jax.sharding.get_abstract_mesh(), cost.pricing_key(cost.target()), generation.numerics(),
            jax.config.jax_default_matmul_precision, jax.config.jax_enable_x64,
            jax.config.jax_numpy_rank_promotion, jax.config.jax_numpy_dtype_promotion, ALGEBRA.value.checked,
            ALGEBRA.value.without, ALGEBRA.value.dev)

FORMS_HELD = 64
"""The derived forms one declaration keeps, its least recently traced dropped past them: a process's distinct
meshes, targets, numerics and tile shapes, each a saturation's seconds to derive again."""


def derive(declaration: Reduction, *, accurate: bool, saturation: program.Saturation) -> Derived:
    """`declaration`'s monoid forms along its walked axis (the module's account), saturated beside the
    engine's sets `accurate` admits under `saturation`'s limits, each tile shape's rewrite in the kernel
    report (`report.rewrites`); refused (`Refused`) where a form keeps a reading no rule rewrites."""
    names = tuple(declaration.operands)
    walked_axes = {name: axes.index(declaration.walked) for name, axes in declaration.operands.items()
                   if declaration.walked in axes}
    # The declaration's own numerics its forms evaluate by (`Reduction.exponential`, `rescale`).
    declared = {"exponential": None if declaration.exponential is None else
                (EXPONENTIALS[declaration.exponent], declaration.exponential),
                "rescale": declaration.rescale}
    held: collections.OrderedDict = collections.OrderedDict()
    classes: collections.OrderedDict[Hashable, Held[dict | None]] = collections.OrderedDict()

    def derived_at(keyed: tuple) -> _Forms:
        shapes = {name: (shape, dtype) for name, shape, dtype in keyed}
        made = _derived(declaration, names, walked_axes, shapes, accurate, saturation, classes)
        kernel_report.record_rewrite(made.report)
        return made

    def forms(shapes: Mapping) -> _Forms:
        # The engine's sets saturate beside the derivation's own (`_derived`), so a set registered after a
        # shape's forms were derived derives them afresh, as a program's windows do (`program._deriving`);
        # held through `_cache.holding`, the latest FORMS_HELD.
        keyed = tuple((name, tuple(shapes[name][0]), np.dtype(shapes[name][1])) for name in names)
        return _cache.holding(held, (context(), rules.admitted_names(accurate), keyed),
                              lambda: derived_at(keyed), FORMS_HELD)

    def shaped(values):
        return {name: (value.shape, value.dtype) for name, value in values.items()}

    def init(shapes):
        made = forms(shapes)
        avals = [jax.ShapeDtypeStruct(*shapes[name]) for name in names]
        slot_terms = (*made.reductions, *made.anchors, *made.scales, *made.scales)
        slot_floors = (*made.floors, *(math.inf,) * len(made.anchors), *(0.0,) * len(made.scales),
                       *(1.0,) * len(made.scales))
        typed = jax.eval_shape(lambda *values: tuple(evaluated(made.region, reduction,
                                                               [*made.consts, *values], {})
                                                     for reduction in slot_terms), *avals)
        return tuple(jnp.full(aval.shape, _identity(reduction, aval.dtype) if floor is None else floor,
                              aval.dtype)
                     for reduction, aval, floor in zip(slot_terms, typed, slot_floors, strict=True))

    def carried(made: _Forms, *states) -> dict:
        return {terms.key(Term.state(i64(index), i64(side))): state[index]
                for side, state in enumerate(states)
                for index in range(len(made.steps))}

    def step(state, tiles, *, product=None, stepped=None, outputs: Sequence[int] = ()):
        made = forms(shaped(tiles))
        value = evaluator(made.region, [*made.consts, *(tiles[name] for name in names)],
                          carried(made, state), product=product, **declared)
        # The tile's outputs along the walked axis `outputs` names (by place), by the same evaluation: a
        # term both read, once.
        along = []
        for index in outputs:
            if not made.mapped[index] or (output := made.outputs[index]) is None:
                raise Refused(f"output {index} is no output along the walked axis")
            along.append(value(output))
        # The mainloop's own step where it gives one, once, over each reduction's parts as the rules state
        # them (each value evaluated once), every reduction it leaves (None) by its form. Every value the
        # step computes whatever the mainloop takes (each reduction of no parts, the outputs along the walk)
        # before it: none written while its products run (a Hopper group's registers, which ptxas otherwise
        # serializes, C7513).
        parts = ((None,) * len(made.steps) if stepped is None else
                 tuple(_parts_of(value, accumulated, added)
                       for accumulated, added in zip(made.accumulated, made.added, strict=True)))
        steps = [value(form) if part is None else None for part, form in zip(parts, made.steps, strict=True)]
        taken = ((None,) * len(made.steps) if stepped is None
                 else tuple(stepped(tuple(state), parts, tuple(steps))))
        stepped = [own if own is not None else made_step if made_step is not None else value(form)
                   for own, made_step, form in zip(taken, steps, made.steps, strict=True)]
        return tuple(stepped) if not outputs else (tuple(stepped), tuple(along))

    def merge(a, b, operands=None, *, shapes: Mapping, merged=None):
        made = forms(shapes)
        # A merge reads no walked operand (`_derived`), its unwalked ones whole.
        leaves = [*made.consts, *(None if operands is None or name in walked_axes else operands[name]
                                  for name in names)]
        value = evaluator(made.region, leaves, carried(made, a, b), **declared)
        # The mainloop's own merge where it gives one, once, over each rescaled sum's two factors (`step`'s).
        taken = (None,) * len(made.merges)
        if merged is not None:
            parts = tuple(None if factors is None else Merged(*map(value, factors))
                          for factors in made.merged)
            taken = tuple(merged(tuple(a), tuple(b), parts))
        return tuple(value(form) if own is None else own
                     for own, form in zip(taken, made.merges, strict=True))

    def finish(state, operands=None, *, shapes: Mapping):
        made = forms(shapes)
        leaves = [*made.consts, *(None if operands is None else operands[name] for name in names)]
        value = evaluator(made.region, leaves, carried(made, state), **declared)
        # A literal output is its literal; one along the walked axis, `mapped`'s.
        outputs = [jnp.asarray(literal) if term is None else None if form is None else value(form)
                   for term, literal, form in zip(made.region.outputs, made.region.literal_outputs,
                                                  made.finishes, strict=True)]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def mapped(tiles):
        made = forms(shaped(tiles))
        value = evaluator(made.region, [*made.consts, *(tiles[name] for name in names)], {}, **declared)
        outputs = [value(form) if along and form is not None else None
                   for form, along in zip(made.outputs, made.mapped, strict=True)]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def slots(shapes) -> tuple[tuple[str, tuple[int, ...]], ...]:
        made = forms(shapes)
        return (*((_role(reduction), made.region.shape(reduction)) for reduction in made.reductions),
                *(("anchor", made.region.shape(term)) for term in made.anchors),
                *(("hi", made.region.shape(term)) for term in made.scales),
                *(("scale", made.region.shape(term)) for term in made.scales))

    return Derived(init, step, merge, finish, mapped,
                   {"rounding": "reorders", "walked": declaration.walked, "exponent": declaration.exponent,
                    "rules": ("walking", "decoupled")}, slots=slots)


def _role(reduction: Term) -> str:
    """A walked reduction's role in the carried state, as the declaration's reduction is made: a maximum's
    `max`, a sum's `sum`, a product's contracting the walked axis `contraction` (each the constructor that
    made it, `terms.reduce_of`)."""
    if terms.args_of(reduction, Term.dot) is not None:
        return "contraction"
    found = terms.reduce_of(reduction)
    if found is None or found[0] not in ("max", "sum"):
        raise Refused(f"{reduction} is no walked maximum, sum or contraction")
    return found[0]


def _traced(declaration: Reduction, names: tuple[str, ...], shapes: Mapping) -> tuple:
    """`declaration` traced at one tile's `shapes` (a model's intercepted quantized products not among it: a
    declaration is the algebra's statement), and its region (`terms.region`)."""
    def named(*values):
        return declaration.function(**dict(zip(names, values, strict=True)))

    avals = [jax.ShapeDtypeStruct(*shapes[name]) for name in names]
    ranges = {index: declaration.ranges[name] for index, name in enumerate(names)
              if name in declaration.ranges}
    with jax.disable_jit():
        closed = jax.make_jaxpr(unintercepted(inlined(named, ranges=ranges)))(*avals)
    return closed, terms.region(closed, ranges)


def _derived(declaration: Reduction, names: tuple[str, ...], walked_axes: Mapping[str, int],
             shapes: Mapping, accurate: bool, saturation: program.Saturation,
             classes: collections.OrderedDict | None = None) -> _Forms:
    """The forms at one tile's `shapes` (the module's account): its class's (`_classed`: the forms derived at
    the class's canonical sizes, held in `classes` by the class, bound at these), where a caller holds its
    classes, else stored or derived afresh (`_held`)."""
    began = time.perf_counter()
    with recording() as facts:
        closed, region = _traced(declaration, names, shapes)
    admitted = rules.admitted(accurate)
    tile = _traced_tile(declaration, names, walked_axes, shapes, closed, region, accurate, saturation, began,
                        facts)
    classed = None if classes is None else _classed(tile, admitted, classes)
    return classed if classed is not None else _held(tile, admitted)[0]


@dataclasses.dataclass(frozen=True)
class _Tile:
    """A declaration traced at one tile (`_derived`): what derives its forms there, stores them, and binds a
    record over its region."""

    declaration: Reduction
    names: tuple[str, ...]
    walked_axes: Mapping[str, int]
    shapes: Mapping
    closed: jcore.Jaxpr
    region: terms.Region
    accurate: bool
    saturation: program.Saturation
    began: float
    facts: Facts
    """The facts the tile's trace read (`_facts.recording`): what a class its own sizes seed holds."""
    traced: tuple[int, int, int]
    """The region's tables as traced, before a derivation appends to them: its parameters, literals and
    shapes (`_traced_tile`)."""

    @property
    def rescaled(self) -> frozenset[str]:
        """The exponentials a stored rescale may read (`stored._operated`)."""
        return frozenset() if self.declaration.rescale is None else frozenset(EXPONENTIALS.values())

    def assembled(self, spent: float = 0.0) -> Callable[..., _Forms]:
        """The forms a record's decoded terms assemble over the region (`_assembled`), timed from the tile's
        trace less `spent` (a class's canonical derivation the tile waited on, its own report's)."""
        return functools.partial(_assembled, region=self.region, closed=self.closed,
                                 walked_leaves=_walked_leaves(self.closed, self.names, self.walked_axes),
                                 began=self.began + spent)


def _traced_tile(declaration: Reduction, names: tuple[str, ...], walked_axes: Mapping[str, int],
                 shapes: Mapping, closed: jcore.Jaxpr, region: terms.Region, accurate: bool,
                 saturation: program.Saturation, began: float, facts: Facts) -> _Tile:
    """A tile of `region` as traced (`_Tile`), its tables counted before any derivation appends to them."""
    return _Tile(declaration, names, walked_axes, shapes, closed, region, accurate, saturation, began, facts,
                 (len(region.params), len(region.literals), len(region.shapes)))


def _held(tile: _Tile, admitted: tuple[rules.Rules, ...]) -> tuple[_Forms, dict | None]:
    """The forms at `tile`, stored or derived afresh and stored, and the record the store read or wrote
    (none where it reads and writes none): the extractor's oracle derives each afresh and reads and writes no
    store."""
    declaration, names, shapes = tile.declaration, tile.names, tile.shapes
    if ALGEBRA.value.checked:
        return _derived_afresh(declaration, names, tile.walked_axes, tile.closed, tile.region, admitted,
                               tile.saturation, tile.began), None
    # Stored across processes, keyed by everything the forms read (`stored.key`), its sets by their names
    # where the source registers them all (`rules.sourced`). Both derive from the one live set of rules.
    sourced = [sets.name for sets in admitted]
    name = None if not rules.sourced(admitted) else stored.key(tile.closed, tile.region, sourced, (
        names, tuple(sorted(tile.walked_axes.items())),
        tuple((n, tuple(shapes[n][0]), np.dtype(shapes[n][1]).name) for n in names), tile.accurate,
        (tile.saturation.growth, tile.saturation.depth), declaration.walked, declaration.exponent,
        tuple(sorted(declaration.ranges.items())), declaration.exact, declaration.rescale is None, context()))
    if name is not None and (
            read := stored.loaded(name, tile.region, tile.assembled(), rescaled=tile.rescaled)) is not None:
        return read
    made = _derived_afresh(declaration, names, tile.walked_axes, tile.closed, tile.region, admitted,
                           tile.saturation, tile.began)
    if name is None:
        kernel_report.record_stored("unstorable")
        return made, None
    return made, stored.stored(name, made.region, *tile.traced, _encoded(made))


def _derived_afresh(declaration: Reduction, names: tuple[str, ...], walked_axes: Mapping[str, int], closed,
                    region: terms.Region, admitted: tuple[rules.Rules, ...], saturation: program.Saturation,
                    began: float) -> _Forms:
    """The forms of `region`, `declaration` traced (`closed`) at one tile, derived by saturation and
    extraction (`_derived`'s, where no store holds them), on a fresh e-graph whose rule sets, the ones its
    saturations run, are declared once a thread (`program.fresh_graph`)."""
    walking = WALKING if declaration.rescale is None else WALKING_SCALED
    decoupled = DECOUPLED if declaration.rescale is None else DECOUPLED_RESCALED
    with program.fresh_graph(walking, PLAIN, rules.FACTS, SHAPES, FLOORED, decoupled.ruleset,
                             *(sets.ruleset for sets in admitted)) as graph:
        return _derived_on(graph, declaration, names, walked_axes, closed, region, admitted, saturation,
                           began, walking, decoupled)


def _derived_on(graph: EGraph, declaration: Reduction, names: tuple[str, ...],
                walked_axes: Mapping[str, int], closed, region: terms.Region,
                admitted: tuple[rules.Rules, ...], saturation: program.Saturation, began: float,
                walking: Ruleset, decoupled: rules.Rules) -> _Forms:
    """`_derived_afresh`'s forms, derived on `graph`."""
    _register(graph, region, closed, names, walked_axes)
    graph.run(walking.saturate())
    _register_local_opaques(graph, region, walking)
    graph.run(PLAIN.saturate())
    found = [term for term, _ in region.shapes if _holds(graph, summed(term)) or _holds(graph, peaked(term))]
    reductions = tuple({terms.key(term): term for term in found}.values())
    anchors = tuple(term for term in reductions if _holds(graph, anchored_first(term)))
    scales = tuple(term for term in reductions if _holds(graph, scaled_second(term)))
    _register_reduction_slots(graph, region, reductions, anchors, scales)
    outputs = list(region.outputs)
    along = [term is not None and _walked(graph, term) for term in outputs]
    for term, is_along in zip(outputs, along, strict=True):
        if is_along and term is not None and not _holds(graph, local(term)):
            raise Refused("an output along the walked axis reads a walked reduction: no tile holds it")
    roots = ([reduction.stepped() for reduction in reductions]
             + [moment_anchor(term).stepped() for term in anchors]
             + [moment_hi(term).stepped() for term in scales]
             + [moment_scale(term).stepped() for term in scales]
             + [reduction.merged() for reduction in reductions]
             + [moment_anchor(term).merged() for term in anchors]
             + [moment_hi(term).merged() for term in scales]
             + [moment_scale(term).merged() for term in scales]
             + [term.finished() for term, is_along in zip(outputs, along, strict=True)
                if term is not None and not is_along])
    declared = Term.bundle(Vec[Term](*region.roots()))
    bundle = Term.bundle(Vec[Term](*roots) if roots else Vec[Term].empty())
    graph.register(declared, bundle)
    schedule = cost.walk(cost.target())
    _, before = extract.cheapest(graph, declared, cost.model(region, schedule))
    floors = _floors(graph, region, reductions)
    facts = seq(rules.FACTS.saturate(), SHAPES.saturate(), walking.saturate(), FLOORED.saturate())
    _decoupled_through(graph, region, facts, decoupled)
    saturated = program.saturate(graph, region, (*admitted, decoupled),
                                 saturation, facts)
    # Each step a rule wrote as a mainloop's accumulator or its own addition takes it, its parts extracted
    # with the forms.
    def stated(part, reduction: Term) -> bool:
        return graph.lookup_function_value(part(reduction.stepped())) is not None

    accumulating = [index for index, reduction in enumerate(reductions) if stated(weighed, reduction)]
    adding = [index for index, reduction in enumerate(reductions) if stated(summand, reduction)]
    rescaling = [index for index, reduction in enumerate(reductions) if stated(rescaled_by, reduction)]
    parts = [part(reductions[index].stepped()) for index in range(len(reductions))
             for part in ((*((rescaled_by,) if index in rescaling else ()), weighed, weighted)
                          if index in accumulating
                          else (rescaled_by, summand, summed_terms) if index in adding else ())]
    merging = [index for index, reduction in enumerate(reductions)
               if graph.lookup_function_value(merge_factors(reduction.merged())) is not None]
    parts += [part(reductions[index].merged()) for index in merging for part in (merge_factors, merge_second)]
    if parts:
        bundle = Term.bundle(Vec[Term](*roots, *parts))
        graph.register(bundle)
    chosen, after = program.extracted(graph, bundle, _priced(region, schedule), exact=declaration.exact,
                                      checked=ALGEBRA.value.checked)
    held = iter(chosen[len(roots):])
    accumulated, added = [], []
    for index in range(len(reductions)):
        accumulated.append(((next(held) if index in rescaling else None), next(held), next(held))
                           if index in accumulating else None)
        added.append((next(held), next(held), next(held)) if index in adding else None)
    merged = [(next(held), next(held)) if index in merging else None for index in range(len(reductions))]
    chosen = chosen[:len(roots)]
    if not math.isfinite(after.value):
        unwritten = next(root for root, form in zip(roots, chosen, strict=True) if not _written(form))
        raise Refused(f"no rule writes {unwritten} in the language: a walked reduction read other than by a "
                      "monoid or a decoupled exponent")
    report = program.AlgebraReport(region.equations, tuple(saturated.fired), saturated.fired,
                                   saturated.limits, saturated.iterations, program.node_count(graph),
                                   saturated.limited, before.value, after.value, time.perf_counter() - began)
    extra_pad = (None,) * (len(anchors) + 2 * len(scales))
    count = len(reductions) + len(anchors) + 2 * len(scales)
    steps, merges, finished = chosen[:count], chosen[count:2 * count], iter(chosen[2 * count:])
    finishes = tuple(None if term is None or is_along else next(finished)
                     for term, is_along in zip(outputs, along, strict=True))
    walked_leaves = _walked_leaves(closed, names, walked_axes)
    if (unread := _unread(finishes, merges, walked_leaves)) is not None:
        raise Refused(unread)
    return _Forms(region, reductions, tuple(steps), tuple(merges), finishes, tuple(along), tuple(outputs),
                  walked_leaves, tuple(closed.consts), report, (*accumulated, *extra_pad),
                  (*added, *extra_pad), (*merged, *extra_pad), floors, anchors, scales)


def _walked_leaves(closed, names: tuple[str, ...], walked_axes: Mapping[str, int]) -> frozenset[int]:
    """The region's leaves the walked operands occupy (`_Forms.walked_leaves`), past its captured
    constants."""
    first = len(closed.consts)
    return frozenset(first + index for index, name in enumerate(names) if name in walked_axes)


def _unread(finishes: Sequence[Term | None], merges: Sequence[Term],
            walked_leaves: frozenset[int]) -> str | None:
    """Why a finish or a merge reads a walked operand, which no tile holds there; None where none does."""
    if any(form is not None and _reads_leaves(form) & walked_leaves for form in finishes):
        return "an output reads a walked operand outside every walked reduction"
    if any(_reads_leaves(form) & walked_leaves for form in merges):
        return "a merge reads a walked operand: no tile is there to read"
    return None


_TERMS = ("reductions", "steps", "merges", "anchors", "scales")
"""The forms' fields of terms (`_Forms`), each a tuple of them."""
_PARTS = ("accumulated", "added", "merged")
"""The forms' fields of a reduction's mainloop parts (`_Forms`), each a tuple of a part's terms or None."""


def _encoded(made: _Forms) -> Callable[..., Mapping[str, object]]:
    """`made`'s forms as the store writes them (`stored.stored`), given its codec: each term by `term` (its
    index in the entry's table), a term or None by `maybe`, a value by `value`; its region traced again on
    reading."""
    def encoded(*, term: Callable, maybe: Callable, value: Callable) -> Mapping[str, object]:
        report = made.report
        return {**{field: [term(form) for form in getattr(made, field)] for field in _TERMS},
                "finishes": [maybe(form) for form in made.finishes],
                **{field: [None if part is None else [maybe(form) for form in part]
                           for part in getattr(made, field)] for field in _PARTS},
                "floors": [value(floor) for floor in made.floors],
                "mapped": list(made.mapped),
                "report": [report.equations, list(report.admitted),
                           [list(pair) for pair in sorted(report.fired.items())],
                           list(report.limits), report.iterations, report.nodes, report.limited,
                           value(report.before), value(report.after)]}
    return encoded


def _assembled(record: Mapping, *, decoded: stored._Decoded, value: Callable, region: terms.Region, closed,
               walked_leaves: frozenset[int], began: float) -> _Forms:
    """The forms a stored `record` holds (`stored.loaded`) over `region`, `closed`'s region (its appended
    parameters, literals and terms already decoded beside it): each form by its term's index, reading only
    the states its form reads (`decoded`'s sides: a region's term none, a step's, a step's parts' and a
    finish's the walk's own, a merge's and its factors' either walk's), each count, part's arity and report's
    field the forms state, a finish for each output a tile's walk finishes (none for a literal or a mapped
    one) and no finish or merge reading a walked operand (`_unread`, of `walked_leaves`), as a derivation
    holds them (`ValueError` otherwise); its report the derivation's as stored, reused
    (`AlgebraReport.reused`), its seconds the reading's."""
    count = len(record["reductions"]) + len(record["anchors"]) + 2 * len(record["scales"])
    outputs = len(region.outputs)
    counts = (len(record["steps"]), len(record["merges"]), *(len(record[field]) for field in _PARTS))
    if (counts != (count,) * 5 or (len(record["finishes"]), len(record["mapped"])) != (outputs, outputs)
            or len(record["floors"]) != len(record["reductions"])
            or not all(isinstance(mapped, bool) and not (mapped and output is None)
                       for mapped, output in zip(record["mapped"], region.outputs, strict=True))):
        raise ValueError("an entry whose forms' counts are not its region's")
    equations, admitted, fired, limits, iterations, nodes, limited, before, after = record["report"]
    if not (all(isinstance(held, list) for held in (admitted, fired, limits))
            and all(stored.integral(number) for number in (equations, iterations, nodes, *limits))
            and len(limits) == 2 and isinstance(limited, bool)
            and all(isinstance(name, str) for name in admitted)
            and all(isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str)
                    and stored.integral(pair[1]) for pair in fired)):
        raise ValueError("an entry whose report is not a derivation's")
    floors = tuple(map(value, record["floors"]))
    before, after = value(before), value(after)
    if not all(number is None or isinstance(number, numbers.Real) for number in (*floors, before, after)):
        raise ValueError("an entry whose floors or costs are not numbers")
    report = program.AlgebraReport(equations, tuple(admitted), dict(fired), tuple(limits), iterations, nodes,
                                   limited, before, after, time.perf_counter() - began, reused=True)

    def parts(field: str, sides: tuple[int, ...], arity: int, optional: frozenset[int]) -> tuple:
        # Each part of a reduction's, None or `arity` terms, the ones `optional` places may be None.
        made = []
        for part in record[field]:
            if part is not None and len(part) != arity:
                raise ValueError(f"a {field} part of {len(part)} terms, not {arity}")
            made.append(None if part is None else tuple(
                decoded.maybe(form, sides=sides) if at in optional else decoded(form, sides=sides)
                for at, form in enumerate(part)))
        return tuple(made)

    held = {field: tuple(decoded(form, sides=()) for form in record[field])
            for field in ("reductions", "anchors", "scales")}
    merges = tuple(decoded(form, sides=(0, 1)) for form in record["merges"])
    finishes = tuple(decoded.maybe(form, sides=(0,)) for form in record["finishes"])
    if any((form is None) != (output is None or mapped)
           for form, output, mapped in zip(finishes, region.outputs, record["mapped"], strict=True)):
        raise ValueError("an entry whose finishes are not its outputs'")
    if (unread := _unread(finishes, merges, walked_leaves)) is not None:
        raise ValueError(unread)
    return _Forms(region, held["reductions"], tuple(decoded(form, sides=(0,)) for form in record["steps"]),
                  merges, finishes, tuple(record["mapped"]), tuple(region.outputs), walked_leaves,
                  tuple(closed.consts), report, parts("accumulated", (0,), 3, frozenset({0})),
                  parts("added", (0,), 3, frozenset()), parts("merged", (0, 1), 2, frozenset()),
                  floors, held["anchors"], held["scales"])


SIZED_PARAMETERS = ("shape", "new_sizes", "slice_sizes")
"""The parameters an equation states extents by (a broadcast's, a reshape's, a dynamic slice's): symbols in a
class's key (`_shape_of`), and its canonical sizes' mapped to another size of the class (`_classed`); every
other parameter's value its own."""


@dataclasses.dataclass(frozen=True)
class _Shape:
    """A traced region's structure as a class of its sizes keys it (`_shape_of`): its key, and the extents
    past one it holds in the order they are first read (each a symbol in the key)."""

    key: Hashable
    sizes: tuple[int, ...]


class Diverged(AssertionError):
    """A class's forms bound at sizes of it state tables a derivation afresh there does not (`_checked`): the
    facts its key reads of the sizes (`_shape_of`) do not hold every fact a derivation states of them."""


def _shape_of(closed, region: terms.Region, parts: Hashable) -> _Shape | None:
    """`region`'s structure (`closed`'s, read beside `parts`, held as they are) with each extent past one a
    symbol, numbered as first read, and what the facts a derivation states read of its sizes beside them:
    each symbol's power of two (`rules.log2_of`), the order of their values (a table built in it,
    `program._products`), each reduction's length's power and whether it counts exactly, and each product's
    contracted length's power and whether it is at most 256 (`program.reduced`, `program.contracted`, the
    facts' own); every other value its own.
    Two regions of one key differ in their sizes alone, and a derivation states of each the same facts but
    for its extents (an extent of none or one its own in the key), each rule reading an extent only as one
    (equal to another, ordered, its power): forms derived at one compute the declaration at the other
    (`_classed`). None where the region holds an equation outside the language (whose parameters state
    what the key does not read), a captured constant no host holds (an outer trace's), or a part a key
    holds by identity."""
    if region.opaque or not all(isinstance(const, (np.ndarray, np.generic, bool, int, float))
                                for const in closed.consts):
        return None
    inventory = stored._inventory(region)
    if inventory is None:
        return None
    sizes: dict[int, int] = {}

    def symbol(size) -> int | tuple[str, int]:
        size = int(size)
        return size if size < 2 else ("extent", sizes.setdefault(size, len(sizes)))

    table, outputs, opaque, shaped, params = inventory
    shaped = [[term, [symbol(size) for size in extent]] for term, extent in shaped]
    params = [(name, {field: (tuple(symbol(size) for size in value)
                              if field in SIZED_PARAMETERS and value is not None else value)
                      for field, value in values.items()}) for name, values in params]
    coarse = [(program.reduced(region, term), program.contracted(region, term)) for term, _ in region.shapes]
    held = list(sizes)
    stated = terms.plain_key((
        table, outputs, opaque, shaped, params, coarse, [max(size - 1, 0).bit_length() for size in held],
        sorted(range(len(held)), key=held.__getitem__), region.leaves, region.depth, region.equations,
        [stored.encoded_value(np.asarray(value)) for value in region.literals],
        [stored.encoded_value(np.asarray(value)) for value in region.literal_outputs],
        sorted((repr(key), np.dtype(dtype).name) for key, dtype in region.dtypes.items()),
        sorted((repr(key), repr(held_bound)) for key, held_bound in region.magnitudes.items()),
        sorted((repr(key), repr(apart)) for key, apart in region.floors.items()),
        sorted(map(repr, region.nonnegative)), sorted(map(repr, region.unplaced_full_fp32)),
        [(terms.key(narrow), terms.key(original)) for narrow, original in region.exact_roundtrips],
        sorted((index, repr(product)) for index, product in region.products.items()),
        sorted(region.stock.items()), len(closed.jaxpr.constvars),
        [stored.encoded_value(np.asarray(const)) for const in closed.consts]))
    return None if stated is None else _Shape((stated, parts), tuple(held))


def _canonical(sizes: tuple[int, ...]) -> tuple[int, ...] | None:
    """The canonical sizes of a class (`_shape_of`'s, of `sizes`, its extents past one): each extent the top
    of its power of two's range (`2**p`, `p` its `rules.log2_of`), extents of one power apart below it, the
    largest highest, so their order holds; None where they are more than its range holds. A function of the
    class alone: every size of it derives at the same sizes, whichever the process derives first."""
    powers: dict[int, list[int]] = {}
    for size in sizes:
        powers.setdefault((size - 1).bit_length(), []).append(size)
    canonical: dict[int, int] = {}
    for power, members in powers.items():
        for rank, size in enumerate(sorted(members, reverse=True)):
            if (value := (1 << power) - rank) <= 1 << (power - 1):
                return None
            canonical[size] = value
    return tuple(canonical[size] for size in sizes)


def _resized(values: Mapping, sizes: Mapping[int, int]) -> dict:
    """A parameter's `values` with each extent of its sized fields (`SIZED_PARAMETERS`) its size in `sizes`
    (one past none its own)."""
    return {field: (tuple(int(size) if int(size) < 2 else sizes[int(size)] for size in value)
                    if field in SIZED_PARAMETERS and value is not None else value)
            for field, value in values.items()}


def _classed(tile: _Tile, admitted: tuple[rules.Rules, ...], classes: collections.OrderedDict
             ) -> _Forms | None:
    """The forms at `tile` its class's canonical sizes hold (`_canonical`), bound over its region as a stored
    entry's are (`stored.bound`, each appended parameter's extents mapped to its sizes, `_resized`): the same
    forms for every size of a class however a process came to them, one derivation a class. A class's record
    is held in `classes` by the class's key (its traced structure and parts, `_shape_of`, `context` among
    them: never by the canonical sizes alone, whose trace may read what a shape does not), through
    `_cache.holding`, with the facts its canonical trace read checked where it is taken; a size of a known
    class traces nothing more, an unknown one derives its class (`_class_record`). A rebound size is held by
    the oracle to a derivation afresh there (`_checked`). None where the class has no canonical sizes or
    record; where a set the source does not register is admitted (a set registered at run time states no
    class's contract, `rules.sourced`)."""
    if not rules.sourced(admitted):
        return None
    parts = (context(), tuple(sets.name for sets in admitted))
    shaped = _shape_of(tile.closed, tile.region, parts)
    sizes = None if shaped is None else _canonical(shaped.sizes)
    if shaped is None or sizes is None:
        return None
    # The canonical sizes asked themselves derive in the making, their own forms taken as they are.
    own: list[_Forms] = []
    began = time.perf_counter()
    record = _cache.holding(classes, shaped.key,
                            lambda: _class_record(tile, admitted, parts, shaped, sizes, own), FORMS_HELD)
    if own:
        return own[0]
    if record is None:
        return None
    spent = time.perf_counter() - began
    back = dict(zip(sizes, shaped.sizes, strict=True))
    made = stored.bound(record, tile.region, tile.assembled(spent), rescaled=tile.rescaled,
                        resized=None if sizes == shaped.sizes else functools.partial(_resized, sizes=back))
    if ALGEBRA.value.checked:
        _checked(record, back, tile, admitted)
    return made


def _class_record(tile: _Tile, admitted: tuple[rules.Rules, ...], parts: Hashable, shaped: _Shape,
                  sizes: tuple[int, ...], own: list[_Forms]) -> dict | None:
    """A class's record (`stored.record_of`'s, the one the store read or wrote where it did), its forms
    derived at its canonical `sizes` (`_held`): `tile`'s own where they are its sizes (its forms put in
    `own`, the facts its trace read recorded as this making's), else the declaration traced there (a probe of
    sizes no caller asked for: whatever it raises leaves the class to each size's own derivation), held to
    the class's key and to those sizes, its report recorded as a derivation's and timed apart. None where it
    does not hold or the codec states no record."""
    there = tile
    if sizes == shaped.sizes:
        replay(tile.facts)
    else:
        to = dict(zip(shaped.sizes, sizes, strict=True))
        shapes = {name: (tuple(int(size) if int(size) < 2 else to[int(size)] for size in shape), dtype)
                  for name, (shape, dtype) in tile.shapes.items()}
        began = time.perf_counter()
        try:
            # Within the class's making, whose recording holds what this trace reads.
            closed, region = _traced(tile.declaration, tile.names, shapes)
        except Exception as untraced:
            _log.info("a class's canonical sizes do not trace, derived at its own: %s", untraced)
            return None
        traced_there = _shape_of(closed, region, parts)
        if traced_there is None or traced_there.key != shaped.key or traced_there.sizes != sizes:
            return None
        there = _traced_tile(tile.declaration, tile.names, tile.walked_axes, shapes, closed, region,
                             tile.accurate, tile.saturation, began, {})
    made, record = _held(there, admitted)
    if there is tile:
        own.append(made)
    else:
        kernel_report.record_rewrite(made.report)
    if record is not None:
        return record
    try:
        return stored.record_of(made.region, *there.traced, _encoded(made))
    except stored.Unstated as unstated:
        _log.info("a class's forms not bound past its canonical sizes: %s", unstated)
        return None


def _checked(record: Mapping, back: Mapping[int, int], tile: _Tile,
             admitted: tuple[rules.Rules, ...]) -> None:
    """The oracle's hold of a class's `record` bound at `tile` (`_classed`'s, its extents mapped by `back`)
    to a derivation afresh there: the tables it appends are the ones the class's bound (each fact it states
    of these sizes the one the class's canonical sizes stated, mapped); `Diverged` otherwise."""
    closed, region = _traced(tile.declaration, tile.names, tile.shapes)
    traced = (len(region.params), len(region.literals), len(region.shapes))
    made = _derived_afresh(tile.declaration, tile.names, tile.walked_axes, closed, region, admitted,
                           tile.saturation, tile.began)
    try:
        fresh = stored.record_of(made.region, *traced, _encoded(made))
    except stored.Unstated as unstated:
        _log.info("a class's tables not held to a derivation's: %s", unstated)
        return
    bound = [stored.encoded_parameter(primitive, _resized(values, back))
             for primitive, values in map(stored.parameter, record["params"])]
    if (fresh["params"], fresh["literals"]) != (bound, record["literals"]):
        raise Diverged(f"a class's tables at {tile.shapes} are not a derivation's there")


def _decoupled_through(graph: EGraph, region: terms.Region, facts: Schedule, decoupled: rules.Rules) -> None:
    """The derivation's readings run through `region`'s terms alone first (`DECOUPLED`, its facts saturated
    after each pass): each reading meets the language at the region's depth, before the engine's sets grow
    the graph toward their limits, so a limit the engine reaches never stops a form."""
    for _ in range(4 * max(region.depth, 1) + 8):
        graph.run(facts)
        if not graph.run(decoupled.ruleset).updated:
            break
    graph.run(facts)


def _register_reduction_slots(graph: EGraph, region: terms.Region, reductions: tuple[Term, ...],
                              anchors: tuple[Term, ...], scales: tuple[Term, ...]) -> None:
    """Assign state slots, companion anchor/scale slots, retypes, state shapes, and broadcast layouts."""
    anchored_keys = {terms.key(term) for term in anchors}
    for index, reduction in enumerate(reductions):
        graph.register(set_(slot(reduction)).to(i64(index)))
        if _holds(graph, local_of(reduction)) and terms.key(reduction) not in anchored_keys:
            graph.register(plain_local(reduction))
        region.params.append((jax.lax.convert_element_type_p, {
            "new_dtype": region.dtypes[terms.key(reduction)], "weak_type": False, "sharding": None}))
        graph.register(set_(retype(i64(index))).to(i64(len(region.params) - 1)))
        for side in (0, 1):
            graph.register(set_(rules.shape(Term.state(i64(index), i64(side)))).to(
                Vec[i64](*(i64(size) for size in region.shape(reduction)))))
    for offset, term in enumerate(anchors):
        a_index = len(reductions) + offset
        graph.register(set_(anchor_slot(term)).to(i64(a_index)))
        for side in (0, 1):
            graph.register(set_(rules.shape(Term.state(i64(a_index), i64(side)))).to(
                Vec[i64](*(i64(size) for size in region.shape(term)))))
    for offset, term in enumerate(scales):
        h_index = len(reductions) + len(anchors) + offset
        s_index = h_index + len(scales)
        graph.register(set_(hi_slot(term)).to(i64(h_index)), set_(scale_slot(term)).to(i64(s_index)))
        for slot_index in (h_index, s_index):
            for side in (0, 1):
                graph.register(set_(rules.shape(Term.state(i64(slot_index), i64(side)))).to(
                    Vec[i64](*(i64(size) for size in region.shape(term)))))
    _lift(graph, region, reductions)
    if anchors or scales or any(_holds(graph, summed(reduction)) and not _holds(graph, local_of(reduction))
                                and graph.lookup_function_value(anchored(reduction)) is None
                                for reduction in reductions):
        derive_moments.register_moment_resources(graph, region, reductions, walked)


def _register_param_moves(graph: EGraph, region: terms.Region) -> None:
    """Batch how each parameter in `region.params` moves, inserts, reduces, or contracts axes."""
    facts = []
    for index, (primitive, params) in enumerate(region.params):
        parameter = i64(index)
        declared = terms.OPERATIONS.get(primitive)
        kind = None if declared is None else declared.kind
        if kind in ("binary", "unary", "comparison", "cast"):
            facts.append(keeps(parameter))
            if primitive is jax.lax.integer_pow_p and params.get("y") == 2:
                facts.append(is_square(parameter))
        elif primitive is jax.lax.broadcast_in_dim_p:
            dims = tuple(int(d) for d in params["broadcast_dimensions"])
            after = tuple(int(s) for s in params["shape"])
            facts.extend(lays(parameter, i64(axis), i64(to)) for axis, to in enumerate(dims))
            if dims == tuple(range(len(after))):
                facts.append(identity_broadcast(parameter))
            elif len(dims) == len(after) - 1:
                missing = [axis for axis in range(len(after)) if axis not in dims]
                if len(missing) == 1 and dims == tuple(p for p in range(len(after)) if p != missing[0]):
                    facts.append(inserts_axis(parameter, i64(missing[0])))
        elif primitive is jax.lax.transpose_p:
            permutation = list(params["permutation"])
            facts.extend(lays(parameter, i64(axis), i64(permutation.index(axis)))
                         for axis in range(len(permutation)))
        elif primitive in (jax.lax.squeeze_p, jax.lax.reshape_p):
            facts.extend(lays(parameter, i64(axis), i64(to))
                         for axis, to in _moved(primitive, params, region, index))
            if primitive is jax.lax.reshape_p and params.get("dimensions") is None:
                facts.extend(inserts_axis(parameter, i64(axis))
                             for axis in _inserted_reshape_axes(region, index))
        elif kind == "reduction":
            reduced = {int(axis) for axis in params["axes"]}
            facts.extend(reduces(parameter, i64(axis), i64(axis - sum(r < axis for r in reduced)))
                         for axis in range(_rank_before(region, index, reduced)) if axis not in reduced)
        elif kind == "product":
            (left_contracted, right_contracted), (left_batch, right_batch) = params["dimension_numbers"]
            left_rank, right_rank = _dot_ranks(region, index)
            left_free = [axis for axis in range(left_rank) if axis not in left_contracted and
                         axis not in left_batch]
            right_free = [axis for axis in range(right_rank) if axis not in right_contracted and
                          axis not in right_batch]
            for position, (left, right) in enumerate(zip(left_batch, right_batch, strict=True)):
                facts.append(left_of(parameter, i64(int(left)), i64(position)))
                facts.append(right_of(parameter, i64(int(right)), i64(position)))
            left_at = len(left_batch)
            right_at = left_at + len(left_free)
            facts.extend(left_of(parameter, i64(axis), i64(left_at + position))
                         for position, axis in enumerate(left_free))
            facts.extend(right_of(parameter, i64(axis), i64(right_at + position))
                         for position, axis in enumerate(right_free))
            facts.extend(contracts_left(parameter, i64(int(axis))) for axis in left_contracted)
            facts.extend(contracts_right(parameter, i64(int(axis))) for axis in right_contracted)
    if facts:
        graph.register(*facts)


def _register(graph: EGraph, region: terms.Region, closed, names, walked_axes) -> None:
    """The facts the rules read that each parameter's own values state (the engine's, `program.stated`, and
    the derivation's own), and each input's walked axis."""
    program.stated(graph, region)
    _least(graph, region)
    _register_param_moves(graph, region)
    facts: list = []
    # A scalar anchor's axis (`scales`' -1) is none, through every equation.
    none = i64(-1)
    for index in range(len(region.params)):
        facts.extend((lays(i64(index), none, none), reduces(i64(index), none, none),
                      left_of(i64(index), none, none), right_of(i64(index), none, none)))
    for index in range(len(closed.jaxpr.constvars)):
        leaf = Term.leaf(i64(index))
        facts.extend((local(leaf), unwalked(leaf)))
    first = len(closed.jaxpr.constvars)
    for index, input_name in enumerate(names):
        leaf = Term.leaf(i64(first + index))
        facts.append(local(leaf))
        if input_name in walked_axes:
            facts.append(set_(walked(leaf)).to(i64(walked_axes[input_name])))
        else:
            facts.append(unwalked(leaf))
    for term, _ in region.shapes:
        named = terms.key(term)
        if region.dtypes.get(named) == np.bool_:
            facts.append(boolean(term))
        if named in region.nonnegative:
            facts.append(nonnegative(term))
    for index, value in enumerate(region.literals):
        literal = Term.literal(i64(index))
        facts.extend((local(literal), unwalked(literal)))
        if not terms.typed_key(value):
            literal_array = np.asarray(value)
            if np.all(literal_array == 0):
                facts.append(zero(literal))
            elif np.all(literal_array == 1):
                facts.extend((unit_array(literal), positive_guard(literal)))
            elif (literal_array.dtype != np.bool_ and np.all(np.isfinite(literal_array))
                  and np.all(literal_array > 0)):
                facts.append(positive_guard(literal))
    if facts:
        graph.register(*facts)


def _least(graph: EGraph, region: terms.Region) -> None:
    """Each floating value's dtype's least finite value (`least`), stated of every value a walked maximum
    may anchor (its own, or one made from it: a broadcast, a difference), one literal a dtype: maxima of two
    dtypes are each corrected by their own, where one value for the whole program refused them."""
    floors: dict[np.dtype, int] = {}
    facts = []
    for term, _ in region.shapes:
        dtype = region.dtypes.get(terms.key(term))
        if dtype is None or not jnp.issubdtype(dtype, jnp.floating):
            continue
        if dtype not in floors:
            region.literals.append(np.asarray(jnp.finfo(dtype).min, dtype))
            floors[dtype] = len(region.literals) - 1
            literal = Term.literal(i64(floors[dtype]))
            region.shaped(literal, ())
            facts.append(set_(rules.shape(literal)).to(Vec[i64]()))
        facts.append(set_(least(term)).to(i64(floors[dtype])))
    if facts:
        graph.register(*facts)


def _inserted_reshape_axes(region: terms.Region, index: int) -> list[int]:
    """Every axis `a` that a `reshape` of parameters `index` inserts as size 1 across all terms using `index`,
    keeping every other axis in order."""
    pairs = []
    for term, after in region.shapes:
        if (args := terms.args_of(term, Term.shaped)) is not None and terms.integer(args[2]) == index:
            pairs.append((region.shape(args[0]), tuple(after)))
    if not pairs:
        return []
    after = pairs[0][1]
    if any(len(after) != len(before) + 1 or end != after for before, end in pairs):
        return []
    return [axis for axis in range(len(after))
            if after[axis] == 1 and all(before == (*after[:axis], *after[axis + 1:]) for before, _ in pairs)]


def _moved(primitive, params, region: terms.Region, index: int) -> list[tuple[int, int]]:
    """A squeeze's or a reshape's moves of its operand's axes, where each axis of extent
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
        args = terms.args_of(term, Term.shaped) or terms.args_of(term, Term.reduce)
        if args is not None and terms.integer(args[-1]) == index:
            return region.extents.get(terms.key(args[0])), extent
    return None, None


def _rank_before(region: terms.Region, index: int, reduced: set[int]) -> int:
    before, after = _operand_and_result_shapes(region, index)
    return len(before) if before is not None else (len(after) + len(reduced) if after is not None else 0)


def _dot_ranks(region: terms.Region, index: int) -> tuple[int, int]:
    for term, _ in region.shapes:
        args = terms.args_of(term, Term.dot)
        if args is not None and terms.integer(args[-1]) == index:
            return len(region.shape(args[0])), len(region.shape(args[1]))
    return 0, 0


def _register_local_opaques(graph: EGraph, region: terms.Region, walking: Ruleset) -> None:
    """The outputs of equations outside the language (`terms.Opaque`), each over local, unwalked inputs alone:
    local themselves, in program order, so an equation over an earlier one's outputs reads them as local. An
    equation outside the language over a value the walk reads is refused: over a walked value its axis is
    unknown, and over one a walked reduction gave (a total's `log2`) no form computes it from the walk's
    state, where evaluating it from the operands would read them whole."""
    for opaque in region.opaque:
        inputs = [term for term in opaque.inputs if term is not None]
        name = opaque.equation.primitive.name
        if any(_walked(graph, term) for term in inputs):
            raise Refused(f"{name}: an equation outside the language over a walked value")
        if not all(_holds(graph, local(term)) and _holds(graph, unwalked(term)) for term in inputs):
            raise Refused(f"{name}: an equation outside the language over a value a walked reduction gave")
        facts = [fact(Term.leaf(i64(leaf))) for leaf in opaque.outputs for fact in (local, unwalked)]
        if facts:
            graph.register(*facts)
            graph.run(walking.saturate())


def _lift(graph: EGraph, region: terms.Region, reductions) -> None:
    """Each broadcast a rescaled sum may need (`lifts`), its parameters added to the region's table: a scalar
    anchor's to the sum's shape along no axis, a vector's along each of the sum's axes of its extent, of which
    the rules take the one its anchor's lies along (`anchored`). An anchor of more axes has none: its sum's
    correction is no broadcast of these, and the declaration is refused."""
    # In the region's order, which no size changes (a set's of their extents a hash's).
    anchors = dict.fromkeys(extent for term, extent in region.shapes if _holds(graph, anchor(term)))
    lifted = set()
    facts = []
    for reduction in reductions:
        to = region.extents.get(terms.key(reduction))
        if to is None:
            continue
        for given in anchors:
            if len(given) > 1:
                continue
            for at in ((-1,) if not given else tuple(axis for axis, extent in enumerate(to)
                                                      if extent == given[0])):
                if (given, tuple(to), at) in lifted:
                    continue
                lifted.add((given, tuple(to), at))
                dimensions = () if at < 0 else (at,)
                params = {"shape": tuple(to), "broadcast_dimensions": dimensions, "sharding": None}
                region.params.append((jax.lax.broadcast_in_dim_p, params))
                facts.append(set_(lifts(Vec[i64](*(i64(size) for size in given)),
                                        Vec[i64](*(i64(size) for size in to)), i64(at))).to(
                    i64(len(region.params) - 1)))
    if facts:
        graph.register(*facts)


def _walked(graph: EGraph, term: Term) -> bool:
    """Whether a value has a walked axis (`walked`)."""
    return graph.lookup_function_value(walked(term)) is not None


def _holds(graph: EGraph, fact) -> bool:
    try:
        graph.check(fact)
    except EggSmolError:
        return False
    return True


def _floors(graph: EGraph, region: terms.Region, reductions: Sequence[Term]) -> tuple[float | None, ...]:
    """Each walked maximum's floor where the declaration reads it only floored, `max(m, c)` at one finite
    literal value `c` (attention's `max(m, least)`), else None: the walk carries it floored from `c` (its
    identity, `init`), so each of its states is at least `c` (`at_least`) and a floor of one is the state
    itself (`FLOORED`), the per-step floor and the correction's shift dropped (row 3520: the hand walk
    carries its maximum floored from the least finite value)."""
    places = {terms.key(reduction): index for index, reduction in enumerate(reductions)
              if _role(reduction) == "max"}
    floors: dict[int, float] = {}
    barred = {places[terms.key(term)] for term in region.outputs
              if term is not None and terms.key(term) in places}

    def literal_of(term: Term) -> float | None:
        if (args := terms.args_of(term, Term.literal)) is None:
            return None
        value = np.asarray(region.literals[terms.integer(args[0])])
        return value.item() if value.shape == () and np.isfinite(value) else None

    for term, _ in region.shapes:
        if (args := terms.args_of(term, terms.LANGUAGE_REFS)) is None:
            continue
        for child in (arg for arg in args if isinstance(arg, Term)):
            if (index := places.get(terms.key(child))) is None:
                continue
            binary = terms.args_of(term, Term.binary)
            floor = None
            if binary is not None and terms.text(binary[1]) == "max":
                other = binary[2] if terms.key(binary[0]) == terms.key(child) else binary[0]
                floor = literal_of(other)
            if floor is None or floors.setdefault(index, floor) != floor:
                barred.add(index)
    held = {index: floor for index, floor in floors.items() if index not in barred}
    classes = {floor: number for number, floor in enumerate(dict.fromkeys(held.values()))}
    for literal, value in enumerate(region.literals):
        value = np.asarray(value)
        if value.shape == () and value.item() in classes:
            graph.register(set_(floor_of(Term.literal(i64(literal)))).to(i64(classes[value.item()])))
    for index, floor in held.items():
        for side in (0, 1):
            graph.register(set_(at_least(Term.state(i64(index), i64(side)))).to(i64(classes[floor])))
    return tuple(held.get(index) for index in range(len(reductions)))


def _identity(reduction: Term, dtype) -> float:
    """A walked reduction's empty value, its monoid's identity: a sum's zero; a maximum's -inf (its dtype's
    least value where it has no infinity). A correction never reads it unshifted (`least`)."""
    if _role(reduction) != "max":
        return 0.0
    return -math.inf if jnp.issubdtype(dtype, jnp.floating) else float(jnp.iinfo(dtype).min)


def _reads_leaves(form: Term) -> set[int]:
    """The region's leaves `form` reads."""
    return {terms.integer(args[0]) for term in terms.subterms(form)
            if (args := terms.args_of(term, Term.leaf)) is not None}


def _written(form: Term) -> bool:
    """Whether `form` keeps none of the derivation's readings: a form in the language."""
    return not any(terms.made_by(term) in _READINGS for term in terms.subterms(form))
