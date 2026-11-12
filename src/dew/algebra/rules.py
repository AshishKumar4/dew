"""The rules Dew's rewrite applies (docs/design/algebra.md §6.4), in named sets, and the facts they read
(§6.3): each set an egglog ruleset with its rounding class, registered by name.

`shape(t)` is each value's shape, as the equation that made it states it (`terms.region`), carried to every
form the rules make: a rule joins or moves values only where their shapes say the result is the program's.
`along(params, axis)` holds where a broadcast lays its operand along every axis of its result but `axis`: a
value it makes is constant along `axis`, so a sum along that axis may take it out."""

from __future__ import annotations

import collections
import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from egglog import (
    Ruleset,
    String,
    StringLike,
    Unit,
    Vec,
    birewrite,
    declarations,
    eq,
    function,
    i64,
    i64Like,
    rewrite,
    rule,
    ruleset,
    set_,
    union,
    var,
    vars_,
)

from dew.algebra import terms
from dew.algebra.terms import Term


@function(merge=lambda old, new: old)
def shape(term: Term) -> Vec[i64]: ...


@function
def along(params: i64, axis: i64) -> Unit: ...


@function
def matrix_product(params: i64) -> Unit: ...
"""A product's parameters (`Term.dot`) contracting its left operand's last axis with its right's first, no
batch, in its operands' dtype (`plain`): `x @ y` of matrices (`matrix`)."""


@function
def matrix(term: Term) -> Unit: ...
"""A value of rank two (`program.stated`): each operand a rule moves a matrix product past is one, as a
product's parameters hold whatever its operands' ranks."""


@function(merge=lambda old, new: old)
def lifted(extent: Vec[i64], axis: i64Like) -> i64: ...
"""The parameters of a broadcast of a vector to a matrix of `extent`, the vector along `axis`
(`program.stated` states each the region's matrices take)."""


@function
def plain(params: i64) -> Unit: ...
"""Holds where an arithmetic equation's parameters are its defaults (no `out_dtype`): it computes in its
operands' dtype, so a rule may move it past the arithmetic beside it (`program.rewrite` states it)."""


@function
def stock(name: StringLike) -> i64: ...
"""The index of a primitive a stable rule writes at its defaults (`terms.Region.stock`), by its name."""


@function
def unit(term: Term) -> Unit: ...
"""Holds where a term is the scalar literal one, which `log(1 + x)` and `exp(x) - 1` read."""


@function
def positive_scale(term: Term) -> Unit: ...
"""Holds where a term is a scalar literal past zero: a scale by it keeps every comparison, so a maximum or a
minimum of a scaled value is the scaled maximum or minimum (`_linear`)."""


@function
def exact_scale(term: Term) -> Unit: ...
"""Holds where a term is a scalar literal of a power of two: a multiply by it rounds nothing (but past the
dtype's range), so moving it through a contraction leaves every rounding as it was (`program.rewrite`
states it). A scale of any other value moved so reorders rounding, which the gate refused on the gated delta
rule's adjoint (1.575 times the reference's error, past 1.5): it waits for error-aware extraction (§1.5)."""


@function(merge=lambda old, new: old)
def log2_bound(term: Term) -> i64: ...
"""A value's magnitude's power of two (`terms.Magnitude`), a number bound: the region's (`program.stated`), or
what a reordering rule states of the intermediate it makes (`made`). A value no number bounds has none."""


@function(merge=lambda old, new: old)
def reach() -> i64: ...
"""The largest power of two a value of the program may take finite, the narrowest float's (`program.stated`):
a reordering's new intermediate under it is finite wherever the program is, as a bound of numbers alone holds
the program's values (§6.3, `dominated`)."""


@function(merge=lambda old, new: old)
def log2_of(size: i64Like) -> i64: ...
"""`ceil(log2(size))` of each extent the region's shapes hold: a contraction's or a sum's length's power."""


@function(merge=lambda old, new: old)
def contracted_length(product: Term) -> i64: ...
"""A contraction's (`Term.dot`'s, by its term) length's power of two, `ceil(log2(...))` of the elements each
of its outputs sums: a rule that makes a contraction bounds its products' and its sum's magnitude by its
operands' and this (`product_magnitude`), read off the product it replaces, which contracts the same
elements. Stated per product, not per parameters' index, which two products of different lengths share
(row 3577). A contraction of no stated length (an operand of no known shape) has none, so a rule that asks it
does not fire."""


@function(merge=lambda old, new: old)
def sum_length(term: Term) -> i64: ...
"""A reduction's length's power of two, `ceil(log2(...))` of the elements it sums into each of its own
(`program.stated`): a reordering that sums anew over the same axes of a value of the same shape bounds its
new sum by it."""


def magnitude(term: Term, name: str) -> tuple[list, i64]:
    """The fact naming `term`'s magnitude (`log2_bound`) and its power's variable."""
    power = var(f"{name}_log2", i64)
    return [eq(power).to(log2_bound(term))], power


def dominated(power, name: str) -> list:
    """The facts that hold where a magnitude of `power` is under the program's reach (`reach`): the bible's
    §6 rule, no infinity where the program is finite, as a reordering rule reads it."""
    reached = var(f"{name}_reach", i64)
    return [eq(reached).to(reach()), power <= reached]


def made(term: Term, power) -> list:
    """The action stating a new intermediate's magnitude, for later rules to read."""
    return [set_(log2_bound(term)).to(power)]


def summed_magnitude(left: tuple, right: tuple) -> tuple[list, i64]:
    """A sum's magnitude from its terms' (`magnitude`'s): twice the larger power."""
    (facts, power), (more, other) = left, right
    return facts + more, power.max(other) + 1


def product_magnitude(left: tuple, right: tuple, length) -> tuple[list, i64]:
    """A product's magnitude from its factors' (`magnitude`'s), its contraction's length's power `length`
    (0 for an elementwise product): the powers added."""
    (facts, power), (more, other) = left, right
    return facts + more, power + other + length


def shared_factored(t: Term, x: Term, y: Term, shared: Term, params, summed, *, combined: str, side: str,
                    length: tuple[list, i64]) -> tuple[list, list]:
    """Two products of one shared operand summed, as the summed operands' one product with it: `x @ shared
    (+|-) y @ shared` as `(x (+|-) y) @ shared` (the shared operand on the `right`), or `shared @ x (+|-)
    shared @ y` as `shared @ (x (+|-) y)` (on the `left`); `combined` the sum's operation. Its facts and its
    actions, the one declaration of the identity and of what makes it safe, which each rule set that factors
    takes and adds its own guards to (the linear set's plain products, the carried set's matrices that read
    the state): the summands of one shape, and the new sum and its contraction with the shared operand each
    under the program's reach (`dominated`), the contraction of `length`'s power (its facts and its
    variable: what each of the two products it replaces contracts). The new sum states its magnitude
    (`made`)."""
    def product(operand: Term) -> Term:
        return operand.dot(shared, params) if side == "right" else shared.dot(operand, params)

    facts, power = summed_magnitude(magnitude(x, "x"), magnitude(y, "y"))
    new = x.binary(combined, y, summed)
    bounds, contracted = product_magnitude(([], power), magnitude(shared, "shared"), length[1])
    return ([eq(t).to(product(x).binary(combined, product(y), summed)), eq(shape(x)).to(shape(y)), *facts,
             *dominated(power, "new"), *length[0], *bounds, *dominated(contracted, "factored")],
            [union(t).with_(product(new)), *made(new, power)])


type Draw = Callable[[int], tuple[jax.Array, ...]]
"""A rewrite's operands by seed."""


@dataclasses.dataclass(frozen=True)
class Drawn:
    """Draws of a rewrite's operands of `shapes` by seed: each element's magnitude across six decades, a large
    and a small term together in every row, the same family each seed; `scale`'s draw instead where a rule's
    operands lie in a domain of their own, each operand drawn after those before it, which it is given (a
    fold's maximum and total are its logits'). Its adversarial families (`adversarial`) are the same draws
    with the operands of its first's shape, its values, changed; an operand of another shape (a scale, a
    maximum, a divisor) keeps its domain's draw."""

    shapes: tuple[tuple[int, ...], ...]
    scale: Callable[[np.random.Generator, tuple[int, ...], list[np.ndarray]], np.ndarray] | None = None

    def _values(self, seed: int) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        drawn: list[np.ndarray] = []
        for shape in self.shapes:
            drawn.append(np.asarray(rng.standard_normal(shape) * 10.0 ** rng.uniform(-3, 3, shape)
                                    if self.scale is None else self.scale(rng, shape, drawn), np.float32))
        return drawn

    def __call__(self, seed: int) -> tuple[jax.Array, ...]:
        return tuple(jnp.asarray(value) for value in self._values(seed))

    def adversarial(self) -> Mapping[str, Draw]:
        """Its adversarial families (docs/design/algebra.md §3), by name: every other row of its values an
        exact zero; one element a row, the rest exact zeros; and, where it draws two values or more, each
        later one the first's negation (their sum cancels exactly) and the first's next float (near-equal
        operands)."""
        values = [index for index, shape in enumerate(self.shapes) if shape == self.shapes[0]]

        def changed(change: Callable[[int, np.ndarray, np.ndarray], np.ndarray]) -> Draw:
            def drawn(seed: int) -> tuple[jax.Array, ...]:
                operands = self._values(seed)
                first = operands[values[0]]
                for place, index in enumerate(values):
                    operands[index] = change(place, operands[index], first)
                return tuple(jnp.asarray(value) for value in operands)
            return drawn

        def zero_rows(place, value, first):
            return np.where((np.arange(len(value)) % 2 == 0)[:, None], np.float32(0), value)

        def one_a_row(place, value, first):
            kept = np.arange(value.shape[-1])[None, :] == (np.arange(len(value)) % value.shape[-1])[:, None]
            return np.where(kept, value, np.float32(0))

        families = {"rows of zero": changed(zero_rows), "one element a row": changed(one_a_row)}
        if len(values) > 1:
            families["cancelling operands"] = changed(lambda place, value, first: -first if place else value)
            families["near-equal operands"] = changed(
                lambda place, value, first: np.nextafter(first, np.float32(np.inf)) if place else value)
        return families


Instance = tuple[Callable, Callable, Drawn]
"""A rewrite written as JAX: its left side, its right side, and draws of its operands by seed."""


@dataclasses.dataclass(frozen=True)
class RuleFamily:
    """One rewrite of a set: its rules (egglog's, each direction and layout of it) and its `instance`, the
    rewrite written as JAX, which its accuracy test holds the right side to the left side by."""

    name: str
    rules: tuple
    instance: Instance


@dataclasses.dataclass(frozen=True)
class Rules:
    """A named set of rules: its rounding class (`exact`: each rewrite rounds as the form it replaces;
    `reorders`: it reorders rounding, admitted under default numerics alone), its egglog ruleset, the
    measurement it ships with, and the rewrites its rules are declared as (`families`), each with the instance
    that qualifies it (`declared`)."""

    name: str
    rounding: Literal["exact", "reorders", "bounded"]
    ruleset: Ruleset
    receipt: str
    families: tuple[RuleFamily, ...] = ()
    """Its rewrites, every rule of its ruleset one of theirs where it is `declared`: a registered set's
    (`register`); a derivation's own set (`derive.DECOUPLED`, `carry.CARRIED`), whose rules read its readings
    and never run among the registry's, is qualified by its derivation's tests."""
    dev: bool = False
    """Whether it runs at dev time alone, where each candidate's error is measured before it is taken
    (`dew.algebra.error.choose`): a set whose forms are chosen for their accuracy, at a cost the program pass
    would never pay."""
    aid: bool = False
    """Whether the set only aids the others' saturation and changes no program on its own (commutativity,
    whose operand orders XLA's HloCSE reads alike): a pass whose admitted sets are all aids does not run
    (`program.rewritten`). Every other set's rewrite applies wherever the cost model prices it cheaper, with
    no condition on XLA (docs/design/algebra.md §6.5)."""

    @classmethod
    def declared(cls, name: str, rounding: Literal["exact", "reorders", "bounded"],
                 families: tuple[RuleFamily, ...], receipt: str, *, dev: bool = False,
                 aid: bool = False) -> Rules:
        """A set whose ruleset is its `families`' rules, every one of them declared with its instance."""
        made = ruleset(*(each for family in families for each in family.rules), name=name)
        return cls(name, rounding, made, receipt, families, dev=dev, aid=aid)

    @property
    def instances(self) -> Mapping[str, Instance]:
        """Each of its rewrites written as JAX, by its name (`RuleFamily.instance`)."""
        return {family.name: family.instance for family in self.families}


def _facts() -> Ruleset:
    """The shape of each value a rule makes: an elementwise term's over operands of one shape theirs, or over
    an operand and a scalar the operand's, a layout-keeping one its operand's, a sum along one axis its
    operand's without that axis, a matrix's transpose its axes swapped, a matrix product its left operand's
    rows by its right's columns; and each value of two axes a matrix (`matrix`)."""
    term, x, y = vars_("term x y", Term)
    shaped, axes, left, right = vars_("shaped axes left right", Vec[i64])
    name = var("name", String)
    params = var("params", i64)
    scalar = Vec[i64].empty()
    return ruleset(
        rule(eq(term).to(x.binary(name, y, params)), eq(shaped).to(shape(x)), eq(shape(y)).to(shaped)).then(
            set_(shape(term)).to(shaped)),
        rule(eq(term).to(x.binary(name, y, params)), eq(shaped).to(shape(x)), eq(shape(y)).to(scalar)).then(
            set_(shape(term)).to(shaped)),
        rule(eq(term).to(x.binary(name, y, params)), eq(shape(x)).to(scalar), eq(shaped).to(shape(y))).then(
            set_(shape(term)).to(shaped)),
        rule(eq(term).to(x.unary(name, params)), eq(shaped).to(shape(x))).then(set_(shape(term)).to(shaped)),
        rule(eq(term).to(x.convert(params)), eq(shaped).to(shape(x))).then(set_(shape(term)).to(shaped)),
        rule(eq(term).to(x.reduce(name, axes, params)), eq(shaped).to(shape(x)), eq(axes.length()).to(i64(1))
             ).then(set_(shape(term)).to(shaped.remove(axes[i64(0)]))),
        rule(eq(term).to(x.shaped(String("transpose"), params)), eq(left).to(shape(x)),
             eq(left.length()).to(i64(2))).then(set_(shape(term)).to(Vec[i64](left[i64(1)], left[i64(0)]))),
        rule(eq(term).to(x.dot(y, params)), matrix_product(params), eq(left).to(shape(x)),
             eq(right).to(shape(y)), eq(left.length()).to(i64(2)), eq(right.length()).to(i64(2))).then(
            set_(shape(term)).to(Vec[i64](left[i64(0)], right[i64(1)]))),
        rule(eq(shaped).to(shape(term)), eq(shaped.length()).to(i64(2))).then(matrix(term)),
        name="facts")


_ROWS, _SCALED = Drawn(((64, 1024), (64, 1024), (64, 1024))), Drawn(((64, 1024), (64,)))
_SMALL = Drawn(((64, 1024),), scale=lambda rng, shape, drawn: (rng.uniform(-0.5, 0.5, shape)
                                                               * 10.0 ** rng.uniform(-6, 0, shape)))


def _fold(rng: np.random.Generator, shape: tuple[int, ...], drawn: list[np.ndarray]) -> np.ndarray:
    """A fold's operands in turn: its logits, their maximum along the row, and the total of their
    exponentials less it, which holds the maximum's unit weight and so is at least one."""
    if not drawn:
        return rng.standard_normal(shape) * 4.0
    peak = drawn[0].max(-1, keepdims=True)
    return peak if len(drawn) == 1 else np.exp(drawn[0] - peak).sum(-1, keepdims=True)


_FOLDED = Drawn(((64, 1024), (64, 1), (64, 1)), scale=_fold)
"""A fold's logits with their own maximum and total (`_fold`), a softmax's. The unfolded quotient is the
bounded form where the total is at least one: under a total below one `exp(x - m)` falls under float32's
normal range where the folded exponent does not (at `x - m = -95` and `t = exp(-10)` the fold gives 1.2e-37
and the quotient a flushed zero), which a declaration of such totals measures at dev time (§1.5) and
refuses."""
_POSITIVE = Drawn(((64, 1024), ()), scale=lambda rng, shape, drawn: (
    rng.standard_normal(shape) * 10.0 ** rng.uniform(-3, 3, shape) if shape
    else np.exp(rng.standard_normal(shape))))
_ROW_SCALED = Drawn(((64, 1024), (64,), (64, 1024)))


def _stable() -> tuple[RuleFamily, ...]:
    """Stable forms (docs/design/algebra.md §1.5): `exp(x - (m + log t))` as `exp(x - m) / t`, so the
    logarithm of a sum is never rounded into a value as large as the logits (a derived attention backward's
    mapped dK and dQ came to 1.6-1.7 times the reference's error without it), bounded where the total is at
    least one (`_FOLDED`); `log(1 + x)` as `log1p(x)`
    and `exp(x) - 1` as `expm1(x)`, which keep a small `x`'s digits. Each costs what it costs: the program
    pass never runs them, and dev time takes a form of theirs only where its measured error asks it."""
    x, m, t, one = vars_("x m t one", Term)
    subbed, added, logged, exped = vars_("subbed added logged exped", i64)
    # The stock primitives' indices are read in a rule's conditions, where egglog allows a lookup.
    divided, plus_one, minus_one = vars_("divided plus_one minus_one", i64)
    return (
        RuleFamily("a sum's logarithm out of the exponent", (
            rewrite(x.binary("sub", m.binary("add", t.unary("log", logged), added), subbed)
                    .unary("exp", exped)).to(
                x.binary("sub", m, subbed).unary("exp", exped).binary("div", t, divided),
                eq(shape(m)).to(shape(x)), eq(shape(t)).to(shape(x)), plain(subbed), plain(added),
                eq(divided).to(stock("div"))),),
            (lambda x, m, t: jnp.exp(x - (m + jnp.log(t))), lambda x, m, t: jnp.exp(x - m) / t, _FOLDED)),
        RuleFamily("the logarithm of one more", (
            rewrite(one.binary("add", x, added).unary("log", logged)).to(
                x.unary("log1p", plus_one), unit(one), plain(added), eq(plus_one).to(stock("log1p"))),
            rewrite(x.binary("add", one, added).unary("log", logged)).to(
                x.unary("log1p", plus_one), unit(one), plain(added), eq(plus_one).to(stock("log1p")))),
            (lambda x: jnp.log(1 + x), jnp.log1p, _SMALL)),
        RuleFamily("one less than the exponential", (
            rewrite(x.unary("exp", exped).binary("sub", one, subbed)).to(
                x.unary("expm1", minus_one), unit(one), plain(subbed), eq(minus_one).to(stock("expm1"))),),
            (lambda x: jnp.exp(x) - 1, jnp.expm1, _SMALL)))


def _linear() -> tuple[RuleFamily, ...]:
    """Distributivity over operands of one shape (the sums' linearity is `_joined`'s), a positive scalar's
    scale taken out of a maximum or a minimum it keeps the order of (`positive_scale`: `max(c x) = c max(x)`,
    each element's one rounding of `c x` kept as the reduced one's, so it reorders no rounding and makes no
    new magnitude), a product's bilinearity
    (two products sharing an operand, one product of the summed others, and a power of two's scale of the
    product on either operand instead), each over arithmetic in its operands' dtype (`plain`), and a value a
    broadcast lays along every axis but a sum's taken out of the sum where the other factor has the product's
    shape (the broadcast's extent along the axis the product's or 1, as an elementwise product broadcasts a
    unit axis), and a matrix product of an operand scaled along its kept axis (a dequant's row or column
    scale) as the product scaled: each reorders rounding, and each fires only where every intermediate it
    makes is under a magnitude the program as written reaches (`dominated`, §6.3), each stating its own
    (`made`)."""
    x, y, a, t = vars_("x y a t", Term)
    axes, extent = vars_("axes extent", Vec[i64])
    params, laid, axis, added, multiplied, summed = vars_("params laid axis added multiplied summed", i64)
    scalar = Vec[i64].empty()
    families = []
    # A contraction scaled by a power of two: the scale on either operand, the product linear in each. The
    # new operand and the new contraction over it each under the program's reach: a scale moved into a
    # contraction's operand scales each of its products before its sum (2**120 x times 2 summed past 2**128).
    for side, scaled, other, made_of, moved in (
            ("left", x, y, lambda s: s.dot(y, params), lambda x, y, a: jnp.dot(x * 0.5, y.T)),
            ("right", y, x, lambda s: x.dot(s, params), lambda x, y, a: jnp.dot(x, (y * 0.5).T))):
        facts, power = product_magnitude(magnitude(scaled, "s"), magnitude(a, "a"), 0)
        new = scaled.binary("mul", a, multiplied)
        bounds, contracted = product_magnitude(([], power), magnitude(other, "other"),
                                               var("replaced_length", i64))
        families.append(RuleFamily(f"a power of two's scale into a contraction's {side} operand", (rule(
            eq(t).to(x.dot(y, params).binary("mul", a, multiplied)), eq(shape(a)).to(scalar), exact_scale(a),
            plain(params), plain(multiplied), *facts, *dominated(power, "new"),
            eq(var("replaced_length", i64)).to(contracted_length(x.dot(y, params))), *bounds,
            *dominated(contracted, "replaced")).then(union(t).with_(made_of(new)), *made(new, power)),),
            (lambda x, y, a: jnp.dot(x, y.T) * 0.5, moved, _ROWS)))
    facts, power = summed_magnitude(magnitude(x, "x"), magnitude(y, "y"))
    new = x.binary("add", y, added)
    families += [
        RuleFamily("distributivity", (rule(
            eq(t).to(a.binary("mul", x, multiplied).binary("add", a.binary("mul", y, multiplied), added)),
            eq(shape(x)).to(shape(y)), plain(multiplied), plain(added), *facts,
            *dominated(power, "new")).then(
            union(t).with_(a.binary("mul", new, multiplied)), *made(new, power)),),
            (lambda x, y, a: a * x + a * y, lambda x, y, a: a * (x + y), _ROWS)),
    ]
    # A shared operand factored out of two contractions (`shared_factored`), each product in its operands'
    # dtype; the factored contraction contracts what either of the two it replaces does.
    shared_length = var("shared_length", i64)
    for side, replaced, written in (
            ("left", a.dot(x, multiplied),
             (lambda x, y, a: jnp.dot(a, x.T) + jnp.dot(a, y.T), lambda x, y, a: jnp.dot(a, (x + y).T))),
            ("right", x.dot(a, multiplied),
             (lambda x, y, a: jnp.dot(x, a.T) + jnp.dot(y, a.T), lambda x, y, a: jnp.dot(x + y, a.T)))):
        conditions, actions = shared_factored(
            t, x, y, a, multiplied, added, combined="add", side=side,
            length=([eq(shared_length).to(contracted_length(replaced))], shared_length))
        families.append(RuleFamily(f"a shared {side} operand out of two products", (
            rule(*conditions, plain(multiplied), plain(added)).then(*actions),), (*written, _ROWS)))
    facts, power = magnitude(x, "x")
    length = var("length", i64)
    sum_of = x.reduce("sum", axes, summed)
    families.append(RuleFamily("a row scalar out of a sum", (rule(
        eq(t).to(a.shaped("broadcast_in_dim", laid).binary("mul", x, multiplied).reduce("sum", axes, summed)),
        along(laid, axis), plain(multiplied), eq(axes).to(Vec[i64](axis)), eq(shape(x)).to(extent),
        eq(shape(a.shaped("broadcast_in_dim", laid).binary("mul", x, multiplied))).to(extent),
        eq(length).to(log2_of(extent[axis])), *facts, *dominated(power + length, "new")).then(
        union(t).with_(a.binary("mul", sum_of, multiplied)), *made(sum_of, power + length)),),
        (lambda x, s: jnp.sum(s[:, None] * x, -1), lambda x, s: s * jnp.sum(x, -1), _SCALED)))
    return (*families, *_extrema(), *_scale_lifts())


def _extrema() -> tuple[RuleFamily, ...]:
    """A positive scalar's scale out of a maximum and a minimum (`_linear`'s)."""
    x, a, t = vars_("x a t", Term)
    axes = var("axes", Vec[i64])
    laid, multiplied, summed = vars_("laid multiplied summed", i64)
    scalar = Vec[i64].empty()
    families = []
    # A positive scale out of a maximum or a minimum: the scaled elements' order is the elements', so the
    # reduced element is one, its one rounding of `c x` the result's; a scale broadcast to the operand's
    # shape (a scalar's) leaves the reduced value's as a scalar's.
    for reduced, extremum, named in (("max", jnp.max, "maximum"), ("min", jnp.min, "minimum")):
        out_of = []
        for scaled in (lambda c: x.binary("mul", c, multiplied), lambda c: c.binary("mul", x, multiplied)):
            out_of += [
                rule(eq(t).to(scaled(a).reduce(reduced, axes, summed)), positive_scale(a), plain(multiplied),
                     eq(shape(a)).to(scalar)).then(
                    union(t).with_(x.reduce(reduced, axes, summed).binary("mul", a, multiplied))),
                rule(eq(t).to(scaled(a.shaped("broadcast_in_dim", laid)).reduce(reduced, axes, summed)),
                     positive_scale(a), plain(multiplied), eq(shape(a)).to(scalar),
                     eq(shape(x)).to(shape(a.shaped("broadcast_in_dim", laid)))).then(
                    union(t).with_(x.reduce(reduced, axes, summed).binary("mul", a, multiplied))),
            ]
        families.append(RuleFamily(f"a positive scale out of a {named}", tuple(out_of), (
            lambda x, c, extremum=extremum: extremum(x * c, -1),
            lambda x, c, extremum=extremum: extremum(x, -1) * c, _POSITIVE)))
    return tuple(families)


def _scale_lifts() -> tuple[RuleFamily, ...]:
    """A matrix product of an operand scaled along its kept axis as the product scaled (`_linear`'s)."""
    x, y, t = vars_("x y t", Term)
    params, laid, multiplied = vars_("params laid multiplied", i64)
    # A matrix product of an operand scaled along its kept axis (a row scale on the left, a column scale on
    # the right, constant along the contraction) as the product scaled: x@w new.
    s = var("s", Term)
    product_extent = var("product_extent", Vec[i64])
    length2, row_lift, column_lift = vars_("length2 row_lift column_lift", i64)
    left = var("left", Vec[i64])
    contracted = (eq(left).to(shape(x)), eq(length2).to(log2_of(left[i64(1)])))
    facts, power = product_magnitude(magnitude(x, "x"), magnitude(y, "y"), length2)
    product = x.dot(y, params)
    lifted_by: dict[str, list] = {"row": [], "column": []}
    for scaled, laid_axis, lift_axis, lift in (
            (lambda: x.binary("mul", s.shaped("broadcast_in_dim", laid), multiplied).dot(y, params), 1, 0,
             row_lift),
            (lambda: s.shaped("broadcast_in_dim", laid).binary("mul", x, multiplied).dot(y, params), 1, 0,
             row_lift),
            (lambda: x.dot(y.binary("mul", s.shaped("broadcast_in_dim", laid), multiplied), params), 0, 1,
             column_lift),
            (lambda: x.dot(s.shaped("broadcast_in_dim", laid).binary("mul", y, multiplied), params), 0, 1,
             column_lift)):
        lifted_by["row" if lift is row_lift else "column"].append(rule(
            eq(t).to(scaled()), matrix_product(params), plain(multiplied), matrix(x), matrix(y),
            along(laid, i64(laid_axis)), eq(product_extent).to(shape(t)),
            eq(lift).to(lifted(product_extent, i64(lift_axis))),
            *contracted, *facts, *dominated(power, "new")).then(
            union(t).with_(s.shaped("broadcast_in_dim", lift).binary("mul", product, multiplied)),
            # The lifted broadcast lays the scale over the product's extent: its shape, so it is priced.
            set_(shape(s.shaped("broadcast_in_dim", lift))).to(product_extent),
            *made(product, power)))
    return (
        RuleFamily("a row scale out of a product", tuple(lifted_by["row"]),
               (lambda x, s, y: jnp.dot(x * s[:, None], y.T), lambda x, s, y: jnp.dot(x, y.T) * s[:, None],
                _ROW_SCALED)),
        RuleFamily("a column scale out of a product", tuple(lifted_by["column"]),
               (lambda x, s, y: jnp.dot(x, (y * s[:, None]).T), lambda x, s, y: jnp.dot(x, y.T) * s[None, :],
                _ROW_SCALED)))


def _joined() -> tuple[RuleFamily, ...]:
    """Two sums along one axis of operands of one shape as one sum of their terms (linearity), in their
    operands' dtype (`plain`), where the new sum of terms and the new reduction of it, its length's power
    (`sum_length`) above them, are each under the program's reach (`dominated`): `sum(x) + sum(y)` may be
    finite where `sum(x + y)` is not (float32's `x = [2**126, 2**126]` and
    `y = [2**126 - 2**102, 2**126 - 2**103]`)."""
    x, y, t = vars_("x y t", Term)
    axes = var("axes", Vec[i64])
    params, added = vars_("params added", i64)
    length = var("length", i64)
    facts, power = summed_magnitude(magnitude(x, "x"), magnitude(y, "y"))
    new = x.binary("add", y, added)
    reduced = new.reduce("sum", axes, params)
    summed_x = x.reduce("sum", axes, params)
    return (RuleFamily("two sums along an axis", (rule(
        eq(t).to(summed_x.binary("add", y.reduce("sum", axes, params), added)), eq(shape(x)).to(shape(y)),
        plain(added), eq(length).to(sum_length(summed_x)), *facts, *dominated(power + length, "new")).then(
        union(t).with_(reduced), *made(new, power), *made(reduced, power + length)),),
        (lambda x, y, a: jnp.sum(x, -1) + jnp.sum(y, -1), lambda x, y, a: jnp.sum(x + y, -1), _ROWS)),)


CONTRACTED = {"mul": "fmul", "add": "fadd", "sub": "fsub"}
"""A fused multiply-add's names (`_contracted`): its multiply's and the add's or subtract's that reads it,
each bound as the primitive it names (its params'), the multiply priced nothing beside its reader
(`cost.work`)."""


def _contracted() -> tuple[RuleFamily, ...]:
    """A multiply an add or a subtract reads first, contracted into it as a fused multiply-add: the compiler
    fuses the two into one instruction rounding the product once (NVPTX's `fma`), so the pair costs one
    operation an element where the multiply has no other reader. A multiply another node reads too (a scaled
    tile whose maximum is taken) is made on its own and its reader's add is one more: the contracted multiply
    is a class the multiply's is not, which its contracted reader alone reads, and each add contracts one
    product (the first operand, an add's either by commuting it). Each rewrite binds the primitives it
    replaces, so it rounds as the form it replaces."""
    x, y, z = vars_("x y z", Term)
    multiplied, added = vars_("multiplied added", i64)
    fused = x.binary(CONTRACTED["mul"], y, multiplied)
    return (
        RuleFamily("a multiply an add reads", (
            rewrite(x.binary("mul", y, multiplied).binary("add", z, added)).to(
                fused.binary(CONTRACTED["add"], z, added)),
            rewrite(z.binary("add", x.binary("mul", y, multiplied), added)).to(
                fused.binary(CONTRACTED["add"], z, added))),
            (lambda x, y, a: a * x + y, lambda x, y, a: a * x + y, _ROWS)),
        RuleFamily("a multiply a subtract reads", (
            rewrite(x.binary("mul", y, multiplied).binary("sub", z, added)).to(
                fused.binary(CONTRACTED["sub"], z, added)),),
            (lambda x, y, a: a * x - y, lambda x, y, a: a * x - y, _ROWS)))


def _identities() -> tuple[RuleFamily, ...]:
    """Commutativity of addition and multiplication: each rounds as the form it replaces."""
    x, y = vars_("x y", Term)
    params = var("params", i64)
    return (RuleFamily("addition's commutativity", (birewrite(x.binary("add", y, params)).to(
        y.binary("add", x, params)),), (lambda x, y, a: x + y, lambda x, y, a: y + x, _ROWS)),
            RuleFamily("multiplication's commutativity", (birewrite(x.binary("mul", y, params)).to(
                y.binary("mul", x, params)),), (lambda x, y, a: x * y, lambda x, y, a: y * x, _ROWS)))


FACTS = _facts()
RULES: dict[str, Rules] = {}
_HEADS: dict[str, set[tuple[str, ...]]] = {}
"""What each registered rule's left side reads at its head (`_head`), by the operation it heads at: the
operations that make its operands, where the left side states them. The program pass opens a window at an
equation one of them matches (`heads`, `program.WINDOW` deep), so a rule is reachable from the moment it is
registered; a set that only aids the others (`Rules.aid`: commutativity leaves every price as it was) or runs
at dev time alone (`Rules.dev`) opens none."""


def _operation(pattern) -> declarations.CallDecl | None:
    """A left side's term where it writes an operation: a call of a method of the term it reads
    (`Term.binary`, `Term.dot`); None for a variable, which any value answers, and for a leaf, a literal or
    a state, the class's own constructors."""
    held = pattern.expr if isinstance(pattern, declarations.TypedExprDecl) else pattern
    if isinstance(held, declarations.CallDecl) and isinstance(held.callable, declarations.MethodRef):
        return held
    return None


def _spoken(pattern) -> str | None:
    """The symbol a left side's term is spoken as (`terms.written_as`), None where it writes no operation
    (`_operation`)."""
    call = _operation(pattern)
    if call is None or not isinstance(call.callable, declarations.MethodRef):
        return None
    named = call.args[1].expr if len(call.args) > 1 else None
    name = named.value if isinstance(named, declarations.LitDecl) else None
    return terms.written_as(call.callable.method_name, name if isinstance(name, str) else None)


def _left_side(decl):
    """A rule's left side: a rewrite's own; a rule's the term its union rewrites, read from the equality
    that binds the root it unions (`eq(t).to(...)`, either way round and wherever among its facts), or the
    union's own left where it is written out; None where it states none."""
    if not isinstance(decl, declarations.RuleDecl):
        return decl.lhs
    rewritten = next((action.lhs for action in decl.head if isinstance(action, declarations.UnionDecl)), None)
    if rewritten is None or isinstance(rewritten, declarations.CallDecl):
        return rewritten
    for fact in decl.body:
        if isinstance(fact, declarations.EqDecl) and rewritten in (fact.left, fact.right):
            return fact.right if fact.left == rewritten else fact.left
    return None


def _head(written) -> tuple[str, tuple[str, ...]] | None:
    """The head of a rule's left side (`_left_side`): the operation it heads at and those that make its
    operands where it states them, in order of their names; None where it heads at no operation."""
    left = _left_side(written.decl)
    call, root = (None, None) if left is None else (_operation(left), _spoken(left))
    if call is None or root is None:
        return None
    made = (_spoken(operand) for operand in call.args if isinstance(operand, declarations.TypedExprDecl))
    return root, tuple(sorted(symbol for symbol in made if symbol is not None))


def register(rules: Rules) -> None:
    """Add a set of rules to the registry, which runs them in the order they were registered: a set states
    its rewrites, each with the instance that qualifies it (`Rules.declared`, `instances`), or is refused;
    each rule of a set that may change a program opens the pass's windows at its own left side's head
    (`_HEADS`), and one whose left side heads at no operation of the language, which no window could open
    for, is refused."""
    if rules.name in RULES:
        raise ValueError(f"a rule set named {rules.name!r} is registered already")
    if not rules.families:
        raise ValueError(f"the rule set {rules.name!r} declares no rewrites to qualify it by")
    found = [] if rules.aid or rules.dev else [
        (family.name, _head(written)) for family in rules.families for written in family.rules]
    headless = [name for name, head in found if head is None]
    if headless:
        raise ValueError(f"the rule set {rules.name!r}: the left side of {headless} heads at no operation "
                         f"of the language, so no window of a program opens for it")
    RULES[rules.name] = rules
    for root, made in (head for _, head in found if head is not None):
        _HEADS.setdefault(root, set()).add(made)


register(Rules.declared("identities", "exact", _identities(), "docs/design/algebra.md §1", aid=True))
register(Rules.declared("contracted", "exact", _contracted(),
                        "docs/design/algebra.md §6.5; the scaled softmax's maximum, ledger 3721", aid=True))
register(Rules.declared("stable", "bounded", _stable(),
                        "docs/design/algebra.md §1.5; the log-sum-exp fold, feat/attention-derived 12f9398ba",
                        dev=True))
register(Rules.declared("linear", "reorders", _linear(), "docs/design/algebra.md §1; ledger 1308"))
register(Rules.declared("joined", "reorders", _joined(), "docs/design/algebra.md §1; ledger 1308"))


def heads(primitive: str, operands: Sequence[str | None]) -> bool:
    """Whether an equation binding `primitive` over operands made by `operands` (each producer's primitive, or
    None for a value from outside) is a head a registered rule may lower the price at: some rule's left side
    heads at that operation over operands of those makers (`_HEADS`, each rule's own as it is registered),
    each of the equation's operands answering one of the left side's at most, in any order (commutativity
    may have swapped them)."""
    made = collections.Counter(operand for operand in operands if operand is not None)
    return any(not collections.Counter(needed) - made for needed in _HEADS.get(primitive, ()))


def reorderable(region: terms.Region) -> bool:
    """Whether a reordering rule may fire in `region`: each makes its new intermediate of operands the
    program states a number bound of (`dominated`, `made`: a sum's terms, a product's moved factor), so it
    needs a bounded term other than a literal, or two literals one value is multiplied by, whose sum
    distributivity takes (`a * 2 + a * 4` as `a * 6`). A literal's scale an extremum reduces moves out of it
    making no new magnitude (`positive_scale`), so a value scaled by a literal under a maximum or a minimum
    admits it too. A program's own math over its unbounded values (most
    of a training step's: a gemma3n step's 2879 regions each fired commutativity alone) admits none, and is
    left as traced. It reads the region's records alone, so a window's structure (`terms.Skeleton`) decides it
    before its terms are built."""
    literals = region.literal_keys
    if any(bound is not None and named not in literals for named, bound in region.magnitudes.items()):
        return True
    scaled: set = set()
    by_literal: set = set()
    for term, _ in region.shapes:
        found = terms.binary_of(term)
        if found is None or found[0] != "mul":
            continue
        _, left, right = found
        for operand, other in ((left, right), (right, left)):
            if other in literals:
                if operand in scaled:
                    return True
                scaled.add(operand)
                by_literal.add(terms.key(term))
    extrema = (terms.reduce_of(term) for term, _ in region.shapes)
    return any(found is not None and found[0] in ("max", "min") and found[1] in by_literal
               for found in extrema)


def admitted(accurate: bool, *, dev: bool = False) -> tuple[Rules, ...]:
    """The rule sets a program's numerics admit: every set under default numerics, the `exact` ones alone
    under the accurate numerics (`precision="accurate"`); the dev-time sets (`Rules.dev`) only where `dev`,
    whose choice measures each candidate's error instead."""
    return tuple(rules for rules in RULES.values()
                 if (dev or not rules.dev) and (dev or not accurate or rules.rounding == "exact"))


def instances() -> Mapping[str, Instance]:
    """Every registered rule written as JAX, by its set's name and its own (`Rules.instances`): the
    qualification inventory its accuracy test holds each right side to its left side over."""
    return {f"{rules.name}: {name}": instance for rules in RULES.values()
            for name, instance in rules.instances.items()}
