"""The rules Dew's rewrite applies (docs/design/algebra.md §6.4), in named sets, and the facts they read
(§6.3): each set an egglog ruleset with its rounding class, registered by name.

`shape(t)` is each value's shape, as the equation that made it states it (`terms.region`), carried to every
form the rules make: a rule joins or moves values only where their shapes say the result is the program's.
`along(params, axis)` holds where a broadcast lays its operand along every axis of its result but `axis`: a
value it makes is constant along `axis`, so a sum along that axis may take it out."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import jax.numpy as jnp
import numpy as np
from egglog import (
    Ruleset,
    String,
    StringLike,
    Unit,
    Vec,
    birewrite,
    eq,
    function,
    i64,
    rewrite,
    rule,
    ruleset,
    set_,
    var,
    vars_,
)

from dew.algebra.terms import Term


@function(merge=lambda old, new: old)
def shape(term: Term) -> Vec[i64]: ...


@function
def along(params: i64, axis: i64) -> Unit: ...


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
def exact_scale(term: Term) -> Unit: ...
"""Holds where a term is a scalar literal of a power of two: a multiply by it rounds nothing (but past the
dtype's range), so moving it through a contraction leaves every rounding as it was (`program.rewrite`
states it). A scale of any other value moved so reorders rounding, which the gate refused on the gated delta
rule's adjoint (1.575 times the reference's error, past 1.5): it waits for error-aware extraction (§1.5)."""


@function
def tame(term: Term) -> Unit: ...
"""Holds where a value's magnitude, bounded by the region's range facts (`terms.Region.bounds`, §6.3), times
the region's largest extent is at most the root of its narrowest dtype's largest value: a sum of two tame
values, a sum of one along an axis and a product of two contracted along one all stay finite. A reordering
that makes an intermediate of its operands fires only where they are tame, so no form is infinite where the
program is finite (bible §6); with no range fact, it does not fire (`program.stated` states it)."""


@dataclasses.dataclass(frozen=True)
class Rules:
    """A named set of rules: its rounding class (`exact`: each rewrite rounds as the form it replaces;
    `reorders`: it reorders rounding, admitted under default numerics alone), its egglog ruleset, and the
    measurement it ships with."""

    name: str
    rounding: Literal["exact", "reorders", "bounded"]
    ruleset: Ruleset
    receipt: str
    dev: bool = False
    """Whether it runs at dev time alone, where each candidate's error is measured before it is taken
    (`dew.algebra.error.choose`): a set whose forms are chosen for their accuracy, at a cost the program pass
    would never pay."""


def _facts() -> Ruleset:
    """The shape of each value a rule makes: an elementwise term's over operands of one shape theirs, or over
    an operand and a scalar the operand's, a layout-keeping one its operand's, a sum along one axis its
    operand's without that axis."""
    term, x, y = vars_("term x y", Term)
    shaped, axes = vars_("shaped axes", Vec[i64])
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
        name="facts")


def _stable() -> Ruleset:
    """Stable forms (docs/design/algebra.md §1.5): `exp(x - (m + log t))` as `exp(x - m) / t`, so the
    logarithm of a sum is never rounded into a value as large as the logits (a derived attention backward's
    mapped dK and dQ came to 1.6-1.7 times the reference's error without it); `log(1 + x)` as `log1p(x)`
    and `exp(x) - 1` as `expm1(x)`, which keep a small `x`'s digits. Each costs what it costs: the program
    pass never runs them, and dev time takes a form of theirs only where its measured error asks it."""
    x, m, t, one = vars_("x m t one", Term)
    subbed, added, logged, exped = vars_("subbed added logged exped", i64)
    # The stock primitives' indices are read in a rule's conditions, where egglog allows a lookup.
    divided, plus_one, minus_one = vars_("divided plus_one minus_one", i64)
    return ruleset(
        rewrite(x.binary("sub", m.binary("add", t.unary("log", logged), added), subbed)
                .unary("exp", exped)).to(
            x.binary("sub", m, subbed).unary("exp", exped).binary("div", t, divided),
            eq(shape(m)).to(shape(x)), eq(shape(t)).to(shape(x)), plain(subbed), plain(added),
            eq(divided).to(stock("div"))),
        rewrite(one.binary("add", x, added).unary("log", logged)).to(
            x.unary("log1p", plus_one), unit(one), plain(added), eq(plus_one).to(stock("log1p"))),
        rewrite(x.binary("add", one, added).unary("log", logged)).to(
            x.unary("log1p", plus_one), unit(one), plain(added), eq(plus_one).to(stock("log1p"))),
        rewrite(x.unary("exp", exped).binary("sub", one, subbed)).to(
            x.unary("expm1", minus_one), unit(one), plain(subbed), eq(minus_one).to(stock("expm1"))),
        name="stable")


def _linear() -> Ruleset:
    """Linearity of sums (two sums along one axis of operands of one shape, one sum of their terms),
    distributivity over operands of one shape, a product's bilinearity (two products sharing an operand, one
    product of the summed others, and a power of two's scale of the product on either operand instead), each
    over arithmetic in its operands' dtype (`plain`), and a value a broadcast lays along every axis but a
    sum's taken out of the sum where the other factor has the product's shape (the broadcast's extent along
    the axis the product's or 1, as an elementwise product broadcasts a unit axis): each reorders rounding,
    and each fires only where the operands its new intermediate reads are tame (`tame`)."""
    x, y, a = vars_("x y a", Term)
    axes, extent = vars_("axes extent", Vec[i64])
    params, laid, axis, added, multiplied, summed = vars_("params laid axis added multiplied summed", i64)
    scalar = Vec[i64].empty()
    return ruleset(
        # A contraction scaled by a power of two: the scale on either operand, the product linear in each.
        rewrite(x.dot(y, params).binary("mul", a, multiplied)).to(
            x.binary("mul", a, multiplied).dot(y, params), eq(shape(a)).to(scalar), exact_scale(a),
            plain(params), plain(multiplied), tame(x), tame(a)),
        rewrite(x.dot(y, params).binary("mul", a, multiplied)).to(
            x.dot(y.binary("mul", a, multiplied), params), eq(shape(a)).to(scalar), exact_scale(a),
            plain(params), plain(multiplied), tame(y), tame(a)),
        rewrite(x.reduce("sum", axes, params).binary("add", y.reduce("sum", axes, params), added)).to(
            x.binary("add", y, added).reduce("sum", axes, params), eq(shape(x)).to(shape(y)), plain(added),
            tame(x), tame(y)),
        rewrite(a.binary("mul", x, multiplied).binary("add", a.binary("mul", y, multiplied), added)).to(
            a.binary("mul", x.binary("add", y, added), multiplied), eq(shape(x)).to(shape(y)),
            plain(multiplied), plain(added), tame(x), tame(y)),
        rewrite(a.dot(x, multiplied).binary("add", a.dot(y, multiplied), added)).to(
            a.dot(x.binary("add", y, added), multiplied), eq(shape(x)).to(shape(y)), plain(multiplied),
            plain(added), tame(x), tame(y)),
        rewrite(x.dot(a, multiplied).binary("add", y.dot(a, multiplied), added)).to(
            x.binary("add", y, added).dot(a, multiplied), eq(shape(x)).to(shape(y)), plain(multiplied),
            plain(added), tame(x), tame(y)),
        rewrite(a.shaped("broadcast_in_dim", laid).binary("mul", x, multiplied).reduce(
            "sum", axes, summed)).to(
            a.binary("mul", x.reduce("sum", axes, summed), multiplied), along(laid, axis), plain(multiplied),
            eq(axes).to(Vec[i64](axis)), eq(shape(x)).to(extent), tame(x),
            eq(shape(a.shaped("broadcast_in_dim", laid).binary("mul", x, multiplied))).to(extent)),
        name="linear")


def _identities() -> Ruleset:
    """Commutativity of addition and multiplication: each rounds as the form it replaces."""
    x, y = vars_("x y", Term)
    params = var("params", i64)
    return ruleset(birewrite(x.binary("add", y, params)).to(y.binary("add", x, params)),
                   birewrite(x.binary("mul", y, params)).to(y.binary("mul", x, params)), name="identities")


FACTS = _facts()
RULES: dict[str, Rules] = {}


def register(rules: Rules) -> None:
    """Add a set of rules to the registry, which runs them in the order they were registered."""
    if rules.name in RULES:
        raise ValueError(f"a rule set named {rules.name!r} is registered already")
    RULES[rules.name] = rules


register(Rules("identities", "exact", _identities(), receipt="docs/design/algebra.md §1"))
register(Rules("stable", "bounded", _stable(), dev=True,
               receipt="docs/design/algebra.md §1.5; the log-sum-exp fold, feat/attention-derived 12f9398ba"))
register(Rules("linear", "reorders", _linear(), receipt="docs/design/algebra.md §1; ledger 1308"))


HEADS = {"add": frozenset({"reduce_sum", "mul", "dot_general"}), "reduce_sum": frozenset({"mul"}),
         "mul": frozenset({"dot_general"})}
"""What a rule's left side that may lower a price reads at its head: a sum (`add`) of two reductions, two
products or two contractions of one kind (`linear`), a reduction of a product, or a contraction scaled.
Commutativity alone (`identities`) leaves every price as it was."""


def heads(primitive: str, operands: Sequence[str | None]) -> bool:
    """Whether an equation binding `primitive` over operands made by `operands` (each producer's primitive, or
    None for a value from outside) is a head a rule may lower the price at (`HEADS`)."""
    kinds = HEADS.get(primitive)
    if kinds is None:
        return False
    if primitive == "add":
        return len(operands) == 2 and operands[0] == operands[1] and operands[0] in kinds
    return any(operand in kinds for operand in operands)


def admitted(accurate: bool, *, dev: bool = False) -> tuple[Rules, ...]:
    """The rule sets a program's numerics admit: every set under default numerics, the `exact` ones alone
    under `--precision accurate`; the dev-time sets (`Rules.dev`) only where `dev`, whose choice measures
    each candidate's error instead."""
    return tuple(rules for rules in RULES.values()
                 if (dev or not rules.dev) and (dev or not accurate or rules.rounding == "exact"))


def instances() -> Mapping[str, tuple[Callable, Callable, Callable]]:
    """Each rule written as JAX, its left side and its right side, with draws of its operands (drawn rows and
    adversarial ones: a large and a small term together), for its accuracy test."""
    def draw(*shapes):
        def drawn(seed):
            # Each element's magnitude drawn across six decades, a large and a small term together in every
            # row, the same family each seed.
            rng = np.random.default_rng(seed)
            return tuple(jnp.asarray(rng.standard_normal(shape) * 10.0 ** rng.uniform(-3, 3, shape),
                                     jnp.float32) for shape in shapes)
        return drawn

    rows, scaled = draw((64, 1024), (64, 1024), (64, 1024)), draw((64, 1024), (64,))
    return {
        "linear: two sums along an axis": (lambda x, y, a: jnp.sum(x, -1) + jnp.sum(y, -1),
                                           lambda x, y, a: jnp.sum(x + y, -1), rows),
        "linear: distributivity": (lambda x, y, a: a * x + a * y, lambda x, y, a: a * (x + y), rows),
        "linear: a row scalar out of a sum": (lambda x, s: jnp.sum(s[:, None] * x, -1),
                                              lambda x, s: s * jnp.sum(x, -1), scaled),
        "linear: a scale through a contraction": (lambda x, y, a: jnp.dot(x, y.T) * 0.3,
                                                  lambda x, y, a: jnp.dot(x * 0.3, y.T), rows),
        "identities: commutativity": (lambda x, y, a: x + y, lambda x, y, a: y + x, rows),
    }
