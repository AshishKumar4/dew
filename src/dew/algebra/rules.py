"""The rules Dew's rewrite applies (docs/design/algebra.md §6.4), in named sets, and the facts they read
(§6.3): each set an egglog ruleset with its rounding class, registered by name.

`shape(t)` is each value's shape, as the equation that made it states it (`terms.region`), carried to every
form the rules make: a rule joins or moves values only where their shapes say the result is the program's.
`along(params, axis)` holds where a broadcast lays its operand along every axis of its result but `axis`: a
value it makes is constant along `axis`, so a sum along that axis may take it out."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Literal

import jax.numpy as jnp
import numpy as np
from egglog import (
    Ruleset,
    String,
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


@dataclasses.dataclass(frozen=True)
class Rules:
    """A named set of rules: its rounding class (`exact`: each rewrite rounds as the form it replaces;
    `reorders`: it reorders rounding, admitted under default numerics alone), its egglog ruleset, and the
    measurement it ships with."""

    name: str
    rounding: Literal["exact", "reorders", "bounded"]
    ruleset: Ruleset
    receipt: str


def _facts() -> Ruleset:
    """The shape of each value a rule makes: an elementwise term's over operands of one shape theirs, a
    layout-keeping one its operand's, a sum along one axis its operand's without that axis."""
    term, x, y = vars_("term x y", Term)
    shaped, axes = vars_("shaped axes", Vec[i64])
    name = var("name", String)
    params = var("params", i64)
    return ruleset(
        rule(eq(term).to(x.binary(name, y, params)), eq(shaped).to(shape(x)), eq(shape(y)).to(shaped)).then(
            set_(shape(term)).to(shaped)),
        rule(eq(term).to(x.unary(name, params)), eq(shaped).to(shape(x))).then(set_(shape(term)).to(shaped)),
        rule(eq(term).to(x.convert(params)), eq(shaped).to(shape(x))).then(set_(shape(term)).to(shaped)),
        rule(eq(term).to(x.reduce(name, axes, params)), eq(shaped).to(shape(x)), eq(axes.length()).to(i64(1))
             ).then(set_(shape(term)).to(shaped.remove(axes[i64(0)]))),
        name="facts")


def _linear() -> Ruleset:
    """Linearity of sums (two sums along one axis of operands of one shape, one sum of their terms),
    distributivity over operands of one shape, a product's bilinearity (two products sharing an operand, one
    product of the summed others), each over arithmetic in its operands' dtype (`plain`), and a
    value a broadcast lays along every axis but a sum's
    taken out of the sum where the other factor has the product's shape (the broadcast's extent along the axis
    the product's or 1, as an elementwise product broadcasts a unit axis): each reorders rounding."""
    x, y, a = vars_("x y a", Term)
    axes, extent = vars_("axes extent", Vec[i64])
    params, laid, axis, added, multiplied, summed = vars_("params laid axis added multiplied summed", i64)
    return ruleset(
        rewrite(x.reduce("sum", axes, params).binary("add", y.reduce("sum", axes, params), added)).to(
            x.binary("add", y, added).reduce("sum", axes, params), eq(shape(x)).to(shape(y)), plain(added)),
        rewrite(a.binary("mul", x, multiplied).binary("add", a.binary("mul", y, multiplied), added)).to(
            a.binary("mul", x.binary("add", y, added), multiplied), eq(shape(x)).to(shape(y)),
            plain(multiplied), plain(added)),
        rewrite(a.dot(x, multiplied).binary("add", a.dot(y, multiplied), added)).to(
            a.dot(x.binary("add", y, added), multiplied), eq(shape(x)).to(shape(y)), plain(multiplied),
            plain(added)),
        rewrite(x.dot(a, multiplied).binary("add", y.dot(a, multiplied), added)).to(
            x.binary("add", y, added).dot(a, multiplied), eq(shape(x)).to(shape(y)), plain(multiplied),
            plain(added)),
        rewrite(a.shaped("broadcast_in_dim", laid).binary("mul", x, multiplied).reduce(
            "sum", axes, summed)).to(
            a.binary("mul", x.reduce("sum", axes, summed), multiplied), along(laid, axis), plain(multiplied),
            eq(axes).to(Vec[i64](axis)), eq(shape(x)).to(extent),
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
register(Rules("linear", "reorders", _linear(), receipt="docs/design/algebra.md §1; ledger 1308"))


def may_rewrite(primitives: Mapping[str, int]) -> bool:
    """Whether a region holding `primitives` (each primitive's name, how many equations bind it) holds what a
    rule's left side reads that may lower its price: a sum of two reductions, two products or two
    contractions (`linear`), or a reduction of a product. Commutativity alone (`identities`) leaves every
    price as it was, so a region of none is kept as traced, unsaturated."""
    adds, sums = primitives.get("add", 0), primitives.get("reduce_sum", 0)
    products, contractions = primitives.get("mul", 0), primitives.get("dot_general", 0)
    return (adds > 0 and (sums > 1 or products > 1 or contractions > 1)) or (sums > 0 and products > 0)


def admitted(accurate: bool) -> tuple[Rules, ...]:
    """The rule sets a program's numerics admit: every set under default numerics, the `exact` ones alone
    under `--precision accurate`."""
    return tuple(rules for rules in RULES.values() if not accurate or rules.rounding == "exact")


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
        "identities: commutativity": (lambda x, y, a: x + y, lambda x, y, a: y + x, rows),
    }
