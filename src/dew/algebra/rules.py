"""The rules Dew's rewrite applies (docs/design/algebra.md §6.4), in named sets, and the facts they read
(§6.3): each set an egglog ruleset with its rounding class, registered by name.

`elements(t)` is each value's number of elements, which the cost model prices passes by: a leaf, a broadcast
and a reduction state theirs, every elementwise term takes its operands'. `along(params, axis)` holds where
a broadcast lays its operand along every axis of its result but `axis`: a value it makes is constant along
`axis`, so a sum along that axis may take it out."""

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
    f64,
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


@function(merge=lambda old, new: old.max(new))
def elements(term: Term) -> i64: ...


@function
def along(params: i64, axis: i64) -> Unit: ...
"""Holds where the broadcast whose parameters are `params` (`Region.params`) lays its operand along every axis
of its result but `axis`, in order, so its result is constant along `axis` (`program.rewritten` states it)."""


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
    """The `elements` fact of every term from its operands', or the count it states."""
    term, left, right, operand = vars_("term left right operand", Term)
    count, other, params = vars_("count other params", i64)
    name = var("name", String)
    axes = var("axes", Vec[i64])
    value = var("value", f64)
    return ruleset(
        rule(eq(term).to(Term.leaf(params, count))).then(set_(elements(term)).to(count)),
        rule(eq(term).to(Term.literal(value, params))).then(set_(elements(term)).to(i64(1))),
        rule(eq(term).to(left.binary(name, right)), eq(count).to(elements(left)),
             eq(other).to(elements(right))).then(set_(elements(term)).to(count.max(other))),
        rule(eq(term).to(left.compare(name, right)), eq(count).to(elements(left)),
             eq(other).to(elements(right))).then(set_(elements(term)).to(count.max(other))),
        rule(eq(term).to(operand.select(left, right)), eq(count).to(elements(left)),
             eq(other).to(elements(right))).then(set_(elements(term)).to(count.max(other))),
        rule(eq(term).to(operand.unary(name, params)), eq(count).to(elements(operand))
             ).then(set_(elements(term)).to(count)),
        rule(eq(term).to(operand.convert(params)), eq(count).to(elements(operand))
             ).then(set_(elements(term)).to(count)),
        rule(eq(term).to(operand.shaped(name, params, count))).then(set_(elements(term)).to(count)),
        rule(eq(term).to(operand.reduce(name, axes, params, count))).then(set_(elements(term)).to(count)),
        name="facts")


def _linear() -> Ruleset:
    """Linearity of sums (two sums along one axis, one sum of their terms), distributivity, and a value
    constant along a sum's axis taken out of it: each reorders rounding."""
    x, y, a = vars_("x y a", Term)
    axes = var("axes", Vec[i64])
    params, count, laid, spread, axis = vars_("params count laid spread axis", i64)
    return ruleset(
        rewrite(x.reduce("sum", axes, params, count).binary("add", y.reduce("sum", axes, params, count))).to(
            x.binary("add", y).reduce("sum", axes, params, count)),
        birewrite(a.binary("mul", x).binary("add", a.binary("mul", y))).to(
            a.binary("mul", x.binary("add", y))),
        rewrite(a.shaped("broadcast_in_dim", laid, spread).binary("mul", x).reduce(
            "sum", Vec[i64](axis), params, count)).to(
            a.binary("mul", x.reduce("sum", Vec[i64](axis), params, count)), along(laid, axis)),
        name="linear")


def _identities() -> Ruleset:
    """Commutativity of addition and multiplication: each rounds as the form it replaces."""
    x, y = vars_("x y", Term)
    return ruleset(rewrite(x.binary("add", y)).to(y.binary("add", x)),
                   rewrite(x.binary("mul", y)).to(y.binary("mul", x)), name="identities")


FACTS = _facts()
RULES: dict[str, Rules] = {}


def register(rules: Rules) -> None:
    """Add a set of rules to the registry, which runs them in the order they were registered."""
    if rules.name in RULES:
        raise ValueError(f"a rule set named {rules.name!r} is registered already")
    RULES[rules.name] = rules


register(Rules("identities", "exact", _identities(), receipt="docs/design/algebra.md §1"))
register(Rules("linear", "reorders", _linear(), receipt="docs/design/algebra.md §1; ledger 1308"))


def admitted(accurate: bool) -> tuple[Rules, ...]:
    """The rule sets a program's numerics admit: every set under default numerics, the `exact` ones alone
    under `--precision accurate`."""
    return tuple(rules for rules in RULES.values() if not accurate or rules.rounding == "exact")


def instances() -> Mapping[str, tuple[Callable, Callable, Callable]]:
    """Each rule written as JAX, its left side and its right side, with draws of its operands (drawn rows and
    adversarial ones: a large and a small term together), for its accuracy test."""
    def draw(*shapes):
        def drawn(seed):
            rng = np.random.default_rng(seed)
            scale = 10.0 ** rng.integers(-3, 4, size=len(shapes))
            return tuple(jnp.asarray(rng.standard_normal(shape) * scale[index], jnp.float32)
                         for index, shape in enumerate(shapes))
        return drawn

    rows = draw((8, 64), (8, 64), (8, 64))
    return {
        "linear: two sums along an axis": (lambda x, y, a: jnp.sum(x, -1) + jnp.sum(y, -1),
                                           lambda x, y, a: jnp.sum(x + y, -1), rows),
        "linear: distributivity": (lambda x, y, a: a * x + a * y, lambda x, y, a: a * (x + y), rows),
        "identities: commutativity": (lambda x, y, a: x + y, lambda x, y, a: y + x, rows),
    }
