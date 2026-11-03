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
    aid: bool = False
    """Whether the set only aids the others' saturation and changes no program on its own (commutativity,
    whose operand orders XLA's HloCSE reads alike): a pass whose admitted sets are all aids does not run
    (`program.rewritten`). Every other set's rewrite applies wherever the cost model prices it cheaper, with
    no condition on XLA (docs/design/algebra.md §6.5)."""


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
    sets = []
    # A contraction scaled by a power of two: the scale on either operand, the product linear in each. The
    # new operand and the new contraction over it each under the program's reach: a scale moved into a
    # contraction's operand scales each of its products before its sum (2**120 x times 2 summed past 2**128).
    for scaled, other, made_of in ((x, y, lambda s: s.dot(y, params)), (y, x, lambda s: x.dot(s, params))):
        facts, power = product_magnitude(magnitude(scaled, "s"), magnitude(a, "a"), 0)
        new = scaled.binary("mul", a, multiplied)
        bounds, contracted = product_magnitude(([], power), magnitude(other, "other"),
                                               var("replaced_length", i64))
        sets.append(rule(eq(t).to(x.dot(y, params).binary("mul", a, multiplied)), eq(shape(a)).to(scalar),
                         exact_scale(a), plain(params), plain(multiplied), *facts, *dominated(power, "new"),
                         eq(var("replaced_length", i64)).to(contracted_length(x.dot(y, params))), *bounds,
                         *dominated(contracted, "replaced")).then(
            union(t).with_(made_of(new)), *made(new, power)))
    facts, power = summed_magnitude(magnitude(x, "x"), magnitude(y, "y"))
    new = x.binary("add", y, added)
    # A shared operand factored out of two contractions: the summed operand and its contraction with the
    # shared one each under the program's reach.
    shared_bound = var("shared_bound", i64)
    shared_length = var("shared_length", i64)
    def factored(product: Term) -> tuple:
        # The factored contraction contracts what `product`, one of the two it replaces, does.
        return (eq(shared_bound).to(log2_bound(a)), eq(shared_length).to(contracted_length(product)),
                *dominated(power + shared_bound + shared_length, "factored"))

    sets += [
        rule(eq(t).to(a.binary("mul", x, multiplied).binary("add", a.binary("mul", y, multiplied), added)),
             eq(shape(x)).to(shape(y)), plain(multiplied), plain(added), *facts,
             *dominated(power, "new")).then(
            union(t).with_(a.binary("mul", new, multiplied)), *made(new, power)),
        rule(eq(t).to(a.dot(x, multiplied).binary("add", a.dot(y, multiplied), added)),
             eq(shape(x)).to(shape(y)), plain(multiplied), plain(added), *facts,
             *dominated(power, "new"), *factored(a.dot(x, multiplied))).then(
            union(t).with_(a.dot(new, multiplied)), *made(new, power)),
        rule(eq(t).to(x.dot(a, multiplied).binary("add", y.dot(a, multiplied), added)),
             eq(shape(x)).to(shape(y)), plain(multiplied), plain(added), *facts,
             *dominated(power, "new"), *factored(x.dot(a, multiplied))).then(
            union(t).with_(new.dot(a, multiplied)), *made(new, power)),
    ]
    # A positive scale out of a maximum or a minimum: the scaled elements' order is the elements', so the
    # reduced element is one, its one rounding of `c x` the result's; a scale broadcast to the operand's
    # shape (a scalar's) leaves the reduced value's as a scalar's.
    for reduced in ("max", "min"):
        for scaled in (lambda c: x.binary("mul", c, multiplied), lambda c: c.binary("mul", x, multiplied)):
            sets += [
                rule(eq(t).to(scaled(a).reduce(reduced, axes, summed)), positive_scale(a), plain(multiplied),
                     eq(shape(a)).to(scalar)).then(
                    union(t).with_(x.reduce(reduced, axes, summed).binary("mul", a, multiplied))),
                rule(eq(t).to(scaled(a.shaped("broadcast_in_dim", laid)).reduce(reduced, axes, summed)),
                     positive_scale(a), plain(multiplied), eq(shape(a)).to(scalar),
                     eq(shape(x)).to(shape(a.shaped("broadcast_in_dim", laid)))).then(
                    union(t).with_(x.reduce(reduced, axes, summed).binary("mul", a, multiplied))),
            ]
    facts, power = magnitude(x, "x")
    length = var("length", i64)
    sum_of = x.reduce("sum", axes, summed)
    sets.append(rule(
        eq(t).to(a.shaped("broadcast_in_dim", laid).binary("mul", x, multiplied).reduce("sum", axes, summed)),
        along(laid, axis), plain(multiplied), eq(axes).to(Vec[i64](axis)), eq(shape(x)).to(extent),
        eq(shape(a.shaped("broadcast_in_dim", laid).binary("mul", x, multiplied))).to(extent),
        eq(length).to(log2_of(extent[axis])), *facts, *dominated(power + length, "new")).then(
        union(t).with_(a.binary("mul", sum_of, multiplied)), *made(sum_of, power + length)))
    # A matrix product of an operand scaled along its kept axis (a row scale on the left, a column scale on
    # the right, constant along the contraction) as the product scaled: x@w new.
    s = var("s", Term)
    product_extent = var("product_extent", Vec[i64])
    length2, row_lift, column_lift = vars_("length2 row_lift column_lift", i64)
    left = var("left", Vec[i64])
    contracted = (eq(left).to(shape(x)), eq(length2).to(log2_of(left[i64(1)])))
    facts, power = product_magnitude(magnitude(x, "x"), magnitude(y, "y"), length2)
    product = x.dot(y, params)
    for scaled, laid_axis, lift_axis, lift in (
            (lambda: x.binary("mul", s.shaped("broadcast_in_dim", laid), multiplied).dot(y, params), 1, 0,
             row_lift),
            (lambda: s.shaped("broadcast_in_dim", laid).binary("mul", x, multiplied).dot(y, params), 1, 0,
             row_lift),
            (lambda: x.dot(y.binary("mul", s.shaped("broadcast_in_dim", laid), multiplied), params), 0, 1,
             column_lift),
            (lambda: x.dot(s.shaped("broadcast_in_dim", laid).binary("mul", y, multiplied), params), 0, 1,
             column_lift)):
        sets.append(rule(
            eq(t).to(scaled()), matrix_product(params), plain(multiplied), matrix(x), matrix(y),
            along(laid, i64(laid_axis)), eq(product_extent).to(shape(t)),
            eq(lift).to(lifted(product_extent, i64(lift_axis))),
            *contracted, *facts, *dominated(power, "new")).then(
            union(t).with_(s.shaped("broadcast_in_dim", lift).binary("mul", product, multiplied)),
            # The lifted broadcast lays the scale over the product's extent: its shape, so it is priced.
            set_(shape(s.shaped("broadcast_in_dim", lift))).to(product_extent),
            *made(product, power)))
    return ruleset(*sets, name="linear")


def _joined() -> Ruleset:
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
    return ruleset(
        rule(eq(t).to(summed_x.binary("add", y.reduce("sum", axes, params), added)),
             eq(shape(x)).to(shape(y)), plain(added), eq(length).to(sum_length(summed_x)), *facts,
             *dominated(power + length, "new")).then(
            union(t).with_(reduced), *made(new, power), *made(reduced, power + length)),
        name="joined")


CONTRACTED = {"mul": "fmul", "add": "fadd", "sub": "fsub"}
"""A fused multiply-add's names (`_contracted`): its multiply's and the add's or subtract's that reads it,
each bound as the primitive it names (its params'), the multiply priced nothing beside its reader
(`cost.work`)."""


def _contracted() -> Ruleset:
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
    return ruleset(
        rewrite(x.binary("mul", y, multiplied).binary("add", z, added)).to(
            fused.binary(CONTRACTED["add"], z, added)),
        rewrite(z.binary("add", x.binary("mul", y, multiplied), added)).to(
            fused.binary(CONTRACTED["add"], z, added)),
        rewrite(x.binary("mul", y, multiplied).binary("sub", z, added)).to(
            fused.binary(CONTRACTED["sub"], z, added)),
        name="contracted")


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


register(Rules("identities", "exact", _identities(), receipt="docs/design/algebra.md §1", aid=True))
register(Rules("contracted", "exact", _contracted(), aid=True,
               receipt="docs/design/algebra.md §6.5; the scaled softmax's maximum, ledger 3721"))
register(Rules("stable", "bounded", _stable(), dev=True,
               receipt="docs/design/algebra.md §1.5; the log-sum-exp fold, feat/attention-derived 12f9398ba"))
register(Rules("linear", "reorders", _linear(), receipt="docs/design/algebra.md §1; ledger 1308"))
register(Rules("joined", "reorders", _joined(), receipt="docs/design/algebra.md §1; ledger 1308"))


HEADS = {"add": frozenset({"reduce_sum", "mul", "dot_general"}), "reduce_sum": frozenset({"mul"}),
         "reduce_max": frozenset({"mul"}), "reduce_min": frozenset({"mul"}),
         "mul": frozenset({"dot_general"}), "dot_general": frozenset({"mul"})}
"""What a rule's left side that may lower a price reads at its head: a sum (`add`) of two reductions, two
products or two contractions of one kind (`linear`), a reduction of a product (a sum's, or an extremum's of a
positive scale), a contraction scaled, or a contraction of a scaled operand (a dequant's matrix scale, its
broadcast within `program.WINDOW`). Commutativity alone (`identities`) leaves every price as it was."""


def heads(primitive: str, operands: Sequence[str | None]) -> bool:
    """Whether an equation binding `primitive` over operands made by `operands` (each producer's primitive, or
    None for a value from outside) is a head a rule may lower the price at (`HEADS`)."""
    kinds = HEADS.get(primitive)
    if kinds is None:
        return False
    if primitive == "add":
        return len(operands) == 2 and operands[0] == operands[1] and operands[0] in kinds
    return any(operand in kinds for operand in operands)


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
