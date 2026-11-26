"""The rules Dew's rewrite applies (docs/design/algebra.md §6.4), in named sets, and the facts they read
(§6.3): each set an egglog ruleset with its rounding class, registered by name.

`shape(t)` is each value's shape, as the equation that made it states it (`terms.region`), carried to every
form the rules make: a rule joins or moves values only where their shapes say the result is the program's.
`along(params, axis)` holds where a broadcast lays its operand along every axis of its result but `axis`: a
value it makes is constant along `axis`, so a sum along that axis may take it out."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import importlib
from collections.abc import Callable, Iterator, Mapping, Sequence
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
from jax.typing import DTypeLike

from dew._cache import ALGEBRA
from dew.algebra import terms
from dew.algebra.terms import Term


@function(merge=lambda old, new: old)
def shape(term: Term) -> Vec[i64]: ...


@function(merge=lambda old, new: old)
def dtype(term: Term) -> String: ...
"""The scalar dtype name of `term` (`np.dtype(...).name`: `'float32'`, `'bfloat16'`, `'bool'`), stated for
every traced term by `program.stated` and propagated by `FACTS` so `cost.work` prices conversions and
divisions at their real dtypes (row 12974)."""


@function(merge=lambda old, new: old)
def param_dtype(params: i64) -> String: ...
"""The fixed output dtype name of `params` (`convert_element_type`'s `new_dtype`, `dot_general`'s
`Product.made`, `is_finite`'s `'bool'`, or a binary primitive's explicit `out_dtype`)."""


@function
def same_dtype(params: i64) -> Unit: ...
"""Holds where a unary or binary operation's parameters preserve its operands' dtype (every unary except
`is_finite`, and every binary without `out_dtype`)."""


@function
def along(params: i64, axis: i64) -> Unit: ...


@function(merge=lambda old, new: old)
def matrix_product(params: i64) -> i64: ...
"""The RHS kept axis of a plain matrix product with the left columns contracted."""


@function
def exact_int8_product(term: Term) -> Unit: ...
"""This short full-fp32 dot, exact on byte casts; a parameter index does not encode its length."""


@function
def int8_values(term: Term) -> Unit: ...
"""Float32 values obtained by widening intact integer bytes."""


@function
def exact_roundtrip(narrow: Term, original: Term) -> Unit: ...
"""The actual casts preserve the input's shape, placement and floating-point bits."""


@function
def dyadic_scale(term: Term) -> Unit: ...
"""Every scale is a positive power of two from 2**-100 through 2**100."""


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
def counting_sum(term: Term) -> Unit: ...
"""A sum of floating 0/1 values whose worst-case count doubled fits exactly in its dtype."""


@function(merge=lambda old, new: old)
def log2_bound(term: Term) -> i64: ...
"""A value's magnitude's power of two (`terms.Magnitude`), a number bound: the region's (`program.stated`), or
what a reordering rule states of the intermediate it makes (`made`). A value no number bounds has none."""


@function(merge=lambda old, new: old)
def log2_floor(term: Term) -> i64: ...
"""A value's floor away from zero's power of two, `2**floor <= |x|` of a number (`terms.Apart.floor`, the
region's `floors`, `program.stated`): what its reciprocal is at most, which a guarded reuse reads (`_reuse`).
A value nothing holds away from zero has none."""


@function
def positive(term: Term) -> Unit: ...
"""Holds where a value is above zero (`terms.Apart.positive`): a root's operand, and its reciprocal root's,
in their domain."""


@function(merge=lambda old, new: old)
def least() -> i64: ...
"""The least power of two a normal value of the program's narrowest float takes (`program.stated`): a
reciprocal under it would flush to zero on a lane that flushes, where the quotient it replaces keeps its
value (§6.3)."""


@function(merge=lambda old, new: old)
def narrowed(term: Term) -> i64: ...
"""The index of the conversion back to a value's own float (`stock`'s `convert_<dtype>`), stated of a value
of a float narrower than float32 (`terms.WIDENED`, `program.stated`): a widened sum's result returned in
it."""


@function(merge=lambda old, new: old)
def reach() -> i64: ...
"""The largest power of two a value of the program may take finite, the narrowest float's (`program.stated`):
a reordering's new intermediate under it is finite wherever the program is, as a bound of numbers alone holds
the program's values (§6.3, `dominated`)."""


@function(merge=lambda old, new: old)
def log2_of(size: i64Like) -> i64: ...
"""`ceil(log2(size))` of each extent the region's shapes hold: a contraction's or a sum's length's power."""


@function(merge=lambda old, new: old)
def mode_power(params: i64) -> i64: ...
"""The powers a contraction of parameters `params` adds to its bound for its mode (`terms.product_slack`: none
for one that rounds each operand once and accumulates, one where it splits its operands into parts whose
products it sums), stated for each product of a region whose mode's rounding is stated and for none other: a
rule that makes a product at `params` adds it to the product's length's power (`contracted_length` holds the
two together for a traced product), so no rule makes a product of a mode nothing bounds."""


@function(merge=lambda old, new: old)
def contracted_length(product: Term) -> i64: ...
"""A contraction's (`Term.dot`'s, by its term) length's power of two, `ceil(log2(...))` of the elements each
of its outputs sums: a rule that makes a contraction bounds its products' and its sum's magnitude by its
operands' and this (`product_magnitude`), read off the product it replaces, which contracts the same
elements. Stated per product, not per parameters' index, which two products of different lengths share
(row 3577). Its mode's powers are in it (`terms.contraction_slack`: one more where the mode splits its
operands into parts), so the product a rule makes in that mode is bounded as the mode computes it. A
contraction of no stated length (an operand of no known shape, a mode whose rounding nothing states) has none,
so a rule that asks it does not fire."""


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
    dtype: DTypeLike = np.float32
    """The float its operands are drawn in (a narrow float's sum's, `jnp.bfloat16`)."""

    def _values(self, seed: int) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        drawn: list[np.ndarray] = []
        for shape in self.shapes:
            drawn.append(np.asarray(rng.standard_normal(shape) * 10.0 ** rng.uniform(-3, 3, shape)
                                    if self.scale is None else self.scale(rng, shape, drawn), self.dtype))
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
            return np.where((np.arange(len(value)) % 2 == 0)[:, None], np.zeros((), value.dtype), value)

        def one_a_row(place, value, first):
            kept = np.arange(value.shape[-1])[None, :] == (np.arange(len(value)) % value.shape[-1])[:, None]
            return np.where(kept, value, np.zeros((), value.dtype))

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
    """The shape and dtype of each value a rule makes: an elementwise term's shape over operands of one shape
    theirs, or over an operand and a scalar the operand's, a layout-keeping one its operand's, a sum along
    one axis its operand's without that axis, a matrix's transpose its axes swapped, a matrix product its
    left operand's rows by its right's columns, each value of two axes a matrix (`matrix`), and each term's
    dtype (`dtype`) propagated from operands and parameter dtypes (`same_dtype`, `param_dtype`)."""
    term, x, y, z = vars_("term x y z", Term)
    shaped, axes, left, right = vars_("shaped axes left right", Vec[i64])
    name, dt = vars_("name dt", String)
    params, kept = vars_("params kept", i64)
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
        rule(eq(term).to(x.dot(y, params)), eq(kept).to(matrix_product(params)), eq(left).to(shape(x)),
             eq(right).to(shape(y)), eq(left.length()).to(i64(2)), eq(right.length()).to(i64(2))).then(
            set_(shape(term)).to(Vec[i64](left[i64(0)], right[kept]))),
        rule(eq(shaped).to(shape(term)), eq(shaped.length()).to(i64(2))).then(matrix(term)),
        # Dtype propagation across every Term constructor (`cost.dtype_of`, row 12974).
        rule(eq(term).to(x.binary(name, y, params)), same_dtype(params), eq(dt).to(dtype(x)),
             eq(dtype(y)).to(dt)).then(set_(dtype(term)).to(dt)),
        rule(eq(term).to(x.binary(name, y, params)), eq(dt).to(param_dtype(params))).then(
            set_(dtype(term)).to(dt)),
        rule(eq(term).to(x.compare(name, y, params))).then(set_(dtype(term)).to(String("bool"))),
        rule(eq(term).to(x.unary(name, params)), same_dtype(params), eq(dt).to(dtype(x))).then(
            set_(dtype(term)).to(dt)),
        rule(eq(term).to(x.unary(name, params)), eq(dt).to(param_dtype(params))).then(
            set_(dtype(term)).to(dt)),
        rule(eq(term).to(x.convert(params)), eq(dt).to(param_dtype(params))).then(set_(dtype(term)).to(dt)),
        rule(eq(term).to(x.shaped(name, params)), eq(dt).to(dtype(x))).then(set_(dtype(term)).to(dt)),
        rule(eq(term).to(x.reduce(name, axes, params)), eq(dt).to(dtype(x))).then(set_(dtype(term)).to(dt)),
        rule(eq(term).to(x.dot(y, params)), eq(dt).to(param_dtype(params))).then(set_(dtype(term)).to(dt)),
        rule(eq(term).to(z.select(x, y)), eq(dt).to(dtype(x)), eq(dtype(y)).to(dt)).then(
            set_(dtype(term)).to(dt)),
        rule(eq(term).to(z.clamp(x, y)), eq(dt).to(dtype(x))).then(set_(dtype(term)).to(dt)),
        name="facts")


_ROWS, _SCALED = Drawn(((64, 1024), (64, 1024), (64, 1024))), Drawn(((64, 1024), (64,)))
_SMALL = Drawn(((64, 1024),), scale=lambda rng, shape, drawn: (rng.uniform(-0.5, 0.5, shape)
                                                               * 10.0 ** rng.uniform(-6, 0, shape)))


def _half_normals(rng: np.random.Generator, shape: tuple[int, ...], drawn: list[np.ndarray]) -> np.ndarray:
    """Draw both signs of f16's normal encodings; the adversarial families add zeros."""
    del drawn
    magnitude = rng.integers(0x0400, 0x7C00, shape, dtype=np.uint16)
    sign = rng.integers(0, 2, shape, dtype=np.uint16) << 15
    return (sign | magnitude).view(np.float16)


_HALF_ROWS = Drawn(_ROWS.shapes, scale=_half_normals, dtype=jnp.float16)


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


_LOGITS = Drawn(((64, 1024), (64, 1024)), scale=lambda rng, shape, drawn: rng.standard_normal(shape) * 5.0)
"""Logits of a spread of five, whose exponentials stay in float32's range, and their quotients and products
too where the adversarial families cancel or repeat them (`Drawn.adversarial`): a logarithm of them is the
logit, which the form in log space keeps and the exponential rounds."""
_NEAR = Drawn(((64, 1024), (64, 1024), (64, 1024)), scale=lambda rng, shape, drawn: (
    rng.standard_normal(shape) * (1e4 if not drawn else 1.0)))
"""A large term and two small ones: the large one's rounding is what a sum of each carries, which their
difference cancels exactly where the large one is taken out."""
_CONTRACTED_SCALE = Drawn(((64, 128), (1, 128), (32, 128)), scale=lambda rng, shape, drawn: (
    rng.uniform(1.0, 1e3, shape) if shape[0] == 1 else rng.standard_normal(shape)))
"""A matrix product's left operand, a positive scale along its contraction (a softmax's totals, each at
least one) as a row, and its right operand transposed, so each operand's leading axis is one no other
shares (check 2 reads each operand's first rows)."""
_PER_ROW = Drawn(((64, 1024), (64, 1024), (64, 1)), scale=lambda rng, shape, drawn: (
    rng.standard_normal(shape) if shape[-1] > 1 else rng.uniform(0.5, 2.0, shape)))
"""Two operands of unit scale and a divisor a row, held away from zero (a row's norm): a regrouping's
roundings compared where no element's quotient dominates the error, and no adversarial family (each the
first operand's shape's, `Drawn.adversarial`) zeroes the divisor."""
_PER_ROW_DIVISORS = Drawn(((64, 1024), (64, 1), (64, 1)), scale=lambda rng, shape, drawn: (
    rng.standard_normal(shape) if shape[-1] > 1 else rng.uniform(0.5, 2.0, shape)))
"""An operand of unit scale and two divisors a row, held away from zero."""
_NARROW = Drawn(((64, 4096),), dtype=jnp.bfloat16)
"""Rows of bfloat16 long enough that a sum of its terms rounds in its own precision at every step."""


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
            (lambda x: jnp.exp(x) - 1, jnp.expm1, _SMALL)),
        *_log_space(), *_cancelled(), *_ranged(), *_moved_scales(), *_widened())


def _log_space() -> tuple[RuleFamily, ...]:
    """Logarithms of exponentials taken in log space (§1.4's log-space divergences: a KL's `log(p / q)` of
    two softmaxes' exponentials): `log(exp(a))` as `a`, `log(exp(a) / exp(b))` as `a - b` and
    `log(exp(a) exp(b))` as `a + b`, finite wherever the form as written is and wherever an exponential
    leaves the float's range."""
    a, b = vars_("a b", Term)
    exped, other, logged, joined, written = vars_("exped other logged joined written", i64)
    families = [RuleFamily("a logarithm of an exponential", (
        rewrite(a.unary("exp", exped).unary("log", logged)).to(a),),
        (lambda a, b: jnp.log(jnp.exp(a)), lambda a, b: a, _LOGITS))]
    for name, between, made, of, as_written in (
            ("a quotient", "div", "sub", lambda a, b: jnp.log(jnp.exp(a) / jnp.exp(b)), lambda a, b: a - b),
            ("a product", "mul", "add", lambda a, b: jnp.log(jnp.exp(a) * jnp.exp(b)), lambda a, b: a + b)):
        families.append(RuleFamily(f"a logarithm of {name} of exponentials", (
            rewrite(a.unary("exp", exped).binary(between, b.unary("exp", other), joined).unary("log", logged))
            .to(a.binary(made, b, written), plain(joined), eq(shape(a)).to(shape(b)),
                eq(written).to(stock(made))),), (of, as_written, _LOGITS)))
    return tuple(families)


def _cancelled() -> tuple[RuleFamily, ...]:
    """Differences of near-equal sums rearranged to cancel exactly (§1.4): `(x + y) - (x + z)` as `y - z` and
    `(x + y) - x` as `y`, the shared term's rounding in each sum never made (addition's commutativity, an
    aid, finds the other orders)."""
    x, y, z = vars_("x y z", Term)
    added, other, subbed = vars_("added other subbed", i64)
    return (
        RuleFamily("a difference of sums sharing a term", (
            rewrite(x.binary("add", y, added).binary("sub", x.binary("add", z, other), subbed)).to(
                y.binary("sub", z, subbed), plain(added), plain(other), plain(subbed),
                eq(shape(x)).to(shape(y)), eq(shape(x)).to(shape(z))),),
            (lambda x, y, z: (x + y) - (x + z), lambda x, y, z: y - z, _NEAR)),
        RuleFamily("a sum less its own term", (
            rewrite(x.binary("add", y, added).binary("sub", x, subbed)).to(
                y, plain(added), plain(subbed), eq(shape(x)).to(shape(y))),),
            (lambda x, y, z: (x + y) - x, lambda x, y, z: y, _NEAR)))


def _ranged() -> tuple[RuleFamily, ...]:
    """Products and quotients regrouped to stay in range (§1.4): `(a b) / c` as `a (b / c)` and `a / (b c)`
    as `(a / b) / c`, each over one shape, whichever keeps its intermediates in the float's range measured
    (`dew.algebra.error.choose`)."""
    a, b, c = vars_("a b c", Term)
    multiplied, divided = vars_("multiplied divided", i64)
    same = (eq(shape(a)).to(shape(b)), eq(shape(a)).to(shape(c)))
    return (
        RuleFamily("a product's quotient as a product of a quotient", (
            rewrite(a.binary("mul", b, multiplied).binary("div", c, divided)).to(
                a.binary("mul", b.binary("div", c, divided), multiplied), plain(multiplied), plain(divided),
                *same),),
            (lambda a, b, c: (a * b) / c, lambda a, b, c: a * (b / c), _PER_ROW)),
        RuleFamily("a quotient by a product as two quotients", (
            rewrite(a.binary("div", b.binary("mul", c, multiplied), divided)).to(
                a.binary("div", b, divided).binary("div", c, divided), plain(multiplied), plain(divided),
                *same),),
            (lambda a, b, c: a / (b * c), lambda a, b, c: (a / b) / c, _PER_ROW_DIVISORS)))


def _moved_scales() -> tuple[RuleFamily, ...]:
    """A scale along a matrix product's contraction moved to its other operand (§1.5's first acceptance
    case: a softmax's normalization before an attention backward's dV product or after it on the cotangent,
    1.64 times the oracle's error against on par): `(x op s) @ y` as `x @ (y op s)`, `s` broadcast along
    `x`'s contracted axis and laid along `y`'s, `op` a product or a quotient, and back; each reorders its
    rounding, and dev time measures which is kept."""
    x, y, s, t = vars_("x y s t", Term)
    params, laid, scaled, spread = vars_("params laid scaled spread", i64)
    extent = var("extent", Vec[i64])
    families = []
    for op, scale_of in (("mul", lambda v, s: v * s), ("div", lambda v, s: v / s)):
        moved = []
        for on_left in (True, False):
            # The scale lies along the scaled operand's contracted axis (the left's columns, the right's
            # rows): its broadcast misses the other axis, and its new one lies along the other operand's.
            source, target = (x, y) if on_left else (y, x)
            before = source.binary(op, s.shaped("broadcast_in_dim", laid), scaled)
            after = target.binary(op, s.shaped("broadcast_in_dim", spread), scaled)
            written, made = (before.dot(y, params), x.dot(after, params)) if on_left else (
                x.dot(before, params), after.dot(y, params))
            # The other operand's extent bound first: a rule's actions look up no function's value.
            moved.append(rule(
                eq(t).to(written), eq(matrix_product(params)).to(i64(1)), plain(scaled), matrix(x), matrix(y),
                along(laid, i64(0 if on_left else 1)), eq(extent).to(shape(target)),
                eq(spread).to(lifted(extent, i64(0 if on_left else 1)))).then(
                union(t).with_(made), set_(shape(s.shaped("broadcast_in_dim", spread))).to(extent)))
        families.append(RuleFamily(
            f"a {'scale' if op == 'mul' else 'divisor'} along a contraction moved to the other operand",
            tuple(moved), (lambda x, s, w, scale_of=scale_of: jnp.dot(scale_of(x, s), w.T),
                           lambda x, s, w, scale_of=scale_of: jnp.dot(x, scale_of(w, s).T),
                           _CONTRACTED_SCALE)))
    return tuple(families)


def _widened() -> tuple[RuleFamily, ...]:
    """A sum of a float narrower than float32 added in float32 and returned in its own (§1.4's widened sums,
    `terms.WIDENED`): each of its additions rounds in float32's precision, its result once in its own."""
    x = var("x", Term)
    axes = var("axes", Vec[i64])
    params, wide, back = vars_("params wide back", i64)
    return (RuleFamily("a narrow float's sum added in float32", (
        rewrite(x.reduce("sum", axes, params)).to(
            x.convert(wide).reduce("sum", axes, params).convert(back), eq(back).to(narrowed(x)),
            eq(wide).to(stock("convert_float32"))),),
        (lambda x: jnp.sum(x, -1), lambda x: jnp.sum(x.astype(jnp.float32), -1).astype(x.dtype), _NARROW)),)


_FLOORED = Drawn(((64, 1024), (64, 1024)), scale=lambda rng, shape, drawn: (
    rng.standard_normal(shape) if not drawn else np.maximum(np.abs(rng.standard_normal(shape)), 1e-3)))
"""A dividend and a divisor held away from zero (a norm floored by its epsilon, `max(|y|, 1e-3)`)."""
_BROADCAST_FLOORED = Drawn(((64, 1024), (64, 1)), scale=lambda rng, shape, drawn: (
    rng.standard_normal(shape) if not drawn else np.maximum(np.abs(rng.standard_normal(shape)), 1e-3)))
"""A dividend and a per-row broadcast divisor held away from zero (a softmax or row-norm denominator)."""


def _reuse() -> tuple[RuleFamily, ...]:
    """A value the program already computed read in a new one's place (§1.4's reuse) and guarded hoisting of
    broadcast divisors and reciprocal roots: `a / y` as `a (1 / y)` when `1 / y` is already held,
    `a / broadcast(y)` as `a * broadcast(y^-1)` when `y` is broadcast (e.g. softmax `e / sum(e)`), and
    `a / sqrt(v)` (directly or broadcast) as `a * rsqrt(v)` (merging into any sibling `rsqrt(v)`). Each
    reorders a rounding, and each fires only where the reciprocal is a normal finite float and the product
    finite (§6.3's dominance rule): the divisor held away from zero (`log2_floor`: its reciprocal at most
    `2**-floor`, under the program's reach) and bounded (`log2_bound`: its reciprocal at least `2**-bound`,
    above the narrowest float's least normal power, so no flushing lane reads it as zero), and either the
    divisor is at least 1 (`floor >= 0`, so `0 < |1/y| <= 1` can never overflow any finite dividend `a`) or
    the dividend's bound with the reciprocal's stays under the reach."""
    a, u, w, v, one, t, r = vars_("a u w v one t r", Term)
    axes, extent = vars_("axes extent", Vec[i64])
    divided, inverted, rooted, reciprocal, multiplied, made, laid, inv = vars_(
        "divided inverted rooted reciprocal multiplied made laid inv", i64)
    floor, bound, held, reached, lowest = vars_("floor bound held reached lowest", i64)
    common = (eq(reached).to(reach()), eq(lowest).to(least()), eq(multiplied).to(stock("mul")),
              plain(divided), eq(shape(a)).to(shape(t)))
    # A divisor's reciprocal is a normal finite float (above least normal and under reach).
    normal = (i64(0) - floor <= reached, lowest <= i64(0) - bound - 1)
    reciprocals = []
    makers = (u.binary("max", w, made), u.binary("add", w, made), u.reduce("sum", axes, made),
              u.clamp(w, v))
    r_laid = r.shaped("broadcast_in_dim", laid)
    for y in makers:
        held_forms = ((eq(r).to(one.binary("div", y, inverted)), unit(one), plain(inverted)),
                      (eq(inv).to(stock("reciprocal")), eq(r).to(y.unary("integer_pow", inv))))
        for observed in held_forms:
            for dividend_guard in ((eq(held).to(log2_bound(a)), held - floor <= reached), (i64(0) <= floor,)):
                divisor = (eq(floor).to(log2_floor(y)), eq(bound).to(log2_bound(y)))
                reciprocals += [
                    rule(eq(t).to(a.binary("div", y, divided)), *observed, eq(shape(r)).to(shape(t)),
                         *divisor, *common, *normal, *dividend_guard).then(
                        union(t).with_(a.binary("mul", r, multiplied))),
                    rule(eq(t).to(a.binary("div", y.shaped("broadcast_in_dim", laid), divided)),
                         *observed, eq(shape(r)).to(shape(y)),
                         eq(extent).to(shape(y.shaped("broadcast_in_dim", laid))),
                         *divisor, *common, *normal, *dividend_guard).then(
                        union(t).with_(a.binary("mul", r_laid, multiplied)), set_(shape(r_laid)).to(extent)),
                ]
    inv_v = v.unary("integer_pow", inv)
    inv_laid = inv_v.shaped("broadcast_in_dim", laid)
    hoisted_reciprocals = [
        rule(
            eq(t).to(a.binary("div", v.shaped("broadcast_in_dim", laid), divided)),
            eq(inv).to(stock("reciprocal")), eq(extent).to(shape(v.shaped("broadcast_in_dim", laid))),
            eq(floor).to(log2_floor(v)), eq(bound).to(log2_bound(v)), *common, *normal,
            *dividend_guard).then(
            union(t).with_(a.binary("mul", inv_laid, multiplied)),
            set_(shape(inv_laid)).to(extent))
        for dividend_guard in ((eq(held).to(log2_bound(a)), held - floor <= reached), (i64(0) <= floor,))
    ]
    rsq = v.unary("rsqrt", reciprocal)
    rsq_laid = rsq.shaped("broadcast_in_dim", laid)
    roots = []
    for dividend_guard in ((eq(held).to(log2_bound(a)), held + (i64(0) - floor).max(i64(0)) <= reached),
                           (i64(0) <= floor,)):
        roots += [
            rule(
                eq(t).to(a.binary("div", v.unary("sqrt", rooted), divided)),
                eq(reciprocal).to(stock("rsqrt")), positive(v),
                eq(floor).to(log2_floor(v)), eq(bound).to(log2_bound(v)), *common, *normal,
                *dividend_guard).then(union(t).with_(a.binary("mul", rsq, multiplied))),
            rule(
                eq(t).to(a.binary("div", v.unary("sqrt", rooted).shaped("broadcast_in_dim", laid), divided)),
                eq(reciprocal).to(stock("rsqrt")), positive(v),
                eq(extent).to(shape(v.unary("sqrt", rooted).shaped("broadcast_in_dim", laid))),
                eq(floor).to(log2_floor(v)), eq(bound).to(log2_bound(v)), *common, *normal,
                *dividend_guard).then(
                union(t).with_(a.binary("mul", rsq_laid, multiplied)), set_(shape(rsq_laid)).to(extent)),
            rule(
                eq(t).to(a.binary("div", v.shaped("broadcast_in_dim", laid).unary("sqrt", rooted), divided)),
                eq(reciprocal).to(stock("rsqrt")), positive(v),
                eq(extent).to(shape(v.shaped("broadcast_in_dim", laid))),
                eq(floor).to(log2_floor(v)), eq(bound).to(log2_bound(v)), *common, *normal,
                *dividend_guard).then(
                union(t).with_(a.binary("mul", rsq_laid, multiplied)), set_(shape(rsq_laid)).to(extent)),
        ]
    return (
        RuleFamily("a quotient by a value whose reciprocal the program holds", tuple(reciprocals),
                   (lambda a, y: a / jnp.maximum(y, 1e-3), lambda a, y: a * (1 / jnp.maximum(y, 1e-3)),
                    _FLOORED)),
        RuleFamily("a quotient by a broadcast divisor hoisted to a reciprocal product",
                   tuple(hoisted_reciprocals),
                   (lambda a, y: a / y, lambda a, y: a * jax.lax.integer_pow(y, -1), _BROADCAST_FLOORED)),
        RuleFamily("a quotient by a root as a reciprocal root product", tuple(roots),
                   (lambda a, v: a / jnp.sqrt(v), lambda a, v: a * jax.lax.rsqrt(v), _FLOORED)))


def _linear() -> tuple[RuleFamily, ...]:
    """Distributivity over operands of one shape, a positive scalar's scale out of a maximum or minimum
    (the same element wins, with its original rounding), and two matrix products sharing an operand as
    one product with the others summed. A scalar stays on its original side of a contraction: a power
    of two can flush an operand or product even when the final written result is normal (row 13162).
    A value a
    broadcast lays along every axis but a sum's taken out of the sum where the other factor has the product's
    shape (the broadcast's extent along the axis the product's or 1, as an elementwise product broadcasts a
    unit axis), and a matrix product of an operand scaled along its kept axis (a dequant's row or column
    scale) as the product scaled: each reorders rounding, and each fires only where every intermediate it
    makes, and the value it gives in the written one's place, is under a magnitude the program as written
    reaches (`dominated`, §6.3), each stating its own (`made`)."""
    x, y, a, t = vars_("x y a t", Term)
    axes, extent = vars_("axes extent", Vec[i64])
    laid, axis, added, multiplied, summed = vars_("laid axis added multiplied summed", i64)
    facts, power = summed_magnitude(magnitude(x, "x"), magnitude(y, "y"))
    new = x.binary("add", y, added)
    # The factored product too: `a (x + y)` rounds `x + y` before the product, so it may pass the float's
    # largest where `a x + a y` does not (float32's `a = (2**24 - 4) 2**104`, `x = 1`, `y = 3 2**-24`:
    # `x + y` rounds to `1 + 2**-22`, and `a (1 + 2**-22)` overflows where the written sum rounds to the
    # largest). Under the reach it is finite for every `a` its bound holds: `|a| 2**(max(x, y) + 1)`.
    more, factored = product_magnitude(magnitude(a, "a"), ([], power), 0)
    families = [
        RuleFamily("distributivity", (rule(
            eq(t).to(a.binary("mul", x, multiplied).binary("add", a.binary("mul", y, multiplied), added)),
            eq(shape(x)).to(shape(y)), plain(multiplied), plain(added), *facts,
            *dominated(power, "new"), *more, *dominated(factored, "factored")).then(
            union(t).with_(a.binary("mul", new, multiplied)), *made(new, power),
            *made(a.binary("mul", new, multiplied), factored)),),
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
    # The scaled sum too, which rounds the sum before its product (distributivity's).
    more, scaled = product_magnitude(magnitude(a, "a"), ([], power + length), 0)
    families.append(RuleFamily("a row scalar out of a sum", (rule(
        eq(t).to(a.shaped("broadcast_in_dim", laid).binary("mul", x, multiplied).reduce("sum", axes, summed)),
        along(laid, axis), plain(multiplied), eq(axes).to(Vec[i64](axis)), eq(shape(x)).to(extent),
        eq(shape(a.shaped("broadcast_in_dim", laid).binary("mul", x, multiplied))).to(extent),
        eq(length).to(log2_of(extent[axis])), *facts, *dominated(power + length, "new"), *more,
        *dominated(scaled, "scaled")).then(
        union(t).with_(a.binary("mul", sum_of, multiplied)), *made(sum_of, power + length),
        *made(a.binary("mul", sum_of, multiplied), scaled)),),
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
    # The lifted product contracts x's columns in the mode of the one it replaces (`mode_power`).
    contracted = (eq(left).to(shape(x)), eq(length2).to(log2_of(left[i64(1)]) + mode_power(params)))
    facts, power = product_magnitude(magnitude(x, "x"), magnitude(y, "y"), length2)
    product = x.dot(y, params)
    # The scaled product too, which rounds the product before its scale (distributivity's).
    more, scaled_power = product_magnitude(magnitude(s, "s"), ([], power), 0)
    lifted_by: tuple[list, list] = ([], [])
    for kept_axis in (1, 0):
        proof = () if kept_axis == 1 else (
            exact_int8_product(t), int8_values(x), int8_values(y), dyadic_scale(s))
        for lift_axis, laid_axis, lift in ((0, 1, row_lift), (1, 1 - kept_axis, column_lift)):
            side = s.shaped("broadcast_in_dim", laid)
            operand = x if lift_axis == 0 else y
            for scaled_operand in (operand.binary("mul", side, multiplied),
                                   side.binary("mul", operand, multiplied)):
                written = scaled_operand.dot(y, params) if lift_axis == 0 else x.dot(scaled_operand, params)
                lifted_by[lift_axis].append(rule(
                    eq(t).to(written), eq(matrix_product(params)).to(i64(kept_axis)), *proof,
                    plain(multiplied), matrix(x), matrix(y), along(laid, i64(laid_axis)),
                    eq(shape(scaled_operand)).to(shape(operand)),
                    eq(product_extent).to(shape(t)), eq(lift).to(lifted(product_extent, i64(lift_axis))),
                    *contracted, *facts, *dominated(power, "new"), *more,
                    *dominated(scaled_power, "scaled")).then(
                    union(t).with_(s.shaped("broadcast_in_dim", lift).binary("mul", product, multiplied)),
                    set_(shape(s.shaped("broadcast_in_dim", lift))).to(product_extent),
                    *made(product, power)))
    return (
        RuleFamily("a row scale out of a product", tuple(lifted_by[0]),
               (lambda x, s, y: jnp.dot(x * s[:, None], y.T), lambda x, s, y: jnp.dot(x, y.T) * s[:, None],
                _ROW_SCALED)),
        RuleFamily("a column scale out of a product", tuple(lifted_by[1]),
               (lambda x, s, y: jnp.dot(x, (y * s[:, None]).T), lambda x, s, y: jnp.dot(x, y.T) * s[None, :],
                _ROW_SCALED)))


def _joined() -> tuple[RuleFamily, ...]:
    """Join two counts of the same shape and axes when every partial sum in both orders is exact.

    An upper magnitude alone does not make ordinary floating-point sums safe to join. Adding their
    entries first can erase a contribution which would survive cancellation (row 13161)."""
    x, y, t = vars_("x y t", Term)
    axes = var("axes", Vec[i64])
    params, added = vars_("params added", i64)
    length = var("length", i64)
    power = i64(1)
    new = x.binary("add", y, added)
    reduced = new.reduce("sum", axes, params)
    left, right = x.reduce("sum", axes, params), y.reduce("sum", axes, params)
    return (RuleFamily("two counts along an axis", (rule(
        eq(t).to(left.binary("add", right, added)), eq(shape(x)).to(shape(y)), plain(added),
        counting_sum(left), counting_sum(right), eq(length).to(sum_length(left))).then(
        union(t).with_(reduced), *made(new, power), *made(reduced, power + length)),),
        (lambda x, y, a: jnp.sum((x > a).astype(x.dtype), -1) + jnp.sum((y > a).astype(y.dtype), -1),
         lambda x, y, a: jnp.sum((x > a).astype(x.dtype) + (y > a).astype(y.dtype), -1), _ROWS)),)


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
    """Exact identities within a window, not heads that open windows on their own."""
    x, y = vars_("x y", Term)
    params, wide, narrow = vars_("params wide narrow", i64)
    return (RuleFamily("addition's commutativity", (birewrite(x.binary("add", y, params)).to(
        y.binary("add", x, params)),), (lambda x, y, a: x + y, lambda x, y, a: y + x, _ROWS)),
            RuleFamily("multiplication's commutativity", (birewrite(x.binary("mul", y, params)).to(
                y.binary("mul", x, params)),), (lambda x, y, a: x * y, lambda x, y, a: y * x, _ROWS)),
            RuleFamily("two float32 negations", (rule(
                eq(y).to(x.unary("neg", params).unary("neg", params)),
                eq(dtype(x)).to(String("float32"))).then(union(y).with_(x)),),
                (lambda x, y, a: jax.lax.neg(jax.lax.neg(x)), lambda x, y, a: x, _ROWS)),
            RuleFamily("an exact widening round trip", (rule(
                eq(y).to(x.convert(wide).convert(narrow)), exact_roundtrip(y, x)).then(union(y).with_(x)),),
                (lambda x, y, a: x.astype(jnp.float32).astype(x.dtype), lambda x, y, a: x, _HALF_ROWS)))


FACTS = _facts()
RULES: dict[str, Rules] = {}
_HEAD_PATTERNS: dict[str, set[tuple[tuple[str, int], ...]]] = {}
"""What each registered rule's left side reads at its head (`_head`), indexed by root operation to required
operand-producer multisets (as sorted `(symbol, count)` tuples). The program pass opens a window at an
equation one of them matches (`heads`, `program.WINDOW` deep), and derives active root primitives directly
from its keys (`head_primitives`); a set that only aids the others (`Rules.aid`) or runs at dev time alone
(`Rules.dev`) opens none unless enabled via `enabling_dev`."""
_DEV_PATTERNS: dict[str, dict[str, set[tuple[tuple[str, int], ...]]]] = {}
"""Window head patterns for each dev-time rule set (`Rules.dev`), active only while named in
`ALGEBRA.value.dev`."""


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
        _HEAD_PATTERNS.setdefault(root, set()).add(tuple(sorted(collections.Counter(made).items())))
    if rules.dev:
        dev_patterns: dict[str, set[tuple[tuple[str, int], ...]]] = {}
        for family in rules.families:
            for written in family.rules:
                head = _head(written)
                if head is not None:
                    dev_patterns.setdefault(head[0], set()).add(
                        tuple(sorted(collections.Counter(head[1]).items())))
        _DEV_PATTERNS[rules.name] = dev_patterns


_MEASURED = ("measured M-DC-1 (four diffusion training steps, GH200) and M-DC-3 (six GPU flows' training and "
             "decode programs, a DPO step, GH200 and A100): within each cell's A/A")
"""The arms that measured the engine's registered sets (records/measurements.md), with each and without it."""
register(Rules.declared("identities", "exact", _identities(), f"docs/design/algebra.md §1; {_MEASURED}",
                        aid=True))
register(Rules.declared("contracted", "exact", _contracted(),
                        f"docs/design/algebra.md §6.5; the scaled softmax's maximum, ledger 3721; "
                        f"{_MEASURED}", aid=True))
register(Rules.declared("stable", "bounded", _stable(),
                        "docs/design/algebra.md §1.5; the log-sum-exp fold, feat/attention-derived "
                        "12f9398ba; a dev-time set, each form it proposes measured from the float64 truth "
                        "before it is taken (dew.algebra.error.choose)",
                        dev=True))
register(Rules.declared("linear", "reorders", _linear(),
                        "docs/design/algebra.md §1; ledger 1308; measured M-DC-3 (six GPU flows' training "
                        "and decode programs, GH200 and A100): within each cell's A/A"))
register(Rules.declared("reuse", "reorders", _reuse(),
                        "docs/design/algebra.md §1.4; the reuse probe of 2026-11-05 (benchmark_step and the "
                        "parallelism matrix: 42 to 452 of 423 to 794 divisions a suite share a divisor), "
                        "each reciprocal guarded by its divisor's floor and bound (row 9035). Dev time "
                        "until its receipt is measured (row 9777): of the nine perf-gate steps compiled "
                        "ahead for an H100, it changes finetune-4b's program alone (the survey of "
                        "2026-11-20, every set against it left out, A/A digests equal), whose timing with "
                        "and without it is M-DC-14's",
                        dev=True))
register(Rules.declared("joined", "reorders", _joined(),
                        "docs/design/algebra.md §1; ledger 1308; the rule-set survey of 2026-11-14: it "
                        "changes no shipped program, and the pass runs 12% slower without it (13,604 "
                        "passes)"))


def head_primitives() -> frozenset[str]:
    """Operation symbols where any active registered rule's left side heads (`_HEAD_PATTERNS`, enabled dev
    sets)."""
    enabled = ALGEBRA.value.dev
    if not enabled:
        return frozenset(_HEAD_PATTERNS)
    return frozenset(_HEAD_PATTERNS).union(*(_DEV_PATTERNS.get(name, ()) for name in enabled))


def _matches_head(patterns: set[tuple[tuple[str, int], ...]],
                  operands: Sequence[str | None]) -> bool:
    """Whether `operands` satisfies any pre-counted required producer multiset in `patterns`."""
    for needed in patterns:
        if not needed or all(operands.count(symbol) >= count for symbol, count in needed):
            return True
    return False


def heads(primitive: str, operands: Sequence[str | None]) -> bool:
    """Whether an equation binding `primitive` over operands made by `operands` (each producer's primitive, or
    None for a value from outside) is a head a registered rule may lower the price at: some rule's left side
    heads at that operation over operands of those makers (`_HEADS`, each rule's own as it is registered),
    each of the equation's operands answering one of the left side's at most, in any order (commutativity
    may have swapped them)."""
    if (patterns := _HEAD_PATTERNS.get(primitive)) and _matches_head(patterns, operands):
        return True
    enabled = ALGEBRA.value.dev
    if not enabled:
        return False
    return any(_matches_head(dev_patterns, operands)
               for name in enabled
               if (dev_patterns := _DEV_PATTERNS.get(name, {}).get(primitive)))


def reorderable(region: terms.Region) -> bool:
    """Whether a reordering rule may fire in `region`: each makes its new intermediate, and the value it
    gives, of operands the program states a number bound of (`dominated`, `made`: a sum's terms, a product's
    moved factor), so it needs a bounded term other than a literal. A literal's scale an extremum reduces
    moves out of it making no new magnitude (`positive_scale`), so a value scaled by a literal under a
    maximum or a minimum admits it too. A program's own math over its unbounded values (most
    of a training step's: a gemma3n step's 2879 regions each fired commutativity alone) admits none, and is
    left as traced. It reads the region's records alone, so a window's structure (`terms.Skeleton`) decides it
    before its terms are built."""
    literals = region.literal_keys
    if any(bound is not None and named not in literals for named, bound in region.magnitudes.items()):
        return True
    by_literal: set = set()
    for term, _ in region.shapes:
        found = terms.binary_of(term)
        if found is None or found[0] != "mul":
            continue
        _, left, right = found
        if left in literals or right in literals:
            by_literal.add(terms.key(term))
    extrema = (terms.reduce_of(term) for term, _ in region.shapes)
    return any(found is not None and found[0] in ("max", "min") and found[1] in by_literal
               for found in extrema)


@contextlib.contextmanager
def enabling_dev(*names: str) -> Iterator[None]:
    """Enable registered cost-qualified dev-time rule sets `names` (both their rules and their window heads)
    in JAX's trace context (`ALGEBRA.value.dev`) without mutating global registries; `bounded` sets such as
    `stable`, which require measured candidate selection (`dew.algebra.error.choose`), are refused."""
    invalid = [name for name in names
               if name not in RULES or not RULES[name].dev or RULES[name].rounding == "bounded"]
    if invalid:
        raise ValueError(f"not cost-qualified dev-time rule sets: {invalid}")
    with ALGEBRA(dataclasses.replace(ALGEBRA.value, dev=ALGEBRA.value.dev | frozenset(names))):
        yield


def admitted(accurate: bool, *, dev: bool = False) -> tuple[Rules, ...]:
    """The rule sets a program's numerics admit: every set under default numerics, the `exact` ones alone
    under the accurate numerics (`precision="accurate"`); the dev-time sets (`Rules.dev`) only where `dev`
    or named in `enabling_dev` (`ALGEBRA.value.dev`); none the pass leaves out (`dew._cache.Passes`'
    `without`, a receipt's arm)."""
    enabled = ALGEBRA.value.dev
    return tuple(rules for rules in RULES.values()
                 if (dev or not rules.dev or rules.name in enabled)
                 and (dev or not accurate or rules.rounding == "exact")
                 and admits(rules))


ARMS: dict[str, Rules] = {}
"""Each derivation's own set a receipt's arm may leave out (`program.passes`' `without`) beside the registered
sets, by name: one whose derivation still makes a program without it (`carry.CARRIED`: the split's traced
forms). A derivation's defining set (`derive.DECOUPLED`, a scan's own identities, `linear_scan`'s, which no
`Rules` holds) makes its program, which no arm has without it, and a dev-time set runs at dev time alone:
neither is an arm's."""


def armed(sets: Rules) -> Rules:
    """`sets`, a derivation's own, nameable by a receipt's arm (`ARMS`)."""
    ARMS[sets.name] = sets
    return sets


def nameable() -> frozenset[str]:
    """The sets a receipt's arm may leave out: every registered set but a dev-time one, and each `ARMS`, a
    derivation's set its own module arms as it loads (`dew.algebra.carry`'s): loaded first, so the names are
    every one of them whatever this process has read of the algebra (its package loads each module where it
    is first read)."""
    importlib.import_module("dew.algebra.carry")
    return frozenset(name for name, sets in RULES.items() if not sets.dev) | frozenset(ARMS)


def admits(sets: Rules) -> bool:
    """Whether the pass runs `sets` (none an arm leaves out, `dew._cache.Passes`' `without`)."""
    return sets.name not in ALGEBRA.value.without


def admitted_names(accurate: bool) -> tuple[str, ...]:
    """The admitted sets by name, in their order: what a derivation kept keys on beside its program, so a set
    registered later derives afresh (the program pass's windows, a reduction's forms)."""
    return tuple(sets.name for sets in admitted(accurate))


def firing(sets: tuple[Rules, ...], region: terms.Region) -> tuple[Rules, ...]:
    """The sets of `sets` that may change `region`: every one where a reordering rule may fire
    (`reorderable`); else the `exact` ones, whose rewrites read no bound, and none where those are saturation
    aids alone, which change no program by themselves (`Rules.aid`)."""
    if reorderable(region):
        return sets
    exact = tuple(held for held in sets if held.rounding == "exact")
    return () if all(held.aid for held in exact) else exact


def instances() -> Mapping[str, Instance]:
    """Every registered rule written as JAX, by its set's name and its own (`Rules.instances`): the
    qualification inventory its accuracy test holds each right side to its left side over."""
    return {f"{rules.name}: {name}": instance for rules in RULES.values()
            for name, instance in rules.instances.items()}
