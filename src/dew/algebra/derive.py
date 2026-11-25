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
import itertools
import math
import time
from collections.abc import Callable, Mapping, Sequence
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
    get_callable_args,
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

from dew import _cache
from dew._cache import ALGEBRA, unintercepted
from dew.algebra import cost, extract, program, rules, terms
from dew.algebra.cost import Work
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
    exponent: Literal[2, "e"]
    ranges: Mapping[str, float] = dataclasses.field(default_factory=dict)
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


@function
def unit_array(term: Term) -> Unit: ...
"""A value of ones (a literal 1, or a layout or cast of one)."""


@function
def boolean(term: Term) -> Unit: ...
"""A boolean value (a mask)."""


@function
def is_square(params: i64) -> Unit: ...
"""An `integer_pow` of exponent 2."""


@function
def inserts_axis(params: i64, axis: i64) -> Unit: ...
"""A broadcast or reshape inserting `axis` while keeping every other axis in order."""


@function
def identity_broadcast(params: i64) -> Unit: ...
"""A `broadcast_in_dim` keeping every axis in place (`broadcast_dimensions == range(rank)`)."""


@function(merge=lambda old, new: old)
def zero_of(term: Term) -> i64: ...
"""The literal index of 0 in `term`'s floating dtype."""


@function(merge=lambda old, new: old)
def one_of(term: Term) -> i64: ...
"""The literal index of 1 in `term`'s floating dtype."""


@function(merge=lambda old, new: old)
def filled(to: Vec[i64]) -> i64: ...
"""The `broadcast_in_dim` parameters lifting a scalar to shape `to`."""


@function(merge=lambda old, new: old)
def unreduced(to: Vec[i64], axis: i64) -> i64: ...
"""The `broadcast_in_dim` parameters lifting a shape without `axis` back to `to`."""


@function(merge=lambda old, new: old)
def comp_gt() -> i64: ...
"""The parameters index of `lax.gt_p`."""


@function(merge=lambda old, new: old)
def logic_and() -> i64: ...
"""The parameters index of `lax.and_p`."""


@function
def indicator(term: Term, mask: Term) -> Unit: ...
"""A 0/1 indicator of boolean `mask` (`mask.astype(dtype)` or `where(mask, 1, 0)`)."""


@function
def counts(total: Term, weight: Term, axis: i64) -> Unit: ...
"""`total` is the walked sum of local `weight` along `axis`."""


@function
def masked_counts(total: Term, mask: Term, axis: i64) -> Unit: ...
"""`total` is the walked sum of `mask`'s 0/1 indicator along `axis`."""


@function
def first_sum(total: Term, value: Term, weight: Term, axis: i64) -> Unit: ...
"""`total` is the walked sum of `weight * value` along `axis`."""


@function
def masked_first_sum(total: Term, value: Term, mask: Term, axis: i64) -> Unit: ...
"""`total` is the walked sum of `where(mask, value, 0)` along `axis`."""


@function
def safe_count(term: Term, count: Term) -> Unit: ...
"""`term` equals `count` wherever `count > 0` (`count` itself, or guarded against zero)."""


@function
def count_lifted(term: Term, count: Term, axis: i64) -> Unit: ...
"""`term` is `count` lifted across `axis` (`keepdims`)."""


@function
def safe_count_lifted(term: Term, count: Term, axis: i64) -> Unit: ...
"""`term` is a zero-safe `count` lifted across `axis` (`keepdims`)."""


@function
def first_lifted(term: Term, total: Term, value: Term, weight: Term, axis: i64) -> Unit: ...
"""`term` is `first_sum` `total` lifted across `axis` (`keepdims`)."""


@function
def masked_first_lifted(term: Term, total: Term, value: Term, mask: Term, axis: i64) -> Unit: ...
"""`term` is `masked_first_sum` `total` lifted across `axis` (`keepdims`)."""


@function
def mean_of(mu: Term, total: Term, count: Term, value: Term, weight: Term, axis: i64) -> Unit: ...
"""`mu` is the weighted mean `total / count` of `value` under `weight` along `axis`."""


@function
def masked_mean_of(mu: Term, total: Term, count: Term, value: Term, mask: Term, axis: i64) -> Unit: ...
"""`mu` is the masked mean `total / count` of `value` under `mask` along `axis`."""


@function
def mean_lifted(mu: Term, total: Term, count: Term, value: Term, weight: Term, axis: i64) -> Unit: ...
"""`mu` is `mean_of` broadcast along `axis` keeping all other axes in order."""


@function
def masked_mean_lifted(mu: Term, total: Term, count: Term, value: Term, mask: Term, axis: i64) -> Unit: ...
"""`mu` is `masked_mean_of` broadcast along `axis` keeping all other axes in order."""


@function
def centered(diff: Term, value: Term, total: Term, count: Term, weight: Term, axis: i64, sign: i64) -> Unit: ...
"""`diff` is `sign * (value - mean)` along `axis` under `weight`."""


@function
def masked_centered(diff: Term, value: Term, total: Term, count: Term, mask: Term, axis: i64,
                    sign: i64) -> Unit: ...
"""`diff` is `sign * (value - mean)` along `axis` under `mask`."""


@function
def centered_pair(term: Term, sx: Term, sy: Term, count: Term, x: Term, y: Term, weight: Term, axis: i64,
                  sign: i64) -> Unit: ...
"""`term` is `sign * (x - mu_x) * (y - mu_y)` along `axis` under `weight`."""


@function
def masked_centered_pair(term: Term, sx: Term, sy: Term, count: Term, x: Term, y: Term, mask: Term, axis: i64,
                         sign: i64) -> Unit: ...
"""`term` is `sign * (x - mu_x) * (y - mu_y)` along `axis` under `mask`."""


@function
def weighted_central(term: Term, sx: Term, sy: Term, count: Term, x: Term, y: Term, weight: Term, axis: i64,
                     sign: i64) -> Unit: ...
"""`term` is `sign * weight * (x - mu_x) * (y - mu_y)` along `axis`."""


@function
def masked_weighted_central(term: Term, sx: Term, sy: Term, count: Term, x: Term, y: Term, mask: Term,
                            z: Term, axis: i64, sign: i64) -> Unit: ...
"""`term` is `where(mask, sign * (x - mu_x) * (y - mu_y), z)` along `axis` (`z` zero)."""


@function
def central_moment(term: Term, sx: Term, sy: Term, count: Term, x: Term, y: Term, weight: Term, axis: i64,
                   sign: i64) -> Unit: ...
"""`term` is the walked central second moment `sum(weight * (x - mu_x) * (y - mu_y))` along `axis`."""


@function
def masked_central_moment(term: Term, sx: Term, sy: Term, count: Term, x: Term, y: Term, mask: Term,
                          z: Term, axis: i64, sign: i64) -> Unit: ...
"""`term` is the walked masked central second moment `sum(where(mask, (x - mu_x) * (y - mu_y), z))`."""


def _moment_walking() -> list:
    """The facts' rules recognizing weighted and masked central second moments (variance and covariance)
    along the walked axis, for Chan-Golub-LeVeque decoupling (`_moment_decoupled`)."""
    x, y, z, w, mask, ind, t, n, s, sx, sy, d, c, mu, dx, dy, pair, m = vars_(
        "x y z w mask ind t n s sx sy d c mu dx dy pair m", Term)
    layout = var("layout", String)
    p, q, r, a, sign = vars_("p q r a sign", i64)
    axes = var("axes", Vec[i64])
    shape = rules.shape
    found = [
        # Ones and booleans through layouts and casts; 0/1 mask indicators.
        rule(eq(t).to(x.shaped(layout, p)), unit_array(x)).then(unit_array(t)),
        rule(eq(t).to(x.convert(p)), unit_array(x)).then(unit_array(t)),
        rule(eq(t).to(x.shaped(layout, p)), boolean(x)).then(boolean(t)),
        rule(eq(t).to(mask.convert(p)), boolean(mask)).then(indicator(t, mask)),
        rule(eq(t).to(mask.select(z, x)), boolean(mask), zero(z), unit_array(x)).then(indicator(t, mask)),
        # Counts and first sums along the walked axis.
        rule(eq(n).to(w.reduce(String("sum"), axes, p)), local(w), eq(walked(w)).to(a),
             eq(axes).to(Vec[i64](a))).then(counts(n, w, a), safe_count(n, n)),
        rule(eq(n).to(ind.reduce(String("sum"), axes, p)), indicator(ind, mask), local(mask),
             eq(walked(mask)).to(a), eq(axes).to(Vec[i64](a))).then(masked_counts(n, mask, a),
                                                                    safe_count(n, n)),
        rule(eq(s).to(w.binary(String("mul"), x, q).reduce(String("sum"), axes, p)), local(x), local(w),
             eq(walked(x)).to(a), eq(walked(w)).to(a), eq(axes).to(Vec[i64](a))).then(first_sum(s, x, w, a)),
        rule(eq(s).to(x.binary(String("mul"), w, q).reduce(String("sum"), axes, p)), local(x), local(w),
             eq(walked(x)).to(a), eq(walked(w)).to(a), eq(axes).to(Vec[i64](a))).then(first_sum(s, x, w, a)),
        rule(eq(s).to(mask.select(z, x).reduce(String("sum"), axes, p)), boolean(mask), zero(z), local(x),
             local(mask), eq(walked(x)).to(a), eq(walked(mask)).to(a),
             eq(axes).to(Vec[i64](a))).then(masked_first_sum(s, x, mask, a)),
        # Zero-safe count denominators (`where(n > 0, n, c)`, `where(0 < n, n, c)`, `where(n != 0, n, c)`,
        # `where(n == 0, c, n)`).
        rule(eq(d).to(n.compare(String("gt"), z, p).select(c, n)), safe_count(n, n), zero(z)).then(
            safe_count(d, n)),
        rule(eq(d).to(z.compare(String("lt"), n, p).select(c, n)), safe_count(n, n), zero(z)).then(
            safe_count(d, n)),
        rule(eq(d).to(n.compare(String("ne"), z, p).select(c, n)), safe_count(n, n), zero(z)).then(
            safe_count(d, n)),
        rule(eq(d).to(n.compare(String("eq"), z, p).select(n, c)), safe_count(n, n), zero(z)).then(
            safe_count(d, n)),
        rule(eq(d).to(z.compare(String("eq"), n, p).select(n, c)), safe_count(n, n), zero(z)).then(
            safe_count(d, n)),
        # Keepdims lifts (inserting the walked axis at its exact place before dividing).
        rule(eq(t).to(n.shaped(layout, p)), counts(n, w, a), inserts_axis(p, a)).then(
            count_lifted(t, n, a), safe_count(t, t)),
        rule(eq(t).to(n.shaped(layout, p)), masked_counts(n, mask, a), inserts_axis(p, a)).then(
            count_lifted(t, n, a), safe_count(t, t)),
        rule(eq(t).to(d.shaped(layout, p)), safe_count(d, n), inserts_axis(p, a)).then(
            safe_count_lifted(t, n, a)),
        rule(count_lifted(c, n, a), safe_count(d, c)).then(safe_count_lifted(d, n, a)),
        rule(eq(t).to(s.shaped(layout, p)), first_sum(s, x, w, a), inserts_axis(p, a)).then(
            first_lifted(t, s, x, w, a)),
        rule(eq(t).to(s.shaped(layout, p)), masked_first_sum(s, x, mask, a), inserts_axis(p, a)).then(
            masked_first_lifted(t, s, x, mask, a)),
        # Weighted and masked means, plus axis-aligned broadcasts back to `x`'s shape.
        rule(eq(mu).to(s.binary(String("div"), d, p)), first_sum(s, x, w, a), counts(n, w, a),
             safe_count(d, n), eq(shape(s)).to(shape(n))).then(mean_of(mu, s, n, x, w, a)),
        rule(eq(mu).to(s.binary(String("div"), d, p)), masked_first_sum(s, x, mask, a),
             masked_counts(n, mask, a), safe_count(d, n),
             eq(shape(s)).to(shape(n))).then(masked_mean_of(mu, s, n, x, mask, a)),
        rule(eq(t).to(mu.shaped(layout, p)), mean_of(mu, s, n, x, w, a), inserts_axis(p, a)).then(
            mean_lifted(t, s, n, x, w, a)),
        rule(eq(t).to(mu.shaped(layout, p)), masked_mean_of(mu, s, n, x, mask, a), inserts_axis(p, a)).then(
            masked_mean_lifted(t, s, n, x, mask, a)),
        rule(eq(mu).to(c.binary(String("div"), d, p)), first_lifted(c, s, x, w, a),
             safe_count_lifted(d, n, a), counts(n, w, a),
             eq(shape(s)).to(shape(n))).then(mean_lifted(mu, s, n, x, w, a)),
        rule(eq(mu).to(c.binary(String("div"), d, p)), masked_first_lifted(c, s, x, mask, a),
             safe_count_lifted(d, n, a), masked_counts(n, mask, a),
             eq(shape(s)).to(shape(n))).then(masked_mean_lifted(mu, s, n, x, mask, a)),
        rule(eq(t).to(mu.shaped(String("broadcast_in_dim"), p)), mean_lifted(mu, s, n, x, w, a),
             identity_broadcast(p), eq(shape(t)).to(shape(x))).then(mean_lifted(t, s, n, x, w, a)),
        rule(eq(t).to(mu.shaped(String("broadcast_in_dim"), p)), masked_mean_lifted(mu, s, n, x, mask, a),
             identity_broadcast(p), eq(shape(t)).to(shape(x))).then(masked_mean_lifted(t, s, n, x, mask, a)),
        # Centered differences (both `x - mu` and `mu - x`).
        rule(eq(dx).to(x.binary(String("sub"), mu, p)), mean_lifted(mu, s, n, x, w, a),
             eq(shape(x)).to(shape(mu))).then(centered(dx, x, s, n, w, a, i64(1))),
        rule(eq(dx).to(mu.binary(String("sub"), x, p)), mean_lifted(mu, s, n, x, w, a),
             eq(shape(x)).to(shape(mu))).then(centered(dx, x, s, n, w, a, i64(-1))),
        rule(eq(dx).to(x.binary(String("sub"), mu, p)), masked_mean_lifted(mu, s, n, x, mask, a),
             eq(shape(x)).to(shape(mu))).then(masked_centered(dx, x, s, n, mask, a, i64(1))),
        rule(eq(dx).to(mu.binary(String("sub"), x, p)), masked_mean_lifted(mu, s, n, x, mask, a),
             eq(shape(x)).to(shape(mu))).then(masked_centered(dx, x, s, n, mask, a, i64(-1))),
        # Centered squares (`square(dx)` and `dx ** 2`).
        rule(eq(pair).to(dx.unary(String("square"), p)), centered(dx, x, sx, n, w, a, sign)).then(
            centered_pair(pair, sx, sx, n, x, x, w, a, i64(1))),
        rule(eq(pair).to(dx.unary(String("integer_pow"), p)), is_square(p),
             centered(dx, x, sx, n, w, a, sign)).then(centered_pair(pair, sx, sx, n, x, x, w, a, i64(1))),
        rule(eq(pair).to(dx.unary(String("square"), p)), masked_centered(dx, x, sx, n, mask, a, sign)).then(
            masked_centered_pair(pair, sx, sx, n, x, x, mask, a, i64(1))),
        rule(eq(pair).to(dx.unary(String("integer_pow"), p)), is_square(p),
             masked_centered(dx, x, sx, n, mask, a, sign)).then(
            masked_centered_pair(pair, sx, sx, n, x, x, mask, a, i64(1))),
    ]
    # Guarded zero-means (`where(n > 0, mu, 0)` and `where(n == 0, 0, mu)`).
    for mean_fact, ref in ((mean_of, w), (masked_mean_of, mask)):
        for guard in (lambda: n.compare(String("gt"), z, p).select(c, mu),
                      lambda: z.compare(String("lt"), n, p).select(c, mu),
                      lambda: n.compare(String("ne"), z, p).select(c, mu),
                      lambda: n.compare(String("eq"), z, p).select(mu, c),
                      lambda: z.compare(String("eq"), n, p).select(mu, c)):
            found.append(rule(eq(t).to(guard()), mean_fact(mu, s, n, x, ref, a), zero(z), zero(c)).then(
                mean_fact(t, s, n, x, ref, a)))
    # Centered products `dx * dy` and every association of `w * dx * dy`.
    for s1 in (1, -1):
        for s2 in (1, -1):
            prod_sign = i64(s1 * s2)
            found += [
                rule(eq(pair).to(dx.binary(String("mul"), dy, p)),
                     centered(dx, x, sx, n, w, a, i64(s1)), centered(dy, y, sy, n, w, a, i64(s2))).then(
                    centered_pair(pair, sx, sy, n, x, y, w, a, prod_sign)),
                rule(eq(pair).to(dx.binary(String("mul"), dy, p)),
                     masked_centered(dx, x, sx, n, mask, a, i64(s1)),
                     masked_centered(dy, y, sy, n, mask, a, i64(s2))).then(
                    masked_centered_pair(pair, sx, sy, n, x, y, mask, a, prod_sign)),
                rule(eq(t).to(w.binary(String("mul"), dx, q).binary(String("mul"), dy, p)),
                     centered(dx, x, sx, n, w, a, i64(s1)), centered(dy, y, sy, n, w, a, i64(s2))).then(
                    weighted_central(t, sx, sy, n, x, y, w, a, prod_sign)),
                rule(eq(t).to(dx.binary(String("mul"), w, q).binary(String("mul"), dy, p)),
                     centered(dx, x, sx, n, w, a, i64(s1)), centered(dy, y, sy, n, w, a, i64(s2))).then(
                    weighted_central(t, sx, sy, n, x, y, w, a, prod_sign)),
                rule(eq(t).to(dx.binary(String("mul"), w.binary(String("mul"), dy, q), p)),
                     centered(dx, x, sx, n, w, a, i64(s1)), centered(dy, y, sy, n, w, a, i64(s2))).then(
                    weighted_central(t, sx, sy, n, x, y, w, a, prod_sign)),
                rule(eq(t).to(dx.binary(String("mul"), dy.binary(String("mul"), w, q), p)),
                     centered(dx, x, sx, n, w, a, i64(s1)), centered(dy, y, sy, n, w, a, i64(s2))).then(
                    weighted_central(t, sx, sy, n, x, y, w, a, prod_sign)),
            ]
    found += [
        rule(eq(t).to(w.binary(String("mul"), pair, p)),
             centered_pair(pair, sx, sy, n, x, y, w, a, sign)).then(
            weighted_central(t, sx, sy, n, x, y, w, a, sign)),
        rule(eq(t).to(pair.binary(String("mul"), w, p)),
             centered_pair(pair, sx, sy, n, x, y, w, a, sign)).then(
            weighted_central(t, sx, sy, n, x, y, w, a, sign)),
        rule(eq(t).to(mask.select(z, pair)), boolean(mask), zero(z),
             masked_centered_pair(pair, sx, sy, n, x, y, mask, a, sign)).then(
            masked_weighted_central(t, sx, sy, n, x, y, mask, z, a, sign)),
        rule(eq(m).to(t.reduce(String("sum"), axes, p)), eq(axes).to(Vec[i64](a)),
             weighted_central(t, sx, sy, n, x, y, w, a, sign)).then(
            summed(m), central_moment(m, sx, sy, n, x, y, w, a, sign)),
        rule(eq(m).to(t.reduce(String("sum"), axes, p)), eq(axes).to(Vec[i64](a)),
             masked_weighted_central(t, sx, sy, n, x, y, mask, z, a, sign)).then(
            summed(m), masked_central_moment(m, sx, sy, n, x, y, mask, z, a, sign)),
    ]
    return found


def _moment_decoupled() -> list:
    """Chan-Golub-LeVeque step and merge rules for weighted and masked central second moments
    (`central_moment`, `masked_central_moment`):
    `M_AB = M_A + M_B + where((n_A > 0) & (n_B > 0), d_x * d_y * (n_A / (n_A + n_B)) * n_B, 0)`
    with zero-safe denominators so empty states and wholly-masked tiles introduce no `0 / 0` NaNs."""
    t, u, sx, sy, n, x, y, w, mask, z = vars_("t u sx sy n x y w mask z", Term)
    axes = var("axes", Vec[i64])
    to_x, to_y, to_n = vars_("to_x to_y to_n", Vec[i64])
    p, a, i, i_n, i_sx, i_sy = vars_("p a i i_n i_sx i_sy", i64)
    added, subtracted, multiplied, divided = vars_("added subtracted multiplied divided", i64)
    gt_idx, and_idx, z_n, o_n = vars_("gt_idx and_idx z_n o_n", i64)
    fill_n, lift_x, lift_y = vars_("fill_n lift_x lift_y", i64)
    stock = rules.stock
    shape = rules.shape

    def combine(m0, n0, sx0, sy0, m1, n1, sx1, sy1, sign_val: int):
        zero_n = Term.literal(z_n).shaped(String("broadcast_in_dim"), fill_n)
        one_n = Term.literal(o_n).shaped(String("broadcast_in_dim"), fill_n)
        pos0 = n0.compare(String("gt"), zero_n, gt_idx)
        pos1 = n1.compare(String("gt"), zero_n, gt_idx)
        both = pos0.binary(String("and"), pos1, and_idx)
        safe0 = pos0.select(one_n, n0)
        safe1 = pos1.select(one_n, n1)
        total = n0.binary(String("add"), n1, added)
        safe_total = both.select(one_n, total)
        mu_x0 = sx0.binary(String("div"), safe0, divided)
        mu_x1 = sx1.binary(String("div"), safe1, divided)
        mu_y0 = sy0.binary(String("div"), safe0, divided)
        mu_y1 = sy1.binary(String("div"), safe1, divided)
        dx = mu_x1.binary(String("sub"), mu_x0, subtracted)
        dy = mu_y1.binary(String("sub"), mu_y0, subtracted)
        weight = n0.binary(String("div"), safe_total, divided).binary(String("mul"), n1, multiplied)
        corr = both.select(zero_n, dx.binary(String("mul"), dy, multiplied).binary(String("mul"), weight,
                                                                                   multiplied))
        summed_m = m0.binary(String("add"), m1, added)
        return (summed_m.binary(String("add"), corr, added) if sign_val == 1
                else summed_m.binary(String("sub"), corr, subtracted))

    def tile_pair(sign_val: int):
        zero_n = Term.literal(z_n).shaped(String("broadcast_in_dim"), fill_n)
        one_n = Term.literal(o_n).shaped(String("broadcast_in_dim"), fill_n)
        safe_tile = n.compare(String("gt"), zero_n, gt_idx).select(one_n, n)
        mu_x_tile = sx.binary(String("div"), safe_tile, divided).shaped(String("broadcast_in_dim"), lift_x)
        mu_y_tile = sy.binary(String("div"), safe_tile, divided).shaped(String("broadcast_in_dim"), lift_y)
        dx_tile = x.binary(String("sub"), mu_x_tile, subtracted)
        dy_tile = (y.binary(String("sub"), mu_y_tile, subtracted) if sign_val == 1
                   else mu_y_tile.binary(String("sub"), y, subtracted))
        return dx_tile.binary(String("mul"), dy_tile, multiplied)

    common = (
        eq(t).to(u.reduce(String("sum"), axes, p)),
        eq(i).to(slot(t)), eq(i_n).to(slot(n)), eq(i_sx).to(slot(sx)), eq(i_sy).to(slot(sy)),
        eq(added).to(stock("add")), eq(subtracted).to(stock("sub")),
        eq(multiplied).to(stock("mul")), eq(divided).to(stock("div")),
        eq(gt_idx).to(comp_gt()), eq(and_idx).to(logic_and()),
        eq(z_n).to(zero_of(n)), eq(o_n).to(one_of(n)),
        eq(z_n).to(zero_of(sx)), eq(z_n).to(zero_of(sy)), eq(z_n).to(zero_of(t)),
        eq(to_n).to(shape(n)), eq(shape(sx)).to(to_n), eq(shape(sy)).to(to_n), eq(shape(t)).to(to_n),
        eq(fill_n).to(filled(to_n)),
        eq(to_x).to(shape(x)), eq(to_y).to(shape(y)),
        eq(lift_x).to(unreduced(to_x, a)), eq(lift_y).to(unreduced(to_y, a)),
    )
    made = []
    for sign_val in (1, -1):
        w_tile = w.binary(String("mul"), tile_pair(sign_val), multiplied).reduce(String("sum"), axes, p)
        m_tile = mask.select(z, tile_pair(sign_val)).reduce(String("sum"), axes, p)
        w_cond = (central_moment(t, sx, sy, n, x, y, w, a, i64(sign_val)), *common)
        m_cond = (masked_central_moment(t, sx, sy, n, x, y, mask, z, a, i64(sign_val)), *common)
        for cond, tile_m in ((w_cond, w_tile), (m_cond, m_tile)):
            made += [
                rewrite(t.stepped()).to(
                    combine(Term.state(i, 0), Term.state(i_n, 0), Term.state(i_sx, 0), Term.state(i_sy, 0),
                            tile_m, n, sx, sy, sign_val), *cond),
                rewrite(t.merged()).to(
                    combine(Term.state(i, 0), Term.state(i_n, 0), Term.state(i_sx, 0), Term.state(i_sy, 0),
                            Term.state(i, 1), Term.state(i_n, 1), Term.state(i_sx, 1), Term.state(i_sy, 1),
                            sign_val), *cond),
            ]
    return made


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
    for power in ("exp", "exp2"):
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
    ] + _moment_walking()
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
    derived.append(rule(eq(stepped).to(t.stepped()), eq(t).to(x.dot(y, p)), summed(t), local_of(t),
                        slotted).then(set_(weighed(stepped)).to(x.stepped()),
                                      set_(weighted(stepped)).to(y.stepped())))
    for combined, which, index in (("max", peaked, largest), ("add", summed, added)):
        stocked = eq(index).to(stock(combined))
        derived += [
            rewrite(t.stepped()).to(Term.state(i, 0).binary(combined, t, index), which(t), local_of(t),
                                    stocked, slotted),
            rewrite(t.merged()).to(Term.state(i, 0).binary(combined, Term.state(i, 1), index), which(t),
                                   local_of(t), stocked, slotted),
            rewrite(t.finished()).to(Term.state(i, 0), which(t), slotted),
        ]
    derived += [
        rewrite(t.held(side)).to(Term.state(i, side), peaked(t), slotted),
        rewrite(t.held(side)).to(Term.state(i, side), summed(t), local_of(t), slotted),
    ]
    derived += _moment_decoupled()

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
        product = get_callable_args(tile, Term.dot)
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


EXPONENTIALS = {2: "exp2", "e": "exp"}
"""The language's exponential of each base a declaration states (`Reduction.exponent`)."""


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


_READINGS = (Term.stepped, Term.merged, Term.finished, Term.held)


def _priced(region: terms.Region, schedule) -> DagCostModel[cost.Seconds]:
    """The template's tile walk's price of each node (`cost.model` under `cost.walk`: a form the template's
    algebra takes runs between its mainloop's products, not as XLA fuses it), each of the derivation's
    readings infinite in every schedule (`cost.Seconds.extra`: a form that keeps one is no form, whatever
    rates the target states)."""
    timed = cost.model(region, schedule)

    def marginal(graph: EGraph, node) -> cost.Seconds:
        if any(get_callable_args(node, reading) is not None for reading in _READINGS):
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


def context() -> tuple:
    """What a derivation reads where it runs beside its declaration and shapes, the key of every derivation a
    declaration holds (`derive`'s forms, beside the rule sets its numerics admit, `rules.admitted_names`;
    `scan.derive_scan`'s carry splits): forms traced under one abstract
    mesh hold its avals in their literals (a sharded step's bound under a local one's mesh refused to lower),
    and are priced for one target (`cost.target`), under one run's numerics and one matmul precision, with
    the extractor oracle on or off (`Passes.checked`), by every registered rule set but those the pass leaves
    out (`Passes.without`, a receipt's arm, which shares no derivation or recognition with another arm). Not
    the whole trace context, whose identity-keyed collections (`dew._cache.KERNEL_ROUTES`, `KERNEL_SEEN`) are
    new in each lowering: every lowering derived again and the cache grew without end."""
    return (jax.sharding.get_abstract_mesh(), cost.target(), generation.numerics(),
            jax.config.jax_default_matmul_precision, ALGEBRA.value.checked, ALGEBRA.value.without)


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

    def derived_at(keyed: tuple) -> _Forms:
        shapes = {name: (shape, dtype) for name, shape, dtype in keyed}
        made = _derived(declaration, names, walked_axes, shapes, accurate, saturation)
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
        typed = jax.eval_shape(lambda *values: tuple(evaluated(made.region, reduction,
                                                               [*made.consts, *values], {})
                                                     for reduction in made.reductions), *avals)
        return tuple(jnp.full(aval.shape, _identity(reduction, aval.dtype) if floor is None else floor,
                              aval.dtype)
                     for reduction, aval, floor in zip(made.reductions, typed, made.floors, strict=True))

    def carried(made: _Forms, *states) -> dict:
        return {terms.key(Term.state(i64(index), i64(side))): state[index]
                for side, state in enumerate(states)
                for index in range(len(made.reductions))}

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
        taken = (None,) * len(made.steps) if stepped is None else tuple(stepped(tuple(state), parts))
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
        return tuple((_role(reduction), made.region.shape(reduction)) for reduction in made.reductions)

    return Derived(init, step, merge, finish, mapped,
                   {"rounding": "reorders", "walked": declaration.walked, "exponent": declaration.exponent,
                    "rules": ("walking", "decoupled")}, slots=slots)


def _role(reduction: Term) -> str:
    """A walked reduction's role in the carried state, as the declaration's reduction is made: a maximum's
    `max`, a sum's `sum`, a product's contracting the walked axis `contraction` (each the constructor that
    made it, `terms.reduce_of`)."""
    if get_callable_args(reduction, Term.dot) is not None:
        return "contraction"
    found = terms.reduce_of(reduction)
    if found is None or found[0] not in ("max", "sum"):
        raise Refused(f"{reduction} is no walked maximum, sum or contraction")
    return found[0]


def _derived(declaration: Reduction, names: tuple[str, ...], walked_axes: Mapping[str, int],
             shapes: Mapping, accurate: bool, saturation: program.Saturation) -> _Forms:
    """The forms at one tile's `shapes` (the module's account)."""
    began = time.perf_counter()
    def named(*values):
        return declaration.function(**dict(zip(names, values, strict=True)))

    avals = [jax.ShapeDtypeStruct(*shapes[name]) for name in names]
    # The declaration is the algebra's statement, no model's products: traced inside a quantized model's
    # method, a provider's interception would have quantized it, a form no rule writes (row 4858).
    with jax.disable_jit():
        closed = jax.make_jaxpr(unintercepted(inlined(named)))(*avals)
    # The ranges by input, as the region reads them (its constants apart).
    region = terms.region(closed, {index: declaration.ranges[name] for index, name in enumerate(names)
                                   if name in declaration.ranges})
    graph = EGraph()
    _register(graph, region, closed, names, walked_axes)
    walking = WALKING if declaration.rescale is None else WALKING_SCALED
    graph.run(walking.saturate())
    for term in _opaque_locals(graph, region, walking):
        graph.register(local(term))
    graph.run(walking.saturate())
    graph.run(PLAIN.saturate())
    found = [term for term, _ in region.shapes if _holds(graph, summed(term)) or _holds(graph, peaked(term))]
    reductions = tuple({terms.key(term): term for term in found}.values())
    for index, reduction in enumerate(reductions):
        graph.register(set_(slot(reduction)).to(i64(index)))
        region.params.append((jax.lax.convert_element_type_p, {
            "new_dtype": region.dtypes[terms.key(reduction)], "weak_type": False, "sharding": None}))
        graph.register(set_(retype(i64(index))).to(i64(len(region.params) - 1)))
        for side in (0, 1):
            graph.register(set_(rules.shape(Term.state(i64(index), i64(side)))).to(
                Vec[i64](*(i64(size) for size in region.shape(reduction)))))
    _lift(graph, region, reductions)
    _moment_layouts(graph, region, reductions)
    outputs = list(region.outputs)
    along = [term is not None and _walked(graph, term) for term in outputs]
    for term, is_along in zip(outputs, along, strict=True):
        if is_along and term is not None and not _holds(graph, local(term)):
            raise Refused("an output along the walked axis reads a walked reduction: no tile holds it")
    roots = ([reduction.stepped() for reduction in reductions]
             + [reduction.merged() for reduction in reductions]
             + [term.finished() for term, is_along in zip(outputs, along, strict=True)
                if term is not None and not is_along])
    declared = Term.bundle(Vec[Term](*region.roots()))
    bundle = Term.bundle(Vec[Term](*roots) if roots else Vec[Term].empty())
    graph.register(declared, bundle)
    schedule = cost.walk(cost.target())
    _, before = extract.cheapest(graph, declared, cost.model(region, schedule))
    floors = _floors(graph, region, reductions)
    facts = seq(rules.FACTS.saturate(), SHAPES.saturate(), walking.saturate(), FLOORED.saturate())
    decoupled = DECOUPLED if declaration.rescale is None else DECOUPLED_RESCALED
    _decoupled_through(graph, region, facts, decoupled)
    saturated = program.saturate(graph, region, (*rules.admitted(accurate), decoupled),
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
                  walked_leaves, tuple(closed.consts), report, tuple(accumulated), tuple(added),
                  tuple(merged), floors)


def _decoupled_through(graph: EGraph, region: terms.Region, facts: Schedule, decoupled: rules.Rules) -> None:
    """The derivation's readings run through `region`'s terms alone first (`DECOUPLED`, its facts saturated
    after each pass): each reading meets the language at the region's depth, before the engine's sets grow
    the graph toward their limits, so a limit the engine reaches never stops a form."""
    for _ in range(4 * max(region.depth, 1) + 8):
        graph.run(facts)
        if not graph.run(decoupled.ruleset).updated:
            break
    graph.run(facts)


def _register(graph: EGraph, region: terms.Region, closed, names, walked_axes) -> None:
    """The facts the rules read that each parameter's own values state (the engine's, `program.stated`, and
    the derivation's own), and each input's walked axis."""
    program.stated(graph, region)
    _least(graph, region)
    _dtype_constants(graph, region)
    for index, (primitive, params) in enumerate(list(region.params)):
        declared = terms.OPERATIONS.get(primitive)
        kind = None if declared is None else declared.kind
        if kind in ("binary", "unary", "comparison", "cast"):
            graph.register(keeps(i64(index)))
            if primitive is jax.lax.integer_pow_p and params.get("y") == 2:
                graph.register(is_square(i64(index)))
        elif primitive is jax.lax.broadcast_in_dim_p:
            dims = tuple(int(d) for d in params["broadcast_dimensions"])
            after = tuple(int(s) for s in params["shape"])
            for axis, to in enumerate(dims):
                graph.register(lays(i64(index), i64(axis), i64(to)))
            if dims == tuple(range(len(after))):
                graph.register(identity_broadcast(i64(index)))
            elif len(dims) == len(after) - 1:
                missing = [axis for axis in range(len(after)) if axis not in dims]
                if len(missing) == 1 and dims == tuple(p for p in range(len(after)) if p != missing[0]):
                    graph.register(inserts_axis(i64(index), i64(missing[0])))
        elif primitive is jax.lax.transpose_p:
            permutation = list(params["permutation"])
            for axis in range(len(permutation)):
                graph.register(lays(i64(index), i64(axis), i64(permutation.index(axis))))
        elif primitive in (jax.lax.squeeze_p, jax.lax.reshape_p):
            for axis, to in _moved(primitive, params, region, index):
                graph.register(lays(i64(index), i64(axis), i64(to)))
            if primitive is jax.lax.reshape_p and params.get("dimensions") is None:
                for axis in _inserted_reshape_axes(region, index):
                    graph.register(inserts_axis(i64(index), i64(axis)))
        elif kind == "reduction":
            reduced = {int(axis) for axis in params["axes"]}
            for axis in range(_rank_before(region, index, reduced)):
                if axis not in reduced:
                    graph.register(reduces(i64(index), i64(axis), i64(axis - sum(r < axis for r in reduced))))
        elif kind == "product":
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
    # A scalar anchor's axis (`scales`' -1) is none, through every equation.
    for index in range(len(region.params)):
        none = i64(-1)
        graph.register(lays(i64(index), none, none), reduces(i64(index), none, none),
                       left_of(i64(index), none, none), right_of(i64(index), none, none))
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
    region.params.append((jax.lax.gt_p, {}))
    graph.register(set_(comp_gt()).to(i64(len(region.params) - 1)))
    region.params.append((jax.lax.and_p, {}))
    graph.register(set_(logic_and()).to(i64(len(region.params) - 1)))
    for index, value in enumerate(region.literals):
        literal = Term.literal(i64(index))
        graph.register(local(literal), unwalked(literal))
        if not terms.typed_key(value):
            if np.all(np.asarray(value) == 0):
                graph.register(zero(literal))
            elif np.all(np.asarray(value) == 1):
                graph.register(unit_array(literal))


def _least(graph: EGraph, region: terms.Region) -> None:
    """Each floating value's dtype's least finite value (`least`), stated of every value a walked maximum
    may anchor (its own, or one made from it: a broadcast, a difference), one literal a dtype: maxima of two
    dtypes are each corrected by their own, where one value for the whole program refused them."""
    floors: dict[np.dtype, int] = {}
    for term, _ in region.shapes:
        dtype = region.dtypes.get(terms.key(term))
        if dtype is None or not jnp.issubdtype(dtype, jnp.floating):
            continue
        if dtype not in floors:
            region.literals.append(np.asarray(jnp.finfo(dtype).min, dtype))
            floors[dtype] = len(region.literals) - 1
            literal = Term.literal(i64(floors[dtype]))
            region.shaped(literal, ())
            graph.register(set_(rules.shape(literal)).to(Vec[i64]()))
        graph.register(set_(least(term)).to(i64(floors[dtype])))


def _dtype_constants(graph: EGraph, region: terms.Region) -> None:
    """Each floating value's dtype's 0 and 1 literals (`zero_of`, `one_of`) and each boolean value (`boolean`),
    registered for Chan-Golub-LeVeque zero-count guards and masked indicators."""
    zeros: dict[np.dtype, int] = {}
    ones: dict[np.dtype, int] = {}
    for term, _ in list(region.shapes):
        dtype = region.dtypes.get(terms.key(term))
        if dtype is None:
            continue
        if dtype == np.bool_:
            graph.register(boolean(term))
            continue
        if not jnp.issubdtype(dtype, jnp.floating):
            continue
        if dtype not in zeros:
            for table, value in ((zeros, 0.0), (ones, 1.0)):
                region.literals.append(np.asarray(value, dtype))
                table[dtype] = len(region.literals) - 1
                literal = Term.literal(i64(table[dtype]))
                region.shaped(literal, ())
                graph.register(set_(rules.shape(literal)).to(Vec[i64]()))
        graph.register(set_(zero_of(term)).to(i64(zeros[dtype])),
                       set_(one_of(term)).to(i64(ones[dtype])))


def _inserted_reshape_axes(region: terms.Region, index: int) -> list[int]:
    """Every axis `a` that a `reshape` of parameters `index` inserts as size 1 across all terms using `index`,
    keeping every other axis in order."""
    pairs = []
    for term, after in region.shapes:
        if (args := get_callable_args(term, Term.shaped)) is not None and terms.integer(args[2]) == index:
            pairs.append((region.shape(args[0]), tuple(after)))
    if not pairs:
        return []
    after = pairs[0][1]
    if any(len(after) != len(before) + 1 or end != after for before, end in pairs):
        return []
    return [axis for axis in range(len(after))
            if after[axis] == 1 and all(before == (*after[:axis], *after[axis + 1:]) for before, _ in pairs)]


def _moment_layouts(graph: EGraph, region: terms.Region, reductions) -> None:
    """Register `filled(to)` broadcasts (scalar to each reduction's shape) and `unreduced(to, axis)`
    broadcasts (reduced shape back to each walked shape) for central-moment rules."""
    done_fill: set[tuple[int, ...]] = set()
    for reduction in reductions:
        to = tuple(region.shape(reduction))
        if to in done_fill:
            continue
        done_fill.add(to)
        region.params.append((jax.lax.broadcast_in_dim_p,
                              {"shape": to, "broadcast_dimensions": (), "sharding": None}))
        graph.register(set_(filled(Vec[i64](*(i64(size) for size in to)))).to(
            i64(len(region.params) - 1)))
    done_unreduced: set[tuple[tuple[int, ...], int]] = set()
    for term, extent in list(region.shapes):
        axis_val = graph.lookup_function_value(walked(term))
        if axis_val is None:
            continue
        axis = terms.integer(axis_val)
        to = tuple(extent)
        if (to, axis) in done_unreduced:
            continue
        done_unreduced.add((to, axis))
        dims = tuple(p for p in range(len(to)) if p != axis)
        region.params.append((jax.lax.broadcast_in_dim_p,
                              {"shape": to, "broadcast_dimensions": dims, "sharding": None}))
        graph.register(set_(unreduced(Vec[i64](*(i64(size) for size in to)), i64(axis))).to(
            i64(len(region.params) - 1)))


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
        args = get_callable_args(term, Term.shaped) or get_callable_args(term, Term.reduce)
        if args is not None and terms.integer(args[-1]) == index:
            return region.extents.get(terms.key(args[0])), extent
    return None, None


def _rank_before(region: terms.Region, index: int, reduced: set[int]) -> int:
    before, after = _operand_and_result_shapes(region, index)
    return len(before) if before is not None else (len(after) + len(reduced) if after is not None else 0)


def _dot_ranks(region: terms.Region, index: int) -> tuple[int, int]:
    for term, _ in region.shapes:
        args = get_callable_args(term, Term.dot)
        if args is not None and terms.integer(args[-1]) == index:
            return len(region.shape(args[0])), len(region.shape(args[1]))
    return 0, 0


def _opaque_locals(graph: EGraph, region: terms.Region, walking: Ruleset) -> list[Term]:
    """The outputs of equations outside the language (`terms.Opaque`), each over local, unwalked inputs alone:
    local themselves, in program order, so an equation over an earlier one's outputs reads them as local. An
    equation outside the language over a value the walk reads is refused: over a walked value its axis is
    unknown, and over one a walked reduction gave (a total's `log2`) no form computes it from the walk's
    state, where evaluating it from the operands would read them whole."""
    found = []
    for opaque in region.opaque:
        inputs = [term for term in opaque.inputs if term is not None]
        name = opaque.equation.primitive.name
        if any(_walked(graph, term) for term in inputs):
            raise Refused(f"{name}: an equation outside the language over a walked value")
        if not all(_holds(graph, local(term)) and _holds(graph, unwalked(term)) for term in inputs):
            raise Refused(f"{name}: an equation outside the language over a value a walked reduction gave")
        for leaf in opaque.outputs:
            found.append(Term.leaf(i64(leaf)))
            graph.register(local(Term.leaf(i64(leaf))), unwalked(Term.leaf(i64(leaf))))
        graph.run(walking.saturate())
    return found


def _lift(graph: EGraph, region: terms.Region, reductions) -> None:
    """Each broadcast a rescaled sum may need (`lifts`), its parameters added to the region's table: a scalar
    anchor's to the sum's shape along no axis, a vector's along each of the sum's axes of its extent, of which
    the rules take the one its anchor's lies along (`anchored`). An anchor of more axes has none: its sum's
    correction is no broadcast of these, and the declaration is refused."""
    anchors = {extent for term, extent in region.shapes if _holds(graph, anchor(term))}
    lifted = set()
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
                graph.register(set_(lifts(Vec[i64](*(i64(size) for size in given)),
                                          Vec[i64](*(i64(size) for size in to)), i64(at))).to(
                    i64(len(region.params) - 1)))


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
        if (args := get_callable_args(term, Term.literal)) is None:
            return None
        value = np.asarray(region.literals[terms.integer(args[0])])
        return value.item() if value.shape == () and np.isfinite(value) else None

    for term, _ in region.shapes:
        for constructor in terms.LANGUAGE:
            if (args := get_callable_args(term, constructor)) is None:
                continue
            children = [arg for arg in args if isinstance(arg, Term)]
            for child in children:
                if (index := places.get(terms.key(child))) is None:
                    continue
                binary = get_callable_args(term, Term.binary)
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
            if (args := get_callable_args(term, Term.leaf)) is not None}


def _written(form: Term) -> bool:
    """Whether `form` keeps none of the derivation's readings: a form in the language."""
    return not any(get_callable_args(term, reading) is not None for term in terms.subterms(form)
                   for reading in _READINGS)
