"""Normalized anchored Chan-Golub-LeVeque walking, decoupled SSA rewrite rules, and resource registration
for central second moments (`dew.algebra.derive`, docs/design/algebra.md §6.9).

Split from `dew.algebra.derive` along the central-moment seam: all moment-specific egglog predicates, linear
SSA combine/finish/tile rewrite rules, and per-reduction literal/layout resource registration live here,
taking `derive`'s shared walking predicates (`local`, `summed`, `walked`, `zero`, `slot`) as parameters to
keep imports one-way."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from egglog import EGraph, String, Unit, Vec, eq, function, i64, rewrite, rule, set_, union, var, vars_

from dew.algebra import rules, terms
from dew.algebra.terms import Term


@function
def unit_array(term: Term) -> Unit: ...
"""A value of ones (a literal 1, or a layout or cast of one)."""


@function
def boolean(term: Term) -> Unit: ...
"""A boolean value (a mask)."""


@function
def nonnegative(term: Term) -> Unit: ...
"""A value proven nonnegative (`terms.Region.nonnegative`: declared or computed `>= 0`)."""


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
def half_of(term: Term) -> i64: ...
"""The literal index of 0.5 in `term`'s floating dtype."""


@function(merge=lambda old, new: old)
def quarter_max_of(term: Term) -> i64: ...
"""The literal index of `0.25 * finfo(dtype).max` in `term`'s floating dtype."""


@function(merge=lambda old, new: old)
def max_of(term: Term) -> i64: ...
"""The literal index of `finfo(dtype).max` in `term`'s floating dtype."""


@function(merge=lambda old, new: old)
def sqrt_max_of(term: Term) -> i64: ...
"""The literal index of `0.5 * sqrt(finfo(dtype).max)` in `term`'s floating dtype."""


@function(merge=lambda old, new: old)
def ratio_floor_of(term: Term) -> i64: ...
"""The literal index of `min(sqrt(finfo(dtype).tiny), 0.25 * finfo(dtype).eps)` in `term`'s floating dtype."""


@function(merge=lambda old, new: old)
def filled(to: Vec[i64]) -> i64: ...
"""The `broadcast_in_dim` parameters lifting a scalar to shape `to`."""


@function(merge=lambda old, new: old)
def unreduced(to: Vec[i64], axis: i64) -> i64: ...
"""The `broadcast_in_dim` parameters lifting a shape without `axis` back to `to`."""


@function(merge=lambda old, new: old)
def reduce_min_on(axes: Vec[i64]) -> i64: ...
"""The `reduce_min_p` parameters reducing along `axes`."""


@function(merge=lambda old, new: old)
def reduce_max_on(axes: Vec[i64]) -> i64: ...
"""The `reduce_max_p` parameters reducing along `axes`."""


@function(merge=lambda old, new: old)
def comp_gt() -> i64: ...
"""The parameters index of `lax.gt_p`."""


@function(merge=lambda old, new: old)
def comp_ge() -> i64: ...
"""The parameters index of `lax.ge_p`."""


@function(merge=lambda old, new: old)
def logic_and() -> i64: ...
"""The parameters index of `lax.and_p`."""


@function(merge=lambda old, new: old)
def logic_or() -> i64: ...
"""The parameters index of `lax.or_p`."""


@function(merge=lambda old, new: old)
def min_op() -> i64: ...
"""The parameters index of `lax.min_p`."""


@function(merge=lambda old, new: old)
def abs_op() -> i64: ...
"""The parameters index of `lax.abs_p`."""


@function
def positive_guard(term: Term) -> Unit: ...
"""A finite strictly positive constant (or a layout/cast of one), safe as a zero-count denominator guard."""


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
def guarded_count(term: Term, count: Term) -> Unit: ...
"""`term` equals `count` wherever `count > 0` and a finite positive constant where `count == 0`."""


@function
def count_lifted(term: Term, count: Term, axis: i64) -> Unit: ...
"""`term` is `count` lifted across `axis` (`keepdims`)."""


@function
def safe_count_lifted(term: Term, count: Term, axis: i64) -> Unit: ...
"""`term` is a zero-safe `count` lifted across `axis` (`keepdims`)."""


@function
def guarded_count_lifted(term: Term, count: Term, axis: i64) -> Unit: ...
"""`term` is a zero-guarded `count` lifted across `axis` (`keepdims`)."""


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
def centered(diff: Term, value: Term, total: Term, count: Term, weight: Term, axis: i64,
             sign: i64) -> Unit: ...
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


@function
def moment_anchor(term: Term) -> Term: ...
"""The carried base-mean anchor of an anchored first-moment reduction (`moment_first`), rewritten to a form
in the language alongside `stepped` and `merged` and never extracted itself."""


@function(merge=lambda old, new: old)
def anchor_slot(term: Term) -> i64: ...
"""The state slot holding `term`'s base-mean anchor (`moment_anchor`)."""


@function
def moment_hi(term: Term) -> Term: ...
"""The carried scaled overflow band of a zero-mass central second moment, kept separate from the unscaled
plain band so large entries never underflow small entries in the same tile or merge."""


@function(merge=lambda old, new: old)
def hi_slot(term: Term) -> i64: ...
"""The state slot holding `term`'s zero-mass scaled overflow band (`moment_hi`)."""


@function
def moment_scale(term: Term) -> Term: ...
"""The carried zero-mass scale factor (`>= 1`) of a central second moment, rewritten alongside `stepped`
and `merged` so zero-mass quadratic tangents cancel before unscaling."""


@function(merge=lambda old, new: old)
def scale_slot(term: Term) -> i64: ...
"""The state slot holding `term`'s zero-mass scale (`moment_scale`)."""


@function
def moment_first(sx: Term, count: Term, value: Term, weight: Term, axis: i64) -> Unit: ...
"""`sx` is a weighted first-moment sum participating in a recognized central second moment."""


@function
def masked_moment_first(sx: Term, count: Term, value: Term, mask: Term, z: Term, axis: i64) -> Unit: ...
"""`sx` is a masked first-moment sum participating in a recognized central second moment."""


@function
def anchored_first(term: Term) -> Unit: ...
"""`term` is carried as an anchored `(anchor, normalized_residual)` pair rather than a raw partial sum."""


@function
def scaled_second(term: Term) -> Unit: ...
"""`term` is a central second moment carrying a companion zero-mass scale slot (`moment_scale`)."""


@function
def plain_local(term: Term) -> Unit: ...
"""`term` is a local walked reduction carried directly by its monoid without a companion anchor slot."""


@function
def moment_first_of(sx: Term, count: Term, axis: i64) -> Unit: ...
"""`sx` is a shape-validated first-moment sum (weighted or masked) participating in a central moment."""


@function
def moment_second_of(term: Term, sx: Term, sy: Term, count: Term, axis: i64, sign: i64) -> Unit: ...
"""`term` is a shape-validated central second moment (weighted or masked)."""


@function
def _m_part(tag: String, ref: Term, n0: Term, n1: Term) -> Term: ...
@function
def _m_diff(tag: String, ref: Term, a0: Term, a1: Term, u0: Term, u1: Term) -> Term: ...
@function
def _m_double_if(doubled: Term, value: Term) -> Term: ...
@function
def _m_prod_div(zero_t: Term, one_t: Term, product: Term, factor: Term, denom: Term) -> Term: ...
@function
def _m_ext_mul(ref: Term, extreme: Term, k_min: Term, safe_max: Term, factor: Term) -> Term: ...
@function
def _m_ext_mul_body(ref: Term, extreme: Term, ext_n: Term, ext_v: Term, big_n: Term,
                    safe_max: Term) -> Term: ...
@function
def _m_ext_corr(ref: Term, extreme: Term, k_min: Term, safe_max: Term, big_d: Term,
                small_d: Term) -> Term: ...
@function
def _m_ext_corr_body(ref: Term, extreme: Term, ext_k: Term, ext_b: Term, ext_s: Term,
                     div_first: Term, safe_max: Term) -> Term: ...
@function
def _m_zero_anchor(ref: Term, a0: Term, a1: Term) -> Term: ...
@function
def _m_tile_anchor(ref: Term, x: Term, axes: Vec[i64]) -> Term: ...
@function
def _m_pair_scale(ref: Term, dx: Term, dy: Term) -> Term: ...
@function
def _m_shift_plain(ref: Term, c: Term, n: Term, dx: Term, dy: Term, ux: Term, uy: Term,
                   sign: i64) -> Term: ...
@function
def _m_shift_hi(ref: Term, hi: Term, sc: Term, n: Term, dx: Term, dy: Term, ux: Term, uy: Term,
                scale_out: Term, sign: i64) -> Term: ...
@function
def _m_combine_zero_u(ref: Term, either: Term, n0: Term, ax0: Term, ux0: Term, n1: Term, ax1: Term,
                      ux1: Term) -> Term: ...
@function
def _m_combine_u(ref: Term, n0: Term, ax0: Term, ux0: Term, n1: Term, ax1: Term, ux1: Term) -> Term: ...
@function
def _m_combine_u_body(ref: Term, either: Term, both: Term, extreme: Term, n_min: Term, n_drop: Term,
                      safe_max: Term, kept_a: Term, drop_a: Term, kept_u: Term, drop_u: Term,
                      p_kept: Term, p_drop: Term, u_zero: Term) -> Term: ...
@function
def _m_combine_base(ref: Term, extreme: Term, keep0: Term, n_min: Term, safe_max: Term, c0: Term,
                    c1: Term, p0: Term, p1: Term) -> Term: ...
@function
def _m_combine_corr(ref: Term, both: Term, extreme: Term, n_min: Term, safe_max: Term, p_prod: Term,
                    dx_eff: Term, big_x: Term, dy_eff: Term, big_y: Term) -> Term: ...
@function
def _m_combine_corr_body(ref: Term, both: Term, extreme: Term, k_min: Term, safe_max: Term,
                         eff_p: Term, big_d: Term, small_d: Term) -> Term: ...
@function
def _m_combine_s(ref: Term, n0: Term, sc0: Term, ax0: Term, ay0: Term, n1: Term, sc1: Term, ax1: Term,
                 ay1: Term) -> Term: ...
@function
def _m_combine_hi(ref: Term, hi0: Term, sc0: Term, n0: Term, ax0: Term, ux0: Term, ay0: Term, uy0: Term,
                  hi1: Term, sc1: Term, n1: Term, ax1: Term, ux1: Term, ay1: Term, uy1: Term,
                  sign: i64) -> Term: ...
@function
def _m_combine_zero_c(ref: Term, either: Term, c0: Term, n0: Term, ax0: Term, ux0: Term, ay0: Term,
                      uy0: Term, c1: Term, n1: Term, ax1: Term, ux1: Term, ay1: Term, uy1: Term,
                      sign: i64) -> Term: ...
@function
def _m_combine_one_c(ref: Term, either: Term, both: Term, safe_max: Term, c_kept: Term, c_drop: Term,
                     hi_drop: Term, s_drop: Term, p_kept: Term, n_drop: Term, dx_eff: Term, bx: Term,
                     dy_eff: Term, by: Term, drop_ux: Term, drop_uy: Term, sign: i64) -> Term: ...
@function
def _m_combine_c(ref: Term, c0: Term, hi0: Term, sc0: Term, n0: Term, ax0: Term, ux0: Term, ay0: Term,
                 uy0: Term, c1: Term, hi1: Term, sc1: Term, n1: Term, ax1: Term, ux1: Term, ay1: Term,
                 uy1: Term, sign: i64) -> Term: ...
@function
def _m_finish_u(tag: String, ref: Term, n0: Term, ax0: Term, ux0: Term) -> Term: ...
@function
def _m_finish_c(tag: String, ref: Term, c0: Term, hi0: Term, sc0: Term, n0: Term, ax0: Term, ux0: Term,
                ay0: Term, uy0: Term, sign: i64) -> Term: ...
@function
def _m_wcov_ovf(ref: Term, x: Term, y: Term, axes: Vec[i64], lift_x: i64, fill_x: i64) -> Term: ...
@function
def _m_wcov_scale(ref: Term, x: Term, y: Term, axes: Vec[i64], lift_x: i64, fill_x: i64) -> Term: ...
@function
def _m_safe_x(ref: Term, lift_x: i64) -> Term: ...
@function
def _m_wmul_first(ref: Term, weight: Term, lift_x: i64, fill_x: i64) -> Term: ...
@function
def _m_wmean(tag: String, ref: Term, values: Term, weight: Term, axes: Vec[i64], p: i64, lift_x: i64,
             fill_x: i64) -> Term: ...
@function
def _m_mmean(tag: String, ref: Term, values: Term, mask: Term, z: Term, axes: Vec[i64], p: i64,
             lift_x: i64, fill_x: i64) -> Term: ...
@function
def _m_wcov_raw(ref: Term, dx: Term, dy_s: Term, sc_x: Term, weight: Term, axes: Vec[i64], p: i64,
                lift_x: i64, fill_x: i64) -> Term: ...
@function
def _m_wcov_term(ref: Term, big_d: Term, small_d: Term, sc_x: Term, weight: Term, axes: Vec[i64], p: i64,
                 lift_x: i64, fill_x: i64) -> Term: ...
@function
def _m_wcov_tile(tag: String, ref: Term, x: Term, y: Term, weight: Term, axes: Vec[i64], p: i64,
                 lift_x: i64, fill_x: i64, sign: i64) -> Term: ...
@function
def _m_mcov_raw(ref: Term, dx: Term, dy_s: Term, mask: Term, axes: Vec[i64], p: i64, lift_x: i64,
                fill_x: i64) -> Term: ...
@function
def _m_mcov_tile(ref: Term, x: Term, y: Term, mask: Term, z: Term, axes: Vec[i64], p: i64, lift_x: i64,
                 fill_x: i64, sign: i64) -> Term: ...


MOMENT_SSA = (
    _m_part, _m_diff, _m_double_if, _m_prod_div, _m_ext_mul, _m_ext_mul_body, _m_ext_corr,
    _m_ext_corr_body, _m_zero_anchor, _m_tile_anchor, _m_pair_scale, _m_shift_plain, _m_shift_hi,
    _m_combine_zero_u, _m_combine_u, _m_combine_u_body, _m_combine_base, _m_combine_corr,
    _m_combine_corr_body, _m_combine_s, _m_combine_hi, _m_combine_zero_c, _m_combine_one_c, _m_combine_c,
    _m_finish_u, _m_finish_c, _m_wcov_ovf, _m_wcov_scale, _m_safe_x, _m_wmul_first, _m_wmean, _m_mmean,
    _m_wcov_raw, _m_wcov_term, _m_wcov_tile, _m_mcov_raw, _m_mcov_tile,
)


def _moment_counts(local, walked, zero) -> list:
    """Rules recognizing indicators, counts, first sums, zero-safe denominators, and keepdims lifts."""
    x, z, w, mask, ind, t, n, s, d, c = vars_("x z w mask ind t n s d c", Term)
    layout = var("layout", String)
    p, q, a = vars_("p q a", i64)
    axes = var("axes", Vec[i64])
    return [
        # Ones and booleans through layouts and casts; 0/1 mask indicators.
        rule(eq(t).to(x.shaped(layout, p)), unit_array(x)).then(unit_array(t)),
        rule(eq(t).to(x.convert(p)), unit_array(x)).then(unit_array(t)),
        rule(eq(t).to(x.shaped(layout, p)), positive_guard(x)).then(positive_guard(t)),
        rule(eq(t).to(x.convert(p)), positive_guard(x)).then(positive_guard(t)),
        rule(eq(t).to(x.shaped(layout, p)), boolean(x)).then(boolean(t)),
        rule(eq(t).to(mask.convert(p)), boolean(mask)).then(indicator(t, mask)),
        rule(eq(t).to(mask.select(z, x)), boolean(mask), zero(z), unit_array(x)).then(indicator(t, mask)),
        # Counts and first sums along the walked axis.
        rule(eq(n).to(w.reduce(String("sum"), axes, p)), local(w), nonnegative(w), eq(walked(w)).to(a),
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
        rule(eq(d).to(n.compare(String("gt"), z, p).select(c, n)), safe_count(n, n), zero(z),
             positive_guard(c)).then(guarded_count(d, n)),
        rule(eq(d).to(z.compare(String("lt"), n, p).select(c, n)), safe_count(n, n), zero(z)).then(
            safe_count(d, n)),
        rule(eq(d).to(z.compare(String("lt"), n, p).select(c, n)), safe_count(n, n), zero(z),
             positive_guard(c)).then(guarded_count(d, n)),
        rule(eq(d).to(n.compare(String("ne"), z, p).select(c, n)), safe_count(n, n), zero(z)).then(
            safe_count(d, n)),
        rule(eq(d).to(n.compare(String("ne"), z, p).select(c, n)), safe_count(n, n), zero(z),
             positive_guard(c)).then(guarded_count(d, n)),
        rule(eq(d).to(n.compare(String("eq"), z, p).select(n, c)), safe_count(n, n), zero(z)).then(
            safe_count(d, n)),
        rule(eq(d).to(n.compare(String("eq"), z, p).select(n, c)), safe_count(n, n), zero(z),
             positive_guard(c)).then(guarded_count(d, n)),
        rule(eq(d).to(z.compare(String("eq"), n, p).select(n, c)), safe_count(n, n), zero(z)).then(
            safe_count(d, n)),
        rule(eq(d).to(z.compare(String("eq"), n, p).select(n, c)), safe_count(n, n), zero(z),
             positive_guard(c)).then(guarded_count(d, n)),
        # Keepdims lifts (inserting the walked axis at its exact place before dividing).
        rule(eq(t).to(n.shaped(layout, p)), counts(n, w, a), inserts_axis(p, a)).then(
            count_lifted(t, n, a), safe_count(t, t)),
        rule(eq(t).to(n.shaped(layout, p)), masked_counts(n, mask, a), inserts_axis(p, a)).then(
            count_lifted(t, n, a), safe_count(t, t)),
        rule(eq(t).to(d.shaped(layout, p)), safe_count(d, n), inserts_axis(p, a)).then(
            safe_count_lifted(t, n, a)),
        rule(eq(t).to(d.shaped(layout, p)), guarded_count(d, n), inserts_axis(p, a)).then(
            guarded_count_lifted(t, n, a)),
        rule(count_lifted(c, n, a), safe_count(d, c)).then(safe_count_lifted(d, n, a)),
        rule(count_lifted(c, n, a), guarded_count(d, c)).then(guarded_count_lifted(d, n, a)),
        rule(eq(d).to(c.shaped(String("squeeze"), p)), count_lifted(c, n, a),
             eq(rules.shape(d)).to(rules.shape(n))).then(safe_count(d, n)),
        rule(eq(d).to(c.shaped(String("squeeze"), p)), guarded_count_lifted(c, n, a),
             eq(rules.shape(d)).to(rules.shape(n))).then(guarded_count(d, n)),
        rule(eq(t).to(s.shaped(layout, p)), first_sum(s, x, w, a), inserts_axis(p, a)).then(
            first_lifted(t, s, x, w, a)),
        rule(eq(t).to(s.shaped(layout, p)), masked_first_sum(s, x, mask, a), inserts_axis(p, a)).then(
            masked_first_lifted(t, s, x, mask, a)),
    ]


def _moment_means(zero) -> list:
    """Rules recognizing weighted and masked means, zero-guards, and axis-aligned mean broadcasts."""
    x, z, w, mask, t, n, s, d, c, mu = vars_("x z w mask t n s d c mu", Term)
    layout = var("layout", String)
    p, a = vars_("p a", i64)
    shape = rules.shape
    found = [
        rule(eq(mu).to(s.binary(String("div"), d, p)), first_sum(s, x, w, a), counts(n, w, a),
             safe_count(d, n), eq(shape(s)).to(shape(n))).then(mean_of(mu, s, n, x, w, a)),
        rule(eq(mu).to(s.binary(String("div"), d, p)), masked_first_sum(s, x, mask, a),
             masked_counts(n, mask, a), safe_count(d, n),
             eq(shape(s)).to(shape(n))).then(masked_mean_of(mu, s, n, x, mask, a)),
        rule(mean_of(mu, s, n, x, w, a), eq(shape(mu)).to(Vec[i64].empty())).then(
            mean_lifted(mu, s, n, x, w, a)),
        rule(masked_mean_of(mu, s, n, x, mask, a), eq(shape(mu)).to(Vec[i64].empty())).then(
            masked_mean_lifted(mu, s, n, x, mask, a)),
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
    ]
    guards = (
        n.compare(String("gt"), z, p).select(c, mu),
        z.compare(String("lt"), n, p).select(c, mu),
        n.compare(String("ne"), z, p).select(c, mu),
        n.compare(String("eq"), z, p).select(mu, c),
        z.compare(String("eq"), n, p).select(mu, c),
    )
    found.extend(
        rule(eq(t).to(guard), mean_fact(mu, s, n, x, ref, a), zero(z), zero(c)).then(
            mean_fact(t, s, n, x, ref, a))
        for mean_fact, ref in ((mean_of, w), (masked_mean_of, mask))
        for guard in guards)
    return found


def _moment_centered(summed, zero) -> list:
    """Rules recognizing centered differences, squares/products, weighted terms, and second central sums."""
    x, y, z, w, mask, t, n, s, sx, sy, mu, dx, dy, pair, m = vars_(
        "x y z w mask t n s sx sy mu dx dy pair m", Term)
    p, q, a, sign = vars_("p q a sign", i64)
    axes = var("axes", Vec[i64])
    shape = rules.shape
    found = [
        rule(eq(dx).to(x.binary(String("sub"), mu, p)), mean_lifted(mu, s, n, x, w, a),
             eq(shape(dx)).to(shape(x))).then(centered(dx, x, s, n, w, a, i64(1))),
        rule(eq(dx).to(mu.binary(String("sub"), x, p)), mean_lifted(mu, s, n, x, w, a),
             eq(shape(dx)).to(shape(x))).then(centered(dx, x, s, n, w, a, i64(-1))),
        rule(eq(dx).to(x.binary(String("sub"), mu, p)), masked_mean_lifted(mu, s, n, x, mask, a),
             eq(shape(dx)).to(shape(x))).then(masked_centered(dx, x, s, n, mask, a, i64(1))),
        rule(eq(dx).to(mu.binary(String("sub"), x, p)), masked_mean_lifted(mu, s, n, x, mask, a),
             eq(shape(dx)).to(shape(x))).then(masked_centered(dx, x, s, n, mask, a, i64(-1))),
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
            summed(m), central_moment(m, sx, sy, n, x, y, w, a, sign),
            moment_first(sx, n, x, w, a), moment_first(sy, n, y, w, a),
            anchored_first(sx), anchored_first(sy), scaled_second(m)),
        rule(eq(m).to(t.reduce(String("sum"), axes, p)), eq(axes).to(Vec[i64](a)),
             masked_weighted_central(t, sx, sy, n, x, y, mask, z, a, sign)).then(
            summed(m), masked_central_moment(m, sx, sy, n, x, y, mask, z, a, sign),
            masked_moment_first(sx, n, x, mask, z, a), masked_moment_first(sy, n, y, mask, z, a),
            anchored_first(sx), anchored_first(sy), scaled_second(m)),
        rule(central_moment(m, sx, sy, n, x, y, w, a, sign), eq(shape(w)).to(shape(x)),
             eq(shape(y)).to(shape(x))).then(
            moment_second_of(m, sx, sy, n, a, sign), moment_first_of(sx, n, a), moment_first_of(sy, n, a)),
        rule(masked_central_moment(m, sx, sy, n, x, y, mask, z, a, sign), eq(shape(mask)).to(shape(x)),
             eq(shape(y)).to(shape(x))).then(
            moment_second_of(m, sx, sy, n, a, sign), moment_first_of(sx, n, a), moment_first_of(sy, n, a)),
    ]
    return found


def moment_walking(local, summed, walked, zero) -> list:
    """The facts' rules recognizing weighted and masked central second moments (variance and covariance)
    along the walked axis, for Chan-Golub-LeVeque decoupling (`_moment_decoupled`)."""
    return [*_moment_counts(local, walked, zero), *_moment_means(zero), *_moment_centered(summed, zero)]


def _moment_ops() -> tuple[dict[str, i64], dict[str, tuple]]:
    """Shared egglog op-index variables and binding clauses for moment SSA rules."""
    added, subtracted, multiplied, divided = vars_("added subtracted multiplied divided", i64)
    max_idx, min_idx, abs_idx, gt_idx, ge_idx, and_idx, or_idx = vars_(
        "max_idx min_idx abs_idx gt_idx ge_idx and_idx or_idx", i64)
    stock = rules.stock
    ops = {"add": added, "sub": subtracted, "mul": multiplied, "div": divided, "max": max_idx,
           "min": min_idx, "abs": abs_idx, "gt": gt_idx, "ge": ge_idx, "and": and_idx, "or": or_idx}
    binds = {
        "add": (eq(added).to(stock("add")),), "sub": (eq(subtracted).to(stock("sub")),),
        "mul": (eq(multiplied).to(stock("mul")),), "div": (eq(divided).to(stock("div")),),
        "max": (eq(max_idx).to(stock("max")),), "min": (eq(min_idx).to(min_op()),),
        "abs": (eq(abs_idx).to(abs_op()),), "gt": (eq(gt_idx).to(comp_gt()),),
        "ge": (eq(ge_idx).to(comp_ge()),), "and": (eq(and_idx).to(logic_and()),),
        "or": (eq(or_idx).to(logic_or()),),
    }
    return ops, binds


def _moment_ref_consts(ref: Term) -> tuple[dict[str, i64], dict[str, Term], tuple, dict[str, tuple]]:
    """Scalar literal indices, broadcast constants, and binding clauses keyed by count reduction `ref`."""
    to_n = var("to_n", Vec[i64])
    fill_n, z_n, o_n, h_n, q_n, m_n, sq_n, r_flr = vars_("fill_n z_n o_n h_n q_n m_n sq_n r_flr", i64)
    lits = {"zero": z_n, "one": o_n, "half": h_n, "qmax": q_n, "fmax": m_n, "sqmax": sq_n, "rflr": r_flr}
    consts = {name: Term.literal(idx).shaped(String("broadcast_in_dim"), fill_n)
              for name, idx in lits.items()}
    fill_bind = (eq(to_n).to(rules.shape(ref)), eq(fill_n).to(filled(to_n)))
    lit_binds = {
        "zero": (eq(z_n).to(zero_of(ref)),), "one": (eq(o_n).to(one_of(ref)),),
        "half": (eq(h_n).to(half_of(ref)),), "qmax": (eq(q_n).to(quarter_max_of(ref)),),
        "fmax": (eq(m_n).to(max_of(ref)),), "sqmax": (eq(sq_n).to(sqrt_max_of(ref)),),
        "rflr": (eq(r_flr).to(ratio_floor_of(ref)),),
    }
    return lits, consts, fill_bind, lit_binds


def _need(table: dict[str, tuple], *names: str) -> tuple:
    return tuple(clause for name in names for clause in table[name])


def _moment_ssa_prob_rules(ops: dict[str, i64], binds: dict[str, tuple]) -> list:
    """Linear SSA rules for merge probabilities, adaptive scaled differences, and guarded product-div."""
    ref, n0, n1, a0, a1, u0, u1 = vars_("ref n0 n1 a0 a1 u0 u1", Term)
    doubled, value, zero_t, one_t, product, factor, denom = vars_(
        "doubled value zero_t one_t product factor denom", Term)
    _, consts, fill_bind, lit_binds = _moment_ref_consts(ref)
    zero_n, one_n, half_n = consts["zero"], consts["one"], consts["half"]
    qmax, rflr = consts["qmax"], consts["rflr"]
    both = _m_part(String("both"), ref, n0, n1)
    either = _m_part(String("either"), ref, n0, n1)
    smax = _m_part(String("safe_max"), ref, n0, n1)
    usum = _m_part(String("u_sum"), ref, n0, n1)
    big_d = _m_diff(String("big"), ref, a0, a1, u0, u1)
    raw = big_d.select(a1, zero_n).binary(String("sub"), big_d.select(a0, zero_n), ops["sub"]).binary(
        String("add"),
        big_d.select(u1, zero_n).binary(String("sub"), big_d.select(u0, zero_n), ops["sub"]), ops["add"])
    halved = a1.binary(String("mul"), half_n, ops["mul"]).binary(
        String("sub"), a0.binary(String("mul"), half_n, ops["mul"]), ops["sub"]).binary(
        String("add"),
        u1.binary(String("mul"), half_n, ops["mul"]).binary(
            String("sub"), u0.binary(String("mul"), half_n, ops["mul"]), ops["sub"]), ops["add"])
    big_p = product.unary(String("abs"), ops["abs"]).compare(String("gt"), one_t, ops["gt"])
    return [
        rewrite(_m_part(String("keep0"), ref, n0, n1)).to(
            n0.compare(String("gt"), zero_n, ops["gt"]).binary(
                String("and"), n0.compare(String("ge"), n1, ops["ge"]), ops["and"]),
            *fill_bind, *lit_binds["zero"], *_need(binds, "gt", "ge", "and")),
        rewrite(_m_part(String("both"), ref, n0, n1)).to(
            n0.compare(String("gt"), zero_n, ops["gt"]).binary(
                String("and"), n1.compare(String("gt"), zero_n, ops["gt"]), ops["and"]),
            *fill_bind, *lit_binds["zero"], *_need(binds, "gt", "and")),
        rewrite(_m_part(String("either"), ref, n0, n1)).to(
            n0.compare(String("gt"), zero_n, ops["gt"]).binary(
                String("or"), n1.compare(String("gt"), zero_n, ops["gt"]), ops["or"]),
            *fill_bind, *lit_binds["zero"], *_need(binds, "gt", "or")),
        rewrite(_m_part(String("safe_max"), ref, n0, n1)).to(
            either.select(one_n, n0.binary(String("max"), n1, ops["max"])),
            *fill_bind, *lit_binds["one"], *_need(binds, "max")),
        rewrite(_m_part(String("extreme"), ref, n0, n1)).to(
            both.binary(String("and"), rflr.compare(
                String("gt"), both.select(zero_n, n0.binary(String("min"), n1, ops["min"]).binary(
                    String("div"), smax, ops["div"])), ops["gt"]), ops["and"]),
            *fill_bind, *_need(lit_binds, "zero", "rflr"), *_need(binds, "min", "div", "gt", "and")),
        rewrite(_m_part(String("u_sum"), ref, n0, n1)).to(
            either.select(one_n, either.select(zero_n, n0.binary(String("div"), smax, ops["div"])).binary(
                String("add"), either.select(zero_n, n1.binary(String("div"), smax, ops["div"])),
                ops["add"])),
            *fill_bind, *_need(lit_binds, "zero", "one"), *_need(binds, "div", "add")),
        rewrite(_m_part(String("p0"), ref, n0, n1)).to(
            either.select(zero_n, n0.binary(String("div"), smax, ops["div"])).binary(
                String("div"), usum, ops["div"]),
            *fill_bind, *lit_binds["zero"], *_need(binds, "div")),
        rewrite(_m_part(String("p1"), ref, n0, n1)).to(
            either.select(zero_n, n1.binary(String("div"), smax, ops["div"])).binary(
                String("div"), usum, ops["div"]),
            *fill_bind, *lit_binds["zero"], *_need(binds, "div")),
        rewrite(_m_diff(String("big"), ref, a0, a1, u0, u1)).to(
            a0.unary(String("abs"), ops["abs"]).binary(
                String("max"), a1.unary(String("abs"), ops["abs"]), ops["max"]).binary(
                String("max"), u0.unary(String("abs"), ops["abs"]).binary(
                    String("max"), u1.unary(String("abs"), ops["abs"]), ops["max"]), ops["max"]).compare(
                String("gt"), qmax, ops["gt"]),
            *fill_bind, *lit_binds["qmax"], *_need(binds, "abs", "max", "gt")),
        rewrite(_m_diff(String("val"), ref, a0, a1, u0, u1)).to(
            big_d.select(raw, halved),
            *fill_bind, *_need(lit_binds, "zero", "half"), *_need(binds, "add", "sub", "mul")),
        rewrite(_m_double_if(doubled, value)).to(
            doubled.select(value, value.binary(String("add"), value, ops["add"])), *_need(binds, "add")),
        rewrite(_m_prod_div(zero_t, one_t, product, factor, denom)).to(
            big_p.select(
                big_p.select(product, zero_t).binary(String("mul"), factor, ops["mul"]).binary(
                    String("div"), denom, ops["div"]),
                product.binary(String("div"), denom, ops["div"]).binary(String("mul"), factor, ops["mul"])),
            *_need(binds, "abs", "gt", "mul", "div")),
    ]


def _moment_ssa_extreme_rules(ops: dict[str, i64], binds: dict[str, tuple]) -> list:
    """Linear SSA rules for double-where guarded extreme-ratio products and corrections."""
    ref, extreme, k_min, safe_max, factor = vars_("ref extreme k_min safe_max factor", Term)
    ext_n, ext_v, big_n, big_d, small_d = vars_("ext_n ext_v big_n big_d small_d", Term)
    ext_k, ext_b, ext_s, div_first = vars_("ext_k ext_b ext_s div_first", Term)
    _, consts, fill_bind, lit_binds = _moment_ref_consts(ref)
    zero_n, one_n = consts["zero"], consts["one"]
    ext_n_e = extreme.select(zero_n, k_min)
    ext_v_e = extreme.select(zero_n, factor)
    max_lo_m = extreme.select(one_n, big_n.select(safe_max, one_n))
    max_hi_m = extreme.select(one_n, big_n.select(one_n, safe_max))
    arm_lo_m = big_n.select(ext_n, zero_n).binary(String("mul"), ext_v, ops["mul"]).binary(
        String("div"), max_lo_m, ops["div"])
    arm_hi_m = ext_v.binary(String("div"), max_hi_m, ops["div"]).binary(
        String("mul"), big_n.select(zero_n, ext_n), ops["mul"])
    ext_k_e = extreme.select(zero_n, k_min)
    ext_b_e = extreme.select(zero_n, big_d)
    ext_s_e = extreme.select(zero_n, small_d)
    div_first_e = ext_k_e.compare(String("gt"), one_n, ops["gt"]).binary(
        String("and"), ext_b_e.unary(String("abs"), ops["abs"]).compare(String("gt"), one_n, ops["gt"]),
        ops["and"])
    max_hi_c = extreme.select(one_n, div_first.select(one_n, safe_max))
    max_lo_c = extreme.select(one_n, div_first.select(safe_max, one_n))
    arm_hi_c = ext_b.binary(String("div"), max_hi_c, ops["div"]).binary(
        String("mul"), div_first.select(zero_n, ext_k), ops["mul"]).binary(
        String("mul"), ext_s, ops["mul"])
    t1 = div_first.select(ext_k, zero_n).binary(
        String("mul"), div_first.select(ext_b, zero_n), ops["mul"])
    arm_lo_c = _m_prod_div(zero_n, one_n, t1, ext_s, max_lo_c)
    return [
        rewrite(_m_ext_mul(ref, extreme, k_min, safe_max, factor)).to(
            _m_ext_mul_body(ref, extreme, ext_n_e, ext_v_e, ext_n_e.compare(String("gt"), one_n, ops["gt"]),
                            safe_max),
            *fill_bind, *_need(lit_binds, "zero", "one"), *_need(binds, "gt")),
        rewrite(_m_ext_mul_body(ref, extreme, ext_n, ext_v, big_n, safe_max)).to(
            extreme.select(zero_n, big_n.select(arm_lo_m, arm_hi_m)),
            *fill_bind, *_need(lit_binds, "zero", "one"), *_need(binds, "mul", "div")),
        rewrite(_m_ext_corr(ref, extreme, k_min, safe_max, big_d, small_d)).to(
            _m_ext_corr_body(ref, extreme, ext_k_e, ext_b_e, ext_s_e, div_first_e, safe_max),
            *fill_bind, *_need(lit_binds, "zero", "one"), *_need(binds, "gt", "abs", "and")),
        rewrite(_m_ext_corr_body(ref, extreme, ext_k, ext_b, ext_s, div_first, safe_max)).to(
            extreme.select(zero_n, div_first.select(arm_lo_c, arm_hi_c)),
            *fill_bind, *_need(lit_binds, "zero", "one"), *_need(binds, "mul", "div")),
    ]


def _moment_combine_parts(ref: Term, n0: Term, n1: Term, ops: dict[str, i64]) -> tuple[Term, ...]:
    """Shared merge-partition SSA terms `(keep0, either, both, extreme, safe_max, p0, p1, n_min, n_drop)`."""
    k0 = _m_part(String("keep0"), ref, n0, n1)
    return (
        k0, _m_part(String("either"), ref, n0, n1), _m_part(String("both"), ref, n0, n1),
        _m_part(String("extreme"), ref, n0, n1), _m_part(String("safe_max"), ref, n0, n1),
        _m_part(String("p0"), ref, n0, n1), _m_part(String("p1"), ref, n0, n1),
        n0.binary(String("min"), n1, ops["min"]), k0.select(n0, n1),
    )


def _moment_one_side_scale(either: Term, both: Term, safe_max: Term, n_drop: Term, zero_n: Term,
                           one_n: Term, ops: dict[str, i64]) -> tuple[Term, ...]:
    """Double-where partition of one-sided denominator `smax_1` across `< 1` vs `>= 1` so reverse-mode
    VJP neither overflows `ct / tiny` nor sends `inf * 0` into `smax_1` on `L = 2^70`."""
    smax_1 = either.select(one_n, both.select(safe_max, one_n))
    g1_n = either.select(zero_n, both.select(n_drop, zero_n))
    small_s = one_n.compare(String("gt"), smax_1, ops["gt"])
    return (smax_1, small_s, small_s.select(smax_1, one_n), small_s.select(one_n, smax_1),
            small_s.select(g1_n, zero_n), small_s.select(zero_n, g1_n))


def _moment_ssa_anchor_rules(ops: dict[str, i64], binds: dict[str, tuple]) -> list:
    """Linear SSA rules for closest-to-zero anchors, overflow scales, and split plain/scaled shifts."""
    ref, a0, a1, x, dx, dy, c, hi, sc, n, ux, uy, scale_out = vars_(
        "ref a0 a1 x dx dy c hi sc n ux uy scale_out", Term)
    axes = var("axes", Vec[i64])
    p_min, p_max = vars_("p_min p_max", i64)
    _, consts, fill_bind, lit_binds = _moment_ref_consts(ref)
    zero_n, one_n, qmax_n = consts["zero"], consts["one"], consts["qmax"]
    fmax_n, sqmax_n = consts["fmax"], consts["sqmax"]
    seen0 = fmax_n.compare(String("ge"), a0.unary(String("abs"), ops["abs"]), ops["ge"])
    seen1 = fmax_n.compare(String("ge"), a1.unary(String("abs"), ops["abs"]), ops["ge"])
    a0_g, a1_g = seen0.select(zero_n, a0), seen1.select(zero_n, a1)
    lo_a, hi_a = a0_g.binary(String("min"), a1_g, ops["min"]), a0_g.binary(String("max"), a1_g, ops["max"])
    z_both = lo_a.binary(String("max"), hi_a.binary(String("min"), zero_n, ops["min"]), ops["max"])
    lo_x, hi_x = x.reduce(String("min"), axes, p_min), x.reduce(String("max"), axes, p_max)
    z_tile = lo_x.binary(String("max"), hi_x.binary(String("min"), zero_n, ops["min"]), ops["max"])
    adx, ady = dx.unary(String("abs"), ops["abs"]), dy.unary(String("abs"), ops["abs"])
    x_ge_y = adx.compare(String("ge"), ady, ops["ge"])
    dbig, dsml = x_ge_y.select(dy, dx), x_ge_y.select(dx, dy)
    ab, as_ = x_ge_y.select(ady, adx), x_ge_y.select(adx, ady)
    ovf_p = ab.compare(String("gt"), one_n, ops["gt"]).binary(
        String("and"),
        as_.compare(String("gt"), qmax_n.binary(String("div"), ab.binary(String("max"), one_n, ops["max"]),
                                                ops["div"]), ops["gt"]), ops["and"])
    hi_dy = ady.compare(String("gt"), sqmax_n, ops["gt"])
    hi_dx = adx.compare(String("gt"), sqmax_n, ops["gt"])
    need_sc = ab.compare(String("gt"), sqmax_n, ops["gt"]).binary(String("or"), ovf_p, ops["or"])
    tsh_p = ovf_p.select(n, zero_n).binary(String("mul"), ovf_p.select(dx, zero_n), ops["mul"]).binary(
        String("mul"), ovf_p.select(dy, zero_n), ops["mul"])
    tux_p = hi_dy.select(dy, zero_n).binary(String("mul"), hi_dy.select(ux, zero_n), ops["mul"])
    tuy_p = hi_dx.select(dx, zero_n).binary(String("mul"), hi_dx.select(uy, zero_n), ops["mul"])
    plain_sum = tsh_p.binary(String("add"), tux_p, ops["add"]).binary(String("add"), tuy_p, ops["add"])
    tc = hi.binary(String("mul"), sc.binary(String("div"), scale_out, ops["div"]), ops["mul"])
    tsh_h = ovf_p.select(zero_n, n).binary(
        String("mul"), dbig.binary(String("div"), scale_out, ops["div"]), ops["mul"]).binary(
        String("mul"), dsml, ops["mul"])
    tux_h = hi_dy.select(zero_n, dy).binary(String("div"), scale_out, ops["div"]).binary(
        String("mul"), ux, ops["mul"])
    tuy_h = hi_dx.select(zero_n, dx).binary(String("div"), scale_out, ops["div"]).binary(
        String("mul"), uy, ops["mul"])
    hi_sum = tsh_h.binary(String("add"), tux_h, ops["add"]).binary(String("add"), tuy_h, ops["add"])
    made = [
        rewrite(_m_zero_anchor(ref, a0, a1)).to(
            seen0.select(a1, seen1.select(a0, z_both)),
            *fill_bind, *_need(lit_binds, "zero", "fmax"), *_need(binds, "abs", "ge", "min", "max")),
        rewrite(_m_tile_anchor(ref, x, axes)).to(
            z_tile, *fill_bind, *lit_binds["zero"], *_need(binds, "min", "max"),
            eq(p_min).to(reduce_min_on(axes)), eq(p_max).to(reduce_max_on(axes))),
        rewrite(_m_pair_scale(ref, dx, dy)).to(
            need_sc.select(one_n, ab), *fill_bind, *_need(lit_binds, "one", "qmax", "sqmax"),
            *_need(binds, "abs", "ge", "gt", "and", "or", "max", "div")),
    ]
    for sign_i, op in ((1, "add"), (-1, "sub")):
        s_ops = ("abs", "ge", "gt", "and", "max", "add", "mul", "div") if sign_i == 1 else (
            "abs", "ge", "gt", "and", "max", "add", "sub", "mul", "div")
        made += [
            rewrite(_m_shift_plain(ref, c, n, dx, dy, ux, uy, i64(sign_i))).to(
                c.binary(String(op), plain_sum, ops[op]),
                *fill_bind, *_need(lit_binds, "zero", "one", "qmax", "sqmax"), *_need(binds, *s_ops)),
            rewrite(_m_shift_hi(ref, hi, sc, n, dx, dy, ux, uy, scale_out, i64(sign_i))).to(
                tc.binary(String(op), hi_sum, ops[op]),
                *fill_bind, *_need(lit_binds, "zero", "one", "qmax", "sqmax"), *_need(binds, *s_ops)),
        ]
    return made


def _moment_ssa_combine_u_rules(ops: dict[str, i64], binds: dict[str, tuple]) -> list:
    """Linear SSA rules combining normalized first-moment residuals across zero-mass, single-sided, and
    two-sided boundaries."""
    ref, n0, ax0, ux0, n1, ax1, ux1, u_zero = vars_("ref n0 ax0 ux0 n1 ax1 ux1 u_zero", Term)
    either, both, extreme, n_min, n_drop, safe_max = vars_("either both extreme n_min n_drop safe_max", Term)
    kept_a, drop_a, kept_u, drop_u, p_kept, p_drop = vars_("kept_a drop_a kept_u drop_u p_kept p_drop", Term)
    _, consts, fill_bind, lit_binds = _moment_ref_consts(ref)
    zero_n, one_n, fmax_n = consts["zero"], consts["one"], consts["fmax"]
    k0_p, either_p, both_p, ext_p, smax_p, p0_p, p1_p, nmin_p, nd_p = _moment_combine_parts(ref, n0, n1, ops)
    merged_a = _m_zero_anchor(ref, ax0, ax1)
    seen0 = fmax_n.compare(String("ge"), ax0.unary(String("abs"), ops["abs"]), ops["ge"])
    seen1 = fmax_n.compare(String("ge"), ax1.unary(String("abs"), ops["abs"]), ops["ge"])
    seen_n = fmax_n.compare(String("ge"), merged_a.unary(String("abs"), ops["abs"]), ops["ge"])
    a0_g = seen0.select(zero_n, ax0)
    a1_g = seen1.select(zero_n, ax1)
    an_g = seen_n.select(zero_n, merged_a)
    d0_z = either.select(seen0.select(zero_n, a0_g.binary(String("sub"), an_g, ops["sub"])), zero_n)
    d1_z = either.select(seen1.select(zero_n, a1_g.binary(String("sub"), an_g, ops["sub"])), zero_n)
    u_zero_rhs = ux0.binary(String("add"), ux1, ops["add"]).binary(
        String("add"),
        either.select(n0, zero_n).binary(String("mul"), d0_z, ops["mul"]).binary(
            String("add"), either.select(n1, zero_n).binary(String("mul"), d1_z, ops["mul"]), ops["add"]),
        ops["add"])
    d_eff = _m_diff(String("val"), ref, both.select(zero_n, kept_a), both.select(zero_n, drop_a),
                    zero_n, both.select(zero_n, drop_u))
    big_u = _m_diff(String("big"), ref, both.select(zero_n, kept_a), both.select(zero_n, drop_a),
                    zero_n, both.select(zero_n, drop_u))
    shift_norm = both.select(zero_n, _m_double_if(big_u, p_drop).binary(String("mul"), d_eff, ops["mul"]))
    shift_ext = _m_ext_mul(ref, extreme, _m_double_if(big_u, n_min), safe_max, d_eff)
    base_u = extreme.select(p_kept.binary(String("mul"), kept_u, ops["mul"]), kept_u)
    u_both = base_u.binary(String("add"), extreme.select(shift_norm, shift_ext), ops["add"])
    smax_1, small_s, smax_hi, smax_lo, n_hi, n_lo = _moment_one_side_scale(
        either, both, safe_max, n_drop, zero_n, one_n, ops)
    seen_d = fmax_n.compare(String("ge"), drop_a.unary(String("abs"), ops["abs"]), ops["ge"])

    def g1(term: Term) -> Term:
        return either.select(zero_n, both.select(seen_d.select(zero_n, term), zero_n))

    d_zero = _m_diff(String("val"), ref, g1(kept_a), g1(drop_a), zero_n, zero_n)
    big_z = _m_diff(String("big"), ref, g1(kept_a), g1(drop_a), zero_n, zero_n)
    u_shift_hi = _m_double_if(big_z, n_hi).binary(String("mul"), d_zero, ops["mul"]).binary(
        String("div"), smax_hi, ops["div"])
    u_shift_lo = _m_double_if(big_z, n_lo.binary(String("div"), smax_lo, ops["div"])).binary(
        String("mul"), d_zero, ops["mul"])
    u_one = p_kept.binary(String("mul"), kept_u, ops["mul"]).binary(
        String("add"), g1(drop_u).binary(String("div"), smax_1, ops["div"]), ops["add"]).binary(
        String("add"), small_s.select(u_shift_hi, u_shift_lo), ops["add"])
    return [
        rewrite(_m_combine_zero_u(ref, either, n0, ax0, ux0, n1, ax1, ux1)).to(
            u_zero_rhs, *fill_bind, *_need(lit_binds, "zero", "fmax"),
            *_need(binds, "abs", "ge", "add", "sub", "mul")),
        rewrite(_m_combine_u(ref, n0, ax0, ux0, n1, ax1, ux1)).to(
            _m_combine_u_body(ref, either_p, both_p, ext_p, nmin_p, nd_p, smax_p, k0_p.select(ax1, ax0),
                              k0_p.select(ax0, ax1), k0_p.select(ux1, ux0), k0_p.select(ux0, ux1),
                              k0_p.select(p1_p, p0_p), k0_p.select(p0_p, p1_p),
                              _m_combine_zero_u(ref, either_p, n0, ax0, ux0, n1, ax1, ux1)),
            *_need(binds, "min")),
        rewrite(_m_combine_u_body(ref, either, both, extreme, n_min, n_drop, safe_max, kept_a, drop_a,
                                  kept_u, drop_u, p_kept, p_drop, u_zero)).to(
            either.select(u_zero, both.select(u_one, u_both)),
            *fill_bind, *_need(lit_binds, "zero", "one", "fmax"),
            *_need(binds, "abs", "ge", "gt", "add", "mul", "div")),
    ]


def _moment_ssa_combine_one_c_rules(ops: dict[str, i64], binds: dict[str, tuple]) -> list:
    """Linear SSA rules for one-sided zero-mass recentering of plain and scaled central second moments."""
    ref, either, both, safe_max = vars_("ref either both safe_max", Term)
    ck_v, cd_v, hi_drop, s_drop, p_kept, n_drop = vars_("ck_v cd_v hi_drop s_drop p_kept n_drop", Term)
    dx_eff, bx, dy_eff, by, drop_ux, drop_uy = vars_("dx_eff bx dy_eff by drop_ux drop_uy", Term)
    _, consts, fill_bind, lit_binds = _moment_ref_consts(ref)
    zero_n, one_n, half_n, qmax_n = consts["zero"], consts["one"], consts["half"], consts["qmax"]
    _, small_s, smax_hi, smax_lo, n_hi, n_lo = _moment_one_side_scale(
        either, both, safe_max, n_drop, zero_n, one_n, ops)

    def g1(term: Term) -> Term:
        return either.select(zero_n, both.select(term, zero_n))

    def scale_in(term: Term) -> Term:
        g = g1(term)
        return small_s.select(g, g.binary(String("div"), smax_lo, ops["div"]))

    c_plain_out = scale_in(cd_v).binary(String("div"), smax_hi, ops["div"])
    hi_in = scale_in(hi_drop)
    n_in = small_s.select(n_hi, n_lo.binary(String("div"), smax_lo, ops["div"]))
    ux_in, uy_in = scale_in(drop_ux), scale_in(drop_uy)
    adx, ady = dx_eff.unary(String("abs"), ops["abs"]), dy_eff.unary(String("abs"), ops["abs"])
    x_ge_y = adx.compare(String("ge"), ady, ops["ge"])
    d_big, d_sml = x_ge_y.select(dy_eff, dx_eff), x_ge_y.select(dx_eff, dy_eff)
    sc_1 = s_drop.binary(String("max"), _m_pair_scale(ref, dx_eff, dy_eff), ops["max"])
    qtr_n = half_n.binary(String("mul"), half_n, ops["mul"])
    k_pad = sc_1.compare(String("gt"), qmax_n, ops["gt"]).select(one_n, qtr_n)
    h_x, h_y = bx.select(one_n, half_n), by.select(one_n, half_n)
    k_c = k_pad.binary(String("mul"), h_x, ops["mul"]).binary(String("mul"), h_y, ops["mul"])
    k_ux, k_uy = k_pad.binary(String("mul"), h_x, ops["mul"]), k_pad.binary(String("mul"), h_y, ops["mul"])
    term_hi = hi_in.binary(String("mul"), s_drop.binary(String("mul"), k_c, ops["mul"]).binary(
        String("div"), sc_1, ops["div"]), ops["mul"])
    term_shift = n_in.binary(String("mul"), d_big.binary(String("mul"), k_pad, ops["mul"]).binary(
        String("div"), sc_1, ops["div"]), ops["mul"]).binary(String("mul"), d_sml, ops["mul"])
    dy_pad = dy_eff.binary(String("mul"), k_ux, ops["mul"])
    dx_pad = dx_eff.binary(String("mul"), k_uy, ops["mul"])
    big_dy = ady.binary(String("mul"), k_ux, ops["mul"]).compare(String("gt"), one_n, ops["gt"])
    big_dx = adx.binary(String("mul"), k_uy, ops["mul"]).compare(String("gt"), one_n, ops["gt"])
    term_ux = big_dy.select(
        ux_in.binary(String("div"), sc_1, ops["div"]).binary(String("mul"), dy_pad, ops["mul"]),
        dy_pad.binary(String("div"), sc_1, ops["div"]).binary(String("mul"), ux_in, ops["mul"]))
    term_uy = big_dx.select(
        uy_in.binary(String("div"), sc_1, ops["div"]).binary(String("mul"), dx_pad, ops["mul"]),
        dx_pad.binary(String("div"), sc_1, ops["div"]).binary(String("mul"), uy_in, ops["mul"]))
    # Plain zero-mass band bypasses `sc_1` completely so large shifts never underflow small terms; the
    # scaled band cancels against shift/cross terms before unscaling.
    num_pos = term_hi.binary(String("sub"), term_ux, ops["sub"]).binary(
        String("add"), term_shift.binary(String("sub"), term_uy, ops["sub"]), ops["add"])
    num_neg = term_hi.binary(String("add"), term_ux, ops["add"]).binary(
        String("add"), term_uy.binary(String("sub"), term_shift, ops["sub"]), ops["add"])
    one_ops = ("abs", "ge", "gt", "max", "add", "sub", "mul", "div")
    made = []
    for sign_i, num_s in ((1, num_pos), (-1, num_neg)):
        unscaled = _m_double_if(by, _m_double_if(bx, num_s.binary(String("div"), k_pad, ops["div"]))).binary(
            String("div"), smax_hi, ops["div"]).binary(String("mul"), sc_1, ops["mul"])
        c_one_rhs = p_kept.binary(String("mul"), ck_v, ops["mul"]).binary(
            String("add"), c_plain_out.binary(String("add"), unscaled, ops["add"]), ops["add"])
        made.append(rewrite(_m_combine_one_c(
            ref, either, both, safe_max, ck_v, cd_v, hi_drop, s_drop, p_kept, n_drop, dx_eff, bx, dy_eff,
            by, drop_ux, drop_uy, i64(sign_i))).to(
            c_one_rhs, *fill_bind, *_need(lit_binds, "zero", "one", "half", "qmax"),
            *_need(binds, *one_ops)))
    return made


def _moment_ssa_combine_zero_rules(ops: dict[str, i64], binds: dict[str, tuple]) -> list:
    """Linear SSA rules combining zero-mass plain band, scaled overflow band, and scale slot."""
    ref, c0, hi0, sc0, n0, ax0, ux0, ay0, uy0 = vars_("ref c0 hi0 sc0 n0 ax0 ux0 ay0 uy0", Term)
    c1, hi1, sc1, n1, ax1, ux1, ay1, uy1 = vars_("c1 hi1 sc1 n1 ax1 ux1 ay1 uy1", Term)
    either = var("either", Term)
    _, consts, fill_bind, lit_binds = _moment_ref_consts(ref)
    zero_n, one_n, fmax_n = consts["zero"], consts["one"], consts["fmax"]
    either_p = _m_part(String("either"), ref, n0, n1)
    merged_ax, merged_ay = _m_zero_anchor(ref, ax0, ax1), _m_zero_anchor(ref, ay0, ay1)
    seen0 = fmax_n.compare(String("ge"), ax0.unary(String("abs"), ops["abs"]), ops["ge"])
    seen1 = fmax_n.compare(String("ge"), ax1.unary(String("abs"), ops["abs"]), ops["ge"])
    seen_n = fmax_n.compare(String("ge"), merged_ax.unary(String("abs"), ops["abs"]), ops["ge"])
    dax0 = seen0.select(zero_n, seen0.select(zero_n, ax0).binary(
        String("sub"), seen_n.select(zero_n, merged_ax), ops["sub"]))
    day0 = seen0.select(zero_n, seen0.select(zero_n, ay0).binary(
        String("sub"), seen_n.select(zero_n, merged_ay), ops["sub"]))
    dax1 = seen1.select(zero_n, seen1.select(zero_n, ax1).binary(
        String("sub"), seen_n.select(zero_n, merged_ax), ops["sub"]))
    day1 = seen1.select(zero_n, seen1.select(zero_n, ay1).binary(
        String("sub"), seen_n.select(zero_n, merged_ay), ops["sub"]))
    s_sh = _m_pair_scale(ref, dax0, day0).binary(String("max"), _m_pair_scale(ref, dax1, day1), ops["max"])
    s_01 = sc0.binary(String("max"), sc1, ops["max"]).binary(String("max"), s_sh, ops["max"])
    made = [
        rewrite(_m_combine_s(ref, n0, sc0, ax0, ay0, n1, sc1, ax1, ay1)).to(
            either_p.select(s_01, one_n), *fill_bind, *_need(lit_binds, "zero", "one", "fmax"),
            *_need(binds, "abs", "ge", "sub", "max")),
    ]
    for sign_i in (1, -1):
        c_zero_rhs = _m_shift_plain(
            ref, either.select(c0, zero_n), either.select(n0, zero_n), either.select(dax0, zero_n),
            either.select(day0, zero_n), either.select(ux0, zero_n), either.select(uy0, zero_n),
            i64(sign_i)).binary(
            String("add"),
            _m_shift_plain(ref, either.select(c1, zero_n), either.select(n1, zero_n),
                           either.select(dax1, zero_n), either.select(day1, zero_n),
                           either.select(ux1, zero_n), either.select(uy1, zero_n), i64(sign_i)),
            ops["add"])
        hi_zero_rhs = _m_shift_hi(
            ref, either_p.select(hi0, zero_n), sc0, either_p.select(n0, zero_n),
            either_p.select(dax0, zero_n), either_p.select(day0, zero_n), either_p.select(ux0, zero_n),
            either_p.select(uy0, zero_n), s_01, i64(sign_i)).binary(
            String("add"),
            _m_shift_hi(ref, either_p.select(hi1, zero_n), sc1, either_p.select(n1, zero_n),
                        either_p.select(dax1, zero_n), either_p.select(day1, zero_n),
                        either_p.select(ux1, zero_n), either_p.select(uy1, zero_n), s_01, i64(sign_i)),
            ops["add"])
        made += [
            rewrite(_m_combine_zero_c(ref, either, c0, n0, ax0, ux0, ay0, uy0, c1, n1, ax1, ux1, ay1,
                                      uy1, i64(sign_i))).to(
                c_zero_rhs, *fill_bind, *_need(lit_binds, "zero", "fmax"),
                *_need(binds, "abs", "ge", "sub", "add")),
            rewrite(_m_combine_hi(ref, hi0, sc0, n0, ax0, ux0, ay0, uy0, hi1, sc1, n1, ax1, ux1, ay1,
                                  uy1, i64(sign_i))).to(
                either_p.select(hi_zero_rhs, zero_n), *fill_bind, *_need(lit_binds, "zero", "fmax"),
                *_need(binds, "abs", "ge", "sub", "max", "add")),
        ]
    return made


def _moment_ssa_combine_c_rules(ops: dict[str, i64], binds: dict[str, tuple]) -> list:
    """Linear SSA rules combining normalized central second moments across all mass regimes."""
    ref, c0, hi0, sc0, n0, ax0, ux0, ay0, uy0 = vars_("ref c0 hi0 sc0 n0 ax0 ux0 ay0 uy0", Term)
    c1, hi1, sc1, n1, ax1, ux1, ay1, uy1 = vars_("c1 hi1 sc1 n1 ax1 ux1 ay1 uy1", Term)
    both, extreme, keep0, n_min, safe_max = vars_("both extreme keep0 n_min safe_max", Term)
    p0, p1, p_prod, dx_eff, big_x, dy_eff, big_y = vars_("p0 p1 p_prod dx_eff big_x dy_eff big_y", Term)
    k_min, eff_p, big_d, small_d = vars_("k_min eff_p big_d small_d", Term)
    _, consts, fill_bind, lit_binds = _moment_ref_consts(ref)
    zero_n, fmax_n = consts["zero"], consts["fmax"]
    k0_p, either_p, both_p, ext_p, smax_p, p0_p, p1_p, nmin_p, nd_p = _moment_combine_parts(ref, n0, n1, ops)
    c_kept, c_drop = keep0.select(c1, c0), keep0.select(c0, c1)
    pk_c, pd_c = keep0.select(p1, p0), keep0.select(p0, p1)
    base_norm = pk_c.binary(String("mul"), c_kept, ops["mul"]).binary(
        String("add"), pd_c.binary(String("mul"), c_drop, ops["mul"]), ops["add"])
    base_ext = c_kept.binary(
        String("add"), _m_ext_mul(ref, extreme, n_min, safe_max, c_drop), ops["add"])
    x_ge_y = dx_eff.unary(String("abs"), ops["abs"]).compare(
        String("ge"), dy_eff.unary(String("abs"), ops["abs"]), ops["ge"])
    corr_norm = both.select(zero_n, eff_p).binary(
        String("mul"), both.select(zero_n, big_d), ops["mul"]).binary(
        String("mul"), both.select(zero_n, small_d), ops["mul"])
    c_base = _m_combine_base(ref, ext_p, k0_p, nmin_p, smax_p, c0, c1, p0_p, p1_p)
    c_corr = _m_combine_corr(
        ref, both_p, ext_p, nmin_p, smax_p, p0_p.binary(String("mul"), p1_p, ops["mul"]),
        _m_diff(String("val"), ref, both_p.select(zero_n, ax0), both_p.select(zero_n, ax1),
                both_p.select(zero_n, ux0), both_p.select(zero_n, ux1)),
        _m_diff(String("big"), ref, both_p.select(zero_n, ax0), both_p.select(zero_n, ax1),
                both_p.select(zero_n, ux0), both_p.select(zero_n, ux1)),
        _m_diff(String("val"), ref, both_p.select(zero_n, ay0), both_p.select(zero_n, ay1),
                both_p.select(zero_n, uy0), both_p.select(zero_n, uy1)),
        _m_diff(String("big"), ref, both_p.select(zero_n, ay0), both_p.select(zero_n, ay1),
                both_p.select(zero_n, uy0), both_p.select(zero_n, uy1)))
    kax, dax = k0_p.select(ax1, ax0), k0_p.select(ax0, ax1)
    kay, day = k0_p.select(ay1, ay0), k0_p.select(ay0, ay1)
    kux, kuy = k0_p.select(ux1, ux0), k0_p.select(uy1, uy0)
    seen_d = fmax_n.compare(String("ge"), dax.unary(String("abs"), ops["abs"]), ops["ge"])

    def g1p(term: Term) -> Term:
        return either_p.select(zero_n, both_p.select(seen_d.select(zero_n, term), zero_n))

    dx1_v = _m_diff(String("val"), ref, g1p(dax), g1p(kax), zero_n, g1p(kux))
    dx1_b = _m_diff(String("big"), ref, g1p(dax), g1p(kax), zero_n, g1p(kux))
    dy1_v = _m_diff(String("val"), ref, g1p(day), g1p(kay), zero_n, g1p(kuy))
    dy1_b = _m_diff(String("big"), ref, g1p(day), g1p(kay), zero_n, g1p(kuy))
    made = [
        rewrite(_m_combine_base(ref, extreme, keep0, n_min, safe_max, c0, c1, p0, p1)).to(
            extreme.select(base_norm, base_ext), *_need(binds, "add", "mul")),
        rewrite(_m_combine_corr(ref, both, extreme, n_min, safe_max, p_prod, dx_eff, big_x,
                                dy_eff, big_y)).to(
            _m_combine_corr_body(ref, both, extreme, _m_double_if(big_y, _m_double_if(big_x, n_min)),
                                 safe_max, _m_double_if(big_y, _m_double_if(big_x, p_prod)),
                                 x_ge_y.select(dy_eff, dx_eff), x_ge_y.select(dx_eff, dy_eff)),
            *_need(binds, "abs", "ge")),
        rewrite(_m_combine_corr_body(ref, both, extreme, k_min, safe_max, eff_p, big_d, small_d)).to(
            extreme.select(corr_norm, _m_ext_corr(ref, extreme, k_min, safe_max, big_d, small_d)),
            *fill_bind, *lit_binds["zero"], *_need(binds, "mul")),
    ]
    for sign_i, op in ((1, "add"), (-1, "sub")):
        c_one_call = _m_combine_one_c(
            ref, either_p, both_p, smax_p, k0_p.select(c1, c0), k0_p.select(c0, c1), k0_p.select(hi0, hi1),
            k0_p.select(sc0, sc1), k0_p.select(p1_p, p0_p), nd_p, dx1_v, dx1_b, dy1_v, dy1_b,
            k0_p.select(ux0, ux1), k0_p.select(uy0, uy1), i64(sign_i))
        c_both_rhs = c_base.binary(String(op), c_corr, ops[op])
        c_ops = ("abs", "ge", "min", "mul", "add") if sign_i == 1 else (
            "abs", "ge", "min", "mul", "add", "sub")
        made.append(rewrite(_m_combine_c(ref, c0, hi0, sc0, n0, ax0, ux0, ay0, uy0, c1, hi1, sc1, n1, ax1,
                                         ux1, ay1, uy1, i64(sign_i))).to(
            either_p.select(_m_combine_zero_c(ref, either_p, c0, n0, ax0, ux0, ay0, uy0, c1, n1, ax1,
                                              ux1, ay1, uy1, i64(sign_i)),
                            both_p.select(c_one_call, c_both_rhs)),
            *fill_bind, *_need(lit_binds, "zero", "fmax"), *_need(binds, *c_ops)))
    return made


def _moment_ssa_finish_rules(ops: dict[str, i64], binds: dict[str, tuple]) -> list:
    """Linear SSA finish rules recentering and unscaling zero-mass partials with positive mass exact."""
    ref, c0, hi0, sc0, n0, ax0, ux0, ay0, uy0 = vars_("ref c0 hi0 sc0 n0 ax0 ux0 ay0 uy0", Term)
    _, consts, fill_bind, lit_binds = _moment_ref_consts(ref)
    zero_n, fmax_n = consts["zero"], consts["fmax"]
    pos_f = n0.compare(String("gt"), zero_n, ops["gt"])

    def z0(term: Term) -> Term:
        return pos_f.select(term, zero_n)

    seen_x = fmax_n.compare(String("ge"), ax0.unary(String("abs"), ops["abs"]), ops["ge"])
    seen_y = fmax_n.compare(String("ge"), ay0.unary(String("abs"), ops["abs"]), ops["ge"])
    ax0_g, ay0_g = seen_x.select(zero_n, ax0), seen_y.select(zero_n, ay0)
    mu_pos = ax0_g.binary(String("add"), ux0, ops["add"])
    mu_zero = z0(ux0).binary(String("add"), z0(n0).binary(String("mul"), z0(ax0_g), ops["mul"]), ops["add"])
    s_f = sc0.binary(String("max"), _m_pair_scale(ref, z0(ax0_g), z0(ay0_g)), ops["max"])
    made = [
        rewrite(_m_finish_u(String("mu"), ref, n0, ax0, ux0)).to(
            pos_f.select(mu_zero, mu_pos), *fill_bind, *_need(lit_binds, "zero", "fmax"),
            *_need(binds, "abs", "ge", "gt", "add", "mul")),
        rewrite(_m_finish_u(String("sx"), ref, n0, ax0, ux0)).to(
            pos_f.select(mu_zero, mu_pos.binary(String("mul"), n0, ops["mul"])),
            *fill_bind, *_need(lit_binds, "zero", "fmax"), *_need(binds, "abs", "ge", "gt", "add", "mul")),
    ]
    for sign_i in (1, -1):
        cov_zero = _m_shift_plain(
            ref, z0(c0), z0(n0), z0(ax0_g), z0(ay0_g), z0(ux0), z0(uy0), i64(sign_i)).binary(
            String("add"),
            _m_shift_hi(ref, z0(hi0), sc0, z0(n0), z0(ax0_g), z0(ay0_g), z0(ux0), z0(uy0), s_f,
                        i64(sign_i)).binary(String("mul"), s_f, ops["mul"]),
            ops["add"])
        made += [
            rewrite(_m_finish_c(String("cov"), ref, c0, hi0, sc0, n0, ax0, ux0, ay0, uy0, i64(sign_i))).to(
                pos_f.select(cov_zero, c0), *fill_bind, *_need(lit_binds, "zero", "fmax"),
                *_need(binds, "abs", "ge", "gt", "max", "add", "mul")),
            rewrite(_m_finish_c(String("sum"), ref, c0, hi0, sc0, n0, ax0, ux0, ay0, uy0, i64(sign_i))).to(
                pos_f.select(cov_zero, c0.binary(String("mul"), n0, ops["mul"])),
                *fill_bind, *_need(lit_binds, "zero", "fmax"),
                *_need(binds, "abs", "ge", "gt", "max", "add", "mul")),
        ]
    return made


def _moment_ssa_tile_scale_rules(ops: dict[str, i64], binds: dict[str, tuple]) -> list:
    """Linear SSA rules detecting per-element zero-mass overflow and its tile scale factor."""
    ref, x, y = vars_("ref x y", Term)
    axes = var("axes", Vec[i64])
    p_max, lift_x, fill_x = vars_("p_max lift_x fill_x", i64)
    lits, consts, fill_bind, lit_binds = _moment_ref_consts(ref)
    zero_n, one_n = consts["zero"], consts["one"]
    one_x = Term.literal(lits["one"]).shaped(String("broadcast_in_dim"), fill_x)
    qmax_x = Term.literal(lits["qmax"]).shaped(String("broadcast_in_dim"), fill_x)
    pos = ref.compare(String("gt"), zero_n, ops["gt"])
    dx_z = x.binary(String("sub"), _m_tile_anchor(ref, x, axes).shaped(String("broadcast_in_dim"), lift_x),
                    ops["sub"])
    dy_z = y.binary(String("sub"), _m_tile_anchor(ref, y, axes).shaped(String("broadcast_in_dim"), lift_x),
                    ops["sub"])
    ax_el, ay_el = dx_z.unary(String("abs"), ops["abs"]), dy_z.unary(String("abs"), ops["abs"])
    big_el = ax_el.binary(String("max"), ay_el, ops["max"])
    sml_el = ax_el.binary(String("min"), ay_el, ops["min"])
    ovf_el = big_el.compare(String("gt"), one_x, ops["gt"]).binary(
        String("and"),
        sml_el.compare(String("gt"), qmax_x.binary(String("div"), big_el.binary(String("max"), one_x,
                                                                                ops["max"]), ops["div"]),
                       ops["gt"]), ops["and"])
    sc_raw = _m_wcov_ovf(ref, x, y, axes, lift_x, fill_x).select(one_x, big_el).reduce(
        String("max"), axes, p_max)
    return [
        rewrite(_m_wcov_ovf(ref, x, y, axes, lift_x, fill_x)).to(
            ovf_el, *_need(lit_binds, "one", "qmax"),
            *_need(binds, "sub", "abs", "max", "min", "gt", "and", "div")),
        rewrite(_m_wcov_scale(ref, x, y, axes, lift_x, fill_x)).to(
            pos.select(sc_raw, one_n), *fill_bind, *_need(lit_binds, "zero", "one"),
            *_need(binds, "sub", "abs", "max", "gt"), eq(p_max).to(reduce_max_on(axes))),
    ]


def _moment_ssa_tile_rules(ops: dict[str, i64], binds: dict[str, tuple]) -> list:
    """Linear SSA rules for tile-local weighted and masked means, residuals, and central second moments."""
    ref, x, y, w, mask, z, dx, dy_s, big_d, small_d, sc_x = vars_(
        "ref x y w mask z dx dy_s big_d small_d sc_x", Term)
    axes = var("axes", Vec[i64])
    p, lift_x, fill_x = vars_("p lift_x fill_x", i64)
    lits, consts, fill_bind, lit_binds = _moment_ref_consts(ref)
    zero_n, one_n = consts["zero"], consts["one"]
    zero_x = Term.literal(lits["zero"]).shaped(String("broadcast_in_dim"), fill_x)
    one_x = Term.literal(lits["one"]).shaped(String("broadcast_in_dim"), fill_x)
    pos = ref.compare(String("gt"), zero_n, ops["gt"])
    pos_x = pos.shaped(String("broadcast_in_dim"), lift_x)
    safe_x = _m_safe_x(ref, lift_x)
    mf = _m_wmul_first(ref, w, lift_x, fill_x)
    pw_w = w.binary(String("div"), safe_x, ops["div"])
    pw_m = mask.select(zero_x, one_x.binary(String("div"), safe_x, ops["div"]))
    arm_mf_avg = mf.select(zero_x, w).binary(String("mul"), x, ops["mul"]).binary(
        String("div"), mf.select(one_x, safe_x), ops["div"])
    arm_df_avg = pw_w.binary(String("mul"), x, ops["mul"])
    ax_w = _m_wmean(String("avg"), ref, x, w, axes, p, lift_x, fill_x)
    ay_w = _m_wmean(String("avg"), ref, y, w, axes, p, lift_x, fill_x)
    ax_m = _m_mmean(String("avg"), ref, x, mask, z, axes, p, lift_x, fill_x)
    ay_m = _m_mmean(String("avg"), ref, y, mask, z, axes, p, lift_x, fill_x)
    x_ge_y = dx.unary(String("abs"), ops["abs"]).compare(
        String("ge"), dy_s.unary(String("abs"), ops["abs"]), ops["ge"])
    big_s = big_d.binary(String("div"), sc_x, ops["div"])
    t1_w = mf.select(zero_x, w).binary(String("mul"), big_s, ops["mul"])
    arm_mf_c = _m_prod_div(zero_x, one_x, t1_w, small_d, mf.select(one_x, safe_x))
    arm_df_c = pw_w.binary(String("mul"), big_s, ops["mul"]).binary(String("mul"), small_d, ops["mul"])
    made = [
        rewrite(_m_safe_x(ref, lift_x)).to(
            pos.select(one_n, ref).shaped(String("broadcast_in_dim"), lift_x),
            *fill_bind, *_need(lit_binds, "zero", "one"), *_need(binds, "gt")),
        rewrite(_m_wmul_first(ref, w, lift_x, fill_x)).to(
            one_x.compare(String("ge"), w, ops["ge"]).binary(
                String("and"), safe_x.compare(String("gt"), one_x, ops["gt"]), ops["and"]),
            *lit_binds["one"], *_need(binds, "ge", "gt", "and")),
        rewrite(_m_wmean(String("raw"), ref, x, w, axes, p, lift_x, fill_x)).to(
            mf.select(arm_df_avg, arm_mf_avg).reduce(String("sum"), axes, p),
            *_need(lit_binds, "zero", "one"), *_need(binds, "mul", "div")),
        rewrite(_m_wmean(String("avg"), ref, x, w, axes, p, lift_x, fill_x)).to(
            pos.select(_m_tile_anchor(ref, x, axes), _m_wmean(String("raw"), ref, x, w, axes, p, lift_x,
                                                             fill_x)),
            *fill_bind, *lit_binds["zero"], *_need(binds, "gt")),
        rewrite(_m_wmean(String("res"), ref, x, w, axes, p, lift_x, fill_x)).to(
            _m_wmean(String("raw"), ref,
                     x.binary(String("sub"), ax_w.shaped(String("broadcast_in_dim"), lift_x), ops["sub"]),
                     w, axes, p, lift_x, fill_x),
            *_need(binds, "sub")),
        rewrite(_m_mmean(String("avg"), ref, x, mask, z, axes, p, lift_x, fill_x)).to(
            pw_m.binary(String("mul"), mask.select(z, x), ops["mul"]).reduce(String("sum"), axes, p),
            *_need(lit_binds, "zero", "one"), *_need(binds, "div", "mul")),
        rewrite(_m_mmean(String("res"), ref, x, mask, z, axes, p, lift_x, fill_x)).to(
            pw_m.binary(String("mul"), mask.select(
                z, x.binary(String("sub"), ax_m.shaped(String("broadcast_in_dim"), lift_x), ops["sub"])),
                ops["mul"]).reduce(String("sum"), axes, p),
            *_need(lit_binds, "zero", "one"), *_need(binds, "div", "mul", "sub")),
        rewrite(_m_wcov_raw(ref, dx, dy_s, sc_x, w, axes, p, lift_x, fill_x)).to(
            _m_wcov_term(ref, x_ge_y.select(dy_s, dx), x_ge_y.select(dx, dy_s), sc_x, w, axes, p, lift_x,
                         fill_x),
            *_need(binds, "abs", "ge")),
        rewrite(_m_wcov_term(ref, big_d, small_d, sc_x, w, axes, p, lift_x, fill_x)).to(
            mf.select(arm_df_c, arm_mf_c).reduce(String("sum"), axes, p),
            *_need(lit_binds, "zero", "one"), *_need(binds, "mul", "div")),
        rewrite(_m_mcov_raw(ref, dx, dy_s, mask, axes, p, lift_x, fill_x)).to(
            pw_m.binary(String("mul"), x_ge_y.select(dy_s, dx), ops["mul"]).binary(
                String("mul"), x_ge_y.select(dx, dy_s), ops["mul"]).reduce(String("sum"), axes, p),
            *_need(lit_binds, "zero", "one"), *_need(binds, "abs", "ge", "div", "mul")),
    ]
    ax_wl, ay_wl = ax_w.shaped(String("broadcast_in_dim"), lift_x), ay_w.shaped(
        String("broadcast_in_dim"), lift_x)
    ax_ml, ay_ml = ax_m.shaped(String("broadcast_in_dim"), lift_x), ay_m.shaped(
        String("broadcast_in_dim"), lift_x)
    ovf_x = _m_wcov_ovf(ref, x, y, axes, lift_x, fill_x)
    sc_tile_x = _m_wcov_scale(ref, x, y, axes, lift_x, fill_x).shaped(String("broadcast_in_dim"), lift_x)
    dx_w = x.binary(String("sub"), ax_wl, ops["sub"])
    dx_plain = pos_x.select(ovf_x.select(dx_w, zero_x), dx_w)
    dx_hi = pos_x.select(ovf_x.select(zero_x, dx_w), zero_x)
    corr_w = pos.select(zero_n, _m_wmean(String("res"), ref, x, w, axes, p, lift_x, fill_x).binary(
        String("mul"), _m_wmean(String("res"), ref, y, w, axes, p, lift_x, fill_x), ops["mul"]))
    corr_m = _m_mmean(String("res"), ref, x, mask, z, axes, p, lift_x, fill_x).binary(
        String("mul"), _m_mmean(String("res"), ref, y, mask, z, axes, p, lift_x, fill_x), ops["mul"])
    for sign_i, op in ((1, "sub"), (-1, "add")):
        dy_w = y.binary(String("sub"), ay_wl, ops["sub"]) if sign_i == 1 else ay_wl.binary(
            String("sub"), y, ops["sub"])
        dy_m = mask.select(z, y.binary(String("sub"), ay_ml, ops["sub"]) if sign_i == 1 else
                           ay_ml.binary(String("sub"), y, ops["sub"]))
        made += [
            rewrite(_m_wcov_tile(String("plain"), ref, x, y, w, axes, p, lift_x, fill_x, i64(sign_i))).to(
                _m_wcov_raw(ref, dx_plain, dy_w, one_x, w, axes, p, lift_x, fill_x).binary(
                    String(op), corr_w, ops[op]),
                *fill_bind, *_need(lit_binds, "zero", "one"), *_need(binds, "gt", "sub", "mul", op)),
            rewrite(_m_wcov_tile(String("hi"), ref, x, y, w, axes, p, lift_x, fill_x, i64(sign_i))).to(
                pos.select(_m_wcov_raw(ref, dx_hi, dy_w, sc_tile_x, w, axes, p, lift_x, fill_x), zero_n),
                *fill_bind, *lit_binds["zero"], *_need(binds, "gt", "sub")),
            rewrite(_m_mcov_tile(ref, x, y, mask, z, axes, p, lift_x, fill_x, i64(sign_i))).to(
                pos.select(zero_n, _m_mcov_raw(
                    ref, mask.select(z, x.binary(String("sub"), ax_ml, ops["sub"])), dy_m, mask,
                    axes, p, lift_x, fill_x).binary(String(op), corr_m, ops[op])),
                *fill_bind, *lit_binds["zero"], *_need(binds, "gt", "sub", "mul", op)),
        ]
    return made


def _moment_first_decoupled(slot) -> list:
    """Step, merge, and finish rules for normalized anchored first-moment reductions `(anchor, ux)`."""
    u, sx, n, x, w, mask, z, d, c, mu = vars_("u sx n x w mask z d c mu", Term)
    axes = var("axes", Vec[i64])
    to_x, to_n, to_c = vars_("to_x to_n to_c", Vec[i64])
    layout = var("layout", String)
    p, p_div, p_lift, a, i_n, i_sx, i_ax = vars_("p p_div p_lift a i_n i_sx i_ax", i64)
    z_n, fill_x, lift_x = vars_("z_n fill_x lift_x", i64)
    shape = rules.shape
    shared = (
        moment_first_of(sx, n, a),
        eq(i_n).to(slot(n)), eq(i_sx).to(slot(sx)), eq(i_ax).to(anchor_slot(sx)),
        eq(z_n).to(zero_of(n)), eq(z_n).to(zero_of(sx)),
        eq(to_n).to(shape(n)), eq(shape(sx)).to(to_n),
    )
    step_extra = (
        eq(sx).to(u.reduce(String("sum"), axes, p)),
        eq(to_x).to(shape(x)), eq(fill_x).to(filled(to_x)), eq(lift_x).to(unreduced(to_x, a)), *shared,
    )
    n0, ax0, ux0 = Term.state(i_n, 0), Term.state(i_ax, 0), Term.state(i_sx, 0)
    n1, ax1 = Term.state(i_n, 1), Term.state(i_ax, 1)
    mu_done = _m_finish_u(String("mu"), n, n0, ax0, ux0)
    mu_lifted = mu_done.shaped(layout, p_lift)
    ax_w = _m_wmean(String("avg"), n, x, w, axes, p, lift_x, fill_x)
    ux_w = _m_wmean(String("res"), n, x, w, axes, p, lift_x, fill_x)
    ax_m = _m_mmean(String("avg"), n, x, mask, z, axes, p, lift_x, fill_x)
    ux_m = _m_mmean(String("res"), n, x, mask, z, axes, p, lift_x, fill_x)
    return [
        rewrite(moment_anchor(sx).stepped()).to(
            _m_part(String("either"), n, n0, n).select(
                _m_zero_anchor(n, ax0, ax_w), _m_part(String("keep0"), n, n0, n).select(ax_w, ax0)),
            moment_first(sx, n, x, w, a), eq(shape(w)).to(to_x), *step_extra),
        rewrite(moment_anchor(sx).stepped()).to(
            _m_part(String("either"), n, n0, n).select(
                _m_zero_anchor(n, ax0, ax_m), _m_part(String("keep0"), n, n0, n).select(ax_m, ax0)),
            masked_moment_first(sx, n, x, mask, z, a), eq(shape(mask)).to(to_x), *step_extra),
        rewrite(sx.stepped()).to(
            _m_combine_u(n, n0, ax0, ux0, n, ax_w, ux_w),
            moment_first(sx, n, x, w, a), eq(shape(w)).to(to_x), *step_extra),
        rewrite(sx.stepped()).to(
            _m_combine_u(n, n0, ax0, ux0, n, ax_m, ux_m),
            masked_moment_first(sx, n, x, mask, z, a), eq(shape(mask)).to(to_x), *step_extra),
        rewrite(moment_anchor(sx).merged()).to(
            _m_part(String("either"), n, n0, n1).select(
                _m_zero_anchor(n, ax0, ax1), _m_part(String("keep0"), n, n0, n1).select(ax1, ax0)),
            *shared),
        rewrite(sx.merged()).to(
            _m_combine_u(n, n0, ax0, ux0, n1, ax1, Term.state(i_sx, 1)), *shared),
        rewrite(sx.finished()).to(_m_finish_u(String("sx"), n, n0, ax0, ux0), *shared),
        rewrite(mu.finished()).to(
            mu_done, eq(mu).to(sx.binary(String("div"), d, p_div)), guarded_count(d, n), *shared),
        rule(eq(mu).to(c.binary(String("div"), d, p_div)), eq(c).to(sx.shaped(layout, p_lift)),
             eq(to_c).to(shape(c)), inserts_axis(p_lift, a), guarded_count_lifted(d, n, a), *shared).then(
            union(mu.finished()).with_(mu_lifted), set_(shape(mu_lifted)).to(to_c)),
    ]


def _moment_second_decoupled(slot) -> list:
    """Step, merge, and finish rules for normalized central second moments `(c, hi, scale)`."""
    t, u, sx, sy, n, x, y, w, mask, z, d, c, cov = vars_("t u sx sy n x y w mask z d c cov", Term)
    axes = var("axes", Vec[i64])
    to_x, to_n, to_c = vars_("to_x to_n to_c", Vec[i64])
    layout = var("layout", String)
    p, p_div, p_lift, a, sign = vars_("p p_div p_lift a sign", i64)
    i, i_hi, i_sc, i_n, i_sx, i_sy, i_ax, i_ay = vars_("i i_hi i_sc i_n i_sx i_sy i_ax i_ay", i64)
    z_n, o_n, fill_n, fill_x, lift_x = vars_("z_n o_n fill_n fill_x lift_x", i64)
    shape = rules.shape
    shared = (
        moment_second_of(t, sx, sy, n, a, sign),
        eq(i).to(slot(t)), eq(i_hi).to(hi_slot(t)), eq(i_sc).to(scale_slot(t)), eq(i_n).to(slot(n)),
        eq(i_sx).to(slot(sx)), eq(i_sy).to(slot(sy)),
        eq(i_ax).to(anchor_slot(sx)), eq(i_ay).to(anchor_slot(sy)),
        eq(z_n).to(zero_of(n)), eq(z_n).to(zero_of(sx)), eq(z_n).to(zero_of(sy)), eq(z_n).to(zero_of(t)),
        eq(to_n).to(shape(n)), eq(shape(sx)).to(to_n), eq(shape(sy)).to(to_n), eq(shape(t)).to(to_n),
    )
    step_extra = (
        eq(t).to(u.reduce(String("sum"), axes, p)),
        eq(to_x).to(shape(x)), eq(shape(y)).to(to_x),
        eq(fill_x).to(filled(to_x)), eq(lift_x).to(unreduced(to_x, a)), *shared,
    )
    zero_n = Term.literal(z_n).shaped(String("broadcast_in_dim"), fill_n)
    one_n = Term.literal(o_n).shaped(String("broadcast_in_dim"), fill_n)
    one_bind = (eq(o_n).to(one_of(n)), eq(fill_n).to(filled(to_n)))
    n0, hi0, sc0 = Term.state(i_n, 0), Term.state(i_hi, 0), Term.state(i_sc, 0)
    ax0, ay0 = Term.state(i_ax, 0), Term.state(i_ay, 0)
    s0 = (Term.state(i, 0), hi0, sc0, n0, ax0, Term.state(i_sx, 0), ay0, Term.state(i_sy, 0))
    s1 = (Term.state(i, 1), Term.state(i_hi, 1), Term.state(i_sc, 1), Term.state(i_n, 1), Term.state(i_ax, 1),
          Term.state(i_sx, 1), Term.state(i_ay, 1), Term.state(i_sy, 1))
    h0, h1 = s0[1:], s1[1:]
    sc_w = _m_wcov_scale(n, x, y, axes, lift_x, fill_x)
    ax_w = _m_wmean(String("avg"), n, x, w, axes, p, lift_x, fill_x)
    ux_w = _m_wmean(String("res"), n, x, w, axes, p, lift_x, fill_x)
    ay_w = _m_wmean(String("avg"), n, y, w, axes, p, lift_x, fill_x)
    uy_w = _m_wmean(String("res"), n, y, w, axes, p, lift_x, fill_x)
    ax_m = _m_mmean(String("avg"), n, x, mask, z, axes, p, lift_x, fill_x)
    ux_m = _m_mmean(String("res"), n, x, mask, z, axes, p, lift_x, fill_x)
    ay_m = _m_mmean(String("avg"), n, y, mask, z, axes, p, lift_x, fill_x)
    uy_m = _m_mmean(String("res"), n, y, mask, z, axes, p, lift_x, fill_x)
    hi_w = _m_wcov_tile(String("hi"), n, x, y, w, axes, p, lift_x, fill_x, sign)
    cov_done = _m_finish_c(String("cov"), n, *s0, sign)
    c_lifted = cov_done.shaped(layout, p_lift)
    return [
        rewrite(moment_scale(t).stepped()).to(
            _m_combine_s(n, n0, sc0, ax0, ay0, n, sc_w, ax_w, ay_w),
            central_moment(t, sx, sy, n, x, y, w, a, sign), eq(shape(w)).to(to_x), *step_extra),
        rewrite(moment_scale(t).stepped()).to(
            _m_combine_s(n, n0, sc0, ax0, ay0, n, one_n, ax_m, ay_m),
            masked_central_moment(t, sx, sy, n, x, y, mask, z, a, sign), eq(shape(mask)).to(to_x),
            *step_extra, *one_bind),
        rewrite(moment_scale(t).merged()).to(
            _m_combine_s(n, n0, sc0, ax0, ay0, Term.state(i_n, 1), Term.state(i_sc, 1),
                         Term.state(i_ax, 1), Term.state(i_ay, 1)), *shared),
        rewrite(moment_hi(t).stepped()).to(
            _m_combine_hi(n, *h0, hi_w, sc_w, n, ax_w, ux_w, ay_w, uy_w, sign),
            central_moment(t, sx, sy, n, x, y, w, a, sign), eq(shape(w)).to(to_x), *step_extra),
        rewrite(moment_hi(t).stepped()).to(
            _m_combine_hi(n, *h0, zero_n, one_n, n, ax_m, ux_m, ay_m, uy_m, sign),
            masked_central_moment(t, sx, sy, n, x, y, mask, z, a, sign), eq(shape(mask)).to(to_x),
            *step_extra, *one_bind),
        rewrite(moment_hi(t).merged()).to(_m_combine_hi(n, *h0, *h1, sign), *shared),
        rewrite(t.stepped()).to(
            _m_combine_c(n, *s0, _m_wcov_tile(String("plain"), n, x, y, w, axes, p, lift_x, fill_x, sign),
                         hi_w, sc_w, n, ax_w, ux_w, ay_w, uy_w, sign),
            central_moment(t, sx, sy, n, x, y, w, a, sign), eq(shape(w)).to(to_x), *step_extra),
        rewrite(t.stepped()).to(
            _m_combine_c(n, *s0, _m_mcov_tile(n, x, y, mask, z, axes, p, lift_x, fill_x, sign), zero_n,
                         one_n, n, ax_m, ux_m, ay_m, uy_m, sign),
            masked_central_moment(t, sx, sy, n, x, y, mask, z, a, sign), eq(shape(mask)).to(to_x),
            *step_extra, *one_bind),
        rewrite(t.merged()).to(_m_combine_c(n, *s0, *s1, sign), *shared),
        rewrite(t.finished()).to(_m_finish_c(String("sum"), n, *s0, sign), *shared),
        rewrite(cov.finished()).to(
            cov_done, eq(cov).to(t.binary(String("div"), d, p_div)), guarded_count(d, n), *shared),
        rule(eq(cov).to(c.binary(String("div"), d, p_div)), eq(c).to(t.shaped(layout, p_lift)),
             eq(to_c).to(shape(c)), inserts_axis(p_lift, a), guarded_count_lifted(d, n, a), *shared).then(
            union(cov.finished()).with_(c_lifted), set_(shape(c_lifted)).to(to_c)),
    ]


def moment_decoupled(slot) -> list:
    """Normalized anchored Chan-Golub-LeVeque step, merge, and finish rules for weighted and masked central
    second moments (`central_moment`, `masked_central_moment`), decomposed into linear SSA egglog helpers so
    each rule RHS stays O(1) in AST size while deriving `(anchor, ux, c, hi, scale)` forms."""
    ops, binds = _moment_ops()
    return [
        *_moment_ssa_prob_rules(ops, binds),
        *_moment_ssa_extreme_rules(ops, binds),
        *_moment_ssa_anchor_rules(ops, binds),
        *_moment_ssa_combine_u_rules(ops, binds),
        *_moment_ssa_combine_one_c_rules(ops, binds),
        *_moment_ssa_combine_zero_rules(ops, binds),
        *_moment_ssa_combine_c_rules(ops, binds),
        *_moment_ssa_finish_rules(ops, binds),
        *_moment_ssa_tile_scale_rules(ops, binds),
        *_moment_ssa_tile_rules(ops, binds),
        *_moment_first_decoupled(slot),
        *_moment_second_decoupled(slot),
    ]


def register_moment_resources(graph: EGraph, region: terms.Region, reductions: tuple[Term, ...],
                              walked) -> None:
    """Register synthetic ops (`comp_gt`, `comp_ge`, `logic_and`, `logic_or`, `min_op`, `abs_op`),
    per-reduction literals (`zero_of`, `one_of`, `half_of`, `quarter_max_of`, `max_of`, `ratio_floor_of`),
    and broadcast/reduction layouts (`filled`, `unreduced`, `reduce_min_on`, `reduce_max_on`) in one batch."""
    facts: list = []
    for primitive, params, setter in (
        (jax.lax.gt_p, {}, comp_gt), (jax.lax.ge_p, {}, comp_ge),
        (jax.lax.and_p, {}, logic_and), (jax.lax.or_p, {}, logic_or),
        (jax.lax.min_p, {}, min_op), (jax.lax.abs_p, {}, abs_op),
    ):
        region.params.append((primitive, params))
        facts.append(set_(setter()).to(i64(len(region.params) - 1)))
    zeros: dict[np.dtype, int] = {}
    ones: dict[np.dtype, int] = {}
    halfs: dict[np.dtype, int] = {}
    qmaxs: dict[np.dtype, int] = {}
    fmaxs: dict[np.dtype, int] = {}
    sqmaxs: dict[np.dtype, int] = {}
    rflrs: dict[np.dtype, int] = {}
    done_fill: set[tuple[int, ...]] = set()
    for reduction in reductions:
        dtype = region.dtypes.get(terms.key(reduction))
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
            if dtype not in zeros:
                float_info = jnp.finfo(dtype)
                fmax_val = np.asarray(float_info.max, dtype)
                qmax_val = fmax_val * np.asarray(0.25, dtype)
                sqmax_val = np.asarray(0.5 * math.sqrt(float(float_info.max)), dtype)
                rflr_val = np.asarray(min(math.sqrt(float(float_info.tiny)),
                                          0.25 * float(float_info.eps)), dtype)
                for table, value in ((zeros, 0.0), (ones, 1.0), (halfs, 0.5), (qmaxs, qmax_val),
                                     (fmaxs, fmax_val), (sqmaxs, sqmax_val), (rflrs, rflr_val)):
                    region.literals.append(np.asarray(value, dtype))
                    table[dtype] = len(region.literals) - 1
                    literal = Term.literal(i64(table[dtype]))
                    region.shaped(literal, ())
                    facts.append(set_(rules.shape(literal)).to(Vec[i64]()))
            facts.extend((set_(zero_of(reduction)).to(i64(zeros[dtype])),
                          set_(one_of(reduction)).to(i64(ones[dtype])),
                          set_(half_of(reduction)).to(i64(halfs[dtype])),
                          set_(quarter_max_of(reduction)).to(i64(qmaxs[dtype])),
                          set_(max_of(reduction)).to(i64(fmaxs[dtype])),
                          set_(sqrt_max_of(reduction)).to(i64(sqmaxs[dtype])),
                          set_(ratio_floor_of(reduction)).to(i64(rflrs[dtype]))))
        to = tuple(region.shape(reduction))
        if to not in done_fill:
            done_fill.add(to)
            region.params.append((jax.lax.broadcast_in_dim_p,
                                  {"shape": to, "broadcast_dimensions": (), "sharding": None}))
            facts.append(set_(filled(Vec[i64](*(i64(size) for size in to)))).to(
                i64(len(region.params) - 1)))
    done_unreduced: set[tuple[tuple[int, ...], int]] = set()
    done_axes: set[int] = set()
    for term, extent in list(region.shapes):
        axis_val = graph.lookup_function_value(walked(term))
        if axis_val is None:
            continue
        axis = terms.integer(axis_val)
        to = tuple(extent)
        if axis not in done_axes:
            done_axes.add(axis)
            region.params.append((jax.lax.reduce_min_p, {"axes": (axis,), "out_sharding": None}))
            facts.append(set_(reduce_min_on(Vec[i64](i64(axis)))).to(i64(len(region.params) - 1)))
            region.params.append((jax.lax.reduce_max_p, {"axes": (axis,), "out_sharding": None}))
            facts.append(set_(reduce_max_on(Vec[i64](i64(axis)))).to(i64(len(region.params) - 1)))
        if to not in done_fill:
            done_fill.add(to)
            region.params.append((jax.lax.broadcast_in_dim_p,
                                  {"shape": to, "broadcast_dimensions": (), "sharding": None}))
            facts.append(set_(filled(Vec[i64](*(i64(size) for size in to)))).to(
                i64(len(region.params) - 1)))
        if (to, axis) in done_unreduced:
            continue
        done_unreduced.add((to, axis))
        dims = tuple(p for p in range(len(to)) if p != axis)
        region.params.append((jax.lax.broadcast_in_dim_p,
                              {"shape": to, "broadcast_dimensions": dims, "sharding": None}))
        facts.append(set_(unreduced(Vec[i64](*(i64(size) for size in to)), i64(axis))).to(
            i64(len(region.params) - 1)))
    if facts:
        graph.register(*facts)
