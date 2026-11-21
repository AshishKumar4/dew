"""The convex mean (docs/design/algebra.md §6.12): finished states joined, finished and resumed by one law
over scaled values (`dew.algebra.scaled`), its invariants held by the construct rather than by a case of its
operands'.

A finished state of a row is `(o, m, l)`: its output `o`, a mean over its channels, its anchor `m` and its
total `l` at that anchor. Finished states of disjoint walks join as their outputs' convex combination, each
weighed by its mass `w_k = l_k f_k` at the joined anchor `M = max m_k` (`f_k` the walk's own factor from
`m_k` to `M`), over the masses' sum `W`:

    merged((o_k, m_k, l_k)_k) = (sum_k w_k o_k / W, M, W)

The law is one ratio of sums of products of its float operands (`ratio_p`, `Ratio`): `sum_k o_k l_k f_k /
sum_k l_k f_k` (`s` the units' scale, `Units`: `f_k = exp(s (m_k - M))`), each term formed exactly as a
double float at a power of two (`_product`), the terms summed compensated at the power of two that puts the
row's largest under the format's largest (`_summed`), the quotient's residual carried (`_quotient`),
landed once, so no mass, no share and no gap is a float apart from the sum it is in, and terms past the
format that cancel cancel. The mean is held to the hull of the outputs of positive mass (its value alone).
A ratio's derivative is a ratio again, over its operands, their tangents and its own value,

    d (N / D) = (dN - R dD) / D:   d o = sum_k (do_k l_k f_k + (o_k - o)(dl_k f_k + s l_k f_k dm_k)) / W

and its transpose too (`_ratio_jvp`, `_ratio_transpose`), so every order of the law, forward and reverse,
is one such ratio: no tangent is a float formed apart from its sum, the class the degenerate sweep found in
the scaled tangents before (a coefficient's tangent flushed or overflowed beside the terms it cancels
against, `tests/test_algebra_scaled_edges.py`). An order reads the order beneath it as its landed value
(`R`, one rounding), so a derivative is within its terms' compensated bound, `u |d| + gamma sum |terms| /
|W|`, plus `u |R| |dW / W|` an order. The anchor `M` reads no tangent: the mean is the same at any anchor.

The emitted arithmetic's error (the proof, algebra.md §6.12): each term is within `gamma_m` of its real value,
`m` its roundings (the factor's, `Units`, the significand product by the total, the quotient, the product
by `o_k` and its landing,
exact but where it falls beneath the least normal, `2^-277` of the row's largest);
the compensated sum is within `u |S| + gamma_(K-1) sum |e_i|` of the exact sum of the rounded terms, `e_i`
its partial sums' exact errors; the real terms' sum lies within the outputs' hull. So the sum is within the
hull widened by `gamma_m max |o_k| + u |S| + gamma_(K-1) sum |e_i|`, and the hull clip moves it by no more.
Exactly opposite terms (equal masses, exactly opposite outputs) round to exactly opposite floats, whose
errors in the compensated sum are exact, so the rest's sum is one rounding of itself in any order: V4's
`t / 3` beside `+-L / 3`."""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from dew.algebra import scaled
from dew.algebra.ratios import Ratio, _ratio_sum, ratio
from dew.algebra.windows import Units, _arrays, _laid_to, _landed, _quotient, _streamed, _Sum, _without_axis
from dew.nn.derivatives import contracted
from dew.nn.precision import two_product, two_sum


def _laid(array: jax.Array, like: jax.Array) -> jax.Array:
    """`array`, a row's value, laid along `like`'s trailing channels."""
    return array.reshape(array.shape + (1,) * (like.ndim - array.ndim))



def _weighed(totals, anchors, axis):
    """The one policy of a state's weight, the merged output's and total's alike: the states of positive
    mass, each total and anchor as the law weighs it, and their joined anchor (the largest of the states
    holding mass, held: the law is the same at any anchor). A total of zero is kept as it is, its own (the
    sum of the masses moves along it, a derivative's), a negative or non-finite one zero, and an empty
    walk's (its anchor not finite) moves nothing; an anchor its own where finite (a zero total's factor its
    own, the derivative along its total, base review F2), else the joined one (an empty walk's, -inf, its
    factor 1 beside its total of none)."""
    return _weights(totals, anchors, _settled(_unsettled(totals, anchors, axis)))


def _unsettled(totals, anchors, axis: int) -> jax.Array:
    """A merge's joined anchor before it is settled (`_joined_anchor`), its totals and anchors laid together
    (anchors common to the states beside totals of a batch, or the other way)."""
    laid = jnp.broadcast_shapes(jnp.shape(totals), jnp.shape(anchors))
    totals, anchors = jnp.broadcast_to(totals, laid), jnp.broadcast_to(anchors, laid)
    return _joined_anchor(jnp.max(jnp.where(_mass(totals)[0], anchors, -jnp.inf), axis=axis, keepdims=True),
                          jnp.max(jnp.where(jnp.isfinite(anchors), anchors, -jnp.inf), axis=axis,
                                  keepdims=True))


def _joined_anchor(massive: jax.Array, finite: jax.Array) -> jax.Array:
    """A merge's joined anchor, chosen inside the law before any mass or tangent is taken at it: the largest
    anchor of its states holding mass (`massive`), and of a row none holds mass in, the largest of its
    states' finite anchors (`finite`), the anchor its walk's start holds; -inf where none is finite. Chosen
    outside the law, an empty row's mass and tangent were taken at one anchor and the row returned at
    another (the merge P0's base review 10, F1: -exp(2) for -exp(1))."""
    return jnp.where(jnp.isfinite(massive), massive, finite)


def _row_anchor(landing: jax.Array, joined: jax.Array) -> jax.Array:
    """The anchor a merged row returns (`merged_total`, `merged_state`): its landing (`_landing`), but -inf
    where no state has a finite anchor (`joined`, before `_settled`), as its walk's start holds: an empty
    walk's, whose mass moves with no total."""
    return jnp.where(jnp.isfinite(joined), landing, -jnp.inf)


def _settled(joined: jax.Array) -> jax.Array:
    """The joined anchor (`_weighed`'s) as the law holds it: the largest finite anchor of a state holding
    mass, 0 where none holds any."""
    return jax.lax.stop_gradient(jnp.where(jnp.isfinite(joined), joined, 0.0))


def holding(totals: jax.Array) -> jax.Array:
    """Where each total holds mass (`_mass`): the one predicate a state's reader beside the law takes
    (`attention_algebra.initial_state`)."""
    return _mass(totals)[0]


def _mass(totals: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Where each total holds mass (positive, its infinity and a subnormal included; a NaN none) and where it
    is one the law weighs (that, or a zero of either sign: its own, the masses' sum moving along it), read
    from its significand as the law's split reads it (`scaled.scaled`: a subnormal's normalized from its
    bits, a zero's +0), a normal float a comparison reads whole. The total compared itself on a lane that
    flushes a subnormal operand (XLA's CPU) reads as a zero of its sign, so `total > 0` dropped a subnormal
    mass (a finished state of total `2^-149` divided by one) and `total >= 0` weighed a negative subnormal
    one, which the split reads as negative (`2^-125` beside `-2^-127` merged to `3 2^-127`: the merge P0's
    head review 2, both); its stored bits compared within a fused program, XLA:CPU's LLVM folds into that
    same comparison. The one reading of a state's mass: the law's weights (`_weights`), its joined anchor
    (`_unsettled`), a chunked merge's (`merged_state_chunked`) and a finished state's (`finished`)."""
    # A predicate, no value's derivative: the split carries no derivative rule (`scaled`'s account).
    significand = scaled.scaled(jax.lax.stop_gradient(totals)).significand
    holds = significand > 0
    return holds, holds | (significand == 0)


def _weights(totals, anchors, joined):
    """Each state's weight given the `joined` anchor (`_weighed`'s): where it holds mass, its total and its
    anchor as the law weighs them (a state at a time, so a chunk's are its own, `merged_state_chunked`)."""
    holds, weighed = _mass(totals)
    placed = holds | jnp.isfinite(anchors)
    totals = jnp.where(weighed, totals, 0.0)
    totals = jnp.where(placed, totals, jax.lax.stop_gradient(totals))
    return holds, totals, jnp.where(placed, anchors, joined), joined



_MEAN = Ratio(numerator=((1.0, (("v", 0), ("v", 1), ("e", 2, 3))),),
              denominator=((1.0, (("v", 1), ("e", 2, 3))),), axis=0, reduced=True, units=Units())
"""The law's mean (`merged_output`) over the outputs, totals, anchors and the joined anchor, its axis and
units each call's."""


@contracted("test_the_convex_mean_holds_every_order")
def merged_output(outputs: jax.Array, totals: jax.Array, anchors: jax.Array, units: Units,
                  axis: int) -> jax.Array:
    """Finished states' merged output (the module's law): `outputs` `[..., K, ..., D]` along `axis` (each a
    row's mean over its `D` channels), `totals` and `anchors` the rows' `[..., K, ...]` in the walk's
    `units` (`Units`, a value: `ratio_p` keys its compiled programs on it). A state of no total adds
    nothing; a row of none is 0. One ratio (`ratio_p`), held to the hull of the outputs of positive mass
    (its value alone); its derivative the ratio's at every order."""
    axis = axis % jnp.ndim(totals)
    laid, operands = _mean_operands(outputs, totals, anchors, axis)
    mean = ratio(operands, dataclasses.replace(_MEAN, axis=axis, units=units))
    return _in_hull(mean, operands[0], laid, axis)


def _mean_operands(outputs, totals, anchors, axis: int):
    """The law's mean's operands (`_MEAN`'s: outputs, totals, anchors, the joined anchor, `_weighed`'s) and
    where each state holds mass, laid along the outputs' channels."""
    totals = jnp.asarray(totals, outputs.dtype)
    holds, totals, anchors, joined = _weighed(totals, jnp.asarray(anchors, outputs.dtype), axis)
    return _laid_mean(outputs, holds, totals, anchors, joined)


def _laid_mean(outputs, holds, totals, anchors, joined):
    """The mean's operands of weighed states (`_weights`'): where each holds mass, laid along the outputs'
    channels, and a state of no mass's output read where finite, its gap a total's derivative's."""
    laid = _laid(holds, outputs)
    return laid, [jnp.where(laid | jnp.isfinite(outputs), outputs, 0.0), totals, anchors, joined]


def _in_hull(mean, outputs, laid, axis: int) -> jax.Array:
    """The law's `mean` held to the hull of the `outputs` of positive mass (`laid`), its value alone; a row of
    none 0."""
    least = jnp.min(jnp.where(laid, outputs, jnp.inf), axis=axis)
    most = jnp.max(jnp.where(laid, outputs, -jnp.inf), axis=axis)
    held = jnp.where(jnp.any(laid, axis=axis), jnp.clip(mean, least, most), 0.0)
    return mean + jax.lax.stop_gradient(held - mean)


def merged_total(totals: jax.Array, anchors: jax.Array, units: Units,
                 axis: int) -> tuple[jax.Array, jax.Array]:
    """Finished states' merged total (`merged_output`'s `W`) and the anchor it is landed at: `(total,
    anchor)`. The anchor is the joined one (the largest of the states holding a total), but where the
    masses' sum there passes fp32's largest, where it is one above it by more than the powers of two the sum
    passes by and one (`_landing`), the total each mass's factor to it summed (`ratio_p`), the state's
    meaning kept (its output the same at any anchor). The caller takes the anchor as it is: one moved by
    the powers of two at its own anchor's spacing rounded to none (an anchor of 2^30, its neighbours 128
    apart), the total lost half its mass (base review F2). Neither anchor moves with an operand: the total's
    derivative is its own at the landed anchor. An anchor whose spacing alone, in its units, passes the
    format's range (about 180 natural units in float32, an anchor past 2^31 at a scale of one) lands the
    total past the format's least, as no anchor of the format holds it."""
    axis = axis % totals.ndim
    unsettled = _unsettled(totals, anchors, axis)
    _, totals, anchors, joined = _weighed(totals, anchors, axis)
    form = _total_form(axis, units)
    # The masses' sum at the joined anchor as the total sums and lands it (`_ratio_sum`), the one sum the
    # landing reads: a sequential rounded sum lost three masses of 2^102 beside the format's largest the
    # ratio keeps, which then landed past it (the law's review).
    masses = _ratio_sum(list(jax.lax.stop_gradient((totals, anchors, joined))), form)
    landing = jax.lax.stop_gradient(_landing(units, joined, masses, axis))
    total = _in_domain(ratio([totals, anchors, landing], form), masses, landing, joined, axis)
    return total, jnp.squeeze(_row_anchor(landing, unsettled), axis)


def _in_domain(total, masses, landing, joined, axis: int):
    """A merged `total`, or the format's infinity where it lies outside the law's domain (docs/design/
    algebra.md 6.12: a merge's mass within the format at its joined anchor's resolution, as attention's
    always is, each total a softmax denominator, at most its keys): its anchor moved (`_landing`) and its
    total landed beneath the format's normals though its states hold mass (`masses`, at the joined anchor),
    the anchor's next value farther than the move it needed (two totals of 2^127 at an anchor of 2^31: the
    next, 256 past it, flushed their mass to zero). So a run's finiteness checks catch it, never a finite
    wrong mass (the base review's F2)."""
    moved = jnp.squeeze(landing != joined, axis)
    lost = moved & (jnp.reshape(masses[0], total.shape) != 0) & (jnp.abs(total) < jnp.finfo(total.dtype).tiny)
    return jnp.where(lost, jnp.inf, total)


def _total_form(axis: int, units: Units) -> Ratio:
    """The merged total's ratio (`merged_total`): the masses' sum, each a total by its factor to the anchor
    its total lands at; its one term the law's mean's denominator's (`_MEAN`), formed alike."""
    return Ratio(((1.0, (("v", 0), ("e", 1, 2))),), (), axis, reduced=True, units=units)


def _landing(units: Units, joined: jax.Array, masses, axis: int) -> jax.Array:
    """The anchor a merged total at `joined` lands at (`merged_total`), its masses' sum there unlanded
    (`masses`, `_ratio_sum`'s): `joined` where it lands within the format (a total of 2^127 at an anchor of
    2^31 kept there: a move by its spacing, 256, landed it at 0, base review F1); else past `joined + (shift
    + 1) ln 2 / s` (`shift` the powers of two the sum passes the format's largest by, `s` the units' scale)
    by one place of its format, so the total lands below the format's largest; and where that anchor passes
    the format, the least that holds the mass, `joined + ln(sum / largest) / s` read from the unlanded sum
    (its significand's logarithm and its power of two's) and a margin of `2^-16`, by a place of its format:
    32 totals of 2^127 at a scale of 2^-126 moved by `6 ln 2 / s` to an infinite anchor where `3 2^126` holds
    them (the merge P0's base review 9, F4). Infinity only where no anchor of the format holds the mass."""
    if units.scale == 0:
        # Every factor one: no anchor holds the total within the format where the joined one does not.
        return joined
    high, low, exponent = masses
    landed = scaled.scaled(high + low)
    binade = jnp.expand_dims(exponent + landed.exponent, axis)
    largest = scaled.format_of(joined.dtype).largest
    shift = jnp.maximum(binade - largest, 0).astype(joined.dtype)
    # The least anchor of the format at or past `joined + move / s`, the joined anchor and the move formed
    # together in the units' own scale (`s` times an anchor) as one exact double float (`two_product`,
    # `two_sum`), its quotient by `s` stepped one place up where an exact product shows it short: formed
    # apart, a move past the format overflowed beside an anchor far beneath it (64 totals of 2^127 at
    # -2^127, a scale of 2^-126: +2^127 holds them; the merge P0's base review 10, F2), and an anchor at 2^30
    # moves one place, 128, as its sum needs (any further, its mass underflowed).
    scale = np.asarray(units.scale, joined.dtype)
    split = joined.dtype == jnp.float32
    start, start_error = two_product(joined, scale, bit_split=split)
    ln2 = np.asarray(math.log(2), joined.dtype)
    significand = jnp.expand_dims(jnp.abs(landed.significand), axis)
    least = (jnp.log(significand) + np.asarray(2.0 ** -16, joined.dtype)
             - np.asarray(np.log(float(jnp.finfo(joined.dtype).max) / 2.0 ** largest), joined.dtype))

    def landed_at(move):
        high, low = two_sum(start, move)
        low = low + start_error
        candidate = (high + low) / scale
        product, residual = two_product(candidate, scale, bit_split=split)
        short = (product - high) + (residual - low) < 0
        return jnp.where(short, jnp.nextafter(candidate, np.asarray(np.inf, joined.dtype)), candidate)

    safe = landed_at((shift + 1) * ln2)
    held = landed_at(shift * ln2 + least)
    return jnp.where(shift > 0, jnp.where(jnp.isfinite(safe), safe, held), joined)


def _merged_state_value(outputs, totals, anchors, *, units: Units, axis: int):
    """Finished states' merged output, total and the anchor the total is landed at (`merged_state`), a chunk
    of the states at a time (`merged_state_chunked`, its loads `_arrays`'s): `merged_output`'s and
    `merged_total`'s values bit for bit, its operands laid as `merged_state` binds them. Its derivative its
    parts' (`_merged_state_parts`, `_merged_state_jvp`)."""
    # The totals first: a chunk's slots laid as they are (`_arrays`).
    load, chunks = _arrays([totals, anchors, outputs], axis)

    def chunk(c):
        (totals, anchors, outputs), valid = load(c)
        return outputs, totals, anchors, valid

    return merged_state_chunked(chunk, chunks, units, axis)


def merged_state_chunked(load, chunks: int, units: Units, axis: int):
    """Finished states' merged output, total and the anchor the total lands at (`merged_state`'s values) of
    states `load(c)` gives `chunks` at a time: each `(outputs, totals, anchors, valid)`, a chunk of the
    states along `axis` (`chunk_width` of them; the slots `valid`, laid as the totals are, does not hold
    read as no state's, whatever they hold), the outputs its channels after them. The law's one value
    function of a merge, XLA's (`_merged_state_value`, its loads `_arrays`'s) and a kernel's (its own loads
    of its refs: the decode merge's, whose live set is then a chunk's) alike, bit for bit, its carries one a
    row whatever the states: a pass joining the anchors of the states of mass and reading their outputs'
    hull (order-free); the mean's two sums and the masses' (`_streamed`, its two passes), the masses' sum
    landing the total; and where that landing moves, the total's sum at it (`lax.cond`, its two passes
    again)."""
    outputs, totals, _, _ = load(0)
    dtype = outputs.dtype
    without = tuple(size for place, size in enumerate(totals.shape) if place != axis)
    channels = tuple(size for place, size in enumerate(outputs.shape) if place != axis)

    def joining(c, carry):
        joined, finite, least, most, held = carry
        outputs, totals, anchors, valid = load(c)
        holds = valid & _mass(totals)[0]
        laid = _laid(holds, outputs)
        return (jnp.maximum(joined, jnp.max(jnp.where(holds, anchors, -jnp.inf), axis=axis, keepdims=True)),
                jnp.maximum(finite, jnp.max(jnp.where(valid & jnp.isfinite(anchors), anchors, -jnp.inf),
                                            axis=axis, keepdims=True)),
                jnp.minimum(least, jnp.min(jnp.where(laid, outputs, jnp.inf), axis=axis)),
                jnp.maximum(most, jnp.max(jnp.where(laid, outputs, -jnp.inf), axis=axis)),
                held | jnp.any(laid, axis=axis))

    kept = tuple(1 if place == axis else size for place, size in enumerate(totals.shape))
    start = (jnp.full(kept, -jnp.inf, dtype), jnp.full(kept, -jnp.inf, dtype),
             jnp.full(channels, jnp.inf, dtype), jnp.full(channels, -jnp.inf, dtype),
             jnp.zeros(channels, bool))
    # The first chunk joined apart, so the loop's carry is typed as its step makes it (a mesh's sharding).
    joined, finite, least, most, held = jax.lax.fori_loop(1, chunks, joining, joining(0, start))
    unsettled = _joined_anchor(joined, finite)
    joined = _settled(unsettled)
    rank = outputs.ndim

    def weighed(c):
        outputs, totals, anchors, valid = load(c)
        _, operands = _laid_mean(outputs, *_weights(totals, anchors, joined))
        return [_laid_to(operand, rank) for operand in operands], _laid_to(valid, rank)

    count = chunks * outputs.shape[axis]
    mean = dataclasses.replace(_MEAN, axis=axis, units=units)
    numerator, masses = _streamed(weighed, chunks, [_Sum(mean.numerator, axis, (), count),
                                                    _Sum(mean.denominator, axis, (), count)], units)
    merged = jnp.broadcast_to(_landed(_quotient(numerator, masses), dtype), channels)
    # The masses' sum a row (the mean's denominator, laid along the outputs' channels, one) as the total's.
    masses = tuple(jnp.reshape(part, without) for part in masses)
    landing = _landing(units, joined, masses, axis)

    def moved():
        def at_landing(c):
            _, totals, anchors, valid = load(c)
            _, totals, anchors, _ = _weights(totals, anchors, joined)
            return [totals, anchors, landing], valid

        form = _total_form(axis, units)
        (summed,) = _streamed(at_landing, chunks, [_Sum(form.numerator, axis, (), count)], units)
        return _in_domain(_landed(summed, dtype), masses, landing, joined, axis)

    # Where none moved, the cond's other branch the total at the joined anchor itself, of its type (its
    # sharding a mesh's), which the `where` takes alike.
    at_joined = _landed(masses, dtype)
    total = jnp.where(jnp.squeeze(landing == joined, axis), at_joined,
                      jax.lax.cond(jnp.any(landing != joined), moved, lambda: at_joined))
    hull = jnp.where(held, jnp.clip(merged, least, most), 0.0)
    anchor = jnp.squeeze(_row_anchor(landing, unsettled), axis)
    return merged + jax.lax.stop_gradient(hull - merged), total, anchor


def _merged_state_parts(outputs, totals, anchors, *, units: Units, axis: int):
    """Finished states' merged output, total and the anchor the total is landed at (`merged_state`) as its
    parts compute them, `merged_output`'s and `merged_total`'s primitives, whose derivative rules its own
    reads (`_merged_state_jvp`); its value (`_merged_state_value`) theirs bit for bit."""
    total, anchor = merged_total(totals, anchors, units, axis)
    return merged_output(outputs, totals, anchors, units, axis), total, anchor


def _merged_state_abstract(outputs, totals, anchors, *, units: Units, axis: int):
    """The merged state's result types (`_without_axis`): the output's the outputs', the total's and its
    anchor's the totals', their layout one (`merged_state`), each varying along every manual axis an operand
    its value reads varies along (`scaled.varying`): the output reads all three, the total and its anchor
    the totals and anchors."""
    total = scaled.varying(_without_axis(totals, axis), totals, anchors)
    return scaled.varying(_without_axis(outputs, axis), outputs, totals, anchors), total, total


def _merged_state_jvp(primals, tangents, *, units: Units, axis: int):
    """`merged_state_p`'s derivative, its parts' (`_merged_state_parts`: `merged_output`'s and
    `merged_total`'s primitives'), along the operands that move alone: one whose tangent is a symbolic zero
    is held, never moved by an instantiated zero, as the separate primitives hold it (`_ratio_jvp` reads a
    zero tangent as no term: a zero-mass state's term, 0 times a factor past the format, kept out at every
    order); the anchor's derivative none."""
    moving = [index for index, tangent in enumerate(tangents) if type(tangent) is not ad.Zero]

    def value(*moved):
        operands = list(primals)
        for index, operand in zip(moving, moved, strict=True):
            operands[index] = operand
        return _merged_state_parts(*operands, units=units, axis=axis)

    values, moved = jax.jvp(value, tuple(primals[index] for index in moving),
                            tuple(tangents[index] for index in moving))
    # The landed anchor moves with no operand (`merged_total`).
    return values, [*moved[:2], ad.Zero(jax.typeof(values[2]).to_tangent_aval())]


merged_state_p = scaled.primitive("convex_merged_state", _merged_state_value, _merged_state_abstract,
                                  scaled.along_axis(lambda: merged_state_p), several=True)
"""Finished states' merged output, total and its landed anchor (`merged_state`), one operation: lowered as
its value is on every platform but where a template registers its own kernel of that value (a GPU's merge,
one launch for the three), its derivative its value's primitives' (`_merged_state_jvp`)."""
ad.primitive_jvps[merged_state_p] = _merged_state_jvp


def merged_state(outputs: jax.Array, totals: jax.Array, anchors: jax.Array, units: Units,
                 axis: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Finished states' merged output, merged total and the anchor the total is landed at (`merged_output`,
    `merged_total`), as one operation (`merged_state_p`), which a template may run as one kernel. Its
    operands bound in the layout the law weighs them in, so each result's type is its value's: anchors
    common to the states, or totals shared by a batch, laid as each other, and the outputs' rows along them
    (the merge P0's base review F3: totals `[1, 2]` beside anchors `[2, 2]` declared a total of one row
    where its value held two, which lowering refused)."""
    totals, anchors = (jnp.asarray(x, outputs.dtype) for x in (totals, anchors))
    layout = jnp.broadcast_shapes(totals.shape, anchors.shape)
    totals, anchors = (jnp.broadcast_to(x, layout) for x in (totals, anchors))
    outputs = jnp.broadcast_to(outputs, jnp.broadcast_shapes(
        outputs.shape, layout + (1,) * (outputs.ndim - len(layout))))
    return tuple(merged_state_p.bind(outputs, totals, anchors, units=units, axis=axis % totals.ndim))


_FINISHED = Ratio(((1.0, (("v", 0),)),), ((1.0, (("v", 1),)),), 0, reduced=True, units=Units())
"""A state's contraction over its total (`finished`), along a leading axis of one."""

_RESUMED = Ratio(((1.0, (("v", 0), ("v", 1))),), (), 0, reduced=True, units=Units())
"""A state's output times its total (`resumed`), along a leading axis of one."""


def finished(accumulated: jax.Array, total: jax.Array) -> jax.Array:
    """A walk's own state finished: its accumulated contraction `[..., D]` over its total `[...]` (`o = acc /
    l`), one ratio (`ratio_p`), so no division by a broadcast total flushes (V5); a state of no total its
    contraction, 0."""
    # A total of none (zero, negative or NaN) holds one; a subnormal one is the state's (`_mass`).
    total = jnp.asarray(total, accumulated.dtype)
    held = jnp.where(_mass(total)[0], total, 1.0)
    return ratio([accumulated[None], held[None]], _FINISHED)


def resumed(output: jax.Array, total: jax.Array) -> jax.Array:
    """A finished state's contraction `acc = o l` (`finished`'s inverse, a resumed state's), one ratio
    (`ratio_p`, V8)."""
    return ratio([output[None], jnp.asarray(total, output.dtype)[None]], _RESUMED)
