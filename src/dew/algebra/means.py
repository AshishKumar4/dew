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
import functools
import itertools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad, batching
from jax.sharding import PartitionSpec

from dew.algebra import scaled
from dew.algebra.scaled import Scaled
from dew.nn import float_bits, precision
from dew.nn.derivatives import contracted
from dew.nn.precision import correctly_rounded_divide, two_product, two_sum


@dataclasses.dataclass(frozen=True)
class Units:
    """A walk's units of anchor: its factor from a state's anchor `m` to another `M` is `exp(scale (m - M))`
    in the reals. `compensated`, the accurate form's: the anchors' difference taken exactly (`two_sum`), its
    product by `scale` too (`two_product`) and its conversion to base 2 (`scaled.base2`), so the factor's
    exponent is the reals' to within its own rounding; else a fold's (FlashAttention 2's): `(m - M)` times
    `scale log2(e)` rounded, as its kernels weigh a key. A value (`scaled._valued`), so the law's primitive
    keys its programs on it alone."""

    scale: float = 1.0
    compensated: bool = True


FLOOR = -(1 << 16)
"""The least power of two a factor is held exactly above: below `2^FLOOR` it is zero. A factor held at a
floor of positive value weighed a negligible state with mass a large output made observable (2^-378, the
floor before: outputs 0 and 2^127, totals 2^-126 and 2^127, anchors 0 and -1000 merged to 4, their mean
2^-1063, base review F1); one below `2^FLOOR` weighs a state, its total at most the format's largest beside
the joined state's of at least its least subnormal (a ratio of 2^277 in float32), its output and any
derivative's tangent and output gap each at most 2^129, its units' scale `s`, at under `2^(FLOOR + 534) s`
a derivative order: none any order under a hundred reads beside a result's least subnormal. Its exponent
bound is the factor's `reach` (`_FACTOR_REACH`)."""

CEILING = 1 << 22
"""The greatest power of two a factor is held exactly beneath: past `2^CEILING` it is NaN, the law's refusal
of a state beyond its reach. Only a state of no total has a factor past one (its anchor above the joined
one, `_weighed`), and only its total's derivative reads it; two such states' factors keep their distinct
exponents through the cancellation between them (anchors 100000 and 100001: `e^100000 - e^100001`, not the
zero that clamping both at one power gave, the review's F6)."""

_FACTOR_REACH = max(-FLOOR, CEILING) + 1
"""A factor's exponent bound (`Scaled.reach`): within `2^CEILING`, a term's few other factors' hundreds of
places beside it under the construct's 2^24 (`scaled.REACH`)."""

_REACH = 1 << 23
"""The bound of every exponent a ratio places by (`precision.power2`): a term's (a factor's and its other
factors'), and a sum's beneath its largest."""


def _laid(array: jax.Array, like: jax.Array) -> jax.Array:
    """`array`, a row's value, laid along `like`'s trailing channels."""
    return array.reshape(array.shape + (1,) * (like.ndim - array.ndim))


def _factors_value(anchors: jax.Array, joined: jax.Array, *, units: Units):
    """Each state's factor from its anchor to `joined` (the largest, kept on its axis) in `units`, a scaled
    value aligned once: its base-2 exponent's nearest integer the value's power of two and its remainder, in
    [-1/2, 1/2] with the exponent's residual joined, the significand's exponential, so no factor underflows
    above `2^FLOOR` (V6: an anchor 100 below the joined one, its mass of 2^-24 kept); one below it zero.
    Anchors whose difference passes the format, its scaled value within it (`+-2^127` at a scale of
    `2^-120`: -256), are taken at half each, exact, their scaled difference doubled after: the same bits
    wherever the difference is finite, a power of two moving no rounding (the base review's F2: merged to 0
    where float64's merge is 1629.34)."""
    spilled = jnp.isinf(anchors - joined)
    half, twice = (jnp.where(spilled, np.asarray(power, anchors.dtype), np.asarray(1, anchors.dtype))
                   for power in (0.5, 2.0))
    if units.compensated:
        difference, error = two_sum(anchors * half, -(joined * half))
        if units.scale != 1.0:
            difference, rounding = two_product(difference, np.asarray(units.scale, anchors.dtype),
                                               bit_split=anchors.dtype == jnp.float32)
            error = rounding + error * units.scale
        high, low = scaled.base2(difference * twice)
        low = low + (error * twice) * math.log2(math.e)
    else:
        high = (anchors * half - joined * half) * np.asarray(units.scale * math.log2(math.e),
                                                             anchors.dtype) * twice
        low = jnp.zeros_like(high)
    held = high > FLOOR
    # A factor past 2^CEILING is NaN (`CEILING`), its exponent held within the factor's reach meanwhile.
    refused = high >= CEILING
    high, low = jnp.where(held & ~refused, high, 0.0), jnp.where(held & ~refused, low, 0.0)
    whole = jnp.floor(high)
    whole = whole + (high - whole >= 0.5).astype(high.dtype)
    # 2^f as XLA lowers exp2, the exponential of f ln 2 (one rounded product, no sum beside it to contract):
    # a Pallas kernel's exp2 is another library's, its exp XLA's own, so the law lands alike in both.
    fraction = (high - whole) + low
    significand = jnp.where(held, jnp.exp(fraction * np.asarray(math.log(2), fraction.dtype)), 0.0)
    made = scaled.normalized(jnp.where(refused, jnp.nan, significand), whole.astype(jnp.int32), _FACTOR_REACH)
    return made.significand, made.exponent


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


@dataclasses.dataclass(frozen=True)
class Ratio:
    """A ratio of sums of products of float operands (`ratio_p`): `sum numerator / sum denominator`, each a
    tuple of terms `(coefficient, factors)`, a factor `("v", i)` operand `i`, `("e", i, j)` the factor
    `exp(s (operand_i - operand_j))` of `units` (`_factors_value`) or `("s",)` the units' scale `s` in the
    operands' format. The numerator is summed over the operands' `axis` where `reduced` (the result without
    it), else taken a state at a time; the denominator is always summed over it (none: one). The numerator
    is summed too over `over`, further axes of the operands' domain, each term over its own extent there (a
    term of none counted once), the result one along them: a transpose's cotangent summed over the axes its
    operand is broadcast along before it lands (opposite channels' `+-2^130` landed apart and summed were
    NaN where the sum is 0, the base review's F1). Its quotients, each its numerator's over its own
    denominator, are summed `across` further axes before they land, the result one along them: a
    transpose's along the axes its operand is broadcast along where its denominator varies (`sum_c N_c /
    D_c`: `2^126 / 0.25 - 2^127 / 0.5` landed apart and summed was NaN where it is 0, the merge P0's base
    review F1). Its denominator's sum is raised to `power`. Its numerator is, where `products` names one, a
    sum of products of sums, `sum_t c_t prod_j S_tj` (`products`: each `(c_t, (S_t1, ...))`, each `S_tj` a
    `Summed`), its `numerator` then none: a derivative's whose denominator moves, `(dP D - q P dD) / D^(q+1)`
    (`_ratio_jvp`), each sum formed once, unlanded, so no quotient is landed apart from the terms it cancels
    against (the merge P0's base review 9, F2: a landed quotient of 2^-156 lost the derivative's -2^-66).
    A zero lands as +0. A value (`scaled._valued`)."""

    numerator: tuple
    denominator: tuple
    axis: int
    reduced: bool
    units: Units
    over: tuple[int, ...] = ()
    across: tuple[int, ...] = ()
    power: int = 1
    products: tuple = ()


@dataclasses.dataclass(frozen=True)
class Summed:
    """A sum a ratio's numerator multiplies (`Ratio.products`): its `terms` (`Ratio`'s), summed over the
    states' axis where `reduced` (none along it after), else taken a state at a time, and over its further
    axes `over`, each term over its own extent there (`Ratio.over`'s account)."""

    terms: tuple
    reduced: bool
    over: tuple[int, ...] = ()


def _products(form: Ratio) -> tuple:
    """A ratio's numerator as products of sums (`Ratio.products`): its own, or its one sum's."""
    return form.products or ((1.0, (Summed(form.numerator, form.reduced, form.over),)),)


def _canonical(products) -> tuple:
    """Products of sums (`Ratio.products`) written once each: a sum of one term its coefficient taken into
    its product's (a sum of `c f` is `c` times the sum of `f`), its sums in one order, products of the same
    sums one, their coefficients added, none of coefficient zero; so a sum formed in several products, or
    differing from another by its sign or a scale alone (`dD`, `-q dD`), is formed once
    (`_products_summed`)."""
    merged: dict = {}
    for coefficient, sums in products:
        made = []
        for each in sums:
            if len(each.terms) == 1:
                own, factors = each.terms[0]
                coefficient *= own
                each = dataclasses.replace(each, terms=((1.0, factors),))
            made.append(each)
        key = tuple(sorted(made, key=repr))
        merged[key] = merged.get(key, 0.0) + coefficient
    return tuple((coefficient, sums) for sums, coefficient in merged.items() if coefficient != 0)


def _classic(form: Ratio) -> Ratio:
    """`form` with a numerator of one product of one sum at its own flags written as that sum
    (`Ratio.numerator`), its coefficient in its terms: the form every value and derivative whose
    denominator does not move takes, bit for bit as before products."""
    if len(form.products) == 1 and len(form.products[0][1]) == 1:
        coefficient, (only,) = form.products[0]
        if only.reduced == form.reduced and only.over == form.over:
            return dataclasses.replace(form, products=(), numerator=tuple(
                (coefficient * each, factors) for each, factors in only.terms))
    return form


def _laid_to(operand: jax.Array, rank: int) -> jax.Array:
    """`operand` laid along a ratio's trailing axes (a row's value beside its channels), of `rank`."""
    return operand.reshape(operand.shape + (1,) * (rank - operand.ndim))


def _factor_values(operands, units: Units, rank: int) -> Callable[[tuple], Scaled]:
    """Each factor's value (`Ratio`'s, a scaled value) as a ratio's terms read it, formed once a ratio
    whatever the terms reading it: an operand's (`("v", i)`), the units' scale (`("s",)`) or an
    exponential (`("e", i, j)`, `_factors_value`), the numerator's and the denominator's alike (the mean's
    exponential formed once for both)."""
    held: dict[tuple, Scaled] = {}

    def value(factor: tuple) -> Scaled:
        if factor not in held:
            if factor[0] == "v":
                held[factor] = scaled.scaled(_laid_to(operands[factor[1]], rank))
            elif factor[0] == "s":
                held[factor] = scaled.scaled(jnp.asarray(units.scale, operands[0].dtype))
            else:
                significand, power2 = _factors_value(_laid_to(operands[factor[1]], rank),
                                                     _laid_to(operands[factor[2]], rank), units=units)
                held[factor] = Scaled(significand, power2, _FACTOR_REACH)
        return held[factor]
    return value


def _product(term, value: Callable[[tuple], Scaled], dtype):
    """A term's value (`Ratio`) as a double float at a power of two, `(high, low, exponent)`: the
    coefficient's significand times each factor's (`value`, `_factor_values`) in turn, each product exact
    (`two_product`) and its residual carried, the exponents summed, then renormalized into [1, 2) once: a
    product of significands in [1, 2) stays far within the format, and each step's power of two moves no
    rounding, so it is the term renormalized after every factor, bit for bit. The term is within `u^2` of its
    real value but for its factors' own roundings (an exponential's, `_factors_value`). A subnormal operand
    is read from its bits (`scaled.scaled`); a zero term's exponent is no other's (`_summed` reads it as
    none). A term with a zero factor is zero whatever its others: a state of no mass contributes none, its
    exponential past its reach (NaN) included (the base review's F3: NaN where the merge is 1); a term of
    its derivative without that factor keeps its own."""
    coefficient, factors = term
    fraction, power = math.frexp(coefficient)
    high, low = jnp.asarray(2 * fraction, dtype), jnp.zeros((), dtype)
    exponent = jnp.asarray(power - 1, jnp.int32)
    zero = jnp.zeros((), bool)
    for count, factor in enumerate(factors, 1):
        factor_value = value(factor)
        zero = zero | (factor_value.significand == 0)
        product, residual = two_product(high, factor_value.significand)
        high, low = two_sum(product, residual + low * factor_value.significand)
        exponent = exponent + factor_value.exponent
        if count % _HELD == 0:
            high, low, exponent = _renormalized(high, low, exponent, dtype)
    high, low, exponent = _renormalized(high, low, exponent, dtype)
    return jnp.where(zero, 0.0, high), jnp.where(zero, 0.0, low), exponent


_HELD = 64
"""The factors a term's product takes before it is renormalized (`_product`): each significand under 2, its
high part stays under `2^64`, which the exact product's split (Dekker's, a multiply by `2^12 + 1`) keeps
within float32 (a term of 128 factors just under 2 overflowed it to NaN, the law's review)."""


def _renormalized(high, low, exponent, dtype):
    """A double float `(high, low)` at `exponent` with its high part moved into [1, 2), its low part and its
    exponent by the same power of two (exact)."""
    moved = scaled.scaled(high)
    return moved.significand, low * float_bits.exact_power2(-moved.exponent, dtype), exponent + moved.exponent


_NONE = np.iinfo(np.int32).min // 4
"""A zero term's exponent: beneath every other's, so a row's largest is a nonzero term's."""

_CEILING = np.iinfo(np.int32).max // 4
"""The first window's ceiling (`_windows`): above every term's exponent, so it holds them all."""


CHUNK = 8
"""The states a sum reads a step at a time (`_streamed`): a step's program and live set are a chunk's, the
loop takes a row's states in `ceil(S / 8)` trips, so no program grows with the states. Eight: a step's halving
trees are three levels deep and its tile `[8, C]` (a term's exact product holds about a dozen such arrays
live), where the decode merge's kernel loaded its row's whole `[S, C]` and unrolled every state, its
compile 236 s at 64 slices and past 15 minutes at 256 (the attention lane's sm90 probe); its widest row, 256
slices (B1 x 131072 at 512 keys a slice), takes 32 trips."""


def chunk_width(states: int) -> int:
    """The states a sum reads a step at a time for a row of `states` (`_arrays`, and a kernel's loads, which
    must read the same so its bits are XLA's): `CHUNK`, or the least power of two holding a shorter row (a
    decode merge of four splits reads four, not eight, its work not doubled by padding), a power of two so
    its halving trees split evenly (`_halves`, which Pallas Triton lowers)."""
    return min(CHUNK, 1 << max(states - 1, 0).bit_length())


def _halved(operation: Callable, x: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    """`x` reduced by `operation` over each of `axes` (one along them after) in halves: a tree over each
    axis's order, its program the size of the axis's logarithm, an even extent's halves split (`_halves`)."""
    for axis in axes:
        while x.shape[axis] > 1:
            low, high, rest = _halves(x, axis)
            pair = operation(low, high)
            x = pair if rest is None else jnp.concatenate([pair, rest], axis=axis)
    return x


def _halves(x: jax.Array, axis: int):
    """`x`'s two halves along `axis` and its odd last element (None where its extent is even): an even
    extent's by `lax.split`, which Pallas Triton lowers as a register shuffle (a chunk's extent,
    `chunk_width`, a power of two at every level: it lowers no other slice of a register tile but one
    element's, each a masked reduction), an odd one's by slices (a transpose's further axes, `Ratio.over`,
    XLA's alone)."""
    count = x.shape[axis]
    half = count // 2
    if count % 2 == 0:
        low, high = jax.lax.split(x, (half, half), axis=axis)
        return low, high, None
    return (jax.lax.slice_in_dim(x, 0, half, axis=axis), jax.lax.slice_in_dim(x, half, 2 * half, axis=axis),
            jax.lax.slice_in_dim(x, 2 * half, count, axis=axis))


def _halved_compensated(x: jax.Array, axes: tuple[int, ...]):
    """`x` summed over `axes` in halves (`_halved`), each pair's sum exact (`two_sum`) and its error kept, the
    errors summed plainly: `(sum, error)`, a cascade compensated as the compensated sum is."""
    error = x - x
    for axis in axes:
        while x.shape[axis] > 1:
            (a, b, rest), (ea, eb, error_rest) = _halves(x, axis), _halves(error, axis)
            pair, made = two_sum(a, b)
            made = made + ea + eb
            # An odd extent's last element, the sum's and its error's alike (one shape, so both or neither).
            if rest is not None and error_rest is not None:
                pair = jnp.concatenate([pair, rest], axis=axis)
                made = jnp.concatenate([made, error_rest], axis=axis)
            x, error = pair, made
    return x, error


@dataclasses.dataclass(frozen=True)
class _Sum:
    """One sum a step forms (`_streamed`): `terms` (`Ratio`'s), over the states' `axis` where given (else a
    state at a time) and over the further axes `over` (`Ratio.over`), its states `count` slots."""

    terms: tuple
    axis: int | None
    over: tuple[int, ...]
    count: int

    def axes(self) -> tuple[int, ...]:
        return tuple(sorted(({self.axis} if self.axis is not None else set()) | set(self.over)))

    def summands(self, operands, value, valid, dtype):
        """Each term's product over a chunk's `operands` (`_product`, its factors `value`'s), broadcast to the
        sum's extent (its read operands', its states' axis whole, its own along `over`), the slots `valid`
        does not hold read as zero, an exponent none where its product is; each formed once
        (`lax.optimization_barrier`): an exact product is hundreds of operations an element, which a compiler
        fusing it into each reduction formed four times (XLA:CPU's loop vectorizer held one such program past
        240 s, row 7925)."""
        rank = max(operand.ndim for operand in operands)
        read = {operand for _, factors in self.terms for factor in factors for operand in factor[1:]}
        shape = list(np.broadcast_shapes((1,) * rank, *(operands[index].shape for index in read)))
        if self.axis is not None:
            shape[self.axis] = max(operand.shape[self.axis] for operand in operands)
        made = []
        for term in self.terms:
            parts = _product(term, value, dtype)
            extent = np.broadcast_shapes((1,) * rank, *(np.shape(part) for part in parts))
            own = tuple(extent[place] if place in self.over else size for place, size in enumerate(shape))
            high, low, exponent = (jnp.broadcast_to(part, own) for part in parts)
            if valid is not None:
                high, low = jnp.where(valid, high, 0.0), jnp.where(valid, low, 0.0)
            made.append((high, low.astype(dtype), jnp.where(high != 0, exponent, _NONE)))
        return jax.lax.optimization_barrier(tuple(made))

    def bits(self, extents) -> int:
        """The bits its summands' count takes (each term's `extents`): their magnitudes' sum, each under 2,
        lies under `2^bits`."""
        spread = max(int(np.prod([extent[place] for place in self.over])) for extent in extents)
        return (self.count * len(self.terms) * spread).bit_length()


def _placing(summands, axes, ceiling, carry):
    """A step of a sum's placement (a window's first pass, `_windows`): the largest exponent so far of its
    summands beneath `ceiling` (the window's, those above it summed already) and their magnitudes' sum at it,
    moved by a power of two when a later state's is larger. Formed from the summands themselves: placed from
    its factors' exponents alone (an upper bound, no product formed), the programs were smaller and compiled
    two to three times slower (a merged state's gradient 4.2 s of CPU against 1.7 s at four states, its
    second derivative 12.5 s against 3.9 s)."""
    top, magnitude = carry
    exponents = [jnp.where(exponent < ceiling, exponent, _NONE) for _, _, exponent in summands]
    moved = jnp.maximum(top, _folded(jnp.maximum, [_halved(jnp.maximum, exponent, axes)
                                                    for exponent in exponents]))
    magnitude = precision.power2(magnitude, top - moved, reach=_REACH) + _folded(jnp.add, [
        _halved(jnp.add, precision.power2(jnp.abs(high) + jnp.abs(low), exponent - moved, reach=_REACH), axes)
        for (high, low, _), exponent in zip(summands, exponents, strict=True)])
    return moved, magnitude


def _summing(summands, axes, top, ceiling, form: scaled.Format, carry):
    """A step of a sum's window (its second pass, `_windows`): its summands beneath `ceiling` whose
    placement at `top`'s power of two a binade beneath the format's largest keeps them normal, summed there
    compensated (`two_sum`, its errors plainly), their low parts with them; and the largest exponent of those
    that placement puts beneath the normals (`below`), the next window's, none of them rounded here: summed
    in the window, a pair's half-ulp rounded away before its larger term's opposite cancelled it
    (`2^-102 + 2^-126 - 2^-102` was 0, the merge P0's base review 11, F1)."""
    held, error, low, below = carry
    placement = form.largest - 1
    present = [(exponent != _NONE) & (exponent < ceiling) for _, _, exponent in summands]
    inside = [each & (exponent - top + placement >= form.least + form.mantissa + 1)
              for each, (_, _, exponent) in zip(present, summands, strict=True)]

    def placed(index):
        return [jnp.where(each, precision.power2(summand[index], summand[2] - top + placement, reach=_REACH),
                          0.0)
                for each, summand in zip(inside, summands, strict=True)]

    pairs = [_halved_compensated(each, axes) for each in placed(0)]
    chunk, errors = scaled.compensated([pair for pair, _ in pairs])
    held, made = two_sum(held, chunk)
    error = error + made + errors + _folded(jnp.add, [made for _, made in pairs])
    low = low + _folded(jnp.add, [_halved(jnp.add, each, axes) for each in placed(1)])
    below = jnp.maximum(below, _folded(jnp.maximum, [
        _halved(jnp.maximum, jnp.where(here & ~kept, exponent, _NONE), axes)
        for here, kept, (_, _, exponent) in zip(present, inside, summands, strict=True)]))
    return held, error, low, below


def _windows(summands_of, chunks: int, axes: list[tuple[int, ...]], bits: list[int], dtype):
    """Sums of states `summands_of(c)` gives `chunks` at a time (each sum's `(high, low, exponent)` terms
    along its `axes`, one along them after), each `(high, low, exponent)`: summed a window at a time, from its
    largest summands down (`lax.while_loop`), each window two passes over the chunks (`lax.fori_loop`, a
    step's program a chunk's, its carry one a row whatever the states): the first placing it (`_placing`:
    its summands' largest exponent beneath the windows before it, and their magnitudes' sum there, the power
    of two that puts that sum a binade beneath the format's largest, so no partial sum passes the format),
    the second summing those it keeps normal there (`_summing`), the window joined to those before
    (`_joined`). Another window follows wherever summands lie beneath it within a double float's reach of
    the sum so far (`bits`: their count's headroom), or the sum so far is zero, so a band that cancels never
    drops the bands beneath it, however far (`[1, -1, 1, -1, 1]` at anchors `[1000, 1000, 200, 200, 0]` sums
    to `1`, the merge P0's base review 13, F2: two bands, it was 0). A row the windows have resolved takes
    those that follow for the others, which add nothing it holds or beneath its double float's reach."""
    form = scaled.format_of(dtype)
    placement, reach = form.largest - 1, form.least + form.mantissa + 1

    def looped(step, carry):
        if chunks == 1:
            return step(0, carry)
        return jax.lax.fori_loop(0, chunks, step, carry)

    # Each carry typed as the step's own reductions of the first chunk (their shape, their sharding under an
    # explicit mesh, their varying axes in a shard map): a carry of a shape alone was refused beside them.
    zeros = [(lambda reduced: reduced - reduced)(_folded(jnp.maximum, [
        _halved(jnp.maximum, exponent, each) for _, _, exponent in summands]))
        for summands, each in zip(summands_of(0), axes, strict=True)]

    def window(state):
        totals, ceilings, _ = state
        placed = looped(lambda c, carry: tuple(
            _placing(summands, each, ceiling, part)
            for summands, each, ceiling, part in zip(summands_of(c), axes, ceilings, carry, strict=True)),
            tuple((zero + _NONE, zero.astype(dtype)) for zero in zeros))
        tops = [jnp.where(top == _NONE, 0, top) + scaled.scaled(magnitude).exponent
                for top, magnitude in placed]
        summed = looped(lambda c, carry: tuple(
            _summing(summands, each, top, ceiling, form, part)
            for summands, each, top, ceiling, part
            in zip(summands_of(c), axes, tops, ceilings, carry, strict=True)),
            tuple((*(zero.astype(dtype),) * 3, zero + _NONE) for zero in zeros))
        made, below = [], []
        for total, top, (held, error, low, under) in zip(totals, tops, summed, strict=True):
            high, rest = two_sum(held, low + error)
            made.append(_joined(total, (high, rest, top - placement)))
            below.append(under)
        # A sum's ceiling only falls: one whose window found no summand beneath it keeps its own, where its
        # empty window's top raised it and a window after read the summands it had summed again (`2^-120`
        # for `2^-121` beside a sum still windowing, the merge P0's head review, F1).
        lowered = [jnp.where(found == _NONE, ceiling, jnp.minimum(ceiling, top - placement + reach))
                   for (found, _), top, ceiling in zip(placed, tops, ceilings, strict=True)]
        return made, lowered, below

    def unresolved(state):
        totals, _, below = state
        wanting = [(under != _NONE) & ((high == 0) | (
            under + count + 1 >= exponent + scaled.scaled(high).exponent - 2 * (form.mantissa + 1)))
                   for (high, _, exponent), under, count in zip(totals, below, bits, strict=True)]
        return functools.reduce(jnp.logical_or, [jnp.any(each) for each in wanting])

    # The first window taken by the loop itself, its start a zero sum with summands beneath every ceiling
    # (`below` past `_NONE`): one window's program in the loop, where the first called before it compiled
    # every sum's two passes twice (a merged state's file past 10 GiB of compile memory).
    start = ([(zero.astype(dtype), zero.astype(dtype), zero) for zero in zeros],
             [zero + _CEILING for zero in zeros], list(zeros))
    totals, _, _ = jax.lax.while_loop(unresolved, window, start)
    return totals


def _streamed(load, chunks: int, sums: list[_Sum], units: Units):
    """`sums` of states a `load(c)` gives `chunks` at a time (each `(operands, valid)`: a chunk's operands
    laid to one rank, its states along each sum's axis, and the slots it holds), each `(high, low,
    exponent)` without its states' axis: summed a window at a time, each two passes over the chunks
    (`_windows`). The one value function of the law's sums, XLA's (`_arrays`'s loads) and a kernel's (its
    own loads of its refs) alike, a chunk's factors formed once for all its sums (`_factor_values`)."""
    operands, _ = load(0)
    dtype = operands[0].dtype
    rank = max(operand.ndim for operand in operands)

    def summands_of(c):
        operands, valid = load(c)
        value = _factor_values(operands, units, rank)
        return [each.summands(operands, value, valid, dtype) for each in sums]

    bits = [each.bits([high.shape for high, _, _ in summands])
            for each, summands in zip(sums, summands_of(0), strict=True)]
    made = _windows(summands_of, chunks, [each.axes() for each in sums], bits, dtype)
    return [tuple(jnp.squeeze(part, each.axis) if each.axis is not None else part for part in joined)
            for each, joined in zip(sums, made, strict=True)]


def _quotients_summed(summed, axes: tuple[int, ...]):
    """A ratio's quotients before they land (`_ratio_sum`'s, `(high, low, exponent)`, each under 2 at its
    power of two) summed over `axes` (`Ratio.across`), one along them, landing none: summed a window at a
    time as a sum's terms are (`_windows`), one chunk of them, so opposite quotients past the format cancel
    before the sum lands. The quotients formed once before the windows read them
    (`lax.optimization_barrier`): fused into each, a ratio of five products of three sums compiled in 47 s
    where its quotients alone take 4."""
    high, low, exponent = jax.lax.optimization_barrier(summed)
    shape = np.broadcast_shapes(*(np.shape(part) for part in summed))
    summands = [(jnp.broadcast_to(high, shape), jnp.broadcast_to(low, shape),
                 jnp.broadcast_to(jnp.where(high != 0, exponent, _NONE), shape))]
    bits = int(np.prod([shape[place] for place in axes])).bit_length()
    (made,) = _windows(lambda c: [summands], 1, [axes], [bits], high.dtype)
    return made


def _arrays(operands, axis: int | None):
    """A load of `operands` (each laid to one rank) a chunk of the states along `axis` at a time
    (`_streamed`), `(load, chunks)`: each operand of the states' extent padded to whole chunks (zeros, its
    slots then held by none) and sliced, one of extent one whole; along no axis, one chunk, the
    operands whole."""
    if axis is None:
        return (lambda c: (list(operands), None)), 1
    states = max(operand.shape[axis] for operand in operands)
    # Whole chunks of `chunk_width`, a kernel's (its tiles a power of two), so its chunks are XLA's.
    width = chunk_width(states)
    chunks = -(-states // width)
    extra = chunks * width - states
    # Padded along the states alone (a pad slices nothing, so a batch split over a mesh stays whole), its
    # slots zeros, which `valid` holds none of.
    padded = [jax.lax.pad(operand, jnp.zeros((), operand.dtype),
                          [(0, extra if place == axis else 0, 0) for place in range(operand.ndim)])
              if operand.shape[axis] == states and extra else operand for operand in operands]

    def load(c):
        start = c * width
        sliced = [jax.lax.dynamic_slice_in_dim(operand, start, width, axis)
                  if operand.shape[axis] == chunks * width else operand for operand in padded]
        index = start + jax.lax.broadcasted_iota(jnp.int32, tuple(width if place == axis else 1
                                                                   for place in range(sliced[0].ndim)), axis)
        return sliced, index < states

    return load, chunks


def _folded(operation: Callable, values: list):
    """`values` reduced by `operation` in pairs, a tree over their order: `log2 n` dependent steps where a
    fold in their order is `n` (a kernel's program's span, the decode merge's), stacking none."""
    while len(values) > 1:
        values = [operation(values[at], values[at + 1]) for at in range(0, len(values) - 1, 2)] + (
            [values[-1]] if len(values) % 2 else [])
    return values[0]


def _joined(a, b):
    """Two double floats at powers of two (`_streamed`'s bands) summed, `(high, low, exponent)`: each moved to
    the
    larger's power (exact, or beneath the larger's `u^2`), their high parts' sum exact (`two_sum`), its error
    carried in the low part; a zero sum where both are."""
    a, b = _normal(*a), _normal(*b)
    exponents = [jnp.where(high != 0, exponent, _NONE) for high, _, exponent in (a, b)]
    top = jnp.maximum(*exponents)
    moved = [(precision.power2(high, exponent - top, reach=_REACH),
              precision.power2(low, exponent - top, reach=_REACH))
             for (high, low, _), exponent in zip((a, b), exponents, strict=True)]
    high, error = two_sum(moved[0][0], moved[1][0])
    high, low = two_sum(high, error + moved[0][1] + moved[1][1])
    return high, low, jnp.where(top == _NONE, 0, top)


def _normal(high, low, exponent):
    """A double float `(high, low)` at `exponent` with its high part moved into [1, 2) (exact)."""
    moved = scaled.scaled(high)
    return moved.significand, precision.power2(low, -moved.exponent, reach=_REACH), exponent + moved.exponent


def _quotient(numerator, denominator):
    """A double float's quotient by another's (`_summed`'s): the high parts' correctly rounded quotient and
    its exact residual (`two_product`) giving the low part, `(high, low, exponent)`; zero over a zero
    denominator (a row of no mass)."""
    n_high, n_low, n_exponent = _normal(*numerator)
    d_high, d_low, d_exponent = _normal(*denominator)
    none = d_high == 0
    divisor = jnp.where(none, 1.0, d_high).astype(n_high.dtype)
    quotient = (correctly_rounded_divide(n_high, divisor) if n_high.dtype == jnp.float32
                else n_high / divisor)
    product, residual = two_product(quotient, divisor)
    low = (((n_high - product) - residual) + n_low - quotient * d_low) / divisor
    return (jnp.where(none, 0.0, quotient), jnp.where(none, 0.0, low),
            jnp.where(none, 0, n_exponent - d_exponent))


def _ratio_sums(operands, form: Ratio):
    """A ratio's numerator's and denominator's sums (`_streamed`'s), each `(high, low, exponent)`, its
    denominator's None where it has none (`Ratio`); a reduced form's both without its axis, its sums over the
    states read a chunk at a time and their factors formed once for both (the mean's exponential), a sum a
    state at a time (an unreduced numerator's) whole. A numerator of products (`Ratio.products`) its
    products' sum (`_products_summed`)."""
    if form.products:
        return _products_summed(operands, form)
    rank = max(operand.ndim for operand in operands)
    laid = [_laid_to(operand, rank) for operand in operands]
    states = max(operand.shape[form.axis] for operand in laid)
    count = -(-states // chunk_width(states)) * chunk_width(states)
    over_states = [_Sum(form.numerator, form.axis, form.over, count)] if form.reduced else []
    if form.denominator:
        over_states.append(_Sum(form.denominator, form.axis, (), count))
    made = _streamed(*_arrays(laid, form.axis), over_states, form.units) if over_states else []
    numerator = made.pop(0) if form.reduced else _streamed(*_arrays(laid, None), [_Sum(
        form.numerator, None, form.over, 1)], form.units)[0]
    return numerator, (made[0] if form.denominator else None)


def _products_summed(operands, form: Ratio):
    """A numerator of products of sums (`Ratio.products`) and its denominator's sum, each `(high, low,
    exponent)` (`_ratio_sums`'s): each distinct sum formed once (`_streamed`: those over the states in one
    pass with the denominator, their factors formed once for all, those a state at a time in another), each
    product its coefficient times its sums in turn (`_times`, a double float's product, landing none), the
    products joined (`_joined`); a reduced sum laid along the states' axis again where the form keeps it."""
    rank = max(operand.ndim for operand in operands)
    laid = [_laid_to(operand, rank) for operand in operands]
    states = max(operand.shape[form.axis] for operand in laid)
    count = -(-states // chunk_width(states)) * chunk_width(states)
    # The denominator's own sum where a product takes it (the quotient rule's `D`), formed once for both.
    own = Summed(form.denominator, reduced=True) if form.denominator else None
    distinct = list(dict.fromkeys(each for _, sums in form.products for each in sums))
    reduced = [each for each in distinct if each.reduced and each != own]
    entrywise = [each for each in distinct if not each.reduced]
    over_states = [_Sum(each.terms, form.axis, each.over, count) for each in reduced]
    if form.denominator:
        over_states.append(_Sum(form.denominator, form.axis, (), count))
    made = _streamed(*_arrays(laid, form.axis), over_states, form.units) if over_states else []
    values = dict(zip(reduced, made[:len(reduced)], strict=True))
    denominator = None
    if own is not None:
        denominator = values[own] = made[-1]
    if entrywise:
        values.update(zip(entrywise, _streamed(*_arrays(laid, None), [_Sum(each.terms, None, each.over, 1)
                                                                     for each in entrywise], form.units),
                          strict=True))
    if not form.reduced:
        values = {each: tuple(jnp.expand_dims(part, form.axis) for part in value) if each.reduced else value
                  for each, value in values.items()}
    dtype = laid[0].dtype
    products = []
    for coefficient, sums in form.products:
        fraction, power = math.frexp(coefficient)
        product = (jnp.asarray(2 * fraction, dtype), jnp.zeros((), dtype), jnp.asarray(power - 1, jnp.int32))
        for each in sums:
            product = _times(product, values[each])
        products.append(product)
    if len(products) == 1:
        return products[0], denominator
    # The products summed as a sum's terms are (`_quotients_summed`, banded and compensated), each unlanded:
    # joined in turn, a product beneath its neighbours' binade was lost before they cancelled ([2^127,
    # 2^-126, -2^127] summed to 0; the merge P0's base review 10, F5).
    shape = np.broadcast_shapes(*(np.shape(part) for product in products for part in product))
    stacked = tuple(jnp.stack([jnp.broadcast_to(product[place], shape) for product in products], axis=-1)
                    for place in range(3))
    summed = _quotients_summed(stacked, (len(shape),))
    return tuple(jnp.squeeze(part, -1) for part in summed), denominator


def _times(a, b):
    """Two double floats at powers of two multiplied (`_products_summed`), `(high, low, exponent)`: their
    normal forms' high parts' product exact (`two_product`), the low parts' products carried beside it, the
    exponents summed, renormalized into [1, 2): within `2 u^2` of the real product, landing none, so a product
    of sums past the format, or beneath it, keeps its value; zero where either is."""
    a_high, a_low, a_exponent = _normal(*a)
    b_high, b_low, b_exponent = _normal(*b)
    product, residual = two_product(a_high, b_high)
    high, low = two_sum(product, residual + a_high * b_low + a_low * b_high)
    high, low, exponent = _renormalized(high, low, a_exponent + b_exponent, high.dtype)
    zero = (a_high == 0) | (b_high == 0)
    return jnp.where(zero, 0.0, high), jnp.where(zero, 0.0, low), jnp.where(zero, 0, exponent)


def _ratio_sum(operands, form: Ratio):
    """A ratio's value before it lands (`Ratio`), `(high, low, exponent)`: its numerator's sum
    (`_ratio_sums`) over its denominator's (`_quotient`), where it has one, its quotients summed across
    their further axes (`Ratio.across`, `_quotients_summed`)."""
    numerator, denominator = _ratio_sums(operands, form)
    if denominator is None:
        summed = numerator
    else:
        if form.power != 1:
            denominator = _raised(denominator, form.power)
        if not form.reduced:
            denominator = tuple(jnp.expand_dims(part, form.axis) for part in denominator)
        summed = _quotient(numerator, denominator)
    if not form.across:
        return summed
    # The axes as the quotients lie, a reduced form's without its states' axis.
    return _quotients_summed(summed, tuple(place - (form.reduced and place > form.axis)
                                           for place in form.across))


def _raised(summed, power: int):
    """A double float at a power of two (`_ratio_sums`'s) raised to `power`, `(high, low, exponent)`: its
    normal form multiplied by itself in turn as a term's factors are (`_product`): each product of high parts
    exact (`two_product`), the low parts' products carried beside it, the exponents summed, renormalized into
    [1, 2) a step at a time, so the power is within `power u^2` of its real value; zero's zero."""
    high, low, exponent = _normal(*summed)
    made = (high, low, exponent)
    for _ in range(power - 1):
        product, residual = two_product(made[0], high)
        joined, error = two_sum(product, residual + made[0] * low + made[1] * high)
        made = _renormalized(joined, error, made[2] + exponent, high.dtype)
    return made


def _landed(summed, dtype) -> jax.Array:
    """A ratio's value before it lands (`_ratio_sum`'s) landed once into `dtype`, its residual with it (a
    quotient's `1 / (1 + 2^-24)`, the review's F3), a zero as +0."""
    high, low, exponent = summed
    landed = precision.power2(high + low, exponent, reach=_REACH).astype(dtype)
    return jnp.where(landed == 0, jnp.zeros_like(landed), landed)


def _binade(summed, axis: int) -> jax.Array:
    """The power of two a reduced ratio's sum (`_ratio_sum`'s) lands at, as it lands (its high and low parts
    added), laid along `axis` again."""
    high, low, exponent = summed
    return jnp.expand_dims(exponent + scaled.scaled(high + low).exponent, axis)


def _ratio_value(*operands, form: Ratio):
    """A ratio's value (`Ratio`): its sum (`_ratio_sum`) landed once (`_landed`), over its operands' whole
    extent (`_ratio_abstract`'s), each of its sums broadcast to it."""
    landed = _landed(_ratio_sum(operands, form), operands[0].dtype)
    rank = max(operand.ndim for operand in operands)
    shape = np.broadcast_shapes(*(_laid_to(operand, rank).shape for operand in operands))
    whole = tuple(1 if place in (*form.over, *form.across) else size for place, size in enumerate(shape)
                  if not (form.reduced and place == form.axis))
    return jnp.broadcast_to(landed, whole)


def _ratio_abstract(*operands, form: Ratio):
    """A ratio's result type: its operands' broadcast (each laid along the trailing axes, `_laid_to`), its
    sharding the full operand's, without the reduced axis (`_without_axis`), varying along every manual axis
    an operand varies along (`scaled.varying`), weak where every operand is."""
    rank = max(operand.ndim for operand in operands)
    shape = np.broadcast_shapes(*(operand.shape + (1,) * (rank - operand.ndim) for operand in operands))
    full = next((operand for operand in operands if operand.shape == shape), None)
    typed = scaled.varying((full if full is not None else operands[0].update(shape=shape)).update(
        weak_type=all(operand.weak_type for operand in operands)), *operands)
    summed = (*form.over, *form.across)
    if summed:
        spec = (*typed.sharding.spec, *(None,) * (typed.ndim - len(typed.sharding.spec)))
        ones = tuple(1 if place in summed else size for place, size in enumerate(shape))
        held = PartitionSpec(*(None if place in summed else name for place, name in enumerate(spec)))
        typed = typed.update(shape=ones, sharding=typed.sharding.update(spec=held))
    return _without_axis(typed, form.axis) if form.reduced else typed


def _ratio_batched(arguments, dimensions, *, form: Ratio):
    size = next(argument.shape[dimension] for argument, dimension in zip(arguments, dimensions, strict=True)
                if dimension is not None)
    rank = max(argument.ndim - (dimension is not None)
               for argument, dimension in zip(arguments, dimensions, strict=True))
    # Each operand laid to the ratio's rank before its batch axis is put first, so its trailing laying holds.
    fronted = [batching.bdim_at_front(argument if dimension is None else jnp.moveaxis(argument, dimension, 0),
                                      None if dimension is None else 0, size)
               for argument, dimension in zip(arguments, dimensions, strict=True)]
    fronted = [x.reshape(x.shape[:1] + x.shape[1:] + (1,) * (rank + 1 - x.ndim)) for x in fronted]
    return ratio_p.bind(*fronted, form=dataclasses.replace(
        form, axis=form.axis + 1, over=tuple(place + 1 for place in form.over),
        across=tuple(place + 1 for place in form.across),
        products=tuple((coefficient, tuple(_shifted(each) for each in sums))
                       for coefficient, sums in form.products))), 0


def _shifted(each: Summed) -> Summed:
    """A sum of a ratio's products (`Summed`) with its further axes one on, beneath a batch axis put first."""
    return dataclasses.replace(each, over=tuple(place + 1 for place in each.over))


def _derived(term, tangent_of: dict[int, int]) -> list:
    """A term's derivative (`Ratio`), terms again: each factor in turn replaced by its tangent's operand (an
    exponential's kept beside it, times the units' scale, and minus that along its reference), a factor of
    a held operand, and the scale, moving nothing."""
    coefficient, factors = term
    made = []
    for place, factor in enumerate(factors):
        if factor[0] == "s":
            continue
        if factor[0] == "v":
            if factor[1] in tangent_of:
                moved = ("v", tangent_of[factor[1]])
                made.append((coefficient, (*factors[:place], moved, *factors[place + 1:])))
            continue
        _, moving, reference = factor
        # The units' scale a factor of its own (`("s",)`), so its powers multiply exactly in the format.
        if moving in tangent_of:
            made.append((coefficient, (*factors, ("s",), ("v", tangent_of[moving]))))
        if reference in tangent_of:
            made.append((-coefficient, (*factors, ("s",), ("v", tangent_of[reference]))))
    return made


def _compacted(operands: list, form: Ratio):
    """`operands` and `form` with every operand no term reads dropped, the terms' indices renumbered, but one
    that spans the operands' domain where those read do not: a derivative's operands, its tangents', keep the
    domain its primal sums run over, their states' extent among it (the merge P0's base review 9, F3: of
    `sum(x + y)` over `x [1, 1]` and `y [2, 3]`, the derivative along `x` alone, `y` dropped, was `[1]`
    where it is `[2, 2, 2]`)."""
    terms = [*form.numerator, *form.denominator,
             *(term for _, sums in form.products for each in sums for term in each.terms)]
    read = {factor[index] for _, factors in terms for factor in factors for index in range(1, len(factor))}
    rank = max(operand.ndim for operand in operands)

    def domain(indices):
        laid = [tuple(operands[index].shape) + (1,) * (rank - operands[index].ndim) for index in indices]
        return np.broadcast_shapes((1,) * rank, *laid)

    whole = domain(range(len(operands)))
    for index in range(len(operands)):
        if index not in read and domain(read) != whole and domain(read | {index}) != domain(read):
            read.add(index)
    # And its rank: an operand of the domain's rank where none read holds it (of `sum(x + y)` over `x` of
    # shape [1, 1] and a scalar `y`, the derivative along `y` alone kept the scalar, and its states' axis
    # was none: the merge P0's base review 10, F3).
    if read and max(operands[index].ndim for index in read) < rank:
        read.add(next(index for index, operand in enumerate(operands) if operand.ndim == rank))
    renumbered = {old: new for new, old in enumerate(sorted(read))}
    to = renumbered.__getitem__
    return ([operands[index] for index in sorted(read)],
            dataclasses.replace(form, numerator=_remapped(form.numerator, to),
                                denominator=_remapped(form.denominator, to),
                                products=tuple((coefficient, tuple(dataclasses.replace(
                                    each, terms=_remapped(each.terms, to)) for each in sums))
                                    for coefficient, sums in form.products)))


def ratio(operands: list, form: Ratio) -> jax.Array:
    """`ratio_p` over the operands its form reads (`_compacted`)."""
    operands, form = _compacted(list(operands), form)
    return ratio_p.bind(*operands, form=form)


def _remapped(terms, to: Callable[[int], int]) -> tuple:
    """`terms` (`Ratio`'s) with each operand index `k` read as `to(k)`."""
    return tuple((coefficient, tuple((factor[0], *(to(index) for index in factor[1:])) for factor in factors))
                 for coefficient, factors in terms)


def _ratio_jvp(primals, tangents, *, form: Ratio):
    """A ratio's derivative, a ratio again over its operands and their tangents: of `P / D^q` (`P` a sum of
    products of sums, `Ratio.products`), `dP / D^q` where its denominator holds still, else the quotient's
    closed rule `(dP D - q P dD) / D^(q+1)`, each `d` of a sum its terms' (`_derived`), `dP` the products'
    rule (each sum in turn by its derivative): products of sums again, each sum formed once and unlanded,
    so no quotient, and no tangent, is a float formed apart from the terms it cancels against (the merge
    P0's base reviews 4 and 9, F2: a landed quotient, or a tangent of one, past the format or beneath it).
    Every term of it holds one tangent, so it is linear in them, and its transpose is a ratio again
    (`_ratio_transpose`): closed at every order, forward and reverse, each sum's work its states' (no grid
    of them)."""
    value = ratio_p.bind(*primals, form=form)
    operands = list(primals)
    tangent_of = {}
    for index, tangent in enumerate(tangents):
        if type(tangent) is not ad.Zero:
            tangent_of[index] = len(operands)
            operands.append(tangent)
    moved = tuple(made for term in form.denominator for made in _derived(term, tangent_of))
    held = (Summed(form.denominator, reduced=True),) if moved else ()
    products = []
    for coefficient, sums in _products(form):
        for place, each in enumerate(sums):
            derived = tuple(made for term in each.terms for made in _derived(term, tangent_of))
            if derived:
                products.append((coefficient, (*sums[:place], dataclasses.replace(each, terms=derived),
                                               *sums[place + 1:], *held)))
        if moved:
            products.append((-form.power * coefficient, (*sums, Summed(moved, reduced=True))))
    # Its products written once (`_canonical`), where they may cancel whole: `sum(x) / sum(x)`'s two are one
    # product of opposite coefficients, an empty numerator, its tangent zero (the merge P0's base review 11,
    # F2: the empty numerator reached its sum's bits).
    products = _canonical(products)
    if not products:
        return value, ad.Zero(jax.typeof(value).to_tangent_aval())
    derived_form = _classic(dataclasses.replace(form, numerator=(), products=products,
                                                power=form.power + bool(moved)))
    return value, ratio(operands, derived_form)


def _ratio_transpose(cotangent, *operands, form: Ratio):
    """A ratio linear in its undefined operands (`_ratio_jvp`'s), its numerator a sum of products of sums
    (`Ratio.products`; one sum alone the law's own forms, `_products`), each product holding them in one of
    its sums, each term of that sum one of them, the denominator none: each one's cotangent the ratio, a
    state at a time, of each such product with that sum's terms holding it, it replaced by the cotangent
    (laid across the operands' domain, so an operand that alone carried an extent keeps it, the review's
    F5), its other sums and the denominator as they are, summed over the axes the operand is broadcast
    along: within the replaced sum along an axis where the denominator and every other sum of each product
    holding the operand are one (`Summed.over`, a channel's terms summed over one denominator, the base
    review's F1), else its quotients across it (`Ratio.across`: landed apart, opposite quotients past the
    format were NaN, the merge P0's base review F1), landed once; one product of one sum is written as that
    sum (`_classic`), its form the law's own."""
    linear = [ad.is_undefined_primal(operand) for operand in operands]

    def holds(each):
        return any(factor[0] == "v" and linear[factor[1]] for _, factors in each.terms for factor in factors)

    for _, sums in _products(form):
        holding = [each for each in sums if holds(each)]
        if len(holding) != 1 or any(
                sum(factor[0] == "v" and linear[factor[1]] for factor in factors) != 1 or any(
                    factor[0] == "e" and (linear[factor[1]] or linear[factor[2]]) for factor in factors)
                for _, factors in holding[0].terms):
            raise TypeError("a ratio is linear in one operand a term, in one sum a product, never in an "
                            "exponential's")
    if any(linear[index] for _, factors in form.denominator for factor in factors for index in factor[1:]):
        raise TypeError("a ratio's denominator holds no operand it is linear in")
    if type(cotangent) is ad.Zero:
        return [None] * len(operands)
    paired = list(zip(operands, linear, strict=True))
    known = [jnp.zeros((), cotangent.dtype) if moving else operand for operand, moving in paired]
    rank = max((operand.aval if moving else operand).ndim for operand, moving in paired)
    domain = np.broadcast_shapes(*(tuple((operand.aval if moving else operand).shape)
                                   + (1,) * (rank - (operand.aval if moving else operand).ndim)
                                   for operand, moving in paired))
    laid = jnp.expand_dims(cotangent, form.axis) if form.reduced else cotangent
    laid = jnp.broadcast_to(_laid_to(laid, rank), tuple(1 if place in form.over else size
                                                         for place, size in enumerate(domain)))

# The transposed ratio's operands: the known ones, the cotangent (`held`) and a one (`unit`), which
    # takes an operand's place in a sum over the states whose cotangent is summed apart (`_cotangent_of`).
    full, held = [*known, laid, jnp.ones((), cotangent.dtype)], len(known)
    return [_cotangent_of(index, operands[index].aval, form, full, held, rank, domain) if moving else None
            for index, moving in enumerate(linear)]


def _extent(full: list, rank: int, form: Ratio, terms, summed: Summed | None = None) -> tuple[int, ...]:
    """The extent of `terms`' operands in `full`, laid to the ratio's `rank` (past it where an operand is laid
    to fresh axes, `_moved_apart`), each padded after its own axes: a sum's one along its own sums' axes."""
    read = {index for _, factors in terms for factor in factors for index in factor[1:]}
    width = max([rank, *(np.ndim(full[i]) for i in read)])
    made = np.broadcast_shapes((1,) * width, *(
        tuple(np.shape(full[i])) + (1,) * (width - np.ndim(full[i])) for i in read))
    if summed is None:
        return made
    along = (*summed.over, *((form.axis,) if summed.reduced else ()))
    return tuple(1 if place in along else size for place, size in enumerate(made))


def _cotangent_of(index: int, aval, form: Ratio, full: list, held: int, rank: int, domain: tuple[int, ...]):
    """`_ratio_transpose`'s cotangent of its operand `index` (of `aval`'s shape and dtype), `full` the
    transposed ratio's operands (the known ones, the cotangent laid across the domain at `held`, a one after
    it, then any laid to fresh axes), or None where no product holds it or its products cancel whole."""
    unit = held + 1
    shape = aval.shape
    own = shape + (1,) * (rank - len(shape))
    broadcast = [place for place, (size, mine) in enumerate(zip(domain, own, strict=True))
                 if mine == 1 and size != 1]
    holding = [(coefficient, sums, place) for coefficient, sums in _products(form)
               for place, each in enumerate(sums)
               if any(("v", index) in factors for _, factors in each.terms)]
    if not holding:
        return None
    # The axes the sums holding the operand sum it over (`Summed.over`): the cotangent and every factor
    # outside those sums (each product's other sums, the denominator) read the ratio's positions along them,
    # the operand's own positions apart, so they are taken to fresh axes and summed across there (of
    # `sum_k x_k sum_d y_id z_ic`, `y_id`'s cotangent `sum_c t_ic (sum_k x_k) z_ic`, the same at every `d`:
    # the merge P0's base review 13, F1).
    reduced = {sums[place].over for _, sums, place in holding}
    if len(reduced) != 1:
        raise TypeError("a ratio's operand is summed over other axes in one product than in another")
    (own_axes,) = reduced
    fresh = {axis: rank + place for place, axis in enumerate(own_axes)}
    outside = functools.partial(_taken_apart, full, fresh, {})
    denominator_terms = outside(form.denominator)
    cotangent_at = _moved_apart(full, held, fresh) if fresh else held
    # Each product's terms holding the operand, it replaced: by the cotangent where the product's value at a
    # state reads the operand at that state (its sum a state at a time, or the ratio reduced over the states,
    # its cotangent one along them); by a one where its sum is over the states and the ratio is not, the
    # cotangent then summed over the states against the product's sums a state at a time (`folded`): of
    # `P_j sum_k t_k x_k`, `x_i`'s is `t_i sum_j c_j P_j`, never `c_i P_i t_i`.
    built = []
    for coefficient, sums, place in holding:
        folding = sums[place].reduced and not form.reduced
        put = ("v", unit if folding else cotangent_at)
        replaced = tuple((term, tuple(put if factor == ("v", index) else factor for factor in factors))
                         for term, factors in sums[place].terms if ("v", index) in factors)
        # Each other sum's axes, those taken apart at their fresh ones (never 0: past the ratio's rank).
        others = [dataclasses.replace(each, terms=outside(each.terms),
                                      over=tuple(sorted(fresh.get(axis) or axis for axis in each.over)))
                  for spot, each in enumerate(sums) if spot != place]
        if folding:
            # Their further axes, over which the product of their sums is one sum of their terms' products
            # (of `sum_k x_k sum_c y_ic`, `x`'s cotangent `sum_i t_i sum_c y_ic`: the merge P0's base review
            # 10, F4), each sum's own fresh axes (`_apart_axes`), apart from one another's and from every
            # axis a factor beside them or the cotangent keeps (base reviews 11 F3, 12 F1).
            apart = _apart_axes([each for each in others if not each.reduced], full, rank + len(fresh))
            folded = Summed(tuple(
                (math.prod(term for term, _ in combination),
                 (*(factor for _, factors in combination for factor in factors), ("v", cotangent_at)))
                for combination in itertools.product(*(each.terms for each in apart))), reduced=True,
                over=tuple(sorted(axis for each in apart for axis in each.over)))
            others = [*(each for each in others if each.reduced), folded]
        built.append((coefficient, replaced, others))
    # Within the replaced sum where the denominator and every other factor of each product holding the
    # operand are one along the axis, its quotients summed across it elsewhere.
    denominator = _extent(full, rank, form, denominator_terms)
    within = tuple(axis for axis in broadcast if denominator[axis] == 1 and all(
        _extent(full, rank, form, other.terms, other)[axis] == 1
        for _, _, others in built for other in others))
    products = _canonical((coefficient, (Summed(replaced, reduced=False, over=within), *others))
                          for coefficient, replaced, others in built)
    if not products:
        # Its products cancelled whole (`_canonical`): its cotangent zero.
        return None
    transposed = _classic(dataclasses.replace(
        form, numerator=(), denominator=denominator_terms, products=products, reduced=False, over=within,
        across=(*(axis for axis in broadcast if axis not in within), *fresh.values())))
    made = ratio(full, transposed)
    # Laid to the ratio's rank: along a fresh axis (a sum's channels', `_apart_axes`, or the operand's sum's,
    # summed across) the value one or broadcast, so its first place is the whole.
    made = _laid_to(made, rank) if made.ndim < rank else made[(..., *(0,) * (made.ndim - rank))]
    return jnp.broadcast_to(made, own).reshape(shape).astype(aval.dtype)


def _taken_apart(full: list, fresh: dict[int, int], moved: dict[int, int], terms):
    """`terms` (a sum's, `Ratio`'s) reading their operands laid to `fresh`'s axes (`_moved_apart`, each
    operand moved once, `moved` its index in `full` so far): the terms outside the sum a transposed operand is
    summed in, at the ratio's positions along that sum's own axes (`_ratio_transpose`); `terms` as they are
    where `fresh` is empty."""
    if not fresh:
        return terms

    def at(index: int) -> int:
        if index not in moved:
            moved[index] = _moved_apart(full, index, fresh)
        return moved[index]

    return tuple((coefficient, tuple((factor[0], *(at(index) for index in factor[1:])) for factor in factors))
                 for coefficient, factors in terms)


def _apart_axes(sums: list[Summed], full: list, rank: int) -> list[Summed]:
    """`sums`, each over further axes of its own (`Summed.over`): every axis a sum reduces taken to a fresh
    axis past the ratio's `rank` and every one taken before, the operands its terms read laid so (each
    appended to `full`, its extent along the axis moved there), so the product of the sums, with the
    factors beside them and the cotangent along the axes those keep, is one sum of their terms' products over
    all their axes (`_ratio_transpose`'s folded sum): two sums over one channel are their product (of
    `sum_k x_k sum_c y_ic sum_c z_ic`, `x`'s cotangent `sum_i t_i (sum_c y_ic)(sum_c z_ic)`: the merge P0's
    base review 11, F3), and a channel sum beside a factor that keeps the channel is its product with that
    factor, not their contraction (of `sum_k x_k sum_d y_id z_ic`, `x`'s cotangent `sum_ic t_ic (sum_d y_id)
    z_ic`: base review 12, F1)."""
    made, top = [], rank - 1
    for each in sums:
        if not each.over:
            made.append(each)
            continue
        fresh = {axis: top + 1 + place for place, axis in enumerate(each.over)}
        top += len(each.over)
        moved: dict[int, int] = {}
        for index in dict.fromkeys(index for _, factors in each.terms for factor in factors
                                   for index in factor[1:]):
            moved[index] = _moved_apart(full, index, fresh)
        terms = tuple((coefficient, tuple((factor[0], *(moved[index] for index in factor[1:]))
                                          for factor in factors)) for coefficient, factors in each.terms)
        made.append(dataclasses.replace(each, terms=terms, over=tuple(sorted(fresh.values()))))
    return made


def _moved_apart(full: list, index: int, fresh: dict[int, int]) -> int:
    """The index in `full` of operand `index` laid to the fresh axes (`_apart_axes`), its extent along each
    of `fresh`'s axes moved to the fresh one it names (appended)."""
    operand = full[index]
    operand = operand.reshape(np.shape(operand) + (1,) * (max(fresh.values()) + 1 - np.ndim(operand)))
    for axis, to in fresh.items():
        operand = jnp.swapaxes(operand, axis, to)
    full.append(operand)
    return len(full) - 1


ratio_p = scaled.primitive("convex_ratio", _ratio_value, _ratio_abstract, _ratio_batched)
"""A ratio of sums of products of float operands (`Ratio`, `_ratio_value`), an operation of its own closed
under differentiation (`_ratio_jvp`) and transposition (`_ratio_transpose`): the law's every value and
every derivative is one, its terms formed exactly and summed compensated at a power of two common to the
row, landed once, so no tangent is a float formed apart from its sum (the degenerate sweep's class: a
coefficient's tangent flushed or overflowed apart from the terms it cancels against)."""
ad.primitive_jvps[ratio_p] = _ratio_jvp
ad.primitive_transposes[ratio_p] = _ratio_transpose


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


def _without_axis(operand, axis: int):
    """`operand`'s type without its states' `axis` (its shape and its sharding's spec), its dtype, weak type,
    memory and manual axes (`shard_map`'s varying ones) its own: a merged result's (base review F3: a spec
    kept whole beside a shape without the axis is refused under an explicit mesh)."""
    spec = (*operand.sharding.spec, *(None,) * (operand.ndim - len(operand.sharding.spec)))
    kept = PartitionSpec(*spec[:axis], *spec[axis + 1:])
    return operand.update(shape=tuple(size for place, size in enumerate(operand.shape) if place != axis),
                          sharding=operand.sharding.update(spec=kept))


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
