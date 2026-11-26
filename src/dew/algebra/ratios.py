"""The `Ratio` primitive (`ratio_p`) and its forward/reverse derivatives (`_ratio_jvp`, `_ratio_transpose`).

Every value and every derivative of the convex mean law (`dew.algebra.means`) is a ratio of sums of products
of its float operands evaluated through `dew.algebra.windows`'s windowed compensated-sum engine."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.lax import lax as lax_internal
from jax.interpreters import ad, batching
from jax.sharding import PartitionSpec

from dew.algebra import bands, scaled, windows
from dew.algebra.windows import (
    Units,
    _arrays,
    _laid_to,
    _landed,
    _quotient,
    _quotients_summed,
    _raised,
    _streamed,
    _Sum,
    _times,
    _without_axis,
    chunk_width,
)
from dew.nn.float_bits import raw_fields
from dew.nn.precision import at_least_fp32


@dataclasses.dataclass(frozen=True)
class Ratio:
    """A ratio of sums of products of float operands (`ratio_p`): `sum numerator / sum denominator`, each a
    tuple of terms `(coefficient, factors)`, a factor `("v", i)` operand `i`, `("e", i, j)` the factor
    `exp(s (operand_i - operand_j))` of `units` (`_factors_value`) or `("s",)` the units' scale `s` in the
    operands' format, or `("m", i)` a held zero/one term mask. The numerator is summed over the operands'
    `axis` where `reduced` (the result without
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
    With track_directions, a single affine direction is streamed with its coefficients. Independent
    directional sums keep their dyadic bands until all products have met the final reduction.
    A plain product sum can retain IEEE nonfinite terms rather than annihilate them with zero
    (ieee_products, a portable recurrence pairing). This choice follows the term through AD.
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
    track_directions: bool = False
    directions: tuple[int, ...] = ()
    ieee_products: bool = False


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


def _product_of(coefficient, sums, values, dtype):
    fraction, power = math.frexp(coefficient)
    if sums and abs(fraction) == 0.5:
        hi, lo, exponent = values[sums[0]]
        sums = sums[1:]
        if fraction < 0:
            hi, lo = -hi, -lo
        product = hi, lo, exponent if power == 1 else exponent + (power - 1)
    else:
        product = (jnp.asarray(2 * fraction, dtype), jnp.zeros((), dtype), jnp.asarray(power - 1, jnp.int32))
    for each in sums:
        product = _times(product, values[each])
    return product


def _carries_direction(each: Summed, active: frozenset[int]) -> bool:
    return each.reduced and not each.over and any(
        active.intersection(factor[1:]) for _, factors in each.terms for factor in factors)


def _directional_sums(form: Ratio) -> tuple[Summed, ...]:
    """Keep independent direction sums, and sums that a pullback subtracts from entrywise values."""
    active = frozenset(form.directions)
    sums = [tuple(each for each in term if _carries_direction(each, active)) for _, term in form.products]
    if form.reduced and not any(len(term) > 1 for term in sums):
        return ()
    return tuple(dict.fromkeys(each for term in sums for each in term))


def _directional_products(form: Ratio):
    """Each product's one affine sum, separated from its coefficients."""
    if not form.reduced or not any(form.directions):
        return ()
    active = frozenset(form.directions)
    plans = []
    for coefficient, sums in form.products:
        chosen = tuple(at for at, each in enumerate(sums) if _carries_direction(each, active))
        if len(chosen) != 1:
            return ()
        (at,) = chosen
        plans.append((coefficient, sums[at], sums[:at] + sums[at + 1:]))
    return plans


def _directions_summed(plans, values, arrays, count: int, rank: int, form: Ratio, dtype):
    """Cancel a linear direction's coefficients before reducing across samples."""
    load, chunks = arrays
    summed = [(_Sum(direction.terms, form.axis, (), count), tuple(jnp.expand_dims(part, form.axis)
               for part in _product_of(coefficient, sums, values, dtype)))
              for coefficient, direction, sums in plans]

    def summands_of(chunk):
        inputs, valid = load(chunk)
        factors = windows._factor_values(inputs, form.units, rank)
        terms = []
        for each, multiplier in summed:
            for parts in each._placed_summands(inputs, factors, valid, dtype, empty_exponent=0):
                hi, lo, exponent = _times(parts, multiplier)
                terms.append((hi, lo, jnp.where(hi != 0, exponent, windows._NONE)))
        return [jax.lax.optimization_barrier(tuple(terms))]

    bits = (count * sum(len(each.terms) for each, _ in summed)).bit_length()
    (value,) = windows._windows(summands_of, chunks, [(form.axis,)], [bits], dtype)
    return tuple(jnp.squeeze(part, form.axis) for part in value)


def _banded_sums(expanded: tuple[Summed, ...], arrays, count: int, rank: int, form: Ratio, dtype):
    """Keep each directional sum's bins separate, sharing factors and digit extraction across sums
    that have the same number of terms. The common grid encloses every sum in a group."""
    load, chunks = arrays
    number = scaled.format_of(dtype)
    capacity = min(jnp.iinfo(jnp.int32).bits - 1, number.mantissa + 1)
    groups: dict[int, list[Summed]] = {}
    for each in expanded:
        groups.setdefault(len(each.terms), []).append(each)
    made = {}
    for size, group in groups.items():
        terms_per_bin = 2 * count * size
        width = max(1, min(16, capacity - terms_per_bin.bit_length()))
        powers = [(math.frexp(coefficient)[1], len(factors))
                  for each in group for coefficient, factors in each.terms]
        lower = min(exponent - (degree + 2) * number.reach - number.mantissa for exponent, degree in powers)
        upper = max(exponent + (degree + 1) * (number.largest + 2) for exponent, degree in powers)
        may_round = terms_per_bin * ((1 << width) - 1) > 1 << (number.mantissa + 1)
        summations = tuple(_Sum(each.terms, form.axis, (), count) for each in group)

        def terms(chunk, summations=summations):
            inputs, valid = load(chunk)
            factors = windows._factor_values(inputs, form.units, rank)
            terms_by_sum = [each._placed_summands(inputs, factors, valid, dtype, empty_exponent=0)
                            for each in summations]
            if len(terms_by_sum) == 1:
                return terms_by_sum[0]
            return tuple(tuple(jnp.stack(jnp.broadcast_arrays(*part), axis=-1)
                               for part in zip(*pieces, strict=True))
                         for pieces in zip(*terms_by_sum, strict=True))

        placed, live = bands.summed(terms, chunks, axis=form.axis, width=width, lower=lower,
                                    bins=(upper - lower) // width + 1, dtype=dtype, may_round=may_round)
        if len(group) == 1:
            made[group[0]] = placed, live
        else:
            for at, each in enumerate(group):
                made[each] = tuple(part[..., at, :] for part in placed), live[..., at]
    return made


def _band_products(products, values, expanded, dtype):
    """Multiply independent sums over their bin combinations, without materializing a sample grid."""
    plans = [(coefficient, sums, math.prod(expanded[each][1] for each in sums if each in expanded),
              math.prod(expanded[each][0][0].shape[-1] for each in sums if each in expanded))
             for coefficient, sums in products]
    length = max(bound for _, _, _, bound in plans)
    width = chunk_width(length)
    live_length = functools.reduce(jnp.maximum,
                                   [jnp.max(jnp.asarray(steps), initial=0) for _, _, steps, _ in plans])

    def stacked(arrays):
        return jnp.stack(jnp.broadcast_arrays(*arrays), axis=-2)

    def block(chunk):
        term = chunk * width + jnp.arange(width, dtype=jnp.int32)
        by_arity: dict[int, list] = {}
        for coefficient, sums, steps, _ in plans:
            taken = []
            index = term
            for each in sums:
                if each in expanded:
                    parts, count = expanded[each]
                    index, pos = jnp.divmod(index, jnp.maximum(count[..., None], 1))
                    taken.append(tuple(jnp.take_along_axis(part, pos, axis=-1) for part in parts))
                else:
                    taken.append(tuple(part[..., None] for part in values[each]))
            first = _product_of(coefficient, (0,) if taken else (), taken, dtype)
            by_arity.setdefault(len(taken), []).append((term < jnp.asarray(steps)[..., None],
                                                        (first, *taken[1:])))
        found = []
        # One multiplier per factor position and arity. A separate multiplier graph per product
        # duplicates compensated arithmetic in both passes of every window and for every derivative.
        for group in by_arity.values():
            masks, factors = zip(*group, strict=True)
            operands = [tuple(stacked(arrays) for arrays in zip(*factor, strict=True))
                        for factor in zip(*factors, strict=True)]
            hi, lo, exponent = functools.reduce(_times, operands)
            live = stacked(masks) & (hi != 0)
            found.append((jnp.where(live, hi, 0), jnp.where(live, lo, 0),
                          jnp.where(live, exponent, windows._NONE)))
        return [jax.lax.optimization_barrier(tuple(found))]

    rank = next(iter(expanded.values()))[0][0].ndim - 1
    axes = (rank, rank + 1)
    bits = sum(bound for _, _, _, bound in plans).bit_length()
    (value,) = windows._windows(block, (live_length + width - 1) // width, [axes], [bits], dtype)
    return tuple(jnp.squeeze(part, axes) for part in value)


def _products_summed(operands, form: Ratio):
    """A numerator of products of sums (`Ratio.products`) and its denominator's sum, each `(high, low,
    exponent)` (`_ratio_sums`'s): each distinct sum formed once (`_streamed`: those over the states in one
    pass with the denominator, their factors formed once for all, those a state at a time in another), each
    product its coefficient times its sums in turn (`_times`, a double float's product, landing none), the
    products joined (`joined`); a reduced sum laid along the states' axis again where the form keeps it."""
    rank = max(operand.ndim for operand in operands)
    laid = [_laid_to(operand, rank) for operand in operands]
    states = max(operand.shape[form.axis] for operand in laid)
    count = -(-states // chunk_width(states)) * chunk_width(states)
    # The denominator's own sum where a product takes it (the quotient rule's `D`), formed once for both.
    own = Summed(form.denominator, reduced=True) if form.denominator else None
    expanded = _directional_sums(form)
    directional = () if expanded else _directional_products(form)
    products = (sums for _, _, sums in directional) if directional else (sums for _, sums in form.products)
    distinct = list(dict.fromkeys(each for sums in products for each in sums if each not in expanded))
    reduced = [each for each in distinct if each.reduced and each != own]
    entrywise = [each for each in distinct if not each.reduced]
    over_states = [_Sum(each.terms, form.axis, each.over, count) for each in reduced]
    if form.denominator:
        over_states.append(_Sum(form.denominator, form.axis, (), count))
    arrays = _arrays(laid, form.axis)
    made = _streamed(*arrays, over_states, form.units) if over_states else []
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
    dtype = at_least_fp32(laid[0].dtype)
    if expanded:
        values_by_band = _banded_sums(expanded, arrays, count, rank, form, dtype)
        if not form.reduced:
            values_by_band = {each: jax.tree.map(lambda part: jnp.expand_dims(part, form.axis), band)
                              for each, band in values_by_band.items()}
        return _band_products(form.products, values, values_by_band, dtype), denominator
    if directional:
        return _directions_summed(directional, values, arrays, count, rank, form, dtype), denominator
    products = [_product_of(coefficient, sums, values, dtype) for coefficient, sums in form.products]
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



def _exceptional_products(operands, form: Ratio, rank: int):
    """IEEE nonfinite products, summed on the same axes as the plain product sum."""
    held = []
    for operand in operands:
        laid = _laid_to(operand, rank)
        finite = jnp.isfinite(laid)
        _, _, exponent, mantissa = raw_fields(laid)
        sign = jnp.copysign(jnp.where((exponent == 0) & (mantissa == 0), 0.0, 1.0), laid)
        held.append((finite, jnp.where(finite, sign, laid)))
    axes = (*form.over, form.axis) if form.reduced else form.over
    terms = []
    for coefficient, factors in form.numerator:
        values = [held[index] for kind, index in factors if kind != "m"]
        exceptional = ~functools.reduce(jnp.logical_and, (keep for keep, _ in values))
        for kind, index in factors:
            if kind == "m":
                exceptional = exceptional & (_laid_to(operands[index], rank) != 0)
        sign = functools.reduce(jnp.multiply, (value for _, value in values), coefficient)
        terms.append(jnp.sum(jnp.where(exceptional, sign, 0.0), axis=axes, keepdims=True))
    total = functools.reduce(jnp.add, terms)
    total = jnp.sum(total, axis=form.across, keepdims=True)
    return jnp.squeeze(total, form.axis) if form.reduced else total


def _ratio_value(*operands, form: Ratio):
    """A ratio's value (`Ratio`): its sum (`_ratio_sum`) landed once (`_landed`), over its operands' whole
    extent (`_ratio_abstract`'s), each of its sums broadcast to it."""
    rank = max(operand.ndim for operand in operands)
    if form.ieee_products:
        finite = tuple(jnp.where(jnp.isfinite(operand), operand, 0.0) for operand in operands)
        total = _landed(_ratio_sum(finite, form), operands[0].dtype)
        special = _exceptional_products(operands, form, rank)
        landed = jnp.where(jnp.isfinite(special), total, special)
    else:
        landed = _landed(_ratio_sum(operands, form), operands[0].dtype)
    shape = np.broadcast_shapes(*(_laid_to(operand, rank).shape for operand in operands))
    whole = tuple(1 if place in (*form.over, *form.across) else size for place, size in enumerate(shape)
                  if not (form.reduced and place == form.axis))
    return jnp.broadcast_to(landed, whole)


def _ratio_abstract(*operands, form: Ratio):
    """A ratio's broadcast type over the operands laid along trailing axes, without the reduced axis;
    varying along every manual axis an input varies along and weak where every operand is."""
    rank = max(operand.ndim for operand in operands)
    shape = np.broadcast_shapes(*(operand.shape + (1,) * (rank - operand.ndim) for operand in operands))
    full = next((operand for operand in operands if operand.shape == shape), None)
    base = full if full is not None else operands[0].update(shape=shape)
    if full is None and any(operand.sharding.mesh.explicit_axes for operand in operands):
        laid = [operand.update(shape=operand.shape + (1,) * (rank - operand.ndim)) for operand in operands]
        base = base.update(sharding=lax_internal.broadcasting_sharding_rule("algebra_ratio", *laid))
    typed = scaled.varying(base.update(weak_type=all(operand.weak_type for operand in operands)), *operands)
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
        if factor[0] in ("s", "m"):
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
                                directions=tuple(to(i) for i in form.directions if i in read),
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
    directions = (*form.directions, *tangent_of.values()) if form.track_directions else ()
    derived_form = _classic(dataclasses.replace(form, numerator=(), products=products,
                                                power=form.power + bool(moved), directions=directions))
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
    moved: dict[int, int] = {}
    outside = functools.partial(_taken_apart, full, fresh, moved)
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
        across=(*(axis for axis in broadcast if axis not in within), *fresh.values()),
        directions=tuple(direction for i in form.directions if i != index
                         for direction in ((i, moved[i]) if i in moved else (i,)))))
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

