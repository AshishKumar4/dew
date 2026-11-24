"""The aligned sum of exact products and its cotangents.

`aligned_sum` returns the exact sum of its terms' exact products, rounded once to nearest, ties to even. Its
tangent and transpose are formed as its value is. `aligned_cotangents` returns the cotangents its transpose
gives each factor, for a backward that takes no autodiff. Both run integer arithmetic on the factors' bits
(`dew.nn.float_bits`), so a Pallas TPU kernel forms them in place."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from numpy.lib.array_utils import normalize_axis_tuple

from dew.nn.derivatives import Protected, contracted, declared_tangent
from dew.nn.float_bits import cast_back, float_bits, split_bits, widened_exactly


def aligned_sum(terms: Sequence[Sequence[jax.Array]], axis: int | tuple[int, ...] | None = None) -> jax.Array:
    """The sum of `terms` in float32 or float64 (their widest, at least float32), each the broadcast product
    of its factors, summed over `axis` of each product too where given (kept): the exact sum of the exact
    products, rounded once to nearest, ties to even (`_exact_sum`), so a sum in range is its value though its
    terms leave float32's range and cancel: entry writes of keys of +-2^60 and values of 2^-20 under a weight
    of 2^100, each term 2^140 and their sum zero, where summed as formed it was inf - inf, NaN (base review
    ead11c83a F1), and no order of terms that cancel drops a smaller one, as floats summed did, each sum
    rounding (base reviews 224fcf53a to af2cca564, row 8757). A product a term at a time (`twice_scaled`)
    holds each term's range, not a sum's. Its tangent is its terms' own, each factor's direction times its
    term's other factors, and each factor's gradient the cotangent times its terms' other factors, each summed
    aligned as its value is (`_aligned_directions`): its bits carry none, so differentiated as formed every
    factor's was zero, and summed plainly a cancelling entry write's terms of 2^160 were inf - inf (row 8501).
    A factor several terms share (one array: an entry write's weight) is one factor, its gradient one aligned
    sum over them. Elementwise and over a 2-D array's axes, so a Pallas TPU kernel forms it in place. At most
    `2^31 - 1` products a value in float32 (`2^63 - 1` in float64), else a ValueError (row 8757)."""
    return _aligned_terms(terms, ((axis, None),) * len(terms))


def aligned_segment_sum(terms: Sequence[Sequence[jax.Array]], segments: jax.Array | None, count: int,
                        axes: tuple[str, ...] = ()) -> jax.Array:
    """`aligned_sum`'s exact sum of `terms`' products, each product's rows (its first axis) summed into the
    row `segments` names among `count` (int32, one a row: a gather's table, its transpose summing each route
    into its token; each row its own, placed as it is, where None), then over the manual axes `axes` names
    where its products vary over them (each
    device's routes of a token every device holds), rounded once to nearest, ties to even: routes of +-2^140
    from two devices sum to 0 beside a 2^-38 one, which no order of floats summed keeps. Its value alone: a
    caller declares its derivative. Every device of `axes` walks the same digits, their bounds agreed
    (`_levels`), as each digit's sum is a collective over them."""
    _, factors, shape, spec = _laid_out(terms, ((None, None),) * len(terms))
    return _exact_sum(_regrouped(shape, factors), spec, factors[0].dtype,
                      segments=(segments, count, tuple(axes)))


def _segmented(segments: jax.Array, count: int, axes: tuple[str, ...]) -> Callable:
    """`_reduction`'s counterpart for `aligned_segment_sum`: a term's rows scatter-added into `count`
    segments by `segments`, then summed over each of `axes` its sums vary over (each device's sum alone where
    every device holds the same), and the most products one sum holds."""
    from dew.nn.sharding import varying_like

    devices = math.prod(jax.lax.axis_size(axis) for axis in axes)

    def reduction(shape: tuple[int, ...], axis, laid) -> tuple[Callable, int]:
        del axis, laid

        def reduced(x: jax.Array, op: Callable = jnp.sum, **options) -> jax.Array:
            assert op is jnp.sum, "a segment's digits are summed"
            x = jnp.asarray(x, options.get("dtype", jnp.asarray(x).dtype))
            x = jnp.broadcast_to(x, shape)
            if segments is None:
                summed = x
            else:
                zeros = varying_like(jnp.zeros((count, *shape[1:]), x.dtype), x, segments)
                summed = zeros.at[segments].add(x)
            over = tuple(axis for axis in axes if axis in jax.typeof(summed).mat.varying)
            return jax.lax.psum(summed, over) if over else summed

        return reduced, (1 if segments is None else shape[0]) * devices

    return reduction


def _aligned_terms(terms: Sequence[Sequence[jax.Array]], spec: tuple) -> jax.Array:
    """`aligned_sum` of `terms`, each product summed over its `spec` entry's axis (kept) and, where it gives
    `(held, shape)`, broadcast to `held` and reshaped to `shape` (a gradient's term in its factor's)."""
    _, factors, shape, spec = _laid_out(terms, spec)
    from dew.nn.kernels.launches import interpreting

    # A compiled kernel's body bare, as Mosaic lowers no symbolic zeros' rule; an interpreted one declared.
    return (_aligned_value if interpreting() is False else _aligned)(shape, spec, *factors)


def _laid_out(terms: Sequence[Sequence[jax.Array]], spec: tuple) -> tuple[list, list, tuple, tuple]:
    """`terms`' distinct factors (one array several terms share, by identity, once, in order of first
    appearance) as given and widened exactly to their sum's type, each term's factors' places among them and
    `spec` with each term's axes normalized: what `aligned_sum` and its transpose read."""
    dtype = jnp.result_type(jnp.float32, *(factor for term in terms for factor in term))
    originals = []
    # Each factor once, by identity: a shared factor's gradient sums its terms' together, aligned.
    factors, places, shape = [], {}, []
    for term in terms:
        indices = []
        for factor in term:
            if id(factor) not in places:
                places[id(factor)] = len(factors)
                originals.append(factor)
                # Its tangent the cast's, a subnormal kept where XLA:CPU's cast flushes it (row 8966).
                factors.append(widened_exactly(factor, dtype))
            indices.append(places[id(factor)])
        shape.append(tuple(indices))
    # Each term's axes checked and made one sorted tuple of nonnegative axes once, as numpy normalizes them
    # (an AxisError, a ValueError, for one out of range or repeated), for its value and transpose alike
    # (rows 8837, 8974).
    ranks = [max(map(np.ndim, term)) for term in terms]
    spec = tuple((axis if axis is None else _normalized_axes(axis, rank), laid)
                 for rank, (axis, laid) in zip(ranks, spec, strict=True))
    return originals, factors, tuple(shape), spec


def _normalized_axes(axis: int | Sequence[int], rank: int) -> tuple[int, ...]:
    """`axis` of a term of `rank` as numpy normalizes it (`normalize_axis_tuple`), sorted: a ValueError for
    one out of range or repeated, an axis past a C int's range one too, where numpy's conversion of it
    raised an OverflowError (row 9086)."""
    try:
        return tuple(sorted(normalize_axis_tuple(axis, rank)))
    except OverflowError as error:
        raise ValueError(f"axis {axis} is out of bounds for a term of rank {rank}") from error


def _regrouped(shape: tuple[tuple[int, ...], ...], flat: Sequence[jax.Array]) -> list[tuple[jax.Array, ...]]:
    """`aligned_sum`'s terms from its factors, `shape` each term's factors' places among them."""
    return [tuple(flat[at] for at in term) for term in shape]


def _aligned_value(shape: tuple[tuple[int, ...], ...], spec: tuple, *flat: jax.Array) -> jax.Array:
    """`_aligned_terms` of its terms' factors in a row (`_regrouped`), summed exactly (`_exact_sum`)."""
    return _exact_sum(_regrouped(shape, flat), spec, flat[0].dtype)


class _Product(NamedTuple):
    """A term's product, exactly (`_product`): its limbs, its lowest bit's place, its sign and IEEE class."""

    limbs: tuple[jax.Array, ...]
    place: jax.Array
    negative: jax.Array
    nan: jax.Array
    infinite: jax.Array
    zero: jax.Array


def _product(term: Sequence[jax.Array], ints) -> _Product:
    """`term`'s product (`_Product`), exactly, from its factors' bits: each significand two digits of half
    its bits, convolved and carried a factor at a time (`k` factors `k` limbs); NaN, or a signed infinity, as
    IEEE's, no finite factors' rounded product (`2^-75 2^-75` zero) meeting the infinity (row 8757)."""
    limits = jnp.finfo(term[0].dtype)
    half, top = split_bits(term[0].dtype)[2], 2 * limits.maxexp - 1
    low = (1 << half) - 1
    read = [float_bits(factor) for factor in term]
    special = [biased == top for _, biased, *_ in read]
    zeros = [(biased == 0) & (fraction == 0) for _, biased, fraction, *_ in read]
    nan, infinite, zero = (functools.reduce(jnp.logical_or, flags) for flags in (
        [out & (fraction != 0) for out, (_, _, fraction, *_) in zip(special, read, strict=True)],
        [out & (fraction == 0) for out, (_, _, fraction, *_) in zip(special, read, strict=True)], zeros))
    negative = functools.reduce(jnp.logical_xor, [sign != 0 for sign, _, _, _, _ in read])
    place = functools.reduce(jnp.add, [(exponent - limits.nmant).astype(ints) for *_, exponent in read])
    digits: list[jax.Array] = []
    for (_, _, _, normalized, _), out, nothing in zip(read, special, zeros, strict=True):
        mantissa = jnp.where(out | nothing, 0, normalized | (1 << limits.nmant)).astype(ints)
        pair = (mantissa & low, mantissa >> half)
        if not digits:
            digits = list(pair)
            continue
        sums: list = [0] * (len(digits) + 1)
        for at, digit in enumerate(digits):
            for by, other in enumerate(pair):
                sums[at + by] = sums[at + by] + digit * other
        digits, carry = [], 0
        for total in sums:
            total = total + carry
            digits.append(total & low)
            carry = total >> half
        digits.append(jnp.asarray(carry, ints))
    limbs = tuple(digits[at] | (digits[at + 1] << half) for at in range(0, len(digits), 2))
    shape = jnp.broadcast_shapes(*(np.shape(limb) for limb in limbs), np.shape(place), np.shape(negative))
    return _Product(tuple(jnp.broadcast_to(limb, shape) for limb in limbs), jnp.broadcast_to(place, shape),
                    jnp.broadcast_to(negative, shape), nan | (infinite & zero), infinite, zero)


def _reduction(shape: tuple[int, ...], axis, laid) -> tuple[Callable, int]:
    """A term's reduction over its axes one at a time (Mosaic TPU x64, 8807) and laid, and its count."""
    axes = axis or ()

    def reduced(x: jax.Array, op: Callable = jnp.sum, **options) -> jax.Array:
        for a in axes:
            x = op(x, axis=a, keepdims=True, **options)
        return x if laid is None else jnp.broadcast_to(x, laid[0]).reshape(laid[1])

    return reduced, math.prod(shape[a] for a in axes)


def _exact_sum(terms, spec, dtype, segments: tuple | None = None) -> jax.Array:
    """The sum `_aligned_terms` states, exact: every product exact (`_product`), summed as whole digits and
    rounded once to nearest, ties to even (`_nearest`); IEEE's non-finite sums and signed zeros (row 8757).
    With `segments` (`aligned_segment_sum`'s table, count and axes), each product's rows summed into theirs
    (`_segmented`)."""
    reduction = _reduction if segments is None else _segmented(*segments)
    axes = () if segments is None else segments[2]
    limits = jnp.finfo(dtype)
    ints = jnp.dtype(f"int{limits.bits}")
    factors = max(map(len, terms))
    parts, nan, positive, negative, live, held = [], False, False, False, False, False
    for term, (axis, laid) in zip(terms, spec, strict=True):
        product = _product(term, ints)
        reduced, extent = reduction(product.place.shape, axis, laid)
        parts.append((product, reduced, extent))
        # Sum in `ints` (not `jnp.any`, float64 under Mosaic TPU x64, 8807); negative zeros alone stay -0.
        flags = (product.nan, product.infinite & ~product.negative, product.infinite & product.negative,
                 ~(product.zero & product.negative))
        nan, positive, negative, live = (p | (reduced(f.astype(ints), jnp.sum, dtype=ints) != 0)
                                         for p, f in zip((nan, positive, negative, live), flags, strict=True))
        if segments is not None:
            # Whether a segment holds any product: one that holds none sums to +0, as an empty sum does.
            held = held | (reduced(jnp.ones(product.place.shape, ints), jnp.sum, dtype=ints) != 0)
    count = sum(extent for _, _, extent in parts)
    live = True if not count else live
    if segments is not None:
        live = live | ~held
    # The widest digit whose one term's sum `widest (2^w - 1)` the integers hold, its quotient carried apart
    # as it adds (`_nearest`, row 8757).
    widest, most = max(extent for _, _, extent in parts), (1 << (limits.bits - 1)) - 1
    digit = max([0] + [w for w in range(1, limits.bits - 1) if widest * ((1 << w) - 1) <= most])
    if digit < 1 or count > most:  # A level's carry lies in [-count, count - 1] (row 8757).
        raise ValueError(f"an aligned sum of {count} products a value, past its cap of {most}")
    lowest = factors * (limits.minexp - 2 * limits.nmant)
    highest = factors * (limits.maxexp + 1) + count.bit_length() + 2
    levels = -(-(highest - lowest) // digit)
    exact = _nearest(parts, lowest, digit, levels, limits, axes)
    exact = jnp.where(live, exact, -jnp.zeros_like(exact))
    special = jnp.where(positive, jnp.asarray(jnp.inf, dtype), jnp.asarray(-jnp.inf, dtype))
    special = jnp.where(nan | (positive & negative), jnp.asarray(jnp.nan, dtype), special)
    return jnp.where(nan | positive | negative, special, exact).astype(dtype)


def _nearest(parts, lowest: int, digit: int, levels: int, limits, axes: tuple[str, ...] = ()) -> jax.Array:
    """The parts' exact sum as the nearest float, ties to even: each of `levels` digits of `digit` bits from
    bit `lowest` every limb's bits there summed, carried as the number's and its negation's, the last carry
    its sign; the magnitude's highest digits and a sticky bit kept, then written as its bits (`_written`).
    The sticky bit and the bits under the window are integers, zero or one, each read a nonzero test or an
    OR: as bools, a select taking them as values (`jnp.where(nonzero, under, sticky)`, the sign's choice of
    states) took Mosaic TPU an i1 vector of a splat's layout, which its widening to i8 could not relayout,
    every forked TPU scan with an entry write refused (rows 12124, 12125)."""
    ints = jnp.dtype(f"int{limits.bits}")
    # A limb's bits, two of `split_bits`' low halves (`_product`).
    bits, mask, width = limits.bits, (1 << digit) - 1, 2 * split_bits(limits.dtype)[2]
    held = 2 + -(-(limits.nmant + 3) // digit)

    def digit_sums(level):
        low = lowest + level * digit
        sums = []
        for product, reduced, _ in parts:
            taken = 0
            for at, limb in enumerate(product.limbs):
                shift = low - (product.place + at * width)
                right = (limb >> jnp.clip(shift, 0, bits - 1)) & mask
                kept = limb & ((1 << jnp.clip(digit + shift, 0, bits - 1)) - 1)
                left = kept << jnp.clip(-shift, 0, bits - 1)
                taken = taken + jnp.where(shift >= 0, right, jnp.where(digit + shift > 0, left, 0))
            sums.append(reduced(jnp.where(product.negative, -taken, taken), jnp.sum, dtype=ints))
        return sums

    def rolled(state, sums, level):
        # Each term's sum carried as added; the window's last into its sticky part; kept where it leads.
        carry, window, under, lead, kept, sticky = state
        value, carry = carry & mask, carry >> digit
        for total in sums:
            value, carry = value + (total & mask), carry + (total >> digit)  # No partial past one term's.
            carry, value = carry + (value >> digit), value & mask
        under = under | (window[-1] != 0).astype(under.dtype)
        window = (value, *window[:-1])
        nonzero = window[0] != 0
        return (carry, window, under, jnp.where(nonzero, level, lead),
                tuple(jnp.where(nonzero, new, old) for new, old in zip(window, kept, strict=True)),
                jnp.where(nonzero, under, sticky))

    def step(level, states):
        sums = digit_sums(level)
        return rolled(states[0], sums, level), rolled(states[1], [-total for total in sums], level)

    zeros = functools.reduce(jnp.add, [reduced(jnp.zeros_like(product.place), jnp.sum, dtype=ints)
                                       for product, reduced, _ in parts])
    start = (zeros, (zeros,) * held, zeros, zeros - 1, (zeros,) * held, zeros)
    first, last = _levels(parts, lowest, digit, levels, limits, axes)
    # A counter of the value's integers: x64 traced a 64-bit index where Mosaic's is 32 (row 8757).
    _, states = jax.lax.while_loop(lambda carried: carried[0] < last,
                                   lambda carried: (carried[0] + 1, step(carried[0], carried[1])),
                                   (jnp.asarray(first, ints), (start, start)))
    negative = states[0][0] < 0
    _, _, _, lead, kept, sticky = jax.tree.map(lambda p, n: jnp.where(negative, n, p), *states)
    return _written(lead, kept, sticky, negative, lowest, digit, limits)


def _levels(parts, lowest: int, digit: int, levels: int, limits, axes: tuple[str, ...] = ()) -> tuple:
    """The digits `_nearest` walks: a kernel's every one, and every one where a limb may lie across devices
    (`_may_split`: the extrema of independent sums' limbs would be a collective across their shards, row
    8841), else the nonzero limbs' and a carry's room, agreed over the manual `axes` its digits' sums are
    collectives over (`aligned_segment_sum`'s), so every device walks the same digits."""
    from dew.nn.kernels.launches import interpreting

    if interpreting() is not None or _may_split(limb for product, _, _ in parts for limb in product.limbs):
        return 0, levels
    ints = jnp.dtype(f"int{limits.bits}")
    width = 2 * split_bits(limits.dtype)[2]
    least, most = jnp.iinfo(ints).max, jnp.iinfo(ints).min
    for product, _, _ in parts:
        for at, limb in enumerate(product.limbs):
            place = product.place + at * width
            least = jnp.minimum(least, jnp.min(jnp.where(limb != 0, place, least), initial=least))
            most = jnp.maximum(most, jnp.max(jnp.where(limb != 0, place + width, most), initial=most))
    if axes:
        # Varying over every one of `axes` first: a bound every device holds alike agrees as it is.
        least, most = (jax.lax.pcast(bound, missing, to="varying") if (missing := tuple(sorted(
            set(axes) - jax.typeof(bound).mat.varying))) else bound for bound in (least, most))
        least, most = jax.lax.pmin(least, axes), jax.lax.pmax(most, axes)
    count = sum(extent for _, _, extent in parts)
    first = jnp.clip((least - lowest) // digit, 0, levels)
    last = jnp.clip((most + count.bit_length() + 2 - lowest) // digit + 2, 0, levels)
    return first, jnp.maximum(first, last)


def _may_split(limbs) -> bool:
    """Whether a limb may lie across devices: its type's sharding names a mesh axis (an Explicit one), or
    the trace's mesh or its type's has an Auto axis of more than one device, which the compiler may split it
    over and its type does not state."""
    def split(mesh) -> bool:
        return any(kind == jax.sharding.AxisType.Auto and size > 1
                   for kind, size in zip(mesh.axis_types, mesh.axis_sizes, strict=True))

    kinds = [jax.typeof(limb).sharding for limb in limbs]
    return split(jax.sharding.get_abstract_mesh()) or any(
        split(kind.mesh) or any(entry is not None for entry in kind.spec) for kind in kinds)


def _written(lead, kept, sticky, negative, lowest: int, digit: int, limits) -> jax.Array:
    """The magnitude's digits from its highest nonzero one at `lead` (`kept`), `sticky` (an integer, zero or
    one) where any under them is nonzero, as the nearest float's bits, ties to even, signed where `negative`,
    rounded in integers."""
    ints = lead.dtype
    bits = jnp.iinfo(ints).bits
    length = bits - jax.lax.clz(kept[0])
    top = lead * digit + length - 1
    least = limits.minexp - limits.nmant
    # The rounding place, two bits below the significand's lowest, relative to the lowest digit's bit.
    place = jnp.maximum(lowest + top - limits.nmant, least) - 2 - lowest
    whole = jnp.zeros_like(lead)
    for at, value in enumerate(kept):
        offset = (lead - at) * digit - place
        up = value << jnp.clip(offset, 0, bits - 1)
        down = value >> jnp.clip(-offset, 0, bits - 1)
        whole = whole + jnp.where(offset >= 0, up, jnp.where(-offset < digit, down, 0))
        below = value & ((1 << jnp.clip(-offset, 0, digit)) - 1)
        dropped = jnp.where(offset >= 0, 0, jnp.where(-offset < digit, below, value))
        sticky = sticky | (dropped != 0).astype(ints)
    whole = whole | sticky
    tail, significand = whole & 3, whole >> 2
    significand = significand + ((tail > 2) | ((tail == 2) & ((significand & 1) == 1))).astype(ints)
    # The float's bits: its significand (a carry into the next binade its exponent's unit) at its place.
    field = lowest + place + 2 - least
    finite = (jnp.clip(field, 0, 2 * limits.maxexp - 3) << limits.nmant) + significand
    infinite = (2 * limits.maxexp - 1) << limits.nmant
    magnitude = jnp.where(lead < 0, 0, jnp.where(field > 2 * limits.maxexp - 3, infinite, finite))
    signed = jnp.where(negative & (lead >= 0), magnitude | np.array(-(1 << (bits - 1)), ints), magnitude)
    return jax.lax.bitcast_convert_type(signed, jnp.dtype(limits.dtype))


def _aligned_tangent(shape: tuple[tuple[int, ...], ...], spec: tuple, primals, tangents):
    """`_aligned_terms`' value (`_aligned`'s) and tangent in the factors it reaches."""
    from jax.custom_derivatives import SymbolicZero

    value = _aligned(shape, spec, *primals)
    live = tuple(index for index, direction in enumerate(tangents) if not isinstance(direction, SymbolicZero))
    if not live:
        return value, jnp.zeros_like(value)
    return value, _aligned_directions(shape, spec, primals, [tangents[index] for index in live],
                                      jax.typeof(value), live)


def _placed_as(x: jax.Array, like: ShapedArray) -> jax.Array:
    """`x` placed as the type `like`, whole too, where its mesh has an Explicit axis (rows 8757, 8849)."""
    from dew.nn.sharding import explicit_placement

    placed = explicit_placement(like)
    return x if placed is None else jax.sharding.reshard(x, placed)


def _transposed_term(term: Sequence[jax.Array], spot: int, axis, cotangent: jax.Array) -> tuple:
    """`aligned_sum`'s cotangent of one term's factor at `spot` as products and that factor's shape in their
    space: summed over the axes the shape is one along, then broadcast to it. The term's product reduced
    over `axis` (its own axes) and broadcast into the value, transposed in reverse: the value's broadcast
    summed, each of the reduction's elements taking the cotangent, the factor's broadcast summed, as one sum
    of the cotangent times the term's other factors in a space holding the value's axes and, beside each
    reduced one, the product's own (row 8501; mixed ranks reduced against another term's axis, base review
    f5e6e678d F2)."""
    rank = max(np.ndim(factor) for factor in term)
    reduced = set(axis or ())
    lead = cotangent.ndim - rank

    def laid(x):
        # A factor in the space: its reduced axes beside the value's, which it is broadcast along.
        x = jnp.reshape(x, (1,) * (rank - np.ndim(x)) + np.shape(x))
        return x.reshape((1,) * lead + tuple(d for k, size in enumerate(x.shape)
                                             for d in ((1, size) if k in reduced else (size,))))

    spread = cotangent.reshape(cotangent.shape[:lead] + tuple(
        d for k, size in enumerate(cotangent.shape[lead:]) for d in ((size, 1) if k in reduced else (size,))))
    products = (spread, *(laid(factor) for other, factor in enumerate(term) if other != spot))
    return products, laid(term[spot]).shape


def _cotangent_term(term: Sequence[jax.Array], spot: int, axis, cotangent: jax.Array, shape: tuple
                    ) -> tuple[tuple, tuple]:
    """The factor at `spot` of `term`'s cotangent as one aligned sum's term (`_transposed_term`): its products
    and their spec entry, summed along the axes the factor's shape is one along and laid in `shape`."""
    products, held = _transposed_term(term, spot, axis, cotangent)
    along = tuple(d for d, size in enumerate(held) if size == 1)
    return products, (along if along else None, (held, shape))


def aligned_cotangents(terms: Sequence[Sequence[jax.Array]], cotangent: jax.Array,
                       axis: int | tuple[int, ...] | None = None) -> list[jax.Array | np.ndarray]:
    """The cotangent `jax.vjp` of `aligned_sum(terms, axis)` gives each distinct factor at `cotangent`, in
    order of first appearance: a factor several terms share (one array, by identity) takes one, its terms'
    products summed together aligned before they round, as the transpose sums them
    (`_factor_cotangents`). Each is in its factor's type and placement, an integer factor's `float0`. A
    backward that takes no autodiff (a Pallas TPU kernel's) forms an entry write's cotangents so (row 8666).
    """
    originals, factors, shape, spec = _laid_out(terms, ((axis, None),) * len(terms))
    # The cotangent in the sum's type, as reverse mode hands the transpose its output's cotangent.
    given = jnp.asarray(cotangent, jnp.result_type(*factors))
    held = [jax.typeof(jnp.asarray(original)) for original in originals]
    # The factors a cotangent reaches, as the declared rule's `live`: an integer factor takes `float0`.
    live = tuple(place for place, aval in enumerate(held) if jnp.issubdtype(aval.dtype, jnp.inexact))
    pulled = dict(zip(live, _factor_cotangents(shape, spec, factors, given, live), strict=True))
    # Each cast back to its factor's type, as the widening's transpose returns it (`float_bits.cast_back`).
    return [cast_back(pulled[place], aval) if place in pulled else np.zeros(aval.shape, jax.dtypes.float0)
            for place, aval in enumerate(held)]


def _factor_cotangents(shape: tuple[tuple[int, ...], ...], spec: tuple, factors, cotangent: jax.Array,
                       live: tuple[int, ...]) -> list[jax.Array]:
    """Each `live` factor's cotangent of the aligned sum of `shape`'s terms of `factors` at `cotangent`: each
    of the factor's places a term of one aligned sum in its shape (`_transposed_term`), the cotangent inside
    its products, so an overflowing partial sum or one bit-formed is none, in the factor's type and
    placement."""
    gradients = []
    for place in live:
        factor = factors[place]
        terms, laid = [], []
        for index, term in enumerate(shape):
            axis, value = spec[index]
            given = cotangent if value is None else cotangent.reshape(value[0])
            for spot, at in enumerate(term):
                if at == place:
                    others = [factors[other] for other in term]
                    products, held = _cotangent_term(others, spot, axis, given, factor.shape)
                    terms.append(products)
                    laid.append(held)
        gradients.append(_placed_as(_aligned_terms(terms, tuple(laid)).astype(factor.dtype),
                                    jax.typeof(factor)))
    return gradients


def _aligned_directions(shape: tuple[tuple[int, ...], ...], spec: tuple, primals, tangents, value,
                        live: tuple[int, ...]):
    """`aligned_sum`'s tangent in its `live` factors' `tangents` (`value` the sum's type alone, row 8757)."""
    from dew.nn.kernels import generation

    def applied(factors, directions):
        # A place's terms side by side, the first factor's direction in every term, then the second's...
        moving = dict(zip(live, directions, strict=True))
        placed = [(spot, index) for spot in range(max(map(len, shape)))
                  for index, term in enumerate(shape) if spot < len(term) and term[spot] in moving]
        moved = [(*(factors[at] for at in shape[index][:spot]), moving[shape[index][spot]],
                  *(factors[at] for at in shape[index][spot + 1:])) for spot, index in placed]
        # In the value's layout inside the sum, which reverse mode then sums inside it too (row 8757).
        laid = tuple((axis, (value.shape, value.shape) if held is None else held)
                     for axis, held in (spec[index] for _, index in placed))
        return _placed_as(_aligned_terms(moved, laid), value)

    def pulled(factors, cotangent):
        return _factor_cotangents(shape, spec, factors, cotangent, live)

    return generation.linear_map(applied, pulled, list(primals), list(tangents),
                                 out_aval=value.to_tangent_aval(), traced_tangent=True)


_aligned = contracted("test_an_aligned_sum_holds_every_order")(
    declared_tangent(Protected(), _aligned_value, _aligned_tangent, nondiff_argnums=(0, 1),
                     symbolic_zeros=True))
