"""Windowed compensated summation over scaled double-float terms (`dew.algebra.means`, `ratios`).

Each summand is formed as a double float at a power of two (`_product`) and summed a window at a time from
the largest summands down (`_windows`, `_streamed`), so terms spanning more than the format's range cancel
without overflowing or flushing beneath the format's least normal."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

from dew.algebra import scaled
from dew.algebra.scaled import Scaled
from dew.nn.float_bits import exact_power2
from dew.nn.precision import correctly_rounded_divide, power2, two_product, two_sum


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
"""The bound of every exponent a ratio places by (`dew.nn.precision.power2`): a term's (a factor's and its
other factors'), and a sum's beneath its largest."""



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
    return moved.significand, low * exact_power2(-moved.exponent, dtype), exponent + moved.exponent


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
    """`x` summed over `axes` in halves (`_halved`), each pair a compensated sum's step
    (`scaled.compensated_pair`): `(sum, error)`, a cascade compensated as the compensated sum is."""
    error = x - x
    for axis in axes:
        while x.shape[axis] > 1:
            (a, b, rest), (ea, eb, error_rest) = _halves(x, axis), _halves(error, axis)
            pair, made = scaled.compensated_pair(a, ea, b, eb)
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
    magnitude = power2(magnitude, top - moved, reach=_REACH) + _folded(jnp.add, [
        _halved(jnp.add, power2(jnp.abs(high) + jnp.abs(low), exponent - moved, reach=_REACH), axes)
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
        return [jnp.where(each, power2(summand[index], summand[2] - top + placement, reach=_REACH), 0.0)
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
    (`joined`). Another window follows wherever summands lie beneath it within a double float's reach of
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
            made.append(joined(total, (high, rest, top - placement)))
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
        if chunks == 1:
            # One chunk is the operands whole: no slice is bound, which Pallas's Triton lowering has no rule
            # for (row 11661).
            start = 0
            sliced = padded
        else:
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


def joined(a, b):
    """Two double floats at powers of two (`_streamed`'s bands) summed, `(high, low, exponent)`: each moved to
    the larger's power (exact, or beneath the larger's `u^2`), one compensated step of the two
    (`scaled.compensated_pair`: their high parts' sum exact, its error carried with their low parts), then
    renormalized (`two_sum`); a zero sum where both are."""
    a, b = _normal(*a), _normal(*b)
    exponents = [jnp.where(high != 0, exponent, _NONE) for high, _, exponent in (a, b)]
    top = jnp.maximum(*exponents)
    moved = [(power2(high, exponent - top, reach=_REACH), power2(low, exponent - top, reach=_REACH))
             for (high, low, _), exponent in zip((a, b), exponents, strict=True)]
    high, low = two_sum(*scaled.compensated_pair(*moved[0], *moved[1]))
    return high, low, jnp.where(top == _NONE, 0, top)


def _normal(high, low, exponent):
    """A double float `(high, low)` at `exponent` with its high part moved into [1, 2) (exact)."""
    moved = scaled.scaled(high)
    return moved.significand, power2(low, -moved.exponent, reach=_REACH), exponent + moved.exponent


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
    landed = power2(high + low, exponent, reach=_REACH).astype(dtype)
    return jnp.where(landed == 0, jnp.zeros_like(landed), landed)


def _binade(summed, axis: int) -> jax.Array:
    """The power of two a reduced ratio's sum (`_ratio_sum`'s) lands at, as it lands (its high and low parts
    added), laid along `axis` again."""
    high, low, exponent = summed
    return jnp.expand_dims(exponent + scaled.scaled(high + low).exponent, axis)


def _without_axis(operand, axis: int):
    """`operand`'s type without its states' `axis` (its shape and its sharding's spec), its dtype, weak type,
    memory and manual axes (`shard_map`'s varying ones) its own: a merged result's (base review F3: a spec
    kept whole beside a shape without the axis is refused under an explicit mesh)."""
    spec = (*operand.sharding.spec, *(None,) * (operand.ndim - len(operand.sharding.spec)))
    kept = PartitionSpec(*spec[:axis], *spec[axis + 1:])
    return operand.update(shape=tuple(size for place, size in enumerate(operand.shape) if place != axis),
                          sharding=operand.sharding.update(spec=kept))
