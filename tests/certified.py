"""Check 2 as a gate: a site's computation run one primitive at a time (`certified`), each equation
held on the operands its execution gave it to its primitive's contract (CONTRACTS), from the exact
oracles of tests/op_certificates.py; a primitive no contract covers fails the site
(`assert_certified`), and the lane's arithmetic is held as it computes, measured on it (`lane`)."""

import dataclasses
import fractions
import functools
import math

import ml_dtypes
import numpy as np
from op_certificates import (
    EXP_RANGE,
    FORMATS,
    Format,
    exact_sum,
    gamma,
    interval_ends,
    interval_precision,
    magnitude_sum,
    rounded,
    two_sum,
)

STORAGE = {"float32": np.uint32, "bfloat16": np.uint16}
"""Each format's storage bits, which a move of data is compared by."""

VALUES = {"float32": np.float32, "bfloat16": ml_dtypes.bfloat16}
"""Each format's NumPy type."""

UNREACHED = np.iinfo(np.int64).max
"""The level (`LEVELS`) of a result no reach admits: a NaN where the exact value is a number or the
reverse, a wrong exact case (an infinite or named result), or a value off the lane's grid (a subnormal
from a lane that flushes)."""


@dataclasses.dataclass(frozen=True)
class Lane:
    """The float arithmetic of the lane check 2 runs on, as measured there (`lane`).

    `flushes`: whether an arithmetic operation or comparison reads a subnormal operand as a zero of its
    sign and a result tiny after rounding (below the least normal once rounded to its precision with an
    unbounded exponent) is a zero of the exact result's sign, as XLA on the CPU does; XLA on a GPU keeps
    subnormals unless `--xla_gpu_ftz` is set. A move of data, a sign operation (neg, abs) and a cast
    between float formats are bit manipulation: they keep subnormals on every lane.

    `reach`: by rounding primitive and format, the largest level (`LEVELS`) of the lane's own results
    over its probes (`_probes`): 0 where every one was its exact value rounded to nearest, 1 where each
    was one of the two neighbours of its exact value on the lane's grid (faithful), n where each was
    within n - 1 steps of the grid past them."""
    flushes: bool
    reach: dict[tuple[str, str], int]


def _grid_index(value, fmt: Format, flushes: bool) -> np.ndarray:
    """Each value of `fmt` (as float64, finite or infinite) as its place on the lane's grid, in steps
    from zero, negative below it: its storage bits' magnitude, less the subnormals the grid skips where
    the lane flushes, so that the least normal is one step from zero there."""
    value = np.asarray(value, np.float64)
    magnitude = np.abs(value).astype(VALUES[fmt.name]).view(STORAGE[fmt.name]).astype(np.int64)
    if flushes:
        magnitude = np.where(magnitude > 0, magnitude - (2 ** (fmt.precision - 1) - 1), 0)
    return np.where(np.signbit(value), -magnitude, magnitude)


def _format(dtype) -> Format | None:
    """`dtype`'s float format of ours, or None (an integer, a boolean, a typed key)."""
    return FORMATS.get(getattr(dtype, "name", str(dtype)))


def _inexact(dtype) -> bool:
    """A float, complex or narrow float type, whether or not a format of ours (ml_dtypes' float8 is a
    void kind to NumPy)."""
    import jax.numpy as jnp

    return bool(jnp.issubdtype(dtype, jnp.inexact))


def _host(value):
    """A value on the host: an array as NumPy's, a typed key as it is (its bits are its data)."""
    import jax

    if jax.dtypes.issubdtype(getattr(value, "dtype", np.float32), jax.dtypes.prng_key):
        return value
    return np.asarray(value)


def _bits(value) -> np.ndarray:
    """A value's storage bits, element by element: what a move of data keeps."""
    value = np.asarray(value)
    if value.dtype == bool:
        return value.astype(np.uint8)
    return value.view(np.dtype(f"u{value.dtype.itemsize}"))


def _daz(value, fmt: Format, flushes: bool) -> np.ndarray:
    """`value` (float64) with every subnormal of `fmt` read as a zero of its sign where the lane
    `flushes`, as its arithmetic reads it."""
    value = np.asarray(value, np.float64)
    if not flushes:
        return value
    return np.where(np.abs(value) < 2.0 ** fmt.emin, np.copysign(0.0, value), value)


def _read(value, flushes: bool) -> np.ndarray:
    """An operand as the lane's arithmetic reads it: a float of a known format as float64, flushed
    where the lane flushes (`_daz`); any other as it is."""
    fmt = _format(np.asarray(value).dtype)
    return np.asarray(value) if fmt is None else _daz(np.asarray(value, np.float64), fmt, flushes)


def _same(native, expected) -> np.ndarray:
    """Equal values, element by element, zeros by sign and NaN by NaN: an arithmetic result's identity
    (a NaN's payload is no part of IEEE's contract)."""
    native, expected = np.asarray(native, np.float64), np.asarray(expected, np.float64)
    return ((native == expected) & (np.signbit(native) == np.signbit(expected))) | (
        np.isnan(native) & np.isnan(expected))


_UNBOUNDED = {name: Format(f"{name} unbounded", fmt.precision, -1022, fmt.emax)
              for name, fmt in FORMATS.items()}
"""Each format's precision at every exponent: what the lane's tininess is read from."""


def _rounded_on_lane(hi, lo, fmt: Format, flushes: bool) -> np.ndarray:
    """RN of the exact hi + lo in `fmt` as the lane's arithmetic rounds it: gradually, or where it
    `flushes` a zero of the exact value's sign wherever the value rounded with an unbounded exponent is
    below the least normal (tininess after rounding)."""
    gradual = rounded(hi, lo, fmt)
    if not flushes:
        return gradual
    unbounded = rounded(hi, lo, _UNBOUNDED[fmt.name])
    tiny = np.isfinite(unbounded) & (np.abs(unbounded) < 2.0 ** fmt.emin)
    return np.where(tiny, np.copysign(0.0, np.where(hi != 0, hi, gradual)), gradual)


def _step(value, fmt: Format, direction: float, flushes: bool) -> np.ndarray:
    """The next value of `fmt` after each `value` (a value of `fmt`, as float64) toward `direction`,
    on the lane's grid: where it `flushes` its outputs hold no subnormal, so the next value past zero is
    the least normal and the next below the least normal is zero."""
    dtype = VALUES[fmt.name]
    value = np.asarray(value, np.float64)
    stepped = np.nextafter(value.astype(dtype), np.full(value.shape, direction, dtype)).astype(np.float64)
    if flushes:
        least = 2.0 ** fmt.emin
        tiny = np.abs(stepped) < least
        # Past zero toward a subnormal the grid's next value is the least normal; from the least
        # normal toward zero it is the zero of its sign.
        jumped = np.where(np.abs(value) >= least, np.copysign(0.0, value), np.copysign(least, direction))
        stepped = np.where(tiny, jumped, stepped)
    return stepped


def _past(native, down, up, fmt: Format, flushes: bool) -> np.ndarray:
    """Each native value's level against the two neighbours `down` and `up` of its exact value on the
    lane's grid (equal where the exact value is on it): 1 at either, one more for each step of the grid
    past them; UNREACHED for a NaN, a value off the grid, or a zero of the sign the exact value's side
    does not take (the side of `up`, a negative value or a negative zero, or else positive)."""
    native, up = np.asarray(native, np.float64), np.asarray(up, np.float64)
    index = _grid_index(native, fmt, flushes)
    past = np.maximum(_grid_index(down, fmt, flushes) - index, index - _grid_index(up, fmt, flushes))
    negative = (up < 0) | ((up == 0) & np.signbit(up))
    off = (np.isnan(native) | (flushes & (native != 0) & (np.abs(native) < 2.0 ** fmt.emin))
           | ((native == 0) & (np.signbit(native) != negative)))
    return np.where(off, UNREACHED, 1 + np.maximum(past, 0))


def _neighbours(low, high, fmt: Format, flushes: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The greatest value of the lane's grid at or below an exact value enclosed in [low, high]
    (float64, outward), the least at or above it, and whether the enclosure decides them: one that
    straddles a value of the grid, its ends not two neighbours, decides nothing, and is refined by the
    caller."""
    down = _rounded_on_lane(low, np.zeros_like(low), fmt, flushes)
    down = np.where(down > low, _step(down, fmt, -np.inf, flushes), down)
    up = _rounded_on_lane(high, np.zeros_like(high), fmt, flushes)
    up = np.where(up < high, _step(up, fmt, np.inf, flushes), up)
    exact = (low == high) & (down == up)
    return down, up, exact | (_step(down, fmt, np.inf, flushes) == up)


def _outward(end: fractions.Fraction, direction: float) -> float:
    """The float64 nearest an exact end, moved one step `direction` where rounding took it inward, and
    infinite past float64's range."""
    try:
        value = float(end)
    except OverflowError:
        return -math.inf if end < 0 else math.inf
    inward = fractions.Fraction(value) > end if direction < 0 else fractions.Fraction(value) < end
    return math.nextafter(value, direction) if inward else value


def _interval(function):
    """An outward float64 enclosure, at `bits`, of an mpmath interval `function` at each finite value."""
    def enclose(values, bits):
        low, high = np.empty(values.shape), np.empty(values.shape)
        with interval_precision(bits) as iv:
            for index, value in enumerate(values.flat):
                a, b = interval_ends(function(iv, iv.mpf(float(value))), bits)
                low.flat[index], high.flat[index] = _outward(a, -math.inf), _outward(b, math.inf)
        return low, high
    return enclose


def _exp_interval(values, bits):
    """exp's outward enclosure at each finite value, taken at EXP_RANGE's nearer end beyond it: past
    either end every exp lies between the same two neighbours of a format no wider than fp32, so the
    end's enclosure gives the same verdict, and no end is built of an exponent float64 cannot hold
    (exp(1e30) is 2^(1.4e30))."""
    return _interval(lambda iv, x: iv.exp(x))(np.clip(values, *EXP_RANGE), bits)


def _erfinv_enclosed(values, bits):
    """An outward enclosure of erfinv at each value in (-1, 1): mpmath's erfinv at `bits`, widened by
    2^-(bits - 24) of itself, verified at its two ends by erfc at `bits` + 64 bits, 2^40 times finer
    than the width (erfinv is odd and erfc decreasing: erfc(b) < 1 - |x| < erfc(a) for |x|'s ends)."""
    import mpmath

    low, high = np.empty(values.shape), np.empty(values.shape)
    for index, value in enumerate(values.flat):
        with mpmath.workprec(bits):
            estimate = mpmath.erfinv(mpmath.mpf(float(value)))
            width = abs(estimate) * mpmath.mpf(2) ** -(bits - 24) + mpmath.mpf(2) ** -(bits + 40)
            ends = float(estimate - width), float(estimate + width)
        a, b = math.nextafter(ends[0], -math.inf), math.nextafter(ends[1], math.inf)
        with mpmath.workprec(bits + 64):
            near, far = (a, b) if value >= 0 else (-b, -a)
            rest = 1 - abs(mpmath.mpf(float(value)))
            if not (mpmath.erfc(mpmath.mpf(far)) < rest < mpmath.erfc(mpmath.mpf(near))):
                raise ArithmeticError(f"erfinv({value!r}) not enclosed at {bits} bits")
        low.flat[index], high.flat[index] = a, b
    return low, high


def _log_special(value):
    """log's limits at 0 and infinity, its NaN below zero, and its one exact value, log(1) = +0, named."""
    named = np.isnan(value) | (value <= 0) | (value == np.inf) | (value == 1)
    expected = np.select([value < 0, value == 0, value == np.inf, value == 1], [np.nan, -np.inf, np.inf, 0.0],
                         np.nan)
    return expected, named


def _exp_special(value):
    """exp's limits at the infinities and its one exact value, exp(±0) = 1, named."""
    named = ~np.isfinite(value) | (value == 0)
    return np.select([value == np.inf, value == -np.inf, value == 0], [np.inf, 0.0, 1.0], np.nan), named


def _erfinv_special(value):
    """erfinv's poles at ±1 and its zero, exactly ±0 (it is odd), named; NaN outside (-1, 1)."""
    expected = np.where(np.abs(value) == 1, np.copysign(np.inf, value), np.where(value == 0, value, np.nan))
    return expected, ~(np.abs(value) < 1) | (value == 0)


def _transcendental(enclose, special):
    """A transcendental's levels (`LEVELS`): where `special(value)` names an exact result (a limit, a
    pole, outside the domain), 0 at that result and UNREACHED elsewhere; at every other value at least
    1, its place against the two neighbours of `enclose(values, bits)` (`_past`), the enclosure refined
    at rising precision wherever it decides nothing."""
    def levels(operands, result, fmt, flushes):
        (value,) = _floats(operands, flushes)
        result = np.asarray(result, np.float64)
        expected, named = special(value)
        found = np.where(_same(result, expected), 0, UNREACHED)
        rest = np.flatnonzero(~named)
        for bits in (96, 192, 384, 768):
            if not rest.size:
                break
            low, high = enclose(value.flat[rest], bits)
            down, up, decided = _neighbours(low, high, fmt, flushes)
            found.flat[rest[decided]] = _past(result.flat[rest], down, up, fmt, flushes)[decided]
            rest = rest[~decided]
        if rest.size:
            raise ArithmeticError(f"undecided at 768 bits at {rest[:4]}")
        return found
    return levels


def _terms_hold(native, terms, fmt: Format, flushes: bool) -> np.ndarray:
    """Whether each native sum of `terms` (exact float64 values, summed along the last axis) is within
    the bound of any order of summing them in `fmt`, fused multiply-adds included: γ_n Σ|terms| (Higham,
    Accuracy and Stability of Numerical Algorithms, 2002, §3.1: n terms, at most n - 1 roundings on any
    path), around the terms' float64-rounded exact sum (`exact_sum`, within 2^-53 of it relative to
    Σ|terms|, in the allowance), plus what underflow loses past that relative bound at each rounding:
    where the lane flushes, a flushed partial's least normal; where it keeps subnormals, half their
    spacing (a product, or a sum carried wider and rounded into `fmt`). An empty sum is the additive
    identity, +0."""
    native = np.asarray(native, np.float64)
    count = terms.shape[-1]
    if count == 0:
        return _same(native, np.zeros_like(native))
    magnitude = magnitude_sum(terms, -1)
    underflow = 2.0 ** fmt.emin if flushes else 2.0 ** (fmt.emin - fmt.precision)
    allowance = (gamma(count, fmt) + 2.0 ** -52) * magnitude + count * underflow
    error = np.abs(native - exact_sum(terms, -1))
    return np.isfinite(error) & (error <= np.nextafter(allowance, np.inf))


def _floats(operands, flushes: bool):
    return [_read(value, flushes) for value in operands]


def _wrapping(operator):
    """An integer operation of the result's width, wrapping as two's complement does: computed modulo
    2^64 and narrowed, which is exact modulo the narrower width too."""
    def integral(result, *operands):
        wide = [np.asarray(value).astype(np.uint64) for value in operands]
        with np.errstate(over="ignore"):
            return np.asarray(operator(*wide)).astype(np.asarray(result).dtype)
    return integral


def _arithmetic(build):
    """An IEEE operation's levels (`LEVELS`), its exact result `build(*operands)` as a double-double: 0
    where the result is that rounded to nearest on the lane (`_rounded_on_lane`), else its place against
    the exact result's two neighbours (`_past`); an exactly infinite or undefined result (an infinite
    operand, a division by zero) 0 where the lane's is the same and UNREACHED elsewhere."""
    def levels(operands, result, fmt, flushes):
        hi, lo = build(*_floats(operands, flushes))
        result = np.asarray(result, np.float64)
        nearest = _rounded_on_lane(hi, lo, fmt, flushes)
        # Rounding took the exact value up, or down: the other neighbour is one step of the grid away.
        above = (nearest > hi) | ((nearest == hi) & (lo < 0))
        below = (nearest < hi) | ((nearest == hi) & (lo > 0))
        down = np.where(above, _step(nearest, fmt, -np.inf, flushes), nearest)
        up = np.where(below, _step(nearest, fmt, np.inf, flushes), nearest)
        placed = np.where(np.isfinite(hi), _past(result, down, up, fmt, flushes), UNREACHED)
        return np.where(_same(result, nearest), 0, placed)
    return levels


def _rounding(name: str, integral):
    """A rounding primitive's contract: each float result's level (`LEVELS[name]`) within the lane's
    reach for it (`lane`); an integer result bit for bit (`integral`), where the primitive has one."""
    def contract(params, operands, result, fmt):
        if fmt is None:
            return _bits(result) == _bits(integral(result, *operands))
        measured = lane()
        return LEVELS[name](operands, result, fmt, measured.flushes) <= measured.reach[name, fmt.name]
    return contract


def _product(a, b):
    product = a * b
    return product, np.zeros_like(product)


def _quotient(a, b):
    quotient = a / b
    return quotient, np.zeros_like(quotient)


def _root(a):
    root = np.sqrt(a)
    # float64's correctly rounded root of a p-bit value is never within 2^-(2p+2) of a p-bit midpoint.
    return root, np.zeros_like(root)


def _exact_integer(values) -> tuple[np.ndarray, np.ndarray]:
    """An integer array as a double-double of its exact values: hi the nearest float64, lo the rest."""
    values = np.asarray(values)
    hi = values.astype(np.float64)
    rests = [float(int(value) - int(near)) for value, near in zip(values.flat, hi.flat, strict=True)]
    lo = np.array(rests, np.float64).reshape(hi.shape)
    return hi, lo


def _converted(params, operands, result, fmt):
    """A cast, as XLA's CPU lowering casts: between float formats correctly rounded, subnormals kept
    (bit manipulation); to a float from an integer or boolean correctly rounded from the exact integer;
    to an integer from a float truncated toward zero, saturated at the integer's range and NaN to 0; to
    a boolean nonzero; between integers wrapped to the result's width."""
    (value,) = operands
    value, result = np.asarray(value), np.asarray(result)
    source = _format(value.dtype)
    if fmt is not None:
        exact = _exact_integer(value) if source is None else (value.astype(np.float64), np.zeros(value.shape))
        return _same(result, rounded(*exact, fmt))
    if result.dtype == bool:
        return result == (_read(value, lane().flushes) != 0)
    if source is not None:
        limits = np.iinfo(result.dtype)
        # Each element's exact truncation, clamped in exact integers (64-bit limits are not float64s).
        read = _read(value, lane().flushes)
        expected = [0 if math.isnan(v) else limits.max if v == math.inf else limits.min if v == -math.inf else
                    min(max(int(v), int(limits.min)), int(limits.max)) for v in read.flat]
        return result == np.array(expected, dtype=object).reshape(result.shape)
    return result == value.astype(result.dtype)


def _reduced_sum(params, operands, result, fmt):
    (terms,) = operands
    axes = tuple(params["axes"])
    if fmt is None:
        summed = _wrapping(lambda t: np.sum(t, axis=axes, dtype=np.uint64))
        return _bits(result) == _bits(summed(result, terms))
    flushes = lane().flushes
    moved = np.moveaxis(_read(terms, flushes), axes, tuple(range(-len(axes), 0)))
    kept = moved.shape[:moved.ndim - len(axes)]
    count = int(np.prod(moved.shape[len(kept):], dtype=np.int64))
    return _terms_hold(result, moved.reshape(*kept, count), fmt, flushes)


def _cumulative(params, operands, result, fmt):
    """A cumulative sum along `axis`, forward or reversed: each prefix's (or suffix's) sum of its terms
    within its bound in any order (`_terms_hold`, every prefix taken as one of the axis's length, which
    bounds each shorter one); for integers, each wrapped to the result's width."""
    (terms,) = operands
    axis, reverse = params["axis"], params["reverse"]
    if fmt is None:
        summed = _wrapping(lambda t: np.flip(np.cumsum(np.flip(t, axis), axis=axis, dtype=np.uint64), axis)
                           if reverse else np.cumsum(t, axis=axis, dtype=np.uint64))
        return _bits(result) == _bits(summed(result, terms))
    flushes = lane().flushes
    moved = np.moveaxis(_read(terms, flushes), axis, -1)
    count = moved.shape[-1]
    taken = np.arange(count)[None, :] >= np.arange(count)[:, None] if reverse else \
        np.arange(count)[None, :] <= np.arange(count)[:, None]
    # [..., k, j]: the j-th term where the k-th prefix (or suffix) takes it, zero elsewhere.
    prefixes = np.where(taken, moved[..., None, :], 0.0)
    held = _terms_hold(np.moveaxis(np.asarray(result), axis, -1), prefixes, fmt, flushes)
    return np.moveaxis(held, -1, axis)


def _dot(params, operands, result, fmt):
    """A dot_general's products, exact in float64 for operands no wider than fp32, summed (`_terms_hold`)
    over the contracting dimensions in any order; for integers, wrapped to the result's width."""
    (contract_l, contract_r), (batch_l, batch_r) = params["dimension_numbers"]
    left, right = (np.asarray(value) for value in operands)
    free_l = [axis for axis in range(left.ndim) if axis not in contract_l and axis not in batch_l]
    free_r = [axis for axis in range(right.ndim) if axis not in contract_r and axis not in batch_r]
    left = np.transpose(left, [*batch_l, *free_l, *contract_l])
    right = np.transpose(right, [*batch_r, *free_r, *contract_r])
    batch = left.shape[:len(batch_l)]

    def size(shape):
        return int(np.prod(shape, dtype=np.int64))

    head = len(batch)
    rows = size(left.shape[head:head + len(free_l)])
    columns = size(right.shape[head:head + len(free_r)])
    width = size(left.shape[head + len(free_l):])

    def paired(lhs, rhs):
        return lhs.reshape(*batch, rows, 1, width) * rhs.reshape(*batch, 1, columns, width)

    if fmt is None:
        summed = _wrapping(lambda lhs, rhs: paired(lhs, rhs).sum(-1))
        return _bits(result) == _bits(summed(result, left, right).reshape(result.shape))
    flushes = lane().flushes
    products = paired(_read(left, flushes), _read(right, flushes))
    native = np.asarray(result, np.float64).reshape(products.shape[:-1])
    return _terms_hold(native, products, fmt, flushes).reshape(result.shape)


def _moved(oracle):
    """A move of data or a sign operation: the result is `oracle(params, *operands)` in storage bits,
    subnormals and NaN payloads kept."""
    def contract(params, operands, result, fmt):
        expected = np.asarray(oracle(params, *(np.asarray(value) for value in operands)))
        result = np.asarray(result)
        return np.broadcast_to(_bits(result) == _bits(expected.astype(result.dtype)), result.shape)
    return contract


def _compared(oracle):
    """A comparison or logical operation on values as the lane's arithmetic reads them (`_read`)."""
    def contract(params, operands, result, fmt):
        expected = np.asarray(oracle(params, *(_read(value, lane().flushes) for value in operands)))
        return np.asarray(result) == expected.astype(np.asarray(result).dtype)
    return contract


def _signed_max(a, b):
    """The lane's max: NaN propagates and -0 is below +0."""
    both = (a == 0) & (b == 0)
    return np.where(both, np.where(np.signbit(a) & np.signbit(b), -0.0, 0.0), np.maximum(a, b))


def _signed_min(a, b):
    """The lane's min: NaN propagates and -0 is below +0."""
    both = (a == 0) & (b == 0)
    return np.where(both, np.where(np.signbit(a) | np.signbit(b), -0.0, 0.0), np.minimum(a, b))


def _extreme(choose, integral):
    """max or min of the operands as the lane's arithmetic reads them (`_read`), by its own order of
    zeros and NaN (`choose`), as values; of integers or booleans, `integral`."""
    def contract(params, operands, result, fmt):
        if fmt is None:
            return np.asarray(result) == integral(*(np.asarray(value) for value in operands))
        return _same(result, choose(*_floats(operands, lane().flushes)))
    return contract


def _extremum(choose, integral):
    """reduce_max or reduce_min over the reduced axes, by `choose` folded pairwise, as values; of
    integers or booleans, `integral` (its identity the type's own)."""
    def contract(params, operands, result, fmt):
        axes = tuple(params["axes"])
        (value,) = operands
        if fmt is None:
            return np.asarray(result) == integral(np.asarray(value), axes)
        moved = np.moveaxis(_read(value, lane().flushes), axes, tuple(range(-len(axes), 0)))
        flat = moved.reshape(*moved.shape[:moved.ndim - len(axes)], -1)
        if flat.shape[-1] == 0:
            identity = -np.inf if choose is _signed_max else np.inf
            return _same(result, np.full(flat.shape[:-1], identity))
        folded = flat[..., 0]
        for index in range(1, flat.shape[-1]):
            folded = choose(folded, flat[..., index])
        return _same(result, folded)
    return contract


def _integral_extremum(reduce, extreme: str):
    """An integer or boolean extremum reduction from its type's identity: the least value for a max,
    the greatest for a min (False and True for booleans)."""
    def reduced(value, axes):
        if np.dtype(value.dtype) == np.dtype(bool):
            start = extreme == "min"
        else:
            limits = np.iinfo(value.dtype)
            start = limits.max if extreme == "min" else limits.min
        return reduce(value, axis=axes, initial=start)
    return reduced


def _selected(params, which, *cases):
    return np.choose(np.asarray(which).astype(np.int64), cases)


def _broadcast(params, value):
    shape, dimensions = params["shape"], params["broadcast_dimensions"]
    expanded = np.reshape(value, [value.shape[dimensions.index(axis)] if axis in dimensions else 1
                                  for axis in range(len(shape))])
    return np.broadcast_to(expanded, shape)


def _reshaped(params, value):
    if params.get("dimensions") is not None:
        value = np.transpose(value, params["dimensions"])
    return np.reshape(value, params["new_sizes"])


def _padded(params, value, fill):
    """lax.pad's result: each axis dilated by its interior padding, then given its low and high padding,
    a negative one cropping."""
    for axis, (low, high, interior) in enumerate(params["padding_config"]):
        low, high, interior = int(low), int(high), int(interior)
        if interior and value.shape[axis] > 1:
            size = value.shape[axis] + (value.shape[axis] - 1) * interior
            dilated = np.full((*value.shape[:axis], size, *value.shape[axis + 1:]), fill, value.dtype)
            index = [slice(None)] * value.ndim
            index[axis] = slice(None, None, interior + 1)
            dilated[tuple(index)] = value
            value = dilated
        widths = [(0, 0)] * value.ndim
        widths[axis] = (max(low, 0), max(high, 0))
        value = np.pad(value, widths, constant_values=fill)
        index = [slice(None)] * value.ndim
        index[axis] = slice(max(-low, 0), value.shape[axis] - max(-high, 0))
        value = value[tuple(index)]
    return value


def _iota(params):
    shape, axis = tuple(params["shape"]), params["dimension"]
    counted = np.arange(shape[axis], dtype=np.dtype(params["dtype"]))
    return np.broadcast_to(counted.reshape([shape[axis] if dim == axis else 1 for dim in range(len(shape))]),
                           shape)


def _barrier(params, operands, result, fmt):
    """An optimization barrier's results are its operands, unchanged: each result one of them, bit for bit
    (`_held` hands a contract each result alone)."""
    result = np.asarray(result)
    same = [np.asarray(value) for value in operands
            if np.shape(value) == result.shape and np.asarray(value).dtype == result.dtype]
    held = any(np.array_equal(_bits(value), _bits(result)) for value in same)
    return np.full(result.shape, held)


def _gathered(params, operand, indices):
    """lax.gather's result, by XLA's semantics: each output index's batch coordinates pick a start vector
    from `indices` (its last axis the index vector), placed on the operand's `start_index_map` axes and
    its batching axes, clamped so the slice fits (the clip and promise-in-bounds modes) or the whole slice
    the fill value where it does not (fill-or-drop), then offset by the output's offset coordinates on the
    operand's axes a slice keeps."""
    import jax

    numbers = params["dimension_numbers"]
    sizes, mode = tuple(params["slice_sizes"]), params["mode"]
    fill = jax.lax.GatherScatterMode.FILL_OR_DROP
    shape = operand.shape
    out_rank = len(numbers.offset_dims) + indices.ndim - 1
    offset_axes = [axis for axis in range(operand.ndim)
                   if axis not in numbers.collapsed_slice_dims and axis not in numbers.operand_batching_dims]
    batch_dims = [dim for dim in range(out_rank) if dim not in numbers.offset_dims]
    kept = [size for axis, size in enumerate(sizes) if axis in offset_axes]
    out_shape = [0] * out_rank
    for dim, size in zip(numbers.offset_dims, kept, strict=True):
        out_shape[dim] = size
    for dim, size in zip(batch_dims, indices.shape[:-1], strict=True):
        out_shape[dim] = size
    result = np.empty(out_shape, operand.dtype)
    if params.get("fill_value") is not None:
        filler = params["fill_value"]
    elif np.issubdtype(operand.dtype, np.inexact):
        filler = np.nan
    elif np.issubdtype(operand.dtype, np.bool_):
        filler = True
    else:
        filler = np.iinfo(operand.dtype).min if np.issubdtype(operand.dtype, np.signedinteger) else \
            np.iinfo(operand.dtype).max
    for out in np.ndindex(*out_shape):
        batch = tuple(out[dim] for dim in batch_dims)
        vector = indices[batch]
        start = [0] * operand.ndim
        for position, axis in enumerate(numbers.start_index_map):
            start[axis] = int(vector[position])
        for axis, batch_axis in zip(numbers.operand_batching_dims, numbers.start_indices_batching_dims,
                                    strict=True):
            start[axis] = batch[batch_axis]
        if mode == fill and any(start[axis] < 0 or start[axis] + sizes[axis] > shape[axis]
                                for axis in range(operand.ndim)):
            result[out] = filler
            continue
        start = [min(max(begin, 0), shape[axis] - sizes[axis]) for axis, begin in enumerate(start)]
        offsets = [out[dim] for dim in numbers.offset_dims]
        for axis, offset in zip(offset_axes, offsets, strict=True):
            start[axis] += offset
        result[out] = operand[tuple(start)]
    return result


def _bitcast(params, value):
    """lax.bitcast_convert_type's shapes: a narrower type adds a trailing axis of the width ratio, a
    wider one takes the trailing axis away."""
    new = np.dtype(params["new_dtype"])
    old = value.dtype
    if new.itemsize == old.itemsize:
        return value.view(new)
    if new.itemsize < old.itemsize:
        return np.ascontiguousarray(value).view(new).reshape(*value.shape, old.itemsize // new.itemsize)
    return np.ascontiguousarray(value).view(new).reshape(value.shape[:-1])


def _logical_shift(direction):
    """A logical shift: on the operand's unsigned bits, zero at a shift of the width or more."""
    def shifted(params, a, b):
        unsigned = np.asarray(a).view(np.dtype(f"u{np.asarray(a).dtype.itemsize}"))
        width = 8 * unsigned.dtype.itemsize
        count = np.asarray(b).astype(np.int64)
        operator = np.right_shift if direction > 0 else np.left_shift
        moved = operator(unsigned, np.clip(count, 0, width - 1).astype(unsigned.dtype))
        kept = np.where((count >= width) | (count < 0), 0, moved).astype(unsigned.dtype)
        return kept.view(np.asarray(a).dtype)
    return shifted


def _random_bits(params, operands, result, fmt):
    """Random bits are integers, whatever program computes them: the bits a separate `jax.random.bits`
    of the same key draws under the equation's own context, bit for bit."""
    import jax

    (key,) = operands
    dtype = np.dtype(f"uint{params['bit_width']}")
    expected = jax.jit(lambda key: jax.random.bits(key, params["shape"], dtype))(key)
    return np.asarray(result) == np.asarray(expected)


def _wrapped(params, operands, result, fmt):
    """A key wrapped from its data: the key's data is the operand, bit for bit."""
    import jax

    return _bits(np.asarray(jax.random.key_data(result))) == _bits(operands[0])


def _unwrapped(params, operands, result, fmt):
    import jax

    return _bits(result) == _bits(np.asarray(jax.random.key_data(operands[0])))


LEVELS = {
    "add": _arithmetic(two_sum),
    "sub": _arithmetic(lambda a, b: two_sum(a, -b)),
    "mul": _arithmetic(_product),
    "square": _arithmetic(lambda a: _product(a, a)),
    "div": _arithmetic(_quotient),
    "sqrt": _arithmetic(_root),
    "log": _transcendental(_interval(lambda iv, x: iv.log(x)), _log_special),
    "exp": _transcendental(_exp_interval, _exp_special),
    "erf_inv": _transcendental(_erfinv_enclosed, _erfinv_special),
}
"""Each rounding primitive's measure, `levels(operands, result, fmt, flushes) -> level`, element by
element, on a lane that flushes or not: 0 where the result is its exact value rounded to nearest (an
IEEE operation's), 1 at either neighbour of its exact value on the lane's grid (faithful), n at n - 1
steps of the grid past them; UNREACHED where no reach admits it. The lane's reach for each is measured
(`lane`), and its contract is its reach (CONTRACTS)."""

PROBES = 2048
"""The operands drawn for each rounding primitive in each format that measure the lane's reach for it
(`lane`, `_probes`): the reach is the largest level over them, so a rarer rounding a site meets past
it is a finding there, and the probes grow; few enough that the transcendentals' exact enclosures,
one interval evaluation each, take seconds once a process."""


def _probes(name: str, fmt: Format) -> list[np.ndarray]:
    """PROBES operands of the rounding primitive `name` in `fmt`, over its domain at every scale its
    results reach: sums of every alignment, a quarter of them nearly cancelling; products and quotients
    from about 2^-140 (subnormal on a lane that keeps subnormals) to 2^126; roots of every binade,
    subnormals included; logarithms of every binade and a quarter near one; exponentials across fp32's
    range and a quarter near zero; inverse error functions across (-1, 1) and a quarter near its poles;
    and each transcendental at its exact value (exp(0), log(1), erf_inv(0)), which a lane must return
    exactly. One fixed draw per primitive and format."""
    rng = np.random.default_rng([*name.encode(), fmt.precision])
    quarter = PROBES // 4

    def scaled(low: int, high: int) -> np.ndarray:
        signs = rng.choice([-1.0, 1.0], PROBES)
        return signs * rng.uniform(1, 2, PROBES) * 2.0 ** rng.integers(low, high, PROBES)

    if name in ("add", "sub"):
        a, b = scaled(-30, 30), scaled(-30, 30)
        near = a[:quarter] * (1 + rng.uniform(-2.0 ** -10, 2.0 ** -10, quarter))
        b[:quarter] = -near if name == "add" else near
        drawn = [a, b]
    elif name in ("mul", "square"):
        drawn = [scaled(-70, 63) for _ in range(1 if name == "square" else 2)]
    elif name == "div":
        drawn = [scaled(-70, 63), scaled(-63, 70)]
    elif name == "sqrt":
        drawn = [np.abs(scaled(-149, 127))]
    elif name == "log":
        value = np.abs(scaled(-126, 127))
        value[:quarter] = 1 + rng.uniform(-2.0 ** -8, 2.0 ** -8, quarter)
        value[0] = 1.0
        drawn = [value]
    elif name == "exp":
        value = rng.uniform(-104, 89, PROBES)
        value[:quarter] = rng.uniform(-2.0 ** -8, 2.0 ** -8, quarter)
        value[0] = 0.0
        drawn = [value]
    else:
        value = rng.uniform(-1, 1, PROBES)
        poles = 1 - 2.0 ** -rng.uniform(1, fmt.precision, quarter)
        value[:quarter] = rng.choice([-1.0, 1.0], quarter) * poles
        value[0] = 0.0
        drawn = [value]
    return [value.astype(VALUES[fmt.name]) for value in drawn]


@functools.cache
def lane() -> Lane:
    """The lane check 2 runs on (the process's default backend), measured once: whether it reads a
    subnormal operand as zero (2^-140 times 2^100), and each rounding primitive's reach in each format,
    the largest level (`LEVELS`) of the lane's own results on its probes (`_probes`), each operation
    run as `certified` runs it, eagerly and alone. A lane result that breaks an exact case (UNREACHED)
    is a finding no reach admits, raised here."""
    import jax

    flushes = float(jax.lax.mul(np.float32(2.0 ** -140), np.float32(2.0 ** 100))) == 0.0
    reach = {}
    for name, levels in LEVELS.items():
        for fmt in FORMATS.values():
            operands = _probes(name, fmt)
            # Each primitive is `jax.lax`'s function of its own name.
            found = levels(operands, np.asarray(getattr(jax.lax, name)(*operands)), fmt, flushes)
            if np.any(found == UNREACHED):
                at = int(np.argmax(found == UNREACHED))
                raise ArithmeticError(f"the lane's {name} in {fmt.name} breaks an exact case at "
                                      f"{[operand[at] for operand in operands]}")
            reach[name, fmt.name] = int(np.max(found))
    return Lane(flushes, reach)


CONTRACTS = {
    "add": _rounding("add", _wrapping(np.add)),
    "sub": _rounding("sub", _wrapping(np.subtract)),
    "mul": _rounding("mul", _wrapping(np.multiply)),
    "square": _rounding("square", _wrapping(lambda a: a * a)),
    "div": _rounding("div", None),
    "sqrt": _rounding("sqrt", None),
    "neg": _moved(lambda params, a: -a),
    "abs": _moved(lambda params, a: np.abs(a)),
    "max": _extreme(_signed_max, np.maximum),
    "min": _extreme(_signed_min, np.minimum),
    "convert_element_type": _converted,
    "log": _rounding("log", None),
    "exp": _rounding("exp", None),
    "erf_inv": _rounding("erf_inv", None),
    "reduce_sum": _reduced_sum,
    "cumsum": _cumulative,
    "dot_general": _dot,
    "eq": _compared(lambda params, a, b: np.equal(a, b)),
    "ne": _compared(lambda params, a, b: np.not_equal(a, b)),
    "lt": _compared(lambda params, a, b: np.less(a, b)),
    "le": _compared(lambda params, a, b: np.less_equal(a, b)),
    "gt": _compared(lambda params, a, b: np.greater(a, b)),
    "ge": _compared(lambda params, a, b: np.greater_equal(a, b)),
    "and": _moved(lambda params, a, b: np.bitwise_and(a, b)),
    "or": _moved(lambda params, a, b: np.bitwise_or(a, b)),
    "not": _moved(lambda params, a: np.invert(a)),
    "shift_right_logical": _moved(_logical_shift(1)),
    "shift_left": _moved(_logical_shift(-1)),
    "bitcast_convert_type": _moved(_bitcast),
    "reduce_and": _moved(lambda params, a: np.all(a, axis=tuple(params["axes"]))),
    "reduce_or": _moved(lambda params, a: np.any(a, axis=tuple(params["axes"]))),
    "reduce_max": _extremum(_signed_max, _integral_extremum(np.max, "max")),
    "reduce_min": _extremum(_signed_min, _integral_extremum(np.min, "min")),
    "select_n": _moved(_selected),
    "broadcast_in_dim": _moved(_broadcast),
    "reshape": _moved(_reshaped),
    "squeeze": _moved(lambda params, a: np.squeeze(a, tuple(params["dimensions"]))),
    "expand_dims": _moved(lambda params, a: np.expand_dims(a, tuple(params["dimensions"]))),
    "transpose": _moved(lambda params, a: np.transpose(a, params["permutation"])),
    "slice": _moved(lambda params, a: a[tuple(slice(start, limit, step) for start, limit, step in zip(
        params["start_indices"], params["limit_indices"], params["strides"] or (1,) * a.ndim, strict=True))]),
    "concatenate": _moved(lambda params, *parts: np.concatenate(parts, axis=params["dimension"])),
    "copy": _moved(lambda params, a: a),
    "copy_p": _moved(lambda params, a: a),
    "optimization_barrier": _barrier,
    "pad": _moved(_padded),
    "iota": _moved(_iota),
    "gather": _moved(_gathered),
    "stack": _moved(lambda params, *parts: np.stack(parts, axis=params["axis"])),
    "random_bits": _random_bits,
    "random_wrap": _wrapped,
    "random_unwrap": _unwrapped,
}
"""Each primitive's contract, `contract(params, operands, result, fmt) -> ok`, the result's elements
that hold it, `fmt` the result's float format (None for an integer or boolean result): a rounding
primitive within the lane's measured reach for it (`LEVELS`, `lane`: correctly rounded, faithful or a
stated number of steps past, as the lane computes it); a sum within its bound in any order; an integer
operation wrapped to its width; or a result with no rounding at all, bit for bit (a move of data, a
sign operation, a comparison), every operand read as the lane reads it (`Lane.flushes`). A primitive
missing here cannot be certified, which fails the site's coverage, as does an integer result of a
primitive with no integer form (`div`, `sqrt`, a transcendental)."""


def certified(fn, *arguments) -> tuple:
    """`fn` on `arguments` run one primitive at a time, each executed alone and natively (an eager
    bind, its own compiled program, under its equation's own context), and each held to its contract
    (CONTRACTS) on the operands it was given: `(outputs, findings)`, the outputs as that run computed
    them, and every equation whose result breaks its contract or that has none, by its path, with its
    first element that breaks. This certifies the op-by-op program; a fused program's internals are
    check 1's."""
    import jax

    closed, shape = jax.make_jaxpr(fn, return_shape=True)(*arguments)
    findings: list[str] = []
    outputs = _run(closed.jaxpr, closed.consts, jax.tree.leaves(arguments), findings)
    return jax.tree.unflatten(jax.tree.structure(shape), outputs), findings


def _run(jaxpr, consts, arguments, findings, where: str = "") -> list:
    from jax.extend import core
    from reference_error import _body

    env = {}

    def read(var):
        # A literal at its traced type: its Python value may be wider (a float64 scalar for a float32).
        return np.asarray(var.val, var.aval.dtype) if isinstance(var, core.Literal) else env[var]

    env.update(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, arguments, strict=True))
    for index, eqn in enumerate(jaxpr.eqns):
        operands = [read(var) for var in eqn.invars]
        at = f"{where}{eqn.primitive.name}#{index}"
        with eqn.ctx.manager:
            body = _body(eqn)
            if eqn.primitive.name == "cond":
                branch = eqn.params["branches"][int(np.asarray(operands[0]))]
                results = _run(branch.jaxpr, branch.consts, operands[1:], findings, at + "/")
            elif eqn.primitive.name == "scan":
                results = _scanned(eqn.params, operands, findings, at + "/")
            elif body is not None:
                results = _run(*body, operands, findings, at + "/")
            else:
                results = eqn.primitive.bind(*operands, **eqn.params)
                results = list(results) if eqn.primitive.multiple_results else [results]
                _held(eqn, operands, results, findings, at)
        env.update(zip(eqn.outvars, results, strict=True))
    return [read(var) for var in jaxpr.outvars]


def _scanned(params, operands, findings, where: str) -> list:
    """A scan run a step at a time (`_run` of its body), its carry threaded from step to step and its
    outputs stacked, every step's equations held to their contracts as the program's are, each by its
    step's path."""
    import jax.numpy as jnp

    body = params["jaxpr"]
    constants, carries, _ = (len(group) for group in params["ft_in"].update(range(len(operands))).unpack())
    consts, carry, xs = (operands[:constants], operands[constants:constants + carries],
                         operands[constants + carries:])
    ys = []
    steps = range(params["length"])
    for step in reversed(steps) if params["reverse"] else steps:
        outputs = _run(body.jaxpr, body.consts, [*consts, *carry, *(x[step] for x in xs)], findings,
                       f"{where}{step}/")
        carry = outputs[:carries]
        ys.append(outputs[carries:])
    if params["reverse"]:
        ys.reverse()
    if ys:
        stacked = [jnp.stack(column) for column in zip(*ys, strict=True)]
    else:
        stacked = [jnp.zeros((0, *var.aval.shape), var.aval.dtype) for var in body.jaxpr.outvars[carries:]]
    return [*carry, *stacked]


def _held(eqn, operands, results, findings, at) -> None:
    import jax

    name = eqn.primitive.name
    if name not in CONTRACTS:
        findings.append(f"{at}: uncertifiable, no contract for {name}")
        return
    hosted = [_host(value) for value in operands]
    for value in [*hosted, *(_host(result) for result in results)]:
        dtype = getattr(value, "dtype", None)
        keyed = dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)
        if dtype is not None and not keyed and _inexact(dtype) and _format(dtype) is None:
            findings.append(f"{at}: uncertifiable, no format for {dtype}")
            return
    for result in results:
        fmt = _format(getattr(_host(result), "dtype", np.float32))
        try:
            ok = np.asarray(CONTRACTS[name](eqn.params, hosted, _host(result), fmt))
        except TypeError:
            findings.append(f"{at}: uncertifiable, {name} of these operands has no contract")
            return
        if not np.all(ok):
            hosted_result = _host(result)
            keyed = not isinstance(hosted_result, np.ndarray)
            shown = np.asarray(jax.random.key_data(result)) if keyed else hosted_result
            first = np.unravel_index(int(np.argmin(ok.ravel())), ok.shape) if ok.ndim else ()
            findings.append(f"{at}: {int(ok.size - ok.sum())} of {ok.size} elements break the contract, "
                            f"first at {first}: {shown[first]!r}")


def assert_certified(fn, draw, label: str, *, seeds: int) -> None:
    """Check 2 of the gate: `fn` on `draw(seed)`'s inputs for each of `seeds`, run one primitive at a
    time (`certified`), every element of every equation within its own operation's contract and every
    primitive covered by one."""
    for seed in range(seeds):
        _, findings = certified(fn, *draw(seed))
        assert not findings, f"{label}, seed {seed}: " + "; ".join(findings[:8])
