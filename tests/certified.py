"""Check 2 as a gate: a site's computation run one primitive at a time (`certified`), each equation
held on the operands its execution gave it to its primitive's contract (CONTRACTS), from the exact
oracles of tests/op_certificates.py; a primitive no contract covers fails the site
(`assert_certified`), and the lane's arithmetic is held as it computes, measured on it (`lane`)."""

import contextlib
import dataclasses
import fractions
import functools
import math

import jax
import numpy as np
from jax.extend.core import primitives
from op_certificates import (
    DTYPES,
    EXP_RANGE,
    FORMATS,
    Format,
    exact_sum,
    gamma,
    interval_ends,
    interval_precision,
    magnitude_sum,
    nearest,
    rounded,
    stepped as oc_stepped,
    two_sum,
)

from dew.nn.precision import correctly_rounded_divide_p

STORAGE = {"float32": np.uint32, "bfloat16": np.uint16}
"""Each format's storage bits, which a move of data is compared by."""

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

    `reach`: by rounding primitive (the object, JAX's own) and format, the largest level (`LEVELS`) of
    the lane's own results
    over its probes (`_probes`): 0 where every one was its exact value rounded to nearest, 1 where each
    was one of the two neighbours of its exact value on the lane's grid (faithful), n where each was
    within n - 1 steps of the grid past them; a division by a size-1 broadcast divisor its own form's
    (`BROADCAST`).

    `kept`: the rounding primitives and formats that read a subnormal operand as it is on a lane that
    `flushes` (`SUBNORMAL_OPERANDS`): XLA on the CPU flushes per primitive, its expm1 and log1p returning
    a subnormal argument where its multiply reads one as zero (ledger 4023). Each is held as a lane that
    keeps subnormals (`flushes_in`)."""
    flushes: bool
    reach: dict[tuple[object, str], int]
    kept: frozenset[tuple[object, str]] = frozenset()

    def flushes_in(self, primitive, fmt: Format) -> bool:
        """Whether the rounding `primitive` flushes in `fmt` on this lane: the lane's, but where it reads a
        subnormal operand as it is (`kept`)."""
        return self.flushes and (primitive, fmt.name) not in self.kept


def _grid_index(value, fmt: Format, flushes: bool) -> np.ndarray:
    """Each value of `fmt` (as float64, finite or infinite) as its place on the lane's grid, in steps
    from zero, negative below it: its storage bits' magnitude, less the subnormals the grid skips where
    the lane flushes, so that the least normal is one step from zero there."""
    value = np.asarray(value, np.float64)
    magnitude = np.abs(value).astype(DTYPES[fmt.name]).view(STORAGE[fmt.name]).astype(np.int64)
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
    if jax.dtypes.issubdtype(getattr(value, "dtype", np.float32), jax.dtypes.prng_key):
        return value
    return np.asarray(value)


def _bits(value) -> np.ndarray:
    """A value's storage bits, element by element: what a move of data keeps."""
    value = np.asarray(value)
    if value.dtype == bool:
        return value.astype(np.uint8)
    if value.dtype.itemsize == 0:
        # float0, a symbolic zero's type (an integer input's tangent), holds no bits.
        return np.zeros(value.shape, np.uint8)
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


def rounded_on_lane(hi, lo, fmt: Format, flushes: bool) -> np.ndarray:
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
    value = np.asarray(value, np.float64)
    stepped = oc_stepped(value, fmt, direction)
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
    down = rounded_on_lane(low, np.zeros_like(low), fmt, flushes)
    down = np.where(down > low, _step(down, fmt, -np.inf, flushes), down)
    up = rounded_on_lane(high, np.zeros_like(high), fmt, flushes)
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


def _near_zero_interval(function):
    """An outward float64 enclosure of an mpmath interval `function` of `1 + x` (log1p's) or of `exp(x) - 1`
    (expm1's) at each finite value, its precision raised by the value's own exponent below one, so `1 + x`
    is exact at every scale down to fp32's least subnormal and the enclosure's width stays below the value's
    own last bit."""
    def enclose(values, bits):
        low, high = np.empty(values.shape), np.empty(values.shape)
        for index, value in enumerate(values.flat):
            raised = bits + max(0, -math.frexp(float(value))[1]) if value else bits
            with interval_precision(raised) as iv:
                a, b = interval_ends(function(iv, iv.mpf(float(value))), raised)
            low.flat[index], high.flat[index] = _outward(a, -math.inf), _outward(b, math.inf)
        return low, high
    return enclose


def _expm1_interval(values, bits):
    """expm1's outward enclosure at each finite value, taken at EXP_RANGE's nearer end beyond it, as exp's
    (`_exp_interval`): past its upper end the same two neighbours bound every expm1, and past its lower end
    every expm1 lies within a format's last step of -1."""
    return _near_zero_interval(lambda iv, x: iv.exp(x) - 1)(np.clip(values, *EXP_RANGE), bits)


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


def _log1p_special(value):
    """log1p's limit at -1 and infinity, its NaN below -1, and its exact values at ±0 (log1p(±0) = ±0),
    named."""
    named = np.isnan(value) | (value <= -1) | (value == np.inf) | (value == 0)
    expected = np.select([value < -1, value == -1, value == np.inf, value == 0],
                         [np.nan, -np.inf, np.inf, value], np.nan)
    return expected, named


def _expm1_special(value):
    """expm1's limits at the infinities and its exact values at ±0 (expm1(±0) = ±0), named."""
    named = ~np.isfinite(value) | (value == 0)
    return np.select([value == np.inf, value == -np.inf, value == 0], [np.inf, -1.0, value], np.nan), named


def _exp_special(value):
    """exp's limits at the infinities and its one exact value, exp(±0) = 1, named."""
    named = ~np.isfinite(value) | (value == 0)
    return np.select([value == np.inf, value == -np.inf, value == 0], [np.inf, 0.0, 1.0], np.nan), named


def _erfinv_special(value):
    """erfinv's poles at ±1 and its zero, exactly ±0 (it is odd), named; NaN outside (-1, 1)."""
    expected = np.where(np.abs(value) == 1, np.copysign(np.inf, value), np.where(value == 0, value, np.nan))
    return expected, ~(np.abs(value) < 1) | (value == 0)


def _sin_special(value):
    """sin's exact values at ±0 (sin(±0) = ±0), and its NaN at the infinities, named."""
    named = ~np.isfinite(value) | (value == 0)
    return np.where(value == 0, value, np.nan), named


def _cos_special(value):
    """cos's one exact value, cos(±0) = 1, and its NaN at the infinities, named."""
    named = ~np.isfinite(value) | (value == 0)
    return np.where(value == 0, 1.0, np.nan), named


def _logistic_interval(values, bits):
    """logistic's outward enclosure, 1 / (1 + exp(-x)), at each finite value, taken at EXP_RANGE's nearer
    end beyond it: below it every logistic is under exp(-110), between +0 and the least subnormal of a
    format no wider than fp32, and above it within exp(-90) of one, between one and the largest value
    below it, the verdicts of the ends'."""
    return _interval(lambda iv, x: 1 / (1 + iv.exp(-x)))(np.clip(values, *EXP_RANGE), bits)


def _logistic_special(value):
    """logistic's limits at the infinities and its one exact value, logistic(±0) = 1/2, named."""
    named = ~np.isfinite(value) | (value == 0)
    return np.select([value == np.inf, value == -np.inf, value == 0], [1.0, 0.0, 0.5], np.nan), named


def _rsqrt_special(value):
    """rsqrt's pole at zero, of the zero's sign, its zero at infinity, NaN below zero, and its exact
    values, at the powers of four (2^(2k), whose root's reciprocal 2^-k is on every grid; at no other
    value is it), named."""
    mantissa, exponent = np.frexp(np.where(np.isfinite(value) & (value > 0), value, 1.0))
    four = np.isfinite(value) & (value > 0) & (mantissa == 0.5) & ((exponent - 1) % 2 == 0)
    named = np.isnan(value) | (value <= 0) | (value == np.inf) | four
    expected = np.select([value < 0, value == 0, value == np.inf, four],
                         [np.nan, np.copysign(np.inf, value), 0.0, np.ldexp(1.0, -(exponent - 1) // 2)],
                         np.nan)
    return expected, named


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


def _terms_hold(native, terms, fmt: Format, flushes: bool, counts=None) -> np.ndarray:
    """Whether each native sum of `terms` (exact float64 values, summed along the last axis) is within
    the bound of any order of summing them in `fmt`, fused multiply-adds included: γ_n Σ|terms| (Higham,
    Accuracy and Stability of Numerical Algorithms, 2002, §3.1: n terms, at most n - 1 roundings on any
    path), around the terms' float64-rounded exact sum (`exact_sum`, within 2^-53 of it relative to
    Σ|terms|, in the allowance), plus what underflow loses past that relative bound at each rounding:
    where the lane flushes, a flushed partial's least normal; where it keeps subnormals, half their
    spacing (a product, or a sum carried wider and rounded into `fmt`). An empty sum is the additive
    identity, +0. `counts`, where given, is each sum's own count of terms, its padding zeros left out (a
    scatter's, whose elements take different numbers of updates)."""
    native = np.asarray(native, np.float64)
    count = terms.shape[-1]
    if count == 0:
        return _same(native, np.zeros_like(native))
    underflow = 2.0 ** fmt.emin if flushes else 2.0 ** (fmt.emin - fmt.precision)
    error = np.abs(native - exact_sum(terms, -1))
    slack = allowance(magnitude_sum(terms, -1), count if counts is None else counts, fmt, underflow)
    return np.isfinite(error) & (error <= np.nextafter(slack, np.inf))


def allowance(magnitude, count: int, fmt: Format, underflow: float):
    """How far any order's sum of `count` terms of total magnitude `magnitude` lies from their exact sum in
    `fmt`: γ_n of it (Higham, §3.1), with 2^-52 for the exact sum's own float64 rounding (`exact_sum`), and
    `underflow` lost at each rounding below the normal range. Check 2's sums (`_terms_hold`) and a native
    GEMM's (`native_certificates.native_dot_ends`) are held within it; `count` an int or each element's
    count, an array (an add of an exact zero rounding nothing)."""
    gammas = (np.vectorize(lambda n: gamma(int(n), fmt), otypes=[np.float64])(count) if np.ndim(count)
              else gamma(count, fmt))
    return (gammas + 2.0 ** -52) * magnitude + count * underflow


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
    where the result is that rounded to nearest on the lane (`rounded_on_lane`), else its place against
    the exact result's two neighbours (`_past`); an exactly infinite or undefined result (an infinite
    operand, a division by zero) 0 where the lane's is the same and UNREACHED elsewhere."""
    def levels(operands, result, fmt, flushes):
        hi, lo = build(*_floats(operands, flushes))
        result = np.asarray(result, np.float64)
        nearest = rounded_on_lane(hi, lo, fmt, flushes)
        # Rounding took the exact value up, or down: the other neighbour is one step of the grid away.
        above = (nearest > hi) | ((nearest == hi) & (lo < 0))
        below = (nearest < hi) | ((nearest == hi) & (lo > 0))
        down = np.where(above, _step(nearest, fmt, -np.inf, flushes), nearest)
        up = np.where(below, _step(nearest, fmt, np.inf, flushes), nearest)
        placed = np.where(np.isfinite(hi), _past(result, down, up, fmt, flushes), UNREACHED)
        return np.where(_same(result, nearest), 0, placed)
    return levels


def _truncated(result, dividend, divisor):
    """An integer division's quotient as `lax.div` gives it, truncated toward zero, wrapped to the result's
    width; by zero every bit set (-1 signed), as XLA defines it (its elemental emitter's integer division:
    `x / 0 == -1`, `INT_MIN / -1 == INT_MIN`, the latter the wrap)."""
    dtype = np.asarray(result).dtype
    dividends, divisors = (np.asarray(value).ravel().tolist()
                           for value in np.broadcast_arrays(dividend, divisor))
    # Python's integers hold every magnitude (int64's -2**63, uint64's past 2**63) exactly; the quotient wraps
    # to the result's width once.
    bits = 8 * dtype.itemsize
    quotients = [((abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1) if b else -1) % (1 << bits)
                 for a, b in zip(dividends, divisors, strict=True)]
    return np.asarray(quotients, np.uint64 if bits == 64 else np.int64).astype(dtype).reshape(
        np.shape(result))


def _remainder(dividend, divisor):
    """A remainder as `lax.rem` gives it, the dividend's sign; an integer one by zero the dividend and by -1
    zero, as XLA defines them (its elemental emitter's: `x % 0 == x`, `INT_MIN % -1 == 0`)."""
    if not np.issubdtype(np.result_type(dividend, divisor), np.integer):
        return np.fmod(dividend, divisor)
    dividend, divisor = np.broadcast_arrays(dividend, divisor)
    defined = (divisor != 0) & (divisor != -1)
    remainders = np.fmod(dividend, np.where(defined, divisor, 1))
    return np.where(divisor == 0, dividend, np.where(defined, remainders, 0))


def _rounding(primitive, integral):
    """A rounding primitive's contract: each float result's level (`LEVELS[primitive]`) within the lane's
    reach for it (`lane`); an integer result bit for bit (`integral`), where the primitive has one."""
    def contract(params, operands, result, fmt):
        if fmt is None:
            return _bits(result) == _bits(integral(result, *operands))
        measured = lane()
        # A division by a divisor broadcast along a size-1 axis of its own rank is held at that form's own
        # reach (`BROADCAST`); a scalar divisor's (a literal's, ledger 1402) is the division's.
        divisor = np.shape(operands[-1])
        broadcast = (primitive is jax.lax.div_p and len(divisor) == np.ndim(result)
                     and divisor != np.shape(result))
        reach = measured.reach[(primitive, BROADCAST) if broadcast else primitive, fmt.name]
        return LEVELS[primitive](operands, result, fmt, measured.flushes_in(primitive, fmt)) <= reach
    return contract


def _correctly_divided(params, operands, result, fmt):
    """Dew's correctly rounded division's contract (`dew.nn.precision.correctly_rounded_divide_p`): its exact
    quotient rounded to nearest, whatever the lane's own division reaches, a subnormal operand read as the
    lane's division reads it and a subnormal quotient kept, which it never flushes."""
    hi, lo = _quotient(*_floats(operands, lane().flushes_in(jax.lax.div_p, fmt)))
    return _same(np.asarray(result, np.float64), rounded_on_lane(hi, lo, fmt, flushes=False))


BROADCAST = "broadcast"
"""A division's form whose divisor is broadcast along a size-1 axis within the one equation (lax's implicit
broadcast), its reach measured apart (`lane`, under `(jax.lax.div_p, BROADCAST)`): XLA on the CPU computes
it as a multiply by the divisor's reciprocal, two roundings. A divisor broadcast by an equation of its
own (`broadcast_in_dim`) reaches its division materialized, which check 2 runs alone and the lane rounds
correctly (row 4011: none of 2048 such quotients off, though the same jitted whole was off at 545)."""


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


def _scatter_targets(params, operand, indices, updates):
    """Where each element of a lax.scatter's `updates` lands in `operand`, by XLA's semantics: an
    `[updates.size, operand.ndim]` array of operand indices, -1 rows for an update its window drops. Each
    update's scatter coordinates pick a start vector from `indices` (its last axis the index vector),
    placed on `scatter_dims_to_operand_dims` and the operand's batching axes, clamped so the window fits
    (the clip mode) or dropped whole where it does not (every other mode), then offset by the update's
    window coordinates on the operand's axes a window keeps."""
    numbers = params["dimension_numbers"]
    clip = params["mode"] == jax.lax.GatherScatterMode.CLIP
    window_axes = [axis for axis in range(operand.ndim)
                   if axis not in numbers.inserted_window_dims and axis not in numbers.operand_batching_dims]
    scatter_dims = [dim for dim in range(updates.ndim) if dim not in numbers.update_window_dims]
    window = [1] * operand.ndim
    for dim, axis in zip(numbers.update_window_dims, window_axes, strict=True):
        window[axis] = updates.shape[dim]
    targets = np.full((updates.size, operand.ndim), -1, np.int64)
    for flat, at in enumerate(np.ndindex(*updates.shape)):
        batch = tuple(at[dim] for dim in scatter_dims)
        vector = indices[batch]
        start = [0] * operand.ndim
        for position, axis in enumerate(numbers.scatter_dims_to_operand_dims):
            start[axis] = int(vector[position])
        for axis, batch_axis in zip(numbers.operand_batching_dims, numbers.scatter_indices_batching_dims,
                                    strict=True):
            start[axis] = batch[batch_axis]
        if clip:
            start = [min(max(begin, 0), operand.shape[axis] - window[axis])
                     for axis, begin in enumerate(start)]
        elif any(start[axis] < 0 or start[axis] + window[axis] > operand.shape[axis]
                 for axis in range(operand.ndim)):
            continue
        for dim, axis in zip(numbers.update_window_dims, window_axes, strict=True):
            start[axis] += at[dim]
        targets[flat] = start
    return targets


def _scattered(params, operand, indices, updates):
    """lax.scatter's result: `operand` with each update written where it lands (`_scatter_targets`), in
    their order, so of two updates of one element the later stands."""
    result = np.array(operand)
    targets = _scatter_targets(params, operand, indices, updates)
    for update, target in zip(np.asarray(updates).ravel(), targets, strict=True):
        if target[0] >= 0:
            result[tuple(target)] = update
    return result


def _scatter_added(params, operands, result, fmt):
    """A scatter-add's result: each operand element plus every update that lands on it (`_scatter_targets`),
    summed within its bound in any order (`_terms_hold`, each element's own count of terms); for integers,
    wrapped to the result's width."""
    operand, indices, updates = (np.asarray(value) for value in operands)
    targets = _scatter_targets(params, operand, indices, updates)
    landed = targets[:, 0] >= 0
    places = np.ravel_multi_index(targets[landed].T, operand.shape) if operand.ndim else np.zeros(
        int(landed.sum()), np.int64)
    values = updates.ravel()[landed]
    if fmt is None:
        summed = operand.astype(np.uint64).ravel()
        with np.errstate(over="ignore"):
            np.add.at(summed, places, values.astype(np.uint64))
        return _bits(result) == _bits(summed.astype(result.dtype).reshape(operand.shape))
    flushes = lane().flushes
    counts = np.bincount(places, minlength=operand.size)
    width = int(counts.max(initial=0)) + 1
    terms = np.zeros((operand.size, width))
    terms[:, 0] = _read(operand, flushes).ravel()
    slot = np.zeros(operand.size, np.int64)
    for place, value in zip(places, _read(values, flushes), strict=True):
        slot[place] += 1
        terms[place, slot[place]] = value
    native = np.asarray(result, np.float64).ravel()
    return _terms_hold(native, terms, fmt, flushes, counts=counts + 1).reshape(operand.shape)


def _ragged_dot(params, operands, result, fmt):
    """A ragged_dot_general's products, exact in float64, summed within their bound in any order
    (`_terms_hold`): the lhs's rows split into consecutive groups of `group_sizes`, either each row times
    its group's rhs over the contracting axis (a ragged row axis, the rhs grouped) or each group's rows
    alone summed (a ragged contracting axis, the result grouped), as the product's dimensions state it
    (`dew.nn.programs.ragged_layout`); a row past every group a zero. Other forms have no contract here."""
    from dew.nn.programs import RaggedLayout, ragged_layout

    lhs, rhs, sizes = (np.asarray(value) for value in operands)
    layout = ragged_layout(lhs.ndim, rhs.ndim, params["ragged_dot_dimension_numbers"])
    if layout not in (RaggedLayout.ROWS, RaggedLayout.GROUPS):
        raise TypeError("a ragged_dot_general of this form")
    group = np.repeat(np.arange(sizes.size), sizes)
    group = np.concatenate([group, np.full(lhs.shape[0] - group.size, -1)])
    flushes = lane().flushes
    left, right = _read(lhs, flushes), _read(rhs, flushes)
    if layout is RaggedLayout.ROWS:
        # [rows, columns, terms]: each row times its group's rhs, a row of no group times zeros.
        weights = np.where(group[:, None, None] >= 0, np.moveaxis(right, 1, 2)[np.maximum(group, 0)], 0.0)
        products = left[:, None, :] * weights
    else:
        # [groups, lhs columns, rhs columns, rows]: each group's rows' products, every other row a zero.
        member = group[None, :] == np.arange(sizes.size)[:, None]
        products = np.where(member[:, None, None, :],
                            left.T[None, :, None, :] * right.T[None, None, :, :], 0.0)
    native = np.asarray(result, np.float64).reshape(products.shape[:-1])
    return _terms_hold(native, products, fmt, flushes).reshape(np.shape(result))


def _power_ends(size: float, power: int, fmt: Format, lost: float) -> tuple[float, float]:
    """The least and the greatest magnitude the lane's `|x|^y` can take for `size = |x|`, finite and not
    zero, as JAX computes it: `acc = x^|y|` by squarings and multiplies from the low bit of `|y|`, and
    `1 / acc` where `y` is negative.

    A squaring's rounding is squared again by every squaring after it, so the j-th squaring's relative
    error enters the bit k's power with multiplicity 2^(k-j), and enters `acc` once for every set bit of
    `|y|` above it: `|y| >> j` times. Over the squarings that is `|y|` less its set bits (Legendre), and the
    multiplies of the set bits' powers add those bits less one: `|y| - 1` factors `(1 + δ)`, `|δ| ≤ u`, as
    `|y| - 1` plain multiplications make, so `acc` is within γ_(|y|-1) of the exact power (Higham,
    Accuracy and Stability of Numerical Algorithms, 2002, Lemma 3.1), plus what underflow loses at each
    multiplication, which no later one grows (a chain that underflows multiplies by magnitudes below one).
    The count of instructions, a square and a multiply a bit, undercounts it: x^8 is three squarings and
    seven factors.

    The reciprocal is one more rounding of `1 / acc` for whatever `acc` the chain made: the ends are the
    reciprocals of `acc`'s, each a rounding off, and so follow `acc` where it left the range, which a bound
    relative to the exact power does not: an `acc` that can round to an infinity (`Format.overflow`) has a
    reciprocal as small as zero, and one lost to underflow a reciprocal as large as an infinity. The ends
    carry 2^-50 for float64's own roundings in forming them."""
    count = max(abs(power) - 1, 0)
    try:
        exact = float(fractions.Fraction(size) ** abs(power))
    except OverflowError:
        exact = math.inf
    slack = allowance(exact, count, fmt, lost) if math.isfinite(exact) else 0.0
    low, high = max(exact - slack, 0.0), exact + slack
    if power < 0:
        near = 0.0 if high >= fmt.overflow else 1.0 / high
        far = 1.0 / low if low > 0 else math.inf
        low = max(near - allowance(near, 1, fmt, lost), 0.0)
        high = far + allowance(far, 1, fmt, lost) if math.isfinite(far) else math.inf
    return low * (1 - 2.0 ** -50), high * (1 + 2.0 ** -50)


def _integer_power(params, operands, result, fmt):
    """An integer_pow's result: each magnitude within what the lane's chain of multiplications, and its
    reciprocal where y is negative, can make of the operand's (`_power_ends`), an infinity where the chain
    can pass the format's range, and the power's sign exactly, a zero's among them; a zero's or a
    non-finite operand's power the lane's same (a pole of the zero's sign where the power is odd), an
    undefined one undefined. Integers wrap to the result's width."""
    (value,) = operands
    power = params["y"]
    if fmt is None:
        powered = _wrapping(lambda a: a ** abs(power))
        return _bits(result) == _bits(powered(result, value))
    read = _read(value, lane().flushes)
    lost = _underflow(fmt)
    low, high = np.empty(read.shape), np.empty(read.shape)
    for index, element in enumerate(read.flat):
        if np.isfinite(element) and element != 0:
            low.flat[index], high.flat[index] = _power_ends(abs(float(element)), power, fmt, lost)
        else:
            # A zero's or a non-finite value's power is exact in float64.
            with np.errstate(divide="ignore"):
                low.flat[index] = high.flat[index] = abs(np.float64(element) ** power)
    native = np.asarray(result, np.float64)
    size = np.abs(native)
    undefined = np.isnan(low)
    within = ((size >= low) & (size <= high)) | (np.isinf(size) & (high >= fmt.overflow))
    signed = np.signbit(native) == (np.signbit(read) & (power % 2 == 1))
    return np.where(undefined, np.isnan(native), within & signed)


def _keyed(value) -> bool:
    dtype = getattr(value, "dtype", None)
    return dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)


def _underflow(fmt: Format) -> float:
    """What one rounding loses below `fmt`'s normal range: its least normal where the lane flushes, else
    half its least subnormal."""
    return 2.0 ** fmt.emin if lane().flushes else 2.0 ** (fmt.emin - fmt.precision)


def _pair_held(params, operands, results, fmt) -> list[np.ndarray]:
    """A compensated sum's `(hi, lo)` pairs (`compensated_add`'s reduction of `(values, residuals)`): `hi +
    lo` within 5u²·n·Σ|terms| of the terms' exact sum, plus what underflow loses (8n of `_underflow`), and
    `lo` within a rounding of `hi`. Each combine's TwoSums are exact and its residuals' two additions
    round (Ogita, Rump and Oishi, Accurate Sum and Dot Product, 2005, §4: an error of at most
    2.01u(|d| + |l1| + |l2|) ≤ 4.03u²(|h1| + |h2|) a combine), at most n combines on a term's path. A
    TwoSum overflows only where its rounded sum does (Boldo, Graillat and Muller, On the robustness of
    the 2Sum and Fast2Sum algorithms, 2017, §5), and a TwoSum of an infinite sum takes inf - inf for its
    residual, so a combine reaching a non-finite value makes both halves NaN, and every combine after it:
    a sum of a non-finite term, or one some order of whose combines certainly overflows (its positive or
    negative terms, summed first, past the format's overflow threshold by γ_2n), may also be a NaN pair;
    any other sum's pair is finite and held, a NaN one a finding where no order is proven to make it."""
    flushes = lane().flushes
    axes = tuple(params["dimensions"])
    # A reduce's operands are its arrays, then each one's initial value, every output's first term.
    arrays, initial = operands[:len(operands) // 2], operands[len(operands) // 2:]
    terms = [np.moveaxis(_read(value, flushes), axes, range(-len(axes), 0)) for value in arrays]
    terms = np.concatenate([value.reshape(*value.shape[:value.ndim - len(axes)], -1) for value in terms], -1)
    starts = np.broadcast_to(np.stack([_read(value, flushes) for value in initial]), (*terms.shape[:-1], 2))
    terms = np.concatenate([starts, terms], -1)
    count = terms.shape[-1] // 2
    high, low = (np.asarray(value, np.float64) for value in results)
    exact, magnitude = exact_sum(terms, -1), magnitude_sum(terms, -1)
    u, lost = fmt.unit_roundoff, _underflow(fmt)
    bound = 5 * u * u * count * magnitude + 8 * count * lost + 2.0 ** -52 * (np.abs(high) + np.abs(exact))
    one_signed = np.maximum(exact_sum(np.maximum(terms, 0), -1), exact_sum(np.maximum(-terms, 0), -1))
    overflows = ~np.all(np.isfinite(terms), -1) | (one_signed * (1 - gamma(2 * count, fmt)) >= fmt.overflow)
    near = np.abs(high + low - exact) <= bound
    renormalized = np.abs(low) <= u * np.abs(high) + lost
    pair = np.isfinite(high) & np.isfinite(low) & near & renormalized
    held = pair | (overflows & np.isnan(high) & np.isnan(low))
    return [held, held]


def _softmax_held(params, operands, results, fmt) -> list[np.ndarray]:
    """A row's online softmax (`chunked._row_terms`' reduction of `(logits, ones, rest[, columns])`): its
    maximum exact, its maximum's first column exact, and each sum of exponentials scaled to the maximum
    within its bound of the exact Σ s_j e^(x_j - M): a term's path through at most n combines, each a
    subtraction of maxima (its error in the exponent, u|m - M|, summing along the path to at most
    u(M - x_j)), an exponential (the lane's reach for it, `lane`, relative r), a product and a sum, so the
    term within e^(u(M - x_j) + n(r + 3u)) - 1 of itself, plus what underflow loses (4n of `_underflow`).
    A row of a NaN may hold any sums; a row whose maximum is -inf keeps its sums as they are, summed
    (γ_n). Its initial values are one more term of every row, its maximum's column the least of the
    columns holding the maximum, as the combine keeps the lesser on a tie."""
    flushes = lane().flushes
    (axis,) = params["dimensions"]
    arrays, initial = operands[:len(operands) // 2], operands[len(operands) // 2:]
    values = []
    for value, start in zip(arrays, initial, strict=True):
        laid = np.moveaxis(_read(value, flushes), axis, -1)
        start = np.broadcast_to(_read(start, flushes), (*laid.shape[:-1], 1))
        values.append(np.concatenate([start, laid], -1))
    logits, scales = values[0], values[1:3]
    count = logits.shape[-1]
    peak = np.max(logits, -1)
    held = [_same(np.asarray(results[0], np.float64), peak)]
    nan = np.any(np.isnan(logits), -1)
    u, lost = fmt.unit_roundoff, _underflow(fmt)
    reach = 2 * u * lane().reach[jax.lax.exp_p, fmt.name]
    empty = peak == -np.inf
    with np.errstate(invalid="ignore", over="ignore"):
        shifted = np.where(empty[..., None], 0.0, logits - peak[..., None])
        terms = np.exp(shifted)
        spread = np.exp(np.minimum(-u * shifted + count * (reach + 3 * u), 700.0)) - 1
    for scale, result in zip(scales, results[1:3], strict=True):
        weighted = scale * terms
        exact, magnitude = exact_sum(weighted, -1), magnitude_sum(weighted, -1)
        bound = (np.sum(np.abs(weighted) * spread, -1) + 4 * count * lost
                 + gamma(count, fmt) * magnitude * empty + 2.0 ** -50 * magnitude)
        native = np.asarray(result, np.float64)
        held.append(nan | (np.isfinite(native) & (np.abs(native - exact) <= bound)))
    if len(results) == 4:
        least = np.min(np.where(logits == peak[..., None], values[3], np.iinfo(values[3].dtype).max), -1)
        held.append(nan | (np.asarray(results[3]) == least))
    return held


@functools.cache
def _reductions() -> dict:
    """Dew's variadic reductions check 2 holds, each by its combine, to its contract over all its results
    together. A `reduce` equation carries the combine it was given (its `computation`, the function its
    owner handed `jax.lax.reduce`), and each of these owners hands a function of its module: the
    compensated pair sum (`dew.nn.precision.compensated_add`) and a row's online softmax with and without
    its maximum's column (`dew.objectives.lm.chunked`). A reduction is one of these where its combine is
    that function, whatever its traced program prints as."""
    from dew.nn.precision import compensated_add
    from dew.objectives.lm import chunked

    return {compensated_add: _pair_held, chunked._online_softmax: _softmax_held,
            chunked._online_softmax_predicted: _softmax_held}


def _reduced(eqn, operands, results) -> list[np.ndarray]:
    """A variadic `reduce`'s held results, by the contract of the reduction its combine is (`_reductions`);
    a TypeError for one it is none of, and for one of operands that are no floats."""
    contract = _reductions().get(eqn.params["computation"])
    if contract is None or not np.issubdtype(np.asarray(operands[0]).dtype, np.floating):
        raise TypeError("a reduction of none of Dew's combines")
    return contract(eqn.params, operands, results, _format(np.asarray(results[0]).dtype))


def _moved(oracle):
    """A move of data or a sign operation: the result is `oracle(params, *operands)` in storage bits,
    subnormals and NaN payloads kept. A move of PRNG keys moves their data: each word of a key's data moved
    as its keys are (`oracle` over that word of every keyed operand), so its every word is held."""
    def contract(params, operands, result, fmt):
        if _keyed(result):
            words = np.asarray(jax.random.key_data(result))
            data = [np.asarray(jax.random.key_data(value)) if _keyed(value) else None for value in operands]

            def word(at):
                parts = (np.asarray(value) if part is None else part[..., at]
                         for value, part in zip(operands, data, strict=True))
                return np.asarray(oracle(params, *parts))
            expected = np.stack([word(at) for at in range(words.shape[-1])], axis=-1)
            return np.all(_bits(words) == _bits(expected.astype(words.dtype)), axis=-1)
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


def _dynamic_sliced(params, operand, *starts):
    """lax.dynamic_slice's result: the window of `slice_sizes` at `starts`, each start clamped so the window
    fits in the operand, as XLA clamps it."""
    sizes = params["slice_sizes"]
    at = [min(max(int(start), 0), extent - size)
          for start, extent, size in zip(starts, np.shape(operand), sizes, strict=True)]
    window = tuple(slice(begin, begin + size) for begin, size in zip(at, sizes, strict=True))
    return np.asarray(operand)[window]


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


def _gathered(params, operand, indices):
    """lax.gather's result, by XLA's semantics: each output index's batch coordinates pick a start vector
    from `indices` (its last axis the index vector), placed on the operand's `start_index_map` axes and
    its batching axes, clamped so the slice fits (the clip and promise-in-bounds modes) or the whole slice
    the fill value where it does not (fill-or-drop), then offset by the output's offset coordinates on the
    operand's axes a slice keeps."""
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
    filler = None
    if mode == fill:
        if params.get("fill_value") is not None:
            filler = params["fill_value"]
        elif jax.dtypes.issubdtype(operand.dtype, np.inexact):
            filler = np.nan
        elif operand.dtype == np.bool_:
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


def _ordered(value) -> np.ndarray:
    """Each value's place in the lane's total order of its type, as uint64, over the type's whole range: a
    float's sign and magnitude bits, so -NaN < -inf < ... < -0 < +0 < ... < +inf < NaN as XLA's sort and
    top_k order them; a signed integer's value past its type's least (its sign bit flipped), an unsigned
    one's and a boolean's as it is. A signed key overflowed at the ends: an unsigned 2^63 read negative,
    and the least int64 was its own negation."""
    value = np.asarray(value)
    if value.dtype == np.bool_ or np.issubdtype(value.dtype, np.unsignedinteger):
        return value.astype(np.uint64)
    half = np.uint64(1) << np.uint64(63)
    if np.issubdtype(value.dtype, np.integer):
        return value.astype(np.int64).view(np.uint64) ^ half
    bits = _bits(value).astype(np.uint64)
    sign = np.uint64(1) << np.uint64(8 * value.dtype.itemsize - 1)
    magnitude = bits & (sign - np.uint64(1))
    # A negative float below every positive one, the larger its magnitude the lower.
    return np.where(bits & sign, half - np.uint64(1) - magnitude, half + magnitude)


def _sort_key(value) -> np.ndarray:
    """A sort key as lax.sort compares it: a float's zeros one +0 and its NaNs one +NaN, after every other
    value (`_canonicalize_float_for_sort`, jax/_src/lax/lax.py); any other value as it is."""
    value = np.asarray(value)
    if _format(value.dtype) is None:
        return value
    zeroed = np.where(value == 0, value.dtype.type(0), value)
    return np.where(np.isnan(value), np.array(np.nan, value.dtype), zeroed)


def _sorted(params, operands) -> list[np.ndarray]:
    """lax.sort's results: every operand reordered along `dimension` by its first `num_keys` operands'
    total order (`_ordered`) of their canonical keys (`_sort_key`), the first key most significant,
    stably (ties in their places' order, as the lane's sort keeps them whether or not it is asked to)."""
    axis = params["dimension"]
    keys = [np.moveaxis(_ordered(_sort_key(value)), axis, -1) for value in operands[:params["num_keys"]]]
    order = np.lexsort(keys[::-1], axis=-1)
    return [np.moveaxis(np.take_along_axis(np.moveaxis(np.asarray(value), axis, -1), order, -1), -1, axis)
            for value in operands]


def _top(params, operand) -> list[np.ndarray]:
    """lax.top_k's results: along `axis`, its `k` greatest values in the total order (`_ordered`), the
    greatest first and of equal ones the earlier first, and their indices, int32."""
    axis = params.get("axis", -1)
    moved = np.moveaxis(np.asarray(operand), axis, -1)
    # The order reversed by each key's complement, which no key overflows as its negation did.
    order = np.argsort(~_ordered(moved), axis=-1, kind="stable")[..., :params["k"]]
    return [np.moveaxis(np.take_along_axis(moved, order, -1), -1, axis),
            np.moveaxis(order.astype(np.int32), -1, axis)]


def _rounded_to(params, value) -> np.ndarray:
    """lax.reduce_precision's result, as XLA computes it: `value`'s significand rounded to nearest even at
    `mantissa_bits`, in its own binade; where `exponent_bits` narrows its type's exponent, then past the
    narrow format's largest value infinite and below its least normal value a zero of the value's sign (the
    narrow format's subnormals flushed); NaN kept; in `value`'s own type."""
    import ml_dtypes

    value = np.asarray(value)
    exponent, mantissa = params["exponent_bits"], params["mantissa_bits"]
    own = ml_dtypes.finfo(value.dtype)
    wide = value.astype(np.float64)
    if exponent >= own.nexp:
        kept = Format("significand", mantissa + 1, own.minexp, own.maxexp - 1)
        return nearest(wide, kept).astype(value.dtype)
    largest = 2 ** (exponent - 1) - 1
    # No least exponent: the significand rounds in the value's own binade, the narrow range applied after.
    rounded = nearest(wide, Format(f"e{exponent}m{mantissa}", mantissa + 1, -(2 ** 20), largest))
    flushed = np.abs(rounded) < 2.0 ** (1 - largest)
    return np.where(flushed, np.copysign(0.0, wide), rounded).astype(value.dtype)


def _rounded_integral(params, value) -> np.ndarray:
    """lax.round's result: each value's nearest integer, a tie away from zero or to even
    (`rounding_method`), in the value's type, its sign kept."""
    value = np.asarray(value)
    wide = value.astype(np.float64)
    if params["rounding_method"] == jax.lax.RoundingMethod.TO_NEAREST_EVEN:
        found = np.rint(wide)
    else:
        found = np.copysign(np.floor(np.abs(wide) + 0.5), wide)
    return found.astype(value.dtype)


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
    (key,) = operands
    dtype = np.dtype(f"uint{params['bit_width']}")
    expected = jax.jit(lambda key: jax.random.bits(key, params["shape"], dtype))(key)
    return np.asarray(result) == np.asarray(expected)


def _unset(params, operands, result, fmt):
    """An uninitialized buffer (`empty2`, JAX's filler for the residual slots a `cond`'s other branch defines:
    conditionals.py `_join_cond_outputs`): no arithmetic, and the taken branch never reads the other
    branch's slots, so no value of it is any computation's."""
    return np.ones(np.shape(result), bool)


def _wrapped(params, operands, result, fmt):
    """A key wrapped from its data: the key's data is the operand, bit for bit."""
    return _bits(np.asarray(jax.random.key_data(result))) == _bits(operands[0])


def _unwrapped(params, operands, result, fmt):
    return _bits(result) == _bits(np.asarray(jax.random.key_data(operands[0])))


def _seeded(params, operands, result, fmt):
    """A key made from a seed is integers: its data is a separate `jax.random.key` of the seed's, of the
    equation's own implementation, bit for bit."""
    expected = jax.random.key_data(jax.random.key(operands[0], impl=params["impl"]))
    return _bits(np.asarray(jax.random.key_data(result))) == _bits(np.asarray(expected))


def _folded(params, operands, result, fmt):
    """A key folded with data is integers: its data is a separate `jax.random.fold_in` of the same key and
    data's, bit for bit."""
    key, data = operands
    expected = jax.jit(jax.random.fold_in)(key, data)
    return _bits(np.asarray(jax.random.key_data(result))) == _bits(np.asarray(jax.random.key_data(expected)))


LEVELS = {
    jax.lax.add_p: _arithmetic(two_sum),
    jax.lax.sub_p: _arithmetic(lambda a, b: two_sum(a, -b)),
    jax.lax.mul_p: _arithmetic(_product),
    jax.lax.square_p: _arithmetic(lambda a: _product(a, a)),
    jax.lax.div_p: _arithmetic(_quotient),
    jax.lax.sqrt_p: _arithmetic(_root),
    jax.lax.log_p: _transcendental(_interval(lambda iv, x: iv.log(x)), _log_special),
    jax.lax.exp_p: _transcendental(_exp_interval, _exp_special),
    jax.lax.log1p_p: _transcendental(_near_zero_interval(lambda iv, x: iv.log(1 + x)), _log1p_special),
    jax.lax.expm1_p: _transcendental(_expm1_interval, _expm1_special),
    jax.lax.erf_inv_p: _transcendental(_erfinv_enclosed, _erfinv_special),
    jax.lax.rsqrt_p: _transcendental(_interval(lambda iv, x: 1 / iv.sqrt(x)), _rsqrt_special),
    jax.lax.logistic_p: _transcendental(_logistic_interval, _logistic_special),
    # mpmath's interval sine and cosine reduce their argument at the enclosure's own precision, rigorously:
    # an argument too large for it leaves the enclosure wide, undecided, and its precision is raised.
    jax.lax.sin_p: _transcendental(_interval(lambda iv, x: iv.sin(x)), _sin_special),
    jax.lax.cos_p: _transcendental(_interval(lambda iv, x: iv.cos(x)), _cos_special),
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


def _probes(primitive, fmt: Format) -> list[np.ndarray]:
    """PROBES operands of the rounding `primitive` in `fmt`, over its domain at every scale its
    results reach: sums of every alignment, a quarter of them nearly cancelling; products and quotients
    from about 2^-140 (subnormal on a lane that keeps subnormals) to 2^126; roots of every binade,
    subnormals included; logarithms of every binade, a quarter near one, a quarter in [1, 4) and the
    witness a site found past the reach; exponentials across fp32's
    range and a quarter near zero; log1p and expm1 at every normal binade near zero and across their
    domains; inverse error functions across (-1, 1) and a quarter near its poles;
    reciprocal roots of every binade, a quarter at the powers of four where they are exact; logistics
    across fp32's range and a quarter near zero; sines and cosines of every binade a rotary table reaches
    and a quarter across a few periods; and each transcendental at its exact value (exp(0), log(1),
    erf_inv(0), logistic(0), sin(0), cos(0)), which a lane must return exactly. One fixed draw per primitive
    and format."""
    rng = np.random.default_rng([*primitive.name.encode(), fmt.precision])
    quarter = PROBES // 4

    def scaled(low: int, high: int) -> np.ndarray:
        signs = rng.choice([-1.0, 1.0], PROBES)
        return signs * rng.uniform(1, 2, PROBES) * 2.0 ** rng.integers(low, high, PROBES)

    if primitive in (jax.lax.add_p, jax.lax.sub_p):
        a, b = scaled(-30, 30), scaled(-30, 30)
        near = a[:quarter] * (1 + rng.uniform(-2.0 ** -10, 2.0 ** -10, quarter))
        b[:quarter] = -near if primitive is jax.lax.add_p else near
        drawn = [a, b]
    elif primitive in (jax.lax.mul_p, jax.lax.square_p):
        drawn = [scaled(-70, 63) for _ in range(1 if primitive is jax.lax.square_p else 2)]
    elif primitive is jax.lax.div_p:
        drawn = [scaled(-70, 63), scaled(-63, 70)]
    elif primitive is jax.lax.sqrt_p:
        drawn = [np.abs(scaled(-149, 127))]
    elif primitive is jax.lax.log_p:
        value = np.abs(scaled(-126, 127))
        value[:quarter] = 1 + rng.uniform(-2.0 ** -8, 2.0 ** -8, quarter)
        # A quarter where a logsumexp's sums of exponentials sit, and XLA:CPU's float32 log a step past
        # faithful at 2.6954336 (0x402c81fc: 0.99155915 against 0.991559088..., a quantization-aware
        # step's chunked cross entropy, check 2), which the probes grew by.
        value[quarter:2 * quarter] = rng.uniform(1, 4, quarter)
        value[1] = np.float32(np.uint32(0x402C81FC).view(np.float32))
        value[0] = 1.0
        drawn = [value]
    elif primitive in (jax.lax.exp_p, jax.lax.logistic_p):
        value = rng.uniform(-104, 89, PROBES)
        value[:quarter] = rng.uniform(-2.0 ** -8, 2.0 ** -8, quarter)
        value[0] = 0.0
        drawn = [value]
    elif primitive in (jax.lax.log1p_p, jax.lax.expm1_p):
        # Every normal binade near zero, where the functions are their argument to within a last bit (their
        # enclosures' precision raised there, `_near_zero_interval`); a quarter densest across (-0.5, 1),
        # where XLA's log1p rounds farthest, and out to 8; and their domains' reach: log1p's down to just
        # above -1 and up to fp32's largest, expm1's across fp32's range. Their subnormal binades too, which a
        # lane reads as its own primitive does (`Lane.flushes_in`: XLA:CPU's expm1 keeps one its multiply
        # flushes, ledger 4023).
        value = scaled(fmt.emin - fmt.precision + 1, 0)
        value[quarter:2 * quarter] = (rng.uniform(-1, 0, quarter) if primitive is jax.lax.log1p_p
                                      else rng.uniform(-104, 89, quarter))
        value[2 * quarter:3 * quarter] = rng.uniform(-0.5, 1, quarter) * 2.0 ** rng.integers(0, 4, quarter)
        value[3 * quarter:] = np.abs(scaled(0, 127))[:PROBES - 3 * quarter]
        value[0] = 0.0
        drawn = [value]
    elif primitive is jax.lax.rsqrt_p:
        value = np.abs(scaled(-149, 127))
        value[:quarter] = 4.0 ** rng.integers(-74, 64, quarter)
        drawn = [value]
    elif primitive in (jax.lax.sin_p, jax.lax.cos_p):
        # Every binade from 2^-30 to 2^20, the angles a rotary table reaches, and a quarter across a few
        # periods, where the functions turn.
        value = scaled(-30, 21)
        value[:quarter] = rng.uniform(-8, 8, quarter)
        value[0] = 0.0
        drawn = [value]
    else:
        value = rng.uniform(-1, 1, PROBES)
        poles = 1 - 2.0 ** -rng.uniform(1, fmt.precision, quarter)
        value[:quarter] = rng.choice([-1.0, 1.0], quarter) * poles
        value[0] = 0.0
        drawn = [value]
    return [value.astype(DTYPES[fmt.name]) for value in drawn]


SUBNORMAL_OPERANDS = {
    jax.lax.add_p: lambda tiny: (tiny, tiny), jax.lax.sub_p: lambda tiny: (tiny, -tiny),
    jax.lax.mul_p: lambda tiny: (tiny, 2.0 ** 100), jax.lax.div_p: lambda tiny: (tiny, 2.0 ** -100),
    jax.lax.sqrt_p: lambda tiny: (tiny,), jax.lax.rsqrt_p: lambda tiny: (tiny,),
    jax.lax.log_p: lambda tiny: (tiny,), jax.lax.log1p_p: lambda tiny: (tiny,),
    jax.lax.expm1_p: lambda tiny: (-tiny,),
    jax.lax.erf_inv_p: lambda tiny: (tiny,), jax.lax.sin_p: lambda tiny: (tiny,),
}
"""Each rounding primitive's operands at a subnormal `tiny` whose result tells a subnormal read as it is from
one read as zero (`lane`'s `kept`); exp, logistic, square and cos return the same result either way, so a
lane's flushing holds them as it holds every other operation."""


def _function(primitive):
    """`jax.lax`'s function of a rounding primitive, which bears the primitive's own name: what binds it
    with the parameters it needs (a transcendental's `accuracy`)."""
    return getattr(jax.lax, primitive.name)


def _keeps_subnormals(primitive, fmt: Format) -> bool:
    """Whether the lane's `primitive` in `fmt` reads a subnormal operand as it is: its result at
    `SUBNORMAL_OPERANDS`' subnormal differs from its result with that operand zero."""
    tiny = 2.0 ** (fmt.emin - 4)
    subnormal = [np.asarray(value, DTYPES[fmt.name]) for value in SUBNORMAL_OPERANDS[primitive](tiny)]
    # Each subnormal zeroed with its sign, as a lane that flushes it reads it.
    zeroed = [np.asarray(np.copysign(0.0, value) if abs(float(value)) == tiny else value, DTYPES[fmt.name])
              for value in SUBNORMAL_OPERANDS[primitive](tiny)]
    run = _function(primitive)
    return bool(np.any(_bits(np.asarray(run(*subnormal))) != _bits(np.asarray(run(*zeroed)))))


@functools.cache
def lane() -> Lane:
    """The lane check 2 runs on (the process's default backend), measured once: whether it reads a
    subnormal operand as zero (2^-140 times 2^100), the rounding primitives that read one as it is where
    it does (`_keeps_subnormals`), and each rounding primitive's reach in each format, the largest level
    (`LEVELS`) of the lane's own results on its probes (`_probes`), each operation run as `certified`
    runs it, eagerly and alone, at the primitive's own flushing. A lane result that breaks an exact case
    (UNREACHED) is a finding no reach admits, raised here."""
    flushes = float(jax.lax.mul(np.float32(2.0 ** -140), np.float32(2.0 ** 100))) == 0.0
    kept = frozenset((primitive, fmt.name) for primitive in SUBNORMAL_OPERANDS for fmt in FORMATS.values()
                     if flushes and _keeps_subnormals(primitive, fmt))
    measured = Lane(flushes, {}, kept)
    reach = measured.reach
    for primitive, levels in LEVELS.items():
        for fmt in FORMATS.values():
            operands = _probes(primitive, fmt)
            found = levels(operands, np.asarray(_function(primitive)(*operands)), fmt,
                           measured.flushes_in(primitive, fmt))
            if np.any(found == UNREACHED):
                at = int(np.argmax(found == UNREACHED))
                raise ArithmeticError(f"the lane's {primitive.name} in {fmt.name} breaks an exact case at "
                                      f"{[operand[at] for operand in operands]}")
            reach[primitive, fmt.name] = int(np.max(found))
    for fmt in FORMATS.values():
        # A division whose divisor is broadcast along a size-1 axis, a row's scale's, which XLA on the CPU
        # computes as a multiply by the divisor's reciprocal, two roundings: measured as a form of its own.
        dividend, divisor = _probes(jax.lax.div_p, fmt)
        rows = PROBES // 32
        operands = [dividend[:rows * 32].reshape(rows, 32), divisor[:rows].reshape(rows, 1)]
        found = LEVELS[jax.lax.div_p](operands, np.asarray(jax.lax.div(*operands)), fmt,
                                      measured.flushes_in(jax.lax.div_p, fmt))
        if np.any(found == UNREACHED):
            raise ArithmeticError(f"the lane's broadcast div in {fmt.name} breaks an exact case")
        reach[(jax.lax.div_p, BROADCAST), fmt.name] = int(np.max(found))
    return measured


CONTRACTS = {
    jax.lax.add_p: _rounding(jax.lax.add_p, _wrapping(np.add)),
    # A cotangents' sum, the add its transpose emits, lowered as `add` is: held at add's reach.
    primitives.add_jaxvals_p: _rounding(jax.lax.add_p, _wrapping(np.add)),
    jax.lax.sub_p: _rounding(jax.lax.sub_p, _wrapping(np.subtract)),
    jax.lax.mul_p: _rounding(jax.lax.mul_p, _wrapping(np.multiply)),
    jax.lax.square_p: _rounding(jax.lax.square_p, _wrapping(lambda a: a * a)),
    jax.lax.div_p: _rounding(jax.lax.div_p, _truncated),
    correctly_rounded_divide_p: _correctly_divided,
    # A remainder (C's, the dividend's sign) is exact in IEEE arithmetic and in integers, and a sign is a
    # sign operation, each kept zero's sign: bit for bit.
    jax.lax.rem_p: _moved(lambda params, a, b: _remainder(a, b)),
    jax.lax.sign_p: _moved(lambda params, a: np.where(a == 0, a, np.sign(a))),
    jax.lax.sqrt_p: _rounding(jax.lax.sqrt_p, None),
    jax.lax.neg_p: _moved(lambda params, a: -a),
    jax.lax.abs_p: _moved(lambda params, a: np.abs(a)),
    jax.lax.max_p: _extreme(_signed_max, np.maximum),
    jax.lax.min_p: _extreme(_signed_min, np.minimum),
    jax.lax.convert_element_type_p: _converted,
    jax.lax.log_p: _rounding(jax.lax.log_p, None),
    jax.lax.exp_p: _rounding(jax.lax.exp_p, None),
    jax.lax.log1p_p: _rounding(jax.lax.log1p_p, None),
    jax.lax.expm1_p: _rounding(jax.lax.expm1_p, None),
    jax.lax.erf_inv_p: _rounding(jax.lax.erf_inv_p, None),
    jax.lax.rsqrt_p: _rounding(jax.lax.rsqrt_p, None),
    jax.lax.logistic_p: _rounding(jax.lax.logistic_p, None),
    jax.lax.sin_p: _rounding(jax.lax.sin_p, None),
    jax.lax.cos_p: _rounding(jax.lax.cos_p, None),
    jax.lax.reduce_sum_p: _reduced_sum,
    jax.lax.cumsum_p: _cumulative,
    jax.lax.dot_general_p: _dot,
    jax.lax.eq_p: _compared(lambda params, a, b: np.equal(a, b)),
    jax.lax.ne_p: _compared(lambda params, a, b: np.not_equal(a, b)),
    jax.lax.lt_p: _compared(lambda params, a, b: np.less(a, b)),
    jax.lax.le_p: _compared(lambda params, a, b: np.less_equal(a, b)),
    jax.lax.gt_p: _compared(lambda params, a, b: np.greater(a, b)),
    jax.lax.ge_p: _compared(lambda params, a, b: np.greater_equal(a, b)),
    jax.lax.and_p: _moved(lambda params, a, b: np.bitwise_and(a, b)),
    jax.lax.or_p: _moved(lambda params, a, b: np.bitwise_or(a, b)),
    jax.lax.not_p: _moved(lambda params, a: np.invert(a)),
    jax.lax.shift_right_logical_p: _moved(_logical_shift(1)),
    jax.lax.shift_left_p: _moved(_logical_shift(-1)),
    jax.lax.bitcast_convert_type_p: _moved(_bitcast),
    jax.lax.reduce_and_p: _moved(lambda params, a: np.all(a, axis=tuple(params["axes"]))),
    jax.lax.reduce_or_p: _moved(lambda params, a: np.any(a, axis=tuple(params["axes"]))),
    jax.lax.reduce_max_p: _extremum(_signed_max, _integral_extremum(np.max, "max")),
    jax.lax.reduce_min_p: _extremum(_signed_min, _integral_extremum(np.min, "min")),
    jax.lax.select_n_p: _moved(_selected),
    jax.lax.broadcast_in_dim_p: _moved(_broadcast),
    jax.lax.reshape_p: _moved(_reshaped),
    jax.lax.squeeze_p: _moved(lambda params, a: np.squeeze(a, tuple(params["dimensions"]))),
    jax.lax.transpose_p: _moved(lambda params, a: np.transpose(a, params["permutation"])),
    jax.lax.slice_p: _moved(lambda params, a: a[tuple(slice(start, limit, step) for start, limit, step in zip(
        params["start_indices"], params["limit_indices"], params["strides"] or (1,) * a.ndim, strict=True))]),
    jax.lax.concatenate_p: _moved(lambda params, *parts: np.concatenate(parts, axis=params["dimension"])),
    jax.lax.copy_p: _moved(lambda params, a: a),
    jax.lax.optimization_barrier_p: _moved(lambda params, a: a),
    primitives.name_p: _moved(lambda params, a: a),
    jax.lax.rev_p: _moved(lambda params, a: np.flip(a, tuple(params["dimensions"]))),
    jax.lax.split_p: _moved(lambda params, a: a),
    jax.lax.pad_p: _moved(_padded),
    jax.lax.iota_p: _moved(_iota),
    jax.lax.gather_p: _moved(_gathered),
    jax.lax.stack_p: _moved(lambda params, *parts: np.stack(parts, axis=params["axis"])),
    jax.lax.dynamic_slice_p: _moved(_dynamic_sliced),
    # Exact functions of a value as the lane's arithmetic reads it (a flushed subnormal a zero).
    jax.lax.floor_p: _moved(lambda params, a: np.floor(_read(a, lane().flushes)).astype(a.dtype)),
    jax.lax.nextafter_p: _moved(lambda params, a, b: np.nextafter(a, b)),
    jax.lax.scatter_p: _moved(_scattered),
    jax.lax.scatter_add_p: _scatter_added,
    primitives.ragged_dot_general_p: _ragged_dot,
    jax.lax.integer_pow_p: _integer_power,
    jax.lax.sort_p: _moved(lambda params, part: part),
    jax.lax.top_k_p: _moved(lambda params, part: part),
    jax.lax.unstack_p: _moved(lambda params, part: part),
    jax.lax.tile_p: _moved(lambda params, a: np.tile(a, params["reps"])),
    jax.lax.stop_gradient_p: _moved(lambda params, a: a),
    jax.lax.xor_p: _moved(lambda params, a, b: np.bitwise_xor(a, b)),
    # A shift by the operand's width or more (an amount read unsigned) leaves its sign in every bit.
    jax.lax.shift_right_arithmetic_p: _moved(lambda params, a, b: np.right_shift(a, np.minimum(
        b.astype(a.dtype).view(f"u{a.dtype.itemsize}").astype(np.uint64), 8 * a.dtype.itemsize - 1)
        .astype(a.dtype))),
    jax.lax.round_p: _moved(_rounded_integral),
    jax.lax.reduce_precision_p: _moved(_rounded_to),
    jax.lax.is_finite_p: _compared(lambda params, a: np.isfinite(a)),
    primitives.random_seed_p: _seeded,
    primitives.random_fold_in_p: _folded,
    primitives.random_bits_p: _random_bits,
    primitives.empty2_p: _unset,
    primitives.random_wrap_p: _wrapped,
    primitives.random_unwrap_p: _unwrapped,
}
"""Each primitive's contract, `contract(params, operands, result, fmt) -> ok`, the result's elements
that hold it, `fmt` the result's float format (None for an integer or boolean result): a rounding
primitive within the lane's measured reach for it (`LEVELS`, `lane`: correctly rounded, faithful or a
stated number of steps past, as the lane computes it); a sum within its bound in any order; an integer
operation wrapped to its width; or a result with no rounding at all, bit for bit (a move of data, a
sign operation, a comparison), every operand read as the lane reads it (`Lane.flushes`). A primitive
missing here cannot be certified, which fails the site's coverage, as does an integer result of a
primitive with no integer form (`div`, `sqrt`, a transcendental)."""
def _piece(params, value, index):
    """lax.split's `index`-th result: its operand's run of `sizes[index]` along `axis`, after the earlier
    sizes'."""
    start = sum(params["sizes"][:index])
    return np.take(value, np.arange(start, start + params["sizes"][index]), axis=params["axis"])


RESULT_OPERANDS = {
    jax.lax.optimization_barrier_p: lambda params, operands, index: [operands[index]],
    jax.lax.split_p: lambda params, operands, index: [_piece(params, operands[0], index)],
    jax.lax.unstack_p: lambda params, operands, index: [np.take(operands[0], index, axis=params["axis"])],
    jax.lax.sort_p: lambda params, operands, index: [_sorted(params, operands)[index]],
    jax.lax.top_k_p: lambda params, operands, index: [_top(params, operands[0])[index]],
}
"""The primitives of several results whose each result is a move of its operands of its own position,
`parts(params, operands, index)`, which its contract reads alone (`_held`): an optimization barrier's i-th
operand, a split's i-th piece, an unstack's i-th slice, a sort's i-th operand reordered (`_sorted`), a
top_k's values or their indices (`_top`)."""


def certified(fn, *arguments) -> tuple:
    """`fn` on `arguments` run one primitive at a time, each executed alone and natively (an eager
    bind, its own compiled program, under its equation's own context), and each held to its contract
    (CONTRACTS) on the operands it was given: `(outputs, findings)`, the outputs as that run computed
    them, and every equation whose result breaks its contract or that has none, by its path, with its
    first element that breaks. The program is walked as the gate walks every program
    (`float64_terms.walked`, by `Certifying`): its calls, branches and scans inside, a scan a step at a
    time, each step's equations by the step's path, and a call of one of JAX's high-level primitives as the
    program JAX lowers it to (`float64_terms.lowered`), so no equation is held as a call it cannot open.
    This certifies the op-by-op program; a fused program's internals are check 1's."""
    from float64_terms import evaluated_on, lowered, walked

    closed, shape = jax.make_jaxpr(fn, return_shape=True)(*arguments)
    closed = lowered(closed)
    findings: list[str] = []
    leaves = jax.tree.leaves(arguments)
    with evaluated_on(leaves):
        outputs = walked(closed.jaxpr, closed.consts, leaves, Certifying(findings))
    return jax.tree.unflatten(jax.tree.structure(shape), outputs), findings


@dataclasses.dataclass(frozen=True)
class Certifying:
    """The walk's rule of check 2 (`float64_terms.Rule`, `certified`), unpruned: every equation executed is
    held, its outputs read or not; each primitive bound natively on its operands and held to its contract
    (`_held`), its findings appended to `findings` by its path: each call's equation `name#index`, a scan's
    step its index, joined by `/`."""

    findings: list
    path: tuple[str, ...] = ()
    pruned = False

    def at(self, index: int, name: str | None = None) -> "Certifying":
        return Certifying(self.findings, (*self.path, str(index) if name is None else f"{name}#{index}"))

    def held(self, dtype):
        return dtype

    def primitive(self, eqn, operands: list) -> list:
        results = eqn.primitive.bind(*operands, **eqn.params)
        results = list(results) if eqn.primitive.multiple_results else [results]
        _held(eqn, operands, results, self.findings, "/".join(self.path))
        return results


_OBSERVING: list[list] = []


@contextlib.contextmanager
def observing():
    """Every equation `certified` holds within, `(primitive, params, operands, results)` in the order held:
    what each contract was given (`_held`), read where it is given."""
    seen: list = []
    _OBSERVING.append(seen)
    try:
        yield seen
    finally:
        # By identity: two observers that have seen nothing yet are equal lists.
        del _OBSERVING[next(index for index, observer in enumerate(_OBSERVING) if observer is seen)]


def _held(eqn, operands, results, findings, at) -> None:
    # An operation is its primitive object: one of another library's or a user's that bears a name of
    # JAX's has no contract here, and is refused by that name.
    primitive, name = eqn.primitive, eqn.primitive.name
    for seen in _OBSERVING:
        seen.append((primitive, eqn.params, list(operands), list(results)))
    if primitive not in CONTRACTS and primitive is not jax.lax.reduce_p:
        findings.append(f"{at}: uncertifiable, no contract for {name}")
        return
    hosted = [_host(value) for value in operands]
    for value in [*hosted, *(_host(result) for result in results)]:
        dtype = getattr(value, "dtype", None)
        keyed = dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)
        if dtype is not None and not keyed and _inexact(dtype) and _format(dtype) is None:
            findings.append(f"{at}: uncertifiable, no format for {dtype}")
            return
    if primitive is jax.lax.reduce_p:
        # A variadic reduction's results are held together, by the reduction its combine is.
        try:
            held = _reduced(eqn, hosted, [_host(result) for result in results])
        except TypeError:
            findings.append(f"{at}: uncertifiable, no contract for this reduce's combine")
            return
        for ok in held:
            if not np.all(ok):
                findings.append(f"{at}: {int(ok.size - ok.sum())} of {ok.size} elements break the contract")
        return
    for index, result in enumerate(results):
        fmt = _format(getattr(_host(result), "dtype", np.float32))
        # A result of its own position reads its own part of the operands alone (a barrier's, a split's).
        given = (RESULT_OPERANDS[primitive](eqn.params, hosted, index) if primitive in RESULT_OPERANDS
                 else hosted)
        try:
            ok = np.asarray(CONTRACTS[primitive](eqn.params, given, _host(result), fmt))
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
