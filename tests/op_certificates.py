"""Check 2 of the gate: every element within a certified roundoff bound for its own operation.

A site's computation runs one primitive at a time, each executed alone and natively, and each equation
is held on the operands that execution gave it to its primitive's contract (tests/certified.py's
`certified` and CONTRACTS, from this module's exact oracles):

- a rounding operation (add, sub, mul, divide, square root, a transcendental) must lie within the
  reach the lane's own results show for it, measured on the lane (tests/certified.py's `lane`): the
  round to nearest even of its exact result in its own format where the lane rounds so, one of the two
  values of its format around the exact value (faithful), or a stated number of steps past them, its
  operands and results read as the lane computes (flushing subnormals or not);
- a cast between formats must be the round to nearest even of its value;
- a sum of n terms, a dot product's included, must be within γ_n of its terms' magnitudes of their
  exact sum, in any order;
- a comparison, a selection, a move of data or an integer operation must be its result bit for bit.

A primitive no contract covers fails the site (tests/certified.py's `assert_certified`). This
certifies the program run one operation at a time; a fused program's internals are check 1's
(tests/reference_error.py).

The exact results come from exact arithmetic, not from float64 rounding: a product of two fp32 or
bf16 values is exact in float64, a sum or a fused multiply-add is carried as a double-double by
TwoSum, and rounding to the node's format reads the low part only where the high part is itself a
midpoint of the format, where it decides the direction. float64's own rounding of `a*b + c` can land
on an fp32 midpoint the exact value is not on (a = b = 1 + 2^-12, c = 2^-80), so a single float64
evaluation is not an oracle for a fused multiply-add.

Bounds are computed with outward rounding: γ(k) = ku/(1-ku) in exact rationals, rounded up to a
float64, and every float64 sum of magnitudes widened by its own worst-case error.
"""

import contextlib
import dataclasses
import fractions
import math

import ml_dtypes
import numpy as np


@dataclasses.dataclass(frozen=True)
class Format:
    """A binary floating-point format: its precision in bits (the leading one counted), its least
    normal exponent and its greatest exponent."""
    name: str
    precision: int
    emin: int
    emax: int

    @property
    def unit_roundoff(self) -> float:
        return 2.0 ** -self.precision

    @property
    def largest(self) -> float:
        return (2.0 - 2.0 ** (1 - self.precision)) * 2.0 ** self.emax


FLOAT32 = Format("float32", 24, -126, 127)
BFLOAT16 = Format("bfloat16", 8, -126, 127)
FORMATS = {"float32": FLOAT32, "bfloat16": BFLOAT16}
TFLOAT32 = Format("tfloat32", 11, -126, 127)
"""TF32: fp32's range at 11 bits, the operands a GPU's tensor cores multiply fp32 products in at the default
precision (src/dew/nn/kernels/templates/product.py); no array holds it, so it is no dtype's (FORMATS)."""


def two_sum(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Knuth's TwoSum in float64: hi = fl(a + b) and lo with hi + lo = a + b exactly."""
    hi = a + b
    b_virtual = hi - a
    lo = (a - (hi - b_virtual)) + (b - b_virtual)
    return hi, lo


def _quantum(value: np.ndarray, fmt: Format) -> np.ndarray:
    """The spacing of `fmt` at each value's binade, the subnormal spacing below the least normal."""
    _, exponent = np.frexp(value)
    binade = np.maximum(exponent - 1, fmt.emin)
    return np.ldexp(1.0, binade - (fmt.precision - 1))


def rounded(hi: np.ndarray, lo: np.ndarray, fmt: Format) -> np.ndarray:
    """Round to nearest even in `fmt` of the exact value hi + lo (a double-double, abs(lo) at most
    half a float64 ulp of hi), as float64.

    A midpoint of `fmt` is a float64, so any midpoint strictly between hi and the exact value would
    be a float64 nearer it than hi: rounding hi alone is right except where hi is a midpoint, where
    the sign of lo, and only a zero lo, leaves it a tie."""
    hi, lo = np.asarray(hi, np.float64), np.asarray(lo, np.float64)
    finite = np.isfinite(hi)
    quantum = _quantum(np.where(finite, hi, 0.0), fmt)
    scaled = np.where(finite, hi, 0.0) / quantum
    floor = np.floor(scaled)
    # floor rounds toward -inf, so above a midpoint is floor + 1 whatever hi's sign.
    tie = (scaled - floor == 0.5) & (lo != 0)
    nearest = np.where(tie, np.where(lo > 0, floor + 1, floor), np.rint(scaled))
    result = np.copysign(nearest * quantum, hi)
    # Overflow: past the largest finite value plus half its spacing, or on it unless lo pulls the
    # exact value back below (the largest significand is odd, so its tie goes to the next binade).
    edge = fmt.largest + 2.0 ** (fmt.emax - fmt.precision)
    inward = (lo != 0) & ((lo > 0) != (hi > 0))
    beyond = (np.abs(hi) > edge) | ((np.abs(hi) == edge) & ~inward)
    return np.where(~finite, hi, np.where(beyond, np.copysign(np.inf, hi), result))


def _quantized(value, fmt: Format, offset: float) -> np.ndarray:
    """Each exact value's magnitude in units of `fmt`'s spacing at it, plus `offset`, floored, as a value of
    `fmt` of the value's sign (as float64); infinite past `fmt`'s range. For values no wider than half of
    float64's precision every step is exact: the scaling by a power of two, the offset's add and the floor."""
    value = np.asarray(value, np.float64)
    finite = np.isfinite(value)
    quantum = _quantum(np.where(finite, value, 0.0), fmt)
    result = np.copysign(np.floor(np.abs(np.where(finite, value, 0.0)) / quantum + offset) * quantum, value)
    overflows = np.abs(result) > fmt.largest
    return np.where(~finite, value, np.where(overflows, np.copysign(np.inf, value), result))


def rounded_away(value, fmt: Format) -> np.ndarray:
    """Round to nearest in `fmt`, a tie away from zero (PTX's `.rna`), of exact values, as float64."""
    return _quantized(value, fmt, 0.5)


def truncated(value, fmt: Format) -> np.ndarray:
    """Exact values rounded toward zero in `fmt`, as float64."""
    return _quantized(value, fmt, 0.0)


def exact_add(a, b, fmt: Format) -> np.ndarray:
    """RN(a + b) in `fmt` for values of a format no wider than fp32."""
    return rounded(*two_sum(np.asarray(a, np.float64), np.asarray(b, np.float64)), fmt)


def exact_mul(a, b, fmt: Format) -> np.ndarray:
    """RN(a·b) in `fmt`: the product of two fp32 values is exact in float64 (48 bits)."""
    product = np.asarray(a, np.float64) * np.asarray(b, np.float64)
    return rounded(product, np.zeros_like(product), fmt)


def exact_fma(a, b, c, fmt: Format) -> np.ndarray:
    """RN(a·b + c) in `fmt`, one rounding: the exact product, then TwoSum with c."""
    return rounded(*two_sum(np.asarray(a, np.float64) * np.asarray(b, np.float64), np.asarray(c, np.float64)),
                   fmt)


def exact_div(a, b, fmt: Format) -> np.ndarray:
    """RN(a/b) in `fmt`. A quotient of two p-bit values is never a midpoint at precision p, and
    float64's correctly rounded quotient has more than 2p + 2 bits, so rounding it again is exact
    (Figueroa, 1995)."""
    quotient = np.asarray(a, np.float64) / np.asarray(b, np.float64)
    return rounded(quotient, np.zeros_like(quotient), fmt)


def exact_cast(value, fmt: Format) -> np.ndarray:
    """RN of an exact value (float64, fp32 or bf16) in `fmt`."""
    value = np.asarray(value, np.float64)
    return rounded(value, np.zeros_like(value), fmt)


def fraction_rounded(value: fractions.Fraction, fmt: Format) -> float:
    """Round to nearest even in `fmt` of an exact rational, by exact arithmetic: the reference the
    float64 oracle is checked against."""
    if value == 0:
        return 0.0
    sign, magnitude = (-1.0 if value < 0 else 1.0), abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if fractions.Fraction(2) ** exponent > magnitude:
        exponent -= 1
    quantum = fractions.Fraction(2) ** (max(exponent, fmt.emin) - (fmt.precision - 1))
    steps = magnitude / quantum
    whole = steps.numerator // steps.denominator
    rest = steps - whole
    if rest > fractions.Fraction(1, 2) or (rest == fractions.Fraction(1, 2) and whole % 2 == 1):
        whole += 1
    result = whole * quantum
    if result > fractions.Fraction(fmt.largest):
        return sign * math.inf
    return sign * float(result)


def gamma(terms: int, fmt: Format) -> float:
    """γ(k) = ku/(1-ku), rounded up to a float64, or inf where ku ≥ 1."""
    product = terms * fractions.Fraction(fmt.unit_roundoff)
    if product >= 1:
        return math.inf
    value = product / (1 - product)
    nearest = float(value)
    return nearest if fractions.Fraction(nearest) >= value else math.nextafter(nearest, math.inf)


def magnitude_sum(terms: np.ndarray, axis: int) -> np.ndarray:
    """Σ abs(terms) along `axis` in float64, widened by float64's own worst case so it is an upper
    bound of the exact sum."""
    count = terms.shape[axis]
    total = np.sum(np.abs(np.asarray(terms, np.float64)), axis=axis)
    widen = 1.0 + 2.0 * count * 2.0 ** -53
    return np.nextafter(total * widen, np.inf)


def exact_sum(terms: np.ndarray, axis: int) -> np.ndarray:
    """Σ terms along `axis`, correctly rounded to float64 (Shewchuk's exact partials, `math.fsum`)."""
    moved = np.moveaxis(np.asarray(terms, np.float64), axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    return np.array([math.fsum(row) for row in flat]).reshape(moved.shape[:-1])


def mismatches(native, oracle, fmt: Format) -> np.ndarray:
    """Where a native value is not bit for bit its oracle, in the native format's own bits: signed
    zeros and infinities compared as bits, and a NaN on either side a mismatch."""
    dtype = {"float32": np.float32, "bfloat16": ml_dtypes.bfloat16}[fmt.name]
    native = np.asarray(native).astype(dtype)
    oracle = np.asarray(oracle, np.float64).astype(dtype)
    bits = np.uint32 if fmt.name == "float32" else np.uint16
    return (native.view(bits) != oracle.view(bits)) | np.isnan(native.astype(np.float64)) | np.isnan(
        oracle.astype(np.float64))


def _decided(low: np.ndarray, high: np.ndarray, fmt: Format) -> tuple[np.ndarray, np.ndarray]:
    """RN in `fmt` of each end of an enclosure of the exact value, and where the ends agree, so the
    rounding of every value between them is that one (round to nearest is monotone)."""
    lower, upper = exact_cast(low, fmt), exact_cast(high, fmt)
    return lower, (lower == upper) | (np.isnan(lower) & np.isnan(upper))


@contextlib.contextmanager
def interval_precision(bits: int):
    """mpmath's interval context at `bits` of precision, restored after (it has no workprec)."""
    import mpmath

    held = mpmath.iv.prec
    mpmath.iv.prec = bits
    try:
        yield mpmath.iv
    finally:
        mpmath.iv.prec = held


def interval_ends(interval, bits: int) -> tuple[fractions.Fraction, fractions.Fraction]:
    """An mpmath interval's two ends as exact rationals, through mpmath's public interface: each end
    (`.a`, `.b`) read as an mpf at the `bits` the interval was computed at, so exactly, then its
    mantissa's magnitude and exponent (`.man`, `.exp`) and its sign."""
    import mpmath

    def exact(end) -> fractions.Fraction:
        with mpmath.workprec(bits):
            value = mpmath.mpf(end)
        # `.man` is the mantissa's magnitude: the sign is the value's own.
        magnitude = fractions.Fraction(int(value.man)) * fractions.Fraction(2) ** int(value.exp)
        return -magnitude if value < 0 else magnitude

    return exact(interval.a), exact(interval.b)


def _fast_two_sum(a, b):
    """s + e = a + b exactly, for |a| >= |b| or a = 0 (Dekker)."""
    total = a + b
    return total, b - (total - a)


def _split(a):
    """a = hi + lo, each of at most 26 significant bits (Veltkamp, by 2^27 + 1)."""
    scaled = 134217729.0 * a
    hi = scaled - (scaled - a)
    return hi, a - hi


def _two_product(a, b):
    """p + e = a·b exactly (Dekker's product, no fused multiply-add)."""
    product = a * b
    a_hi, a_lo = _split(a)
    b_hi, b_lo = _split(b)
    return product, ((a_hi * b_hi - product) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo


def _dd_mul(a_hi, a_lo, b_hi, b_lo):
    """A double-double product, within 7·2^-106 of the exact one relative to it (Joldes, Muller and
    Popescu, ACM TOMS 44(2), 2017, DWTimesDW1: 7u^2)."""
    product, error = _two_product(a_hi, b_hi)
    return _fast_two_sum(product, error + (a_hi * b_lo + a_lo * b_hi))


def _dd_add(a_hi, a_lo, b_hi, b_lo):
    """A double-double sum, within 4·2^-106 of the exact one relative to it, whatever the operands'
    signs (Joldes, Muller and Popescu 2017, AccurateDWPlusDW: 3u^2 + 13u^3 < 4u^2)."""
    total, error = two_sum(a_hi, b_hi)
    low, low_error = two_sum(a_lo, b_lo)
    total, error = _fast_two_sum(total, error + low)
    return _fast_two_sum(total, error + low_error)


_LN2 = fractions.Fraction(
    "0.6931471805599453094172321214581765680755001343602552541206800094933936219696947156058633269964186875")
"""ln 2 to 100 digits (OEIS A002162), within 2^-330 of it, far below every bound here; the tests hold it
to an exact enclosure of ln 2 = 2 atanh(1/3) (`ln2_enclosure`)."""
_LN2_HI = float(fractions.Fraction(round(_LN2 * 2 ** 44), 2 ** 44))
"""ln 2 to 44 fractional bits, 0x1.62e42fefa3a00p-1: k·_LN2_HI is exact in float64 for |k| <= 160."""
_LN2_LO = float(_LN2 - fractions.Fraction(_LN2_HI))
"""The rest of ln 2, -0x1.0ca86c3898d00p-49."""
_TAYLOR = 24
_COEFFICIENTS = [(float(c), float(c - fractions.Fraction(float(c))))
                 for c in (fractions.Fraction(1, math.factorial(n)) for n in range(_TAYLOR + 1))]
"""1/n! for n <= _TAYLOR as double-doubles, each within 2^-106 of it relative to it."""
EXP_BOUND = 2.0 ** -80
"""A relative bound on `_exp_enclosed`'s error, from its terms (the arithmetic's 2^-89 and below):
- the reduced argument r = x - k·ln 2 for fp32 or bf16 x in [-110, 90]: x - k·_LN2_HI is exact
  (k·_LN2_HI is exact for |k| <= 160, and the difference is exact by Sterbenz at every k = ±1 edge and
  beyond, and is x for k = 0); |x - k·_LN2_HI| / |k·_LN2_LO| >= 6e5 (fp32) and 1e10 (bf16), so
  `_fast_two_sum`'s precondition holds; k·_LN2_LO is within 2^-53 of itself, |k·_LN2_LO| <= 160·2^-44,
  and _LN2_LO within 2^-98 of ln 2 - _LN2_HI: r is within 2^-89 of the exact x - k·ln 2, |r| <= 0.3466,
  so exp(r) within 2^-88.9 relative;
- the Taylor remainder at |r| <= 0.347 after 25 terms: 0.347^25/25!·e^0.347 < 2^-118;
- the double-double Horner evaluation, 25 steps each of one product (7u^2) and one sum (4u^2 relative
  to it), with 2^-106 coefficient roundings. Each step adds c_n = 1/n! to r times a partial value at
  most 0.347/(n + 1)·e^0.347 of it, so there is no severe cancellation: for r < 0 every partial value
  P_n >= 0.653·c_n, so c_n/P_n <= 1/0.653 < 1.6, and for r >= 0 the sum has one sign. Below
  25·(7 + 4 + 1)·1.6·2^-106 < 2^-97;
- scaling by 2^k is exact (the results stay normal in float64).
Total below 2^-88; the bound is 2^-80 to keep 2^8 of slack. Measured against mpmath at 300 bits on
3,000 fp32 arguments over the range: 2^-94.7 at worst (tests/test_op_certificates.py holds it).
`_exp2_enclosed` holds it too: its reduced argument r = f·ln 2 for the fraction f = x - rint(x),
|f| <= 1/2 and exact, is f·_LN2_HI exactly (`_two_product`, its low part at most 2^-55) plus
f·_LN2_LO (below 2^-49, rounded within 2^-103), the two parts' sum rounded within 2^-103, and
f·(_LN2_HI + _LN2_LO) is within 2^-99 of f·ln 2: r is within 2^-98 of f·ln 2, |r| <= 0.3466, so
exp(r) within 2^-97.9 relative, and the same series and exact scaling by 2^rint(x) follow."""


def ln2_enclosure(terms: int = 110) -> tuple[fractions.Fraction, fractions.Fraction]:
    """Exact rationals around ln 2 = 2 atanh(1/3) = 2 Σ (1/3)^(2k+1) / (2k+1): the series to `terms`
    terms, and that plus a bound on the rest, 2 (1/3)^(2n+1) / (2n+1) / (1 - 1/9) for n = `terms`."""
    third = fractions.Fraction(1, 3)
    partial = 2 * sum(third ** (2 * k + 1) / (2 * k + 1) for k in range(terms))
    rest = 2 * third ** (2 * terms + 1) / (2 * terms + 1) / (1 - third ** 2)
    return partial, partial + rest


def _exp_series(reduced_hi, reduced_lo, k):
    """exp(r)·2^k for the double-double r, |r| <= 0.347, by its Taylor series in double-double."""
    def constant(value):
        return np.full_like(reduced_hi, value)

    value_hi, value_lo = constant(_COEFFICIENTS[-1][0]), constant(_COEFFICIENTS[-1][1])
    for hi, lo in reversed(_COEFFICIENTS[:-1]):
        value_hi, value_lo = _dd_mul(value_hi, value_lo, reduced_hi, reduced_lo)
        value_hi, value_lo = _dd_add(value_hi, value_lo, constant(hi), constant(lo))
    return np.ldexp(value_hi, k.astype(np.int64)), np.ldexp(value_lo, k.astype(np.int64))


def _exp_enclosed(x):
    """exp(x) for fp32 or bf16 x in [-110, 90], as a double-double within EXP_BOUND of it relative to it."""
    k = np.rint(x / float(_LN2))
    return _exp_series(*_fast_two_sum(x - k * _LN2_HI, -k * _LN2_LO), k)


def _exp2_enclosed(x):
    """2^x for fp32 or bf16 x in (-150, 128), as a double-double within EXP_BOUND of it relative to it:
    2^f·2^n for x = n + f, n = rint(x), 2^f = exp(f·ln 2)."""
    n = np.rint(x)
    fraction = x - n
    product, error = _two_product(fraction, _LN2_HI)
    return _exp_series(*_fast_two_sum(product, error + fraction * _LN2_LO), n)


def _enclosure_rounded(x, hi, lo, inside, beyond, exact, name: str, fmt: Format) -> np.ndarray:
    """RN in `fmt` of a function at each of `x`, from its double-double enclosure `hi + lo` within
    EXP_BOUND where `inside`: an element whose enclosure rounds one way takes that rounding, and
    mpmath's interval `exact(iv, x)` at rising precision decides the rest. Outside, inf where `beyond`
    and 0 elsewhere; NaN is NaN."""
    slack = np.abs(hi) * EXP_BOUND
    # Each end renormalized: `rounded` reads a double-double whose low part is within half an ulp of hi.
    lower, upper = (rounded(*two_sum(hi, lo + offset), fmt) for offset in (-slack, slack))
    result = np.where(inside, lower, np.where(beyond, np.inf, 0.0))
    result = np.where(np.isnan(x), np.nan, result)
    decided = ~inside | (lower == upper)
    for index in np.flatnonzero(~decided):
        for precision in (96, 192, 384, 768):
            with interval_precision(precision) as iv:
                ends = interval_ends(exact(iv, iv.mpf(float(x.flat[index]))), precision)
            low_end, high_end = (fraction_rounded(end, fmt) for end in ends)
            if low_end == high_end:
                result.flat[index] = low_end
                break
        else:
            raise ArithmeticError(f"{name}({x.flat[index]!r}) undecided at 768 bits")
    return result


EXP_RANGE = (-110.0, 90.0)
"""Where exp's results in a format no wider than fp32 lie strictly between the same two neighbours:
below it every exp is under exp(-110), below half fp32's least subnormal (bf16's is larger), so
between +0 and the least subnormal, and above it every exp is over exp(90), past fp32's and bf16's
largest finite values, so between the largest finite value and inf."""


def exp_rounded(x, fmt: Format) -> np.ndarray:
    """RN(exp(x)) in `fmt` for values of a format no wider than fp32, verified: a double-double exp
    with a proven relative bound (`_exp_enclosed`, EXP_BOUND) decides every element whose enclosure
    rounds one way, and mpmath's interval exp at rising precision decides the rest (exp of a nonzero
    rational is never a midpoint, so the precision needed is finite). No library exp is trusted.
    Beyond EXP_RANGE the result is 0 or inf in every format no wider than fp32; NaN is NaN."""
    x = np.asarray(x, np.float64)
    low, high = EXP_RANGE
    inside = (x >= low) & (x <= high)
    hi, lo = _exp_enclosed(np.where(inside, x, 0.0))
    return _enclosure_rounded(x, hi, lo, inside, x > high, lambda iv, value: iv.exp(value), "exp", fmt)


def exp2_rounded(x, fmt: Format) -> np.ndarray:
    """RN(2^x) in `fmt` for values of a format no wider than fp32, verified as `exp_rounded` is, from
    `_exp2_enclosed`: 2^x of a non-integer rational is irrational, never a midpoint, and of an integer
    is a power of two, rounded exactly (`fraction_rounded`, which takes a tie, half the least
    subnormal, to even). At -150 and below the result is 0 in every format no wider than fp32 (2^-150
    is half fp32's least subnormal, and bf16's is larger), at 128 and above inf; NaN is NaN."""
    x = np.asarray(x, np.float64)
    inside = (x > -150) & (x < 128)
    whole = inside & (x == np.rint(x))
    hi, lo = _exp2_enclosed(np.where(inside & ~whole, x, 0.0))
    result = _enclosure_rounded(x, hi, lo, inside & ~whole, x >= 128,
                                lambda iv, value: iv.exp(value * iv.log(2)), "exp2", fmt)
    for index in np.flatnonzero(whole):
        result.flat[index] = fraction_rounded(fractions.Fraction(2) ** int(x.flat[index]), fmt)
    return result


def rsqrt_rounded(x, fmt: Format) -> np.ndarray:
    """RN(1/sqrt(x)) in `fmt` for positive values of a format no wider than fp32, exactly: float64's
    1/sqrt rounds twice, so it is within 2^-51 of the exact value relative to it, and an element whose
    enclosure straddles a midpoint m of `fmt` is decided by the sign of m²·x - 1 in exact rationals."""
    x = np.asarray(x, np.float64)
    estimate = 1 / np.sqrt(x)
    margin = 2.0 ** -51
    result, decided = _decided(estimate * (1 - margin), estimate * (1 + margin), fmt)
    for index in np.flatnonzero(~decided & np.isfinite(estimate) & (x > 0)):
        value = float(x.flat[index])
        below = float(exact_cast(estimate.flat[index] * (1 - margin), fmt))
        above = below + float(_quantum(np.asarray(below), fmt))
        midpoint = (fractions.Fraction(below) + fractions.Fraction(above)) / 2
        side = midpoint * midpoint * fractions.Fraction(value) - 1
        # 1/sqrt(x) > m exactly when m²·x < 1; a tie (m²·x = 1) goes to the even neighbour.
        if side < 0:
            result.flat[index] = above
        elif side > 0:
            result.flat[index] = below
        else:
            result.flat[index] = float(fraction_rounded(midpoint, fmt))
    return result
