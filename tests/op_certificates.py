"""Per-operation certificates (a prototype, not a gate the suite uses): each value a kernel rounds is
checked on its own native parents against a contract for that one operation.

A node of a kernel's graph (`dew.nn.kernels.graph`) is one value the kernel rounds and holds. Given
the node's parents as the kernel computed them, its contract says what the native value may be:

- an IEEE operation (add, sub, mul, fma, a correctly rounded divide or square root, a cast) must be
  the round to nearest even of its exact result in its own format, subnormal cells included
  (`Exact`);
- an approximate or library primitive must be within its vendor-specified error of the true value
  (`Ulps`);
- a sum of n terms must be within the bound of the algorithm that sums it (`Sum`, `Dot`).

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
  beyond, and is x for k = 0); |x - k·_LN2_HI| / |k·_LN2_LO| >= 6.3e5 (fp32) and 1.9e10 (bf16), so
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
3,000 fp32 arguments over the range: 2^-94.7 at worst (tests/test_op_certificates.py holds it)."""


def ln2_enclosure(terms: int = 110) -> tuple[fractions.Fraction, fractions.Fraction]:
    """Exact rationals around ln 2 = 2 atanh(1/3) = 2 Σ (1/3)^(2k+1) / (2k+1): the series to `terms`
    terms, and that plus a bound on the rest, 2 (1/3)^(2n+1) / (2n+1) / (1 - 1/9) for n = `terms`."""
    third = fractions.Fraction(1, 3)
    partial = 2 * sum(third ** (2 * k + 1) / (2 * k + 1) for k in range(terms))
    rest = 2 * third ** (2 * terms + 1) / (2 * terms + 1) / (1 - third ** 2)
    return partial, partial + rest


def _exp_enclosed(x):
    """exp(x) for fp32 or bf16 x in [-110, 90], as a double-double within EXP_BOUND of it relative to it."""
    k = np.rint(x / float(_LN2))
    reduced_hi, reduced_lo = _fast_two_sum(x - k * _LN2_HI, -k * _LN2_LO)
    value_hi, value_lo = np.full_like(x, _COEFFICIENTS[-1][0]), np.full_like(x, _COEFFICIENTS[-1][1])
    for hi, lo in reversed(_COEFFICIENTS[:-1]):
        value_hi, value_lo = _dd_mul(value_hi, value_lo, reduced_hi, reduced_lo)
        value_hi, value_lo = _dd_add(value_hi, value_lo, np.full_like(x, hi), np.full_like(x, lo))
    return np.ldexp(value_hi, k.astype(np.int64)), np.ldexp(value_lo, k.astype(np.int64))


def exp_rounded(x, fmt: Format) -> np.ndarray:
    """RN(exp(x)) in `fmt` for values of a format no wider than fp32, verified: a double-double exp
    with a proven relative bound (`_exp_enclosed`, EXP_BOUND) decides every element whose enclosure
    rounds one way, and mpmath's interval exp at rising precision decides the rest (exp of a nonzero
    rational is never a midpoint, so the precision needed is finite). No library exp is trusted.
    Beyond [-110, 90] the result is 0 or inf in every format no wider than fp32 (exp(-110) is below
    half fp32's least subnormal, exp(90) past its largest finite value); NaN is NaN."""
    x = np.asarray(x, np.float64)
    inside = (x >= -110) & (x <= 90)
    hi, lo = _exp_enclosed(np.where(inside, x, 0.0))
    slack = np.abs(hi) * EXP_BOUND
    lower, upper = rounded(hi, lo - slack, fmt), rounded(hi, lo + slack, fmt)
    result = np.where(x > 90, np.inf, np.where(x < -110, 0.0, lower))
    result = np.where(np.isnan(x), np.nan, result)
    decided = ~inside | (lower == upper)
    for index in np.flatnonzero(~decided):
        for precision in (96, 192, 384, 768):
            with interval_precision(precision) as iv:
                ends = interval_ends(iv.exp(iv.mpf(float(x.flat[index]))), precision)
            low_end, high_end = (fraction_rounded(end, fmt) for end in ends)
            if low_end == high_end:
                result.flat[index] = low_end
                break
        else:
            raise ArithmeticError(f"exp({x.flat[index]!r}) undecided at 768 bits")
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
