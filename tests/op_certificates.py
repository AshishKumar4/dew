"""Check 2 of the gate: every element within a certified roundoff bound for its own operation.

A site's computation runs one primitive at a time, each executed alone and natively (`certified`), and
each equation is held on the operands that execution gave it to its primitive's contract (CONTRACTS):

- an IEEE operation (add, sub, mul, a correctly rounded divide or square root, a cast) must be the
  round to nearest even of its exact result in its own format, read as the lane computes (`FLUSHES`);
- a library primitive, a transcendental included, must be faithful: one of the two values of its
  format around the exact value, whatever its vendor documents;
- a sum of n terms, a dot product's included, must be within γ_n of its terms' magnitudes of their
  exact sum, in any order;
- a comparison, a selection, a move of data or an integer operation must be its result bit for bit.

A primitive no contract covers fails the site (`assert_certified`). This certifies the program run one
operation at a time; a fused program's internals are check 1's (tests/reference_error.py).

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
    return _enclosure_rounded(x, hi, lo, inside, x > 90, lambda iv, value: iv.exp(value), "exp", fmt)


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



# Check 2 as a gate: a site's computation run one primitive at a time, each held to its contract.

FLUSHES = True
"""Whether the lane's float arithmetic flushes, as XLA on the CPU lane does, jitted or eager, with
YNNPACK's fusions or without (tests/test_check2.py holds each fact, so a JAX that changes one fails
there first): an arithmetic operation or comparison reads a subnormal operand as a zero of its sign,
and a result that is tiny after rounding (below the least normal once rounded to its precision with an
unbounded exponent) is a zero of the exact result's sign. A move of data, a sign operation (neg, abs)
and a cast between float formats are bit manipulation: they keep subnormals."""

STORAGE = {"float32": np.uint32, "bfloat16": np.uint16}
"""Each format's storage bits, which a move of data is compared by."""


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


def _daz(value, fmt: Format) -> np.ndarray:
    """`value` (float64) with every subnormal of `fmt` read as a zero of its sign, as the lane's
    arithmetic reads it."""
    value = np.asarray(value, np.float64)
    if not FLUSHES:
        return value
    return np.where(np.abs(value) < 2.0 ** fmt.emin, np.copysign(0.0, value), value)


def _read(value) -> np.ndarray:
    """An operand as the lane's arithmetic reads it: a float of a known format as float64, flushed
    (`_daz`); any other as it is."""
    fmt = _format(np.asarray(value).dtype)
    return np.asarray(value) if fmt is None else _daz(np.asarray(value, np.float64), fmt)


def _same(native, expected) -> np.ndarray:
    """Equal values, element by element, zeros by sign and NaN by NaN: an arithmetic result's identity
    (a NaN's payload is no part of IEEE's contract)."""
    native, expected = np.asarray(native, np.float64), np.asarray(expected, np.float64)
    return ((native == expected) & (np.signbit(native) == np.signbit(expected))) | (
        np.isnan(native) & np.isnan(expected))


_UNBOUNDED = {name: Format(f"{name} unbounded", fmt.precision, -1022, fmt.emax)
              for name, fmt in FORMATS.items()}
"""Each format's precision at every exponent: what the lane's tininess is read from."""


def _rounded_on_lane(hi, lo, fmt: Format) -> np.ndarray:
    """RN of the exact hi + lo in `fmt` as the lane's arithmetic rounds it: gradually, or with FLUSHES a
    zero of the exact value's sign wherever the value rounded with an unbounded exponent is below the
    least normal (tininess after rounding)."""
    gradual = rounded(hi, lo, fmt)
    if not FLUSHES:
        return gradual
    unbounded = rounded(hi, lo, _UNBOUNDED[fmt.name])
    tiny = np.isfinite(unbounded) & (np.abs(unbounded) < 2.0 ** fmt.emin)
    return np.where(tiny, np.copysign(0.0, np.where(hi != 0, hi, gradual)), gradual)


def _step(value, fmt: Format, direction: float) -> np.ndarray:
    """The next value of `fmt` after each `value` (a value of `fmt`, as float64) toward `direction`,
    on the lane's grid: with FLUSHES its outputs hold no subnormal, so the next value past zero is the
    least normal and the next below the least normal is zero."""
    dtype = {"float32": np.float32, "bfloat16": ml_dtypes.bfloat16}[fmt.name]
    value = np.asarray(value, np.float64)
    stepped = np.nextafter(value.astype(dtype), np.full(value.shape, direction, dtype)).astype(np.float64)
    if FLUSHES:
        least = 2.0 ** fmt.emin
        tiny = np.abs(stepped) < least
        # Past zero toward a subnormal the grid's next value is the least normal; from the least
        # normal toward zero it is zero.
        jumped = np.where(np.abs(value) >= least, 0.0, np.copysign(least, direction))
        stepped = np.where(tiny, jumped, stepped)
    return stepped


def _faithful(native, low, high, fmt: Format) -> tuple[np.ndarray, np.ndarray]:
    """Whether `native` is faithful to an exact value enclosed in [low, high] (float64, outward): the
    greatest value of the lane's grid at or below the value or the least at or above it. Returns
    `(holds, decided)`: an enclosure that straddles a value of the grid, its ends not two neighbours,
    decides nothing, and is refined by the caller."""
    native = np.asarray(native, np.float64)
    down = _rounded_on_lane(low, np.zeros_like(low), fmt)
    down = np.where(down > low, _step(down, fmt, -np.inf), down)
    up = _rounded_on_lane(high, np.zeros_like(high), fmt)
    up = np.where(up < high, _step(up, fmt, np.inf), up)
    exact = (low == high) & (down == up)
    decided = exact | (_step(down, fmt, np.inf) == up)
    holds = _same(native, down) | _same(native, up) | (np.isnan(native) & np.isnan(low))
    return holds, decided


def _outward(end: fractions.Fraction, direction: float) -> float:
    """The float64 nearest an exact end, moved one step `direction` where rounding took it inward, and
    infinite past float64's range."""
    try:
        value = float(end)
    except OverflowError:
        return math.copysign(math.inf, end)
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
    named = np.isnan(value) | (value <= 0) | (value == np.inf)
    return np.select([value < 0, value == 0, value == np.inf], [np.nan, -np.inf, np.inf], np.nan), named


def _exp_special(value):
    return np.select([value == np.inf, value == -np.inf], [np.inf, 0.0], np.nan), ~np.isfinite(value)


def _erfinv_special(value):
    return np.where(np.abs(value) == 1, np.copysign(np.inf, value), np.nan), ~(np.abs(value) < 1)


def _transcendental(enclose, special):
    """A faithful contract: where `special(value)` names an exact result (a limit, a pole, outside the
    domain), that result; elsewhere faithful to `enclose(values, bits)`, refined at rising precision
    wherever it decides nothing."""
    def contract(params, operands, result, fmt):
        (value,) = _floats(operands)
        expected, named = special(value)
        holds = _same(result, expected)
        rest = np.flatnonzero(~named)
        for bits in (96, 192, 384, 768):
            if not rest.size:
                break
            low, high = enclose(value.flat[rest], bits)
            held, decided = _faithful(np.asarray(result, np.float64).flat[rest], low, high, fmt)
            holds.flat[rest[decided]] = held[decided]
            rest = rest[~decided]
        if rest.size:
            raise ArithmeticError(f"undecided at 768 bits at {rest[:4]}")
        return holds
    return contract


def _terms_hold(native, terms, fmt: Format) -> np.ndarray:
    """Whether each native sum of `terms` (exact float64 values, summed along the last axis) is within
    the bound of any order of summing them in `fmt`, fused multiply-adds included: γ_n Σ|terms| (Higham,
    Accuracy and Stability of Numerical Algorithms, 2002, §3.1: n terms, at most n - 1 roundings on any
    path), plus a flushed partial's least normal at each rounding, around the terms' float64-rounded
    exact sum (`exact_sum`, within 2^-53 of it relative to Σ|terms|, in the allowance). An empty sum is
    the additive identity, +0."""
    native = np.asarray(native, np.float64)
    count = terms.shape[-1]
    if count == 0:
        return _same(native, np.zeros_like(native))
    magnitude = magnitude_sum(terms, -1)
    allowance = (gamma(count, fmt) + 2.0 ** -52) * magnitude + count * 2.0 ** fmt.emin
    error = np.abs(native - exact_sum(terms, -1))
    return np.isfinite(error) & (error <= np.nextafter(allowance, np.inf))


def _floats(operands):
    return [_read(value) for value in operands]


def _wrapping(operator):
    """An integer operation of the result's width, wrapping as two's complement does: computed modulo
    2^64 and narrowed, which is exact modulo the narrower width too."""
    def integral(result, *operands):
        wide = [np.asarray(value).astype(np.uint64) for value in operands]
        with np.errstate(over="ignore"):
            return np.asarray(operator(*wide)).astype(np.asarray(result).dtype)
    return integral


def _arithmetic(build, integral):
    """A correctly rounded contract (`build(*operands)` the exact result as a double-double), or an
    integer one where the result is an integer (`integral`, bit for bit)."""
    def contract(params, operands, result, fmt):
        if fmt is None:
            return _bits(result) == _bits(integral(result, *operands))
        return _same(result, _rounded_on_lane(*build(*_floats(operands)), fmt))
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
        return result == (_read(value) != 0)
    if source is not None:
        limits = np.iinfo(result.dtype)
        read = np.trunc(_read(value))
        expected = np.where(np.isnan(read), 0, np.clip(read, limits.min, limits.max))
        return result == np.array([int(v) for v in expected.flat], dtype=object).reshape(result.shape)
    return result == value.astype(result.dtype)


def _reduced_sum(params, operands, result, fmt):
    (terms,) = operands
    axes = tuple(params["axes"])
    if fmt is None:
        summed = _wrapping(lambda t: np.sum(t, axis=axes, dtype=np.uint64))
        return _bits(result) == _bits(summed(result, terms))
    moved = np.moveaxis(_read(terms), axes, tuple(range(-len(axes), 0)))
    kept = moved.shape[:moved.ndim - len(axes)]
    count = int(np.prod(moved.shape[len(kept):], dtype=np.int64))
    return _terms_hold(result, moved.reshape(*kept, count), fmt)


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
    products = paired(_read(left), _read(right))
    native = np.asarray(result, np.float64).reshape(products.shape[:-1])
    return _terms_hold(native, products, fmt).reshape(result.shape)


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
        expected = np.asarray(oracle(params, *(_read(value) for value in operands)))
        return np.asarray(result) == expected.astype(np.asarray(result).dtype)
    return contract


def _extreme(choose):
    """max or min: the chosen value of the operands as the lane reads them (`_read`), a zero of either
    sign where both read as zero."""
    def contract(params, operands, result, fmt):
        if fmt is None:
            return np.asarray(result) == choose(*(np.asarray(value) for value in operands))
        a, b = _floats(operands)
        expected = choose(a, b)
        zeros = (a == 0) & (b == 0) & (np.asarray(result, np.float64) == 0)
        return _same(result, expected) | zeros
    return contract


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


def _lowest(dtype):
    return -np.inf if np.dtype(dtype).kind == "f" or _format(dtype) else np.iinfo(dtype).min


def _highest(dtype):
    return np.inf if np.dtype(dtype).kind == "f" or _format(dtype) else np.iinfo(dtype).max


CONTRACTS = {
    "add": _arithmetic(two_sum, _wrapping(np.add)),
    "sub": _arithmetic(lambda a, b: two_sum(a, -b), _wrapping(np.subtract)),
    "mul": _arithmetic(_product, _wrapping(np.multiply)),
    "square": _arithmetic(lambda a: _product(a, a), _wrapping(lambda a: a * a)),
    "div": _arithmetic(_quotient, None),
    "sqrt": _arithmetic(_root, None),
    "neg": _moved(lambda params, a: -a),
    "abs": _moved(lambda params, a: np.abs(a)),
    "max": _extreme(np.maximum),
    "min": _extreme(np.minimum),
    "convert_element_type": _converted,
    "log": _transcendental(_interval(lambda iv, x: iv.log(x)), _log_special),
    "exp": _transcendental(_interval(lambda iv, x: iv.exp(x)), _exp_special),
    "erf_inv": _transcendental(_erfinv_enclosed, _erfinv_special),
    "reduce_sum": _reduced_sum,
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
    "reduce_max": _compared(lambda params, a: np.max(a, axis=tuple(params["axes"]),
                                                     initial=_lowest(a.dtype))),
    "reduce_min": _compared(lambda params, a: np.min(a, axis=tuple(params["axes"]),
                                                     initial=_highest(a.dtype))),
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
    "random_bits": _random_bits,
    "random_wrap": _wrapped,
    "random_unwrap": _unwrapped,
}
"""Each primitive's contract, `contract(params, operands, result, fmt) -> ok`, the result's elements
that hold it, `fmt` the result's float format (None for an integer or boolean result): a correctly
rounded result (IEEE arithmetic, as the lane rounds: `FLUSHES`), a faithful one (a transcendental,
within its two neighbours of the exact value whatever its vendor documents), a sum within its bound in
any order, an integer operation wrapped to its width, or a result with no rounding at all, bit for bit
(a move of data, a sign operation, a comparison). A primitive missing here cannot be certified, which
fails the site's coverage, as does an integer result of `div` or `sqrt`."""


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
            elif body is not None:
                results = _run(*body, operands, findings, at + "/")
            else:
                results = eqn.primitive.bind(*operands, **eqn.params)
                results = list(results) if eqn.primitive.multiple_results else [results]
                _held(eqn, operands, results, findings, at)
        env.update(zip(eqn.outvars, results, strict=True))
    return [read(var) for var in jaxpr.outvars]


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
