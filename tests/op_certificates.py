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
"""Whether the platform's float arithmetic flushes: XLA on the CPU lane treats a subnormal operand as
zero and flushes a subnormal result to zero (FTZ and DAZ), jitted or eager, with YNNPACK's fusions or
without (tests/test_op_certificates.py holds it, so a JAX that changes it fails there first). Every
contract reads operands and results so; gradual underflow is IEEE's, not this lane's."""

STRUCTURAL = frozenset({"jit", "pjit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call",
                        "custom_vjp_call_jaxpr", "remat", "checkpoint", "cond"})
"""Primitives that only call a jaxpr of their own, which `certified` runs equation by equation in
place of binding them (so a call's body is never compiled fused)."""


def _format(dtype) -> Format | None:
    return FORMATS.get(np.dtype(dtype).name)


def _host(value):
    """A value on the host: an array as NumPy's, a typed key as it is (its bits are its data)."""
    import jax

    if jax.dtypes.issubdtype(getattr(value, "dtype", np.float32), jax.dtypes.prng_key):
        return value
    return np.asarray(value)


def _daz(value, fmt: Format) -> np.ndarray:
    """`value` (float64) with every subnormal of `fmt` read as a zero of its sign, as the lane reads it."""
    value = np.asarray(value, np.float64)
    if not FLUSHES:
        return value
    return np.where(np.abs(value) < 2.0 ** fmt.emin, np.copysign(0.0, value), value)


def _same(native, expected) -> np.ndarray:
    """Bit for bit, element by element, zeros by sign and NaN by NaN."""
    native, expected = np.asarray(native, np.float64), np.asarray(expected, np.float64)
    return ((native == expected) & (np.signbit(native) == np.signbit(expected))) | (
        np.isnan(native) & np.isnan(expected))


def _flushed(native, rounded_value, fmt: Format) -> np.ndarray:
    """Whether `native` is `rounded_value` (a result rounded with gradual underflow) as the lane holds
    it: itself where it is normal, and where it is below the least normal or rounds onto it (a tiny
    result, which the lane may detect before rounding) itself or a zero of its sign."""
    native = np.asarray(native, np.float64)
    exact = _same(native, rounded_value)
    if not FLUSHES:
        return exact
    tiny = np.abs(rounded_value) <= 2.0 ** fmt.emin
    return exact | (tiny & (native == 0) & (np.signbit(native) == np.signbit(rounded_value)))


def _correctly_rounded(native, hi, lo, fmt: Format) -> np.ndarray:
    return _flushed(native, rounded(hi, lo, fmt), fmt)


def _faithful(native, low, high, fmt: Format) -> np.ndarray:
    """Whether `native` is one of the two values of `fmt` around an exact value enclosed in [low, high]
    (float64, outward), or the value itself where it is one: below the enclosure's least neighbour or
    above its greatest is unfaithful. A zero of either sign is faithful to a tiny value."""
    native = np.asarray(native, np.float64)
    down = rounded(low, np.zeros_like(low), fmt)
    down = np.where(down > low, down - _quantum(np.abs(down), fmt), down)
    up = rounded(high, np.zeros_like(high), fmt)
    up = np.where(up < high, up + _quantum(np.abs(up), fmt), up)
    spacing = _quantum(np.maximum(np.abs(down), np.abs(up)), fmt)
    inside = (native >= down) & (native <= up) & (up - down <= 2 * spacing)
    tiny = FLUSHES & (np.minimum(np.abs(low), np.abs(high)) <= 2.0 ** fmt.emin) & (native == 0)
    return inside | tiny | (np.isnan(native) & np.isnan(low))


def _outward(end: fractions.Fraction, direction: float) -> float:
    """The float64 nearest an exact end, moved one step `direction` where rounding took it inward."""
    value = float(end)
    inward = fractions.Fraction(value) > end if direction < 0 else fractions.Fraction(value) < end
    return math.nextafter(value, direction) if inward else value


def _enclosed(function, values, bits: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """An outward float64 enclosure of `function` (an mpmath interval function) at each value."""
    low, high = np.empty(values.shape), np.empty(values.shape)
    with interval_precision(bits) as iv:
        for index, value in enumerate(values.flat):
            if not np.isfinite(value):
                low.flat[index] = high.flat[index] = np.nan
                continue
            a, b = interval_ends(function(iv, iv.mpf(float(value))), bits)
            low.flat[index] = _outward(a, -math.inf)
            high.flat[index] = _outward(b, math.inf)
    return low, high


def _erfinv_enclosed(values, bits: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """An outward enclosure of erfinv at each value in (-1, 1): mpmath's erfinv at `bits`, widened by
    2^-(bits - 24) of itself, verified at its two ends by erfc at `bits` + 64 bits, 2^40 times finer
    than the width (erfinv is odd and erfc decreasing: erfc(b) < 1 - |x| < erfc(a) for |x|'s ends)."""
    import mpmath

    low, high = np.empty(values.shape), np.empty(values.shape)
    for index, value in enumerate(values.flat):
        if not -1 < value < 1:
            low.flat[index] = high.flat[index] = np.copysign(np.inf, value) if abs(value) == 1 else np.nan
            continue
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


def _sum_holds(native, terms, axis, fmt: Format) -> np.ndarray:
    """Whether each native sum of `terms` (exact float64 values) along `axis` is within the bound of any
    order of summing them in `fmt`, fused multiply-adds included: γ_n Σ|terms| (Higham, Accuracy and
    Stability, 2002, §3.1, for n terms, n - 1 roundings at most along any path), plus a flushed partial
    sum's 2^emin at each of the n roundings, around the terms' float64-rounded exact sum (`exact_sum`,
    within 2^-53 of it relative to Σ|terms|, added to the allowance)."""
    count = terms.shape[axis]
    magnitude = magnitude_sum(terms, axis)
    allowance = (gamma(max(count, 1), fmt) + 2.0 ** -52) * magnitude + count * 2.0 ** fmt.emin
    error = np.abs(np.asarray(native, np.float64) - exact_sum(terms, axis))
    return np.isfinite(error) & (error <= np.nextafter(allowance, np.inf))


def _read(value) -> np.ndarray:
    """An operand as float64, as the lane reads it (`_daz`) where it is a float of a known format."""
    fmt = _format(np.asarray(value).dtype)
    return np.asarray(value, np.float64) if fmt is None else _daz(np.asarray(value, np.float64), fmt)


def _floats(operands, fmt):
    return [_daz(np.asarray(value, np.float64), fmt) for value in operands]


def _rn(build):
    """A correctly rounded contract: `build(*operands)` the exact result as a double-double."""
    def contract(params, operands, result, fmt):
        hi, lo = build(*_floats(operands, fmt))
        return _correctly_rounded(result, hi, lo, fmt)
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


def _converted(params, operands, result, fmt):
    """A cast: correctly rounded to a float format, and to an integer or boolean as NumPy casts it
    (truncating a float, nonzero to true), bit for bit."""
    (value,) = operands
    if fmt is None:
        return np.asarray(result) == _read(value).astype(np.asarray(result).dtype)
    source = _format(np.asarray(value).dtype)
    value = np.asarray(value, np.float64)
    if source is not None:
        value = _daz(value, source)
    return _correctly_rounded(result, value, np.zeros_like(value), fmt)


def _transcendental(enclose):
    def contract(params, operands, result, fmt):
        (value,) = _floats(operands, fmt)
        return _faithful(result, *enclose(value), fmt)
    return contract


def _reduced_sum(params, operands, result, fmt):
    (terms,) = _floats(operands, fmt)
    axes = tuple(params["axes"])
    moved = np.moveaxis(terms, axes, tuple(range(-len(axes), 0)))
    flat = moved.reshape(*moved.shape[:moved.ndim - len(axes)], -1)
    return _sum_holds(result, flat, -1, fmt)


def _dot(params, operands, result, fmt):
    """A dot_general's products, exact in float64 for operands no wider than fp32, summed (`_sum_holds`)
    over the contracting dimensions in any order."""
    (contract_l, contract_r), (batch_l, batch_r) = params["dimension_numbers"]
    left, right = (_read(value) for value in operands)
    free_l = [axis for axis in range(left.ndim) if axis not in contract_l and axis not in batch_l]
    free_r = [axis for axis in range(right.ndim) if axis not in contract_r and axis not in batch_r]
    left = np.transpose(left, [*batch_l, *free_l, *contract_l])
    right = np.transpose(right, [*batch_r, *free_r, *contract_r])
    batch = left.shape[:len(batch_l)]
    width = int(np.prod(left.shape[len(batch) + len(free_l):], dtype=np.int64))
    left = left.reshape(*batch, -1, 1, width)
    right = right.reshape(*batch, 1, -1, width)
    products = left * right
    native = np.asarray(result, np.float64).reshape(products.shape[:-1])
    return _sum_holds(native, products, -1, fmt).reshape(np.shape(result))


def _exactly(oracle):
    """A contract with no rounding: the result is `oracle(params, operands)` bit for bit (a comparison,
    a selection, a move of data, an integer operation), floats read as the lane reads them."""
    def contract(params, operands, result, fmt):
        read = [_read(value).astype(np.asarray(value).dtype) if _format(np.asarray(value).dtype) else
                np.asarray(value) for value in operands]
        expected = np.asarray(oracle(params, *read))
        native = np.asarray(result)
        if native.dtype.kind in "biu" or expected.dtype.kind in "biu":
            return np.broadcast_to(native == expected.astype(native.dtype), native.shape)
        return _same(native, expected)
    return contract


def _selected(params, which, *cases):
    return np.choose(np.asarray(which).astype(np.int64), cases)


def _broadcast(params, value):
    shape, dimensions = params["shape"], params["broadcast_dimensions"]
    expanded = np.reshape(value, [value.shape[dimensions.index(axis)] if axis in dimensions else 1
                                  for axis in range(len(shape))])
    return np.broadcast_to(expanded, shape)


def _random_bits(params, operands, result, fmt):
    """Random bits are integers, whatever program computes them: the bits a separate `jax.random.bits`
    of the same key draws, bit for bit."""
    import jax

    (key,) = operands
    dtype = np.dtype(f"uint{params['bit_width']}")
    expected = jax.jit(lambda key: jax.random.bits(key, params["shape"], dtype))(key)
    return np.asarray(result) == np.asarray(expected)


def _comparison(operator):
    def compared(params, a, b):
        return operator(*(np.asarray(value, np.float64) if _format(value.dtype) else value
                          for value in (a, b)))
    return _exactly(compared)


CONTRACTS = {
    "add": _rn(two_sum),
    "sub": _rn(lambda a, b: two_sum(a, -b)),
    "mul": _rn(_product),
    "square": _rn(lambda a: _product(a, a)),
    "div": _rn(_quotient),
    "sqrt": _rn(_root),
    "neg": _exactly(lambda params, a: -a),
    "abs": _exactly(lambda params, a: np.abs(a)),
    "max": _exactly(lambda params, a, b: np.maximum(a, b)),
    "min": _exactly(lambda params, a, b: np.minimum(a, b)),
    "convert_element_type": _converted,
    "log": _transcendental(lambda value: _enclosed(lambda iv, x: iv.log(x), value)),
    "exp": _transcendental(lambda value: _enclosed(lambda iv, x: iv.exp(x), value)),
    "erf_inv": _transcendental(_erfinv_enclosed),
    "reduce_sum": _reduced_sum,
    "dot_general": _dot,
    "eq": _comparison(np.equal),
    "ne": _comparison(np.not_equal),
    "lt": _comparison(np.less),
    "le": _comparison(np.less_equal),
    "gt": _comparison(np.greater),
    "ge": _comparison(np.greater_equal),
    "and": _exactly(lambda params, a, b: np.bitwise_and(a, b)),
    "or": _exactly(lambda params, a, b: np.bitwise_or(a, b)),
    "not": _exactly(lambda params, a: np.invert(a)),
    "shift_right_logical": _exactly(lambda params, a, b: np.right_shift(a, b)),
    "shift_left": _exactly(lambda params, a, b: np.left_shift(a, b)),
    "bitcast_convert_type": _exactly(lambda params, a: a.view(params["new_dtype"])),
    "reduce_and": _exactly(lambda params, a: np.all(a, axis=tuple(params["axes"]))),
    "reduce_or": _exactly(lambda params, a: np.any(a, axis=tuple(params["axes"]))),
    "reduce_max": _exactly(lambda params, a: np.max(a, axis=tuple(params["axes"]))),
    "reduce_min": _exactly(lambda params, a: np.min(a, axis=tuple(params["axes"]))),
    "select_n": _exactly(_selected),
    "broadcast_in_dim": _exactly(_broadcast),
    "reshape": _exactly(lambda params, a: np.reshape(a, params["new_sizes"])),
    "squeeze": _exactly(lambda params, a: np.squeeze(a, tuple(params["dimensions"]))),
    "expand_dims": _exactly(lambda params, a: np.expand_dims(a, tuple(params["dimensions"]))),
    "transpose": _exactly(lambda params, a: np.transpose(a, params["permutation"])),
    "copy": _exactly(lambda params, a: a),
    "copy_p": _exactly(lambda params, a: a),
    "random_bits": _random_bits,
    "random_wrap": None,
    "random_unwrap": None,
}
"""Each primitive's contract, `contract(params, operands, result, fmt) -> ok`, the result's elements
that hold it, `fmt` the result's float format (None for an integer or boolean result): a correctly
rounded result (IEEE arithmetic), a faithful one (a transcendental, within its two neighbours of the
exact value whatever its vendor documents), a sum within its bound in any order, or a result with no
rounding at all, bit for bit. A key's wrapping holds no arithmetic (None: it is data, moved). A
primitive missing here cannot be certified, which fails the site's coverage."""


def certified(fn, *arguments) -> tuple:
    """`fn` on `arguments` run one primitive at a time, each executed alone and natively (an eager
    bind, its own compiled program), and each held to its contract (CONTRACTS) on the operands it was
    given: `(outputs, findings)`, the outputs as that run computed them, and every equation whose
    result breaks its contract or that has none, by primitive, with its first element that breaks.
    This certifies the op-by-op program; a fused program's internals are check 1's."""
    import jax

    closed, shape = jax.make_jaxpr(fn, return_shape=True)(*arguments)
    findings: list[str] = []
    outputs = _run(closed.jaxpr, closed.consts, jax.tree.leaves(arguments), findings)
    return jax.tree.unflatten(jax.tree.structure(shape), outputs), findings


def _run(jaxpr, consts, arguments, findings, where: str = "") -> list:
    from jax.extend import core

    env = {}

    def read(var):
        return var.val if isinstance(var, core.Literal) else env[var]

    env.update(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, arguments, strict=True))
    for index, eqn in enumerate(jaxpr.eqns):
        operands = [read(var) for var in eqn.invars]
        name = eqn.primitive.name
        at = f"{where}{name}#{index}"
        if name in STRUCTURAL:
            results = _called(eqn, operands, findings, at)
        else:
            results = eqn.primitive.bind(*operands, **eqn.params)
            results = list(results) if eqn.primitive.multiple_results else [results]
            _held(eqn, operands, results, findings, at)
        env.update(zip(eqn.outvars, results, strict=True))
    return [read(var) for var in jaxpr.outvars]


def _called(eqn, operands, findings, at) -> list:
    params = eqn.params
    if eqn.primitive.name == "cond":
        branch = params["branches"][int(np.asarray(operands[0]))]
        return _run(branch.jaxpr, branch.consts, operands[1:], findings, at + "/")
    body = next(params[key] for key in ("jaxpr", "call_jaxpr", "fun_jaxpr") if key in params)
    jaxpr, consts = (body.jaxpr, body.consts) if hasattr(body, "consts") else (body, ())
    return _run(jaxpr, consts, operands, findings, at + "/")


def _held(eqn, operands, results, findings, at) -> None:
    name = eqn.primitive.name
    if name not in CONTRACTS:
        findings.append(f"{at}: uncertifiable, no contract for {name}")
        return
    contract = CONTRACTS[name]
    if contract is None:
        return
    for result in results:
        result = _host(result)
        if not isinstance(result, np.ndarray):
            findings.append(f"{at}: uncertifiable, a key result of {name}")
            return
        fmt = _format(result.dtype)
        if fmt is None and np.asarray(result).dtype.kind == "f":
            findings.append(f"{at}: uncertifiable, no format for {np.asarray(result).dtype}")
            return
        ok = np.asarray(contract(eqn.params, [_host(value) for value in operands], result, fmt))
        if not np.all(ok):
            first = np.unravel_index(int(np.argmin(ok.ravel())), ok.shape) if ok.ndim else ()
            findings.append(f"{at}: {int(ok.size - ok.sum())} of {ok.size} elements break the contract, "
                            f"first at {first}: {np.asarray(result)[first]!r}")


def assert_certified(fn, draw, label: str, *, seeds: int) -> None:
    """Check 2 of the gate: `fn` on `draw(seed)`'s inputs for each of `seeds`, run one primitive at a
    time (`certified`), every element of every equation within its own operation's contract and every
    primitive covered by one."""
    for seed in range(seeds):
        _, findings = certified(fn, *draw(seed))
        assert not findings, f"{label}, seed {seed}: " + "; ".join(findings[:8])
