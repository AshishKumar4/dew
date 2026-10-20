"""The per-operation certificates' exact oracle (tests/op_certificates.py) against exact rational
rounding: random operands, and the operands where a float64 evaluation would round wrongly or a
format's edges decide (ties, a fused multiply-add's low part, subnormal cells, overflow, signed zero)."""

import fractions

import ml_dtypes
import numpy as np
import op_certificates as oc
import pytest

FORMATS = [oc.FLOAT32, oc.BFLOAT16]
Fraction = fractions.Fraction


def _values(rng, count: int, fmt: oc.Format) -> np.ndarray:
    """Values of `fmt` across its range: random signs, exponents from below the least subnormal to
    the largest binade, random significands."""
    exponents = rng.integers(fmt.emin - fmt.precision - 2, fmt.emax, count)
    raw = np.ldexp(rng.random(count) + 1.0, exponents) * rng.choice([-1.0, 1.0], count)
    return oc.exact_cast(raw, fmt)


def _assert_rounds(got, exact: list, fmt: oc.Format):
    """`got` is each exact value rounded to nearest even in `fmt`, bit for bit. An exactly zero
    value takes its sign from each operation's IEEE rule, not from the value, so there only the
    value is compared."""
    expected = np.array([oc.fraction_rounded(value, fmt) for value in exact])
    zero = np.array([value == 0 for value in exact])
    wrong = np.where(zero, got != 0, oc.mismatches(got, expected, fmt))
    assert not wrong.any(), (np.flatnonzero(wrong)[:8], got[wrong][:8], expected[wrong][:8])


def _exact(values) -> list:
    return [Fraction(float(value)) for value in values]


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name)
def test_each_operation_rounds_its_exact_result_on_random_operands(fmt):
    rng = np.random.default_rng(0)
    a, b, c = (_values(rng, 20000, fmt) for _ in range(3))
    # Operands near one another's negation, so sums cancel.
    b[::3] = oc.exact_cast(-a[::3] * (1 + rng.random(len(a[::3])) * 2.0 ** -6), fmt)
    x, y, z = _exact(a), _exact(b), _exact(c)
    _assert_rounds(oc.exact_add(a, b, fmt), [p + q for p, q in zip(x, y, strict=True)], fmt)
    _assert_rounds(oc.exact_mul(a, b, fmt), [p * q for p, q in zip(x, y, strict=True)], fmt)
    _assert_rounds(oc.exact_fma(a, b, c, fmt), [p * q + r for p, q, r in zip(x, y, z, strict=True)], fmt)
    kept = b != 0
    _assert_rounds(oc.exact_div(a[kept], b[kept], fmt),
                   [p / q for p, q, keep in zip(x, y, kept, strict=True) if keep], fmt)


def test_a_fused_multiply_add_off_a_midpoint_rounds_where_float64_does_not():
    """a·b sits on an fp32 midpoint whose lower neighbour is even; c moves the exact value off it,
    which float64's own a·b + c loses."""
    a = b = np.array([1 + 2.0 ** -12])
    c = np.array([2.0 ** -80])
    product = Fraction(float(a[0])) ** 2
    for sign in (1, -1):
        exact = product + sign * Fraction(float(c[0]))
        _assert_rounds(oc.exact_fma(a, b, sign * c, oc.FLOAT32), [exact], oc.FLOAT32)
    above = oc.fraction_rounded(product + Fraction(float(c[0])), oc.FLOAT32)
    assert np.float32(a[0] * b[0] + c[0]) != np.float32(above)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name)
def test_ties_round_to_even_and_a_low_part_breaks_them(fmt):
    rng = np.random.default_rng(1)
    base = np.abs(_values(rng, 4000, fmt))
    base = base[(base > 2.0 ** (fmt.emin + 1)) & (base < fmt.largest / 4)]
    midpoints = base + np.ldexp(1.0, np.frexp(base)[1] - fmt.precision) / 2
    for sign in (1.0, -1.0):
        hi = sign * midpoints
        # A low part well inside half a float64 ulp of hi, of either sign, or none.
        for low_sign in (0.0, 1.0, -1.0):
            lows = low_sign * np.abs(hi) * 2.0 ** -60
            exact = [p + q for p, q in zip(_exact(hi), _exact(lows), strict=True)]
            _assert_rounds(oc.rounded(hi, lows, fmt), exact, fmt)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name)
def test_subnormal_cells_overflow_and_signed_zero(fmt):
    least = 2.0 ** (fmt.emin - fmt.precision + 1)
    edge = fmt.largest + 2.0 ** (fmt.emax - fmt.precision)
    cases = [(least / 2, least * 2.0 ** -40), (-least / 2, 0.0), (3 * least / 2, 0.0), (least / 4, 0.0),
             (-least / 4, 0.0), (2.0 ** fmt.emin * 0.75, 0.0), (edge, 0.0), (-edge, 0.0),
             (edge, -edge * 2.0 ** -60), (np.nextafter(edge, 0), 0.0), (fmt.largest, 0.0)]
    hi, lo = (np.array(column) for column in zip(*cases, strict=True))
    got = oc.rounded(hi, lo, fmt)
    _assert_rounds(got, [p + q for p, q in zip(_exact(hi), _exact(lo), strict=True)], fmt)
    assert np.signbit(got[4]) and not np.signbit(got[3])
    zeros = oc.rounded(np.array([0.0, -0.0]), np.zeros(2), fmt)
    assert not np.signbit(zeros[0]) and np.signbit(zeros[1])


def test_a_cast_matches_the_format_libraries():
    """float64 to fp32 by numpy and fp32 to bf16 by ml_dtypes, each rounding to nearest even once."""
    rng = np.random.default_rng(2)
    raw = np.ldexp(rng.random(50000) + 1, rng.integers(-150, 127, 50000)) * rng.choice([-1.0, 1.0], 50000)
    single = raw.astype(np.float32)
    assert not oc.mismatches(oc.exact_cast(raw, oc.FLOAT32), single.astype(np.float64), oc.FLOAT32).any()
    brain = single.astype(ml_dtypes.bfloat16).astype(np.float64)
    assert not oc.mismatches(oc.exact_cast(single, oc.BFLOAT16), brain, oc.BFLOAT16).any()


def test_a_sum_bound_holds_its_own_worst_case():
    """Sequential fp32 summation of n positive terms stays within γ(n-1)·Σ|t| of the exact sum, and
    γ(k) is at least ku/(1-ku)."""
    rng = np.random.default_rng(3)
    terms = rng.random((64, 7168)).astype(np.float32)
    native = np.zeros(64, np.float32)
    for column in terms.T:
        native = native + column
    exact = oc.exact_sum(terms, axis=1)
    bound = oc.gamma(terms.shape[1] - 1, oc.FLOAT32) * oc.magnitude_sum(terms, axis=1)
    assert (np.abs(native.astype(np.float64) - exact) <= bound).all()
    k, u = 7167, Fraction(2) ** -24
    assert Fraction(oc.gamma(k, oc.FLOAT32)) >= k * u / (1 - k * u)


def _interval_rounded(function, values, fmt: oc.Format) -> np.ndarray:
    """RN in `fmt` of `function` (of mpmath's interval context and an interval) at each value, by
    interval arithmetic alone, at rising precision."""
    out = []
    for value in values:
        for precision in (96, 192, 384):
            with oc.interval_precision(precision) as iv:
                ends = function(iv, iv.mpf(float(value)))._mpi_
            lower, upper = (oc.fraction_rounded(oc._fraction(end), fmt) for end in ends)
            if lower == upper:
                out.append(lower)
                break
        else:
            raise AssertionError(value)
    return np.array(out)


def _near_midpoints(values, estimate, fmt: oc.Format, count: int) -> np.ndarray:
    """The `count` values whose estimate lies nearest a midpoint of `fmt`, relative to it."""
    spacing = np.ldexp(1.0, np.maximum(np.frexp(estimate)[1] - 1, fmt.emin) - (fmt.precision - 1))
    offset = np.abs((estimate / spacing) % 1 - 0.5) * spacing / np.abs(estimate)
    return values[np.argsort(offset)[:count]]


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name)
def test_exp_rounds_as_interval_arithmetic_does(fmt):
    """Random arguments over exp's whole range in `fmt`, the ones whose float64 exp lies nearest a
    midpoint (the oracle's interval path), and the edges: underflow to subnormals and zero, overflow."""
    rng = np.random.default_rng(4)
    values = oc.exact_cast(rng.uniform(-110.0, 90.0, 1 << 20), fmt)
    values = values[np.isfinite(values)]
    hard = _near_midpoints(values, np.exp(values), fmt, 64)
    edges = oc.exact_cast(np.array([-103.97, -103.28, -87.34, -87.33, 88.72, 88.73, 0.0, -0.0, 1e-30]), fmt)
    sample = np.concatenate([values[:2000], hard, edges])
    expected = _interval_rounded(lambda iv, x: iv.exp(x), sample, fmt)
    assert not oc.mismatches(oc.exp_rounded(sample, fmt), expected, fmt).any()


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name)
def test_rsqrt_rounds_as_interval_arithmetic_does(fmt):
    rng = np.random.default_rng(5)
    values = np.abs(_values(rng, 1 << 20, fmt))
    values = values[values > 0]
    hard = _near_midpoints(values, 1 / np.sqrt(values), fmt, 64)
    least = 2.0 ** (fmt.emin - fmt.precision + 1)
    sample = np.concatenate([values[:2000], hard, [1.0, 4.0, least]])
    expected = _interval_rounded(lambda iv, x: 1 / iv.sqrt(x), sample, fmt)
    assert not oc.mismatches(oc.rsqrt_rounded(sample, fmt), expected, fmt).any()
