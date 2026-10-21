"""Check 2 at the elements of float64 truth exactly zero (tests/zero_elements.py) on computations whose
answer is known: what its own rounding admits, what a defect past it fails, and what it does not
certify (the specification's fixtures, numbered as there)."""

import fractions

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from zero_elements import FLOAT64, Interpreter, _widened_sum, assert_zero_elements, zero_elements

TWO_24 = np.float32(2.0 ** 24)


def _sequential(values):
    """A sum taken in its written order, one rounding an addition."""
    def summed(x):
        total = x[0]
        for index in range(1, x.shape[0]):
            total = total + x[index]
        return total
    return summed, (jnp.asarray(values, jnp.float32),)


def _fraction(value) -> fractions.Fraction:
    """An mpmath number exactly, its sign included (its mantissa is unsigned)."""
    mantissa, exponent = value.man_exp
    magnitude = fractions.Fraction(mantissa) * fractions.Fraction(2) ** exponent
    return -magnitude if value < 0 else magnitude


def _interpreted(function, *inputs):
    interpreter = Interpreter()
    [out] = interpreter.run(jax.make_jaxpr(function)(*inputs), *inputs)
    return out, interpreter.blocked


def test_1_a_cancellation_is_held_to_its_own_rounding_and_a_defect_past_it_fails():
    """s, the fp32 sum of 2^24, 1, -2^24 and -1 in that order, is -1 where float64's is 0; s·s is 1
    against a truth of 0, within the rounding of the first addition carried through the product. The
    same program with 2^30 added fails: its exact result is 2^30."""
    summed, inputs = _sequential([TWO_24, 1.0, -TWO_24, -1.0])

    def squared(x):
        s = summed(x)
        return {"f": s * s}

    assert float(squared(*inputs)["f"]) == 1.0
    coverage = assert_zero_elements(squared, squared, inputs, "cancelled")
    assert coverage.certified == 1

    def defective(x):
        s = summed(x)
        return {"f": s * s + jnp.float32(2.0 ** 30)}

    with pytest.raises(AssertionError):
        assert_zero_elements(defective, squared, inputs, "defect")


def _one_key(dew_extra=None):
    """Attention on given scores where every query keeps one key and the third key no query sees: each
    kept key's weight exp(0) = 1 and every other exp(-inf) = 0, so the softmax is exact and the third
    key's value gradient a structural zero."""
    keep = np.array([0, 1, 3, 0])
    visible = jnp.asarray(np.arange(4)[None, :] == keep[:, None])

    def attention(scores, d_out):
        scores = jnp.where(visible, scores, -jnp.inf)
        weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
        p = weights / jnp.sum(weights, axis=-1, keepdims=True)
        d_v = p.T @ d_out
        if dew_extra is not None:
            d_v = d_v + dew_extra
        return {"d_v": d_v}

    rng = np.random.default_rng(0)
    inputs = tuple(jnp.asarray(rng.normal(size=shape), jnp.float32) for shape in ((4, 4), (4, 8)))
    return attention, inputs


def test_2_a_key_no_query_sees_has_a_value_gradient_of_exactly_zero():
    """The third key's value gradient is zero by structure: zero passes, and 1e-30 added there fails."""
    attention, inputs = _one_key()
    coverage = assert_zero_elements(attention, attention, inputs, "masked")
    assert coverage.certified == 8 and coverage.uncertified == 0
    injected = jnp.zeros((4, 8), jnp.float32).at[2].set(1e-30)
    faulty, _ = _one_key(injected)
    with pytest.raises(AssertionError):
        assert_zero_elements(faulty, attention, inputs, "masked, injected")


def _single_key_backward(d_from="own", injected=None):
    """Attention's query gradient where every query keeps one key: its scores given, so its softmax is
    exact; D summed from the output as FlashAttention does (dO·O, O = P·V rounded), or from another
    row's (`d_from="next"`), and `injected` added to dQ. The reference sums D from P·dP, as XLA's
    softmax VJP does, so its dQ is exactly zero: P·(dP - D) is zero where P is one and D is dP."""
    keep = np.array([0, 2, 1, 3])
    visible = jnp.asarray(np.arange(4)[None, :] == keep[:, None])

    def probabilities(scores):
        scores = jnp.where(visible, scores, -jnp.inf)
        weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
        return weights / jnp.sum(weights, axis=-1, keepdims=True)

    def dew(scores, k, v, d_out):
        p = probabilities(scores)
        d_p = d_out @ v.T
        delta = jnp.sum(d_out * (p @ v), axis=-1, keepdims=True)
        if d_from == "next":
            delta = jnp.roll(delta, 1, axis=0)
        d_q = (p * (d_p - delta)) @ k * jnp.float32(0.125)
        return {"d_q": d_q if injected is None else d_q + injected}

    def reference(scores, k, v, d_out):
        p = probabilities(scores)
        d_p = d_out @ v.T
        delta = jnp.sum(p * d_p, axis=-1, keepdims=True)
        return {"d_q": (p * (d_p - delta)) @ k * 0.125}

    rng = np.random.default_rng(3)
    shapes = ((4, 4), (4, 16), (4, 16), (4, 16))
    return dew, reference, tuple(jnp.asarray(rng.normal(size=shape), jnp.float32) for shape in shapes)


def test_3_a_single_key_query_gradient_with_flashattention_s_d_passes_on_its_own_bound():
    """dQ of a query that keeps one key is zero in float64 and rounding natively, D from the rounded
    output: held to Dew's own bound it passes; 1e-3 added there fails."""
    dew, reference, inputs = _single_key_backward()
    coverage = assert_zero_elements(dew, reference, inputs, "one key")
    assert coverage.certified == 64 and coverage.uncertified == 0
    faulty, _, _ = _single_key_backward(injected=jnp.full((4, 16), 1e-3, jnp.float32))
    with pytest.raises(AssertionError):
        assert_zero_elements(faulty, reference, inputs, "one key, injected")


def test_13_d_from_another_row_fails_past_its_own_bound():
    dew, reference, inputs = _single_key_backward(d_from="next")
    with pytest.raises(AssertionError):
        assert_zero_elements(dew, reference, inputs, "another row's D")


def test_4_a_selection_its_predicate_does_not_decide_is_not_certified():
    """where(s > 0, 1, 0) at s = -1 natively, 0 exactly, its bound reaching past zero: undecided, so
    not certified, and nothing fails."""
    summed, inputs = _sequential([TWO_24, 1.0, -TWO_24, -1.0])

    def chosen(x):
        return {"y": jnp.where(summed(x) > 0, jnp.float32(1.0), jnp.float32(0.0))}

    coverage = zero_elements(chosen, chosen, inputs, "undecided")
    assert (coverage.certified, coverage.uncertified, coverage.over) == (0, 1, [])


def test_5_a_primitive_without_a_contract_is_named_and_its_values_not_certified():
    def cancelled(x):
        return {"y": jax.lax.erf(x) - jax.lax.erf(x)}

    coverage = zero_elements(cancelled, cancelled, (jnp.ones(3, jnp.float32),), "erf")
    assert coverage.certified == 0 and coverage.uncertified == 3
    assert "erf" in coverage.blocked


def test_6_a_slot_routing_drops_is_exactly_zero_and_a_least_normal_there_fails():
    """A slot whose score is decided below its threshold carries an exact zero; the least normal added
    there fails (a smaller subnormal a flushing backend would read as zero)."""
    scores = jnp.asarray([0.9, 0.1, 0.7], jnp.float32)

    def routed(score, value, extra=0.0):
        return {"y": jnp.where(score > 0.5, value, jnp.float32(0.0)) + jnp.asarray(extra, jnp.float32)}

    values = jnp.asarray([2.0, 3.0, 4.0], jnp.float32)
    assert_zero_elements(routed, routed, (scores, values), "dropped")

    def faulty(score, value):
        return routed(score, value, np.asarray([0.0, 2.0 ** -126, 0.0], np.float32))

    with pytest.raises(AssertionError):
        assert_zero_elements(faulty, routed, (scores, values), "dropped, injected")


def test_7_a_maximum_takes_its_operands_bound():
    """max(s, 0) for s the fp32 sum of -2^24, -1, 2^24 and 0.5: 0.5 natively, 0 in float64, within
    the bound s carries."""
    summed, inputs = _sequential([-TWO_24, -1.0, TWO_24, 0.5])

    def clipped(x):
        return {"y": jnp.maximum(summed(x), jnp.float32(0.0))}

    assert float(clipped(*inputs)["y"]) == 0.5
    assert assert_zero_elements(clipped, clipped, inputs, "clipped").certified == 1


def test_8_an_underflow_is_held_to_the_least_normal_and_a_defect_past_it_fails():
    """t·0.5 + t·0.5 - t at t = 2^-149 is -t under gradual underflow and 0 where the backend flushes,
    within its rounding either way; 2^-100 added fails."""
    t = (jnp.asarray([2.0 ** -149], jnp.float32),)

    def halves(x):
        return {"y": (x * 0.5) + (x * 0.5) - x}

    assert_zero_elements(halves, halves, t, "underflow")

    def defective(x):
        return {"y": (x * 0.5) + (x * 0.5) - x + jnp.float32(2.0 ** -100)}

    with pytest.raises(AssertionError):
        assert_zero_elements(defective, halves, t, "underflow, defect")


def test_9_a_square_counts_its_rounding_and_erf_has_no_contract():
    x = jnp.asarray([1 + 2.0 ** -12], jnp.float32)
    out, _ = _interpreted(lambda v: v ** 2, x)
    exact = fractions.Fraction(float(x[0])) ** 2
    assert out.certified.all()
    assert fractions.Fraction(float(out.error[0])) >= abs(fractions.Fraction(float(out.value[0])) - exact) > 0
    erf, blocked = _interpreted(jax.lax.erf, x)
    assert not erf.certified.any() and "erf" in blocked


def test_10_a_comparison_of_exact_operands_is_decided():
    out, _ = _interpreted(lambda v: v > 0, jnp.zeros(2, jnp.float32))
    assert out.certified.all() and out.exact.all()


def test_11_an_injected_subnormal_on_an_unused_slot_fails_where_the_backend_keeps_it():
    """x·0 plus 2^-149 at one slot: an exact zero and an exact subnormal, which fails against a truth of
    zero wherever the backend holds it; a backend that flushes it holds zero, which passes."""
    x = (jnp.ones(3, jnp.float32),)

    def clean(v):
        return {"y": v * 0}

    def injected(v):
        return {"y": v * 0 + jnp.asarray([0.0, 2.0 ** -149, 0.0], jnp.float32)}

    if float(injected(*x)["y"][1]) != 0.0:
        with pytest.raises(AssertionError):
            assert_zero_elements(injected, clean, x, "subnormal")
    else:
        assert_zero_elements(injected, clean, x, "subnormal, flushed")


def test_12_a_bound_sum_encloses_the_exact_sum():
    terms = np.array([1.0, *([2.0 ** -54] * 32767)])
    exact = sum(fractions.Fraction(term) for term in terms)
    assert fractions.Fraction(float(_widened_sum(np.sum(terms), terms.size, terms.size))) >= exact


def test_15_an_element_its_native_output_does_not_witness_is_not_certified():
    """An interpretation certifies only the execution it reproduces: a function traced to x·0 that then
    runs as x·0 - 1 is not certified, and does not fail."""
    calls = []

    def unlike(v):
        calls.append(None)
        return {"y": v * 0 if len(calls) == 1 else v * 0 - 1}

    coverage = zero_elements(unlike, lambda v: {"y": v * 0}, (jnp.ones(2, jnp.float32),), "unwitnessed")
    assert (coverage.certified, coverage.uncertified, coverage.over) == (0, 2, [])


def test_16_a_one_product_dot_counts_its_products_rounding():
    a = jnp.asarray([1 + 2.0 ** -12], jnp.float32)
    out, _ = _interpreted(jnp.dot, a, a)
    assert out.certified.all() and float(out.error) >= 2.0 ** -24


def test_the_truths_own_rounding_is_propagated_at_float64():
    """τ: the float64 truth of 1e16 + 1 - 1e16 is 0 against an exact 1, and its bound covers that."""
    with jax.enable_x64(new_val=True):
        inputs = (jnp.asarray([1e16, 1.0, -1e16], jnp.float64),)
        out, _ = _interpreted(lambda x: (x[0] + x[1]) + x[2], *inputs)
    assert FLOAT64.precision == 53
    assert out.certified.all() and float(out.error) >= 1.0


def _float64(function, *values):
    with jax.enable_x64(new_val=True):
        inputs = tuple(jnp.asarray(value, jnp.float64) for value in values)
        return _interpreted(function, *inputs)


def test_a_bound_survives_underflow_in_products_quotients_and_dots():
    """Exactness is never read off an underflowed residual, and a positive bound never rounds to zero:
    2^-540·2^-540, z/2^500 and z·2^-500 for z a cancellation's inexact zero all keep a positive bound."""
    product, _ = _float64(lambda a, b: a * b, [2.0 ** -540], [2.0 ** -540])
    assert product.certified.all() and not product.exact.any()
    assert fractions.Fraction(float(product.error[0])) >= fractions.Fraction(2) ** -1080

    def cancelled(x):
        return (x[0] + x[1]) - x[0]

    inputs = ([2.0 ** -600, 2.0 ** -1000],)
    quotient, _ = _float64(lambda x: cancelled(x) / 2.0 ** 500, *inputs)
    dotted, _ = _float64(lambda x: jnp.dot(jnp.stack([cancelled(x)]), jnp.asarray([2.0 ** -500])), *inputs)
    # A subnormal bound fed to the bound dot, which XLA may read as zero.
    def scaled_back(x):
        return jnp.dot(jnp.stack([cancelled(x) / 2.0 ** 500]), jnp.asarray([2.0 ** 500]))

    flushed, _ = _float64(scaled_back, [2.0 ** -600, 2.0 ** -654])
    # A subnormal input amplified past the normal range, which XLA:CPU's denormal flushing reads as zero:
    # its bound encloses the exact product.
    amplified, _ = _float64(jnp.dot, [2.0 ** -1074], [2.0 ** 1000])
    for out in (quotient, dotted, flushed, amplified):
        assert not out.exact.any() and (out.error > 0).all()
    assert float(flushed.error) >= 2.0 ** -654
    apart = abs(fractions.Fraction(2) ** -74 - fractions.Fraction(float(amplified.value)))
    assert fractions.Fraction(float(amplified.error)) >= apart


def test_a_correctly_rounded_float64_quotient_is_certified():
    quotient, _ = _float64(lambda a, b: a / b, [1.0], [3.0])
    assert quotient.certified.all() and not quotient.exact.any()


def test_a_subnormal_constant_is_held_to_its_size_where_a_backend_may_read_it_as_zero():
    tiny = np.float32(2.0 ** -149)
    constant = jnp.asarray([tiny, 0.0], jnp.float32)
    summed, _ = _interpreted(lambda x: jnp.sum(x + constant), jnp.zeros(2, jnp.float32))
    assert not summed.exact.any() and float(summed.error) >= 2.0 ** -149
    compared, _ = _interpreted(lambda x: x < tiny, jnp.zeros(1, jnp.float32))
    assert not compared.certified.any()


def test_values_without_a_contract_are_not_certified():
    """A complex modulus, a value that is not a number, a sum whose bound overflows, and a dot by an
    algorithm, which may convert its operands (not run: the CPU has none to run)."""
    from zero_elements import _probed_dot

    modulus, _ = _interpreted(jnp.abs, jnp.asarray([1 + 1j], jnp.complex64))
    passed, _ = _interpreted(lambda x: x, jnp.asarray([np.nan], jnp.float32))
    overflowing, _ = _float64(jnp.sum, [2.0 ** 1023, -(2.0 ** 1023)])
    for out in (modulus, passed, overflowing):
        assert not out.certified.any()
    assert not _probed_dot(jax.lax.DotAlgorithmPreset.F16_F16_F16, None, np.float32)
    assert _probed_dot(None, None, np.float32)


def test_exp_is_held_to_a_verified_enclosure_at_every_element_and_carries_its_operands_bound():
    """exp at ordinary fp32 and float64 arguments: certified, its bound at least its distance from the
    exact value (mpmath at 200 bits); past its range, not certified."""
    import mpmath

    for dtype, values in ((jnp.float32, [0.3, -5.0, 20.0]), (jnp.float64, [0.3, -5.0, 20.0])):
        with jax.enable_x64(new_val=dtype == jnp.float64):
            x = jnp.asarray(values, dtype)
            out, _ = _interpreted(jnp.exp, x)
        assert out.certified.all()
        for at in range(len(values)):
            with mpmath.workprec(200):
                exact = _fraction(mpmath.exp(mpmath.mpf(float(x[at]))))
            apart = abs(exact - fractions.Fraction(float(out.value[at])))
            assert fractions.Fraction(float(out.error[at])) >= apart
    overflowing, _ = _interpreted(jnp.exp, jnp.asarray([100.0], jnp.float32))
    assert not overflowing.certified.any()


def test_a_single_key_query_gradient_with_computed_scores_is_certified():
    """The single-key query gradient with its scores computed from q·kᵀ, so exp is taken at a bounded
    argument: certified on its own bound, and FlashAttention's D passes there."""
    keep = np.array([0, 2, 1, 3])
    visible = jnp.asarray(np.arange(4)[None, :] == keep[:, None])

    def probabilities(q, k):
        scores = jnp.where(visible, q @ k.T * jnp.float32(0.25), -jnp.inf)
        weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
        return weights / jnp.sum(weights, axis=-1, keepdims=True)

    def dew(q, k, v, d_out):
        p = probabilities(q, k)
        delta = jnp.sum(d_out * (p @ v), axis=-1, keepdims=True)
        return {"d_q": (p * (d_out @ v.T - delta)) @ k * jnp.float32(0.25)}

    def reference(q, k, v, d_out):
        p = probabilities(q, k)
        d_p = d_out @ v.T
        return {"d_q": (p * (d_p - jnp.sum(p * d_p, axis=-1, keepdims=True))) @ k * 0.25}

    rng = np.random.default_rng(5)
    inputs = tuple(jnp.asarray(rng.normal(size=(4, 16)), jnp.float32) for _ in range(4))
    coverage = assert_zero_elements(dew, reference, inputs, "computed scores")
    assert coverage.certified == 64, coverage


def test_a_function_is_held_to_its_interval_enclosure_over_its_operands_bound():
    """log, log1p, sqrt, rsqrt, sin, cos, tanh and the logistic at bounded arguments: certified, each
    element's bound at least its distance from the exact value at the native argument (mpmath at 200
    bits); log of a negative, not certified."""
    import mpmath

    cases = {jnp.log: mpmath.log, jnp.log1p: mpmath.log1p, jnp.sqrt: mpmath.sqrt,
             jax.lax.rsqrt: lambda t: 1 / mpmath.sqrt(t), jnp.sin: mpmath.sin, jnp.cos: mpmath.cos,
             jnp.tanh: mpmath.tanh, jax.nn.sigmoid: lambda t: 1 / (1 + mpmath.exp(-t))}
    terms = [np.float32(0.3), np.float32(2.0 ** -25), np.float32(0.4)]
    summed, inputs = _sequential(terms)
    for function, exact_function in cases.items():
        out, _ = _interpreted(lambda x, f=function: f(summed(x)), *inputs)
        assert out.certified.all(), function
        with mpmath.workprec(200):
            exact = _fraction(exact_function(sum((mpmath.mpf(float(term)) for term in terms), mpmath.mpf(0))))
        apart = abs(exact - fractions.Fraction(float(out.value)))
        assert fractions.Fraction(float(out.error)) >= apart, function
    negative, _ = _interpreted(jnp.log, jnp.asarray([-1.0], jnp.float32))
    assert not negative.certified.any()


def test_a_scan_carries_its_bound_through_every_step_in_its_order():
    """A running fp32 sum by `scan`, forward and reversed: certified, its final carry's bound at least
    its distance from the exact sum, and each stacked step's its own."""
    x = jnp.asarray([1.0, 2.0 ** -24, 2.0 ** -24, -1.0], jnp.float32)
    exact = sum(fractions.Fraction(float(v)) for v in x)
    for reverse in (False, True):
        def running(v, reverse=reverse):
            return jax.lax.scan(lambda c, t: (c + t, c), jnp.float32(0.0), v, reverse=reverse)

        interpreter = Interpreter()
        total, steps = interpreter.run(jax.make_jaxpr(running)(x), x)
        assert total.certified.all() and steps.certified.all()
        assert steps.value.shape == (4,)
        assert fractions.Fraction(float(total.error)) >= abs(exact - fractions.Fraction(float(total.value)))


def test_a_loop_whose_count_rounding_decides_is_not_certified_and_a_counted_one_is():
    """s, a cancellation that is -1 natively and 0 exactly: `while s < 0: s += 1` runs once natively and
    not at all exactly, its first predicate undecided, so nothing is certified; a `fori_loop` counted by
    an exact integer is."""
    summed, inputs = _sequential([TWO_24, 1.0, -TWO_24, -1.0])

    def counted_by_rounding(x):
        return jax.lax.while_loop(lambda s: s < 0, lambda s: s + 1, summed(x))

    out, blocked = _interpreted(counted_by_rounding, *inputs)
    assert not out.certified.any() and any(name.startswith("while") for name in blocked)

    def counted(x):
        return jax.lax.fori_loop(0, 3, lambda i, s: s + x[i], jnp.float32(0.0))

    out, _ = _interpreted(counted, *inputs)
    assert out.certified.all()


def test_a_branch_rounding_decides_is_not_certified_and_a_decided_one_is():
    """`cond(s > -0.5, ...)` with s the cancellation above takes the other branch natively than exactly:
    undecided, so not certified; on an exact predicate it is."""
    summed, inputs = _sequential([TWO_24, 1.0, -TWO_24, -1.0])

    def flipped(x):
        s = summed(x)
        return jax.lax.cond(s > -0.5, lambda: s * 2, lambda: s * 3)

    out, blocked = _interpreted(flipped, *inputs)
    assert not out.certified.any() and any(name.startswith("cond") for name in blocked)

    def decided(x):
        return jax.lax.cond(x[0] > 0, lambda: x[1] * 2, lambda: x[1] * 3)

    out, _ = _interpreted(decided, *inputs)
    assert out.certified.all() and out.exact.all()
