"""Check 2 as a gate (tests/certified.py `certified`, `assert_certified`): a computation run one
primitive at a time, natively, each equation held to its own operation's contract, as the CPU lane
computes (its subnormals flushed), and a primitive without a contract refused."""

import certified as ct
import jax
import jax.numpy as jnp
import numpy as np
import op_certificates as oc
import pytest

SUBNORMAL = np.float32(2.0 ** -140)


def test_the_lane_flushes_subnormal_operands_and_results():
    """XLA on the CPU lane reads a subnormal operand as zero and flushes a subnormal result, jitted and
    eager, with YNNPACK's fusions and without (`FLUSHES`, which every contract reads)."""
    from reference_error import without_ynn

    for run in (jax.jit, lambda fn: fn, without_ynn):
        assert float(run(lambda x: jnp.multiply(x, np.float32(2.0 ** 100)))(SUBNORMAL)) == 0.0
        assert float(run(jnp.multiply)(np.float32(2.0 ** -70), np.float32(2.0 ** -70))) == 0.0
    assert ct.FLUSHES


@pytest.mark.parametrize("name, fn, operands", [
    ("add", jnp.add, 2), ("sub", jnp.subtract, 2), ("mul", jnp.multiply, 2), ("div", jnp.divide, 2),
    ("sqrt", jnp.sqrt, 1), ("log", jnp.log, 1), ("exp", jnp.exp, 1)])
def test_each_contract_holds_the_lanes_own_operation_and_refuses_a_step_off_it(name, fn, operands):
    """The lane's own result of each operation on random float32 operands, subnormal products and
    quotients included, holds its contract; the same result one step of float32 off at an element
    breaks it there and nowhere else."""
    rng = np.random.default_rng(1)
    values = [np.float32(rng.uniform(0.01, 4, 256) * rng.choice([1, 2.0 ** -70], 256))
              for _ in range(operands)]
    if name in ("sqrt", "log"):
        values = [np.abs(values[0])]
    result = np.asarray(jax.jit(fn)(*values))
    contract = ct.CONTRACTS[name]
    assert np.all(contract({}, values, result, oc.FLOAT32))
    stepped = result.copy()
    stepped[3] = np.nextafter(stepped[3], np.float32(np.inf)) if stepped[3] else np.float32(1e-30)
    stepped[7] = np.nextafter(np.nextafter(stepped[7], np.float32(-np.inf)), np.float32(-np.inf))
    held = contract({}, values, stepped, oc.FLOAT32)
    assert not held[7] and np.all(np.delete(held, [3, 7]))
    if name not in ("log", "exp"):
        assert not held[3]


def test_a_sum_holds_in_any_order_and_a_lost_term_breaks_it():
    """A float32 sum taken left to right, pairwise or the lane's own way holds the sum's bound; a sum
    missing one term of a cancelling row breaks it."""
    rng = np.random.default_rng(2)
    terms = np.float32(rng.standard_normal((64, 512)))
    terms[0] = np.float32([1.0, 2.0 ** -12, -1.0, *([0.0] * 509)])
    params = {"axes": (1,)}
    sequential = np.array([np.float32(0) + np.cumsum(row, dtype=np.float32)[-1] for row in terms])
    for native in (np.asarray(jnp.sum(jnp.asarray(terms), 1)), sequential,
                   np.float32(terms.reshape(64, 2, 256).sum(2, dtype=np.float32).sum(1, dtype=np.float32))):
        assert np.all(ct.CONTRACTS["reduce_sum"](params, [terms], native, oc.FLOAT32))
    lost = sequential.copy()
    lost[0] = np.float32(0.0)
    assert not ct.CONTRACTS["reduce_sum"](params, [terms], lost, oc.FLOAT32)[0]


def test_a_dot_holds_its_bound_and_an_element_twice_past_it_breaks_it():
    rng = np.random.default_rng(3)
    a, b = np.float32(rng.standard_normal((5, 300))), np.float32(rng.standard_normal((300, 4)))
    params = {"dimension_numbers": (((1,), (0,)), ((), ()))}
    native = np.asarray(jax.jit(lambda a, b: jnp.dot(a, b, precision="highest"))(a, b))
    assert np.all(ct.CONTRACTS["dot_general"](params, [a, b], native, oc.FLOAT32))
    off = native.copy()
    off[1, 2] += 2 * oc.gamma(300, oc.FLOAT32) * float(np.abs(a[1]) @ np.abs(b[:, 2]))
    held = ct.CONTRACTS["dot_general"](params, [a, b], off, oc.FLOAT32)
    assert not held[1, 2] and held.sum() == held.size - 1


def test_a_computation_is_certified_through_its_calls_and_a_primitive_without_a_contract_refused():
    """`certified` runs a jitted call's and a cond's bodies one primitive at a time, and finds the
    primitive no contract covers, by its path."""
    def computed(x, flag):
        inner = jax.jit(lambda x: jnp.log(x * x + 1))(x)
        return jax.lax.cond(flag, lambda v: v * 2, lambda v: v - 1, inner)

    x = np.float32(np.linspace(0.1, 3, 32))
    outputs, findings = ct.certified(computed, x, np.bool_(1))
    assert findings == []
    # The op-by-op run's outputs are the operations' own: x·x, + 1, log, · 2, each the lane's.
    stepped = np.asarray(jnp.multiply(jnp.log(jnp.add(jnp.multiply(x, x), np.float32(1))), np.float32(2)))
    np.testing.assert_array_equal(outputs, stepped)
    _, findings = ct.certified(lambda x: jnp.sin(x), x)
    assert len(findings) == 1 and "no contract for sin" in findings[0]
    with pytest.raises(AssertionError):
        ct.assert_certified(lambda x: jnp.sin(x), lambda seed: (x,), "sine", seeds=1)


def test_the_lanes_unfaithful_operations_are_findings_with_their_paths():
    """A transcendental more than its two neighbours from the exact value breaks its contract whatever
    its vendor documents (erf_inv as the lane computes it, inside `jax.random.normal`), and so does a
    division by a constant, which XLA computes as a product with the constant's reciprocal, not the
    correctly rounded quotient; a division by a value the program is given is correctly rounded."""
    rng = np.random.default_rng(4)
    u = np.float32(rng.uniform(-0.99, 0.99, 512))
    _, findings = ct.certified(jax.lax.erf_inv, u)
    assert len(findings) == 1 and findings[0].startswith("erf_inv#0")
    x = np.float32(rng.uniform(0.1, 3, 512))
    _, findings = ct.certified(lambda x: x / 3, x)
    assert len(findings) == 1 and findings[0].startswith("div#0")
    assert ct.certified(jnp.divide, x, np.float32(np.full(512, 3)))[1] == []


def test_the_lane_detects_tininess_after_rounding_and_keeps_subnormals_in_moves_and_casts():
    """A product exactly 2^-126 - 2^-172 rounds to the least normal at an unbounded exponent, and the lane
    keeps it (tininess after rounding); a reshape, a negation and a cast between bf16 and float32 keep a
    subnormal; a comparison reads it as zero; a float32 past int32's range saturates and NaN casts to 0."""
    import ml_dtypes

    a, b = np.float32(1 - 2.0 ** -23), np.float32(2.0 ** -126 + 2.0 ** -149)
    native = np.asarray(jax.jit(jnp.multiply)(a, b))
    assert float(native) == 2.0 ** -126
    assert np.all(ct.CONTRACTS["mul"]({}, [a, b], native, oc.FLOAT32))
    assert not np.all(ct.CONTRACTS["mul"]({}, [a, b], np.float32(0), oc.FLOAT32))
    flushed = [np.float32(2.0 ** -70)] * 2
    assert not np.all(ct.CONTRACTS["mul"]({}, flushed, np.float32(2.0 ** -140), oc.FLOAT32))
    tiny = np.float32([2.0 ** -140, -(2.0 ** -130)])
    assert np.array_equal(np.asarray(jax.jit(lambda x: -x.reshape(2, 1))(tiny)).ravel(), -tiny)
    assert np.all(ct.CONTRACTS["neg"]({}, [tiny], -tiny, oc.FLOAT32))
    narrow = np.array([2.0 ** -130], ml_dtypes.bfloat16)
    assert float(jax.jit(lambda x: x.astype(jnp.float32))(narrow)[0]) == 2.0 ** -130
    assert float(jax.jit(lambda x: x.astype(jnp.bfloat16))(np.float32(2.0 ** -130))) == 2.0 ** -130
    assert bool(jax.jit(jnp.equal)(np.float32(2.0 ** -140), np.float32(0)))
    saturated = jax.jit(lambda x: x.astype(jnp.int32))(np.float32([2.0 ** 31, -(2.0 ** 32), np.nan]))
    assert np.array_equal(saturated, [2 ** 31 - 1, -(2 ** 31), 0])
    beyond = np.float32([2.0 ** 31, -(2.0 ** 32), np.nan, 2.7])
    assert ct.certified(lambda x: x.astype(jnp.int32), beyond)[1] == []


def test_a_faithful_result_is_one_of_its_two_neighbours_on_the_lanes_grid():
    """exp(-2^-25) lies between 1 - 2^-24 and 1: both are faithful, and 1 - 2^-23, the value below the
    next binade's spacing, is not; a tiny exact value's neighbours on the flushing grid are zero and the
    least normal; specials are their limits."""
    x = np.float32([-(2.0 ** -25)])
    assert np.all(ct.CONTRACTS["exp"]({}, [x], np.float32([1 - 2.0 ** -24]), oc.FLOAT32))
    assert np.all(ct.CONTRACTS["exp"]({}, [x], np.float32([1.0]), oc.FLOAT32))
    assert not np.any(ct.CONTRACTS["exp"]({}, [x], np.float32([1 - 2.0 ** -23]), oc.FLOAT32))
    deep = np.float32([-100.0])
    assert np.all(ct.CONTRACTS["exp"]({}, [deep], np.float32([0.0]), oc.FLOAT32))
    assert np.all(ct.CONTRACTS["exp"]({}, [deep], np.float32([2.0 ** -126]), oc.FLOAT32))
    specials = np.float32([np.inf, -np.inf, np.nan])
    assert np.all(ct.CONTRACTS["exp"]({}, [specials], np.float32([np.inf, 0.0, np.nan]), oc.FLOAT32))
    limits = np.float32([-np.inf, np.inf, np.nan])
    assert np.all(ct.CONTRACTS["log"]({}, [np.float32([0.0, np.inf, -1.0])], limits, oc.FLOAT32))
    poles = np.float32([np.inf, -np.inf])
    assert np.all(ct.CONTRACTS["erf_inv"]({}, [np.float32([1.0, -1.0])], poles, oc.FLOAT32))


def test_each_equation_runs_under_its_own_context():
    """Random bits drawn under a local threefry setting other than the caller's are held to the bits
    that setting draws."""
    key = jax.random.key(3)

    def drawn(key):
        with jax.threefry_partitionable(not jax.config.jax_threefry_partitionable):
            return jax.random.bits(key, (64,))

    outputs, findings = ct.certified(drawn, key)
    assert findings == []
    assert np.array_equal(outputs, jax.jit(drawn)(key))


def test_moves_keep_their_bits_and_integer_operations_wrap():
    """A bitcast of a NaN keeps its payload, and one with another payload breaks; a reshape by a
    permutation, a logical shift of a negative integer, a narrowing bitcast's trailing axis, integer
    sums that wrap and an empty sum are each their own results."""
    payload = np.uint32([0x7FC00001])
    native = np.asarray(jax.lax.bitcast_convert_type(payload, jnp.float32))
    params = {"new_dtype": np.float32}
    assert np.all(ct.CONTRACTS["bitcast_convert_type"](params, [payload], native, oc.FLOAT32))
    other = np.uint32([0x7FC00002]).view(np.float32)
    assert not np.any(ct.CONTRACTS["bitcast_convert_type"](params, [payload], other, oc.FLOAT32))
    ints = np.int32([[-1, 2 ** 31 - 1], [5, -7]])
    for fn in (lambda a: jax.lax.reshape(a, (4,), dimensions=(1, 0)),
               lambda a: jax.lax.shift_right_logical(a, np.int32(1)),
               lambda a: jax.lax.bitcast_convert_type(a, jnp.int16),
               lambda a: jnp.sum(a * a + a, axis=0),
               lambda a: jnp.sum(a.astype(jnp.float32)[:, :0], axis=1)):
        assert ct.certified(fn, ints)[1] == [], fn


def test_calls_are_certified_inside_and_unsupported_types_refused():
    """A rematerialised body is run one primitive at a time; a float16, float64-free complex or float8
    value has no contract of ours, and a key's wrapping is held to its data."""
    x = np.float32(np.linspace(1, 2, 8))
    _, findings = ct.certified(jax.checkpoint(lambda x: jnp.log(x) * x), x)
    assert findings == []
    _, findings = ct.certified(jax.checkpoint(lambda x: jnp.sin(x)), x)
    assert findings and "remat" in findings[0] and "sin" in findings[0]
    for dtype in (jnp.float16, jnp.complex64, jnp.float8_e4m3fn):
        _, findings = ct.certified(lambda x, dtype=dtype: x.astype(dtype).reshape(2, 4), x)
        assert findings and "no format" in findings[0], dtype
    data = np.uint32([0, 7])
    assert ct.certified(lambda data: jax.random.key_data(jax.random.wrap_key_data(data)), data)[1] == []


def test_extrema_order_zeros_and_propagate_nan_as_the_lane_does():
    """max and min take -0 below +0 and propagate NaN, alone and reduced; a flipped zero breaks them;
    boolean reductions, a scalar log, erf_inv's zero and an overflowing exp are each their own results."""
    zeros = np.float32([0.0, -0.0, -0.0])
    others = np.float32([-0.0, 0.0, -0.0])
    for name, fn in (("max", jnp.maximum), ("min", jnp.minimum)):
        native = np.asarray(jax.jit(fn)(zeros, others))
        assert np.all(ct.CONTRACTS[name]({}, [zeros, others], native, oc.FLOAT32))
        assert not np.any(ct.CONTRACTS[name]({}, [zeros, others], -native, oc.FLOAT32))
    rows = np.float32([[-0.0, 0.0], [1.0, np.nan], [-0.0, -0.0]])
    for fn in (lambda a: jnp.max(a, 1), lambda a: jnp.min(a, 1), lambda a: jnp.any(a > 0, 1),
               lambda a: jnp.all(a > 0, 1)):
        assert ct.certified(fn, rows)[1] == [], fn
    assert ct.certified(jnp.log, np.float32(2.0))[1] == []
    assert ct.certified(jax.lax.erf_inv, np.float32([0.0, -0.0, 2.0 ** -140]))[1] == []
    assert not np.any(ct.CONTRACTS["erf_inv"]({}, [np.float32([0.0])], np.float32([2.0 ** -126]), oc.FLOAT32))
    assert ct.certified(jnp.exp, np.float32([1000.0, 88.0, 89.0]))[1] == []
