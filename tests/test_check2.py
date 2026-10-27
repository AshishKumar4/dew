"""Check 2 as a gate (tests/certified.py `certified`, `assert_certified`): a computation run one
primitive at a time, natively, each equation held to its own operation's contract, as the lane
computes (`lane`, measured on it), and a primitive without a contract refused."""

import certified as ct
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import op_certificates as oc
import pytest


@pytest.mark.parametrize("dtype, subnormal, halves", [
    (np.float32, 2.0 ** -140, 2.0 ** -70), (ml_dtypes.bfloat16, 2.0 ** -130, 2.0 ** -66)])
def test_the_lane_flushes_operands_and_results_alike_however_it_compiles(dtype, subnormal, halves):
    """In float32 and in bfloat16, the lane reads a subnormal operand as zero exactly where it flushes a
    subnormal product (of two `halves`) and where a comparison reads it as zero, jitted and eager, with
    YNNPACK's fusions and without: the one fact of subnormals its measure takes (`Lane.flushes`), which
    every contract reads."""
    from reference_error import without_ynn

    flushes, tiny, half = ct.lane().flushes, dtype(subnormal), dtype(halves)
    for run in (jax.jit, lambda fn: fn, without_ynn):
        assert (float(run(lambda x: jnp.multiply(x, dtype(2.0 ** 100)))(tiny)) == 0.0) == flushes
        assert (float(run(jnp.multiply)(half, half)) == 0.0) == flushes
        assert bool(run(jnp.equal)(tiny, dtype(0))) == flushes


@pytest.mark.parametrize("name, fn, operands", [
    ("add", jnp.add, 2), ("sub", jnp.subtract, 2), ("mul", jnp.multiply, 2), ("div", jnp.divide, 2),
    ("sqrt", jnp.sqrt, 1), ("log", jnp.log, 1), ("exp", jnp.exp, 1)])
def test_each_contract_holds_the_lanes_own_operation_and_refuses_a_step_off_it(name, fn, operands):
    """The lane's own result of each operation on random float32 operands, subnormal products and
    quotients included, holds its contract at the lane's measured reach; the same result moved past
    that reach at an element (twice the reach's steps, past both neighbours of the exact value and every
    step the reach allows beyond them) breaks it there and nowhere else, and one step off breaks a
    correctly rounded operation's."""
    rng = np.random.default_rng(1)
    values = [np.float32(rng.uniform(0.01, 4, 256) * rng.choice([1, 2.0 ** -70], 256))
              for _ in range(operands)]
    if name in ("sqrt", "log"):
        values = [np.abs(values[0])]
    result = np.asarray(jax.jit(fn)(*values))
    contract = ct.CONTRACTS[name]
    assert np.all(contract({}, values, result, oc.FLOAT32))
    reach = ct.lane().reach[name, "float32"]
    stepped = result.copy()
    stepped[3] = np.nextafter(stepped[3], np.float32(np.inf)) if stepped[3] else np.float32(1e-30)
    for _ in range(2 * max(reach, 1)):
        stepped[7] = np.nextafter(stepped[7], np.float32(-np.inf))
    held = contract({}, values, stepped, oc.FLOAT32)
    assert not held[7] and np.all(np.delete(held, [3, 7]))
    if reach == 0:
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


def test_an_equation_past_the_lanes_reach_is_a_finding_by_its_path():
    """A division by a literal, which XLA may compute as a product with the literal's rounded reciprocal
    (ledger 1402), is held to the lane's own division's reach: it is a finding, named by its equation,
    exactly where its results' levels pass that reach; a division by a value the program is given is the
    lane's own division and holds."""
    x = np.float32(np.random.default_rng(4).uniform(0.1, 3, 512))
    outputs, findings = ct.certified(lambda x: x / 3, x)
    levels = ct.LEVELS["div"]([x, np.float32(3)], outputs, oc.FLOAT32, ct.lane().flushes)
    past = int(np.count_nonzero(levels > ct.lane().reach["div", "float32"]))
    assert len(findings) == (1 if past else 0)
    named = f"div#0: {past} of 512 elements break the contract"
    assert all(finding.startswith(named) for finding in findings)
    assert ct.certified(jnp.divide, x, np.float32(np.full(512, 3)))[1] == []


def test_a_sum_loses_a_subnormal_only_where_the_lane_flushes_it():
    """A sum of one subnormal returned as zero holds its contract where the lane flushes (its operand
    reads as zero) and breaks it where the lane keeps subnormals, whose sums lose nothing to underflow;
    the lane's own sum holds either way."""
    terms, params = np.float32([2.0 ** -140]), {"axes": (0,)}
    lost = ct.CONTRACTS["reduce_sum"](params, [terms], np.float32(0), oc.FLOAT32)
    assert bool(lost) == ct.lane().flushes
    assert ct.CONTRACTS["reduce_sum"](params, [terms], np.asarray(jnp.sum(terms)), oc.FLOAT32)


def test_the_lane_detects_tininess_after_rounding_and_keeps_subnormals_in_moves_and_casts():
    """A product exactly 2^-126 - 2^-172 rounds to the least normal, at an unbounded exponent and on the
    gradual grid alike, and the lane keeps it (tininess after rounding); a subnormal product holds the
    contract as the lane rounds it, zero where it flushes and the subnormal where it keeps them, and the
    other breaks it; a reshape, a negation and a cast between bf16 and float32 keep a subnormal; a
    float32 past int32's range saturates and NaN casts to 0."""
    a, b = np.float32(1 - 2.0 ** -23), np.float32(2.0 ** -126 + 2.0 ** -149)
    native = np.asarray(jax.jit(jnp.multiply)(a, b))
    assert float(native) == 2.0 ** -126
    assert np.all(ct.CONTRACTS["mul"]({}, [a, b], native, oc.FLOAT32))
    assert not np.all(ct.CONTRACTS["mul"]({}, [a, b], np.float32(0), oc.FLOAT32))
    tiny_product = [np.float32(2.0 ** -70)] * 2
    flushed, kept = np.float32(0), np.float32(2.0 ** -140)
    own, other = (flushed, kept) if ct.lane().flushes else (kept, flushed)
    assert np.all(ct.CONTRACTS["mul"]({}, tiny_product, own, oc.FLOAT32))
    assert not np.any(ct.CONTRACTS["mul"]({}, tiny_product, other, oc.FLOAT32))
    tiny = np.float32([2.0 ** -140, -(2.0 ** -130)])
    assert np.array_equal(np.asarray(jax.jit(lambda x: -x.reshape(2, 1))(tiny)).ravel(), -tiny)
    assert np.all(ct.CONTRACTS["neg"]({}, [tiny], -tiny, oc.FLOAT32))
    narrow = np.array([2.0 ** -130], ml_dtypes.bfloat16)
    assert float(jax.jit(lambda x: x.astype(jnp.float32))(narrow)[0]) == 2.0 ** -130
    assert float(jax.jit(lambda x: x.astype(jnp.bfloat16))(np.float32(2.0 ** -130))) == 2.0 ** -130
    saturated = jax.jit(lambda x: x.astype(jnp.int32))(np.float32([2.0 ** 31, -(2.0 ** 32), np.nan]))
    assert np.array_equal(saturated, [2 ** 31 - 1, -(2 ** 31), 0])
    beyond = np.float32([2.0 ** 31, -(2.0 ** 32), np.nan, 2.7])
    assert ct.certified(lambda x: x.astype(jnp.int32), beyond)[1] == []


@pytest.mark.parametrize("flushes", [True, False])
def test_a_results_level_counts_its_grid_steps_past_the_exact_values_neighbours(flushes):
    """exp(-2^-25) lies between 1 - 2^-24 and 1: both are level 1 (faithful) and 1 - 2^-23, a step
    below, is level 2, on either grid; exp(-100)'s neighbours are zero and the least normal on the
    flushing grid, and two subnormals on the gradual one, where zero is a step past for each subnormal
    below them; a special at its limit is level 0 and at any other value no reach admits it."""
    exp = ct.LEVELS["exp"]
    x = np.float32([-(2.0 ** -25)] * 3)
    around = np.float32([1 - 2.0 ** -24, 1.0, 1 - 2.0 ** -23])
    assert exp([x], around, oc.FLOAT32, flushes).tolist() == [1, 1, 2]
    deep = exp([np.float32([-100.0])], np.float32([0.0]), oc.FLOAT32, flushes)
    below = int(np.exp(-100.0) / 2.0 ** -149)
    assert deep.tolist() == ([1] if flushes else [1 + below])
    if flushes:
        assert exp([np.float32([-100.0])], np.float32([2.0 ** -126]), oc.FLOAT32, flushes).tolist() == [1]
    specials = np.float32([np.inf, -np.inf, np.nan])
    assert exp([specials], np.float32([np.inf, 0.0, np.nan]), oc.FLOAT32, flushes).tolist() == [0, 0, 0]
    assert np.all(exp([specials], np.float32([0.0, np.inf, 1.0]), oc.FLOAT32, flushes) == ct.UNREACHED)
    limits = np.float32([-np.inf, np.inf, np.nan])
    log = ct.LEVELS["log"]([np.float32([0.0, np.inf, -1.0])], limits, oc.FLOAT32, flushes)
    assert log.tolist() == [0, 0, 0]
    poles = np.float32([np.inf, -np.inf])
    assert ct.LEVELS["erf_inv"]([np.float32([1.0, -1.0])], poles, oc.FLOAT32, flushes).tolist() == [0, 0]


@pytest.mark.parametrize("flushes", [True, False])
@pytest.mark.parametrize("fmt", [oc.FLOAT32, oc.BFLOAT16], ids=["float32", "bfloat16"])
def test_a_zero_of_the_wrong_sign_or_a_missed_exact_value_is_no_rounding(flushes, fmt):
    """exp(-1000) is positive and under the least value past zero: +0 is level 1, and -0 no rounding of
    it; sqrt(-0) is -0 exactly and +0 no rounding of it; exp(±0) is 1 and log(1) is +0 exactly, so a step
    of the grid off either, which a measured reach would otherwise admit, is no rounding either."""
    dtype = ct.VALUES[fmt.name]

    def at(values):
        return np.asarray(values, np.float64).astype(dtype)

    exp, log, sqrt = ct.LEVELS["exp"], ct.LEVELS["log"], ct.LEVELS["sqrt"]
    assert exp([at([-1000.0, -1000.0])], at([0.0, -0.0]), fmt, flushes).tolist() == [1, ct.UNREACHED]
    assert sqrt([at([-0.0, -0.0])], at([-0.0, 0.0]), fmt, flushes).tolist() == [0, ct.UNREACHED]
    above_one = float(np.nextafter(dtype(1), dtype(2)))
    ones = exp([at([0.0, -0.0, 0.0])], at([1.0, 1.0, above_one]), fmt, flushes)
    assert ones.tolist() == [0, 0, ct.UNREACHED]
    least = 2.0 ** fmt.emin if flushes else 2.0 ** (fmt.emin - fmt.precision + 1)
    assert log([at([1.0, 1.0])], at([0.0, least]), fmt, flushes).tolist() == [0, ct.UNREACHED]


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


def test_a_high_level_primitive_is_certified_through_its_lowering_and_is_finite_is_exact():
    """A `jax.experimental.hijax` primitive (Dew's reverse-mode seam, `generation.linear_map`) is run as
    it lowers, one primitive at a time, in a vjp too; `is_finite` is exact, held to the value as stored,
    and a false result where the value is finite breaks it."""
    from dew.nn.kernels.generation import linear_map

    x = np.float32(np.linspace(0.5, 2, 8))

    def mapped(x):
        return linear_map(lambda scale, moving: moving * scale, lambda scale, cotangent: cotangent * scale,
                          np.float32(1.5), x)

    outputs, findings = ct.certified(mapped, x)
    assert findings == [] and np.array_equal(outputs, x * np.float32(1.5))
    _, findings = ct.certified(lambda x: jax.vjp(mapped, x)[1](x)[0], x)
    assert findings == []
    _, findings = ct.certified(lambda x: linear_map(lambda scale, moving: jnp.sin(moving) * scale,
                                                    lambda scale, cotangent: cotangent, np.float32(1), x), x)
    assert findings and "sin" in findings[0]
    values = np.float32([1.0, np.inf, -np.inf, np.nan, 1e-45, 0.0])
    assert np.all(ct.CONTRACTS["is_finite"]({}, [values], np.isfinite(values), None))
    assert not np.all(ct.CONTRACTS["is_finite"]({}, [values], np.zeros(6, bool), None))


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


def test_an_exp_past_its_range_is_held_at_its_neighbours_however_far():
    """Past `EXP_RANGE` every exp lies between the same two neighbours of the grid: a finite fp32 as far
    as ±1e30 or ±3e38 certifies as exp(±100) does, with no enclosure of its exponent, and the largest
    finite value is as faithful above as inf, and below the least value past zero on the grid (the least
    normal where the lane flushes, the least subnormal where it does not) as zero."""
    far = np.float32([1e30, -1e30, 3e38, -3e38, 1000.0, -1000.0])
    assert ct.certified(jnp.exp, far)[1] == []
    biggest = np.finfo(np.float32).max
    for flushes, least in ((True, np.finfo(np.float32).tiny), (False, np.float32(2.0 ** -149))):
        for native in (np.float32([np.inf, 0.0] * 3), np.float32([biggest, least] * 3)):
            assert ct.LEVELS["exp"]([far], native, oc.FLOAT32, flushes).tolist() == [1] * 6
    assert not np.any(ct.CONTRACTS["exp"]({}, [far], np.float32([1.0] * 6), oc.FLOAT32))
