"""Check 2 as a gate (tests/certified.py `certified`, `assert_certified`): a computation run one
primitive at a time, natively, each equation held to its own operation's contract, as the lane
computes (`lane`, measured on it), and a primitive without a contract refused."""

import itertools

import certified as ct
import interval_certificates as ic
import jax
import jax.numpy as jnp
import ml_dtypes
import native_certificates as nc
import numpy as np
import op_certificates as oc
import pytest
import recurrence_witness as rw
from generation_support import INTERPRETS
from jax.ad_checkpoint import checkpoint_name
from jax.extend import core as jcore
from jax.extend.core import primitives

from dew.nn.precision import correctly_rounded_divide_p


def _named(value):
    """A parametrized primitive's id, its own name; any other argument's as pytest names it."""
    return value.name if isinstance(value, jcore.Primitive) else None


def test_an_operation_is_its_primitive_and_another_bearing_its_name_is_refused():
    """Check 2 knows an operation by its primitive object: one of another library's or a user's that bears
    the name of JAX's `add`, and here even computes a sum, has no contract and is refused, where a lookup by
    name certified it by JAX's sum's contract; JAX's own add of the same operands certifies."""
    named = jcore.Primitive("add")
    named.def_impl(lambda a, b: np.add(a, b))
    named.def_abstract_eval(lambda a, b: a)
    operands = np.float32([1.5, 2.25]), np.float32([3.0, -4.5])
    assert len(ct.certified(lambda a, b: named.bind(a, b), *operands)[1]) == 1
    assert ct.certified(jnp.add, *operands)[1] == []


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


@pytest.mark.parametrize("primitive, fn, operands", [
    (jax.lax.add_p, jnp.add, 2), (primitives.add_jaxvals_p, jnp.add, 2), (jax.lax.sub_p, jnp.subtract, 2),
    (jax.lax.mul_p, jnp.multiply, 2), (jax.lax.div_p, jnp.divide, 2), (jax.lax.sqrt_p, jnp.sqrt, 1),
    (jax.lax.rsqrt_p, jax.lax.rsqrt, 1), (jax.lax.log_p, jnp.log, 1), (jax.lax.exp_p, jnp.exp, 1),
    (jax.lax.logistic_p, jax.lax.logistic, 1), (jax.lax.log1p_p, jnp.log1p, 1),
    (jax.lax.expm1_p, jnp.expm1, 1), (jax.lax.sin_p, jnp.sin, 1), (jax.lax.cos_p, jnp.cos, 1)], ids=_named)
def test_each_contract_holds_the_lanes_own_operation_and_refuses_a_step_off_it(primitive, fn, operands):
    """The lane's own result of each operation on random float32 operands, subnormal products and
    quotients included, holds its contract at the lane's measured reach; the same result moved past
    that reach at an element (twice the reach's steps, past both neighbours of the exact value and every
    step the reach allows beyond them) breaks it there and nowhere else, and one step off breaks a
    correctly rounded operation's."""
    rng = np.random.default_rng(1)
    values = [np.float32(rng.uniform(0.01, 4, 256) * rng.choice([1, 2.0 ** -70], 256))
              for _ in range(operands)]
    if primitive in (jax.lax.sqrt_p, jax.lax.rsqrt_p, jax.lax.log_p):
        values = [np.abs(values[0])]
    result = np.asarray(jax.jit(fn)(*values))
    contract = ct.CONTRACTS[primitive]
    assert np.all(contract({}, values, result, oc.FLOAT32))
    # A cotangents' sum is held at add's reach, as it lowers as add does.
    reach = ct.lane().reach[jax.lax.add_p if primitive is primitives.add_jaxvals_p else primitive, "float32"]
    stepped = result.copy()
    stepped[3] = np.nextafter(stepped[3], np.float32(np.inf)) if stepped[3] else np.float32(1e-30)
    for _ in range(2 * max(reach, 1)):
        stepped[7] = np.nextafter(stepped[7], np.float32(-np.inf))
    held = contract({}, values, stepped, oc.FLOAT32)
    assert not held[7] and np.all(np.delete(held, [3, 7]))
    if reach == 0:
        assert not held[3]


@pytest.mark.parametrize("primitive, fn", [(jax.lax.expm1_p, jnp.expm1), (jax.lax.log1p_p, jnp.log1p)],
                         ids=_named)
def test_a_primitive_keeping_a_subnormal_its_lanes_multiply_flushes_is_held_as_it_computes(primitive, fn):
    """A lane flushes per primitive (ledger 4023): XLA's CPU multiply reads a subnormal as zero, its expm1
    and log1p return one as it is. Each holds its contract at subnormal arguments of either sign, as the lane
    computes it, and a result moved a step off it breaks it there."""
    x = np.float32([-(2.0 ** -148), 2.0 ** -140, -(2.0 ** -130), 2.0 ** -127, 2.0 ** -149])
    ct.assert_certified(fn, lambda seed: (x,), primitive, seeds=1)
    result = np.asarray(fn(x))
    stepped = result.copy()
    for _ in range(2 * max(ct.lane().reach[primitive, "float32"], 1)):
        stepped[1] = np.nextafter(stepped[1], np.float32(np.inf))
    held = ct.CONTRACTS[primitive]({}, [x], stepped, oc.FLOAT32)
    assert not held[1] and np.all(np.delete(held, 1))


def test_a_division_by_a_broadcast_divisor_is_held_at_its_own_forms_reach():
    """A division whose divisor is broadcast along a size-1 axis (a row's scale), which XLA may compute as
    a multiply by the divisor's reciprocal, is held at that form's measured reach (`certified.BROADCAST`):
    the lane's own such division holds, its result moved past that reach at an element breaks it there,
    and a same-shape division keeps its own reach."""
    rng = np.random.default_rng(2)
    dividend = np.float32(rng.uniform(0.01, 4, (16, 64)))
    divisor = np.float32(rng.uniform(0.01, 4, (16, 1)))
    result = np.asarray(jax.jit(jnp.divide)(dividend, divisor))
    contract = ct.CONTRACTS[jax.lax.div_p]
    assert np.all(contract({}, [dividend, divisor], result, oc.FLOAT32))
    reach = ct.lane().reach[(jax.lax.div_p, ct.BROADCAST), "float32"]
    stepped = result.copy()
    for _ in range(2 * max(reach, 1)):
        stepped[0, 7] = np.nextafter(stepped[0, 7], np.float32(-np.inf))
    held = contract({}, [dividend, divisor], stepped, oc.FLOAT32)
    assert not held[0, 7] and np.all(np.delete(held.ravel(), 7))
    same = np.asarray(jax.jit(jnp.divide)(dividend, np.broadcast_to(divisor, dividend.shape)))
    assert np.all(contract({}, [dividend, np.broadcast_to(divisor, dividend.shape)], same, oc.FLOAT32))


def test_a_correctly_rounded_division_is_held_at_its_exact_quotient():
    """Dew's correctly rounded division (`dew.nn.precision.correctly_rounded_divide`) is held at its exact
    quotient rounded to nearest, whatever the lane's own division reaches: its result holds, one step off at
    an element breaks it there and nowhere else, and its gradient over a constant divisor, the cotangent over
    that divisor (row 5277), is certified through."""
    from dew.nn import precision

    rng = np.random.default_rng(5277)
    dividend, weights = (np.float32(rng.uniform(-4, 4, 256)) for _ in range(2))
    divisor = np.float32(rng.uniform(0.01, 4, 256))
    result = np.asarray(jax.jit(precision.correctly_rounded_divide)(dividend, divisor))
    contract = ct.CONTRACTS[correctly_rounded_divide_p]
    assert np.all(contract({}, [dividend, divisor], result, oc.FLOAT32))
    stepped = result.copy()
    stepped[7] = np.nextafter(stepped[7], np.float32(np.inf))
    held = contract({}, [dividend, divisor], stepped, oc.FLOAT32)
    assert not held[7] and np.all(np.delete(held, 7))
    ct.assert_certified(jax.grad(lambda x: jnp.sum(precision.divided(x, 6.0) * weights)),
                        lambda seed: (dividend,), "a correctly rounded division's gradient", seeds=1)


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
        assert np.all(ct.CONTRACTS[jax.lax.reduce_sum_p](params, [terms], native, oc.FLOAT32))
    lost = sequential.copy()
    lost[0] = np.float32(0.0)
    assert not ct.CONTRACTS[jax.lax.reduce_sum_p](params, [terms], lost, oc.FLOAT32)[0]


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_a_cumulative_sum_holds_each_prefixs_bound_and_a_lost_term_breaks_it(reverse):
    """A cumulative sum, forward and reversed, of float32 terms over seven binades and of integers that
    wrap: the lane's own holds at every prefix; one prefix missing its largest term breaks there; an integer
    one off breaks."""
    rng = np.random.default_rng(6)
    terms = np.float32(rng.normal(size=(3, 17)) * 2.0 ** rng.integers(-3, 4, (3, 17)))
    native = np.asarray(jax.lax.cumsum(terms, axis=1, reverse=reverse))
    params = {"axis": 1, "reverse": reverse}
    assert np.all(ct.CONTRACTS[jax.lax.cumsum_p](params, [terms], native, oc.FLOAT32))
    taken = np.arange(8, 17) if reverse else np.arange(0, 9)
    lost = native.copy()
    lost[1, 8] -= terms[1, taken[np.argmax(np.abs(terms[1, taken]))]]
    assert not ct.CONTRACTS[jax.lax.cumsum_p](params, [terms], lost, oc.FLOAT32)[1, 8]
    ints = np.int32([[2 ** 31 - 1, 1, 5], [-7, 3, 2]])
    native = np.asarray(jax.lax.cumsum(ints, axis=1, reverse=reverse))
    assert np.all(ct.CONTRACTS[jax.lax.cumsum_p](params, [ints], native, None))
    assert not np.all(ct.CONTRACTS[jax.lax.cumsum_p](params, [ints], native + np.int32(1), None))


def test_a_dot_holds_its_bound_and_an_element_twice_past_it_breaks_it():
    rng = np.random.default_rng(3)
    a, b = np.float32(rng.standard_normal((5, 300))), np.float32(rng.standard_normal((300, 4)))
    params = {"dimension_numbers": (((1,), (0,)), ((), ()))}
    native = np.asarray(jax.jit(lambda a, b: jnp.dot(a, b, precision="highest"))(a, b))
    assert np.all(ct.CONTRACTS[jax.lax.dot_general_p](params, [a, b], native, oc.FLOAT32))
    off = native.copy()
    off[1, 2] += 2 * oc.gamma(300, oc.FLOAT32) * float(np.abs(a[1]) @ np.abs(b[:, 2]))
    held = ct.CONTRACTS[jax.lax.dot_general_p](params, [a, b], off, oc.FLOAT32)
    assert not held[1, 2] and held.sum() == held.size - 1


def test_a_scatter_add_holds_its_bound_and_a_lost_update_breaks_it():
    """A scatter-add of float32 updates, several landing on one element and one dropped out of range: the
    lane's own result holds each element's sum in any order, and an element missing its largest update
    breaks there alone; an integer scatter-add wraps, and one off breaks."""
    rng = np.random.default_rng(7)
    operand = np.float32(rng.standard_normal((6, 3)))
    indices = np.int32([[1], [4], [1], [1], [9]])
    updates = np.float32(rng.standard_normal((5, 3)) * 2.0 ** rng.integers(-6, 6, (5, 3)))

    def added(operand, indices, updates):
        return operand.at[indices[:, 0]].add(updates, mode="drop")

    params, given, (native,) = _first_equation(added, [operand, indices, updates], jax.lax.scatter_add_p)
    native = np.asarray(native)
    contract = ct.CONTRACTS[jax.lax.scatter_add_p]
    assert np.all(contract(params, given, native, oc.FLOAT32))
    lost = native.copy()
    lost[1, 0] -= updates[[0, 2, 3], 0][np.argmax(np.abs(updates[[0, 2, 3], 0]))]
    held = contract(params, given, lost, oc.FLOAT32)
    assert not held[1, 0] and held.sum() == held.size - 1
    ints = [np.int32([[2 ** 31 - 1], [0]]), np.int32([[0], [0]]), np.int32([[1], [5]])]
    params, given, (native,) = _first_equation(lambda o, i, u: o.at[i[:, 0]].add(u), ints,
                                               jax.lax.scatter_add_p)
    assert np.all(contract(params, given, np.asarray(native), None))
    assert not np.all(contract(params, given, np.asarray(native) + np.int32(1), None))


@pytest.mark.parametrize("ragged", ["rows", "contracting"])
def test_a_ragged_dot_holds_each_groups_bound_and_an_element_twice_past_it_breaks_it(ragged):
    """A ragged dot in each form a mixture of experts takes: each row times its group's weights (its
    forward, its input's gradient) and each group's rows alone summed (its weights' gradient), with rows
    past every group. The lane's own result holds every element's sum in any order, and one element moved
    twice its bound breaks there alone."""
    rng = np.random.default_rng(8)
    sizes = np.int32([3, 0, 4])
    lhs = np.float32(rng.standard_normal((9, 5)))
    if ragged == "rows":
        rhs = np.float32(rng.standard_normal((3, 5, 4)))

        def product(lhs, rhs, sizes):
            return jax.lax.ragged_dot(lhs, rhs, sizes, precision="highest")
    else:
        rhs = np.float32(rng.standard_normal((9, 4)))
        numbers = jax.lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(((0,), (0,)), ((), ())),
                                                     lhs_ragged_dimensions=(0,), rhs_group_dimensions=())

        def product(lhs, rhs, sizes):
            return jax.lax.ragged_dot_general(lhs, rhs, sizes, numbers, precision="highest")

    params, given, (native,) = _first_equation(product, [lhs, rhs, sizes], primitives.ragged_dot_general_p)
    native = np.asarray(native)
    contract = ct.CONTRACTS[primitives.ragged_dot_general_p]
    assert np.all(contract(params, given, native, oc.FLOAT32))
    off = native.copy()
    at = (1, 2) if ragged == "rows" else (2, 1, 2)
    off[at] += 2 * oc.gamma(9, oc.FLOAT32) * float(np.abs(lhs).sum() * np.abs(rhs).max())
    held = contract(params, given, off, oc.FLOAT32)
    assert not held[at] and held.sum() == held.size - 1


@pytest.mark.parametrize("power", [-7, -2, 3, 8])
def test_an_integer_power_holds_its_chain_of_roundings_and_a_value_twice_past_it_breaks_it(power):
    """x to an integer power, by squarings and multiplies and a reciprocal where the power is negative: the
    lane's own result holds within its roundings' bound of the exact power, a zero's pole and an infinity's
    power named exactly; a value twice the bound off, or a negative zero's power of the other sign, breaks
    there alone."""
    rng = np.random.default_rng(9)
    x = np.float32(rng.uniform(0.25, 4, 64) * rng.choice([-1, 1], 64))
    x[:3] = np.float32([0.0, np.inf, -0.0])
    native = np.asarray(jax.lax.integer_pow(x, power))
    contract = ct.CONTRACTS[jax.lax.integer_pow_p]
    params = {"y": power}
    assert np.all(contract(params, [x], native, oc.FLOAT32))
    off = native.copy()
    off[5] = np.float32(float(off[5]) * (1 + 2 * oc.gamma(abs(power) + 1, oc.FLOAT32)))
    held = contract(params, [x], off, oc.FLOAT32)
    assert not held[5] and held.sum() == held.size - 1
    # A negative zero's power, of its sign or a pole of it, with its sign flipped.
    off = native.copy()
    off[2] = -off[2]
    held = contract(params, [x], off, oc.FLOAT32)
    assert not held[2] and held.sum() == held.size - 1


def test_an_integer_power_is_held_through_its_squarings_and_its_reciprocal():
    """x^8 is three squarings, and each squaring's rounding is squared again by those after it: seven
    factors of rounding, where its three instructions count three. The lane's own eighth power of a value
    whose error is past three roundings' bound holds (1.2637746: 6.5066457 against the exact 6.5066444,
    1.295e-6 off where three roundings allow 1.163e-6). A negative power is the reciprocal of what the chain
    made, so it follows the chain out of the range: of a square past the format's largest, a zero, where
    the exact power is a small number; of a square below the normal range, whatever that square's own loss
    makes of it."""
    contract = ct.CONTRACTS[jax.lax.integer_pow_p]
    x = np.float32([1.2637746334075928])
    native = np.asarray(jax.lax.integer_pow(x, 8))
    assert contract({"y": 8}, [x], native, oc.FLOAT32).all()
    past = np.float32([1e20, -3e19])
    native = np.asarray(jax.lax.integer_pow(past, -2))
    assert np.all(native == 0) and contract({"y": -2}, [past], native, oc.FLOAT32).all()
    below = np.float32(np.random.default_rng(11).uniform(2.0 ** -64, 2.0 ** -63, 256))
    native = np.asarray(jax.lax.integer_pow(below, -2))
    assert contract({"y": -2}, [below], native, oc.FLOAT32).all()


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
    _, findings = ct.certified(lambda x: jnp.tan(x), x)
    assert len(findings) == 1 and "no contract for tan" in findings[0]
    with pytest.raises(AssertionError):
        ct.assert_certified(lambda x: jnp.tan(x), lambda seed: (x,), "tangent", seeds=1)


def test_an_equation_whose_result_nothing_reads_is_held_too():
    """Check 2 holds every equation the program executes: a primitive without a contract whose result the
    output never reads (a tangent dropped before the return, inside a call too) is still a finding."""
    x = np.float32(np.linspace(0.1, 3, 32))
    for discarded in (lambda x: (jnp.tan(x), x + 1)[1],
                      lambda x: jax.jit(lambda x: (jnp.tan(x), x * 2))(x)[1]):
        _, findings = ct.certified(discarded, x)
        assert len(findings) == 1 and "no contract for tan" in findings[0], findings


def test_an_equation_past_the_lanes_reach_is_a_finding_by_its_path():
    """A division by a literal, which XLA may compute as a product with the literal's rounded reciprocal
    (ledger 1402), is held to the lane's own division's reach: it is a finding, named by its equation,
    exactly where its results' levels pass that reach; a division by a value the program is given is the
    lane's own division and holds."""
    x = np.float32(np.random.default_rng(4).uniform(0.1, 3, 512))
    outputs, findings = ct.certified(lambda x: x / 3, x)
    levels = ct.LEVELS[jax.lax.div_p]([x, np.float32(3)], outputs, oc.FLOAT32, ct.lane().flushes)
    past = int(np.count_nonzero(levels > ct.lane().reach[jax.lax.div_p, "float32"]))
    assert len(findings) == (1 if past else 0)
    named = f"div#0: {past} of 512 elements break the contract"
    assert all(finding.startswith(named) for finding in findings)
    assert ct.certified(jnp.divide, x, np.float32(np.full(512, 3)))[1] == []


def test_a_sum_loses_a_subnormal_only_where_the_lane_flushes_it():
    """A sum of one subnormal returned as zero holds its contract where the lane flushes (its operand
    reads as zero) and breaks it where the lane keeps subnormals, whose sums lose nothing to underflow;
    the lane's own sum holds either way."""
    terms, params = np.float32([2.0 ** -140]), {"axes": (0,)}
    lost = ct.CONTRACTS[jax.lax.reduce_sum_p](params, [terms], np.float32(0), oc.FLOAT32)
    assert bool(lost) == ct.lane().flushes
    assert ct.CONTRACTS[jax.lax.reduce_sum_p](params, [terms], np.asarray(jnp.sum(terms)), oc.FLOAT32)


def test_the_lane_detects_tininess_after_rounding_and_keeps_subnormals_in_moves_and_casts():
    """A product exactly 2^-126 - 2^-172 rounds to the least normal, at an unbounded exponent and on the
    gradual grid alike, and the lane keeps it (tininess after rounding); a subnormal product holds the
    contract as the lane rounds it, zero where it flushes and the subnormal where it keeps them, and the
    other breaks it; a reshape, a negation and a cast between bf16 and float32 keep a subnormal; a
    float32 past int32's range saturates and NaN casts to 0."""
    a, b = np.float32(1 - 2.0 ** -23), np.float32(2.0 ** -126 + 2.0 ** -149)
    native = np.asarray(jax.jit(jnp.multiply)(a, b))
    assert float(native) == 2.0 ** -126
    assert np.all(ct.CONTRACTS[jax.lax.mul_p]({}, [a, b], native, oc.FLOAT32))
    assert not np.all(ct.CONTRACTS[jax.lax.mul_p]({}, [a, b], np.float32(0), oc.FLOAT32))
    tiny_product = [np.float32(2.0 ** -70)] * 2
    flushed, kept = np.float32(0), np.float32(2.0 ** -140)
    own, other = (flushed, kept) if ct.lane().flushes else (kept, flushed)
    assert np.all(ct.CONTRACTS[jax.lax.mul_p]({}, tiny_product, own, oc.FLOAT32))
    assert not np.any(ct.CONTRACTS[jax.lax.mul_p]({}, tiny_product, other, oc.FLOAT32))
    tiny = np.float32([2.0 ** -140, -(2.0 ** -130)])
    assert np.array_equal(np.asarray(jax.jit(lambda x: -x.reshape(2, 1))(tiny)).ravel(), -tiny)
    assert np.all(ct.CONTRACTS[jax.lax.neg_p]({}, [tiny], -tiny, oc.FLOAT32))
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
    exp = ct.LEVELS[jax.lax.exp_p]
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
    log = ct.LEVELS[jax.lax.log_p]([np.float32([0.0, np.inf, -1.0])], limits, oc.FLOAT32, flushes)
    assert log.tolist() == [0, 0, 0]
    poles = np.float32([np.inf, -np.inf])
    erf_inv = ct.LEVELS[jax.lax.erf_inv_p]
    assert erf_inv([np.float32([1.0, -1.0])], poles, oc.FLOAT32, flushes).tolist() == [0, 0]


@pytest.mark.parametrize("flushes", [True, False])
@pytest.mark.parametrize("fmt", [oc.FLOAT32, oc.BFLOAT16], ids=["float32", "bfloat16"])
def test_a_zero_of_the_wrong_sign_or_a_missed_exact_value_is_no_rounding(flushes, fmt):
    """exp(-1000) is positive and under the least value past zero: +0 is level 1, and -0 no rounding of
    it; sqrt(-0) is -0 exactly and +0 no rounding of it; exp(±0) is 1 and log(1) is +0 exactly, so a step
    of the grid off either, which a measured reach would otherwise admit, is no rounding either."""
    dtype = oc.DTYPES[fmt.name]

    def at(values):
        return np.asarray(values, np.float64).astype(dtype)

    exp, log, sqrt = ct.LEVELS[jax.lax.exp_p], ct.LEVELS[jax.lax.log_p], ct.LEVELS[jax.lax.sqrt_p]
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
    assert np.all(ct.CONTRACTS[jax.lax.bitcast_convert_type_p](params, [payload], native, oc.FLOAT32))
    other = np.uint32([0x7FC00002]).view(np.float32)
    assert not np.any(ct.CONTRACTS[jax.lax.bitcast_convert_type_p](params, [payload], other, oc.FLOAT32))
    ints = np.int32([[-1, 2 ** 31 - 1], [5, -7]])
    for fn in (lambda a: jax.lax.reshape(a, (4,), dimensions=(1, 0)),
               lambda a: jax.lax.shift_right_logical(a, np.int32(1)),
               lambda a: jax.lax.bitcast_convert_type(a, jnp.int16),
               lambda a: jnp.sum(a * a + a, axis=0),
               lambda a: jnp.sum(a.astype(jnp.float32)[:, :0], axis=1)):
        assert ct.certified(fn, ints)[1] == [], fn


def test_an_integer_division_is_held_at_every_64_bit_magnitude():
    """An integer division truncates toward zero and wraps to its width, and its remainder takes the
    dividend's sign: int64's least value, -2^63, over 3 is -3074457345618258602, over -1 itself, uint64's
    values past 2^63 divide as themselves, and a division by zero is every bit set and its remainder the
    dividend, each held as XLA computes it (row 1966's head review); so a program guarding its division
    (`where(n > 0, total // n, 0)`, which computes both sides) holds check 2."""
    with jax.enable_x64(new_val=True):
        least = -(2 ** 63)
        signed = (np.int64([least, least, least, 7, -7, 2 ** 63 - 1, 5, -5, least]),
                  np.int64([3, -3, -1, -2, 2, -1, 0, 0, 0]))
        unsigned = (np.uint64([2 ** 64 - 1, 2 ** 63 + 5, 9]), np.uint64([3, 2, 0]))
        narrow = (np.int32([-(2 ** 31), 5, -5, 0]), np.int32([-1, 0, 0, 0]))
        for operands in (signed, unsigned, narrow):
            assert ct.certified(jax.lax.div, *operands)[1] == [], operands
            assert ct.certified(jax.lax.rem, *operands)[1] == [], operands
    totals, counts = np.int32([12, 7, -9, 0]), np.int32([4, 0, 2, 0])
    assert ct.certified(lambda total, n: jnp.where(n > 0, total // n, 0), totals, counts)[1] == []


def _first_equation(fn, operands, primitive):
    """The first equation binding `primitive` check 2 holds running `fn` on `operands`
    (`certified.observing`): its params, the operands its contract was given and its results."""
    with ct.observing() as seen:
        ct.certified(fn, *operands)
    found = next(((params, given, results) for held, params, given, results in seen if held is primitive),
                 None)
    assert found is not None, f"no {primitive.name} equation"
    return found



def _off_by_one_element(result):
    """`result` with its first element moved by one step of its storage (a flipped low bit)."""
    wrong = np.array(result, copy=True)
    flat = wrong.reshape(-1)
    if flat.dtype == np.bool_:
        flat[0] = not flat[0]
    elif np.issubdtype(flat.dtype, np.integer):
        flat[0] = flat[0] + 1
    else:
        bits = flat.view(ct._bits(flat).dtype)
        bits[0] ^= 1
    return wrong


@pytest.mark.parametrize("primitive, fn, operands", [
    (jax.lax.pad_p, lambda a: jax.lax.pad(a, np.float32(-3), ((1, 2, 0), (-1, 1, 2))),
     [np.float32(np.arange(12).reshape(3, 4))]),
    (jax.lax.iota_p, lambda: jax.lax.broadcasted_iota(jnp.int32, (3, 5), 1), []),
    (jax.lax.gather_p, lambda a, i: a[i],
     [np.float32(np.arange(20).reshape(5, 4)), np.int32([[4], [0], [2]])]),
    (jax.lax.gather_p, lambda a, i: jnp.take_along_axis(a, i, axis=1, mode="clip"),
     [np.float32(np.arange(20).reshape(5, 4)), np.int32([[0, 9], [3, -2], [1, 1], [2, 0], [3, 3]])]),
    (jax.lax.gather_p, lambda a, i: jnp.take(a, i, axis=0, mode="fill", fill_value=-1.0),
     [np.float32(np.arange(20).reshape(5, 4)), np.int32([1, 7, -1, 4])]),
    (jax.lax.gather_p, lambda a, i: jax.vmap(lambda row, at: row[at])(a, i),
     [np.float32(np.arange(20).reshape(5, 4)), np.int32([3, 0, 1, 2, 3])]),
    (jax.lax.gather_p, lambda a, i: jnp.take_along_axis(a, i, axis=1, mode="clip"),
     [np.arange(20).reshape(5, 4).astype(ml_dtypes.bfloat16),
      np.int32([[0, 9], [3, -2], [1, 1], [2, 0], [3, 3]])]),
    (jax.lax.gather_p, lambda a, i: a[i],
     [np.arange(20).reshape(5, 4).astype(ml_dtypes.bfloat16), np.int32([[4], [0], [2]])]),
    (jax.lax.stack_p, lambda a, b: jnp.stack([a, b], axis=1), [np.float32([1, 2, 3]), np.float32([4, 5, 6])]),
    (jax.lax.optimization_barrier_p, lambda a, b: jax.lax.optimization_barrier((a, b)),
     [np.float32([1, 2]), np.float32([[3, 4, 5]])]),
    (jax.lax.rev_p, lambda a: jax.lax.rev(a, (0, 1)), [np.float32(np.arange(12).reshape(3, 4))]),
    (jax.lax.split_p, lambda a: jax.lax.split(a, (1, 3), axis=1), [np.float32(np.arange(12).reshape(3, 4))]),
    (primitives.name_p, lambda a: checkpoint_name(a, "held"), [np.float32([1, 2, 3])]),
    (jax.lax.dynamic_slice_p, lambda a, i: jax.lax.dynamic_slice(a, (i, 1), (2, 2)),
     [np.float32(np.arange(12).reshape(4, 3)), np.int32(3)]),
    (jax.lax.sort_p, jax.lax.sort, [np.float32([2, -0.0, 0.0, np.nan, -np.inf, 1, 1, -1])]),
    (jax.lax.sort_p, lambda k, v: jax.lax.sort((k, v), num_keys=1, is_stable=True),
     [np.int32([3, 1, 3, 0, 1]), np.int32([0, 1, 2, 3, 4])]),
    (jax.lax.sort_p, lambda k, v: jax.lax.sort((k, v), num_keys=1, is_stable=True),
     [np.float32([0.0, -np.nan, -0.0, 1, np.nan]), np.int32([0, 1, 2, 3, 4])]),
    (jax.lax.top_k_p, lambda a: jax.lax.top_k(a, 3),
     [np.float32([[1, np.nan, -0.0, 0.0, 2], [1, 1, 1, -1, 0]])]),
    (jax.lax.unstack_p, lambda a: jnp.unstack(a, axis=1), [np.float32(np.arange(12).reshape(3, 4))]),
    (jax.lax.tile_p, lambda a: jnp.tile(a, 3), [np.int32([1, 2])]),
    (jax.lax.stop_gradient_p, jax.lax.stop_gradient, [np.float32([1, -0.0, 3])]),
    (jax.lax.xor_p, jnp.bitwise_xor, [np.uint32([1, 5, 2 ** 31]), np.uint32([3, 3, 1])]),
    (jax.lax.shift_right_arithmetic_p, jax.lax.shift_right_arithmetic,
     [np.int32([-8, 8, -1, 5]), np.int32([1, 2, 40, -1])]),
    (jax.lax.rem_p, jax.lax.rem, [np.int32([-7, 7, -7, 9]), np.int32([3, -3, -3, 4])]),
    (jax.lax.rem_p, jax.lax.rem, [np.float32([-7.5, 7.25, 1e10, -0.0]), np.float32([2, -3, 3, 5])]),
    (jax.lax.sign_p, jax.lax.sign, [np.float32([-0.0, 0.0, -2.5, np.nan, np.inf])]),
    (jax.lax.sign_p, jax.lax.sign, [np.int32([-5, 0, 7])]),
    (jax.lax.round_p, lambda a: jax.lax.round(a, jax.lax.RoundingMethod.TO_NEAREST_EVEN),
     [np.float32([0.5, 1.5, 2.5, -0.5, -2.5, 0.49999997, 1e10])]),
    (jax.lax.round_p, lambda a: jax.lax.round(a, jax.lax.RoundingMethod.AWAY_FROM_ZERO),
     [np.float32([0.5, 1.5, 2.5, -0.5, -2.5, 0.49999997, 1e10])]),
    (jax.lax.reduce_precision_p, lambda a: jax.lax.reduce_precision(a, exponent_bits=8, mantissa_bits=7),
     [np.float32([2.0 ** -130, 2.0 ** -140, 1 + 2.0 ** -8, 1 + 3 * 2.0 ** -8, 3.4e38, np.nan])]),
    (jax.lax.reduce_precision_p, lambda a: jax.lax.reduce_precision(a, exponent_bits=4, mantissa_bits=3),
     [np.float32([2.0 ** -9, 2.0 ** -12, 1 + 2.0 ** -4, 1 + 3 * 2.0 ** -4, 250.0, -248.0])]),
    (jax.lax.scatter_p, lambda a, i, u: a.at[i[:, 0]].set(u, mode="drop"),
     [np.float32(np.arange(12).reshape(4, 3)), np.int32([[2], [0], [7]]),
      np.float32(-np.arange(9).reshape(3, 3))]),
    (jax.lax.scatter_p, lambda a, i, u: a.at[i[:, 0]].set(u),
     [np.zeros((4, 2), bool), np.int32([[3], [1]]), np.ones((2, 2), bool)]),
], ids=["pad", "iota", "gather", "gather clipped", "gather filled", "gather batched", "gather clipped bf16",
        "gather bf16", "stack", "barrier", "rev", "split", "name", "dynamic slice clamped", "sort",
        "sort keyed", "sort keyed zeros and nans", "top_k", "unstack",
        "tile", "stop_gradient", "xor", "shift", "rem", "rem float", "sign", "sign int", "round even",
        "round away", "reduce to bf16", "reduce to e4m3", "scatter", "scatter bool"])
def test_each_move_holds_its_own_result_and_refuses_one_element_off_it(primitive, fn, operands):
    """Every move check 2 certifies by a numpy oracle of XLA's semantics (`CONTRACTS`), padding with
    interior and negative edges, an iota, a gather in each mode and batched, a stack, an optimization
    barrier of two operands, a reversal, a split in two, a checkpoint primitive, a dynamic slice clamped to
    fit,
    a sort in the total order (its zeros and NaNs placed) and keyed (its zeros tied and its NaNs, either
    sign, last), a top_k, an unstack, a tile, a
    stop_gradient, the integer bit operations, a remainder, a sign, a rounding to an integral value either
    way, a reduction of precision to bfloat16 and to E4M3 (subnormals, ties and overflow among them) and a
    scatter: the program certifies,
    its contract holding the lane's own result at every element, and the contract refuses that result
    with one element a step off."""
    assert ct.certified(fn, *operands)[1] == []
    params, given, results = _first_equation(fn, operands, primitive)
    contract = ct.CONTRACTS[primitive]
    for index, result in enumerate(results):
        native = np.asarray(result)
        # As `_held` hands it: a result of its own position reads its own part of the operands alone.
        operands = (ct.RESULT_OPERANDS[primitive](params, given, index) if primitive in ct.RESULT_OPERANDS
                    else given)
        assert np.all(contract(params, operands, native, None)), primitive
        assert not np.all(contract(params, operands, _off_by_one_element(native), None)), primitive


def test_a_move_of_keys_holds_their_data_and_a_word_off_breaks_it():
    """A broadcast of PRNG keys, as a step's key reaches every row of a draw, moves each key's data whole:
    the lane's own result holds, and one word of one key's data off breaks it there alone."""
    keys = jax.random.split(jax.random.key(3), 2)

    def broadcast(keys):
        return jax.random.key_data(jax.lax.broadcast_in_dim(keys, (3, 2), (1,)))

    assert ct.certified(broadcast, keys)[1] == []
    params, given, (result,) = _first_equation(lambda keys: jax.lax.broadcast_in_dim(keys, (3, 2), (1,)),
                                               [keys], jax.lax.broadcast_in_dim_p)
    contract = ct.CONTRACTS[jax.lax.broadcast_in_dim_p]
    assert np.all(contract(params, given, result, None))
    words = np.array(jax.random.key_data(result))
    words[1, 0, 0] ^= 1
    held = contract(params, given, jax.random.wrap_key_data(words), None)
    assert not held[1, 0] and held.sum() == held.size - 1


def test_a_compensated_sum_holds_its_pairs_and_a_pair_a_step_off_breaks():
    """A compensated sum's reduction (`dew.nn.precision.compensated_sum`'s, its combine
    `compensated_add`) certifies: its `(hi, lo)` pairs hold the exact sum to fp32's precision squared, and a
    pair whose `hi` is a step off, an error a plain sum's rounding could make, breaks there alone, as a
    non-finite `hi` of a sum near the range's top that cannot overflow does."""
    from dew.nn.precision import compensated_sum

    rng = np.random.default_rng(10)
    values = np.float32(rng.standard_normal((3, 64)) * 2.0 ** rng.integers(-8, 8, (3, 64)))
    assert ct.certified(lambda x: compensated_sum(x, (1,)), values)[1] == []
    params, given, results = _first_equation(lambda x: compensated_sum(x, (1,)), [values], jax.lax.reduce_p)
    held = ct._pair_held(params, given, [np.asarray(result) for result in results], oc.FLOAT32)
    assert all(np.all(ok) for ok in held)
    high = _off_by_one_element(np.asarray(results[0]))
    held = ct._pair_held(params, given, [high, np.asarray(results[1])], oc.FLOAT32)
    assert not held[0][0] and held[0].sum() == held[0].size - 1
    # Sums near the range's top that no order overflows, the largest value alone and as two halves among
    # them: a non-finite `hi` breaks each.
    largest = np.finfo(np.float32).max
    top = np.float32([[2.0 ** 127, 0], [largest, 0], [largest / 2, largest / 2]])
    params, given, results = _first_equation(lambda x: compensated_sum(x, (1,)), [top], jax.lax.reduce_p)
    native = [np.asarray(result) for result in results]
    assert all(np.all(ok) for ok in ct._pair_held(params, given, native, oc.FLOAT32))
    for wrong in (np.nan, np.inf, -np.inf):
        high = np.full(3, wrong, np.float32)
        held = ct._pair_held(params, given, [high, np.asarray(results[1])], oc.FLOAT32)
        assert not held[0].any(), wrong
    # Two largest values overflow in every order, a TwoSum of an infinite sum making both halves NaN: the
    # lane's NaN pair holds, an infinity of either sign breaks it.
    twice = np.float32([[largest, largest]])
    params, given, results = _first_equation(lambda x: compensated_sum(x, (1,)), [twice], jax.lax.reduce_p)
    native = [np.asarray(result) for result in results]
    assert np.isnan(native[0]).all()
    assert all(np.all(ok) for ok in ct._pair_held(params, given, native, oc.FLOAT32))
    for wrong in (np.inf, -np.inf):
        held = ct._pair_held(params, given, [np.float32([wrong]), native[1]], oc.FLOAT32)
        assert not held[0].any(), wrong


def test_a_variadic_reduction_is_held_by_the_combine_it_was_given():
    """A variadic reduction is one of Dew's where its combine is the function its owner hands
    `jax.lax.reduce`: a pair sum over `compensated_add` certifies, and the same sum given another function
    that computes as it does has no contract, a finding at that reduction and never a result passed
    unheld."""
    from dew.nn.precision import compensated_add

    values = np.float32(np.random.default_rng(12).standard_normal((3, 64)))

    def summed(combine):
        zero = np.float32(0)
        return lambda x: jax.lax.reduce((x, jnp.zeros_like(x)), (zero, zero), combine, (1,))

    assert ct.certified(summed(compensated_add), values)[1] == []
    assert len(ct.certified(summed(lambda left, right: compensated_add(left, right)), values)[1]) == 1


@pytest.mark.parametrize("predict", [False, True])
def test_a_rows_online_softmax_holds_its_terms_and_a_sum_or_column_off_breaks(predict, monkeypatch):
    """A row's online softmax (`dew.objectives.lm.chunked._row_terms`' reduction), as its users run it,
    without the suite's float64 CPU reductions: its maximum exact, its sums of exponentials within their
    bound and, where it predicts, the maximum's first column; a sum a thousandth off, or a column one off,
    breaks that row alone."""
    from dew.nn.precision import WIDE_CPU_ARITHMETIC
    from dew.objectives.lm.chunked import _row_terms

    monkeypatch.delenv(WIDE_CPU_ARITHMETIC, raising=False)

    rng = np.random.default_rng(11)
    logits = np.float32(rng.standard_normal((4, 48)) * 4)
    logits[1, 7] = logits[1, 30] = logits[1].max() + 1
    labels = np.int32([3, 0, 47, 12])

    def terms(logits, labels):
        return _row_terms(logits, predict, labels)

    assert ct.certified(terms, logits, labels)[1] == []
    params, given, results = _first_equation(terms, [logits, labels], jax.lax.reduce_p)
    native = [np.asarray(result) for result in results]
    assert all(np.all(ok) for ok in ct._softmax_held(params, given, native, oc.FLOAT32))
    wrong = [value.copy() for value in native]
    wrong[1][2] *= np.float32(1.001)
    held = ct._softmax_held(params, given, wrong, oc.FLOAT32)
    assert not held[1][2] and held[1].sum() == held[1].size - 1
    if predict:
        wrong = [value.copy() for value in native]
        wrong[3][1] += 1
        held = ct._softmax_held(params, given, wrong, oc.FLOAT32)
        assert not held[3][1] and held[3].sum() == held[3].size - 1


def test_a_rows_online_softmax_keeps_the_least_column_of_a_tie_and_reads_its_initial_values():
    """The online softmax's contract over operands in any order: of two equal maxima its column is the
    lesser (the combine's tie rule), not the first in the row; an initial value past every logit is the
    maximum and its column."""
    params = {"dimensions": (1,)}
    ones = np.float32([[1, 1]])
    logits, columns, tail = np.float32([[0, 0]]), np.int32([[7, 3]]), 1 + 2 * np.exp(-5)
    for initial, found, wrong in (((-np.inf, 0, 0, 2 ** 31 - 1), (0, 2, 2, 3), 7),
                                  ((5, 1, 1, 9), (5, tail, tail, 9), 3)):
        given = [logits, ones, ones, columns, *np.float32(initial[:3]), np.int32(initial[3])]
        results = [*np.float32([[value] for value in found[:3]]), np.int32([found[3]])]
        assert all(np.all(ok) for ok in ct._softmax_held(params, given, results, oc.FLOAT32))
        assert not ct._softmax_held(params, given, [*results[:3], np.int32([wrong])], oc.FLOAT32)[3][0]


def test_a_key_made_and_folded_holds_its_bits_and_one_off_breaks():
    """A key made from a seed and folded with data, inside a program, holds the bits a separate key of the
    same seed and data has; one bit off breaks it."""
    seed, data = np.uint32(7), np.uint32(42)

    def keyed(seed, data):
        return jax.random.key_data(jax.random.fold_in(jax.random.key(seed), data))

    assert ct.certified(keyed, seed, data)[1] == []
    for primitive in (primitives.random_seed_p, primitives.random_fold_in_p):
        params, given, (result,) = _first_equation(keyed, [seed, data], primitive)
        contract = ct.CONTRACTS[primitive]
        assert np.all(contract(params, given, result, None)), primitive
        data_of = np.array(jax.random.key_data(result))
        data_of.reshape(-1)[0] ^= 1
        assert not np.all(contract(params, given, jax.random.wrap_key_data(data_of), None)), primitive


def test_an_integer_division_truncates_wraps_and_refuses_one_off():
    """An integer division truncates toward zero, its most negative value over -1 wrapping to itself, at
    every width: the lane's own result holds, and one off breaks."""
    a = np.int32([-7, 7, -7, 9, -(2 ** 31)])
    b = np.int32([2, -2, -2, 4, -1])
    native = np.asarray(jax.lax.div(a, b))
    contract = ct.CONTRACTS[jax.lax.div_p]
    assert np.all(contract({}, [a, b], native, None))
    assert not np.all(contract({}, [a, b], native + np.int32(1), None))
    # At a width past int64's magnitudes: the most negative int64 halved, an unsigned value past int64.
    for a, b, quotient in ((np.int64([-(2 ** 63)]), np.int64([2]), np.int64([-(2 ** 62)])),
                           (np.uint64([2 ** 64 - 2]), np.uint64([2]), np.uint64([2 ** 63 - 1]))):
        assert contract({}, [a, b], quotient, None).all()
        assert not contract({}, [a, b], quotient + quotient.dtype.type(1), None).any()


def test_a_split_holds_each_piece_to_its_own_run():
    """A split into two pieces of one shape: its pieces in place hold, the two swapped break, though each
    is a run of the operand."""
    a = np.float32(np.arange(8).reshape(2, 4))

    def split(a):
        return jax.lax.split(a, (2, 2), axis=1)

    _, given, results = _first_equation(split, [a], jax.lax.split_p)
    eqn = jax.make_jaxpr(split)(a).jaxpr.eqns[0]
    findings = []
    ct._held(eqn, given, results, findings, "split")
    assert findings == []
    ct._held(eqn, given, results[::-1], findings, "split")
    assert len(findings) == 2, findings


def test_rsqrt_and_logistic_name_their_exact_values():
    """rsqrt is exact at the powers of four and logistic at zero, which their contracts hold bit for bit:
    a result a step off there breaks, as do a wrong pole sign and a missed limit."""
    # Normal operands: a lane that flushes reads a subnormal one as zero, whose rsqrt is the pole.
    fours = np.float32([1.0, 4.0, 0.25, 2.0 ** -126, 2.0 ** 126])
    roots = np.float32([1.0, 0.5, 2.0, 2.0 ** 63, 2.0 ** -63])
    assert np.all(ct.CONTRACTS[jax.lax.rsqrt_p]({}, [fours], roots, oc.FLOAT32))
    off = np.nextafter(roots, np.float32(np.inf))
    assert not np.any(ct.CONTRACTS[jax.lax.rsqrt_p]({}, [fours], off, oc.FLOAT32))
    poles = np.float32([0.0, -0.0, np.inf])
    assert np.all(ct.CONTRACTS[jax.lax.rsqrt_p]({}, [poles], np.float32([np.inf, -np.inf, 0.0]), oc.FLOAT32))
    flipped = np.float32([-np.inf, np.inf, 1e-30])
    assert not np.any(ct.CONTRACTS[jax.lax.rsqrt_p]({}, [poles], flipped, oc.FLOAT32))
    limits = np.float32([0.0, np.inf, -np.inf])
    assert np.all(ct.CONTRACTS[jax.lax.logistic_p]({}, [limits], np.float32([0.5, 1.0, 0.0]), oc.FLOAT32))
    assert not np.any(ct.CONTRACTS[jax.lax.logistic_p]({}, [limits],
                                               np.nextafter(np.float32([0.5, 1.0, 0.0]), np.float32(0.25)),
                                               oc.FLOAT32))


def test_a_barrier_holds_each_result_to_its_own_operand():
    """An optimization barrier of two operands of one shape: its results in place hold, the two swapped
    break, though each is one of its operands."""
    a, b = np.float32([1, 2]), np.float32([3, 4])
    _, given, results = _first_equation(lambda a, b: jax.lax.optimization_barrier((a, b)), [a, b],
                                        jax.lax.optimization_barrier_p)
    eqn = jax.make_jaxpr(lambda a, b: jax.lax.optimization_barrier((a, b)))(a, b).jaxpr.eqns[0]
    findings = []
    ct._held(eqn, given, results, findings, "barrier")
    assert findings == []
    ct._held(eqn, given, results[::-1], findings, "barrier")
    assert len(findings) == 2, findings


def test_a_scan_is_certified_a_step_at_a_time_and_a_steps_finding_named_by_it():
    """`certified` runs a scan's body once a step (forward and reversed), its carry threaded and its
    outputs stacked, giving the lane's own scan's results; a body primitive no contract covers is found at
    each step, by the step's path."""
    x = np.float32(np.linspace(0.5, 2, 12).reshape(4, 3))

    def cumulative(reverse):
        def run(x):
            return jax.lax.scan(lambda carry, row: (carry * row + 1, carry - row), np.float32(np.ones(3)), x,
                                reverse=reverse)
        return run

    def stepped(reverse):
        # The body's own operations, eagerly, a step at a time: the lane's, unfused.
        carry, ys = np.float32(np.ones(3)), []
        for row in (x[::-1] if reverse else x):
            ys.append(jnp.subtract(carry, row))
            carry = jnp.add(jnp.multiply(carry, row), np.float32(1))
        return carry, jnp.stack(ys[::-1] if reverse else ys)

    for reverse in (False, True):
        outputs, findings = ct.certified(cumulative(reverse), x)
        assert findings == []
        for found, native in zip(jax.tree.leaves(outputs), jax.tree.leaves(stepped(reverse)), strict=True):
            np.testing.assert_array_equal(np.asarray(found), np.asarray(native))
    # No steps, and steps over no inputs: the carry passes through, the outputs are empty or stacked.
    empty, findings = ct.certified(lambda x: jax.lax.scan(lambda c, row: (c + row, c * row), x[0], x[:0]), x)
    assert findings == [] and np.asarray(empty[1]).shape == (0, 3)
    np.testing.assert_array_equal(np.asarray(empty[0]), x[0])
    counted, findings = ct.certified(lambda c: jax.lax.scan(lambda c, _: (c * np.float32(2), c), c, None,
                                                            length=3), x[0])
    assert findings == [] and np.asarray(counted[1]).shape == (3, 3)
    _, findings = ct.certified(lambda x: jax.lax.scan(lambda c, row: (c + jnp.tan(row), None),
                                                      np.float32(np.zeros(3)), x), x)
    assert len(findings) == 4 and all("scan#" in finding and f"/{step}/tan" in finding
                                      for step, finding in enumerate(findings)), findings


def test_calls_are_certified_inside_and_unsupported_types_refused():
    """A rematerialised body is run one primitive at a time; a float16, float64-free complex or float8
    value has no contract of ours, and a key's wrapping is held to its data."""
    x = np.float32(np.linspace(1, 2, 8))
    _, findings = ct.certified(jax.checkpoint(lambda x: jnp.log(x) * x), x)
    assert findings == []
    _, findings = ct.certified(jax.checkpoint(lambda x: jnp.tan(x)), x)
    assert findings and "remat" in findings[0] and "tan" in findings[0]
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
    for name, fn in ((jax.lax.max_p, jnp.maximum), (jax.lax.min_p, jnp.minimum)):
        native = np.asarray(jax.jit(fn)(zeros, others))
        assert np.all(ct.CONTRACTS[name]({}, [zeros, others], native, oc.FLOAT32))
        assert not np.any(ct.CONTRACTS[name]({}, [zeros, others], -native, oc.FLOAT32))
    rows = np.float32([[-0.0, 0.0], [1.0, np.nan], [-0.0, -0.0]])
    for fn in (lambda a: jnp.max(a, 1), lambda a: jnp.min(a, 1), lambda a: jnp.any(a > 0, 1),
               lambda a: jnp.all(a > 0, 1)):
        assert ct.certified(fn, rows)[1] == [], fn
    assert ct.certified(jnp.log, np.float32(2.0))[1] == []
    assert ct.certified(jax.lax.erf_inv, np.float32([0.0, -0.0, 2.0 ** -140]))[1] == []
    tiny = np.float32([2.0 ** -126])
    assert not np.any(ct.CONTRACTS[jax.lax.erf_inv_p]({}, [np.float32([0.0])], tiny, oc.FLOAT32))
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
            assert ct.LEVELS[jax.lax.exp_p]([far], native, oc.FLOAT32, flushes).tolist() == [1] * 6
    assert not np.any(ct.CONTRACTS[jax.lax.exp_p]({}, [far], np.float32([1.0] * 6), oc.FLOAT32))


def _truncating_sum(values: np.ndarray) -> np.float32:
    """`values` summed in fp32 in their order, each add truncated toward zero, as a tensor core adds: the
    exact sum as TwoSum's high and low parts, its nearest fp32 value stepped toward zero wherever that
    overshoots the exact sum, low part included."""
    total = np.float32(0)
    for value in values:
        high, low = oc.two_sum(np.float64(total), np.float64(value))
        stepped = np.float32(high)
        # The nearest value less the exact sum is (stepped - high) - low, its first difference exact.
        beyond = np.float64(stepped) - high
        if (high > 0 and beyond > low) or (high < 0 and beyond < low):
            stepped = np.nextafter(stepped, np.float32(0))
        total = stepped
    return total


def _outer(left, right) -> np.ndarray:
    """`[rows, columns, depth]`: each element's products, its row of `left` times its row of `right`."""
    return left[:, None, :] * right[None, :, :]


def _without(products: nc.Terms, at) -> nc.Terms:
    """`products` with each element's term at `at` (`[...]`, an index along the last axis) lost."""
    exact, magnitudes = products.exact.copy(), products.magnitudes.copy()
    for found in (exact, magnitudes):
        np.put_along_axis(found, np.asarray(at)[..., None], 0.0, -1)
    return nc.Terms(exact, magnitudes)


def _truncating_sums(products: np.ndarray, rng) -> np.ndarray:
    """Each element's `products` (summed along the last axis) summed by `_truncating_sum` in an order of its
    own, as float64."""
    flat = products.reshape(-1, products.shape[-1])
    return np.asarray([_truncating_sum(row[rng.permutation(row.size)]) for row in flat],
                      np.float64).reshape(products.shape[:-1])


def _stepped_sums(products: np.ndarray, width: int, rng, exact: bool) -> list[np.ndarray]:
    """The fp32 accumulator of each element's `products` (summed along the last axis) after each depth step
    of `width` of them, from zero: each step's partial summed from zero by truncating adds in an order of its
    own and added by an IEEE add (`exact`, as exact sums promote it), or each step's products added into the
    accumulator itself by truncating adds, as the tensor cores add without exact sums."""
    flat = products.reshape(-1, products.shape[-1])
    states = [np.zeros(flat.shape[0], np.float32)]
    for start in range(0, flat.shape[-1], width):
        rows = (row[rng.permutation(row.size)] for row in flat[:, start:start + width])
        states.append(np.asarray([np.float32(running + _truncating_sum(row)) if exact else
                                  _truncating_sum(np.concatenate([[running], row]))
                                  for running, row in zip(states[-1], rows, strict=True)], np.float32))
    return [np.asarray(state, np.float64).reshape(products.shape[:-1]) for state in states]


def _steps_held(states: list, products: nc.Terms, width: int) -> list[np.ndarray]:
    """Each depth step's witness (`promoted_step_holds`) of `states`, the accumulators `_stepped_sums` gives,
    against the true `products`."""
    return [nc.promoted_step_holds(before, after, products[..., j * width:(j + 1) * width], flushes=False)
            for j, (before, after) in enumerate(itertools.pairwise(states))]


def _steps_apart(states: list, products: nc.Terms, width: int, fault_ends) -> list[np.ndarray]:
    """Each depth step's separation (`separated`) of the witness's interval on the true `products`
    (`promoted_step_ends`) from a fault's, `fault_ends(before, step)` of the accumulator before the step and
    the step's slice of the depth."""
    def apart(j, before):
        step = slice(j * width, (j + 1) * width)
        return nc.separated(nc.promoted_step_ends(before, before, products[..., step], flushes=False),
                            fault_ends(before, step))

    return [apart(j, before) for j, before in enumerate(states[:-1])]


def test_the_truncating_model_keeps_the_low_part_of_an_add():
    """1 less 2^-80, both exact bf16 products, truncates to fp32's value below one, which a float64 add
    of them, rounding to one, would hide."""
    assert _truncating_sum(np.asarray([1.0, -2.0 ** -80])) == np.nextafter(np.float32(1), np.float32(0))


def test_a_native_dot_holds_products_that_truncate_away_below_the_normal_range():
    """128 products of 1.5 * 2^-75 and 2^-75, each a subnormal fp32 sum truncates to zero: the output zero
    holds, as a truncating add below the normal range loses up to a whole subnormal unit."""
    products = nc.terms(np.multiply, np.full((1, 128), 1.5 * 2.0 ** -75), np.full((1, 128), 2.0 ** -75),
                        fmt=oc.BFLOAT16)
    native = np.asarray([_truncating_sum(products.exact[0])], np.float64)
    assert native[0] == 0
    assert np.all(nc.native_dot_holds(native, products, oc.FLOAT32, flushes=False))


def test_a_native_dot_aligns_a_subnormal_operand_at_its_least_normal_and_no_bare_product_of_one():
    """The native-dot contract's magnitudes (row 2084, link 3): the tensor cores' sum of `SUBNORMAL_CASE`'s
    five fp16 products, one of the subnormal 2^-24, holds where that operand counts at fp16's least normal
    (`nc.aligned`) and is refused at the products' own magnitudes; bare products, which a subnormal operand
    may have made, are refused rather than take the tight bound."""
    left, right, native = nc.SUBNORMAL_CASE
    products = nc.terms(np.multiply, np.asarray([left]), np.asarray([right]), fmt=oc.FLOAT16)
    assert nc.native_dot_holds([native], products, oc.FLOAT32, flushes=False).tolist() == [True]
    tight = nc.Terms(products.exact, np.abs(products.exact))
    assert nc.native_dot_holds([native], tight, oc.FLOAT32, flushes=False).tolist() == [False]
    for bare in (products.exact, np.full((1, 4), 4.0)):
        with pytest.raises(TypeError):
            nc.native_dot_ends(bare, oc.FLOAT32, flushes=False)


def test_a_native_dot_holds_truncating_sums_in_any_order_and_refuses_a_lost_product_or_a_narrow_partial():
    """The native-dot contract (row 2084's link 1): a GEMM's output summed from its exact bf16 products in
    fp32 by truncating adds in any order, then rounded once to bf16 (or kept in fp32), holds at every
    element, same-sign products included, whose truncations all err one way. Each fault is refused where it
    lands (`assert_refused`): the same sums with their largest product lost, at more than 95% of the
    elements; and the first half's partial rounded to bf16 before an IEEE add of the rest's, as the native
    fault adds them, wherever that rounding outweighs the sum's own: at more than 65% of the elements of
    Gaussian draws, whose partials sometimes cancel to a bf16 value's neighbourhood, and more than 90% of
    draws of one sign."""
    rng = np.random.default_rng(3)
    for signed in (True, False):
        left, right = (np.asarray(ml_dtypes.bfloat16(rng.normal(size=shape) if signed else
                                                     rng.uniform(0.5, 2.0, size=shape)), np.float64)
                       for shape in ((16, 128), (4, 128)))
        products = nc.terms(_outer, left, right, fmt=oc.BFLOAT16)
        summed = _truncating_sums(products.exact, rng)
        for out, native in ((oc.FLOAT32, summed), (oc.BFLOAT16, summed.astype(ml_dtypes.bfloat16))):
            held = nc.native_dot_holds(np.asarray(native, np.float64), products, out, flushes=False)
            assert np.all(held), (signed, out.name)
        lost = _without(products, np.argmax(np.abs(products.exact), -1))
        first, rest = (_truncating_sums(half.exact, rng) for half in (products[..., :64], products[..., 64:]))
        narrow = np.float32(first.astype(ml_dtypes.bfloat16)) + np.float32(rest)
        narrowed = nc.narrow_partial_ends(products[..., :64], products[..., 64:], flushes=False)
        truth = nc.native_dot_ends(products, oc.FLOAT32, flushes=False)
        fault_lost = nc.native_dot_ends(lost, oc.FLOAT32, flushes=False)
        # One sign's partials never cancel near a bf16 value: the emulation's claim there is 0.9.
        for faulty, fault, share in ((_truncating_sums(lost.exact, rng), fault_lost, nc.LOST_SHARE),
                                     (narrow, narrowed, nc.NARROW_SHARE if signed else 0.9)):
            held = nc.native_dot_holds(np.asarray(faulty, np.float64), products, oc.FLOAT32, flushes=False)
            nc.assert_refused(held, nc.separated(truth, fault), share)


def test_a_tf32_operand_rounds_to_nearest_a_tie_away_from_zero():
    """TF32 keeps 11 of fp32's 24 bits, each value read here of either sign: 1 + 2^-11, a tie, reads as
    1 + 2^-10 (away from zero, where to even it is 1); a value just below the tie as 1; 1 + 3 2^-11, a tie at
    an odd last bit, as 1 + 2^-9 (to even too); and a value TF32 holds as itself. At its edges: the subnormal
    tie 2^-137 as TF32's least subnormal 2^-136, and an fp32 step below it as zero; 2^-126 - 2^-137, the tie
    below the least normal, as 2^-126; the binade's tie 2 - 2^-11 as 2; TF32's largest value, 2^128 - 2^117,
    as itself; and the tie past it, 2^128 - 2^116, as infinity, where an fp32 step below it reads as the
    largest."""
    largest, past = 2.0 ** 128 - 2.0 ** 117, 2.0 ** 128 - 2.0 ** 116
    cases = [(1 + 2 ** -11, 1 + 2 ** -10), (1 + 2 ** -11 - 2 ** -23, 1.0), (1 + 3 * 2 ** -11, 1 + 2 ** -9),
             (1 + 2 ** -10, 1 + 2 ** -10), (2.0 ** -137, 2.0 ** -136), (2.0 ** -137 - 2.0 ** -149, 0.0),
             (2.0 ** -126 - 2.0 ** -137, 2.0 ** -126), (2 - 2.0 ** -11, 2.0), (largest, largest),
             (past, np.inf), (past - 2.0 ** 104, largest)]
    values, expected = (np.asarray([[sign * case[side] for case in cases] for sign in (1.0, -1.0)])
                        for side in (0, 1))
    read = nc.tf32_operand(np.float32(values))
    assert np.array_equal(read, expected) and np.array_equal(np.signbit(read), np.signbit(expected))


def test_an_operand_read_as_tf32_is_enclosed_under_either_tie_and_truncation():
    """An fp32 value a tensor core reads as TF32 from its register, no conversion stated
    (`interval_certificates.operand`'s 'read', warpgroup MMA's): its enclosure's ends lie on TF32's grid,
    within one TF32 step outside the value's, and hold the value rounded to nearest with a tie to even, with
    a tie away from zero, and truncated, at fp32 values across the range, both signs, ties, subnormals and
    the largest. TF32 is no array's format, and the enclosure once stepped through a NumPy dtype it lacks
    (KeyError 'tfloat32' on an H100)."""
    rng = np.random.default_rng(2084)
    drawn = np.float32(rng.standard_normal(4096) * 2.0 ** rng.integers(-140, 127, 4096)).astype(np.float64)
    edges = np.asarray([0.0, 1 + 2.0 ** -11, 1 + 3 * 2.0 ** -11, 2 - 2.0 ** -11, 2.0 ** -137, 2.0 ** -126,
                        2.0 ** 128 - 2.0 ** 117, 2.0 ** 128 - 2.0 ** 104])
    values = np.concatenate([drawn, edges, -edges])
    lane = ic.Lane(reach=1, flushes=False, block=0, wide=True, tf32="read")
    read = ic.operand(ic.point(values), lane)
    for end in read:
        on_grid = np.isinf(end) | (oc.truncated(end, oc.TFLOAT32) == end)
        assert on_grid.all()
    for rounding in (oc.nearest, oc.rounded_away, oc.truncated):
        as_read = rounding(values, oc.TFLOAT32)
        finite = np.isfinite(as_read)
        assert np.all((read.lo <= as_read) | ~finite) and np.all((as_read <= read.hi) | ~finite)
    step = oc.above(np.nextafter(np.abs(values), np.inf), oc.TFLOAT32) - oc.below(np.abs(values), oc.TFLOAT32)
    assert np.all(read.lo <= values) and np.all(values <= read.hi)
    assert np.all((read.hi - read.lo <= step) | np.isinf(read.hi) | np.isinf(read.lo))


def test_a_native_dot_holds_tf32_products_and_refuses_bf16_ones_tf32_ones_for_bf16_and_a_truncation():
    """The TF32 contract (row 2084's link 2): a GEMM of fp32 operands at the default precision, its products
    of their TF32 values (`tf32_operand`) summed in fp32 by truncating adds in any order, holds at every
    element; the products of its operands' bf16 values where TF32 is declared, its TF32 products where bf16
    ones are, and the products of its operands truncated to TF32 where the tensor cores round them are each
    refused where they land (`assert_refused`): the first two at more than 85% of the elements, the
    truncation at more than 75% of Gaussian draws, whose products' errors partly cancel, and at more than 95%
    of draws of one sign, whose products all err toward zero."""
    rng = np.random.default_rng(5)

    def bf16(value):
        return np.asarray(value.astype(ml_dtypes.bfloat16), np.float64)

    for signed in (True, False):
        left, right = (np.asarray((rng.normal(size=shape) if signed else rng.uniform(0.5, 2.0, size=shape))
                                  .astype(np.float32), np.float64) for shape in ((32, 128), (8, 128)))
        tf32, wide, truncated = (nc.terms(_outer, read(left), read(right), fmt=fmt) for read, fmt in (
            (nc.tf32_operand, oc.TFLOAT32), (bf16, oc.BFLOAT16),
            (lambda value: oc.truncated(value, oc.TFLOAT32), oc.TFLOAT32)))
        native = _truncating_sums(tf32.exact, rng)
        assert np.all(nc.native_dot_holds(native, tf32, oc.FLOAT32, flushes=False)), signed
        swapped, other, cut = nc.TF32_SHARES[signed]
        for faulty, declared, given, share in ((_truncating_sums(wide.exact, rng), tf32, wide, swapped),
                                               (native, wide, tf32, other),
                                               (_truncating_sums(truncated.exact, rng), tf32, truncated,
                                                cut)):
            held = nc.native_dot_holds(faulty, declared, oc.FLOAT32, flushes=False)
            nc.assert_refused(held, nc.separated(*(nc.native_dot_ends(products, oc.FLOAT32, flushes=False)
                                                   for products in (declared, given))), share)


def test_a_promoted_step_holds_each_partials_ieee_add_and_refuses_a_truncating_add_or_a_lost_product():
    """The exact-sums witness (row 2084's link 2): an fp32 accumulator to which each depth step's partial,
    summed from zero by truncating adds in any order, is added by an IEEE add holds at every step, on draws at
    one scale and falling 2^-12 past the first step. Each fault is refused where it lands (`assert_refused`):
    a step with its largest product lost, at more than 95% of that step's elements, and nowhere else; and,
    on a falling draw of one sign, the same steps' products added into the accumulator by truncating adds,
    as the tensor cores add without exact sums (`truncating_step_ends`), at more than 40% of the later
    steps' elements, about those whose exact sum lies past the midpoint between two fp32 values, which a
    truncation leaves below and a rounding above."""
    rng = np.random.default_rng(7)
    width = 16
    for signed in (True, False):
        left, right = (np.asarray(ml_dtypes.bfloat16(rng.normal(size=shape) if signed else
                                                     rng.uniform(0.5, 2.0, size=shape)), np.float64)
                       for shape in ((16, 64), (4, 64)))
        for fall in (1.0, 2.0 ** -12):
            products = nc.terms(_outer, left * np.where(np.arange(64) < width, 1.0, fall), right,
                                fmt=oc.BFLOAT16)
            held = _steps_held(_stepped_sums(products.exact, width, rng, exact=True), products, width)
            assert all(np.all(step) for step in held), (signed, fall)
        lost = _without(products, 2 * width + np.argmax(np.abs(products.exact[..., 2 * width:3 * width]), -1))
        states = _stepped_sums(lost.exact, width, rng, exact=True)
        held = _steps_held(states, products, width)
        apart = _steps_apart(states, products, width, lambda before, step, lost=lost: nc.promoted_step_ends(
            before, before, lost[..., step], flushes=False))
        nc.assert_refused(held[2], apart[2], nc.LOST_SHARE)
        assert all(np.all(held[j]) for j in (0, 1, 3)), signed
        if not signed:
            # A truncation's bound toward zero holds where the accumulator and every product share a sign.
            states = _stepped_sums(products.exact, width, rng, exact=False)
            held = _steps_held(states, products, width)
            apart = _steps_apart(states, products, width, lambda before, step, terms=products: (
                nc.truncating_step_ends(before, terms[..., step], flushes=False)))
            assert not np.any(np.stack(held) & np.stack(apart))
            nc.assert_refused(np.stack(held[1:]), np.stack(apart[1:]), nc.TRUNCATED_SHARE)


@INTERPRETS
@pytest.mark.parametrize("regime", rw.REGIMES)
def test_the_recurrence_step_witness_holds_the_interpreted_mainloop(regime):
    """Check 2's step witness of the recurrence template (row 2084, link 3) on Mosaic GPU's interpreter, the
    mainloop's own code on the CPU's IEEE dots: every record of a bf16 call of three chunks, its inverses,
    weighted keys, corrected values, outputs and stored and fp32 memories, lies inside its launch step's
    enclosure on the native records the step reads (`recurrence_witness.enclosed`), each chunk's from the
    fp32 memory it enters, under each regime of decay (`REGIMES`); the certify build is the production
    program bit for bit (`recorded`), and each chunk's stored entering memory is the fp32 memory the chunks
    before it leave, rounded as stored (end_j == start_{j+1})."""
    run = rw.recorded(*rw.draw(0, regime))
    held = rw.witness(run, rw.interpreted_lane())
    for name in rw.RECORDS:
        assert np.all(held[name]), (name, np.mean(held[name]))


@INTERPRETS
@pytest.mark.parametrize("begins", [1.0, 1.25], ids=["on the grid", "inside a chunk"])
def test_the_recurrence_step_witness_holds_a_packed_call(begins):
    """The step witness (row 2084 (e)) on a packed bf16 call of three chunks whose second document begins at
    the second chunk or a quarter into it: the launches lay each document's chunks (`rw.laid_chunks`), its
    first chunk entering the zero memory a document starts from, held as stored, and every record lies
    inside its step's enclosure."""
    import jax.numpy as jnp

    operands = rw.draw(0, "mild")
    chunk = rw.recorded(*operands, carried=False).chunk
    starts = jnp.zeros((1, operands[0].shape[1]), bool).at[0, int(begins * chunk)].set(True)
    run = rw.recorded(*operands, starts=starts, max_segments=2)
    held = rw.witness(run, rw.interpreted_lane())
    for name in rw.RECORDS:
        assert np.all(held[name]), (name, np.mean(held[name]))


@INTERPRETS
def test_the_recurrence_step_witness_holds_a_decay_a_key_dimension_and_refuses_one_a_head():
    """The step witness (row 2084 (e)) under a decay a key dimension (`rw.assert_key_decays_witnessed`):
    every record of a bf16 call of three chunks inside its step's enclosure, and a kernel reading each
    token's mean decay for all its keys refused."""
    rw.assert_key_decays_witnessed(rw.interpreted_lane())


@INTERPRETS
def test_the_recurrence_step_witness_holds_a_write_gate_on_the_keys_side_and_refuses_one_dropped():
    """The step witness (row 2084 (e)) where the write gate takes the keys' side, Mamba2's SSD under the fast
    forms (`rw.assert_keys_gated_witnessed`): every record a bf16 call of three chunks keeps inside its step's
    enclosure, and a kernel reading every write gate as one refused."""
    rw.assert_keys_gated_witnessed(rw.interpreted_lane())


@INTERPRETS
def test_the_recurrence_step_witness_holds_values_passed_in_parts_and_refuses_them_rounded():
    """The step witness (row 2084 (c), (d), (e)) where the launches hold the values they pass in parts
    (`rw.assert_parts_witnessed`): an fp32 call with the delta rule and without it, and a bf16 call without it
    unrounded (Lightning's), every record each keeps inside its step's enclosure; the fp32 call run in bf16
    refused."""
    rw.assert_parts_witnessed(rw.interpreted_lane())


@INTERPRETS
def test_the_recurrence_step_witness_holds_the_fused_inference_walk_and_refuses_a_rounded_carry():
    """The step witness (row 2084 (e)) of the fused inference walk (`rw.assert_walk_witnessed`): a bf16 call
    no gradient follows, with the delta rule and without it, every record it keeps inside its step's
    enclosure, and the walk from a memory rounded to bf16 where fp32 is held refused."""
    rw.assert_walk_witnessed(rw.interpreted_lane())


@INTERPRETS
def test_the_recurrence_step_witness_holds_the_delta_rules_fast_backward():
    """The step witness (row 2084 (e)) of the delta rule's fast backward (`rw.assert_backward_witnessed`):
    every record each of its launches keeps, for a bf16 call of three chunks, inside its step's enclosure."""
    rw.assert_backward_witnessed(rw.interpreted_lane())


@INTERPRETS
def test_the_recurrence_step_witness_holds_a_decay_a_key_dimensions_fast_backward():
    """The step witness (row 2084 (e)) of a decay a key dimension's fast backward
    (`rw.assert_key_backward_witnessed`): every record each launch keeps inside its step's enclosure."""
    rw.assert_key_backward_witnessed(rw.interpreted_lane())


@INTERPRETS
def test_the_recurrence_step_witness_refuses_a_dropped_carry_and_a_state_rounded_to_bf16():
    """The step witness's faults on the interpreted mainloop, each refused where it lands
    (`rw.assert_carry_faults_refused`): a dropped carry and a state rounded to bf16 where fp32 is held."""
    rw.assert_carry_faults_refused(rw.interpreted_lane())


@INTERPRETS
def test_the_recurrence_step_witness_refuses_rounded_cumulative_sums_and_an_off_by_one_boundary():
    """The step witness's faults on the interpreted mainloop, each refused where it lands
    (`rw.assert_step_faults_refused`): decays from a difference of fp32 cumulative sums (row 2066's defect)
    and each chunk's boundary a token early."""
    rw.assert_step_faults_refused(rw.interpreted_lane())
