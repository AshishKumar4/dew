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
from jax.ad_checkpoint import checkpoint_name


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
    ("add", jnp.add, 2), ("add_any", jnp.add, 2), ("sub", jnp.subtract, 2), ("mul", jnp.multiply, 2),
    ("div", jnp.divide, 2), ("sqrt", jnp.sqrt, 1), ("rsqrt", jax.lax.rsqrt, 1), ("log", jnp.log, 1),
    ("exp", jnp.exp, 1), ("logistic", jax.lax.logistic, 1)])
def test_each_contract_holds_the_lanes_own_operation_and_refuses_a_step_off_it(name, fn, operands):
    """The lane's own result of each operation on random float32 operands, subnormal products and
    quotients included, holds its contract at the lane's measured reach; the same result moved past
    that reach at an element (twice the reach's steps, past both neighbours of the exact value and every
    step the reach allows beyond them) breaks it there and nowhere else, and one step off breaks a
    correctly rounded operation's."""
    rng = np.random.default_rng(1)
    values = [np.float32(rng.uniform(0.01, 4, 256) * rng.choice([1, 2.0 ** -70], 256))
              for _ in range(operands)]
    if name in ("sqrt", "rsqrt", "log"):
        values = [np.abs(values[0])]
    result = np.asarray(jax.jit(fn)(*values))
    contract = ct.CONTRACTS[name]
    assert np.all(contract({}, values, result, oc.FLOAT32))
    # A cotangents' sum is held at add's reach, as it lowers as add does.
    reach = ct.lane().reach["add" if name == "add_any" else name, "float32"]
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


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_a_cumulative_sum_holds_each_prefixs_bound_and_a_lost_term_breaks_it(reverse):
    """A cumulative sum, forward and reversed, of float32 terms over seven binades and of integers that
    wrap: the lane's own holds at every prefix; one prefix missing its largest term breaks there; an integer
    one off breaks."""
    rng = np.random.default_rng(6)
    terms = np.float32(rng.normal(size=(3, 17)) * 2.0 ** rng.integers(-3, 4, (3, 17)))
    native = np.asarray(jax.lax.cumsum(terms, axis=1, reverse=reverse))
    params = {"axis": 1, "reverse": reverse}
    assert np.all(ct.CONTRACTS["cumsum"](params, [terms], native, oc.FLOAT32))
    taken = np.arange(8, 17) if reverse else np.arange(0, 9)
    lost = native.copy()
    lost[1, 8] -= terms[1, taken[np.argmax(np.abs(terms[1, taken]))]]
    assert not ct.CONTRACTS["cumsum"](params, [terms], lost, oc.FLOAT32)[1, 8]
    ints = np.int32([[2 ** 31 - 1, 1, 5], [-7, 3, 2]])
    native = np.asarray(jax.lax.cumsum(ints, axis=1, reverse=reverse))
    assert np.all(ct.CONTRACTS["cumsum"](params, [ints], native, None))
    assert not np.all(ct.CONTRACTS["cumsum"](params, [ints], native + np.int32(1), None))


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


def _first_equation(fn, operands, name):
    """The first `name` equation `fn` runs on `operands`, calls included: its params, the operands it was
    given and its results, the program run one equation at a time as `certified` runs it."""
    from jax.extend import core
    from reference_error import _body

    def run(jaxpr, consts, arguments):
        env = dict(zip(jaxpr.constvars, consts, strict=True))
        env.update(zip(jaxpr.invars, arguments, strict=True))

        def read(var):
            return np.asarray(var.val, var.aval.dtype) if isinstance(var, core.Literal) else env[var]

        for eqn in jaxpr.eqns:
            given, body = [read(var) for var in eqn.invars], _body(eqn)
            if body is not None:
                found = run(*body, given)
                if found is not None:
                    return found
                results = jax.core.eval_jaxpr(*body, *given)
            else:
                results = eqn.primitive.bind(*given, **eqn.params)
                results = list(results) if eqn.primitive.multiple_results else [results]
                if eqn.primitive.name == name:
                    return eqn.params, given, results
            env.update(zip(eqn.outvars, results, strict=True))
        return None

    closed = jax.make_jaxpr(fn)(*operands)
    found = run(closed.jaxpr, closed.consts, operands)
    assert found is not None, f"no {name} equation"
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


@pytest.mark.parametrize("name, fn, operands", [
    ("pad", lambda a: jax.lax.pad(a, np.float32(-3), ((1, 2, 0), (-1, 1, 2))),
     [np.float32(np.arange(12).reshape(3, 4))]),
    ("iota", lambda: jax.lax.broadcasted_iota(jnp.int32, (3, 5), 1), []),
    ("gather", lambda a, i: a[i], [np.float32(np.arange(20).reshape(5, 4)), np.int32([[4], [0], [2]])]),
    ("gather", lambda a, i: jnp.take_along_axis(a, i, axis=1, mode="clip"),
     [np.float32(np.arange(20).reshape(5, 4)), np.int32([[0, 9], [3, -2], [1, 1], [2, 0], [3, 3]])]),
    ("gather", lambda a, i: jnp.take(a, i, axis=0, mode="fill", fill_value=-1.0),
     [np.float32(np.arange(20).reshape(5, 4)), np.int32([1, 7, -1, 4])]),
    ("gather", lambda a, i: jax.vmap(lambda row, at: row[at])(a, i),
     [np.float32(np.arange(20).reshape(5, 4)), np.int32([3, 0, 1, 2, 3])]),
    ("gather", lambda a, i: jnp.take_along_axis(a, i, axis=1, mode="clip"),
     [np.arange(20).reshape(5, 4).astype(ml_dtypes.bfloat16),
      np.int32([[0, 9], [3, -2], [1, 1], [2, 0], [3, 3]])]),
    ("gather", lambda a, i: a[i],
     [np.arange(20).reshape(5, 4).astype(ml_dtypes.bfloat16), np.int32([[4], [0], [2]])]),
    ("stack", lambda a, b: jnp.stack([a, b], axis=1), [np.float32([1, 2, 3]), np.float32([4, 5, 6])]),
    ("optimization_barrier", lambda a, b: jax.lax.optimization_barrier((a, b)),
     [np.float32([1, 2]), np.float32([[3, 4, 5]])]),
    ("rev", lambda a: jax.lax.rev(a, (0, 1)), [np.float32(np.arange(12).reshape(3, 4))]),
    ("split", lambda a: jax.lax.split(a, (1, 3), axis=1), [np.float32(np.arange(12).reshape(3, 4))]),
    ("name", lambda a: checkpoint_name(a, "held"), [np.float32([1, 2, 3])]),
], ids=["pad", "iota", "gather", "gather clipped", "gather filled", "gather batched", "gather clipped bf16",
        "gather bf16", "stack", "barrier", "rev", "split", "name"])
def test_each_move_holds_its_own_result_and_refuses_one_element_off_it(name, fn, operands):
    """Every move check 2 certifies by a numpy oracle of XLA's semantics (`CONTRACTS`), padding with
    interior and negative edges, an iota, a gather in each mode and batched, a stack, an optimization
    barrier of two operands, a reversal, a split in two and a checkpoint name: the program certifies,
    its contract holding the lane's own result at every element, and the contract refuses that result
    with one element a step off."""
    assert ct.certified(fn, *operands)[1] == []
    params, given, results = _first_equation(fn, operands, name)
    contract = ct.CONTRACTS[name]
    for index, result in enumerate(results):
        native = np.asarray(result)
        # As `_held` hands it: a result of its own position reads its own part of the operands alone.
        operands = ct.RESULT_OPERANDS[name](params, given, index) if name in ct.RESULT_OPERANDS else given
        assert np.all(contract(params, operands, native, None)), name
        assert not np.all(contract(params, operands, _off_by_one_element(native), None)), name


def test_a_split_holds_each_piece_to_its_own_run():
    """A split into two pieces of one shape: its pieces in place hold, the two swapped break, though each
    is a run of the operand."""
    a = np.float32(np.arange(8).reshape(2, 4))

    def split(a):
        return jax.lax.split(a, (2, 2), axis=1)

    _, given, results = _first_equation(split, [a], "split")
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
    assert np.all(ct.CONTRACTS["rsqrt"]({}, [fours], roots, oc.FLOAT32))
    off = np.nextafter(roots, np.float32(np.inf))
    assert not np.any(ct.CONTRACTS["rsqrt"]({}, [fours], off, oc.FLOAT32))
    poles = np.float32([0.0, -0.0, np.inf])
    assert np.all(ct.CONTRACTS["rsqrt"]({}, [poles], np.float32([np.inf, -np.inf, 0.0]), oc.FLOAT32))
    assert not np.any(ct.CONTRACTS["rsqrt"]({}, [poles], np.float32([-np.inf, np.inf, 1e-30]), oc.FLOAT32))
    limits = np.float32([0.0, np.inf, -np.inf])
    assert np.all(ct.CONTRACTS["logistic"]({}, [limits], np.float32([0.5, 1.0, 0.0]), oc.FLOAT32))
    assert not np.any(ct.CONTRACTS["logistic"]({}, [limits],
                                               np.nextafter(np.float32([0.5, 1.0, 0.0]), np.float32(0.25)),
                                               oc.FLOAT32))


def test_a_barrier_holds_each_result_to_its_own_operand():
    """An optimization barrier of two operands of one shape: its results in place hold, the two swapped
    break, though each is one of its operands."""
    a, b = np.float32([1, 2]), np.float32([3, 4])
    _, given, results = _first_equation(lambda a, b: jax.lax.optimization_barrier((a, b)), [a, b],
                                        "optimization_barrier")
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
    _, findings = ct.certified(lambda x: jax.lax.scan(lambda c, row: (c + jnp.sin(row), None),
                                                      np.float32(np.zeros(3)), x), x)
    assert len(findings) == 4 and all("scan#" in finding and f"/{step}/sin" in finding
                                      for step, finding in enumerate(findings)), findings


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


def _truncating_sum(values: np.ndarray) -> np.float32:
    """`values` summed in fp32 in their order, each add truncated toward zero, as a tensor core adds."""
    total = np.float32(0)
    for value in values:
        exact = np.float64(total) + np.float64(value)
        stepped = np.float32(exact)
        if abs(np.float64(stepped)) > abs(exact):
            stepped = np.nextafter(stepped, np.float32(0))
        total = stepped
    return total


def test_a_native_dot_holds_truncating_sums_in_any_order_and_refuses_a_lost_product_or_a_narrow_partial():
    """The native-dot contract (row 2084's link 1): a GEMM's output summed from its exact bf16 products in
    fp32 by truncating adds in any order, then rounded once to bf16 (or kept in fp32), holds at every
    element, same-sign products included, whose truncations all err one way; the same sums with their
    largest product lost break it at nearly every element, and with the first half's partial rounded to
    bf16 wherever that rounding outweighs the sum's own (about six elements in seven on Gaussian draws,
    whose partials sometimes cancel to a bf16 value's neighbourhood; nearly all of one sign)."""
    rng = np.random.default_rng(3)
    for signed in (True, False):
        left, right = (np.asarray(ml_dtypes.bfloat16(rng.normal(size=shape) if signed else
                                                     rng.uniform(0.5, 2.0, size=shape)), np.float64)
                       for shape in ((16, 128), (4, 128)))
        products = left[:, None, :] * right[None, :, :]
        orders = [rng.permutation(128) for _ in range(products.shape[0] * products.shape[1])]
        flat = products.reshape(-1, 128)
        summed = np.asarray([_truncating_sum(row[order]) for row, order in zip(flat, orders, strict=True)])
        for out, native in ((oc.FLOAT32, summed), (oc.BFLOAT16, summed.astype(ml_dtypes.bfloat16))):
            held = ct.native_dot_holds(np.asarray(native, np.float64).reshape(products.shape[:2]), products,
                                       out, flushes=False)
            assert np.all(held), (signed, out.name)
        lost = np.asarray([_truncating_sum(np.delete(row, np.argmax(np.abs(row)))) for row in flat])
        narrow = np.asarray([_truncating_sum(np.concatenate([
            [np.float32(ml_dtypes.bfloat16(_truncating_sum(row[:64])))], row[64:]])) for row in flat])
        for faulty, most in ((lost, 0.05), (narrow, 0.25 if signed else 0.05)):
            held = ct.native_dot_holds(faulty.reshape(products.shape[:2]).astype(np.float64), products,
                                       oc.FLOAT32, flushes=False)
            assert np.mean(held) < most, (signed, np.mean(held))
