"""Check 2 as a gate (tests/op_certificates.py `certified`, `assert_certified`): a computation run one
primitive at a time, natively, each equation held to its own operation's contract, as the CPU lane
computes (its subnormals flushed), and a primitive without a contract refused."""

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
    assert oc.FLUSHES


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
    contract = oc.CONTRACTS[name]
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
        assert np.all(oc.CONTRACTS["reduce_sum"](params, [terms], native, oc.FLOAT32))
    lost = sequential.copy()
    lost[0] = np.float32(0.0)
    assert not oc.CONTRACTS["reduce_sum"](params, [terms], lost, oc.FLOAT32)[0]


def test_a_dot_holds_its_bound_and_an_element_twice_past_it_breaks_it():
    rng = np.random.default_rng(3)
    a, b = np.float32(rng.standard_normal((5, 300))), np.float32(rng.standard_normal((300, 4)))
    params = {"dimension_numbers": (((1,), (0,)), ((), ()))}
    native = np.asarray(jax.jit(lambda a, b: jnp.dot(a, b, precision="highest"))(a, b))
    assert np.all(oc.CONTRACTS["dot_general"](params, [a, b], native, oc.FLOAT32))
    off = native.copy()
    off[1, 2] += 2 * oc.gamma(300, oc.FLOAT32) * float(np.abs(a[1]) @ np.abs(b[:, 2]))
    held = oc.CONTRACTS["dot_general"](params, [a, b], off, oc.FLOAT32)
    assert not held[1, 2] and held.sum() == held.size - 1


def test_a_computation_is_certified_through_its_calls_and_a_primitive_without_a_contract_refused():
    """`certified` runs a jitted call's and a cond's bodies one primitive at a time, and finds the
    primitive no contract covers, by its path."""
    def computed(x, flag):
        inner = jax.jit(lambda x: jnp.log(x * x + 1))(x)
        return jax.lax.cond(flag, lambda v: v * 2, lambda v: v - 1, inner)

    x = np.float32(np.linspace(0.1, 3, 32))
    outputs, findings = oc.certified(computed, x, np.bool_(1))
    assert findings == []
    np.testing.assert_allclose(outputs, np.log(x.astype(np.float64) ** 2 + 1) * 2, rtol=1e-5)
    _, findings = oc.certified(lambda x: jnp.sin(x), x)
    assert len(findings) == 1 and "no contract for sin" in findings[0]
    with pytest.raises(AssertionError):
        oc.assert_certified(lambda x: jnp.sin(x), lambda seed: (x,), "sine", seeds=1)


def test_the_lanes_unfaithful_operations_are_findings_with_their_paths():
    """A transcendental more than its two neighbours from the exact value breaks its contract whatever
    its vendor documents (erf_inv as the lane computes it, inside `jax.random.normal`), and so does a
    division by a constant, which XLA computes as a product with the constant's reciprocal, not the
    correctly rounded quotient; a division by a value the program is given is correctly rounded."""
    rng = np.random.default_rng(4)
    u = np.float32(rng.uniform(-0.99, 0.99, 512))
    _, findings = oc.certified(jax.lax.erf_inv, u)
    assert len(findings) == 1 and findings[0].startswith("erf_inv#0")
    x = np.float32(rng.uniform(0.1, 3, 512))
    _, findings = oc.certified(lambda x: x / 3, x)
    assert len(findings) == 1 and findings[0].startswith("div#0")
    assert oc.certified(jnp.divide, x, np.float32(np.full(512, 3)))[1] == []
