"""Dew's algebraic rewrite of a program (`dew.algebra`, docs/design/algebra.md): a jaxpr held as an e-graph,
its rules applied, and the cheapest form under the cost model extracted, which computes the program's values.
The rewritten programs are held to the accuracy gate against the unrewritten ones (check 1), every rule to its
own left side, and the rewrite to one output for one input in any process."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.core import eval_jaxpr
from reference_error import assert_as_exact

from dew import algebra
from dew.nn import precision

SOURCE = Path(__file__).resolve().parents[1] / "src"


def rows(seed: int, *shapes) -> tuple[jax.Array, ...]:
    """Rows of `shapes` drawn from `seed`, in float32."""
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.standard_normal(shape), jnp.float32) for shape in shapes)


def unit_backward(x, dy):
    """A unit norm's backward written as autodiff writes it: the norm's inverse, then the inner product of the
    unit rows with the cotangent, each waiting on the one before."""
    _, pull = jax.vjp(precision.l2norm, x)
    return pull(dy)[0]


def two_sums(x, y, a):
    """Two reductions of one scale's products of bounded values (each a tanh), which linearity joins into one
    pass over their sum: the range facts bound that new sum, so the rewrite admits it (§6.3)."""
    x, y, a = jnp.tanh(x), jnp.tanh(y), jnp.tanh(a)
    return jnp.sum(a * x, axis=-1) + jnp.sum(a * y, axis=-1)


def test_a_rewritten_program_computes_what_the_program_computes():
    """A rewritten program is on par with the program it rewrites against the float64 truth (check 1), for
    a unit norm's backward and for two sums linearity joins."""
    def draw_unit(seed):
        return rows(seed, (8, 64), (8, 64))

    def draw_sums(seed):
        return rows(seed, (8, 64), (8, 64), (8, 64))

    for fn, draw in ((unit_backward, draw_unit), (two_sums, draw_sums)):
        assert_as_exact(jax.jit(algebra.rewritten(fn)), jax.jit(fn), draw, fn.__name__, seeds=16)


def test_linearity_joins_two_reductions_into_one_pass():
    """Two sums of one scale's products become one sum over the joined products: the rewritten program reads
    its rows in one reduction where the program read them in two."""
    x, y, a = rows(0, (8, 64), (8, 64), (8, 64))
    before = jax.make_jaxpr(two_sums)(x, y, a)
    after = jax.make_jaxpr(algebra.rewritten(two_sums))(x, y, a)

    def reductions(closed) -> int:
        return sum(equation.primitive.name == "reduce_sum" for equation in closed.jaxpr.eqns)

    assert (reductions(before), reductions(after)) == (2, 1)


def test_each_rule_holds_its_right_side_to_its_left_side():
    """Every registered rule's right side is on par with its left side against the float64 truth (check 1),
    over drawn and adversarial operands, as its set states its instances (`algebra.rules.instances`)."""
    for name, (left, right, draw) in algebra.rules.instances().items():
        assert_as_exact(jax.jit(right), jax.jit(left), draw, name, seeds=16)


def test_every_registered_rule_set_is_qualified_by_instances_of_its_own():
    """The qualification inventory is the registry's (`rules.instances`): every registered set's instances
    name it, every set has some, and a set registered with none is refused, so no set enters the pass
    unqualified and no instance stands for a rule the registry does not hold."""
    from dew.algebra import rules

    named = {name.split(": ", 1)[0] for name in rules.instances()}
    assert named == set(rules.RULES)
    with pytest.raises(ValueError):
        rules.register(rules.Rules.declared("unqualified", "exact", (), ""))


def test_each_rewrite_of_a_registered_set_is_declared_with_its_instance():
    """A registered set is declared as its rewrites (`rules.Family`), each its rules (every direction and
    layout of it) and its instance written as JAX, so no rule of the registry runs without a case of its own
    qualifying it: the inventory (`rules.instances`), which the accuracy test holds each right side to its
    left side over, is every family's."""
    from dew.algebra import rules

    declared = {f"{sets.name}: {family.name}" for sets in rules.RULES.values() for family in sets.families}
    assert set(rules.instances()) == declared
    assert all(family.rules for sets in rules.RULES.values() for family in sets.families)


def test_a_compensated_sums_residual_survives_a_rewritten_program():
    """What a user of an error-free transform relies on (`dew.nn.precision.compensated_add`): its low word,
    which reassociation would cancel, is what the program computes once rewritten and rewritten again. The
    protection itself is proven against a control the pass does rewrite
    (test_algebra_program.py's protected span)."""
    def compensated(a, b):
        return precision.compensated_add((a, jnp.zeros_like(a)), (b, jnp.zeros_like(b)))[1]

    a, b = jnp.asarray([1e8], jnp.float32), jnp.asarray([1.0], jnp.float32)
    low = float(jax.jit(compensated)(a, b)[0])
    for rewrite in (algebra.rewritten(compensated), algebra.rewritten(algebra.rewritten(compensated))):
        assert float(jax.jit(rewrite)(a, b)[0]) == low != 0.0


def test_a_complex_programs_literals_are_read_by_the_rules():
    """A program of complex values, its literals complex (`1 + 0j` a unit, as a complex weight's update
    writes them), rewritten: the facts its literals state are read whatever their dtype, and it computes
    what it computes, its sums joined, real and imaginary parts alike."""
    def complex_sums(x, y, a, u, v, b):
        xs, ys, scale = x + 1j * u, y + 1j * v, a + 1j * b
        summed = jnp.sum(scale * xs * (1.0 + 0j), axis=-1) + jnp.sum(scale * ys * (1.0 + 0j), axis=-1)
        return jnp.stack([summed.real, summed.imag])

    def draw(seed):
        return rows(seed, *[(8, 64)] * 6)

    assert_as_exact(jax.jit(algebra.rewritten(complex_sums)), jax.jit(complex_sums), draw, "complex sums",
                    seeds=16)


LARGEST = float(np.finfo(np.float32).max)


def filled(*given):
    """Float32 operands, each a shape and its fill or its rows."""
    return tuple(jnp.full(shape, value, jnp.float32) if np.ndim(value) == 0
                 else jnp.broadcast_to(jnp.asarray(value, jnp.float32), shape) for shape, value in given)


NEAR_LARGEST = {
    "a constant factor taken out of a sum": (
        lambda a, x, y: 0.5 * x + 0.5 * y,
        lambda: filled(((8,), 0.0), ((8,), 3e38), ((8,), 3e38))),
    "a product's operands summed first": (
        lambda a, x, y: a * x + a * y,
        lambda: filled(((8,), 0.25), ((8,), LARGEST * 0.6), ((8,), LARGEST * 0.6))),
    "two sums' terms summed first": (
        lambda a, x, y: a * (jnp.sum(x, -1) + jnp.sum(y, -1)),
        lambda: filled(((2,), 1.0), ((2, 2), [LARGEST * 0.6, -LARGEST * 0.6]),
                       ((2, 2), [LARGEST * 0.6, -LARGEST * 0.6]))),
    "two contractions' operands summed first": (
        lambda a, x, y: a @ x + a @ y,
        lambda: filled(((4, 4), 0.01), ((4, 4), LARGEST * 0.6), ((4, 4), LARGEST * 0.6))),
    "a scale moved onto an operand": (
        lambda a, x, y: (x @ a) * 2.0,
        lambda: filled(((4, 4), 1e-38), ((1, 4), LARGEST * 0.6), ((1,), 0.0))),
    "a factor taken out of a sum": (
        lambda a, x, y: jnp.sum(a[:, :1] * x, axis=-1),
        lambda: filled(((2, 4), 0.25), ((2, 4), LARGEST * 0.6), ((1,), 0.0))),
}
"""Programs a reordering rule rewrites whose rewritten intermediate would pass float32's largest where the
program's never does, each with its operands (a, x, y)."""


@pytest.mark.parametrize("name", list(NEAR_LARGEST))
def test_a_rewritten_program_is_finite_wherever_the_program_is(name):
    """A sum of products (by a constant, `0.5 x + 0.5 y` at 3e38, and by an operand), of sums, of
    contractions, a scale after a contraction and a factor in a sum, their operands near float32's largest:
    each program is finite; rewritten, it stays finite (bible §6: no infinity where the reference is
    finite), whether or not a rule fires."""
    program, given = NEAR_LARGEST[name]
    operands_ = given()
    assert np.all(np.isfinite(program(*operands_)))
    # The rewritten program's equations run one at a time, so no compiler's own reassociation (XLA's
    # algebraic simplifier factors a shared constant) stands in for the rewrite's.
    closed = jax.make_jaxpr(algebra.rewritten(program))(*operands_)
    assert np.all(np.isfinite(jax.core.eval_jaxpr(closed.jaxpr, closed.consts, *operands_)[0]))


def cos_at(values) -> jax.Array:
    """Float32 arguments whose `cos` is each of `values` exactly, as XLA computes it: the first of a fine
    grid of each, so a witness's precondition is a fact of the function itself."""
    grid = jnp.linspace(0.0, 1e-3, 1_000_001, dtype=jnp.float32)
    found = np.asarray(jnp.cos(grid))
    at = [int(np.argmax(found == np.float32(value))) for value in values]
    assert all(found[index] == np.float32(value) for index, value in zip(at, values, strict=True))
    return grid[jnp.asarray(at)]


def test_two_joined_sums_stay_finite_wherever_the_program_is():
    """Two sums of bounded terms beside a scale multiplied into two bounded terms: `2^126 cos(u)` and
    `2^126 cos(v)`, under 2^126 each, at `[2^126, 2^126]` and `[2^126 - 2^102, 2^126 - 2^103]` sum to
    float32's largest, while their terms summed first, `[2^127, 2^127 - 2^103]` once rounded, sum past it.
    The joined sum's bound counts its reduction's length, so it stays apart, finite as the program is (bible
    §6), whatever else the program rewrites."""
    def program(u, v, x, y, a):
        x, y, a = jnp.tanh(x), jnp.tanh(y), jnp.tanh(a)
        return jnp.sum(2.0 ** 126 * jnp.cos(u)) + jnp.sum(2.0 ** 126 * jnp.cos(v)), a * x + a * y

    u = jnp.zeros((2,), jnp.float32)
    v = cos_at([1.0 - 2.0 ** -24, 1.0 - 2.0 ** -23])
    x, y, a = rows(11, (8,), (8,), (8,))
    operands_ = (u, v, x, y, a)
    assert np.isfinite(program(*operands_)[0])
    closed = jax.make_jaxpr(algebra.rewritten(program))(*operands_)
    assert np.isfinite(eval_jaxpr(closed.jaxpr, closed.consts, *operands_)[0])


def count(closed, primitive: str) -> int:
    """The equations of `closed` binding `primitive`."""
    return sum(equation.primitive.name == primitive for equation in closed.jaxpr.eqns)


def test_sums_of_operands_of_other_shapes_stay_apart():
    """Two sums along one axis of operands of different extents are not joined: one sum of their terms would
    broadcast the narrower operand along the axis, or add operands of no common shape."""
    def apart(x, y):
        return jnp.sum(x, axis=1) + jnp.sum(y, axis=1)

    for shapes in (((2, 1), (2, 4)), ((2, 3), (2, 4))):
        x, y = rows(1, *shapes)
        closed = jax.make_jaxpr(algebra.rewritten(apart))(x, y)
        assert count(closed, "reduce_sum") == 2
        np.testing.assert_array_equal(np.asarray(algebra.rewritten(apart)(x, y)), np.asarray(apart(x, y)))


def test_distributivity_factors_a_shared_matrix_where_its_sum_is_bounded():
    """A matrix multiplied into two bounded terms (tanh's) that are then added is multiplied once into their
    sum, which the range facts bound (§6.3); over two unbounded terms, whose sum may pass float32's largest
    where the program's terms do not, it stays as the program wrote it. A shared elementwise scale is not the
    case: the sum of its two products is a multiply and a fused multiply-add (`rules._contracted`), as many
    operations as the factored sum's add and multiply, so no price falls."""
    def bounded(x, y, a):
        x, y, a = jnp.tanh(x), jnp.tanh(y), jnp.tanh(a)
        return a @ x + a @ y

    def free(x, y, a):
        return a @ x + a @ y

    x, y, a = rows(2, (64, 32), (64, 32), (8, 64))
    assert (count(jax.make_jaxpr(bounded)(x, y, a), "dot_general"),
            count(jax.make_jaxpr(algebra.rewritten(bounded))(x, y, a), "dot_general")) == (2, 1)
    assert count(jax.make_jaxpr(algebra.rewritten(free))(x, y, a), "dot_general") == 2


def test_a_row_scalar_leaves_a_sum_along_its_rows():
    """A per-row scale broadcast along a row and summed with it is taken out of the sum: the rewritten program
    multiplies the row sums, not the rows."""
    def weighted(x, s):
        return jnp.sum(jnp.tanh(s)[:, None] * jnp.tanh(x), axis=1)

    x, (s,) = rows(3, (8, 64))[0], rows(4, (8,))
    closed = jax.make_jaxpr(algebra.rewritten(weighted))(x, s)
    multiplied = [equation for equation in closed.jaxpr.eqns if equation.primitive.name == "mul"]
    assert [equation.outvars[0].aval.shape for equation in multiplied] == [(8,)]


def test_literals_keep_their_exact_values_and_types():
    """A program's literals pass through the rewrite as the program wrote them: integers, booleans and floats
    each in its own type and value, int32's least among them."""
    def literal(x, n):
        return (jnp.where(n > np.int32(2_000_000_011), x * 3.5, x - 1.25), n + np.int32(2_000_000_011),
                jnp.sum(n * np.int32(-(2 ** 31))))

    x, n = rows(5, (4,))[0], jnp.asarray([2_000_000_013, 1, 2_000_000_011, 7], jnp.int32)
    for given, got in zip(literal(x, n), jax.jit(algebra.rewritten(literal))(x, n), strict=True):
        assert given.dtype == got.dtype
        np.testing.assert_array_equal(np.asarray(given), np.asarray(got))


def test_a_values_products_by_two_literals_are_one_product_by_their_sum():
    """`a * 2 + a * 4`, whose only bounded terms are its literals, is `a * 6`: one product, exactly."""
    def doubled(a):
        return a * 2.0 + a * 4.0

    (a,) = rows(8, (8, 64))
    closed = jax.make_jaxpr(algebra.rewritten(doubled))(a)
    assert count(closed, "mul") == 1
    np.testing.assert_array_equal(np.asarray(jax.jit(algebra.rewritten(doubled))(a)), np.asarray(doubled(a)))


def test_products_that_widen_keep_their_widening():
    """Products that widen their operands (an explicit `out_dtype`) are not factored: the sum the factoring
    would add in the narrow dtype overflows where the program's sum of widened products does not."""
    def widened(a, x, y):
        return (jax.lax.mul(a, x, out_dtype=jnp.float32) + jax.lax.mul(a, y, out_dtype=jnp.float32))

    a, x = jnp.asarray(1.0, jnp.float16), jnp.asarray(40000.0, jnp.float16)
    assert float(jax.jit(algebra.rewritten(widened))(a, x, x)) == float(widened(a, x, x)) == 80000.0


def test_a_weak_literal_stays_weak():
    """A Python scalar the program adds stays weakly typed through the rewrite, so the result promotes with a
    narrower array as the program's does."""
    def plus_one(x):
        return x + 1.0

    rewritten = jax.jit(algebra.rewritten(plus_one))(1.0)
    assert (rewritten + jnp.ones(2, jnp.float16)).dtype == (plus_one(1.0) + jnp.ones(2, jnp.float16)).dtype


def test_custom_derivatives_survive_the_rewrite():
    """A program calling a function with a custom JVP or a custom VJP computes the same values rewritten, and
    the rewritten program differentiates through the same rule."""
    @jax.custom_vjp
    def clipped(x):
        return jnp.clip(x, -1.0, 1.0)

    clipped.defvjp(lambda x: (clipped(x), None), lambda _, g: (g,))
    (x,) = rows(6, (16,))
    for fn in (jax.nn.relu, clipped):
        np.testing.assert_array_equal(np.asarray(jax.jit(algebra.rewritten(fn))(x)), np.asarray(fn(x)))
    np.testing.assert_array_equal(np.asarray(jax.grad(lambda v: jnp.sum(algebra.rewritten(clipped)(v)))(x)),
                                  np.ones(16, np.float32))


def test_outputs_sharing_a_value_keep_it_shared():
    """A program returning a sum and that sum added to another keeps the shared sum, two reductions in all,
    where joining the second output's sums would leave the first's to compute apart."""
    def shared(x, y):
        first = jnp.sum(x, axis=-1)
        return first + jnp.sum(y, axis=-1), first

    x, y = rows(7, (8, 64), (8, 64))
    assert count(jax.make_jaxpr(algebra.rewritten(shared))(x, y), "reduce_sum") == 2


REWRITE = """
import json, sys
import jax, jax.numpy as jnp
from dew import algebra
from dew.nn.kernels.templates import report


def program(x, y, a, b):
    x, y, a = jnp.tanh(x), jnp.tanh(y), jnp.tanh(a)
    return jnp.sum(a * x, -1) + jnp.sum(a * y, -1), jnp.dot(x, y.T) * 0.5 + jnp.dot(b, y.T) * 0.5

shape = jax.ShapeDtypeStruct((8, 64), jnp.float32)
with report.refusals():
    lowered = jax.jit(algebra.rewritten(program)).lower(shape, shape, shape, shape)
    made = [{"nodes": rewrite.nodes, "iterations": rewrite.iterations, "spliced": rewrite.spliced,
             "fired": dict(rewrite.fired)} for rewrite in report.rewrites()]
print(json.dumps({"lowered": lowered.as_text(), "rewrites": made}))
"""
"""A fresh process lowering a rewritten program whose windows the engine saturates, extracts and splices:
two sums of a shared bounded factor, and two scaled contractions of one right operand."""


def test_one_input_rewrites_to_one_program_in_any_process():
    """Two fresh processes of different string hashing rewrite one program to the same lowered program,
    as a pool of processes needs each to compile, through the same saturation, extraction and splices: each
    window's e-graph as large, its rules fired as often and its forms landing alike."""
    found = []
    for seed in ("1", "2"):
        done = subprocess.run([sys.executable, "-c", REWRITE], capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONHASHSEED": seed, "JAX_PLATFORMS": "cpu",
                                   "PYTHONPATH": str(SOURCE)})
        found.append(json.loads(done.stdout.strip().splitlines()[-1]))
    assert found[0] == found[1]
    assert any(rewrite["nodes"] and rewrite["spliced"] for rewrite in found[0]["rewrites"])
