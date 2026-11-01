"""The algebra pass where Dew compiles its programs (docs/design/algebra.md §2, §6.7): every outer program and
the resident training step jit their function rewritten, under the switch the gate opens per arm (`passes`);
the bodies of nested programs, scans, loops, conditionals and checkpoints are rewritten in place; products
(`dot_general`) are terms the rules read; and each rewrite is recorded in the one kernel report."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, equations

from dew import algebra
from dew.nn.kernels import options
from dew.nn.kernels.templates import report


def rows(seed: int, *shapes) -> tuple[jax.Array, ...]:
    """Rows of `shapes` drawn from `seed`, in float32."""
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.standard_normal(shape), jnp.float32) for shape in shapes)


def two_sums(x, y, a):
    """Two reductions of one scale's products, which linearity joins into one pass over x + y."""
    return jnp.sum(a * x, axis=-1) + jnp.sum(a * y, axis=-1)


def reductions(closed) -> int:
    """The sums a program's jaxpr holds, its bodies' included."""
    return sum(equation.primitive.name == "reduce_sum" for equation in equations(closed))


def nested(kind: str):
    """`two_sums` inside a body of `kind`: a nested jit, a checkpoint, a scan over two steps, a conditional,
    a loop of one turn."""
    def body(x, y, a):
        if kind == "jit":
            return jax.jit(two_sums)(x, y, a)
        if kind == "checkpoint":
            return jax.checkpoint(two_sums)(x, y, a)
        if kind == "scan":
            return jax.lax.scan(lambda carry, step: (carry + two_sums(x * step, y, a), None),
                                jnp.zeros(x.shape[:-1], x.dtype), jnp.arange(2.0, dtype=x.dtype))[0]
        if kind == "cond":
            return jax.lax.cond(x[0, 0] > 0, two_sums, lambda x, y, a: two_sums(y, x, a), x, y, a)
        return jax.lax.while_loop(lambda state: state[0] < 1,
                                  lambda state: (state[0] + 1, two_sums(x, y, a)),
                                  (0, jnp.zeros(x.shape[:-1], x.dtype)))[1]
    return body


KINDS = ("jit", "checkpoint", "scan", "cond", "while")


@pytest.mark.parametrize("kind", KINDS)
def test_a_bodys_math_is_rewritten_in_place(kind):
    """Two sums inside a nested jit, a checkpoint, a scan, a conditional or a loop are joined into one inside
    that body, the body kept a body (never unrolled), and the program is on par with the unrewritten one
    against the float64 truth (check 1)."""
    fn = nested(kind)
    x, y, a = rows(0, (8, 64), (8, 64), (8, 64))
    before, after = jax.make_jaxpr(fn)(x, y, a), jax.make_jaxpr(algebra.rewritten(fn))(x, y, a)
    branches = 2 if kind == "cond" else 1
    assert (reductions(before), reductions(after)) == (2 * branches, branches)
    assert [equation.primitive for equation in after.jaxpr.eqns] == [
        equation.primitive for equation in before.jaxpr.eqns]

    def draw(seed):
        return rows(seed, (8, 64), (8, 64), (8, 64))

    assert_as_exact(jax.jit(algebra.rewritten(fn)), jax.jit(fn), draw, f"two sums in a {kind}", seeds=16)


def test_products_with_a_shared_operand_join_into_one():
    """Two products with one shared operand (`dot_general`), summed, become one product of the shared operand
    with the summed other: a contraction is bilinear, so the rewritten program reads the shared rows once."""
    def contracted(a, x, y):
        return jnp.einsum("rk,rk->r", a, x) + jnp.einsum("rk,rk->r", a, y)

    a, x, y = rows(1, (8, 64), (8, 64), (8, 64))
    after = jax.make_jaxpr(algebra.rewritten(contracted))(a, x, y)
    assert sum(equation.primitive.name == "dot_general" for equation in equations(after)) == 1

    def draw(seed):
        return rows(seed, (8, 64), (8, 64), (8, 64))

    assert_as_exact(jax.jit(algebra.rewritten(contracted)), jax.jit(contracted), draw, "two products",
                    seeds=16)


def test_every_outer_program_is_rewritten_unless_its_arm_turns_the_pass_off():
    """An outer program (`options.program`) traces its function rewritten, so its lowering, its trace and its
    executable read one program; under `algebra.passes(on=False)`, which the gate opens for a reference arm,
    the same program traces unrewritten, and a program already held retraces as the switch changes."""
    x, y, a = rows(2, (8, 64), (8, 64), (8, 64))
    program = options.program(two_sums)
    assert reductions(program.trace(x, y, a).jaxpr) == 1
    with algebra.passes(on=False):
        assert reductions(program.trace(x, y, a).jaxpr) == 2
        off = program.executable(x, y, a)
    on = program.executable(x, y, a)
    assert off is not on
    np.testing.assert_array_equal(np.asarray(program(x, y, a)), np.asarray(on(x, y, a)))


def test_a_rewrite_is_recorded_in_the_kernel_report():
    """Each program the pass rewrites is recorded in the one kernel report (`report.rewrites`): the equations
    it read, the rule sets that fired and the nodes they introduced, the region's price before and after, and
    the pass's seconds."""
    x, y, a = rows(3, (8, 64), (8, 64), (8, 64))
    with report.refusals():
        options.program(two_sums).compiled(x, y, a)
        [made] = report.rewrites()
    assert made.fired.get("linear", 0) > 0
    assert made.after < made.before
    assert made.seconds > 0
