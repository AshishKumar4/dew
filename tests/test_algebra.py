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
    """Two reductions of one scale's products, which linearity joins into one pass over x + y."""
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
    over drawn and adversarial operands, as `algebra.rules.checked` instantiates each."""
    for name, (left, right, draw) in algebra.rules.instances().items():
        assert_as_exact(jax.jit(right), jax.jit(left), draw, name, seeds=16)


def test_a_compensated_sum_keeps_its_residual():
    """An error-free transform's residual survives the rewrite (`dew.nn.precision.compensated_add`): the
    rewritten program still computes the low word reassociation would cancel."""
    def compensated(a, b):
        return precision.compensated_add((a, jnp.zeros_like(a)), (b, jnp.zeros_like(b)))[1]

    a, b = jnp.asarray([1e8], jnp.float32), jnp.asarray([1.0], jnp.float32)
    low = float(jax.jit(compensated)(a, b)[0])
    assert float(jax.jit(algebra.rewritten(compensated))(a, b)[0]) == low != 0.0


REWRITE = """
import json, sys
import jax, jax.numpy as jnp
from dew import algebra
from dew.nn import precision

def unit_backward(x, dy):
    _, pull = jax.vjp(precision.l2norm, x)
    return pull(dy)[0]

shape = jax.ShapeDtypeStruct((8, 64), jnp.float32)
lowered = jax.jit(algebra.rewritten(unit_backward)).lower(shape, shape)
print(json.dumps(lowered.as_text()))
"""
"""A fresh process lowering a unit norm's rewritten backward."""


def test_one_input_rewrites_to_one_program_in_any_process():
    """Two fresh processes of different string hashing rewrite one program to the same lowered program, as a
    pool of processes needs each to compile."""
    lowered = []
    for seed in ("1", "2"):
        done = subprocess.run([sys.executable, "-c", REWRITE], capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONHASHSEED": seed, "JAX_PLATFORMS": "cpu",
                                   "PYTHONPATH": str(SOURCE)})
        lowered.append(json.loads(done.stdout.strip().splitlines()[-1]))
    assert lowered[0] == lowered[1]
