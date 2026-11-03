"""A carried function split by extraction (`dew.algebra.carry.split`, docs/design/algebra.md §6.9): its
prepared values, its carry and its emitted outputs together compute the function, against the float64 truth
(check 1), whatever its products' accumulation, its sums' broadcasting and its operands' ranks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.algebra import SATURATION, carry
from dew.nn.precision import FP32_SUMS


def split_run(function, carried):
    """`function(h, *operands)` through its split: the prepared values from the operands, then the carried
    and the emitted outputs from them and the state, in the function's order."""
    def run(h, *values):
        made = carry.split(function, jax.ShapeDtypeStruct(h.shape, h.dtype),
                           [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values], carried,
                           saturation=SATURATION)
        prepared = made.prepare(*values)
        carried_outputs, emitted = iter(made.carry(prepared, h)), iter(made.emit(prepared, h))
        return tuple(next(carried_outputs) if index in carried else next(emitted)
                     for index in range(len(jax.eval_shape(function, h, *values))))

    return run


def widened(h, a, b):
    """Two products of float16 matrices with the state, each accumulated in float32, summed."""
    return (jnp.matmul(a, h, preferred_element_type=jnp.float32)
            + jnp.matmul(b, h, preferred_element_type=jnp.float32),)


def test_products_accumulated_wide_keep_their_accumulation_through_the_split():
    """Float16 matrices near half float16's largest, each product with the state accumulated in float32:
    their sum is finite where adding the matrices first in float16 overflows; the split computes the
    function."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.uniform(0.5, 1.0, (3, 2)), jnp.float16),
                *(jnp.asarray(rng.uniform(30000, 40000, (3, 3)), jnp.float16) for _ in range(2)))

    assert_as_exact(jax.jit(split_run(widened, (0,))), jax.jit(widened), draw, "widened products",
                    seeds=8)


def broadcast_sum(h, a, b, c):
    """A product of a matrix with a row broadcast onto the state's product: `a @ (b + c @ h)`."""
    return (a @ (b + c @ h),)


def test_a_product_over_a_broadcast_sum_computes_through_the_split():
    """`a @ (b + c @ h)` with `b` one row broadcast across the sum: the split computes the function."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.standard_normal((3, 2)), jnp.float32),
                jnp.asarray(rng.standard_normal((3, 3)), jnp.float32),
                jnp.asarray(rng.standard_normal((1, 2)), jnp.float32),
                jnp.asarray(rng.standard_normal((3, 3)), jnp.float32))

    assert_as_exact(jax.jit(split_run(broadcast_sum, (0,))), jax.jit(broadcast_sum), draw,
                    "a broadcast sum's product", seeds=8)


def ranked(h, a, b, d):
    """A rank-three operand's product with a matrix product of the state, and a matrix's product with the
    state, each contracting its left operand's axis 1 with its right's axis 0."""
    contracted = (((1,), (0,)), ((), ()))
    return (jax.lax.dot_general(a, jax.lax.dot_general(b, h, contracted), contracted),
            jax.lax.dot_general(d, h, contracted))


def test_products_of_one_contraction_over_operands_of_other_ranks_compute_through_the_split():
    """Products sharing one contraction's parameters, one over a rank-three operand and one over matrices:
    the split computes both outputs."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.standard_normal(shape), jnp.float32)
                     for shape in ((5, 6), (2, 3, 4), (3, 5), (7, 5)))

    assert_as_exact(jax.jit(split_run(ranked, (0, 1))), jax.jit(ranked), draw, "ranked products",
                    seeds=8)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("overflow", ["product", "sum"])
def test_association_keeps_finite_carried_contraction_products(compiled, overflow):
    """The prepared product x@y is finite, but its contraction with the state overflows a term product or
    a partial sum where the declared grouping cancels exactly before reading x."""
    length = 2 if overflow == "product" else 4

    def associated(h):
        x = jnp.ones((1, 2), jnp.float32)
        y = jnp.ones((2, length), jnp.float32)
        return (x @ (y @ h),)

    magnitude = 2.0 ** (127 if overflow == "product" else 126)
    h = jnp.repeat(jnp.array([magnitude, -magnitude], jnp.float32), length // 2)
    h = jnp.broadcast_to(h[:, None], (length, 128))
    original, rewritten = associated, split_run(associated, (0,))
    if compiled:
        original, rewritten = jax.jit(original), jax.jit(rewritten)
    with jax.default_matmul_precision(FP32_SUMS):
        expected, = original(h)
        np.testing.assert_array_equal(expected, jnp.zeros_like(expected))
        actual, = rewritten(h)
        np.testing.assert_array_equal(actual, expected)
