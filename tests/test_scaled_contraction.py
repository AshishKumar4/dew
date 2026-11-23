"""The scaled contraction (`scaled_contraction_p`): a sum of contractions of products and quotients formed
apart from their binades, its every order the primitive again."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.kernels.templates.scaled_contraction import Form, contract

QUOTIENT = Form(("", ""), ((1, (((0, 1), (1, -1)),)),), "")
"""`x / y` of two scalars."""


def test_a_quotients_product_rule_lands_once():
    """`x / y` at `x = y = 2^-100` moved by `2^100` each: the quotient 1, its tangent 0, where each product
    rule term landed apart was `+inf` and `-inf` (NaN)."""
    tiny = jnp.float32(2.0 ** -100)
    out, tangent = jax.jvp(lambda x, y: contract(QUOTIENT, x, y), (tiny, tiny),
                           (jnp.float32(2.0 ** 100), jnp.float32(2.0 ** 100)))
    np.testing.assert_array_equal(np.asarray(out), 1.0)
    np.testing.assert_array_equal(np.asarray(tangent), 0.0)


def test_a_subnormal_quotient_keeps_its_bits():
    """`2^-127 / 2^-127` is 1, its operands read from their bits, where `frexp` on a process flushing
    subnormals read `0 / 0`."""
    sub = jnp.float32(2.0 ** -127)
    np.testing.assert_array_equal(np.asarray(contract(QUOTIENT, sub, sub)), 1.0)


def test_a_sums_gradient_reaches_the_letters_it_reduces():
    """The gradient of a sum over `i` (`i ->`) is ones along `i`: the transpose takes the letter no other
    operand reads by a unit operand, at first and second order."""
    form = Form(("i",), ((1, (((0, 1),),)),), "")
    x = jnp.arange(1.0, 4.0, dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(jax.grad(lambda v: contract(form, v))(x)), 1.0)
    hessian = jax.jacfwd(jax.grad(lambda v: contract(form, v) ** 2))(x)
    np.testing.assert_array_equal(np.asarray(hessian), 2.0)


def test_a_subscript_naming_a_letter_twice_is_refused():
    """`ii ->`, a trace, is no form of the sum's: refused by name."""
    with pytest.raises(ValueError):
        contract(Form(("ii",), ((1, (((0, 1),),)),), ""), jnp.ones((2, 2), jnp.float32))


def test_a_batched_contraction_is_each_examples():
    """`jax.vmap` of a matrix product's sum over quotients is each example's, a fresh letter on the batch."""
    form = Form(("ik", "i", "kj"), ((1, (((0, 1), (1, -1)), ((2, 1),))),), "ij")
    rng = np.random.default_rng(12388)
    a, d, b = (jnp.asarray(rng.normal(size=shape) + 3.0, jnp.float32)
               for shape in ((3, 2, 4), (3, 2), (3, 4, 5)))
    batched = jax.vmap(lambda *x: contract(form, *x))(a, d, b)
    for example in range(3):
        np.testing.assert_array_equal(np.asarray(batched[example]),
                                      np.asarray(contract(form, a[example], d[example], b[example])))
