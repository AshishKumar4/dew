"""`rounding_bound.rounding_scale`: a JAX program's rounding scale through the calls it makes."""

import jax
import jax.numpy as jnp
import numpy as np
from rounding_bound import rounding_scale

from dew.nn.kernels.generation import linear_map


def test_a_program_through_a_chained_linear_map_is_scaled_as_its_lowering():
    """A program whose product is Dew's chained linear map (`generation.linear_map`, one hijax primitive
    call) is weighed through the program it lowers to: its float64 value the product's, every element of
    it rounding, as the plain product's does, where the walk refused the call (row 9780: Gemma 3n's
    gradients ran through one)."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((4, 8)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((8, 3)), jnp.float32)

    def mapped(x, w):
        return linear_map(lambda residuals, moving: moving @ residuals,
                          lambda residuals, cotangent: cotangent @ residuals.T, w, x)

    value, scale = rounding_scale(mapped, x, w)
    plain, plain_scale = rounding_scale(lambda x, w: x @ w, x, w)
    np.testing.assert_array_equal(value, plain)
    assert np.all(scale > 0) and np.all(plain_scale > 0)
    gradient_value, gradient_scale = rounding_scale(jax.grad(lambda x, w: jnp.sum(mapped(x, w) ** 2)), x, w)
    np.testing.assert_array_equal(gradient_value, rounding_scale(
        jax.grad(lambda x, w: jnp.sum((x @ w) ** 2)), x, w)[0])
    assert np.all(gradient_scale > 0)
