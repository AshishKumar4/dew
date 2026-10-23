"""Dew's CPU policy for float32 products (`dew.nn.precision.accumulated`): a CPU sums them in float64
and rounds once, through every product a layer forms, its gradient's and tangent's included, with x64
off; 16-bit products and other backends keep their own arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import with_gradients

from dew.nn.dense import Dense
from dew.nn.precision import float32_dot_general, float32_einsum, float32_matmul

pytestmark = pytest.mark.skipif(jax.default_backend() != "cpu", reason="the policy is the CPU's")

CANCELLING = np.asarray([2.0 ** 24, 1.0, -(2.0 ** 24)], np.float32)
"""Terms whose float32 sum, in the order given, loses the 1 (2^24 + 1 rounds to 2^24), and whose
float64 sum is 1 exactly."""


def products():
    """Each public product over `CANCELLING` against ones, as a [1, 3] by [3, 1] contraction."""
    lhs, rhs = CANCELLING[None], np.ones((3, 1), np.float32)
    return {
        "dot_general": lambda: float32_dot_general(lhs, rhs, (((1,), (0,)), ((), ()))),
        "einsum": lambda: float32_einsum("ik,kj->ij", lhs, rhs),
        "matmul": lambda: float32_matmul(lhs, rhs),
    }


@pytest.mark.parametrize("name", sorted(products()))
def test_a_float32_product_sums_in_float64_and_rounds_once(name):
    """2^24 + 1 - 2^24 is 1: the float64 sum rounded once, where float32 accumulation in that order
    gives 0; under jit and without, x64 off."""
    assert not jax.config.jax_enable_x64
    product = products()[name]
    assert np.asarray(product()).tolist() == [[1.0]]
    result = jax.jit(product)()
    assert result.dtype == jnp.float32 and np.asarray(result).tolist() == [[1.0]]


def test_a_layers_gradient_and_tangent_sum_in_float64():
    """A Dense layer forming its product as Dew's layers do (`float32_dot_general`): its kernel
    gradient sums the inputs' products over the batch in float64, and its output tangent sums the
    input tangent's, both 1 exactly over the cancelling terms."""
    layer = Dense(1, use_bias=False, dot_general=float32_dot_general)
    x = CANCELLING[:, None]
    params = {"params": {"kernel": np.ones((1, 1), np.float32)}}
    _, gradient = with_gradients(lambda params: layer.apply(params, x))(params, np.ones((3, 1), np.float32))
    assert np.asarray(gradient["params"]["kernel"]).tolist() == [[1.0]]
    rows = Dense(1, use_bias=False, dot_general=float32_dot_general)
    weights = {"params": {"kernel": np.ones((3, 1), np.float32)}}
    _, tangent = jax.jvp(lambda x: rows.apply(weights, x), (np.zeros((1, 3), np.float32),),
                         (CANCELLING[None],))
    assert np.asarray(tangent).tolist() == [[1.0]]


def test_16_bit_products_keep_their_float32_accumulation():
    """bfloat16 products, exact in float32, accumulate as the backend's dot accumulates them, and
    come back in the dtype asked."""
    lhs = jnp.asarray([[1.0, 2.0, 3.0]], jnp.bfloat16)
    rhs = jnp.ones((3, 1), jnp.bfloat16)
    dims = (((1,), (0,)), ((), ()))
    for preferred in (None, jnp.float32):
        ours = float32_dot_general(lhs, rhs, dims, preferred_element_type=preferred)
        theirs = jax.lax.dot_general(lhs, rhs, dims, preferred_element_type=preferred)
        assert ours.dtype == theirs.dtype
        np.testing.assert_array_equal(np.asarray(ours, np.float32), np.asarray(theirs, np.float32))


def test_a_float32_product_returns_the_dtype_asked():
    """A float32 product asked for a float32 result returns one, rounded once from the float64 sum."""
    lhs, rhs = CANCELLING[None], np.ones((3, 1), np.float32)
    dims = (((1,), (0,)), ((), ()))
    result = jax.jit(lambda: float32_dot_general(lhs, rhs, dims, preferred_element_type=jnp.float32))()
    assert result.dtype == jnp.float32 and float(result[0, 0]) == 1.0
