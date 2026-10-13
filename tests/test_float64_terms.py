"""What `reference_error.float64_terms` takes for a leaf's terms: the root sum
of squares of the terms its final sums add, carried into it, and nothing behind them."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import float64_terms


def loss_of(keys, query, values):
    scores = jnp.einsum("bd,bkd->bk", query, keys)
    return jnp.sum(jax.nn.softmax(scores, axis=-1)[..., None] * values)


def test_a_key_bias_gradient_sums_terms_that_cancel():
    """A key bias shifts every logit of a query's softmax alike, so its
    gradient is zero in exact arithmetic, while the cotangents it sums are
    not: its terms are their root sum of squares over the rows and keys, as
    the bias's gradient sums them."""
    rng = np.random.default_rng(0)
    query, keys, values = (rng.normal(size=shape) for shape in ((2, 4), (2, 6, 4), (2, 6, 3)))

    with jax.enable_x64(new_val=True):
        gradient, terms = float64_terms(jax.grad(lambda bias: loss_of(keys + bias, query, values)),
                                        jnp.zeros(4))
        cotangent = np.asarray(jax.grad(loss_of)(jnp.asarray(keys), query, values))
    np.testing.assert_allclose(gradient, 0, atol=1e-15)
    np.testing.assert_allclose(terms, np.sqrt(np.sum(cotangent**2, axis=(0, 1))), rtol=1e-12)
    assert np.all(terms > 1e-3)


def test_a_sum_of_one_sign_has_its_own_size_and_an_update_its_steps_terms():
    """A weight's gradient of positive inputs sums positive terms, whose root
    sum of squares is below the sum itself. Through a scan of momentum
    steps, the momentum's terms are its gradients' and a parameter's its
    updates', not the value it started at."""
    x = np.abs(np.random.default_rng(1).normal(size=(8, 3)))

    def gradient(weight):
        return jax.grad(lambda w: jnp.sum(x @ w))(weight)

    def steps(weight):
        def step(carry, _):
            weight, trace = carry
            trace = 0.9 * trace + gradient(weight)
            return (weight - 0.1 * trace, trace), None
        return jax.lax.scan(step, (weight, jnp.zeros_like(weight)), None, length=2)[0]

    with jax.enable_x64(new_val=True):
        weight = jnp.ones((3, 2))
        value, terms = float64_terms(gradient, weight)
        _, (moved_terms, trace_terms) = float64_terms(steps, weight)
    energy = np.sum(x**2, axis=0)[:, None] * np.ones((1, 2))
    np.testing.assert_allclose(terms, np.sqrt(energy), rtol=1e-12)
    assert np.all(terms < np.asarray(value))
    # The momentum's second value is 0.9 of the first gradient and all of the second.
    np.testing.assert_allclose(trace_terms, np.sqrt((0.81 + 1) * energy), rtol=1e-12)
    # The weight moved by 0.1 of each step's momentum; the value it started at carries none.
    np.testing.assert_allclose(moved_terms, np.sqrt(0.01 * (1 + 0.81 + 1) * energy), rtol=1e-12)
