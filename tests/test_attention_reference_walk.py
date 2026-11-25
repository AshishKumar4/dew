"""The reference attention as a kernel's portable rule states its walk (`reference_attention`'s
`query_scale`, `logit_floor` and `skipped`; Splash's `_splash_portable`, row 12848)."""

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.attention import reference_attention


def _calls(seed: int = 0):
    """A call of one row, four queries over four keys, one head eight wide, fp32."""
    query, key, value = (jax.random.normal(stream, (1, 4, 1, 8), jnp.float32)
                         for stream in jax.random.split(jax.random.key(seed), 3))
    return query, key, value


def test_a_walks_starting_maximum_weighs_logits_far_beneath_it_zero():
    """Logits of -200 under a walk's starting maximum of 0 (Splash's mask value, its walk's start) weigh
    nothing, every output zero, where a softmax from their own maximum takes their mean."""
    query = jnp.full((1, 4, 1, 8), 5.0)
    key = jnp.full((1, 4, 1, 8), -5.0)
    value = _calls()[2]
    floored = reference_attention(query, key, value, query_scale=1.0, logit_floor=0.0)
    unfloored = reference_attention(query, key, value, query_scale=1.0)
    np.testing.assert_array_equal(np.asarray(floored), np.zeros_like(floored))
    assert np.abs(np.asarray(unfloored)).max() > 0


def test_skipped_keys_non_finite_values_and_tangents_reach_no_output_or_derivative():
    """Keys a walk skips (two query blocks of two rows, the last two keys skipped by both) holding NaN
    keys, values and tangents leave the output, its tangent and its query gradient those of the visited
    keys alone."""
    query, key, value = _calls()
    skipped = jnp.asarray([[[False, False, True, True], [False, False, True, True]]])
    poisoned_key, poisoned_value = (array.at[:, 2:].set(jnp.nan) for array in (key, value))

    def attend(q, k, v):
        return reference_attention(q, k, v, skipped=skipped)

    out = attend(query, poisoned_key, poisoned_value)
    visited = reference_attention(query, key[:, :2], value[:, :2])
    np.testing.assert_array_equal(np.asarray(out), np.asarray(visited))
    tangents = tuple(jnp.ones_like(array).at[:, 2:].set(jnp.nan) for array in (key, value))
    _, tangent = jax.jvp(lambda k, v: attend(query, k, v), (poisoned_key, poisoned_value), tangents)
    gradient = jax.grad(lambda q: attend(q, poisoned_key, poisoned_value).sum())(query)
    assert np.isfinite(np.asarray(tangent)).all() and np.isfinite(np.asarray(gradient)).all()


def test_a_query_block_every_key_of_which_is_skipped_weighs_nothing():
    """A query block whose every key the walk skips outputs zero, never NaN, beside a block that visits."""
    query, key, value = _calls()
    skipped = jnp.asarray([[[False, False, False, False], [True, True, True, True]]])
    out = np.asarray(reference_attention(query, key, value, skipped=skipped))
    np.testing.assert_array_equal(out[:, 2:], np.zeros_like(out[:, 2:]))
    np.testing.assert_array_equal(out[:, :2], np.asarray(reference_attention(query, key, value))[:, :2])
