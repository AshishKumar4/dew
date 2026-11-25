"""The reference attention as a kernel's portable rule states its walk (`reference_attention`'s
`query_scale`, `logit_floor` and `skipped`; Splash's `_splash_portable`, row 12848)."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import FEWEST_SEEDS, assert_as_exact

from dew.nn.attention import reference_attention

SKIPPED = jnp.asarray([[[False, False, True, True], [False, False, True, True]]])
"""Two query blocks of two rows, the last two of four keys skipped by both."""


def _draw(seed: int):
    """One row of four queries over four keys, one head eight wide, fp32, and the keys' and values'
    tangents."""
    return tuple(jax.random.normal(stream, (1, 4, 1, 8), jnp.float32)
                 for stream in jax.random.split(jax.random.key(seed), 5))


def test_a_walks_starting_maximum_weighs_logits_far_beneath_it_zero():
    """Logits of -200 under a walk's starting maximum of 0 (Splash's mask value, its walk's start) weigh
    nothing, every output zero, where a softmax from their own maximum weighs them."""
    query = jnp.full((1, 4, 1, 8), 5.0)
    key = jnp.full((1, 4, 1, 8), -5.0)
    value = _draw(0)[2]
    floored = reference_attention(query, key, value, query_scale=1.0, logit_floor=0.0)
    np.testing.assert_array_equal(np.asarray(floored), np.zeros_like(floored))
    assert np.abs(np.asarray(reference_attention(query, key, value, query_scale=1.0))).max() > 0


def test_skipped_keys_non_finite_values_and_tangents_reach_no_output_or_derivative():
    """Keys a walk skips holding NaN keys, values and tangents leave the output, its tangent and its query
    gradient those of the visited keys alone (check 1 against the call over the visited keys)."""
    def poisoned(array):
        return array.at[:, 2:].set(jnp.nan)

    def walked(query, key, value, key_tangent, value_tangent):
        def attend(q, k, v):
            return reference_attention(q, k, v, skipped=SKIPPED)

        out, tangent = jax.jvp(lambda k, v: attend(query, k, v), (poisoned(key), poisoned(value)),
                               (poisoned(key_tangent), poisoned(value_tangent)))
        gradient = jax.grad(lambda q: attend(q, poisoned(key), poisoned(value)).sum())(query)
        return out, tangent, gradient

    def visited(query, key, value, key_tangent, value_tangent):
        def attend(q, k, v):
            return reference_attention(q, k[:, :2], v[:, :2])

        out, tangent = jax.jvp(lambda k, v: attend(query, k, v), (key, value), (key_tangent, value_tangent))
        return out, tangent, jax.grad(lambda q: attend(q, key, value).sum())(query)

    assert_as_exact(walked, visited, _draw, "a walk's skipped keys", seeds=FEWEST_SEEDS)


def test_a_query_block_every_key_of_which_is_skipped_weighs_nothing():
    """A query block whose every key the walk skips outputs zero, never NaN, beside a block that visits
    them all (check 1 against the call without skipping)."""
    skipped = jnp.asarray([[[False, False, False, False], [True, True, True, True]]])
    query, key, value = _draw(0)[:3]
    out = np.asarray(reference_attention(query, key, value, skipped=skipped))
    np.testing.assert_array_equal(out[:, 2:], np.zeros_like(out[:, 2:]))
    assert_as_exact(lambda q, k, v, *_: reference_attention(q, k, v, skipped=skipped)[:, :2],
                    lambda q, k, v, *_: reference_attention(q, k, v)[:, :2], _draw,
                    "a walk's visiting block", seeds=FEWEST_SEEDS)


def test_a_walk_beside_a_sink_of_minus_infinity_whose_every_key_is_masked_weighs_nothing():
    """A walk beside a sink of -inf whose every key a mask removes (-inf fill: Splash's segment ids masking
    a visited tile whole) outputs zero, never NaN (row 12848's review)."""
    query, key, value = _draw(0)[:3]
    mask = jnp.zeros((1, 1, 4, 4), bool)
    out = reference_attention(query, key, value, mask=mask, mask_fill=-jnp.inf,
                              sinks=jnp.full((1,), -jnp.inf), logit_floor=-jnp.inf)
    np.testing.assert_array_equal(np.asarray(out), np.zeros_like(out))


def test_a_grouped_query_walk_shares_its_keys_and_holds_on_par():
    """A grouped-query walk (four query heads over two key heads) whose skipped tiles every head shares
    takes its keys and values whole, grouped by key head, NaN keys skipped as ever, and holds on par
    (check 1) with the call over the visited keys alone (row 12848's review: its keys repeated a query head
    each)."""
    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 3)
        return (jax.random.normal(streams[0], (1, 4, 4, 8), jnp.float32),
                *(jax.random.normal(stream, (1, 4, 2, 8), jnp.float32) for stream in streams[1:]))

    def walked(query, key, value):
        poisoned = (array.at[:, 2:].set(jnp.nan) for array in (key, value))
        return reference_attention(query, *poisoned, skipped=SKIPPED)

    def visited(query, key, value):
        return reference_attention(query, key[:, :2], value[:, :2])

    assert_as_exact(walked, visited, draw, "a grouped-query walk", seeds=FEWEST_SEEDS)
