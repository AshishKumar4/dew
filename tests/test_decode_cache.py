"""Per-row cache progression through real attention, including paused rows."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.nn.attention import (
    causal_attention_mask,
    open_kv_cache,
    scaled_dot_product_attention,
)


class CachedAttention(nn.Module):
    @nn.compact
    def __call__(self, values, valid):
        positions, append = open_kv_cache(self, values, 8, valid=valid)
        keys, cached_values = append(values, values)
        mask = causal_attention_mask(
            positions, keys.shape[1], key_valid=self.get_variable("cache", "cache_valid"))
        return scaled_dot_product_attention(values, keys, cached_values,
                                            implementation="xla", mask=mask)


def test_valid_rows_prefill_pause_and_resume_without_reading_padding():
    attention = CachedAttention()
    hidden = jax.random.normal(jax.random.key(0), (2, 6, 1, 4))
    padded = hidden[:, :3].at[0, :2].set(50.0)
    valid = jnp.array([[False, False, True], [True, True, True]])
    variables = attention.init(jax.random.key(1), padded[:, :1], valid[:, :1])
    apply = jax.jit(lambda state, x, live: attention.apply(state, x, live, mutable=["cache"]))
    actual, variables = apply(variables, padded, valid)
    histories = [hidden[0:1, 2:3], hidden[1:2, :3]]
    for row in range(2):
        expected = scaled_dot_product_attention(
            hidden[row:row + 1, 2:3], histories[row], histories[row], implementation="xla")
        np.testing.assert_allclose(actual[row, 2], expected[0, 0], atol=1e-6, rtol=1e-6)

    for token_index, active in [(3, [True, False]), (4, [True, True]), (5, [False, True])]:
        actual, variables = apply(variables, hidden[:, token_index:token_index + 1],
                                  jnp.asarray(active)[:, None])
        for row, live in enumerate(active):
            if not live:
                continue
            token = hidden[row:row + 1, token_index:token_index + 1]
            histories[row] = jnp.concatenate([histories[row], token], axis=1)
            expected = scaled_dot_product_attention(token, histories[row], histories[row],
                                                     implementation="xla")
            np.testing.assert_allclose(actual[row, 0], expected[0, 0], atol=1e-6, rtol=1e-6)
