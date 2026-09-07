"""Per-row cache progression through real attention, including paused rows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.nn.attention import (
    causal_attention_mask,
    open_kv_cache,
    scaled_dot_product_attention,
)


from dew.nn.inputs import AttentionMetadata
from dew.nn.linear import GatedDeltaNet
from dew.nn.mla import MultiHeadLatentAttention


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





@pytest.mark.parametrize("kind", ["latent", "sparse", "recurrent"])
def test_mixer_prefill_and_resume_follow_each_rows_real_tokens(kind):
    if kind == "recurrent":
        module = GatedDeltaNet(emb_features=16, num_k_heads=2, num_v_heads=2,
                               head_k_dim=4, head_v_dim=4, conv_kernel=4,
                               max_seq_len=8, dtype=jnp.float32)
    else:
        sparse = dict(index_topk=3, index_n_heads=2, index_head_dim=8) if kind == "sparse" else {}
        module = MultiHeadLatentAttention(
            emb_features=16, num_heads=2, max_seq_len=8, q_lora_rank=8,
            kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
            dtype=jnp.float32, **sparse)
    hidden = jax.random.normal(jax.random.key(3), (2, 6, 16))
    params = module.init(jax.random.key(4), hidden)
    cache = module.apply(params, hidden[:, :1], decode=True, mutable=["cache"])[1]["cache"]
    apply = jax.jit(lambda state, x, valid: module.apply(
        {**params, "cache": state}, x, decode=True,
        attention_metadata=AttentionMetadata(valid=valid), mutable=["cache"]))
    prefill = hidden[:, :3].at[0, :2].set(50.0)
    actual, mutated = apply(cache, prefill, jnp.array([[False, False, True], [True, True, True]]))
    histories = [hidden[0:1, 2:3], hidden[1:2, :3]]
    for row in range(2):
        expected = module.apply(params, histories[row])
        np.testing.assert_allclose(actual[row, -1], expected[0, -1], atol=3e-6, rtol=3e-6)
    for index, valid in [(3, [True, False]), (4, [False, True]), (5, [True, True])]:
        actual, mutated = apply(mutated["cache"], hidden[:, index:index + 1],
                                 jnp.asarray(valid)[:, None])
        for row, active in enumerate(valid):
            if not active:
                continue
            histories[row] = jnp.concatenate([histories[row], hidden[row:row + 1, index:index + 1]], 1)
            expected = module.apply(params, histories[row])
            np.testing.assert_allclose(actual[row, 0], expected[0, -1], atol=3e-6, rtol=3e-6)

