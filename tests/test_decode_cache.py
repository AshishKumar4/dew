"""Per-row cache progression through real attention, including paused rows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import TINY_DECODER, rows_prefill_pause_and_resume

from dew.batch import coordinates
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.attention_masks import causal_attention_mask
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.kv_cache import cached_validity, open_kv_cache
from dew.nn.linear import GatedDeltaNet
from dew.nn.mla import MultiHeadLatentAttention


class CachedAttention(nn.Module):
    @nn.compact
    def __call__(self, values, valid):
        positions, append = open_kv_cache(self, values, 8, valid=valid)
        keys, cached_values = append(values, values)
        mask = causal_attention_mask(
            positions, keys.shape[1], key_valid=cached_validity(self, keys.shape[1]))
        return scaled_dot_product_attention(values, keys, cached_values,
                                            implementation="xla", mask=mask)


PREFILL_VALID = jnp.array([[False, False, True], [True, True, True]])
"""Two rows' prefill: the first holds one real token behind two padded ones."""


def test_valid_rows_prefill_pause_and_resume_without_reading_padding():
    """A raw attention cache: each row's output is attention over its own real tokens alone, through the
    prefill and as rows pause and resume (`rows_prefill_pause_and_resume`)."""
    attention = CachedAttention()
    hidden = jax.random.normal(jax.random.key(0), (2, 6, 1, 4))
    variables = attention.init(jax.random.key(1), hidden[:, :1], PREFILL_VALID[:, :1])
    apply = jax.jit(lambda state, x, live: attention.apply(state, x, live, mutable=["cache"]))

    def alone(history):
        return scaled_dot_product_attention(history[:, -1:], history, history, implementation="xla")[0, 0]

    schedule = [(3, [True, False]), (4, [True, True]), (5, [False, True])]
    scenario = rows_prefill_pause_and_resume(apply, variables, alone, hidden, PREFILL_VALID, schedule)
    for actual, expected in scenario:
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("kind", ["latent", "sparse", "recurrent"])
def test_mixer_prefill_and_resume_follow_each_rows_real_tokens(kind):
    """A latent, sparse or recurrent mixer: each row's cached output is the mixer over its own real tokens
    alone, through the prefill and as rows pause and resume (`rows_prefill_pause_and_resume`)."""
    if kind == "recurrent":
        module = GatedDeltaNet(emb_features=16, num_k_heads=2, num_v_heads=2,
                               head_k_dim=4, head_v_dim=4, conv_kernel=4,
                               dtype=jnp.float32)
    else:
        sparse = {"index_topk": 3, "index_n_heads": 2, "index_head_dim": 8} if kind == "sparse" else {}
        module = MultiHeadLatentAttention(
            emb_features=16, num_heads=2, max_seq_len=8, q_lora_rank=8,
            kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
            dtype=jnp.float32, **sparse)
    hidden = jax.random.normal(jax.random.key(3), (2, 6, 16))
    params = module.init(jax.random.key(4), hidden)
    cache = module.apply(params, hidden[:, :1], MixerCall(decode=True), mutable=["cache"])[1]["cache"]
    apply = jax.jit(lambda state, x, valid: module.apply(
        {**params, "cache": state}, x,
        MixerCall(AttentionMetadata(coordinates=coordinates(valid=valid)), decode=True), mutable=["cache"]))

    def step(state, x, valid):
        actual, mutated = apply(state, x, valid)
        return actual, mutated["cache"]

    for actual, expected in rows_prefill_pause_and_resume(
            step, cache, lambda history: module.apply(params, history)[0, -1], hidden, PREFILL_VALID,
            [(3, [True, False]), (4, [False, True]), (5, [True, True])]):
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)


PACKED_VOCAB = 29
"""The packed continuation's vocabulary."""


@pytest.mark.parametrize("kind", ["full", "window"])
def test_a_packed_prompts_continuation_reads_its_last_document_alone(kind):
    """A causal decoder prefilled with a packed prompt, documents A then B (`coordinates`' segments), then
    continued with no segments, as generation continues a prompt: two decode steps on the greedy tokens and
    a two-token call (a speculative replay) read B's keys alone, so their logits, the generated tokens'
    likelihoods among them, are the same bits whatever A holds (the cache's own document identity,
    `KVCache.documents`), under full attention and under a window the prompt fits."""
    kinds = {"sliding_attention": {"window": 12}} if kind == "window" else {"full_attention": {}}
    types = tuple(kinds) * TINY_DECODER["num_layers"]
    decoder = CausalTransformer(vocab_size=PACKED_VOCAB, **TINY_DECODER, num_kv_heads=2, dtype="float32",
                                attention_impl="xla", layer_types=types, kinds=kinds)
    segments = jnp.asarray([[1] * 4 + [2] * 3])
    b = jax.random.randint(jax.random.key(1), (1, 3), 0, PACKED_VOCAB)
    params = decoder.init(jax.random.key(0), jnp.zeros((1, 7), jnp.int32))

    def continued(a):
        _, made = decoder.apply(params, 1, method=CausalTransformer.init_cache, mutable=["cache"])
        prompt = jnp.concatenate([a, b], axis=1)
        _, held = decoder.apply({**params, "cache": made["cache"]}, prompt, decode=True,
                                coordinates=coordinates(segments=segments), mutable=["cache"])
        cache, token, logits = held["cache"], None, []
        block, _ = decoder.apply({**params, "cache": cache}, jnp.asarray([[3, 5]]), decode=True,
                                 mutable=["cache"])
        logits.append(block)
        token = jnp.asarray([[7]])
        for _ in range(2):
            step, held = decoder.apply({**params, "cache": cache}, token, decode=True, mutable=["cache"])
            cache, token = held["cache"], jnp.argmax(step[:, -1:], axis=-1)
            logits.append(step)
        return [np.asarray(x) for x in logits]

    first = continued(jax.random.randint(jax.random.key(2), (1, 4), 0, PACKED_VOCAB))
    other = continued(jax.random.randint(jax.random.key(3), (1, 4), 0, PACKED_VOCAB))
    for ours, theirs in zip(first, other, strict=True):
        assert ours.tobytes() == theirs.tobytes()
