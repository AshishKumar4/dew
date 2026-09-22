"""DeepSeek sparse attention executed over the selection, not a mask.

`sparse_latent_attention` attends each query's top-k keys in MLA's latent
space. The reference is the attention the V3.2 reference computes: expanded
keys and values under the selection folded into a `[B, S, T]` mask, which
`tests/test_mla.py` holds against transformers 5.16.1 (the V3.2 fixture's
seven tokens exceed its top-k of four, so that parity test runs this path).
"""

import jax
import jax.numpy as jnp
import numpy as np

from dew.lora import LoRA, Target
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.mla import (
    MultiHeadLatentAttention,
    selection_mask,
    sparse_latent_attention,
    top_k_selection,
)

HEADS, RANK, NOPE, ROPE, VALUE = 4, 24, 16, 8, 12


def pieces(length, batch=2, seed=0):
    keys = jax.random.split(jax.random.key(seed), 7)
    q_nope = jax.random.normal(keys[0], (batch, length, HEADS, NOPE))
    q_rot = jax.random.normal(keys[1], (batch, length, HEADS, ROPE))
    latent = jax.random.normal(keys[2], (batch, length, RANK))
    rot = jax.random.normal(keys[3], (batch, length, ROPE))
    key_weight = jax.random.normal(keys[4], (RANK, HEADS, NOPE)) / np.sqrt(RANK)
    value_weight = jax.random.normal(keys[5], (RANK, HEADS, VALUE)) / np.sqrt(RANK)
    scores = jax.random.normal(keys[6], (batch, length, length))
    return q_nope, q_rot, latent, rot, key_weight, value_weight, scores


def expanded_reference(q_nope, q_rot, latent, rot, key_weight, value_weight, mask):
    """The V3.2 reference's arithmetic: per-head keys and values, masked softmax."""
    nope = jnp.einsum('btr,rhn->bthn', latent, key_weight)
    key = jnp.concatenate([nope, jnp.broadcast_to(rot[:, :, None], (*nope.shape[:3], ROPE))], -1)
    value = jnp.einsum('btr,rhv->bthv', latent, value_weight)
    query = jnp.concatenate([q_nope, q_rot], -1)
    return scaled_dot_product_attention(query, key, value, implementation="reference",
                                        mask=mask[:, None])


def causal_selection(scores, top_k, segments=None):
    length = scores.shape[-1]
    keep = jnp.tril(jnp.ones((length, length), bool))[None]
    if segments is not None:
        keep = keep & (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0)
    return top_k_selection(scores, keep, top_k)


def test_the_selection_attends_as_the_masked_expanded_reference():
    """Forward and every input gradient at fp32. A query whose causal set is
    smaller than k carries unselected slots, which read nothing. The block
    size of 5 splits 23 queries unevenly, so padding rides the last block."""
    length, top_k = 23, 6
    q_nope, q_rot, latent, rot, key_weight, value_weight, scores = pieces(length)
    indices, chosen = causal_selection(scores, top_k)
    mask = selection_mask(indices, chosen, length)
    scale = 1 / np.sqrt(NOPE + ROPE)
    cotangent = jax.random.normal(jax.random.key(9), (2, length, HEADS, VALUE))

    def sparse(*args):
        return jnp.sum(sparse_latent_attention(*args, indices, chosen, scale=scale, block=5) * cotangent)

    def dense(*args):
        return jnp.sum(expanded_reference(*args, mask) * cotangent)

    args = (q_nope, q_rot, latent, rot, key_weight, value_weight)
    np.testing.assert_allclose(sparse(*args), dense(*args), rtol=1e-5)
    for actual, expected in zip(jax.grad(sparse, argnums=tuple(range(6)))(*args),
                                jax.grad(dense, argnums=tuple(range(6)))(*args), strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5, rtol=0)


def test_packed_documents_select_inside_their_own_document():
    length, top_k = 19, 4
    q_nope, q_rot, latent, rot, key_weight, value_weight, scores = pieces(length, seed=1)
    segments = jnp.asarray(np.repeat([[1, 2, 3, 0]], [6, 7, 4, 2], axis=1).repeat(2, 0))
    indices, chosen = causal_selection(scores, top_k, segments)
    mask = selection_mask(indices, chosen, length)
    actual = sparse_latent_attention(q_nope, q_rot, latent, rot, key_weight, value_weight,
                                     indices, chosen, scale=1 / np.sqrt(NOPE + ROPE))
    expected = expanded_reference(q_nope, q_rot, latent, rot, key_weight, value_weight, mask)
    real = np.asarray(segments) != 0
    np.testing.assert_allclose(np.asarray(actual)[real], np.asarray(expected)[real], atol=2e-6)


def test_the_layer_runs_the_selection_and_matches_its_masked_kernel():
    """The V3.2 layer past its top-k takes the sparse path; an open `qk`
    collection sends the same call through the masked dense kernel."""
    layer = MultiHeadLatentAttention(
        emb_features=32, num_heads=HEADS, max_seq_len=64, q_lora_rank=16, kv_lora_rank=RANK,
        qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE, v_head_dim=VALUE,
        index_topk=5, index_n_heads=2, index_head_dim=16, attention_impl="reference")
    hidden = jax.random.normal(jax.random.key(2), (2, 21, 32))
    variables = layer.init(jax.random.key(3), hidden)
    sparse = layer.apply(variables, hidden)
    dense, _ = layer.apply(variables, hidden, mutable=["qk"])
    np.testing.assert_allclose(np.asarray(sparse), np.asarray(dense), atol=2e-5, rtol=0)


def test_an_adapter_dropping_out_kv_b_proj_keeps_the_masked_kernel():
    """LoRA dropout on kv_b_proj draws a mask per token and latent dimension.
    The absorbed path applies kv_b_proj to the identity, where the same draw
    would be one mask for every token, so such a training call runs the
    masked kernel: it matches the same call with `qk` open, rng for rng."""
    layer = MultiHeadLatentAttention(
        emb_features=32, num_heads=HEADS, max_seq_len=64, q_lora_rank=16, kv_lora_rank=RANK,
        qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE, v_head_dim=VALUE,
        index_topk=5, index_n_heads=2, index_head_dim=16, attention_impl="reference")
    adapter = LoRA({("params", "kv_b_proj"): Target(rank=4, alpha=8.0)}, dropout=0.5)
    adapted = adapter.adapt(layer)
    hidden = jax.random.normal(jax.random.key(2), (2, 21, 32))
    variables = adapted.init(jax.random.key(3), hidden)
    factor = variables["params"]["kv_b_proj"]["lora_B"]
    variables["params"]["kv_b_proj"]["lora_B"] = jax.random.normal(jax.random.key(4), factor.shape)
    rngs = {"dropout": jax.random.key(5)}
    trained = adapted.apply(variables, hidden, rngs=rngs)
    masked, _ = adapted.apply(variables, hidden, rngs=rngs, mutable=["qk"])
    np.testing.assert_allclose(np.asarray(trained), np.asarray(masked), atol=2e-5, rtol=0)


def test_memory_follows_the_selection_not_the_sequence():
    """At 2048 keys and a top-k of 64 the masked kernel holds `[B, H, S, T]`
    logits; the selection's temporaries are a fraction of that. Both measured
    from the compiled executables."""
    length, top_k = 2048, 64
    q_nope, q_rot, latent, rot, key_weight, value_weight, scores = pieces(length, batch=1)
    indices, chosen = causal_selection(scores, top_k)
    args = (q_nope, q_rot, latent, rot, key_weight, value_weight)
    mask = selection_mask(indices, chosen, length)

    def temporaries(fn, *inputs):
        return jax.jit(fn).lower(*inputs).compile().memory_analysis().temp_size_in_bytes

    dense = temporaries(expanded_reference, *args, mask)
    selected = temporaries(lambda *a: sparse_latent_attention(
        *a, scale=1.0), *args, indices, chosen)
    assert dense >= HEADS * length * length * 4
    assert selected * 8 < dense
