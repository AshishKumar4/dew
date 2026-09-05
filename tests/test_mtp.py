"""Multi-token-prediction depths: tree and head sharing.

`num_nextn_predict_layers` stacks MTP depths after the final norm; 0 leaves
the tree unchanged. A depth reads the previous hidden states and the token
embeddings and scores one token further out through the shared head. A plain
init holds every depth: Flax creates parameters where a call reaches them,
and the forward reaches the depths while initializing, so no caller has to
know a second init method.
"""

import jax
import jax.numpy as jnp
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer

VOCAB = 37


def tiny(**overrides):
    config = dict(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=4,
                  mlp_features=64, max_seq_len=16)
    return CausalTransformer(**{**config, **overrides})


def both_paths(model, key, ids):
    """The variables tree a plain init builds, main path and every depth."""
    return model.init(key, ids)


def test_no_depths_leaves_the_tree_unchanged():
    params = tiny().init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))

    assert tiny().num_nextn_predict_layers == 0
    assert [key for key in params["params"]] == [
        "embed_tokens", "layers_0", "layers_1", "norm"]


def test_depths_land_in_the_tree_with_the_fused_projection():
    params = both_paths(tiny(num_nextn_predict_layers=2), jax.random.key(0),
                        jnp.ones((1, 8), jnp.int32))["params"]

    assert params["mtp_0"]["eh_proj"]["kernel"].shape == (64, 32)
    assert params["mtp_1"]["eh_proj"]["kernel"].shape == (64, 32)
    for depth in ("mtp_0", "mtp_1"):
        assert set(params[depth]) == {"enorm", "hnorm", "eh_proj", "block", "final_norm"}
        assert "q_proj" in params[depth]["block"]["self_attn"]


def test_a_negative_depth_count_is_refused():
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        tiny(num_nextn_predict_layers=-1).init(
            jax.random.key(0), jnp.ones((1, 8), jnp.int32))


def test_mtp_logits_score_one_token_further_per_depth():
    """Two fp32 logit arrays, finite, and the depths disagree with each other."""
    model = tiny(num_nextn_predict_layers=2)
    ids = jnp.ones((2, 8), jnp.int32)
    params = both_paths(model, jax.random.key(0), ids)
    hidden = model.apply(params, ids, method=CausalTransformer.hidden_states)

    depths = model.apply(params, hidden, ids, method=CausalTransformer.mtp_logits)

    assert len(depths) == 2
    for logits in depths:
        assert logits.shape == (2, 8, VOCAB)
        assert logits.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(logits))
    assert jnp.any(depths[0] != depths[1])


def test_depths_share_the_main_head():
    """The main forward is the shared head over the final states."""
    model = tiny(num_nextn_predict_layers=1)
    ids = jnp.ones((2, 8), jnp.int32)
    params = both_paths(model, jax.random.key(0), ids)
    hidden = model.apply(params, ids, method=CausalTransformer.hidden_states)

    assert jnp.array_equal(
        model.apply(params, ids),
        model.apply(params, hidden, method=lambda m, x: m._logits(x)))


def test_no_depths_scores_nothing():
    model = tiny()
    ids = jnp.ones((2, 8), jnp.int32)
    params = model.init(jax.random.key(0), ids)
    hidden = model.apply(params, ids, method=CausalTransformer.hidden_states)

    assert model.apply(params, hidden, ids, method=CausalTransformer.mtp_logits) == []
