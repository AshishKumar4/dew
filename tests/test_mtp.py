"""Multi-token-prediction depths: tree, head sharing, the shift, the loss term.

`num_nextn_predict_layers` stacks MTP depths after the final norm; 0 leaves
the tree unchanged. Depth d pairs the previous depth's state at position p
with the embedding of the token at p + d and scores the token after p + d
through the shared head (arXiv 2412.19437, section 2.2), so each depth is one
position shorter than the last. A plain init holds every depth: Flax creates
parameters where a call reaches them, and the forward reaches the depths
while initializing, so no caller has to know a second init method.

`LMObjective(mtp_weight=lambda)` adds lambda times the mean over the depths
of each depth's cross entropy to the training loss (eq. 24 of the paper);
unset, the loss is the plain shifted cross entropy and the depths get no
gradient.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective, TEXT_KEY

VOCAB = 37
SEQ = 8


def tiny(**overrides):
    config = dict(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=4,
                  mlp_features=64, max_seq_len=16)
    return CausalTransformer(**{**config, **overrides})


def test_no_depths_leaves_the_tree_unchanged():
    params = tiny().init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))

    assert tiny().num_nextn_predict_layers == 0
    assert [key for key in params["params"]] == [
        "embed_tokens", "layers_0", "layers_1", "norm"]


def test_depths_land_in_the_tree_with_the_fused_projection():
    params = tiny(num_nextn_predict_layers=2).init(
        jax.random.key(0), jnp.ones((1, 8), jnp.int32))["params"]

    assert params["mtp_0"]["eh_proj"]["kernel"].shape == (64, 32)
    assert params["mtp_1"]["eh_proj"]["kernel"].shape == (64, 32)
    for depth in ("mtp_0", "mtp_1"):
        assert set(params[depth]) == {"enorm", "hnorm", "eh_proj", "block", "final_norm"}
        assert "q_proj" in params[depth]["block"]["self_attn"]


def test_a_negative_depth_count_is_refused():
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        tiny(num_nextn_predict_layers=-1).init(
            jax.random.key(0), jnp.ones((1, 8), jnp.int32))


def test_a_sequence_with_no_position_a_depth_out_is_refused():
    with pytest.raises(ValueError, match="2 prediction depths need more than 2 tokens"):
        tiny(num_nextn_predict_layers=2).init(
            jax.random.key(0), jnp.ones((1, 2), jnp.int32))


def test_mtp_logits_score_one_token_further_per_depth():
    """One fp32 logit array per depth, each a position shorter than the
    last, finite, and the depths disagree with each other where they overlap."""
    model = tiny(num_nextn_predict_layers=2)
    ids = jnp.ones((2, 8), jnp.int32)
    params = model.init(jax.random.key(0), ids)
    hidden = model.apply(params, ids, method=CausalTransformer.hidden_states)

    depths = model.apply(params, hidden, ids, method=CausalTransformer.mtp_logits)

    assert [logits.shape for logits in depths] == [(2, 7, VOCAB), (2, 6, VOCAB)]
    for logits in depths:
        assert logits.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(logits))
    assert jnp.any(depths[0][:, :6] != depths[1])


def test_a_depth_reads_the_token_after_its_position():
    """Depth 1 at position p pairs the state at p with the token at p + 1
    (eq. 21 of the paper). Changing the last input token leaves every main
    state before it alone, so a depth that read the token at p would leave
    position S - 2 alone too; it changes here, and nothing earlier does."""
    model = tiny(num_nextn_predict_layers=1)
    ids = jnp.asarray(np.random.RandomState(0).randint(0, VOCAB, (1, SEQ)), jnp.int32)
    params = model.init(jax.random.key(0), ids)
    changed = ids.at[0, -1].set((ids[0, -1] + 1) % VOCAB)

    def depth_logits(tokens):
        hidden = model.apply(params, tokens, method=CausalTransformer.hidden_states)
        return model.apply(params, hidden, tokens, method=CausalTransformer.mtp_logits)[0]

    before, after = depth_logits(ids), depth_logits(changed)
    assert jnp.array_equal(before[:, :-1], after[:, :-1])
    assert jnp.any(before[:, -1] != after[:, -1])


def test_depths_share_the_main_head():
    """The main forward is the shared head over the final states."""
    model = tiny(num_nextn_predict_layers=1)
    ids = jnp.ones((2, 8), jnp.int32)
    params = model.init(jax.random.key(0), ids)
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


@pytest.mark.mesh
def test_a_depth_places_on_a_sharded_mesh():
    """The fused projection's input is two embed widths concatenated, so it
    cannot carry the embed name twice: flax refuses a repeated logical name,
    and a declaration with it failed every MTP model on a mesh with fsdp above
    one once the kernel crossed min_shard."""
    from dew.training import Layout, MeshSpec
    from dew.training.distributed import build_mesh

    model = tiny(num_nextn_predict_layers=1)
    params = model.init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    mesh = build_mesh(MeshSpec(fsdp=8))
    placement = Layout(min_shard=1).shardings(mesh, params)

    assert placement["params"]["mtp_0"]["eh_proj"]["kernel"].spec == jax.sharding.PartitionSpec(None, "fsdp")


# --------------------------------------------------------------------------
# The loss term
# --------------------------------------------------------------------------

def step_at(key=1):
    return Step(step=jnp.asarray(0), key=jax.random.key(key), ema=None)


def token_batch(batch=4, seed=0):
    ids = np.random.RandomState(seed).randint(0, VOCAB, (batch, SEQ + 1))
    return {TEXT_KEY: jnp.asarray(ids, jnp.int32)}


def cross_entropy(logits, targets):
    """Per-position cross entropy in float64 from the model's own logits."""
    logits = np.asarray(logits, np.float64)
    largest = logits.max(axis=-1, keepdims=True)
    log_probs = logits - largest - np.log(np.exp(logits - largest).sum(axis=-1, keepdims=True))
    return -np.take_along_axis(log_probs, np.asarray(targets)[..., None], axis=-1)[..., 0]


def depth_cross_entropies(model, params, tokens):
    """Each depth's cross entropy by hand: depth d's logits at p against the
    token at p + d + 1, summed and divided by the main term's count."""
    inputs = jnp.asarray(tokens[:, :-1], jnp.int32)
    hidden = model.apply(params, inputs, method=CausalTransformer.hidden_states)
    depths = model.apply(params, hidden, inputs, method=CausalTransformer.mtp_logits)
    count = tokens.shape[0] * (tokens.shape[1] - 1)
    return [cross_entropy(logits, tokens[:, depth + 1:]).sum() / count
            for depth, logits in enumerate(depths, start=1)]


def test_the_term_is_off_by_default():
    """Without mtp_weight the loss is the shifted cross entropy alone, the
    same number a depth-less model computes on the same trunk, and the
    depths receive no gradient."""
    model = tiny(num_nextn_predict_layers=1)
    objective = LMObjective(model, SEQ)
    params = objective.init(jax.random.key(0))
    batch = token_batch()

    (loss, aux), grads = jax.value_and_grad(
        lambda p: objective.loss(p, batch, step_at()), has_aux=True)(params)

    assert float(loss) == float(aux.metrics["ce"])
    assert "mtp_ce" not in aux.metrics
    plain = {"params": {key: value for key, value in params["params"].items()
                        if key != "mtp_0"}}
    plain_loss, _ = LMObjective(tiny(), SEQ).loss(plain, batch, step_at())
    assert float(loss) == pytest.approx(float(plain_loss), rel=1e-6)
    assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(grads["params"]["mtp_0"]))


def test_the_term_adds_the_weighted_mean_depth_cross_entropy():
    """With mtp_weight the loss is ce + lambda times the mean over the depths
    of each depth's cross entropy, each computed here by hand from the
    depth's logits against the targets d + 1 positions out, and the depths
    receive gradient."""
    model = tiny(num_nextn_predict_layers=2)
    objective = LMObjective(model, SEQ, mtp_weight=0.3)
    params = objective.init(jax.random.key(0))
    batch = token_batch()
    tokens = np.asarray(batch[TEXT_KEY])

    (loss, aux), grads = jax.value_and_grad(
        lambda p: objective.loss(p, batch, step_at()), has_aux=True)(params)

    ce = float(aux.metrics["ce"])
    expected = depth_cross_entropies(model, params, tokens)
    assert float(aux.metrics["mtp_ce"]) == pytest.approx(float(np.mean(expected)), rel=1e-5)
    assert float(loss) == pytest.approx(ce + 0.3 * float(np.mean(expected)), rel=1e-5)
    assert abs(float(loss) - ce) > 1e-2
    # Scored against the wrong tokens (the main targets, unshifted) the term
    # is a different number, which is the mistake this guards.
    inputs = jnp.asarray(tokens[:, :-1], jnp.int32)
    hidden = model.apply(params, inputs, method=CausalTransformer.hidden_states)
    first = model.apply(params, hidden, inputs, method=CausalTransformer.mtp_logits)[0]
    unshifted = cross_entropy(first, tokens[:, 1:-1]).sum() / (tokens.shape[0] * SEQ)
    assert abs(unshifted - expected[0]) > 1e-3
    for depth in ("mtp_0", "mtp_1"):
        assert any(jnp.any(leaf) for leaf in jax.tree.leaves(grads["params"][depth]))


def test_a_packed_batch_keeps_the_depths_inside_their_documents():
    """A depth's target counts only when the state that predicts it sits in
    the same document: depth 1 drops the last two transitions of every
    document and everything into the padding, one more than the main term."""
    model = tiny(num_nextn_predict_layers=1)
    objective = LMObjective(model, SEQ, mtp_weight=0.3)
    params = objective.init(jax.random.key(0))
    ids = np.random.RandomState(0).randint(1, VOCAB, (1, SEQ + 1)).astype(np.int32)
    # Two documents of four and three tokens, then padding (segment 0).
    segments = np.array([[1, 1, 1, 1, 2, 2, 2, 0, 0]], np.int32)
    positions = np.array([[0, 1, 2, 3, 0, 1, 2, 0, 0]], np.int32)
    batch = {TEXT_KEY: jnp.asarray(ids), "segment_ids": jnp.asarray(segments),
             "positions": jnp.asarray(positions)}

    _, _, _, _, depths = objective.token_scores(
        params, jnp.asarray(ids), segment_ids=jnp.asarray(segments),
        positions=jnp.asarray(positions), depths=True)
    (_, weights), = depths

    # State at p predicts the target at p + 2: only p = 0, 1 (document 1)
    # and p = 4 (document 2) stay inside a document.
    assert weights.tolist() == [[1, 1, 0, 0, 1, 0, 0]]
    loss, aux = objective.loss(params, batch, step_at())
    assert np.isfinite(float(loss)) and float(aux.metrics["mtp_ce"]) > 0


def test_mtp_weight_needs_depths_and_a_positive_weight():
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        LMObjective(tiny(), SEQ, mtp_weight=0.3)
    with pytest.raises(ValueError, match="positive weight"):
        LMObjective(tiny(num_nextn_predict_layers=1), SEQ, mtp_weight=0.0)


def test_a_routed_depth_balances_only_when_the_depths_run():
    """A depth routes like the trunk's last layer, so its balancing bias
    moves with the load its router saw; without mtp_weight the depth never
    runs, observes no load, and its bias stays where it was."""
    model = tiny(num_nextn_predict_layers=1, num_kv_heads=4,
                 mixture={"experts": 4, "top_k": 2, "score_function": "sigmoid", "bias": True})
    batch = token_batch()
    still = LMObjective(model, SEQ, balance_rate=0.1)
    params = still.init(jax.random.key(0))
    _, aux = still.loss(params, batch, step_at())
    assert aux.variables is not None
    moved = aux.variables["moe"]
    assert jnp.array_equal(moved["mtp_0"]["block"]["mlp"]["gate"]["e_score_correction_bias"],
                           params["moe"]["mtp_0"]["block"]["mlp"]["gate"]["e_score_correction_bias"])
    assert not jnp.array_equal(moved["layers_1"]["mlp"]["gate"]["e_score_correction_bias"],
                               params["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"])

    _, aux = LMObjective(model, SEQ, balance_rate=0.1, mtp_weight=0.3).loss(params, batch, step_at())
    assert aux.variables is not None
    assert not jnp.array_equal(
        aux.variables["moe"]["mtp_0"]["block"]["mlp"]["gate"]["e_score_correction_bias"],
        params["moe"]["mtp_0"]["block"]["mlp"]["gate"]["e_score_correction_bias"])

