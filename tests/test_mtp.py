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
from model_support import TINY_DECODER
from token_support import token_batch_of

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import ModelInputs, coordinates
from dew.objectives.base import Step
from dew.objectives.lm import TEXT_KEY, LMObjective
from dew.objectives.routing import ExpertBalance

VOCAB = 37
SEQ = 8


def tiny(**overrides):
    return CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, **overrides})


def test_mean_stream_trunk_exposes_normalized_states_to_plain_prediction():
    model = tiny(num_nextn_predict_layers=1, dtype=jnp.float32,
                 hyper_connections={'hc_mult': 4, 'hc_eps': 1e-6,
                                    'hc_sinkhorn_iters': 20, 'head': 'mean'})
    ids = token_batch(batch=2)[TEXT_KEY]
    variables = model.init(jax.random.key(0), ids)
    hidden = jnp.asarray(model.apply(variables, ids, method=CausalTransformer.hidden_states))
    state, _ = model.apply(variables, ids, method=CausalTransformer.states_and_logits)
    state = jnp.asarray(state)
    assert state.shape == (*ids.shape, model.emb_features)
    np.testing.assert_array_equal(state, hidden)
    doubled = {**variables, 'params': {**variables['params'], 'norm': {
        **variables['params']['norm'], 'scale': variables['params']['norm']['scale'] * 2}}}
    louder, _ = model.apply(doubled, ids, method=CausalTransformer.states_and_logits)
    np.testing.assert_allclose(jnp.asarray(louder), 2 * state, atol=1e-4, rtol=0)
    expected = model.apply(variables, hidden, ids, method=CausalTransformer.mtp_logits)[0]
    cache = model.apply(variables, ids.shape[0], method=CausalTransformer.init_mtp_cache,
                        mutable=['cache'])[1]['cache']
    (logits, carried), _ = model.apply(
        {**variables, 'cache': cache}, state[:, :1], ids[:, 1:2],
        method=CausalTransformer.mtp_step, decode=True, mutable=['cache'])
    np.testing.assert_allclose(jnp.asarray(logits), jnp.asarray(expected)[:, :1], atol=1e-4, rtol=0)
    assert jnp.asarray(carried).shape == (ids.shape[0], 1, model.emb_features)


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


def test_the_depths_score_through_the_main_head():
    """A depth owns no head: doubling the model's untied head doubles the main
    logits and every depth's, which a depth scoring through a head of its own
    would not."""
    model = tiny(num_nextn_predict_layers=2, tie_embeddings=False)
    ids = jnp.ones((2, 8), jnp.int32)
    params = model.init(jax.random.key(0), ids)
    doubled = {"params": {**params["params"],
                          "lm_head": {"kernel": params["params"]["lm_head"]["kernel"] * 2.0}}}

    def scored(variables):
        hidden = model.apply(variables, ids, method=CausalTransformer.hidden_states)
        return [model.apply(variables, ids),
                *model.apply(variables, hidden, ids, method=CausalTransformer.mtp_logits)]

    for plain, louder in zip(scored(params), scored(doubled), strict=True):
        assert float(jnp.max(jnp.abs(plain))) > 1e-3
        assert jnp.allclose(2.0 * plain, louder, rtol=1e-6)


# --------------------------------------------------------------------------
# The loss term
# --------------------------------------------------------------------------

def step_at(key=1):
    return Step(step=jnp.asarray(0), key=jax.random.key(key), ema=None)


def token_batch(batch=4, seed=0):
    return token_batch_of(VOCAB, SEQ, batch, seed)


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
        lambda p: objective.scalar_loss(p, batch, step_at()), has_aux=True)(params)

    assert float(loss) == float(aux.metrics["ce"])
    assert "mtp_ce" not in aux.metrics
    plain = {"params": {key: value for key, value in params["params"].items()
                        if key != "mtp_0"}}
    plain_loss, _ = LMObjective(tiny(), SEQ).scalar_loss(plain, batch, step_at())
    assert float(loss) == pytest.approx(float(plain_loss), rel=1e-6)
    assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(grads["params"]["mtp_0"]))


@pytest.mark.parametrize('hyper_connections', [
    None, {'hc_mult': 4, 'hc_eps': 1e-6, 'hc_sinkhorn_iters': 20, 'head': 'mean'},
], ids=['plain_trunk', 'mean_stream_trunk'])
def test_the_term_adds_the_weighted_mean_depth_cross_entropy(hyper_connections):
    """With mtp_weight the loss is ce + lambda times the mean over the depths
    of each depth's cross entropy, each computed here by hand from the
    depth's logits against the targets d + 1 positions out, and the depths
    receive gradient."""
    model = tiny(num_nextn_predict_layers=2, hyper_connections=hyper_connections)
    objective = LMObjective(model, SEQ, mtp_weight=0.3)
    params = objective.init(jax.random.key(0))
    batch = token_batch()
    tokens = np.asarray(batch[TEXT_KEY])

    (loss, aux), grads = jax.value_and_grad(
        lambda p: objective.scalar_loss(p, batch, step_at()), has_aux=True)(params)

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


def packed_row():
    """One row of two documents, four and three tokens, then padding (segment 0)."""
    ids = np.random.RandomState(0).randint(1, VOCAB, (1, SEQ + 1)).astype(np.int32)
    segments = np.array([[1, 1, 1, 1, 2, 2, 2, 0, 0]], np.int32)
    positions = np.array([[0, 1, 2, 3, 0, 1, 2, 0, 0]], np.int32)
    return jnp.asarray(ids), jnp.asarray(segments), jnp.asarray(positions)


@pytest.mark.parametrize("depths", [1, 2])
def test_packing_carried_by_the_model_inputs_scores_the_depths_the_same(depths):
    """Coordinates may ride the `ModelInputs` instead of beside the batch's
    token ids, and the losses are the same number either way."""
    objective = LMObjective(tiny(num_nextn_predict_layers=depths), SEQ, mtp_weight=0.3)
    params = objective.init(jax.random.key(0))
    ids, segments, positions = packed_row()
    laid = coordinates(segments=segments, positions=positions, max_segments=ids.shape[1],
                       right_padded=True, xp=np)
    columns = {TEXT_KEY: ids, "text_coordinates": laid}
    carried = {TEXT_KEY: ModelInputs(ids, {"coordinates": laid})}

    _, by_columns = objective.scalar_loss(params, columns, step_at())
    _, by_inputs = objective.scalar_loss(params, carried, step_at())

    assert float(by_inputs.metrics["ce"]) == float(by_columns.metrics["ce"])
    assert float(by_inputs.metrics["mtp_ce"]) == float(by_columns.metrics["mtp_ce"])
    options = {"train": True, "depths": True, "rngs": {"dropout": step_at().key}}
    supplied = objective.token_scores(params, carried[TEXT_KEY], **options)
    explicit = objective.token_scores(params, ids, segment_ids=segments, positions=positions, **options)
    for ours, reference in zip((supplied.losses, supplied.weights, *supplied.depths),
                              (explicit.losses, explicit.weights, *explicit.depths), strict=True):
        for left, right in zip(jax.tree.leaves(ours), jax.tree.leaves(reference), strict=True):
            np.testing.assert_array_equal(left, right)


def test_a_routed_depth_balances_only_when_the_depths_run():
    """A depth routes like the trunk's last layer, so its balancing bias
    moves with the load its router saw; without mtp_weight the depth never
    runs, observes no load, and its bias stays where it was."""
    model = tiny(num_nextn_predict_layers=1, num_kv_heads=4,
                 mixture={"experts": 4, "top_k": 2, "score_function": "sigmoid", "bias": True})
    batch = token_batch()
    still = LMObjective(model, SEQ, balance=ExpertBalance(balance_rate=0.1))
    params = still.init(jax.random.key(0))
    _, aux = still.scalar_loss(params, batch, step_at())
    assert aux.effects is not None
    moved = still.apply_effects(params, aux.effects)["moe"]
    assert jnp.array_equal(moved["mtp_0"]["block"]["mlp"]["gate"]["e_score_correction_bias"],
                           params["moe"]["mtp_0"]["block"]["mlp"]["gate"]["e_score_correction_bias"])
    assert not jnp.array_equal(moved["layers_1"]["mlp"]["gate"]["e_score_correction_bias"],
                               params["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"])

    running = LMObjective(model, SEQ, balance=ExpertBalance(balance_rate=0.1), mtp_weight=0.3)
    _, aux = running.scalar_loss(params, batch, step_at())
    assert aux.effects is not None
    moved = running.apply_effects(params, aux.effects)["moe"]
    assert not jnp.array_equal(
        moved["mtp_0"]["block"]["mlp"]["gate"]["e_score_correction_bias"],
        params["moe"]["mtp_0"]["block"]["mlp"]["gate"]["e_score_correction_bias"])
