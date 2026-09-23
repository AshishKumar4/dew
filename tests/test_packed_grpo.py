"""GRPO over packed rows against GRPO over one unmerged row per model call.

A multi-call rollout whose history stays append-only merges into one chain,
and several chains share a row. Each sampled id must still be scored with
exactly the prefix its call saw, so the packed loss, its gradient and its
proximal log-probabilities equal the ones computed call by call.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Step, mean_loss
from dew.objectives.rl import GRPOObjective
from dew.objectives.rl.rollouts import (
    ADVANTAGES_KEY,
    BEHAVIOR_LOG_PROBS_KEY,
    CALL_INDEX_KEY,
    IDS_KEY,
    OLD_LOG_PROBS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    ROLLOUT_INDEX_KEY,
    SEGMENT_IDS_KEY,
    Call,
    Rollout,
    Status,
    advantages,
    pack,
)

VOCAB = 16
WIDTH = 24


def _model():
    return CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2,
                             mlp_features=32, max_seq_len=64, dtype="float32", attention_impl="xla")


def _rollouts():
    rng = np.random.default_rng(0)

    def ids(count):
        return tuple(int(value) for value in rng.integers(1, VOCAB, count))

    def logps(count):
        return tuple(float(value) for value in -rng.random(count))

    rollouts = []
    for sample in range(4):
        first = Call(ids(3), ids(2), logps(2), "tool_calls", 0)
        # Append-only: the second prompt holds the first call's prompt and sampled ids.
        second = Call(first.prompt_ids + first.sampled_ids + ids(2), ids(3), logps(3), "tool_calls", 0)
        # A rewrite: the third call's history drops the second call's sampled ids.
        third = Call(second.prompt_ids + ids(1), ids(2), logps(2), "stop", 0)
        rollouts.append(Rollout("t", str(sample // 2), sample, 0, (first, second, third),
                                Status.COMPLETED, float(sample % 2)))
    return rollouts


def _per_call(rollouts):
    """One chain per row, one row per call: the unmerged layout, each call's
    prompt then its sampled ids, carrying its rollout's advantage."""
    values = advantages(rollouts)
    calls = [(call, values[index]) for index, rollout in enumerate(rollouts) for call in rollout.calls]
    shape = (len(calls), WIDTH)
    ids, segments, positions = np.zeros(shape, np.int32), np.zeros(shape, np.int32), np.zeros(shape, np.int32)
    mask, behavior, advantage = np.zeros(shape, np.float32), np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    for row, (call, value) in enumerate(calls):
        size, count = len(call.prompt_ids), len(call.sampled_ids)
        ids[row, :size + count] = (*call.prompt_ids, *call.sampled_ids)
        segments[row, :size + count] = 1
        positions[row, :size + count] = np.arange(size + count)
        mask[row, size:size + count] = 1
        behavior[row, size:size + count] = call.behavior_log_probs
        advantage[row, :size + count] = value
    return {IDS_KEY: ids, SEGMENT_IDS_KEY: segments, POSITIONS_KEY: positions, RESPONSE_MASK_KEY: mask,
            OLD_LOG_PROBS_KEY: behavior, BEHAVIOR_LOG_PROBS_KEY: behavior, ADVANTAGES_KEY: advantage}


def _loss(objective, params, batch, reference):
    step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=reference)
    return mean_loss(objective.loss(params, batch, step)[0])[0]


@pytest.mark.parametrize("policy_loss", ["ppo", "cispo"])
def test_packed_grpo_equals_per_call_grpo_on_the_unmerged_chains(policy_loss):
    rollouts = _rollouts()
    packed = pack(rollouts, WIDTH)
    assert packed[IDS_KEY].shape[0] < sum(len(rollout.calls) for rollout in rollouts)
    assert packed[SEGMENT_IDS_KEY].max() >= 2, "rows share chains"
    packed[OLD_LOG_PROBS_KEY] = packed[BEHAVIOR_LOG_PROBS_KEY]
    unmerged = _per_call(rollouts)
    objective = GRPOObjective(_model(), WIDTH - 1, beta=0.1, policy_loss=policy_loss)
    params = objective.init(jax.random.key(1))
    reference = jax.tree.map(lambda leaf: leaf * 0.9, params)

    a, grad_a = jax.value_and_grad(lambda p: _loss(objective, p, packed, reference))(params)
    b, grad_b = jax.value_and_grad(lambda p: _loss(objective, p, unmerged, reference))(params)
    assert float(a) == pytest.approx(float(b), abs=1e-6)
    for left, right in zip(jax.tree.leaves(grad_a), jax.tree.leaves(grad_b), strict=True):
        np.testing.assert_allclose(left, right, atol=1e-6)


def test_packed_log_probs_score_each_id_with_its_own_calls_prefix():
    rollouts = _rollouts()
    packed = pack(rollouts, WIDTH)
    unmerged = _per_call(rollouts)
    objective = GRPOObjective(_model(), WIDTH - 1)
    params = objective.init(jax.random.key(2))
    scored = np.asarray(objective.packed_log_probs(params, packed))
    alone = np.asarray(objective.packed_log_probs(params, unmerged))
    expected = alone[unmerged[RESPONSE_MASK_KEY] != 0]
    order = [(index, number) for index, rollout in enumerate(rollouts) for number in range(len(rollout.calls))]
    placed = np.concatenate([scored[(packed[ROLLOUT_INDEX_KEY] == index) & (packed[CALL_INDEX_KEY] == number)]
                             for index, number in order])
    np.testing.assert_allclose(placed, expected, atol=1e-5)
    assert (scored[packed[RESPONSE_MASK_KEY] == 0] == 0).all()


def test_corrections_without_proximal_log_probs_read_the_current_policy():
    """verl's bypass mode: with behavior standing in for the old policy, the
    band and the sequence masks compare the detached current policy with
    behavior, rather than behavior with itself."""
    packed = pack(_rollouts(), WIDTH)
    objective = GRPOObjective(_model(), WIDTH - 1, behavior_band=(0.5, 5.0))
    params = objective.init(jax.random.key(3))
    step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None)
    policy = np.asarray(objective.packed_log_probs(params, packed))
    mask = packed[RESPONSE_MASK_KEY] != 0
    ratio = np.exp(policy - packed[BEHAVIOR_LOG_PROBS_KEY])[mask]
    outside = float(np.mean((ratio < 0.5) | (ratio > 5.0)))
    assert 0 < outside < 1, "the fixture puts some tokens outside the band"
    _, aux = objective.loss(params, packed, step)
    assert float(aux.metrics["masked/band"]) == pytest.approx(outside)
    plain, _ = GRPOObjective(_model(), WIDTH - 1).loss(params, packed, step)
    banded, _ = objective.loss(params, packed, step)
    assert abs(float(mean_loss(plain)[0]) - float(mean_loss(banded)[0])) > 1e-4
    geometric = GRPOObjective(_model(), WIDTH - 1, geometric_mask=(0.99, 1.01))
    _, aux = geometric.loss(params, packed, step)
    assert float(aux.metrics["masked/geometric"]) == 1.0
    capped = GRPOObjective(_model(), WIDTH - 1, behavior_importance_cap=2.0)
    with pytest.raises(ValueError, match="old_log_probs"):
        capped.loss(params, packed, step)


def test_mismatch_metrics_describe_every_trainable_token_whatever_a_mask_rejects():
    """verl computes IS weights and off-policy metrics over the response mask
    and applies rejection to the loss alone."""
    packed = pack(_rollouts(), WIDTH)
    model = _model()
    params = GRPOObjective(model, WIDTH - 1).init(jax.random.key(3))
    packed[OLD_LOG_PROBS_KEY] = np.asarray(GRPOObjective(model, WIDTH - 1).packed_log_probs(params, packed))
    step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None)
    _, open_aux = GRPOObjective(model, WIDTH - 1, behavior_band=(0.5, 5.0)).loss(params, packed, step)
    _, masked_aux = GRPOObjective(model, WIDTH - 1, behavior_band=(0.5, 5.0),
                                  sequence_mask=(0.99, 1.01)).loss(params, packed, step)
    assert float(masked_aux.metrics["masked/sequence"]) == 1.0
    for key in ("mismatch/kl", "mismatch/k3_kl", "mismatch/ess", "masked/band"):
        assert float(masked_aux.metrics[key]) == pytest.approx(float(open_aux.metrics[key]))
    assert float(open_aux.metrics["mismatch/kl"]) > 0.1
