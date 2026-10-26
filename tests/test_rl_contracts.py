"""Policy loss terms and replicated-reader rollout contracts."""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from process_support import run_pool
from test_indexer_training import SEQ, VOCAB, deepseek_stack

from dew.data.preferences import MASK_KEY
from dew.nn.backbones.decoder_block import Mixture
from dew.objectives.base import Step
from dew.objectives.rl import DPOObjective, GRPOObjective
from dew.objectives.rl.sessions import (
    ADVANTAGES_KEY,
    BEHAVIOR_LOG_PROBS_KEY,
    IDS_KEY,
    OLD_LOG_PROBS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    SEGMENT_IDS_KEY,
)

WORKER = Path(__file__).with_name("rl_contracts_worker.py")


def scored_batch(objective) -> tuple[dict, jax.Array]:
    """Two rows of SEQ + 1 ids, each with a four-id prompt: a pair for DPO, two one-chain packed
    rows with drawn likelihoods and advantages for GRPO. With them, the rows the loss scores."""
    rng = np.random.default_rng(0)
    ids = rng.integers(0, VOCAB, (2, SEQ + 1)).astype(np.int32)
    mask = np.ones((2, SEQ + 1), np.float32)
    mask[:, :4] = 0
    if objective is DPOObjective:
        pair = {IDS_KEY: jnp.asarray(ids[None]), MASK_KEY: jnp.asarray(mask[None], jnp.int32)}
        return pair, jnp.asarray(ids)
    drawn = np.where(mask != 0, -rng.random((2, SEQ + 1)), 0).astype(np.float32)
    return {
        IDS_KEY: jnp.asarray(ids),
        SEGMENT_IDS_KEY: jnp.ones((2, SEQ + 1), jnp.int32),
        POSITIONS_KEY: jnp.tile(jnp.arange(SEQ + 1, dtype=jnp.int32), (2, 1)),
        OLD_LOG_PROBS_KEY: jnp.asarray(drawn),
        BEHAVIOR_LOG_PROBS_KEY: jnp.asarray(drawn),
        ADVANTAGES_KEY: jnp.asarray(np.where(mask != 0, rng.normal(0, 1, (2, SEQ + 1)), 0), jnp.float32),
        RESPONSE_MASK_KEY: jnp.asarray(mask),
    }, jnp.asarray(ids)


@pytest.mark.parametrize("objective", [GRPOObjective, DPOObjective])
def test_policy_losses_score_through_the_scorers_head_and_report_the_policys_qk_maxima(objective):
    """A policy or preference loss over a routed decoder with prediction depths scores its tokens
    through the token scorer. The head's chunks and tile change how the head runs, not the loss
    or its gradient; `qk_stats` reports, for the optimizer's QK-Clip, the per-head logit maxima of
    the policy's own forward, not of the frozen reference it also runs, and nothing without it."""
    model = deepseek_stack(None, mixture=Mixture(experts=4, top_k=2), num_nextn_predict_layers=1)
    plain = objective(model, SEQ, 0.1)
    variables = plain.init(jax.random.key(0))
    reference = jax.tree.map(lambda leaf: leaf * 0.9, variables)
    batch, rows = scored_batch(objective)
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=reference)

    def trained(built, step):
        (loss, aux), gradient = jax.value_and_grad(built.scalar_loss, has_aux=True)(variables, batch, step)
        return loss, gradient, aux.qk_stats

    loss, gradient, qk = trained(plain, step)
    assert qk is None
    headed = objective(model, SEQ, 0.1, head_chunks=2, head_tile="whole")
    headed_loss, headed_gradient, qk = trained(headed, dataclasses.replace(step, qk_stats=True))
    np.testing.assert_allclose(headed_loss, loss, rtol=1e-5)
    jax.tree.map(lambda found, expected: np.testing.assert_allclose(found, expected, rtol=1e-4, atol=1e-6),
                 headed_gradient, gradient)
    own, frozen = (plain.token_scores(tree, rows, qk_stats=True).qk for tree in (variables, reference))
    jax.tree.map(np.testing.assert_allclose, qk, own)
    assert not all(np.allclose(found, other) for found, other in zip(
        jax.tree.leaves(qk), jax.tree.leaves(frozen), strict=True))


@pytest.mark.mesh
@pytest.mark.parametrize("mode", ["native", "ppo"])
def test_episode_rollouts_refuse_replicated_readers_before_any_tool_runs(tmp_path, mode):
    reports = run_pool(mode, tmp_path, 2, script=WORKER, devices=1, timeout=120)
    for report in reports:
        assert report["readers"] == 2 and report["shares"] == 1
        assert report["refused"] and report["opened"] == report["closed"] == 0
        assert report["verified"] == 0
