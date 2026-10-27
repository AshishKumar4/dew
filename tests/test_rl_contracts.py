"""Policy loss terms and replicated-reader rollout contracts."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from process_support import run_pool
from reference_error import assert_as_exact
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


def scored_batch(objective, seed: int = 0) -> tuple[dict, jax.Array]:
    """Two rows of SEQ + 1 ids, each with a four-id prompt, drawn by `seed`: a pair for DPO, two
    one-chain packed rows with drawn likelihoods and advantages for GRPO. With them, the rows the loss
    scores."""
    rng = np.random.default_rng(seed)
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


HEAD_SEEDS = {GRPOObjective: 64, DPOObjective: 128}
"""Check 1's declared seeds for the headed policy losses, sized by `seeds_for` paired on armada: the
headed and plain heads run one model over one batch, so they share their upstream rounding."""


@pytest.mark.parametrize("objective", [GRPOObjective, DPOObjective])
def test_policy_losses_score_through_the_scorers_head_and_report_the_policys_qk_maxima(objective):
    """A policy or preference loss over a routed decoder with prediction depths scores its tokens
    through the token scorer. The head's chunks and tile change how the head runs, not the loss
    or its gradient: over each drawn model, reference and batch, the headed loss and every gradient
    leaf are as exact as the plain head's (check 1), the truth the plain loss over the model's float64
    twin. `qk_stats` reports, for the optimizer's QK-Clip, the per-head logit maxima of the policy's
    own forward, those its own scores give, not the frozen reference's it also runs, and nothing
    without it."""
    def built(dtype, **head):
        model = deepseek_stack(None, mixture=Mixture(experts=4, top_k=2), num_nextn_predict_layers=1,
                               dtype=dtype)
        return objective(model, SEQ, 0.1, **head)

    plain, headed, wide = built(None), built(None, head_chunks=2, head_tile="whole"), built(jnp.float64)
    initialize = jax.jit(plain.init)

    def draw(seed):
        variables = initialize(jax.random.key(seed))
        reference = jax.tree.map(lambda leaf: leaf * 0.9, variables)
        return variables, reference, scored_batch(objective, int(seed))[0]

    def trained(scored, *, qk_stats=False):
        def run(variables, reference, batch):
            step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=reference, qk_stats=qk_stats)
            scalar = jax.value_and_grad(scored.scalar_loss, has_aux=True)
            (loss, aux), gradient = scalar(variables, batch, step)
            return {"loss": loss, "gradient": gradient}, aux.qk_stats
        return run

    def leaves(scored):
        return lambda *drawn: trained(scored)(*drawn)[0]

    def truth(*drawn):
        # Traced by the gate, so widened in JAX: every floating leaf to float64, exactly.
        def wider(leaf):
            return leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf
        return leaves(wide)(*jax.tree.map(wider, drawn))

    # The gate runs the truth in float64 on the draw widened.
    assert_as_exact(leaves(headed), leaves(plain), draw, f"{objective.__name__} head",
                    seeds=HEAD_SEEDS[objective], truth_fn=truth)
    variables, reference, batch = draw(0)
    assert trained(plain)(variables, reference, batch)[1] is None
    _, qk = trained(headed, qk_stats=True)(variables, reference, batch)
    rows = scored_batch(objective)[1]
    own, frozen = (plain.token_scores(tree, rows, qk_stats=True).qk for tree in (variables, reference))
    jax.tree.map(np.testing.assert_array_equal, qk, own)
    assert not all(np.array_equal(found, other) for found, other in zip(
        jax.tree.leaves(qk), jax.tree.leaves(frozen), strict=True))


@pytest.mark.mesh
@pytest.mark.parametrize("mode", ["native", "ppo"])
def test_episode_rollouts_refuse_replicated_readers_before_any_tool_runs(tmp_path, mode):
    reports = run_pool(mode, tmp_path, 2, script=WORKER, devices=1, timeout=120)
    for report in reports:
        assert report["readers"] == 2 and report["shares"] == 1
        assert report["refused"] and report["opened"] == report["closed"] == 0
        assert report["verified"] == 0
