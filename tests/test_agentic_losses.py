"""The packed GRPO loss variants against verl 12ebe0c and Agent Lightning ff94575.

tests/fixtures/rl/agentic.npz holds four sequences as verl sees them, one
per row, and what verl's `core_algos` and `rollout_corr_helper` (and Agent
Lightning's per-rollout normalization) compute from them, gradients
included; tools/parity_agentic.py writes it. Here the same sequences are
laid out as packed chains, two to a row, so every sequence reduction has to
find its chain through the segment ids, and one rollout spans both rows.
The policy's log-probabilities are the parameters, so the gradient each
reference took with torch autograd is the gradient Dew's loss takes.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Step, mean_loss
from dew.objectives.rl import GRPOObjective
from dew.objectives.rl.rollout import (
    ADVANTAGES_KEY,
    BEHAVIOR_LOG_PROBS_KEY,
    IDS_KEY,
    OLD_LOG_PROBS_KEY,
    RESPONSE_MASK_KEY,
)
from dew.objectives.rl.rollouts import POSITIONS_KEY, ROLLOUT_WEIGHTS_KEY, SEGMENT_IDS_KEY

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rl" / "agentic.npz"
CHAIN = 7
"""A chain is one prompt id and the six scored ids."""
WIDTH = 2 * CHAIN
TOLERANCE = 2e-6


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURE, allow_pickle=False))


def _place(values, fill=0.0):
    """Four [6] sequences into two packed rows, chains at columns 1-6 and 8-13."""
    values = np.asarray(values)
    out = np.full((2, WIDTH), fill, values.dtype)
    for sequence in range(4):
        row, column = divmod(sequence, 2)
        out[row, column * CHAIN + 1:(column + 1) * CHAIN] = values[sequence]
    return out


def _take(grid):
    grid = np.asarray(grid)
    return np.stack([grid[s // 2, (s % 2) * CHAIN + 1:(s % 2 + 1) * CHAIN] for s in range(4)])


def _batch(reference, rollouts):
    mask = reference["mask"]
    counts = {}
    for sequence, rollout in enumerate(rollouts):
        counts[rollout] = counts.get(rollout, 0) + mask[sequence].sum()
    weights = np.stack([mask[s] / counts[rollouts[s]] for s in range(4)])
    segments = np.repeat([[1] * CHAIN + [2] * CHAIN], 2, axis=0).astype(np.int32)
    return {
        IDS_KEY: np.ones((2, WIDTH), np.int32), SEGMENT_IDS_KEY: segments,
        POSITIONS_KEY: np.tile(np.arange(CHAIN, dtype=np.int32), (2, 2)),
        RESPONSE_MASK_KEY: _place(mask), OLD_LOG_PROBS_KEY: _place(reference["old"]),
        BEHAVIOR_LOG_PROBS_KEY: _place(reference["behavior"]),
        ADVANTAGES_KEY: _place(np.repeat(reference["advantages"][:, None], 6, axis=1)),
        ROLLOUT_WEIGHTS_KEY: _place(weights.astype(np.float32)),
    }


class Fixed(GRPOObjective):
    """GRPO whose policy log-probabilities are its parameters."""

    def packed_log_probs(self, params, batch):
        return jnp.where(jnp.asarray(batch[RESPONSE_MASK_KEY]) != 0, params["current"], 0.0)


def _objective(reference, **options):
    model = CausalTransformer(vocab_size=8, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=8, max_seq_len=WIDTH, dtype="float32", attention_impl="xla")
    return Fixed(model, WIDTH - 1, epsilon_low=float(reference["epsilon_low"]),
                 epsilon_high=float(reference["epsilon_high"]),
                 dual_clip=float(reference["dual_clip"]), **options)


def _run(reference, rollouts=("a", "b", "c", "d"), **options):
    objective = _objective(reference, **options)
    batch = _batch(reference, list(rollouts))
    step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None)

    def scalar(current):
        loss, aux = objective.loss({"current": current}, batch, step)
        return mean_loss(loss)[0], aux.metrics

    (loss, metrics), grad = jax.value_and_grad(scalar, has_aux=True)(jnp.asarray(_place(reference["current"])))
    return float(loss), _take(grad), {key: float(value) for key, value in metrics.items()}


def _check(reference, name, loss, grad, metrics=None):
    assert loss == pytest.approx(float(reference[f"{name}_loss"]), abs=TOLERANCE)
    np.testing.assert_allclose(grad * reference["mask"], reference[f"{name}_grad"], atol=TOLERANCE)
    for key in ("pg_clipfrac", "ppo_kl", "pg_clipfrac_lower"):
        if metrics is not None and f"{name}_{key}" in reference:
            assert metrics[f"actor/{key}"] == pytest.approx(float(reference[f"{name}_{key}"]), abs=TOLERANCE)


@pytest.mark.parametrize("policy_loss", ["ppo", "gspo", "cispo"])
@pytest.mark.parametrize(("aggregation", "suffix"), [("token-mean", "token"), ("rollout-mean", "sequence")])
def test_each_policy_loss_matches_verl_over_packed_chains(reference, policy_loss, aggregation, suffix):
    """Rollout-mean with one rollout per chain is verl's seq-mean-token-mean.
    Largest observed difference: under 3e-7 on every loss and gradient."""
    loss, grad, metrics = _run(reference, policy_loss=policy_loss, aggregation=aggregation)
    _check(reference, f"{policy_loss}_{suffix}", loss, grad, metrics)


def test_gspo_pools_each_chain_on_its_own(reference):
    """Pooling the whole row instead of its chain moves the GSPO loss."""
    loss, _, _ = _run(reference, policy_loss="gspo")
    batch = _batch(reference, ["a", "b", "c", "d"])
    batch[SEGMENT_IDS_KEY] = np.where(batch[SEGMENT_IDS_KEY] > 0, 1, 0).astype(np.int32)
    objective = _objective(reference, policy_loss="gspo")
    merged, _ = objective.loss({"current": jnp.asarray(_place(reference["current"]))}, batch,
                               Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None))
    assert abs(float(mean_loss(merged)[0]) - loss) > 1e-3


def test_token_corrections_match_verl_rollout_correction(reference):
    """TIS caps the proximal/behavior ratio, IcePop zeroes it outside the band."""
    loss, grad, metrics = _run(reference, behavior_importance_cap=float(reference["tis_cap"]))
    _check(reference, "ppo_tis_token", loss, grad)
    assert metrics["mismatch/ess"] == pytest.approx(float(reference["tis_ess"]), abs=1e-5)
    low, high = (float(value) for value in reference["band"])
    loss, grad, metrics = _run(reference, behavior_band=(low, high))
    _check(reference, "ppo_band_token", loss, grad)
    assert metrics["mismatch/ess"] == pytest.approx(float(reference["band_ess"]), abs=1e-5)
    assert metrics["masked/band"] == pytest.approx(float(reference["band_oob"]), abs=1e-6)
    assert metrics["mismatch/kl"] == pytest.approx(float(reference["offpolicy_kl"]), abs=1e-6)
    assert metrics["mismatch/k3_kl"] == pytest.approx(float(reference["offpolicy_k3_kl"]), abs=1e-6)


@pytest.mark.parametrize("name", ["sequence", "geometric"])
@pytest.mark.parametrize(("aggregation", "suffix"), [("token-mean", "token"), ("rollout-mean", "sequence")])
def test_sequence_masks_match_verl_rejection(reference, name, aggregation, suffix):
    """seq_sum_k1 and seq_mean_k1 reject whole chains; the fixture's bands
    reject rows 1 and 2 for the sum and row 1 alone for the mean."""
    band = tuple(float(value) for value in reference[f"{name}_band"])
    loss, grad, metrics = _run(reference, aggregation=aggregation, **{f"{name}_mask": band})
    _check(reference, f"ppo_{name}_{suffix}", loss, grad)
    rejected = 1 - reference[f"{name}_mask"]
    expected = (rejected * reference["mask"]).sum() / reference["mask"].sum()
    assert metrics[f"masked/{name}"] == pytest.approx(expected, abs=1e-6)


def test_rollout_mean_matches_agent_lightning_per_rollout_mean(reference):
    """Chains 0 and 3 are one rollout, placed in different rows. Agent
    Lightning divides by the row count where Dew divides by the rollout
    count, a constant factor: 4 rows, 3 rollouts."""
    rollouts = [str(value) for value in reference["rollouts"]]
    loss, grad, _ = _run(reference, rollouts=rollouts, aggregation="rollout-mean")
    factor = len(set(rollouts)) / len(rollouts)
    assert loss * factor == pytest.approx(float(reference["per_rollout_loss"]), abs=TOLERANCE)
    np.testing.assert_allclose(grad * factor * reference["mask"], reference["per_rollout_grad"], atol=TOLERANCE)


def test_the_fixture_names_its_references(reference):
    assert str(reference["verl_revision"]) == "12ebe0cb4d300c58449fb6c675379e8700015c51"
    assert str(reference["lightning_revision"]) == "ff9457587fb6ec900e16e93be9ad2d77409afa08"
