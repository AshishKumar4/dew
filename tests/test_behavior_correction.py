"""Token behavior-importance correction against pinned verl and real rollouts."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.objectives.base import Step, scalar_loss
from dew.objectives.rl import GRPOObjective
from dew.rl import behavior_importance_weights, token_log_ratio, token_mean
from dew.rl.surrogate import clipped_surrogate_terms
from test_tool_episodes import build, collect, PROMPT, RESPONSE

FIXTURE = Path(__file__).parent / "fixtures/rl/behavior.npz"


def test_token_correction_matches_pinned_verl_loss_and_gradient():
    """verl d040717 token TIS: weights exact, loss 1.20e-7, gradient 1.50e-8 max differences."""
    with np.load(FIXTURE, allow_pickle=False) as reference:
        old, behavior, current, advantages, mask = [jnp.asarray(reference[name])
            for name in ("old", "behavior", "current", "advantages", "mask")]
        weights = behavior_importance_weights(old, behavior, mask, float(reference["cap"]))
        np.testing.assert_allclose(weights, reference["weights"], atol=1e-12, rtol=2e-6)

        def loss(current):
            # The coefficient changes the measure, while PPO clipping still
            # compares current versus the recorded raw policy.
            terms, _ = clipped_surrogate_terms(token_log_ratio(current, old), advantages, mask)
            return token_mean(weights * terms, mask)

        actual, gradient = jax.value_and_grad(loss)(current)
        np.testing.assert_allclose(actual, reference["corrected"], atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(gradient, reference["gradient"], atol=2e-7, rtol=2e-6)
        assert abs(float(actual) - float(reference["plain"])) > .1
        detached = jax.grad(lambda values: behavior_importance_weights(values, behavior, mask, 2.).sum())(old)
        np.testing.assert_array_equal(detached, 0)


def test_correction_is_explicit_and_uses_actual_recorded_behavior():
    trainer, rollout = build()
    assert isinstance(trainer.objective, GRPOObjective)
    state = trainer.initial_state()
    batch = rollout.project(collect(rollout, state))
    corrected = GRPOObjective(trainer.objective.model, PROMPT + RESPONSE - 1,
                              beta=.05, behavior_importance_cap=2.)
    info = Step(state.microstep, jax.random.key(4), state.ema)

    ordinary_loss, _ = scalar_loss(trainer.objective, state.params, batch, info)
    corrected_loss, _ = scalar_loss(corrected, state.params, batch, info)

    mask = batch["response_mask"]
    weights = np.minimum(np.exp(np.clip(batch["old_log_probs"] - batch["behavior_log_probs"], -20, 20)), 2.)
    expected = np.sum(-batch["advantages"] * mask * weights) / mask.sum()
    np.testing.assert_allclose(corrected_loss, expected, atol=2e-7)
    assert abs(float(corrected_loss) - float(ordinary_loss)) > 1e-4
    without_behavior = {key: value for key, value in batch.items() if key != "behavior_log_probs"}
    unchanged, _ = scalar_loss(trainer.objective, state.params, without_behavior, info)
    np.testing.assert_array_equal(unchanged, ordinary_loss)
    with pytest.raises(ValueError, match="behavior_log_probs"):
        corrected.loss(state.params, without_behavior, info)
    with pytest.raises(ValueError, match="shape"):
        corrected.loss(state.params, {**batch, "behavior_log_probs": jnp.zeros((1, 1))}, info)


def test_corrected_objective_changes_a_real_trainer_update():
    import itertools
    import optax
    from dew.data import Dataset
    from dew.training import Trainer

    original, rollout = build()
    assert isinstance(original.objective, GRPOObjective)
    state = original.initial_state()
    batch = rollout.project(collect(rollout, state))
    batch = {name: np.repeat(value, 2, axis=0) for name, value in batch.items()}
    copies = []
    for cap in (None, 2.):
        objective = GRPOObjective(original.objective.model, PROMPT + RESPONSE - 1,
                                  behavior_importance_cap=cap)
        trainer = Trainer(objective, optax.sgd(.05), key=jax.random.key(19))
        data = Dataset(train=lambda: itertools.repeat(batch), val=None, records=None,
                       batch=batch["input_ids"].shape[0])
        final = trainer.fit(data, steps=1, log_every=1)
        assert int(final.updates) == 1
        copies.append(np.asarray(final.params["params"]["table"]).copy())
    assert np.max(np.abs(copies[0] - copies[1])) > 1e-5
