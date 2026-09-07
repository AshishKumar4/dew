"""PPO actor/critic math against pinned verl and a real episode training run."""

from dataclasses import replace
import itertools
from pathlib import Path

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.data import Dataset
from dew.objectives.base import Step, scalar_loss
from dew.objectives.rl import EpisodeRollout, PPOObjective, PPORollout, ValueHead
from dew.objectives.rl.ppo import OLD_VALUES_KEY, RETURNS_KEY
from dew.rl import clipped_surrogate, clipped_value_loss_terms, gae, token_log_ratio, token_mean
from dew.training import Trainer
from test_tool_episodes import EOS, GROUPS, PROMPT, RESPONSE, SAMPLING, TURNS, VOCAB, Harness, ToolPolicy, verify

FIXTURE = Path(__file__).parent / "fixtures/rl/ppo.npz"


class TokenFeatures(nn.Module):
    @nn.compact
    def hidden_states(self, tokens, train=False, *, attention_mask=None):
        scale = self.param("scale", nn.initializers.ones, (VOCAB,))
        return jax.nn.one_hot(tokens, VOCAB) * scale


def build_ppo():
    objective = PPOObjective(ToolPolicy(), PROMPT + RESPONSE - 1,
                             critic=ValueHead(TokenFeatures()), beta=.03)
    trainer = Trainer(objective, optax.sgd(.05), key=jax.random.key(19))
    source = EpisodeRollout(objective.policy(objective.init(jax.random.key(19))), Harness(), verify,
                            PROMPT, RESPONSE, TURNS, sampling=SAMPLING, groups=GROUPS)
    rollout = PPORollout(objective, source, gamma=.97, lam=.9)
    trainer.rollout = rollout
    return trainer, rollout


def test_ppo_losses_gradients_and_gae_match_pinned_verl():
    """verl d040717: loss/actor gradient 5.97e-8, critic gradient 7.46e-9.

    GAE advantage and return maximum differences were 3.58e-7 and 1.20e-7.
    """
    with np.load(FIXTURE) as reference:
        f = {name: jnp.asarray(reference[name]) for name in reference.files if name != "revision"}
        advantages, returns = gae(f["rewards"], f["old_values"], f["mask"], float(f["gamma"]), float(f["lam"]))
        np.testing.assert_allclose(advantages, f["advantages"], atol=2e-6)
        np.testing.assert_allclose(returns, f["returns"], atol=2e-6)
        def loss(current, predicted):
            actor, _ = clipped_surrogate(token_log_ratio(current, f["old"]), advantages, f["mask"])
            critic = token_mean(clipped_value_loss_terms(predicted, returns, f["old_values"], float(f["clip"])), f["mask"])
            return actor + f["coefficient"] * critic
        actual, gradients = jax.value_and_grad(loss, argnums=(0, 1))(f["current"], f["predicted"])
        assert float(actual) == pytest.approx(float(f["loss"]), abs=2e-6)
        np.testing.assert_allclose(gradients[0], f["actor_gradient"], atol=2e-6)
        np.testing.assert_allclose(gradients[1], f["critic_gradient"], atol=2e-6)
        assert abs(float(f["critic"] - f["unclipped"])) > .01
        np.testing.assert_array_equal(np.asarray(gradients[0])[np.asarray(f["mask"]) == 0], 0)
        np.testing.assert_array_equal(np.asarray(gradients[1])[np.asarray(f["mask"]) == 0], 0)


def test_episode_gae_crosses_turns_without_discounting_observations_or_padding():
    """verl episode GAE: advantage 2.39e-7, return 5.97e-8 maximum differences."""
    trainer, rollout = build_ppo()
    state = trainer.initial_state()
    batch = rollout(state, {"task_id": np.array([31], np.int32)}, jax.random.key(23))
    with np.load(FIXTURE) as reference:
        for name in (OLD_VALUES_KEY, RETURNS_KEY, "advantages"):
            np.testing.assert_allclose(batch[name], reference[f"episode_{name}"], atol=2e-6)
    critic = state.params["params"]["critic"]
    table = np.asarray(critic["backbone"]["scale"]) * np.asarray(critic["value"]["kernel"])[:, 0]
    expected = table[batch["input_ids"][:, PROMPT - 1:PROMPT + RESPONSE - 1]] + np.asarray(critic["value"]["bias"])[0]
    np.testing.assert_allclose(batch[OLD_VALUES_KEY], expected, atol=1e-7)
    changed = {**batch, RETURNS_KEY: batch[RETURNS_KEY] + .25}
    info = Step(jnp.array(0), jax.random.key(1), None)
    no_kl = PPOObjective(ToolPolicy(), PROMPT + RESPONSE - 1, critic=ValueHead(TokenFeatures()))
    before = scalar_loss(no_kl, state.params, batch, info)[0]
    after = scalar_loss(no_kl, state.params, changed, info)[0]
    assert abs(float(before - after)) > .001


def test_ppo_trains_policy_and_critic_with_a_frozen_policy_reference():
    trainer, rollout = build_ppo()
    initial = trainer.initial_state()
    data = Dataset(train=lambda: itertools.repeat({"task_id": np.arange(jax.device_count(), dtype=np.int32)}),
                   val=None, records=None, batch=jax.device_count())
    final = trainer.fit(data, steps=2, log_every=1)
    assert int(final.updates) == 2
    for part in ("policy", "critic"):
        differences = [np.max(np.abs(np.asarray(after) - np.asarray(before))) for before, after in
                       zip(jax.tree.leaves(initial.params["params"][part]), jax.tree.leaves(final.params["params"][part]), strict=True)]
        assert max(differences) > 1e-5
    for before, after in zip(jax.tree.leaves(initial.ema), jax.tree.leaves(final.ema), strict=True):
        np.testing.assert_array_equal(before, after)
    fixed = rollout(initial, {"task_id": np.array([31], np.int32)}, jax.random.key(23))
    keep = fixed["response_mask"] != 0
    old_error = np.mean((np.asarray(rollout.objective.values(initial.params, fixed))[keep] - fixed[RETURNS_KEY][keep]) ** 2)
    new_error = np.mean((np.asarray(rollout.objective.values(final.params, fixed))[keep] - fixed[RETURNS_KEY][keep]) ** 2)
    assert new_error < old_error


@pytest.mark.parametrize("field", [OLD_VALUES_KEY, RETURNS_KEY])
def test_ppo_refuses_missing_critic_targets(field):
    trainer, rollout = build_ppo()
    state = trainer.initial_state()
    batch = rollout(state, {"task_id": np.array([31], np.int32)}, jax.random.key(23))
    batch.pop(field)
    with pytest.raises(ValueError, match=field):
        rollout.objective.loss(state.params, batch, Step(jnp.array(0), jax.random.key(1), None))


def test_composite_objective_and_parameter_gradients_match_verl():
    """verl combined loss differs by 3.26e-8; parameter gradients by at most 1.50e-8."""
    trainer, rollout = build_ppo()
    state = trainer.initial_state()
    batch = rollout(state, {"task_id": np.array([31], np.int32)}, jax.random.key(23))
    with np.load(FIXTURE) as reference:
        params = {"policy": {"table": jnp.asarray(reference["objective_policy"])},
                  "critic": {"backbone": {"scale": jnp.asarray(reference["objective_scale"])},
                             "value": {"kernel": jnp.asarray(reference["objective_kernel"]),
                                       "bias": jnp.asarray(reference["objective_bias"])}}}
        info = Step(jnp.array(0), jax.random.key(1), state.averaged)
        def loss(parameters):
            return scalar_loss(rollout.objective, {"params": parameters}, batch, info)[0]
        actual, gradient = jax.value_and_grad(loss)(params)
        assert float(actual) == pytest.approx(float(reference["objective_loss"]), abs=2e-6)
        observed = {"policy": gradient["policy"]["table"], "kernel": gradient["critic"]["value"]["kernel"],
                    "bias": gradient["critic"]["value"]["bias"], "scale": gradient["critic"]["backbone"]["scale"]}
        for name, value in observed.items():
            difference = np.max(np.abs(np.asarray(value) - reference[f"objective_{name}_gradient"]))
            assert difference < 2e-6, f"verl {name} gradient maximum difference {difference}"


@pytest.mark.parametrize("active", [0, 1])
def test_ppo_refuses_undefined_gae_whitening(active):
    from contextlib import contextmanager
    from dew.objectives.rl import Observation, EpisodeStatus

    class Terminal:
        def __init__(self, sample):
            self.sample = sample

        def reset(self):
            return (Observation((EOS,)) if self.sample < active else
                    Observation((), EpisodeStatus.COMPLETED))

        def step(self, action):
            return Observation((), EpisodeStatus.COMPLETED)

    @contextmanager
    def environment(identity):
        yield Terminal(identity.sample)

    trainer, rollout = build_ppo()
    rollout = replace(rollout, episodes=replace(rollout.episodes, environment=environment, verifier=lambda episode: 1.))
    with pytest.raises(ValueError, match="at least two action tokens"):
        rollout(trainer.initial_state(), {"task_id": np.array([31], np.int32)}, jax.random.key(23))
