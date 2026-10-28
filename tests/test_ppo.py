"""PPO actor/critic math against pinned verl and a real episode training run."""

import functools
import itertools
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from episode_support import SAMPLING, Harness, collect, per_call, verify
from flax import linen as nn
from reference_error import as_given, assert_as_exact
from rl_support import clipped_surrogate, token_mean
from tool_policy import GROUPS, PROMPT, RESPONSE, TURNS, VOCAB, ToolPolicy

from dew.batch import part
from dew.data.dataset import Loader
from dew.objectives.base import Step
from dew.objectives.rl import EpisodeCollector, EpisodeStatus, PPOObjective, PPORollout, ValueHead
from dew.objectives.rl.ppo import OLD_VALUES_KEY, RETURNS_KEY
from dew.objectives.rl.sessions import (
    ADVANTAGES_KEY,
    BEHAVIOR_LOG_PROBS_KEY,
    OLD_LOG_PROBS_KEY,
    RESPONSE_MASK_KEY,
    SEGMENT_IDS_KEY,
    SESSION_WEIGHTS_KEY,
)
from dew.rl import clipped_value_loss_terms, gae, token_log_ratio
from dew.training import Trainer

FIXTURE = Path(__file__).parent / "fixtures/rl/ppo.npz"


class TokenFeatures(nn.Module):
    @nn.compact
    def hidden_states(self, tokens, train=False, *, segment_ids=None, positions=None):
        scale = self.param("scale", nn.initializers.ones, (VOCAB,))
        return jax.nn.one_hot(tokens, VOCAB) * scale


def build_ppo(verifier=verify, gamma=.97, *, lam=.9, split=None, **policy_options):
    objective = PPOObjective(ToolPolicy(), PROMPT + RESPONSE - 1,
                             critic=ValueHead(TokenFeatures()), beta=.03, **policy_options)
    trainer = Trainer(objective, optax.sgd(.05), key=jax.random.key(19), split=split)
    episodes = EpisodeCollector(objective.policy(objective.init(jax.random.key(19))), Harness(), verifier,
                                PROMPT, RESPONSE, TURNS, sampling=SAMPLING)
    rollout = PPORollout(objective, episodes, groups=GROUPS, gamma=gamma, lam=lam)
    trainer.rollout = rollout
    return trainer, rollout


def test_ppo_losses_gradients_and_gae_match_pinned_verl():
    """verl d040717: loss/actor gradient 5.97e-8, critic gradient 7.46e-9.

    GAE advantage and return maximum differences were 3.58e-7 and 1.20e-7.
    """
    with np.load(FIXTURE) as reference:
        f = {name: jnp.asarray(reference[name]) for name in reference.files if name != "revision"}
        advantages, returns = gae(
            f["rewards"], f["old_values"], f["mask"], float(f["gamma"]), float(f["lam"])
        )
        np.testing.assert_allclose(advantages, f["advantages"], atol=2e-6)
        np.testing.assert_allclose(returns, f["returns"], atol=2e-6)
        def loss(current, predicted):
            actor, _ = clipped_surrogate(token_log_ratio(current, f["old"]), advantages, f["mask"])
            critic = token_mean(
                clipped_value_loss_terms(predicted, returns, f["old_values"], float(f["clip"])), f["mask"]
            )
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
    episodes = collect(rollout.episodes, state)
    with np.load(FIXTURE) as reference:
        # The fixture holds verl's GAE in the old one-row-per-call layout,
        # whose unsupported slots carry the recursion's neighbouring values.
        actions = per_call(batch, "response_mask", episodes) != 0
        for name in (OLD_VALUES_KEY, RETURNS_KEY, "advantages"):
            np.testing.assert_allclose(per_call(batch, name, episodes)[actions],
                                       reference[f"episode_{name}"][actions], atol=2e-6)
    critic = state.variables["params"]["critic"]
    table = np.asarray(critic["backbone"]["scale"]) * np.asarray(critic["value"]["kernel"])[:, 0]
    previous = np.concatenate([np.zeros_like(batch["input_ids"][:, :1]), batch["input_ids"][:, :-1]], axis=1)
    expected = np.where(batch["response_mask"] != 0,
                        table[previous] + np.asarray(critic["value"]["bias"])[0], 0)
    np.testing.assert_allclose(batch[OLD_VALUES_KEY], expected, atol=1e-7)
    changed = {**batch, RETURNS_KEY: batch[RETURNS_KEY] + .25}
    info = Step(jnp.array(0), jax.random.key(1), None)
    no_kl = PPOObjective(ToolPolicy(), PROMPT + RESPONSE - 1, critic=ValueHead(TokenFeatures()))
    before = no_kl.scalar_loss(state.variables, batch, info)[0]
    after = no_kl.scalar_loss(state.variables, changed, info)[0]
    assert abs(float(before - after)) > .001


def episode_returns(rewards, values, gamma, lam, dtype):
    """verl's masked GAE returns over one episode's action ids in draw order, in `dtype`: each
    id's `rewards` and the critic's `values`, with no bootstrap after the last."""
    rewards, values = np.asarray(rewards, dtype), np.asarray(values, dtype)
    advantage, following, out = dtype(0), dtype(0), np.zeros_like(values)
    for t in range(len(values) - 1, -1, -1):
        delta = rewards[t] + dtype(gamma) * following - values[t]
        advantage = delta + dtype(gamma * lam) * advantage
        out[t], following = advantage + values[t], values[t]
    return out


STEP_REWARD_SEEDS = 8
"""The step rewards' returns' declared seeds for check 1, each a rollout (`assert_as_exact`)."""


def test_each_step_reward_lands_on_the_last_action_of_the_call_it_paid_for():
    """A square tool that pays 0.5 for its tool step and 0.25 for the answer step, beside the
    verifier's reward: GAE's returns over each episode's actions are verl's with each step reward
    on the last action id of its call and the verifier's on the episode's last, on par with the same
    recursion in float32 from float64 over rollouts each declared seed draws (check 1)."""
    trainer, rollout = build_ppo()
    rollout = replace(rollout, episodes=replace(rollout.episodes, environment=Harness(paid=(0.5, 0.25))))
    state = trainer.initial_state()
    task = {"task_id": np.array([31], np.int32)}

    @functools.cache
    def drawn(seed):
        """One rollout's returns and, per completed episode, its rewards and values."""
        batch = rollout(state, task, jax.random.key(seed))
        episodes = collect(rollout.episodes, state, seed, tasks=task["task_id"])
        mine, rewards_of, values_of = [], [], []
        for index, episode in enumerate(episodes):
            ids = [(turn, position) for turn, transition in enumerate(episode.transitions)
                   for position in range(len(transition.action.tokens))]
            rows = [index * TURNS + turn for turn, _ in ids]
            if episode.status is not EpisodeStatus.COMPLETED:
                continue
            rewards = np.zeros(len(ids))
            for turn, transition in enumerate(episode.transitions):
                last = max(at for at, (call, _) in enumerate(ids) if call == turn)
                rewards[last] += transition.observation.reward
            rewards[-1] += episode.reward
            positions = [position for _, position in ids]
            rewards_of.append(rewards)
            values_of.append(per_call(batch, OLD_VALUES_KEY, episodes)[rows, positions])
            mine.append(per_call(batch, RETURNS_KEY, episodes)[rows, positions])
        assert mine
        return np.concatenate(mine), rewards_of, values_of

    def recursion(dtype):
        return as_given(lambda seed: {"returns": np.concatenate([
            episode_returns(rewards, values, rollout.gamma, rollout.lam, dtype)
            for rewards, values in zip(*drawn(int(seed))[1:], strict=True)])})

    assert_as_exact(as_given(lambda seed: {"returns": drawn(int(seed))[0]}), recursion(np.float32),
                    lambda seed: (np.int32(seed),), "episode returns with step rewards",
                    seeds=STEP_REWARD_SEEDS, truth_fn=recursion(np.float64))


def test_episode_advantages_hold_at_values_and_rewards_far_above_them():
    """PPORollout whitens the GAE recursion's own advantages, from centred moments over every process.
    With every value and every reward 1e8 higher, all exact in float32, each temporal difference is
    the same, and so is each whitened advantage, bit for bit, where returns less values would round
    every advantage below 4 away."""
    def advantages(value_shift, reward_shift, lam=.9):
        trainer, rollout = build_ppo(lambda episode: reward_shift + 8 * verify(episode), gamma=1., lam=lam)
        state = trainer.initial_state()
        params = state.variables["params"]
        value = params["critic"]["value"]
        level = {"kernel": jnp.zeros_like(value["kernel"]), "bias": jnp.full_like(value["bias"], value_shift)}
        variables = {**state.variables, "params": {**params, "critic": {**params["critic"], "value": level}}}
        batch = rollout(state.replace(variables=variables), {"task_id": np.array([31], np.int32)},
                        jax.random.key(23))
        actions = batch["response_mask"] != 0
        assert np.all(batch[OLD_VALUES_KEY][actions] == np.float32(value_shift))
        return batch["advantages"][actions]

    level = advantages(0., 0.)
    assert np.unique(level).size > 2
    np.testing.assert_array_equal(advantages(1e8, 1e8), level)


def whitened(rewards, values, gamma, lam, dtype):
    """verl's masked GAE over each episode's actions (`episode_returns` less the values), then
    whitened over all of them in `dtype`: centred on their mean, scaled by their unbiased deviation,
    as Tunix and verl whiten."""
    raw = np.concatenate([episode_returns(reward, value, gamma, lam, dtype) - np.asarray(value, dtype)
                          for reward, value in zip(rewards, values, strict=True)])
    count = dtype(raw.size)
    mean = raw.sum(dtype=dtype) / (count + dtype(1e-8))
    variance = np.square(raw - mean).sum(dtype=dtype) / (count + dtype(1e-8))
    return (raw - mean) / np.sqrt(variance * (count / (count - 1)) + dtype(1e-8))


PPO_SHIFT_SEEDS = 16
"""The shifted-reward site's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for`
(2026-10-21, on armada)."""


def test_whitened_advantages_of_rewards_far_above_the_values_are_as_exact_as_the_equations():
    """Undiscounted (gamma = lambda = 1), every reward 10001 higher than the values at zero: each
    advantage is 10001 higher and whitening, which a shift leaves alone, gives the same advantages,
    where moments of the raw squares cancel to a wrong variance. Over rollouts each declared seed
    draws, PPORollout's whitened advantages are as exact as verl's equations in float32 (check 1),
    truth the same equations in float64 on the same rewards and values."""
    trainer, rollout = build_ppo(lambda episode: 10001. + 8 * verify(episode), gamma=1., lam=1.)
    initial = trainer.initial_state()
    params = initial.variables["params"]
    value = params["critic"]["value"]
    zero = {"kernel": jnp.zeros_like(value["kernel"]), "bias": jnp.zeros_like(value["bias"])}
    state = initial.replace(variables={**initial.variables,
                                       "params": {**params, "critic": {**params["critic"], "value": zero}}})
    # Sixteen prompts a draw: over one, the reference compared with itself did not resolve within 512.
    task = {"task_id": np.arange(16, dtype=np.int32)}

    @functools.cache
    def drawn(seed):
        """One rollout's whitened advantages and, per completed episode, its rewards and values."""
        batch = rollout(state, task, jax.random.key(seed))
        episodes = collect(rollout.episodes, state, seed, tasks=task["task_id"])
        advantages, rewards, values = [], [], []
        for index, episode in enumerate(episodes):
            ids = [(turn, position) for turn, transition in enumerate(episode.transitions)
                   for position in range(len(transition.action.tokens))]
            if not ids:
                continue
            rows, positions = [index * TURNS + turn for turn, _ in ids], [position for _, position in ids]
            # An episode the rollout truncated scores nothing, as PPORollout scores it.
            paid = np.zeros(len(ids))
            paid[-1] = 0. if episode.reward is None else episode.reward
            rewards.append(paid)
            values.append(per_call(batch, OLD_VALUES_KEY, episodes)[rows, positions])
            advantages.append(per_call(batch, "advantages", episodes)[rows, positions])
        assert sum(len(reward) for reward in rewards) == np.count_nonzero(batch["response_mask"])
        return np.concatenate(advantages), rewards, values

    def equations(dtype):
        return as_given(lambda seed: {"advantages": whitened(*drawn(int(seed))[1:], 1., 1., dtype)})

    assert_as_exact(as_given(lambda seed: {"advantages": drawn(int(seed))[0]}), equations(np.float32),
                    lambda seed: (np.int32(seed),), "whitened advantages of shifted rewards",
                    seeds=PPO_SHIFT_SEEDS, truth_fn=equations(np.float64))


def test_ppo_trains_policy_and_critic_with_a_frozen_policy_reference():
    trainer, rollout = build_ppo()
    initial = trainer.initial_state()
    data = Loader(
        train=lambda partition: itertools.repeat({"task_id": np.arange(jax.device_count(), dtype=np.int32)}),
        val=None,
        records=None,
        batch=jax.device_count(),
    )
    final = trainer.fit(data, steps=2, log_every=1)
    assert int(final.updates) == 2
    for name in ("policy", "critic"):
        differences = [
            np.max(np.abs(np.asarray(after) - np.asarray(before)))
            for before, after in zip(
                jax.tree.leaves(initial.variables["params"][name]),
                jax.tree.leaves(final.variables["params"][name]),
                strict=True,
            )
        ]
        assert max(differences) > 1e-5
    for before, after in zip(jax.tree.leaves(initial.ema), jax.tree.leaves(final.ema), strict=True):
        np.testing.assert_array_equal(before, after)
    fixed = rollout(initial, {"task_id": np.array([31], np.int32)}, jax.random.key(23))
    keep = fixed["response_mask"] != 0
    old_error = np.mean(
        (np.asarray(rollout.objective.values(initial.variables, fixed))[keep] - fixed[RETURNS_KEY][keep]) ** 2
    )
    new_error = np.mean(
        (np.asarray(rollout.objective.values(final.variables, fixed))[keep] - fixed[RETURNS_KEY][keep]) ** 2
    )
    assert new_error < old_error


def test_muonclip_steps_both_the_actor_and_a_transformer_critic():
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.optim import OptimConfig

    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1,
                              num_heads=2, mlp_features=16, max_seq_len=8, qk_norm=False)
    objective = PPOObjective(model, 5, critic=ValueHead(model))
    optimizer = OptimConfig(optimizer="muonclip", learning_rate=0.,
                            optimizer_opts={"qk_clip_threshold": .01}).build(1)
    trainer = Trainer(objective, optimizer, key=jax.random.key(0))
    initial = trainer.initial_state()
    ids = jnp.tile(jnp.arange(6, dtype=jnp.int32), (jax.device_count(), 1))
    mask = jnp.broadcast_to(jnp.arange(6) >= 3, ids.shape).astype(jnp.float32)
    batch = {"input_ids": ids, "text_segment_ids": jnp.ones_like(ids),
             "text_positions": jnp.broadcast_to(jnp.arange(6), ids.shape),
             "response_mask": mask, "advantages": mask,
             "behavior_log_probs": jnp.full(ids.shape, -3.),
             "old_values": jnp.zeros_like(mask), "returns": mask}
    observing = Step(jnp.array(0), jax.random.key(1), None, qk_stats=True)
    _, aux = objective.loss(initial.variables, batch, observing)
    assert aux.qk_stats is not None and set(aux.qk_stats) == {"policy", "critic"}
    for name, path in (("policy", ()), ("critic", ("backbone",))):
        stats = aux.qk_stats[name]
        for component in path:
            stats = stats[component]
        assert np.max(stats["layers_0"]["self_attn"]["max_logits"][0]) > .01
    start = jax.tree.map(np.asarray, initial.variables["params"])
    compiled = trainer.compile(initial, batch)
    final, *_ = compiled(initial, batch)
    assert int(final.updates) == 1
    for name, path in (("policy", ()), ("critic", ("backbone",))):
        before, after = start[name], final.variables["params"][name]
        for component in path:
            before, after = before[component], after[component]
        for projection in ("q_proj", "k_proj"):
            own = before["layers_0"]["self_attn"][projection]["kernel"]
            changed = after["layers_0"]["self_attn"][projection]["kernel"]
            assert np.linalg.norm(changed) < np.linalg.norm(own)


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
            return rollout.objective.scalar_loss({"params": parameters}, batch, info)[0]
        actual, gradient = jax.value_and_grad(loss)(params)
        assert float(actual) == pytest.approx(float(reference["objective_loss"]), abs=2e-6)
        observed = {
            "policy": gradient["policy"]["table"],
            "kernel": gradient["critic"]["value"]["kernel"],
            "bias": gradient["critic"]["value"]["bias"],
            "scale": gradient["critic"]["backbone"]["scale"],
        }
        for name, value in observed.items():
            difference = np.max(np.abs(np.asarray(value) - reference[f"objective_{name}_gradient"]))
            assert difference < 2e-6, f"verl {name} gradient maximum difference {difference}"


def ppo_equations(objective, options):
    """verl's PPO loss written out for the actor `options` the scenarios set, over the policy's and the
    frozen reference's scores and the critic's values (the models' own forwards), every term at the
    scores' precision: the dual-clipped surrogate (`compute_policy_loss`), the k3 KL to the reference
    clamped as `kl_penalty_forward` clamps it, the rollout's sequence and geometric rejection
    (`compute_rollout_rejection_mask`, seq_sum_k1 and seq_mean_k1) over each packed chain, token-mean or
    session-mean weights, and half the larger squared value error against the clipped one
    (`compute_value_loss`), added at the value coefficient. Log ratios are clamped to +-20 before their
    exponential, as verl's `negative_approx_kl` is."""
    actor = objective.actor

    def chains(values, mask, segments):
        """Each token's chain's total of `values` and count of trained tokens."""
        ids = jnp.arange(segments.shape[1] + 1)
        member = (segments[..., None] == ids) & (mask[..., None] != 0)
        totals = jnp.sum(jnp.where(member, values[..., None], 0), axis=1, keepdims=True)
        counts = jnp.sum(member, axis=1, keepdims=True).astype(values.dtype)
        pick = segments[..., None] == ids
        return jnp.sum(jnp.where(pick, totals, 0), -1), jnp.sum(jnp.where(pick, counts, 0), -1)

    def loss(variables, ema, batch):
        policy = actor.packed_log_probs(part(variables, "policy"), batch)
        frozen = actor.packed_log_probs(part(ema, "policy"), batch)
        dtype = policy.dtype
        columns = (RESPONSE_MASK_KEY, OLD_LOG_PROBS_KEY, BEHAVIOR_LOG_PROBS_KEY, ADVANTAGES_KEY)
        mask, old, behavior, advantages = (jnp.asarray(batch[key]).astype(dtype) for key in columns)
        segments = jnp.asarray(batch[SEGMENT_IDS_KEY])
        keep = jnp.ones_like(mask)
        for band, geometric in ((options.get("sequence_mask"), False), (options.get("geometric_mask"), True)):
            if band is not None:
                total, count = chains(jnp.clip(behavior - old, -20, 20), mask, segments)
                statistic = total / (count + 1e-8) if geometric else total
                inside = (statistic >= np.log(band[0])) & (statistic <= np.log(band[1]))
                keep = keep * jnp.where(inside | (mask == 0), 1, 0).astype(dtype)
        ratio = jnp.exp(jnp.clip(policy - old, -20, 20))
        bounded_ratio = jnp.clip(ratio, 1 - actor.epsilon_low, 1 + actor.epsilon_high)
        clipped = jnp.maximum(-advantages * ratio, -advantages * bounded_ratio)
        surrogate = jnp.where(advantages < 0, jnp.minimum(-advantages * actor.dual_clip, clipped), clipped)
        drift = jnp.clip(frozen - policy, -20, 20)
        kl = jnp.clip(jnp.exp(drift) - drift - 1, -10, 10)
        effective = mask * keep
        weights = (effective if options.get("aggregation", "token-mean") == "token-mean"
                   else jnp.asarray(batch[SESSION_WEIGHTS_KEY]).astype(dtype) * keep * (effective != 0))
        mass = jax.lax.stop_gradient(jnp.sum(weights))

        def aggregated(values):
            # verl's agg_loss: each term's weighted sum over the mass, zero where nothing is kept.
            return jnp.where(mass > 0, jnp.sum(jnp.where(weights != 0, values, 0) * weights)
                             / jnp.where(mass > 0, mass, 1), 0)

        policy_loss = aggregated(surrogate) + actor.beta * aggregated(kl)
        values = objective.values(variables, batch).astype(dtype)
        returns, recorded = (jax.lax.stop_gradient(jnp.asarray(batch[key]).astype(dtype))
                             for key in (RETURNS_KEY, OLD_VALUES_KEY))
        bounded = jnp.clip(values, recorded - objective.value_clip, recorded + objective.value_clip)
        value_terms = .5 * jnp.maximum(jnp.square(values - returns), jnp.square(bounded - returns))
        value_loss = jnp.sum(jnp.where(mask != 0, value_terms, 0) * mask) / jnp.sum(mask)
        return policy_loss + objective.value_coefficient * value_loss

    return loss


ACTOR_CONTROL_SEEDS = 8
"""Check 1's declared seeds for PPO's composed loss under each actor control, sized by `seeds_for` paired
on armada: PPO and the equations score one rollout through one model, sharing its upstream rounding."""


@pytest.mark.parametrize("options", [
    {},
    {"aggregation": "session-mean"},
    {"sequence_mask": (.8, 1.2)},
    {"geometric_mask": (.8, 1.2)},
    {"aggregation": "session-mean", "sequence_mask": (.8, 1.2), "geometric_mask": (.8, 1.2)},
], ids=["default", "sessions", "sequence", "geometric", "combined"])
def test_actor_controls_leave_the_critics_normalization_and_gradient_unchanged(options):
    """PPO's loss and every gradient leaf under each actor control are as exact as verl's PPO loss written
    out (`ppo_equations`) over rollouts each declared seed draws (check 1), the truth those equations in
    float64 over the same rollout and weights: the controls reshape the actor's terms and leave the
    critic's normalization and gradient its own. A rollout whose behavior sits far from its old policy
    is rejected whole by either band, so the actor contributes nothing and the critic carries the loss."""
    trainer, rollout = build_ppo(**options)
    objective = rollout.objective
    state = trainer.initial_state()
    # Sixteen prompts a draw: over one, the critic's bias gradient, a sum whose signs cancel, did not
    # resolve within 512 seeds.
    rolled = functools.cache(lambda seed: rollout(state, {"task_id": np.arange(16, dtype=np.int32)},
                                                  jax.random.key(seed)))

    def draw(seed):
        batch = rolled(int(seed))
        old = objective.actor.packed_log_probs(part(state.variables, "policy"), batch)
        mask = jnp.asarray(batch[RESPONSE_MASK_KEY])
        return {**batch, OLD_LOG_PROBS_KEY: old, BEHAVIOR_LOG_PROBS_KEY: old - 2 * mask},

    step = Step(jnp.array(0), jax.random.key(1), state.averaged)

    def dew(batch):
        value, gradient = jax.value_and_grad(
            lambda params: objective.scalar_loss({**state.variables, "params": params}, batch, step)[0])(
            state.variables["params"])
        return {"loss": value, "gradient": gradient}

    def equations(variables, ema, batch):
        written = ppo_equations(objective, options)
        def scored(params):
            return written({**variables, "params": params}, ema, batch)
        value, gradient = jax.value_and_grad(scored)(variables["params"])
        return {"loss": value, "gradient": gradient}

    def wider(leaf):
        return leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf

    # The gate runs the truth in float64 on the draw widened; the state it reads beside the draw too.
    assert_as_exact(dew, lambda batch: equations(state.variables, state.averaged, batch), draw,
                    f"PPO actor controls {options}", seeds=ACTOR_CONTROL_SEEDS,
                    truth_fn=lambda batch: equations(
                        *jax.tree.map(wider, (state.variables, state.averaged, batch))))
    if "sequence_mask" in options or "geometric_mask" in options:
        _, aux = objective.scalar_loss(state.variables, draw(0)[0], step)
        assert float(aux.metrics["pg"]) == 0


@pytest.mark.parametrize("coefficient", [0., .5])
@pytest.mark.parametrize("split", [1, 2])
@pytest.mark.parametrize("rejection", ["sequence", "band"])
def test_rejected_actor_tokens_leave_only_the_weighted_critic_support(coefficient, split, rejection):
    """Every actor token is rejected, by a sequence mask against the old log probs or, with none,
    by the behavior importance band against the live ones; the critic still trains on all of its
    own, unless its coefficient is zero."""
    options = {"sequence_mask": (.8, 1.2)} if rejection == "sequence" else {"behavior_importance": (.5, 2.)}
    trainer, rollout = build_ppo(split=split, **options)
    objective = rollout.objective
    objective.value_coefficient = coefficient
    state = trainer.initial_state()
    batch = rollout(state, {"task_id": np.arange(31, 35, dtype=np.int32)}, jax.random.key(23))
    old = objective.actor.packed_log_probs(part(state.variables, "policy"), batch)
    batch = {name: value for name, value in batch.items() if name != OLD_LOG_PROBS_KEY}
    if rejection == "sequence":
        batch[OLD_LOG_PROBS_KEY] = old
    batch[BEHAVIOR_LOG_PROBS_KEY] = old - 2 * jnp.asarray(batch["response_mask"])
    before = jax.tree.map(np.asarray, state.variables["params"])
    state, _, _, _, accepted, _ = trainer.compile(state, batch)(state, batch)
    assert bool(accepted)
    assert int(state.updates) == int(coefficient > 0)
    for start, end in zip(jax.tree.leaves(before["policy"]),
                          jax.tree.leaves(state.variables["params"]["policy"]), strict=True):
        np.testing.assert_array_equal(start, end)
    changed = any(not np.array_equal(start, end) for start, end in zip(
        jax.tree.leaves(before["critic"]), jax.tree.leaves(state.variables["params"]["critic"]), strict=True))
    assert changed == (coefficient > 0)
