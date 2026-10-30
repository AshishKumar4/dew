"""Environments as pure functions (`dew.rl.env`), their replay (`dew.rl.replay`) and PPO over them
(`dew.objectives.rl.control`): each held to a step-by-step loop, an exact index bookkeeping or a
float64 transcription of the arithmetic, apart from Dew's code."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.objectives.rl.control import (
    ActorCritic,
    ControlPPOObjective,
    ControlRollout,
    episode_return,
    tanh_normal_log_prob,
)
from dew.rl.env import Batched, TimeLimit, TimeStep, env_rollout, start
from dew.rl.replay import ReplayBuffer


@struct.dataclass
class Walk:
    """A point on a line pushed by its action toward the origin, paid minus its distance squared, its
    episode terminated where it leaves [-limit, limit]: integer arithmetic on small values, so any
    order computes it exactly. It starts at an integer drawn from [-2, 2]."""

    limit: float = struct.field(pytree_node=False, default=4.0)
    action_size: int = struct.field(pytree_node=False, default=1)

    def reset(self, key):
        x = jax.random.randint(key, (), -2, 3).astype(jnp.float32)
        return x, TimeStep(x[None], jnp.zeros(()), jnp.zeros((), bool), jnp.zeros((), bool))

    def step(self, x, action):
        x = x + jnp.round(2 * action[0])
        return x, TimeStep(x[None], -jnp.square(x), jnp.abs(x) > self.limit, jnp.zeros((), bool))


def pushes(params, observation, key):
    """A policy pushing each point by the sign of a drawn number, half a step a time."""
    return 0.5 * jnp.sign(jax.random.normal(key, (observation.shape[0], 1))), {}


def test_a_rollout_steps_restarts_and_records_as_a_step_by_step_loop_does():
    """`env_rollout` over a batch of time-limited walks takes the steps a Python loop of the same
    environment's `step` takes, each from the same key, restarting an episode that ended, cut or
    terminated, from its own reset before the next step and recording the observation the episode
    ended at, bit for bit."""
    env = Batched(TimeLimit(Walk(), episode_length=3), 5)
    carry = start(env, jax.random.key(0))
    ended, steps = env_rollout(env, pushes, None, carry, 11)

    state, observation, key = carry.state, carry.observation, carry.key
    for index in range(11):
        key, act_key, reset_key = jax.random.split(key, 3)
        action, _ = pushes(None, observation, act_key)
        state, following = env.step(state, action)
        for name in ("observation", "reward", "terminated", "truncated"):
            np.testing.assert_array_equal(getattr(steps.following, name)[index], getattr(following, name))
        np.testing.assert_array_equal(steps.observation[index], observation)
        fresh_state, fresh = env.reset(reset_key)
        done = np.asarray(following.done)
        state = jax.tree.map(lambda new, old, done=done: jnp.where(done.reshape(-1, *(1,) * (new.ndim - 1)),
                                                                   new, old), fresh_state, state)
        observation = jnp.where(done[:, None], fresh.observation, following.observation)
    np.testing.assert_array_equal(ended.observation, observation)
    # A walk starting within 2 of the origin cannot leave [-4, 4] before its third step, where every
    # episode ends, cut or terminated.
    np.testing.assert_array_equal(np.asarray(steps.following.done),
                                  np.broadcast_to((np.arange(11) % 3 == 2)[:, None], (11, 5)))


def test_a_replays_windows_are_consecutive_steps_it_still_holds():
    """After 23 steps of 3 environments into a replay of 10, every window of 4 it samples is 4
    consecutive steps of one environment, all among the last 10 written, and over many draws every
    such window appears."""
    replay = ReplayBuffer.empty({"t": jnp.zeros(3, jnp.int32), "env": jnp.zeros(3, jnp.int32)}, 10)
    for written in itertools.batched(range(23), 5):
        times = np.asarray(written)[:, None].repeat(3, axis=1)
        replay = replay.add({"t": jnp.asarray(times), "env": jnp.broadcast_to(jnp.arange(3), times.shape)})
    windows = jax.jit(lambda key: replay.sample(key, 512, 4))(jax.random.key(1))
    t, env = np.asarray(windows["t"]), np.asarray(windows["env"])
    np.testing.assert_array_equal(np.diff(t, axis=1), 1)
    assert (env == env[:, :1]).all() and t.min() >= 13 and t.max() <= 22
    assert {(int(row[0]), int(rows[0])) for row, rows in zip(t, env, strict=True)} == {
        (start, environment) for start in range(13, 20) for environment in range(3)}


def test_the_squashed_normals_log_density_is_the_change_of_variables():
    """The log-density of tanh(u), u normal, is the normal's less log(1 - tanh(u)^2), within float64
    rounding of that change of variables computed directly."""
    rng = np.random.default_rng(3)
    raw, location = rng.normal(size=(64, 3)) * 2, rng.normal(size=(64, 3))
    scale = rng.uniform(0.1, 2.0, size=(64, 3))
    with jax.enable_x64(new_val=True):
        found = np.asarray(tanh_normal_log_prob(raw, location, scale))
        roundings = chain_roundings(jax.make_jaxpr(tanh_normal_log_prob)(raw, location, scale))
    normal = -0.5 * ((raw - location) / scale) ** 2 - np.log(scale * np.sqrt(2 * np.pi))
    expected = np.sum(normal - np.log1p(-np.tanh(raw) ** 2), axis=-1)
    assert_computes_the_oracle(found, expected, "the squashed normal's log-density", roundings=roundings)


def test_advantages_bootstrap_a_cut_episode_and_end_a_terminated_one():
    """GAE over sequences: the TD error discounts the following value except after a terminal state,
    and the recursion stops at either end of an episode, within float64 rounding of the recursion
    run step by step in Python."""
    rng = np.random.default_rng(4)
    shape = (6, 9)
    values, following, rewards = (rng.normal(size=shape) for _ in range(3))
    terminated, truncated = rng.uniform(size=shape) < 0.15, rng.uniform(size=shape) < 0.15
    batch = {"reward": rewards, "terminated": terminated, "truncated": truncated & ~terminated}
    objective = ControlPPOObjective(ActorCritic(1), 1, discount=0.97, lam=0.9, reward_scaling=2.0)

    def computed(values, following):
        return objective.advantages(values, following, batch)

    with jax.enable_x64(new_val=True):
        advantages, targets = (np.asarray(found) for found in computed(values, following))
        roundings = chain_roundings(jax.make_jaxpr(computed)(values, following))
    expected = np.zeros(shape)
    for row in range(shape[0]):
        running = 0.0
        for t in reversed(range(shape[1])):
            end = terminated[row, t] or batch["truncated"][row, t]
            delta = 2 * rewards[row, t] + 0.97 * (not terminated[row, t]) * following[row, t] - values[row, t]
            running = delta + 0.97 * 0.9 * (not end) * running
            expected[row, t] = running
    assert_computes_the_oracle(advantages, expected, "advantages", roundings=roundings)
    assert_computes_the_oracle(targets, expected + values, "value targets", roundings=roundings + 1)


def test_ppo_raises_the_greedy_policys_return_on_the_walk():
    """A few PPO updates of a fresh actor-critic on walks cut at 8 steps raise its greedy policy's
    mean return above the fresh one's, the collection served as two passes of two minibatches."""
    from dew.data.dataset import Loader
    from dew.training import Trainer

    env = Batched(TimeLimit(Walk(limit=50.0), episode_length=8), 64)
    objective = ControlPPOObjective(ActorCritic(1, (16, 16), (32, 32)), 1, discount=0.9, entropy_cost=1e-3)
    rollout = ControlRollout(objective, env, unroll_length=8, epochs=2, minibatches=2)
    trainer = Trainer(objective, optax.adam(3e-3), key=jax.random.key(5), rollout=rollout)
    initial = trainer.place()[0]
    data = Loader(train=lambda partition: itertools.repeat({"step": np.zeros((4,), np.float32)}), val=None,
                  records=4, batch=4)
    before = episode_return(objective, env, initial.variables, jax.random.key(6), 8)
    final = trainer.fit(data, steps=60, log_every=60)
    after = episode_return(objective, env, final.variables, jax.random.key(6), 8)
    assert after > before, (before, after)


@pytest.mark.parametrize("asked", [{"discount": 1.5}, {"lam": -0.1}, {"clip": 0.0}])
def test_a_discount_lambda_or_clip_outside_its_range_is_refused(asked):
    with pytest.raises(ValueError):
        ControlPPOObjective(ActorCritic(1), 1, **asked)
