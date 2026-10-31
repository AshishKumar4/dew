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
from dew.rl.simulator import STACKED


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


@struct.dataclass
class Counter:
    """A count raised by one each step and paid as its value, its episode ended by the environment
    itself: terminated at `stop`, truncated at `cut`."""

    stop: int = struct.field(pytree_node=False)
    cut: int = struct.field(pytree_node=False)
    action_size: int = struct.field(pytree_node=False, default=1)

    def reset(self, key):
        return jnp.zeros(()), TimeStep(jnp.zeros(1), jnp.zeros(()), jnp.zeros((), bool), jnp.zeros((), bool))

    def step(self, x, action):
        x = x + 1
        return x, TimeStep(x[None], x, x == self.stop, x == self.cut)


@pytest.mark.parametrize(("stop", "cut", "repeat"),
                         [(99, 2, 1), (2, 99, 3), (99, 2, 3), (99, 99, 2), (3, 3, 2)])
def test_a_time_limit_keeps_the_ends_its_environment_reports(stop, cut, repeat):
    """A time limit of 4 steps over a count its environment terminates or truncates by itself, each step
    repeating its action: every step's observation, reward and ends are a step-by-step loop's until the
    episode ends, the environment's own end reported where it falls, a step's repeats after it neither
    stepping nor paying, and the limit's own cut where the environment's falls later."""
    env = TimeLimit(Counter(stop, cut), episode_length=4, action_repeat=repeat)
    state, _ = env.reset(jax.random.key(0))
    step = jax.jit(env.step)
    x = 0
    for _ in range(4):
        reward, terminated, truncated = 0, False, False
        for _ in range(repeat):
            if not (terminated or truncated):
                x += 1
                reward, terminated, truncated = reward + x, x == stop, x == cut
        state, stepped = step(state, jnp.zeros(1))
        truncated = not terminated and (truncated or int(state.steps) >= 4)
        np.testing.assert_array_equal(np.asarray(stepped.observation), [x])
        assert (float(stepped.reward), bool(stepped.terminated), bool(stepped.truncated)) == (
            reward, terminated, truncated)
        if terminated or truncated:
            break


def pushes(params, observation, key, memory):
    """A policy pushing each point by the sign of a drawn number, half a step a time."""
    return 0.5 * jnp.sign(jax.random.normal(key, (observation.shape[0], 1))), {}, None


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
        action, _, _ = pushes(None, observation, act_key, None)
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
    # Within 3 of zero, where 1 - tanh(u)^2 keeps the change of variables' own float64 digits.
    raw, location = rng.uniform(-3, 3, size=(64, 3)), rng.normal(size=(64, 3))
    scale = rng.uniform(0.1, 2.0, size=(64, 3))
    with jax.enable_x64(new_val=True):
        found = np.asarray(tanh_normal_log_prob(raw, location, scale))
        roundings = chain_roundings(jax.make_jaxpr(tanh_normal_log_prob)(raw, location, scale))
    normal = -0.5 * ((raw - location) / scale) ** 2 - np.log(scale * np.sqrt(2 * np.pi))
    expected = np.sum(normal - np.log1p(-np.tanh(raw) ** 2), axis=-1)
    assert_computes_the_oracle(found, expected, "the squashed normal's log-density", roundings=roundings)


def test_an_actor_critics_spread_multiplies_its_policys_scale():
    """An actor-critic of spread 0.2 answers the location of one of spread 1 with the same weights, and a
    fifth of its scale, bit for bit."""
    observation = jnp.asarray(np.random.default_rng([532, 10]).normal(size=(8, 5)), jnp.float32)
    held = ActorCritic(3).init(jax.random.key(0), observation)
    location, scale = ActorCritic(3).apply(held, observation, method=ActorCritic.policy)
    narrow = ActorCritic(3, spread=0.2)
    narrow_location, narrow_scale = narrow.apply(held, observation, method=ActorCritic.policy)
    np.testing.assert_array_equal(np.asarray(narrow_location), np.asarray(location))
    np.testing.assert_array_equal(np.asarray(narrow_scale), np.asarray(scale * np.float32(0.2)))


def test_advantages_are_brax_two_passes_over_each_sequence():
    """The value targets are GAE's lambda-returns, the TD error discounting the following value
    except after a terminal state and the recursion stopping at either end of an episode; the actor's
    advantage is a TD step from each value through the next step's target, or the following value
    where the episode ended (brax/training/agents/ppo/losses.py:59-98): within float64 rounding of
    both passes run step by step in Python."""
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
    ended = terminated | batch["truncated"]
    expected_targets, expected = np.zeros(shape), np.zeros(shape)
    for row in range(shape[0]):
        running = 0.0
        for t in reversed(range(shape[1])):
            delta = 2 * rewards[row, t] + 0.97 * (not terminated[row, t]) * following[row, t] - values[row, t]
            running = delta + 0.97 * 0.9 * (not ended[row, t]) * running
            expected_targets[row, t] = running + values[row, t]
        for t in range(shape[1]):
            after = following[row, t] if ended[row, t] or t == shape[1] - 1 else expected_targets[row, t + 1]
            expected[row, t] = 2 * rewards[row, t] + 0.97 * (not terminated[row, t]) * after - values[row, t]
    assert_computes_the_oracle(targets, expected_targets, "value targets", roundings=roundings)
    assert_computes_the_oracle(advantages, expected, "advantages", roundings=roundings)


@pytest.mark.parametrize("simulated", [False, True], ids=["env", "simulator"])
def test_ppo_raises_the_greedy_policys_return_on_the_walk(simulated):
    """A few PPO updates of a fresh actor-critic on walks cut at 8 steps, a JAX environment or a simulator
    stepped from the host, raise its greedy policy's mean return above the fresh one's, the collection
    served as two passes of two minibatches."""
    from dew.data.dataset import Loader
    from dew.training import Trainer

    walks = Batched(TimeLimit(Walk(limit=50.0), episode_length=8), 64)
    env = HostWalks(64, 8, limit=50.0, drawn=True) if simulated else walks
    objective = ControlPPOObjective(ActorCritic(1, (16, 16), (32, 32)), 1, discount=0.9, entropy_cost=1e-3)
    rollout = ControlRollout(objective, env, unroll_length=8, epochs=2, minibatches=2)
    trainer = Trainer(objective, optax.adam(3e-3), key=jax.random.key(5), rollout=rollout)
    initial = trainer.place()[0]
    data = Loader(train=lambda partition: itertools.repeat({"step": np.zeros((8,), np.float32)}), val=None,
                  records=8, batch=8)
    before = episode_return(objective, env, initial.variables, jax.random.key(6), 8)
    final = trainer.fit(data, steps=60, log_every=60)
    after = episode_return(objective, env, final.variables, jax.random.key(6), 8)
    assert after > before, (before, after)


@pytest.mark.parametrize("asked", [{"discount": 1.5}, {"lam": -0.1}, {"clip": 0.0}])
def test_a_discount_lambda_or_clip_outside_its_range_is_refused(asked):
    with pytest.raises(ValueError):
        ControlPPOObjective(ActorCritic(1), 1, **asked)


@struct.dataclass
class Settled(Walk):
    """`Walk` restarting at 1 every time, so a reset draws nothing."""

    def reset(self, key):
        x = jnp.ones((), jnp.float32)
        return x, TimeStep(x[None], jnp.zeros(()), jnp.zeros((), bool), jnp.zeros((), bool))


class HostWalks:
    """`Settled` walks cut after `length` steps, simulated on the host in NumPy as a simulator outside JAX
    simulates them: it reads each action through DLPack, answers NumPy arrays, and restarts the episodes
    it is told to. `drawn` walks start, as `Walk`'s do, at integers drawn from [-2, 2]."""

    action_size = 1

    def __init__(self, count: int, length: int, limit: float = 4.0, *, drawn: bool = False):
        self.count, self.length, self.limit, self.drawn = count, length, limit, drawn

    def starts(self) -> np.ndarray:
        if self.drawn:
            return self.rng.integers(-2, 3, self.count).astype(np.float32)
        return np.ones(self.count, np.float32)

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.x, self.steps = self.starts(), np.zeros(self.count, np.int32)
        return self.x[:, None].copy()

    def step(self, action):
        self.x = (self.x + np.round(2 * np.from_dlpack(action)[:, 0])).astype(np.float32)
        self.steps = self.steps + 1
        terminated = np.abs(self.x) > self.limit
        truncated = ~terminated & (self.steps >= self.length)
        return TimeStep(self.x[:, None].copy(), -np.square(self.x), terminated, truncated)

    def restart(self, mask):
        mask = np.asarray(mask)
        self.x = np.where(mask, self.starts(), self.x).astype(np.float32)
        self.steps = np.where(mask, 0, self.steps)
        return self.x[:, None].copy()


class ShardedWalks(HostWalks):
    """`HostWalks` answering its observations as device arrays laid out over a one-device mesh's data axis, as
    a simulator holding its batch on a mesh answers them."""

    def __init__(self, count: int, length: int):
        super().__init__(count, length)
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("data",))
        self.layout = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

    def reset(self, seed: int):
        return jax.device_put(super().reset(seed), self.layout)

    def step(self, action):
        following = super().step(action)
        return TimeStep(jax.device_put(following.observation, self.layout), following.reward,
                        following.terminated, following.truncated)

    def restart(self, mask):
        return jax.device_put(super().restart(mask), self.layout)


def test_a_simulator_answering_arrays_laid_out_over_a_mesh_rolls_out_as_one_answering_numpy():
    """A simulator answering arrays laid out over a mesh's data axis rolls out, its policy's key placed whole
    where its observations are: it records what the same walks answered as NumPy arrays record, the
    observations, actions, rewards and flags bit for bit."""
    from dew.rl.simulator import simulator_rollout, simulator_start

    recorded = []
    for simulator in (HostWalks(5, 3), ShardedWalks(5, 3)):
        carry = simulator_start(simulator, jax.random.key(4))
        recorded.append(simulator_rollout(simulator, pushes, None, carry, 11)[1])
    plain, sharded = recorded
    for name in ("observation", "action"):
        np.testing.assert_array_equal(getattr(sharded, name), getattr(plain, name))
    for name in ("observation", "reward", "terminated", "truncated"):
        np.testing.assert_array_equal(getattr(sharded.following, name), getattr(plain.following, name))


def heads_home(params, observation, key, memory):
    """A policy pushing each point a step toward the origin, its key unread."""
    return -0.5 * jnp.sign(observation), {"seen": observation}, None


@pytest.mark.parametrize("steps", [11, 2 * STACKED + 3])
@pytest.mark.parametrize("policy", [heads_home, pushes], ids=["settled", "drawn"])
def test_a_simulator_outside_jax_rolls_out_as_its_env_does(policy, steps):
    """A simulator stepped from the host (`simulator_rollout`), its arrays crossing by DLPack and its
    ended episodes restarted before the next step, records what `env_rollout` records of the same walks
    as a JAX environment: the observations acted on, the actions and the policy's extras, every step's reward,
    terminal and cut flags and the observation each episode ended at, bit for bit. Where the policy
    draws, each action is drawn from the key the rollout's carry hands it, a step's split of it. A rollout
    longer than a record's stack (`STACKED`) records its steps in order across the stacks it joins."""
    from dew.rl.simulator import simulator_rollout, simulator_start

    env = Batched(TimeLimit(Settled(), episode_length=3), 5)
    simulator = HostWalks(5, 3)
    carry = simulator_start(simulator, jax.random.key(4))
    ended, simulated = simulator_rollout(simulator, policy, None, carry, steps)

    # The same walks in JAX, each action drawn from the key the simulator's rollout drew it from.
    state, observation, key = env.reset(jax.random.key(5))[0], carry.observation, carry.key
    for index in range(steps):
        key, act_key = jax.random.split(key)
        action, extras, _ = policy(None, observation, act_key, None)
        state, following = env.step(state, action)
        np.testing.assert_array_equal(simulated.observation[index], observation)
        np.testing.assert_array_equal(simulated.action[index], action)
        jax.tree.map(np.testing.assert_array_equal, jax.tree.map(lambda leaf, index=index: leaf[index],
                                                                 simulated.extras), extras)
        for name in ("observation", "reward", "terminated", "truncated"):
            np.testing.assert_array_equal(getattr(simulated.following, name)[index], getattr(following, name))
        done = np.asarray(following.done)
        fresh_state, fresh = env.reset(jax.random.key(index))
        state = jax.tree.map(lambda new, old, done=done: jnp.where(done.reshape(-1, *(1,) * (new.ndim - 1)),
                                                                   new, old), fresh_state, state)
        observation = jnp.where(done[:, None], fresh.observation, following.observation)
    np.testing.assert_array_equal(ended.observation, observation)
    assert np.asarray(simulated.following.done).any()


def test_a_driven_simulator_records_what_a_driven_env_records():
    """A simulator driven for a world-model learner (`DrivenSimulator`), each ended episode answered as a
    step of its own and restarted a step later, records what `Driven` records of the same walks as a JAX
    environment: every step's observation, reward, first, last and terminal flags and the actions taken,
    bit for bit."""
    from dew.rl.env import Driven
    from dew.rl.simulator import DrivenSimulator, simulator_rollout, simulator_start

    def home(params, observed, key, memory):
        return heads_home(params, observed["observation"], key, memory)

    driven = Driven(Batched(TimeLimit(Settled(), episode_length=3), 5))
    _, expected = env_rollout(driven, home, None, start(driven, jax.random.key(4)), 11, reset=False)
    simulator = DrivenSimulator(HostWalks(5, 3))
    _, simulated = simulator_rollout(simulator, home, None, simulator_start(simulator, jax.random.key(4)), 11,
                                     reset=False)
    jax.tree.map(np.testing.assert_array_equal, (simulated.observation, simulated.action),
                 (expected.observation, expected.action))
    assert np.asarray(expected.observation["first"])[1:].any()
