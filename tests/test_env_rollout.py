"""Environments as pure functions (`dew.rl.env`), their replay (`dew.rl.replay`) and PPO over them
(`dew.objectives.rl.control`): each held to a step-by-step loop, an exact index bookkeeping or a
float64 transcription of the arithmetic, apart from Dew's code."""

import contextlib
import copy
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from recording import bits
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.objectives.base import Shown
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
    from dew.config.env import idle
    from dew.training import Trainer

    walks = Batched(TimeLimit(Walk(limit=50.0), episode_length=8), 64)
    env = HostWalks(64, 8, limit=50.0, drawn=True) if simulated else walks
    objective = ControlPPOObjective(ActorCritic(1, (16, 16), (32, 32)), 1, discount=0.9, entropy_cost=1e-3)
    rollout = ControlRollout(objective, env, unroll_length=8, epochs=2, minibatches=2)
    trainer = Trainer(objective, optax.adam(3e-3), key=jax.random.key(5), rollout=rollout)
    initial = trainer.place()[0]
    data = idle()
    before = episode_return(objective, env, initial.variables, jax.random.key(6), 8)
    final = trainer.fit(data, steps=60, log_every=60)
    after = episode_return(objective, env, final.variables, jax.random.key(6), 8)
    assert after > before, (before, after)


def test_one_unbatched_environments_episode_return_is_the_reward_its_first_episode_earns():
    """`episode_return` of one environment, unbatched, is the reward its greedy policy's first episode
    earns, summed as the episode is flown by hand a step at a time; walks' rewards are integers, so
    every order sums them exactly."""
    objective = ControlPPOObjective(ActorCritic(1, (16, 16), (32, 32)), 1, discount=0.9)
    variables = objective.init(jax.random.key(0))
    env, key = TimeLimit(Walk(limit=4.0), episode_length=8), jax.random.key(6)
    carry = start(env, key)
    state, observation, earned = carry.state, carry.observation, 0.0
    for _ in range(12):
        state, following = env.step(state, objective.greedy(variables, observation, key, None)[0])
        earned += float(following.reward)
        if bool(following.terminated | following.truncated):
            break
        observation = following.observation
    assert episode_return(objective, env, variables, key, 12) == earned


def test_a_simulators_episodes_are_the_rewards_its_first_episodes_earn():
    """`episodes` of walks a host simulator steps (`HostWalks`, drawn starts) scores each walk's first
    episode as flown by hand a step at a time from the same seed: its undiscounted return, its steps and
    its position where it ended, averaged over the walks; walks' rewards and positions are integers, so
    every order sums them exactly."""
    from dew.rl.evaluation import episodes
    from dew.rl.simulator import simulator_start

    class Seen:
        """`HostWalks` that show their position as a view, `x`."""

        action_size = 1

        def __init__(self, walks: HostWalks):
            self.walks, self.count = walks, walks.count

        def reset(self, seed: int):
            return {"x": self.walks.reset(seed)}

        def step(self, action):
            following = self.walks.step(action)
            return TimeStep({"x": following.observation}, following.reward, following.terminated,
                            following.truncated)

        def restart(self, mask):
            return {"x": self.walks.restart(mask)}

    objective = ControlPPOObjective(ActorCritic(1, (16, 16), (32, 32)), 1, discount=0.9, reads=("x",))
    variables, key = objective.init(jax.random.key(0)), jax.random.key(6)
    walks = Seen(HostWalks(16, 8, limit=3.0, drawn=True))
    scored = episodes(objective.greedy, walks, 12, reached=("x",))(variables, key)
    observation = simulator_start(walks, key).observation
    earned, lasted, at = np.zeros(16), np.zeros(16), np.zeros(16)
    going = np.ones(16, bool)
    for _ in range(12):
        action = objective.greedy(variables, observation, key, None)[0]
        following = walks.step(action)
        reward, x = np.asarray(following.reward), np.asarray(following.observation["x"])[:, 0]
        earned, lasted = earned + np.where(going, reward, 0), lasted + going
        at = np.where(going, x, at)
        going &= ~(np.asarray(following.terminated) | np.asarray(following.truncated))
        observation = {"x": jnp.asarray(following.observation["x"])}
    assert scored == {"return": earned.mean(), "steps": lasted.mean(), "x": at.mean()}


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


class Lent:
    """A buffer a simulator lends through DLPack, another framework's array as it hands it over."""

    def __init__(self, held: np.ndarray):
        self.held = held

    def __dlpack__(self, **options):
        return self.held.__dlpack__(**options)

    def __dlpack_device__(self):
        return self.held.__dlpack_device__()


def aligned(shape: tuple[int, ...]) -> np.ndarray:
    """A float32 buffer of `shape` on a 64-byte boundary, which XLA:CPU reads in place through DLPack
    (an unaligned one it copies)."""
    size = int(np.prod(shape))
    held = np.zeros(size + 16, np.float32)
    start = (-held.ctypes.data % 64) // held.itemsize
    return held[start:start + size].reshape(shape)


class ReusingWalks(HostWalks):
    """`HostWalks` answering every observation and reward from one buffer each, lent through DLPack and
    written again in place every step, as a GPU simulator's fixed output buffers are."""

    def __init__(self, count: int, length: int):
        super().__init__(count, length)
        self.observed, self.rewarded = aligned((count, 1)), aligned((count,))

    def _lent(self, observation) -> Lent:
        self.observed[...] = observation
        return Lent(self.observed)

    def reset(self, seed: int):
        return self._lent(super().reset(seed))

    def step(self, action):
        following = super().step(action)
        self.rewarded[...] = following.reward
        return TimeStep(self._lent(following.observation), Lent(self.rewarded), following.terminated,
                        following.truncated)

    def restart(self, mask):
        return self._lent(super().restart(mask))


def test_a_simulator_writing_each_step_into_one_buffer_records_every_step_it_answered():
    """A simulator that lends its observations and rewards from the same buffers every step, writing each
    step over the last (`ReusingWalks`), records what the same walks answered in fresh arrays record, every
    step's observation and reward bit for bit: a rollout keeps an owned copy of each step it takes in
    (`dew.rl.simulator.taken`), not a view a later step overwrites (row 4641)."""
    from dew.rl.simulator import simulator_rollout, simulator_start

    recorded = []
    for simulator in (HostWalks(5, 3), ReusingWalks(5, 3)):
        carry = simulator_start(simulator, jax.random.key(4))
        recorded.append(simulator_rollout(simulator, pushes, None, carry, 11)[1])
    fresh, reused = recorded
    np.testing.assert_array_equal(reused.observation, fresh.observation)
    for name in ("observation", "reward"):
        np.testing.assert_array_equal(getattr(reused.following, name), getattr(fresh.following, name))
    assert np.unique(np.asarray(fresh.observation), axis=0).shape[0] > 1


@contextlib.contextmanager
def kept_as_left(walks):
    """A host simulator's walks stepped in a cooldown branch, then put back as the run left them: their
    positions, steps and draws (a simulator's `branch`, `dew.rl.simulator.Branching`)."""
    kept = walks.x.copy(), walks.steps.copy(), copy.deepcopy(walks.rng)
    try:
        yield
    finally:
        walks.x, walks.steps, walks.rng = kept


class KeptWalks(HostWalks):
    """`HostWalks` that a cooldown branch steps and then leaves as the run left them (`kept_as_left`)."""

    def branch(self):
        return kept_as_left(self)


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


def test_a_simulators_policy_step_compiles_under_the_options_the_run_holds_open():
    """A simulator's policy step compiles through Dew's compile point, so under an option the run holds
    open (`compiled_with`, the accuracy gate's seam) that XLA does not know it is refused, as any program
    of Dew's is (row 3408: the step was a bare outer program, which took no option)."""
    from dew.nn.kernels.options import compiled_with
    from dew.rl.simulator import simulator_rollout, simulator_start

    simulator = ShardedWalks(5, 3)
    carry = simulator_start(simulator, jax.random.key(4))
    with compiled_with({"xla_dew_no_such_option": True}), pytest.raises(jax.errors.JaxRuntimeError):
        simulator_rollout(simulator, pushes, {}, carry, 3)


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


@pytest.mark.parametrize("where", ["persistent", "local"])
def test_a_control_run_resumed_from_its_checkpoint_ends_as_the_run_that_never_stopped(tmp_path, where):
    """A PPO run on walks, checkpointed every update to its run's directory or to its local one, stopped at
    its third update and resumed from that checkpoint by a fresh rollout over fresh environments: the
    checkpoint holds the rollout's walks mid episode, the collection it serves and its minibatches' order
    (`ControlRollout.saved`), so the resumed run's last state is the uninterrupted run's, every leaf bit
    for bit, the key's as its raw bits; a run whose checkpoints were written without the rollout's state
    resumes to another state, its walks begun again."""
    from dew.checkpoints import Checkpoints, Keep
    from dew.config.env import idle
    from dew.training import Trainer

    def walks():
        return Batched(TimeLimit(Walk(limit=50.0), episode_length=8), 16)

    objective = ControlPPOObjective(ActorCritic(1, (16, 16), (32, 32)), 1, discount=0.9, entropy_cost=1e-3)
    data = idle()

    def trained(directory, steps, rollout):
        local = {"local_directory": str(directory / "local"), "local_every": 1} if where == "local" else {}
        trainer = Trainer(objective, optax.adam(3e-3), key=jax.random.key(5), rollout=rollout,
                          checkpoints=Checkpoints(str(directory / "run"), keep=Keep(latest=8), **local))
        # A local run keeps its local checkpoint alone, so the resume reads it.
        every = {"checkpoint_every": 1} if where == "persistent" else {"checkpoint_final": False}
        return trainer.fit(data, steps=steps, log_every=1, **every)

    def resumed(directory, made):
        trained(directory, 3, made())
        return trained(directory, 6, made())

    def made():
        return ControlRollout(objective, walks(), unroll_length=8, epochs=2, minibatches=2)

    whole = trained(tmp_path / "whole", 6, made())
    again = resumed(tmp_path / "stopped", made)
    for one, other in zip(bits(whole), bits(again), strict=True):
        np.testing.assert_array_equal(one, other)

    class Forgetting(ControlRollout):
        """The rollout saving nothing of what it carries, as one did before checkpoints held it."""

        def saved(self):
            return None

    def forgetting():
        return Forgetting(objective, walks(), unroll_length=8, epochs=2, minibatches=2)

    afresh = resumed(tmp_path / "forgot", forgetting)
    assert not all(np.array_equal(one, other)
                   for one, other in zip(bits(whole), bits(afresh), strict=True))


class Counted(ControlRollout):
    """`ControlRollout`, keeping each collection it makes and each batch it serves."""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "made", [])
        object.__setattr__(self, "served", [])

    def __call__(self, state, batch, key):
        before = self._collected
        out = super().__call__(state, batch, key)
        if self._collected is not before:
            self.made.append(self._collected)
        self.served.append(out)
        return out


def test_a_control_run_accumulating_two_minibatches_an_update_serves_each_collection_once_a_pass():
    """A PPO run whose update pools two microbatches (`accumulation=2`) serves its rollout's collection in
    one pass of its two minibatches: two calls an update, each the next minibatch, a collection each update,
    every sequence of each collection served once. The cursor is the rollout's own, apart from the updates
    the optimizer commits, where reading it off them collected afresh on every call of an update and
    served the first minibatch of each (row 3932)."""
    from dew.config.env import idle
    from dew.training import Trainer

    objective = ControlPPOObjective(ActorCritic(1, (16, 16), (32, 32)), 1, discount=0.9)
    rollout = Counted(objective, Batched(TimeLimit(Walk(limit=50.0), episode_length=8), 16), unroll_length=8,
                      epochs=1, minibatches=2)
    Trainer(objective, optax.adam(3e-3), key=jax.random.key(5), rollout=rollout, accumulation=2).fit(
        idle(), steps=4, log_every=4)
    assert len(rollout.made) == 2 and len(rollout.served) == 4
    for made, pair in zip(rollout.made, (rollout.served[:2], rollout.served[2:]), strict=True):
        rewards = np.sort(np.concatenate([np.asarray(batch["reward"]) for batch in pair]), axis=0)
        np.testing.assert_array_equal(rewards, np.sort(np.asarray(made["reward"]), axis=0))


def test_a_control_runs_cooldowns_leave_its_rollout_as_the_run_left_it(tmp_path):
    """A PPO run on walks with a cooldown at its fourth update ends, every leaf bit for bit, where the run
    with no cooldown does: the branch's updates run on the rollout's walks and collection and leave them as
    the run left them (`ControlRollout.branch`). Preempted as its branch begins, the run saves its step's
    state with the rollout's as the run left it, and its resume runs the branch again and ends there too."""
    from recording import preempting

    from dew.checkpoints import Checkpoints, Keep
    from dew.config.env import idle
    from dew.rl.evaluation import episodes
    from dew.training import Episodes, EvalSuite, Preempted, Trainer
    from dew.training.cooldown import Cooldowns

    objective = ControlPPOObjective(ActorCritic(1, (16, 16), (32, 32)), 1, discount=0.9, entropy_cost=1e-3)
    data = idle()
    # A cooldown branch is scored by the run's validation: here the walks the greedy policy flies.
    flights = {"flight": EvalSuite(Episodes(episodes(objective.greedy, Batched(TimeLimit(
        Walk(limit=50.0), episode_length=8), 8), 8), {"return": Shown(better="higher"), "steps": Shown()}),
        every=8)}

    def trained(directory=None, **options):
        walks = Batched(TimeLimit(Walk(limit=50.0), episode_length=8), 16)
        checkpoints = None if directory is None else Checkpoints(str(directory), keep=Keep(latest=8))
        # A cooldown scales the rate the optimizer's state holds.
        trainer = Trainer(objective, optax.inject_hyperparams(optax.adam)(learning_rate=3e-3),
                          key=jax.random.key(5), checkpoints=checkpoints,
                          rollout=ControlRollout(objective, walks, unroll_length=8, epochs=2, minibatches=2))
        return trainer.fit(data, steps=8, log_every=1, validation=flights, **options)

    plain = bits(trained())
    cooldowns = Cooldowns(at=(4,), length=2)
    for one, other in zip(plain, bits(trained(cooldowns=cooldowns)), strict=True):
        np.testing.assert_array_equal(one, other)
    with pytest.raises(Preempted):
        trained(tmp_path / "run", cooldowns=cooldowns, checkpoint_every=1, observe=preempting(3))
    assert Checkpoints(str(tmp_path / "run")).latest == 4
    again = trained(tmp_path / "run", cooldowns=cooldowns, checkpoint_every=1)
    for one, other in zip(plain, bits(again), strict=True):
        np.testing.assert_array_equal(one, other)


def test_a_control_runs_cooldown_over_a_simulator_leaves_it_as_the_run_left_it():
    """A PPO run on walks a simulator steps, with a cooldown at its fourth update whose branch collects
    steps of its own, ends, every leaf bit for bit, where the run with no cooldown does: the rollout keeps
    its state and the simulator its own walks (their `branch()`, `dew.rl.simulator.branched`). A simulator
    with no `branch()` is refused the cooldown."""
    from dew.config.env import idle
    from dew.rl.evaluation import episodes
    from dew.training import Episodes, EvalSuite, Trainer
    from dew.training.cooldown import Cooldowns

    objective = ControlPPOObjective(ActorCritic(1, (16, 16), (32, 32)), 1, discount=0.9, entropy_cost=1e-3)
    data = idle()
    flights = {"flight": EvalSuite(Episodes(episodes(objective.greedy, Batched(TimeLimit(
        Walk(limit=50.0), episode_length=8), 8), 8), {"return": Shown(better="higher"), "steps": Shown()}),
        every=8)}

    def trained(walks, **options):
        # A cooldown scales the rate the optimizer's state holds.
        trainer = Trainer(objective, optax.inject_hyperparams(optax.adam)(learning_rate=3e-3),
                          key=jax.random.key(5),
                          rollout=ControlRollout(objective, walks, unroll_length=8, epochs=2, minibatches=2))
        return trainer.fit(data, steps=8, log_every=1, validation=flights, **options)

    plain = bits(trained(KeptWalks(16, 8, limit=50.0, drawn=True)))
    cooldowns = Cooldowns(at=(4,), length=2)
    cooled = trained(KeptWalks(16, 8, limit=50.0, drawn=True), cooldowns=cooldowns)
    for one, other in zip(plain, bits(cooled), strict=True):
        np.testing.assert_array_equal(one, other)
    with pytest.raises(TypeError):
        trained(HostWalks(16, 8, limit=50.0, drawn=True), cooldowns=cooldowns)


@struct.dataclass
class Seen(Walk):
    """`Walk` observed through named views: its `position`, its `distance` from the origin, and the `push`
    it was last given (none at its start), which a policy's weights move continuously."""

    def reset(self, key):
        x, _ = super().reset(key)
        unended = jnp.zeros((), bool)
        return x, TimeStep(self.seen(x, jnp.zeros(1)), jnp.zeros(()), unended, unended)

    def step(self, x, action):
        x, stepped = super().step(x, action)
        return x, TimeStep(self.seen(x, action), stepped.reward, stepped.terminated, stepped.truncated)

    @staticmethod
    def seen(x, push):
        return {"position": x[None], "distance": jnp.abs(x)[None], "push": push}


def flown_by_hand(objective, variables, key, count: int, steps: int) -> dict[str, float]:
    """The greedy policy flown over `count` fresh `Seen` walks for `steps` steps from `key`, scored by
    hand: each walk's first episode's undiscounted return, its steps, and its distance and last push at its
    last step, each a mean over the walks."""
    env = Batched(TimeLimit(Seen(limit=50.0), episode_length=8), count)
    taken = env_rollout(env, objective.greedy, variables, start(env, key), steps, reset=False)[1]
    done = np.asarray(taken.following.done)
    inside = (np.cumsum(done, 0) - done) == 0
    last = inside.sum(0) - 1

    def at_last(name):
        value = np.asarray(taken.following.observation[name]).reshape(steps, count)
        return float(value[last, np.arange(count)].mean())

    return {"return": float(np.where(inside, np.asarray(taken.following.reward), 0).sum(0).mean()),
            "steps": float(inside.sum(0).mean()), "distance": at_last("distance"), "push": at_last("push")}


def evaluation_key(run_key, step: int):
    """The key a split's episodes fly from at `step`: the run's key folded with 'EVAL', the step and
    'EPIS', as `dew.training.Episodes` says."""
    folded = jax.random.fold_in(jax.random.fold_in(run_key, 0x4556414C), step)
    return jax.random.fold_in(folded, 0x45504953)


SHOWN = {"return": Shown(better="higher"), "steps": Shown(better="higher"),
         "distance": Shown(better="lower"), "push": Shown()}
"""How a walk's episode scores show: a longer and better-paid episode, nearer the origin, is better."""


def test_a_split_scored_by_its_episodes_reports_its_averaged_weights_first_episodes(tmp_path):
    """A PPO run on walks that keeps averaged weights (an EMA of decay 0.5), validated by its episodes
    (`EvalSuite(Episodes(dew.rl.evaluation.episodes(...)))`), reports at its end `flight/return`,
    `flight/steps` and `flight/distance`: what its averaged weights earn flown, from the key the run's key
    gives at its last step, over the walks' first episodes scored by hand; its live weights' flight is
    another. A warming flight before the first step reports nothing."""
    from dew.config.env import idle
    from dew.objectives.base import EMASpec
    from dew.rl.evaluation import episodes
    from dew.training import Episodes, EvalSuite, Trainer
    from dew.training.tracker import LocalTracker

    objective = ControlPPOObjective(ActorCritic(1, (16, 16), (32, 32)), 1, discount=0.9, entropy_cost=1e-3,
                                    reads=("position",))
    objective.ema = EMASpec(lambda step: 0.5)
    walks = Batched(TimeLimit(Seen(limit=50.0), episode_length=8), 32)
    flown = episodes(objective.greedy, walks, 12, reached=("distance", "push"))
    suite = EvalSuite(Episodes(flown, SHOWN), every=4)
    data = idle()
    tracker = LocalTracker(str(tmp_path / "tracked"))
    rollout = ControlRollout(objective, Batched(TimeLimit(Seen(limit=50.0), episode_length=8), 16),
                             unroll_length=8, epochs=2, minibatches=2)
    state = Trainer(objective, optax.adam(3e-3), key=jax.random.key(5), rollout=rollout,
                    tracker=tracker).fit(data, steps=4, log_every=2, validation={"flight": suite})
    logged = [(step, scalars) for step, scalars in tracker.scalars() if "flight/return" in scalars]
    tracker.close()
    assert [step for step, _ in logged] == [4]
    key = evaluation_key(state.key, 4)
    expected = flown_by_hand(objective, state.averaged, key, 32, 12)
    assert {name: logged[0][1][f"flight/{name}"] for name in expected} == expected
    assert flown_by_hand(objective, state.variables, key, 32, 12) != expected


@pytest.mark.parametrize("selected", ["by name", "by metric", "by its only metric"])
def test_a_run_ranks_its_checkpoints_by_a_score_its_episodes_report(tmp_path, selected):
    """A PPO run on walks validated by its episodes every other update keeps, as its best checkpoint (better
    higher as its `Episodes` declares), a step whose logged return is the highest any evaluation logged:
    selected by the score's name (`Best("flight/return")`), by its metric (`Best(split.metrics[0])`, the
    very one the run declares), or by that metric where the suite asks for it alone, which logs it alone
    though the episodes report every score they declare."""
    from dew.checkpoints import Checkpoints, Keep
    from dew.config.env import idle
    from dew.rl.evaluation import episodes
    from dew.training import Best, Episodes, EvalSuite, Trainer
    from dew.training.tracker import LocalTracker

    objective = ControlPPOObjective(ActorCritic(1, (16, 16), (32, 32)), 1, discount=0.9, entropy_cost=1e-3,
                                    reads=("position",))
    walks = Batched(TimeLimit(Seen(limit=50.0), episode_length=8), 32)
    flown = episodes(objective.greedy, walks, 12, reached=("distance", "push"))
    split = Episodes(flown, SHOWN)
    asked = split.metrics[:1] if selected == "by its only metric" else ()
    suite = EvalSuite(split, metrics=asked, every=2)
    best = Best("flight/return") if selected == "by name" else Best(split.metrics[0])
    data = idle()
    tracker = LocalTracker(str(tmp_path / "tracked"))
    checkpoints = Checkpoints(str(tmp_path / "run"), keep=Keep(latest=1))
    rollout = ControlRollout(objective, Batched(TimeLimit(Seen(limit=50.0), episode_length=8), 16),
                             unroll_length=8, epochs=2, minibatches=2)
    Trainer(objective, optax.adam(3e-2), key=jax.random.key(5), rollout=rollout, tracker=tracker,
            checkpoints=checkpoints).fit(data, steps=8, log_every=2, validation={"flight": suite},
                                         best=best, checkpoint_every=2)
    returns = {step: scalars["flight/return"] for step, scalars in tracker.scalars()
               if "flight/return" in scalars}
    flown_names = {name for _, scalars in tracker.scalars() for name in scalars if name.startswith("flight/")}
    tracker.close()
    checkpoints.wait()
    assert sorted(returns) == [2, 4, 6, 8]
    assert returns[checkpoints.best] == max(returns.values())
    assert flown_names == ({"flight/return"} if asked else {f"flight/{name}" for name in SHOWN})


def test_a_pool_is_refused_a_rollout_that_checkpoints_its_state_as_its_run_starts(tmp_path):
    """A trainer over a rollout that checkpoints the state it carries, made from an integer seed before
    its process joins a pool (made without opening JAX's backend, which a pool forms only before), is
    refused as its fit starts, before it restores or steps, on every process of a pool of two, and
    trains on a pool of one; a pool of two placing the one's checkpointed run is refused it too, before
    its rollout restores: each process holds its own environments, a checkpoint the pool's tree."""
    from pathlib import Path

    from process_support import run_pool

    script = Path(__file__).with_name("rollout_state_worker.py")
    pooled = run_pool("fitted", tmp_path / "two", 2, script=script, devices=1, timeout=300)
    alone = run_pool("fitted", tmp_path / "one", 1, script=script, devices=1, timeout=300)
    assert pooled == [{"processes": 2, "refused": "ValueError"}] * 2
    assert alone == [{"processes": 1, "refused": None}]
    # The one process's run, its rollout's state checkpointed, placed by a pool of two: refused the
    # rollout that would restore that state, where a trainer whose rollout keeps none places it.
    placed = run_pool("placed", tmp_path / "one", 2, script=script, devices=1, timeout=300)
    assert placed == [{"processes": 2, "refused": {"stateful": "ValueError", "stateless": None}}] * 2


def test_a_pools_split_scored_by_its_episodes_agrees_and_fails_together(tmp_path):
    """A pool of two evaluates a split scored by its episodes as one: every process reports process 0's
    scores and one event key; a scorer that fails on one process alone, or reports there fewer scores than
    its split declares, fails the evaluation on both, that process raising its own error and its peer a
    `PeerFailure`, rather than leaving the peer waiting."""
    from pathlib import Path

    from process_support import run_pool

    reports = run_pool("flown", tmp_path, 2, script=Path(__file__).with_name("rollout_state_worker.py"),
                       devices=1, timeout=300)
    assert reports[0]["scores"] == reports[1]["scores"] and set(reports[0]["scores"]) == {"flight/return",
                                                                                          "flight/steps"}
    assert reports[0]["event"] == reports[1]["event"]
    peer, failing = (report["failed"] for report in reports)
    assert peer == {"raising": "PeerFailure", "incomplete": "PeerFailure"}
    assert failing == {"raising": "RuntimeError", "incomplete": "ValueError"}

