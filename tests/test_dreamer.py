"""DreamerV3 (`dew.objectives.rl.dreamer`, `dew.rl.distributions`) against its code's arithmetic
(danijar/dreamerv3 at e01491fad6434b2245a3b8ca201dd7faedcc458c) transcribed here in NumPy float64,
an episode driver's records against the steps it took, and an agent that learns its world."""

import dataclasses
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from recording import bits
from reference_error import assert_computes_the_oracle, chain_roundings
from test_env_rollout import kept_as_left

from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective, LatentEnv, WorldModel, lambda_return
from dew.objectives.rl.tssm import TSSM
from dew.rl.distributions import symexp, symlog, twohot_bins, twohot_loss, twohot_mean, unimix_log_probs
from dew.rl.env import Batched, Carry, Driven, TimeLimit, TimeStep, env_rollout, start


def test_two_hot_predicts_zero_at_uniform_odds_and_trains_toward_its_encoding():
    """Uniform logits predict exactly zero over the symmetric bins; the loss is the cross-entropy of
    the target's two-hot encoding, its weight split between the neighbouring bins in inverse
    proportion to their distances (outs.py:311-323), within float64 rounding of that encoding built
    bin by bin; and symexp inverts symlog."""
    bins = np.asarray(twohot_bins(255), np.float64)
    assert float(twohot_mean(jnp.zeros(255), jnp.asarray(bins, jnp.float32))) == 0.0
    rng = np.random.default_rng(0)
    targets = np.concatenate([symexp(rng.uniform(-8, 8, 30)), [bins[0] * 2, bins[-1] * 2, bins[130]]])
    logits = rng.normal(size=(len(targets), 255))
    with jax.enable_x64(new_val=True):
        found = np.asarray(twohot_loss(logits, targets, jnp.asarray(bins)))
        roundings = chain_roundings(jax.make_jaxpr(twohot_loss)(logits, targets, jnp.asarray(bins)))
        inverted = np.asarray(symexp(symlog(jnp.asarray(targets))))
    log_softmax = logits - np.log(np.sum(np.exp(logits - logits.max(-1, keepdims=True)), -1,
                                         keepdims=True)) - logits.max(-1, keepdims=True)
    expected = []
    for target, row in zip(targets, log_softmax, strict=True):
        below = min(max(int(np.sum(bins <= target)) - 1, 0), 254)
        above = min(max(255 - int(np.sum(bins > target)), 0), 254)
        if below == above:
            expected.append(-row[below])
            continue
        span = bins[above] - bins[below]
        share = (bins[above] - target) / span
        expected.append(-(share * row[below] + (1 - share) * row[above]))
    assert_computes_the_oracle(found, np.asarray(expected), "the two-hot loss", roundings=roundings)
    assert_computes_the_oracle(inverted, targets, "symexp of symlog", roundings=8)


@pytest.mark.parametrize("unimix", [0.0, 0.01])
def test_a_categoricals_log_probabilities_stay_finite_where_a_class_is_far_below(unimix):
    """A latent's log-probabilities (`unimix_log_probs`) of logits a thousand below their row's largest,
    with and without the uniform mixture, are finite in float32, so the world model's KL over them stays
    finite, and in float64 the mixture's written from the log-softmax, where a probability's log is -inf
    once its softmax underflows (below about -745 in float64)."""
    logits = np.asarray([[0.0, -1000.0, -3.0, 2.5], [-40.0, 0.0, -1000.0, 1.0]], np.float32)
    assert np.isfinite(np.asarray(unimix_log_probs(jnp.asarray(logits), unimix))).all()
    wide = logits.astype(np.float64)
    with jax.enable_x64(new_val=True):
        found = np.asarray(unimix_log_probs(jnp.asarray(wide), unimix))
    log_softmax = wide - wide.max(-1, keepdims=True)
    log_softmax = log_softmax - np.log(np.exp(log_softmax).sum(-1, keepdims=True))
    expected = log_softmax if not unimix else np.log((1 - unimix) * np.exp(log_softmax) + unimix / 4)
    assert_computes_the_oracle(found, expected, "the latent's log-probabilities", roundings=16)


def test_lambda_returns_are_the_codes_recursion():
    """The lambda-returns over sequences that end and terminate within them, within float64 rounding
    of the code's Python loop (agent.py:482-490) run here in NumPy."""
    rng = np.random.default_rng(1)
    shape = (5, 9)
    last, term = (rng.uniform(size=shape) < 0.2).astype(float), (rng.uniform(size=shape) < 0.2).astype(float)
    reward, value, boot = (rng.normal(size=shape) for _ in range(3))
    with jax.enable_x64(new_val=True):
        found = np.asarray(lambda_return(last, term, reward, value, boot, 0.997, 0.95))
        roundings = chain_roundings(jax.make_jaxpr(
            lambda *a: lambda_return(*a, 0.997, 0.95))(last, term, reward, value, boot))
    rets = [boot[:, -1]]
    live = (1 - term)[:, 1:] * 0.997
    cont = (1 - last)[:, 1:] * 0.95
    interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    expected = np.stack(list(reversed(rets))[:-1], 1)
    assert_computes_the_oracle(found, expected, "lambda-returns", roundings=roundings)


@struct.dataclass
class Target:
    """A point on a line the action moves by itself, paid minus its squared distance from 1, its
    episode terminated past 3 from the origin."""

    action_size: int = struct.field(pytree_node=False, default=1)

    def reset(self, key):
        x = jax.random.uniform(key, (), minval=-1, maxval=1)
        return x, TimeStep(x[None], jnp.zeros(()), jnp.zeros((), bool), jnp.zeros((), bool))

    def step(self, x, action):
        x = x + 0.5 * action[0]
        return x, TimeStep(x[None], -jnp.square(x - 1), jnp.abs(x) > 3, jnp.zeros((), bool))


@struct.dataclass
class SharedTarget:
    """`Target`, its state holding its point twice, one array in two fields (as a Playground state holds
    one zero in several)."""

    action_size: int = struct.field(pytree_node=False, default=1)

    def reset(self, key):
        x, step = Target().reset(key)
        return {"x": x, "again": x}, step

    def step(self, state, action):
        x, step = Target().step(state["x"], action)
        return {"x": x, "again": x}, step


def test_a_rollout_advances_an_environment_whose_state_shares_an_array():
    """A Dreamer rollout donates its live state to the steps that advance it, each buffer once, so it runs
    over an environment whose state holds one array in two fields: three updates at a train ratio of 32 over
    windows of 16 in batches of 8 take the prefill and exactly the steps they owe."""
    from dew.objectives.rl.config import idle
    from dew.training import Trainer

    env = Batched(TimeLimit(SharedTarget(), episode_length=20), 8)
    objective = DreamerV3Objective(tiny())
    rollout = DreamerRollout(objective, env, capacity=400, batch=8, length=16, train_ratio=32.0, prefill=24)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(7), rollout=rollout)
    data = idle()
    trainer.fit(data, steps=3, log_every=3)
    owed = -(-3 * 8 * 16 // 32)
    assert rollout.environment_steps == 8 * 24 + 8 * -(-owed // 8)


def test_a_driver_records_an_episodes_last_observation_then_restarts():
    """Driven answers each step whole: an episode's last observation is a step of its own, `last`,
    with the reward that came with it, and the step after it, whatever the action, starts the next
    episode, `first`, with no reward."""
    env = Driven(Batched(TimeLimit(Target(), episode_length=3), 4))

    def push(params, observed, key, memory):
        return jnp.ones((4, 1)), {}, None

    _, recorded = env_rollout(env, push, None, start(env, jax.random.key(2)), 9, reset=False)
    first, last = np.asarray(recorded.observation["first"]), np.asarray(recorded.observation["last"])
    np.testing.assert_array_equal(first, np.broadcast_to((np.arange(9) % 4 == 0)[:, None], (9, 4)))
    np.testing.assert_array_equal(last, np.broadcast_to((np.arange(9) % 4 == 3)[:, None], (9, 4)))
    rewards = np.asarray(recorded.observation["reward"])
    assert (rewards[first] == 0).all() and (rewards[~first] != 0).all()


@struct.dataclass
class Seen:
    """`Target` seen as a 16-pixel-square image, a bright column where the point is on [-3, 3], beside the
    point's position, privileged (`state`)."""

    action_size: int = struct.field(pytree_node=False, default=1)

    @staticmethod
    def image(x):
        column = jnp.clip(jnp.round((x + 3) / 6 * 15), 0, 15).astype(jnp.int32)
        row = (jnp.arange(16) == column).astype(jnp.uint8) * 255
        return jnp.broadcast_to(row[None, :, None], (16, 16, 1))

    def reset(self, key):
        x, step = Target().reset(key)
        return x, step.replace(observation={"image": self.image(x), "state": step.observation})

    def step(self, x, action):
        x, step = Target().step(x, action)
        return x, step.replace(observation={"image": self.image(x), "state": step.observation})


def tiny(observation_size=1, action_size=1, **options):
    return WorldModel(observation_size, action_size, deter=16, hidden=8, stoch=4, classes=4, blocks=2,
                      units=8, **options)


def transformer():
    return TSSM(action_size=1, stoch=4, classes=4, width=16, layers=1, heads=2, context=16)


class HostTarget:
    """`Target`'s walks, eight of them cut after 20 steps, simulated on the host in NumPy as a simulator
    outside JAX simulates them, restarting the episodes it is told to (`dew.rl.simulator.Simulator`)."""

    action_size, count = 1, 8

    def reset(self, seed: int) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self.x, self.steps = self.rng.uniform(-1, 1, self.count).astype(np.float32), np.zeros(self.count, int)
        return self.x[:, None].copy()

    def step(self, action) -> TimeStep:
        self.x = (self.x + 0.5 * np.from_dlpack(action)[:, 0]).astype(np.float32)
        self.steps += 1
        terminated = np.abs(self.x) > 3
        truncated = ~terminated & (self.steps >= 20)
        return TimeStep(self.x[:, None].copy(), -np.square(self.x - 1), terminated, truncated)

    def restart(self, mask) -> np.ndarray:
        mask = np.asarray(mask)
        self.x = np.where(mask, self.rng.uniform(-1, 1, self.count), self.x).astype(np.float32)
        self.steps = np.where(mask, 0, self.steps)
        return self.x[:, None].copy()

    def branch(self):
        """A cooldown branch's steps, after which the targets stand as the run left them (`kept_as_left`)."""
        return kept_as_left(self)


def walks(walk):
    """Eight of `walk`'s episodes cut after 20 steps, batched."""
    return lambda: Batched(TimeLimit(walk(), episode_length=20), 8)


AGENTS = {"gru": (walks(Target), lambda: tiny()), "gru-simulator": (HostTarget, lambda: tiny()),
          "transformer": (walks(Target), lambda: tiny(dynamics=transformer())),
          "transformer-pixels": (walks(Seen), lambda: tiny(0, image=(16, 16, 1), conv_depth=2,
                                                           reads="image", auxiliary=(("state", 1, 8),),
                                                           dynamics=transformer()))}
"""Each agent's environments and world model: the block GRU's and TransDreamer's dynamics over the walk's
position, the GRU's again over the walks simulated on the host, and TransDreamer's over the walk seen as
pixels alone, its position predicted from the latent by an auxiliary head."""


@pytest.mark.parametrize("agent", ["gru", "gru-simulator"])
def test_a_rollout_writes_its_replay_in_place(agent):
    """A Dreamer rollout's steps are written to its replay in place: the replay it held when a collection
    began, over an environment on the device or a simulator on the host, is donated to the program that
    writes them and deleted once it ran, so the replay is never held twice."""
    from dew.training import Trainer

    environments, model = AGENTS[agent]
    objective = DreamerV3Objective(model())
    rollout = DreamerRollout(objective, environments(), capacity=400, batch=8, length=16, train_ratio=8.0,
                             prefill=24)
    state = Trainer(objective, optax.adam(1e-3), key=jax.random.key(7), rollout=rollout).place()[0]
    rollout(state, {}, jax.random.key(1))
    begun = rollout.held_replay
    assert begun is not None
    # The next update owes steps the prefill did not take.
    rollout(dataclasses.replace(state, updates=state.updates + 1), {}, jax.random.key(2))
    assert all(leaf.is_deleted() for leaf in jax.tree.leaves(begun.held))


def test_imagination_is_the_world_models_prior_stepped_in_an_env_rollout():
    """Imagining through `env_rollout` over the `LatentEnv` takes the GRU and the prior's draw each
    step, as stepping the model's own functions one at a time from the same keys does, bit for bit."""
    model = tiny()
    objective = DreamerV3Objective(model)
    variables = objective.init(jax.random.key(3))
    bins = twohot_bins(model.bins)
    env = LatentEnv(model, variables, bins)
    deter, stoch = model.initial((6,))
    state, begin = env.at(deter, stoch, jax.random.key(4))
    actions = jnp.asarray(np.random.default_rng(5).uniform(-1, 1, (4, 6, 1)), jnp.float32)

    def scripted(params, feature, key, memory):
        return actions[memory], {}, memory + 1

    _, imagined = jax.jit(lambda carry: env_rollout(env, scripted, None, carry, 4, reset=False))(
        Carry(state, begin, jax.random.key(6), jnp.zeros((), jnp.int32)))

    @jax.jit
    def stepped(state):
        features = []
        for index in range(4):
            state, step = env.step(state, actions[index])
            features.append(step.observation)
        return jnp.stack(features)

    np.testing.assert_array_equal(imagined.following.observation, stepped(state))


@pytest.mark.parametrize("agent", AGENTS)
def test_an_agent_learns_its_world_from_its_replay(agent):
    """A small DreamerV3 agent acting in batched target walks and trained from its replay at a fixed
    ratio, its actor and critic learning in imagination beside it, predicts the replay's observations
    and rewards better than it did at the start, on the same replayed windows: with the block GRU's
    dynamics or TransDreamer's, over the walk's position, the walks simulated on the host, or its
    pixels, the pixels' agent predicting the privileged position it never reads better too. Its policy's
    return is the acceptance run's to judge, on a task at the steps the paper reports."""
    from dew.objectives.base import Step
    from dew.objectives.rl.config import idle
    from dew.training import Trainer

    environments, model = AGENTS[agent]
    env = environments()
    objective = DreamerV3Objective(model())
    rollout = DreamerRollout(objective, env, capacity=400, batch=8, length=16, train_ratio=32.0, prefill=24)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(7), rollout=rollout)
    initial = jax.device_get(trainer.place()[0])
    data = idle()
    final = trainer.fit(data, steps=150, log_every=150)
    windows = rollout.sample(jax.random.key(8))

    def losses(state):
        step = Step(state.microstep, jax.random.key(9), state.ema)
        _, aux = objective.loss(state.variables, windows, step)
        return {name: float(value) for name, value in aux.metrics.items()}

    before, after = losses(initial), losses(final)
    for name in ("decoder", "reward", *(entry for entry, _, _ in objective.model.auxiliary)):
        assert after[f"loss_{name}"] < before[f"loss_{name}"], (name, before, after)


def test_each_replayed_step_draws_its_posterior_from_its_own_key():
    """The replayed windows' posterior pass draws each step's categoricals from its own key, split
    from the pass's key: the classes drawn are those stepping the posterior a step at a time with those
    keys draws; a pass that reused one key would repeat its draws."""
    model = tiny()
    objective = DreamerV3Objective(model)
    variables = objective.init(jax.random.key(11))
    rng = np.random.default_rng(12)
    observation = jnp.asarray(rng.normal(size=(3, 6, 1)), jnp.float32)
    previous = jnp.asarray(rng.uniform(-1, 1, (3, 6, 1)), jnp.float32)
    first = jnp.zeros((3, 6), bool).at[:, 0].set(True)
    _, stoch, _ = jax.jit(objective.observed)(variables, observation, previous, first, jax.random.key(13))

    @jax.jit
    def stepped(variables):
        deter, drawn = model.initial((3,))
        draws = []
        for t, key in enumerate(jax.random.split(jax.random.key(13), 6)):
            at = model.bind(variables, rngs={"latent": key})
            inputs = observation[:, t], previous[:, t], first[:, t]
            deter, drawn, _ = objective.observe(at, deter, drawn, *inputs)
            draws.append(drawn)
        return jnp.stack(draws, 1)

    # The draws, not the straight-through values around them, which the two programs round apart.
    np.testing.assert_array_equal(np.argmax(stoch, -1), np.argmax(stepped(variables), -1))


def test_the_worlds_kl_losses_are_the_codes_and_the_representation_one_spares_the_prior():
    """A window's dynamics and representation losses are the code's KLs between each step's posterior and
    prior (rssm.py:125-129), KL(sg(post) || prior) and KL(post || sg(prior)), summed over the latent's
    categoricals and floored at the free nat step by step: within float64 rounding of that formula run in
    NumPy over the posterior and prior the model drew. The prior's heads take a gradient from the dynamics
    loss and exactly none from the representation loss, which trains the posterior alone."""
    from dew.objectives.base import Step

    model = tiny(dtype=jnp.float64)
    objective = DreamerV3Objective(model, scales={"policy": 0.0, "value": 0.0, "replay_value": 0.0})
    rng = np.random.default_rng(14)
    first = np.zeros((3, 6), bool)
    first[:, 0] = first[1, 3] = True
    with jax.enable_x64(new_val=True):
        variables = objective.init(jax.random.key(15))
        windows = {"observation": jnp.asarray(3 * rng.normal(size=(3, 6, 1))),
                   "action": jnp.asarray(rng.uniform(-1, 1, (3, 6, 1))),
                   "reward": jnp.asarray(rng.normal(size=(3, 6))), "first": jnp.asarray(first),
                   "last": jnp.zeros((3, 6), bool), "terminal": jnp.zeros((3, 6), bool)}
        step = Step(jnp.asarray(0), jax.random.key(16), None)

        def kls(prior_head):
            params = {**variables["params"], "prior_logits": prior_head}
            metrics = objective.loss({**variables, "params": params}, windows, step)[1].metrics
            return jnp.stack([metrics["loss_dynamics"], metrics["loss_representation"]])

        head = variables["params"]["prior_logits"]
        found, moved = np.asarray(kls(head)), jax.jacrev(kls)(head)
        roundings = chain_roundings(jax.make_jaxpr(kls)(head))
        # The loss draws its posterior from the first of its step key's two halves.
        world_key = jax.random.split(step.key)[0]
        previous = jnp.concatenate([jnp.zeros_like(windows["action"][:, :1]), windows["action"][:, :-1]], 1)
        deter, _, posterior = objective.observed(variables, model.seen(windows["observation"]), previous,
                                                 windows["first"], world_key)
        prior = model.bind(variables, rngs={"latent": world_key}).prior(deter)
    post, prior = np.asarray(posterior, np.float64), np.asarray(prior, np.float64)
    divergence = np.sum(np.exp(post) * (post - prior), axis=(-1, -2))
    expected = np.mean(np.maximum(divergence, objective.free_nats))
    assert_computes_the_oracle(found, np.full(2, expected), "the world's KL losses", roundings=roundings)
    gradients = [np.asarray(leaf) for leaf in jax.tree.leaves(moved)]
    assert all(np.any(gradient[0]) for gradient in gradients)
    assert not any(np.any(gradient[1]) for gradient in gradients)


def demonstrations(count: int, length: int, seed: int) -> list[dict]:
    """`count` target walks of `length` steps each, a scripted pilot steering toward 1, recorded in the
    agent's convention: each row's observation, the reward that came with it, and the action taken at
    it."""
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(count):
        x, rows = rng.uniform(-1, 1), {"observation": [], "action": [], "reward": []}
        reward = 0.0
        for _ in range(length):
            action = float(np.clip(2 * (1 - x), -1, 1))
            rows["observation"].append([x])
            rows["action"].append([action])
            rows["reward"].append(reward)
            x = x + 0.5 * action
            reward = -(x - 1) ** 2
        episodes.append({name: np.asarray(value, np.float32) for name, value in rows.items()})
    return episodes


def test_a_rollout_from_demonstrations_takes_no_prefill_and_writes_on_them():
    """A Dreamer rollout started from a replay of recorded steps takes no random prefill: after 12 updates
    at a train ratio of 32 over windows of 16 in batches of 8, its environments have taken exactly the
    steps those updates owe, written after the recorded ones."""
    from dew.objectives.rl.config import idle
    from dew.rl.replay import ReplayBuffer
    from dew.training import Trainer

    env = Batched(TimeLimit(Target(), episode_length=20), 4)
    objective = DreamerV3Objective(tiny())
    def still(params, observed, key, memory):
        return jnp.zeros((4, 1)), {}, None

    driven = Driven(env)
    _, recorded = env_rollout(driven, still, None, start(driven, jax.random.key(1)), 40, reset=False)
    stream = {**recorded.observation, "action": recorded.action}
    seeded = ReplayBuffer.empty(jax.tree.map(lambda leaf: leaf[0], stream), 400).add(stream)
    rollout = DreamerRollout(objective, env, batch=8, length=16, train_ratio=32.0, prefill=0, replay=seeded)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(7), rollout=rollout)
    data = idle()
    trainer.fit(data, steps=12, log_every=12)
    owed = -(-12 * 8 * 16 // 32)
    assert rollout.environment_steps == 4 * -(-owed // 4)
    held = rollout.held_replay
    assert held is not None and int(held.filled) == 40 + rollout.environment_steps // 4


def test_a_demonstration_replay_holds_each_episode_in_the_agents_convention():
    """A replay of demonstrations holds every episode's rows in order, each row's action the one its
    own observation was answered with, the episode's first and last steps flagged; its windows are
    consecutive steps of the stream. A terminal step before an episode's last is refused."""
    from dew.objectives.rl.dreamer import demonstration_replay

    episodes = [{"observation": np.arange(n, dtype=np.float32)[:, None] + 10 * index,
                 "action": -np.arange(n, dtype=np.float32)[:, None] - 10 * index,
                 "reward": np.arange(n, dtype=np.float32)} for index, n in enumerate((5, 3, 4))]
    replay = demonstration_replay(episodes)
    data = jax.tree.map(lambda leaf: np.asarray(leaf)[:, 0], replay.held)
    np.testing.assert_array_equal(data["observation"], np.concatenate([e["observation"] for e in episodes]))
    np.testing.assert_array_equal(data["action"], -data["observation"])
    np.testing.assert_array_equal(np.flatnonzero(data["first"]), [0, 5, 8])
    np.testing.assert_array_equal(np.flatnonzero(data["last"]), [4, 7, 11])
    windows = np.asarray(replay.sample(jax.random.key(0), 64, 4)["observation"])[..., 0]
    starts = [np.flatnonzero(data["observation"][:, 0] == window[0])[0] for window in windows]
    np.testing.assert_array_equal(windows, np.stack([data["observation"][s:s + 4, 0] for s in starts]))
    with pytest.raises(ValueError):
        demonstration_replay([{**episodes[0], "terminal": np.asarray([0, 1, 0, 0, 0], bool)}])


def pretrained_on(objective, replay):
    """`objective` trained 150 updates on 8 windows of 16 steps an update drawn from `replay`, as pretraining
    reads a replay: its state before and after, and 8 windows drawn apart to score both on."""
    from dew.data.dataset import Loader
    from dew.training import Trainer

    keys = iter(jax.random.split(jax.random.key(11), 1000))
    data = Loader(train=lambda partition: (jax.device_get(replay.sample(next(keys), 8, 16))
                                           for _ in itertools.count()), val=None, records=8, batch=8)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(7))
    initial = jax.device_get(trainer.place()[0])
    final = trainer.fit(data, steps=150, log_every=150)
    return initial, final, jax.device_get(replay.sample(jax.random.key(8), 8, 16))


def test_pretraining_from_demonstrations_learns_the_world_and_clones_the_pilot():
    """A world model pretrained from a pilot's demonstrations alone, its actor cloning the pilot and its
    critic untrained, predicts the demonstrations' observations and rewards better than at the start
    and gives the pilot's actions more likelihood, on the same windows; imagination is skipped."""
    from dew.objectives.base import Step
    from dew.objectives.rl.dreamer import demonstration_replay

    replay = demonstration_replay(demonstrations(8, 20, 3))
    objective = DreamerV3Objective(tiny(), scales={"policy": 0.0, "value": 0.0, "replay_value": 0.0},
                                   imitation=1.0)
    initial, final, windows = pretrained_on(objective, replay)

    def losses(state):
        _, aux = objective.loss(state.variables, windows, Step(state.microstep, jax.random.key(9), state.ema))
        return {name: float(value) for name, value in aux.metrics.items()}

    before, after = losses(initial), losses(final)
    assert "loss_policy" not in after
    for name in ("loss_decoder", "loss_reward", "loss_imitation"):
        assert after[name] < before[name], (name, before[name], after[name])


@pytest.mark.parametrize("name", ["dynamics", "representation", "decoder", "reward", "continue", "imitation"])
def test_an_auxiliary_head_named_as_a_world_model_loss_is_refused(name):
    """An auxiliary head whose name is one of the objective's own losses is refused, where its regression
    would replace that loss (a head named `dynamics` dropped the KL that trains the prior); one named apart
    builds."""
    def headed(head):
        return tiny(0, image=(16, 16, 1), conv_depth=2, reads="image", auxiliary=((head, 1, 8),))

    DreamerV3Objective(headed("physics"))
    with pytest.raises(ValueError):
        DreamerV3Objective(headed(name))


def dreamer_targets(objective, *, checkpointed, replay=None):
    """A Dreamer rollout over eight targets, its replay the prefill's (400 steps) or `replay`."""
    targets = Batched(TimeLimit(Target(), episode_length=20), 8)
    return DreamerRollout(objective, targets, capacity=400, batch=8, length=16, train_ratio=32.0, prefill=24,
                          checkpointed_replay=checkpointed, replay=replay)


def dreamer_fit(directory, objective, rollout, steps, **options):
    """A Dreamer run of `steps` updates over `rollout`, checkpointed every update to `directory`."""
    from dew.checkpoints import Checkpoints, Keep
    from dew.objectives.rl.config import idle
    from dew.training import Trainer

    data = idle()
    checkpoints = None if directory is None else Checkpoints(str(directory), keep=Keep(latest=8))
    trainer = Trainer(objective, options.pop("optimizer", optax.adam(1e-3)), key=jax.random.key(7),
                      rollout=rollout, checkpoints=checkpoints)
    return trainer.fit(data, steps=steps, log_every=1, checkpoint_every=None if directory is None else 1,
                       **options)


@pytest.mark.parametrize("given", ["prefilled", "supplied"])
def test_a_dreamer_run_resumed_from_its_checkpoint_owes_its_steps_and_with_its_replay_ends_alike(tmp_path,
                                                                                                  given):
    """A DreamerV3 run on targets, checkpointed every update, stopped at its third and resumed by a fresh
    rollout: the checkpoint holds the environment steps it took, so the resumed rollout takes, update by
    update, the steps the uninterrupted one did, its environments and replay begun again (its prefill
    taken again); with `checkpointed_replay` it also holds the replay (the prefill's, or one the rollout
    was given, of another capacity), the targets mid episode and their returns, so the resumed run takes
    the very steps and its last state is the uninterrupted run's, every leaf bit for bit."""
    supplied = None
    if given == "supplied":
        # A replay of 300 steps, filled by another rollout's prefill: the rollout's own capacity is 400.
        seed = DreamerRollout(DreamerV3Objective(tiny()), Batched(TimeLimit(Target(), episode_length=20), 8),
                              capacity=300, batch=8, length=16, train_ratio=32.0, prefill=24)
        dreamer_fit(None, seed.objective, seed, 1)
        supplied = seed.held_replay
    for checkpointed in (False, True):
        objective = DreamerV3Objective(tiny())

        def made(objective=objective, checkpointed=checkpointed):
            replay = None if supplied is None else jax.tree.map(jnp.copy, supplied)
            return dreamer_targets(objective, checkpointed=checkpointed, replay=replay)

        uninterrupted = made()
        whole = dreamer_fit(tmp_path / f"whole-{checkpointed}", objective, uninterrupted, 5)
        dreamer_fit(tmp_path / f"stopped-{checkpointed}", objective, made(), 3)
        resumed = made()
        again = dreamer_fit(tmp_path / f"stopped-{checkpointed}", objective, resumed, 5)
        prefilled = 0 if checkpointed or supplied is not None else 24 * 8
        assert resumed.environment_steps == uninterrupted.environment_steps + prefilled
        if checkpointed:
            for one, other in zip(bits(whole), bits(again), strict=True):
                np.testing.assert_array_equal(one, other)


def test_a_dreamer_rollouts_step_counts_past_32_bits_come_back_from_its_checkpoint(tmp_path):
    """A Dreamer rollout's counts of steps past 2**31 and 2**32, saved in a run's checkpoint with JAX in 32
    bits, come back exact into a fresh rollout (`DreamerRollout.restored`)."""
    from dew.checkpoints import Checkpoints
    from dew.training._fit import _rollout_resumed, _rollout_saved

    objective = DreamerV3Objective(tiny())
    rollout = dreamer_targets(objective, checkpointed=False)
    state = dreamer_fit(None, objective, rollout, 1)
    rollout._taken, rollout._prefilled = 2**33 + 2**31 + 7, 2**31 + 5
    checkpoints = Checkpoints(str(tmp_path / "run"))
    checkpoints.save(1, state, rollout=_rollout_saved(rollout))
    checkpoints.wait()
    back = dreamer_targets(objective, checkpointed=False)
    _rollout_resumed(back, checkpoints, 1, state.key.sharding)
    assert (back.environment_steps, back._prefilled) == (2**33 + 2**31 + 7, 2**31 + 5)


def test_a_dreamer_rollouts_steps_saved_without_its_pacing_come_back_from_its_checkpoint(tmp_path):
    """A Dreamer rollout's step counts saved in a run's checkpoint without where it owes its updates from (as
    an earlier Dew saved them) come back into a fresh rollout through the run's restore
    (`_rollout_resumed`), which then owes from them."""
    from dew.checkpoints import Checkpoints
    from dew.training._fit import _rollout_resumed, _rollout_saved

    objective = DreamerV3Objective(tiny())
    rollout = dreamer_targets(objective, checkpointed=False)
    state = dreamer_fit(None, objective, rollout, 2)
    saved = {name: leaf for name, leaf in _rollout_saved(rollout).items() if name in ("taken", "prefilled")}
    checkpoints = Checkpoints(str(tmp_path / "run"))
    checkpoints.save(2, state, rollout=saved)
    checkpoints.wait()
    back = dreamer_targets(objective, checkpointed=False)
    _rollout_resumed(back, checkpoints, 2, state.key.sharding)
    assert (back.environment_steps, back._prefilled) == (rollout.environment_steps, rollout._prefilled)


def test_a_dreamer_run_resumed_from_a_step_without_its_state_owes_only_the_steps_after_it(tmp_path):
    """A DreamerV3 run resumed at its third update from checkpoints written without its rollout's state
    (as an earlier Dew wrote them) begins its targets and replay again and owes the steps of its fourth
    and fifth updates alone: after its prefill it takes at least the steps those updates train on, and
    less than one step of every target more, where taking the run's whole history again would take the
    steps of all five."""
    class Forgetting(DreamerRollout):
        """The rollout saving nothing of what it carries, as one did before checkpoints held it."""

        def saved(self):
            return None

    objective = DreamerV3Objective(tiny())
    targets = Batched(TimeLimit(Target(), episode_length=20), 8)

    def forgetting():
        return Forgetting(objective, targets, capacity=400, batch=8, length=16, train_ratio=32.0, prefill=24)

    dreamer_fit(tmp_path / "run", objective, forgetting(), 3)
    resumed = forgetting()
    dreamer_fit(tmp_path / "run", objective, resumed, 5)
    owed = [math.ceil(updates * 8 * 16 / 32.0) for updates in (3, 5)]
    taken = resumed.environment_steps - 24 * 8 - owed[0]
    assert owed[1] - owed[0] <= taken < owed[1] - owed[0] + 8


def test_a_dreamer_run_resumed_with_longer_windows_owes_the_updates_after_it_at_their_pace(tmp_path):
    """A DreamerV3 run of 16-step windows, resumed at its third update by a rollout drawing 64-step windows
    (as a later phase of training lengthens them), owes its fourth and fifth updates the steps their
    windows train on alone: after its prefill it takes at least those steps, and less than one step of
    every target more, where owing the whole run's updates at the longer windows' pace would take twice
    as many."""
    objective = DreamerV3Objective(tiny())
    targets = Batched(TimeLimit(Target(), episode_length=20), 8)

    def drawing(length):
        return DreamerRollout(objective, targets, capacity=400, batch=8, length=length, train_ratio=32.0,
                              prefill=64)

    first = drawing(16)
    dreamer_fit(tmp_path / "run", objective, first, 3)
    resumed = drawing(64)
    dreamer_fit(tmp_path / "run", objective, resumed, 5)
    owed = 2 * 8 * 64 // 32
    taken = resumed.environment_steps - first.environment_steps - 64 * 8
    assert owed <= taken < owed + 8


def test_a_dreamer_rollout_restored_after_collecting_for_an_uncommitted_update_owes_it_none_again():
    """A Dreamer rollout called twice for its third update (an accumulation window's microsteps, or an
    update the trainer rejected and draws again), its state saved after the first call and restored with
    its replay into a fresh rollout, takes on the second call and the updates after it the steps the
    uninterrupted rollout takes: the steps it took for the update not yet committed stand paid."""
    from types import SimpleNamespace

    objective = DreamerV3Objective(tiny())
    variables = objective.init(jax.random.key(0))
    batch = {}

    def called(rollout, updates):
        for at, update in enumerate(updates):
            rollout(SimpleNamespace(variables=variables, updates=update), batch, jax.random.key(at))

    uninterrupted = dreamer_targets(objective, checkpointed=True)
    called(uninterrupted, (0, 1, 2, 2, 3, 4))
    stopped = dreamer_targets(objective, checkpointed=True)
    called(stopped, (0, 1, 2))
    resumed = dreamer_targets(objective, checkpointed=True)
    resumed.restored(stopped.saved())
    called(resumed, (2, 3, 4))
    assert resumed.environment_steps == uninterrupted.environment_steps


def test_a_dreamer_rollout_restored_in_a_later_phase_owes_each_update_at_its_own_pace_once():
    """A Dreamer rollout of 16-step windows restored at 64 (a later phase), called twice for its fourth
    update, takes the steps that update trains on once: restored between the two calls, it takes on the
    second and the update after it the steps the rollout restored once takes. One of 32-step windows
    restored at 16 takes for its next update the steps that update trains on at 16 (4, a step of each of
    its eight targets)."""
    from types import SimpleNamespace

    objective = DreamerV3Objective(tiny())
    variables = objective.init(jax.random.key(0))
    targets = Batched(TimeLimit(Target(), episode_length=20), 8)

    def drawing(length, restored=None):
        rollout = DreamerRollout(objective, targets, capacity=400, batch=8, length=length, train_ratio=32.0,
                                 prefill=64, checkpointed_replay=True)
        if restored is not None:
            # A copy, as a checkpoint holds it: the restored rollout's calls donate what it carries.
            rollout.restored(jax.tree.map(jnp.copy, restored.saved()))
        return rollout

    def called(rollout, updates):
        for update in updates:
            rollout(SimpleNamespace(variables=variables, updates=update), {}, jax.random.key(update))
        return rollout

    earlier = called(drawing(16), (0, 1, 2))
    once = called(drawing(64, earlier), (3, 3, 4))
    stopped = called(drawing(64, earlier), (3,))
    resumed = called(drawing(64, stopped), (3, 4))
    assert resumed.environment_steps == once.environment_steps
    longer = called(drawing(32), (0, 1))
    shorter = called(drawing(16, longer), (2,))
    assert shorter.environment_steps - longer.environment_steps == 8


def test_a_dreamer_runs_cooldown_over_a_simulator_leaves_it_as_the_run_left_it():
    """A DreamerV3 run on targets a simulator steps, with a cooldown at its third update whose branch
    collects steps of its own, ends, every leaf bit for bit, where the run with no cooldown does: the
    branch runs on copies of the rollout's replay, and the simulator keeps its own targets and the driver
    the episodes it is to restart (`DrivenSimulator.branch`)."""
    from dew.objectives.base import Shown
    from dew.rl.evaluation import episodes
    from dew.training import Episodes, EvalSuite
    from dew.training.cooldown import Cooldowns

    objective = DreamerV3Objective(tiny())
    flights = {"flight": EvalSuite(Episodes(episodes(objective.act, Driven(Batched(TimeLimit(
        Target(), episode_length=20), 4)), 20, memory=objective.memory),
        {"return": Shown(better="higher"), "steps": Shown()}), every=6)}
    options = {"optimizer": optax.inject_hyperparams(optax.adam)(learning_rate=1e-3), "validation": flights}

    def simulated():
        # Every update owes two steps of each target, so the branch's collect.
        return DreamerRollout(objective, HostTarget(), capacity=400, batch=8, length=16, train_ratio=8.0,
                              prefill=24)

    plain = bits(dreamer_fit(None, objective, simulated(), 6, **options))
    cooled = dreamer_fit(None, objective, simulated(), 6, cooldowns=Cooldowns(at=(3,), length=2), **options)
    for one, other in zip(plain, bits(cooled), strict=True):
        np.testing.assert_array_equal(one, other)


def test_a_dreamer_runs_cooldown_leaves_its_replay_as_the_run_left_it(tmp_path):
    """A DreamerV3 run with a checkpointed replay and a cooldown at its third update ends, every leaf bit
    for bit, where the run with no cooldown does: the branch runs on copies of the rollout's targets and
    replay (`DreamerRollout.branch`), the run's own donated buffers kept aside. Preempted as its branch
    begins, the run saves its step's state with the rollout's as the run left it, and its resume runs the
    branch again and ends there too."""
    from recording import preempting

    from dew.checkpoints import Checkpoints
    from dew.objectives.base import Shown
    from dew.rl.evaluation import episodes
    from dew.training import Episodes, EvalSuite, Preempted
    from dew.training.cooldown import Cooldowns

    objective = DreamerV3Objective(tiny())
    # The agent acts on the steps its driver records (`Driven`).
    flights = {"flight": EvalSuite(Episodes(episodes(objective.act, Driven(Batched(TimeLimit(
        Target(), episode_length=20), 4)), 20, memory=objective.memory),
        {"return": Shown(better="higher"), "steps": Shown()}), every=6)}
    options = {"optimizer": optax.inject_hyperparams(optax.adam)(learning_rate=1e-3), "validation": flights}
    plain = bits(dreamer_fit(None, objective, dreamer_targets(objective, checkpointed=True), 6, **options))
    cooldowns = Cooldowns(at=(3,), length=2)
    cooled = dreamer_fit(None, objective, dreamer_targets(objective, checkpointed=True), 6,
                         cooldowns=cooldowns, **options)
    for one, other in zip(plain, bits(cooled), strict=True):
        np.testing.assert_array_equal(one, other)
    with pytest.raises(Preempted):
        dreamer_fit(tmp_path / "run", objective, dreamer_targets(objective, checkpointed=True), 6,
                    cooldowns=cooldowns, observe=preempting(2), **options)
    assert Checkpoints(str(tmp_path / "run")).latest == 3
    again = dreamer_fit(tmp_path / "run", objective, dreamer_targets(objective, checkpointed=True), 6,
                        cooldowns=cooldowns, **options)
    for one, other in zip(plain, bits(again), strict=True):
        np.testing.assert_array_equal(one, other)
