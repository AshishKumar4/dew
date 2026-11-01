"""A student's own steps labelled by a privileged teacher (`dew.rl.aggregation`), written by a Dreamer
rollout beside the actions it took (`DreamerRollout(teacher=)`), imitated in their place
(`DreamerV3Objective(imitates=)`), and a Dreamer agent scored greedy (`DreamerV3Objective.greedy`)."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_dreamer import HostTarget, Target, demonstrations, dreamer_fit, tiny

from dew.objectives.base import Step
from dew.objectives.rl.control import ActorCritic, ControlPPOObjective
from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective, demonstration_replay
from dew.rl.aggregation import LABEL, Teacher, labelled, paired
from dew.rl.env import Batched, Carry, Driven, TimeLimit, env_rollout, start
from dew.rl.replay import ReplayBuffer


def pilot(params, observation, key, memory):
    """A privileged pilot steering a target toward 1: twice its distance, scaled by its `gain` and bounded,
    arithmetic every order computes alike."""
    return jnp.clip(params["gain"] * 2 * (1 - observation), -1, 1), {}, None


def test_a_labelled_student_acts_as_itself_and_each_step_holds_the_teachers_action():
    """A PPO student labelled by a pilot takes, in walks of targets, the very steps it takes alone, its own
    extras kept, every leaf bit for bit; each step's label is the pilot's action at the observation the
    student acted on."""
    objective = ControlPPOObjective(ActorCritic(1, (8,), (8,)), 1, discount=0.9)
    variables = objective.init(jax.random.key(0))
    env = Batched(TimeLimit(Target(), episode_length=8), 16)
    taught = {"gain": jnp.asarray(0.75)}

    def flown(policy, params):
        return jax.jit(lambda params: env_rollout(env, policy, params, start(env, jax.random.key(1)), 24))(
            params)[1]

    alone = flown(objective.act, variables)
    together = flown(labelled(objective.act, pilot), paired(variables, taught))
    for one, other in zip(jax.tree.leaves((alone.observation, alone.action, alone.extras, alone.following)),
                          jax.tree.leaves((together.observation, together.action,
                                           {name: value for name, value in together.extras.items()
                                            if name != LABEL}, together.following)), strict=True):
        np.testing.assert_array_equal(one, other)
    expected = np.clip(0.75 * 2 * (1 - np.asarray(together.observation)), -1, 1)
    np.testing.assert_array_equal(np.asarray(together.extras[LABEL]), expected.astype(np.float32))


@pytest.mark.parametrize("simulated", [False, True])
def test_a_dreamer_rollout_with_a_teacher_writes_each_steps_label_beside_the_action_it_took(simulated):
    """A Dreamer rollout given a pilot as its teacher, over targets in JAX or simulated on the host, writes
    every step it takes, its random prefill's and its agent's, with the pilot's action at the step's
    observation, which the agent's own actions are not."""
    objective = DreamerV3Objective(tiny())
    env = HostTarget() if simulated else Batched(TimeLimit(Target(), episode_length=20), 8)
    rollout = DreamerRollout(objective, env, capacity=400, batch=8, length=16, train_ratio=8.0, prefill=24,
                             teacher=Teacher(pilot, {"gain": jnp.asarray(0.75)}))
    dreamer_fit(None, objective, rollout, 4)
    replay = rollout.held_replay
    assert replay is not None
    filled = int(replay.filled)
    assert filled > 24
    held = jax.device_get(replay.held)
    observation = np.asarray(held["observation"][:filled])
    expected = np.clip(0.75 * 2 * (1 - observation), -1, 1).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(held[LABEL][:filled]), expected)
    assert not np.array_equal(np.asarray(held["action"][:filled]), expected)


def test_a_replay_given_to_a_rollout_holds_a_label_exactly_where_it_has_a_teacher():
    """A rollout with a teacher is refused a replay of unlabelled steps, and one without a teacher a replay
    of labelled steps: each writes the steps it takes on the replay it is given."""
    objective = DreamerV3Objective(tiny())
    env = Batched(TimeLimit(Target(), episode_length=20), 8)
    seeded = demonstration_replay(demonstrations(8, 20, 3))
    held = seeded.held
    labelled_replay = ReplayBuffer({**held, LABEL: held["action"]}, seeded.cursor, seeded.filled)
    teacher = Teacher(pilot, {"gain": jnp.asarray(0.75)})
    DreamerRollout(objective, env, replay=labelled_replay, teacher=teacher)
    DreamerRollout(objective, env, replay=seeded)
    with pytest.raises(ValueError):
        DreamerRollout(objective, env, replay=seeded, teacher=teacher)
    with pytest.raises(ValueError):
        DreamerRollout(objective, env, replay=labelled_replay)


def test_an_objective_imitating_the_label_reads_it_in_the_imitation_alone():
    """Imitating each step's label changes the imitation loss alone: the world model's losses read the
    actions taken, bit for bit those of the objective that imitates them; and where every label is the
    action taken, the two objectives' losses are one, bit for bit."""
    replay = demonstration_replay(demonstrations(8, 20, 3))
    windows = jax.device_get(replay.sample(jax.random.key(8), 8, 16))
    rng = np.random.default_rng(4)
    relabelled = {**windows, LABEL: rng.uniform(-1, 1, np.shape(windows["action"])).astype(np.float32)}
    scales = {"policy": 0.0, "value": 0.0, "replay_value": 0.0}
    taken = DreamerV3Objective(tiny(), scales=scales, imitation=1.0)
    taught = DreamerV3Objective(tiny(), scales=scales, imitation=1.0, imitates=LABEL)
    variables = taken.init(jax.random.key(3))
    step = Step(jnp.zeros((), jnp.int32), jax.random.key(9), None)

    def metrics(objective, batch):
        _, aux = objective.loss(variables, batch, step)
        return {name: np.asarray(value) for name, value in aux.metrics.items()}

    actions, labels = metrics(taken, relabelled), metrics(taught, relabelled)
    assert set(actions) == set(labels)
    for name in actions:
        if name != "loss_imitation":
            np.testing.assert_array_equal(actions[name], labels[name])
    assert actions["loss_imitation"] != labels["loss_imitation"]
    same = metrics(taught, {**windows, LABEL: windows["action"]})
    for name in actions:
        np.testing.assert_array_equal(metrics(taken, windows)[name], same[name])


def test_an_agent_trained_to_imitate_labels_learns_the_labels_not_the_actions_taken():
    """An agent pretrained on targets whose steps took uniformly random actions, each labelled with a pilot's,
    imitating the labels gives the pilot's actions more likelihood than it did, and more than it gives the
    actions taken."""
    from dew.data.dataset import Loader
    from dew.training import Trainer

    env = Batched(TimeLimit(Target(), episode_length=20), 8)
    seeding = DreamerRollout(DreamerV3Objective(tiny()), env, capacity=400, batch=8, length=16,
                             train_ratio=8.0, prefill=200, teacher=Teacher(pilot, {"gain": jnp.asarray(1.0)}))
    dreamer_fit(None, seeding.objective, seeding, 1)
    replay = seeding.held_replay
    assert isinstance(replay, ReplayBuffer)
    objective = DreamerV3Objective(tiny(), scales={"policy": 0.0, "value": 0.0, "replay_value": 0.0},
                                   imitation=1.0, imitates=LABEL)
    keys = iter(jax.random.split(jax.random.key(11), 1000))
    data = Loader(train=lambda partition: (jax.device_get(replay.sample(next(keys), 8, 16))
                                           for _ in itertools.count()), val=None, records=8, batch=8)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(7))
    initial = jax.device_get(trainer.place()[0])
    final = trainer.fit(data, steps=150, log_every=150)
    windows = jax.device_get(replay.sample(jax.random.key(8), 8, 16))

    def imitation(state, field):
        batch = {**windows, LABEL: windows[field]}
        _, aux = objective.loss(state.variables, batch, Step(state.microstep, jax.random.key(9), state.ema))
        return float(aux.metrics["loss_imitation"])

    assert imitation(final, LABEL) < imitation(initial, LABEL)
    assert imitation(final, LABEL) < imitation(final, "action")


def test_a_greedy_step_is_the_agents_step_without_its_draw():
    """An agent whose policy spreads every action by 0.25 acts, from any key, as its greedy step plus 0.25
    times the key's draw, bit for bit, the same latent drawn by both; the greedy step remembers the action
    it took."""
    objective = DreamerV3Objective(tiny(min_std=0.25, max_std=0.25))
    variables = objective.init(jax.random.key(0))
    driven = Driven(Batched(TimeLimit(Target(), episode_length=20), 8))
    carry = start(driven, jax.random.key(1), objective.memory(8))
    for index in range(3):
        key = jax.random.fold_in(jax.random.key(2), index)
        acted, _, _ = objective.act(variables, carry.observation, key, carry.memory)
        greedy, _, remembered = objective.greedy(variables, carry.observation, key, carry.memory)
        draw = jax.random.normal(jax.random.split(key)[1], greedy.shape)
        np.testing.assert_array_equal(np.asarray(acted), np.asarray(jnp.clip(greedy + 0.25 * draw, -1, 1)))
        np.testing.assert_array_equal(np.asarray(remembered.previous), np.asarray(greedy))
        state, following = driven.step(carry.state, greedy)
        carry = Carry(state, following.observation, carry.key, remembered)
