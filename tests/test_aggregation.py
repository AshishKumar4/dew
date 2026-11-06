"""A student's own steps labelled by a privileged teacher (`dew.rl.aggregation`), written by a Dreamer
rollout beside the actions it took (`DreamerRollout(teacher=)`), imitated in their place
(`DreamerV3Objective(imitates=)`), and a Dreamer agent scored greedy (`DreamerV3Objective.greedy`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_dreamer import HostTarget, Target, demonstrations, dreamer_fit, pretrained_on, tiny

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


def prefill(params, observed, key, memory):
    """A Dreamer rollout's prefill policy over a driven step (`DreamerRollout`): each action uniform in
    [-1, 1]."""
    return jax.random.uniform(key, (observed["reward"].shape[0], 1), minval=-1, maxval=1), {}, memory


def driven_pilot(params, observed, key, memory):
    """`pilot` reading the observation a driven step holds, as a rollout's teacher reads it."""
    return pilot(params, observed["observation"], key, memory)


def test_an_agent_trained_to_imitate_labels_learns_the_labels_not_the_actions_taken():
    """An agent pretrained on targets whose steps took uniformly random actions, each labelled with a pilot's,
    imitating the labels gives the pilot's actions more likelihood than it did, and more than it gives the
    actions taken. The replay is a labelled prefill's, 200 steps of 8 walks written as a Dreamer rollout
    writes them (its policy labelled by its teacher, `labelled`, over `Driven` steps), flown as one scan
    rather than through a rollout's first update, which nothing here reads."""
    env = Driven(Batched(TimeLimit(Target(), episode_length=20), 8))
    policy = labelled(prefill, driven_pilot)
    _, flown = jax.jit(lambda params: env_rollout(env, policy, params, start(env, jax.random.key(1)), 200,
                                                  reset=False))(paired({}, {"gain": jnp.asarray(1.0)}))
    steps = {**flown.observation, "action": flown.action, LABEL: flown.extras[LABEL]}
    replay = ReplayBuffer.empty(jax.tree.map(lambda leaf: leaf[0], steps), 400).add(steps)
    objective = DreamerV3Objective(tiny(), scales={"policy": 0.0, "value": 0.0, "replay_value": 0.0},
                                   imitation=1.0, imitates=LABEL)
    initial, final, windows = pretrained_on(objective, replay)

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


@pytest.mark.parametrize("simulated", [False, True])
@pytest.mark.parametrize("taught", [False, True])
def test_a_demonstrated_prefill_writes_the_demonstrators_actions_and_a_teachers_labels(simulated, taught):
    """A Dreamer rollout whose prefill a pilot demonstrates, over targets in JAX or simulated on the host,
    writes each prefill step with the pilot's action at its observation, and, given a second pilot as its
    teacher, that one's action as its label; the agent then acts on from the memory it started with, its
    own steps the pilot's actions no more."""
    objective = DreamerV3Objective(tiny())
    env = HostTarget() if simulated else Batched(TimeLimit(Target(), episode_length=20), 8)
    teacher = Teacher(pilot, {"gain": jnp.asarray(0.5)}) if taught else None
    rollout = DreamerRollout(objective, env, capacity=400, batch=8, length=16, train_ratio=8.0, prefill=24,
                             demonstrator=Teacher(pilot, {"gain": jnp.asarray(0.75)}), teacher=teacher)
    dreamer_fit(None, objective, rollout, 2)
    written = jax.device_get(rollout.held_replay)
    filled, held = int(written.filled), written.held
    assert filled > 24
    observation, action = np.asarray(held["observation"][:filled]), np.asarray(held["action"][:filled])

    def piloted(gain):
        return np.clip(gain * 2 * (1 - observation), -1, 1).astype(np.float32)

    np.testing.assert_array_equal(action[:24], piloted(0.75)[:24])
    assert not np.array_equal(action[24:], piloted(0.75)[24:])
    assert (LABEL in held) == taught
    if taught:
        np.testing.assert_array_equal(np.asarray(held[LABEL][:filled]), piloted(0.5))


@pytest.mark.parametrize("demonstrated", [False, True])
def test_a_prefill_flown_a_window_at_a_time_writes_the_replay_one_pass_writes(demonstrated):
    """A prefill of 24 steps over targets that restart within it, uniform or demonstrated by a PPO policy's
    draws, flown in windows of 8 writes the replay, every leaf bit for bit, that one flown in one window of
    24 writes: each window carries the environments, their keys and the replay on from where the last left
    them. Pretraining on it (`train_ratio` infinite) takes no step past the prefill."""
    policy = ControlPPOObjective(ActorCritic(1, (8,), (8,)), 1)
    demonstrator = Teacher(policy.act, policy.init(jax.random.key(4))) if demonstrated else None

    def prefilled(length):
        objective = DreamerV3Objective(tiny())
        rollout = DreamerRollout(objective, Batched(TimeLimit(Target(), episode_length=20), 8), capacity=400,
                                 batch=8, length=length, train_ratio=float("inf"), prefill=24,
                                 demonstrator=demonstrator)
        dreamer_fit(None, objective, rollout, 2)
        assert rollout.environment_steps == 24 * 8
        return jax.device_get(rollout.held_replay)

    for one, other in zip(jax.tree.leaves(prefilled(8)), jax.tree.leaves(prefilled(24)), strict=True):
        np.testing.assert_array_equal(one, other)


def test_a_demonstrator_beside_a_given_replay_is_refused():
    """A demonstrator flies the prefill, which a replay given to the rollout replaces: a rollout given both
    is refused, and either alone is not."""
    objective = DreamerV3Objective(tiny())
    env = Batched(TimeLimit(Target(), episode_length=20), 8)
    seeded = demonstration_replay(demonstrations(8, 20, 3))
    demonstrator = Teacher(pilot, {"gain": jnp.asarray(0.75)})
    DreamerRollout(objective, env, replay=seeded)
    DreamerRollout(objective, env, demonstrator=demonstrator)
    with pytest.raises(ValueError):
        DreamerRollout(objective, env, replay=seeded, demonstrator=demonstrator)
