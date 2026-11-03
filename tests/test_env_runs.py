"""Environment runs built from their records (`dew.objectives.rl.config.EnvRunConfig`): an agent trained in
the gate race as the same run composed in Python trains, its record read back training it again, a teacher
saved as a run labelling and demonstrating the steps its own policy would, stages chained from the run before
them, a resume ending where the run that never stopped ends, and each setup the kind refuses."""

import dataclasses
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from recording import bits
from test_env_rollout import evaluation_key

from dew.config import ObjectiveConfig, RunConfig, TrainerConfig
from dew.nn.kernels import generation
from dew.objectives.rl.config import EnvConfig, EnvRunConfig, EpisodesConfig, RolloutConfig, TeacherConfig
from dew.objectives.rl.control import ActorCritic, ControlPPOObjective, ControlRollout
from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective, WorldModel
from dew.objectives.rl.tssm import TSSM
from dew.optim import OptimConfig
from dew.registry import ModelConfig, envs
from dew.rl.aggregation import LABEL, Teacher
from dew.rl.env import Batched
from dew.rl.evaluation import episodes
from dew.rl.flight import TEACHER, FlightEnv, GateRace, Racing
from dew.training import Trainer

WORLD_MODEL, ACTOR_CRITIC = "dew.objectives.rl.dreamer:WorldModel", "dew.objectives.rl.control:ActorCritic"
DREAMER_V3 = "dew.objectives.rl.dreamer:DreamerV3Objective"
CONTROL_PPO = "dew.objectives.rl.control:ControlPPOObjective"
"""The agent's networks and objectives, by their import paths."""

RACING = {"families": [["loop", 1.0]], "obstacles": 0, "episode_seconds": 0.4, "hz": 30.0}
"""A race of loops, each episode cut after 12 steps, so the runs' few steps restart episodes."""
VIEWS = ("state", "teacher")
"""What a race shows its agent: the drone's state, which the student reads, and the teacher's view."""
TINY = {"deter": 16, "hidden": 8, "stoch": 4, "classes": 4, "blocks": 2, "units": 8}
"""A world model's widths at a test's size."""
DYNAMICS = {"action_size": 4, "stoch": 4, "classes": 4, "width": 16, "layers": 1, "heads": 2, "context": 8}
"""TransDreamer's dynamics at a test's size, over windows of 8 steps."""
OPTIM = OptimConfig(optimizer="adam", learning_rate=1e-3)


def racing() -> Racing:
    return Racing(families=(("loop", 1.0),), obstacles=0, episode_seconds=0.4, hz=30.0)


def trainer(root: Path, steps: int, **fields) -> TrainerConfig:
    return TrainerConfig(checkpoint_dir=str(root), steps=steps, log_every=1, eval_every=None,
                         checkpoint_every=None, multi_host=False, **fields)


def flight(count: int = 16) -> EnvConfig:
    return EnvConfig("flight", {"racing": RACING, "views": list(VIEWS)}, count=count)


def teacher_run(root: Path, steps: int = 2) -> EnvRunConfig:
    """PPO over the race's privileged view, as tools/fpv_race.py's teacher, at a test's size."""
    model = ModelConfig(ACTOR_CRITIC, {"policy_layers": [16], "value_layers": [16]})
    return EnvRunConfig(env=flight(), model=model,
                        objective=ObjectiveConfig(CONTROL_PPO, {"discount": 0.9, "reads": ["teacher"]}),
                        rollout=RolloutConfig("control", {"unroll_length": 4, "epochs": 1, "minibatches": 2}),
                        optim=OPTIM, trainer=trainer(root, steps))


def teacher_policy() -> ControlPPOObjective:
    """`teacher_run`'s objective composed in Python."""
    return ControlPPOObjective(ActorCritic(4, (16,), (16,)), TEACHER, discount=0.9, reads=("teacher",))


def student(root: Path, steps: int, **fields) -> EnvRunConfig:
    """A Dreamer agent of the drone's state in the race, at a test's size."""
    rollout = {"batch": 8, "length": 8, "train_ratio": 16.0, "prefill": 8, "capacity": 64}
    run = EnvRunConfig(env=flight(8), model=ModelConfig(WORLD_MODEL, {**TINY, "reads": "state"}),
                       rollout=RolloutConfig("dreamer", rollout), optim=OPTIM, trainer=trainer(root, steps))
    return dataclasses.replace(run, **fields)


def world_model(**fields) -> WorldModel:
    """`student`'s world model composed in Python, computing in the run's numerics' bfloat16."""
    return WorldModel(13, 4, reads="state", dtype=jnp.bfloat16, **{**TINY, **fields})


def fitted(run: EnvRunConfig, objective, rollout, steps: int):
    """`objective` trained `steps` updates over `rollout` by a trainer composed in Python, on the loader the
    run's rollout replaces, under the run's numerics, as `RunConfig.train` traces its programs."""
    with generation.run_precision(run.trainer.precision):
        return Trainer(objective, run.optim, key=0, rollout=rollout).fit(run.prepare().dataset, steps=steps,
                                                                         log_every=1)


def assert_bitwise(one, other) -> None:
    for left, right in zip(bits(one), bits(other), strict=True):
        np.testing.assert_array_equal(left, right)


def ppo(root: Path):
    run = dataclasses.replace(teacher_run(root, 3), episodes={"episodes": EpisodesConfig(steps=16)})
    objective = teacher_policy()
    env = Batched(FlightEnv(racing(), VIEWS), 16)
    return run, objective, ControlRollout(objective, env, unroll_length=4, epochs=1, minibatches=2)


def dreamer(root: Path):
    run = student(root, 3)
    objective = DreamerV3Objective(world_model())
    env = Batched(FlightEnv(racing(), VIEWS), 8)
    return run, objective, DreamerRollout(objective, env, capacity=64, batch=8, length=8, train_ratio=16.0,
                                          prefill=8)


def transdreamer_race(root: Path):
    """TransDreamer's agent in the race a host simulator steps (`GateRace`, no images)."""
    race = {"renderer": None, "racing": RACING, "views": list(VIEWS), "count": 8}
    model = ModelConfig(WORLD_MODEL, {**TINY, "reads": "state", "dynamics": DYNAMICS})
    run = student(root, 3, env=EnvConfig("gate_race", race), model=model)
    objective = DreamerV3Objective(world_model(dynamics=TSSM(**DYNAMICS)))
    simulator = GateRace(None, racing(), views=VIEWS, count=8)
    return run, objective, DreamerRollout(objective, simulator, capacity=64, batch=8, length=8,
                                          train_ratio=16.0, prefill=8)


@pytest.mark.parametrize("composed", [ppo, dreamer, transdreamer_race], ids=lambda made: made.__name__)
def test_a_run_from_its_record_trains_as_the_run_composed_in_python(tmp_path, composed):
    """An agent trained in the race through its record (`prepare().train`), PPO over the privileged view
    validated by its greedy flights, DreamerV3's block GRU over the drone's state, and TransDreamer's in a
    race a host simulator steps, ends, every leaf of its state bit for bit, where the same objective and
    rollout composed in Python end; the run.json it writes reads back as the run, which trains again under
    another name to the same state."""
    run, objective, rollout = composed(tmp_path)
    state = run.prepare().train("record")
    assert_bitwise(state, fitted(run, objective, rollout, 3))
    saved = RunConfig.load(str(tmp_path / "record"))
    assert isinstance(saved, EnvRunConfig)
    assert EnvRunConfig.from_dict(saved.to_dict()) == saved
    assert_bitwise(state, saved.prepare().train("again"))


@pytest.mark.parametrize("drawn", [False, True], ids=["greedy", "drawn"])
@pytest.mark.parametrize("agent", ["ppo", "dreamer", "dreamer in a simulator"])
def test_a_runs_episodes_score_its_flights_from_the_evaluations_key(tmp_path, agent, drawn):
    """A run validated by its episodes logs at its end `episodes/return` and `episodes/steps`: the first
    episodes its greedy policy earns, or with `drawn` the policy's draws as it acts (`act`), flown 16 steps
    from the key the run's key gives its last step (`dew.rl.evaluation.episodes`), on its averaged weights
    where it keeps them; a PPO policy over the race's observation, a Dreamer agent over the driven step
    (`Driven`) from the memory it starts with, and a Dreamer agent trained in a race a host simulator
    steps, flown in a race of the split's own (`DrivenSimulator`)."""
    from dew.rl.env import Driven
    from dew.rl.simulator import DrivenSimulator
    from dew.training.tracker import LocalTracker

    run, objective, _ = {"ppo": ppo, "dreamer": dreamer, "dreamer in a simulator": transdreamer_race}[agent](
        tmp_path)
    count = 16 if agent == "ppo" else 8
    simulated = {"renderer": None, "racing": RACING, "views": list(VIEWS), "count": count}
    own = EnvConfig("gate_race", simulated) if agent == "dreamer in a simulator" else None
    split = EpisodesConfig(steps=16, env=own, drawn=drawn)
    run = dataclasses.replace(run, episodes={"episodes": split})
    state = run.prepare().train("scored")
    logged = {name: value for _, scalars in LocalTracker(str(tmp_path / "scored" / "tracking")).scalars()
              for name, value in scalars.items() if name.startswith("episodes/")}
    policy = objective.act if drawn else objective.greedy
    if agent == "ppo":
        race = Batched(FlightEnv(racing(), VIEWS), count)
        flown = episodes(policy, race, 16)
    else:
        race = (DrivenSimulator(GateRace(None, racing(), views=VIEWS, count=count)) if own is not None
                else Driven(Batched(FlightEnv(racing(), VIEWS), count)))
        flown = episodes(policy, race, 16, memory=objective.memory)
    with generation.run_precision(run.trainer.precision):
        expected = flown(state.variables if state.ema is None else state.averaged,
                         evaluation_key(state.key, 3))
    assert logged == {f"episodes/{name}": value for name, value in expected.items()}


def test_a_command_line_names_the_run_its_record_does(tmp_path):
    """A run named on the command line, each optional setting turned on by its own flags (`--env.`,
    `--teacher.`), the model's and objective's fields one flag each and an environment's argument set
    alone (`--set env.difficulty=`), is the run its record states."""
    args = ["--env.name", "flight", "--env.count", "16", "--env.fields", json.dumps(
                {"racing": RACING, "views": list(VIEWS)}),
            "--model", ACTOR_CRITIC, "--model.policy-layers", "16", "--model.value-layers", "16",
            "--objective", CONTROL_PPO, "--objective.discount", "0.9", "--objective.reads", "teacher",
            "--rollout.name", "control", "--rollout.fields", json.dumps(
                {"unroll_length": 4, "epochs": 1, "minibatches": 2}),
            "--optim.optimizer", "adam", "--optim.learning-rate", "1e-3",
            "--trainer.checkpoint-dir", str(tmp_path), "--trainer.steps", "2", "--trainer.log-every", "1",
            "--trainer.multi-host", "False", "--teacher.run", str(tmp_path / "teacher"),
            "--teacher.demonstrates", "--set", "env.difficulty=0.5"]
    run = teacher_run(tmp_path)
    assert run.env is not None
    raced = dataclasses.replace(run.env, fields={**run.env.fields, "difficulty": 0.5})
    taught = TeacherConfig(str(tmp_path / "teacher"), demonstrates=True)
    stated = dataclasses.replace(run, env=raced, teacher=taught)
    assert EnvRunConfig.cli(args) == stated


def test_stages_from_their_records_learn_from_their_saved_teacher_as_composed_in_python(tmp_path):
    """A teacher trained as an environment run (PPO over the race's privileged view), then two stages of a
    student, each built from its record: pretraining, whose prefill the teacher demonstrates with its
    policy's draws and which trains on that prefill alone (`train_ratio` inf), and DAgger from it, the
    student flying and the teacher labelling each step with its greedy action, which the student imitates.
    Each trains, every leaf of its state and of its replay bit for bit, as the stage composed in Python does
    with the teacher's own policy and weights; the DAgger stage starts from the pretraining stage's last
    weights, and each names the steps it read of the runs before it."""
    teacher = teacher_run(tmp_path).prepare().train("teacher")
    taught = teacher_policy()
    scales = {"policy": 0.0, "value": 0.0, "replay_value": 0.0}
    pretrains = {"batch": 8, "length": 8, "train_ratio": float("inf"), "prefill": 16, "capacity": 64}
    pretraining = student(tmp_path, 2, teacher=TeacherConfig(str(tmp_path / "teacher"), demonstrates=True),
                          objective=ObjectiveConfig(DREAMER_V3, {"imitation": 1.0, "scales": scales}),
                          rollout=RolloutConfig("dreamer", pretrains))
    prepared = pretraining.prepare()
    assert prepared.run.teacher is not None and prepared.run.teacher.step == 2
    pretrained = prepared.train("pretrain")
    objective = DreamerV3Objective(world_model(), imitation=1.0, scales=scales)
    race = Batched(FlightEnv(racing(), VIEWS), 8)
    rollout = DreamerRollout(objective, race, capacity=64, batch=8, length=8, train_ratio=float("inf"),
                             prefill=16, demonstrator=Teacher(taught.act, teacher.variables))
    assert_bitwise(pretrained, fitted(pretraining, objective, rollout, 2))
    assert isinstance(prepared.rollout, DreamerRollout)
    assert_bitwise(prepared.rollout.held_replay, rollout.held_replay)

    dagger = student(tmp_path, 2, teacher=TeacherConfig(str(tmp_path / "teacher")),
                     pretrained=str(tmp_path / "pretrain"), model=EnvRunConfig().model,
                     objective=ObjectiveConfig(DREAMER_V3, {"imitation": 1.0, "imitates": LABEL}))
    prepared = dagger.prepare()
    assert prepared.run.source.step == 2
    assert_bitwise(prepared.objective.variables, pretrained.variables)
    state = prepared.train("dagger")
    objective = DreamerV3Objective(world_model(), imitation=1.0, imitates=LABEL,
                                   variables=pretrained.variables)
    rollout = DreamerRollout(objective, Batched(FlightEnv(racing(), VIEWS), 8), capacity=64, batch=8,
                             length=8, train_ratio=16.0, prefill=8,
                             teacher=Teacher(taught.greedy, teacher.variables))
    assert_bitwise(state, fitted(dagger, objective, rollout, 2))
    assert isinstance(prepared.rollout, DreamerRollout)
    labelled = jax.device_get(prepared.rollout.held_replay)
    assert_bitwise(labelled, rollout.held_replay)
    filled = int(labelled.filled)
    assert not np.array_equal(labelled.held[LABEL][:filled], labelled.held["action"][:filled])


@pytest.mark.parametrize("agent", ["ppo", "dreamer"])
def test_a_resumed_run_ends_as_the_run_that_never_stopped_and_keeps_what_it_trains(tmp_path, agent):
    """A run in the race checkpointed every update, PPO or DreamerV3 with its replay in its checkpoints,
    trained to its second update and resumed under its name to its fourth, ends where the run that never
    stopped ends, every leaf bit for bit. Its environment is what it reads, so a resume at another difficulty
    trains on (a curriculum); one with another discount, which says what it trains, is refused."""
    replayed = {"batch": 8, "length": 8, "train_ratio": 16.0, "prefill": 8, "capacity": 64,
                "checkpointed_replay": True}
    run = (teacher_run(tmp_path, 4) if agent == "ppo" else
           student(tmp_path, 4, rollout=RolloutConfig("dreamer", replayed)))
    run = dataclasses.replace(run, trainer=dataclasses.replace(run.trainer, checkpoint_every=1))
    whole = run.prepare().train("whole")
    run.override({"trainer.steps": 2}).prepare().train("stopped")
    assert_bitwise(whole, run.prepare().train("stopped"))
    run.override({"trainer.steps": 6, "env.difficulty": 0.5}).prepare().train("stopped")
    trained = {"objective.discount": 0.5} if agent == "ppo" else {"objective.horizon": 100.0}
    with pytest.raises(ValueError):
        run.override({"trainer.steps": 8, **trained}).prepare().train("stopped")


def test_a_dreamer_run_resumed_in_a_later_phase_trains_on_and_at_another_horizon_is_refused(tmp_path):
    """A DreamerV3 run trained to its second update and resumed under its name in a later phase of its
    training, drawing longer windows at a lower entropy (SkyDreamer's last phases), trains on to its fourth:
    the rollout and the entropy are how it learns. Resumed again at another horizon, which says what its
    critic estimates, it is refused."""
    run = student(tmp_path, 2)
    run.prepare().train("phased")
    later = dataclasses.replace(
        run, rollout=RolloutConfig("dreamer", {"batch": 8, "length": 16, "train_ratio": 16.0, "prefill": 16,
                                               "capacity": 64}),
        objective=ObjectiveConfig(DREAMER_V3, {"entropy": 1e-5}))
    assert int(later.override({"trainer.steps": 4}).prepare().train("phased").step) == 4
    with pytest.raises(ValueError):
        later.override({"trainer.steps": 6, "objective.horizon": 100.0}).prepare().train("phased")


@pytest.fixture(scope="module")
def saved_teacher(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("teacher")
    teacher_run(root).prepare().train("teacher")
    return root / "teacher"


REFUSED = {
    "no environment": {"env": None},
    "labels imitated without a teacher": {"objective": ObjectiveConfig(DREAMER_V3, {"imitation": 1.0,
                                                                                      "imitates": LABEL})},
    "a teacher that neither labels nor demonstrates": {"teacher": True},
    "a width the environment states": {"model": ModelConfig(WORLD_MODEL,
                                                            {**TINY, "reads": "state", "action_size": 4})},
    "an auxiliary head's width": {"model": ModelConfig(WORLD_MODEL, {**TINY, "reads": "state",
                                                                     "auxiliary": [["teacher", 80, 8]]})},
    "a proprioceptive entry's width": {"model": ModelConfig(WORLD_MODEL, {**TINY, "reads": "state",
                                                                          "proprio": [["teacher", 80]]})},
    "a simulator batched again": {"env": EnvConfig("gate_race", {"renderer": None, "racing": RACING,
                                                                 "views": list(VIEWS), "count": 8}, count=8)},
    "episodes the run's own simulator flies": {
        "env": EnvConfig("gate_race", {"renderer": None, "racing": RACING, "views": list(VIEWS), "count": 8}),
        "episodes": {"episodes": EpisodesConfig(steps=4)}},
    "an unbatched environment": {"env": EnvConfig("flight", {"racing": RACING, "views": list(VIEWS)})},
    "another objective's rollout": {"rollout": RolloutConfig("control")},
    "a teacher for a rollout that takes none": {"teacher": "demonstrates", "objective": ObjectiveConfig(
        CONTROL_PPO, {"reads": ["state"]}), "model": ModelConfig(ACTOR_CRITIC),
        "rollout": RolloutConfig("control", {"unroll_length": 4, "minibatches": 2})},
    "a model beside the run it starts from": {"pretrained": True},
}
"""Each setup an environment run refuses, as the one change to a run that prepares."""


def test_a_world_models_named_entries_take_their_widths_from_the_environment(tmp_path):
    """A record naming a world model's auxiliary heads by their views and units builds the model with each
    head's width the race's view of it, one drone's flattened: the drone's state and the teacher's view."""
    model = ModelConfig(WORLD_MODEL, {**TINY, "reads": "state", "auxiliary": [["state", 8], ["teacher", 16]]})
    run = student(tmp_path, 1, model=model)
    assert run.env is not None
    race = run.env.build()
    shown = jax.eval_shape(race.reset, jax.random.key(0))[1].observation
    agent = run.agent(race)
    assert isinstance(agent, DreamerV3Objective)
    assert agent.model.auxiliary == (("state", math.prod(shown["state"].shape[1:]), 8),
                                     ("teacher", math.prod(shown["teacher"].shape[1:]), 16))


@pytest.mark.parametrize("change", list(REFUSED))
def test_an_environment_run_refuses_a_setup_it_cannot_train(tmp_path, saved_teacher, change):
    """A Dreamer agent of the drone's state prepares; changed in one thing, each setup the kind refuses is
    refused as it prepares (`REFUSED`)."""
    run = student(tmp_path, 1)
    run.prepare()
    fields = dict(REFUSED[change])
    if "teacher" in fields:
        demonstrates = fields["teacher"] == "demonstrates"
        fields["teacher"] = TeacherConfig(str(saved_teacher), demonstrates=demonstrates)
    if "pretrained" in fields:
        fields["pretrained"] = str(saved_teacher)
    with pytest.raises(ValueError):
        dataclasses.replace(run, **fields).prepare()


@pytest.mark.parametrize("name", ["playground", "dew.rl.playground:playground"])
def test_an_environment_a_function_builds_is_named_again_by_its_record(name):
    """An environment a function builds, the table's (`playground`, `Playground.load`) or any other declared
    to return one (`dew.rl.playground.playground`), which a record holds as the function's import path, is
    named by that record again, read back with its run."""
    held = EnvConfig(name, {"name": "CartpoleBalance"} | ({} if name == "playground" else {"count": 4}))
    run = EnvRunConfig(env=held)
    again = EnvRunConfig.from_dict(json.loads(json.dumps(run.to_dict())))
    # A classmethod is bound afresh at each read: the same function of the same class, equal, not identical.
    assert again.env is not None and again.env.member == held.member == envs[name]

