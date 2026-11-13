"""Environment runs built from their records (`dew.config.env.EnvRunConfig`): an agent trained in
the gate race as the same run composed in Python trains, its record read back training it again, a teacher
saved as a run labelling and demonstrating the steps its own policy would, stages chained from the run before
them, a resume ending where the run that never stopped ends, and each setup the kind refuses."""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from recording import bits
from test_env_rollout import evaluation_key

from dew.config import ObjectiveConfig, RunConfig, TrainerConfig
from dew.config.env import EnvConfig, EnvRunConfig, EpisodesConfig, RolloutConfig, TeacherConfig
from dew.nn.kernels import generation
from dew.objectives.rl.control import ActorCritic, ControlPPOObjective, ControlRollout
from dew.objectives.rl.dreamer import DreamerRollout, DreamerV3Objective, WorldModel
from dew.objectives.rl.tssm import TSSM
from dew.optim import OptimConfig
from dew.registry import ModelConfig
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
    from dew.telemetry.compiles import Compiles

    run, objective, rollout = composed(tmp_path)
    state = run.prepare().train("record")
    assert_bitwise(state, fitted(run, objective, rollout, 3))
    saved = RunConfig.load(str(tmp_path / "record"))
    with Compiles("step") as again:
        retrained = saved.prepare().train("again")
    assert [program for program in again.programs if program.kind == "traced"] == []
    assert_bitwise(state, retrained)
    if composed is dreamer:
        with Compiles("step") as changed:
            saved.override({"objective.horizon": 100.0, "trainer.steps": 1}).prepare().train("horizon")
        assert any(program.kind == "traced" for program in changed.programs)

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



def test_a_student_resumed_by_its_command_keeps_the_teacher_step_it_pinned(tmp_path):
    """A student whose teacher's step it leaves to its pinning (`TeacherConfig.step` None) pins the
    teacher's latest as it starts; once the teacher has trained on, the same command resumed reads the
    step it pinned (`EnvRunConfig.BUILT`, `RunConfig.resumed`) and trains on, where it pinned the
    teacher's new latest and its resume was refused (row 3931). A run of no teacher stays one."""
    teacher_run(tmp_path).prepare().train("teacher")
    taught = TeacherConfig(str(tmp_path / "teacher"), demonstrates=True)
    pretrains = {"batch": 8, "length": 8, "train_ratio": float("inf"), "prefill": 16, "capacity": 64}
    # Named in its trainer, as `dew train` names a run by its file.
    command = student(tmp_path, 1, teacher=taught, rollout=RolloutConfig("dreamer", pretrains),
                      objective=ObjectiveConfig(DREAMER_V3, {"imitation": 1.0}),
                      trainer=trainer(tmp_path, 1, name="taught"))
    command.prepare().train("taught")
    teacher_run(tmp_path, steps=4).prepare().train("teacher")
    again = dataclasses.replace(command, trainer=trainer(tmp_path, 2, name="taught"))
    prepared = again.prepare()
    assert prepared.run.teacher is not None and prepared.run.teacher.step == 2
    assert int(prepared.train("taught").step) == 2
    untaught = student(tmp_path, 1, trainer=trainer(tmp_path, 1, name="taught"))
    assert untaught.resumed().teacher is None


@pytest.fixture(scope="module")
def trained_teacher(tmp_path_factory):
    """A teacher trained as an environment run once for the module's tests, its saved run and its state."""
    root = tmp_path_factory.mktemp("teacher")
    return root / "teacher", teacher_run(root).prepare().train("teacher")


@pytest.fixture(scope="module")
def saved_teacher(trained_teacher) -> Path:
    return trained_teacher[0]


def test_a_pretraining_stage_from_its_record_learns_from_its_saved_teacher_as_composed_in_python(
        tmp_path, trained_teacher):
    """A teacher trained as an environment run (PPO over the race's privileged view), then a student's
    pretraining stage built from its record, whose prefill the teacher demonstrates with its policy's draws
    and which trains on that prefill alone (`train_ratio` inf): it trains, every leaf of its state and of its
    replay bit for bit, as the stage composed in Python does with the teacher's own policy and weights, and
    names the teacher step it read; a DAgger stage from it starts from its last weights."""
    saved, teacher = trained_teacher
    # A step of the stage: one update reads, writes and records all a run of more does.
    scales = {"policy": 0.0, "value": 0.0, "replay_value": 0.0}
    pretrains = {"batch": 8, "length": 8, "train_ratio": float("inf"), "prefill": 16, "capacity": 64}
    pretraining = student(tmp_path, 1, teacher=TeacherConfig(str(saved), demonstrates=True),
                          objective=ObjectiveConfig(DREAMER_V3, {"imitation": 1.0, "scales": scales}),
                          rollout=RolloutConfig("dreamer", pretrains))
    prepared = pretraining.prepare()
    assert prepared.run.teacher is not None and prepared.run.teacher.step == 2
    pretrained = prepared.train("pretrain")
    objective = DreamerV3Objective(world_model(), imitation=1.0, scales=scales)
    race = Batched(FlightEnv(racing(), VIEWS), 8)
    rollout = DreamerRollout(objective, race, capacity=64, batch=8, length=8, train_ratio=float("inf"),
                             prefill=16, demonstrator=Teacher(teacher_policy().act, teacher.variables))
    assert_bitwise(pretrained, fitted(pretraining, objective, rollout, 1))
    assert_bitwise(prepared.rollout.held_replay, rollout.held_replay)
    dagger = student(tmp_path, 1, teacher=TeacherConfig(str(saved)),
                     pretrained=str(tmp_path / "pretrain"), model=EnvRunConfig().model,
                     objective=ObjectiveConfig(DREAMER_V3, {"imitation": 1.0, "imitates": LABEL}))
    prepared = dagger.prepare()
    assert prepared.run.source.step == 1
    assert_bitwise(prepared.objective.variables, pretrained.variables)


def test_a_dagger_stage_from_its_record_labels_its_flights_with_its_saved_teachers_actions(tmp_path,
                                                                                         trained_teacher):
    """A teacher trained as an environment run, then a student's DAgger stage built from its record: the
    student flies, the teacher read from its saved run labels each step with its greedy action, and the
    student imitates the labels. It trains, every leaf of its state and of its replay bit for bit, as the
    stage composed in Python does with the teacher's own greedy policy and weights (row 4359), its labels
    the teacher's, not the actions flown. The labelling composed in Python is test_aggregation's; this
    stage's record wires it to the saved teacher."""
    saved, teacher = trained_teacher
    dagger = student(tmp_path, 1, teacher=TeacherConfig(str(saved)),
                     objective=ObjectiveConfig(DREAMER_V3, {"imitation": 1.0, "imitates": LABEL}))
    prepared = dagger.prepare()
    assert prepared.run.teacher is not None and prepared.run.teacher.step == 2
    state = prepared.train("dagger")
    objective = DreamerV3Objective(world_model(), imitation=1.0, imitates=LABEL)
    rollout = DreamerRollout(objective, Batched(FlightEnv(racing(), VIEWS), 8), capacity=64, batch=8,
                             length=8, train_ratio=16.0, prefill=8,
                             teacher=Teacher(teacher_policy().greedy, teacher.variables))
    assert_bitwise(state, fitted(dagger, objective, rollout, 1))
    labelled = jax.device_get(prepared.rollout.held_replay)
    assert_bitwise(labelled, rollout.held_replay)
    filled = int(labelled.filled)
    assert filled and not np.array_equal(labelled.held[LABEL][:filled], labelled.held["action"][:filled])


def resumable(root: Path, agent: str) -> EnvRunConfig:
    """A run of `agent`'s, PPO or DreamerV3 with its replay in its checkpoints, to its fourth update in the
    race, checkpointed every update."""
    replayed = {"batch": 8, "length": 8, "train_ratio": 16.0, "prefill": 8, "capacity": 64,
                "checkpointed_replay": True}
    run = (teacher_run(root, 4) if agent == "ppo" else
           student(root, 4, rollout=RolloutConfig("dreamer", replayed)))
    return dataclasses.replace(run, trainer=dataclasses.replace(run.trainer, checkpoint_every=1))


@pytest.mark.parametrize("agent", ["ppo", "dreamer"])
def test_a_resumed_run_ends_as_the_run_that_never_stopped(tmp_path, agent):
    """A run in the race checkpointed every update (`resumable`), trained to its second update and resumed
    under its name to its fourth, ends where the run that never stopped ends, every leaf bit for bit;
    resumed further in a harder race (and for Dreamer with longer windows and lower entropy) it trains on,
    while changing discount or horizon is refused."""
    run = resumable(tmp_path, agent)
    prepared_whole = run.prepare()
    whole = prepared_whole.train("whole")
    if agent == "dreamer":
        assert isinstance(prepared_whole.rollout, DreamerRollout)
        assert [path for path, leaf in jax.tree_util.tree_leaves_with_path(prepared_whole.rollout.saved())
                if getattr(leaf, "weak_type", False)] == []
    run.override({"trainer.steps": 2}).prepare().train("stopped")
    assert_bitwise(whole, run.prepare().train("stopped"))
    next_run = run.override({"trainer.steps": 5, "env.difficulty": 0.5})
    if agent == "dreamer":
        next_run = dataclasses.replace(
            next_run,
            rollout=RolloutConfig("dreamer", {"batch": 8, "length": 16, "train_ratio": 16.0, "prefill": 16,
                                              "capacity": 64, "checkpointed_replay": True}),
            objective=ObjectiveConfig(DREAMER_V3, {"entropy": 1e-5}))
    assert int(next_run.prepare().train("stopped").step) == 5
    refused = {"objective.discount": 0.5} if agent == "ppo" else {"objective.horizon": 100.0}
    with pytest.raises(ValueError):
        next_run.override({"trainer.steps": 6, **refused}).prepare().train("stopped")


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


