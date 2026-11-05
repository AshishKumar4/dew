"""The environment run as one typed record: the environment an agent acts in, the agent's networks and
objective, the rollout that collects its batches there, and a teacher that labels or demonstrates its
steps (docs/design/post-training.md §14).

The environment is its class and arguments (`EnvConfig`) and the rollout its class and settings
(`RolloutConfig`), each held as its import path, the arguments the run states and the class's other
defaults as they were recorded, as a run holds its model and its objective. What the environment states,
the run derives: the agent's action width, the width a policy reads of each observation and the image a
world model sees. The rollout replaces every batch whole, so the run reads no data, and the trainer is
handed a loader of empty rows (`idle`).
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, ClassVar, Self, TypedDict

import jax
import numpy as np
import tyro

from dew.batch import Variables
from dew.checkpoints import Checkpoints
from dew.config import ObjectiveConfig, Prepared, RunConfig, SourceOptions, TrainerConfig
from dew.data.dataset import Loader, replayed
from dew.nn.sharding import abstract
from dew.objectives.base import Shown
from dew.objectives.rl.control import ControlPPOObjective
from dew.objectives.rl.dreamer import DreamerV3Objective
from dew.registry import (
    Configured,
    ModelConfig,
    Registered,
    declared_type,
    envs,
    models,
    parameters,
    read_arguments,
    record_argument,
    rollouts,
)
from dew.rl.aggregation import LABEL, Teacher
from dew.rl.env import Batched, Driven, Env, Observation, TimeLimit, view
from dew.rl.evaluation import episodes
from dew.rl.simulator import DrivenSimulator, Simulator, received
from dew.training.evaluation import Episodes
from dew.training.rollout import Rollout

type Agent = DreamerV3Objective | ControlPPOObjective
"""What an environment run trains: an agent that acts, a world-model learner's or a policy's."""

_BUILT = ("objective", "env", "replay", "teacher", "demonstrator")
"""A rollout's arguments the run builds, or no record holds (a replay), which its settings leave out."""


@dataclasses.dataclass(frozen=True)
class EnvConfig(Registered):
    """An environment's class and the arguments the run gives it, and how the run wraps it.

    `name` is an alias of `dew.registry.envs` or an import path, held as the path: a JAX `Env` class, a host
    `Simulator` class, or a function or classmethod that returns one (`Playground.load`). `fields` are its
    arguments as given, a nested value as its record (a race's `Racing`, a `GateRace`'s renderer as
    `{"class": ..., "fields": ...}`), set one at a time by `--set env.<argument>=...`; `defaults` holds
    every other argument the class defaults, as a model's are held. An `Env` of one environment is wrapped
    as `dew.rl.playground.playground` wraps one: each episode cut after `episode_length` steps of
    `action_repeat` repeats where that is set (`TimeLimit`), then `count` of them stepped together
    (`Batched`). A `Simulator` is a batch of its own, and takes neither."""

    name: str
    count: int | None = None
    episode_length: int | None = None
    action_repeat: int = 1

    _recorded: ClassVar[bool] = False

    @staticmethod
    def _member(name: str) -> type | Callable[..., Configured]:
        return envs[name]

    def build(self) -> Env | Simulator:
        """The environment the record names, wrapped as it says."""
        built = envs.build(self.name, self.arguments)
        if isinstance(built, Simulator):
            if self.count is not None or self.episode_length is not None or self.action_repeat != 1:
                raise ValueError(f"{self.name} is a Simulator, a batch of environments stepped from the host "
                                 "that cuts its own episodes; leave count, episode_length and action_repeat "
                                 "unset")
            return built
        if self.episode_length is not None:
            built = TimeLimit(built, self.episode_length, self.action_repeat)
        elif self.action_repeat != 1:
            raise ValueError("action_repeat repeats an action within a time limit's step; set episode_length")
        return built if self.count is None else Batched(built, self.count)


@dataclasses.dataclass(frozen=True)
class RolloutConfig(Registered):
    """The rollout that collects the run's batches (`dew.training.rollout.Rollout`): `name` an alias of
    `dew.registry.rollouts` (`dreamer`, `control`) or an import path, held as the path, and `fields` the
    settings the run gives it, as records: a `DreamerRollout`'s `batch`, `length`, `train_ratio`,
    `prefill`, `capacity` and `checkpointed_replay`, a `ControlRollout`'s `unroll_length`, `unrolls`,
    `epochs` and `minibatches`. `defaults` holds the class's others as recorded. The run builds the rest,
    the objective, the environment and its teacher (`_BUILT`); a `train_ratio` of `inf` trains on the
    prefill alone."""

    name: str = "dreamer"

    @staticmethod
    def _member(name: str) -> type:
        return rollouts[name]

    def build(self, objective: Agent, env: Env | Simulator, **taught: Teacher) -> Rollout:
        """The rollout of `objective` in `env`, with the teacher and demonstrator `taught` names. Refuses a
        setting the run builds, an objective of another class than the rollout's, and a teacher it takes
        none of."""
        stated = sorted(set(self.fields) & set(_BUILT))
        if stated:
            raise ValueError(f"the run builds the rollout's {stated}; leave them out of its settings")
        member = rollouts[self.name]
        declared = declared_type(member, "objective")
        if isinstance(declared, type) and not isinstance(objective, declared):
            raise ValueError(f"{self.name} collects for a {declared.__name__}, and the run trains a "
                             f"{type(objective).__name__}; name the rollout of its objective")
        untaken = sorted(set(taught) - set(parameters(member)[0]))
        if untaken:
            raise ValueError(f"{self.name} takes no {untaken}; a teacher labels and demonstrates a Dreamer "
                             "rollout's steps")
        given = {**read_arguments(member, self.arguments), "objective": objective, "env": env, **taught}
        return member(**given)


@dataclasses.dataclass(frozen=True)
class TeacherConfig:
    """A privileged teacher: the saved environment run in the directory `run`, a `ControlPPOObjective`'s,
    read at its checkpoint `step` (a step, `latest`, `best` or `best:<metric>`; None for its latest), which
    the run pins as it starts, so its record names the step it read (`EnvRunConfig.pinned`). Its policy acts
    in the run's own environment and reads there what it read in its own. It labels every step the rollout
    takes with its greedy action where the run's objective imitates labels (`imitates="label"`, DAgger),
    and where it `demonstrates`, it flies the rollout's prefill with its policy's draws (DART), the replay's
    first steps. A teacher that does neither is refused."""

    run: str
    step: int | str | None = None
    demonstrates: bool = False


@dataclasses.dataclass(frozen=True)
class EpisodesConfig:
    """A validation split scored by flying the agent (`dew.training.Episodes` over
    `dew.rl.evaluation.episodes`): its greedy policy, or with `drawn` the policy's draws as it acts in
    training (`act`), flown `steps` steps from fresh episodes of `env`, the run's own where None, each
    environment's first episode scored, logged as `<split>/return` and `<split>/steps`, and for each
    observation view in `reached` its mean at each first episode's last step (a race's gates `passed`),
    `<split>/<view>`. A JAX `Env` is flown in one program; a `Simulator` from the host, and only one the
    record names for the split: the run's own simulator holds the rollout's environments mid episode,
    which a flight would move."""

    steps: int
    env: EnvConfig | None = None
    reached: tuple[str, ...] = ()
    drawn: bool = False

    def split(self, objective: Agent, env: Env | Simulator) -> Episodes:
        """The split of `objective`'s flights, in `env` where the record names no environment of its own."""
        flown = env if self.env is None else self.env.build()
        if isinstance(flown, Simulator) and self.env is None:
            raise ValueError("a run's simulator holds its rollout's environments mid episode, which a flight "
                             "would move; name a simulator of the split's own (env)")
        policy = objective.act if self.drawn else objective.greedy
        if isinstance(objective, DreamerV3Objective):
            driven = DrivenSimulator(flown) if isinstance(flown, Simulator) else Driven(flown)
            score = episodes(policy, driven, self.steps, memory=objective.memory, reached=self.reached,
                             models=objective.towers())
        else:
            score = episodes(policy, flown, self.steps, reached=self.reached, models=objective.towers())
        reached = {view: Shown(better="higher") for view in self.reached}
        return Episodes(score, {"return": Shown(better="higher"), "steps": Shown(), **reached})


@dataclasses.dataclass(frozen=True)
class EnvRunConfig(RunConfig):
    """A run that trains an agent in an environment: the shared run fields, the environment (`env`), the
    agent's networks (`model`: a `WorldModel`, its `dynamics` a nested `TSSM` record where it has one, or an
    `ActorCritic`, each by its import path) and objective (`objective`: a `DreamerV3Objective` or a
    `ControlPPOObjective`, by its import path, its
    imitation among its arguments), the rollout's settings (`rollout`), an optional teacher (`teacher`), the
    saved run the agent starts from (`pretrained`) and the episodes each validation flies (`episodes`).

    The environment is what the run reads, as data is a run's, and the rollout how it collects and replays
    it, so a run resumed under its name may change them (`RESUMABLE`): a curriculum resumes the run at a
    new difficulty, a later phase of training at longer windows (SkyDreamer's), the rollout owing its
    updates from there at its own pace. The run has no epochs, so its
    trainer validates and checkpoints at the cadences it names, none by default, and at its end."""

    _FLAG_SELECTED: ClassVar[Mapping[str, str]] = {**RunConfig._FLAG_SELECTED, "env": "env",
                                                   "teacher": "teacher"}
    RESUMABLE: ClassVar[tuple[str, ...]] = (*RunConfig.RESUMABLE, "env", "episodes", "rollout")
    BUILT: ClassVar[tuple[str, ...]] = (*RunConfig.BUILT, "teacher.step")

    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("dew.objectives.rl.dreamer:WorldModel"))
    """The agent's networks. The run derives from its environment each width the class declares: its
    `action_size`, and a world model's `observation_size` and `image` from the view it `reads`; a record
    that states one is refused. A run that starts from a saved run trains that run's model, and names none."""
    objective: ObjectiveConfig = dataclasses.field(
        default_factory=lambda: ObjectiveConfig("dew.objectives.rl.dreamer:DreamerV3Objective"))
    """The agent's objective and its arguments (`--objective.imitation 1 --objective.imitates label`); a
    `ControlPPOObjective`'s `observation_size` is derived, the width of the views it `reads`."""
    trainer: TrainerConfig = dataclasses.field(
        default_factory=lambda: TrainerConfig(eval_every=None, checkpoint_every=None))
    env: Annotated[EnvConfig, tyro.conf.subcommand("env")] | None = None
    """The environment the agent acts in (`EnvConfig`): `--env.name flight --env.count 4096 --env.fields
    '{"racing": {"hz": 60}}'`, an argument at a time `--set env.difficulty=0.5`."""
    rollout: RolloutConfig = dataclasses.field(default_factory=RolloutConfig)
    """The rollout and its settings (`RolloutConfig`): `--rollout.name control --rollout.fields
    '{"unroll_length": 64}'`, a setting at a time `--set rollout.train_ratio=128`."""
    teacher: Annotated[TeacherConfig, tyro.conf.subcommand("teacher")] | None = None
    """A privileged teacher's saved run (`TeacherConfig`), turned on by its flags: `--teacher.run runs/teacher
    --teacher.demonstrates`."""
    pretrained: str | None = None
    """A saved environment run's directory the agent starts from, its model and its weights at the step
    `source.step` selects (the latest unless it names one), which the run pins as it starts; its live
    weights, or its average with `source.ema`. A stage of a chain starts from the stage before it."""
    episodes: Annotated[Mapping[str, EpisodesConfig], record_argument(Mapping[str, EpisodesConfig])] = (
        dataclasses.field(default_factory=dict))
    """The episodes each validation flies, a split of each (`EpisodesConfig`) by its name, one JSON value
    on the command line: `--episodes '{"episodes": {"steps": 1000}}'`; at `trainer.eval_every`, or a
    split's own cadence in `val_every`."""

    def build_run(self) -> Prepared:
        """The agent the record names, acting in its environment through its rollout, on the loader the
        rollout replaces (`idle`), validated by its episodes where it names them."""
        if self.data is not None or self.validation is not None or self.lora is not None:
            raise ValueError("an environment run trains its whole agent on the batches its rollout collects; "
                             "leave data, validation and lora unset, and score it by its episodes")
        if self.env is None:
            raise ValueError("EnvRunConfig names no environment; name one, as --env.name flight")
        env = self.env.build()
        if not isinstance(env, Simulator) and not jax.eval_shape(env.reset, jax.random.key(0))[1].reward.ndim:
            raise ValueError(f"a rollout steps a batch of environments, and {self.env.name} is one; set "
                             "env.count")
        objective = self.agent(env, self.started())
        rollout = self.rollout.build(objective, env, **self.taught(objective, env))
        validation = {name: split.split(objective, env) for name, split in self.episodes.items()} or None
        return Prepared(self, objective, idle(), rollout=rollout, validation=validation)

    def trained_model(self) -> ModelConfig:
        """The model the run trains: its own, or where it starts from a saved run that run's."""
        return self.model if self.pretrained is None else EnvRunConfig.load(self.pretrained).trained_model()

    def agent(self, env: Env | Simulator, variables: Variables | None = None) -> Agent:
        """The run's objective over its model (`trained_model`) acting in `env`, from `variables`, a fresh
        draw where None: each width `env` states derived for both, and refused where the record states it.
        Variables that do not fit the agent are refused."""
        model = self.trained_model()
        observation = _observed(env)
        model = dataclasses.replace(model, fields={**model.fields, **_filled(observation, model.fields)})
        declared = {field.name for field in dataclasses.fields(models[model.name])}
        derived: dict[str, Configured] = {"action_size": env.action_size}
        if declared & {"observation_size", "image"}:
            # A world model's encoder reads a vector of `observation_size`, or an image of its shape.
            seen = _seen(observation, _reads({**model.defaults, **model.fields}))
            derived |= {"observation_size": seen[0] if len(seen) == 1 else 0,
                        "image": seen if len(seen) == 3 else None}
        derived = {name: value for name, value in derived.items() if name in declared}
        stated = sorted(set(model.fields) & set(derived))
        if stated:
            raise ValueError(f"the run derives the model's {stated} from its environment; leave them out")
        built = dataclasses.replace(self, model=model).numerics_fields()
        widths = {}
        if "observation_size" in parameters(self.objective.member)[0]:
            read = _seen(observation, _reads({**self.objective.defaults, **self.objective.fields}))
            if len(read) != 1:
                raise ValueError(f"{self.objective.name} reads a flat observation, and its views are {read}")
            widths["observation_size"] = read[0]
        given = {} if variables is None else {"variables": variables}
        objective = self.objective.build(model=model.build(**built, **derived), **widths, **given)
        if not isinstance(objective, DreamerV3Objective | ControlPPOObjective):
            raise ValueError(f"an environment run trains an agent that acts in it, a DreamerV3Objective or a "
                             f"ControlPPOObjective, not a {type(objective).__name__}")
        if variables is not None:
            fresh = abstract(lambda key: objective.fresh_variables(key, None), jax.random.key(0))
            if jax.tree.structure(fresh) != jax.tree.structure(variables) or any(
                    np.shape(one) != np.shape(other)
                    for one, other in zip(jax.tree.leaves(fresh), jax.tree.leaves(variables), strict=True)):
                raise ValueError(f"the saved weights do not fit the {type(objective).__name__} this run "
                                 "builds in its environment: it reads other views, or other widths of them")
        return objective

    def started(self) -> Variables | None:
        """The weights the agent starts from: those of the run `pretrained` names, at the step `source.step`
        selects, live unless `source.ema`; None where it names none, its `source` then refused
        (`built_fresh`). A model the run names beside it, or a source option other than the step and the
        average, is refused."""
        if self.pretrained is None:
            self.built_fresh()
            return None
        if dataclasses.replace(self.source, step=None, ema=None) != SourceOptions():
            raise ValueError("a run that starts from a saved environment run reads its step and its average "
                             "alone (--source.step, --source.ema)")
        if self.model != type(self)().model:
            raise ValueError(f"{self.pretrained} decides the model the run trains; name none")
        with Checkpoints(self.pretrained) as checkpoints:
            return checkpoints.variables(step=self.source.step, ema=bool(self.source.ema))

    def taught(self, objective: Agent, env: Env | Simulator) -> dict[str, Teacher]:
        """The rollout's teacher and demonstrator (`TeacherConfig`), each the teacher run's policy acting in
        `env`: its greedy action labelling each step where `objective` imitates labels, and its draws flying
        the prefill where it demonstrates. An objective imitating labels with no teacher, a teacher that
        neither labels nor demonstrates, and a teacher run of another objective than PPO's are refused."""
        labels = isinstance(objective, DreamerV3Objective) and objective.imitates == LABEL
        held = self.teacher
        if held is None:
            if labels:
                raise ValueError("the objective imitates a teacher's labels, and the run names no teacher; "
                                 "name one (--teacher.run DIR)")
            return {}
        if not labels and not held.demonstrates:
            raise ValueError("a teacher labels the steps of an objective that imitates its labels "
                             f"(imitates={LABEL!r}) or demonstrates the prefill (teacher.demonstrates), and "
                             "this run's does neither")
        saved = EnvRunConfig.load(held.run)
        with Checkpoints(held.run) as checkpoints:
            variables = checkpoints.variables(step=held.step, ema=False)
        policy = saved.agent(env, variables)
        if not isinstance(policy, ControlPPOObjective):
            raise ValueError(f"{held.run} trained a {type(policy).__name__}; a teacher is a PPO policy, "
                             "which acts from the environment's observation alone")
        taught = {"teacher": Teacher(policy.greedy, variables)} if labels else {}
        return taught | ({"demonstrator": Teacher(policy.act, variables)} if held.demonstrates else {})

    def pinned(self) -> Self:
        """This run with its sources pinned (`RunConfig.pinned`), and its teacher's step and the step of the
        run it starts from named as each resolves now (`Checkpoints.pinned`: the latest, or the best), so
        its record names the steps it read and a resume reads them again: a resumed run that leaves its
        teacher's step to this takes the saved run's first (`BUILT`, `RunConfig.resumed`, row 3931)."""
        run = super().pinned()
        teacher = run.teacher
        if teacher is not None and not isinstance(teacher.step, int):
            run = dataclasses.replace(run, teacher=dataclasses.replace(
                teacher, step=_pinned(teacher.run, teacher.step)))
        if run.pretrained is not None and not isinstance(run.source.step, int):
            run = dataclasses.replace(run, source=dataclasses.replace(
                run.source, step=_pinned(run.pretrained, run.source.step)))
        return run


def _pinned(directory: str, step: int | str | None) -> int:
    """The step of the saved run in `directory` that `step` selects now (`Checkpoints.pinned`)."""
    with Checkpoints(directory) as checkpoints:
        return checkpoints.pinned(step)


def _observed(env: Env | Simulator) -> Observation:
    """Every environment's observation as an episode of `env` starts, its arrays' shapes the facts read of
    it: an `Env`'s traced, a `Simulator`'s reset, which its rollout resets again as it begins."""
    if isinstance(env, Simulator):
        return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), received(env.reset(0)))
    return jax.eval_shape(env.reset, jax.random.key(0))[1].observation


class Filled(TypedDict, total=False):
    """A world model's proprioceptive entries and auxiliary heads with their widths (`_filled`)."""

    proprio: tuple[tuple[str, int], ...]
    auxiliary: tuple[tuple[str, int, int], ...]


def _filled(observation: Observation, fields: Mapping[str, object]) -> Filled:
    """A world model's proprioceptive entries (`proprio`, by name) and auxiliary heads (`auxiliary`, each its
    entry's name and its units) as its record names them, each with the width of the environment's view of
    it, one environment's flattened; a record that states a width is refused, the environment declaring
    it."""
    def width(name: str) -> int:
        return int(np.prod(np.shape(view(observation, name))[1:]))

    filled: Filled = {}
    proprio = fields.get("proprio")
    if proprio is not None:
        if not isinstance(proprio, Sequence) or not all(isinstance(entry, str) for entry in proprio):
            raise ValueError(f"a record names a world model's proprioceptive entries, whose widths the "
                             f"environment declares: {proprio}")
        filled["proprio"] = tuple((name, width(name)) for name in proprio)
    auxiliary = fields.get("auxiliary")
    if auxiliary is not None:
        heads = list(auxiliary) if isinstance(auxiliary, Sequence) else [auxiliary]
        if not all(isinstance(head, Sequence) and not isinstance(head, str) and len(head) == 2
                   for head in heads):
            raise ValueError(f"a record names each auxiliary head's entry and its units, the environment "
                             f"declaring its width: {auxiliary}")
        filled["auxiliary"] = tuple((str(head[0]), width(str(head[0])), int(head[1])) for head in heads
                                    if isinstance(head, Sequence))
    return filled


def _reads(arguments: Mapping[str, object]) -> tuple[str, ...]:
    """The views a reader's recorded `arguments` name it reading (`reads`): a world model's one, a
    policy's several, or none, where it reads the observation whole."""
    reads = arguments.get("reads")
    if reads is None or isinstance(reads, str):
        return () if reads is None else (reads,)
    if not isinstance(reads, Sequence):
        raise TypeError(f"reads names views, not {reads!r}")
    return tuple(str(name) for name in reads)


def _seen(observation: Observation, names: tuple[str, ...]) -> tuple[int, ...]:
    """The shape a reader of the views `names` sees of each environment's `observation`: the one view a
    world model reads, the views a policy reads joined along their last axis, or the observation whole
    where it names none."""
    shapes = [tuple(np.shape(view(observation, name))[1:]) for name in names] or [
        tuple(np.shape(view(observation))[1:])]
    return shapes[0] if len(shapes) == 1 else (sum(shape[-1] for shape in shapes),)


def idle() -> Loader:
    """The loader of a run whose rollout makes every batch whole (§14): a batch of no records, one value for
    the step, which any mesh places whole, the same each time it opens (`replayed`), so a resume reads on;
    with no record count, so no cadence is an epoch of it. A run composed in Python passes it to
    `Trainer.fit` as the run kind does."""
    return Loader(train=replayed(lambda partition: itertools.repeat({"step": np.zeros((), np.float32)})),
                  val=None, records=None, batch=1)


__all__ = ["EnvConfig", "EnvRunConfig", "EpisodesConfig", "RolloutConfig", "TeacherConfig", "idle"]
