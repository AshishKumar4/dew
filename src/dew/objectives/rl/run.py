"""What an environment run builds of its environment: the environment its record names, wrapped as it says,
the widths an agent reads of its observation, the teacher that labels or demonstrates its steps, and the
flights a validation scores (docs/design/post-training.md §14).

The run's record is `dew.config.env.EnvRunConfig`, a declaration; the reinforcement-learning packages build
from it here, so the configuration layer names an environment by its registered name and fields and never
imports one (tools/lint_layers.py, row 4267).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypedDict

import jax
import numpy as np

from dew.batch import Variables
from dew.objectives.rl.control import ControlPPOObjective
from dew.objectives.rl.dreamer import DreamerV3Objective
from dew.registry import envs
from dew.rl.aggregation import LABEL, Teacher
from dew.rl.env import Batched, Driven, Env, Observation, TimeLimit, view
from dew.rl.evaluation import episodes
from dew.rl.simulator import DrivenSimulator, Simulator, received

type Agent = DreamerV3Objective | ControlPPOObjective
"""What an environment run trains: an agent that acts, a world-model learner's or a policy's."""

type Environment = Env | Simulator
"""Where an environment run's agent acts: a JAX `Env` batch, or a host `Simulator`."""


def built(name: str, arguments: Mapping[str, object], *, count: int | None, episode_length: int | None,
          action_repeat: int) -> Environment:
    """The environment `name` (`dew.registry.envs`) of `arguments`, wrapped as its run's record says
    (`dew.config.env.EnvConfig`): an `Env` of one environment cut after `episode_length` steps of
    `action_repeat` repeats where that is set (`TimeLimit`), then `count` of them stepped together
    (`Batched`); a `Simulator` is a batch of its own, and takes neither."""
    made = envs.build(name, arguments)
    if isinstance(made, Simulator):
        if count is not None or episode_length is not None or action_repeat != 1:
            raise ValueError(f"{name} is a Simulator, a batch of environments stepped from the host that "
                             "cuts its own episodes; leave count, episode_length and action_repeat unset")
        return made
    if episode_length is not None:
        made = TimeLimit(made, episode_length, action_repeat)
    elif action_repeat != 1:
        raise ValueError("action_repeat repeats an action within a time limit's step; set episode_length")
    return made if count is None else Batched(made, count)


def batched(env: Environment) -> bool:
    """Whether `env` steps a batch of environments, as a rollout does: a `Simulator`, or an `Env` whose
    reward has a batch axis."""
    return isinstance(env, Simulator) or bool(jax.eval_shape(env.reset, jax.random.key(0))[1].reward.ndim)


def observed(env: Environment) -> Observation:
    """Every environment's observation as an episode of `env` starts, its arrays' shapes the facts read of
    it: an `Env`'s traced, a `Simulator`'s reset, which its rollout resets again as it begins."""
    if isinstance(env, Simulator):
        return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), received(env.reset(0)))
    return jax.eval_shape(env.reset, jax.random.key(0))[1].observation


class Filled(TypedDict, total=False):
    """A world model's proprioceptive entries and auxiliary heads with their widths (`filled`)."""

    proprio: tuple[tuple[str, int], ...]
    auxiliary: tuple[tuple[str, int, int], ...]


def filled(observation: Observation, fields: Mapping[str, object]) -> Filled:
    """A world model's proprioceptive entries (`proprio`, by name) and auxiliary heads (`auxiliary`, each its
    entry's name and its units) as its record names them, each with the width of the environment's view of
    it, one environment's flattened; a record that states a width is refused, the environment declaring
    it."""
    def width(name: str) -> int:
        return int(np.prod(np.shape(view(observation, name))[1:]))

    held: Filled = {}
    proprio = fields.get("proprio")
    if proprio is not None:
        if not isinstance(proprio, Sequence) or not all(isinstance(entry, str) for entry in proprio):
            raise ValueError(f"a record names a world model's proprioceptive entries, whose widths the "
                             f"environment declares: {proprio}")
        held["proprio"] = tuple((name, width(name)) for name in proprio)
    auxiliary = fields.get("auxiliary")
    if auxiliary is not None:
        heads = list(auxiliary) if isinstance(auxiliary, Sequence) else [auxiliary]
        if not all(isinstance(head, Sequence) and not isinstance(head, str) and len(head) == 2
                   for head in heads):
            raise ValueError(f"a record names each auxiliary head's entry and its units, the environment "
                             f"declaring its width: {auxiliary}")
        held["auxiliary"] = tuple((str(head[0]), width(str(head[0])), int(head[1])) for head in heads
                                  if isinstance(head, Sequence))
    return held


def reads(arguments: Mapping[str, object]) -> tuple[str, ...]:
    """The views a reader's recorded `arguments` name it reading (`reads`): a world model's one, a
    policy's several, or none, where it reads the observation whole."""
    named = arguments.get("reads")
    if named is None or isinstance(named, str):
        return () if named is None else (named,)
    if not isinstance(named, Sequence):
        raise TypeError(f"reads names views, not {named!r}")
    return tuple(str(name) for name in named)


def seen(observation: Observation, names: tuple[str, ...]) -> tuple[int, ...]:
    """The shape a reader of the views `names` sees of each environment's `observation`: the one view a
    world model reads, the views a policy reads joined along their last axis, or the observation whole
    where it names none."""
    shapes = [tuple(np.shape(view(observation, name))[1:]) for name in names] or [
        tuple(np.shape(view(observation))[1:])]
    return shapes[0] if len(shapes) == 1 else (sum(shape[-1] for shape in shapes),)


def imitates_labels(objective: Agent) -> bool:
    """Whether `objective` imitates a teacher's labels (`imitates="label"`, DAgger)."""
    return isinstance(objective, DreamerV3Objective) and objective.imitates == LABEL


class Taught(TypedDict, total=False):
    """A rollout's teacher, which labels each step, and demonstrator, which flies its prefill (`taught`)."""

    teacher: Teacher
    demonstrator: Teacher


def taught(policy: Agent, variables: Variables, *, labels: bool, demonstrates: bool) -> Taught:
    """A rollout's teacher and demonstrator, the saved teacher run's `policy` over its `variables`: its greedy
    action labelling each step where the run's objective imitates `labels`, and its draws flying the prefill
    where it `demonstrates`. A teacher run of another objective than PPO's is refused."""
    if not isinstance(policy, ControlPPOObjective):
        raise ValueError(f"a teacher is a PPO policy, which acts from the environment's observation alone, "
                         f"not a {type(policy).__name__}")
    held: Taught = {}
    if labels:
        held["teacher"] = Teacher(policy.greedy, variables)
    if demonstrates:
        held["demonstrator"] = Teacher(policy.act, variables)
    return held


def flights(objective: Agent, env: Environment, steps: int, *, reached: tuple[str, ...], drawn: bool,
            own: bool) -> Callable[[Variables, jax.Array], dict[str, float]]:
    """The scoring of `objective`'s flights in `env` (`dew.rl.evaluation.episodes`): its greedy policy, or
    with `drawn` the policy's draws, flown `steps` steps from fresh episodes. A `Simulator` the run's own
    (`own`) is refused: it holds the rollout's environments mid episode, which a flight would move."""
    if isinstance(env, Simulator) and own:
        raise ValueError("a run's simulator holds its rollout's environments mid episode, which a flight "
                         "would move; name a simulator of the split's own (env)")
    policy = objective.act if drawn else objective.greedy
    if isinstance(objective, DreamerV3Objective):
        driven = DrivenSimulator(env) if isinstance(env, Simulator) else Driven(env)
        return episodes(policy, driven, steps, memory=objective.memory, reached=reached,
                        models=objective.towers())
    return episodes(policy, env, steps, reached=reached, models=objective.towers())

