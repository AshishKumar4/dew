"""A recipe chains pretraining, SFT, preference and online RL stages over one model.

A `Recipe` is an ordered sequence of `Stage`s sharing one built decoder.
Each stage trains its own data with its own objective in its own checkpoint
directory, and every stage after the first starts from the model's part of
the previous stage's final variables, held as the starting tree of the
objective that trains the model. So the chain is pretraining, then SFT,
then RL with nothing reloaded by hand. The optimizer starts fresh every
stage; a stage never resumes another stage's step count.

The existing `--pretrained` flag cannot express this: it reads a Hugging
Face decoder, not a dew run directory. The chain instead hands each stage
the tree the previous one trained, in memory. A GRPO stage names its reward callable and a PPO
stage its environment and verifier; callables do not survive a command
line, and neither does a chain: this recipe is a Python value, built where
the reward lives.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

import jax
import optax
from flax import linen as nn

from dew.batch import Variables
from dew.data.chat import Role
from dew.data.pipeline import Dataset
from dew.data.text import tokenizer_for
from dew.objectives import DistillationObjective
from dew.objectives.base import OMITTED, Objective, Omitted, Source
from dew.objectives.lm import LMObjective
from dew.objectives.rl import (
    DPOObjective,
    EnvironmentFactory,
    EpisodeCollector,
    GRPOObjective,
    PPOObjective,
    PPORollout,
    RolloutScheduler,
    Verifier,
)
from dew.objectives.rl.rollout import Reward
from dew.sampling import Sampling, with_ids_of
from dew.state import TrainState
from dew.training import Checkpoints, Layout, Rollout, Trainer


@dataclasses.dataclass(frozen=True)
class Teacher:
    """A frozen model that a pretraining or SFT stage distills into the chain's decoder.

    `model` is a loaded model with its weights, such as `Pretrained.load`'s
    or `Pretrained.from_run`'s, over the decoder's vocabulary. `alpha` and
    `temperature` are `DistillationObjective`'s, each a constant or a
    schedule of the step: alpha weights the KL against the stage's own
    loss, and the temperature softens both distributions.
    """

    model: Source
    alpha: float | optax.Schedule
    temperature: float | optax.Schedule

    def __post_init__(self) -> None:
        if not isinstance(self.model, Source):
            raise TypeError("a teacher is a loaded model with its weights, such as Pretrained.load's")


@dataclasses.dataclass(frozen=True)
class Episodes:
    """The environment a PPO stage acts in and the critic it trains beside the decoder.

    Each `task_id` row of the stage's data opens the stage's `groups`
    episodes from `environment`, and `verifier` scores each one. A call's
    context holds at most `max_prompt_tokens` ids and an episode takes at
    most `max_turns` actions. `critic` returns one value per position, as
    `ValueHead` over a decoder does. `gamma` and `lam` are the GAE discount
    and lambda.
    """

    environment: EnvironmentFactory
    verifier: Verifier
    critic: nn.Module
    max_prompt_tokens: int
    max_turns: int
    gamma: float
    lam: float


@dataclasses.dataclass(frozen=True)
class Stage:
    """One link: `data` trained by `loss` for `steps` gradient steps.

    `data` is a chain ending in its loss's preset (`dew.data.presets`), batched by its own batch stage:
    "pretrain" trains next-token pretraining on `presets.lm`, "sft" on `presets.sft` (assistant tokens
    alone), "dpo" on `presets.preferences`, "grpo" on `presets.prompts`, and "ppo" on rows of
    `task_id`s over the stage's `episodes`, which no other stage takes; `Recipe.run` refuses data that
    does not carry its loss's fields (`Dataset.facts`). A `teacher` distills a pretraining or SFT
    stage, and any other stage refuses one. `beta` is the KL strength of DPO, GRPO and PPO, 0.1 for
    DPO and 0.0 otherwise when None; a pretraining or SFT stage refuses one. A GRPO stage refuses to
    run without a `reward`: a rollout no rule scores is steps in the dark. Its reward reads completions
    decoded by the prompts' tokenizer, and its `sampling` stops at that tokenizer's EOS unless it names
    its own. A PPO stage's `sampling` names the EOS that ends an action. `groups` and
    `max_new_tokens` size both online stages' sampling, and `estimator` GRPO's advantages.
    `Recipe.run` applies these rules.
    """

    name: str
    loss: Literal["pretrain", "sft", "dpo", "grpo", "ppo"]
    data: Dataset
    steps: int = 100
    beta: float | None = None
    reward: Reward | None = None
    groups: int = 4
    max_new_tokens: int = 32
    estimator: str = "group"
    sampling: Sampling = dataclasses.field(default_factory=Sampling)
    episodes: Episodes | None = None
    teacher: Teacher | None = None

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"stage {self.name!r} runs {self.steps} steps: at least one")


@dataclasses.dataclass(frozen=True)
class Recipe:
    """The chain: `stages` in order, one decoder, one directory.

    `model` is the built decoder every stage trains; `optimizer` restarts
    each stage; `key` seeds the first stage and folds forward from there;
    each stage's data batches itself; `layout` places every stage's
    state. `run` builds every stage's objective and rollout before the
    first stage trains, so a stage whose settings its data or objective
    refuses fails before anything trains, and returns each stage's final
    state, oldest first.
    """

    model: nn.Module
    optimizer: optax.GradientTransformation
    key: jax.Array
    stages: tuple[Stage, ...]
    directory: str
    layout: Layout = dataclasses.field(default_factory=Layout)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a recipe links at least one stage")

    def run(self) -> list[TrainState]:
        built = [self._build(stage) for stage in self.stages]
        states: list[TrainState] = []
        variables = None
        for index, (stage, (objective, rollout, start)) in enumerate(zip(self.stages, built, strict=True)):
            start.variables = variables
            trainer = Trainer(
                objective,
                self.optimizer,
                key=jax.random.fold_in(self.key, index),
                layout=self.layout,
                checkpoints=Checkpoints(f"{self.directory}/{stage.name}"),
                **({"rollout": rollout} if rollout is not None else {}),
            )
            print(f"Stage {index + 1}/{len(self.stages)}: {stage.name} "
                  f"({type(objective).__name__}, {stage.steps} steps)")
            states.append(trainer.fit(stage.data, steps=stage.steps))
            variables = objective.model_variables(states[-1].variables)
        return states

    def _build(self, stage: Stage) -> tuple[Objective, Rollout | None, Objective]:
        """The stage's objective over the shared model, its rollout for GRPO
        and PPO, and the objective whose `variables` the stage starts from:
        a distillation's student, PPO's actor, or the objective itself.

        This is the one place a stage's data picks its loss, so it also
        refuses the stage's settings that loss does not take or lacks
        (`Stage`'s rules): `ValueError`, or `TypeError` for data no loss
        reads.
        """
        # A chain whose store states no fields (records in memory, say) is read as its loss reads it,
        # which refuses a missing field by name; one that states them is checked here.
        facts = stage.data.facts()
        stated = facts.get("fields")
        fields = None if stated is None else _stated(facts, "fields", Mapping)
        # A conversation's rows carry their roles, whose assistant turns alone SFT scores; a preference
        # pair's are as `preferences.PreferencePair` lays them out.
        needed = {"pretrain": ("text",), "sft": ("text", "text_roles"),
                  "dpo": ("input_ids", "completion_mask"), "grpo": ("prompt",),
                  "ppo": ("task_id",)}[stage.loss]
        if fields is not None and (missing := [name for name in needed if name not in fields]):
            raise ValueError(f"stage {stage.name!r} trains {stage.loss}, whose data carries "
                             f"{', '.join(needed)}; this chain carries no {', '.join(missing)}")
        if (stage.loss == "ppo") != (stage.episodes is not None):
            raise ValueError(f"stage {stage.name!r}: a PPO stage and episodes go together")
        supervised = stage.loss in ("pretrain", "sft")
        if not supervised and stage.teacher is not None:
            raise ValueError(f"stage {stage.name!r} names a teacher, which distills only pretraining or SFT")
        if supervised:
            if stage.beta is not None:
                raise ValueError(f"stage {stage.name!r} sets beta on a stage with no KL term")
            student = _scored(self.model, stage, facts, None)
            if stage.teacher is None:
                return student, None, student
            teacher = stage.teacher
            return DistillationObjective(student, _scored(teacher.model, stage, facts), alpha=teacher.alpha,
                                         temperature=teacher.temperature), None, student
        if stage.loss == "dpo":
            beta = 0.1 if stage.beta is None else stage.beta
            objective = DPOObjective(self.model, _stated(facts, "seq_len", int) - 1, beta=beta,
                                     variables=None)
            return objective, None, objective
        beta = 0.0 if stage.beta is None else stage.beta
        if stage.loss == "grpo":
            if stage.reward is None:
                raise ValueError(f"stage {stage.name!r} samples without a reward; name one")
            prompt = _last_dimension(_stated(_stated(facts, "fields", Mapping), "prompt", Sequence))
            objective = GRPOObjective(self.model, prompt + stage.max_new_tokens - 1, beta=beta,
                                      variables=None)
            tokenizer = tokenizer_for(_stated(facts, "tokenizer", str))
            sampling = with_ids_of(stage.sampling, Sampling(eos_token_ids=tokenizer.eos_id))
            return objective, RolloutScheduler.sampled(
                objective, stage.reward, decode=tokenizer.decode, groups=stage.groups,
                max_new_tokens=stage.max_new_tokens, estimator=stage.estimator, sampling=sampling), objective
        episodes = stage.episodes
        assert episodes is not None, "a PPO stage holds its episodes, as checked above"
        objective = PPOObjective(self.model, episodes.max_prompt_tokens + stage.max_new_tokens - 1,
                                 critic=episodes.critic, beta=beta, variables=None)
        # Every collection binds the step's variables, so the policy here holds none.
        collector = EpisodeCollector(objective.policy({}), episodes.environment, episodes.verifier,
                                     episodes.max_prompt_tokens, stage.max_new_tokens, episodes.max_turns,
                                     stage.sampling)
        rollout = PPORollout(objective, collector, groups=stage.groups, gamma=episodes.gamma,
                             lam=episodes.lam)
        return objective, rollout, objective.actor


def _stated[Value](facts: Mapping[str, object], name: str, kind: type[Value]) -> Value:
    """The fact `name` the chain states (`Dataset.facts`), refused where it is not a `kind`."""
    value = facts.get(name)
    if not isinstance(value, kind):
        raise ValueError(f"the chain states its {name} as {value!r}, not a {kind.__name__}")
    return value


def _last_dimension(shape: Sequence[object]) -> int:
    """The last dimension of a stated field shape (a list or a tuple), refused unless it is a whole number."""
    length = shape[-1] if shape else None
    if not isinstance(length, int) or isinstance(length, bool):
        raise ValueError(f"the chain states the prompt's shape as {shape!r}, which ends in no length")
    return length


def _scored(model: nn.Module | Source, stage: Stage, facts: Mapping[str, object],
            variables: Variables | None | Omitted = OMITTED) -> LMObjective:
    """The next-token objective over the stage's token windows, counting only assistant tokens of SFT."""
    return LMObjective(model, _stated(facts, "seq_len", int),
                       loss_role=Role.ASSISTANT if stage.loss == "sft" else None, variables=variables)
