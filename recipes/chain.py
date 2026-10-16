"""Chained training: pretraining, SFT, then preference or online RL, one model.

A `Recipe` is an ordered sequence of `Stage`s sharing one built decoder.
Each stage trains its own data with its own objective in its own checkpoint
directory, and every stage after the first initializes from the model's part
of the previous stage's final variables through the objective's `pretrained`
mechanism, so the chain is pretraining, then SFT, then RL with nothing
reloaded by hand. The optimizer starts fresh every stage; a stage never
resumes another stage's step count.

The existing `--pretrained` flag cannot express this: it reads a Hugging
Face decoder, not a dew run directory. The chain instead restores the dew
checkpoint it just wrote. A GRPO stage names its reward callable and a PPO
stage its environment and verifier; callables do not survive a command
line, and neither does a chain: this recipe is a Python value, built where
the reward lives.
"""

from __future__ import annotations

import dataclasses

import jax
import optax
from flax import linen as nn

from dew.config.data import ChatData, PreferenceData, PromptData, RecordsData, TextData, loader
from dew.data import Loading
from dew.data.chat import Role
from dew.data.text import tokenizer_for
from dew.objectives import DistillationObjective
from dew.objectives.base import OMITTED, Objective, Omitted, Source, Variables
from dew.objectives.lm import LMObjective
from dew.objectives.rl import (
    DPOObjective,
    EnvironmentFactory,
    EpisodeRollout,
    GRPOObjective,
    PPOObjective,
    PPORollout,
    SampledRollout,
    Verifier,
)
from dew.objectives.rl.rollout import Reward
from dew.sampling import Sampling, with_ids_of
from dew.training import Checkpoints, Layout, Rollout, Trainer, TrainState


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
    """One link: `data` trained for `steps` gradient steps.

    The data decides the loss: `TextData` trains next-token pretraining,
    `ChatData` SFT, `PreferenceData` DPO, `PromptData` GRPO, and
    `RecordsData` of `task_id` rows PPO over the stage's `episodes`, which
    no other stage takes. A `teacher` distills a pretraining or SFT stage,
    and any other stage refuses one. `beta` is the KL strength of DPO, GRPO
    and PPO, 0.1 for DPO and 0.0 otherwise when None; a pretraining or SFT
    stage refuses one. A GRPO stage refuses to run without a `reward`: a
    rollout no rule scores is steps in the dark. Its reward reads
    completions decoded by the prompts' tokenizer, and its `sampling` stops
    at that tokenizer's EOS unless it names its own. A PPO stage's
    `sampling` names the EOS that ends an action. `groups` and
    `max_new_tokens` size both online stages' sampling, and `estimator`
    GRPO's advantages. `Recipe.run` applies these rules.
    """

    name: str
    data: TextData | ChatData | PreferenceData | PromptData | RecordsData
    steps: int = 100
    beta: float | None = None
    reward: Reward | None = None
    groups: int = 4
    max_new_tokens: int = 32
    estimator: str = "group"
    seed: int = 0
    loading: Loading = dataclasses.field(default_factory=Loading)
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
    `batch` sizes every stage's batches; `layout` places every stage's
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
    batch: int = 8
    layout: Layout = dataclasses.field(default_factory=Layout)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a recipe links at least one stage")

    def run(self) -> list[TrainState]:
        for stage in self.stages:
            self._build(stage, None)
        states: list[TrainState] = []
        variables = None
        for index, stage in enumerate(self.stages):
            objective, rollout = self._build(stage, variables)
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
            states.append(trainer.fit(
                loader(stage.data, batch=self.batch, read="memory", seed=stage.seed, loading=stage.loading),
                steps=stage.steps))
            variables = objective.model_variables(states[-1].variables)
        return states

    def _build(self, stage: Stage, variables: Variables | None) -> tuple[Objective, Rollout | None]:
        """The stage's objective over the shared model, continuing `variables`
        past the first stage, with its rollout beside it for GRPO and PPO.

        This is the one place a stage's data picks its loss, so it also
        refuses the stage's settings that loss does not take or lacks
        (`Stage`'s rules): `ValueError`, or `TypeError` for data no loss
        reads.
        """
        data = stage.data
        if isinstance(data, RecordsData) != (stage.episodes is not None):
            raise ValueError(f"stage {stage.name!r}: RecordsData and episodes make a PPO stage together")
        supervised = isinstance(data, (TextData, ChatData))
        if not supervised and stage.teacher is not None:
            raise ValueError(f"stage {stage.name!r} names a teacher, which distills only pretraining or SFT")
        if supervised:
            if stage.beta is not None:
                raise ValueError(f"stage {stage.name!r} sets beta on a stage with no KL term")
            objective = _scored(self.model, data, variables)
            if stage.teacher is None:
                return objective, None
            teacher = stage.teacher
            return DistillationObjective(objective, _scored(teacher.model, data), alpha=teacher.alpha,
                                         temperature=teacher.temperature), None
        if isinstance(data, PreferenceData):
            beta = 0.1 if stage.beta is None else stage.beta
            return DPOObjective(self.model, data.seq_len - 1, beta=beta, variables=variables), None
        beta = 0.0 if stage.beta is None else stage.beta
        if isinstance(data, PromptData):
            if stage.reward is None:
                raise ValueError(f"stage {stage.name!r} samples without a reward; name one")
            objective = GRPOObjective(self.model, data.max_prompt_len + stage.max_new_tokens - 1,
                                      beta=beta, variables=variables)
            tokenizer = tokenizer_for(data.tokenizer)
            sampling = with_ids_of(stage.sampling, Sampling(eos_token_ids=tokenizer.eos_id))
            return objective, SampledRollout(objective, stage.reward, decode=tokenizer.decode,
                                             groups=stage.groups, max_new_tokens=stage.max_new_tokens,
                                             estimator=stage.estimator, sampling=sampling)
        episodes = stage.episodes
        if episodes is None:
            raise TypeError(f"stage {stage.name!r} trains {type(data).__name__}, which no loss reads")
        objective = PPOObjective(self.model, episodes.max_prompt_tokens + stage.max_new_tokens - 1,
                                 critic=episodes.critic, beta=beta, variables=variables)
        # Every collection binds the step's variables, so the policy here holds none.
        source = EpisodeRollout(objective.policy({}), episodes.environment, episodes.verifier,
                                episodes.max_prompt_tokens, stage.max_new_tokens, episodes.max_turns,
                                stage.sampling, groups=stage.groups)
        return objective, PPORollout(objective, source, gamma=episodes.gamma, lam=episodes.lam)


def _scored(model: nn.Module | Source, data: TextData | ChatData,
            variables: Variables | None | Omitted = OMITTED) -> LMObjective:
    """The next-token objective over `data`'s rows, counting only assistant tokens of chat."""
    return LMObjective(model, data.seq_len, loss_role=Role.ASSISTANT if isinstance(data, ChatData) else None,
                       variables=variables)
