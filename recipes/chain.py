"""Chained post-training: SFT then preference or online RL, one model.

A `Recipe` is an ordered sequence of `Stage`s sharing one built decoder.
Each stage trains its own data with its own objective in its own checkpoint
directory, and every stage after the first initializes from the previous
stage's final parameters through the objective's `pretrained` mechanism, so
the chain is pretraining, then SFT, then RL with nothing reloaded by hand.
The optimizer starts fresh every stage; a stage never resumes another
stage's step count.

The existing `--pretrained` flag cannot express this: it reads a Hugging
Face decoder, not a dew run directory. The chain instead restores the dew
checkpoint it just wrote. A GRPO stage samples with a `SampledRollout`, so
it names its reward callable; rewards do not survive a command line, and
neither does a chain: this recipe is a Python value, built where the reward
lives.
"""

from __future__ import annotations

import dataclasses

import jax
import optax
from flax import linen as nn

from dew.data import ChatMessages, PreferencePairs, Prompts
from dew.data.chat import Role
from dew.objectives.base import Variables
from dew.objectives.lm import LMObjective
from dew.objectives.rl import DPOObjective, GRPOObjective, SampledRollout
from dew.objectives.rl.rollout import Reward
from dew.training import Checkpoints, Layout, Rollout, Trainer, TrainState

KINDS = ("sft", "dpo", "grpo")
"""The stage losses a chain links: supervised fine-tuning, preference
optimization, online RL."""


@dataclasses.dataclass(frozen=True)
class Stage:
    """One link: `data` trained with `objective` for `steps` gradient steps.

    `beta` is the DPO or GRPO KL strength, defaulting per loss (0.1 for DPO,
    0.0 for GRPO) when None; an SFT stage refuses one. A GRPO stage names its
    `reward` and sampling sizes, and refuses to run without a reward: a
    rollout no rule scores is steps in the dark.
    """

    name: str
    data: ChatMessages | PreferencePairs | Prompts
    objective: str = "sft"
    steps: int = 100
    beta: float | None = None
    reward: Reward | None = None
    groups: int = 4
    max_new_tokens: int = 32
    sample: str = "group"

    def __post_init__(self) -> None:
        if self.objective not in KINDS:
            raise ValueError(
                f"stage {self.name!r} trains {self.objective!r}; "
                f"the chain links {list(KINDS)}")
        if self.steps < 1:
            raise ValueError(f"stage {self.name!r} runs {self.steps} steps: at least one")
        pairs = {"sft": ChatMessages, "dpo": PreferencePairs, "grpo": Prompts}
        if not isinstance(self.data, pairs[self.objective]):
            raise ValueError(
                f"stage {self.name!r} trains {self.objective} on "
                f"{type(self.data).__name__}; a {self.objective} stage reads "
                f"{pairs[self.objective].__name__}")
        if self.objective == "sft" and self.beta is not None:
            raise ValueError(
                f"stage {self.name!r} sets beta on an SFT stage, which has no KL term")
        if self.objective == "grpo" and self.reward is None:
            raise ValueError(
                f"stage {self.name!r} samples without a reward; name one")


@dataclasses.dataclass(frozen=True)
class Recipe:
    """The chain: `stages` in order, one decoder, one directory.

    `model` is the built decoder every stage trains; `optimizer` restarts
    each stage; `key` seeds the first stage and folds forward from there;
    `batch` sizes every stage's batches; `layout` places every stage's
    state. `run` returns each stage's final state, oldest first.
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
                  f"({stage.objective}, {stage.steps} steps)")
            states.append(trainer.fit(
                stage.data.load(batch=self.batch), steps=stage.steps))
            variables = states[-1].params
        return states

    def _build(self, stage: Stage,
               variables: Variables | None) -> tuple[LMObjective, Rollout | None]:
        """The stage's objective over the shared model, continuing `variables`
        past the first stage, with its rollout beside it for GRPO."""
        if stage.objective == "sft":
            assert isinstance(stage.data, ChatMessages)
            return LMObjective(self.model, stage.data.seq_len, loss_role=Role.ASSISTANT,
                               pretrained=variables), None
        if stage.objective == "dpo":
            assert isinstance(stage.data, PreferencePairs)
            beta = 0.1 if stage.beta is None else stage.beta
            return DPOObjective(self.model, stage.data.seq_len - 1, beta=beta,
                                pretrained=variables), None
        assert isinstance(stage.data, Prompts)
        assert stage.reward is not None
        beta = 0.0 if stage.beta is None else stage.beta
        objective = GRPOObjective(
            self.model, stage.data.max_prompt_len + stage.max_new_tokens - 1,
            beta=beta, pretrained=variables)
        rollout = SampledRollout(objective, stage.reward, groups=stage.groups,
                                 max_new_tokens=stage.max_new_tokens, sample=stage.sample)
        return objective, rollout
