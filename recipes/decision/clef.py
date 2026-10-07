"""A decision model on a frozen decoder under LoRA with Clef's joint head, as Clef is built.

    python recipes/decision/clef.py --decontaminate eval-index.npz --trainer.checkpoint-dir runs
    python recipes/decision/clef.py --pretrained Qwen/Qwen3.5-9B \\
        --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --lora.rank 256

Qwen3.5-4B (Apache-2.0) stays frozen under rank-64 LoRA factors on every
projection, the full-attention layers' q, k, v and o, the Gated DeltaNet
layers' qkv, z and output projections, and every MLP's, while a fresh joint
schema head of Clef-flash's shape learns with them. Clef's joint layout asks
every question of an example in one row of up to 2,048 tokens, in a new
order on each read, and the head pools each question's instructions and
each option's tokens and lets the questions attend to each other. Clef
trains rank-256 factors over Qwen3.8-27B and Clef-flash over Qwen3.5-9B; the
second command is the size-matched comparison with Clef-flash, for an H100.

It reads the same mixture as the encoder recipe, decontaminated the same
way, and the same proper scoring rules. AdamW at 1e-4 warms up over 200
steps and decays on a cosine to step 6,000, with 8 rows a step and every
block recomputed in the backward pass, which fits a 40 GB A100.
"""

from dataclasses import dataclass, field
from typing import Annotated

import tyro
from encoder import RULES
from sources import Mixture
from train import DecisionRunConfig, JointShape, main

from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, TrainerConfig
from dew.lora import LoRA
from dew.training.optim import Cosine

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
               "gate_proj", "up_proj", "down_proj")
"""Every projection of a Qwen3.5 decoder layer, attention, Gated DeltaNet and MLP alike."""


@dataclass(frozen=True)
class ClefRun(DecisionRunConfig):
    """The Clef recipe's run: the module's description, every value a flag."""

    pretrained: str = "Qwen/Qwen3.5-4B"
    revision: str | None = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig("causal_transformer", {"dtype": "bfloat16"}))
    lora: Annotated[LoRA, tyro.conf.subcommand("lora")] | None = field(
        default_factory=lambda: LoRA(rank=64, modules=PROJECTIONS))
    head: JointShape | None = field(default_factory=JointShape)
    remat: str | None = "full"
    mixture: Mixture | None = field(default_factory=Mixture)
    objective: ObjectiveConfig = field(default_factory=lambda: ObjectiveConfig(
        "decision", {"loss": RULES, "label_smoothing": 0.05}))
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(
        learning_rate=1e-4, schedule=Cosine(peak=1e-4, warmup_steps=200)))
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(
        batch_size=8, steps=6_000, eval_every=500, checkpoint_every=500))
    max_len: int | None = 2048


if __name__ == "__main__":
    main(ClefRun.cli())
