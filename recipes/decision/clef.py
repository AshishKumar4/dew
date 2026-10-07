"""A decision model on a frozen decoder under LoRA with Clef's joint head, as Clef is built.

    python recipes/decision/sources.py --out data/mixture --decontaminate eval-index.npz
    python recipes/decision/clef.py --mixture.root data/mixture --trainer.checkpoint-dir runs
    python recipes/decision/clef.py --mixture.root data/mixture --pretrained Qwen/Qwen3.5-9B \\
        --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --lora.rank 256

Qwen3.5-4B (Apache-2.0) stays frozen under rank-64 LoRA factors on every
projection, the full-attention layers' q, k, v and o, the Gated DeltaNet
layers' qkv, z and output projections, and every MLP's, while a fresh joint
schema head of Clef-flash's shape learns with them. Clef's joint layout asks
every question of an example in one row of up to 4,096 tokens, in a new
order on each read, and the head pools each question's instructions and
each option's tokens and lets the questions attend to each other. Clef
trains rank-256 factors over Qwen3.8-27B and Clef-flash over Qwen3.5-9B; the
third command is the size-matched comparison with Clef-flash, for an H100.

It reads the same sets as the encoder recipe, with the same proper scoring
rules. An example whose questions alone pass 4,096 tokens is dropped and
counted (`DecisionRunConfig`'s summary, `unfit`). AdamW at 1e-4 warms up
over 400 steps and decays on a cosine to step 12,000, with 4 rows a step;
where a step does not fit the device, the trainer recomputes more of each
block and compiles it again.
"""

from encoder import RULES

from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, TrainerConfig
from dew.decision import DecisionMixture
from dew.decision.config import DecisionRunConfig, JointShape
from dew.lora import LoRA
from dew.training.optim import Cosine

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
               "gate_proj", "up_proj", "down_proj")
"""Every projection of a Qwen3.5 decoder layer, attention, Gated DeltaNet and MLP alike."""


def run_config() -> DecisionRunConfig:
    """The Clef recipe's run, every value a flag."""
    return DecisionRunConfig(
        pretrained="Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        model=ModelConfig("causal_transformer", {"dtype": "bfloat16"}),
        lora=LoRA(rank=64, modules=PROJECTIONS), head=JointShape(),
        mixture=DecisionMixture("data/mixture"),
        objective=ObjectiveConfig("decision", {"loss": RULES, "label_smoothing": 0.05}),
        optim=OptimConfig(learning_rate=1e-4, schedule=Cosine(peak=1e-4, warmup_steps=400)),
        trainer=TrainerConfig(batch_size=4, steps=12_000, eval_every=1_000, checkpoint_every=1_000),
        max_len=4096)


if __name__ == "__main__":
    DecisionRunConfig.cli(default=run_config()).run()
