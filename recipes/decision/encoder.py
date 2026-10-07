"""A decision model on a pretrained encoder with a fresh head, as Laya and OpenDecider-nano are built.

    python recipes/decision/encoder.py --decontaminate eval-index.npz --trainer.checkpoint-dir runs

ModernBERT-large (Apache-2.0, 396M parameters) is trained whole under a
fresh Laya head, reading Laya's layout: the question, a marker before each
option, then the state. It trains on `sources.Mixture`: Open-Jev, the
typed-decisions train split, gliclass, BANKING77, CLINC150, WANLI,
HellaSwag, ARC, BoolQ and GSM8K, each at its share of every step, with
every example that shares content with an evaluation item dropped first
(`--decontaminate`, built by `contamination.py`). Rows are 1,024 tokens;
the question and its options share 768 of them, an option keeps up to 256.
A served model widens these (`examples/serve_decisions.py --max-len`).

The loss is the log loss plus half the Brier score, plus the ranked
probability score on ordered levels, against the gold distribution with
5% of it spread over the options: the proper scoring rules Laya's RLCD and
Clef's post-training reward. AdamW at 2.5e-5, Laya's rate, warms up over
500 steps and decays on a cosine to step 15,000, with 32 rows a step. The
temperatures are then fitted to Open-Jev's calibration split and the
typed-decisions cases held back.

`--pretrained jhu-clsp/ettin-encoder-400m --revision 7662476d60abb071a5bd319c9f3074f3072c062d`
trains OpenDecider-nano's encoder instead.
"""

from dataclasses import dataclass, field

from sources import Mixture
from train import DecisionRunConfig, main

from dew.config import ObjectiveConfig, OptimConfig, TrainerConfig
from dew.decision import Brier, LogLoss, RankedProbability
from dew.training.optim import Cosine

RULES = LogLoss() + 0.5 * Brier() + RankedProbability()
"""The proper scoring rules the decision recipes train against."""


@dataclass(frozen=True)
class EncoderRun(DecisionRunConfig):
    """The encoder recipe's run: the module's description, every value a flag."""

    pretrained: str = "answerdotai/ModernBERT-large"
    revision: str | None = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
    mixture: Mixture | None = field(default_factory=Mixture)
    objective: ObjectiveConfig = field(default_factory=lambda: ObjectiveConfig(
        "decision", {"loss": RULES, "label_smoothing": 0.05}))
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(
        learning_rate=2.5e-5, schedule=Cosine(peak=2.5e-5, warmup_steps=500)))
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(
        batch_size=32, steps=15_000, eval_every=1_000, checkpoint_every=1_000))
    max_len: int | None = 1024
    head_max_len: int | None = 768
    option_tokens: int | None = 256


if __name__ == "__main__":
    main(EncoderRun.cli())
