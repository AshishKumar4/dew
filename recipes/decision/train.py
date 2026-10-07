"""Fine-tune a decision model on a table of labelled text, as `laya-train --data tickets.csv` does.

    python recipes/decision/train.py --data.path tickets.csv --data.text text --data.label label \\
        --trainer.batch-size 32 --trainer.epochs 4 --optim.learning-rate 2.5e-5

By default the run starts from Laya's released English checkpoint, encoder
and head. `--pretrained` names another: a Laya-style repository (one with
an `rl_agent_config.json`), or any Hugging Face model, whose head is then
drawn fresh, with `--lora.rank 16 --lora.modules q_proj v_proj` to train an
adapter in place of the whole model:

    python recipes/decision/train.py --pretrained Qwen/Qwen3-0.6B --lora.rank 16 \\
        --lora.modules q_proj v_proj --data.path PolyAI/banking77 --data.validation-split test \\
        --data.question intent --trainer.batch-size 32 --trainer.steps 2000

The loss is the log loss plus `--brier`, `--spherical` and
`--ranked-probability` times their rules. A tenth of the rows, at most 400,
are held out (`--data.held-out`); validation scores them, and after
training they fit the temperatures the task divides its logits by, saved
into the run, so `Decide.from_run` and `dew.pipeline` answer calibrated.
A question type's temperature needs ten held-out answers and a bucket's
2000, Laya's floors.
"""

import json
from dataclasses import dataclass, field, replace

from dew.config import ModelConfig, OptimConfig, RunConfig
from dew.data.text import HFTokenizer
from dew.decision import (
    AURC,
    ECE,
    Accuracy,
    Brier,
    Decide,
    DecisionObjective,
    DecisionTable,
    LayaCheckpoint,
    LogLoss,
    RankedProbability,
    ScoringRule,
    Spherical,
)
from dew.interop import Pretrained
from dew.training import TrainState, prepare_process, run_timestamp


@dataclass(frozen=True)
class DecisionRunConfig(RunConfig):
    """A run, plus the decision objective's own knobs."""

    objective: str = "decision"
    data: DecisionTable = field(default_factory=DecisionTable)
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(learning_rate=2.5e-5))
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig("causal_transformer", {"dtype": "bfloat16"}))
    pretrained: str = "convaiinnovations/laya"
    """A Laya-style repository, read whole, or a Hugging Face model, read as the backbone."""
    subfolder: str | None = None
    """The checkpoint's folder inside a repository that bundles several, such as `multilingual`."""
    revision: str | None = None
    brier: float = 0.0
    spherical: float = 0.0
    ranked_probability: float = 0.0
    label_smoothing: float = 0.0
    shuffle_options: bool = True
    none_of_the_above: float = 0.0
    calibrate: bool = True
    """Fit the task's temperatures on the held-out rows after training."""
    max_len: int | None = None
    """The row's length in tokens; None keeps the checkpoint's (Laya's 512)."""
    head_max_len: int | None = None
    """The tokens the question and its options share; None keeps the
    checkpoint's (Laya's 192). A question of many options wants more, or
    its options are cut to a few tokens each."""

    def loss(self) -> ScoringRule:
        """The log loss, plus each other rule at its weight."""
        rule: ScoringRule = LogLoss()
        for weight, other in ((self.brier, Brier()), (self.spherical, Spherical()),
                              (self.ranked_probability, RankedProbability())):
            if weight > 0:
                rule = rule + weight * other
        return rule


def backbone(config: DecisionRunConfig) -> Decide | Pretrained:
    """Laya's checkpoint whole, or a Hugging Face model as the backbone, adapted when `--lora` asks.

    The checkpoint decides the backbone; the model flags choose only its
    compute dtype and attention kernel."""
    settings = ("dtype", "attention_impl")
    decided = sorted(set(config.model.fields) - set(settings))
    if decided:
        raise ValueError(f"--model sets {decided}, which {config.pretrained} decides; only "
                         f"{', '.join(settings)} are choices")
    dtype = str(config.model.fields.get("dtype") or "float32")
    attention_impl = str(config.model.fields.get("attention_impl", "auto"))
    if LayaCheckpoint.exists(config.pretrained, subfolder=config.subfolder, revision=config.revision):
        if config.lora is not None:
            raise ValueError("--lora adapts a Hugging Face backbone; a Laya checkpoint trains whole")
        return Decide.from_pretrained(config.pretrained, subfolder=config.subfolder, revision=config.revision,
                                      dtype=dtype, attention_impl=attention_impl)
    source = Pretrained.load(config.pretrained, revision=config.revision, dtype=dtype,
                             attention_impl=attention_impl)
    return source if config.lora is None else source.adapt(config.lora, key=config.trainer.key)


def main(config: DecisionRunConfig) -> TrainState:
    prepare_process(config.trainer.wandb, config.trainer.multi_host,
                    config.trainer.xla_flags, config.trainer.compilation_cache_dir,
                    layout=config.trainer.layout)
    start = backbone(config)
    tokenizer = None if isinstance(start, Decide) else HFTokenizer(config.pretrained)
    layout = start.layout if isinstance(start, Decide) else None
    if layout is not None:
        layout = replace(layout, max_len=config.max_len or layout.max_len,
                         head_max_len=config.head_max_len or layout.head_max_len)
    elif config.max_len or config.head_max_len:
        raise ValueError("--max-len and --head-max-len resize a Laya checkpoint's layout; "
                         "another backbone takes the default layout")
    objective = DecisionObjective(start, loss=config.loss(), tokenizer=tokenizer, layout=layout,
                                  label_smoothing=config.label_smoothing,
                                  shuffle_options=config.shuffle_options,
                                  none_of_the_above=config.none_of_the_above)
    train, held_out = config.data.examples()
    data = objective.dataset(train, batch=config.trainer.batch_size, validation=held_out or None,
                             seed=config.data.seed, loading=config.data.loading)
    name = config.trainer.name or (
        f"decision-{config.pretrained.replace('/', '-')}/{config.data.path.replace('/', '-')}/"
        f"lr-{config.optim.learning_rate}/date-{run_timestamp()}")
    validation = (Accuracy(), ECE(), AURC(), LogLoss())
    summary = {"pretrained": config.pretrained, "records": data.records, "loss": objective.loss_rule.name}
    state = config.train(objective, data, name=name, metrics=validation, summary=summary)
    if config.calibrate and held_out:
        run = f"{config.trainer.checkpoint_dir}/{name}"
        calibrated = replace(objective.pipeline(state).calibrated(held_out), name=name.rsplit("/", 1)[-1])
        calibrated.save(run)
        print(json.dumps({"temperatures": dict(calibrated.calibration.temperatures.types)}))
    return state


if __name__ == "__main__":
    main(DecisionRunConfig.cli())
