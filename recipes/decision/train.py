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

The loss is the log loss unless the objective names another rule, or a
weighted sum of rules (`--objective.loss '{"class": "dew.decision.scoring:Combined",
"fields": {"terms": [[1.0, {"class": "log_loss"}], [0.5, {"class": "brier"}]]}}'`).
A tenth of the rows, at most 400,
are held out (`--data.held-out`); validation scores them, and after
training they fit the temperatures the task divides its logits by, saved
into the run, so `Decide.from_run` and `dew.pipeline` answer calibrated.
A question type's temperature needs ten held-out answers and a bucket's
2000, Laya's floors.
"""

import dataclasses
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, RunConfig
from dew.decision import (
    AURC,
    ECE,
    Accuracy,
    Decide,
    DecisionModel,
    DecisionTable,
    JointLayout,
    JointSchemaHead,
    LayaCheckpoint,
    LogLoss,
    MarkerLayout,
    StateFirstLayout,
)
from dew.interop import Pretrained
from dew.nn.backbones.decoder_block import remat_policy
from dew.training import TrainState, prepare_process, run_timestamp

sys.path.insert(0, str(Path(__file__).parent))
from contamination import Overlaps
from sources import Mixture, weights


@dataclass(frozen=True)
class JointShape:
    """The sizes of Clef's joint schema head (`dew.decision.JointSchemaHead`), Clef-flash's by default.

    Its input width is the backbone's, which the run reads from the model.
    """

    width: int = 1024
    routing_layers: int = 2
    layers: int = 4
    heads: int = 16
    feedforward: int = 4096


@dataclass(frozen=True)
class DecisionRunConfig(RunConfig):
    """A run, plus the decision objective's own knobs."""

    objective: ObjectiveConfig = field(default_factory=lambda: ObjectiveConfig("decision"))
    """The decision objective and its arguments: its scoring rule (`loss`),
    label smoothing, option shuffling and none-of-the-above rate."""
    data: DecisionTable = field(default_factory=DecisionTable)
    mixture: Mixture | None = None
    """The labelled sets to train on, each at its weight (`sources.Mixture`),
    in place of `data`."""
    decontaminate: str | None = None
    """An evaluation index (`contamination.py --out`): examples that share
    content with an evaluation item are dropped before training."""
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(learning_rate=2.5e-5))
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig("causal_transformer", {"dtype": "bfloat16"}))
    pretrained: str = "convaiinnovations/laya"
    """A Laya-style repository, read whole, or a Hugging Face model, read as the backbone."""
    subfolder: str | None = None
    """The checkpoint's folder inside a repository that bundles several, such as `multilingual`."""
    revision: str | None = None
    head: JointShape | None = None
    """Clef's joint schema head over a fresh backbone, which then reads Clef's
    joint layout; None draws Laya's head."""
    remat: str | None = None
    """The backbone's rematerialization policy (`REMAT_POLICIES`), such as
    "full" for a decoder whose activations would not fit."""
    calibrate: bool = True
    """Fit the task's temperatures on the held-out rows after training."""
    max_len: int | None = None
    """The row's length in tokens; None keeps the checkpoint's (Laya's 512) or the layout's default."""
    head_max_len: int | None = None
    """The tokens the question and its options share; None keeps the
    checkpoint's (Laya's 192). A question of many options wants more, or
    its options are cut to a few tokens each."""
    option_tokens: int | None = None
    """The most tokens one option keeps in a layout of one row per question."""


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
    if config.remat is not None:
        source = replace(source, model=_rematerialized(source.model, config.remat))
    return source if config.lora is None else source.adapt(config.lora, key=config.trainer.key)


def _rematerialized(model, policy: str):
    """`model` recomputing its blocks under `policy`: a decoder's own field, or a
    multimodal wrapper's decoder's."""
    named = {entry.name for entry in dataclasses.fields(model)}
    if "remat" in named:
        return model.clone(remat=remat_policy(policy))
    if "language_model" in named:
        return model.clone(language_model=model.language_model.clone(remat=remat_policy(policy)))
    raise ValueError(f"a {type(model).__name__} has no blocks to recompute")


def layout_of(config: DecisionRunConfig, start: Decide | Pretrained):
    """The layout the run lays its rows out with, or None for the objective's default.

    A Laya checkpoint keeps its own, resized by the run's budgets; a joint head
    reads Clef's joint layout; any other backbone reads the layout its order
    asks for, at the run's budgets.
    """
    stated = {"max_len": config.max_len, "head_max_len": config.head_max_len,
              "option_tokens": config.option_tokens}
    budgets = {name: value for name, value in stated.items() if value is not None}
    if isinstance(start, Decide):
        return replace(start.layout, **budgets)
    if config.head is not None:
        return JointLayout(**{name: value for name, value in budgets.items() if name == "max_len"})
    if not budgets:
        return None
    return (StateFirstLayout if start.model.causal else MarkerLayout)(**budgets)


def main(config: DecisionRunConfig) -> TrainState:
    prepare_process(config.trainer.wandb, config.trainer.multi_host,
                    config.trainer.xla_flags, config.trainer.compilation_cache_dir,
                    layout=config.trainer.layout)
    start = backbone(config)
    derived = {"backbone": start}
    layout = layout_of(config, start)
    if layout is not None:
        derived["layout"] = layout
    if config.head is not None:
        if isinstance(start, Decide):
            raise ValueError("--head draws Clef's head over a fresh backbone; a Laya checkpoint keeps "
                             "its own")
        width, dtype = DecisionModel.head_size(start.model)
        derived["head"] = JointSchemaHead(hidden_size=width, dtype=dtype, **dataclasses.asdict(config.head))
    objective = config.objective.build(**derived)
    overlaps = None if config.decontaminate is None else Overlaps.load(config.decontaminate)
    if config.mixture is not None:
        train, held_out = config.mixture.read(None if overlaps is None else overlaps.keep)
        source_name = "mixture"
    else:
        table_train, held_out = config.data.examples()
        train = table_train if overlaps is None else overlaps.keep("data", table_train, natural=True)
        source_name = config.data.path.replace("/", "-")
    data = objective.dataset(train, batch=config.trainer.batch_size, validation=held_out or None,
                             seed=config.data.seed, loading=config.data.loading)
    name = config.trainer.name or (
        f"decision-{config.pretrained.replace('/', '-')}/{source_name}/"
        f"lr-{config.optim.learning_rate}/date-{run_timestamp()}")
    validation = (Accuracy(), ECE(), AURC(), LogLoss())
    summary = {"pretrained": config.pretrained, "records": data.records, "loss": objective.loss_rule.name}
    if config.mixture is not None:
        summary["mixture"] = dict(weights(config.mixture))
    if overlaps is not None:
        summary["decontamination"] = overlaps.report
    state = config.train(objective, data, name=name, metrics=validation, summary=summary)
    if config.calibrate and held_out:
        run = f"{config.trainer.checkpoint_dir}/{name}"
        calibrated = replace(objective.pipeline(state).calibrated(held_out), name=name.rsplit("/", 1)[-1])
        calibrated.save(run)
        print(json.dumps({"temperatures": dict(calibrated.calibration.temperatures.types)}))
    return state


if __name__ == "__main__":
    main(DecisionRunConfig.cli())
