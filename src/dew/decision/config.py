"""The decision run, as one typed record: a decision model fine-tuned on a
table of labelled text, as `laya-train --data tickets.csv` does.

By default the run starts from Laya's released English checkpoint, encoder
and head. `pretrained` names another: a Laya-style repository (one with an
`rl_agent_config.json`), or any Hugging Face model, whose head is then drawn
fresh, with `lora` to train an adapter in place of the whole model. A tenth
of the rows, at most 400, are held out (`DecisionTable.held_out`);
validation scores them, and after training they fit the temperatures the
task divides its logits by, saved into the run, so `Decide.from_run` and
`dew.pipeline` answer calibrated.
"""

from __future__ import annotations

import dataclasses
import os

from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, Prepared, RunConfig
from dew.data.text import HFTokenizer
from dew.decision.data import DecisionTable
from dew.decision.laya import LayaCheckpoint
from dew.decision.metrics import AURC, ECE, Accuracy
from dew.decision.scoring import LogLoss
from dew.decision.task import Decide
from dew.training.state import TrainState


@dataclasses.dataclass(frozen=True)
class DecisionRunConfig(RunConfig):
    """A run, plus the decision objective's own knobs."""

    objective: ObjectiveConfig = dataclasses.field(default_factory=lambda: ObjectiveConfig("decision"))
    """The decision objective and its arguments: its scoring rule (`loss`),
    label smoothing, option shuffling and none-of-the-above rate."""
    data: DecisionTable = dataclasses.field(default_factory=DecisionTable)
    optim: OptimConfig = dataclasses.field(default_factory=lambda: OptimConfig(learning_rate=2.5e-5))
    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("causal_transformer", {"dtype": "bfloat16"}))
    pretrained: str = "convaiinnovations/laya"
    """A Laya-style repository, read whole, or a Hugging Face model, read as the backbone."""
    subfolder: str | None = None
    """The checkpoint's folder inside a repository that bundles several, such as `multilingual`."""
    revision: str | None = None
    calibrate: bool = True
    """Fit the task's temperatures on the held-out rows after training."""
    max_len: int | None = None
    """The row's length in tokens; None keeps the checkpoint's (Laya's 512)."""
    head_max_len: int | None = None
    """The tokens the question and its options share; None keeps the
    checkpoint's (Laya's 192). A question of many options wants more, or
    its options are cut to a few tokens each."""

    def prepare(self) -> Prepared:
        """The objective over the backbone `pretrained` names, on the table's
        rows, scored on the held-out ones, which calibrate the run after it
        trains. The checkpoint decides the backbone; the model flags choose
        only its compute dtype and attention kernel."""
        from dew.interop import Pretrained

        decided = sorted(set(self.model.fields) - {"dtype", "attention_impl"})
        if decided:
            raise ValueError(f"--model sets {decided}, which {self.pretrained} decides; only dtype and "
                             "attention_impl are choices")
        dtype = str(self.model.fields.get("dtype") or "float32")
        attention_impl = str(self.model.fields.get("attention_impl", "auto"))
        if LayaCheckpoint.exists(self.pretrained, subfolder=self.subfolder, revision=self.revision):
            if self.lora is not None:
                raise ValueError("--lora adapts a Hugging Face backbone; a Laya checkpoint trains whole")
            start = Decide.from_pretrained(self.pretrained, subfolder=self.subfolder, revision=self.revision,
                                           dtype=dtype, attention_impl=attention_impl)
            layout, tokenizer = dataclasses.replace(
                start.layout, max_len=self.max_len or start.layout.max_len,
                head_max_len=self.head_max_len or start.layout.head_max_len), None
        else:
            if self.max_len or self.head_max_len:
                raise ValueError("--max-len and --head-max-len resize a Laya checkpoint's layout; another "
                                 "backbone takes the default layout")
            start = Pretrained.load(self.pretrained, revision=self.revision, dtype=dtype,
                                    attention_impl=attention_impl)
            start = start if self.lora is None else start.adapt(self.lora, key=self.trainer.key)
            layout, tokenizer = None, HFTokenizer(self.pretrained)
        objective = self.objective.build(backbone=start, tokenizer=tokenizer, layout=layout)
        train, held_out = self.data.examples()
        dataset = objective.dataset(train, batch=self.trainer.batch_size, validation=held_out or None,
                                    seed=self.data.seed, loading=self.data.loading)

        def calibrate(state: TrainState, directory: str) -> None:
            """Fit the task's temperatures on the held-out rows and save them into the run."""
            calibrated = objective.pipeline(state).calibrated(held_out)
            dataclasses.replace(calibrated, name=os.path.basename(directory)).save(directory)

        return Prepared(self, objective, dataset, metrics=(Accuracy(), ECE(), AURC(), LogLoss()),
                        after=calibrate if self.calibrate and held_out else None)


__all__ = ["DecisionRunConfig"]
