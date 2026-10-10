"""The decision run, as one typed record: a decision model fine-tuned on a
table of labelled text, as `laya-train --data tickets.csv` does.

By default the run starts from Laya's released English checkpoint, encoder
and head. `pretrained` names another: a Laya-style repository (one with an
`rl_agent_config.json`), or any Hugging Face model, whose head is then drawn
fresh, with `lora` to train an adapter in place of the whole model. A fresh
head is Laya's, reading the layout the backbone's order asks for, or with
`head` Clef's joint schema head, reading every question of an example in one
row (`JointLayout`). `mixture` trains on several labelled sets, each at its
share of every step, in place of `data`'s table. A tenth
of the rows, at most 400, are held out (`DecisionTable.held_out`);
validation scores them, and after training they fit the temperatures the
task divides its logits by, saved into the run, so `Decide.from_run` and
`dew.pipeline` answer calibrated. A question type's temperature needs ten
held-out answers and a bucket's 2000, Laya's floors.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import TYPE_CHECKING

import jax

from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, Prepared, RunConfig
from dew.data.text import HFTokenizer
from dew.decision.data import DecisionMixture, DecisionTable, Example, Weighted
from dew.decision.head import JointSchemaHead
from dew.decision.laya import LayaCheckpoint
from dew.decision.layout import JointLayout, Layout, MarkerLayout, StateFirstLayout
from dew.decision.metrics import AURC, ECE, Accuracy
from dew.decision.model import DecisionModel
from dew.decision.objective import DecisionObjective
from dew.decision.scoring import LogLoss
from dew.decision.task import Decide
from dew.training.state import TrainState

if TYPE_CHECKING:
    from dew.interop import Pretrained


@dataclasses.dataclass(frozen=True)
class JointShape:
    """The sizes of Clef's joint schema head (`JointSchemaHead`), Clef-flash's by default.

    Its input width is the backbone's, which the run reads from the model.
    """

    width: int = 1024
    routing_layers: int = 2
    layers: int = 4
    heads: int = 16
    feedforward: int = 4096


@dataclasses.dataclass(frozen=True)
class DecisionRunConfig(RunConfig):
    """A run, plus the decision objective's own knobs."""

    objective: ObjectiveConfig = dataclasses.field(default_factory=lambda: ObjectiveConfig("decision"))
    """The decision objective and its arguments: its scoring rule (`loss`),
    label smoothing, option shuffling and none-of-the-above rate."""
    data: DecisionTable = dataclasses.field(default_factory=DecisionTable)
    mixture: DecisionMixture | None = None
    """Labelled sets to train on, each at its share of every step, in place of `data`."""
    optim: OptimConfig = dataclasses.field(default_factory=lambda: OptimConfig(learning_rate=2.5e-5))
    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("causal_transformer", {"dtype": "bfloat16"}))
    pretrained: str = "convaiinnovations/laya"
    """A Laya-style repository, read whole, or a Hugging Face model, read as the backbone."""
    param_dtype: str = "float32"
    """The loaded checkpoint's parameter storage dtype; bfloat16 halves a frozen base's memory,
    leaving fresh LoRA factors and the head in float32."""
    subfolder: str | None = None
    """The checkpoint's folder inside a repository that bundles several, such as `multilingual`."""
    revision: str | None = None
    loop_steps: int | None = None
    """The passes a looped checkpoint runs its stack (`CausalTransformer.loop`), such
    as Ouro's at 1, 2 or 4; None keeps the checkpoint's own."""
    head: JointShape | None = None
    """Clef's joint schema head over a fresh backbone, which then reads Clef's
    joint layout; None draws Laya's head."""
    calibrate: bool = True
    """Fit the task's temperatures on the held-out rows after training."""
    max_len: int | None = None
    """The row's length in tokens; None keeps the checkpoint's (Laya's 512) or the layout's own."""
    head_max_len: int | None = None
    """The tokens the question and its options share; None keeps the
    checkpoint's (Laya's 192). A question of many options wants more, or
    its options are cut to a few tokens each."""
    option_tokens: int | None = None
    """The most tokens one option keeps, in a layout of one row per question."""

    def prepare(self) -> Prepared:
        """The objective over the backbone `pretrained` names, on the table's
        or the mixture's rows, scored on the held-out ones, which calibrate the
        run after it trains. The checkpoint decides the backbone; the model
        flags choose only its compute dtype and attention kernel;
        `param_dtype` chooses the loaded checkpoint's storage dtype."""
        from dew.interop import Pretrained, sources

        decided = sorted(set(self.model.fields) - {"dtype", "attention_impl"})
        if decided:
            raise ValueError(f"--model sets {decided}, which {self.pretrained} decides; only dtype and "
                             "attention_impl are choices")
        dtype = str(self.model.fields.get("dtype") or "float32")
        attention_impl = str(self.model.fields.get("attention_impl", "auto"))
        stated = {"max_len": self.max_len, "head_max_len": self.head_max_len,
                  "option_tokens": self.option_tokens}
        budgets = {name: value for name, value in stated.items() if value}
        derived = {}
        if LayaCheckpoint.exists(self.pretrained, subfolder=self.subfolder, revision=self.revision):
            if self.lora is not None or self.head is not None or self.loop_steps is not None:
                raise ValueError("--lora, --head and --loop-steps shape a fresh backbone; a Laya checkpoint "
                                 "trains whole under its own head")
            start = Decide.from_pretrained(self.pretrained, subfolder=self.subfolder, revision=self.revision,
                                           dtype=dtype, param_dtype=self.param_dtype,
                                           attention_impl=attention_impl)
            derived["layout"] = dataclasses.replace(start.layout, **budgets)
        else:
            start = Pretrained.load(self.pretrained, revision=self.revision, dtype=dtype,
                                    param_dtype=self.param_dtype, attention_impl=attention_impl)
            start = start if self.loop_steps is None else looping(start, self.loop_steps)
            start = start if self.lora is None else start.adapt(self.lora, key=self.trainer.key)
            # The tokenizer of the commit the weights come from.
            root = sources.snapshot(self.pretrained, self.revision, weights=False)
            derived["tokenizer"] = HFTokenizer(str(root), local_files_only=root != Path(self.pretrained))
            if self.head is not None:
                width, head_dtype = DecisionModel.head_size(start.model)
                derived["head"] = JointSchemaHead(hidden_size=width, dtype=head_dtype,
                                                  **dataclasses.asdict(self.head))
            derived["layout"] = _fresh_layout(start.model.causal, self.head is not None, budgets)
        objective = self.objective.build(backbone=start, **derived)
        assert isinstance(objective, DecisionObjective), "a decision run's objective is a DecisionObjective"
        summary: dict[str, object] = {"pretrained": self.pretrained}
        if self.mixture is None:
            train, held_out = self.data.examples()
            weighted = {"examples": Weighted(train, 1.0)}
        else:
            weighted, held_out = self.mixture.read()
            summary["mixture"] = self.mixture.shares()
            summary["made"] = self.mixture.recorded()
        weighted, held_out, unfit = _fitting(objective, weighted, held_out)
        if any(unfit.values()):
            summary["unfit"] = unfit
        examples = weighted if self.mixture is not None else weighted["examples"].examples
        dataset = objective.dataset(examples, batch=self.trainer.batch_size, validation=held_out or None,
                                    seed=self.data.seed, loading=self.data.loading)
        summary["records"] = dataset.records

        def calibrate(state: TrainState, directory: str) -> None:
            """Fit the task's temperatures on the held-out rows; process zero saves them into the run."""
            task = objective.pipeline(state)
            assert isinstance(task, Decide), "a decision objective's task is a Decide"
            calibrated = dataclasses.replace(task.calibrated(held_out), name=os.path.basename(directory))
            if jax.process_index() == 0:
                calibrated.save(directory)

        metrics = (Accuracy(), ECE(), AURC(), LogLoss())
        return Prepared(self, lambda name: self.train(objective, dataset, name=name, metrics=metrics,
                                                      summary=summary),
                        after=calibrate if self.calibrate and held_out else None)


def looping(start: Pretrained, steps: int) -> Pretrained:
    """A loaded looped checkpoint whose stack runs `steps` passes; one that does not loop is refused."""
    from dew.nn.protocols import Looping

    looped = start.model.with_passes(steps) if isinstance(start.model, Looping) else None
    if looped is None:
        raise ValueError(f"--loop-steps sets a looped checkpoint's passes, and this "
                         f"{type(start.model).__name__} runs its stack once")
    return dataclasses.replace(start, model=looped)


def _fresh_layout(causal: bool, joint: bool, budgets: dict[str, int]) -> Layout:
    """The layout a fresh head reads: Clef's joint one, or the one a backbone's order asks for."""
    if joint:
        return JointLayout() if "max_len" not in budgets else JointLayout(max_len=budgets["max_len"])
    return dataclasses.replace(StateFirstLayout() if causal else MarkerLayout(), **budgets)


def _fitting(objective: DecisionObjective, weighted: dict[str, Weighted],
             held: list[Example]) -> tuple[dict[str, Weighted], list[Example], dict[str, int]]:
    """The examples the objective's layout can lay out, and how many of each set it cannot.

    A joint row holds every question of an example, and one whose questions
    alone pass the layout's `max_len` cannot be laid out at all: it is dropped
    and counted here, rather than stopping the run when a worker reaches it. A
    layout of one row per question lays out every example.
    """
    if not objective.layout.joint:
        return weighted, held, {}

    def fits(example: Example) -> bool:
        try:
            objective.layout.rows(objective.tokenizer, objective.specials, example.state, example.questions)
        except ValueError:
            return False
        return True

    kept, unfit = {}, {}
    for name, entry in weighted.items():
        examples = [example for example in map(Example.of, entry.examples) if fits(example)]
        unfit[name] = len(entry.examples) - len(examples)
        kept[name] = Weighted(examples, entry.weight)
    held_kept = [example for example in held if fits(example)]
    unfit["held out"] = len(held) - len(held_kept)
    return kept, held_kept, unfit


__all__ = ["DecisionRunConfig", "JointShape"]
