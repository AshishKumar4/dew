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
import logging
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
from dew.nn.backbones.decoder_block import remat_policy
from dew.nn.protocols import RematChoosing
from dew.training.state import TrainState

_log = logging.getLogger(__name__)

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
    """The passes a looped checkpoint runs its block (`CausalTransformer.loop`), such
    as Ouro's at 1, 2 or 4; None keeps the checkpoint's own."""
    loop_layers: tuple[int, int] | None = None
    """The block of layers [first, end) that `loop_steps` repeats: a pretrained
    decoder with no loop gains one over its own layers, as Qwen 3.5's middle block
    repeated twice; None keeps a looped checkpoint's block."""
    loop_backprop_steps: int | None = None
    """Train through the last k passes of the loop alone, with earlier passes
    cut from backpropagation; None keeps the loop's own choice."""
    loop_remat: bool = False
    """Checkpoint every pass of the loop, including the first, and recompute
    it in the backward pass; requires a loop and leaves decoding unchanged."""
    remat: str | None = None
    """A `REMAT_POLICIES` name for the loaded backbone's layers, applied before
    looping and LoRA. The trainer's fitting ladder starts here; a policy
    outside `DECODER_REMAT` cannot climb. None keeps the checkpoint's policy."""
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
    max_state_tokens: int | None = None
    """The most tokens a joint row's state keeps, whatever room `max_len` leaves
    it: a row is its state, cut to this, and its schema, so a large schema
    (a hundred options, or a document's many hypotheses) fits under a `max_len`
    set for it without lengthening every other row's state. None lets the
    state fill the row."""

    def prepare(self) -> Prepared:
        """The objective over the backbone `pretrained` names, on the table's
        or the mixture's rows, scored on the held-out ones, which calibrate the
        run after it trains. The checkpoint decides the backbone; the model
        flags choose only its compute dtype and attention kernel;
        `param_dtype` chooses the loaded checkpoint's storage dtype. `remat`
        chooses its layer recomputation before looping or attaching LoRA."""
        from dew.interop import Pretrained, sources

        decided = sorted(set(self.model.fields) - {"dtype", "attention_impl"})
        if decided:
            raise ValueError(f"--model sets {decided}, which {self.pretrained} decides; only dtype and "
                             "attention_impl are choices")
        dtype = str(self.model.fields.get("dtype") or "float32")
        attention_impl = str(self.model.fields.get("attention_impl", "auto"))
        stated = {"max_len": self.max_len, "head_max_len": self.head_max_len,
                  "option_tokens": self.option_tokens, "max_state_tokens": self.max_state_tokens}
        budgets = {name: value for name, value in stated.items() if value}
        derived = {}
        if LayaCheckpoint.exists(self.pretrained, subfolder=self.subfolder, revision=self.revision):
            if any(choice is not None for choice in (self.lora, self.head, self.loop_steps,
                                                    self.loop_layers, self.loop_backprop_steps,
                                                    self.remat)) or self.loop_remat:
                raise ValueError("--lora, --head, --remat and the loop shape a fresh backbone; "
                                 "a Laya checkpoint trains whole under its own head")
            start = Decide.from_pretrained(self.pretrained, subfolder=self.subfolder, revision=self.revision,
                                           dtype=dtype, param_dtype=self.param_dtype,
                                           attention_impl=attention_impl)
            derived["layout"] = _budgeted(start.layout, budgets)
        else:
            start = Pretrained.load(self.pretrained, revision=self.revision, dtype=dtype,
                                    param_dtype=self.param_dtype, attention_impl=attention_impl)
            if self.remat is not None:
                if not isinstance(start.model, RematChoosing):
                    raise ValueError(f"--remat chooses a decoder's layer policy (RematChoosing), which "
                                     f"{type(start.model).__name__} does not support")
                start = dataclasses.replace(start, model=start.model.with_remat(remat_policy(self.remat)))
            if (self.loop_steps is not None or self.loop_layers is not None
                    or self.loop_backprop_steps is not None or self.loop_remat):
                start = looping(start, self.loop_steps, self.loop_layers,
                                backprop_steps=self.loop_backprop_steps, remat=self.loop_remat or None)
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
        fitting, held_out, unfit = _fitting(objective, weighted, held_out)
        if any(unfit.values()):
            summary["unfit"] = unfit
        left = sorted(set(weighted) - set(fitting))
        if left:
            summary["left out"] = left
            _log.warning("%s left out of the mixture: none of their rows fits a joint row of %d tokens",
                         ", ".join(left), objective.layout.max_len)
        weighted = fitting
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
        fitted = calibrate if self.calibrate and held_out else None
        return Prepared(self, lambda name: self.train(objective, dataset, name=name, metrics=metrics,
                                                      summary=summary, after=fitted))


def looping(start: Pretrained, steps: int | None, layers: tuple[int, int] | None, *,
            backprop_steps: int | None = None, remat: bool | None = None) -> Pretrained:
    """A loaded checkpoint whose block runs `steps` passes: a looped one's own block, or `layers`,
    which gives a decoder with no loop one over its own layers. Refused where neither names a block,
    or a new block's passes are not stated. `backprop_steps` and `remat` choose
    the loop's backward memory; None keeps a looped checkpoint's own value."""
    from dew.nn.protocols import Looping

    looped = (start.model.with_passes(steps, layers, backprop_steps=backprop_steps, remat=remat)
              if isinstance(start.model, Looping) else None)
    if looped is None:
        raise ValueError(f"this {type(start.model).__name__} has no loop: --loop-layers names the block of "
                         "its layers to repeat and --loop-steps how many times")
    return dataclasses.replace(start, model=looped)


def _fresh_layout(causal: bool, joint: bool, budgets: dict[str, int]) -> Layout:
    """The layout a fresh head reads: Clef's joint one, or the one a backbone's order asks for."""
    return _budgeted(JointLayout() if joint else StateFirstLayout() if causal else MarkerLayout(), budgets)


def _budgeted(layout: Layout, budgets: dict[str, int]) -> Layout:
    """`layout` with the stated `budgets`, refusing one it has no field for: a joint row has
    no question budget, and a row per question no state budget, its state taking what is left."""
    fields = {field.name for field in dataclasses.fields(layout)}
    unknown = sorted(f"--{name.replace('_', '-')}" for name in budgets if name not in fields)
    if unknown:
        raise ValueError(f"{', '.join(unknown)} budget a row that {type(layout).__name__} does not lay out")
    return dataclasses.replace(layout, **budgets)


def _fitting(objective: DecisionObjective, weighted: dict[str, Weighted],
             held: list[Example]) -> tuple[dict[str, Weighted], list[Example], dict[str, int]]:
    """The examples the objective's layout can lay out, and how many of each set it cannot.

    A joint row holds every question of an example, and one whose questions
    alone pass the layout's `max_len` cannot be laid out at all: it is dropped
    and counted here, rather than stopping the run when a worker reaches it. A
    layout of one row per question lays out every example.

    A set none of whose rows fit has nothing to train on and is left out of
    the mixture, its rows counted, since grain refuses to mix an empty set;
    the run's summary names it (`prepare`). A budget no set fits is refused.
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
        if examples:
            kept[name] = Weighted(examples, entry.weight)
    if not kept:
        raise ValueError(
            f"no row of {sorted(weighted)} fits a joint row of {objective.layout.max_len} tokens: each "
            f"one's questions alone pass it; raise --max-len")
    held_kept = [example for example in held if fits(example)]
    unfit["held out"] = len(held) - len(held_kept)
    return kept, held_kept, unfit


__all__ = ["DecisionRunConfig", "JointShape"]
