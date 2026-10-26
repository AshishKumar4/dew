"""The decision run, as one typed record: a decision model fine-tuned on a
table of labelled text, as `laya-train --data tickets.csv` does.

By default the run starts from Laya's released English checkpoint, encoder
and head. `pretrained` names another: a Laya-style repository (one with an
`rl_agent_config.json`), or any Hugging Face model, whose head is then drawn
fresh, with `lora` to train an adapter in place of the whole model. A fresh
head is Laya's, reading the layout the backbone's order asks for, or with
`head` the joint schema head, reading every question of an example in one
row (`JointLayout`). Its data is a chain through `decisions` over a table, or `mixture`'s over a
mixture manifest whose sets contribute their recorded shares, and its validation the rows a
`holdout` keeps or a store's own split; validation scores them, and after training they fit the
temperatures the
task divides its logits by, saved into the run, so `Decide.from_run` and
`dew.pipeline` answer calibrated. A question type's temperature needs ten
held-out answers and a bucket's 2000, Laya's floors.
"""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Annotated, ClassVar

import jax

from dew._records import number, record
from dew.config import DecoderRunConfig, MetricConfig, ObjectiveConfig, Prepared
from dew.decision.head import DecisionHead, JointSchemaHead, LabelReadout
from dew.decision.laya import LAYA
from dew.decision.layout import Framing, Labels, Layout, QuestionLayout, Schema, Segment, default_layout
from dew.decision.model import DecisionModel
from dew.decision.objective import DecisionObjective
from dew.decision.scoring import default_metrics
from dew.decision.task import Decide
from dew.nn.backbones.decoder_block import remat_policy
from dew.nn.cross_attention import ContextEncoder, CrossAttention
from dew.nn.protocols import RematChoosing, block_chunks_of
from dew.optim import OptimConfig
from dew.registry import ModelConfig, import_path, record_argument
from dew.state import TrainState

if TYPE_CHECKING:
    pass


@dataclasses.dataclass(frozen=True)
class JointShape:
    """The sizes of the joint schema head (`JointSchemaHead`), its own defaults.

    Its input width is the backbone's, which the run reads from the model.
    """

    width: int = JointSchemaHead.width
    routing_layers: int = JointSchemaHead.routing_layers
    layers: int = JointSchemaHead.layers
    heads: int = JointSchemaHead.heads
    feedforward: int = JointSchemaHead.feedforward


@dataclasses.dataclass(frozen=True)
class Letters:
    """The letter readout (`LabelReadout`): each option scored by the backbone's own logit at its
    label's token, read at the row's answer cue, as a slot release reads it; its rows lettered as
    `labels` letter them, each label one token of the run's tokenizer. The head holds `count`
    labels, and its rows letter no more: a question of more options, one the objective's
    none-of-the-above adds among them, is refused where its row is laid out. `untied` gives each
    label a trainable output row of its own, drawn from the backbone's."""

    labels: Labels = dataclasses.field(default_factory=Labels)
    count: int = 26
    untied: bool = False


@dataclasses.dataclass(frozen=True)
class DecisionRunConfig(DecoderRunConfig):
    """A run, plus the decision objective's own knobs."""

    RESUMABLE: ClassVar[tuple[str, ...]] = (*DecoderRunConfig.RESUMABLE, "calibrate")

    objective: ObjectiveConfig = dataclasses.field(default_factory=lambda: ObjectiveConfig("decision"))
    """The decision objective and its arguments: its scoring rule (`loss`),
    label smoothing, option shuffling and none-of-the-above rate."""
    optim: OptimConfig = dataclasses.field(default_factory=lambda: OptimConfig(learning_rate=2.5e-5))
    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("causal_transformer", {"dtype": "bfloat16"}))
    pretrained: str = LAYA
    """A Laya-style repository, read whole, or a Hugging Face model, read as the backbone."""
    param_dtype: str = "float32"
    """The loaded checkpoint's parameter storage dtype; bfloat16 halves a frozen base's memory,
    leaving fresh LoRA factors and the head in float32."""
    subfolder: str | None = None
    """The checkpoint's folder inside a repository that bundles several, such as `multilingual`."""
    revision: str | None = None
    remat: str | None = None
    """A `REMAT_POLICIES` name for the loaded backbone's layers, applied before
    looping and LoRA. The trainer's fitting ladder starts here; a policy
    outside `DECODER_REMAT` cannot climb. None keeps the checkpoint's policy."""
    head: JointShape | Letters | None = None
    """The head over a fresh backbone: the joint schema head, which then reads the joint layout, or
    the letter readout of the backbone's own logits (`Letters`), over lettered rows; None draws
    Laya's head."""
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
    segments: Annotated[tuple[Segment, ...], record_argument(tuple[Segment, ...])] | None = None
    """A joint row's user turn, its parts in order (`JointLayout.segments`): the schema first, or
    read as the row's context through `context_layers` (`Schema(stream="context")`), its
    `context_length` holding the whole schema. None keeps the layout's own, the release's turn."""
    framing: Framing | None = None
    """How a joint row frames itself as a conversation (`Framing`): its system message, the
    assistant's prefill, the reasoning switch, a Jinja chat template in place of the tokenizer's
    own, and the template's other variables; recorded with the run's layout, so the run reloads it.
    None keeps the layout's own, the tokenizer's template with thinking off."""
    context_layers: tuple[int, ...] | None = None
    """The backbone layers that read each example's context (`Example.context`)
    through gated cross-attention, the context encoded as `context_encoder` says;
    the blocks train from zero gates beside the head, the loaded backbone frozen
    (`Pretrained.with_cross_attention`), with `lora` its adapter beside them. None
    reads no context."""
    context_length: int = 256
    """The most tokens of an example's context the cross-attention reads; an
    example with a longer context refuses the run where its data reads it,
    before training for data read into memory, as a request with one is
    refused. With `context_layers` alone."""
    context_encoder: ContextEncoder = "table"
    """How a context becomes the states its cross-attention reads
    (`DecisionModel.context_encoder`): the backbone's token table, or the
    backbone's own pass over it. With `context_layers` alone."""
    def prepare(self) -> Prepared:
        """The objective over the backbone `pretrained` names, on the table's
        or the mixture's rows, scored on the held-out ones, which calibrate the
        run after it trains. The checkpoint decides the backbone; the model
        flags choose only its compute dtype and attention kernel;
        `param_dtype` chooses the loaded checkpoint's storage dtype. `remat`
        chooses its layer recomputation before looping or attaching LoRA."""
        from dew.interop import Pretrained

        decided = sorted(set(self.model.fields) - {"dtype", "attention_impl"})
        if decided:
            raise ValueError(f"--model sets {decided}, which {self.pretrained} decides; only dtype and "
                             "attention_impl are choices")
        dtype = str(self.model.fields.get("dtype") or "float32")
        attention_impl = str(self.model.fields.get("attention_impl", "auto"))
        budgets = {"max_len": self.max_len, "head_max_len": self.head_max_len,
                   "option_tokens": self.option_tokens, "max_state_tokens": self.max_state_tokens,
                   "framing": self.framing, "segments": self.segments}
        in_context = any(isinstance(segment, Schema) and segment.stream == "context"
                         for segment in self.segments or ())
        if in_context and self.context_layers is None:
            raise ValueError("a schema read as the row's context is read by context layers; set "
                             "--context-layers")
        if self.context_encoder != "table" and self.context_layers is None:
            raise ValueError("--context-encoder says how context layers read a context; set --context-layers")
        derived = {}
        if self.context_layers is not None:
            derived["context_encoder"] = self.context_encoder
        from dew.decision.sources import PretrainedDecision

        frozen = self.lora is not None or self.context_layers is not None
        start = Pretrained.load(self.pretrained, subfolder=self.subfolder, revision=self.revision,
                                dtype=dtype, param_dtype=self.param_dtype, attention_impl=attention_impl,
                                **self.source_keywords(frozen=frozen))
        release = isinstance(start, PretrainedDecision)
        if release and self.head is not None:
            raise ValueError("--head draws a head over a fresh backbone; a decision release reads with "
                             "its own")
        # The backbone's own transformations, a release's as a fresh backbone's: its head and layout
        # stay as the release states them.
        if self.remat is not None:
            if not isinstance(start.model, RematChoosing):
                raise ValueError(f"--remat chooses a decoder's layer policy (RematChoosing), which "
                                 f"{type(start.model).__name__} does not support")
            start = dataclasses.replace(start, model=start.model.with_remat(remat_policy(self.remat)))
        start = self.looped(start)
        if self.context_layers is not None:
            # Drawn from a key of their own, apart from the adapter's and the head's.
            blocks = CrossAttention(self.context_layers, self.context_length)
            start = start.with_cross_attention(blocks, key=jax.random.fold_in(
                jax.random.key(self.trainer.key), 1))
        start = start if self.lora is None else start.adapt(self.lora, key=self.trainer.key)
        if isinstance(start, PretrainedDecision):
            # A release lays its rows out as it states, its blocks where its overrides read them.
            start = Decide.of(start)
            derived["layout"] = _budgeted(start.layout, budgets)
        else:
            labels = None
            if isinstance(self.head, JointShape):
                width, head_dtype = DecisionModel.head_size(start.model)
                derived["head"] = JointSchemaHead(hidden_size=width, dtype=head_dtype,
                                                  **dataclasses.asdict(self.head))
            elif isinstance(self.head, Letters):
                # Its ids are its labels' under the objective's tokenizer, which lays the rows out too.
                letters, (width, head_dtype) = self.head.labels, DecisionModel.head_size(start.model)
                labels = dataclasses.replace(letters, names=letters.names_for(self.head.count))
                derived["head"] = LabelReadout(features=width, label_ids=(), untied=self.head.untied,
                                               dtype=head_dtype)
            head = derived.get("head", DecisionHead)
            # Lettered rows lie one question to a row, as the backbone's order asks.
            layout = _fresh_layout(start.model.causal, head.one_question or labels is not None, budgets)
            derived["layout"] = _laid(layout, bool(block_chunks_of(start.model)), labels)
        objective = self.objective.build(backbone=start, **derived)
        assert isinstance(objective, DecisionObjective), "a decision run's objective is a DecisionObjective"
        summary: dict[str, object] = {"pretrained": self.pretrained}
        # A mixture's weights and the manifest it was made by, as the chain states them (`Dataset.facts`).
        facts = self.chain().facts()
        if "mixture_weights" in facts:
            weights = record(facts["mixture_weights"], "mixture_weights")
            summary["mixture"] = {name: value for name, value in weights.items()
                                  if number(value, f"mixture_weights.{name}") > 0}
            summary["made"] = record(facts.get("mixture_made", {}), "mixture_made")
        # Compiled once: `train` reads the run's length and cadences off this loader and trains on it.
        dataset, held = self.compiled(objective.data(), self.chain(), self.validation)
        summary["records"] = dataset.records
        unfit = {name: counter.rejected.get("unfit", 0) for name, counter in dataset.counts.items()}
        unfit["held out"] = sum(counter.rejected.get("unfit",
            0) for counter in dataset.validation_counts.values())
        if any(unfit.values()):
            summary["unfit"] = unfit

        def calibrate(state: TrainState, directory: str) -> None:
            """Fit the task's temperatures on the held-out rows; process zero saves them into the run."""
            task = objective.pipeline(state)
            assert isinstance(task, Decide), "a decision objective's task is a Decide"
            assert held
            # The first held-out split the run scores is the one it calibrates on.
            calibrated = dataclasses.replace(task.calibrated(next(iter(held.values()))),
                                             name=os.path.basename(directory))
            if jax.process_index() == 0:
                calibrated.save(directory)

        metrics = self.validation_metrics(validates=bool(held))
        after = calibrate if self.calibrate and held else None
        return Prepared(self, lambda name: self.train(objective, dataset, name=name, metrics=metrics,
                                                      summary=summary, after=after, validation=held))

    def default_metrics(self, *, validates: bool) -> tuple[MetricConfig, ...]:
        """The task's own held-out suite (`default_metrics`), where the data holds held-out rows."""
        if not validates:
            return ()
        return tuple(MetricConfig(import_path(type(metric))) for metric in default_metrics())


type Budgets = dict[str, int | Framing | tuple[Segment, ...] | None]
"""A run's budgets of a row's layout, by the layout's field names (`_budgeted`)."""


def _fresh_layout(causal: bool, one_question: bool, budgets: Budgets) -> Layout:
    """The layout a fresh head reads (`default_layout`), with the run's `budgets`."""
    return _budgeted(default_layout(causal, one_question), budgets)


def _laid(layout: Layout, blocked: bool, labels: Labels | None) -> Layout:
    """A fresh head's `layout` with its rows in blocks where the backbone reads blocks whole
    (`blocked`, `block_chunks_of`, as the load's overrides or the checkpoint's own made it) and
    lettered by `labels` where the run reads letters; a layout whose rows can do neither, a joint row
    of every question, is refused."""
    stated = {**({"blocks": True} if blocked else {}), **({} if labels is None else {"labels": labels})}
    if stated and not isinstance(layout, QuestionLayout):
        raise ValueError(f"{type(layout).__name__} lays every question in one row, which lies in no blocks "
                         "and carries no labels; a block_bidirectional --source.attention and the letter "
                         "readout read a question to a row")
    return dataclasses.replace(layout, **stated)


def _budgeted(layout: Layout, budgets: Budgets) -> Layout:
    """`layout` with the `budgets` the run states, each but those left None, which keep the
    layout's own; zero is a budget too (a joint row of no state). A budget the layout has no
    field for is refused: a joint row has no question budget, and a row per question no state
    budget, its state taking what is left."""
    stated = {name: value for name, value in budgets.items() if value is not None}
    fields = {field.name for field in dataclasses.fields(layout)}
    unknown = sorted(f"--{name.replace('_', '-')}" for name in stated if name not in fields)
    if unknown:
        raise ValueError(f"{', '.join(unknown)} budget a row that {type(layout).__name__} does not lay out")
    return dataclasses.replace(layout, **stated)


__all__ = ["DecisionRunConfig", "JointShape", "Letters"]
