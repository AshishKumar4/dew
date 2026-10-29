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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

import jax

from dew._records import integer, number, record
from dew.config import DecoderRunConfig, MetricConfig, ObjectiveConfig, Prepared
from dew.data.pipeline import Dataset, cut_unit
from dew.decision.head import DecisionHead, JointSchemaHead, LabelReadout
from dew.decision.laya import LAYA
from dew.decision.layout import (
    Framing,
    Labels,
    Layout,
    QuestionLayout,
    RowOrder,
    RowSegment,
    Schema,
    default_layout,
)
from dew.decision.model import DecisionModel
from dew.decision.objective import DecisionObjective
from dew.decision.scoring import default_metrics
from dew.decision.task import Decide
from dew.nn.attention import attention_implementation
from dew.nn.backbones.decoder_block import remat_policy
from dew.nn.cross_attention import ContextEncoder, CrossAttention
from dew.nn.protocols import RematChoosing
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
    `labels` letter them, each label one token of the run's tokenizer: where `labels` names none,
    the first `count` of `LABEL_NAMES` the tokenizer spells as one token (`Labels.spelled`). The head
    holds `count` labels, and its rows letter no more: a question of more options, one the
    objective's none-of-the-above adds among them, is refused where its row is laid out. `untied`
    gives each label a trainable output row of its own, drawn from the backbone's."""

    labels: Labels = dataclasses.field(default_factory=Labels)
    count: int | None = None
    """The labels the head holds; None resolves them from the run: the most options a question of its
    data asks (its `slots` facts, training's and validation's), one more where its objective adds "none
    of the above", or where the data states none, every label its tokenizer spells as one token."""
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
    """The tokens the question and its options may share, a declared cut; None
    keeps the checkpoint's (Laya's release declares 192) or a fresh head's,
    which keeps every question whole. Each row it cuts is counted in the run's
    data counters, and a row whose cut leaves two options alike is refused."""
    option_tokens: int | None = None
    """The most tokens one option keeps, in a layout of one row per question, a
    declared cut counted as `head_max_len`'s is; None keeps the checkpoint's
    (Laya's 48) or a fresh head's, each option whole."""
    max_state_tokens: int | None = None
    """The most tokens a joint row's state keeps, whatever room `max_len` leaves
    it: a row is its state, cut to this, and its schema, so a large schema
    (a hundred options, or a document's many hypotheses) fits under a `max_len`
    set for it without lengthening every other row's state. None lets the
    state fill the row."""
    segments: Annotated[tuple[RowSegment, ...], record_argument(tuple[RowSegment, ...])] | None = None
    """The row's parts in order, as its layout reads them. A joint row's user turn
    (`JointLayout.segments`): the schema first, or read as the row's context through
    `context_layers` (`Schema(stream="context")`), its `context_length` holding the whole schema. A
    question row's text (`QuestionLayout.segments`): its instructions, option lines and answer cue
    beside the state, the question first or the state first, its blocks following that order
    (`QuestionLayout.blocks`). None keeps the layout's own, the release's turn."""
    order: RowOrder | None = None
    """The order a question row's own text writes its parts in (`QuestionLayout.order`): the state
    first, the question first, or the state, the question and the state again (`re_read`), each part as
    the built-in row writes it. None keeps the layout's own, a causal backbone's state first."""
    split: Literal["part", "question", "whole"] | None = None
    """Where a question row splits into its blocks, where its backbone reads blocks (a
    block_bidirectional `--source.attention`, `QuestionLayout.split`): at its second part, a
    question-first row's state; at its question; or nowhere, the whole row one block its backbone
    reads both ways (`whole`). None keeps the layout's own law."""
    framing: Framing | None = None
    """How a joint row frames itself as a conversation (`Framing`): its system message, the
    assistant's prefill, the reasoning switch, a Jinja chat template in place of the tokenizer's
    own, and the template's other variables; recorded with the run's layout, so the run reloads it.
    None keeps the layout's own, the tokenizer's template with thinking off."""
    context_layers: tuple[int, ...] | None = None
    """The backbone layers that read each example's context (`Example.context`)
    through gated cross-attention, the context encoded as `context_encoder` says;
    the blocks train from zero gates beside the head, the loaded backbone frozen
    (`Pretrained.with_cross_attention`) unless `context_frozen` is off, with
    `lora` its adapter beside them. None reads no context."""
    context_frozen: bool = True
    """Whether the context layers freeze the backbone beside their blocks, which
    then train alone with the head; off, they freeze nothing more, so the
    backbone trains as the run trains it: whole, a full fine-tune, or through
    its `lora` adapter over the frozen base (`--no-context-frozen`). With
    `context_layers` alone."""
    context_length: int = 256
    """The most tokens of an example's context the cross-attention reads; an
    example with a longer context refuses the run where its data reads it,
    before training for data read into memory, as a request with one is
    refused. With `context_layers` alone."""
    context_encoder: ContextEncoder = "table"
    """How a context becomes the states its cross-attention reads
    (`DecisionModel.context_encoder`): the backbone's token table, or the
    backbone's own pass over it. With `context_layers` alone."""
    def pinned(self) -> DecisionRunConfig:
        """This run with its Hub sources at their commits (`RunConfig.pinned`), its backbone's `revision`
        among them (`SourceReference.pinned`), so its record names the commit it read."""
        from dew.interop.sources import SourceReference

        run = super().pinned()
        revision = SourceReference(self.pretrained, self.revision, self.subfolder).pinned().revision
        return run if revision == self.revision else dataclasses.replace(run, revision=revision)

    def build_run(self) -> Prepared:
        """The objective over the backbone `pretrained` names, on the table's
        or the mixture's rows, scored on the held-out ones, which calibrate the
        run after it trains. The checkpoint decides the backbone; the model
        flags choose only its compute dtype and attention kernel;
        `param_dtype` chooses the loaded checkpoint's storage dtype. `remat`
        chooses its layer recomputation before looping or attaching LoRA."""
        from dew.interop import Pretrained

        self.sourced_model()
        decided = sorted(set(self.model.fields) - {"dtype", "attention_impl"})
        if decided:
            raise ValueError(f"--model sets {decided}, which {self.pretrained} decides; only dtype and "
                             "attention_impl are choices")
        dtype = str(self.model.fields.get("dtype") or "float32")
        attention_impl = attention_implementation(str(self.model.fields.get("attention_impl", "auto")))
        budgets = {"max_len": self.max_len, "head_max_len": self.head_max_len,
                   "option_tokens": self.option_tokens, "max_state_tokens": self.max_state_tokens,
                   "framing": self.framing, "segments": self.segments, "order": self.order,
                   "split": self.split}
        in_context = any(isinstance(segment, Schema) and segment.stream == "context"
                         for segment in self.segments or ())
        if in_context and self.context_layers is None:
            raise ValueError("a schema read as the row's context is read by context layers; set "
                             "--context-layers")
        if self.context_encoder != "table" and self.context_layers is None:
            raise ValueError("--context-encoder says how context layers read a context; set --context-layers")
        if not self.context_frozen and self.context_layers is None:
            raise ValueError("--no-context-frozen says how context layers hold the backbone; set "
                             "--context-layers")
        derived = {}
        if self.context_layers is not None:
            derived["context_encoder"] = self.context_encoder
        from dew.decision.slot_source import reads_blocks
        from dew.decision.sources import PretrainedDecision

        frozen = self.lora is not None or (self.context_layers is not None and self.context_frozen)
        start = Pretrained.load(self.pretrained, subfolder=self.subfolder, revision=self.revision,
                                dtype=dtype, param_dtype=self.param_dtype, attention_impl=attention_impl,
                                **self.source_keywords(frozen=frozen))
        release = isinstance(start, PretrainedDecision)
        # Whether a fresh head's rows lie in blocks, as the loaded checkpoint and the run declare it.
        blocked = not release and reads_blocks(start.config, self.source.overrides())
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
                jax.random.key(self.trainer.key), 1), frozen=self.context_frozen)
        start = start if self.lora is None else start.adapt(self.lora, key=self.trainer.key)
        if isinstance(start, PretrainedDecision):
            # A release lays its rows out as it states, its blocks where its overrides read them.
            start = Decide.of(start)
            derived["layout"] = _budgeted(start.layout, budgets)
        else:
            derived.update(self._fresh_head(start, blocked, budgets))
        objective = self.objective.build(backbone=start, **derived)
        assert isinstance(objective, DecisionObjective), "a decision run's objective is a DecisionObjective"
        summary: dict[str, object] = {"pretrained": self.pretrained,
                                      "capacities": self._capacities(objective, release)}
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
        # The rows each layout refused, by reason, and the rows and tokens each cut removed, as the
        # run's data counters count them (`Layout.refusal`, `Encoded.cuts`).
        for reason in ("unfit", "options_alike"):
            refused = {name: counter.refused.get(reason, 0) for name, counter in dataset.counts.items()}
            refused["held out"] = sum(counter.refused.get(reason, 0)
                                      for counter in dataset.validation_counts.values())
            if any(refused.values()):
                summary[reason] = refused
        cut = {name: {kind: {"examples": counter.cut_examples[kind], "rows": rows,
                             "removed": counter.cut_removed[kind], "unit": cut_unit(kind)}
                      for kind, rows in counter.cut_rows.items()}
               for name, counter in {**dataset.counts, **{f"{name} (held out)": counter for name, counter
                                                           in dataset.validation_counts.items()}}.items()
               if counter.cut_rows}
        if cut:
            summary["cut"] = cut
        # The rows whose "none of the above" a training read declines where it draws it, by why their
        # layout refuses it (`FittedRow.declined`); each such read trains on the row as it was admitted.
        declined = {name: dict(counter.declined) for name, counter in dataset.counts.items()
                    if counter.declined}
        if declined:
            summary["none_of_the_above_declined"] = declined

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

    def _fresh_head(self, start, blocked: bool, budgets: Budgets) -> Mapping[str, object]:
        """The head a fresh backbone `start` reads and its rows' layout: the joint schema head, the
        letter readout of its own logits (`Letters`), its labels named by the backbone's tokenizer, or
        Laya's head, over the layout their rows take, in blocks where `blocked` (`_fresh_layout`)."""
        labels, head = None, None
        if isinstance(self.head, JointShape):
            width, head_dtype = DecisionModel.head_size(start.model)
            head = JointSchemaHead(hidden_size=width, dtype=head_dtype, **dataclasses.asdict(self.head))
        elif isinstance(self.head, Letters):
            # Its ids are its labels' under the objective's tokenizer, which lays the rows out too and
            # names the labels it spells as one token each where they name none (`LabelReadout.count`).
            labels, (width, head_dtype) = self.head.labels, DecisionModel.head_size(start.model)
            count = self.head.count if self.head.count is not None else self._options()
            head = LabelReadout(features=width, label_ids=(), untied=self.head.untied, dtype=head_dtype,
                                count=count or 0)
        # Lettered rows lie one question to a row, as the backbone's order asks.
        one_question = (DecisionHead if head is None else head).one_question or labels is not None
        layout = _fresh_layout(start.model.causal, one_question, budgets, blocked, labels)
        return {"layout": layout} if head is None else {"head": head, "layout": layout}

    def _capacities(self, objective: DecisionObjective, release: bool) -> Mapping[str, Mapping[str, object]]:
        """Each capacity the run's rows take and where it came from, beside the cuts it causes
        (`summary["cut"]`): the row's window in tokens, stated, the release's, or the layout's default;
        the context the cross-attention reads, where it reads one; and the labels a letter readout
        holds, stated, from the data's questions, or every label the tokenizer letters."""
        layout = objective.layout
        found: dict[str, dict[str, object]] = {"window": {
            "tokens": layout.max_len,
            "from": "stated" if self.max_len is not None else "release" if release else "default"}}
        if self.context_layers is not None:
            stated = self.context_length != type(self).context_length
            found["context"] = {"tokens": self.context_length, "from": "stated" if stated else "default"}
        # A lettered layout's labels are named by now, as its head reads them (`Head.labelled`).
        labels = layout.labels if isinstance(layout, QuestionLayout) else None
        named = None if labels is None else labels.names
        if isinstance(self.head, Letters) and named is not None:
            origin = ("stated" if self.head.count is not None else "data" if self._options() is not None
                      else "tokenizer")
            found["labels"] = {"count": len(named), "from": origin}
        return found

    def _options(self) -> int | None:
        """The most options a question of the run's data asks, as every dataset it reads states them
        (`slots`), training's and validation's, with "none of the above" where its objective adds it;
        None where any dataset leaves it unknown until it is read, so a split the run has not measured
        never takes fewer labels than it asks."""
        datasets = [self.chain(), *([] if self.validation is None else
                                    self.validation.values() if isinstance(self.validation, Mapping)
                                    else [self.validation])]
        stated = [record(dataset.facts().get("slots", {}), "slots").get("options")
                  if isinstance(dataset, Dataset) else None for dataset in datasets]
        if any(value is None for value in stated):
            return None
        widest = max(integer(value, "slots.options") for value in stated)
        added = number(self.objective.fields.get("none_of_the_above") or 0.0, "none_of_the_above") > 0
        return widest + int(added)

    def default_metrics(self, *, validates: bool) -> tuple[MetricConfig, ...]:
        """The task's own held-out suite (`default_metrics`), where the data holds held-out rows."""
        if not validates:
            return ()
        return tuple(MetricConfig(import_path(type(metric))) for metric in default_metrics())


type Budgets = dict[str, int | bool | str | Framing | Labels | tuple[RowSegment, ...] | None]
"""A run's budgets of a row's layout, by the layout's field names (`_budgeted`)."""


def _fresh_layout(causal: bool, one_question: bool, budgets: Budgets, blocked: bool = False,
                  labels: Labels | None = None) -> Layout:
    """The layout a fresh head reads (`default_layout`), with the run's `budgets`, its rows in blocks
    where its backbone's text reads each of a row's blocks whole, as the checkpoint's attention and the
    run's `--source.attention` declare it (`blocked`, `reads_blocks`), not where a checkpoint reads its
    own media so, and lettered by `labels` where the run reads letters, all set at once, so that a
    question row's segments read under the labels they need; a layout whose rows can do neither, a
    joint row of every question, is refused."""
    layout = default_layout(causal, one_question)
    stated = {**({"blocks": True} if blocked else {}), **({} if labels is None else {"labels": labels})}
    if stated and not isinstance(layout, QuestionLayout):
        raise ValueError(f"{type(layout).__name__} lays every question in one row, which lies in no blocks "
                         "and carries no labels; a block_bidirectional --source.attention and the letter "
                         "readout read a question to a row")
    return _budgeted(layout, {**budgets, **stated})


def _budgeted(layout: Layout, budgets: Budgets) -> Layout:
    """`layout` with the `budgets` the run states, each but those left None, which keep the
    layout's own; zero is a budget too (a joint row of no state). A budget the layout has no
    field for is refused: a joint row has no question budget, and a row per question no state
    budget, its state taking what is left."""
    stated = {name: value for name, value in budgets.items() if value is not None}
    if isinstance(layout, QuestionLayout) and "framing" in stated and (
            stated.get("segments", layout.segments) is None):
        raise ValueError("--framing frames a joint row or a question row of --segments; a question layout's "
                         "own row is written as its release writes it, unframed")
    fields = {field.name for field in dataclasses.fields(layout)}
    unknown = sorted(f"--{name.replace('_', '-')}" for name in stated if name not in fields)
    if unknown:
        raise ValueError(f"{', '.join(unknown)} budget a row that {type(layout).__name__} does not lay out")
    return dataclasses.replace(layout, **stated)


__all__ = ["DecisionRunConfig", "JointShape", "Letters"]
