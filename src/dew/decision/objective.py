"""`DecisionObjective`: training a decision model on labelled questions."""

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.artifacts import Decisions
from dew.batch import Batch, Variables
from dew.data.pipeline import Configured, ObjectiveData, Pack, Rejected
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.calibration import Temperatures
from dew.decision.data import Example
from dew.decision.head import DecisionHead, Head
from dew.decision.layout import (
    OPTION_TILE,
    Bounds,
    DecisionInputs,
    Layout,
    QuestionLayout,
    Specials,
    compacted,
    default_layout,
    question_fields,
)
from dew.decision.model import DecisionModel
from dew.decision.questions import KINDS, Choice, Question, Score
from dew.decision.scoring import LogLoss, ScoringRule
from dew.decision.task import Decide, DecideSettings, Weights, laid_out
from dew.inference.pipeline import RunProcessor
from dew.inference.tasks import Processor as TaskProcessor
from dew.inputs import Field, InputSpec
from dew.interop.processors import Processor
from dew.lora import Adapter
from dew.nn.flops import MissingFlopsCount, counted_flops
from dew.nn.inputs import ModelInputs
from dew.nn.protocols import CountedDecisions, CountedTraining, TokenModel
from dew.objectives.base import (
    OMITTED,
    Aux,
    Objective,
    Omitted,
    ProgramModule,
    Ratio,
    Source,
    Step,
    joined,
    model_rngs,
    part,
)
from dew.records import integer, record

if TYPE_CHECKING:
    from dew.train_state import TrainState

NONE_OF_THE_ABOVE = "none of the above"

_SCORE = KINDS.index(Score)


@dataclass(frozen=True)
class Encoding:
    """How a labelled example becomes the row a step packs and trains on.

    A row is its own tokens, at most `max_len`, one slot per question it lays
    out and `width` option slots each. A layout of one row per question lays
    out one answered question; a joint layout lays out every question of the
    example, scoring the answered ones. With `shuffle`, a choice's options are
    laid out in a fresh random order every time the row is read (Laya's
    `--shuffle-options`), while a score's levels and a noul's two answers keep
    their order, and a joint row asks its questions in a fresh order too.
    With `none_of_the_above` p, a choice with a single right answer gains a
    "none of the above" option with probability p, and half of those also
    lose their right option, which makes "none of the above" the right
    answer. That teaches the model to say
    that no option fits instead of picking the nearest wrong one. Each scored
    slot's target is its gold distribution (`Example.distribution`) in the
    order the row shows the options. A reader passes the state's `tokens` it
    fitted the example with (`Fit`), which the row lays out again.
    """

    layout: Layout
    tokenizer: Tokenizer
    specials: Specials
    width: int
    shuffle: bool = True
    none_of_the_above: float = 0.0

    def __call__(self, example: Example, names: Sequence[str], rng: np.random.Generator | None,
                 tokens: np.ndarray | None = None) -> Batch:
        labels = example.labels()
        if rng is not None and self.shuffle and self.layout.joint:
            names = [names[int(index)] for index in rng.permutation(len(names))]
        asked: dict[str, Question] = {}
        golds: dict[str, np.ndarray] = {}
        for name in names:
            question = example.questions[name]
            gold = example.distribution(name) if name in labels else None
            if (
                name in example.answers
                and isinstance(question, Choice)
                and rng is not None
                and rng.random() < self.none_of_the_above
            ):
                question, label = _none_of_the_above(question, labels[name], drop=rng.random() < 0.5)
                gold = np.eye(len(question.options))[label]
            asked[name] = question
            if gold is not None:
                golds[name] = gold
        orders = (
            {
                name: [int(slot) for slot in rng.permutation(len(question.options))]
                for name, question in asked.items()
                if isinstance(question, Choice)
            }
            if rng is not None and self.shuffle
            else None
        )
        rows = self.layout.rows(self.tokenizer, self.specials, example.state, asked, orders=orders,
                                tokens=None if tokens is None else tokens.tolist())
        if len(rows) != 1:
            raise ValueError(f"an encoding lays out one row, and the layout gave {len(rows)}")
        encoded = rows[0]
        length = self.layout.max_len
        if not self.layout.fits(encoded):
            raise ValueError(f"the row needs {len(encoded.tokens)} tokens, more than max_len={length}")
        for laid in encoded.questions:
            count = len(asked[laid.name].options)
            if len(laid.options) != count:
                raise ValueError(
                    f"{count - len(laid.options)} of question {laid.name!r}'s {count} options "
                    f"fall past max_len={length}; raise it or shorten the question"
                )
            if count > self.width:
                raise ValueError(f"a question of {count} options outgrows the encoding's {self.width} slots")
        questions = len(encoded.questions)
        row: Batch = {
            "tokens": np.asarray(encoded.tokens, np.int32),
            **question_fields(encoded, questions, self.width),
            "labels": np.zeros(questions, np.int32),
            "scored": np.zeros(questions, bool),
            "targets": np.zeros((questions, self.width), np.float32),
        }
        for slot, laid in enumerate(encoded.questions):
            count = len(laid.options)
            if laid.name in golds:
                shown = golds[laid.name][list(laid.order)]
                row["targets"][slot, :count] = shown
                row["labels"][slot] = int(np.argmax(shown))
                row["scored"][slot] = True
        if encoded.positions is not None and encoded.slots is not None:
            row["positions"] = np.asarray(encoded.positions, np.int32)
            row["slots"] = np.asarray(encoded.slots, np.int32)
        return row


def _none_of_the_above(question: Choice, label: int, *, drop: bool) -> tuple[Choice, int]:
    """`question` with a "none of the above" option last, and without its
    right option when `drop`; the new label."""
    described = dict(zip(question.options, question.descriptions, strict=True))
    right = question.options[label]
    if drop:
        described.pop(right)
    described[NONE_OF_THE_ABOVE] = "none of the other options fits"
    options = list(described)
    return Choice(question.instructions, described), options.index(NONE_OF_THE_ABOVE if drop else right)


def _tokenizer_of(processor: TaskProcessor | None) -> Tokenizer | None:
    """The tokenizer a loaded model's text processor encodes with: a run's
    own, or a checkpoint's Hugging Face tokenizer, alone or inside a
    multimodal processor, read from the directory it was loaded from, at the
    revision the model was. A processor without a tokenizer gives none, and
    `tokenizer=` names one."""
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

    match processor:
        case RunProcessor():
            return processor.tokenizer
        case (
            Processor(reference=PreTrainedTokenizerBase() as tokenizer)
            | Processor(reference=ProcessorMixin(tokenizer=PreTrainedTokenizerBase() as tokenizer))
        ):
            return HFTokenizer(str(tokenizer.name_or_path), local_files_only=True)
        case _:
            return None


@dataclass(frozen=True)
class FittedRow:
    example: Example
    names: tuple[str, ...]
    length: int
    truncated: bool
    tokens: np.ndarray
    """The state's tokens, which the encoding lays out again, as int32: a read
    held in memory keeps them for every example."""


@dataclass(frozen=True)
class Fit(pygrain.MapTransform):
    layout: Layout
    tokenizer: Tokenizer
    specials: Specials
    head: Head
    rejects = True

    def map(self, element: Batch) -> Batch | Rejected:
        example = Example.of(element.get("example", element))
        labels = example.labels()
        if not labels:
            return Rejected("unlabelled")
        asked = example.questions if self.layout.joint else {name: example.questions[name] for name in labels}
        state = self.layout.state(self.tokenizer, self.specials, example.state)
        laid = self.layout.rows(self.tokenizer, self.specials, example.state, asked, tokens=state)
        if not all(map(self.layout.fits, laid)):
            return Rejected("unfit")
        self.head.validate_rows(laid)
        tokens = np.asarray(state, np.int32)
        fitted = tuple(
            FittedRow(
                example,
                tuple(question.name for question in row.questions),
                len(row.tokens),
                row.state_kept < row.state_tokens,
                tokens,
            )
            for row in laid
        )
        return (
            {
                "decision": fitted[0],
                "_dew_length": fitted[0].length,
                "_dew_truncated": fitted[0].truncated,
                "_dew_bounds": {
                    "slots": {
                        "questions": len(asked),
                        "options": max(len(question.options) for question in asked.values()),
                    }
                },
            }
            if self.layout.joint
            else {"decision_rows": fitted}
        )


@dataclass(frozen=True)
class QuestionRows:
    def flat_map(self, element: Batch) -> Sequence[Batch]:
        return tuple(
            {"decision": row, "_dew_length": row.length, "_dew_truncated": row.truncated}
            | {
                "_dew_bounds": {
                    "slots": {
                        "questions": 1,
                        "options": max(len(row.example.questions[name].options) for name in row.names),
                    }
                }
            }
            for row in element["decision_rows"]
        )


def _fitted(element: Batch) -> FittedRow:
    fitted = element["decision"]
    if not isinstance(fitted, FittedRow):
        raise ValueError("decision encoding needs the fitted decision row")
    return fitted


@dataclass(frozen=True)
class LayoutRows(pygrain.RandomMapTransform):
    encoding: Encoding

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        fitted = _fitted(element)
        return self.encoding(fitted.example, fitted.names, rng, fitted.tokens)


@dataclass(frozen=True)
class EvaluateRows(pygrain.MapTransform):
    encoding: Encoding

    def map(self, element: Batch) -> Batch:
        fitted = _fitted(element)
        return self.encoding(fitted.example, fitted.names, None, fitted.tokens)


@dataclass(frozen=True)
class ConfigureLayout:
    layout: Layout
    tokenizer: Tokenizer
    specials: Specials
    shuffle: bool
    none_of_the_above: float
    train: bool

    def __call__(self, metadata: Mapping[str, object]) -> LayoutRows | EvaluateRows:
        slots = record(metadata["slots"], "decision slots")
        width = integer(slots.get("options"), "slots.options") + int(self.none_of_the_above > 0)
        if width < 1:
            raise ValueError("decision source slot maxima must be positive")
        encoding = Encoding(self.layout, self.tokenizer, self.specials, width, self.shuffle,
                            self.none_of_the_above)
        return LayoutRows(encoding) if self.train else EvaluateRows(encoding)


_ALONG_THE_QUESTIONS = ("kinds", "questions", "spans", "option_spans", "options", "labels", "scored",
                        "targets")
"""The fields of an encoded row that run along its questions."""


def packing(layout: Layout, bounds: Bounds) -> Pack:
    """Rows of `layout` packed first-fit into windows of its `max_len`, each
    window's rows, questions and options within `bounds` (`Layout.bounds`)."""
    parallel = isinstance(layout, QuestionLayout) and layout.parallel
    along = tuple((name, bounds.questions) for name in _ALONG_THE_QUESTIONS)
    return Pack(layout.max_len, ("tokens", "positions", "slots") if parallel else ("tokens",),
                segments=bounds.segments, along=along)


def _rows(batch: Batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each packed row's tokens, questions and options, window by window."""
    segments = np.asarray(batch["segments"])
    owners = np.where(batch["questions"],
                      np.take_along_axis(segments, np.asarray(batch["spans"])[..., 0], axis=1), 0)
    options = np.sum(np.asarray(batch["options"], bool), axis=-1)
    counted = []
    for window, owner, option in zip(segments, owners, options, strict=True):
        rows = int(window.max()) + 1
        counted.append(np.stack([np.bincount(window, minlength=rows), np.bincount(owner, minlength=rows),
                                 np.bincount(owner, weights=option, minlength=rows).astype(np.int64)])[:, 1:])
    lengths, questions, real = np.concatenate(counted, axis=1)
    return lengths, questions, real


@dataclass(frozen=True)
class PackedRows(pygrain.MapTransform):
    """A window of packed rows (`packing`) as `DecisionInputs` reads it: each
    row's spans moved to where it starts in the window, and with
    `option_slots` the window's real options compacted to that many slots
    (`compacted`). Every window is `max_length` tokens, which a token budget
    divides into rows."""

    max_length: int
    option_slots: int | None = None

    def map(self, element: Batch) -> Batch:
        segments = np.asarray(element["tokens_segment_ids"])
        owners = np.asarray(element["kinds_segment_ids"])
        starts = np.concatenate([[0], np.cumsum(np.bincount(segments)[1:])]).astype(np.int32)
        moved = np.where(owners > 0, starts[np.maximum(owners - 1, 0)], 0)
        options = np.asarray(element["options"])
        laid = {name: element[name] for name in ("tokens", *_ALONG_THE_QUESTIONS)} | {
            "segments": segments,
            "positions": element["positions"] if "slots" in element else element["tokens_positions"],
            "spans": element["spans"] + moved[:, None],
            "option_spans": element["option_spans"] + (moved[:, None] * options)[..., None],
        }
        if "slots" in element:
            laid["slots"] = element["slots"]
        if self.option_slots is not None:
            laid["option_slots"], laid["slotted"] = compacted(options, self.option_slots)
        return laid


class DecisionObjective(Objective[Ratio]):
    """Trains a decision model on questions with known answers.

    The model is any backbone with final states (`dew.nn.protocols.HiddenStates`),
    read by a head, Laya's `DecisionHead` unless `head` says otherwise.
    `backbone` is a model built from scratch, a loaded model
    (`Pretrained.load`, adapted with LoRA or partly frozen through `freeze`, as
    `LMObjective` accepts one), or a whole `Decide` task, such as Laya's released
    checkpoint, which brings its head and layout with it. Unless `layout` says
    otherwise, a head that reads several questions a row reads them in one
    row (`JointLayout`), and one that reads one question a row reads Laya's
    `MarkerLayout` over a bidirectional backbone and the `StateFirstLayout`
    over a causal one (`default_layout`). `tokenizer` and `specials` default
    to the loaded model's.

    The loss is `loss`, a proper scoring rule or a sum of them, computed against
    each row's right option, with `label_smoothing` spread evenly over its
    options. `shuffle_options` and `none_of_the_above` augment the rows as
    `Encoding` describes, and `data` supplies their pipeline steps. The
    training stream and the validation pass pack rows first-fit into windows
    of the layout's `max_len` tokens (`packing`), as many rows, questions and
    options as its `bounds` let a window hold, so a step pays for its tokens
    rather than `max_len` per row, and every batch has one shape. A window
    holds at most `segments` rows, which the backbone's sequence kernel then
    works on a window (`Coordinates.max_segments`) whether or not they come:
    16 by default, a few more than a window of mixed tasks' rows was measured
    to hold, where a window full of the shortest rows would declare tens. A
    mix of rows shorter than `max_len / segments` tokens fills its windows
    less under it and takes a larger one; None lets a window hold as many as
    the layout's shortest row fits. The loss is the mean over the batch's
    scored questions.

    Evaluation returns the model's `Decisions` on the validation rows, which
    `Accuracy`, `ECE`, `AURC` and the scoring rules read, its probabilities
    divided by `temperatures` (none by default: a `Decide`'s calibration fits
    its logits before training moves them, so it is not inherited). The saved
    run loads as a `Decide` task.
    """

    saved_task = Decide
    settings = DecideSettings
    _held_parts = True

    counts_tokens = True

    def real_lengths(self, batch: Batch) -> np.ndarray:
        """Each packed row's own length, not its window's."""
        return _rows(batch)[0]

    def model_flops(self, model, batch: Batch) -> float:
        """One backbone pass plus the head's real question/option geometry. Forming the vocabulary
        rows a head reads (an adapter's update merged into them) is execution, as recompute is."""
        model = self.model if model is None else model
        return counted_flops(model.backbone, self.real_lengths(batch)) + self._head_flops(model, batch)

    def _head_flops(self, model, batch: Batch) -> float:
        """Forward and training work read the same layout geometry and head capability."""
        tokens = batch.get("tokens")
        if isinstance(tokens, ModelInputs) and tokens.conditioning:
            raise MissingFlopsCount("DecisionObjective needs its image tower's multi-sequence geometry")
        if not isinstance(model.head, CountedDecisions):
            raise MissingFlopsCount(f"{type(model.head).__name__} has no declared decision_flops")
        return model.head.decision_flops(*_rows(batch))

    def training_flops(self, model, batch: Batch) -> float:
        """The adapted frozen/trainable backbone plus the trained decision head."""
        model = self.model if model is None else model
        if not isinstance(model.backbone, CountedTraining):
            return super().training_flops(model, batch)
        return (model.backbone.training_flops(self.real_lengths(batch))
                + 3 * self._head_flops(model, batch))

    def __init__(self, backbone: nn.Module | Source | Adapter | Decide, *,
                 loss: ScoringRule | None = None, tokenizer: Tokenizer | None = None,
                 specials: Specials | None = None, layout: Layout | None = None,
                 head: Head | None = None, variables: Variables | None | Omitted = OMITTED,
                 label_smoothing: float = 0.0, shuffle_options: bool = True,
                 none_of_the_above: float = 0.0, temperatures: Temperatures | None = None,
                 segments: int | None = 16):
        held: Variables | None = None
        self.image_processor = None
        match backbone:
            case Decide():
                model, held = backbone.model, backbone.variables
                self.image_processor = backbone.processor
                head = head or model.head
                layout = layout or backbone.layout
                tokenizer = tokenizer or backbone.tokenizer
                specials = specials or backbone.specials
                module = model.backbone
            case Adapter() | Source():
                module, held = backbone.model, joined({"backbone": backbone.variables})
                if isinstance(backbone, Source):
                    tokenizer = tokenizer or _tokenizer_of(backbone.text_processor)
            case _:
                module = backbone
        if variables is not OMITTED:
            held = variables
        if tokenizer is None:
            raise ValueError("a decision model lays its rows out with a tokenizer; pass tokenizer=")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing is a share of the target, in [0, 1)")
        if not 0.0 <= none_of_the_above <= 1.0:
            raise ValueError("none_of_the_above is a probability")
        if segments is not None and segments < 1:
            raise ValueError(f"a window holds at least one row; got segments={segments}")
        if head is None:
            width, dtype = DecisionModel.head_size(module)
            head = DecisionHead(width, dtype=dtype)
        self.model = DecisionModel(module, head)
        if layout is None:
            if not isinstance(module, TokenModel):
                raise ValueError(
                    f"a {type(module).__name__} does not say whether it reads its tokens in "
                    "order; pass layout= (StateFirstLayout if it does, MarkerLayout if not)"
                )
            layout = default_layout(module.causal, head.one_question)
        self.layout = layout
        self.tokenizer = tokenizer
        self.specials = specials or Specials.of(tokenizer)
        self.loss_rule = LogLoss() if loss is None else loss
        self.variables = held
        self.label_smoothing = label_smoothing
        self.shuffle_options = shuffle_options
        self.none_of_the_above = none_of_the_above
        self.temperatures = temperatures
        self.segments = segments
        self.inputs = InputSpec(sample=Field("tokens", (self.layout.max_len,)))

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The backbone, then the head, both trained."""
        return (
            ProgramModule(self.model.backbone, None, trained=True),
            ProgramModule(self.model.head, None, trained=True),
        )

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        backbone, head = modules
        self.model = dataclasses.replace(self.model, backbone=backbone, head=head)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        """Nothing beyond the held parts: `complete_variables` draws the part the tree lacks."""
        return held or {}

    def complete_variables(self, key: jax.Array, tree: Variables) -> Variables:
        """Draw whichever of the backbone and the head `tree` does not hold."""
        backbone_key, head_key = jax.random.split(key)
        tokens = jnp.zeros((1, self.layout.max_len), jnp.int32)
        parts = {
            name: part(tree, name)
            for name in ("backbone", "head")
            if any(name in collection for collection in tree.values())
        }
        if "backbone" not in parts:
            parts["backbone"] = self.model.backbone.init(backbone_key, tokens)
        if "head" not in parts:
            # The width the head was built for (its parameters read the states' width, not their dtype),
            # not a trace of the backbone inside every trace of the state.
            head = self.model.head
            states = jnp.zeros((1, self.layout.max_len, getattr(head, head.width_field)), jnp.float32)
            one = jnp.ones((1, 1), bool)
            inputs = DecisionInputs(
                tokens=tokens,
                segments=jnp.ones_like(tokens),
                kinds=jnp.zeros((1, 1), jnp.int32),
                questions=one,
                spans=jnp.asarray([[[0, 1]]]),
                option_spans=jnp.asarray([[[[0, 1]]]]),
                options=one[..., None],
            )
            table = (
                self.model.table(joined({"backbone": parts["backbone"]}))
                if self.model.head.reads_table
                else None
            )
            parts["head"] = self.model.head.init(head_key, states, inputs, table)
        return joined(parts)

    @functools.cached_property
    def bounds(self) -> Bounds:
        """The most rows, questions and options a window packs (`Layout.bounds`),
        at most `segments` rows."""
        bounds = self.layout.bounds(self.tokenizer, self.specials)
        if self.segments is None:
            return bounds
        return dataclasses.replace(bounds, segments=min(self.segments, bounds.segments))

    def loss(self, variables: Variables, batch: Batch, step: Step):
        inputs = laid_out(batch, self.bounds.segments)
        logits = self.model.logits(
            variables, inputs, train=step.training, rngs=model_rngs(step.key, training=step.training)
        )
        return self._charged(logits, inputs, batch)

    def _charged(self, logits: jax.Array, inputs: DecisionInputs, batch: Batch) -> tuple[Ratio, Aux]:
        """The loss and its accuracy report over the questions' option logits."""
        options, scored = inputs.options, jnp.asarray(batch["scored"]).astype(jnp.float32)
        # A padding question has no options; it is scored nothing over a
        # finite, even distribution, so no NaN reaches its gradient.
        empty = ~jnp.any(options, axis=-1, keepdims=True)
        logits, options = jnp.where(empty, 0.0, logits), options | empty
        count = jnp.sum(options, axis=-1, keepdims=True)
        gold = jnp.asarray(batch["targets"])
        target = jnp.where(options, (1.0 - self.label_smoothing) * gold + self.label_smoothing / count, 0.0)
        charge = self.loss_rule.charge(logits, target, options, inputs.kinds == _SCORE)
        correct = (jnp.argmax(logits, axis=-1) == batch["labels"]).astype(jnp.float32)
        accuracy, _ = self.accuracy(correct, batch, scored).mean()
        loss = Ratio(self.row_mean(charge * scored, batch).total, self.row_mean(scored, batch).total)
        return loss, Aux(metrics={"accuracy": accuracy})

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Decisions:
        inputs = laid_out(batch, self.bounds.segments)
        return self._decisions(self._logits(self.evaluation_variables(params, step), inputs), inputs, batch)

    def _validation_scores(self, params: Variables, batch: Batch, step: Step) -> tuple[Ratio, Decisions]:
        inputs = laid_out(batch, self.bounds.segments)
        logits = self.model.logits(params, inputs)
        return self._charged(logits, inputs, batch)[0], self._decisions(logits, inputs, batch)

    def _decisions(self, logits: jax.Array, inputs: DecisionInputs, batch: Batch) -> Decisions:
        """The questions' option probabilities, each at its type's temperature, with what scores them."""
        if self.temperatures is not None:
            # Each question's temperature by its type and its count of real
            # options, as `Temperatures.of` gives it; a padding question has none.
            width = logits.shape[-1]
            table = np.asarray(
                [
                    [
                        1.0 if count == 0 else self.temperatures.of(kind.kind, count)
                        for count in range(width + 1)
                    ]
                    for kind in KINDS
                ],
                np.float32,
            )
            logits = logits / jnp.asarray(table)[inputs.kinds, jnp.sum(inputs.options, axis=-1)][..., None]
        empty = ~jnp.any(inputs.options, axis=-1, keepdims=True)
        probabilities = jax.nn.softmax(jnp.where(empty, 0.0, logits), axis=-1)
        return Decisions(
            probabilities=probabilities,
            options=inputs.options,
            labels=batch["labels"],
            targets=batch["targets"],
            ordinal=inputs.kinds == _SCORE,
            scored=batch["scored"],
        )

    @property
    def _logits(self):
        return self.program('logits', lambda variables, inputs: self.model.logits(variables, inputs))

    def data(self, tokenizer: Tokenizer | None = None) -> ObjectiveData:
        """Fit rows before mixing, augment training layouts, and pack training
        and evaluation rows alike, by the objective's own tokenizer, which its
        `bounds` count in."""
        if tokenizer not in (None, self.tokenizer):
            raise ValueError("a decision objective lays its rows out with its own tokenizer; "
                             "pass tokenizer= when building it")
        training = ConfigureLayout(self.layout, self.tokenizer, self.specials, self.shuffle_options,
                                   self.none_of_the_above, train=True)
        evaluating = dataclasses.replace(training, train=False)
        fit = Fit(self.layout, self.tokenizer, self.specials, self.model.head)
        expand = () if self.layout.joint else (QuestionRows(),)
        options = self.bounds.options if self.layout.joint else None
        compact = None if options is None else -(-options // OPTION_TILE) * OPTION_TILE
        return ObjectiveData(
            train=(fit, *expand, Configured(training, needs=("slots",), scope="run", random=True)),
            evaluate=(fit, *expand, Configured(evaluating, needs=("slots",), scope="run")),
            pack=packing(self.layout, self.bounds),
            packed=(PackedRows(self.layout.max_len, compact),),
        )

    def task_settings(self) -> DecideSettings:
        """What a saved run rebuilds its task from beside the backbone.

        That is the head, layout, tokenizer, special tokens and the image
        processor's source, independently of the text tokenizer. Interop
        reloads that source's processor files and derives its media geometry
        from the source config; the decision record owns neither format.
        """
        from dew.inference.tasks import recorded_tokenizer
        from dew.registry import to_record

        return DecideSettings(
            self.model.head.record(),
            {"name": type(self.layout).__name__, "fields": to_record(self.layout, type(self.layout))},
            self.specials, recorded_tokenizer(RunProcessor(self.tokenizer)),
            # A decision task reads text through its own tokenizer, so only a media processor is kept.
            None if self.image_processor is None or self.image_processor.directory is None
            else self.image_processor.source)

    def recorded_model(self) -> nn.Module:
        """The backbone; the head is a task setting of its own."""
        return self.model.backbone

    def pipeline(
        self,
        state: "TrainState",
        *,
        ema: bool | None = None,
        processor: TaskProcessor | None | Omitted = OMITTED,
    ) -> Decide:
        """Return the trained model as a `Decide` task over the state's weights.

        A `Decide` encodes with the objective's own tokenizer and layout, so it
        takes no processor; it reads images with the backbone's own processor,
        which a task the objective fine-tunes brings.
        """
        if processor is not OMITTED:
            raise TypeError("a Decide task encodes with the objective's tokenizer and takes no processor")
        averaged = not (ema is False or (ema is None and state.ema is None))
        return Decide(
            self.model,
            self._pipeline_weights(state, ema),
            self.layout,
            self.tokenizer,
            self.specials,
            weights=Weights(int(state.step), averaged),
            processor=self.image_processor,
        )
