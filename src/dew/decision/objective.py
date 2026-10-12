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
from dew.data.dataset import LengthGroups
from dew.data.pipeline import Configured, ObjectiveData, Rejected, TokenField
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.calibration import Temperatures
from dew.decision.data import Example
from dew.decision.head import DecisionHead, Head
from dew.decision.layout import DecisionInputs, Layout, MarkerLayout, Specials, StateFirstLayout, compacted
from dew.decision.model import DecisionModel
from dew.decision.questions import KINDS, Choice, Question, Score
from dew.decision.scoring import LogLoss, ScoringRule
from dew.decision.task import Decide, Weights, laid_out
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
    Batch,
    Objective,
    Omitted,
    ProgramModule,
    Ratio,
    Source,
    Step,
    Variables,
    joined,
    model_rngs,
    part,
)
from dew.records import integer, record
from dew.registry import import_path

if TYPE_CHECKING:
    from dew.training.state import TrainState

NONE_OF_THE_ABOVE = "none of the above"

_SCORE = KINDS.index(Score)


@dataclass(frozen=True)
class Encoding:
    """How a labelled example becomes the fixed-size row a step trains on.

    Every row is `max_len` tokens, `questions` question slots and `width` option
    slots. A layout of one row per question lays out one answered question; a
    joint layout lays out every question of the example, scoring the answered
    ones. With `shuffle`, a choice's options are laid out in a fresh random order
    every time the row is read (Laya's `--shuffle-options`), while a score's
    levels and a noul's two answers keep their order, and a joint row asks its
    questions in a fresh order too, as Clef's training permutes field orders.
    With `none_of_the_above` p, a choice with a single right answer gains a
    "none of the above" option with probability p, and half of those also lose
    their right option, which makes "none of the above" the right answer. That
    teaches the model to say that no option fits instead of picking the nearest
    wrong one. Each scored slot's target is its gold distribution (`Example.distribution`)
    in the order the row shows the options. With `option_slots`, a row also
    carries its real options compacted to that many slots (`compacted`), which
    a joint head computes alone.
    """

    layout: Layout
    tokenizer: Tokenizer
    specials: Specials
    width: int
    questions: int = 1
    shuffle: bool = True
    none_of_the_above: float = 0.0
    option_slots: int | None = None

    def __call__(self, example: Example, names: Sequence[str], rng: np.random.Generator | None) -> Batch:
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
        rows = self.layout.rows(self.tokenizer, self.specials, example.state, asked, orders=orders)
        if len(rows) != 1 or len(rows[0].questions) > self.questions:
            raise ValueError(
                f"an encoding lays out one row of at most {self.questions} questions, "
                f"and the layout gave {len(rows)} rows"
            )
        encoded = rows[0]
        length = self.layout.max_len
        if not self.layout.fits(encoded):
            raise ValueError(f"the row needs {len(encoded.tokens)} tokens, more than max_len={length}")
        row: Batch = {
            "tokens": _padded(encoded.tokens, length, self.specials.pad),
            "valid": _padded((True,) * len(encoded.tokens), length, fill=False),
            "kinds": np.zeros(self.questions, np.int32),
            "questions": np.zeros(self.questions, bool),
            "spans": np.zeros((self.questions, 2), np.int32),
            "option_spans": np.zeros((self.questions, self.width, 2), np.int32),
            "options": np.zeros((self.questions, self.width), bool),
            "labels": np.zeros(self.questions, np.int32),
            "scored": np.zeros(self.questions, bool),
            "targets": np.zeros((self.questions, self.width), np.float32),
        }
        for slot, laid in enumerate(encoded.questions):
            count = len(asked[laid.name].options)
            if len(laid.options) != count:
                raise ValueError(
                    f"{count - len(laid.options)} of question {laid.name!r}'s {count} options "
                    f"fall past max_len={length}; raise it or shorten the question"
                )
            if count > self.width:
                raise ValueError(f"a question of {count} options outgrows the encoding's {self.width} slots")
            row["kinds"][slot], row["questions"][slot], row["spans"][slot] = laid.kind, True, laid.span
            row["option_spans"][slot, :count] = laid.options
            row["options"][slot, :count] = True
            if laid.name in golds:
                shown = golds[laid.name][list(laid.order)]
                row["targets"][slot, :count] = shown
                row["labels"][slot] = int(np.argmax(shown))
                row["scored"][slot] = True
        if encoded.positions is not None and encoded.slots is not None:
            row["positions"] = _padded(encoded.positions, length, 0)
            row["slots"] = _padded(encoded.slots, length, 0)
        if self.option_slots is not None:
            row["option_slots"], row["slotted"] = compacted(row["options"], self.option_slots)
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


_ALONG_THE_ROW = ("tokens", "valid", "positions", "slots")
"""The fields of an encoded row that run along its tokens."""


@dataclass(frozen=True)
class _Cut:
    """A batch's fields along its rows cut after its longest row, rounded up to
    a multiple of `bucket` and at most `length`: the padding no row reaches."""

    bucket: int
    length: int
    single_reader = True

    def __call__(self, batch: Batch) -> Batch:
        longest = int(np.max(np.sum(batch["valid"], axis=1)))
        size = min(self.length, -(-longest // self.bucket) * self.bucket)
        return {name: value[:, :size] if name in _ALONG_THE_ROW else value for name, value in batch.items()}


def _padded(values: Sequence[int | bool], length: int, fill: int | bool = 0) -> np.ndarray:
    dtype = np.bool_ if isinstance(fill, bool) else np.int32
    row = np.full((length,), fill, dtype)
    row[: len(values)] = values
    return row


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


@dataclass(frozen=True)
class Fit(pygrain.MapTransform):
    layout: Layout
    tokenizer: Tokenizer
    specials: Specials
    rejects = True

    def map(self, element: Batch) -> Batch | Rejected:
        example = Example.of(element.get("example", element))
        labels = example.labels()
        if not labels:
            return Rejected("unlabelled")
        asked = example.questions if self.layout.joint else {name: example.questions[name] for name in labels}
        laid = self.layout.rows(self.tokenizer, self.specials, example.state, asked)
        if not all(map(self.layout.fits, laid)):
            return Rejected("unfit")
        fitted = tuple(
            FittedRow(
                example,
                tuple(question.name for question in row.questions),
                len(row.tokens),
                row.state_kept < row.state_tokens,
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
                        "total_options": sum(len(question.options) for question in asked.values()),
                        "choice_questions": sum(isinstance(question, Choice) for question in asked.values()),
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
                        "total_options": sum(len(row.example.questions[name].options) for name in row.names),
                        "choice_questions": sum(
                            isinstance(row.example.questions[name], Choice) for name in row.names
                        ),
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
        encoded = self.encoding(fitted.example, fitted.names, rng)
        return {**encoded, self.length_field: np.int32(np.sum(encoded["valid"]))}

    @property
    def max_length(self) -> int:
        return self.encoding.layout.max_len

    @property
    def token_fields(self) -> tuple[TokenField, ...]:
        return tuple(
            TokenField(name, pad=self.encoding.specials.pad if name == "tokens" else 0)
            for name in _ALONG_THE_ROW
        )

    @property
    def length_field(self) -> str:
        return "_dew_length"


@dataclass(frozen=True)
class EvaluateRows(pygrain.MapTransform):
    encoding: Encoding

    def map(self, element: Batch) -> Batch:
        fitted = _fitted(element)
        encoded = self.encoding(fitted.example, fitted.names, None)
        return {**encoded, "_dew_length": np.int32(np.sum(encoded["valid"]))}


@dataclass(frozen=True)
class StoredLength:
    def __call__(self, element: Batch) -> int:
        return _fitted(element).length if "decision" in element else int(np.sum(element["valid"]))


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
        questions = integer(slots.get("questions"), "slots.questions") if self.layout.joint else 1
        if min(width, questions) < 1:
            raise ValueError("decision source slot maxima must be positive")
        encoding = Encoding(
            self.layout, self.tokenizer, self.specials, width, questions, self.shuffle, self.none_of_the_above
        )
        total = slots.get("total_options")
        if total is not None:
            count = integer(total, "slots.total_options")
            if self.none_of_the_above > 0:
                count += integer(slots.get("choice_questions"), "slots.choice_questions")
            capacity = -(-count // 128) * 128
            if capacity < questions * width:
                encoding = dataclasses.replace(encoding, option_slots=capacity)
        return LayoutRows(encoding) if self.train else EvaluateRows(encoding)


class DecisionObjective(Objective[Ratio]):
    """Trains a decision model on questions with known answers.

    The model is any backbone with final states (`dew.nn.protocols.HiddenStates`),
    read by a head, Laya's `DecisionHead` unless `head` says otherwise.
    `backbone` is a model built from scratch, a loaded model
    (`Pretrained.load`, adapted with LoRA or partly frozen through `freeze`, as
    `LMObjective` accepts one), or a whole `Decide` task, such as Laya's released
    checkpoint, which brings its head and layout with it. A bidirectional
    backbone reads Laya's `MarkerLayout` and a causal one the `StateFirstLayout`,
    unless `layout` says otherwise. `tokenizer` and `specials` default to the
    loaded model's.

    The loss is `loss`, a proper scoring rule or a sum of them, computed against
    each row's right option, with `label_smoothing` spread evenly over its
    options. `shuffle_options` and `none_of_the_above` augment the rows as
    `Encoding` describes, and `data` supplies their pipeline steps. With `bucket`, the
    training stream and the validation pass batch rows of like length
    (`LengthGroups`) and cut each batch after its longest row, rounded up to a
    multiple of `bucket`, so a step pays for little more than its tokens
    instead of `max_len` per row; each length compiles its own step, and both
    are read on one process.

    Evaluation returns the model's `Decisions` on the validation rows, which
    `Accuracy`, `ECE`, `AURC` and the scoring rules read, its probabilities
    divided by `temperatures` (none by default: a `Decide`'s calibration fits
    its logits before training moves them, so it is not inherited). The saved
    run loads as a `Decide` task.
    """

    artifact = Decisions
    saved_task = Decide
    _held_parts = True

    counts_tokens = True

    def real_lengths(self, batch: Batch) -> np.ndarray:
        """The layout's unpadded row lengths, not its bucket or maximum length."""
        return np.sum(np.asarray(batch["valid"], bool), axis=-1, dtype=np.int64)

    def model_flops(self, model, batch: Batch) -> float:
        """One backbone pass plus the head's real question/option geometry."""
        model = self.model if model is None else model
        return counted_flops(model.backbone, self.real_lengths(batch)) + self._head_flops(model, batch)

    def _head_flops(self, model, batch: Batch) -> float:
        """Forward and training work read the same layout geometry and head capability."""
        tokens = batch.get("tokens")
        if isinstance(tokens, ModelInputs) and tokens.conditioning:
            raise MissingFlopsCount("DecisionObjective needs its image tower's multi-sequence geometry")
        if not isinstance(model.head, CountedDecisions):
            raise MissingFlopsCount(f"{type(model.head).__name__} has no declared decision_flops")
        lengths = self.real_lengths(batch)
        questions = np.sum(np.asarray(batch["questions"], bool), axis=-1, dtype=np.int64)
        options = np.sum(np.asarray(batch["options"], bool), axis=(-2, -1), dtype=np.int64)
        return model.head.decision_flops(lengths, questions, options)

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
                 bucket: int | None = None):
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
        if bucket is not None and bucket < 1:
            raise ValueError(
                f"bucket is the multiple a batch's length rounds up to, at least 1, not {bucket}"
            )
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
            layout = StateFirstLayout() if module.causal else MarkerLayout()
        self.layout = layout
        self.tokenizer = tokenizer
        self.specials = specials or Specials.of(tokenizer)
        self.loss_rule = LogLoss() if loss is None else loss
        self.variables = held
        self.label_smoothing = label_smoothing
        self.shuffle_options = shuffle_options
        self.none_of_the_above = none_of_the_above
        self.temperatures = temperatures
        self.bucket = bucket
        self.inputs = InputSpec(sample=Field("tokens", (self.layout.max_len,), variable_axes=(0,)))

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
            width, dtype = DecisionModel.head_size(self.model.backbone)
            states = jnp.zeros((1, self.layout.max_len, width), dtype or jnp.float32)
            one = jnp.ones((1, 1), bool)
            inputs = DecisionInputs(
                tokens=tokens,
                valid=jnp.ones_like(tokens, bool),
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

    def loss(self, variables: Variables, batch: Batch, step: Step):
        inputs = laid_out(batch)
        logits = self.model.logits(
            variables, inputs, train=step.training, rngs=model_rngs(step.key, training=step.training)
        )
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
        from dew.data.batching import TOKEN_COUNTS

        weight = (
            self.row_mean(batch[TOKEN_COUNTS], batch).total
            if TOKEN_COUNTS in batch
            else self.row_mean(scored, batch).total
        )
        loss = Ratio(self.row_mean(charge * scored, batch).total, weight)
        return loss, Aux(metrics={"accuracy": accuracy})

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Decisions:
        inputs = laid_out(batch)
        weights = self.evaluation_variables(params, step)
        logits = self._logits(weights, inputs)
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
            ordinal=inputs.kinds == _SCORE,
            scored=batch["scored"],
        )

    @functools.cached_property
    def _logits(self):
        return jax.jit(lambda variables, inputs: self.model.logits(variables, inputs))

    def data(self, tokenizer: Tokenizer | None = None) -> ObjectiveData:
        """Fit rows before mixing, augment training layouts, and group and cut
        training and evaluation batches alike."""
        vocabulary = self.tokenizer if tokenizer is None else tokenizer
        training = ConfigureLayout(
            self.layout, vocabulary, self.specials, self.shuffle_options, self.none_of_the_above, train=True
        )
        evaluating = dataclasses.replace(training, train=False)
        fit = Fit(self.layout, vocabulary, self.specials)
        expand = () if self.layout.joint else (QuestionRows(),)
        cut = () if self.bucket is None else (_Cut(self.bucket, self.layout.max_len),)
        return ObjectiveData(
            train=(fit, *expand, Configured(training, needs=("slots",), scope="run", random=True)),
            evaluate=(fit, *expand, Configured(evaluating, needs=("slots",), scope="run")),
            group=None if self.bucket is None else LengthGroups(StoredLength()),
            train_batch=cut,
            evaluate_batch=cut,
        )

    def inference_record(self):
        """Return what a saved run rebuilds its task from.

        That is the backbone, head, layout, tokenizer, special tokens and the
        image processor's source, independently of the text tokenizer. Interop
        reloads that source's processor files and derives its media geometry
        from the source config; the decision record owns neither format.
        """
        from dew.config import ModelConfig
        from dew.inference.tasks import recorded_tokenizer
        from dew.registry import to_record

        return {
            "objective": import_path(type(self)),
            "model": to_record(ModelConfig.from_model(self.model.backbone), ModelConfig),
            "head": self.model.head.record(),
            "layout": {
                "name": type(self.layout).__name__,
                "fields": to_record(self.layout, type(self.layout)),
            },
            "specials": to_record(self.specials, Specials),
            "tokenizer": recorded_tokenizer(RunProcessor(self.tokenizer)),
            "processor": None if self.image_processor is None else self.image_processor.source,
        }

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
