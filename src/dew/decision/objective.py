"""`DecisionObjective`: training a decision model on labelled questions."""

import dataclasses
import functools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.artifacts import Decisions
from dew.data.dataset import Dataset, Loading, train_stream, validation_pass
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.data import DecisionTable, Example
from dew.decision.head import DecisionHead, Head
from dew.decision.layout import DecisionInputs, Layout, MarkerLayout, Specials, StateFirstLayout
from dew.decision.model import DecisionModel
from dew.decision.questions import KINDS, Choice, Question, Score
from dew.decision.scoring import LogLoss, ScoringRule
from dew.decision.task import Decide, Weights, laid_out
from dew.inference.pipeline import RunProcessor
from dew.inference.tasks import Processor as TaskProcessor
from dew.inputs import Field, InputSpec
from dew.interop.processors import Processor
from dew.lora import Adapter
from dew.nn.protocols import TokenModel
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
    part,
)
from dew.registry import objectives

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
    levels and a noul's two answers keep their order. With `none_of_the_above`
    p, an answered choice gains a "none of the above" option with probability p,
    and half of those also lose their right option, which makes "none of the
    above" the right answer. That teaches the model to say that no option fits
    instead of picking the nearest wrong one.
    """

    layout: Layout
    tokenizer: Tokenizer
    specials: Specials
    width: int
    questions: int = 1
    shuffle: bool = True
    none_of_the_above: float = 0.0

    def __call__(self, example: Example, names: Sequence[str], rng: np.random.Generator | None) -> Batch:
        labels = example.labels()
        asked: dict[str, Question] = {}
        answers: dict[str, int] = {}
        for name in names:
            question, label = example.questions[name], labels.get(name)
            if (label is not None and isinstance(question, Choice) and rng is not None
                    and rng.random() < self.none_of_the_above):
                question, label = _none_of_the_above(question, label, drop=rng.random() < 0.5)
            asked[name] = question
            if label is not None:
                answers[name] = label
        orders = ({name: [int(slot) for slot in rng.permutation(len(question.options))]
                   for name, question in asked.items() if isinstance(question, Choice)}
                  if rng is not None and self.shuffle else None)
        rows = self.layout.rows(self.tokenizer, self.specials, example.state, asked, orders=orders)
        if len(rows) != 1 or len(rows[0].questions) > self.questions:
            raise ValueError(f"an encoding lays out one row of at most {self.questions} questions, "
                             f"and the layout gave {len(rows)} rows")
        encoded = rows[0]
        length = self.layout.max_len
        row: Batch = {
            "tokens": _padded(encoded.tokens, length, self.specials.pad),
            "valid": _padded((True,) * len(encoded.tokens), length, fill=False),
            "kinds": np.zeros(self.questions, np.int32), "questions": np.zeros(self.questions, bool),
            "spans": np.zeros((self.questions, 2), np.int32),
            "option_spans": np.zeros((self.questions, self.width, 2), np.int32),
            "options": np.zeros((self.questions, self.width), bool),
            "labels": np.zeros(self.questions, np.int32), "scored": np.zeros(self.questions, bool),
        }
        for slot, laid in enumerate(encoded.questions):
            count = len(asked[laid.name].options)
            if len(laid.options) != count:
                raise ValueError(f"{count - len(laid.options)} of question {laid.name!r}'s {count} options "
                                 f"fall past max_len={length}; raise it or shorten the question")
            if count > self.width:
                raise ValueError(f"a question of {count} options outgrows the encoding's {self.width} slots")
            row["kinds"][slot], row["questions"][slot], row["spans"][slot] = laid.kind, True, laid.span
            row["option_spans"][slot, :count] = laid.options
            row["options"][slot, :count] = True
            if laid.name in answers:
                row["labels"][slot] = laid.order.index(answers[laid.name])
                row["scored"][slot] = True
        if encoded.positions is not None and encoded.slots is not None:
            row["positions"] = _padded(encoded.positions, length, 0)
            row["slots"] = _padded(encoded.slots, length, 0)
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


def _padded(values: Sequence[int | bool], length: int, fill: int | bool = 0) -> np.ndarray:
    dtype = np.bool_ if isinstance(fill, bool) else np.int32
    row = np.full((length,), fill, dtype)
    row[:len(values)] = values
    return row


class _Rows:
    """The rows of a set of examples, each an example and the questions it lays
    out, read by index as `{"row": i}`."""

    def __init__(self, rows: Sequence[tuple[Example, tuple[str, ...]]], origin: str):
        self.rows = rows
        self.origin = origin

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Batch:
        return {"row": np.int64(index)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.origin})"


class _Encode(pygrain.RandomMapTransform):
    """`Encoding` as the grain step that lays out each row inside the workers."""

    def __init__(self, encoding: Encoding, rows: _Rows, *, augment: bool):
        self.encoding, self.rows, self.augment = encoding, rows, augment

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        example, names = self.rows.rows[int(element["row"])]
        return self.encoding(example, names, rng if self.augment else None)


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
        case (Processor(reference=PreTrainedTokenizerBase() as tokenizer)
              | Processor(reference=ProcessorMixin(tokenizer=PreTrainedTokenizerBase() as tokenizer))):
            return HFTokenizer(str(tokenizer.name_or_path), local_files_only=True)
        case _:
            return None


def _laid_rows(examples: Iterable[Example | Mapping[str, object]], *,
               joint: bool) -> list[tuple[Example, tuple[str, ...]]]:
    """Each row an example lays out: one per answered question, or, for a joint
    layout, one asking all its questions where any is answered."""
    held = [Example.of(example) for example in examples]
    if joint:
        return [(example, tuple(example.questions)) for example in held if example.labels()]
    return [(example, (name,)) for example in held for name in example.labels()]


@objectives("decision")
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
    `Encoding` describes, and `dataset` builds them.

    Evaluation returns the model's `Decisions` on the validation rows, which
    `Accuracy`, `ECE`, `AURC` and the scoring rules read. The saved run loads as
    a `Decide` task.
    """

    artifact = Decisions
    saved_task = Decide
    _held_parts = True

    def __init__(self, backbone: nn.Module | Source | Adapter | Decide, *,
                 loss: ScoringRule | None = None, tokenizer: Tokenizer | None = None,
                 specials: Specials | None = None, layout: Layout | None = None,
                 head: Head | None = None, variables: Variables | None | Omitted = OMITTED,
                 label_smoothing: float = 0.0, shuffle_options: bool = True,
                 none_of_the_above: float = 0.0):
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
        if head is None:
            width, dtype = DecisionModel.head_size(module)
            head = DecisionHead(width, dtype=dtype)
        self.model = DecisionModel(module, head)
        if layout is None:
            if not isinstance(module, TokenModel):
                raise ValueError(f"a {type(module).__name__} does not say whether it reads its tokens in "
                                 "order; pass layout= (StateFirstLayout if it does, MarkerLayout if not)")
            layout = StateFirstLayout() if module.causal else MarkerLayout()
        self.layout = layout
        self.tokenizer = tokenizer
        self.specials = specials or Specials.of(tokenizer)
        self.loss_rule = LogLoss() if loss is None else loss
        self.variables = held
        self.label_smoothing = label_smoothing
        self.shuffle_options = shuffle_options
        self.none_of_the_above = none_of_the_above
        self.inputs = InputSpec(sample=Field("tokens", (self.layout.max_len,)))

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The backbone, then the head, both trained."""
        return (ProgramModule(self.model.backbone, None, trained=True),
                ProgramModule(self.model.head, None, trained=True))

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
        parts = {name: part(tree, name) for name in ("backbone", "head")
                 if any(name in collection for collection in tree.values())}
        if "backbone" not in parts:
            parts["backbone"] = self.model.backbone.init(backbone_key, tokens)
        if "head" not in parts:
            width, dtype = DecisionModel.head_size(self.model.backbone)
            states = jnp.zeros((1, self.layout.max_len, width), dtype or jnp.float32)
            one = jnp.ones((1, 1), bool)
            inputs = DecisionInputs(tokens=tokens, valid=jnp.ones_like(tokens, bool),
                                    kinds=jnp.zeros((1, 1), jnp.int32), questions=one,
                                    spans=jnp.asarray([[[0, 1]]]), option_spans=jnp.asarray([[[[0, 1]]]]),
                                    options=one[..., None])
            table = (self.model.table(joined({"backbone": parts["backbone"]})) if self.model.head.reads_table
                     else None)
            parts["head"] = self.model.head.init(head_key, states, inputs, table)
        return joined(parts)

    def loss(self, variables: Variables, batch: Batch, step: Step):
        inputs = laid_out(batch)
        logits = self.model.logits(variables, inputs, train=True, rngs={"dropout": step.key})
        options, scored = inputs.options, jnp.asarray(batch["scored"]).astype(jnp.float32)
        # A padding question has no options; it is scored nothing over a
        # finite, even distribution, so no NaN reaches its gradient.
        empty = ~jnp.any(options, axis=-1, keepdims=True)
        logits, options = jnp.where(empty, 0.0, logits), options | empty
        count = jnp.sum(options, axis=-1, keepdims=True)
        right = jax.nn.one_hot(batch["labels"], options.shape[-1])
        target = jnp.where(options, (1.0 - self.label_smoothing) * right + self.label_smoothing / count, 0.0)
        charge = self.loss_rule.charge(logits, target, options, inputs.kinds == _SCORE)
        correct = (jnp.argmax(logits, axis=-1) == batch["labels"]).astype(jnp.float32)
        accuracy, _ = self.accuracy(correct, batch, scored).mean()
        loss = Ratio(self.row_mean(charge * scored, batch).total, self.row_mean(scored, batch).total)
        return loss, Aux(metrics={"accuracy": accuracy})

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Decisions:
        inputs = laid_out(batch)
        weights = self.evaluation_variables(params, step)
        logits = self._logits(weights, inputs)
        empty = ~jnp.any(inputs.options, axis=-1, keepdims=True)
        probabilities = jax.nn.softmax(jnp.where(empty, 0.0, logits), axis=-1)
        return Decisions(probabilities=probabilities, options=inputs.options, labels=batch["labels"],
                         ordinal=inputs.kinds == _SCORE, scored=batch["scored"])

    @functools.cached_property
    def _logits(self):
        return jax.jit(lambda variables, inputs: self.model.logits(variables, inputs))

    def dataset(self, examples: Iterable[Example | Mapping[str, object]] | DecisionTable, *, batch: int,
                validation: Iterable[Example | Mapping[str, object]] | None = None, seed: int = 0,
                loading: Loading | None = None) -> Dataset:
        """Return the training stream and validation pass over labelled questions.

        The training stream holds one row for each answered question in `examples`,
        augmented afresh on every read, and the validation pass holds `validation`'s
        rows as they are. A `DecisionTable` provides both, with its held-out rows as
        the second. The option slots fit the widest question either one holds.
        """
        if isinstance(examples, DecisionTable):
            if validation is not None:
                raise ValueError("a decision table holds out its own validation rows")
            examples, validation = examples.examples()
        loading = Loading() if loading is None else loading
        joint = self.layout.joint
        train = _laid_rows(examples, joint=joint)
        held = None if validation is None else _laid_rows(validation, joint=joint)
        width = max(len(example.questions[name].options) + (self.none_of_the_above > 0)
                    for example, names in train + (held or []) for name in names)
        questions = max(len(names) for _, names in train + (held or []))
        encoding = Encoding(self.layout, self.tokenizer, self.specials, width, questions,
                            self.shuffle_options, self.none_of_the_above)
        rows = _Rows(train, f"{len(train)} rows")
        if len(rows) < batch:
            raise ValueError(f"{len(rows)} rows of answered questions, fewer than one batch of {batch}")
        scoring = None
        if held is not None:
            held_rows = _Rows(held, f"{len(held)} held-out rows")
            scoring = validation_pass(held_rows, [_Encode(encoding, held_rows, augment=False)], batch=batch,
                                      seed=seed, loading=loading)
        return Dataset(train=train_stream(rows, [_Encode(encoding, rows, augment=True)], batch=batch,
                                          seed=seed, loading=loading),
                       val=scoring, records=len(rows), batch=batch)

    def inference_record(self):
        """Return what a saved run rebuilds its task from.

        That is the backbone, head, layout, tokenizer and special tokens.
        """
        from dew.config import ModelConfig
        from dew.inference.tasks import recorded_tokenizer
        from dew.registry import to_record

        return {
            "objective": objectives.name_of(type(self)),
            "model": to_record(ModelConfig.from_model(self.model.backbone), ModelConfig),
            "head": self.model.head.record(),
            "layout": {"name": type(self.layout).__name__,
                       "fields": to_record(self.layout, type(self.layout))},
            "specials": to_record(self.specials, Specials),
            "tokenizer": recorded_tokenizer(RunProcessor(self.tokenizer)),
        }

    def pipeline(self, state: "TrainState", *, ema: bool | None = None,
                 processor: TaskProcessor | None | Omitted = OMITTED) -> Decide:
        """Return the trained model as a `Decide` task over the state's weights.

        A `Decide` encodes with the objective's own tokenizer and layout, so it
        takes no processor; it reads images with the backbone's own processor,
        which a task the objective fine-tunes brings.
        """
        if processor is not OMITTED:
            raise TypeError("a Decide task encodes with the objective's tokenizer and takes no processor")
        averaged = not (ema is False or (ema is None and state.ema is None))
        return Decide(self.model, self._pipeline_weights(state, ema), self.layout, self.tokenizer,
                      self.specials, weights=Weights(int(state.step), averaged),
                      processor=self.image_processor)

