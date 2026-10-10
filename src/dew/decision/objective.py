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
from dew.data.dataset import (
    Corpus,
    Dataset,
    LengthGroups,
    Loading,
    Reader,
    mixed_records,
    mixed_stream,
    train_stream,
    validation_pass,
)
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.calibration import Temperatures
from dew.decision.data import DecisionTable, Example, Weighted
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
    training_rngs,
)
from dew.registry import import_path

if TYPE_CHECKING:
    from dew.training.state import TrainState

NONE_OF_THE_ABOVE = "none of the above"

type Labelled = Iterable[Example | Mapping[str, object]]
"""Labelled examples: `Example`s, or rows `Example.of` reads."""
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
    in the order the row shows the options.
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
        if rng is not None and self.shuffle and self.layout.joint:
            names = [names[int(index)] for index in rng.permutation(len(names))]
        asked: dict[str, Question] = {}
        golds: dict[str, np.ndarray] = {}
        for name in names:
            question = example.questions[name]
            gold = example.distribution(name) if name in labels else None
            if (name in example.answers and isinstance(question, Choice) and rng is not None
                    and rng.random() < self.none_of_the_above):
                question, label = _none_of_the_above(question, labels[name], drop=rng.random() < 0.5)
                gold = np.eye(len(question.options))[label]
            asked[name] = question
            if gold is not None:
                golds[name] = gold
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
            "targets": np.zeros((self.questions, self.width), np.float32),
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
            if laid.name in golds:
                shown = golds[laid.name][list(laid.order)]
                row["targets"][slot, :count] = shown
                row["labels"][slot] = int(np.argmax(shown))
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


_ALONG_THE_ROW = ("tokens", "valid", "positions", "slots")
"""The fields of an encoded row that run along its tokens."""


@dataclass(frozen=True)
class _Cut:
    """A batch's fields along its rows cut after its longest row, rounded up to
    a multiple of `bucket` and at most `length`: the padding no row reaches."""

    bucket: int
    length: int

    def __call__(self, batch: Batch) -> Batch:
        longest = int(np.max(np.sum(batch["valid"], axis=1)))
        size = min(self.length, -(-longest // self.bucket) * self.bucket)
        return {name: value[:, :size] if name in _ALONG_THE_ROW else value for name, value in batch.items()}


def _padded(values: Sequence[int | bool], length: int, fill: int | bool = 0) -> np.ndarray:
    dtype = np.bool_ if isinstance(fill, bool) else np.int32
    row = np.full((length,), fill, dtype)
    row[:len(values)] = values
    return row


class _Rows:
    """The rows of a set of examples, each an example and the questions it lays
    out, read by index as `{"corpus": c, "row": i}`, c naming the set in a mixture."""

    def __init__(self, rows: Sequence[tuple[Example, tuple[str, ...]]], origin: str, corpus: int = 0):
        self.rows = rows
        self.origin = origin
        self.corpus = corpus

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Batch:
        return {"corpus": np.int64(self.corpus), "row": np.int64(index)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.origin})"


class _Tokens:
    """A record's length as `LengthGroups` ranks it: the tokens of its row laid
    out without augmentation, counted when the order first reads it and kept,
    so a later pass does not lay it out twice."""

    def __init__(self, encoding: Encoding, corpora: Sequence[_Rows]):
        self.encoding, self.corpora = encoding, corpora
        self.counted: dict[tuple[int, int], int] = {}

    def __call__(self, record: Batch) -> int:
        key = int(record["corpus"]), int(record["row"])
        count = self.counted.get(key)
        if count is None:
            example, names = self.corpora[key[0]].rows[key[1]]
            count = self.counted[key] = int(np.sum(self.encoding(example, names, None)["valid"]))
        return count


class _Encode(pygrain.RandomMapTransform):
    """`Encoding` as the grain step that lays out each row inside the workers."""

    def __init__(self, encoding: Encoding, corpora: Sequence[_Rows], *, augment: bool):
        self.encoding, self.corpora, self.augment = encoding, corpora, augment

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        example, names = self.corpora[int(element["corpus"])].rows[int(element["row"])]
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


def _held_pass(held: Sequence[tuple[Example, tuple[str, ...]]], encoding: Encoding, *, batch: int, seed: int,
               loading: Loading) -> Reader:
    """The validation pass over held-out rows: each once, laid out unaugmented."""
    rows = _Rows(held, f"{len(held)} held-out rows")
    return validation_pass(rows, [_Encode(encoding, [rows], augment=False)], batch=batch, seed=seed,
                           loading=loading)


def _laid_rows(examples: Iterable[Example | Mapping[str, object]], *,
               joint: bool) -> list[tuple[Example, tuple[str, ...]]]:
    """Each row an example lays out: one per answered question, or, for a joint
    layout, one asking all its questions where any is answered."""
    held = [Example.of(example) for example in examples]
    if joint:
        return [(example, tuple(example.questions)) for example in held if example.labels()]
    return [(example, (name,)) for example in held for name in example.labels()]


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
    `Encoding` describes, and `dataset` builds them. With `bucket`, the
    training stream batches rows of like length (`LengthGroups`) and cuts each
    batch after its longest row, rounded up to a multiple of `bucket`, so a
    step pays for little more than its tokens instead of `max_len` per row;
    each length compiles its own step, and the stream is read on one process.

    Evaluation returns the model's `Decisions` on the validation rows, which
    `Accuracy`, `ECE`, `AURC` and the scoring rules read, its probabilities
    divided by `temperatures` (none by default: a `Decide`'s calibration fits
    its logits before training moves them, so it is not inherited). The saved
    run loads as a `Decide` task.
    """

    artifact = Decisions
    saved_task = Decide
    _held_parts = True

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
            raise ValueError(f"bucket is the multiple a batch's length rounds up to, at least 1, "
                             f"not {bucket}")
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
        self.temperatures = temperatures
        self.bucket = bucket
        # A cut batch is as long as its longest row, so only an uncut stream declares its length.
        self.inputs = None if bucket else InputSpec(sample=Field("tokens", (self.layout.max_len,)))

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
        logits = self.model.logits(variables, inputs, train=True, rngs=training_rngs(step.key))
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
        inputs = laid_out(batch)
        weights = self.evaluation_variables(params, step)
        logits = self._logits(weights, inputs)
        if self.temperatures is not None:
            # Each question's temperature by its type and its count of real
            # options, as `Temperatures.of` gives it; a padding question has none.
            width = logits.shape[-1]
            table = np.asarray([[1.0 if count == 0 else self.temperatures.of(kind.kind, count)
                                 for count in range(width + 1)] for kind in KINDS], np.float32)
            logits = logits / jnp.asarray(table)[inputs.kinds, jnp.sum(inputs.options, axis=-1)][..., None]
        empty = ~jnp.any(inputs.options, axis=-1, keepdims=True)
        probabilities = jax.nn.softmax(jnp.where(empty, 0.0, logits), axis=-1)
        return Decisions(probabilities=probabilities, options=inputs.options, labels=batch["labels"],
                         ordinal=inputs.kinds == _SCORE, scored=batch["scored"])

    @functools.cached_property
    def _logits(self):
        return jax.jit(lambda variables, inputs: self.model.logits(variables, inputs))

    def dataset(self, examples: Labelled | DecisionTable | Mapping[str, Weighted], *, batch: int,
                validation: Labelled | None = None, seed: int = 0, loading: Loading | None = None) -> Dataset:
        """Return the training stream and validation pass over labelled questions.

        The training stream holds one row for each answered question in `examples`,
        augmented afresh on every read, and the validation pass holds `validation`'s
        rows as they are. A `DecisionTable` provides both, with its held-out rows as
        the second. A mapping of names to `Weighted` sets is a mixture: every step
        takes each set's share of its rows, whatever the sets' lengths, and each set
        comes round again at its own rate (`dew.data.dataset.mixed_stream`). The
        option slots fit the widest question any of them holds.
        """
        match examples:
            case DecisionTable():
                if validation is not None:
                    raise ValueError("a decision table holds out its own validation rows")
                examples, validation = examples.examples()
                weighted = {"examples": Weighted(examples, 1.0)}
            case Mapping():
                weighted: dict[str, Weighted] = {}
                for name, entry in examples.items():
                    match name, entry:
                        case str(), Weighted():
                            weighted[name] = entry
                        case _:
                            raise TypeError("a mixture maps names to Weighted example sets")
            case _:
                weighted = {"examples": Weighted(list(examples), 1.0)}
        loading = Loading() if loading is None else loading
        joint = self.layout.joint
        names = sorted(weighted)
        corpora = []
        for corpus, name in enumerate(names):
            laid = _laid_rows(weighted[name].examples, joint=joint)
            corpora.append(_Rows(laid, f"{name}: {len(laid)} rows", corpus))
        held = None if validation is None else _laid_rows(validation, joint=joint)
        # One shape for the training and the held-out rows, so both compile once.
        encoding = self._encoding([row for rows in corpora for row in rows.rows] + (held or []))
        records = sum(len(rows) for rows in corpora)
        if records < batch:
            raise ValueError(f"{records} rows of answered questions, fewer than one batch of {batch}")
        scoring = (None if held is None
                   else _held_pass(held, encoding, batch=batch, seed=seed, loading=loading))
        encode = [_Encode(encoding, corpora, augment=True)]
        bucketed = {} if self.bucket is None else {"groups": LengthGroups(_Tokens(encoding, corpora)),
                                                   "cut": _Cut(self.bucket, self.layout.max_len)}
        if len(corpora) == 1:
            return Dataset(train=train_stream(corpora[0], encode, batch=batch, seed=seed, loading=loading,
                                              **bucketed),
                           val=scoring, records=len(corpora[0]), batch=batch)
        mixed = [Corpus(name, rows, weighted[name].weight) for name, rows in zip(names, corpora, strict=True)]
        return Dataset(train=mixed_stream(mixed, encode, batch=batch, seed=seed, loading=loading, **bucketed),
                       val=scoring, records=mixed_records(mixed), batch=batch)

    def held_out(self, examples: Labelled, *, batch: int, loading: Loading | None = None) -> Reader:
        """Return the validation pass over the answered questions of `examples`,
        each row once, as `dataset` holds out its `validation`, however few."""
        held = _laid_rows(examples, joint=self.layout.joint)
        if not held:
            raise ValueError("scoring needs examples with answers")
        return _held_pass(held, self._encoding(held), batch=batch, seed=0,
                          loading=Loading() if loading is None else loading)

    def _encoding(self, rows: Sequence[tuple[Example, tuple[str, ...]]]) -> Encoding:
        """The layout of `rows`, with as many option and question slots as the widest row needs."""
        width = max(len(example.questions[name].options) + (self.none_of_the_above > 0)
                    for example, names in rows for name in names)
        questions = max(len(names) for _, names in rows)
        return Encoding(self.layout, self.tokenizer, self.specials, width, questions,
                        self.shuffle_options, self.none_of_the_above)

    def inference_record(self):
        """Return what a saved run rebuilds its task from.

        That is the backbone, head, layout, tokenizer and special tokens.
        """
        from dew.config import ModelConfig
        from dew.inference.tasks import recorded_tokenizer
        from dew.registry import to_record

        return {
            "objective": import_path(type(self)),
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

