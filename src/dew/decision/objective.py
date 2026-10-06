"""`DecisionObjective`: training a decision model on labelled questions."""

import functools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np

from dew.artifacts import Decisions
from dew.data.dataset import Dataset, Loading, train_stream, validation_pass
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.data import DecisionTable, Example
from dew.decision.head import KINDS, DecisionHead, kind_of
from dew.decision.layout import DecisionInputs, Layout, MarkerLayout, Specials, StateFirstLayout
from dew.decision.model import DecisionModel
from dew.decision.questions import Choice, Score
from dew.decision.scoring import LogLoss, ScoringRule
from dew.decision.task import Decide, Weights
from dew.inference.pipeline import RunProcessor
from dew.inference.tasks import Processor as TaskProcessor
from dew.inputs import Field, InputSpec
from dew.interop.processors import Processor
from dew.lora import Adapter
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import (
    OMITTED,
    Aux,
    Batch,
    Objective,
    Omitted,
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
    """How a labelled question becomes the fixed-width row a step trains on.

    Every row is `max_len` tokens and `width` option slots. With `shuffle`, a
    choice's options are laid out in a fresh random order every time the row is
    read (Laya's `--shuffle-options`), while a score's levels and a noul's two
    answers keep their order. With `none_of_the_above` p, a choice row gains a
    "none of the above" option with probability p, and half of those rows also
    lose their right option, which makes "none of the above" the right answer.
    That teaches the model to say that no option fits instead of picking the
    nearest wrong one.
    """

    layout: Layout
    tokenizer: Tokenizer
    specials: Specials
    width: int
    shuffle: bool = True
    none_of_the_above: float = 0.0

    def __call__(self, example: Example, name: str, rng: np.random.Generator | None) -> Batch:
        question, label = example.questions[name], example.labels()[name]
        if isinstance(question, Choice) and rng is not None and rng.random() < self.none_of_the_above:
            question, label = _none_of_the_above(question, label, drop=rng.random() < 0.5)
        count = len(question.options)
        order = (rng.permutation(count) if rng is not None and self.shuffle and isinstance(question, Choice)
                 else np.arange(count))
        tokens = self.layout.state(self.tokenizer, self.specials, example.state)
        match example.state:
            case list():
                conversation = True
            case _:
                conversation = False
        encoded = self.layout.encode(self.tokenizer, self.specials, question, tokens,
                                     conversation=conversation, order=[int(slot) for slot in order])
        if len(encoded.markers) != count:
            raise ValueError(f"{count - len(encoded.markers)} of a question's {count} options fall past "
                             f"max_len={self.layout.max_len}; raise it or lower head_max_len")
        if count > self.width:
            raise ValueError(f"a question of {count} options outgrows the encoding's {self.width} slots")
        length = self.layout.max_len
        row: Batch = {
            "tokens": _padded(encoded.tokens, length, self.specials.pad),
            "valid": _padded((True,) * len(encoded.tokens), length, fill=False),
            "markers": _padded(encoded.markers, self.width, 0),
            "options": _padded((True,) * count, self.width, fill=False),
            "kinds": np.int32(kind_of(question)),
            "labels": np.int32(int(np.flatnonzero(order == label)[0])),
        }
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
    """The answered questions of a set of examples, read by index as `{"row": i}`."""

    def __init__(self, rows: Sequence[tuple[Example, str]], origin: str):
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
        example, name = self.rows.rows[int(element["row"])]
        return self.encoding(example, name, rng if self.augment else None)


def _tokenizer_of(processor: TaskProcessor | None) -> Tokenizer | None:
    """The tokenizer a loaded model's text processor encodes with: a run's
    own, or a checkpoint's Hugging Face tokenizer read from the directory it
    was loaded from, at the revision the model was. A processor that is not
    a tokenizer's gives none, and `tokenizer=` names one."""
    from transformers import PreTrainedTokenizerBase

    match processor:
        case RunProcessor():
            return processor.tokenizer
        case Processor(reference=PreTrainedTokenizerBase() as reference):
            return HFTokenizer(str(reference.name_or_path), local_files_only=True)
        case _:
            return None


def _answered(examples: Iterable[Example | Mapping[str, object]]) -> list[tuple[Example, str]]:
    return [(example, name) for example in map(Example.of, examples) for name in example.labels()]


@objectives("decision")
class DecisionObjective(Objective[Ratio]):
    """Trains a decision model on questions with known answers.

    The model is a backbone with `hidden_states`, read by a `DecisionHead`.
    `backbone` is a `CausalTransformer` built from scratch, a loaded model
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

    def __init__(self, backbone: CausalTransformer | Source[CausalTransformer] | Adapter | Decide, *,
                 loss: ScoringRule | None = None, tokenizer: Tokenizer | None = None,
                 specials: Specials | None = None, layout: Layout | None = None,
                 head: DecisionHead | None = None, variables: Variables | None | Omitted = OMITTED,
                 label_smoothing: float = 0.0, shuffle_options: bool = True,
                 none_of_the_above: float = 0.0):
        held: Variables | None = None
        match backbone:
            case Decide():
                model, held = backbone.model, backbone.variables
                head = head or model.head
                layout = layout or backbone.layout
                tokenizer = tokenizer or backbone.tokenizer
                specials = specials or backbone.specials
                module = model.backbone
            case Adapter():
                module, held = backbone.model, joined({"backbone": backbone.variables})
            case Source():
                module, held = backbone.model, joined({"backbone": backbone.variables})
                tokenizer = tokenizer or _tokenizer_of(backbone.text_processor)
            case _:
                module = backbone
        if not isinstance(module, CausalTransformer):
            raise TypeError(f"a decision model reads a CausalTransformer's hidden states, "
                            f"not a {type(module).__name__}'s")
        if variables is not OMITTED:
            held = variables
        if tokenizer is None:
            raise ValueError("a decision model lays its rows out with a tokenizer; pass tokenizer=")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing is a share of the target, in [0, 1)")
        if not 0.0 <= none_of_the_above <= 1.0:
            raise ValueError("none_of_the_above is a probability")
        self.model = DecisionModel(module, head or DecisionHead(module.emb_features, dtype=module.dtype))
        self.layout = layout or (StateFirstLayout() if module.causal else MarkerLayout())
        self.tokenizer = tokenizer
        self.specials = specials or Specials.of(tokenizer)
        self.loss_rule = LogLoss() if loss is None else loss
        self.variables = held
        self.label_smoothing = label_smoothing
        self.shuffle_options = shuffle_options
        self.none_of_the_above = none_of_the_above
        self.inputs = InputSpec(sample=Field("tokens", (self.layout.max_len,)))

    def held_variables(self) -> Variables | None:
        return self.variables

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
        given = self.held_variables() if variables is None else variables
        backbone_key, head_key = jax.random.split(key)
        tokens = jnp.zeros((1, self.layout.max_len), jnp.int32)
        valid = jnp.ones((1, self.layout.max_len), bool)
        parts = {name: part(given, name) for name in ("backbone", "head")
                 if given is not None and any(name in tree for tree in given.values())}
        if "backbone" not in parts:
            parts["backbone"] = self.model.backbone.init(backbone_key, tokens)
        if "head" not in parts:
            states = jnp.zeros((1, self.layout.max_len, self.model.backbone.emb_features),
                               self.model.backbone.dtype or jnp.float32)
            parts["head"] = self.model.head.init(head_key, states, valid, jnp.zeros((1, 1), jnp.int32),
                                                 jnp.zeros((1,), jnp.int32))
        return joined(parts)

    def _laid_out(self, batch: Batch) -> DecisionInputs:
        return DecisionInputs(tokens=batch["tokens"], valid=batch["valid"], markers=batch["markers"],
                              options=batch["options"], kinds=batch["kinds"],
                              positions=batch.get("positions"), slots=batch.get("slots"))

    def loss(self, variables: Variables, batch: Batch, step: Step):
        inputs = self._laid_out(batch)
        logits = self.model.logits(variables, inputs, train=True, rngs={"dropout": step.key})
        options = inputs.options
        count = jnp.sum(options, axis=-1, keepdims=True)
        right = jax.nn.one_hot(batch["labels"], options.shape[1])
        target = jnp.where(options, (1.0 - self.label_smoothing) * right + self.label_smoothing / count, 0.0)
        charge = self.loss_rule.charge(logits, target, options, inputs.kinds == _SCORE)
        correct = (jnp.argmax(logits, axis=-1) == batch["labels"]).astype(jnp.float32)
        accuracy, _ = self.accuracy(correct, batch).mean()
        return self.row_mean(charge, batch), Aux(metrics={"accuracy": accuracy})

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Decisions:
        inputs = self._laid_out(batch)
        weights = params if step.ema is None else step.ema
        probabilities = jax.nn.softmax(self._logits(weights, inputs), axis=-1)
        return Decisions(probabilities=probabilities, options=inputs.options, labels=batch["labels"],
                         ordinal=inputs.kinds == _SCORE)

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
        train = _answered(examples)
        held = None if validation is None else _answered(validation)
        width = max(len(example.questions[name].options) + (self.none_of_the_above > 0)
                    for example, name in train + (held or []))
        encoding = Encoding(self.layout, self.tokenizer, self.specials, width, self.shuffle_options,
                            self.none_of_the_above)
        rows = _Rows(train, f"{len(train)} answered questions")
        if len(rows) < batch:
            raise ValueError(f"{len(rows)} answered questions, fewer than one batch of {batch}")
        scoring = None
        if held is not None:
            held_rows = _Rows(held, f"{len(held)} held-out answered questions")
            if len(held_rows) < batch:
                raise ValueError(f"{len(held_rows)} held-out answered questions, "
                                 f"fewer than one batch of {batch}")
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

        head = self.model.head
        return {
            "objective": objectives.name_of(type(self)),
            "model": to_record(ModelConfig.from_model(self.model.backbone), ModelConfig),
            "head": {"layers": head.layers, "dropout_rate": head.dropout_rate},
            "layout": {"name": type(self.layout).__name__,
                       "fields": to_record(self.layout, type(self.layout))},
            "specials": to_record(self.specials, Specials),
            "tokenizer": recorded_tokenizer(RunProcessor(self.tokenizer)),
        }

    def pipeline(self, state: "TrainState", *, ema: bool | None = None) -> Decide:
        """Return the trained model as a `Decide` task over the state's weights."""
        averaged = not (ema is False or (ema is None and state.ema is None))
        return Decide(self.model, self._pipeline_weights(state, ema), self.layout, self.tokenizer,
                      self.specials, weights=Weights(int(state.step), averaged))

