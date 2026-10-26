"""`DecisionObjective`: training a decision model on labelled questions."""

import dataclasses
import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct

from dew._records import integer, record
from dew.artifacts import Decisions
from dew.batch import FROZEN, Batch, Variables, part
from dew.data.pipeline import Configured, ObjectiveData, Pack, Rejected
from dew.data.text import HFTokenizer, Tokenizer
from dew.decision.calibration import Temperatures
from dew.decision.data import Example, GroupedExamples
from dew.decision.head import DecisionHead, Head
from dew.decision.layout import (
    OPTION_TILE,
    OPTIONS,
    QUESTION_FIELDS,
    Bounds,
    DecisionInputs,
    Encoded,
    JointLayout,
    Layout,
    QuestionField,
    QuestionLayout,
    Specials,
    compacted,
    default_layout,
    labelled_rows,
    question_fields,
    question_owners,
    with_context,
)
from dew.decision.model import DecisionModel
from dew.decision.questions import KINDS, Choice, Question, Score
from dew.decision.scoring import LogLoss, ScoringRule
from dew.decision.task import Decide, DecideSettings, Weights, laid_out
from dew.inference.loading import RunProcessor
from dew.inference.tasks import Processor as TaskProcessor
from dew.inputs import Field, InputSpec
from dew.interop.processors import Processor
from dew.lora import Adapter
from dew.nn import categorical
from dew.nn.cross_attention import ContextEncoder, crosses_alone
from dew.nn.flops import MissingFlopsCount, counted_flops, frozen_training_flops, parameter_flops
from dew.nn.inputs import ModelInputs
from dew.nn.protocols import CountedDecisions, CountedTraining, CrossAttending, TokenModel
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
)

if TYPE_CHECKING:
    from dew.state import TrainState

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

    With a `context_length`, a row also carries its example's context as
    `context`, the tokens of its text (`render`), a context longer than
    `context_length` refused (`Fit`), and none for an example without one.
    """

    layout: Layout
    tokenizer: Tokenizer
    specials: Specials
    width: int
    shuffle: bool = True
    none_of_the_above: float = 0.0
    context_length: int | None = None
    grouped: bool = False
    """Whether each row carries its questions' consistency groups (`_grouped`), an objective with a
    consistency term's rows; every row then carries them, an example of none with none in each."""

    def __call__(self, example: Example, names: Sequence[str], rng: np.random.Generator | None,
                 tokens: np.ndarray | None = None, group_of: "GroupOf | None" = None) -> Batch:
        labels = example.labels()
        if rng is not None and self.shuffle and self.layout.joint:
            names = [names[int(index)] for index in rng.permutation(len(names))]
        asked: dict[str, Question] = {}
        golds: dict[str, np.ndarray] = {}
        dropped: dict[str, str | None] = {}
        for name in names:
            question = example.questions[name]
            gold = example.distribution(name) if name in labels else None
            if (
                name in example.answers
                and isinstance(question, Choice)
                and rng is not None
                and rng.random() < self.none_of_the_above
            ):
                original = question
                question, label = _none_of_the_above(question, labels[name], drop=rng.random() < 0.5)
                gold = np.eye(len(question.options))[label]
                gone = [key for key in original.options if key not in question.options]
                # The augmentation's own option: it stands for the option it replaced, or for none.
                dropped[name] = gone[0] if gone else None
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
        # A run over a backbone that reads no context leaves an example's own context unread.
        rows = with_context(rows, self.tokenizer, None if self.context_length is None else example.context,
                            self.context_length)
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
            **question_fields(encoded, questions, self.width, labelled_rows([encoded])),
            **{field.name: field.empty(questions, self.width) for field in TRAINING_FIELDS
               if field.when == "trained"},
        }
        if self.grouped:
            row["groups"], row["canonical"] = _grouped(group_of or _own_groups(example), encoded, asked,
                                                       dropped, self.width)
            # A row of its own is a unit of one row (`_unit`).
            row["row"] = np.zeros(len(row["tokens"]), np.int32)
            row["question_row"] = np.zeros(questions, np.int32)
            row["unit_rows"] = np.zeros(1, np.int32)
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
        if encoded.blocks is not None:
            row["blocks"] = np.asarray(encoded.blocks, np.int32)
        if self.context_length is not None:
            row["context"] = np.asarray(encoded.context or (), np.int32)
        return row



type GroupOf = Callable[[str], tuple[int, Mapping[str, str]] | None]
"""The group a row's question stands in, by its name: the group's index in its unit and the question's
options mapped onto the group's canonical keys, or None."""


def _own_groups(example: Example) -> GroupOf:
    """An example's own groups (`Example.groups`), numbered in the example."""
    def found(name: str) -> tuple[int, Mapping[str, str]] | None:
        held = example.group_of(name)
        return None if held is None else (held[0], held[1].members[name])
    return found


def _grouped(group_of: GroupOf, encoded: Encoded, asked: Mapping[str, Question],
             dropped: Mapping[str, str | None], width: int) -> tuple[np.ndarray, np.ndarray]:
    """The row's consistency fields along its questions: each question's group in its unit
    (`group_of`), -1 for none, and each of its option slots' canonical option in that group,
    the slot as the row shows it (`Laid.order`), -1 for padding and for an option the group does not
    map. A "none of the above" the augmentation added (`dropped` names the question) that stands for
    the option it dropped maps to that option's canonical one, and one added beside every option to
    none; a question's own option of that name maps as its group says."""
    declared = {field.name: field for field in TRAINING_FIELDS}
    groups, canonical = (declared[name].empty(len(encoded.questions), width)
                         for name in ("groups", "canonical"))
    for slot, laid in enumerate(encoded.questions):
        found = group_of(laid.name)
        if found is None:
            continue
        index, mapped = found
        keys = sorted(set(mapped.values()))
        groups[slot] = index
        options = asked[laid.name].options
        for shown, option in enumerate(laid.order):
            key = options[option]
            if key == NONE_OF_THE_ABOVE and laid.name in dropped:
                key = dropped[laid.name]
            if key is not None and key in mapped:
                canonical[slot, shown] = keys.index(mapped[key])
    return groups, canonical


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
    context_length: int | None = None
    """The most context tokens the backbone's cross-attention reads, or None where it reads none: an
    example whose context is longer refuses the run (ValueError) where the data reads it, which the
    data plan's compile meets before training for data read into memory and a streamed or indexed
    read meets at that record, so no example is silently cut or dropped."""
    grouped: bool = False
    """Whether the objective reads consistency groups, whose members may stand in rows of their own: an
    example's questions under a layout of a question a row, or grouped examples' (`GroupedExamples`),
    whose rows then train as one unit (`_unit`)."""
    bounds: Bounds | None = None
    """The most rows and questions a packed window holds, which a unit of rows fits in or is rejected."""
    none_of_the_above: float = 0.0
    """The rate the training encoding adds a "none of the above" at (`Encoding`), for which a unit's
    rows are budgeted at their longest."""
    rejects = True

    def map(self, element: Batch) -> Batch | Rejected:
        held = element.get("example", element)
        if isinstance(held, GroupedExamples) or (isinstance(held, Mapping) and "examples" in held):
            grouped = GroupedExamples.of(held)
            # Without a consistency term the examples are rows of their own, which no window ties.
            return self._unit(grouped) if self.grouped else self._apart(grouped)
        example = Example.of(held)
        if self.grouped and example.groups and not self.layout.joint:
            # Its grouped questions stand in rows of their own, which train as one unit.
            return self._unit(GroupedExamples((example,)))
        labels = example.labels()
        if not labels:
            return Rejected("unlabelled")
        asked = example.questions if self.layout.joint else {name: example.questions[name] for name in labels}
        fitted = self._rows(example, asked)
        if isinstance(fitted, Rejected):
            return fitted
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

    def _rows(self, example: Example, asked: Mapping[str, Question]) -> tuple[FittedRow, ...] | Rejected:
        """`example`'s rows of the questions `asked`, as the layout lays them out, or why none fit."""
        state = self.layout.state(self.tokenizer, self.specials, example.state)
        laid = self.layout.rows(self.tokenizer, self.specials, example.state, asked, tokens=state)
        if not all(map(self.layout.fits, laid)):
            return Rejected("unfit")
        if self.context_length is not None:
            with_context(laid, self.tokenizer, example.context, self.context_length)
        self.head.validate_rows(laid)
        tokens = np.asarray(state, np.int32)
        return tuple(FittedRow(example, tuple(question.name for question in row.questions), len(row.tokens),
                               row.state_kept < row.state_tokens, tokens) for row in laid)

    def _apart(self, grouped: GroupedExamples) -> Batch | Rejected:
        """Grouped examples' labelled rows, each as an example of its own lays it out (`QuestionRows`
        spreads them), where no consistency term reads their groups."""
        rows = []
        for example in grouped.examples:
            labels = example.labels()
            if not labels:
                continue
            asked = (example.questions if self.layout.joint
                     else {name: example.questions[name] for name in labels})
            fitted = self._rows(example, asked)
            if isinstance(fitted, Rejected):
                return fitted
            rows.extend(fitted)
        return {"decision_rows": tuple(rows)} if rows else Rejected("unlabelled")

    def _longest(self, row: FittedRow) -> int:
        """The most tokens `row` encodes to: each answered choice given a "none of the above" beside
        its options where the training encoding may add one (`Encoding`), the longest it lays out."""
        example = row.example
        if not self.none_of_the_above:
            return row.length
        labels = example.labels()
        asked = {name: (_none_of_the_above(question, labels[name], drop=False)[0]
                        if name in example.answers and isinstance(question, Choice) else question)
                 for name in row.names for question in (example.questions[name],)}
        laid = self.layout.rows(self.tokenizer, self.specials, example.state, asked,
                                tokens=row.tokens.tolist())
        return sum(len(each.tokens) for each in laid)

    def _unit(self, grouped: GroupedExamples) -> Batch | Rejected:
        """Grouped examples' rows as one unit, which the packer places in one window: each example's
        rows as the layout lays them out, with the example's index (`FittedUnit`), a row of each of its
        labelled and grouped questions under a layout of a question a row, its one row under a joint
        layout. A unit with no labelled question, or that no window holds (its tokens, rows or
        questions past the window's), is rejected."""
        if self.context_length is not None:
            raise ValueError("a backbone that reads context packs one row a window, so rows that train "
                             "together do not fit one; train them without context or without groups")
        if not any(example.labels() for example in grouped.examples):
            return Rejected("unlabelled")
        rows: list[tuple[int, FittedRow]] = []
        for index, example in enumerate(grouped.examples):
            named = set(example.labels()) | {name for name in example.questions
                                             if grouped.group_of(index, name) is not None}
            if not named:
                continue
            asked = (example.questions if self.layout.joint else
                     {name: question for name, question in example.questions.items() if name in named})
            fitted = self._rows(example, asked)
            if isinstance(fitted, Rejected):
                return fitted
            rows.extend((index, row) for row in fitted)
        length = sum(self._longest(row) for _, row in rows)
        questions = sum(len(row.names) for _, row in rows)
        bounds = self.bounds
        if length > self.layout.max_len or (bounds is not None and (len(rows) > bounds.segments
                                                                    or questions > bounds.questions)):
            return Rejected("unfit")
        options = max(len(grouped.examples[index].questions[name].options)
                      for index, row in rows for name in row.names)
        return {"decision_unit": FittedUnit(grouped, tuple(rows)), "_dew_length": length,
                "_dew_truncated": any(row.truncated for _, row in rows),
                "_dew_bounds": {"slots": {"questions": questions, "options": options}}}


@dataclass(frozen=True)
class FittedUnit:
    """Rows that train together in one window (`Fit._unit`): each the index of its example in `grouped`
    and its fitted row."""

    grouped: GroupedExamples
    rows: tuple[tuple[int, FittedRow], ...]


def _unit_row(encoded: Sequence[Batch]) -> Batch:
    """A unit's encoded rows as one document the packer places whole: the fields along the tokens and
    along the questions laid end to end, with each token's and each question's row in the unit (`row`,
    `question_row`), and one entry a row along an axis of its own (`unit_rows`), which bounds a window's
    rows. Each row's spans stay its own; the window counts them from where the row starts in it
    (`PackedRows`)."""
    joined = {name: np.concatenate([np.asarray(row[name]) for row in encoded])
              for name in encoded[0] if name not in ("row", "question_row", "unit_rows")}
    joined["row"] = np.concatenate([np.full(len(row["tokens"]), index, np.int32)
                                    for index, row in enumerate(encoded)])
    joined["question_row"] = np.concatenate([np.full(len(row["kinds"]), index, np.int32)
                                             for index, row in enumerate(encoded)])
    joined["unit_rows"] = np.zeros(len(encoded), np.int32)
    return joined


def _encoded_unit(encoding: "Encoding", unit: FittedUnit, rng: np.random.Generator | None) -> Batch:
    """A unit's rows encoded each as `encoding` encodes a row, its groups numbered in the unit."""
    def group_of(index: int) -> GroupOf:
        return lambda name: unit.grouped.group_of(index, name)
    return _unit_row([encoding(row.example, row.names, rng, row.tokens, group_of(index))
                      for index, row in unit.rows])


@dataclass(frozen=True)
class QuestionRows:
    def flat_map(self, element: Batch) -> Sequence[Batch]:
        if "decision_rows" not in element:
            # A joint layout's row, or a unit of rows the packer places whole.
            return (element,)
        return tuple(
            {"decision": row, "_dew_length": row.length, "_dew_truncated": row.truncated}
            | {
                "_dew_bounds": {
                    "slots": {
                        "questions": len(row.names),
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
        if "decision_unit" in element:
            return _encoded_unit(self.encoding, element["decision_unit"], rng)
        fitted = _fitted(element)
        return self.encoding(fitted.example, fitted.names, rng, fitted.tokens)


@dataclass(frozen=True)
class EvaluateRows(pygrain.MapTransform):
    encoding: Encoding

    def map(self, element: Batch) -> Batch:
        if "decision_unit" in element:
            return _encoded_unit(self.encoding, element["decision_unit"], None)
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
    context_length: int | None = None
    grouped: bool = False

    def __call__(self, metadata: Mapping[str, object]) -> LayoutRows | EvaluateRows:
        slots = record(metadata["slots"], "decision slots")
        width = integer(slots.get("options"), "slots.options") + int(self.none_of_the_above > 0)
        if width < 1:
            raise ValueError("decision source slot maxima must be positive")
        encoding = Encoding(self.layout, self.tokenizer, self.specials, width, self.shuffle,
                            self.none_of_the_above, self.context_length, self.grouped)
        return LayoutRows(encoding) if self.train else EvaluateRows(encoding)


TRAINING_FIELDS = (QuestionField("labels", (), np.int32, when="trained"),
                   QuestionField("scored", (), np.bool_, when="trained"),
                   QuestionField("targets", (OPTIONS,), np.float32, when="trained"),
                   QuestionField("groups", (), np.int32, fill=-1, when="grouped"),
                   QuestionField("canonical", (OPTIONS,), np.int32, fill=-1, when="grouped"),
                   QuestionField("question_row", (), np.int32, when="grouped"))
"""The fields along a training row's questions beside the model's (`QUESTION_FIELDS`): each question's
gold `labels`, whether it is `scored`, its soft `targets` over its option slots, and, where its
questions stand in consistency groups (`Example.groups`, `GroupedExamples`), each one's group and each
option slot's `canonical` option, -1 for none, and its row in its unit of rows (`question_row`, `_unit`)."""


def _along(layout: Layout, grouped: bool = False) -> tuple[str, ...]:
    """The fields along the questions of `layout`'s training rows (`QUESTION_FIELDS`,
    `TRAINING_FIELDS`): its readout's where its options stand under labels, and their consistency
    groups' where they carry them (`grouped`)."""
    labelled = isinstance(layout, QuestionLayout) and layout.labels is not None
    carried = ("always", "trained", *(("labelled",) if labelled else ()), *(("grouped",) if grouped else ()))
    declared = (*QUESTION_FIELDS, *TRAINING_FIELDS)
    # The model's and training's fields, then the labelled readout, then the groups, as rows carry them.
    return tuple(field.name for when in carried for field in declared if field.when == when)


def _along_fields(element: Batch, when: tuple[str, ...] = ("always", "trained")) -> tuple[str, ...]:
    """The declared fields along the questions (`QUESTION_FIELDS`, `TRAINING_FIELDS`) that `element`
    carries and a row carries `when`, in their declared order."""
    return tuple(field.name for field in (*QUESTION_FIELDS, *TRAINING_FIELDS)
                 if field.name in element and field.when in when)


def packing(layout: Layout, bounds: Bounds, context_length: int | None = None, grouped: bool = False) -> Pack:
    """Rows of `layout` packed first-fit into windows of its `max_len`, each
    window's rows, questions and options within `bounds` (`Layout.bounds`),
    and with a `context_length` each row's context along an axis of that
    many tokens."""
    parallel = isinstance(layout, QuestionLayout) and layout.parallel
    blocked = isinstance(layout, QuestionLayout) and layout.blocks
    along = tuple((name, bounds.questions) for name in _along(layout, grouped))
    if context_length is not None:
        along = (*along, ("context", context_length))
    if grouped:
        # A unit of rows packs as one document, one entry a row on an axis of the window's rows.
        along = (*along, ("unit_rows", bounds.segments))
    fields = ("tokens", *(("positions", "slots") if parallel else ()), *(("blocks",) if blocked else ()),
              *(("row",) if grouped else ()))
    return Pack(layout.max_len, fields, segments=bounds.segments, along=along)


def _rows(batch: Batch, spans_in_context: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each packed row's tokens, questions and options, window by window: a question the row its span
    starts in, or, where the spans are the context's, the window's one row (`question_owners`)."""
    segments = np.asarray(batch["segments"])
    owners = question_owners(segments, batch["spans"], np.asarray(batch["questions"]), spans_in_context)
    options = np.sum(np.asarray(batch["options"], bool), axis=-1)
    counted = []
    for window, owner, option in zip(segments, owners, options, strict=True):
        rows = int(window.max()) + 1
        counted.append(np.stack([np.bincount(window, minlength=rows), np.bincount(owner, minlength=rows),
                                 np.bincount(owner, weights=option, minlength=rows).astype(np.int64)])[:, 1:])
    lengths, questions, real = np.concatenate(counted, axis=1)
    return lengths, questions, real


def _unit_rows(element: Batch, units: np.ndarray, owned: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A window's rows, numbered from 1 in the order they stand and 0 for padding, from the packer's
    documents (`units`, `owned` each question's) and each token's and question's row in its unit
    (`row`, `question_row`; a unit of one row where the window holds none); each question's row; and
    each token's position in its row."""
    if "row" not in element:
        return units, owned, np.asarray(element["tokens_positions"])
    inner, question_inner = np.asarray(element["row"]), np.asarray(element["question_row"])
    width = int(max(inner.max(initial=0), question_inner.max(initial=0))) + 1
    keys = units.astype(np.int64) * width + inner
    numbered = np.unique(keys[units > 0])
    segments = np.where(units > 0, np.searchsorted(numbered, keys) + 1, 0).astype(units.dtype)
    question_keys = owned.astype(np.int64) * width + question_inner
    owners = np.where(owned > 0, np.searchsorted(numbered, question_keys) + 1, 0).astype(owned.dtype)
    starts = np.concatenate([[0], np.cumsum(np.bincount(segments)[1:])])
    positions = np.where(segments > 0, np.arange(len(segments)) - starts[np.maximum(segments - 1, 0)], 0)
    return segments, owners, positions.astype(np.int32)


def _window_groups(element: Batch, owners: np.ndarray, options: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """A window's consistency groups numbered apart across its units, each unit's group `g` (`owners`:
    each question's unit) the window's own index of `(unit, g)`, and the padding the packer added back
    at -1: a question slot of no unit is in no group, and an option slot of no option maps to no
    canonical option."""
    groups = np.asarray(element["groups"]).astype(np.int64)
    real = (owners > 0) & (groups >= 0)
    keys = np.where(real, owners.astype(np.int64) * (groups.shape[-1] + 1) + groups, -1)
    numbered = np.full(groups.shape, -1, np.int32)
    if real.any():
        _, index = np.unique(keys[real], return_inverse=True)
        numbered[real] = index.astype(np.int32)
    canonical = np.where(options & (owners > 0)[..., None], np.asarray(element["canonical"]), -1)
    return numbered, canonical.astype(np.int32)


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
        units = np.asarray(element["tokens_segment_ids"])
        owned = np.asarray(element["kinds_segment_ids"])
        segments, owners, positions = _unit_rows(element, units, owned)
        starts = np.concatenate([[0], np.cumsum(np.bincount(segments)[1:])]).astype(np.int32)
        moved = np.where(owners > 0, starts[np.maximum(owners - 1, 0)], 0)
        options = np.asarray(element["options"])
        laid = {name: element[name] for name in ("tokens", *_along_fields(element))} | {
            "segments": segments,
            "positions": element["positions"] if "slots" in element else positions,
            "spans": element["spans"] + moved[:, None],
            "option_spans": element["option_spans"] + (moved[:, None] * options)[..., None],
        }
        if "readout" in element:
            # A question's answer cue moves with its row in the window, as its spans do.
            laid["readout"] = element["readout"] + moved
        if "groups" in element:
            # A group is its unit's: the packer's documents, before they are cut into their rows.
            laid["groups"], laid["canonical"] = _window_groups(element, owned, options)
        if "slots" in element:
            laid["slots"] = element["slots"]
        if "blocks" in element:
            # Each row keeps its own blocks where the window packed it; the padding is in none.
            laid["blocks"] = np.where(segments > 0, np.asarray(element["blocks"]), -1).astype(np.int32)
        if "context" in element:
            # A conditioned window holds one row (`DecisionObjective.bounds`), whose context it is.
            laid["context"] = element["context"]
            laid["context_lengths"] = np.int32(np.count_nonzero(np.asarray(element["context_segment_ids"])))
        if self.option_slots is not None:
            laid["option_slots"], laid["slotted"] = compacted(options, self.option_slots)
        return laid


@struct.dataclass
class DecisionStatistics:
    """A decision loss's statistics where it adds a consistency term: the label loss and the groups'
    divergence, each its own `Ratio`, their weighted sum the objective's value
    (`DecisionObjective.reduce_loss`)."""

    label: Ratio
    consistency: Ratio


class DecisionObjective(Objective[Ratio | DecisionStatistics]):
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
    scored questions. With `consistency` w > 0 it adds w times the mean over
    the batch's consistency groups (`Example.groups`) of the generalized
    Jensen-Shannon divergence of their members' answers over the group's
    canonical options (`categorical.group_divergence`), reported as the
    `consistency` metric; a group's members stand in one row, so a layout of a
    question a row refuses an example with groups under it. At 0 the groups
    are not read, and the loss is the label loss alone.

    Evaluation returns the model's `Decisions` on the validation rows, which
    `Accuracy`, `ECE`, `AURC` and the scoring rules read, its probabilities
    divided by `temperatures` (none by default: a `Decide`'s calibration fits
    its logits before training moves them, so it is not inherited). The saved
    run loads as a `Decide` task.
    """

    saved_task = Decide
    settings = DecideSettings
    _held_parts = True
    # The record is the backbone's; the head beside it is a task setting.
    _recorded_part = "backbone"

    counts_tokens = True

    def real_lengths(self, batch: Batch) -> np.ndarray:
        """Each packed row's own length, not its window's."""
        return _rows(batch, self.schema_in_context)[0]

    def model_flops(self, model, batch: Batch) -> float:
        """One backbone pass, its cross-attention over each row's context where rows carry one
        (`CrossAttending.cross_flops`), plus the head's real question/option geometry. Forming the
        vocabulary rows a head reads (an adapter's update merged into them) is execution, as
        recompute is."""
        model = self.model if model is None else model
        return (counted_flops(model.backbone, self.real_lengths(batch)) + self._context_flops(model, batch)
                + self._cross_flops(model, batch) + self._head_flops(model, batch))

    def _context_lengths(self, batch: Batch) -> np.ndarray | None:
        """Each window row's context length where the backbone reads the context in a pass of its own,
        as it reads a schema in context (`schema_in_context`) or encodes every context
        (`DecisionModel.context_encoder`); None where it reads none."""
        passes = self.schema_in_context or self.model.context_encoder == "backbone"
        if not passes or "context_lengths" not in batch:
            return None
        return np.asarray(batch["context_lengths"])

    def _context_flops(self, model, batch: Batch) -> float:
        """The backbone's forward over each row's context, where it reads one in a pass of its own."""
        lengths = self._context_lengths(batch)
        return 0.0 if lengths is None else counted_flops(model.backbone, lengths)

    def _cross_flops(self, model, batch: Batch) -> float:
        """The backbone's cross-attention work over each window row's context: none where rows carry
        none. A conditioned window holds one row (`bounds`), so its rows are the windows."""
        if "context_lengths" not in batch:
            return 0.0
        backbone = model.backbone
        assert isinstance(backbone, CrossAttending)
        return backbone.cross_flops(self.real_lengths(batch), np.asarray(batch["context_lengths"]))

    def _head_flops(self, model, batch: Batch) -> float:
        """Forward and training work read the same layout geometry and head capability."""
        tokens = batch.get("tokens")
        if isinstance(tokens, ModelInputs) and tokens.conditioning:
            raise MissingFlopsCount("DecisionObjective needs its image tower's multi-sequence geometry")
        if not isinstance(model.head, CountedDecisions):
            raise MissingFlopsCount(f"{type(model.head).__name__} has no declared decision_flops")
        # Spans in the context pool over its tokens, not the row's.
        pooled = self._context_lengths(batch) if self.schema_in_context else None
        return model.head.decision_flops(*_rows(batch, self.schema_in_context),
                                         **({} if pooled is None else {"pooled": pooled}))

    def training_flops(self, model, batch: Batch) -> float:
        """The backbone's frozen and trainable work, then three passes of the trained cross-attention
        blocks and decision head: an adapted backbone's own count (`CountedTraining`); a backbone whose
        held parameters train nothing but its cross-attention blocks, as `with_cross_attention` holds a
        source, its forward, input-gradient and no weight-gradient work (`frozen_training_flops`);
        else, trained whole, three passes of everything."""
        model = self.model if model is None else model
        lengths = self.real_lengths(batch)
        context = self._context_lengths(batch)
        trained = 3 * (self._cross_flops(model, batch) + self._head_flops(model, batch))
        if isinstance(model.backbone, CountedTraining):
            # An adapter trains through the context's pass too.
            own = 0.0 if context is None else model.backbone.training_flops(context)
            return model.backbone.training_flops(lengths) + own + trained
        if self._backbone_frozen():
            # A context's pass trains nothing, so it runs forward alone (`encode_context`).
            forward = counted_flops(model.backbone, lengths)
            return (frozen_training_flops(forward, parameter_flops(model.backbone, lengths))
                    + self._context_flops(model, batch) + trained)
        return super().training_flops(model, batch)

    def _backbone_frozen(self) -> bool:
        """Whether the held tree trains none of the backbone's own parameters: every backbone
        parameter it trains is a cross-attention block's (`crossing`), the rest held frozen."""
        held = self.variables
        if held is None or FROZEN not in held or "backbone" not in held[FROZEN]:
            return False
        return crosses_alone(held.get("params", {}).get("backbone", {}))

    def __init__(self, backbone: nn.Module | Source | Adapter | Decide, *,
                 loss: ScoringRule | None = None, tokenizer: Tokenizer | None = None,
                 specials: Specials | None = None, layout: Layout | None = None,
                 head: Head | None = None, variables: Variables | None | Omitted = OMITTED,
                 label_smoothing: float = 0.0, shuffle_options: bool = True,
                 none_of_the_above: float = 0.0, temperatures: Temperatures | None = None,
                 segments: int | None = 16, context_encoder: ContextEncoder | None = None,
                 consistency: float = 0.0):
        held: Variables | None = None
        self.image_processor = None
        match backbone:
            case Decide():
                model, held = backbone.model, backbone.variables
                self.image_processor = backbone.processor
                head = head or model.head
                context_encoder = context_encoder or model.context_encoder
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
        self.model = DecisionModel(module, head, context_encoder or "table")
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
        if not consistency >= 0.0:
            raise ValueError(f"consistency weighs a divergence, so it is nonnegative, got {consistency}")
        self.consistency = consistency
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
            parts["head"] = self.model.initialize_head(head_key, parts["backbone"], tokens)
        return joined(parts)

    @property
    def context_length(self) -> int | None:
        """The most context tokens a row carries (`DecisionModel.context_length`)."""
        return self.model.context_length

    @property
    def schema_in_context(self) -> bool:
        """Whether a row's schema is its context (`JointLayout.schema_in_context`)."""
        return isinstance(self.layout, JointLayout) and self.layout.schema_in_context

    @functools.cached_property
    def bounds(self) -> Bounds:
        """The most rows, questions and options a window packs (`Layout.bounds`),
        at most `segments` rows, and one where rows carry a context: a backbone's
        cross-attention reads one context a window row, the row's own."""
        bounds = self.layout.bounds(self.tokenizer, self.specials, self.context_length)
        most = 1 if self.context_length is not None else self.segments
        if most is None:
            return bounds
        return dataclasses.replace(bounds, segments=min(most, bounds.segments))

    def loss(self, variables: Variables, batch: Batch, step: Step):
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.schema_in_context)
        logits = self.model.logits(
            variables, inputs, train=step.training, rngs=model_rngs(step.key, training=step.training)
        )
        return self._charged(logits, inputs, batch)

    def _charged(self, logits: jax.Array, inputs: DecisionInputs, batch: Batch
                 ) -> tuple[Ratio | DecisionStatistics, Aux]:
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
        if not self.consistency:
            return loss, Aux(metrics={"accuracy": accuracy})
        # Each question's group in its window, and each option slot's canonical option: the group's
        # members' answers over the canonical options (`categorical.group_divergence`).
        divergence, members = categorical.group_divergence(logits, jnp.asarray(batch["groups"]),
                                                           jnp.asarray(batch["canonical"]), options)
        term = Ratio(self.row_mean(divergence, batch).total, self.row_mean(members, batch).total)
        return (DecisionStatistics(loss, term),
                Aux(metrics={"accuracy": accuracy, "consistency": term.mean()[0]}))

    def reduce_loss(self, stats: Ratio | DecisionStatistics) -> tuple[jax.Array, jax.Array]:
        """The label loss, plus `consistency` times the groups' mean divergence where it adds one."""
        if isinstance(stats, Ratio):
            return stats.mean()
        label, active = stats.label.mean()
        divergence, grouped = stats.consistency.mean()
        return label + self.consistency * divergence, active | grouped

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Decisions:
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.schema_in_context)
        return self._decisions(self._logits(self.evaluation_variables(params, step), inputs), inputs, batch)

    def _validation_scores(self, params: Variables, batch: Batch, step: Step
                           ) -> tuple[Ratio | DecisionStatistics, Decisions]:
        inputs = laid_out(batch, self.bounds.segments, spans_in_context=self.schema_in_context)
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
        grouped = bool(self.consistency)
        training = ConfigureLayout(self.layout, self.tokenizer, self.specials, self.shuffle_options,
                                   self.none_of_the_above, train=True, context_length=self.context_length,
                                   grouped=grouped)
        evaluating = dataclasses.replace(training, train=False)
        fit = Fit(self.layout, self.tokenizer, self.specials, self.model.head, self.context_length, grouped,
                  self.bounds, self.none_of_the_above)
        held = dataclasses.replace(fit, none_of_the_above=0.0)
        # Rows a record spreads into: a layout of a question a row's, or grouped examples' apart.
        expand = (QuestionRows(),)
        options = self.bounds.options if self.layout.joint else None
        compact = None if options is None else -(-options // OPTION_TILE) * OPTION_TILE
        return ObjectiveData(
            train=(fit, *expand, Configured(training, needs=("slots",), scope="run", random=True)),
            evaluate=(held, *expand, Configured(evaluating, needs=("slots",), scope="run")),
            pack=packing(self.layout, self.bounds, self.context_length, grouped),
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

        return DecideSettings(
            self.model.head.record(),
            self.layout,
            self.specials, recorded_tokenizer(RunProcessor(self.tokenizer)),
            # A decision task reads text through its own tokenizer, so only a media processor is kept.
            None if self.image_processor is None or self.image_processor.directory is None
            else self.image_processor.source,
            self.model.context_encoder)

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
