"""The decision objective's training rows: examples fitted to the layout's rows and grouped examples' units
(`Fit`), encoded with their augmentations (`Encoding`), packed into windows (`PackedRows`), and the
fields a row carries (`TRAINING_FIELDS`); `DecisionObjective.data` assembles them."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import grain.python as pygrain
import numpy as np

from dew._records import integer, record
from dew.batch import Batch
from dew.data.pipeline import Pack, Rejected
from dew.data.text import Tokenizer
from dew.decision.data import Example, GroupedExamples
from dew.decision.head import Head
from dew.decision.layout import (
    OPTIONS,
    QUESTION_FIELDS,
    Bounds,
    Encoded,
    Layout,
    QuestionField,
    QuestionLayout,
    Specials,
    compacted,
    labelled_rows,
    question_fields,
    question_owners,
    with_context,
)
from dew.decision.questions import Choice, Question

NONE_OF_THE_ABOVE = "none of the above"


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
                augmented, label = _none_of_the_above(question, labels[name], drop=rng.random() < 0.5)
                # A question the augmentation leaves as it is (its own "none of the above" right, or
                # beside the right option) keeps its own option and the canonical one its group maps.
                if augmented.options != question.options:
                    gone = [key for key in question.options if key not in augmented.options]
                    # The augmentation's own option: it stands for the option it replaced, or for none.
                    dropped[name] = gone[0] if gone else None
                    question, gold = augmented, np.eye(len(augmented.options))[label]
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
    right option when `drop`; the new label. A question that offers the option
    already keeps it where it stands, as its own description writes it, and
    one whose right option it is stays as it is, so the option added never
    shortens a question, nor does dropping lengthen it past the one added."""
    described = dict(zip(question.options, question.descriptions, strict=True))
    right = question.options[label]
    if drop and right != NONE_OF_THE_ABOVE:
        described.pop(right)
    described.setdefault(NONE_OF_THE_ABOVE, "none of the other options fits")
    options = list(described)
    return Choice(question.instructions, described), options.index(NONE_OF_THE_ABOVE if drop else right)


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
        spreads them), where no consistency term reads their groups: an example that does not fit is
        left out as it would be alone, its siblings kept."""
        rows, reasons = [], set()
        for example in grouped.examples:
            labels = example.labels()
            if not labels:
                reasons.add("unlabelled")
                continue
            asked = (example.questions if self.layout.joint
                     else {name: example.questions[name] for name in labels})
            fitted = self._rows(example, asked)
            if isinstance(fitted, Rejected):
                reasons.add(fitted.reason)
                continue
            rows.extend(fitted)
        if rows:
            return {"decision_rows": tuple(rows)}
        return Rejected("unfit" if "unfit" in reasons else "unlabelled")

    def _longest(self, row: FittedRow) -> int:
        """The most tokens `row` encodes to where the training encoding may give each answered choice a
        "none of the above" (`Encoding`): the longest of the row as it is, with the option added beside
        every answered choice's options, and with it in place of each one's right option. A row of a
        question each is these three exactly, since its layout may trim its options together
        (`QuestionLayout`), so that adding one can shorten the row; a row of several questions lays
        them out untrimmed, where adding the option beside every one is the longest: it lengthens each
        question by the option or leaves it as it is (`_none_of_the_above`), and the state takes the
        room the rest leaves."""
        if not self.none_of_the_above:
            return row.length
        example = row.example
        labels = example.labels()

        def laid_out(drop: bool) -> int:
            asked = {name: (_none_of_the_above(question, labels[name], drop=drop)[0]
                            if name in example.answers and isinstance(question, Choice) else question)
                     for name in row.names for question in (example.questions[name],)}
            laid = self.layout.rows(self.tokenizer, self.specials, example.state, asked,
                                    tokens=row.tokens.tolist())
            return sum(len(each.tokens) for each in laid)

        return max(row.length, laid_out(drop=False), laid_out(drop=True))

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
