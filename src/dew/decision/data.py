"""Labelled requests: what a decision model trains and calibrates on."""

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dew.data.dataset import Dataset, DatasetSpec, Tokenize
from dew.data.sources.hf import HFDatasetSource, HFOptions
from dew.decision.questions import Choice, Question
from dew.records import JSON, json_value, record


@dataclass(frozen=True)
class Example:
    """One state, its questions, and the right answer to some of them.

    An answer names an option by its key or its index (`Question.index`), so
    a dataset's integer label column can be used as it is. A question with no
    answer is asked but not scored.
    """

    state: JSON
    questions: Mapping[str, Question]
    answers: Mapping[str, str | int] = field(default_factory=dict)

    def __post_init__(self):
        unknown = set(self.answers) - set(self.questions)
        if unknown:
            raise ValueError(f"answers name questions the example does not ask: {sorted(unknown)}")
        for name, label in self.answers.items():
            self.questions[name].index(label)

    @classmethod
    def of(cls, row: "Example | Mapping[str, object]") -> "Example":
        """Return an example as it is, or build one from a row with `state`, `questions` and `answers`.

        The questions may be `Question`s or in Jev's wire format.
        """
        if isinstance(row, Example):
            return row
        questions = {}
        for name, question in record(row.get("questions"), "questions").items():
            questions[name] = (question if isinstance(question, Question)
                               else Question.from_wire(record(question, f"questions.{name}")))
        answers = {}
        for name, label in record(row.get("answers", {}), "answers").items():
            if isinstance(label, bool) or not isinstance(label, (str, int)):
                raise ValueError(f"answers.{name} names an option by its key or index, got {label!r}")
            answers[name] = label
        return cls(json_value(row.get("state"), "state"), questions, answers)

    def labels(self) -> dict[str, int]:
        """Return each answered question's right option, as an index."""
        return {name: self.questions[name].index(label) for name, label in self.answers.items()}


_FORMATS = {".csv": "csv", ".json": "json", ".jsonl": "json", ".parquet": "parquet"}


@dataclass(frozen=True)
class DecisionTable(DatasetSpec):
    """A table of labelled text, read as examples of one choice question.

    This is how Laya's `laya-train --data tickets.csv` reads a table. `path` is a
    .csv, .json, .jsonl or .parquet file, or a Hugging Face dataset id read at
    `split`; every format is read through `datasets`. Each row's `text` column is
    the state, and its `label` column is the answer to the question `question`,
    asked with `instructions` over `options`. Without `options`, the options are
    the label column's class names where it has them (BANKING77's, CLINC150's),
    and otherwise its distinct values in the order they first appear. A row whose
    label is an int names its option by index. A table with `state` and
    `questions` columns is read as whole examples instead (`Example.of`),
    including its `answers` column.

    A `held_out` share of the rows, at most `held_out_most`, is drawn from `seed`
    for validation and calibration (Laya's 10% and 400), unless
    `validation_split` names a split to hold out instead.
    `DecisionObjective.dataset` lays the examples out; `load` alone cannot,
    because the layout depends on the objective's tokenizer.
    """

    path: str = "tickets.csv"
    split: str = "train"
    text: str = "text"
    label: str = "label"
    question: str = "label"
    instructions: str = "Classify the input into the correct category."
    options: tuple[str, ...] = ()
    held_out: float = 0.1
    held_out_most: int = 400
    validation_split: str | None = None
    revision: str | None = None
    """The Hub dataset's commit."""

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        raise TypeError("a decision table is laid out by its objective's tokenizer: "
                        "DecisionObjective.dataset(table, batch=...)")

    def examples(self) -> tuple[list[Example], list[Example]]:
        """Return the training examples and the held-out ones."""
        rows = self._rows(self.split)
        if self.validation_split is not None:
            return self._read(rows), self._read(self._rows(self.validation_split))
        examples = self._read(rows)
        order = list(range(len(examples)))
        random.Random(self.seed).shuffle(order)
        count = min(self.held_out_most, int(len(examples) * self.held_out))
        held = set(order[:count])
        return ([example for index, example in enumerate(examples) if index not in held],
                [example for index, example in enumerate(examples) if index in held])

    def _rows(self, split: str) -> HFDatasetSource:
        suffix = Path(self.path).suffix.lower()
        if suffix in _FORMATS:
            files = HFOptions(data_files={split: self.path})
            return HFDatasetSource(_FORMATS[suffix], split=split, options=files)
        return HFDatasetSource(self.path, split=split, options=HFOptions(revision=self.revision))

    def _read(self, source: HFDatasetSource) -> list[Example]:
        rows = [source[index] for index in range(len(source))]
        if {"state", "questions"} <= set(source.columns):
            return [Example.of(row) for row in rows]
        missing = {self.text, self.label} - set(source.columns)
        if missing:
            raise ValueError(f"{self.path} has no column {sorted(missing)}; its columns are {source.columns}")
        names = self.options or _class_names(source, self.label) or tuple(
            dict.fromkeys(str(row[self.label]) for row in rows))
        question = Choice(self.instructions, list(names))
        return [Example(json_value(row[self.text], self.text), {self.question: question},
                        {self.question: _label(row[self.label])}) for row in rows]


def _class_names(source: HFDatasetSource, column: str) -> tuple[str, ...]:
    """A `datasets` ClassLabel column's names, or nothing for any other column."""
    from datasets import ClassLabel

    feature = source.features[column]
    return tuple(feature.names) if isinstance(feature, ClassLabel) else ()


def _label(value: str | int | float | bool) -> str | int:
    """A label as an option's key, or its index when the column holds ints."""
    match value:
        case bool() | float():
            return str(value)
        case _:
            return value
