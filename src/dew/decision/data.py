"""Labelled requests: what a decision model trains and calibrates on."""

import dataclasses
import functools
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np

from dew import records
from dew.data.dataset import Dataset, DatasetSpec, Tokenize
from dew.data.pipeline import Configured, HostStep, Pipeline
from dew.data.recipes import HoldOut, Split, Validation, _branches, source_argument
from dew.data.sources.base import Read, ShardManifest, Source
from dew.data.sources.hf import HFDatasetSource, HFOptions
from dew.decision.questions import Choice, Question
from dew.records import JSON, json_value, record


@dataclass(frozen=True)
class Example:
    """One state, its questions, and the right answer to some of them.

    An answer names an option by its key or its index (`Question.index`), so
    a dataset's integer label column can be used as it is. Where the gold is a
    distribution, as a teacher's or a panel's is, `targets` gives it instead:
    one probability per option, in the question's own order. A question with
    neither is asked but not scored.
    """

    state: JSON
    questions: Mapping[str, Question]
    answers: Mapping[str, str | int] = field(default_factory=dict)
    targets: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self):
        unknown = (set(self.answers) | set(self.targets)) - set(self.questions)
        if unknown:
            raise ValueError(f"answers name questions the example does not ask: {sorted(unknown)}")
        both = set(self.answers) & set(self.targets)
        if both:
            raise ValueError(f"questions {sorted(both)} have both an answer and a target distribution")
        for name, label in self.answers.items():
            self.questions[name].index(label)
        for name, target in self.targets.items():
            count = len(self.questions[name].options)
            if len(target) != count or min(target) < 0 or abs(sum(target) - 1) > 1e-3:
                raise ValueError(
                    f"targets.{name} is a distribution over the question's {count} options, "
                    f"got {list(target)}"
                )

    @classmethod
    def of(cls, row: "Example | Mapping[str, object]") -> "Example":
        """Return an example as it is, or build one from a row with `state`, `questions` and `answers`.

        The questions may be `Question`s or in Jev's wire format.
        """
        if isinstance(row, Example):
            return row
        questions = {}
        for name, question in record(row.get("questions"), "questions").items():
            questions[name] = (
                question
                if isinstance(question, Question)
                else Question.from_wire(record(question, f"questions.{name}"))
            )
        answers = {}
        for name, label in record(row.get("answers", {}), "answers").items():
            if isinstance(label, bool) or not isinstance(label, (str, int)):
                raise ValueError(f"answers.{name} names an option by its key or index, got {label!r}")
            answers[name] = label
        targets = {}
        for name, target in record(row.get("targets", {}), "targets").items():
            if name in questions:
                where = f"targets.{name}"
                targets[name] = _distribution(questions[name], json_value(target, where), where)
        return cls(json_value(row.get("state"), "state"), questions, answers, targets)

    def labels(self) -> dict[str, int]:
        """Return each answered question's right option, as an index: a target's most likely one."""
        labels = {name: self.questions[name].index(label) for name, label in self.answers.items()}
        return labels | {name: int(np.argmax(target)) for name, target in self.targets.items()}

    def distribution(self, name: str) -> np.ndarray:
        """Return question `name`'s gold as a distribution over its options.

        That is its target, or else all of the mass on its answer.
        """
        if name in self.targets:
            target = np.asarray(self.targets[name], np.float64)
            return target / target.sum()
        return np.eye(len(self.questions[name].options))[self.labels()[name]]


@dataclass(frozen=True)
class Weighted:
    """Examples that fill `weight`, a share of every training step, in a mixture of several.

    `DecisionObjective.dataset` reads a mapping of names to these as one stream,
    each set at its share whatever its length (`dew.data.dataset.mixture`).
    """

    examples: Sequence["Example | Mapping[str, object]"]
    """`Example`s, or rows `Example.of` reads."""
    weight: float


@dataclass(frozen=True)
class DecisionMixture:
    """Labelled sets of examples, each at its share of every training step.

    `root` holds each set as JSON-lines rows `Example.of` reads, `<name>.jsonl`,
    and the rows it holds back for validation and calibration as
    `<name>.held.jsonl`, where it holds some back. Each set's share is the one
    `<root>/mixture.json` records, as `recipes/decision/sources.py` writes it
    beside the sets, unless `weights` gives another: a weight there adds a
    set of your own, and a weight of 0 leaves a set out.
    """

    root: str = "data/mixture"
    weights: dict[str, float] = field(default_factory=dict)

    def recorded(self) -> JSON:
        """What `<root>/mixture.json` records about how the sets were made, or nothing."""
        path = Path(self.root) / "mixture.json"
        return json_value(json.loads(path.read_text()), "mixture.json") if path.is_file() else {}

    def shares(self) -> dict[str, float]:
        """Each set's share by name, those of weight 0 left out."""
        recorded = record(record(self.recorded(), "mixture.json").get("weights", {}), "weights")
        stated = {name: records.number(weight, f"weights.{name}") for name, weight in recorded.items()}
        shares = {name: float(weight) for name, weight in (stated | self.weights).items() if weight > 0}
        if not shares:
            raise ValueError(
                f"no set has a weight: give --mixture.weights, or write {self.root}/mixture.json"
            )
        return shares

    def read(self) -> tuple[dict[str, Weighted], list[Example]]:
        """Every set at its share, and the rows they hold back, together."""
        weighted, held = {}, []
        for name, weight in self.shares().items():
            weighted[name] = Weighted(_lines(Path(self.root) / f"{name}.jsonl"), weight)
            kept = Path(self.root) / f"{name}.held.jsonl"
            if kept.is_file():
                held.extend(_lines(kept))
        return weighted, held


def _lines(path: Path) -> list[Example]:
    with path.open() as file:
        return [Example.of(json.loads(line)) for line in file if line.strip()]


def _distribution(question: Question, target: JSON, name: str) -> tuple[float, ...]:
    """A target distribution from a row: probabilities in the question's option order, or keyed by option."""
    match target:
        case list():
            return tuple(records.number(value, name) for value in target)
        case dict():
            if set(target) != set(question.options):
                raise ValueError(f"{name} keys {sorted(target)}, not the options {list(question.options)}")
            return tuple(records.number(target[option], f"{name}.{option}") for option in question.options)
        case _:
            raise ValueError(f"{name} is a list of probabilities or a mapping of options to them")


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
        raise TypeError(
            "a decision table is laid out by its objective's tokenizer: "
            "DecisionObjective.dataset(table, batch=...)"
        )

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
        return (
            [example for index, example in enumerate(examples) if index not in held],
            [example for index, example in enumerate(examples) if index in held],
        )

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
        names = (
            self.options
            or _class_names(source, self.label)
            or tuple(dict.fromkeys(str(row[self.label]) for row in rows))
        )
        question = Choice(self.instructions, list(names))
        return [
            Example(
                json_value(row[self.text], self.text),
                {self.question: question},
                {self.question: _label(row[self.label])},
            )
            for row in rows
        ]


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


@dataclass(frozen=True)
class ParseExample:
    def __call__(self, row: Mapping[str, object]) -> Mapping[str, object]:
        return {"example": Example.of(row)}


@dataclass(frozen=True)
class ParseTable:
    options: tuple[str, ...]
    text: str = "text"
    label: str = "label"
    question: str = "label"
    instructions: str = "Classify the input into the correct category."

    def __call__(self, row: Mapping[str, object]) -> Mapping[str, object]:
        if "state" in row and "questions" in row:
            return ParseExample()(row)
        label = row[self.label]
        if not isinstance(label, (str, int, float, bool)):
            raise ValueError(f"decision label {self.label} must name an option")
        question = Choice(self.instructions, list(self.options))
        return {
            "example": Example(
                json_value(row[self.text], self.text),
                {self.question: question},
                {self.question: _label(label)},
            )
        }


@dataclass(frozen=True)
class _DecisionSource:
    source: Source
    slots: Mapping[str, int] | None = None
    table: ParseTable | None = None

    def describe(self) -> str:
        return self.source.describe()

    def snapshot(self) -> Mapping[str, object]:
        return self.source.snapshot()

    def supports(self, read: Read) -> bool:
        return self.source.supports(read)

    def records(self, read: Read):
        records = self.source.records(read)
        if read == "memory" and self.slots is None and not self._facts:
            if self.table is not None:
                stated = self.source.metadata().get("class_names", {})
                names = record(stated, "class names").get(self.table.label)
                first = records[0] if len(records) else None
                if isinstance(names, (list, tuple)):
                    self._classes.extend(str(name) for name in names)
                elif isinstance(first, Mapping) and "questions" not in first and "example" not in first:
                    if self.table.label not in first:
                        raise ValueError(f"decision table needs column {self.table.label!r}")
                    self._classes.extend(dict.fromkeys(str(record(records[index], "decision row")[self.table.label])
                                                      for index in range(len(records))))
            self._memory_slots(records)
        return records

    @functools.cached_property
    def _classes(self) -> list[str]:
        return []

    @functools.cached_property
    def _facts(self) -> dict[str, int]:
        return dict(self.slots or {})

    def _memory_slots(self, rows) -> None:
        for index in range(len(rows)):
            row = rows[index]
            if row is None:
                continue
            if self.table is not None and "questions" not in row and "example" not in row:
                self._facts.update(questions=1, options=len(self.table.options or self._classes))
                continue
            example = row.get("example")
            example = Example.of(row) if example is None else Example.of(example)
            self._facts["questions"] = max(self._facts.get("questions", 0), len(example.questions))
            self._facts["options"] = max(
                self._facts.get("options", 0),
                max((len(question.options) for question in example.questions.values()), default=0),
            )

    def metadata(self) -> Mapping[str, object]:
        stated = self.source.metadata()
        if self.table is not None:
            names = (self.table.options or tuple(self._classes) or
                     records.strings(record(stated.get("class_names", {}), "class names").get(
                         self.table.label, ()), "class names"))
            if names:
                return {**stated, "slots": {"questions": 1, "options": len(names)}, "decision_options": names}
        return {**stated, "slots": self._facts} if self._facts else stated

    def lengths(self) -> np.ndarray | None:
        return self.source.lengths()

    def manifest(self) -> ShardManifest:
        return self.source.manifest()


@dataclass(frozen=True)
class Decisions:
    """Decision rows and their corpus shares, with layout supplied by the objective."""

    source: Annotated[str | Source | Sequence[Example | Mapping[str, object]] | Mapping[
        str, Sequence[Example | Mapping[str, object]]], source_argument()] = "tickets.csv"
    validation: Annotated[Validation | Sequence[Example | Mapping[str, object]], source_argument()] = None
    weights: Mapping[str, float] = field(default_factory=dict)
    options: tuple[str, ...] = ()
    text: str = "text"
    label: str = "label"
    question: str = "label"
    instructions: str = "Classify the input into the correct category."
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        from dew.data.sources import Files, Memory, source
        from dew.data.sources.concatenate import Concatenate
        from dew.data.storage import Store

        table = ParseTable(self.options, self.text, self.label, self.question, self.instructions)
        if isinstance(self.source, Mapping):
            branches = [dataclasses.replace(
                dataclasses.replace(self, source=rows, validation=None, weights={}).pipelines()[0], name=name)
                        for name, rows in sorted(self.source.items())]
            weights = [self.weights.get(name, 1.) for name in sorted(self.source)]
            train = branches[0] if len(branches) == 1 else branches[0].mix(*branches[1:], weights=weights)
            held = None if self.validation is None else dataclasses.replace(
                self, source=(), weights={}).pipelines()[1]
            return train, held, ()
        if isinstance(self.source, str):
            root = Store(self.source)
            fs, path = root.resolve()
            mixture = path.rstrip("/") + "/mixture.json"
            if fs.exists(mixture):
                with fs.open(mixture, "rb") as handle:
                    stated = record(json.load(handle), "mixture.json")
                shares = {
                    name: records.number(value, f"weights.{name}")
                    for name, value in record(stated.get("weights", {}), "weights").items()
                } | dict(self.weights)
                slots = {
                    name: records.integer(value, f"slots.{name}")
                    for name, value in record(stated.get("slots", {}), "slots").items()
                }
                files = record(stated.get("files", {}), "mixture files")
                training, held, weights = [], [], []
                for name, weight in sorted(shares.items()):
                    if weight <= 0:
                        continue
                    corpus = record(files.get(name, {}), f"files.{name}")
                    for split, destination in (("train", training), ("held", held)):
                        entry = record(corpus.get(split, {}), f"files.{name}.{split}")
                        shards = entry.get("shards")
                        if isinstance(shards, list):
                            paths = tuple(
                                self.source.rstrip("/")
                                + "/"
                                + records.text(record(shard, "shard")["path"], "shard.path")
                                for shard in shards
                            )
                        else:
                            url = (
                                self.source.rstrip("/") + f"/{name}{'.held' if split == 'held' else ''}.jsonl"
                            )
                            shard_fs, shard_path = Store(url).resolve()
                            paths = (url,) if shard_fs.exists(shard_path) else ()
                        if paths:
                            index = record(entry.get("index", {}), "index").get("path")
                            source_index = (
                                None
                                if index is None
                                else self.source.rstrip("/") + "/" + records.text(index, "index.path")
                            )
                            dataset = _DecisionSource(
                                Files(paths, format="jsonl", index=source_index), slots or None
                            )
                            destination.append(Pipeline(dataset, name=name).map(ParseExample()))
                    weights.append(weight)
                if not training:
                    raise ValueError(f"no corpus of {self.source} has a positive weight")
                train = training[0] if len(training) == 1 else training[0].mix(*training[1:], weights=weights)
                evaluated = (
                    None
                    if not held
                    else Pipeline(
                        _DecisionSource(
                            Concatenate(
                                tuple(child.source for child in held if isinstance(child.source, Source))
                            ),
                            slots or None,
                        )
                    ).map(ParseExample())
                )
                return train, evaluated, ()
        if isinstance(self.source, Sequence) and not isinstance(self.source, str):
            rows = tuple({"example": row} if isinstance(row, Example) else row for row in self.source)
            dataset = _DecisionSource(Memory(rows), table=table)
            dataset.records("memory")
            if any("example" not in row for row in rows):
                dataset.records("memory")
                train = Pipeline(dataset, name="examples").map(Configured(_ParseConfigured(table), needs=()))
            else:
                train = Pipeline(dataset, name="examples")
        else:
            dataset = _DecisionSource(
                source(self.source, split=self.split),
                {"questions": 1, "options": len(self.options)} if self.options else None,
                table=table,
            )
            train = Pipeline(dataset, name="examples").map(Configured(_ParseConfigured(table), needs=()))
        if self.validation is None:
            evaluated = None
        elif isinstance(self.validation, HoldOut):
            train, evaluated = _branches(dataset, self.validation, split=self.split, prefix=train.stages)
        elif isinstance(self.validation, Pipeline):
            evaluated = self.validation
        elif isinstance(self.validation, Split):
            held_source = _DecisionSource(source(self.source, split=self.validation.name), table=table)
            evaluated = Pipeline(held_source).map(Configured(_ParseConfigured(table), needs=()))
        elif isinstance(self.validation, Sequence) and not isinstance(self.validation, str):
            rows = tuple({"example": Example.of(row)} for row in self.validation)
            held_source = _DecisionSource(Memory(rows))
            held_source._memory_slots(rows)
            evaluated = Pipeline(held_source, name="held out")
        else:
            evaluated = Pipeline(_DecisionSource(source(self.validation), table=table)).map(
                Configured(_ParseConfigured(table), needs=()))
        return train, evaluated, ()


@dataclass(frozen=True)
class _ParseConfigured:
    table: ParseTable

    def __call__(self, metadata: Mapping[str, object]) -> ParseTable | ParseExample:
        names = metadata.get("decision_options")
        if names is None:
            if "slots" in metadata:
                return ParseExample()
            raise ValueError("decision flat table needs options, ClassLabel metadata or read='memory'")
        return dataclasses.replace(self.table, options=records.strings(names, "decision_options"))
