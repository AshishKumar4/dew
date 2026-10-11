"""Labelled requests: what a decision model trains and calibrates on."""

import dataclasses
import functools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np

from dew import records
from dew.data.pipeline import Configured, HostStep, Pipeline
from dew.data.recipes import HoldOut, Split, Validation, _branches, source_argument
from dew.data.sources.base import Read, ShardManifest, Source
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
