"""Labelled requests: what a decision model trains and calibrates on."""

import dataclasses
import functools
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np

from dew import _records
from dew._records import JSON, json_value, record
from dew.data.lowering import prepared_rows
from dew.data.pipeline import Configured, Dataset, Pipeline, Stage, mix
from dew.data.sources.base import Read, Rows, Source, View, _consumed, stated as store_facts
from dew.decision.questions import Choice, Question


@dataclass(frozen=True)
class Group:
    """Questions of one example that ask the same thing (negated, paraphrased, its options re-ordered),
    named `name`: `members` maps each one's every option key onto a canonical key the members share,
    so a consistency term (`DecisionObjective.consistency`) can hold their answers to one distribution
    over the canonical keys."""

    name: str
    members: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class Example:
    """One state, its questions, and the right answer to some of them.

    An answer names an option by its key or its index (`Question.index`), so
    a dataset's integer label column can be used as it is. Where the gold is a
    distribution, as a teacher's or a panel's is, `targets` gives it instead:
    one probability per option, in the question's own order. A question with
    neither is asked but not scored.

    `context` is a sequence the decision reads beside its state, through the
    backbone's cross-attention layers (`DecisionObjective`'s context), rendered
    as text; None, or a backbone without those layers, reads none.

    `groups` name questions that ask the same thing (`Group`), each question in
    one at most; an objective without a consistency term reads none.
    """

    state: JSON
    questions: Mapping[str, Question]
    answers: Mapping[str, str | int] = field(default_factory=dict)
    targets: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    context: JSON | None = None
    groups: tuple[Group, ...] = ()

    def __post_init__(self):
        _asked(self.questions, [*self.answers, *self.targets])
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
        for group in self.groups:
            _asked(self.questions, list(group.members))
        def question(name: str) -> Question:
            return self.questions[name]
        _checked_groups([(group.name, group.members) for group in self.groups], question)

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
        stated = record(row.get("targets", {}), "targets")
        _asked(questions, stated)
        targets = {}
        for name, target in stated.items():
            where = f"targets.{name}"
            targets[name] = _distribution(questions[name], json_value(target, where), where)
        context = row.get("context")
        return cls(json_value(row.get("state"), "state"), questions, answers, targets,
                   None if context is None else json_value(context, "context"),
                   groups_of(json_value(row.get("groups"), "groups")))

    def group_of(self, name: str) -> tuple[int, Group] | None:
        """The index and group question `name` stands in, or None."""
        return next(((index, group) for index, group in enumerate(self.groups) if name in group.members),
                    None)

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


def _checked_groups(groups: Sequence[tuple[str, Mapping[str, Mapping[str, str]]]],
                    question: Callable[[str], Question]) -> None:
    """Refuse groups that do not ask one thing: each `(name, members)` two members or more, no member
    in two groups, and each member mapping every option of its question (`question(member)`) onto
    canonical keys its group's members share."""
    grouped: set[str] = set()
    for name, members in groups:
        if len(members) < 2:
            raise ValueError(f"group {name!r} holds {len(members)} question; a group holds two or more")
        if grouped & set(members):
            raise ValueError(f"questions {sorted(grouped & set(members))} stand in two groups")
        grouped |= set(members)
        canonical = None
        for member, mapped in members.items():
            options = question(member).options
            if set(mapped) != set(options):
                raise ValueError(f"group {name!r} maps question {member!r}'s options {sorted(mapped)}, "
                                 f"not its {list(options)}")
            keys = set(mapped.values())
            if canonical is not None and keys != canonical:
                raise ValueError(f"group {name!r}'s members map onto other canonical keys: "
                                 f"{sorted(canonical)} and {sorted(keys)}")
            canonical = keys


def _member(key: str) -> tuple[int, str]:
    """A grouped examples' member key, `"<example index>.<question>"`, as its example and question."""
    index, _, question = key.partition(".")
    if not index.isdigit() or index != str(int(index)) or not question:
        raise ValueError(f"a member of grouped examples is named '<example index>.<question>', its index "
                         f"written once (0, not 00), got {key!r}")
    return int(index), question


@dataclass(frozen=True)
class GroupedExamples:
    """Examples whose rows train together, so a consistency term (`DecisionObjective.consistency`)
    holds questions asked across them to one answer: the same question negated, paraphrased or its
    options re-ordered in rows of their own, under any layout.

    `groups` names each set of such questions (`Group`), its members keyed `"<example index>.<question>"`
    by the example's index in `examples` and the question's name there, each mapping its question's
    every option key onto canonical keys the members share, as an example's own groups do. An example's
    own groups hold too, numbered after these. The examples' rows train in one packed window, as a unit
    the packer does not split (`DecisionObjective.data`): a unit too long for a window is refused."""

    examples: tuple[Example, ...]
    groups: tuple[Group, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "examples", tuple(Example.of(example) for example in self.examples))
        if not self.examples:
            raise ValueError("grouped examples hold one example or more")
        own = {f"{index}.{member}" for index, example in enumerate(self.examples)
               for group in example.groups for member in group.members}
        for group in self.groups:
            for key in group.members:
                index, asked = _member(key)
                if index >= len(self.examples) or asked not in self.examples[index].questions:
                    raise ValueError(f"group {group.name!r} names question {asked!r} of example {index}, "
                                     f"which the {len(self.examples)} examples do not ask")
            if own & set(group.members):
                raise ValueError(f"group {group.name!r} holds questions {sorted(own & set(group.members))} "
                                 "an example already groups")

        def question(key: str) -> Question:
            index, asked = _member(key)
            return self.examples[index].questions[asked]
        _checked_groups([(group.name, group.members) for group in self.groups], question)

    def group_of(self, index: int, name: str) -> tuple[int, Mapping[str, str]] | None:
        """The unit-wide index of the group example `index`'s question `name` stands in, counting each
        example's own groups after these, and the member's map onto its canonical keys, or None."""
        key = f"{index}.{name}"
        for number, group in enumerate(self.groups):
            if key in group.members:
                return number, group.members[key]
        number = len(self.groups) + sum(len(example.groups) for example in self.examples[:index])
        found = self.examples[index].group_of(name)
        return None if found is None else (number + found[0], found[1].members[name])

    @classmethod
    def of(cls, row: "GroupedExamples | Mapping[str, object]") -> "GroupedExamples":
        """Return grouped examples as they are, or build them from a row with `examples` (each as
        `Example.of` reads one) and `groups` (`groups_of`), its members named
        `"<example index>.<question>"`."""
        if isinstance(row, GroupedExamples):
            return row
        held = json_value(row.get("examples"), "examples")
        if not isinstance(held, list):
            raise ValueError(f"examples is a list of examples, got {type(held).__name__}")
        examples = [Example.of(record(example, f"examples.{index}")) for index, example in enumerate(held)]
        return cls(tuple(examples), groups_of(json_value(row.get("groups"), "groups")))


def groups_of(value: JSON) -> tuple[Group, ...]:
    """A row's `groups`, as `Example.groups` holds them: a mapping of each group's name to its members,
    each member's option keys mapped to canonical keys, or None for none."""
    if value is None:
        return ()
    groups = []
    for name, members in record(value, "groups").items():
        mapped = {}
        for question, options in record(members, f"groups.{name}").items():
            mapped[question] = {key: _records.text(canonical, f"groups.{name}.{question}.{key}")
                                for key, canonical in record(options, f"groups.{name}.{question}").items()}
        groups.append(Group(name, mapped))
    return tuple(groups)


def _asked(questions: Mapping[str, Question], named: Iterable[str]) -> None:
    """Refuse an answer or a target to a question the example does not ask."""
    unknown = set(named) - set(questions)
    if unknown:
        raise ValueError(f"answers or targets name questions the example does not ask: {sorted(unknown)}")


def _distribution(question: Question, target: JSON, name: str) -> tuple[float, ...]:
    """A target distribution from a row: probabilities in the question's option order, or keyed by option."""
    match target:
        case list():
            return tuple(_records.number(value, name) for value in target)
        case dict():
            if set(target) != set(question.options):
                raise ValueError(f"{name} keys {sorted(target)}, not the options {list(question.options)}")
            return tuple(_records.number(target[option], f"{name}.{option}") for option in question.options)
        case _:
            raise ValueError(f"{name} is a list of probabilities or a mapping of options to them")


def _label(value: str | int | float | bool) -> str | int:
    """A label as an option's key, or its index when the column holds ints."""
    match value:
        case bool() | float():
            return str(value)
        case _:
            return value



def _slots(stated: Mapping[str, object]) -> dict[str, int]:
    """The question and option slots a mixture's manifest states its rows need."""
    return {name: _records.integer(value, f"slots.{name}")
            for name, value in record(stated.get("slots", {}), "slots").items()}

@dataclass(frozen=True)
class ParseExample:
    """A row of an example's fields as its `Example`, a row of grouped examples' fields (`examples`) as
    its `GroupedExamples`, or a row that holds either already as it is."""

    def __call__(self, row: Mapping[str, object]) -> Mapping[str, object]:
        held = row.get("example")
        if isinstance(held, (Example, GroupedExamples)):
            return {"example": held}
        return {"example": GroupedExamples.of(row) if "examples" in row else Example.of(row)}


@dataclass(frozen=True)
class ParseTable:
    """A flat table's row as a one-question `Example`. A label names its
    option by key, or by index where it is an int; where the options are the
    label column's own values (`by_value`), every label names its option by
    value, an int one included. A `context` column's value is the example's
    context (`Example.context`), none where the column holds None. A row's
    `groups` field reaches the example as `Example.of` reads it, so a group
    over the row's one question is refused as any group of one member is."""

    options: tuple[str, ...]
    text: str = "text"
    label: str = "label"
    question: str = "label"
    instructions: str = "Classify the input into the correct category."
    by_value: bool = False
    context: str | None = None

    def __call__(self, row: Mapping[str, object]) -> Mapping[str, object]:
        if ("state" in row and "questions" in row) or "examples" in row:
            return ParseExample()(row)
        label = row[self.label]
        if not isinstance(label, (str, int, float, bool)):
            raise ValueError(f"decision label {self.label} must name an option")
        question = Choice(self.instructions, list(self.options))
        column = self.context
        context = None if column is None else row[column]
        return {
            "example": Example(
                json_value(row[self.text], self.text),
                {self.question: question},
                {self.question: str(label) if self.by_value else _label(label)},
                context=None if context is None or column is None else json_value(context, column),
                groups=groups_of(json_value(row.get("groups"), "groups")),
            )
        }


def _named_by(rows, label: str, names: Sequence[object]) -> bool:
    """Whether every row's `label` names one of `names`, by its index or its name."""
    known = {str(name) for name in names}
    for index in range(len(rows)):
        row = rows[index]
        value = row.get(label) if isinstance(row, Mapping) else None
        by_index = isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(names)
        if not (by_index or (not isinstance(value, bool) and str(value) in known)):
            return False
    return True


@dataclass(frozen=True)
class _DecisionSource(View[Source]):
    """A store's rows read as decisions: `table`'s options and the slots its rows need are inferred
    in a memory read, from the rows as the stages `before` the decisions make them (a rename, a
    relabelling); `mixture` is the facts a decision mixture states of itself (`mixture`)."""

    rows: ClassVar[Rows] = "described"
    source: Source
    slots: Mapping[str, int] | None = None
    table: ParseTable | None = None
    descriptor: Mapping[str, object] = field(default_factory=dict)
    before: tuple[Stage, ...] = ()
    mixture: Mapping[str, object] = field(default_factory=dict)

    def stated(self) -> Mapping[str, object]:
        """What the store states of its records (`Dataset.facts`), with a mixture's weights and the
        manifest it was made by."""
        return {**store_facts(self.source), **self.mixture}

    def consumed(self, read: Read) -> "_DecisionSource":
        consumed = dataclasses.replace(self, source=_consumed(self.source, read))
        consumed._classes[:] = self._classes
        consumed._valued[:] = self._valued
        consumed._facts.update(self._facts)
        return consumed

    def fingerprint(self) -> Mapping[str, object]:
        return {**self.source.fingerprint(), **({"descriptor": self.descriptor} if self.descriptor else {})}

    def records(self, read: Read):
        records = self.source.records(read)
        if read == "memory" and self.slots is None and not self._facts:
            rows = prepared_rows(records, self.before) if self.before else records
            if self.table is not None:
                stated = self.source.metadata().get("class_names", {})
                names = record(stated, "class names").get(self.table.label)
                first = rows[0] if len(rows) else None
                # The store's class names hold where the stages before keep its labels: a relabelling
                # map's values are read from its rows instead.
                if isinstance(names, (list, tuple)) and (not self.before or _named_by(rows, self.table.label,
                                                                                        names)):
                    self._classes[:] = [str(name) for name in names]
                elif isinstance(first, Mapping) and not {"questions", "example", "examples"} & set(first):
                    if self.table.label not in first:
                        raise ValueError(f"decision table needs column {self.table.label!r}")
                    self._classes[:] = sorted({
                            str(record(rows[index], "decision row")[self.table.label])
                            for index in range(len(rows))
                    })
                    self._valued[:] = [True]
            self._memory_slots(rows)
        return records

    @functools.cached_property
    def _classes(self) -> list[str]:
        return []

    @functools.cached_property
    def _valued(self) -> list[bool]:
        """`[True]` where the options are the label column's own values, read
        whole, so each label names its option by value."""
        return []

    @functools.cached_property
    def _facts(self) -> dict[str, int]:
        return dict(self.slots or {})

    def _memory_slots(self, rows) -> None:
        for index in range(len(rows)):
            row = rows[index]
            if row is None:
                continue
            if self.table is not None and not {"questions", "example", "examples"} & set(row):
                self._facts.update(questions=1, options=len(self.table.options or self._classes))
                continue
            held = row.get("example")
            held = row if held is None else held
            # Grouped examples train as one unit, whose questions a window holds together.
            if isinstance(held, GroupedExamples) or (isinstance(held, Mapping) and "examples" in held):
                examples = GroupedExamples.of(held).examples
            else:
                examples = (Example.of(held),)
            asked = [question for example in examples for question in example.questions.values()]
            self._facts["questions"] = max(self._facts.get("questions", 0), len(asked))
            self._facts["options"] = max(self._facts.get("options", 0),
                                         max((len(question.options) for question in asked), default=0))

    def metadata(self) -> Mapping[str, object]:
        stated = self.source.metadata()
        if self.table is not None:
            names = (
                self.table.options
                or tuple(self._classes)
                or _records.strings(
                    record(stated.get("class_names", {}), "class names").get(self.table.label, ()),
                    "class names",
                )
            )
            if names:
                valued = not self.table.options and bool(self._valued)
                return {**stated, "slots": {"questions": 1, "options": len(names)}, "decision_options": names,
                        **({"decision_options_by_value": True} if valued else {})}
        return {**stated, "slots": self._facts} if self._facts else stated




def decisions(dataset: Dataset, *, options: Sequence[str] = (), text: str = "text", label: str = "label",
              question: str = "label",
              instructions: str = "Classify the input into the correct category.",
              context: str | None = None) -> Pipeline:
    """`dataset`'s rows as decisions, each an `Example` under `example`, for a chain to batch: a flat
    table's `text` and `label` columns read as one `question` over `options` (its ClassLabel names, or
    read whole in memory, where none are given), its `context` column, where named, each example's
    context, or rows that hold their examples already. Its stages
    before this one, a holdout's selection among them, stand; a decision chain is a store's rows, and
    a mixture of them mixes decision chains (`mixture`). Raises ValueError for a mixture or phases
    under it."""
    from dew.data.sources.base import Source

    chain = dataset._chain()
    if not isinstance(chain.source, Source):
        raise ValueError("decisions reads a store's rows; mix decision chains rather than reading a mixture")
    table = ParseTable(tuple(options), text, label, question, instructions, context=context)
    wrapped = _DecisionSource(chain.source, {"questions": 1, "options": len(options)} if options else None,
                              table=table, before=tuple(chain.stages))
    return dataclasses.replace(chain, source=wrapped).map(Configured(_ParseConfigured(table), needs=()))


def mixture(root_url: str, *, weights: Mapping[str, float] | None = None,
            descriptor: str | None = None) -> tuple[Pipeline, Pipeline | None]:
    """The decision corpora of the mixture directory `root_url` (its `mixture.json`): every corpus of a
    positive weight, the manifest's weights over `weights`, mixed by them for training, and their held
    rows together, or None where none has any. `descriptor` pins a replaced manifest by its URL."""
    from dew.data.sources import Source
    from dew.data.sources.concatenate import Concatenate

    stated, pinned = _stated(root_url, descriptor)
    shares = {
        name: _records.number(value, f"weights.{name}")
        for name, value in record(stated.get("weights", {}), "weights").items()
    } | dict(weights or {})
    training, held, kept = [], [], []
    for name, weight in sorted(shares.items()):
        if weight <= 0:
            continue
        for split, destination in (("train", training), ("held", held)):
            read = _corpus(root_url, stated, pinned, name, split)
            if read is not None:
                destination.append(read)
        kept.append(weight)
    if not training:
        raise ValueError(f"no corpus of {root_url} has a positive weight")
    # Each corpus states the mixture it is read in: its positive weights and the manifest that made
    # it, which a run's summary reports (`DecisionRunConfig`).
    made = {"mixture_weights": {name: weight for name, weight in sorted(shares.items()) if weight > 0},
            "mixture_made": dict(stated)}
    training = [dataclasses.replace(child, source=dataclasses.replace(child.source, mixture=made))
                for child in training]
    train = training[0] if len(training) == 1 else mix(training, kept)
    evaluated = None if not held else Pipeline(_DecisionSource(
        Concatenate(tuple(child.source for child in held if isinstance(child.source, Source))),
        _slots(stated) or None, descriptor=pinned)).map(ParseExample())
    return train, evaluated


def corpus(root_url: str, name: str, split: Literal["train", "held"] = "held",
           descriptor: str | None = None) -> Pipeline | None:
    """Corpus `name`'s `split` of the mixture directory `root_url` as decision loading reads it, from the
    shards and index its manifest names (or the corpus's one JSONL file), each row an `Example` under
    `example`; None when the corpus has no such split. A benchmark reads one corpus's held rows through
    this, whatever its weight."""
    stated, pinned = _stated(root_url, descriptor)
    return _corpus(root_url, stated, pinned, name, split)


def _stated(root_url: str, descriptor: str | None) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """The mixture's manifest, as `descriptor` pins it, and that pin."""
    from dew.data.storage import Store

    fs, path = Store(root_url).resolve()
    with fs.open(path.rstrip("/") + "/mixture.json", "rb") as handle:
        stated = record(json.load(handle), "mixture.json")
    return _descriptor(root_url, stated, descriptor)


def _corpus(root_url: str, stated: Mapping[str, object], pinned: Mapping[str, object],
            name: str, split: str) -> Pipeline | None:
    """Corpus `name`'s `split` from manifest `stated` (`corpus`)."""
    from dew.data.sources import Files
    from dew.data.storage import Store

    root = root_url.rstrip("/")
    corpus = record(record(stated.get("files", {}), "mixture files").get(name, {}), f"files.{name}")
    entry = record(corpus.get(split, {}), f"files.{name}.{split}")
    shards = entry.get("shards")
    if isinstance(shards, list):
        listed = [record(shard, "shard") for shard in shards]
        declared: dict[str, Mapping[str, object]] | None = {
            root + "/" + _records.text(shard["path"], "shard.path"): shard for shard in listed
        }
        paths = tuple(declared)
    else:
        declared = None
        url = root + f"/{name}{'.held' if split == 'held' else ''}.jsonl"
        shard_fs, shard_path = Store(url).resolve()
        paths = (url,) if shard_fs.exists(shard_path) else ()
    if not paths:
        return None
    index = record(entry.get("index", {}), "index").get("path")
    source_index = None if index is None else root + "/" + _records.text(index, "index.path")
    if declared is not None and source_index is not None:
        declared[source_index] = record(entry["index"], "index")
    dataset = _DecisionSource(Files(paths, format="jsonl", index=source_index, file_info=declared),
                              _slots(stated) or None, descriptor=pinned)
    return Pipeline(dataset, name=name).map(ParseExample())


def _descriptor(root_url: str, stated: Mapping[str, object],
                descriptor: str | None) -> tuple[Mapping[str, object], Mapping[str, object]]:
    from dew.data.storage import Store

    root = Store(root_url)
    pinned: dict[str, JSON] = {}
    if descriptor is not None:
        payload = Store(descriptor).read()
        replaced = record(json.loads(payload), "decision descriptor")
        provenance = record(replaced.get("provenance"), "descriptor provenance")
        root_manifest = root.child("manifest.sha256").read().decode().strip()
        if provenance.get("original_manifest_sha") != root_manifest:
            raise ValueError("decision descriptor refers to a different canonical source manifest")
        original_files = record(stated.get("files", {}), "original files")
        for name, value in record(replaced.get("files", {}), "descriptor files").items():
            old = record(original_files.get(name), f"original files.{name}")
            for partition, entry in record(value, name).items():
                if record(entry, partition).get("sha256") != record(old.get(partition), partition).get(
                    "sha256"
                ):
                    raise ValueError(f"decision descriptor changes raw corpus {name}/{partition}")
        stated = replaced
        pinned = {"url": descriptor, "sha256": hashlib.sha256(payload).hexdigest()}
    return stated, pinned


@dataclass(frozen=True)
class _ParseConfigured:
    table: ParseTable

    def __call__(self, metadata: Mapping[str, object]) -> ParseTable | ParseExample:
        names = metadata.get("decision_options")
        if names is None:
            if "slots" in metadata:
                return ParseExample()
            raise ValueError("decision flat table needs options, ClassLabel metadata or read='memory'")
        return dataclasses.replace(self.table, options=_records.strings(names, "decision_options"),
                                   by_value=metadata.get("decision_options_by_value") is True)
