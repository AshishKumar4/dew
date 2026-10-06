"""The three questions a decision model answers, and its answers to them.

The questions are TypeSafe Jev's (docs.typesafe.ai/api): a yes-or-no `Noul`, a
`Choice` among named options, or a `Score` over ordered levels. Each names its
options in a fixed order, and the model returns one probability per option in
that order. An answer holds that distribution and computes the reported
fields from it: a Noul's probability of yes, a Choice's most likely option, a
Score's expected level. Its `confidence` is a statistic of the distribution,
chosen by a `Confidence` value.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from dew.records import JSON, json_value, record, strings


class Question(ABC):
    """One question about a state, with its options in the order the model scores them."""

    kind: ClassVar[str]
    """Jev's name for the question type: noul, choice or score."""
    instructions: JSON

    @property
    @abstractmethod
    def options(self) -> tuple[str, ...]:
        """Return the answer keys, one per option, in the order the model scores them."""

    @property
    @abstractmethod
    def descriptions(self) -> tuple[JSON, ...]:
        """Return what each option means, as the request gave it, with None where it gave nothing."""

    @abstractmethod
    def answer(self, probabilities: np.ndarray, confidence: "Confidence") -> "Answer":
        """Return this question's answer for `probabilities`, one per option."""

    @abstractmethod
    def wire(self) -> dict[str, JSON]:
        """Return the question in a Jev request's format."""

    @staticmethod
    def from_wire(request: Mapping[str, object]) -> "Question":
        """Build the question from a Jev request's type, instructions and criteria."""
        kind = request.get("type")
        unknown = set(request) - {"type", "instructions", "criteria"}
        if unknown:
            raise ValueError(f"a question has a type, instructions and criteria, not {sorted(unknown)}")
        instructions = json_value(request.get("instructions"), "instructions")
        criteria = request.get("criteria")
        if kind == "noul":
            return Noul(instructions, {} if criteria is None else {
                key: json_value(value, f"criteria.{key}")
                for key, value in record(criteria, "criteria").items()})
        if kind == "choice":
            if isinstance(criteria, (list, tuple)):
                return Choice(instructions, strings(criteria, "criteria"))
            return Choice(instructions, {key: json_value(value, f"criteria.{key}")
                                         for key, value in record(criteria, "criteria").items()})
        if kind == "score":
            if not isinstance(criteria, (list, tuple)):
                raise ValueError("a score's criteria are a list of level descriptions, lowest first")
            return Score(instructions, [json_value(level, "criteria") for level in criteria])
        raise ValueError(f"a question's type is noul, choice or score, got {kind!r}")

    def index(self, label: object) -> int:
        """Return the position of the option that `label` names, given as its key or as an int index."""
        if isinstance(label, bool) or not isinstance(label, (int, str)):
            raise ValueError(f"an answer names an option by its key or its index, got {label!r}")
        if isinstance(label, int):
            if not 0 <= label < len(self.options):
                raise ValueError(f"option {label} is past this question's {len(self.options)} options")
            return label
        if label not in self.options:
            raise ValueError(f"{label!r} is none of this question's options {self.options}")
        return self.options.index(label)


@dataclass(frozen=True)
class Noul(Question):
    """A yes-or-no question. `criteria` may say what `true` and `false` mean.

    Its options are `false` then `true`, so an answer's second probability is
    the probability of yes.
    """

    instructions: JSON
    criteria: Mapping[str, JSON] = field(default_factory=dict)
    kind: ClassVar[str] = "noul"

    def __post_init__(self):
        _require_instructions(self.instructions)
        unknown = set(self.criteria) - {"true", "false"}
        if unknown:
            raise ValueError(f"a noul's criteria describe only true and false, got {sorted(unknown)}")
        object.__setattr__(self, "criteria", dict(self.criteria))

    @property
    def options(self) -> tuple[str, ...]:
        return ("false", "true")

    @property
    def descriptions(self) -> tuple[JSON, ...]:
        return (self.criteria.get("false"), self.criteria.get("true"))

    def answer(self, probabilities: np.ndarray, confidence: "Confidence") -> "NoulAnswer":
        return NoulAnswer(_distribution(probabilities, 2), confidence(self, probabilities))

    def wire(self) -> dict[str, JSON]:
        request: dict[str, JSON] = {"type": self.kind, "instructions": self.instructions}
        if self.criteria:
            request["criteria"] = dict(self.criteria)
        return request


@dataclass(frozen=True)
class Choice(Question):
    """A choice among named options, 1 to 255 of them.

    `criteria` maps each option's key to what it means, or to None where the
    key says enough; a list of keys describes none of them.
    """

    instructions: JSON
    criteria: Mapping[str, JSON] | Sequence[str]
    kind: ClassVar[str] = "choice"

    def __post_init__(self):
        _require_instructions(self.instructions)
        criteria = self.criteria
        if isinstance(criteria, str):
            raise ValueError("a choice's criteria are a mapping of option to description, "
                             "or a list of options")
        if isinstance(criteria, Mapping):
            pairs = tuple(criteria.items())
        else:
            pairs = tuple((key, None) for key in criteria)
            if len({key for key, _ in pairs}) != len(pairs):
                raise ValueError("a choice's options must be distinct")
        if any(not isinstance(key, str) for key, _ in pairs):
            raise ValueError("a choice's options are named by strings, which its answer reports")
        if not 1 <= len(pairs) <= 255:
            raise ValueError(f"a choice takes 1 to 255 options, got {len(pairs)}")
        object.__setattr__(self, "criteria", dict(pairs))

    @property
    def options(self) -> tuple[str, ...]:
        return tuple(self.criteria)

    @property
    def descriptions(self) -> tuple[JSON, ...]:
        assert isinstance(self.criteria, Mapping)
        return tuple(self.criteria.values())

    def answer(self, probabilities: np.ndarray, confidence: "Confidence") -> "ChoiceAnswer":
        return ChoiceAnswer(self.options, _distribution(probabilities, len(self.options)),
                            confidence(self, probabilities))

    def wire(self) -> dict[str, JSON]:
        return {"type": self.kind, "instructions": self.instructions,
                "criteria": dict(zip(self.options, self.descriptions, strict=True))}


@dataclass(frozen=True)
class Score(Question):
    """A rating on ordered levels, 2 to 10 of them, lowest first.

    `criteria` describes each level; the answer's score is the expected
    level, so it can land between two.
    """

    instructions: JSON
    criteria: Sequence[JSON]
    kind: ClassVar[str] = "score"

    def __post_init__(self):
        _require_instructions(self.instructions)
        if isinstance(self.criteria, (str, Mapping)):
            raise ValueError("a score's criteria are a list of level descriptions, lowest first")
        levels = tuple(self.criteria)
        if not 2 <= len(levels) <= 10:
            raise ValueError(f"a score takes 2 to 10 levels, got {len(levels)}")
        if any(level is None for level in levels):
            raise ValueError("every score level needs a description")
        object.__setattr__(self, "criteria", levels)

    @property
    def options(self) -> tuple[str, ...]:
        return tuple(str(level) for level in range(len(self.criteria)))

    @property
    def descriptions(self) -> tuple[JSON, ...]:
        return tuple(self.criteria)

    def answer(self, probabilities: np.ndarray, confidence: "Confidence") -> "ScoreAnswer":
        return ScoreAnswer(self.descriptions, _distribution(probabilities, len(self.options)),
                           confidence(self, probabilities))

    def wire(self) -> dict[str, JSON]:
        return {"type": self.kind, "instructions": self.instructions, "criteria": list(self.descriptions)}


@dataclass(frozen=True)
class NoulAnswer:
    """The answer to a Noul: P(false) and P(true), and a confidence computed from them."""

    probabilities: np.ndarray
    confidence: float
    abstained: bool = False
    """Whether a calibration's abstention threshold gated the answer out."""
    kind: ClassVar[str] = "noul"

    @property
    def noul(self) -> float:
        """The probability that the answer is yes."""
        return float(self.probabilities[1])


@dataclass(frozen=True)
class ChoiceAnswer:
    """One probability per option, and the most likely option."""

    options: tuple[str, ...]
    probabilities: np.ndarray
    confidence: float
    abstained: bool = False
    kind: ClassVar[str] = "choice"

    @property
    def choice(self) -> str:
        """The most likely option; the first of several equally likely ones."""
        return self.options[int(np.argmax(self.probabilities))]


@dataclass(frozen=True)
class ScoreAnswer:
    """One probability per level, lowest first, and the expected level."""

    legend: tuple[JSON, ...]
    probabilities: np.ndarray
    confidence: float
    abstained: bool = False
    kind: ClassVar[str] = "score"

    @property
    def score(self) -> float:
        """The expected level, which can land between two."""
        return float(np.dot(np.arange(len(self.probabilities)), self.probabilities))


type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer

KINDS: tuple[type[Question], ...] = (Choice, Score, Noul)
"""The question types in the order a laid-out row numbers them, Laya's
(laya/common.py QTYPES: choice 0, score 1, noul 2). A head whose type
embedding orders them otherwise maps these numbers to its own rows."""


def kind_of(question: Question) -> int:
    """Return the number a laid-out row gives `question`'s type (`KINDS`)."""
    return KINDS.index(type(question))


class Confidence(ABC):
    """A statistic of an answer's distribution, between 0 and 1."""

    @abstractmethod
    def __call__(self, question: Question, probabilities: np.ndarray) -> float: ...


class JevConfidence(Confidence):
    """TypeSafe Jev's confidence (docs.typesafe.ai/confidence).

    For a Choice of n options, it is how far the top probability sits above an
    even split, (p_max - 1/n) / (1 - 1/n). For a Score, it weighs each level's
    distance from the most likely level m, 1 - sum_i p_i |i - m| / MAD, where MAD
    is the same average distance for an even spread measured from the middle
    level, floored at 0. Jev reports none for a Noul; this gives the Choice
    formula over its two options, |2 p - 1|, as Jev's page suggests. A choice
    with one option is certain.
    """

    def __call__(self, question: Question, probabilities: np.ndarray) -> float:
        p = np.asarray(probabilities, np.float64)
        n = len(p)
        if isinstance(question, Score):
            peak = int(np.argmax(p))
            levels = np.arange(n)
            spread = float(np.dot(p, np.abs(levels - peak)))
            even = float(np.mean(np.abs(levels - (n - 1) / 2)))
            return min(1.0, max(0.0, 1.0 - spread / even))
        if n == 1:
            return 1.0
        return min(1.0, max(0.0, (n * float(p.max()) - 1.0) / (n - 1)))


class EntropyConfidence(Confidence):
    """Laya's confidence: one minus the distribution's entropy over log n
    (laya/common.py, `confidence_from_probs`), for every type but a Noul,
    which Laya gives its top probability."""

    def __call__(self, question: Question, probabilities: np.ndarray) -> float:
        p = np.asarray(probabilities, np.float64)
        if isinstance(question, Noul):
            return float(p.max())
        if len(p) < 2:
            return 1.0
        entropy = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
        return min(1.0, max(0.0, 1.0 - entropy / math.log(len(p))))


class TopProbability(Confidence):
    """The answer's own probability, the largest one.

    This is what a temperature fit calibrates and what expected calibration error
    measures.
    """

    def __call__(self, question: Question, probabilities: np.ndarray) -> float:
        return float(np.max(probabilities))


def _distribution(probabilities: np.ndarray, count: int) -> np.ndarray:
    p = np.asarray(probabilities, np.float64)
    if p.shape != (count,):
        raise ValueError(f"the question has {count} options, the distribution {p.shape}")
    return p


def _require_instructions(instructions: JSON) -> None:
    # A mapping pattern matches every mapping that holds its keys, so `{}`
    # would match them all; an empty one is named by its length.
    match instructions:
        case None | "" | []:
            raise ValueError("a question needs instructions: the text, or the data, the model answers about")
        case dict() if not instructions:
            raise ValueError("a question needs instructions: the text, or the data, the model answers about")
