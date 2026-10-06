"""Post-hoc calibration of a decision model: temperatures, binning and abstention.

All three are Laya's (laya/calibrate.py and laya/common.py at
NandhaKishorM/laya a4a8921), fitted on held-out answers and kept per
bucket: a question type and its option count, as `bucket` names it.

- `Temperatures` divides each question's logits by its bucket's
  temperature, or its type's where the bucket has too few answers.
- `Binning` maps the calibrated top probability to the accuracy of the
  held-out answers in its histogram bin.
- `Abstention` sets each bucket's lowest confidence an answer needs, where
  the held-out answers at or above it err at most a target rate.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from dew.decision.questions import Question


def bucket(question: Question) -> str:
    """Return the calibration bucket of `question`: its type and its option count.

    The counts are grouped as 2, 3-5, 6-10 or 11+ (Laya's `temp_bucket`).
    """
    return _bucket(question.kind, len(question.options))


def _bucket(kind: str, count: int) -> str:
    size = "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
    return f"{kind}:{size}"


@dataclass(frozen=True)
class Scored:
    """One held-out answer, with its question's type and the model's raw logits.

    It holds the question's type (`Question.kind`), the model's raw logits over
    its options, and the index of the right option.
    """

    kind: str
    logits: np.ndarray
    label: int

    @property
    def bucket(self) -> str:
        return _bucket(self.kind, len(self.logits))


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, np.float64) / temperature
    shifted = np.exp(scaled - scaled.max())
    return shifted / shifted.sum()


@dataclass(frozen=True)
class Temperatures:
    """The temperature each question's logits are divided by.

    The temperatures are held within [`lowest`, `highest`], as Laya's agent holds
    a checkpoint's: it refuses to sharpen past 0.5 (its `clamp_temperature`).
    """

    types: Mapping[str, float] = field(default_factory=dict)
    buckets: Mapping[str, float] = field(default_factory=dict)
    lowest: float = 0.5
    highest: float = 5.0

    def of(self, kind: str, count: int) -> float:
        """Return the temperature for a question of type `kind` (`Question.kind`) with `count` options.

        That is its bucket's temperature, or else its type's, or else 1.
        """
        return self._held(self.buckets.get(_bucket(kind, count), self.types.get(kind, 1.0)))

    def applied(self) -> "Temperatures":
        """Return these temperatures as `of` applies them, each held within [`lowest`, `highest`].

        A reader that does not hold them itself, as llama.cpp's server does not,
        then answers as this does.
        """
        return replace(self, types={kind: self._held(value) for kind, value in self.types.items()},
                       buckets={name: self._held(value) for name, value in self.buckets.items()})

    def _held(self, value: float) -> float:
        return min(self.highest, max(self.lowest, value))

    @classmethod
    def fit(cls, scored: Sequence[Scored], *, type_minimum: int = 10, bucket_minimum: int = 2000,
            lowest: float = 0.5, highest: float = 5.0) -> "Temperatures":
        """Return the temperatures that minimize held-out log loss.

        There is one per question type with at least `type_minimum` answers, and one
        per bucket with at least `bucket_minimum`; these are Laya's minimums
        (laya/calibrate.py MIN_TYPE_N and MIN_BUCKET_N).
        """
        def fitted(group: list[Scored], minimum: int) -> float | None:
            return None if len(group) < minimum else _temperature(group)

        types = {kind: fitted([held for held in scored if held.kind == kind], type_minimum)
                 for kind in sorted({held.kind for held in scored})}
        buckets = {name: fitted([held for held in scored if held.bucket == name], bucket_minimum)
                   for name in sorted({held.bucket for held in scored})}
        return cls({name: value for name, value in types.items() if value is not None},
                   {name: value for name, value in buckets.items() if value is not None}, lowest, highest)


def _temperature(group: Sequence[Scored]) -> float:
    """The temperature minimizing mean log loss over `group`, by Newton's
    method on its inverse, in which the loss is convex: its first derivative
    is the mean over answers of E_p[z] - z_label and its second the mean of
    Var_p[z]. Rows of fewer options pad with logits that take no mass."""
    width = max(len(held.logits) for held in group)
    z = np.zeros((len(group), width))
    real = np.zeros((len(group), width), bool)
    for row, held in enumerate(group):
        z[row, :len(held.logits)] = held.logits
        real[row, :len(held.logits)] = True
    labelled = z[np.arange(len(group)), [held.label for held in group]]
    inverse = 1.0
    for _ in range(100):
        scaled = np.where(real, z * inverse, -np.inf)
        p = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        mean = np.sum(p * z, axis=1)
        slope = float(np.mean(mean - labelled))
        curvature = float(np.mean(np.sum(p * (z - mean[:, None]) ** 2, axis=1)))
        step = slope / max(curvature, 1e-12)
        inverse = max(inverse - step, 1e-3)
        if abs(step) < 1e-10 * inverse:
            break
    return 1.0 / inverse


@dataclass(frozen=True)
class Binning:
    """Histogram binning of the calibrated top probability, per bucket.

    `bins` equal bins on [0, 1] each map to the held-out accuracy of the answers
    that fell in them (Laya's `fit_binning_map`). A bucket without a map keeps
    the probability.
    """

    maps: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def apply(self, kind: str, count: int, top: float) -> float:
        values = self.maps.get(_bucket(kind, count))
        if values is None:
            return top
        return values[min(len(values) - 1, max(0, int(top * len(values))))]

    @classmethod
    def fit(cls, scored: Sequence[Scored], temperatures: Temperatures, *, bins: int = 15,
            bucket_minimum: int = 200) -> "Binning":
        """Return a map for every bucket with at least `bucket_minimum` answers.

        An empty bin keeps its midpoint.
        """
        maps = {}
        for name in sorted({held.bucket for held in scored}):
            group = [held for held in scored if held.bucket == name]
            if len(group) < bucket_minimum:
                continue
            tops, correct = _tops(group, temperatures)
            index = np.clip((tops * bins).astype(int), 0, bins - 1)
            maps[name] = tuple(float(correct[index == b].mean()) if np.any(index == b) else (b + 0.5) / bins
                               for b in range(bins))
        return cls(maps)


@dataclass(frozen=True)
class Abstention:
    """The lowest calibrated top probability an answer needs, per bucket, or else `default`.

    An answer below it abstains. None gates nothing.
    """

    thresholds: Mapping[str, float] = field(default_factory=dict)
    default: float | None = None

    def threshold(self, kind: str, count: int) -> float | None:
        return self.thresholds.get(_bucket(kind, count), self.default)

    @classmethod
    def fit(cls, scored: Sequence[Scored], temperatures: Temperatures, binning: Binning | None = None, *,
            target_error: float = 0.10, bucket_minimum: int = 100, conservative: bool = True) -> "Abstention":
        """Return the smallest threshold that meets `target_error`, per bucket.

        Only buckets with at least `bucket_minimum` answers get one. A threshold
        meets the target when the accepted answers err at most `target_error`,
        counting a group of tied answers as a whole; with `conservative`, the error
        is estimated as (errors + 1) / (accepted + 1). A bucket that no threshold
        satisfies gets 1.0 (Laya's `fit_abstention_thresholds`).
        """
        thresholds = {}
        for name in sorted({held.bucket for held in scored}):
            group = [held for held in scored if held.bucket == name]
            if len(group) < bucket_minimum:
                continue
            tops, correct = _tops(group, temperatures, binning)
            confidence = np.round(tops, 4)
            chosen = 1.0
            for level in sorted(set(confidence.tolist()), reverse=True):
                accepted = confidence >= level
                errors = float(np.sum(~correct[accepted]))
                count = float(np.sum(accepted))
                rate = (errors + 1) / (count + 1) if conservative else errors / count
                if rate <= target_error:
                    chosen = level
            thresholds[name] = chosen
        return cls(thresholds)


def _tops(group: Sequence[Scored], temperatures: Temperatures,
          binning: Binning | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each answer's calibrated (and binned) top probability, and whether its top option is right."""
    tops, correct = [], []
    for held in group:
        p = softmax(held.logits, temperatures.of(held.kind, len(held.logits)))
        top = float(p.max())
        tops.append(top if binning is None else binning.apply(held.kind, len(held.logits), top))
        correct.append(int(np.argmax(p)) == held.label)
    return np.asarray(tops), np.asarray(correct, bool)


@dataclass(frozen=True)
class Calibration:
    """How a decision model's raw logits become reported answers."""

    temperatures: Temperatures = Temperatures()
    binning: Binning | None = None
    abstention: Abstention | None = None

    def probabilities(self, question: Question, logits: np.ndarray) -> np.ndarray:
        return softmax(logits, self.temperatures.of(question.kind, len(question.options)))

    def abstains(self, question: Question, probabilities: np.ndarray) -> bool:
        """Return whether the answer's calibrated top probability is below its threshold."""
        kind, count = question.kind, len(question.options)
        threshold = None if self.abstention is None else self.abstention.threshold(kind, count)
        if threshold is None:
            return False
        top = float(np.max(probabilities))
        return (top if self.binning is None else self.binning.apply(kind, count, top)) < threshold

