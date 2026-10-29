"""Metrics of a decision model's answers over a validation pass.

Each reads the `Decisions` artifact `DecisionObjective.evaluate` returns: the
model's probabilities over every question's options and the right option,
counting the questions with a known answer. The proper scoring rules
(`LogLoss`, `Brier`, `Spherical`, `RankedProbability`) are metrics too.
"""

from dataclasses import dataclass

import numpy as np

from dew.artifacts import Artifact, Decisions
from dew.batch import Batch
from dew.eval.calibration import expected_calibration_error, reliability_bins, risk_coverage_area
from dew.objectives.base import Shown, mean_of_totals, merge_totals


@dataclass(frozen=True)
class Answered:
    """The questions of a `Decisions` with a known answer, one row each: their
    probabilities `[M, K]`, real options `[M, K]`, right options `[M]`, gold
    distributions `[M, K]` and whether `[M]` they ask a score."""

    probabilities: np.ndarray
    options: np.ndarray
    labels: np.ndarray
    targets: np.ndarray
    ordinal: np.ndarray

    @classmethod
    def of(cls, artifact: Artifact) -> "Answered":
        if not isinstance(artifact, Decisions):
            raise TypeError(f"a decision metric reads Decisions, not {type(artifact).__name__}")
        decisions = artifact
        scored = np.asarray(decisions.scored, bool)
        return cls(np.asarray(decisions.probabilities, np.float64)[scored],
                   np.asarray(decisions.options, bool)[scored], np.asarray(decisions.labels)[scored],
                   np.asarray(decisions.targets, np.float64)[scored],
                   np.asarray(decisions.ordinal, bool)[scored])


def _tops(artifact: Artifact) -> tuple[np.ndarray, np.ndarray]:
    """Each answered question's top probability, and whether its top option is the right one."""
    answered = Answered.of(artifact)
    probabilities = np.where(answered.options, answered.probabilities, -1.0)
    return probabilities.max(axis=-1), probabilities.argmax(axis=-1) == answered.labels


@dataclass(frozen=True)
class Accuracy:
    """The share of answered questions whose most likely option is the right one."""

    name = "accuracy"
    reads = Decisions
    shown = Shown(better="higher")

    def __call__(self, artifact: Artifact, batch: Batch, /) -> tuple[float, float]:
        _, correct = _tops(artifact)
        return float(np.sum(correct)), float(correct.size)

    merge = staticmethod(merge_totals)
    finalize = staticmethod(mean_of_totals)


@dataclass(frozen=True)
class MacroF1:
    """The mean over classes of each class's F1, the classes those the split's answered questions are
    or are answered as (sklearn's `f1_score(average="macro")`): a class is a question's option
    (`Decisions.classes`), the same in every batch and split that asks it, so each class's true and
    false positives and false negatives add over the whole pass before its F1 is taken."""

    name = "macro_f1"
    reads = Decisions
    shown = Shown(better="higher")

    def __call__(self, artifact: Artifact, batch: Batch, /) -> dict[int, np.ndarray]:
        """Return each class's true positives, false positives and false negatives in the batch."""
        if not isinstance(artifact, Decisions) or artifact.classes is None:
            raise TypeError("macro F1 reads Decisions that carry each option's class (a run's rows)")
        scored = np.asarray(artifact.scored, bool)
        lanes = np.asarray(artifact.classes, np.int64)[scored]
        classes = (lanes[..., 0] << 32) | (lanes[..., 1] & 0xFFFFFFFF)
        probabilities = np.where(np.asarray(artifact.options, bool)[scored],
                                 np.asarray(artifact.probabilities, np.float64)[scored], -1.0)
        rows = np.arange(len(classes))
        predicted = classes[rows, probabilities.argmax(axis=-1)]
        gold = classes[rows, np.asarray(artifact.labels)[scored]]
        counts: dict[int, np.ndarray] = {}
        for guess, truth in zip(predicted.tolist(), gold.tolist(), strict=True):
            if guess == truth:
                counts.setdefault(truth, np.zeros(3, np.int64))[0] += 1
            else:
                counts.setdefault(guess, np.zeros(3, np.int64))[1] += 1
                counts.setdefault(truth, np.zeros(3, np.int64))[2] += 1
        return counts

    def merge(self, accumulated: dict[int, np.ndarray],
              contribution: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        merged = dict(accumulated)
        for key, counts in contribution.items():
            merged[key] = merged[key] + counts if key in merged else counts
        return merged

    def finalize(self, accumulated: dict[int, np.ndarray]) -> float:
        scores = [2 * true / (2 * true + false + missed) for true, false, missed in accumulated.values()]
        return float(np.mean(scores)) if scores else float("nan")


@dataclass(frozen=True)
class ScoredQuestions:
    """How many questions with a known answer the split scored, the count every other metric of it
    is taken over."""

    name = "scored_questions"
    reads = Decisions
    shown = Shown()

    def __call__(self, artifact: Artifact, batch: Batch, /) -> int:
        if not isinstance(artifact, Decisions):
            raise TypeError(f"a decision metric reads Decisions, not {type(artifact).__name__}")
        return int(np.sum(np.asarray(artifact.scored, bool)))

    def merge(self, accumulated: int, contribution: int) -> int:
        return accumulated + contribution

    def finalize(self, accumulated: int) -> float:
        return float(accumulated)


@dataclass(frozen=True)
class ECE:
    """Expected calibration error: the gap between the top probability and the accuracy.

    It is taken over `bins` equal bins on (0, 1], closed on the right, and
    weighted by how many rows each bin holds (Laya's `ece_score`, 15 bins).
    """

    bins: int = 15
    name = "ece"
    reads = Decisions
    shown = Shown(better="lower")

    def __call__(self, artifact: Artifact, batch: Batch, /) -> np.ndarray:
        """Return, per bin, the row count, the summed top probability and the number correct."""
        return self.binned(*_tops(artifact))

    def binned(self, tops: np.ndarray, correct: np.ndarray) -> np.ndarray:
        """Per bin, the row count, the summed top probability `tops` and the
        number `correct`, for answers however they were drawn: `finalize`
        reads these, from a run's validation pass or a served benchmark's (`reliability_bins`)."""
        return reliability_bins(tops, correct, self.bins)

    def merge(self, accumulated: np.ndarray, contribution: np.ndarray) -> np.ndarray:
        return accumulated + contribution

    def finalize(self, accumulated: np.ndarray) -> float:
        return expected_calibration_error(accumulated)


@dataclass(frozen=True)
class AURC:
    """The area under the risk-coverage curve.

    It is the error rate among the rows at or above each confidence, weighted by
    how many rows that confidence adds, so a group of equal confidences counts as
    a whole in any row order (Laya's `aurc`, laya/evals.py). Lower is better; it
    rewards a confidence that ranks right answers above wrong ones.
    """

    name = "aurc"
    reads = Decisions
    shown = Shown(better="lower")

    def __call__(self, artifact: Artifact, batch: Batch, /) -> tuple[np.ndarray, np.ndarray]:
        return _tops(artifact)

    def merge(self, accumulated: tuple[np.ndarray, np.ndarray],
              contribution: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return (np.concatenate([accumulated[0], contribution[0]]),
                np.concatenate([accumulated[1], contribution[1]]))

    def finalize(self, accumulated: tuple[np.ndarray, np.ndarray]) -> float:
        return risk_coverage_area(*accumulated)
