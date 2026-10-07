"""Metrics of a decision model's answers over a validation pass.

Each reads the `Decisions` artifact `DecisionObjective.evaluate` returns: the
model's probabilities over every question's options and the right option,
counting the questions with a known answer. The proper scoring rules
(`LogLoss`, `Brier`, `Spherical`, `RankedProbability`) are metrics too.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from dew.artifacts import Artifact, Decisions
from dew.eval.common import metric_device
from dew.objectives.base import Batch, Objective, Shown, mean_of_totals, merge_totals


@dataclass(frozen=True)
class Answered:
    """The questions of a `Decisions` with a known answer, one row each: their
    probabilities `[M, K]`, real options `[M, K]`, right options `[M]` and
    whether `[M]` they ask a score."""

    probabilities: np.ndarray
    options: np.ndarray
    labels: np.ndarray
    ordinal: np.ndarray

    @classmethod
    def of(cls, artifact: Artifact) -> "Answered":
        if not isinstance(artifact, Decisions):
            raise TypeError(f"a decision metric reads Decisions, not {type(artifact).__name__}")
        decisions = artifact
        scored = np.asarray(decisions.scored, bool)
        return cls(np.asarray(decisions.probabilities, np.float64)[scored],
                   np.asarray(decisions.options, bool)[scored], np.asarray(decisions.labels)[scored],
                   np.asarray(decisions.ordinal, bool)[scored])


def _tops(artifact: Artifact) -> tuple[np.ndarray, np.ndarray]:
    """Each answered question's top probability, and whether its top option is the right one."""
    answered = Answered.of(artifact)
    probabilities = np.where(answered.options, answered.probabilities, -1.0)
    return probabilities.max(axis=-1), probabilities.argmax(axis=-1) == answered.labels


@dataclass(frozen=True)
class Accuracy:
    """The share of rows whose most likely option is the right one."""

    name = "accuracy"
    reads = Decisions
    shown = Shown(better="higher")

    def __call__(self, artifact: Artifact, batch: Batch, /) -> tuple[float, float]:
        _, correct = _tops(artifact)
        with metric_device():
            share = Objective.accuracy(jnp.asarray(correct, jnp.float32), {})
        return float(share.total), float(share.mass)

    def merge(self, accumulated: tuple[float, float],
              contribution: tuple[float, float]) -> tuple[float, float]:
        return merge_totals(accumulated, contribution)

    def finalize(self, accumulated: tuple[float, float]) -> float:
        return mean_of_totals(accumulated)


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
        tops, correct = _tops(artifact)
        edges = np.linspace(0.0, 1.0, self.bins + 1)
        index = np.clip(np.searchsorted(edges, tops, side="left") - 1, 0, self.bins - 1)
        return np.stack([np.bincount(index, minlength=self.bins).astype(np.float64),
                         np.bincount(index, tops, self.bins), np.bincount(index, correct, self.bins)])

    def merge(self, accumulated: np.ndarray, contribution: np.ndarray) -> np.ndarray:
        return accumulated + contribution

    def finalize(self, accumulated: np.ndarray) -> float:
        rows, confidence, correct = accumulated
        filled = rows > 0
        return float(np.sum(np.abs(confidence[filled] - correct[filled])) / rows.sum())


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
        tops, correct = accumulated
        order = np.argsort(-tops, kind="mergesort")
        tops, correct = tops[order], correct[order]
        last = np.flatnonzero(np.concatenate((tops[1:] != tops[:-1], [True])))
        accepted = last + 1.0
        risk = 1.0 - np.cumsum(correct)[last] / accepted
        return float(np.sum(risk * np.diff(np.concatenate(([0.0], accepted)))) / len(tops))
