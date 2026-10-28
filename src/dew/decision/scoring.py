"""Proper scoring rules: the losses a decision model trains on and the scores it is judged by.

A scoring rule charges a forecast distribution p for the outcome that
happened. A proper rule charges least, in expectation, when p is the true
distribution, so training on it rewards honest probabilities over
overconfident ones. Each rule here serves both roles: called on logits inside
a step it is a per-row loss, and as a `Metric` it averages the same charge
over a validation pass's `Decisions`. Rules can be added and scaled, as in
`LogLoss() + 0.5 * Brier()`. Each rule's charge is the categorical's (`dew.nn.categorical`), which any
objective's criterion composes on.

The charge is against a target distribution t over the row's options,
one-hot or soft, smoothed in training, and is zero for a certain, correct
forecast of a certain target. A held-out pass charges each question's gold
distribution, as its training does.
"""

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from dew.artifacts import Artifact, Decisions
from dew.batch import Batch
from dew.decision.metrics import Answered
from dew.nn import categorical
from dew.objectives.base import Metric, Shown, mean_of_totals, merge_totals


class ScoringRule(ABC):
    """A proper scoring rule, lower is better."""

    name: str
    reads = Decisions
    shown = Shown(better="lower")

    @abstractmethod
    def charge(self, logits: jax.Array, target: jax.Array, options: jax.Array,
               ordinal: jax.Array) -> jax.Array:
        """Return the `[B]` charge of each row.

        The inputs are the `[B, K]` logits (any value past a row's options), the
        `[B, K]` target distributions, the `[B, K]` real options, and the `[B]` rows
        whose options are ordered levels.
        """

    def __add__(self, other: "ScoringRule") -> "Combined":
        return Combined(((1.0, self),)) + other

    def __mul__(self, weight: float) -> "Combined":
        return Combined(((float(weight), self),))

    __rmul__ = __mul__

    def __call__(self, artifact: Artifact, batch: Batch, /) -> tuple[float, float]:
        answered = Answered.of(artifact)
        logits = np.log(np.clip(answered.probabilities, 1e-300, None))
        charged = np.asarray(self.charge(jnp.asarray(logits), jnp.asarray(answered.targets),
                                         jnp.asarray(answered.options), jnp.asarray(answered.ordinal)),
                             np.float64)
        return float(charged.sum()), float(charged.size)

    merge = staticmethod(merge_totals)
    finalize = staticmethod(mean_of_totals)


@dataclass(frozen=True)
class LogLoss(ScoringRule):
    """The cross entropy, -sum_k t_k log p_k: the logarithmic score."""

    name = "log_loss"

    def charge(self, logits, target, options, ordinal):
        return categorical.soft_cross_entropy(logits, target, options)


@dataclass(frozen=True)
class Brier(ScoringRule):
    """The quadratic score, sum_k (p_k - t_k)^2, between 0 and 2."""

    name = "brier"

    def charge(self, logits, target, options, ordinal):
        return categorical.brier(logits, target, options)


@dataclass(frozen=True)
class Spherical(ScoringRule):
    """The spherical score as a charge, 1 - (t . p) / |p|, between 0 and 1."""

    name = "spherical"

    def charge(self, logits, target, options, ordinal):
        return categorical.spherical(logits, target, options)


@dataclass(frozen=True)
class RankedProbability(ScoringRule):
    """The ranked probability score of ordered levels.

    It sums the squared gaps between the forecast's and the target's cumulative
    distributions over the K - 1 cuts between K levels, sum_k (P_k - T_k)^2 /
    (K - 1). It reads only a score question's levels and charges any other row
    nothing, since its options have no order. As a metric it averages over the
    scored rows.
    """

    name = "rps"

    def charge(self, logits, target, options, ordinal):
        return jnp.where(ordinal, categorical.ranked_probability(logits, target, options), 0.0)

    def __call__(self, artifact: Artifact, batch: Batch, /) -> tuple[float, float]:
        total, _ = super().__call__(artifact, batch)
        return total, float(np.sum(Answered.of(artifact).ordinal))


@dataclass(frozen=True)
class Combined(ScoringRule):
    """A weighted sum of rules, itself proper when every weight is positive."""

    terms: tuple[tuple[float, ScoringRule], ...]

    def __post_init__(self):
        if any(weight <= 0 for weight, _ in self.terms):
            raise ValueError("a sum of proper scoring rules stays proper only with positive weights")
        object.__setattr__(self, "name", "+".join(rule.name if weight == 1 else f"{weight:g}*{rule.name}"
                                                  for weight, rule in self.terms))

    def charge(self, logits, target, options, ordinal):
        charges = [weight * rule.charge(logits, target, options, ordinal) for weight, rule in self.terms]
        return functools.reduce(jnp.add, charges)

    def __add__(self, other: ScoringRule) -> "Combined":
        added = other.terms if isinstance(other, Combined) else ((1.0, other),)
        return Combined(self.terms + added)

    def __mul__(self, weight: float) -> "Combined":
        return Combined(tuple((float(weight) * held, rule) for held, rule in self.terms))

    __rmul__ = __mul__


def default_metrics() -> list[Metric]:
    """The held-out suite a decision run scores by default, fresh: accuracy, calibration error,
    the risk-coverage area and the log loss, in that order."""
    from dew.decision.metrics import AURC, ECE, Accuracy

    return [Accuracy(), ECE(), AURC(), LogLoss()]
