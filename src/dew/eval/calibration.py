"""Calibration of a classifier's forecasts over K options, and the scores of it: the math any model's
calibration reads.

A forecast is a row of logits, or of probabilities, over its real options. Each function reads
held-out forecasts as arrays, so a model keeps its own policy of which forecasts it calibrates together
(`dew.decision.calibration`'s buckets of a question type and its option count):

- `fit_temperature` is the temperature that minimizes the held-out log loss.
- `histogram_map` maps a top probability to the held-out accuracy of its bin (`histogram_mapped`).
- `abstention_threshold` is the lowest top probability whose accepted forecasts err at most a target.
- `reliability_bins` counts forecasts, confidence and right answers per bin of the top probability,
  additive over passes, and `expected_calibration_error` reads them.
- `risk_coverage_area` is the area under the risk-coverage curve, over `tied_prefixes`.

The two binnings are their references': a histogram map's bin is `floor(top * bins)`, as a binning
calibrator's (Laya's `fit_binning_map`), and a reliability bin is closed on the right, as the expected
calibration error's (Guo et al. 2017, Laya's `ece_score`), so a top probability on an edge falls in the
upper bin of one and the lower of the other.
"""

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """The options' probabilities of `logits` at `temperature`, over the last axis, in float64; an option
    of logit -inf takes no mass."""
    scaled = np.asarray(logits, np.float64) / temperature
    shifted = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    return shifted / np.sum(shifted, axis=-1, keepdims=True)


def fit_temperature(logits: np.ndarray, targets: np.ndarray, real: np.ndarray) -> float:
    """The temperature minimizing mean log loss of `logits` `[N, K]` against the gold distributions
    `targets` `[N, K]` over the `real` options `[N, K]`, by SciPy's bounded L-BFGS-B on its inverse, at
    least 1e-3, in which the loss is convex: its derivative is the mean over forecasts of
    E_p[z] - E_gold[z]. Each gold row is read as a distribution, to unit mass: a batch packs it in
    float32, a little short of unit mass, and the shortfall would scale with the logits' common
    offset."""
    from scipy import optimize, special

    real = np.asarray(real, bool)
    z = np.where(real, np.asarray(logits, np.float64), 0.0)
    gold = np.where(real, np.asarray(targets, np.float64), 0.0)
    gold = gold / np.sum(gold, axis=1, keepdims=True)
    labelled = np.sum(gold * z, axis=1)

    def loss(inverse: np.ndarray) -> tuple[float, np.ndarray]:
        scaled = np.where(real, z * inverse[0], -np.inf)
        slope = np.mean(np.sum(special.softmax(scaled, axis=1) * z, axis=1) - labelled)
        return float(np.mean(special.logsumexp(scaled, axis=1) - inverse[0] * labelled)), np.asarray([slope])

    fitted = optimize.minimize(loss, x0=np.ones(1), jac=True, method="L-BFGS-B", bounds=[(1e-3, None)],
                               options={"gtol": 1e-12, "ftol": 1e-15})
    return 1.0 / float(fitted.x[0])


def histogram_map(tops: np.ndarray, correct: np.ndarray, bins: int) -> tuple[float, ...]:
    """The held-out accuracy of each of `bins` equal bins of the top probability on [0, 1], a forecast in
    the bin `floor(top * bins)` (the last holds 1); an empty bin keeps its midpoint. A top probability
    maps through it as `histogram_mapped` reads it."""
    index = np.clip((np.asarray(tops) * bins).astype(int), 0, bins - 1)
    correct = np.asarray(correct, bool)
    return tuple(float(correct[index == b].mean()) if np.any(index == b) else (b + 0.5) / bins
                 for b in range(bins))


def histogram_mapped(values: tuple[float, ...], top: float) -> float:
    """`top` mapped through a histogram's bins (`histogram_map`)."""
    return values[min(len(values) - 1, max(0, int(top * len(values))))]


def tied_prefixes(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each distinct `confidence`, highest first, with how many forecasts are at or above it and how many
    of those are `correct`: a group of tied forecasts counts as a whole, in any order. Abstention's
    thresholds and the risk-coverage area read their accepted forecasts off these."""
    order = np.argsort(-confidence, kind="mergesort")
    confidence, correct = confidence[order], correct[order]
    last = np.flatnonzero(np.concatenate((confidence[1:] != confidence[:-1], [True])))
    return confidence[last], last + 1, np.cumsum(correct)[last]


def abstention_threshold(tops: np.ndarray, correct: np.ndarray, *, target_error: float,
                         conservative: bool = True) -> float:
    """The smallest top probability at which the forecasts at or above it err at most `target_error`,
    a group of tied forecasts counted as a whole (`tied_prefixes`); with `conservative`, the error is
    estimated as (errors + 1) / (accepted + 1). 1.0 where no threshold meets the target."""
    levels, accepted, right = tied_prefixes(np.asarray(tops), np.asarray(correct, bool))
    errors = accepted - right
    rate = (errors + 1) / (accepted + 1) if conservative else errors / accepted
    met = rate <= target_error
    return float(levels[met].min()) if met.any() else 1.0


def reliability_bins(tops: np.ndarray, correct: np.ndarray, bins: int) -> np.ndarray:
    """Per bin of `bins` equal bins on (0, 1], closed on the right, the forecast count, the summed top
    probability `tops` and the number `correct`, `[3, bins]`: additive, so passes merge by sum."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.searchsorted(edges, tops, side="left") - 1, 0, bins - 1)
    return np.stack([np.bincount(index, minlength=bins).astype(np.float64),
                     np.bincount(index, tops, bins), np.bincount(index, correct, bins)])


def expected_calibration_error(binned: np.ndarray) -> float:
    """The gap between confidence and accuracy, per bin weighted by its forecasts (`reliability_bins`)."""
    rows, confidence, correct = binned
    filled = rows > 0
    return float(np.sum(np.abs(confidence[filled] - correct[filled])) / rows.sum())


def risk_coverage_area(tops: np.ndarray, correct: np.ndarray) -> float:
    """The area under the risk-coverage curve: the error rate among the forecasts at or above each
    confidence, weighted by how many forecasts that confidence adds, so a group of equal confidences
    counts as a whole in any order. Lower is better; it rewards a confidence that ranks right answers
    above wrong ones."""
    _, accepted, right = tied_prefixes(np.asarray(tops), np.asarray(correct, bool))
    risk = 1.0 - right / accepted
    return float(np.sum(risk * np.diff(np.concatenate(([0], accepted)))) / len(tops))


__all__ = ["abstention_threshold", "expected_calibration_error", "fit_temperature", "histogram_map",
           "histogram_mapped", "reliability_bins", "risk_coverage_area", "softmax", "tied_prefixes"]
