"""The learning gate: whether training does what a learning claim says, over independent runs
(`assert_learns`). A trained model's claim ("the samples take both modes at their shares", "the loss
halves") is about what training does on average, which no float64 run computes and one run measures
only with its own luck: a run that met a hand margin says little of the next, and a run that missed it
says little of the trainer. This gate tests the claim over runs.

Each run is an independent unit (its own initialization, data and noise) and states each claim as a
contrast, positive where the run meets the claim: for an improvement, the paired `before - 2 * after` of
one run; for a value held within its old margin of a target, the two one-sided contrasts `within` makes,
the margin less the run's departure on each side, never the margin less its absolute departure, whose
long left tail would pass a trainer at the margin far more often than LEVEL. Over a site's declared
runs, each claim's mean contrast is held above zero by the one-sided t test: its t, the mean over its
standard error, at least the t law's quantile at 1 - LEVEL on its runs' degrees of freedom. A trainer
whose runs sit exactly at a claim's margin passes that claim with chance LEVEL, where the old one-run
check passed it half the time; a trainer that meets its claims with room passes as its runs' spread
allows, which the site's runs are declared for (tests/test_learning_gate.py measures both)."""

import math

import numpy as np
from scipy import stats

LEVEL = 1e-4
"""The chance a trainer whose runs meet a claim exactly at its old margin, on average, passes it: one
verdict in ten thousand, the sampling gate's false-red rate on the other side."""

FEWEST_RUNS = 3
"""The least runs a site declares: with two, the t law's quantile at LEVEL is 3183 (71 with three), so
no spread is small enough to judge by."""


class LearningFailure(AssertionError):
    """A claim the gate finds training does not meet over its runs."""


def within(name: str, value, target, margin) -> dict[str, float]:
    """A run's two contrasts for `value` held within `margin` of `target`, one a side: `margin` less its
    departure above, and less its departure below (two one-sided tests, each linear in the value)."""
    departure = float(value) - float(target)
    return {f"{name} below its upper margin": margin - departure,
            f"{name} above its lower margin": margin + departure}


def t_statistics(contrasts: np.ndarray) -> np.ndarray:
    """Each claim's one-sided t over its runs (the first axis): the mean contrast over its standard
    error; a claim whose runs agree exactly has none, and its t is infinite of the mean's sign (zero
    where the mean is)."""
    contrasts = np.asarray(contrasts, np.float64)
    mean, spread = contrasts.mean(axis=0), contrasts.std(axis=0, ddof=1)
    error = spread / math.sqrt(contrasts.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(error > 0, mean / error, np.sign(mean) * np.inf)


def threshold(runs: int) -> float:
    """The t a claim's runs must reach: the t law's quantile at 1 - LEVEL on runs - 1 degrees of freedom."""
    return float(stats.t.isf(LEVEL, runs - 1))


def assert_learns(contrasts_of, label: str, *, runs: int) -> None:
    """`contrasts_of(run)`, one independent run's contrasts by claim (a mapping of each claim's name to
    its old margin less the run's miss, positive where the run meets it), over `runs` runs: each claim's
    t (`t_statistics`) at least `threshold(runs)`.

    Raises LearningFailure naming each claim that falls short, with its mean contrast, its runs' spread
    and its t; refuses a site of fewer than FEWEST_RUNS runs, runs that name other claims, and a contrast
    that is not finite."""
    if runs < FEWEST_RUNS:
        raise ValueError(f"{label}: {runs} runs, fewer than {FEWEST_RUNS}; declare more")
    first = dict(contrasts_of(0))
    claims = sorted(first)
    measured = [first] + [dict(contrasts_of(run)) for run in range(1, runs)]
    for run, contrasts in enumerate(measured):
        if sorted(contrasts) != claims:
            raise ValueError(f"{label}: run {run} states claims {sorted(contrasts)}, not {claims}")
    table = np.asarray([[float(contrasts[claim]) for claim in claims] for contrasts in measured])
    if not np.all(np.isfinite(table)):
        raise ValueError(f"{label}: contrasts must be finite, not {table.tolist()}")
    t, needed = t_statistics(table), threshold(runs)
    short = [index for index, value in enumerate(t) if not value >= needed]
    if short:
        missed = "; ".join(f"{claims[index]}: mean contrast {table[:, index].mean():.4g}, spread "
                           f"{table[:, index].std(ddof=1):.3g}, t {t[index]:.3g}" for index in short)
        raise LearningFailure(f"{label}: over {runs} runs, {missed}; the gate needs t at least {needed:.3g}, "
                              f"which a trainer at the claim's margin reaches with chance {LEVEL}")
