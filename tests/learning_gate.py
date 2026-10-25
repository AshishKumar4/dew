"""The learning gate: whether training does what a learning claim says, over independent units
(`assert_learns`). A trained model's claim ("the samples take both modes at their shares", "training
lowers the loss") is about what training does on average, which no float64 run computes and one run
measures only with its own luck: a run that met a hand margin says little of the next, and a run that
missed it says little of the trainer. This gate tests the claim over units.

A unit is an independent training run (its own initialization, data and noise) or a trained model's
independent draw (a noise level and noise it is scored at). Each states each claim as a contrast,
positive where the unit meets the claim: for an improvement, the paired `before - 2 * after` of one
unit; for a value held within its old margin of a target, the two one-sided contrasts `within` makes,
the margin less the unit's departure on each side, never the margin less its absolute departure, whose
long left tail passes a claim at its margin more often than the t law says. Over a site's declared
units, each claim's mean contrast is held above zero by the one-sided t test: its t, the mean over its
standard error, at least the t law's quantile at 1 - LEVEL on its units' degrees of freedom
(`threshold`). A trainer whose units meet a claim exactly at its old margin on average passes it with
chance LEVEL, where the old one-run check passed it half the time. A trainer that meets its claims with
room passes as its units' spread allows: `units_for` sizes a site from a pilot of its units so a
trainer as good as the pilot's fails at most FALSE_RED of its verdicts."""

import math
from collections.abc import Mapping

import numpy as np
from scipy import stats

LEVEL = 1e-9
"""The chance a trainer whose units meet a claim exactly at its old margin, on average, passes it: one
verdict in a billion, the rate check 1's conversions state for a law."""

FALSE_RED = 1e-4
"""The chance a trainer as good as its site's pilot fails a verdict, summed over its claims, at the
units `units_for` declares: the sampling gate's false-red rate."""

FEWEST_UNITS = 8
"""The least units a site declares: the t law's quantile at LEVEL is 57 at seven units and 39 at eight,
so far in its tail that fewer units judge a trainer by how normal its contrasts are there."""


class LearningFailure(AssertionError):
    """A claim the gate finds training does not meet over its units."""


def within(name: str, value, target, margin) -> dict[str, np.ndarray]:
    """The contrasts for `value` (one a unit) held within `margin` of `target`, one a side: `margin` less
    the departure above, and less the departure below (two one-sided tests, each linear in the value)."""
    departure = np.asarray(value, np.float64) - target
    return {f"{name} below its upper margin": margin - departure,
            f"{name} above its lower margin": margin + departure}


def t_statistics(contrasts: np.ndarray) -> np.ndarray:
    """Each claim's one-sided t over its units (the first axis): the mean contrast over its standard
    error; a claim whose units agree exactly has none, and its t is infinite of the mean's sign (zero
    where the mean is)."""
    contrasts = np.asarray(contrasts, np.float64)
    mean, spread = contrasts.mean(axis=0), contrasts.std(axis=0, ddof=1)
    error = spread / math.sqrt(contrasts.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(error > 0, mean / error, np.sign(mean) * np.inf)


def threshold(units: int) -> float:
    """The t a claim's units must reach: the t law's quantile at 1 - LEVEL on units - 1 degrees of freedom."""
    return float(stats.t.isf(LEVEL, units - 1))


def _table(contrasts: Mapping[str, np.ndarray], label: str) -> tuple[list[str], np.ndarray]:
    """`contrasts` by claim as its claims' names and a table, a row a unit and a column a claim."""
    claims = sorted(contrasts)
    table = np.stack([np.asarray(contrasts[claim], np.float64) for claim in claims], axis=-1)
    if table.ndim != 2 or not np.all(np.isfinite(table)):
        raise ValueError(f"{label}: contrasts must be finite, one a unit, not {table.tolist()}")
    return claims, table


def assert_learns(contrasts_of, label: str, *, units: int) -> None:
    """`contrasts_of(numbers)`, the contrasts of the independent units numbered `numbers` (an integer
    array) by claim: a mapping of each claim's name to its contrasts, one a unit, each the old margin
    less the unit's miss, positive where the unit meets it. Over `units` units numbered from zero, each
    claim's t (`t_statistics`) at least `threshold(units)`. A site whose units are training runs trains
    them together, one program for all of them.

    Raises LearningFailure naming each claim that falls short, with its mean contrast, its units'
    spread and its t; refuses a site of fewer than FEWEST_UNITS units, contrasts of another count, and a
    contrast that is not finite."""
    if units < FEWEST_UNITS:
        raise ValueError(f"{label}: {units} units, fewer than {FEWEST_UNITS}; declare more")
    claims, table = _table(contrasts_of(np.arange(units)), label)
    if table.shape[0] != units:
        raise ValueError(f"{label}: {table.shape[0]} contrasts a claim for {units} units")
    t, needed = t_statistics(table), threshold(units)
    short = [index for index, value in enumerate(t) if not value >= needed]
    if short:
        missed = "; ".join(f"{claims[index]}: mean contrast {table[:, index].mean():.4g}, spread "
                           f"{table[:, index].std(ddof=1):.3g}, t {t[index]:.3g}" for index in short)
        raise LearningFailure(f"{label}: over {units} units, {missed}; the gate needs t at least "
                              f"{needed:.3g}, which a trainer at the claim's margin reaches with chance "
                              f"{LEVEL}")


def false_red(effects: np.ndarray, units: int) -> float:
    """The chance a verdict on `units` units fails a trainer whose claims' contrasts have `effects`,
    each claim's mean over its spread, summed over its claims: each claim's t is the noncentral t law's
    on units - 1 degrees of freedom, its noncentrality the effect times the root of `units`."""
    effects = np.asarray(effects, np.float64)
    needed = threshold(units)
    return float(np.sum(stats.nct.cdf(needed, units - 1, np.sqrt(units) * effects)))


def units_for(contrasts: Mapping[str, np.ndarray], label: str, *, most: int = 1024) -> int:
    """The least units, from FEWEST_UNITS, at which a trainer whose claims' contrasts have a pilot's
    effects fails a verdict with chance at most FALSE_RED (`false_red`). The pilot is `contrasts` by
    claim, one a unit, many more units than the answer (its effects' own error, about one over the
    root of its units, moves the answer as the square of the effect does). Raises past `most`, a
    trainer whose pilot does not meet its claims with room any site could judge."""
    claims, table = _table(contrasts, label)
    spread = table.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        effects = np.where(spread > 0, table.mean(axis=0) / spread, np.sign(table.mean(axis=0)) * np.inf)
    weakest = claims[int(np.argmin(effects))]
    if not np.all(effects > 0):
        raise AssertionError(f"{label}: the pilot misses {weakest} on average; nothing to size")
    units = FEWEST_UNITS
    while false_red(effects[np.isfinite(effects)], units) > FALSE_RED:
        units += 1
        if units > most:
            raise AssertionError(f"{label}: {weakest}'s effect {effects.min():.3g} needs more than {most} "
                                 f"units")
    return units
