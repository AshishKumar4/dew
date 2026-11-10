"""Check 1, Dew's accuracy contract (docs/design/testing.md): a run is on par with its reference where
its error from the float64 truth is the reference's within PARITY, judged over drawn seeds. This module is
its one implementation, the suite's (tests/reference_error.py's `assert_as_exact`, which compiles and draws
the arms) and the library's (dew.interop.verify's load-time check): its factors and tails, its findings, the
pooled ratio of Dew's error to the reference's and its interval, a leaf's status, and the tests of a site
that stacks examples (its examples' paired ratios, a scalar leaf's known null, each example's worst row
against the reference's on the same example). Each a function of the errors a site's draws recorded, apart
from the arms and the draws that record them.
"""

import dataclasses
import functools
import hashlib
import itertools
import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dew._accuracy import distance
from dew.parity_stages import staged_units

FACTOR = 2.0

ROW_FACTOR = 4.0
"""How far Dew's worst row may sit from float64 against the reference's worst
row (`rows_over`; check 1's at rows of one or two values is NARROW_FACTORS').
An RMS over every entry averages a few wrong rows into a
large tensor, so each comparison also holds Dew's worst row to the
reference's. The worst of many rows is an extreme value, not an average,
and parts further by chance: over the suite's 3,660 comparisons whose rows
differ from float64 at all (99 files), Dew's worst row over the
reference's sat at a median of 0.21, p99 1.55 and at most 2.42 (Mixtral's
exported logits). A row that differs rather than rounds moves further: the
suite's deliberate failures, a GDN chunk's plain fp32 prefix and a MiMo tower
with its sinks zeroed, put their worst rows at 7.1 and 3,419 times the
reference's."""


PARITY = 1.5
"""The gate's on-par factor: each leaf's RMS error from float64 within 1.5 times the
reference's, on the same inputs, with no floor of any kind. Check 1 (`assert_as_exact`) holds it now;
the one-draw rule's FACTOR joins it once every site that rule would then fail has moved to check 1."""

BOOTSTRAP = 2000
"""Resamples of a site's seeds for the pooled ratio's interval (`_interval`), drawn by one fixed key so
an assertion decides alike on every run."""
CONFIDENCE = 0.999
"""Check 1's stated confidence in its ratio's decisions: a pass certifies a leaf's RMS ratio no more
than PARITY², and an over by its ratio that it is above one, each but for a nominal 5e-4 (LOWER_TAIL,
UPPER_TAIL). A leaf failed by a row, a row of float64 zero or a value not finite is failed by that
rule, which certifies no ratio."""
LOWER_TAIL = 5e-4
"""The lower tail of a leaf's bootstrap interval, the over side's whole 5e-4, spent at stage 2 alone, where
an over is decided (`_staged`). Nominal: the rates hold as far as the percentile bootstrap's endpoint
coverage does, and 2,000 resamples interpolate an endpoint this far out."""
UPPER_TAIL = 2.5e-4
"""The upper tail at each of check 1's two stages: the pass side's 5e-4 spent equally over them, a pass
possible at either (nominal, as LOWER_TAIL)."""
FEWEST_SEEDS = 8
"""The fewest seeds check 1 judges by, an engineering minimum and no statistical threshold: over very few
seeds a bootstrap has so few distinct resamples that its interval can sit wholly to one side of PARITY by
the draws' luck, so fewer is Inconclusive whatever their ratio reads; above it the measured interval
decides. `seeds_for` never declares fewer."""


@dataclasses.dataclass(frozen=True)
class Finding:
    """One finding of the gate as data, beside its message: the leaf it names (the site's label and the
    leaf's path), the rule it fails (RULES), and its measure, finite and larger worse: the log of Dew's
    error over the reference's for a ratio's rule (`log_ratio`, `_log_sum`), the seeds a value is not
    finite on, Dew's own distance from float64 where the reference is exactly float64, and for too few
    seeds the fewest over the given. What a train matches a candidate's failures to its parent's by:
    the same leaf and rule, a measure no larger."""
    leaf: str
    rule: str
    measure: float
    message: str

    def __str__(self) -> str:
        return self.message


RULES = frozenset({"pooled", "row", "zero rows", "not finite", "exact reference", "no measure",
                   "inconclusive"})
"""The rules a finding names: the pooled RMS ratio over its factor, the worst row over its factor, a row
of float64 zero further from it than the reference's, a value not finite, a reference exactly float64
where Dew is not (measured by Dew's own distance from it), a site where both are (nothing to measure,
measured 0), and a leaf its seeds leave undecided."""


LOG_ZERO = -1075 * math.log(2)
"""What a finding's measure reads as the log of an error of exactly zero (`log_ratio`): below the log of
the least positive float64, 2^-1074, so a reference error of zero orders under every positive one and
a Dew error of zero under every positive one. An order among findings, never a bound any verdict
reads."""


def _logs(values) -> np.ndarray:
    """Each nonnegative value's natural log, exact for every positive float64, subnormals included, and
    LOG_ZERO for zero."""
    values = np.asarray(values, np.float64)
    with np.errstate(divide="ignore"):
        return np.where(values > 0, np.log(np.where(values > 0, values, 1.0)), LOG_ZERO)


LOG_LARGEST = math.log(np.finfo(np.float64).max)
"""The log of float64's largest value: a quotient of finite evidence past it is held there where it is
recorded or shown (`_over`), never where it is judged (`log_over`)."""


def log_over(numerators, denominators) -> np.ndarray:
    """The log of each nonnegative numerator over its positive denominator (`_logs`), exact and finite for
    every finite pair, so a quotient past float64's range (an error of 1e154 over a row of 1e-160) keeps
    its order against every other; LOG_ZERO for a zero numerator, and not a number for one that is not."""
    numerators = np.asarray(numerators, np.float64)
    logs = np.where(numerators > 0, _logs(numerators) - _logs(denominators), LOG_ZERO)
    return np.where(np.isnan(numerators), np.nan, logs)


def _over(numerators, denominators) -> np.ndarray:
    """Each quotient of `log_over` as a value, held at float64's largest: a recorded reference's worst
    row (`RecordedLeaf.worst`) and a message's figure; zero over anything is zero."""
    quotient = np.exp(np.minimum(log_over(numerators, denominators), LOG_LARGEST))
    return np.where(np.asarray(numerators, np.float64) > 0, quotient, 0.0)


def _shown(log: float) -> str:
    """A log quotient as a message shows it, held at float64's largest."""
    return f"{float(np.exp(np.minimum(log, LOG_LARGEST))):.3e}"


def _scaled_rms(rows, xp=np):
    """Each row's root mean square of finite values, the row scaled by its largest first, so no square
    overflows (row 1628): zero for a row of zeros. In numpy, or in `xp` (jax.numpy, inside check 1's
    programs on a GPU, `reference_error._computed_here`)."""
    rows = xp.abs(xp.asarray(rows, xp.float64))
    largest = xp.max(rows, axis=1, keepdims=True) if rows.shape[1] else xp.zeros((len(rows), 1))
    scale = xp.where(largest > 0, largest, 1.0)
    return scale[:, 0] * xp.sqrt(xp.mean(xp.square(rows / scale), axis=1))


def _log_sum(values) -> float:
    """The natural log of the sum of nonnegative values, scaled by their largest first, so no sum of
    finite values overflows on the way: LOG_ZERO for a sum of zeros."""
    values = np.asarray(values, np.float64)
    largest = float(np.max(values)) if values.size else 0.0
    if not largest > 0:
        return LOG_ZERO
    return math.log(largest) + math.log(float(np.sum(values / largest)))


def _root_mean(values) -> float:
    """The root of the mean of nonnegative values (each seed's mean squared error), scaled by their
    largest first, so no sum of finite values overflows on the way: finite wherever they are."""
    values = np.asarray(values, np.float64)
    largest = float(np.max(values)) if values.size else 0.0
    if not largest > 0:
        return 0.0
    return math.sqrt(largest) * math.sqrt(float(np.mean(values / largest)))


def log_ratio(mine, theirs) -> float:
    """A ratio rule's measure: the natural log of Dew's error over the reference's, the largest over its
    elements, each log exact for every positive error (`_logs`). Finite at every finite error and
    growing with Dew's, where the ratio itself overflows past an error of 4 over a reference of zero."""
    return float(np.max(_logs(mine) - _logs(theirs)))


class GateFailure(AssertionError):
    """A site the gate fails, its findings as data (`Finding`), its message theirs, one a line, unless a
    rule words its findings together. A test's report carries them as the JUnit property
    `gate_findings` (tests/conftest.py)."""

    def __init__(self, findings, message: str | None = None):
        self.findings = tuple(findings)
        super().__init__("\n".join(map(str, self.findings)) if message is None else message)


class Inconclusive(GateFailure):
    """Check 1 could not decide a leaf: its seeds do not resolve PARITY. Never a pass; the site takes
    larger inputs or more seeds (`seeds_for`)."""


def _pooled(mine: np.ndarray, theirs: np.ndarray) -> float:
    """sqrt(sum a / sum b): the RMS ratio over every draw and element, each draw's mean squared error
    weighed alike, as the owner's estimand reads it."""
    return float(np.sqrt(np.sum(mine) / np.sum(theirs))) if np.sum(theirs) > 0 else (
        0.0 if np.sum(mine) == 0 else float("inf"))


def _interval(mine: np.ndarray, theirs: np.ndarray,
              tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL)) -> tuple[float, float]:
    """The pooled ratio's bootstrap interval over seeds, its lower and upper `tails` left out: Dew's and
    the reference's errors resampled by the same seeds."""
    picks = np.random.default_rng(0).integers(0, len(mine), (BOOTSTRAP, len(mine)))
    return _quantiles(np.sqrt(mine[picks].sum(axis=1) / np.maximum(theirs[picks].sum(axis=1), 1e-300)), tails)


def _control_interval(mine: np.ndarray, control: np.ndarray,
                      tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL)) -> tuple[float, float]:
    """`_interval` of the reference compared with itself, an equally exact pair: its errors on two disjoint
    seed sets, `mine` and `control`, resampled apart."""
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(mine), (BOOTSTRAP, len(mine)))
    other = rng.integers(0, len(control), (BOOTSTRAP, len(control)))
    ratios = np.sqrt(mine[picks].sum(axis=1) / np.maximum(control[other].sum(axis=1), 1e-300))
    return _quantiles(ratios, tails)


def _quantiles(ratios: np.ndarray, tails: tuple[float, float]) -> tuple[float, float]:
    """`ratios`' quantiles at its lower tail and one less its upper."""
    return float(np.quantile(ratios, tails[0])), float(np.quantile(ratios, 1 - tails[1]))


QUALIFYING_SPREAD = 0.40
"""The most a leaf's draws' log RMS ratios (`_log_ratios`) may spread, their standard deviation, for its
typical draw to pass it (`_typical_status`): at it and under it, check 1's tail test (`_tail_sign`) with the
spread itself refuses Dew three times the reference's RMS error on a fifth of the draws in at least 95% of
runs at 256 seeds and more, Dew as exact passing (check1-qualifying-sim.py,
the gate's own functions; ledger 2469). A fault on some draws only widens the spread, so a leaf whose fault
the tail test misses is held inconclusive by it."""
TAIL_SHARE = 0.1
"""The share of a leaf's draws, the farthest from their median, the tail test reads (`_tail_sign`)."""


def _log_ratios(mine: np.ndarray, theirs: np.ndarray) -> np.ndarray | None:
    """Each draw's log RMS ratio, half the log of Dew's squared error over the reference's, the draws where
    both are exactly float64 left out; None where one arm is exact on a draw and the other is not."""
    if np.any((mine > 0) != (theirs > 0)):
        return None
    kept = mine > 0
    return 0.5 * np.log(mine[kept] / theirs[kept])


FAULTS = ((2.0, 0.1), (3.0, 0.2), (10.0, 0.1))
"""The faults the tail test must refuse wherever it reads (`_tail_sign`), each Dew `factor` times the
reference's RMS error on `share` of a leaf's draws or examples: the smallest, twice off on a tenth, is the
reach check 1's fallback states (docs/design/testing.md); the larger move a tail's median along with it."""

TAIL_POWER = 0.95
"""The power the tail test keeps against every fault of FAULTS at a leaf's own noise and count."""

FLOOR_REPLICATES = 32
"""The faulted samples of a leaf's own noise its tail share is averaged over (`_tail_floor`)."""


def _tail_share(centred: np.ndarray) -> tuple[int, int]:
    """Of the values in the farthest TAIL_SHARE of `centred` from their median, the count above it and
    their count."""
    centred = centred - np.median(centred)
    tail = centred[np.abs(centred) >= np.quantile(np.abs(centred), 1 - TAIL_SHARE)]
    return int(np.sum(tail > 0)), len(tail)


def _tail_floor(ratios: np.ndarray, alpha: float) -> float:
    """The tail test's null share (`_tail_sign`), derived from its fault model at the leaf's own noise
    (Main, 2026-10-31): the leaf's log ratios about their median, each sign drawn anew so no asymmetry of
    their own counts, take each fault of FAULTS (log `factor` added on `share` of them) over
    FLOOR_REPLICATES draws; the share of the faulted tail above its median, q, is what that fault shows; the
    floor is the largest share from ½ at which a binomial test of the tail at `alpha` still refuses q with
    TAIL_POWER, the least over FAULTS, and ½ where none clears it at ½. Equal exactness puts the share at ½,
    so a floor above it costs no false refusal; it gives the test the effect size its statistic needs, where
    against ½ alone every heavy tail a little uneven is refused once a leaf holds enough draws."""
    from scipy import stats

    rng = np.random.default_rng([2469, len(ratios)])
    magnitudes = np.abs(np.asarray(ratios, np.float64) - np.median(ratios))
    floor = 1.0
    for factor, share in FAULTS:
        found = [_tail_share(magnitudes * rng.choice((-1.0, 1.0), len(magnitudes))
                             + np.where(rng.random(len(magnitudes)) < share, math.log(factor), 0.0))
                 for _ in range(FLOOR_REPLICATES)]
        q, count = float(np.mean([above / max(total, 1) for above, total in found])), found[0][1]
        shares = np.arange(0.5, q, 0.001)
        # The least count above the median the test refuses at each share, and a fault's chance to reach it.
        refused = stats.binom.isf(alpha, count, shares) + 1
        powered = shares[stats.binom.sf(refused - 1, count, q) >= TAIL_POWER]
        floor = min(floor, float(powered[-1]) if len(powered) else 0.5)
    return max(floor, 0.5)


def _tail_sign(ratios: np.ndarray, alpha: float = LOWER_TAIL) -> float:
    """The tail test's p-value: of the draws in the farthest TAIL_SHARE of their log ratios from the median,
    the count above it against Binomial(n, f), f the share above the median its fault model derives at the
    leaf's noise and count (`_tail_floor`, at `alpha`; ½ where it can afford none). Where the two arms are
    equally exact their roundings are exchangeable about the typical ratio, so a draw is as likely above it
    as below whatever their spread (distribution-free); Dew worse by a factor on some draws puts their tail
    above it, by at least what each fault of FAULTS shows there."""
    from scipy import stats

    above, total = _tail_share(np.asarray(ratios, np.float64))
    return float(stats.binom.sf(above - 1, total, _tail_floor(ratios, alpha)))


def _example_medians(examples: list[np.ndarray]) -> np.ndarray:
    """Each draw's statistic at a site that stacks examples (`assert_as_exact`'s `examples`): the median of
    its examples' paired log RMS ratios (`_example_ratios`). Both arms see the same example, heavy-tailed
    across examples as it is, so its ratio cancels most of that tail, and the median of a draw's examples
    narrows as their count grows where the draw's own ratio, its RMS ruled by its worst example, does not
    (ledger 2469: 0.11 to 0.28 at 32 examples where the draw's ratio spread 0.43 to 1.65). A draw whose
    examples both arms hold exactly is left out."""
    return np.asarray([float(np.median(ratios)) for ratios in examples if len(ratios)])


def _within_draws(examples: list[np.ndarray]) -> np.ndarray:
    """Every example's paired log ratio less its draw's median (`_example_ratios`): a fault on some of each
    draw's examples puts these past zero on its side, which the draws' medians alone do not show."""
    return np.concatenate([np.zeros(0), *(ratios - np.median(ratios) for ratios in examples if len(ratios))])


EXAMPLE_POWER = ((0.4, 1024), (0.6, 2048), (0.8, 8192), (0.9, 16384))
"""The examples a site's leaf needs, by its examples' spread about their draws' medians (`_within_draws`,
their standard deviation), for the examples' tail test (`_tail_sign`) to refuse Dew three times the
reference's RMS error on a fifth of each draw's examples in at least 95% of runs, measured with the gate's
own test (check1-examples-power.txt; ledger 2469): a leaf whose examples
spread wider than the last, or that holds fewer than its spread's count (interpolated in its log), is not
passed by its examples (`_typical_status`)."""


SCALAR_NULL_QUANTILE = 0.9
"""The share of a scalar leaf's examples (one value each) under the known null's quantile the scalar tail test
reads above (`_scalar_tail`)."""

SCALAR_EXAMPLES = 4096
"""The examples a scalar leaf's tail test needs: at it, with the gate's own `_staged`, Dew three times off on
a fifth of its examples passed by them in no run of 100, at 32 or 16 examples a draw, and equally exact arms
in every run at 32 (check1-scalar-sim.txt). A scalar leaf's draws' medians
spread about 0.35 at 32 examples a draw and 0.49 at 16, past QUALIFYING_SPREAD: it wants 32."""


def _scalar_null_quantile() -> float:
    """The known null's SCALAR_NULL_QUANTILE quantile: a scalar example's log RMS ratio under equal
    exactness, each arm's rounding of it one normal (the central limit over the terms it sums), so
    log|N1| - log|N2|, its spread pi/2 whatever their scale; drawn once, many millions."""
    rng = np.random.default_rng(0)
    null = np.log(np.abs(rng.standard_normal(4_000_000))) - np.log(np.abs(rng.standard_normal(4_000_000)))
    return float(np.quantile(null, SCALAR_NULL_QUANTILE))


def _scalar_tail(examples: list[np.ndarray]) -> float:
    """The scalar tail test's p-value: of a scalar leaf's examples' log ratios about their median over every
    draw (a draw's own median, of a few dozen, too noisy a centre against a fault on some of them), the count
    above the known null's quantile (`_scalar_null_quantile`) against Binomial(n, 1 - SCALAR_NULL_QUANTILE).
    The null is exact for a scalar, each arm's rounding a normal and its per-example ratio cancelling its
    scale: the sign test, distribution-free, cannot see a fault through its pi/2 spread within MOST_SEEDS,
    and this test, the null's own quantile, does."""
    from scipy import stats

    every = np.concatenate([np.zeros(0), *examples])
    above = int(np.sum(every - np.median(every) > _scalar_null_quantile()))
    return float(stats.binom.sf(above - 1, len(every), 1 - SCALAR_NULL_QUANTILE))


def _examples_needed(spread: float) -> float:
    """The examples a leaf's tail test needs at its examples' `spread` (`EXAMPLE_POWER`), infinite past
    the widest measured."""
    spreads, counts = zip(*EXAMPLE_POWER, strict=True)
    if spread > spreads[-1]:
        return float("inf")
    return float(np.exp(np.interp(spread, spreads, np.log(counts))))


def _geometric(ratios: np.ndarray, tails: tuple[float, float]) -> tuple[float, float, float]:
    """The geometric mean of log ratios `ratios` (one a draw) and its bootstrap interval over the draws, its
    lower and upper `tails` left out."""
    if not len(ratios):
        return float("nan"), float("nan"), float("nan")
    picks = np.random.default_rng(0).integers(0, len(ratios), (BOOTSTRAP, len(ratios)))
    means = ratios[picks].mean(axis=1)
    return (float(np.exp(ratios.mean())), float(np.exp(np.quantile(means, tails[0]))),
            float(np.exp(np.quantile(means, 1 - tails[1]))))


def _examples_refuse(examples: list[np.ndarray], scalar: bool, tail: float = LOWER_TAIL) -> str | None:
    """Why a leaf's examples do not pass it, or None: their ratios about their draws' medians
    (`_within_draws`) too few for the tail test at their spread (`EXAMPLE_POWER`; SCALAR_EXAMPLES for a
    `scalar` leaf, one value an example), or their tail above the medians (`_tail_sign`; `_scalar_tail`,
    the known null's, for a scalar leaf)."""
    within = _within_draws(examples)
    needed = SCALAR_EXAMPLES if scalar else _examples_needed(float(np.std(within)))
    if len(within) < needed:
        return (f"its {len(within)} examples spread {float(np.std(within)):.2f} about their draws' medians, "
                f"where the tail test needs {needed:.0f}")
    if (found := _scalar_tail(examples) if scalar else _tail_sign(within, tail)) < tail:
        return f"its examples' tail lies above their median (p {found:.1e})"
    return None


def _typical_status(mine: np.ndarray, theirs: np.ndarray,
                    tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL),
                    examples: list[np.ndarray] | None = None,
                    scalar: bool = False) -> tuple[str, float, float, float, str]:
    """A leaf's verdict by its draws' paired ratios, asked only where its pooled ratio leaves it
    inconclusive: "pass" where the typical draw's ratio passes (the draws' geometric mean, its interval
    `_status`), the draws' log ratios spread no wider than QUALIFYING_SPREAD (where the tail test sees a
    fault on a fifth of the draws) and their tail is not above the median's (`_tail_sign`, at the lower
    tail), else "inconclusive", never "over"; with the reason it is not passed. At a site that stacks
    examples (`examples`, each seed's per-example log ratios) each draw's ratio is its examples' median
    (`_example_medians`), and the examples' own tail about their draws' medians is asked too
    (`_within_draws`), so a fault on some of each draw's examples is refused. A pass certifies the typical
    draw's (or example's) ratio under PARITY and no fault the tail tests see at that spread; a `scalar` leaf
    (one value an example) takes the known null's tail test (`_scalar_tail`)."""
    ratios = _log_ratios(mine, theirs) if examples is None else _example_medians(examples)
    if ratios is None or not len(ratios):
        return "inconclusive", float("nan"), float("nan"), float("nan"), "its draws hold no paired ratio"
    ratio, low, high = _geometric(ratios, tails)
    if _status(ratio, low, high) != "pass":
        return "inconclusive", ratio, low, high, "the typical draw's ratio does not pass"
    if (spread := float(np.std(ratios))) > QUALIFYING_SPREAD:
        return ("inconclusive", ratio, low, high,
                f"its draws' log ratios spread {spread:.2f}, past the {QUALIFYING_SPREAD} the tail test "
                f"sees at")
    if (tail := _tail_sign(ratios, tails[0])) < tails[0]:
        return "inconclusive", ratio, low, high, f"its draws' tail lies above their median (p {tail:.1e})"
    if examples is not None and (reason := _examples_refuse(examples, scalar, tails[0])) is not None:
        return "inconclusive", ratio, low, high, reason
    return "pass", ratio, low, high, ""


def _sized(mine: np.ndarray, theirs: np.ndarray, typical: bool = False,
           examples: list[np.ndarray] | None = None, scalar: bool = False) -> bool:
    """Whether a paired sizing takes a leaf as sized by these draws, Dew's squared errors `mine` against
    the reference's `theirs`: its pooled ratio's interval resolved (no wider than PARITY), or, with
    `typical` (a leaf the pooled ratio does not size within MOST_SEEDS, `seeds_for`), its typical draw's
    interval resolved and its draws qualifying as `_typical_status` asks (`examples` as it reads them)."""
    low, high = _interval(mine, theirs)
    if _resolves(low, high):
        return True
    if not typical:
        return False
    ratios = _log_ratios(mine, theirs) if examples is None else _example_medians(examples)
    if ratios is None or not len(ratios):
        return False
    _, low, high = _geometric(ratios, (LOWER_TAIL, UPPER_TAIL))
    return (_resolves(low, high) and float(np.std(ratios)) <= QUALIFYING_SPREAD
            and _tail_sign(ratios) >= LOWER_TAIL
            and (examples is None or _examples_refuse(examples, scalar) is None))


def _resolves(low: float, high: float) -> bool:
    """Whether a pooled ratio's interval is no wider than PARITY, so it tells an equally exact
    computation from one PARITY off."""
    return high <= PARITY * max(low, 1e-300)


def _status(ratio: float, low: float, high: float, *, contained: bool = True) -> str:
    """A leaf's verdict by its pooled ratio and interval: "pass", "over" or "inconclusive". The interval
    is taken to hold the ratio (low and high widened to it, which no endpoint's coverage loses by), so a
    pass certifies the ratio's high end at most PARITY², whether the interval lies wholly at or under
    PARITY or is resolved (no wider than PARITY) with the ratio under it, and an over its low end above
    one. Straddling PARITY and wider than it, the leaf is inconclusive, as is a ratio or end that is not
    a number. Not `contained`, the earlier rule: the interval as it is, a leaf over PARITY
    by its ratio wherever the interval does not straddle it unresolved."""
    if not contained:
        if low <= PARITY < high and not _resolves(low, high):
            return "inconclusive"
        return "over" if ratio > PARITY else "pass"
    if not all(np.isfinite(value) for value in (ratio, low, high)):
        return "inconclusive"
    low, high = min(low, ratio), max(high, ratio)
    if high <= PARITY:
        return "pass"
    if low > PARITY:
        return "over"
    if _resolves(low, high):
        return "pass" if ratio <= PARITY else "over"
    return "inconclusive"


ZeroRows = tuple[np.ndarray, np.ndarray, np.ndarray]
"""One seed's rows of float64 zero of a leaf that check 1 judges by the reference's error on them
(`_paired`): their indices, Dew's RMS errors on them and the reference's."""


def pooled_zero_rows(evidence: list[ZeroRows]) -> list[tuple[int, float, float]]:
    """The rows of float64 zero Dew is further from than ROW_FACTOR times the reference over a run's
    seeds, as (row, Dew's RMS error, the reference's): each row's squared errors summed over the seeds
    it is zero on, Dew's within ROW_FACTOR² of the reference's. A row of float64 zero is all residue,
    float32's rounding of a sum float64 cancels, with no size to judge it at, so it is held to the
    reference's residue on the same row; one seed's residues are heavy-tailed, and their ratio, a
    ratio of two, failed equally exact pairs seed by seed where the pooled one does not, while one
    row held ten times further still fails. Where the reference is exactly the float64 zero on a row
    over every seed, Dew must be too.

    Calibrated on attention's packed bias gradients, whose key rows are zero (`python
    tools/gate_null_runs.py attention-key-bias --paired --seeds K --runs N --chunk C`, and `--fault
    zero-row-10x`): 0 of 15,000 runs failed at 8 seeds (a red rate at most 4.6e-4 at 99.9%) and 0 of
    3,000 at 40 (at most 2.3e-3), and the 10x row failed every run at both. Judged seed by seed, the
    same pairs failed on 6.6% of runs of 8 seeds and 42% of 64."""
    mine: dict[int, list[float]] = {}
    theirs: dict[int, list[float]] = {}
    for rows, my_errors, their_errors in evidence:
        for row, my_error, their_error in zip(rows.tolist(), my_errors.tolist(), their_errors.tolist(),
                                              strict=True):
            mine.setdefault(row, []).append(my_error)
            theirs.setdefault(row, []).append(their_error)
    # Each row's errors pooled as a scaled root mean square and compared by their logs, so errors near
    # float64's largest pool and order without a square or a product overflowing.
    pooled = [(row, float(_scaled_rms([mine[row]])[0]), float(_scaled_rms([theirs[row]])[0]))
              for row in sorted(mine)]
    return [(row, my_rms, their_rms) for row, my_rms, their_rms in pooled
            if not _logs(my_rms) <= math.log(ROW_FACTOR) + _logs(their_rms)]


def _paired_rows(paired: np.ndarray, factor: float) -> float:
    """The paired worst-row test's p-value at a site that stacks examples: of the examples whose worst rows'
    log ratio (`_example_worst_rows`) lies past `factor` either way, the count with Dew's past `factor`
    times the reference's against Binomial(n, 1/2). Under equal exactness the two arms are exchangeable on
    each example, so its ratio is as likely past `factor` as under its inverse whatever their own spread
    (the known null, distribution-free): Dew whose worst rows sit past `factor` times the reference's on more
    examples than the reverse is refused, where a maximum over a leaf's every row against the reference's is
    an extreme of two heavy-tailed samples and no measure of either (ledger 2469)."""
    from scipy import stats

    above, below = int(np.sum(paired > math.log(factor))), int(np.sum(paired < -math.log(factor)))
    return float(stats.binom.sf(above - 1, above + below, 0.5)) if above else 1.0


NARROW_FACTORS = {1: 90.0, 2: 11.0}
"""The worst-row rule's factor at rows of one or two values, in place of ROW_FACTOR. At either width
ROW_FACTOR fails an equally exact Dew far more often than LOWER_TAIL as sites run check 1 (1.5e-2 of
runs at one value a row, 1.5e-3 at two): a row of few values is sized by a few float64 values, which sit
near zero far more often than a wide row's RMS does, and the worst row over a site's rows and seeds is
the row whose truth is smallest. Each factor sits above the 1 - 1e-4 quantile of the factor that would
fail equally exact pairs at both stages, both arms on one truth (`tools/gate_null_runs.py --rows`, its
`row_controls`, through `_staged` itself), at every layout measured, 16 to 4,096 rows a draw and 8 to 512
declared seeds: that quantile is 40 to 53 at one value and 7 to 8 at two (10,000 to 600,000 runs a
layout), so 90 and 11 stand about 1.7 and 1.4 times above it, at measured rates near 2e-5 and at most
3.3e-5; 28,000 fresh runs at the 16-row and 256-row layouts bound them within LOWER_TAIL at 99.9% (at
most 4.7e-4). Rows of three values keep ROW_FACTOR at a measured rate of at most 2.9e-4 over 16 to 2,048
rows, a point rate, bounded within LOWER_TAIL at 256 rows and 8 seeds (80,000 runs) and not at 2,048. The
rate is a leaf's: a site of L narrow leaves may fail an equally exact Dew L times as often.

The power this gives up is local: a narrow row whose error is between ROW_FACTOR and these factors
times the reference's worst no longer fails by its row (the pooled ratio and the leaf's other rows still
judge it), and a narrow false alarm no longer draws the second stage for the site's other leaves, which
that stage exists to judge on real findings. Check 2 (tests/certified.py `assert_certified`) holds
each element within its own operation's certified bound where a site runs it, which resolves such an
error at a pointwise step or a short sum: a sum of n terms is held within γ_n Σ|terms|, about n·u
Σ|terms| for u the unit roundoff, of its exact value, so its reach in units of the sum's own last place grows
with n and with the sum's cancellation (for standard normal terms, the smallest error it flagged in a
review's measurement, the median over 200 sums, was 37 units in the sum's last place at 16 terms, 918 at
128 and 22,966 at 1,024): past a few dozen terms of that kind an error in this window passes both checks
wherever the pooled ratio does not see it."""


RULED_SHARE = 0.9
"""The share of a worst row's squared error its largest element carries past which one value rules the row:
where both Dew's worst row and the reference's, over a stage's draws, are so ruled, a leaf of more than two
values a row judges its maximum at a row of one value's factor (NARROW_FACTORS[1]; `_Draws.factor`, row
3684). Such rows (an input amplified through 2^k, the weighed check's case) part as one value's error does:
at a share s each row's RMS error is within sqrt(1/s) of its largest element's over the row's width, so the
two rows' ratio is the two elements' within 1.054 either way at 0.9. Calibrated through the gate's own
verdict (`tools/gate_null_runs.py` `row_controls`, equally exact pairs, one element a row amplified by 2^k,
8 seeds, 32,000 runs a layout over 1 and 4 rows a draw, 3, 4 and 16 values a row, k apart in each arm or
shared; armada vq-g17-calib against vq-g17-calib-base): with k up to 24 (rows ruled far past the cutoff)
it fails at most 2 of 32,000 runs, where the width's factor alone failed up to 154 of 4,000; with k up to
4, which holds the shares about the cutoff, at most 14 of 32,000 (1 row of 3 values, k apart), where the
width's factor alone failed 185, and 0 to 3 at 16 values. Dew's worst row must be ruled too, so a row of
Dew's spread over its values (a row fault) keeps its width's factor beside a reference one value rules. A
Gaussian row of eight values or more is almost never ruled; rows of three or four values sometimes are,
where a fault ruled alike is refused less often. A value's own error in a ruled row is check 2's to bound.
A stacked-examples site's paired rows keep their width's factor (`_judgement`), and a recording's summary,
which keeps no row's elements, keeps it too."""


def row_factor(width: int) -> float:
    """The worst-row rule's factor for rows of `width` values: NARROW_FACTORS' at one or two, else ROW_FACTOR;
    every path of check 1 reads its rows' factor here (a weighed class's rows by the class's values a row)."""
    return NARROW_FACTORS.get(width, ROW_FACTOR)


PREVIOUS_TAILS = (5e-4, 5e-4)
"""The interval's tails before check 1's two stages (`_staged`'s `two_stage=False`)."""


def _unresolved_against_itself(errors: np.ndarray, names, count: int) -> list[str]:
    """The leaves the reference compared with itself does not resolve at `count` seeds: `errors`, its
    squared errors a seed (rows) and a leaf (columns), on seeds 0 to `count` against `count` to twice
    it (`_interval`'s control); a leaf with no error on either half is resolved."""
    halves = errors[:count], errors[count:2 * count]
    return [name for index, name in enumerate(names)
            if np.sum(halves[0][:, index]) > 0 and np.sum(halves[1][:, index]) > 0
            and not _resolves(*_control_interval(halves[0][:, index], halves[1][:, index]))]


def _verdict(over: list[Finding], unresolved: list[Finding]) -> str:
    """A judgement's verdict: "over" where any leaf failed, else "inconclusive" where any is unresolved,
    else "pass"."""
    return "over" if over else "inconclusive" if unresolved else "pass"


RESOLUTION = 1024 * float(np.finfo(np.float64).eps)
"""The smallest row float64 resolves, over its leaf's size: a row below it is
float64's own rounding of the leaf (a packed row's padding at 4e-19 of a
leaf at 1e-3), a zero `rows_over` sizes by the reference's row. 1024
roundings is the longest chain the suite's oracles allow
(`assert_computes_the_oracle`); a row at a float32 resolution is ten orders
above it."""


def row_errors(value, truth, terms=None) -> tuple[np.ndarray, np.ndarray]:
    """Each row's float64 size and `value`'s RMS distance from float64 on it,
    a row every axis but the last. A row's size is the RMS of its float64
    values, or, where `terms` gives each entry's float64 terms
    (`float64_terms`), the larger of that and the RMS of what the row sums,
    so a row that cancels is sized by its terms' rounding and not by its
    near-zero result; a row below float64's `RESOLUTION` of its leaf has size
    zero. No other row's size enters: a small row beside large ones has its
    own. A value of rank one or less is one row, which `distance` already
    judges, and has none here."""
    truth = np.asarray(truth, np.float64)
    if truth.ndim < 2:
        return np.zeros(0), np.zeros(0)
    return _row_sizes(truth, terms)[1], _rows_rms(np.asarray(value, np.float64) - truth, truth.shape[-1])


def _rows_rms(values, width: int) -> np.ndarray:
    """Each row's RMS of `values`, a row `width` wide (`_scaled_rms`)."""
    return _scaled_rms(np.reshape(values, (-1, width)))


def _row_sizes(truth: np.ndarray, terms=None) -> tuple[np.ndarray, np.ndarray]:
    """A truth leaf's rows, of rank two or more: each row's own RMS, and its size as `row_errors` reads
    it, the RMS (or, with `terms`, the larger of it and its terms') where it is past RESOLUTION of the
    leaf, else zero."""
    own = _rows_rms(truth, truth.shape[-1])
    sized = own if terms is None else np.maximum(own, _rows_rms(terms, truth.shape[-1]))
    leaf = max(distance(truth, 0), 0.0 if terms is None else distance(terms, 0))
    return own, np.where(sized > RESOLUTION * leaf, sized, 0.0)

NO_ZERO_ROWS: ZeroRows = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0))


RECORDED_KINDS = {
    "reference": "output", "truth": "output",
    "mse": "seed", "worst": "seed", "finite": "seed", "sketch": "seed", "example_mse": "seed",
    "example_worst": "seed", "scale": "seed",
    "zero_rows": "span", "zero_errors": "span", "zero_offsets": "offsets",
    "shape": "whole", "dtype": "whole",
}
"""A recording's entries kept one a leaf, `<site>.<kind>.<leaf>` (a pilot's `<site>.pilot.<kind>.<leaf>`),
each kind by how it is held: a full recording's outputs and truths over its seeds (`output`); a summarized
one's evidence, one row a seed (`seed`: a `RecordedLeaf`'s summary, its truth's sketch, a stacked site's
examples, a weighed site's rounding scales); its rows of float64 zero joined over the seeds (`span`), each
seed's span of them between consecutive `offsets`; and what the whole recording holds once a leaf
(`whole`). The one statement of the layout: the recorder writes and cuts by it
(tools/recorded_references.py) and the gate reads by it (tests/reference_error.py). Every other entry
(inputs, digests, counts, provenance) is the whole recording's."""


def recorded_kinds(*held: str) -> tuple[str, ...]:
    """The kinds of a recording's leaf entries held as one of `held` (`RECORDED_KINDS`), in its order."""
    return tuple(kind for kind, how in RECORDED_KINDS.items() if how in held)


@dataclasses.dataclass(frozen=True)
class RecordedLeaf:
    """A recorded reference's evidence for one leaf at one seed, all check 1 reads of it (`_drawn`): its
    mean squared error from its own float64 truth, its worst sized row over its size, its RMS error on
    each row the truth sizes zero (below RESOLUTION of its leaf) by the row's index, and whether it and
    its truth were finite; read from a leaf here (`leaf_summary`), also the truth's own RMS on each of
    those rows, which decides whether a row a live reference measures is zero (`_paired`)."""
    mse: float
    worst: float
    zero_rows: np.ndarray
    zero_errors: np.ndarray
    finite: bool
    dtype: str
    zero_sizes: np.ndarray | None = None
    worst_log: float | None = None
    share: float | None = None
    """The share of its worst sized row's squared error the row's largest element carries (`_Draws.factor`),
    zero where no row is sized; None where the leaf was recorded as rows alone (`rows_summary`)."""
    example_mse: np.ndarray | None = None
    example_worst: np.ndarray | None = None
    """At a site whose leaves stack independent examples, each example's evidence (`example_mse`,
    `example_worst`); None elsewhere."""

    @property
    def log_worst(self) -> float:
        """The log of `worst`, exact where this leaf was read here (`leaf_summary`), which keeps its
        order past float64's range; a recording's, from the value it holds."""
        return self.worst_log if self.worst_log is not None else float(log_over(self.worst, 1.0))


def leaf_summary(reference, truth) -> RecordedLeaf:
    """`RecordedLeaf` of a leaf against its float64 truth: check 1's evidence of the reference's leaf, or
    of Dew's (`_drawn` reads both so, a recorded reference's as recorded), read as `_evidence` reads a
    draw's (`leaf_summaries`). An empty leaf is exact."""
    return leaf_summaries([reference], [truth])[0]


def leaf_summaries(references, truths) -> list[RecordedLeaf]:
    """Each reference leaf's `leaf_summary` against its truth, read at once (`_summaries`), each held to
    its truth's shape first: a recording's record (tools/recorded_references.py)."""
    for reference, truth in zip(references, truths, strict=True):
        if np.shape(reference) != np.shape(truth):
            raise AssertionError(f"a leaf {np.shape(reference)} against a truth {np.shape(truth)}")
    # Every reference given, so every leaf has its summary (`_summaries`' None is a reference of None's).
    return [summary for summary in _summaries(references, _LaidTruths.of(truths)) if summary is not None]


def _missed(mse, missed):
    """Each mean squared error where its leaf `missed` its truth somewhere: a leaf whose every error is below
    2^-537 squares to zero, so it takes the least positive float64 instead, and an arm that misses an exact
    reference is never read as exact (check 1's exact-reference rule, `_judgement`)."""
    return np.where((np.asarray(mse) == 0) & missed, np.nextafter(0.0, 1.0), mse)


def rows_summary(rows, truth, *, finite: bool, dtype: str, mse: float | None = None) -> RecordedLeaf:
    """`leaf_summary` of a leaf from each row's RMS distance from its float64 truth `rows` alone, where the
    reference ran apart and kept those (a recorder whose outputs are too large to keep, its rows its
    leaf's every axis but the last, in C order): its mean squared error the rows' mean square, every row
    one width, unless `mse` gives it; `finite` whether the reference's leaf was, and `dtype` its dtype.
    The truth's rows are read as `_summaries` reads them (`_LaidTruths`), and an empty leaf is exact,
    as `leaf_summary` reads one."""
    if not np.size(truth):
        return RecordedLeaf(0.0, 0.0, np.zeros(0, np.int64), np.zeros(0), finite=finite, dtype=dtype)
    held, rows = _LaidTruths.of([truth]), np.asarray(rows, np.float64)
    sized = held.sizes > 0
    logged = float(np.max(log_over(rows[sized], held.sizes[sized]))) if sized.any() else LOG_ZERO
    return RecordedLeaf(float(_missed(np.mean(np.square(rows)), np.any(rows != 0))) if mse is None else mse,
                        float(np.max(_over(rows[sized], held.sizes[sized]))) if sized.any() else 0.0,
                        np.flatnonzero(~sized), rows[~sized], finite=finite and bool(held.finite.all()),
                        dtype=dtype, zero_sizes=held.own[~sized], worst_log=logged)


def _computed_here(program, *trees):
    """`program(*trees, xp=...)` in float64, its small results on the host: in one compiled program where
    every array lives on a GPU, whose float64 arithmetic keeps subnormals as numpy does (IEEE's; XLA:CPU
    reads a subnormal float64 as zero, `1e-310 != 0` false there, and a TPU has no float64), else in numpy,
    on a CPU the arrays' own memory. Where the arrays live decides it, not the default backend: a program
    over arrays a GPU host placed on its CPU would run on XLA:CPU."""
    arrays = [leaf for leaf in jax.tree.leaves(trees) if leaf is not None]
    if not arrays or not all(isinstance(leaf, jax.Array) and all(device.platform == "gpu"
                                                                 for device in leaf.devices())
                             for leaf in arrays):
        return program(*trees, xp=np)
    with jax.enable_x64(new_val=True):
        return jax.device_get(_compiled(program)(*trees))


@functools.cache
def _compiled(statistics):
    """`statistics` over jax.numpy compiled through Dew's compile point (`options.program`) once a process,
    its cache keyed by the leaves' layouts."""
    from dew.nn.kernels import options

    return options.program(functools.partial(statistics, xp=jnp))


def _width(leaf) -> int:
    """A leaf's row width: its last axis's, 0 for a leaf of rank under two, which has no rows."""
    return np.shape(leaf)[-1] if np.ndim(leaf) >= 2 else 0


def _laid(leaf, xp):
    """`leaf` in float64, in C order: its sums run alike whatever layout it came in (numpy's follow
    memory order; a compiled program's leaves are C-ordered)."""
    return np.ascontiguousarray(leaf, np.float64) if xp is np else jnp.asarray(leaf, jnp.float64)


def _truth_rows(truths, terms, *, xp):
    """Each truth leaf's rows' own RMS (`_scaled_rms`) and size, the larger of it and its terms' RMS
    where `terms` holds the leaf's float64 terms, the RMS past RESOLUTION of the leaf's own RMS, else zero;
    and whether its every value is finite. A leaf of rank under two has no rows."""
    found = []
    for truth, term in zip(truths, terms, strict=True):
        truth = _laid(truth, xp)
        width = _width(truth)
        if not width:
            found.append((np.zeros(0), np.zeros(0), xp.isfinite(truth).all()))
            continue
        rows = truth.reshape(-1, width)
        own = _scaled_rms(rows, xp)
        sized = own if term is None else xp.maximum(
            own, _scaled_rms(xp.abs(_laid(term, xp)).reshape(rows.shape), xp))
        resolved = RESOLUTION * xp.sqrt(xp.mean(xp.square(truth)))
        found.append((own, xp.where(sized > resolved, sized, 0.0), xp.isfinite(truth).all()))
    return found


def _error_rows(references, truths, *, xp):
    """Each reference leaf against its truth: its mean squared error, whether it differs anywhere, whether
    its every value is finite, and its error's rows' RMS (`_scaled_rms`) and largest share
    (`_largest_share`)."""
    found = []
    for reference, truth in zip(references, truths, strict=True):
        values, truth = _laid(reference, xp), _laid(truth, xp)
        errors = values - truth
        width = _width(truth)
        rows = errors.reshape(-1, width) if width else xp.zeros((0, 1))
        found.append((xp.mean(xp.square(errors)), xp.any(values != truth), xp.isfinite(values).all(),
                      _scaled_rms(rows, xp), _largest_share(rows, xp)))
    return found


def _largest_share(rows, xp=np):
    """Each row's largest element's share of the row's squared error, the row scaled by its largest first so
    no square overflows: one for a row one element holds alone, zero for a row of zeros. In numpy, or in
    `xp` (jax.numpy, inside check 1's programs)."""
    rows = xp.abs(xp.asarray(rows, xp.float64))
    largest = xp.max(rows, axis=1, keepdims=True) if rows.shape[1] else xp.zeros((len(rows), 1))
    scaled = xp.square(rows / xp.where(largest > 0, largest, 1.0))
    total = xp.sum(scaled, axis=1)
    return xp.where(total > 0, 1.0 / xp.where(total > 0, total, 1.0), 0.0)


@dataclasses.dataclass(frozen=True)
class _LaidTruths:
    """A draw's truth leaves read once for both arms (`_summaries`), every leaf that is not empty in its
    order, its rows in C order (its last axis's, none for a leaf of rank under two). Each row has its own
    RMS (`_scaled_rms`) and its size, the RMS where it is past RESOLUTION of its leaf, else zero
    (`row_errors`'), computed where the leaves live in one program a draw (`_truth_rows`): only these
    rows' statistics leave the device. `truths` is every leaf as given, `laid` the leaves that are not
    empty, `counts` each laid leaf's rows, `own` and `sizes` their rows' own RMS and size, and `finite`
    whether each laid leaf's every value is."""

    truths: list
    laid: list[int]
    counts: np.ndarray
    own: np.ndarray
    sizes: np.ndarray
    finite: np.ndarray

    @staticmethod
    def of(truths, terms=None) -> "_LaidTruths":
        """`truths` laid, each row sized by its own RMS or, given each leaf's float64 `terms`
        (`float64_terms`), by the larger of it and its terms' RMS, as `row_errors` sizes a row that
        cancels (`assert_as_exact`'s `terms`); a leaf no sum made is its own size."""
        laid = [at for at, truth in enumerate(truths) if np.size(truth)]
        found = _computed_here(_truth_rows, [truths[at] for at in laid],
                               [None if terms is None else terms[at] for at in laid])
        return _LaidTruths(truths, laid, np.asarray([np.size(truths[at]) // _width(truths[at])
                                                     if _width(truths[at]) else 0 for at in laid], np.int64),
                           np.concatenate([np.zeros(0), *(np.asarray(own) for own, _, _ in found)]),
                           np.concatenate([np.zeros(0), *(np.asarray(sizes) for _, sizes, _ in found)]),
                           np.asarray([finite for _, _, finite in found], bool))


def _summaries(references, held: _LaidTruths) -> list[RecordedLeaf | None]:
    """Each reference leaf's evidence against its truth in `held` (`_LaidTruths.of`) as check 1 reads it
    (`RecordedLeaf`), every leaf read in one program a draw where the leaves live (`_error_rows`); None
    for a reference of None. The caller holds each reference to its truth's shape (`leaf_summaries`,
    `_evidence`)."""
    names = {}

    def dtype(reference) -> str:
        kind = reference.dtype if hasattr(reference, "dtype") else np.asarray(reference).dtype
        if kind not in names:
            names[kind] = str(jnp.dtype(kind))
        return names[kind]

    # An empty leaf is exact; `held` lays every other.
    summaries: list[RecordedLeaf | None] = [None] * len(references)
    laid = set(held.laid)
    for at, reference in enumerate(references):
        if reference is not None and at not in laid:
            summaries[at] = RecordedLeaf(0.0, 0.0, np.zeros(0, np.int64), np.zeros(0), finite=True,
                                         dtype=dtype(reference))
    read = [references[at] is not None for at in held.laid]
    if not any(read):
        return summaries
    # A leaf of no reference lies as its truth, a zero error never read.
    found = _computed_here(_error_rows, [references[at] if ok else held.truths[at]
                                         for ok, at in zip(read, held.laid, strict=True)],
                           [held.truths[at] for at in held.laid])
    mse = _missed(np.asarray([mse for mse, *_ in found], np.float64),
                  np.asarray([missed for _, missed, *_ in found], bool))
    finite = np.asarray([finite for _, _, finite, *_ in found], bool)
    rows = np.concatenate([np.zeros(0), *(np.asarray(rows) for *_, rows, _ in found)])
    shares = np.concatenate([np.zeros(0), *(np.asarray(shares) for *_, shares in found)])
    # Each laid leaf's rows, its first among them, and those sized past RESOLUTION.
    counts = held.counts
    starts = np.cumsum(counts) - counts
    sized = held.sizes > 0
    before = np.concatenate([[0], np.cumsum(sized)])
    taken = before[starts + counts] - before[starts]
    logs, over = log_over(rows[sized], held.sizes[sized]), _over(rows[sized], held.sizes[sized])
    # The worst sized row of each leaf that has one: the largest of its run of the sized rows.
    some = taken > 0
    worst_log, worst, share = np.full(len(counts), LOG_ZERO), np.zeros(len(counts)), np.zeros(len(counts))
    if some.any():
        firsts = (np.cumsum(taken) - taken)[some]
        worst_log[some] = np.maximum.reduceat(logs, firsts)
        worst[some] = np.maximum.reduceat(over, firsts)
        # The worst row's largest element's share of its squared error, each leaf's worst sized row's.
        sized_shares = shares[sized]
        share[some] = [sized_shares[first + int(np.argmax(logs[first:first + count]))]
                       for first, count in zip(firsts.tolist(), taken[some].tolist(), strict=True)]
    # The rows of float64 zero, each leaf's its span of them.
    unsized = np.flatnonzero(~sized)
    zero_rows = unsized - np.repeat(starts, counts - taken)
    zero_errors, zero_sizes = rows[unsized], held.own[unsized]
    ends = list(itertools.accumulate((counts - taken).tolist(), initial=0))
    mse, worst, worst_log, share, finite = mse.tolist(), worst.tolist(), worst_log.tolist(), share.tolist(), (
        finite & held.finite).tolist()
    for index, at in enumerate(held.laid):
        if read[index]:
            span = slice(ends[index], ends[index + 1])
            summaries[at] = RecordedLeaf(mse[index], worst[index], zero_rows[span], zero_errors[span],
                                         finite=finite[index], dtype=dtype(references[at]),
                                         zero_sizes=zero_sizes[span], worst_log=worst_log[index],
                                         share=share[index])
    return summaries


@dataclasses.dataclass
class _Draws:
    """Each leaf's errors over a run of seeds: Dew's and the reference's mean squared error from
    float64 (seed by leaf), the log of each seed's worst sized row over its size (Dew's, the
    reference's; `log_over`), each
    seed's rows of float64 zero with Dew's and the reference's errors on them (`ZeroRows`, seed by leaf;
    `pooled_zero_rows`), whether every value of the leaf was finite on every seed, and each leaf's row
    width (its last axis's length, 0 for a leaf of no rows; NARROW_FACTORS)."""
    names: list[str]
    mine: np.ndarray
    theirs: np.ndarray
    worst: np.ndarray
    zero_rows: list[list[ZeroRows]]
    finite: np.ndarray
    widths: tuple[int, ...]
    examples: list[list[np.ndarray | None]] | None = None
    """A site whose leaves stack independent examples on their first axis (`assert_as_exact`'s
    `examples`): each seed's per-example log RMS ratios of each leaf (`_example_ratios`), else None."""
    example_rows: list[list[np.ndarray | None]] | None = None
    """At such a site, each seed's per-example log ratio of Dew's worst row over the reference's worst row
    on the same example, each over its size (`_example_worst_rows`), else None."""
    shares: np.ndarray | None = None
    """The share of each seed's worst row's squared error its largest element carries, Dew's and the
    reference's (seed by leaf by arm, as `worst`), zero where no row is sized and NaN where it is unknown (a
    recording's summary keeps no row's elements, a row of float64 zero a live reference measured); None
    where none was kept (`held_shares`)."""

    def factor(self, index: int) -> float:
        """The worst-row rule's factor for leaf `index` (its maximum's, `_judgement`): its rows' width's
        (`row_factor`), but a row of one value's at rows of more than two values where one element carries
        past RULED_SHARE of both Dew's worst row and the reference's over these draws, the two rows one
        value rules alike. A share unknown keeps the width's."""
        width = self.widths[index]
        if width > 2:
            shares = self.held_shares()
            mine, theirs = (int(np.argmax(self.worst[:, index, arm])) for arm in (0, 1))
            if shares[mine, index, 0] > RULED_SHARE and shares[theirs, index, 1] > RULED_SHARE:
                return row_factor(1)
        return row_factor(width)

    def held_shares(self) -> np.ndarray:
        """`shares`, every cell unknown (NaN) where none was kept, so draws join and pick per cell."""
        return np.full(self.worst.shape, np.nan) if self.shares is None else self.shares

    def joined(self, other: "_Draws") -> "_Draws":
        examples, example_rows = ((getattr(self, field) + getattr(other, field)
                                   if getattr(self, field) is not None and getattr(other, field) is not None
                                   else None) for field in ("examples", "example_rows"))
        shares = np.concatenate([self.held_shares(), other.held_shares()])
        return _Draws(self.names, np.concatenate([self.mine, other.mine]),
                      np.concatenate([self.theirs, other.theirs]), np.concatenate([self.worst, other.worst]),
                      self.zero_rows + other.zero_rows, np.concatenate([self.finite, other.finite]),
                      self.widths, examples, example_rows, shares)

    def between(self, start: int, stop: int) -> "_Draws":
        """These draws' seeds `start` to `stop`, counted from their first."""
        cut = slice(start, stop)
        examples, example_rows = (None if getattr(self, field) is None else getattr(self, field)[cut]
                                  for field in ("examples", "example_rows"))
        return _Draws(self.names, self.mine[cut], self.theirs[cut], self.worst[cut], self.zero_rows[cut],
                      self.finite[cut], self.widths, examples, example_rows,
                      None if self.shares is None else self.shares[cut])

    def of_examples(self, index: int) -> list[np.ndarray] | None:
        """Leaf `index`'s per-example log ratios a seed, or None where the site stacks none or a seed's
        are not all paired (`_example_ratios`)."""
        if self.examples is None:
            return None
        found = [ratios for seed in self.examples if (ratios := seed[index]) is not None]
        return found if len(found) == len(self.examples) else None

    def of_example_rows(self, index: int) -> np.ndarray | None:
        """Leaf `index`'s per-example paired worst-row log ratios over every seed, or None where the site
        stacks no examples or the leaf has no rows (`_example_worst_rows`)."""
        if self.example_rows is None:
            return None
        found = [rows for seed in self.example_rows if (rows := seed[index]) is not None]
        return np.concatenate(found) if len(found) == len(self.example_rows) else None

    def scalar(self, index: int) -> bool:
        """Whether leaf `index` holds one value an example (its width, the values a row, one)."""
        return self.widths[index] <= 1

    def against_larger(self, library: "_Draws", leaves: Sequence[str]) -> "_Draws":
        """These draws judged against the larger of their reference's error and `library`'s, the same
        Dew on the same draws against the library arm (`assert_as_exact`'s `library_fn`), on the
        exception's `leaves` alone: seed by seed and leaf by leaf, the reference's evidence (its error,
        its worst row, its rows of float64 zero) is whichever of the two arms' error is the larger there,
        and a value must be finite in both. Every other leaf is the reference's alone."""
        if library.names != self.names or len(library.mine) != len(self.mine):
            raise AssertionError("the library arm's draws are not the reference's")
        unknown = sorted(set(leaves) - set(self.names))
        if unknown or not leaves:
            raise ValueError(f"a library arm names the leaves it covers, of {self.names}; it names "
                             f"{list(leaves)}")
        covered = np.isin(self.names, leaves)
        pick = (library.theirs > self.theirs) & covered
        worst = self.worst.copy()
        worst[..., 1] = np.where(pick, library.worst[..., 1], self.worst[..., 1])
        zero_rows = [[(theirs if chosen else mine)
                      for mine, theirs, chosen in zip(seed, other, picked, strict=True)]
                     for seed, other, picked in zip(self.zero_rows, library.zero_rows, pick, strict=True)]
        # A library arm's leaves are judged by their pooled ratio alone: their examples are not Dew's against
        # one reference.
        # A picked cell's worst row is the library arm's, and so is its share.
        shares = self.held_shares().copy()
        shares[..., 1] = np.where(pick, library.held_shares()[..., 1], shares[..., 1])
        return _Draws(self.names, self.mine, np.where(pick, library.theirs, self.theirs), worst, zero_rows,
                      self.finite & (library.finite | ~covered), self.widths, shares=shares)


def _shares(found: list[list], count: int) -> np.ndarray:
    """The worst rows' shares over a run's seeds (`_Draws.shares`, seed by leaf by arm), NaN where one is
    unknown (a recording's summary keeps no row's elements)."""
    return np.asarray([[[np.nan if share is None else share for share in pair] for pair in seed]
                       for seed in found], np.float64).reshape(count, -1, 2)


def example_mse(arm, truth) -> np.ndarray | None:
    """An arm's mean squared error from its float64 `truth` on each of a leaf's independent examples, its
    first axis (`judge`'s `examples`); None for a leaf of no axis."""
    truth = np.asarray(truth, np.float64)
    if truth.ndim == 0:
        return None
    return np.mean(np.square(np.asarray(arm, np.float64) - truth), axis=tuple(range(1, truth.ndim)))


def example_worst(arm, truth) -> np.ndarray | None:
    """An arm's worst sized row on each of a leaf's independent examples: the log of its RMS error from its
    float64 `truth` over the row's size, the rows those of the leaf's last axis, sized by their truth's RMS,
    rows under RESOLUTION of their example's size left out. None for a leaf under rank two (no rows), or
    one with an example of no sized row."""
    truth = np.asarray(truth, np.float64)
    if truth.ndim < 2:
        return None
    rows = truth.reshape(truth.shape[0], -1, truth.shape[-1])
    sizes = np.stack([_scaled_rms(example) for example in rows])
    sized = sizes > RESOLUTION * np.sqrt(np.mean(np.square(rows), axis=(1, 2)))[:, None]
    if not sized.any(axis=1).all():
        return None
    errors = (np.asarray(arm, np.float64) - truth).reshape(rows.shape)
    logs = log_over(np.stack([_scaled_rms(example) for example in errors]), np.where(sized, sizes, 1.0))
    return np.max(np.where(sized, logs, LOG_ZERO), axis=1)


def _theirs(reference, truth, evidence, field: str):
    """The reference's per-example `evidence` of a leaf: a recorded leaf's as its recording holds it
    (`RecordedLeaf.<field>`), a live one's read against `truth`."""
    return getattr(reference, field) if isinstance(reference, RecordedLeaf) else evidence(reference, truth)


def _example_ratios(leaves, references, truths) -> list[np.ndarray | None]:
    """Each leaf's per-example log RMS ratios of Dew (`leaves`) against the reference (`references`, a
    tree of the truth's structure, live or recorded): half the log of each example's mean squared error
    from its float64 truth over the reference's (`example_mse`), examples both arms hold exactly left out.
    None for a leaf of no axis, or where one arm is exact on an example and the other is not."""
    found: list[np.ndarray | None] = []
    for leaf, reference, truth in zip(leaves, jax.tree.leaves(references), truths, strict=True):
        mine, theirs = example_mse(leaf, truth), _theirs(reference, truth, example_mse, "example_mse")
        if mine is None or theirs is None or np.any((mine > 0) != (theirs > 0)):
            found.append(None)
            continue
        kept = mine > 0
        found.append(0.5 * np.log(mine[kept] / theirs[kept]))
    return found


def _example_worst_rows(leaves, references, truths) -> list[np.ndarray | None]:
    """Each leaf's per-example paired worst rows at a site that stacks examples (`judge`'s `examples`): on
    each example, Dew's worst sized row less the reference's on the same example (`example_worst`, the
    reference's live or recorded). None for a leaf under rank two (no rows), or an example with no sized
    row."""
    found: list[np.ndarray | None] = []
    for leaf, reference, truth in zip(leaves, jax.tree.leaves(references), truths, strict=True):
        mine, theirs = example_worst(leaf, truth), _theirs(reference, truth, example_worst, "example_worst")
        found.append(None if mine is None or theirs is None else mine - theirs)
    return found


def _leaf_paths(tree) -> list[str]:
    return [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(tree)]


@dataclasses.dataclass(frozen=True)
class _Prepared:
    """Dew's side of a draw's evidence, read once for every reference (`_evidence`): its leaves and its
    truth's where they live, their paths, the truth's laid rows and Dew's summaries against them."""
    truth: object
    paths: list
    leaves: list
    truths: list
    held: _LaidTruths
    mine: list


def _term_leaves(terms, truth):
    """A truth's terms tree (`float64_terms`) as its leaves on the host, one a leaf of `truth`, a leaf no
    sum made None; or None. Read by the truth's own structure, so an output the truth holds as None (a
    gradient of an input given as None) takes no leaf of the terms either."""
    if terms is None:
        return None
    return jax.tree.structure(truth).flatten_up_to(terms)


def _prepared(dew, truth, terms=None) -> _Prepared:
    """Dew's leaves and its truth's where they live, held alike (the same leaves at the same paths, each
    the shape of its truth), the truth's rows laid once and Dew's summaries read against them
    (`_summaries`): only their rows' statistics leave the device."""
    paths = _leaf_paths(truth)
    if _leaf_paths(dew) != paths:
        raise AssertionError(f"dew's leaves are {_leaf_paths(dew)}, the truth's {paths}")
    leaves, truths = jax.tree.leaves(dew), jax.tree.leaves(truth)
    for path, leaf, true in zip(paths, leaves, truths, strict=True):
        if np.shape(leaf) != np.shape(true):
            raise AssertionError(f"{path}: dew's shape is {np.shape(leaf)}, the truth's {np.shape(true)}")
    held = _LaidTruths.of(truths, _term_leaves(terms, truth))
    return _Prepared(truth, paths, leaves, truths, held, _summaries(leaves, held))


def _evidence(prepared: _Prepared, reference, their_truth, *, shared: bool, their_terms=None) -> list[tuple]:
    """One draw's `_paired` evidence of every leaf of Dew (`prepared`, read once a draw) and `reference`
    against their truths: the reference's leaves (a recorded one's `RecordedLeaf`) and its truth held to
    Dew's truth's leaves, paths and shapes, so no leaf is paired with another's truth or compared through
    broadcasting; its truth the same tree as Dew's where `shared`."""
    their_truth = prepared.truth if shared else their_truth
    paths, leaves, truths = prepared.paths, prepared.leaves, prepared.truths
    for name, tree in (("the reference", reference), ("the reference's truth", their_truth)):
        if _leaf_paths(tree) != paths:
            raise AssertionError(f"{name}'s leaves are {_leaf_paths(tree)}, the truth's {paths}")
    references, their_truths = jax.tree.leaves(reference), jax.tree.leaves(their_truth)
    recorded = [isinstance(their_leaf, RecordedLeaf) for their_leaf in references]
    for path, their_leaf, true, their_true, was in zip(paths, references, truths, their_truths, recorded,
                                                       strict=True):
        shapes = {"the reference's truth": np.shape(their_true)}
        if not was:
            shapes["the reference"] = np.shape(their_leaf)
        for name, shape in shapes.items():
            if shape != np.shape(true):
                raise AssertionError(f"{path}: {name}'s shape is {shape}, the truth's {np.shape(true)}")
    # The truths' rows are read once a draw, for both arms where the two truths are the same values.
    same = all(their_true is true for their_true, true in zip(their_truths, truths, strict=True))
    theirs = _summaries([None if was else leaf for leaf, was in zip(references, recorded, strict=True)],
                        prepared.held if same else _LaidTruths.of(their_truths,
                                                                   _term_leaves(their_terms, their_truth)))
    row = []
    for path, leaf, their_leaf, true, their_true, was, summary, their_summary in zip(
            paths, leaves, references, truths, their_truths, recorded, prepared.mine, theirs, strict=True):
        their = their_leaf if was else their_summary
        if np.size(leaf) == 0:
            # An empty leaf (a one-tap convolution's history) is exact by construction: held to the
            # reference's dtype (its shape is held above; the truth's dtype, float64, is its own), it
            # adds no error and resolves nothing.
            kept = str(jnp.dtype(leaf.dtype))
            assert kept == their.dtype, (path, kept, their.dtype)
            row.append((0.0, 0.0, LOG_ZERO, LOG_ZERO, NO_ZERO_ROWS, True, (0.0, 0.0)))
            continue
        # Rows pair where the two truths are the same values: the same draw, or an orbit's draws.
        paired = was or shared or np.array_equal(np.asarray(true), np.asarray(their_true))
        row.append(_paired(path, summary, their, paired=paired))
    return row


def _paired(name: str, mine: RecordedLeaf, theirs: RecordedLeaf, *, paired: bool = True) -> tuple:
    """One seed's evidence of a leaf for check 1 (`_Draws`), Dew's and the reference's alike from
    `leaf_summary`: their mean squared errors, their worst sized rows, the rows of float64 zero with
    both arms' errors on them (`ZeroRows`, pooled over the run's seeds by `pooled_zero_rows`), and
    whether every value was finite. A row the truth sizes zero (below RESOLUTION of its leaf)
    is one of float64 zero unless a live reference measures it: its error there below the row's own
    float64 RMS, so it carries the row better than zero would. The two are taken by one arithmetic
    (`_scaled_rms`), so a reference exactly zero on a row is exactly the row's size from it and never
    measures it (ledger 1831: two arithmetics broke that tie, and Dew's ordinary rounding on a residue
    row read 1e8 of the row). That is an engineering definition of a measured row, not a proof of
    float64's accuracy on it; a sum that cancels to float64's residue is never measured so, and a
    product far below its leaf, which float64 holds to its own precision, is.
    A measured row is judged at its own size as any row is, its error over its size entering both worst
    rows: against the reference's worst row over the leaf, which holds it with less power than the zero
    rows' rule held it against the reference's own error on it. Both arms are classified alike, from the
    one truth they share. A recorded reference keeps no row's own size, and the recording's sketch holds
    the truth at the leaf's scale, not a tiny row's, so its rows keep the zero rows' rule. A row of zero
    in Dew's truth the reference's lacks (a recorded truth recomputed otherwise) is refused. Not
    `paired`, the two held to truths of their own draws (`null_runs`), no row is paired."""
    if not paired:
        return (mine.mse, theirs.mse, mine.log_worst, theirs.log_worst, NO_ZERO_ROWS,
                mine.finite and theirs.finite, (mine.share, theirs.share))
    at_zero = dict(zip(theirs.zero_rows.tolist(), theirs.zero_errors.tolist(), strict=True))
    unrecorded = [at for at in mine.zero_rows.tolist() if at not in at_zero]
    if unrecorded:
        raise AssertionError(f"{name}: the truth holds rows of float64 zero the reference's did not, "
                             f"{unrecorded[:4]}")
    sizes = (dict(zip(theirs.zero_rows.tolist(), theirs.zero_sizes.tolist(), strict=True))
             if theirs.zero_sizes is not None else {})
    worst, their_worst, zero = mine.log_worst, theirs.log_worst, []
    for at, error in zip(mine.zero_rows.tolist(), mine.zero_errors.tolist(), strict=True):
        size = sizes.get(at, 0.0)
        if at_zero[at] < size:
            worst = max(worst, float(log_over(error, size)))
            their_worst = max(their_worst, float(log_over(at_zero[at], size)))
        elif error or at_zero[at]:
            # A row both arms hold exactly scales both pooled errors alike and passes on every seed: no
            # evidence, not carried (a window's untouched embedding rows, 283k a draw at 2469's chunk sites).
            zero.append((at, error, at_zero[at]))
    rows, errors, their_errors = (np.asarray(column, dtype) for column, dtype in zip(
        zip(*zero, strict=True) if zero else ((), (), ()), (np.int64, np.float64, np.float64), strict=True))
    # A measured row of float64 zero that became an arm's worst keeps no share of its own: unknown.
    shares = (mine.share if worst == mine.log_worst else None,
              theirs.share if their_worst == theirs.log_worst else None)
    return (mine.mse, theirs.mse, worst, their_worst, (rows, errors, their_errors),
            mine.finite and theirs.finite, shares)


def _judged(draws: _Draws, index: int, tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL)
            ) -> tuple[float, float, float]:
    """A leaf's pooled ratio and its interval over the draws, its `tails` left out."""
    a, b = draws.mine[:, index], draws.theirs[:, index]
    return (_pooled(a, b), *_interval(a, b, tails=tails))


def _at_parity(draws: _Draws, index: int, leaf: str, tails: tuple[float, float], *, contained: bool,
               typical: bool) -> Finding | None:
    """Check 1's rule for a leaf's pooled ratio (`_judgement`'s `ratio_rule`): its finding where the leaf
    is over PARITY (`_status`) or its draws leave it inconclusive, none where it passes; with `typical` (a
    final verdict), a leaf its pooled ratio leaves inconclusive passed by its typical draw where that
    decides (`_typical_status`). A leaf both hold exactly on every seed has no ratio and passes."""
    a, b, count = draws.mine[:, index], draws.theirs[:, index], len(draws.mine)
    if np.sum(b) == 0:
        return None
    ratio, low, high = _judged(draws, index, tails)
    status = _status(ratio, low, high, contained=contained)
    # The ratio's log from its sums' logs, finite where a sum or the ratio itself would overflow.
    measure = (_log_sum(a) - _log_sum(b)) / 2
    found = None
    if status == "inconclusive" and contained and typical:
        # The pooled ratio's sums unresolved: its draws' paired ratios pass it where they qualify.
        found = _typical_status(a, b, tails, draws.of_examples(index), draws.scalar(index))
        status = found[0]
    if status == "inconclusive":
        return Finding(
            leaf, "inconclusive", measure, f"{leaf}: inconclusive: the pooled ratio {ratio:.3f}'s "
            f"interval over {count} seeds is [{low:.3f}, {high:.3f}], wider than {PARITY}"
            + ("" if found is None else f", and the typical draw's {found[1]:.3f}'s "
               f"[{found[2]:.3f}, {found[3]:.3f}]: {found[4]}")
            + "; give the site larger inputs or more seeds (`seeds_for`)")
    if status == "over":
        return Finding(leaf, "pooled", measure, f"{leaf}: dew's RMS error from float64 is "
                       f"{ratio:.3f} times the reference's over {count} seeds, interval "
                       f"[{low:.3f}, {high:.3f}] (allowed {PARITY})")
    return None


def _below_one(draws: _Draws, index: int, leaf: str, tails: tuple[float, float], *, contained: bool,
               typical: bool) -> Finding | None:
    """The FP8 products' rule for a leaf's pooled ratio against a named arm (`_judgement`'s `ratio_rule`,
    docs/design/testing.md, "FP8 products"): none where Dew's error is below the arm's, its ratio's interval
    (`_judged`, taken to hold the ratio) wholly under one; a "pooled" finding where it lies wholly at or over
    one; else the leaf is inconclusive, as a ratio or an end that is not a number is. Below is certified by
    the whole interval at either stage: no interval is resolved by its width and no typical draw passes a
    leaf, as check 1's parity allows (`contained`, `typical`). A leaf both hold exactly on every seed has
    nothing to be below and fails with no measure, a form at parity being check 1's to hold: a finding as
    any other, so a first stage of exact seeds draws the second before it is the verdict."""
    a, b, count = draws.mine[:, index], draws.theirs[:, index], len(draws.mine)
    if np.sum(b) == 0:
        return Finding(leaf, "no measure", 0.0, f"{leaf}: dew and the named arm both equal float64 on "
                       f"every one of {count} seeds, so nothing is below; a form at parity holds check 1 "
                       f"(`assert_as_exact`)")
    ratio, low, high = _judged(draws, index, tails)
    measure = (_log_sum(a) - _log_sum(b)) / 2
    found = (f"{leaf}: dew's RMS error from float64 is {ratio:.3f} times the named arm's over {count} seeds, "
             f"interval [{low:.3f}, {high:.3f}]")
    if not all(np.isfinite(value) for value in (ratio, low, high)):
        return Finding(leaf, "inconclusive", measure, f"{found}: inconclusive, not a number")
    low, high = min(low, ratio), max(high, ratio)
    if high < 1:
        return None
    if low >= 1:
        return Finding(leaf, "pooled", measure, f"{found} (it must be below one)")
    return Finding(leaf, "inconclusive", measure, f"{found}: inconclusive, the interval reaches one; give "
                   f"the site more seeds")


def _judgement(draws: _Draws, label: str, tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL), *,
               contained: bool = True, typical: bool = False,
               ratio_rule=_at_parity) -> tuple[list[Finding], list[Finding]]:
    """Check 1's findings over `draws` (`assert_as_exact`): each leaf its pooled ratio fails (`ratio_rule`:
    over PARITY, `_at_parity`; under the FP8 products' contract not below a named arm, `_below_one`), a
    row or a value it fails, and each leaf its draws leave inconclusive; with `typical` (a final verdict),
    a leaf its pooled ratio leaves inconclusive passed by its typical draw where that decides."""
    over, unresolved = [], []
    for index, name in enumerate(draws.names):
        leaf, count = label + name, len(draws.mine)
        a, b = draws.mine[:, index], draws.theirs[:, index]
        if not draws.finite[:, index].all():
            nonfinite = int(np.sum(~draws.finite[:, index]))
            over.append(Finding(leaf, "not finite", nonfinite,
                                f"{leaf}: a value is not finite on {nonfinite} of {count} seeds"))
            continue
        if np.sum(b) == 0 and np.sum(a) > 0:
            # No ratio to judge, but rows over an orbit's other orders still are; Dew's own RMS error is
            # what a change can make worse.
            over.append(Finding(leaf, "exact reference", _root_mean(a),
                                f"{leaf}: the reference equals float64 on every seed and dew does not"))
        else:
            found = ratio_rule(draws, index, leaf, tails, contained=contained, typical=typical)
            if found is not None:
                (unresolved if found.rule == "inconclusive" else over).append(found)
        # The paired examples' test keeps its width's factor (its null is exchangeability, whatever a row's
        # error is); the maximum's takes a row one value rules at one value's (`_Draws.factor`).
        factor = row_factor(draws.widths[index])
        paired = draws.of_example_rows(index) if contained else None
        if paired is not None and (found_rows := _paired_rows(paired, factor)) < tails[0]:
            # A site that stacks examples: each example's worst row against the reference's on it.
            over.append(Finding(
                leaf, "row", float(np.quantile(paired, 1 - tails[0])),
                f"{leaf}: dew's worst row is past {factor} times the reference's on the same example on "
                f"{int(np.sum(paired > math.log(factor)))} of {len(paired)} examples, the reference's past "
                f"it on {int(np.sum(paired < -math.log(factor)))} (p {found_rows:.1e}; `_paired_rows`)"))
        my_row, their_row = float(np.max(draws.worst[:, index, 0])), float(np.max(draws.worst[:, index, 1]))
        factor = draws.factor(index)
        if paired is None and not my_row <= math.log(factor) + their_row:
            over.append(Finding(leaf, "row", my_row - their_row,
                                f"{leaf}: the worst row over {count} seeds is {_shown(my_row)} from float64 "
                                f"over its size, the reference's {_shown(their_row)}, allowed "
                                f"{_shown(math.log(factor) + their_row)} ({factor} times it)"))
        zero = pooled_zero_rows([seed[index] for seed in draws.zero_rows])
        if zero:
            # The measure is the worst of them, the first the example.
            at, my_error, their_error = zero[0]
            worst = log_ratio([row[1] for row in zero], [row[2] for row in zero])
            over.append(Finding(leaf, "zero rows", worst,
                                f"{leaf}: {len(zero)} rows of float64 zero are further from it over "
                                f"{count} seeds than ROW_FACTOR times the reference's own, row {at} by "
                                f"{my_error:.3e} against {their_error:.3e} (`pooled_zero_rows`)"))
    return over, unresolved


def _staged(drawn, seeds: int, label: str, *, two_stage: bool = True,
            ratio_rule=_at_parity) -> tuple[list[Finding], list[Finding], int]:
    """Check 1's two pre-declared stages over `drawn(start, stop)`, the draws of seeds start to stop: at
    `seeds`, a site whose every leaf passes (`_judgement`: its ratio by `ratio_rule`, its rows and values)
    passes; any leaf that does not draws the second stage, REDRAW times as many further seeds, and every
    leaf is judged again over all of them, that verdict final. Its findings, the leaves left inconclusive
    and the stage reached. With `two_stage` false, the procedure before it, for comparison: tails
    PREVIOUS_TAILS, the interval as it is, only a leaf whose ratio is over PARITY drawing again."""
    tails = (LOWER_TAIL, UPPER_TAIL) if two_stage else PREVIOUS_TAILS
    draws = drawn(0, seeds)
    if two_stage:
        further = any(_judgement(draws, label, tails, ratio_rule=ratio_rule))
    else:
        further = any(np.sum(draws.theirs[:, index]) > 0 and _judged(draws, index, tails)[0] > PARITY
                      for index in range(len(draws.names)))
    if further:
        draws = draws.joined(drawn(seeds, staged_units(seeds)))
    # The stage drawn by the pooled ratio alone; the final verdict's leaves it leaves inconclusive asked of
    # their typical draw (`_judgement`'s `typical`).
    return (*_judgement(draws, label, tails, contained=two_stage, typical=two_stage, ratio_rule=ratio_rule),
            2 if further else 1)


# Check 1's conditioning (ledger 1714): a site declares the weighed check's own seeds beside its `seeds`;
# the unweighted check runs first, whole, and decides where it can. Only where it is still Inconclusive
# with its budget spent is the site judged again, weighed: each arm's element e at its truth t read as the
# residual (e - t) / c against a truth of zeros, every row of unit size, c the reference program's
# rounding scale there, the same for every arm (_dew_torch.rounding's model, tests/rounding_bound.py's
# for a JAX program). A leaf's heavy elements (`heavy_masks`) are judged as a leaf of their own beside the
# light rest, each class's worst rows by the gate's factor for its values a row (`row_factor`); that check
# must decide a pass. An element no rounding reaches (c = 0) must be exact in every arm. Where a
# computation amplifies its rounding unevenly (an inner step past its stable rate), the unweighted pool is
# the amplified elements' and its interval stays wide; weighed, a fault on the well-conditioned rest
# shows, and one on the amplified elements shows on their own class. Check 2 is never weighed.


def float_leaf(leaf) -> bool:
    """Whether `leaf` holds floats, bfloat16's among them, as JAX's dtypes read it."""
    return jnp.issubdtype(jnp.result_type(leaf), jnp.floating)


@dataclasses.dataclass(frozen=True)
class Weighed:
    """One truth's weighing for check 1's conditioning (`Conditioning`): the `truth` on the host, the
    reference program's rounding `scale` at its inputs (each element's RMS displacement under its program's
    roundings, _dew_torch.rounding), applied to every arm alike, and the truth of zeros and unit sizes
    (`zeros`, `ones`) every weighed arm is held to."""

    truth: object
    scale: object
    zeros: object
    ones: object
    heavy: list | None = None
    """Each floating leaf's heavy elements (`_heavy`), a mask, None for a leaf of none: its residuals
    judged as a leaf of their own beside the light ones' (`classed`)."""

    def classed(self, tree):
        """`tree`, of the truth's structure (its leaves by the truth's treedef, a structural None kept out),
        each floating leaf with heavy elements (`heavy`) split into its classes, each class's rows compacted
        by the count of its values a row: `{"heavy": {k: [rows of k heavy values, k]}, "light": {...}}`,
        so each class is judged as its own leaves, every row at its own width (`row_factor`) and its RMS
        over its own values; a leaf of rank one splits into its classes' values, of no rows."""
        if self.heavy is None:
            return tree
        structure = jax.tree.structure(self.truth)
        split = [leaf if mask is None or leaf is None else _compacted(np.asarray(leaf), mask)
                 for leaf, mask in zip(structure.flatten_up_to(tree), self.heavy, strict=True)]
        return structure.unflatten(split)

    def weighed(self, tree, arm: str):
        """`tree`, an arm's outputs, each floating element e at its truth t and scale c read as its residual
        (e - t) / c, c the element's RMS displacement under the reference program's roundings, against a
        truth of zeros: its error in units of the rounding the reference's program does there, every row
        of unit size (`ones`, the
        terms `_prepared` sizes rows by). Each leaf's shape is checked before any arithmetic. An element
        no rounding reaches (c = 0) is exact in the reference's program, and an arm not exact there fails,
        by name, as does an element whose scale is not finite, which no error can be read against."""
        structure = jax.tree.structure(self.truth)
        paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(self.truth)]
        leaves = structure.flatten_up_to(jax.device_get(tree))
        truths = structure.flatten_up_to(self.truth)
        for path, leaf, true in zip(paths, leaves, truths, strict=True):
            if np.shape(leaf) != np.shape(true):
                raise AssertionError(f"{arm}, {path}: its shape is {np.shape(leaf)}, the truth's "
                                     f"{np.shape(true)}")
        weighed = []
        for path, leaf, true, scale in zip(paths, leaves, truths, structure.flatten_up_to(self.scale),
                                           strict=True):
            if scale is None:
                # A leaf of no float (an index, a count) is judged as it is.
                weighed.append(leaf)
                continue
            value, true, scale = (np.asarray(part, np.float64) for part in (leaf, true, scale))
            unweighable = ~np.isfinite(scale)
            if unweighable.any():
                at = tuple(int(index) for index in np.argwhere(unweighable)[0])
                raise AssertionError(f"{arm}, {path}{list(at)}: the reference's rounding scale is "
                                     f"{scale[at]!r} there, which no error can be weighed against")
            exact = scale == 0
            wrong = exact & ~(value == true)
            if wrong.any():
                # A decided failure of an exact reference's element (`Finding`), the most an arm misses by.
                at = tuple(int(index) for index in np.argwhere(wrong)[0])
                raise GateFailure([Finding(
                    f"{arm}, {path}", "exact reference", float(np.max(np.abs(value - true)[wrong])),
                    f"{arm}, {path}{list(at)}: no rounding of the reference's program reaches it (its "
                    f"rounding scale is zero), and the arm is not exact: {value[at]!r} against "
                    f"{true[at]!r}")])
            weighed.append(np.where(exact, 0.0, (value - true) / np.where(exact, 1.0, scale)))
        return self.classed(structure.unflatten(weighed)), self.classed(self.zeros)


def heavy_masks(runs) -> list:
    """Each floating output leaf's heavy elements over `runs`, the reference's outputs beside their float64
    truths draw by draw, as a mask, None for a leaf of none: those whose share of the reference's squared
    error summed over the draws exceeds their uniform share of their row (1/n, n its length along the leaf's
    last axis). The unweighted RMS, which defines on par, is theirs, so the weighed check judges them apart
    from the light rest (`Weighed.classed`): an error moved onto them never hides behind the rest's."""
    totals = None
    for ran, true in runs:
        structure = jax.tree.structure(true)
        squares = [np.square(np.asarray(arm, np.float64) - np.asarray(truth, np.float64))
                   if float_leaf(truth) and np.ndim(truth) else None
                   for arm, truth in zip(structure.flatten_up_to(ran), structure.flatten_up_to(true),
                                         strict=True)]
        totals = squares if totals is None else [None if total is None else total + square
                                                 for total, square in zip(totals, squares, strict=True)]
    masks = []
    for total in totals or []:
        heavy = None if total is None else total * total.shape[-1] > np.sum(total, axis=-1, keepdims=True)
        masks.append(heavy if heavy is not None and heavy.any() else None)
    return masks


@dataclasses.dataclass(frozen=True)
class Conditioning:
    """Check 1's conditioning (`judge`'s `conditioning`, tests/reference_error.py's `assert_as_exact`'s):
    `scale_of(inputs, seed)`, the reference program's rounding scale at a draw's inputs, which weighs every
    arm's error there (`Weighed`), and each floating output leaf's `heavy` elements (`heavy_masks`), which
    the weighed check judges apart."""

    scale_of: Callable
    heavy: list | None = None

    def digest(self) -> str:
        """The classes' digest (`heavy`, each mask's shape and values), which every map task of a sharded
        site records and its judge holds one (tests/reference_error.py)."""
        found = hashlib.sha256()
        for mask in self.heavy or []:
            found.update(b"none" if mask is None else repr(mask.shape).encode() + np.packbits(mask).tobytes())
        return found.hexdigest()

    def at(self, cases) -> dict[int, Weighed]:
        """Each truth's weighing by its id, from `cases` of (truth, its inputs, its seed), a truth held
        twice weighed once."""
        weighed: dict[int, Weighed] = {}
        for truth, inputs, seed in cases:
            if id(truth) not in weighed:
                weighed[id(truth)] = self.of(truth, inputs, seed)
        return weighed

    def of(self, truth, inputs, seed) -> Weighed:
        """The weighing of `truth`, the truth of `inputs`, at the draw `seed`."""
        truth = jax.device_get(truth)
        scale = self.scale_of(inputs, seed)
        zeros = jax.tree.map(lambda leaf: np.zeros(np.shape(leaf)) if float_leaf(leaf) else leaf, truth)
        ones = jax.tree.map(lambda leaf: np.ones(np.shape(leaf)) if float_leaf(leaf) else None, truth)
        weighing = Weighed(truth, scale, zeros, ones, self.heavy)
        return dataclasses.replace(weighing, ones=weighing.classed(ones))


def _compacted(leaf: np.ndarray, mask: np.ndarray) -> dict:
    """`leaf`'s heavy (`mask`) and light values apart, each class's rows grouped by how many of its values
    a row holds, each group `[its rows, that count]` in row order (`Weighed.classed`); a leaf of rank one
    gives each class's values."""
    if leaf.ndim < 2:
        return {"heavy": leaf[mask], "light": leaf[~mask]}
    rows, chosen = leaf.reshape(-1, leaf.shape[-1]), mask.reshape(-1, mask.shape[-1])
    classes = {}
    for name, held in (("heavy", chosen), ("light", ~chosen)):
        counts = held.sum(axis=-1)
        classes[name] = {int(count): rows[counts == count][held[counts == count]].reshape(-1, int(count))
                         for count in np.unique(counts[counts > 0])}
    return classes


def _judged_leaves(dew, weighing) -> tuple[list[str], tuple[int, ...]]:
    """The names and row widths of the leaves check 1 judges of Dew's arm `dew`: its own, or with a
    `weighing` each floating leaf's classes, each compacted by its values a row (`Weighed.classed`)."""
    judged = dew if weighing is None else weighing.classed(jax.device_get(dew))
    return ([jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(judged)],
            tuple(int(np.shape(leaf)[-1]) if np.ndim(leaf) >= 2 else 0 for leaf in jax.tree.leaves(judged)))


def _drawn_from(names: list[str], widths: tuple[int, ...], evidence: list[list[tuple]], examples=None,
               example_rows=None) -> _Draws:
    """A site's draws (`_Draws`) from each seed's evidence (`_evidence`, one entry a leaf), the leaves'
    `names` and row `widths` (`_judged_leaves`), and where its leaves stack examples, each seed's
    examples' ratios and worst rows (`_example_ratios`, `_example_worst_rows`)."""
    count = len(evidence)

    def each(at: int) -> list[list]:
        return [[entry[at] for entry in row] for row in evidence]

    worst = np.asarray([[(entry[2], entry[3]) for entry in row] for row in evidence]).reshape(count, -1, 2)
    return _Draws(names, np.asarray(each(0)), np.asarray(each(1)), worst, each(4), np.asarray(each(5)),
                  widths, examples, example_rows, _shares(each(6), count))


SIZING_FROM = 1 << 20
"""The first seed a paired sizing draws (`seeds_for` with `dew_fn`): past every site's verdict seeds
and below every null run's (tools/gate_null_runs.py, from 2^16 runs of at least 80 seeds on), so the
count a site declares is fixed by draws its verdict never sees."""


MOST_SEEDS = 512
"""The most seeds a site's sizing declares (`seeds_for`, `seeds_resolving`): a site that needs more
takes larger inputs or fewer shared factors, not more seeds."""


def _drawn_over(draw, examples: bool, conditioning: Conditioning | None = None):
    """`drawn(start, stop)` over `draw(seed)`, each seed's three trees alike, ours, the reference's and
    the float64 truth's: the site's draws (`_drawn_from`) of seeds start to stop, its leaves named by the
    first draw's. With `conditioning`, each seed's draw adds its inputs, and both arms are weighed by the
    reference's rounding there (`Conditioning.of`), their leaves split into their classes."""
    leaves: list = []

    def drawn(start: int, stop: int) -> _Draws:
        evidence, stacked, stacked_rows = [], [], []
        for seed in range(start, stop):
            if conditioning is None:
                (ours, reference, truth), weighing = draw(seed), None
            else:
                ours, reference, truth, inputs = draw(seed)
                weighing = conditioning.of(truth, inputs, seed)
            if not leaves:
                leaves.append(_judged_leaves(ours, weighing))
            if weighing is None:
                prepared = _prepared(ours, truth)
            else:
                prepared = _prepared(*weighing.weighed(ours, "Dew's arm"), weighing.ones)
                reference, truth = weighing.weighed(reference, "the reference")
            if examples:
                held = jax.device_get(reference)
                stacked.append(_example_ratios(prepared.leaves, held, prepared.truths))
                stacked_rows.append(_example_worst_rows(prepared.leaves, held, prepared.truths))
            evidence.append(_evidence(prepared, reference, truth, shared=True))
        names, widths = leaves[0]
        return _drawn_from(names, widths, evidence, stacked if examples else None,
                          stacked_rows if examples else None)
    return drawn


def _enough_seeds(seeds: int, label: str) -> None:
    """Refuses a run of fewer than FEWEST_SEEDS seeds as Inconclusive, whatever their ratio: check 1 judges
    no fewer (`seeds_for` sizes a site from FEWEST_SEEDS on)."""
    if seeds < FEWEST_SEEDS:
        raise Inconclusive([Finding(label, "inconclusive", FEWEST_SEEDS / max(seeds, 1), f"{label}: {seeds} "
                                    f"seeds cannot be judged; check 1 takes at least {FEWEST_SEEDS} "
                                    f"(`seeds_for`)")])


def judge(draw, seeds: int, label: str, *, examples: bool = False,
          conditioning: Conditioning | None = None) -> None:
    """Check 1 over `draw(seed)`, each a draw's three trees of arrays alike, ours, the reference's and the
    float64 truth's (`_evidence`): its two stages (`_staged`, REDRAW times `seeds` further seeds drawn where
    a leaf does not pass at `seeds`), every leaf's pooled ratio, rows and values judged, and with `examples`
    each leaf's first axis its independent examples. With `conditioning`, the weighed check (`Weighed`):
    each draw adds its inputs, `(ours, reference, truth, inputs)`, and both arms are judged in units of the
    reference's rounding there, a caller's second judgement where the unweighted one stays Inconclusive.
    Raises GateFailure where a leaf is over, Inconclusive where one is left unresolved, as is a run of
    fewer than FEWEST_SEEDS seeds; returns where every leaf passes."""
    _enough_seeds(seeds, label)
    over, unresolved, _ = _staged(_drawn_over(draw, examples, conditioning), seeds, label)
    if over:
        raise GateFailure(over + unresolved)
    if unresolved:
        raise Inconclusive(unresolved)


def seeds_for(draw, label: str, *, start: int = FEWEST_SEEDS, most: int = MOST_SEEDS,
              examples: bool = False, conditioning: Conditioning | None = None) -> int:
    """The seeds `judge` declares over `draw`: ours paired with the reference on sizing seeds from
    SIZING_FROM on, which no verdict draws (`_sized_count`), weighed as `judge` is with `conditioning`."""
    drawn = _drawn_over(draw, examples, conditioning)
    return _sized_count(lambda first, stop: drawn(SIZING_FROM + first, SIZING_FROM + stop), label,
                        start=start, most=most, paired=True)


def _leaf_examples(examples: list, start: int, count: int, index: int) -> list[np.ndarray] | None:
    """Leaf `index`'s per-example log ratios over seeds `start` to `start + count` of a sizing's draws
    (`_Draws.examples`), or None where a seed's are not paired."""
    found = [seed[index] for seed in examples[start:start + count]]
    return None if any(ratios is None for ratios in found) else found


def _confirmed(names, theirs: np.ndarray, count: int, mine: np.ndarray | None = None,
               typical: tuple = (), examples: list | None = None, scalars: Sequence[bool] = ()) -> list[str]:
    """The leaves `seeds_for` leaves unresolved at `count`: on the count's own block of draws or on the
    next block of as many (`seeds_for`'s confirmation), each judged as the sizing judges it: paired,
    `mine` Dew's squared errors against the reference's (`theirs`, seed by leaf) on the same draws, or
    the reference compared with itself (`_unresolved_against_itself`, each block twice the count); the
    leaves named in `typical` sized by their typical draw too (`_sized`), `scalars` saying which leaves hold
    one value an example."""
    wide: list[str] = []
    for start in (0, count) if mine is not None else (0, 2 * count):
        if mine is not None:
            held, ours = theirs[start:start + count], mine[start:start + count]
            unresolved = [name for index, name in enumerate(names) if np.sum(held[:, index]) > 0
                          and not _sized(ours[:, index], held[:, index], name in typical,
                                         None if examples is None else _leaf_examples(examples, start, count,
                                                                                      index),
                                         bool(scalars) and scalars[index])]
        else:
            unresolved = _unresolved_against_itself(theirs[start:start + 2 * count], names, count)
        wide += [name for name in unresolved if name not in wide]
    return wide


def seeds_resolving(errors: np.ndarray, names, label: str, *, start: int = FEWEST_SEEDS,
                    most: int = MOST_SEEDS) -> int:
    """`seeds_for`'s count for the reference compared with itself, from its errors already measured:
    `errors` holds its squared errors a seed (rows, from seed 0) and a leaf (columns), such as a
    recording's merged shards, and the count is the fewest, doubling from `start` to `most`, that
    resolves every leaf on its own block and the next (`_confirmed`), as `seeds_for` doubles. Fails past
    `most`, as `seeds_for` does, and where the rows held run out first: the recording then holds too
    few seeds."""
    count, wide = max(start, FEWEST_SEEDS), []
    while count <= most and 4 * count <= len(errors):
        wide = _confirmed(names, errors, count)
        if not wide:
            return count
        count *= 2
    if count > most:
        raise _unresolved(label, count / most, f"{label}: the reference compared with itself does not "
                          f"resolve {PARITY} within {most} seeds at {wide[:4]}; give the site larger inputs "
                          f"or fewer shared factors")
    raise _unresolved(label, 4 * count / max(len(errors), 1), f"{label}: the reference compared with "
                      f"itself does not resolve {PARITY} within the {len(errors)} seeds recorded, "
                      f"{len(errors) // 4} a half of each block")


def _unresolved(label: str, measure: float, message: str) -> Inconclusive:
    """A sizing that resolves no count it may declare, as Inconclusive: its seeds leave the site undecided,
    `measure` the count it reached over the most it may take."""
    return Inconclusive([Finding(label, "inconclusive", measure, message)])


def _sized_count(drawn, label: str, *, start: int, most: int, paired: bool) -> int:
    """`seeds_for`'s count over `drawn(start, stop)`, the draws of sizing seeds start to stop (`_Draws`):
    doubling from `start`, the first whose draws resolve every leaf on its own block and the next
    (`_confirmed`), paired (Dew's errors against the reference's) or the reference compared with itself
    (each block twice the count); where paired, the leaves the pooled ratio leaves unresolved within `most`
    sized by their typical draw on the draws held. Fails past `most`."""
    count = max(start, FEWEST_SEEDS)
    if count > most:
        raise AssertionError(f"{label}: sizing starts at {count} seeds, past the {most} it may take; "
                             f"give the site more pilots or start lower")

    def through(count: int) -> int:
        """The sizing seeds a count's draws run to: its block and the confirming one, each twice the count
        for the reference compared with itself."""
        return 2 * (count if paired else 2 * count)

    held = drawn(0, through(count))
    while True:
        wide = _confirmed(held.names, held.theirs, count, held.mine if paired else None)
        if not wide:
            return count
        count *= 2
        if count > most:
            break
        held = held.joined(drawn(len(held.theirs), through(count)))
    if paired:
        # The leaves the pooled ratio does not size within `most`, its draws ruled by their largest (each arm
        # rounding apart, ledger 2469), sized by their typical draw on the draws already held, every other
        # leaf by its pooled ratio as before: the first count at which each is.
        stubborn, count = tuple(wide), max(start, FEWEST_SEEDS)
        while count <= most:
            wide = _confirmed(held.names, held.theirs, count, held.mine, typical=stubborn,
                              examples=held.examples,
                              scalars=[held.scalar(index) for index in range(len(held.names))])
            if not wide:
                return count
            count *= 2
    compared = "Dew paired with the reference" if paired else "the reference compared with itself"
    raise _unresolved(label, count / most, f"{label}: {compared} does not resolve {PARITY} within {most} "
                      f"seeds at {wide[:4]}; give the site larger inputs or fewer shared factors")
