"""Check 1's statistics (tests/reference_error.py's `assert_as_exact`): its factors and tails, its
findings, the pooled ratio of Dew's error to the reference's and its interval, a leaf's status, and the
tests of a site that stacks examples (its examples' paired ratios, a scalar leaf's known null, each
example's worst row against the reference's on the same example). Each a function of the errors a site's
draws recorded, apart from the arms and the draws that record them.
"""

import dataclasses
import math

import numpy as np

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
    error over the reference's for a ratio's rule (`log_ratio`, `log_sum`), the seeds a value is not
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


def shown(log: float) -> str:
    """A log quotient as a message shows it, held at float64's largest."""
    return f"{float(np.exp(np.minimum(log, LOG_LARGEST))):.3e}"


def _scaled_rms(rows) -> np.ndarray:
    """Each row's root mean square of finite values, the row scaled by its largest first, so no square
    overflows (row 1628): zero for a row of zeros."""
    rows = np.abs(np.asarray(rows, np.float64))
    largest = np.max(rows, axis=1, keepdims=True) if rows.shape[1] else np.zeros((len(rows), 1))
    scale = np.where(largest > 0, largest, 1.0)
    return scale[:, 0] * np.sqrt(np.mean(np.square(rows / scale), axis=1))


def log_sum(values) -> float:
    """The natural log of the sum of nonnegative values, scaled by their largest first, so no sum of
    finite values overflows on the way: LOG_ZERO for a sum of zeros."""
    values = np.asarray(values, np.float64)
    largest = float(np.max(values)) if values.size else 0.0
    if not largest > 0:
        return LOG_ZERO
    return math.log(largest) + math.log(float(np.sum(values / largest)))


def root_mean(values) -> float:
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


def _interval(mine: np.ndarray, theirs: np.ndarray | None = None, control: np.ndarray | None = None,
              tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL)) -> tuple[float, float]:
    """The pooled ratio's bootstrap interval over seeds, its lower and upper `tails` left out: Dew's and
    the reference's errors resampled by the same seeds, or, for `control`, the reference's errors on
    two disjoint seed sets resampled apart (the reference compared with itself, an equally exact
    pair)."""
    rng = np.random.default_rng(0)
    count = len(mine)
    picks = rng.integers(0, count, (BOOTSTRAP, count))
    if control is None:
        ratios = np.sqrt(mine[picks].sum(axis=1) / np.maximum(theirs[picks].sum(axis=1), 1e-300))
    else:
        other = rng.integers(0, len(control), (BOOTSTRAP, len(control)))
        ratios = np.sqrt(mine[picks].sum(axis=1) / np.maximum(control[other].sum(axis=1), 1e-300))
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
3684). Derived: a row whose largest element e carries a share s of its squared error has its RMS error
within sqrt(1/s) of |e|'s own over the row's width, so at s past 0.9 the two rows' ratio is the two
elements' within a factor of at most 1.054 either way, inside the width-one factor's headroom over the
quantile it was calibrated at (90 against 40 to 53, NARROW_FACTORS). Such a row (an input amplified through
2^k, k up to 24, the weighed check's case) parts from its reference as one value's error does, a ratio of
two roundings, which ROW_FACTOR failed on equally exact pairs in up to 5% of runs at one row a draw (1 row of
16 values, 8 seeds: 154 of 4,000 runs with k apart in each arm, 35 with one k for both). Dew's worst row
must be ruled too, so a row of Dew's spread over its values (a row fault) keeps its width's factor beside a
reference one value rules. A Gaussian row of eight values or more is almost never ruled; rows of three or
four values sometimes are, where a fault ruled alike is refused less often. A value's own error in a ruled
row is check 2's to bound. A stacked-examples site's paired rows keep their width's factor (`_judgement`),
and a recording's summary, which keeps no row's elements, keeps it too."""


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
            and not _resolves(*_interval(halves[0][:, index], control=halves[1][:, index]))]


def _verdict(over: list[Finding], unresolved: list[Finding]) -> str:
    """A judgement's verdict: "over" where any leaf failed, else "inconclusive" where any is unresolved,
    else "pass"."""
    return "over" if over else "inconclusive" if unresolved else "pass"
