"""The sampling gate: whether a sampler draws each outcome at its declared chance, as a fair draw
would, over drawn seeds (`assert_draws_as_declared`). Check 1 holds a value against float64 truth; a
sampler's claim is of another kind, a share it promises on average, which no float64 run computes:
held to an independent draw by check 1, a share favoured by 0.02 of 500 records passed 19 verdicts in
20 (ledger 1305). This gate tests the share itself.

Each seed's draws are counted by outcome, the counts pooled over the seeds, and the pooled counts
held to the declared chances by Pearson's chi-square test at FALSE_RED, the chance a fair sampler
fails: a share off its chance by more than its draws' spread fails, the more surely the more draws
there are. The seeds are a sampler's independent runs (a holdout's corpora, a mixture's streams), so
their counts add as one larger draw's. A design note stays in the head that adds it (ledger 1305)."""

import math

import numpy as np
from scipy import stats

FALSE_RED = 1e-4
"""The chance a sampler that draws exactly its declared chances fails the gate: one verdict in ten
thousand, so a suite of a hundred such sites goes red on a fair sampler once in a hundred runs."""

FEWEST_EXPECTED = 5
"""The least count any outcome is expected to receive over all the seeds: below it the chi-square law
the verdict reads is not the statistic's, so the gate refuses rather than judges."""


class SamplingFailure(AssertionError):
    """A sampler the gate finds drawing its outcomes off their declared chances."""


def assert_draws_as_declared(counts_of, chances, label: str, *, seeds: int) -> None:
    """`counts_of(seed)`, the count of each outcome over one seed's draws (a part's records, a member's
    rows), drawn at `chances` (each a share of their sum) as a fair sampler would, over `seeds` seeds:
    Pearson's chi-square of the pooled counts against `chances` times their total passes at FALSE_RED.

    Raises SamplingFailure naming each outcome's share against its chance; refuses a site whose
    expected counts fall below FEWEST_EXPECTED or whose seeds draw nothing."""
    chances = np.asarray(chances, np.float64)
    if chances.ndim != 1 or np.any(chances <= 0):
        raise ValueError(f"{label}: chances must be positive, not {chances.tolist()}")
    # A split's chances as its shares of their exact sum: 0.85, 0.05 and 0.1 add to one only in decimal.
    chances = chances / math.fsum(chances.tolist())
    pooled = np.zeros(chances.size, np.int64)
    for seed in range(seeds):
        counts = np.asarray(counts_of(seed))
        if counts.shape != chances.shape or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError(f"{label}: seed {seed}'s counts must be {chances.size} nonnegative integers, "
                             f"not {counts.tolist()}")
        pooled += counts
    total = int(pooled.sum())
    expected = chances * total
    if total == 0 or expected.min() < FEWEST_EXPECTED:
        raise AssertionError(f"{label}: {total} draws over {seeds} seeds expect {expected.min():.1f} of the "
                             f"rarest outcome, fewer than {FEWEST_EXPECTED}; give the site more seeds")
    statistic, p_value = stats.chisquare(pooled, expected)
    if p_value < FALSE_RED:
        shares = ", ".join(f"{count / total:.4f} against {chance:.4f}"
                           for count, chance in zip(pooled, chances, strict=True))
        raise SamplingFailure(f"{label}: {total} draws over {seeds} seeds take shares {shares}: chi-square "
                              f"{statistic:.1f} on {chances.size - 1} degrees of freedom, p = {p_value:.2e} "
                              f"(a fair sampler fails at {FALSE_RED})")
