"""The sampling gate: whether a sampler draws each outcome at its declared chance, as a fair draw
would, over drawn seeds (`assert_draws_as_declared`). Check 1 holds a value against float64 truth; a
sampler's claim is of another kind, a share it promises on average, which no float64 run computes:
held to an independent draw by check 1, a share favoured by 0.02 of 500 records passed 19 verdicts in
20 (ledger 1305). This gate tests the share itself.

Each seed's draws are counted by outcome and the counts pooled over the seeds (a sampler's independent
runs add as one larger draw). Pearson's statistic of the pooled counts against the declared chances is
held to its own law under a fair sampler, NULL_DRAWS multinomial draws at those chances and that total
(`null_statistics`), not to the chi-square law, which at sparse expected counts rejects a fair sampler
more often than its level: 100 draws at (0.95, 0.05) fail a fair sampler 4.6e-4 of the time by the
chi-square law at 1e-4. The site fails where the share of null statistics at least its own, counted as
(1 + exceedances) / (NULL_DRAWS + 1), is below FALSE_RED: a valid test at any counts, its false-red
rate FALSE_RED's at most whatever the draws' spread (tests/test_sampling_gate.py computes it exactly)."""

import functools
import math

import numpy as np

FALSE_RED = 1e-4
"""The chance a sampler that draws exactly its declared chances fails the gate: one verdict in ten
thousand, so a suite of a hundred such sites goes red on a fair sampler once in a hundred runs."""

FEWEST_EXPECTED = 5
"""The least count any outcome is expected to receive over all the seeds: a site whose rarest outcome
expects fewer can barely be tested at all, so the gate refuses it rather than pass it unexamined."""

NULL_DRAWS = 999_999
"""The fair sampler's draws a verdict's statistic is held to: (1 + exceedances) / (NULL_DRAWS + 1)
resolves FALSE_RED to a hundredth of itself."""

_CHUNK = 100_000
"""The null draws taken at a time, bounding a verdict's memory at many outcomes."""


class SamplingFailure(AssertionError):
    """A sampler the gate finds drawing its outcomes off their declared chances."""


def statistics(counts: np.ndarray, chances: np.ndarray) -> np.ndarray:
    """Pearson's statistic of each row of `counts` (one draw's count of each outcome a row) against
    `chances` times the row's total: the sum of each outcome's squared departure over its expected
    count, the same arithmetic for a site's counts and for the null's."""
    counts = np.asarray(counts, np.float64)
    expected = chances * counts.sum(axis=-1, keepdims=True)
    return np.sum((counts - expected) ** 2 / expected, axis=-1)


@functools.lru_cache(maxsize=32)
def null_statistics(total: int, chances: tuple[float, ...]) -> np.ndarray:
    """Pearson's statistics of NULL_DRAWS draws of `total` at `chances` by a fair sampler, sorted: the
    statistic's own law under the gate's null, the same for every site of that total and those chances,
    from a fixed seed so every verdict on the same counts is the same."""
    rng = np.random.default_rng([0x5A3D, total, *np.asarray(chances).view(np.uint64).tolist()])
    held = np.asarray(chances)
    drawn = [statistics(rng.multinomial(total, held, size=min(_CHUNK, NULL_DRAWS - start)), held)
             for start in range(0, NULL_DRAWS, _CHUNK)]
    return np.sort(np.concatenate(drawn))


def p_value(statistic: float, total: int, chances: tuple[float, ...]) -> float:
    """The share of the fair sampler's statistics at least `statistic`, as an exact Monte Carlo test
    counts it: (1 + exceedances) / (NULL_DRAWS + 1)."""
    null = null_statistics(total, chances)
    exceedances = null.size - np.searchsorted(null, statistic, side="left")
    return (1 + int(exceedances)) / (NULL_DRAWS + 1)


def assert_draws_as_declared(counts_of, chances, label: str, *, seeds: int) -> None:
    """`counts_of(seed)`, the count of each outcome over one seed's draws (a part's records, a member's
    rows), drawn at `chances` (each a share of their sum) as a fair sampler would, over `seeds` seeds:
    the pooled counts' statistic is no rarer than FALSE_RED under a fair sampler (`p_value`).

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
    if total == 0 or chances.min() * total < FEWEST_EXPECTED:
        raise AssertionError(f"{label}: {total} draws over {seeds} seeds expect {chances.min() * total:.1f} "
                             f"of the rarest outcome, fewer than {FEWEST_EXPECTED}; give the site more seeds")
    statistic = float(statistics(pooled[None], chances)[0])
    rarity = p_value(statistic, total, tuple(chances.tolist()))
    if rarity < FALSE_RED:
        shares = ", ".join(f"{count / total:.4f} against {chance:.4f}"
                           for count, chance in zip(pooled, chances, strict=True))
        raise SamplingFailure(f"{label}: {total} draws over {seeds} seeds take shares {shares}: Pearson's "
                              f"statistic {statistic:.1f}, as rare as {rarity:.1e} under a fair sampler "
                              f"(a fair sampler fails at {FALSE_RED})")
