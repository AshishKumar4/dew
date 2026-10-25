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
(1 + exceedances) / (NULL_DRAWS + 1), is below LEVEL, half of FALSE_RED: a valid test at any counts in
expectation over its null draws, and at the fixed draws a verdict uses, its false-red rate under FALSE_RED
by about seven of its deviations (tests/test_sampling_gate.py computes it exactly at four sparse sites).

A site whose draws are several independent categoricals at once (a token's mask at each of three times,
each position's token) declares their chances as rows and counts each row's outcomes: its statistic is
the rows' statistics summed, held to the same null drawn row by row at each row's own total, one verdict
over all of them, so a site of many rows is one more chance of a false red, not one a row."""

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

LEVEL = FALSE_RED / 2
"""The rarity below which a site fails. A test held to null draws from a fixed seed fails a fair
sampler at that seed's own rate, which spreads about its level by the draws' count: at FALSE_RED
itself, (0.5, 0.25, 0.25) over 40 draws failed 1.004e-4 of the time, computed exactly. At half of it
the rate's spread (about 7e-6 at a million draws) leaves FALSE_RED about seven of its deviations
away: an overshoot that unlikely, not impossible, and computed absent at the sparse sites tested."""

_CHUNK = 4_000_000
"""The counts a batch of null draws holds (draws times rows times outcomes), bounding a verdict's memory
at many outcomes."""


class SamplingFailure(AssertionError):
    """A sampler the gate finds drawing its outcomes off their declared chances."""


def statistics(counts: np.ndarray, chances: np.ndarray) -> np.ndarray:
    """Pearson's statistic of `counts` against `chances`: one categorical's outcomes on `chances`' last
    axis, independent categoricals (rows) on the axis before it where `chances` has one, and draws on any
    axes before those. Each outcome's squared departure from its row's total times its chance, over that
    expected count, summed over the outcomes and the rows: the same arithmetic for a site's counts and for
    the null's."""
    counts = np.asarray(counts, np.float64)
    expected = chances * counts.sum(axis=-1, keepdims=True)
    return np.sum((counts - expected) ** 2 / expected, axis=tuple(range(-chances.ndim, 0)))


def _rows(totals, chances) -> tuple[np.ndarray, np.ndarray]:
    """`totals` and `chances` as rows: one categorical a row of each, a single one as one row."""
    held = np.asarray(chances, np.float64)
    return np.atleast_1d(np.asarray(totals, np.int64)), held.reshape(-1, held.shape[-1])


@functools.lru_cache(maxsize=32)
def null_statistics(totals, chances) -> np.ndarray:
    """Pearson's statistics of NULL_DRAWS draws by a fair sampler, each row's `totals` drawn at its
    `chances` (`totals` an int and `chances` a tuple for one categorical, a tuple of each for rows),
    sorted: the statistic's own law under the gate's null, the same for every site of those totals and
    chances, from a fixed seed so every verdict on the same counts is the same."""
    counts, held = _rows(totals, chances)
    rng = np.random.default_rng([0x5A3D, *counts.tolist(), *held.ravel().view(np.uint64).tolist()])
    batch = max(1, _CHUNK // held.size)
    drawn = [statistics(rng.multinomial(counts, held, size=(min(batch, NULL_DRAWS - start), counts.size)),
                        held)
             for start in range(0, NULL_DRAWS, batch)]
    return np.sort(np.concatenate(drawn))


def p_value(statistic: float, totals, chances) -> float:
    """The share of the fair sampler's statistics at least `statistic`, as an exact Monte Carlo test
    counts it: (1 + exceedances) / (NULL_DRAWS + 1)."""
    null = null_statistics(totals, chances)
    exceedances = null.size - np.searchsorted(null, statistic, side="left")
    return (1 + int(exceedances)) / (NULL_DRAWS + 1)


def assert_draws_as_declared(counts_of, chances, label: str, *, seeds: int) -> None:
    """`counts_of(seed)`, the count of each outcome over one seed's draws (a part's records, a member's
    rows), drawn at `chances` (probabilities summing to one) as a fair sampler would, over `seeds` seeds:
    the pooled counts' statistic is no rarer than LEVEL under a fair sampler (`p_value`). Chances of two
    axes declare independent categoricals, a row of outcomes each (a token's mask at each of three times,
    each position's draw), and `counts_of` returns their counts in the same shape: one verdict over all of
    them, their statistics summed and their null drawn row by row.

    Raises SamplingFailure naming each outcome's share against its chance; refuses a site whose
    expected counts fall below FEWEST_EXPECTED or whose seeds draw nothing."""
    chances = np.asarray(chances, np.float64)
    if chances.ndim not in (1, 2) or not np.all(np.isfinite(chances)) or np.any(chances <= 0):
        raise ValueError(f"{label}: chances must be finite and positive, a row of outcomes or rows of them, "
                         f"not {chances.tolist()}")
    rows = chances.reshape(-1, chances.shape[-1])
    # Chances are probabilities: a declaration whose sum is off one by more than its decimals' rounding
    # is refused, never repaired (0.85, 0.05 and 0.1 add to one in decimal, to 1 + 2^-52 in binary).
    sums = np.asarray([math.fsum(row) for row in rows.tolist()])
    if np.any(np.abs(sums - 1.0) > rows.shape[-1] * 2.0 ** -52):
        raise ValueError(f"{label}: each row of chances must sum to one, not {sums.tolist()}")
    rows = rows / sums[:, None]
    pooled = np.zeros(rows.shape, np.int64)
    for seed in range(seeds):
        counts = np.asarray(counts_of(seed))
        if counts.shape != chances.shape or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError(f"{label}: seed {seed}'s counts must be nonnegative integers shaped "
                             f"{chances.shape}, not {counts.tolist()}")
        pooled += counts.reshape(rows.shape)
    totals = pooled.sum(axis=-1)
    rarest = float(np.min(rows * totals[:, None]))
    if np.any(totals == 0) or rarest < FEWEST_EXPECTED:
        raise AssertionError(f"{label}: {totals.tolist()} draws over {seeds} seeds expect {rarest:.1f} of "
                             f"the rarest outcome, fewer than {FEWEST_EXPECTED}; give the site more seeds")
    statistic = float(statistics(pooled[None], rows)[0])
    rarity = p_value(statistic, tuple(totals.tolist()), tuple(map(tuple, rows.tolist())))
    if rarity < LEVEL:
        shares = "; ".join(", ".join(f"{count / total:.4f} against {chance:.4f}"
                                     for count, chance in zip(row, chance_row, strict=True))
                           for row, chance_row, total in zip(pooled, rows, totals, strict=True))
        raise SamplingFailure(f"{label}: {totals.tolist()} draws over {seeds} seeds take shares {shares}: "
                              f"Pearson's statistic {statistic:.1f}, as rare as {rarity:.1e} under a fair "
                              f"sampler (the gate fails below {LEVEL}, a fair sampler at most {FALSE_RED})")
