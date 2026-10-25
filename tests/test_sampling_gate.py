"""The sampling gate (tests/sampling_gate.py): a fair sampler passes, a favoured outcome fails, and a site
that cannot be judged is refused."""

import itertools

import numpy as np
import pytest
from sampling_gate import FALSE_RED, SamplingFailure, assert_draws_as_declared, p_value, statistics
from scipy import stats

SHARES = (0.85, 0.05, 0.1)


def drawn(shares, records: int = 500):
    """A seed's counts of `records` drawn independently at `shares`."""
    return lambda seed: np.random.default_rng([0x5A3, seed]).multinomial(records, shares)


def test_a_fair_sampler_passes():
    """Records drawn at the declared shares pass over 64 seeds of 500."""
    assert_draws_as_declared(drawn(SHARES), SHARES, "fair", seeds=64)


def test_a_share_favoured_by_two_hundredths_fails():
    """A part drawn at 0.07 where 0.05 is declared, 0.02 of the records taken from the first part,
    fails over 64 seeds of 500: the bias a hand bound of 0.02 at 4,000 records refused."""
    with pytest.raises(SamplingFailure):
        assert_draws_as_declared(drawn((0.83, 0.07, 0.1)), SHARES, "favoured", seeds=64)


def test_a_site_too_small_to_judge_is_refused():
    """Eight records a seed over two seeds expect less than one of the rarest part: refused, not
    judged; and counts of another shape, or a chance that is not positive, are refused too."""
    with pytest.raises(AssertionError) as raised:
        assert_draws_as_declared(drawn(SHARES, 8), SHARES, "small", seeds=2)
    assert raised.type is AssertionError
    with pytest.raises(ValueError):
        assert_draws_as_declared(lambda seed: np.array([1.5, 2, 3]), SHARES, "floats", seeds=1)
    with pytest.raises(ValueError):
        assert_draws_as_declared(drawn(SHARES), (0.95, 0.05, 0.0), "chances", seeds=1)


@pytest.mark.parametrize("total, chances", [(100, (0.95, 0.05)), (100, (0.9, 0.05, 0.05)),
                                            (100, (0.85, 0.05, 0.1)), (40, (0.5, 0.25, 0.25))])
def test_a_fair_sampler_fails_at_most_false_red_even_at_sparse_counts(total, chances):
    """The gate's false-red rate computed exactly at sparse sites, the rarest outcome expecting five
    draws or a few more: every composition of `total` into the outcomes, its chance under a fair
    sampler, and whether the gate fails it; the chances of those it fails sum to at most FALSE_RED.
    The chi-square law's own region at 100 draws at (0.95, 0.05) sums to 4.6e-4."""
    chances = np.asarray(chances)
    compositions = np.asarray([(*head, total - sum(head))
                               for head in itertools.product(range(total + 1), repeat=chances.size - 1)
                               if sum(head) <= total])
    statistic = statistics(compositions, chances)
    failed = np.asarray([p_value(value, total, tuple(chances.tolist())) < FALSE_RED for value in statistic])
    rate = stats.multinomial.pmf(compositions[failed], total, chances).sum() if failed.any() else 0.0
    assert failed.any() and rate <= FALSE_RED
