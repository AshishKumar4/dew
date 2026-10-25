"""The sampling gate (tests/sampling_gate.py): a fair sampler passes, a favoured outcome fails, and a site
that cannot be judged is refused."""

import numpy as np
import pytest
from sampling_gate import SamplingFailure, assert_draws_as_declared

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
    with pytest.raises(SamplingFailure, match=r"0\.07"):
        assert_draws_as_declared(drawn((0.83, 0.07, 0.1)), SHARES, "favoured", seeds=64)


def test_a_site_too_small_to_judge_is_refused():
    """Eight records a seed over two seeds expect less than one of the rarest part: refused, not
    judged; and counts of another shape, or chances that do not sum to one, are refused too."""
    with pytest.raises(AssertionError, match="more seeds") as raised:
        assert_draws_as_declared(drawn(SHARES, 8), SHARES, "small", seeds=2)
    assert raised.type is AssertionError
    with pytest.raises(ValueError, match="nonnegative integers"):
        assert_draws_as_declared(lambda seed: np.array([1.5, 2, 3]), SHARES, "floats", seeds=1)
    with pytest.raises(ValueError, match="sum to one"):
        assert_draws_as_declared(drawn(SHARES), (0.8, 0.05, 0.1), "chances", seeds=1)
