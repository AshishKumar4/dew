"""Rows held over a batch of exact-symmetry orders (`reference_error.assert_rows_as_exact_over_orders`) and
the orders check 1's orbits draw (tests/residual_orders.py)."""

import numpy as np
import pytest
from reference_error import FACTOR, ORDERS, assert_rows_as_exact_over_orders


def test_a_row_rounding_more_fails_over_orders_where_the_global_rms_passes():
    """Rows over ORDERS orders (`assert_rows_as_exact_over_orders`): Dew rounding each of 64 rows as the
    reference does passes; one row rounding six times as far, which moves the RMS over every row by
    little, fails as its worst row."""
    rng = np.random.default_rng(5)
    reference = np.abs(rng.normal(size=(ORDERS, 64)))
    dew = np.abs(rng.normal(size=(ORDERS, 64)))
    assert_rows_as_exact_over_orders(dew, reference, "alike")
    dew[:, 3] *= 6
    assert np.sqrt(np.mean(dew ** 2)) <= FACTOR * np.sqrt(np.mean(reference ** 2))
    with pytest.raises(AssertionError):
        assert_rows_as_exact_over_orders(dew, reference, "one row")


def test_each_seed_draws_its_own_order_and_a_group_moves_whole():
    """Check 1's orbit draws: each seed its own order of the units, the same on every call, and with a
    group of 4 every block of 4 moves whole and is permuted within."""
    from residual_orders import order

    drawn = [order(64, seed) for seed in range(8)]
    assert all(sorted(each.tolist()) == list(range(64)) for each in drawn)
    assert len({each.tobytes() for each in drawn}) == 8
    assert np.array_equal(order(64, 3), drawn[3])
    with pytest.raises(ValueError):
        order(10, 0, group=4)
    grouped = order(64, 5, group=4)
    assert sorted(grouped.tolist()) == list(range(64))
    assert all(sorted((grouped[at:at + 4] // 4).tolist()) == [grouped[at] // 4] * 4 for at in range(0, 64, 4))
