"""The float64 gate over orders (tests/reference_ensemble.py): a comparison held over a whole batch of
exact-symmetry orders, its elements against the reference's own error at each, flagged elements
confirmed over fresh batches, and its rows over the orders."""

import jax
import numpy as np
import pytest
from reference_ensemble import REFERENCE, STAGES, assert_elements_as_exact, assert_leaves_as_exact_over_orders
from reference_error import FACTOR, ORDERS, ROW_FACTOR, assert_rows_as_exact_over_orders, distance


def drawn(spread: float, rounding: float = 1.0, *, unstable: bool = False):
    """A comparison's draw by order: a float64 truth of 512 sums, the reference rounding each by
    `rounding` float32 units of what it sums and Dew by `spread` times that. Where `unstable`, order
    0's reference rounds a tenth as far, as one lucky draw of a sum's rounding does."""
    unit = float(np.finfo(np.float32).eps)

    def draw(order):
        rng = np.random.default_rng(order)
        truth = rng.normal(size=(16, 32))
        luck = 0.1 if unstable and order == 0 else 1.0
        reference = truth + luck * rounding * unit * rng.normal(size=truth.shape)
        dew = truth + spread * rounding * unit * rng.normal(size=truth.shape)
        return {"leaf": dew}, {REFERENCE: {"leaf": reference}}, {"leaf": truth}
    return draw


def counted(draw, calls: list):
    """`draw`, each order it is asked for appended to `calls`."""
    def run(order):
        calls.append(order)
        return draw(order)
    return run


def test_a_comparison_is_held_over_a_whole_batch_of_orders_and_one_rounding_more_fails():
    """The K-order gate (`assert_leaves_as_exact_over_orders`) takes ORDERS orders before it passes
    anything, as one draw within its factors is no evidence that every element is, and an ensemble of
    the reference alone draws no admission sample: an equally exact port passes on them, as does one
    whose order 0 meets a reference rounding by luck a tenth as far; one rounding three times as far
    fails."""
    calls = []
    assert_leaves_as_exact_over_orders(counted(drawn(1.0), calls), "equal", rows=False)
    assert calls == list(range(ORDERS))
    assert_leaves_as_exact_over_orders(drawn(1.0, unstable=True), "lucky", rows=False)
    with pytest.raises(AssertionError):
        assert_leaves_as_exact_over_orders(drawn(3.0), "rounds more", rows=False)


def test_a_localized_defect_that_the_leaf_and_row_rules_pass_fails_element_by_element():
    """A port 100 times the reference's rounding on 4 of a row's 4096 elements (0.1%) moves that
    row's RMS by sqrt(10.999), 3.3 times, inside ROW_FACTOR, and its leaf's by less, inside FACTOR;
    the elements are held one by one and fail at every stage."""
    unit = float(np.finfo(np.float32).eps)

    def draw(order):
        rng = np.random.default_rng(order)
        truth = rng.normal(size=(8, 4096))
        reference = truth + unit * rng.normal(size=truth.shape)
        error = unit * rng.normal(size=truth.shape)
        error[2, :4] *= 100
        return {"leaf": truth + error}, {REFERENCE: {"leaf": reference}}, {"leaf": truth}

    (dew,), (reference,), (truth,) = (jax.tree.leaves(tree) for tree in draw(0))
    assert distance(dew[2], truth[2]) < ROW_FACTOR * distance(reference[2], truth[2])
    calls = []
    with pytest.raises(AssertionError):
        assert_leaves_as_exact_over_orders(counted(draw, calls), "localized")
    assert calls == list(range(STAGES * ORDERS))


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


def staged(dew, reference):
    """A `batch` for `assert_elements_as_exact` from `[stages, ...]` sums, each stage asked for appended
    to the returned list."""
    asked = []

    def batch(stage):
        asked.append(stage)
        return dew[stage], {REFERENCE: reference[stage]}
    return batch, asked


def test_an_element_flagged_by_chance_clears_at_a_confirmation_and_one_rounding_more_does_not():
    """Each element's mean absolute error over ORDERS orders against FACTOR times the reference's
    there (`assert_elements_as_exact`): an element three times the reference's at stage 0 alone is
    re-measured over fresh orders and clears; one three times it at every stage fails, after
    STAGES; elements both compute exactly pass; one Dew rounds where the reference is exact in
    every order fails; a non-finite one fails at the stage it is measured, before any other."""
    rng = np.random.default_rng(6)
    dew, reference = (np.sum(np.abs(rng.standard_normal((STAGES, ORDERS, 64, 256))), axis=1)
                      for _ in range(2))
    batch, asked = staged(dew, reference)
    assert_elements_as_exact(batch, "alike")
    assert asked == [0]
    once = dew.copy()
    once[0, 3, 5] = 3 * reference[0, 3, 5]
    batch, asked = staged(once, reference)
    assert_elements_as_exact(batch, "once")
    assert asked == [0, 1]
    always = dew.copy()
    always[:, 3, 5] = 3 * reference[:, 3, 5]
    batch, asked = staged(always, reference)
    with pytest.raises(AssertionError):
        assert_elements_as_exact(batch, "always")
    assert asked == list(range(STAGES))
    exact, both = reference.copy(), dew.copy()
    exact[:, :, :8], both[:, :, :8] = 0, 0
    assert_elements_as_exact(staged(both, exact)[0], "both exact")
    with pytest.raises(AssertionError):
        assert_elements_as_exact(staged(dew, exact)[0], "exact only in the reference")
    broken = dew.copy()
    broken[0, 0, 0] = np.nan
    batch, asked = staged(broken, reference)
    with pytest.raises(AssertionError):
        assert_elements_as_exact(batch, "nan")
    assert asked == [0]


@pytest.mark.parametrize("tails", ["gaussian", "laplace", "student-t5"])
def test_two_ports_equally_exact_pass_element_by_element_on_tails_heavier_than_a_gaussians(tails):
    """Two runs as exact as each other over 200000 elements, their errors Gaussian, Laplace or
    Student t of 5 degrees (kurtosis 9, past Dory's median rows): stage 0 flags some by chance, a
    few hundred on the heaviest, and the confirmations clear them. A check of the mechanism, not the
    rule's calibration, which real order traces at the real element count set (STAGES)."""
    rng = np.random.default_rng(9)
    draw = {"gaussian": rng.standard_normal, "laplace": lambda size: rng.laplace(size=size),
            "student-t5": lambda size: rng.standard_t(5, size=size)}[tails]
    sums = [[np.sum(np.abs(draw((ORDERS, 200_000))), axis=0) for _ in range(STAGES)] for _ in range(2)]
    assert_elements_as_exact(lambda stage: (sums[0][stage], {REFERENCE: sums[1][stage]}), tails)


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
