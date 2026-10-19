"""The float64 accuracy rule (tests/reference_error.py) on comparisons whose answer is known: what it
must refuse, and the rounding it must admit."""

import jax
import numpy as np
import pytest
from reference_error import (
    FACTOR,
    ORDERS,
    ROW_FACTOR,
    STAGES,
    assert_as_exact_as_the_reference,
    assert_computes_the_oracle,
    assert_elements_as_exact,
    assert_leaves_as_exact,
    assert_leaves_as_exact_over_orders,
    assert_rows_as_exact_over_orders,
    distance,
)


def test_a_small_row_beside_large_ones_is_held_to_its_own_size():
    """Dew doubles a row of 5s beside rows of 1e8 that the reference rounds by 8. Every
    tensor-wide distance passes it: its RMS error, 3.5, sits under twice the reference's, 5.7.
    Held to its own size, the row is 100% wrong against the reference's 8e-8."""
    truth = np.array([[1e8, 1e8], [5.0, 5.0]])
    reference = np.array([[1e8 + 8, 1e8 + 8], [5.0, 5.0]])
    dew = np.array([[1e8, 1e8], [10.0, 10.0]])
    assert distance(dew, truth) <= FACTOR * distance(reference, truth)
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(dew, reference, truth, "small row")
    with pytest.raises(AssertionError):
        assert_leaves_as_exact({"rows": dew}, {"rows": reference}, {"rows": truth})


def test_a_small_element_is_held_to_its_terms_in_a_column_or_a_vector():
    """The small row transposed, a column of 5s beside columns of 1e8, puts each wrong element in
    a row whose RMS it barely moves; a vector is one row. Its terms (`float64_terms`) say the 5s
    sum terms of their own size, so held to them each element is 100% wrong against the
    reference's 8e-8, though every tensor-wide and row-wide distance passes it."""
    truth = np.array([[1e8, 5.0], [1e8, 5.0]])
    reference = np.array([[1e8 + 8, 5.0], [1e8 + 8, 5.0]])
    dew = np.array([[1e8, 10.0], [1e8, 10.0]])
    for value, wrong, exact in ((truth, dew, reference), (truth[0], dew[0], reference[0])):
        assert distance(wrong, value) <= FACTOR * distance(exact, value)
        with pytest.raises(AssertionError):
            assert_as_exact_as_the_reference(wrong, exact, value, "small element", terms=np.abs(value))
        with pytest.raises(AssertionError):
            assert_leaves_as_exact({"elements": wrong}, {"elements": exact}, {"elements": value},
                                   terms={"elements": np.abs(value)})


def test_an_element_near_zero_without_terms_is_held_to_its_row():
    """Sums of terms of either sign land near zero somewhere in a large tensor, and there one
    rounding against another parts by any ratio: two runs rounding alike, each by a float32 unit
    of what the element sums, put their worst element over its own value apart by more than four
    times in 6.2% of comparisons (2000 draws of these sums), and in none sized by the row. With no
    terms to size it, an element is sized by its row, and two such runs pass, over 200 draws."""
    rng = np.random.default_rng(2)
    unit = float(np.finfo(np.float32).eps)
    for _ in range(200):
        terms = rng.normal(size=(64, 32, 16))
        truth = terms.sum(axis=1)
        spread = np.sqrt(32) * unit
        dew, reference = (truth + spread * rng.normal(size=truth.shape) for _ in range(2))
        assert_as_exact_as_the_reference(dew, reference, truth, "near zero")


def test_a_row_that_cancels_is_held_to_what_it_sums():
    """A query that keeps one key has a zero gradient: its terms, about 1 each, cancel. Both
    sides round them by about a float32 unit, which over the row's near-zero float64 value is a
    ratio near 1e10 and over its terms (`float64_terms`) one unit, so Dew rounding as the
    reference does passes. Dew ten times further in that row alone moves the RMS over 512 rows
    by 9%, which the tensor-wide rule admits; the row, held to its terms, does not pass."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(512, 64))
    truth[7] = 1e-17
    terms = np.abs(truth)
    terms[7] = 1.0
    unit = float(np.finfo(np.float32).eps)

    def rounded(cancelling=1.0):
        noise = unit * terms * rng.choice([-1.0, 1.0], size=truth.shape)
        noise[7] *= cancelling
        return truth + noise

    reference, further = rounded(), rounded(10.0)
    assert_leaves_as_exact({"dq": rounded()}, {"dq": reference}, {"dq": truth}, terms={"dq": terms})
    assert distance(further, truth) <= FACTOR * distance(reference, truth)
    with pytest.raises(AssertionError):
        assert_leaves_as_exact({"dq": further}, {"dq": reference}, {"dq": truth}, terms={"dq": terms})


def test_a_row_of_float64_zero_is_held_to_the_references_own_row():
    """A sum that cancels exactly before anything its terms see, as a query that keeps one key
    has a zero gradient through the flash formula's delta, has no size of its own: its distance
    is held to the reference's on that same row. Dew at the reference's residue passes; Dew with a
    residue where the reference computes the float64 zero exactly does not, though the tensor-wide
    RMS admits it."""
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(64, 16))
    truth[0] = 0.0
    unit = float(np.finfo(np.float32).eps)
    reference = truth + unit * rng.normal(size=truth.shape)
    dew = truth + unit * rng.normal(size=truth.shape)
    assert_as_exact_as_the_reference(dew, reference, truth, "zero row")
    reference[0], dew[0] = 0.0, unit
    assert distance(dew, truth) <= FACTOR * distance(reference, truth)
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(dew, reference, truth, "zero row")


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
        return {"leaf": dew}, {"leaf": reference}, {"leaf": truth}, None
    return draw


def counted(draw, calls: list):
    """`draw`, each order it is asked for appended to `calls`."""
    def run(order):
        calls.append(order)
        return draw(order)
    return run


def test_a_comparison_is_held_over_a_whole_batch_of_orders_and_one_rounding_more_fails():
    """The K-order gate (`assert_leaves_as_exact_over_orders`) takes ORDERS orders before it passes
    anything, as one draw within its factors is no evidence that every element is: an equally
    exact port passes on them, as does one whose order 0 meets a reference rounding by luck a tenth
    as far; one rounding three times as far fails."""
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
        return {"leaf": truth + error}, {"leaf": reference}, {"leaf": truth}, None

    (dew,), (reference,), (truth,), _ = (jax.tree.leaves(tree) for tree in draw(0))
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
        return dew[stage], reference[stage]
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
    assert_elements_as_exact(lambda stage: (sums[0][stage], sums[1][stage]), tails)


def test_the_oracle_check_finds_a_wrong_small_entry_its_rms_dilutes():
    """A float64 twin of 8 rows of 131072 logits a few ulps from the oracle passes, a row the oracle
    computes as exact zeros a few ulps of the logits' scale off them too; one entry of size 1e-3
    wrong by 1e-11, which the global RMS bound passes (it moves the twin's RMS by 1e-14 of its
    scale), fails entry by entry (`assert_computes_the_oracle`)."""
    rng = np.random.default_rng(8)
    truth = rng.standard_normal((8, 131072))
    eps = float(np.finfo(np.float64).eps)
    twin = truth * (1 + 4 * eps * rng.standard_normal(truth.shape))
    assert_computes_the_oracle(twin, truth, "a few ulps", roundings=64)
    truth[2, 5] = 1e-3
    truth[3] = 0
    twin = truth * (1 + 4 * eps * rng.standard_normal(truth.shape))
    twin[3] = 4 * eps * rng.standard_normal(truth.shape[1])
    assert_computes_the_oracle(twin, truth, "a cancelled row", roundings=64)
    wrong = twin.copy()
    wrong[2, 5] = truth[2, 5] + 1e-11
    # The global RMS bound alone passes it.
    assert distance(wrong, truth) <= FACTOR * 64 * eps * float(np.sqrt(np.mean(np.square(truth))))
    with pytest.raises(AssertionError):
        assert_computes_the_oracle(wrong, truth, "one entry", roundings=64)
