"""The float64 accuracy rule (tests/reference_error.py) on comparisons whose answer is known: what it
must refuse, and the rounding it must admit."""

from pathlib import Path

import numpy as np
import pytest
from reference_error import (
    FACTOR,
    ORDERS,
    ORDERS_DIR,
    ORDERS_REQUESTED,
    assert_as_exact_as_the_reference,
    assert_computes_the_oracle,
    assert_elements_as_exact_over_orders,
    assert_leaves_as_exact,
    assert_leaves_as_exact_in_two_stages,
    assert_rows_as_exact_over_orders,
    distance,
    element_factor,
    reduce,
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


def test_a_draw_within_the_factor_passes_and_an_unstable_one_is_held_over_orders(monkeypatch):
    """The two-stage rule: an equally exact port passes on order 0 alone; one whose order 0 meets a
    reference rounding by luck a tenth as far goes on to ORDERS draws and passes there; one rounding
    three times as far fails over the orders as it does on one draw."""
    monkeypatch.delenv(ORDERS_REQUESTED, raising=False)
    calls = []

    def counted(draw):
        def run(order):
            calls.append(order)
            return draw(order)
        return run

    assert_leaves_as_exact_in_two_stages(counted(drawn(1.0)), "equal", rows=False)
    assert calls == [0]
    calls.clear()
    assert_leaves_as_exact_in_two_stages(counted(drawn(1.0, unstable=True)), "lucky", rows=False)
    assert calls == list(range(ORDERS))
    with pytest.raises(AssertionError):
        assert_leaves_as_exact_in_two_stages(drawn(3.0, unstable=True), "rounds more", rows=False)


def test_orders_recorded_apart_reduce_to_the_rule_run_in_one_process(monkeypatch, tmp_path):
    """CI's two passes (`record_orders`, `reduce`): order 0 recorded alone names the case for its other
    orders where it is over its factor; all ORDERS recorded by items run apart hold it as one process
    would, and a missing order fails it."""
    monkeypatch.setenv(ORDERS_DIR, str(tmp_path))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_x.py::test_case (call)")
    monkeypatch.setenv(ORDERS_REQUESTED, "0")
    assert_leaves_as_exact_in_two_stages(drawn(1.0, unstable=True), "lucky", rows=False)
    assert reduce(tmp_path, stage1=True) == ["tests/test_x.py::test_case"]
    for group in range(1, ORDERS, 13):
        monkeypatch.setenv(ORDERS_REQUESTED, ",".join(map(str, range(group, min(group + 13, ORDERS)))))
        assert_leaves_as_exact_in_two_stages(drawn(1.0, unstable=True), "lucky", rows=False)
    assert reduce(tmp_path, stage1=False) == []
    next(tmp_path.glob("*/7.json")).unlink()
    assert reduce(tmp_path, stage1=False)


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


def test_elements_held_over_orders_pass_alike_and_fail_one_that_rounds_more():
    """Elements over ORDERS orders (`assert_elements_as_exact_over_orders`), each held to the
    reference's own error at it: 64 rows of 4096 elements rounding alike pass, as do elements both
    compute exactly; an element Dew rounds where the reference is exact in every order fails, as does
    one whose error the reference has elsewhere in its row, one rounding five times as far and a
    non-finite one. At 43.5M elements the derived factor is 3.16."""
    rng = np.random.default_rng(6)
    shape, spread = (64, 4096), 3e-6

    def squares():
        return np.sum(np.square(spread * rng.normal(size=(ORDERS, *shape))), axis=0)

    dew, reference = squares(), squares()
    assert_elements_as_exact_over_orders(dew, reference, "alike")
    both, exact = dew.copy(), reference.copy()
    both[:, :8], exact[:, :8] = 0, 0
    assert_elements_as_exact_over_orders(both, exact, "both exact")
    with pytest.raises(AssertionError):
        assert_elements_as_exact_over_orders(dew, exact, "exact only in the reference")
    moved, misplaced = reference.copy(), dew.copy()
    moved[3, 0], moved[3, 1] = ORDERS * (30 * spread) ** 2, 0
    misplaced[3, 0], misplaced[3, 1] = 0, ORDERS * (30 * spread) ** 2
    with pytest.raises(AssertionError):
        assert_elements_as_exact_over_orders(misplaced, moved, "misplaced")
    wrong = dew.copy()
    wrong[5, 17] = ORDERS * (5 * 4 * spread) ** 2
    with pytest.raises(AssertionError):
        assert_elements_as_exact_over_orders(wrong, reference, "one element")
    broken = dew.copy()
    broken[0, 0] = np.nan
    with pytest.raises(AssertionError):
        assert_elements_as_exact_over_orders(broken, reference, "nan")
    assert round(element_factor(332 * 131072), 2) == 3.16


def test_the_rule_reduces_a_directory_of_recorded_orders_from_its_command_line(tmp_path):
    """`python tests/reference_error.py reduce DIR` reads a directory of recorded orders and exits
    zero where it holds none to fail."""
    import subprocess
    import sys

    found = subprocess.run([sys.executable, str(Path(__file__).parent / "reference_error.py"), "reduce",
                            str(tmp_path)], capture_output=True, text=True, check=False, timeout=120)
    assert found.returncode == 0, found.stderr


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
    with pytest.raises(AssertionError, match="entries"):
        assert_computes_the_oracle(wrong, truth, "one entry", roundings=64)
