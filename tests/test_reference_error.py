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
    error_spread,
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


def _summed_over_orders(errors):
    """Each element's absolute and squared errors summed over its orders, `errors` `[ORDERS, ...]`."""
    return np.sum(np.abs(errors), axis=0), np.sum(np.square(errors), axis=0)


@pytest.mark.parametrize("tails", ["gaussian", "laplace", "student-t5"])
def test_elements_held_over_orders_pass_alike_and_fail_one_that_rounds_more(tails):
    """Elements over ORDERS orders (`assert_elements_as_exact_over_orders`), each held to the
    reference's own mean absolute error at it, for errors of a Gaussian's tails and heavier ones, as
    a bf16 rounding's flips give: 64 rows of 4096 elements rounding alike pass, as do elements both
    compute exactly; an element Dew rounds where the reference is exact in every order fails, as do
    one whose error the reference has elsewhere in its row, one rounding twice its row's derived
    factor as far, and a non-finite one."""
    rng = np.random.default_rng(6)
    shape = (64, 4096)
    draw = {"gaussian": lambda: rng.standard_normal((ORDERS, *shape)),
            "laplace": lambda: rng.laplace(size=(ORDERS, *shape)),
            "student-t5": lambda: rng.standard_t(5, size=(ORDERS, *shape))}[tails]
    spread = 3e-6
    dew, reference = (_summed_over_orders(spread * draw()) for _ in range(2))

    def held(dew, reference, label):
        assert_elements_as_exact_over_orders(dew[0], reference[0], label, dew_squares=dew[1],
                                             reference_squares=reference[1])

    held(dew, reference, "alike")
    both, exact = (tuple(side.copy() for side in sums) for sums in (dew, reference))
    for side in (*both, *exact):
        side[:, :8] = 0
    held(both, exact, "both exact")
    with pytest.raises(AssertionError):
        held(dew, exact, "exact only in the reference")
    moved, misplaced = (tuple(side.copy() for side in sums) for sums in (reference, dew))
    for sums, (at, away) in ((moved, (0, 1)), (misplaced, (1, 0))):
        sums[0][3, at], sums[1][3, at] = ORDERS * 30 * spread, ORDERS * (30 * spread) ** 2
        sums[0][3, away], sums[1][3, away] = 0, 0
    with pytest.raises(AssertionError):
        held(misplaced, moved, "misplaced")
    spreads = error_spread(dew[0] + reference[0], dew[1] + reference[1])
    factor = float(element_factor(dew[0].size, spreads[5]))
    wrong = tuple(side.copy() for side in dew)
    wrong[0][5, 17], wrong[1][5, 17] = (2 * factor * reference[0][5, 17],
                                        (2 * factor) ** 2 * reference[1][5, 17])
    with pytest.raises(AssertionError):
        held(wrong, reference, "one element")
    broken = tuple(side.copy() for side in dew)
    broken[0][0, 0] = np.nan
    with pytest.raises(AssertionError):
        held(broken, reference, "nan")


def test_the_element_factor_widens_with_the_errors_tails_and_the_element_count():
    """The derived factor (`element_factor`): 3.06 at 43.5M elements of a Gaussian's spread, wider
    for a Laplace's (spread 2), and at least FACTOR; the spread is each row's pooled over its
    elements (`error_spread`), a Gaussian's where none moves."""
    assert round(float(element_factor(332 * 131072)), 2) == 3.06
    assert float(element_factor(332 * 131072, 2.0)) > float(element_factor(332 * 131072)) >= FACTOR
    assert float(error_spread(np.zeros((1, 4)), np.zeros((1, 4)))[0]) == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("degrees, allowed", [(5, 2e-3), (3, 3e-3)])
def test_elements_held_over_orders_keep_their_false_failures_on_heavy_tails(degrees, allowed):
    """Two runs as exact as each other, their errors Student-t over ORDERS orders (5 degrees, kurtosis
    9, past Dory's median rows; 3, of unbounded kurtosis), held at a per-element rate of 1e-3 with
    the spread the rule measures: of 200000 elements under 2e-3 fail at 5 degrees and under 3e-3 at
    3, where the RMS ratio's F(ORDERS, ORDERS) bound failed 4.5e-3 and 1.4e-2."""
    from scipy import stats

    rng = np.random.default_rng(9)
    n = 200_000
    (mine, mine_squares), (theirs, theirs_squares) = (
        _summed_over_orders(rng.standard_t(degrees, size=(ORDERS, n))) for _ in range(2))
    spread = float(error_spread((mine + theirs)[None], (mine_squares + theirs_squares)[None])[0])
    factor = np.exp(stats.norm.isf(1e-3) * np.sqrt(2 * (spread - 1) / ORDERS))
    assert np.mean(mine > factor * theirs) < allowed


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
    # The global RMS bound alone passes it.
    assert distance(wrong, truth) <= FACTOR * 64 * eps * float(np.sqrt(np.mean(np.square(truth))))
    with pytest.raises(AssertionError):
        assert_computes_the_oracle(wrong, truth, "one entry", roundings=64)
