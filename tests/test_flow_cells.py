"""A GPU flow's runner (tests/gpu_flow.py) on cells whose outcomes are known: every cell runs, every
failure is kept and raised together, a cell's own skip is reported and the cells after it still run,
and a flow of no cell that runs is skipped, never passed."""

import gpu_flow
import pytest
from gpu_flow import Cell, CellsFailed, cells_of, run_cells


def test_a_flow_runs_every_cell_and_raises_every_failure_together():
    """A failing cell is kept, the cells after it run, and every failure is raised together after the
    last, each by its label."""
    ran = []

    def failing(name):
        def run():
            ran.append(name)
            raise AssertionError(f"{name} is off")
        return run

    with pytest.raises(CellsFailed) as raised:
        run_cells([Cell("first", failing("first")), Cell("passing", lambda: ran.append("passing")),
                   Cell("last", failing("last"))])
    assert ran == ["first", "passing", "last"]
    assert raised.value.labels == ["first", "last"]


def test_a_cell_that_skips_itself_is_reported_and_the_cells_after_it_run():
    """A cell's own skip is reported by its label and ends neither the flow nor the cells after it."""
    ran = []
    skipped = run_cells([Cell("absent", lambda: pytest.skip("its package is missing")),
                         Cell("present", lambda: ran.append("present"))])
    assert ran == ["present"] and skipped == ["absent"]


def test_a_flow_of_no_cell_that_runs_is_skipped():
    """A flow none of whose cells ran is skipped, never passed."""
    with pytest.raises(pytest.skip.Exception):
        run_cells([Cell("absent", lambda: pytest.skip("its package is missing"))])


ran_with = []
TOO_WIDE = True


FORMS = {"p": ("plain", 1.0), "s": ("scaled", 2.0)}


@pytest.mark.parametrize("width", [64, pytest.param(128, marks=pytest.mark.skipif(TOO_WIDE, reason="wide"))])
@pytest.mark.parametrize(("form", "scale"), FORMS.values(), ids=FORMS)
def gridded(form, scale, width, fixture):
    """A cell function whose test kept its grid in its own marks."""
    ran_with.append((form, scale, width, fixture))


def test_a_cell_function_gives_a_cell_a_parameter_set_and_skips_where_its_set_does():
    """`cells_of`: the product of the function's own parametrize marks, each set run with its arguments and
    the fixtures given, a set whose skipif holds skipped by its reason."""
    cells = cells_of(gridded, fixture="given")
    assert sorted(cell.label for cell in cells) == ["gridded: 128-p", "gridded: 128-s", "gridded: 64-p",
                                                    "gridded: 64-s"]
    ran_with.clear()
    skipped = run_cells(cells)
    assert sorted(ran_with) == [("plain", 1.0, 64, "given"), ("scaled", 2.0, 64, "given")]
    assert sorted(skipped) == ["gridded: 128-p", "gridded: 128-s"]


def test_a_flow_reports_its_counts_once_however_its_cells_nest():
    """A flow's counts (`gpu_flow.REPORTED`, the JUnit property `cells` CI reads): an outermost flow's alone,
    a sweep's inner flow counted within its cell, and a flow none of whose cells ran reported with none
    passed before it skips, so it reads as unqualified rather than green."""
    gpu_flow.REPORTED.clear()
    run_cells([Cell("a sweep", lambda: run_cells([Cell("one", lambda: None), Cell("two", lambda: None)])),
               Cell("absent", lambda: pytest.skip("not here"))])
    with pytest.raises(pytest.skip.Exception):
        run_cells([Cell("absent", lambda: pytest.skip("not here"))])
    assert [{key: run[key] for key in ("passed", "failed", "skipped")} for run in gpu_flow.REPORTED] == [
        {"passed": 1, "failed": 0, "skipped": 1}, {"passed": 0, "failed": 0, "skipped": 1}]
    gpu_flow.REPORTED.clear()


def test_a_cell_failing_by_pytests_own_failures_ends_no_flow_and_is_counted():
    """A cell's `pytest.fail`, and a `pytest.raises` that saw nothing, are its failures (they derive from
    BaseException): each is kept and raised with the rest, the cells after it still run, and the flow's
    counts are reported (`gpu_flow.REPORTED`) however it ends."""
    ran = []

    def unraised():
        with pytest.raises(ValueError):
            ran.append("unraised")

    gpu_flow.REPORTED.clear()
    with pytest.raises(CellsFailed) as raised:
        run_cells([Cell("failed", lambda: pytest.fail("its pool never ended")), Cell("unraised", unraised),
                   Cell("after", lambda: ran.append("after"))])
    assert raised.value.labels == ["failed", "unraised"] and ran == ["unraised", "after"]
    assert [{key: run[key] for key in ("passed", "failed", "skipped")} for run in gpu_flow.REPORTED] == [
        {"passed": 1, "failed": 2, "skipped": 0}]
    gpu_flow.REPORTED.clear()


def test_a_repeated_step_holds_its_bits_and_a_changed_one_fails():
    """`gpu_flow.repeats_bit_for_bit`: a step that gives the same bits each run passes, compiled afresh after
    its first; one whose bits change between runs fails, naming its leaf."""
    import jax.numpy as jnp

    gpu_flow.repeats_bit_for_bit(lambda x: {"sum": x + 1, "product": x * 2}, jnp.arange(4.0), runs=3)
    counter = iter(range(10))

    def drifting(x):
        return {"drift": x + next(counter) * 1e-6}

    with pytest.raises(AssertionError, match="drift"):
        gpu_flow.repeats_bit_for_bit(drifting, jnp.arange(4.0))


def test_an_isolated_cell_takes_its_own_directory_and_the_suites_per_test_settings(tmp_path_factory):
    """`gpu_flow.isolated`: a cell runs in a directory of its own, with its own rung records, data cache and
    Hub answers beneath it, the fixtures it names passed, and each setting undone after it."""
    import os

    seen = {}

    def check(tmp_path, monkeypatch):
        seen.update(directory=tmp_path, rungs=os.environ["DEW_RUNG_RECORDS"],
                    cache=os.environ["DEW_DATA_CACHE"], answers=os.environ["DEW_HUB_ANSWERS"],
                    patch=monkeypatch)

    before = os.environ.get("DEW_RUNG_RECORDS")
    gpu_flow.isolated(tmp_path_factory)(check, "own")()
    directory = seen["directory"]
    assert seen["rungs"] == str(directory / "rungs") and seen["cache"] == str(directory / "data-cache")
    assert seen["answers"] == str(directory / "hub-answers") and os.environ.get("DEW_RUNG_RECORDS") == before

