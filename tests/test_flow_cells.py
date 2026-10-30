"""A GPU flow's runner (tests/gpu_flow.py) on cells whose outcomes are known: every cell runs, every
failure is kept and raised together, a cell's own skip is reported and the cells after it still run,
and a flow of no cell that runs is skipped, never passed."""

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


@pytest.mark.parametrize("width", [64, pytest.param(128, marks=pytest.mark.skipif(TOO_WIDE, reason="wide"))])
@pytest.mark.parametrize(("form", "scale"), [("plain", 1.0), ("scaled", 2.0)], ids=["p", "s"])
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

