"""A GPU flow's runner (tests/gpu_flow.py) on cells whose outcomes are known: every cell runs, every
failure is kept and raised together, a cell's own skip is reported and the cells after it still run,
and a flow of no cell that runs is skipped, never passed."""

import pytest
from gpu_flow import Cell, CellsFailed, run_cells


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
    assert "first is off" in str(raised.value) and "last is off" in str(raised.value)
    assert "passing" not in str(raised.value)


def test_a_cell_that_skips_itself_is_reported_and_the_cells_after_it_run(capsys):
    """A cell's own skip is printed with its reason and ends neither the flow nor the cells after it."""
    ran = []
    run_cells([Cell("absent", lambda: pytest.skip("its package is missing")),
               Cell("present", lambda: ran.append("present"))])
    assert ran == ["present"]
    assert "cell skipped: absent: its package is missing" in capsys.readouterr().out


def test_a_flow_of_no_cell_that_runs_is_skipped():
    """A flow none of whose cells ran is skipped, never passed."""
    with pytest.raises(pytest.skip.Exception):
        run_cells([Cell("absent", lambda: pytest.skip("its package is missing"))])
