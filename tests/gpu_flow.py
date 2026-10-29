"""A GPU flow (G14, the owner's rule: no unit tests on GPUs): one GPU test that runs many native cells in
one process, each cell a route's property on its generation (`Cell`).

Each cell's label is printed before it runs, so a hang or a fatal device error names the cell it stopped
in (the session's own deadline ends the process). Each cell's failure is kept, and every failure is raised
together after the last cell (`CellsFailed`), so no failure hides another. A cell whose generation this
device cannot run (`generation_support.runnable`), or that skips itself (`pytest.skip`: a missing
optional package, too little free device memory), is reported skipped with its reason, never passed,
and the cells after it still run."""

import dataclasses
import traceback
from collections.abc import Callable, Iterable

import pytest
from generation_support import runnable


@dataclasses.dataclass(frozen=True)
class Cell:
    """One native route's property: `run` asserts it, under `generation` where it names one."""

    label: str
    run: Callable[[], None]
    generation: str | None = None


class CellsFailed(AssertionError):
    """Every failed cell of a flow, each by its label beside its error's own report."""


def run_cells(cells: Iterable[Cell]) -> None:
    """Run every cell in order (`Cell`): each label printed first; every failure kept and raised together
    after the last; the cells this device cannot run, or that skip themselves, reported skipped."""
    failed, skipped, passed = [], [], 0
    for cell in cells:
        if cell.generation is not None and not runnable(cell.generation):
            skipped.append(cell.label)
            continue
        print(f"cell: {cell.label}", flush=True)
        try:
            cell.run()
        except pytest.skip.Exception as skip:
            print(f"cell skipped: {cell.label}: {skip.msg}", flush=True)
            skipped.append(cell.label)
        except Exception as error:
            # Kept with its label, and raised with every other cell's after the last.
            failed.append((cell.label, "".join(traceback.format_exception_only(error)).strip()))
        else:
            passed += 1
    print(f"cells: {passed} passed, {len(failed)} failed, {len(skipped)} skipped on this device", flush=True)
    if failed:
        raise CellsFailed("\n\n".join(f"{label}:\n{report}" for label, report in failed))
    if not passed:
        pytest.skip(f"no cell runs on this device ({len(skipped)} skipped)")
