"""A GPU flow (G14, the owner's rule: no unit tests on GPUs): one GPU test that runs many native cells in
one process, each cell a route's property on its generation (`Cell`).

Each cell's label is printed before it runs, so a hang or a fatal device error names the cell it stopped
in (the session's own deadline ends the process). Each cell's failure is kept, and every failure is raised
together after the last cell (`CellsFailed`), so no failure hides another. A cell whose generation this
device cannot run (`generation_support.runnable`), or that skips itself (`pytest.skip`: a missing
optional package, too little free device memory), is reported skipped with its reason, never passed,
and the cells after it still run. A cell function that keeps its test's grid in
its own `pytest.mark.parametrize` marks gives a cell a parameter set (`cells_of`)."""

import dataclasses
import functools
import inspect
import itertools
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
    """Every failed cell of a flow, each by its label beside its error's own report; `labels` the failed
    cells' in order."""

    def __init__(self, failed: list[tuple[str, str]]):
        super().__init__("\n\n".join(f"{label}:\n{report}" for label, report in failed))
        self.labels = [label for label, _ in failed]


REPORTED: list[dict] = []
"""The counts of each outermost flow run since the last test's report took them (`run_cells`), which
tests/conftest.py reports as the test's JUnit property `cells`: a flow none of whose cells ran is
unqualified on the GPU it was run for, never green (ledger 2643)."""

_depth = 0
"""How many flows enclose the one running: a sweep's inner flow (`run_cells` within a cell) counts within
its cell, not apart."""


def _device() -> str:
    """The device a flow runs on, as its counts name it."""
    import jax

    return jax.devices()[0].device_kind


def run_cells(cells: Iterable[Cell]) -> list[str]:
    """Run every cell in order (`Cell`): each label printed first; every failure kept and raised together
    after the last; the cells this device cannot run, or that skip themselves, reported skipped, their
    labels returned; an outermost flow's counts reported (`REPORTED`)."""
    global _depth
    _depth += 1
    failed, skipped, passed = [], [], [0]
    try:
        _ran(cells, failed, skipped, passed)
    finally:
        # Counted however the flow ends, so a flow stopped early still reports what it ran.
        _depth -= 1
        if not _depth:
            REPORTED.append({"passed": passed[0], "failed": len(failed), "skipped": len(skipped),
                             "device": _device()})
    passed = passed[0]
    print(f"cells: {passed} passed, {len(failed)} failed, {len(skipped)} skipped on this device", flush=True)
    if failed:
        raise CellsFailed(failed)
    if not passed:
        pytest.skip(f"no cell runs on this device ({len(skipped)} skipped)")
    return skipped


def _ran(cells: Iterable[Cell], failed: list[tuple[str, str]], skipped: list[str], passed: list[int]) -> None:
    """Each cell run (`run_cells`), into `failed` (by label and report), `skipped` (labels) and `passed` (its
    one count): a cell's failure is any exception, pytest's own failures among them (`pytest.fail`, and a
    `pytest.raises` that saw nothing), which derive from BaseException and would otherwise end the flow."""
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
        except (Exception, pytest.fail.Exception) as error:
            # Kept with its label, and raised with every other cell's after the last.
            failed.append((cell.label, "".join(traceback.format_exception_only(error)).strip()))
        else:
            passed[0] += 1


def _skipped(marks, reason_of: str) -> Callable[[], None] | None:
    """`marks`' first `skipif` whose condition holds, as a run that skips for its reason; or None."""
    for mark in marks:
        if mark.name == "skipif" and mark.args and bool(mark.args[0]):
            reason = mark.kwargs.get("reason", reason_of)

            def skip(reason=reason):
                pytest.skip(reason)
            return skip
    return None


def _parameter_sets(mark) -> list[tuple[str, dict, list]]:
    """A `parametrize` mark's sets: each its id, its arguments by name, and its own marks."""
    names = mark.args[0]
    names = [name.strip() for name in names.split(",")] if isinstance(names, str) else list(names)
    ids = mark.kwargs.get("ids")
    if ids is not None and not callable(ids):
        ids = list(ids)  # a mapping's ids are its keys, as pytest reads them
    sets = []
    for at, value in enumerate(list(mark.args[1])):
        given = value if isinstance(value, type(pytest.param(None))) else None
        values = given.values if given else (value,) if len(names) == 1 else tuple(value)
        if given and given.id is not None:
            label = given.id
        elif callable(ids):
            label = "-".join(str(ids(item)) for item in values)
        elif ids is not None:
            label = str(ids[at])
        else:
            label = "-".join(getattr(item, "__name__", str(item)) for item in values)
        sets.append((label, dict(zip(names, values, strict=True)), list(given.marks) if given else []))
    return sets


def cells_of(check, generation: str | None = None, **fixtures) -> list[Cell]:
    """A cell for each parameter set `check`'s own `pytest.mark.parametrize` marks give (their product, as
    pytest collects them), each run with that set and those of `fixtures` it takes by name, under
    `generation`; a `skipif` mark of `check`'s own, or of a set's, whose condition holds skips its cell
    with its reason, as pytest would. So a cell function keeps its test's grid where its test kept it,
    and a flow names the function alone."""
    marks = list(getattr(check, "pytestmark", []))
    taken = inspect.signature(check).parameters
    grids = [_parameter_sets(mark) for mark in reversed(marks) if mark.name == "parametrize"]
    name = check.__name__.replace("_", " ")
    cells = []
    for combination in itertools.product(*grids) if grids else [()]:
        arguments = {name: value for name, value in fixtures.items() if name in taken}
        own, labels = list(marks), []
        for label, values, set_marks in combination:
            labels.append(label)
            arguments.update(values)
            own += set_marks
        run = _skipped(own, name) or functools.partial(check, **arguments)
        cells.append(Cell(f"{name}: {'-'.join(labels)}" if labels else name, run, generation))
    return cells


def when(holds: bool, reason: str, check: Callable[[], None]) -> Callable[[], None]:
    """`check` where `holds`, else skipped for `reason`: a cell gated on a capability (`bf16_dot_runs`, a
    package) rather than on a generation's launch sizes (`Cell.generation`)."""
    def run():
        if not holds:
            pytest.skip(reason)
        check()
    return run


def isolated(tmp_path_factory) -> Callable[..., Callable[[], None]]:
    """A flow's `in_own(check, name, *, devices=1, **given)`: `check` run as the suite runs a test, in a
    directory of its own (`tmp_path`) with tests/conftest.py's per-test settings, its own rung records, data
    cache, Hub answers and Hub cache, and a `monkeypatch` undone after it, each passed where it takes them;
    skipped where this process holds fewer than `devices` devices."""
    def in_own(check, name: str, *, devices: int = 1, **given) -> Callable[[], None]:
        def run():
            import jax
            from huggingface_hub import constants

            if jax.device_count() < devices:
                pytest.skip(f"needs {devices} devices; this process has {jax.device_count()}")
            directory = tmp_path_factory.mktemp(name)
            with pytest.MonkeyPatch.context() as patch:
                patch.setenv("DEW_RUNG_RECORDS", str(directory / "rungs"))
                patch.setenv("DEW_DATA_CACHE", str(directory / "data-cache"))
                patch.setenv("DEW_HUB_ANSWERS", str(directory / "hub-answers"))
                patch.setattr(constants, "HF_HUB_CACHE", str(directory / "hub-cache"))
                fixtures = {"tmp_path": directory, "monkeypatch": patch}
                taken = inspect.signature(check).parameters
                check(**{key: value for key, value in fixtures.items() if key in taken}, **given)
        return run
    return in_own


def repeats_bit_for_bit(step: Callable, *inputs, runs: int = 2, names: Iterable[str] | None = None) -> None:
    """`step(*inputs)` run `runs` times, the second compiled afresh (`jax.clear_caches`): every leaf of every
    run's output the first run's bits, so a schedule the compiler picks repeats as well as one it is handed
    (kernel-templates.md's reproducibility); each leaf named by `names` where given, else by its path."""
    import jax
    import numpy as np

    first = jax.tree.map(np.asarray, step(*inputs))
    paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(first)]
    labels = list(names) if names is not None else paths
    for run in range(1, runs):
        if run == 1:
            jax.clear_caches()
        again = jax.tree.leaves(jax.tree.map(np.asarray, step(*inputs)))
        for label, one, other in zip(labels, jax.tree.leaves(first), again, strict=True):
            np.testing.assert_array_equal(one.view(np.uint8), other.view(np.uint8),
                                          err_msg=f"{label} on run {run + 1}")

