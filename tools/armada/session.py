"""A CI task's pytest session, as the plan (ci.py) and the plugins a task loads into it (split.py,
executed.py) share it: how pytest runs and what it sees (`pytest_run`, `pytest_environment`), how long it
runs (`DEADLINE`, `GRACE`), which tests take another path on a GPU (`MARKS`), and how a split file's tests
are cut (`groups`). What it decides changes a task's outcome, so the test cache keys every file's record on
it (ci.py's `SHARED`); the plan's own code does not, and a change to it reruns nothing. The standard library
alone: the plan imports it where pytest is not installed."""

import os
from collections.abc import Mapping
from pathlib import Path

DURATIONS = Path("tests/test_durations.json")


SELECTION = ("-m", "not network")
"""Which tests CI runs: every one but those marked network, which reach the Hub (`CLOSED_HUB`)."""

CLOSED_HUB = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
              "HF_ENDPOINT": "http://127.0.0.1:9"}
"""What a task's tests see of the Hugging Face Hub, which none of them may reach (`SELECTION`): the
libraries offline, and the Hub's address a closed port for any that would ask it regardless. A test
that needs the Hub is marked network; one that reads a model or dataset otherwise brings it."""

DEADLINE = 1680.0
"""Seconds a task's tests run before they are interrupted and the files they hold are red: below armada's own
task timeout (.armada.json's), so the verdict is written."""
GRACE = 60.0
"""Seconds an interrupted task's tests have to end before their process group is killed."""


def pytest_run(python: str, report: Path, *measured: str) -> list[str]:
    """CI's pytest under `python`, as every task runs it: CI's `SELECTION`, failures and errors summarized
    with short tracebacks, a file that does not collect beside those that do, unbuffered (-u) so the output
    holds every test that finished when a run is cut short, and pytest's JUnit report written to `report`.
    No timed dump of every thread's stack runs beside the tests, as walking threads that run killed the
    process mid-dump (exit 245, a SIGSEGV, in 3 of 5 runs of a multithreaded interpreted kernel's test, Main
    2026-10-25): a caller's deadline ends a hang (ci.py's `tested`: SIGINT, whose report pytest prints where
    it can handle it, then the group killed past the grace), and a fatal signal still prints the stacks
    (pytest's faulthandler)."""
    return [f".venv-{python}/bin/python", "-u", *measured, "-m", "pytest", "-q", *SELECTION, "-rfE",
            "--tb=short", "--continue-on-collection-errors", f"--junitxml={report}"]


def pytest_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """What CI's pytest runs under: this process's environment, `environ` over it, the Hub closed, and
    tools/armada on the path, where the plugins a task loads (split.py, executed.py) import from."""
    path = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH")]))
    return {**os.environ, **environ, **CLOSED_HUB, "PYTHONPATH": path}


MARKS = ("gpu", "kernel")
"""The marks of tests a GPU runs otherwise: those only a GPU runs, and a kernel's, which a CPU interprets."""


def starts(weights: list[int], capacity: int) -> list[int]:
    """Where each run of consecutive `weights` begins, each run filled up to `capacity`, a heavier
    weight alone."""
    begins, held = [0], 0
    for index, weight in enumerate(weights):
        if held and held + weight > capacity:
            begins.append(index)
            held = 0
        held += weight
    return begins


def milliseconds(seconds: list[float]) -> list[int]:
    return [round(max(value, 0.0) * 1000) for value in seconds]


def groups(seconds: list[float], count: int) -> list[range]:
    """`count` runs of consecutive `seconds`, each filled up to the least capacity that needs no more
    runs, a heavier test alone: how every task of a file split `count` ways cuts the file's tests
    (split.py), in whole milliseconds, so each container cuts them alike. The runs beside a test
    heavier than the rest are no heavier than they must be. A run past the last test is empty."""
    weights = milliseconds(seconds)
    low, high = 0, sum(weights)
    while low < high:
        middle = (low + high) // 2
        low, high = (low, middle) if len(starts(weights, middle)) <= count else (middle + 1, high)
    bounds = [*starts(weights, low), len(weights)]
    bounds += [len(weights)] * (count + 1 - len(bounds))
    return [range(bounds[index], bounds[index + 1]) for index in range(count)]
