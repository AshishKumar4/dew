"""The records of a compile gate's tests run in a pytest process of their own
(`kernel_compile.compiled_apart`), and each test's outcome there: a test's records are admitted only where it
passed (row 1877). Light, for the conftest's hooks: the session's tests (`SESSION`), and in a process of its
own each test's outcome beside its records (`reported`)."""

import hashlib
import json
import os
from pathlib import Path

RECORDS = "DEW_MOSAIC_RECORDS"
"""The directory a process of its own writes its tests' records and outcomes to."""

SESSION: list = []
"""This session's tests (the conftest's `pytest_collection_finish`)."""


def _named(test: str) -> str:
    """`test`'s node id as a file name."""
    return hashlib.sha256(test.encode()).hexdigest()[:24]


def written(directory: Path, test: str, call: int, found: list[dict]) -> None:
    """`test`'s `call`-th records, as its process of its own compiled them."""
    (directory / f"{_named(test)}-{call}.json").write_text(json.dumps(found))


def reported(report) -> None:
    """In a process of its own (`RECORDS`), a test's outcome beside its records: passed once its call passed,
    failed, with its report, where a phase of it failed, and failed where it ended with no call that passed
    (a skip)."""
    if RECORDS not in os.environ:
        return
    outcome = Path(os.environ[RECORDS]) / f"{_named(report.nodeid)}.outcome"
    if report.failed:
        outcome.write_text(json.dumps({"passed": False, "when": report.when,
                                       "report": str(report.longrepr)[-3000:]}))
    elif report.when == "call" and report.passed:
        outcome.write_text(json.dumps({"passed": True}))
    elif report.when == "teardown" and not outcome.exists():
        outcome.write_text(json.dumps({"passed": False, "when": "call", "report": "no call of it passed"}))


def ran(directory: Path, test: str) -> bool:
    """Whether `test` finished in the process of its own that wrote `directory`."""
    return (directory / f"{_named(test)}.outcome").exists()


def read(directory: Path, test: str, call: int) -> list[dict]:
    """`test`'s `call`-th records from the process of its own that wrote `directory`: refused where the test
    did not pass there or wrote none, the process's output (`output.txt`) or the test's report saying why."""
    outcome = directory / f"{_named(test)}.outcome"
    records = directory / f"{_named(test)}-{call}.json"
    if not outcome.exists():
        output = (directory / "output.txt").read_text()[-3000:]
        raise RuntimeError(f"{test} did not finish in a process of its own:\n{output}")
    said = json.loads(outcome.read_text())
    if not said["passed"]:
        raise RuntimeError(f"{test} in a process of its own failed at its {said['when']}:\n{said['report']}")
    if not records.exists():
        raise RuntimeError(f"{test} in a process of its own passed, but wrote no records of compile {call}")
    return json.loads(records.read_text())
