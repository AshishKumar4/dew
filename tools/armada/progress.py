"""A pytest plugin that keeps a task's progress where armada keeps a recalled attempt's artifacts:
each test that passed whole (its call and its teardown), a line of PROGRESS under ARMADA_ARTIFACTS as it
ends, so the task's next attempt, recalled while it ran, deselects them and goes on from there
(tools/armada/ci.py `resumed`, armada's ARMADA_RESUME). Loaded by a CI task's pytest beside executed.py;
writes nothing outside a task."""

import json
import os
from pathlib import Path

PROGRESS = "progress.jsonl"
"""The file of the tests a task passed, a JSON line each: its node id and its call's seconds."""

_called: dict[str, float] = {}


def pytest_runtest_logreport(report) -> None:
    """A test's call passed is kept once its teardown passed too; one whose teardown failed is no progress."""
    artifacts = os.environ.get("ARMADA_ARTIFACTS")
    if not artifacts:
        return
    if report.when == "call" and report.passed:
        _called[report.nodeid] = report.duration
    elif report.when == "teardown" and report.nodeid in _called:
        seconds = _called.pop(report.nodeid)
        if report.passed:
            Path(artifacts).mkdir(parents=True, exist_ok=True)
            with (Path(artifacts) / PROGRESS).open("a") as progress:
                progress.write(json.dumps({"node": report.nodeid, "seconds": seconds}) + "\n")
