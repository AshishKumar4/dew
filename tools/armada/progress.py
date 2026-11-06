"""A pytest plugin that keeps a task's progress where armada keeps a recalled attempt's artifacts:
each test that passed whole (its call and its teardown), a line of PROGRESS under ARMADA_ARTIFACTS as it
ends, so the task's next attempt, recalled while it ran, sets them aside and goes on from there
(tools/armada/ci.py `resumed`, armada's ARMADA_RESUME). Loaded by a CI task's pytest beside executed.py;
writes nothing outside a task."""

import json
import os
from pathlib import Path

PROGRESS = "progress.jsonl"
"""The file of the tests a task passed, a JSON line each: its node id and its call's seconds."""
SET_ASIDE = "DEW_SET_ASIDE"
"""What names the JSON list of the node ids a resumed attempt sets aside (ci.py's `tested`): exactly those,
never a test whose id one of them begins, as pytest's --deselect, which matches by prefix, would (train
671's review: test_host_placement's `..._trains_the_same_model` would have set aside its
`..._over_a_scaled_window`)."""


def pytest_collection_modifyitems(config, items) -> None:
    """The items whose node ids SET_ASIDE names taken out of the session, said deselected as pytest says its
    own, which executed.py counts. This module imports no pytest: ci.py, which a planner with none runs,
    reads PROGRESS and SET_ASIDE from it."""
    named = os.environ.get(SET_ASIDE)
    aside = set(json.loads(Path(named).read_text())) if named else set()
    if dropped := [item for item in items if item.nodeid in aside]:
        items[:] = [item for item in items if item.nodeid not in aside]
        config.hook.pytest_deselected(items=dropped)


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
