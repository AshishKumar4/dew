"""The pytest plugin that keeps one group of a split test file (tools/armada/ci.py).

    python -m pytest -p split --ci-group=G/N tests/test_heavy.py

with tools/armada on the path, as `ci.py task --split=G/N` runs it. Every
container collects the same tests and cuts them as the plan cut the file's
recorded ones (`session.cut`, in node-id order, each weighing its time in
tests/test_durations.json, those marked with one shared setup one weight,
`session.SHARED_SETUP`, into N runs of consecutive tests whose heaviest is as
light as N runs can be; a recorded test the file no longer defines is no part of
it, `session.standing`), each recorded test kept in its planned group and a test
the record lacks spread across them (`session.kept`). This keeps the Gth group
and deselects the rest, after CI's own selection (`session.SELECTION`), and writes each file
the record does not wholly name to DEW_UNRECORDED, whose group's row is kept
with no key (ci.py's `record`). A GPU tier's groups weigh the seconds its GPUs
took instead (`session.cut_record`), and are keyed by the tests that cut gives
them as any group is by its record's.
"""

import json
import os
from pathlib import Path

import pytest
import session


def pytest_addoption(parser):
    parser.addoption("--ci-group", help="G/N: run the Gth of N runs of the collected tests")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    group = config.getoption("ci_group")
    if group is None:
        return
    index, count = (int(part) for part in group.split("/"))
    durations = session.cut_record()
    recorded = json.loads(durations.read_text()) if durations.is_file() else {}
    nodes = sorted(item.nodeid for item in items)
    known = [recorded[node] for node in nodes if node in recorded]
    mean = sum(known) / len(known) if known else 1.0
    # The plan's groups of each file's recorded tests (`cut`, ci.py's `members`), which this task keeps.
    kept_nodes: set[str] = set()
    for name in sorted({node.split("::")[0] for node in nodes}):
        path = Path(name)
        sites = session.declared_marks(name, path.read_text()) if path.is_file() else {}
        of_file = {node: seconds for node, seconds in sorted(recorded.items()) if node.split("::")[0] == name}
        record = session.own(session.standing(name, of_file), sites)
        planned = session.cut(list(record), list(record.values()), count)
        collected = [node for node in nodes if node.split("::")[0] == name]
        groups = session.kept(collected, planned, mean)
        # Every collected test in exactly one group: each task cuts the file whole and alike, so the groups
        # together run each test once, and a cut that would not is no task's to run.
        if sorted(node for members in groups for node in members) != collected:
            raise pytest.UsageError(f"{name}: its {count} groups do not hold each of its {len(collected)} "
                                    "collected tests once (session.kept)")
        kept_nodes |= set(groups[index - 1])
    if unrecorded := os.environ.get("DEW_UNRECORDED"):
        with open(unrecorded, "w") as out:
            json.dump(sorted({node.split("::")[0] for node in nodes if node not in recorded}), out)
    kept = {id(item) for item in items if item.nodeid in kept_nodes}
    config.hook.pytest_deselected(items=[item for item in items if id(item) not in kept])
    items[:] = [item for item in items if id(item) in kept]
