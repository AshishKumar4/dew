"""The pytest plugin that keeps one group of a split test file (tools/armada/ci.py).

    python -m pytest -p split --ci-group=G/N tests/test_heavy.py

with tools/armada on the path, as `ci.py task --split=G/N` runs it. Every
container collects the same tests and cuts them alike (`session.cut`): in
node-id order, each weighing its time in tests/test_durations.json and a test
that file lacks the mean of the others, those it records marked with one shared
setup one weight (`session.SHARED_SETUP`), into N runs of consecutive tests whose
heaviest is as light as N runs can be. This keeps the Gth run and deselects
the rest, after CI's own selection (`session.SELECTION`), and writes each file
the record does not wholly name to DEW_UNRECORDED, whose group's row is kept
with no key (ci.py's `record`).
"""

import json
import os

import pytest
from session import DURATIONS, cut


def pytest_addoption(parser):
    parser.addoption("--ci-group", help="G/N: run the Gth of N runs of the collected tests")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    group = config.getoption("ci_group")
    if group is None:
        return
    index, count = (int(part) for part in group.split("/"))
    recorded = json.loads(DURATIONS.read_text()) if DURATIONS.is_file() else {}
    nodes = sorted(item.nodeid for item in items)
    known = [recorded[node] for node in nodes if node in recorded]
    mean = sum(known) / len(known) if known else 1.0
    kept_nodes = set(cut(nodes, [recorded.get(node, mean) for node in nodes], count, recorded)[index - 1])
    if unrecorded := os.environ.get("DEW_UNRECORDED"):
        with open(unrecorded, "w") as out:
            json.dump(sorted({node.split("::")[0] for node in nodes if node not in recorded}), out)
    kept = {id(item) for item in items if item.nodeid in kept_nodes}
    config.hook.pytest_deselected(items=[item for item in items if id(item) not in kept])
    items[:] = [item for item in items if id(item) in kept]
