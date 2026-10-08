"""The pytest plugin that keeps one group of a split test file (tools/armada/ci.py).

    python -m pytest -p split --ci-group=G/N tests/test_heavy.py

with tools/armada on the path, as `ci.py task --split=G/N` runs it. Every
container collects the same tests and cuts them alike (`ci.groups`): in
node-id order, each weighing its time in tests/test_durations.json and a test
that file lacks the mean of the others, into N runs of consecutive tests whose
heaviest is as light as N runs can be. This keeps the Gth run and deselects
the rest, after CI's own selection (`-m "not network"`).
"""

import json

import pytest
from ci import DURATIONS, groups


def pytest_addoption(parser):
    parser.addoption("--ci-group", help="G/N: run the Gth of N runs of the collected tests")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    group = config.getoption("ci_group")
    if group is None:
        return
    index, count = (int(part) for part in group.split("/"))
    recorded = json.loads(DURATIONS.read_text()) if DURATIONS.is_file() else {}
    ordered = sorted(items, key=lambda item: item.nodeid)
    known = [recorded[item.nodeid] for item in ordered if item.nodeid in recorded]
    mean = sum(known) / len(known) if known else 1.0
    run = groups([recorded.get(item.nodeid, mean) for item in ordered], count)[index - 1]
    kept = {id(item) for item in ordered[run.start:run.stop]}
    config.hook.pytest_deselected(items=[item for item in items if id(item) not in kept])
    items[:] = [item for item in items if id(item) in kept]
