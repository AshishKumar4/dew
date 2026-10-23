"""The pytest plugin that keeps one group of a split test file (tools/armada/ci.py).

    python -m pytest -p split --ci-group=G/N --ci-bounds=B tests/test_heavy.py

with tools/armada on the path, as `ci.py task --split=G/N --bounds=B` runs it. `B` is the group's range of
node ids, the plan's (`ci.ranges`): JSON [low, high), either end open as null. It keeps the collected tests,
after CI's own selection (`ci.SELECTION`), whose node ids fall in that range, so a file's groups run each
test once, a test the plan's times lack included. With no range, every container cuts the collected tests
alike, in node-id order, into N runs of as near equal counts as N runs allow (`ci.groups`), and keeps the
Gth.
"""

import json

import pytest
from ci import groups


def pytest_addoption(parser):
    parser.addoption("--ci-group", help="G/N: run the Gth of N runs of the collected tests")
    parser.addoption("--ci-bounds", default="", help="the group's range of node ids, JSON [low, high)")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    group = config.getoption("ci_group")
    if group is None:
        return
    bounds = config.getoption("ci_bounds")
    if bounds:
        low, high = json.loads(bounds)
        kept = {id(item) for item in items
                if (low is None or low <= item.nodeid) and (high is None or item.nodeid < high)}
    else:
        index, count = (int(part) for part in group.split("/"))
        ordered = sorted(items, key=lambda item: item.nodeid)
        run = groups([1.0] * len(ordered), count)[index - 1]
        kept = {id(item) for item in ordered[run.start:run.stop]}
    config.hook.pytest_deselected(items=[item for item in items if id(item) not in kept])
    items[:] = [item for item in items if id(item) in kept]
