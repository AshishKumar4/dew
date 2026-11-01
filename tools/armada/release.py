"""The pytest plugin that keeps the tests of one release check (tools/armada/ci.py).

    python -m pytest -p release --ci-release=REQUIREMENT tests/test_remote_code.py

with tools/armada on the path, as `ci.py task --release=REQUIREMENT` runs it, under uv's overlay of
REQUIREMENT on the suite's environment: the tests marked release_env(REQUIREMENT) are kept and every
other test deselected, after CI's own selection (`session.SELECTION`).
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--ci-release", help="REQUIREMENT: run the tests marked release_env(REQUIREMENT) alone")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    wanted = config.getoption("ci_release")
    if wanted is None:
        return
    kept = [item for item in items
            if (mark := item.get_closest_marker("release_env")) is not None and mark.args[:1] == (wanted,)]
    held = {id(item) for item in kept}
    dropped = [item for item in items if id(item) not in held]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
