"""The pytest plugin that says which GPU generations each collected test file's tests are pinned to
(tools/armada/collection.py, for `ci.py gpu-tier`).

    python -m pytest --collect-only -m gpu -p pins --pins-out=FILE tests/

with tools/armada on the path. A test whose every cell runs on some GPU generations alone says so with the
mark `generations` (`@pytest.mark.generations("sm90")`, on the test, its class, its module or a parameter of
it); a test without it runs on every generation, its cells skipping what a device cannot run. FILE is
written as JSON: each file with a selected test, by the generations its tests are pinned to together, and by
none where any of them is pinned to none, so a tier's composer gives a GPU kind every file but those pinned
to other generations than its own.
"""

import json

MARK = "generations"


def pytest_addoption(parser):
    parser.addoption("--pins-out", help="where each collected file's pinned GPU generations are written")


def pytest_collection_finish(session):
    out = session.config.getoption("pins_out")
    if out is None:
        return
    pinned: dict[str, set[str] | None] = {}
    for item in session.items:
        name = item.nodeid.split("::")[0]
        own = {str(generation) for mark in item.iter_markers(MARK) for generation in mark.args}
        if not own:
            pinned[name] = None
        elif pinned.setdefault(name, set()) is not None:
            pinned[name] |= own
    with open(out, "w") as file:
        json.dump({name: sorted(found or ()) for name, found in sorted(pinned.items())}, file)
