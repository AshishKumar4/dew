"""A pytest plugin that records, for each test file a task runs, the checkout's Python files its tests
executed, for the test cache (`ci.py`): a file whose recorded files are unchanged since it ran green
is not run again.

It counts a file once per test file through sys.monitoring's function-start event, disabled after
the first and restarted at each test file, so it costs about nothing. What collection and
session-scoped fixtures execute serves every test file after them, so it joins every file's record.
A test file that starts another process runs code no event sees, so its record says so, and the
cache never skips it.

It also records the functions of the checkout that read the backend (`READS`) as each test file is
collected or its tests run, and which of its tests are marked to run on a GPU: a file with either takes
another path on a GPU, so the H100 tier runs it (`ci.py gpu-tier`)."""

import json
import os
import sys
from pathlib import Path

import pytest

TOOL = 3
"""sys.monitoring's tool ids 0-2 and 5 are the debugger's, coverage's, the profiler's and the optimizer's."""
SPAWNS = {"subprocess.Popen", "os.fork", "os.forkpty", "os.posix_spawn", "os.spawn", "os.exec"}
"""Audit events of a process start."""
READS = frozenset({"default_backend", "platform"})
"""The names code reads the backend by: `jax.default_backend()` and a device's `platform`."""
MARKS = ("gpu", "kernel")
"""The marks of tests a GPU runs otherwise: those only a GPU runs, and a kernel's, which a CPU interprets."""

root = str(Path.cwd()) + os.sep
shared: set[str] = set()
"""What collection and session-scoped fixtures executed."""
own: dict[str, set[str]] = {}
spawned: set[str] = set()
current: list[set[str]] = [shared]
running: list[str] = []
reads: dict[str, set[str]] = {}
"""Each test file's functions that read the backend, and its marks of `MARKS`."""
reader: list[str] = []
"""The test file being collected or run: conftest.py reads the backend for every test, so it is left out."""


def started(code, offset):
    name = code.co_filename
    # The environment's packages, which CI installs in the checkout, are the locks' to key.
    if name.startswith(root) and name.endswith(".py") and "site-packages" not in name:
        current[-1].add(name[len(root):])
        if reader and not READS.isdisjoint(code.co_names) and not name.endswith(f"{os.sep}conftest.py"):
            reads.setdefault(reader[-1], set()).add(f"{name[len(root):]}:{code.co_qualname}")
    return sys.monitoring.DISABLE


def audited(event, args):
    if running and event in SPAWNS:
        spawned.add(running[-1])


# Recording starts as `-p executed` imports this, before pytest loads the initial conftests, which
# import Dew; repository modules imported even earlier join what every file shares.
shared.update(name[len(root):] for module in list(sys.modules.values())
              if (name := getattr(module, "__file__", None) or "").startswith(root) and name.endswith(".py")
              and "site-packages" not in name)
sys.monitoring.use_tool_id(TOOL, "dew-test-cache")
sys.monitoring.register_callback(TOOL, sys.monitoring.events.PY_START, started)
sys.monitoring.set_events(TOOL, sys.monitoring.events.PY_START)
sys.addaudithook(audited)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    name = item.nodeid.split("::")[0]
    if not running or running[-1] != name:
        running[:] = [name]
        current[:] = [own.setdefault(name, set())]
        sys.monitoring.restart_events()
    reads.setdefault(name, set()).update(f"mark:{mark}" for mark in MARKS if item.get_closest_marker(mark))
    reader.append(name)
    try:
        yield
    finally:
        reader.pop()


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """A test module's own code runs as it is collected, its marks' conditions among it."""
    if not isinstance(collector, pytest.Module):
        yield
        return
    reader.append(collector.nodeid)
    sys.monitoring.restart_events()
    try:
        yield
    finally:
        reader.pop()


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    if fixturedef.scope != "session":
        yield
        return
    current.append(shared)
    sys.monitoring.restart_events()
    try:
        yield
    finally:
        current.pop()
        sys.monitoring.restart_events()


def pytest_sessionfinish(session):
    sys.monitoring.set_events(TOOL, 0)
    out = os.environ.get("DEW_EXECUTED")
    if out:
        recorded = {name: {"depends": None if name in spawned else sorted(files | shared),
                           "reads": sorted(reads.get(name, ()))} for name, files in own.items()}
        Path(out).write_text(json.dumps(recorded))
