"""A pytest plugin that records, for each test file a task runs, the checkout's Python files its tests
executed, for the test cache (`ci.py`): a file whose recorded files are unchanged since it ran green
is not run again.

It counts a file once per test file through sys.monitoring's function-start event, disabled after
the first and restarted at each test file, so it costs about nothing. What collection and
session-scoped fixtures execute serves every test file after them, so it joins every file's record.
A test file that starts another process runs code no event sees, so its record says so, and the
cache never skips it.

It also records where each test file's run read the backend (`reads_backend`) and which of its tests
are marked to run on a GPU (`MARKS`): a file with either takes another path on a GPU, so the H100 tier
runs it (`ci.py gpu-tier`). A read is that of the test file being collected or run, or of the
session-scoped fixture being set up, which every file that requests it shares. A module's body reads
once, as it is imported: for its own test file too, and for every file where nothing is collected yet."""

import dis
import json
import os
import sys
import types
from pathlib import Path

import pytest

TOOL = 3
"""sys.monitoring's tool ids 0-2 and 5 are the debugger's, coverage's, the profiler's and the optimizer's."""
SPAWNS = {"subprocess.Popen", "os.fork", "os.forkpty", "os.posix_spawn", "os.spawn", "os.exec"}
"""Audit events of a process start."""
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
"""Each test file's reads of the backend, as `file:function`, and its marks of `MARKS`."""
module_reads: dict[str, set[str]] = {}
imported: set[str] = set()
"""What module bodies read before collection began."""
fixture_reads: dict[str, set[str]] = {}
requested: dict[str, set[str]] = {}
"""The fixtures each test file's tests requested."""
readers: list[set[str]] = []
"""Where a function's read is kept. conftest.py's reads are every test's, so they are left out."""
decided: dict[types.CodeType, bool] = {}


def started(code, offset):
    name = code.co_filename
    # The environment's packages, which CI installs in the checkout, are the locks' to key.
    if name.startswith(root) and name.endswith(".py") and "site-packages" not in name:
        current[-1].add(name[len(root):])
        if reads_backend(code) and not name.endswith(f"{os.sep}conftest.py"):
            site = f"{name[len(root):]}:{code.co_qualname}"
            if code.co_name == "<module>":
                module_reads.setdefault(name[len(root):], set()).add(site)
            if readers:
                readers[-1].add(site)
            elif code.co_name == "<module>":
                imported.add(site)
    return sys.monitoring.DISABLE


def reads_backend(code: types.CodeType) -> bool:
    """Whether `code` calls `default_backend` (jax's), by any name table, or reads a `platform`
    attribute (a device's), as `platform` alone is the module."""
    if code not in decided:
        decided[code] = any(step.argval == "default_backend"
                            or (step.opname == "LOAD_ATTR" and step.argval == "platform")
                            for step in dis.get_instructions(code))
    return decided[code]


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
    requested.setdefault(name, set()).update(item.fixturenames)
    readers.append(reads[name])
    try:
        yield
    finally:
        readers.pop()


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """A test module's own code runs as it is collected, its marks' conditions among it."""
    if not isinstance(collector, pytest.Module):
        yield
        return
    readers.append(reads.setdefault(collector.nodeid, set()))
    sys.monitoring.restart_events()
    try:
        yield
    finally:
        readers.pop()


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    if fixturedef.scope != "session":
        yield
        return
    current.append(shared)
    readers.append(fixture_reads.setdefault(fixturedef.argname, set()))
    sys.monitoring.restart_events()
    try:
        yield
    finally:
        current.pop()
        readers.pop()
        sys.monitoring.restart_events()


def pytest_sessionfinish(session):
    sys.monitoring.set_events(TOOL, 0)
    out = os.environ.get("DEW_EXECUTED")
    if out:
        recorded = {name: {"depends": None if name in spawned else sorted(files | shared),
                           "reads": sorted(reads.get(name, set()) | module_reads.get(name, set()) | imported
                                           | set().union(*(fixture_reads.get(fixture, set())
                                                           for fixture in requested.get(name, ()))))}
                    for name, files in own.items()}
        Path(out).write_text(json.dumps(recorded))
