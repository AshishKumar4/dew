"""A pytest plugin that tells the test cache's recorder (lifetime.py) whose each thing a task's pytest runs
is: a test file's while its tests run, every file's (`lifetime.SHARED`) while collection, conftests and a
session- or package-scoped fixture's making and ending run, and from the session's end on; and where each test
file's run read the backend (`followed.reads_backend`) and which of its tests are marked to run on a GPU
(`MARKS`), which `ci.py gpu-tier --evidence` reports; the H100 tier runs the marked files alone (`ci.py
gpu-tier`). A read is the test file's being collected or run; what a session-scoped fixture reads is every
file's, and what runs under a module's body, which runs once, as it is imported, is its own test file's too,
and every file's before collection.

What the recorder logs is everything the process executed, read, imported and started, as it happened
(lifetime.py), which the supervisor reads once the process has exited: this plugin decides no record. It
writes how collection found each file (`DEW_COLLECTED`), which accounts for a file whose every test the
selection set aside."""

import ctypes
import json
import os
import sys

import lifetime
import pytest
from session import MARKS

if lifetime.log is None:
    raise pytest.UsageError("executed.py records under lifetime.py's boot alone (session.py's pytest_run)")

# This process's id, written whole to the file DEW_PYTEST_PID names, whose every thread's stack the task
# reads from outside it once its deadline passes, and puts in its row before its interrupt (ci.py's
# `dumped`): a stall names its call site, one in collection or a fixture's setup too, which pytest's report
# of the interrupt cannot (2026-11-03: a group stalled 28 minutes in a read with no deadline, its row saying
# only that no tests ran). pytest's own id. Taken
# out of the environment, so no pytest a test starts writes over it.
if written := os.environ.pop("DEW_PYTEST_PID", None):
    reader = int(os.environ.pop("DEW_STACKS_READER"))
    # Where Yama lets a process trace its own descendants alone (ptrace_scope 1, the GPU hosts' Ubuntu), this
    # one names the task as its tracer (PR_SET_PTRACER, "Yama"), whose py-spy, the task's child, may read it.
    if os.path.exists("/proc/sys/kernel/yama/ptrace_scope"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(0x59616D61, ctypes.c_ulong(reader), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), f"executed.py could not name the task ({reader}) its tracer")
    with open(f"{written}.part", "w") as pid:
        pid.write(str(os.getpid()))
    os.replace(f"{written}.part", written)

collected: dict[str, int] = {}
"""Each test file's tests as collection found them, before any hook chose among them; a module of none
is 0."""
deselected: dict[str, int] = {}
"""Each test file's tests the selection set aside, as pytest reports it (`pytest_deselected`)."""
SHARED_SCOPES = ("session", "package")
"""The fixture scopes whose making and ending serve every file after them: they run for every file."""


# pytest put its assertion rewriter first before it loaded this plugin.
lifetime.leading()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    name = item.nodeid.split("::")[0]
    if lifetime.owners[0] != name:
        lifetime.switch(name)
    for mark in MARKS:
        if item.get_closest_marker(mark):
            lifetime.write(name, f"{lifetime.READ}mark:{mark}")
    lifetime.readers.append(name)
    try:
        yield
    finally:
        lifetime.readers.pop()


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """A test module's own code runs as it is collected, its marks' conditions among it."""
    if not isinstance(collector, pytest.Module):
        yield
        return
    lifetime.readers.append(collector.nodeid)
    # Each module body sees every reader it calls, however many bodies called it before.
    sys.monitoring.restart_events()
    try:
        yield
    finally:
        lifetime.readers.pop()


def pytest_collectreport(report):
    if report.passed and report.nodeid.endswith(".py"):
        collected.setdefault(report.nodeid, 0)


def tally(counts: dict[str, int], items) -> None:
    """Each item's test file counted once more in `counts`."""
    for item in items:
        name = item.nodeid.split("::")[0]
        counts[name] = counts.get(name, 0) + 1


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    tally(collected, items)


def pytest_deselected(items):
    tally(deselected, items)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    """A session- or package-scoped fixture's making and its ending are every file's (`lifetime.SHARED`),
    whichever test first asked for it and whichever test's teardown, or the session's end, ends it: its
    finalizers run last in, first out, so `shared` registered before its own and `owned` after them bracket
    them."""
    if fixturedef.scope not in SHARED_SCOPES:
        yield
        return
    fixturedef.addfinalizer(owned)
    shared()
    try:
        yield
    finally:
        owned()
        fixturedef.addfinalizer(shared)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_fixture_post_finalizer(fixturedef, request):
    """What runs as a session- or package-scoped fixture's ending is done (pytest's hook after its
    finalizers) is every file's too, a conftest's wrapper of it among it: this wrapper runs first, around
    theirs."""
    if fixturedef.scope not in SHARED_SCOPES:
        yield
        return
    shared()
    try:
        yield
    finally:
        owned()


def shared() -> None:
    """What runs from here is every file's, until `owned`."""
    lifetime.push(lifetime.SHARED)
    lifetime.readers.append(lifetime.SHARED)


def owned() -> None:
    lifetime.readers.pop()
    lifetime.pop()


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session):
    """From the session's end on, what runs is every file's; and how collection found each file."""
    lifetime.switch(lifetime.SHARED)
    if counted := os.environ.get("DEW_COLLECTED"):
        with open(counted, "w") as out:
            json.dump({name: {"collected": count, "deselected": deselected.get(name, 0)}
                       for name, count in collected.items()}, out)
