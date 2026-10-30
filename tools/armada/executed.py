"""A pytest plugin that tells the test cache's recorder (lifetime.py) whose each thing a task's pytest runs
is: a test file's while its tests run, every file's (`lifetime.SHARED`) while collection, conftests and a
session- or package-scoped fixture's making and ending run, and from the session's end on; and where each test
file's run read the backend (`followed.reads_backend`) and which of its tests are marked to run on a GPU
(`MARKS`): a file with either takes another path on a GPU, so the H100 tier runs it (`ci.py gpu-tier`). A read
is the test file's being collected or run; what a session-scoped fixture reads is every file's, and what runs
under a module's body, which runs once, as it is imported, is its own test file's too, and every file's before
collection.

What the recorder logs is everything the process executed, read, imported and started, as it happened
(lifetime.py), which the supervisor reads once the process has exited: this plugin decides no record. It
writes how collection found each file (`DEW_COLLECTED`), which accounts for a file whose every test the
selection set aside, and which tests carry the marks a row names (`whole_suite`, `release_env`)."""

import json
import os
import sys

import lifetime
import pytest
from session import MARKS

if lifetime.log is None:
    raise pytest.UsageError("executed.py records under lifetime.py's boot alone (session.py's pytest_run)")

collected: dict[str, int] = {}
"""Each test file's tests as collection found them, before any hook chose among them; a module of none
is 0."""
deselected: dict[str, int] = {}
"""Each test file's tests the selection set aside, as pytest reports it (`pytest_deselected`)."""
released: dict[str, dict[str, list[str]]] = {}
"""Each test file's release checks, by the requirement each runs under, as pytest resolves a node's
release_env mark, recorded before any selection: its row names them, and a whole suite's run plans their
tasks from it (`ci.py release-plan`)."""
whole_suite: dict[str, list[str]] = {}
"""Each test file's tests marked whole_suite, by their node ids, as pytest resolves a node's marks (its own,
its class's, its module's, a parameter set's, a conftest's), every one collected, those the selection sets
aside among them: the file's row names its cells (`ci.py slow`, verdict.sh's record of what main holds)."""
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
    # Every cell collected, those a train's run sets aside among them: main holds its cells though its train
    # CI never times one, so a whole-suite run of one main has is no new slow test (verdict.sh).
    for item in items:
        if item.get_closest_marker("whole_suite"):
            whole_suite.setdefault(item.nodeid.split("::")[0], []).append(item.nodeid)
    # Before any selection sets them aside, as every run but a release check's does (`ci.py selected`).
    # A mark that names no one requirement string fails the collection: set aside from every run, no
    # release task would plan its test, which would run nowhere.
    for item in items:
        marks = list(item.iter_markers("release_env"))
        if not marks:
            continue
        named = {mark.args[0] for mark in marks
                 if len(mark.args) == 1 and isinstance(mark.args[0], str) and not mark.kwargs}
        if len(named) != 1 or any(len(mark.args) != 1 or mark.kwargs for mark in marks):
            raise pytest.UsageError(
                f"{item.nodeid}: release_env takes one requirement string, as "
                f"release_env(\"transformers==5.18.0\"); it has "
                f"{', '.join(f'args {mark.args} kwargs {mark.kwargs}' for mark in marks)}")
        checks = released.setdefault(item.nodeid.split("::")[0], {})
        checks.setdefault(named.pop(), []).append(item.nodeid)


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
            json.dump({name: {"collected": count, "deselected": deselected.get(name, 0),
                              **({"whole_suite": whole_suite[name]} if name in whole_suite else {}),
                              **({"release_env": released[name]} if name in released else {})}
                       for name, count in collected.items()}, out)
