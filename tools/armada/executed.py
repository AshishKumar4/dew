"""A pytest plugin that records, for each test file a task runs, the checkout's Python files its tests
executed and the checkout's files and directories they read, for the test cache (`ci.py`): a file
whose recorded files are unchanged since it ran green is not run again. A read is a file opened to
read, as Python's `open` audits it (a documentation page a test trains from, a fixture), or a
directory listed (the pages a test globs, so a page added is a change); the import system's and
pytest's collection's own listings are not, as what they find is recorded as it runs. `os.open`
is wrapped so that a path it opens relative to a directory descriptor is placed (`followed.opening`).
Hidden directories (the checkout's .git and environments) and compiled bytecode are never inputs.

A native reader raises no audit event (Rust's tokenizers and safetensors, tensorstore, a child's C
code): the data files a task's processes open are its supervisor's to see (`watched`, held by
`ci.py`'s `tested` around the whole run), each the input of every file the task ran.

It counts a file once per test file through sys.monitoring's function-start event, disabled after
the first and restarted at each test file, so it costs about nothing. What collection and
session-scoped fixtures execute serves every test file after them, so it joins every file's record.
A Python child a test file starts records what it executes and reads as it exits (`followed`), and
its files, its reads of the backend and its GPU pool join the file's record. Any other process start
runs code no event sees, so its record says so, and the cache never skips it; but for the native
tools Dew and its tests ask about the host (`followed.TOOLS`), which run none of the checkout's code.

It also records where each test file's run read the backend (`reads_backend`) and which of its tests
are marked to run on a GPU (`MARKS`): a file with either takes another path on a GPU, so the H100 tier
runs it (`ci.py gpu-tier`), each file in a process of its own, which records its peak memory on the
GPU (`gpu_peak`) and its children's pools. A read is the test file's being collected or run. As with
the files they execute, what session-scoped fixtures read is every file's, and what runs under a
module's body, which runs once, as it is imported, is its own test file's too, and every file's
before collection."""

import json
import os
import shutil
import sys
import tempfile
from multiprocessing.process import BaseProcess
from pathlib import Path

import followed
import pytest
from followed import CHILDREN, checkout, reads_backend
from session import DURATIONS, MARKS

TOOL = 3
"""sys.monitoring's tool ids 0-2 and 5 are the debugger's, coverage's, the profiler's and the optimizer's."""
CHILD_START = BaseProcess.start.__code__
"""Where every multiprocessing context starts a child, a Python process: Python 3.12 audits no spawn
or forkserver start, and a forkserver's later children start in its own process."""

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
"""What ran under each test module's body."""
everywhere: set[str] = set()
"""What session-scoped fixtures read, and module bodies before collection."""
readers: list[set[str]] = []
"""Where a read is kept. conftest.py's reads are every test's, so they are left out."""
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
its class's, its module's, a parameter set's, a conftest's), which `ci.py slow` reads in the file's row."""
starts: dict[str, int] = {}
"""How many followed children each test file's own process started (`followed.start`)."""
children = Path(tempfile.mkdtemp(prefix="dew-children-"))


def started(code, offset):
    if code is CHILD_START:
        # Never disabled: a thread's start that disabled it as the next test file began would leave
        # that file's starts unseen. Process starts are few.
        if running:
            spawned.add(running[-1])
        return None
    name = code.co_filename
    if not checkout(name, root):
        return sys.monitoring.DISABLE
    current[-1].add(name[len(root):])
    if conftest(name):
        return sys.monitoring.DISABLE
    if code.co_name == "<module>":
        # Each module body sees every reader it calls, however many bodies called it before.
        sys.monitoring.restart_events()
    if reads_backend(code):
        site = f"{name[len(root):]}:{code.co_qualname}"
        if readers:
            readers[-1].add(site)
        # The callback runs in the reader's frame, under the bodies of the modules being imported.
        frame = sys._getframe(1)
        while frame is not None:
            body = frame.f_code.co_filename
            if frame.f_code.co_name == "<module>" and checkout(body, root) and not conftest(body):
                if Path(body).name.startswith("test_"):
                    module_reads.setdefault(body[len(root):], set()).add(site)
                elif not readers:
                    everywhere.add(site)
            frame = frame.f_back
    return sys.monitoring.DISABLE


def conftest(name: str) -> bool:
    return Path(name).name == "conftest.py"


def kept(read: str | None) -> None:
    # The durations record is split.py's, which cuts a split file from it, read or probed: the file's own
    # entries are its groups' input (ci.py's `cache_key`), and no other file's.
    if read is not None and read.removeprefix(followed.PROBED) != str(DURATIONS):
        current[-1].add(read)


def audited(event, args):
    if event in followed.READS:
        kept(followed.read(event, args, root))
        return
    kind = followed.start(event, args, str(children)) if running else None
    if kind == "followed":
        starts[running[-1]] = starts.get(running[-1], 0) + 1
    elif kind == "unfollowed":
        spawned.add(running[-1])


# Recording starts as `-p executed` imports this, before pytest loads the initial conftests, which
# import Dew; repository modules imported even earlier join what every file shares.
shared.update(name[len(root):] for module in list(sys.modules.values())
              if checkout(name := getattr(module, "__file__", None) or "", root))
# The children this process starts record themselves (`followed`), the GPU share it was given beside.
followed.installed()
os.environ |= {CHILDREN: str(children), "DEW_EXECUTED_ROOT": root,
               "DEW_EXECUTED_FOLLOWER": str(Path(followed.__file__).resolve()),
               "DEW_EXECUTED_SHARE": os.environ.get(followed.FRACTION, "")}
followed.opening(kept, root)


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
        os.environ["DEW_EXECUTED_FILE"] = name
        sys.monitoring.restart_events()
    reads.setdefault(name, set()).update(f"mark:{mark}" for mark in MARKS if item.get_closest_marker(mark))
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


def pytest_collection_finish(session):
    for item in session.items:
        if item.get_closest_marker("whole_suite"):
            whole_suite.setdefault(item.nodeid.split("::")[0], []).append(item.nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    if fixturedef.scope != "session":
        yield
        return
    current.append(shared)
    readers.append(everywhere)
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
        found = followed.gathered(children, starts)
        shutil.rmtree(children)
        lost = {name for name, child in found.items() if not child["followed"]}
        recorded = {}
        # What the files this process ran before a file executed, read or probed is that file's input too: a
        # module an earlier one imported lazily, a value it cached, a reader that probed a fixture, serve the
        # file without running again (row 2228; the bundle's review). A child's are its own process's.
        earlier: set[str] = set()
        for name, files in own.items():
            child = found.get(name, {"files": set(), "reads": set(), "peak": None, "shares": 0, "pools": 0})
            whole = name not in spawned | lost
            # The share a process of this file takes: its own peak, or a child's that took its fraction.
            peaks = [held for held in (gpu_peak(), child["peak"]) if held is not None]
            peak = max(peaks) if len(own) == 1 and whole and peaks else None
            recorded[name] = {"depends": sorted(files | shared | child["files"] | earlier) if whole else None,
                              "reads": sorted(reads[name] | module_reads.get(name, set()) | everywhere
                                              | child["reads"]),
                              "peak": peak, "shares": child["shares"], "pools": child["pools"]}
            earlier |= files
        Path(out).write_text(json.dumps(recorded))
    # Each file's collection, which accounts for a file whose every test the selection set aside (`ci.py`).
    if counted := os.environ.get("DEW_COLLECTED"):
        Path(counted).write_text(json.dumps({
            name: {"collected": count, "deselected": deselected.get(name, 0),
                   **({"whole_suite": whole_suite[name]} if name in whole_suite else {}),
                   **({"release_env": released[name]} if name in released else {})}
            for name, count in collected.items()}))


def gpu_peak() -> int | None:
    """The most memory this process held at once on a GPU of its own, in bytes, or None on another
    backend: a process that ran one test file holds that file's peak (`ci.py gpu-lane`)."""
    jax = sys.modules.get("jax")
    if jax is None or jax.default_backend() != "gpu":
        return None
    return max(device.memory_stats()["peak_bytes_in_use"] for device in jax.local_devices())
