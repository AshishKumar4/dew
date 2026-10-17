"""A pytest plugin that records, for each test file a task runs, the checkout's Python files its tests
executed, for the test cache (`ci.py`): a file whose recorded files are unchanged since it ran green
is not run again.

It counts a file once per test file through sys.monitoring's function-start event, disabled after
the first and restarted at each test file, so it costs about nothing. What collection and
session-scoped fixtures execute serves every test file after them, so it joins every file's record.
A Python child a test file starts records itself as it exits (`followed`), and its files, reads and
GPU pool join the file's record. Any other process start runs code no event sees, so its record says
so, and the cache never skips it; but for the native tools Dew and its tests ask about the host
(`followed.TOOLS`), which run none of the checkout's code.

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
from ci import MARKS
from followed import CHILDREN, checkout, reads_backend

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


def audited(event, args):
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
        peak = gpu_peak() if len(own) == 1 and not spawned | lost else None
        recorded = {}
        for name, files in own.items():
            child = found.get(name, {"files": set(), "reads": set(), "shares": 0, "pools": 0})
            whole = name not in spawned | lost
            recorded[name] = {"depends": sorted(files | shared | child["files"]) if whole else None,
                              "reads": sorted(reads[name] | module_reads.get(name, set()) | everywhere
                                              | child["reads"]),
                              "peak": peak, "shares": child["shares"], "pools": child["pools"]}
        Path(out).write_text(json.dumps(recorded))


def gpu_peak() -> int | None:
    """The most memory this process held at once on a GPU of its own, in bytes, or None on another
    backend: a process that ran one test file holds that file's peak (`ci.py gpu-lane`)."""
    jax = sys.modules.get("jax")
    if jax is None or jax.default_backend() != "gpu":
        return None
    return max(device.memory_stats()["peak_bytes_in_use"] for device in jax.local_devices())
