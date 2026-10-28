"""A pytest plugin that records, for each test file a task runs, the checkout's Python files its tests
executed and the checkout's files and directories they read, for the test cache (`ci.py`): a file
whose recorded files are unchanged since it ran green is not run again. A read is a file opened to
read, as Python's `open` audits it (a documentation page a test trains from, a fixture), or a
directory listed (the pages a test globs, so a page added is a change); the import system's and
pytest's collection's own listings are not, as what they find is recorded as it runs. `os.open`
is wrapped so that a path it opens relative to a directory descriptor is placed (`followed.opening`).
Hidden directories (the checkout's .git and environments) and compiled bytecode are never inputs.

A native reader raises no audit event (Rust's tokenizers and safetensors, tensorstore, a child's C
code), so every directory under tests/ and src/ is watched for opens (`Opens`, inotify): each open of
a file there, by any process, joins the record of the test file running as the watch is read, at each
change of test file, so an open a thread makes after its file ended widens the next file's record,
never narrows one. Where the watches could not all be set (fs.inotify.max_user_watches) or the
kernel's queue overflowed, no record has a key: a record that saw some opens and not others would be
partial (owner, 2026-10-27).

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

import ctypes
import json
import os
import shutil
import struct
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


class Opens:
    """Every open of a data file under tests/ and src/ of the checkout at `root`, by any process, as inotify
    reports it (`IN_OPEN` on each directory there): `drained` gives those since it was last read, relative to
    the checkout. Python files are code, which the record takes from what runs, and a directory's own open
    is a listing, which the audit hook places by who listed it (pytest's collection and the import system
    are no file's input). `blind` where a directory could not be watched or the kernel's queue overflowed:
    what was opened then is not known."""

    OPEN, OVERFLOW, DIRECTORY = 0x20, 0x4000, 0x40000000
    EVENT = struct.Struct("iIII")

    def __init__(self, root: str) -> None:
        self.blind = False
        self.watched: dict[int, str] = {}
        libc = ctypes.CDLL(None, use_errno=True)
        self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            self.blind = True
            return
        for top in ("tests", "src"):
            for directory, names, _ in os.walk(os.path.join(root, top)):
                names[:] = [name for name in names if name != "__pycache__" and not name.startswith(".")]
                watch = libc.inotify_add_watch(self.fd, directory.encode(), self.OPEN)
                if watch < 0:
                    self.blind = True
                    return
                self.watched[watch] = os.path.relpath(directory, root)

    def drained(self) -> set[str]:
        opened: set[str] = set()
        while self.fd >= 0:
            try:
                data = os.read(self.fd, 1 << 16)
            except BlockingIOError:
                break
            at = 0
            while at < len(data):
                watch, mask, _, length = self.EVENT.unpack_from(data, at)
                start = at + self.EVENT.size
                name = data[start:start + length].split(b"\0", 1)[0].decode()
                at += self.EVENT.size + length
                if mask & self.OVERFLOW:
                    self.blind = True
                elif watch in self.watched and name and not mask & self.DIRECTORY and not name.endswith(
                        (".py", ".pyc")):
                    opened.add(os.path.join(self.watched[watch], name))
        opened.discard(str(DURATIONS))
        return opened


def conftest(name: str) -> bool:
    return Path(name).name == "conftest.py"


def audited(event, args):
    if event in followed.READS:
        read = followed.read(event, args, root)
        # The durations record is split.py's, which cuts a split file from it: the file's own entries are its
        # groups' input (ci.py's `cache_key`), and no other file's.
        if read is not None and read != str(DURATIONS):
            current[-1].add(read)
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
followed.opening(lambda read: current[-1].add(read), root)
opens = Opens(root)


def settled() -> None:
    """The opens since the watch was last read, the record running now's (`Opens`)."""
    current[-1].update(opens.drained())


sys.monitoring.use_tool_id(TOOL, "dew-test-cache")
sys.monitoring.register_callback(TOOL, sys.monitoring.events.PY_START, started)
sys.monitoring.set_events(TOOL, sys.monitoring.events.PY_START)
sys.addaudithook(audited)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    name = item.nodeid.split("::")[0]
    if not running or running[-1] != name:
        settled()
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


def pytest_deselected(items):
    tally(deselected, items)


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    if fixturedef.scope != "session":
        yield
        return
    settled()
    current.append(shared)
    readers.append(everywhere)
    sys.monitoring.restart_events()
    try:
        yield
    finally:
        settled()
        current.pop()
        readers.pop()
        sys.monitoring.restart_events()


def pytest_sessionfinish(session):
    settled()
    sys.monitoring.set_events(TOOL, 0)
    out = os.environ.get("DEW_EXECUTED")
    if out:
        found = followed.gathered(children, starts)
        shutil.rmtree(children)
        lost = {name for name, child in found.items() if not child["followed"]}
        recorded = {}
        for name, files in own.items():
            child = found.get(name, {"files": set(), "reads": set(), "peak": None, "shares": 0, "pools": 0})
            whole = name not in spawned | lost and not opens.blind
            # The share a process of this file takes: its own peak, or a child's that took its fraction.
            peaks = [held for held in (gpu_peak(), child["peak"]) if held is not None]
            peak = max(peaks) if len(own) == 1 and whole and peaks else None
            recorded[name] = {"depends": sorted(files | shared | child["files"]) if whole else None,
                              "reads": sorted(reads[name] | module_reads.get(name, set()) | everywhere
                                              | child["reads"]),
                              "peak": peak, "shares": child["shares"], "pools": child["pools"]}
        Path(out).write_text(json.dumps(recorded))
    # Each file's collection, which accounts for a file whose every test the selection set aside (`ci.py`).
    if counted := os.environ.get("DEW_COLLECTED"):
        Path(counted).write_text(json.dumps({name: {"collected": count, "deselected": deselected.get(name, 0)}
                                             for name, count in collected.items()}))


def gpu_peak() -> int | None:
    """The most memory this process held at once on a GPU of its own, in bytes, or None on another
    backend: a process that ran one test file holds that file's peak (`ci.py gpu-lane`)."""
    jax = sys.modules.get("jax")
    if jax is None or jax.default_backend() != "gpu":
        return None
    return max(device.memory_stats()["peak_bytes_in_use"] for device in jax.local_devices())
