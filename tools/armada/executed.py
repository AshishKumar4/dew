"""A pytest plugin that records, for each test file a task runs, the checkout's Python files its tests
executed, for the test cache (`ci.py`): a file whose recorded files are unchanged since it ran green
is not run again.

It counts a file once per test file through sys.monitoring's function-start event, disabled after
the first and restarted at each test file, so it costs about nothing. What collection and
session-scoped fixtures execute serves every test file after them, so it joins every file's record.
A test file that starts another process runs code no event sees, so its record says so, and the
cache never skips it; but for the native tools Dew and its tests ask about the host (`TOOLS`), which
run none of the checkout's code.

It also records where each test file's run read the backend (`reads_backend`) and which of its tests
are marked to run on a GPU (`MARKS`): a file with either takes another path on a GPU, so the H100 tier
runs it (`ci.py gpu-tier`), each file in a process of its own, which records its peak memory on the
GPU (`gpu_peak`). A read is the test file's being collected or run. As with the files they
execute, what session-scoped fixtures read is every file's, and what runs under a module's body,
which runs once, as it is imported, is its own test file's too, and every file's before collection."""

import dis
import json
import os
import sys
import types
from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest
from ci import MARKS

TOOL = 3
"""sys.monitoring's tool ids 0-2 and 5 are the debugger's, coverage's, the profiler's and the optimizer's."""
TOOLS = {"git", "nvidia-smi", "uname", "file"}
"""The programs a test file may start and keep its key: git (a run's commit), nvidia-smi (the GPUs)
and uname and file (the `platform` module's answers). Any other start, a shell's or a launcher's
included, may run the checkout's code."""
SPAWNS = {"subprocess.Popen", "_posixsubprocess.fork_exec", "os.posix_spawn", "os.spawn", "os.exec",
          "os.fork", "os.forkpty"}
"""Audit events of a process start; a multiprocessing child's is `CHILD_START`'s."""
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
decided: dict[types.CodeType, bool] = {}


def started(code, offset):
    if code is CHILD_START:
        if running:
            spawned.add(running[-1])
        return sys.monitoring.DISABLE
    name = code.co_filename
    if not checkout(name):
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
            if frame.f_code.co_name == "<module>" and checkout(body) and not conftest(body):
                if Path(body).name.startswith("test_"):
                    module_reads.setdefault(body[len(root):], set()).add(site)
                elif not readers:
                    everywhere.add(site)
            frame = frame.f_back
    return sys.monitoring.DISABLE


def checkout(name: str) -> bool:
    """Whether `name` is a Python file of the checkout. The environment's packages, which CI installs
    in the checkout, are the locks' to key."""
    return name.startswith(root) and name.endswith(".py") and "site-packages" not in name


def conftest(name: str) -> bool:
    return Path(name).name == "conftest.py"


def reads_backend(code: types.CodeType) -> bool:
    """Whether `code` calls `default_backend` (jax's), by any name table, or reads a device's
    `platform`: an attribute of anything but the `sys` and `platform` modules."""
    if code not in decided:
        steps = list(dis.get_instructions(code))
        decided[code] = any(
            step.argval == "default_backend"
            or (step.opname == "LOAD_ATTR" and step.argval == "platform" and not (
                index and steps[index - 1].opname in ("LOAD_GLOBAL", "LOAD_NAME")
                and steps[index - 1].argval in ("sys", "platform")))
            for index, step in enumerate(steps))
    return decided[code]


def audited(event, args):
    if running and event in SPAWNS and not native(programs(event, args)):
        spawned.add(running[-1])


def programs(event: str, args: tuple) -> list:
    """The programs a process start of `event` may run, by the paths its audit names: `subprocess`'s
    executable (its command's first word where it names none), `_posixsubprocess`'s candidates, and
    none for a fork, whose child runs this process's own code."""
    if event in ("subprocess.Popen", "os.posix_spawn", "os.exec"):
        return [args[0]]
    if event == "_posixsubprocess.fork_exec":
        return list(args[0] or ())
    if event == "os.spawn":
        return [args[1]]
    return []


def native(names: list) -> bool:
    """Whether every one of `names`, and at least one, is a program of `TOOLS`. A start that names
    its program by a file descriptor (`os.execve`'s), or by nothing, is no tool's."""
    return bool(names) and all(isinstance(name, (str, bytes, os.PathLike))
                               and os.path.basename(os.fsdecode(name)) in TOOLS for name in names)


# Recording starts as `-p executed` imports this, before pytest loads the initial conftests, which
# import Dew; repository modules imported even earlier join what every file shares.
shared.update(name[len(root):] for module in list(sys.modules.values())
              if checkout(name := getattr(module, "__file__", None) or ""))
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
        # A file that started a process may have used the GPU there, which no allocator here saw.
        peak = gpu_peak() if len(own) == 1 and not spawned else None
        recorded = {name: {"depends": None if name in spawned else sorted(files | shared),
                           "reads": sorted(reads[name] | module_reads.get(name, set()) | everywhere),
                           "peak": peak} for name, files in own.items()}
        Path(out).write_text(json.dumps(recorded))


def gpu_peak() -> int | None:
    """The most memory this process held at once on a GPU of its own, in bytes, or None on another
    backend: a process that ran one test file holds that file's peak (`ci.py gpu-lane`)."""
    jax = sys.modules.get("jax")
    if jax is None or jax.default_backend() != "gpu":
        return None
    return max(device.memory_stats()["peak_bytes_in_use"] for device in jax.local_devices())
