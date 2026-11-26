"""A task's pytest that ends between two tests once it holds more than its task lets it (`RECYCLE_AT`, bytes
resident), or before or after a test marked `fresh_process`, so its task runs the tests left in a fresh
process (ci.py's `tested`): a process keeps much of what each test compiled (JAX's traces, XLA's and MLIR's
native memory, which neither its caches' release nor a collection frees: a recurrence family held 1-3.6 GB
past its release, scratch pk-rsstrace; eight higher-order algebra compiles 5.3 to 10 GB, row 13063), and
only its end returns it. It writes the test it ended after to `RECYCLED`, which its task reads as its word to
run again. This module imports no pytest: ci.py reads its names."""

import ctypes
import ctypes.util
import json
import os
from pathlib import Path
from typing import Any

RECYCLE_AT = "DEW_RECYCLE_AT"
"""What names the resident bytes past which a task's pytest ends after the test running."""
FRESH = "fresh_process"
"""The mark of a test that runs in a process of its own: one that needs more than any process but a fresh
one leaves its container (row 13063: a gradient of a gradient's compile peaks at 9.06 GiB of 10 GiB)."""
RECYCLED = "DEW_RECYCLED"
"""What names the file a pytest that ended so writes the node id of its last test to."""
SELECTED = "DEW_RECYCLE_SELECTED"
"""What names the file a row's first process writes the node ids its session selected to
(`pytest_collection_finish`): every test the row must report, whatever a later process's collection finds
(ci.py's `tested`)."""
CONTINUES = "DEW_CONTINUES_ROW"
"""Set in each later process of one row's task (ci.py's `tested`): the shared XLA cache's bundles it writes
keep every program it fetched, its row's earlier processes' among them (tests/remote_cache.py)."""

_session: dict[str, Any] = {}


def resident() -> int:
    """The bytes this process holds resident (`/proc/self/status`'s VmRSS)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def trimmed() -> int:
    """`resident` after glibc returns its arenas' free pages to the system (`malloc_trim(0)`): what a process
    keeps past a test is largely freed native memory its allocator still holds (row 13063: 5.02 GB resident
    after the first orders' compiles, 0.93 GB trimmed, JAX's caches' release returning none of it)."""
    libc = ctypes.util.find_library("c")
    if libc is not None:
        ctypes.CDLL(libc).malloc_trim(0)
    return resident()


def pytest_sessionstart(session) -> None:
    _session["it"] = session


def pytest_collection_finish(session) -> None:
    """The first process of a row writes what its session selected (`SELECTED`); a later one, nothing."""
    if (selected := os.environ.get(SELECTED)) and not Path(selected).exists():
        Path(selected).write_text(json.dumps({"selected": [item.nodeid for item in session.items]}))


def pytest_runtest_logfinish(nodeid: str, location) -> None:
    """After a test, end the session where a test is left and this process holds more than `RECYCLE_AT`, the
    test was marked `FRESH`, or the next one is."""
    at, recycled, session = os.environ.get(RECYCLE_AT), os.environ.get(RECYCLED), _session.get("it")
    if not at or not recycled or session is None or not session.items or session.items[-1].nodeid == nodeid:
        return
    at_item = next((index for index, item in enumerate(session.items) if item.nodeid == nodeid), None)
    fresh = at_item is not None and any(item.get_closest_marker(FRESH) is not None
                                        for item in session.items[at_item:at_item + 2])
    # Past the bound only what the allocator cannot return: a process recycled is a row run again by each train.
    if fresh or (resident() > int(at) and trimmed() > int(at)):
        Path(recycled).write_text(nodeid)
        why = ("a test beside it runs in a fresh process" if fresh
               else f"this process holds more than {at} bytes")
        session.shouldstop = f"recycled after {nodeid}: {why}"
