"""A dry prepare's pytest that ends between two tests once it holds more than its task lets it
(`RECYCLE_AT`, bytes resident), so its task runs the tests left in a fresh process (ci.py's `tested`): a
process keeps much of what each test compiled (JAX's traces, XLA's and MLIR's native memory, which neither
its caches' release nor a collection frees: a recurrence family held 1-3.6 GB past its release, scratch
pk-rsstrace), and only its end returns it. It writes the test it ended after to `RECYCLED`, which its task
reads as its word to run again. This module imports no pytest: ci.py reads its names."""

import os
from pathlib import Path

RECYCLE_AT = "DEW_RECYCLE_AT"
"""What names the resident bytes past which a dry prepare's pytest ends after the test running."""
RECYCLED = "DEW_RECYCLED"
"""What names the file a pytest that ended so writes the node id of its last test to."""
CONTINUES = "DEW_CONTINUES_ROW"
"""Set in each later process of one row's task (ci.py's `tested`): the shared XLA cache's bundles it writes
keep every program it fetched, its row's earlier processes' among them (tests/remote_cache.py)."""

_session: dict[str, object] = {}


def resident() -> int:
    """The bytes this process holds resident (`/proc/self/status`'s VmRSS)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def pytest_sessionstart(session) -> None:
    _session["it"] = session


def pytest_runtest_logfinish(nodeid: str, location) -> None:
    """After a test, end the session where this process holds more than `RECYCLE_AT` and a test is left."""
    at, recycled, session = os.environ.get(RECYCLE_AT), os.environ.get(RECYCLED), _session.get("it")
    if not at or not recycled or session is None or not session.items or session.items[-1].nodeid == nodeid:
        return
    if resident() > int(at):
        Path(recycled).write_text(nodeid)
        session.shouldstop = f"recycled after {nodeid}: this process holds more than {at} bytes"
