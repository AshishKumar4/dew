"""A CI task's pytest past its deadline: every thread's stacks as it stood (`dumped`), and its whole record
kept among the task's artifacts (`kept`), beside the host it stood on. A tool, not Dew's library."""

import re
import subprocess
import time
from pathlib import Path

DUMP_SECONDS = 10.0
"""How long py-spy gets to read a stalled pytest's stacks (`dumped`): it stops the process and reads every
thread's frames out of its memory, in under a second for a test's threads (0.14 s on armada, 2026-11-07)."""


def dumped(pid: Path, spy: Path) -> str:
    """Every thread's stack of a task's pytest past its deadline, its Python frames and the native ones under
    them, read from outside it by `spy` (py-spy, the task environment's) while it is stopped: nothing runs
    in the stalled process, so one stalled in the interpreter, or in native code holding the GIL or not (an
    XLA compile, a wait on a device, a collective), says where it stood alike (row 4828). pytest's id is the
    one executed.py wrote to `pid` as it loaded; none where it never loaded."""
    if not pid.is_file():
        return ""
    try:
        read = subprocess.run([str(spy), "dump", "--native", "--pid", pid.read_text()],
                              capture_output=True, text=True, timeout=DUMP_SECONDS)
    except subprocess.TimeoutExpired:
        return f"py-spy read no stacks of pytest in {DUMP_SECONDS:.0f} s"
    return read.stdout or f"py-spy read no stacks of pytest: {read.stderr.strip()}"


def kept(artifacts: Path, name: str, dump: str, began: float, host: str) -> Path:
    """A pytest past its deadline kept whole under the task's `artifacts`, `stalls/<name>-<n>.txt`, `n` the
    first ordinal no record of the name holds yet (made exclusively, so a task's runs of the same rows each
    keep their own): every thread's stacks as `dumped` read them, Python and native, whole, when the task
    began (`began`, epoch seconds) and stalled, and the `host` it stood on (host.py's `state`). A row keeps a
    stall's first 12,000 characters and a reader of armada's tail fewer, which held the main thread's first
    frames alone (rows 5452, 5454)."""
    stamp = "%Y-%m-%dT%H:%M:%SZ"
    held = artifacts / "stalls"
    held.mkdir(parents=True, exist_ok=True)
    named = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    ordinal = 0
    while True:
        record = held / f"{named}-{ordinal}.txt"
        try:
            with record.open("x") as out:
                out.write(f"{dump}\n\ntask began {time.strftime(stamp, time.gmtime(began))}, stalled "
                          f"{time.strftime(stamp, time.gmtime())}\n{host}")
            return record
        except FileExistsError:
            ordinal += 1
