"""A CI task's pytest past its deadline: every thread's stacks as it stood (`dumped`), and its whole record
kept among the task's artifacts (`kept`), beside the host it stood on. A tool, not Dew's library."""

import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

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


STAMP = "%Y-%m-%dT%H:%M:%SZ"
"""A stall record's times (`stood`, `kept`), UTC."""

ACTIVITY_SECONDS = 1.0
"""How long a stalled pytest's threads are watched (`activity`): long enough for a thread still computing to
show its CPU time, the kernel's clock tick 0.01 s."""


class Thread(NamedTuple):
    """One thread of a stalled pytest as the kernel accounts it (`activity`): its id, its name, its state (R
    running, S sleeping, D in uninterruptible wait), the CPU it used over the watch (user and system,
    seconds) and the kernel function it waits in (0 where it runs)."""
    tid: int
    name: str
    state: str
    cpu: float
    waits: str


def activity(pid: Path, seconds: float = ACTIVITY_SECONDS) -> tuple[Thread, ...]:
    """Each thread of a task's pytest past its deadline over `seconds` (`/proc/<pid>/task`), in its id's
    order: a process still computing shows its threads' CPU (XLA's pool threads among them, which py-spy's
    `dump` does not list, holding no Python frame), a stalled one none (rows 5452, 5454). None where pytest
    never loaded or has gone."""
    if not pid.is_file():
        return ()
    tasks = Path("/proc") / pid.read_text().strip() / "task"
    tick = os.sysconf("SC_CLK_TCK")

    def sampled() -> dict[int, tuple[str, str, int]]:
        found = {}
        for path in tasks.glob("*/stat"):
            try:
                stat = path.read_text()
            except OSError:
                continue
            name, _, rest = stat[stat.index("(") + 1:].rpartition(")")
            fields = rest.split()
            # After the name: the state, then the user and system ticks at the stat's 14th and 15th fields.
            found[int(path.parent.name)] = (name, fields[0], int(fields[11]) + int(fields[12]))
        return found

    first = sampled()
    time.sleep(seconds)
    second = sampled()
    threads = []
    for tid in sorted(first.keys() & second.keys()):
        name, state, ticks = second[tid]
        try:
            waits = (tasks / str(tid) / "wchan").read_text().strip()
        except OSError:
            waits = "?"
        threads.append(Thread(tid, name, state, (ticks - first[tid][2]) / tick, waits))
    return tuple(threads)


def stood(pid: Path, spy: Path, host: Callable[[], str]) -> str:
    """A task's pytest past its deadline as it stood, all of it read before anything stops it: its threads'
    activity over `ACTIVITY_SECONDS` (`activity`), every thread's stacks (`dumped`), then the `host` it stood
    on (host.py's `state`), its processes the stalled pytest among them: read after the interrupt, the host's
    processes held pytest no more and said nothing of what it did (the qwen35/text record of 11-11)."""
    stalled = time.strftime(STAMP, time.gmtime())
    threads = activity(pid)
    lines = [f"stalled {stalled}", "",
             f"threads over {ACTIVITY_SECONDS:.1f} s, {sum(thread.cpu for thread in threads):.2f} s of CPU:"]
    lines += [f"  {thread.tid} {thread.name}: {thread.state}, {thread.cpu:.2f} s, waits in {thread.waits}"
              for thread in threads]
    return "\n".join([*lines, "", dumped(pid, spy), host()])


def kept(artifacts: Path, name: str, record: str, began: float) -> Path:
    """A pytest past its deadline kept whole under the task's `artifacts`, `stalls/<name>-<n>.txt`, `n` the
    first ordinal no record of the name holds yet (made exclusively, so a task's runs of the same rows each
    keep their own): its `record` as it stood (`stood`) after when the task began (`began`, epoch seconds).
    A row keeps a stall's first 12,000 characters and a reader of armada's tail fewer, which held the
    main thread's first frames alone (rows 5452, 5454)."""
    held = artifacts / "stalls"
    held.mkdir(parents=True, exist_ok=True)
    named = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    ordinal = 0
    while True:
        path = held / f"{named}-{ordinal}.txt"
        try:
            with path.open("x") as out:
                out.write(f"task began {time.strftime(STAMP, time.gmtime(began))}\n{record}")
            return path
        except FileExistsError:
            ordinal += 1
