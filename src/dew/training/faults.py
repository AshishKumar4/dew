"""A run's native fault record (row 6497): Python's fault handler armed for every thread into a file of the
run's directory, so a fatal signal in a fit (a segfault in any thread, native code's or Python's) leaves each
thread's Python stack, and on Python 3.14 the faulting thread's C frames by library and offset (`c_stack`),
where the run's checkpoints are. Nothing runs until a fatal signal arrives.

A fit arms it as it starts and again once its first step has run (`Trainer.fit`): NCCL's UCX transport
installs its own handler as it starts and replaces one armed before it, while one armed after it chains to
UCX's, which still prints its backtrace after the Python stacks. A fit ends with the process's handler as it
found it."""

import faulthandler
import sys
from pathlib import Path
from typing import TextIO

import jax

_RECORDS: dict[Path, TextIO] = {}
"""Each record this process armed, open for the process's life: the handler writes to its file descriptor."""


def record_path(directory: str, process: int) -> Path:
    """Process `process`'s fault record in the run directory `directory`."""
    return Path(directory) / "faults" / f"process-{process}.txt"


def armed(directory: str | None) -> Path | None:
    """Arm the fault handler for every thread into this process's record of the run directory `directory`
    (`record_path`), appended to, and return its path; none where the run has no directory on this host's
    file system (no checkpoints, or a bucket's), the handler left as it is."""
    if directory is None or "://" in directory:
        return None
    path = record_path(directory, jax.process_index())
    if path not in _RECORDS:
        path.parent.mkdir(parents=True, exist_ok=True)
        _RECORDS[path] = path.open("a")
    # Python 3.14 prints the faulting thread's C frames too, by default (`c_stack`).
    faulthandler.enable(_RECORDS[path], all_threads=True)
    return path


def restored(enabled: bool) -> None:
    """The process's fault handler as a fit found it (`enabled`, `faulthandler.is_enabled()` then): its
    stacks to the process's standard error (`sys.__stderr__`, whatever captures `sys.stderr`), or none."""
    if enabled and sys.__stderr__ is not None:
        faulthandler.enable(sys.__stderr__, all_threads=True)
    else:
        faulthandler.disable()
