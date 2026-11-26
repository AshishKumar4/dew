"""A run's native fault record (row 6497): Python's fault handler for every thread, the process's one owner of
its fatal signals from its start (`owned`), its stacks written into a file of the run's directory while a fit
runs (`recording`), so a fatal signal in it (a segfault in any thread, native code's or Python's) leaves each
thread's Python stack, and on Python 3.14 the faulting thread's C frames by library and offset
(`faulthandler`'s `c_stack`), where the run's checkpoints are. Nothing runs until a fatal signal.

The handler is enabled once, as the process is prepared, before its pool's join (`runtime.prepare_process`),
readied (`runtime.ready_process`) or a fit records (`recording`), whichever comes first, and never disabled: a
handler installed over it (UCX's, as NCCL's transport starts, each saving the one it replaced and, on a
signal, putting it back and raising again) hands each signal back to it, which writes its stacks and hands
on to what it found. A fit re-points it at the run's file and back at standard error after, which installs
nothing. The process's policy asks UCX for no handler of its own where its environment names none
(`UCX_ERROR_SIGNALS` empty, read once as UCX starts: a transport already started keeps the policy it read),
as Dew's other environment defaults are; a process whose fault handler another owner enabled (pytest,
`PYTHONFAULTHANDLER`) keeps it, and records nothing."""

import contextlib
import faulthandler
import logging
import os
import socket
import sys
from pathlib import Path
from typing import TextIO

_log = logging.getLogger(__name__)

_OWNED: list[TextIO | None] = []
"""The stream this process's fault handler writes to where it is Dew's, None where another owner's, decided
once (`owned`)."""


def owned() -> None:
    """The process's fatal signals Python's fault handler's, for every thread, to standard error, and UCX
    asked for none where the environment names its own policy (`UCX_ERROR_SIGNALS`): once a process,
    before NCCL starts. Where another owner enabled the handler, it stays theirs."""
    if _OWNED:
        return
    if faulthandler.is_enabled() or (stream := sys.__stderr__) is None:
        _OWNED.append(None)
        return
    os.environ.setdefault("UCX_ERROR_SIGNALS", "")
    faulthandler.enable(stream, all_threads=True)
    _OWNED.append(stream)


def record_path(directory: str) -> Path:
    """This process's fault record in the run directory `directory`: by host and process id, which every
    process of a job knows before its JAX starts (no rank read initializes it early)."""
    return Path(directory) / "faults" / f"{socket.gethostname()}-{os.getpid()}.txt"


@contextlib.contextmanager
def recording(directory: str | None):
    """The process's fault handler writing to its record in the run directory `directory` for the block,
    to standard error again after it: re-pointed, nothing installed, the handler the process's from here
    on where nothing made it so before (`owned`: no JAX read). None where the handler is another owner's,
    the run keeps no directory on this host's file system (no checkpoints, or a bucket's), or its record
    cannot be made there, which a warning names; an error of the block is the block's own."""
    owned()
    if (stream := _OWNED[0]) is None or directory is None or "://" in directory:
        yield None
        return
    path = record_path(directory)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = path.open("a")
    except OSError as error:
        # A run directory it cannot keep a record in (a regular file named as its directory beside the
        # checkpoints, a read-only mount): the fit runs, its faults on standard error (row 13017).
        _log.warning("no fault record at %s (%s): a fatal signal's stacks go to standard error", path, error)
        yield None
        return
    with record:
        faulthandler.enable(record, all_threads=True)
        try:
            yield path
        finally:
            faulthandler.enable(stream, all_threads=True)
