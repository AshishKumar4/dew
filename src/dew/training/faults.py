"""A run's native fault record (row 6497): Python's fault handler armed for every thread into a file of the
run's directory for a fit's whole length, so a fatal signal in it (a segfault in any thread, native code's or
Python's) leaves each thread's Python stack, and on Python 3.14 the faulting thread's C frames by library and
offset (`faulthandler`'s `c_stack`), where the run's checkpoints are. Nothing runs until a fatal signal.

NCCL's UCX transport installs handlers of its own for the fatal signals as its first collectives start,
replacing the record's. CPython's `faulthandler.enable` keeps its first installation (it reinstalls nothing
once enabled) and its `disable` puts back what it found, so a record taken again past UCX's (`rearmed`) holds
the handlers it found, disables, puts them back, and enables over them: the record writes its stacks first,
then hands the signal to UCX's, which prints its own backtrace. A process whose fault handler another owner
enabled (pytest, `PYTHONFAULTHANDLER`) keeps it: the record is that owner's stream, untouched."""

import contextlib
import ctypes
import faulthandler
import logging
import os
import signal
import socket
import sys
from pathlib import Path

_log = logging.getLogger(__name__)

FATAL = (signal.SIGSEGV, signal.SIGFPE, signal.SIGABRT, signal.SIGBUS, signal.SIGILL)
"""The signals `faulthandler` handles."""

_SIGACTION_BYTES = 256
"""Room for a `struct sigaction` (152 bytes on x86-64 and aarch64 Linux), read and written whole."""


def record_path(directory: str) -> Path:
    """This process's fault record in the run directory `directory`: by host and process id, which every
    process of a job knows before its JAX starts (no rank read initializes it early)."""
    return Path(directory) / "faults" / f"{socket.gethostname()}-{os.getpid()}.txt"


def _libc():
    return ctypes.CDLL(None, use_errno=True) if sys.platform.startswith("linux") else None


def dispositions() -> dict[int, bytes] | None:
    """Each fatal signal's installed `struct sigaction`, as the kernel holds it; None off Linux."""
    if (libc := _libc()) is None:
        return None
    held = {}
    for number in FATAL:
        action = ctypes.create_string_buffer(_SIGACTION_BYTES)
        if libc.sigaction(int(number), None, action) != 0:
            return None
        held[int(number)] = action.raw
    return held


def _installed(held: dict[int, bytes]) -> None:
    """Install each of `held`'s actions for its signal."""
    libc = _libc()
    for number, action in held.items():
        libc.sigaction(number, ctypes.create_string_buffer(action, _SIGACTION_BYTES), None)


def _handlers(held: dict[int, bytes] | None) -> dict[int, bytes] | None:
    """Each action's handler, its first word: what tells one owner's installation from another's."""
    return None if held is None else {number: action[:ctypes.sizeof(ctypes.c_void_p)]
                                      for number, action in held.items()}


class Recording:
    """A fit's fault record (`recording`): the file it writes to, None where it records nothing (no run
    directory on this host's file system, or the handler another owner's), and the handlers it installed."""

    def __init__(self, path: Path | None):
        self.path = path
        self._file = None
        self._ours: dict[int, bytes] | None = None

    def start(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a")
        faulthandler.enable(self._file, all_threads=True)
        self._ours = _handlers(dispositions())

    def rearmed(self) -> None:
        """The record installed again past handlers installed over it since (UCX's, as a fit's first step
        starts NCCL), which it then hands each signal on to; nothing where its own still hold."""
        if self._file is None or (found := dispositions()) is None or _handlers(found) == self._ours:
            return
        faulthandler.disable()
        _installed(found)
        faulthandler.enable(self._file, all_threads=True)
        self._ours = _handlers(dispositions())

    def stop(self) -> None:
        """The handler's disabled, what it found put back, and what was installed over it since kept."""
        if self._file is None:
            return
        found = dispositions()
        faulthandler.disable()
        if found is not None and _handlers(found) != self._ours:
            _installed(found)
        self._file.close()
        self._file = None


@contextlib.contextmanager
def recording(directory: str | None):
    """A fit's fault record in `directory` for the block's length (`Recording`): none where the run keeps no
    directory on this host's file system (no checkpoints, or a bucket's), or where another owner enabled
    the process's fault handler, which keeps it."""
    if directory is None or "://" in directory:
        path = None
    elif faulthandler.is_enabled():
        _log.info("the fault handler is the process's own owner's: no fault record in %s", directory)
        path = None
    else:
        path = record_path(directory)
    held = Recording(path)
    held.start()
    try:
        yield held
    finally:
        held.stop()
