"""A run's native fault record (row 6497): Python's fault handler armed for every thread into a file of the
run's directory for a fit's whole length, so a fatal signal in it (a segfault in any thread, native code's or
Python's) leaves each thread's Python stack, and on Python 3.14 the faulting thread's C frames by library and
offset (`faulthandler`'s `c_stack`), where the run's checkpoints are. Nothing runs until a fatal signal.

The record is the fatal signals' one owner for the fit. NCCL's UCX transport would install handlers of its own
over it as its first collectives start, each saving the one it replaced and, on a signal, putting back what it
saved and raising again; a chain of the two re-raises into a handler already spent. So a fit asks UCX for none
(`UCX_ERROR_SIGNALS` empty, where its environment names none); a process whose UCX started before the fit
keeps UCX's, which the record, enabled over it, hands each signal on to. A process whose fault handler another
owner enabled (pytest, `PYTHONFAULTHANDLER`) keeps it: the record is that owner's stream, untouched."""

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


def _actions() -> dict[int, bytes]:
    """Each fatal signal's installed `struct sigaction`, as the kernel holds it; none off Linux."""
    if not sys.platform.startswith("linux"):
        return {}
    libc, held = ctypes.CDLL(None, use_errno=True), {}
    for number in FATAL:
        action = ctypes.create_string_buffer(_SIGACTION_BYTES)
        if libc.sigaction(int(number), None, action) == 0:
            held[int(number)] = action.raw
    return held


def _handler(action: bytes) -> bytes:
    """An action's handler, its first word: what tells one owner's installation from another's."""
    return action[:ctypes.sizeof(ctypes.c_void_p)]


@contextlib.contextmanager
def recording(directory: str | None):
    """A fit's fault record in `directory` for the block's length: none where the run keeps no directory on
    this host's file system (no checkpoints, or a bucket's), or where another owner enabled the process's
    fault handler, which keeps it. Ending, it disables the handler it enabled, keeping any action installed
    over it since for its signal, and puts the environment's UCX setting back."""
    if directory is None or "://" in directory:
        yield None
        return
    if faulthandler.is_enabled():
        _log.info("the fault handler is the process's own owner's: no fault record in %s", directory)
        yield None
        return
    path = record_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    asked = "UCX_ERROR_SIGNALS" not in os.environ
    with path.open("a") as record:
        if asked:
            os.environ["UCX_ERROR_SIGNALS"] = ""
        faulthandler.enable(record, all_threads=True)
        ours = {number: _handler(action) for number, action in _actions().items()}
        try:
            yield path
        finally:
            installed = {number: action for number, action in _actions().items()
                         if _handler(action) != ours.get(number)}
            faulthandler.disable()
            libc = ctypes.CDLL(None) if installed else None
            for number, action in installed.items():
                libc.sigaction(number, ctypes.create_string_buffer(action, _SIGACTION_BYTES), None)
            if asked:
                os.environ.pop("UCX_ERROR_SIGNALS", None)
