"""Apply process limits before exec without a threaded parent's preexec_fn."""

import ctypes
import os
import resource
import signal
import sys


def main() -> None:
    started_coverage()
    cpu, memory, parent = (int(value) for value in sys.argv[1:4])
    command = sys.argv[4:]
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    # Linux clears this setting on fork. Install it in the launched child,
    # then check the parent again to close the race with parent termination.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot install parent-death signal")
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGKILL)
    saved_coverage()
    os.execvpe(command[0], command, os.environ)


def started_coverage() -> None:
    """Where the parent is measured (`launched` passes its coverage startup on), this launcher is measured
    too: `-I` reads no startup of its own."""
    if "COVERAGE_PROCESS_START" in os.environ or "COVERAGE_PROCESS_CONFIG" in os.environ:
        import coverage

        coverage.process_startup()


def saved_coverage() -> None:
    """A measured launcher's data saved before the exec replaces the process: an exec skips the exit, where
    coverage saves."""
    measured = sys.modules.get("coverage")
    current = None if measured is None else measured.Coverage.current()
    if current is not None:
        current.stop()
        current.save()


if __name__ == "__main__":
    main()
