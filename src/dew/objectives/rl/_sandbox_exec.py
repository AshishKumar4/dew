"""Apply process limits before exec without a threaded parent's preexec_fn."""

import ctypes
import os
import resource
import signal
import sys


def main() -> None:
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
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
