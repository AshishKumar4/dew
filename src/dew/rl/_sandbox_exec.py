"""Apply process limits before exec without a threaded parent's preexec_fn."""

import os
import resource
import sys


def main() -> None:
    started_coverage()
    cpu, memory = (int(value) for value in sys.argv[1:3])
    command = sys.argv[3:]
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    # No parent-death signal: the worker's lifetime is its group's lifeline's (`dew._cleanup.owned_group`),
    # where the signal, which fires when the thread that forked the process ends (a short-lived starter's,
    # `dew._cleanup.start`), killed a healthy worker at its start (the models review, row 12172).
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
