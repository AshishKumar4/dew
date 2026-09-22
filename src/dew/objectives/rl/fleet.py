"""A fleet of sandboxed workers that run untrusted programs for verifiable rewards.

A `Program` is files written into a fresh temporary directory, an argv run
there without a shell, and the text fed to its stdin. A runner executes one
program under `SandboxLimits` and reports an `Outcome`: how it ended, its
exit code, its capped output and the seconds it took. `SandboxFleet` runs
programs on `workers` threads at once, each program in its own process.

Two runners exist. `ProcessRunner` starts the program through the same
launcher as `SubprocessEnvironment`: RLIMIT_CPU and RLIMIT_AS, no core
dumps, its own session, SIGKILL on parent death, a minimal environment, and
a process-group kill at the wall deadline. It keeps the caller's user and
filesystem, so it bounds resources, not access. `ContainerRunner` runs the
program in a fresh Docker or Podman container with no network, a read-only
root, the job directory mounted read-only, no capabilities, an unprivileged
user and memory, CPU, process and CPU-time limits; that is the boundary for
hostile code.

A program that fails is an outcome, not an exception: the reward decides
what a timeout or a crash is worth. Only a runner that cannot start a
program at all raises.
"""

from __future__ import annotations

import contextlib
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from .sandbox import SandboxLimits, launch


class Verdict(Enum):
    """How a program ended."""

    COMPLETED = "completed"
    """Exited with status zero."""
    FAILED = "failed"
    """Exited with a nonzero status, an uncaught exception or MemoryError included."""
    TIMEOUT = "timeout"
    """Ran past the wall deadline or its CPU-time limit and was killed."""
    CRASHED = "crashed"
    """Killed by a signal it did not ask for."""
    OUTPUT_LIMIT = "output_limit"
    """Wrote more than `message_bytes` to stdout and stderr together and was killed."""


@dataclass(frozen=True)
class Program:
    """Files to write, the argv to run beside them, and its stdin."""

    files: Mapping[str, str]
    command: tuple[str, ...]
    stdin: str = ""

    def __post_init__(self) -> None:
        if not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise ValueError("a program command is a nonempty argv tuple")
        for name in self.files:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"program file {name!r} must be a relative path inside the job directory")
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))


@dataclass(frozen=True)
class Outcome:
    verdict: Verdict
    exit_code: int | None
    stdout: str
    stderr: str
    seconds: float


class Runner(Protocol):
    """Run one program to its end under `limits`; raise only when it cannot start.

    `python` is the argv that runs a Python file where this runner runs programs.
    """

    @property
    def python(self) -> tuple[str, ...]: ...

    def __call__(self, program: Program, limits: SandboxLimits) -> Outcome: ...


def _written(program: Program, directory: str) -> None:
    for name, text in program.files.items():
        path = Path(directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@dataclass
class _Streams:
    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)


def _collected(process: subprocess.Popen[bytes], stdin: bytes, deadline: float,
               limit: int) -> tuple[_Streams, Verdict | None]:
    """Feed stdin, read both output streams until they close, and wait for the exit.

    Returns what was read, and TIMEOUT or OUTPUT_LIMIT when the process had
    to be stopped for it, else None. The caller kills and reaps.
    """
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    streams = _Streams()
    sent = 0
    with selectors.DefaultSelector() as ready:
        if stdin:
            os.set_blocking(process.stdin.fileno(), False)
            ready.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            process.stdin.close()
        ready.register(process.stdout, selectors.EVENT_READ, streams.stdout)
        ready.register(process.stderr, selectors.EVENT_READ, streams.stderr)
        while any(key.data != "stdin" for key in ready.get_map().values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return streams, Verdict.TIMEOUT
            for key, _ in ready.select(min(remaining, .1)):
                if key.data == "stdin":
                    try:
                        sent += os.write(key.fd, stdin[sent:])
                    except BrokenPipeError:
                        sent = len(stdin)
                    if sent == len(stdin):
                        ready.unregister(key.fileobj)
                        process.stdin.close()
                    continue
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    ready.unregister(key.fileobj)
                    continue
                key.data.extend(chunk)
                if len(streams.stdout) + len(streams.stderr) > limit:
                    return streams, Verdict.OUTPUT_LIMIT
    try:
        # Closed streams are not an exit: wait for it, so a kill that follows
        # never lands between a program's last write and its exit status.
        process.wait(timeout=max(deadline - time.monotonic(), 0))
    except subprocess.TimeoutExpired:
        return streams, Verdict.TIMEOUT
    return streams, None


def _outcome(process: subprocess.Popen[bytes], streams: _Streams, stopped: Verdict | None,
             started: float, deadline: float) -> Outcome:
    """Reap the process and name how it ended."""
    code = process.wait(timeout=max(deadline - time.monotonic(), 0) + 5)
    seconds = time.monotonic() - started
    if stopped is not None:
        verdict = stopped
    elif code == 0:
        verdict = Verdict.COMPLETED
    elif code in (-signal.SIGXCPU, -signal.SIGKILL):
        # The fleet kills only after an exit or a stop it reports itself, so
        # a SIGKILL here is the kernel's: the launcher sets RLIMIT_CPU's soft
        # and hard limits equal, and the hard limit kills with SIGKILL.
        verdict = Verdict.TIMEOUT
    elif code < 0:
        verdict = Verdict.CRASHED
    else:
        verdict = Verdict.FAILED
    return Outcome(verdict, None if stopped is not None else code,
                   streams.stdout.decode("utf-8", errors="replace"),
                   streams.stderr.decode("utf-8", errors="replace"), seconds)


def _closed(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


@dataclass(frozen=True)
class ProcessRunner:
    """Run each program as a resource-limited Linux process in a temporary directory.

    The whole process group is killed at the wall deadline, on excess
    output, and after a normal exit, so no child it forked outlives it
    unless it left the group on purpose. `python` is this interpreter,
    isolated from the environment, site-packages and user site.
    """

    python: tuple[str, ...] = (sys.executable, "-I", "-S")

    def __call__(self, program: Program, limits: SandboxLimits) -> Outcome:
        with tempfile.TemporaryDirectory(prefix="dew-fleet-") as directory:
            _written(program, directory)
            started = time.monotonic()
            deadline = started + limits.wall_seconds
            process = launch(program.command, limits, directory)
            try:
                streams, stopped = _collected(process, program.stdin.encode(), deadline, limits.message_bytes)
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            try:
                return _outcome(process, streams, stopped, started, deadline)
            finally:
                _closed(process)


@dataclass(frozen=True)
class ContainerRunner:
    """Run each program in a fresh, network-less container of `image`.

    `runtime` is the Docker-compatible CLI (`docker` or `podman`). The job
    directory is mounted read-only at `/work`, the working directory; `/tmp`
    is a small writable tmpfs. Memory is capped at `memory_bytes` with no
    swap, CPU time at `cpu_seconds` (SIGXCPU, then SIGKILL a second later), the processor share at `cpus` and the
    process count at `pids`. At the wall deadline the client that runs
    it is killed, then the container is force-removed by name. A runtime that exits without
    creating the container (no daemon, no permission, no image) raises.
    """

    image: str
    runtime: str = "docker"
    cpus: float = 1.0
    pids: int = 64
    user: str = "65534:65534"
    python: tuple[str, ...] = ("python", "-I", "-S")
    """The image's own interpreter; the host's path does not exist inside it."""

    def command(self, program: Program, limits: SandboxLimits, directory: str, name: str, cidfile: str) -> list[str]:
        """The runtime argv that runs `program` from `directory` in a container called `name`.

        The runtime writes the container's id to `cidfile` once it creates one.
        """
        return [self.runtime, "run", "--rm", "-i", "--name", name, "--cidfile", cidfile, "--network", "none",
                "--memory", str(limits.memory_bytes), "--memory-swap", str(limits.memory_bytes),
                "--cpus", str(self.cpus), "--pids-limit", str(self.pids),
                "--ulimit", f"cpu={limits.cpu_seconds}:{limits.cpu_seconds + 1}", "--ulimit", "core=0",
                "--read-only", "--tmpfs", "/tmp:size=64m", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--user", self.user,
                "--volume", f"{directory}:/work:ro", "--workdir", "/work",
                "--env", "PYTHONDONTWRITEBYTECODE=1", self.image, *program.command]

    def __call__(self, program: Program, limits: SandboxLimits) -> Outcome:
        with tempfile.TemporaryDirectory(prefix="dew-fleet-") as scratch:
            # The job is mounted; the id file beside it is not.
            directory, cidfile = os.path.join(scratch, "job"), os.path.join(scratch, "cid")
            os.mkdir(directory)
            _written(program, directory)
            # The container user is not the caller, so it needs to read the files.
            os.chmod(directory, 0o755)
            for path in Path(directory).rglob("*"):
                os.chmod(path, 0o755 if path.is_dir() else 0o644)
            name = f"dew-fleet-{uuid.uuid4().hex}"
            started = time.monotonic()
            # The pull and the container start count against the deadline, so
            # a cold image is a timeout, not an unbounded wait.
            deadline = started + limits.wall_seconds
            process = subprocess.Popen(self.command(program, limits, directory, name, cidfile),
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True)
            try:
                streams, stopped = _collected(process, program.stdin.encode(), deadline, limits.message_bytes)
            finally:
                if process.poll() is None:
                    # The client dies first so it makes no further API call;
                    # `rm --force` then removes the container whether it is
                    # still being created, created or running.
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    subprocess.run([self.runtime, "rm", "--force", name], capture_output=True, timeout=60, check=False)
            try:
                outcome = _outcome(process, streams, stopped, started, deadline)
            finally:
                _closed(process)
            if stopped is None and not (os.path.exists(cidfile) and Path(cidfile).read_text().strip()):
                # The client ended without creating a container: the runtime
                # failed, and the program never ran to be scored.
                raise RuntimeError(f"{self.runtime} created no container (exit {outcome.exit_code}): "
                                   f"{outcome.stderr.strip()}")
            # `docker run` exits with 128 plus the signal that ended the
            # container's process. The CPU soft limit sits one second under
            # the hard one, so running out of CPU time is SIGXCPU; SIGKILL is
            # the memory cgroup's OOM killer.
            code = outcome.exit_code
            if code is not None and code > 128 and outcome.verdict is Verdict.FAILED:
                verdict = Verdict.TIMEOUT if code == 128 + signal.SIGXCPU else Verdict.CRASHED
                return Outcome(verdict, code, outcome.stdout, outcome.stderr, outcome.seconds)
            return outcome


class SandboxFleet:
    """Run programs on `workers` concurrent sandboxed workers.

    Use it as a context manager, or call `close`. Programs queue when every
    worker is busy; `submit` returns at once.
    """

    def __init__(self, runner: Runner = ProcessRunner(), *, limits: SandboxLimits = SandboxLimits(),
                 workers: int = os.cpu_count() or 1):
        if type(workers) is not int or workers < 1:
            raise ValueError("a fleet needs at least one worker")
        self.runner, self.limits = runner, limits
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dew-fleet")

    def submit(self, program: Program) -> Future[Outcome]:
        return self._pool.submit(self.runner, program, self.limits)

    def run(self, programs: Iterable[Program]) -> list[Outcome]:
        """Run every program, concurrently, and return their outcomes in order."""
        futures = [self.submit(program) for program in programs]
        return [future.result() for future in futures]

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> SandboxFleet:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def outputs_match(outcome: Outcome, expected: str) -> bool:
    """A completed program whose stdout equals `expected`, up to surrounding whitespace per line."""
    lines = lambda text: [line.rstrip() for line in text.strip().splitlines()]
    return outcome.verdict is Verdict.COMPLETED and lines(outcome.stdout) == lines(expected)
