"""Explicit Linux subprocess environments with bounded CPU, memory, time and IO.

The worker runs under the caller's OS permissions with bounded resources.
It has no filesystem or network isolation, so hostile code needs an outer
boundary. Dew never selects or executes this environment by default.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time

from .episodes import Action, Environment, EpisodeId, EpisodeStatus, Observation


@dataclass(frozen=True)
class SandboxLimits:
    """Per-process RLIMIT_CPU/RLIMIT_AS and parent-enforced session/IO limits.

    Forked children inherit the resource limits. Process-group cleanup handles
    descendants that stay in that group; this is not a cgroup aggregate limit.
    """

    wall_seconds: float = 30.
    cpu_seconds: int = 10
    memory_bytes: int = 256 * 1024 ** 2
    message_bytes: int = 1024 ** 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("sandbox wall_seconds must be finite and positive")
        for name in ("cpu_seconds", "memory_bytes", "message_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"sandbox {name} must be a positive integer")


def _observation(value: object) -> Observation:
    if not isinstance(value, Mapping):
        raise ValueError("sandbox response must be an observation object")
    context, status, detail = value.get("context"), value.get("status"), value.get("detail", "")
    if not isinstance(context, list) or not isinstance(status, str) or not isinstance(detail, str):
        raise ValueError("sandbox observation needs context ids, a status name and string detail")
    ids: list[int] = []
    for token in context:
        if type(token) is not int:
            raise ValueError("sandbox context must contain integer token ids")
        ids.append(token)
    try:
        state = EpisodeStatus[status.upper()]
    except KeyError:
        raise ValueError(f"unknown sandbox observation status {status!r}") from None
    return Observation(tuple(ids), state, detail)


class _ProcessEnvironment:
    def __init__(self, command: tuple[str, ...], limits: SandboxLimits,
                 identity: EpisodeId, directory: str):
        self.identity, self.limits = identity, limits
        self.deadline = time.monotonic() + limits.wall_seconds
        self.output = bytearray()
        helper = str(Path(__file__).with_name("_sandbox_exec.py"))
        self.process = subprocess.Popen(
            [sys.executable, "-I", helper, str(limits.cpu_seconds), str(limits.memory_bytes),
             str(os.getpid()), *command], cwd=directory,
            env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        assert self.process.stdin is not None and self.process.stdout is not None and self.process.stderr is not None
        self.stdin, self.stdout, self.stderr = self.process.stdin, self.process.stdout, self.process.stderr
        for stream in (self.stdin, self.stdout, self.stderr):
            os.set_blocking(stream.fileno(), False)

    def _request(self, operation: str, payload: Mapping[str, object]) -> object:
        request = json.dumps({"operation": operation, **payload}, allow_nan=False).encode() + b"\n"
        if len(request) > self.limits.message_bytes:
            raise ValueError("sandbox request exceeds message_bytes")
        sent = 0
        diagnostic = bytearray()
        with selectors.DefaultSelector() as ready:
            ready.register(self.stdin, selectors.EVENT_WRITE, "stdin")
            ready.register(self.stdout, selectors.EVENT_READ, "stdout")
            ready.register(self.stderr, selectors.EVENT_READ, "stderr")
            while True:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("sandbox exceeded wall_seconds")
                if sent == len(request) and b"\n" in self.output:
                    line, _, rest = self.output.partition(b"\n")
                    self.output = rest
                    try:
                        return json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        raise ValueError("sandbox returned malformed JSON") from error
                for event, _ in ready.select(min(remaining, .1)):
                    descriptor = event.fd
                    if event.data == "stdin":
                        try:
                            sent += os.write(descriptor, request[sent:])
                        except BrokenPipeError:
                            ready.unregister(self.stdin)
                        else:
                            if sent == len(request):
                                ready.unregister(self.stdin)
                        continue
                    chunk = os.read(descriptor, 65536)
                    if not chunk:
                        ready.unregister(event.fileobj)
                    elif event.data == "stdout":
                        self.output.extend(chunk)
                    else:
                        diagnostic.extend(chunk)
                    if len(self.output) + len(diagnostic) > self.limits.message_bytes:
                        raise ValueError("sandbox response exceeds message_bytes")
                if self.process.poll() is not None and not ready.get_map():
                    raise ChildProcessError(
                        f"sandbox exited with code {self.process.returncode}: "
                        f"{diagnostic.decode('utf-8', errors='replace')}")

    def reset(self) -> Observation:
        return _observation(self._request("reset", {"episode": asdict(self.identity)}))

    def step(self, action: Action) -> Observation:
        record = {"context": list(action.context), "tokens": list(action.tokens),
                  "terminated": action.terminated, "policy_step": action.policy_step}
        return _observation(self._request("step", {"action": record}))

    def close(self) -> None:
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=5)
        finally:
            for stream in (self.stdin, self.stdout, self.stderr):
                stream.close()


@dataclass(frozen=True)
class SubprocessEnvironment:
    """A user-selected JSON-lines worker implementing reset and step.

    command is an argv tuple, executed without a shell in a temporary working
    directory. Each request has an operation and either an episode identity
    or an action record. Replies contain context (integer ids), status
    (running/completed/truncated/cancelled/error) and optional string detail.

    Session exit kills the process group on success, error or cancellation.
    The direct worker also receives SIGKILL if its parent dies. Neither
    mechanism replaces filesystem/network isolation or controls descendants
    that deliberately leave the group.
    """

    command: tuple[str, ...]
    limits: SandboxLimits = SandboxLimits()

    def __post_init__(self) -> None:
        if sys.platform != "linux":
            raise NotImplementedError("SubprocessEnvironment currently requires Linux resource and parent-death limits")
        if not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise ValueError("sandbox command must be a nonempty argv tuple")

    @contextmanager
    def __call__(self, identity: EpisodeId) -> Iterator[Environment]:
        with tempfile.TemporaryDirectory(prefix="dew-episode-") as directory:
            worker = _ProcessEnvironment(self.command, self.limits, identity, directory)
            try:
                yield worker
            finally:
                worker.close()
