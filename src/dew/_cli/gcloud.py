"""The subprocesses dew tpu starts: every one is built, printed and run here.

A dry run prints the command and returns success without touching gcloud.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Sequence

from dew._records import JSON

DEAD_STATES = frozenset({"TERMINATED", "PREEMPTED", "DELETING"})
"""The states a TPU does not come back from to READY."""
READY_TIMEOUT = 1800
POLL_SECONDS = 10

#: The binary every command in this module goes through.
GCLOUD = "gcloud"

_PRINT = threading.Lock()


def emit(line: str) -> None:
    """Print one whole line, even when workers report at the same time."""
    with _PRINT:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def said(done: subprocess.CompletedProcess[str]) -> str:
    """What a finished command printed, its output then its errors, where it was captured."""
    return ((done.stdout or "") + (done.stderr or "")).strip()


@dataclasses.dataclass(frozen=True, slots=True)
class Node:
    """The parts of a describe response that dew tpu reads."""

    name: str
    zone: str
    accelerator_type: str
    state: str
    health: str
    runtime_version: str
    topology: str
    ips: tuple[str, ...]
    internal_ips: tuple[str, ...]
    spot: bool
    queued_resource: str

    @property
    def workers(self) -> int:
        return max(1, len(self.internal_ips))

    @classmethod
    def parse(cls, payload: dict) -> Node:
        endpoints = payload.get("networkEndpoints") or []
        scheduling = payload.get("schedulingConfig") or {}
        accelerator = payload.get("acceleratorConfig") or {}
        path = payload.get("name", "")
        queued = payload.get("queuedResource", "")
        return cls(
            name=path.rsplit("/", 1)[-1],
            zone=_segment(path, "locations"),
            accelerator_type=payload.get("acceleratorType", ""),
            state=payload.get("state", "UNKNOWN"),
            health=payload.get("health", ""),
            runtime_version=payload.get("runtimeVersion", ""),
            topology=accelerator.get("topology", ""),
            ips=tuple((e.get("accessConfig") or {}).get("externalIp", "") for e in endpoints),
            internal_ips=tuple(e.get("ipAddress", "") for e in endpoints),
            spot=bool(scheduling.get("spot") or scheduling.get("preemptible")),
            queued_resource=queued.rsplit("/", 1)[-1] if queued else "",
        )


def _segment(path: str, key: str) -> str:
    parts = path.split("/")
    return parts[parts.index(key) + 1] if key in parts else ""


class Gcloud:
    """Builds gcloud argv, runs it, reads its JSON."""

    def __init__(self, project: str = "", dry_run: bool = False):
        self.project = project
        self.dry_run = dry_run

    def argv(self, *args: str) -> list[str]:
        """A gcloud command line, with the project when the config names one."""
        line = [GCLOUD, *args]
        if self.project:
            line.append(f"--project={self.project}")
        return line

    def run(self, argv: Sequence[str], *, capture: bool = True, check: bool = False,
            timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        """Run one command, killed after `timeout` seconds (`subprocess.TimeoutExpired`). A dry run
        prints it and reports success."""
        argv = list(argv)
        if self.dry_run:
            emit(shlex.join(argv))
            return subprocess.CompletedProcess(argv, 0, "", "")
        done = subprocess.run(argv, capture_output=capture, text=True, timeout=timeout)
        if check and done.returncode:
            raise SystemExit(said(done) or f"{argv[0]} failed with code {done.returncode}")
        return done

    def json(self, *args: str, timeout: float | None = None) -> JSON:
        """Run a gcloud command that speaks JSON and parse it: None in a dry run, which prints the
        command instead. A failed command stops dew with gcloud's message, as an answer of nothing
        would read as nothing there (an authentication failure as no TPUs)."""
        outcome = self.run(self.argv(*args, "--format=json"), check=True, timeout=timeout)
        return json.loads(outcome.stdout) if outcome.stdout.strip() else None

    def nodes(self, zone: str, timeout: float | None = None) -> list[Node]:
        """The TPUs `zone` lists, asked for at most `timeout` seconds: none in a dry run. A TPU absent
        from a listing gcloud gave is not there, which a failed describe cannot say apart from gcloud
        failing."""
        payload = self.json("compute", "tpus", "tpu-vm", "list", f"--zone={zone}", timeout=timeout)
        if payload is None:
            return []
        if not isinstance(payload, list):
            raise ValueError(f"gcloud listed {payload!r} in {zone}, which is not a list of TPU records")
        nodes = []
        for entry in payload:
            if not isinstance(entry, dict):
                raise ValueError(f"gcloud listed {entry!r} in {zone}, which is not a TPU record")
            nodes.append(Node.parse(entry))
        return nodes

    def config_value(self, name: str) -> str:
        """Read one value from the local gcloud config.

        Runs in a dry run too: it reads and changes nothing, and the argv a dry
        run prints can depend on the answer.
        """
        done = subprocess.run([GCLOUD, "config", "get-value", name],
                              capture_output=True, text=True)
        value = done.stdout.strip()
        return "" if done.returncode or value == "(unset)" else value

    def stream(self, argv: Sequence[str], prefix: str = "") -> subprocess.CompletedProcess[str]:
        """Run one command, tagging every output line with a prefix."""
        argv = list(argv)
        if self.dry_run:
            emit(shlex.join(argv))
            return subprocess.CompletedProcess(argv, 0, "", "")
        # Popen as a context manager closes the pipe it opened; left to the
        # collector, a fanout over a pod leaks one file per worker.
        with subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1,
        ) as process:
            stdout = process.stdout
            if stdout is None:
                raise RuntimeError(f"{argv[0]} opened without the pipe it was asked for")
            for line in stdout:
                emit(prefix + line.rstrip("\n"))
            return subprocess.CompletedProcess(argv, process.wait())

    def fanout(
        self,
        jobs: Sequence[tuple[str, Sequence[str]]],
        *,
        capture: bool = False,
    ) -> list[subprocess.CompletedProcess[str]]:
        """Run one command per worker at the same time, in job order."""
        if not jobs:
            return []
        if self.dry_run:
            return [self.run(argv) for _, argv in jobs]

        def one(job: tuple[str, Sequence[str]]) -> subprocess.CompletedProcess[str]:
            prefix, argv = job
            if not capture:
                return self.stream(argv, prefix)
            outcome = self.run(argv)
            if outcome.returncode:
                # Captured output goes nowhere, so a failure would be silent.
                emit(prefix + (said(outcome) or f"exit {outcome.returncode}"))
            return outcome

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            return list(pool.map(one, jobs))


def exit_code(outcomes: Iterable[subprocess.CompletedProcess[str]]) -> int:
    """The first failure, so a fan-out reports what went wrong."""
    return next((outcome.returncode for outcome in outcomes if outcome.returncode), 0)


@dataclasses.dataclass(frozen=True, slots=True)
class Tpu:
    """A named TPU in a zone, and the gcloud calls that reach it."""

    gcloud: Gcloud
    name: str
    zone: str
    user: str = ""

    @property
    def host(self) -> str:
        return f"{self.user}@{self.name}" if self.user else self.name

    def vm(self, *args: str) -> list[str]:
        return self.gcloud.argv("compute", "tpus", "tpu-vm", *args)

    def ssh_argv(
        self,
        worker: str = "0",
        command: str = "",
        ssh_flags: Sequence[str] = (),
    ) -> list[str]:
        args = ["ssh", self.host, f"--zone={self.zone}", f"--worker={worker}"]
        if command:
            args.append(f"--command={command}")
        args += [f"--ssh-flag={option}" for option in ssh_flags]
        return self.vm(*args)

    def scp_argv(self, sources: Sequence[str], target: str, worker: str = "0",
                 recurse: bool = False) -> list[str]:
        args = ["scp", *sources, target, f"--zone={self.zone}", f"--worker={worker}"]
        if recurse:
            args.append("--recurse")
        return self.vm(*args)

    def describe(self, timeout: float | None = None) -> Node | None:
        """This TPU as its zone lists it, asked for at most `timeout` seconds, or None where the zone
        has none of its name: not created yet, or deleted."""
        return next((node for node in self.gcloud.nodes(self.zone, timeout) if node.name == self.name), None)

    def ready(self, wait: float = READY_TIMEOUT) -> Node | None:
        """This TPU once its zone lists it READY, asked every POLL_SECONDS for up to `wait` seconds
        (none in a dry run): a queued resource lists its node only once it is provisioned."""
        if self.gcloud.dry_run:
            return None
        deadline = time.monotonic() + wait
        state = "PENDING"
        while True:
            # Each listing has what is left of the wait, so a hung gcloud cannot hold it past its end.
            try:
                node = self.describe(timeout=max(deadline - time.monotonic(), 0.0))
            except subprocess.TimeoutExpired:
                raise SystemExit(f"{self.name} is still {state} after {wait}s; gcloud did not "
                                 "answer within the wait") from None
            state = node.state if node else "PENDING"
            if state == "READY":
                return node
            if state in DEAD_STATES:
                raise SystemExit(f"{self.name} is {state}")
            if time.monotonic() >= deadline:
                raise SystemExit(f"{self.name} is still {state} after {wait}s")
            emit(f"{self.name} is {state}")
            time.sleep(POLL_SECONDS)

    def fanout(self, jobs: Sequence[tuple[int, str]], *, capture: bool = False,
               ssh_flags: Sequence[str] = ()) -> list[subprocess.CompletedProcess[str]]:
        """Run one remote command per worker, prefixing output with the worker."""
        prepared = [
            (f"[worker {index}] ", self.ssh_argv(str(index), command, ssh_flags))
            for index, command in jobs
        ]
        return self.gcloud.fanout(prepared, capture=capture)
