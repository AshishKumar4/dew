"""The one seam between dew-tpu and the outside world.

Every subprocess dew-tpu starts is built, printed and run here. A dry run
prints the command and returns success without touching gcloud.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import shlex
import subprocess
import sys
import threading
from collections.abc import Iterable, Sequence

#: The binary every command in this module goes through.
GCLOUD = "gcloud"

_PRINT = threading.Lock()


def emit(line: str) -> None:
    """Print one whole line, even when workers report at the same time."""
    with _PRINT:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


@dataclasses.dataclass(frozen=True, slots=True)
class Result:
    """What a finished subprocess left behind."""

    argv: tuple[str, ...]
    code: int
    out: str = ""
    err: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def text(self) -> str:
        return (self.out + self.err).strip()


@dataclasses.dataclass(frozen=True, slots=True)
class Node:
    """The parts of a describe response that dew-tpu reads."""

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

    def run(self, argv: Sequence[str], *, capture: bool = True, check: bool = False) -> Result:
        """Run one command. A dry run prints it and reports success."""
        argv = list(argv)
        if self.dry_run:
            emit(shlex.join(argv))
            return Result(tuple(argv), 0)
        if capture:
            done = subprocess.run(argv, capture_output=True, text=True)
            result = Result(tuple(argv), done.returncode, done.stdout, done.stderr)
        else:
            result = Result(tuple(argv), subprocess.call(argv))
        if check and not result.ok:
            raise SystemExit(result.text or f"{argv[0]} failed with code {result.code}")
        return result

    def json(self, *args: str, default: object = None) -> object:
        """Run a gcloud command that speaks JSON and parse it."""
        result = self.run(self.argv(*args, "--format=json"))
        if not result.ok or not result.out.strip():
            return default
        return json.loads(result.out)

    def config_value(self, name: str) -> str:
        """Read one value from the local gcloud config.

        Runs in a dry run too: it reads and changes nothing, and the argv a dry
        run prints can depend on the answer.
        """
        done = subprocess.run([GCLOUD, "config", "get-value", name],
                              capture_output=True, text=True)
        value = done.stdout.strip()
        return "" if done.returncode or value == "(unset)" else value

    def stream(self, argv: Sequence[str], prefix: str = "") -> Result:
        """Run one command, tagging every output line with a prefix."""
        argv = list(argv)
        if self.dry_run:
            emit(shlex.join(argv))
            return Result(tuple(argv), 0)
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:  # type: ignore[union-attr]
            emit(prefix + line.rstrip("\n"))
        return Result(tuple(argv), process.wait())

    def fanout(
        self,
        jobs: Sequence[tuple[str, Sequence[str]]],
        *,
        capture: bool = False,
    ) -> list[Result]:
        """Run one command per worker at the same time, in job order."""
        if not jobs:
            return []
        if self.dry_run:
            return [self.run(argv) for _, argv in jobs]

        def one(job: tuple[str, Sequence[str]]) -> Result:
            prefix, argv = job
            if not capture:
                return self.stream(argv, prefix)
            result = self.run(argv)
            if not result.ok:
                # Captured output goes nowhere, so a failure would be silent.
                emit(prefix + (result.text or f"exit {result.code}"))
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            return list(pool.map(one, jobs))


def exit_code(results: Iterable[Result]) -> int:
    """The first failure, so a fan-out reports what went wrong."""
    return next((result.code for result in results if not result.ok), 0)


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
        args += [f"--ssh-flag={flag}" for flag in ssh_flags]
        return self.vm(*args)

    def scp_argv(self, sources: Sequence[str], target: str, worker: str = "0",
                 recurse: bool = False) -> list[str]:
        args = ["scp", *sources, target, f"--zone={self.zone}", f"--worker={worker}"]
        if recurse:
            args.append("--recurse")
        return self.vm(*args)

    def describe(self) -> Node | None:
        payload = self.gcloud.json(
            "compute", "tpus", "tpu-vm", "describe", self.name, f"--zone={self.zone}")
        return Node.parse(payload) if isinstance(payload, dict) and payload else None

    def fanout(self, jobs: Sequence[tuple[int, str]], *, capture: bool = False,
               ssh_flags: Sequence[str] = ()) -> list[Result]:
        """Run one remote command per worker, prefixing output with the worker."""
        prepared = [
            (f"[worker {index}] ", self.ssh_argv(str(index), command, ssh_flags))
            for index, command in jobs
        ]
        return self.gcloud.fanout(prepared, capture=capture)
