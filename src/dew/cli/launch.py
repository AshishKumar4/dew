"""dew launch: start one program as a jax.distributed process pool.

Every process of a pool runs the same program and joins through
`dew.training.runtime.prepare_process`, which calls
`jax.distributed.initialize`. That call finds its coordinator, the process
count and its own rank in the environment. A TPU VM, a Slurm step and an
Open MPI launch leave those in variables jax reads by itself. A plain set of
machines leaves nothing, and this command is what fills the gap: it starts
the processes over ssh, or directly for the machine it runs on, with the
three values in the variables `dew.pool` names.

Under Slurm, `--slurm` hands the launch to `srun` and jax reads the rank
from Slurm's own variables, so the same program runs unchanged either way.

Nothing here imports an array library, like the rest of the package.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import types
from collections.abc import Sequence
from typing import Annotated, TextIO

import tyro

from dew.cli.gcloud import emit
from dew.pool import COORDINATOR, LOCAL_DEVICES, PROCESS_COUNT, PROCESS_ID

VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""What an `--env` name may be: it lands unquoted in the remote shell line."""
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
POLL_SECONDS = 0.2

Positional = tyro.conf.Positional
Hosts = Annotated[tuple[str, ...], tyro.conf.arg(metavar="HOST")]
Variables = Annotated[tuple[str, ...], tyro.conf.arg(metavar="NAME=VALUE"), tyro.conf.UseAppendAction]
Address = Annotated[str | None, tyro.conf.arg(metavar="HOST")]
Directory = Annotated[str | None, tyro.conf.arg(metavar="DIR")]
Count = Annotated[int | None, tyro.conf.arg(metavar="N")]


@dataclasses.dataclass(frozen=True)
class Process:
    """One member of the pool: where it runs and the argv that starts it."""

    rank: int
    host: str
    argv: tuple[str, ...]
    env: dict[str, str]


@dataclasses.dataclass(frozen=True)
class Launch:
    """Run a program on every host as one jax.distributed process pool."""

    command: Positional[tuple[str, ...]]
    """The program and its arguments, the same on every host. Put `--` before
    it so its own flags stay its own."""
    hosts: Hosts = ("localhost",)
    """The machines, process 0's first. localhost runs here without ssh;
    every other name is reached with `ssh -o BatchMode=yes`, so keys must
    already be in place. The remote shell reads no login profile, so give
    the program as an absolute path, or its PATH through `--env`."""
    processes_per_host: int = 1
    """Processes each host runs. One per host is the usual layout; one per
    GPU needs `devices_per_process` so each takes its own devices."""
    devices_per_process: Count = None
    """Accelerators each process takes, counted from 0 on its host in rank
    order, through JAX_LOCAL_DEVICE_IDS. Unset leaves every process every
    local device, which is right for one process per host. Not for
    `--slurm`, where jax assigns one GPU per task itself."""
    port: int = 43217
    """The coordinator's port on process 0's host."""
    coordinator: Address = None
    """The address the other hosts reach process 0's host at; unset takes
    the first host, or this machine's name when that is localhost and other
    hosts are listed."""
    env: Variables = ()
    """Variables every process gets, beside the pool's own."""
    cwd: Directory = None
    """The directory each process starts in; unset is this one, which must
    then exist on every host at the same path."""
    slurm: bool = False
    """Launch with srun inside a Slurm allocation: `processes_per_host`
    tasks on every allocated node, ranks from Slurm's variables. jax gives
    each task the one GPU at its SLURM_LOCALID, so `processes_per_host` is
    the GPUs a node has."""
    dry_run: bool = False
    """Print the commands that would run, then exit."""

    def __post_init__(self):
        if not self.command:
            raise ValueError("dew launch needs the program to run, after --")
        if self.processes_per_host < 1:
            raise ValueError(f"processes_per_host must be positive, got {self.processes_per_host}")
        if self.devices_per_process is not None and self.devices_per_process < 1:
            raise ValueError(
                f"devices_per_process must be positive, got {self.devices_per_process}")
        for variable in self.env:
            name = variable.split("=", 1)[0]
            if "=" not in variable or not VARIABLE_NAME.fullmatch(name):
                raise ValueError(
                    f"--env takes NAME=VALUE with a shell variable name "
                    f"([A-Za-z_][A-Za-z0-9_]*), got {variable!r}")
        if self.slurm and (self.hosts != ("localhost",) or self.coordinator is not None):
            raise ValueError("--slurm takes its hosts and coordinator from the allocation")
        if self.slurm and self.devices_per_process is not None:
            raise ValueError(
                "--slurm runs one task per GPU: jax's Slurm detection gives each task "
                "the GPU at its SLURM_LOCALID. Set --processes-per-host to the GPUs a "
                "node has and leave --devices-per-process unset.")

    def extra_env(self) -> dict[str, str]:
        return dict(variable.split("=", 1) for variable in self.env)

    def coordinator_address(self) -> str:
        host = self.coordinator or self.hosts[0]
        if host in LOCAL_HOSTS and any(name not in LOCAL_HOSTS for name in self.hosts):
            host = socket.getfqdn()
        return f"{host}:{self.port}"

    def pool(self) -> list[Process]:
        """Every process, in rank order: host by host, then within a host."""
        count = len(self.hosts) * self.processes_per_host
        shared = {**self.extra_env(), COORDINATOR: self.coordinator_address(),
                  PROCESS_COUNT: str(count)}
        processes = []
        for rank in range(count):
            host = self.hosts[rank // self.processes_per_host]
            env = {**shared, PROCESS_ID: str(rank)}
            if self.devices_per_process is not None:
                first = (rank % self.processes_per_host) * self.devices_per_process
                env[LOCAL_DEVICES] = ",".join(
                    str(device) for device in range(first, first + self.devices_per_process))
            processes.append(Process(rank, host, self.remote_argv(host, env), env))
        return processes

    def remote_argv(self, host: str, env: dict[str, str]) -> tuple[str, ...]:
        """The argv that starts the command on `host`: the command itself
        here, and anywhere else the user's shell over ssh, which is not a
        login shell, so profile setup such as a virtualenv does not run and
        the command's interpreter is best given as an absolute path."""
        if host in LOCAL_HOSTS:
            return self.command
        cwd = self.cwd or os.getcwd()
        assignments = " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items())
        script = f"cd {shlex.quote(cwd)} && exec env {assignments} {shlex.join(self.command)}"
        # -tt gives the remote command a terminal, so closing the connection,
        # which stopping the pool does, delivers SIGHUP to it.
        return ("ssh", "-tt", "-o", "BatchMode=yes", host, script)

    def srun_argv(self) -> tuple[str, ...]:
        """srun's argv. The `--env` variables travel in its environment,
        which --export=ALL hands to every task: Slurm reads --export itself
        as a comma-separated list, and would cut a value at its commas."""
        argv = ["srun", f"--ntasks-per-node={self.processes_per_host}",
                "--kill-on-bad-exit=1", "--export=ALL"]
        if self.cwd is not None:
            argv.append(f"--chdir={self.cwd}")
        return (*argv, *self.command)

    def run_command(self) -> int:
        if self.slurm:
            argv = self.srun_argv()
            if self.dry_run:
                emit(" ".join([*(f"{name}={shlex.quote(value)}"
                                 for name, value in self.extra_env().items()), shlex.join(argv)]))
                return 0
            os.execvpe(argv[0], argv, {**os.environ, **self.extra_env()})
        processes = self.pool()
        if self.dry_run:
            for process in processes:
                local = process.host in LOCAL_HOSTS
                prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in process.env.items())
                emit(f"{prefix} {shlex.join(process.argv)}" if local else shlex.join(process.argv))
            return 0
        return supervise(processes, self.cwd)


def _relay(rank: int, stream: TextIO, out: TextIO, lock: threading.Lock) -> None:
    for line in stream:
        with lock:
            out.write(f"[{rank}] {line.rstrip()}\n")
            out.flush()


def _stop(running: Sequence[subprocess.Popen]) -> None:
    """SIGTERM every process group still alive, then SIGKILL after 10 s."""
    for child in running:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for child in running:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


class Stopped(BaseException):
    """The launcher was told to stop by a signal; `code` is what it exits."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.code = 128 + signum


def _raise_stopped(signum: int, _frame: types.FrameType | None) -> None:
    raise Stopped(signum)


def supervise(processes: Sequence[Process], cwd: str | None) -> int:
    """Run the pool with each line prefixed by its rank, and return the
    first failure's exit code, or 0. A rank killed by signal N counts as
    128 + N, the way a shell reports it.

    A process that fails leaves its peers waiting in a collective for a
    partner that is gone, which on most backends is a hang rather than an
    error. So the first non-zero exit stops the rest.

    Every rank runs in a session of its own, so no signal meant for the
    launcher's terminal or process group reaches it. The launcher stops the
    pool itself on SIGINT, SIGTERM (a scheduler's cancel) and SIGHUP (its
    terminal or ssh session closing), and returns 128 plus the signal, as a
    shell reports it. Any other exception stops the pool before it
    propagates.
    """
    lock = threading.Lock()
    running: list[subprocess.Popen] = []
    relays = []
    handlers = {signum: signal.signal(signum, _raise_stopped)
                for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        for process in processes:
            local = process.host in LOCAL_HOSTS
            child = subprocess.Popen(
                process.argv, cwd=cwd if local else None,
                env={**os.environ, **process.env} if local else None,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", start_new_session=True)
            running.append(child)
            assert child.stdout is not None, "stdout=PIPE always opens the stream"
            relay = threading.Thread(target=_relay, args=(process.rank, child.stdout, sys.stdout, lock),
                                     daemon=True)
            relay.start()
            relays.append(relay)
        while True:
            codes = [child.poll() for child in running]
            failed = [code for code in codes if code not in (None, 0)]
            if failed:
                emit(f"a process exited {failed[0]}; stopping the pool")
                _stop(running)
                # Popen reports death by signal N as -N; a shell reports 128 + N.
                return 128 - failed[0] if failed[0] < 0 else failed[0]
            if all(code == 0 for code in codes):
                return 0
            time.sleep(POLL_SECONDS)
    except BaseException as error:
        # A second signal while the pool stops would leave it half stopped.
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        _stop(running)
        if isinstance(error, Stopped):
            return error.code
        raise
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)
        for relay in relays:
            relay.join(timeout=5)
