"""dew launch: start one program as a jax.distributed process pool.

Every process of a pool runs the same program and joins through
`dew.training.runtime.prepare_process`, which calls
`jax.distributed.initialize`. That call finds the coordinator, the process
count and its own rank in the environment. A Slurm step, an Open MPI rank, a
Cloud TPU VM and a Kubernetes job leave those where jax's own cluster
detection reads them, and this command then only runs the program. A plain
set of machines leaves nothing, and this command fills the gap: it starts the
processes over ssh, or directly for the machine it runs on, with the three
values in the variables `dew.pool` names. A Cloud TPU named with `--tpu` is
reached through gcloud, one process per worker, and jax reads its metadata.

The launcher imports jax only to ask its cluster detection, and only when no
target is named.
"""

from __future__ import annotations

import collections
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
from pathlib import Path
from typing import Annotated, TextIO

import tyro

from dew.cli import tpu_setup
from dew.cli.gcloud import emit
from dew.cli.tpu import reach
from dew.pool import (
    COORDINATOR,
    LOCAL_DEVICES,
    PREEMPTED_EXIT,
    PROCESS_COUNT,
    PROCESS_ID,
    Cluster,
    detected_cluster,
    listed_gpus,
    local_gpu_count,
    refuse_idle_gpus,
    runs_on_gpu,
    slurm_tasks_here,
    visible_gpus,
)

VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""What an `--env` name may be: it lands unquoted in the remote shell line."""
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
POLL_SECONDS = 0.2
TAIL_LINES = 20
FAILURE_GRACE = 10.0
"""Seconds the other ranks of a pool whose rank failed get between SIGTERM and
SIGKILL. They wait in a collective for the failed rank, so none reaches a
preemption checkpoint, and it is the SIGKILL that stops a JAX rank."""
PREEMPTION_GRACE = 300.0
"""Seconds the ranks of a pool the launcher was signalled to stop get between
SIGTERM and SIGKILL: time for `Trainer.fit` to reach the step every rank
agrees on and write its checkpoint and data position. A scheduler's own
grace, often shorter, still ends the pool, and a second signal stops it at
once."""
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
"""Lines of a failed rank's output printed again once the pool has stopped."""
MEGASCALE_PORT = "8081"
"""The port slices of a multislice run meet on, the one Ray's TPU support uses."""

Positional = tyro.conf.Positional
Names = Annotated[tuple[str, ...], tyro.conf.arg(metavar="NAME")]
Variables = Annotated[tuple[str, ...], tyro.conf.arg(metavar="NAME=VALUE"), tyro.conf.UseAppendAction]
Address = Annotated[str | None, tyro.conf.arg(metavar="HOST")]
Directory = Annotated[str | None, tyro.conf.arg(metavar="DIR")]
File = Annotated[Path | None, tyro.conf.arg(metavar="FILE")]
Zone = Annotated[str | None, tyro.conf.arg(metavar="ZONE")]
Count = Annotated[int | None, tyro.conf.arg(metavar="N")]


@dataclasses.dataclass(frozen=True)
class Process:
    """One member of the pool: where it runs and the argv that starts it."""

    rank: int
    host: str
    argv: tuple[str, ...]
    env: dict[str, str]
    devices: str = ""
    """The local devices it takes, as JAX_LOCAL_DEVICE_IDS lists them."""

    @property
    def local(self) -> bool:
        return self.host in LOCAL_HOSTS


@dataclasses.dataclass(frozen=True)
class Launch:
    """Run a program on every accelerator of a cluster, as one jax.distributed pool.

    dew launch -- python train.py

    dew launch --hosts node0,node1 -- /opt/dew/.venv/bin/python train.py

    dew launch --tpu my-v5e-16 --cwd dew -- python recipes/lm/train.py

    Without --hosts, --hostfile or --tpu it runs the program in place where a
    cluster already started several ranks (a Slurm step, mpirun, a TPU pod
    worker, a Kubernetes job); starts it with srun inside a Slurm
    allocation; and otherwise runs one process per GPU on this machine.
    """

    command: Positional[tuple[str, ...]]
    """The program and its arguments; put `--` before it."""
    hosts: Names = ()
    """Machines to run on over ssh, process 0's first, separated by commas or spaces."""
    hostfile: File = None
    """A file of machines to run on, one per line."""
    tpu: Names = ()
    """Cloud TPU VMs to run on, every worker of each; several names make one multislice pool."""
    zone: Zone = None
    """Zone of the --tpu VMs; found from the dew tpu config when unset."""
    processes_per_host: Count = None
    """Processes on each host; unset is one per GPU, or one when a host has no more than one GPU."""
    devices_per_process: Count = None
    """GPUs each process takes; unset splits a host's GPUs evenly between its processes."""
    port: Count = None
    """Coordinator port on process 0's host; unset takes a free one."""
    coordinator: Address = None
    """Address the other hosts reach process 0's host at; unset is its name."""
    env: Variables = ()
    """A variable every process gets; repeat for more."""
    cwd: Directory = None
    """Directory every process starts in; unset is this one, or the home directory on --tpu."""
    dry_run: bool = False
    """Print what would run, then exit."""

    def __post_init__(self):
        if not self.command:
            raise ValueError("dew launch needs the program to run, after --")
        for name, value in (("processes_per_host", self.processes_per_host),
                            ("devices_per_process", self.devices_per_process)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        for variable in self.env:
            name = variable.split("=", 1)[0]
            if "=" not in variable or not VARIABLE_NAME.fullmatch(name):
                raise ValueError(
                    f"--env takes NAME=VALUE with a shell variable name "
                    f"([A-Za-z_][A-Za-z0-9_]*), got {variable!r}")
        if self.hosts and self.hostfile is not None:
            raise ValueError("name the hosts with --hosts or with --hostfile, not both")
        if self.tpu:
            given = [option for option, value in (
                ("--hosts", self.hosts), ("--hostfile", self.hostfile),
                ("--processes-per-host", self.processes_per_host),
                ("--devices-per-process", self.devices_per_process),
                ("--port", self.port), ("--coordinator", self.coordinator)) if value]
            if given:
                raise ValueError(
                    f"--tpu runs one process on every worker and jax finds the coordinator "
                    f"itself; drop {', '.join(given)}")
        elif self.zone is not None:
            raise ValueError("--zone names the zone of the --tpu VMs")

    def extra_env(self) -> dict[str, str]:
        return dict(variable.split("=", 1) for variable in self.env)

    def host_names(self) -> tuple[str, ...]:
        """The --hosts names split at commas, or the first word of each
        hostfile line, so an MPI hostfile's `slots=` is ignored."""
        if self.hostfile is not None:
            lines = (line.split("#", 1)[0].split() for line in self.hostfile.read_text().splitlines())
            return tuple(words[0] for words in lines if words)
        return split_names(self.hosts)

    def run_command(self) -> int:
        if self.tpu:
            return self.start(self.tpu_pool())
        hosts = self.host_names()
        if self.hostfile is not None and not hosts:
            raise ValueError(f"{self.hostfile} names no hosts")
        if hosts:
            return self.start(self.pool(hosts))
        cluster = detected_cluster()
        if cluster is not None and cluster.count > 1:
            return self.in_place(cluster)
        if cluster is None and "SLURM_JOB_ID" in os.environ:
            return self.srun()
        # Nothing placed a pool here, or a cluster placed this one process
        # alone, as a one-task Slurm step does: this machine's GPUs are the
        # launcher's to share out.
        return self.start(self.pool(("localhost",)))

    def pool_flags(self) -> list[str]:
        """The flags that shape a pool dew launch starts itself."""
        return [option for option, value in (
            ("--processes-per-host", self.processes_per_host),
            ("--devices-per-process", self.devices_per_process),
            ("--port", self.port), ("--coordinator", self.coordinator)) if value is not None]

    def in_place(self, cluster: Cluster) -> int:
        """Run the program as this process: whatever started it on every
        node already set what jax reads. A Slurm step with fewer tasks on
        this node than the GPUs its task sees is refused, as srun is."""
        refused = self.pool_flags()
        if refused:
            raise ValueError(f"{cluster.name} already placed this process; jax reads the pool "
                             f"from it, so drop {', '.join(refused)}")
        tasks = slurm_tasks_here()
        if cluster.name == "slurm" and tasks is not None and runs_on_gpu(os.environ):
            refuse_idle_gpus(tasks, local_gpu_count(), "this slurm step")
        emit(f"{cluster.name}: process {cluster.process} of {cluster.count}, placed by the "
             f"cluster; running {shlex.join(self.command)}")
        if cluster.count > 1 and cluster.process == 0 and cluster.name in ("gcetpu", "gketpu"):
            emit(f"every worker has to run it; `dew launch --tpu NAME` starts all {cluster.count}")
        return self.execute(self.command)

    def srun(self) -> int:
        """Start the program with srun in this Slurm allocation. jax gives
        each Slurm task the one GPU at its SLURM_LOCALID, so a node runs a
        task per GPU: `--processes-per-host`, else the allocation's own
        tasks per node, else its GPUs per node, else the GPUs this node
        sees, which in a batch step are the ones Slurm gave it. Fewer tasks
        a node than GPUs is refused: the rest would sit idle."""
        if self.devices_per_process is not None or self.port is not None \
                or self.coordinator is not None:
            raise ValueError(
                "under Slurm jax gives each task the GPU at its SLURM_LOCALID and picks the "
                "coordinator; use --processes-per-host for the GPUs a node has, and drop "
                "--devices-per-process, --port and --coordinator")
        allocation = f"slurm allocation {os.environ['SLURM_JOB_ID']}"
        gpus: int | None = 0
        if runs_on_gpu({**os.environ, **self.extra_env()}):
            per_node = os.environ.get("SLURM_GPUS_PER_NODE")
            # --gpus-per-node reads `[type:]count`, several types separated by commas.
            gpus = (sum(int(part.rsplit(":", 1)[-1]) for part in per_node.split(","))
                    if per_node else local_gpu_count())
        tasks = self.processes_per_host
        if tasks is None and "SLURM_NTASKS_PER_NODE" in os.environ:
            refuse_idle_gpus(int(os.environ["SLURM_NTASKS_PER_NODE"]), gpus, allocation)
        elif tasks is not None:
            refuse_idle_gpus(tasks, gpus, allocation)
        else:
            tasks = gpus or None
        argv = ["srun", "--kill-on-bad-exit=1", "--export=ALL", "--label"]
        if tasks is not None:
            argv.append(f"--ntasks-per-node={tasks}")
        emit(f"{allocation}: starting the program with srun")
        return self.execute((*argv, *self.command))

    def execute(self, argv: Sequence[str]) -> int:
        """Replace this process with `argv`, with the `--env` variables and
        in `--cwd`. srun hands its environment to every task with
        --export=ALL, so values holding commas, which Slurm's own --export
        list would cut, arrive whole."""
        extra = self.extra_env()
        if not self.dry_run:
            if self.cwd is not None:
                os.chdir(self.cwd)
            os.execvpe(argv[0], list(argv), {**os.environ, **extra})
        prefix = [f"cd {shlex.quote(self.cwd)} &&"] if self.cwd else []
        emit(" ".join([*prefix, *(f"{name}={shlex.quote(value)}" for name, value in extra.items()),
                       shlex.join(argv)]))
        return 0

    def coordinator_address(self, hosts: Sequence[str]) -> str:
        host = self.coordinator or hosts[0]
        if host in LOCAL_HOSTS and any(name not in LOCAL_HOSTS for name in hosts):
            host = socket.getfqdn()
        return f"{host}:{self.port if self.port is not None else free_port(hosts[0])}"

    def layout(self, first_host: str) -> tuple[int, int | None]:
        """Processes per host and GPUs per process. A host's GPUs are
        counted on the first host, when the flags leave them to decide or
        name some, and not at all when JAX_PLATFORMS keeps the pool off GPUs,
        as a CPU rehearsal on a GPU machine does. A pool that names more GPUs
        than the host shows is refused before any rank starts. GPUs are named
        only where several processes split a host's: one process holds every
        GPU and names none, and prepare_process keeps a cluster's detection
        from narrowing a pool the launcher placed."""
        processes, devices = self.processes_per_host, self.devices_per_process
        if processes == 1 and devices is None:
            return processes, devices
        # What the pool's processes will see: the --env values, and on this
        # machine the launcher's own environment, which ssh does not carry.
        env = {**(os.environ if first_host in LOCAL_HOSTS else {}), **self.extra_env()}
        on_gpu = runs_on_gpu(env)
        # None where the GPUs could not be counted, which is not zero of them.
        gpus = gpu_count(first_host, env.get("CUDA_VISIBLE_DEVICES")) if on_gpu else 0
        if processes is None:
            processes = max(1, (gpus or 0) // (devices or 1))
        # More processes than GPUs, as in a rehearsal of programs that do
        # not use them, leaves every process every GPU.
        if devices is None and gpus is not None and 1 < processes <= gpus:
            if gpus % processes:
                raise ValueError(f"{processes} processes do not split {first_host}'s {gpus} "
                                 "GPUs evenly; set --devices-per-process")
            devices = gpus // processes
        if not on_gpu or devices is None:
            return processes, devices
        if gpus is None:
            sys.stderr.write(
                f"could not count {first_host}'s GPUs (nvidia-smi is missing there or failed); "
                f"starting {processes} process{'es' if processes > 1 else ''} with {devices} "
                f"GPU{'s' if devices > 1 else ''} each without checking that they exist\n")
        elif processes * devices > gpus:
            raise ValueError(
                f"a pool of {processes} process{'es' if processes > 1 else ''} with {devices} "
                f"GPU{'s' if devices > 1 else ''} each needs {processes * devices} GPUs on "
                f"{first_host}, which shows {gpus}; ask for fewer with --processes-per-host "
                f"or --devices-per-process, or keep the pool off GPUs with "
                f"--env JAX_PLATFORMS=cpu")
        return processes, devices

    def pool(self, hosts: Sequence[str]) -> list[Process]:
        """Every process, in rank order: host by host, then within a host."""
        per_host, devices = self.layout(hosts[0])
        count = len(hosts) * per_host
        address = self.coordinator_address(hosts)
        shared = {**self.extra_env(), COORDINATOR: address, PROCESS_COUNT: str(count)}
        if devices is not None:
            holds = f"{devices} GPU{'s' if devices > 1 else ''} each"
        else:
            holds = "each with every local device" if per_host > 1 else "with every local device"
        emit(f"pool: {count} process{'es' if count > 1 else ''} on {', '.join(hosts)}, "
             f"{holds}, coordinator {address}")
        processes = []
        for rank in range(count):
            host = hosts[rank // per_host]
            env = {**shared, PROCESS_ID: str(rank)}
            if devices is not None:
                first = (rank % per_host) * devices
                env[LOCAL_DEVICES] = ",".join(str(device) for device in range(first, first + devices))
            processes.append(Process(rank, host, self.remote_argv(host, env), env,
                                     env.get(LOCAL_DEVICES, "")))
        return processes

    def remote_argv(self, host: str, env: dict[str, str]) -> tuple[str, ...]:
        """The argv that starts the command on `host`: the command itself
        here, and anywhere else the user's shell over ssh, which is not a
        login shell, so profile setup such as a virtualenv does not run and
        the command's interpreter is best given as an absolute path."""
        if host in LOCAL_HOSTS:
            return self.command
        script = remote_script(self.command, env, self.cwd or os.getcwd())
        # -tt gives the remote command a terminal, so closing the connection,
        # which stopping the pool does, delivers SIGHUP to it.
        return ("ssh", "-tt", "-o", "BatchMode=yes", host, script)

    def tpu_pool(self) -> list[Process]:
        """One process per worker of every named TPU, each started over
        gcloud ssh in a shell that sources the environment `dew tpu setup`
        wrote. jax reads each worker's rank from the TPU metadata; several
        slices also get the MEGASCALE variables that join them."""
        slices = [reach(name, self.zone, self.dry_run) for name in split_names(self.tpu)]
        processes = []
        for slice_id, (tpu, addresses) in enumerate(slices):
            env = self.extra_env()
            if len(slices) > 1:
                env |= {"MEGASCALE_COORDINATOR_ADDRESS": slices[0][1][0],
                        "MEGASCALE_PORT": MEGASCALE_PORT,
                        "MEGASCALE_NUM_SLICES": str(len(slices)),
                        "MEGASCALE_SLICE_ID": str(slice_id)}
            script = tpu_setup.wrap(remote_script(self.command, env, self.cwd))
            for worker in range(len(addresses)):
                argv = tpu.ssh_argv(str(worker), script, ("-tt",))
                processes.append(Process(len(processes), f"{tpu.name} worker {worker}",
                                         tuple(argv), env))
        emit(f"pool: {len(processes)} process{'es' if len(processes) > 1 else ''} on "
             f"{', '.join(tpu.name for tpu, _ in slices)}, one per worker")
        return processes

    def start(self, processes: Sequence[Process]) -> int:
        if not self.dry_run:
            return supervise(processes, self.cwd)
        for process in processes:
            prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in process.env.items())
            emit(f"{prefix} {shlex.join(process.argv)}" if process.local
                 else shlex.join(process.argv))
        return 0


def split_names(entries: Sequence[str]) -> tuple[str, ...]:
    """Names given as `a,b`, `a b` or both."""
    return tuple(name for entry in entries for name in entry.split(",") if name)


def remote_script(command: Sequence[str], env: dict[str, str], cwd: str | None) -> str:
    """The shell line that runs `command` with `env` in `cwd` on another
    machine; a relative `cwd` is under the login directory."""
    assignments = "".join(f"{name}={shlex.quote(value)} " for name, value in env.items())
    run = f"exec env {assignments}{shlex.join(command)}" if env else f"exec {shlex.join(command)}"
    return f"cd {shlex.quote(cwd)} && {run}" if cwd else run


def gpu_count(host: str, visible: str | None) -> int | None:
    """GPUs a pool process on `host` may use: those `visible` lists when
    the pool sets CUDA_VISIBLE_DEVICES itself, else what `host` shows,
    asked over ssh when it is another machine; None when they could not be
    counted there."""
    if visible is not None:
        return visible_gpus(visible)
    if host in LOCAL_HOSTS:
        return local_gpu_count()
    return listed_gpus(("ssh", "-o", "BatchMode=yes", host, "nvidia-smi -L"))


def free_port(host: str) -> int:
    """A TCP port nothing listens on at `host` now: port 0 bound there, which
    has the kernel pick one, and released. Another host is asked over ssh,
    with the python3 its shell finds."""
    if host in LOCAL_HOSTS:
        with socket.socket() as probe:
            probe.bind(("", 0))
            return probe.getsockname()[1]
    probe = "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])"
    found = subprocess.run(("ssh", "-o", "BatchMode=yes", host, f"python3 -c {shlex.quote(probe)}"),
                           capture_output=True, text=True, timeout=60)
    if found.returncode != 0 or not found.stdout.strip().isdigit():
        raise ValueError(f"no free port found on {host} with python3 over ssh: "
                         f"{found.stderr.strip()[-300:]}; name one with --port")
    return int(found.stdout)


def _relay(rank: int, stream: TextIO, tail: collections.deque[str]) -> None:
    for line in stream:
        text = line.rstrip()
        tail.append(text)
        emit(f"[{rank}] {text}")


def _stop(running: Sequence[subprocess.Popen], grace: float, forced: threading.Event) -> None:
    """SIGTERM every process group still alive, give them `grace` seconds to
    exit, or until `forced` is set, and SIGKILL those still alive, rank 0
    last.

    A JAX process of a pool takes SIGTERM as a preemption notice
    (jax_enable_preemption_service): `Trainer.fit` checkpoints at the step
    every rank agrees on and exits, and any other program runs on, so the
    SIGKILL is what stops it. Rank 0's process holds the pool's coordination
    service, and a rank that outlives it aborts in XLA's error polling, with
    a Check failure that reads as a crash of its own.
    """
    for child in running:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while (not forced.is_set() and time.monotonic() < deadline
           and any(child.poll() is None for child in running)):
        time.sleep(POLL_SECONDS)
    for child in (*running[1:], *running[:1]):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


class Stopped(BaseException):
    """The launcher was told to stop by a signal; `code` is what it exits."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.code = 128 + signum


def _raise_stopped(signum: int, _frame: types.FrameType | None) -> None:
    raise Stopped(signum)


def _escalate(forced: threading.Event) -> None:
    """Make every stop signal from here set `forced`, which ends a stop's
    grace, instead of raising into the launcher."""
    for signum in STOP_SIGNALS:
        signal.signal(signum, lambda _signum, _frame: forced.set())


def _exit_text(code: int) -> str:
    """How a Popen return code reads: a signal by name, else the code."""
    return f"was killed by {signal.Signals(-code).name}" if code < 0 else f"exited {code}"


def supervise(processes: Sequence[Process], cwd: str | None) -> int:
    """Run the pool with each line prefixed by its rank, and return the
    first failure's exit code, or 0. A rank killed by signal N counts as
    128 + N, the way a shell reports it.

    A process that fails leaves its peers waiting in a collective for a
    partner that is gone, which on most backends is a hang rather than an
    error. So the first non-zero exit stops the rest, and the failed rank's
    last lines are printed again after the others' shutdown output.

    Every rank runs in a session of its own, so no signal meant for the
    launcher's terminal or process group reaches it. The launcher stops the
    pool itself on SIGINT, SIGTERM (a scheduler's cancel) and SIGHUP (its
    terminal or ssh session closing), giving the ranks PREEMPTION_GRACE to
    checkpoint where a training run agrees, or until a second signal, and
    returns 128 plus the signal, as a shell reports it. A rank that exits
    with PREEMPTED_EXIT stopped at such a checkpoint, and the others get the
    same grace. Any other exception stops the pool before it propagates.
    """
    running: list[subprocess.Popen] = []
    relays: list[threading.Thread] = []
    tails = [collections.deque[str](maxlen=TAIL_LINES) for _ in processes]
    handlers = {signum: signal.signal(signum, _raise_stopped) for signum in STOP_SIGNALS}
    forced = threading.Event()
    try:
        for process in processes:
            child = subprocess.Popen(
                process.argv, cwd=cwd if process.local else None,
                # Unbuffered, so a local Python rank's prints reach the relay
                # when they happen and in order with its warnings; a remote
                # rank writes to the terminal ssh -tt gives it, which Python
                # line-buffers already.
                env={"PYTHONUNBUFFERED": "1", **os.environ, **process.env} if process.local
                else None,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", start_new_session=True)
            running.append(child)
            stdout = child.stdout
            if stdout is None:
                raise RuntimeError(f"{process.argv[0]} opened without the pipe it was asked for")
            devices = f", GPU {process.devices}" if process.devices else ""
            emit(f"[{process.rank}] rank {process.rank} of {len(processes)} on "
                 f"{process.host}{devices}, pid {child.pid}")
            relay = threading.Thread(target=_relay, args=(process.rank, stdout, tails[process.rank]),
                                     daemon=True)
            relay.start()
            relays.append(relay)
        while True:
            codes = [child.poll() for child in running]
            failed = next(((rank, code) for rank, code in enumerate(codes) if code), None)
            if failed is not None:
                rank, code = failed
                others = sum(1 for other, returned in enumerate(codes) if other != rank and returned is None)
                # A rank preempted at the step the pool agreed on leaves the
                # others writing the same checkpoint.
                preempted = code == PREEMPTED_EXIT
                emit(f"rank {rank} on {processes[rank].host} "
                     + ("stopped at a preemption checkpoint" if preempted else _exit_text(code))
                     + (f"; stopping the other {others}" if others else ""))
                _escalate(forced)
                _stop(running, PREEMPTION_GRACE if preempted else FAILURE_GRACE, forced)
                for relay in relays:
                    relay.join(timeout=5)
                emit(f"last lines of rank {rank}:")
                for line in tails[rank]:
                    emit(f"[{rank}] {line}")
                # Popen reports death by signal N as -N; a shell reports 128 + N.
                return 128 - code if code < 0 else code
            if all(code == 0 for code in codes):
                emit(f"all {len(running)} ranks exited 0")
                return 0
            time.sleep(POLL_SECONDS)
    except BaseException as error:
        # From here a signal cuts the stop's grace short, rather than raising
        # into it and leaving the pool half stopped.
        _escalate(forced)
        if isinstance(error, Stopped) and running:
            emit(f"stopping the pool on {signal.Signals(error.code - 128).name}: a training run "
                 f"checkpoints at the step its ranks agree on, within {PREEMPTION_GRACE:.0f} s; "
                 "signal again to stop it now")
        _stop(running, PREEMPTION_GRACE if isinstance(error, Stopped) else FAILURE_GRACE, forced)
        if isinstance(error, Stopped):
            return error.code
        raise
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)
        for relay in relays:
            relay.join(timeout=5)
