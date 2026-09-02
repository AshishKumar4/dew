"""dew-tpu: create Cloud TPUs, set them up for dew, and run on every worker.

Reads its defaults from ~/.config/dew/tpu.toml. Every command takes --dry-run,
which prints the commands it would run and exits.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import getpass
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import tyro

from dew.cli import config, tpu_setup
from dew.cli.gcloud import Gcloud, Node, Tpu, emit, exit_code

#: States a create never recovers from.
DEAD_STATES = frozenset({"TERMINATED", "PREEMPTED", "DELETING"})
READY_TIMEOUT = 1800
POLL_SECONDS = 10

Positional = tyro.conf.Positional

#: Optional flags, named so --help reads like documentation instead of {None}|STR.
Zone = Annotated[str | None, tyro.conf.arg(metavar="ZONE")]
Zones = Annotated[str | None, tyro.conf.arg(metavar="ZONES")]
Project = Annotated[str | None, tyro.conf.arg(metavar="PROJECT")]
Kind = Annotated[str | None, tyro.conf.arg(metavar="TYPE")]
Version = Annotated[str | None, tyro.conf.arg(metavar="VERSION")]
User = Annotated[str | None, tyro.conf.arg(metavar="USER")]
Bucket = Annotated[str | None, tyro.conf.arg(metavar="BUCKET")]
Disk = Annotated[str | None, tyro.conf.arg(metavar="DISK")]
PyVersion = Annotated[str | None, tyro.conf.arg(metavar="X.Y")]
Worker = Annotated[str, tyro.conf.arg(metavar="N|all")]
Job = Annotated[str, tyro.conf.arg(metavar="JOB")]
Extras = Annotated[str, tyro.conf.arg(metavar="LIST")]
Release = Annotated[str, tyro.conf.arg(metavar="VERSION")]


@dataclasses.dataclass(kw_only=True)
class Base:
    """Flags shared by every command that talks to a TPU."""

    zone: Zone = None
    """Zone of the TPU. Searched across the configured zones when omitted."""
    dry_run: bool = False
    """Print the commands that would run, then exit."""


# ----------------------------------------------------------------- shared work


def _open(cmd: Base) -> tuple[config.TpuConfig, Gcloud]:
    cfg = config.load()
    return cfg, Gcloud(project=cfg.project, dry_run=cmd.dry_run)


def _zone(gcloud: Gcloud, cfg: config.TpuConfig, name: str, wanted: str | None) -> str:
    """The zone a TPU lives in: the flag, the cache, or the first zone that has it."""
    if wanted:
        # Not cached: an unverified flag would send every later command astray.
        return wanted
    cached = config.cached_zone(name)
    if cached:
        return cached
    if gcloud.dry_run:
        return cfg.zones[0]
    for zone in cfg.zones:
        found = gcloud.run(gcloud.argv(
            "compute", "tpus", "tpu-vm", "describe", name,
            f"--zone={zone}", "--format=value(name)"))
        if found.ok:
            config.cache_zone(name, zone)
            return zone
    raise SystemExit(f"{name} is in none of {', '.join(cfg.zones)}")


def _missing(tpu: Tpu) -> SystemExit:
    return SystemExit(f"{tpu.name} is not in {tpu.zone}. Pass --zone to look elsewhere.")


def _tpu(cmd: Base, cfg: config.TpuConfig, gcloud: Gcloud, name: str) -> Tpu:
    return Tpu(gcloud, name, _zone(gcloud, cfg, name, cmd.zone), cfg.ssh_user)


def _slice(tpu: Tpu, cfg: config.TpuConfig, type_hint: str = "") -> tuple[int, str]:
    """Worker count and accelerator type. A dry run reads them from the flags."""
    if tpu.gcloud.dry_run:
        kind = type_hint or cfg.accelerator_type
        return config.worker_count(kind), kind
    node = tpu.describe()
    if node is None:
        raise _missing(tpu)
    return node.workers, node.accelerator_type or type_hint or cfg.accelerator_type


def _workers(spec: str, tpu: Tpu, cfg: config.TpuConfig, type_hint: str = "") -> list[int]:
    """`all` means every worker in the slice, anything else is one index."""
    if spec != "all":
        return [int(spec)]
    return list(range(_slice(tpu, cfg, type_hint)[0]))


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Print a table. Every command that shows one comes through here."""
    columns = [headers, *rows]
    widths = [max(len(str(row[index])) for row in columns) for index in range(len(headers))]
    for row in columns:
        emit("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip())


def _worker_table(node: Node | None) -> None:
    if node is None:
        return
    _table(("WORKER", "INTERNAL IP", "EXTERNAL IP"), [
        (str(index), internal, node.ips[index] if index < len(node.ips) else "-")
        for index, internal in enumerate(node.internal_ips)
    ])


def _wait_ready(tpu: Tpu) -> Node | None:
    """Poll until the TPU answers READY. Returns the node it saw."""
    if tpu.gcloud.dry_run:
        return None
    deadline = time.monotonic() + READY_TIMEOUT
    while True:
        node = tpu.describe()
        state = node.state if node else "PENDING"
        if state == "READY":
            return node
        if state in DEAD_STATES:
            raise SystemExit(f"{tpu.name} is {state}")
        if time.monotonic() >= deadline:
            raise SystemExit(f"{tpu.name} is still {state} after {READY_TIMEOUT}s")
        emit(f"{tpu.name} is {state}")
        time.sleep(POLL_SECONDS)


def _git_root() -> Path:
    done = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True)
    if done.returncode:
        raise SystemExit("sync needs a git working tree in the current directory")
    return Path(done.stdout.strip())


def _rsync_argv(root: Path, user: str, host: str, excludes: Sequence[str]) -> list[str]:
    key = Path.home() / ".ssh/google_compute_engine"
    remote_shell = (f"ssh -i {key} -o StrictHostKeyChecking=no "
                    "-o UserKnownHostsFile=/dev/null")
    return [
        "rsync", "-az", "--delete", "--exclude=.git",
        # Per-directory merge rule: every .gitignore in the tree excludes.
        "--filter=:- .gitignore",
        *(f"--exclude={pattern}" for pattern in excludes),
        "-e", remote_shell,
        f"{root}/", f"{user}@{host}:~/{root.name}/",
    ]


def _sync(tpu: Tpu, cfg: config.TpuConfig, count: int, excludes: Sequence[str]) -> tuple[Path, int]:
    """Copy the git working tree to ~/<repo> on every worker."""
    root = _git_root()
    user = cfg.ssh_user or getpass.getuser()
    hosts = _hosts(tpu, count)
    jobs = [
        (f"[worker {index}] ", _rsync_argv(root, user, host, excludes))
        for index, host in enumerate(hosts)
    ]
    return root, exit_code(tpu.gcloud.fanout(jobs))


def _hosts(tpu: Tpu, count: int) -> list[str]:
    """The address of each worker. A dry run names them instead."""
    if tpu.gcloud.dry_run:
        return [f"<worker-{index}-ip>" for index in range(count)]
    node = tpu.describe()
    if node is None:
        raise _missing(tpu)
    return [ip or node.internal_ips[index] for index, ip in enumerate(node.ips)]


def _job_id(name: str) -> str:
    return f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"


# -------------------------------------------------------------------- commands


@dataclasses.dataclass
class Init:
    """Write ~/.config/dew/tpu.toml from flags, asking for what is missing."""

    project: Project = None
    """Google Cloud project that owns the TPUs."""
    zones: Zones = None
    """Zones to search, in order, comma separated."""
    accelerator_type: Kind = None
    """Default accelerator type, for example v5e-8."""
    runtime_version: Version = None
    """Runtime version, or auto to pick it from the accelerator generation."""
    ssh_user: User = None
    """User to log in as. Empty lets gcloud choose."""
    gcs_bucket: Bucket = None
    """Bucket to mount with gcsfuse during setup."""
    data_disk: Disk = None
    """Persistent disk to attach on create."""
    python_version: PyVersion = None
    """Python version of the venv on the workers."""
    dry_run: bool = False
    """Print the config that would be written, then exit."""

    def run(self, rest: list[str]) -> int:
        cfg = config.load()
        answers: dict[str, object] = {}
        for field in dataclasses.fields(config.TpuConfig):
            given = getattr(self, field.name)
            if given is None and sys.stdin.isatty():
                current = getattr(cfg, field.name)
                shown = ",".join(current) if isinstance(current, tuple) else current
                given = input(f"{field.name} [{shown}]: ").strip() or None
            if given is None:
                continue
            answers[field.name] = (
                tuple(part.strip() for part in given.split(",") if part.strip())
                if field.name == "zones" else given
            )
        new = dataclasses.replace(cfg, **answers)
        if self.dry_run:
            emit(f"# {config.config_path()}")
            emit(config.dumps(new).rstrip())
            return 0
        emit(f"wrote {config.save(new)}")
        return 0


@dataclasses.dataclass
class Create(Base):
    """Create a TPU VM or pod slice and wait until it is ready."""

    name: Positional[str]
    """Name of the TPU."""
    type: Kind = None
    """Accelerator type, for example v5e-16."""
    spot: bool = False
    """Ask for a spot TPU, which costs less and can be preempted."""
    queued: bool = False
    """Go through the queued resources API instead of creating directly."""
    version: Version = None
    """Runtime version, or auto to pick it from the accelerator generation."""
    disk: Disk = None
    """Persistent disk to attach, mounted on the worker at /mnt/persist."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        kind = self.type or cfg.accelerator_type
        zone = self.zone or cfg.zones[0]
        asked = self.version or cfg.runtime_version
        version = config.runtime_for(kind) if asked in ("", "auto") else asked
        disk = cfg.data_disk if self.disk is None else self.disk
        api_type = config.api_type(kind)

        if self.queued:
            args = ["compute", "tpus", "queued-resources", "create", self.name,
                    f"--zone={zone}", f"--accelerator-type={api_type}",
                    f"--runtime-version={version}", f"--node-id={self.name}"]
        else:
            args = ["compute", "tpus", "tpu-vm", "create", self.name,
                    f"--zone={zone}", f"--accelerator-type={api_type}",
                    f"--version={version}"]
        if self.spot:
            args.append("--spot")
        if disk:
            source = f"projects/{cfg.project}/zones/{zone}/disks/{disk}"
            args.append(f"--data-disk=mode=read-write,source={source}")
            args.append(f"--metadata-from-file=startup-script={_startup_script()}")
        gcloud.run(gcloud.argv(*args), capture=False, check=True)
        config.cache_zone(self.name, zone)

        tpu = Tpu(gcloud, self.name, zone, cfg.ssh_user)
        emit(f"{self.name} in {zone}: {api_type} on {config.worker_count(kind)} worker(s)")
        _worker_table(_wait_ready(tpu))
        return 0


def _startup_script() -> Path:
    """The data disk mount script, on disk because gcloud reads it from a file."""
    path = config.config_dir() / "disk-startup.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tpu_setup.disk_startup_script())
    return path


@dataclasses.dataclass
class Delete(Base):
    """Delete a TPU, and the queued resource that holds it."""

    name: Positional[str]
    """Name of the TPU."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        node = None if gcloud.dry_run else tpu.describe()
        if node is not None and node.queued_resource:
            argv = gcloud.argv("compute", "tpus", "queued-resources", "delete",
                               node.queued_resource, f"--zone={tpu.zone}", "--force", "--quiet")
        else:
            argv = tpu.vm("delete", self.name, f"--zone={tpu.zone}", "--quiet")
        code = gcloud.run(argv, capture=False).code
        if not code:
            config.forget_zone(self.name)
        return code


@dataclasses.dataclass
class Start(Base):
    """Start a stopped TPU."""

    name: Positional[str]
    """Name of the TPU."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        code = gcloud.run(tpu.vm("start", self.name, f"--zone={tpu.zone}"), capture=False).code
        if not code:
            _worker_table(_wait_ready(tpu))
        return code


@dataclasses.dataclass
class Stop(Base):
    """Stop a TPU. It keeps its disks and its name."""

    name: Positional[str]
    """Name of the TPU."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        return gcloud.run(tpu.vm("stop", self.name, f"--zone={tpu.zone}"), capture=False).code


@dataclasses.dataclass
class List(Base):
    """List TPUs in every configured zone."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        zones = cfg.zones if self.zone in (None, "all") else (self.zone,)
        rows = []
        for zone in zones:
            payload = gcloud.json("compute", "tpus", "tpu-vm", "list",
                                  f"--zone={zone}", default=[])
            for item in payload:
                node = Node.parse(item)
                rows.append((node.name, node.accelerator_type, node.state,
                             node.health or "-", str(node.workers), zone,
                             "yes" if node.spot else "-"))
        if not rows:
            emit(f"no TPUs in {', '.join(zones)}")
            return 0
        _table(("NAME", "TYPE", "STATE", "HEALTH", "WORKERS", "ZONE", "SPOT"), rows)
        return 0


@dataclasses.dataclass
class Describe(Base):
    """Show what a TPU is and where its workers are."""

    name: Positional[str]
    """Name of the TPU."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        node = tpu.describe()
        if node is None:
            if gcloud.dry_run:
                return 0
            raise _missing(tpu)
        _table(("FIELD", "VALUE"), [
            ("name", node.name),
            ("zone", node.zone),
            ("type", node.accelerator_type),
            ("topology", node.topology or "-"),
            ("runtime", node.runtime_version),
            ("state", node.state),
            ("health", node.health or "-"),
            ("spot", "yes" if node.spot else "-"),
            ("queued resource", node.queued_resource or "-"),
        ])
        emit("")
        _worker_table(node)
        return 0


@dataclasses.dataclass
class Ssh(Base):
    """Open a shell on one worker, with ports forwarded."""

    name: Positional[str]
    """Name of the TPU."""
    worker: int = 0
    """Worker to connect to."""
    forward: Annotated[
        tyro.conf.UseAppendAction[tuple[int, ...]],
        tyro.conf.arg(aliases=("-L",), metavar="PORT"),
    ] = ()
    """Ports to forward from the worker. Repeat for each port."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        flags = [f"-L {port}:localhost:{port}" for port in self.forward]
        argv = tpu.ssh_argv(str(self.worker), ssh_flags=flags)
        if rest:
            argv += ["--", *rest]
        return gcloud.run(argv, capture=False).code


@dataclasses.dataclass
class Run(Base):
    """Run a command on the workers. Put the command after --."""

    name: Positional[str]
    """Name of the TPU."""
    worker: Worker = "all"
    """Worker index, or all for every worker at once."""
    detach: bool = False
    """Start the command under nohup and return the job id."""
    job: Job = ""
    """Name of the detached job. A timestamp by default."""

    def run(self, rest: list[str]) -> int:
        if not rest:
            raise SystemExit("dew-tpu run NAME -- COMMAND")
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        workers = _workers(self.worker, tpu, cfg)
        command = shlex.join(rest)
        if not self.detach:
            return exit_code(tpu.fanout([(index, tpu_setup.wrap(command)) for index in workers]))
        job = self.job or _job_id(self.name)
        results = tpu.fanout([
            (index, tpu_setup.detached(command, job, index)) for index in workers])
        emit(f"job {job}: dew-tpu logs {self.name} {job} --follow")
        return exit_code(results)


@dataclasses.dataclass
class Logs(Base):
    """Show the log of a detached job."""

    name: Positional[str]
    """Name of the TPU."""
    job: Positional[str]
    """Job id that run or train printed."""
    worker: Worker = "0"
    """Worker to read, or all for every worker."""
    follow: bool = False
    """Keep the log open and print new lines."""
    lines: int = 200
    """How much of the tail to print."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        workers = _workers(self.worker, tpu, cfg)
        return exit_code(tpu.fanout([
            (index, tpu_setup.tail(self.job, index, self.follow, self.lines))
            for index in workers]))


@dataclasses.dataclass
class Copy(Base):
    """Copy a local file or directory to the workers."""

    name: Positional[str]
    """Name of the TPU."""
    src: Positional[str]
    """Local path."""
    dst: Positional[str]
    """Remote path."""
    worker: Worker = "all"
    """Worker to copy to, or all."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        argv = tpu.scp_argv([self.src], f"{tpu.host}:{self.dst}", worker=self.worker,
                            recurse=Path(self.src).is_dir())
        return gcloud.run(argv, capture=False).code


@dataclasses.dataclass
class Sync(Base):
    """Copy the git working tree to ~/<repo> on every worker."""

    name: Positional[str]
    """Name of the TPU."""
    exclude: Annotated[
        tyro.conf.UseAppendAction[tuple[str, ...]],
        tyro.conf.arg(metavar="PATTERN"),
    ] = ()
    """Extra rsync exclude patterns. .gitignore is already honoured. Repeatable."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        count, _ = _slice(tpu, cfg)
        root, code = _sync(tpu, cfg, count, self.exclude)
        if not code:
            emit(f"{root} is on {count} worker(s) at ~/{root.name}")
        return code


@dataclasses.dataclass
class Setup(Base):
    """Install uv, a venv, jax and dew on every worker, then count the devices."""

    name: Positional[str]
    """Name of the TPU."""
    from_source: bool = False
    """Sync the working tree and install it in editable mode."""
    version: Release = ""
    """Release of dew-ml to install. The newest by default."""
    extras: Extras = ""
    """Extras to install, for example tfds,av."""
    gcs_bucket: Bucket = None
    """Bucket to mount with gcsfuse."""
    python_version: PyVersion = None
    """Python version of the venv."""
    type: Kind = None
    """Accelerator type to expect, for the device check in a dry run."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        count, kind = _slice(tpu, cfg, self.type or "")

        source = ""
        if self.from_source:
            root, code = _sync(tpu, cfg, count, ())
            if code:
                return code
            source = root.name
        spec, editable = tpu_setup.package_spec(source, self.extras, self.version)
        script = tpu_setup.render(
            python_version=self.python_version or cfg.python_version,
            package_spec=spec,
            editable=editable,
            gcs_bucket=cfg.gcs_bucket if self.gcs_bucket is None else self.gcs_bucket,
        )
        path = config.config_dir() / f"setup-{self.name}.sh"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script)

        copy = gcloud.run(tpu.scp_argv([str(path)], f"{tpu.host}:~/dew-setup.sh", worker="all"),
                          capture=False)
        if not copy.ok:
            return copy.code
        code = exit_code(tpu.fanout([(index, "bash ~/dew-setup.sh") for index in range(count)]))
        if code:
            return code
        return _verify_devices(tpu, count, config.device_count(kind))


def _verify_devices(tpu: Tpu, count: int, expected: int) -> int:
    """Every worker must see the whole slice before a run is worth starting."""
    results = tpu.fanout([(index, tpu_setup.DEVICE_COUNT) for index in range(count)],
                         capture=True)
    if tpu.gcloud.dry_run:
        return 0
    rows, failed = [], 0
    for index, result in enumerate(results):
        reported = result.out.split()
        devices = int(reported[0]) if reported else 0
        local = reported[1] if len(reported) > 1 else "-"
        good = devices == expected
        failed += not good
        rows.append((str(index), str(devices) if devices else "-", local,
                     "ok" if good else f"want {expected}"))
    _table(("WORKER", "DEVICES", "LOCAL", "CHECK"), rows)
    return 1 if failed else 0


@dataclasses.dataclass
class Train(Base):
    """Sync the tree and start a recipe on every worker. Recipe after --."""

    name: Positional[str]
    """Name of the TPU."""
    job: Job = ""
    """Name of the job. A timestamp by default."""

    def run(self, rest: list[str]) -> int:
        if not rest:
            raise SystemExit("dew-tpu train NAME -- recipes/lm/train.py [FLAGS]")
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        count, _ = _slice(tpu, cfg)
        root, code = _sync(tpu, cfg, count, ())
        if code:
            return code
        job = self.job or _job_id(self.name)
        command = shlex.join(["python", *rest, "--trainer.multi-host", "True"])
        results = tpu.fanout([
            (index, tpu_setup.detached(command, job, index, cwd=f"~/{root.name}"))
            for index in range(count)])
        code = exit_code(results)
        if code:
            return code
        emit(f"job {job} on {count} worker(s), following worker 0")
        return exit_code(tpu.fanout([(0, tpu_setup.tail(job, 0, follow=True))]))


@dataclasses.dataclass
class Status(Base):
    """Show what each worker is doing."""

    name: Positional[str]
    """Name of the TPU."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        count, _ = _slice(tpu, cfg)
        node = None if gcloud.dry_run else tpu.describe()
        if node is not None:
            emit(f"{node.name} {node.accelerator_type} {node.state} "
                 f"{node.health or 'health unknown'} in {node.zone}")
        results = tpu.fanout([(index, tpu_setup.STATUS) for index in range(count)], capture=True)
        if gcloud.dry_run:
            return 0
        rows = []
        for index, result in enumerate(results):
            fields = (result.out.strip().split("|") + ["-"] * 4)[:4]
            rows.append((str(index), fields[0] or "-", fields[1], fields[2], fields[3]))
        _table(("WORKER", "UPTIME", "DEW PROCS", "DEVICES BUSY", "DEVICES"), rows)
        return exit_code(results)


@dataclasses.dataclass
class Reset(Base):
    """Kill whatever holds the accelerators on every worker."""

    name: Positional[str]
    """Name of the TPU."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        tpu = _tpu(self, cfg, gcloud, self.name)
        count, _ = _slice(tpu, cfg)
        return exit_code(tpu.fanout([(index, tpu_setup.RESET) for index in range(count)]))


@dataclasses.dataclass
class Spawn(Base):
    """Create N independent TPUs, set them up, and start a command on each."""

    base: Positional[str]
    """Name prefix. The TPUs are base-0, base-1 and so on."""
    count: Positional[int]
    """How many TPUs to create."""
    type: Kind = None
    """Accelerator type for all of them."""
    spot: bool = False
    """Ask for spot TPUs."""
    queued: bool = False
    """Go through the queued resources API."""
    extras: Extras = ""
    """Extras to install during setup."""

    def run(self, rest: list[str]) -> int:
        cfg, gcloud = _open(self)
        names = [f"{self.base}-{index}" for index in range(self.count)]
        zone = self.zone
        command = shlex.join(rest)

        def one(name: str) -> tuple[str, str, str]:
            emit(f"[{name}] create")
            Create(name=name, type=self.type, spot=self.spot, queued=self.queued,
                   zone=zone, dry_run=self.dry_run).run([])
            emit(f"[{name}] setup")
            Setup(name=name, extras=self.extras, type=self.type,
                  zone=zone, dry_run=self.dry_run).run([])
            if not command:
                return name, "ready", "-"
            tpu = _tpu(self, cfg, gcloud, name)
            count, _ = _slice(tpu, cfg, self.type or "")
            job = _job_id(name)
            emit(f"[{name}] run {job}")
            tpu.fanout([(index, tpu_setup.detached(command, job, index))
                        for index in range(count)])
            return name, "ready", job

        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, self.count)) as pool:
            futures = {pool.submit(one, name): name for name in names}
            for future in concurrent.futures.as_completed(futures):
                try:
                    rows.append(future.result())
                except (SystemExit, OSError) as error:
                    rows.append((futures[future], f"failed: {error}", "-"))
        _table(("NAME", "STATE", "JOB"), sorted(rows))
        return 1 if any(row[1].startswith("failed") for row in rows) else 0


COMMANDS = {
    "init": Init,
    "create": Create,
    "delete": Delete,
    "start": Start,
    "stop": Stop,
    "list": List,
    "describe": Describe,
    "ssh": Ssh,
    "run": Run,
    "logs": Logs,
    "copy": Copy,
    "sync": Sync,
    "setup": Setup,
    "train": Train,
    "status": Status,
    "reset": Reset,
    "spawn": Spawn,
}

CONFIG = (
    tyro.conf.FlagCreatePairsOff,
    tyro.conf.PositionalMetavarFromFieldName,
)


def _split(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Everything after the first bare -- is the remote command."""
    argv = list(argv)
    if "--" not in argv:
        return argv, []
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1:]


def main(argv: Sequence[str] | None = None) -> int:
    flags, rest = _split(sys.argv[1:] if argv is None else argv)
    command = tyro.extras.subcommand_cli_from_dict(
        COMMANDS, args=flags, prog="dew-tpu", description=__doc__, config=CONFIG)
    return command.run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
