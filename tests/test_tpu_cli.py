"""dew tpu and dew launch --tpu against a fake gcloud: argv per command, fan-out, pools, dry runs, config."""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dew.builds import PACKAGE_INDEX
from dew.cli import config as tpu_config, tpu as tpu_cli, tpu_setup
from dew.cli.launch import remote_script

#: Stands in for gcloud and rsync. Records every argv, answers from a state file. Where the state names
#: `hosts`, each worker is a home directory under it, and what reaches a worker reaches that home: a
#: command over ssh runs there under bash, until every process it started has ended, with the worker's
#: tools first on its PATH (`remote_path`); scp and rsync copy into it, rsync at the worker's address.
FAKE = '''#!/usr/bin/env python3
import fnmatch
import json
import os
import shutil
import sys
import subprocess

argv = sys.argv[1:]
program = os.path.basename(sys.argv[0])
state = json.load(open(os.environ["DEW_FAKE_STATE"]))
hosts = state.get("hosts")
with open(os.environ["DEW_FAKE_CALLS"], "a") as log:
    log.write(json.dumps([program, *argv]) + "\\n")


def die(message, code=1):
    sys.stderr.write(message + "\\n")
    raise SystemExit(code)


def flag(name, default=""):
    for item in argv:
        if item.startswith(f"--{name}="):
            return item.split("=", 1)[1]
    return default


def homes(name, worker):
    """The homes of TPU `name`'s workers that `worker` names, an index or all."""
    count = len(state["nodes"][name]["payload"]["networkEndpoints"])
    workers = range(count) if worker == "all" else [worker]
    found = [os.path.join(hosts, name, str(index)) for index in workers]
    for home in found:
        os.makedirs(home, exist_ok=True)
    return found


def under(home, path):
    return os.path.join(home, path.removeprefix("~/").lstrip("/"))


fail = state.get("fail", "")
if fail and fail in " ".join(argv):
    die("the fake was told to fail here", 3)
hang = state.get("hang", "")
if hang and hang in " ".join(argv):
    import time
    time.sleep(600)

if program == "rsync":
    if hosts:
        source, (address, path) = argv[-2], argv[-1].split("@")[-1].split(":", 1)
        name, index = next((name, index) for name, node in state["nodes"].items()
                           for index, end in enumerate(node["payload"]["networkEndpoints"])
                           if address in (end["ipAddress"], end["accessConfig"]["externalIp"]))
        target = under(homes(name, index)[0], path)
        excluded = [item.split("=", 1)[1] for item in argv if item.startswith("--exclude=")]
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(*excluded), dirs_exist_ok=True)
        for directory, _, files in os.walk(target):
            for file in files:
                kept = os.path.relpath(os.path.join(directory, file), target)
                spared = any(fnmatch.fnmatch(part, pattern)
                             for part in kept.split(os.sep) for pattern in excluded)
                if "--delete" in argv and not os.path.exists(os.path.join(source, kept)) and not spared:
                    os.remove(os.path.join(directory, file))
    raise SystemExit(0)

group = argv[:3]
verb = argv[3] if len(argv) > 3 else ""
if argv[:2] == ["config", "get-value"]:
    print(state.get("gcloud_project", ""))
    raise SystemExit(0)

if group == ["compute", "tpus", "queued-resources"]:
    raise SystemExit(0)
if group != ["compute", "tpus", "tpu-vm"]:
    die("the fake does not know " + " ".join(argv))

if verb == "list":
    # A node's `states` are what its listings say, in turn, the last for good.
    for node in state["nodes"].values():
        if node["zone"] == flag("zone") and node.get("states"):
            node["payload"]["state"] = node["states"].pop(0) if len(node["states"]) > 1 else node["states"][0]
            json.dump(state, open(os.environ["DEW_FAKE_STATE"], "w"))
    listed = state.get("listing", {}).get(flag("zone"), [])
    listed += [node["payload"] for node in state["nodes"].values() if node["zone"] == flag("zone")]
    print(json.dumps(listed))
    raise SystemExit(0)

if verb == "ssh":
    name = argv[4].split("@")[-1]
    node = state["nodes"].get(name, {})
    command = flag("command")
    if "jax.device_count" in command:
        print(node.get("devices", "16 8"))
    elif hosts and command and "dew-setup.sh" not in command:
        [home] = homes(name, int(flag("worker")))
        done = subprocess.run(["bash", "-c", command + "\\nwait"], cwd=home,
                              env={"HOME": home, "PATH": state["remote_path"]})
        raise SystemExit(done.returncode)
    elif command.startswith("printf"):
        print(node.get("status", "up 2 hours|0|0|8"))
    elif command:
        print("ran " + command)
    raise SystemExit(0)

if verb == "scp" and hosts:
    *sources, target = [item for item in argv[4:] if not item.startswith("--")]
    name, path = target.split("@")[-1].split(":", 1)
    worker = flag("worker")
    for home in homes(name, worker if worker == "all" else int(worker)):
        for source in sources:
            os.makedirs(os.path.dirname(under(home, path)), exist_ok=True)
            if os.path.isdir(source):
                if "--recurse" not in argv:
                    die(source + ": not a regular file")
                shutil.copytree(source, under(home, path), dirs_exist_ok=True)
            else:
                shutil.copy(source, under(home, path))
    raise SystemExit(0)

if verb in ("create", "delete", "start", "stop", "scp", "attach-disk"):
    raise SystemExit(0)
die("the fake does not know " + " ".join(argv))
'''

NODE = {
    "name": "projects/my-project/locations/us-central2-b/nodes/slice",
    "acceleratorType": "v5litepod-16",
    "acceleratorConfig": {"topology": "4x4", "type": "V5LITE_POD"},
    "runtimeVersion": "v2-alpha-tpuv5-lite",
    "state": "READY",
    "health": "HEALTHY",
    "schedulingConfig": {},
    "networkEndpoints": [
        {"ipAddress": "10.0.0.1", "accessConfig": {"externalIp": "34.0.0.1"}, "port": 8470},
        {"ipAddress": "10.0.0.2", "accessConfig": {"externalIp": "34.0.0.2"}, "port": 8470},
    ],
}

CONFIG = tpu_config.TpuConfig(
    project="my-project",
    zones=("us-central2-b", "europe-west4-a", "us-east1-d"),
    accelerator_type="v5e-16",
    runtime_version="auto",
    ssh_user="you",
    gcs_bucket="my-bucket",
    data_disk="",
    python_version="3.12",
)


class Fake:
    """The fake gcloud on PATH, plus the state it answers from."""

    def __init__(self, root: Path):
        self.root = root
        self.calls_path = root / "calls.jsonl"
        self.state_path = root / "state.json"
        self.state: dict[str, Any] = {"nodes": {}, "listing": {}}
        self.flush()

    def flush(self) -> None:
        self.state_path.write_text(json.dumps(self.state))

    def offer(self, name: str, zone: str, payload: dict = NODE, **extra) -> None:
        """Tell the fake that a TPU exists, and what its zone's listing should hold of it."""
        node = dict(zone=zone, payload=json.loads(json.dumps(payload)), **extra)
        node["payload"]["name"] = f"projects/my-project/locations/{zone}/nodes/{name}"
        self.state["nodes"][name] = node
        self.flush()

    def listing(self, per_zone: dict) -> None:
        self.state["listing"] = per_zone
        self.flush()

    def fail_on(self, fragment: str) -> None:
        self.state["fail"] = fragment
        self.flush()

    def hang_on(self, fragment: str) -> None:
        """Make every call whose arguments hold `fragment` hang, as a gcloud that never answers."""
        self.state["hang"] = fragment
        self.flush()

    def host(self, tools: dict[str, str]) -> None:
        """Make each TPU's workers home directories here (`home`), which what reaches a worker reaches,
        with `tools` (a name to a shell script's body) as the worker system's own, ahead of this one's."""
        binaries = self.root / "worker-bin"
        binaries.mkdir()
        for name, body in tools.items():
            (binaries / name).write_text(f"#!/bin/sh\n{body}\n")
            (binaries / name).chmod(0o755)
        self.state.update(hosts=str(self.root / "workers"), remote_path=os.pathsep.join(
            (str(binaries), "/usr/bin", "/bin")))
        self.flush()

    def home(self, name: str, worker: int) -> Path:
        """TPU `name`'s worker `worker`'s home (`host`)."""
        return self.root / "workers" / name / str(worker)

    @property
    def calls(self) -> list[list[str]]:
        if not self.calls_path.is_file():
            return []
        return [json.loads(line) for line in self.calls_path.read_text().splitlines()]

    def gcloud_calls(self) -> list[list[str]]:
        return [call[1:] for call in self.calls if call[0] == "gcloud"]

    def rsync_calls(self) -> list[list[str]]:
        return [call[1:] for call in self.calls if call[0] == "rsync"]


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("gcloud", "rsync"):
        path = binaries / name
        path.write_text(FAKE)
        path.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("DEW_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DEW_FAKE_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("DEW_FAKE_STATE", str(tmp_path / "state.json"))
    # The CLI polls a TPU through the clock, which the fake answers at once.
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    tpu_config.save(CONFIG)
    return Fake(tmp_path)


def run(*argv: str) -> int:
    return tpu_cli.main(list(argv))


def gcloud(*args: str) -> list[str]:
    return [*args, "--project=my-project"]


def ssh(worker: str, command: str, name: str = "slice",
        zone: str = "us-central2-b") -> list[str]:
    return gcloud("compute", "tpus", "tpu-vm", "ssh", f"you@{name}", f"--zone={zone}",
                  f"--worker={worker}", f"--command={command}")


def env_prefix() -> str:
    return ". $HOME/.dew-env 2>/dev/null; "


def config_dir() -> Path:
    return Path(os.environ["DEW_CONFIG_DIR"])


def zones_seen(calls: list[list[str]], verb: str) -> list[str]:
    return [next(part.split("=", 1)[1] for part in call if part.startswith("--zone="))
            for call in calls if verb in call]


def only(calls: list[list[str]], verb: str) -> list[list[str]]:
    return [call for call in calls if len(call) > 3 and call[3] == verb]


def worker_tools(**more: str) -> dict[str, str]:
    """A TPU worker's system as `Fake.host` stands it in: this machine's Python; a tail that ends where
    it would follow, the job having ended; an uptime of its home's name, the worker's index; two of
    Dew's processes; two accelerators that two processes hold, one of them both; a sudo that runs what
    it is given; and a kill that records its arguments in the home. `more` adds to them."""
    return {
        "python": f'exec "{sys.executable}" "$@"',
        "tail": f'exec "{shutil.which("tail")}" $(printf "%s\\n" "$@" | grep -vx -- -f)',
        "uptime": 'echo "up $(basename "$HOME") hours"',
        "pgrep": "echo 2",
        "ls": "printf '/dev/accel0\\n/dev/accel1\\n'",
        "lsof": "printf '4242\\n4343\\n4242\\n'",
        "sudo": '[ "$1" = -n ] && shift; exec "$@"',
        "kill": 'echo "$*" >> "$HOME/killed"',
        **more,
    }


def project(root: Path) -> Path:
    """A git working tree at `root` holding a recipe that prints the arguments it ran with."""
    root.mkdir()
    (root / "recipe.py").write_text("import sys\nprint('recipe', *sys.argv[1:])\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def printed(capsys) -> list[str]:
    """The lines the CLI printed since it was last asked."""
    return capsys.readouterr().out.splitlines()


# --------------------------------------------------------------------- lifecycle


def test_create_uses_the_configured_defaults(fake):
    fake.offer("slice", "us-central2-b")
    assert run("create", "slice") == 0
    assert fake.gcloud_calls()[0] == gcloud(
        "compute", "tpus", "tpu-vm", "create", "slice", "--zone=us-central2-b",
        "--accelerator-type=v5litepod-16", "--version=v2-alpha-tpuv5-lite")
    assert json.loads((config_dir() / "zones.json").read_text()) == {"slice": "us-central2-b"}


def test_a_queued_spot_slice_is_created_once_its_node_is_ready_and_a_preempted_one_stops_it(fake):
    """`create --queued --spot` asks the queued resources API for a spot node named for the TPU, then
    waits through the listings until the node is READY, and caches its zone; a node preempted while it
    waits stops the command at that listing instead."""
    fake.offer("slice", "us-central2-b", states=["CREATING", "CREATING", "READY"])
    assert run("create", "slice", "--queued", "--spot") == 0
    [created] = only(fake.gcloud_calls(), "create")
    assert created[:5] == ["compute", "tpus", "queued-resources", "create", "slice"]
    assert {"--spot", "--node-id=slice"} <= set(created)
    assert len(only(fake.gcloud_calls(), "list")) == 3
    assert json.loads((config_dir() / "zones.json").read_text()) == {"slice": "us-central2-b"}
    fake.offer("other", "us-central2-b", states=["CREATING", "PREEMPTED"])
    with pytest.raises(SystemExit):
        run("create", "other", "--queued", "--spot", "--zone", "us-central2-b")
    assert len(only(fake.gcloud_calls(), "list")) == 5


def test_a_wait_on_a_gcloud_that_never_answers_ends_at_its_deadline(fake):
    """Each readiness probe has what is left of the wait, so a listing that hangs is killed there and
    the wait ends with the TPU unready, instead of running past its deadline."""
    from dew.cli.gcloud import Gcloud, Tpu

    fake.offer("slice", "us-central2-b")
    fake.hang_on("list")
    began = time.monotonic()
    with pytest.raises(SystemExit):
        Tpu(Gcloud("my-project"), "slice", "us-central2-b").ready(wait=2)
    assert time.monotonic() - began < 60


def test_a_dry_run_writes_nothing_it_would_keep(fake):
    """A dry run prints the commands and changes no file: a create with a data disk leaves the
    zone cache and writes no startup script, and a setup writes no setup script."""
    fake.offer("slice", "us-central2-b")
    (config_dir() / "zones.json").write_text(json.dumps({"slice": "europe-west4-a"}))
    before = {path: path.read_bytes() for path in config_dir().rglob("*") if path.is_file()}
    assert run("create", "slice", "--dry-run", "--disk", "data") == 0
    assert run("setup", "slice", "--dry-run", "--zone", "us-central2-b") == 0
    assert {path: path.read_bytes() for path in config_dir().rglob("*") if path.is_file()} == before


def test_ssh_config_refuses_a_block_without_its_end_marker_and_keeps_the_file(fake):
    """The TPU's ~/.ssh/config block lies between its two markers. A file holding the start marker
    without the end one is refused, and every entry in it is kept as it was; with the end marker
    back, the block is replaced and the entries after it are kept. An end marker on the file's last
    line, with no newline after it, closes the block too."""
    fake.offer("slice", "us-central2-b")
    config = Path(os.environ["HOME"]) / ".ssh" / "config"
    config.parent.mkdir()
    block = "# dew tpu slice\nHost slice\n    HostName 10.0.0.1\n"
    other = "Host other\n    HostName 10.9.9.9\n"
    config.write_text(block + other)
    with pytest.raises(ValueError):
        run("ssh-config", "slice")
    assert config.read_text() == block + other
    config.write_text(block + "# end dew tpu slice\n" + other)
    assert run("ssh-config", "slice") == 0
    assert other in config.read_text() and config.read_text().count("# dew tpu slice") == 1
    config.write_text(other + block + "# end dew tpu slice")
    assert run("ssh-config", "slice") == 0
    assert config.read_text().startswith(other) and config.read_text().count("# dew tpu slice") == 1


def test_delete_removes_the_queued_resource_that_holds_the_node(fake):
    fake.offer("slice", "us-central2-b", dict(
        NODE, queuedResource="projects/p/locations/us-central2-b/queuedResources/qr-slice"))
    assert run("delete", "slice") == 0
    assert fake.gcloud_calls()[-1] == gcloud(
        "compute", "tpus", "queued-resources", "delete", "qr-slice",
        "--zone=us-central2-b", "--force", "--quiet")
    assert json.loads((config_dir() / "zones.json").read_text()) == {}


def test_stop_then_start(fake):
    fake.offer("slice", "us-central2-b")
    assert run("stop", "slice") == 0
    assert run("start", "slice") == 0
    assert only(fake.gcloud_calls(), "stop") == [gcloud(
        "compute", "tpus", "tpu-vm", "stop", "slice", "--zone=us-central2-b")]
    assert only(fake.gcloud_calls(), "start") == [gcloud(
        "compute", "tpus", "tpu-vm", "start", "slice", "--zone=us-central2-b")]
    # Each command confirms where the TPU is in its zone's listing, and start
    # waits for READY afterwards.
    assert [call[3] for call in fake.gcloud_calls()] == [
        "list", "stop", "list", "start", "list"]


# ------------------------------------------------------------------------ access


def test_ssh_forwards_ports_and_passes_extra_args(fake):
    fake.offer("slice", "us-central2-b")
    assert run("ssh", "slice", "-L", "8888", "-L", "6006", "--", "-vv") == 0
    assert fake.gcloud_calls()[-1] == [
        *gcloud(
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            "you@slice",
            "--zone=us-central2-b",
            "--worker=0",
            "--ssh-flag=-L 8888:localhost:8888",
            "--ssh-flag=-L 6006:localhost:6006",
        ),
        "--",
        "-vv",
    ]


def test_list_shows_every_tpu_of_the_zones_it_reads(fake, capsys):
    """`list` reads every configured zone, a row a TPU with its zone and state, `--zone` one zone, and
    a zone with no TPU lists none and succeeds."""
    fake.offer("slice", "us-central2-b")
    fake.offer("spare", "europe-west4-a", dict(NODE, state="STOPPED", schedulingConfig={"spot": True}))
    assert run("list") == 0
    rows = {row[0]: row[1:] for row in map(str.split, printed(capsys)[1:])}
    assert {name: (row[1], row[4], row[5]) for name, row in rows.items()} == {
        "slice": ("READY", "us-central2-b", "-"), "spare": ("STOPPED", "europe-west4-a", "yes")}
    assert run("list", "--zone", "europe-west4-a") == 0
    assert [row.split()[0] for row in printed(capsys)[1:]] == ["spare"]
    assert run("list", "--zone", "us-east1-d") == 0
    assert not {"slice", "spare"} & set(" ".join(printed(capsys)).split())


def test_help_shows_the_configured_defaults_of_the_fields_its_command_reads(fake, capsys):
    """A command's --help ends with the configured value of each field it falls back to, and of no
    other: setup reads the bucket it mounts and not the disk create attaches, and create the reverse."""
    tpu_config.save(replace(CONFIG, data_disk="scratch-disk"))
    for command, read, unread in (("setup", "my-bucket", "scratch-disk"),
                                  ("create", "scratch-disk", "my-bucket")):
        with pytest.raises(SystemExit) as helped:
            run(command, "--help")
        assert helped.value.code == 0
        defaults = capsys.readouterr().out.split(str(config_dir() / "tpu.toml"))[-1]
        assert read in defaults and unread not in defaults and "my-project" in defaults


# ----------------------------------------------------------------------- fan-out


def test_run_reports_the_first_worker_that_failed(fake):
    fake.offer("slice", "us-central2-b")
    fake.fail_on("--worker=1")
    assert run("run", "slice", "--", "false") == 3


def test_a_command_reaches_the_workers_it_names_and_a_detached_one_logs_on_each(fake, capsys, tmp_path):
    """`run` runs its command on every worker, or on the one `--worker` names; `--detach` starts it
    under a job whose log each worker keeps, which `logs` reads back from worker 0, or from each with
    `--worker all`; a job name that is no directory name is refused before any worker is reached. `copy`
    puts a file on every worker, and a directory, whole, on the one it names."""
    fake.offer("slice", "us-central2-b")
    fake.host(worker_tools())
    homes = [fake.home("slice", worker) for worker in (0, 1)]
    assert run("run", "slice", "--", "touch", "reached") == 0
    assert run("run", "slice", "--worker", "1", "--", "touch", "one") == 0
    assert [(home / "reached").is_file() for home in homes] == [True, True]
    assert [(home / "one").is_file() for home in homes] == [False, True]
    assert run("run", "slice", "--detach", "--job", "greet", "--", "sh", "-c", "echo hello from $HOME") == 0
    assert [(home / "dew-runs" / "greet" / f"worker-{worker}.log").read_text()
            for worker, home in enumerate(homes)] == [f"hello from {home}\n" for home in homes]
    printed(capsys)
    assert run("logs", "slice", "greet") == 0
    assert printed(capsys) == [f"[worker 0] hello from {homes[0]}"]
    assert run("logs", "slice", "greet", "--worker", "all", "--follow") == 0
    assert sorted(printed(capsys)) == [f"[worker {worker}] hello from {home}"
                                       for worker, home in enumerate(homes)]
    reached = len(fake.calls)
    with pytest.raises(SystemExit):
        run("logs", "slice", "../greet")
    assert len(fake.calls) == reached
    (tmp_path / "data.txt").write_text("rows")
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards" / "0.jsonl").write_text("{}")
    assert run("copy", "slice", str(tmp_path / "data.txt"), "~/data.txt") == 0
    assert run("copy", "slice", str(tmp_path / "shards"), "~/shards", "--worker", "1") == 0
    assert [(home / "data.txt").read_text() for home in homes] == ["rows", "rows"]
    assert [(home / "shards" / "0.jsonl").is_file() for home in homes] == [False, True]


def test_status_and_reset_read_and_free_each_workers_accelerators(fake, capsys):
    """`status` shows a row a worker, in worker order, of its uptime, Dew's processes, the processes
    holding its accelerators and its accelerators, as that worker reports them; `reset` kills every
    process holding a worker's accelerators, each once, on every worker."""
    fake.offer("slice", "us-central2-b")
    fake.host(worker_tools())
    assert run("status", "slice") == 0
    rows = [line.split() for line in printed(capsys)[2:]]
    assert rows == [[str(worker), str(worker), "hours", "2", "2", "2"] for worker in (0, 1)]
    assert run("reset", "slice") == 0
    assert [(fake.home("slice", worker) / "killed").read_text().split() for worker in (0, 1)] == [
        ["-9", "4242", "4343"]] * 2


@pytest.mark.parametrize("project", ["configured", "gcloud"])
def test_attach_disk_mounts_the_disk_on_every_worker_once_it_is_attached(fake, project):
    """`attach-disk` attaches the zone's disk of the project the config names, or else gcloud's own,
    to the TPU, read-only where asked, then mounts it on every worker, read-only too; an attach that
    fails mounts nothing. Each worker's system tools are stand-ins, as no test is a TPU VM's root."""
    if project != "configured":
        tpu_config.save(replace(CONFIG, project=""))
        fake.state["gcloud_project"] = "their-project"
    fake.offer("slice", "us-central2-b")
    disk = {name: f'echo "{name} $*" >> "$HOME/mounted"' for name in ("mkdir", "mount", "chmod")}
    fake.host(worker_tools(**disk, mountpoint="exit 1", grep="exit 0"))
    assert run("attach-disk", "slice", "shared", "--read-only") == 0
    [attached] = only(fake.gcloud_calls(), "attach-disk")
    named = "my-project" if project == "configured" else "their-project"
    assert {f"--disk=projects/{named}/zones/us-central2-b/disks/shared", "--mode=read-only"} <= set(attached)
    mounted = [fake.home("slice", worker) / "mounted" for worker in (0, 1)]
    assert [path.read_text().splitlines() for path in mounted] == [
        ["mkdir -p /mnt/persist", "mount -o ro /dev/sdb /mnt/persist"]] * 2
    for path in mounted:
        path.unlink()
    fake.fail_on("attach-disk")
    assert run("attach-disk", "slice", "shared") == 3
    assert not any(path.exists() for path in mounted)


# ------------------------------------------------------------------ dew launch --tpu


def launch(*argv: str) -> int:
    from dew.cli.main import main

    return main(["launch", *argv])


def pool_ssh(worker: int, name: str, command: str) -> list[str]:
    return gcloud("compute", "tpus", "tpu-vm", "ssh", f"you@{name}", "--zone=us-central2-b",
                  f"--worker={worker}", f"--command={env_prefix()}{command}", "--ssh-flag=-tt")


def test_launch_on_several_tpus_joins_them_as_one_multislice_pool(fake):
    """Each slice's workers learn the slice count, their slice and the first
    worker of slice 0, which is where the slices meet and jax's coordinator."""
    for name in ("left", "right"):
        fake.offer(name, "us-central2-b")
    assert launch("--tpu", "left,right", "--", "python", "train.py") == 0
    calls = only(fake.gcloud_calls(), "ssh")
    expected = []
    for slice_id, name in enumerate(("left", "right")):
        env = {"MEGASCALE_COORDINATOR_ADDRESS": "10.0.0.1", "MEGASCALE_PORT": "8081",
               "MEGASCALE_NUM_SLICES": "2", "MEGASCALE_SLICE_ID": str(slice_id)}
        command = remote_script(("python", "train.py"), env, None)
        expected += [pool_ssh(0, name, command), pool_ssh(1, name, command)]
    assert sorted(calls) == sorted(expected)


def test_launch_reaches_a_tpu_only_once_it_is_ready_in_its_zone(fake):
    """`dew launch --tpu` starts nothing on a slice that is still being created, nor on one the zone
    it names does not list; once the slice is READY it starts its program on every worker."""
    fake.offer("slice", "us-central2-b", dict(NODE, state="CREATING"))
    with pytest.raises(SystemExit):
        launch("--tpu", "slice", "--", "python", "train.py")
    fake.offer("slice", "us-central2-b")
    with pytest.raises(SystemExit):
        launch("--tpu", "slice", "--zone", "us-east1-d", "--", "python", "train.py")
    assert only(fake.gcloud_calls(), "ssh") == []
    assert launch("--tpu", "slice", "--", "python", "train.py") == 0
    assert sorted(call[6] for call in only(fake.gcloud_calls(), "ssh")) == ["--worker=0", "--worker=1"]


# ----------------------------------------------------------------------- syncing


def test_train_runs_the_synced_recipe_on_every_worker_and_follows_worker_zero(fake, capsys, tmp_path,
                                                                             monkeypatch):
    """`train` copies the working tree to every worker, each at its own address, starts the recipe there
    as one multi-host job, each worker logging its own output, and follows worker 0's log. A sync keeps
    what a worker holds beside the tree, and mirrors the tree with `--delete`; what it excludes is left
    alone either way."""
    monkeypatch.chdir(project(tmp_path / "project"))
    fake.offer("slice", "us-central2-b")
    fake.host(worker_tools())
    homes = [fake.home("slice", worker) / "project" for worker in (0, 1)]
    assert run("train", "slice", "--job", "fit", "--", "recipe.py", "--steps", "3") == 0
    ran = "recipe --steps 3 --trainer.multi-host True"
    assert [(home.parent / "dew-runs" / "fit" / f"worker-{worker}.log").read_text()
            for worker, home in enumerate(homes)] == [f"{ran}\n"] * 2
    assert f"[worker 0] {ran}" in printed(capsys)
    for home in homes:
        (home / "outputs").mkdir()
        (home / "outputs" / "kept.txt").write_text("")
        (home / "notes.txt").write_text("")
    assert run("sync", "slice", "--exclude", "outputs") == 0
    assert all((home / "notes.txt").is_file() for home in homes)
    assert run("sync", "slice", "--exclude", "outputs", "--delete") == 0
    assert [sorted(path.relative_to(home).as_posix() for path in home.rglob("*") if path.is_file())
            for home in homes] == [["outputs/kept.txt", "recipe.py"]] * 2


# ------------------------------------------------------------------------- setup


def on_a_worker(tmp_path: Path, script: str, scripts: dict[str, str]) -> list[list[str]]:
    """`script` run as a worker runs it, under bash, with stand-ins on PATH for its package manager and
    privileged commands, each recording being called, and `scripts` (a name to its body after the line
    that records the call); the calls, each its command and arguments."""
    home, binaries, calls = tmp_path / "home", tmp_path / "bin", tmp_path / "calls"
    home.mkdir()
    binaries.mkdir()
    for name in ("sudo", "curl", "gcsfuse", "rsync", "lsb_release", *scripts):
        body = scripts.get(name, "")
        (binaries / name).write_text(f'#!/bin/sh\necho "$(basename "$0") $*" >> {calls}\n{body}\n')
        (binaries / name).chmod(0o755)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60, env={
        "HOME": str(home), "PATH": f"{binaries}:/usr/bin:/bin", "STAND_INS": str(tmp_path)})
    assert done.returncode == 0, done.stderr
    return [line.split() for line in calls.read_text().splitlines()]


UV = ('[ "$1" = --version ] && echo "uv @RELEASE@"; if [ "$1" = venv ]; then mkdir -p "$4/bin"; '
      'printf \'#!/bin/sh\\necho 3.12\\n\' > "$4/bin/python"; chmod +x "$4/bin/python"; fi')
"""uv: it names release @RELEASE@, and makes the venv it is asked for, whose interpreter answers its
version."""


@pytest.mark.parametrize("source, version", [("", "1.2.3"), ("dew", "")], ids=["release", "editable"])
def test_a_worker_installs_dew_with_its_tpu_extra_in_one_resolution(tmp_path, source, version):
    """The setup script a worker runs installs Dew once, a release or the synced source, with the tpu
    extra that pins the JAX range and the libtpu release Dew is built against beside the extras asked
    for, and Dew's package index beside PyPI for Dew's JAX build: no JAX resolved apart first, which
    would pick its own TPU runtime and leave it to a second resolve. Run as a worker runs it, with its
    package manager and privileged commands stand-ins that record being called."""
    spec, editable = tpu_setup.package_spec(source, "wandb", version)
    script = tpu_setup.render(python_version="3.12", package_spec=spec, editable=editable)
    calls = on_a_worker(tmp_path, script, {"uv": UV.replace("@RELEASE@", "99.0.0")})
    installs = [call[1:] for call in calls if call[:3] == ["uv", "pip", "install"]]
    target = f"{tmp_path}/home/dew[tpu,wandb]" if editable else "dewml[tpu,wandb]==1.2.3"
    assert installs == [["pip", "install", "--quiet", "--python", f"{tmp_path}/home/dew-venv/bin/python",
                         "--extra-index-url", PACKAGE_INDEX, *(["-e"] if editable else []), target]]


def test_a_fresh_worker_installs_the_least_uv_first_and_dew_with_it(tmp_path):
    """A worker with no uv runs astral's installer for the least uv (pyproject.toml's [tool.uv]
    required-version) under the script's own `set -euo pipefail`, and the uv it installed then installs
    Dew. curl stands in for astral's installer, putting a uv of the release its URL names in ~/.local/bin,
    which records its calls under that release's name."""
    spec, editable = tpu_setup.package_spec("", "wandb", "1.2.3")
    script = tpu_setup.render(python_version="3.12", package_spec=spec, editable=editable)
    (tmp_path / "uv.template").write_text(
        f'#!/bin/sh\necho "uv-@RELEASE@ $*" >> {tmp_path}/calls\n{UV}\n')
    curl = ('for url; do :; done; release=${url#*/uv/}; release=${release%/install.sh}; '
            'printf \'mkdir -p "$HOME/.local/bin"; sed "s/@RELEASE@/%s/g" "$STAND_INS/uv.template" '
            '> "$HOME/.local/bin/uv"; chmod +x "$HOME/.local/bin/uv"\\n\' "$release"')
    calls = on_a_worker(tmp_path, script, {"curl": curl})
    [fetched] = [call[-1] for call in calls
                 if call[0] == "curl" and call[-1].startswith("https://astral.sh/uv/")]
    release = fetched.removeprefix("https://astral.sh/uv/").removesuffix("/install.sh")
    installs = [call[:3] for call in calls if call[1:3] == ["pip", "install"]]
    assert installs == [[f"uv-{release}", "pip", "install"]]


def test_setup_from_source_installs_the_synced_tree_where_every_worker_sees_the_slice(fake, tmp_path,
                                                                                     monkeypatch):
    """`setup --from-source` syncs the working tree to every worker before it sends them the setup
    script, which, run as a worker runs it, installs that tree editable with the extras asked for; then
    every worker runs it and is asked how many devices it sees, and a worker that sees half the slice
    fails the setup."""
    monkeypatch.chdir(project(tmp_path / "project"))
    fake.offer("slice", "us-central2-b", devices="16 8")
    assert run("setup", "slice", "--from-source", "--extras", "tfds") == 0
    reached = [call[0] if call[0] == "rsync" else call[4] for call in fake.calls
               if call[0] == "rsync" or call[4] in ("scp", "ssh")]
    assert reached == ["rsync", "rsync", "scp", *["ssh"] * 4]
    [sent] = only(fake.gcloud_calls(), "scp")
    assert "--worker=all" in sent
    assert sorted(call[6] for call in only(fake.gcloud_calls(), "ssh")) == ["--worker=0", "--worker=0",
                                                                           "--worker=1", "--worker=1"]
    (tmp_path / "worker").mkdir()
    uv = {"uv": UV.replace("@RELEASE@", "99.0.0")}
    calls = on_a_worker(tmp_path / "worker", Path(sent[4]).read_text(), uv)
    [install] = [call[1:] for call in calls if call[:3] == ["uv", "pip", "install"]]
    assert install[-2:] == ["-e", f"{tmp_path}/worker/home/project[tpu,tfds]"]
    fake.offer("slice", "us-central2-b", devices="8 8")
    assert run("setup", "slice") == 1


# ------------------------------------------------------------------------- spawn


def test_spawn_sets_up_and_starts_each_tpu_it_created_and_reports_one_that_failed(fake, capsys):
    """`spawn` creates its TPUs, sets each up and starts the command on each one's workers as a job of
    its own, and ends with a row a TPU; a TPU whose creation fails, or whose setup does as a worker
    sees too few devices, is reported failed and runs nothing, while the others run the command."""
    single = dict(NODE, acceleratorType="v5litepod-8", networkEndpoints=NODE["networkEndpoints"][:1])
    for name, devices in (("sw-0", "8 8"), ("sw-1", "8 8"), ("sw-2", "4 4")):
        fake.offer(name, "us-central2-b", single, devices=devices)
    fake.host(worker_tools())
    fake.fail_on("create sw-1")
    assert run("spawn", "sw", "3", "--type", "v5e-8", "--", "touch", "started") == 1
    lines = printed(capsys)
    header = max(index for index, line in enumerate(lines) if line.split()[:2] == ["NAME", "STATE"])
    table = lines[header + 1:]
    assert {row.split()[0]: row.split()[1] for row in table} == {
        "sw-0": "ready", "sw-1": "failed:", "sw-2": "failed:"}
    assert [(fake.home(name, 0) / "started").is_file() for name in ("sw-0", "sw-1", "sw-2")] == [
        True, False, False]


# ------------------------------------------------------------------- zone lookup


def test_zone_search_follows_the_configured_order_and_caches(fake):
    fake.offer("slice", "europe-west4-a")
    assert run("describe", "slice") == 0
    assert zones_seen(fake.gcloud_calls(), "list") == [
        "us-central2-b", "europe-west4-a", "europe-west4-a"]
    assert json.loads((config_dir() / "zones.json").read_text()) == {"slice": "europe-west4-a"}
    fake.calls_path.unlink()
    assert run("describe", "slice") == 0
    assert zones_seen(fake.gcloud_calls(), "list") == [
        "europe-west4-a", "europe-west4-a"]


def test_a_cached_zone_that_stopped_answering_is_replaced(fake):
    """A TPU recreated in another zone is found by the zone search and its
    cache entry replaced; every command would otherwise fail with 'not in Z.
    Pass --zone' until the cache file was edited by hand."""
    tpu_config.cache_zone("slice", "us-east1-d")
    fake.offer("slice", "europe-west4-a")
    assert run("describe", "slice") == 0
    assert zones_seen(fake.gcloud_calls(), "list") == [
        "us-east1-d", "us-central2-b", "europe-west4-a", "europe-west4-a"]
    assert json.loads((config_dir() / "zones.json").read_text()) == {
        "slice": "europe-west4-a"}


def test_a_tpu_in_none_of_the_zones_reaches_no_worker_and_leaves_no_cached_zone(fake):
    """A TPU that no configured zone lists stops the command before any worker is asked to run it, and
    the zone it was cached in, which no longer lists it, is forgotten."""
    tpu_config.cache_zone("gone", "us-east1-d")
    with pytest.raises(SystemExit):
        run("run", "gone", "--", "true")
    assert zones_seen(fake.gcloud_calls(), "list") == ["us-east1-d", "us-central2-b", "europe-west4-a"]
    assert only(fake.gcloud_calls(), "ssh") == []
    assert json.loads((config_dir() / "zones.json").read_text()) == {}


@pytest.mark.parametrize("command", [("list",), ("describe", "slice"),
                                     ("describe", "slice", "--zone", "us-east1-d")])
def test_a_gcloud_that_fails_stops_the_command(fake, command):
    """gcloud failing (an expired login, a missing permission) stops the command, not an answer of
    nothing there: no TPUs listed and success, or a TPU in none of the zones, or one not yet READY."""
    fake.offer("slice", "us-east1-d")
    fake.fail_on("list")
    with pytest.raises(SystemExit) as stopped:
        run(*command)
    assert stopped.value.code not in (0, None)


def test_the_zone_flag_skips_the_search_and_is_not_cached(fake):
    fake.offer("slice", "us-east1-d")
    assert run("describe", "slice", "--zone", "us-east1-d") == 0
    assert zones_seen(fake.gcloud_calls(), "list") == ["us-east1-d"]
    # An unverified flag must not decide where later commands look.
    assert not (config_dir() / "zones.json").exists()


# ---------------------------------------------------------------------- config


def test_config_round_trips_through_toml(fake, tmp_path):
    cfg = replace(CONFIG, zones=("a", "b"), gcs_bucket='has "quotes"', data_disk="d")
    path = tpu_config.save(cfg, tmp_path / "round.toml")
    assert tpu_config.load(path) == cfg


def test_init_writes_the_config_from_flags(fake):
    assert run("init", "--project", "p2", "--zones", "us-central2-a, us-east1-d",
               "--accelerator-type", "v5e-8") == 0
    assert tpu_config.load() == replace(
        CONFIG, project="p2", zones=("us-central2-a", "us-east1-d"),
        accelerator_type="v5e-8")


def _write_process_zone_cache(directory, worker):
    import os

    from dew.cli import config

    os.environ["DEW_CONFIG_DIR"] = directory
    for index in range(12):
        config.cache_zone(f"slice-{worker}-{index}", "us-central2-b")
    config.forget_zone(f"old-{worker}")


def test_zone_cache_preserves_parallel_process_updates(tmp_path, monkeypatch):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    directory = tmp_path / "zones"
    monkeypatch.setenv("DEW_CONFIG_DIR", str(directory))
    for worker in range(3):
        tpu_config.cache_zone(f"old-{worker}", "us-east1-d")
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_write_process_zone_cache, str(directory), worker)
                   for worker in range(3)]
        for future in futures:
            future.result(timeout=30)
    expected = {f"slice-{worker}-{index}": "us-central2-b"
                for worker in range(3) for index in range(12)}
    assert json.loads((directory / "zones.json").read_text()) == expected


def test_zone_cache_readers_never_observe_partial_writes(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    directory = tmp_path / "zones"
    monkeypatch.setenv("DEW_CONFIG_DIR", str(directory))
    tpu_config.cache_zone("stable", "us-east1-d")
    start = Barrier(4)

    def write_and_read(worker):
        start.wait(timeout=10)
        for index in range(24):
            tpu_config.cache_zone(f"slice-{worker}-{index}", "us-central2-b")
            assert tpu_config.cached_zone("stable") == "us-east1-d"

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write_and_read, range(4)))
    expected = {f"slice-{worker}-{index}": "us-central2-b"
                for worker in range(4) for index in range(24)}
    expected["stable"] = "us-east1-d"
    assert json.loads((directory / "zones.json").read_text()) == expected


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("mounted", [True, False])
def test_the_disk_script_opens_and_records_the_mount_once_the_disk_is_mounted(tmp_path, mounted, read_only):
    """The attached disk's script (a VM's boot, and attach-disk on every worker) mounts the disk, then
    opens the mount to every user and names it in fstab; a disk attached read-only, which several
    workers share, it mounts read-only and leaves as it is. A disk that does not mount fails the
    script before either, so nothing writes to the boot disk's directory as though it were the disk.
    The system's tools are stood in for on PATH, as no test runs as a TPU VM's root."""
    import shutil
    import subprocess

    from dew.cli import tpu_setup

    tools = tmp_path / "bin"
    tools.mkdir()
    calls = tmp_path / "calls"
    answers = {"mkdir": 0, "mountpoint": 1, "mount": 0 if mounted else 32, "chmod": 0, "grep": 0}
    for name, code in answers.items():
        (tools / name).write_text(f'#!/bin/sh\necho "{name} $*" >> {calls}\nexit {code}\n')
        (tools / name).chmod(0o755)
    shell = shutil.which("bash")
    assert shell is not None
    done = subprocess.run([shell, "-c", tpu_setup.disk_startup_script(read_only)], env={"PATH": str(tools)},
                          capture_output=True, text=True)
    ran = calls.read_text().splitlines()
    mount = "mount -o ro /dev/sdb /mnt/persist" if read_only else "mount /dev/sdb /mnt/persist"
    if mounted:
        assert done.returncode == 0, done.stderr
        assert [line.split()[0] for line in ran] == ["mkdir", "mountpoint", "mount",
                                                     *(() if read_only else ("chmod",)), "grep"]
        assert mount in ran
    else:
        assert done.returncode == 32
        assert [line.split()[0] for line in ran] == ["mkdir", "mountpoint", "mount"]


def test_copy_sends_a_directory_recursively_to_every_worker(fake, tmp_path):
    fake.offer("slice", "us-central2-b")
    (tmp_path / "payload").mkdir()
    assert run("copy", "slice", str(tmp_path / "payload"), "~/payload") == 0
    assert only(fake.gcloud_calls(), "scp") == [gcloud(
        "compute", "tpus", "tpu-vm", "scp", str(tmp_path / "payload"), "you@slice:~/payload",
        "--zone=us-central2-b", "--worker=all", "--recurse")]


def test_sync_rsyncs_the_working_tree_to_each_worker_with_its_excludes(fake, tmp_path, monkeypatch):
    """`sync` copies the git working tree to each worker's address, its excludes and `--delete`
    passed to every rsync, the workers at once in any order."""
    tree = project(tmp_path / "tree")
    monkeypatch.chdir(tree)
    fake.offer("slice", "us-central2-b")
    assert run("sync", "slice", "--exclude", "runs", "--delete") == 0
    synced = fake.rsync_calls()
    assert sorted(call[-1] for call in synced) == [f"you@34.0.0.{index}:~/tree/" for index in (1, 2)]
    assert all("--delete" in call and "--exclude=runs" in call and call[-2] == f"{tree}/"
               for call in synced)
