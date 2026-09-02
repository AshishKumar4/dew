"""dew-tpu against a fake gcloud: argv per command, fan-out, dry runs, config."""

import json
import os
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from dew.cli import config as tpu_config
from dew.cli import tpu as tpu_cli
from dew.cli import tpu_setup
from dew.cli.gcloud import Node, Result, exit_code

REPO = Path(__file__).resolve().parents[1]

#: Stands in for gcloud and rsync. Records every argv, answers from a state file.
FAKE = '''#!/usr/bin/env python3
import json
import os
import sys

argv = sys.argv[1:]
program = os.path.basename(sys.argv[0])
state = json.load(open(os.environ["DEW_FAKE_STATE"]))
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


fail = state.get("fail", "")
if fail and fail in " ".join(argv):
    die("the fake was told to fail here", 3)

if program == "rsync":
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
    print(json.dumps(state.get("listing", {}).get(flag("zone"), [])))
    raise SystemExit(0)

if verb == "describe":
    node = state["nodes"].get(argv[4])
    if node is None or node["zone"] != flag("zone"):
        die("NOT_FOUND: the fake has no such node")
    if "--format=json" not in argv:
        print(argv[4])
        raise SystemExit(0)
    states = node.get("states")
    if states:
        # Each describe sees the next state, and stays on the last one.
        node["payload"]["state"] = states[0]
        if len(states) > 1:
            node["states"] = states[1:]
            json.dump(state, open(os.environ["DEW_FAKE_STATE"], "w"))
    print(json.dumps(node["payload"]))
    raise SystemExit(0)

if verb == "ssh":
    node = state["nodes"].get(argv[4].split("@")[-1], {})
    command = flag("command")
    if "jax.device_count" in command:
        print(node.get("devices", "16 8"))
    elif command.startswith("printf"):
        print(node.get("status", "up 2 hours|0|0|8"))
    elif command:
        print("ran " + command)
    raise SystemExit(0)

if verb in ("create", "delete", "start", "stop", "scp"):
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
        self.state = {"nodes": {}, "listing": {}}
        self.flush()

    def flush(self) -> None:
        self.state_path.write_text(json.dumps(self.state))

    def offer(self, name: str, zone: str, payload: dict = NODE, **extra) -> None:
        """Tell the fake that a TPU exists, and what describe should answer."""
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
    monkeypatch.setattr(tpu_cli, "POLL_SECONDS", 0)
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


def ssh_commands(calls: list[list[str]]) -> list[str]:
    """Every --command value, in the order the calls happened."""
    return [part.split("=", 1)[1] for call in calls for part in call
            if part.startswith("--command=")]


def only(calls: list[list[str]], verb: str) -> list[list[str]]:
    return [call for call in calls if len(call) > 3 and call[3] == verb]


# --------------------------------------------------------------------- lifecycle


def test_create_uses_the_configured_defaults(fake):
    fake.offer("slice", "us-central2-b")
    assert run("create", "slice") == 0
    assert fake.gcloud_calls()[0] == gcloud(
        "compute", "tpus", "tpu-vm", "create", "slice", "--zone=us-central2-b",
        "--accelerator-type=v5litepod-16", "--version=v2-alpha-tpuv5-lite")
    assert json.loads((config_dir() / "zones.json").read_text()) == {"slice": "us-central2-b"}


def test_create_waits_for_ready_then_prints_the_workers(fake, capsys):
    fake.offer("slice", "us-central2-b", states=["CREATING", "READY"])
    assert run("create", "slice") == 0
    out = capsys.readouterr().out
    assert "slice is CREATING" in out
    assert "v5litepod-16 on 2 worker(s)" in out
    assert "10.0.0.2" in out and "34.0.0.2" in out
    describes = [call for call in fake.gcloud_calls() if "describe" in call]
    assert len(describes) == 2


def test_create_gives_up_on_a_dead_state(fake):
    fake.offer("slice", "us-central2-b", states=["PREEMPTED"])
    with pytest.raises(SystemExit, match="PREEMPTED"):
        run("create", "slice")


def test_create_spot_queued_disk_and_explicit_version(fake):
    fake.offer("slice", "europe-west4-a")
    assert run("create", "slice", "--type", "v4-32", "--spot", "--queued",
               "--version", "v2-alpha-tpuv4", "--disk", "disk-1",
               "--zone", "europe-west4-a") == 0
    script = config_dir() / "disk-startup.sh"
    assert fake.gcloud_calls()[0] == gcloud(
        "compute", "tpus", "queued-resources", "create", "slice",
        "--zone=europe-west4-a", "--accelerator-type=v4-32",
        "--runtime-version=v2-alpha-tpuv4", "--node-id=slice", "--spot",
        "--data-disk=mode=read-write,"
        "source=projects/my-project/zones/europe-west4-a/disks/disk-1",
        f"--metadata-from-file=startup-script={script}")
    assert "mount /dev/sdb /mnt/persist" in script.read_text()


def test_create_disk_reads_the_project_from_gcloud_when_the_config_has_none(fake):
    """The disk source is the only argv that needs the project spelled out, and
    an empty one builds projects//zones/... which gcloud rejects at create."""
    tpu_config.save(replace(CONFIG, project=""))
    fake.state["gcloud_project"] = "from-gcloud"
    fake.flush()
    fake.offer("slice", "us-central2-b")
    assert run("create", "slice", "--disk", "d1") == 0
    created = only(fake.gcloud_calls(), "create")[0]
    assert ("--data-disk=mode=read-write,"
            "source=projects/from-gcloud/zones/us-central2-b/disks/d1") in created


def test_create_disk_says_so_when_no_project_is_configured_anywhere(fake):
    tpu_config.save(replace(CONFIG, project=""))
    fake.offer("slice", "us-central2-b")
    with pytest.raises(SystemExit, match="needs a project"):
        run("create", "slice", "--disk", "d1")
    assert only(fake.gcloud_calls(), "create") == []


def test_delete_removes_the_queued_resource_that_holds_the_node(fake):
    fake.offer("slice", "us-central2-b", dict(
        NODE, queuedResource="projects/p/locations/us-central2-b/queuedResources/qr-slice"))
    assert run("delete", "slice") == 0
    assert fake.gcloud_calls()[-1] == gcloud(
        "compute", "tpus", "queued-resources", "delete", "qr-slice",
        "--zone=us-central2-b", "--force", "--quiet")
    assert json.loads((config_dir() / "zones.json").read_text()) == {}


def test_delete_plain_node(fake):
    fake.offer("slice", "us-central2-b")
    assert run("delete", "slice") == 0
    assert fake.gcloud_calls()[-1] == gcloud(
        "compute", "tpus", "tpu-vm", "delete", "slice", "--zone=us-central2-b", "--quiet")


def test_stop_then_start(fake):
    fake.offer("slice", "us-central2-b")
    assert run("stop", "slice") == 0
    assert run("start", "slice") == 0
    assert only(fake.gcloud_calls(), "stop") == [gcloud(
        "compute", "tpus", "tpu-vm", "stop", "slice", "--zone=us-central2-b")]
    assert only(fake.gcloud_calls(), "start") == [gcloud(
        "compute", "tpus", "tpu-vm", "start", "slice", "--zone=us-central2-b")]
    # start waits for READY again, so it describes afterwards.
    assert [call[3] for call in fake.gcloud_calls()] == [
        "describe", "stop", "start", "describe"]


def test_list_reads_every_configured_zone(fake, capsys):
    fake.listing({"us-east1-d": [dict(NODE, name="p/locations/us-east1-d/nodes/one",
                                      acceleratorType="v5e-8", health="",
                                      schedulingConfig={"spot": True},
                                      networkEndpoints=NODE["networkEndpoints"][:1])]})
    assert run("list") == 0
    assert fake.gcloud_calls() == [gcloud(
        "compute", "tpus", "tpu-vm", "list", f"--zone={zone}", "--format=json")
        for zone in CONFIG.zones]
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["NAME", "TYPE", "STATE", "HEALTH", "WORKERS", "ZONE", "SPOT"]
    assert lines[1].split() == ["one", "v5e-8", "READY", "-", "1", "us-east1-d", "yes"]


def test_list_says_so_when_there_is_nothing(fake, capsys):
    assert run("list") == 0
    assert "no TPUs in us-central2-b, europe-west4-a, us-east1-d" in capsys.readouterr().out


def test_describe_shows_the_fields_and_the_workers(fake, capsys):
    fake.offer("slice", "us-central2-b")
    assert run("describe", "slice") == 0
    assert fake.gcloud_calls() == [
        gcloud("compute", "tpus", "tpu-vm", "describe", "slice", "--zone=us-central2-b",
               "--format=value(name)"),
        gcloud("compute", "tpus", "tpu-vm", "describe", "slice", "--zone=us-central2-b",
               "--format=json"),
    ]
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert ["name", "slice"] in rows
    assert ["state", "READY"] in rows
    assert ["topology", "4x4"] in rows
    assert ["1", "10.0.0.2", "34.0.0.2"] in rows


def test_a_tpu_in_no_zone_is_an_error(fake):
    with pytest.raises(SystemExit, match="ghost is in none of"):
        run("describe", "ghost")
    assert zones_seen(fake.gcloud_calls(), "describe") == list(CONFIG.zones)


# ------------------------------------------------------------------------ access


def test_ssh_forwards_ports_and_passes_extra_args(fake):
    fake.offer("slice", "us-central2-b")
    assert run("ssh", "slice", "-L", "8888", "-L", "6006", "--", "-vv") == 0
    assert fake.gcloud_calls()[-1] == gcloud(
        "compute", "tpus", "tpu-vm", "ssh", "you@slice", "--zone=us-central2-b",
        "--worker=0", "--ssh-flag=-L 8888:localhost:8888",
        "--ssh-flag=-L 6006:localhost:6006") + ["--", "-vv"]


def test_copy_sends_a_file_to_every_worker(fake):
    fake.offer("slice", "us-central2-b")
    assert run("copy", "slice", "/tmp/one", "~/one") == 0
    assert fake.gcloud_calls()[-1] == gcloud(
        "compute", "tpus", "tpu-vm", "scp", "/tmp/one", "you@slice:~/one",
        "--zone=us-central2-b", "--worker=all")


def test_copy_recurses_a_directory(fake, tmp_path):
    fake.offer("slice", "us-central2-b")
    tree = tmp_path / "tree"
    tree.mkdir()
    assert run("copy", "slice", str(tree), "~/", "--worker", "1") == 0
    assert fake.gcloud_calls()[-1] == gcloud(
        "compute", "tpus", "tpu-vm", "scp", str(tree), "you@slice:~/",
        "--zone=us-central2-b", "--worker=1", "--recurse")


# ----------------------------------------------------------------------- fan-out


def test_run_reaches_every_worker_and_prefixes_each_line(fake, capsys):
    fake.offer("slice", "us-central2-b")
    assert run("run", "slice", "--", "echo", "hi there") == 0
    command = env_prefix() + "echo 'hi there'"
    # The workers run at the same time, so compare the set of calls.
    assert sorted(only(fake.gcloud_calls(), "ssh")) == sorted(
        [ssh("0", command), ssh("1", command)])
    out = capsys.readouterr().out.splitlines()
    assert sorted(out) == [f"[worker 0] ran {command}", f"[worker 1] ran {command}"]


def test_run_can_name_one_worker(fake):
    fake.offer("slice", "us-central2-b")
    assert run("run", "slice", "--worker", "1", "--", "uptime") == 0
    assert fake.gcloud_calls()[-1] == ssh("1", env_prefix() + "uptime")


def test_run_reports_the_first_worker_that_failed(fake):
    fake.offer("slice", "us-central2-b")
    fake.fail_on("--worker=1")
    assert run("run", "slice", "--", "false") == 3


def test_run_detached_writes_a_log_per_worker(fake, capsys):
    fake.offer("slice", "us-central2-b")
    assert run("run", "slice", "--detach", "--job", "job1", "--", "python", "t.py") == 0
    for worker in (0, 1):
        command = (
            "mkdir -p $HOME/dew-runs/job1 && "
            f"nohup bash -c '{env_prefix()}python t.py' "
            f"> $HOME/dew-runs/job1/worker-{worker}.log 2>&1 < /dev/null & "
            'echo "job job1 pid $!"')
        assert ssh(str(worker), command) in fake.gcloud_calls()
    assert "dew-tpu logs slice job1 --follow" in capsys.readouterr().out


def test_logs_tails_one_worker_by_default_and_all_on_request(fake):
    fake.offer("slice", "us-central2-b")
    assert run("logs", "slice", "job1", "--lines", "50") == 0
    assert fake.gcloud_calls()[-1] == ssh(
        "0", "tail -n 50 $HOME/dew-runs/job1/worker-0.log")
    assert run("logs", "slice", "job1", "--worker", "all", "--follow") == 0
    assert sorted(fake.gcloud_calls()[-2:]) == sorted(
        ssh(str(worker), f"tail -f -n 200 $HOME/dew-runs/job1/worker-{worker}.log")
        for worker in (0, 1))


def test_status_reads_each_worker(fake, capsys):
    fake.offer("slice", "us-central2-b", status="3 days|2|1|8")
    assert run("status", "slice") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "slice v5litepod-16 READY HEALTHY in us-central2-b"
    assert lines[1].split() == ["WORKER", "UPTIME", "DEW", "PROCS", "DEVICES", "BUSY", "DEVICES"]
    assert lines[-1].split() == ["1", "3", "days", "2", "1", "8"]


def test_a_worker_that_fails_a_captured_command_says_why(fake, capsys):
    fake.offer("slice", "us-central2-b")
    fake.fail_on("uptime -p")
    assert run("status", "slice") == 3
    out = capsys.readouterr().out
    assert "[worker 0] the fake was told to fail here" in out
    assert "[worker 1] the fake was told to fail here" in out


def test_reset_kills_what_holds_the_accelerators_on_every_worker(fake):
    fake.offer("slice", "us-central2-b")
    assert run("reset", "slice") == 0
    commands = [call[-2].split("=", 1)[1] for call in fake.gcloud_calls()[-2:]]
    assert len(commands) == 2
    for command in commands:
        assert "/dev/accel*" in command and "kill -9 $pids" in command


# ----------------------------------------------------------------------- syncing


def test_sync_rsyncs_the_working_tree_to_every_worker(fake, capsys):
    fake.offer("slice", "us-central2-b")
    assert run("sync", "slice") == 0
    shell = (f"ssh -i {Path.home()}/.ssh/google_compute_engine "
             "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")
    assert sorted(fake.rsync_calls()) == sorted(
        ["-az", "--exclude=.git", "--filter=:- .gitignore", "-e", shell,
         f"{REPO}/", f"you@34.0.0.{worker + 1}:~/{REPO.name}/"] for worker in (0, 1))
    assert f"~/{REPO.name}" in capsys.readouterr().out


def test_sync_mirrors_only_when_asked(fake):
    """rsync --delete removes remote files that are not in the local tree, so a
    worker's datasets and outputs survive a sync that did not ask to mirror."""
    fake.offer("slice", "us-central2-b")
    assert run("sync", "slice") == 0
    assert "--delete" not in fake.rsync_calls()[0]
    fake.calls_path.unlink()
    assert run("sync", "slice", "--delete") == 0
    assert fake.rsync_calls()[0][:2] == ["-az", "--delete"]


def test_train_and_setup_from_source_do_not_mirror_by_default(fake):
    fake.offer("slice", "us-central2-b")
    assert run("train", "slice", "--job", "run-1", "--", "recipes/lm/train.py") == 0
    assert run("setup", "slice", "--from-source") == 0
    assert all("--delete" not in call for call in fake.rsync_calls())
    fake.calls_path.unlink()
    assert run("train", "slice", "--job", "run-2", "--delete", "--",
               "recipes/lm/train.py") == 0
    assert all("--delete" in call for call in fake.rsync_calls())


def test_sync_adds_the_extra_excludes(fake):
    fake.offer("slice", "us-central2-b")
    assert run("sync", "slice", "--exclude", "wandb", "--exclude", "*.pth") == 0
    call = fake.rsync_calls()[0]
    assert call[1:5] == ["--exclude=.git", "--filter=:- .gitignore",
                         "--exclude=wandb", "--exclude=*.pth"]


# ------------------------------------------------------------------------- setup


def test_setup_from_source_syncs_installs_and_counts_the_devices(fake, capsys):
    fake.offer("slice", "us-central2-b", devices="16 8")
    assert run("setup", "slice", "--from-source", "--extras", "tfds,av") == 0
    script = config_dir() / "setup-slice.sh"
    assert len(fake.rsync_calls()) == 2
    assert only(fake.gcloud_calls(), "scp") == [gcloud(
        "compute", "tpus", "tpu-vm", "scp", str(script), "you@slice:~/dew-setup.sh",
        "--zone=us-central2-b", "--worker=all")]
    assert ssh_commands(fake.gcloud_calls()) == (
        ["bash ~/dew-setup.sh"] * 2 + [tpu_setup.DEVICE_COUNT] * 2)
    assert f"PACKAGE_SPEC='{REPO.name}[tfds,av]'" in script.read_text()
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert ["WORKER", "DEVICES", "LOCAL", "CHECK"] in rows
    assert ["0", "16", "8", "ok"] in rows


def test_setup_fails_when_a_worker_cannot_see_the_slice(fake, capsys):
    fake.offer("slice", "us-central2-b", devices="8 8")
    assert run("setup", "slice") == 1
    assert ["1", "8", "8", "want", "16"] in [
        line.split() for line in capsys.readouterr().out.splitlines()]


def test_setup_installs_a_release_when_asked(fake):
    fake.offer("slice", "us-central2-b")
    assert run("setup", "slice", "--version", "0.2.1", "--gcs-bucket", "") == 0
    text = (config_dir() / "setup-slice.sh").read_text()
    assert "PACKAGE_SPEC=dew-ml==0.2.1" in text
    assert "EDITABLE=0" in text
    assert "GCS_BUCKET=''" in text


def test_setup_script_renders_for_the_python_version_and_the_source(fake):
    script = tpu_setup.render(python_version="3.11", package_spec="dew[tfds]",
                              editable=True, gcs_bucket="bucket-1")
    assert script.startswith("#!/bin/bash\n")
    assert "PYTHON_VERSION=3.11" in script
    assert "JAX_SPEC='jax[tpu]'" in script
    assert "PACKAGE_SPEC='dew[tfds]'" in script
    assert "EDITABLE=1" in script
    assert "GCS_BUCKET=bucket-1" in script
    assert 'uv venv --python "$PYTHON_VERSION" "$VENV"' in script
    assert 'uv pip install --quiet --python "$PY" "$JAX_SPEC"' in script
    assert 'uv pip install --quiet --python "$PY" -e "$HOME/$PACKAGE_SPEC"' in script
    # The three things the old setup_tpu.sh existed for.
    assert "* soft nofile 1048576" in script and "DefaultLimitNOFILE=1048576" in script
    assert "ulimit -n 1048576" in script
    assert "gcsfuse --config-file" in script and "max-size-mb: 40960" in script
    assert "export TOKENIZERS_PARALLELISM=false" in script
    assert "export WANDB_CACHE_DIR=/tmp/wandb-cache" in script


def test_setup_script_is_bash_and_every_step_guards_itself(fake):
    script = tpu_setup.render(python_version="3.12", package_spec="dew-ml")
    checked = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert checked.returncode == 0, checked.stderr
    for guard in ("command -v gcsfuse", 'grep -q "dew nofile"', 'command -v uv',
                  '[ "$have" = "$PYTHON_VERSION" ]', "grep -q '.dew-env'",
                  'mountpoint -q "$MOUNT_PATH"', '"$(cat $override 2>/dev/null)" != "$wanted"'):
        assert guard in script


def test_package_spec_reads_source_extras_and_version():
    assert tpu_setup.package_spec("dew", "tfds,av", "") == ("dew[tfds,av]", True)
    assert tpu_setup.package_spec("", "tfds", "0.2.1") == ("dew-ml[tfds]==0.2.1", False)
    assert tpu_setup.package_spec("", "", "") == ("dew-ml", False)


def test_train_syncs_detaches_on_all_workers_then_follows_worker_zero(fake, capsys):
    fake.offer("slice", "us-central2-b")
    assert run("train", "slice", "--job", "run-1", "--",
               "recipes/lm/train.py", "--trainer.epochs", "4") == 0
    assert len(fake.rsync_calls()) == 2
    commands = ssh_commands(fake.gcloud_calls())
    started = [
        "mkdir -p $HOME/dew-runs/run-1 && "
        f"nohup bash -c '{env_prefix()}cd $HOME/{REPO.name} && python recipes/lm/train.py "
        "--trainer.epochs 4 --trainer.multi-host True' "
        f"> $HOME/dew-runs/run-1/worker-{worker}.log 2>&1 < /dev/null & "
        'echo "job run-1 pid $!"'
        for worker in (0, 1)]
    assert sorted(commands[:2]) == started
    assert commands[2] == "tail -f -n 200 $HOME/dew-runs/run-1/worker-0.log"
    assert "following worker 0" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ("train", "slice", "--job", "j 1", "--", "recipes/lm/train.py"),
    ("run", "slice", "--detach", "--job", "j;1", "--", "true"),
    ("logs", "slice", "../etc/passwd"),
])
def test_a_job_name_that_would_break_the_remote_path_is_refused(fake, argv):
    """The job is a directory under ~/dew-runs and a redirect target, so a name
    with a space or a separator sends the log somewhere logs cannot read it."""
    fake.offer("slice", "us-central2-b")
    flags, rest = tpu_cli._split(argv)
    with pytest.raises(SystemExit, match="job"):
        run(*flags, *(["--", *rest] if rest else []))
    assert not any("dew-runs" in command for command in ssh_commands(fake.gcloud_calls()))


def test_a_repo_path_with_a_space_reaches_the_remote_shell_quoted(fake, monkeypatch):
    fake.offer("slice", "us-central2-b")
    monkeypatch.setattr(tpu_cli, "_git_root", lambda: Path("/tmp/my repo"))
    assert run("train", "slice", "--job", "run-1", "--", "recipes/lm/train.py") == 0
    assert fake.rsync_calls()[0][-1] == "you@34.0.0.1:~/'my repo'/"
    # The inner command is quoted again for bash -c, so read it back as a shell
    # word instead of matching the escaped form.
    tokens = shlex.split(ssh_commands(fake.gcloud_calls())[0])
    assert "cd $HOME/'my repo' &&" in tokens[tokens.index("-c") + 1]


def test_spawn_creates_sets_up_and_runs_on_each_tpu(fake, capsys):
    for name in ("sw-0", "sw-1"):
        fake.offer(name, "us-central2-b",
                   dict(NODE, acceleratorType="v5litepod-8",
                        networkEndpoints=NODE["networkEndpoints"][:1]),
                   devices="8 8")
    assert run("spawn", "sw", "2", "--type", "v5e-8", "--", "python", "t.py") == 0
    created = sorted(call[4] for call in fake.gcloud_calls() if call[3] == "create")
    assert created == ["sw-0", "sw-1"]
    scripts = sorted(Path(call[4]).name for call in fake.gcloud_calls() if call[3] == "scp")
    assert scripts == ["setup-sw-0.sh", "setup-sw-1.sh"]
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert ["NAME", "STATE", "JOB"] in rows
    assert sum(row[:2] == ["sw-0", "ready"] or row[:2] == ["sw-1", "ready"] for row in rows) == 2


def test_spawn_reports_a_tpu_that_failed(fake, capsys):
    for name in ("sw-0", "sw-1"):
        fake.offer(name, "us-central2-b", devices="16 8")
    fake.fail_on("create sw-1")
    assert run("spawn", "sw", "2") == 1
    assert "failed" in capsys.readouterr().out


def test_spawn_does_not_launch_when_setup_fails_the_device_check(fake, capsys):
    """The device check is why setup has an exit code. A worker that sees half
    the slice has to stop the run, not be reported ready with a job."""
    for name in ("sw-0", "sw-1"):
        fake.offer(name, "us-central2-b", devices="8 8")
    assert run("spawn", "sw", "2", "--", "python", "t.py") == 1
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert ["sw-0", "failed:", "setup", "exit", "1", "-"] in rows
    assert not any("dew-runs" in command for command in ssh_commands(fake.gcloud_calls()))


# ----------------------------------------------------------------------- dry run


@pytest.mark.parametrize("argv,expected", [
    (("create", "slice"), "gcloud compute tpus tpu-vm create slice --zone=us-central2-b"),
    (("delete", "slice"), "gcloud compute tpus tpu-vm delete slice"),
    (("start", "slice"), "gcloud compute tpus tpu-vm start slice"),
    (("stop", "slice"), "gcloud compute tpus tpu-vm stop slice"),
    (("list",), "gcloud compute tpus tpu-vm list --zone=us-east1-d"),
    (("describe", "slice"), "describe slice --zone=us-central2-b --format=json"),
    (("ssh", "slice", "-L", "8888"), "--ssh-flag=-L 8888:localhost:8888"),
    (("run", "slice", "--", "uptime"), "--worker=1"),
    (("logs", "slice", "job1"), "worker-0.log"),
    (("copy", "slice", "a", "b"), "compute tpus tpu-vm scp a you@slice:b"),
    (("sync", "slice"), "rsync -az --exclude=.git"),
    (("setup", "slice"), "dew-setup.sh"),
    (("train", "slice", "--", "recipes/lm/train.py"), "--trainer.multi-host True"),
    (("status", "slice"), "uptime -p"),
    (("reset", "slice"), "/dev/accel*"),
    (("spawn", "base", "1"), "create base-0"),
    (("init",), 'project = "my-project"'),
])
def test_dry_run_prints_the_plan_and_calls_nothing(fake, capsys, argv, expected):
    flags, rest = tpu_cli._split(argv)
    assert run(*flags, "--dry-run", *(["--", *rest] if rest else [])) == 0
    assert expected in capsys.readouterr().out
    assert fake.calls == []


def test_dry_run_needs_no_cluster_and_reads_the_slice_from_the_config(fake, capsys):
    assert run("run", "unknown-tpu", "--dry-run", "--", "hostname") == 0
    out = capsys.readouterr().out
    assert out.count("gcloud compute tpus tpu-vm ssh") == 2  # v5e-16 is two workers
    assert fake.calls == []


# ------------------------------------------------------------------- zone lookup


def test_zone_search_follows_the_configured_order_and_caches(fake):
    fake.offer("slice", "europe-west4-a")
    assert run("describe", "slice") == 0
    assert zones_seen(fake.gcloud_calls(), "describe") == [
        "us-central2-b", "europe-west4-a", "europe-west4-a"]
    assert json.loads((config_dir() / "zones.json").read_text()) == {"slice": "europe-west4-a"}
    fake.calls_path.unlink()
    assert run("describe", "slice") == 0
    assert zones_seen(fake.gcloud_calls(), "describe") == ["europe-west4-a"]


def test_the_zone_flag_skips_the_search_and_is_not_cached(fake):
    fake.offer("slice", "us-east1-d")
    assert run("describe", "slice", "--zone", "us-east1-d") == 0
    assert zones_seen(fake.gcloud_calls(), "describe") == ["us-east1-d"]
    # An unverified flag must not decide where later commands look.
    assert not (config_dir() / "zones.json").exists()


# ---------------------------------------------------------------------- config


def test_config_round_trips_through_toml(fake, tmp_path):
    cfg = replace(CONFIG, zones=("a", "b"), gcs_bucket='has "quotes"', data_disk="d")
    path = tpu_config.save(cfg, tmp_path / "round.toml")
    assert tpu_config.load(path) == cfg


def test_a_missing_config_is_the_defaults(fake, tmp_path):
    assert tpu_config.load(tmp_path / "nothing.toml") == tpu_config.TpuConfig()


def test_an_unknown_config_key_is_an_error(fake, tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('project = "p"\nnope = 1\n')
    with pytest.raises(SystemExit, match="unknown keys nope"):
        tpu_config.load(path)


def test_init_writes_the_config_from_flags(fake):
    assert run("init", "--project", "p2", "--zones", "us-central2-a, us-east1-d",
               "--accelerator-type", "v5e-8") == 0
    assert tpu_config.load() == replace(
        CONFIG, project="p2", zones=("us-central2-a", "us-east1-d"),
        accelerator_type="v5e-8")


def test_init_dry_run_shows_the_file_without_writing_it(fake, capsys):
    assert run("init", "--project", "p2", "--dry-run") == 0
    assert 'project = "p2"' in capsys.readouterr().out
    assert tpu_config.load().project == "my-project"


# ------------------------------------------------------------- accelerator types


@pytest.mark.parametrize("kind,workers,devices,runtime", [
    ("v5e-8", 1, 8, "v2-alpha-tpuv5-lite"),
    ("v5e-16", 2, 16, "v2-alpha-tpuv5-lite"),
    ("v5e-32", 4, 32, "v2-alpha-tpuv5-lite"),
    ("v5litepod-16", 2, 16, "v2-alpha-tpuv5-lite"),
    ("v5p-8", 1, 4, "v2-alpha-tpuv5"),
    ("v5p-32", 4, 16, "v2-alpha-tpuv5"),
    ("v4-8", 1, 4, "tpu-ubuntu2204-base"),
    ("v4-32", 4, 16, "tpu-ubuntu2204-base"),
    ("v6e-8", 1, 8, "v2-alpha-tpuv6e"),
    ("v6e-256", 32, 256, "v2-alpha-tpuv6e"),
    ("v3-8", 1, 8, "tpu-ubuntu2204-base"),
])
def test_what_an_accelerator_type_implies(kind, workers, devices, runtime):
    assert tpu_config.worker_count(kind) == workers
    assert tpu_config.device_count(kind) == devices
    assert tpu_config.runtime_for(kind) == runtime


def test_v5e_is_v5litepod_on_the_wire():
    assert tpu_config.api_type("v5e-16") == "v5litepod-16"
    assert tpu_config.api_type("v5litepod-16") == "v5litepod-16"
    assert tpu_config.api_type("v4-32") == "v4-32"


def test_an_accelerator_type_without_a_size_is_an_error():
    with pytest.raises(SystemExit, match="cannot read a size"):
        tpu_config.worker_count("v5e")


# ------------------------------------------------------------------ describe json


def test_node_reads_the_describe_schema():
    node = Node.parse(NODE)
    assert (node.name, node.zone, node.state, node.health) == (
        "slice", "us-central2-b", "READY", "HEALTHY")
    assert node.accelerator_type == "v5litepod-16" and node.topology == "4x4"
    assert node.ips == ("34.0.0.1", "34.0.0.2")
    assert node.internal_ips == ("10.0.0.1", "10.0.0.2")
    assert node.workers == 2 and node.spot is False and node.queued_resource == ""


def test_node_reads_spot_preemptible_and_queued_resources():
    spot = Node.parse(dict(NODE, schedulingConfig={"spot": True}))
    old = Node.parse(dict(NODE, schedulingConfig={"preemptible": True}))
    queued = Node.parse(dict(NODE, queuedResource="p/locations/z/queuedResources/qr-1"))
    assert spot.spot is True and old.spot is True
    assert queued.queued_resource == "qr-1"


def test_the_first_failure_is_the_exit_code():
    assert exit_code([Result(("a",), 0), Result(("b",), 3), Result(("c",), 1)]) == 3
    assert exit_code([Result(("a",), 0)]) == 0
