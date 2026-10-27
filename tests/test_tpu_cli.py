"""dew tpu and dew launch --tpu against a fake gcloud: argv per command, fan-out, pools, dry runs, config."""

import json
import os
import shlex
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dew.builds import PACKAGE_INDEX
from dew.cli import config as tpu_config, tpu as tpu_cli, tpu_setup
from dew.cli.launch import remote_script

#: Stands in for gcloud and rsync. Records every argv, answers from a state file.
FAKE = '''#!/usr/bin/env python3
import json
import os
import sys
import subprocess

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
hang = state.get("hang", "")
if hang and hang in " ".join(argv):
    import time
    time.sleep(600)

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
    listed = state.get("listing", {}).get(flag("zone"), [])
    listed += [node["payload"] for node in state["nodes"].values() if node["zone"] == flag("zone")]
    print(json.dumps(listed))
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


# --------------------------------------------------------------------- lifecycle


def test_create_uses_the_configured_defaults(fake):
    fake.offer("slice", "us-central2-b")
    assert run("create", "slice") == 0
    assert fake.gcloud_calls()[0] == gcloud(
        "compute", "tpus", "tpu-vm", "create", "slice", "--zone=us-central2-b",
        "--accelerator-type=v5litepod-16", "--version=v2-alpha-tpuv5-lite")
    assert json.loads((config_dir() / "zones.json").read_text()) == {"slice": "us-central2-b"}


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


# ----------------------------------------------------------------------- fan-out


def test_run_reports_the_first_worker_that_failed(fake):
    fake.offer("slice", "us-central2-b")
    fake.fail_on("--worker=1")
    assert run("run", "slice", "--", "false") == 3


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


# ----------------------------------------------------------------------- syncing


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


# ----------------------------------------------------------------------- dry run


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


# ------------------------------------------------------------- accelerator types


# ------------------------------------------------------------------ describe json


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


# ------------------------------------------------------------ every other command


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """A git working tree as the current directory, the one `sync` copies."""
    root = tmp_path / "tree"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    monkeypatch.chdir(root)
    return root


def commands(fake: Fake, verb: str) -> list[tuple[str, str, str]]:
    """The remote commands the fake was asked to run over ssh, sorted, each its TPU, its worker and
    the command, the venv prefix off: a fan-out's workers run at once, in any order."""
    def value(call, flag):
        return next(part.split("=", 1)[1] for part in call if part.startswith(f"--{flag}="))

    ran = [call for call in only(fake.gcloud_calls(), verb)
           if any(part.startswith("--command=") for part in call)]
    return sorted((call[4].split("@")[-1], value(call, "worker"),
                   value(call, "command").removeprefix(env_prefix())) for call in ran)


def each(command: str, name: str = "slice", workers: int = 2) -> list[tuple[str, str, str]]:
    """`command` run once on each of a TPU's workers, as `commands` reads them."""
    return [(name, str(worker), command) for worker in range(workers)]


def test_list_asks_every_configured_zone_and_reports_none_found(fake, capsys):
    """`list` asks each configured zone's listing once and reports every TPU found in any, its name,
    type, state and zone; with none it reports no row and succeeds."""
    fake.offer("slice", "us-central2-b")
    fake.offer("other", "us-east1-d", dict(NODE, state="STOPPED"))
    assert run("list") == 0
    assert zones_seen(only(fake.gcloud_calls(), "list"), "list") == list(CONFIG.zones)
    header, *rows = capsys.readouterr().out.splitlines()
    assert header.split()[:6] == ["NAME", "TYPE", "STATE", "HEALTH", "WORKERS", "ZONE"]
    assert sorted((cells[0], cells[1], cells[2], cells[5]) for cells in map(str.split, rows)) == [
        ("other", "v5litepod-16", "STOPPED", "us-east1-d"),
        ("slice", "v5litepod-16", "READY", "us-central2-b")]
    fake.state["nodes"] = {}
    fake.flush()
    assert run("list", "--zone", "europe-west4-a") == 0
    assert zones_seen(only(fake.gcloud_calls(), "list"), "list")[-1] == "europe-west4-a"
    shown = capsys.readouterr().out.splitlines()
    assert not any(line.startswith(("NAME", "slice", "other")) for line in shown)


def test_attach_disk_attaches_the_disk_then_mounts_it_on_every_worker(fake):
    fake.offer("slice", "us-central2-b")
    assert run("attach-disk", "slice", "data", "--read-only") == 0
    assert only(fake.gcloud_calls(), "attach-disk") == [gcloud(
        "compute", "tpus", "tpu-vm", "attach-disk", "slice", "--zone=us-central2-b",
        "--disk=projects/my-project/zones/us-central2-b/disks/data", "--mode=read-only")]
    mount = f"sudo -n bash -c {shlex.quote(tpu_setup.disk_startup_script(read_only=True))}"
    assert commands(fake, "ssh") == each(mount)


def test_copy_sends_a_directory_recursively_to_every_worker(fake, tmp_path):
    fake.offer("slice", "us-central2-b")
    (tmp_path / "payload").mkdir()
    assert run("copy", "slice", str(tmp_path / "payload"), "~/payload") == 0
    assert only(fake.gcloud_calls(), "scp") == [gcloud(
        "compute", "tpus", "tpu-vm", "scp", str(tmp_path / "payload"), "you@slice:~/payload",
        "--zone=us-central2-b", "--worker=all", "--recurse")]


def test_sync_rsyncs_the_working_tree_to_each_worker_with_its_excludes(fake, tree):
    fake.offer("slice", "us-central2-b")
    assert run("sync", "slice", "--exclude", "runs", "--delete") == 0
    synced = fake.rsync_calls()
    # The workers sync at once, in whatever order they finish.
    assert sorted(call[-1] for call in synced) == [f"you@34.0.0.{index}:~/tree/" for index in (1, 2)]
    assert all("--delete" in call and "--exclude=runs" in call and call[-2] == f"{tree}/"
               for call in synced)


def test_setup_from_source_syncs_installs_and_checks_every_workers_devices(fake, tree):
    """`setup --from-source` syncs the tree, copies its rendered script to every worker, runs it on
    each, and checks each sees the slice's 16 devices; a worker that sees fewer fails it."""
    fake.offer("slice", "us-central2-b")
    assert run("setup", "slice", "--from-source") == 0
    assert len(fake.rsync_calls()) == 2
    assert [call[4:6] for call in only(fake.gcloud_calls(), "scp")] == [
        [str(config_dir() / "setup-slice.sh"), "you@slice:~/dew-setup.sh"]]
    assert commands(fake, "ssh") == sorted(each("bash ~/dew-setup.sh") + each(tpu_setup.DEVICE_COUNT))
    fake.offer("slice", "us-central2-b", devices="8 8")
    assert run("setup", "slice") == 1


def test_train_syncs_starts_the_recipe_on_every_worker_and_follows_worker_zero(fake, tree):
    fake.offer("slice", "us-central2-b")
    assert run("train", "slice", "--job", "j1", "--", "recipes/lm/train.py", "--steps", "3") == 0
    recipe = "python recipes/lm/train.py --steps 3 --trainer.multi-host True"
    assert commands(fake, "ssh") == sorted(
        [("slice", str(index), tpu_setup.detached(recipe, "j1", index, home_dir="tree")) for index in (0, 1)]
        + [("slice", "0", tpu_setup.tail("j1", 0, follow=True))])
    with pytest.raises(SystemExit):
        run("train", "slice")


def test_run_detached_starts_a_named_job_and_logs_reads_its_tail(fake):
    fake.offer("slice", "us-central2-b")
    assert run("run", "slice", "--detach", "--job", "j2", "--", "echo", "hi") == 0
    started = [("slice", str(index), tpu_setup.detached("echo hi", "j2", index)) for index in (0, 1)]
    assert commands(fake, "ssh") == started
    assert run("logs", "slice", "j2", "--worker", "all", "--lines", "5") == 0
    tails = [("slice", str(index), tpu_setup.tail("j2", index, follow=False, lines=5)) for index in (0, 1)]
    assert commands(fake, "ssh") == sorted(started + tails)
    with pytest.raises(SystemExit):
        run("run", "slice")


def test_status_and_reset_ask_every_worker(fake):
    fake.offer("slice", "us-central2-b")
    assert run("status", "slice") == 0
    assert run("reset", "slice") == 0
    assert commands(fake, "ssh") == sorted(each(tpu_setup.STATUS) + each(tpu_setup.RESET))


def test_create_queued_spot_goes_through_the_queued_resources_api(fake):
    fake.offer("slice", "us-central2-b")
    assert run("create", "slice", "--queued", "--spot") == 0
    assert fake.gcloud_calls()[0] == gcloud(
        "compute", "tpus", "queued-resources", "create", "slice", "--zone=us-central2-b",
        "--accelerator-type=v5litepod-16", "--runtime-version=v2-alpha-tpuv5-lite", "--node-id=slice",
        "--spot")


def test_spawn_creates_sets_up_and_starts_a_command_on_each_tpu(fake):
    for name in ("pool-0", "pool-1"):
        fake.offer(name, "us-central2-b")
    assert run("spawn", "pool", "2", "--spot", "--", "echo", "hi") == 0
    created = sorted(call[4] for call in only(fake.gcloud_calls(), "create"))
    assert created == ["pool-0", "pool-1"]
    started = [(name, worker, command) for name, worker, command in commands(fake, "ssh")
               if "nohup" in command]
    assert [(name, worker) for name, worker, _ in started] == [
        ("pool-0", "0"), ("pool-0", "1"), ("pool-1", "0"), ("pool-1", "1")]


def test_help_lists_the_config_defaults_a_command_reads(fake, capsys):
    with pytest.raises(SystemExit) as done:
        run("spawn", "--help")
    assert done.value.code == 0
    shown = dict(line.split(None, 1) for line in capsys.readouterr().out.splitlines()
                 if line.startswith("  ") and len(line.split(None, 1)) == 2)
    expected = {field: ", ".join(value) if isinstance(value, tuple) else (value or "(unset)")
                for field, value in ((field, getattr(CONFIG, field)) for field in tpu_cli.Spawn.reads)}
    assert {field: shown[field].strip() for field in expected} == expected


def test_a_tpu_no_zone_lists_and_one_not_ready_are_refused(fake):
    """A name no configured zone lists is refused after asking each, and forgotten from the zone
    cache; a TPU that is TERMINATED does not come back to READY, so `start` stops at once."""
    (config_dir() / "zones.json").write_text(json.dumps({"ghost": "us-east1-d"}))
    with pytest.raises(SystemExit):
        run("describe", "ghost")
    assert sorted(zones_seen(only(fake.gcloud_calls(), "list"), "list")) == sorted(CONFIG.zones)
    assert json.loads((config_dir() / "zones.json").read_text()) == {}
    fake.offer("slice", "us-central2-b", dict(NODE, state="TERMINATED"))
    listed = len(only(fake.gcloud_calls(), "list"))
    with pytest.raises(SystemExit):
        run("start", "slice")
    # One listing finds the zone, one after the start finds it TERMINATED: no poll waits on it.
    assert len(only(fake.gcloud_calls(), "list")) - listed == 2
