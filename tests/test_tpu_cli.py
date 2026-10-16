"""dew tpu and dew launch --tpu against a fake gcloud: argv per command, fan-out, pools, dry runs, config."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from dew.cli import config as tpu_config, tpu as tpu_cli

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
    monkeypatch.setenv("DEW_TPU_POLL_SECONDS", "0")
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
    # Each command confirms where the TPU is with a describe, and start waits
    # for READY afterwards.
    assert [call[3] for call in fake.gcloud_calls()] == [
        "describe", "stop", "describe", "start", "describe"]


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
        command = ("exec env MEGASCALE_COORDINATOR_ADDRESS=10.0.0.1 MEGASCALE_PORT=8081 "
                   f"MEGASCALE_NUM_SLICES=2 MEGASCALE_SLICE_ID={slice_id} python train.py")
        expected += [pool_ssh(0, name, command), pool_ssh(1, name, command)]
    assert sorted(calls) == sorted(expected)


# ----------------------------------------------------------------------- syncing


# ------------------------------------------------------------------------- setup


# ----------------------------------------------------------------------- dry run


# ------------------------------------------------------------------- zone lookup


def test_zone_search_follows_the_configured_order_and_caches(fake):
    fake.offer("slice", "europe-west4-a")
    assert run("describe", "slice") == 0
    assert zones_seen(fake.gcloud_calls(), "describe") == [
        "us-central2-b", "europe-west4-a", "europe-west4-a"]
    assert json.loads((config_dir() / "zones.json").read_text()) == {"slice": "europe-west4-a"}
    fake.calls_path.unlink()
    assert run("describe", "slice") == 0
    assert zones_seen(fake.gcloud_calls(), "describe") == [
        "europe-west4-a", "europe-west4-a"]


def test_a_cached_zone_that_stopped_answering_is_replaced(fake):
    """A TPU recreated in another zone is found by the zone search and its
    cache entry replaced; every command would otherwise fail with 'not in Z.
    Pass --zone' until the cache file was edited by hand."""
    tpu_config.cache_zone("slice", "us-east1-d")
    fake.offer("slice", "europe-west4-a")
    assert run("describe", "slice") == 0
    assert zones_seen(fake.gcloud_calls(), "describe") == [
        "us-east1-d", "us-central2-b", "europe-west4-a", "europe-west4-a"]
    assert json.loads((config_dir() / "zones.json").read_text()) == {
        "slice": "europe-west4-a"}


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
