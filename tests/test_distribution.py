"""Hybrid sharded data parallelism and context parallelism across real
processes, started by `dew launch`.

Each pool is two processes of four simulated CPU devices on this machine,
launched the way a two-node run is: `dew launch` puts the coordinator, the
process count and the rank in the environment, and the worker joins through
`prepare_process`. The reference is the same worker as one process of eight
devices on plain fsdp. A mesh whose fsdp axis stays inside one process and
whose data axis crosses the two is hybrid sharding; the losses it trains to
have to be the reference's.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("distribution_worker.py")
# Three adam steps on another topology; observed 2.4e-7.
TOLERANCE = 1e-5


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def launch(*arguments: str, devices: int, timeout: float = 600) -> subprocess.CompletedProcess:
    """`dew launch` on this machine, with `devices` CPU devices a process."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "dew.cli.main", "launch", "--port", str(free_port()),
         "--env", f"XLA_FLAGS=--xla_force_host_platform_device_count={devices}", *arguments],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def train(tmp_path: Path, mesh: dict, processes: int) -> dict:
    out = tmp_path / f"{len(list(tmp_path.iterdir()))}.json"
    done = launch("--processes-per-host", str(processes), "--",
                  sys.executable, str(WORKER), "--out", str(out), "--mesh", json.dumps(mesh),
                  devices=8 // processes)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> dict:
    return train(tmp_path_factory.mktemp("reference"), {"fsdp": 8}, processes=1)


def test_hybrid_sharding_keeps_fsdp_inside_a_host_and_trains_like_fsdp(tmp_path, reference):
    """fsdp=4 with replicas=2 over two processes: every fsdp group sits on
    one process, the data axis crosses them, and the losses are plain fsdp's
    over all eight devices."""
    pool = train(tmp_path, {"fsdp": 4, "replicas": 2}, processes=2)
    assert pool["processes"] == 2 and pool["mesh"]["data"] == 2
    assert pool["fsdp_groups"] == [[0], [1]]
    assert max(abs(a - b) for a, b in zip(pool["losses"], reference["losses"], strict=True)) < TOLERANCE


@pytest.mark.parametrize("exchange", ["all_to_all", "all_gather"])
def test_context_parallelism_trains_like_whole_sequences_across_hosts(tmp_path, reference, exchange):
    """A packed batch with its sequence split two ways inside each host and
    the replicas across the hosts trains to the reference's losses under
    either exchange."""
    pool = train(tmp_path, {"fsdp": 2, "sequence": 2, "replicas": 2,
                            "sequence_exchange": exchange}, processes=2)
    assert pool["fsdp_groups"] == [[0], [0], [1], [1]]
    assert max(abs(a - b) for a, b in zip(pool["losses"], reference["losses"], strict=True)) < TOLERANCE


def test_more_replicas_than_hosts_are_refused(tmp_path):
    done = launch("--", sys.executable, str(WORKER), "--out", str(tmp_path / "x.json"),
                  "--mesh", json.dumps({"fsdp": 4, "replicas": 2}), devices=8)
    assert done.returncode != 0
    assert "replicas 2 must divide both the 1 granules" in done.stdout


def test_a_failed_process_stops_the_pool_with_its_exit_code():
    """Rank 1 exits 3 while rank 0 would wait ten minutes, as a process
    stuck in a collective with a dead peer does. The launch returns 3 within
    seconds, having stopped rank 0."""
    program = ("import os, sys, time\n"
               "if os.environ['DEW_PROCESS_ID'] == '1': sys.exit(3)\n"
               "time.sleep(600)\n")
    started = time.monotonic()
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program,
                  devices=1, timeout=120)
    assert done.returncode == 3, done.stdout + done.stderr
    assert time.monotonic() - started < 60
