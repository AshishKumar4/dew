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

import dataclasses
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from dew.nn.sharding import MESH_AXES
from dew.training import MeshSpec
from dew.training.distributed import hybrid_devices

# Needs the eight simulated CPU devices conftest configures, except the tests
# that name fewer, which a four-GPU run takes as one GPU a process.
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
    """`dew launch` on this machine with `devices` devices a process: simulated
    CPU devices, or on a GPU run that many GPUs of the machine's own."""
    import jax

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    if jax.default_backend() == "gpu":
        placement = ["--devices-per-process", str(devices)]
    else:
        env["JAX_PLATFORMS"] = "cpu"
        placement = ["--env", f"XLA_FLAGS=--xla_force_host_platform_device_count={devices}"]
    process = subprocess.Popen(
        [sys.executable, "-m", "dew.cli.main", "launch", "--port", str(free_port()), *placement,
         *arguments],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # SIGTERM, not the SIGKILL a timed-out run sends: the launcher stops
        # its ranks, which run in sessions of their own, before it exits.
        process.terminate()
        stdout, stderr = process.communicate(timeout=60)
        pytest.fail(f"the pool was still running after {timeout}s\n{stdout}{stderr}")
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def train(tmp_path: Path, mesh: dict, processes: int, devices: int = 8) -> dict:
    """The worker's record of a pool of `processes` over `devices` CPU devices."""
    out = tmp_path / f"{len(list(tmp_path.iterdir()))}.json"
    done = launch("--processes-per-host", str(processes), "--",
                  sys.executable, str(WORKER), "--out", str(out), "--mesh", json.dumps(mesh),
                  devices=devices // processes)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> dict:
    return train(tmp_path_factory.mktemp("reference"), {"fsdp": 8}, processes=1)


def test_hybrid_sharding_groups_hosts_into_replicas_and_trains_like_fsdp(tmp_path, reference):
    """Four processes of two devices, fsdp=2 over replicas=2: each replica is
    two processes and fsdp crosses the pair, where `jax.make_mesh` would put
    every fsdp group on one process. The losses are plain fsdp's over all
    eight devices."""
    pool = train(tmp_path, {"fsdp": 2, "replicas": 2}, processes=4)
    assert pool["processes"] == 4 and pool["mesh"]["data"] == 4
    assert pool["fsdp_groups"] == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert max(abs(a - b) for a, b in zip(pool["losses"], reference["losses"], strict=True)) < TOLERANCE


@dataclasses.dataclass(frozen=True)
class StandIn:
    """What `hybrid_devices` reads of a device, for a topology this machine
    does not have."""

    id: int
    process_index: int
    slice_index: int
    platform: str = "cpu"
    device_kind: str = "cpu"


def standins(slices: int, processes: int, per_process: int) -> list[StandIn]:
    """Devices numbered process-major, `processes` hosts to each slice."""
    return [StandIn(slice_ * processes * per_process + process * per_process + local,
                    slice_ * processes + process, slice_)
            for slice_ in range(slices) for process in range(processes)
            for local in range(per_process)]


def groups(array: np.ndarray, attribute: str, axis: int) -> list[list[int]]:
    """Along `axis` of a (data, expert, fsdp, tensor, sequence, stage)
    device array, the distinct `attribute` values of every group."""
    moved = np.moveaxis(array, axis, -1)
    return [sorted({getattr(device, attribute) for device in group})
            for group in moved.reshape(-1, moved.shape[-1])]


def test_replicas_over_processes_split_fsdp_across_the_hosts_of_one_replica():
    """Four hosts of two devices on one slice, fsdp=2 over replicas=2: fsdp
    pairs hosts 0-1 and 2-3, and the data axis's outer half crosses the
    replicas."""
    array = hybrid_devices(MeshSpec(fsdp=2, replicas=2), (4, 1, 2, 1, 1, 1), standins(1, 4, 2))
    assert groups(array, "process_index", 2) == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert [sorted({d.process_index for d in array[i].flat}) for i in range(4)] == [
        [0, 1], [0, 1], [2, 3], [2, 3]]


def test_replicas_over_slices_keep_every_axis_but_data_inside_a_slice():
    """Two slices of two hosts: fsdp=4 over replicas=2 keeps each fsdp group
    in one slice across its two hosts, and only the data axis crosses the
    slices, whose devices report a slice each."""
    array = hybrid_devices(MeshSpec(fsdp=4, replicas=2), (2, 1, 4, 1, 1, 1), standins(2, 2, 2))
    assert groups(array, "slice_index", 2) == [[0], [1]]
    assert groups(array, "process_index", 2) == [[0, 1], [2, 3]]
    assert groups(array, "slice_index", 0) == [[0, 1]] * 4


def test_gpu_hosts_are_the_granules_however_many_processes_each_runs():
    """Two GPU hosts of four one-device processes, which XLA gives a slice
    index per host: the granules are the two hosts, so fsdp=4 over two
    replicas holds each host's four processes, and four replicas are more
    than the hosts."""
    hosts = [StandIn(index, index, index // 4) for index in range(8)]
    array = hybrid_devices(MeshSpec(fsdp=4, replicas=2), (2, 1, 4, 1, 1, 1), hosts)
    assert groups(array, "slice_index", 2) == [[0], [1]]
    with pytest.raises(ValueError, match="the 2 granules"):
        hybrid_devices(MeshSpec(fsdp=2, replicas=4), (4, 1, 2, 1, 1, 1), hosts)


def test_replicas_the_granules_cannot_hold_are_refused():
    with pytest.raises(ValueError, match="replicas 3 must divide both the 4 granules"):
        hybrid_devices(MeshSpec(fsdp=2, replicas=3), (4, 1, 2, 1, 1, 1), standins(1, 4, 2))
    with pytest.raises(ValueError, match="fsdp 1 does not divide over them"):
        hybrid_devices(MeshSpec(replicas=2), (8, 1, 1, 1, 1, 1), standins(1, 4, 2))


def test_context_parallelism_trains_like_whole_sequences_across_hosts(tmp_path, reference):
    """A packed batch with its sequence split two ways inside each host and
    the replicas across the hosts trains to the reference's losses."""
    pool = train(tmp_path, {"fsdp": 2, "sequence": 2, "replicas": 2}, processes=2)
    assert pool["fsdp_groups"] == [[0], [0], [1], [1]]
    assert max(abs(a - b) for a, b in zip(pool["losses"], reference["losses"], strict=True)) < TOLERANCE


@pytest.mark.parametrize("mesh", [{"fsdp": 2, "stage": 2}, {"fsdp": 2, "sequence": 2}],
                         ids=["stage", "sequence"])
def test_processes_that_hold_the_same_rows_read_the_same_share(tmp_path, mesh):
    """Four processes of one device, with the pipeline's stage axis or a split
    sequence between them: each pair holds one row shard, so the pair reads
    one share of the batch and the pool trains the losses one process of
    four devices trains. Every process reading its own quarter of the rows,
    as the loaders once did, trained this stage mesh to 4.42, 4.03, 3.91
    against 4.60, 3.47, 2.87, and fed the sequence mesh a batch of half its
    rows. A leaf the sequence axis splits is placed from the share whole."""
    alone = train(tmp_path, mesh, processes=1, devices=4)
    pool = train(tmp_path, mesh, processes=4, devices=4)

    assert pool["partition"] == {"count": 2, "readers": 2}
    assert alone["partition"] == {"count": 1, "readers": 1}
    assert pool["placed_whole"] and alone["placed_whole"]
    assert max(abs(a - b) for a, b in zip(pool["losses"], alone["losses"], strict=True)) < TOLERANCE


def test_more_replicas_than_hosts_are_refused(tmp_path):
    done = launch("--", sys.executable, str(WORKER), "--out", str(tmp_path / "x.json"),
                  "--mesh", json.dumps({"fsdp": 4, "replicas": 2}), devices=8)
    assert done.returncode != 0
    assert "replicas 2 must divide both the 1 granules" in done.stdout


@dataclasses.dataclass(frozen=True)
class StandIn:
    """What the mesh builders read of a GPU device on one of two hosts, each
    host its own slice, as XLA numbers GPU slices per boot."""
    id: int
    process_index: int
    slice_index: int
    platform: str = "gpu"
    device_kind: str = "NVIDIA GeForce RTX 3090"


def crossing(spec: MeshSpec) -> list[str]:
    """The axes of `spec`'s device grid over two hosts of four GPUs whose
    groups span both hosts."""
    sharded = spec.fsdp * spec.expert * spec.tensor * spec.sequence * spec.stage
    shape = (8 // sharded, spec.expert, spec.fsdp, spec.tensor, spec.sequence, spec.stage)
    grid = hybrid_devices(spec, shape, [StandIn(index, index // 4, index // 4) for index in range(8)])
    hosts = np.vectorize(lambda device: device.slice_index)(grid)
    return [name for axis, name in enumerate(MESH_AXES)
            if len(set(np.moveaxis(hosts, axis, -1).reshape(-1, hosts.shape[axis])[0])) > 1]


def test_two_gpu_hosts_split_the_data_axis_unless_asked_otherwise():
    """Every GPU host is its own slice. Without replicas the hosts take the
    data axis where it divides, and fsdp only where it does not; before,
    plain data parallelism and a tensor mesh were refused on two hosts."""
    assert crossing(MeshSpec()) == ["data"]
    assert crossing(MeshSpec(tensor=4)) == ["data"]
    assert crossing(MeshSpec(fsdp=2, tensor=2)) == ["data"]
    assert crossing(MeshSpec(fsdp=8)) == ["fsdp"]
    with pytest.raises(ValueError, match="only the fsdp axis may cross"):
        crossing(MeshSpec(tensor=8))


@pytest.mark.mesh(devices=4)
def test_a_named_device_list_is_laid_out_in_its_own_order():
    """A caller naming the devices chooses which share an axis, as
    benchmark_step's device_order puts a size-2 axis across or along an
    NVLink pair. jax.make_mesh sorts GPU devices by id, which built the same
    mesh for every order of four GPUs."""
    import jax

    from dew.training import build_mesh

    first, second, third, fourth = jax.devices()[:4]
    order = [first, third, second, fourth]
    mesh = build_mesh(MeshSpec(fsdp=2, tensor=2), order)
    assert list(mesh.devices.flat) == order


def test_replicas_in_a_process_outside_any_pool_are_refused_by_granule():
    """A process that never joined a pool has CPU devices without a
    `slice_index`; it is one granule, so replicas are refused by the same
    rule, not by the missing attribute."""
    from dew.training import build_mesh

    with pytest.raises(ValueError, match="replicas 2 must divide both the 1 granules"):
        build_mesh(MeshSpec(fsdp=4, replicas=2))


@pytest.mark.mesh(devices=2)
def test_a_rank_that_raises_between_collectives_stops_the_pool():
    """Rank 1 raises before its fourth step while rank 0 is inside that
    step's reduction, waiting for a partner that is gone, which no backend
    times out on its own for minutes. The launch stops rank 0 and returns
    rank 1's failure within a bound."""
    program = ("from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "import jax, numpy as np\n"
               "from jax.sharding import NamedSharding, PartitionSpec\n"
               "from dew.training import MeshSpec, build_mesh\n"
               "mesh = build_mesh(MeshSpec(fsdp=jax.device_count()))\n"
               "rows = NamedSharding(mesh, PartitionSpec('fsdp'))\n"
               "whole = np.ones((jax.device_count() * 256, 256), np.float32)\n"
               "x = jax.make_array_from_callback(whole.shape, rows, lambda index: whole[index])\n"
               "step = jax.jit(lambda x: x / x.sum(), out_shardings=rows)\n"
               "for index in range(100000):\n"
               "    if index == 3 and jax.process_index() == 1:\n"
               "        raise RuntimeError('injected failure')\n"
               "    x = step(x)\n"
               "    x.block_until_ready()\n")
    started = time.monotonic()
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program,
                  devices=1, timeout=600)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "RuntimeError: injected failure" in done.stdout
    assert time.monotonic() - started < 120, done.stdout + done.stderr


@pytest.mark.mesh(devices=2)
def test_a_rank_whose_data_fails_mid_fit_stops_the_pool(tmp_path):
    """Rank 1's loader raises on its fourth batch while rank 0 has gone on to
    that step, whose collectives wait for rank 1 for ever on a GPU. Rank 1's
    fit reaches its cleanup agreement alone; the pool still has to end, with
    rank 1's error, within a bound."""
    started = time.monotonic()
    done = launch("--processes-per-host", "2", "--", sys.executable, str(WORKER),
                  "--out", str(tmp_path / "failed.json"), "--mesh", json.dumps({"fsdp": 2}),
                  "--steps", "50", "--fail-at", "3", devices=1, timeout=600)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "injected failure reading batch 3" in done.stdout, done.stdout
    assert time.monotonic() - started < 180, done.stdout + done.stderr


@pytest.mark.mesh(devices=2)
def test_a_pool_refuses_a_checkpoint_directory_its_processes_do_not_share(tmp_path):
    """Each process names its own disk's copy of the run directory, as hosts
    with nothing shared do. Orbax writes one checkpoint across the pool and
    reads any process's shards back, so process 0 would commit steps missing
    the others' shards while they saw none, and a resume would train
    different states. Every process refuses, naming the one that cannot see
    the directory."""
    program = ("import os, sys\n"
               "from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "from dew.checkpoints import Checkpoints\n"
               "host = os.path.join(sys.argv[1], 'host' + os.environ['DEW_PROCESS_ID'])\n"
               "print('latest', Checkpoints(os.path.join(host, 'run')).latest)\n")
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program,
                  str(tmp_path), devices=1, timeout=300)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "is not shared: process(es) [1] of 2 do not see" in done.stdout, done.stdout


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
