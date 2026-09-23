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
