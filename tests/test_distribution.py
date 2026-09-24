"""Hybrid sharded data parallelism, context parallelism and rollout
scheduling across real processes, started by `dew launch`.

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
# that name fewer: those of stand-in devices need none, and a four-GPU run
# takes the pool tests of two as one GPU a process.
pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("distribution_worker.py")
SCHEDULER = Path(__file__).with_name("scheduler_worker.py")
# Three adam steps on another topology; observed 2.4e-7.
TOLERANCE = 1e-5


def start(*arguments: str, devices: int) -> subprocess.Popen:
    """`dew launch` on this machine with `devices` devices a process: simulated
    CPU devices, or on a GPU run that many GPUs of the machine's own."""
    import jax

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    if jax.default_backend() == "gpu":
        placement = ["--devices-per-process", str(devices)]
    else:
        env["JAX_PLATFORMS"] = "cpu"
        placement = ["--env", f"XLA_FLAGS=--xla_force_host_platform_device_count={devices}"]
    return subprocess.Popen(
        [sys.executable, "-m", "dew.cli.main", "launch", *placement, *arguments],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finished(process: subprocess.Popen, timeout: float = 600) -> subprocess.CompletedProcess:
    """A started launch once it has ended, stopping it after `timeout` seconds."""
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # SIGTERM, not the SIGKILL a timed-out run sends: the launcher stops
        # its ranks, which run in sessions of their own, before it exits.
        process.terminate()
        stdout, stderr = process.communicate(timeout=60)
        pytest.fail(f"the pool was still running after {timeout}s\n{stdout}{stderr}")
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def launch(*arguments: str, devices: int, timeout: float = 600) -> subprocess.CompletedProcess:
    return finished(start(*arguments, devices=devices), timeout)


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
    """What the mesh builders read of a GPU device, for a topology this
    machine does not have; XLA numbers GPU slices per host boot."""

    id: int
    process_index: int
    slice_index: int
    platform: str = "gpu"
    device_kind: str = "NVIDIA GeForce RTX 3090"


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


@pytest.mark.mesh(devices=0)
def test_replicas_over_processes_split_fsdp_across_the_hosts_of_one_replica():
    """Four hosts of two devices on one slice, fsdp=2 over replicas=2: fsdp
    pairs hosts 0-1 and 2-3, and the data axis's outer half crosses the
    replicas."""
    array = hybrid_devices(MeshSpec(fsdp=2, replicas=2), (4, 1, 2, 1, 1, 1), standins(1, 4, 2))
    assert groups(array, "process_index", 2) == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert [sorted({d.process_index for d in array[i].flat}) for i in range(4)] == [
        [0, 1], [0, 1], [2, 3], [2, 3]]


@pytest.mark.mesh(devices=0)
def test_replicas_over_slices_keep_every_axis_but_data_inside_a_slice():
    """Two slices of two hosts: fsdp=4 over replicas=2 keeps each fsdp group
    in one slice across its two hosts, and only the data axis crosses the
    slices, whose devices report a slice each."""
    array = hybrid_devices(MeshSpec(fsdp=4, replicas=2), (2, 1, 4, 1, 1, 1), standins(2, 2, 2))
    assert groups(array, "slice_index", 2) == [[0], [1]]
    assert groups(array, "process_index", 2) == [[0, 1], [2, 3]]
    assert groups(array, "slice_index", 0) == [[0, 1]] * 4


@pytest.mark.mesh(devices=0)
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


@pytest.mark.mesh(devices=0)
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


def crossing(spec: MeshSpec) -> list[str]:
    """The axes of `spec`'s device grid over two hosts of four GPUs whose
    groups span both hosts."""
    sharded = spec.fsdp * spec.expert * spec.tensor * spec.sequence * spec.stage
    shape = (8 // sharded, spec.expert, spec.fsdp, spec.tensor, spec.sequence, spec.stage)
    grid = hybrid_devices(spec, shape, [StandIn(index, index // 4, index // 4) for index in range(8)])
    hosts = np.vectorize(lambda device: device.slice_index)(grid)
    return [name for axis, name in enumerate(MESH_AXES)
            if len(set(np.moveaxis(hosts, axis, -1).reshape(-1, hosts.shape[axis])[0])) > 1]


@pytest.mark.mesh(devices=0)
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


@pytest.mark.mesh(devices=2)
def test_a_named_device_list_is_laid_out_in_its_own_order():
    """A caller naming the devices chooses which share an axis, as
    benchmark_step's device_order puts a size-2 axis across or along an
    NVLink pair. jax.make_mesh sorts GPU devices by id, which built the same
    mesh for every order."""
    import jax

    from dew.training import build_mesh

    order = jax.devices()[:2][::-1]
    mesh = build_mesh(MeshSpec(fsdp=2), order)
    assert list(mesh.devices.flat) == order


def test_replicas_in_a_process_outside_any_pool_are_refused_by_granule():
    """A process that never joined a pool has CPU devices without a
    `slice_index`; it is one granule, so replicas are refused by the same
    rule, not by the missing attribute."""
    from dew.training import build_mesh

    with pytest.raises(ValueError, match="replicas 2 must divide both the 1 granules"):
        build_mesh(MeshSpec(fsdp=4, replicas=2))


def stepping_pool(rank_one: str, *, execution_timeout: str | None = None) -> str:
    """A program that steps a reduction over every device of the pool, where
    rank 1 runs the statement `rank_one` before its fourth step.
    `execution_timeout` shortens the pool's bound on one execution."""
    shortened = ("import dew.training.runtime as runtime\n"
                 f"runtime.EXECUTION_TIMEOUT = {execution_timeout!r}\n") if execution_timeout else ""
    return (shortened
            + "import threading\n"
            "from dew.training.runtime import prepare_process\n"
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
            f"        {rank_one}\n"
            "    x = step(x)\n"
            "    x.block_until_ready()\n")


ONE_SLURM_TASK = {"SLURM_JOB_ID": "4242", "SLURM_NTASKS": "1", "SLURM_PROCID": "0",
                  "SLURM_LOCALID": "0", "SLURM_STEP_NODELIST": "dew-no-such-host",
                  "SLURM_STEP_NUM_NODES": "1"}
"""What srun sets for a step of one task, on a node no container resolves."""


@pytest.mark.mesh(devices=0)
def test_one_slurm_task_joins_no_pool():
    """srun with one task sets SLURM_JOB_ID, and JAX's Slurm detection then
    starts a one-process pool whose coordinator is the node's name. A
    container on the node need not resolve that name (the box's did not):
    the process waited 300 s to register and aborted. One task is no pool,
    so the process starts on its own."""
    env = {**os.environ, **ONE_SLURM_TASK, "PYTHONPATH": str(REPO_ROOT / "src"), "JAX_PLATFORMS": "cpu"}
    program = ("from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "import jax\n"
               "print('processes', jax.process_count())\n")
    done = subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "processes 1" in done.stdout, done.stdout


@pytest.mark.mesh(devices=0)
def test_an_mpirun_inside_one_slurm_task_forms_its_pool():
    """mpirun started inside a one-task allocation sets Open MPI's rank
    variables beside Slurm's. JAX's detection reads Open MPI's first and
    forms the pool mpirun started, so the one-task rule has to leave the
    process to it, here a pool of one rank."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, **ONE_SLURM_TASK, "PYTHONPATH": str(REPO_ROOT / "src"), "JAX_PLATFORMS": "cpu",
           "OMPI_MCA_orte_hnp_uri": "1531576320.0;tcp://127.0.0.1:34911",
           "OMPI_COMM_WORLD_SIZE": "1", "OMPI_COMM_WORLD_RANK": "0",
           "OMPI_COMM_WORLD_LOCAL_RANK": "0", "JAX_COORDINATOR_PORT": str(port)}
    program = ("from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "import jax\n"
               "print('pool', jax.distributed.is_initialized())\n")
    done = subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "pool True" in done.stdout, done.stdout


@pytest.mark.mesh(devices=2)
def test_a_launched_process_keeps_every_device_inside_a_slurm_step():
    """`dew launch` without --devices-per-process gives its one process every
    local device. Inside a Slurm step, JAX's detection would still place it:
    it pins a process it finds in Slurm's variables to the GPU at the step's
    SLURM_LOCALID, one device of the launcher's several. A process dew launch
    started takes its placement from the launcher."""
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("a cluster pins device ids on GPU; the CPU backend takes none")
    env = {**os.environ, **ONE_SLURM_TASK, "PYTHONPATH": str(REPO_ROOT / "src")}
    program = ("from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "import jax\n"
               "print('local devices', jax.local_device_count())\n")
    done = finished(subprocess.Popen(
        [sys.executable, "-m", "dew.cli.main", "launch", "--", sys.executable, "-c", program],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"local devices {jax.local_device_count()}" in done.stdout, done.stdout


@pytest.mark.mesh(devices=2)
def test_a_rank_whose_backend_fails_to_open_after_joining_ends_the_pool():
    """Rank 1 joins the pool and then cannot open its backend, as a GPU with
    no memory left fails to, while rank 0 waits two minutes for its devices'
    topology. Rank 1 has to leave at once, not wait in jax.distributed's
    shutdown barrier for the peer that waits for it, so the launch ends with
    its error within a bound."""
    program = ("import os\n"
               "if os.environ['DEW_PROCESS_ID'] == '1':\n"
               "    os.environ['JAX_PLATFORMS'] = 'nosuchplatform'\n"
               "from dew.training.runtime import prepare_process\n"
               "prepare_process()\n")
    started = time.monotonic()
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program,
                  devices=1, timeout=600)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "nosuchplatform" in done.stdout, done.stdout
    assert time.monotonic() - started < 60, done.stdout + done.stderr


@pytest.mark.mesh(devices=1)
def test_a_one_process_pool_that_fails_still_runs_its_exit_handlers(tmp_path):
    """A pool of one process has no peer waiting in a collective for it, so a
    failure takes Python's own exit, which runs the atexit handlers: those
    are where HarborTrials stops its trials. The failure watch leaves through
    os._exit, which skips them, so only a pool of several processes takes it."""
    marker = tmp_path / "exited"
    program = ("import atexit, pathlib, sys\n"
               "atexit.register(pathlib.Path(sys.argv[1]).write_text, 'ran')\n"
               "from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "raise RuntimeError('the program fails')\n")
    done = launch("--processes-per-host", "1", "--", sys.executable, "-c", program, str(marker),
                  devices=1, timeout=300)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "RuntimeError: the program fails" in done.stdout, done.stdout
    assert marker.exists(), done.stdout + done.stderr


@pytest.mark.mesh(devices=2)
def test_a_rank_that_raises_between_collectives_stops_the_pool():
    """Rank 1 raises before its fourth step while rank 0 is inside that
    step's reduction, waiting for a partner that is gone, which no backend
    times out on its own for minutes. The launch stops rank 0 and returns
    rank 1's failure within a bound."""
    started = time.monotonic()
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c",
                  stepping_pool("raise RuntimeError('injected failure')"), devices=1, timeout=600)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "RuntimeError: injected failure" in done.stdout
    assert time.monotonic() - started < 120, done.stdout + done.stderr


@pytest.mark.mesh(devices=2)
def test_a_rank_that_stalls_between_collectives_ends_the_pool():
    """Rank 1 stops before its fourth step without failing, as a rank blocked
    on a read, or on a compile that waits for its peers, does. Rank 0 waits
    inside that step's reduction. Every process stays alive and none fails,
    so only a bound on one execution can end the pool: rank 0's runs past
    the pool's execution timeout, shortened here, and the launch ends."""
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("the execution bound is XLA's GPU watchdog; CPU pools have none")
    started = time.monotonic()
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c",
                  stepping_pool("threading.Event().wait()", execution_timeout="20s"),
                  devices=1, timeout=600)
    assert done.returncode != 0, done.stdout + done.stderr
    assert time.monotonic() - started < 150, done.stdout + done.stderr


@pytest.mark.mesh(devices=2)
def test_a_rank_busy_on_its_host_before_an_agreement_leaves_the_pool_running():
    """Rank 1 spends longer on its host than the pool's execution bound, as
    process 0 uploading a checkpoint to W&B does, before it agrees with rank
    0, which has arrived. Rank 0 has to wait for it on the host: inside the
    agreement's device collective, the bound would end it."""
    program = ("import dew.training.runtime as runtime\n"
               "runtime.EXECUTION_TIMEOUT = '20s'\n"
               "runtime.prepare_process()\n"
               "import time, jax\n"
               "from dew.artifacts import agreed\n"
               "if jax.process_index() == 1:\n"
               "    time.sleep(45)\n"
               "print('agreed', agreed('upload', lambda: jax.process_index()))\n")
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program,
                  devices=1, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.mesh(devices=2)
def test_a_pool_gathers_a_tree_in_groups_to_every_host_or_to_process_zero():
    """A pool brings a tree of 25 small global leaves home. Every group size
    gives the same leaves and dtypes as one leaf a group. With the default
    size the tree takes one transfer agreement, four in all, where one a leaf
    took 28 round trips: a weight push of Qwen3-0.6B's 310 leaves spent 4.6 s
    of its 6.8 s in them on 4x RTX 3090. With `held_by="first"`, process 0
    holds the tree and process 1, which took part in every agreement, holds
    none."""
    program = ("import json\n"
               "import dew.training.runtime as runtime\n"
               "runtime.prepare_process()\n"
               "import jax, jax.numpy as jnp, numpy as np\n"
               "from jax.sharding import NamedSharding, PartitionSpec\n"
               "from dew import artifacts\n"
               "from dew.training import MeshSpec, build_mesh\n"
               "mesh = build_mesh(MeshSpec(fsdp=jax.device_count()))\n"
               "rows, rng = jax.device_count(), np.random.default_rng(0)\n"
               "values = {'replicated': rng.standard_normal((3, 5)).astype(np.float32)}\n"
               "for i in range(8):\n"
               "    values[f'f32 {i}'] = rng.standard_normal((2 * rows, 3)).astype(np.float32)\n"
               "    values[f'bf16 {i}'] = rng.standard_normal((4 * rows,)).astype(jnp.bfloat16)\n"
               "    values[f'i32 {i}'] = rng.integers(0, 100, (rows, 2)).astype(np.int32)\n"
               "def placed(name, value):\n"
               "    spec = PartitionSpec() if name == 'replicated' else PartitionSpec('fsdp')\n"
               "    return jax.make_array_from_callback(value.shape, NamedSharding(mesh, spec),"
               " lambda index: value[index])\n"
               "tree = {name: placed(name, value) for name, value in values.items()}\n"
               "tree['host'] = np.arange(3)\n"
               "agreements = []\n"
               "agree = artifacts.agree_process_phase\n"
               "def counted(error, *, phase, available=True):\n"
               "    agreements.append(phase)\n"
               "    return agree(error, phase=phase, available=available)\n"
               "artifacts.agree_process_phase = counted\n"
               "def same(got):\n"
               "    return got is not None and np.array_equal(got['host'], np.arange(3)) and all(\n"
               "        type(got[name]) is np.ndarray and got[name].dtype == value.dtype\n"
               "        and np.array_equal(got[name], value) for name, value in values.items())\n"
               "report = {}\n"
               "for size in ('default', 'one leaf'):\n"
               "    if size == 'one leaf':\n"
               "        artifacts.GATHER_BYTES = 1\n"
               "    agreements.clear()\n"
               "    report[f'{size} every'] = [same(artifacts.collective_host(tree, phase='t')), len(agreements)]\n"
               "    print('gathered', jax.process_index(), json.dumps(report), flush=True)\n"
               "    agreements.clear()\n"
               "    got = artifacts.collective_host(tree, phase='t', held_by='first')\n"
               "    report[f'{size} first'] = [None if got is None else same(got), len(agreements)]\n"
               "    print('gathered', jax.process_index(), json.dumps(report), flush=True)\n")
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program, devices=1, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    # Each rank prints its report after every gather; the last line is the whole report.
    reports = dict(line.split("] gathered ", 1)[1].split(" ", 1)
                   for line in done.stdout.splitlines() if "] gathered " in line)
    assert {rank: json.loads(report) for rank, report in reports.items()} == {
        "0": {"default every": [True, 4], "default first": [True, 4],
              "one leaf every": [True, 28], "one leaf first": [True, 28]},
        "1": {"default every": [True, 4], "default first": [None, 4],
              "one leaf every": [True, 28], "one leaf first": [None, 28]}}, done.stdout


@pytest.mark.mesh(devices=2)
def test_a_rank_that_fails_in_a_gather_group_ends_the_gather_at_that_groups_agreement():
    """Rank 1's host copy of the second of four groups fails. Both ranks
    finished that group's computation, so rank 0 waits at the group's
    agreement rather than inside the next computation, hears the failure
    there, and never starts the third group."""
    program = ("import dew.training.runtime as runtime\n"
               "runtime.prepare_process()\n"
               "import jax, numpy as np\n"
               "from jax.sharding import NamedSharding, PartitionSpec\n"
               "from dew import artifacts\n"
               "from dew.training import MeshSpec, build_mesh\n"
               "mesh = build_mesh(MeshSpec(fsdp=jax.device_count()))\n"
               "value = np.arange(4 * jax.device_count(), dtype=np.float32)\n"
               "tree = [jax.make_array_from_callback(value.shape, NamedSharding(mesh, PartitionSpec('fsdp')),"
               " lambda index: value[index]) for _ in range(4)]\n"
               "artifacts.GATHER_BYTES = 1\n"
               "started = []\n"
               "gathered = artifacts._gathered\n"
               "def failing(group, held):\n"
               "    started.append(group)\n"
               "    whole = gathered(group, held)\n"
               "    if jax.process_index() == 1 and len(started) == 2:\n"
               "        raise RuntimeError('injected failure copying the second group')\n"
               "    return whole\n"
               "artifacts._gathered = failing\n"
               "try:\n"
               "    artifacts.collective_host(tree, phase='t')\n"
               "finally:\n"
               "    print('groups started', jax.process_index(), len(started), flush=True)\n")
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program, devices=1, timeout=300)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "groups started 0 2" in done.stdout, done.stdout
    assert "Process phase t transfer leaves 1-1 failed on rank 1" in done.stdout, done.stdout
    assert "RuntimeError: injected failure copying the second group" in done.stdout, done.stdout


@pytest.mark.mesh(devices=2)
def test_a_rank_whose_leaf_cannot_be_read_reports_at_the_gather_preflight():
    """Rank 1 has lost one of its shards (deleted, as a failed or donated
    computation leaves it). The gather's preflight finds it on rank 1 before
    either rank enters a gather computation, and both ranks end there with
    rank 1's error rather than waiting in a collective rank 1 never joins."""
    program = ("import dew.training.runtime as runtime\n"
               "runtime.prepare_process()\n"
               "import jax, numpy as np\n"
               "from jax.sharding import NamedSharding, PartitionSpec\n"
               "from dew import artifacts\n"
               "from dew.training import MeshSpec, build_mesh\n"
               "mesh = build_mesh(MeshSpec(fsdp=jax.device_count()))\n"
               "value = np.arange(4 * jax.device_count(), dtype=np.float32)\n"
               "tree = [jax.make_array_from_callback(value.shape, NamedSharding(mesh, PartitionSpec('fsdp')),"
               " lambda index: value[index]) for _ in range(2)]\n"
               "if jax.process_index() == 1:\n"
               "    tree[1].delete()\n"
               "artifacts.collective_host(tree, phase='t')\n")
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program, devices=1, timeout=300)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "Process phase t transfer preflight failed on rank 1" in done.stdout, done.stdout


@pytest.mark.mesh(devices=2)
def test_a_rank_that_holds_nothing_copies_none_of_the_tree_to_its_host():
    """With held_by="first" only process 0 holds the gathered tree. Process 1
    takes part in every computation and agreement, and copies none of the
    tree's values to its host: its preflight waits on each shard, which raises
    a failed computation's error as a copy would. The copy moved every rank's
    whole replica of a data-parallel policy to its host, 1.2 GiB of
    Qwen3-0.6B's weight push on each of three RTX 3090s. JAX's transfer guard
    logs every implicit device-to-host copy on a GPU, where the count is
    taken; process 0's copies of the tree show that the log is on."""
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("a CPU array reaches its host without a device-to-host transfer")
    program = ("import dew.training.runtime as runtime\n"
               "runtime.prepare_process()\n"
               "import jax, jax.numpy as jnp, numpy as np\n"
               "from jax.sharding import NamedSharding, PartitionSpec\n"
               "from dew import artifacts\n"
               "from dew.training import MeshSpec, build_mesh\n"
               "mesh = build_mesh(MeshSpec(fsdp=jax.device_count()))\n"
               "def placed(value, spec):\n"
               "    return jax.make_array_from_callback(value.shape, NamedSharding(mesh, spec),"
               " lambda index: value[index])\n"
               "tree = {'sharded': [placed(np.full((2 * jax.device_count(), 37), i, np.float32),"
               " PartitionSpec('fsdp')) for i in range(3)],\n"
               "        'replicated': [placed(np.full((7, 37), i, np.float32), PartitionSpec()) for i in range(3)],\n"
               "        'local': jnp.ones((5, 37))}\n"
               "with jax.transfer_guard_device_to_host('log'):\n"
               "    held = artifacts.collective_host(tree, phase='t', held_by='first')\n"
               "print('held', jax.process_index(), held is not None, flush=True)\n")
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program, devices=1, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "held 0 True" in done.stdout and "held 1 False" in done.stdout, done.stdout
    # Every leaf of the tree, and every shard of one, is (n, 37); nothing else the gather moves is.
    copies = {rank: sum(line.startswith(f"[{rank}] ") and "device-to-host transfer: shape=(" in line
                        and ",37)" in line for line in done.stdout.splitlines()) for rank in "01"}
    assert copies["0"] > 0 and copies["1"] == 0, (copies, done.stdout)


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
def test_two_pools_on_one_machine_take_a_free_port_each():
    """Two pools start together on one machine with no port named. Each
    coordinator listens on a port that was free when its launch began, where
    a fixed default had the second pool's rank dial the first pool's
    coordinator and fail."""
    program = ("from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "import time\n"
               "time.sleep(20)\n")
    pools = [start("--", sys.executable, "-c", program, devices=1) for _ in range(2)]
    for done in [finished(pool, timeout=300) for pool in pools]:
        assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.mesh(devices=2)
def test_a_pools_second_run_loads_what_its_first_compiled(tmp_path):
    """A pool runs twice over one persistent compilation cache, and the second
    run loads its step on every process. jax 0.11.2 keyed an executable by the
    compiling process's topology fingerprint, which on a GPU describes the
    device down to its NVLink links, and only process 0 writes entries. On the
    box, whose GPUs 0 and 1 share NVLink and 2 and 3 do not, a pool of GPUs 1
    and 2 loaded the step on one process and compiled it on the other, whose
    sharded autotuning then waited for ever for its peer's share. Each process
    here reports a fingerprint of its own, as those two did, whatever devices
    the run has; the pinned jax hashes the fingerprints of every process a
    computation spans."""
    cache, records = tmp_path / "cache", tmp_path / "records"
    program = ("import json, sys\n"
               "from pathlib import Path\n"
               "import jax\n"
               "from jax._src.lib import xla_client\n"
               "topology_of = xla_client.get_topology_for_devices\n"
               "class Apart:\n"
               "    def __init__(self, topology):\n"
               "        self.topology = topology\n"
               "    def fingerprint(self):\n"
               "        return self.topology.fingerprint() ^ (jax.process_index() + 1)\n"
               "xla_client.get_topology_for_devices = lambda devices: Apart(topology_of(devices))\n"
               "events = []\n"
               "jax.monitoring.register_event_listener(lambda event, **_: events.append(event))\n"
               "from dew.training.runtime import prepare_process\n"
               "prepare_process(compilation_cache_dir=sys.argv[1])\n"
               "import jax.numpy as jnp, numpy as np\n"
               "from jax.sharding import NamedSharding, PartitionSpec\n"
               "from dew.training import MeshSpec, build_mesh\n"
               "mesh = build_mesh(MeshSpec(fsdp=jax.device_count()))\n"
               "def placed(value, spec):\n"
               "    return jax.make_array_from_callback(value.shape, NamedSharding(mesh, spec),"
               " lambda index: value[index])\n"
               "weights = [placed((np.random.default_rng(i).standard_normal((1024, 1024)) / 32)"
               ".astype(jnp.bfloat16), PartitionSpec()) for i in range(4)]\n"
               "x = placed(np.ones((jax.device_count() * 512, 1024), jnp.bfloat16), PartitionSpec('fsdp'))\n"
               "def loss(x, weights):\n"
               "    for weight in weights:\n"
               "        x = jax.nn.gelu(x @ weight)\n"
               "    return (x.astype(jnp.float32) ** 2).mean()\n"
               "before = len(events)\n"
               "float(jax.jit(lambda x, weights: jax.grad(loss)(x, weights).astype(jnp.float32).sum())(x, weights))\n"
               "step = events[before:]\n"
               "Path(sys.argv[2], f'{jax.process_index()}.json').write_text(json.dumps(\n"
               "    [step.count('/jax/compilation_cache/compile_requests_use_cache'),\n"
               "     step.count('/jax/compilation_cache/cache_hits')]))\n")
    runs = []
    for run in range(2):
        out = records / str(run)
        out.mkdir(parents=True)
        done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program,
                      str(cache), str(out), devices=1, timeout=150)
        assert done.returncode == 0, done.stdout + done.stderr
        runs.append([json.loads(path.read_text()) for path in sorted(out.iterdir())])
    # [lookups, hits] a process: the first run compiles every lookup, the second loads it.
    compiled = runs[0][0][0]
    assert compiled > 0 and runs == [[[compiled, 0]] * 2, [[compiled, compiled]] * 2], runs


@pytest.mark.mesh(devices=2)
def test_a_gpu_pool_whose_jax_keys_its_processes_apart_compiles_without_the_cache():
    """A jax installed around Dew's pin, such as an image's own, may key a
    computation that spans processes apart on each of them. A GPU pool on one
    turns the persistent cache off before its first compile: a rank that
    compiled a step its peers loaded would wait for ever for their shares of
    its autotuning. A CPU pool keeps the cache, since its compiles wait on no
    peer."""
    import jax

    program = ("import jax\n"
               "jax.config.update('jax_enable_compilation_cache', True)\n"
               "import dew.training.runtime as runtime\n"
               "runtime._pool_keys_alike = lambda: False\n"
               "runtime.prepare_process()\n"
               "print('cache', jax.process_index(), jax.config.jax_enable_compilation_cache, flush=True)\n")
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program, devices=1, timeout=150)
    assert done.returncode == 0, done.stdout + done.stderr
    kept = jax.default_backend() != "gpu"
    assert f"cache 0 {kept}" in done.stdout and f"cache 1 {kept}" in done.stdout, done.stdout


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


def scheduled(tmp_path: Path, mesh: dict, processes: int, *flags: str) -> tuple[dict, dict[str, np.ndarray]]:
    """The record and the final parameters of a RolloutScheduler run over two devices."""
    out = tmp_path / f"{len(list(tmp_path.iterdir()))}.json"
    done = launch("--processes-per-host", str(processes), "--", sys.executable, str(SCHEDULER),
                  "--out", str(out), "--mesh", json.dumps(mesh), *flags, devices=2 // processes)
    assert done.returncode == 0, done.stdout + done.stderr
    with np.load(out.with_suffix(".npz")) as params:
        return json.loads(out.read_text()), {name: params[name] for name in params.files}


def handed(record: dict, process: int) -> dict[str, np.ndarray]:
    """The rows `process` handed the step."""
    return {name: np.asarray(rows[process]) for name, rows in record["rows"].items()}


@pytest.mark.mesh(devices=2)
def test_every_process_schedules_its_own_rollouts_and_the_pool_trains_on_all_of_them(tmp_path):
    """fsdp across two processes: each schedules the task rows of its own
    share and packs its own rows. Together they pack the rows one process
    packs over the whole batch, the proximal rescoring over the pool's rows
    gives each row the likelihood one process gives it, and the update moves
    the parameters one process moves."""
    pool, pooled = scheduled(tmp_path, {"fsdp": 2}, 2)
    alone, single = scheduled(tmp_path, {"fsdp": 2}, 1)
    assert pool["partition"] == {"count": 2, "readers": 1} and pool["step"] == 1

    def rows(record: dict) -> dict[str, np.ndarray]:
        """Every row every process handed the step, sorted: groups complete in any order."""
        stacked = {name: np.concatenate(value) for name, value in record["rows"].items()}
        order = np.lexsort(stacked["input_ids"].T[::-1])
        return {name: value[order] for name, value in stacked.items()}

    together, reference = rows(pool), rows(alone)
    for name in ("input_ids", "response_mask", "behavior_log_probs"):
        np.testing.assert_array_equal(together[name], reference[name])
    np.testing.assert_allclose(together["old_log_probs"], reference["old_log_probs"], atol=TOLERANCE)
    for name, value in single.items():
        np.testing.assert_allclose(pooled[name], value, atol=TOLERANCE)


@pytest.mark.mesh(devices=2)
def test_the_readers_of_one_share_train_on_the_rows_its_first_reader_sampled(tmp_path):
    """tensor across two processes: both read the one share, and their
    engines would draw differently (`--vary`). The first reader samples and
    the second trains on its rows, so the step's two tensor shards hold one
    batch, the batch one process samples, and the update is that process's.
    Both sampling, the shards held different rows and the step trained on
    them as one batch."""
    pool, pooled = scheduled(tmp_path, {"tensor": 2}, 2, "--vary")
    alone, single = scheduled(tmp_path, {"tensor": 2}, 1, "--vary")
    assert pool["partition"] == {"count": 1, "readers": 2} and pool["step"] == 1
    first, second, reference = handed(pool, 0), handed(pool, 1), handed(alone, 0)
    for name, value in first.items():
        np.testing.assert_array_equal(second[name], value)
    for name in ("input_ids", "response_mask", "behavior_log_probs"):
        np.testing.assert_array_equal(first[name], reference[name])
    np.testing.assert_allclose(first["old_log_probs"], reference["old_log_probs"], atol=TOLERANCE)
    for name, value in single.items():
        np.testing.assert_allclose(pooled[name], value, atol=TOLERANCE)
