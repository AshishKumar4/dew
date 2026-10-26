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

import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from lane_environment import outside_any_cluster

from dew.training import MeshSpec
from dew.training.distributed import parameter_axes

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
    """The worker's record of a pool of `processes` over `devices` CPU devices,
    whose trained parameters are split over every parameter axis the mesh
    has (`sharded.assert_sharded`)."""
    out = tmp_path / f"{len(list(tmp_path.iterdir()))}.json"
    done = launch("--processes-per-host", str(processes), "--",
                  sys.executable, str(WORKER), "--out", str(out), "--mesh", json.dumps(mesh),
                  devices=devices // processes)
    assert done.returncode == 0, done.stdout + done.stderr
    record = json.loads(out.read_text())
    wanted = {axis for axis in parameter_axes(record["mesh"]) if record["mesh"][axis] > 1}
    assert wanted <= set(record["split_axes"]), (wanted, record["split_axes"])
    return record


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> dict:
    return train(tmp_path_factory.mktemp("reference"), {"fsdp": 8}, processes=1)


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "apart"])
def test_a_pool_caches_by_default_only_where_its_processes_share_the_directory(tmp_path, monkeypatch, shared):
    """A pool of two processes caches its compilations under the user's cache directory where both
    read it as one, and leaves the cache off where each has its own: JAX writes entries from process
    0 alone, so caches apart would find a step on one process and compile it on the other."""
    program = (
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "base = Path(sys.argv[1])\n"
        "own = base / os.environ['DEW_PROCESS_ID']\n"
        "os.environ['XDG_CACHE_HOME'] = str(base if sys.argv[3] == 'shared' else own)\n"
        "from dew.training.runtime import prepare_process\n"
        "prepare_process()\n"
        "import jax\n"
        "held = jax.config.jax_compilation_cache_dir\n"
        "Path(sys.argv[2], f'{jax.process_index()}.json').write_text(json.dumps(held))\n"
    )
    out = tmp_path / "out"
    out.mkdir()
    for name in ("JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE"):
        monkeypatch.delenv(name, raising=False)
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program,
                  str(tmp_path / "home"), str(out), "shared" if shared else "apart", devices=1, timeout=150)
    assert done.returncode == 0, done.stdout + done.stderr
    directories = [json.loads(path.read_text()) for path in sorted(out.iterdir())]
    minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    expected = str(tmp_path / "home" / "dew" / "xla" / minor) if shared else None
    assert directories == [expected] * 2


def test_hybrid_sharding_groups_hosts_into_replicas_and_trains_like_fsdp(tmp_path, reference):
    """Four processes of two devices, fsdp=2 over replicas=2: each replica is
    two processes and fsdp crosses the pair, where `jax.make_mesh` would put
    every fsdp group on one process. The losses are plain fsdp's over all
    eight devices."""
    pool = train(tmp_path, {"fsdp": 2, "replicas": 2}, processes=4)
    assert pool["processes"] == 4 and pool["mesh"]["data"] == 4
    assert pool["fsdp_groups"] == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert max(abs(a - b) for a, b in zip(pool["losses"], reference["losses"], strict=True)) < TOLERANCE


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


def test_replicas_in_a_process_outside_any_pool_are_refused_by_granule():
    """A process that never joined a pool has CPU devices without a
    `slice_index`; it is one granule, so replicas are refused by the same
    rule, not by the missing attribute."""
    with pytest.raises(ValueError):
        MeshSpec(fsdp=4, replicas=2).build()


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
            "from dew.training import MeshSpec\n"
            "mesh = MeshSpec(fsdp=jax.device_count()).build()\n"
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

@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="pins the child to four cores")
def test_eight_cpu_devices_on_four_cores_run_collectives_ahead_of_the_host():
    """A loop that dispatches collective steps faster than eight simulated
    CPU devices run them fills each device's 32 computations in flight. XLA:CPU
    then blocks the next launch's dispatch on the pool that runs the devices,
    and with a thread per device, a 4-core runner's pool, the launch it waits
    for never gets its eighth: the rendezvous aborted the process after 40 s,
    as tests/test_discrete.py's toy run did on CI. The suite's environment
    (`configure_lane`) gives that pool room for both."""
    if os.environ["JAX_PLATFORMS"] != "cpu":
        pytest.skip("the CPU lane's simulated devices")
    cores = sorted(os.sched_getaffinity(0))[:4]
    program = (f"import os\nos.sched_setaffinity(0, {cores})\n"
               "import jax, jax.numpy as jnp\n"
               "from jax.sharding import Mesh, NamedSharding, PartitionSpec as P\n"
               "rows = NamedSharding(Mesh(jax.devices(), ('d',)), P('d'))\n"
               "step = jax.jit(lambda x: x / jnp.sum(jnp.tanh(x @ x.T)), out_shardings=rows)\n"
               "x = jax.device_put(jnp.ones((8 * 256, 256), jnp.float32), rows)\n"
               "for _ in range(400):\n"
               "    x = step(x)\n"
               "x.block_until_ready()\n"
               "print('devices', jax.device_count())\n")
    done = subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr[-3000:]


@pytest.mark.mesh(devices=0)
def test_one_slurm_task_joins_no_pool():
    """srun with one task sets SLURM_JOB_ID, and JAX's Slurm detection then
    starts a one-process pool whose coordinator is the node's name. A
    container on the node need not resolve that name (the box's did not):
    the process waited 300 s to register and aborted. One task is no pool,
    so the process starts on its own."""
    env = {**outside_any_cluster(os.environ), **ONE_SLURM_TASK, "PYTHONPATH": str(REPO_ROOT / "src"),
           "JAX_PLATFORMS": "cpu"}
    program = ("from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "import jax\n"
               "print('processes', jax.process_count())\n")
    done = subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.mesh(devices=0)
def test_a_slurm_step_that_leaves_gpus_idle_is_refused_before_it_joins():
    """srun started one task a node on nodes of four GPUs, not through dew
    launch. jax gives each Slurm task the one GPU at its SLURM_LOCALID, so
    the run would train on one GPU of four and nothing would say so; the
    process refuses before it joins, naming the task count."""
    env = {**outside_any_cluster(os.environ), **ONE_SLURM_TASK, "PYTHONPATH": str(REPO_ROOT / "src"),
           "SLURM_NTASKS": "2", "SLURM_STEP_TASKS_PER_NODE": "1(x2)", "SLURM_NODEID": "0",
           "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    env.pop("JAX_PLATFORMS", None)
    program = ("from dew.training.runtime import prepare_process\n"
               "prepare_process()\n")
    done = subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode != 0, done.stdout


@pytest.mark.mesh(devices=0)
def test_an_mpirun_inside_one_slurm_task_forms_its_pool():
    """mpirun started inside a one-task allocation sets Open MPI's rank
    variables beside Slurm's. JAX's detection reads Open MPI's first and
    forms the pool mpirun started, so the one-task rule has to leave the
    process to it, here a pool of one rank."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**outside_any_cluster(os.environ), **ONE_SLURM_TASK, "PYTHONPATH": str(REPO_ROOT / "src"),
           "JAX_PLATFORMS": "cpu",
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


@pytest.mark.mesh(devices=0)
def test_a_declared_pool_without_a_coordinator_is_refused():
    """DEW_PROCESS_COUNT declares a pool of two and nothing names its
    coordinator, as when a launcher's environment reached a rank in part.
    The rank refuses rather than train alone as if it were the pool; with
    no pool declared, the same process runs alone."""
    program = ("from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "import jax\n"
               "print('processes', jax.process_count())\n")
    env = {**outside_any_cluster(os.environ), "PYTHONPATH": str(REPO_ROOT / "src"), "JAX_PLATFORMS": "cpu"}
    declared = subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT, capture_output=True, text=True,
                              env={**env, "DEW_PROCESS_COUNT": "2", "DEW_PROCESS_ID": "1"}, timeout=120)
    alone = subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT, capture_output=True, text=True,
                           env=env, timeout=120)
    assert declared.returncode != 0, declared.stdout
    assert alone.returncode == 0, alone.stdout + alone.stderr


@pytest.mark.gpu(reason="a cluster pins device ids on GPU; the CPU backend takes none")
@pytest.mark.mesh(devices=2)
def test_a_launched_process_keeps_every_device_inside_a_slurm_step():
    """`dew launch --processes-per-host 1` gives its one process every local
    device, and names none of them. Inside a Slurm step, JAX's detection would
    still place it:
    it pins a process it finds in Slurm's variables to the GPU at the step's
    SLURM_LOCALID, one device of the launcher's several. A process dew launch
    started takes its placement from the launcher."""

    env = {**outside_any_cluster(os.environ), **ONE_SLURM_TASK, "PYTHONPATH": str(REPO_ROOT / "src")}
    program = ("from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "import jax\n"
               "print('local devices', jax.local_device_count())\n")
    done = finished(subprocess.Popen(
        [sys.executable, "-m", "dew.cli.main", "launch", "--processes-per-host", "1", "--",
         sys.executable, "-c", program],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr


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
    assert time.monotonic() - started < 120, done.stdout + done.stderr


@pytest.mark.mesh(devices=2)
def test_a_pool_forms_and_fails_together_after_orbaxs_public_multihost_loads():
    """Importing orbax.checkpoint.multihost replaces the private module that
    `from orbax.checkpoint import multihost` had named, and the public one
    has no accessor for jax.distributed's client: every rank of such a pool
    failed to start. It forms, and rank 1's failure still ends it."""
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c",
                  "import orbax.checkpoint.multihost\n"
                  + stepping_pool("raise RuntimeError('injected failure')"), devices=1, timeout=600)
    assert done.returncode == 1, done.stdout + done.stderr


@pytest.mark.mesh(devices=2)
def test_a_failure_no_agreement_hears_ends_the_peer_after_the_grace():
    """Rank 1 publishes a failure and ends its program; rank 0 waits on its
    host and never fails. Only the coordination service carries the failure
    to rank 0, whose watch ends it once the grace has passed unheard."""
    program = ("import orbax.checkpoint.multihost\n"
               "import os, threading\n"
               "import dew._coordination as engine\n"
               "engine.FAILURE_GRACE_SECONDS = 3.0\n"
               "engine.FAILURE_POLL_SECONDS = 0.5\n"
               "from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "import jax\n"
               "if jax.process_index() == 1:\n"
               "    engine.publish_failure(RuntimeError('injected failure'), 'a step')\n"
               "else:\n"
               "    threading.Event().wait()\n")
    started = time.monotonic()
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program, devices=1, timeout=300)
    output = done.stdout + done.stderr
    assert done.returncode != 0, output
    heard = [line for line in output.splitlines() if line.startswith("[0]") and "injected failure" in line]
    assert heard, output
    assert time.monotonic() - started < 120, output


def four_rank_pool(program: str) -> tuple[int, str]:
    """`program` run by every rank of a pool of four processes of one device each: the launch's exit
    code and everything it printed."""
    done = launch("--processes-per-host", "4", "--", sys.executable, "-c", program, devices=1, timeout=300)
    return done.returncode, done.stdout + done.stderr


@pytest.mark.mesh(devices=4)
def test_a_stopped_pool_ends_its_ranks_without_aborts():
    """Rank 3 fails and the launch stops ranks 0 to 2, which wait. A JAX
    process takes SIGTERM as a preemption notice and runs on, so each is
    SIGKILLed after the grace. Rank 0's process holds the pool's coordination
    service: killed first, it left the others to abort in XLA's error
    polling, with "Check failure" stack traces that read as crashes of their
    own (DistSequence's land7 run, stopped by its timeout). Stopped last, it
    leaves no rank to abort."""
    program = ("import os, time\n"
               "from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "if os.environ['DEW_PROCESS_ID'] == '3':\n"
               "    raise RuntimeError('rank 3 fails')\n"
               "time.sleep(600)\n")
    code, output = four_rank_pool(program)
    assert code == 1, output


def stopped_by_sigterm(process: subprocess.Popen, after: str, timeout: float = 300) -> tuple[int, str]:
    """SIGTERM `process` once its output shows `after`, as a scheduler stops
    a job, and return its exit code and everything it printed."""
    assert process.stdout is not None and process.stderr is not None
    lines: list[str] = []

    def read(stream) -> None:
        # Extended a line at a time, so the loop below sees each as it comes.
        lines.extend(stream)

    readers = [threading.Thread(target=read, args=(stream,), daemon=True)
               for stream in (process.stdout, process.stderr)]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    while not any(after in line for line in lines):
        assert process.poll() is None and time.monotonic() < deadline, "".join(lines)
        time.sleep(0.1)
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        pytest.fail(f"still running {timeout}s after SIGTERM\n{''.join(lines)}")
    for reader in readers:
        reader.join(timeout=30)
    return process.returncode, "".join(lines)


def resumed_where_it_stopped(tmp_path: Path, run) -> None:
    """Train the worker's varying stream for eight steps uninterrupted; again,
    SIGTERMed after its second step; and once more from its checkpoints.
    `run(arguments, stop)` runs the worker with `arguments` and returns its
    exit code and output, SIGTERMed after the second step when `stop`."""
    steps, checkpoints = 8, tmp_path / "checkpoints"
    worker = ["--steps", str(steps), "--vary"]
    code, output = run([*worker, "--out", str(tmp_path / "whole.json")], stop=False)
    assert code == 0, output
    code, output = run([*worker, "--out", str(tmp_path / "stopped.json"), "--checkpoints", str(checkpoints),
                        "--step-seconds", "1"], stop=True)
    assert code == 128 + signal.SIGTERM, output
    stopped = re.search(r"Preempted at step (\d+)", output)
    assert stopped is not None, output
    at = int(stopped[1])
    assert 2 <= at < steps, output
    code, output = run([*worker, "--out", str(tmp_path / "resumed.json"), "--checkpoints", str(checkpoints)],
                       stop=False)
    assert code == 0, output
    whole, resumed = (json.loads((tmp_path / name).read_text()) for name in ("whole.json", "resumed.json"))
    # The state, the key and the data position all came back: every loss
    # after the stop is the uninterrupted run's, to the bit, and so is every
    # leaf of the state it ends with.
    assert resumed["loss_steps"] == list(range(at + 1, steps + 1)), output
    assert resumed["losses"] == whole["losses"][at:], output
    assert resumed["state_digest"] == whole["state_digest"], output


@pytest.mark.parametrize("processes", [pytest.param(2, marks=pytest.mark.mesh(devices=2)),
                                       pytest.param(4, marks=pytest.mark.mesh(devices=4))])
def test_a_pool_stopped_by_sigterm_checkpoints_and_resumes_where_it_stopped(tmp_path, processes):
    """A scheduler stops a job with SIGTERM and SIGKILLs it a grace later:
    Slurm's KillWait, Kubernetes' termination grace, a spot VM's notice.
    JAX's preemption service takes the signal in every rank of a pool, so the
    ranks ran on until the SIGKILL, and everything since the last checkpoint
    was lost. The fit now stops at the step the ranks agree on, writes that
    step's state and data position, and exits with SIGTERM's code; run
    again, it resumes there, bit for bit the run it continues. A pool of four
    over fsdp 4 is issue #11's: its resume against a continuous pool of four,
    which reduces in the same order."""
    def run(arguments: list[str], stop: bool) -> tuple[int, str]:
        pool = start("--processes-per-host", str(processes), "--", sys.executable, str(WORKER),
                     "--mesh", json.dumps({"fsdp": processes}), *arguments, devices=1)
        if stop:
            return stopped_by_sigterm(pool, "] step 2/")
        done = finished(pool, timeout=300)
        return done.returncode, done.stdout + done.stderr

    resumed_where_it_stopped(tmp_path, run)


@pytest.mark.mesh(devices=1)
def test_a_lone_process_stopped_by_sigterm_checkpoints_and_resumes_where_it_stopped(tmp_path):
    """A process in no pool has no preemption service: SIGTERM's default
    ended it where it stood. The fit catches it and stops as a pool does."""
    import jax

    # No cluster around the test makes a pool of it: not Slurm's, Open MPI's,
    # a Cloud TPU VM's or a Kubernetes pod's.
    env = {**outside_any_cluster(os.environ), "PYTHONPATH": str(REPO_ROOT / "src")}
    if jax.default_backend() != "gpu":
        # A TPU belongs to the one process that opened it, and this one has:
        # the worker aborted with "The TPU is already in use". It takes CPU
        # devices, as the pool tests' ranks do off a GPU (`start`).
        env["JAX_PLATFORMS"] = "cpu"
    # Each line as it is printed, as `dew launch` has its ranks write: a piped
    # stdout holds the step lines until exit, and the signal would come after
    # the fit had ended.
    env["PYTHONUNBUFFERED"] = "1"

    def run(arguments: list[str], stop: bool) -> tuple[int, str]:
        process = subprocess.Popen([sys.executable, str(WORKER), "--mesh", "{}", *arguments],
                                   cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        if stop:
            return stopped_by_sigterm(process, "step 2/")
        done = finished(process, timeout=300)
        return done.returncode, done.stdout + done.stderr

    resumed_where_it_stopped(tmp_path, run)


@pytest.mark.mesh(devices=4)
def test_a_pool_whose_program_ends_while_its_failure_watches_read_exits_cleanly():
    """Each rank's failure watch reads the pool's coordination service from
    a daemon thread, inside jaxlib with the GIL released. A read still under
    way when Python finalizes returns to a runtime that ends the thread with
    pthread_exit, and the unwind through jaxlib's GIL guard aborted ranks
    whose programs had run to their end: SIGABRT and "terminate called ..."
    in the pool tests of two CI runs (ci/train-2, ci/commit-select). Every
    watch here reads without a pause, so each rank ends mid-read."""
    program = ("import dew._coordination as engine\n"
               "engine.FAILURE_POLL_SECONDS = 0\n"
               "from dew.training.runtime import prepare_process\n"
               "prepare_process()\n")
    code, output = four_rank_pool(program)
    assert code == 0, output


@pytest.mark.gpu(reason="the execution bound is XLA's GPU watchdog; CPU pools have none")
@pytest.mark.mesh(devices=2)
def test_a_rank_that_stalls_between_collectives_ends_the_pool():
    """Rank 1 stops before its fourth step without failing, as a rank blocked
    on a read, or on a compile that waits for its peers, does. Rank 0 waits
    inside that step's reduction. Every process stays alive and none fails,
    so only a bound on one execution can end the pool: rank 0's runs past
    the pool's execution timeout, shortened here, and the launch ends."""
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
               "from dew._coordination import agreed\n"
               "if jax.process_index() == 1:\n"
               "    time.sleep(45)\n"
               "print('agreed', agreed('upload', lambda: jax.process_index()))\n")
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program,
                  devices=1, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.mesh(devices=4)
def test_every_process_of_a_pool_takes_the_same_link_bandwidth(tmp_path):
    """A tensor axis over two processes' devices. Each process times its own
    part of the gathers, and processes that placed the step from different
    figures could compile different programs and hang in their collectives,
    so every process takes the pool's one figure. A later pool on the same
    devices, as a resume is, takes that figure from beside the compilation
    cache rather than a new measurement that noise moves."""
    program = ("import dew.training.runtime as runtime\n"
               f"runtime.prepare_process(compilation_cache_dir={str(tmp_path / 'cache')!r})\n"
               "import jax\n"
               "from dew.training import MeshSpec\n"
               "from dew.training.distributed import link_bandwidth\n"
               "mesh = MeshSpec(tensor=jax.device_count()).build()\n"
               "print('measured', jax.process_index(), repr(link_bandwidth(mesh, 'tensor')), flush=True)\n")
    pools = []
    for _ in range(2):
        done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program, devices=2,
                      timeout=300)
        assert done.returncode == 0, done.stdout + done.stderr
        figures = dict(line.split("] measured ", 1)[1].split(" ", 1)
                       for line in done.stdout.splitlines() if "] measured " in line)
        assert sorted(figures) == ["0", "1"], done.stdout
        assert figures["0"] == figures["1"] and float(figures["0"]) > 0, figures
        pools.append(figures["0"])
    assert pools[0] == pools[1] and len(list((tmp_path / "cache" / "dew-links").iterdir())) == 1


@pytest.mark.mesh(devices=4)
def test_a_link_figure_is_kept_for_its_placement_and_carried_by_a_checkpoint(tmp_path):
    """Two tensor meshes of one shape over the same four devices, grouped otherwise along the axis,
    are other links: the second measures its own figure beside the compilation cache rather than
    take the first's, and a mesh built again over the first placement takes the first's. A
    checkpoint's rung carries each figure its trace decided from, by that placement, for a resumed
    run to decide alike."""
    import jax
    import optax
    from jax.sharding import Mesh

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer
    from dew.training.distributed import link_bandwidth, link_topology

    devices = np.asarray(jax.devices()[:4])
    paired = Mesh(devices.reshape(2, 2), ("data", "tensor"))
    crossed = Mesh(devices.reshape(2, 2).T, ("data", "tensor"))
    assert link_topology(paired, "tensor") != link_topology(crossed, "tensor")
    assert link_topology(paired, "tensor") == link_topology(Mesh(devices.reshape(2, 2), ("data", "tensor")),
                                                            "tensor")
    # The figures go beside this test's own compilation cache.
    kept = jax.config.jax_compilation_cache_dir
    jax.config.update("jax_compilation_cache_dir", str(tmp_path / "cache"))
    try:
        first = link_bandwidth(paired, "tensor", size=1 << 16)
        link_bandwidth(crossed, "tensor", size=1 << 16)
        assert len(list((tmp_path / "cache" / "dew-links").iterdir())) == 2
        again = link_bandwidth(Mesh(devices.reshape(2, 2), ("data", "tensor")), "tensor", size=1 << 16)
        assert again == first and len(list((tmp_path / "cache" / "dew-links").iterdir())) == 2
    finally:
        jax.config.update("jax_compilation_cache_dir", kept)

    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=8)

    def trainer():
        return Trainer(LMObjective(model, seq_len=8), optax.sgd(1e-2), key=jax.random.key(0),
                       mesh=MeshSpec(tensor=2))

    saved = trainer()
    saved._bandwidths[paired, "tensor"] = first
    rung = saved._checkpoint_rung()
    assert rung["links"] == {link_topology(paired, "tensor"): first}
    resumed = trainer()
    resumed._resume_rung(rung)
    assert resumed._resumed_links == {link_topology(paired, "tensor"): first}


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
               "from dew import distributed as coordination\n"
               "from dew.training import MeshSpec\n"
               "mesh = MeshSpec(fsdp=jax.device_count()).build()\n"
               "value = np.arange(4 * jax.device_count(), dtype=np.float32)\n"
               "tree = [jax.make_array_from_callback(value.shape, NamedSharding(mesh, PartitionSpec('fsdp')),"
               " lambda index: value[index]) for _ in range(2)]\n"
               "if jax.process_index() == 1:\n"
               "    tree[1].delete()\n"
               "coordination.collective_host(tree, phase='t')\n")
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program, devices=1, timeout=300)
    assert done.returncode != 0, done.stdout + done.stderr


@pytest.mark.gpu(reason="a CPU array reaches its host without a device-to-host transfer")
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
    program = (
        "import dew.training.runtime as runtime\n"
        "runtime.prepare_process()\n"
        "import jax, jax.numpy as jnp, numpy as np\n"
        "from jax.sharding import NamedSharding, PartitionSpec\n"
        "from dew import distributed as coordination\n"
        "from dew.training import MeshSpec\n"
        "mesh = MeshSpec(fsdp=jax.device_count()).build()\n"
        "def placed(value, spec):\n"
        "    return jax.make_array_from_callback(value.shape, NamedSharding(mesh, spec),"
        " lambda index: value[index])\n"
        "tree = {'sharded': [placed(np.full((2 * jax.device_count(), 37), i, np.float32),"
        " PartitionSpec('fsdp')) for i in range(3)],\n"
        "        'replicated': [placed(np.full((7, 37), i, np.float32), PartitionSpec()) "
        "for i in range(3)],\n"
        "        'local': jnp.ones((5, 37))}\n"
        "with jax.transfer_guard_device_to_host('log_explicit'):\n"
        "    held = coordination.collective_host(tree, phase='t', held_by='first')\n"
        "print('held', jax.process_index(), held is not None, flush=True)\n"
    )
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program, devices=1, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
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
def test_a_pool_refuses_a_checkpoint_directory_its_processes_do_not_share(tmp_path):
    """Each process names its own disk's copy of the run directory, as hosts
    with nothing shared do. Orbax writes one checkpoint across the pool and
    reads any process's shards back, so process 0 would commit steps missing
    the others' shards while they saw none, and a resume would train
    different states. Asking for the latest step reads only; at the first
    write every process refuses, naming the one that cannot see the
    directory."""
    program = ("import os, sys\n"
               "from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "import jax.numpy as jnp\n"
               "from dew.checkpoints import Checkpoints\n"
               "host = os.path.join(sys.argv[1], 'host' + os.environ['DEW_PROCESS_ID'])\n"
               "checkpoints = Checkpoints(os.path.join(host, 'run'))\n"
               "print('latest', checkpoints.latest, flush=True)\n"
               "checkpoints.save(1, {'params': {'w': jnp.zeros(1)}})\n")
    done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program,
                  str(tmp_path), devices=1, timeout=300)
    assert done.returncode != 0 and all(f"[{rank}] latest None" in done.stdout for rank in (0, 1)), \
        done.stdout + done.stderr


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
def test_a_share_whose_calls_recorded_nothing_holds_the_records_its_peer_packed(tmp_path):
    """fsdp across two processes over a mixture, every call recording its
    routing and its sampler's support: the second share's tasks all
    truncate, so its calls train on nothing and pack no record. It holds
    the records' empty forms, so the pool hands the step one tree, and the
    pool trains on the rows and moves the parameters one process does."""
    pool, pooled = scheduled(tmp_path, {"fsdp": 2}, 2, "--records")
    alone, single = scheduled(tmp_path, {"fsdp": 2}, 1, "--records")
    assert pool["partition"] == {"count": 2, "readers": 1} and pool["step"] == 1
    trained = handed(pool, 1)["response_mask"]
    assert not trained.any(), "the second share was meant to train on nothing"
    together = {name: np.concatenate(value) for name, value in pool["rows"].items()}
    reference = {name: np.concatenate(value) for name, value in alone["rows"].items()}
    order, expected = (np.lexsort(rows["input_ids"].T[::-1]) for rows in (together, reference))
    for name in ("input_ids", "response_mask", "behavior_log_probs"):
        np.testing.assert_array_equal(together[name][order], reference[name][expected])
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


@pytest.mark.mesh(devices=2)
def test_a_pools_second_run_loads_what_its_first_compiled(tmp_path):
    """A pool runs twice over one persistent compilation cache: process 0 writes
    the entries, and its second run loads its step. Another process loads it
    where its key is process 0's. JAX keys an executable by its process's
    topology fingerprint: on a GPU, whose pool shares one topology (jaxlib 0.12
    starts none whose fingerprints differ, openxla/xla#49677;
    docs/guides/multi-node.md), every process does; on the CPU, where a
    process's topology is its own devices, the others compile the step again,
    as upstream JAX does (Main, 2026-10-26: no fork code for the CPU alone)."""
    cache, records = tmp_path / "cache", tmp_path / "records"
    program = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "import jax\n"
        "events = []\n"
        "jax.monitoring.register_event_listener(lambda event, **_: events.append(event))\n"
        "jax.config.update('jax_explain_cache_misses', True)\n"
        "from dew.training.runtime import prepare_process\n"
        "prepare_process(compilation_cache_dir=sys.argv[1])\n"
        "import jax.numpy as jnp, numpy as np\n"
        "from jax.sharding import NamedSharding, PartitionSpec\n"
        "from dew.training import MeshSpec\n"
        "mesh = MeshSpec(fsdp=jax.device_count()).build()\n"
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
        "float(jax.jit(lambda x, weights: jax.grad(loss)(x, weights).astype(jnp.float32).sum())"
        "(x, weights))\n"
        "step = events[before:]\n"
        "Path(sys.argv[2], f'{jax.process_index()}.json').write_text(json.dumps(\n"
        "    [step.count('/jax/compilation_cache/compile_requests_use_cache'),\n"
        "     step.count('/jax/compilation_cache/cache_hits'), jax.default_backend()]))\n"
    )
    runs, logged = [], []
    for run in range(2):
        out = records / str(run)
        out.mkdir(parents=True)
        done = launch("--processes-per-host", "2", "--", sys.executable, "-c", program,
                      str(cache), str(out), devices=1, timeout=150)
        assert done.returncode == 0, done.stdout + done.stderr
        runs.append([json.loads(path.read_text()) for path in sorted(out.iterdir())])
        # What each rank's jax said of the cache: the key it missed, the entry
        # it wrote or could not, which says why a second run loads nothing.
        logged += [f"run {run}: {line}" for line in (done.stdout + done.stderr).splitlines()
                   if "cache" in line.lower()]
    # [lookups, hits, backend] a process: the first run compiles every lookup; the
    # second loads them on process 0, and on process 1 where it keys as process 0.
    compiled, backend = runs[0][0][0], runs[0][0][2]
    keyed_alike = backend != "cpu"
    assert compiled > 0 and runs == [
        [[compiled, 0, backend]] * 2,
        [[compiled, compiled, backend], [compiled, compiled if keyed_alike else 0, backend]]], (
        runs, "\n".join(logged))
