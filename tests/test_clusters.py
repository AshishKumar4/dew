"""The schedulers a run's processes start under (`dew.clusters`): what each reads of its own variables, which
one a process runs under, and a pool joined through a scheduler of the user's own."""

import os
import sys
from pathlib import Path

import pytest
from lane_environment import outside_any_cluster
from process_support import coordinator_port, output, owned

from dew.clusters import CloudTPU, Joining, Kubernetes, Launched, OpenMPI, Slurm, detected_environment

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_TASK = {"SLURM_JOB_ID": "4242", "SLURM_NTASKS": "4", "SLURM_PROCID": "3", "SLURM_LOCALID": "1",
              "SLURM_STEP_NODELIST": "node[1-2]"}
"""What srun sets on the fourth task of a step of four, the second on its node."""
MPI_RANK = {"OMPI_COMM_WORLD_SIZE": "8", "OMPI_COMM_WORLD_RANK": "5", "OMPI_COMM_WORLD_LOCAL_RANK": "1"}
"""What mpirun sets on the sixth of eight ranks, the second on its host."""


def test_each_scheduler_joins_by_jaxs_detection_and_states_its_place_for_dew():
    """A scheduler JAX detects joins by JAX's detection of it, named, which derives every fact; its place,
    which Dew's own decisions read, states the pool's size, this process's rank and its accelerators as JAX
    places them: the accelerator at its local rank, or the ones JAX_LOCAL_DEVICE_IDS names. `dew launch`'s
    pool is stated whole, JAX's detection off."""
    assert Slurm(SLURM_TASK).detected() and Slurm(SLURM_TASK).joining() == Joining(detection="slurm")
    assert Slurm(SLURM_TASK).place() == Joining(None, 4, 3, (1,))
    assert Slurm({**SLURM_TASK, "JAX_LOCAL_DEVICE_IDS": "0"}).place() == Joining(None, 4, 3, (0,))
    mpi = {**MPI_RANK, "OMPI_MCA_orte_hnp_uri": "1531576320.0;tcp://10.96.0.1:34911"}
    assert OpenMPI(mpi).joining() == Joining(detection="ompi")
    assert OpenMPI(mpi).place() == Joining(None, 8, 5, (1,))
    launched = Launched.environ_of("host-0:8476", 2, 1, (2, 3))
    assert Launched(launched).joining() == Joining("host-0:8476", 2, 1, (2, 3), detection="deactivate")


def test_open_mpi_is_its_legacy_launchers_or_version_fives_as_jax_reads_it():
    """Open MPI is found where JAX's detection finds it: its legacy launcher's ORTE URI, or Open MPI 5's ranks
    with a major version of 5 or more, or with no version and no ORTE URI; its world size alone, or an
    Open MPI 4's ranks without its URI, is none (review of 4480)."""
    assert OpenMPI({"OMPI_MCA_orte_hnp_uri": "1.0;tcp://h:1"}).detected()
    assert OpenMPI({**MPI_RANK, "OMPI_VERSION": "5.0.3"}).detected()
    assert OpenMPI(MPI_RANK).detected()
    assert not OpenMPI({**MPI_RANK, "OMPI_VERSION": "4.1.6"}).detected()
    assert not OpenMPI({"OMPI_COMM_WORLD_SIZE": "8"}).detected()


def test_a_tpu_joins_by_jaxs_detection_on_a_tpu_vm_alone_and_leaves_its_place_to_jax():
    """A Cloud TPU worker joins by JAX's detection, GKE's by name where the environment lists its slice's
    workers (in JAX's order of sources) and JAX's own order otherwise, TPU_SKIP_MDS_QUERY unset; its place
    is JAX's to read as it joins, so Dew states none. On a host JAX runs as no Cloud TPU VM (no libtpu or
    chip, no JAX_FORCE_TPU_INIT) the same variables are no TPU (review of 4480)."""
    gke = {"TPU_WORKER_HOSTNAMES": "w-0,w-1", "TPU_PROCESS_ADDRESSES": "a-0,a-1,a-2,a-3",
           "TPU_WORKER_ID": "2"}
    forced = CloudTPU({**gke, "JAX_FORCE_TPU_INIT": "1"})
    assert forced.detected() and forced.joining() == Joining(detection="gketpu")
    assert forced.place() == Joining()
    # Any other TPU VM joins in JAX's own order, which reads GCE's metadata and, where it names no worker,
    # goes on to a Kubernetes Job's (review of the TPU's place).
    assert CloudTPU({"JAX_FORCE_TPU_INIT": "1"}).detected()
    assert CloudTPU({"JAX_FORCE_TPU_INIT": "1"}).joining() == Joining()
    assert _tpu_vm() or not CloudTPU(gke).detected()
    # An empty addresses path names no GKE worker list, as JAX reads it (`_get_worker_host_names_env_var`).
    assert not CloudTPU({"TPU_PROCESS_ADDRESSES_PATH": "", "JAX_FORCE_TPU_INIT": "1",
                         "TPU_SKIP_MDS_QUERY": "1"}).detected()


def test_a_kubernetes_job_is_found_where_jax_can_read_its_api_and_joins_in_jaxs_order():
    """A Kubernetes pod is a pool's where the `kubernetes` package lets JAX's detection read the cluster's
    API, whatever its Job's completions (JAX refuses a Job whose are not indexed), and joins in JAX's own
    order, which JAX names no detection of (review of 4480)."""
    import importlib.util

    pod = Kubernetes({"KUBERNETES_SERVICE_HOST": "10.0.0.1"})
    assert pod.detected() == (importlib.util.find_spec("kubernetes") is not None)
    assert pod.joining() == Joining() and pod.place() == Joining()


def _tpu_vm() -> bool:
    """Whether this host is a Cloud TPU VM JAX runs as one (libtpu and a TPU chip)."""
    import importlib.util

    from dew.clusters import tpu_chips
    return importlib.util.find_spec("libtpu") is not None and tpu_chips() > 0


def test_mpirun_inside_a_slurm_allocation_is_open_mpis_pool_and_dew_launchs_is_its_own():
    """A process under several schedulers joins the first in JAX's own order, Open MPI's ranks before the
    Slurm step's tasks `mpirun` runs inside, and a pool `dew launch` declared before either."""
    both = {**SLURM_TASK, **MPI_RANK, "OMPI_MCA_orte_hnp_uri": "1.0;tcp://h:1", "TPU_SKIP_MDS_QUERY": "1"}
    assert detected_environment(environ=both).name == "ompi"
    launched = {**both, **Launched.environ_of("host-0:8476", 2, 0)}
    assert detected_environment(environ=launched).name == "dew launch"
    assert detected_environment(environ={"TPU_SKIP_MDS_QUERY": "1"}) is None


def test_a_pool_joins_through_a_scheduler_of_the_users_own():
    """Two processes a scheduler Dew does not know started, each telling its rank in variables of its own,
    join one pool through `prepare_process(cluster=...)`, its environment a class of the user's."""
    port = coordinator_port()
    program = ("import dataclasses, os\n"
               "from dew.clusters import Joining\n"
               "from dew.training.runtime import prepare_process\n"
               "@dataclasses.dataclass(frozen=True)\n"
               "class Mine:\n"
               "    name = 'mine'\n"
               "    def detected(self):\n"
               "        return 'MY_RANK' in os.environ\n"
               "    def joining(self):\n"
               "        rank = int(os.environ['MY_RANK'])\n"
               "        return Joining(os.environ['MY_COORDINATOR'], 2, rank, detection='deactivate')\n"
               "prepare_process(cluster=Mine())\n"
               "import jax\n"
               "print('pool', jax.process_count(), jax.process_index())\n")
    env = {**outside_any_cluster(os.environ), "PYTHONPATH": str(REPO_ROOT / "src"), "JAX_PLATFORMS": "cpu",
           "MY_COORDINATOR": f"127.0.0.1:{port}"}
    # Each rank's output read as it comes (`owned`), as one read to its end before the other's would leave
    # the other blocked on a full pipe beside a peer waiting on it (row 5434).
    ranks = [owned([sys.executable, "-c", program], {**env, "MY_RANK": str(rank)}) for rank in range(2)]
    outputs = [output(rank, 300) for rank in ranks]
    assert [rank.returncode for rank in ranks] == [0, 0], outputs
    pools = sorted(line for written in outputs for line in written.splitlines() if line.startswith("pool"))
    assert pools == ["pool 2 0", "pool 2 1"], outputs



LISTING = """GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-0a11)
GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-1b22)
GPU 2: NVIDIA H100 80GB HBM3 (UUID: GPU-2c33)
GPU 3: NVIDIA H100 80GB HBM3 (UUID: GPU-3d44)
"""
"""A host's four GPUs as `nvidia-smi -L` lists them."""


def test_dew_launch_shows_each_process_its_own_gpus_alone(monkeypatch):
    """`dew launch`'s processes on a host each see only the GPUs they take (`CUDA_VISIBLE_DEVICES`), numbered
    from 0 for JAX, so a process of one GPU opens no neighbour's whether or not it joins them and keeps
    XLA's CUDA graphs (row 12761: every rank counted the host's eight): the host's own ordinals where the pool
    states none; else its places among the GPUs the pool's list shows as CUDA enumerates it on the host's
    `nvidia-smi -L` listing, by index or UUID, up to an entry naming none of them (row 13104: `0,2,-1,1`
    shows 0 and 2); refused before any process starts where the list shows too few, or the host lists none
    of its GPUs. A pool kept off its GPUs takes its devices by their places. The host answers through
    `subprocess`, whatever machine runs this."""
    import subprocess

    from dew._cli import launch as launching
    from dew._cli.launch import Launch
    from dew.clusters import LOCAL_DEVICES

    def pool(listing: str | None, *env: str) -> list[tuple[str, str]]:
        def run(argv, **_):
            if listing is None or "nvidia-smi -L" not in " ".join(argv):
                raise OSError("no answer")
            return subprocess.CompletedProcess(argv, 0, stdout=listing, stderr="")

        monkeypatch.setattr(subprocess, "run", run)
        launching.listed_on.cache_clear()
        launch = Launch(command=("python", "train.py"), processes_per_host=2, devices_per_process=1,
                        port=8476, env=env)
        return [(process.env.get("CUDA_VISIBLE_DEVICES", ""), process.env[LOCAL_DEVICES])
                for process in launch.pool(["gpu-host.invalid"])]

    assert pool(None, "JAX_PLATFORMS=cuda") == [("0", "0"), ("1", "0")]
    assert pool(LISTING, "JAX_PLATFORMS=cuda", "CUDA_VISIBLE_DEVICES=0,2,-1,1") == [("0", "0"), ("2", "0")]
    by_uuid = pool(LISTING, "JAX_PLATFORMS=cuda", "CUDA_VISIBLE_DEVICES=GPU-3d44,GPU-1b")
    assert by_uuid == [("GPU-3d44", "0"), ("GPU-1b", "0")]
    for listing, visible in ((LISTING, "0,99,1"), (None, "0,1"), (LISTING, "GPU-9f,0")):
        with pytest.raises(ValueError):
            pool(listing, "JAX_PLATFORMS=cuda", f"CUDA_VISIBLE_DEVICES={visible}")
    assert pool(None, "JAX_PLATFORMS=cpu", "CUDA_VISIBLE_DEVICES=") == [("", "0"), ("", "1")]
