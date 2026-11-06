"""The schedulers a run's processes start under (`dew.clusters`): what each reads of its own variables, which
one a process runs under, and a pool joined through a scheduler of the user's own."""

import os
import socket
import subprocess
import sys
from pathlib import Path

from lane_environment import outside_any_cluster

from dew.clusters import CloudTPU, Joining, Launched, OpenMPI, Slurm, detected_environment

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_TASK = {"SLURM_JOB_ID": "4242", "SLURM_NTASKS": "4", "SLURM_PROCID": "3", "SLURM_LOCALID": "1",
              "SLURM_STEP_NODELIST": "node[1-2]"}
"""What srun sets on the fourth task of a step of four, the second on its node."""
MPI_RANK = {"OMPI_COMM_WORLD_SIZE": "8", "OMPI_COMM_WORLD_RANK": "5", "OMPI_COMM_WORLD_LOCAL_RANK": "1"}
"""What mpirun sets on the sixth of eight ranks, the second on its host."""


def test_each_scheduler_reads_its_own_place_and_leaves_the_rest_to_jax():
    """A Slurm task and an Open MPI rank state their pool's count, their rank and the accelerator they take,
    and leave the coordinator to JAX's detection of the same scheduler; `dew launch`'s pool is stated whole,
    JAX's detection off; a GKE TPU worker states its pool's size and rank alone."""
    assert Slurm(SLURM_TASK).detected() and Slurm(SLURM_TASK).joining() == Joining(None, 4, 3, (1,))
    assert OpenMPI(MPI_RANK).joining() == Joining(None, 8, 5, (1,))
    launched = Launched.environ_of("host-0:8476", 2, 1, (2, 3))
    assert Launched(launched).joining() == Joining("host-0:8476", 2, 1, (2, 3), fills=False)
    tpu = CloudTPU({"TPU_WORKER_HOSTNAMES": "w-0,w-1,w-2,w-3", "TPU_WORKER_ID": "2"})
    assert tpu.detected() and tpu.joining() == Joining(None, 4, 2)
    quiet = {"TPU_SKIP_MDS_QUERY": "1"}
    assert not any(kind(quiet).detected() for kind in (Slurm, OpenMPI, Launched, CloudTPU))


def test_mpirun_inside_a_slurm_allocation_is_open_mpis_pool_and_dew_launchs_is_its_own():
    """A process under several schedulers joins the first in JAX's own order, Open MPI's ranks before the
    Slurm step's tasks `mpirun` runs inside, and a pool `dew launch` declared before either."""
    both = {**SLURM_TASK, **MPI_RANK, "TPU_SKIP_MDS_QUERY": "1"}
    assert detected_environment(environ=both).name == "ompi"
    launched = {**both, **Launched.environ_of("host-0:8476", 2, 0)}
    assert detected_environment(environ=launched).name == "dew launch"
    assert detected_environment(environ={"TPU_SKIP_MDS_QUERY": "1"}) is None


def test_a_pool_joins_through_a_scheduler_of_the_users_own():
    """Two processes a scheduler Dew does not know started, each telling its rank in variables of its own,
    join one pool through `prepare_process(cluster=...)`, its environment a class of the user's."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
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
               "        return Joining(os.environ['MY_COORDINATOR'], 2, rank, fills=False)\n"
               "prepare_process(cluster=Mine())\n"
               "import jax\n"
               "print('pool', jax.process_count(), jax.process_index())\n")
    env = {**outside_any_cluster(os.environ), "PYTHONPATH": str(REPO_ROOT / "src"), "JAX_PLATFORMS": "cpu",
           "MY_COORDINATOR": f"127.0.0.1:{port}"}
    ranks = [subprocess.Popen([sys.executable, "-c", program], cwd=REPO_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env={**env, "MY_RANK": str(rank)})
             for rank in range(2)]
    outputs = [rank.communicate(timeout=300) for rank in ranks]
    assert [rank.returncode for rank in ranks] == [0, 0], outputs
    pools = sorted(line for stdout, _ in outputs for line in stdout.splitlines() if line.startswith("pool"))
    assert pools == ["pool 2 0", "pool 2 1"], outputs
