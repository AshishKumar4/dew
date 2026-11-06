"""The schedulers a run's processes start under (`dew.clusters`): what each reads of its own variables, which
one a process runs under, and a pool joined through a scheduler of the user's own."""

import os
import socket
import subprocess
import sys
from pathlib import Path

from lane_environment import outside_any_cluster

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


def test_a_tpus_place_is_its_slices_workers_and_its_global_rank_on_a_tpu_vm_alone():
    """A GKE TPU worker's place is its slice's workers times its slices, its rank its slice's offset plus its
    rank in the slice, its workers read in JAX's order of sources (TPU_PROCESS_ADDRESSES before
    TPU_WORKER_HOSTNAMES); on a host JAX runs as no Cloud TPU VM (no libtpu or chip, no
    JAX_FORCE_TPU_INIT) the same variables are no TPU (review of 4480)."""
    gke = {"TPU_WORKER_HOSTNAMES": "w-0,w-1", "TPU_PROCESS_ADDRESSES": "a-0,a-1,a-2,a-3",
           "TPU_WORKER_ID": "2", "MEGASCALE_NUM_SLICES": "2", "MEGASCALE_SLICE_ID": "1"}
    forced = CloudTPU({**gke, "JAX_FORCE_TPU_INIT": "1"})
    assert forced.detected() and forced.joining() == Joining(detection="gketpu")
    assert forced.place() == Joining(None, 8, 6)
    assert _tpu_vm() or not CloudTPU(gke).detected()
    # An empty addresses path names no GKE worker list, as JAX reads it (`_get_worker_host_names_env_var`).
    assert not CloudTPU({"TPU_PROCESS_ADDRESSES_PATH": "", "JAX_FORCE_TPU_INIT": "1",
                         "TPU_SKIP_MDS_QUERY": "1"}).detected()


def test_a_gke_tpus_slices_read_through_its_metadata_where_its_environment_names_none():
    """A GKE TPU slice of one worker whose multislice topology its metadata states, not its environment, is
    one process of a pool across the slices, as JAX reads MEGASCALE_NUM_SLICES and MEGASCALE_SLICE_ID
    through the metadata's `tpu-env` (`get_tpu_env_value`), so `dew launch` runs it in place rather than as a
    pool of its own (review of 4480)."""
    import http.server
    import threading

    class Metadata(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"MEGASCALE_NUM_SLICES: '2'\nMEGASCALE_SLICE_ID: '1'\n"
            found = self.path.endswith("/tpu-env") and self.headers.get("Metadata-Flavor") == "Google"
            self.send_response(200 if found else 404)
            self.end_headers()
            self.wfile.write(body if found else b"")

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Metadata)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        worker = CloudTPU({"TPU_WORKER_HOSTNAMES": "w-0", "TPU_WORKER_ID": "0", "JAX_FORCE_TPU_INIT": "1",
                           "GCE_METADATA_IP": f"127.0.0.1:{server.server_address[1]}"})
        assert worker.place() == Joining(None, 2, 1)
    finally:
        server.shutdown()


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

    from dew.clusters import _tpu_chips
    return importlib.util.find_spec("libtpu") is not None and _tpu_chips() > 0


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
               "        return Joining(os.environ['MY_COORDINATOR'], 2, rank, detection='deactivate')\n"
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
