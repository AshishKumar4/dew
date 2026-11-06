"""The scheduler a run's processes start under, and the place each one takes in the run's process pool.

A run across hosts is one JAX process pool: `jax.distributed.initialize` takes the coordinator's address, the
pool's process count, this process's rank and the accelerators it takes on its host. A scheduler that starts
the processes knows them: Slurm, Open MPI, a Kubernetes indexed Job, a Cloud TPU's runtime, or `dew launch`,
which writes them into each process's environment (`Launched`). A `ClusterEnvironment` says how a process
joins under one scheduler; `prepare_process(cluster=...)` joins the pool it describes, or, given none, the
first of `ENVIRONMENTS` this process runs under (`detected_environment`). A scheduler Dew does not know is a
class of the user's own:

    @dataclass(frozen=True)
    class MyScheduler:
        name = "my scheduler"

        def detected(self) -> bool:
            return "MY_RANK" in os.environ

        def joining(self) -> Joining:
            return Joining(os.environ["MY_COORDINATOR"], int(os.environ["MY_WORLD"]),
                           int(os.environ["MY_RANK"]), detection="deactivate")

    prepare_process(cluster=MyScheduler())

A scheduler JAX detects itself joins by JAX's own detection of it (`Joining.detection`, its
`cluster_detection_method`), which derives every fact JAX's way: a Slurm step's coordinator from its node
list, a multislice TPU's global ranks, the local devices `JAX_LOCAL_DEVICE_IDS` names. Dew reads a
scheduler's presence as JAX's detection does (jax/_src/clusters, jax-ml/jax 19a48d1de), so a process Dew finds
under one JAX finds under it too, and its place where Dew's own decisions need it (`Scheduler.place`: a Slurm
step of one task, the pool `dew launch` runs in place). This module imports only the standard library, so
`dew launch` reads it without JAX.
"""

from __future__ import annotations

import dataclasses
import functools
import glob
import importlib.util
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

COORDINATOR = "JAX_COORDINATOR_ADDRESS"
"""host:port of process 0's coordinator service, which JAX reads itself too."""
LOCAL_DEVICES = "JAX_LOCAL_DEVICE_IDS"
"""The accelerators a process takes on its host, comma-separated, which JAX reads itself too."""
PROCESS_COUNT = "DEW_PROCESS_COUNT"
"""How many processes a pool `dew launch` started holds, which JAX has no variable for."""
PROCESS_ID = "DEW_PROCESS_ID"
"""This process's rank in a pool `dew launch` started."""


@dataclasses.dataclass(frozen=True)
class Joining:
    """What a process passes `jax.distributed.initialize` to join its pool: each fact its environment
    states, None where it states none, and `detection`, JAX's `cluster_detection_method`: a scheduler JAX
    detects by its name ("slurm", "ompi", "gcetpu", "gketpu"), which fills every fact left None, or
    "deactivate" where the environment states the pool whole, so a scheduler around it (a Slurm
    allocation `dew launch` runs in) places the process no further; None is JAX's own order of detection."""

    coordinator_address: str | None = None
    process_count: int | None = None
    process_id: int | None = None
    local_device_ids: tuple[int, ...] | None = None
    detection: str | None = None


@runtime_checkable
class ClusterEnvironment(Protocol):
    """A scheduler that starts a run's processes and tells each its place: `detected` where this process
    runs under it, `joining` how it joins the pool."""

    name: str

    def detected(self) -> bool: ...

    def joining(self) -> Joining: ...


@runtime_checkable
class Scheduler(ClusterEnvironment, Protocol):
    """A scheduler Dew detects itself (`SCHEDULERS`), whose `place` states the pool's size, this process's
    rank and its accelerators where the scheduler tells them without JAX: what Dew's own decisions read,
    `dew launch`'s running in place among them, none of which reaches JAX."""

    def place(self) -> Joining: ...


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _devices(listed: str | None) -> tuple[int, ...] | None:
    return None if not listed else tuple(int(device) for device in listed.split(","))


@dataclasses.dataclass(frozen=True)
class Launched:
    """A pool `dew launch` started: it writes each process's place into its environment (`environ_of`),
    the whole of it, so JAX detects nothing more."""

    environ: Mapping[str, str] | None = None
    name = "dew launch"

    def detected(self) -> bool:
        return PROCESS_COUNT in _environ(self.environ)

    def joining(self) -> Joining:
        environ = _environ(self.environ)
        return Joining(environ.get(COORDINATOR), int(environ[PROCESS_COUNT]), int(environ[PROCESS_ID]),
                       _devices(environ.get(LOCAL_DEVICES)), detection="deactivate")

    def place(self) -> Joining:
        return self.joining()

    @staticmethod
    def environ_of(coordinator_address: str, process_count: int, process_id: int | None = None,
                   local_device_ids: Sequence[int] | None = None) -> dict[str, str]:
        """The variables `dew launch` writes into a process for `Launched` to read: the pool's, and this
        process's rank and accelerators where given."""
        return {COORDINATOR: coordinator_address, PROCESS_COUNT: str(process_count),
                **({} if process_id is None else {PROCESS_ID: str(process_id)}),
                **({} if local_device_ids is None
                   else {LOCAL_DEVICES: ",".join(str(device) for device in local_device_ids)})}


def _ranked(environ: Mapping[str, str], count: str, rank: str, local: str) -> Joining:
    """A process a scheduler ranked in `environ`: its pool's size, its rank and the accelerator at its rank
    on its host, or the ones `JAX_LOCAL_DEVICE_IDS` names, as JAX places it."""
    return Joining(None, int(environ[count]), int(environ[rank]),
                   _devices(environ.get(LOCAL_DEVICES)) or (int(environ[local]),))


_OMPI_RANKS = ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_LOCAL_RANK")


@dataclasses.dataclass(frozen=True)
class OpenMPI:
    """Processes `mpirun` started: Open MPI's legacy launcher, which states its ORTE URI, or Open MPI 5's,
    which states its ranks and a major version of 5 or more, or, stating no version, no ORTE URI
    (`OmpiCluster.is_env_present`, `_is_ompi_5`); the coordinator JAX's detection reads off the launcher."""

    environ: Mapping[str, str] | None = None
    name = "ompi"

    def detected(self) -> bool:
        environ = _environ(self.environ)
        if "OMPI_MCA_orte_hnp_uri" in environ:
            return True
        if not all(variable in environ for variable in _OMPI_RANKS):
            return False
        major = environ.get("OMPI_VERSION", "").split(".", maxsplit=1)[0]
        return int(major) >= 5 if major.isdigit() else "OMPI_MCA_orte_hnp_uri" not in environ

    def joining(self) -> Joining:
        return Joining(detection=self.name)

    def place(self) -> Joining:
        environ = _environ(self.environ)
        if not all(variable in environ for variable in _OMPI_RANKS):
            return Joining()
        return _ranked(environ, *_OMPI_RANKS)


_SLURM = ("SLURM_JOB_ID", "SLURM_STEP_NODELIST", "SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID")
"""The variables a Slurm step sets on each task, every one of which JAX's detection reads (`SlurmCluster`)."""


@dataclasses.dataclass(frozen=True)
class Slurm:
    """The tasks of a Slurm step: SLURM_NTASKS of them, this one SLURM_PROCID, taking the accelerator at
    SLURM_LOCALID, as JAX places them; the coordinator JAX's detection names after the step's first node."""

    environ: Mapping[str, str] | None = None
    name = "slurm"

    def detected(self) -> bool:
        environ = _environ(self.environ)
        return all(variable in environ for variable in _SLURM)

    def joining(self) -> Joining:
        return Joining(detection=self.name)

    def place(self) -> Joining:
        return _ranked(_environ(self.environ), "SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID")


_GOOGLE_PCI_VENDOR = "0x1ae0"
_TPU_PCI_DEVICES = frozenset({"0x0027", "0x0056", "0x005e", "0x0062", "0x0063", "0x006f", "0x0076", "0x0083",
                              "0x007c"})
"""The PCI device ids of Google's TPU chips, v3 through TPU8, as JAX's hardware detection lists them
(jax/_src/hardware_utils.py `_TPU_PCI_DEVICE_IDS`, Apache-2.0, jax-ml/jax 19a48d1de)."""


def _tpu_chips(devices: str = "/sys/bus/pci/devices") -> int:
    """How many TPU chips this host's PCI bus holds."""
    found = 0
    for vendor in glob.glob(f"{devices}/*/vendor"):
        device = Path(vendor).with_name("device")
        if Path(vendor).read_text().strip() == _GOOGLE_PCI_VENDOR and device.exists():
            found += device.read_text().strip() in _TPU_PCI_DEVICES
    return found


def _in_cloud_tpu_vm(environ: Mapping[str, str]) -> bool:
    """Whether JAX runs this process as a Cloud TPU VM's: libtpu installed (TPU_LIBRARY_PATH an existing
    file, or the `libtpu` package) and a TPU chip on the bus, or JAX_FORCE_TPU_INIT set
    (`cloud_tpu_init`, jax/_src/cloud_tpu_init.py, jax-ml/jax 19a48d1de)."""
    if "JAX_FORCE_TPU_INIT" in environ:
        return True
    library = environ.get("TPU_LIBRARY_PATH")
    installed = library is not None and os.path.isfile(library)
    return (installed or importlib.util.find_spec("libtpu") is not None) and _tpu_chips() > 0


def _gke_workers(environ: Mapping[str, str]) -> str | None:
    """A GKE TPU slice's workers, comma-separated, in JAX's order of sources: the file
    TPU_PROCESS_ADDRESSES_PATH names, then TPU_PROCESS_ADDRESSES, then TPU_WORKER_HOSTNAMES
    (`GkeTpuCluster._get_worker_host_names_env_var`)."""
    path = environ.get("TPU_PROCESS_ADDRESSES_PATH")
    if path:
        return Path(path).read_text().strip()
    return environ.get("TPU_PROCESS_ADDRESSES", environ.get("TPU_WORKER_HOSTNAMES"))


@dataclasses.dataclass(frozen=True)
class CloudTPU:
    """A Cloud TPU's workers, which JAX's detection places whole, multislice ranks among them: on GKE a TPU
    VM whose worker list the environment states (`GkeTpuCluster`), else on GCE one whose metadata server JAX
    asks, TPU_SKIP_MDS_QUERY unset (`GceTpuCluster`). Its place is JAX's to read as it joins (its metadata,
    its slices): Dew decides nothing by it before, so it states none, and `dew launch` runs every worker's
    program in place."""

    environ: Mapping[str, str] | None = None
    name = "tpu"

    @functools.cached_property
    def _kind(self) -> str | None:
        """JAX's detection of it: "gketpu", "gcetpu", or None where it runs on neither."""
        environ = _environ(self.environ)
        if not _in_cloud_tpu_vm(environ):
            return None
        if _gke_workers(environ) is not None:
            return "gketpu"
        return None if "TPU_SKIP_MDS_QUERY" in environ else "gcetpu"

    def detected(self) -> bool:
        return self._kind is not None

    def joining(self) -> Joining:
        return Joining(detection=self._kind)

    def place(self) -> Joining:
        return Joining()


@dataclasses.dataclass(frozen=True)
class Kubernetes:
    """The pods of a Kubernetes Job, where the `kubernetes` package is installed for JAX's detection to read
    the cluster's API (`K8sCluster`), which refuses a Job whose completions are not indexed."""

    environ: Mapping[str, str] | None = None
    name = "kubernetes"

    def detected(self) -> bool:
        return ("KUBERNETES_SERVICE_HOST" in _environ(self.environ)
                and importlib.util.find_spec("kubernetes") is not None)

    def joining(self) -> Joining:
        # JAX names no detection of it, so it is found in JAX's own order, as here.
        return Joining()

    def place(self) -> Joining:
        index = _environ(self.environ).get("JOB_COMPLETION_INDEX")
        return Joining(process_id=None if index is None else int(index))


type Reader = Callable[[Mapping[str, str] | None], Scheduler]
"""A scheduler's class, which reads a process's variables (the process's own given None)."""

SCHEDULERS: tuple[Reader, ...] = (OpenMPI, Slurm, CloudTPU, Kubernetes)
"""The schedulers Dew reads, in JAX's own detection order (jax/_src/clusters/__init__.py): Open MPI's ranks
before a Slurm step's tasks, which `mpirun` inside a Slurm allocation sets both of."""

ENVIRONMENTS: tuple[Reader, ...] = (Launched, *SCHEDULERS)
"""Every environment `prepare_process` detects, `dew launch`'s first: a pool it declared is placed whole."""


def detected_environment(environments: Sequence[Reader] = ENVIRONMENTS,
                         environ: Mapping[str, str] | None = None) -> Scheduler | None:
    """The first of `environments` this process runs under, reading `environ` (the process's own where
    None), or None."""
    for kind in environments:
        environment = kind(environ)
        if environment.detected():
            return environment
    return None


__all__ = ["ENVIRONMENTS", "SCHEDULERS", "CloudTPU", "ClusterEnvironment", "Joining", "Kubernetes",
           "Launched", "OpenMPI", "Scheduler", "Slurm", "detected_environment"]
