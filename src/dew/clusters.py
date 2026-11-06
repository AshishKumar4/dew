"""The scheduler a run's processes start under, and the place each one takes in the run's process pool.

A run across hosts is one JAX process pool: `jax.distributed.initialize` takes the coordinator's address, the
pool's process count, this process's rank and the accelerators it takes on its host. A scheduler that starts
the processes knows them: Slurm, Open MPI, a Kubernetes indexed Job, a Cloud TPU's runtime, or `dew launch`,
which writes them into each process's environment (`Launched`). A `ClusterEnvironment` reads them for one
scheduler; `prepare_process(cluster=...)` joins the pool it describes, or, given none, the first of
`ENVIRONMENTS` this process runs under (`detected_environment`). A scheduler Dew does not know is a class of
the user's own:

    @dataclass(frozen=True)
    class MyScheduler:
        name = "my scheduler"

        def detected(self) -> bool:
            return "MY_RANK" in os.environ

        def joining(self) -> Joining:
            return Joining(os.environ["MY_COORDINATOR"], int(os.environ["MY_WORLD"]),
                           int(os.environ["MY_RANK"]), fills=False)

    prepare_process(cluster=MyScheduler())

An environment states what it reads and leaves the rest to JAX's own detection of the same scheduler
(`Joining.fills`), which finds a Slurm step's or an Open MPI run's coordinator and a Cloud TPU's whole place
from its metadata. Detection reads variables and the PCI bus alone, no network and no client. This module
imports only the standard library, so `dew launch` reads it without JAX.
"""

from __future__ import annotations

import dataclasses
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
    reads, None where it reads none. Where `fills`, JAX's own detection of the scheduler fills the facts
    left None (a Slurm step's coordinator, from its node list); otherwise the environment states the pool
    whole and JAX detects nothing, so a scheduler around it (a Slurm allocation `dew launch` runs in) does
    not place the process again."""

    coordinator_address: str | None = None
    process_count: int | None = None
    process_id: int | None = None
    local_device_ids: tuple[int, ...] | None = None
    fills: bool = True

    def arguments(self) -> dict[str, object]:
        """`jax.distributed.initialize`'s arguments: the facts it holds, and JAX's detection turned off
        where it fills none (`fills`)."""
        stated = {"coordinator_address": self.coordinator_address, "num_processes": self.process_count,
                  "process_id": self.process_id,
                  "local_device_ids": None if self.local_device_ids is None else list(self.local_device_ids)}
        return {**{name: value for name, value in stated.items() if value is not None},
                **({} if self.fills else {"cluster_detection_method": "deactivate"})}


@runtime_checkable
class ClusterEnvironment(Protocol):
    """A scheduler that starts a run's processes and tells each its place: `detected` where this process
    runs under it, `joining` its place in the pool."""

    name: str

    def detected(self) -> bool: ...

    def joining(self) -> Joining: ...


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
                       _devices(environ.get(LOCAL_DEVICES)), fills=False)

    @staticmethod
    def environ_of(coordinator_address: str, process_count: int, process_id: int | None = None,
                   local_device_ids: Sequence[int] | None = None) -> dict[str, str]:
        """The variables `dew launch` writes into a process for `Launched` to read: the pool's, and this
        process's rank and accelerators where given."""
        return {COORDINATOR: coordinator_address, PROCESS_COUNT: str(process_count),
                **({} if process_id is None else {PROCESS_ID: str(process_id)}),
                **({} if local_device_ids is None
                   else {LOCAL_DEVICES: ",".join(str(device) for device in local_device_ids)})}


@dataclasses.dataclass(frozen=True)
class OpenMPI:
    """Processes `mpirun` started: Open MPI's ranks (OMPI_COMM_WORLD_SIZE, _RANK, _LOCAL_RANK), the
    coordinator JAX's detection reads off the launcher (`OmpiCluster`)."""

    environ: Mapping[str, str] | None = None
    name = "ompi"

    def detected(self) -> bool:
        return "OMPI_COMM_WORLD_SIZE" in _environ(self.environ)

    def joining(self) -> Joining:
        environ = _environ(self.environ)
        return Joining(None, int(environ["OMPI_COMM_WORLD_SIZE"]), int(environ["OMPI_COMM_WORLD_RANK"]),
                       (int(environ["OMPI_COMM_WORLD_LOCAL_RANK"]),))


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
        environ = _environ(self.environ)
        return Joining(None, int(environ["SLURM_NTASKS"]), int(environ["SLURM_PROCID"]),
                       (int(environ["SLURM_LOCALID"]),))


_GOOGLE_PCI_VENDOR = "0x1ae0"
_TPU_PCI_DEVICES = frozenset({"0x0027", "0x0056", "0x005e", "0x0062", "0x0063", "0x006f", "0x0076", "0x0083",
                              "0x007c"})
"""The PCI device ids of Google's TPU chips, v3 through TPU8, as JAX's hardware detection lists them
(jax/_src/hardware_utils.py `_TPU_PCI_DEVICE_IDS`, Apache-2.0, jax-ml/jax 19a48d1de): a Cloud TPU VM's
chips, which its metadata server describes (`CloudTPU`)."""


def _tpu_chips(devices: str = "/sys/bus/pci/devices") -> int:
    """How many TPU chips this host's PCI bus holds."""
    found = 0
    for vendor in glob.glob(f"{devices}/*/vendor"):
        device = Path(vendor).with_name("device")
        if Path(vendor).read_text().strip() == _GOOGLE_PCI_VENDOR and device.exists():
            found += device.read_text().strip() in _TPU_PCI_DEVICES
    return found


@dataclasses.dataclass(frozen=True)
class CloudTPU:
    """A Cloud TPU's workers: on GKE the worker list in the environment (TPU_WORKER_HOSTNAMES,
    TPU_PROCESS_ADDRESSES), across slices Megascale's coordinator, and on a TPU VM its chips, whose
    metadata server JAX's detection reads every fact from (`GceTpuCluster`, `GkeTpuCluster`).
    TPU_SKIP_MDS_QUERY, which says the host is no GCE TPU, leaves a TPU VM undetected, as JAX does."""

    environ: Mapping[str, str] | None = None
    name = "tpu"

    def detected(self) -> bool:
        environ = _environ(self.environ)
        listed = ("TPU_WORKER_HOSTNAMES", "TPU_PROCESS_ADDRESSES", "TPU_PROCESS_ADDRESSES_PATH",
                  "MEGASCALE_COORDINATOR_ADDRESS")
        return (any(variable in environ for variable in listed)
                or ("TPU_SKIP_MDS_QUERY" not in environ and _tpu_chips() > 0))

    def joining(self) -> Joining:
        environ = _environ(self.environ)
        hosts = environ.get("TPU_WORKER_HOSTNAMES")
        worker = environ.get("TPU_WORKER_ID")
        return Joining(None, None if not hosts else len(hosts.split(",")),
                       None if worker is None else int(worker))


@dataclasses.dataclass(frozen=True)
class Kubernetes:
    """The pods of a Kubernetes indexed Job: this one's index (JOB_COMPLETION_INDEX), the rest JAX's
    detection reads from the cluster's API through the `kubernetes` package (`K8sCluster`), which it
    needs."""

    environ: Mapping[str, str] | None = None
    name = "kubernetes"

    def detected(self) -> bool:
        environ = _environ(self.environ)
        return ("KUBERNETES_SERVICE_HOST" in environ and "JOB_COMPLETION_INDEX" in environ
                and importlib.util.find_spec("kubernetes") is not None)

    def joining(self) -> Joining:
        return Joining(None, None, int(_environ(self.environ)["JOB_COMPLETION_INDEX"]))


type Reader = Callable[[Mapping[str, str] | None], ClusterEnvironment]
"""An environment's class, which reads a process's variables (the process's own given None)."""

SCHEDULERS: tuple[Reader, ...] = (OpenMPI, Slurm, CloudTPU, Kubernetes)
"""The schedulers Dew reads, in JAX's own detection order: Open MPI's ranks before a Slurm step's tasks,
which `mpirun` inside a Slurm allocation sets both of."""

ENVIRONMENTS: tuple[Reader, ...] = (Launched, *SCHEDULERS)
"""Every environment `prepare_process` detects, `dew launch`'s first: a pool it declared is placed whole."""


def detected_environment(environments: Sequence[Reader] = ENVIRONMENTS,
                         environ: Mapping[str, str] | None = None) -> ClusterEnvironment | None:
    """The first of `environments` this process runs under, reading `environ` (the process's own where
    None), or None."""
    for kind in environments:
        environment = kind(environ)
        if environment.detected():
            return environment
    return None


__all__ = ["ENVIRONMENTS", "SCHEDULERS", "CloudTPU", "ClusterEnvironment", "Joining", "Kubernetes",
           "Launched", "OpenMPI", "Slurm", "detected_environment"]
