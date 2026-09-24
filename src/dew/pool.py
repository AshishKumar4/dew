"""The environment a `dew launch` process pool starts in.

`jax.distributed.initialize` needs the coordinator's address, the process
count and this process's rank. A TPU VM, a Slurm step and an Open MPI launch
leave those in variables jax reads by itself. A plain set of machines leaves
nothing, so `dew.cli.launch` sets the four variables below on every process
and `dew.training.runtime.prepare_process` reads them back. This module is
the contract between the two. It imports only the standard library, so
neither side pulls in the other's dependencies; `detected_cluster` imports
jax's cluster detection when it is asked.
"""

from __future__ import annotations

import dataclasses

COORDINATOR = "JAX_COORDINATOR_ADDRESS"
"""host:port of process 0's coordinator service, which jax reads itself."""
LOCAL_DEVICES = "JAX_LOCAL_DEVICE_IDS"
"""The accelerators a process takes on its host, which jax reads itself."""
PROCESS_COUNT = "DEW_PROCESS_COUNT"
"""How many processes the pool holds; `prepare_process` passes it to jax."""
PROCESS_ID = "DEW_PROCESS_ID"
"""This process's rank in the pool; `prepare_process` passes it to jax."""


@dataclasses.dataclass(frozen=True)
class Cluster:
    """A cluster jax's own detection finds this process running in."""

    name: str
    process: int
    count: int


def detected_cluster() -> Cluster | None:
    """The cluster `jax.distributed.initialize` would take, in jax's own
    order, which reads Open MPI's ranks before a Slurm step's tasks, and
    leaving out the opt-in mpi4py method as it does. The launcher and
    `prepare_process` both ask here, so they cannot disagree on the order."""
    from jax._src import clusters

    for kind in clusters.ClusterEnv._cluster_types:
        if not kind.opt_in_only_method and kind.is_env_present():
            return Cluster(kind.name, kind.get_process_id(), kind.get_process_count())
    return None
