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
import logging
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence

_log = logging.getLogger(__name__)

COORDINATOR = "JAX_COORDINATOR_ADDRESS"
"""host:port of process 0's coordinator service, which jax reads itself."""
LOCAL_DEVICES = "JAX_LOCAL_DEVICE_IDS"
"""The accelerators a process takes on its host, which jax reads itself."""
PROCESS_COUNT = "DEW_PROCESS_COUNT"
"""How many processes the pool holds; `prepare_process` passes it to jax."""
PROCESS_ID = "DEW_PROCESS_ID"
"""This process's rank in the pool; `prepare_process` passes it to jax."""
PREEMPTED_EXIT = 128 + signal.SIGTERM
"""The exit status of a run stopped at a preemption, SIGTERM's as a shell
reports it: a scheduler reads a stopped job rather than a finished one, and
Kubernetes' pod failure policy can ignore it by this code. `Trainer.fit`
ends with it, and `dew launch` reads a rank's as that rank's preemption."""


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



def runs_on_gpu(env: Mapping[str, str]) -> bool:
    """Whether JAX_PLATFORMS in `env` leaves jax the GPUs; unset lets it pick them."""
    platforms = env.get("JAX_PLATFORMS", "")
    return not platforms or any(name in platforms for name in ("cuda", "gpu"))


def listed_gpus(argv: Sequence[str]) -> int | None:
    """GPUs in the `nvidia-smi -L` output of `argv`, whose MIG lines are
    indented; None when they could not be counted: nvidia-smi missing, or
    failing, or not answering within a minute, as over ssh a host whose
    non-interactive PATH lacks it, or that cannot be reached, gives. None is
    not zero GPUs, and every check that reads a count skips on it."""
    try:
        found = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as failure:
        _log.debug("%s could not list GPUs: %s", " ".join(argv), failure)
        return None
    if found.returncode != 0:
        _log.debug("%s exited %d: %s", " ".join(argv), found.returncode, found.stderr.strip()[-300:])
        return None
    return sum(line.startswith("GPU ") for line in found.stdout.splitlines())


def visible_gpus(visible: str) -> int:
    """GPUs a CUDA_VISIBLE_DEVICES value lists."""
    return sum(1 for device in visible.split(",") if device.strip())


def local_gpu_count() -> int | None:
    """GPUs this process may use: CUDA_VISIBLE_DEVICES when it is set,
    since jax numbers only those, else every GPU nvidia-smi lists, None when
    it cannot list them."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return listed_gpus(("nvidia-smi", "-L")) if visible is None else visible_gpus(visible)


def slurm_tasks_here() -> int | None:
    """Tasks the Slurm step or allocation around this process runs on its
    node, from the per-node counts Slurm writes as `4` or `2(x3),1`; None
    where Slurm wrote none."""
    counts = os.environ.get("SLURM_STEP_TASKS_PER_NODE") or os.environ.get("SLURM_TASKS_PER_NODE")
    if counts is None:
        return None
    nodes: list[int] = []
    for entry in counts.split(","):
        count, _, repeat = entry.partition("(x")
        nodes += [int(count)] * (int(repeat.rstrip(")")) if repeat else 1)
    return nodes[int(os.environ.get("SLURM_NODEID", "0"))]


def refuse_idle_gpus(tasks: int, gpus: int | None, where: str) -> None:
    """Refuse a Slurm placement of `tasks` tasks on a node of `gpus` GPUs
    that leaves some idle: jax gives each Slurm task the one GPU at its
    SLURM_LOCALID, so a node running fewer tasks than it has GPUs trains on
    that many, and nothing reports the rest. A node whose GPUs could not be
    counted (None) is not refused."""
    if gpus is not None and 0 < tasks < gpus:
        raise ValueError(
            f"{where} runs {tasks} task{'s' if tasks > 1 else ''} on a node of {gpus} GPUs, and jax "
            f"gives each Slurm task only the GPU at its SLURM_LOCALID, so {gpus - tasks} would sit "
            f"idle; allocate with --ntasks-per-node={gpus}, or start it with "
            f"dew launch --processes-per-host {gpus}")
