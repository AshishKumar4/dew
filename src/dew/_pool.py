"""The machine a process pool runs on: its GPUs, as the driver counts them, a Slurm step's tasks on this
node, and whether a process joins a pool at all (`joins_pool`), from the scheduler it runs under
(`dew.clusters`). It imports only the standard library, so `dew launch` reads it without JAX.
"""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence

from dew.clusters import COORDINATOR, ClusterEnvironment

_log = logging.getLogger(__name__)

PREEMPTED_EXIT = 128 + signal.SIGTERM
"""The exit status of a run stopped at a preemption, SIGTERM's as a shell
reports it: a scheduler reads a stopped job rather than a finished one, and
Kubernetes' pod failure policy can ignore it by this code. `Trainer.fit`
ends with it, and `dew launch` reads a rank's as that rank's preemption."""


def joins_pool(multi_host: bool | None, environment: ClusterEnvironment | None) -> bool:
    """Whether a process joins a pool, read before a backend opens: the run asks for one
    (multi_host=True), or the environment declares one (a scheduler's `environment`, `dew launch`'s among
    them, or a coordinator). A Slurm step of one task forms none unless the run asks, nor any run asking
    for none."""
    one_task = (multi_host is None and environment is not None and environment.name == "slurm"
                and environment.joining().process_count == 1)
    if multi_host is False or one_task:
        return False
    return multi_host is True or environment is not None or COORDINATOR in os.environ


def runs_on_gpu(env: Mapping[str, str]) -> bool:
    """Whether `env` leaves jax the GPUs: JAX_PLATFORMS unset or naming them,
    and CUDA_VISIBLE_DEVICES, where set, naming some (empty or -1 hides all)."""
    platforms = env.get("JAX_PLATFORMS", "")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    hidden = visible is not None and visible.strip() in ("", "-1")
    return not hidden and (not platforms or any(name in platforms for name in ("cuda", "gpu")))


def answered(argv: Sequence[str], timeout: float = 60) -> str | None:
    """What `argv` (an nvidia-smi query, or one over ssh) printed, or None
    where it did not answer: missing, failing, or not done within `timeout`
    seconds, as over ssh a host whose non-interactive PATH lacks it, or that
    cannot be reached, gives. Each failure is logged at debug level."""
    try:
        found = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as failure:
        _log.debug("%s did not answer: %s", " ".join(argv), failure)
        return None
    if found.returncode != 0:
        _log.debug("%s exited %d: %s", " ".join(argv), found.returncode, found.stderr.strip()[-300:])
        return None
    return found.stdout


def listed_gpus(argv: Sequence[str]) -> int | None:
    """GPUs in the `nvidia-smi -L` output of `argv`, whose MIG lines are
    indented; None when they could not be counted (`answered`). None is not
    zero GPUs, and every check that reads a count skips on it."""
    listed = answered(argv)
    return None if listed is None else sum(line.startswith("GPU ") for line in listed.splitlines())


def visible_gpus(visible: str) -> int:
    """GPUs a CUDA_VISIBLE_DEVICES value lists."""
    return sum(1 for device in visible.split(",") if device.strip())


def cuda_driver() -> ctypes.CDLL | None:
    """The CUDA user-mode driver library JAX's CUDA plugin loads, or None where none loads."""
    try:
        return ctypes.CDLL("libcuda.so.1")
    except OSError as error:  # ctypes' answer for a library the loader does not find
        _log.debug("no CUDA driver library loads: %s", error)
        return None


NO_DEVICE = 100
"""`CUDA_ERROR_NO_DEVICE` (cuda.h): a driver that starts and finds no GPU this process may use."""


ALL_GPUS = "all"
"""JAX's selection of no particular GPUs (`jax_cuda_visible_devices`'s default): it opens every one the
process may use."""


def selected_gpus(selected: str) -> tuple[int, ...] | None:
    """The GPUs JAX's own selection names (`jax_cuda_visible_devices`, set in the process: a comma list of
    the driver's ordinals, as JAX reads it), or None where it selects `ALL_GPUS`."""
    return None if selected == ALL_GPUS else tuple(int(ordinal) for ordinal in selected.split(","))


def local_gpus(selected: str = ALL_GPUS) -> tuple[int, ...] | None:
    """The GPUs a process of JAX's selection `selected` opens, by the CUDA driver's ordinals, which its
    memory and placement reads take (`dew._devices.gpu_memory`): those `selected` names, else every one the
    process may use (`local_gpu_count`); None where they could not be counted."""
    named = selected_gpus(selected)
    if named is not None:
        return named
    count = local_gpu_count()
    return None if count is None else tuple(range(count))


def local_gpu_count() -> int | None:
    """GPUs this process may use: CUDA_VISIBLE_DEVICES when it is set,
    since jax numbers only those, else every GPU the CUDA driver finds
    (`cuDeviceGetCount`), whether or not the host ships nvidia-smi; None
    where no driver library loads or it fails to start. JAX opens those its
    own selection names of them (`local_gpus`)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return visible_gpus(visible)
    cuda = cuda_driver()
    if cuda is None:
        return None
    started, count = cuda.cuInit(0), ctypes.c_int()
    if started == NO_DEVICE:
        return 0
    return None if started or cuda.cuDeviceGetCount(ctypes.byref(count)) else count.value


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
