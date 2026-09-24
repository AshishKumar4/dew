"""Process setup every recipe runs before it builds anything.

rlimits, the XLA flags, the compilation cache, the JAX distributed pool and
the env vars wandb and the tokenizers read are the same in every recipe.
That makes them library wiring, and the recipes call this once at the top of
main().
"""

from __future__ import annotations

import importlib.util
import os
import resource
from datetime import datetime
from typing import TYPE_CHECKING

import jax
from jax._src.distributed import global_state
from jax.experimental import multihost_utils

from dew.artifacts import broadcast_from_process_zero, end_pool_on_failure
from dew.pool import (
    PROCESS_COUNT,
    PROCESS_ID,
    detected_cluster,
    local_gpu_count,
    refuse_idle_gpus,
    runs_on_gpu,
    slurm_tasks_here,
)
from dew.telemetry.devices import apply_xla_flags, xla_flag
from dew.telemetry.instrumentation import enable_compilation_cache

if TYPE_CHECKING:
    from dew.config import Wandb
    from dew.training.distributed import Layout


EXECUTION_TIMEOUT = "30m"
"""How long one device execution of a process pool may run before XLA ends
its process.

A rank that stalls without failing, blocked on a read or on a compile that
waits for peers, leaves the other ranks inside a collective or a
communicator's setup. No GPU backend times that out and no process reports it,
so the pool would hang for ever with every process alive. XLA's execution
watchdog ends a process whose execution runs past this, and `dew launch`,
srun or the scheduler then stops the rest. An execution is a whole step or
sampling loop, which can run for minutes, so the bound is generous;
`--xla_gpu_execution_terminate_timeout` in XLA_FLAGS or `xla_flags` sets
another."""


def prepare_process(wandb: Wandb | None = None,
                    multi_host: bool | None = None,
                    xla_flags: str | None = None,
                    compilation_cache_dir: str | None = None,
                    *, layout: Layout | None = None) -> None:
    """Raise the fd/core limits, set the env vars, join the JAX process pool.

    `wandb` is the run's `dew.config.Wandb`, or None for a run without a
    tracker. Only its offline switch is read, and it has to be read before
    wandb opens a run.

    jax.distributed.initialize() finds the coordinator from the environment on
    TPU pods and Slurm/GKE/Open MPI clusters. `dew launch` leaves the process
    count and rank in DEW_PROCESS_COUNT and DEW_PROCESS_ID, which jax has no
    variable for, and those are passed to it with JAX's cluster detection
    off: the launcher placed its processes, and a Slurm step around it would
    otherwise pin each to the GPU at SLURM_LOCALID. On a machine with no cluster
    environment it raises a ValueError naming the missing coordinator
    address, the single-host signature. Every other failure propagates,
    since a pod run would otherwise continue on one host. multi_host=True
    requires the pool, multi_host=False never asks for it. A Slurm step of
    one task forms no pool unless the run asks for one with multi_host=True
    or mpirun started its ranks there, which JAX's detection reads before
    Slurm's: JAX would still start a pool of that one task, at a coordinator
    named after the node, which a container on the node need not resolve.
    A Slurm step of several tasks with fewer on this node than the GPUs its
    task sees is refused: JAX gives each task the GPU at its SLURM_LOCALID,
    and the others would sit idle.

    xla_flags reaches XLA through the environment, which XLA reads when it
    opens a backend. So this call has to come before the first JAX call in
    the process, which makes it a recipe's first line. A library user, who
    never runs a recipe, sets XLA_FLAGS in the environment.

    The same Layout passed to Trainer selects CPU transaction ownership when
    host includes params. JAX_PLATFORMS must then permit CPU beside the
    accelerator. JAX_NUM_CPU_DEVICES, or the existing XLA flags, must
    establish one CPU device per local accelerator before this call.
    Validation never changes backend configuration after initialization.

    A GPU pool keeps the persistent compilation cache when its jax keys a
    computation that spans processes alike on every one of them, as the jax
    Dew pins does (`_pool_keys_alike`). With another jax it compiles without
    the cache: some ranks would load a step that the others compile, and that
    compile waits for every rank for ever.
    """
    if wandb is not None and wandb.offline:
        os.environ['WANDB_MODE'] = 'offline'
    # HF tokenizers fork a thread pool; grain's workers fork the process.
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    apply_xla_flags(xla_flags)
    if compilation_cache_dir:
        enable_compilation_cache(compilation_cache_dir)

    resource.setrlimit(
        resource.RLIMIT_CORE,
        (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))

    # The cluster JAX's detection would take, in its own order: Open MPI's
    # ranks when mpirun started them, then Slurm's tasks.
    cluster = None if multi_host is False or PROCESS_COUNT in os.environ else detected_cluster()
    one_task = (multi_host is None and cluster is not None and cluster.name == "slurm"
                and cluster.count == 1)
    tasks = slurm_tasks_here()
    if (cluster is not None and cluster.name == "slurm" and cluster.count > 1 and tasks is not None
            and runs_on_gpu(os.environ)):
        # Before joining: a pool whose ranks see GPUs they will not use
        # trains on fewer than it was given, and nothing reports it.
        refuse_idle_gpus(tasks, local_gpu_count(), "this slurm step")
    if multi_host is not False and not one_task:
        try:
            if PROCESS_COUNT in os.environ:
                jax.distributed.initialize(num_processes=int(os.environ[PROCESS_COUNT]),
                                           process_id=int(os.environ[PROCESS_ID]),
                                           cluster_detection_method="deactivate")
            else:
                jax.distributed.initialize()
        except ValueError as e:
            if multi_host or "coordinator_address" not in str(e):
                raise
        else:
            # Before the backend opens, which can fail on one process of a
            # pool that has formed, a GPU with no memory left for one. The
            # watch leaves through os._exit, past the atexit handlers, which
            # only a peer waiting in a collective justifies; a pool of one
            # process keeps Python's own exit. The count is the pool's as it
            # formed: jax.process_count() would open the backend.
            if global_state.num_processes > 1:
                end_pool_on_failure()
            # XLA reads its flags when the backend opens, which the first
            # line below does. The watchdog is the CUDA plugin's, and a TPU
            # host's libtpu need not know its flag.
            if cuda_plugin() and xla_flag("xla_gpu_execution_terminate_timeout") is None:
                apply_xla_flags(f"--xla_gpu_execution_terminate_timeout={EXECUTION_TIMEOUT}")
            print(f"Joined the JAX process pool: process {jax.process_index()} "
                  f"of {jax.process_count()}")
            if jax.process_count() > 1 and jax.default_backend() == "gpu" and not _pool_keys_alike():
                # Before the first compile, which fixes whether the cache is used.
                jax.config.update("jax_enable_compilation_cache", val=False)
            # One collective while the processes are still in lockstep;
            # initialize() returns on every process once the last one has
            # connected. On CPU, collectives rendezvous through the
            # coordinator with a 30 second deadline. Without this the first
            # collective would fall inside orbax's checkpoint-manager barrier
            # in the trainer. By then the processes are as far apart as a
            # wandb init and their model builds, and one that arrives late
            # dies in gloo before the run can report it.
            multihost_utils.sync_global_devices("dew process pool joined")
    if layout is not None and "params" in layout.host:
        from dew.training.distributed import build_mesh
        from dew.training.host import companion_mesh
        companion_mesh(build_mesh())
    print(f"Number of devices: {jax.device_count()}")


def _pool_keys_alike() -> bool:
    """Whether this jax keys a computation that spans processes alike on
    every one of them (jax-ml/jax#40940).

    jax 0.11.2 keys an executable by the compiling process's own topology
    fingerprint, which on a GPU describes the device down to its NVLink
    links. In a pool across GPUs linked differently the processes keyed a
    step apart, and on the next run some loaded it while the others compiled
    it and waited for ever for their shares of its sharded autotuning. Dew
    pins a jax that hashes every process's fingerprint. A jax installed
    around the pin, such as an image's own or a `--no-deps` install, may
    lack it.
    """
    from jax._src import cache_key

    return hasattr(cache_key, "_shared_fingerprints")


def cuda_plugin() -> bool:
    """Whether JAX's CUDA plugin is installed, the one reader of XLA's GPU
    flags; asked before the backend opens, which no other question can be."""
    return (importlib.util.find_spec("jax_plugins") is not None
            and any(importlib.util.find_spec(f"jax_plugins.xla_cuda{major}") is not None
                    for major in (12, 13)))


def run_timestamp() -> str:
    """Return process 0's wall clock as `%Y-%m-%d_%H:%M:%S`, on every process.

    A default run name carries it, and the name is the checkpoint directory
    every process writes into, so a process that read its own clock a second
    later would write into a different directory.
    """
    return broadcast_from_process_zero(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
