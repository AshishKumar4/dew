"""Process setup that every recipe runs before it builds anything.

Every recipe needs the same setup: the file descriptor limit, the XLA flags,
the compilation cache, the JAX distributed pool, and the environment
variables that wandb and the tokenizers read. So the library does it in
`prepare_process`, and each recipe calls that once at the top of main().
"""

from __future__ import annotations

import logging
import os
import resource
import signal
import threading
import types
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

import jax
from jax.experimental import multihost_utils

from dew.cache import enable_compilation_cache
from dew.coordination import broadcast_from_process_zero, end_pool_on_failure, pool_size, preemption_service
from dew.pool import (
    PREEMPTED_EXIT,
    PROCESS_COUNT,
    PROCESS_ID,
    detected_cluster,
    local_gpu_count,
    refuse_idle_gpus,
    runs_on_gpu,
    slurm_tasks_here,
)
from dew.telemetry.devices import apply_xla_flags, cuda_plugin, unpartition_gpu_pool, xla_flag

_log = logging.getLogger(__name__)

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
    """Set the environment variables and XLA flags, raise the soft file
    descriptor limit, and join the JAX process pool.

    `wandb` is the run's `dew.config.Wandb`, or None for a run without a
    tracker. Only its offline switch is read, and it has to be read before
    wandb opens a run. `multi_host=True` requires a pool, False never joins
    one, and None joins the pool the environment describes, if any.
    `compilation_cache_dir`, when set, turns on JAX's persistent compilation
    cache in that directory.

    `xla_flags` reaches XLA through the environment, which XLA reads when it
    opens a backend. So this call has to come before the first JAX call in
    the process, and it is the first line of every recipe. If you use the
    library without a recipe, set XLA_FLAGS in the environment yourself.

    `layout` is the same `Layout` you pass to `Trainer`. When its `host`
    includes "variables", the master copy of the whole train state is kept
    on the CPU and the optimizer update runs there, so this call checks the
    CPU devices that this requires.
    JAX_PLATFORMS must then allow CPU beside the accelerator, and
    JAX_NUM_CPU_DEVICES or the existing XLA flags must give one CPU device
    per local accelerator before this call. The check raises `ValueError`
    when they do not, and it never changes the backend configuration after
    JAX has initialized.
    """
    _set_environment(wandb, xla_flags, compilation_cache_dir)
    _raise_limits()
    _join_process_pool(multi_host)
    if layout is not None and "variables" in layout.host:
        from dew.training.distributed import MeshSpec
        from dew.training.host import companion_mesh
        companion_mesh(MeshSpec().build())
    _log.info("Number of devices: %s", jax.device_count())


def _set_environment(wandb: Wandb | None, xla_flags: str | None,
                     compilation_cache_dir: str | None) -> None:
    """The env vars wandb and the tokenizers read, the run's XLA flags and
    Dew's defaults beside them, and the compilation cache: everything read
    before a backend opens."""
    if wandb is not None and wandb.offline:
        os.environ['WANDB_MODE'] = 'offline'
    # HF tokenizers fork a thread pool; grain's workers fork the process.
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    apply_xla_flags(xla_flags)
    unpartition_gpu_pool()
    if compilation_cache_dir:
        enable_compilation_cache(compilation_cache_dir)


_DESCRIPTORS = 65535
"""The open files a run asks room for: data loaders' and checkpoints' descriptors."""


def _raise_limits() -> None:
    """Raise the soft descriptor limit toward `_DESCRIPTORS`, as far as the
    hard limit the process was started with allows. The hard limit, which
    only a privileged process can raise, and a soft limit already higher are
    left as they are."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = _DESCRIPTORS if hard == resource.RLIM_INFINITY else min(_DESCRIPTORS, hard)
    if soft != resource.RLIM_INFINITY and soft < wanted:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))


def _join_process_pool(multi_host: bool | None) -> None:
    """Join the JAX process pool the environment describes, if any.

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
    """
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
    if multi_host is False or one_task:
        return
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
        _joined()


def _joined() -> None:
    """What a process does once it has joined the pool.

    A GPU pool keeps the persistent compilation cache when its jax keys a
    computation that spans processes alike on every one of them, as the jax
    constraints.txt names does (`_pool_keys_alike`). With another jax, such
    as the 0.11.2 release, it compiles without the cache and says so on every
    process: some ranks would load a step that the others compile, and that
    compile waits for every rank for ever.
    """
    # Before the backend opens, which can fail on one process of a
    # pool that has formed, a GPU with no memory left for one. The
    # watch leaves through os._exit, past the atexit handlers, which
    # only a peer waiting in a collective justifies; a pool of one
    # process keeps Python's own exit. The count is the pool's as it
    # formed: jax.process_count() would open the backend.
    if pool_size() > 1:
        end_pool_on_failure()
    # XLA reads its flags when the backend opens, which the first
    # line below does. The watchdog is the CUDA plugin's, and a TPU
    # host's libtpu need not know its flag.
    if cuda_plugin() and xla_flag("xla_gpu_execution_terminate_timeout") is None:
        apply_xla_flags(f"--xla_gpu_execution_terminate_timeout={EXECUTION_TIMEOUT}")
    _log.info(
        "Joined the JAX process pool: process %s of %s", jax.process_index(), jax.process_count()
    )
    if jax.process_count() > 1 and jax.default_backend() == "gpu" and not _pool_keys_alike():
        # Before the first compile, which fixes whether the cache is used.
        jax.config.update("jax_enable_compilation_cache", val=False)
        _log.warning("This jax keys a computation that spans processes apart on each of them "
                     "(jax-ml/jax#40940), so the pool compiles without the persistent compilation "
                     "cache; docs/installation.md names the jax that keeps it")
    # One collective while the processes are still in lockstep;
    # initialize() returns on every process once the last one has
    # connected. On CPU, collectives rendezvous through the
    # coordinator with a 30 second deadline. Without this the first
    # collective would fall inside orbax's checkpoint-manager barrier
    # in the trainer. By then the processes are as far apart as a
    # wandb init and their model builds, and one that arrives late
    # dies in gloo before the run can report it.
    multihost_utils.sync_global_devices("dew process pool joined")


def _pool_keys_alike() -> bool:
    """Whether this jax keys a computation that spans processes alike on
    every one of them (jax-ml/jax#40940).

    jax 0.11.2 keys an executable by the compiling process's own topology
    fingerprint, which on a GPU describes the device down to its NVLink
    links. In a pool across GPUs linked differently the processes keyed a
    step apart, and on the next run some loaded it while the others compiled
    it and waited for ever for their shares of its sharded autotuning. The
    jax constraints.txt names hashes every process's fingerprint; the 0.11.2
    release, which the published dewml installs, does not, so the private
    module is asked at this boundary.
    """
    from jax._src import cache_key

    return hasattr(cache_key, "_shared_fingerprints")


class Preempted(SystemExit):
    """Raised by `Trainer.fit` when it stops at a preemption notice.

    `step` is the step it stopped at, and `fit` has written that step's
    checkpoint and data position before raising. Uncaught, it ends the
    program with exit status `PREEMPTED_EXIT` and no traceback, as SIGTERM
    itself would have. Run the same program again to resume from the
    checkpoint.
    """

    def __init__(self, step: int):
        super().__init__(PREEMPTED_EXIT)
        self.step = step


class PreemptionNotice:
    """Reports once a step whether a preemption notice has reached the run,
    from its creation until `close`.

    A scheduler stops a job by sending SIGTERM and then SIGKILL after a grace
    period, such as Slurm's KillWait, Kubernetes' termination grace period
    or a spot VM's notice. In a process pool, JAX's preemption service
    receives the SIGTERM; XLA's notifier replaces the signal handler, so the
    process keeps running. The service shares the notice through the
    coordination service, and `reached_preemption_sync_point` picks one step
    that every process agrees on, so the checkpoint written there is
    complete. A process outside any pool has no such service, so the notice
    is SIGTERM itself, which this object catches until `close`. Only the main
    thread can set a signal handler, so a notice created on another thread
    outside a pool leaves SIGTERM's default effect. A pool with the
    preemption service turned off (jax_enable_preemption_service) gets no
    notice, and SIGTERM ends it as usual.
    """

    def __init__(self):
        self._pool = jax.distributed.is_initialized()
        self._signalled = False
        self._previous: Callable[[int, types.FrameType | None], object] | int | None = None
        self._installed = False
        # Only the main thread may set a handler; a fit run on another thread
        # keeps the default, as before.
        if not self._pool and threading.current_thread() is threading.main_thread():
            self._previous = signal.signal(signal.SIGTERM, self._caught)
            self._installed = True

    def close(self) -> None:
        """Put back the SIGTERM handler this notice replaced."""
        if self._installed:
            # A handler set outside Python reads back as None, and the
            # default stands in for it.
            signal.signal(signal.SIGTERM, signal.SIG_DFL if self._previous is None else self._previous)
            self._installed = False

    def _caught(self, _signum: int, _frame: types.FrameType | None) -> None:
        self._signalled = True

    def reached(self, step: int) -> bool:
        """Return whether to stop at `step`.

        In a pool, that is whether every process agreed to stop there, and a
        pool without the preemption service never stops. A process outside a
        pool stops once SIGTERM has arrived.
        """
        if not self._pool:
            return self._signalled
        if not preemption_service():
            return False
        return multihost_utils.reached_preemption_sync_point(step)


def run_timestamp() -> str:
    """Return process 0's wall clock as `%Y-%m-%d_%H:%M:%S`, on every process.

    The default run name includes it, and that name is the checkpoint
    directory every process writes into. So a process that read its own
    clock a second later would write into a different directory.
    """
    return broadcast_from_process_zero(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))


__all__ = ["Preempted", "PreemptionNotice", "prepare_process", "run_timestamp"]
