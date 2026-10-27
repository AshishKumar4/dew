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

from dew._coordination import broadcast_from_process_zero, end_pool_on_failure, pool_size, preemption_service
from dew.cache import enable_compilation_cache, persist_compilations
from dew.devices import apply_xla_flags, cuda_plugin, preallocate_gpu_pool, refuse_cpu_only_jax, xla_flag
from dew.logging import configure as configure_logging
from dew.pool import (
    PREEMPTED_EXIT,
    PROCESS_COUNT,
    PROCESS_ID,
    joins_pool,
    local_gpu_count,
    pool_cluster,
    refuse_idle_gpus,
    runs_on_gpu,
    slurm_tasks_here,
)

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dew.training.tracker import Wandb


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
                    compilation_cache_dir: str | None = None) -> None:
    """Set the environment variables and XLA flags, raise the soft file
    descriptor limit, and join the JAX process pool.

    `wandb` is the run's `dew.training.tracker.Wandb`, or None for a run without a
    tracker. Only its offline switch is read, and it has to be read before
    wandb opens a run. `multi_host=True` requires a pool, False never joins
    one, and None joins the pool the environment describes, if any.
    `compilation_cache_dir` names the directory of JAX's persistent
    compilation cache, which a pool across hosts must keep on storage they
    share; without one the process caches under the user's cache directory
    where its pool shares it (`dew.cache.persist_compilations`), and
    JAX_ENABLE_COMPILATION_CACHE=false turns the cache off.

    `xla_flags` reaches XLA through the environment, which XLA reads when it
    opens a backend. So this call has to come before the first JAX call in
    the process, and it is the first line of every recipe. If you use the
    library without a recipe, set XLA_FLAGS in the environment yourself.
    A state kept on the CPU (`Layout(host=("variables",))`) needs one CPU
    device per local accelerator, which the trainer checks on its own mesh
    (`dew.distributed.host.companion_mesh`).
    """
    _set_environment(wandb, xla_flags, compilation_cache_dir)
    _raise_limits()
    _join_process_pool(multi_host)
    ready_process()
    _log.info("Number of devices: %s", jax.device_count())


def unprepared(multi_host: bool | None, xla_flags: str | None,
               compilation_cache_dir: str | None) -> list[str]:
    """Which of `prepare_process`'s options a run states that this process does not carry: a flag of
    `xla_flags` its XLA_FLAGS lacks, a compilation cache JAX keeps elsewhere, a pool it is not in
    (`multi_host=True`) or is in (False). Each acts before JAX opens a backend, so a run trained in a
    process no `prepare_process` set up with them would run without them."""
    missed = []
    if xla_flags and not set(xla_flags.split()) <= set(os.environ.get('XLA_FLAGS', '').split()):
        missed.append("xla_flags")
    if compilation_cache_dir and jax.config.jax_compilation_cache_dir != compilation_cache_dir:
        missed.append("compilation_cache_dir")
    if multi_host is not None and jax.distributed.is_initialized() != multi_host:
        missed.append("multi_host")
    return missed


def ready_process() -> None:
    """What every Dew run needs of its process and `import dew` leaves alone: Dew's log handler
    (`dew.logging.configure`), the persistent compilation cache under the user's cache directory
    (`dew.cache.persist_compilations`), which a run's own directory, a pool whose processes do not
    share it, or JAX_ENABLE_COMPILATION_CACHE=false leaves alone, and the refusal of a machine with
    an NVIDIA GPU whose JAX opened with none (`dew.devices.refuse_cpu_only_jax`), which
    opens the backend, so the cache's flags come first. `prepare_process` readies the process once
    its flags are set and its pool joined, and `Trainer.fit` readies one no `prepare_process` did;
    each step is idempotent."""
    configure_logging()
    persist_compilations()
    refuse_cpu_only_jax()


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
    preallocate_gpu_pool()
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
    otherwise pin each to the GPU at SLURM_LOCALID. A process that nothing
    declares a pool for (no cluster JAX would detect, no process count or
    coordinator a launcher set) runs alone without asking JAX; every failure
    to join a pool propagates, since a pod run would otherwise continue on
    one host. multi_host=True
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
    cluster = pool_cluster(multi_host)
    tasks = slurm_tasks_here()
    if (cluster is not None and cluster.name == "slurm" and cluster.count > 1 and tasks is not None
            and runs_on_gpu(os.environ)):
        # Before joining: a pool whose ranks see GPUs they will not use
        # trains on fewer than it was given, and nothing reports it.
        refuse_idle_gpus(tasks, local_gpu_count(), "this slurm step")
    if not joins_pool(multi_host, cluster):
        return
    if PROCESS_COUNT in os.environ:
        jax.distributed.initialize(num_processes=int(os.environ[PROCESS_COUNT]),
                                   process_id=int(os.environ[PROCESS_ID]),
                                   cluster_detection_method="deactivate")
    else:
        jax.distributed.initialize()
    _joined()


def _joined() -> None:
    """What a process does once it has joined the pool."""
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
    # One collective while the processes are still in lockstep;
    # initialize() returns on every process once the last one has
    # connected. On CPU, collectives rendezvous through the
    # coordinator with a 30 second deadline. Without this the first
    # collective would fall inside orbax's checkpoint-manager barrier
    # in the trainer. By then the processes are as far apart as a
    # wandb init and their model builds, and one that arrives late
    # dies in gloo before the run can report it.
    multihost_utils.sync_global_devices("dew process pool joined")


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
        # The notice's checks, each process's in step with its peers': the pool's sync points count
        # them, so a run's cooldown branches ask between their own steps too (`reached`).
        self._asked = 0
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

    def reached(self) -> bool:
        """Return whether to stop here, at this check.

        In a pool, that is whether every process agreed to stop at this check,
        the sync point counting each check every process makes in step (each
        step's, and each step of a cooldown branch), so it grows across a
        run however its steps run; a pool without the preemption service
        never stops. A process outside a pool stops once SIGTERM has arrived.
        """
        if not self._pool:
            return self._signalled
        if not preemption_service():
            return False
        self._asked += 1
        return multihost_utils.reached_preemption_sync_point(self._asked)


def run_timestamp() -> str:
    """Return process 0's wall clock as `%Y-%m-%d_%H:%M:%S`, on every process.

    The default run name includes it, and that name is the checkpoint
    directory every process writes into. So a process that read its own
    clock a second later would write into a different directory.
    """
    return broadcast_from_process_zero(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))


__all__ = ["Preempted", "PreemptionNotice", "prepare_process", "run_timestamp", "unprepared"]
