"""Process setup every recipe runs before it builds anything.

rlimits, the XLA flags, the compilation cache, the JAX distributed pool and
the env vars wandb and the tokenizers read are the same in every recipe:
library wiring, not recipe behavior. The recipes call this once at the top of
main().
"""

from __future__ import annotations

import os
import resource
from datetime import datetime
from typing import TYPE_CHECKING

import jax
from jax.experimental import multihost_utils

from dew.telemetry.devices import apply_xla_flags
from dew.telemetry.instrumentation import enable_compilation_cache
from dew.training.distributed import broadcast_from_process_zero

if TYPE_CHECKING:
    from dew.config import Wandb


def prepare_process(wandb: Wandb | None = None,
                    multi_host: bool | None = None,
                    xla_flags: str | None = None,
                    compilation_cache_dir: str | None = None) -> None:
    """Raise the fd/core limits, set the env vars, join the JAX process pool.

    `wandb` is the run's `dew.config.Wandb`, or None for a run without a
    tracker; only its offline switch is read, and it has to be read before
    wandb opens a run.

    jax.distributed.initialize() finds the coordinator from the environment on
    TPU pods and Slurm/GKE clusters. On a machine with no cluster environment
    it raises the one ValueError below, which is the single-host signature;
    every other failure means a pod run would otherwise continue on one host,
    so it propagates. multi_host=True requires the pool, multi_host=False
    never asks for it.

    xla_flags reaches XLA through the environment, which it reads when it
    opens a backend, so this call has to come before the first JAX call in the
    process. That is what makes it a recipe's first line and why a library
    user, who never runs a recipe, sets XLA_FLAGS themselves.
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

    if multi_host is not False:
        try:
            jax.distributed.initialize()
        except ValueError as e:
            if multi_host or "coordinator_address" not in str(e):
                raise
        else:
            print(f"Joined the JAX process pool: process {jax.process_index()} "
                  f"of {jax.process_count()}")
            # One collective while the processes are still in lockstep, which
            # they are only here: initialize() returns on every process once
            # the last one has connected. On CPU, collectives rendezvous
            # through the coordinator with a 30 second deadline, and the
            # first one otherwise falls inside orbax's checkpoint-manager
            # barrier in the trainer, by which time the processes are as far
            # apart as a wandb init and their model builds. A process that
            # arrives late dies in gloo rather than in anything the run can
            # report.
            multihost_utils.sync_global_devices("dew process pool joined")
    print(f"Number of devices: {jax.device_count()}")


def run_timestamp() -> str:
    """Process 0's wall clock as `%Y-%m-%d_%H:%M:%S`, on every process.

    A default run name carries it, and the name is the checkpoint directory
    every process writes into, so a process that read its own clock a second
    later would write into a directory of its own.
    """
    return broadcast_from_process_zero(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
