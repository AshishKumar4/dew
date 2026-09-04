"""Process setup every recipe runs before it builds anything.

rlimits, the XLA flags, the JAX distributed pool and the augmenter/wandb env
vars are the same in every recipe: library wiring, not recipe behavior. The
recipes call this once at the top of main().
"""

import os
import resource
from datetime import datetime
from typing import Optional

import jax

from dew.telemetry.devices import apply_xla_flags
from dew.training.distributed import broadcast_from_process_zero


def prepare_process(augmentation_mode: str, wandb_offline: bool = False,
                    multi_host: Optional[bool] = None,
                    xla_flags: Optional[str] = None):
    """Raise the fd/core limits, set the env vars, join the JAX process pool.

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
    # The image augmenters read this at MapTransform construction time
    os.environ['FLAXDIFF_AUGMENT_MODE'] = augmentation_mode
    if wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'
    apply_xla_flags(xla_flags)

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
    print(f"Number of devices: {jax.device_count()}")


def run_timestamp() -> str:
    """Process 0's wall clock as `%Y-%m-%d_%H:%M:%S`, on every process.

    A default run name carries it, and the name is the checkpoint directory
    every process writes into, so a process that read its own clock a second
    later would write into a directory of its own.
    """
    return broadcast_from_process_zero(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
