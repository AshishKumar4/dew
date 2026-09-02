"""Process setup every recipe runs before it builds anything.

rlimits, the JAX distributed pool and the augmenter/wandb env vars are the
same in every recipe: library wiring, not recipe behavior. The recipes call
this once at the top of main().
"""

import os
import resource
from typing import Optional

import jax


def prepare_process(augmentation_mode: str, wandb_offline: bool = False,
                    multi_host: Optional[bool] = None):
    """Raise the fd/core limits, set the env vars, join the JAX process pool.

    jax.distributed.initialize() finds the coordinator from the environment on
    TPU pods and Slurm/GKE clusters. On a machine with no cluster environment
    it raises the one ValueError below, which is the single-host signature;
    every other failure means a pod run would otherwise continue on one host,
    so it propagates. multi_host=True requires the pool, multi_host=False
    never asks for it.
    """
    # The image augmenters read this at MapTransform construction time
    os.environ['FLAXDIFF_AUGMENT_MODE'] = augmentation_mode
    if wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    resource.setrlimit(
        resource.RLIMIT_CORE,
        (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_OFILE, (65535, 65535))

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
