"""Process setup every recipe runs before it builds anything.

rlimits, the JAX distributed pool and the augmenter/wandb env vars are the
same in every recipe: library wiring, not recipe behavior. The recipes call
this once at the top of main().
"""

import os
import resource

import jax


def prepare_process(augmentation_mode: str, wandb_offline: bool = False,
                    multi_host: bool = False):
    """Raise the fd/core limits, join the device pool, set the env vars.

    multi_host joins the JAX process pool, and a failure to join is fatal: a
    pod run that quietly falls back to one process trains on a slice of the
    data with a slice of the devices and reports it as a full run.
    """
    # The image augmenters read this at MapTransform construction time
    os.environ['FLAXDIFF_AUGMENT_MODE'] = augmentation_mode
    if wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    resource.setrlimit(
        resource.RLIMIT_CORE,
        (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_OFILE, (65535, 65535))

    if multi_host:
        jax.distributed.initialize()
        print(f"Joined the JAX process pool: process {jax.process_index()} "
              f"of {jax.process_count()}")
    print(f"Number of devices: {jax.device_count()}")
