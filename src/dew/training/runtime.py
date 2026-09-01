"""Process setup every recipe runs before it builds anything.

rlimits, the JAX distributed pool and the augmenter/wandb env vars are the
same in every recipe: library wiring, not recipe behavior. The recipes call
this once at the top of main().
"""

import os
import resource

import jax


def prepare_process(augmentation_mode: str, wandb_offline: bool = False):
    """Raise the fd/core limits, join the device pool, set the env vars.

    jax.distributed.initialize() is still swallowed when it fails - a single
    process must keep training - but the reason is printed so the failure is
    at least visible.
    """
    # The image augmenters read this at MapTransform construction time
    os.environ['FLAXDIFF_AUGMENT_MODE'] = augmentation_mode
    if wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    resource.setrlimit(
        resource.RLIMIT_CORE,
        (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_OFILE, (65535, 65535))

    print("Initializing JAX")
    try:
        jax.distributed.initialize()
    except Exception as e:
        print(f"jax.distributed.initialize() failed, continuing on one process: {e}")
    print(f"Number of devices: {jax.device_count()}")
