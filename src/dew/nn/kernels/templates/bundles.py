"""The Mosaic GPU Family bundles every template's launches derive their kernels on (`mosaic.Family`), one
registry keyed by the device's instruction family (`generation.family`): Hopper's warpgroup MMA's
(`mosaic_sm90.WGMMA`) and Ampere's mma.sync's (`mosaic_sm80.Mma`). A device of a family with no bundle
here (Blackwell's tcgen05, consumer Blackwell's mma.sync fed by TMA, a TPU's) runs no Mosaic GPU mainloop:
every template's Mosaic launch is ineligible there, and its call takes the next route by legality."""

import jax.numpy as jnp

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic, mosaic_sm80, mosaic_sm90


def of(generation_name: str, dtype) -> mosaic.Family | None:
    """The bundle a template's Mosaic GPU mainloop runs on a device of `generation_name` (`'sm80'`, the
    assumed or a real one's) for operands of `dtype`, by its instruction family (`generation.family`); None
    where its family has none. mma.sync's bundle stages its blocks by cp.async (`mosaic_sm80`), so a device
    whose pipelines move them by TMA (`generation.Architecture.tma`, consumer Blackwell's) takes none."""
    architecture = generation.architecture(generation_name)
    if architecture is None or generation.capability(generation_name) is None:
        return None
    if architecture.family == mosaic_sm90.WGMMA.name:
        return mosaic_sm90.WGMMA
    if architecture.family == mosaic_sm80.Mma.name and not architecture.tma:
        return mosaic_sm80.Mma(jnp.dtype(dtype))
    return None
