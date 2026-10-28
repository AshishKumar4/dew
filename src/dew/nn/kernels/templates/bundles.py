"""The Mosaic GPU Family bundles every template's launches derive their kernels on (`mosaic.Family`), one
registry keyed by the device's instruction family (`generation.family`): Hopper's warpgroup MMA's
(`mosaic_sm90.WGMMA`, and its layout without a producer, `mosaic_sm90.FED`) and Ampere's mma.sync's
(`mosaic_sm80.Mma`); a launch names its bundle by the bundle's `name` (`named`). A device of a family with
no bundle here (Blackwell's tcgen05, consumer Blackwell's mma.sync fed by TMA, a TPU's) runs no Mosaic GPU
mainloop: every template's Mosaic launch is ineligible there, and its call takes the next route by
legality."""

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


def layouts(generation_name: str, dtype) -> tuple[mosaic.Family, ...]:
    """The bundles a launch on a device of `generation_name` may take for operands of `dtype`, in preference
    order: its instruction family's (`of`) first, a producer's layout, whose loads overlap no consumer's
    issue; then, on warpgroup MMA, its layout without a producer (`mosaic_sm90.FED`), whose consumers hold
    the program's entry registers, for a tile whose consumers need more than a producer leaves them. None
    where its family has no bundle."""
    first = of(generation_name, dtype)
    if first is None:
        return ()
    return (first, mosaic_sm90.FED) if first is mosaic_sm90.WGMMA else (first,)


def named(name: str, dtype) -> mosaic.Family:
    """The bundle named `name` (`mosaic.Family.name`, a launch's tile's) for operands of `dtype`."""
    if name == mosaic_sm80.Mma.name:
        return mosaic_sm80.Mma(jnp.dtype(dtype))
    for bundle in (mosaic_sm90.WGMMA, mosaic_sm90.FED):
        if bundle.name == name:
            return bundle
    raise ValueError(f"no Mosaic GPU bundle is named {name!r}")
