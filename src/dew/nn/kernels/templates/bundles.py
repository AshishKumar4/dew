"""The Mosaic GPU Family bundles every template's launches derive their kernels on (`mosaic.Family`), one
registry keyed by the device's instruction family (`generation.family`), each bundle an issuer composed with
an execution policy: Hopper's warpgroup MMA's (`mosaic_sm90.WGMMA`, and its layout without a producer,
`mosaic_sm90.FED`), Ampere's mma.sync's on cp.async (`mosaic_sm80.Mma`) and consumer Blackwell's mma.sync on
TMA rings (`MmaFed`); a launch names its bundle by the bundle's `name` (`named`). Blackwell's tcgen05, whose
products accumulate in tensor memory, has its own bundle (`mosaic_sm100.TCGEN05`), which the templates'
Blackwell mainloops, not yet in any route, import themselves. A device of a family with no bundle here (a
TPU's) runs no Mosaic GPU mainloop: every template's Mosaic launch is ineligible there, and its call takes the
next route by legality."""

import dataclasses

import jax.numpy as jnp

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic, mosaic_sm80, mosaic_sm90

INTERPRETED = mosaic_sm90.WGMMA
"""The bundle Mosaic GPU's interpreter runs a template's mainloop by off a GPU of a family a bundle claims (a
CPU): wgmma's, whose producer role, TMA rings and barriers the interpreter models (its products the dots they
compute)."""


@dataclasses.dataclass(frozen=True)
class MmaFed(mosaic_sm80.MmaIssuer, mosaic.ConsumerFed):
    """mma.sync's issuer and register home (`mosaic_sm80.MmaIssuer`) on TMA rings its consumer feeds
    (`mosaic.ConsumerFed`): consumer Blackwell's (sm120), whose mma.sync moves its blocks by TMA and has no
    warpgroup MMA, composed from the two and no template's. One consumer a program, as mma.sync's on an
    A100, where two programs a multiprocessor ran faster than two consumers (CONSUMERS1, CONSUMERS3);
    unmeasured on sm120."""

    name = "mma.sync.fed"
    consumers = 1


def of(generation_name: str, dtype) -> mosaic.Family | None:
    """The bundle a template's Mosaic GPU mainloop runs on a device of `generation_name` (`'sm80'`, the
    assumed or a real one's) for operands of `dtype`, by its instruction family (`generation.family`); None
    where its family has none. mma.sync's bundle stages its blocks by cp.async (`mosaic_sm80.Mma`), or on a
    device whose blocks move by TMA (`generation.Architecture.tma`, consumer Blackwell's) through TMA rings
    (`MmaFed`)."""
    architecture = generation.architecture(generation_name)
    if architecture is None or generation.capability(generation_name) is None:
        return None
    if architecture.family == mosaic_sm90.WGMMA.name:
        return mosaic_sm90.WGMMA
    if architecture.family == mosaic_sm80.Mma.name:
        return (MmaFed if architecture.tma else mosaic_sm80.Mma)(jnp.dtype(dtype))
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
    for composed in (mosaic_sm80.Mma, MmaFed):
        if composed.name == name:
            return composed(jnp.dtype(dtype))
    for bundle in (mosaic_sm90.WGMMA, mosaic_sm90.FED):
        if bundle.name == name:
            return bundle
    raise ValueError(f"no Mosaic GPU bundle is named {name!r}")
