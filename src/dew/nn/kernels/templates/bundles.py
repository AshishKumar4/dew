"""The Mosaic GPU Family bundles every template's launches derive their kernels on (`mosaic.Family`), one
registry keyed by the device's instruction family (`generation.family`), each bundle an issuer composed with
an execution policy: Hopper's warpgroup MMA's (`mosaic_sm90.WGMMA`, and its layout without a producer,
`mosaic_sm90.FED`), Ampere's mma.sync's on cp.async (`mosaic_sm80.Mma`) and consumer Blackwell's mma.sync on
TMA rings (`MmaFed`); a launch names its bundle by the bundle's `name` (`named`). Blackwell's tcgen05, whose
products accumulate in tensor memory, composes its issuer with the TMA rings (`mosaic_sm100.RINGS`), which a
launch names; `of` gives it to sm100 and sm103 once the templates' persistent launches state their
accumulators (`Transport.persist`'s `accumulators`). A TPU core's bundle composes the MXU's issuer with VMEM
rings (`mosaic_tpu.MXU`), which runs the toolkit's persistent walk; `of` gives it to a TPU once the templates'
launches run on it. Until then a TPU runs no shared mainloop: every template's Mosaic launch is ineligible
there, and its call takes the next route by legality."""

import dataclasses

import jax.numpy as jnp

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic, mosaic_sm80, mosaic_sm90, mosaic_sm100, mosaic_tpu

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
        return _mma(architecture, dtype)
    return None


def _mma(architecture: generation.Architecture, dtype) -> mosaic.Family:
    """mma.sync's bundle on a device of `architecture`: its blocks staged by cp.async (`mosaic_sm80.Mma`),
    or through TMA rings where the device moves blocks by TMA (`MmaFed`)."""
    return (MmaFed if architecture.tma else mosaic_sm80.Mma)(jnp.dtype(dtype))


def useful_rows(family: mosaic.Family, rows: int) -> float:
    """The share of the rows `family`'s products issue for a launch of `rows` rows that are the launch's: its
    rows padded to the issuer's least (`Issuer.rows`), so 8 rows fill half of mma.sync's 16 and an eighth of
    wgmma's 64."""
    return rows / (-(-rows // family.rows) * family.rows)


def layouts(generation_name: str, dtype, rows: int | None = None) -> tuple[mosaic.Family, ...]:
    """The bundles a launch on a device of `generation_name` may take for operands of `dtype`, in preference
    order: its instruction family's (`of`) first, a producer's layout, whose loads overlap no consumer's
    issue; then, on warpgroup MMA, its layout without a producer (`mosaic_sm90.FED`), whose consumers hold
    the program's entry registers, for a tile whose consumers need more than a producer leaves them. Given a
    launch's `rows`, also mma.sync's where the device issues it beside its own family
    (`generation.issuers`), every bundle ordered by the share of its padded rows the launch fills
    (`useful_rows`), that order on a tie: a decode step's 8 rows take mma.sync's 16 before wgmma's 64 (row
    4042), where a launch's tile pads its rows to the bundle's least. None where its family has no bundle."""
    first, architecture = of(generation_name, dtype), generation.architecture(generation_name)
    if first is None or architecture is None:
        return ()
    found = (first, mosaic_sm90.FED) if first is mosaic_sm90.WGMMA else (first,)
    if rows is None:
        return found
    if first.name != mosaic_sm80.Mma.name and mosaic_sm80.Mma.name in generation.issuers(generation_name):
        found = (*found, _mma(architecture, dtype))
    return tuple(sorted(found, key=lambda family: -useful_rows(family, rows)))


def named(name: str, dtype) -> mosaic.Family:
    """The bundle named `name` (`mosaic.Family.name`, a launch's tile's) for operands of `dtype`."""
    for composed in (mosaic_sm80.Mma, MmaFed):
        if composed.name == name:
            return composed(jnp.dtype(dtype))
    for bundle in (mosaic_sm90.WGMMA, mosaic_sm90.FED, mosaic_sm100.RINGS, mosaic_tpu.MXU):
        if bundle.name == name:
            return bundle
    raise ValueError(f"no Mosaic GPU bundle is named {name!r}")
