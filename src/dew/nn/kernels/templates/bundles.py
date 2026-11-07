"""The Mosaic Family bundles every template's launches derive their kernels on (`mosaic.Family`), composed
from shared components rather than one bundle a generation: a matrix issuer with its accumulator home and
execution (mma.sync's warps into registers, `mosaic_sm80.MmaIssuer`; wgmma's warpgroups into registers,
`mosaic_sm90.WgmmaIssuer`; tcgen05's warp into tensor memory with its 2-CTA pairs and block scales,
`mosaic_sm100.Tcgen05Issuer`; a TPU core's MXU) and a loader (cp.async, `mosaic_sm80.CpAsync`; TMA rings a
producer fills, `mosaic.RingFamily`, multicast across a cluster; TMA rings their consumers fill,
`fed.ConsumerFed`; a TPU core's VMEM rings). Each composition reuses its components and overrides only what
differs: `mosaic_sm80.Mma` (mma.sync on cp.async), `MmaFed` (mma.sync on consumer-fed TMA rings),
`mosaic_sm90.WGMMA` and `FED` (wgmma on producer- and consumer-fed rings), `mosaic_sm100.RINGS` (tcgen05 on
producer rings), `mosaic_tpu.MXU`. A device takes the most capable legal component of each kind its facts
state (`compositions`: its issuers and loaders, stated per device in `generation.Architecture`, never implied
by a generation's number), each composition as the device runs it (`Family.on`: its cluster and block
scales the device's), and a launch the first composition, by issuer, that takes its call (`groups`), so no
CUDA device from sm80 leaves Mosaic (row 4727, row 4738). A launch names its bundle by the bundle's `name`
(`named`). A TPU core's bundle is a launch's written wholly against the family's seams (`tpu`: its staging,
reads, grid coordinates and turns the family's, the attention mainloop's); every other template's Mosaic
launch is ineligible on a TPU, and its call takes the next route by legality."""

import dataclasses

import jax.numpy as jnp

from dew.nn.kernels import generation
from dew.nn.kernels.templates import fed, mosaic, mosaic_sm80, mosaic_sm90, mosaic_sm100, mosaic_tpu

INTERPRETED = mosaic_sm90.WGMMA
"""The bundle Mosaic GPU's interpreter runs a template's mainloop by off a GPU of a family a bundle claims (a
CPU): wgmma's, whose producer role, TMA rings and barriers the interpreter models (its products the dots they
compute)."""


@dataclasses.dataclass(frozen=True)
class MmaFed(mosaic_sm80.MmaIssuer, fed.ConsumerFed):
    """mma.sync's issuer and register home (`mosaic_sm80.MmaIssuer`) on TMA rings its consumer feeds
    (`fed.ConsumerFed`), on every device that moves its blocks by TMA (`legal`), composed from the two and no
    template's. One consumer a program, as mma.sync's on an
    A100, where two programs a multiprocessor ran faster than two consumers (CONSUMERS1, CONSUMERS3);
    unmeasured on sm120."""

    name = "mma.sync.fed"
    consumers = 1


def groups(generation_name: str, dtype, *, stated: bool = False, tpu: bool = False
           ) -> tuple[tuple[mosaic.Family, ...], ...]:
    """The bundles a launch on a device of `generation_name` (`'sm80'`, the assumed or a real one's) may run
    for operands of `dtype`, by matrix issuer, most capable first (`generation.Architecture.issuers`, stated
    per device, never implied by a generation's number), each issuer composed with every loader, its
    accumulator home and execution with it, in the issuer's order, those the device runs (`legal`):

    - tcgen05 (`mosaic_sm100.RINGS`): TMA rings into tensor memory, for a launch that states its tiles'
      accumulators (`stated`, `Transport.persist`'s `accumulators`: the GEMM template's); a launch that does
      not takes the next issuer.
    - wgmma: TMA rings a producer fills (`mosaic_sm90.WGMMA`), then rings its consumers fill
      (`mosaic_sm90.FED`).
    - mma.sync, which every NVIDIA GPU from sm80 issues: TMA rings its consumer fills (`MmaFed`), then
      cp.async (`mosaic_sm80.Mma`).
    - A TPU core's MXU (`mosaic_tpu.MXU`), for a launch written wholly against the family's seams (`tpu`,
      the module's account).

    A launch takes the first group whose bundles take its call, so a device's own issuer runs it where it can
    and the next where not (row 4727: a B200's recurrence runs on mma.sync). None where the device issues
    none of them (a CPU, a GPU before Ampere)."""
    architecture = generation.architecture(generation_name)
    if architecture is None or (architecture.family != "tpu"
                                and generation.capability(generation_name) is None):
        return ()
    composed = {mosaic_sm100.RINGS.name: (mosaic_sm100.RINGS,) if stated else (),
                mosaic_sm90.WGMMA.name: (mosaic_sm90.WGMMA, mosaic_sm90.FED),
                mosaic_sm80.Mma.name: (MmaFed(jnp.dtype(dtype)), mosaic_sm80.Mma(jnp.dtype(dtype))),
                "tpu": (mosaic_tpu.MXU,) if tpu else ()}
    found = (tuple(family for family in composed[issuer] if legal(family, architecture))
             for issuer in architecture.issuers)
    return tuple(group for group in found if group)


def legal(family: mosaic.Family, architecture: generation.Architecture) -> bool:
    """Whether a device of `architecture` runs `family`'s components: its TMA rings where the device moves
    blocks by TMA (`Architecture.tma`), its cluster within the multiprocessors the device's multicast reaches
    (`Architecture.cluster`), its block-scaled products where its tensor instructions read such operands
    (`Architecture.block_scales`)."""
    return ((architecture.tma or not isinstance(family, mosaic.RingFamily))
            and family.cluster <= architecture.cluster
            and (architecture.block_scales or not family.takes_block_scales))


def of(generation_name: str, dtype, *, stated: bool = False, tpu: bool = False) -> mosaic.Family | None:
    """The most capable bundle a launch on a device of `generation_name` may run for operands of `dtype`
    (`groups`' first), None where there is none."""
    found = groups(generation_name, dtype, stated=stated, tpu=tpu)
    return found[0][0] if found else None


def useful_rows(family: mosaic.Family, rows: int) -> float:
    """The share of the rows `family`'s products issue for a launch of `rows` rows that are the launch's: its
    rows padded to the issuer's least (`Issuer.rows`), so 8 rows fill half of mma.sync's 16 and an eighth of
    wgmma's 64."""
    return rows / (-(-rows // family.rows) * family.rows)


def layouts(generation_name: str, dtype, rows: int | None = None, *, stated: bool = False, tpu: bool = False
            ) -> tuple[mosaic.Family, ...]:
    """The bundles a launch on a device of `generation_name` may take for operands of `dtype`, in preference
    order: its most capable issuer's (`groups`' first), a producer's layout before one whose consumers fill
    their own rings. Given a launch's `rows`, every issuer's, ordered by the share of its padded rows the
    launch fills (`useful_rows`), the issuers' order on a tie: a decode step's 8 rows take mma.sync's 16
    before wgmma's 64 (row 4042), where a launch's tile pads its rows to the bundle's least. None where the
    device has none; `stated` and `tpu` as `groups`'."""
    found = groups(generation_name, dtype, stated=stated, tpu=tpu)
    if rows is None:
        return found[0] if found else ()
    flat = [family for group in found for family in group]
    return tuple(sorted(flat, key=lambda family: -useful_rows(family, rows)))


def named(name: str, dtype) -> mosaic.Family:
    """The bundle named `name` (`mosaic.Family.name`, a launch's tile's) for operands of `dtype`."""
    for composed in (mosaic_sm80.Mma, MmaFed):
        if composed.name == name:
            return composed(jnp.dtype(dtype))
    for bundle in (mosaic_sm90.WGMMA, mosaic_sm90.FED, mosaic_sm100.RINGS, mosaic_tpu.MXU):
        if bundle.name == name:
            return bundle
    raise ValueError(f"no Mosaic GPU bundle is named {name!r}")
