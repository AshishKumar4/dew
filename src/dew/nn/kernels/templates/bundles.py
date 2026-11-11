"""The Mosaic Family bundles every template's launches derive their kernels on (`mosaic.Family`), composed
from shared components rather than one bundle a generation: a matrix issuer with its accumulator home and
execution (mma.sync's warps into registers, `mosaic_sm80.MmaIssuer`; wgmma's warpgroups into registers,
`mosaic_sm90.WgmmaIssuer`; tcgen05's warp into tensor memory with its 2-CTA pairs and block scales,
`mosaic_sm100.Tcgen05Issuer`; a TPU core's MXU) and a loader (cp.async, `mosaic_sm80.CpAsync`; TMA rings a
producer fills, `mosaic.RingFamily`, multicast across a cluster; TMA rings their consumers fill,
`fed.ConsumerFed`; a TPU core's VMEM rings). Each composition reuses its components and overrides only what
differs: `mosaic_sm80.Mma` (mma.sync on cp.async), `MmaFed` (mma.sync on consumer-fed TMA rings),
`mosaic_sm90.WGMMA` and `FED` (wgmma on producer- and consumer-fed rings), `mosaic_sm100.RINGS`, `PAIR` and
`CHUNKS` (tcgen05 on producer rings, its accumulators persisted on one CTA or a pair's, or a product's in
registers), `mosaic_tpu.MXU`. A
device takes the most capable legal component of each kind its facts state (`compositions`: its issuers and
loaders, stated per device in `generation.Architecture`, never implied by a generation's number), each
composition as the device runs it (`Family.on`: its cluster and block scales the device's), and a launch the
first composition, by issuer, that takes its call (`groups`), so no CUDA device from sm80 leaves Mosaic (row
4727, row 4738). A launch names its bundle by the bundle's `name` (`named`). A TPU core's bundle is a launch's
written wholly against the family's seams (`tpu`: its staging, reads, grid coordinates and turns the family's,
the attention mainloop's); every other template's Mosaic launch is ineligible on a TPU, and its call takes the
next route by legality."""

import dataclasses
from collections.abc import Callable

import jax.numpy as jnp

from dew.nn.kernels import generation
from dew.nn.kernels.templates import fed, mosaic, mosaic_sm80, mosaic_sm90, mosaic_sm100, mosaic_tpu


@dataclasses.dataclass(frozen=True)
class MmaFed(mosaic_sm80.MmaIssuer, fed.ConsumerFed):
    """mma.sync's issuer and register home (`mosaic_sm80.MmaIssuer`) on TMA rings its consumer feeds
    (`fed.ConsumerFed`), on every device that moves its blocks by TMA (`legal`), composed from the two and no
    template's. One consumer a program, as mma.sync's on an
    A100, where two programs a multiprocessor ran faster than two consumers (CONSUMERS1, CONSUMERS3);
    unmeasured on sm120."""

    name = "mma.sync.fed"
    consumers = 1
    held_registers = 65
    """mma.sync's measured whole (`MmaIssuer`'s 40) and the 25 its consumer-fed launches held past it on a
    B200: ptxas's least register cap less the model over every recurrence gate launch at base e346aa4af72
    (FS-CENSUSFED-100d), at most +25 (key_values_back), +24 (states), +22 (keys_back, values_back), +17
    (query_key), where the same launches on cp.async held at most +21 and an RTX PRO 6000's at most +7
    (FS-CENSUSFED-120): the ring each consumer fills, its slots, barrier phases and copy coordinates live
    in its own registers, warpgroup MMA's producer's uniform ones."""


@dataclasses.dataclass(frozen=True)
class Launch:
    """What a launch states of its call that a composition reads (`Composition.takes`): it states its tiles'
    accumulators (`stated`, `Transport.persist`'s `accumulators`: the GEMM template's), its program scopes the
    issuer's product scratch (`scoped`, `Issuer.products_in`: the recurrence template's), it is written
    wholly against the family's seams (`tpu`, the module's account)."""

    stated: bool = False
    scoped: bool = False
    tpu: bool = False


@dataclasses.dataclass(frozen=True)
class Composition:
    """A bundle, declared once: `build(dtype)` composes it for operands of `dtype`, and `takes(launch)` says
    whether a launch's call may run on it (`Launch`)."""

    build: Callable[[jnp.dtype], mosaic.Family]
    takes: Callable[[Launch], bool] = lambda launch: True


COMPOSITIONS: dict[str, tuple[Composition, ...]] = {
    # tcgen05 on TMA rings, its accumulators in tensor memory: persisted a tile in full-lane tensor memory
    # and read out in chunks for a launch that states its tiles' accumulators; scoped a product in half-lane
    # tensor memory and read back into registers in the WGMMA layout (row 4747) for a launch holding its
    # 64-row tiles in registers between products whose program scopes the issuer's product scratch. A
    # launch stating neither takes the next issuer. A launch stating its accumulators takes the pair too
    # (`mosaic_sm100.PAIR`, CUTLASS's 2-SM UMMA, a cluster of two each product spans), by its tile's config,
    # after the one CTA's.
    mosaic_sm100.RINGS.name: (Composition(lambda dtype: mosaic_sm100.RINGS, lambda launch: launch.stated),
                              Composition(lambda dtype: mosaic_sm100.PAIR, lambda launch: launch.stated),
                              Composition(lambda dtype: mosaic_sm100.CHUNKS,
                                          lambda launch: launch.scoped and not launch.stated)),
    # wgmma: TMA rings a producer fills, then rings its consumers fill.
    mosaic_sm90.WGMMA.name: (Composition(lambda dtype: mosaic_sm90.WGMMA),
                             Composition(lambda dtype: mosaic_sm90.FED)),
    # mma.sync, which every NVIDIA GPU from sm80 issues: TMA rings its consumer fills where the device moves
    # blocks by TMA, cp.async where it does not (`legal`).
    mosaic_sm80.Mma.name: (Composition(MmaFed), Composition(mosaic_sm80.Mma)),
    # A TPU core's MXU, for a launch written wholly against the family's seams.
    "tpu": (Composition(lambda dtype: mosaic_tpu.MXU, lambda launch: launch.tpu),),
}
"""Every bundle, by matrix issuer (`generation.Architecture.issuers`' names), each issuer's in preference
order: the one declaration a launch's selection (`groups`) and its tile's name (`named`) both read."""


def groups(generation_name: str, dtype, *, stated: bool = False, scoped: bool = False, tpu: bool = False
           ) -> tuple[tuple[mosaic.Family, ...], ...]:
    """The bundles a launch on a device of `generation_name` (`'sm80'`, the assumed or a real one's) may run
    for operands of `dtype`, by matrix issuer, most capable first (`generation.Architecture.issuers`, stated
    per device, never implied by a generation's number): each issuer's compositions (`COMPOSITIONS`) that
    take the launch (`Launch`: `stated`, `scoped`, `tpu`) and the device runs (`legal`), in the issuer's
    order. A launch takes the first group whose bundles take its call, so a device's own issuer runs it where
    it can and the next where not. None where the device issues none of them (a CPU, a GPU before
    Ampere)."""
    architecture = generation.architecture(generation_name)
    if architecture is None or (architecture.family != "tpu"
                                and generation.capability(generation_name) is None):
        return ()
    launch = Launch(stated=stated, scoped=scoped, tpu=tpu)
    found = (tuple(family for family in (composition.build(jnp.dtype(dtype))
                                         for composition in COMPOSITIONS[issuer] if composition.takes(launch))
                   if legal(family, architecture))
             for issuer in architecture.issuers)
    return tuple(group for group in found if group)


def legal(family: mosaic.Family, architecture: generation.Architecture) -> bool:
    """Whether a device of `architecture` runs `family`'s components: its loader the copy Mosaic GPU moves the
    device's blocks by (`Architecture.tma`: TMA rings from sm90, cp.async before it, whose waits Mosaic
    lowers for no TMA device), its cluster within the multiprocessors the device's multicast reaches
    (`Architecture.cluster`), its block-scaled products where its tensor instructions read such operands
    (`Architecture.block_scales`)."""
    copies = {mosaic.RingFamily: True, mosaic_sm80.CpAsync: False}
    by_tma = next((tma for loader, tma in copies.items() if isinstance(family, loader)), architecture.tma)
    return (by_tma == architecture.tma and family.cluster <= architecture.cluster
            and (architecture.block_scales or not family.block_formats))


def of(generation_name: str, dtype, *, stated: bool = False, scoped: bool = False, tpu: bool = False
       ) -> mosaic.Family | None:
    """The most capable bundle a launch on a device of `generation_name` may run for operands of `dtype`
    (`groups`' first), None where there is none."""
    found = groups(generation_name, dtype, stated=stated, scoped=scoped, tpu=tpu)
    return found[0][0] if found else None


def useful_rows(family: mosaic.Family, rows: int) -> float:
    """The share of the rows `family`'s products issue for a launch of `rows` rows that are the launch's: its
    rows padded to the issuer's least (`Issuer.rows`), so 8 rows fill half of mma.sync's 16 and an eighth of
    wgmma's 64."""
    return rows / (-(-rows // family.rows) * family.rows)


def layouts(generation_name: str, dtype, rows: int | None = None, *, stated: bool = False,
            scoped: bool = False, tpu: bool = False) -> tuple[mosaic.Family, ...]:
    """The bundles a launch on a device of `generation_name` may take for operands of `dtype`, in preference
    order: its most capable issuer's (`groups`' first), a producer's layout before one whose consumers fill
    their own rings. Given a launch's `rows`, every issuer's, ordered by the share of its padded rows the
    launch fills (`useful_rows`), the issuers' order on a tie: a decode step's 8 rows take mma.sync's 16
    before wgmma's 64 (row 4042), where a launch's tile pads its rows to the bundle's least. None where the
    device has none; `stated`, `scoped` and `tpu` as `groups`'."""
    found = groups(generation_name, dtype, stated=stated, scoped=scoped, tpu=tpu)
    if rows is None:
        return found[0] if found else ()
    flat = [family for group in found for family in group]
    return tuple(sorted(flat, key=lambda family: -useful_rows(family, rows)))


def named(name: str, dtype) -> mosaic.Family:
    """The bundle named `name` (`mosaic.Family.name`, a launch's tile's) for operands of `dtype`: the
    composition declared so (`COMPOSITIONS`)."""
    for compositions in COMPOSITIONS.values():
        for composition in compositions:
            if (family := composition.build(jnp.dtype(dtype))).name == name:
                return family
    raise ValueError(f"no Mosaic GPU bundle is named {name!r}")
