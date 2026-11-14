"""Dew: a GEMM tile's facts and its walk's prices on a device (`mosaic_gpu`'s launches): the bundle a tile
runs on and the rows and columns it spans (`family_of`, `tile_rows`, `tile_columns`), one step's time on a
multiprocessor (`step_seconds`, its shared memory's bytes `step_shared_bytes`), and the walks the derivation
prices from them: a contraction's split (`split_count`, `split_walk`) and a dense product's balanced last
wave (`balanced_tail`, `handoff_steps`, `fixes_up`)."""

import dataclasses
import math
from typing import TYPE_CHECKING

import jax.numpy as jnp

from dew.nn.kernels.templates import bundles, cost as unit_cost, mosaic, tail as tail_of

if TYPE_CHECKING:
    from dew.nn.kernels.templates.gemm.mosaic_gpu import Tiles


def family_of(tiles: "Tiles", dtype) -> mosaic.Family:
    """The bundle a tile runs on: the one it names (`Tiles.family`, `bundles.named`), for a consumer's share
    of the tile's rows and the columns its accumulators hold (`tile_columns`, `mosaic.Issuer.specialized`)."""
    family = bundles.named(tiles.family, dtype)
    return family.specialized(tiles.share, tile_columns(tiles, family))


def tile_columns(tiles: "Tiles", family: mosaic.Family) -> int:
    """The output columns a tile's accumulators hold: its `block_n`, or both CTAs' of a pair whose family's
    products span it (`mosaic.Transport.pairs_product`, tcgen05's pair: its tile twice a CTA's rows and
    columns, each CTA's accumulators its own rows by all the pair's columns)."""
    return tiles.block_n * (tiles.cluster if family.pairs_product else 1)


def walk_columns(tiles: "Tiles", family: mosaic.Family) -> int:
    """The output columns a cluster's tile spans: each program's `tile_columns`, side by side along the
    cluster's columns (`Tiles.cluster_columns`)."""
    return tile_columns(tiles, family) * tiles.cluster_columns


def tile_rows(tiles: "Tiles", family: mosaic.Family) -> int:
    """The output rows a cluster's tile spans: each CTA's `block_m` its own, side by side, whether a pair's
    products span them (`tile_columns`) or its CTAs multicast the blocks they read alike."""
    del family
    return tiles.block_m * tiles.cluster


def step_seconds(tiles: "Tiles", target, itemsizes: tuple[float, float],
                 decoded: tuple = (None, None)) -> float | None:
    """One step of `tiles` on one multiprocessor, in seconds, the unit the kernel gradient's split
    (`split_count`) and a balanced tail (`handoff_steps`) price their walks in: the longer of its products'
    FLOPs at the multiprocessor's 16-bit tensor rate and its operands' bytes through its shared memory at
    that memory's rate (`templates.cost.rates`, each the architecture's a clock at the part's clock; their
    units overlapped, `cost.units`), a step's `step_shared_bytes` of operands of `itemsizes` (the left's
    and the right's) in their `decoded` forms; None where the target states no tensor rate. A pair's step is
    a CTA's: its rows by the pair's columns (`tile_columns`)."""
    rates = unit_cost.rates(target, device=False)
    tensor, shared = rates[0], rates[5]
    if tensor is None:
        return None
    family = family_of(tiles, jnp.bfloat16)
    products = 2 * tiles.block_m * tile_columns(tiles, family) * tiles.block_k / tensor
    moved = step_shared_bytes(tiles, family, itemsizes, decoded)
    return products if shared is None else max(products, moved / shared)


def step_shared_bytes(tiles: "Tiles", family: mosaic.Family, itemsizes: tuple[float, float],
                      decoded: tuple = (None, None)) -> float:
    """The bytes a step of `tiles` passes through one multiprocessor's shared memory: its rows' and its
    matrix's blocks staged once (each CTA its own: a pair's CTA half the pair's columns), then read by its
    products as its issuer reads them (`Issuer.shared_reads`: a fragment's warps' loads, or a descriptor's
    reads an instruction), each consumer its share of the rows and the matrix's block. A pair's CTA's half of
    the matrix is counted read by both CTAs' products (`Transport.pairs_product`): an assumption of this
    model, not a measured fact, which M-EM-35's pinned one-CTA arm against the derived pair tests (rows
    5671, 5363). An operand whose stage is decoded (`decoded`, each operand's `(itemsize, copies)`, its
    decode `packed.lanes`) is read from its stage by its decoders, `copies` times whole (`packed.decoding`:
    once where its consumers share the decode, `shared_decode`, each its columns; each consumer the whole
    stage where not), then written as `copies` stages of the decoded `itemsize` the products read (an NVFP4
    matrix's bf16 stage, `gmm`'s `scales`), or, at itemsize 0, decoded in registers into the products,
    which read it from shared memory never (`gmm`'s `packed_rows`); None, read as staged."""
    share = tiles.block_m // tiles.warpgroups
    lhs_reads, rhs_reads = family.specialized(share, tile_columns(tiles, family)).shared_reads(share)
    readers = tiles.warpgroups * (2 if family.pairs_product else 1)
    elements = (tiles.block_m * tiles.block_k, tiles.block_n * tiles.block_k)
    moved = 0.0
    for count, staged, form, reads in zip(elements, itemsizes, decoded, (lhs_reads, rhs_reads * readers),
                                          strict=True):
        if form is None:
            moved += count * staged * (1 + reads)
            continue
        itemsize, copies = form
        moved += count * (staged + copies * (staged + itemsize) + itemsize * reads)
    return moved


def fixes_up(tiles: "Tiles", dtype) -> bool:
    """Whether a tile's steps may be walked by several programs at `tiles`, each leaving its partial and the
    last to arrive adding them (`mosaic.Family.fixed_up`): its family's, where its consumers hold the tile's
    accumulator in registers, which `fixed_up` leaves and adds whole; not tcgen05's, whose accumulator stays
    in tensor memory (`Issuer.tensor_columns`), read out a chunk of columns at a time; nor a ping-pong tile's,
    whose consumers each finish their own tiles, which no last wave's partials share."""
    family = family_of(tiles, dtype)
    return family.fixes_up and not family.tensor_columns(tiles.block_n, 1) and not tiles.pingpong


def balanced_tail(tiles: "Tiles", grid: tuple[int, int], steps: int, itemsizes: tuple[int, int], target,
                  programs: int) -> tail_of.Tail | None:
    """A dense product's last wave balanced over `programs` (`tail.balanced`, the launch's: its
    multiprocessors times `tiles.programs`) at `tiles` over its output tiles' `grid` and `steps` a tile,
    priced from the target's facts: its MMA step (`step_seconds`), its
    memory's rate and L2 (`GpuTarget.memory_rate`, `l2_bytes`), its operands' blocks a step (`itemsizes`,
    the left's and the right's) and its hand-off (`handoff_steps`); walked where the target states no
    rate."""
    step, rate = step_seconds(tiles, target, itemsizes), target.memory_rate
    if step is None or rate is None:
        return None
    left, right = itemsizes
    traffic = tail_of.Traffic(grid, tiles.grid_block_n, tiles.block_m * tiles.block_k * left,
                              tiles.block_n * tiles.block_k * right, step,
                              target.multiprocessors or programs, rate, target.l2_bytes)
    return tail_of.balanced(grid[0] * grid[1], steps, programs, traffic,
                            handoff_steps=handoff_steps(tiles, target, itemsizes))


def handoff_steps(tiles: "Tiles", target, itemsizes: tuple[int, int]) -> float:
    """A balanced tail's hand-off of a partial between programs (`GpuTarget.handoff_seconds`) in MMA steps
    of `tiles` (`step_seconds`), as `tail.balanced` prices it; none where the part's hand-off or its rate is
    unmeasured."""
    step, handoff = step_seconds(tiles, target, itemsizes), target.handoff_seconds
    return 0.0 if step is None or handoff is None else handoff / step


def split_count(tiles: "Tiles", *, groups: int, rows: int, depth: int, width: int, itemsizes: tuple[int, int],
                multiprocessors: int, step_seconds: float | None, bandwidth: float | None,
                launch_seconds: float | None, partial_rows: int | None = None) -> int:
    """Dew: the slices a contraction of `rows` splits into at `tiles` (`Tiles.splits`: a kernel gradient's,
    `tgmm`, and a swapped decode product's, `gmm`), derived from the call's extents and the device's facts
    in MMA steps (`step_seconds`). Over a wave of P programs (`multiprocessors` x
    `tiles.programs`) and `rows` in steps a group (on average), the busiest program walks ceil(O s / P) of
    the O output tiles at ceil(steps / s) steps each at s slices, and streams as many shares of the operands
    (`rows` of `depth` and of `width` at `itemsizes` bytes, each read once) at its multiprocessor's share
    of `bandwidth`, beside that multiprocessor's other programs' shares (`SplitWalk.walk`): the walk is the
    longer of the two, W unsplit. A split adds its fp32 partials' 2s + 1 passes over the output
    (`partial_rows` of it where they hold fewer than its `groups` x `depth`: a decode step's experts' few
    rows), p steps each at `bandwidth`, and the launch their sum adds (`launch_seconds`, the part's measured
    launch in a program, row 3613; none where unmeasured). The count is the one the walk prices least
    (`SplitWalk.seconds`) among those whose slices each keep at least the ring's stages and one more steps,
    its runs sharing the steps whole (`split_runs`, as `gmm`'s runs share them): one where no split prices
    below the walk unsplit, where the device states no rate, or where the call has no whole tile. A closed
    form for its k whole waves (the last wave idling half a tile's slices on average) took counts the walk
    itself priced above their neighbours: key's 2560 x 1024 kernel gradient over 24,576 rows on an A100 at
    six slices, priced 485 us against four's 465 and two's 444, ran 1.02-1.04x four's time (M-EM-25). Its
    steps priced at their MMA alone, a decode product's few tiles walked unsplit: an A100's 2048 x 2048
    input gradient at 1-32 rows ran 12 programs on 108 multiprocessors, 2.8-3.0x XLA's 128 (M-NS-12, row
    483), where its matrix's stream is the walk."""
    walk = split_walk(tiles, groups=groups, rows=rows, depth=depth, width=width, itemsizes=itemsizes,
                      multiprocessors=multiprocessors, step_seconds=step_seconds, bandwidth=bandwidth,
                      launch_seconds=launch_seconds, partial_rows=partial_rows)
    if walk is None:
        return 1
    steps = walk.steps
    # Its runs share the steps whole: as many runs as a share's ceil(steps / s) steps cover.
    counts = sorted({-(-steps // -(-steps // count)) for count in range(1, steps // (tiles.stages + 1) + 1)})
    return min([1, *counts], key=walk.seconds)


@dataclasses.dataclass(frozen=True)
class SplitWalk:
    """Dew: a contraction's walk at a tile as `split_count` prices it (`split_walk`): its `output` tiles
    over a wave of `wave` programs (`multiprocessors` x the tile's programs), `steps` MMA steps a tile, each
    `step_seconds` long at a multiprocessor's whole tensor rate, its operands' `operand_bytes` streamed once
    at `bandwidth`, and, split, its fp32 partials' passes over the output (`passes` steps each) and their
    sum's `launch` (in steps)."""

    output: int
    wave: int
    steps: int
    multiprocessors: int
    operand_bytes: float
    bandwidth: float
    step_seconds: float
    passes: float
    launch: float

    def walk(self, splits: int) -> float:
        """The busiest program's walk at `splits` slices a tile, in steps: its waves' share of every tile's
        slices, each step at its share of its multiprocessor's tensor rate (the programs a wave runs on each
        at once), or its multiprocessor's share of the operands' stream where that is longer: each
        multiprocessor streams at its share of the memory's rate, its programs' shares in turn, so a wave of
        more programs than multiprocessors but not twice them streams two shares on some. Priced at a rate
        shared by programs instead, an A100's 2048 x 2048 decode product at 16 rows took a seven-way split
        of its 16 tiles (112 programs on 108 multiprocessors) over the six-way swapped walk (96 programs),
        which ran 0.88x its kernel time at the same DRAM bytes, 0.79x its call's (M-EM-45b-a100; the split
        count's own share of that, M-EM-45c). Priced at the whole rate, two programs a multiprocessor read as
        twice the part's peak: an RTX PRO 6000's dense 64 x 64 walks at two programs priced 0.21-0.23 of
        their measured time (M-FS-9), where an underfilled wave, one program a multiprocessor, runs at the
        whole rate."""
        programs = self.output * splits
        waves = -(-programs // self.wave)
        sharing = -(-min(programs, self.wave) // self.multiprocessors)
        busiest = waves * -(-self.steps // splits) * sharing
        # The busiest multiprocessor's programs' shares of the operands, at its share of the memory's rate.
        rate = self.bandwidth / self.multiprocessors
        stream = waves * sharing * self.operand_bytes / programs / rate / self.step_seconds
        return max(busiest, stream)

    def seconds(self, splits: int) -> float:
        """The call's time at `splits` slices a tile: its busiest program's walk and, split, its partials'
        2s + 1 passes over the output and their sum's launch."""
        added = (2 * splits + 1) * self.passes + self.launch if splits > 1 else 0.0
        return (self.walk(splits) + added) * self.step_seconds

    def underfills(self, splits: int) -> bool:
        """Whether the call's programs at `splits` slices a tile fill less than one wave, so its time is its
        busiest program's walk, not its tiles' reads a FLOP, which `derived` balances for a full one."""
        return self.output * splits < self.wave


def split_walk(tiles: "Tiles", *, groups: int, rows: int, depth: int, width: int, itemsizes: tuple[int, int],
               multiprocessors: int, step_seconds: float | None, bandwidth: float | None,
               launch_seconds: float | None, partial_rows: int | None = None) -> SplitWalk | None:
    """Dew: the walk `split_count` prices at `tiles` (its arguments'), None where the device states no rate
    or the call has no whole tile: its wave every program a multiprocessor runs at once, its output tiles
    each program's (a pair's CTA's its rows by the pair's columns, `tile_columns`), `step_seconds` a step
    at the multiprocessor's whole rate (`SplitWalk.walk` shares it among the programs running)."""
    columns = tile_columns(tiles, family_of(tiles, jnp.bfloat16))
    output = groups * -(-depth // tiles.block_m) * (width // columns)
    steps = -(-rows // max(groups, 1) // tiles.block_k)
    if step_seconds is None or bandwidth is None or output == 0:
        return None
    held = groups * depth if partial_rows is None else partial_rows
    return SplitWalk(output, multiprocessors * tiles.programs, steps, multiprocessors,
                     rows * (depth * itemsizes[0] + width * itemsizes[1]), bandwidth, step_seconds,
                     held * width * 4 / bandwidth / step_seconds,
                     0.0 if launch_seconds is None else launch_seconds / step_seconds)


def decode_contraction(tiles: "Tiles", target, *, rows: int, groups: int, depth: int, width: int,
                       columns: int | None, held: int, operands: tuple, multiprocessors: int) -> dict:
    """A decode-sized product's walk at `tiles` as `split_walk` and `split_count` price it (their keywords):
    swapped (`Tiles.swapped`), its matrix the rows, `width` of them, over its `depth`, the call's `rows` its
    `columns` padded to whole column units; else each active group's (at most `rows` of `groups`) one row
    tile over the depth, its partials `held` rows. `operands`, the bytes an element of the
    products' left and right move from device memory and their decoded forms (`step_shared_bytes`)."""
    itemsizes, decoded = operands
    if tiles.swapped:
        assert columns is not None, "a swapped walk's columns are the call's rows, padded"
        shape = {'groups': 1, 'rows': depth, 'depth': width, 'width': columns}
    else:
        active = min(groups, rows)
        shape = {'groups': active, 'rows': depth * active, 'depth': tiles.block_m, 'width': width,
                 'partial_rows': held}
    unswapped = dataclasses.replace(tiles, swapped=False)
    return {**shape, 'itemsizes': itemsizes, 'multiprocessors': multiprocessors,
            'step_seconds': step_seconds(unswapped, target, itemsizes, decoded),
            'bandwidth': target.memory_rate, 'launch_seconds': target.launch_seconds}


def walk_seconds(tiles: "Tiles", contraction: dict) -> float:
    """The seconds `tiles`' walk of a `contraction` (`decode_contraction`) takes at its split, past every
    other walk's where the device states no rate (`split_walk`)."""
    walk = split_walk(dataclasses.replace(tiles, swapped=False), **contraction)
    return math.inf if walk is None else walk.seconds(tiles.splits)
