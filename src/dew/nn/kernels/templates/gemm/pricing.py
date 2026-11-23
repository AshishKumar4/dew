"""Dew: a GEMM tile's facts and its walk's prices on a device (`mosaic_gpu`'s launches): the bundle a tile
runs on and the rows and columns it spans (`family_of`, `tile_rows`, `tile_columns`), one step's time on a
multiprocessor (`step_seconds`, its shared memory's bytes `step_shared_bytes`), and the walks the derivation
prices from them: a contraction's split (`split_count`, `split_walk`) and a dense product's balanced last
wave (`balanced_tail`, `handoff_steps`, `fixes_up`)."""

import contextlib
import dataclasses
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from dew.nn.kernels import generation
from dew.nn.kernels.templates import bundles, cost as unit_cost, mosaic, tail as tail_of
from dew.nn.kernels.templates.gemm.tiling import TilesTemplate as TilesTemplate

if TYPE_CHECKING:
    from dew.nn.kernels.templates.gemm.tiling import Tiles


def family_of(tiles: "Tiles", dtype, target: generation.GpuTarget | None = None) -> mosaic.Family:
    """The bundle a tile runs on: the one it names (`Tiles.family`, `bundles.named`) for `target` (the active
    GPU `generation.gpu_target()` where None), specialized for a consumer's share of the tile's rows and the
    columns its accumulators hold (`tile_columns`, `mosaic.Issuer.specialized`, rows 11885, 12011)."""
    family = bundles.named(tiles.family, dtype, generation.gpu_target() if target is None else target)
    return family.specialized(tiles.share, tile_columns(tiles, family))


def tile_columns(tiles: "Tiles", family: mosaic.Family) -> int:
    """The output columns a tile's accumulators hold: its `block_n`, or both CTAs' of a pair whose family's
    products span it (`mosaic.Transport.pairs_product`, tcgen05's pair: its tile twice a CTA's rows and
    columns, each CTA's accumulators its own rows by all the pair's columns)."""
    return tiles.block_n * (tiles.cluster if family.pairs_product else 1)


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
    tensor = (rates.tf32 or rates.tensor) if itemsizes == (4, 4) else rates.tensor
    shared = rates.shared
    if tensor is None:
        return None
    family = family_of(tiles, jnp.bfloat16, target)
    columns = tile_columns(tiles, family)
    products = 2 * tiles.block_m * columns * tiles.block_k / tensor
    moved = step_shared_bytes(tiles, family, itemsizes, decoded)
    if generation.tpu(target):
        # A TPU core (`multiprocessors == 1`, no inter-SM L2) DMAs each step's operand blocks from HBM into
        # VMEM, moves its fp32 accumulator through VMEM beside the operands, and pays per-step loop overhead.
        left, right = itemsizes
        dma_bytes = (tiles.block_m * left + columns * right) * tiles.block_k
        dma = dma_bytes / target.memory_rate if target.memory_rate else 0.0
        vmem = (moved + 2 * tiles.block_m * columns * 4) / shared if shared is not None else 0.0
        return max(products, dma, vmem) + (target.kernel_step_seconds or 0.0)
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


def merges(tiles: "Tiles", dtype, *, dense: bool, target: generation.GpuTarget | None = None) -> bool:
    """Whether a launch at `tiles` would balance its last wave over the programs, each adding another's
    partial (`tail.balanced`), where its registers hold the merge (`grouped_matmul._Mosaic.balanced`): a dense
    call of one product (`dense`), its products on one CTA (`not family.pairs_product`: Hopper's 2-CTA
    multicast cluster pairs two independent tiles and balances across cluster pairs), and its depth unsplit,
    on a family that fixes a tile up across programs (`fixes_up`)."""
    family = family_of(tiles, dtype, target)
    return dense and tiles.splits == 1 and not family.pairs_product and fixes_up(tiles, dtype, target)


def fixes_up(tiles: "Tiles", dtype, target: generation.GpuTarget | None = None) -> bool:
    """Whether a tile's steps may be walked by several programs at `tiles`, each leaving its partial and the
    last to arrive adding them (`mosaic.Family.fixed_up`): its family's, where its consumers hold the tile's
    accumulator in registers, which `fixed_up` leaves and adds whole; not tcgen05's, whose accumulator stays
    in tensor memory (`Issuer.tensor_columns`), read out a chunk of columns at a time; nor a ping-pong tile's,
    whose consumers each finish their own tiles, which no last wave's partials share."""
    family = family_of(tiles, dtype, target)
    return family.fixes_up and not family.tensor_columns(tiles.block_n, 1) and not tiles.pingpong


def balanced_tail(tiles: "Tiles", grid: tuple[int, int], steps: int, itemsizes: tuple[int, int], target,
                  programs: int, *, dtype=jnp.bfloat16) -> tail_of.Tail | None:
    """A dense product's last wave balanced over `programs // tiles.cluster` cluster pairs (`tail.balanced`)
    at `tiles` over its output tiles' `grid` and `steps` a tile, priced from the target's facts: its MMA step
    (`step_seconds`), its memory's rate and L2 (`GpuTarget.memory_rate`, `l2_bytes`), its operands' blocks a
    step (`itemsizes`, each CTA's row block and the cluster's multicast column block), its semaphore hand-off
    (`handoff_steps`), its `fixed_up` partials' bytes (each tile's fp32 block) and whether its family's walk
    is scoped to its block, which a balanced tail launches persistent (`mosaic.Family.walks_scoped`), each
    priced by `tail.balanced`; None where the target states no rate or walking beats balancing."""
    step, rate = step_seconds(tiles, target, itemsizes), target.memory_rate
    pairs = programs // tiles.cluster
    if step is None or rate is None or pairs <= 0:
        return None
    family = family_of(tiles, dtype, target)
    left, right = itemsizes
    mp = (target.multiprocessors // tiles.cluster) if target.multiprocessors else pairs
    traffic = tail_of.Traffic(grid, tiles.grid_block_n, tile_rows(tiles, family) * tiles.block_k * left,
                              tiles.block_n * tiles.block_k * right, step, mp, rate, target.l2_bytes)
    return tail_of.balanced(grid[0] * grid[1], steps, pairs, traffic,
                            handoff_steps=handoff_steps(tiles, target, itemsizes), scoped=family.walks_scoped,
                            partial_bytes=tile_rows(tiles, family) * tile_columns(tiles, family) * 4)


def dense_tail(tiles: "Tiles", dtype, extents: tuple[int, int, int], itemsizes: tuple[int, int],
               target, programs: int, *, dense: bool = True) -> tail_of.Tail | None:
    """The one balanced-tail decision for a dense GEMM over `(m, n, k)` `extents` at `tiles` on `target`:
    shared by planner (`grouped_matmul._Mosaic`) and kernel (`mosaic_gpu._dense_tail`, row 11945)."""
    if not merges(tiles, dtype, dense=dense, target=target):
        return None
    family = family_of(tiles, dtype, target)
    m, n, k = extents
    grid = (-(-m // tile_rows(tiles, family)), -(-n // tile_columns(tiles, family)))
    pairs = programs // tiles.cluster
    if pairs <= 0:
        return None
    # On Hopper cluster=2, `fixed_up` adds ~7-10% whole-wave overhead across multi-wave persistent launches
    # (M-NG-45b-gh200 20261123T021244Z: F/dX/dW +3.5/+5.1/+3.8% geomean, up to +10.3% on waves >= 3). Restrict
    # cluster > 1 balanced tail to single-wave outliers with >=50% idle tail (e.g. 2048x8192x2560, 14/66).
    waves, rem = divmod(grid[0] * grid[1], pairs)
    if tiles.cluster > 1 and (waves > 1 or rem > pairs // 2):
        return None
    return balanced_tail(tiles, grid, -(-k // tiles.block_k), itemsizes, target, programs, dtype=dtype)


def handoff_steps(tiles: "Tiles", target, itemsizes: tuple[int, int]) -> float:
    """A balanced tail's hand-off of a partial between programs (`GpuTarget.handoff_seconds`) in MMA steps
    of `tiles` (`step_seconds`), as `tail.balanced` prices it; none where the part's hand-off or its rate is
    unmeasured."""
    step, handoff = step_seconds(tiles, target, itemsizes), target.handoff_seconds
    return 0.0 if step is None or handoff is None else handoff / step


def split_count(tiles: "Tiles", *, groups: int, rows: int, depth: int, width: int, itemsizes: tuple[int, int],
                stored: float, multiprocessors: int, step_seconds: float | None, bandwidth: float | None,
                launch_seconds: float | None, partial_rows: int | None = None,
                target: generation.GpuTarget | None = None) -> int:
    """Dew: the slices a contraction of `rows` splits into at `tiles` (`Tiles.splits`: a kernel gradient's,
    `tgmm`, a decode product's or an underfilled dense product's, `gmm`), derived from the call's extents and
    the device's facts in MMA steps (`step_seconds`, rows 10151, 3613, 483)."""
    walk = split_walk(tiles, groups=groups, rows=rows, depth=depth, width=width, itemsizes=itemsizes,
                      stored=stored, multiprocessors=multiprocessors, step_seconds=step_seconds,
                      bandwidth=bandwidth, launch_seconds=launch_seconds, partial_rows=partial_rows,
                      target=target)
    return best_splits(walk, tiles.stages)


def best_splits(walk: "SplitWalk | None", stages: int) -> int:
    """The split count `walk` prices least among whole-step shares keeping at least `stages + 1` steps."""
    if walk is None:
        return 1
    steps = walk.steps
    counts = sorted({-(-steps // -(-steps // count)) for count in range(1, steps // (stages + 1) + 1)})
    return min([1, *counts], key=walk.seconds)


def epilogue_registers(family: mosaic.Family, share: int, columns: int, stored: tuple, *,
                       vectors: int = 0, tile_count: int = 0, has_hook: bool = False) -> int:
    """Extra consumer-thread registers held across epilogue hook evaluation and staged stores:
    the single owner shared by `_Mosaic._epilogue_registers` and `syrk_registers` (row 11982)."""
    from dew.nn.kernels.templates import resources, stores

    scalar = family.scalar_registers()
    def _staging(dtype) -> int:
        if family.stores_through_registers:
            return 0
        chunk = stores.output_staging(columns, (dtype,))[0]
        return (-(-share * chunk * jnp.dtype(dtype).itemsize // (4 * resources.THREADS))
                + resources.CONTROL_REGISTERS)

    if not has_hook:
        return max(0, max((_staging(dtype) for dtype in stored), default=0) - scalar)
    accumulator = -(-share * family.chunk_columns(columns) // resources.THREADS)
    tiles = tile_count * accumulator
    live = [-(-accumulator * jnp.dtype(dtype).itemsize // 4) for dtype in stored]
    hook = max(0, resources.CONTROL_REGISTERS + vectors + tiles - scalar) + sum(live[1:])
    staged = max((max(0, _staging(dtype) - scalar) + sum(live[i + 1:])
                  for i, dtype in enumerate(stored)), default=0)
    return max(hook, staged)


def max_programs(family: mosaic.Family, consumers: int, target=None) -> int:
    """Most resident programs/SM for `consumers` warpgroups at `family`'s scalar-register floor, evaluated
    under `target` when given (shared by `granular_tile` and `MosaicGmm._tile`)."""
    ctx = contextlib.nullcontext() if target is None else generation.as_target(target)
    with ctx:
        return family.programs(consumers, family.scalar_registers() if family.budgets_registers else None)


def granular_tile(family: mosaic.Family, config: "Tiles", dtype, *, target,
                  steps: tuple[int, ...] | None = None, accumulators: int = 1, exact_sums: bool = False,
                  stream_blocks: Callable[[], dict[str, tuple[tuple[int, ...], object]]] | None = None
                  ) -> bool:
    """Whether `config` satisfies `family`'s granularity, register-program residency, fixed accumulator rows,
    and power-of-two tensor-memory column budget (shared by `_Mosaic` and `MosaicSyrk`, rows 11982, 12302)."""
    from dew.nn.kernels.templates import resources
    from dew.nn.kernels.templates.gemm.tiling import CONSUMERS

    if (config.warpgroups > min(family.consumers, CONSUMERS)
            or config.programs > max_programs(family, config.warpgroups, target) or config.stages < 2
            or config.block_m % (family.role_rows * config.sharing)
            or config.block_n % family.column_unit(dtype)
            or config.block_k not in (steps or family.depths(dtype))
            or family.accumulator_rows not in (None, config.share)):
        return False
    tensor = family.tensor_columns(tile_columns(config, family), accumulators, exact_sums)
    if tensor and stream_blocks is not None:
        tensor += family.stream_tensor_columns(stream_blocks(), config.stages)
    limit = target.tmem_columns or 0
    return tensor <= limit and resources.tensor_memory_allocated(tensor) * config.programs <= limit


def prefetch_registers(family: mosaic.Family, config: "Tiles", *, ba) -> int:
    """Extra registers held across a prefetching `mma.sync` step boundary (`products_across`): +16 regs at
    32-deep steps, or +32 regs at 64-deep steps (rows 5271, 11986)."""
    from . import mosaic_gpu

    if not family.prefetches_fragments:
        return 0
    interpret = bool(ba.arguments['interpret'])
    summed = bool(ba.arguments.get('exact_sums')) and not family.sums_in_persist
    scaled = bool(ba.arguments.get('scales')) or bool(ba.arguments.get('step_scales'))
    hooks = ba.arguments['hooks']
    trans_lhs = (bool(ba.arguments.get('packed_rows')) or bool(ba.arguments.get('trans_lhs'))
                 or config.swapped)
    trans_rhs = bool(ba.arguments.get('trans_rhs')) or config.swapped
    active = mosaic_gpu.prefetches(
        family, matrices=1 + (ba.arguments.get('paired') is not None), prologue=hooks.prologue,
        scaled=scaled, chained=bool(ba.arguments.get('chained')), interpret=interpret,
        transposed=(trans_lhs, trans_rhs), summed=summed)
    return (32 if config.block_k > 32 else 16) if active else 0


BOUNDED_REGISTERS = 16
"""Extra consumer registers held by `MmaIssuer.bounded` on one-consumer grouped `mma.sync` tiles
(`mosaic_gpu._row_bound`, rows 5623, 7091): the runtime `int32` row bound and per-M-block predication
pushed `sm80` `64x192x64` normed biased forward from 244 modeled registers to 255 with 32 bytes spilled."""


def bounded_registers(family: mosaic.Family, config: "Tiles", dtype, *, dense: bool,
                      staged_rows: bool) -> int:
    """Extra consumer registers for `family.bounded(_row_bound)` (uses `mosaic_gpu.uses_row_bound`)."""
    from . import mosaic_gpu

    return BOUNDED_REGISTERS if mosaic_gpu.uses_row_bound(
        family, config, dtype, dense=dense, staged_rows=staged_rows) else 0


@dataclasses.dataclass(frozen=True)
class SplitWalk:
    """Dew: a contraction's walk at a tile as `split_count` prices it (`split_walk`): its `output` tiles
    over a wave of `wave` programs (`multiprocessors` x the tile's programs), `steps` MMA steps a tile, each
    `step_seconds` long at a multiprocessor's whole tensor rate, its operands' `operand_bytes` streamed once
    at `bandwidth`, its epilogue's store of the output (`stored` steps), and, split, its fp32 partials'
    passes over the output (`passes` steps each) and their sum's `launch` (in steps)."""

    output: int
    wave: int
    steps: int
    multiprocessors: int
    operand_bytes: float
    bandwidth: float
    step_seconds: float
    passes: float
    launch: float
    stored: float
    drift_horizon: float = 0.0
    drift_extra: float = 0.0

    def walk(self, splits: int) -> float:
        """The busiest program's walk at `splits` slices a tile, in steps: its waves' share of every tile's
        slices, each step at its share of its multiprocessor's tensor rate (the programs a wave runs on each
        at once), or its multiprocessor's share of the operands' stream where that is longer: each
        multiprocessor streams at its share of the memory's rate, its programs' shares in turn, so a wave of
        more programs than multiprocessors but not twice them streams two shares on some. For a `walks_scoped`
        full-wave dense walk whose K-slice passes its L2 drift horizon (`drift_horizon`, `tail.snaked`, row
        5363), steps past the horizon add the unshared L2-miss penalty (`drift_extra`). Priced at a rate
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
        slice_steps = -(-self.steps // splits)
        busiest = waves * slice_steps * sharing
        # The busiest multiprocessor's programs' shares of the operands, at its share of the memory's rate.
        rate = self.bandwidth / self.multiprocessors
        stream = waves * sharing * self.operand_bytes / programs / rate / self.step_seconds
        base = max(busiest, stream)
        if self.drift_horizon > 0.0 and slice_steps > self.drift_horizon:
            base += waves * (slice_steps - self.drift_horizon) * self.drift_extra
        return base

    def seconds(self, splits: int) -> float:
        """The call's time at `splits` slices a tile: its busiest program's walk, its epilogue's store of the
        output after its products (`stored`: a consumer stores its tile once its products are done), and,
        split, its partials' 2s passes over the output and their sum's launch. Priced at its walk alone, a
        GH200's input gradient of three 1024- and 4096-deep products chained through their fp32 output
        (each adding the last's in its epilogue) took 1.145 ms against the 0.955 its walks priced, where one
        6144-deep product, its output stored once, took the 0.855 its walk's 0.790 priced (M-EM-70d)."""
        added = 2 * splits * self.passes + self.launch if splits > 1 else 0.0
        return (self.walk(splits) + self.stored + added) * self.step_seconds

    def underfills(self, splits: int) -> bool:
        """Whether the call's programs at `splits` slices a tile fill less than one wave, so its time is its
        busiest program's walk, not its tiles' reads a FLOP, which `derived` balances for a full one."""
        return self.output * splits < self.wave


def split_walk(tiles: "Tiles", *, groups: int, rows: int, depth: int, width: int, itemsizes: tuple[int, int],
               stored: float, multiprocessors: int, step_seconds: float | None, bandwidth: float | None,
               launch_seconds: float | None, partial_rows: int | None = None,
               target: generation.GpuTarget | None = None) -> SplitWalk | None:
    """Dew: the walk `split_count` prices at `tiles` (its arguments'), None where the device states no rate
    or the call has no whole tile: its wave every program a multiprocessor runs at once, its output tiles
    each program's (a pair's CTA's its rows by the pair's columns, `tile_columns`), `step_seconds` a step at
    the multiprocessor's whole rate (`SplitWalk.walk` shares it among the programs running), its output
    stored at `stored` bytes an element (`SplitWalk.seconds`, rows 10151, 12006)."""
    family = family_of(tiles, jnp.bfloat16, target)
    columns = tile_columns(tiles, family)
    grid_m, grid_n = -(-depth // tiles.block_m), width // columns
    output = groups * grid_m * grid_n
    steps = -(-rows // max(groups, 1) // tiles.block_k)
    if step_seconds is None or bandwidth is None or output == 0:
        return None
    wave = multiprocessors * tiles.programs
    drift_horizon, drift_extra = 0.0, 0.0
    if (groups == 1 and target is not None and target.l2_bytes is not None
            and family.walks_scoped and output >= wave):
        row_idx, col_idx = tail_of.snaked(np.arange(wave), (grid_m, grid_n), 1, max(tiles.grid_block_n, 1))
        row_bytes = tiles.block_m * tiles.block_k * itemsizes[0]
        col_bytes = columns * tiles.block_k * itemsizes[1]
        shared_step = len(np.unique(row_idx)) * row_bytes + len(np.unique(col_idx)) * col_bytes
        unshared_step = wave * (row_bytes + col_bytes)
        # M-FS-6 (A100 SXM, 40 MB L2): wave SMs drift past L2 reuse window after ~6x L2/shared_step steps.
        drift_horizon = 6.0 * target.l2_bytes / max(shared_step, 1)
        drift_extra = max(0.0, unshared_step / bandwidth / step_seconds - tiles.programs)
    held = groups * depth if partial_rows is None else partial_rows
    return SplitWalk(output, wave, steps, multiprocessors,
                     rows * (depth * itemsizes[0] + width * itemsizes[1]), bandwidth, step_seconds,
                     held * width * 4 / bandwidth / step_seconds,
                     0.0 if launch_seconds is None else launch_seconds / step_seconds,
                     held * width * stored / bandwidth / step_seconds,
                     drift_horizon, drift_extra)


def decode_contraction(tiles: "Tiles", target, *, rows: int, groups: int, depth: int, width: int,
                       columns: int | None, held: int, operands: tuple, stored: float,
                       multiprocessors: int) -> dict:
    """A product's walk at `tiles` as `split_walk` and `split_count` price it (their keywords): swapped
    (`Tiles.swapped`), its matrix the rows (`width`) over its `depth`, the call's `rows` its padded `columns`;
    multi-row-tile calls (`rows > groups * block_m`), all `rows` over `depth` (row 10151); else each
    active group's (at most `rows` of `groups`) one row tile over `depth`, its partials `held` rows."""
    itemsizes, decoded = operands
    if tiles.swapped:
        assert columns is not None, "a swapped walk's columns are the call's rows, padded"
        shape = {'groups': 1, 'rows': depth, 'depth': width, 'width': columns}
    elif groups > 0 and rows > groups * tiles.block_m:
        shape = {'groups': groups, 'rows': depth * groups, 'depth': -(-rows // groups), 'width': width,
                 'partial_rows': rows}
    else:
        active = min(groups, rows)
        shape = {'groups': active, 'rows': depth * active, 'depth': tiles.block_m, 'width': width,
                 'partial_rows': held}
    unswapped = dataclasses.replace(tiles, swapped=False)
    return {**shape, 'itemsizes': itemsizes, 'stored': stored, 'multiprocessors': multiprocessors,
            'step_seconds': step_seconds(unswapped, target, itemsizes, decoded),
            'bandwidth': target.memory_rate, 'launch_seconds': target.launch_seconds, 'target': target}


def walk_seconds(tiles: "Tiles", contraction: dict) -> float:
    """The seconds `tiles`' walk of a `contraction` (`decode_contraction`) takes at its split, past every
    other walk's where the device states no rate (`split_walk`)."""
    walk = split_walk(dataclasses.replace(tiles, swapped=False), **contraction)
    return math.inf if walk is None else walk.seconds(tiles.splits)
