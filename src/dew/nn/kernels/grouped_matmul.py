"""The grouped matmul's kernel launches, which the GEMM template (`templates.gemm.template`) runs and
differentiates (`dew.nn.moe.expert_projection` through it), and XLA's portable projection.

Three products, in MaxText's megablox shape: the forward `gmm`, the input gradient `gmm` against the
transposed kernel, and the kernel gradient `tgmm`. Each is a template launch (`templates.base`) with
two implementations, chosen per call by the cost model (`templates.cost`) among those that run it:
JAX's Pallas Triton kernels (`ragged_dot.py`, vendored), on every CUDA GPU from sm80 and on a CPU that
interprets them, and Dew's Pallas Mosaic GPU kernels for Hopper (`templates.gemm.mosaic_gpu`), whose
warpgroup MMA Triton's lowering does not reach. `ragged_dot_refusal` says which products they compute
exactly. They run only for a CUDA lowering, chosen where the call lowers
(`jax.lax.platform_dependent`), so a CPU computation on a GPU host lowers to `jax.lax.ragged_dot`; an
explicit request interprets them on the CPU, as the CPU suite checks them.

Each launch's config is its cost model's fastest that fits the GPU, the same on every run; a tuning
run measures its space (`tuning.tune`). JAX's one tile for every shape left a wide expert's products
at half an A100's cuBLAS rate (a 4096 by 14336 expert's forward 10.4 ms where 5.3 ms is the tuned
tile's, the kernel gradient 18.3 against 8.4).
"""

import abc
import dataclasses
import functools
import itertools
import math
from collections.abc import Hashable
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax._src.pallas.mosaic import core as tpu_core, lowering as mosaic_tpu_lowering
from jax._src.pallas.mosaic_gpu import (
    core as plgpu_core,  # the lowering's semantics (`_lowers`)
    lowering as mosaic_gpu_lowering,
)
from jax.core import ShapedArray
from jax.custom_derivatives import SymbolicZero
from jax.experimental.pallas import mosaic_gpu as plgpu, tpu as pltpu
from jax.extend import core as jax_core

from ..precision import rounded_operand
from . import generation
from .templates import base, cost, hopper
from .templates.gemm import mosaic_gpu, mosaic_tpu
from .templates.product import declared


def ragged_dot_refusal(compute: Dtype, operands: tuple[Dtype, ...],
                       precision: PrecisionLike) -> str | None:
    """Why the kernels do not compute the product a caller asked for, or None where they do: the
    declared product's refusal (`templates.product.declared`)."""
    return declared(compute, precision, operands=operands).refusal


@dataclasses.dataclass(frozen=True)
class Tile:
    """One Triton launch's output tile, its depth step, and its warps and pipeline stages."""
    block_m: int
    block_n: int
    block_k: int
    warps: int
    stages: int


TILES = tuple(Tile(*shape) for shape in itertools.product((64, 128, 256), (128, 256), (128, 64, 32), (8, 4),
                                                            (3, 4, 2)))
"""Every tile the Triton launches may take: the ones the A100 and H100 sweeps found fastest lie inside.
Where the cost model ties, the first wins: the deeper step, which synchronizes its pipeline fewer times
a tile (at 128 x 256 on an H100, a step of 64 took 1.56 ms where 32 took 1.78, GMM-h100-2), and the
more warps."""

_COOPERATIVE = itertools.product((128,), (256, 128), (64,), (4, 3, 5), (8, 4, 2, 1), (1, 2))
_ALONE = itertools.product((128, 192, 64), (128, 192, 64), (64,), (4, 5, 2), (8, 4, 2, 1))
MOSAIC_TILES = (tuple(mosaic_gpu.Tiles(*shape) for shape in _ALONE)
                + tuple(mosaic_gpu.Tiles(*shape, warpgroups=2, cluster=pair)
                        for *shape, pair in _COOPERATIVE))
"""Every tile the Mosaic GPU launches may take: the sweep JAX ships with its one-warpgroup kernels
(`ragged_dot_mgpu.main`), where an H100's fastest lay (GMM-h100-3), then tiles two consumer warpgroups
compute, CUTLASS's cooperative sm90 schedule (128 x 256, a pair of multiprocessors sharing the blocks of
one operand). The cost model prices no term the cooperative schedule or a pair changes yet (its MMA
issue, the second consumer's reads of the shared stage, the pair's multicast), so where it ties, the
first wins: a one-warpgroup tile, then a cooperative tile alone (shared memory staging's barriers, a few
hundred bytes, are not counted against the block either)."""

TPU_BLOCKS = tuple(mosaic_tpu.Blocks(*shape) for shape in itertools.product((128, 256, 512), repeat=3))
"""Every block the TPU launches may take: multiples of the MXU's 128 (Megablox's default is 128 cubed),
none past the 16 MiB a v5e's kernel holds at XLA's default (`_MosaicTpu.staged_bytes`)."""

INTERPRETED = plgpu.InterpretGPUParams(out_of_bounds_reads="uninitialized")
"""How a CPU interprets the Mosaic GPU launches: a read past an operand's rows, which a TMA load zeroes,
gives the poison every buffer starts with (NaN), as an output a kernel fails to store does, so the
interpreter shows both."""

TPU_INTERPRETED = pltpu.InterpretParams()
"""How a CPU interprets the TPU launches: as a TPU runs them, its DMAs and VMEM, every buffer starting
as NaN and a read out of bounds raising."""


def _mode(dtype) -> str:
    """The tensor cores' arithmetic for `dtype` operands (`GpuTarget.tensor_rates`): 16-bit products
    summed in fp32, or TF32 for fp32."""
    return '16bit' if jnp.dtype(dtype).itemsize == 2 else 'tf32'


def _row_tiles(rows: int, groups: int, tile: int) -> float:
    """The tiles of `tile` rows that cover `rows` rows split into `groups` contiguous groups, each from
    its first row, expected over a routing of unknown sizes: each of the nonempty groups (at most
    `rows`), of `rows / nonempty` rows on average, takes one tile, or a tile per `tile` rows and half a
    partial tile more; no count leaves `[ceil(rows / tile), min(rows, ceil(rows / tile) + groups)]`."""
    least = math.ceil(rows / tile)
    nonempty = min(groups, rows)
    expected = nonempty * max(1.0, rows / (nonempty * tile) + 0.5)
    return min(max(expected, least), rows, least + groups)


def _block_tiles(rows: int, groups: int, tile: int) -> int:
    """The tiles of `tile` rows block-aligned launches run over `rows` rows in `groups` groups: each
    block once, and once more for each group boundary inside a block (`mosaic_gpu.GroupInfo`), at most
    one for each group after the first, and never more tiles than rows."""
    return min(math.ceil(rows / tile) + max(min(groups, rows) - 1, 0), rows)


def _routed(rows: int, groups: int) -> float:
    """The share of `groups` matrices a launch over `rows` rows reads: at most one group a row."""
    return min(groups, rows) / groups


@dataclasses.dataclass(frozen=True)
class Hooks:
    """A launch's prologue and epilogue (`templates.gemm`): each a JAX function of a tile and its side
    tiles, or None, with each side operand's launch axes ('rows', 'depth', 'columns'), the positions of
    each side operand's dimensions of one the launch squeezed out (the hook gets them back), and the
    dtypes of the epilogue's outputs where it returns several. The side operands themselves are the
    launch's `prologue_operands` and `epilogue_operands`. A measured pick stores a module-level hook
    (`tuning.encoded`)."""
    prologue: object = None
    prologue_axes: tuple = ()
    epilogue: object = None
    epilogue_axes: tuple = ()
    out_dtypes: tuple | None = None
    prologue_broadcast: tuple = ()
    epilogue_broadcast: tuple = ()

    def before(self):
        """The prologue as a kernel traces it: on the squeezed side tiles."""
        return _unsqueezing(self.prologue, self.prologue_broadcast)

    def after(self):
        """The epilogue as a kernel traces it: on the squeezed side tiles."""
        return _unsqueezing(self.epilogue, self.epilogue_broadcast)

    def kernel_options(self) -> dict:
        return {'prologue': self.before(), 'prologue_axes': self.prologue_axes, 'epilogue': self.after(),
                'epilogue_axes': self.epilogue_axes, 'out_dtypes': self.out_dtypes}


def _unsqueezing(hook, positions: tuple):
    """`hook` given its side tiles' dimensions of one back at `positions` (one tuple per side operand)."""
    if hook is None or not any(positions):
        return hook
    return functools.partial(_unsqueezed, hook, positions)


def _unsqueezed(hook, positions: tuple, tile, *sides):
    """`hook` on `tile` and its side tiles with their dimensions of one back at `positions`."""
    return hook(tile, *(jnp.expand_dims(side, where) if where else side
                        for side, where in zip(sides, positions, strict=True)))


NO_HOOKS = Hooks()

EXACT_STEP = 64
"""The most of the contraction one step of a launch with `exact_sums` takes: the tensor cores truncate
each add within a step's MMA chain, and exact sums promote only between steps."""

TRANSCENDENTALS = ('exp', 'exp2', 'log', 'logistic', 'rsqrt', 'sqrt', 'tanh', 'erf', 'erfc', 'erf_inv', 'pow')
"""The primitives a multiprocessor's special-function units evaluate (`cost.EXPONENTIALS_PER_CLOCK`)."""


def _transcendentals(hook, tile, sides) -> float:
    """The special-function evaluations `hook` makes per element of `tile`, counted in its jaxpr on
    shapes alone: each transcendental primitive's output elements over the tile's."""
    if hook is None:
        return 0.0
    jaxpr = jax.make_jaxpr(hook)(tile, *sides)
    avals = (var.aval for equation in _equations(jaxpr.jaxpr) if equation.primitive.name in TRANSCENDENTALS
             for var in equation.outvars)
    counted = sum(math.prod(aval.shape) for aval in avals if isinstance(aval, ShapedArray))
    return counted / math.prod(tile.shape)


def _equations(jaxpr):
    """Every equation of `jaxpr` and of the programs nested in it, at any depth."""
    yield from jaxpr.eqns
    for inner in jax_core.subjaxprs(jaxpr):
        yield from _equations(inner)


def _lowers(hook, tile, sides, rules) -> bool:
    """Whether a Pallas lowering's `rules` hold a rule for every primitive of `hook` on `tile` and its
    side tiles: Mosaic GPU's, at warpgroup semantics, have none for a transpose, which Triton's lowering
    takes."""
    if hook is None:
        return True
    return all(equation.primitive in rules
               for equation in _equations(jax.make_jaxpr(hook)(tile, *sides).jaxpr))


def _hooks_lower(ba, dtype, rules) -> bool:
    """Whether a lowering's `rules` take the launch's hooks (`_lowers`), traced on 128-wide tiles: the
    primitives a hook makes do not depend on its tile's extents."""
    hooks = ba.arguments['hooks']

    def tiles(name, axes_of):
        return [jax.ShapeDtypeStruct((128,) * len(axes), side.dtype)
                for side, axes in zip(ba.arguments.get(name, ()), axes_of, strict=True)]

    return (_lowers(hooks.before(), jax.ShapeDtypeStruct((128, 128), dtype),
                    tiles('prologue_operands', hooks.prologue_axes), rules)
            and _lowers(_traced_epilogue(ba), jax.ShapeDtypeStruct((128, 128), jnp.float32),
                        tiles('epilogue_operands', hooks.epilogue_axes), rules))


def _traced_epilogue(ba):
    """The launch's epilogue as a function of one accumulator tile, as `_lowers` and `_hook_work` read it:
    a paired product's reads that tile for both of its accumulators, the same primitives on the same
    shapes."""
    epilogue = ba.arguments['hooks'].after()
    if epilogue is None or ba.arguments.get('paired') is None:
        return epilogue
    return lambda acc, *sides: epilogue(acc, acc, *sides)


def _hook_work(hook, dtype, sides, axes_of, tile: tuple, extents: dict) -> float:
    """`_transcendentals` of `hook` on a `tile` of `dtype` and its side tiles at `extents`."""
    if hook is None:
        return 0.0
    side_tiles = [jax.ShapeDtypeStruct(tuple(extents[axis] for axis in axes), side.dtype)
                  for side, axes in zip(sides, axes_of, strict=True)]
    return _transcendentals(hook, jax.ShapeDtypeStruct(tile, dtype), side_tiles)


def _side_bytes(operands, axes_of, rereads: dict) -> float:
    """The device bytes of the side operands: each read once, and again for each block of a launch axis
    in `rereads` (its block count) that it does not follow, as the launch's own operands are."""
    return sum(operand.size * jnp.dtype(operand.dtype).itemsize
               * math.prod(blocks for axis, blocks in rereads.items() if axis not in axes)
               for operand, axes in zip(operands, axes_of, strict=True))


def _fetched_bytes(operands, axes_of, grid: tuple) -> float:
    """The HBM bytes a TPU launch's pipeline fetches of its side operands over `grid` (its axes, outer to
    inner, each with its block count): a block is fetched again only where its index changes between
    consecutive grid steps, so an operand is read again for each block of the axes it does not follow
    outside the innermost it does."""
    def rereads(axes) -> float:
        innermost = max(index for index, (axis, _) in enumerate(grid) if axis in axes)
        return math.prod(blocks for axis, blocks in grid[:innermost] if axis not in axes)

    return sum(operand.size * jnp.dtype(operand.dtype).itemsize * rereads(axes)
               for operand, axes in zip(operands, axes_of, strict=True))


def _output_bytes(ba) -> int:
    """The bytes of one output element of each of the launch's outputs, together."""
    return sum(jnp.dtype(dtype).itemsize
               for dtype in (ba.arguments['hooks'].out_dtypes or (ba.arguments['out_dtype'],)))


def _side_tile_bytes(operands, axes_of, extents: dict) -> int:
    """The bytes of one tile of each side operand at the block `extents` per launch axis."""
    return sum(math.prod(extents[axis] for axis in axes) * jnp.dtype(operand.dtype).itemsize
               for operand, axes in zip(operands, axes_of, strict=True))


def _warps(rows: int, columns: int, warps: int) -> tuple[int, int]:
    """How Triton lays `warps` warps over a `rows` x `columns` dot (`warpsPerTileV2` in
    lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp): warps along the rows while a warp's 16-row
    slice is no narrower than its 8-column one, then along the columns."""
    along_rows = along_columns = 1
    while along_rows * along_columns < warps:
        if rows // 16 // along_rows >= columns // (8 * along_columns) and along_rows < rows // 16:
            along_rows *= 2
        else:
            along_columns *= 2
    return along_rows, along_columns


def _triton_staged(rows: int, columns: int, depth: int, warps: int, wgmma: bool) -> int:
    """The elements one depth step of a Triton dot of `rows` x `columns` x `depth` moves through shared
    memory: both operand tiles written, then read back by the matrix instructions, each warp its own
    slices (`_warps`), or on Hopper each warpgroup the whole right operand and its rows of the left."""
    if wgmma:
        return (rows + columns) * depth + (rows + max(warps // 4, 1) * columns) * depth
    along_rows, along_columns = _warps(rows, columns, warps)
    return (rows + columns) * depth + depth * (rows * along_columns + columns * along_rows)


def _sizes(specs: list[jax.ShapeDtypeStruct], seed: int, rows: int) -> jax.Array:
    """Group sizes summing to `rows` over the groups the third spec (the launch's group sizes) names,
    drawn unevenly, one group left empty, as a step routes them."""
    groups = specs[2].shape[0]
    shares = np.random.default_rng(seed).dirichlet(np.ones(groups))
    shares[0] = 0.0
    sizes = np.floor(shares / shares.sum() * rows).astype(np.int32)
    sizes[-1] += rows - sizes.sum()
    return jnp.asarray(sizes)


class _GroupedLaunch[Config: Hashable](base.Template[Config]):
    """A grouped launch: its space is a declared tuple, its group sizes drawn as a step routes them."""
    # Annotations stay evaluated (no postponed annotations in this module):
    # tokamax serializes a launch's arguments into the program by their types.

    def keyed(self, ba):
        """`tuning.KernelOp.keyed`, each side operand's dimension that follows the rows bucketed with the
        tokens' (`ROWS`), so the launch a key measures loads side tiles over all its rows."""
        keyed = super().keyed(ba)
        rows = keyed.arguments['tokens'].shape[0]
        hooks = ba.arguments.get('hooks', NO_HOOKS)
        arguments = dict(keyed.arguments)
        for name, axes_of in (('prologue_operands', hooks.prologue_axes),
                              ('epilogue_operands', hooks.epilogue_axes)):
            if name in arguments:
                arguments[name] = tuple(
                    jax.ShapeDtypeStruct(tuple(rows if axis == 'rows' else extent
                                               for axis, extent in zip(axes, side.shape, strict=True)),
                                         side.dtype)
                    for side, axes in zip(arguments[name], axes_of, strict=True))
        return type(keyed)(keyed.op, arguments)

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """`tuning.KernelOp.operands`, the group sizes (every launch's third argument) drawn as a step
        routes them, an integer side operand drawn as any other operand."""
        arrays = super().operands(specs, seed)
        arrays[2] = _sizes(specs, seed, specs[0].shape[0])
        return arrays


class _Triton(_GroupedLaunch[Tile]):
    """JAX's Pallas Triton kernels (`ragged_dot`), on every CUDA GPU from sm80, and on a CPU where the
    call interprets them."""
    name: ClassVar[str] = 'triton'
    config_cls: ClassVar[type[Tile]] = Tile

    def eligible(self, ba, target) -> bool:
        # Every CUDA GPU the template sends the kernels to (`templates.gemm.template`), and
        # a CPU, which interprets them, or traces the CUDA branch a GPU's lowering takes; never a TPU.
        cuda = generation.capability(generation.effective(target).generation) is not None
        return cuda and _rated(target, ba.arguments['tokens'].dtype)

    @staticmethod
    @abc.abstractmethod
    def _dense_extents(ba) -> tuple[int, int]:
        """The depth and width a dense call's tiles divide."""

    def space(self, ba) -> tuple[Tile, ...]:
        return TILES

    def fitting(self, ba, target) -> tuple[Tile, ...]:
        """The tiles that fit `target` (`Template.fitting`) whose accumulator (two with `exact_sums`)
        stays in its threads' registers, and on Hopper those whose dot Triton gives a warpgroup MMA: each
        warpgroup's share of
        the output rows a multiple of wgmma's 64. Either other kind ran 1.2 to 8.5 times slower than the
        fastest tile on an H100 (GMM-h100-2: an accumulator of 256 registers a thread spills; a 32-row
        share falls back to Ampere's MMA)."""
        wgmma = generation.effective(target).wgmma

        exact_sums = ba.arguments.get('exact_sums')
        accumulators = 2 if exact_sums else 1

        def fits(tile: Tile) -> bool:
            rows, columns = self.output(tile)
            held = accumulators * rows * columns // (32 * tile.warps) < cost.THREAD_REGISTERS
            # Exact sums promote between depth steps; a step's own MMA chain truncates its adds, so it takes
            # at most EXACT_STEP of the contraction (H100: a 128-row step left 1.55x IEEE's error, a 64-row
            # one 1.03x; GMMGATE-h100-5, GMMDKX-h100).
            short = not exact_sums or self.contraction(tile) <= EXACT_STEP
            return held and short and not (wgmma and rows // max(tile.warps // 4, 1) % 64)

        fitted = tuple(tile for tile in super().fitting(ba, target) if fits(tile))
        if not ba.arguments.get('dense'):
            return fitted
        # A dense call (`triton_dense`) prefers the fitting tiles dividing its depth and width, where any
        # does: the kernel pads to one that does not, a copy the cost model prices (`_padding_bytes`).
        depth, width = self._dense_extents(ba)
        dividing = tuple(tile for tile in fitted if depth % tile.block_k == 0 and width % tile.block_n == 0)
        return dividing or fitted

    @staticmethod
    @abc.abstractmethod
    def output(tile: Tile) -> tuple[int, int]:
        """The rows and columns of a program's output tile, the dot's M and N."""

    @staticmethod
    @abc.abstractmethod
    def contraction(tile: Tile) -> int:
        """The contraction one step of `tile` takes, the dot's K."""

    def staged_bytes(self, ba, config: Tile) -> int:
        # Pallas Triton's pipeline holds `stages - 1` depth steps, and on Hopper `stages` where it
        # pipelines a wgmma, which reads one step while the next loads; an A100 and an H100 refuse
        # exactly those bytes, so a Hopper tile is sized for the larger. The prologue's side tiles load
        # in the same steps.
        held = 0 if generation.gpu_target().wgmma else 1
        step = self.loaded(config) * jnp.dtype(ba.arguments['tokens'].dtype).itemsize + _side_tile_bytes(
            ba.arguments['prologue_operands'], ba.arguments['hooks'].prologue_axes, self.extents(config))
        return step * (config.stages - held)

    @staticmethod
    @abc.abstractmethod
    def extents(tile: Tile) -> dict:
        """The block each launch axis takes at `tile`."""

    def _resident(self, ba, config: Tile, target) -> int:
        """The programs a multiprocessor holds at `config`: by its shared memory, and by its registers,
        each thread holding its share of the fp32 accumulator and of one step's operand fragments."""
        rows, columns = self.output(config)
        threads = 32 * config.warps
        # One MMA's k of 16 of each operand fragment, the warps' shares together.
        fragments = (rows + columns) * 16 // threads
        return cost.resident(target, self.staged_bytes(ba, config), threads,
                             rows * columns // threads + fragments)

    @staticmethod
    @abc.abstractmethod
    def loaded(tile: Tile) -> int:
        """The elements of both operand tiles one depth step loads."""


@dataclasses.dataclass(frozen=True)
class Gmm(_Triton):
    """Each group's rows of `tokens` `[rows, in]` times its matrix of
    `kernel` `[groups, in, out]`, or of its transpose with `trans_rhs`."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    @staticmethod
    def _dense_extents(ba) -> tuple[int, int]:
        """The depth and width of a dense call's product, which `block_k` and `block_n` tile."""
        kernel = ba.arguments['kernel']
        return (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]

    @staticmethod
    def loaded(tile: Tile) -> int:
        return (tile.block_m + tile.block_n) * tile.block_k

    @staticmethod
    def output(tile: Tile) -> tuple[int, int]:
        return tile.block_m, tile.block_n

    @staticmethod
    def contraction(tile: Tile) -> int:
        return tile.block_k

    @staticmethod
    def extents(tile: Tile) -> dict:
        return {'rows': tile.block_m, 'depth': tile.block_k, 'columns': tile.block_n}

    def costed(self, ba, config: Tile, target) -> cost.Cost:
        # A program is one group's chunk of up to `CHUNK_M` rows (ragged_dot.py) by `block_n` columns,
        # its rows `block_m` at a time, each row tile loading the group's `depth` x `block_n` block
        # again. Programs run chunks before columns, so each column block reads the rows again. The
        # rereads come from device memory, not the L2: so counted, the bytes alone gave an H100's time
        # for every tile of Qwen3-30B-A3B's gate_up within 10% (GMM-h100-3).
        from .ragged_dot import CHUNK_M
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        (rows, depth), groups = tokens.shape, kernel.shape[0]
        width = kernel.shape[1] if ba.arguments['trans_rhs'] else kernel.shape[2]
        size = jnp.dtype(tokens.dtype).itemsize
        # A dense call's program is one row tile (`triton_dense`), a ragged one's a chunk.
        chunks = _row_tiles(rows, groups, config.block_m if ba.arguments.get('dense') else CHUNK_M)
        columns = math.ceil(width / config.block_n)
        tiles = _row_tiles(rows, groups, config.block_m)
        flops = 2 * tiles * config.block_m * depth * width
        steps = tiles * math.ceil(depth / config.block_k) / chunks
        staged = steps * _triton_staged(config.block_m, config.block_n, config.block_k, config.warps,
                                        generation.effective(target).wgmma) * size
        hooks, prologue_sides = ba.arguments['hooks'], ba.arguments['prologue_operands']
        epilogue_sides = ba.arguments['epilogue_operands']
        # Side operands are read again as the operands are: with each column block, and with each row
        # tile where they follow no rows; the hooks' special functions on every element they see.
        rereads = {'rows': tiles, 'columns': columns}
        moved = ((columns * rows * depth + tiles * depth * width) * size + rows * width * _output_bytes(ba)
                 + _side_bytes(prologue_sides, hooks.prologue_axes, rereads)
                 + _side_bytes(epilogue_sides, hooks.epilogue_axes, rereads))
        if ba.arguments.get('dense'):
            moved += _padding_bytes((((rows, depth), ('rows', 'depth'), size),
                                     ((depth, width), ('depth', 'width'), size),
                                     ((rows, width), ('rows', 'width'), _output_bytes(ba)),
                                     *_side_arrays(prologue_sides, hooks.prologue_axes),
                                     *_side_arrays(epilogue_sides, hooks.epilogue_axes)), config)
        extents = self.extents(config)
        exponentials = (
            steps * config.block_m * config.block_k * _hook_work(
                hooks.before(), tokens.dtype, prologue_sides, hooks.prologue_axes,
                (config.block_m, config.block_k), extents)
            + tiles / chunks * config.block_m * config.block_n * _hook_work(
                _traced_epilogue(ba), jnp.float32, epilogue_sides, hooks.epilogue_axes,
                (config.block_m, config.block_n), extents))
        return cost.Cost(
            programs=chunks * columns, device_bytes=moved,
            program_flops=flops / (chunks * columns), program_staged=staged,
            program_exponentials=exponentials, mode=_mode(tokens.dtype), steps=steps, stages=config.stages,
            resident=self._resident(ba, config, target))

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             *, out_dtype: str,
             trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: Tile, return_residuals: bool = False):
        # Imported at first use: the kernels need jax's Pallas Triton backend,
        # which jax 0.11.2 deprecates, and importing a model must not load it.
        if dense:
            # One group of every row: the plain tiled kernel, which finds no group and masks no row.
            from .templates.gemm import triton_dense
            options = hooks.kernel_options()
            outs = triton_dense.gmm(tokens, kernel[0], tile=config, transpose_rhs=trans_rhs,
                                    out_dtypes=tuple(jnp.dtype(d) for d in (options.pop('out_dtypes')
                                                                            or (out_dtype,))),
                                    interpret=interpret, prologue_operands=prologue_operands,
                                    epilogue_operands=epilogue_operands, exact_sums=exact_sums, paired=paired,
                                    **options)
            return (outs if hooks.out_dtypes else outs[0]), None
        from . import ragged_dot
        return ragged_dot.gmm(tokens, kernel, sizes, block_m=config.block_m, block_k=config.block_k,
                              block_n=config.block_n, num_warps=config.warps, num_stages=config.stages,
                              trans_rhs=trans_rhs, interpret=interpret, compute_dtype=tokens.dtype,
                              out_dtype=jnp.dtype(out_dtype), prologue_operands=prologue_operands,
                              epilogue_operands=epilogue_operands, exact_sums=exact_sums, paired=paired,
                              **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class Tgmm(_Triton):
    """Each group's `tokens` rows transposed times its `cotangent` rows: the
    kernel gradient `[groups, in, out]`."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    @staticmethod
    def _dense_extents(ba) -> tuple[int, int]:
        """The depth and width of a dense call's gradient, which `block_k` and `block_n` tile."""
        return ba.arguments['tokens'].shape[1], ba.arguments['cotangent'].shape[1]

    @staticmethod
    def loaded(tile: Tile) -> int:
        return tile.block_m * (tile.block_k + tile.block_n)

    @staticmethod
    def output(tile: Tile) -> tuple[int, int]:
        return tile.block_k, tile.block_n

    @staticmethod
    def contraction(tile: Tile) -> int:
        return tile.block_m

    @staticmethod
    def extents(tile: Tile) -> dict:
        return {'rows': tile.block_m, 'depth': tile.block_k}

    def costed(self, ba, config: Tile, target) -> cost.Cost:
        # A program is one group's `block_k` x `block_n` output tile, summed over the group's rows
        # `block_m` at a time; each output column block reads the rows again, and each output row
        # block the cotangent, from device memory (`Gmm.cost`).
        tokens, cotangent, sizes = ba.arguments['tokens'], ba.arguments['cotangent'], ba.arguments['sizes']
        (rows, depth), width, groups = tokens.shape, cotangent.shape[1], sizes.shape[0]
        size, out = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(ba.arguments['out_dtype']).itemsize
        blocks = math.ceil(depth / config.block_k), math.ceil(width / config.block_n)
        programs = groups * blocks[0] * blocks[1]
        steps = _row_tiles(rows, groups, config.block_m) / groups
        flops = 2 * steps * config.block_m * config.block_k * config.block_n
        staged = steps * _triton_staged(config.block_k, config.block_n, config.block_m, config.warps,
                                        generation.effective(target).wgmma) * size
        hooks, sides = ba.arguments['hooks'], ba.arguments['prologue_operands']
        rereads = {'rows': groups, 'depth': blocks[0], 'columns': blocks[1]}
        moved = ((blocks[1] * rows * depth + blocks[0] * rows * width) * size + groups * depth * width * out
                 + _side_bytes(sides, hooks.prologue_axes, rereads))
        if ba.arguments.get('dense'):
            moved += _padding_bytes((((rows, depth), ('rows', 'depth'), size),
                                     ((rows, width), ('rows', 'width'), size),
                                     ((depth, width), ('depth', 'width'), out),
                                     *_side_arrays(sides, hooks.prologue_axes)), config)
        exponentials = steps * config.block_m * config.block_k * _hook_work(
            hooks.before(), tokens.dtype, sides, hooks.prologue_axes, (config.block_m, config.block_k),
            self.extents(config))
        return cost.Cost(
            programs=programs, device_bytes=moved, program_flops=flops, program_staged=staged,
            program_exponentials=exponentials, mode=_mode(tokens.dtype), steps=steps, stages=config.stages,
            resident=self._resident(ba, config, target))

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: Tile, return_residuals: bool = False):
        if dense:
            # One group of every row: the plain tiled kernel (`Gmm._fwd`).
            from .templates.gemm import triton_dense
            return triton_dense.tgmm(tokens, cotangent, tile=config, out_dtype=jnp.dtype(out_dtype),
                                     interpret=interpret, prologue=hooks.before(),
                                     prologue_operands=prologue_operands, prologue_axes=hooks.prologue_axes,
                                     exact_sums=exact_sums)[None], None
        from . import ragged_dot
        return ragged_dot.tgmm(tokens, cotangent, sizes, block_m=config.block_m, block_k=config.block_k,
                               block_n=config.block_n, num_warps=config.warps, num_stages=config.stages,
                               interpret=interpret, compute_dtype=tokens.dtype,
                               out_dtype=jnp.dtype(out_dtype), prologue=hooks.before(),
                               prologue_operands=prologue_operands, prologue_axes=hooks.prologue_axes,
                               exact_sums=exact_sums), None


class _Mosaic(_GroupedLaunch[mosaic_gpu.Tiles]):
    """JAX's Pallas Mosaic GPU kernels (`templates.gemm.mosaic_gpu`): Hopper's warpgroup MMA, one 16-bit
    dtype, and widths the space's tiles divide. Each runs one warpgroup a multiprocessor, or two
    consumers beside a producer (a cooperative tile), persistent over the output tiles in a planar snake,
    so the rows a band of `grid_block_n` column tiles shares and the matrix block the row tiles in flight
    share come from the L2."""
    name: ClassVar[str] = 'mosaic_gpu'
    config_cls: ClassVar[type[mosaic_gpu.Tiles]] = mosaic_gpu.Tiles
    EXACT_SUMS: ClassVar[bool] = False
    """Whether the kernel adds each depth step's product apart (`exact_sums`); one that does not is not
    eligible where the call asks for it."""

    def eligible(self, ba, target) -> bool:
        first, second = (value.dtype for name, value in ba.arguments.items()
                         if name in ('tokens', 'kernel', 'cotangent'))
        # Interpreted only in a CPU process: the interpreter's callbacks are ordered effects, which a GPU
        # process's lowering of the platform choice cannot carry in the CPU branch it does not take.
        interpretable = not ba.arguments['interpret'] or jax.default_backend() == 'cpu'
        return (target.wgmma and first == second and jnp.dtype(first).itemsize == 2 and interpretable
                and _rated(target, first) and bool(self.space(ba)) and self._hooks_lower(ba, first)
                and (self.EXACT_SUMS or not ba.arguments.get('exact_sums')))

    @staticmethod
    def _hooks_lower(ba, dtype) -> bool:
        """Whether Mosaic GPU's lowering takes the launch's hooks (`_hooks_lower`). A side operand the
        launch squeezes and gives back with its dimension of one (`Hooks.prologue_broadcast`) has no
        layout Mosaic's inference finds for the size-one dimension, so a hook taking one does not lower
        there (GMMGATE-h100-8), where Triton's lowering takes it."""
        hooks = ba.arguments['hooks']
        if hooks.prologue_broadcast or hooks.epilogue_broadcast:
            return False
        return _hooks_lower(ba, dtype, mosaic_gpu_lowering.mosaic_lowering_rules[plgpu_core.WGxWG_SEMANTICS])

    def staged_bytes(self, ba, config: mosaic_gpu.Tiles) -> int:
        # The pipeline's stages of both operand blocks and the prologue's side tiles, the epilogue's side
        # tiles, and the output block each output stores from.
        size = jnp.dtype(ba.arguments['tokens'].dtype).itemsize
        hooks = ba.arguments['hooks']
        outs = [jnp.dtype(dtype).itemsize for dtype in (hooks.out_dtypes or (ba.arguments['out_dtype'],))]
        extents = self.extents(config)
        step = (config.block_m + config.block_n) * config.block_k * size + _side_tile_bytes(
            ba.arguments['prologue_operands'], hooks.prologue_axes, extents)
        epilogue = _side_tile_bytes(ba.arguments.get('epilogue_operands', ()), hooks.epilogue_axes, extents)
        # One warpgroup stages each output in turn; two hold every output's staging buffer at once, and
        # each consumer its own copy of an epilogue side tile without rows (`hopper.staging`; a
        # prologue side's per-consumer copies are within the stages' count of it).
        staged_outs = sum(outs) if config.warpgroups == 2 else max(outs)
        if config.warpgroups == 2:
            epilogue_sides = ba.arguments.get('epilogue_operands', ())
            epilogue += _side_tile_bytes(*_uncut(epilogue_sides, hooks.epilogue_axes, 'rows'), extents)
        return config.stages * step + epilogue + staged_outs * config.block_m * config.block_n

    @staticmethod
    @abc.abstractmethod
    def extents(tile: mosaic_gpu.Tiles) -> dict:
        """The block each launch axis takes at `tile`."""

    def _cost(self, ba, config: mosaic_gpu.Tiles, tiles: float, steps: float, moved: float) -> cost.Cost:
        """`tiles` output tiles of `steps` contraction steps each, `moved` device bytes."""
        tokens, hooks = ba.arguments['tokens'], ba.arguments['hooks']
        size = jnp.dtype(tokens.dtype).itemsize
        out = jnp.dtype(ba.arguments['out_dtype']).itemsize
        extents = self.extents(config)
        prologue_sides = ba.arguments['prologue_operands']
        epilogue_sides = ba.arguments.get('epilogue_operands', ())
        lhs = (extents['rows'], extents['depth'])
        # Each step's blocks are written by TMA and read once by the warpgroup's MMA. A prologue reads
        # the stage of `lhs` and its side tiles and writes the stage back before the MMA may read it
        # (a commit to shared memory, and the consumers' meet): serial with the MMAs, not beside them
        # (GMMSB-h100-main: a gated prologue at 3.7x the Triton launch's time).
        step = 2 * (config.block_m + config.block_n) * config.block_k * size
        rewritten = 0.0
        if hooks.prologue is not None:
            rewritten = steps * (2 * math.prod(lhs) * size
                                 + 2 * _side_tile_bytes(prologue_sides, hooks.prologue_axes, extents))
        prologue_exponentials = steps * math.prod(lhs) * _hook_work(
            hooks.before(), tokens.dtype, prologue_sides, hooks.prologue_axes, lhs, extents)
        epilogue_exponentials = config.block_m * config.block_n * _hook_work(
            _traced_epilogue(ba), jnp.float32, epilogue_sides, hooks.epilogue_axes,
            (config.block_m, config.block_n), extents)
        return cost.Cost(
            programs=tiles, device_bytes=moved,
            program_flops=2 * steps * config.block_m * config.block_n * config.block_k,
            program_staged=steps * step + config.block_m * config.block_n * out,
            program_exponentials=epilogue_exponentials, mode=_mode(tokens.dtype), steps=steps,
            stages=config.stages, serial_staged=rewritten, serial_exponentials=prologue_exponentials)

    def _multiprocessors(self, interpret: bool) -> int:
        # The interpreter runs one block (jax/_src/pallas/mosaic_gpu/interpret).
        return 1 if interpret else _multiprocessors(generation.gpu_target())

    @staticmethod
    def _in_flight(config: mosaic_gpu.Tiles, target) -> float:
        """The row tiles a band of the planar snake keeps in flight at once on `target`."""
        return max(1.0, _multiprocessors(generation.effective(target)) / config.grid_block_n)


@dataclasses.dataclass(frozen=True)
class MosaicGmm(_Mosaic):
    """`Gmm` on Hopper (`mosaic_gpu.gmm`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    def space(self, ba) -> tuple[mosaic_gpu.Tiles, ...]:
        kernel = ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return tuple(tile for tile in _modelled(ba)
                     if depth % tile.block_k == 0 and width % tile.block_n == 0)

    @staticmethod
    def extents(tile: mosaic_gpu.Tiles) -> dict:
        return {'rows': tile.block_m, 'depth': tile.block_k, 'columns': tile.block_n}

    def costed(self, ba, config: mosaic_gpu.Tiles, target) -> cost.Cost:
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        (rows, depth), groups = tokens.shape, kernel.shape[0]
        width = kernel.shape[1] if ba.arguments['trans_rhs'] else kernel.shape[2]
        size = jnp.dtype(tokens.dtype).itemsize
        # A pair of multiprocessors runs two row tiles of one group, its second empty past the group.
        pair = config.cluster
        tiles, columns = pair * _block_tiles(rows, groups, pair * config.block_m), width // config.block_n
        # The rows are read once a band; a group's matrix once for the row tiles of it in flight (a pair's
        # multicast saves the L2's reads, not the device's).
        bands = math.ceil(columns / config.grid_block_n)
        in_flight = self._in_flight(config, target)
        per_group = tiles / max(min(groups, rows), 1)
        matrices = _routed(rows, groups) * kernel.size * size * math.ceil(per_group / in_flight)
        hooks = ba.arguments['hooks']
        rereads = {'rows': tiles, 'columns': bands}
        moved = (bands * rows * depth * size + matrices + rows * width * _output_bytes(ba)
                 + _side_bytes(ba.arguments['prologue_operands'], hooks.prologue_axes, rereads)
                 + _side_bytes(ba.arguments['epilogue_operands'], hooks.epilogue_axes, rereads))
        return self._cost(ba, config, tiles * columns, depth / config.block_k, moved)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             *, out_dtype: str,
             trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: mosaic_gpu.Tiles, return_residuals: bool = False):
        # Never set: the launch is not eligible for exact sums; a dense call is one group, as any.
        del exact_sums, dense
        return mosaic_gpu.gmm(tokens, kernel, sizes, tiles=config, transpose_rhs=trans_rhs,
                              out_dtype=out_dtype, multiprocessors=self._multiprocessors(interpret),
                              interpret=INTERPRETED if interpret else False,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              paired=paired, **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class MosaicTgmm(_Mosaic):
    """`Tgmm` on Hopper (`mosaic_gpu.tgmm`): `block_m` x `block_n` tiles of each group's gradient, its rows
    `block_k` at a time."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')
    EXACT_SUMS: ClassVar[bool] = True

    def space(self, ba) -> tuple[mosaic_gpu.Tiles, ...]:
        depth, width = ba.arguments['tokens'].shape[1], ba.arguments['cotangent'].shape[1]
        # No pair: a tile streaming its stages across tiles outruns the paired tiles' kernel, whose pair
        # never paid for its multicast here.
        return base.dividing(tuple(tile for tile in _modelled(ba) if tile.cluster == 1), (depth, width),
                             lambda tile: (tile.block_m, tile.block_n))

    def fitting(self, ba, target) -> tuple[mosaic_gpu.Tiles, ...]:
        """The tiles that fit `target` (`Template.fitting`), with `exact_sums` those whose two fp32
        accumulators, the running sum and a step's, stay in each consumer warpgroup's registers (a
        warpgroup's 128 threads hold its share of the tile's columns)."""
        fits = super().fitting(ba, target)
        if not ba.arguments.get('exact_sums'):
            return fits
        # A cooperative consumer has the registers the producer's 40 leave it (JAX's
        # `emit_pipeline_warp_specialized`: 232).
        return tuple(tile for tile in fits
                     if 2 * tile.block_m * tile.block_n // (128 * tile.warpgroups)
                     < (hopper.CONSUMER_REGISTERS if tile.warpgroups == 2 else cost.THREAD_REGISTERS))

    @staticmethod
    def extents(tile: mosaic_gpu.Tiles) -> dict:
        return {'rows': tile.block_k, 'depth': tile.block_m}

    def costed(self, ba, config: mosaic_gpu.Tiles, target) -> cost.Cost:
        tokens, cotangent, sizes = ba.arguments['tokens'], ba.arguments['cotangent'], ba.arguments['sizes']
        (rows, depth), width, groups = tokens.shape, cotangent.shape[1], sizes.shape[0]
        size, out = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(ba.arguments['out_dtype']).itemsize
        blocks = depth // config.block_m, width // config.block_n
        # A group's rows are read once a band of column tiles, its cotangent once for the row tiles in
        # flight.
        bands = math.ceil(blocks[1] / config.grid_block_n)
        in_flight = self._in_flight(config, target)
        hooks = ba.arguments['hooks']
        rereads = {'rows': groups, 'depth': blocks[0], 'columns': bands}
        moved = ((bands * rows * depth + math.ceil(blocks[0] / in_flight) * rows * width) * size
                 + groups * depth * width * out
                 + _side_bytes(ba.arguments['prologue_operands'], hooks.prologue_axes, rereads))
        steps = _block_tiles(rows, groups, config.block_k) / groups
        return self._cost(ba, config, groups * blocks[0] * blocks[1], steps, moved)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: mosaic_gpu.Tiles,
             return_residuals: bool = False):
        del dense  # a dense call is one group, as any
        return mosaic_gpu.tgmm(tokens, cotangent, sizes, tiles=config, out_dtype=out_dtype,
                               multiprocessors=self._multiprocessors(interpret),
                               interpret=INTERPRETED if interpret else False, prologue=hooks.before(),
                               prologue_operands=prologue_operands, prologue_axes=hooks.prologue_axes,
                               exact_sums=exact_sums), None



class _MosaicTpu(_GroupedLaunch[mosaic_tpu.Blocks]):
    """The TPU mainloops (`templates.gemm.mosaic_tpu`, Megablox's schedule): bf16 operands, which the
    MXUs multiply exactly and sum in fp32, widths the space's blocks divide, and a TPU whose scoped VMEM a
    measurement states; a CPU interprets them as the TPU runs them. A TPU's own lowering runs XLA's
    ragged dot until a scoreboard row measures these (`_on_platform` has no TPU branch), so an assumed
    TPU interprets them alone."""
    name: ClassVar[str] = 'mosaic_tpu'
    config_cls: ClassVar[type[mosaic_tpu.Blocks]] = mosaic_tpu.Blocks

    def eligible(self, ba, target) -> bool:
        # Not for exact sums: the MXUs add within a step, in fp32 by Google's account, IEEE by none
        # (`kernel_refusal` routes such a product to XLA).
        first, second = (value.dtype for name, value in ba.arguments.items()
                         if name in ('tokens', 'kernel', 'cotangent'))
        interpretable = not ba.arguments['interpret'] or jax.default_backend() == 'cpu'
        rules = mosaic_tpu_lowering.lowering_rules[tpu_core.CoreType.TC]
        return (generation.tpu(target) and target.scoped_vmem is not None
                and first == second == jnp.bfloat16 and interpretable and _rated(target, first)
                and bool(self.space(ba)) and _hooks_lower(ba, first, rules)
                and not ba.arguments.get('exact_sums'))

    def fitting(self, ba, target) -> tuple[mosaic_tpu.Blocks, ...]:
        """The blocks whose double-buffered operands, outputs and accumulator fit the TPU's scoped VMEM
        (`Template.fitting`'s rule, VMEM for a block's shared memory)."""
        budget = generation.effective(target).scoped_vmem
        if budget is None:
            raise ValueError(f"{target.generation} states no scoped VMEM, which the TPU launches fit their "
                             "blocks to")
        space = self.space(ba)
        fits = tuple(blocks for blocks in space if self.staged_bytes(ba, blocks) <= budget)
        if not fits:
            raise base.Unfitted(type(self).__name__, min((self.staged_bytes(ba, blocks) for blocks in space),
                                                         default=None), budget, target.generation)
        return fits

    def staged_bytes(self, ba, config: mosaic_tpu.Blocks) -> int:
        # Pallas's pipeline holds two of each operand's and output's blocks (one in flight), beside the
        # fp32 accumulator.
        size = jnp.dtype(ba.arguments['tokens'].dtype).itemsize
        hooks = ba.arguments['hooks']
        extents = self.extents(config)
        outs = _output_bytes(ba)
        sides = (_side_tile_bytes(ba.arguments['prologue_operands'], hooks.prologue_axes, extents)
                 + _side_tile_bytes(ba.arguments.get('epilogue_operands', ()), hooks.epilogue_axes, extents))
        operands = self.loaded(config) * size + sides
        rows, columns = self.output(config)
        return 2 * (operands + outs * rows * columns) + 4 * rows * columns

    def _lays_out(self, ba, blocks: mosaic_tpu.Blocks) -> bool:
        """Whether Mosaic TPU lays out every side operand's block at `blocks`: a one-dimensional block
        (a power of two here) holds the operand's whole extent, the rows padded to their block, or at
        least a row of 128 lanes of 32-bit words (`_check_block_mappings` in jax's Pallas TPU lowering,
        256 of a 16-bit operand); a block of two dimensions or more here is of whole multiples of 128."""
        hooks = ba.arguments['hooks']
        extents = self.extents(blocks)
        for name, axes_of in (('prologue_operands', hooks.prologue_axes),
                              ('epilogue_operands', hooks.epilogue_axes)):
            for side, axes in zip(ba.arguments.get(name, ()), axes_of, strict=True):
                if len(axes) != 1:
                    continue
                block = extents[axes[0]]
                whole = -(-side.shape[0] // block) * block if axes[0] == 'rows' else side.shape[0]
                if block != whole and block < 128 * 4 // jnp.dtype(side.dtype).itemsize:
                    return False
        return True

    @staticmethod
    @abc.abstractmethod
    def extents(blocks: mosaic_tpu.Blocks) -> dict:
        """The block each launch axis takes at `blocks`."""

    @staticmethod
    @abc.abstractmethod
    def loaded(blocks: mosaic_tpu.Blocks) -> int:
        """The elements of both operand blocks one grid step loads."""

    @staticmethod
    @abc.abstractmethod
    def output(blocks: mosaic_tpu.Blocks) -> tuple[int, int]:
        """The rows and columns of an output block."""

    @staticmethod
    def _cost(steps: float, programs: float, flops: float, moved: float, dtype) -> cost.Cost:
        """`programs` output blocks of `flops` each, over `steps` grid steps each, `moved` HBM bytes. The MXUs
        read their operands from VMEM as fast as they multiply them, so no VMEM bytes are charged; a
        hook's transcendentals go to a unit whose rate no fact states, so none are either."""
        return cost.Cost(programs=programs, device_bytes=moved, program_flops=flops, mode=_mode(dtype),
                         steps=steps)


@dataclasses.dataclass(frozen=True)
class MosaicTpuGmm(_MosaicTpu):
    """`Gmm` on a TPU (`mosaic_tpu.gmm`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    def space(self, ba) -> tuple[mosaic_tpu.Blocks, ...]:
        kernel = ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return tuple(blocks for blocks in TPU_BLOCKS
                     if depth % blocks.block_k == 0 and width % blocks.block_n == 0
                     and self._lays_out(ba, blocks))

    @staticmethod
    def extents(blocks: mosaic_tpu.Blocks) -> dict:
        return {'rows': blocks.block_m, 'depth': blocks.block_k, 'columns': blocks.block_n}

    @staticmethod
    def loaded(blocks: mosaic_tpu.Blocks) -> int:
        return (blocks.block_m + blocks.block_n) * blocks.block_k

    @staticmethod
    def output(blocks: mosaic_tpu.Blocks) -> tuple[int, int]:
        return blocks.block_m, blocks.block_n

    def costed(self, ba, config: mosaic_tpu.Blocks, target) -> cost.Cost:
        # A grid slot is one row block of one group (a block two groups share takes one of each), swept
        # over the depth for each column block: the rows are read once a column block, and each slot's
        # group's matrix once a slot (Megablox's own estimate).
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        (rows, depth), groups = tokens.shape, kernel.shape[0]
        width = kernel.shape[1] if ba.arguments['trans_rhs'] else kernel.shape[2]
        size = jnp.dtype(tokens.dtype).itemsize
        slots, columns = _block_tiles(rows, groups, config.block_m), width // config.block_n
        hooks = ba.arguments['hooks']
        grid = (('columns', columns), ('rows', slots), ('depth', depth // config.block_k))
        moved = ((columns * rows * depth + slots * depth * width) * size + rows * width * _output_bytes(ba)
                 + _fetched_bytes(ba.arguments['prologue_operands'], hooks.prologue_axes, grid)
                 + _fetched_bytes(ba.arguments['epilogue_operands'], hooks.epilogue_axes, grid))
        return self._cost(depth / config.block_k, slots * columns,
                          2 * config.block_m * config.block_n * depth, moved, tokens.dtype)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             *, out_dtype: str,
             trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: mosaic_tpu.Blocks, return_residuals: bool = False):
        # Never set: the launch is not eligible for exact sums; a dense call is one group, as any.
        del exact_sums, dense
        return mosaic_tpu.gmm(tokens, kernel, sizes, blocks=config, transpose_rhs=trans_rhs,
                              out_dtype=out_dtype, interpret=TPU_INTERPRETED if interpret else False,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              paired=paired, **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class MosaicTpuTgmm(_MosaicTpu):
    """`Tgmm` on a TPU (`mosaic_tpu.tgmm`): `block_m` x `block_n` blocks of each group's gradient, its rows
    `block_k` at a time."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    def space(self, ba) -> tuple[mosaic_tpu.Blocks, ...]:
        depth, width = ba.arguments['tokens'].shape[1], ba.arguments['cotangent'].shape[1]
        return tuple(blocks for blocks in TPU_BLOCKS
                     if depth % blocks.block_m == 0 and width % blocks.block_n == 0
                     and self._lays_out(ba, blocks))

    @staticmethod
    def extents(blocks: mosaic_tpu.Blocks) -> dict:
        return {'rows': blocks.block_k, 'depth': blocks.block_m}

    @staticmethod
    def loaded(blocks: mosaic_tpu.Blocks) -> int:
        return blocks.block_k * (blocks.block_m + blocks.block_n)

    @staticmethod
    def output(blocks: mosaic_tpu.Blocks) -> tuple[int, int]:
        return blocks.block_m, blocks.block_n

    def costed(self, ba, config: mosaic_tpu.Blocks, target) -> cost.Cost:
        # Each output block sweeps the grid slots, a row block of one group each (an empty group's one
        # slot zeroes its gradient): the rows are read once a column block, the cotangent once a depth
        # block, each gradient written once.
        tokens, cotangent, sizes = ba.arguments['tokens'], ba.arguments['cotangent'], ba.arguments['sizes']
        (rows, depth), width, groups = tokens.shape, cotangent.shape[1], sizes.shape[0]
        size, out = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(ba.arguments['out_dtype']).itemsize
        slots = _block_tiles(rows, groups, config.block_k)
        blocks = depth // config.block_m, width // config.block_n
        hooks = ba.arguments['hooks']
        grid = (('columns', blocks[1]), ('depth', blocks[0]), ('rows', slots))
        moved = ((blocks[1] * rows * depth + blocks[0] * rows * width) * size + groups * depth * width * out
                 + _fetched_bytes(ba.arguments['prologue_operands'], hooks.prologue_axes, grid))
        flops = 2 * slots * config.block_k * config.block_m * config.block_n
        return self._cost(slots, blocks[0] * blocks[1], flops, moved, tokens.dtype)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: mosaic_tpu.Blocks,
             return_residuals: bool = False):
        del exact_sums, dense  # never set (`eligible`); a dense call is one group, as any
        return mosaic_tpu.tgmm(tokens, cotangent, sizes, blocks=config, out_dtype=out_dtype,
                               interpret=TPU_INTERPRETED if interpret else False, prologue=hooks.before(),
                               prologue_operands=prologue_operands, prologue_axes=hooks.prologue_axes), None

def _padding_bytes(arrays: tuple, config: Tile) -> int:
    """The bytes a dense call's copies move (`triton_dense`): each of `arrays` (its shape, the launch axes
    'rows', 'depth' or 'width' its dimensions follow, None for one it broadcasts along, its itemsize),
    read and written padded to `config`'s tile (rows by `block_m`, depth by `block_k`, width by
    `block_n`) where it is not a multiple of it."""
    block = {'rows': config.block_m, 'depth': config.block_k, 'width': config.block_n}
    moved = 0
    for shape, axes, itemsize in arrays:
        grown = tuple(-(extent // -block.get(axis, 1)) * block.get(axis, 1)
                      for extent, axis in zip(shape, axes, strict=True))
        moved += 2 * math.prod(grown) * itemsize if grown != tuple(shape) else 0
    return moved


def _side_arrays(sides: tuple, axes_of: tuple) -> tuple:
    """Side operands as `_padding_bytes` takes arrays: their dimensions along launch axes, 'columns'
    named 'width'."""
    return tuple((side.shape, tuple('width' if axis == 'columns' else axis for axis in axes),
                  jnp.dtype(side.dtype).itemsize) for side, axes in zip(sides, axes_of, strict=True))


def _uncut(sides: tuple, axes_of: tuple, halved: str) -> tuple[tuple, tuple]:
    """The side operands, and their axes, without the launch axis a cooperative consumer's half cuts."""
    kept = [(side, axes) for side, axes in zip(sides, axes_of, strict=True) if halved not in axes]
    return tuple(side for side, _ in kept), tuple(axes for _, axes in kept)


def _modelled(ba) -> tuple[mosaic_gpu.Tiles, ...]:
    """The Mosaic GPU tiles a launch may take: interpreted, none of a pair of multiprocessors, which the
    interpreter's one block does not model."""
    return tuple(tile for tile in MOSAIC_TILES if not (ba.arguments['interpret'] and tile.cluster == 2))


def _facts(target: generation.GpuTarget) -> generation.GpuTarget:
    """The target a launch on `target` is costed as (`cost._facts`): a CPU as an A100."""
    return target if generation.capability(target.generation) is not None else generation.TARGETS['sm80']


def _multiprocessors(target: generation.GpuTarget) -> int:
    """`target`'s multiprocessors, which the persistent Mosaic GPU launches spread their tiles over."""
    if target.multiprocessors is None:
        raise ValueError(f"{target.generation} states no multiprocessors, which the Mosaic GPU launches read")
    return target.multiprocessors


def _rated(target: generation.GpuTarget, dtype) -> bool:
    """Whether `target` states the tensor rate a launch over `dtype` operands is costed by: a generation
    that states none runs XLA's grouped matmul, needing no estimate (`kernel_refusal`)."""
    return generation.effective(target).tensor_rate(_mode(dtype)) is not None


def kernel_refusal(compute: Dtype, promoted: bool = False) -> str | None:
    """Why the kernels do not run a product computed in `compute` on this target, or None where one of
    them does: its generation states no tensor rate for it (an unseen GPU's), or the product adds each
    step apart (`promoted`) on a TPU, whose launches do not (`_MosaicTpu`)."""
    target = generation.gpu_target()
    if not _rated(target, compute):
        return (f"{target.generation} states no {_mode(compute)} tensor rate, which the kernels' cost model "
                "reads")
    if promoted and generation.tpu(target):
        return f"{target.generation}'s MXUs add within a step, which no measurement states IEEE"
    return None


_GMMS = (MosaicGmm(), MosaicTpuGmm(), Gmm())
_TGMMS = (MosaicTgmm(), MosaicTpuTgmm(), Tgmm())


def launched_gmm(tokens, kernel, sizes, prologue_operands: tuple = (), epilogue_operands: tuple = (),
                 paired=None, *, out_dtype, trans_rhs: bool, hooks: Hooks = NO_HOOKS, interpret: bool,
                 promoted: bool = False, dense: bool = False):
    """Each group's rows of `tokens` times its matrix of `kernel`, rounded to `tokens`'s dtype, through
    the launch that runs it (`_launch`), with the template's hooks (`templates.gemm.template.gemm`).
    `promoted`, the declared product's (`templates.product.Product`), adds each depth step's partial
    apart (`exact_sums`); `dense`, one group of every row (a dense product), the plain tiled kernel where
    Triton runs it. `paired`, a second matrix, makes each tile two products the epilogue combines
    (`mosaic_gpu.gmm`'s)."""
    launches, arguments, options = _gmm_call(tokens, kernel, sizes, prologue_operands, epilogue_operands,
                                             paired, out_dtype=out_dtype, trans_rhs=trans_rhs, hooks=hooks,
                                             interpret=interpret, promoted=promoted, dense=dense)
    return _launch(launches, arguments, options)(*arguments, **options)


def gmm_refusal(tokens, kernel, sizes, prologue_operands: tuple = (), epilogue_operands: tuple = (),
                paired=None, **options) -> str | None:
    """Why no launch takes a `launched_gmm` call (`launch_refusal`), or None where one does."""
    return launch_refusal(*_gmm_call(tokens, kernel, sizes, prologue_operands, epilogue_operands, paired,
                                     **options))


def _gmm_call(tokens, kernel, sizes, prologue_operands: tuple, epilogue_operands: tuple, paired, *, out_dtype,
              trans_rhs: bool, hooks: Hooks = NO_HOOKS, interpret: bool, promoted: bool = False,
              dense: bool = False) -> tuple:
    """`launched_gmm`'s launches, arguments and options."""
    arguments = (tokens, kernel.astype(tokens.dtype), sizes, tuple(prologue_operands),
                 tuple(epilogue_operands), None if paired is None else paired.astype(tokens.dtype))
    options = {'out_dtype': jnp.dtype(out_dtype).name, 'trans_rhs': trans_rhs, 'hooks': hooks,
               'interpret': interpret, 'exact_sums': promoted, 'dense': dense}
    return _GMMS, arguments, options


def launched_tgmm(tokens, cotangent, sizes, prologue_operands: tuple = (), *, out_dtype,
                  hooks: Hooks = NO_HOOKS, interpret: bool, promoted: bool = False, dense: bool = False):
    """Each group's rows of `tokens` transposed times its rows of `cotangent`, rounded to `tokens`'s
    dtype, through the launch that runs it, with the template's prologue
    (`templates.gemm.template.gemm_t`); `dense` as `launched_gmm`'s."""
    launches, arguments, options = _tgmm_call(tokens, cotangent, sizes, prologue_operands,
                                              out_dtype=out_dtype, hooks=hooks, interpret=interpret,
                                              promoted=promoted, dense=dense)
    return _launch(launches, arguments, options)(*arguments, **options)


def tgmm_refusal(tokens, cotangent, sizes, prologue_operands: tuple = (), **options) -> str | None:
    """Why no launch takes a `launched_tgmm` call (`launch_refusal`), or None where one does."""
    return launch_refusal(*_tgmm_call(tokens, cotangent, sizes, prologue_operands, **options))


def _tgmm_call(tokens, cotangent, sizes, prologue_operands: tuple, *, out_dtype, hooks: Hooks = NO_HOOKS,
               interpret: bool, promoted: bool = False, dense: bool = False) -> tuple:
    """`launched_tgmm`'s launches, arguments and options."""
    arguments = (tokens, cotangent.astype(tokens.dtype), sizes, tuple(prologue_operands))
    options = {'out_dtype': jnp.dtype(out_dtype).name, 'hooks': hooks, 'interpret': interpret,
               'exact_sums': promoted, 'dense': dense}
    return _TGMMS, arguments, options


def launch_refusal(launches, arguments, options) -> str | None:
    """Why none of `launches` takes the call on this target, or None where one does (`_launch`): on a
    CUDA GPU, Triton takes every product the kernels compute; a TPU's launches take only the widths their
    blocks divide, the side operands their blocks lay out and the hooks Mosaic TPU lowers."""
    if options['interpret'] and jax.default_backend() != 'cpu':
        return None
    target = generation.gpu_target()
    if any(launch.eligible(launch.bind(*arguments, **options), target) for launch in launches):
        return None
    return f"no launch of {target.generation} takes this call's widths, side operands or hooks"


def _launch(launches, arguments, options):
    """The launch of `launches` that runs the call: the cost model's (`base.choose`), except in a GPU
    process's CPU branch, which interprets Triton's, the one launch that interprets there
    (`_Mosaic.eligible`): a chosen implementation names the GPU's launch, the branch the call takes."""
    if options['interpret'] and jax.default_backend() != 'cpu':
        *_, triton = launches
        generation.route('grouped_matmul', kernel=triton.name)
        return triton
    return base.choose('grouped_matmul', launches, *arguments, **options)


def _on_platform(kernel, fallback, interpret_on_cpu: bool):
    """`kernel` where the call lowers for CUDA, interpreted on the CPU when
    asked for, and `fallback` everywhere else: a TPU's program, which never runs
    on CUDA, holds no CUDA branch."""
    tpu = generation.tpu(generation.gpu_target())
    branches = {} if tpu else {'cuda': functools.partial(kernel, interpret=False)}
    if interpret_on_cpu:
        branches['cpu'] = functools.partial(kernel, interpret=True)
    return branches, fallback


def xla_ragged_dot(tokens: jax.Array, kernel: jax.Array, group_sizes: jax.Array, *,
                   precision: PrecisionLike = None,
                   preferred_element_type: Dtype | None = None) -> jax.Array:
    """`jax.lax.ragged_dot` as every backend runs it alike.

    A 16-bit product is exact at any precision, and XLA's TPU ragged-dot
    kernel refuses 16-bit operands at HIGHEST ("Bad lhs type"), so two of
    them multiply at DEFAULT: the same products, summed in the preferred
    type. A 16-bit operand beside a wider one is widened to it.
    XLA's TPU kernel writes values into the rows past the groups, where
    other backends write zeros, and its lhs cotangent is the same kernel.
    Those rows are zeroed going in and coming out, so neither the output
    nor a dropped row's gradient carries them.
    """
    if all(jnp.finfo(operand.dtype).bits == 16 for operand in (tokens, kernel)):
        precision = jax.lax.Precision.DEFAULT
    elif tokens.dtype != kernel.dtype:
        # A 16-bit operand beside a wider one would reach the TPU kernel
        # 16-bit at HIGHEST too; widened, it is the same values (exactly),
        # and the wider operand keeps the precision the call asked for.
        wide = jnp.promote_types(tokens.dtype, kernel.dtype)
        tokens, kernel = tokens.astype(wide), kernel.astype(wide)
    # bincount widens under x64; TPU ragged-dot only lowers int32 counts.
    # Keep the wider domain for CPU calls whose row count actually needs it.
    index_dtype = jnp.int32 if tokens.shape[0] <= np.iinfo(np.int32).max else jnp.int64
    if group_sizes.dtype == jnp.int64:
        group_sizes = group_sizes.astype(index_dtype)
    grouped = jnp.arange(tokens.shape[0], dtype=index_dtype)[:, None] < jnp.sum(
        group_sizes, dtype=index_dtype
    )
    out = jax.lax.ragged_dot(
        jnp.where(grouped, tokens, 0), kernel, group_sizes, precision=precision,
        preferred_element_type=preferred_element_type)
    return jnp.where(grouped, out, 0)


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5))
def portable_projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                dtype: Dtype | None, precision: PrecisionLike, out_dtype: Dtype | None = None) -> jax.Array:
    """The grouped projection on `xla_ragged_dot`, in every differentiation mode: `x`'s rows `[rows, in]`
    through each group's matrix of `kernel` `[groups, in, out]`, both cast to `dtype` (flax's promotion
    where None), summed in at least fp32 and rounded once. Its tangent is `dx @ Q(kernel) + Q(x) @
    dkernel`, `Q` the rounded operands, summed over the widest of the compute, input and kernel dtypes,
    so an input gradient keeps its input's dtype and a kernel gradient its master's: the portable
    projection `dew.nn.moe.expert_projection`'s XLA route and the GEMM template's portable branch run.
    `out_dtype`, where given, is the dtype its sum rounds to in place of the products' (the template's
    fp32 accumulator, which its epilogue reads)."""
    x, kernel = promote_dtype(x, kernel, dtype=dtype)
    accumulated = xla_ragged_dot(x, kernel, group_sizes, precision=precision,
                                 preferred_element_type=jnp.promote_types(x.dtype, jnp.float32))
    return accumulated.astype(out_dtype or x.dtype)


def _portable_projection_jvp(dtype: Dtype | None, precision: PrecisionLike, out_dtype: Dtype | None,
                    primals: tuple[jax.Array, jax.Array, jax.Array],
                    tangents: tuple[jax.Array, jax.Array, jax.Array]
                    ) -> tuple[jax.Array, jax.Array | SymbolicZero]:
    x, kernel, group_sizes = primals
    dx, dkernel, _ = tangents
    output = jnp.asarray(portable_projection(x, kernel, group_sizes, dtype, precision, out_dtype))
    work = jnp.result_type(output.dtype, x.dtype, kernel.dtype, jnp.float32)
    compute = jnp.dtype(dtype) if dtype is not None else jnp.result_type(x.dtype, kernel.dtype)
    # A symbolic zero tangent contributes no term. Only the others pass the
    # barrier: jax 0.11.2 cannot transpose one that holds a constant zero.
    live = [(name, tangent) for name, tangent in (('x', dx), ('kernel', dkernel))
            if not isinstance(tangent, SymbolicZero)]
    if not live:
        return output, SymbolicZero(jax.typeof(output))
    # The tangents keep their dtypes to this point whatever produced them, so
    # an activation's bf16 cotangent is not fused into a wider expression.
    held = dict(zip((name for name, _ in live),
                    jax.lax.optimization_barrier(tuple(tangent for _, tangent in live)),
                    strict=True))
    terms = []
    # Each rounded operand is held in the work dtype, which holds the rounding whatever the operand's own
    # (an fp16 master's largest value rounds to bf16's 65536, which fp16 does not hold).
    if 'x' in held:
        matrix = jnp.asarray(rounded_operand(kernel.astype(work), compute))
        terms.append(xla_ragged_dot(
            held['x'].astype(work), matrix.astype(work), group_sizes, precision=precision,
            preferred_element_type=work))
    if 'kernel' in held:
        inputs = jnp.asarray(rounded_operand(x.astype(work), compute))
        terms.append(xla_ragged_dot(
            inputs.astype(work), held['kernel'].astype(work), group_sizes, precision=precision,
            preferred_element_type=work))
    total = terms[0] if len(terms) == 1 else terms[0] + terms[1]
    tangent = jax.lax.optimization_barrier(total.astype(output.dtype))
    return output, tangent


portable_projection.defjvp(_portable_projection_jvp, symbolic_zeros=True)


