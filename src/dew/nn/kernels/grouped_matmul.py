"""The grouped matmul's kernel launches, which the GEMM template (`templates.gemm.template`) runs and
differentiates (`dew.nn.moe.expert_projection` through it), and XLA's portable projection.

Three products, in MaxText's megablox shape: the forward `gmm`, the input gradient `gmm` against the
transposed kernel, and the kernel gradient `tgmm`. Each is a template launch (`templates.base`) with
its implementations taken per call in a fixed order among those that run it: Dew's Pallas Mosaic GPU
kernels (`templates.gemm.mosaic_gpu`) on their instruction family's bundle, the TPU's Mosaic kernels
(`templates.gemm.mosaic_tpu`), then JAX's Pallas Triton kernels (`ragged_dot.py`, vendored), on every
CUDA GPU from sm80 and on a CPU that interprets them, for a call neither Mosaic launch takes; and
CUTLASS's grouped GEMMs (`kernels/cutlass`, the dew-cutlass wheel) where it is installed, a vendored
kernel a call runs only by name (`generation.chosen`). The grouped selector (`GROUPED`) reads which
products they compute exactly (the product's declaration, `templates.product.declared`) and which
launch takes a call. They run only for their target's lowering, CUDA's or a TPU's (`_on_platform`),
chosen where the call lowers (`jax.lax.platform_dependent`), so a CPU computation on a GPU host lowers
to `jax.lax.ragged_dot`; an explicit request interprets them on the CPU, as the CPU suite checks them.

Each launch's config is the one it derives for the GPU, or the Triton launches' cost model's fastest
of their fixed tiles that fits it, the same on every run. JAX's one tile for every shape left a wide
expert's products at half an A100's cuBLAS rate (a 4096 by 14336 expert's forward 10.4 ms where 5.3 ms
is the tuned tile's, the kernel gradient 18.3 against 8.4).
"""

import abc
import dataclasses
import functools
import importlib
import importlib.util
import itertools
import math
from collections.abc import Callable, Hashable
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

from ..precision import at_least_fp32, rounded_operand
from . import generation, selection, tuning
from .generation import INHERENT
from .templates import base, bundles, cost, mosaic
from .templates.gemm import mosaic_gpu, mosaic_tpu
from .templates.product import Product


@dataclasses.dataclass(frozen=True)
class Tile:
    """One Triton launch's output tile, its depth step, and its warps and pipeline stages."""
    block_m: int
    block_n: int
    block_k: int
    warps: int
    stages: int


def _doubling(least: int, most: int) -> tuple[int, ...]:
    """`least` and its doublings up to `most`, ascending."""
    return tuple(least << power for power in range(max(most // least, 1).bit_length())
                 if least << power <= most)


TILES = tuple(Tile(*shape) for shape in itertools.product((64, 128, 256), (128, 256), (128, 64, 32), (8, 4),
                                                            (3, 4, 2)))
"""Every tile the Triton launches may take: the ones the A100 and H100 sweeps found fastest lie inside.
Where the cost model ties, the first wins: the deeper step, which synchronizes its pipeline fewer times
a tile (at 128 x 256 on an H100, a step of 64 took 1.56 ms where 32 took 1.78, GMM-h100-2), and the
more warps. The Triton launches are kept, not extended (ops/BIBLE.md): a fixed set the chooser prices
with the model's shared terms, fitted to none of Triton's own."""


def _covering(extent: int) -> int:
    """The least power of two at least `extent`."""
    return 1 << (max(extent, 1) - 1).bit_length()


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
    each side operand's dimensions of one the launch squeezed out (the hook gets them back), the dtypes of
    the epilogue's outputs where it returns several, and the dtypes a kernel traces each hook at (its tile's
    then each side tile's: `prologue_dtypes`, `epilogue_dtypes`). The side operands themselves are the
    launch's `prologue_operands` and `epilogue_operands`. A launch's picks key each hook by what it
    computes (`_GroupedLaunch.traced`), never by the function object."""
    prologue: object = None
    prologue_axes: tuple = ()
    epilogue: object = None
    epilogue_axes: tuple = ()
    out_dtypes: tuple | None = None
    prologue_broadcast: tuple = ()
    epilogue_broadcast: tuple = ()
    prologue_dtypes: tuple = ()
    epilogue_dtypes: tuple = ()

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
            and _lowers(hooks.after(), jax.ShapeDtypeStruct((128, 128), jnp.float32),
                        tiles('epilogue_operands', hooks.epilogue_axes), rules))


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


def _written(elements: int, ba, columns: int) -> float:
    """The device bytes a launch writing `elements` output elements a row segment of `columns` at a time
    moves: each segment at least `cost.WRITE_GRANULE` bytes (`cost.written`)."""
    return sum(cost.written(elements * jnp.dtype(dtype).itemsize, columns * jnp.dtype(dtype).itemsize)
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


TRACED_TILE = 16
"""The square tile a launch's hooks are traced at to key its picks (`_GroupedLaunch.traced`)."""


class _GroupedLaunch[Config: Hashable](base.Template[Config]):
    """A grouped launch: its space is a declared tuple, its group sizes drawn as a step routes them."""
    # Annotations stay evaluated (no postponed annotations in this module):
    # tokamax serializes a launch's arguments into the program by their types.
    HOOKS: ClassVar[tuple[str, ...]] = ('hooks',)

    def traced(self, name: str, value: Callable | str | None) -> str | None:
        """The launch's `Hooks` as its picks key them (`tuning.KernelOp.traced`): each hook by what it
        computes (`tuning.program`) as a kernel traces it (`Hooks.before`, `Hooks.after`), at a
        `TRACED_TILE` square tile and each side tile as its axes lay it out there, in the dtypes the hooks
        state, beside their axes, squeezed positions and output dtypes; an already traced key as it is."""
        if not isinstance(value, Hooks):
            return value if isinstance(value, str) else None

        def program(hook, axes_of: tuple, dtypes: tuple) -> str | None:
            if hook is None:
                return None
            tile, *sides = dtypes
            return tuning.program(hook, jax.ShapeDtypeStruct((TRACED_TILE, TRACED_TILE), tile),
                                  *(jax.ShapeDtypeStruct((TRACED_TILE,) * len(axes), dtype)
                                    for axes, dtype in zip(axes_of, sides, strict=True)))

        return "hooks:" + repr((program(value.before(), value.prologue_axes, value.prologue_dtypes),
                                program(value.after(), value.epilogue_axes, value.epilogue_dtypes),
                                value.prologue_axes, value.epilogue_axes, value.out_dtypes,
                                value.prologue_broadcast, value.epilogue_broadcast))

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
        cuda = generation.capability(target.generation) is not None
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
        wgmma = target.wgmma

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

    def staged_bytes(self, ba, config: Tile, target) -> int:
        # Pallas Triton's pipeline holds `stages - 1` depth steps, and on Hopper `stages` where it
        # pipelines a wgmma, which reads one step while the next loads; an A100 and an H100 refuse
        # exactly those bytes, so a Hopper tile is sized for the larger. The prologue's side tiles load
        # in the same steps.
        held = 0 if target.wgmma else 1
        step = self.loaded(config) * jnp.dtype(ba.arguments['tokens'].dtype).itemsize + _side_tile_bytes(
            ba.arguments['prologue_operands'], ba.arguments['hooks'].prologue_axes, self.extents(config))
        # A contraction of one step has no loop whose dead stages the output's layout conversion reuses:
        # it takes a 16-bit output tile beside them (an L4 refused 114688 bytes for a 64 x 128 tile's
        # 98304 at a depth of one step, either output dtype, GEN6-gpu-l4; at four steps it asked the
        # stages' alone, TSMEM-l4).
        contracted = self._contracted(ba)
        single = contracted is not None and contracted <= self.contraction(config)
        rows, columns = self.output(config)
        return step * (config.stages - held) + (2 * rows * columns if single else 0)

    @staticmethod
    def _contracted(ba) -> int | None:
        """The contraction a call's every tile steps over, None where its groups' rows set it."""
        return None

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
        return cost.resident(target, self.staged_bytes(ba, config, target), threads,
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
    def _contracted(ba) -> int | None:
        return Gmm._dense_extents(ba)[0]

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
                                        target.wgmma) * size
        hooks, prologue_sides = ba.arguments['hooks'], ba.arguments['prologue_operands']
        epilogue_sides = ba.arguments['epilogue_operands']
        # Side operands are read again as the operands are: with each column block, and with each row
        # tile where they follow no rows; the hooks' special functions on every element they see.
        rereads = {'rows': tiles, 'columns': columns}
        moved = ((columns * rows * depth + tiles * depth * width) * size
                 + _written(rows * width, ba, config.block_n)
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
                hooks.after(), jnp.float32, epilogue_sides, hooks.epilogue_axes,
                (config.block_m, config.block_n), extents))
        return cost.Cost(
            programs=chunks * columns, device_bytes=moved,
            program_flops=flops / (chunks * columns), program_staged=staged,
            program_exponentials=exponentials, mode=_mode(tokens.dtype), steps=steps, stages=config.stages,
            resident=self._resident(ba, config, target))

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), *, out_dtype: str,
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
                                    epilogue_operands=epilogue_operands, exact_sums=exact_sums, **options)
            return (outs if hooks.out_dtypes else outs[0]), None
        from . import ragged_dot
        return ragged_dot.gmm(tokens, kernel, sizes, block_m=config.block_m, block_k=config.block_k,
                              block_n=config.block_n, num_warps=config.warps, num_stages=config.stages,
                              trans_rhs=trans_rhs, interpret=interpret, compute_dtype=tokens.dtype,
                              out_dtype=jnp.dtype(out_dtype), prologue_operands=prologue_operands,
                              epilogue_operands=epilogue_operands, exact_sums=exact_sums,
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
                                        target.wgmma) * size
        hooks, sides = ba.arguments['hooks'], ba.arguments['prologue_operands']
        rereads = {'rows': groups, 'depth': blocks[0], 'columns': blocks[1]}
        moved = ((blocks[1] * rows * depth + blocks[0] * rows * width) * size
                 + cost.written(groups * depth * width * out, config.block_n * out)
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
    """JAX's Pallas Mosaic GPU kernels (`templates.gemm.mosaic_gpu`) on their instruction family's bundle
    (Hopper's warpgroup MMA, Ampere's mma.sync), one 16-bit dtype, their one tile generated from the
    family's facts, the device's and the call's extents (`mosaic_gpu.derived`). Each runs one warpgroup a
    multiprocessor, or two consumers beside a producer (a cooperative tile), persistent over the output
    tiles in a planar snake, so the rows a band of `grid_block_n` column tiles shares and the matrix block
    the row tiles in flight share come from the L2."""
    name: ClassVar[str] = 'mosaic_gpu'
    config_cls: ClassVar[type[mosaic_gpu.Tiles]] = mosaic_gpu.Tiles
    GENERATES: ClassVar[bool] = True
    EXACT_SUMS: ClassVar[bool] = False
    """Whether the kernel adds each depth step's product apart (`exact_sums`); one that does not is not
    eligible where the call asks for it."""
    STORES_SERIALLY: ClassVar[bool] = False
    """Whether a one-warpgroup tile's store holds back its next tile's loads (`cost.Cost.serial_stored`):
    `tgmm`'s, whose few steps a tile leave the store most of its time, where `gmm`'s depth hides it and a
    cooperative `tgmm` tile streams its stages across tiles."""

    def eligible(self, ba, target) -> bool:
        first, second = (value.dtype for name, value in ba.arguments.items()
                         if name in ('tokens', 'kernel', 'cotangent'))
        # Hooked and exactly summed calls run on warpgroup MMA alone: mma.sync's kernels take them (its
        # side loads, vector reads and running sum lower), but its register model does not yet count what
        # they hold, and its derived tiles spilled (ns-regs2: sm86's normed forward 16 bytes, exact sums 136).
        special = ba.arguments['hooks'] != NO_HOOKS or ba.arguments.get('exact_sums')
        return (first == second and jnp.dtype(first).itemsize == 2
                and _rated(target, first) and self._family(ba, target) is not None
                and (target.wgmma or not special)
                and self._hooks_lower(ba, first, target)
                and (self.EXACT_SUMS or not ba.arguments.get('exact_sums'))
                and self.derived(ba, target) is not None)

    def registers(self, ba, config: mosaic_gpu.Tiles) -> int:
        # A consumer thread's live set (`mosaic.tile_registers`), and where a prologue rewrites its share of
        # the rows' stage, that share in fp32 beside the accumulator (ptxas spilled 384 bytes at a normed
        # 192 x 128 forward and 156 at a normed 128 x 192 kernel gradient the plain estimate held: the
        # compile gate's H100 run).
        dtype = ba.arguments['tokens'].dtype
        family, block = _bundle(config, dtype), (config.block_m, config.block_n, config.block_k)
        # A transposed matrix (the input gradient's) is loaded a step deep and sliced in registers
        # (`mosaic_sm80.Mma.products`, `mosaic.tile_registers`).
        transposed = bool(ba.arguments.get('trans_rhs', False))
        plain = mosaic.tile_registers(family, config.warpgroups, block, dtype, transposed=transposed)
        if ba.arguments['hooks'].prologue is None:
            return plain
        # The rows' stage a consumer rewrites: its share of a product's rows by the step, or of a kernel
        # gradient's depth by the step's rows, `block_m` by `block_k` over the consumers either way.
        share = config.block_m * config.block_k // config.warpgroups
        return plain + -(-share // mosaic.THREADS)

    def register_budget(self, config: mosaic_gpu.Tiles) -> int:
        # Its family's budget a consumer, whatever the operands' dtype (`mosaic.Family.registers`).
        return _bundle(config, jnp.bfloat16).registers(config.warpgroups)

    def _family(self, ba, target) -> mosaic.Family | None:
        """The bundle the launch runs on `target`: Hopper's warpgroup MMA where it has it, Ampere's
        mma.sync on a CUDA GPU without it where the launch takes it (`_families`), none on any other
        device (a TPU's launches are its own; a CPU interprets a GPU's only where `generation.assumed`
        names it)."""
        if generation.capability(target.generation) is None:
            return None
        wanted = 'wgmma' if target.wgmma else 'mma.sync'
        return next((family for family in self._families(ba) if family.name == wanted), None)

    def _families(self, ba) -> tuple[mosaic.Family, ...]:
        """The bundles the launch takes at its operands' dtype (`bundles.families`)."""
        return bundles.families(ba.arguments['tokens'].dtype)

    @abc.abstractmethod
    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        """A group's output rows (`rows`, on average), the depth the steps run over (None: a group's rows,
        padded, the output's rows then whole, not a group's) and the output's width, as
        `mosaic_gpu.derived` reads them."""

    def derived(self, ba, target) -> mosaic_gpu.Tiles | None:
        family = self._family(ba, target)
        if family is None:
            return None
        rows, depth, width = self._generated_extents(ba)
        return mosaic_gpu.derived(family, rows=rows, depth=depth, width=width,
                                  dtype=ba.arguments['tokens'].dtype,
                                  multiprocessors=_multiprocessors(target),
                                  holds=lambda tile: self.legal(ba, tile, target), grouped=depth is not None)

    def neighbours(self, ba, config: mosaic_gpu.Tiles) -> tuple[mosaic_gpu.Tiles, ...]:
        # The derived tile with one of its choices moved a step either way: a stage more or fewer, the
        # band halved or doubled, the columns or the rows the next block either side (a column unit, a
        # role of rows).
        family = _bundle(config, ba.arguments['tokens'].dtype)
        unit = family.column_unit(ba.arguments['tokens'].dtype)
        role = family.role_rows * config.warpgroups
        band = config.grid_block_n
        moves = [{'stages': config.stages - 1}, {'stages': config.stages + 1},
                 {'grid_block_n': max(1, band // 2)}, {'grid_block_n': band * 2},
                 {'block_n': config.block_n - unit}, {'block_n': config.block_n + unit},
                 {'block_m': config.block_m - role}, {'block_m': config.block_m + role}]
        found = []
        for move in moves:
            if min(move.values()) <= 0:
                continue
            try:
                moved = dataclasses.replace(config, **move)
            except ValueError:
                continue  # a shape no Mosaic tile takes (`Tiles.__post_init__`)
            if moved != config:
                found.append(moved)
        return tuple(found)

    def legal(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        """Whether `config` runs the call on `target` (`Template.legal`): of the launch's family there, in
        its granularity (whole roles of rows for each consumer, whole column units, one of its depths, at
        most its consumers, one multiprocessor a tile) with a ring of two stages at least (a walk keeping
        one step's products in flight holds its slot while the next step's loads fill another), its blocks
        dividing the call's extents, and a one-dimensional side operand's block, copied untiled, at most
        one copy's extent along its axis (`mosaic.Family.copy_extent`)."""
        family = self._family(ba, target)
        if family is None or config.family != family.name or not self._divides(ba, config):
            return False
        dtype = ba.arguments['tokens'].dtype
        if (config.warpgroups > family.consumers or config.cluster != 1 or config.stages < 2
                or config.block_m % (family.role_rows * config.warpgroups)
                or config.block_n % family.column_unit(dtype) or config.block_k not in family.depths(dtype)):
            return False
        hooks = ba.arguments['hooks']
        axes = {axis for name, axes_of in (('prologue_operands', hooks.prologue_axes),
                                           ('epilogue_operands', hooks.epilogue_axes))
                for _, side_axes in zip(ba.arguments.get(name, ()), axes_of, strict=True)
                if len(side_axes) == 1 for axis in side_axes}
        # A vector the family reads from global memory instead (`copies_vectors`) loads in a role's row
        # tiles: its block along its axis whole roles (ns-regs: a 32-element depth vector on sm86 failed
        # to cast to the 64-row layout).
        extent, least = family.copy_extent, 1 if family.copies_vectors else family.role_rows
        copied = all((extent is None or self.extents(config)[axis] <= extent)
                     and self.extents(config)[axis] % least == 0 for axis in axes)
        return copied and super().legal(ba, config, target)

    @abc.abstractmethod
    def _divides(self, ba, config: mosaic_gpu.Tiles) -> bool:
        """Whether `config`'s blocks divide the call's extents as the kernel requires."""

    @staticmethod
    def _hooks_lower(ba, dtype, target) -> bool:
        """Whether Mosaic GPU's lowering takes the launch's hooks (`_hooks_lower`). A side operand the
        launch squeezes and gives back with its dimension of one (`Hooks.prologue_broadcast`) has no
        layout Mosaic's inference finds for the size-one dimension, so a hook taking one does not lower
        there (GMMGATE-h100-8), where Triton's lowering takes it."""
        # Both bundles lower with warpgroup semantics, whose rules a hook's operations need.
        del target
        hooks = ba.arguments['hooks']
        if hooks.prologue_broadcast or hooks.epilogue_broadcast:
            return False
        return _hooks_lower(ba, dtype, mosaic_gpu_lowering.mosaic_lowering_rules[plgpu_core.WGxWG_SEMANTICS])

    def staged_bytes(self, ba, config: mosaic_gpu.Tiles, target) -> int:
        # Each allocation a program scopes, rounded up to Mosaic's alignment (`mosaic.allocated`): each
        # stream's stages (both operand blocks and each prologue side's tile), each output's staging (every
        # output's at once, the columns its family stages at once: `mosaic.Family.output_columns`), each
        # epilogue side's two staging copies (`mosaic.staging`), and the barriers: each stream stage's two
        # and each staged side's two (at a block's edge, a Hopper tile's barriers alone took it 128 bytes
        # past, the reason they are counted).
        size = jnp.dtype(ba.arguments['tokens'].dtype).itemsize
        hooks = ba.arguments['hooks']
        outs = [jnp.dtype(dtype).itemsize for dtype in (hooks.out_dtypes or (ba.arguments['out_dtype'],))]
        extents = self.extents(config)
        prologue_sides = ba.arguments['prologue_operands']
        epilogue_sides = ba.arguments.get('epilogue_operands', ())
        streams = [config.block_m * config.block_k * size, config.block_k * config.block_n * size,
                   *(_side_tile_bytes((side,), (axes,), extents)
                     for side, axes in zip(prologue_sides, hooks.prologue_axes, strict=True))]
        family = _bundle(config, ba.arguments['tokens'].dtype)
        staged = [config.block_m * family.output_columns(config.block_n, (dtype,)) * out
                  for dtype, out in zip(hooks.out_dtypes or (ba.arguments['out_dtype'],), outs, strict=True)]
        sides = [2 * _side_tile_bytes((side,), (axes,), extents)
                 for side, axes in zip(epilogue_sides, hooks.epilogue_axes, strict=True)]
        barriers = 2 * config.stages * len(streams) + 2 * (len(epilogue_sides) + len(prologue_sides) + 1)
        # mma.sync's too, though its output's staging is scoped after its pipeline's stages: Mosaic places
        # both at once (an L4 refused 114688 bytes for a 98304-byte pipeline and its 16384-byte output,
        # GEN4-gpu-l4).
        return (mosaic.allocated(*(config.stages * stream for stream in streams), *staged, *sides)
                + mosaic.BARRIER_BYTES * barriers)

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
        # (a commit to shared memory): serial with the MMAs, not beside them
        # (GMMSB-h100-main: a gated prologue at 3.7x the Triton launch's time).
        family = _bundle(config, tokens.dtype)
        # A step's stage written (its rows' and columns' blocks), and read by the products: each of the
        # family's instructions spans `rows` output rows and the tile's columns, reading its rows' slice
        # and the columns' whole block, so the columns' block is read once a `rows` of the tile (an H100's
        # kernel gradients at 64 columns, which read most a FLOP, ran 1.08 to 1.43 times the best tile,
        # R1/R3-sweep-h100).
        reads = config.block_k * size * (config.block_m + config.block_m * config.block_n / family.rows)
        step = (config.block_m + config.block_n) * config.block_k * size + reads
        fragments = 0.0
        if family.operands_in_registers:
            # Its products read their operands from registers the consumer's warps load from the stage
            # before each MMA, the left operand's rows once and the right operand's whole block in each of
            # its `rhs_copies` warps: in the same warps as the MMAs they feed, serial with them, where
            # warpgroup MMA reads the stage itself beside its products (an A100 ran mma.sync's tiles at
            # 1.2 to 1.5 times Triton's time, an L4 at 0.5 to 0.9, GEN4-route-a100, GEN6-route-l4).
            step = (config.block_m + config.block_n) * config.block_k * size
            fragments = steps * (family.lhs_copies * config.block_m
                                 + family.rhs_copies * config.block_n) * config.block_k * size
        rewritten = 0.0
        if hooks.prologue is not None:
            rewritten = steps * (2 * math.prod(lhs) * size
                                 + 2 * _side_tile_bytes(prologue_sides, hooks.prologue_axes, extents))
        prologue_exponentials = steps * math.prod(lhs) * _hook_work(
            hooks.before(), tokens.dtype, prologue_sides, hooks.prologue_axes, lhs, extents)
        epilogue_exponentials = config.block_m * config.block_n * _hook_work(
            hooks.after(), jnp.float32, epilogue_sides, hooks.epilogue_axes, (config.block_m, config.block_n),
            extents)
        return cost.Cost(
            programs=tiles, device_bytes=moved,
            program_flops=2 * steps * config.block_m * config.block_n * config.block_k,
            program_staged=steps * step + config.block_m * config.block_n * out,
            program_exponentials=epilogue_exponentials, mode=_mode(tokens.dtype), steps=steps,
            stages=config.stages, serial_staged=rewritten + fragments,
            serial_exponentials=prologue_exponentials,
            # Warpgroup MMA's walk keeps a step's products running past it (`mosaic_gpu._IN_FLIGHT`);
            # mma.sync's products are synchronous (`mosaic_sm80.Mma.walk`).
            in_flight=0 if family.operands_in_registers else mosaic_gpu._IN_FLIGHT,
            serial_stored=(config.block_m * config.block_n * out
                           if self.STORES_SERIALLY and config.warpgroups == 1 else 0))

    def program_shared(self, config: mosaic_gpu.Tiles, target) -> int:
        # The programs its family runs on a multiprocessor at once share its shared memory.
        return mosaic.program_shared(super().program_shared(config, target), _programs(config))

    def _launched(self, config: mosaic_gpu.Tiles, interpret: bool) -> int:
        """The programs the persistent kernel launches: its family's programs on each multiprocessor
        (`Family.programs`); the interpreter runs one block (jax/_src/pallas/mosaic_gpu/interpret)."""
        return 1 if interpret else _multiprocessors(generation.gpu_target()) * _programs(config)

    @staticmethod
    def _in_flight(config: mosaic_gpu.Tiles, target) -> float:
        """The row tiles a band of the planar snake keeps in flight at once on `target`."""
        return max(1.0, _multiprocessors(target) * _programs(config) / config.grid_block_n)


@dataclasses.dataclass(frozen=True)
class MosaicGmm(_Mosaic):
    """`Gmm` through the Mosaic GPU kernels (`mosaic_gpu.gmm`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return -(-tokens.shape[0] // max(kernel.shape[0], 1)), depth, width

    def _divides(self, ba, config: mosaic_gpu.Tiles) -> bool:
        kernel = ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return depth % config.block_k == 0 and width % config.block_n == 0

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
        moved = (bands * rows * depth * size + matrices + _written(rows * width, ba, config.block_n)
                 + _side_bytes(ba.arguments['prologue_operands'], hooks.prologue_axes, rereads)
                 + _side_bytes(ba.arguments['epilogue_operands'], hooks.epilogue_axes, rereads))
        return self._cost(ba, config, tiles * columns, depth / config.block_k, moved)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: mosaic_gpu.Tiles, return_residuals: bool = False):
        # Never set: the launch is not eligible for exact sums; a dense call is one group, as any.
        del exact_sums, dense
        return mosaic_gpu.gmm(tokens, kernel, sizes, tiles=config, transpose_rhs=trans_rhs,
                              out_dtype=out_dtype, multiprocessors=self._launched(config, interpret),
                              interpret=INTERPRETED if interpret else False,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class MosaicTgmm(_Mosaic):
    """`Tgmm` on Hopper (`mosaic_gpu.tgmm`): `block_m` x `block_n` tiles of each group's gradient, its rows
    `block_k` at a time."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    EXACT_SUMS: ClassVar[bool] = True
    STORES_SERIALLY: ClassVar[bool] = True

    def _generated_extents(self, ba) -> tuple[int, int | None, int]:
        # A gradient tile's rows are the tokens' width; its steps run over a group's rows, padded to whole
        # steps (`mosaic_gpu.tgmm`).
        return ba.arguments['tokens'].shape[1], None, ba.arguments['cotangent'].shape[1]

    def _divides(self, ba, config: mosaic_gpu.Tiles) -> bool:
        rows, _, width = self._generated_extents(ba)
        return rows % config.block_m == 0 and width % config.block_n == 0

    def legal(self, ba, config: mosaic_gpu.Tiles, target) -> bool:
        """`_Mosaic.legal`, a family whose products read registers loading its transposed stage in whole
        roles of rows (the step, `block_k`: a 32-row stage failed to cast to the 64-row layout, ns-regs2),
        and with `exact_sums` a second fp32 accumulator, the running sum beside a step's, in each
        consumer's live set (`mosaic.tile_registers`) within its family's registers."""
        if not super().legal(ba, config, target):
            return False
        family = _bundle(config, ba.arguments['tokens'].dtype)
        if family.operands_in_registers and config.block_k % family.role_rows:
            return False
        if not ba.arguments.get('exact_sums'):
            return True
        dtype = ba.arguments['tokens'].dtype
        block = (config.block_m, config.block_n, config.block_k)
        running = -(-config.block_m * config.block_n // config.warpgroups // mosaic.THREADS)
        return (mosaic.tile_registers(family, config.warpgroups, block, dtype) + running
                <= family.registers(config.warpgroups))

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
                 + cost.written(groups * depth * width * out, config.block_n * out)
                 + _side_bytes(ba.arguments['prologue_operands'], hooks.prologue_axes, rereads))
        steps = _block_tiles(rows, groups, config.block_k) / groups
        return self._cost(ba, config, groups * blocks[0] * blocks[1], steps, moved)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: mosaic_gpu.Tiles,
             return_residuals: bool = False):
        del dense  # a dense call is one group, as any
        return mosaic_gpu.tgmm(tokens, cotangent, sizes, tiles=config, out_dtype=out_dtype,
                               multiprocessors=self._launched(config, interpret),
                               interpret=INTERPRETED if interpret else False, prologue=hooks.before(),
                               prologue_operands=prologue_operands, prologue_axes=hooks.prologue_axes,
                               exact_sums=exact_sums), None


@functools.cache
def cutlass_architectures() -> tuple[str, ...]:
    """The GPU generations the installed dew_cutlass holds code for, or none where it is not installed."""
    if importlib.util.find_spec('dew_cutlass') is None:
        return ()
    return importlib.import_module('dew_cutlass').ARCHITECTURES


@functools.cache
def _cutlass_registered() -> None:
    """dew_cutlass's FFI targets, registered with jax once, at the first launch that runs one."""
    for target, capsule in importlib.import_module('dew_cutlass').targets().items():
        jax.ffi.register_ffi_target(target, capsule, platform="CUDA")


class _Cutlass(_GroupedLaunch[mosaic_gpu.Tiles]):
    """CUTLASS's grouped GEMMs (`kernels/cutlass`, the dew-cutlass wheel) as XLA FFI calls: on Hopper,
    CUTLASS 3's pointer-array cooperative kernel, a 128 x 256 x 64 tile on a pair of multiprocessors, its
    stages the shared memory its epilogue leaves, bf16 operands and bf16 or fp32 out, each group's shape
    set up on the device. A vendored kernel a call runs only by name (`generation.chosen`), at its one
    `TILE`, which the model does not cost."""
    name: ClassVar[str] = 'cutlass'
    config_cls: ClassVar[type[mosaic_gpu.Tiles]] = mosaic_gpu.Tiles
    TILE: ClassVar[mosaic_gpu.Tiles]

    def eligible(self, ba, target) -> bool:
        operands = [value for name, value in ba.arguments.items()
                    if name in ('tokens', 'kernel', 'cotangent')]
        # 16-byte aligned rows of every operand and output, as its TMA copies and each group's pointers
        # need: every extent a multiple of 8 elements.
        extents = {extent for operand in operands for extent in operand.shape[1:]}
        built = target.generation
        return (built in cutlass_architectures()
                and all(operand.dtype == jnp.bfloat16 for operand in operands)
                and jnp.dtype(ba.arguments['out_dtype']) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))
                and ba.arguments['sizes'].dtype == jnp.int32 and all(extent % 8 == 0 for extent in extents)
                and ba.arguments['hooks'] == NO_HOOKS and not ba.arguments.get('exact_sums')
                and not ba.arguments['interpret'])

    def space(self, ba) -> tuple[mosaic_gpu.Tiles, ...]:
        return (self.TILE,)

    def fitting(self, ba, target) -> tuple[mosaic_gpu.Tiles, ...]:
        # CUTLASS's builder fits the stages to the shared memory (`StageCountAutoCarveout`).
        return self.space(ba)

    def _get_heuristics_config(self, ba) -> mosaic_gpu.Tiles:
        return self.TILE


@dataclasses.dataclass(frozen=True)
class CutlassGmm(_Cutlass):
    """`Gmm` through CUTLASS (`dew_cutlass_grouped_gemm`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    TILE: ClassVar[mosaic_gpu.Tiles] = mosaic_gpu.Tiles(128, 256, 64, 4, 8, warpgroups=2, cluster=2)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: mosaic_gpu.Tiles, return_residuals: bool = False):
        # No hooks, side operands or exact sums: the launch is not eligible for them; a dense call is one
        # group, as any.
        del prologue_operands, epilogue_operands, interpret, hooks, exact_sums, dense, config
        _cutlass_registered()
        width = kernel.shape[1] if trans_rhs else kernel.shape[2]
        out = jax.ShapeDtypeStruct((tokens.shape[0], width), jnp.dtype(out_dtype))
        return jax.ffi.ffi_call('dew_cutlass_grouped_gemm', out)(tokens, kernel, sizes,
                                                                 transpose_w=trans_rhs), None


@dataclasses.dataclass(frozen=True)
class CutlassTgmm(_Cutlass):
    """`Tgmm` through CUTLASS (`dew_cutlass_grouped_gemm_k`): each group's gradient a GEMM over its rows,
    an empty group's zero, its pair splitting the gradient's rows."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    TILE: ClassVar[mosaic_gpu.Tiles] = mosaic_gpu.Tiles(128, 256, 64, 4, 8, warpgroups=2)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: mosaic_gpu.Tiles,
             return_residuals: bool = False):
        del prologue_operands, interpret, hooks, exact_sums, dense, config  # as `CutlassGmm._fwd`'s
        _cutlass_registered()
        out = jax.ShapeDtypeStruct((sizes.shape[0], tokens.shape[1], cotangent.shape[1]),
                                   jnp.dtype(out_dtype))
        return jax.ffi.ffi_call('dew_cutlass_grouped_gemm_k', out)(tokens, cotangent, sizes), None


class _MosaicTpu(_GroupedLaunch[mosaic_tpu.Blocks]):
    """The TPU mainloops (`templates.gemm.mosaic_tpu`, Megablox's schedule): bf16 operands, which the
    MXUs multiply exactly and sum in fp32, widths a vreg's lanes divide, and a TPU whose scoped VMEM a
    measurement states; a CPU interprets them as the TPU runs them, and a TPU's own lowering runs them
    (`_on_platform`'s TPU branch), their native qualification open until a scoreboard row measures them."""
    name: ClassVar[str] = 'mosaic_tpu'
    config_cls: ClassVar[type[mosaic_tpu.Blocks]] = mosaic_tpu.Blocks
    GENERATES: ClassVar[bool] = True
    ROW_AXIS: ClassVar[int]
    """The block axis (`block_m`, `block_n`, `block_k`: 0, 1, 2) that walks the grouped rows."""

    def eligible(self, ba, target) -> bool:
        # Not for exact sums: the MXUs add within a step, in fp32 by Google's account, IEEE by none
        # (`GROUPED`'s 'step_sums' routes such a product to XLA).
        first, second = (value.dtype for name, value in ba.arguments.items()
                         if name in ('tokens', 'kernel', 'cotangent'))
        rules = mosaic_tpu_lowering.lowering_rules[tpu_core.CoreType.TC]
        return (generation.tpu(target) and target.scoped_vmem is not None
                and first == second == jnp.bfloat16 and _rated(target, first)
                and self.derived(ba, target) is not None and _hooks_lower(ba, first, rules)
                and not ba.arguments.get('exact_sums'))

    @abc.abstractmethod
    def spans(self, ba) -> tuple[int, int, int]:
        """The extents the `block_m`, `block_n` and `block_k` axes span at `ba`."""

    def derived(self, ba, target) -> mosaic_tpu.Blocks | None:
        """The one block the launch takes on `target` (docs/design/kernel-templates.md §7), from Mosaic
        TPU's facts and the call's extents alone: each axis the widest doubling of a vreg's lanes
        (`mosaic_tpu.LANES`) that divides its extent, the grouped rows' at most a group's rows doubled
        to cover them (a block a group boundary crosses runs once for each group), each at least what
        its one-dimensional side operands lay out at (`_lays_out`); then, until its buffers fit the
        scoped VMEM (`legal`), its largest axis halved, the step's before the output's, never below that
        least: the output block as square as the budget leaves it, which reads each operand fewest times.
        None where no block is legal."""
        spans, groups = self.spans(ba), ba.arguments['sizes'].shape[0]
        least = self._least_blocks(ba)
        blocks = []
        for axis, extent in enumerate(spans):
            if axis == self.ROW_AXIS:
                widest = max(_covering(-(-extent // max(groups, 1))), least[axis])
            else:
                dividing = [block for block in _doubling(mosaic_tpu.LANES, extent) if extent % block == 0]
                if not dividing:
                    return None
                widest = dividing[-1]
            blocks.append(widest)
        while not self.legal(ba, config := mosaic_tpu.Blocks(*blocks), target):
            halved = [axis for axis in (2, 0, 1) if blocks[axis] // 2 >= least[axis]]
            if not halved:
                return None
            # Half of a block that divides its extent divides it too.
            axis = max(halved, key=lambda axis: blocks[axis])
            blocks[axis] //= 2
        return config

    def neighbours(self, ba, config: mosaic_tpu.Blocks) -> tuple[mosaic_tpu.Blocks, ...]:
        # The derived blocks with one axis halved or doubled.
        return tuple(dataclasses.replace(config, **{name: value})
                     for name in ('block_m', 'block_n', 'block_k')
                     for value in (getattr(config, name) // 2, getattr(config, name) * 2) if value > 0)

    def _least_blocks(self, ba) -> list[int]:
        """The least block of each axis (`block_m`, `block_n`, `block_k`): a vreg's lanes, or the row of
        128 32-bit lanes a one-dimensional side operand along it lays out at short of its whole extent
        (`_lays_out`: 256 elements of a 16-bit one)."""
        least = [mosaic_tpu.LANES] * 3
        index = self.extents(mosaic_tpu.Blocks(0, 1, 2))
        hooks = ba.arguments['hooks']
        for name, axes_of in (('prologue_operands', hooks.prologue_axes),
                              ('epilogue_operands', hooks.epilogue_axes)):
            for side, axes in zip(ba.arguments.get(name, ()), axes_of, strict=True):
                if len(axes) == 1:
                    axis = index[axes[0]]
                    least[axis] = max(least[axis], 128 * 4 // jnp.dtype(side.dtype).itemsize)
        return least

    def legal(self, ba, config: mosaic_tpu.Blocks, target) -> bool:
        """Whether `config`'s blocks are whole vregs' lanes dividing their extents (the grouped rows' any),
        lay out every side operand (`_lays_out`) and, double-buffered with the fp32 accumulator, fit the
        TPU's scoped VMEM (`Template.legal`'s rule, VMEM for a block's shared memory)."""
        budget = target.scoped_vmem
        blocks = (config.block_m, config.block_n, config.block_k)
        return (budget is not None
                and all(block % mosaic_tpu.LANES == 0 and (axis == self.ROW_AXIS or extent % block == 0)
                        for axis, (extent, block) in enumerate(zip(self.spans(ba), blocks, strict=True)))
                and self._lays_out(ba, config) and self.staged_bytes(ba, config, target) <= budget)

    def fitting(self, ba, target) -> tuple[mosaic_tpu.Blocks, ...]:
        """The derived block where it is legal (`Template.fitting`), else raises with the TPU's scoped
        VMEM for the budget."""
        fits = self.fits(ba, target)
        if not fits:
            raise base.Unfitted(type(self).__name__, None, target.scoped_vmem or 0,
                                target.generation)
        return fits

    def staged_bytes(self, ba, config: mosaic_tpu.Blocks, target) -> int:
        # Pallas's pipeline holds two of each operand's and output's blocks (one in flight), beside the
        # fp32 accumulator; Mosaic TPU may hold each operand block's value and the product's fp32 tile as
        # temporaries beside them, at most (a v5e refused 20.33 and 20.95 MiB at a 1024 x 512 x 2048
        # product and kernel gradient, 16.12 at a 256 x 512 x 4096 gradient, whose buffers alone counted
        # 16.00, 16.00 and 13.0: the compile gate's libtpu runs).
        size = jnp.dtype(ba.arguments['tokens'].dtype).itemsize
        hooks = ba.arguments['hooks']
        extents = self.extents(config)
        outs = _output_bytes(ba)
        sides = (_side_tile_bytes(ba.arguments['prologue_operands'], hooks.prologue_axes, extents)
                 + _side_tile_bytes(ba.arguments.get('epilogue_operands', ()), hooks.epilogue_axes, extents))
        operands = self.loaded(config) * size + sides
        rows, columns = self.output(config)
        temporaries = operands + 4 * rows * columns
        return 2 * (operands + outs * rows * columns) + 4 * rows * columns + temporaries

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

    ROW_AXIS: ClassVar[int] = 0

    def spans(self, ba) -> tuple[int, int, int]:
        kernel = ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return ba.arguments['tokens'].shape[0], width, depth

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
             prologue_operands: tuple = (), epilogue_operands: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: mosaic_tpu.Blocks, return_residuals: bool = False):
        # Never set: the launch is not eligible for exact sums; a dense call is one group, as any.
        del exact_sums, dense
        return mosaic_tpu.gmm(tokens, kernel, sizes, blocks=config, transpose_rhs=trans_rhs,
                              out_dtype=out_dtype, interpret=TPU_INTERPRETED if interpret else False,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class MosaicTpuTgmm(_MosaicTpu):
    """`Tgmm` on a TPU (`mosaic_tpu.tgmm`): `block_m` x `block_n` blocks of each group's gradient, its rows
    `block_k` at a time."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    ROW_AXIS: ClassVar[int] = 2

    def spans(self, ba) -> tuple[int, int, int]:
        tokens = ba.arguments['tokens']
        return tokens.shape[1], ba.arguments['cotangent'].shape[1], tokens.shape[0]

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
    that states none runs XLA's grouped matmul, needing no estimate (`GROUPED`'s 'rate')."""
    return target.tensor_rate(_mode(dtype)) is not None


@dataclasses.dataclass(frozen=True)
class GroupedCase:
    """A grouped product as the grouped selector reads it (`GROUPED`): whether the kernels run where
    the call does (`placed`: a GPU's, or a CPU's interpreting them; the template's dispatch runs in the
    branch its platform places), whether they see its operands whole (`whole`,
    `dew.nn.sharding.seen_whole`), the `product` it declares, its `rows` and its launches bound to it
    (`call`)."""

    placed: bool
    whole: bool
    product: Product
    rows: int
    call: base.Call


_Need = selection.Requirement[GroupedCase]
"""A requirement of a `GroupedCase`."""

SUPPORTS = (
    _Need('capability', lambda case: case.placed,
          "the kernels need a GPU of sm80 or later, or a CPU interpreting them where the call asks",
          row=INHERENT),
    _Need('placement', lambda case: case.whole,
          "a mesh axis splits the operands, or a shard_map that checks varying axes holds them, and the "
          "kernels see whole arrays", row='ledger 1306'),
    _Need('product', lambda case: case.product.refusal is None, lambda case: case.product.refusal or '',
          row='ledger 1295'),
    _Need('rate', lambda case: _rated(case.call.target, case.product.compute),
          lambda case: f"{case.call.target.generation} states no {_mode(case.product.compute)} tensor rate, "
                       "which the kernels' cost model reads", row='ledger 1295'),
    _Need('step_sums', lambda case: not (case.product.promoted and generation.tpu(case.call.target)),
          lambda case: f"{case.call.target.generation}'s MXUs add within a step, which no measurement states "
                       "IEEE", row=INHERENT),
    _Need('rows', lambda case: case.rows > 0, "the call has no rows", row=INHERENT),
)
"""What every launch needs of a grouped product: a process that runs it, whole operands, a product the
kernels compute exactly on the target (its declaration's refusal, the target's tensor rate for its
dtype, no step-by-step sums on a TPU, whose launches do not add them apart), and rows to launch."""


def _launch_kernel(name: str, *measured: _Need) -> selection.Kernel[GroupedCase]:
    """`GROUPED`'s kernel for the launches implementation `name` runs: `SUPPORTS`, then its own
    eligibility for the call (`Template.eligible`), then what the targets measured of it (`measured`, which
    a call naming it overrides)."""
    return selection.Kernel(name, (*SUPPORTS, _Need(
        'launch', lambda case: case.call.eligible(name),
        lambda case: f"the {name} launches take no call of these dtypes, widths, side operands or hooks on "
                     f"{case.call.target.generation}", row='ledger 1295'), *measured))


OLD_ROUTES: dict[tuple[str, str, str], tuple[str, str]] = {
    ('mma.sync', 'forward', 'grouped'):
        ('triton', "R10 (852bb4bd3, A100): the template's mma.sync tile 1.017x Triton's time, geomean of 6 "
                   "grouped MoE rows"),
    ('mma.sync', 'input_gradient', 'grouped'):
        ('triton', "R10: 1.263x Triton's time, geomean of 6 grouped rows"),
    ('mma.sync', 'kernel_gradient', 'grouped'):
        ('triton', "R10: 1.093x Triton's time, geomean of 6 grouped rows"),
    ('mma.sync', 'forward', 'dense'):
        ('xla', "DENSE1 (A100, 2d1bbd051): Mosaic 1.923x and Triton 1.947x XLA's dot (cuBLAS), geomean of 10 "
                "projections at 8192 tokens"),
    ('mma.sync', 'input_gradient', 'dense'):
        ('xla', "DENSE1 (A100): Mosaic 2.881x, Triton 1.839x XLA's dot"),
    ('mma.sync', 'kernel_gradient', 'dense'):
        ('xla', "DENSE1 (A100): Mosaic 1.964x, Triton 1.735x XLA's dot"),
    ('wgmma', 'forward', 'dense'):
        ('xla', "DENSE1 (H100, 2d1bbd051): Mosaic 1.067x and Triton 1.616x XLA's dot (cuBLAS), geomean of 10 "
                "projections at 8192 tokens"),
    ('wgmma', 'input_gradient', 'dense'): ('xla', "DENSE1 (H100): Mosaic 1.048x, Triton 1.738x XLA's dot"),
    ('wgmma', 'kernel_gradient', 'dense'): ('xla', "DENSE1 (H100): Mosaic 1.049x, Triton 1.149x XLA's dot"),
}
"""The cells where an old route measured faster than the template, each with its receipt (owner, bible
da64ae5): (instruction family, product, grouped or dense) to (the old route, the receipt). A listed cell
runs its old route: every launch before it in `GROUPED`'s order steps aside (`_measured_slower`). The
template is the default on every other cell; a cell leaves the list once the template measures at least as
fast on it (geomean over the receipt's rows). Nothing is ranked at runtime. A call the template cannot run
(its legality) takes the next route in `GROUPED`'s order, which is no cell of this list."""


def _cell(case: "GroupedCase") -> tuple[str, str, str] | None:
    """The `OLD_ROUTES` cell a call is: its GPU's instruction family, its product (a kernel gradient,
    exactly summed or not, or an input gradient or a forward) and whether it is dense; None off a CUDA
    GPU."""
    target = case.call.target
    if generation.capability(target.generation) is None:
        return None
    arguments = next(iter(case.call.bound.values()))[1].arguments
    if 'trans_rhs' not in arguments:
        product = 'kernel_gradient exact' if arguments.get('exact_sums') else 'kernel_gradient'
    else:
        product = 'input_gradient' if arguments['trans_rhs'] else 'forward'
    shape = 'dense' if arguments.get('dense') else 'grouped'
    return ('wgmma' if target.wgmma else 'mma.sync', product, shape)


def _old_route_reason(case: "GroupedCase") -> str:
    """Why the template leaves a listed cell to its old route: the cell and its receipt."""
    cell = _cell(case)
    route, receipt = OLD_ROUTES[cell] if cell is not None else ('', '')
    return f"{route} measured faster on this cell ({', '.join(cell or ())}): {receipt}"


def _measured_slower(name: str) -> _Need:
    """What keeps launch `name` off an `OLD_ROUTES` cell whose old route is another, unless named, so the
    old route runs it."""
    def holds(case: "GroupedCase") -> bool:
        cell = _cell(case)
        return cell not in OLD_ROUTES or OLD_ROUTES[cell][0] == name

    return _Need('old_route', holds, _old_route_reason, row='bible da64ae5', kind='measured')


NAMED = _Need('named', lambda case: False,
              lambda case: "this launch runs only where a call names it (`generation.chosen`)",
              row="no production route", kind='measured')
"""What keeps CUTLASS's launches off every call that does not name them: a vendored kernel that was never a
production route, which no measurement ranks against the template's."""


"""The bundles by name, as a tile names its (`mosaic_gpu.Tiles.family`), for their class-wide facts."""


def _bundle(config: mosaic_gpu.Tiles, dtype) -> mosaic.Family:
    """The bundle `config` runs on, for operands of `dtype` (`mosaic_gpu.family_of`)."""
    return mosaic_gpu.family_of(config, dtype)


def _programs(config: mosaic_gpu.Tiles) -> int:
    """The programs of a launch at `config` a multiprocessor runs at once (`mosaic.Family.programs`)."""
    return _bundle(config, jnp.bfloat16).programs(config.warpgroups)


_GMMS = (MosaicGmm(), MosaicTpuGmm(), Gmm(), CutlassGmm())
_TGMMS = (MosaicTgmm(), MosaicTpuTgmm(), Tgmm(), CutlassTgmm())

GROUPED: selection.Op[GroupedCase] = selection.Op('grouped_matmul', (
    *(_launch_kernel(launch.name, *((NAMED,) if launch.name == 'cutlass' else
                                    (_measured_slower(launch.name),)))
      for launch in _GMMS),
    selection.Kernel[GroupedCase]('xla', (), generic=True)))
"""The grouped selector, in a fixed order, the template first (owner, bible da64ae5): Mosaic GPU's launch
where it takes the call, the TPU's launch on a TPU, Triton's where they refuse, else XLA's program, but an
`OLD_ROUTES` cell runs its old route; CUTLASS's launches only by name (`NAMED`). Every
implementation shares the template's signature, so each binds the call's own arguments, `_GMMS`' for a
product, `_TGMMS`' for a kernel gradient."""


def grouped_case(launches, arguments, options, *, product: Product, whole: bool,
                 placed: bool = True) -> GroupedCase:
    """The `GroupedCase` of `launches` called with `arguments` and `options` (`gmm_call`, `tgmm_call`)."""
    return GroupedCase(placed, whole, product, arguments[0].shape[0],
                       base.call(launches, *arguments, **options))


def launched(launches, arguments, options, *, product: Product, whole: bool):
    """The launch of `launches` that runs the call (`arguments`, `options`) as `GROUPED` selects it,
    ready to call, or None where XLA's program takes it. A CPU process's GPU branch, which runs only where
    a lowering for the GPU takes it, takes a chosen implementation where it can and the order's next
    launch where it cannot (mma.sync's where this JAX lowers none: `_families`); every branch a process
    runs insists on it."""
    case = grouped_case(launches, arguments, options, product=product, whole=whole)
    unrun = not options['interpret'] and jax.default_backend() == 'cpu'
    named = selection.forced(GROUPED.choice) if unrun else None
    route = selection.select(GROUPED, case, named, insist=not unrun)
    if route.kernel == 'xla':
        return None
    return functools.partial(case.call.implementation(route.kernel), *arguments, **options)


def gmm_call(tokens, kernel, sizes, prologue_operands: tuple = (), epilogue_operands: tuple = (), *,
             out_dtype, trans_rhs: bool, hooks: Hooks = NO_HOOKS, interpret: bool, promoted: bool = False,
             dense: bool = False) -> tuple:
    """The launches, arguments and options of each group's rows of `tokens` times its matrix of `kernel`,
    both of one dtype, with the template's hooks (`templates.gemm.template.gemm`). `promoted`,
    the declared product's (`templates.product.Product`), adds each depth step's partial apart
    (`exact_sums`); `dense`, one group of every row (a dense product), the plain tiled kernel where Triton
    runs it."""
    arguments = (tokens, kernel, sizes, tuple(prologue_operands), tuple(epilogue_operands))
    options = {'out_dtype': jnp.dtype(out_dtype).name, 'trans_rhs': trans_rhs, 'hooks': hooks,
               'interpret': interpret, 'exact_sums': promoted, 'dense': dense}
    return _GMMS, arguments, options


def tgmm_call(tokens, cotangent, sizes, prologue_operands: tuple = (), *, out_dtype,
              hooks: Hooks = NO_HOOKS, interpret: bool, promoted: bool = False, dense: bool = False) -> tuple:
    """The launches, arguments and options of each group's rows of `tokens` transposed times its rows of
    `cotangent`, both of one dtype, with the template's prologue
    (`templates.gemm.template.gemm_t`); `promoted` and `dense` as `gmm_call`'s."""
    arguments = (tokens, cotangent, sizes, tuple(prologue_operands))
    options = {'out_dtype': jnp.dtype(out_dtype).name, 'hooks': hooks, 'interpret': interpret,
               'exact_sums': promoted, 'dense': dense}
    return _TGMMS, arguments, options


def _on_platform(kernel, fallback, interpret_on_cpu: bool):
    """`kernel` where the call lowers for its target's platform (CUDA, or a TPU's), and `fallback`
    everywhere else: a TPU's program, which never runs on CUDA, holds no CUDA branch, and a GPU's no TPU
    branch. A test's `interpret_on_cpu` interprets the kernel in a CPU process, as the GPU or TPU it
    emulates (`generation.assumed`), which it must state: interpreted for no device's facts, a launch
    would have none to derive its tile from, and the call would quietly run `fallback`. A new family
    member runs its template after the hardware-free checks, its native qualification an open item until
    measured (bible §6)."""
    target = generation.gpu_target()
    tpu = generation.tpu(target)
    branches = {'tpu' if tpu else 'cuda': functools.partial(kernel, interpret=False)}
    if interpret_on_cpu and jax.default_backend() == 'cpu':
        if not tpu and generation.capability(target.generation) is None:
            raise ValueError("interpret_on_cpu interprets the kernel as the GPU or TPU this CPU process "
                             "emulates; state it with generation.assumed(...)")
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
                                 preferred_element_type=at_least_fp32(x.dtype))
    return accumulated.astype(out_dtype or x.dtype)


def _portable_projection_jvp(dtype: Dtype | None, precision: PrecisionLike, out_dtype: Dtype | None,
                    primals: tuple[jax.Array, jax.Array, jax.Array],
                    tangents: tuple[jax.Array, jax.Array, jax.Array]
                    ) -> tuple[jax.Array, jax.Array | SymbolicZero]:
    x, kernel, group_sizes = primals
    dx, dkernel, _ = tangents
    output = jnp.asarray(portable_projection(x, kernel, group_sizes, dtype, precision, out_dtype))
    work = at_least_fp32(jnp.result_type(output.dtype, x.dtype, kernel.dtype))
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


