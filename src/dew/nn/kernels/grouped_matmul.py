"""The MoE grouped matmul: its kernels' launches and the custom VJP that trains through them.

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
from flax.typing import Dtype, PrecisionLike
from jax._src.pallas.mosaic_gpu import (
    core as plgpu_core,  # the lowering's semantics (`_lowers_on_mosaic`)
    lowering as mosaic_gpu_lowering,
)
from jax.core import ShapedArray
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import core as jax_core

from ..precision import asks_default_precision
from . import generation
from .templates import base, cost
from .templates.gemm import mosaic_gpu


def ragged_dot_refusal(compute: Dtype, operands: tuple[Dtype, ...],
                       precision: PrecisionLike) -> str | None:
    """Why the kernels do not compute the product a caller asked for, or None where they do.

    The kernels multiply in `compute` and accumulate in fp32. With 16-bit
    compute that gives exact products, summed by the tensor cores, whose fp32
    accumulation truncates each add: the default precision's sums, and an XLA
    dot's at it. A precision above the default (explicitly or through
    `jax_default_matmul_precision`) asks for IEEE fp32 sums, which an fp32
    output takes by adding each depth step's product apart (`exact_sums`);
    a 16-bit output's one rounding hides the difference. With fp32 compute
    the kernels multiply in TF32, which only the default precision asks for.
    An operand or master wider than fp32 needs its gradient summed wider than
    the kernels sum, and x64 widens their int32 group offsets, so both are
    refused.
    """
    if jax.config.jax_enable_x64:
        return "x64 widens the kernels' int32 group offsets"
    if jnp.result_type(compute, *operands, jnp.float32) != jnp.dtype(jnp.float32):
        return "an operand is wider than fp32, and the kernels sum gradients in fp32"
    compute = jnp.dtype(compute)
    if compute in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        return None
    if compute != jnp.dtype(jnp.float32):
        return f"the kernels do not multiply in {compute.name}"
    return None if asks_default_precision(precision, configured=True) else (
        "fp32 products above the default precision, and the kernels multiply in TF32")


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

MOSAIC_TILES = tuple(mosaic_gpu.Tiles(*shape) for shape in itertools.product(
    (128, 192, 64), (128, 192, 64), (64,), (4, 5, 2), (8, 4, 2, 1)))
"""Every tile the Mosaic GPU launches may take: the sweep JAX ships with the kernels
(`ragged_dot_mgpu.main`), where an H100's fastest lie (GMM-h100-3). Where the cost model ties, the first
wins."""

INTERPRETED = plgpu.InterpretGPUParams(out_of_bounds_reads="uninitialized")
"""How a CPU interprets the Mosaic GPU launches: a read past an operand's rows, which a TMA load zeroes,
gives the poison every buffer starts with (NaN), as an output a kernel fails to store does, so the
interpreter shows both."""


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


def _lowers_on_mosaic(hook, tile, sides) -> bool:
    """Whether Pallas's Mosaic GPU lowering has a rule, at warpgroup semantics, for every primitive of
    `hook` on `tile` and its side tiles: it has none for a transpose, which Triton's lowering takes."""
    if hook is None:
        return True
    rules = mosaic_gpu_lowering.mosaic_lowering_rules[plgpu_core.WGxWG_SEMANTICS]
    return all(equation.primitive in rules
               for equation in _equations(jax.make_jaxpr(hook)(tile, *sides).jaxpr))


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
        # Every CUDA GPU the dispatcher sends the kernels to (`dew.nn.moe.grouped_matmul_kernel`), and
        # a CPU, which interprets them, or traces the CUDA branch a GPU's lowering takes.
        return _rated(target, ba.arguments['tokens'].dtype)

    def space(self, ba) -> tuple[Tile, ...]:
        return TILES

    def fitting(self, ba, target) -> tuple[Tile, ...]:
        """The tiles that fit `target` (`Template.fitting`) whose accumulator (two with `exact_sums`)
        stays in its threads' registers, and on Hopper those whose dot Triton gives a warpgroup MMA: each
        warpgroup's share of
        the output rows a multiple of wgmma's 64. Either other kind ran 1.2 to 8.5 times slower than the
        fastest tile on an H100 (GMM-h100-2: an accumulator of 256 registers a thread spills; a 32-row
        share falls back to Ampere's MMA)."""
        wgmma = _facts(target).wgmma

        accumulators = 2 if ba.arguments.get('exact_sums') else 1

        def fits(tile: Tile) -> bool:
            rows, columns = self.output(tile)
            held = accumulators * rows * columns // (32 * tile.warps) < cost.THREAD_REGISTERS
            return held and not (wgmma and rows // max(tile.warps // 4, 1) % 64)

        return tuple(tile for tile in super().fitting(ba, target) if fits(tile))

    @staticmethod
    @abc.abstractmethod
    def output(tile: Tile) -> tuple[int, int]:
        """The rows and columns of a program's output tile, the dot's M and N."""

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
    def loaded(tile: Tile) -> int:
        return (tile.block_m + tile.block_n) * tile.block_k

    @staticmethod
    def output(tile: Tile) -> tuple[int, int]:
        return tile.block_m, tile.block_n

    @staticmethod
    def extents(tile: Tile) -> dict:
        return {'rows': tile.block_m, 'depth': tile.block_k, 'columns': tile.block_n}

    def cost(self, ba, config: Tile, target) -> float:
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
        chunks = _row_tiles(rows, groups, CHUNK_M)
        columns = math.ceil(width / config.block_n)
        tiles = _row_tiles(rows, groups, config.block_m)
        flops = 2 * tiles * config.block_m * depth * width
        steps = tiles * math.ceil(depth / config.block_k) / chunks
        staged = steps * _triton_staged(config.block_m, config.block_n, config.block_k, config.warps,
                                        _facts(target).wgmma) * size
        hooks, prologue_sides = ba.arguments['hooks'], ba.arguments['prologue_operands']
        epilogue_sides = ba.arguments['epilogue_operands']
        # Side operands are read again as the operands are: with each column block, and with each row
        # tile where they follow no rows; the hooks' special functions on every element they see.
        rereads = {'rows': tiles, 'columns': columns}
        moved = ((columns * rows * depth + tiles * depth * width) * size + rows * width * _output_bytes(ba)
                 + _side_bytes(prologue_sides, hooks.prologue_axes, rereads)
                 + _side_bytes(epilogue_sides, hooks.epilogue_axes, rereads))
        extents = self.extents(config)
        exponentials = (
            steps * config.block_m * config.block_k * _hook_work(
                hooks.before(), tokens.dtype, prologue_sides, hooks.prologue_axes,
                (config.block_m, config.block_k), extents)
            + tiles / chunks * config.block_m * config.block_n * _hook_work(
                hooks.after(), jnp.float32, epilogue_sides, hooks.epilogue_axes,
                (config.block_m, config.block_n), extents))
        return cost.seconds(cost.Cost(
            programs=chunks * columns, device_bytes=moved,
            program_flops=flops / (chunks * columns), program_staged=staged,
            program_exponentials=exponentials, mode=_mode(tokens.dtype), steps=steps, stages=config.stages,
            resident=self._resident(ba, config, target)), target)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             config: Tile, return_residuals: bool = False):
        # Imported at first use: the kernels need jax's Pallas Triton backend,
        # which jax 0.11.2 deprecates, and importing a model must not load it.
        from . import ragged_dot
        return ragged_dot.gmm(tokens, kernel, sizes, block_m=config.block_m, block_k=config.block_k,
                              block_n=config.block_n, num_warps=config.warps, num_stages=config.stages,
                              trans_rhs=trans_rhs, interpret=interpret, compute_dtype=tokens.dtype,
                              out_dtype=jnp.dtype(out_dtype), prologue_operands=prologue_operands,
                              epilogue_operands=epilogue_operands, exact_sums=exact_sums,
                              **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class GmmTransposed(Gmm):
    """`Gmm` against each matrix's transpose: the input gradient, tuned apart."""


@dataclasses.dataclass(frozen=True)
class Tgmm(_Triton):
    """Each group's `tokens` rows transposed times its `cotangent` rows: the
    kernel gradient `[groups, in, out]`."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    @staticmethod
    def loaded(tile: Tile) -> int:
        return tile.block_m * (tile.block_k + tile.block_n)

    @staticmethod
    def output(tile: Tile) -> tuple[int, int]:
        return tile.block_k, tile.block_n

    @staticmethod
    def extents(tile: Tile) -> dict:
        return {'rows': tile.block_m, 'depth': tile.block_k}

    def cost(self, ba, config: Tile, target) -> float:
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
                                        _facts(target).wgmma) * size
        hooks, sides = ba.arguments['hooks'], ba.arguments['prologue_operands']
        rereads = {'rows': groups, 'depth': blocks[0], 'columns': blocks[1]}
        moved = ((blocks[1] * rows * depth + blocks[0] * rows * width) * size + groups * depth * width * out
                 + _side_bytes(sides, hooks.prologue_axes, rereads))
        exponentials = steps * config.block_m * config.block_k * _hook_work(
            hooks.before(), tokens.dtype, sides, hooks.prologue_axes, (config.block_m, config.block_k),
            self.extents(config))
        return cost.seconds(cost.Cost(
            programs=programs, device_bytes=moved, program_flops=flops, program_staged=staged,
            program_exponentials=exponentials, mode=_mode(tokens.dtype), steps=steps, stages=config.stages,
            resident=self._resident(ba, config, target)), target)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, config: Tile, return_residuals: bool = False):
        from . import ragged_dot
        return ragged_dot.tgmm(tokens, cotangent, sizes, block_m=config.block_m, block_k=config.block_k,
                               block_n=config.block_n, num_warps=config.warps, num_stages=config.stages,
                               interpret=interpret, compute_dtype=tokens.dtype,
                               out_dtype=jnp.dtype(out_dtype), prologue=hooks.before(),
                               prologue_operands=prologue_operands, prologue_axes=hooks.prologue_axes,
                               exact_sums=exact_sums), None


class _Mosaic(_GroupedLaunch[mosaic_gpu.Tiles]):
    """JAX's Pallas Mosaic GPU kernels (`templates.gemm.mosaic_gpu`): Hopper's warpgroup MMA, one 16-bit
    dtype, and widths the space's tiles divide. Each runs one warpgroup a multiprocessor, persistent over
    the output tiles in a planar snake, so the rows a band of `grid_block_n` column tiles shares and the
    matrix block the row tiles in flight share come from the L2."""
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
        """Whether Mosaic GPU's lowering takes the launch's hooks (`_lowers_on_mosaic`), traced on 64-wide
        tiles: the primitives a hook makes do not depend on its tile's extents."""
        hooks = ba.arguments['hooks']

        def tiles(name, axes_of):
            return [jax.ShapeDtypeStruct((64,) * len(axes), side.dtype)
                    for side, axes in zip(ba.arguments.get(name, ()), axes_of, strict=True)]

        return (_lowers_on_mosaic(hooks.before(), jax.ShapeDtypeStruct((64, 64), dtype),
                                  tiles('prologue_operands', hooks.prologue_axes))
                and _lowers_on_mosaic(hooks.after(), jax.ShapeDtypeStruct((64, 64), jnp.float32),
                                      tiles('epilogue_operands', hooks.epilogue_axes)))

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
        return config.stages * step + epilogue + max(outs) * config.block_m * config.block_n

    @staticmethod
    @abc.abstractmethod
    def extents(tile: mosaic_gpu.Tiles) -> dict:
        """The block each launch axis takes at `tile`."""

    def _cost(self, ba, config: mosaic_gpu.Tiles, target, tiles: float, steps: float, moved: float) -> float:
        """`tiles` output tiles of `steps` contraction steps each, `moved` device bytes."""
        tokens, hooks = ba.arguments['tokens'], ba.arguments['hooks']
        size = jnp.dtype(tokens.dtype).itemsize
        out = jnp.dtype(ba.arguments['out_dtype']).itemsize
        extents = self.extents(config)
        prologue_sides = ba.arguments['prologue_operands']
        epilogue_sides = ba.arguments.get('epilogue_operands', ())
        lhs = (extents['rows'], extents['depth'])
        # Each step's blocks are written by TMA and read once by the warpgroup's MMA; a prologue reads
        # the stage of `lhs` and its side tiles and writes the stage back.
        step = 2 * (config.block_m + config.block_n) * config.block_k * size
        if hooks.prologue is not None:
            sides = _side_tile_bytes(prologue_sides, hooks.prologue_axes, extents)
            step += 2 * math.prod(lhs) * size + 2 * sides
        exponentials = (steps * math.prod(lhs) * _hook_work(hooks.before(), tokens.dtype, prologue_sides,
                                                            hooks.prologue_axes, lhs, extents)
                        + config.block_m * config.block_n * _hook_work(
                            hooks.after(), jnp.float32, epilogue_sides, hooks.epilogue_axes,
                            (config.block_m, config.block_n), extents))
        return cost.seconds(cost.Cost(
            programs=tiles, device_bytes=moved,
            program_flops=2 * steps * config.block_m * config.block_n * config.block_k,
            program_staged=steps * step + config.block_m * config.block_n * out,
            program_exponentials=exponentials, mode=_mode(tokens.dtype), steps=steps,
            stages=config.stages), target)

    def _multiprocessors(self, interpret: bool) -> int:
        # The interpreter runs one block (jax/_src/pallas/mosaic_gpu/interpret).
        return 1 if interpret else _multiprocessors(generation.gpu_target())

    @staticmethod
    def _in_flight(config: mosaic_gpu.Tiles, target) -> float:
        """The row tiles a band of the planar snake keeps in flight at once on `target`."""
        return max(1.0, _multiprocessors(_facts(target)) / config.grid_block_n)


@dataclasses.dataclass(frozen=True)
class MosaicGmm(_Mosaic):
    """`Gmm` on Hopper (`mosaic_gpu.gmm`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    def space(self, ba) -> tuple[mosaic_gpu.Tiles, ...]:
        kernel = ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return tuple(tile for tile in MOSAIC_TILES if depth % tile.block_k == 0 and width % tile.block_n == 0)

    @staticmethod
    def extents(tile: mosaic_gpu.Tiles) -> dict:
        return {'rows': tile.block_m, 'depth': tile.block_k, 'columns': tile.block_n}

    def cost(self, ba, config: mosaic_gpu.Tiles, target) -> float:
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        (rows, depth), groups = tokens.shape, kernel.shape[0]
        width = kernel.shape[1] if ba.arguments['trans_rhs'] else kernel.shape[2]
        size = jnp.dtype(tokens.dtype).itemsize
        tiles, columns = _block_tiles(rows, groups, config.block_m), width // config.block_n
        # The rows are read once a band; a group's matrix once for the row tiles of it in flight.
        bands = math.ceil(columns / config.grid_block_n)
        in_flight = self._in_flight(config, target)
        per_group = tiles / max(min(groups, rows), 1)
        matrices = _routed(rows, groups) * kernel.size * size * math.ceil(per_group / in_flight)
        hooks = ba.arguments['hooks']
        rereads = {'rows': tiles, 'columns': bands}
        moved = (bands * rows * depth * size + matrices + rows * width * _output_bytes(ba)
                 + _side_bytes(ba.arguments['prologue_operands'], hooks.prologue_axes, rereads)
                 + _side_bytes(ba.arguments['epilogue_operands'], hooks.epilogue_axes, rereads))
        return self._cost(ba, config, target, tiles * columns, depth / config.block_k, moved)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             config: mosaic_gpu.Tiles, return_residuals: bool = False):
        del exact_sums  # never set: the launch is not eligible for it
        return mosaic_gpu.gmm(tokens, kernel, sizes, tiles=config, transpose_rhs=trans_rhs,
                              out_dtype=out_dtype, multiprocessors=self._multiprocessors(interpret),
                              interpret=INTERPRETED if interpret else False,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class MosaicGmmTransposed(MosaicGmm):
    """`MosaicGmm` against each matrix's transpose: the input gradient, tuned apart."""


@dataclasses.dataclass(frozen=True)
class MosaicTgmm(_Mosaic):
    """`Tgmm` on Hopper (`mosaic_gpu.tgmm`): `block_m` x `block_n` tiles of each group's gradient, its rows
    `block_k` at a time."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')
    EXACT_SUMS: ClassVar[bool] = True

    def space(self, ba) -> tuple[mosaic_gpu.Tiles, ...]:
        depth, width = ba.arguments['tokens'].shape[1], ba.arguments['cotangent'].shape[1]
        return tuple(tile for tile in MOSAIC_TILES if depth % tile.block_m == 0 and width % tile.block_n == 0)

    def fitting(self, ba, target) -> tuple[mosaic_gpu.Tiles, ...]:
        """The tiles that fit `target` (`Template.fitting`), with `exact_sums` those whose two fp32
        accumulators, the running sum and a step's, stay in the warpgroup's registers."""
        fits = super().fitting(ba, target)
        if not ba.arguments.get('exact_sums'):
            return fits
        return tuple(tile for tile in fits if 2 * tile.block_m * tile.block_n // 128 < cost.THREAD_REGISTERS)

    @staticmethod
    def extents(tile: mosaic_gpu.Tiles) -> dict:
        return {'rows': tile.block_k, 'depth': tile.block_m}

    def cost(self, ba, config: mosaic_gpu.Tiles, target) -> float:
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
        return self._cost(ba, config, target, groups * blocks[0] * blocks[1], steps, moved)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, config: mosaic_gpu.Tiles, return_residuals: bool = False):
        return mosaic_gpu.tgmm(tokens, cotangent, sizes, tiles=config, out_dtype=out_dtype,
                               multiprocessors=self._multiprocessors(interpret),
                               interpret=INTERPRETED if interpret else False, prologue=hooks.before(),
                               prologue_operands=prologue_operands, prologue_axes=hooks.prologue_axes,
                               exact_sums=exact_sums), None


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
    return _facts(target).tensor_rate(_mode(dtype)) is not None


def kernel_refusal(compute: Dtype) -> str | None:
    """Why the kernels do not run a product computed in `compute` on this target, its generation stating
    no tensor rate for it (an unseen GPU's), or None where one of them does."""
    target = generation.gpu_target()
    return None if _rated(target, compute) else (
        f"{target.generation} states no {_mode(compute)} tensor rate, which the kernels' cost model reads")


_GMMS = (MosaicGmm(), Gmm())
_GMMS_TRANSPOSED = (MosaicGmmTransposed(), GmmTransposed())
_TGMMS = (MosaicTgmm(), Tgmm())


def launched_gmm(tokens, kernel, sizes, prologue_operands: tuple = (), epilogue_operands: tuple = (), *,
                 out_dtype, trans_rhs: bool, hooks: Hooks = NO_HOOKS, interpret: bool, exact: bool = False):
    """Each group's rows of `tokens` times its matrix of `kernel`, rounded to `tokens`'s dtype, through
    the launch that runs it (`_launch`), with the template's hooks (`templates.gemm.template.gemm`).
    `exact` (a precision above the default) gives an fp32 output IEEE sums (`ragged_dot_refusal`)."""
    arguments = (tokens, kernel.astype(tokens.dtype), sizes, tuple(prologue_operands),
                 tuple(epilogue_operands))
    exact_sums = _exact_sums(exact=exact, hooks=hooks, out_dtype=out_dtype)
    options = {'out_dtype': jnp.dtype(out_dtype).name, 'trans_rhs': trans_rhs, 'hooks': hooks,
               'interpret': interpret, 'exact_sums': exact_sums}
    return _launch(_GMMS_TRANSPOSED if trans_rhs else _GMMS, arguments, options)(*arguments, **options)


def launched_tgmm(tokens, cotangent, sizes, prologue_operands: tuple = (), *, out_dtype,
                  hooks: Hooks = NO_HOOKS, interpret: bool, exact: bool = False):
    """Each group's rows of `tokens` transposed times its rows of `cotangent`, rounded to `tokens`'s
    dtype, through the launch that runs it, with the template's prologue
    (`templates.gemm.template.gemm_t`)."""
    arguments = (tokens, cotangent.astype(tokens.dtype), sizes, tuple(prologue_operands))
    options = {'out_dtype': jnp.dtype(out_dtype).name, 'hooks': hooks, 'interpret': interpret,
               'exact_sums': _exact_sums(exact=exact, hooks=hooks, out_dtype=out_dtype)}
    return _launch(_TGMMS, arguments, options)(*arguments, **options)


def _exact_sums(*, exact: bool, hooks: Hooks, out_dtype) -> bool:
    """Whether a launch adds each depth step's product apart: asked for IEEE sums, with an fp32 output."""
    return exact and jnp.dtype(jnp.float32) in {jnp.dtype(d) for d in (hooks.out_dtypes or (out_dtype,))}


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
    asked for, and `fallback` everywhere else."""
    branches = {'cuda': functools.partial(kernel, interpret=False)}
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


def _forward(inputs, matrix, sizes, out_dtype, interpret_on_cpu):
    def kernel(inputs, matrix, sizes, *, interpret):
        return launched_gmm(inputs, matrix, sizes, out_dtype=out_dtype, trans_rhs=False, interpret=interpret)

    def fallback(inputs, matrix, sizes):
        summed = xla_ragged_dot(inputs, matrix, sizes, preferred_element_type=_summed(out_dtype))
        return summed.astype(out_dtype)

    branches, default = _on_platform(kernel, fallback, interpret_on_cpu)
    return jax.lax.platform_dependent(inputs, matrix, sizes, default=default, **branches)


def _backward(inputs, matrix, sizes, cotangent, input_dtype, kernel_dtype, interpret_on_cpu, exact):
    def kernel(inputs, matrix, sizes, cotangent, *, interpret):
        return (launched_gmm(cotangent, matrix, sizes, out_dtype=input_dtype, trans_rhs=True,
                             interpret=interpret, exact=exact),
                launched_tgmm(inputs, cotangent, sizes, out_dtype=kernel_dtype, interpret=interpret,
                              exact=exact))

    def fallback(inputs, matrix, sizes, cotangent):
        work = jnp.result_type(_summed(input_dtype), _summed(kernel_dtype))
        d_inputs = xla_ragged_dot(cotangent, jnp.swapaxes(matrix, 1, 2), sizes,
                                  preferred_element_type=work)
        _, pullback = jax.vjp(lambda matrix: xla_ragged_dot(
            inputs, matrix, sizes, preferred_element_type=work), matrix.astype(work))
        return d_inputs.astype(input_dtype), pullback(cotangent.astype(work))[0].astype(kernel_dtype)

    branches, default = _on_platform(kernel, fallback, interpret_on_cpu)
    return jax.lax.platform_dependent(inputs, matrix, sizes, cotangent, default=default,
                                      **branches)


def _summed(dtype) -> jnp.dtype:
    """The dtype a product written in `dtype` is summed in: at least fp32."""
    return jnp.promote_types(dtype, jnp.float32)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def grouped_projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                       compute: Dtype, interpret_on_cpu: bool, exact: bool = False) -> jax.Array:
    """Compute `dew.nn.moe.expert_projection` with the kernels, differentiable in first-order reverse mode.

    The operands are cast to `compute`, and the product accumulates in fp32
    and is rounded once to `compute`. The input gradient takes `x`'s dtype
    and the kernel gradient takes the kernel's. Each is summed in fp32 from
    the compute-dtype cotangent and the rounded operands, which with 16-bit
    compute means exact products, and the kernels write each in its own
    dtype, one rounding with no pass after. Only a CUDA lowering runs the
    kernels (`interpret_on_cpu` adds the CPU's interpreter). Every other
    lowering runs `jax.lax.ragged_dot` under the same contract. `exact` (a
    precision above the default) gives an fp32 gradient IEEE sums
    (`ragged_dot_refusal`).
    """
    return _projection_fwd(x, kernel, group_sizes, compute, interpret_on_cpu, exact)[0]


def _projection_fwd(x, kernel, group_sizes, compute, interpret_on_cpu, exact):
    inputs, matrix = x.astype(compute), kernel.astype(compute)
    sizes = group_sizes.astype(jnp.int32)
    output = _forward(inputs, matrix, sizes, jnp.dtype(compute), interpret_on_cpu)
    # The residuals are the rounded operands, the values the forward
    # multiplied; the dtypes of the originals are what the gradients take.
    residuals = (inputs, matrix, sizes, jnp.zeros((0,), x.dtype), jnp.zeros((0,), kernel.dtype))
    return output, residuals


def _projection_bwd(compute, interpret_on_cpu, exact, residuals, cotangent):
    del compute
    inputs, matrix, sizes, x_like, kernel_like = residuals
    d_inputs, d_matrix = _backward(inputs, matrix, sizes, cotangent.astype(inputs.dtype), x_like.dtype,
                                   kernel_like.dtype, interpret_on_cpu, exact)
    return d_inputs, d_matrix, np.zeros(sizes.shape, jax.dtypes.float0)


grouped_projection.defvjp(_projection_fwd, _projection_bwd)
