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
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype, PrecisionLike
from jax.experimental.pallas import mosaic_gpu as plgpu

from ..precision import asks_default_precision
from . import generation
from .templates import base, cost
from .templates.gemm import mosaic_gpu


def ragged_dot_refusal(compute: Dtype, operands: tuple[Dtype, ...],
                       precision: PrecisionLike) -> str | None:
    """Why the kernels do not compute the product a caller asked for, or None where they do.

    The kernels multiply in `compute`, accumulate in fp32 and ignore
    `precision`. With 16-bit compute that gives exact products summed in
    fp32, which any precision asks for. With fp32 compute it gives TF32,
    which only the default precision asks for (explicitly or through
    `jax_default_matmul_precision`). An operand or master wider than fp32
    needs its gradient summed wider than the kernels sum, and x64 widens
    their int32 group offsets, so both are refused.
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
                                                            (2, 3, 4)))
"""Every tile the Triton launches may take: the ones the A100 and H100 sweeps found fastest lie inside.
Where the cost model ties, the first wins: the deeper step, which synchronizes its pipeline fewer times
a tile (at 128 x 256 on an H100, a step of 64 took 1.56 ms where 32 took 1.78, GMM-h100-2), and the
more warps."""

MOSAIC_TILES = tuple(mosaic_gpu.Tiles(tile_m, tile_n, 64, stages, split, width)
                     for tile_m, tile_n, stages, split, width in itertools.product(
                         (128, 64), (128, 64), (4, 3, 5), ('n', 'm'), (8, 4)))
"""Every tile the Mosaic GPU launches may take, a contraction step of 64 (`mosaic_gpu.Tiles`); where the
cost model ties, the first wins, the larger tile."""

INTERPRETED = plgpu.InterpretGPUParams(out_of_bounds_reads="uninitialized", uninitialized_memory="zero")
"""How a CPU interprets the Mosaic GPU launches: a read past an operand's rows gives zeros, as the TMA
loads they stand for do."""


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


def _routed(rows: int, groups: int) -> float:
    """The share of `groups` matrices a launch over `rows` rows reads: at most one group a row."""
    return min(groups, rows) / groups


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
    """Group sizes summing to `rows` over the groups `specs` name, drawn
    unevenly, one group left empty, as a step routes them."""
    groups = next(spec.shape[0] for spec in specs if jnp.issubdtype(spec.dtype, jnp.integer))
    shares = np.random.default_rng(seed).dirichlet(np.ones(groups))
    shares[0] = 0.0
    sizes = np.floor(shares / shares.sum() * rows).astype(np.int32)
    sizes[-1] += rows - sizes.sum()
    return jnp.asarray(sizes)


class _GroupedLaunch(base.Template):
    """A grouped launch: its space is a declared tuple, its group sizes drawn as a step routes them."""
    # Annotations stay evaluated (no postponed annotations in this module):
    # tokamax serializes a launch's arguments into the program by their types.

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        arrays = super().operands(specs, seed)
        sizes = _sizes(specs, seed, specs[0].shape[0])
        return [sizes if jnp.issubdtype(spec.dtype, jnp.integer) else array
                for spec, array in zip(specs, arrays, strict=True)]


class _Triton(_GroupedLaunch):
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
        """The tiles that fit `target` (`Template.fitting`) whose accumulator stays in its threads'
        registers, and on Hopper those whose dot Triton gives a warpgroup MMA: each warpgroup's share of
        the output rows a multiple of wgmma's 64. Either other kind ran 1.2 to 8.5 times slower than the
        fastest tile on an H100 (GMM-h100-2: an accumulator of 256 registers a thread spills; a 32-row
        share falls back to Ampere's MMA)."""
        wgmma = _facts(target).wgmma

        def fits(tile: Tile) -> bool:
            rows, columns = self.output(tile)
            held = rows * columns // (32 * tile.warps) < cost.THREAD_REGISTERS
            return held and not (wgmma and rows // max(tile.warps // 4, 1) % 64)

        return tuple(tile for tile in super().fitting(ba, target) if fits(tile))

    @staticmethod
    @abc.abstractmethod
    def output(tile: Tile) -> tuple[int, int]:
        """The rows and columns of a program's output tile, the dot's M and N."""

    def staged_bytes(self, ba, config: Tile) -> int:
        # Pallas Triton's pipeline holds `stages - 1` depth steps, and on Hopper `stages` where it
        # pipelines a wgmma, which reads one step while the next loads; an A100 and an H100 refuse
        # exactly those bytes, so a Hopper tile is sized for the larger.
        held = 0 if generation.gpu_target().wgmma else 1
        return self.loaded(config) * jnp.dtype(ba.arguments['tokens'].dtype).itemsize * (config.stages - held)

    @staticmethod
    @abc.abstractmethod
    def loaded(tile: Tile) -> int:
        """The elements of both operand tiles one depth step loads."""

    def _resident(self, ba, config: Tile, target) -> int:
        accumulator = config.block_m * config.block_n // (32 * config.warps)
        return cost.resident(target, self.staged_bytes(ba, config), 32 * config.warps, accumulator + 64)


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
        size, out = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(ba.arguments['out_dtype']).itemsize
        chunks = _row_tiles(rows, groups, CHUNK_M)
        columns = math.ceil(width / config.block_n)
        tiles = _row_tiles(rows, groups, config.block_m)
        flops = 2 * tiles * config.block_m * depth * width
        steps = tiles * math.ceil(depth / config.block_k) / chunks
        staged = steps * _triton_staged(config.block_m, config.block_n, config.block_k, config.warps,
                                        _facts(target).wgmma) * size
        moved = (columns * rows * depth + tiles * depth * width) * size + rows * width * out
        return cost.seconds(cost.Cost(
            programs=chunks * columns, device_bytes=moved,
            program_flops=flops / (chunks * columns), program_staged=staged,
            resident=self._resident(ba, config, target), mode=_mode(tokens.dtype)), target)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array, *, out_dtype: str,
             trans_rhs: bool, interpret: bool, config: Tile, return_residuals: bool = False):
        # Imported at first use: the kernels need jax's Pallas Triton backend,
        # which jax 0.11.2 deprecates, and importing a model must not load it.
        from . import ragged_dot
        return ragged_dot.gmm(tokens, kernel, sizes, block_m=config.block_m, block_k=config.block_k,
                              block_n=config.block_n, num_warps=config.warps, num_stages=config.stages,
                              trans_rhs=trans_rhs, interpret=interpret, compute_dtype=tokens.dtype,
                              out_dtype=jnp.dtype(out_dtype)), None


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
        moved = (blocks[1] * rows * depth + blocks[0] * rows * width) * size + groups * depth * width * out
        return cost.seconds(cost.Cost(
            programs=programs, device_bytes=moved, program_flops=flops, program_staged=staged,
            resident=self._resident(ba, config, target), mode=_mode(tokens.dtype)), target)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array, *, out_dtype: str,
             interpret: bool, config: Tile, return_residuals: bool = False):
        from . import ragged_dot
        return ragged_dot.tgmm(tokens, cotangent, sizes, block_m=config.block_m, block_k=config.block_k,
                               block_n=config.block_n, num_warps=config.warps, num_stages=config.stages,
                               interpret=interpret, compute_dtype=tokens.dtype,
                               out_dtype=jnp.dtype(out_dtype)), None


class _Mosaic(_GroupedLaunch):
    """The Pallas Mosaic GPU kernels (`templates.gemm.mosaic_gpu`): Hopper's warpgroup MMA, one 16-bit
    dtype, and widths the space's tiles divide."""
    name: ClassVar[str] = 'mosaic_gpu'
    config_cls: ClassVar[type[mosaic_gpu.Tiles]] = mosaic_gpu.Tiles

    def eligible(self, ba, target) -> bool:
        first, second = (value.dtype for name, value in ba.arguments.items()
                         if name in ('tokens', 'kernel', 'cotangent'))
        return (target.wgmma and first == second and jnp.dtype(first).itemsize == 2
                and _rated(target, first) and bool(self.space(ba)))

    def staged_bytes(self, ba, config: mosaic_gpu.Tiles) -> int:
        size = jnp.dtype(ba.arguments['tokens'].dtype).itemsize
        return config.stages * self.loaded(config) * size + self.epilogue(config) * jnp.dtype(
            ba.arguments['out_dtype']).itemsize

    @staticmethod
    @abc.abstractmethod
    def loaded(tile: mosaic_gpu.Tiles) -> int:
        """The elements of both operand tiles one contraction step loads."""

    @staticmethod
    @abc.abstractmethod
    def epilogue(tile: mosaic_gpu.Tiles) -> int:
        """The elements of the epilogue's shared-memory slots."""

    def _multiprocessors(self, interpret: bool) -> int:
        # The interpreter runs one block (jax/_src/pallas/mosaic_gpu/interpret).
        return 1 if interpret else generation.gpu_target().multiprocessors


@dataclasses.dataclass(frozen=True)
class MosaicGmm(_Mosaic):
    """`Gmm` on Hopper (`mosaic_gpu.gmm`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    def space(self, ba) -> tuple[mosaic_gpu.Tiles, ...]:
        kernel = ba.arguments['kernel']
        depth, width = (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]
        return tuple(tile for tile in MOSAIC_TILES if depth % tile.tile_k == 0 and width % tile.cta_n == 0)

    @staticmethod
    def loaded(tile: mosaic_gpu.Tiles) -> int:
        return (tile.cta_m + tile.cta_n) * tile.tile_k

    @staticmethod
    def epilogue(tile: mosaic_gpu.Tiles) -> int:
        return 2 * 2 * min(mosaic_gpu.EPILOGUE, tile.tile_m) * min(mosaic_gpu.EPILOGUE, tile.tile_n)

    def cost(self, ba, config: mosaic_gpu.Tiles, target) -> float:
        # Persistent: each multiprocessor takes output tiles in turn.
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        (rows, depth), groups = tokens.shape, kernel.shape[0]
        width = kernel.shape[1] if ba.arguments['trans_rhs'] else kernel.shape[2]
        size, out = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(ba.arguments['out_dtype']).itemsize
        tiles, columns = _row_tiles(rows, groups, config.cta_m), width // config.cta_n
        steps = depth // config.tile_k
        # Each step's tiles are written by TMA and read by both warpgroups' wgmma, each its own rows of
        # the left operand (or all of them) and its own columns of the right (or all of them).
        read = 2 * (config.tile_m + config.tile_n) * config.tile_k
        staged = steps * (self.loaded(config) + read) * size + 2 * config.cta_m * config.cta_n * out
        return cost.seconds(cost.Cost(
            programs=tiles * columns,
            device_bytes=(rows * depth + _routed(rows, groups) * kernel.size) * size + rows * width * out,
            program_flops=2 * config.cta_m * config.cta_n * depth, program_staged=staged,
            mode=_mode(tokens.dtype)), target)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array, *, out_dtype: str,
             trans_rhs: bool, interpret: bool, config: mosaic_gpu.Tiles, return_residuals: bool = False):
        return mosaic_gpu.gmm(tokens, kernel, sizes, tiles=config, transpose_rhs=trans_rhs,
                              out_dtype=out_dtype, multiprocessors=self._multiprocessors(interpret),
                              interpret=INTERPRETED if interpret else False), None


@dataclasses.dataclass(frozen=True)
class MosaicGmmTransposed(MosaicGmm):
    """`MosaicGmm` against each matrix's transpose: the input gradient, tuned apart."""


@dataclasses.dataclass(frozen=True)
class MosaicTgmm(_Mosaic):
    """`Tgmm` on Hopper (`mosaic_gpu.tgmm`): its warpgroups split the output's columns."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    def space(self, ba) -> tuple[mosaic_gpu.Tiles, ...]:
        depth, width = ba.arguments['tokens'].shape[1], ba.arguments['cotangent'].shape[1]
        return tuple(tile for tile in MOSAIC_TILES
                     if tile.split == 'n' and depth % tile.tile_m == 0 and width % tile.cta_n == 0)

    @staticmethod
    def loaded(tile: mosaic_gpu.Tiles) -> int:
        return tile.tile_k * (tile.tile_m + tile.cta_n)

    @staticmethod
    def epilogue(tile: mosaic_gpu.Tiles) -> int:
        return 2 * tile.tile_m * tile.tile_n

    def cost(self, ba, config: mosaic_gpu.Tiles, target) -> float:
        # Persistent over each group's output tiles.
        tokens, cotangent, sizes = ba.arguments['tokens'], ba.arguments['cotangent'], ba.arguments['sizes']
        (rows, depth), width, groups = tokens.shape, cotangent.shape[1], sizes.shape[0]
        size, out = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(ba.arguments['out_dtype']).itemsize
        tiles = groups * (depth // config.tile_m) * (width // config.cta_n)
        steps = _row_tiles(rows, groups, config.tile_k) / groups
        read = 2 * (config.tile_m + config.tile_n) * config.tile_k
        staged = steps * (self.loaded(config) + read) * size + config.tile_m * config.cta_n * out
        moved = rows * (depth + width) * size + groups * depth * width * out
        return cost.seconds(cost.Cost(
            programs=tiles, device_bytes=moved,
            program_flops=2 * steps * config.tile_k * config.tile_m * config.cta_n, program_staged=staged,
            mode=_mode(tokens.dtype)), target)

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array, *, out_dtype: str,
             interpret: bool, config: mosaic_gpu.Tiles, return_residuals: bool = False):
        return mosaic_gpu.tgmm(tokens, cotangent, sizes, tiles=config, out_dtype=out_dtype,
                               multiprocessors=self._multiprocessors(interpret),
                               interpret=INTERPRETED if interpret else False), None


def _facts(target: generation.GpuTarget) -> generation.GpuTarget:
    """The target a launch on `target` is costed as (`cost._facts`): a CPU as an A100."""
    return target if generation.capability(target.generation) is not None else generation.TARGETS['sm80']


def _rated(target: generation.GpuTarget, dtype) -> bool:
    """Whether `target` states the tensor rate a launch over `dtype` operands is costed by: a generation
    that states none runs XLA's grouped matmul, needing no estimate."""
    return _facts(target).tensor_rate(_mode(dtype)) is not None


_GMMS = (MosaicGmm(), Gmm())
_GMMS_TRANSPOSED = (MosaicGmmTransposed(), GmmTransposed())
_TGMMS = (MosaicTgmm(), Tgmm())


def _gmm(tokens, kernel, sizes, out_dtype, *, trans_rhs: bool, interpret: bool):
    arguments = (tokens, kernel.astype(tokens.dtype), sizes)
    options = {'out_dtype': jnp.dtype(out_dtype).name, 'trans_rhs': trans_rhs, 'interpret': interpret}
    launch = base.choose('grouped_matmul', _GMMS_TRANSPOSED if trans_rhs else _GMMS, *arguments, **options)
    return launch(*arguments, **options)


def _tgmm(tokens, cotangent, sizes, out_dtype, *, interpret: bool):
    arguments = (tokens, cotangent.astype(tokens.dtype), sizes)
    options = {'out_dtype': jnp.dtype(out_dtype).name, 'interpret': interpret}
    launch = base.choose('grouped_matmul', _TGMMS, *arguments, **options)
    return launch(*arguments, **options)


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
        return _gmm(inputs, matrix, sizes, out_dtype, trans_rhs=False, interpret=interpret)

    def fallback(inputs, matrix, sizes):
        summed = xla_ragged_dot(inputs, matrix, sizes, preferred_element_type=_summed(out_dtype))
        return summed.astype(out_dtype)

    branches, default = _on_platform(kernel, fallback, interpret_on_cpu)
    return jax.lax.platform_dependent(inputs, matrix, sizes, default=default, **branches)


def _backward(inputs, matrix, sizes, cotangent, input_dtype, kernel_dtype, interpret_on_cpu):
    def kernel(inputs, matrix, sizes, cotangent, *, interpret):
        return (_gmm(cotangent, matrix, sizes, input_dtype, trans_rhs=True, interpret=interpret),
                _tgmm(inputs, cotangent, sizes, kernel_dtype, interpret=interpret))

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


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def grouped_projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                       compute: Dtype, interpret_on_cpu: bool) -> jax.Array:
    """Compute `dew.nn.moe.expert_projection` with the kernels, differentiable in first-order reverse mode.

    The operands are cast to `compute`, and the product accumulates in fp32
    and is rounded once to `compute`. The input gradient takes `x`'s dtype
    and the kernel gradient takes the kernel's. Each is summed in fp32 from
    the compute-dtype cotangent and the rounded operands, which with 16-bit
    compute means exact products, and the kernels write each in its own
    dtype, one rounding with no pass after. Only a CUDA lowering runs the
    kernels (`interpret_on_cpu` adds the CPU's interpreter). Every other
    lowering runs `jax.lax.ragged_dot` under the same contract.
    """
    return _projection_fwd(x, kernel, group_sizes, compute, interpret_on_cpu)[0]


def _projection_fwd(x, kernel, group_sizes, compute, interpret_on_cpu):
    inputs, matrix = x.astype(compute), kernel.astype(compute)
    sizes = group_sizes.astype(jnp.int32)
    output = _forward(inputs, matrix, sizes, jnp.dtype(compute), interpret_on_cpu)
    # The residuals are the rounded operands, the values the forward
    # multiplied; the dtypes of the originals are what the gradients take.
    residuals = (inputs, matrix, sizes, jnp.zeros((0,), x.dtype), jnp.zeros((0,), kernel.dtype))
    return output, residuals


def _projection_bwd(compute, interpret_on_cpu, residuals, cotangent):
    del compute
    inputs, matrix, sizes, x_like, kernel_like = residuals
    d_inputs, d_matrix = _backward(inputs, matrix, sizes, cotangent.astype(inputs.dtype), x_like.dtype,
                                   kernel_like.dtype, interpret_on_cpu)
    return d_inputs, d_matrix, np.zeros(sizes.shape, jax.dtypes.float0)


grouped_projection.defvjp(_projection_fwd, _projection_bwd)
