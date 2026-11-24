"""Dew: a grouped product's Pallas-Triton launches (`Gmm`, `Tgmm`), `grouped_matmul`'s."""

from __future__ import annotations

import abc
import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp

from . import generation
from .grouped_launch import EXACT_STEP, MOST_STAGES, Tile, _GroupedLaunch, _side_tile_bytes
from .templates.gemm.hooks import NO_HOOKS, Hooks


class _Triton(_GroupedLaunch[Tile]):
    """JAX's Pallas Triton kernels (`ragged_dot`), on every CUDA GPU from sm80, and on a CPU where the
    call interprets them."""
    name: ClassVar[str] = 'triton'
    config_cls: ClassVar[type[Tile]] = Tile
    GENERATES: ClassVar[bool] = True

    def eligible(self, ba, target) -> bool:
        # Every CUDA GPU the template sends the kernels to (`kernels.gemm`), and
        # a CPU, which interprets them, or traces the CUDA branch a GPU's lowering takes; never a TPU (a
        # paired product the selector hands on apart: `_takes_pair`); no output stored in a form of its own
        # (`Hooks.out_forms`, the Mosaic template's staged stores alone).
        return (generation.capability(target.generation) is not None and not ba.arguments['hooks'].out_forms
                and self.derived(ba, target) is not None)

    def derived(self, ba, target) -> Tile | None:
        """The one tile the launch runs on `target`, from facts (the ranked set and its cost model deleted,
        bible §6): eight warps, CUTLASS's and cuBLAS's Ampere block; the widest output tile from 128 x 256,
        halved columns first then rows, whose fp32 accumulator (two with exact sums) its threads hold
        (`legal`); the first contraction step the launch takes (`steps`) at which a block holds three stages,
        two loading ahead of the one consumed, else two; and the most stages that step's block holds
        (`staged_bytes`). A dense call takes the first such tile dividing its depth and width where one
        does: the kernel pads to one that does not. On an A100 a product's tile is 128 x 256 a step of 64
        at four stages: at that step three stages took 0.790 (forward) and 0.793 (input gradient) of two's
        time, a geomean over 6 grouped and 4 dense rows, where the ranked set's 128-deep step held two
        (PINS2-triton); PINS3 times the rule's tiles against the ranked set's."""
        rows, columns = 128, 256
        shapes = [(rows, columns)]
        while rows > 16 or columns > 16:
            rows, columns = (rows, columns // 2) if columns > rows else (rows // 2, columns)
            shapes.append((rows, columns))
        extents = self._dense_extents(ba) if ba.arguments.get('dense') else None
        found = None
        for output in shapes:
            for floor in (3, 2):
                for step in self.steps(ba):
                    tile = next((tile for stages in range(MOST_STAGES, floor - 1, -1)
                                 if self.takes(ba, tile := self.tile(output, step, stages), target)), None)
                    if tile is None:
                        continue
                    if extents is None or (extents[0] % self.contraction(tile) == 0
                                           and extents[1] % output[1] == 0):
                        return tile
                    found = found or tile
        return found

    def legal(self, ba, config: Tile, target) -> bool:
        """`Template.legal`, its accumulator (two with exact sums) in its threads' registers, an exactly
        summed step at most `EXACT_STEP` deep (a step's own MMA chain truncates its adds; exact sums
        promote between steps: H100's 128-row step left 1.55x IEEE's error, a 64-row one 1.03x,
        GMMGATE-h100-5), and on Hopper each warpgroup's share of the rows a multiple of wgmma's 64 (a
        32-row share falls back to Ampere's MMA, and an accumulator of 256 registers a thread spills: 1.2 to
        8.5 times slower on an H100, GMM-h100-2)."""
        rows, columns = self.output(config)
        accumulators = 2 if ba.arguments.get('exact_sums') else 1
        held = accumulators * rows * columns // (32 * config.warps) < target.multiprocessor.thread_registers
        short = not ba.arguments.get('exact_sums') or self.contraction(config) <= EXACT_STEP
        shared = not (target.wgmma and rows // max(config.warps // 4, 1) % 64)
        return held and short and shared and super().legal(ba, config, target)

    @staticmethod
    @abc.abstractmethod
    def steps(ba) -> tuple[int, ...]:
        """The contraction steps the launch tries, in order."""

    @staticmethod
    @abc.abstractmethod
    def tile(output: tuple[int, int], step: int, stages: int) -> Tile:
        """The tile of an `output` (rows, columns) tile, a contraction `step` deep, at `stages`, eight
        warps."""

    @staticmethod
    @abc.abstractmethod
    def _dense_extents(ba) -> tuple[int, int]:
        """The depth and width a dense call's tiles divide."""

    @staticmethod
    @abc.abstractmethod
    def output(tile: Tile) -> tuple[int, int]:
        """The rows and columns of a program's output tile, the dot's M and N."""

    @staticmethod
    @abc.abstractmethod
    def contraction(tile: Tile) -> int:
        """The contraction one step of `tile` takes, the dot's K."""

    def staged_bytes(self, ba, config: Tile, target) -> int:
        # Pallas Triton's pipeline holds `stages - 1` depth steps where its MMA reads its operands from
        # registers, and `stages` where it reads them from shared memory as it runs (the device's family's
        # issuer, `mosaic.Issuer.operands_in_registers`: Hopper's wgmma, Blackwell's tcgen05), one step read
        # while the next loads; an A100 and an H100 refuse exactly those bytes, and a B200 refused 262,160
        # bytes for a dense fp32 tile counted at `stages - 1` (gate-2738-gemm-b200, row 4449). The
        # prologue's side tiles load in the same steps.
        # A GPU outside every template family (a Turing part) issues no MMA reading shared memory: its
        # operands are registers' too.
        from .templates import bundles

        dtype, issuer = ba.arguments['tokens'].dtype, generation.family(target.generation)
        held = int(issuer is None or bundles.named(issuer, jnp.bfloat16, None).operands_in_registers)
        step = self.loaded(config) * jnp.dtype(dtype).itemsize + _side_tile_bytes(
            ba.arguments['prologue_operands'], ba.arguments['hooks'].prologue_axes, self.extents(config))
        # A contraction of one step has no loop whose dead stages the output's layout conversion reuses:
        # it takes a 16-bit output tile beside them (an L4 refused 114688 bytes for a 64 x 128 tile's
        # 98304 at a depth of one step, either output dtype, GEN6-gpu-l4; at four steps it asked the
        # stages' alone, TSMEM-l4).
        contracted = self._contracted(ba)
        single = contracted is not None and contracted <= self.contraction(config)
        rows, columns = self.output(config)
        # A hooked call converts its hooks' tiles' layouts through shared memory beside the stages: an A100
        # refused 181504 bytes for a relu producer's tile counted at 147456 (ALIGNED-gpu-a100), within a
        # step's more.
        hooked = ba.arguments['hooks'] != NO_HOOKS
        return step * (config.stages - held + hooked) + (2 * rows * columns if single else 0)

    @staticmethod
    def _contracted(ba) -> int | None:
        """The contraction a call's every tile steps over, None where its groups' rows set it."""
        return None

    @staticmethod
    @abc.abstractmethod
    def extents(tile: Tile) -> dict:
        """The block each launch axis takes at `tile`."""


    @staticmethod
    @abc.abstractmethod
    def loaded(tile: Tile) -> int:
        """The elements of both operand tiles one depth step loads."""


def _product_sides(ba) -> tuple[int, int]:
    """A product's output: its tokens by its kernel's width."""
    kernel = ba.arguments['kernel']
    return ba.arguments['tokens'].shape[0], kernel.shape[1 if ba.arguments['trans_rhs'] else 2]


def _gradient_sides(ba) -> tuple[int, int]:
    """A kernel gradient's output: its tokens' width (its depth) by its cotangent's, the tokens the
    contraction."""
    return ba.arguments['tokens'].shape[1], ba.arguments['cotangent'].shape[1]


@dataclasses.dataclass(frozen=True)
class Gmm(_Triton):
    """Each group's rows of `tokens` `[rows, in]` times its matrix of
    `kernel` `[groups, in, out]`, or of its transpose with `trans_rhs`."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens',)

    def output_sides(self, ba) -> tuple[int, int]:
        return _product_sides(ba)

    @staticmethod
    def _dense_extents(ba) -> tuple[int, int]:
        """The depth and width of a dense call's product, which `block_k` and `block_n` tile."""
        kernel = ba.arguments['kernel']
        return (kernel.shape[2], kernel.shape[1]) if ba.arguments['trans_rhs'] else kernel.shape[1:]

    @staticmethod
    def _contracted(ba) -> int | None:
        return Gmm._dense_extents(ba)[0]

    @staticmethod
    def steps(ba) -> tuple[int, ...]:
        # The deepest first: a dense depth, read a step at a time, synchronizes its pipeline fewer times a
        # tile the deeper the step (at 128 x 256 on an H100, a step of 64 took 1.56 ms where 32 took 1.78,
        # GMM-h100-2).
        del ba
        return 128, 64, 32, 16

    @staticmethod
    def tile(output: tuple[int, int], step: int, stages: int) -> Tile:
        return Tile(output[0], output[1], step, 8, stages)

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


    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             first: jax.Array | None = None, gather: jax.Array | None = None,
             scatter: jax.Array | None = None, joined: jax.Array | None = None, beside: tuple = (),
             chained: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool,
             hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, config: Tile, return_residuals: bool = False):
        # A second matrix, joined rows, tables, matrices side by side, chained products: Mosaic GPU's
        # (`tables`).
        del paired, gather, scatter, joined, beside, chained
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
                              out_dtype=jnp.dtype(out_dtype), first=first,
                              prologue_operands=prologue_operands, epilogue_operands=epilogue_operands,
                              exact_sums=exact_sums, **hooks.kernel_options()), None


@dataclasses.dataclass(frozen=True)
class Tgmm(_Triton):
    """Each group's `tokens` rows transposed times its `cotangent` rows: the
    kernel gradient `[groups, in, out]`."""

    ROWS: ClassVar[tuple[str, ...]] = ('tokens', 'cotangent')

    def output_sides(self, ba) -> tuple[int, int]:
        return _gradient_sides(ba)

    @staticmethod
    def _dense_extents(ba) -> tuple[int, int]:
        """The depth and width of a dense call's gradient, which `block_k` and `block_n` tile."""
        return ba.arguments['tokens'].shape[1], ba.arguments['cotangent'].shape[1]

    @staticmethod
    def loaded(tile: Tile) -> int:
        return tile.block_m * (tile.block_k + tile.block_n)

    @staticmethod
    def steps(ba) -> tuple[int, ...]:
        # Its steps run over a group's rows, each group's padded to whole steps: a step of 64 rows keeps
        # that padding to half of 64 a group.
        del ba
        return 64, 32, 16

    @staticmethod
    def tile(output: tuple[int, int], step: int, stages: int) -> Tile:
        return Tile(step, output[1], output[0], 8, stages)

    @staticmethod
    def output(tile: Tile) -> tuple[int, int]:
        return tile.block_k, tile.block_n

    @staticmethod
    def contraction(tile: Tile) -> int:
        return tile.block_m

    @staticmethod
    def extents(tile: Tile) -> dict:
        return {'rows': tile.block_m, 'depth': tile.block_k}


    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), gather: tuple | None = None, *, out_dtype: str, interpret: bool,
             hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, config: Tile, return_residuals: bool = False):
        del gather  # table rows are Mosaic GPU's alone (`tables`)
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
