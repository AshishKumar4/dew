"""The template's grouped products of quantized formats (`grouped_matmul.MosaicGmm`'s and `MosaicTgmm`'s
specializations): an NVFP4 matrix's packed words and a word-packed one's decoded in the walk
(`MosaicPackedGmm`, `MosaicWordPackedGmm`), and a product whose rows' and matrix's scales its walk applies a
step at a time
(`MosaicStepScaledGmm`, `MosaicStepScaledTgmm`: DeepSeek-V3's blockwise FP8) or a block at a time
(`MosaicMxGmm`: MXFP8)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Protocol

import jax
import jax.numpy as jnp

from . import generation
from .formats import Bf16Window, TernaryWeights
from .grouped_launch import INTERPRETED
from .grouped_matmul import MosaicGmm, MosaicTgmm, _beside, _stored_dtypes
from .templates.gemm import packed, tiling
from .templates.gemm.hooks import NO_HOOKS, Hooks
from .templates.product import NVFP4, SCALE_DEPTH

if TYPE_CHECKING:
    from .templates import mosaic
    from .templates.gemm import mosaic_gpu


class _PackedMatrix(Protocol):
    """A packed matrix's fields as the packed launch streams them (`formats.NvFp4`'s)."""

    words: jax.Array
    scales: jax.Array


@dataclasses.dataclass(frozen=True)
class MosaicPackedGmm(MosaicGmm):
    """`MosaicGmm` of an NVFP4 matrix (`gemm.nvfp4_product`'s packed kernel, row 3621): bfloat16 rows times
    its `kernel`, the bank as Dew stores it (`formats.NvFp4`: its words and block scales, K-major, its shape
    and dtype the bfloat16 matrix it decodes to, `[groups, depth, width]`), streamed as its words and block
    scales and each stage decoded in place into the consumer's buffers (`mosaic_gpu.gmm`'s `scales`), so its
    tile takes `packed.NVFP4_STEP`-deep steps, whole 64-column tiles, the decode's shared memory and
    registers, and no cluster, split, prologue or pair; its epilogue the finished accumulator's, as any
    launch's (its tensor scale's, row 13743). A decode-sized dense call runs swapped (`_swapped`, row 5387):
    the matrix the products' rows, each stage of it decoded in registers into their left operand, the call's
    rows the columns."""

    STEPS: ClassVar[tuple[int, ...] | None] = (packed.NVFP4_STEP,)

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array,
             prologue_operands: tuple = (), epilogue_operands: tuple = (), paired: jax.Array | None = None,
             first: jax.Array | None = None, gather: jax.Array | None = None,
             scatter: jax.Array | None = None, joined: jax.Array | None = None, beside: tuple = (),
             chained: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool,
             hooks: Hooks = NO_HOOKS, exact_sums: bool = False, dense: bool = False,
             config: tiling.Tiles, return_residuals: bool = False, packed_rows: bool = False):
        # `packed_rows` states the swapped call's orientation (`_swapped`), which its derivation reads: the
        # call itself runs as `config` says (`_launched_call`).
        del packed_rows, chained
        return super()._fwd(tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                            gather, scatter, joined, out_dtype=out_dtype, trans_rhs=trans_rhs,
                            interpret=interpret,
                            hooks=hooks, exact_sums=exact_sums, dense=dense, config=config,
                            return_residuals=return_residuals)

    def _streamed_matrix(self, kernel: _PackedMatrix, trans_rhs: bool) -> tuple:
        # Its words, K-major whatever orientation its decoded shape states, and its block scales, which each
        # stage decodes (`mosaic_gpu.gmm`'s `scales`).
        del trans_rhs
        return kernel.words, True, {'scales': kernel.scales}

    def _matrix_transposed(self, ba) -> bool:
        # Its words K-major, or a swapped call's rows read transposed (`_swapped`): its matrix streams
        # transposed either way, whatever the binding holds (an abstract matrix where a derivation binds it).
        del ba
        return True

    def _allocations(self, ba, config: tiling.Tiles, family: mosaic.Family) -> mosaic_gpu.Allocations:
        from .templates.gemm import mosaic_gpu

        dtype = ba.arguments['tokens'].dtype
        rows = ba.arguments['packed_rows']
        # An epilogue's side operands staged as any launch's (the tensor scale its finish reads, row 13743).
        return mosaic_gpu.gmm_allocations(
            config, family, dtype=dtype, rhs_dtype=dtype if rows else jnp.uint32, transpose_rhs=True,
            prologue_axes=(), prologue_dtypes=(), epilogue_axes=ba.arguments['hooks'].epilogue_axes,
            epilogue_dtypes=tuple(side.dtype for side in ba.arguments.get('epilogue_operands', ())),
            dtypes=(ba.arguments['out_dtype'],), packed=True, packed_rows=rows)

    def _legal(self, ba, config: tiling.Tiles, target) -> bool:
        return config.splits == 1 and super()._legal(ba, config, target)
    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
        # A swapped call's consumers each decode 64 rows of the matrix, its fragment's; the matrix's columns
        # take whole 64-column fragments otherwise (`packed.rows_taken`, `packed.taken`).
        whole = (config.block_m // config.warpgroups == 64 if ba.arguments['packed_rows']
                 else config.block_n % 64 == 0)
        return (config.cluster == 1 and config.splits == 1 and not config.swapped and whole
                and super().candidate(ba, config, target))

    SWAPPED_OPTIONS: ClassVar[dict] = {'packed_rows': True}

    def _operand_bytes(self, ba, tile: tiling.Tiles) -> tuple[tuple[float, float], tuple]:
        # Its matrix moves its words and block scales (`formats.NvFp4`): decoded into the stage the products
        # read (`gmm`'s `scales`, its decoded dtype's: one its consumers share or each one's own,
        # `packed.decoding`), or swapped, in registers into them, each consumer its rows (`packed_rows`).
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        # A value's: its code's half byte and its sixteen depths' one E4M3 scale (the format's geometry,
        # whatever the bank's size, an empty one's too).
        stored = 0.5 + jnp.dtype(kernel.scales.dtype).itemsize / NVFP4.depth
        rows, stage = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(kernel.dtype).itemsize
        if tile.swapped:
            return (stored, rows), ((0, 1), None)
        copies = 1 if packed.shared_decode(tile, interpret=False) else tile.warpgroups
        return (rows, stored), (None, (stage, copies))

    def _swaps_on(self, family: mosaic.Family, dtype) -> bool:
        # Swapped, its matrix's stages decode in registers into the products' left operand
        # (`mosaic_gpu.gmm`'s `packed_rows`, row 5387), so no decoded stage passes through shared memory:
        # on a family whose products take a left operand from registers in the decode's layout
        # (`packed.rows_taken`: the warpgroup MMA's fragment).
        from .templates.gemm import packed
        del dtype
        return packed.rows_supported(family)

    def _launched_call(self, tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                       gather, scatter, joined=None, *, out_dtype: str, trans_rhs: bool, interpret: bool,
                       hooks: Hooks, exact_sums: bool, dense: bool, config: tiling.Tiles, formats: dict,
                       target):
        if not config.swapped:
            return super()._launched_call(tokens, kernel, sizes, prologue_operands, epilogue_operands,
                                          paired, first, gather, scatter, joined, out_dtype=out_dtype,
                                          trans_rhs=trans_rhs, interpret=interpret, hooks=hooks,
                                          exact_sums=exact_sums, dense=dense, config=config, formats=formats,
                                          target=target)
        # The matrix's words (and block scales when NVFP4) as rows, decoded in registers (`_swaps_on`).
        return self._swapped_launch(kernel.words[0], tokens, sizes, out_dtype=out_dtype, interpret=interpret,
                                    exact_sums=exact_sums, config=config, target=target,
                                    packed_rows=True, **self._swapped_extra(kernel, formats))

    def _swapped_extra(self, kernel, formats: dict) -> dict:
        return {'scales': kernel.scales[0], **formats}
    def _registers(self, ba, config: tiling.Tiles, *, merging: bool, target) -> int | None:
        # Beside the plain product's live set, a step's decode holds a 128-depth group's words, the code
        # being decoded and the stage's block scales, each a `[columns, 16]` register tile over a role's
        # threads (`packed.lanes`, its columns the tile's, or a consumer's rows where the call runs swapped),
        # and a word's decode in flight, its form's live registers (`bits.BitForm.live`); swapped, the
        # decoded stage too, the products' left operand (`packed.rows_decoded`, bfloat16 a half register).
        from ...algebra import bits
        from .templates import resources

        if ba.arguments['packed_rows']:
            share = config.block_m // config.warpgroups
            decoded = 3 * share * 16 + share * config.block_k // 2
        else:
            decoded = 3 * config.block_n * 16
        return _beside(super()._registers(ba, config, merging=merging, target=target),
                       decoded // resources.THREADS + bits.NVFP4_BFLOAT16_FORM.live)


@dataclasses.dataclass(frozen=True)
class MosaicWordPackedGmm(MosaicPackedGmm):
    """`MosaicPackedGmm` of inline-decoded scale-free word-packed weights (`TernaryWeights`, `Bf16Window`;
    rows 1464, 5387), driven by `packed.PackedWeightFormat`: a format whose decode reads its columns' meta
    halves (a window's bases and escapes) streams them as `gmm`'s `scales`, and runs no swapped call."""
    DEFAULT_FMT: ClassVar[str] = 'ternary'

    @property
    def _scaled(self) -> bool:
        return True

    def _step_sizes(self, ba) -> tuple[int, ...] | None:
        return self._desc(ba).steps

    def _desc(self, ba) -> packed.PackedWeightFormat:
        kernel = ba.arguments['kernel']
        return packed.weight_format(
            kernel.fmt if isinstance(kernel, (TernaryWeights, Bf16Window)) else self.DEFAULT_FMT)

    def _streamed_matrix(self, kernel, trans_rhs: bool) -> tuple:
        del trans_rhs
        assert isinstance(kernel, (TernaryWeights, Bf16Window)), type(kernel)
        return kernel.words, True, self._formats(kernel)

    def _formats(self, kernel: TernaryWeights | Bf16Window) -> dict:
        """`gmm`'s keywords for its packed matrix: its format's name, and a window's meta halves."""
        meta = {'scales': kernel.meta_halves} if isinstance(kernel, Bf16Window) else {}
        return {'packed_format': kernel.fmt, **meta}

    def _swaps_on(self, family: mosaic.Family, dtype) -> bool:
        # A format reading its columns' meta halves decodes a staged matrix alone (`packed.word_taken`).
        return packed.weight_format(self.DEFAULT_FMT).meta is False and super()._swaps_on(family, dtype)

    def _swapped_extra(self, kernel, formats: dict) -> dict:
        assert isinstance(kernel, (TernaryWeights, Bf16Window)), type(kernel)
        return {**self._formats(kernel), **formats}
    def _allocations(self, ba, config: tiling.Tiles, family: mosaic.Family) -> mosaic_gpu.Allocations:
        from .templates.gemm import mosaic_gpu
        dtype, rows, hooks = ba.arguments['tokens'].dtype, ba.arguments['packed_rows'], ba.arguments['hooks']
        return mosaic_gpu.gmm_allocations(
            config, family, dtype=dtype, rhs_dtype=dtype if rows else jnp.uint32, transpose_rhs=True,
            prologue_axes=(), prologue_dtypes=(), epilogue_axes=hooks.epilogue_axes,
            epilogue_dtypes=tuple(side.dtype for side in ba.arguments.get('epilogue_operands', ())),
            dtypes=_stored_dtypes(ba),
            packed_format=self._desc(ba).name, packed_rows=rows)
    def _operand_bytes(self, ba, tile: tiling.Tiles) -> tuple[tuple[float, float], tuple]:
        tokens, kernel = ba.arguments['tokens'], ba.arguments['kernel']
        rows_b, stage = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(kernel.dtype).itemsize
        wb = self._desc(ba).weight_bytes
        if tile.swapped:
            return (wb, rows_b), ((0, 1), None)
        copies = 1 if packed.shared_decode(tile, interpret=False) else tile.warpgroups
        return (rows_b, wb), (None, (stage, copies))

    def _registers(self, ba, config: tiling.Tiles, *, merging: bool, target) -> int | None:
        from .templates import resources
        share = config.block_m // config.warpgroups
        desc = self._desc(ba)
        # A group's blocks and the lane being decoded, a `[columns, 16]` tile each, and a step's meta halves
        # beside them (`PackedWeightFormat.meta_columns`).
        decoded = (2 * share * 16 + share * config.block_k // 2 if ba.arguments['packed_rows']
                   else config.block_n * (16 * (desc.blocks + 1) + desc.meta_columns(config.block_k)))
        return _beside(MosaicGmm._registers(self, ba, config, merging=merging, target=target),
                       decoded // resources.THREADS + desc.decode_live)


@dataclasses.dataclass(frozen=True)
class _ScaledGmm(MosaicGmm):
    """`MosaicGmm` of a product whose rows' and matrix's scales its binding takes as one operand (`scales`),
    reaching the mainloop as the keyword of `mosaic_gpu.gmm`'s its subclass names (`SCALES`)."""

    SCALES: ClassVar[str]

    def _swapped(self, ba, target):
        # Its scales are its rows' and columns': no swapped orientation reads them.
        del ba, target

    def _fwd(self, tokens: jax.Array, kernel: jax.Array, sizes: jax.Array, prologue_operands: tuple = (),
             epilogue_operands: tuple = (), paired: jax.Array | None = None, first: jax.Array | None = None,
             gather: jax.Array | None = None, scatter: jax.Array | None = None,
             joined: jax.Array | None = None, beside: tuple = (), chained: tuple = (), *, out_dtype: str,
             trans_rhs: bool, interpret: bool, hooks: Hooks = NO_HOOKS, exact_sums: bool = False,
             dense: bool = False, scales: tuple = (),
             config: tiling.Tiles,
             return_residuals: bool = False):
        del return_residuals, chained
        return self._forward(tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                             gather, scatter, joined, out_dtype=out_dtype, trans_rhs=trans_rhs,
                             interpret=interpret, hooks=hooks, exact_sums=exact_sums, dense=dense,
                             config=config, formats={self.SCALES: tuple(scales)})


@dataclasses.dataclass(frozen=True)
class MosaicStepScaledGmm(_ScaledGmm):
    """`MosaicGmm` of a blockwise-scaled FP8 product (`gemm.blockwise_fp8_product`, row 4518), planned on the
    call it is: each exactly summed `SCALE_DEPTH`-deep step's partial times its rows' and
    columns' scales before its add (`mosaic_gpu.gmm`'s `step_scales`), its rows' scales streamed beside its
    ring (`gmm_allocations`' `step_scaled`), a column tile within one weight block's columns, and no split,
    swap, prologue or pair (an epilogue reads its sums)."""

    STEPS: ClassVar[tuple[int, ...] | None] = (SCALE_DEPTH,)
    STEP_SCALED: ClassVar[bool] = True
    SCALES: ClassVar[str] = 'step_scales'

    def _fp8(self, ba, dtype, family: mosaic.Family) -> bool:
        # An epilogue reads its exact sums as any product's (a next product's rows quantized there, row 8547):
        # no prologue, its rows FP8 codes no hook forms.
        hooks = ba.arguments['hooks']
        return (jnp.dtype(dtype) in family.fp8_formats and self.FP8 and hooks.prologue is None
                and ba.arguments.get('paired') is None and bool(ba.arguments.get('exact_sums')))

    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
        within = SCALE_DEPTH % config.block_n == 0
        return config.splits == 1 and not config.swapped and within and super().candidate(ba, config, target)

    def _registers(self, ba, config: tiling.Tiles, *, merging: bool, target) -> int | None:
        # Beside the exact product's live set, a step's column scales.
        return _beside(super()._registers(ba, config, merging=merging, target=target),
                       _column_scale_registers(ba, config, target))


@dataclasses.dataclass(frozen=True)
class MosaicMxGmm(_ScaledGmm):
    """`MosaicGmm` of a block-scaled product (`gemm.mxfp8_product`, `gemm.nvfp4_block_product`, row 4519),
    planned on the call it is: FP8 or E2M1 rows times a K-major matrix of theirs, each block of each scaled
    by its scale in the products (`product.block_format`)
    (`mosaic_gpu.gmm`'s `mx_scales`, `mosaic.Issuer.scaled_products`), its block scales streamed beside its
    rings (`gmm_allocations`' `block_scaled`) on a family whose products take them (tcgen05's), its tiles
    whole 128s of rows, columns and depth, and no split, swap, prologue, pair or exact sums: an epilogue
    reads its sums (`gemm.nvfp4_block_finished`'s, its outputs in their forms, row 12721)."""

    BLOCK_SCALED: ClassVar[bool] = True
    SCALES: ClassVar[str] = 'mx_scales'

    def _fp8(self, ba, dtype, family: mosaic.Family) -> bool:
        # Its products scale each block in the instruction, which sums in fp32 itself: no step promoted. Its
        # operands FP8 (MXFP8) or E2M1 (NVFP4), `product.block_format`'s. An epilogue reads its sums as any
        # product's; no prologue forms its codes.
        return (jnp.dtype(dtype) in family.block_formats
                and ba.arguments['trans_rhs'] and ba.arguments['hooks'].prologue is None
                and ba.arguments.get('paired') is None and not ba.arguments.get('exact_sums'))

    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
        whole = all(extent % 128 == 0 for extent in (config.block_m, config.block_n, config.block_k))
        return config.splits == 1 and not config.swapped and whole and super().candidate(ba, config, target)


@dataclasses.dataclass(frozen=True)
class MosaicStepScaledTgmm(MosaicTgmm):
    """`MosaicTgmm` of a blockwise FP8 recipe's kernel gradient (`gemm.blockwise_fp8_kernel_gradient`, row
    4518), planned on the call it is: FP8 operands rows-contiguous (`mosaic_gpu.tgmm`'s `k_major`, as
    Hopper's FP8 tensor cores read both), each exactly summed 128-row step's partial times its depth rows' and
    columns' scales, its rows' scales streamed beside its ring, groups of whole steps, no split or hook."""

    STEPS: ClassVar[tuple[int, ...] | None] = (SCALE_DEPTH,)
    FP8: ClassVar[bool] = True
    K_MAJOR: ClassVar[bool] = True

    def _fwd(self, tokens: jax.Array, cotangent: jax.Array, sizes: jax.Array, prologue_operands: tuple = (),
             gather: tuple | None = None, *, out_dtype: str, interpret: bool, hooks: Hooks = NO_HOOKS,
             exact_sums: bool = False, dense: bool = False, step_scales: tuple = (),
             config: tiling.Tiles, return_residuals: bool = False, whole_steps: bool = False):
        # Bound as a kernel gradient states its operands, rows first: the rows-contiguous operands' transposes
        # (`gemm.blockwise_fp8_kernel_gradient`), which XLA folds into the kernel's own operands, read back as
        # they lie (`mosaic_gpu.tgmm`'s `k_major`); its scales (`step_scales`, the binding's operand) its
        # mainloop's `step_scales`. Its layout's steps are whole by the caller's (`k_major`), unsplit.
        from .templates.gemm import mosaic_gpu

        del prologue_operands, gather, hooks, dense, return_residuals, whole_steps
        target = generation.gpu_target()
        bound = self.bind(tokens, cotangent, sizes, out_dtype=out_dtype, interpret=interpret,
                          exact_sums=exact_sums, step_scales=step_scales)
        out = mosaic_gpu.tgmm(tokens.T, cotangent.T, sizes, tiles=config, out_dtype=out_dtype,
                              multiprocessors=self._launched(config, interpret, target),
                              interpret=INTERPRETED if interpret else False, exact_sums=exact_sums,
                              k_major=True, step_scales=tuple(step_scales),
                              **self._stated(bound, config, target))
        return out, None

    def _fp8(self, ba, dtype, family: mosaic.Family) -> bool:
        # Its operands lie rows-contiguous, as Hopper's FP8 tensor cores read both.
        return (jnp.dtype(dtype) in family.fp8_formats
                and ba.arguments['hooks'] == NO_HOOKS and bool(ba.arguments.get('exact_sums')))

    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
        return config.splits == 1 and super().candidate(ba, config, target)

    def _registers(self, ba, config: tiling.Tiles, *, merging: bool, target) -> int | None:
        # Beside the exact gradient's live set, a step's column scales.
        return _beside(super()._registers(ba, config, merging=merging, target=target),
                       _column_scale_registers(ba, config, target))


def _column_scale_registers(ba, config: tiling.Tiles, target=None) -> int:
    """The registers a step-scaled walk holds beside the exact one's: a step's fp32 column scales in its
    accumulator's column layout (`Family.column_registers`)."""
    from .templates.gemm import pricing

    family = pricing.family_of(config, ba.arguments['tokens'].dtype, target)
    return family.column_registers(config.block_n, jnp.float32)
