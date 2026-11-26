"""FP8 codes' product on a GPU whose tensor cores multiply no FP8 (sm80, sm86; row 8548): the template's
mainloop streams the matrix's codes, a byte each, and converts each stage exactly into bfloat16 before its
products read it (`packed.Decoding`, `mosaic.Family.converted`), so a frozen FP8 site's weights move one
byte a parameter where XLA widens them into a bfloat16 copy first; its rows, the codes of a step's tokens,
widen in XLA. The products are the codes' exact bfloat16 values summed in fp32, XLA's arm's arithmetic
(`xla_products.as_multiplied`), so check 1 holds the same reference. Off until a receipt finds it faster
(`FP8_CODES`)."""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp

from dew.nn.derivatives import contracted, surrogate
from dew.nn.kernels import gemm, generation, grouped_matmul, selection, xla_products
from dew.nn.kernels.templates import mosaic, resources
from dew.nn.kernels.templates.gemm import tiling
from dew.nn.precision import EXACT_FLOAT32
from dew.nn.sharding import automatic_axes

FP8_CODES_ROW = "ledger 8548"

STEP = 64
"""The depth and width an FP8 codes product's operands are padded to whole multiples of (zero codes, zero
products): a 64-column tile's and a bfloat16 step's least (`MosaicConvertedGmm.candidate`)."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Codes:
    """A matrix of FP8 codes as the converting launch binds it (`MosaicConvertedGmm`, `trans_rhs`): `codes`
    K-major, `[groups, width, depth]`, its shape theirs and its dtype the bfloat16 they widen to exactly, the
    products', as a packed matrix states its decoded one (`grouped_matmul._PackedMatrix`)."""

    codes: jax.Array | jax.ShapeDtypeStruct

    @property
    def shape(self) -> tuple[int, ...]:
        return self.codes.shape

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.dtype(jnp.bfloat16)


@dataclasses.dataclass(frozen=True)
class MosaicConvertedGmm(grouped_matmul.MosaicPackedGmm):
    """`MosaicPackedGmm` of a matrix of FP8 codes (`Codes`) beside bfloat16 rows on a family of no FP8
    products: its codes streamed as they lie, K-major, a byte each, each stage converted exactly into the
    consumers' bfloat16 buffers (`packed.CODES`), at the family's own steps. A decode-sized call runs
    swapped where the family swaps a bfloat16 call (mma.sync's: `MosaicGmm._swaps_on`): the codes the
    products' rows, each consumer converting its own rows' stage (`packed.ROW_CODES`), the call's rows the
    columns (row 5387)."""

    STEPS: ClassVar[tuple[int, ...] | None] = None
    SWAPPED_OPTIONS: ClassVar[dict] = {}

    @property
    def _scaled(self) -> bool:
        # Its codes' layout is the caller's, whole steps and tiles (`STEP`): the call is not padded.
        return True

    def _streamed_matrix(self, kernel: Codes | grouped_matmul._PackedMatrix, trans_rhs: bool) -> tuple:
        del trans_rhs
        assert isinstance(kernel, Codes), "a converting launch streams a matrix of codes"
        return kernel.codes, True, {}

    def _allocations(self, ba, config: tiling.Tiles, family: mosaic.Family):
        from dew.nn.kernels.templates.gemm import mosaic_gpu

        rows, matrix = (_stored_dtype(ba.arguments[name]) for name in ('tokens', 'kernel'))
        return mosaic_gpu.gmm_allocations(
            config, family, dtype=rows, rhs_dtype=matrix, transpose_rhs=True, prologue_axes=(),
            prologue_dtypes=(), epilogue_axes=(), epilogue_dtypes=(), dtypes=(ba.arguments['out_dtype'],))

    def _priced_operands(self, ba):
        # A swapped call's binding, its rows the codes (`_swapped`), is priced as its dtype's.
        return None if isinstance(ba.arguments['tokens'], Codes) else super()._priced_operands(ba)

    def _swaps_on(self, family: mosaic.Family, dtype) -> bool:
        return grouped_matmul.MosaicGmm._swaps_on(self, family, dtype)

    def candidate(self, ba, config: tiling.Tiles, target) -> bool:
        # Its stages convert at any width, each consumer its own share of columns or rows where they share
        # none (`packed.shared_decode`): no packed matrix's 64-column fragments, so a swapped call's narrow
        # columns (a decode's rows) take a tile.
        return (config.cluster == 1 and config.splits == 1 and not config.swapped
                and grouped_matmul.MosaicGmm.candidate(self, ba, config, target))

    def _swapped(self, ba, target):
        # The bfloat16 call's swapped binding (`MosaicGmm._swapped`), its rows the matrix's codes.
        swapped = grouped_matmul.MosaicGmm._swapped(self, ba, target)
        if swapped is None:
            return None
        codes = ba.arguments['kernel'].codes
        return type(swapped)(swapped.op, {**swapped.arguments, 'tokens': Codes(jax.ShapeDtypeStruct(
            swapped.arguments['tokens'].shape, codes.dtype))})

    def _launched_call(self, tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first,
                       gather, scatter, joined=None, *, out_dtype: str, trans_rhs: bool, interpret: bool,
                       hooks, exact_sums: bool, dense: bool, config: tiling.Tiles, formats: dict, target):
        if not config.swapped:
            return grouped_matmul.MosaicGmm._launched_call(
                self, tokens, kernel, sizes, prologue_operands, epilogue_operands, paired, first, gather,
                scatter, joined, out_dtype=out_dtype, trans_rhs=trans_rhs, interpret=interpret, hooks=hooks,
                exact_sums=exact_sums, dense=dense, config=config, formats=formats, target=target)
        # The codes the products' rows (`_swapped`), each consumer's converted in its stage (`ROW_CODES`).
        return self._swapped_launch(kernel.codes[0], tokens, sizes, out_dtype=out_dtype, interpret=interpret,
                                    exact_sums=exact_sums, config=config, target=target, **formats)

    def _operand_bytes(self, ba, tile: tiling.Tiles) -> tuple[tuple[float, float], tuple]:
        # Its rows' bytes and its codes' one, each stage of codes converted into bfloat16 buffers: the
        # matrix's its consumers share or each holds (`packed.shared_decode`), a swapped call's rows each
        # consumer's own.
        from dew.nn.kernels.templates.gemm import packed

        rows = jnp.dtype(ba.arguments['tokens'].dtype).itemsize
        if tile.swapped:
            return (1.0, rows), ((rows, tile.warpgroups), None)
        copies = 1 if packed.shared_decode(tile, interpret=False) else tile.warpgroups
        return (rows, 1.0), (None, (rows, copies))

    def _registers(self, ba, config: tiling.Tiles, *, merging: bool, target) -> int | None:
        # Beside the plain product's live set, a conversion's chunk of a consumer's share of its stage of
        # codes, the matrix's columns or a swapped call's rows (`mosaic.Family.converted`: a widened code
        # and its fp32 value an entry).
        share = (config.block_m if isinstance(ba.arguments['tokens'], Codes) else config.block_n)
        converted = 2 * share // config.warpgroups * min(config.block_k, 128)
        return grouped_matmul._beside(grouped_matmul.MosaicGmm._registers(
            self, ba, config, merging=merging, target=target), converted // resources.THREADS)


def _stored_dtype(operand) -> jnp.dtype:
    """An operand's dtype as its stream stores it: a matrix of codes' (`Codes`), else its own."""
    return jnp.dtype(operand.codes.dtype if isinstance(operand, Codes) else operand.dtype)


FP8_CODES: selection.Op[gemm.WordPackedCase] = selection.Op(generation.Choice.GROUPED_MATMUL, (
    selection.Kernel('mosaic', (
        selection.Requirement(
            'sharding', lambda case: case.whole,
            "automatic mesh axes keep XLA's widened product outside a per-shard launch", FP8_CODES_ROW),
        selection.Requirement(
            'measured', lambda case: False,
            "no receipt yet has the template's converted FP8 stages beat XLA's widened product (pending "
            "M-NG-42)", FP8_CODES_ROW, kind='measured'),
        selection.Requirement(
            'launch', lambda case: case.tile is not None,
            "no converting tile holds this shape on the target", FP8_CODES_ROW),
    )),
    selection.Kernel('xla', (), generic=True),
))
"""An FP8 codes product's selector on a GPU of no FP8 products (`fp8_codes_product`): the template's
converting launch where a receipt found it faster and its tile holds, else XLA's widened product."""


def fp8_codes_product(lhs: jax.Array, rhs: jax.Array, *, out_dtype, interpret_on_cpu: bool = False
                      ) -> jax.Array:
    """`lhs @ rhs.T` of FP8 codes, `[rows, depth]` and K-major `[width, depth]` (a frozen FP8 site's,
    `quantization._fp8_rows`), in `out_dtype`: `gemm.gemm`'s FP8 product where the target's tensor cores
    multiply FP8 (the promoted walk); on a CUDA GPU whose do not, each code's exact bfloat16 value, the
    products summed in fp32, through the template's converting launch where its selector takes it
    (`FP8_CODES`), else `gemm.gemm`'s XLA arm, which widens both. One derivative on every route: the rows'
    tangent times the matrix's exact values; the matrix, a frozen site's, takes none."""
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[1]:
        raise ValueError(f"an FP8 codes product is [rows, depth] by [width, depth], not {lhs.shape} by "
                         f"{rhs.shape}")
    options = {'dtype': rhs.dtype, 'out_dtype': jnp.dtype(out_dtype)}
    value = functools.partial(_valued, interpret=interpret_on_cpu, **options)
    product = contracted("test_an_fp8_codes_product_holds_every_order")(
        surrogate(value, functools.partial(_tangent, **options)))
    # The matrix as its bytes: an integer operand, which takes no tangent.
    return product(lhs, jax.lax.bitcast_convert_type(rhs, jnp.uint8))


def _valued(lhs: jax.Array, matrix: jax.Array, *, dtype, out_dtype, interpret: bool) -> jax.Array:
    """`fp8_codes_product`'s value, `matrix` its codes' bytes."""
    rhs = jax.lax.bitcast_convert_type(matrix, dtype)
    converted = _converted(lhs, rhs, interpret)
    if converted is None:
        return gemm.gemm(lhs, rhs, transpose_rhs=True, out_dtype=out_dtype, interpret_on_cpu=interpret)
    return converted.astype(out_dtype)


def _tangent(primals, tangents, value, *, dtype, out_dtype) -> jax.Array:
    """`fp8_codes_product`'s tangent: the rows' tangent times the matrix's exact values, an fp32 product at
    an exact boundary, whatever route the value took."""
    del value
    (_, matrix), (d_lhs, _) = primals, tangents
    rhs = jax.lax.bitcast_convert_type(matrix, dtype).astype(jnp.float32)
    return jax.lax.dot_general(d_lhs.astype(jnp.float32), rhs, (((1,), (1,)), ((), ())),
                               precision=EXACT_FLOAT32, preferred_element_type=jnp.float32).astype(out_dtype)


def _converted(lhs: jax.Array, rhs: jax.Array, interpret: bool) -> jax.Array | None:
    """The converting launch's fp32 sums of `lhs @ rhs.T` where the target is a CUDA GPU of no FP8 products
    and the selector takes it (`FP8_CODES`), each operand padded with zero codes to whole `STEP`s and the
    sums sliced back; None elsewhere."""
    target = generation.gpu_target().generation
    if (generation.capability(target) is None or generation.fp8_tensor_cores(target)
            or lhs.dtype != rhs.dtype or not mosaic.converts(rhs.dtype)):
        return None
    (rows, depth), width = lhs.shape, rhs.shape[0]
    padded_depth, padded_width = -(-depth // STEP) * STEP, -(-width // STEP) * STEP
    launch = MosaicConvertedGmm()

    def bound(rows_of, codes, sizes):
        return launch.bind(rows_of, Codes(codes), sizes, (), (), None, None, None, out_dtype="float32",
                           trans_rhs=True, interpret=interpret, dense=True, exact_sums=False)

    planned = bound(jax.ShapeDtypeStruct((rows, padded_depth), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1, padded_width, padded_depth), rhs.dtype),
                    jax.ShapeDtypeStruct((1,), jnp.int32))
    with xla_products.interpreted_context(interpret):
        case = gemm.WordPackedCase(functools.partial(gemm._planned, launch, planned), not automatic_axes())
        if selection.select(FP8_CODES, case).kernel != "mosaic":
            return None

    def kernel(rows_of, codes, sizes, *, interpret: bool):
        return launch(rows_of, Codes(codes), sizes, (), (), None, None, None, out_dtype='float32',
                      trans_rhs=True, interpret=interpret, dense=True, exact_sums=False)

    def portable(rows_of, codes, sizes):
        del sizes
        return jax.lax.dot_general(rows_of, codes[0].astype(jnp.bfloat16), (((1,), (1,)), ((), ())),
                                   preferred_element_type=jnp.float32)

    codes = jnp.pad(rhs, ((0, padded_width - width), (0, padded_depth - depth)))[None]
    widened = jnp.pad(lhs.astype(jnp.bfloat16), ((0, 0), (0, padded_depth - depth)))
    with xla_products.interpreted_context(interpret):
        summed = xla_products.dispatched(kernel, portable, interpret, widened, codes,
                                         jnp.asarray([rows], jnp.int32))
    return summed[:, :width]
