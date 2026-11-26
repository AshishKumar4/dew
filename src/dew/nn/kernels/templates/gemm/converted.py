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
    consumers' bfloat16 buffers (`mosaic_gpu.gmm_allocations`' `converted`), at the family's own steps,
    never swapped (no register decode takes a code's byte)."""

    STEPS: ClassVar[tuple[int, ...] | None] = None

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

        return mosaic_gpu.gmm_allocations(
            config, family, dtype=ba.arguments['tokens'].dtype, rhs_dtype=ba.arguments['kernel'].codes.dtype,
            transpose_rhs=True, prologue_axes=(), prologue_dtypes=(), epilogue_axes=(), epilogue_dtypes=(),
            dtypes=(ba.arguments['out_dtype'],))

    def _swaps_on(self, family: mosaic.Family, dtype) -> bool:
        del family, dtype
        return False

    def _operand_bytes(self, ba, tile: tiling.Tiles) -> tuple[tuple[float, float], tuple]:
        # Its rows' bytes and its codes' one, each stage converted into a bfloat16 buffer its consumers share
        # or each holds (`packed.shared_decode`).
        from dew.nn.kernels.templates.gemm import packed

        rows = jnp.dtype(ba.arguments['tokens'].dtype).itemsize
        copies = 1 if packed.shared_decode(tile, interpret=False) else tile.warpgroups
        return (rows, 1.0), (None, (rows, copies))

    def _registers(self, ba, config: tiling.Tiles, *, merging: bool, target) -> int | None:
        # Beside the plain product's live set, a conversion's chunk of a consumer's share of the stage's
        # columns (`mosaic.Family.converted`: a widened code and its fp32 value an entry).
        columns = config.block_n // config.warpgroups
        converted = 2 * columns * min(config.block_k, 128)
        return grouped_matmul._beside(grouped_matmul.MosaicGmm._registers(
            self, ba, config, merging=merging, target=target), converted // resources.THREADS)


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
