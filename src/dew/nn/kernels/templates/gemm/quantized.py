"""GEMM output quantization epilogues (`bits.encode_form`, rows 8547, 12037): inline PTX encoding of fp32
accumulators directly into FP8 (`E4M3`/`E5M2`), block-scaled MXFP8 (`formats.MxFp8`, 32-column E8M0 scales),
or block-scaled NVFP4 (`formats.NvFp4Block`, 16-column E4M3 scales at a supplied `tensor_scale`) using pure
2D register ops that lower across sm80/sm90/sm100/TPU."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import math
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike

from dew.algebra import bits
from dew.nn.float_bits import raw_fields
from dew.nn.kernels import launches
from dew.nn.kernels.templates import inline_forms, product as product_formats

if TYPE_CHECKING:
    from dew.nn.kernels.formats import MxFp8, NvFp4Block

_EPILOGUE_LAYOUT: contextvars.ContextVar[Any] = contextvars.ContextVar("gemm_epilogue_layout", default=None)
"""Active GPU accumulator register layout while tracing a compiled GEMM epilogue (`encode_tile`)."""

_MINIFLOAT_DTYPES = {bits.E4M3: jnp.dtype(jnp.float8_e4m3fn), bits.E5M2: jnp.dtype(jnp.float8_e5m2)}


def minifloat_dtype(minifloat: bits.Minifloat) -> jnp.dtype:
    """Storage float8 dtype for `minifloat` (`bits.E4M3` -> `float8_e4m3fn`, `bits.E5M2` -> `float8_e5m2`)."""
    if minifloat not in _MINIFLOAT_DTYPES:
        raise ValueError(f"unsupported FP8 minifloat {minifloat}")
    return _MINIFLOAT_DTYPES[minifloat]

@contextlib.contextmanager
def epilogue_layout(layout):
    """Bind `layout` as the active compiled-epilogue register layout for `encode_tile` (row 8547)."""
    token = _EPILOGUE_LAYOUT.set(layout)
    try:
        yield
    finally:
        _EPILOGUE_LAYOUT.reset(token)


def epilogue_register_layout():
    """The accumulator register layout a compiled GEMM epilogue is being traced in (`epilogue_layout`), None
    outside one."""
    return _EPILOGUE_LAYOUT.get()


def encode_tile(x: jax.Array, minifloat: bits.Minifloat) -> jax.Array:
    """Encode fp32 tile `x` into `minifloat` codes via `bits.encode_form`: inline PTX
    (`inline_forms.registers`) inside a compiled GPU epilogue, or `BitForm.traced` when interpreted or
    running outside a GPU kernel (row 8547)."""
    words = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    form = bits.encode_form(minifloat)
    layout = _EPILOGUE_LAYOUT.get() if launches.interpreting() is False else None
    codes = (form.traced(words)[0] if layout is None
             else inline_forms.registers(form, words, layout)[0]).astype(jnp.uint8)
    target_dtype = _MINIFLOAT_DTYPES.get(minifloat)
    return jax.lax.bitcast_convert_type(codes, target_dtype) if target_dtype is not None else codes


def _block_max(acc: jax.Array, block: int) -> jax.Array:
    """Per-row maximum of `|acc|` over each `block`-column group, broadcast back to `acc.shape`: within a
    kernel (or for a tile of no whole blocks, a launch identity's probe of a narrower one) one row reduction
    a block (`_blocks_reduced`), elsewhere one reduction of the blocks' view."""
    rows, columns = acc.shape
    if _EPILOGUE_LAYOUT.get() is None and columns % block == 0:
        # Its magnitudes' words reduced at its own width, as the quantizer reduces them (`bits.nvfp4_coded`):
        # a float maximum a flushing platform takes (XLA's CPU) read a subnormal accumulator as zero (row
        # 14083's review), and a float64 one narrowed first overflowed (train 1455's: 2^128 to infinity).
        words, sign, _, _ = raw_fields(acc)
        peaks = jnp.max((words ^ sign).reshape(rows, columns // block, block), axis=-1)
        return jnp.repeat(jax.lax.bitcast_convert_type(peaks, acc.dtype), block, axis=1)
    magnitude = jnp.abs(acc)
    block_id = _block_ids(acc.shape, block)
    peaks = jnp.zeros_like(acc)
    for group, peak in enumerate(_blocks_reduced(magnitude, block, jnp.float32(0.0))):
        peaks = jnp.where(block_id == group, jax.lax.broadcast_in_dim(peak, acc.shape, (0,)), peaks)
    return peaks


def _block_ids(shape: tuple[int, ...], block: int) -> jax.Array:
    """Each column's `block`-column group (`block` a power of two) of a tile of `shape`: its column's iota
    shifted, as a warpgroup's accumulator layouts lower it."""
    return jax.lax.broadcasted_iota(jnp.int32, shape, 1) >> jnp.int32(block.bit_length() - 1)


def _blocks_reduced(values: jax.Array, block: int, fill: jax.Array) -> list[jax.Array]:
    """Each `block`-column group's row maxima of `values` (`[rows, columns]`): one row reduction a group, its
    columns selected by their group (`_block_ids`), the others `fill`, the form Mosaic GPU lowers in a
    warpgroup's accumulator layouts (tcgen05's and WGMMA's), where a reshape to the blocks and its transpose
    has no lowering (witness/b2-blockmax-1, row 12721); one group's, its row reduction alone."""
    groups = values.shape[1] // block
    if groups == 1:
        return [jnp.max(values, axis=1)]
    block_id = _block_ids(values.shape, block)
    return [jnp.max(jnp.where(block_id == group, values, fill), axis=1) for group in range(groups)]


def _divided(numerator: jax.Array, denominator: jax.Array | float) -> jax.Array:
    """The IEEE fp32 quotient of a tile by one of its shape or by a constant: PTX's `div.rn.f32` inline in a
    compiled GPU epilogue (`inline_forms.quotients`, one instruction where the software rule issued dozens an
    element), the software rule elsewhere (`precision.lowered_divide`: XLA's arms, interpreted or not, whose
    own division a CPU flushes and a GPU approximates; row 4023), the same quotient."""
    from dew.nn.precision import lowered_divide

    layout = _EPILOGUE_LAYOUT.get() if launches.interpreting() is False else None
    if layout is None:
        return lowered_divide(numerator, jnp.asarray(denominator, jnp.float32), jnp.int32(0))
    return inline_forms.quotients(numerator, denominator, layout)


def _quantized_epilogue(minifloat: bits.Minifloat, acc: jax.Array,
                        scale: jax.Array | None = None) -> jax.Array:
    if scale is None:
        return encode_tile(acc, minifloat)
    s = jax.lax.broadcast_in_dim(scale, acc.shape, (0,)) if scale.ndim == 1 else scale
    return encode_tile(_divided(acc, s), minifloat)


def quantize_epilogue(minifloat: bits.Minifloat):
    """GEMM epilogue packing fp32 accumulators to `minifloat` codes (`bits.encode_form`, row 8547)."""
    return functools.partial(_quantized_epilogue, minifloat)


def _nvfp4_quantize_epilogue(acc: jax.Array, tensor_scale: jax.Array) -> tuple[jax.Array, jax.Array]:
    """GEMM epilogue quantizing each 16-column block of fp32 `acc` (`[rows, cols]`, `cols % 16 == 0`) to
    E2M1 nibbles (`uint8`) and E4M3 block scales (`float8_e4m3fn`, broadcast over each 16-column block) via
    `bits.encode_form` inline PTX and IEEE division (`_divided`), bitwise matching
    `formats.NvFp4Block.quantized` at `tensor_scale` (rows 8547, 12037)."""
    from dew.nn.fake_quant import E2M1_MAX

    block_max = _block_max(acc, product_formats.NVFP4.depth)
    ts_in = tensor_scale.astype(jnp.float32)
    ts = (jax.lax.broadcast_in_dim(ts_in, acc.shape, (0,)) if ts_in.ndim == 1
          else jnp.broadcast_to(ts_in, acc.shape))
    ratio = _divided(_divided(block_max, ts), float(E2M1_MAX))
    nearest_f8 = encode_tile(ratio, bits.E4M3)
    nearest = jax.lax.bitcast_convert_type(nearest_f8, jnp.uint8).astype(jnp.uint32)
    scale_u32 = bits.scales_rounded_up(nearest, ratio)
    each = bits.scales_decoded(scale_u32)
    pos = each > jnp.float32(0.0)
    safe_each = jnp.where(pos, each, jnp.float32(1.0))
    scaled = jnp.where(pos, _divided(_divided(acc, ts), safe_each), jnp.float32(0.0))
    e2m1_codes = encode_tile(scaled, bits.E2M1)
    scale_f8 = jax.lax.bitcast_convert_type(scale_u32.astype(jnp.uint8), jnp.float8_e4m3fn)
    return e2m1_codes, scale_f8


def quantize_nvfp4_epilogue():
    """GEMM epilogue returning `(e2m1_nibbles [uint8], block_scales [float8_e4m3fn])` for each 16-column
    block at side operand `tensor_scale` via `bits.encode_form` inline PTX (rows 8547, 12037)."""
    return _nvfp4_quantize_epilogue


def _mxfp8_quantize_epilogue(acc: jax.Array) -> tuple[jax.Array, jax.Array]:
    """GEMM epilogue quantizing each 32-column block of fp32 `acc` (`[rows, cols]`, `cols % 32 == 0`) to
    E4M3 codes (`float8_e4m3fn`) and OCP MX E8M0 block scales (`float8_e8m0fnu`, broadcast over each
    32-column block) via `bits.encode_form` inline PTX, bitwise matching `formats.MxFp8.quantized` (rows
    8547, 12037)."""
    from dew.nn.kernels.formats import mxfp8_scale_bytes

    # The scale's exponent and E8M0 byte from the block's largest magnitude's bits, as the quantizer takes
    # them (`formats.mxfp8_scale_bytes`; rows 13164, 13249).
    exponent, stored = mxfp8_scale_bytes(_block_max(acc, product_formats.MXFP8.depth))
    # Multiply by the exact normal float32 power 2^-exponent, set in its exponent bits.
    inv_scale = jax.lax.bitcast_convert_type((127 - exponent).astype(jnp.uint32) << jnp.uint32(23),
                                             jnp.float32)
    codes = encode_tile(acc * inv_scale, bits.E4M3)
    scales = jax.lax.bitcast_convert_type(stored.astype(jnp.uint8), jnp.float8_e8m0fnu)
    return codes, scales


def quantize_mxfp8_epilogue():
    """GEMM epilogue returning `(e4m3_codes [float8_e4m3fn], e8m0_scales [float8_e8m0fnu])` for each
    32-column block via `bits.encode_form` inline PTX (rows 8547, 12037)."""
    return _mxfp8_quantize_epilogue


def _blockwise_fp8_quantized(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`values` (`[rows, cols]`, `cols % SCALE_DEPTH == 0`, fp32: the bf16 or fp32 values a recipe quantizes)
    as E4M3 codes a 1 x `SCALE_DEPTH` tile at a time and each tile's power-of-two scale (fp32, broadcast over
    its tile's columns), bitwise `projections.rows_quantized`'s: its scale `fake_quant.fp8_block_scale` of the
    tile's amax, its codes E4M3 of the values over it, the IEEE quotient a product by its exact reciprocal
    (`_reciprocal`, an infinite or NaN block's as the recipe's), by `bits.encode_form` (ties to even,
    saturated), the tile's amax `_block_max`'s (DeepSeek-V3's blockwise FP8 recipe, row 8547)."""
    from dew.nn.fake_quant import fp8_block_scale

    scale = fp8_block_scale(_block_max(values, product_formats.SCALE_DEPTH))
    return encode_tile(values * _reciprocal(scale), bits.E4M3), scale


def _reciprocal(scale: jax.Array) -> jax.Array:
    """`1 / scale` exactly for `fake_quant.fp8_block_scale`'s scales, so a product by it is the IEEE
    quotient: a normal power of two's its exponent negated (every finite block's: its scale between 2**-22 and
    2**120), +0 for its infinite one (an infinite amax's) and -inf for its -0 (a NaN amax's sentinel)."""
    word = jax.lax.bitcast_convert_type(scale, jnp.uint32)
    biased = (word >> jnp.uint32(23)) & jnp.uint32(0xFF)
    inverse = jax.lax.bitcast_convert_type((jnp.uint32(254) - biased) << jnp.uint32(23), jnp.float32)
    infinite, sentinel = biased == jnp.uint32(0xFF), word == jnp.uint32(0x80000000)
    return jnp.where(infinite, jnp.float32(0.0), jnp.where(sentinel, jnp.float32(-jnp.inf), inverse))


def blockwise_fp8_epilogue(activation=None, out_dtype=jnp.bfloat16):
    """A GEMM epilogue returning its product rounded to `out_dtype` beside its next product's rows quantized
    as DeepSeek-V3's blockwise FP8 recipe quantizes them: `activation` of its side tiles and the rounded
    product (`activation(*sides, product)`: a gated MLP's rounded gate product its side, row 12689; the
    product itself where none), rounded to `out_dtype`, as E4M3 codes and 1 x 128 power-of-two scales
    (`_blockwise_fp8_quantized`), bitwise the recipe's `projections.rows_quantized` of that activation, which
    a separate pass formed from the stored products (row 8547)."""
    _read_in_fp32(out_dtype)

    def quantized(acc, *sides):
        rounded, next_rows = _activated(acc, sides, activation, out_dtype)
        return rounded, *_blockwise_fp8_quantized(next_rows)

    return quantized


def nvfp4_epilogue(activation=None, out_dtype=jnp.bfloat16):
    """A GEMM epilogue returning its product rounded to `out_dtype` beside its next product's rows quantized
    as NVFP4 at a given tensor scale, its last side (a delayed-scaling recipe's, each row's the one scale;
    row 12721): `activation` of its other side tiles and the rounded product, rounded to `out_dtype`, as
    E2M1 codes one a byte (uint8) and E4M3 block scales broadcast over each block's 16 columns
    (`_nvfp4_quantize_epilogue`), bitwise `formats.NvFp4Block.quantized` of that activation at the scale,
    which a separate pass formed from the stored products. Its product the fp32 sums a caller has scaled
    (`gemm.nvfp4_block_finished`: its operands' two tensor scales)."""
    _read_in_fp32(out_dtype)

    def quantized(acc, *sides):
        *tiles, tensor_scale = sides
        rounded, next_rows = _activated(acc, tuple(tiles), activation, out_dtype)
        return rounded, *_nvfp4_quantize_epilogue(next_rows, tensor_scale)

    return quantized


def _read_in_fp32(out_dtype) -> None:
    """Refuse a quantizing epilogue's `out_dtype` wider than fp32, which its quantizer reads values in."""
    if jnp.dtype(out_dtype).itemsize > 4:
        raise ValueError(f"a quantizing epilogue reads fp32 at most, not {jnp.dtype(out_dtype)}")


def _activated(acc: jax.Array, sides: tuple, activation, out_dtype) -> tuple[jax.Array, jax.Array]:
    """A quantizing epilogue's product rounded to `out_dtype`, and its next product's rows in fp32:
    `activation` of the side tiles and the rounded product, rounded to `out_dtype` again (the rounded product
    itself where none), as a separate pass forms them from the stored products."""
    rounded = acc.astype(out_dtype)
    next_rows = rounded if activation is None else activation(*sides, rounded).astype(out_dtype)
    return rounded, next_rows.astype(jnp.float32)


class Form(NamedTuple):
    """How a several-output call stores one of its outputs (`Hooks.out_forms`; a tile of its own where None):
    `divisor` of its tile's columns a stored value. Reduced (the default): the value its epilogue broadcast
    across each `divisor`-column block, one a row a block (a next product's block scales, `reduced`),
    `transposed` its rows contiguous (`[columns // divisor, rows]`, as a block-scaled product reads them).
    `packed`: two 4-bit codes a byte (`divisor` 2, row-major), the even column's in its low nibble, as a
    sub-byte array packs them (NVFP4's E2M1 codes, `paired`; row 12743). `narrowed`, where nonzero: its
    epilogue computes it a `divisor`-th as wide as the tile (`[rows, columns // divisor]`, row-major) from
    whole runs of `narrowed` columns, `divisor` of them a stored run (a wide gated product's hidden
    activation, one of each gate and up run, `templates.gemm.wide.GROUP`), stored as computed; a tile, and
    every piece its issuer reads out at once, holds whole `unit`s."""

    divisor: int
    transposed: bool = False
    packed: bool = False
    narrowed: int = 0

    @property
    def unit(self) -> int:
        """The tile columns its epilogue narrows whole (`divisor` runs of `narrowed`), else its `divisor`."""
        return self.divisor * self.narrowed if self.narrowed else self.divisor

    def __repr__(self) -> str:
        # A launch's pick key reads its forms' repr (`grouped_launch._GroupedLaunch.traced`): a form that
        # narrows nothing keys as it did before `narrowed` was a field.
        if not self.narrowed:
            return f"Form(divisor={self.divisor!r}, transposed={self.transposed!r}, packed={self.packed!r})"
        return (f"Form(divisor={self.divisor!r}, transposed={self.transposed!r}, packed={self.packed!r}, "
                f"narrowed={self.narrowed!r})")


def forms_of(out_forms: tuple | None, dtypes: tuple) -> tuple:
    """Each of a call's outputs' form (`Hooks.out_forms`, each a `Form` or its fields' tuple), the outputs'
    `dtypes` its several outputs' (`out_dtypes`): None for a tile of its own; none stated, every output a
    tile. A packed form is two codes a byte, row-major, stored uint8 (its every route's: `_pairs`), the one
    boundary each route's selection reads."""
    if not out_forms:
        return ()
    if len(out_forms) != len(dtypes):
        raise ValueError(f"{len(out_forms)} output forms for {len(dtypes)} outputs")
    forms = _formed(out_forms)
    for form, dtype in zip(forms, dtypes, strict=True):
        if form is not None and form.packed and (form.divisor != 2 or form.transposed
                                                 or jnp.dtype(dtype) != jnp.dtype(jnp.uint8)):
            raise ValueError(f"a packed output is two 4-bit codes a byte, row-major, uint8, not {form} in "
                             f"{jnp.dtype(dtype)}")
        if form is not None and form.narrowed and (form.divisor < 2 or form.transposed or form.packed):
            raise ValueError(f"a narrowed output is a divisor-th of the tile's columns, row-major, not "
                             f"{form}")
    return forms


def _formed(out_forms) -> tuple:
    """Each of `out_forms` as a `Form` (`_form`), None as it is."""
    return tuple(None if form is None else _form(form) for form in out_forms or ())


def _form(form: Form | tuple) -> Form:
    """`form` as a `Form`, its fields' tuple read in their order."""
    return form if isinstance(form, Form) else Form(*form)


def reduced(buffer, piece: jax.Array) -> None:
    """A reduced output's chunk written (`stores.Output.write`): `piece`, a consumer's rows by the chunk's
    columns (its staging's, whole blocks: `formed`), each block's value broadcast across its columns by its
    epilogue, into `buffer` (`[blocks, rows]`, transposed: `forms_refusal`) as each row's maximum over each
    block, a buffer row a block: one row reduction a block of the accumulator's layout (`_blocks_reduced`),
    which Mosaic GPU lowers where a strided read of every `divisor`-th column, or a reshape to the blocks and
    its transpose, does not (witness/packing-1, sm90; witness/b2-blockmax-1, sm90 and sm100). A chunk of
    tcgen05's 64 columns holds four NVFP4 blocks (row 12721), a Hopper one a block."""
    blocks = buffer.shape[0]
    maxima = _blocks_reduced(piece.astype(jnp.float32), piece.shape[1] // blocks, jnp.float32(-jnp.inf))
    for block, peak in enumerate(maxima):
        buffer[block, :] = peak.astype(buffer.dtype)


def paired(buffer, piece: jax.Array) -> None:
    """A packed output's chunk written (`stores.Output.write`): `piece`, a consumer's rows by the chunk's
    columns of 4-bit codes, into `buffer` (`[rows, columns / 2]` uint8) two a byte (`_pairs`)."""
    buffer[...] = _pairs(piece)


def _pairs(codes: jax.Array) -> jax.Array:
    """`codes` (`[rows, columns]`, 4-bit codes) two a byte, the even column's in its low nibble, as a
    sub-byte array packs them: in a compiled GPU epilogue's register layout each pair in the thread that
    holds it (`inline_forms.paired`), elsewhere its columns' view as pairs (interpreted, and XLA's form of a
    packed output's store, `stored`), the same bytes."""
    layout = _EPILOGUE_LAYOUT.get() if launches.interpreting() is False else None
    if layout is not None:
        return inline_forms.paired(codes.astype(jnp.uint8), layout)
    rows, columns = codes.shape
    both = codes.astype(jnp.uint8).reshape(rows, columns // 2, 2)
    return both[..., 0] | (both[..., 1] << jnp.uint8(4))


def outputs(part: tuple[int, int], dtypes: tuple, forms: tuple) -> tuple:
    """A consumer's share `part` of each output as its family stages it: `(shape, dtype)` for a tile,
    `stores.Output` for one a form stores (`forms_of`), written by `paired` where packed, else `reduced`."""
    from dew.nn.kernels.templates import stores

    if not forms:
        return tuple((part, jnp.dtype(dtype)) for dtype in dtypes)

    def output(dtype, form):
        if form is None:
            return part, jnp.dtype(dtype)
        if form.narrowed:
            return stores.Output(part, jnp.dtype(dtype), divisor=form.divisor, narrowed=True)
        return stores.Output(part, jnp.dtype(dtype), divisor=form.divisor,
                             write=paired if form.packed else reduced, transposed=form.transposed)

    return tuple(output(dtype, form) for dtype, form in zip(dtypes, _formed(forms), strict=True))


def declared(dtype, form: Form | tuple, rows: int, columns: int) -> jax.ShapeDtypeStruct:
    """An output a form stores (`forms_of`) as its call declares it: one value a row each `divisor` of
    `columns`, `[columns // divisor, rows]` where transposed, `[rows, columns // divisor]` otherwise."""
    form = _form(form)
    if columns % form.divisor:
        raise ValueError(f"{columns} columns hold no whole {form.divisor}-column blocks")
    stored_columns = columns // form.divisor
    return jax.ShapeDtypeStruct((stored_columns, rows) if form.transposed else (rows, stored_columns),
                                jnp.dtype(dtype))


def _chunk_unit(out_forms: tuple | None) -> int | None:
    """The columns each staged chunk is whole numbers of where a call's outputs state forms its stores write
    (every such form's `divisor`: a reduced output's blocks, a packed one's pairs), None for none: a narrowed
    one is stored as computed (`forms_refusal` holds its divisor)."""
    divisors = [form.divisor for form in _formed(out_forms) if form is not None and not form.narrowed]
    return math.lcm(*divisors) if divisors else None


def formed(config, out_forms: tuple | None):
    """`config` (a GEMM tile) staging its outputs whole units of its forms' columns at a time where the call
    stores outputs in forms (`Hooks.out_forms`; two buffers, `_block_chunk`'s columns), so each chunk a form's
    `write` takes is whole blocks and pairs (`reduced`, `paired`); `config` as it is elsewhere, where it
    states its staging, or where it stages no such chunk (a tcgen05 tile's 128-column FP8 blocks, wider than
    the 64 columns its issuer reads out at once: `Tiles.chunked_refusal`), which `forms_refusal` refuses, so
    its family declines."""
    import dataclasses

    from dew.nn.kernels.templates import bundles, stores

    unit = _chunk_unit(out_forms)
    if unit is None or config.staging is not None:
        return config
    staging = (_block_chunk(config.issued(bundles.named(config.family, jnp.bfloat16, None)), unit),
               stores.OUTPUT_BUFFERS)
    if config.chunked_refusal(staging) is not None:
        return config
    return dataclasses.replace(config, staging=staging)


def _block_chunk(issued: tuple[int, int], unit: int) -> int:
    """The columns a chunk of outputs in forms stages, its tile's `issued` columns and pieces
    (`Tiles.issued`): a piece its issuer reads out narrower than the tile, where whole `unit`s (tcgen05's 64
    columns of tensor memory, four NVFP4 blocks), so no piece is sliced in registers; else one unit, a slice
    of a register accumulator's (row 12721)."""
    columns, piece = issued
    return piece if piece < columns and piece % unit == 0 else unit


def forms_refusal(out_forms: tuple | None, config, *, dense: bool) -> str | None:
    """Why a GEMM launch at `config` stores no output in its form (`Hooks.out_forms`), None where it does or
    the call states none: its staging whole units of the forms' columns a chunk (`formed`, which `reduced`
    and `paired` write), each reduced output transposed (`reduced` writes a block's maxima as its buffer's
    row, no strided column), and its call dense (a transposed store takes a share's whole rows,
    `stores.transposed_rows`, and a packed row's store a share's from its first, `stores._aligned_ladder`: a
    grouped tile's share starts mid-tile); a narrowed output (`Form.narrowed`) where the tile's family stores
    one (`mosaic.Transport.stores_narrowed`), its stated chunk whole multiples of its divisor."""
    from dew.nn.kernels.templates import bundles

    narrowed = [form for form in _formed(out_forms) if form is not None and form.narrowed]
    if narrowed:
        family = bundles.named(config.family, jnp.bfloat16, None)
        if not family.stores_narrowed:
            return f"a {config.family} launch stores no output narrower than its tile"
        if not dense:
            return ("a narrowed output is a dense call's (a scattered one's rows land a row at a time, none "
                    "narrowed)")
        # Its epilogue narrows whole runs (`Form.unit`): every tile's and every piece its issuer reads out at
        # once (tcgen05's 64 columns), and its stated chunk whole stored columns.
        unit = math.lcm(*(form.unit for form in narrowed))
        if any(each % unit for each in config.issued(family)):
            return (f"a {config.issued(family)} tile's columns and pieces hold no whole {unit}-column runs "
                    "of a narrowed output")
        if config.staging is not None and any(config.staging[0] % form.divisor for form in narrowed):
            return f"a chunk of {config.staging[0]} columns holds no whole column groups of a narrowed output"
    unit = _chunk_unit(out_forms)
    if unit is None:
        return None
    if config.staging is None or config.staging[0] % unit:
        return f"a chunk of outputs in forms is whole {unit}-column units, not {config.staging}"
    if any(form is not None and not (form.packed or form.transposed or form.narrowed)
           for form in _formed(out_forms)):
        return "a launch stores a reduced output transposed, a buffer row a block"
    if not dense:
        return "an output in a form is a dense call's"
    return None


def stored(value: jax.Array, form: Form | tuple | None) -> jax.Array:
    """`value` (`[rows, columns]`, an epilogue's output) as its call stores it (`declared`): itself for a
    tile; two codes a byte where `form` packs it (`_pairs`); else a block's first column each block (its
    value broadcast across its block), `[columns // divisor, rows]` where transposed: XLA's form of a form's
    store."""
    if form is None:
        return value
    form = _form(form)
    if form.narrowed:
        return value
    if form.packed:
        return _pairs(value)
    blocks = value[:, ::form.divisor]
    return blocks.T if form.transposed else blocks


def trimmed(outs: tuple, forms: tuple, rows: int | None, columns: int | None = None) -> tuple:
    """A call's outputs (`declared`'s) to their first `rows` rows and `columns` columns where given, along
    each output's own axes: a transposed output's rows its second axis, its columns its first, a `divisor`th
    of them."""
    def one(out, form):
        if form is None:
            return out[:rows] if columns is None else out[:rows, :columns]
        kept = None if columns is None else -(-columns // form.divisor)
        return out[:kept, :rows] if form.transposed else out[:rows, :kept]

    held = _formed(forms) or (None,) * len(outs)
    return tuple(one(out, form) for out, form in zip(outs, held, strict=True))


def typed_outputs(dtypes: tuple, out_forms: tuple | None, config, *, dense: bool) -> tuple[tuple, tuple]:
    """A `gmm` call's outputs' dtypes and forms (`forms_of`), refused where its tile stores none of them
    (`forms_refusal`): a direct call's as a launch's planning refuses it."""
    typed = tuple(jnp.dtype(dtype) for dtype in dtypes)
    forms = forms_of(out_forms, typed)
    if (refusal := forms_refusal(forms, config, dense=dense)) is not None:
        raise ValueError(f"{config}: {refusal}")
    return typed, forms


def output_bytes(dtypes: tuple, out_forms) -> float:
    """`stored_bytes` of a call's outputs' `dtypes` under the forms it names (`forms_of` its `out_forms`)."""
    return stored_bytes(dtypes, forms_of(out_forms, tuple(jnp.dtype(dtype) for dtype in dtypes)))


def stored_bytes(dtypes: tuple, forms: tuple) -> float:
    """The bytes a call's outputs store an element of its output: each dtype's over its form's divisor
    (a packed output's two codes a byte, a reduced one's a value a block), none where a form is None."""
    return sum(jnp.dtype(dtype).itemsize / (1 if form is None else form.divisor)
               for dtype, form in zip(dtypes, forms or (None,) * len(dtypes), strict=True))


def stored_all(outs, forms: tuple):
    """A call's outputs each as its form stores it (`stored`), the outputs as they are where none states
    one: a several-output call's alone (`forms_of`)."""
    if not forms:
        return outs
    return tuple(stored(out, form) for out, form in zip(outs, forms, strict=True))


def declared_all(dtypes: tuple, forms: tuple, splits: int, tile, rows: int, columns: int) -> tuple:
    """A `gmm` call's outputs as it declares them where a form reduces one (`declared`), the others its
    tiles (`tile(dtype)`); a call of no depth split's."""
    if splits != 1:
        raise ValueError("an output a form reduces is a call's of no depth split")
    return tuple(tile(dtype) if form is None else declared(dtype, form, rows, columns)
                 for dtype, form in zip(dtypes, forms, strict=True))


def gemm_quantized(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None = None, *,
                   minifloat: bits.Minifloat | type[MxFp8] | type[NvFp4Block] = bits.E4M3,
                   tensor_scale: jax.Array | float | None = None,
                   transpose_rhs: bool = False, precision: PrecisionLike = None,
                   interpret_on_cpu: bool = False) -> tuple[jax.Array, jax.Array] | MxFp8 | NvFp4Block:
    """GEMM whose epilogue quantizes fp32 accumulators directly to FP8 `(codes, scale)` (`bits.E4M3`/
    `bits.E5M2`, dividing by `tensor_scale` on encode so `codes * scale ≈ acc`), block-scaled MXFP8
    (`formats.MxFp8`, bitwise matching `MxFp8.quantized`), or block-scaled NVFP4 (`bits.E2M1` /
    `formats.NvFp4Block`, bitwise matching `NvFp4Block.quantized` at `tensor_scale`; rows 8547, 12037,
    12160)."""
    from dew.nn.kernels.formats import MxFp8, NvFp4Block
    from dew.nn.kernels.gemm import gemm
    from dew.nn.kernels.templates.hooks import Hook, Side

    width = rhs.shape[-2] if transpose_rhs else rhs.shape[-1]
    if minifloat in (bits.E4M3, bits.E5M2):
        assert isinstance(minifloat, bits.Minifloat)
        out_dtype = minifloat_dtype(minifloat)
        ts = jnp.asarray(1.0 if tensor_scale is None else tensor_scale, jnp.float32)
        sides = (() if tensor_scale is None else (Side(jnp.broadcast_to(ts, (lhs.shape[0],)), ('rows',)),))
        out = gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs,
                   epilogue=Hook(quantize_epilogue(minifloat), sides),
                   out_dtype=out_dtype, precision=precision, interpret_on_cpu=interpret_on_cpu)
        assert isinstance(out, jax.Array)
        return out, ts
    if minifloat is MxFp8:
        if width % product_formats.MXFP8.depth != 0:
            raise ValueError(f"gemm_quantized(MxFp8) requires width divisible by "
                             f"{product_formats.MXFP8.depth}, got {width}")
        out = gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs,
                   epilogue=Hook(quantize_mxfp8_epilogue()), out_dtype=jnp.float8_e4m3fn,
                   out_dtypes=(jnp.float8_e4m3fn, jnp.float8_e8m0fnu), precision=precision,
                   interpret_on_cpu=interpret_on_cpu)
        assert isinstance(out, tuple)
        codes, block_scales = out
        return MxFp8(codes, block_scales[..., ::32])
    if minifloat == bits.E2M1 or minifloat is NvFp4Block:
        if width % product_formats.NVFP4.depth != 0:
            raise ValueError(
                f"gemm_quantized(NvFp4Block) requires width divisible by {product_formats.NVFP4.depth}, "
                f"got {width}")
        # The one boundary every given scale passes, as `NvFp4Block.quantized`'s does.
        ts = bits.nvfp4_given_scale(1.0 if tensor_scale is None else tensor_scale)
        out = gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs,
                   epilogue=Hook(quantize_nvfp4_epilogue(),
                                 (Side(jnp.broadcast_to(ts, (lhs.shape[0],)), ('rows',)),)),
                   out_dtype=jnp.uint8, out_dtypes=(jnp.uint8, jnp.float8_e4m3fn),
                   precision=precision, interpret_on_cpu=interpret_on_cpu)
        assert isinstance(out, tuple)
        nibbles, block_scales = out
        return NvFp4Block.encoded(nibbles, block_scales[..., ::16], ts)
    raise ValueError(f"gemm_quantized supports E4M3, E5M2, E2M1, MxFp8, or NvFp4Block, got {minifloat}")
