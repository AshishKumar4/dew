"""GEMM output quantization epilogues (`bits.encode_form`, rows 8547, 12037): inline PTX encoding of fp32
accumulators directly into FP8 (`E4M3`/`E5M2`), block-scaled MXFP8 (`formats.MxFp8`, 32-column E8M0 scales),
or block-scaled NVFP4 (`formats.NvFp4Block`, 16-column E4M3 scales at a supplied `tensor_scale`) using pure
2D register ops that lower across sm80/sm90/sm100/TPU."""

from __future__ import annotations

import contextlib
import contextvars
import functools
from typing import Any

import jax
import jax.numpy as jnp

from dew.algebra import bits
from dew.nn.kernels import launches
from dew.nn.kernels.templates import inline_forms, product as product_formats

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
    """Per-row maximum of `|acc|` over each `block`-column group, broadcast back to `acc.shape`."""
    shift = block.bit_length() - 1
    block_id = jax.lax.broadcasted_iota(jnp.int32, acc.shape, 1) >> jnp.int32(shift)
    abs_acc = jnp.abs(acc)
    peaks = jnp.zeros_like(acc)
    for group in range(acc.shape[1] // block):
        bmax = jnp.max(jnp.where(block_id == group, abs_acc, jnp.float32(0.0)), axis=1)
        peaks = jnp.where(block_id == group, jax.lax.broadcast_in_dim(bmax, acc.shape, (0,)), peaks)
    return peaks


def _quantized_epilogue(minifloat: bits.Minifloat, acc: jax.Array,
                        scale: jax.Array | None = None) -> jax.Array:
    if scale is None:
        return encode_tile(acc, minifloat)
    from dew.nn.precision import lowered_divide

    s = jax.lax.broadcast_in_dim(scale, acc.shape, (0,)) if scale.ndim == 1 else scale
    return encode_tile(lowered_divide(acc, s, jnp.int32(0)), minifloat)


def quantize_epilogue(minifloat: bits.Minifloat):
    """GEMM epilogue packing fp32 accumulators to `minifloat` codes (`bits.encode_form`, row 8547)."""
    return functools.partial(_quantized_epilogue, minifloat)


def _nvfp4_quantize_epilogue(acc: jax.Array, tensor_scale: jax.Array) -> tuple[jax.Array, jax.Array]:
    """GEMM epilogue quantizing each 16-column block of fp32 `acc` (`[rows, cols]`, `cols % 16 == 0`) to
    E2M1 nibbles (`uint8`) and E4M3 block scales (`float8_e4m3fn`, broadcast over each 16-column block) via
    `bits.encode_form` inline PTX and `precision.lowered_divide`, bitwise matching
    `formats.NvFp4Block.quantized` at `tensor_scale` (rows 8547, 12037)."""
    from dew.nn.fake_quant import E2M1_MAX
    from dew.nn.precision import lowered_divide

    block_max = _block_max(acc, product_formats.NVFP4.depth)
    zero = jnp.int32(0)
    ts_in = tensor_scale.astype(jnp.float32)
    ts = (jax.lax.broadcast_in_dim(ts_in, acc.shape, (0,)) if ts_in.ndim == 1
          else jnp.broadcast_to(ts_in, acc.shape))
    e2m1_max = jnp.full(acc.shape, E2M1_MAX, jnp.float32)
    ratio = lowered_divide(lowered_divide(block_max, ts, zero), e2m1_max, zero)
    nearest_f8 = encode_tile(ratio, bits.E4M3)
    nearest = jax.lax.bitcast_convert_type(nearest_f8, jnp.uint8).astype(jnp.uint32)
    scale_u32 = jnp.where(bits.scales_decoded(nearest) < ratio,
                          jnp.minimum(nearest + jnp.uint32(1), jnp.uint32(bits.E4M3.largest)), nearest)
    each = bits.scales_decoded(scale_u32)
    pos = each > jnp.float32(0.0)
    safe_each = jnp.where(pos, each, jnp.float32(1.0))
    scaled = jnp.where(pos, lowered_divide(lowered_divide(acc, ts, zero), safe_each, zero), jnp.float32(0.0))
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
    block_max = _block_max(acc, product_formats.MXFP8.depth)
    biased = (jax.lax.bitcast_convert_type(block_max, jnp.uint32) >> jnp.uint32(23)) & jnp.uint32(0xFF)
    # OCP MX rule: exponent = clip(max_exp - 8, -127, 127) -> biased E8M0 byte = clip(biased - 8, 0, 254).
    scale_byte = jnp.where(biased > jnp.uint32(8), jnp.minimum(biased - jnp.uint32(8), jnp.uint32(254)),
                           jnp.uint32(0))
    # Multiply by exact normal float32 power 2^(127 - scale_byte) = bitcast((254 - scale_byte) << 23).
    inv_scale = jax.lax.bitcast_convert_type((jnp.uint32(254) - scale_byte) << jnp.uint32(23), jnp.float32)
    codes = encode_tile(acc * inv_scale, bits.E4M3)
    scales = jax.lax.bitcast_convert_type(scale_byte.astype(jnp.uint8), jnp.float8_e8m0fnu)
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
    saturated). Within a kernel the tile's
    amax a row reduction of each tile's columns of the accumulator's layout (`_block_max`), elsewhere one
    reduction of the tiles' view (DeepSeek-V3's blockwise FP8 recipe, row 8547)."""
    from dew.nn.fake_quant import fp8_block_scale

    block = product_formats.SCALE_DEPTH
    rows, cols = values.shape
    if _EPILOGUE_LAYOUT.get() is None and cols % block == 0:
        tiles = jnp.max(jnp.abs(values.reshape(rows, cols // block, block)), axis=-1)
        amax = jnp.repeat(tiles, block, axis=1)
    else:
        # A kernel's accumulator layout, or a launch identity's probe of a narrower tile (`traced`).
        amax = _block_max(values, block)
    scale = fp8_block_scale(amax)
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
    """A GEMM epilogue returning each of its products rounded to `out_dtype` (one, or a pair's two: a gated
    MLP's gate and up, row 12689) beside its next product's rows quantized as DeepSeek-V3's blockwise FP8
    recipe quantizes them: `activation` of the rounded products (the one product itself where none), rounded
    to `out_dtype`, as E4M3 codes and 1 x 128 power-of-two scales (`_blockwise_fp8_quantized`), bitwise the
    recipe's `projections.rows_quantized` of that activation, which a separate pass formed from the stored
    products (row 8547)."""
    if jnp.dtype(out_dtype).itemsize > 4:  # its quantizer reads fp32 (`_blockwise_fp8_quantized`)
        raise ValueError(f"a quantizing epilogue reads fp32 at most, not {jnp.dtype(out_dtype)}")

    def quantized(*accs):
        rounded = tuple(acc.astype(out_dtype) for acc in accs)
        next_rows = rounded[0] if activation is None else activation(*rounded).astype(out_dtype)
        codes, scales = _blockwise_fp8_quantized(next_rows.astype(jnp.float32))
        return *rounded, codes, scales

    return quantized


def forms_of(out_forms: tuple | None, count: int) -> tuple:
    """Each of a call's `count` outputs' form (`Hooks.out_forms`): None for a tile of its own, else
    `(divisor, transposed)`; none stated, every output a tile."""
    if not out_forms:
        return ()
    if count < 2:
        raise ValueError("output forms are a several-output call's (`out_dtypes`)")
    if len(out_forms) != count:
        raise ValueError(f"{len(out_forms)} output forms for {count} outputs")
    return tuple(out_forms)


def reduced(buffer, piece: jax.Array) -> None:
    """A reduced output's chunk written (`stores.Output.write`): `piece`, a consumer's rows by one block's
    columns (the chunk its staging states, one block), its value broadcast across them by its epilogue, as
    each row's maximum, `[1, rows]` where transposed: a row reduction of the accumulator's layout, which
    Mosaic GPU lowers where a strided read of every `divisor`-th column does not (witness/packing-1, sm90)."""
    buffer[0, :] = jnp.max(piece.astype(jnp.float32), axis=1).astype(buffer.dtype)


def outputs(part: tuple[int, int], dtypes: tuple, forms: tuple) -> tuple:
    """A consumer's share `part` of each output as its family stages it: `(shape, dtype)` for a tile,
    `stores.Output` for one a form reduces (`forms_of`), written by `reduced`."""
    from dew.nn.kernels.templates import stores

    if not forms:
        return tuple((part, jnp.dtype(dtype)) for dtype in dtypes)
    return tuple((part, jnp.dtype(dtype)) if form is None
                 else stores.Output(part, jnp.dtype(dtype), divisor=form[0], write=reduced,
                                    transposed=form[1])
                 for dtype, form in zip(dtypes, forms, strict=True))


def declared(dtype, form: tuple[int, bool], rows: int, columns: int) -> jax.ShapeDtypeStruct:
    """An output a form reduces (`forms_of`) as its call declares it: one value a row each `divisor` of
    `columns`, `[columns // divisor, rows]` where transposed, `[rows, columns // divisor]` otherwise."""
    divisor, transposed = form
    if columns % divisor:
        raise ValueError(f"{columns} columns hold no whole {divisor}-column blocks")
    shape = (columns // divisor, rows) if transposed else (rows, columns // divisor)
    return jax.ShapeDtypeStruct(shape, jnp.dtype(dtype))


def _reduced_divisor(out_forms: tuple | None) -> int | None:
    """The one block width a call's outputs a form reduces state (`Hooks.out_forms`), None for none."""
    divisors = {form[0] for form in out_forms or () if form is not None}
    if len(divisors) > 1:
        raise ValueError(f"a call's reduced outputs state one block width, not {sorted(divisors)}")
    return next(iter(divisors), None)


def formed(config, out_forms: tuple | None):
    """`config` (a GEMM tile) staging its outputs a block of a reduced output's columns at a time where the
    call stores one (`Hooks.out_forms`; two buffers), so each chunk a reduced output's `write` takes is one
    block (`reduced`); `config` as it is elsewhere, or where it states its staging."""
    import dataclasses

    from dew.nn.kernels.templates import stores

    divisor = _reduced_divisor(out_forms)
    staging = None if divisor is None else (divisor, stores.OUTPUT_BUFFERS)
    # A tile whose issuer emits narrower pieces than a block (tcgen05's 64 columns) stages no block a chunk
    # (`Tiles.chunked_refusal`): left unstated, which `forms_refusal` refuses, so its family declines.
    if staging is None or config.staging is not None or config.chunked_refusal(staging) is not None:
        return config
    return dataclasses.replace(config, staging=staging)


def forms_refusal(out_forms: tuple | None, config, *, dense: bool) -> str | None:
    """Why a GEMM launch at `config` stores no output a form reduces (`Hooks.out_forms`), None where it does
    or the call stores none: its staging a block of the form's columns a chunk (`formed`, which `reduced`
    writes), and a transposed one's call dense (its stores take a share's whole rows,
    `stores.transposed_rows`: a grouped tile's share starts mid-tile)."""
    divisor = _reduced_divisor(out_forms)
    if divisor is None:
        return None
    if config.staging is None or config.staging[0] != divisor:
        return f"a reduced output's chunk is one {divisor}-column block, not {config.staging}"
    if not dense and any(form is not None and form[1] for form in out_forms or ()):
        return "a transposed output is a dense call's"
    return None


def stored(value: jax.Array, form: tuple[int, bool] | None) -> jax.Array:
    """`value` (`[rows, columns]`, an epilogue's output broadcast across each block's columns where `form`
    reduces it) as its call stores it (`declared`): itself for a tile; a block's first column each block,
    `[columns // divisor, rows]` where transposed: XLA's form of `reduced`'s store."""
    if form is None:
        return value
    divisor, transposed = form
    blocks = value[:, ::divisor]
    return blocks.T if transposed else blocks


def trimmed(outs: tuple, forms: tuple, rows: int | None, columns: int | None = None) -> tuple:
    """A call's outputs (`declared`'s) to their first `rows` rows and `columns` columns where given, along
    each output's own axes: a transposed reduced output's rows its second axis, its columns its first, a
    `divisor`th of them."""
    def one(out, form):
        if form is None:
            return out[:rows] if columns is None else out[:rows, :columns]
        divisor, transposed = form
        blocks = None if columns is None else -(-columns // divisor)
        return out[:blocks, :rows] if transposed else out[:rows, :blocks]

    return tuple(one(out, form) for out, form in zip(outs, forms or (None,) * len(outs), strict=True))


def typed_outputs(dtypes: tuple, out_forms: tuple | None, config, *, dense: bool) -> tuple[tuple, tuple]:
    """A `gmm` call's outputs' dtypes and forms (`forms_of`), refused where its tile stores none of them
    (`forms_refusal`): a direct call's as a launch's planning refuses it."""
    typed = tuple(jnp.dtype(dtype) for dtype in dtypes)
    forms = forms_of(out_forms, len(typed))
    if (refusal := forms_refusal(forms, config, dense=dense)) is not None:
        raise ValueError(f"{config}: {refusal}")
    return typed, forms


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
