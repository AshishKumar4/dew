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
    """`values` (`[rows, cols]`, `cols % SCALE_DEPTH == 0`, the bf16 values a recipe quantizes, as fp32) as
    E4M3 codes a 1 x `SCALE_DEPTH` tile at a time and each tile's power-of-two scale (fp32, broadcast over
    its tile's columns), bitwise `projections.rows_quantized`'s (`fake_quant.fp8_scaled_blocks`: amax floored
    at 1e-4, the scale the ceiling power of two of amax / 448; codes E4M3 by `bits.encode_form`, ties to even,
    saturated). Its scales are normal powers of two for every finite value, so the scaled value is the
    product by the scale's reciprocal, exact, where a quotient would need `lowered_divide` (DeepSeek-V3's
    blockwise FP8 recipe, row 8547)."""
    from dew.nn.fake_quant import E4M3_MAX

    amax = jnp.maximum(_block_max(values, product_formats.SCALE_DEPTH), jnp.float32(1e-4))
    word = jax.lax.bitcast_convert_type(amax * jnp.float32(1 / E4M3_MAX), jnp.uint32)
    # 2 ** ceil(log2(amax / 448)) off its bits (`fake_quant._power_of_two_ceil`): its biased exponent, one
    # more where any fraction bit is set; within the normal range for every finite amax past the floor.
    biased = ((word >> jnp.uint32(23)) & jnp.uint32(0xFF)) + ((word & jnp.uint32(0x7FFFFF)) != 0).astype(
        jnp.uint32)
    scale = jax.lax.bitcast_convert_type(biased << jnp.uint32(23), jnp.float32)
    inverse = jax.lax.bitcast_convert_type((jnp.uint32(254) - biased) << jnp.uint32(23), jnp.float32)
    return encode_tile(values * inverse, bits.E4M3), scale


def blockwise_fp8_epilogue(epilogue=None, out_dtype=jnp.bfloat16):
    """A GEMM epilogue returning the product's value in `out_dtype` (through `epilogue` where given: a
    gated MLP's SwiGLU of its two accumulators) beside that value quantized as DeepSeek-V3's blockwise FP8
    recipe quantizes its next product's rows (`_blockwise_fp8_quantized`: E4M3 codes and 1 x 128 power-of-two
    scales of the value rounded to `out_dtype`), so the next product's operand comes out of this one's
    epilogue, bitwise the recipe's `projections.rows_quantized` of the stored value (row 8547)."""
    def quantized(*tiles):
        value = tiles[0] if epilogue is None else epilogue(*tiles)
        rounded = value.astype(out_dtype)
        codes, scales = _blockwise_fp8_quantized(rounded.astype(jnp.float32))
        return rounded, codes, scales

    return quantized


def forms_of(out_forms: tuple | None, count: int) -> tuple:
    """Each of a call's `count` outputs' form (`Hooks.out_forms`): None for a tile of its own, else
    `(divisor, transposed)`; none stated, every output a tile."""
    if not out_forms:
        return ()
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
    if divisor is None or config.staging is not None:
        return config
    return dataclasses.replace(config, staging=(divisor, stores.OUTPUT_BUFFERS))


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
    if not dense and any(form is not None and form[1] for form in out_forms):
        return "a transposed output is a dense call's"
    return None
