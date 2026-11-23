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
