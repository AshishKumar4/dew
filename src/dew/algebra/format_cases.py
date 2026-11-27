"""Printable components of Dew's numeric formats, held off the runtime path (`dew.algebra.generate`,
docs/design/algebra.md §6.11): FP4, FP8 and E8M0 codes encoded and decoded, MX and NVFP4 block scales read
and applied, the FP8, MXFP8 and NVFP4 scales and epilogues, a blockwise product's partial scaling, and the
rounded operand's and the quantizer's straight-through surrogates."""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra import bits
from dew.algebra.generate import (
    Cases,
    ComponentForm,
    SurrogateBodies,
    SurrogateCase,
    declare,
    float32_cases,
    normal_operands,
)
from dew.nn import fake_quant, precision
from dew.nn.kernels import formats
from dew.nn.kernels.templates import product
from dew.nn.kernels.templates.gemm import quantized


def _rounded(x, *, dtype):
    return precision._rounded(x, dtype)


def _rounded_truth(x, *, dtype):
    return x.astype(dtype).astype(x.dtype)


def _tangent(x, dx, value, *, dtype):
    return precision._rounded_tangent(dtype, (x,), (dx,), value)


def _tangent_truth(x, dx, value):
    return dx


def _rounding_case(name, dtype):
    return SurrogateCase(name, ComponentForm(
        functools.partial(_rounded, dtype=dtype), functools.partial(_rounded_truth, dtype=dtype),
        float32_cases(((16, 32),), ((8, 64),))), {(): ComponentForm(
            functools.partial(_tangent, dtype=dtype), _tangent_truth,
            float32_cases(*[((m, n),) * 3 for m, n in ((16, 32), (8, 64))]))},
                         output_tree=(jax.tree.structure(0),))


ROUNDED = SurrogateBodies(precision._rounded, precision._rounded_tangent, (1,), symbolic_zeros=False, cases={
    (jnp.bfloat16,): _rounding_case("rounded", jnp.bfloat16),
    (jnp.float16,): _rounding_case("rounded_fp16", jnp.float16),
}).declared(__name__)


def _gemm_scaled_truth(partial, rows, columns):
    return partial * rows[:, None] * columns[None, :]


def _gemm_scaled(partial, rows, columns):
    return precision.scaled_partial(partial, rows[:, None], columns[None, :])


def _gemm_scale_draw(seed, operands):
    partial, rows, columns = normal_operands(seed, operands)
    rows, columns = np.abs(rows), np.abs(columns)
    # Exercise row-first and column-first overflow avoidance as well as ordinary scales.
    if seed == 1:
        rows[0], columns[0], partial[0, 0] = 2.0 ** 127, 2.0 ** -80, 4
        partial[0, 1:] = 0
    elif seed == 2:
        columns[0], rows[0], partial[0, 0] = 2.0 ** 127, 2.0 ** -80, 4
        partial[1:, 0] = 0
    return partial, rows, columns


def _fp8_epilogue_truth(acc):
    # The recipe first stores its rounded activation, then scales each 128 columns independently.
    stored = acc.astype(jnp.bfloat16).astype(jnp.float64)
    blocks = stored.reshape(acc.shape[0], -1, 128)
    maximum = jnp.maximum(jnp.max(jnp.abs(blocks), axis=-1, keepdims=True), 1e-4)
    fraction, exponent = jnp.frexp(maximum)
    # The FP8 ceiling 448 is (7/8) × 2^9. Compare the exact fractions at a scale boundary.
    power = jnp.ldexp(jnp.ones_like(maximum), exponent - 9 + (fraction > 0.875))
    encoded = jnp.clip(blocks / power, -448, 448).astype(jnp.float8_e4m3fn).astype(jnp.float64)
    return stored, encoded.reshape(acc.shape), jnp.broadcast_to(power, blocks.shape).reshape(acc.shape)


def _fp8_epilogue_draw(seed, operands):
    (acc,) = normal_operands(seed, operands)
    acc *= np.float32(4 ** (seed % 4))
    acc[0, :128] = 0
    acc[1, :4] = [448, -448, 224, -224]
    return (acc,)


GEMM_PARTIAL_SCALE = declare(__name__, "gemm_partial_scale", ComponentForm(
    _gemm_scaled, _gemm_scaled_truth, dataclasses.replace(
        float32_cases(*[((m, n), (m,), (n,)) for m, n in ((32, 128), (64, 128))]), draw=_gemm_scale_draw)))


FP8_EPILOGUE = declare(__name__, "fp8_epilogue", ComponentForm(
    quantized.blockwise_fp8_epilogue(), _fp8_epilogue_truth,
    dataclasses.replace(float32_cases(((32, 128),), ((64, 256),)), draw=_fp8_epilogue_draw)))


def _mxfp8_scale(raw):
    return formats.mxfp8_scale_bytes(jax.lax.bitcast_convert_type(raw, jnp.float32))


def _mx_scale_truth(value):
    _, e = jnp.frexp(value)
    exponent = jnp.clip(jnp.where(value == 0, -127, e - 9), -127, 127)
    finite = jnp.isfinite(value)
    return jnp.where(finite, exponent, 0), jnp.where(finite, exponent + 127, 255).astype(jnp.uint32)


def _mxfp8_scale_truth(raw):
    return _mx_scale_truth(jax.lax.bitcast_convert_type(raw, jnp.float32).astype(jnp.float64))


def _mx_scale_draw(seed, operands):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2 ** 31, operands[0].shape, dtype=np.uint32)
    raw.flat[:11] = [0, 1, 0x007fffff, 0x00800000, 0x047fffff, 0x04800000,
                     0x3f7fffff, 0x3f800000, 0x7f7fffff, 0x7f800000, 0x7fc00001]
    return (raw,)


def _mxfp8_epilogue(acc):
    codes, scales = quantized.quantize_mxfp8_epilogue()(acc)
    return jax.lax.bitcast_convert_type(codes, jnp.uint8), jax.lax.bitcast_convert_type(scales, jnp.uint8)


def _mxfp8_epilogue_truth(acc):
    blocks = acc.reshape(acc.shape[0], -1, 32)
    exponent, scale = _mx_scale_truth(jnp.max(jnp.abs(blocks), axis=-1, keepdims=True))
    narrowed = jnp.clip(jnp.ldexp(blocks, -exponent), -448, 448).astype(jnp.float8_e4m3fn)
    codes = jax.lax.bitcast_convert_type(narrowed, jnp.uint8)
    scale_codes = jnp.broadcast_to(scale, blocks.shape).astype(jnp.uint8)
    return codes.reshape(acc.shape), scale_codes.reshape(acc.shape)


def _mxfp8_draw(seed, operands):
    (acc,) = normal_operands(seed, operands)
    acc *= np.float32(2.0 ** (seed * 20 - 80))
    acc[0, :32] = 0
    acc[1, :8] = [448, -448, 224, -224, 256, -256, 1, -0.0]
    return (acc,)


def _nvfp4_scale(codes):
    return jax.lax.bitcast_convert_type(bits.scales_decoded(codes), jnp.uint32)


def _nvfp4_scale_truth(codes):
    magnitude = codes.astype(jnp.uint32) & 127
    field, fraction = magnitude >> 3, magnitude & 7
    significand = fraction.astype(jnp.float64) * 0.125 + (field > 0)
    exact = jnp.ldexp(significand, jnp.maximum(field, 1).astype(jnp.int32) - 7)
    rounded = jax.lax.bitcast_convert_type(exact.astype(jnp.float32), jnp.uint32)
    return jnp.where(magnitude == 127, jnp.uint32(0x7fc00000), rounded) | (
        (codes.astype(jnp.uint32) & 0x80) << 24)


def _nvfp4_scale_draw(seed, operands):
    rows, columns = operands[0].shape
    codes = np.arange(rows * columns, dtype=np.uint32).astype(np.uint8)
    return (np.roll(codes, seed).reshape(rows, columns),)


def _nvfp4_payload(codes):
    return bits.nvfp4_codes(bits.packed(codes.T, bits.E2M1).T)


def _nvfp4_payload_truth(codes):
    values = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], jnp.float64)[codes & 7]
    return jnp.where(codes & 8 != 0, -values, values)


def _nvfp4_payload_draw(seed, operands):
    rng = np.random.default_rng(seed)
    (value,) = operands
    codes = rng.integers(0, 16, value.shape, dtype=np.uint32)
    codes[0, :16] = np.arange(16, dtype=np.uint32)
    return (codes,)


MXFP8_SCALE = declare(__name__, "mxfp8_scale", ComponentForm(_mxfp8_scale, _mxfp8_scale_truth, Cases(
    tuple((jax.ShapeDtypeStruct(shape, jnp.uint32),) for shape in ((32, 8), (64, 8))),
    _mx_scale_draw, seeds=8, policies=("default",))))


MXFP8_EPILOGUE = declare(__name__, "mxfp8_epilogue", ComponentForm(_mxfp8_epilogue, _mxfp8_epilogue_truth,
    dataclasses.replace(float32_cases(((32, 128),), ((64, 128),)), draw=_mxfp8_draw)))


NVFP4_SCALE = declare(__name__, "nvfp4_scale", ComponentForm(_nvfp4_scale, _nvfp4_scale_truth, Cases(
    tuple((jax.ShapeDtypeStruct(shape, jnp.uint8),) for shape in ((32, 8), (64, 8))),
    _nvfp4_scale_draw, seeds=8, policies=("default",))))


NVFP4_PAYLOAD = declare(__name__, "nvfp4_payload", ComponentForm(_nvfp4_payload, _nvfp4_payload_truth, Cases(
    tuple((jax.ShapeDtypeStruct(shape, jnp.uint32),) for shape in ((16, 32), (32, 64))),
    _nvfp4_payload_draw, seeds=8, policies=("default",))))


def _ste_truth(x, rounded):
    return rounded.astype(jnp.float64)


def _ste_flat_tangent(x, rounded, dx, dr, value):
    return fake_quant.rounded_tangent((x, rounded), (dx, dr), value)


def _ste_tangent_truth(x, rounded, dx, dr, value):
    return dx


def _ste_draw(seed, operands):
    x, rounded, *rest = normal_operands(seed, operands)
    x[0, :3] = [1, -3, 4096]
    rounded[0, :3] = [-0.0, 0, -1]
    return x, rounded, *rest


def _ste_cases(reverse):
    tile_cases = []
    for shape in ((16, 32), (8, 64)):
        x = jax.ShapeDtypeStruct(shape, jnp.float32)
        rounded = jax.ShapeDtypeStruct(shape, jnp.bfloat16)
        tile_cases.append((x, rounded, x, rounded, x) if reverse else (x, rounded))
    return Cases(tuple(tile_cases), _ste_draw, seeds=8, policies=("default",))


def _minifloat_levels(fmt):
    codes = np.arange(fmt.largest + 1)
    exponent, fraction = np.divmod(codes, 1 << fmt.mantissa)
    significand = (exponent != 0) + fraction * 2.0 ** -fmt.mantissa
    return np.ldexp(significand, np.maximum(exponent, 1) - fmt.bias)


def _encode_truth(raw, *, fmt):
    levels = _minifloat_levels(fmt)
    middle = jnp.asarray((levels[:-1] + levels[1:]) * 0.5)
    magnitude = jnp.abs(jax.lax.bitcast_convert_type(raw, jnp.float32).astype(jnp.float64))
    code = jnp.searchsorted(middle, magnitude, side="left").astype(jnp.uint32)
    tied = (code & 1 != 0) & (magnitude == middle[jnp.minimum(code, len(middle) - 1)])
    code += tied.astype(jnp.uint32)
    nan = fmt.largest if fmt.nan is None else fmt.nan
    sign = raw >> jnp.uint32(31)
    return jnp.where(jnp.isnan(magnitude), jnp.uint32(nan), code) | (sign << (fmt.width - 1))


def _encode_draw(seed, operands, *, fmt):
    levels = _minifloat_levels(fmt)
    middle = ((levels[:-1] + levels[1:]) * 0.5).astype(np.float32)
    boundary = np.concatenate((middle, np.nextafter(middle, -np.inf, dtype=np.float32),
                               np.nextafter(middle, np.inf, dtype=np.float32), levels.astype(np.float32)))
    words = boundary.view(np.uint32)
    edge = np.asarray([0, 1, 0x007fffff, 0x00800000, 0x7f7fffff, 0x7f800000, 0x7f800001, 0x7fc00001],
                      np.uint32)
    words = np.concatenate((edge, words))
    words = np.concatenate((words, words | np.uint32(0x80000000)))
    rng = np.random.default_rng(seed)
    shape = operands[0].shape
    value = rng.integers(0, 2 ** 32, size=shape, dtype=np.uint32)
    value.ravel()[:len(words)] = words
    rng.shuffle(value.ravel())
    return (value,)


def _encoded_words(raw, *, fmt):
    (codes,) = bits.encode_form(fmt).traced(raw)
    return codes


def _encode_case(name, fmt):
    cases = Cases(tuple((jax.ShapeDtypeStruct(shape, jnp.uint32),) for shape in ((16, 128), (32, 128))),
                  functools.partial(_encode_draw, fmt=fmt), seeds=8, policies=("default",))
    return declare(__name__, name, ComponentForm(functools.partial(_encoded_words, fmt=fmt),
                                                 functools.partial(_encode_truth, fmt=fmt), cases))


def _e8m0_bits(codes):
    return jax.lax.bitcast_convert_type(bits.e8m0_decoded(codes), jnp.uint32)


def _e8m0_truth(codes):
    regular = codes.astype(jnp.uint32) << jnp.uint32(23)
    return jnp.select((codes == 0, codes == 255), (jnp.uint32(0x00400000), jnp.uint32(0x7fc00000)), regular)


QUANTIZED_STE = SurrogateBodies(fake_quant._rounded_value, fake_quant.rounded_tangent, (),
                                symbolic_zeros=False, cases={
    (): SurrogateCase("quantized_ste", ComponentForm(
        fake_quant._rounded_value, _ste_truth, _ste_cases(reverse=False)),
                      {(): ComponentForm(_ste_flat_tangent, _ste_tangent_truth, _ste_cases(reverse=True))},
                      output_tree=(jax.tree.structure(0),)),
}).declared(__name__)


ENCODE_E4M3 = _encode_case("encode_e4m3", bits.E4M3)


ENCODE_E5M2 = _encode_case("encode_e5m2", bits.E5M2)


ENCODE_E2M1 = _encode_case("encode_e2m1", bits.E2M1)


E8M0_DECODE = declare(__name__, "e8m0_decode", ComponentForm(_e8m0_bits, _e8m0_truth, Cases(
    tuple((jax.ShapeDtypeStruct(shape, jnp.uint8),) for shape in ((32, 8), (64, 8))),
    _nvfp4_scale_draw, seeds=8, policies=("default",))))


def _decoded_words(codes, *, fmt):
    value = bits.decoded(codes.astype(jnp.uint32), fmt)
    return jax.lax.bitcast_convert_type(value, jnp.uint32)


def _decode_case(name, fmt, host_dtype):
    # Use the host format, not the portable decoder or the device conversion under test.
    codes = np.arange(256, dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        words = codes.view(host_dtype).astype(np.float32).view(np.uint32)
    # The portable read specifies one quiet payload for each NaN sign, not the cast payload.
    nan = np.isnan(codes.view(host_dtype))
    words[nan] = np.uint32(0x7fc00000) | ((codes[nan].astype(np.uint32) & 0x80) << 24)

    def truth(codes):
        return jnp.asarray(words)[codes]

    cases = Cases(tuple((jax.ShapeDtypeStruct(shape, jnp.uint8),) for shape in ((32, 8), (64, 8))),
                  _nvfp4_scale_draw, seeds=8, policies=("default",))
    return declare(__name__, name, ComponentForm(functools.partial(_decoded_words, fmt=fmt), truth, cases))


DECODE_E4M3 = _decode_case("decode_e4m3", bits.E4M3, jnp.float8_e4m3fn)


DECODE_E5M2 = _decode_case("decode_e5m2", bits.E5M2, jnp.float8_e5m2)


def _e8m0_exponents(raw):
    return formats.e8m0_exponents(jax.lax.bitcast_convert_type(raw, jnp.float8_e8m0fnu))


# The host's decodes (ml_dtypes), neither XLA's conversion nor the portable reader under test.
_E8M0 = np.arange(256, dtype=np.uint8).view(jnp.float8_e8m0fnu).astype(np.float64)


_E4M3 = np.arange(256, dtype=np.uint8).view(jnp.float8_e4m3fn).astype(np.float64)


def _e8m0_exponents_truth(raw):
    power = jnp.asarray(_E8M0)[raw]
    # A byte's power's own binary exponent; the NaN byte flagged, at the binade past the largest, 2^128's.
    nan = jnp.isnan(power)
    return jnp.where(nan, 128, jnp.frexp(jnp.where(nan, 1, power))[1] - 1), nan


def _mxfp8_block_decode(codes, raw, *, dtype):
    """`MxFp8.decoded`'s portable body over stored bytes: each block's E8M0 exponent read and repeated over
    its 32 codes, then applied with one destination rounding (`formats._mxfp8_decoded`)."""
    exponent, invalid = _e8m0_exponents(raw)
    exponent, invalid = (jnp.repeat(x, product.MXFP8.depth, axis=-1) for x in (exponent, invalid))
    value = formats._mxfp8_decoded(jnp.dtype(dtype), jax.lax.bitcast_convert_type(codes, jnp.float8_e4m3fn),
                                   exponent, invalid)
    return jax.lax.bitcast_convert_type(value, jnp.dtype(f"uint{jnp.finfo(dtype).bits}"))


def _mxfp8_block_decode_truth(codes, raw, *, dtype):
    limits = jnp.finfo(dtype)
    word = jnp.dtype(f"uint{limits.bits}")
    exact = jnp.ldexp(jnp.asarray(_E4M3)[codes],
                      jnp.repeat(raw.astype(jnp.int32) - 127, product.MXFP8.depth, axis=-1))
    # Four significant bits: exact at every normal of either format; below them rounded to nearest even
    # on the least subnormal's grid, past the largest binade an infinity.
    least = 2.0 ** (limits.minexp - limits.nmant)
    landed = jnp.where(jnp.abs(exact) < 2.0 ** limits.minexp, jnp.round(exact / least) * least, exact)
    landed = jnp.where(jnp.abs(exact) >= 2.0 ** limits.maxexp, jnp.copysign(jnp.inf, exact), landed)
    bits = jax.lax.bitcast_convert_type(landed.astype(dtype), word)
    # OCP MX: a NaN code keeps its sign at a valid scale; the NaN scale byte is a positive quiet NaN.
    unscaled = raw == 255
    held = jnp.repeat(unscaled, product.MXFP8.depth, axis=-1)
    nan = ((codes & 0x7F) == 0x7F) | held
    negative = (codes >= 0x80) & ~held
    quiet = word.type((((1 << limits.nexp) - 1) << limits.nmant) | (1 << (limits.nmant - 1)))
    return jnp.where(nan, quiet | (negative.astype(word) << word.type(limits.bits - 1)), bits)


def _mx_block_draw(seed, operands):
    codes, raw = operands
    rng = np.random.default_rng(seed)
    # All 256 codes and all 256 scale bytes in each draw, the codes shuffled across the blocks by seed.
    held = np.resize(np.arange(256, dtype=np.uint8), codes.shape[0] * codes.shape[1])
    rng.shuffle(held)
    scales = np.roll(np.resize(np.arange(256, dtype=np.uint8), raw.shape[0] * raw.shape[1]), 37 * seed)
    return held.reshape(codes.shape), scales.reshape(raw.shape)


def _mx_block_case(name, dtype):
    cases = Cases(tuple((jax.ShapeDtypeStruct((rows, product.MXFP8.depth * blocks), jnp.uint8),
                         jax.ShapeDtypeStruct((rows, blocks), jnp.uint8))
                        for rows, blocks in ((32, 8), (64, 4))),
                  _mx_block_draw, seeds=8, policies=("default",))
    return declare(__name__, name, ComponentForm(functools.partial(_mxfp8_block_decode, dtype=dtype),
                                                 functools.partial(_mxfp8_block_decode_truth, dtype=dtype),
                                                 cases))


E8M0_EXPONENTS = declare(__name__, "e8m0_exponents", ComponentForm(
    _e8m0_exponents, _e8m0_exponents_truth, Cases(
        tuple((jax.ShapeDtypeStruct(shape, jnp.uint8),) for shape in ((32, 8), (64, 8))),
        _nvfp4_scale_draw, seeds=8, policies=("default",))))


MXFP8_BLOCK_DECODE = _mx_block_case("mxfp8_block_decode", jnp.float32)


MXFP8_BLOCK_DECODE_BF16 = _mx_block_case("mxfp8_block_decode_bf16", jnp.bfloat16)


# E2M1's eight magnitudes, by a code's low three bits, its fourth the sign (OCP MX's table).
_E2M1 = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _nvfp4_block_decode(codes, raw):
    """`bits.nvfp4_decoded` over a tile's codes packed as the quantizer stores them (`bits.packed`) and its
    E4M3 block-scale bytes: each code times its block's scale over 2^8, in bfloat16."""
    words = bits.packed(codes.T, bits.E2M1).T
    return bits.nvfp4_decoded(words, jax.lax.bitcast_convert_type(raw, jnp.float8_e4m3fn))


def _nvfp4_block_decode_truth(codes, raw):
    values = jnp.asarray(_E2M1)[codes & 7]
    signed = jnp.where(codes & 8 != 0, -values, values)
    scales = jnp.asarray(_E4M3)[raw] / 256
    return signed * jnp.repeat(scales, product.NVFP4.depth, axis=0)


def _nvfp4_block_draw(seed, operands):
    codes, raw = operands
    rng = np.random.default_rng(seed)
    held = np.resize(np.arange(16, dtype=np.uint32), codes.shape[0] * codes.shape[1])
    rng.shuffle(held)
    # Every finite E4M3 scale byte, both signs and the subnormals; the NaN bytes no quantizer writes.
    finite = np.asarray([code for code in range(256) if code & 0x7F != 0x7F], np.uint8)
    scales = np.roll(np.resize(finite, raw.shape[0] * raw.shape[1]), 29 * seed)
    return held.reshape(codes.shape), scales.reshape(raw.shape)


NVFP4_BLOCK_DECODE = declare(__name__, "nvfp4_block_decode", ComponentForm(
    _nvfp4_block_decode, _nvfp4_block_decode_truth, Cases(
        tuple((jax.ShapeDtypeStruct((depth, columns), jnp.uint32),
               jax.ShapeDtypeStruct((depth // product.NVFP4.depth, columns), jnp.uint8))
              for depth, columns in ((64, 64), (128, 32))),
        _nvfp4_block_draw, seeds=8, policies=("default",))))


