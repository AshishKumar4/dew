"""Read and write the quantized formats checkpoints ship their weights in.

`source_quantization` reads a config's `quantization_config` into one codec
and refuses every other format by name. A codec knows which tensors of a
checkpoint are the parts of one quantized weight, decodes them in float32
one tensor at a time, and encodes dense weights back into those parts for
`Pretrained.save`:

- DeepSeek's block-scaled FP8 (`quant_method: fp8`; V3, V3.2 and the
  finegrained FP8 of Qwen3 and MiniMax): `<m>.weight` in float8_e4m3fn
  beside `<m>.weight_scale_inv`, one float32 scale per square block.
- DeepSeek-V4's storage of that config, which also declares `expert_dtype`:
  `<m>.weight` beside `<m>.scale`, an E8M0 exponent (a float32 power of two
  in the Base releases) per block of an FP8 Linear, per 32 values of a V4.1
  engram table's row, and per 32 inputs of an FP4 routed expert.
- GPT OSS's MXFP4 (`quant_method: mxfp4`): `<stem>_blocks` and `<stem>_scales`.
- compressed-tensors' `mxfp4-pack-quantized`: `<m>.weight_packed` beside
  `<m>.weight_scale`.
- AutoAWQ's gemm packing (`quant_method: awq`) and GPTQ (`quant_method:
  gptq`): int32-packed codes with fp16 scales and packed zeros per group,
  which save back against the source's own scales and zeros.

Every FP4 format here stores the OCP MX element that `decode_e2m1` and
`encode_e2m1` read and write: E2M1 codes two to a byte, the even element in
the low nibble, bit 3 the sign, 32 consecutive inputs under one E8M0
exponent byte b meaning 2 ** (b - 127). The formats differ in where the
bytes sit, in the rule that picks a group's exponent, and in where a value
halfway between two E2M1 values goes.

The arithmetic is NumPy's on a host copy. XLA on CPU reads and writes
float32 subnormals as zero, and these formats keep them: an MXFP4 group of
2 ** -127 weights encodes to the 0.5 code at the 2 ** -126 scale and
decodes back to 2 ** -127, which the same code under jax.numpy returns as
zeros (measured, jax 0.11 CPU, scale bytes 0 and 1).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax.numpy as jnp
import ml_dtypes
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from dew import records
from dew.nn.text_encoders import checkpoint_array

# --------------------------------------------------------------------------
# E2M1 codes under E8M0 exponents, shared by every FP4 format
# --------------------------------------------------------------------------

GROUP = 32
"""Inputs that share one E8M0 exponent in every MX format: 16 packed bytes."""

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
"""The value of each E2M1 code, in code order."""

_E2M1_BYTES = np.stack((E2M1[np.arange(256) & 15], E2M1[np.arange(256) >> 4]), axis=-1)
"""The two values each packed byte holds, low nibble first."""

_E2M1_MIDPOINTS = (E2M1[:7] + E2M1[1:8]) / 2
"""The magnitudes halfway between neighbouring E2M1 values, 0.25 up to 5."""

_CODE_DTYPES = (np.dtype(np.uint8), np.dtype(np.int8))
"""Packed E2M1 pairs: U8 in GPT OSS and compressed-tensors, I8 in DeepSeek-V4."""


def _bytes(array: ArrayLike, dtypes: tuple[np.dtype, ...]) -> np.ndarray:
    """`array`'s bytes as uint8, if it is stored in one of `dtypes`."""
    values = np.asarray(array)
    if values.dtype not in dtypes:
        raise ValueError(f"expected bytes stored as one of {[dtype.name for dtype in dtypes]}, "
                         f"got {values.dtype}")
    return values.view(np.uint8)


def e8m0_scales(exponents: ArrayLike) -> np.ndarray:
    """The float32 scale 2 ** (b - 127) of each E8M0 exponent byte b,
    uint8 or float8_e8m0fnu: exact down to byte 0's subnormal 2 ** -127,
    and NaN for byte 255, as float8_e8m0fnu reads."""
    stored = _bytes(exponents, (np.dtype(np.uint8), np.dtype(ml_dtypes.float8_e8m0fnu)))
    return stored.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)


def decode_e2m1(packed: ArrayLike, exponents: ArrayLike) -> np.ndarray:
    """Packed codes [..., n / 2] under exponent bytes [..., n / 32] to float32 [..., n].

    Element i is E2M1[code i] * 2 ** (exponents[i // 32] - 127), a product
    float32 holds exactly, down to 2 ** -128 at byte 0; code 8 is -0.0.
    Codes arrive as uint8 or int8, exponents as uint8 or float8_e8m0fnu.
    """
    codes, scales = _bytes(packed, _CODE_DTYPES), e8m0_scales(exponents)
    if codes.shape[:-1] != scales.shape[:-1] or codes.shape[-1] != scales.shape[-1] * (GROUP // 2):
        raise ValueError(f"packed E2M1 codes [..., n / 2] take E8M0 exponents [..., n / {GROUP}], "
                         f"got {codes.shape} and {scales.shape}")
    values = _E2M1_BYTES[codes].reshape(*scales.shape, GROUP)
    values *= scales[..., None]
    return values.reshape(*codes.shape[:-1], 2 * codes.shape[-1])


def encode_e2m1(quotients: np.ndarray, *, ties: Literal['even', 'away']) -> np.ndarray:
    """Values already over their group's scale, [..., n], to packed codes [..., n / 2].

    Each rounds to the nearest E2M1 value and saturates at +-6, and a
    negative that rounds to zero keeps its sign (code 8). A value halfway
    between two E2M1 values goes to the even code under 'even', the IEEE
    rounding of ml_dtypes' float4_e2m1fn cast (which has no infinity and
    saturates, measured, ml_dtypes 0.6), and to the larger magnitude under
    'away'. The even element goes in the low nibble.
    """
    if ties == 'even':
        codes = quotients.astype(ml_dtypes.float4_e2m1fn).view(np.uint8)
    else:
        magnitudes = np.searchsorted(_E2M1_MIDPOINTS, np.abs(quotients), side='right')
        codes = magnitudes.astype(np.uint8) | (np.signbit(quotients).astype(np.uint8) << 3)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def _float_groups(weight: ArrayLike, what: str, dtype: DTypeLike | None = None) -> np.ndarray:
    """A float weight, cast to `dtype` if given, as C-ordered [..., groups, 32]
    along its last axis, so the codes and exponents derived from it are
    C-ordered too; refused unless every value is finite in that dtype."""
    values = np.asarray(weight)
    # jnp's dtype lattice counts ml_dtypes' bfloat16 as floating; NumPy's does not.
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise ValueError(f"{what} encodes float weights, got {values.dtype}")
    if values.ndim < 1 or values.shape[-1] % GROUP:
        raise ValueError(f"{what} takes a weight whose input axis is a multiple of the "
                         f"{GROUP}-value group, got {values.shape}")
    values = np.ascontiguousarray(values, dtype)
    if not np.isfinite(values).all():
        raise ValueError(f"{what} holds no infinite or NaN weight: E2M1 encodes neither, and its "
                         "group would take the reserved E8M0 NaN scale 0xff")
    return values.reshape(*values.shape[:-1], values.shape[-1] // GROUP, GROUP)


# --------------------------------------------------------------------------
# GPT OSS: quant_method mxfp4, <stem>_blocks and <stem>_scales
# --------------------------------------------------------------------------

def _mxfp4_arrays(blocks: ArrayLike, scales: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    blocks, scales = np.asarray(blocks), np.asarray(scales)
    if blocks.ndim != 4 or blocks.shape[-1] != GROUP // 2 or blocks.shape[:-1] != scales.shape:
        raise ValueError("MXFP4 blocks must be [expert, output, group, 16] with one scale per group")
    if blocks.dtype != np.uint8 or scales.dtype != np.uint8:
        raise ValueError("MXFP4 blocks and scales must be uint8")
    return blocks, scales


def dequantize_mxfp4(blocks: ArrayLike, scales: ArrayLike) -> np.ndarray:
    """Packed [expert, output, group, 16] blocks and E8M0 scales to float32 [expert, input, output].

    transformers' `convert_moe_packed_tensors`: the low nibble precedes the
    high nibble, and every value is exact in the bf16 it decodes to.
    """
    blocks, scales = _mxfp4_arrays(blocks, scales)
    return decode_e2m1(blocks.reshape(*scales.shape[:2], blocks.shape[2] * blocks.shape[3]), scales).swapaxes(1, 2)


def quantize_mxfp4(weight: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """[expert, input, output] weights to packed [expert, output, group, 16] blocks and E8M0 scales.

    The released encoder: transformers 5.16.1's `quantize_to_mxfp4`
    (integrations/mxfp4.py:231-234) runs `downcast_to_mxfp_torch` of
    kernels-community/gpt-oss-triton-kernels (numerics_details/mxfp.py at
    v1, 0f351046) on the weight rounded to bf16. A group's scale is its
    largest magnitude over 6 rounded up to a power of two on the float32
    bits (an all-zero group takes the 0x00 byte), and each value over that
    scale rounds to the nearest E2M1 value, ties away from zero: the kernel
    adds one to the magnitude's exponent and two leading mantissa bits and
    halves the sum (mxfp.py:220). The round-up keeps every scaled value at
    or under 6, so the saturation never acts. The scale rule is spelled out
    because no cast performs it: float8_e8m0fnu rounds to nearest and sends
    0 to the NaN byte 0xff.
    """
    values = np.asarray(weight)
    if values.ndim != 3 or values.shape[1] % GROUP:
        raise ValueError(
            "MXFP4 takes an [expert, input, output] weight whose input axis is a "
            f"multiple of the {GROUP}-value group, got {values.shape}")
    groups = _float_groups(values.swapaxes(1, 2), "MXFP4", ml_dtypes.bfloat16).astype(np.float32)
    bits = (np.abs(groups).max(-1) / np.float32(6)).view(np.uint32)
    exponents = ((bits + np.uint32(0x007fffff)) >> 23).astype(np.uint8)
    codes = encode_e2m1(groups / e8m0_scales(exponents)[..., None], ties='away')
    return codes.reshape(*exponents.shape, GROUP // 2), exponents


@dataclass(frozen=True)
class SourceQuantization:
    """One quantized storage format: which tensors of a checkpoint hold one
    weight, how that weight decodes and how a dense weight encodes back.

    `names` finds the weights a checkpoint ships quantized, refusing a pair
    that has lost a part; take it before `dequantize`, after which nothing
    says which tensors arrived quantized. `partners(name)` are the stored
    tensors that hold `name` besides a tensor of that name itself.
    `decode(tensors, name)` is its value in float32 and `encode(name,
    weight)` the stored tensors it writes back. `scale_dtype` reads the
    dtype a source stored its scales in where the format leaves that to the
    checkpoint (DeepSeek-V4), for `source_quantization` to write back.
    `grid(name)` names the partners an integer format encodes against rather
    than recomputes (AWQ's and GPTQ's scales and zeros): the loader keeps
    them, and `source_quantization(..., grid=)` hands them back to `encode`.
    """

    names: Callable[[Mapping[str, np.ndarray]], tuple[str, ...]]
    partners: Callable[[str], tuple[str, ...]]
    decode: Callable[[Mapping[str, np.ndarray], str], np.ndarray]
    encode: Callable[[str, np.ndarray], dict[str, np.ndarray]]
    scale_dtype: Callable[[Mapping[str, np.ndarray]], str | None] = lambda tensors: None
    grid: Callable[[str], tuple[str, ...]] = lambda name: ()

    def tensor_names(self, tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
        """The names left once every quantized weight is decoded."""
        names = self.names(tensors)
        stored = {partner for name in names for partner in self.partners(name)}
        return (tuple(name for name in tensors if name not in stored)
                + tuple(name for name in names if name not in tensors))

    def read(self, tensors: Mapping[str, np.ndarray], name: str) -> np.ndarray:
        """One original tensor, a quantized weight decoded in float32 on
        demand, so alias checks never build a float32 copy of the checkpoint."""
        return self.decode(tensors, name) if self.partners(name)[0] in tensors else tensors[name]

    def dequantize(self, tensors: Mapping[str, np.ndarray], *,
                   param_dtype: str = "float32") -> dict[str, np.ndarray]:
        """Decode each quantized weight in float32, keep it in `param_dtype`
        and drop its parts. Every other tensor passes through unchanged."""
        out = dict(tensors)
        for name in self.names(tensors):
            out[name] = checkpoint_array(self.decode(tensors, name), param_dtype)
            for partner in self.partners(name):
                out.pop(partner)
        return out

    def requantize(self, tensors: Mapping[str, np.ndarray], names: Iterable[str]) -> dict[str, np.ndarray]:
        """Write each of `names` back in the format, for the names a source
        shipped quantized (`names`). Every other tensor passes through as
        itself. A name the caller no longer holds is refused rather than
        written dense under a config that calls it quantized, and so is a
        part already among the tensors, which the encoding would overwrite."""
        out = dict(tensors)
        for name in names:
            if name not in out:
                raise ValueError(f"{name} was quantized in the source and is not among the tensors to write")
            taken = [partner for partner in self.partners(name) if partner in out]
            if taken:
                raise ValueError(f"{', '.join(taken)} is already among the tensors to write, so "
                                 f"quantizing {name} would overwrite it")
            out.update(self.encode(name, np.asarray(out.pop(name))))
        return out


def _paired(tensors: Mapping[str, np.ndarray], suffixes: tuple[str, ...], weight: str) -> tuple[str, ...]:
    """The weights stored as `<stem><suffix>` parts, `<stem><weight>` each, sorted.
    Half a pair is refused: the checkpoint has lost a weight."""
    stems = sorted({name.removesuffix(suffix) for name in tensors
                    for suffix in suffixes if name.endswith(suffix)})
    for stem in stems:
        for suffix in suffixes:
            if stem + suffix not in tensors:
                raise ValueError(f"{stem}{weight} arrives quantized and the checkpoint holds no {stem}{suffix}")
    return tuple(stem + weight for stem in stems)


MXFP4_SUFFIXES = ('_blocks', '_scales')
"""GPT OSS's `<stem>_blocks` and `<stem>_scales`."""

MXFP4 = SourceQuantization(
    lambda tensors: _paired(tensors, MXFP4_SUFFIXES, ''),
    lambda name: tuple(name + suffix for suffix in MXFP4_SUFFIXES),
    lambda tensors, name: dequantize_mxfp4(*(tensors[name + suffix] for suffix in MXFP4_SUFFIXES)),
    lambda name, weight: dict(zip((name + suffix for suffix in MXFP4_SUFFIXES), quantize_mxfp4(weight),
                                  strict=True)))
"""GPT OSS's MXFP4 (`quant_method: mxfp4`), [E, in, out] weights under their stems."""


# --------------------------------------------------------------------------
# compressed-tensors: mxfp4-pack-quantized, <m>.weight_packed and <m>.weight_scale
# --------------------------------------------------------------------------

PACKED_MXFP4_WEIGHTS = {'num_bits': 4, 'type': 'float', 'strategy': 'group', 'group_size': GROUP,
                        'symmetric': True, 'dynamic': False, 'scale_dtype': 'torch.uint8',
                        'actorder': None, 'block_structure': None, 'zp_dtype': None}
"""compressed-tensors' MXFP4 weight scheme (`QuantizationArgs` of the one config
group moonshotai/Kimi-K3 declares at f831ab6): the fields that change what the
codes mean, each with the one value the packed layout below reads."""

PACKED_SUFFIXES = ('.weight_packed', '.weight_scale')
"""compressed-tensors' `mxfp4-pack-quantized` pair beside a Linear's module name."""


def packed_mxfp4_format(quantization: Mapping[str, object]) -> None:
    """Refuse a compressed-tensors config whose tensors are not MXFP4 packed weights.

    Weights alone are quantized, as groups of 32 inputs under one E8M0
    exponent; activations and the KV cache stay in the compute dtype.
    """
    if quantization.get('format') != 'mxfp4-pack-quantized':
        raise ValueError(f"compressed-tensors format {quantization.get('format')!r}: this loader reads "
                         "mxfp4-pack-quantized weights and nothing else")
    if quantization.get('quantization_status', 'compressed') != 'compressed':
        raise ValueError("compressed-tensors quantization_status must be 'compressed'")
    if quantization.get('kv_cache_scheme') is not None:
        raise ValueError("compressed-tensors kv_cache_scheme quantizes the cache, which this loader does not")
    groups = quantization.get('config_groups')
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("compressed-tensors config_groups must name the quantized weights")
    for name, group in groups.items():
        group = records.record(group, f'config_groups.{name}')
        if group.get('format', 'mxfp4-pack-quantized') != 'mxfp4-pack-quantized':
            raise ValueError(f"config_groups.{name}.format must be mxfp4-pack-quantized")
        for side in ('input_activations', 'output_activations'):
            if group.get(side) is not None:
                raise ValueError(f"config_groups.{name}.{side} quantizes activations, which this loader does not")
        weights = records.record(group.get('weights'), f'config_groups.{name}.weights')
        wrong = sorted(key for key, value in PACKED_MXFP4_WEIGHTS.items() if weights.get(key, value) != value)
        if wrong:
            raise ValueError(f"config_groups.{name}.weights {wrong} differ from MXFP4's "
                             f"{ {key: PACKED_MXFP4_WEIGHTS[key] for key in wrong} }")


def quantize_packed_mxfp4(weight: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """[..., output, input] weights to compressed-tensors' packed codes
    [..., output, input / 2] and E8M0 exponent bytes [..., output, input / 32].

    compressed-tensors 0.17.1's own encoder, `calculate_qparams` then
    `MXFP4PackedCompressor.compress`, computed as the library computes it,
    in the weight's dtype, per group of 32 inputs:

    - the largest magnitude rounds to a power of two on its bits
      (`round_to_power_2` adds a quarter of the exponent step to the pattern,
      2 ** 21 in float32, and keeps the exponent: up from a mantissa
      fraction of 0.75, down below it);
    - the byte is 127 + log2 of that power - 2, floor(log2 6)
      (`generate_mx_scales`), clamped to [0, 255], so an all-zero group,
      whose log2 is -inf, takes byte 0;
    - the scale is 2 ** (byte - 127) in the weight's dtype. Where that
      underflows to zero, as the smallest groups' scales do in float16,
      the library puts the eps of a uint8 scale dtype, 1, and writes 127;
    - each weight over the scale, plus the symmetric zero point the
      quantization lifecycle keeps until compression, saturates at 6 and
      rounds to the nearest E2M1 value, ties to even. The zero point makes
      a -0.0 quotient code 0; a negative quotient that rounds to zero keeps
      code 8.
    """
    groups = _float_groups(weight, "compressed-tensors MXFP4")
    largest = np.abs(groups).max(-1)
    mantissa = ml_dtypes.finfo(largest.dtype).nmant
    unsigned = np.dtype(f'u{largest.dtype.itemsize}')
    bits = largest.view(unsigned) + unsigned.type(1 << (mantissa - 2))
    power = (bits & ~unsigned.type((1 << mantissa) - 1)).view(largest.dtype)
    if not np.isfinite(power).all():
        raise ValueError("compressed-tensors MXFP4 rounds a group's largest magnitude up to the "
                         "power of two past its dtype's range, whose E8M0 byte is the reserved NaN 0xff")
    with np.errstate(divide='ignore'):
        exponents = np.clip(127 + np.floor(np.log2(power.astype(np.float64))) - 2, 0, 255).astype(np.uint8)
    scales = e8m0_scales(exponents).astype(groups.dtype)
    underflow = scales == 0
    scales[underflow], exponents[underflow] = 1, 127
    codes = encode_e2m1(groups / scales[..., None] + groups.dtype.type(0), ties='even')
    return codes.reshape(*groups.shape[:-2], groups.shape[-2] * GROUP // 2), exponents


def _packed_names(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """The `<module>.weight` names shipped as a compressed-tensors pair; a
    pair beside a dense weight of the same name is refused too."""
    names = _paired(tensors, PACKED_SUFFIXES, '.weight')
    dense = [name for name in names if name in tensors]
    if dense:
        raise ValueError(f"{dense} arrive both dense and MXFP4 packed")
    return names


def _packed_partners(name: str) -> tuple[str, ...]:
    return tuple(name.removesuffix('.weight') + suffix for suffix in PACKED_SUFFIXES)


def _decode_packed(tensors: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """A packed Linear's weight, FP32 `[output, input]`."""
    packed, scales = (tensors[partner] for partner in _packed_partners(name))
    if (packed.dtype != np.uint8 or scales.dtype != np.uint8 or packed.ndim != 2
            or packed.shape[1] % (GROUP // 2)
            or scales.shape != (packed.shape[0], packed.shape[1] // (GROUP // 2))):
        raise ValueError(
            f"{name} packs U8 [output, input / 2] codes beside U8 [output, input / {GROUP}] scales, "
            f"got {packed.dtype} {packed.shape} and {scales.dtype} {scales.shape}")
    return decode_e2m1(packed, scales)


def _encode_packed(name: str, weight: np.ndarray) -> dict[str, np.ndarray]:
    if weight.ndim != 2:
        raise ValueError(f"{name} packs a Linear's [output, input] weight, got {weight.shape}")
    return dict(zip(_packed_partners(name), quantize_packed_mxfp4(weight), strict=True))


PACKED_MXFP4 = SourceQuantization(_packed_names, _packed_partners, _decode_packed, _encode_packed)
"""compressed-tensors' `mxfp4-pack-quantized`, `<module>.weight` under its pair."""


# --------------------------------------------------------------------------
# DeepSeek: quant_method fp8, <m>.weight and <m>.weight_scale_inv
# --------------------------------------------------------------------------

BLOCK = 128
"""DeepSeek's weight_block_size, [128, 128]."""

SCALE_SUFFIX = '_scale_inv'
"""`layer.weight` scales by `layer.weight_scale_inv`."""

E4M3 = ml_dtypes.float8_e4m3fn
"""safetensors' F8_E4M3."""

E4M3_MAX = 448.0
"""E4M3FN's largest finite value, which a block's amax is scaled to."""

AMAX_FLOOR = 1e-4
"""`per_block_cast_to_fp8`'s `clamp(1e-4)` on a block's amax."""


E4M3_NAMES = (None, 'e4m3', 'float8_e4m3fn')
"""The `fmt` spellings of E4M3 weights. transformers' `FineGrainedFP8Config`
(utils/quantization_config.py:1692-1742, 5.16.1) declares no `fmt` and
always stores float8_e4m3fn, so Qwen3's FP8 repos that omit it are E4M3;
DeepSeek writes 'e4m3' and MiniMax 'float8_e4m3fn'."""


def fp8_format(quantization: Mapping[str, object]) -> tuple[int, bool]:
    """Return the block size and whether the scales are ue8m0, from `quantization`.

    The finegrained format or refused: E4M3 weights (`E4M3_NAMES`) in a
    square block, the scales float32 (`scale_fmt` absent or 'float', V3)
    or ue8m0 (V3.2). A per-tensor or rectangular scale, or a scale format
    with no rounding rule here, is not this format.
    """
    fmt, block, scale_fmt = (quantization.get(key)
                             for key in ('fmt', 'weight_block_size', 'scale_fmt'))
    if (fmt not in E4M3_NAMES or not isinstance(block, list) or len(block) != 2
            or any(type(side) is not int or side < 1 for side in block)
            or block[0] != block[1] or scale_fmt not in (None, 'float', 'ue8m0')
            or quantization.get('weight_per_tensor')):
        raise ValueError(
            f"quantization_config names fp8 with fmt {fmt!r}, weight_block_size "
            f"{block!r} and scale_fmt {scale_fmt!r}; this loader reads e4m3 weights "
            f"in square blocks, not per tensor, with float32 or ue8m0 scales and nothing else")
    return block[0], scale_fmt == 'ue8m0'


def dequantize_fp8_blocks(weight: np.ndarray, scale_inv: np.ndarray,
                          block: int = BLOCK) -> np.ndarray:
    """Return float32(weight[i, j]) * float32(scale_inv[i // block, j // block]).

    `weight` may be in any float dtype, `scale_inv` float32 or E8M0.
    """
    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(
            f"block-scaled dequantization takes a [rows, cols] weight and its "
            f"[row blocks, col blocks] scales, got shapes {weight.shape} and "
            f"{scale_inv.shape}")
    rows, cols = weight.shape
    blocks = (math.ceil(rows / block), math.ceil(cols / block))
    if scale_inv.shape != blocks:
        raise ValueError(
            f"a {weight.shape} weight in {block} x {block} blocks takes a "
            f"{blocks} scale, got {scale_inv.shape}")
    # Scaled one block row at a time, so no scale array of the weight's size
    # is ever built.
    out = weight.astype(np.float32)
    scales = scale_inv.astype(np.float32)
    for index in range(blocks[0]):
        out[index * block:(index + 1) * block] *= np.repeat(scales[index], block)[:cols]
    return out


def _fp8_scale_inv(amax: np.ndarray, ue8m0: bool) -> np.ndarray:
    """`per_block_cast_to_fp8`'s scale of a block from its float32 amax."""
    scale_inv = np.maximum(amax, np.float32(AMAX_FLOOR)) / np.float32(E4M3_MAX)
    return np.exp2(np.ceil(np.log2(scale_inv))) if ue8m0 else scale_inv


def _finite_matrix(weight: ArrayLike, rows: int, cols: int) -> np.ndarray:
    """A weight's host float32 copy, refused unless it is a finite matrix;
    its blocks are `rows` x `cols`."""
    values = np.asarray(weight, np.float32)
    if values.ndim != 2:
        raise ValueError(
            f"block-scaled quantization takes a [rows, cols] weight, got shape "
            f"{values.shape}")
    if type(cols) is not int or cols < 1:
        raise ValueError(
            f"a block covers a positive number of rows and columns, got {cols!r}")
    if not np.isfinite(values).all():
        raise ValueError(
            f"a {values.shape} weight holds "
            f"{int(np.count_nonzero(~np.isfinite(values)))} value(s) that are not "
            f"finite; E4M3FN encodes no infinity, and one of them carries its "
            f"whole {rows} x {cols} block's scale away with it")
    return values


def quantize_fp8_blocks(weight: ArrayLike, block: int = BLOCK, *,
                        ue8m0: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Cast `weight` to float8_e4m3fn blocks and the float32 scales that invert them.

    The inverse of `dequantize_fp8_blocks`: a [rows, cols] weight in any float
    dtype, on any device (widened to a host float32 copy exactly first),
    gives a float8_e4m3fn array of its shape and a [ceil(rows / block),
    ceil(cols / block)] `scale_inv`. `ue8m0` rounds the scales up to powers
    of two, as V3.2's config declares.

    This is DeepGEMM's `per_block_cast_to_fp8` (deep_gemm/utils/math.py),
    the cast that produces weights in this format, ported operation for
    operation so the bytes agree with it exactly:

        amax = max(|x|) over the block, in float32, clamped up to 1e-4
        scale_inv = amax / 448
        scale_inv = 2 ** ceil(log2(scale_inv))    # ue8m0 only, float32 log2
        q = float8_e4m3fn(x * (1 / scale_inv))

    Not `act_quant` in DeepSeek's inference code, the activation rule: it
    reduces over 1 x 128 tiles and divides where the weight cast multiplies
    by a reciprocal. The partial blocks the reference zero-pads are reduced
    in place here, which zeros and the 1e-4 floor make the same reduction.

    The cast needs no clamp. A float32 scale is amax / 448 to within a few
    roundings, and a ue8m0 scale sits at most 2 ** -21 relative under the
    quotient it rounds (float32 log2 lands on the integer for a quotient that
    close above a power of two; measured against torch in tests), so a scaled
    element stays under 448.001. E4M3FN rounds everything up to 464 to its
    448, and that margin is where the libraries part: ml_dtypes sends a value
    above it to NaN where torch saturates. A weight that is not finite is
    refused by name rather than written as a block of NaN. The 1e-4 floor
    keeps every scale a float32 normal.
    """
    values = _finite_matrix(weight, block, block)
    rows, cols = values.shape
    row_starts, col_starts = np.arange(0, rows, block), np.arange(0, cols, block)
    # reduceat's last segment in each dimension is the partial block, over
    # just its own elements.
    amax = np.maximum.reduceat(
        np.maximum.reduceat(np.abs(values), row_starts, axis=0), col_starts, axis=1)
    scale_inv = _fp8_scale_inv(amax, ue8m0)
    out = np.empty(values.shape, E4M3)
    for index, start in enumerate(row_starts):
        # One reciprocal per block, as the reference takes it, repeated across
        # the block row.
        scales = np.repeat(np.float32(1.0) / scale_inv[index], block)[:cols]
        out[start:start + block] = (values[start:start + block] * scales).astype(E4M3)
    return out, scale_inv


def scaled_names(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """The names in `tensors` that have a `<name>_scale_inv` partner, in
    order. A scale whose weight is absent is refused."""
    names = tuple(name.removesuffix(SCALE_SUFFIX) for name in tensors if name.endswith(SCALE_SUFFIX))
    for name in names:
        if name not in tensors:
            raise ValueError(f"{name}{SCALE_SUFFIX} scales {name}, which the checkpoint does not hold")
    return names


def fp8_blocks(block: int = BLOCK, *, ue8m0: bool = False) -> SourceQuantization:
    """DeepSeek's FP8 blocks of `block` x `block` under `_scale_inv`
    partners, encoded back with ue8m0 scales where the source's are."""
    return SourceQuantization(
        scaled_names, lambda name: (name + SCALE_SUFFIX,),
        lambda tensors, name: dequantize_fp8_blocks(tensors[name], tensors[name + SCALE_SUFFIX], block),
        lambda name, weight: dict(zip((name, name + SCALE_SUFFIX),
                                      quantize_fp8_blocks(weight, block, ue8m0=ue8m0), strict=True)))


# --------------------------------------------------------------------------
# DeepSeek-V4: quant_method fp8 beside expert_dtype, <m>.weight and <m>.scale
# --------------------------------------------------------------------------

V4_SCALE_SUFFIX = '.scale'
"""DeepSeek-V4 scales `<m>.weight` by `<m>.scale`."""

V4_SCALE_DTYPES = ('float8_e8m0fnu', 'float32')
"""The dtypes a DeepSeek-V4 FP8 weight's `.scale` ships in: E8M0 exponents
in V4-Flash, V4-Pro and V4.1-Flash, float32 powers of two in V4-Flash-Base
and V4-Pro-Base. The config says neither, so the loader records which."""

_V4_EXPERT = re.compile(r'\.experts\.\d+\.w[123]\.weight$')
"""A routed expert's projection, `layers.N.ffn.experts.E.w1` or under `mtp.N`."""


def deepseek_v4_layout(name: str, fp4_experts: bool) -> Literal['blocks', 'rows', 'fp4']:
    """How the `.scale` beside `name` covers it, as inference/model.py builds
    the module: one per 32 inputs of an FP4 `Linear` for a routed expert
    under expert_dtype 'fp4', one per `block` values of a row of
    `ParallelEngramEmbedding`'s table, one per block of an FP8 `Linear`
    otherwise."""
    if fp4_experts and _V4_EXPERT.search(name):
        return 'fp4'
    return 'rows' if name.endswith('.engram.embed.weight') else 'blocks'


def _v4_names(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """The `<m>.weight` names a DeepSeek-V4 checkpoint ships beside a `<m>.scale`, in order.

    A `.scale` whose weight is absent is refused: the checkpoint has lost it.
    """
    names = []
    for name in tensors:
        if name.endswith(V4_SCALE_SUFFIX):
            weight = name.removesuffix(V4_SCALE_SUFFIX) + '.weight'
            if weight not in tensors:
                raise ValueError(f"{name} scales {weight}, which the checkpoint does not hold")
            names.append(weight)
    return tuple(names)


def _v4_partners(name: str) -> tuple[str, ...]:
    return (name.removesuffix('.weight') + V4_SCALE_SUFFIX,)


def _v4_scale_dtype(tensors: Mapping[str, np.ndarray]) -> str | None:
    """The dtype a DeepSeek-V4 checkpoint stores its `.scale` tensors in, or
    None without any. Scales in more than one dtype are refused."""
    stored = sorted({tensors[_v4_partners(name)[0]].dtype.name for name in _v4_names(tensors)})
    if len(stored) > 1:
        raise ValueError(f"a DeepSeek-V4 checkpoint stores its `.scale` tensors in one dtype, "
                         f"this one in {stored}")
    return stored[0] if stored else None


def _decode_v4(tensors: Mapping[str, np.ndarray], name: str, *, block: int,
               fp4_experts: bool) -> np.ndarray:
    """A scaled weight decoded in FP32.

    The release's own dequantization: an FP8 Linear is float32(weight) *
    float32(scale) per block (convert.py on `wo_a`), an engram row is
    float32(weight) * float32(scale) per `block` values (model.py,
    `ParallelEngramEmbedding.forward`), and an FP4 expert is
    FP4_TABLE[code] * float32(scale) per 32 inputs with element 2i in the
    low nibble (convert.py, `cast_e2m1fn_to_e4m3fn`). The one difference is
    code 8, which FP4_TABLE reads as 0.0 and this reads as -0.0, so that a
    decoded weight encodes back to the same byte.
    """
    value = tensors[name]
    partner, layout = _v4_partners(name)[0], deepseek_v4_layout(name, fp4_experts)
    scale = tensors[partner]
    if layout == 'fp4':
        if (value.dtype not in _CODE_DTYPES or value.ndim != 2 or scale.dtype != ml_dtypes.float8_e8m0fnu
                or scale.shape != (value.shape[0], 2 * value.shape[1] // GROUP) or 2 * value.shape[1] % GROUP):
            raise ValueError(
                f"{name} is a routed expert, which expert_dtype 'fp4' ships as int8 [out, in / 2] E2M1 "
                f"pairs beside a float8_e8m0fnu [out, in / {GROUP}] {partner}, got {value.dtype} "
                f"{value.shape} and {scale.dtype} {scale.shape}")
        return decode_e2m1(value, scale)
    if value.dtype != E4M3 or value.ndim != 2 or scale.dtype.name not in V4_SCALE_DTYPES:
        raise ValueError(
            f"{name} and {partner} are an FP8 weight, float8_e4m3fn beside a scale in one of "
            f"{list(V4_SCALE_DTYPES)}, got {value.dtype} {value.shape} and {scale.dtype}; DeepSeek-V4 "
            f"ships E2M1 pairs for routed experts alone, under expert_dtype 'fp4'")
    if layout == 'blocks':
        return dequantize_fp8_blocks(value, scale, block)
    if value.shape[1] % block or scale.shape != (value.shape[0], value.shape[1] // block):
        raise ValueError(f"{name} is an engram table, [rows, dim] beside a [rows, dim / {block}] "
                         f"{partner}, got {value.shape} and {scale.shape}")
    rows = value.astype(np.float32).reshape(value.shape[0], value.shape[1] // block, block)
    rows *= scale.astype(np.float32)[..., None]
    return rows.reshape(value.shape)


def quantize_deepseek_v4_fp4(weight: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """[..., output, input] weights to DeepSeek-V4's int8 E2M1 pairs
    [..., output, input / 2] and float8_e8m0fnu scales [..., output, input / 32].

    The release's FP4 rule, `fp4_quant_kernel` in inference/kernel.py under
    E8M0 scales, per group of 32 inputs: the weight rounds to the bf16 the
    kernel reads; amax is floored at 6 * 2 ** -126; the scale is 2 **
    ceil(log2(amax * float32(1 / 6))) on the float32 bits of that product
    (`fast_round_scale`: the exponent, plus one if any mantissa bit is
    set); each value over the scale saturates at 6 and rounds to the
    nearest E2M1 value, ties to even. An all-zero group takes byte 1.
    """
    groups = _float_groups(weight, "DeepSeek-V4 FP4", ml_dtypes.bfloat16).astype(np.float32)
    amax = np.maximum(np.abs(groups).max(-1), np.float32(6 * 2.0 ** -126))
    bits = (amax * np.float32(1 / 6)).view(np.uint32)
    exponents = ((bits >> 23) + ((bits & 0x007fffff) != 0)).astype(np.uint8)
    codes = encode_e2m1(groups / e8m0_scales(exponents)[..., None], ties='even')
    return (codes.reshape(*groups.shape[:-2], groups.shape[-2] * GROUP // 2).view(np.int8),
            exponents.view(ml_dtypes.float8_e8m0fnu))


def quantize_fp8_rows(weight: ArrayLike, group: int) -> tuple[np.ndarray, np.ndarray]:
    """Cast a [rows, cols] weight to float8_e4m3fn with one ue8m0 scale per
    `group` consecutive values of a row, [rows, cols / group] float32.

    `quantize_fp8_blocks`' rule with ue8m0 over 1 x `group` blocks, the
    layout V4.1's engram tables ship in.
    """
    values = _finite_matrix(weight, 1, group)
    if values.shape[1] % group:
        raise ValueError(f"row groups of {group} take a weight whose width is a multiple of "
                         f"{group}, got {values.shape}")
    groups = values.reshape(values.shape[0], values.shape[1] // group, group)
    scale_inv = _fp8_scale_inv(np.abs(groups).max(-1), ue8m0=True)
    codes = (groups * (np.float32(1.0) / scale_inv)[..., None]).astype(E4M3)
    return codes.reshape(values.shape), scale_inv


def _encode_v4(name: str, weight: np.ndarray, *, block: int, fp4_experts: bool,
               scale_dtype: str | None) -> dict[str, np.ndarray]:
    """`name` written back as a DeepSeek-V4 weight and `.scale`, in the
    layout `deepseek_v4_layout` gives it. An FP8 weight takes
    `quantize_fp8_blocks` or `quantize_fp8_rows` under ue8m0, its scale
    stored in `scale_dtype`, the dtype the source stored its scales in (E8M0
    when None). An FP4 expert takes `quantize_deepseek_v4_fp4`, whose scales
    are E8M0."""
    partner, layout = _v4_partners(name)[0], deepseek_v4_layout(name, fp4_experts)
    if layout == 'fp4':
        return dict(zip((name, partner), quantize_deepseek_v4_fp4(weight), strict=True))
    codes, scale_inv = (quantize_fp8_rows(weight, block) if layout == 'rows'
                        else quantize_fp8_blocks(weight, block, ue8m0=True))
    return {name: codes, partner: scale_inv.astype(np.dtype(scale_dtype or V4_SCALE_DTYPES[0]))}


def deepseek_v4(block: int, *, fp4_experts: bool, scale_dtype: str | None = None) -> SourceQuantization:
    """DeepSeek-V4's `.scale` storage at `block`, with routed experts as E2M1
    pairs under `fp4_experts`, written back with scales in `scale_dtype`."""
    return SourceQuantization(
        _v4_names, _v4_partners, partial(_decode_v4, block=block, fp4_experts=fp4_experts),
        partial(_encode_v4, block=block, fp4_experts=fp4_experts, scale_dtype=scale_dtype), _v4_scale_dtype)


# --------------------------------------------------------------------------
# Integer groups: AWQ (quant_method awq) and GPTQ (quant_method gptq)
# --------------------------------------------------------------------------
#
# Both store a Linear's [in, out] weight as `bits`-bit codes packed into int32
# words, with fp16 scales and packed zeros per group of `group_size` inputs:
# w = (code - zero) * scale. The product is taken in fp16, as both libraries'
# torch dequantizers take it (autoawq 0.2.9 awq/utils/packing_utils.py
# `dequantize_gemm`, gptqmodel 7.5 nn_modules/qlinear `dequantize_weight`):
# the float32 product of a small integer and an fp16 scale is exact, so one
# rounding to fp16 is theirs. A trained weight encodes back against the
# source's own scales and zeros, `round((w + zero * scale) / scale)`, both
# libraries' pack rule given scales and zeros (autoawq `WQLinear_GEMM.from_linear`,
# gptqmodel `pack_block`). A weight whose code leaves [0, 2**bits) is refused:
# AutoAWQ 0.2.9 does not clamp it, so it would spill into the neighbouring
# nibbles (gemm.py:196-206), and gptqmodel 7.5 clamps it to the grid's edge
# (`int_block.clamp_(0, maxq)`, qlinear/__init__.py:1284), which saves a
# different weight than the one trained.
#
# Re-encoding against the source grid keeps only changes of half a grid step
# or more: a lightly trained model saves back mostly as its source. A
# min/max grid puts codes on 0 and 2**bits - 1 by construction, so a weight
# at an edge that trains outward past half a step is refused, and a
# substantially trained model is refused as a whole; it saves dense.

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
"""The column each nibble of an AWQ gemm word holds, nibble 0 in the low bits."""

GPTQ_SUFFIXES = ('.qweight', '.qzeros', '.scales', '.g_idx')
AWQ_SUFFIXES = ('.qweight', '.qzeros', '.scales')


def _words(values: np.ndarray, bits: int) -> np.ndarray:
    """Unsigned codes [..., n] packed low bits first into int32 words [..., n * bits / 32]."""
    per = 32 // bits
    grouped = values.astype(np.uint32).reshape(*values.shape[:-1], -1, per)
    shifts = np.arange(per, dtype=np.uint32) * bits
    return np.bitwise_or.reduce(grouped << shifts, axis=-1).view(np.int32)


def _codes(words: np.ndarray, bits: int) -> np.ndarray:
    """int32 words [..., m] to their unsigned codes [..., m * 32 / bits], low bits first."""
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    codes = (words.astype(np.int32).view(np.uint32)[..., None] >> shifts) & ((1 << bits) - 1)
    return codes.reshape(*words.shape[:-1], -1).astype(np.int32)


def _fp16_product(codes: np.ndarray, zeros: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return ((codes - zeros).astype(np.float32) * scales.astype(np.float32)).astype(np.float16).astype(np.float32)


def _encoded_codes(name: str, weight: np.ndarray, zeros: np.ndarray, scales: np.ndarray, bits: int) -> np.ndarray:
    """`round((w + zero * scale) / scale)` for [in, out] `weight` against per-input
    zeros and scales, refusing codes the packing cannot hold."""
    w, s = weight.astype(np.float64), scales.astype(np.float64)
    codes = np.round((w + zeros * s) / s)
    outside = int(np.count_nonzero((codes < 0) | (codes >= 1 << bits)))
    if outside:
        raise ValueError(
            f"{name}: {outside} trained values fall outside the source's {bits}-bit grid, which the "
            "libraries' packing would spill or clamp; save dense instead, with dataclasses.replace(pretrained, "
            "config={k: v for k, v in pretrained.config.items() if k != 'quantization_config'}, "
            "quantized_tensors=()).save(directory)")
    return codes.astype(np.int32)


def _gridded(grid: Mapping[str, np.ndarray] | None, name: str, parts: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    if grid is None or any(part not in grid for part in parts):
        raise ValueError(f"{name} encodes against the scales and zeros it was loaded with, which this "
                         "codec was not given; save it through the Pretrained that loaded it")
    return tuple(np.asarray(grid[part]) for part in parts)


def _awq_decode(tensors: Mapping[str, np.ndarray], name: str, *, bits: int, group: int) -> np.ndarray:
    stem = name.removesuffix('.weight')
    order = np.argsort(AWQ_ORDER)

    def columns(words: np.ndarray) -> np.ndarray:
        codes = _codes(words, bits)
        return codes.reshape(*codes.shape[:-1], -1, 32 // bits)[..., order].reshape(codes.shape)

    codes, zeros = columns(tensors[stem + '.qweight']), columns(tensors[stem + '.qzeros'])
    scales = np.asarray(tensors[stem + '.scales'])
    rows = np.arange(codes.shape[0]) // group
    return _fp16_product(codes, zeros[rows], scales[rows]).T


def _awq_encode(name: str, weight: np.ndarray, *, bits: int, group: int,
                grid: Mapping[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    stem = name.removesuffix('.weight')
    parts = (stem + '.qzeros', stem + '.scales')
    packed_zeros, scales = _gridded(grid, name, parts)
    order = np.argsort(AWQ_ORDER)
    zeros = _codes(packed_zeros, bits)
    zeros = zeros.reshape(*zeros.shape[:-1], -1, 32 // bits)[..., order].reshape(zeros.shape)
    rows = np.arange(weight.shape[1]) // group
    codes = _encoded_codes(name, np.asarray(weight).T, zeros[rows], scales[rows], bits)
    packed = codes.reshape(codes.shape[0], -1, 32 // bits)[..., list(AWQ_ORDER)].reshape(codes.shape)
    return {stem + '.qweight': _words(packed, bits), stem + '.qzeros': packed_zeros, stem + '.scales': scales}


def awq(bits: int, group: int, grid: Mapping[str, np.ndarray] | None = None) -> SourceQuantization:
    """AutoAWQ's gemm packing: `<m>.qweight` int32 [in, out * bits / 32] in
    `AWQ_ORDER`, `<m>.qzeros` likewise per group, `<m>.scales` fp16 [in / group, out]."""
    return SourceQuantization(
        lambda tensors: _paired(tensors, AWQ_SUFFIXES, '.weight'),
        lambda name: tuple(name.removesuffix('.weight') + suffix for suffix in AWQ_SUFFIXES),
        partial(_awq_decode, bits=bits, group=group),
        partial(_awq_encode, bits=bits, group=group, grid=grid),
        grid=lambda name: tuple(name.removesuffix('.weight') + suffix for suffix in AWQ_SUFFIXES[1:]))


_V1_ZERO_OFFSET = {2: 0x55555555, 4: 0x11111111, 8: 0x01010101}
"""What gptqmodel adds to each packed zero word of a v1 ('gptq') checkpoint:
one per code, as an integer add over the word (utils/model.py
`convert_gptq_v1_to_v2_format_module`), carries included."""


def _gptq_zeros(packed: np.ndarray, bits: int, v1: bool) -> np.ndarray:
    words = packed.astype(np.int64)
    if v1:
        words = (words + _V1_ZERO_OFFSET[bits]) & 0xFFFFFFFF
    return _codes(words.astype(np.uint32).view(np.int32), bits)


def _gptq_decode(tensors: Mapping[str, np.ndarray], name: str, *, bits: int, v1: bool) -> np.ndarray:
    stem = name.removesuffix('.weight')
    codes = _codes(np.asarray(tensors[stem + '.qweight']).T, bits).T
    zeros = _gptq_zeros(np.asarray(tensors[stem + '.qzeros']), bits, v1)
    groups = np.asarray(tensors[stem + '.g_idx']).astype(np.int64)
    return _fp16_product(codes, zeros[groups], np.asarray(tensors[stem + '.scales'])[groups]).T


def _gptq_encode(name: str, weight: np.ndarray, *, bits: int, v1: bool,
                 grid: Mapping[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    stem = name.removesuffix('.weight')
    packed_zeros, scales, groups = _gridded(grid, name, (stem + '.qzeros', stem + '.scales', stem + '.g_idx'))
    rows = groups.astype(np.int64)
    zeros = _gptq_zeros(packed_zeros, bits, v1)
    codes = _encoded_codes(name, np.asarray(weight).T, zeros[rows], scales[rows], bits)
    return {stem + '.qweight': _words(codes.T, bits).T, stem + '.qzeros': packed_zeros,
            stem + '.scales': scales, stem + '.g_idx': groups}


def gptq(bits: int, *, v1: bool, grid: Mapping[str, np.ndarray] | None = None) -> SourceQuantization:
    """GPTQ's packing: `<m>.qweight` int32 [in * bits / 32, out] along inputs,
    `<m>.qzeros` int32 [groups, out * bits / 32] (stored one below the zero in
    a v1 checkpoint), `<m>.scales` fp16 [groups, out] and `<m>.g_idx` [in],
    the group of each input, which act-order permutes."""
    return SourceQuantization(
        lambda tensors: _paired(tensors, GPTQ_SUFFIXES, '.weight'),
        lambda name: tuple(name.removesuffix('.weight') + suffix for suffix in GPTQ_SUFFIXES),
        partial(_gptq_decode, bits=bits, v1=v1),
        partial(_gptq_encode, bits=bits, v1=v1, grid=grid),
        grid=lambda name: tuple(name.removesuffix('.weight') + suffix for suffix in GPTQ_SUFFIXES[1:]))


def _integer_format(quantization: Mapping[str, object], method: str) -> tuple[int, int]:
    """The `bits` and `group_size` an AWQ or GPTQ config declares, refusing what
    this loader does not decode."""
    if 'bits' not in quantization:
        raise ValueError(f"{method} quantization_config has no bits, the code width its weights are packed "
                         "at; the checkpoint's config.json is incomplete")
    bits, group = records.integer(quantization['bits'], f'{method} bits'), quantization.get('group_size')
    if method == 'awq':
        if quantization.get('version', 'gemm') != 'gemm' or quantization.get('zero_point', True) is not True:
            raise ValueError(f"awq version {quantization.get('version')!r} with zero_point "
                             f"{quantization.get('zero_point')!r}: this loader reads gemm weights with zero "
                             "points and nothing else")
        if bits != 4:
            raise ValueError(f"awq bits {bits!r}: gemm packs 4-bit weights")
    else:
        if bits not in (2, 4, 8):
            raise ValueError(f"gptq bits {bits!r}: this loader reads 2-, 4- and 8-bit packing")
        if quantization.get('checkpoint_format', 'gptq') not in ('gptq', 'gptq_v2'):
            raise ValueError(f"gptq checkpoint_format {quantization.get('checkpoint_format')!r}: this loader "
                             "reads the gptq and gptq_v2 formats")
    # GPTQ's decode reads each input's group from g_idx, so any group_size
    # holds, -1 (one group per row) included; AWQ's reads it from group_size.
    if method == 'awq' and (not isinstance(group, int) or isinstance(group, bool) or group <= 0):
        raise ValueError(f"awq group_size {group!r}: a positive group of inputs per scale")
    if not isinstance(group, int) or isinstance(group, bool):
        raise ValueError(f"{method} group_size {group!r}: an integer")
    return bits, group


# --------------------------------------------------------------------------
# Dispatch from quantization_config
# --------------------------------------------------------------------------

def _refuse_mlx(config: Mapping[str, object]) -> None:
    """Refuse MLX quantization by name.

    mlx-lm writes its affine group quantization as `quantization` (and, in
    older conversions, the same record as `quantization_config`) with
    `group_size` and `bits` and no `quant_method`, over MLX's own tensor
    names (`.scales`, `.biases`).
    """
    for key in ("quantization", "quantization_config"):
        entry = config.get(key)
        if isinstance(entry, Mapping) and "quant_method" not in entry and {"bits", "group_size"} <= entry.keys():
            raise ValueError(
                f"{key} {dict(entry)!r} is MLX quantization ({entry['bits']}-bit weights in groups "
                f"of {entry['group_size']}), which Dew does not dequantize; load the unquantized "
                "safetensors repo it was converted from (the model card's base_model)")


def source_quantization(config: Mapping[str, object], *, scale_dtype: str | None = None,
                        grid: Mapping[str, np.ndarray] | None = None) -> SourceQuantization | None:
    """Return the format a config's `quantization_config` declares, or None.

    A wrapper may declare it on its text_config alone: KimiK3Config lifts
    `text_config.quantization_config` onto itself (configuration_kimi_k3.py:282-283).
    An fp8 config that declares `expert_dtype`, in quantization_config or
    beside it, is DeepSeek-V4's `.scale` storage; `scale_dtype` is the dtype
    the source stored those scales in, which the codec writes back, and
    `grid` the scales and zeros an integer format encodes against
    (`SourceQuantization.grid`).
    """
    _refuse_mlx(config)
    quantization = config.get("quantization_config")
    text = config.get("text_config")
    if quantization is None and isinstance(text, Mapping):
        quantization = text.get("quantization_config")
    if quantization is None:
        return None
    if not isinstance(quantization, Mapping):
        raise ValueError(f"quantization_config must be an object, got {quantization!r}")
    method = quantization.get("quant_method")
    if method == "fp8" and ("expert_dtype" in quantization or "expert_dtype" in config):
        # `fp8_format`'s blocks under ue8m0 scales, the only kind an E8M0
        # `.scale` holds. expert_dtype 'fp4' ships routed experts as E2M1
        # pairs (inference/model.py `Linear` under float4_e2m1fn_x2), 'fp8' or
        # null as FP8 blocks like every other Linear (the Base releases).
        block, ue8m0 = fp8_format(quantization)
        experts = quantization.get("expert_dtype", config.get("expert_dtype"))
        if not ue8m0:
            raise ValueError(
                f"quantization_config declares expert_dtype, DeepSeek-V4's `.scale` storage, with scale_fmt "
                f"{quantization.get('scale_fmt')!r}; those scales are powers of two, scale_fmt 'ue8m0'")
        if experts not in ("fp4", "fp8", None):
            raise ValueError(f"expert_dtype {experts!r}: DeepSeek-V4 ships routed experts as fp4 E2M1 "
                             f"pairs or as fp8 blocks and nothing else")
        return deepseek_v4(block, fp4_experts=experts == "fp4", scale_dtype=scale_dtype)
    if method == "fp8":
        block, ue8m0 = fp8_format(quantization)
        return fp8_blocks(block, ue8m0=ue8m0)
    if method == "mxfp4":
        return MXFP4
    if method == "compressed-tensors":
        packed_mxfp4_format(quantization)
        return PACKED_MXFP4
    if method == "awq":
        bits, group = _integer_format(quantization, method)
        return awq(bits, group, grid)
    if method == "gptq":
        bits, _ = _integer_format(quantization, method)
        return gptq(bits, v1=quantization.get("checkpoint_format", "gptq") == "gptq", grid=grid)
    raise ValueError(
        f"quantization_config names quant_method {method!r}; this loader reads DeepSeek's "
        f"fp8 blocks and V4 `.scale` storage, GPT OSS's mxfp4, compressed-tensors' "
        f"mxfp4-pack-quantized, AutoAWQ's gemm and GPTQ and nothing else")
