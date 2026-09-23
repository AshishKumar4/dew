"""Read and write the quantized formats checkpoints ship their weights in.

`source_quantization` reads a config's `quantization_config` into one codec
and refuses every other format by name. A codec knows which tensors of a
checkpoint are the parts of one quantized weight, decodes them in float32,
a whole tensor or the block-aligned part of one that an index asks for, and
encodes dense weights back into those parts for `Pretrained.save`:

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

Every FP4 format here stores the OCP MX element that `decode_e2m1` and
`encode_e2m1` read and write: E2M1 codes two to a byte, the even element in
the low nibble, bit 3 the sign, 32 consecutive inputs under one E8M0
exponent byte b meaning 2 ** (b - 127). The formats differ in where the
bytes sit and in the rule that picks a group's exponent.

The arithmetic is NumPy's on a host copy. XLA on CPU reads and writes
float32 subnormals as zero, and these formats keep them: an MXFP4 group of
2 ** -127 weights encodes to the 0.5 code at the 2 ** -126 scale and
decodes back to 2 ** -127, which the same code under jax.numpy returns as
zeros (measured, jax 0.11 CPU, scale bytes 0 and 1).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Literal, Protocol

import jax.numpy as jnp
import ml_dtypes
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from dew import records
from dew.nn.text_encoders import checkpoint_array

Index = tuple[slice, ...]
"""A region of a decoded tensor, one slice per leading axis, as
`jax.make_array_from_callback` asks for a shard."""


def _hull(index: Index | None, shape: tuple[int, ...],
          units: tuple[int, ...]) -> tuple[Index, Index, Index]:
    """Split `index` over a decoded `shape` into the unit-aligned region
    that covers it: in units, in elements, and `index` within that region.

    A unit is the run of elements one stored scale covers along an axis (a
    block, a group, or 1), so the region decodes on its own and reads only
    the bytes it covers. A partial last unit stays partial.
    """
    parts = () if index is None else index
    if len(parts) > len(shape):
        raise IndexError(f"a {len(parts)}-axis index into a {len(shape)}-axis tensor")
    covered, span, within = [], [], []
    for axis, (size, unit) in enumerate(zip(shape, units, strict=True)):
        chosen = range(*(parts[axis] if axis < len(parts) else slice(None)).indices(size))
        if not chosen:
            covered.append(slice(0, 0))
            span.append(slice(0, 0))
            within.append(slice(0, 0))
            continue
        first, last = min(chosen) // unit, -(-(max(chosen) + 1) // unit)
        base = first * unit
        covered.append(slice(first, last))
        span.append(slice(base, min(last * unit, size)))
        # A descending slice whose stop falls before the region runs to its start.
        stop = chosen.stop - base
        within.append(slice(chosen.start - base, stop if stop >= 0 else None, chosen.step))
    return tuple(covered), tuple(span), tuple(within)


# --------------------------------------------------------------------------
# E2M1 codes under E8M0 exponents, shared by every FP4 format
# --------------------------------------------------------------------------

GROUP = 32
"""Inputs that share one E8M0 exponent in every MX format: 16 packed bytes."""

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
"""The value of each E2M1 code, in code order."""

_E2M1_BYTES = np.stack((E2M1[np.arange(256) & 15], E2M1[np.arange(256) >> 4]), axis=-1)
"""The two values each packed byte holds, low nibble first."""

_CODE_DTYPES = (np.dtype(np.uint8), np.dtype(np.int8))
"""Packed E2M1 pairs: U8 in GPT OSS and compressed-tensors, I8 in DeepSeek-V4."""

_EXPONENT_DTYPES = (np.dtype(np.uint8), np.dtype(ml_dtypes.float8_e8m0fnu))
"""E8M0 exponent bytes: U8, or F8_E8M0 in DeepSeek-V4."""


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
    return _bytes(exponents, _EXPONENT_DTYPES).view(ml_dtypes.float8_e8m0fnu).astype(np.float32)


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


def encode_e2m1(quotients: np.ndarray) -> np.ndarray:
    """Values already over their group's scale, [..., n], to packed codes [..., n / 2].

    Each saturates at +-6 and rounds to the nearest E2M1 value, ties to
    even; a negative that rounds to zero keeps its sign (code 8). The even
    element goes in the low nibble.
    """
    codes = np.clip(quotients, -6, 6).astype(ml_dtypes.float4_e2m1fn).view(np.uint8)
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

    The released encoder, transformers 5.16.1's `quantize_to_mxfp4` over
    triton_kernels' `downcast_to_mxfp(..., ROUND_UP)`: the weight rounds to
    bf16, a group's scale is its largest magnitude over 6 rounded up to a
    power of two on the float32 bits (an all-zero group takes the 0x00
    byte), and each value over that scale rounds to the nearest E2M1 value,
    ties to even. The round-up keeps every scaled value at or under 6, so
    the saturation never acts. The scale rule is spelled out because no
    cast performs it: float8_e8m0fnu rounds to nearest and sends 0 to the
    NaN byte 0xff.
    """
    values = np.asarray(weight)
    if values.ndim != 3 or values.shape[1] % GROUP:
        raise ValueError(
            "MXFP4 takes an [expert, input, output] weight whose input axis is a "
            f"multiple of the {GROUP}-value group, got {values.shape}")
    groups = _float_groups(values.swapaxes(1, 2), "MXFP4", ml_dtypes.bfloat16).astype(np.float32)
    bits = (np.abs(groups).max(-1) / np.float32(6)).view(np.uint32)
    exponents = ((bits + np.uint32(0x007fffff)) >> 23).astype(np.uint8)
    codes = encode_e2m1(groups / e8m0_scales(exponents)[..., None])
    return codes.reshape(*exponents.shape, GROUP // 2), exponents


def mxfp4_stems(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """The names a checkpoint ships as an MXFP4 `<stem>_blocks`/`<stem>_scales` pair, sorted.

    Taken before `unpack_mxfp4`, after which nothing says which tensors
    arrived packed. Half a pair is refused: the checkpoint has lost a weight.
    """
    stems = sorted({name.removesuffix(suffix) for name in tensors
                    for suffix in ('_blocks', '_scales') if name.endswith(suffix)})
    for stem in stems:
        for suffix in ('_blocks', '_scales'):
            if stem + suffix not in tensors:
                raise ValueError(
                    f"{stem} arrives MXFP4 packed and the checkpoint holds no {stem}{suffix}")
    return tuple(stems)


def mxfp4_tensor_names(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Decoded names, using the codec's validated pair discovery."""
    stems = mxfp4_stems(tensors)
    return tuple(name for name in tensors if not name.endswith(('_blocks', '_scales'))) + tuple(
        stem for stem in stems if stem not in tensors)


def read_mxfp4_tensor(tensors: Mapping[str, np.ndarray], name: str,
                      index: Index | None = None) -> np.ndarray:
    """One original tensor, or the region `index` names; a packed weight
    decodes in FP32 [expert, input, output] from the groups the region covers."""
    if name + '_blocks' not in tensors:
        return tensors[name] if index is None else tensors[name][index]
    blocks, scales = _mxfp4_arrays(tensors[name + '_blocks'], tensors[name + '_scales'])
    experts, outputs, groups = scales.shape
    (expert, group, output), _, within = _hull(index, (experts, groups * GROUP, outputs), (1, GROUP, 1))
    return dequantize_mxfp4(blocks[expert, output, group], scales[expert, output, group])[within]


def unpack_mxfp4(tensors: Mapping[str, np.ndarray], *,
                 param_dtype: str = "float32") -> dict[str, np.ndarray]:
    """Decode each packed pair in FP32, then retain that weight in param_dtype.
    Biases and every other unpaired tensor remain untouched.
    """
    unpacked = dict(tensors)
    for stem in mxfp4_stems(tensors):
        unpacked[stem] = checkpoint_array(read_mxfp4_tensor(tensors, stem), param_dtype)
        unpacked.pop(stem + '_blocks')
        unpacked.pop(stem + '_scales')
    return unpacked


def pack_mxfp4(tensors: Mapping[str, np.ndarray],
               stems: Collection[str]) -> dict[str, np.ndarray]:
    """Replace each named `<stem>` with the `<stem>_blocks` and `<stem>_scales` it encodes to.

    Only the stems a source shipped packed (`mxfp4_stems`): every other
    tensor is written back as itself. A named stem the tensors no longer
    hold is refused, since the config would still promise its blocks.
    """
    packed = dict(tensors)
    for stem in stems:
        if stem not in packed:
            raise ValueError(
                f"{stem} arrived MXFP4 packed and is not among the tensors to write back")
        packed[f'{stem}_blocks'], packed[f'{stem}_scales'] = quantize_mxfp4(packed.pop(stem))
    return packed


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
    codes = encode_e2m1(groups / scales[..., None] + groups.dtype.type(0))
    return codes.reshape(*groups.shape[:-2], groups.shape[-2] * GROUP // 2), exponents


def packed_mxfp4_stems(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """The `<module>.weight` names a checkpoint ships as a compressed-tensors
    MXFP4 pair, sorted. Half a pair, or a pair beside a dense weight of the
    same name, is refused: the checkpoint would hold one weight twice or not at all."""
    modules = sorted({name.removesuffix(suffix) for name in tensors
                      for suffix in PACKED_SUFFIXES if name.endswith(suffix)})
    for module in modules:
        for suffix in PACKED_SUFFIXES:
            if module + suffix not in tensors:
                raise ValueError(f"{module}.weight arrives MXFP4 packed and the checkpoint holds no {module}{suffix}")
        if module + '.weight' in tensors:
            raise ValueError(f"{module}.weight arrives both dense and MXFP4 packed")
    return tuple(module + '.weight' for module in modules)


def packed_mxfp4_tensor_names(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Decoded names: every unpaired tensor, then each packed Linear's weight."""
    stems = packed_mxfp4_stems(tensors)
    return tuple(name for name in tensors if not name.endswith(PACKED_SUFFIXES)) + stems


def _read_e2m1(packed: np.ndarray, exponents: np.ndarray, index: Index | None) -> np.ndarray:
    """The region `index` names of a [rows, input] weight stored as
    [rows, input / 2] codes under [rows, input / 32] exponents."""
    covered, span, within = _hull(index, (packed.shape[0], 2 * packed.shape[1]), (1, GROUP))
    codes = packed[span[0], span[1].start // 2:span[1].stop // 2]
    return decode_e2m1(codes, exponents[covered])[within]


def read_packed_mxfp4_tensor(tensors: Mapping[str, np.ndarray], name: str,
                             index: Index | None = None) -> np.ndarray:
    """One original tensor, or the region `index` names; a packed Linear's
    weight decodes to FP32 `[output, input]` from the groups the region covers."""
    module = name.removesuffix('.weight')
    if name == module or module + PACKED_SUFFIXES[0] not in tensors:
        return tensors[name] if index is None else tensors[name][index]
    packed, scales = tensors[module + PACKED_SUFFIXES[0]], tensors[module + PACKED_SUFFIXES[1]]
    if (packed.dtype != np.uint8 or scales.dtype != np.uint8 or packed.ndim != 2
            or packed.shape[1] % (GROUP // 2)
            or scales.shape != (packed.shape[0], packed.shape[1] // (GROUP // 2))):
        raise ValueError(
            f"{name} packs U8 [output, input / 2] codes beside U8 [output, input / {GROUP}] scales, "
            f"got {packed.dtype} {packed.shape} and {scales.dtype} {scales.shape}")
    return _read_e2m1(packed, scales, index)


def unpack_packed_mxfp4(tensors: Mapping[str, np.ndarray], *,
                        param_dtype: str = "float32") -> dict[str, np.ndarray]:
    """Decode each compressed-tensors pair in FP32 into its Linear's `.weight`,
    retained in param_dtype. Every other tensor remains untouched."""
    unpacked = dict(tensors)
    for stem in packed_mxfp4_stems(tensors):
        unpacked[stem] = checkpoint_array(read_packed_mxfp4_tensor(tensors, stem), param_dtype)
        for suffix in PACKED_SUFFIXES:
            unpacked.pop(stem.removesuffix('.weight') + suffix)
    return unpacked


def pack_packed_mxfp4(tensors: Mapping[str, np.ndarray],
                      stems: Collection[str]) -> dict[str, np.ndarray]:
    """Replace each named `<module>.weight` `[output, input]` with the
    compressed-tensors pair `quantize_packed_mxfp4` encodes it to. Only the
    stems a source shipped packed (`packed_mxfp4_stems`); a named stem the
    tensors no longer hold is refused, since the config would still promise
    its codes."""
    packed = dict(tensors)
    for stem in stems:
        if stem not in packed:
            raise ValueError(f"{stem} arrived MXFP4 packed and is not among the tensors to write back")
        weight = np.asarray(packed.pop(stem))
        if weight.ndim != 2:
            raise ValueError(f"{stem} packs a Linear's [output, input] weight, got {weight.shape}")
        module = stem.removesuffix('.weight')
        packed[module + PACKED_SUFFIXES[0]], packed[module + PACKED_SUFFIXES[1]] = quantize_packed_mxfp4(weight)
    return packed


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


def _check_grid(weight: np.ndarray, scale_inv: np.ndarray, block: int) -> None:
    """Refuse scales that are not one per block of a [rows, cols] weight,
    [ceil(rows / block), ceil(cols / block)]."""
    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(
            f"block-scaled dequantization takes a [rows, cols] weight and its "
            f"[row blocks, col blocks] scales, got shapes {weight.shape} and "
            f"{scale_inv.shape}")
    blocks = (math.ceil(weight.shape[0] / block), math.ceil(weight.shape[1] / block))
    if scale_inv.shape != blocks:
        raise ValueError(
            f"a {weight.shape} weight in {block} x {block} blocks takes a "
            f"{blocks} scale, got {scale_inv.shape}")


def dequantize_fp8_blocks(weight: np.ndarray, scale_inv: np.ndarray,
                          block: int = BLOCK) -> np.ndarray:
    """Return float32(weight[i, j]) * float32(scale_inv[i // block, j // block]).

    `weight` may be in any float dtype, `scale_inv` float32 or E8M0.
    """
    _check_grid(weight, scale_inv, block)
    # Scaled one block row at a time, so no scale array of the weight's size
    # is ever built.
    out = weight.astype(np.float32)
    scales = scale_inv.astype(np.float32)
    for index in range(scales.shape[0]):
        out[index * block:(index + 1) * block] *= np.repeat(scales[index], block)[:weight.shape[1]]
    return out


def fp8_tensor_names(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Return the tensor names that survive decoding.

    A `_scale_inv` partner is metadata, not a model tensor, so it is dropped.
    """
    return tuple(name for name in tensors if not name.endswith(SCALE_SUFFIX))


def _read_blocks(weight: np.ndarray, scale: np.ndarray, block: int, index: Index | None) -> np.ndarray:
    """The region `index` names of a block-scaled [rows, cols] weight. The
    whole scale grid is checked first: a grid of other blocks can fit the
    region it covers."""
    _check_grid(weight, scale, block)
    covered, span, within = _hull(index, weight.shape, (block, block))
    return dequantize_fp8_blocks(weight[span], scale[covered], block)[within]


def read_fp8_tensor(tensors: Mapping[str, np.ndarray], name: str, index: Index | None = None,
                    *, block: int) -> np.ndarray:
    """Return one tensor, or the region `index` names, in its original values,
    a scaled one decoded in FP32 from the blocks the region covers.

    Only the requested tensor is decoded, so alias validation never builds an
    FP32 copy of the checkpoint. An unscaled value keeps its stored dtype.
    """
    value = tensors[name]
    scale = tensors.get(name + SCALE_SUFFIX)
    if scale is None:
        return value if index is None else value[index]
    return _read_blocks(value, scale, block, index)


def dequantize_checkpoint(tensors: Mapping[str, np.ndarray], block: int, *,
                          param_dtype: str = "float32") -> dict[str, np.ndarray]:
    """Apply each scale to its weight, drop the scale, and cast to `param_dtype`.

    The block multiply stays FP32 and each weight is cast as it is written, so
    no whole decoded model is held in FP32. A scale whose weight is absent is
    refused. Unscaled tensors pass through unchanged.
    """
    out = dict(tensors)
    for name in tuple(out):
        if not name.endswith(SCALE_SUFFIX):
            continue
        scaled = name[:-len(SCALE_SUFFIX)]
        if scaled not in out:
            raise ValueError(
                f"{name} scales {scaled}, which the checkpoint does not hold")
        out[scaled] = checkpoint_array(read_fp8_tensor(out, scaled, block=block), param_dtype)
        out.pop(name)
    return out


def scaled_names(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Return the names in `tensors` that have a `<name>_scale_inv` partner, in order.

    Take these before `dequantize_checkpoint` consumes the partners; afterwards
    nothing says which tensors arrived quantized.
    """
    return tuple(name[:-len(SCALE_SUFFIX)] for name in tensors
                 if name.endswith(SCALE_SUFFIX))


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


def pack_fp8(tensors: Mapping[str, np.ndarray], names: Iterable[str], block: int, *,
             ue8m0: bool) -> dict[str, np.ndarray]:
    """Return `tensors` with each of `names` written back as fp8 blocks and scales.

    `dequantize_checkpoint` run backwards, for the names a source shipped
    quantized (`scaled_names`) in the format its own config declares
    (`fp8_format`). Every tensor not named passes through as itself; a name
    the caller no longer holds is refused rather than written dense under a
    config that calls it quantized.
    """
    out = dict(tensors)
    for name in names:
        if name not in out:
            raise ValueError(
                f"{name} was quantized in the source and is not among the tensors "
                f"to write")
        partner = name + SCALE_SUFFIX
        if partner in out:
            raise ValueError(
                f"{partner} is already among the tensors to write, so quantizing "
                f"{name} would overwrite it")
        out[name], out[partner] = quantize_fp8_blocks(out[name], block, ue8m0=ue8m0)
    return out


# --------------------------------------------------------------------------
# DeepSeek-V4: quant_method fp8 beside expert_dtype, <m>.weight and <m>.scale
# --------------------------------------------------------------------------

V4_SCALE_SUFFIX = '.scale'
"""DeepSeek-V4 scales `<m>.weight` by `<m>.scale`."""

V4_SCALE_DTYPES = ('float8_e8m0fnu', 'float32')
"""The dtypes a DeepSeek-V4 FP8 weight's `.scale` ships in: E8M0 exponents
in V4-Flash, V4-Pro and V4.1-Flash, float32 powers of two in V4-Flash-Base
and V4-Pro-Base. The config says neither, so the loader records which."""

V4_FP4_AMAX_FLOOR = 6 * 2.0 ** -126
"""The release's floor on an FP4 group's amax, which keeps its scale at
least 2 ** -126 (E8M0 byte 1)."""

V4Layout = Literal['blocks', 'rows', 'fp4']
"""How a DeepSeek-V4 `.scale` covers its weight: one per block of an FP8
Linear, one per `block` values of an engram table's row, or one per 32
inputs of an FP4 routed expert."""

_V4_EXPERT = re.compile(r'\.experts\.\d+\.w[123]\.weight$')
"""A routed expert's projection, `layers.N.ffn.experts.E.w1` or under `mtp.N`."""

_V4_ENGRAM = '.engram.embed.weight'
"""V4.1's n-gram hash table, `layers.N.engram.embed.weight`."""


def deepseek_v4_format(quantization: Mapping[str, object],
                       config: Mapping[str, object]) -> tuple[int, bool]:
    """Return the block size and whether routed experts are FP4, from a
    DeepSeek-V4 config.

    `fp8_format`'s blocks, with ue8m0 scales, the only kind an E8M0 `.scale`
    holds. `expert_dtype` sits in quantization_config (V4.1) or beside it
    (V4): 'fp4' ships routed experts as E2M1 pairs (inference/model.py
    `Linear` under float4_e2m1fn_x2), 'fp8' or null as FP8 blocks like every
    other Linear (the Base releases).
    """
    block, ue8m0 = fp8_format(quantization)
    if not ue8m0:
        raise ValueError(
            f"quantization_config declares expert_dtype, DeepSeek-V4's `.scale` storage, with "
            f"scale_fmt {quantization.get('scale_fmt')!r}; those scales are powers of two, which "
            f"the release declares as scale_fmt 'ue8m0'")
    experts = quantization.get('expert_dtype', config.get('expert_dtype'))
    if experts not in ('fp4', 'fp8', None):
        raise ValueError(
            f"expert_dtype {experts!r}: DeepSeek-V4 ships routed experts as fp4 E2M1 pairs "
            f"or as fp8 blocks and nothing else")
    return block, experts == 'fp4'


def deepseek_v4_layout(name: str, fp4_experts: bool) -> V4Layout:
    """How the `.scale` beside `name` covers it, as inference/model.py builds
    the module: an FP4 `Linear` for a routed expert under expert_dtype
    'fp4', `ParallelEngramEmbedding` for an engram table, an FP8 `Linear`
    otherwise."""
    if fp4_experts and _V4_EXPERT.search(name):
        return 'fp4'
    return 'rows' if name.endswith(_V4_ENGRAM) else 'blocks'


def deepseek_v4_names(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
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


def deepseek_v4_tensor_names(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Decoded names: every tensor but the `.scale` partners."""
    return tuple(name for name in tensors if not name.endswith(V4_SCALE_SUFFIX))


def deepseek_v4_scale_dtype(tensors: Mapping[str, np.ndarray]) -> str | None:
    """The dtype a DeepSeek-V4 checkpoint stores its `.scale` tensors in, or
    None without any. Scales in more than one dtype are refused."""
    stored = sorted({tensors[name.removesuffix('.weight') + V4_SCALE_SUFFIX].dtype.name
                     for name in deepseek_v4_names(tensors)})
    if len(stored) > 1:
        raise ValueError(f"a DeepSeek-V4 checkpoint stores its `.scale` tensors in one dtype, "
                         f"this one in {stored}")
    return stored[0] if stored else None


def _check_v4(name: str, layout: V4Layout, weight: np.ndarray, scale: np.ndarray, block: int) -> None:
    """Refuse a DeepSeek-V4 pair whose dtypes or shapes are not its layout's."""
    partner = name.removesuffix('.weight') + V4_SCALE_SUFFIX
    packed = weight.dtype in _CODE_DTYPES
    if layout == 'fp4':
        if (not packed or weight.ndim != 2 or scale.dtype != ml_dtypes.float8_e8m0fnu
                or 2 * weight.shape[1] % GROUP
                or scale.shape != (weight.shape[0], 2 * weight.shape[1] // GROUP)):
            raise ValueError(
                f"{name} is a routed expert, which expert_dtype 'fp4' ships as int8 [out, in / 2] "
                f"E2M1 pairs beside a float8_e8m0fnu [out, in / {GROUP}] {partner}, got "
                f"{weight.dtype} {weight.shape} and {scale.dtype} {scale.shape}")
        return
    if packed:
        reason = ("the config's expert_dtype is not 'fp4'" if _V4_EXPERT.search(name)
                  else "it is not a routed expert")
        raise ValueError(
            f"{name} arrives as {weight.dtype} E2M1 pairs, which DeepSeek-V4 ships for routed "
            f"experts under expert_dtype 'fp4' alone, and {reason}")
    if weight.dtype != E4M3 or weight.ndim != 2 or scale.dtype.name not in V4_SCALE_DTYPES:
        raise ValueError(
            f"{name} and {partner} are a DeepSeek-V4 FP8 weight, float8_e4m3fn beside a scale in "
            f"one of {list(V4_SCALE_DTYPES)}, got {weight.dtype} {weight.shape} and {scale.dtype}")
    if layout == 'rows' and (weight.shape[1] % block
                             or scale.shape != (weight.shape[0], weight.shape[1] // block)):
        raise ValueError(
            f"{name} is an engram table, [rows, dim] beside a [rows, dim / {block}] {partner}, "
            f"got {weight.shape} and {scale.shape}")


def read_deepseek_v4_tensor(tensors: Mapping[str, np.ndarray], name: str, index: Index | None = None,
                            *, block: int, fp4_experts: bool) -> np.ndarray:
    """One tensor, or the region `index` names, in its original values; a
    scaled weight decodes in FP32 from the blocks or groups the region covers.

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
    module = name.removesuffix('.weight')
    scale = tensors.get(module + V4_SCALE_SUFFIX) if module != name else None
    if scale is None:
        return value if index is None else value[index]
    layout = deepseek_v4_layout(name, fp4_experts)
    _check_v4(name, layout, value, scale, block)
    if layout == 'fp4':
        return _read_e2m1(value, scale, index)
    if layout == 'blocks':
        return _read_blocks(value, scale, block, index)
    covered, span, within = _hull(index, value.shape, (1, block))
    rows = value[span].astype(np.float32).reshape(*scale[covered].shape, block)
    rows *= scale[covered].astype(np.float32)[..., None]
    return rows.reshape(rows.shape[0], rows.shape[1] * block)[within]


def unpack_deepseek_v4(tensors: Mapping[str, np.ndarray], *, block: int, fp4_experts: bool,
                       param_dtype: str = "float32") -> dict[str, np.ndarray]:
    """Decode each `.scale` pair in FP32 into its weight, retained in
    param_dtype, and drop the `.scale`. Every other tensor remains untouched."""
    unpacked = dict(tensors)
    for name in deepseek_v4_names(tensors):
        unpacked[name] = checkpoint_array(
            read_deepseek_v4_tensor(tensors, name, block=block, fp4_experts=fp4_experts), param_dtype)
        unpacked.pop(name.removesuffix('.weight') + V4_SCALE_SUFFIX)
    return unpacked


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
    amax = np.maximum(np.abs(groups).max(-1), np.float32(V4_FP4_AMAX_FLOOR))
    bits = (amax * np.float32(1 / 6)).view(np.uint32)
    exponents = ((bits >> 23) + ((bits & 0x007fffff) != 0)).astype(np.uint8)
    codes = encode_e2m1(groups / e8m0_scales(exponents)[..., None])
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


def pack_deepseek_v4(tensors: Mapping[str, np.ndarray], names: Iterable[str], *, block: int,
                     fp4_experts: bool, scale_dtype: str | None = None) -> dict[str, np.ndarray]:
    """Return `tensors` with each of `names` written back as a DeepSeek-V4
    weight and `.scale`, in the layout `deepseek_v4_layout` gives it.

    An FP8 weight takes `quantize_fp8_blocks` or `quantize_fp8_rows` under
    ue8m0, its scale stored in `scale_dtype`, the dtype the source stored
    its scales in (`deepseek_v4_scale_dtype`; E8M0 when None). An FP4
    expert takes `quantize_deepseek_v4_fp4`, whose scales are E8M0. A name
    the caller no longer holds is refused, and so is a `.scale` already
    among the tensors.
    """
    stored = np.dtype(scale_dtype or V4_SCALE_DTYPES[0])
    if stored.name not in V4_SCALE_DTYPES:
        raise ValueError(f"a DeepSeek-V4 FP8 `.scale` is stored in one of {list(V4_SCALE_DTYPES)}, "
                         f"not {stored.name}")
    out = dict(tensors)
    for name in names:
        if name not in out:
            raise ValueError(f"{name} was quantized in the source and is not among the tensors to write")
        partner = name.removesuffix('.weight') + V4_SCALE_SUFFIX
        if partner in out:
            raise ValueError(f"{partner} is already among the tensors to write, so quantizing "
                             f"{name} would overwrite it")
        layout = deepseek_v4_layout(name, fp4_experts)
        if layout == 'fp4':
            out[name], out[partner] = quantize_deepseek_v4_fp4(out[name])
            continue
        codes, scale_inv = (quantize_fp8_rows(out[name], block) if layout == 'rows'
                            else quantize_fp8_blocks(out[name], block, ue8m0=True))
        out[name], out[partner] = codes, scale_inv.astype(stored)
    return out


# --------------------------------------------------------------------------
# Dispatch from quantization_config
# --------------------------------------------------------------------------

class TensorReader(Protocol):
    """Decodes one source tensor, or the region `index` names, in its original values."""

    def __call__(self, tensors: Mapping[str, np.ndarray], name: str,
                 index: Index | None = None) -> np.ndarray: ...


def _one_scale_dtype(tensors: Mapping[str, np.ndarray]) -> None:
    """A format whose scales have one dtype records none."""


@dataclass(frozen=True)
class SourceQuantization:
    """Describes a source format the loader undoes and `Pretrained.save` restores.

    `names` reads which tensors arrived quantized off the raw checkpoint,
    before `dequantize` replaces them with dense weights in requested
    storage; `requantize` writes those names back in the format.
    `tensor_names` and `read` expose original values one tensor, or one
    region of one, at a time, for alias checks and streamed loads, so
    neither holds an FP32 model. `scale_dtype` reads the dtype a source
    stored its scales in where the format leaves that to the checkpoint
    (DeepSeek-V4), for `source_quantization` to write back.
    """

    names: Callable[[Mapping[str, np.ndarray]], tuple[str, ...]]
    dequantize: Callable[[Mapping[str, np.ndarray]], dict[str, np.ndarray]]
    requantize: Callable[[Mapping[str, np.ndarray], tuple[str, ...]], dict[str, np.ndarray]]
    tensor_names: Callable[[Mapping[str, np.ndarray]], tuple[str, ...]]
    read: TensorReader
    scale_dtype: Callable[[Mapping[str, np.ndarray]], str | None] = _one_scale_dtype


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


def source_quantization(config: Mapping[str, object], *, param_dtype: str = "float32",
                        scale_dtype: str | None = None) -> SourceQuantization | None:
    """Return the format a config's `quantization_config` declares, or None.

    A wrapper may declare it on its text_config alone: KimiK3Config lifts
    `text_config.quantization_config` onto itself (configuration_kimi_k3.py:282-283).
    An fp8 config that declares `expert_dtype`, in quantization_config or
    beside it, is DeepSeek-V4's `.scale` storage; `scale_dtype` is the dtype
    the source stored those scales in, which the codec writes back.
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
        block, fp4_experts = deepseek_v4_format(quantization, config)
        return SourceQuantization(
            deepseek_v4_names,
            partial(unpack_deepseek_v4, block=block, fp4_experts=fp4_experts, param_dtype=param_dtype),
            partial(pack_deepseek_v4, block=block, fp4_experts=fp4_experts, scale_dtype=scale_dtype),
            deepseek_v4_tensor_names, partial(read_deepseek_v4_tensor, block=block, fp4_experts=fp4_experts),
            deepseek_v4_scale_dtype)
    if method == "fp8":
        block, ue8m0 = fp8_format(quantization)
        return SourceQuantization(
            scaled_names, partial(dequantize_checkpoint, block=block, param_dtype=param_dtype),
            partial(pack_fp8, block=block, ue8m0=ue8m0),
            fp8_tensor_names, partial(read_fp8_tensor, block=block))
    if method == "mxfp4":
        return SourceQuantization(
            mxfp4_stems, partial(unpack_mxfp4, param_dtype=param_dtype), pack_mxfp4,
            mxfp4_tensor_names, read_mxfp4_tensor)
    if method == "compressed-tensors":
        packed_mxfp4_format(quantization)
        return SourceQuantization(
            packed_mxfp4_stems, partial(unpack_packed_mxfp4, param_dtype=param_dtype), pack_packed_mxfp4,
            packed_mxfp4_tensor_names, read_packed_mxfp4_tensor)
    raise ValueError(
        f"quantization_config names quant_method {method!r}; this loader reads DeepSeek's "
        f"fp8 blocks and V4 `.scale` storage, GPT OSS's mxfp4 and compressed-tensors' "
        f"mxfp4-pack-quantized and nothing else")
