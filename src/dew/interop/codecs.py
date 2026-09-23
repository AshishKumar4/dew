"""Read and write the quantized formats checkpoints ship their weights in.

`source_quantization` reads a config's `quantization_config` into one codec
and refuses every other format by name. A codec knows which tensors of a
checkpoint are the parts of one quantized weight, decodes them in float32
one tensor at a time, and encodes dense weights back into those parts for
`Pretrained.save`:

- DeepSeek's block-scaled FP8 (`quant_method: fp8`; V3, V3.2 and the
  finegrained FP8 of Qwen3 and MiniMax): `<m>.weight` in float8_e4m3fn
  beside `<m>.weight_scale_inv`, one float32 scale per square block.
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
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from functools import partial

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

_CODE_DTYPES = (np.dtype(np.uint8),)
"""Packed E2M1 pairs: U8 in GPT OSS and compressed-tensors."""

_EXPONENT_DTYPES = (np.dtype(np.uint8),)
"""E8M0 exponent bytes: U8 in GPT OSS and compressed-tensors."""


def _bytes(array: ArrayLike, dtypes: tuple[np.dtype, ...]) -> np.ndarray:
    """`array`'s bytes as uint8, if it is stored in one of `dtypes`."""
    values = np.asarray(array)
    if values.dtype not in dtypes:
        raise ValueError(f"expected bytes stored as one of {[dtype.name for dtype in dtypes]}, "
                         f"got {values.dtype}")
    return values.view(np.uint8)


def e8m0_scales(exponents: ArrayLike) -> np.ndarray:
    """The float32 scale 2 ** (b - 127) of each E8M0 exponent byte b:
    exact down to byte 0's subnormal 2 ** -127, and NaN for byte 255, as
    float8_e8m0fnu reads."""
    return _bytes(exponents, _EXPONENT_DTYPES).view(ml_dtypes.float8_e8m0fnu).astype(np.float32)


def decode_e2m1(packed: ArrayLike, exponents: ArrayLike) -> np.ndarray:
    """Packed codes [..., n / 2] under exponent bytes [..., n / 32] to float32 [..., n].

    Element i is E2M1[code i] * 2 ** (exponents[i // 32] - 127), a product
    float32 holds exactly, down to 2 ** -128 at byte 0; code 8 is -0.0.
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
    return decode_e2m1(blocks.reshape(*scales.shape[:2], -1), scales).swapaxes(1, 2)


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


def read_mxfp4_tensor(tensors: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """One original tensor, with a packed weight decoded in FP32 on demand."""
    if name + '_blocks' in tensors:
        return dequantize_mxfp4(tensors[name + '_blocks'], tensors[name + '_scales'])
    return tensors[name]


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


def read_packed_mxfp4_tensor(tensors: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """One original tensor; a packed Linear's weight decodes to FP32 `[output, input]`."""
    module = name.removesuffix('.weight')
    if name == module or module + PACKED_SUFFIXES[0] not in tensors:
        return tensors[name]
    packed, scales = tensors[module + PACKED_SUFFIXES[0]], tensors[module + PACKED_SUFFIXES[1]]
    if (packed.dtype != np.uint8 or scales.dtype != np.uint8 or packed.ndim != 2
            or packed.shape[1] % (GROUP // 2)
            or scales.shape != (packed.shape[0], packed.shape[1] // (GROUP // 2))):
        raise ValueError(
            f"{name} packs U8 [output, input / 2] codes beside U8 [output, input / {GROUP}] scales, "
            f"got {packed.dtype} {packed.shape} and {scales.dtype} {scales.shape}")
    return decode_e2m1(packed, scales)


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
    compressed-tensors pair `quantize_mxfp4` encodes it to. Only the stems a
    source shipped packed (`packed_mxfp4_stems`); a named stem the tensors no
    longer hold is refused, since the config would still promise its codes."""
    packed = dict(tensors)
    for stem in stems:
        if stem not in packed:
            raise ValueError(f"{stem} arrived MXFP4 packed and is not among the tensors to write back")
        weight = np.asarray(packed.pop(stem))
        if weight.ndim != 2:
            raise ValueError(f"{stem} packs a Linear's [output, input] weight, got {weight.shape}")
        blocks, scales = quantize_mxfp4(weight.T[None])
        module = stem.removesuffix('.weight')
        packed[module + PACKED_SUFFIXES[0]] = blocks[0].reshape(weight.shape[0], -1)
        packed[module + PACKED_SUFFIXES[1]] = scales[0]
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


def dequantize_fp8_blocks(weight: np.ndarray, scale_inv: np.ndarray,
                          block: int = BLOCK) -> np.ndarray:
    """Return float32(weight[i, j]) * scale_inv[i // block, j // block].

    `weight` may be in any float dtype.
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


def fp8_tensor_names(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Return the tensor names that survive decoding.

    A `_scale_inv` partner is metadata, not a model tensor, so it is dropped.
    """
    return tuple(name for name in tensors if not name.endswith(SCALE_SUFFIX))


def read_fp8_tensor(tensors: Mapping[str, np.ndarray], name: str, *, block: int) -> np.ndarray:
    """Return one tensor in its original values, decoding it in FP32 if it is scaled.

    Only the requested tensor is decoded, so alias validation never builds an
    FP32 copy of the checkpoint. An unscaled value keeps its stored dtype.
    """
    value = tensors[name]
    scale = tensors.get(name + SCALE_SUFFIX)
    return value if scale is None else dequantize_fp8_blocks(value, scale, block)


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
    values = np.asarray(weight, np.float32)
    if values.ndim != 2:
        raise ValueError(
            f"block-scaled quantization takes a [rows, cols] weight, got shape "
            f"{values.shape}")
    if type(block) is not int or block < 1:
        raise ValueError(
            f"a block covers a positive number of rows and columns, got {block!r}")
    if not np.isfinite(values).all():
        raise ValueError(
            f"a {values.shape} weight holds "
            f"{int(np.count_nonzero(~np.isfinite(values)))} value(s) that are not "
            f"finite; E4M3FN encodes no infinity, and one of them carries its "
            f"whole {block} x {block} block's scale away with it")
    rows, cols = values.shape
    row_starts, col_starts = np.arange(0, rows, block), np.arange(0, cols, block)
    # reduceat's last segment in each dimension is the partial block, over
    # just its own elements.
    amax = np.maximum.reduceat(
        np.maximum.reduceat(np.abs(values), row_starts, axis=0), col_starts, axis=1)
    scale_inv = np.maximum(amax, np.float32(AMAX_FLOOR)) / np.float32(E4M3_MAX)
    if ue8m0:
        scale_inv = np.exp2(np.ceil(np.log2(scale_inv)))
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
# Dispatch from quantization_config
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceQuantization:
    """Describes a source format the loader undoes and `Pretrained.save` restores.

    `names` reads which tensors arrived quantized off the raw checkpoint,
    before `dequantize` replaces them with dense weights in requested
    storage; `requantize` writes those names back in the format.
    `tensor_names` and `read` expose original values one tensor at a time,
    for alias checks, so neither holds an FP32 model.
    """

    names: Callable[[Mapping[str, np.ndarray]], tuple[str, ...]]
    dequantize: Callable[[Mapping[str, np.ndarray]], dict[str, np.ndarray]]
    requantize: Callable[[Mapping[str, np.ndarray], tuple[str, ...]], dict[str, np.ndarray]]
    tensor_names: Callable[[Mapping[str, np.ndarray]], tuple[str, ...]]
    read: Callable[[Mapping[str, np.ndarray], str], np.ndarray]


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


def source_quantization(config: Mapping[str, object], *,
                        param_dtype: str = "float32") -> SourceQuantization | None:
    """Return the format a config's `quantization_config` declares, or None.

    A wrapper may declare it on its text_config alone: KimiK3Config lifts
    `text_config.quantization_config` onto itself (configuration_kimi_k3.py:282-283).
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
        f"fp8 blocks, GPT OSS's mxfp4 and compressed-tensors' mxfp4-pack-quantized and nothing else")
