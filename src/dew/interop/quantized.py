"""Read and write quantized checkpoint weights: DeepSeek's block-scaled FP8
and compressed-tensors' MXFP4.

A quantized linear's `weight` is float8_e4m3fn [out, in] and its partner
`weight_scale_inv` is float32 [ceil(out / 128), ceil(in / 128)], one scale
per 128 x 128 block with a partial last block scaled on its own (the config's
quantization_config: fmt e4m3, weight_block_size [128, 128]; V3.2 adds
scale_fmt ue8m0, powers of two stored as float32 all the same).

Reading is DeepSeek's `weight_dequant` (inference/kernel.py of DeepSeek-V3):
one fp32 multiply per element. Writing is DeepGEMM's `per_block_cast_to_fp8`
(deep_gemm/utils/math.py), the cast that produces weights in this format,
ported operation for operation so the bytes agree with it exactly:

    amax = max(|x|) over the block, in float32, clamped up to 1e-4
    scale_inv = amax / 448
    scale_inv = 2 ** ceil(log2(scale_inv))    # ue8m0 only, float32 log2
    q = float8_e4m3fn(x * (1 / scale_inv))

Not `act_quant` in the DeepSeek file, the activation rule: it reduces over
1 x 128 tiles and divides where the weight cast multiplies by a reciprocal.
The partial blocks the reference zero-pads are reduced in place here, which
zeros and the 1e-4 floor make the same reduction.

The cast needs no clamp. A float32 scale is amax / 448 to within a few
roundings, and a ue8m0 scale sits at most 2 ** -21 relative under the
quotient it rounds (float32 log2 lands on the integer for a quotient that
close above a power of two; measured against torch in tests), so a scaled
element stays under 448.001. E4M3FN rounds everything up to 464 to its 448,
and that margin is where the libraries part: ml_dtypes sends a value above
it to NaN where torch saturates. A weight that is not finite is refused by
name rather than written as a block of NaN. The 1e-4 floor keeps every
scale a float32 normal, and the arithmetic is NumPy's on a host copy, clear
of XLA's flush-to-zero on CPU.

compressed-tensors' `mxfp4-pack-quantized` format, which Kimi K3 ships its
routed experts in, stores each torch Linear as `<module>.weight_packed`
`[output, input / 2]` beside `<module>.weight_scale` `[output, input / 32]`:
low-nibble-first E2M1 codes and bias-127 E8M0 exponents over groups of 32
inputs (`pack_fp4_to_uint8`, `unpack_fp4_from_uint8` and
`decompress_mx_scale` of compressed-tensors 0.17.1). Those are GPT OSS's
codes and scales in another layout, one output row per block row, so they
decode through `dew.nn.gpt_oss.dequantize_mxfp4`.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping

import ml_dtypes
import numpy as np
from numpy.typing import ArrayLike

from dew import records
from dew.nn.gpt_oss import GROUP, dequantize_mxfp4, quantize_mxfp4

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
    from dew.nn.text_encoders import checkpoint_array

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
    packed, scales = np.asarray(tensors[module + PACKED_SUFFIXES[0]]), np.asarray(tensors[module + PACKED_SUFFIXES[1]])
    if (packed.ndim != 2 or packed.shape[1] % (GROUP // 2)
            or scales.shape != (packed.shape[0], packed.shape[1] // (GROUP // 2))):
        raise ValueError(
            f"{name} packs [output, input / 2] codes beside [output, input / {GROUP}] scales, "
            f"got {packed.shape} and {scales.shape}")
    blocks = packed.reshape(1, packed.shape[0], -1, GROUP // 2)
    return dequantize_mxfp4(blocks, scales[None])[0].T


def unpack_packed_mxfp4(tensors: Mapping[str, np.ndarray], *,
                        param_dtype: str = "float32") -> dict[str, np.ndarray]:
    """Decode each compressed-tensors pair in FP32 into its Linear's `.weight`,
    retained in param_dtype. Every other tensor remains untouched."""
    from dew.nn.text_encoders import checkpoint_array

    unpacked = dict(tensors)
    for stem in packed_mxfp4_stems(tensors):
        unpacked[stem] = checkpoint_array(read_packed_mxfp4_tensor(unpacked, stem), param_dtype)
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
