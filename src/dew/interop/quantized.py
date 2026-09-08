"""Block-scaled FP8 weights, as the DeepSeek V3 and V3.2 checkpoints ship them.

A quantized linear's `weight` is float8_e4m3fn, [out, in], and its partner
`weight_scale_inv` is float32, [ceil(out / 128), ceil(in / 128)]: one scale
per 128 x 128 block, the inverse of the scale the block was divided by (the
checkpoint's quantization_config names fmt e4m3 and weight_block_size
[128, 128]). V3.2's scales are powers of two (scale_fmt ue8m0), stored as
float32 all the same, so they take the same arithmetic. A row or column
count that is not a multiple of 128 has a partial last block with a scale of
its own: V3's kv_a_proj_with_mqa is [576, 7168] with a [5, 56] scale.

The dequantization is DeepSeek's `weight_dequant` (inference/kernel.py of
deepseek-ai/DeepSeek-V3): y[i, j] = float32(x[i, j]) * s[i // 128, j // 128],
one fp32 multiply per element, the partial blocks masked. transformers'
`Fp8Dequantize` (integrations/finegrained_fp8.py) computes the same product
where the blocks are whole and refuses a partial one.

Writing the format back is `quantize_fp8_blocks`, and its reference is the
one that produces *weights* in it: `per_block_cast_to_fp8` in
deep_gemm/utils/math.py of deepseek-ai/DeepGEMM.

    amax = max(|x|) over the block, in float32, clamped up to 1e-4
    scale_inv = amax / 448                  # 448 is E4M3FN's largest finite
    scale_inv = ceil_to_ue8m0(scale_inv)    # only where the config says ue8m0
    q = float8_e4m3fn(x * (1 / scale_inv))  # one reciprocal per block

That reference and not `act_quant` in the DeepSeek file above, which is the
rule for *activations*: it reduces over 1 x 128 tiles along the last
dimension, which is not the grid a [ceil(rows / 128), ceil(cols / 128)]
weight scale covers, and it divides each element by the scale where the
weight cast multiplies by the block's reciprocal, a last-bit difference in
every element. The two agree on everything the tile shape does not decide:
the amax reduction, the 1e-4 floor, `amax / 448`, and rounding a ue8m0 scale
up to a power of two. `per_block_cast_to_fp8` is ported here operation for
operation, so the bytes agree with it exactly, which is what
tests/test_quantized.py holds it to against torch.

Four details of that reference carry the edge cases, none of them decided
here:

- **Partial blocks.** The reference zero-pads the weight out to whole
  blocks, reduces over the padded blocks, and slices the quantized result
  back. Zeros cannot raise an amax and the 1e-4 floor covers a block that is
  all padding, so the padding changes nothing and the edge blocks are
  reduced in place here instead.
- **The 1e-4 floor** is what makes an all-zero block legal: without it the
  scale would be zero and every element of the block 0/0. A block whose amax
  falls under the floor is scaled by 1e-4 / 448 rather than by itself, so it
  keeps its magnitudes relative to the floor, and one far enough under it
  flushes to zero. That is the reference's behaviour and not a degenerate
  case treated apart.
- **ue8m0** is `ceil_to_ue8m0`, an exact integer rule on the float32 bits:
  the biased exponent, plus one where any mantissa bit is set, clamped to
  [1, 254]. It is 2 ** ceil(log2(scale_inv)) with no libm and no rounding of
  its own, and it moves a scale up, never down.
- **Overflow cannot arise.** scale_inv is amax / 448 to within three
  roundings, so |x * (1 / scale_inv)| <= 448 * (1 + 4 * 2 ** -24), which is
  448.000107; E4M3FN rounds everything below 464 down to its 448. That bound
  is worth stating because it is the one place the libraries part:
  `ml_dtypes.float8_e4m3fn` sends an out-of-range value to NaN where
  `torch.float8_e4m3fn` saturates to 448 (measured, torch 2.14 on CPU), and
  E4M3FN has no infinity for either of them to reach for. Every value this
  encoder casts is inside the range where they agree, so it needs no clamp
  of its own and the bytes do not depend on which library casts them. A
  weight that is not finite has neither the bound nor a representation, and
  is refused by name rather than written as a block of NaN.
- **Underflow cannot arise either.** The 1e-4 floor puts scale_inv at or
  above 1e-4 / 448, which is 2.2e-7 and thirty-one decades clear of
  float32's smallest normal, so neither a scale nor its reciprocal is ever
  subnormal and no flush-to-zero mode can reach them. The arithmetic is
  NumPy's throughout, on a host copy of the weight, so it is also clear of
  XLA's flush-to-zero on CPU: a trained leaf arrives as a jax array or a
  bfloat16 master and the bytes come out the same either way.

Nothing here decides on its own what to quantize. `pack_fp8` writes the
names it is handed, which `scaled_names` reads off the source's own
`_scale_inv` partners before `dequantize_checkpoint` consumes them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import ml_dtypes
import numpy as np

BLOCK = 128
"""The block of DeepSeek's weight_block_size, [128, 128]."""

SCALE_SUFFIX = '_scale_inv'
"""What a quantized tensor's scale partner appends to its name: the weight
`layer.weight` scales by `layer.weight_scale_inv`."""

E4M3 = ml_dtypes.float8_e4m3fn
"""The dtype a quantized weight is stored in, safetensors' F8_E4M3."""

E4M3_MAX = 448.0
"""E4M3FN's largest finite value, 2 ** 8 * 1.75, which a block's amax is
scaled to. OCP FP8's `fn` encoding spends no code point on an infinity, so
the format holds nothing above this."""

AMAX_FLOOR = 1e-4
"""`per_block_cast_to_fp8`'s `clamp(1e-4)` on a block's amax, which is what
keeps a zero or near-zero block's scale positive."""

SCALE_FORMATS = (None, 'ue8m0')
"""The `scale_fmt` a checkpoint declares, of the two this writer produces:
absent for V3's plain float32 scales, `ue8m0` for V3.2's powers of two."""


def fp8_block(hf_config: Mapping[str, Any]) -> int | None:
    """The block of a checkpoint's fp8 `quantization_config`, or None.

    None is a checkpoint with no quantization_config, or one quantized
    another way (GPT-OSS's mxfp4), which its own reader handles; a config
    that names fp8 has to name the square block DeepSeek's scales cover,
    since a per-tensor or a rectangular scale is not the format here.
    """
    quantization = hf_config.get('quantization_config')
    if quantization is None or quantization.get('quant_method') != 'fp8':
        return None
    fmt, block = quantization.get('fmt'), quantization.get('weight_block_size')
    if (fmt != 'e4m3' or not isinstance(block, list) or len(block) != 2
            or any(type(side) is not int or side < 1 for side in block)
            or block[0] != block[1]):
        raise ValueError(
            f"quantization_config names fp8 with fmt {fmt!r} and weight_block_size "
            f"{block!r}; this loader dequantizes DeepSeek's e4m3 weights in square "
            f"blocks and nothing else")
    return block[0]


def dequantize_fp8_blocks(weight: np.ndarray, scale_inv: np.ndarray,
                          block: int = BLOCK) -> np.ndarray:
    """`weight` times its per-block scales, in float32.

    `weight` is [rows, cols] in any float dtype (float8_e4m3fn as read from
    the checkpoint, or already widened, since every fp8 value is exact in
    fp32), `scale_inv` is [ceil(rows / block), ceil(cols / block)], and
    element (i, j) comes out as float32(weight[i, j]) * scale_inv[i // block,
    j // block], one fp32 multiply; the result equals the reference bit for
    bit.
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
    # The widened copy is the output; each block row of it scales in place
    # against one row of the scales, so no scale array of the weight's size
    # is ever built.
    out = weight.astype(np.float32)
    scales = scale_inv.astype(np.float32)
    for index in range(blocks[0]):
        out[index * block:(index + 1) * block] *= np.repeat(scales[index], block)[:cols]
    return out


def dequantize_checkpoint(tensors: Mapping[str, np.ndarray],
                          block: int | None) -> dict[str, np.ndarray]:
    """`tensors` with every `<name>_scale_inv` applied to `<name>` and dropped.

    The loader's hook: a checkpoint read as fp32 arrives here with the block
    its config names (`fp8_block`), and a weight with a scale partner leaves
    as the dequantized fp32 weight the model loads, while every other tensor
    passes through untouched. A scale with no weight to scale is refused,
    since a checkpoint that ships one has lost a tensor, and so is a scale
    in a checkpoint whose config names no block, since the scale's grid
    alone cannot say how large a partial block is.
    """
    out = dict(tensors)
    for name in tuple(out):
        if not name.endswith(SCALE_SUFFIX):
            continue
        scaled = name[:-len(SCALE_SUFFIX)]
        if scaled not in out:
            raise ValueError(
                f"{name} scales {scaled}, which the checkpoint does not hold")
        if block is None:
            raise ValueError(
                f"{name} carries block scales and the checkpoint's config names no "
                f"fp8 quantization_config with a weight_block_size")
        out[scaled] = dequantize_fp8_blocks(out[scaled], out.pop(name), block)
    return out


def scaled_names(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """The names in `tensors` that ship a `<name>_scale_inv` partner, in order.

    Which tensors a source quantized, read off the source itself instead of
    guessed from a name pattern or a shape, and read *before*
    `dequantize_checkpoint` consumes the partners: afterwards a weight that
    arrived as fp8 blocks is an fp32 array like every other one, and nothing
    in the tensors or the config says which it was. A caller that means to
    write the format back keeps this tuple and hands it to `pack_fp8`.
    """
    return tuple(name[:-len(SCALE_SUFFIX)] for name in tensors
                 if name.endswith(SCALE_SUFFIX))


def _ue8m0(scale_inv: np.ndarray) -> np.ndarray:
    """`scale_inv` rounded up to powers of two, DeepGEMM's `ceil_to_ue8m0`.

    A positive float32's biased exponent, plus one wherever its mantissa
    holds anything, is ceil(log2(x)) exactly; writing that exponent back
    with an empty mantissa is 2 ** ceil(log2(x)). The clamp is the
    reference's, and keeps the result a normal number.
    """
    bits = scale_inv.view(np.uint32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
    return (np.clip(exponent, 1, 254).astype(np.uint32) << 23).view(np.float32)


def quantize_fp8_blocks(weight: np.ndarray, block: int = BLOCK, *,
                        scale_fmt: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`weight` as float8_e4m3fn blocks and the float32 scales that invert them.

    The inverse of `dequantize_fp8_blocks`: a [rows, cols] weight in any
    float dtype gives a float8_e4m3fn array of its shape and a
    [ceil(rows / block), ceil(cols / block)] float32 `scale_inv`, the pair a
    checkpoint ships, such that dequantizing them is the weight rounded to
    the format. `scale_fmt` is the source config's: None for V3's float32
    scales, 'ue8m0' for V3.2's powers of two.

    The weight widens to float32 first, exactly, from any narrower float
    dtype, which is the precision the reference reduces and scales in.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"block-scaled quantization takes a [rows, cols] weight, got shape "
            f"{weight.shape}")
    if type(block) is not int or block < 1:
        raise ValueError(
            f"a block covers a positive number of rows and columns, got {block!r}")
    if scale_fmt not in SCALE_FORMATS:
        raise ValueError(
            f"scale_fmt {scale_fmt!r} names no scale format this writer produces; "
            f"DeepSeek's checkpoints declare 'ue8m0' or leave it out")
    values = np.asarray(weight, np.float32)
    if not np.isfinite(values).all():
        raise ValueError(
            f"a {values.shape} weight holds "
            f"{int(np.count_nonzero(~np.isfinite(values)))} value(s) that are not "
            f"finite; E4M3FN encodes no infinity, and one of them carries its "
            f"whole {block} x {block} block's scale away with it")
    rows, cols = values.shape
    row_starts, col_starts = np.arange(0, rows, block), np.arange(0, cols, block)
    # Two max reductions give the per-block amax without padding the weight
    # out to whole blocks the way the reference does: reduceat's last segment
    # in each dimension is the partial block, over just its own elements.
    amax = np.maximum.reduceat(
        np.maximum.reduceat(np.abs(values), row_starts, axis=0), col_starts, axis=1)
    scale_inv = np.maximum(amax, np.float32(AMAX_FLOOR)) / np.float32(E4M3_MAX)
    if scale_fmt == 'ue8m0':
        scale_inv = _ue8m0(scale_inv)
    out = np.empty(values.shape, E4M3)
    for index, start in enumerate(row_starts):
        # One reciprocal per block, repeated across the columns it covers, so
        # a block row costs one [block, cols] product and no scale array of
        # the weight's size is ever built.
        scales = np.repeat(np.float32(1.0) / scale_inv[index], block)[:cols]
        out[start:start + block] = (values[start:start + block] * scales).astype(E4M3)
    return out, scale_inv


def pack_fp8(tensors: Mapping[str, np.ndarray], names: Iterable[str],
             hf_config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """`tensors` with each of `names` written back as fp8 blocks and scales.

    `dequantize_checkpoint` run backwards, for an export that holds a
    source's own list of quantized tensors (`scaled_names`, taken before the
    load dequantized them) and the source's own config. The block and the
    scale format come from that config's fp8 `quantization_config`, so the
    checkpoint is written in the format its config declares rather than one
    inferred from the trained tensors, and every tensor not named passes
    through untouched: a source that left its embeddings or its norms dense
    re-exports them dense. A name the caller no longer holds is refused
    rather than written dense under a config that calls it quantized.
    """
    block = fp8_block(hf_config)
    if block is None:
        raise ValueError(
            "writing fp8 blocks takes the block from the source's fp8 "
            "quantization_config, and this config declares none")
    scale_fmt = hf_config['quantization_config'].get('scale_fmt')
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
        out[name], out[partner] = quantize_fp8_blocks(out[name], block,
                                                      scale_fmt=scale_fmt)
    return out
