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
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

BLOCK = 128
"""The block of DeepSeek's weight_block_size, [128, 128]."""

SCALE_SUFFIX = '_scale_inv'
"""What a quantized tensor's scale partner appends to its name: the weight
`layer.weight` scales by `layer.weight_scale_inv`."""


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
