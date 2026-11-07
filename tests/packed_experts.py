"""Packed NVFP4 expert banks (`dew.nn.kernels.gemm.NvFp4`) and their exact values, for the expert layers'
served-checkpoint tests (row 4516)."""

import jax.numpy as jnp
import numpy as np


def packed_banks(experts: int, depth: int, width: int, seed: int):
    """Packed NVFP4 expert banks of random codes and block scales, each expert's tensor scale a power of two
    (an MXFP4 checkpoint's, `codecs.mxfp4_as_nvfp4`), and their values `[experts, depth, width]` float32,
    exactly."""
    from dew.nn.kernels.gemm import NvFp4

    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, (experts, depth, width), dtype=np.uint8)
    scales = np.ldexp(np.float32(1), rng.integers(-3, 3, (experts, depth // 16, width))).astype(
        jnp.float8_e4m3fn)
    scale = np.ldexp(np.float32(1), rng.integers(-4, 0, (experts,))).astype(np.float32)
    bank = NvFp4.of(codes, scales, scale)
    values = np.asarray(bank.decoded(), np.float32) * np.asarray(bank.factor)[:, None, None]
    return bank, values
