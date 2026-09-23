#!/usr/bin/env python3
"""Write timm's bf16 RmsNorm2d output for the MobileNet norm parity test.

`timm.layers.fast_norm.rms_norm2d` is the norm every RmsNorm2d in
Gemma 3n's MobileNet-v5 runs on CPU. A bf16 tower (`tower.bfloat16()`)
computes it with the input, the square, the mean, the inverse root and both
products in bf16, and the weight rounded to bf16. The fixture holds a fixed
input and fp32 weight with the bf16 output, all NHWC.

Run with timm 1.0.29 and Torch 2.14.0 CPU:
  python tools/timm_rms_norm_reference.py --out tests/fixtures/gemma3n/rms_norm2d_bf16.npz
"""

import argparse

import numpy as np
import timm
import torch
from timm.layers.fast_norm import rms_norm2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if timm.__version__ != "1.0.29":
        raise RuntimeError("This fixture is pinned to timm 1.0.29")
    rng = np.random.default_rng(1729)
    pixels = (rng.standard_normal((2, 256, 4, 4)) * 3).astype(np.float32)
    weight = (1 + 0.1 * rng.standard_normal(256)).astype(np.float32)
    inputs = torch.from_numpy(pixels).bfloat16()
    output = rms_norm2d(inputs, [256], torch.from_numpy(weight).bfloat16(), 1e-6)
    np.savez(args.out, inputs=inputs.float().permute(0, 2, 3, 1).numpy(), weight=weight,
             output=output.float().permute(0, 2, 3, 1).numpy(), timm=timm.__version__,
             torch=torch.__version__)


if __name__ == "__main__":
    main()
