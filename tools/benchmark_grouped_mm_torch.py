#!/usr/bin/env python3
"""torch's grouped GEMM at tools/benchmark_kernels.py projection's shapes: the bar for Dew's grouped matmul.

    python tools/benchmark_grouped_mm_torch.py --rows 8192 --experts 8 --width 768 --features 2048 \\
        [--shape up|down]

`torch._grouped_mm` (torch 2.14) over bf16 operands, rows sorted by expert with the same Dirichlet
group sizes and seed as tools/benchmark_kernels.py's `projection`, forward and forward+backward
(input and weight gradients), median of `--repeats` synchronized calls after warmup.
"""
import argparse
import json
import time

import numpy as np
import torch


def timed(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return {"median_ms": samples[len(samples) // 2], "min_ms": samples[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--shape", choices=("up", "down"), default="up")
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    rows, experts = args.rows, args.experts
    width, features = (args.width, args.features) if args.shape == "up" else (args.features, args.width)
    rng = np.random.default_rng(0)
    sizes = np.floor(rng.dirichlet(np.ones(experts)) * rows).astype(np.int32)
    sizes[-1] += rows - sizes.sum()
    x = torch.tensor(rng.normal(size=(rows, width)), dtype=torch.bfloat16, device="cuda")
    kernel = torch.tensor(rng.normal(size=(experts, width, features)) / np.sqrt(width), dtype=torch.bfloat16,
                          device="cuda")
    cotangent = torch.tensor(rng.normal(size=(rows, features)), dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(np.cumsum(sizes), dtype=torch.int32, device="cuda")

    def forward():
        return torch._grouped_mm(x, kernel, offs=offsets)

    xs, ks = x.detach().requires_grad_(), kernel.detach().requires_grad_()

    def both():
        out = torch._grouped_mm(xs, ks, offs=offsets)
        out.backward(cotangent)

    result = {"impl": "torch._grouped_mm", "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "shape": [rows, width, features, experts]}
    result["forward"] = timed(forward, args.repeats)
    result["forward_backward"] = timed(both, args.repeats)
    print(json.dumps(result))


main()
