#!/usr/bin/env python3
"""Measure the interpreted SSD kernel's check-1 ratio against XLA's scan, with and without XLA:CPU's
YNNPACK fusions, on the host it runs on.

Reference environment: JAX/jaxlib 0.11.2. Run from the checkout whose source is being measured:

    JAX_PLATFORMS=cpu PYTHONPATH=src:tests python tools/ssd_ynn_probe.py [--draws 128] [--fused]

For each host case of tests/test_ssd_kernel.py it prints, per leaf, the two arms' pooled RMS ratio
from the float64 truth over the draws, sqrt(sum of the kernel's mean square errors over the scan's),
the statistic check 1 pools; `--fused` compiles both arms as `jax.jit` does, with the fusions. It
prints the host's CPU family and model, which docs/performance.md's figures name.
"""

import argparse
from pathlib import Path

import jax
import numpy as np
from test_ssd_kernel import CASES, FIXTURES, case_draw, evaluated, parity_functions

from dew.nn.kernels.ssd import ssd_chunk_scan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    lines = Path("/proc/cpuinfo").read_text().splitlines()
    cpu = {key.strip(): value.strip() for key, value in (line.split(":", 1) for line in lines
                                                          if line.startswith(("cpu family", "model\t")))}
    avx512 = any(line.startswith("flags") and " avx512f" in line for line in lines)
    print("cpu", cpu, "avx512f" if avx512 else "no avx512f", "draws", args.draws,
          "fused" if args.fused else "without YNN", flush=True)
    reference = dict(np.load(FIXTURES / "ssd.npz"))
    for param in CASES:
        kind, case, compiled = param.values
        if compiled:
            continue
        if kind == "fixture":
            chunk = case[0]
            _, length, _, width = reference["scan.x"].shape
            roundings = length + width + reference["scan.B"].shape[-1]
        else:
            (_, length, _, width, state, _), chunk = case
            roundings = length + width + state
        mine, xla, truth = parity_functions(chunk, roundings)
        if args.fused:
            mine = jax.jit(lambda operands, seeded, chunk=chunk: evaluated(
                lambda *o: ssd_chunk_scan(*o, chunk, "tpu"), operands, seeded))
            xla = jax.jit(xla.__wrapped__)
        draw, sums = case_draw(reference, kind, case), {}
        for seed in range(args.draws):
            operands, seeded = draw(seed)
            found, theirs = mine(operands, seeded), xla(operands, seeded)
            with jax.enable_x64(new_val=True):
                exact = truth(*jax.tree.map(lambda a: np.asarray(a, np.float64), (operands, seeded)))
            for name, want in exact.items():
                want = np.asarray(want, np.float64)
                for arm, got in (("kernel", found), ("scan", theirs)):
                    sums[name, arm] = sums.get((name, arm), 0.0) + float(
                        np.mean((np.asarray(got[name], np.float64) - want) ** 2))
        ratios = {name: np.sqrt(sums[name, "kernel"] / sums[name, "scan"]) for name, _ in sums}
        print(param.id, " ".join(f"{name} {ratio:.3f}" for name, ratio in ratios.items()), flush=True)


if __name__ == "__main__":
    main()
