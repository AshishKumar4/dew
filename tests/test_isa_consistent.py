"""Probe (not for landing): YNNPACK's consistent arithmetic under --xla_cpu_enable_platform_dependent_math."""
import hashlib
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from test_ssd_kernel import case_draw, evaluated, parity_functions

from dew.nn.kernels.ssd import ssd_chunk_scan


def digest(tree):
    return hashlib.sha256(b"".join(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))).hexdigest()[:12]


def timed(fn, *args, repeats=20):
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def test_probe():
    lines = open("/proc/cpuinfo").read().splitlines()
    cpu = [line.split(":")[1].strip() for line in lines if line.startswith(("cpu family", "model\t"))][:2]
    print("PROBE host", cpu, "avx512f" if any(" avx512f" in line for line in lines if line.startswith("flags")) else "avx2",
          "flags", os.environ.get("XLA_FLAGS", ""), flush=True)
    rng = np.random.default_rng(0)
    for m, k, n in [(16, 48, 4), (8, 2048, 8), (16, 8, 8), (64, 8, 8), (128, 512, 64), (2, 3, 5)]:
        left, right = rng.normal(size=(m, k)).astype(np.float32), rng.normal(size=(k, n)).astype(np.float32)
        print(f"PROBE dot {m}x{k}x{n}", digest(jax.jit(jnp.dot)(left, right)), flush=True)
    rows = rng.normal(size=(64, 512)).astype(np.float32)
    print("PROBE sum", digest(jax.jit(lambda r: r.sum(-1))(rows)), "exp", digest(jax.jit(jnp.exp)(rows)), flush=True)
    reference = dict(np.load("tests/fixtures/mamba2/ssd.npz"))
    shape, chunk = (2, 63, 4, 8, 8, 2), 64
    draw = case_draw(reference, "shape", (shape, chunk))
    xla, truth = parity_functions(chunk, shape[1] + shape[3] + shape[4])
    mine = jax.jit(lambda operands, seeded: evaluated(lambda *o: ssd_chunk_scan(*o, chunk, "tpu"), operands, seeded))
    sums = {}
    for seed in range(int(os.environ.get("PROBE_SEEDS", "32"))):
        operands, seeded = draw(seed)
        found, theirs = mine(operands, seeded), xla(operands, seeded)
        if seed < 2:
            print("PROBE ssd", seed, "kernel", digest(found), "scan", digest(theirs), flush=True)
        with jax.enable_x64(True):
            exact = truth(*jax.tree.map(lambda a: np.asarray(a, np.float64), (operands, seeded)))
        for name in exact:
            want = np.asarray(exact[name], np.float64)
            for arm, got in (("kernel", found), ("scan", theirs)):
                sums[name, arm] = sums.get((name, arm), 0.0) + float(np.mean((np.asarray(got[name], np.float64) - want) ** 2))
    print("PROBE ratio", " ".join(f"{name} {np.sqrt(sums[name, 'kernel'] / sums[name, 'scan']):.3f}"
                                 for name in sorted({n for n, _ in sums})), flush=True)
    big = rng.normal(size=(1024, 1024)).astype(np.float32)
    narrow = rng.normal(size=(4096, 64)).astype(np.float32), rng.normal(size=(64, 16)).astype(np.float32)
    print("PROBE time dot1024", f"{timed(jax.jit(jnp.dot), big, big) * 1e3:.2f}ms",
          "narrow", f"{timed(jax.jit(jnp.dot), *narrow) * 1e3:.3f}ms",
          "sum", f"{timed(jax.jit(lambda r: r.sum(-1)), big) * 1e3:.3f}ms",
          "ssd", f"{timed(xla, *draw(0), repeats=10) * 1e3:.2f}ms", flush=True)
