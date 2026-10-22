"""Probe (not for landing): which arm of the interpreted SSD kernel's check 1 moves with the host's ISA."""
import hashlib
import os

import jax
import jax.numpy as jnp
import numpy as np
from test_ssd_kernel import case_draw, evaluated, parity_functions

from dew.nn.kernels.ssd import ssd_chunk_scan


def digest(tree):
    return hashlib.sha256(b"".join(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))).hexdigest()[:12]


def test_probe():
    print("PROBE XLA_FLAGS", os.environ.get("XLA_FLAGS", ""), flush=True)
    model = [line for line in open("/proc/cpuinfo").read().splitlines() if line.startswith(("model\t", "cpu family", "flags"))]
    print("PROBE cpu", [line.split(":")[1].strip()[:6] for line in model[:2]], "avx512f" in model[2], flush=True)
    rng = np.random.default_rng(0)
    for m, k, n in [(16, 48, 4), (8, 2048, 8), (16, 8, 8), (64, 8, 8), (8, 64, 8)]:
        left, right = rng.normal(size=(m, k)).astype(np.float32), rng.normal(size=(k, n)).astype(np.float32)
        print(f"PROBE dot {m}x{k}x{n}", digest(jax.jit(jnp.dot)(left, right)), flush=True)
    reference = dict(np.load("tests/fixtures/mamba2/ssd.npz"))
    shape, chunk = (2, 63, 4, 8, 8, 2), 64
    draw = case_draw(reference, "shape", (shape, chunk))
    xla, truth = parity_functions(chunk, shape[1] + shape[3] + shape[4])
    mine = jax.jit(lambda operands, seeded: evaluated(lambda *o: ssd_chunk_scan(*o, chunk, "tpu"), operands, seeded))
    sums = {}
    for seed in range(int(os.environ.get("PROBE_SEEDS", "64"))):
        operands, seeded = draw(seed)
        found, theirs = mine(operands, seeded), xla(operands, seeded)
        with jax.enable_x64(True):
            exact = truth(*jax.tree.map(lambda a: np.asarray(a, np.float64), (operands, seeded)))
        if seed == 0:
            print("PROBE digest kernel", digest(found), "xla", digest(theirs), "truth", digest(exact), flush=True)
        for name in exact:
            want = np.asarray(exact[name], np.float64)
            for arm, got in (("kernel", found), ("xla", theirs)):
                sums.setdefault((name, arm), 0.0)
                sums[(name, arm)] += float(np.mean((np.asarray(got[name], np.float64) - want) ** 2))
    for name in sorted({name for name, _ in sums}):
        k, x = sums[(name, "kernel")], sums[(name, "xla")]
        print(f"PROBE leaf {name:10s} kernel {k:.4e} xla {x:.4e} ratio {np.sqrt(k / x):.3f}", flush=True)
