"""Probe (not for landing): do jit compiler_options turn XLA:CPU's dot fusions off per program?"""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np


def digest(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()[:12]


def test_probe():
    rng = np.random.default_rng(0)
    for options in ({}, {"xla_cpu_experimental_ynn_fusion_type": "", "xla_cpu_experimental_xnn_fusion_type": ""},
                    {"xla_cpu_experimental_ynn_fusion_type": ""}):
        rng = np.random.default_rng(0)
        found = []
        for m, k, n in [(16, 48, 4), (8, 2048, 8), (16, 8, 8)]:
            left, right = rng.normal(size=(m, k)).astype(np.float32), rng.normal(size=(k, n)).astype(np.float32)
            try:
                found.append(digest(jax.jit(jnp.dot, compiler_options=options)(left, right)))
            except Exception as error:  # noqa: BLE001
                found.append(f"ERR {type(error).__name__}: {str(error)[:80]}")
        print("PROBE options", options, found, flush=True)
