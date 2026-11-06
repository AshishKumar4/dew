"""Hold M and K fixed while sweeping the YNN dot's output width."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import lane_environment

os.environ.update(lane_environment.configured(os.environ))
import jax
import jax.numpy as jnp
import numpy as np

rng = np.random.default_rng(423)
a = rng.standard_normal((256, 3, 8)).astype(np.float32)
for columns in (4, 8, 16, 17, 32, 64, 128, 256):
    b = rng.standard_normal((256, 8, columns)).astype(np.float32)
    output = np.asarray(jax.jit(lambda x, y: jnp.einsum(
        "btd,bdf->btf", x, y, precision=jax.lax.Precision.HIGHEST))(a, b))
    fma = np.zeros(output.shape, np.float32)
    plain = np.zeros(output.shape, np.float32)
    for k in range(8):
        term = a[..., k, None].astype(np.float64) * b[:, None, k, :].astype(np.float64)
        fma = (fma.astype(np.float64) + term).astype(np.float32)
        plain = (plain + term.astype(np.float32)).astype(np.float32)
    print("WIDTH", columns, "ELEMENTS", output.size, "FMA", int(np.sum(output == fma)),
          "MUL_ADD", int(np.sum(output == plain)), flush=True)
