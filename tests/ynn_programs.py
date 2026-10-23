"""Programs that XLA:CPU otherwise compiles to YNNPACK fusions, a dot, a reduction, a dot under vmap and
one with a static argument, for tests/test_reference_error.py to run in two processes (`outputs`).
Run as a script, it saves their outputs under `jax.jit` to the path it is given."""

import sys

import jax
import jax.numpy as jnp
import numpy as np

rng = np.random.default_rng(0)
x, w = rng.normal(size=(16, 48)).astype(np.float32), rng.normal(size=(48, 4)).astype(np.float32)
rows = rng.normal(size=(64, 512)).astype(np.float32)
batch = rng.normal(size=(3, 16, 48)).astype(np.float32)


def powered(x, w, n):
    for _ in range(n):
        x = jnp.tanh(x @ w @ w.T)
    return x


def outputs(jit) -> dict:
    """Each program's output, jitted by `jit`."""
    return {"dot": jit(jnp.dot)(x, w), "sum": jit(lambda r: r.sum(-1))(rows),
            "vmap": jax.vmap(jit(jnp.dot), (0, None))(batch, w),
            "static": jit(powered, static_argnums=2)(x, w, 3)}


if __name__ == "__main__":
    np.savez(sys.argv[1], **{name: np.asarray(value) for name, value in outputs(jax.jit).items()})
