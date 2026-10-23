"""Warm compiled backward timing at the production geometry; run on armada, not locally."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from probe_aligned_chunk_adjoint_nosysquid import ORIGINAL, chunk_adjoint


@pytest.mark.parametrize("variant", ["current", "chunk-adjoint"])
def test_backward_wall_time(variant):
    fn = {"current": ORIGINAL, "chunk-adjoint": chunk_adjoint}[variant]
    shape = (2, 4096, 16, 128)
    rng = jax.random.split(jax.random.key(20261023), 7)
    q, k, v = (jax.random.normal(key, shape) for key in rng[:3])
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    g = -jax.nn.softplus(jax.random.normal(rng[3], shape[:-1] + (1,)) - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(rng[4], shape[:-1]))
    state = jnp.zeros((2, 16, 128, 128), jnp.float32)
    cotangent = (jax.random.normal(rng[5], shape), jax.random.normal(rng[6], state.shape))
    inputs = q, k, v, g, beta, state

    @jax.jit
    def backward(*inputs):
        _, pull = jax.vjp(lambda *xs: fn(*xs, chunk_size=64), *inputs)
        return pull(cotangent)

    start = time.monotonic()
    jax.block_until_ready(backward(*inputs))
    cold = time.monotonic() - start
    times = []
    for _ in range(5):
        start = time.monotonic()
        jax.block_until_ready(backward(*inputs))
        times.append(time.monotonic() - start)
    print(f"BENCH|{variant}|platform={jax.default_backend()}|shape={shape}|chunk=64"
          f"|seconds={np.median(times):.9g}|samples={times}|cold={cold:.6g}", flush=True)
