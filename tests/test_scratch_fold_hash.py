import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.kernels import fla_gdn


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("packed", [False, True])
def test_hash(dtype, packed):
    dtype = jnp.dtype(dtype)
    key = jax.random.split(jax.random.key(0), 6)
    B, T, H, HV, K, V = 2, 150, 1, 2, 64, 64
    q = jax.random.normal(key[0], (B, T, H, K)).astype(dtype)
    k = jax.random.normal(key[1], (B, T, H, K))
    k = (k / jnp.linalg.norm(k, axis=-1, keepdims=True)).astype(dtype)
    v = jax.random.normal(key[2], (B, T, HV, V)).astype(dtype)
    g = -jax.nn.softplus(jax.random.normal(key[3], (B, T, HV)))
    beta = jax.nn.sigmoid(jax.random.normal(key[4], (B, T, HV))).astype(dtype)
    h0 = jax.random.normal(key[5], (B, HV, K, V))
    cu = jnp.asarray([0, 40, 150, 230, 300], jnp.int32) if packed else None
    (o, final), res = jax.jit(lambda *a: fla_gdn._rule_fwd(*a, cu, K ** -0.5, 64, True))(q, k, v, g, beta, h0)
    w, h, corrected = res[7]
    grads = jax.jit(jax.grad(lambda *a: jnp.sum(fla_gdn.chunk_gated_delta_rule(
        *a, cu, K ** -0.5, 64, True)[0].astype(jnp.float32) ** 2), argnums=(0, 1, 2, 3, 4)))(q, k, v, g, beta, h0)
    leaves = {"o": o, "final": final, "a": res[5], "w": w, "h": h, "c": corrected,
              **{f"d{n}": x for n, x in zip("qkvgb", grads)}}
    line = " ".join(f"{n}={hashlib.sha256(np.asarray(x).tobytes()).hexdigest()[:12]}" for n, x in leaves.items())
    raise AssertionError(f"HASH|{dtype.name}|{packed}|{line}")
