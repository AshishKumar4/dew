"""Debug (not for landing): the cost of a normal draw whose erf_inv runs in float64, rounded once to
float32, against jax.random.normal's float32 erf_inv, at a diffusion noise draw and an initialisation."""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def current(key, shape):
    return jax.random.normal(key, shape, jnp.float32)


def wide(key, shape):
    # jax.random.normal's own path (_normal_real): a uniform in (-1, 1) from the bits, erf_inv, times sqrt 2,
    # with only erf_inv and the scaling in float64.
    lo = np.nextafter(np.float32(-1.0), np.float32(0.0))
    u = jax.random.uniform(key, shape, jnp.float32, lo, 1.0)
    return (jnp.sqrt(2.0) * jax.lax.erf_inv(u.astype(jnp.float64))).astype(jnp.float32)


def timed(fn, key, shape, repeats=20):
    compiled = jax.jit(fn, static_argnums=1)
    compiled(key, shape).block_until_ready()
    times = []
    for index in range(repeats):
        start = time.perf_counter()
        compiled(jax.random.fold_in(key, index), shape).block_until_ready()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


results = {"device": jax.devices()[0].device_kind, "platform": jax.devices()[0].platform}
key = jax.random.key(0)
for name, shape in (("diffusion noise 16x4x128x128", (16, 4, 128, 128)), ("init 8192x8192", (8192, 8192)),
                    ("video noise 2x16x21x90x160", (2, 16, 21, 90, 160))):
    a, b = timed(current, key, shape), timed(wide, key, shape)
    results[name] = {"float32 erf_inv s": a, "float64 erf_inv s": b, "ratio": b / a}
    print(json.dumps({name: results[name]}), flush=True)
print("RESULT", json.dumps(results), flush=True)
