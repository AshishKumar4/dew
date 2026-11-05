"""Whether XLA serves a decode cache kept in pinned host memory the way ledger 1343 needs:
a layer scan that fetches each layer's slice to the device and writes it back to host, inside a
decode while_loop that carries the host buffers, compared against the same program on the device."""
import json
import sys
import time

import jax
import jax.numpy as jnp
from jax.sharding import SingleDeviceSharding

LAYERS, ROWS, CAP, HEADS, DIM, STEPS = 8, 8, 2048, 8, 128, 16
dev = jax.devices()[0]
host = SingleDeviceSharding(dev, memory_kind="pinned_host")
devs = SingleDeviceSharding(dev, memory_kind="device")


def layer(x, k, v, pos):
    # The new key and value in the cache's own dtype, as a decode writes them.
    written = x[:, None, :HEADS * DIM].reshape(ROWS, 1, HEADS, DIM).astype(k.dtype)
    k = jax.lax.dynamic_update_index_in_dim(k, written, pos, 1)
    v = jax.lax.dynamic_update_index_in_dim(v, written, pos, 1)
    s = jnp.einsum("rhd,rchd->rhc", x[:, :HEADS * DIM].reshape(ROWS, HEADS, DIM), k)
    out = jnp.einsum("rhc,rchd->rhd", jax.nn.softmax(s, -1), v).reshape(ROWS, HEADS * DIM)
    return jnp.tanh(out + x), k, v


def step(x, ks, vs, pos, offload):
    def body(carry, i):
        x, ks, vs = carry
        k = jax.lax.dynamic_index_in_dim(ks, i, 0, keepdims=False)
        v = jax.lax.dynamic_index_in_dim(vs, i, 0, keepdims=False)
        if offload:
            k, v = jax.device_put(k, jax.memory.Space.Device), jax.device_put(v, jax.memory.Space.Device)
        x, k, v = layer(x, k, v, pos)
        if offload:
            k, v = jax.device_put(k, jax.memory.Space.Host), jax.device_put(v, jax.memory.Space.Host)
        ks = jax.lax.dynamic_update_index_in_dim(ks, k, i, 0)
        vs = jax.lax.dynamic_update_index_in_dim(vs, v, i, 0)
        return (x, ks, vs), None
    (x, ks, vs), _ = jax.lax.scan(body, (x, ks, vs), jnp.arange(LAYERS))
    return x, ks, vs


def generate(x, ks, vs, offload):
    def cond(c):
        return c[3] < STEPS
    def body(c):
        x, ks, vs, pos = c
        x, ks, vs = step(x, ks, vs, pos, offload)
        return x, ks, vs, pos + 1
    return jax.lax.while_loop(cond, body, (x, ks, vs, jnp.int32(0)))[:3]


report = {"device": str(dev), "jax": jax.__version__}
shape = (LAYERS, ROWS, CAP, HEADS, DIM)
x = jax.random.normal(jax.random.key(0), (ROWS, HEADS * DIM), jnp.float32)
results = {}
for offload in (False, True):
    place = host if offload else devs
    ks = jax.device_put(jnp.zeros(shape, jnp.bfloat16), place)
    vs = jax.device_put(jnp.zeros(shape, jnp.bfloat16), place)
    try:
        f = jax.jit(lambda x, ks, vs: generate(x, ks, vs, offload), donate_argnums=(1, 2),
                    out_shardings=(devs, place, place))
        out = f(x, ks, vs)
        jax.block_until_ready(out)
        ks = jax.device_put(jnp.zeros(shape, jnp.bfloat16), place)
        vs = jax.device_put(jnp.zeros(shape, jnp.bfloat16), place)
        t = time.perf_counter()
        out = f(x, ks, vs)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t
        mem = dev.memory_stats() or {}
        results[str(offload)] = {"seconds": dt, "kinds": [o.sharding.memory_kind for o in out],
                                 "peak_bytes": mem.get("peak_bytes_in_use"), "x": out[0].astype(jnp.float32).sum().item()}
    except Exception as error:  # the probe reports what XLA refuses
        results[str(offload)] = {"error": f"{type(error).__name__}: {str(error)[:2000]}"}
report["results"] = results
report["cache_bytes"] = 2 * LAYERS * ROWS * CAP * HEADS * DIM * 2
print(json.dumps(report, indent=1))
json.dump(report, open(sys.argv[1] if len(sys.argv) > 1 else "probe.json", "w"), indent=1)
