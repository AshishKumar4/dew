"""Run the all-to-all sequence exchange around the real TPU splash kernel.

tests/test_sequence_parallel.py runs splash inside the exchange under pallas's
interpreter. This runs the Mosaic kernel on a TPU instead, through
`exchanged_heads_attention` on the devices this host has: the shard_map, its
all-to-alls and the kernel inside compile and run together, forward and
backward, and agree with XLA's attention in fp32. With one chip the sequence
axis has one shard, so the all-to-alls move nothing; the lowering is what one
chip can prove. With a chip count the heads divide, the sequence axis takes
every chip and the exchange is real.

Prints one JSON line per case: the largest output and gradient differences
against XLA's attention and against whole-sequence splash, and the median
backward time of the exchange and of whole-sequence splash. Run it at the
default matmul precision: splash's Mosaic matmul does not compile fp32
operands under JAX_DEFAULT_MATMUL_PRECISION=highest on jax 0.11.1.
"""

import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.attention import attention_kernel, exchanged_heads_attention
from dew.training import MeshSpec, build_mesh


def timed(fn, *args, repeats: int = 10) -> float:
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def case(batch: int, seq: int, heads: int, kv_heads: int, head_dim: int, dtype) -> dict:
    shards = jax.device_count() if heads % jax.device_count() == 0 else 1
    keys = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(keys[0], (batch, seq, heads, head_dim), dtype)
    key = jax.random.normal(keys[1], (batch, seq, kv_heads, head_dim), dtype)
    value = jax.random.normal(keys[2], (batch, seq, kv_heads, head_dim), dtype)
    splash = functools.partial(attention_kernel, implementation='tpu')
    xla = functools.partial(attention_kernel, implementation='xla')

    def exchanged(q, k, v):
        return exchanged_heads_attention(splash, q, k, v, shards, causal=True,
                                         sliding_window=None, mask=None, bias=None, sinks=None)

    def whole(kernel, q, k, v):
        return kernel(q, k, v, causal=True, sliding_window=None, mask=None, bias=None)

    def loss(fn):
        return lambda q, k, v: jnp.sum(fn(q, k, v).astype(jnp.float32) ** 2)

    mesh = build_mesh(MeshSpec(sequence=shards))
    with jax.set_mesh(mesh):
        forward = jax.jit(exchanged)
        backward = jax.jit(jax.grad(loss(exchanged), argnums=(0, 1, 2)))
        out = forward(query, key, value)
        grads = backward(query, key, value)
        exchange_time = timed(backward, query, key, value)
    reference = functools.partial(whole, xla)
    want = jax.jit(reference)(query, key, value)
    want_grads = jax.jit(jax.grad(loss(reference), argnums=(0, 1, 2)))(query, key, value)
    plain = functools.partial(whole, splash)
    splash_backward = jax.jit(jax.grad(loss(plain), argnums=(0, 1, 2)))
    plain_grads = splash_backward(query, key, value)

    def gap(a, b):
        return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))

    return {
        "device": jax.devices()[0].device_kind, "devices": jax.device_count(),
        "sequence_shards": shards, "dtype": jnp.dtype(dtype).name,
        "shape": [batch, seq, heads, kv_heads, head_dim],
        "output_gap": gap(out, want),
        "gradient_gap": max(gap(a, b) for a, b in zip(grads, want_grads, strict=True)),
        "gap_to_whole_splash": gap(out, jax.jit(plain)(query, key, value)),
        "gradient_gap_to_whole_splash": max(
            gap(a, b) for a, b in zip(grads, plain_grads, strict=True)),
        "gradient_scale": max(float(jnp.max(jnp.abs(g.astype(jnp.float32)))) for g in want_grads),
        "exchange_backward_s": exchange_time,
        "splash_backward_s": timed(splash_backward, query, key, value),
    }


def main() -> None:
    assert jax.default_backend() == 'tpu', jax.default_backend()
    for dtype in (jnp.float32, jnp.bfloat16):
        print(json.dumps(case(2, 2048, 8, 2, 128, dtype)), flush=True)


if __name__ == "__main__":
    main()
