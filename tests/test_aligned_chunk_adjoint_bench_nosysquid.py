"""Warm compiled backward timing at the production geometry; run on armada, not locally."""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from probe_aligned_chunk_adjoint_nosysquid import ORIGINAL, chunk_adjoint, chunk_adjoint_wy_source
from dew.nn.attention import forward_mode_attention
from dew.nn.kernels import recurrence_chunk


def unregrouped(*args, **kwargs):
    with forward_mode_attention():
        return recurrence_chunk.chunk_delta_rule(*args, **kwargs)


@pytest.mark.parametrize("variant", ["current", "chunk-adjoint", "chunk-adjoint-fast",
                                    "chunk-adjoint-uncheckpointed", "comp-core",
                                    "chunk-adjoint-g4", "chunk-adjoint-g8", "chunk-adjoint-g16"])
def test_backward_wall_time(variant):
    if variant.startswith("chunk-adjoint-g"):
        recurrence_chunk.ADJOINT_CHUNKS = int(variant.removeprefix("chunk-adjoint-g"))
    if variant == "comp-core":
        import probe_aligned_dots_nosysquid as dots
        from probe_aligned_compensated_nosysquid import dot
        from test_aligned_precision_nosysquid import CORE
        os.environ["DEW_PROBE_REDUCE"] = "1"
        dots.wide_dot = dot
        dots.install(CORE)
    fn = recurrence_chunk.chunk_delta_rule if variant.startswith("chunk-adjoint-g") else {
          "current": unregrouped, "chunk-adjoint": chunk_adjoint,
          "chunk-adjoint-fast": recurrence_chunk.chunk_delta_rule,
          "chunk-adjoint-uncheckpointed": chunk_adjoint_wy_source, "comp-core": unregrouped}[variant]
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
    compiled = backward.lower(*inputs).compile()
    memory = compiled.memory_analysis()
    peak = (memory.argument_size_in_bytes + memory.output_size_in_bytes
            + memory.temp_size_in_bytes - memory.alias_size_in_bytes)
    print(f"MEMORY|{variant}|peak={peak}|arguments={memory.argument_size_in_bytes}"
          f"|outputs={memory.output_size_in_bytes}|temporary={memory.temp_size_in_bytes}"
          f"|aliases={memory.alias_size_in_bytes}", flush=True)
    jax.block_until_ready(compiled(*inputs))
    cold = time.monotonic() - start
    times = []
    for _ in range(3 if variant == "comp-core" else 5):
        start = time.monotonic()
        jax.block_until_ready(compiled(*inputs))
        times.append(time.monotonic() - start)
    print(f"BENCH|{variant}|platform={jax.default_backend()}|shape={shape}|chunk=64"
          f"|seconds={np.median(times):.9g}|samples={times}|cold={cold:.6g}", flush=True)
