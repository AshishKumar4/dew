"""Sharded cuDNN attention compiles once across processes. jax partitions its cuDNN call through
`custom_partitioning`, whose lowering on several devices names the partitioner by a process address
(`str(id(...))` in the custom call's backend config), which the persistent cache key keeps; Dew runs
the kernel per shard (`dew.nn.attention.per_shard`), where jax lowers no partitioner."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

PROGRAM = """
import json, sys
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from dew.cache import enable_compilation_cache
from dew.nn.attention import scaled_dot_product_attention
from dew.telemetry.instrumentation import compilation_cache_counts, count_compilation_cache
from dew.training._lowering import custom_calls
enable_compilation_cache(sys.argv[1])
count_compilation_cache()
rng = np.random.default_rng(0)
q, k, v = (jnp.asarray(rng.normal(size=(4, 256, 8, 64)), jnp.bfloat16) for _ in range(3))

def loss(q, k, v):
    out = scaled_dot_product_attention(q, k, v, causal=True, implementation='cudnn')
    return jnp.sum(out.astype(jnp.float32) ** 2)

step = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))
whole = step(q, k, v)
mesh = jax.make_mesh((2,), ('data',))
with jax.set_mesh(mesh):
    rows = NamedSharding(mesh, P('data'))
    arrays = [jax.device_put(x, rows) for x in (q, k, v)]
    partitioner = any(call.target == "CustomSPMDPartitioning" for call in custom_calls(step.lower(*arrays)))
    sharded = step(*arrays)
np.save(sys.argv[2], np.stack([np.asarray(x, np.float32) for x in sharded[1]]))
same = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(whole[1], sharded[1]))
print(json.dumps({**compilation_cache_counts(), "partitioner": partitioner,
                  "same": same}))
"""


@pytest.mark.mesh(devices=2)
@pytest.mark.gpu(reason="jax's cuDNN attention runs on a CUDA GPU alone")
def test_a_sharded_cudnn_attention_compiles_once_across_processes(tmp_path, without_deterministic_ops):
    """Two fresh processes train one causal cuDNN attention with its rows over two GPUs against one
    persistent cache: the program holds no partitioner, the second process compiles nothing, both
    take the same gradients, and those match the attention run whole, bit for bit, since every row
    attends alone."""
    reports, gradients = [], []
    for run in range(2):
        # The test's own process holds its share of each GPU; the child takes memory as it needs it.
        done = subprocess.run([sys.executable, "-c", PROGRAM, str(tmp_path / "cache"),
                               str(tmp_path / f"gradients-{run}.npy")], capture_output=True, text=True,
                              timeout=600, env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
        assert done.returncode == 0, done.stderr[-3000:]
        reports.append(json.loads(done.stdout.splitlines()[-1]))
        gradients.append(np.load(tmp_path / f"gradients-{run}.npy"))
    assert not reports[0]["partitioner"] and reports[0]["same"]
    assert reports[0]["train/cache_misses"] >= 1
    assert reports[1]["train/cache_misses"] == 0 and reports[1]["train/cache_hits"] >= 1
    np.testing.assert_array_equal(gradients[0], gradients[1])
