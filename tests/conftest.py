import os

# Tests must run identically on any machine; JAX_PLATFORMS=cuda runs the same
# files on a GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Enough simulated devices to exercise a 4x2 data/fsdp mesh. Must be set before
# jax initialises its backend.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
).strip()
# Parity tests assert fp32 against references computed in fp32. Ampere and
# later GPUs default fp32 matmuls to TF32, a 10-bit mantissa, which puts
# 1e-2 between two correct implementations.
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import jax
import jax.numpy as jnp
import pytest

from dew.telemetry.instrumentation import (
    default_compilation_cache_dir, enable_compilation_cache,
)

# The suite compiles the same kernels every run, on both lanes, in every xdist
# worker. XLA's persistent cache is keyed by the executable, so a second run
# reuses the first one's compilations. DEW_TEST_NO_CACHE=1 measures the cold
# cost, which is what the numbers in docs/performance.md were taken with.
if not os.environ.get("DEW_TEST_NO_CACHE"):
    _cache = default_compilation_cache_dir()
    if _cache:
        enable_compilation_cache(_cache)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def text_context():
    # Shape of the default CLIP-L/14 text context, no need for the actual encoder
    return jnp.ones((2, 77, 768), dtype=jnp.float32)
