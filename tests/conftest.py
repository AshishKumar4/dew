import os

# Tests must run identically on any machine; JAX_PLATFORMS=cuda runs the same
# files on a GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Enough simulated devices to exercise a 4x2 data/fsdp mesh. Must be set before
# jax initialises its backend. A test marked `mesh` needs this many devices;
# a run with fewer, such as one GPU, reports it as skipped with both counts.
MESH_DEVICES = 8
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={MESH_DEVICES}"
).strip()
if os.environ["JAX_PLATFORMS"] == "cuda":
    # Exact state and gradient checks require repeatable CUDA reductions.
    os.environ["XLA_FLAGS"] += " --xla_gpu_deterministic_ops=true"
# Parity tests assert fp32 against references computed in fp32. Ampere and
# later GPUs default fp32 matmuls to TF32, a 10-bit mantissa, which puts
# 1e-2 between two correct implementations.
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import jax
import jax.numpy as jnp
import pytest

from dew.telemetry.instrumentation import default_compilation_cache_dir, enable_compilation_cache

# The suite compiles the same kernels every run, on both lanes, in every xdist
# worker. XLA's persistent cache is keyed by the executable, so a second run
# reuses the first one's compilations. DEW_TEST_NO_CACHE=1 measures the cold
# cost; the numbers in docs/performance.md were taken with it set.
if not os.environ.get("DEW_TEST_NO_CACHE"):
    _cache = default_compilation_cache_dir()
    if _cache:
        enable_compilation_cache(_cache)


def pytest_runtest_setup(item):
    if item.get_closest_marker("mesh") and jax.device_count() < MESH_DEVICES:
        pytest.skip(f"needs a {MESH_DEVICES}-device mesh; this run has "
                    f"{jax.device_count()} {jax.default_backend()} device(s)")


@pytest.fixture
def without_deterministic_ops(monkeypatch):
    """The cuda lane runs the whole suite under --xla_gpu_deterministic_ops,
    which Dew refuses cudnn attention under (openxla/xla#46500). XLA read the
    variable when it opened the backend, so taking the flag back out of the
    environment leaves this executable's reductions as they are and lets a
    test reach the cudnn path it is about."""
    kept = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
            if not flag.startswith("--xla_gpu_deterministic_ops")]
    monkeypatch.setenv("XLA_FLAGS", " ".join(kept))


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def text_context():
    # Shape of the default CLIP-L/14 text context, no need for the actual encoder
    return jnp.ones((2, 77, 768), dtype=jnp.float32)
