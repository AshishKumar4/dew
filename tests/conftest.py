import logging
import os

import jax
import jax.numpy as jnp
import pytest
from lane_environment import (
    MESH_DEVICES,
    configure_lane as configure_lane,
    outside_any_cluster as outside_any_cluster,
)

from dew.telemetry.instrumentation import default_compilation_cache_dir, enable_compilation_cache

# The suite compiles the same kernels every run, on both lanes, in every xdist
# worker. XLA's persistent cache is keyed by the executable, so a second run
# reuses the first one's compilations. DEW_TEST_NO_CACHE=1 measures the cold
# cost; the numbers in docs/performance.md were taken with it set.
if not os.environ.get("DEW_TEST_NO_CACHE"):
    _cache = default_compilation_cache_dir()
    if _cache:
        enable_compilation_cache(_cache)

# XLA parses XLA_FLAGS once, at the first compile, not when the backend
# opens. A test that edits the variable, such as `without_deterministic_ops`
# or the apply_xla_flags tests, would otherwise hand its flags to the whole
# run whenever it happens to run first: the cuda lane then loses
# --xla_gpu_deterministic_ops and its bitwise checks see two compilations of
# one forward disagree. Compiling here fixes the flags set above.
jax.jit(lambda x: x + 1)(0).block_until_ready()


def pytest_runtest_setup(item):
    marker = item.get_closest_marker("mesh")
    needed = marker.kwargs.get("devices", MESH_DEVICES) if marker else 0
    if jax.device_count() < needed:
        pytest.skip(f"needs a {needed}-device mesh; this run has "
                    f"{jax.device_count()} {jax.default_backend()} device(s)")


@pytest.fixture
def without_deterministic_ops(monkeypatch):
    """The cuda lane runs the whole suite under --xla_gpu_deterministic_ops,
    which Dew refuses cudnn attention under (openxla/xla#46500). XLA read the
    variable at the compile above, so taking the flag back out of the
    environment leaves this executable's reductions as they are and lets a
    test reach the cudnn path it is about."""
    kept = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
            if not flag.startswith("--xla_gpu_deterministic_ops")]
    monkeypatch.setenv("XLA_FLAGS", " ".join(kept))


@pytest.fixture
def caplog(caplog):
    """Capture Dew's isolated logger as well as the application's root logger."""
    logger = logging.getLogger("dew")
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def text_context():
    # Shape of the default CLIP-L/14 text context, no need for the actual encoder
    return jnp.ones((2, 77, 768), dtype=jnp.float32)
