import gc
import logging
import os

import lane_environment  # configures the backend before JAX reads the environment
import jax
import pytest
import remote_cache

from dew.cache import default_compilation_cache_dir, enable_compilation_cache

# The suite compiles the same kernels every run, on both lanes, in every xdist
# worker. XLA's persistent cache is keyed by the executable, so a second run
# reuses the first one's compilations. DEW_TEST_NO_CACHE=1 measures the cold
# cost; the numbers in docs/performance.md were taken with it set.
_remote = None
if not os.environ.get("DEW_TEST_NO_CACHE"):
    _cache = default_compilation_cache_dir()
    if _cache:
        enable_compilation_cache(_cache)
        # On armada, where each task's container starts cold, a cache the tasks share.
        _remote = remote_cache.install_from_environment()

# XLA parses XLA_FLAGS once, at the first compile, not when the backend
# opens. A test that edits the variable, such as `without_deterministic_ops`
# or the apply_xla_flags tests, would otherwise hand its flags to the whole
# run whenever it happens to run first: the cuda lane then loses
# --xla_gpu_deterministic_ops and its bitwise checks see two compilations of
# one forward disagree. Compiling here fixes the flags set above.
jax.jit(lambda x: x + 1)(0).block_until_ready()


def pytest_sessionfinish(session, exitstatus):
    if _remote is not None:
        _remote.drain()


def pytest_terminal_summary(terminalreporter):
    if _remote is not None:
        terminalreporter.write_line(_remote.summary())


def pytest_runtest_setup(item):
    marker = item.get_closest_marker("mesh")
    needed = marker.kwargs.get("devices", lane_environment.MESH_DEVICES) if marker else 0
    if jax.device_count() < needed:
        pytest.skip(f"needs a {needed}-device mesh; this run has "
                    f"{jax.device_count()} {jax.default_backend()} device(s)")


@pytest.fixture(autouse=True)
def own_rung_records(monkeypatch, tmp_path_factory):
    """Each test keeps its own rung records (`dew.training.rungs`). The suite
    shares one compilation cache, beside which a run records its step's
    rung, and a rung one test forces would be another's floor."""
    from dew.training import rungs

    directory = []

    def records():
        if not directory:
            directory.append(tmp_path_factory.mktemp("rungs"))
        return directory[0]

    monkeypatch.setattr(rungs, "rung_records", records)


@pytest.fixture(autouse=True, scope="module")
def release_compilations():
    """Each test file's compiled executables are released when it ends.

    XLA:CPU maps every executable's code into the process, and JAX keeps an
    executable for as long as its jit cache does. A run that goes on through
    many files reached 65,391 mappings, past the 65,530 a Linux kernel allows
    by default, and the next compile failed with "Failed to materialize
    symbols". Ubuntu 24.04 allows 1,048,576, so GitHub's runners never saw
    it; Debian and most kernels do.
    """
    yield
    jax.clear_caches()
    gc.collect()


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
