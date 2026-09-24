import os
import subprocess
import sys
from collections.abc import Mapping, MutableMapping

# Enough simulated devices to exercise a 4x2 data/fsdp mesh. A test marked
# `mesh` needs this many devices, or its `devices=`; a run with fewer, such
# as one GPU, reports it as skipped with both counts.
MESH_DEVICES = 8


def configure_lane(environ: MutableMapping[str, str]) -> None:
    """Set the environment a lane runs the suite under, before jax opens a backend.

    Tests must run identically on any machine. The CPU lane is the default,
    with MESH_DEVICES simulated devices. A JAX_PLATFORMS that names an
    accelerator, cuda or tpu, alone or in a list, runs the same files there,
    and the CPU backend stays beside it. Host callbacks (jax.debug.callback,
    io_callback) place their operands on a CPU device, float64 references run
    there, since a TPU has no float64, and a host layout pairs every
    accelerator device with a CPU device of its process
    (`dew.training.host.companion_mesh`), so that backend holds one device
    per local accelerator device. A cuda lane's reductions are made
    repeatable, since exact state and gradient checks require it.
    jax.devices() is still the accelerator's.
    """
    environ.setdefault("JAX_PLATFORMS", "cpu")
    platforms = environ["JAX_PLATFORMS"].split(",")
    flags = [environ.get("XLA_FLAGS", "")]
    local = {"cuda": _local_gpus, "tpu": _local_tpus}
    accelerator = next((name for name in platforms if name in local), None)
    if accelerator is None:
        flags.append(f"--xla_force_host_platform_device_count={MESH_DEVICES}")
    else:
        if accelerator == "cuda":
            flags.append("--xla_gpu_deterministic_ops=true")
        if "cpu" not in platforms:
            environ["JAX_PLATFORMS"] = ",".join([*platforms, "cpu"])
        flags.append(f"--xla_force_host_platform_device_count={local[accelerator](environ)}")
    environ["XLA_FLAGS"] = " ".join(flags).strip()


def _local_gpus(environ: MutableMapping[str, str]) -> int:
    """The GPUs this process will see, counted before any backend opens."""
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([device for device in visible.split(",") if device.strip()])
    listing = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, check=True)
    return len(listing.stdout.splitlines())


# The TPU chips on this machine's PCI bus, as jax's own startup counts them,
# and the TensorCores a chip of that generation shows as devices: two on v2
# and v3, one since v4's megacore.
_TPU_PROBE = ("from jax._src import hardware_utils as h\n"
              "chips, version = h.num_available_tpu_chips_and_device_id()\n"
              "print(chips, 2 if version in (h.TpuVersion.v2, h.TpuVersion.v3) else 1)\n")


def _local_tpus(environ: MutableMapping[str, str]) -> int:
    """The TPU devices this process will see, counted before any backend
    opens: the chips TPU_VISIBLE_CHIPS names, else every one, times a
    chip's devices. A child process asks jax, since importing it here would
    fix its configuration before this function sets it."""
    probe = subprocess.run([sys.executable, "-c", _TPU_PROBE], capture_output=True, text=True,
                           check=True, env={**environ, "JAX_PLATFORMS": "cpu"})
    chips, per_chip = (int(word) for word in probe.stdout.split())
    visible = environ.get("TPU_VISIBLE_CHIPS")
    if visible is not None:
        chips = len([chip for chip in visible.split(",") if chip.strip()])
    return chips * per_chip


def outside_any_cluster(environ: Mapping[str, str]) -> dict[str, str]:
    """`environ` as a process on a machine in no cluster jax detects sees it,
    for a program the test places itself or a launch that stands in for a
    cluster with variables of its own: no Slurm or Open MPI variables, no
    Cloud TPU VM worker list, no Kubernetes pod, and TPU_SKIP_MDS_QUERY,
    jax's switch for a host whose metadata names no TPU cluster (the
    variables `jax._src.clusters` reads). A Colab TPU VM otherwise shows
    jax a TPU cluster of one process ahead of any of them."""
    kept = {name: value for name, value in environ.items()
            if not name.startswith(("SLURM_", "OMPI_"))
            and name not in ("TPU_WORKER_HOSTNAMES", "TPU_PROCESS_ADDRESSES",
                             "TPU_PROCESS_ADDRESSES_PATH", "KUBERNETES_SERVICE_HOST")}
    return {**kept, "TPU_SKIP_MDS_QUERY": "1"}


configure_lane(os.environ)
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
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def text_context():
    # Shape of the default CLIP-L/14 text context, no need for the actual encoder
    return jnp.ones((2, 77, 768), dtype=jnp.float32)
