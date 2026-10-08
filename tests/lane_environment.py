"""Configure test-lane environment before any numerical library is imported."""

import os
import subprocess
import sys
from collections.abc import Mapping, MutableMapping

# Enough simulated devices to exercise a 4x2 data/fsdp mesh. A test marked
# `mesh` needs this many devices, or its `devices=`; a run with fewer, such
# as one GPU, reports it as skipped with both counts.
MESH_DEVICES = 8


REPEATABLE_GPU_FLAGS = ("--xla_gpu_deterministic_ops=true", "--xla_gpu_autotune_level=0")
"""The cuda lane's flags for steps repeatable across processes. Deterministic
ops order the reductions. Autotuning off is a precaution: XLA picks GEMM and
convolution kernels at compile time by live timing, which can differ between
compilations (openxla.org/xla/determinism). Two LADD runs on an RTX 4080
diverged under deterministic ops alone, but 20 fresh DiT processes did not,
so that cause is unconfirmed. The precaution costs a test lane nothing that
matters; in training it cost 8% on a 176M DiT step (docs/guides/checkpoints.md)."""


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
    per local accelerator device. A cuda lane's steps are made repeatable
    across processes (`REPEATABLE_GPU_FLAGS`), since exact state and
    gradient checks require it.
    jax.devices() is still the accelerator's.
    """
    environ.setdefault("JAX_PLATFORMS", "cpu")
    platforms = environ["JAX_PLATFORMS"].split(",")
    flags = [environ.get("XLA_FLAGS", "")]
    if "--xla_allow_excess_precision" not in flags[0]:
        # Every rounding the program states, as a run keeps them
        # (`dew.telemetry.devices.keep_roundings`).
        flags.append("--xla_allow_excess_precision=false")
    local = {"cuda": _local_gpus, "tpu": _local_tpus}
    accelerator = next((name for name in platforms if name in local), None)
    if accelerator is None:
        cpu_devices = MESH_DEVICES
    else:
        if accelerator == "cuda":
            flags.extend(REPEATABLE_GPU_FLAGS)
        if "cpu" not in platforms:
            environ["JAX_PLATFORMS"] = ",".join([*platforms, "cpu"])
        cpu_devices = local[accelerator](environ)
    flags.append(f"--xla_force_host_platform_device_count={cpu_devices}")
    environ["XLA_FLAGS"] = " ".join(flags).strip()
    # XLA:CPU runs every device of a launch on one pool of max(cores,
    # devices) threads (PJRT_NPROC overrides the cores), each held until the
    # launch's collectives meet, and dispatches the next launch on the same
    # pool, where a device with 32 computations in flight blocks its thread.
    # With a thread per device, a 4-core runner's 8, a loop that ran ahead
    # left the launch it waited on one device short, and the rendezvous
    # aborted the process (tests/test_discrete.py's toy run on CI, three
    # times). Two threads per device hold both launches.
    environ.setdefault("PJRT_NPROC", str(max(len(os.sched_getaffinity(0)), 2 * cpu_devices)))


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
    Cloud TPU VM worker list, no Kubernetes pod, no pool a `dew launch`
    declared, and TPU_SKIP_MDS_QUERY, jax's switch for a host whose metadata
    names no TPU cluster (the variables `jax._src.clusters` reads). A Colab
    TPU VM otherwise shows jax a TPU cluster of one process ahead of any of
    them."""
    kept = {name: value for name, value in environ.items()
            if not name.startswith(("SLURM_", "OMPI_"))
            and name not in ("TPU_WORKER_HOSTNAMES", "TPU_PROCESS_ADDRESSES",
                             "TPU_PROCESS_ADDRESSES_PATH", "KUBERNETES_SERVICE_HOST",
                             "DEW_PROCESS_COUNT", "DEW_PROCESS_ID", "JAX_COORDINATOR_ADDRESS")}
    return {**kept, "TPU_SKIP_MDS_QUERY": "1"}


configure_lane(os.environ)
# Parity tests assert fp32 against references computed in fp32. Ampere and
# later GPUs default fp32 matmuls to TF32, a 10-bit mantissa, which puts
# 1e-2 between two correct implementations.
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
