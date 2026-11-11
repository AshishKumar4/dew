"""The test lane's settings, and the environment the suite runs under (`configured`), which
tests/conftest.py sets before any numerical library is imported."""

import os
import subprocess
import sys
from collections.abc import Mapping, MutableMapping

# Enough simulated devices to exercise a 4x2 data/fsdp mesh. A test marked
# `mesh` needs this many devices, or its `devices=`; a run with fewer, such
# as one GPU, reports it as skipped with both counts.
MESH_DEVICES = 8

USERS_ARITHMETIC = "DEW_TEST_USERS_ARITHMETIC"
"""The environment variable that, set to 1, has a lane configure its devices and threads and none of its
arithmetic (`configure_lane`): no matmul precision of the suite's, no wide CPU arithmetic and, on a cuda
lane, none of its deterministic ops (`USERS_GPU_FLAGS`), so a test process computes as a library process
does (`users_process`)."""


DETERMINISTIC_OPS = "--xla_gpu_deterministic_ops=true"
"""Under it a process keeps no picks (`dew._cache.keeps_picks`) and Dew's fused kernels take their ordered
schedules (`dew._devices.deterministic_ops_requested`)."""

REPEATABLE_GPU_FLAGS = (DETERMINISTIC_OPS, "--xla_gpu_autotune_level=0")
"""The cuda lane's flags for steps repeatable across processes. Deterministic
ops order the reductions; autotuning off leaves every cuBLAS and cuDNN call
the configuration it was emitted with, so no process chooses one; and such
a run keeps no stored picks (`dew._cache.keeps_picks`), which XLA and Dew's
kernels would take instead. The lane states nothing of XLA's Triton GEMM
fusions: which backend runs a dot is XLA's choice under these flags, and
Dew's own options for those fusions (`dew.nn.kernels.options`) apply as in
a user's process, which a flag of the lane's would override. Autotuning off
leaves an fp8 convolution's emitted configuration,
which cuDNN refuses on an H100, so that test compiles in its own process
without it (tests/test_quantization.py)."""


USERS_GPU_FLAGS = (f"{DETERMINISTIC_OPS.split('=')[0]}=false", "--xla_gpu_autotune_level=0")
"""A user's process's flags on a cuda lane (`USERS_ARITHMETIC`): XLA's own schedules, its deterministic ops
stated false so the process can show it opened its backend without them (`dew._devices.flags_set`), and
autotuning off as the lane's, so a program compiled ahead for it is the one its run loads."""


def without_flags(flags: str, *names: str) -> str:
    """XLA `flags` without each flag of `names` (`--xla_...`), whatever value it carries: every test that
    runs XLA without one of the lane's flags filters them here."""
    return " ".join(flag for flag in flags.split() if flag.split("=")[0] not in names)


def without_deterministic_ops(flags: str) -> str:
    """XLA `flags` without `DETERMINISTIC_OPS`."""
    return without_flags(flags, DETERMINISTIC_OPS.split("=")[0])


def configure_lane(environ: MutableMapping[str, str]) -> None:
    """Set the environment a lane runs the suite under, before jax opens a backend.

    Tests must run identically on any machine. The CPU lane is the default,
    with MESH_DEVICES simulated devices. A JAX_PLATFORMS that names an
    accelerator, cuda or tpu, alone or in a list, runs the same files there,
    and the CPU backend stays beside it. Host callbacks (jax.debug.callback,
    io_callback) place their operands on a CPU device, float64 references run
    there, since a TPU has no float64, and a host layout pairs every
    accelerator device with a CPU device of its process
    (`dew.distributed.host.companion_mesh`), so that backend holds one device
    per local accelerator device. A cuda lane's steps are made repeatable
    across processes (`REPEATABLE_GPU_FLAGS`), since exact state and
    gradient checks require it; a user's process there runs XLA's own
    schedules (`USERS_GPU_FLAGS`).
    jax.devices() is still the accelerator's.
    """
    environ.setdefault("JAX_PLATFORMS", "cpu")
    platforms = environ["JAX_PLATFORMS"].split(",")
    flags = [environ.get("XLA_FLAGS", "")]
    if "--xla_allow_excess_precision" not in flags[0]:
        # Every rounding the program states, as a run keeps them
        # (`dew._devices.keep_roundings`).
        flags.append("--xla_allow_excess_precision=false")
    local = {"cuda": _local_gpus, "tpu": _local_tpus}
    accelerator = next((name for name in platforms if name in local), None)
    if accelerator is None:
        cpu_devices = MESH_DEVICES
        # XLA:CPU orders the thunks that hold a collective by when each starts, not when it ends
        # (`Resource::kCollectiveCommunicator`, a scheduling edge), and a while loop's thunk returns
        # pending before its inner collective has started: under the default schedule the next loops'
        # collectives then start, their rendezvous block every worker of the pool, and the device the
        # first one waits for never runs (row 5397: a plain JAX program of three such loops on a 2x2 mesh
        # hangs 11 of 12 processes, none with the communicator made a completion dependency; armada job
        # 20261110120401-1a75cf8e; the program is dew-ops upstream/xla_cpu_collective_order_repro.py).
        # The memory-optimized schedule reuses buffers between the loops, whose dependencies keep them
        # apart: tests/test_run_option_matrix.py's staged lm run aborted 3 of 30 runs under the default
        # schedule and 0 of 60 under this one. XLA's CPU backend ships in jaxlib, which Dew takes as
        # released, so the setting is a boundary of that dependency (CONTRIBUTING.md) until a jaxlib
        # release orders CPU collectives by completion.
        flags.append("--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED")
    else:
        if accelerator == "cuda":
            flags.extend(USERS_GPU_FLAGS if environ.get(USERS_ARITHMETIC) == "1" else REPEATABLE_GPU_FLAGS)
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
    # One OpenMP thread, torch's intra-op pool and the host BLAS's: the suite's torch and NumPy arms are
    # references of small ops, which a pool of the machine's cores runs no faster, and whose spinning
    # barriers stall whenever one of its threads is descheduled, by XLA's pool beside it, another xdist
    # worker or a busy host. test_torchax_fallback.py's [bias] case took 39 s on a quiet 4-core container,
    # past 1,500 s beside four busy loops, and 80 s beside them on one thread (ledger 3829). A count a
    # process is given (the GPU lane's share of a node's CPUs, tools/armada/ci.py) stands.
    environ.setdefault("OMP_NUM_THREADS", "1")
    if environ.get(USERS_ARITHMETIC) != "1":
        # Parity tests assert fp32 against references computed in fp32. Ampere and
        # later GPUs default fp32 matmuls to TF32, a 10-bit mantissa, which puts
        # 1e-2 between two correct implementations.
        environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
        # The arithmetic the suite holds a CPU to (`dew.nn.precision.WIDE_CPU_ARITHMETIC`): float32
        # products summed in float64, a loop's running sum of them compensated, and a row's reductions in
        # float64 (`widened_reduction`).
        # Dew's test setting, never the library's default.
        environ.setdefault("DEW_WIDE_CPU_ARITHMETIC", "1")


def configured(environ: Mapping[str, str]) -> dict[str, str]:
    """`environ` as the suite runs under it (`configure_lane`), which tests/conftest.py sets before
    anything imports JAX. Importing this module configures nothing, so a worker process that imports
    the lane's settings keeps the topology its parent gave it."""
    lane = dict(environ)
    configure_lane(lane)
    return lane


def users_process(environ: Mapping[str, str]) -> dict[str, str]:
    """`environ` for a test process that computes on a CPU as a user's does: outside the suite
    (`outside_suite`), then configured by its own conftest with the suite's simulated devices, which a test
    of a mesh needs, and a user's arithmetic (`USERS_ARITHMETIC`). What the suite's own processes cannot
    show: every program a library process builds outside check 1's arms, its shapes, placements, routes
    and footprints (tests/test_users_process.py)."""
    return {**outside_suite(environ), USERS_ARITHMETIC: "1"}


def outside_suite(environ: Mapping[str, str]) -> dict[str, str]:
    """`environ` as a user's process on one CPU device sees it: without every setting `configure_lane` gives
    the suite (its simulated devices and their threads, its one OpenMP thread, its matmul precision, its wide
    CPU arithmetic), JAX on the CPU, for a page, an example or a tool a test runs as a reader would (ledgers
    2510, 2736)."""
    suite = configured({})
    return {**{name: value for name, value in environ.items() if name not in suite}, "JAX_PLATFORMS": "cpu"}


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


def per_test_settings(patch, directory, *, network: bool = False) -> None:
    """A test's own settings, as the suite runs it (tests/conftest.py's `own_settings`) and as a GPU flow
    runs a cell (tests/gpu_flow.py's `isolated`): its own rung records (`dew.training.rungs`: the suite shares
    one compilation cache, beside which a run records its step's rung, and a rung one test forces would be
    another's floor), its own automatic data cache (`dew.data.cache.automatic_root`), its own recorded Hub
    answers (`dew._cache.hub_answer`), and, where it reaches no network, its own Hub cache, whose refs its
    stand-ins' answers write (`dew._cache.hub_commit`), never the runner's. Each is `directory(name)`, set
    through `patch` (a pytest MonkeyPatch, undone after the test)."""
    patch.setenv("DEW_RUNG_RECORDS", str(directory("rungs")))
    patch.setenv("DEW_DATA_CACHE", str(directory("data-cache")))
    patch.setenv("DEW_HUB_ANSWERS", str(directory("hub-answers")))
    if not network:
        from huggingface_hub import constants

        patch.setattr(constants, "HF_HUB_CACHE", str(directory("hub-cache")))
