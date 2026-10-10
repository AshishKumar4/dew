"""The extra XLA flags a run hands to the backend, and what the CUDA driver
says of a GPU.

XLA reads XLA_FLAGS once, when it opens a backend, so a run's flags have to
reach the environment before the first JAX call of the process.
"""

import ctypes
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence

from dew.pool import runs_on_gpu

_log = logging.getLogger(__name__)
_late_policy_warned = False


def apply_xla_flags(flags: str | None) -> None:
    """Append flags to XLA_FLAGS, which XLA reads when it initializes a backend.

    The flags are appended because the environment may already carry some
    (CI sets the host device count). Only useful before the first JAX call;
    `dew.training.prepare_process` calls it there.
    """
    if not flags:
        return
    existing = os.environ.get('XLA_FLAGS', '')
    os.environ['XLA_FLAGS'] = f"{existing} {flags}".strip()


def xla_flag(name: str) -> str | None:
    """The value `--<name>` carries in XLA_FLAGS, or None when it is absent.

    A bare `--<name>` reads as 'true' and the last occurrence wins, which is
    how XLA's own parser resolves a repeated flag. XLA reads the variable
    when it initializes a backend, so this reports what the run asked for,
    not what a live backend was built with.
    """
    value = None
    for token in os.environ.get('XLA_FLAGS', '').split():
        if token == f"--{name}":
            value = 'true'
        elif token.startswith(f"--{name}="):
            value = token.split('=', 1)[1]
    return value


def keep_roundings() -> None:
    """Keep declared narrow-dtype roundings unless the caller names XLA's policy.

    Fusion must not normalize an unrounded FP32 value where the program
    produces a BF16 sum. `import dew` applies it before the process's first
    JAX computation: XLA reads these flags when its backend opens. In an
    already-used notebook, restart with
    `XLA_FLAGS=--xla_allow_excess_precision=false` set before importing JAX.
    """
    global _late_policy_warned
    # Private JAX query pinned by jax<0.11.3; the fresh-process late-import
    # test covers this path without opening a backend just to inspect it.
    bridge = sys.modules.get("jax._src.xla_bridge")
    if bridge is not None and bridge.backends_are_initialized():
        if not _late_policy_warned:
            _late_policy_warned = True
            _log.warning(
                "Dew was imported after the JAX backend opened; its numerical policy cannot take effect. "
                "Restart with XLA_FLAGS=--xla_allow_excess_precision=false set before importing JAX.")
        return
    if xla_flag("xla_allow_excess_precision") is None:
        apply_xla_flags("--xla_allow_excess_precision=false")


def unpartition_gpu_pool() -> None:
    """Turn off XLA's spatial partitioning of a preallocated GPU pool,
    unless the run named it.

    There a step's temporaries can lose their block between steps
    (`dew.training.memory.strands_temporaries`). The partitioning lets the
    pool's upper end hold XLA's collective memory space
    (xla/pjrt/gpu/se_gpu_pjrt_client.cc, `GetStreamExecutorGpuDeviceAllocator`
    at openxla/xla 91888df, the commit jax 0.11.2 builds), which a buffer
    takes only for NCCL user or symmetric buffers, a one-shot ragged
    all-to-all or a Mosaic kernel's symmetric operand
    (xla/service/gpu/gpu_memory_space_assignment.cc); Dew asks for none of
    them. With it off XLA serves that space from an allocator of its own, and
    steps ran as fast on an A100 (a DiT and a decoder within 0.5%).

    `import dew` applies it before the process's first JAX computation, as it
    does `keep_roundings`, so a Trainer built without `prepare_process` (a
    notebook, a tool) gets it too. With the partitioning on, the fit check
    holds room for a step's temporaries twice, and the 99M MoE at 8 x 1024
    on an RTX 4080 tiled its head (98.2 against 78.5 ms a step,
    docs/performance.md)."""
    bridge = sys.modules.get("jax._src.xla_bridge")
    if bridge is not None and bridge.backends_are_initialized():
        return
    if cuda_plugin() and xla_flag("xla_gpu_enable_allocator_spatial_partitioning") is None:
        apply_xla_flags("--xla_gpu_enable_allocator_spatial_partitioning=false")


def cuda_builds() -> list[str]:
    """JAX's CUDA builds installed, `cuda12` and `cuda13` by their plugins;
    asked before the backend opens, which no other question can be."""
    if importlib.util.find_spec("jax_plugins") is None:
        return []
    return [build for build, _, _ in CUDA_BUILDS
            if importlib.util.find_spec(f"jax_plugins.xla_{build}") is not None]


def cuda_plugin() -> bool:
    """Whether JAX's CUDA plugin is installed, the one reader of XLA's GPU flags."""
    return bool(cuda_builds())


CUDA_BUILDS = (("cuda13", 580, (7, 5)), ("cuda12", 525, (5, 2)))
"""JAX's CUDA builds, newest first, each with the least NVIDIA driver major
and GPU SM it runs on: JAX's installation guide, and NVIDIA's CUDA release
notes (Table 3, minor-version compatibility). docs/installation.md, "Which
CUDA build", states the rule, and dewml.dev's install script applies it."""


def runs_on(build: str, driver: str, sm: tuple[int, int]) -> bool:
    """Whether JAX's `build` runs on a driver of version `driver` (`580.82.07`)
    and GPUs whose least SM is `sm` (`CUDA_BUILDS`)."""
    least, least_sm = next((major, s) for name, major, s in CUDA_BUILDS if name == build)
    return int(driver.split(".")[0]) >= least and sm >= least_sm


def nvidia_gpus() -> tuple[str, tuple[int, int]] | None:
    """The driver version and the least SM of the GPUs nvidia-smi lists, or
    None where none answers: no nvidia-smi, no GPU, or a driver too old to
    report an SM."""
    if shutil.which("nvidia-smi") is None:
        return None
    query = ["nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"]
    try:
        found = subprocess.run(query, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = [line.split(",") for line in found.stdout.splitlines() if line.strip()]
    if found.returncode != 0 or not rows:
        return None
    try:
        return rows[0][0].strip(), min(tuple(map(int, sm.strip().split("."))) for _, sm in rows)
    except ValueError:
        return None


def cpu_only_refusal(env: Mapping[str, str], gpus: tuple[str, tuple[int, int]] | None,
                     installed: Sequence[str]) -> str | None:
    """Why a process would run on the CPU on a machine with an NVIDIA GPU,
    and the command that fixes it; None where it would not. `gpus` is
    `nvidia_gpus()`'s answer and `installed` the CUDA builds of JAX present
    (`cuda_builds`). JAX_PLATFORMS naming no GPU asks for the CPU, and is
    not refused."""
    if gpus is None or not runs_on_gpu(env):
        return None
    driver, sm = gpus
    if any(runs_on(build, driver, sm) for build in installed):
        return None
    found = f"This machine has an NVIDIA GPU (driver {driver}, SM {sm[0]}.{sm[1]}), and JAX would use the CPU"
    on_cpu = "To run on the CPU instead, set JAX_PLATFORMS=cpu."
    build = next((name for name, _, _ in CUDA_BUILDS if runs_on(name, driver, sm)), None)
    if build is None:
        needs = ", ".join(f"{name} needs driver {major}+ and SM {least[0]}.{least[1]}+"
                          for name, major, least in CUDA_BUILDS)
        return f"{found}: no CUDA build of JAX runs here ({needs}). Update the NVIDIA driver. {on_cpu}"
    install = f'uv pip install "dewml[{build}]"'
    if not installed:
        return f"{found}: no CUDA build of JAX is installed. Install the one it runs: {install}. {on_cpu}"
    stale = " ".join(f"jax-{name}-plugin jax-{name}-pjrt" for name in installed)
    return (f"{found}: JAX's {' and '.join(installed)} build cannot run on it, and {build} can. "
            f"Replace it: uv pip uninstall {stale} && {install}. {on_cpu}")


def refuse_cpu_only_jax() -> None:
    """Stop a process that would run on the CPU on a machine with an NVIDIA
    GPU, with the install that fixes it (`cpu_only_refusal`). `import dew`
    asks before the backend opens, as `keep_roundings` does, so it is asked
    once for every entry point; once a backend is open, JAX has chosen."""
    bridge = sys.modules.get("jax._src.xla_bridge")
    if (bridge is not None and bridge.backends_are_initialized()) or not runs_on_gpu(os.environ):
        return
    if (refusal := cpu_only_refusal(os.environ, nvidia_gpus(), cuda_builds())) is not None:
        raise RuntimeError(refusal)


def deterministic_ops_requested() -> bool:
    """Whether the run asked XLA for deterministic ops.

    `--xla_gpu_deterministic_ops` orders the reductions of a GPU step.
    Autotuning, which `--xla_gpu_autotune_level=0` turns off, can pick
    different kernels in another compilation (docs/guides/checkpoints.md).
    Kernel selection reads this flag: `dew.nn.attention` keeps
    cudnn's fused attention away from a run that set it.
    """
    return (xla_flag('xla_gpu_deterministic_ops') or '').lower() in ('true', '1')





def primary_context(ordinal: int) -> tuple[ctypes.CDLL, ctypes.c_void_p, ctypes.c_int]:
    """Retain the primary context of the visible CUDA device `ordinal`, the
    one JAX's runtime runs in: the driver library, the context and the
    device, which the caller releases with `cuDevicePrimaryCtxRelease_v2`
    once it is done. Raises OSError where no driver is installed and
    RuntimeError on a driver error."""
    cuda = ctypes.CDLL("libcuda.so.1")
    device, context = ctypes.c_int(), ctypes.c_void_p()
    for call in (lambda: cuda.cuInit(0), lambda: cuda.cuDeviceGet(ctypes.byref(device), ordinal),
                 lambda: cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device)):
        if status := call():
            raise RuntimeError(f"CUDA driver error {status}")
    return cuda, context, device


def gpu_free_bytes(ordinal: int) -> int | None:
    """The bytes the CUDA driver reports free on the visible GPU `ordinal`,
    every process's allocations counted, or None where no driver answers.

    An allocator's limit is a share of the GPU's memory, not memory it holds:
    a pool that grows takes each new region from what is free when it asks,
    and another process on the same GPU may hold the rest. The read makes the
    device's primary context current on this thread for the call."""
    try:
        cuda, context, device = primary_context(ordinal)
    except (OSError, RuntimeError) as error:
        _log.debug("no free memory read for GPU %d: %s", ordinal, error)
        return None
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    try:
        if cuda.cuCtxPushCurrent_v2(context):
            return None
        try:
            return None if cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)) else free.value
        finally:
            cuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
    finally:
        cuda.cuDevicePrimaryCtxRelease_v2(device)
