"""The extra XLA flags a run hands to the backend, and what the CUDA driver
says of a GPU.

XLA reads XLA_FLAGS once, when it opens a backend, so a run's flags have to
reach the environment before the first JAX call of the process.
"""

import ctypes
import logging
import os

_log = logging.getLogger(__name__)


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
    produces a BF16 sum. It takes effect only before the process's first
    JAX computation: XLA reads these flags when its backend opens. In an
    already-used notebook, restart with
    `XLA_FLAGS=--xla_allow_excess_precision=false` set before importing JAX.
    """
    if xla_flag("xla_allow_excess_precision") is None:
        apply_xla_flags("--xla_allow_excess_precision=false")


def deterministic_ops_requested() -> bool:
    """Whether the run asked XLA for deterministic ops.

    `--xla_gpu_deterministic_ops` orders the reductions of a GPU step.
    Autotuning, which `--xla_gpu_autotune_level=0` turns off, can pick
    different kernels in another compilation (docs/guides/checkpoints.md).
    Kernel selection reads this flag: `dew.nn.attention` keeps
    cudnn's fused attention away from a run that set it.
    """
    return (xla_flag('xla_gpu_deterministic_ops') or '').lower() in ('true', '1')


# The generations whose training steps compile with XLA's Triton GEMM fusions
# off, so every dot goes to cuBLAS: on the A100 (sm80) compiles take half as
# long and decoder, MoE and DiT steps run 0-6% faster, and on the RTX 4080
# (sm89) decoder steps run 3-10% faster, and 3x faster where the fusions hit
# a whole-logits head at 4096 tokens (docs/performance.md). A Mamba-2 mixer
# loses 7.7% without them and keeps them (`MixerBase.keeps_triton_gemm`), and
# a step that fits only with them keeps them (`fitting_default`). Unmeasured
# generations, sm86 among them, keep XLA's default.
TRITON_GEMM_OFF_GENERATIONS = frozenset({'sm80', 'sm89'})



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
