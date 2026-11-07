"""The extra XLA flags a run hands to XLA, and what the CUDA driver says of a
GPU.

XLA parses XLA_FLAGS once per runtime it loads (jaxlib's, the CUDA plugin's),
at the first use of its debug options, which can come before any backend opens
(a compile-only topology's compile options). None parses before the process
loads jaxlib, so a run's flags hold for certain where XLA_FLAGS carries them
by then (`flags_hold`). `import dew` imports this module, which writes Dew's
own flags, before anything it imports loads jaxlib.
"""

import ctypes
import importlib.util
import logging
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence

from dew._builds import CUDA_BUILDS, PACKAGE_INDEX, QUERY, listed_gpus, newest_build, runs_on
from dew._pool import answered, cuda_driver, runs_on_gpu

_log = logging.getLogger(__name__)
_late_policy_warned = False


def apply_xla_flags(flags: str | None) -> None:
    """Append flags to XLA_FLAGS, which XLA parses at its first use of its debug options.

    The flags are appended because the environment may already carry some
    (CI sets the host device count). Certain only before the process loads
    jaxlib (`flags_hold`); `dew.training.prepare_process` calls it before the
    backend opens.
    """
    if not flags:
        return
    existing = os.environ.get('XLA_FLAGS', '')
    os.environ['XLA_FLAGS'] = f"{existing} {flags}".strip()
    _seen()


_UNPARSED: str | None = None
"""XLA_FLAGS as Dew last saw or set them before the process loaded jaxlib, None where it had loaded it
before Dew was imported: flags they carry hold for every runtime XLA loads, provided nothing rewrites
XLA_FLAGS without them after."""

_UNOPENED: str | None = None
"""XLA_FLAGS as Dew last saw or set them while JAX's backend was closed, None where it had opened before
Dew was imported: flags a backend that opened since may have read."""


def _seen() -> None:
    """Record XLA_FLAGS where no XLA runtime can have parsed them yet (`_UNPARSED`) and where no backend
    has opened (`_UNOPENED`)."""
    global _UNPARSED, _UNOPENED
    flags = os.environ.get('XLA_FLAGS', '')
    if "jaxlib" not in sys.modules:
        _UNPARSED = flags
    if not backend_open():
        _UNOPENED = flags


def flags_hold(flags: str) -> bool:
    """Whether every flag of `flags` holds for each runtime XLA loads: carried by XLA_FLAGS while the
    process has not loaded jaxlib, or as Dew last saw them before it did (`_UNPARSED`); each read as XLA
    reads `flags` alone, its last occurrence's value, a bare flag's 'true'. A flag set after jaxlib loaded
    is unproven: XLA may have parsed its flags before, with no public way to tell. Flags that held stay
    certain while nothing rewrites XLA_FLAGS without them before a runtime XLA loads later parses them."""
    parsed = os.environ.get('XLA_FLAGS', '') if "jaxlib" not in sys.modules else _UNPARSED
    return parsed is not None and _carries(parsed, flags)


def flags_set(flags: str) -> bool:
    """Whether XLA_FLAGS carried every flag of `flags` while JAX's backend was closed: now, while it is,
    recorded as Dew saw them (`_seen`), or as Dew last saw or set them before it opened (`_UNOPENED`). A
    backend that opened since may have read them, which `flags_hold` proves where XLA_FLAGS carried them
    before jaxlib loaded."""
    if not backend_open():
        _seen()
    return _UNOPENED is not None and _carries(_UNOPENED, flags)


def hold_xla_flags(flags: str | None) -> None:
    """Make every flag of `flags` hold for each runtime XLA loads (`flags_hold`): where one does not, run
    the process again from its start (`sys.orig_argv`) with XLA_FLAGS carrying them, which its `import dew`
    then sees before anything loads jaxlib. Dew's command line does, as its `import dew` loads jaxlib
    before it reads a run's flags. A process that loaded jaxlib before Dew would load it first again, and
    goes on as it is."""
    if not flags or flags_hold(flags) or _UNPARSED is None:
        return
    existing = os.environ.get('XLA_FLAGS', '')
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, sys.orig_argv, {**os.environ, 'XLA_FLAGS': f"{existing} {flags}".strip()})


def _carries(loaded: str, flags: str) -> bool:
    """Whether the XLA_FLAGS `loaded` carry every flag of `flags`, as `flags_hold` reads them."""
    asked = {}
    for token in flags.split():
        name, equals, value = token.removeprefix("--").partition("=")
        asked[name] = value if equals else "true"
    return all(_flag(loaded, name) == value for name, value in asked.items())


def xla_flag(name: str) -> str | None:
    """The value `--<name>` carries in XLA_FLAGS, or None when it is absent.

    A bare `--<name>` reads as 'true' and the last occurrence wins, which is
    how XLA's own parser resolves a repeated flag. XLA parses the variable
    once, so this reports what the run asked for, not what a live backend was
    built with (`flags_hold`).
    """
    return _flag(os.environ.get('XLA_FLAGS', ''), name)


def _flag(flags: str, name: str) -> str | None:
    """The value `--<name>` carries in `flags`, as `xla_flag` reads it."""
    value = None
    for token in flags.split():
        if token == f"--{name}":
            value = 'true'
        elif token.startswith(f"--{name}="):
            value = token.split('=', 1)[1]
    return value


def keep_roundings() -> None:
    """Keep declared narrow-dtype roundings unless the caller names XLA's policy.

    Fusion must not normalize an unrounded FP32 value where the program
    produces a BF16 sum. `import dew` applies it before anything it imports
    loads jaxlib, before which XLA parses no flag (`flags_hold`). In an
    already-used notebook, restart with
    `XLA_FLAGS=--xla_allow_excess_precision=false` set before importing JAX.
    """
    global _late_policy_warned
    if backend_open():
        if not _late_policy_warned:
            _late_policy_warned = True
            _log.warning(
                "Dew was imported after the JAX backend opened; its numerical policy cannot take effect. "
                "Restart with XLA_FLAGS=--xla_allow_excess_precision=false set before importing JAX.")
        return
    if xla_flag("xla_allow_excess_precision") is None:
        apply_xla_flags("--xla_allow_excess_precision=false")


def backend_open() -> bool:
    """Whether JAX's backend has opened, after which XLA reads no flag or pool setting again."""
    # Private JAX query pinned by jax<0.11.3; the fresh-process late-import
    # test covers this path without opening a backend just to inspect it.
    bridge = sys.modules.get("jax._src.xla_bridge")
    return bridge is not None and bridge.backends_are_initialized()


def _gpu_default(name: str, assignment: str) -> None:
    """Pass XLA's GPU compiler `assignment` (`--name=value`) before the backend opens, unless the run
    named `name`."""
    if not backend_open() and cuda_plugin() and xla_flag(name) is None:
        apply_xla_flags(assignment)


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
    _gpu_default("xla_gpu_enable_allocator_spatial_partitioning",
                 "--xla_gpu_enable_allocator_spatial_partitioning=false")


def divide_exactly() -> None:
    """Have the GPU divide fp32 to the nearest float, unless the run named
    XLA's backend options.

    XLA:GPU lowers an fp32 division to PTX `div.full`, up to 2 ulps off
    (`-nvptx-prec-divf32=1`, xla/service/gpu/llvm_gpu_backend/nvptx_backend.cc),
    where a CPU and the references Dew is held to round to the nearest
    (`div.rn`, `-nvptx-prec-divf32=2`). Over a sampler's walk, a schedule's
    weights or a loss those ulps compound: on an H100 adaptive projected
    guidance's Euler walk sat 4.6 times Diffusers' distance from float64, the
    min-SNR weights 2.8 times and a Z-Image flow's forward 5.5e-5 from its
    source, each within the float64 rule with division rounded. XLA reads
    the option from `xla_backend_extra_options` when it compiles, so
    `import dew` sets it before the first computation, as it does
    `keep_roundings`."""
    _gpu_default("xla_backend_extra_options", "--xla_backend_extra_options=-nvptx-prec-divf32=2")


FRACTIONS = ("XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION")
"""The environment variables JAX reads a GPU pool's share from, its current name first; it refuses a run
that sets both (jaxlib's `xla_client.generate_pjrt_gpu_plugin_options`)."""

POOL_NAMES = (*FRACTIONS, "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR")
"""The environment variables a run names its GPU pool by."""

DEW_SHARE, JAX_SHARE = 0.95, 0.75
"""The share of each GPU XLA's pool takes as Dew prepares a process (`preallocate_gpu_pool`), and as JAX
takes it where a run names preallocation or the allocator without a fraction (docs.jax.dev,
gpu_memory_allocation)."""


def fraction_name() -> str:
    """The variable this run names its pool's share by (`FRACTIONS`), the deprecated one where it names
    none, as Dew's own share is written (`preallocate_gpu_pool`)."""
    return next((name for name in FRACTIONS if name in os.environ), FRACTIONS[1])


def pool_bound(*, cuda: bool) -> float | None:
    """The share of each GPU XLA's pool takes as this run names it, which Dew may size it below and never
    above (`dew.communicators`): its fraction where it names one (`fraction_name`); Dew's own where it
    names nothing (`preallocate_gpu_pool`); JAX's where it names preallocation or the allocator without
    one. None where the pool holds no fixed share: a process off CUDA, a pool that grows
    (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) or another allocator than XLA's own BFC pool."""
    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default").lower()
    growing = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true").lower() in ("false", "0")
    if not cuda or growing or allocator not in ("default", "bfc"):
        return None
    named = os.environ.get(fraction_name())
    if named is not None:
        return float(named)
    return JAX_SHARE if any(name in os.environ for name in POOL_NAMES) else DEW_SHARE


def preallocate_gpu_pool() -> None:
    """Preallocate 95% of each GPU (`DEW_SHARE`) as XLA's pool for a training process,
    unless the run named its pool (fraction, preallocation or allocator, `POOL_NAMES`).

    A pool that grows takes a step's temporaries only in a new region from
    the part of its limit it has not taken (`dew.training.memory.placeable`),
    so the regions it holds decide what fits: on an H100 SXM, Qwen3.5-0.8B at
    8 x 3072 found 59.2 GiB left to grow, with 16 GiB held, against 63.4 GiB
    of temporaries, and recomputed under 'minimal'; with 95% preallocated it
    ran without, 351 ms a step against 403 (an A100-40 at 4 x 3072, 389
    against 444 ms; gdn-one-path 99397ff40, runs/parity-*-gdn).
    JAX's default 75% holds less than that step's 71.8 GiB peak.

    It is a training process's (`dew.training.prepare_process`, `dew train`), which owns its GPUs; a
    Trainer fit in a process Dew did not prepare (a notebook, another program's) keeps JAX's 75%,
    which its backend has already taken, and shares its GPU as that program chose."""
    if backend_open():
        return
    if cuda_plugin() and not any(name in os.environ for name in POOL_NAMES):
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(DEW_SHARE)


def cuda_builds() -> list[str]:
    """JAX's CUDA builds installed, `cuda12` and `cuda13` by their plugins;
    asked before the backend opens, which no other question can be."""
    if importlib.util.find_spec("jax_plugins") is None:
        return []
    return [build for build, _, _, _ in CUDA_BUILDS
            if importlib.util.find_spec(f"jax_plugins.xla_{build}") is not None]


def cuda_plugin() -> bool:
    """Whether JAX's CUDA plugin is installed, the one reader of XLA's GPU flags."""
    return bool(cuda_builds())


def nvidia_gpus() -> tuple[str, tuple[int, int]] | None:
    """The driver version and the least SM of the GPUs nvidia-smi lists, or
    None where none answers (`dew._pool.answered`, `listed_gpus`)."""
    if shutil.which("nvidia-smi") is None:
        return None
    return listed_gpus(answered(QUERY, 10))


def cuda_driver_version() -> int | None:
    """The CUDA version the user-mode driver this process loads supports, as
    `cuDriverGetVersion` reports it (13.0 as 13000), the library JAX's CUDA
    plugin loads; None where no driver library loads. NVIDIA's
    forward-compatibility package raises it past what the kernel driver's
    major gives (`CUDA_BUILDS`)."""
    cuda = cuda_driver()
    if cuda is None:
        return None
    version = ctypes.c_int()
    return None if cuda.cuDriverGetVersion(ctypes.byref(version)) else version.value


CLOCK_RATE, MEMORY_CLOCK_RATE, GLOBAL_MEMORY_BUS_WIDTH, L2_CACHE_SIZE = 13, 36, 37, 38
"""`CUdevice_attribute`s (cuda.h): the multiprocessors' peak clock and the device memory's peak clock, in
kHz, the device memory bus's width in bits, and the L2 cache's size in bytes."""


def device_handle(ordinal: int) -> tuple[ctypes.CDLL, ctypes.c_int] | None:
    """The CUDA driver (`dew._pool.cuda_driver`) started, and its handle of the visible GPU `ordinal`: the one
    way every driver read here reaches a device. None where no driver library loads (a CPU host); a driver
    error raises."""
    cuda = cuda_driver()
    if cuda is None:
        return None
    device = ctypes.c_int()
    for status in (cuda.cuInit(0), cuda.cuDeviceGet(ctypes.byref(device), ordinal)):
        if status:
            raise RuntimeError(f"CUDA driver error {status}")
    return cuda, device


def gpu_attribute(ordinal: int, attribute: int) -> int | None:
    """The CUDA driver's `attribute` (a `CUdevice_attribute`) of the visible GPU `ordinal`: a device fact
    JAX's Device does not carry, a GPU's clocks above all. None where no driver library loads (a CPU
    host); a driver error raises."""
    handle = device_handle(ordinal)
    if handle is None:
        return None
    cuda, device = handle
    value = ctypes.c_int()
    if status := cuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, device):
        raise RuntimeError(f"CUDA driver error {status}")
    return value.value


def cpu_only_refusal(env: Mapping[str, str], gpus: tuple[str, tuple[int, int]] | None,
                     installed: Sequence[str], cuda: int | None) -> str | None:
    """Why a process would run on the CPU on a machine with an NVIDIA GPU,
    and the command that fixes it; None where it would not. `gpus` is
    `nvidia_gpus()`'s answer, `installed` the CUDA builds of JAX present
    (`cuda_builds`) and `cuda` the CUDA version the user-mode driver reports
    (`cuda_driver_version`), None where no driver library loads. A build runs
    where that version and the GPUs' least SM are the build's or newer, as its
    plugin's start checks. JAX_PLATFORMS naming no GPU asks for the CPU, and
    is not refused."""
    if gpus is None or not runs_on_gpu(env):
        return None
    driver, sm = gpus
    on_cpu = "To run on the CPU instead, set JAX_PLATFORMS=cpu."
    if cuda is None:
        return (f"This machine has an NVIDIA GPU (driver {driver}, SM {sm[0]}.{sm[1]}), and JAX would use "
                "the CPU: no CUDA driver library (libcuda.so.1) loads. Install the NVIDIA driver's user-mode "
                f"libraries, or put them on the loader's path. {on_cpu}")
    if any(runs_on(build, driver, sm, cuda) for build in installed):
        return None
    found = (f"This machine has an NVIDIA GPU (driver {driver}, CUDA {cuda // 1000}.{cuda % 1000 // 10}, "
             f"SM {sm[0]}.{sm[1]}), and JAX would use the CPU")
    build = newest_build(gpus, cuda)
    if build is None:
        needs = ", ".join(f"{name} needs CUDA {version // 1000}.{version % 1000 // 10}+ (driver {major}+, or "
                          f"an older one with NVIDIA's cuda-compat) and SM {least[0]}.{least[1]}+"
                          for name, major, least, version in CUDA_BUILDS)
        return f"{found}: no CUDA build of JAX runs here ({needs}). Update the NVIDIA driver. {on_cpu}"
    # Dew's JAX build is served from its own index only, beside PyPI.
    install = f'uv pip install --extra-index-url {PACKAGE_INDEX} "dewml[{build}]"'
    if not installed:
        return f"{found}: no CUDA build of JAX is installed. Install the one it runs: {install}. {on_cpu}"
    stale = " ".join(f"jax-{name}-plugin jax-{name}-pjrt" for name in installed)
    return (f"{found}: JAX's {' and '.join(installed)} build cannot run on it, and {build} can. "
            f"Replace it: uv pip uninstall {stale} && {install}. {on_cpu}")


class CPUOnlyJax(RuntimeError):
    """A machine with an NVIDIA GPU whose JAX opened with none (`refuse_cpu_only_jax`)."""


def refuse_cpu_only_jax() -> None:
    """Stop a process whose JAX opened with no GPU on a machine with an NVIDIA GPU, with the
    install that fixes it where one would (`cpu_only_refusal`). It reads the devices JAX opened,
    so it asks once the process's XLA flags are set: `prepare_process` asks it, and `Trainer.fit`
    for a program that never called that (`dew.training.runtime.ready_process`). JAX_PLATFORMS
    naming no GPU asks for the CPU, and is not refused."""
    import jax

    if not runs_on_gpu(os.environ) or any(device.platform == "gpu" for device in jax.devices()):
        return
    gpus = nvidia_gpus()
    if gpus is None:
        return
    refusal = cpu_only_refusal(os.environ, gpus, cuda_builds(), cuda_driver_version())
    raise CPUOnlyJax(refusal or (
        f"This machine has an NVIDIA GPU (driver {gpus[0]}), and JAX opened with no GPU beside a CUDA "
        f"build that runs on it: {jax.devices()}. Read the CUDA plugin's warning above, or set "
        "JAX_PLATFORMS=cuda to see why it failed; to run on the CPU instead, set JAX_PLATFORMS=cpu."))


def deterministic_ops_requested() -> bool:
    """Whether the run asked XLA for deterministic ops.

    `--xla_gpu_deterministic_ops` orders the reductions of a GPU step and
    turns XLA's kernel measurements off, so every compilation picks the same
    kernels (docs/guides/checkpoints.md). Kernel selection reads this flag:
    `dew.nn.attention` keeps cudnn's fused attention away from a run that set
    it, and `dew._cache.enable_compilation_cache` gives it no stored picks.
    """
    return (xla_flag('xla_gpu_deterministic_ops') or '').lower() in ('true', '1')





def primary_context(ordinal: int) -> tuple[ctypes.CDLL, ctypes.c_void_p, ctypes.c_int]:
    """Retain the primary context of the visible CUDA device `ordinal`, the
    one JAX's runtime runs in: the driver library, the context and the
    device, which the caller releases with `cuDevicePrimaryCtxRelease_v2`
    once it is done. Raises OSError where no driver is installed and
    RuntimeError on a driver error."""
    handle = device_handle(ordinal)
    if handle is None:
        raise OSError("no CUDA driver library loads")
    cuda, device = handle
    context = ctypes.c_void_p()
    if status := cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device):
        raise RuntimeError(f"CUDA driver error {status}")
    return cuda, context, device


def gpu_free_bytes(ordinal: int) -> int | None:
    """The bytes the CUDA driver reports free on the visible GPU `ordinal`,
    every process's allocations counted, or None where no driver answers.

    An allocator's limit is a share of the GPU's memory, not memory it holds:
    a pool that grows takes each new region from what is free when it asks,
    and another process on the same GPU may hold the rest. The read makes the
    device's primary context current on this thread for the call."""
    memory = gpu_memory(ordinal)
    return None if memory is None else memory[0]


def gpu_memory(ordinal: int) -> tuple[int, int] | None:
    """The bytes the CUDA driver reports free and in all on the visible GPU
    `ordinal`, as `gpu_free_bytes` reads them, or None where no driver answers."""
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
            if cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)):
                return None
            return free.value, total.value
        finally:
            cuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
    finally:
        cuda.cuDevicePrimaryCtxRelease_v2(device)


def _reported(ordinal: int, size: int, call: Callable[[ctypes.CDLL], Callable[..., int]]) -> str | None:
    """The text a CUDA driver `call` writes of the visible GPU `ordinal`, at most `size` bytes, None where no
    driver library loads; a driver error raises."""
    handle = device_handle(ordinal)
    if handle is None:
        return None
    cuda, device = handle
    text = ctypes.create_string_buffer(size)
    if status := call(cuda)(text, len(text), device):
        raise RuntimeError(f"CUDA driver error {status}")
    return text.value.decode()


def gpu_name(ordinal: int) -> str | None:
    """The visible GPU `ordinal`'s name, as the CUDA driver reports it (`NVIDIA H100 80GB HBM3`)."""
    return _reported(ordinal, 256, lambda cuda: cuda.cuDeviceGetName)


def gpu_place(ordinal: int) -> str | None:
    """The visible GPU `ordinal`'s PCI bus id, as the CUDA driver reports it (`0000:18:00.0`): where it sits,
    which its links to the other GPUs follow."""
    return _reported(ordinal, 32, lambda cuda: cuda.cuDeviceGetPCIBusId)


def nccl_version() -> int | None:
    """The version of the NCCL library this process would load (`ncclGetVersion`: 23203 for 2.32.3), from the
    nvidia-nccl wheel JAX's CUDA plugin depends on, or the loader's path; None where none loads."""
    # The wheel's directory, where its namespace package is installed, found without importing it.
    namespace = importlib.util.find_spec("nvidia")
    spec = importlib.util.find_spec("nvidia.nccl") if namespace is not None else None
    places = [os.path.join(place, "lib", "libnccl.so.2") for place in (spec.submodule_search_locations or [])
              ] if spec is not None else []
    for place in [*places, "libnccl.so.2"]:
        try:
            library = ctypes.CDLL(place)
        except OSError:
            continue
        version = ctypes.c_int()
        return None if library.ncclGetVersion(ctypes.byref(version)) else version.value
    return None


def connect_at_creation() -> None:
    """Have NCCL connect a communicator's rings, trees and NVLS peers as it
    creates the communicator, unless the run named it (NCCL_RUNTIME_CONNECT=0).

    NCCL connects them lazily by default since 2.22, at a collective's first
    launch, and allocates their buffers then. A rank that fails to allocate
    there leaves its peer inside XLA's first-call rendezvous, which aborts the
    process after 40 seconds (xla/service/rendezvous.cc, LOG(FATAL)).
    Connected at creation, those buffers are allocated as XLA acquires the
    step's communicators, before it runs, and XLA returns a failure there to
    every rank as an error (xla/backends/gpu/collectives/nccl_collectives.cc);
    the cost is their buffers at creation, whether or not the program runs a
    collective that uses them. Point-to-point peers, an all-to-all's and a
    permute's, still connect as each is first sent to, whatever this setting:
    `dew.communicators` holds their room, as the 2xH100 LoRA run whose first
    all-to-all one GPU entered and the other never did needed (ledger 2901).
    NCCL reads the variable as it first starts, so `import dew` sets it."""
    os.environ.setdefault("NCCL_RUNTIME_CONNECT", "0")



# `import dew` imports this module, and writes Dew's own flags (dew/__init__.py), before anything it imports
# loads jaxlib.
_seen()
