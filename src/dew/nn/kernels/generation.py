"""The hardware generation every kernel selector in Dew keys on.

One key, read from the default device: `smXY` from a GPU's compute
capability, `v5e`/`v5p`/`v6e`/`v4` from a TPU's device kind, and the
platform name otherwise. A selector names the generations it was measured on
and sends every other one to its portable path. The measurements are in
docs/performance.md, "Kernel choices per generation".
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import logging
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax.sharding import AbstractMesh, Mesh

_log = logging.getLogger(__name__)

ASSUMED: contextvars.ContextVar[str | None] = contextvars.ContextVar("assumed_generation", default=None)
"""The generation `assumed` stands in for the default device's, None outside it."""


@contextlib.contextmanager
def assumed(generation: str | None) -> Iterator[None]:
    """Within the block, choose kernels as on a device of `generation` ('sm80'), whatever device this
    process has: `device_generation`, `bf16_dot_runs` and `triton_runs` answer for it, so every
    `KERNELS` choice and kernel refusal acts as on that GPU while a step traces and lowers for one on
    a CPU (`dew.training._lowering`). None assumes nothing."""
    token = ASSUMED.set(generation)
    try:
        yield
    finally:
        ASSUMED.reset(token)


PALLAS_CHOICES = frozenset({'pallas', 'fla', 'triton'})
"""The `KERNELS` values that run Dew's Pallas or Triton kernels, which lower through JAX's Pallas
Triton backend for a GPU only: what the CPU dry run lowers (`dew.training._lowering`) and the kernel
lab verifies on the GPU. 'flash' and 'cudnn' name CUDA libraries."""


UNLOWERABLE = {
    ('tokamax_grouped_matmul', 'triton'): (
        "tokamax registers its Triton ragged dot on GPU hosts only, so no CPU lowers it, and the "
        "kernel lab's verify on the GPU is its only check"),
}
"""The Pallas choices a `KERNELS` row can name whose Triton lowering no CPU reaches, and why."""


@dataclasses.dataclass(frozen=True)
class GpuTarget:
    """One generation's device prefixes, dense peaks, block memory and measured kernel row.

    `dense_peaks` maps reported device-kind prefixes to precision rates, so
    PCIe/SXM and GeForce/data-center variants retain their own peaks. `device_kind`
    is the canonical device for a generation, or the reported SKU a reader asked
    for. Empty resources remain unknown. The standard-library-only table can
    also be read by a Torch reference's environment without importing JAX.
    """

    generation: str
    device_kind: str
    shared_limit: int | None = None
    kernels: Mapping[str, str] = dataclasses.field(default_factory=dict)
    dense_peaks: Mapping[str, Mapping[str, float]] = dataclasses.field(default_factory=dict)
    aliases: tuple[str, ...] = ()

    @property
    def device_kind_prefixes(self) -> tuple[str, ...]:
        return tuple(self.dense_peaks)

    @property
    def peaks(self) -> Mapping[str, float]:
        keys = [prefix for prefix in self.dense_peaks if self.device_kind.startswith(prefix)]
        return self.dense_peaks[max(keys, key=len)] if keys else {}

    @property
    def bf16_peak(self) -> float | None:
        return self.peaks.get('bf16')

    @property
    def fp8_peak(self) -> float | None:
        return self.peaks.get('fp8')


# CUDA datasheets give sparse tensor peaks; the dense figures below are half.
# A100 bf16 624; H100 SXM bf16/FP8 1979/3958, PCIe 1513/3026, NVL 1671/3341;
# H200 SXM 1979/3958, NVL 1671/3341; L4 242/485; RTX PRO 6000 Blackwell Server 1000/2000 TF/s.
# Existing rounded bf16 rates remain the baseline; FP8 uses the stated dense half.
# https://i.dell.com/sites/csdocuments/App-Merchandizing_Documents/en/us/h100-gpu.pdf
# https://www.nvidia.com/en-us/data-center/h100/
# https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/NVIDIA_H200_Datasheet.pdf (Jul24)
# https://www.nvidia.com/en-us/data-center/l4/
# https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
# GeForce bf16 is dense fp32-accumulate throughput: RTX4080 97.5 (Ada whitepaper),
# RTX3090 71 (GA102 whitepaper), half the fp16-accumulator rate.
# TPU per-chip bf16: v2/v3 45/123 (two JAX devices/chip, halved); v4 275;
# v5e 197, v5p459, v6e918 from Google's architecture and generation pages.
# Opted-in shared memory per block, in KiB: Turing64, A100163, Ampere/Ada/RTX
# Blackwell99, Hopper227. The SM reservation is already excluded:
# https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
# https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
# https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
TARGETS = {target.generation: target for target in (
    GpuTarget('sm75', 'Tesla T4', 64 * 1024, dense_peaks={'Tesla T4': {}, 'NVIDIA Tesla T4': {}},
              aliases=('gpu:t4', 't4')),
    GpuTarget('sm80', 'NVIDIA A100-SXM4-80GB', 163 * 1024,
              {'attention': 'flash', 'grouped_matmul': 'pallas', 'tokamax_grouped_matmul': 'triton',
               'xla_triton_gemm': 'off', 'narrow_copies': 'on'},
              {'NVIDIA A100': {'bf16': 312e12}}, ('gpu:a100', 'a100')),
    GpuTarget('sm86', 'NVIDIA A10', 99 * 1024, {'grouped_matmul': 'pallas'},
              {'NVIDIA A10': {}, 'NVIDIA GeForce RTX 3090': {'bf16': 71e12}}),
    GpuTarget('sm89', 'NVIDIA L4', 99 * 1024,
              {'attention': 'flash', 'grouped_matmul': 'pallas', 'tokamax_grouped_matmul': 'triton',
               'xla_triton_gemm': 'off', 'narrow_copies': 'on'},
              {'NVIDIA L4': {'bf16': 121e12, 'fp8': 242.5e12},
               'NVIDIA GeForce RTX 4080': {'bf16': 97.5e12}}, ('gpu:l4', 'l4')),
    GpuTarget('sm90', 'NVIDIA H100 80GB HBM3', 227 * 1024, dense_peaks={
        'NVIDIA H100': {'bf16': 989e12, 'fp8': 1979e12},
        'NVIDIA H100 PCIe': {'bf16': 756e12, 'fp8': 1513e12},
        'NVIDIA H100 NVL': {'bf16': 835e12, 'fp8': 1670.5e12},
        'NVIDIA H200': {'bf16': 989e12, 'fp8': 1979e12},
        'NVIDIA H200 NVL': {'bf16': 835e12, 'fp8': 1670.5e12}}),
    GpuTarget('sm120', 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 99 * 1024,
              dense_peaks={'NVIDIA RTX PRO 6000 Blackwell': {'bf16': 500e12, 'fp8': 1000e12}},
              aliases=('gpu:g4', 'g4')),
    GpuTarget('v2', 'TPU v2', dense_peaks={'TPU v2': {'bf16': 22.5e12}}),
    GpuTarget('v3', 'TPU v3', dense_peaks={'TPU v3': {'bf16': 61.5e12}}),
    GpuTarget('v4', 'TPU v4', dense_peaks={'TPU v4': {'bf16': 275e12}}),
    GpuTarget('v5e', 'TPU v5 lite',
              kernels={'grouped_matmul': 'xla', 'tokamax_grouped_matmul': 'mosaic_tpu_v2'},
              dense_peaks={'TPU v5 lite': {'bf16': 197e12}}),
    GpuTarget('v5p', 'TPU v5', dense_peaks={'TPU v5': {'bf16': 459e12}, 'TPU v5p': {'bf16': 459e12}}),
    GpuTarget('v6e', 'TPU v6 lite',
              kernels={'grouped_matmul': 'xla', 'tokamax_grouped_matmul': 'mosaic_tpu_v2'},
              dense_peaks={'TPU v6 lite': {'bf16': 918e12}}),
)}


def gpu_target(identifier: str | None = None) -> GpuTarget:
    """The target for a generation, reported device kind or case-insensitive lab alias.

    None is the default device's: its generation is `device_generation`'s, the
    one seam that `assumed` and tests set, and its device kind only picks the
    SKU's peaks within that generation; a real GPU the table names no SKU of
    keeps its own kind, so its peaks stay unknown, while an assumed or patched
    generation takes the generation's canonical device. SKU queries retain
    that SKU's peaks. Unknown kinds have empty resources.
    """
    if identifier is None:
        import jax
        device = jax.devices()[0]
        generation, kind = device_generation(), device.device_kind
        target = gpu_target(kind)
        if target.generation == generation:
            return target
        canonical = TARGETS.get(generation, GpuTarget(generation, kind))
        real = device.platform == 'gpu' and ASSUMED.get() is None
        return dataclasses.replace(canonical, device_kind=kind) if real else canonical
    if identifier in TARGETS:
        return TARGETS[identifier]
    for target in TARGETS.values():
        if identifier.lower() in target.aliases:
            return target
    matches = [(len(prefix), target) for target in TARGETS.values()
               for prefix in target.device_kind_prefixes if identifier.startswith(prefix)]
    if matches:
        return dataclasses.replace(max(matches, key=lambda entry: entry[0])[1], device_kind=identifier)
    return GpuTarget(identifier, identifier)


def gpu_mesh(mesh: Mesh, generation: str) -> AbstractMesh:
    """`mesh`'s axes over abstract devices of the CUDA `generation` ('sm80'), none of which this process
    needs: a program traced on it, under `assumed(generation)`, lowers its Pallas kernels to Triton
    for that GPU on any host (`TrainerConfig.lower_for`; the kernel lab's oracle)."""
    from jax.sharding import AbstractDevice, AbstractMesh
    target = gpu_target(generation)
    if not target.generation.startswith('sm') or target.shared_limit is None:
        raise ValueError(f"no CUDA target for {generation!r}")
    device = AbstractDevice(device_kind=target.device_kind, num_cores=None, platform='cuda')
    return AbstractMesh(mesh.axis_sizes, mesh.axis_names, axis_types=mesh.axis_types, abstract_device=device)


# The first GPU generation with bf16 tensor cores (Ampere). Below it XLA
# rejects the BF16_BF16_F32 dot algorithm at run time ("UNIMPLEMENTED:
# Unsupported algorithm on the current device(s): ALG_DOT_BF16_BF16_F32"),
# cuDNN's fused attention refuses bf16 ("SDPA FP16/BF16 requires SM80"), and
# Triton does not compile; measured on a T4 (sm75), KernelMatrix 2026-09-22.
BF16_GPU = 80


def _compute_capability(device) -> str | None:
    """`device`'s compute capability, '8.9' for an sm89 card. jax's Device
    declares no such field, and a CUDA device carries it, so it is read at
    this boundary; a device without one gives None."""
    return getattr(device, 'compute_capability', None)


def device_generation() -> str:
    """Return the default device's hardware generation.

    The generation is 'sm89' for a GPU of compute capability 8.9, 'v6e' for a
    TPU v6e, and the platform's name otherwise; within `assumed`, the one it
    names.
    """
    import jax
    generation = ASSUMED.get()
    if generation is not None:
        return generation
    device = jax.devices()[0]
    capability = _compute_capability(device) if device.platform == 'gpu' else None
    if capability:
        return 'sm' + capability.replace('.', '')
    if device.platform == 'tpu':
        kind = device.device_kind or 'tpu'
        return gpu_target(kind).generation
    return device.platform


def measured_kernel(choice: str, default: str) -> str:
    """The kernel `KERNELS` names for `choice` on the default device's generation, else `default`."""
    return gpu_target(device_generation()).kernels.get(choice, default)


# Each (choice, kernel, refusal) `ran_kernel` has logged in this process.
_logged: set[tuple[str, str, str]] = set()


def ran_kernel(choice: str, kernel: str, refusal: str, fastest: str | None = None) -> str:
    """Return `kernel`, which a call to `choice` runs because the kernel measured fastest on this
    generation, `fastest` or else the one `KERNELS` names, turned it down for `refusal`. The first such
    call in a process for each kernel and refusal is logged as a warning naming both, so a change that
    sends calls off the measured kernel shows in a run's log, not only in its step time. A generation
    without a measurement, or a call that runs the measured kernel, logs nothing."""
    generation = device_generation()
    measured = fastest or gpu_target(generation).kernels.get(choice)
    if measured not in (None, kernel) and (choice, kernel, refusal) not in _logged:
        _logged.add((choice, kernel, refusal))
        _log.warning("%s runs %r, not %r, the kernel measured fastest on %s: %s",
                     choice, kernel, measured, generation, refusal)
    return kernel


def first_refusal(*checks: tuple[bool, str]) -> str | None:
    """The reason beside the first of `checks` that does not hold, or None where all hold: how a
    kernel's eligibility says why it turns a call down (`ran_kernel`)."""
    return next((reason for holds, reason in checks if not holds), None)


def bf16_dot_runs() -> bool:
    """Return whether the default device multiplies bf16 operands into an fp32 sum as one dot algorithm.

    That algorithm is BF16_BF16_F32. Every TPU and CPU has it, and so does a
    GPU from generation `BF16_GPU` on.
    """
    generation = device_generation()
    return not generation.startswith('sm') or int(generation[2:]) >= BF16_GPU


def backend() -> str:
    """The platform a call's code is chosen for: `jax.default_backend()`, or within `assumed` the
    assumed generation's, 'gpu' for an `smXY` and 'tpu' for a TPU's, so a choice keyed on the
    platform acts as on that device while a step traces for it on another."""
    import jax
    generation = ASSUMED.get()
    if generation is None:
        return jax.default_backend()
    if generation.startswith('sm'):
        return 'gpu'
    if generation in TARGETS and TARGETS[generation].device_kind.startswith('TPU '):
        return 'tpu'
    return generation


def triton_runs() -> bool:
    """Return whether Dew's Pallas GPU (Triton) kernels can run here.

    This is the one eligibility rule for those kernels: a GPU of compute
    capability 8.0 or later, the same bound JAX's own Pallas lowerings apply.
    A T4 fails to compile them ("Triton support is only enabled for
    cc>=8.0"). Within `assumed`, the GPU is the assumed generation's.
    """
    # JAX's bound is `_backend_supports_triton`.
    return backend() == 'gpu' and bf16_dot_runs()
