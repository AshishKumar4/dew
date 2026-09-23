"""The hardware generation every kernel selector in Dew keys on.

One key, read from the default device: `smXY` from a GPU's compute
capability, `v5e`/`v5p`/`v6e`/`v4` from a TPU's device kind, and the
platform name otherwise. A selector names the generations it was measured on
and sends every other one to its portable path. The measurements are in
docs/performance.md, "Kernel choices per generation".
"""

import functools
import warnings

import jax

# `device_kind` of the TPU generations Dew names.
TPU_GENERATIONS = {'TPU v4': 'v4', 'TPU v5 lite': 'v5e', 'TPU v5': 'v5p', 'TPU v5p': 'v5p',
                   'TPU v6 lite': 'v6e'}

# The first GPU generation with bf16 tensor cores (Ampere). Below it XLA
# rejects the BF16_BF16_F32 dot algorithm at run time ("UNIMPLEMENTED:
# Unsupported algorithm on the current device(s): ALG_DOT_BF16_BF16_F32"),
# cuDNN's fused attention refuses bf16 ("SDPA FP16/BF16 requires SM80"), and
# Triton does not compile; measured on a T4 (sm75), KernelMatrix 2026-09-22.
BF16_GPU = 80


def device_generation() -> str:
    """The default device's hardware generation: 'sm89' for a GPU of compute
    capability 8.9, 'v6e' for a TPU v6e, and the platform's name otherwise."""
    device = jax.devices()[0]
    if device.platform == 'gpu' and getattr(device, 'compute_capability', None):
        return 'sm' + device.compute_capability.replace('.', '')
    if device.platform == 'tpu':
        kind = device.device_kind or 'tpu'
        return TPU_GENERATIONS.get(kind, kind)
    return device.platform


def sm_version(generation: str) -> int | None:
    """89 for 'sm89', None for anything that is not a GPU generation."""
    digits = generation[2:]
    return int(digits) if generation.startswith('sm') and digits.isdigit() else None


def bf16_dot_runs(generation: str | None = None) -> bool:
    """Whether `generation` (the default device's when None) multiplies bf16
    operands into an fp32 sum as one dot algorithm, BF16_BF16_F32: every TPU
    and CPU, and a GPU from `BF16_GPU` on."""
    version = sm_version(device_generation() if generation is None else generation)
    return version is None or version >= BF16_GPU


# jax 0.11.2 deprecates the Pallas Triton backend and warns at every
# lowering. Dew's Triton kernels (the grouped matmul, the Mamba-2 SSD scan)
# stay on sm80 to sm89 on purpose: JAX's Mosaic GPU kernels use wgmma, which
# those cards do not have.
TRITON_DEPRECATION = (r"The Pallas Triton backend is deprecated and will be removed in"
                      r" a future JAX version\.")


def triton_runs() -> bool:
    """The one eligibility rule for Dew's Pallas GPU (Triton) kernels: this
    process holds a GPU of compute capability 8.0 or later, the bound JAX's
    own Pallas lowerings apply (`_backend_supports_triton`). A T4 fails to
    compile them ("Triton support is only enabled for cc>=8.0")."""
    gpus = _gpu_versions()
    return bool(gpus) and min(gpus) >= BF16_GPU


def triton_compiles() -> bool:
    """`triton_runs`, or no GPU in the process at all: a pallas_call named
    for 'gpu' on a host without one runs Pallas's interpreter, which any
    host can. Only a GPU older than sm80 refuses."""
    gpus = _gpu_versions()
    return not gpus or min(gpus) >= BF16_GPU


def _gpu_versions() -> list[int]:
    return [int(device.compute_capability.replace('.', ''))
            for device in jax.devices() if device.platform == 'gpu']


@functools.cache
def filter_triton_deprecation() -> None:
    """Ignore `TRITON_DEPRECATION`, only that message and only as a
    DeprecationWarning, once a Triton kernel is first used.

    JAX raises it when a pallas_call is lowered, which is when the jit
    around a whole step compiles, after the kernel's call has returned and
    with none of Dew's frames on the stack. A filter scoped to the call
    cannot see it, so this one lasts for the process, and a process that
    never runs the kernels never installs it."""
    warnings.filterwarnings('ignore', message=TRITON_DEPRECATION, category=DeprecationWarning)
