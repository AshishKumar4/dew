"""What has to be told to the device backend before it is used.

Two things happen in this module, both of them before or around JAX's first
look at the hardware: the Pallas Triton backend learns about a card its table
does not list, and a run's extra XLA flags reach the backend that reads them
from the environment.
"""

import os
from typing import Optional

import jax


def register_pallas_device() -> bool:
    """Teach the Pallas Triton backend the current CUDA device, if it must.

    JAX keeps a table of device kinds it knows the compute capability of and
    refuses to compile a Triton kernel for anything else. Consumer cards are
    mostly absent: 0.11.1 lists the RTX 4090 and not the RTX 4080, which is
    the same AD10x architecture at compute capability 8.9. The entry the table
    wants is exactly what the device already reports, so the miss is a gap in
    a lookup table rather than a hardware limit, and the same registration
    hook JAX itself uses for ROCm cards fills it.

    Returns whether Pallas can lower a Triton kernel for this device now.
    Written against jax 0.11.1 (jax._src.pallas.triton.gpu_info); a release
    that moves the module leaves the private import failing and this a no-op,
    which is the same answer as running on a CPU.
    """
    if jax.default_backend() != 'gpu':
        return False
    try:
        from jax._src.pallas.triton import gpu_info
    except ImportError:
        return False

    kind = gpu_info.get_device_kind()
    if gpu_info.gpu_version_from_device_kind(kind) is not None or kind in gpu_info.registry:
        return True

    capability = getattr(jax.devices()[0], 'compute_capability', None)
    if not capability or '.' not in str(capability):
        return False
    major, minor = str(capability).split('.', 1)
    info = gpu_info.GpuInfo(gpu_version=None, arch_name=f"{major}.{minor}",
                            compute_capability=int(major) * 10 + int(minor))
    gpu_info.registry[kind] = lambda: info
    return True


def apply_xla_flags(flags: Optional[str]):
    """Append flags to XLA_FLAGS, which XLA reads when it initializes a backend.

    Appended rather than assigned: the environment may already carry flags
    (CI sets the host device count), and a run's own flags should add to them
    rather than replace them. Only useful before the first JAX call, which is
    why `dew.training.prepare_process` is the one caller.
    """
    if not flags:
        return
    existing = os.environ.get('XLA_FLAGS', '')
    os.environ['XLA_FLAGS'] = f"{existing} {flags}".strip()
