"""Pallas kernels for the operations XLA does not fuse.

Each module here implements one operation and exports the predicate that
says which shapes and backends its kernel covers. The XLA form that the
kernel replaces stays in the module the operation belongs to. That form is
the reference the kernel's tests compare against, and the path every other
shape and backend takes.

Each re-exported name is loaded where it is first read (PEP 562's module
`__getattr__`), never with the package: importing any submodule under
`dew.nn.kernels` runs this file, and eager imports of `.ssd` and `.generation`
here pulled `ssd`, `algebra.scan`, and `templates.report` into 3,856 test
tasks (Train 1123 `keydiff`).
"""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.nn.kernels.generation import (
        PALLAS_CHOICES,
        TARGETS,
        GpuTarget,
        assumed,
        backend,
        bf16_dot_runs,
        chosen_kernel,
        compile_only_gpus,
        device_generation,
        gpu_mesh,
        gpu_target,
        triton_runs,
    )
    from dew.nn.kernels.ssd import ssd_kernel_platform, ssd_kernel_runs

_EXPORTS = {
    "PALLAS_CHOICES": "dew.nn.kernels.generation",
    "TARGETS": "dew.nn.kernels.generation",
    "GpuTarget": "dew.nn.kernels.generation",
    "assumed": "dew.nn.kernels.generation",
    "backend": "dew.nn.kernels.generation",
    "bf16_dot_runs": "dew.nn.kernels.generation",
    "chosen_kernel": "dew.nn.kernels.generation",
    "compile_only_gpus": "dew.nn.kernels.generation",
    "device_generation": "dew.nn.kernels.generation",
    "gpu_mesh": "dew.nn.kernels.generation",
    "gpu_target": "dew.nn.kernels.generation",
    "triton_runs": "dew.nn.kernels.generation",
    "ssd_kernel_platform": "dew.nn.kernels.ssd",
    "ssd_kernel_runs": "dew.nn.kernels.ssd",
}

__getattr__, __dir__ = _lookup(__name__, _EXPORTS)

__all__ = [
    "PALLAS_CHOICES",
    "TARGETS",
    "GpuTarget",
    "assumed",
    "backend",
    "bf16_dot_runs",
    "chosen_kernel",
    "compile_only_gpus",
    "device_generation",
    "gpu_mesh",
    "gpu_target",
    "ssd_kernel_platform",
    "ssd_kernel_runs",
    "triton_runs",
]
