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

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generation import (
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
    from .ssd import ssd_kernel_platform, ssd_kernel_runs

_NAMED = {
    **dict.fromkeys((
        "PALLAS_CHOICES", "TARGETS", "GpuTarget", "assumed", "backend", "bf16_dot_runs",
        "chosen_kernel", "compile_only_gpus", "device_generation", "gpu_mesh", "gpu_target",
        "triton_runs",
    ), "generation"),
    "ssd_kernel_platform": "ssd",
    "ssd_kernel_runs": "ssd",
}
"""Each name the package re-exports, by the submodule that defines it."""


def __getattr__(name: str):
    """`name` of the package, loaded where it is first read and held from then on."""
    module = _NAMED.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    found = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = found
    return found


__all__ = ["PALLAS_CHOICES", "TARGETS", "GpuTarget", "assumed", "backend", "bf16_dot_runs", "chosen_kernel",
           "compile_only_gpus", "device_generation", "gpu_mesh", "gpu_target", "ssd_kernel_platform",
           "ssd_kernel_runs", "triton_runs"]
