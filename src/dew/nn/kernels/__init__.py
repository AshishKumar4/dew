"""Pallas kernels for the operations XLA does not fuse.

Each module here implements one operation and exports the predicate that
says which shapes and backends its kernel covers. The XLA form that the
kernel replaces stays in the module the operation belongs to. That form is
the reference the kernel's tests compare against, and the path every other
shape and backend takes.
"""

from .generation import (
    KERNELS,
    bf16_dot_runs,
    device_generation,
    first_refusal,
    measured_kernel,
    ran_kernel,
    triton_runs,
)
from .grouped_matmul import grouped_projection, ragged_dot_refusal
from .ssd import ssd_chunk_scan, ssd_kernel_platform, ssd_kernel_runs

__all__ = ["KERNELS", "bf16_dot_runs", "device_generation", "first_refusal", "grouped_projection",
           "measured_kernel", "ragged_dot_refusal", "ran_kernel",
           "ssd_chunk_scan", "ssd_kernel_platform", "ssd_kernel_runs", "triton_runs"]
