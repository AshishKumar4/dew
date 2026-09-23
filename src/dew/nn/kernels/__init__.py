"""Pallas kernels for the operations XLA does not fuse.

Each module here owns one operation, exports the predicate that says which
shapes and backends its kernel covers, and leaves the XLA form that it
replaces in the module the operation belongs to, as the oracle its tests
compare against and as the path every other shape and backend takes.
"""

from .generation import bf16_dot_runs, device_generation, triton_runs
from .grouped_matmul import grouped_projection, ragged_dot_runs
from .ssd import ssd_chunk_scan, ssd_kernel_platform, ssd_kernel_runs

__all__ = ["bf16_dot_runs", "device_generation", "grouped_projection", "ragged_dot_runs",
           "ssd_chunk_scan", "ssd_kernel_platform", "ssd_kernel_runs", "triton_runs"]
