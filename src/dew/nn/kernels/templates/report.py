"""Mosaic GPU's own compile reports, what the templates' resource models are checked against: ptxas's
registers a thread and spilled bytes, and Mosaic's resource estimate (the shared bytes and tensor memory
columns a kernel asks for), written for each kernel Mosaic compiles while they are on (`reported`), at XLA's
compile of the program around it, with a device or a compile-only one (`compile_device`). The CPU suite's
compile gate reads them (tests/kernel_compile.py); a run never does: its resources are derived from the
models (`templates.resources`, owner 10-27)."""

import contextlib
import dataclasses
import functools
import os
import re
import threading
from pathlib import Path

import jax.experimental.topologies as topologies
import numpy as np
from jax.sharding import Mesh

from dew.nn.kernels import generation

DUMPS = {"MOSAIC_GPU_DUMP_PTXAS": "1", "MOSAIC_GPU_DUMP_RESOURCES": "1"}
"""The settings that turn Mosaic GPU's reports on, written under `MOSAIC_GPU_DUMP_TO`."""
DUMPING = threading.RLock()
"""Held while the reports are on (`reported`): the settings are the process's environment, so one scope at
a time sets and restores them."""
REGISTERS = re.compile(r"Used (\d+) registers")
SPILLS = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
SHARED = re.compile(r"smem_scratch_bytes=(\d+)")
TENSOR_MEMORY = re.compile(r"tmem_scratch_cols=(\d+), tmem_collective_scratch_cols=(\d+)")
KERNEL_COUNTER = re.compile(r"_\d+$")
"""The counter Mosaic GPU appends to a kernel's name in its reports' file names, which JAX 0.12 numbers
apart for a kernel's two reports."""


@dataclasses.dataclass(frozen=True)
class Report:
    """One Mosaic GPU kernel's reports: its name, the shared bytes and tensor memory columns it asks for
    (Mosaic's resource estimate), its registers a thread and its spilled bytes (ptxas's, the most over the
    kernel's functions), and what the reports left `unread`: either report without the other, or a field
    neither states, is no evidence the kernel is within its resources."""
    kernel: str
    shared: int | None
    tmem: int | None
    registers: int | None
    spilled: int | None
    unread: tuple[str, ...] = ()

    def failure(self, shared_limit: int, tmem_limit: int) -> str | None:
        """Why the kernel is not within a multiprocessor's `shared_limit` bytes and `tmem_limit` tensor memory
        columns without spilling registers, or None where it is."""
        if self.unread:
            return f"its reports state no {', '.join(self.unread)}"
        if (self.shared is not None and self.tmem is not None and self.shared <= shared_limit
                and self.tmem <= tmem_limit and self.spilled == 0):
            return None
        return (f"{self.shared} shared bytes (limit {shared_limit}), {self.tmem} tensor memory columns "
                f"(limit {tmem_limit}), {self.spilled} bytes spilled")


@contextlib.contextmanager
def reported(directory: str):
    """Mosaic GPU's reports on, into `directory`, for the compiles in the block, under `DUMPING` from before
    the settings are read to after they are restored."""
    with DUMPING:
        before = {name: os.environ.get(name) for name in (*DUMPS, "MOSAIC_GPU_DUMP_TO")}
        os.environ.update(DUMPS, MOSAIC_GPU_DUMP_TO=directory)
        try:
            yield
        finally:
            for name, value in before.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def reports(directory: str) -> list[Report]:
    """Each kernel's `Report` in `directory`, its two reports paired by the kernel's name less the counter
    Mosaic appends (`KERNEL_COUNTER`); the compiles a directory holds name each kernel once."""
    named: dict[str, dict[str, Path]] = {}
    for kind, suffix in (("resources", ".resources.txt"), ("ptxas", ".ptxas")):
        for path in Path(directory).glob(f"*{suffix}"):
            found = named.setdefault(KERNEL_COUNTER.sub("", path.name.removesuffix(suffix)), {})
            # Paired by name: compiles that name a kernel twice would pair the wrong reports.
            assert kind not in found, f"{path.name} and {found[kind].name} compiled under one name"
            found[kind] = path
    found_reports = []
    for name, found in sorted(named.items()):
        text = found["resources"].read_text() if "resources" in found else ""
        log = found["ptxas"].read_text() if "ptxas" in found else ""
        shared, tmem = SHARED.search(text), TENSOR_MEMORY.search(text)
        registers, spills = REGISTERS.findall(log), SPILLS.findall(log)
        unread = tuple(field for field, read in (("shared bytes", shared), ("tensor memory columns", tmem),
                                                 ("registers", registers), ("spills", spills)) if not read)
        found_reports.append(Report(
            name, int(shared.group(1)) if shared else None, sum(map(int, tmem.groups())) if tmem else None,
            max(map(int, registers), default=None),
            max((int(stores) + int(loads) for stores, loads in spills), default=None), unread))
    return found_reports


THREADS_PER_MULTIPROCESSOR = {(8, 0): 2048, (9, 0): 2048, (10, 0): 2048}
"""Resident threads a multiprocessor holds by compute capability, 1536 on the others the targets name
(CUDA C++ Programming Guide, "Technical Specifications per Compute Capability")."""


def target_config(generation_name: str) -> str:
    """The `gpu_device_info` XLA compiles for, from the CUDA `generation`'s target
    (`generation.TARGETS`): its compute capability, the shared memory a block may opt into, its
    multiprocessors, clock and bandwidth; CUDA's per-capability limits for the rest. The fields past
    the capability and shared memory only steer XLA's own fusions, never a Triton kernel's code."""
    target, version = generation.gpu_target(generation_name), generation.capability(generation_name)
    shared, multiprocessors = target.shared_limit, target.multiprocessors
    clock, bandwidth = target.clock_hz, target.memory_bandwidth
    if version is None or shared is None or multiprocessors is None or clock is None or bandwidth is None:
        raise ValueError(f"no CUDA target states {generation_name!r}'s compile facts")
    major, minor = version
    return f"""gpu_device_info {{
  threads_per_block_limit: 1024
  threads_per_warp: 32
  shared_memory_per_block: 49152
  shared_memory_per_core: {shared + 1024}
  threads_per_core_limit: {THREADS_PER_MULTIPROCESSOR.get((major, minor), 1536)}
  core_count: {multiprocessors}
  fpus_per_core: {64 if (major, minor) == (8, 0) else 128}
  block_dim_limit_x: 2147483647
  block_dim_limit_y: 65535
  block_dim_limit_z: 65535
  memory_bandwidth: {int(bandwidth)}
  clock_rate_ghz: {clock / 1e9}
  shared_memory_per_block_optin: {shared}
  cuda_compute_capability {{
    major: {major}
    minor: {minor}
  }}
  registers_per_core_limit: 65536
  registers_per_block_limit: 65536
}}
platform_name: "CUDA"
device_description_str: "{target.device_kind}"
"""


@functools.cache
def compile_device(generation_name: str) -> Mesh:
    """A one-device mesh of a compile-only device of the CUDA `generation_name` (`target_config`): XLA's CUDA
    backend compiles for it with no GPU attached, through the compiler the GPU itself runs."""
    topology = topologies.get_topology_desc("dew-compile", "cuda", topology="1x1x1",
                                            target_config=target_config(generation_name))
    return Mesh(np.array(topology.devices[:1]), ("data",))


REFUSED = re.compile(r"Shared memory size limit exceeded: requested (\d+)")
"""XLA's refusal of a kernel whose shared memory a block of the GPU does not hold."""
