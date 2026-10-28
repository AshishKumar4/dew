"""Mosaic GPU's own compile reports, what the templates' resource models are checked against: ptxas's
registers a thread and spilled bytes, and Mosaic's resource estimate (the shared bytes and tensor memory
columns a kernel asks for), written for each kernel Mosaic compiles while they are on (`reported`), at XLA's
compile of the program around it, with a device or a compile-only one (`dew.training.ahead`'s
`compile_only_gpus`). The CPU suite's compile gate reads them (tests/kernel_compile.py); a run never does:
its resources are derived from the models (`templates.resources`, owner 10-27)."""

import contextlib
import dataclasses
import hashlib
import os
import re
import threading
from pathlib import Path

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
DUMP_NAME = re.compile(r'mosaic_gpu\.dump_basename = "[^"]*"')
"""The one attribute of Mosaic's module dump that names the dump (its counter) rather than the module."""


@dataclasses.dataclass(frozen=True)
class Report:
    """One Mosaic GPU kernel's reports: its name, the shared bytes and tensor memory columns it asks for
    (Mosaic's resource estimate), its registers a thread and its spilled bytes (ptxas's, the most over the
    kernel's functions), what the reports left `unread`: either report without the other, or a field
    neither states, is no evidence the kernel is within its resources; and its `module`, the digest of
    Mosaic's module after layout inference less its dump's name, which two compiles of one module share:
    Mosaic compiles a module once a process, so a later compile of it reports no ptxas run of its own."""
    kernel: str
    shared: int | None
    tmem: int | None
    registers: int | None
    spilled: int | None
    unread: tuple[str, ...] = ()
    module: str | None = None

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
def reported(directory: str, *, ptx: bool = False):
    """Mosaic GPU's reports on, into `directory`, for the compiles in the block, each kernel's PTX too where
    `ptx` asks (the compile gate's register need), under `DUMPING` from before the settings are read to
    after they are restored."""
    with DUMPING:
        dumps = {**DUMPS, **({"MOSAIC_GPU_DUMP_PTX": "1"} if ptx else {})}
        before = {name: os.environ.get(name) for name in (*dumps, "MOSAIC_GPU_DUMP_TO")}
        os.environ.update(dumps, MOSAIC_GPU_DUMP_TO=directory)
        try:
            yield
        finally:
            for name, value in before.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def reports(directory: str) -> list[Report]:
    """Each kernel's `Report` in `directory`, its reports and module dump paired by the kernel's name less the
    counter Mosaic appends (`KERNEL_COUNTER`); the compiles a directory holds name each kernel once."""
    named: dict[str, dict[str, Path]] = {}
    for kind, suffix in (("resources", ".resources.txt"), ("ptxas", ".ptxas"),
                         ("module", ".after_layout_inference.txt")):
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
        module = found.get("module")
        found_reports.append(Report(
            name, int(shared.group(1)) if shared else None, sum(map(int, tmem.groups())) if tmem else None,
            max(map(int, registers), default=None),
            max((int(stores) + int(loads) for stores, loads in spills), default=None), unread,
            hashlib.sha256(DUMP_NAME.sub("", module.read_text()).encode()).hexdigest() if module else None))
    return found_reports


REFUSED = re.compile(r"Shared memory size limit exceeded: requested (\d+)")
"""XLA's refusal of a kernel whose shared memory a block of the GPU does not hold."""
