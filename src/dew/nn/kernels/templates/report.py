"""Mosaic GPU's own compile reports, the resource authority for the templates' kernels: ptxas's registers a
thread and spilled bytes, and Mosaic's resource estimate (the shared bytes and tensor memory columns a kernel
asks for), written for each kernel Mosaic compiles while they are on (`reported`), at XLA's compile of the
program around it, with a device or a compile-only one. The CPU suite's resource gates read them
(tests/kernel_compile.py), as a derivation does to confirm what it planned."""

import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import jax
import jax.experimental.topologies as topologies
import numpy as np
from jax.sharding import AbstractDevice, AbstractMesh, Mesh, NamedSharding, PartitionSpec

from dew import cache
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
    facts = (target.shared_limit, target.multiprocessors, target.clock_hz, target.memory_bandwidth)
    if version is None or any(fact is None for fact in facts):
        raise ValueError(f"no CUDA target states {generation_name!r}'s compile facts")
    major, minor = version
    shared, multiprocessors, clock, bandwidth = facts
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


@functools.cache
def toolchain() -> str:
    """The compilers a report depends on: JAX's and jaxlib's versions and ptxas's own version line, which
    Mosaic GPU runs (JAX's CUDA path, `jax._src.lib.cuda_path`)."""
    from jax._src import lib

    ptxas = Path(lib.cuda_path or "") / "bin" / "ptxas"
    found = subprocess.run([str(ptxas), "--version"], capture_output=True, text=True, check=False) \
        if ptxas.exists() else None
    version = found.stdout.strip().splitlines()[-1] if found and found.stdout.strip() else "no ptxas"
    return f"jax {jax.__version__}, jaxlib {lib.version_str}, {version}"


@functools.cache
def source() -> str:
    """A digest of Dew's own source, each module of the package by its path: what a call's kernels are
    built from beside the toolchain, so reports keyed by the call (`confirmed`'s `call`) hold for this code
    alone."""
    package = Path(cache.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(f"{path.relative_to(package)}\n".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


_CONFIRMED: dict[str, tuple[Report, ...]] = {}
_CONFIRMING = contextvars.ContextVar("dew_confirming", default=False)


def confirming() -> bool:
    """Whether this trace is a confirmation's own (`confirmed`), whose kernel takes the config it is given."""
    return _CONFIRMING.get()


def confirmed(fn: Callable, args, generation_name: str, *, call: str,
              compile: Callable | None = None) -> tuple[Report, ...]:
    """The reports of `fn`'s Mosaic GPU kernels at abstract `args`, compiled for the CUDA `generation_name`
    by the compiler that GPU runs: the compiler's word on a derived config. `call` names the launch whole,
    the same text in every process (its kernel, config, operands and static arguments,
    `tuning.launch_text`), and keys the reports with the generation, the toolchain (`toolchain`) and Dew's
    source (`source`); they persist under Dew's cache (`cache.dew_cache_dir`), so a launch compiles for
    this once per toolchain and checkout, and a process that has them finds them without tracing `fn`
    (0.9 s of a first trace at an A100's grouped forward where the module keyed them, CONFIRMCOST1).
    Where none is kept, `fn` is exported for CUDA as that GPU lowers it (`jax.export`) and compiled in a
    process of its own (`_compiled_apart`), where ptxas runs whatever this process compiled before: a
    kernel compiled here once, or read from JAX's cache, runs no ptxas and states no registers
    (MWARPS3-a100's suite on its own GPU). A compile XLA refuses for its shared memory reports it
    (`Report.unread`); any other failure raises. `compile(exported, generation_name, directory)` compiles
    in place of `_compiled_apart`."""
    digest = _digest(call, generation_name, toolchain(), source())
    if (found := _stored(digest)) is not None:
        return found
    with tempfile.TemporaryDirectory(prefix="dew-confirm-") as directory:
        # Mosaic GPU states a kernel's resources as it lowers it, and ptxas its registers as it compiles.
        with reported(directory):
            exported = _exported(fn, args, generation_name)
        found = (compile or _compiled_apart)(exported, generation_name, directory)
    _store(digest, found)
    return found


def _digest(*parts: str) -> str:
    """The key of a confirmation's reports: `parts`' digest."""
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _stored(digest: str) -> tuple[Report, ...] | None:
    """The reports kept under `digest`, this process's or Dew's cache's, None where neither has them."""
    if digest in _CONFIRMED:
        return _CONFIRMED[digest]
    path = Path(cache.dew_cache_dir()) / "kernel-reports" / f"{digest}.json"
    if not path.exists():
        return None
    found = tuple(Report(**{**entry, "unread": tuple(entry["unread"])})
                  for entry in json.loads(path.read_text()))
    _CONFIRMED[digest] = found
    return found


def _store(digest: str, found: tuple[Report, ...]) -> None:
    """Keep `found` under `digest`, in this process and in Dew's cache, whole or not at all."""
    path = Path(cache.dew_cache_dir()) / "kernel-reports" / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    written = path.with_suffix(f".{os.getpid()}.tmp")
    written.write_text(json.dumps([dataclasses.asdict(entry) for entry in found]))
    written.replace(path)
    _CONFIRMED[digest] = found


def _exported(fn: Callable, args, generation_name: str):
    """`fn` at `args` exported for CUDA as the `generation_name` GPU lowers it, on an abstract device that
    states its compute capability (sm90's too, which `generation.gpu_mesh` leaves unstated; the module is
    the same, STATEDCC1), so Mosaic GPU checks a kernel's shared memory against that GPU's, not an
    attached GPU's of another capability (an A100's 166912 bytes for an H100's kernel, MWARPS3-a100)."""
    mesh = compile_device(generation_name)
    placed = [jax.ShapeDtypeStruct(arg.shape, arg.dtype, sharding=NamedSharding(mesh, PartitionSpec()))
              for arg in args]
    version = generation.capability(generation_name)
    assert version is not None, f"{generation_name} is a CUDA generation (`compile_device`)"
    device = AbstractDevice(device_kind=generation.gpu_target(generation_name).device_kind, num_cores=None,
                            platform="cuda", compute_capability=".".join(map(str, version)))
    stated = AbstractMesh(mesh.axis_sizes, mesh.axis_names, axis_types=mesh.axis_types,
                          abstract_device=device)
    token = _CONFIRMING.set(True)
    try:
        with generation.assumed(generation_name), jax.sharding.use_abstract_mesh(stated):
            return jax.export.export(jax.jit(fn), platforms=("cuda",))(*placed)
    finally:
        _CONFIRMING.reset(token)


def _compiled(exported, generation_name: str, directory: str) -> tuple[Report, ...]:
    """`exported`'s kernels' reports, compiled in this process for a compile-only device of
    `generation_name` into `directory` (beside the resource estimates its lowering stated there): the body
    of `_compiled_apart`'s process."""
    mesh = compile_device(generation_name)
    placed = [jax.ShapeDtypeStruct(aval.shape, aval.dtype, sharding=NamedSharding(mesh, PartitionSpec()))
              for aval in exported.in_avals]
    with reported(directory):
        try:
            with generation.assumed(generation_name), \
                    jax.sharding.use_abstract_mesh(generation.gpu_mesh(mesh, generation_name)):
                jax.jit(exported.call).trace(*placed).lower().compile()
        except jax.errors.JaxRuntimeError as error:
            if (refused := REFUSED.search(str(error))) is None:
                raise
            return (Report("refused", int(refused.group(1)), 0, None, None, ("its compile",)),)
        return reports(directory)


def _compiled_apart(exported, generation_name: str, directory: str) -> tuple[Report, ...]:
    """`_compiled` in a process of its own, on the CPU whatever GPU is attached (none of its memory taken)
    and with JAX's compilation cache off, so ptxas compiles each kernel and reports it; its first Mosaic
    GPU compile is the generation's (Mosaic GPU's binary pass keeps a process's first capability:
    tests/kernel_compile.py)."""
    environment = {name: value for name, value in os.environ.items() if name != "JAX_COMPILATION_CACHE_DIR"}
    environment.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", JAX_ENABLE_COMPILATION_CACHE="false")
    with tempfile.TemporaryDirectory(prefix="dew-confirm-apart-") as apart:
        blob, found = Path(apart) / "exported", Path(apart) / "reports.json"
        blob.write_bytes(exported.serialize())
        ran = subprocess.run([sys.executable, "-m", "dew.nn.kernels.templates.report", generation_name,
                              str(blob), str(found), directory], capture_output=True, text=True, check=False,
                             env=environment)
        if ran.returncode != 0 or not found.exists():
            raise RuntimeError(f"confirming a kernel for {generation_name} failed in its own process:\n"
                               f"{(ran.stdout + ran.stderr)[-3000:]}")
        return tuple(Report(**{**entry, "unread": tuple(entry["unread"])})
                     for entry in json.loads(found.read_text()))


if __name__ == "__main__":
    _generation, _blob, _found, _directory = sys.argv[1:5]
    _reports = _compiled(jax.export.deserialize(bytearray(Path(_blob).read_bytes())), _generation, _directory)
    Path(_found).write_text(json.dumps([dataclasses.asdict(entry) for entry in _reports]))
