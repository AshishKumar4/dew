"""Mosaic GPU's own compile reports, what the templates' resource models are checked against: ptxas's
registers a thread and spilled bytes, and Mosaic's resource estimate (the shared bytes and tensor memory
columns a kernel asks for), written for each kernel Mosaic compiles while they are on (`reported`), at XLA's
compile of the program around it, with a device or a compile-only one (`dew.training.ahead`'s
`compile_only_gpus`). The CPU suite's compile gate reads them (tests/kernel_compile.py), and so does a run's
compile (`compiled`, owner 10-29, bible §6): a launch whose kernel spills is refused (`refused`) and its
program lowered once more, where the launch's derivation takes the next config by its own rule, each such
refusal a model miss (`misses`). Each kernel's name carries the launch that traced it (`named`), so its
reports name the launch and its config."""

import contextlib
import contextvars
import dataclasses
import hashlib
import logging
import os
import re
import tempfile
import threading
import types
from collections.abc import Callable, Hashable, Iterator
from pathlib import Path

import jax
import numpy as np
import tokamax

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


LAUNCH_SUFFIX = re.compile(r"__l([0-9a-f]{10})$")
"""The suffix a kernel's name carries for the launch that traced it (`named`): the digest of its launch."""

_log = logging.getLogger("dew.kernels")


@dataclasses.dataclass(frozen=True)
class Launch:
    """A template launch as it traced: its op (`tuning.KernelOp`), its bound arguments and its config."""
    op: Hashable
    arguments: tokamax.BoundArguments
    config: Hashable


_TRACING: contextvars.ContextVar[str | None] = contextvars.ContextVar("tracing_launch", default=None)
"""The digest of the launch tracing now (`launching`), which its kernels' names carry (`named`)."""
_LAUNCHES: dict[str, Launch] = {}
"""Each launch this process traced, by its digest."""
_REFUSALS: set[tuple[str, Hashable]] = set()
"""Each (launch identity, config) a compile refused this process (`compiled`): its kernel spilled."""


@dataclasses.dataclass(frozen=True)
class Miss:
    """A kernel a run's compile found spilling (`compiled`): its name, its launch's op and config (None
    where no launch of this process traced it), its registers and spilled bytes, and whether its launch's
    config was refused and its program lowered again (`stepped`)."""
    kernel: str
    op: str | None
    config: Hashable | None
    registers: int | None
    spilled: int | None
    stepped: bool


_MISSES: list[Miss] = []


def _identity(op: Hashable, arguments: tokamax.BoundArguments) -> str:
    """A launch's identity apart from its config: its op's type and fields and its arguments' shapes,
    dtypes and static values, a function by its code (closures of one definition share it)."""
    def described(value) -> str | tuple:
        if isinstance(value, types.FunctionType):
            return f"{value.__qualname__}@{id(value.__code__)}"
        if isinstance(value, (jax.Array, jax.ShapeDtypeStruct, np.ndarray)):
            return f"{tuple(value.shape)}:{value.dtype}"
        if isinstance(value, (tuple, list)):
            return tuple(described(part) for part in value)
        if isinstance(value, dict):
            return tuple((key, described(entry)) for key, entry in sorted(value.items()))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return (type(value).__qualname__, tuple((field.name, described(getattr(value, field.name)))
                                                    for field in dataclasses.fields(value)))
        return repr(value)
    return repr((described(op), described(dict(arguments.arguments))))


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=5).hexdigest()


@contextlib.contextmanager
def launching(op: Hashable, arguments: tokamax.BoundArguments, config: Hashable) -> Iterator[None]:
    """Within the block, the kernels a template's launch traces carry its digest in their names (`named`),
    and their reports name the launch (`Report.launch`, `traced`)."""
    digest = _digest(f"{_identity(op, arguments)}|{config!r}")
    _LAUNCHES.setdefault(digest, Launch(op, arguments, config))
    token = _TRACING.set(digest)
    try:
        yield
    finally:
        _TRACING.reset(token)


def named(name: str | None) -> str | None:
    """A kernel's name carrying the launch tracing it (`launching`; `kernel` for an unnamed one), as is
    where none is."""
    digest = _TRACING.get()
    return name if digest is None else f"{name or 'kernel'}__l{digest}"


def traced(digest: str) -> Launch | None:
    """The launch this process traced under `digest` (a report's `launch`), or None."""
    return _LAUNCHES.get(digest)


def refused(op: Hashable, arguments: tokamax.BoundArguments, config: Hashable) -> bool:
    """Whether a compile of this process refused `op`'s launch of `arguments` at `config`: its kernel
    spilled (`compiled`). A template's legality reads it, so its derivation takes the next config."""
    return bool(_REFUSALS) and (_identity(op, arguments), config) in _REFUSALS


def misses() -> tuple[Miss, ...]:
    """The kernels this process's compiles found spilling (`compiled`), for a run's telemetry."""
    return tuple(_MISSES)


@contextlib.contextmanager
def refusals() -> Iterator[None]:
    """Within the block, compiles refuse and record afresh, and the process's refusals and misses are as
    they were after it (a test's scope)."""
    refused_before, misses_before = set(_REFUSALS), list(_MISSES)
    _REFUSALS.clear()
    _MISSES.clear()
    try:
        yield
    finally:
        _REFUSALS.clear()
        _REFUSALS.update(refused_before)
        _MISSES[:] = misses_before


def compiled(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None = None, *,
             relower: Callable[[], jax.stages.Lowered]) -> jax.stages.Compiled:
    """`lowered` compiled, read through Mosaic's reports (`reported`): where a kernel of a template's launch
    spills, that launch's config is refused (`refused`), the miss recorded (`misses`, logged), and the
    program `relower`ed once and compiled; `relower` traces afresh, so each refused launch's derivation
    takes the next config by its own rule. A kernel spilling again is recorded, not stepped again. A
    program loaded from the persistent compilation cache runs no ptxas and reports nothing."""
    executable, spilling = _compiled_reported(lowered, options)
    stepped = False
    for found in spilling:
        launch = None if found.launch is None else traced(found.launch)
        if launch is not None:
            _REFUSALS.add((_identity(launch.op, launch.arguments), launch.config))
            stepped = True
        _missed(found, launch, stepped=launch is not None)
    if not stepped:
        return executable
    executable, spilling = _compiled_reported(relower(), options)
    for found in spilling:
        _missed(found, None if found.launch is None else traced(found.launch), stepped=False)
    return executable


def _compiled_reported(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None
                       ) -> tuple[jax.stages.Compiled, list["Report"]]:
    with tempfile.TemporaryDirectory() as directory:
        with reported(directory):
            executable = lowered.compile(options)
        return executable, [found for found in reports(directory) if found.spilled]


def _missed(found: "Report", launch: Launch | None, *, stepped: bool) -> None:
    miss = Miss(found.kernel, None if launch is None else type(launch.op).__name__,
                None if launch is None else launch.config, found.registers, found.spilled, stepped)
    _MISSES.append(miss)
    _log.warning("kernel %s spilled %s bytes at %s registers (%s at %s); %s", miss.kernel, miss.spilled,
                 miss.registers, miss.op, miss.config,
                 "its launch steps down" if stepped else "no step-down")


@dataclasses.dataclass(frozen=True)
class Report:
    """One Mosaic GPU kernel's reports: its name, the shared bytes and tensor memory columns it asks for
    (Mosaic's resource estimate), its registers a thread and its spilled bytes (ptxas's, the most over the
    kernel's functions), what the reports left `unread`: either report without the other, or a field
    neither states, is no evidence the kernel is within its resources; and its `module`, the digest of
    Mosaic's module after layout inference less its dump's name, which two compiles of one module share:
    Mosaic compiles a module once a process, so a later compile of it reports no ptxas run of its own; and
    the digest of the launch that traced it (`launching`), which its name carried, or None."""
    kernel: str
    shared: int | None
    tmem: int | None
    registers: int | None
    spilled: int | None
    unread: tuple[str, ...] = ()
    module: str | None = None
    launch: str | None = None

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


def kernel_name(stem: str) -> str:
    """A kernel's name from a report file's stem: less Mosaic's counter (`KERNEL_COUNTER`) and the launch
    its name carried (`LAUNCH_SUFFIX`)."""
    return LAUNCH_SUFFIX.sub("", KERNEL_COUNTER.sub("", stem))


def reports(directory: str) -> list[Report]:
    """Each kernel's `Report` in `directory`, its reports and module dump paired by the kernel's name less the
    counter Mosaic appends (`KERNEL_COUNTER`); the compiles a directory holds name each kernel once."""
    by_kernel: dict[str, dict[str, Path]] = {}
    for kind, suffix in (("resources", ".resources.txt"), ("ptxas", ".ptxas"),
                         ("module", ".after_layout_inference.txt")):
        for path in Path(directory).glob(f"*{suffix}"):
            found = by_kernel.setdefault(KERNEL_COUNTER.sub("", path.name.removesuffix(suffix)), {})
            # Paired by name: compiles that name a kernel twice would pair the wrong reports.
            assert kind not in found, f"{path.name} and {found[kind].name} compiled under one name"
            found[kind] = path
    found_reports = []
    for name, found in sorted(by_kernel.items()):
        text = found["resources"].read_text() if "resources" in found else ""
        log = found["ptxas"].read_text() if "ptxas" in found else ""
        shared, tmem = SHARED.search(text), TENSOR_MEMORY.search(text)
        registers, spills = REGISTERS.findall(log), SPILLS.findall(log)
        unread = tuple(field for field, read in (("shared bytes", shared), ("tensor memory columns", tmem),
                                                 ("registers", registers), ("spills", spills)) if not read)
        module = found.get("module")
        launch = LAUNCH_SUFFIX.search(name)
        found_reports.append(Report(
            LAUNCH_SUFFIX.sub("", name), int(shared.group(1)) if shared else None,
            sum(map(int, tmem.groups())) if tmem else None, max(map(int, registers), default=None),
            max((int(stores) + int(loads) for stores, loads in spills), default=None), unread,
            hashlib.sha256(DUMP_NAME.sub("", module.read_text()).encode()).hexdigest() if module else None,
            launch.group(1) if launch else None))
    return found_reports


REFUSED = re.compile(r"Shared memory size limit exceeded: requested (\d+)")
"""XLA's refusal of a kernel whose shared memory a block of the GPU does not hold."""
