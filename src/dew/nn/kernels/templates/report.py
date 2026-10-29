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
import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Callable, Hashable, Iterator
from pathlib import Path

import jax
import tokamax

from dew import cache

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
    """A template launch as it traced: its identity apart from its config (`_identity`), its op's name and
    its config, none of its traced values."""
    identity: str
    op: str
    config: Hashable


_TRACING: contextvars.ContextVar[str | None] = contextvars.ContextVar("tracing_launch", default=None)
"""The digest of the launch tracing now (`launching`), which its kernels' names carry (`named`)."""
_LAUNCHES: dict[str, Launch] = {}
"""Each launch this process traced, by its digest."""
@dataclasses.dataclass
class _Registry:
    """Each (launch identity, config) a compile refused (`compiled`: its kernel spilled), by the names of its
    kernels that spilled, and each miss."""
    refused: dict[tuple[str, Hashable], frozenset[str]] = dataclasses.field(default_factory=dict)
    missed: list["Miss"] = dataclasses.field(default_factory=list)


_PROCESS = _Registry()
"""The process's registry, every thread's, seeded from and kept in `records()` where it is set."""
_SEEDED: list[Path] = []
"""The records file the process's registry was seeded from (`_registry`), once."""
_SCOPE: contextvars.ContextVar[_Registry | None] = contextvars.ContextVar("refusals_scope", default=None)
"""A `refusals` scope's own registry, in its context alone."""
_GUARD = threading.RLock()
"""Serializes the registries' reads and updates between threads (a read may seed the process's)."""


def records() -> Path | None:
    """Where a process keeps its refusals beside the persistent compilation cache, so every process sharing
    the cache shares them and derives the configs it holds (a training pool's, an eval's, a server's):
    DEW_REFUSAL_RECORDS when set, else `dew-refusals.jsonl` in JAX's compilation cache directory, or nowhere
    when the process keeps no cache."""
    return cache.beside_compilation_cache("DEW_REFUSAL_RECORDS", "dew-refusals.jsonl")


def _registry() -> _Registry:
    scoped = _SCOPE.get()
    if scoped is not None:
        return scoped
    path = records()
    if path is not None and path not in _SEEDED:
        with _GUARD:
            if path not in _SEEDED:
                _SEEDED.append(path)
                for identity, config, kernels in _read(path):
                    _PROCESS.refused[(identity, config)] = (
                        _PROCESS.refused.get((identity, config), frozenset()) | kernels)
    return _PROCESS


def _read(path: Path) -> list[tuple[str, Hashable, frozenset[str]]]:
    """The refusals `path` keeps, a JSON line each; a line this checkout cannot rebuild (another checkout's
    config) is skipped."""
    from dew.nn.kernels import tuning  # tuning traces launches under this module's names
    if not path.exists():
        return []
    found = []
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
            found.append((entry["identity"], tuning.decoded(entry["config"]), frozenset(entry["kernels"])))
        except (ValueError, KeyError, TypeError, tuning.Unreadable):
            _log.warning("a refusal in %s this checkout cannot read: %.120s", path, line)
    return found


def _kept(identity: str, config: Hashable, kernels: frozenset[str]) -> None:
    """A process refusal appended to `records()`, one line written at once (O_APPEND), so processes sharing
    the file append whole lines."""
    from dew.nn.kernels import tuning  # tuning traces launches under this module's names
    path = records()
    if path is None:
        return
    line = json.dumps({"identity": identity, "config": tuning.encoded(config), "kernels": sorted(kernels)})
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, (line + "\n").encode())
    finally:
        os.close(descriptor)


class Exhausted(ValueError):
    """A derivation that finds no config left for a launch (`base.Unfitted`): a relowering after a spill
    that ends so keeps the first program (`compiled`)."""


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


def _identity(op, arguments: tokamax.BoundArguments) -> str:
    """A launch's identity apart from its config, the same in every process: its op as its picks name it
    (`tuning.KernelOp.keyed`) and its arguments exactly, its arrays by their own shapes and dtypes (no
    key's bucket: two launches a key shares compile two kernels) and its hooks by what they compute
    (`traced`)."""
    def described(name, value):
        if name in op.HOOKS:
            return op.traced(name, value)
        return jax.tree.map(lambda leaf: (tuple(leaf.shape), str(leaf.dtype)) if hasattr(leaf, "dtype")
                            and hasattr(leaf, "shape") else leaf, value)
    return repr((op.keyed(arguments).op, tuple((name, described(name, value))
                                               for name, value in sorted(arguments.arguments.items()))))


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=5).hexdigest()


@contextlib.contextmanager
def launching(op, arguments: tokamax.BoundArguments, config: Hashable) -> Iterator[None]:
    """Within the block, the kernels a template's launch traces carry its digest in their names (`named`),
    the same in every process, and their reports name the launch (`Report.launch`, `traced`)."""
    identity = _identity(op, arguments)
    digest = _digest(f"{identity}|{config!r}")
    with _GUARD:
        _LAUNCHES.setdefault(digest, Launch(identity, type(op).__name__, config))
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


def refused(op, arguments: tokamax.BoundArguments, config: Hashable) -> bool:
    """Whether a compile of this process refused `op`'s launch of `arguments` at `config`: its kernel
    spilled (`compiled`). A template's legality reads it, so its derivation takes the next config."""
    return bool(spilled(op, arguments, config))


def spilled(op, arguments: tokamax.BoundArguments, config: Hashable) -> frozenset[str]:
    """The names of `op`'s kernels a compile of this process found spilling at `config` for `arguments`
    (`refused` where any): a launch of several kernels (a recurrence's forward and backward walks) steps
    the one that spilled."""
    registry = _registry()
    if not registry.refused:
        return frozenset()
    identity = (_identity(op, arguments), config)
    with _GUARD:
        return registry.refused.get(identity, frozenset())


def refused_launches() -> tuple[tuple[str, Hashable, frozenset[str]], ...]:
    """Every (launch identity, config, spilled kernel names) refused in this context (a `refusals` scope's or
    the process's), ordered: a run records them with its plan, so every process loading it seeds them
    (`refuse`) and derives the stepped-down configs alike, and the compilation cache only holds those."""
    with _GUARD:
        return tuple(sorted(((identity, config, kernels)
                             for (identity, config), kernels in _registry().refused.items()), key=repr))


def refuse(entries) -> None:
    """Each `(identity, config, kernels)` of `entries` (`refused_launches`') refused in this context, as a
    compile's spill refuses it: a run seeding a plan's refusals before it lowers."""
    for identity, config, kernels in entries:
        _record((identity, config), None, frozenset(kernels))


def misses() -> tuple[Miss, ...]:
    """The kernels this process's compiles found spilling (`compiled`), for a run's telemetry: a `refusals`
    scope's own within it."""
    with _GUARD:
        return tuple(_registry().missed)


@contextlib.contextmanager
def refusals() -> Iterator[None]:
    """Within the block, this context's compiles refuse and record apart from the process's (a test's
    scope): its refusals and misses start empty and go with it, other threads' and the process's left as
    they are."""
    token = _SCOPE.set(_Registry())
    try:
        yield
    finally:
        _SCOPE.reset(token)


def _record(refusal: tuple[str, Hashable] | None, miss: "Miss | None",
            kernels: frozenset[str] = frozenset()) -> None:
    registry = _registry()
    with _GUARD:
        if refusal is not None:
            known = registry.refused.get(refusal, frozenset())
            registry.refused[refusal] = known | kernels
            if registry is _PROCESS and not kernels <= known:
                _kept(*refusal, kernels)
        if miss is not None:
            registry.missed.append(miss)


def compiled(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None = None, *,
             relower: Callable[[], jax.stages.Lowered]) -> jax.stages.Compiled:
    """`lowered` compiled, read through Mosaic's reports (`reported`): where a kernel of a template's launch
    spills, that launch's config is refused (`refused`), the miss recorded (`misses`, logged), and the
    program `relower`ed once and compiled; `relower` traces afresh, so each refused launch's derivation
    takes the next config by its own rule. A kernel spilling again is recorded, neither refused nor stepped
    again, so every process derives the program this compile keeps; a relowering whose derivations find no
    config left (`Exhausted`) keeps the first program, its misses recorded as not stepped. A program loaded from the persistent compilation cache runs no ptxas
    and reports nothing."""
    executable, spilling = _compiled_reported(lowered, options)
    found = _misses_of(spilling)
    refusals = [(refusal, report.kernel) for refusal, report, _ in found if refusal is not None]
    for refusal, kernel in refusals:
        _record(refusal, None, frozenset({kernel}))
    if not refusals:
        _recorded(found, stepped=False)
        return executable
    try:
        relowered = relower()
    except Exhausted as failed:
        _log.warning("relowering after a spill found no config: %s; the first program runs", failed)
        _recorded(found, stepped=False)
        return executable
    _recorded(found, stepped=True)
    executable, spilling = _compiled_reported(relowered, options)
    # The kept program's own spills are misses alone: refused, they would have every process seeded with
    # them derive a third program, not the one this compile keeps and runs.
    _recorded(_misses_of(spilling), stepped=False)
    return executable


def _misses_of(spilling: list["Report"]) -> list[tuple[tuple[str, Hashable] | None, "Report", Launch | None]]:
    """Each spilling kernel's refusal (its launch's identity and config, None for a kernel no launch of
    this process traced), its report and its launch."""
    found = []
    for report in spilling:
        launch = None if report.launch is None else traced(report.launch)
        found.append((None if launch is None else (launch.identity, launch.config), report, launch))
    return found


def _recorded(found: list, *, stepped: bool) -> None:
    """Each of `found`'s misses recorded and logged, stepped down where `stepped` and its kernel a
    launch's."""
    for _, report, launch in found:
        miss = Miss(report.kernel, None if launch is None else launch.op,
                    None if launch is None else launch.config, report.registers, report.spilled,
                    stepped and launch is not None)
        _record(None, miss)
        _log.warning("kernel %s spilled %s bytes at %s registers (%s at %s); %s", miss.kernel, miss.spilled,
                     miss.registers, miss.op, miss.config,
                     "its launch steps down" if miss.stepped else "no step-down")


def _compiled_reported(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None
                       ) -> tuple[jax.stages.Compiled, list["Report"]]:
    with tempfile.TemporaryDirectory() as directory:
        with reported(directory):
            executable = lowered.compile(options)
        return executable, [found for found in reports(directory) if found.spilled]


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
