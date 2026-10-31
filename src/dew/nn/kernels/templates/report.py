"""What Mosaic GPU's compiles say of each kernel, what the templates' resource models are checked against.
A run's compile (`compiled`, owner 10-29, bible §6) reads each kernel's registers a thread and its stack
frame from the binary its executable carries (`binaries`): a launch whose kernel spills is refused
(`refused`) and its program lowered once more, where the launch's derivation takes the next config by its
own rule, each such refusal a model miss (`misses`). The CPU suite's compile gate reads Mosaic's dumped
reports and PTX instead (tests/kernel_compile.py), as it scans ptxas's caps and needs Mosaic's resource
estimate. Each kernel's name carries the launch that traced it (`named`), so a binary names its launch and
its config."""

import contextlib
import contextvars
import dataclasses
import hashlib
import logging
import re
import struct
import threading
from collections.abc import Callable, Hashable, Iterator

import jax
import tokamax
from google.protobuf import empty_pb2
from google.protobuf.message import DecodeError
from google.protobuf.unknown_fields import UnknownFieldSet
from jax.extend.mlir import ir

from dew.nn.kernels import generation

LAUNCH_SUFFIX = re.compile(r"__l([0-9a-f]{10})$")
"""The suffix a kernel's name carries for the launch that traced it (`named`): the digest of its launch."""
KERNEL_FUNCTION = re.compile(r"(?P<kernel>.*)_mosaic_gpu_kernel")
"""A Mosaic GPU kernel's function in its binary: its kernel's name, the host function's `_mosaic_gpu`
(jax/experimental/mosaic/gpu/core.py) and GPU outlining's `_kernel`."""
KERNEL_PROTO_VERSION = 4
"""The `MosaicGpuKernelProto` version jaxlib serializes (mosaic/gpu/custom_call.cc), which it alone reads."""
MOSAIC_TARGET = b"mosaic_gpu_v2"
"""The custom call target of a Mosaic GPU kernel, which its thunk names (`_kernels`)."""
REGISTER_COUNT, FRAME_SIZE = 0x2F, 0x11
"""A cubin's `.nv.info` attributes for a function's registers a thread and its stack frame's bytes
(EIATTR_REGCOUNT, EIATTR_FRAME_SIZE), as ptxas states them: its frame holds what its registers spill."""

_log = logging.getLogger("dew.kernels")


@dataclasses.dataclass(frozen=True)
class Launch:
    """A template launch as it traced: its identity apart from its config (`_identity`), its op's name and
    its config, none of its traced values."""
    identity: str
    op: str
    config: Hashable


@dataclasses.dataclass
class _Pending:
    """The launch tracing now. Its identity is evaluated only if a Mosaic kernel asks for its name;
    bound arguments live in this tracing scope, never in the process's launch registry."""
    identity: Callable[[], str]
    op: str
    config: Hashable
    digest: str | None = None


_TRACING: contextvars.ContextVar[_Pending | None] = contextvars.ContextVar("tracing_launch", default=None)
"""The pending launch whose Mosaic kernels acquire their digest through `named`."""
_LAUNCHES: dict[str, Launch] = {}
"""Each launch this process traced, by its digest."""
@dataclasses.dataclass
class _Registry:
    """Each (launch identity, config) a compile refused (`compiled`: its kernel spilled), by the names of its
    kernels that spilled, and each miss."""
    refused: dict[tuple[str, Hashable], frozenset[str]] = dataclasses.field(default_factory=dict)
    missed: list["Miss"] = dataclasses.field(default_factory=list)


_PROCESS = _Registry()
"""The process's registry, every thread's. A shared executable cache does not share these decisions:
only an explicitly handed plan seeds them, after validating its program and target."""
_SCOPE: contextvars.ContextVar[_Registry | None] = contextvars.ContextVar("refusals_scope", default=None)
"""A `refusals` scope's own registry, in its context alone."""
_GUARD = threading.RLock()
"""Serializes the registries' reads and updates between threads."""


def _registry() -> _Registry:
    return _SCOPE.get() or _PROCESS


class Exhausted(ValueError):
    """A derivation that finds no config left for a launch (`base.Unfitted`): a relowering after a spill
    that ends so keeps the first program (`compiled`)."""


@dataclasses.dataclass(frozen=True)
class Miss:
    """A kernel a run's compile found spilling (`compiled`): its name, its launch's op and config (None
    where no launch of this process traced it), its registers and its stack frame's bytes a thread
    (`Binary.frame`), and whether its launch's config was refused and its program lowered again
    (`stepped`)."""
    kernel: str
    op: str | None
    config: Hashable | None
    registers: int | None
    frame: int | None
    stepped: bool


def _identity(op, arguments: tokamax.BoundArguments) -> str:
    """A launch's identity apart from its config, the same in every process: its op as its picks name it
    (`tuning.KernelOp.keyed`) and its arguments exactly, its arrays by their own shapes and dtypes (no
    key's bucket: two launches a key shares compile two kernels) and its hooks by what they compute
    (`traced`), and the target capacities that size its generated program. Clock rates are not code inputs
    once its config is chosen: an attached device and its compile-only target may report different rates
    while emitting exactly that config."""
    def described(name, value):
        if name in op.HOOKS:
            return op.traced(name, value)
        return jax.tree.map(lambda leaf: (tuple(leaf.shape), str(leaf.dtype)) if hasattr(leaf, "dtype")
                            and hasattr(leaf, "shape") else leaf, value)
    target = generation.gpu_target()
    hardware = ((generation.device_generation(),) if target is None else
                (target.generation, target.multiprocessors, target.shared_limit))
    operands = tuple((name, described(name, value)) for name, value in sorted(arguments.arguments.items()))
    return repr((hardware, op.keyed(arguments).op, operands))


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=5).hexdigest()


@contextlib.contextmanager
def launching(op, arguments: tokamax.BoundArguments, config: Hashable) -> Iterator[None]:
    """Within the block, the kernels a template's launch traces carry its digest in their names (`named`),
    the same in every process, and their reports name the launch (`Report.launch`, `traced`)."""
    token = _TRACING.set(_Pending(lambda: _identity(op, arguments), type(op).__name__, config))
    try:
        yield
    finally:
        _TRACING.reset(token)


def named(name: str | None) -> str | None:
    """A kernel's name carrying the launch tracing it (`launching`; `kernel` for an unnamed one), as is
    where none is."""
    pending = _TRACING.get()
    if pending is None:
        return name
    if pending.digest is None:
        identity = pending.identity()
        digest = _digest(f"{identity}|{pending.config!r}")
        with _GUARD:
            _LAUNCHES.setdefault(digest, Launch(identity, pending.op, pending.config))
        pending.digest = digest
    return f"{name or 'kernel'}__l{pending.digest}"


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
    the process's), ordered. A validated plan can hand them to another process explicitly (`refuse` within
    `refusals`); a shared compilation cache alone must not transfer them between programs."""
    with _GUARD:
        return tuple(sorted(((identity, config, kernels)
                             for (identity, config), kernels in _registry().refused.items()), key=repr))


def refuse(entries) -> None:
    """Each `(identity, config, kernels)` of `entries` (`refused_launches`') refused in this context, as a
    compile's spill refuses it. A handed plan seeds these inside its own `refusals` scope, not unioned with
    unrelated decisions that the receiving process already holds."""
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


@contextlib.contextmanager
def _proposing(entries: list[tuple[tuple[str, Hashable], str]]) -> Iterator[None]:
    """Refusals visible while deriving and compiling a replacement, without changing the kept program's
    registry or its file. A successful replacement commits them after this scope ends."""
    registry = _registry()
    with _GUARD:
        proposed = dict(registry.refused)
    for identity, kernel in entries:
        proposed[identity] = proposed.get(identity, frozenset()) | {kernel}
    token = _SCOPE.set(_Registry(refused=proposed))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def _record(refusal: tuple[str, Hashable] | None, miss: "Miss | None",
            kernels: frozenset[str] = frozenset()) -> None:
    registry = _registry()
    with _GUARD:
        if refusal is not None:
            registry.refused[refusal] = registry.refused.get(refusal, frozenset()) | kernels
        if miss is not None:
            registry.missed.append(miss)


def compiled(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None = None, *,
             relower: Callable[[], jax.stages.Lowered]) -> jax.stages.Compiled:
    """`lowered` compiled, read through its kernels' binaries (`binaries`): where a kernel of a template's
    launch spills, that launch's config is refused (`refused`), the miss recorded (`misses`, logged), and the
    program `relower`ed once and compiled; `relower` traces afresh, so each refused launch's derivation
    takes the next config by its own rule. A kernel spilling again is recorded, neither refused nor stepped
    again, so every process derives the program this compile keeps; a relowering whose derivations find no
    config left (`Exhausted`) keeps the first program, its misses recorded as not stepped. Refusals commit
    only when a replacement compiles, so a later process derives the program kept here. A program loaded
    from the persistent compilation cache carries its kernels' binaries as one compiled here does."""
    executable, spilling = _compiled_reported(lowered, options)
    found = _misses_of(spilling)
    refusals = [(refusal, report.kernel) for refusal, report, _ in found if refusal is not None]
    if not refusals:
        _recorded(found, stepped=False)
        return executable
    try:
        with _proposing(refusals):
            relowered = relower()
            replacement, spilling = _compiled_reported(relowered, options)
    except Exhausted as failed:
        _log.warning("relowering after a spill found no config: %s; the first program runs", failed)
        _recorded(found, stepped=False)
        return executable
    for refusal, kernel in refusals:
        _record(refusal, None, frozenset({kernel}))
    _recorded(found, stepped=True)
    # The kept program's own spills are misses alone: refused, they would have every process seeded with
    # them derive a third program, not the one this compile keeps and runs.
    _recorded(_misses_of(spilling), stepped=False)
    return replacement


def _misses_of(spilling: list["Binary"]) -> list[tuple[tuple[str, Hashable] | None, "Binary", Launch | None]]:
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
                    None if launch is None else launch.config, report.registers, report.frame,
                    stepped and launch is not None)
        _record(None, miss)
        _log.warning("kernel %s spills to a %s-byte stack frame at %s registers (%s at %s); %s", miss.kernel,
                     miss.frame, miss.registers, miss.op, miss.config,
                     "its launch steps down" if miss.stepped else "no step-down")


def _compiled_reported(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None
                       ) -> tuple[jax.stages.Compiled, list["Binary"]]:
    from dew.training._lowering import kernel_calls

    executable = lowered.compile(options)
    if not any(call.kind == "mosaic_gpu" for call in kernel_calls(lowered)):
        return executable, []
    return executable, [found for found in binaries(executable) if found.frame]


@dataclasses.dataclass(frozen=True)
class Binary:
    """A Mosaic GPU kernel's binary as its executable carries it (`binaries`): its name and the launch that
    traced it (`Report.kernel`, `Report.launch`), its registers a thread and its stack frame's bytes a
    thread, nonzero where its registers spill (`REGISTER_COUNT`, `FRAME_SIZE`)."""
    kernel: str
    launch: str | None
    registers: int
    frame: int


def binaries(executable: jax.stages.Compiled) -> list[Binary]:
    """Every Mosaic GPU kernel binary `executable` carries, read from its own serialized form: XLA keeps each
    kernel's cubin in its custom call's state (`MosaicGpuKernelProto`), whether this compile ran ptxas for
    it, another thread's program compiled it first (Mosaic compiles a module once a process), or the
    executable was loaded from the persistent cache. Nothing is read from or written to the process's
    environment or disk: a compile that set Mosaic's dump settings around itself handed its directory to
    every other thread's compile and removed it under them ('Directory not empty', row 2790), and the
    settings turn on every dump Mosaic has (row 2816)."""
    from jax._src.thunky import thunky

    found: list[Binary] = []
    context = thunky.make_ir_context()
    with context:
        module = thunky.jax_executable_to_mlir(executable, context)

        def visit(operation: ir.Operation) -> ir.WalkResult:
            if operation.name == "thunky.call_thunk_proto":
                kernels, calls = _kernels(ir.StringAttr(operation.attributes["thunk_proto"]).value_bytes)
                if len(kernels) != calls:
                    # A call whose binary is missing would read as a kernel that does not spill.
                    raise ValueError(f"a thunk names {calls} Mosaic GPU calls but carries {len(kernels)} "
                                     "kernel binaries")
                found.extend(kernels)
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)
    return found


def _kernels(message: bytes) -> tuple[list[Binary], int]:
    """The kernels of each `MosaicGpuKernelProto` within a thunk's serialized proto (a message of jaxlib's
    version, field 1, whose binary, 8, is an ELF and whose kernel name, 9, its function's), and how many
    Mosaic GPU custom calls it names (`MOSAIC_TARGET`, a custom call thunk's target)."""
    parsed = empty_pb2.Empty()
    try:
        parsed.ParseFromString(message)
    except DecodeError:
        return [], 0
    fields: dict[int, list] = {}
    for field in UnknownFieldSet(parsed):
        fields.setdefault(field.field_number, []).append(field.data)
    binary, name = fields.get(8, []), fields.get(9, [])
    if (fields.get(1) == [KERNEL_PROTO_VERSION] and len(binary) == len(name) == 1
            and isinstance(binary[0], bytes) and binary[0][:4] == b"\x7fELF" and isinstance(name[0], bytes)):
        return [_binary(name[0].decode(), binary[0])], 0
    found, calls = [], 0
    for values in fields.values():
        for value in values:
            if value == MOSAIC_TARGET:
                calls += 1
            elif isinstance(value, bytes):
                kernels, named = _kernels(value)
                found += kernels
                calls += named
    return found, calls


def _binary(function: str, elf: bytes) -> Binary:
    """The `Binary` of the kernel `function` from its cubin `elf`: its `.nv.info` attributes' registers and
    frame (`REGISTER_COUNT`, `FRAME_SIZE`; a Mosaic module compiles one kernel function)."""
    (sections_at,) = struct.unpack_from("<Q", elf, 0x28)
    entry, count, names = struct.unpack_from("<HHH", elf, 0x3A)
    headers = [struct.unpack_from("<IIQQQQIIQQ", elf, sections_at + index * entry) for index in range(count)]
    names_at = headers[names][4]
    stated: dict[int, int] = {}
    for name, _, _, _, offset, size, *_ in headers:
        title = elf[names_at + name:elf.index(b"\0", names_at + name)]
        if title != b".nv.info" and not title.startswith(b".nv.info."):
            continue
        at = offset
        while at + 4 <= offset + size:
            form, attribute, length = struct.unpack_from("<BBH", elf, at)
            # A sized attribute (EIFMT_SVAL) holds its symbol and value after its header; the others are
            # their header alone.
            if form == 0x04:
                if attribute in (REGISTER_COUNT, FRAME_SIZE):
                    (value,) = struct.unpack_from("<I", elf, at + 8)
                    stated[attribute] = max(stated.get(attribute, 0), value)
                at += 4 + length
            else:
                at += 4
    if REGISTER_COUNT not in stated:
        raise ValueError(f"the binary of {function} states no registers")
    named = KERNEL_FUNCTION.fullmatch(function)
    kernel = named.group("kernel") if named else function
    launch = LAUNCH_SUFFIX.search(kernel)
    return Binary(LAUNCH_SUFFIX.sub("", kernel), launch.group(1) if launch else None,
                  stated[REGISTER_COUNT], stated.get(FRAME_SIZE, 0))
