"""What Mosaic GPU's compiles say of each kernel, what the templates' resource models are checked against.
A run's compile (`compiled`, owner 10-29, bible §6) reads each kernel's registers a thread and its stack
frame from the binary its executable carries (`binaries`): a launch whose kernel spills is refused
(`refused`) and its program lowered once more, where the launch's derivation takes the next config by its
own rule, each such refusal a model miss (`misses`). The CPU suite's compile gate reads the binaries too,
beside the resource estimate Pallas prints as it lowers a kernel and the PTX Mosaic prints as it compiles
one, whose registers and spills ptxas states at the GPU's own compile (tests/kernel_compile.py). Each
kernel's name carries the launch that traced it (`named`), so a binary names its launch and its config."""

import contextlib
import contextvars
import dataclasses
import hashlib
import importlib.util
import logging
import re
import shutil
import struct
import subprocess
import tempfile
import threading
from collections.abc import Callable, Hashable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import overload

import jax
import tokamax
from google.protobuf import empty_pb2
from google.protobuf.message import DecodeError
from google.protobuf.unknown_fields import UnknownFieldSet
from jax.extend.mlir import ir

from dew import communicators
from dew.nn.kernels import generation

SHARED_REFUSED = re.compile(r"Shared memory size limit exceeded: requested (\d+)")
"""XLA's refusal of a program one of whose Triton kernels asks more shared memory than a block holds,
which names no kernel (`shared_asked`)."""


def shared_asked(error: BaseException) -> int | None:
    """The shared bytes a Triton kernel of a program XLA refused asked past a block, or None where `error`
    is not that refusal: the one reading of XLA's text (`kept`, `tuning`'s fitting, the compile-only
    harness)."""
    asked = SHARED_REFUSED.search(str(error))
    return None if asked is None else int(asked.group(1))


LAUNCH_SUFFIX = re.compile(r"__l([0-9a-f]{10})$")
"""The suffix a kernel's name carries for the launch that traced it (`named`): the digest of its launch."""
KERNEL_FUNCTION = re.compile(r"(?P<kernel>.*)_mosaic_gpu_kernel")
"""A Mosaic GPU kernel's function in its binary: its kernel's name, the host function's `_mosaic_gpu`
(jax/experimental/mosaic/gpu/core.py) and GPU outlining's `_kernel`."""


def kernel_of(name: str) -> tuple[str, str | None]:
    """`(kernel, launch)` of a kernel's `name` as it traced (`named`), its function in its binary or its PTX
    entry (`KERNEL_FUNCTION`) or its call's location: its name less the launch's suffix, and the launch's
    digest, None where it carries none."""
    function = KERNEL_FUNCTION.fullmatch(name)
    kernel = function.group("kernel") if function else name
    launch = LAUNCH_SUFFIX.search(kernel)
    return LAUNCH_SUFFIX.sub("", kernel), launch.group(1) if launch else None
KERNEL_PROTO_VERSION = 4
"""The `MosaicGpuKernelProto` version jaxlib serializes (mosaic/gpu/custom_call.cc), which it alone reads."""
MOSAIC_TARGET = b"mosaic_gpu_v2"
"""The custom call target of a Mosaic GPU kernel, which its thunk names (`_kernels`)."""
SHARED_BYTES = 10
"""The `MosaicGpuKernelProto` field of the dynamic shared memory a kernel's launch asks for (`smem_bytes`)."""
CUSTOM_KERNEL_THUNK = 36
"""The `ThunkProto` field of a custom kernel's thunk, a Pallas-Triton launch's (`CustomKernelThunkProto`): its
kernel (3, `CustomKernelProto`), whose loader spec (2) holds its cubin (2, `CudaCubinProto`; 1) and which
states the dynamic shared memory its launch asks for (6, `shared_memory_bytes`)."""
ELF = b"\x7fELF"
"""A cubin's first bytes."""
REGISTER_COUNT, FRAME_SIZE = 0x2F, 0x11
"""A cubin's `.nv.info` attributes for a function's registers a thread and its stack frame's bytes
(EIATTR_REGCOUNT, EIATTR_FRAME_SIZE), as ptxas states them: its frame holds what its registers spill."""

_log = logging.getLogger("dew.kernels")


@dataclasses.dataclass(frozen=True)
class Launch:
    """A template launch as it traced: its identity apart from its config (`_identity`), its op's name and
    its config, none of its traced values; and the share of its tile grid its tails pad
    (`tuning.KernelOp.padded`), None where its op states none."""
    identity: str
    op: str
    config: Hashable
    padded: float | None = None


@dataclasses.dataclass
class _Pending:
    """The launch tracing now. Its identity is evaluated only if a Mosaic kernel asks for its name;
    bound arguments live in this tracing scope, never in the process's launch registry."""
    identity: Callable[[], str]
    op: str
    config: Hashable
    padded: Callable[[], float | None]
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
    rewritten: list = dataclasses.field(default_factory=list)
    """Each region the algebra pass rewrote, its report (`dew.algebra.program.AlgebraReport`)."""
    passes: list[float] = dataclasses.field(default_factory=list)
    """Each program's whole pass, in seconds: its walks of its bodies and windows beside its regions'."""


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
    """A kernel a run's compile found spilling, or a program whose block XLA refused (`compiled`): its
    kernel's name, its launch's op and config (None where no launch of this process traced it), its
    registers and its stack frame's bytes a thread (`Binary.frame`) or the shared bytes it asked
    (`shared`), and whether its launch's config was refused and its program lowered again (`stepped`)."""
    kernel: str
    op: str | None
    config: Hashable | None
    registers: int | None
    frame: int | None
    stepped: bool
    shared: int | None = None


def _identity(op, arguments: tokamax.BoundArguments) -> str:
    """A launch's identity apart from its config, the same in every process: its op as its picks name it
    (`tuning._identity`) and its arguments exactly, its arrays by their own shapes and dtypes (no key's
    bucket: two launches a key shares compile two kernels), its hooks, the op's and its arguments', by what
    they compute as their parts tell it (`tuning.KernelOp.identified`: every launch reads it, so it traces
    nothing), and the target capacities that size its generated program. Clock rates are not code inputs
    once its config is chosen: an attached device and its compile-only target may report different rates
    while emitting exactly that config."""
    def described(name, value):
        if name in op.HOOKS:
            return op.identified(name, value)
        return jax.tree.map(lambda leaf: (tuple(leaf.shape), str(leaf.dtype)) if hasattr(leaf, "dtype")
                            and hasattr(leaf, "shape") else leaf, value)
    target = generation.gpu_target()
    hardware = ((generation.device_generation(),) if target is None else
                (target.generation, target.multiprocessors, target.shared_limit))
    # The tuner imports this module: read at a call.
    from dew.nn.kernels import tuning

    operands = tuple((name, described(name, value)) for name, value in sorted(arguments.arguments.items()))
    return repr((hardware, tuning._identity(op, traced=False), operands))


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=5).hexdigest()


@contextlib.contextmanager
def launching(op, arguments: tokamax.BoundArguments, config: Hashable) -> Iterator[None]:
    """Within the block, the kernels a template's launch traces carry its digest in their names (`named`),
    the same in every process, and their binaries name the launch (`Binary.launch`, `traced`)."""
    token = _TRACING.set(_Pending(lambda: _identity(op, arguments), type(op).__name__, config,
                                  lambda: op.padded(arguments, config)))
    try:
        yield
    finally:
        _TRACING.reset(token)


@contextlib.contextmanager
def relaunched(digest: str | None) -> Iterator[None]:
    """Within the block, the kernels traced carry the launch `digest` names in their names, as `launching`'s
    do: a launch's derivative traced after its forward's scope ended (a custom VJP's backward, which JAX
    traces on its pullback) is named for the launch that traced the forward, so a compile's report of its
    kernels refuses that launch (`spilled`). A digest this process traced no launch under, or None, names
    nothing."""
    launch = None if digest is None else _LAUNCHES.get(digest)
    if launch is None:
        yield
        return
    token = _TRACING.set(_Pending(lambda: launch.identity, launch.op, launch.config, lambda: launch.padded,
                                  digest))
    try:
        yield
    finally:
        _TRACING.reset(token)


@overload
def named(name: str) -> str: ...


@overload
def named(name: None) -> str | None: ...


def named(name: str | None) -> str | None:
    """A kernel's name carrying the launch tracing it (`launching`; `kernel` for an unnamed one), as is
    where none is."""
    digest = launch_digest()
    return name if digest is None else launch_name(name or "kernel", digest)


def launch_name(kernel: str, launch: str | None) -> str:
    """`kernel`'s name carrying `launch`'s digest (`LAUNCH_SUFFIX`), as is where None: `kernel_of`'s
    inverse."""
    return kernel if launch is None else f"{kernel}__l{launch}"


def launch_digest() -> str | None:
    """The digest of the launch tracing (`launching`), which its kernels' names carry (`named`), or None
    where none is."""
    pending = _TRACING.get()
    if pending is None:
        return None
    if pending.digest is None:
        identity = pending.identity()
        digest = _digest(f"{identity}|{pending.config!r}")
        with _GUARD:
            _LAUNCHES.setdefault(digest, Launch(identity, pending.op, pending.config, pending.padded()))
        pending.digest = digest
    return pending.digest


def traced(digest: str) -> Launch | None:
    """The launch this process traced under `digest` (a report's `launch`), or None."""
    return _LAUNCHES.get(digest)


def derivation_inputs(op, arguments: tokamax.BoundArguments, target) -> Hashable:
    """What a launch's derivation of its config for `arguments` on `target` reads of the launch and its
    compiles: the launch's identity (`_identity`: the op, its arguments' shapes and dtypes, its hooks), the
    whole target, whose facts price its schedules, and each config this context's compiles refused it
    (`refused`), so a refusal recorded after a derivation derives it again."""
    identity = _identity(op, arguments)
    with _GUARD:
        refusals = frozenset((config, kernels) for (launch, config), kernels in _registry().refused.items()
                             if launch == identity)
    return identity, target, refusals


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


def refuse_launch(op, arguments: tokamax.BoundArguments, config: Hashable, kernels) -> None:
    """`op`'s launch of `arguments` at `config` refused in this context as a compile that found `kernels`
    spilling refuses it (`settled`), so its derivation takes the next config: a compile gate's, which
    compiles a launch apart from `compiled`, stepping as a run's compile steps (row 4727)."""
    _record((_identity(op, arguments), config), None, frozenset(kernels))


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


def record_rewrite(made) -> None:
    """Record one region the algebra pass rewrote (`dew.algebra.program.AlgebraReport`) beside the kernels'
    misses, as each program traces."""
    with _GUARD:
        _registry().rewritten.append(made)


def record_pass(seconds: float) -> None:
    """Record one program's whole algebra pass, its seconds (`dew.algebra.program.rewritten_program`)."""
    with _GUARD:
        _registry().passes.append(seconds)


def passes() -> tuple[float, ...]:
    """Each program's whole algebra pass, in seconds, as this process's programs traced (`record_pass`): its
    regions' (`rewrites`) and its walks of the program around them."""
    with _GUARD:
        return tuple(_registry().passes)


def rewrites() -> tuple:
    """The regions the algebra pass rewrote as this process's programs traced (`record_rewrite`), for a
    run's telemetry: a `refusals` scope's own within it."""
    with _GUARD:
        return tuple(_registry().rewritten)


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
def _proposing(entries: list[tuple[tuple[str, Hashable], str]]) -> Iterator[_Registry]:
    """Refusals visible while deriving and compiling a replacement, without changing the kept program's
    registry or its file; the scope's registry, whose rewrites the replacement's trace recorded. A successful
    replacement commits both after this scope ends (`_commit`)."""
    registry = _registry()
    with _GUARD:
        proposed = dict(registry.refused)
    for identity, kernel in entries:
        proposed[identity] = proposed.get(identity, frozenset()) | {kernel}
    scope = _Registry(refused=proposed)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def _commit(proposal: _Registry) -> None:
    """A successful replacement's rewrites (`_proposing`'s scope) recorded as the kept program's."""
    with _GUARD:
        _registry().rewritten.extend(proposal.rewritten)


def _record(refusal: tuple[str, Hashable] | None, miss: "Miss | None",
            kernels: frozenset[str] = frozenset()) -> None:
    registry = _registry()
    with _GUARD:
        if refusal is not None:
            registry.refused[refusal] = registry.refused.get(refusal, frozenset()) | kernels
        if miss is not None:
            registry.missed.append(miss)


def compiled(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None = None, *,
             relower: Callable[[], jax.stages.Lowered], first: "Attempt | None" = None
             ) -> jax.stages.Compiled:
    """`lowered` compiled (`kept`), its communicators admitted against the room the process's pool left
    them before it runs (`dew.communicators.admit`): every program Dew compiles passes here. `first` is
    its first compile where another thread made it (`attempts`)."""
    executable = kept(lowered, options, relower=relower, first=first)
    communicators.admit(executable)
    return executable


def compiled_at_once(programs: Sequence[tuple[jax.stages.Lowered, jax.stages.CompilerOptions | None,
                                             Callable[[], jax.stages.Lowered]]]) -> list[jax.stages.Compiled]:
    """Each of `programs`, a lowering, its options and its `relower`, compiled as `compiled` compiles it,
    their first compiles at once, each in a thread of its own (`attempted`): XLA's compile releases the
    GIL and runs one program on about one and a half cores, so two compiled at once took 1.94 times the
    programs a second of two compiled in turn, four 2.2 times, on an 8-core host (a Qwen3.5-4B step for a
    compile-only H100, FS-2440-CONCURRENT2). Each is then kept (`settled`) and admitted in turn, in this
    thread, in the order given, as `compiled` keeps them one after another: a spill's refusal commits,
    and its program relowers, under this thread's contexts, so the programs kept are the ones compiled in
    turn keep: a program lowered before an earlier one's refusals committed is lowered again past them and
    compiled in turn, as it would have been lowered after them. At most `COMPILE_THREADS` compile at once."""
    firsts = attempts([(lowered, options) for lowered, options, _ in programs])
    held = refused_launches()
    kept_programs = []
    for (lowered, options, relower), first in zip(programs, firsts, strict=True):
        if refused_launches() != held:
            lowered, first = relower(), None
        kept_programs.append(compiled(lowered, options, relower=relower, first=first))
    return kept_programs


COMPILE_THREADS = 4
"""The most programs `compiled_at_once` compiles at once: four took 2.2 times the programs a second of one
on an 8-core host, each 1.8 times slower than alone, where two took 1.94 times (FS-2440-CONCURRENT2), so
more would share the host's cores among more compiles without finishing more."""


@dataclasses.dataclass(frozen=True)
class Attempt:
    """A program's first compile (`attempted`): its executable, or the error XLA raised of it, which its
    settlement raises in its caller's turn (`settled`) where it is no refusal of a Triton kernel asking
    shared bytes past a block (`shared_asked`)."""

    executable: jax.stages.Compiled | None
    error: Exception | None = None
    """A `jax.errors.JaxRuntimeError`, which JAX names as a value, no type."""


def attempted(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None) -> Attempt:
    """`lowered` compiled once under `options`: XLA's compile alone, which reads and writes nothing of this
    process's but the executable it makes and the persistent compilation cache, so a thread of its own
    runs it (`compiled_at_once`); its kernels' binaries are read where it is kept (`settled`), in its
    caller's thread, as reading them walks the executable's MLIR (`binaries`), which two threads at once
    failed in (a Qwen3.5-4B step and its sampled step for a compile-only H100: `MLIRError: Failure while
    executing pass pipeline`)."""
    try:
        return Attempt(lowered.compile(options))
    except jax.errors.JaxRuntimeError as error:
        return Attempt(None, error)


def attempts(programs: Sequence[tuple[jax.stages.Lowered, jax.stages.CompilerOptions | None]]
             ) -> list[Attempt]:
    """Each of `programs` compiled once (`attempted`), in order, all at once, at most `COMPILE_THREADS`
    at a time; one alone in this thread. JAX's cache key reads a module's context without writing it from
    0.12.0.1.post54 (pyproject's pin), so programs one thread lowered in one context compile together."""
    if len(programs) < 2:
        return [attempted(*program) for program in programs]
    with ThreadPoolExecutor(min(len(programs), COMPILE_THREADS), thread_name_prefix="dew-compile") as pool:
        return [each.result() for each in [pool.submit(attempted, *program) for program in programs]]


def reads_binaries(lowered: jax.stages.Lowered) -> bool:
    """Whether a compile of `lowered` reads its kernels' binaries: where it runs a Mosaic GPU or a
    Pallas-Triton launch. A program of Pallas-Triton launches alone spills as a Mosaic one does: both
    kinds' binaries are read (`binaries`, a Triton launch's by its custom kernel's tag), so a launch's
    feedback is the same whatever kernels share its program (row 3960: a Triton-only program's
    15,752-byte frame went unread)."""
    from dew.nn.kernels.lowered import kernel_calls

    return any(call.kind in ("mosaic_gpu", "triton") for call in kernel_calls(lowered))


def kept(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None = None, *,
         relower: Callable[[], jax.stages.Lowered], first: Attempt | None = None) -> jax.stages.Compiled:
    """`lowered` compiled and kept (`settled`), its first compile `first` or one in this thread."""
    return settled(lowered, options, relower, attempted(lowered, options) if first is None else first)


def settled(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
            relower: Callable[[], jax.stages.Lowered], first: Attempt) -> jax.stages.Compiled:
    """The program kept of `lowered`'s first compile (`first`), read through its kernels' binaries
    (`binaries`): where a kernel of a template's
    launch spills, that launch's config is refused (`refused`), the miss recorded (`misses`, logged), and the
    program `relower`ed once and compiled; `relower` traces afresh, so each refused launch's derivation
    takes the next config by its own rule. A kernel spilling again is recorded, neither refused nor stepped
    again, so every process derives the program this compile keeps; a relowering whose derivations find no
    config left (`Exhausted`) keeps the first program, its misses recorded as not stepped. Refusals commit
    only when a replacement compiles, so a later process derives the program kept here. A program loaded
    from the persistent compilation cache carries its kernels' binaries as one compiled here does. A
    program whose block XLA refuses steps its asking Triton launches down (`_unblocked`)."""
    if first.error is not None:
        asked = shared_asked(first.error)
        if asked is None:
            raise first.error
        return _unblocked(lowered, options, relower, asked, first.error)
    executable = first.executable
    assert executable is not None
    found = _misses_of(_spilling(lowered, executable))
    refusals = [(refusal, report.kernel) for refusal, report, _ in found if refusal is not None]
    if not refusals:
        _recorded(found, stepped=False)
        return executable
    try:
        with _proposing(refusals) as proposal:
            relowered = relower()
            replacement, spilling = _compiled_reported(relowered, options)
    except Exhausted as failed:
        _log.warning("relowering after a spill found no config: %s; the first program runs", failed)
        _recorded(found, stepped=False)
        return executable
    _commit(proposal)
    for refusal, kernel in refusals:
        _record(refusal, None, frozenset({kernel}))
    _recorded(found, stepped=True)
    # The kept program's own spills are misses alone: refused, they would have every process seeded with
    # them derive a third program, not the one this compile keeps and runs.
    _recorded(_misses_of(spilling), stepped=False)
    return replacement


def _unblocked(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
               relower: Callable[[], jax.stages.Lowered], asked: int,
               error: Exception) -> jax.stages.Compiled:
    """The program XLA refused for a Triton kernel asking `asked` shared bytes past a block, compiled once
    more with every one of its Triton launches' configs refused at once (each its next derived config, as a
    spill refuses one, `kept`): XLA's refusal names no kernel, and a search for the asking launch would
    compile the whole program a launch at a time (row 3584). Each launch's miss is recorded and logged at
    the bytes asked; where the relowered program is refused again, or a derivation finds no config left
    (`Exhausted`), the refusal stands, each a miss not stepped."""
    from dew.nn.kernels.lowered import kernel_calls

    tried: dict[tuple[str, Hashable], tuple[str, Launch]] = {}
    # Every Triton kernel XLA compiles, a branch no run reaches too (a guarded forward's rewalk).
    for call in kernel_calls(lowered, unrun=True):
        kernel, digest = kernel_of(call.name)
        launch = None if call.kind != "triton" or digest is None else traced(digest)
        if launch is not None:
            tried.setdefault((launch.identity, launch.config), (kernel, launch))
    try:
        with _proposing([(refusal, kernel) for refusal, (kernel, _) in tried.items()]) as proposal:
            replacement, spilling = _compiled_reported(relower(), options)
    except (Exhausted, jax.errors.JaxRuntimeError) as again:
        if isinstance(again, jax.errors.JaxRuntimeError) and shared_asked(again) is None:
            raise
        for kernel, launch in tried.values():
            _missed_block(kernel, launch, asked, stepped=False)
        raise error from again
    _commit(proposal)
    for refusal, (kernel, launch) in tried.items():
        _record(refusal, None, frozenset({kernel}))
        _missed_block(kernel, launch, asked, stepped=True)
    _recorded(_misses_of(spilling), stepped=False)
    return replacement


def _missed_block(kernel: str, launch: Launch, asked: int, *, stepped: bool) -> None:
    """A Triton launch's shared-memory miss recorded and logged (`_unblocked`), under its first kernel's
    name: XLA's refusal names none."""
    _record(None, Miss(kernel, launch.op, launch.config, None, None, stepped, shared=asked))
    _log.warning("a program asks %d shared bytes past its block; Triton kernel %s (%s at %s): %s", asked,
                 kernel, launch.op, launch.config, "its launch steps down" if stepped else "no step-down")


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
    executable = lowered.compile(options)
    return executable, _spilling(lowered, executable)


def _spilling(lowered: jax.stages.Lowered, executable: jax.stages.Compiled) -> list["Binary"]:
    """The kernels of `executable`, `lowered` compiled, that spill, where it runs a template's
    (`reads_binaries`)."""
    if not reads_binaries(lowered):
        return []
    return [found for found in binaries(executable) if found.frame]


@dataclasses.dataclass(frozen=True)
class Binary:
    """A Mosaic GPU or Pallas-Triton kernel's binary as its executable carries it (`binaries`): its name less
    the launch it carries (`LAUNCH_SUFFIX`) and that launch's digest (`named`), its registers a thread and its
    stack frame's bytes a thread, nonzero where its registers spill (`REGISTER_COUNT`, `FRAME_SIZE`), the
    dynamic shared memory its launch asks for (a Mosaic kernel proto's `smem_bytes`, a custom kernel's
    `shared_memory_bytes`), None where its proto states none, and, read `decoded`, what its machine code
    breaks of the SASS rules (`sass_findings`), None where no disassembler read it."""
    kernel: str
    launch: str | None
    registers: int
    frame: int
    shared: int | None = None
    sass: tuple[str, ...] | None = ()


def binaries(executable: jax.stages.Compiled, *, decoded: bool = False) -> list[Binary]:
    """Every Mosaic GPU and Pallas-Triton kernel binary `executable` carries, read from its own serialized
    form: XLA keeps a Mosaic kernel's cubin in its custom call's state (`MosaicGpuKernelProto`), whether this
    compile ran ptxas for it, another thread's program compiled it first (Mosaic compiles a module once a
    process), or the executable was loaded from the persistent cache, and a Pallas-Triton template
    launch's in its custom kernel's (`custom_kernel`), its kernel naming its launch: the thunk's own
    `binary` is its module's, which in a program holding a constant is that constant's module alone, no
    kernel's code in it. Nothing is read from or written to the process's environment or disk: a compile
    that set Mosaic's dump settings around itself handed its directory to every other thread's compile and
    removed it under them ('Directory not empty', row 2790), and the settings turn on every dump Mosaic has
    (row 2816). `decoded`, each binary's SASS findings too (`sass_findings`, the compile gate's), each
    cubin disassembled from a file of its own."""
    from jax._src.thunky import thunky

    found: list[Binary] = []
    context = thunky.make_ir_context()
    with context:
        module = thunky.jax_executable_to_mlir(executable, context)

        def visit(operation: ir.Operation) -> ir.WalkResult:
            if operation.name == "thunky.call_thunk_proto":
                attributes = operation.attributes
                proto = ir.StringAttr(attributes["thunk_proto"]).value_bytes
                found.extend(thunk_kernels(proto, decoded=decoded))
                name = ir.StringAttr(attributes["kernel_name"]).value
                # XLA's own kernels (its fusions, its Triton GEMMs) carry their cubins too: a template
                # launch's names its launch (`named`).
                kernel = custom_kernel(proto) if LAUNCH_SUFFIX.search(name) else None
                if kernel is not None:
                    elf, shared = kernel
                    found.append(dataclasses.replace(_binary(name, elf, decoded), shared=shared))
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)
    return found


def thunk_kernels(proto: bytes, *, decoded: bool = False) -> list[Binary]:
    """The kernel binaries of a thunk's serialized proto (`_kernels`), one for each Mosaic GPU call it names:
    a call whose binary is missing would read as a kernel that does not spill, so it raises."""
    kernels, calls = _kernels(proto, decoded)
    if len(kernels) != calls:
        raise ValueError(f"a thunk names {calls} Mosaic GPU calls but carries {len(kernels)} kernel binaries")
    return kernels


def custom_kernel(proto: bytes) -> tuple[bytes, int] | None:
    """The cubin of the custom kernel a thunk's serialized proto launches and the shared bytes its launch
    asks for (`CUSTOM_KERNEL_THUNK`), None where the thunk launches no custom kernel's cubin."""
    kernels = [kernel for thunk in _fields(proto).get(CUSTOM_KERNEL_THUNK, [])
               for kernel in _fields(thunk).get(3, [])]
    if len(kernels) != 1:
        return None
    kernel = _fields(kernels[0])
    cubins = [elf for spec in kernel.get(2, []) for cubin in _fields(spec).get(2, [])
              for elf in _fields(cubin).get(1, [])]
    if len(cubins) != 1 or cubins[0][:4] != ELF:
        return None
    # proto3 omits a zero.
    return cubins[0], kernel.get(6, [0])[0]


def _fields(message: bytes) -> dict[int, list]:
    """A serialized proto's fields by number, each its values (a varint's int, a length-delimited field's
    bytes), read without its schema; none where `message` parses as no proto."""
    parsed = empty_pb2.Empty()
    try:
        parsed.ParseFromString(message)
    except DecodeError:
        return {}
    fields: dict[int, list] = {}
    for field in UnknownFieldSet(parsed):
        fields.setdefault(field.field_number, []).append(field.data)
    return fields


def _kernels(message: bytes, decoded: bool = False) -> tuple[list[Binary], int]:
    """The kernels of each `MosaicGpuKernelProto` within a thunk's serialized proto (a message of jaxlib's
    version, field 1, whose binary, 8, is an ELF, whose kernel name, 9, its function's, and whose
    `SHARED_BYTES` its launch's dynamic shared memory), and how many Mosaic GPU custom calls it names
    (`MOSAIC_TARGET`, a custom call thunk's target)."""
    fields = _fields(message)
    binary, name, shared = fields.get(8, []), fields.get(9, []), fields.get(SHARED_BYTES, [])
    if (fields.get(1) == [KERNEL_PROTO_VERSION] and len(binary) == len(name) == 1
            and isinstance(binary[0], bytes) and binary[0][:4] == ELF and isinstance(name[0], bytes)):
        stated = shared[0] if len(shared) == 1 and isinstance(shared[0], int) else None
        return [dataclasses.replace(_binary(name[0].decode(), binary[0], decoded), shared=stated)], 0
    found, calls = [], 0
    for values in fields.values():
        for value in values:
            if value == MOSAIC_TARGET:
                calls += 1
            elif isinstance(value, bytes):
                kernels, named = _kernels(value, decoded)
                found += kernels
                calls += named
    return found, calls


def _binary(function: str, elf: bytes, decoded: bool = False) -> Binary:
    """The `Binary` of the kernel `function` from its cubin `elf`: its `.nv.info` attributes' registers and
    frame (`REGISTER_COUNT`, `FRAME_SIZE`; a Mosaic module compiles one kernel function), and `decoded` its
    SASS findings (`sass_findings`)."""
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
    return Binary(*kernel_of(function), stated[REGISTER_COUNT], stated.get(FRAME_SIZE, 0),
                  sass=sass_findings(elf) if decoded else ())


def disassembler() -> Path | None:
    """NVIDIA's nvdisasm, which the SASS rules read a kernel's machine code through (`sass_findings`): the
    nvdisasm wheel's (`nvidia/cu13/bin`), which the `ci` extra installs in every lock the suite runs on, or
    the PATH's; None where neither is installed, a user's install among them."""
    spec = importlib.util.find_spec("nvidia")
    for root in (spec.submodule_search_locations or ()) if spec else ():
        if (found := Path(root) / "cu13" / "bin" / "nvdisasm").is_file():
            return found
    found = shutil.which("nvdisasm")
    return None if found is None else Path(found)


def sass_findings(elf: bytes) -> tuple[str, ...] | None:
    """What the machine code of the cubin `elf` breaks of the SASS rules (`sass_findings_of`, nvdisasm's
    listing of it); None where no disassembler is installed (`disassembler`), the rules unread."""
    tool = disassembler()
    if tool is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin:
        cubin.write(elf)
        cubin.flush()
        listing = subprocess.run([str(tool), "-c", cubin.name], capture_output=True, text=True, check=True)
    return sass_findings_of(listing.stdout)


_INSTRUCTION = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
_LABEL = re.compile(r"^\s*(\.L_\w+):")
_WATERFALL = re.compile(r"\bBRA\.U\.ANY\s+`?\(?(\.L_\w+)\)?`?")


def sass_findings_of(listing: str) -> tuple[str, ...]:
    """What nvdisasm's `listing` of a kernel breaks of the kernel challenge's SASS rules (row 4071, its
    harness/ptxas checks; notes/ptxas-re.md §2): an elect waterfall (a backward BRA.U.ANY) holding an R2UR, a
    copy issued lane by lane because ptxas could not prove its operand uniform."""
    instructions, labels, pending = [], {}, []
    for line in listing.splitlines():
        if label := _LABEL.match(line):
            pending.append(label.group(1))
        elif found := _INSTRUCTION.search(line):
            labels.update(dict.fromkeys(pending, len(instructions)))
            pending = []
            instructions.append(" ".join(found.group(2).split()))
    serialized = 0
    for at, text in enumerate(instructions):
        head = labels.get(branch.group(1)) if (branch := _WATERFALL.search(text)) else None
        if head is not None and head <= at:
            serialized += any("R2UR" in held for held in instructions[head:at + 1])
    if not serialized:
        return ()
    return (f"{serialized} elect waterfall holding an R2UR, a copy issued lane by lane",)
