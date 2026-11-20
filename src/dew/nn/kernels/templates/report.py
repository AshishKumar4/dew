"""What Mosaic GPU's compiles say of each kernel, what the templates' resource models are checked against.
A run's compile (`compiled`, owner 10-29, bible §6) reads each kernel's registers a thread and its stack
frame from the binary its executable carries (`binaries`): a launch whose kernel spills is refused
(`refused`) and its program lowered again, where the launch's derivation takes the next config by its own
rule, until none spills or the launch has none left (`settled`), each such refusal a model miss (`misses`).
The CPU suite's compile gate reads the binaries too, beside the resource estimate Pallas prints as it lowers
a kernel and the PTX Mosaic prints as it compiles one, whose registers and spills ptxas states at the GPU's
own compile (tests/kernel_compile.py). Each kernel's name carries the launch that traced it (`named`), so a
binary names its launch and its config."""

import collections
import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import re
import shutil
import struct
import subprocess
import tempfile
import threading
from collections.abc import Callable, Collection, Hashable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, overload

import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from google.protobuf import empty_pb2
from google.protobuf.message import DecodeError
from google.protobuf.unknown_fields import UnknownFieldSet
from jax._src.thunky import thunky
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import stablehlo

from dew import _cache, communicators
from dew._devices import deterministic_ops_requested, deviceless, xla_flag
from dew.nn.kernels import generation

if TYPE_CHECKING:
    from dew.nn.kernels.lowered import KernelCall
    from dew.nn.kernels.templates.resources import RegisterModel
    from dew.nn.kernels.tuning import KernelOp

SHARED_REFUSED = re.compile(r"Shared memory size limit exceeded: requested (\d+)")
"""XLA's refusal of a program one of whose Triton kernels asks more shared memory than a block holds,
which names no kernel (`shared_asked`)."""


def shared_asked(error: BaseException) -> int | None:
    """The shared bytes a Triton kernel of a program XLA refused asked past a block, or None where `error`
    is not that refusal: the one reading of XLA's text (`kept`, `tuning`'s fitting, the compile-only
    harness)."""
    asked = SHARED_REFUSED.search(str(error))
    return None if asked is None else int(asked.group(1))


LAUNCH_SUFFIX = re.compile(r"__l([0-9a-f]{10})(?:__(\d+))?$")
"""The suffix a kernel's name carries within the launch that traced it (`named`): the digest of its identity
there, its launch's digest where the launch states no kernel's part (`tuning.KernelOp.emitted`)."""
KERNEL_FUNCTION = re.compile(r"(?P<kernel>.*)_mosaic_gpu_kernel")
"""A Mosaic GPU kernel's function in its binary: its kernel's name, the host function's `_mosaic_gpu`
(jax/experimental/mosaic/gpu/core.py) and GPU outlining's `_kernel`."""


def kernel_of(name: str) -> tuple[str, str | None, str | None]:
    """`(kernel, identity, index)` of a kernel's `name` as it traced (`named`), its function in its binary
    or its PTX entry (`KERNEL_FUNCTION`) or its call's location: its name less its suffix, the suffix's
    digest (`LAUNCH_SUFFIX`), and any XLA uniquification index (`__<digits>`), None where absent."""
    function = KERNEL_FUNCTION.fullmatch(name)
    kernel = function.group("kernel") if function else name
    if not (launch := LAUNCH_SUFFIX.search(kernel)):
        return kernel, None, None
    return kernel[:launch.start()], launch.group(1), launch.group(2)
KERNEL_PROTO_VERSION = 4
"""The `MosaicGpuKernelProto` version jaxlib serializes (mosaic/gpu/custom_call.cc), which it alone reads."""
MOSAIC_TARGET = b"mosaic_gpu_v2"
"""The custom call target of a Mosaic GPU kernel, which its thunk names (`_kernels`)."""
SHARED_BYTES = 10
"""The `MosaicGpuKernelProto` field of the dynamic shared memory a kernel's launch asks for (`smem_bytes`)."""
CALLER_METADATA = 12
"""The `MosaicGpuKernelProto` field of what its call's caller stated of the launch (`caller_metadata`, Dew's
JAX build's kernel-metadata): Dew's launch's digest (`launch_metadata`)."""
SERIALIZED = frozenset({*(f"C751{digit}" for digit in range(6)), "C7519"})
"""The codes of ptxas's notes (`AssemblerNote`) of a kernel's warpgroup MMAs serialized, each product waited
for before the next issues (C7510 to C7515; C7519, a dependence in a divergent path): a run refuses a kernel
so noted as a spilling one (`Binary.refused`), and the compile gate fails it (tests/kernel_compile.py): the
ranking's sm90 attention backward held its registers and serialized its products (C7512)."""
IGNORED_RAISE = "C7507"
"""ptxas's note of a kernel's setmaxnreg ignored, its warp roles' paths rejoining after the split and its
threads left at the entry's registers (row 4071): the compile gate fails a derived kernel for it, a template's
structure that leaves its roles their registers; a run does not refuse it, as a role short of registers there
spills, which `Binary.refused` reads."""
ASSEMBLER_NOTES = 13
"""The `MosaicGpuKernelProto` field of what ptxas noted of a loss of performance as it assembled the kernel
(`assembler_notes`, Dew's JAX build's mosaic-gpu-kernel-facts), each note a message of its code (1) and its
message (2) (`AssemblerNote`)."""
CUSTOM_KERNEL_THUNK = 36
"""The `ThunkProto` field of a custom kernel's thunk, a Pallas-Triton launch's (`CustomKernelThunkProto`): its
kernel (3, `CustomKernelProto`), whose loader spec (2) holds its cubin (2, `CudaCubinProto`; 1) and which
states the dynamic shared memory its launch asks for (6, `shared_memory_bytes`)."""
ELF = b"\x7fELF"
"""A cubin's first bytes."""
REGISTER_COUNT, FRAME_SIZE, MAX_THREADS = 0x2F, 0x11, 0x05
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
    kernels: dict[str, "RegisterModel"] = dataclasses.field(default_factory=dict, compare=False)
    """Each kernel the launch traced that stated its register model as it was named (`named`'s `model`), by
    its name less the launch's suffix (`kernel_of`'s): the launch's stated register allocation, whose
    producer the compile gate checks its emission lowers to."""
    emitted: Callable[[Hashable, str], Hashable] | None = dataclasses.field(default=None, compare=False)
    """What each kernel's emission reads of a config it is emitted at (`config`, or a level of it the launch
    runs a call at, `forming`'s `at`), by its name (`tuning.KernelOp.emitted`)."""
    reverse: bool = dataclasses.field(default=False, compare=False)
    """Whether it traced as reverse mode runs it (`pulled`), the form its certification compiles
    (`certified`): a kernel's own entry (`named`) its trace's, the launch's its first trace's."""


@dataclasses.dataclass
class _Pending:
    """The launch tracing now. Its identity is evaluated only if a Mosaic kernel asks for its name;
    bound arguments live in this tracing scope, never in the process's launch registry."""
    identity: Callable[[], str]
    op: str
    config: Hashable
    padded: Callable[[], float | None]
    emitted: Callable[[Hashable, str], Hashable]
    digest: str | None = None
    reverse: bool = False


_TRACING: contextvars.ContextVar[_Pending | None] = contextvars.ContextVar("tracing_launch", default=None)
"""The pending launch whose Mosaic kernels acquire their digest through `named`."""
_NAMING: contextvars.ContextVar[list | None] = contextvars.ContextVar("kernel_naming", default=None)
"""The kernels named within a `naming` block, in order, as `named` was asked and answered."""


@contextlib.contextmanager
def naming() -> Iterator[list[tuple[str | None, "RegisterModel | None", str | None]]]:
    """Each kernel named within the block (`named`): the name and model it was asked with and the name it
    took, which a held trace names again as it is bound for another launch (`renamed`)."""
    names: list = []
    token = _NAMING.set(names)
    try:
        yield names
    finally:
        _NAMING.reset(token)


@contextlib.contextmanager
def replayed() -> Iterator[None]:
    """Within the block, no launch traces: the state a held trace is kept under (`tuning.lowered_once`),
    which reads the launch tracing it only as it names its kernels and carries the launch on their calls:
    its key holds the launch (`launch_metadata`), so a held trace is the launch's own, its names replayed as
    it is bound (`renamed`)."""
    tracing, form, naming = _TRACING.set(None), _FORM.set(((), None)), _NAMING.set(None)
    try:
        yield
    finally:
        _NAMING.reset(naming)
        _FORM.reset(form)
        _TRACING.reset(tracing)


def renamed(names: Sequence[tuple[str | None, "RegisterModel | None", str | None]]) -> None:
    """The kernels a held trace named (`naming`) named again for the launch tracing now, each registered on
    it as `named` registers a kernel traced afresh: each takes the name it took, as its trace's key holds what
    its launch was given (`tuning.lowered_once`); one that takes another is refused, its name no digest of
    what its body read."""
    for name, model, took in names:
        if (now := named(name, model=model)) != took:
            raise ValueError(f"a held kernel named {took} is named {now} for the launch tracing now: its "
                             "name reads what its launch was not given (`KernelOp.emitted`, `lowered_once`)")


_FORM: contextvars.ContextVar[tuple[Hashable, Hashable]] = contextvars.ContextVar("kernel_form",
                                                                               default=((), None))
"""The form of the kernel tracing and the config it is emitted at (`forming`)."""


@contextlib.contextmanager
def forming(form: Hashable, at: Hashable = None) -> Iterator[None]:
    """Within the block, the kernels traced are of `form`, what their emission reads beside their launch's
    config (`tuning.lowered_once`: a launch's static arguments the config does not carry, a forward's keeping
    its inverses for a backward or not), emitted `at` a config, the launch's own where None or a level of it
    a call runs at (a training config's forward tiles, which a forward alone takes): their names hold what
    their emission reads of it (`named`), so a launch tracing one kernel at two levels of its config (a
    training call's forward and a forward alone, an SSD's states at 32 and 16 value columns) names two."""
    token = _FORM.set((form, at))
    try:
        yield
    finally:
        _FORM.reset(token)


_LAUNCHES: dict[str, Launch] = {}
"""Each launch this process traced, by its digest, for the process's life: a program's compile reads its
launches whenever it comes (`compiled`), a pullback's trace whenever JAX makes it (`relaunched`), and JAX
hands a cached trace back without tracing it again, so no trace states when its launches are read last."""
RECORDS_HELD = 4096
"""The misses, rewrites and pass timings a registry keeps (`_Registry`), the latest; a reader of one
search's own takes them in a `recording` scope."""
Refused = tuple[str, tuple[str, ...] | None]
"""A kernel a compile refused, by its name, and the roles its spill stood in where the compile read them
(`Binary.spilled_in`: `()` read and spilling none, its warpgroup MMAs serialized alone; None unread): one
record of both, a kernel once a refusal (`_record` joins its roles), so whatever keys, copies or hands on a
refusal (`derivation_inputs`, `_proposing`, `refusal_entries`) carries what a launch's ladder steps by."""


@dataclasses.dataclass
class _Registry:
    """Each (launch identity, config) a compile refused (`compiled`: its kernel spilled), by its kernels that
    spilled with their roles (`Refused`), and each miss."""
    refused: dict[tuple[str, Hashable], frozenset[Refused]] = dataclasses.field(default_factory=dict)
    missed: collections.deque["Miss"] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=RECORDS_HELD))
    rewritten: collections.deque = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=RECORDS_HELD))
    """Each region the algebra pass rewrote, its report (`dew.algebra.program.AlgebraReport`)."""
    passes: collections.deque[float] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=RECORDS_HELD))
    """Each program's whole pass, in seconds: its walks of its bodies and windows beside its regions'."""
    by_launch: dict[str, frozenset] = dataclasses.field(default_factory=dict)
    """`refusals_of`'s answers by launch identity, each dropped as a refusal of its launch is recorded: read
    for every config a derivation weighs, each answer gathered from every launch's refusals took 426 of a
    refusal walk's 685 profiled seconds, 250,267 reads over its 355 derivations (row 6551)."""


@dataclasses.dataclass
class Recorded:
    """The misses, rewrites and pass timings recorded within a `recording` scope, every one: one search's
    own, which its plan reports."""
    missed: list["Miss"] = dataclasses.field(default_factory=list)
    rewritten: list = dataclasses.field(default_factory=list)
    passes: list[float] = dataclasses.field(default_factory=list)


_RECORDING: contextvars.ContextVar[tuple[Recorded, ...]] = _cache.untraced(contextvars.ContextVar("recording",
                                                                                                  default=()))
"""The `recording` scopes open in this context, innermost last."""


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
    that ends so refuses the program (`compiled`), which no compile keeps spilling."""


@dataclasses.dataclass(frozen=True)
class Miss:
    """A kernel a run's compile refused (`Binary.refused`: spilling, or its warpgroup MMAs serialized), or a
    program whose block XLA refused (`compiled`): its kernel's name, its launch's op and config (None where
    no launch of this process traced it), its registers and its stack frame's bytes a thread
    (`Binary.frame`) or the shared bytes it asked (`shared`), and whether its launch's config was refused
    and its program lowered again (`stepped`)."""
    kernel: str
    op: str | None
    config: Hashable | None
    registers: int | None
    frame: int | None
    stepped: bool
    shared: int | None = None
    serialized: tuple[str, ...] = ()
    """The codes of ptxas's notes of its warpgroup MMAs serialized (`SERIALIZED`), a reason it was refused
    beside a spill (`frame`)."""
    modelled: int | None = None
    """The registers a consumer thread holds as its launch's model stated them (`RegisterModel.registers`,
    `named`), beside the assembler's (`registers`); None where its launch stated none."""

    @property
    def reason(self) -> str:
        """Why its kernel was refused, as a log and a plan state it."""
        if self.shared is not None:
            return f"its block asks {self.shared} shared bytes"
        reasons = ([f"spills to a {self.frame}-byte stack frame"] if self.frame else []) + (
            [f"serializes its warpgroup MMAs ({', '.join(self.serialized)})"] if self.serialized else [])
        return " and ".join(reasons) or "is refused"


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
    the same in every process, and their binaries name the launch (`Binary.launched`, `traced`)."""
    token = _TRACING.set(_Pending(lambda: _identity(op, arguments), type(op).__name__, config,
                                  lambda: op.padded(arguments, config), op.emitted,
                                  reverse=_PULLED.get() and op.CARRIES_DERIVATIVE))
    try:
        yield
    finally:
        _TRACING.reset(token)


@contextlib.contextmanager
def relaunched(digest: str | None) -> Iterator[None]:
    """Within the block, the kernels traced carry the launch `digest` names in their names, as `launching`'s
    do: a launch's derivative traced after its forward's scope ended (a custom VJP's backward, which JAX
    traces on its pullback) is named for the launch that traced the forward, so a compile's report of its
    kernels refuses that launch (`refused_at`). A digest this process traced no launch under, or None, names
    nothing."""
    launch = None if digest is None else _LAUNCHES.get(digest)
    if launch is None:
        yield
        return
    assert launch.emitted is not None
    token = _TRACING.set(_Pending(lambda: launch.identity, launch.op, launch.config, lambda: launch.padded,
                                  launch.emitted, digest, launch.reverse))
    try:
        yield
    finally:
        _TRACING.reset(token)


@overload
def named(name: str, *, model: "RegisterModel | None" = None) -> str: ...


@overload
def named(name: None, *, model: "RegisterModel | None" = None) -> str | None: ...


def named(name: str | None, *, model: "RegisterModel | None" = None) -> str | None:
    """A kernel's name carrying its identity within the launch tracing it (`launching`; `kernel` for an
    unnamed one), as is where none is: the digest of the launch's identity and of what the kernel's emission
    reads of the launch's config (`tuning.KernelOp.emitted`, the whole config unless the op states a kernel's
    part) and of its form (`forming`), so a kernel a change to its launch leaves alone keeps its name, its
    module, and the lowering cache's and Mosaic's compile of it. The launch itself travels beside the name, as
    a Mosaic GPU call's metadata (`launch_metadata`); a kernel whose call carries none (Triton's) is read back
    to its launch by its name alone, which is its launch's digest wherever its emission reads the whole config
    in no form. Its register `model`, where given, is recorded on that launch (`Launch.kernels`)."""
    took = _named(name, model)
    if (names := _NAMING.get()) is not None:
        names.append((name, model, took))
    return took


def _named(name: str | None, model: "RegisterModel | None") -> str | None:
    digest = launch_digest()
    if digest is None:
        return name
    kernel = name or "kernel"
    pending = _TRACING.get()
    assert pending is not None
    with _GUARD:
        launch = _LAUNCHES[digest]
        if model is not None:
            launch.kernels[kernel] = model
        form, at = _FORM.get()
        part = pending.emitted(pending.config if at is None else at, kernel)
        own = _digest(f"{launch.identity}|{part!r}|{form!r}")
        # The kernel's identity names its launch too (`traced`), for a reader holding its name alone, in the
        # mode it traced in.
        _LAUNCHES.setdefault(own, dataclasses.replace(launch, reverse=pending.reverse))
    return launch_name(kernel, own)


def launch_metadata() -> bytes:
    """The launch tracing (`launching`) as its kernels' calls carry it beside their names (`named`): its
    digest, a Mosaic GPU call's caller metadata (`plgpu.CompilerParams.metadata`, kept in its compiled call's
    serialization, `Binary.caller`; not its module's, so no lowering or compile of the kernel reads it);
    none where no launch traces."""
    digest = launch_digest()
    return b"" if digest is None else digest.encode()


def launch_name(kernel: str, launch: str | None, index: str | None = None) -> str:
    """`kernel`'s name carrying the digest `launch` (`LAUNCH_SUFFIX`) and optional XLA uniquification
    `index`, as is where None: the lossless key a kernel's evidence goes by beside its launch."""
    named = kernel if launch is None else f"{kernel}__l{launch}"
    return named if index is None else f"{named}__{index}"


class KernelsOfOneName(ValueError):
    """Two kernels of one name and launch (`kernel_bodies`), whose evidence would be one's."""


def kernel_bodies(calls: Iterable["KernelCall"]) -> dict[str, "KernelCall"]:
    """Each kernel call by its kernel's name and launch (`launch_name`), the key every record of a kernel's
    evidence reads (its module, binary, PTX and register model): two calls of one name and launch whose
    kernels differ (`_kernel`) raise, as each record would hold one kernel's evidence for both.
    Dew names a kernel where it launches it (`named`: the name it is given, its launch's digest beside), so
    two forms of one launch given one name share it whatever their kernels: a training call's keys-alone
    re-forming beside its weighted launch, both `recurrence_weighted`, whose certification read the one and
    missed the other's 8-byte spill, and a Triton attention forward's rewalk beside its plain launch, both
    `attention_forward`. Each form is named apart where Dew names it; this refuses one that is not."""
    found: dict[str, KernelCall] = {}
    for call in calls:
        name = launch_name(call.name, call.launch)
        if _kernel(found.setdefault(name, call)) != _kernel(call):
            raise KernelsOfOneName(f"{name} names two kernels: its evidence would be one's")
    return found


def _kernel(call: "KernelCall") -> Hashable:
    """A call's kernel as its compiler tells kernels apart: a Mosaic GPU kernel's hash (`KernelCall.config`,
    which Mosaic takes of its module without debug information), any other's IR and config."""
    return call.config if call.kind == "mosaic_gpu" else (call.ir, call.config)


def kernel_modules(calls: Iterable["KernelCall"]) -> dict[str, str]:
    """Each Mosaic GPU call's module (its kernel hash, `KernelCall.config`) by its kernel's name and launch,
    one kernel a name and launch (`kernel_bodies`)."""
    return {name: call.config
            for name, call in kernel_bodies(call for call in calls if call.kind == "mosaic_gpu").items()}


def _launched(identity: str, config: Hashable, reverse: bool) -> str:
    """The digest of the launch of `identity` at `config` (`launch_digest`): a kernel's identity in its
    default form (`named`), what its emission reads the whole config, its form none; a gradient's product's
    digest apart from its call's own, as its verdict is (`_held`)."""
    return _digest(f"{identity}|{config!r}|{()!r}{'|reverse' if reverse else ''}")


def launch_digest() -> str | None:
    """The digest of the launch tracing (`launching`), its identity, config and mode (`Launch.reverse`: a
    gradient's product or a call's own), which its calls carry beside their kernels' names
    (`launch_metadata`), or None where none is. The mode is the launch's alone: a kernel's name reads its
    launch's identity and config (`named`), the body both modes share, so a kernel whose call carries no
    caller (Triton's) names its launch's forward."""
    pending = _TRACING.get()
    if pending is None:
        return None
    if pending.digest is None:
        identity = pending.identity()
        digest = _launched(identity, pending.config, pending.reverse)
        with _GUARD:
            _LAUNCHES.setdefault(digest, Launch(identity, pending.op, pending.config, pending.padded(),
                                                emitted=pending.emitted, reverse=pending.reverse))
        pending.digest = digest
    return pending.digest


def called(kernel: "KernelCall | Binary") -> Launch | None:
    """The launch that traced `kernel`, a call or a binary (`launched`, the digest of the launch that traced
    it); None where this process traced none."""
    return None if kernel.launched is None else traced(kernel.launched)


def traced(digest: str) -> Launch | None:
    """The launch this process traced under `digest`, its own or a kernel's identity within it (`named`, a
    report's `launch`), or None."""
    return _LAUNCHES.get(digest)


def derivation_inputs(op, arguments: tokamax.BoundArguments, target
                      ) -> tuple[str, Hashable, frozenset[tuple[Hashable, frozenset[Refused]]], bool]:
    """What a launch's derivation of its config for `arguments` on `target` reads of the launch and its
    compiles: the launch's identity (`_identity`: the op, its arguments' shapes and dtypes, its hooks), the
    whole target, whose facts price its schedules, each config this context's compiles refused it with the
    roles its kernels spilled in (`refused`, `Refused`), so a refusal recorded after a derivation derives it
    again, and whether the run asked XLA for
    deterministic ops, which a launch's legality reads (a fused attention backward's atomic dQ sums,
    `attention_mosaic.MosaicBackward`), so a flag flipped within a process derives it again."""
    identity = _identity(op, arguments)
    return identity, target, refusals_of(identity), deterministic_ops_requested()


@generation.traced_fact
def refusals_of(identity: str) -> frozenset[tuple[Hashable, frozenset[Refused]]]:
    """Each config this context's compiles refused the launch of `identity` (`_identity`), with the kernels
    refused at it and their roles (`Refused`): what a launch's derivation and legality read of the
    refusals, the one way a trace reads
    them (`traced_fact`), so a trace kept past a refusal of another launch holds."""
    with _GUARD:
        registry = _registry()
        if (found := registry.by_launch.get(identity)) is None:
            found = registry.by_launch[identity] = frozenset(
                (config, kernels) for (launch, config), kernels in registry.refused.items()
                if launch == identity)
        return found


def refused(op, arguments: tokamax.BoundArguments, config: Hashable) -> bool:
    """Whether a compile of this process refused `op`'s launch of `arguments` at `config`: a kernel of it
    spilled or serialized its warpgroup MMAs (`refused_at`). A template's legality reads it, so its derivation
    takes the next config."""
    return bool(refused_at(op, arguments, config))


def refused_any(op, arguments: tokamax.BoundArguments) -> bool:
    """Whether a compile of this process refused any config of `op`'s launch of `arguments` (`refused`)."""
    return bool(refusals_of(_identity(op, arguments)))


def refused_at(op, arguments: tokamax.BoundArguments, config: Hashable) -> frozenset[str]:
    """The names of `op`'s kernels a compile of this process refused at `config` for `arguments`
    (`Binary.refused`: spilling, or their warpgroup MMAs serialized): a launch of several kernels (a
    recurrence's forward and backward walks) steps the one refused."""
    return names(dict(refusals_of(_identity(op, arguments))).get(config, ()))


def names(kernels: Iterable[Refused]) -> frozenset[str]:
    """The names of refused `kernels` (`Refused`): what a refusal's readers step by, its roles aside."""
    return frozenset(kernel for kernel, _ in kernels)


def spilled_in(op, arguments: tokamax.BoundArguments, config: Hashable) -> dict[str, tuple[str, ...]]:
    """The roles each of `op`'s kernels a compile refused at `config` for `arguments` spilled in, as its
    refusal states them (`Refused`), by kernel name: what a launch's ladder steps it by (row 6744). A kernel
    whose roles its compile left unread is absent; one read and spilling none (its warpgroup MMAs serialized
    alone) states `()`."""
    found = dict(refusals_of(_identity(op, arguments))).get(config, ())
    return {kernel: roles for kernel, roles in found if roles is not None}


def refuse_launch(op, arguments: tokamax.BoundArguments, config: Hashable, kernels,
                  spilled_in: Mapping[str, tuple[str, ...] | None] | None = None) -> None:
    """`op`'s launch of `arguments` at `config` refused in this context as a compile that found `kernels`
    spilling refuses it (`settled`), so its derivation takes the next config: a compile gate's, which
    compiles a launch apart from `compiled`, stepping as a run's compile steps (row 4727). `spilled_in`, the
    roles each kernel's spill stood in where the compile read them (`Binary.spilled_in`), kept with it
    (`Refused`) for its ladder (`spilled_in`)."""
    roles = spilled_in or {}
    _record((_identity(op, arguments), config), None,
            frozenset((kernel, roles.get(kernel)) for kernel in kernels))


_CERTIFYING: contextvars.ContextVar = contextvars.ContextVar("certifying_device", default=None)
"""The device a launch's derived config is certified on (`certified`) where a block names one
(`certifying`)."""

_TARGETED: contextvars.ContextVar = contextvars.ContextVar("certified_target", default=None)
"""The mesh a run targets (`targeting`), whose GPU a trace of the run outside any mesh certifies on."""


@dataclasses.dataclass(frozen=True)
class Certification:
    """A launch's config certified (`certified`): its op's name and config, whether as reverse mode runs it
    (`pulled`), each Mosaic GPU kernel it compiled alone, by its name and launch (`launch_name`): its module
    (the kernel hash a lowered call states, `KernelCall.config`, by which the program's compile reuses it,
    Mosaic compiling a module once a process) and its binary, and whether its verdict was read back from the
    picks' store (`stored`), compiled by an earlier process, rather than compiled here."""
    op: str
    config: Hashable
    pulled: bool
    kernels: dict[str, tuple[str, "Binary"]]
    stored: bool
    device: Hashable
    """The GPU it was certified on (`_certifying_device`, a `jax.Device`), whose assembler's answer it is."""

    @property
    def refused(self) -> tuple["Binary", ...]:
        """Its kernels the GPU's assembler refuses (`Binary.refused`)."""
        return tuple(binary for _, binary in self.kernels.values() if binary.refused)


_CERTIFIED: dict[tuple, Certification] = {}
"""Each launch certified (`certified`), by its identity, config, device and product."""

_UNCERTIFIED: dict[tuple, str] = {}
"""Each launch certification could not compile alone (`certified`), by its identity, config, device and
product: why."""


_PULLED: contextvars.ContextVar = contextvars.ContextVar("certified_pulled", default=False)
"""Whether a launch traced here runs as reverse mode runs it (`pulled`)."""


def pulling() -> bool:
    """Whether a launch traced here runs as reverse mode runs it (`pulled`)."""
    return _PULLED.get()


@contextlib.contextmanager
def pulled() -> Iterator[None]:
    """Within the block, a gradient's product is traced: a launch whose call carries its own derivative
    (`tuning.KernelOp.CARRIES_DERIVATIVE`, the recurrence template's) is certified as reverse mode runs it,
    its forward and its pullback (`certified`), the kernels a gradient runs where they are not its call's
    without one (`recurrence_chunk._per_product`). A launch whose derivative is its caller's (a port's raw
    launch inside its rule's custom VJP) is certified as it is called."""
    token = _PULLED.set(True)
    try:
        yield
    finally:
        _PULLED.reset(token)


@contextlib.contextmanager
def certifying(device) -> Iterator[None]:
    """Within the block, launches' derived configs are certified on `device` (`certified`): a compile-only
    GPU's, which a process attached to no GPU names for the target it compiles for."""
    token = _CERTIFYING.set(device)
    try:
        yield
    finally:
        _CERTIFYING.reset(token)


@dataclasses.dataclass(frozen=True)
class _AsideLaunch:
    """A launch certified aside (`certifying_aside`): what `certified` read of it, and, once the block's
    trace reached its call (`capture`), its replay's kernels to come (`replay.replayed`: its kernels, their
    verdict's key, and whether that verdict was read back)."""
    op: "KernelOp"
    arguments: tokamax.BoundArguments
    config: Hashable
    reverse: bool
    device: Hashable
    stored: str | None
    kernels: Future | None = None


@dataclasses.dataclass
class Aside:
    """The launches a block certified aside (`certifying_aside`), by `certified`'s key, compiling on
    `pool`; and each whose call the block's trace has not reached yet (`awaiting`), by its launch's digest
    (`launch_digest`) and mode: its `pending` key."""
    pool: ThreadPoolExecutor
    pending: dict[tuple, _AsideLaunch] = dataclasses.field(default_factory=dict)
    awaiting: dict[tuple[str, bool], tuple] = dataclasses.field(default_factory=dict)


_ASIDE: contextvars.ContextVar[Aside | None] = contextvars.ContextVar("certifying_aside", default=None)


@contextlib.contextmanager
def certifying_aside() -> Iterator[Aside]:
    """Within the block, a launch whose derivation has refused none of its configs is certified aside
    (`certified`): lowered as the trace reaches it, Python that holds the GIL, and compiled on
    `compile_threads` threads while the trace runs on, XLA and the GPU's assembler releasing it, its config
    taken as holding until `settled_aside` reads its compile; the block's end waits for every compile. A
    launch with a refusal certifies in turn, its ladder read as it climbs (`tuning.KernelOp._certified`). A
    cold start's 47 launches certified in turn took 48 s, 18 of them compiling (M-MC-21b), and every config
    the gates derive holds (FS-DECISION: an attention forward's 68 and a backward's 45 on sm90), so a
    refusal's relowering is rare."""
    with ThreadPoolExecutor(compile_threads(), thread_name_prefix="dew-certify") as pool:
        aside = Aside(pool)
        token = _ASIDE.set(aside)
        try:
            yield aside
        finally:
            _ASIDE.reset(token)


def settled_aside(aside: Aside) -> bool:
    """Each launch `aside` certified, kept as `certified` keeps one: a launch whose kernels the assembler
    refuses refused (`refuse_launch`) and its misses recorded, stepped where its derivation offers another
    config now, so the program lowered within the block holds a refused config and is lowered afresh.
    Whether any was refused."""
    refused = False
    for key, each in aside.pending.items():
        # A launch a derivation certified (a route's eligibility) and no trace called: no program holds it.
        # One certified in turn meanwhile (a refusal's ladder, `tuning.KernelOp._certified`) is certified.
        with _GUARD:
            done = key in _CERTIFIED
        if each.kernels is None or done:
            continue
        try:
            kernels, lowered, read = each.kernels.result()
        except jax.errors.JaxRuntimeError as error:
            _uncertified_launch(key, each.op, each.config, error)
            continue
        # Its verdict stored by its tree's key and, compiled here, its kernels' (`_lowered_verdict_key`).
        stored = (each.stored,) if read else (each.stored, lowered)
        spilling = _kept_certification(key, each.op, each.config, each.reverse, kernels, stored, each.device,
                                       read=read).refused
        if spilling:
            refuse_launch(each.op, each.arguments, each.config, {binary.kernel for binary in spilling},
                          {binary.kernel: binary.spilled_in for binary in spilling})
            missed(each.op, each.arguments, each.config, spilling,
                   stepped=each.op.remaining(each.arguments) is not None)
            refused = True
    return refused


def _refused_before(op, arguments: tokamax.BoundArguments) -> bool:
    """Whether `op`'s launch of `arguments` had a config refused in this context (`refused_launches`)."""
    identity = _identity(op, arguments)
    return any(refused == identity for refused, _, _ in refused_launches())


@contextlib.contextmanager
def targeting(mesh: jax.sharding.Mesh) -> Iterator[None]:
    """Within the block, a trace outside any mesh certifies as one under `mesh` does (`_certifying_device`):
    a run's trace outside its mesh (its start's, `Trainer._started`) on the GPU the run targets. Certified on
    the host's own, a GH200's start compiling ahead for a compile-only H100 certified five launches twice,
    apart, its step's trace again on the mesh's (row 6824)."""
    token = _TARGETED.set(mesh)
    try:
        yield
    finally:
        _TARGETED.reset(token)


def certified(op, arguments: tokamax.BoundArguments, config: Hashable) -> tuple["Binary", ...] | None:
    """`op`'s kernels at `config` for `arguments` the GPU's assembler refuses (`Binary.refused`: spilling, or
    their warpgroup MMAs serialized), each launch compiled
    alone before its program is: its kernels' binaries (`binaries`) as the plugin's own compiler makes them
    for the device the process compiles for (`_certifying_device`: its own GPU, or a compile-only one a
    block names), which the program's compile then reuses (Mosaic compiles a module once a process). A
    launch's derivation refuses a config that spills and takes its next (`tuning.KernelOp.taken`), with no
    program retraced, where a compile's refusal retraces the whole program (`settled`): a B200's 4B step
    stepped its attention forward 19 times that way, 42 minutes, every rewalk a step's trace. None on a
    process compiling for no GPU or an interpreted launch; None, read nowhere, for a launch XLA or the GPU's
    compiler refuses to compile alone though it lowered (a collective's, its devices the program's; an
    assembly the PTX compiler rejects), which its program's compile reads as before (`settled`) and
    `uncertified` names;
    a launch Mosaic refuses as it lowers (a block past its shared memory) raises here as its program's
    lowering would. Once a process, by launch (`certifications`), and once a compile target a
    machine: each verdict, an accepted config's and a refused one's, is stored beside the kernel picks
    (`_verdict_key`, `verdict_target`), so a warm start traces and compiles nothing to certify."""
    device = _certifying_device()
    if device is None or arguments.arguments.get("interpret"):
        return ()
    from dew.nn.kernels import tuning

    reverse = _PULLED.get() and op.CARRIES_DERIVATIVE
    key = (_identity(op, arguments), config, device.id, device.device_kind, reverse)
    with _GUARD:
        if key in _CERTIFIED:
            return _CERTIFIED[key].refused
        if key in _UNCERTIFIED:
            return None
    stored = _verdict_key(_identity(op, arguments), config, reverse)
    verdict = None if stored is None else tuning.stored_verdict(verdict_target(device), stored)
    if isinstance(verdict, dict):
        return _kept_certification(key, op, config, reverse, verdict, (), device, read=True).refused
    aside = _ASIDE.get()
    if aside is not None and key in aside.pending:
        return ()
    if aside is not None and not _refused_before(op, arguments):
        # Taken as holding, nothing lowered here: the block's trace's own call of the launch is replayed and
        # compiled aside (`capture`), the block's end settling it (`settled_aside`), once a block.
        aside.pending[key] = _AsideLaunch(op, arguments, config, reverse, device, stored)
        aside.awaiting[(_launched(key[0], config, reverse), reverse)] = key
        return ()
    # Its verdict where its kernels' was stored however their Dew tree named it (`_lowered_verdict_key`),
    # stored by this tree's key too, so its next start reads it tracing nothing.
    lowered, modules, lowered_key = _lowered_launch(op, arguments, config, device, reverse)
    verdict = None if lowered_key is None else tuning.stored_verdict(verdict_target(device), lowered_key)
    if isinstance(verdict, dict):
        return _kept_certification(key, op, config, reverse, verdict, (stored,), device, read=True).refused
    unstored = (stored, lowered_key)
    try:
        kernels = _launch_binaries(lowered, modules)
    except jax.errors.JaxRuntimeError as error:
        _uncertified_launch(key, op, config, error)
        return None
    return _kept_certification(key, op, config, reverse, kernels, unstored, device).refused


def _uncertified_launch(key: tuple, op, config: Hashable, error: Exception) -> None:
    """XLA refuses the launch alone (a collective's devices): its program's compile reads it as it did
    before certification (`settled`), and the gate names it (`uncertified`)."""
    _log.warning("launch %s at %s is not certified: XLA does not compile it alone (%s)",
                 type(op).__name__, config, str(error)[:200])
    with _GUARD:
        _UNCERTIFIED[key] = str(error)[:400]


def _kept_certification(key: tuple, op, config: Hashable, reverse: bool, kernels: dict,
                        stored: tuple[str | None, ...], device, *, read: bool = False) -> "Certification":
    """A launch's certification recorded once a process (`certifications`): its kernels `read` back from the
    picks' store, else compiled here, their verdict stored for the GPU model by each of `stored`'s keys that
    found none (None where no key states the installation's code)."""
    from dew.nn.kernels import tuning

    for each in stored:
        if each is not None:
            tuning.store_verdict(verdict_target(device), each, kernels)
    found = Certification(type(op).__name__, config, reverse, kernels, read, device)
    with _GUARD:
        _CERTIFIED[key] = found
    return found


def verdict_target(device) -> str:
    """The compile target whose verdicts a certification on `device` stores and reads
    (`tuning.stored_verdict`): its generation (`generation.device_target`), which names what the assembler
    builds a launch's kernels for, the verdict's one hardware input. An H100, a GH200 and a compile-only H100
    share sm90's, so a build certifying for a compile-only GPU stores the verdicts a run on its GPU reads,
    where by GPU model a GH200's run read none of them (row 6717)."""
    return generation.device_target(device).generation


def _lowered_verdict_key(calls) -> str | None:
    """What decides a lowered launch's certification on a GPU model, as its stored verdict's key once it has
    lowered (`tuning.stored_verdict`): each Mosaic GPU and Pallas-Triton kernel it compiles, by its name, its
    IR's digest and its config, the installed JAX, jaxlib and CUDA plugins, whose compiler assembles them (the
    Pallas lowering cache's seal), and the environment a kernel's compile reads (that cache's policy); the
    store's file names the GPU model. Its kernels decide its verdict, not the Dew tree that lowered them: a
    tree whose own key (`_verdict_key`) finds none, an edit's anywhere outside them, lowers the launch and
    reads its verdict, compiling nothing (row 8650: 23 of an edited tree's 51 seconds certifying a hybrid
    decoder's step ahead, FS-8650-SPLIT). None where the installation has no seal."""
    from jax._src.pallas.mosaic_gpu.lowering_cache import _keyed_environment
    from jax._src.pallas.mosaic_gpu.lowering_cache_seal import seal

    installed = seal().digest
    if installed is None:
        return None
    kernels = sorted((call.kind, call.name, hashlib.sha256(call.ir).hexdigest(), call.config)
                     for call in calls if call.kind in ("mosaic_gpu", "triton"))
    stated = ["lowered", installed.hex(), kernels, _keyed_environment()]
    return hashlib.sha256(json.dumps(stated, sort_keys=True).encode()).hexdigest()


def _verdict_key(identity: str, config: Hashable, reverse: bool) -> str | None:
    """What decides a launch's certification for a compile target (the store's file, `verdict_target`,
    `tuning.stored_verdict`) before it traces, as its stored verdict's key: the launch's `identity`
    (`_identity`, the same in every process), its config, whether reverse mode runs it, Dew's code
    (`package_digest`: a Dew edit certifies again), the installed JAX, jaxlib and CUDA plugins, whose Mosaic
    GPU and XLA lower and compile it (the Pallas lowering cache's seal of them), the ptxas that assembles it
    (`_assemblers`), and the environment a kernel's lowering and compile read (that cache's policy). None
    where the installation has no seal (an edited or a source install of JAX), whose code no key states: its
    verdicts are not stored."""
    from jax._src.pallas.mosaic_gpu.lowering_cache import _keyed_environment
    from jax._src.pallas.mosaic_gpu.lowering_cache_seal import seal

    from dew._cache import package_digest
    from dew.nn.kernels.tuning import encoded

    installed = seal().digest
    if installed is None:
        return None
    stated = [package_digest(), installed.hex(), _assemblers(), identity, encoded(config), reverse,
              _keyed_environment()]
    return hashlib.sha256(json.dumps(stated, sort_keys=True).encode()).hexdigest()


@functools.cache
def _assemblers() -> list[str]:
    """The installed distributions holding the ptxas a kernel's PTX is assembled with, CUDA's nvcc wheels
    (`nvidia-cuda-nvcc`, `nvidia-cuda-nvcc-cu12`), each as `name==version`: a relock that moves one
    certifies again, where the seal, which holds the CUDA plugins alone, kept a verdict another ptxas
    judged. A system CUDA's ptxas, outside every wheel, none names (row 7309: the ptxas Mosaic GPU's own
    assembler names, `_get_assembler`, with its version, once Dew's JAX build states it)."""
    named = (((distribution.metadata.get("Name") or "").lower(), distribution)
             for distribution in importlib.metadata.distributions())
    return sorted(f"{name}=={distribution.version}" for name, distribution in named
                  if re.fullmatch(r"nvidia-cuda-nvcc(-cu\d+)?", name))


def certifications() -> tuple[Certification, ...]:
    """Each launch certified in this process (`certified`)."""
    with _GUARD:
        return tuple(_CERTIFIED.values())


def uncertified() -> dict[tuple, str]:
    """Each launch certification did not compile alone in this process (`certified`), and why."""
    with _GUARD:
        return dict(_UNCERTIFIED)


def missed(op, arguments: tokamax.BoundArguments, config: Hashable, refused: Sequence["Binary"],
           *, stepped: bool) -> None:
    """Each of `refused`'s kernels (`certified`'s, at `config`) recorded as a miss (`misses`), as `settled`
    records a compile's: `stepped` where its launch's derivation took another config after its refusal
    (`refuse_launch`), not where it had none left."""
    launch = Launch(_identity(op, arguments), type(op).__name__, config)
    _recorded([(None, binary, launch) for binary in refused], stepped=stepped)


def _certifying_device():
    """The device a launch is certified on (`certified`): a block's (`certifying`), else the GPU the trace's
    program runs on, the concrete mesh's (`jax.set_mesh`: a compile-only GPU's where a run's start compiles
    ahead for one, `dew.training.ahead`), else the mesh's its run targets (`targeting`), else, nothing
    naming a target, JAX's default device, which a call no mesh places runs on (a plain jit's); either where
    it compiles for that GPU's own generation (`generation.device_target`, whatever `assumed` says: an A100
    tracing for an assumed sm90 compiled the sm90 launch on itself, and Mosaic refused its TMA copies, rows
    6677 and 6678), else none. A trace under a compile-only GPU's mesh, or a run's outside it, certified on
    the host's own GPU certified each launch a second time, apart, where the start's compile certified it on
    the mesh's (rows 6791, 6824)."""
    from jax._src import mesh as mesh_lib

    named = _CERTIFYING.get()
    if named is not None:
        return named
    placed = mesh_lib.get_concrete_mesh()
    if placed.empty:
        placed = _TARGETED.get()
    if placed is None:
        default = jax.config.jax_default_device
        devices = [default if isinstance(default, jax.Device) else jax.devices(default)[0]]
    else:
        devices = [device for device in placed.devices.flat if device.process_index == jax.process_index()]
    # A device of another generation (a CPU's, an A100's under an assumed sm90) certifies nothing.
    if not devices or generation.device_target(devices[0]).generation != generation.gpu_target().generation:
        return None
    return devices[0]


def _lowered_launch(op, arguments: tokamax.BoundArguments, config: Hashable, device, reverse: bool
                    ) -> tuple[jax.stages.Lowered, dict, str | None]:
    """`op`'s launch of `arguments` at `config` lowered alone for `device` (as reverse mode runs it where
    `reverse`), beside its kernels' modules (a Triton call's config) by name, and their verdict's key
    (`_lowered_verdict_key`)."""
    from dew.nn.kernels.options import owned_program
    from dew.nn.programs import meshless

    mesh, manual = _certifying_placement(device)
    placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # A program of its own on one device (`meshless`): none of the enclosing trace's mesh nor its axes, its
    # operands' too, but the mesh its program runs on where that is one device's (`_certifying_placement`),
    # within a map holding the axes manual its program's call holds manual (`sharding.manual_map`'s).
    with meshless(mesh):
        launch, shapes = op.lowering(arguments, config)
        if manual:
            launch = jax.shard_map(launch, mesh=mesh, in_specs=jax.sharding.PartitionSpec(),
                                   out_specs=jax.sharding.PartitionSpec(), axis_names=manual, check_vma=False)
        specs = [jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=placed) for shape in shapes]

        def pulled_back(*operands):
            # Its outputs too, as a step reads them: dropped, the kernels writing only them were dead code
            # to XLA and went uncertified (the 4B decision step's recurrence_output, LF-PROMO3, whose
            # program then stepped its producer down a whole-step relowering a rung).
            out, pullback = jax.vjp(launch, *operands)
            return out, pullback(jax.tree.map(_unit_cotangent, out))

        def certified(*operands):
            return (pulled_back if reverse else launch)(*operands)

        certified.__name__ = certified.__qualname__ = _certification_name(type(op).__name__,
                                                                          _identity(op, arguments), config,
                                                                          reverse)
        lowered = owned_program(certified).trace(*specs).lower()
    return lowered, *_lowered_kernels(lowered)


def _certification_name(op: str, identity: str, config: Hashable, reverse: bool) -> str:
    """A certification's program's name in JAX's trace, lowering and compile events, so a start's receipt
    tells each certification's program apart: its op, its launch's identity and its config, digested."""
    return f"certified_{op}_l{_digest(identity)}_c{_digest(repr(config))}{'_reverse' if reverse else ''}"


def _lowered_kernels(lowered: jax.stages.Lowered) -> tuple[dict, str | None]:
    """A certification's lowered launch's Mosaic GPU and Pallas-Triton kernels' modules (a Triton call's
    config) by name, and their verdict's key (`_lowered_verdict_key`). A Mosaic GPU kernel compiled alone is
    the one its program loads (Mosaic compiles a module once a process). A Pallas-Triton kernel compiles again
    with each executable, the same kernel the same binary: read here too, its call's config its identity, so
    its derivation steps a spilling config at trace as a Mosaic launch's does (an H100's float32 attention
    forward at width 256 spilled at every four-warp tile and held at eight, row 6507), rather than ending its
    program's compile Exhausted."""
    from dew.nn.kernels.lowered import kernel_calls

    calls = kernel_calls(lowered, unrun=True)
    modules = kernel_modules(calls) | {name: call.config for name, call in
                                       kernel_bodies(call for call in calls if call.kind == "triton").items()}
    return modules, _lowered_verdict_key(calls)


def awaiting() -> tuple | None:
    """The `pending` key of the launch tracing here (`launching`) where the block it traces in certifies it
    aside and no trace has called it yet (`certified`), taken: its call is captured
    (`tuning.KernelOp.__call__`, `capture`). None for every other call, which runs as it is."""
    aside, pending, digest = _ASIDE.get(), _TRACING.get(), launch_digest()
    if aside is None or pending is None or digest is None:
        return None
    with _GUARD:
        return aside.awaiting.pop((digest, pending.reverse), None)


def capture(key: tuple, closed) -> None:
    """The launch certified aside under `key` (`certified`), its call as its block's trace made it
    (`closed`, its `ClosedJaxpr`: at the operands' own abstract values, a shard's inside a shard_map),
    replayed and compiled on the block's pool (`replay.replayed`) on the mesh it was traced on
    (`replay.replay_mesh`): no kernel body traced a second time, and its kernels lowered as its program
    lowers them, which the lowering cache then holds once (row 8720: 22 of a hybrid decoder's 36 kernels
    lowered twice, a launch alone on one device and a shard's in its program). A launch whose operands are
    placed on meshes of two shapes certifies by its program's compile (`settled`)."""
    from dew.nn.kernels.templates.replay import replay_mesh, replayed

    aside = _ASIDE.get()
    assert aside is not None, "a capture is of a block's trace"
    with _GUARD:
        each = aside.pending.get(key)
        done = key in _CERTIFIED
    if each is None or each.kernels is not None or done:
        return
    mesh = replay_mesh(closed, each.device)
    if mesh is None:
        with _GUARD:
            aside.pending.pop(key, None)
            _UNCERTIFIED[key] = "its trace placed its operands on meshes of two shapes"
        return
    name = _certification_name(type(each.op).__name__, key[0], each.config, each.reverse)
    compiling = aside.pool.submit(contextvars.copy_context().run, replayed, closed, mesh, each.reverse,
                                  each.device, name)
    aside.pending[key] = dataclasses.replace(each, kernels=compiling)


def _launch_binaries(lowered: jax.stages.Lowered, modules: dict) -> dict[str, tuple[str, "Binary"]]:
    """A launch `_lowered_launch` lowered, compiled: its kernels' modules (`modules`) and binaries, by name
    and launch, none for a launch of none."""
    if not modules:
        return {}
    # Compiled to be read, never run (`kernels_read`): its kernels' binaries are what it states, so a prepare
    # for devices no run has tuned (`for_devices`) certifies as any compile does, its fusions untuned, and an
    # attached GPU's certification is the compile a deviceless build of its generation made (row 8372).
    with kernels_read():
        compiled = xla_compiled(lowered, None)
    return {name: (modules[name], binary) for binary in binaries(compiled, spills=True)
            if (name := launch_name(binary.kernel, binary.launch)) in modules}


def _certifying_placement(device) -> tuple[jax.sharding.Mesh, frozenset]:
    """The mesh certification places a launch's operands on `device` on and sets its program on
    (`_lowered_launch`), and the axes its launch holds manual: its program's where that runs
    on `device` alone (the concrete mesh, `jax.set_mesh`'s, its axes manual where the call is in a map
    holding them, `sharding.manual_map`'s), so the launch lowers to the jaxpr and the mesh its program's call
    does (the GPU it states and each jit's recorded mesh among them) and the program reads its kernels'
    lowerings back (the Pallas lowering cache keys both); else a mesh of its own on `device`, as a program of
    no mesh or a launch of many devices' (row 6327): its own compile cannot hold the program's devices, so
    its program lowers its kernels again."""
    from jax._src import mesh as mesh_lib

    placed = mesh_lib.get_concrete_mesh()
    if not placed.empty and list(placed.devices.flat) == [device]:
        return placed, frozenset(jax.sharding.get_abstract_mesh().manual_axes)
    return jax.sharding.Mesh([device], ("launch",)), frozenset()


def _unit_cotangent(out: jax.Array) -> jax.Array | np.ndarray:
    """A cotangent of ones for `out`, float0's zeros for an integer output."""
    if jnp.issubdtype(out.dtype, jnp.inexact):
        return jnp.ones_like(out)
    return np.zeros(out.shape, jax.dtypes.float0)


@generation.traced_fact
def refusal_entries() -> tuple[tuple[str, Hashable, frozenset[Refused]], ...]:
    """Every (launch identity, config, spilled kernels with their roles, `Refused`) refused in this context (a
    `refusals` scope's or the process's), ordered. A validated plan can hand them to another process
    explicitly, the roles with them (`refuse` within `refusals`); a shared compilation cache alone must not
    transfer them between programs. A trace after a refusal derives the next config, so a program kept after
    one is another than a process without it traces (`traced_fact`): a trace's key holds them, roles and
    all (`options.held_context`)."""
    with _GUARD:
        return tuple(sorted(((identity, config, kernels)
                             for (identity, config), kernels in _registry().refused.items()), key=repr))


def refused_launches() -> tuple[tuple[str, Hashable, frozenset[str]], ...]:
    """`refusal_entries`, each launch's spilled kernels by their names: what a plan's record lists
    (`dew.training.planning.refusal_record`)."""
    return tuple((identity, config, names(kernels)) for identity, config, kernels in refusal_entries())


def refuse(entries) -> None:
    """Each `(identity, config, kernels)` of `entries` (`refusal_entries`', its kernels with their roles,
    `Refused`, or a pair a decoded record gives as a list; a kernel by its name alone states none) refused in
    this context, as a compile's spill refuses it. A handed plan seeds these inside its own `refusals` scope,
    not unioned with unrelated decisions that the receiving process already holds."""
    for identity, config, kernels in entries:
        _record((identity, config), None, frozenset(map(_refused, kernels)))


def _refused(kernel) -> Refused:
    """A kernel `refuse` is handed as its record (`Refused`): its name alone, no roles stated, or its name and
    roles, a tuple's or a list's."""
    if isinstance(kernel, str):
        return kernel, None
    name, roles = kernel
    return name, None if roles is None else tuple(roles)


def misses() -> tuple[Miss, ...]:
    """The kernels this process's compiles found spilling (`compiled`), the latest `RECORDS_HELD`, for a run's
    telemetry: a `refusals` scope's own within it."""
    with _GUARD:
        return tuple(_registry().missed)


def record_rewrite(made) -> None:
    """Record one region the algebra pass rewrote (`dew.algebra.program.AlgebraReport`) beside the kernels'
    misses, as each program traces."""
    _add(rewritten=(made,))


def record_pass(seconds: float) -> None:
    """Record one program's whole algebra pass, its seconds (`dew.algebra.program._passed`)."""
    _add(passes=(seconds,))


def passes() -> tuple[float, ...]:
    """Each program's whole algebra pass, in seconds, as this process's programs traced (`record_pass`), the
    latest `RECORDS_HELD`: its regions' (`rewrites`) and its walks of the program around them."""
    with _GUARD:
        return tuple(_registry().passes)


def rewrites() -> tuple:
    """The regions the algebra pass rewrote as this process's programs traced (`record_rewrite`), the latest
    `RECORDS_HELD`, for a run's telemetry: a `refusals` scope's own within it."""
    with _GUARD:
        return tuple(_registry().rewritten)


@contextlib.contextmanager
def recording() -> Iterator[Recorded]:
    """Within the block, every miss, rewrite and pass timing this context records (`Recorded`), as well as
    its registry's: a search's own, whatever the registry holds besides. A replacement's are recorded once
    it is kept (`_commit`)."""
    recorded = Recorded()
    token = _RECORDING.set((*_RECORDING.get(), recorded))
    try:
        yield recorded
    finally:
        _RECORDING.reset(token)


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
def _proposing(entries: list[tuple[tuple[str, Hashable], Refused]]) -> Iterator[Recorded]:
    """Refusals visible while deriving and compiling a replacement, each kernel with its roles (`Refused`),
    without changing the kept program's registry or its file; every rewrite and pass the replacement's trace
    records, which a successful replacement commits after this scope ends (`_commit`)."""
    registry = _registry()
    with _GUARD:
        proposed = dict(registry.refused)
    for identity, kernel in entries:
        proposed[identity] = _joined(proposed.get(identity, frozenset()) | {kernel})
    # A replacement's records reach the registry and a `recording` only once it is kept (`_commit`).
    proposal = Recorded()
    token, held = _SCOPE.set(_Registry(refused=proposed)), _RECORDING.set((proposal,))
    try:
        yield proposal
    finally:
        _RECORDING.reset(held)
        _SCOPE.reset(token)


def _commit(proposal: Recorded) -> None:
    """A successful replacement's rewrites and its passes' seconds (`_proposing`'s) recorded as the kept
    program's."""
    _add(rewritten=proposal.rewritten, passes=proposal.passes)


def _add(*, missed: Sequence["Miss"] = (), rewritten: Sequence = (), passes: Sequence[float] = ()) -> None:
    """`Recorded` entries added to this context's registry and to every `recording` open in it: a miss once
    a holder (`_record`), each rewrite and pass as often as it is made."""
    with _GUARD:
        for held in (_registry(), *_RECORDING.get()):
            held.missed.extend(miss for miss in missed if miss not in held.missed)
            held.rewritten.extend(rewritten)
            held.passes.extend(passes)


def _joined(kernels: frozenset[Refused]) -> frozenset[Refused]:
    """Refused `kernels` (`Refused`) a kernel once: each kernel's roles every record of it read, joined, None
    where none read them."""
    roles: dict[str, set[str] | None] = {}
    for kernel, read in kernels:
        if read is not None:
            roles[kernel] = (roles.get(kernel) or set()) | set(read)
        else:
            roles.setdefault(kernel, None)
    return frozenset((kernel, None if read is None else tuple(sorted(read)))
                     for kernel, read in roles.items())


def _record(refusal: tuple[str, Hashable] | None, miss: "Miss | None",
            kernels: frozenset[Refused] = frozenset()) -> bool:
    """`refusal` and `miss` recorded; whether `miss` is new to this context's registry. A miss is recorded
    once a holder, the registry and each `recording` open: a launch's derivation asked again (its route's
    eligibility, then its call, `tuning.KernelOp.taken`) reads its certification back and finds the same
    misses, and a recording opened since the first (a later plan's admission, `planning.admitted`) holds
    the misses its own trace meets, those the process met before among them."""
    registry = _registry()
    with _GUARD:
        if refusal is not None:
            registry.refused[refusal] = _joined(registry.refused.get(refusal, frozenset()) | kernels)
            registry.by_launch.pop(refusal[0], None)
        if miss is None:
            return False
        new = miss not in registry.missed
        _add(missed=(miss,))
        return new


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
    compiled in turn, as it would have been lowered after them. At most `compile_threads()` compile at
    once."""
    firsts = attempts([(lowered, options) for lowered, options, _ in programs])
    held = refusal_entries()
    kept_programs = []
    for (lowered, options, relower), first in zip(programs, firsts, strict=True):
        if refusal_entries() != held:
            lowered, first = relower(), None
        kept_programs.append(compiled(lowered, options, relower=relower, first=first))
    return kept_programs


def compile_threads() -> int:
    """The most programs `compiled_at_once` compiles at once, and launches `certifying_aside` compiles: half
    the CPUs this process may use (`cgroups.usable_cpus`, its affinity and container quota), a compile
    threading its LLVM passes over about two (four on an 8-core host took 2.2 times the programs a second of
    one, each 1.8 times slower than alone, where two took 1.94 times, FS-2440-CONCURRENT2): a 4-vCPU
    container's two, a 72-core Grace's 36."""
    from dew import _cgroups as cgroups
    return max(1, cgroups.usable_cpus() // 2)


@dataclasses.dataclass(frozen=True)
class Attempt:
    """A program's first compile (`attempted`): its executable, or the error XLA raised of it, which its
    settlement raises in its caller's turn (`settled`) where it is no refusal of a Triton kernel asking
    shared bytes past a block (`shared_asked`)."""

    executable: jax.stages.Compiled | None
    error: Exception | None = None
    """A `jax.errors.JaxRuntimeError`, which JAX names as a value, no type."""


def attempted(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None, *,
              background: bool = False) -> Attempt:
    """`lowered` compiled once under `options`: XLA's compile alone, which reads and writes nothing of this
    process's but the executable it makes and the persistent compilation cache, so a thread of its own
    runs it (`compiled_at_once`), `background` where no step waits on it (`compile_slots`); its kernels'
    binaries are read where it is kept (`settled`), in its caller's thread, as reading them walks the
    executable's MLIR (`binaries`), which two threads at once failed in (a Qwen3.5-4B step and its sampled
    step for a compile-only H100: `MLIRError: Failure while executing pass pipeline`)."""
    try:
        return Attempt(xla_compiled(lowered, options, background=background))
    except jax.errors.JaxRuntimeError as error:
        return Attempt(None, error)


UNTUNED = ("No cached config found for HLO instr:", "Failed to get configs for:",
           "Missing deviceless CUB scratch size data")
"""What XLA says where a compile without a device, every tuned choice required from the autotune cache, meets
an instruction no device run has tuned yet (config_assigner.cc's NotFound, sort_rewriter.cc's CUB scratch)."""


class Untuned(RuntimeError):
    """A compile without a device for a device to run (`for_devices`: a job's prepare,
    `dew.training.ahead.AheadTarget`) met an instruction no device run has tuned yet (`UNTUNED`): the
    program (`lowered`) is not preparable until one has, and its run compiles it on the device,
    which tunes it for the next. No memory refusal: the plan's search stops at it."""

    def __init__(self, said: str, lowered: jax.stages.Lowered):
        super().__init__(said)
        self.lowered = lowered


class CompiledOnDevice(Untuned):
    """A compile without a device for a device to run met a choice only the device makes (`at_run_time`):
    the programs it names (`programs`; the compile's own where it names none) are not preparable at all, and
    their run compiles each on its GPU, which picks every choice itself. Refused before XLA compiles, so no
    executable leaving the choice to run time reaches the persistent cache under the key the run's own
    compile reads. An `Untuned` by intent: each says a choice the device's own compile makes, where a
    handler of either compiles the program on its device; one naming how to tune it catches this first."""

    def __init__(self, lowered: jax.stages.Lowered, programs: Sequence[str] = ()):
        super().__init__("a cuDNN convolution compiled without its GPU leaves its algorithm to run time "
                         "(DEVICELESS_CONVOLUTION_PICKS): compiled on its GPU"
                         + (f" ({', '.join(programs)})" if programs else ""), lowered)
        self.programs = tuple(programs)


DEVICELESS_CONVOLUTION_PICKS = False
"""Whether this jaxlib's compile without a device (`deviceless`) can carry a cuDNN convolution's algorithm.
It cannot: a compile-only GPU has no cuDNN to ask, so the convolution keeps algorithm -1, which cuDNN picks
at run time, and on an A100 not the same each run (row 7941's UNet fp32 cell, two runs of one executable
apart). XLA records no convolution's pick to read instead: on the A100 a device compile measuring among
deterministic algorithms (deterministic ops off, autotuner at level 4) wrote nothing to its per-fusion
autotune cache in UPDATE mode, and an empty autotune results dump, while every device compile, level 0 to 4,
picked the same algorithms and ran bit for bit (jaxlib 0.12, UB-PICKS2-a100, 11-19). A jaxlib that stores
them (row 8527) retires `at_run_time`'s refusal."""


def at_run_time(lowered: jax.stages.Lowered) -> bool:
    """Whether `lowered`, compiled for its target, leaves a choice to run time that its device's own compile
    makes: a convolution, which XLA's GPU compiles to a cuDNN call, compiled for compile-only GPUs
    (`DEVICELESS_CONVOLUTION_PICKS`). Read from its StableHLO before XLA compiles it."""
    devices = target_devices(lowered)
    if DEVICELESS_CONVOLUTION_PICKS or not devices or not deviceless(next(iter(devices))):
        return False
    return any_op(lowered, lambda op: op.name == "stablehlo.convolution")


def any_op(lowered: jax.stages.Lowered, holds: Callable[[ir.Operation], bool]) -> bool:
    """Whether an operation of `lowered`'s StableHLO `holds`, its walk stopped at the first."""
    found = False

    def visit(op: ir.Operation) -> ir.WalkResult:
        nonlocal found
        found = holds(op)
        return ir.WalkResult.INTERRUPT if found else ir.WalkResult.ADVANCE

    module = lowered.compiler_ir("stablehlo")
    assert module is not None, "a lowered program states its StableHLO module"
    module.operation.walk(visit)
    return found


def left_at_run(compiled: jax.stages.Compiled) -> bool:
    """Whether `compiled` holds a cuDNN convolution whose algorithm its compile left to run time (its backend
    config's `algo_id` -1), as every compile without the GPU leaves it (`DEVICELESS_CONVOLUTION_PICKS`):
    an executable compiled ahead that its load must not take (`dew.training.ahead.AheadOfTime.at_run`),
    each module's instructions read through XLA's own schema (`jax.extend.xla.instructions`) and a call's
    kind by the one table of custom call targets (`instrumentation.custom_call_kind`)."""
    from jax.extend import xla

    from dew.telemetry.instrumentation import custom_call_kind

    executable = compiled.runtime_executable()
    for module in [] if executable is None else executable.hlo_modules():
        for read in xla.instructions(module):
            proto = read.proto
            if read.opcode != "custom-call" or custom_call_kind(proto.custom_call_target) != "convolution":
                continue
            config = json.loads(proto.backend_config or "{}")
            if config.get("cudnn_conv_backend_config", {}).get("algorithm", {}).get("algo_id") == "-1":
                return True
    return False


def target_devices(lowered: jax.stages.Lowered) -> Sequence | None:
    """The GPUs `lowered` compiles for, None where it compiles for none."""
    from jax._src.interpreters import pxla

    lowering = lowered._lowering
    devices = lowering._device_list if isinstance(lowering, pxla.MeshComputation) else None
    return None if devices is None or next(iter(devices)).platform != "gpu" else devices


def untuned(code: str, message: str) -> bool:
    """Whether XLA refused a deviceless compile for want of a tuned result (`UNTUNED`)."""
    return code == "NOT_FOUND" and any(said in message for said in UNTUNED)


class CompileSlots:
    """The compiles this process runs at once (`compile_threads`), shared by every thread that compiles
    (`xla_compiled`): one a step waits on takes any free slot; one no step waits on (`background`: a
    start's program compiled beside its work, `options.compiling`) takes one only while one is free and no
    other waits, so on a host of one slot a step's compile never queues behind it."""

    def __init__(self, count: int):
        self._free, self._waiting = count, 0
        self._changed = threading.Condition()

    @contextlib.contextmanager
    def held(self, *, background: bool) -> Iterator[None]:
        """A slot held for the block, taken as its priority allows."""
        with self._changed:
            if background:
                self._changed.wait_for(lambda: self._free > 0 and self._waiting == 0)
            else:
                self._waiting += 1
                try:
                    self._changed.wait_for(lambda: self._free > 0)
                finally:
                    self._waiting -= 1
            self._free -= 1
            self._changed.notify_all()
        try:
            yield
        finally:
            with self._changed:
                self._free += 1
                self._changed.notify_all()


@functools.cache
def compile_slots() -> CompileSlots:
    """This process's compile slots (`CompileSlots`), counted once at first use (`compile_threads`)."""
    return CompileSlots(compile_threads())


def xla_compiled(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None, *,
                 background: bool = False) -> jax.stages.Compiled:
    """`lowered` compiled by XLA under `options` for its target (`targeted`), in one of this process's
    compile slots (`compile_slots`, `background` where no step waits on it): Dew's one call of XLA's
    compile, where an instruction no device run has tuned yet is said so (`Untuned`), and a choice only the
    device makes, for a device to run, is refused before XLA compiles (`CompiledOnDevice`)."""
    devices = target_devices(lowered)
    if devices is not None and not offline(devices) and at_run_time(lowered):
        raise CompiledOnDevice(lowered)
    compiled_with = targeted(lowered, options)
    # Its options as JAX's persistent cache keys its compile by them (tests/cache_keys.py reads them),
    # written only where the log is read.
    _log.debug("a compile's options: %s", Said(compiled_with))
    try:
        with compile_slots().held(background=background):
            return lowered.compile(compiled_with)
    except jax.errors.JaxRuntimeError as error:
        if untuned(error.error_code_string, str(error)):
            raise Untuned(str(error), lowered) from error
        raise


class Said:
    """A compile's options as a log line writes them, made only where the line is written (`str`)."""

    def __init__(self, options: jax.stages.CompilerOptions | None):
        self.options = options

    def __str__(self) -> str:
        return json.dumps(sorted((self.options or {}).items()), default=str)


CUB_NO_FALLBACK = 3
"""`xla_gpu_deviceless_cub_mode`'s DEVICELESS_CUB_NO_FALLBACK: a deviceless compile's CUB sort reads its
scratch from XLA's table or fails (NOT_FOUND), never keeping another sort under the device compile's key; a
device compile returns before it reads the mode (sort_rewriter.cc), so every GPU compile takes it and the
keys stay equal."""
AUTOTUNE_CACHE_READ = 2
"""`xla_gpu_experimental_autotune_cache_mode`'s READ: a deviceless compile takes each tuned choice from the
autotune cache beside the persistent cache (`--xla_gpu_per_fusion_autotune_cache_dir`), which device compiles
write (the mode's default, UPDATE)."""

_FOR_DEVICES: contextvars.ContextVar[bool] = contextvars.ContextVar("dew_for_devices", default=False)



@contextlib.contextmanager
def for_devices() -> Iterator[None]:
    """Within the block, a deviceless GPU compile makes a program a device will run (a step compiled
    ahead, into a file or the persistent cache a run reads, `dew.training.ahead`): it takes every tuned
    choice from the autotune cache a device run wrote and fails where one is missing (`targeted`), so no
    untuned program reaches a device under a tuned key (row 6046)."""
    held = _FOR_DEVICES.set(True)
    try:
        yield
    finally:
        _FOR_DEVICES.reset(held)


@contextlib.contextmanager
def read_offline() -> Iterator[None]:
    """Within the block, a deviceless GPU compile is one only read for its memory or text, whatever
    `for_devices` holds around it (a call's plan judging a program no device runs, inside a prepare that
    compiles a step ahead): it skips the autotuner and keeps XLA's sorts as they fall back (`offline`,
    `targeted`'s offline compile), so it needs no device run's tuning."""
    held = _FOR_DEVICES.set(False)
    try:
        yield
    finally:
        _FOR_DEVICES.reset(held)


_KERNELS_READ: contextvars.ContextVar[bool] = contextvars.ContextVar("dew_kernels_read", default=False)


@contextlib.contextmanager
def kernels_read() -> Iterator[None]:
    """Within the block, a GPU compile is one read for its kernels' binaries alone and run nowhere (a
    certification's, `_launch_binaries`): read offline on an attached GPU as on a compile-only one
    (`offline`), so a run's certifications and a deviceless build's are one compile under one key."""
    held = _KERNELS_READ.set(True)
    try:
        yield
    finally:
        _KERNELS_READ.reset(held)


def offline(devices: Sequence) -> bool:
    """Whether a program compiled for `devices` is read offline, for its memory or text alone, no device to
    run it (`targeted`): compile-only GPUs outside `for_devices` (or within `read_offline`), and any GPU's
    compile read for its kernels alone (`kernels_read`)."""
    device = next(iter(devices), None)
    return device is not None and device.platform == "gpu" and (
        _KERNELS_READ.get() or (deviceless(device) and not _FOR_DEVICES.get()))


def compiled_as_here(devices: Sequence, *, tuned: bool, at_run: bool) -> bool:
    """Whether a program compiled ahead runs as `devices`' own compile of it would: none of its choices left
    to run time (`at_run`, `at_run_time`) and, where `devices`' compile tunes (`autotuned`), tuned. A
    program that does not is compiled on `devices` in its place."""
    return not at_run and (tuned or not autotuned(devices))


def autotuned(devices: Sequence) -> bool:
    """Whether a program compiled for `devices` takes XLA's tuned choices, as `targeted` compiles it: for a
    device to run (not `offline`), where the run's XLA_FLAGS leaves the autotuner on. One read offline skips
    it, its workspaces XLA's defaults, which a tuned convolution's on the device may exceed."""
    return not offline(devices) and xla_flag("xla_gpu_autotune_level") != "0"


def targeted(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None
             ) -> jax.stages.CompilerOptions | None:
    """`options` for the target `lowered` compiles for. A program a device runs, compiled on it or for it,
    keeps a CUB sort's choice off its key (`CUB_NO_FALLBACK`), the same on both sides. On compile-only GPUs
    (`deviceless`) nothing runs to autotune a product on: a program a device will run (`for_devices`) takes
    each tuned choice the device's runs left in the autotune cache, every one required
    (`xla_gpu_require_complete_aot_autotune_results`; a miss fails as `Untuned`), the executable a device
    compile makes under the same key; one only read for its memory or text (a plan offline, a validation,
    a step compiled into a file, a probe) skips the autotuner (`xla_gpu_autotune_level` 0, its own key,
    which no device's run reads) and keeps XLA's sorts as they fall back, its memory the same (a frontier
    search's every probe and rung, 142 s to 99 s on armada, 11-02). A program holding a dot XLA's Triton
    GEMM computes wrong on the compile's own target (`triton_wrong`) compiles its GEMMs on cuBLAS, on a
    device or for one alike, read from the program and its device alone, opening no backend. A run's own
    XLA_FLAGS setting keeps `options` as given. Every program compiled here takes it, a run's step and
    every program it compiles ahead alike."""
    devices = target_devices(lowered)
    if devices is None:
        return options
    device = next(iter(devices))
    if (xla_flag("xla_gpu_enable_triton_gemm") is None
            and triton_wrong(lowered, generation.device_target(device))):
        options = {**(options or {}), "xla_gpu_enable_triton_gemm": False}
    if offline(devices):
        return options if xla_flag("xla_gpu_autotune_level") is not None else {
            **(options or {}), "xla_gpu_autotune_level": 0}
    if xla_flag("xla_gpu_deviceless_cub_mode") is None:
        options = {**(options or {}), "xla_gpu_deviceless_cub_mode": CUB_NO_FALLBACK}
    if deviceless(device) and xla_flag("xla_gpu_autotune_level") is None:
        return {**(options or {}), "xla_gpu_experimental_autotune_cache_mode": AUTOTUNE_CACHE_READ,
                "xla_gpu_require_complete_aot_autotune_results": True}
    return options


def triton_wrong(lowered: jax.stages.Lowered, target: generation.GpuTarget) -> bool:
    """Whether XLA's Triton GEMM computes one of `lowered`'s own dots (a kernel's products are its custom
    call's) wrong on `target`, as the program states each dot, whatever stated it:
    a product of an int8 operand, which it computes as two or four times the product (openxla/xla#51016)
    on every GPU measured (`GpuTarget.int8_triton_gemm`); with CUDA 13 cuBLAS refuses some int8 shapes
    (on an A100 a contraction of 16 or 48), and then the program does not compile rather than compute
    wrong;
    or fp32 operands at IEEE fp32 products (a precision of `HIGHEST` on either operand, XLA's GPU taking the
    higher of a pair, or an algorithm of fp32 operands one product a term, the F32_F32_F32 preset), which it
    sums farther from float64 than cuBLAS, whose sum a float32 reference is
    (`dew.nn.precision.FP32_PRODUCTS`), as JAX lowers each dot at the precision it resolved: the run's
    default, a model's, an operation's own."""
    from dew.nn.precision import FP32_PRODUCTS, declares_fp32_products
    from dew.nn.programs import algorithm_arithmetic, dot_arithmetic

    def wrong(op: ir.Operation) -> bool:
        if op.name != "stablehlo.dot_general":
            return False
        types = [ir.RankedTensorType(operand.type).element_type for operand in list(op.operands)[:2]]
        if any(isinstance(each, ir.IntegerType) and ir.IntegerType(each).width == 8 for each in types):
            return not target.int8_triton_gemm
        if not all(isinstance(each, ir.F32Type) for each in types):
            return False
        attributes = op.attributes
        if "precision_config" in attributes:
            # Each operand's precision as JAX's enum names it, the pair read as a run's precision is.
            lhs, rhs = (jax.lax.Precision[stablehlo.PrecisionAttr(each).value]
                        for each in ir.ArrayAttr(attributes["precision_config"]))
            if declares_fp32_products((lhs, rhs)):
                return True
        if "algorithm" in attributes:
            stated = algorithm_arithmetic(stablehlo.DotAlgorithm(attributes["algorithm"]))
            return any(stated == dot_arithmetic(name, jnp.float32, jnp.float32) for name in FP32_PRODUCTS)
        return False

    return any_op(lowered, wrong)


def attempts(programs: Sequence[tuple[jax.stages.Lowered, jax.stages.CompilerOptions | None]]
             ) -> list[Attempt]:
    """Each of `programs` compiled once (`attempted`), in order, all at once, at most `compile_threads()`
    at a time; one alone in this thread. JAX's cache key reads a module's context without writing it from
    0.12.0.1.post54 (pyproject's pin), so programs one thread lowered in one context compile together."""
    if len(programs) < 2:
        return [attempted(*program) for program in programs]
    with ThreadPoolExecutor(min(len(programs), compile_threads()), thread_name_prefix="dew-compile") as pool:
        # Each in the submitting thread's context: the compile's target reads it (`for_devices`).
        return [each.result() for each in [pool.submit(contextvars.copy_context().run, attempted, *program)
                                           for program in programs]]


def reads_binaries(lowered: jax.stages.Lowered) -> bool:
    """Whether a compile of `lowered` reads its kernels' binaries: where it runs a Mosaic GPU or a
    Pallas-Triton launch. A program of Pallas-Triton launches alone spills as a Mosaic one does: both
    kinds' binaries are read (`binaries`, a Triton launch's by its custom kernel's tag), so a launch's
    feedback is the same whatever kernels share its program (row 3960: a Triton-only program's
    15,752-byte frame went unread)."""
    from dew.nn.kernels.lowered import kernel_calls

    return any(call.kind in ("mosaic_gpu", "triton") for call in kernel_calls(lowered))


def kept(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None = None, *,
         relower: Callable[[], jax.stages.Lowered], first: Attempt | None = None,
         read: Callable[[jax.stages.Compiled], list["Binary"]] | None = None) -> jax.stages.Compiled:
    """`lowered` compiled and kept (`settled`), its first compile `first` or one in this thread."""
    return settled(lowered, options, relower, attempted(lowered, options) if first is None else first,
                   read=read)


def settled(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
            relower: Callable[[], jax.stages.Lowered], first: Attempt, *,
            read: Callable[[jax.stages.Compiled], list["Binary"]] | None = None
            ) -> jax.stages.Compiled:
    """The program kept of `lowered`'s first compile (`first`), read through its kernels' binaries
    (`read`, `binaries` where None): where a kernel of a template's launch spills, that launch's config is
    refused (`refused`), the miss recorded (`misses`, logged), and the program `relower`ed and compiled,
    again for as long as a
    launch spills at a config not yet refused; `relower` traces afresh, so each refused launch's derivation
    takes the next config by its own rule, through every composition its device has (`bundles.groups`), and
    a launch whose every legal config spilled derives none, its call taking the next route (it leaves the
    template only so). The last compile's own spills are recorded, neither refused nor stepped, so every
    process derives the program this compile keeps; a relowering whose derivations find no config left
    (`Exhausted`: a launch a call pins) raises, its misses recorded as not stepped: a derived kernel never
    spills and a pin is legal (docs/design/kernel-templates.md), so no program found spilling is kept. Each
    relowering's refused launches compile alone until none spills (`_stepped_alone`), the whole program once
    after them. Refusals commit only as a replacement compiles,
    so a later process derives the program kept here. A program loaded from the persistent compilation
    cache carries its kernels' binaries as one compiled here does. A program whose block XLA refuses steps
    its asking Triton launches down (`_unblocked`). A relowering that launches every refused config again
    is the program just compiled: it is kept as compiled, its spills misses not stepped, each refusal that
    did not take named (`_untaken`)."""
    if first.error is not None:
        asked = shared_asked(first.error)
        if asked is None:
            raise first.error
        return _unblocked(lowered, options, relower, asked, first.error)
    executable = first.executable
    assert executable is not None
    found, compiled_from = _misses_of(_refused_binaries(lowered, executable, read)), lowered
    while True:
        # Each step refuses a config no step refused before, of the finitely many its launches derive.
        held = _registry().refused
        refusals = [(refusal, (report.kernel, report.spilled_in)) for refusal, report, _ in found
                    if refusal is not None and report.kernel not in names(held.get(refusal, ()))]
        if not refusals:
            break
        try:
            with _proposing(refusals) as proposal:
                program, alone = _stepped_alone(relower, options, {identity for (identity, _), _ in refusals})
                # A relowering that launches every refused config again is the program just compiled: it is
                # kept, not compiled again (a B200's 4B step compiled the same jit_step twice, 150 s, PF03).
                untaken = _untaken(program, refusals)
                replacement, spilling = ((None, []) if len(untaken) == len(refusals)
                                         else _compiled_reported(program, options, read))
        except Exhausted as failed:
            _recorded(found, stepped=False)
            kernels = ", ".join(sorted({report.kernel for _, report, _ in found}))
            raise Exhausted(f"kernels {kernels} are refused (a spill, or warpgroup MMAs serialized) and "
                            f"their launches have no config left: {failed}") from failed
        if replacement is None:
            break
        _commit(proposal)
        for refusal, kernel in [*refusals, *((refusal, (report.kernel, report.spilled_in))
                                             for refusal, report, _ in alone)]:
            _record(refusal, None, frozenset({kernel}))
        _held(found, compiled_from)
        _held(alone, program)
        _recorded([*found, *alone], stepped=True)
        executable, found, compiled_from = replacement, _misses_of(spilling), program
    # The kept program's own spills are misses alone: refused, they would have every process seeded with
    # them derive another program, not the one this compile keeps and runs.
    _recorded(found, stepped=False)
    return executable


TRITON_TARGET = "__gpu$xla.gpu.triton"
"""The custom call target of a Pallas-Triton kernel, whose backend config names it."""
MOSAIC_CALL = MOSAIC_TARGET.decode()
"""The custom call target of a Mosaic GPU kernel (`MOSAIC_TARGET`), as a lowered module names it."""


@dataclasses.dataclass(frozen=True)
class Alone:
    """A kernel call compiled in a module of its own (`compiled_alone`): the binaries its executable carries
    (`binaries`' reading), or the error XLA raised of it."""
    binaries: tuple["Binary", ...] = ()
    error: Exception | None = None


def compiled_alone(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
                   wanted: Callable[["KernelCall"], bool]) -> dict[str, Alone] | None:
    """Each Pallas-Triton and Mosaic GPU kernel call of `lowered` that `wanted` takes, compiled alone, by its
    kernel's name: the call cloned into a module of its own, every attribute kept but the program's
    shardings (its operands' aliasing and its side effects among them), its operands the module's parameters
    and its results the module's, compiled by `lowered`'s own backend for one of its devices (a compile-only
    device's too) under the program's own compile options (its lowering's and `options`), so its binary is
    the one the program's compile makes: a kernel's code compiles apart from the program around it. None
    where a wanted call reads its program's devices (a Mosaic call over collective metadata or replicas),
    which no module of one device holds."""
    import jaxlib.xla_client as xla_client
    from jax._src import compiler
    from jax._src.interpreters import pxla
    from jax._src.lib.mlir.dialects import func

    from dew.nn.kernels.lowered import KernelCall, kernel_call

    # The compile's own backend, devices and options: a jit's lowering holds them (`pxla.MeshComputation`);
    # `Lowered` states none of them.
    lowering = lowered._lowering
    if not isinstance(lowering, pxla.MeshComputation) or lowering._device_list is None:
        raise TypeError(f"kernels are compiled alone from a jit's lowering over its devices, not "
                        f"{type(lowering).__name__}")
    backend, device = lowering.compile_args["backend"], next(iter(lowering._device_list))
    overrides = dict(lowering._compiler_options_kvs + tuple((options or {}).items()))
    module = lowering.stablehlo()
    operations: dict[KernelCall, ir.Operation] = {}

    def walk(operation) -> None:
        for region in operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    inner = inner.operation
                    target = (ir.StringAttr(inner.attributes["call_target_name"]).value
                              if inner.name == "stablehlo.custom_call" else None)
                    if target in (TRITON_TARGET, MOSAIC_CALL):
                        called = kernel_call(target, inner)
                        if called is not None and wanted(called):
                            operations.setdefault(called, inner)
                    walk(inner)

    walk(module.operation)
    # Each kernel once, by its name and launch, as its program's evidence names it (`kernel_bodies`).
    calls = {name: operations[call] for name, call in kernel_bodies(operations).items()}
    for call in calls.values():
        config = ir.DictAttr(call.attributes["mhlo.backend_config"])
        if "xla_replica_ids" in config or ("uses_xla_collective_metadata" in config
                                           and ir.BoolAttr(config["uses_xla_collective_metadata"]).value):
            return None
    compiled: dict[str, Alone] = {}
    for name, call in calls.items():
        with module.context, ir.Location.unknown():
            alone = ir.Module.create()
            alone.operation.attributes["sym_name"] = ir.StringAttr.get("alone")
            with ir.InsertionPoint(alone.body):
                kind = ir.FunctionType.get([value.type for value in call.operands],
                                           [value.type for value in call.results])
                entry = func.FuncOp("main", kind)
                entry.attributes["sym_visibility"] = ir.StringAttr.get("public")
                block = entry.add_entry_block()
                with ir.InsertionPoint(block):
                    clone = call.clone()
                    for index, argument in enumerate(block.arguments):
                        clone.operands[index] = argument
                    # A sharding names the program's mesh, which the module alone holds none of.
                    held = [clone.attributes[index].name for index in range(len(clone.attributes))]
                    for attribute in [name for name in held if name.startswith("sdy.")]:
                        del clone.attributes[attribute]
                    func.ReturnOp(clone.results)
        built = compiler.get_compile_options(num_replicas=1, num_partitions=1, device_assignment=(device,),
                                             env_options_overrides=overrides, backend=backend)
        try:
            executable = compiler.backend_compile_and_load(backend, alone, xla_client.DeviceList((device,)),
                                                           built, [])
        except jax.errors.JaxRuntimeError as error:
            compiled[name] = Alone(error=error)
            continue
        compiled[name] = Alone(tuple(_carried(executable.serialize())))
    return compiled


def refused_alone(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None = None
                  ) -> dict[str, int]:
    """The Pallas-Triton kernels of `lowered` XLA refuses for their shared memory each compiled alone
    (`compiled_alone`), by name, with the bytes each asked (`shared_asked`). A kernel alone that compiles,
    or that XLA refuses for another reason, is none of them; a Triton call reads no collective metadata, so
    none is left to its program's compile (`compiled_alone`'s None)."""
    alone = compiled_alone(lowered, options, lambda call: call.kind == "triton") or {}
    return {name: asked for name, each in alone.items()
            if each.error is not None and (asked := shared_asked(each.error)) is not None}


def _stepped_alone(relower: Callable[[], jax.stages.Lowered], options: jax.stages.CompilerOptions | None,
                   stepped: set[str]) -> tuple[jax.stages.Lowered, list]:
    """The program `relower`ed past the refusals in scope (`_proposing`'s), the launches `stepped` names (by
    identity) relowered again past each config whose kernel spills compiled alone (`compiled_alone`), until
    none of them spills alone; and those misses, each now refused in the scope (`_misses_of`'s form). A
    whole program's compile a step cost an sm100 step's step-down 18 s of its 30, eight of them spilling
    recurrence launches (row 5359), where a kernel alone costs its own compile. A stepped launch whose call
    cannot compile alone steps at the whole program's compile, as before; the whole compile after these is
    the net (`settled`)."""
    missed: list = []
    while True:
        program = relower()

        def wanted(call) -> bool:
            launch = called(call)
            return launch is not None and launch.identity in stepped

        alone = compiled_alone(program, options, wanted)
        if alone is None or any(each.error is not None for each in alone.values()):
            return program, missed
        held = _registry().refused
        found = [(refusal, report, launch)
                 for refusal, report, launch in _misses_of([binary for each in alone.values()
                                                            for binary in each.binaries if binary.refused])
                 if refusal is not None and report.kernel not in names(held.get(refusal, ()))]
        if not found:
            return program, missed
        for refusal, report, _ in found:
            _record(refusal, None, frozenset({(report.kernel, report.spilled_in)}))
        missed += found


def _held(found: list, program: jax.stages.Lowered) -> None:
    """Each config `found` refuses (a compile's misses of `program`'s kernels, `_misses_of`'s form: each
    refusal its launch's identity and config, the launch the one its call ran, `called`) stored as that
    launch's certification verdict (`_verdict_key`, in the mode it traced in), as `certified` stores its own,
    whatever other config a kernel of one body shares its name with (`named`): the program's compile is the
    authority, its kernels the bodies that run, so its refusal replaces an alone compile's acceptance, which
    no later process compiles again to write back (`certified` reads a stored verdict first). A later
    process's derivation refuses the config at trace, no program compiled and relowered to learn it again (LF-
    PROMO3: a warm start walked the 4B's whole step-down again, 375 of its 577 s)."""
    from dew.nn.kernels import tuning
    from dew.nn.kernels.lowered import kernel_calls

    device = _certifying_device()
    if device is None:
        return
    modules = kernel_modules(kernel_calls(program, unrun=True))
    verdicts: dict[tuple[str, Hashable, bool], dict[str, tuple[str, Binary]]] = {}
    for refusal, binary, launch in found:
        if refusal is None or launch is None:
            continue
        identity, config = refusal
        name = launch_name(binary.kernel, binary.launch)
        verdicts.setdefault((identity, config, launch.reverse), {})[name] = (modules.get(name, ""), binary)
    for (identity, config, reverse), kernels in verdicts.items():
        if (key := _verdict_key(identity, config, reverse)) is not None:
            tuning.store_verdict(verdict_target(device), key, kernels)


def _untaken(program: jax.stages.Lowered, refusals: list[tuple[tuple[str, Hashable], Refused]]
             ) -> list[tuple[tuple[str, Hashable], Refused]]:
    """Each of `refusals` (a launch's identity and config, and its spilling kernel, `Refused`) that `program`,
    relowered under them, launches again: its launch's derivation did not read the refusal. Each is logged,
    so the launch whose rule ignores its refusals is named."""
    from dew.nn.kernels.lowered import kernel_calls

    # Every call JAX compiles, a branch's no run takes among them (a forward's scaled op's).
    launched = {(launch.identity, launch.config) for call in kernel_calls(program, unrun=True)
                if (launch := called(call)) is not None}
    untaken = [(refusal, kernel) for refusal, kernel in refusals if refusal in launched]
    for (_, config), (kernel, _) in untaken:
        _log.warning("kernel %s's refusal at %s did not take: the program relowered under it launches it "
                     "again, its launch's derivation not reading it", kernel, config)
    return untaken


def _unblocked(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
               relower: Callable[[], jax.stages.Lowered], asked: int,
               error: Exception) -> jax.stages.Compiled:
    """The program XLA refused for a Triton kernel asking `asked` shared bytes past a block, compiled once
    more with the configs of the Triton launches XLA refuses each compiled alone refused (`refused_alone`,
    each its next derived config, as a spill refuses one, `kept`), or every Triton launch's where none is
    refused alone: XLA's refusal names no kernel, and a search compiling the whole program a launch at a
    time costs a program's compile a launch (row 3584), where a kernel's call compiled alone costs its own
    (audit group 71). Each launch's miss is recorded and logged at the bytes it asked; where the relowered
    program is refused again, or a derivation finds no config left (`Exhausted`), the refusal stands, each
    a miss not stepped."""
    from dew.nn.kernels.lowered import kernel_calls

    tried: dict[tuple[str, Hashable], tuple[str, Launch]] = {}
    alone = refused_alone(lowered, options)
    # Every Triton kernel XLA compiles, a branch no run reaches too (a guarded forward's rewalk).
    for call in kernel_calls(lowered, unrun=True):
        kernel = kernel_of(call.name)[0]
        launch = called(call) if call.kind == "triton" else None
        if launch is not None and (not alone or call.name in alone):
            tried.setdefault((launch.identity, launch.config), (kernel, launch))
    asked_by = {kernel_of(name)[0]: bytes_asked for name, bytes_asked in alone.items()}
    try:
        with _proposing([(refusal, (kernel, None)) for refusal, (kernel, _) in tried.items()]) as proposal:
            replacement, spilling = _compiled_reported(relower(), options)
    except (Exhausted, jax.errors.JaxRuntimeError) as again:
        if isinstance(again, jax.errors.JaxRuntimeError) and shared_asked(again) is None:
            raise
        for kernel, launch in tried.values():
            _missed_block(kernel, launch, asked_by.get(kernel, asked), stepped=False)
        raise error from again
    _commit(proposal)
    for refusal, (kernel, launch) in tried.items():
        _record(refusal, None, frozenset({(kernel, None)}))
        _missed_block(kernel, launch, asked_by.get(kernel, asked), stepped=True)
    _recorded(_misses_of(spilling), stepped=False)
    return replacement


def _missed_block(kernel: str, launch: Launch, asked: int, *, stepped: bool) -> None:
    """A Triton launch's shared-memory miss recorded and logged (`_unblocked`), under its first kernel's
    name: XLA's refusal names none."""
    if _record(None, Miss(kernel, launch.op, launch.config, None, None, stepped, shared=asked)):
        _log.warning("a program asks %d shared bytes past its block; Triton kernel %s (%s at %s): %s", asked,
                     kernel, launch.op, launch.config, "its launch steps down" if stepped else "no step-down")


def _misses_of(refused: list["Binary"]
               ) -> list[tuple[tuple[str, Hashable] | None, "Binary", Launch | None]]:
    """Each refused kernel's refusal (its launch's identity and config, None for a kernel no launch of this
    process traced), its report and its launch."""
    found = []
    for report in refused:
        launch = called(report)
        found.append((None if launch is None else (launch.identity, launch.config), report, launch))
    return found


def _recorded(found: list, *, stepped: bool) -> None:
    """Each of `found`'s misses recorded and logged, stepped down where `stepped` and its kernel a
    launch's."""
    for _, report, launch in found:
        stated = None if report.launch is None else traced(report.launch)
        model = None if stated is None else stated.kernels.get(report.kernel)
        miss = Miss(report.kernel, None if launch is None else launch.op,
                    None if launch is None else launch.config, report.registers, report.frame,
                    stepped and launch is not None,
                    serialized=tuple(note.code for note in report.notes if note.code in SERIALIZED),
                    modelled=None if model is None else model.registers)
        if _record(None, miss):
            _log.warning("kernel %s %s at %s registers against its model's %s (%s at %s); %s", miss.kernel,
                         miss.reason, miss.registers, miss.modelled, miss.op, miss.config,
                         "its launch steps down" if miss.stepped else "no step-down")


def _compiled_reported(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
                       read: Callable[[jax.stages.Compiled], list["Binary"]] | None = None
                       ) -> tuple[jax.stages.Compiled, list["Binary"]]:
    executable = xla_compiled(lowered, options)
    return executable, _refused_binaries(lowered, executable, read)


def _refused_binaries(lowered: jax.stages.Lowered, executable: jax.stages.Compiled,
                      read: Callable[[jax.stages.Compiled], list["Binary"]] | None = None) -> list["Binary"]:
    """The kernels of `executable`, `lowered` compiled, its assembler refuses (`Binary.refused`), read by
    `read` (`binaries` where None), where it runs a template's (`reads_binaries`)."""
    if not reads_binaries(lowered):
        return []
    return [found for found in (read or binaries)(executable) if found.refused]


@dataclasses.dataclass(frozen=True)
class AssemblerNote:
    """A note ptxas made of a loss of performance in a kernel's assembly, as it states it: its code (`C7512`,
    a warpgroup MMA serialized for want of registers) and its message."""
    code: str
    message: str


def refuses(frame: int, codes: Collection[str]) -> bool:
    """Whether a kernel the assembler built with a `frame`-byte stack frame, noting `codes`
    (`ASSEMBLER_NOTES`' codes), is refused for its config: it spills, or its warpgroup MMAs were serialized
    (`SERIALIZED`). Certification's verdict (`Binary.refused`) and the compile gate's alike."""
    return bool(frame) or bool(SERIALIZED & set(codes))

@dataclasses.dataclass(frozen=True)
class Binary:
    """A Mosaic GPU or Pallas-Triton kernel's binary as its executable carries it (`binaries`): its name less
    the launch it carries (`LAUNCH_SUFFIX`) and that launch's digest (`named`), its registers a thread and its
    stack frame's bytes a thread, nonzero where its registers spill (`REGISTER_COUNT`, `FRAME_SIZE`), the
    dynamic shared memory its launch asks for (a Mosaic kernel proto's `smem_bytes`, a custom kernel's
    `shared_memory_bytes`), None where its proto states none, and, read `decoded`, what its machine code
    breaks of the SASS rules (`sass_findings_of`), None where no disassembler read it."""
    kernel: str
    launch: str | None
    registers: int
    frame: int
    kind: str
    shared: int | None = None
    sass: tuple[str, ...] | None = ()
    caller: str | None = None
    """The launch that traced it, as its call's caller stated it (`CALLER_METADATA`, a Mosaic GPU kernel's);
    None where its call stated none."""
    notes: tuple["AssemblerNote", ...] = ()
    """What ptxas noted of a loss of performance as it assembled it, in its order (`ASSEMBLER_NOTES`, a
    Mosaic GPU kernel's); none where its call states none (a Pallas-Triton kernel's, a build without them)."""
    spilled_in: tuple[str, ...] | None = ()
    """The roles whose code its spill stands in where it spills (`spilled_roles`: `PRODUCER`, `CONSUMERS`),
    read `decoded` or for its `spills` (`binaries`, certification's), none where it spills none; None where
    it was not read, or no disassembler listed it (`disassembler`)."""
    threads: int | None = None
    """Its threads a block, as its `.nv.info` states them (`MAX_THREADS`, its PTX's `.maxntid`); None where
    it states none."""
    decreased: int | None = None
    raised: int | None = None
    """The registers its producer lowers to and its consumers raise to, as its machine code reconfigures
    them (`reconfigured`: USETMAXREG's deallocation and allocation, its PTX's `setmaxnreg.dec` and `.inc`);
    None where it reconfigures none that way, or no disassembler read it."""
    uses_tmem: bool | None = None
    """Whether its decoded SASS issues Blackwell tensor-memory ops (`_uses_tmem`); None undecoded."""
    index: str | None = None
    """XLA uniquification suffix (`__1`) where a second cubin shares `(kernel, launch)`; None otherwise."""

    @property
    def launched(self) -> str | None:
        """The digest of the launch that traced it: its caller's, else its name's, which is its launch's where
        the launch states no kernel's part (`named`)."""
        return self.caller or self.launch

    @property
    def refused(self) -> bool:
        """Whether its config is refused for it (`refuses`)."""
        return refuses(self.frame, [note.code for note in self.notes])


def binaries(executable: jax.stages.Compiled, *, decoded: bool = False, spills: bool = False) -> list[Binary]:
    """Every Mosaic GPU and Pallas-Triton kernel binary `executable` carries, read from its own serialized
    form: XLA keeps a Mosaic kernel's cubin in its custom call's state (`MosaicGpuKernelProto`), whether this
    compile ran ptxas for it, another thread's program compiled it first (Mosaic compiles a module once a
    process), or the executable was loaded from the persistent cache, and a Pallas-Triton template
    launch's in its custom kernel's (`custom_kernel`), its kernel naming its launch: the thunk's own
    `binary` is its module's, which in a program holding a constant is that constant's module alone, no
    kernel's code in it. Nothing is read from or written to the process's environment or disk: a compile
    that set Mosaic's dump settings around itself handed its directory to every other thread's compile and
    removed it under them ('Directory not empty', row 2790), and the settings turn on every dump Mosaic has
    (row 2816). `decoded`, each binary's SASS findings too (`sass_findings_of`, the compile gate's), each
    cubin disassembled from a file of its own; `spills`, each spilling binary's roles its spill stands in
    (`Binary.spilled_in`, certification's: its ladder steps by them), its cubin disassembled alike."""
    runtime = executable.runtime_executable()
    if runtime is None:
        raise ValueError("a compiled program with no runtime executable carries no kernel binaries to read")
    return _carried(runtime.serialize(), decoded, spills)


def _carried(serialized: bytes, decoded: bool = False, spills: bool = False) -> list[Binary]:
    """`binaries`' reading of an executable as XLA serializes it (a program's, or a kernel's compiled alone,
    `compiled_alone`)."""
    found: list[Binary] = []
    for name, proto in _thunk_protos(serialized):
        found.extend(thunk_kernels(proto, decoded=decoded, spills=spills))
        # XLA's own kernels (its fusions, its Triton GEMMs) carry their cubins too: a template launch's
        # names its launch (`named`).
        kernel = custom_kernel(proto) if LAUNCH_SUFFIX.search(name) else None
        if kernel is not None:
            elf, shared = kernel
            found.append(dataclasses.replace(_binary(name, elf, decoded, spills, kind="triton"),
                                             shared=shared))
    return found


def custom_kernels(executable: jax.stages.Compiled) -> list[tuple[str, bytes, int]]:
    """Every custom kernel `executable` launches, by the name its thunk gives it, with its cubin and the
    shared bytes its launch asks for (`custom_kernel`): a Pallas-Triton call's whatever its name (`binaries`
    keeps a template launch's), and XLA's own custom kernels, read from its serialized form as `binaries`."""
    runtime = executable.runtime_executable()
    if runtime is None:
        raise ValueError("a compiled program with no runtime executable launches no kernel to read")
    return [(name, *kernel) for name, proto in _thunk_protos(runtime.serialize())
            if (kernel := custom_kernel(proto)) is not None]


def launched_thunks(executable: jax.stages.Compiled) -> list[str]:
    """Every thunk `executable` launches, in order, by its kernel's name (`_thunk_protos`): XLA's own (a
    fusion of one operation's `wrapped_<operation>`) beside the custom kernels, a run's every launch."""
    runtime = executable.runtime_executable()
    if runtime is None:
        raise ValueError("a compiled program with no runtime executable launches no kernel to read")
    return [name for name, _ in _thunk_protos(runtime.serialize())]


def launched_kernels(executable: jax.stages.Compiled) -> list[tuple[str, int]]:
    """Every custom kernel `executable` launches (`custom_kernels`), by its name, with the shared bytes its
    launch asks for."""
    return [(name, shared) for name, _, shared in custom_kernels(executable)]


def _thunk_protos(serialized: bytes) -> list[tuple[str, bytes]]:
    """Each thunk an executable XLA serialized as `serialized` calls (its thunk sequence, read by thunky's
    reader of the serialized form, which its public `jax_executable_to_mlir` reads from a
    `jax.stages.Compiled`, where a kernel's compiled alone is a backend's `LoadedExecutable`), by its
    kernel's name, with its serialized proto."""
    # thunky is imported with this module, not at a first reading: its import registers a discharge rule
    # (`call_thunky_nested`) the Pallas lowering cache keys every lowering after it by, so a process that
    # certified a launch before its step lowered keyed the step's lowerings apart from a warm start's, which
    # reads none (FS-CERTREG: a warm start's first process lowered 26 kernels afresh).
    found: list[tuple[str, bytes]] = []
    thunky._check_thunky_available()
    context = thunky.make_ir_context()
    with context:
        module = thunky._thunky.jax_executable_to_mlir(serialized, context)

        def visit(operation: ir.Operation) -> ir.WalkResult:
            if operation.name == "thunky.call_thunk_proto":
                attributes = operation.attributes
                found.append((ir.StringAttr(attributes["kernel_name"]).value,
                              ir.StringAttr(attributes["thunk_proto"]).value_bytes))
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)
    return found


def thunk_kernels(proto: bytes, *, decoded: bool = False, spills: bool = False) -> list[Binary]:
    """The kernel binaries of a thunk's serialized proto (`_kernels`), one for each Mosaic GPU call it names:
    a call whose binary is missing would read as a kernel that does not spill, so it raises."""
    kernels, calls = _kernels(proto, decoded, spills)
    if len(kernels) != calls:
        raise ValueError(f"a thunk names {calls} Mosaic GPU calls but carries {len(kernels)} kernel binaries")
    return kernels


def custom_kernel(proto: bytes) -> tuple[bytes, int] | None:
    """The cubin of the custom kernel a thunk's serialized proto launches and the shared bytes its launch
    asks for (`CUSTOM_KERNEL_THUNK`), None where the thunk launches no custom kernel's cubin."""
    kernels = [kernel for thunk in proto_fields(proto).get(CUSTOM_KERNEL_THUNK, [])
               for kernel in proto_fields(thunk).get(3, [])]
    if len(kernels) != 1:
        return None
    kernel = proto_fields(kernels[0])
    cubins = [elf for spec in kernel.get(2, []) for cubin in proto_fields(spec).get(2, [])
              for elf in proto_fields(cubin).get(1, [])]
    if len(cubins) != 1 or cubins[0][:4] != ELF:
        return None
    # proto3 omits a zero.
    return cubins[0], kernel.get(6, [0])[0]


def proto_fields(message: bytes) -> dict[int, list]:
    """A serialized proto's fields by number, each its values (a varint's int, a length-delimited field's
    bytes), read without its schema; none where `message` parses as no proto. Dew's one reader of XLA's
    serialized protos: a thunk's (`custom_kernel`, `_kernels`) and a GPU topology's
    (`dew.training.ahead.topology_memory`)."""
    parsed = empty_pb2.Empty()
    try:
        parsed.ParseFromString(message)
    except DecodeError:
        return {}
    fields: dict[int, list] = {}
    for field in UnknownFieldSet(parsed):
        fields.setdefault(field.field_number, []).append(field.data)
    return fields


def _kernels(message: bytes, decoded: bool = False, spills: bool = False) -> tuple[list[Binary], int]:
    """The kernels of each `MosaicGpuKernelProto` within a thunk's serialized proto (a message of jaxlib's
    version, field 1, whose binary, 8, is an ELF, whose kernel name, 9, its function's, whose
    `SHARED_BYTES` its launch's dynamic shared memory, whose `CALLER_METADATA` the launch its call carried
    (`launch_metadata`), the launch its name states where it carries none, and whose `ASSEMBLER_NOTES` what
    ptxas noted of a loss of performance), and how many Mosaic GPU custom calls it names
    (`MOSAIC_TARGET`, a custom call thunk's target). A message holding neither the target's bytes nor an ELF's
    holds neither, and is not parsed: most of a program's thunks and their messages (a 4B decoder's step read
    223,561 messages, 1.8 of its compile point's 3.5 seconds)."""
    if MOSAIC_TARGET not in message and ELF not in message:
        return [], 0
    fields = proto_fields(message)
    binary, name, shared = fields.get(8, []), fields.get(9, []), fields.get(SHARED_BYTES, [])
    if (fields.get(1) == [KERNEL_PROTO_VERSION] and len(binary) == len(name) == 1
            and isinstance(binary[0], bytes) and binary[0][:4] == ELF and isinstance(name[0], bytes)):
        stated = shared[0] if len(shared) == 1 and isinstance(shared[0], int) else None
        caller = fields.get(CALLER_METADATA, [])
        stated_caller = caller[0].decode() if len(caller) == 1 and isinstance(caller[0], bytes) else None
        notes = tuple(AssemblerNote(*(part[0].decode() for part in (note.get(1, [b""]), note.get(2, [b""]))))
                      for note in map(proto_fields, fields.get(ASSEMBLER_NOTES, [])))
        return [dataclasses.replace(_binary(name[0].decode(), binary[0], decoded, spills, kind="mosaic_gpu"),
                                    shared=stated,
                                    caller=stated_caller, notes=notes)], 0
    found, calls = [], 0
    for values in fields.values():
        for value in values:
            if value == MOSAIC_TARGET:
                calls += 1
            elif isinstance(value, bytes):
                kernels, named = _kernels(value, decoded, spills)
                found += kernels
                calls += named
    return found, calls


def _binary(function: str, elf: bytes, decoded: bool = False, spills: bool = False, *,
            kind: str = "mosaic_gpu") -> Binary:
    """The `Binary` of the kernel `function` from its cubin `elf`: its `.nv.info` attributes' registers,
    frame and threads (`REGISTER_COUNT`, `FRAME_SIZE`, `MAX_THREADS`; a Mosaic module compiles one kernel
    function), `decoded` its SASS findings (`sass_findings_of`), and its register reconfiguration
    (`reconfigured`) and, where it spills, the roles its spill stands in (`spilled_roles`) wherever its one
    listing is read (`sass_listing`: `decoded`, or `spills` where it spills)."""
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
                elif attribute == MAX_THREADS:
                    # A function's own section's: its block's three extents, no symbol before them.
                    stated[attribute] = math.prod(struct.unpack_from("<III", elf, at + 4))
                at += 4 + length
            else:
                at += 4
    if REGISTER_COUNT not in stated:
        raise ValueError(f"the binary of {function} states no registers")
    frame = stated.get(FRAME_SIZE, 0)
    listing = sass_listing(elf) if decoded or (spills and frame) else None
    decreased, raised = (None, None) if listing is None else reconfigured(listing)
    sass = (None if listing is None else sass_findings_of(listing)) if decoded else ()
    uses_tmem = None if listing is None else _uses_tmem(listing)
    kernel, launch, index = kernel_of(function)
    return Binary(kernel, launch, stated[REGISTER_COUNT], frame, kind=kind, sass=sass,
                  spilled_in=(None if listing is None else spilled_roles(listing)) if frame else (),
                  threads=stated.get(MAX_THREADS), decreased=decreased, raised=raised,
                  uses_tmem=uses_tmem, index=index)


def disassembler() -> Path | None:
    """NVIDIA's nvdisasm, which the SASS rules read a kernel's machine code through (`sass_listing`): the
    nvdisasm wheel's (`nvidia/cu13/bin`), which the `ci` extra installs in every lock the suite runs on, or
    the PATH's; None where neither is installed, a user's install among them."""
    spec = importlib.util.find_spec("nvidia")
    for root in (spec.submodule_search_locations or ()) if spec else ():
        if (found := Path(root) / "cu13" / "bin" / "nvdisasm").is_file():
            return found
    found = shutil.which("nvdisasm")
    return None if found is None else Path(found)


def sass_listing(elf: bytes) -> str | None:
    """nvdisasm's listing of the machine code of the cubin `elf`; None where no disassembler is installed
    (`disassembler`)."""
    tool = disassembler()
    if tool is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin:
        cubin.write(elf)
        cubin.flush()
        listing = subprocess.run([str(tool), "-c", cubin.name], capture_output=True, text=True, check=True)
    return listing.stdout


PRODUCER, CONSUMERS = "producer", "consumers"
"""A warpgroup-specialized kernel's roles, as its SASS places them (`spilled_roles`)."""


def spilled_roles(listing: str) -> tuple[str, ...]:
    """The roles whose code a kernel's spill stands in, from nvdisasm's `listing`: each local-memory store
    and load (STL, LDL) the role of the last setmaxnreg before it, ptxas laying each role's branch out after
    its own, the producer's drop (`USETMAXREG.DEALLOC`) or the consumers' raise (`USETMAXREG.TRY_ALLOC`); one
    before either, a prologue's, neither role's (row 6744: a B200's decayed keys spilled after its producer's
    drop to the least, an H100's output walk between the consumers' raise and the producer's drop)."""
    roles, role = set(), None
    for line in listing.splitlines():
        if not (found := _INSTRUCTION.search(line)):
            continue
        opcode = _parsed(found.group(2))[0]
        if opcode.startswith("USETMAXREG"):
            role = PRODUCER if ".DEALLOC" in opcode else CONSUMERS
        elif role is not None and opcode.split(".")[0] in ("STL", "LDL"):
            roles.add(role)
    return tuple(sorted(roles))


_RECONFIGURED = re.compile(r"\bUSETMAXREG\.(DEALLOC|ALLOC|TRY_ALLOC)\S*\s+(?:UP\d+,\s*)?(0x[0-9a-f]+|\d+)")


def _uses_tmem(listing: str) -> bool:
    """Whether nvdisasm's `listing` issues Blackwell tensor-memory ops (`LDTM`, `STTM`, `UTC*`)."""
    return any(_parsed(found.group(2))[0].split(".")[0].startswith(("LDTM", "STTM", "UTC"))
               for line in listing.splitlines() if (found := _INSTRUCTION.search(line)))


def reconfigured(listing: str) -> tuple[int | None, int | None]:
    """The registers a kernel's machine code (nvdisasm's `listing`) lowers its producer to and raises its
    consumers to: its first USETMAXREG deallocation's and allocation's (a try's too) counts, as its PTX's
    `setmaxnreg.dec` and `.inc` state them; None for one it makes none of."""
    found: dict[str, int] = {}
    for kind, count in _RECONFIGURED.findall(listing):
        found.setdefault("dec" if kind == "DEALLOC" else "inc", int(count, 0))
    return found.get("dec"), found.get("inc")


_INSTRUCTION = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
WARPGROUP_MMAS = frozenset({"HGMMA", "QGMMA", "IGMMA"})
"""Hopper's warpgroup MMA opcodes, by their operands: 16-bit floats, 8-bit floats (fp8) and integers."""


def _dead_mma(instruction: str) -> bool:
    """Whether a SASS instruction is a warpgroup MMA writing no register (its destination RZ): what ptxas
    leaves of a kernel's MMAs whose results it found dead, the products never read (row 4346: ptxas 12.9 folds
    an integer op on an accumulator through its value before the MMA, so a negated product read stores
    -0.0). Read from the instruction's opcode and first operand, a predicate guard aside."""
    words = instruction.split()
    if words and words[0].startswith("@"):
        words = words[1:]
    return len(words) > 1 and words[0].split(".")[0] in WARPGROUP_MMAS and words[1].rstrip(",") == "RZ"
_LABEL = re.compile(r"^\s*(\.L_\w+):")
_WATERFALL = re.compile(r"\bBRA\.U\.ANY\s+`?\(?(\.L_\w+)\)?`?")
_BRANCH = re.compile(r"\bBRA(?:\.\S+)?\s+`?\(?(\.L_\w+)\)?`?")
_REGISTER = re.compile(r"R(\d+)")
REGISTER_A = 4
"""The registers a thread holds a warpgroup MMA's register operand A in (its 64-row fragment's: eight 16-bit
values, or sixteen 8-bit)."""


def _parsed(instruction: str) -> tuple[str, list[str]]:
    """A SASS instruction's opcode and its operands, its predicate (`@P0`) apart."""
    words = instruction.split()
    if words and words[0].startswith("@"):
        words = words[1:]
    return (words[0] if words else ""), " ".join(words[1:]).split(", ")


def _written(opcode: str, operands: list[str]) -> set[int]:
    """The registers a SASS instruction writes: its destination, the first operand, and the ones after it
    that its width takes (a 128-bit load's or a four-matrix `LDSM`'s four, a 64-bit one's, a two-matrix
    `LDSM`'s or a `.WIDE` product's two)."""
    if not (target := _REGISTER.fullmatch(operands[0])):
        return set()
    matrices = opcode.startswith("LDSM")
    pair = ".64" in opcode or ".WIDE" in opcode or opcode.startswith("CS2R") or (matrices and ".2" in opcode)
    count = 4 if ".128" in opcode or (matrices and ".4" in opcode) else 2 if pair else 1
    return set(range(int(target.group(1)), int(target.group(1)) + count))


def _predicated(instruction: str) -> bool:
    """Whether a SASS instruction runs under a predicate other than the always-true one (`@P0`, `@!P1`;
    not `@PT`)."""
    first = instruction.split(maxsplit=1)[0] if instruction.strip() else ""
    return first.startswith("@") and first.lstrip("@!") != "PT"


def _reused_register_a(instructions: list[str], labels: dict[str, int]) -> int:
    """The writes in a loop (a backward branch's body, walked over two trips) to the registers of a warpgroup
    MMA's register operand A whose group is still pending: PTX leaves an operand A read until a
    `wgmma.wait_group` retires its group (PTX ISA, "wgmma.mma_async"), so a write before it is undefined
    behaviour, the product reading whichever value its warp reached (row 5416: a prologue's next step
    rewrote A under `wait_group 1`, its forward differing run to run on a GH200). A predicated wait
    (`@P0 WARPGROUP.DEPBAR.LE`) retires nothing on the walk, which takes no predicate as known."""
    reused = 0
    for at, text in enumerate(instructions):
        head = labels.get(branch.group(1)) if (branch := _BRANCH.search(text)) else None
        if head is None or head > at:
            continue
        body = [(*_parsed(instruction), instruction) for instruction in instructions[head:at + 1]]
        if not any(opcode.split(".")[0] in WARPGROUP_MMAS for opcode, _, _ in body):
            continue
        committed: list[set[int]] = []
        issuing: set[int] = set()
        for _ in range(2):
            for opcode, operands, instruction in body:
                if opcode.split(".")[0] in WARPGROUP_MMAS:
                    if len(operands) > 1 and (left := _REGISTER.fullmatch(operands[1])):
                        issuing |= set(range(int(left.group(1)), int(left.group(1)) + REGISTER_A))
                    if "gsb0" in instruction:
                        committed, issuing = [*committed, issuing], set()
                elif opcode == "WARPGROUP.DEPBAR.LE" and not _predicated(instruction):
                    pending = int(operands[-1], 0)
                    committed = committed[len(committed) - pending:] if pending else []
                else:
                    reused += bool(_written(opcode, operands) & set().union(issuing, *committed))
    return reused


def sass_findings_of(listing: str) -> tuple[str, ...]:
    """What nvdisasm's `listing` of a kernel breaks of the kernel challenge's SASS rules (row 4071, its
    harness/ptxas checks; notes/ptxas-re.md §2): an elect waterfall (a backward BRA.U.ANY) holding an R2UR, a
    copy issued lane by lane because ptxas could not prove its operand uniform; a warpgroup MMA writing
    no register (`_dead_mma`), a product ptxas dropped as never read (row 4346); and a loop writing a
    pending warpgroup MMA's register operand A (`_reused_register_a`, row 5416)."""
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
    dead = sum(1 for text in instructions if _dead_mma(text))
    reused = _reused_register_a(instructions, labels)
    found = []
    if serialized:
        found.append(f"{serialized} elect waterfall holding an R2UR, a copy issued lane by lane")
    if dead:
        found.append(f"{dead} warpgroup MMA writing no register, its products dropped as never read")
    if reused:
        found.append(f"{reused} write to a warpgroup MMA's register operand A before its group retired")
    return tuple(found)
