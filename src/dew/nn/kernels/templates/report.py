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
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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

from dew import _cache, communicators
from dew._devices import deterministic_ops_requested, deviceless, xla_flag
from dew.nn.kernels import generation

if TYPE_CHECKING:
    from jax.extend import core as jax_core

    from dew.nn.kernels.lowered import KernelCall
    from dew.nn.kernels.templates.resources import RegisterModel

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
"""The suffix a kernel's name carries within the launch that traced it (`named`): the digest of its identity
there, its launch's digest where the launch states no kernel's part (`tuning.KernelOp.emitted`)."""
KERNEL_FUNCTION = re.compile(r"(?P<kernel>.*)_mosaic_gpu_kernel")
"""A Mosaic GPU kernel's function in its binary: its kernel's name, the host function's `_mosaic_gpu`
(jax/experimental/mosaic/gpu/core.py) and GPU outlining's `_kernel`."""


def kernel_of(name: str) -> tuple[str, str | None]:
    """`(kernel, identity)` of a kernel's `name` as it traced (`named`), its function in its binary or its PTX
    entry (`KERNEL_FUNCTION`) or its call's location: its name less its suffix, and the suffix's digest
    (`LAUNCH_SUFFIX`: its launch's, where its launch states no kernel's part; a Mosaic GPU call carries its
    launch beside, `launch_metadata`), None where it carries none."""
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
    which reads the launch tracing it only as it names its kernels and carries the launch on their calls,
    each replayed as it is bound (`renamed`, `carrying`)."""
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
@dataclasses.dataclass
class _Registry:
    """Each (launch identity, config) a compile refused (`compiled`: its kernel spilled), by the names of its
    kernels that spilled, and each miss."""
    refused: dict[tuple[str, Hashable], frozenset[str]] = dataclasses.field(default_factory=dict)
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


def carrying(closed: "jax_core.Jaxpr", metadata: bytes) -> "jax_core.Jaxpr":
    """`closed` with each Mosaic GPU call in it carrying `metadata` (`launch_metadata`), the launch it is
    bound for: a held trace (`tuning.lowered_once`) bound for another launch than the one it was traced
    for, whose calls' compiled serializations then name that launch (`Binary.caller`)."""
    from jax._src.pallas.mosaic_gpu import core as plgpu_core

    from dew.nn import programs

    def carried_jaxpr(jaxpr: "jax_core.Jaxpr") -> "jax_core.Jaxpr":
        # Its constants with it (`ClosedJaxpr` is `Jaxpr`); itself where no equation's parameters change: a
        # kernel's body, unchanged, keeps its identity, by which its lowering's cache key is kept.
        equations = [carried_equation(equation) for equation in jaxpr.eqns]
        if all(new is old for new, old in zip(equations, jaxpr.eqns, strict=True)):
            return jaxpr
        return jaxpr.replace(eqns=equations)

    def carried_equation(equation: "jax_core.JaxprEqn") -> "jax_core.JaxprEqn":
        params = {key: carried(value) for key, value in
                  programs.with_bodies(equation.params, carried_jaxpr).items()}
        if all(params[key] is equation.params[key] for key in params):
            return equation
        return equation.replace(params=params)

    def carried(value: Hashable | Mapping) -> Hashable | Mapping:
        # A Mosaic GPU call's parameters carrying the launch, where they stand alone or by their backend.
        if isinstance(value, plgpu_core.CompilerParams):
            return dataclasses.replace(value, metadata=metadata)
        if isinstance(value, Mapping) and any(isinstance(each, plgpu_core.CompilerParams)
                                              for each in value.values()):
            return {key: carried(each) for key, each in value.items()}
        return value

    return carried_jaxpr(closed)


def launch_metadata() -> bytes:
    """The launch tracing (`launching`) as its kernels' calls carry it beside their names (`named`): its
    digest, a Mosaic GPU call's caller metadata (`plgpu.CompilerParams.metadata`, kept in its compiled call's
    serialization, `Binary.caller`; not its module's, so no lowering or compile of the kernel reads it);
    none where no launch traces."""
    digest = launch_digest()
    return b"" if digest is None else digest.encode()


def launch_name(kernel: str, launch: str | None) -> str:
    """`kernel`'s name carrying the digest `launch` (`LAUNCH_SUFFIX`), as is where None: `kernel_of`'s
    inverse, the key a kernel's evidence goes by beside its launch."""
    return kernel if launch is None else f"{kernel}__l{launch}"


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
        # A kernel's identity in its default form (`named`): what its emission reads the whole config, its
        # form none; a gradient's product's digest apart from its call's own, as its verdict is (`_held`).
        mode = "|reverse" if pending.reverse else ""
        digest = _digest(f"{identity}|{pending.config!r}|{()!r}{mode}")
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
                      ) -> tuple[str, Hashable, frozenset[tuple[Hashable, frozenset[str]]], bool]:
    """What a launch's derivation of its config for `arguments` on `target` reads of the launch and its
    compiles: the launch's identity (`_identity`: the op, its arguments' shapes and dtypes, its hooks), the
    whole target, whose facts price its schedules, each config this context's compiles refused it
    (`refused`), so a refusal recorded after a derivation derives it again, and whether the run asked XLA for
    deterministic ops, which a launch's legality reads (a fused attention backward's atomic dQ sums,
    `attention_mosaic.MosaicBackward`), so a flag flipped within a process derives it again."""
    identity = _identity(op, arguments)
    return identity, target, refusals_of(identity), deterministic_ops_requested()


@generation.traced_fact
def refusals_of(identity: str) -> frozenset[tuple[Hashable, frozenset[str]]]:
    """Each config this context's compiles refused the launch of `identity` (`_identity`), with the kernels
    refused at it: what a launch's derivation and legality read of the refusals, the one way a trace reads
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
    return dict(refusals_of(_identity(op, arguments))).get(config, frozenset())


def refuse_launch(op, arguments: tokamax.BoundArguments, config: Hashable, kernels) -> None:
    """`op`'s launch of `arguments` at `config` refused in this context as a compile that found `kernels`
    spilling refuses it (`settled`), so its derivation takes the next config: a compile gate's, which
    compiles a launch apart from `compiled`, stepping as a run's compile steps (row 4727)."""
    _record((_identity(op, arguments), config), None, frozenset(kernels))


_CERTIFYING: contextvars.ContextVar = contextvars.ContextVar("certifying_device", default=None)
"""The device a launch's derived config is certified on (`certified`) where a block names one
(`certifying`)."""


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
    round: int
    """The round of certification it was compiled or read in (`certified_together`): configs certified at once
    share one."""
    device: Hashable
    """The GPU it was certified on (`_certifying_device`, a `jax.Device`), whose assembler's answer it is."""

    @property
    def refused(self) -> tuple["Binary", ...]:
        """Its kernels the GPU's assembler refuses (`Binary.refused`)."""
        return tuple(binary for _, binary in self.kernels.values() if binary.refused)


_ROUNDS = itertools.count()
"""Each round of certification's number (`Certification.round`)."""

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


def certified(op, arguments: tokamax.BoundArguments, config: Hashable, *, round: int | None = None
              ) -> tuple["Binary", ...] | None:
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
    lowering would. Once a process, by launch (`certifications`), and once a GPU model a
    machine: each verdict, an accepted config's and a refused one's, is stored with the kernel picks
    (`_verdict_key`), so a warm start traces and compiles nothing to certify."""
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
    verdict = None if stored is None else tuning.stored_verdict(device.device_kind, stored)
    if isinstance(verdict, dict):
        kernels = verdict
    else:
        try:
            kernels = _compiled_launch(op, arguments, config, device, reverse)
        except jax.errors.JaxRuntimeError as error:
            # XLA refuses the launch alone (a collective's devices): its program's compile reads it as it did
            # before certification (`settled`), and the gate names it (`uncertified`).
            _log.warning("launch %s at %s is not certified: XLA does not compile it alone (%s)",
                         type(op).__name__, config, str(error)[:200])
            with _GUARD:
                _UNCERTIFIED[key] = str(error)[:400]
            return None
        if stored is not None:
            tuning.store_verdict(device.device_kind, stored, kernels)
    found = Certification(type(op).__name__, config, reverse, kernels, verdict is not None,
                          next(_ROUNDS) if round is None else round, device)
    with _GUARD:
        _CERTIFIED[key] = found
    return found.refused


def certified_together(op, arguments: tokamax.BoundArguments, configs: Sequence[Hashable]
                       ) -> list[tuple["Binary", ...] | None]:
    """Each of `configs` certified (`certified`) in one round, all at once on as many threads, each tracing
    under this thread's states (`dew._cache.carried`: the assumed GPU, the kernel choices, JAX's per-thread
    settings) in a copy of this context, on the device this thread certifies on: a search's rungs, which the
    assembler answers only one yes or no each (`rung_round` bounds a round by the host's cores)."""
    from jax._src import mesh as mesh_lib

    round = next(_ROUNDS)
    # The device this thread certifies on and the meshes its program is set on and traced under (its axes
    # manual within a map), named to each thread: JAX holds both per thread (`jax.set_mesh`,
    # `jax.sharding.use_abstract_mesh`), which no other thread sees (row 6791; train 909's review F4), and a
    # launch certified off them lowers its kernels as its program does not (`_certifying_placement`).
    device = _certifying_device()
    placed, abstract = mesh_lib.get_concrete_mesh(), jax.sharding.get_abstract_mesh()
    if len(configs) < 2 or device is None:
        return [certified(op, arguments, config, round=round) for config in configs]
    entered = _cache.carried()

    def asked(config: Hashable) -> tuple["Binary", ...] | None:
        with (entered(), certifying(device),
              contextlib.nullcontext() if placed.empty else jax.set_mesh(placed),
              contextlib.nullcontext() if abstract.empty else jax.sharding.use_abstract_mesh(abstract)):
            return certified(op, arguments, config, round=round)

    with ThreadPoolExecutor(len(configs), thread_name_prefix="dew-certify") as pool:
        submitted = [pool.submit(contextvars.copy_context().run, asked, config) for config in configs]
        return [each.result() for each in submitted]


def rung_round() -> int:
    """The most rungs a round of certification compiles at once (`certified_together`): the cores this
    process may run on, so a small host takes fewer rungs a round rather than sharing its cores among more
    compiles."""
    return max(1, len(os.sched_getaffinity(0)))


def _verdict_key(identity: str, config: Hashable, reverse: bool) -> str | None:
    """What decides a launch's certification on a GPU model before it traces, as its stored verdict's key
    (`tuning.stored_verdict`): the launch's `identity` (`_identity`, the same in every process), its config,
    whether reverse mode runs it, Dew's code (`package_digest`: a Dew edit certifies again), the installed
    JAX, jaxlib and CUDA plugins, whose compiler assembles it (the Pallas lowering cache's seal of them), and
    the environment a kernel's lowering and compile read (that cache's policy); the store's file names the
    GPU model. None where the installation has no seal (an edited or a source install of JAX), whose code
    no key states: its verdicts are not stored."""
    from jax._src.pallas.mosaic_gpu.lowering_cache import _keyed_environment
    from jax._src.pallas.mosaic_gpu.lowering_cache_seal import seal

    from dew._cache import package_digest
    from dew.nn.kernels.tuning import encoded

    installed = seal().digest
    if installed is None:
        return None
    stated = [package_digest(), installed.hex(), identity, encoded(config), reverse, _keyed_environment()]
    return hashlib.sha256(json.dumps(stated, sort_keys=True).encode()).hexdigest()


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
    ahead for one, `dew.training.ahead`), else this process's own GPU; either where it compiles for that
    GPU's own generation (`generation.device_target`, whatever `assumed` says: an A100 tracing for an
    assumed sm90 compiled the sm90 launch on itself, and Mosaic refused its TMA copies, rows 6677 and 6678),
    else none. A trace under a compile-only GPU's mesh certified on the host's own GPU certified each launch a
    second time, apart, where the start's compile certified it on the mesh's (row 6791)."""
    from jax._src import mesh as mesh_lib

    named = _CERTIFYING.get()
    if named is not None:
        return named
    placed = mesh_lib.get_concrete_mesh()
    devices = jax.local_devices() if placed.empty else [
        device for device in placed.devices.flat if device.process_index == jax.process_index()]
    # A device of another generation (a CPU's, an A100's under an assumed sm90) certifies nothing.
    if not devices or generation.device_target(devices[0]).generation != generation.gpu_target().generation:
        return None
    return devices[0]


def _compiled_launch(op, arguments: tokamax.BoundArguments, config: Hashable, device,
                     reverse: bool) -> dict[str, tuple[str, "Binary"]]:
    """`op`'s launch of `arguments` at `config` as a resource derivation lowers it (`KernelOp.lowering`),
    compiled alone on `device`, as a call without a gradient runs it or, `reverse`, as reverse mode does (its
    forward and its pullback of a unit cotangent): its Mosaic GPU and Pallas-Triton kernels' modules (a Triton
    call's config) and binaries, by name and launch, none for a launch of none. A launch that does not trace
    alone raises: certification is no optional pass."""
    from dew.nn.kernels.lowered import kernel_calls
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

        # Named by its launch and config in JAX's trace, lowering and compile events, so a start's receipt
        # tells each certification's program apart: its op, its launch's identity and its config, digested.
        certified.__name__ = certified.__qualname__ = (
            f"certified_{type(op).__name__}_l{_digest(_identity(op, arguments))}_c{_digest(repr(config))}"
            f"{'_reverse' if reverse else ''}")
        lowered = owned_program(certified).trace(*specs).lower()
    # A Mosaic GPU kernel compiled alone is the one its program loads (Mosaic compiles a module once a
    # process). A Pallas-Triton kernel compiles again with each executable, the same kernel the same binary:
    # read here too, its call's config its identity, so its derivation steps a spilling config at trace as a
    # Mosaic launch's does (an H100's float32 attention forward at width 256 spilled at every four-warp tile
    # and held at eight, row 6507), rather than ending its program's compile Exhausted.
    calls = kernel_calls(lowered, unrun=True)
    modules = kernel_modules(calls) | {name: call.config for name, call in
                                       kernel_bodies(call for call in calls if call.kind == "triton").items()}
    if not modules:
        return {}
    return {name: (modules[name], binary)
            for binary in binaries(xla_compiled(lowered, None))
            if (name := launch_name(binary.kernel, binary.launch)) in modules}


def _certifying_placement(device) -> tuple[jax.sharding.Mesh, frozenset]:
    """The mesh certification places a launch's operands on `device` on and sets its program on
    (`_compiled_launch`), and the axes its launch holds manual: its program's where that runs
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
def refused_launches() -> tuple[tuple[str, Hashable, frozenset[str]], ...]:
    """Every (launch identity, config, spilled kernel names) refused in this context (a `refusals` scope's or
    the process's), ordered. A validated plan can hand them to another process explicitly (`refuse` within
    `refusals`); a shared compilation cache alone must not transfer them between programs. A trace after a
    refusal derives the next config, so a program kept after one is another than a process without it
    traces (`traced_fact`)."""
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
def _proposing(entries: list[tuple[tuple[str, Hashable], str]]) -> Iterator[Recorded]:
    """Refusals visible while deriving and compiling a replacement, without changing the kept program's
    registry or its file; every rewrite and pass the replacement's trace records, which a successful
    replacement commits after this scope ends (`_commit`)."""
    registry = _registry()
    with _GUARD:
        proposed = dict(registry.refused)
    for identity, kernel in entries:
        proposed[identity] = proposed.get(identity, frozenset()) | {kernel}
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


def _add(**records: Iterable) -> None:
    """Each of `records`, a `Recorded` field's entries, added to this context's registry and to every
    `recording` open in it."""
    with _GUARD:
        for held in (_registry(), *_RECORDING.get()):
            for name, entries in records.items():
                getattr(held, name).extend(entries)


def _record(refusal: tuple[str, Hashable] | None, miss: "Miss | None",
            kernels: frozenset[str] = frozenset()) -> bool:
    """`refusal` and `miss` recorded; whether `miss` is new. A miss is recorded once: a launch's derivation
    asked again (its route's eligibility, then its call, `tuning.KernelOp.taken`) reads its certification
    back and finds the same misses."""
    registry = _registry()
    with _GUARD:
        if refusal is not None:
            registry.refused[refusal] = registry.refused.get(refusal, frozenset()) | kernels
            registry.by_launch.pop(refusal[0], None)
        if miss is None or miss in registry.missed:
            return False
        _add(missed=(miss,))
        return True


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
        return Attempt(xla_compiled(lowered, options))
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


def untuned(code: str, message: str) -> bool:
    """Whether XLA refused a deviceless compile for want of a tuned result (`UNTUNED`)."""
    return code == "NOT_FOUND" and any(said in message for said in UNTUNED)


def xla_compiled(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None
                 ) -> jax.stages.Compiled:
    """`lowered` compiled by XLA under `options` for its target (`targeted`): Dew's one call of XLA's
    compile, where an instruction no device run has tuned yet is said so (`Untuned`)."""
    try:
        return lowered.compile(targeted(lowered, options))
    except jax.errors.JaxRuntimeError as error:
        if untuned(error.error_code_string, str(error)):
            raise Untuned(str(error), lowered) from error
        raise


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


def offline(devices: Sequence) -> bool:
    """Whether a program compiled for `devices` is read offline, for its memory or text alone, no device to
    run it (`targeted`): compile-only GPUs outside `for_devices` (or within `read_offline`)."""
    device = next(iter(devices), None)
    return device is not None and device.platform == "gpu" and deviceless(device) and not _FOR_DEVICES.get()


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
    search's every probe and rung, 142 s to 99 s on armada, 11-02). A run's own XLA_FLAGS setting keeps
    `options` as given. Every program compiled here takes it, a run's step and every program it compiles
    ahead alike."""
    from jax._src.interpreters import pxla

    lowering = lowered._lowering
    devices = lowering._device_list if isinstance(lowering, pxla.MeshComputation) else None
    if devices is None or next(iter(devices)).platform != "gpu":
        return options
    device = next(iter(devices))
    if offline(devices):
        return options if xla_flag("xla_gpu_autotune_level") is not None else {
            **(options or {}), "xla_gpu_autotune_level": 0}
    if xla_flag("xla_gpu_deviceless_cub_mode") is None:
        options = {**(options or {}), "xla_gpu_deviceless_cub_mode": CUB_NO_FALLBACK}
    if deviceless(device) and xla_flag("xla_gpu_autotune_level") is None:
        return {**(options or {}), "xla_gpu_experimental_autotune_cache_mode": AUTOTUNE_CACHE_READ,
                "xla_gpu_require_complete_aot_autotune_results": True}
    return options


def attempts(programs: Sequence[tuple[jax.stages.Lowered, jax.stages.CompilerOptions | None]]
             ) -> list[Attempt]:
    """Each of `programs` compiled once (`attempted`), in order, all at once, at most `COMPILE_THREADS`
    at a time; one alone in this thread. JAX's cache key reads a module's context without writing it from
    0.12.0.1.post54 (pyproject's pin), so programs one thread lowered in one context compile together."""
    if len(programs) < 2:
        return [attempted(*program) for program in programs]
    with ThreadPoolExecutor(min(len(programs), COMPILE_THREADS), thread_name_prefix="dew-compile") as pool:
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
        refusals = [(refusal, report.kernel) for refusal, report, _ in found
                    if refusal is not None and report.kernel not in held.get(refusal, frozenset())]
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
        for refusal, kernel in [*refusals, *((refusal, report.kernel) for refusal, report, _ in alone)]:
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
                 if refusal is not None and report.kernel not in held.get(refusal, frozenset())]
        if not found:
            return program, missed
        for refusal, report, _ in found:
            _record(refusal, None, frozenset({report.kernel}))
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
            tuning.store_verdict(device.device_kind, key, kernels)


def _untaken(program: jax.stages.Lowered, refusals: list[tuple[tuple[str, Hashable], str]]
             ) -> list[tuple[tuple[str, Hashable], str]]:
    """Each of `refusals` (a launch's identity and config, and its spilling kernel's name) that `program`,
    relowered under them, launches again: its launch's derivation did not read the refusal. Each is logged,
    so the launch whose rule ignores its refusals is named."""
    from dew.nn.kernels.lowered import kernel_calls

    # Every call JAX compiles, a branch's no run takes among them (a forward's scaled op's).
    launched = {(launch.identity, launch.config) for call in kernel_calls(program, unrun=True)
                if (launch := called(call)) is not None}
    untaken = [(refusal, kernel) for refusal, kernel in refusals if refusal in launched]
    for (_, config), kernel in untaken:
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
        with _proposing([(refusal, kernel) for refusal, (kernel, _) in tried.items()]) as proposal:
            replacement, spilling = _compiled_reported(relower(), options)
    except (Exhausted, jax.errors.JaxRuntimeError) as again:
        if isinstance(again, jax.errors.JaxRuntimeError) and shared_asked(again) is None:
            raise
        for kernel, launch in tried.values():
            _missed_block(kernel, launch, asked_by.get(kernel, asked), stepped=False)
        raise error from again
    _commit(proposal)
    for refusal, (kernel, launch) in tried.items():
        _record(refusal, None, frozenset({kernel}))
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
        miss = Miss(report.kernel, None if launch is None else launch.op,
                    None if launch is None else launch.config, report.registers, report.frame,
                    stepped and launch is not None,
                    serialized=tuple(note.code for note in report.notes if note.code in SERIALIZED))
        if _record(None, miss):
            _log.warning("kernel %s %s at %s registers (%s at %s); %s", miss.kernel, miss.reason,
                         miss.registers, miss.op, miss.config,
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
    caller: str | None = None
    """The launch that traced it, as its call's caller stated it (`CALLER_METADATA`, a Mosaic GPU kernel's);
    None where its call stated none."""
    notes: tuple["AssemblerNote", ...] = ()
    """What ptxas noted of a loss of performance as it assembled it, in its order (`ASSEMBLER_NOTES`, a
    Mosaic GPU kernel's); none where its call states none (a Pallas-Triton kernel's, a build without them)."""

    @property
    def launched(self) -> str | None:
        """The digest of the launch that traced it: its caller's, else its name's, which is its launch's where
        the launch states no kernel's part (`named`)."""
        return self.caller or self.launch

    @property
    def refused(self) -> bool:
        """Whether its config is refused for it: it spills (`frame`), or the assembler serialized its
        warpgroup MMAs (`SERIALIZED`)."""
        return bool(self.frame) or any(note.code in SERIALIZED for note in self.notes)


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
    runtime = executable.runtime_executable()
    if runtime is None:
        raise ValueError("a compiled program with no runtime executable carries no kernel binaries to read")
    return _carried(runtime.serialize(), decoded)


def _carried(serialized: bytes, decoded: bool = False) -> list[Binary]:
    """`binaries`' reading of an executable as XLA serializes it (a program's, or a kernel's compiled alone,
    `compiled_alone`)."""
    found: list[Binary] = []
    for name, proto in _thunk_protos(serialized):
        found.extend(thunk_kernels(proto, decoded=decoded))
        # XLA's own kernels (its fusions, its Triton GEMMs) carry their cubins too: a template launch's
        # names its launch (`named`).
        kernel = custom_kernel(proto) if LAUNCH_SUFFIX.search(name) else None
        if kernel is not None:
            elf, shared = kernel
            found.append(dataclasses.replace(_binary(name, elf, decoded), shared=shared))
    return found


def launched_kernels(executable: jax.stages.Compiled) -> list[tuple[str, int]]:
    """Every custom kernel `executable` launches, by the name its thunk gives it, with the shared bytes its
    launch asks for (`custom_kernel`): a Pallas-Triton call's whatever its name (`binaries` keeps a
    template launch's), and XLA's own custom kernels, read from its serialized form as `binaries`."""
    runtime = executable.runtime_executable()
    if runtime is None:
        raise ValueError("a compiled program with no runtime executable launches no kernel to read")
    return [(name, kernel[1]) for name, proto in _thunk_protos(runtime.serialize())
            if (kernel := custom_kernel(proto)) is not None]


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


def _kernels(message: bytes, decoded: bool = False) -> tuple[list[Binary], int]:
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
        return [dataclasses.replace(_binary(name[0].decode(), binary[0], decoded), shared=stated,
                                    caller=stated_caller, notes=notes)], 0
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
