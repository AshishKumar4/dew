"""What Mosaic GPU's compiles say of each kernel, what the templates' resource models are checked against.
A run's compile (`compiled`, owner 10-29, bible §6) reads each kernel's registers a thread and the bytes it
spills from the binary its executable carries (`binaries`): a launch whose kernel spills is refused
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
import json
import logging
import re
import threading
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal, Protocol, get_args, overload

import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import stablehlo

from dew import _cache, _facts, communicators
from dew._code import op_provenance, provenance_code
from dew._devices import deterministic_ops_requested, deviceless, xla_flag
from dew.nn.kernels import generation
from dew.nn.kernels.templates import assemblers
from dew.nn.kernels.templates.assembled import (
    MOSAIC_TARGET,
    SERIALIZED,
    Binary,
    binaries,
    carried,
    kernel_of,
    launch_name,
)

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
    provenance: frozenset[str] | None = dataclasses.field(default=None, compare=False)
    """The modules of Dew's whose code its records key on (`_provenance`), None for the whole package's."""


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
    provenance: Callable[[], frozenset[str] | None] = lambda: None


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
refusal (`derivation_inputs`, `_proposing`, `refusal_snapshot`) carries what a launch's ladder steps by."""


StoredOutcome = Literal["hit", "miss", "invalid", "unstorable"]
"""A stored derivation's lookup (`dew.algebra.stored`): read from the store, derived afresh and written,
derived afresh past an entry that failed to read, or derived afresh and not written (a program or a value
its key or codec does not state)."""


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
    stored: collections.Counter[StoredOutcome] = dataclasses.field(default_factory=collections.Counter)
    """Each stored derivation's lookup by its outcome (`record_stored`)."""
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
    stored: collections.Counter[StoredOutcome] = dataclasses.field(default_factory=collections.Counter)


_RECORDING: contextvars.ContextVar[tuple[Recorded, ...]] = _cache.untraced(contextvars.ContextVar("recording",
                                                                                                  default=()))
"""The `recording` scopes open in this context, innermost last."""


_STORED_RECORDING: contextvars.ContextVar[tuple[Recorded, ...]] = _cache.untraced(
    contextvars.ContextVar("stored_recording", default=()))
"""The `recording` scopes open in this context that a stored derivation's outcome reaches (`record_stored`):
`_RECORDING`'s, but for a replacement's proposal (`_proposing`), whose stored outcomes, the cache's work done
whether or not the replacement is kept, are the scopes' it was proposed in."""


_PROCESS = _Registry()
"""The process's registry, every thread's. A shared executable cache does not share these decisions:
only an explicitly handed plan seeds them, after validating its program and target."""
_SCOPE: contextvars.ContextVar[_Registry | None] = _cache.untraced(
    contextvars.ContextVar("refusals_scope", default=None))
"""A `refusals` scope's own registry, in its context alone: a trace reads it through its facts alone
(`refusals_of`, `refusal_snapshot`, under `_GUARD`), so a held trace's key leaves it out (`_cache.untraced`),
where its records' walk raced a compile worker filling them and a first lookup made the key stale."""
_GUARD = threading.RLock()
"""Serializes the registries' reads and updates between threads."""


def _registry() -> _Registry:
    return _SCOPE.get() or _PROCESS


@dataclasses.dataclass(eq=False)
class Reading:
    """The refusals a trace reads (`reading`): those in effect as it began, and those it records itself
    since (a launch it certifies inline, its derivation stepping past its own refusal), never another
    thread's, recorded into the registry it shares as it traces (theme K01-e's review P0: a held jit's
    trace at a new shape read another scope's refusal, which JAX's trace cache then held for its own).
    A `frozen` one takes none of its own: a refusal recorded within it ends its trace (`ReadingRefused`)."""
    refused: dict[tuple[str, Hashable], frozenset[Refused]]
    frozen: bool = False
    by_launch: dict[str, frozenset] = dataclasses.field(default_factory=dict)


class ReadingRefused(BaseException):
    """A refusal recorded within a frozen reading (`reading`'s `frozen`: a held jit's trace), recorded into
    the registry: the trace ends here, before JAX's trace cache takes the program the refusal would make
    under the refusals its jit is held for, and its caller traces afresh by a jit of its own (theme K01-e's
    review P0: a held jit's trace, its cached program dropped, recorded an inline refusal, read it, and its
    cache kept the refusal's program for every call it answered). Not an `Exception`: no handler on the way
    takes it for a launch's failure."""


_READING: contextvars.ContextVar[Reading | None] = _cache.untraced(
    contextvars.ContextVar("refusals_read", default=None))
"""The `reading` a trace in this context reads its refusals from, None outside one."""


@contextlib.contextmanager
def reading(entries: tuple, *, frozen: bool = False) -> Iterator[Reading]:
    """A trace's reads of the refusals (`refusals_of`, `refusal_snapshot`) answered from `entries` (a
    `refusal_snapshot` snapshot) and those it records itself within, for as long as it traces; `frozen`, none
    of its own, a refusal recorded within it ending its trace (`ReadingRefused`). One within a frozen one is
    frozen too: a trace inside a held jit's makes part of its program."""
    outer = _READING.get()
    frozen = frozen or (outer is not None and outer.frozen)
    held = Reading({(identity, config): kernels for identity, config, kernels in entries}, frozen)
    token = _READING.set(held)
    try:
        yield held
    finally:
        _READING.reset(token)


class Exhausted(ValueError):
    """A derivation that finds no config left for a launch (`base.Unfitted`): a relowering after a spill
    that ends so refuses the program (`compiled`), which no compile keeps spilling."""


@dataclasses.dataclass(frozen=True)
class Miss:
    """A kernel a run's compile refused (`Binary.refused`: spilling, or its warpgroup MMAs serialized), or a
    program whose block XLA refused (`compiled`): its kernel's name, its launch's op and config (None where
    no launch of this process traced it), its registers and the bytes it spills a thread (`Binary.spills`)
    or the shared bytes it asked (`shared`), and whether its launch's config was refused and its program
    lowered again (`stepped`)."""
    kernel: str
    op: str | None
    config: Hashable | None
    registers: int | None
    spilled: int | None
    stepped: bool
    shared: int | None = None
    serialized: tuple[str, ...] = ()
    """The codes of ptxas's notes of its warpgroup MMAs serialized (`SERIALIZED`), a reason it was refused
    beside a spill (`spilled`)."""
    modelled: int | None = None
    """The registers a consumer thread holds as its launch's model stated them (`RegisterModel.registers`,
    `named`), beside the assembler's (`registers`); None where its launch stated none."""

    @property
    def reason(self) -> str:
        """Why its kernel was refused, as a log and a plan state it."""
        if self.shared is not None:
            return f"its block asks {self.shared} shared bytes"
        reasons = ([f"spills {self.spilled} bytes"] if self.spilled else []) + (
            [f"serializes its warpgroup MMAs ({', '.join(self.serialized)})"] if self.serialized else [])
        return " and ".join(reasons) or "is refused"


def _identity(op, arguments: tokamax.BoundArguments, target: "generation.GpuTarget | None") -> str:
    """A launch's identity apart from its config, the same in every process: its op as its picks name it
    (`tuning._identity`) and its arguments exactly, its arrays by their own shapes and dtypes (no key's
    bucket: two launches a key shares compile two kernels), its hooks, the op's and its arguments', by what
    they compute as their parts tell it (`tuning.KernelOp.identified`: every launch reads it, so it traces
    nothing), and the capacities of the `target` it is for that size its generated program (its caller's:
    a derivation's own, a template's `launch_target`, the GPU in effect; None for no GPU's). Clock rates are
    not code inputs once its config is chosen: an attached device and its compile-only target may report
    different rates while emitting exactly that config."""
    def described(name, value):
        if name in op.HOOKS:
            return op.identified(name, value)
        return jax.tree.map(lambda leaf: (tuple(leaf.shape), str(leaf.dtype)) if hasattr(leaf, "dtype")
                            and hasattr(leaf, "shape") else leaf, value)
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
    token = _TRACING.set(_Pending(lambda: _identity(op, arguments, generation.gpu_target()),
                                  type(op).__name__, config,
                                  lambda: op.padded(arguments, config), op.emitted,
                                  reverse=_reversed(op),
                                  provenance=lambda: op_provenance(op, arguments.arguments)))
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
                                  launch.emitted, digest, launch.reverse, lambda: launch.provenance))
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


def launch_tracing() -> bool:
    """Whether a launch is tracing (`launching`), asked without its digest, whose identity a launch traces
    its own body to form (`launch_digest`): a span within that body asking it recursed without end."""
    return _TRACING.get() is not None


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
                                                emitted=pending.emitted, reverse=pending.reverse,
                                                provenance=pending.provenance()))
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
    identity = _identity(op, arguments, target)
    return identity, target, refusals_of(identity), deterministic_ops_requested()


@_facts.traced_fact
def refusals_of(identity: str) -> frozenset[tuple[Hashable, frozenset[Refused]]]:
    """Each config this context's compiles refused the launch of `identity` (`_identity`), with the kernels
    refused at it and their roles (`Refused`): what a launch's derivation and legality read of the
    refusals, the one way a trace reads
    them (`traced_fact`), so a trace kept past a refusal of another launch holds."""
    with _GUARD:
        registry = _READING.get() or _registry()
        if (found := registry.by_launch.get(identity)) is None:
            found = registry.by_launch[identity] = frozenset(
                (config, kernels) for (launch, config), kernels in registry.refused.items()
                if launch == identity)
        return found


def refusals_at(op, arguments: tokamax.BoundArguments, target: "generation.GpuTarget | None"
                ) -> frozenset[tuple[Hashable, frozenset[Refused]]]:
    """Each config a compile of this process refused `op`'s launch of `arguments` on `target` at, with the
    kernels refused there and their roles (`refusals_of` of its identity, its cached snapshot as is): the one
    reading of a launch's refusals, which `refused`, `refused_at` and `spilled_in` read at a config
    (`_refused_kernels`), and a launch none of whose configs is refused reads as empty (row 6999: a fifth
    reader asked that alone)."""
    return refusals_of(_identity(op, arguments, target))


def _refused_kernels(found: frozenset[tuple[Hashable, frozenset[Refused]]], config: Hashable
                     ) -> frozenset[Refused]:
    """The kernels a launch's refusals (`refusals_at`) refused at `config`, none where it refused none."""
    return next((kernels for refused_config, kernels in found if refused_config == config), frozenset())


def refused(op, arguments: tokamax.BoundArguments, config: Hashable,
            target: "generation.GpuTarget | None") -> bool:
    """Whether a compile of this process refused `op`'s launch of `arguments` at `config` on `target`: a
    kernel of it spilled or serialized its warpgroup MMAs (`refused_at`). A template's legality reads it, so
    its derivation takes the next config."""
    return bool(refused_at(op, arguments, config, target))


def refused_at(op, arguments: tokamax.BoundArguments, config: Hashable,
               target: "generation.GpuTarget | None") -> frozenset[str]:
    """The names of `op`'s kernels a compile of this process refused at `config` for `arguments` on `target`
    (`Binary.refused`: spilling, or their warpgroup MMAs serialized): a launch of several kernels (a
    recurrence's forward and backward walks) steps the one refused."""
    return names(_refused_kernels(refusals_at(op, arguments, target), config))


def names(kernels: Iterable[Refused]) -> frozenset[str]:
    """The names of refused `kernels` (`Refused`): what a refusal's readers step by, its roles aside."""
    return frozenset(kernel for kernel, _ in kernels)


def spilled_in(op, arguments: tokamax.BoundArguments, config: Hashable,
               target: "generation.GpuTarget | None") -> dict[str, tuple[str, ...]]:
    """The roles each of `op`'s kernels a compile refused at `config` for `arguments` spilled in, as its
    refusal states them (`Refused`), by kernel name: what a launch's ladder steps it by (row 6744). A kernel
    whose roles its compile left unread is absent; one read and spilling none (its warpgroup MMAs serialized
    alone) states `()`."""
    found = _refused_kernels(refusals_at(op, arguments, target), config)
    return {kernel: roles for kernel, roles in found if roles is not None}


def refuse_launch(op, arguments: tokamax.BoundArguments, config: Hashable, kernels,
                  spilled_in: Mapping[str, tuple[str, ...] | None] | None = None) -> None:
    """`op`'s launch of `arguments` at `config` refused in this context as a compile that found `kernels`
    spilling refuses it (`settled`), so its derivation takes the next config: a compile gate's, which
    compiles a launch apart from `compiled`, stepping as a run's compile steps (row 4727). `spilled_in`, the
    roles each kernel's spill stood in where the compile read them (`Binary.spilled_in`), kept with it
    (`Refused`) for its ladder (`spilled_in`)."""
    _refused_launch(_identity(op, arguments, generation.gpu_target()), config, kernels, spilled_in)


def _refused_launch(identity: str, config: Hashable, kernels,
                    spilled_in: Mapping[str, tuple[str, ...] | None] | None) -> None:
    """The launch of `identity` (`_identity`: its op, arguments and target) refused at `config`, as
    `refuse_launch` refuses it: what a settle of launches certified aside records by the identity it queued
    each under, whatever GPU is in effect as it drains (row 12031)."""
    roles = spilled_in or {}
    _record((identity, config), None, frozenset((kernel, roles.get(kernel)) for kernel in kernels))


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


def _reversed(op) -> bool:
    """Whether `op`'s launch runs, and is certified and named, as reverse mode runs it: traced in a
    gradient's product (`pulled`) or a pulled op (`tuning.KernelOp.pulled`, its tuner's candidates certified
    outside a gradient's trace, `tuning._candidates`, where a spilling backward passed), of an op whose call
    carries its derivative."""
    return (_PULLED.get() or op.pulled) and op.CARRIES_DERIVATIVE


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
    """A launch certified aside (`certifying_aside`): what `certified` read of it, each reporting scope it
    was taken in (`_reporting`; the first its queueing's, another for each scope that took it again, which
    its one replay's verdict is recorded in too, row 12007), and its replay's kernels to come once its call
    was reached (`capture`)."""
    op: "KernelOp"
    arguments: tokamax.BoundArguments
    config: Hashable
    reverse: bool
    device: Hashable
    stored: str | None
    reporting: list[tuple]
    kernels: Future | None = None


@dataclasses.dataclass
class Aside:
    """The launches a block certified aside (`certifying_aside`), by `certified`'s key, compiling on
    `pool`; and each whose call the block's trace has not reached yet (`awaiting`), by its launch's digest
    (`launch_digest`) and mode: its `pending` key."""
    pool: ThreadPoolExecutor
    pending: dict[tuple, _AsideLaunch] = dataclasses.field(default_factory=dict)
    awaiting: dict[tuple[str, bool], tuple] = dataclasses.field(default_factory=dict)


# Untraced (`_cache.untraced`), as `_RECORDING`: the block's pending launches change as it captures them, and
# a trace keyed on them lowered a second layer's calls apart from its first's, so a cold build lowered
# another program than a warm run (rows 7322, 8299, 8372).
_ASIDE: contextvars.ContextVar[Aside | None] = _cache.untraced(
    contextvars.ContextVar("certifying_aside", default=None))


@contextlib.contextmanager
def certifying_aside() -> Iterator[None]:
    """Within the block, a launch whose derivation has refused none of its configs is certified aside
    (`certified`): lowered as the trace reaches it, Python that holds the GIL, and compiled on
    `compile_threads` threads while the trace runs on, XLA and the GPU's assembler releasing it, its config
    taken as holding until the block's end reads its compile (`_settled_aside`), within a settle that records
    its misses (`settling`), every compile waited for. A launch with a refusal certifies in turn, its ladder
    read as it climbs (`tuning.KernelOp._certified`). A cold start's 47 launches certified in turn took 48 s,
    18 of them compiling (M-MC-21b), and every config the gates derive holds (FS-DECISION: an attention
    forward's 68 and a backward's 45 on sm90), so a refusal's relowering is rare. The one way a caller
    certifies aside, its steps in order (row 12015); a lowering that relowers past a refusal settles wider
    (`lowered_aside`)."""
    with settling():
        with _aside_block() as aside:
            yield
        _settled_aside(aside)


@contextlib.contextmanager
def _aside_block() -> Iterator[Aside]:
    """A block certifying aside (`certifying_aside`) whose launches its caller settles (`_settled_aside`):
    its compiles' pool, each compile waited for as it ends."""
    with ThreadPoolExecutor(compile_threads(), thread_name_prefix="dew-certify") as pool:
        aside = Aside(pool)
        token = _ASIDE.set(aside)
        try:
            yield aside
        finally:
            _ASIDE.reset(token)


def _settled_aside(aside: Aside) -> frozenset[tuple]:
    """Each launch `aside` certified, kept as `certified` keeps one and dropped from the block's pending: a
    launch whose kernels the assembler refuses refused (`refuse_launch`), so a trace that took its config is
    traced afresh (`settled_trace`) and a program lowered at it lowered afresh (`refused_in`), its misses kept
    for that retry to record (`settling`, which every caller settles within). The keys of those refused."""
    settlement = _SETTLING.get()
    assert settlement is not None
    refused = set()
    for key, each in list(aside.pending.items()):
        # A launch a derivation certified (a route's eligibility) and no trace called: no program holds it.
        # One certified in turn meanwhile (a refusal's ladder, `tuning.KernelOp._certified`) is certified.
        with _GUARD:
            done = key in _CERTIFIED
        if each.kernels is None or done:
            continue
        with _GUARD:
            aside.pending.pop(key, None)
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
            refused.add(key)
            with _GUARD:
                scopes = tuple(each.reporting)
            for reporting in scopes:
                # Refused in each scope its call was taken in, whatever scope drains it.
                with _reported_in(reporting):
                    _refused_launch(key[0], each.config, {binary.kernel for binary in spilling},
                                    {binary.kernel: binary.spilled_in for binary in spilling})
                # Held by the settle (`settling`) until a retry, under its own context, steps past it (row
                # 11786).
                settlement.deferred.append((key, Launch(key[0], type(each.op).__name__, each.config),
                                            tuple(spilling), reporting))
    return frozenset(refused)


def settled_trace[T](trace: Callable[[], T], retrace: Callable[[], T]) -> T:
    """`trace()`'s value, within a block certifying aside the launches it reached settled before it lowers
    (`_settled_aside`, within `settling`), and traced afresh (`retrace()`, past any trace cache) while they
    refuse a config it took: it lowers once, at the configs its certification settled (FS-ASIDE-SM100C).
    Each trace runs in the caller's context, which JAX's trace cache keys on, so a program compiled ahead is
    found by its trace (row 11766); an abandoned trace's routes are withdrawn (`generation.withdrawn`), a
    nested program retraces past the refusal (`options.OuterProgram._nested`, row 11783), and a failed trace
    keeps the routes it reported, as JAX's does. Outside such a block, `trace()`'s."""
    aside = _ASIDE.get()
    if aside is None:
        return trace()
    with settling():
        marks = generation.route_marks()
        value = trace()
        seen: set = set()
        while (refused := _settled_aside(aside)) and not refused <= seen:
            seen |= refused
            generation.withdrawn(marks)
            value = retrace()
    return value


def lowered_aside[T](lower: Callable[[], T], relower: Callable[[], T],
                     program: Callable[[T], jax.stages.Lowered | None]) -> T:
    """`lower()`'s value, the launches its trace reaches certified aside (`certifying_aside`, settled before
    it lowers where it traces through `settled_trace`, else as it ends: `_settled_aside`), and `relower()`'s
    afresh for as long as its program (`program`, None where it holds none) launches a refused config
    (`refused_in`), each refused launch's derivation stepping past its config and certifying the next in
    turn: its own certifications' refusals, or another caller's committed after its trace took the config
    (row 8603: another thread's drain left a program compiling at it). A relowering that launches a config
    refused before again (a derivation reading no refusal, a pin's) is its compile's to read (`settled`).
    Every lowering a compile point settles lowers through it: a program's own (`options.OuterProgram`) and a
    step's planner's and fit check's (`planning.FitSearch.lowering`), whose launches certified in turn took
    64 of an sm90 plan's 82 seconds (FS-P5324-MAIN, row 5324); a program's trace alone certifies aside too
    (`OuterProgram.trace`)."""
    with settling():
        with _aside_block() as aside:
            value = lower()
        _settled_aside(aside)
        seen: set = set()
        while ((lowered := program(value)) is not None and (refused := refused_in(lowered))
               and not refused & seen):
            seen |= refused
            with _aside_block() as aside:
                value = relower()
            _settled_aside(aside)
    return value


@dataclasses.dataclass
class _Settlement:
    """A settle's misses (`_settled_aside`) by certification key and queueing scope, held until the launch
    holds at another config in that scope's registry (stepped) or the settle ends, recorded in their own
    scope either way (row 11786)."""

    scope: tuple
    deferred: list[tuple[tuple, Launch, tuple["Binary", ...], tuple]] = dataclasses.field(
        default_factory=list)

    def certified(self, key: tuple) -> None:
        """Its misses of `key`'s launch, device and mode at other configs, in this registry, stepped."""
        identity, config, *placement = key
        registry = _SCOPE.get()
        with _GUARD:
            past = [entry for entry in self.deferred
                    if entry[0][0] == identity and list(entry[0][2:]) == placement and entry[0][1] != config
                    and entry[3][0] is registry]
            self.deferred = [entry for entry in self.deferred if entry not in past]
        for _, launch, kernels, reporting in past:
            with _reported_in(reporting):
                _recorded([(None, binary, launch) for binary in kernels], stepped=True)

    def ended(self) -> None:
        """Its misses no certification stepped past recorded, not stepped."""
        with _GUARD:
            deferred, self.deferred = self.deferred, []
        for _, launch, kernels, reporting in deferred:
            with _reported_in(reporting):
                _recorded([(None, binary, launch) for binary in kernels], stepped=False)


def _reporting() -> tuple:
    """This context's `refusals` registry (None: the process's) and open `recording`s."""
    return _SCOPE.get(), _RECORDING.get()


@contextlib.contextmanager
def _reported_in(reporting: tuple) -> Iterator[None]:
    """Within the block, refusals and misses recorded in `reporting` (`_reporting`'s)."""
    registry, recordings = reporting
    scope, recorded = _SCOPE.set(registry), _RECORDING.set(recordings)
    try:
        yield
    finally:
        _RECORDING.reset(recorded)
        _SCOPE.reset(scope)


_SETTLING: contextvars.ContextVar[_Settlement | None] = _cache.untraced(
    contextvars.ContextVar("settling_aside", default=None))


@contextlib.contextmanager
def settling() -> Iterator[None]:
    """A settle of blocks certifying aside and its retries (`_Settlement`), its misses recorded as it ends,
    even where it raises; within another of the same scope, the enclosing one's."""
    scope = (id(_registry()), *map(id, _RECORDING.get()))
    enclosing = _SETTLING.get()
    if enclosing is not None and enclosing.scope == scope:
        yield
        return
    settlement = _Settlement(scope)
    token = _SETTLING.set(settlement)
    try:
        yield
    finally:
        _SETTLING.reset(token)
        settlement.ended()


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

    reverse = _reversed(op)
    # Apart by the judge remaking it (`judging`): a run's verdicts, its build's, no other's.
    key = (_identity(op, arguments, generation.gpu_target()), config, device.id, device.device_kind, reverse,
           _cache.KERNEL_JUDGED.value)
    with _GUARD:
        held = _CERTIFIED.get(key)
        if held is None and key in _UNCERTIFIED:
            return None
    if held is not None:
        if not held.refused:
            _progressed(key)
        return held.refused
    provenance = op_provenance(op, arguments.arguments)
    target = verdict_target(device)
    stored = _verdict_key(key[0], config, reverse, provenance, target)
    verdict = None if stored is None else tuning.stored_verdict(target, stored)
    if isinstance(verdict, dict):
        return _kept_certification(key, op, config, reverse, verdict, (), device, read=True).refused
    aside = _ASIDE.get()
    reporting = _reporting()
    with _GUARD:
        taken = None if aside is None else aside.pending.get(key)
        if taken is not None and not any(held[0] is reporting[0] and held[1] is reporting[1]
                                         for held in taken.reporting):
            taken.reporting.append(reporting)
    if taken is not None:
        return ()
    # A launch a config of which was refused here is certified now; asked of its own refusals alone, so a
    # trace held past another launch's refusal holds (row 12167).
    if aside is not None and not refusals_of(key[0]) and _traced_for(device):
        # Taken as holding, nothing lowered here: the block's trace's own call of the launch is replayed and
        # compiled aside (`capture`), the block's end settling it (`_settled_aside`), once a block.
        aside.pending[key] = _AsideLaunch(op, arguments, config, reverse, device, stored, [reporting])
        aside.awaiting[(_launched(key[0], config, reverse), reverse)] = key
        return ()
    # Its verdict where its kernels' was stored however their Dew tree named it (`_lowered_verdict_key`),
    # stored by this tree's key too, so its next start reads it tracing nothing.
    lowered, modules, lowered_key = _lowered_launch(op, arguments, config, device, reverse)
    verdict = None if lowered_key is None else tuning.stored_verdict(target, lowered_key)
    if isinstance(verdict, dict):
        return _kept_certification(key, op, config, reverse, verdict, (stored,), device, read=True).refused
    unstored = (stored, lowered_key)
    judged = _cache.KERNEL_JUDGED.value
    try:
        # Compiled under its judge's watch (`Judge.compiling`), which reads what this compile alone prints.
        with contextlib.nullcontext() if judged is None else judged[1].compiling():
            kernels = _launch_binaries(lowered, modules)
    except jax.errors.JaxRuntimeError as error:
        _uncertified_launch(key, op, config, error)
        return None
    return _kept_certification(key, op, config, reverse, kernels, unstored, device).refused


def _traced_for(device) -> bool:
    """Whether the trace calling a launch lowers its Mosaic GPU kernels for `device`: the architecture
    Mosaic GPU infers where the trace runs (`mosaic.gpu.utils.get_arch`, from its abstract mesh's device or
    the default device, Hopper's where neither states one) is the device's capability, so its call is
    replayed for `device` (`capture`). A trace lowering for another (an A100 step traced on its host with its
    generation assumed, whose kernels took Hopper's TMA copies and failed to lower for sm80,
    FS-ASIDE-EVERY80C) certifies its launch in turn, the launch traced for the device (`_lowered_launch`)."""
    from jax.experimental.mosaic.gpu import utils as mosaic_utils

    capability = generation.capability(generation.device_target(device).generation)
    if capability is None:  # no GPU's: a TPU's launches lower for the device the trace names
        return True
    arch = mosaic_utils.get_arch()
    return (arch.major, arch.minor) == capability


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
    picks' store, else compiled here (as the build judging them assembles them, `judging`), their verdict
    stored for the GPU model by each of `stored`'s keys that found none (None where no key states the
    installation's code)."""
    from dew.nn.kernels import tuning

    if not read and (judged := _cache.KERNEL_JUDGED.value) is not None:
        kernels = judged[1](kernels)
    for each in stored:
        if each is not None:
            tuning.store_verdict(verdict_target(device), each, kernels)
    found = Certification(type(op).__name__, config, reverse, kernels, read, device)
    with _GUARD:
        _CERTIFIED[key] = found
    if not found.refused:
        _progressed(key)
    return found


def _progressed(key: tuple) -> None:
    """A launch certified at a config that holds (`key`): its settle's misses past it stepped."""
    if (settlement := _SETTLING.get()) is not None:
        settlement.certified(key)


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
    Pallas lowering cache's seal), the assemblers of its kernels' kinds (`assemblers.named`: another ptxas's
    spills and serialized products are another verdict, row 10782), and the environment a
    kernel's compile reads (that cache's policy); the store's file names the GPU model. Its kernels decide its
    verdict, not the Dew tree that lowered them: a tree whose own key (`_verdict_key`) finds none, an edit's
    anywhere outside them, lowers the launch and reads its verdict, compiling nothing (row 8650: 23 of an
    edited tree's 51 seconds certifying a hybrid decoder's step ahead, FS-8650-SPLIT). None where the
    installation has no seal, its kernels' IR read and hashed for none (row 12009), or an assembler is
    unknown."""
    def stated() -> tuple | None:
        kernels = sorted((call.kind, call.name, hashlib.sha256(call.ir).hexdigest(), call.config)
                         for call in calls if call.kind in ("mosaic_gpu", "triton"))
        if (assembling := assemblers.named({kind for kind, *_ in kernels})) is None:
            return None
        return "lowered", assembling, _verdict_fields(), kernels

    return _sealed_key(stated)


def stored_key(provenance: frozenset[str] | None, *fields) -> str | None:
    """What decides a stored launch decision (`attention_mosaic_derivation._choice_key`) for a compile
    target before it traces: its code (`dew._code.provenance_code`: an edit of it invalidates the stored
    entry), the installed JAX, jaxlib and CUDA plugins, whose Mosaic GPU and XLA lower and compile it (the
    Pallas lowering cache's seal of them), caller-specific `fields`, and the environment a kernel's lowering
    and compile read (that cache's policy). None where the installation has no seal (an edited or a source
    install of JAX), whose code no key states: decisions are not stored."""
    return _sealed_key(lambda: (provenance_code(__name__, provenance), *fields))


def _sealed_key(stated: Callable[[], tuple | None]) -> str | None:
    """A stored decision's key in the installation's sealed frame, the one owner of it (`stored_key`'s, a
    verdict's, row 11653): what `stated()` gives (its head, its code's provenance or what names its kind,
    then its fields) beside the installed JAX, jaxlib and CUDA plugins (the Pallas lowering cache's seal of
    them) and the environment a kernel's lowering and compile read (that cache's policy); None where the
    installation has no seal, `stated` called only where it has one (row 12009), or where `stated` gives
    None."""
    from jax._src.pallas.mosaic_gpu.lowering_cache import _keyed_environment
    from jax._src.pallas.mosaic_gpu.lowering_cache_seal import seal

    installed = seal().digest
    if installed is None or (state := stated()) is None:
        return None
    head, *fields = state
    keyed = [head, installed.hex(), *fields, _keyed_environment()]
    return hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()


def _verdict_key(identity: str, config: Hashable, reverse: bool, provenance: frozenset[str] | None,
                 target: str) -> str | None:
    """What decides a launch's certification for a compile target (`verdict_target`'s `target`,
    `tuning.stored_verdict`) before it traces: its code's provenance, as a stored decision's
    (`stored_key`), what a stored verdict states (`_verdict_fields`), and a CUDA target's assemblers of
    either kind of kernel its launch may compile (`assemblers.named`); another target's names none, a TPU's
    read on an install with no Mosaic GPU binding. Each is read only where the installation is sealed; None
    where it is not or an assembler is unknown."""
    from dew.nn.kernels.tuning import encoded

    def stated() -> tuple | None:
        kinds = ("mosaic_gpu", "triton") if generation.capability(target) else ()
        if (assembling := assemblers.named(kinds)) is None:
            return None
        return (provenance_code(__name__, provenance), assembling, _verdict_fields(), identity,
                encoded(config), reverse)

    return _sealed_key(stated)


def _verdict_fields() -> list[str]:
    """What a stored verdict states of a kernel, its `Binary`'s fields: a field added keys every verdict
    afresh, where one stored before it decoded without it (row 11796: `Binary.spilled`)."""
    return [field.name for field in dataclasses.fields(Binary)]


def certifications() -> tuple[Certification, ...]:
    """Each launch certified in this process (`certified`)."""
    with _GUARD:
        return tuple(_CERTIFIED.values())


@contextlib.contextmanager
def judging(build: str, judge: "Judge") -> Iterator[None]:
    """Within the block, each launch certified here is compiled under `judge`'s watch (`Judge.compiling`) and
    judged by it, which remakes its kernels' binaries as JAX's CUDA `build` assembles them (a compile gate's,
    from the PTX Mosaic printed as it compiled): its verdict that build's, certified apart from every other
    judge's (`certified`'s key, its identity), and a program traced within traces again
    (`dew._cache.KERNEL_JUDGED`), its derivations stepping by those verdicts as a GPU running that build
    steps (row 10766)."""
    with _cache.KERNEL_JUDGED((build, judge)):
        yield


class Judge(Protocol):
    """What certification judges its launches' kernels by within `judging`: a compile watched
    (`compiling`, a launch's alone, failing or not), then its kernels remade (`__call__`, of a compile that
    held)."""

    def compiling(self) -> contextlib.AbstractContextManager[None]: ...

    def __call__(self, kernels: dict[str, tuple[str | None, "Binary"]]
                 ) -> dict[str, tuple[str | None, "Binary"]]: ...


def uncertified() -> dict[tuple, str]:
    """Each launch certification did not compile alone in this process (`certified`), and why."""
    with _GUARD:
        return dict(_UNCERTIFIED)


def missed(op, arguments: tokamax.BoundArguments, config: Hashable, refused: Sequence["Binary"],
           *, stepped: bool) -> None:
    """Each of `refused`'s kernels (`certified`'s, at `config`) recorded as a miss (`misses`), as `settled`
    records a compile's: `stepped` where its launch's derivation took another config after its refusal
    (`refuse_launch`), not where it had none left."""
    launch = Launch(_identity(op, arguments, generation.gpu_target()), type(op).__name__, config)
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
        # JAX's own default for a plain call: the device named, else this process's first of the platform
        # named (`jax.local_devices`), never another process's (rows 7211, 7224: rank k certified on
        # process 0's device, which it cannot address).
        default = jax.config.jax_default_device
        devices = [default] if isinstance(default, jax.Device) else jax.local_devices(backend=default)[:1]
    else:
        devices = [device for device in placed.devices.flat if device.process_index == jax.process_index()]
    # A device of another generation (a CPU's, an A100's under an assumed sm90) certifies nothing.
    if not devices or generation.device_target(devices[0]).generation != generation.gpu_target().generation:
        return None
    return devices[0]


def _lowered_launch(op, arguments: tokamax.BoundArguments, config: Hashable, device, reverse: bool
                    ) -> tuple[jax.stages.Lowered, dict, str | None]:
    """`op`'s launch of `arguments` at `config` lowered alone, certified for `device` (as reverse mode runs it
    where `reverse`), beside its kernels' modules (a Triton call's config) by name, and their verdict's key
    (`_lowered_verdict_key`)."""
    from dew.nn.kernels.options import owned_program
    from dew.nn.programs import meshless

    mesh, manual = _certifying_placement(device)
    placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # A program of its own (`meshless`): none of the enclosing trace's mesh nor its axes, its operands' too,
    # but the mesh its program runs on where that holds `device` (`_certifying_placement`), its operands
    # whole on each device, within a map holding the axes manual its program's call holds manual
    # (`sharding.manual_map`'s).
    with meshless(mesh):
        launch, shapes = op.lowering(arguments, config)
        if manual:
            launch = jax.shard_map(launch, mesh=mesh, in_specs=jax.sharding.PartitionSpec(),
                                   out_specs=jax.sharding.PartitionSpec(), axis_names=manual, check_vma=False)
        specs = [jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=placed) for shape in shapes]

        def certified(*operands):
            return (reversed_run(launch) if reverse else launch)(*operands)

        identity = _identity(op, arguments, generation.gpu_target())
        certified.__name__ = certified.__qualname__ = _certification_name(type(op).__name__, identity, config,
                                                                          reverse)
        lowered = owned_program(certified).trace(*specs).lower()
    return lowered, *_lowered_kernels(lowered)


def reversed_run(launch: Callable) -> Callable:
    """`launch` as reverse mode runs it, the one construction certification compiles (`_lowered_launch`) and
    the tuner times (`tuning._measured`) a launch whose call carries its derivative by: its forward and its
    pullback at unit cotangents, differentiated along its inexact operands alone, the others closed over.
    Its outputs too, as a step reads them: dropped, the kernels writing only them were dead code to XLA and
    went uncertified (the 4B decision step's recurrence_output, LF-PROMO3, whose program then stepped its
    producer down a whole-step relowering a rung)."""
    def run(*operands):
        inexact = [jnp.issubdtype(jnp.result_type(operand), jnp.inexact) for operand in operands]

        def along(*varied):
            given = iter(varied)
            return launch(*(next(given) if moves else operand
                            for operand, moves in zip(operands, inexact, strict=True)))

        out, pullback = jax.vjp(along, *(operand for operand, moves in zip(operands, inexact, strict=True)
                                         if moves))
        return out, pullback(jax.tree.map(_unit_cotangent, out))

    return run


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
    # Under the trace's own states, its context variables and what holds per thread beside them
    # (`_cache.carried`: the GPU `generation.assumed` stands in for, which a pool thread holds none of, so a
    # replayed derivative's launches derived for no GPU of the trace's: an A100 step's TMA copies).
    held = _cache.carried()

    def replayed_held() -> tuple[dict, str | None, bool]:
        with held():
            return replayed(closed, mesh, each.reverse, each.device, name)

    compiling = aside.pool.submit(contextvars.copy_context().run, replayed_held)
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
    (`_lowered_launch`), and the axes its launch holds manual: its program's where that holds `device` (the
    concrete mesh, `jax.set_mesh`'s, one GPU's or many's, its axes manual where the call is in a map holding
    them, `sharding.manual_map`'s), so the launch lowers to the jaxpr and the mesh its program's call does
    (the GPU it states and each jit's recorded mesh among them) and the program reads its kernels' lowerings
    back (the Pallas lowering cache keys both: a shard's launch on two GPUs certified on a mesh of its own was
    lowered twice, rows 6327 and 12462), where the call holds every axis of more than one device manual, as a
    kernel's launch needs (`sharding.per_shard`: it reads its operands whole, each device its own rows); else
    a mesh of its own on `device`, as a program of no mesh, of other devices, or certified outside its call's
    map (a run's compile: a recurrence's states launch certified there on its program's two-device data axis,
    automatic, was refused, train 1275)."""
    from jax._src import mesh as mesh_lib

    placed = mesh_lib.get_concrete_mesh()
    manual = frozenset(jax.sharding.get_abstract_mesh().manual_axes)
    if (not placed.empty and device in placed.devices.flat
            and all(size == 1 or name in manual for name, size in placed.shape.items())):
        return placed, manual
    return jax.sharding.Mesh([device], ("launch",)), frozenset()


def _unit_cotangent(out: jax.Array) -> jax.Array | np.ndarray:
    """A cotangent of ones for `out`, float0's zeros for an integer output."""
    if jnp.issubdtype(out.dtype, jnp.inexact):
        return jnp.ones_like(out)
    return np.zeros(out.shape, jax.dtypes.float0)


def refusal_snapshot() -> tuple[tuple[str, Hashable, frozenset[Refused]], ...]:
    """Every (launch identity, config, spilled kernels with their roles, `Refused`) refused in this context (a
    `refusals` scope's, the process's, or the reading a trace reads, `reading`), ordered, read as bookkeeping
    and no fact: a program's keys of its executables and jits (`options.OuterProgram`), what a plan hands on.
    A trace reads a launch's own refusals (`refusals_of`), never the whole registry, so a held value is made
    again by a refusal of a launch it read alone (row 12296: the whole registry a trace's fact, any refusal
    made every held value and executable again). A validated plan can hand them to another process
    explicitly, the roles with them (`refuse` within `refusals`); a shared compilation cache alone must not
    transfer them between programs."""
    with _GUARD:
        refused = (_READING.get() or _registry()).refused
        return tuple(sorted(((identity, config, kernels) for (identity, config), kernels in refused.items()),
                            key=repr))


def refused_launches() -> tuple[tuple[str, Hashable, frozenset[str]], ...]:
    """`refusal_snapshot`, each launch's spilled kernels by their names: what a plan's record lists
    (`dew.training.planning.refusal_record`)."""
    return tuple((identity, config, names(kernels)) for identity, config, kernels in refusal_snapshot())


def refuse(entries) -> None:
    """Each `(identity, config, kernels)` of `entries` (`refusal_snapshot`', its kernels with their roles,
    `Refused`, or a pair a decoded record gives as a list; a kernel by its name alone states none) refused in
    this context, as a compile's spill refuses it. A handed plan seeds these inside its own `refusals` scope,
    not unioned with unrelated decisions that the receiving process already holds."""
    refused_within = False
    for identity, config, kernels in entries:
        refused_within |= _entered((identity, config), None, frozenset(map(_refused, kernels)))[1]
    if refused_within:
        raise ReadingRefused(tuple(entries))


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


def record_stored(outcome: StoredOutcome) -> None:
    """Record one stored derivation's lookup by its `outcome`: the cache's reading or writing, done whether or
    not a replacement it was made for is kept, recorded as the kept program's at once (`_STORED_RECORDING`,
    a proposal's registry sharing its parent's count, `_proposing`)."""
    if outcome not in get_args(StoredOutcome):
        raise ValueError(f"no stored derivation's outcome {outcome!r}")
    with _GUARD:
        for held in (_registry(), *_STORED_RECORDING.get()):
            held.stored[outcome] += 1


def stored() -> Mapping[StoredOutcome, int]:
    """Each stored derivation's lookup this process's programs made, by its outcome (`record_stored`)."""
    with _GUARD:
        return dict(_registry().stored)


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
    stored = _STORED_RECORDING.set((*_STORED_RECORDING.get(), recorded))
    try:
        yield recorded
    finally:
        _STORED_RECORDING.reset(stored)
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
    # A replacement's records reach the registry and a `recording` only once it is kept (`_commit`); its
    # stored derivations' outcomes, the cache's work done either way, reach them at once: its registry counts
    # them in its parent's count, and the scopes' `recording`s are left open to them (`record_stored`).
    proposal = Recorded()
    token = _SCOPE.set(_Registry(refused=proposed, stored=registry.stored))
    held = _RECORDING.set((proposal,))
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
    """`_entered`'s, its refusal ending a frozen reading's trace where it adds to the reading's refusals
    (`ReadingRefused`); whether `miss` is new."""
    new, refused_within = _entered(refusal, miss, kernels)
    if refused_within:
        raise ReadingRefused(refusal)
    return new


def _entered(refusal: tuple[str, Hashable] | None, miss: "Miss | None",
             kernels: frozenset[Refused] = frozenset()) -> tuple[bool, bool]:
    """`refusal` and `miss` recorded; whether `miss` is new to this context's registry. A miss is recorded
    once a holder, the registry and each `recording` open: a launch's derivation asked again (its route's
    eligibility, then its call, `tuning.KernelOp.taken`) reads its certification back and finds the same
    misses, and a recording opened since the first (a later plan's admission, `planning.admitted`) holds
    the misses its own trace meets, those the process met before among them. Beside it, whether `refusal`
    adds to a frozen reading's refusals (`reading`), which ends its trace; one it holds already does not, so
    a held jit's trace refusing again what it read goes on."""
    registry, read = _registry(), _READING.get()
    refused_within = False
    with _GUARD:
        if refusal is not None:
            if read is not None and read.frozen:
                held = read.refused.get(refusal, frozenset())
                refused_within = _joined(held | kernels) != held
            # Into the registry, and the reading a trace here reads, which reads its own (`reading`).
            for held in (registry, None if read is None or read.frozen else read):
                if held is not None:
                    held.refused[refusal] = _joined(held.refused.get(refusal, frozenset()) | kernels)
                    held.by_launch.pop(refusal[0], None)
        new = miss is not None and miss not in registry.missed
        if miss is not None:
            _add(missed=(miss,))
    return new, refused_within


def compiled(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None = None, *,
             relower: Callable[[], jax.stages.Lowered], first: "Attempt | None" = None,
             kept_from: list[jax.stages.Lowered] | None = None) -> jax.stages.Compiled:
    """`lowered` compiled (`kept`), its communicators admitted against the room the process's pool left
    them before it runs (`dew.communicators.admit`): every program Dew compiles passes here. `first` is
    its first compile where another thread made it (`attempts`); `kept_from` given, the lowering the kept
    executable was compiled from is appended to it (`settled`'s)."""
    executable = kept(lowered, options, relower=relower, first=first, kept_from=kept_from)
    communicators.admit(executable)
    return executable


def compiled_at_once(programs: Sequence[tuple[jax.stages.Lowered, jax.stages.CompilerOptions | None,
                                             Callable[[], jax.stages.Lowered]]],
                     kept_from: Sequence[list[jax.stages.Lowered]] | None = None
                     ) -> list[jax.stages.Compiled]:
    """Each of `programs`, a lowering, its options and its `relower`, compiled as `compiled` compiles it,
    their first compiles at once, each in a thread of its own (`attempted`): XLA's compile releases the
    GIL and runs one program on about one and a half cores, so two compiled at once took 1.94 times the
    programs a second of two compiled in turn, four 2.2 times, on an 8-core host (a Qwen3.5-4B step for a
    compile-only H100, FS-2440-CONCURRENT2). Each is then kept (`settled`) and admitted in turn, in this
    thread, in the order given, as `compiled` keeps them one after another: a spill's refusal commits,
    and its program relowers, under this thread's contexts, so the programs kept are the ones compiled in
    turn keep: a program lowered before an earlier one's refusals committed is lowered again past them and
    compiled in turn, as it would have been lowered after them. At most `compile_threads()` compile at
    once. `kept_from` given, each program's kept lowering is appended to its own list (`compiled`'s)."""
    firsts = attempts([(lowered, options) for lowered, options, _ in programs])
    held = refusal_snapshot()
    kept_programs = []
    lists = [None] * len(programs) if kept_from is None else kept_from
    for (lowered, options, relower), first, keeping in zip(programs, firsts, lists, strict=True):
        # A program launching a config an earlier one's compile refused after its trace took it, each program
        # by what it launches (row 8603); refusals held before every trace, its derivations read.
        if refusal_snapshot() != held and refused_in(lowered):
            lowered, first = relower(), None
        kept_programs.append(compiled(lowered, options, relower=relower, first=first, kept_from=keeping))
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
    """A `jax.errors.JaxRuntimeError` (which JAX names as a value, no type), or an `Untuned` or a
    `CompiledOnDevice` refusal of a compile for a device to run: each its own program's, so a caller of
    several (`attempts`) knows which raised it."""


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
    # `Untuned` and its `CompiledOnDevice`, the latter raised before XLA compiles, held as XLA's own errors
    # are (the review of 3606, F3: one escaped several compiles unnamed).
    except (jax.errors.JaxRuntimeError, Untuned) as error:
        return Attempt(None, error)


UNTUNED = ("No cached config found for HLO instr:", "Failed to get configs for:",
           "Missing deviceless CUB scratch size data")
"""What XLA says where a compile without a device, every tuned choice required from the autotune cache, meets
an instruction no device run has tuned yet (config_assigner.cc's NotFound, sort_rewriter.cc's CUB scratch)."""


class Untuned(RuntimeError):
    """A compile without a device for a device to run (`for_devices`: a job's prepare,
    `dew.training.targets.AheadTarget`) met an instruction no device run has tuned yet (`UNTUNED`): the
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
    each module's instructions read through XLA's own schema (`dew.telemetry.hlo.instructions`) and a call's
    kind by the one table of custom call targets (`instrumentation.custom_call_kind`)."""
    from dew.telemetry import hlo
    from dew.telemetry.instrumentation import custom_call_kind

    executable = compiled.runtime_executable()
    for module in [] if executable is None else executable.hlo_modules():
        for read in hlo.instructions(module):
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
            # The names that are a preset's, the ones stating an algorithm to compare ('HIGHEST' names a
            # `Precision`, which JAX's conversion reads by no such name).
            stated = algorithm_arithmetic(stablehlo.DotAlgorithm(attributes["algorithm"]))
            presets = jax.lax.DotAlgorithmPreset.__members__
            return any(stated == dot_arithmetic(presets[name], jnp.float32, jnp.float32)
                       for name in FP32_PRODUCTS if name in presets)
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
         read: Callable[[jax.stages.Compiled], list["Binary"]] | None = None,
         kept_from: list[jax.stages.Lowered] | None = None) -> jax.stages.Compiled:
    """`lowered` compiled and kept (`settled`), its first compile `first` or one in this thread."""
    return settled(lowered, options, relower, attempted(lowered, options) if first is None else first,
                   read=read, kept_from=kept_from)


def settled(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
            relower: Callable[[], jax.stages.Lowered], first: Attempt, *,
            read: Callable[[jax.stages.Compiled], list["Binary"]] | None = None,
            kept_from: list[jax.stages.Lowered] | None = None) -> jax.stages.Compiled:
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
    did not take named (`_untaken`). `kept_from` given, the lowering the kept executable was compiled from
    is appended to it: `lowered`, or the relowering it kept (a caller keys the executable by what that one
    traced, row 12165)."""
    if first.error is not None:
        asked = shared_asked(first.error)
        if asked is None:
            raise first.error
        return _unblocked(lowered, options, relower, asked, first.error, kept_from)
    executable = first.executable
    assert executable is not None
    found, compiled_from = _misses_of(_refused_binaries(lowered, executable, read)), lowered
    # The refusals this compile relowered past: a spill at a config another caller refused after this
    # program's trace took it (another thread's compile, a certification elsewhere) is relowered past as
    # any, never read as held, which kept a spilling program (row 8603).
    past: set[tuple[tuple[str, Hashable], str]] = set()
    while True:
        # Each step refuses a config no step of this compile did, of the finitely many its launches derive.
        refusals = [(refusal, (report.kernel, report.spilled_in)) for refusal, report, _ in found
                    if refusal is not None and (refusal, report.kernel) not in past]
        if not refusals:
            break
        past.update((refusal, kernel) for refusal, (kernel, _) in refusals)
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
    if kept_from is not None:
        kept_from.append(compiled_from)
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
        compiled[name] = Alone(tuple(carried(executable.serialize())))
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
    verdicts: dict[tuple[str, Hashable, bool, frozenset[str] | None], dict[str, tuple[str, Binary]]] = {}
    for refusal, binary, launch in found:
        if refusal is None or launch is None:
            continue
        identity, config = refusal
        name = launch_name(binary.kernel, binary.launch)
        verdicts.setdefault((identity, config, launch.reverse, launch.provenance), {})[name] = (
            modules.get(name, ""), binary)
    target = verdict_target(device)
    for (identity, config, reverse, provenance), kernels in verdicts.items():
        if (key := _verdict_key(identity, config, reverse, provenance, target)) is not None:
            tuning.store_verdict(target, key, kernels)


def _launches(program: jax.stages.Lowered) -> set[tuple[str, Hashable]]:
    """Each launch `program` makes, its identity and config (`called`): every call JAX compiles, a branch's
    no run takes among them (a forward's scaled op's)."""
    from dew.nn.kernels.lowered import kernel_calls

    return {(launch.identity, launch.config) for call in kernel_calls(program, unrun=True)
            if (launch := called(call)) is not None}


def refused_in(program: jax.stages.Lowered) -> set[tuple[str, Hashable]]:
    """The launches `program` makes at a config refused in this context (`refusal_snapshot`): a refusal
    committed after its trace took the config (another program's compile, another thread's certification),
    past which it is lowered afresh before it compiles (row 8603)."""
    refused = {(identity, config) for identity, config, _ in refusal_snapshot()}
    return _launches(program) & refused if refused else set()


def _untaken(program: jax.stages.Lowered, refusals: list[tuple[tuple[str, Hashable], Refused]]
             ) -> list[tuple[tuple[str, Hashable], Refused]]:
    """Each of `refusals` (a launch's identity and config, and its spilling kernel, `Refused`) that `program`,
    relowered under them, launches again: its launch's derivation did not read the refusal. Each is logged,
    so the launch whose rule ignores its refusals is named."""
    launched = _launches(program)
    untaken = [(refusal, kernel) for refusal, kernel in refusals if refusal in launched]
    for (_, config), (kernel, _) in untaken:
        _log.warning("kernel %s's refusal at %s did not take: the program relowered under it launches it "
                     "again, its launch's derivation not reading it", kernel, config)
    return untaken


def _unblocked(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
               relower: Callable[[], jax.stages.Lowered], asked: int, error: Exception,
               kept_from: list[jax.stages.Lowered] | None = None) -> jax.stages.Compiled:
    """The program XLA refused for a Triton kernel asking `asked` shared bytes past a block, compiled once
    more with the configs of the Triton launches XLA refuses each compiled alone refused (`refused_alone`,
    each its next derived config, as a spill refuses one, `kept`), or every Triton launch's where none is
    refused alone: XLA's refusal names no kernel, and a search compiling the whole program a launch at a
    time costs a program's compile a launch (row 3584), where a kernel's call compiled alone costs its own
    (audit group 71). Each launch's miss is recorded and logged at the bytes it asked; where the relowered
    program is refused again, or a derivation finds no config left (`Exhausted`), the refusal stands, each
    a miss not stepped. `kept_from` given, the relowering it keeps is appended to it (`settled`'s)."""
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
            program = relower()
            replacement, spilling = _compiled_reported(program, options)
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
    if kept_from is not None:
        kept_from.append(program)
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
                    None if launch is None else launch.config, report.registers, report.spills,
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
