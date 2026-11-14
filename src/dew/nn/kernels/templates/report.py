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
import json
import logging
import re
import shutil
import struct
import subprocess
import tempfile
import threading
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
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

from dew import communicators
from dew._devices import deviceless, xla_flag
from dew.nn.kernels import generation

if TYPE_CHECKING:
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
    kernels: dict[str, "RegisterModel"] = dataclasses.field(default_factory=dict, compare=False)
    """Each kernel the launch traced that stated its register model as it was named (`named`'s `model`), by
    its name less the launch's suffix (`kernel_of`'s): the launch's stated register allocation, whose
    producer the compile gate checks its emission lowers to."""


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


@dataclasses.dataclass
class Recorded:
    """The misses, rewrites and pass timings recorded within a `recording` scope, every one: one search's
    own, which its plan reports."""
    missed: list["Miss"] = dataclasses.field(default_factory=list)
    rewritten: list = dataclasses.field(default_factory=list)
    passes: list[float] = dataclasses.field(default_factory=list)


_RECORDING: contextvars.ContextVar[tuple[Recorded, ...]] = contextvars.ContextVar("recording", default=())
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
def named(name: str, *, model: "RegisterModel | None" = None) -> str: ...


@overload
def named(name: None, *, model: "RegisterModel | None" = None) -> str | None: ...


def named(name: str | None, *, model: "RegisterModel | None" = None) -> str | None:
    """A kernel's name carrying the launch tracing it (`launching`; `kernel` for an unnamed one), as is
    where none is; its register `model`, where given, recorded on that launch (`Launch.kernels`)."""
    digest = launch_digest()
    if digest is None:
        return name
    kernel = name or "kernel"
    if model is not None:
        with _GUARD:
            _LAUNCHES[digest].kernels[kernel] = model
    return launch_name(kernel, digest)


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

    @property
    def spilling(self) -> tuple["Binary", ...]:
        """Its kernels that spill as the GPU assembles them."""
        return tuple(binary for _, binary in self.kernels.values() if binary.frame)


_CERTIFIED: dict[tuple, Certification] = {}
"""Each launch certified (`certified`), by its identity, config, device and product."""

_UNCERTIFIED: dict[tuple, str] = {}
"""Each launch certification could not compile alone (`certified`), by its identity, config, device and
product: why."""


_PULLED: contextvars.ContextVar = contextvars.ContextVar("certified_pulled", default=False)
"""Whether a launch traced here runs as reverse mode runs it (`pulled`)."""


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


def certified(op, arguments: tokamax.BoundArguments, config: Hashable) -> tuple["Binary", ...]:
    """`op`'s kernels at `config` for `arguments` that spill as the GPU assembles them, each launch compiled
    alone before its program is: its kernels' binaries (`binaries`) as the plugin's own compiler makes them
    for the device the process compiles for (`_certifying_device`: its own GPU, or a compile-only one a
    block names), which the program's compile then reuses (Mosaic compiles a module once a process). A
    launch's derivation refuses a config that spills and takes its next (`tuning`'s `_chosen`), with no
    program retraced, where a compile's refusal retraces the whole program (`settled`): a B200's 4B step
    stepped its attention forward 19 times that way, 42 minutes, every rewalk a step's trace. None on a
    process compiling for no GPU or an interpreted launch; none for a launch XLA or the GPU's compiler
    refuses to compile alone though it lowered (a collective's, its devices the program's; an assembly the
    PTX compiler rejects), which its program's compile reads as before (`settled`) and `uncertified` names;
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
            return _CERTIFIED[key].spilling
    stored = _verdict_key(op, arguments, config, reverse)
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
            return ()
        if stored is not None:
            tuning.store_verdict(device.device_kind, stored, kernels)
    found = Certification(type(op).__name__, config, reverse, kernels, verdict is not None)
    with _GUARD:
        _CERTIFIED[key] = found
    return found.spilling


def _verdict_key(op, arguments: tokamax.BoundArguments, config: Hashable, reverse: bool) -> str | None:
    """What decides a launch's certification on a GPU model before it traces, as its stored verdict's key
    (`tuning.stored_verdict`): the launch's identity (`_identity`, the same in every process), its config,
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
    stated = [package_digest(), installed.hex(), _identity(op, arguments), encoded(config), reverse,
              _keyed_environment()]
    return hashlib.sha256(json.dumps(stated, sort_keys=True).encode()).hexdigest()


def certifications() -> tuple[Certification, ...]:
    """Each launch certified in this process (`certified`)."""
    with _GUARD:
        return tuple(_CERTIFIED.values())


def uncertified() -> dict[tuple, str]:
    """Each launch certification did not compile alone in this process (`certified`), and why."""
    with _GUARD:
        return dict(_UNCERTIFIED)


def refuse_spilling(op, arguments: tokamax.BoundArguments, config: Hashable,
                    spilling: Sequence["Binary"]) -> None:
    """`op`'s launch of `arguments` at `config` refused as certification found `spilling` (`certified`):
    refused as a compile's spill refuses it (`refuse_launch`), each kernel's miss recorded, its launch
    stepped down (`misses`), as `settled` records a compile's."""
    refuse_launch(op, arguments, config, {binary.kernel for binary in spilling})
    launch = Launch(_identity(op, arguments), type(op).__name__, config)
    _recorded([(None, binary, launch) for binary in spilling], stepped=True)


def _certifying_device():
    """The device a launch is certified on (`certified`): a block's (`certifying`), else this process's GPU
    where it compiles for that GPU's generation, else none."""
    named = _CERTIFYING.get()
    if named is not None:
        return named
    if jax.default_backend() != "gpu":
        return None
    if generation.device_generation() != generation.gpu_target().generation:
        return None
    return jax.local_devices()[0]


def _compiled_launch(op, arguments: tokamax.BoundArguments, config: Hashable, device,
                     reverse: bool) -> dict[str, tuple[str, "Binary"]]:
    """`op`'s launch of `arguments` at `config` as a resource derivation lowers it (`KernelOp.lowering`),
    compiled alone on `device`, as a call without a gradient runs it or, `reverse`, as reverse mode does (its
    forward and its pullback of a unit cotangent): its Mosaic GPU kernels' modules and binaries, by name and
    launch, none for a launch of none. A launch that does not trace alone raises: certification is no
    optional pass."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew.nn.kernels.lowered import kernel_calls
    from dew.nn.kernels.options import owned_program
    from dew.nn.programs import meshless

    mesh = Mesh([device], ("launch",))
    placed = NamedSharding(mesh, PartitionSpec())
    # A program of its own on one device (`meshless`): none of the enclosing trace's mesh nor its axes (a
    # checked shard_map's body, whose abstract values the launch's own must not vary over), its operands' too.
    with meshless(), jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        launch, shapes = op.lowering(arguments, config)
        specs = [jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=placed) for shape in shapes]

        def pulled_back(*operands):
            out, pullback = jax.vjp(launch, *operands)
            return pullback(jax.tree.map(_unit_cotangent, out))

        lowered = owned_program(pulled_back if reverse else launch).trace(*specs).lower()
    # A Mosaic GPU kernel compiled alone is the one its program loads (Mosaic compiles a module once a
    # process); a Triton kernel compiles again with each executable, so its program's compile reads it.
    modules = {launch_name(call.name, call.launch): call.config
               for call in kernel_calls(lowered, unrun=True) if call.kind == "mosaic_gpu"}
    if not modules:
        return {}
    return {name: (modules[name], binary)
            for binary in binaries(lowered.compile(targeted(lowered, None)))
            if (name := launch_name(binary.kernel, binary.launch)) in modules}


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
            kernels: frozenset[str] = frozenset()) -> None:
    registry = _registry()
    with _GUARD:
        if refusal is not None:
            registry.refused[refusal] = registry.refused.get(refusal, frozenset()) | kernels
        if miss is not None:
            _add(missed=(miss,))


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
        return Attempt(lowered.compile(targeted(lowered, options)))
    except jax.errors.JaxRuntimeError as error:
        return Attempt(None, error)


def targeted(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None
             ) -> jax.stages.CompilerOptions | None:
    """`options` for the target `lowered` compiles for: on compile-only GPUs (`deviceless`) nothing runs to
    autotune a product on, and XLA's config assigner already gives each matmul its default config there
    while its autotuner still compiles the candidates it cannot
    time, and a GEMM fusion whose candidates are cuBLASLt's fails ("CublasLtBackend cannot enumerate
    configs in deviceless mode"), so the program skips them (`xla_gpu_autotune_level` 0), its memory the
    same (a frontier search's every probe and rung, 142 s to 99 s on armada, 11-02). A run's own XLA_FLAGS
    setting, and devices that run, keep `options` as given. Every program compiled here takes it, a
    run's step and every program it compiles ahead alike."""
    from jax._src.interpreters import pxla

    lowering = lowered._lowering
    devices = lowering._device_list if isinstance(lowering, pxla.MeshComputation) else None
    device = None if devices is None else next(iter(devices))
    if (device is None or device.platform != "gpu" or not deviceless(device)
            or xla_flag("xla_gpu_autotune_level") is not None):
        return options
    return {**(options or {}), "xla_gpu_autotune_level": 0}


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
    found = _misses_of(_spilling(lowered, executable, read))
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
            raise Exhausted(f"kernels {kernels} spill and their launches have no config left: {failed}"
                            ) from failed
        if replacement is None:
            break
        _commit(proposal)
        for refusal, kernel in [*refusals, *((refusal, report.kernel) for refusal, report, _ in alone)]:
            _record(refusal, None, frozenset({kernel}))
        _recorded([*found, *alone], stepped=True)
        executable, found = replacement, _misses_of(spilling)
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

    from dew.nn.kernels.lowered import kernel_call

    # The compile's own backend, devices and options: a jit's lowering holds them (`pxla.MeshComputation`);
    # `Lowered` states none of them.
    lowering = lowered._lowering
    if not isinstance(lowering, pxla.MeshComputation) or lowering._device_list is None:
        raise TypeError(f"kernels are compiled alone from a jit's lowering over its devices, not "
                        f"{type(lowering).__name__}")
    backend, device = lowering.compile_args["backend"], next(iter(lowering._device_list))
    overrides = dict(lowering._compiler_options_kvs + tuple((options or {}).items()))
    module = lowering.stablehlo()
    calls: dict[str, ir.Operation] = {}

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
                            calls.setdefault(launch_name(called.name, called.launch), inner)
                    walk(inner)

    walk(module.operation)
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
            launch = traced(call.launch or kernel_of(call.name)[1] or "")
            return launch is not None and launch.identity in stepped

        alone = compiled_alone(program, options, wanted)
        if alone is None or any(each.error is not None for each in alone.values()):
            return program, missed
        held = _registry().refused
        found = [(refusal, report, launch)
                 for refusal, report, launch in _misses_of([binary for each in alone.values()
                                                            for binary in each.binaries if binary.frame])
                 if refusal is not None and report.kernel not in held.get(refusal, frozenset())]
        if not found:
            return program, missed
        for refusal, report, _ in found:
            _record(refusal, None, frozenset({report.kernel}))
        missed += found


def _untaken(program: jax.stages.Lowered, refusals: list[tuple[tuple[str, Hashable], str]]
             ) -> list[tuple[tuple[str, Hashable], str]]:
    """Each of `refusals` (a launch's identity and config, and its spilling kernel's name) that `program`,
    relowered under them, launches again: its launch's derivation did not read the refusal. Each is logged,
    so the launch whose rule ignores its refusals is named."""
    from dew.nn.kernels.lowered import kernel_calls

    # Every call JAX compiles, a branch's no run takes among them (a forward's scaled op's).
    launched = {(launch.identity, launch.config) for call in kernel_calls(program, unrun=True)
                if (launch := traced(call.launch or kernel_of(call.name)[1] or "")) is not None}
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
        kernel, digest = kernel_of(call.name)
        launch = None if call.kind != "triton" or digest is None else traced(digest)
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


def _compiled_reported(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
                       read: Callable[[jax.stages.Compiled], list["Binary"]] | None = None
                       ) -> tuple[jax.stages.Compiled, list["Binary"]]:
    executable = lowered.compile(targeted(lowered, options))
    return executable, _spilling(lowered, executable, read)


def _spilling(lowered: jax.stages.Lowered, executable: jax.stages.Compiled,
              read: Callable[[jax.stages.Compiled], list["Binary"]] | None = None) -> list["Binary"]:
    """The kernels of `executable`, `lowered` compiled, that spill, read by `read` (`binaries` where None),
    where it runs a template's (`reads_binaries`)."""
    if not reads_binaries(lowered):
        return []
    return [found for found in (read or binaries)(executable) if found.frame]


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
    version, field 1, whose binary, 8, is an ELF, whose kernel name, 9, its function's, and whose
    `SHARED_BYTES` its launch's dynamic shared memory), and how many Mosaic GPU custom calls it names
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
