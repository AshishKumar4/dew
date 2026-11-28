"""JAX's events of a trace, a lowering and a backend compile, and their one listener in this process: the
blocks that count the programs jitted functions need (`Compiles`) and the threads held to running loaded
programs alone (`executable_only`)."""

import contextlib
import contextvars
import inspect
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import ClassVar

import jax

from dew import _cache


def executable_platform(executable: jax.stages.Compiled) -> str:
    """The backend a compiled `executable` targets ('cpu', 'cuda', 'rocm', 'tpu'): a loaded one's devices'
    (`communicators.loaded_program`, its runtime's, no tree of its rebuilt), else one compiled for devices
    this process has not (a compile-only GPU's) read from its arguments' and results' placements, whose
    reading rebuilds its trees (whose nodes may refuse leaves of another kind: a dataclass validating its
    fields); "none" for one that places neither."""
    from dew.communicators import loaded_program

    loaded = loaded_program(executable)
    if loaded is not None and loaded.devices:
        return loaded.devices[0].platform
    placed = [*jax.tree.leaves(executable.input_shardings), *jax.tree.leaves(executable.output_shardings)]
    return next(iter(placed[0].device_set)).platform if placed else "none"


class Span:
    """A compile through Dew's compile point (`dew.nn.kernels.options.OuterProgram`) and the work JAX does
    for it, wherever its context runs (a thread it is copied to among them): the program it compiles
    (`called`, `OuterProgram.called`), whether it built its own executable (`built`: compiled or read from
    JAX's persistent cache, not handed by its fingerprint, `dew.nn.kernels.options.taking`), and that
    executable once compiled (`executable`), which states the backend it targets (`platform`)."""

    __slots__ = ("built", "called", "executable")

    def __init__(self, called: str) -> None:
        self.called = called
        self.built = False
        self.executable: jax.stages.Compiled | None = None

    @property
    def platform(self) -> str | None:
        """The backend its executable targets (`executable_platform`), read where a ledger asks; None before
        it is compiled."""
        return None if self.executable is None else executable_platform(self.executable)


class Owner:
    """A block whose work JAX's events are told by (`owning`): equal to itself alone."""


_SPAN: contextvars.ContextVar[Span | None] = _cache.untraced(
    contextvars.ContextVar("dew_compile_span", default=None))
_OWNER: contextvars.ContextVar[Owner | None] = _cache.untraced(
    contextvars.ContextVar("dew_compile_owner", default=None))


@contextlib.contextmanager
def _held_as[Held](variable: contextvars.ContextVar[Held | None], held: Held) -> Iterator[Held]:
    """`variable` set to `held` within the block, and `held` yielded."""
    token = variable.set(held)
    try:
        yield held
    finally:
        variable.reset(token)


def spanning(called: str) -> contextlib.AbstractContextManager[Span]:
    """Within the block, the work JAX does in this context is the compile of the program `called` the
    `Span` it yields stands for: each program a block hears there carries it (`Program.span`). Dew's
    compile point opens one for exactly the span of a compile (`OuterProgram._compiled_digested`)."""
    return _held_as(_SPAN, Span(called))


def owning() -> contextlib.AbstractContextManager[Owner]:
    """Within the block, the work JAX does in this context, and in every context copied from it (a compile
    begun on a thread of its own, a certification's), is the block's own: each program a block hears there
    carries the `Owner` it yields (`Program.owner`), which another block's work, a concurrent run's or
    another thread's own, never does. What a host step's startup and its steps read their own compiles by
    (`dew.training.compiler.HostStartup`, `HostRun`)."""
    return _held_as(_OWNER, Owner())


@dataclass(frozen=True)
class Program:
    """One program a block needed: `traced` (its jaxpr) or `compiled` (by the backend, or fetched
    from the persistent compilation cache), its function's name, its seconds, where it was asked
    for, the innermost frame outside JAX's and this module's, the thread that made it, which a
    compile at once runs on its own (`report.compiled_at_once`), the compile through Dew's compile point it
    was work of (`spanning`), and the block whose work it was (`owning`), None where none's."""

    kind: str
    name: str
    seconds: float
    where: str
    thread: str = ""
    span: Span | None = None
    owner: Owner | None = None

    @property
    def called(self) -> str:
        """Its function's name as its trace names it, a backend compile's `jit(<name>)` unwrapped."""
        return self.name[4:-1] if self.name.startswith("jit(") and self.name.endswith(")") else self.name


EVENTS: Mapping[str, str] = {"/jax/core/compile/jaxpr_trace_duration": "traced",
                              "/jax/core/compile/jaxpr_to_mlir_module_duration": "lowered",
                              "/jax/core/compile/backend_compile_duration": "compiled"}
"""JAX's events of a trace, a lowering and a backend compile (`jax.monitoring`'s durations), by what each is.
This module's one listener (`_heard`) hears them for every open `Compiles` and `executable_only` block."""

_open: list["Compiles"] = []
"""The `Compiles` blocks open in this process, which `_heard` tells each event."""
_open_lock = threading.Lock()


class ExecutableOnly(RuntimeError):
    """A trace, lowering or compile in a process held to running loaded programs alone (`executable_only`)."""


class _Held(threading.local):
    """How many `executable_only` blocks this thread is within."""

    depth = 0


_held = _Held()


def _heard(event: str, duration_secs: float, **metadata: str | int) -> None:
    """`EVENTS`' one listener: an event is told to every open `Compiles`, as JAX records it once the work is
    done, and then raises `ExecutableOnly` on a thread within `executable_only`. It is told under the lock a
    block closes under, so a block closes once the events told to it are held."""
    kind = EVENTS.get(event)
    if kind is None:
        return
    with _open_lock:
        for compiles in _open:
            compiles.heard(kind, duration_secs, metadata)
    if _held.depth:
        raise ExecutableOnly(f"{event} in a process held to loaded programs alone (executable_only): a "
                             "program compiled ahead was missing, or code traced around it")


_listener_lock = threading.Lock()
_listener: list[bool] = []
"""Whether `_heard` is registered with JAX, which appends a listener each time it is given one."""


def _listening() -> None:
    """`_heard` registered with JAX once a process, however many threads first open a block at once."""
    with _listener_lock:
        if not _listener:
            jax.monitoring.register_event_duration_secs_listener(_heard)
            _listener.append(True)


@contextlib.contextmanager
def executable_only() -> Iterator[None]:
    """Within the block, on this thread, a trace, a lowering or a backend compile raises `ExecutableOnly` as
    JAX records it (`EVENTS`), and a loaded program (`dew.training.ahead.AheadPrograms.loaded`) runs: a GPU
    node measuring programs compiled ahead stops where one is missing (ledger 2716). JAX records an event as
    its step ends, so this stops a run, not the step itself; that no compile ran anywhere in a run, children
    included, is its compilers' own dumps left empty (`dew.training.ahead.dumps`, `compiled_in`)."""
    _listening()
    _held.depth += 1
    try:
        yield
    finally:
        _held.depth -= 1


class Compiles:
    """Counts the programs jitted functions named `name` need while it is open (every function's with
    None), each compiled or fetched from the persistent compilation cache, as JAX reports them
    (`EVENTS`' backend compile, which names a function `jit(<name>)`): what a warm-up costs, or a compile
    a measured run should not have paid. `programs` holds every program it heard, the ones traced beside
    the ones compiled.

        with Compiles("_stepped") as compiles:
            server.run()
        assert compiles.count == 0
    """

    COUNTED: ClassVar[frozenset[str]] = frozenset({"traced", "compiled"})
    """The kinds of `EVENTS` it holds a program for."""

    def __init__(self, name: str | None) -> None:
        self.name = name
        self.programs: list[Program] = []
        """Each program it heard, in the order they were needed."""

    @property
    def seconds(self) -> list[float]:
        """Each compiled program's seconds, in the order they were needed."""
        return [program.seconds for program in self.programs if program.kind == "compiled"]

    @property
    def count(self) -> int:
        return len(self.seconds)

    def __enter__(self) -> "Compiles":
        _listening()
        with _open_lock:
            _open.append(self)
        return self

    def __exit__(self, *_) -> None:
        with _open_lock:
            _open.remove(self)

    def heard(self, kind: str, duration_secs: float, metadata: Mapping[str, str | int]) -> None:
        """An event of kind `kind` (`EVENTS`') that took `duration_secs`, as JAX's metadata names it."""
        # A trace names its function bare, a backend compile `jit(<name>)`.
        named = self.name is None or metadata.get("fun_name") in (self.name, f"jit({self.name})")
        if kind in self.COUNTED and named:
            # Its callers' frames walked bare from this one's caller, no source line read for any
            # (`traceback.extract_stack` stats each frame's file and reads its line: up to 0.9 s a step of
            # compile events, row 5556); unbounded, whatever `sys.tracebacklimit` limits a traceback to.
            frame = inspect.currentframe()
            frame = None if frame is None else frame.f_back
            while frame is not None and ("/jax/" in (name := frame.f_code.co_filename) or "/jaxlib/" in name
                                         or name == __file__ or "contextlib" in name):
                frame = frame.f_back
            where = "?" if frame is None else f"{frame.f_code.co_filename}:{frame.f_lineno}"
            self.programs.append(Program(kind, str(metadata.get("fun_name")), duration_secs, where,
                                         threading.current_thread().name, _SPAN.get(), _OWNER.get()))


class Ledger(Compiles):
    """Every program JAX traces, lowers or compiles while it is open, each kind of its events held: the
    one ledger of what work took (a plan's, `dew.training.planning.spending`; a host step's startup's,
    `dew.training.compiler.HostStartup`)."""

    COUNTED = frozenset(EVENTS.values())
