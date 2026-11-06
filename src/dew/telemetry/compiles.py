"""JAX's events of a trace, a lowering and a backend compile, and their one listener in this process: the
blocks that count the programs jitted functions need (`Compiles`) and the threads held to running loaded
programs alone (`executable_only`)."""

import contextlib
import threading
import traceback
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import ClassVar

import jax


@dataclass(frozen=True)
class Program:
    """One program a block needed: `traced` (its jaxpr) or `compiled` (by the backend, or fetched
    from the persistent compilation cache), its function's name, its seconds, where it was asked
    for, the innermost frame outside JAX's and this module's, and the thread that made it, which a
    compile at once runs on its own (`report.compiled_at_once`)."""

    kind: str
    name: str
    seconds: float
    where: str
    thread: str = ""


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
            frame = next((frame for frame in reversed(traceback.extract_stack()[:-1])
                          if "/jax/" not in frame.filename and "/jaxlib/" not in frame.filename
                          and frame.filename != __file__ and "contextlib" not in frame.filename), None)
            where = "?" if frame is None else f"{frame.filename}:{frame.lineno}"
            self.programs.append(Program(kind, str(metadata.get("fun_name")), duration_secs, where,
                                         threading.current_thread().name))
