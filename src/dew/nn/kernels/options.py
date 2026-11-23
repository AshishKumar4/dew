"""The XLA options a program compiles with, and the one place a program over a run's models is compiled
(`program`).

`dew.nn.kernels.generation` stays free of Dew's imports, since benchmark
environments load its target table without Dew or JAX; this reads its
facts with the run's own XLA flags.
"""

from __future__ import annotations

import collections
import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import json
import logging
import threading
import types
import weakref
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Protocol, TypedDict, Unpack

import jax
import numpy as np
from jax._src import api_util, config as jax_config, core as jax_core
from jax._src.array import ArrayImpl
from jax._src.interpreters import pxla
from jax.experimental.layout import Format
from jax.sharding import AbstractMesh, NamedSharding, Sharding

from dew import _cache, _facts, _stated, algebra, communicators
from dew._devices import xla_flag
from dew._facts import Facts, stated, traced_fact
from dew._stated import Unestablished
from dew.nn.kernels.generation import route_marks, withdrawn
from dew.nn.nested import staged
from dew.telemetry.benchmark import UNGRAPHED_FLAG, options_of
from dew.telemetry.compiles import Span, spanning

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from dew.nn.kernels.templates.report import ReadingRefused


type Option = bool | int | str
"""An XLA compiler option's value, as `jax.jit(..., compiler_options=)` takes it."""

_COMPILED_WITH: contextvars.ContextVar[tuple[Mapping[str, Option], ...]] = contextvars.ContextVar(
    "compiled_with", default=())
"""The options `compiled_with` holds open in this context, innermost last."""


@contextlib.contextmanager
def compiled_with(options: Mapping[str, Option]) -> Iterator[None]:
    """Every program Dew compiles while this is open adds `options` to its XLA options
    (`compiler_options`), each but those the run's XLA_FLAGS sets itself: the accuracy gate's seam,
    through which an arm that runs Dew's own compiled programs (a trainer's step, a task's forward)
    compiles them as the gate compiles a Dew arm it compiles itself (tests/reference_error.py,
    `dew_programs`). A program compiled before it opened keeps its own options. The options are the
    context's (a `ContextVar`): another thread, or a worker not handed this context, compiles
    without them."""
    token = _COMPILED_WITH.set((*_COMPILED_WITH.get(), dict(options)))
    try:
        yield
    finally:
        _COMPILED_WITH.reset(token)


def compiler_options() -> dict[str, Option]:
    """The XLA options a program Dew compiles takes beyond its target's (`report.targeted`, where a GEMM's
    backend is chosen from the program's own dots and the device it compiles for), for its
    `jax.jit(..., compiler_options=)`: those `compiled_with` holds open, each but those the run's
    XLA_FLAGS sets itself. JAX takes compiler options on a program's outermost jit alone, so a module's
    own jitted method runs under the options of the program that calls it. It reads no device: a process
    may form a program's options before it joins its peers (`jax.distributed.initialize`)."""
    options: dict[str, Option] = {}
    for opened in _COMPILED_WITH.get():
        options.update({name: value for name, value in opened.items() if xla_flag(name) is None})
    return options


type Placements = (Sharding | Format | None | tuple[Placements, ...] | Mapping[str, Placements]
                   | DataclassInstance)
"""Where a program's arguments or results sit, as `jax.jit` takes them: one sharding, one format (a
sharding and a layout), none (where they are), or a tree of them, one for each, a registered dataclass's
among them (a server's resident slots)."""


class JitArguments(TypedDict, total=False):
    """The arguments of `jax.jit` a program takes (`program`): its static arguments, and its placements,
    whose trees `program` holds by their structure and leaves, each hashable."""

    static_argnums: int | tuple[int, ...]
    static_argnames: str | tuple[str, ...]
    donate_argnums: int | tuple[int, ...]
    donate_argnames: str | tuple[str, ...]
    in_shardings: Placements
    out_shardings: Placements


_log = logging.getLogger("dew.kernels")

class Releasing(Protocol):
    """What keeps compiled programs alive for the process, or lowerings whose live modules key JAX's
    loaded executables of them: a program (`OuterProgram`) or a cache of lowerings (a trainer's kept traced
    steps, `dew.training._candidates`), which `release` asks to let them go."""

    def released(self) -> None: ...


_HOLDERS: weakref.WeakSet[Releasing] = weakref.WeakSet()
"""Every holder alive (`held_until_release`), whose executables and lowerings `release` drops."""
_HOLDERS_LOCK = threading.Lock()
"""Held as a holder is added and as `release` takes its snapshot of them: a set walked as another thread adds
to it raises (row 8411's review)."""


def held_until_release(holder: Releasing) -> None:
    """Register `holder` for `release`, held weakly: a program as it is made, a cache as it first keeps a
    lowering."""
    with _HOLDERS_LOCK:
        _HOLDERS.add(holder)

type Taken = Mapping[str, Callable[[], jax.stages.Compiled]]
"""Executables compiled ahead or kept by another process (`dew.training.ahead`), each loaded on demand, by
the fingerprint of the program it runs: its digest under the options it compiled with (`program_digest`)."""

_TAKEN: contextvars.ContextVar[Taken | None] = _cache.untraced(
    contextvars.ContextVar("dew_taken_programs", default=None))


def program_key(lowered: jax.stages.Lowered) -> Hashable:
    """A program as `lowered` holds it, within this process: its text's digest, its arguments' and results'
    trees as JAX holds them (a node's type the class itself, its data by equality), its arguments' types and
    the host callbacks it calls, the objects themselves (`host_callbacks`). Two lowerings of one program
    agree, whichever devices of one target they were lowered over, and two whose texts agree only as JAX
    prunes an argument neither reads do not, nor two that call other callbacks from one text (an Optax
    update's `jax.pure_callback` over another closure); any tree has one, a class made inside a function
    among its nodes. The key of every match within a process (the executables it holds across trainers, a
    begun compile's lowering); `program_digest` is the one a file of programs holds."""
    types = tuple((aval.shape, str(aval.dtype), aval.weak_type) for aval in jax.tree.leaves(lowered.in_avals))
    return _text_digest(lowered), lowered.in_tree, lowered.out_tree, types, host_callbacks(lowered)


def host_callbacks(lowered: jax.stages.Lowered) -> tuple:
    """The host callbacks `lowered` calls (`jax.pure_callback`, `jax.debug.callback`, an `io_callback`), the
    objects its executable calls, as JAX's own compile reads them from the lowering: its text names each by
    its place in the program alone, so two programs of one text may call other callbacks. JAX's jit keys
    them by the function traced, and its persistent cache keeps no program that calls one."""
    return tuple(lowered._lowering.compile_args.get("host_callbacks", ()))


def _text_digest(lowered: jax.stages.Lowered) -> str:
    return hashlib.sha256(lowered.as_text().encode()).hexdigest()


def program_digest(lowered: jax.stages.Lowered, options: jax.stages.CompilerOptions | None = None) -> str:
    """A program as `lowered` holds it, as a file of programs holds it across processes: its text's digest
    beside its arguments' and results' trees and its arguments' types, as `program_key` holds them within
    one, and the XLA options it compiles under where it states any (`options`): its fingerprint, the one
    thing a program handed an executable another process compiled is held to before it runs (`taken`). Two
    lowerings of one program agree, whichever devices of one target they were lowered over, and two whose
    texts agree only as JAX prunes an argument neither reads, which their executables take in other trees,
    do not (a step's commit with its sample's summaries and without). Refused (`dew._stated.Unestablished`)
    where a tree holds a node no process names (a class made inside a function), or the program calls a host
    callback (`host_callbacks`), an object of this process its text names by place alone: no file can hold
    that program, and each reader of a file's programs takes it as one the file holds none of (`taken`), or
    refuses it, naming why, where it must keep it (a step compiled ahead)."""
    if called := host_callbacks(lowered):
        raise Unestablished(f"the program calls {len(called)} host callbacks (`jax.pure_callback`, "
                            "`jax.debug.callback`), which no executable serializes, as JAX's persistent "
                            "cache keeps no program that calls one")
    # Its trees node by node, each node's data by value as an identity states it (`dew._stated.stated`), never
    # as JAX prints them, which holds a node's data by its printed form: a checkified program's error tree
    # holds a traceback printed by its address, another in every process. A handed executable gives its
    # results in its own output tree, so a node's data that labels its leaves (a key of no plain form) is
    # part of the program.
    types = jax.tree.map(lambda aval: (aval.shape, str(aval.dtype), aval.weak_type), lowered.in_avals)
    trees = json.dumps([_stated.stated(lowered.in_tree, "a program's arguments' tree"),
                        _stated.stated(lowered.out_tree, "a program's results' tree")], sort_keys=True)
    stated = "" if not options else json.dumps(sorted((name, repr(value)) for name, value in options.items()))
    return hashlib.sha256(f"{_text_digest(lowered)}\n{trees}\n{types}{stated}".encode()).hexdigest()


def program_name(lowered: jax.stages.Lowered) -> str:
    """A program's one name where it is counted rather than kept (a step no device run has tuned,
    `dew.training.ahead.CompiledAhead.untuned`): its digest as a file holds it (`stated_digest`), or its
    text's where its trees state no form (a host step's VJP results)."""
    return stated_digest(lowered) or _text_digest(lowered)


def stated_digest(lowered: jax.stages.Lowered,
                  options: jax.stages.CompilerOptions | None = None) -> str | None:
    """`lowered`'s digest under `options` as a file of programs holds it (`program_digest`), None (logged,
    with the tree) where its trees hold a node no process names: a program no file holds, so one no file's
    is, which a reader handed a file compiles (`taken`)."""
    try:
        return program_digest(lowered, options)
    except Unestablished as refusal:
        _log.info("a program no file of programs states is none a file holds: %s", refusal)
        return None


def taken(lowered: jax.stages.Lowered,
          options: jax.stages.CompilerOptions | None) -> Callable[[], jax.stages.Compiled] | None:
    """The executable a block hands (`taking`) for the program `lowered` is under `options`, by its
    fingerprint (`program_digest`), None where the block hands none or holds none for it: a program whose
    trees no file can hold (`stated_digest`) among them."""
    if (programs := _TAKEN.get()) is None or (digest := stated_digest(lowered, options)) is None:
        return None
    return programs.get(digest)


@contextlib.contextmanager
def taking(programs: Taken) -> Iterator[None]:
    """Within the block, a program compiled through Dew's compile point (`OuterProgram.compiled`) whose
    fingerprint is one of `programs` (`program_digest` under its options) runs that executable in place of
    compiling it, its own trace and lowering what proves it the same program: a run handed its start's
    programs (`dew.training.ahead.AheadOfTime.starts`), or a process handed the programs another kept
    (`dew.training.ahead.kept`), compiles none of them on its devices. One the block holds none of
    compiles as without it."""
    token = _TAKEN.set({**(_TAKEN.get() or {}), **programs})
    try:
        yield
    finally:
        _TAKEN.reset(token)


type Kept = Callable[["OuterProgram", jax.stages.Lowered, jax.stages.Compiled], None]
"""What hears each program the compile point compiles (`keeping`): the program, the lowering it compiled
(its last relowering's, past a spill) and its executable."""

_KEEPING: contextvars.ContextVar[tuple[Kept, ...]] = _cache.untraced(
    contextvars.ContextVar("dew_keeping_programs", default=()))


@contextlib.contextmanager
def keeping(kept: Kept) -> Iterator[None]:
    """Within the block, `kept` hears each executable Dew's compile point compiles (`OuterProgram.compiled`,
    `compiled_together`), with its lowering, on the thread that keeps it: what a save of a process's
    programs reads (`dew.training.ahead.keeping`). One taken, compiled elsewhere, is not heard. A block
    within another is heard by both, the outer first (`_heard`): a run's programs kept within a save of its
    process's are kept by each."""
    token = _KEEPING.set((*_KEEPING.get(), kept))
    try:
        yield
    finally:
        _KEEPING.reset(token)


def _heard(program: OuterProgram, lowered: jax.stages.Lowered, executable: jax.stages.Compiled) -> None:
    """`executable`, `program`'s compile of `lowered`, heard by every `keeping` block this thread is in."""
    for kept in _KEEPING.get():
        kept(program, lowered, executable)


@dataclasses.dataclass(frozen=True)
class Compiling:
    """A program's first compile running in a thread of its own (`compiling`): the program, what its call
    lowered from (`OuterProgram.abstract_of`, None for a plain jit) and its lowering, which the program's
    compile through Dew's compile point takes where it compiles the same program over the same, lowering
    nothing again, else where it lowers the same program (`program_key`) (`attempting`), and whether
    a call has taken its lowering (`taken`): a start that lowered again, its algebra passed twice, says
    so."""

    program: Callable
    abstract: Hashable | None
    lowered: jax.stages.Lowered
    attempt: Future
    weak_types: tuple[bool, ...]
    """Its trace's outputs' weak types, which its lowering's `out_info` drops (`_weak_types`)."""
    taken: threading.Event = dataclasses.field(default_factory=threading.Event, compare=False)
    refused: tuple | None = None
    """The launches refused as its trace traced (`report.refusal_snapshot`), a Dew program's; None for
    another's."""
    reads: Facts = dataclasses.field(default_factory=dict)
    """The facts its trace read (`_facts.recording`), a Dew program's."""


def compiling(program: OuterProgram, *args, **kwargs) -> Compiling:
    """`program` over these arguments (arrays or shapes) lowered here and its first compile begun in a
    daemon thread of its own (`report.attempted`: XLA's compile alone, which releases the GIL), so it
    compiles while this thread runs on: a run's start beside its step's planning, which does not read it.
    `attempting` takes it; the process's exit waits for it where it still runs, saying so
    (`dew._coordination.stop_at_exit`), so a run refused or stopped before it takes its start exits once the
    compile ends, never through a compile XLA's exit tears down under it."""
    from dew.nn.kernels.templates import report

    # Dew's program's trace beside the launches refused as it traced and the facts it read, by which its
    # begun lowering is taken (`_refusals_hold`).
    traced, refused, reads = (program._snapshot_trace(args, kwargs) if isinstance(program, OuterProgram)
                              else (program.trace(*args, **kwargs), None, {}))
    lowered = traced.lower()
    attempt: Future = Future()
    # In this thread's context: the compile's target reads it (`report.for_devices`).
    context = contextvars.copy_context()

    # Its compile runs inside jaxlib: the process's exit waits for it (`stop_at_exit`), so neither JAX's exit
    # cleanup nor the C runtime's exit tears XLA down under it (row 6837: 12 of 48 refused runs' exits ended
    # by a segmentation fault).
    from dew._coordination import stop_at_exit

    def said() -> None:
        _log.warning("the process waits to exit for a program's compile it began beside its work, still "
                     "running")

    def attempted() -> None:
        attempt.set_running_or_notify_cancel()
        try:
            attempt.set_result(context.run(report.attempted, lowered, None, background=True))
        except BaseException as error:
            # Its reader raises it (`attempting`).
            attempt.set_exception(error)
        finally:
            withdraw()

    thread = threading.Thread(target=attempted, name="dew-compile-ahead", daemon=True)
    withdraw = stop_at_exit(thread, said, timeout=None)
    thread.start()
    abstract = program.abstract_of(args, kwargs) if isinstance(program, OuterProgram) else None
    return Compiling(program, abstract, lowered, attempt, _weak_types(traced), refused=refused, reads=reads)


_ATTEMPTING: contextvars.ContextVar[Compiling | None] = _cache.untraced(
    contextvars.ContextVar("dew_attempting", default=None))


@contextlib.contextmanager
def attempting(begun: Compiling | None) -> Iterator[None]:
    """Within the block, a program compiled through Dew's compile point whose lowering is `begun`'s takes
    its first compile from that thread (`compiling`), waiting for it there, and keeps it as one it compiled
    itself (`report.compiled`'s `first`): the same program over the same abstract inputs takes the begun
    lowering too, tracing nothing again; any other compiles as it would. None begins nothing."""
    token = _ATTEMPTING.set(begun)
    try:
        yield
    finally:
        _ATTEMPTING.reset(token)


_COLLECTED: contextvars.ContextVar[dict[str, jax.stages.Compiled] | None] = _cache.untraced(
    contextvars.ContextVar("dew_collected_programs", default=None))


def shapes_of(tree):
    """`tree`'s arrays or shapes as their shapes, dtypes, shardings and weak types, as a call over arrays of
    those shardings lowers (`collecting`): an array no one committed to its devices (a fresh
    `jax.random.key`) as an unplaced shape, which jit places as it places the array (JAX's `pjit` lowers an
    uncommitted array's sharding as unspecified); a committed array's or a shape's device-local layout
    where it is not its devices' default (`_stated_layout`), which jit lowers as the array's own and its
    dispatcher holds the call to (a program compiled over another layout refuses the array, JAX's
    `check_array_xla_sharding_layout_match`), the default left unstated as an array's own."""
    def placed(leaf, layout):
        return leaf.sharding if layout is None else Format(layout, leaf.sharding)

    def shaped(leaf):
        if not isinstance(leaf, jax.Array):
            layout = _stated_layout(leaf.format.layout, leaf.sharding, leaf.shape, leaf.dtype)
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=placed(leaf, layout),
                                        weak_type=leaf.weak_type)
        if not leaf.committed:
            # Its abstract value's weak type: a key array states none of its own.
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=None, weak_type=leaf.aval.weak_type)
        # A typed key array states no layout (`_signature`).
        layout = (_stated_layout(leaf.format.layout, leaf.sharding, leaf.shape, leaf.dtype)
                  if isinstance(leaf, ArrayImpl) else None)
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=placed(leaf, layout),
                                    weak_type=leaf.aval.weak_type)

    return jax.tree.map(shaped, tree)


def _stated_layout(layout, sharding, shape, dtype):
    """`layout`, a device-local layout of an array of `shape` and `dtype` placed by `sharding`, where it is
    not those devices' default; None where it is, or none is stated (JAX's own reading,
    `pxla.is_default_layout`, as its dispatcher takes a default layout as unstated)."""
    if layout is None or pxla.is_default_layout(layout, sharding, jax_core.ShapedArray(shape, dtype)):
        return None
    return layout


@contextlib.contextmanager
def collecting() -> Iterator[dict[str, jax.stages.Compiled]]:
    """Within the block, a call of a program through Dew's compile point (`OuterProgram`) compiles it over
    its arguments, shapes where a caller hands shapes (`jax.ShapeDtypeStruct`), and answers with its
    results' shapes (`shapes_of`), running nothing: the mapping it yields holds each executable by its
    program's digest (`program_digest`). A run compiled ahead compiles so the programs it runs on its
    step's results, the code that runs them called as the run calls it (`dew.training.ahead.started`)."""
    collected: dict[str, jax.stages.Compiled] = {}
    token = _COLLECTED.set(collected)
    try:
        yield collected
    finally:
        _COLLECTED.reset(token)


_LOWERING: contextvars.ContextVar[tuple[tuple[str, ...], list[tuple[str, jax.stages.Lowered]]] | None] = (
    _cache.untraced(contextvars.ContextVar("dew_lowering_programs", default=None)))


@contextlib.contextmanager
def lowering(platforms: tuple[str, ...]) -> Iterator[list[tuple[str, jax.stages.Lowered]]]:
    """Within the block, a call of a program through Dew's compile point (`OuterProgram`) traces and lowers
    it over its arguments, shapes where a caller hands shapes, and answers with its results' shapes, as a
    call within `collecting` does, compiling and running nothing: one whose arguments sit on an abstract
    mesh (the devices of a target this host is not, `generation.gpu_mesh`) lowers for `platforms`, any
    other for its own devices' backend. The list it yields holds each lowering, in the order of the calls,
    with its program's name (`OuterProgram.called`): a step lowered for another target whose programs are
    several (a host step's, `Trainer(lower_for=)`)."""
    lowered: list[tuple[str, jax.stages.Lowered]] = []
    token = _LOWERING.set((platforms, lowered))
    try:
        yield lowered
    finally:
        _LOWERING.reset(token)


type Around = Callable[["OuterProgram", jax.stages.Compiled, tuple, dict], contextlib.AbstractContextManager]
"""What `running` runs each call's executable inside: given its program, the executable and the call's
arguments (its static ones among them), a context."""

_RUNNING: contextvars.ContextVar[tuple[Around, ...]] = _cache.untraced(
    contextvars.ContextVar("dew_running_programs", default=()))


def _ran(program: OuterProgram, executable: jax.stages.Compiled, args: tuple, kwargs: dict
         ) -> contextlib.AbstractContextManager:
    """The contexts every enclosing `running` block runs `executable`, `program`'s for these arguments,
    inside, the outermost first, each entered here, at the call, not at the caller's `with`, which leaves
    them; none outside any."""
    stack = contextlib.ExitStack()
    with stack:
        for around in _RUNNING.get():
            stack.enter_context(around(program, executable, args, kwargs))
        return stack.pop_all()


@contextlib.contextmanager
def running(around: Around) -> Iterator[None]:
    """Within the block, each executable a call through Dew's compile point runs over arrays (one its
    program keeps, one compiled over shapes, one taken, or one compiled at the call) runs inside
    `around(program, executable, args, kwargs)`, and inside every enclosing block's, the outermost first: an
    admission that refuses it by raising as it enters (a host run's rollout's, `dew.training.rollout
    .programs_admitted`), or a profiled trial's record of what it ran (`dew.training.planning.Trials
    .profiled_programs`); an executable fetched to be called later (`OuterProgram.executable`) as it is
    handed over. Nothing else of the call changes, and a call within `lowering`, which runs nothing, enters
    none."""
    token = _RUNNING.set((*_RUNNING.get(), around))
    try:
        yield
    finally:
        _RUNNING.reset(token)


def _abstractly_placed(tree) -> bool:
    """Whether a leaf of `tree` sits on an abstract mesh: a target's devices this process holds none of."""
    return any(isinstance(getattr(leaf, "sharding", None), NamedSharding)
               and isinstance(leaf.sharding.mesh, AbstractMesh) for leaf in jax.tree.leaves(tree))


type Relowered = list[tuple[jax.stages.Wrapped, jax.stages.Lowered, tuple, Facts]]
"""The relowerings a spill made of a call (`OuterProgram.compiled`'s), each with the jit it traced by, the
launches refused when it traced (`report.refusal_snapshot`) and the facts its trace read
(`_facts.recording`)."""

@dataclasses.dataclass
class _Lowering:
    """A call lowered within `compiled_together`: its program, the reservation its executable is kept by
    (its call's key and the refusals it began under, `OuterProgram._reserved`) and the compile it owns there
    (`OuterProgram._kept`'s future), its lowering, its arguments, and the relowerings a spill made of it."""

    program: OuterProgram
    key: tuple[Hashable, tuple]
    pending: Future[jax.stages.Compiled]
    lowered: jax.stages.Lowered
    args: tuple
    kwargs: dict
    refused: tuple
    """The launches refused as its lowering traced (`report.refusal_snapshot`)."""
    reads: Facts
    """The facts its lowering's trace read (`_facts.recording`)."""
    relowered: Relowered = dataclasses.field(default_factory=list)
    kept_from: list[jax.stages.Lowered] = dataclasses.field(default_factory=list)
    """The lowering its compile kept (`report.compiled`'s `kept_from`)."""


_TOGETHER: contextvars.ContextVar[dict[tuple[int, Hashable], _Lowering] | None] = _cache.untraced(
    contextvars.ContextVar("dew_programs_compiled_together", default=None))


@traced_fact
def rule_sets() -> dict[str, tuple]:
    """The algebra's registered rule sets, each by what it states plainly: its rounding class, the
    measurement it ships with, whether it is a dev-time set or an aid, and its rewrites' names. They are
    the sets the pass a program's trace runs chooses among (`dew.algebra.rules.admitted`), which a module
    registers as it is imported, so two processes that imported other modules hold other sets; a set's
    rules are the code that registered it, which the process's record states."""
    return {name: (rules.rounding, rules.receipt, rules.dev, rules.aid,
                   tuple(family.name for family in rules.families))
            for name, rules in algebra.rules.RULES.items()}


@contextlib.contextmanager
def compiled_together() -> Iterator[None]:
    """Within the block, a call of a program through Dew's compile point (`OuterProgram`) answers with its
    results' shapes (`shapes_of`), running nothing, and one whose executable the program keeps none of is
    traced and lowered on this thread, under its contexts, as the call would be; as the block ends,
    every lowering is compiled at once (`report.compiled_at_once`: XLA's compiles in threads of their
    own, a spill's refusal relowered here) and kept by its program under its call's key, so the same
    call after the block runs it, compiling nothing: a scoring task's programs compiled before its
    passes (`dew.decision.task.Decide.compile_scoring`). Each call's compile is its program's as `_kept`
    reserves it, and a call of a key this block already lowered lowers it once; a block waits on no
    compile it does not own (a key another thread reserved answers its call's traced shapes), and a block
    within another is the outer one's, so no two blocks wait on each other. A block that raises compiles
    nothing, and a compile that fails fails every call waiting for it, the program keeping none of them."""
    from dew.nn.kernels.templates import report

    if _TOGETHER.get() is not None:
        yield
        return
    lowerings: dict[tuple[int, Hashable], _Lowering] = {}
    token = _TOGETHER.set(lowerings)
    try:
        try:
            yield
        finally:
            _TOGETHER.reset(token)
        held = list(lowerings.values())
        executables = report.compiled_at_once([
            (call.lowered, None, functools.partial(call.program._relowered, call.args, call.kwargs,
                                                   call.relowered))
            for call in held], [call.kept_from for call in held])
    except BaseException as failure:
        for call in lowerings.values():
            call.program._released(call.key, call.pending, failure)
        raise
    for call, executable in zip(held, executables, strict=True):
        settled = _accepted(call.kept_from, call.lowered, call.relowered, call.refused, call.reads)
        call.program._settle(call.key, call.pending, executable, settled, call.args, call.kwargs)
    # Heard once every compile is settled, so a listener that raises leaves no call waiting.
    for call, executable in zip(held, executables, strict=True):
        _heard(call.program, call.kept_from[-1] if call.kept_from else call.lowered, executable)

@dataclasses.dataclass
class _Kept:
    """An executable a program keeps under a call's key (`OuterProgram._ready`): the facts its trace read
    (`reads`), the refusals it was last found to hold under (`under`, `report.refusal_snapshot`), and the
    compile reservation it settled (`reserved`, `OuterProgram._executables`), which `_bounded` lets go of
    with it."""

    executable: jax.stages.Compiled
    reads: Facts
    under: tuple
    reserved: Hashable


def _refusals_hold(reads: Facts) -> bool:
    """Whether the refusals of each launch a trace read (`report.refusals_of`, among `reads`) answer now as
    it read them: what an executable, a begun compile or a program compiled over shapes of it holds by, a
    refusal of another launch none of it (rows 12311, 12328: the whole registry keyed every program, so any
    refusal compiled each again, an evaluation's spill the training step)."""
    from dew.nn.kernels.templates import report

    return _facts.still(_facts.of(report.refusals_of, reads))


def _holding(kept: Sequence[_Kept], refusals: tuple) -> jax.stages.Compiled | None:
    """The executable of `kept`, a key's, that a call under `refusals` (`report.refusal_snapshot`) runs: one
    found to hold under them, else one whose trace's refusals answer now as it read them (`_refusals_hold`),
    found so from here on; None where none does. A key keeps one a refusal scope its calls ran in, so two
    scopes in turn compile none again."""
    for each in kept:
        if each.under == refusals:
            return each.executable
    for each in kept:
        if _refusals_hold(each.reads):
            each.under = refusals
            return each.executable
    return None


class _Resettled(Exception):
    """A compile reserved under one key kept under another (`OuterProgram._settle`): the key of the refusals
    its kept program traced under, which its waiters, joined on the reservation's, look past."""


def _accepted(kept_from: list[jax.stages.Lowered], lowered: jax.stages.Lowered, relowered: Relowered,
              refused: tuple, reads: Facts) -> tuple[jax.stages.Wrapped | None, tuple, Facts]:
    """The jit, refusals and facts read of the lowering a compile kept (`kept_from`'s, else `lowered`): a
    relowering's, or None, `refused` and `reads`, those `lowered` traced under and read, where it kept
    `lowered` (a relowering it discarded, one launching every refused config again, is no program it runs:
    theme K01-e's review P0)."""
    accepted = kept_from[-1] if kept_from else lowered
    return next(((fresh, under, read) for fresh, relowering, under, read in relowered
                 if relowering is accepted), (None, refused, reads))


_PASSING: contextvars.ContextVar[bool] = contextvars.ContextVar("outer_program_passing", default=False)
"""Whether an `OuterProgram`'s trace, which the algebra pass reads whole, encloses the current call."""


def release() -> None:
    """Drop every live program's compiled executables (`OuterProgram.released`) and the process's held ones
    (`held`), as `jax.clear_caches` drops jit's own, and what else keeps a lowering whose live module keys
    JAX's loaded executable of it (`held_until_release`): XLA:CPU maps each executable's code into the
    process, and a program keeps up to `OuterProgram.CACHED` of them for as long as it lives, past a cache
    JAX clears. A process that compiles on and on (a test suite) frees the mappings this way; each program
    compiles again at its next call."""
    # A snapshot taken under the lock, each holder asked outside it: a holder registers under its own lock
    # (`candidate_for` takes `_HOLDERS_LOCK` within `_KEEPING`), so a release that asked it within ours
    # would take the two the other way round.
    with _HOLDERS_LOCK:
        holders = list(_HOLDERS)
    for holder in holders:
        holder.released()
    unhold()


def unhold(key: Hashable | None = None) -> None:
    """Drop the executable the process holds under `key` (`hold`), or every one it holds where None, every
    program's own kept: one a plan judged and passed over, or what a fresh process holds of them, which a
    test of what a start compiles begins from."""
    with _HELD_LOCK:
        if key is None:
            _HELD.clear()
        else:
            _HELD.pop(key, None)


_HELD: collections.OrderedDict[Hashable, jax.stages.Compiled] = collections.OrderedDict()
"""The executables a process compiled from a lowering it may lower again (`hold`, `held`), the latest
`OuterProgram.CACHED`, by the program within the process, its options and its devices."""
_HELD_LOCK = threading.Lock()


def held(key: Hashable) -> jax.stages.Compiled | None:
    """The executable this process compiled under `key` (`hold`), None where it holds none."""
    with _HELD_LOCK:
        return _HELD.get(key)


def hold(key: Hashable, executable: jax.stages.Compiled) -> None:
    """Hold `executable` under `key` (the program within the process, `program_key`, its options and the
    devices it loads on) for the process: a plan of the same step by a later trainer of the process takes
    it (`held`), where each new trainer read it back from the persistent compilation cache (4.4 s of a
    small model's 11.9 s start on CPU, row 6995). Its kernels' report is its lowering's, which the later
    plan reads as it lowers; a held executable compiles nothing, so the compile telemetry names no compile
    of it (`dew.telemetry.compiles`). A plan holds the executables that fit, which a run may step on, a
    guided one in its unguided one's place (`dew.training.planning.Planned.guided`); one that missed only
    where its plan judged devices as they will not stand (`FitSearch.provisional`: before the state is
    placed, a later phase's off the loop), for the plan made on them to judge again, which drops it where it
    misses there too: its device code is held for nothing after. The latest `OuterProgram.CACHED` are held,
    as a program keeps its own; `release` drops them."""
    with _HELD_LOCK:
        _HELD[key] = executable
        while len(_HELD) > OuterProgram.CACHED:
            _HELD.popitem(last=False)


class OuterProgram:
    """`function` jitted (`jax.jit`, under `arguments`), every compile of it read through the kernel
    templates' reports (`dew.nn.kernels.templates.report.compiled`): where a launch's kernel spills, its
    config is refused and the function traced afresh, by a new jit of a new callable past JAX's trace
    cache, and compiled once more, the launch's derivation taking the next config. Called as its jit
    is, it runs the executable that what jit keys its own on chooses (`key_of`): the static arguments,
    the others' structure, shapes, dtypes and placements, and the trace context, the `CACHED` latest compiled;
    `lower` and `trace` are its jit's. Called within another trace (a `jax.grad`, a `jax.vmap`, an
    enclosing jit), it is its jit, which the enclosing program compiles as its own."""

    CACHED: ClassVar[int] = 4096
    """The executables a program keeps, the latest compiled, so its cache is bounded as jit's is: a call of
    one compiled before reads it without refreshing its place (row 3326)."""

    def __init__(self, function: Callable, **arguments):
        self.function, self.arguments = function, arguments
        # Its jits, each by the abstract key of the one call its trace was validated for (`abstract_of`: its
        # shapes, trace context and the refusals it read), the most recent `JITS_HELD` (`_held_jit`).
        self._jits: collections.OrderedDict[Hashable, jax.stages.Wrapped] = collections.OrderedDict()
        # The facts each program its jits traced read, by the program's jaxpr, the one JAX's trace cache
        # answers a later trace by the jit with (`_stable_trace`): held once its trace returns, so a trace
        # answered by a program whose facts are not held (another thread's, between JAX's cache taking it and
        # its trace returning) is no program a call takes (row 12296's review P0).
        self._programs: weakref.WeakKeyDictionary[jax_core.Jaxpr, Facts] = weakref.WeakKeyDictionary()
        # Its jits within another trace and under an eager transformation, one each (`_held_call`), which
        # every layer calling it shares: one program, lowered once.
        self._nested_jits: dict[Hashable, jax.stages.Wrapped] = {}
        # The static positions and names as jit resolves them, the ones given and those it infers.
        numbers, names = arguments.get("static_argnums"), arguments.get("static_argnames")
        signature = api_util.fun_signature(function)
        if signature is not None:
            numbers, names = api_util.infer_argnums_and_argnames(signature, numbers, names)
        self.numbers, self.names = _given_tuple(numbers), frozenset(_given_tuple(names))
        # Each compile reserved and settled by its call's key and the refusals it began under (`_reserved`),
        # the least recently used first.
        self._executables: collections.OrderedDict[tuple[Hashable, tuple], Future[jax.stages.Compiled]] = (
            collections.OrderedDict())
        # The executables kept under each call's key, read without the lock: a call of a signature compiled
        # before costs its key and a lookup (row 3326), and where a refusal moved the registry since, the
        # refusals its trace read (`_holding`).
        self._ready: dict[Hashable, tuple[_Kept, ...]] = {}
        # Each executable compiled within `collecting` over shapes (or arrays) by the call's abstract
        # signature (`abstract_of`: shapes, dtypes, weak types, shardings, layouts and the static arguments),
        # which a later call over arrays alike in it runs, tracing nothing (`_shaped_for`): a run's startup
        # compiles a host step's programs over its state's shapes, and its steps run them (a host step
        # compiled ahead, `dew.training.compiler`). A call whose arrays differ in any of it (a sharding the
        # shapes did not state) misses and traces; it never runs another program's executable. An entry
        # leaves on its first use for `_ready`, under that call's key, and `CACHED` bounds the two together.
        # Each with the digest a collection holds it by (`collecting`; None for a program none can state),
        # which every collection that meets it again records (`__call__`).
        self._shaped: collections.OrderedDict[Hashable, tuple[jax.stages.Compiled, str | None,
                                                              tuple[bool, ...]]] = (
            collections.OrderedDict())
        # The facts each executable's trace read, by the executable, read again by every call it answers with
        # no trace (row 12296's review P0: a warm call recorded none, a value made from it held past them).
        self._reads: weakref.WeakKeyDictionary[jax.stages.Compiled, Facts] = weakref.WeakKeyDictionary()
        self._compiling = threading.Lock()
        held_until_release(self)

    def released(self) -> None:
        """Drop every executable it keeps (`release`), each compiled again at its next call."""
        with self._compiling:
            self._ready.clear()
            self._executables.clear()
            self._shaped.clear()

    def _jitted(self) -> jax.stages.Wrapped:
        """Its jit, of a callable of its own, past JAX's trace cache (a relowering's, `compiled`)."""
        return jax.jit(_fresh(self.function), **self.arguments)

    JITS_HELD = 8
    """The jits it holds, each the one call's it was validated for (`_validated`), the least recently taken
    let go past it."""

    def _held_jit(self, key: Hashable) -> jax.stages.Wrapped | None:
        """The jit whose trace was validated for the call of abstract key `key` (`_validated`), None where
        none was: read, never claimed, so a call at another key traces by a jit of its own and a held jit
        only ever answers its own call, from JAX's trace cache (theme K01-e's reviews' P0s: a held jit traced
        at a new shape cached the program another thread's refusal made, for every call it answered)."""
        with self._compiling:
            held = self._jits.get(key)
            if held is not None:
                self._jits.move_to_end(key)
            return held

    def _held_key(self, args: tuple, kwargs: dict) -> Hashable | None:
        """The key a jit is held by for a call of `args` and `kwargs` (`abstract_of`), None for a call within
        another trace, which holds none: its trace is the enclosing one's, whose key it cannot read (its
        values tracers, and JAX's concrete mesh refused within it). A refusal is none of it: a trace its
        cache answers is taken where its program's facts hold (`_programs`)."""
        if not jax_core.trace_state_clean():
            return None
        return self.abstract_of(args, kwargs)

    def _validated(self, key: Hashable, made: jax.stages.Wrapped) -> None:
        """`made`, a jit whose trace of the call of `key` read its refusals alone, the one that call takes
        from here on, where none is yet; the most recent `JITS_HELD` held."""
        with self._compiling:
            self._jits.setdefault(key, made)
            self._jits.move_to_end(key)
            while len(self._jits) > self.JITS_HELD:
                self._jits.popitem(last=False)

    def _invalidated(self, key: Hashable, held: jax.stages.Wrapped) -> None:
        """`held`, the jit validated for `key`, let go where a trace by it read other refusals (JAX's trace
        cache dropped its program and its trace raced, or its settle retraced past one)."""
        with self._compiling:
            if self._jits.get(key) is held:
                del self._jits[key]

    def _traced(self, held: tuple[jax.stages.Wrapped, tuple], args: tuple, kwargs: dict, *, frozen: bool
                ) -> tuple[jax.stages.Traced, tuple[jax.stages.Wrapped, tuple], jax_core.Jaxpr] | (
                    ReadingRefused):
        """Its trace of a call by `held`'s jit, its program the algebra pass's (`algebra.rewritten_program`),
        so its lowering, its trace, its executable and every relowering read one program
        (docs/design/algebra.md §6.7): the pass reads the trace jit made, which lowers as it is where no
        rule changes it; a program a rule changes lowers from its rewritten program, never traced again
        (`program.traced_as`). Its switch is in JAX's trace context, so a held program traces afresh as it
        changes. A program called within the trace is part of this one, which the pass reads whole
        (`_nested`). Within a block certifying aside, the launches an outermost program's trace reaches
        settle before it is lowered, the program traced afresh past a refused config by a jit of its own,
        past JAX's trace cache (`_jitted`; `report.settled_trace`), so it lowers once; a program traced
        within another's (a launch's own, as its capture traces it) settles nothing mid-trace. Beside its
        trace, the jit that made it and the launches refused as it traced (`held`'s, or a settle's own
        retrace's), and the program its jit traced, before the pass reads it (the one JAX's trace cache
        answers that jit with again, `_stable_trace`). Each trace reads those refusals alone and its own
        (`report.reading`), never another thread's recorded as it traces, so the program a jit's trace cache
        keeps is the one they make; `held`'s, where it is a held jit (`frozen`), none of its own, its cache
        only ever its key's: where its trace recorded one, the refusal that ended it taking nothing
        (`report.ReadingRefused`)."""
        from dew.nn.kernels.templates import report

        outermost = not _PASSING.get()
        token = _PASSING.set(True)
        made = held

        def traced_by(by: tuple[jax.stages.Wrapped, tuple], *, frozen: bool) -> jax.stages.Traced:
            with report.reading(by[1], frozen=frozen):
                return by[0].trace(*args, **kwargs)

        def retraced() -> jax.stages.Traced:
            nonlocal made
            made = (self._jitted(), report.refusal_snapshot())
            return traced_by(made, frozen=False)

        try:
            trace = functools.partial(traced_by, held, frozen=frozen)
            traced = report.settled_trace(trace, retraced) if outermost else trace()
            rewritten = algebra.rewritten_program(traced.jaxpr)
            if rewritten is traced.jaxpr:
                return traced, made, traced.jaxpr
            return algebra.program.traced_as(traced, rewritten), made, traced.jaxpr
        except report.ReadingRefused as refused:
            # Its own freeze's alone: an enclosing held jit's, which this trace's reading inherits, is that
            # trace's to retrace past, its snapshot one this trace's retrying never moves.
            if not frozen:
                raise
            return refused
        finally:
            _PASSING.reset(token)

    @functools.cached_property
    def _rewriting(self) -> Callable:
        """The function rewritten as it traces (`algebra.rewritten`): where no trace of it is read first, the
        pass reads the program it makes there."""
        @functools.wraps(self.function)
        def traced(*args, **kwargs):
            return algebra.rewritten(self.function, static_argnums=self.numbers,
                                     static_argnames=tuple(self.names), kernels=True)(*args, **kwargs)
        return traced

    def _eager(self) -> jax.stages.Wrapped:
        """A jit of it called under an eager transformation, which traces it where no trace of it can be read
        first: the function rewritten as it traces (`_rewriting`), one program as its compiles'."""
        return jax.jit(_fresh(self._rewriting), **self.arguments)

    def _nested(self, whole: bool) -> jax.stages.Wrapped:
        """A jit of it within another trace, the enclosing program's own options the ones it compiles under,
        as JAX takes compiler options on the outermost jit alone: as written where the pass reads the
        enclosing program whole (`_PASSING`; one of these, `_traced`); rewritten as it traces where it does
        not, so a transformation outside it (`jax.grad` of it within a plain `jax.jit`) reads the pass's
        program, the templates' derivatives among it (row 3450)."""
        arguments = {name: value for name, value in self.arguments.items() if name != "compiler_options"}
        return jax.jit(_fresh(self.function if whole else self._rewriting), **arguments)

    def _held_call(self, mode: Hashable, made: Callable[[], jax.stages.Wrapped], args: tuple, kwargs: dict):
        """A call within another trace or under an eager transformation by the jit it holds for `mode`
        (`made`'s, `_nested_jits`), so every call alike shares one program, which lowers once however many
        layers call it. Its trace is taken first, its facts by the program it made (`_programs`): those its
        trace read where its function ran, recorded into the recordings open there; those a trace JAX's cache
        answered with read when it was made, read again; and that program is the one the call binds
        (`_bound`), never its jit's cache looked up again, which a concurrent trace at its signature may have
        answered otherwise between. A program JAX's cache answers with whose facts are not held (another
        thread's trace not yet returned) or answer otherwise now (a refusal it read moved, its launches
        reading the enclosing trace's) answers no call: the call runs a jit of its own, and the held one is
        let go for the next; a trace that ran is taken as it ran (a fact it read under a context of its own
        answers otherwise outside it). A trace whose facts were another signature's, or not yet held where a
        second caller took it, answers no call (row 12296's reviews' P0s)."""
        with self._compiling:
            held = self._nested_jits.get(mode)
            if held is None:
                held = self._nested_jits[mode] = made()
        traced, reads = _read_by(lambda: held.trace(*args, **kwargs))
        program = traced.jaxpr
        if reads is None:
            with self._compiling:
                reads = self._programs.get(program)
            if reads is None or not _facts.still(reads):
                with self._compiling:
                    if self._nested_jits.get(mode) is held:
                        self._nested_jits[mode] = made()
                return made()(*args, **kwargs)
            _facts.replay(reads)
        else:
            with self._compiling:
                self._programs[program] = reads
        return self._bound(traced, args, kwargs)

    def _bound(self, traced: jax.stages.Traced, args: tuple, kwargs: dict):
        """`traced`'s program called at `args` and `kwargs` in the trace around it, bound as its jit's call
        binds it (`jax._src.pjit`'s: its constants, then its dynamic arguments flattened as its trace took
        them, a static position counted from the end as jit counts it, its placements and options its own)."""
        from jax._src import pjit as jax_pjit

        static = {at + len(args) if at < 0 else at for at in self.numbers}
        dynamic = tuple(arg for at, arg in enumerate(args) if at not in static)
        named = {name: value for name, value in kwargs.items() if name not in self.names}
        outputs = jax_pjit.jit_p.bind(*traced._consts, *jax.tree.leaves((dynamic, named)), **traced._params)
        return jax.tree.unflatten(traced.out_tree, outputs)

    def __call__(self, *args, **kwargs):
        from dew.nn.kernels.templates import report

        # Within a staging trace the enclosing program compiles this one as part of its own, under its
        # own options (`_nested`); under an eager transformation alone, or `jax.disable_jit`, its jit
        # compiles itself with its options, or runs the function as Python.
        if jax_config.disable_jit.value:
            # Its function run as Python, every fact it reads read as it runs: as written where the pass reads
            # an enclosing program whole, else rewritten (`_nested`, `_eager`).
            return (self.function if staged() and _PASSING.get() else self._rewriting)(*args, **kwargs)
        if staged():
            whole = _PASSING.get()
            return self._held_call(("nested", whole), lambda: self._nested(whole), args, kwargs)
        if not jax_core.trace_state_clean():
            return self._held_call(("eager",), self._eager, args, kwargs)
        if (lowering := _LOWERING.get()) is not None:
            platforms, lowerings = lowering
            traced = self.trace(*args, **kwargs)
            target = platforms if _abstractly_placed((args, kwargs)) else None
            lowered = traced.lower(lowering_platforms=target)
            lowerings.append((self.called, lowered))
            return _outputs(lowered.out_info, _weak_types(traced))
        # The launches refused, read once a call: the executable it runs is found holding under one snapshot
        # (theme K01-e's review P0).
        refusals = report.refusal_snapshot()
        if (collected := _COLLECTED.get()) is not None:
            abstract = self.abstract_of(args, kwargs)
            with self._compiling:
                held = self._shaped.get(abstract)
            # Compiled once over these shapes: a pass over shapes again (a host step's next step's, whose
            # state its first returned; another collection's in this process) takes the one it holds where
            # the refusals its trace read hold (`_refusals_hold`), and its collection records it as the
            # compile recorded it in the first.
            if held is None or not _refusals_hold(self._reads.get(held[0], {})):
                executable, digest, weak_types, (_, _, reads) = self._compiled_digested(*args, **kwargs)
                held = executable, digest, weak_types
                with self._compiling:
                    self._reads[executable] = reads
                    self._shaped[abstract] = held
                    self._bounded()
            executable, digest, weak_types = held
            _facts.replay(self._reads.get(executable, {}))
            if digest is not None:
                collected[digest] = executable
            # Each result with the weak type its compile's own trace gave it, never a trace again (row 11766:
            # a trace after a settle's own missed JAX's cache and traced every program twice).
            return _outputs(executable.out_info, weak_types)
        key, dynamic, named = self.key_of(args, kwargs)
        if (together := _TOGETHER.get()) is not None:
            return self._together(together, (key, refusals), args, kwargs)
        ready = _holding(self._ready.get(key, ()), refusals)
        if ready is None:
            ready = self._shaped_for(key, args, kwargs, refusals) or self._kept(key, args, kwargs, refusals)
        # The facts its trace read, which no trace of this call reads (`_settle`).
        _facts.replay(self._reads.get(ready, {}))
        return self.invoked(ready, args, kwargs, *dynamic, **named)

    def _shaped_for(self, key: Hashable, args: tuple, kwargs: dict, refusals: tuple
                    ) -> jax.stages.Compiled | None:
        """The executable a compile within `collecting` made for a call alike in its abstract signature
        (`_shaped`), kept from here on under this call's `key` and run, tracing nothing; None where none
        was made, or the refusals its trace read hold no longer (`_refusals_hold`). It stays held by its
        abstract signature too: a later call alike in it under another key (an array its last step committed
        where the first call's was not, or placed under an equal sharding of another object) runs it, where
        it compiled the program again (a host step's EMA update at its second step). Kept as a compile under
        `refusals` settled it (`_settle`)."""
        if not self._shaped:
            return None
        abstract = self.abstract_of(args, kwargs)
        with self._compiling:
            held = self._shaped.get(abstract)
            if held is None or not _refusals_hold(reads := self._reads.get(held[0], {})):
                return None
            self._shaped.move_to_end(abstract)
            executable = held[0]
            settled: Future[jax.stages.Compiled] = Future()
            settled.set_result(executable)
            self._executables[key, refusals] = settled
            self._kept_under(key, _Kept(executable, reads, refusals, (key, refusals)))
            self._bounded()
        return executable

    def _bounded(self) -> None:
        """Hold no more than `CACHED` executables, kept (`_executables`) and compiled over shapes
        (`_shaped`) together, one a call alike in its abstract signature runs too (`_shaped_for`) counted
        once: the least recently made over shapes go first, then the least recently used kept. Called with
        `_compiling` held."""
        # Its executables at most `CACHED`, and the keys and abstract signatures that reach them at most twice
        # that: a handed program reached under many keys holds no index without bound.
        while self._held() > self.CACHED or len(self._executables) + len(self._shaped) > 2 * self.CACHED:
            ready = {id(kept.executable) for kept_all in self._ready.values() for kept in kept_all}
            # The least recently made over shapes that no call runs yet, else the least recently used kept,
            # with every entry over shapes of its executable no other key still runs.
            unaliased = next((abstract for abstract, held in self._shaped.items()
                              if id(held[0]) not in ready), None)
            if unaliased is not None:
                self._shaped.pop(unaliased)
                continue
            reserved, _ = self._executables.popitem(last=False)
            key = reserved[0]
            evicted = [kept.executable for kept in self._ready.get(key, ()) if kept.reserved == reserved]
            left = tuple(kept for kept in self._ready.get(key, ()) if kept.reserved != reserved)
            if left:
                self._ready[key] = left
            else:
                self._ready.pop(key, None)
            running = {id(kept.executable) for kept_all in self._ready.values() for kept in kept_all}
            for executable in (executable for executable in evicted if id(executable) not in running):
                for abstract in [abstract for abstract, held in self._shaped.items()
                                 if held[0] is executable]:
                    self._shaped.pop(abstract)

    def _held(self) -> int:
        """How many executables it holds, kept, compiling and compiled over shapes, each once however many
        dispatch keys or abstract signatures run it."""
        held = {id(kept.executable) for kept_all in self._ready.values() for kept in kept_all}
        held |= {id(entry[0]) for entry in self._shaped.values()}
        return len(held) + sum(not pending.done() for pending in self._executables.values())

    def _kept_under(self, key: Hashable, kept: _Kept) -> None:
        """Keep `kept` under the call's `key` (`_ready`), first, past any it replaces (its executable's or its
        reservation's). Called with `_compiling` held."""
        self._ready[key] = (kept, *(each for each in self._ready.get(key, ()) if each.executable is not
                                    kept.executable and each.reserved != kept.reserved))

    @property
    def options(self) -> jax.stages.CompilerOptions | None:
        """The XLA options every compile of the program takes (`jax.jit`'s `compiler_options`), part of the
        fingerprint an executable is handed by (`program_digest`)."""
        return self.arguments.get("compiler_options")

    @property
    def called(self) -> str:
        """The program's function by its module and name, a callable object's by its class's: what a record
        of its compiles names it by."""
        function = self.function
        named = (function if isinstance(function, types.FunctionType | types.MethodType | type)
                 else type(function))
        return f"{named.__module__}.{named.__qualname__}"

    def executable(self, *args, **kwargs) -> jax.stages.Compiled:
        """The executable a call with these arguments runs, compiled at the first: called with the dynamic
        arguments alone, as a caller that calls it every step does, building no dispatch key, each call
        within the contexts of the `running` blocks it runs in (`ran`)."""
        from dew.nn.kernels.templates import report

        refusals = report.refusal_snapshot()
        key, _, _ = self.key_of(args, kwargs)
        ready = _holding(self._ready.get(key, ()), refusals) or self._kept(key, args, kwargs, refusals)
        _facts.replay(self._reads.get(ready, {}))
        return ready

    def invoked(self, executable: jax.stages.Compiled, args: tuple, kwargs: dict, /, *dynamic, **named):
        """Call `executable`, fetched for these arguments (`executable`), on `dynamic` and `named` inside
        every enclosing `running` block's contexts, entered here, as a call through the program enters them
        (row 8480: a rollout's executables fetched once and called each step, `dew.rl.simulator`, `Fetched`,
        a server's `dispatched`, ran past its admission and a trial's record), its first run taken as run
        once it completes (`dew.communicators.called`): the one call of a fetched executable."""
        with _ran(self, executable, args, kwargs):
            return communicators.called(executable, lambda: executable(*dynamic, **named))

    def _reserved(self, key: tuple[Hashable, tuple]) -> tuple[Future[jax.stages.Compiled], bool]:
        """The compile kept under `key` and whether this caller owns it: a future its owner settles
        (`_keep`, `_released`), reserved where none is, the bound `CACHED` held over everything it keeps
        (`_bounded`); the map alone held, so calls of other keys compile at once."""
        with self._compiling:
            pending = self._executables.get(key)
            owner = pending is None
            if owner:
                pending = self._executables[key] = Future()
            self._executables.move_to_end(key)
            if owner:
                # One bound over every executable it holds, kept and compiled over shapes (`_bounded`):
                # this key, the most recently used, is never the one it lets go.
                self._bounded()
        return pending, owner

    def _kept(self, key: Hashable, args: tuple, kwargs: dict, refusals: tuple) -> jax.stages.Compiled:
        """The executable a call of `key` under `refusals` runs, compiled where none is (`compiled`), by the
        caller owning its reservation (`_reserved`, of the key and the refusals)."""
        from dew.nn.kernels.templates import report

        reserved = (key, refusals)
        pending, owner = self._reserved(reserved)
        if owner:
            try:
                executable, _, _, settled = self._compiled_digested(*args, **kwargs)
            except BaseException as error:
                self._released(reserved, pending, error)
            else:
                self._settle(reserved, pending, executable, settled, args, kwargs)
                return executable
        try:
            return pending.result()
        except _Resettled:
            # Kept under other refusals than this call's (`_settle`): looked up, or compiled, under its own.
            return self._kept(key, args, kwargs, report.refusal_snapshot())

    def _lowered_aside(self, args: tuple, kwargs: dict, relowered: Relowered
                       ) -> tuple[jax.stages.Lowered, tuple[bool, ...], tuple, Facts]:
        """The call's program lowered, its launches certified aside while it traces
        (`report.certifying_aside`), and lowered afresh (`_relowered`) for as long as it launches a refused
        config (`report.refused_in`), each refused launch's derivation stepping past its config and certifying
        the next in turn: its own certifications' refusals, or another caller's committed after its trace took
        the config (row 8603: another thread's drain left this program compiling at it). A relowering that
        launches a config refused before again (a derivation reading no refusal, a pin's) is its compile's to
        read (`report.settled`). Beside it, its outputs' weak types and the launches refused as its program
        traced and the facts it read (its last relowering's, else its trace's)."""
        from dew.nn.kernels.templates import report

        traces: list[tuple[jax.stages.Traced, tuple, Facts]] = []

        def lowered() -> jax.stages.Lowered:
            traces.append(self._snapshot_trace(args, kwargs))
            return traces[-1][0].lower()

        program = report.lowered_aside(lowered, lambda: self._relowered(args, kwargs, relowered),
                                       lambda lowered: lowered)
        # Its outputs' weak types, its first trace's: a relowering steps a launch's config, which changes
        # no output. Its refusals and facts the program's own: a relowering's, else its trace's.
        under, reads = next(((refused, reads) for _, relowering, refused, reads in relowered
                             if relowering is program), traces[-1][1:])
        return program, _weak_types(traces[0][0]), under, reads

    def _relowered(self, args: tuple, kwargs: dict, relowered: Relowered) -> jax.stages.Lowered:
        """A call's program traced afresh past a spill's refused config (`report.compiled`'s `relower`), on
        the thread that keeps it, each relowering with its jit and the launches refused as it traced kept in
        `relowered`."""
        traced, refused, fresh, reads = self._stable_trace(args, kwargs, fresh=True)
        relowering = traced.lower()
        relowered.append((fresh, relowering, refused, reads))
        return relowering

    def _together(self, together: dict[tuple[int, Hashable], _Lowering], key: tuple[Hashable, tuple],
                  args: tuple, kwargs: dict):
        """A call within `compiled_together`: its results' shapes, its executable reserved (`_reserved`) and
        its lowering kept for the block's compile where it owns the reservation; the
        shapes of a compile another owns, once it is done; one this block lowered, its lowering's."""
        if (lowering := together.get((id(self), key))) is not None:
            return shapes_of(lowering.lowered.out_info)
        pending, owner = self._reserved(key)
        if not owner:
            # Another's compile, done or not: its shapes as the call traces, none waited for; a done one's
            # executable's facts read again.
            if pending.done() and pending.exception() is None:
                _facts.replay(self._reads.get(pending.result(), {}))
                return shapes_of(pending.result().out_info)
            return shapes_of(self.trace(*args, **kwargs).out_info)
        try:
            traced, refused, reads = self._snapshot_trace(args, kwargs)
            lowered = traced.lower()
        except BaseException as failure:
            self._released(key, pending, failure)
            raise
        # A program compiled ahead or kept for this target runs as it was compiled (`taking`).
        if (ahead := taken(lowered, self.options)) is not None:
            self._settle(key, pending, ahead(), (None, refused, reads), args, kwargs)
        else:
            together[id(self), key] = _Lowering(self, key, pending, lowered, args, kwargs, refused, reads)
        return shapes_of(lowered.out_info)

    def _settle(self, reserved: tuple[Hashable, tuple], pending: Future[jax.stages.Compiled],
                executable: jax.stages.Compiled, settled: tuple[jax.stages.Wrapped | None, tuple, Facts],
                args: tuple, kwargs: dict) -> None:
        """Settle the compile reserved under `reserved` (a call's key and the refusals it began under,
        `_reserved`) with `executable`, its waiters given it, and keep it under the call's key with the facts
        its kept program's trace read and the refusals it traced under (`settled`'s, `_accepted`), reserved
        under them: the reservation's where they are its, else theirs alone, a compile that refused a
        launch's config relowering past it, which a call after it, of those refusals, runs (row 12165), and
        no call of the reservation's refusals runs a program traced past others (theme K01-e's review P0).
        Its jit (`settled`'s, a relowering's) traces under its refusals from here on; the facts its kept
        program's trace read are read again by every call it answers."""
        fresh, refused, reads = settled
        if fresh is not None and (held := self._held_key(args, kwargs)) is not None:
            self._validated(held, fresh)
        key = reserved[0]
        accepted = (key, refused)
        kept: Future[jax.stages.Compiled] = Future()
        kept.set_result(executable)
        with self._compiling:
            self._reads[executable] = reads
            if accepted == reserved:
                if reserved in self._executables:
                    self._kept_under(key, _Kept(executable, reads, refused, reserved))
            else:
                if self._executables.get(reserved) is pending:
                    self._executables.pop(reserved)
                if accepted not in self._executables:
                    self._executables[accepted] = kept
                    self._kept_under(key, _Kept(executable, reads, refused, accepted))
            self._bounded()
        # Its waiters, joined on its reservation: given it where it is kept under its refusals, else sent to
        # look up or compile under their own (`_Resettled`), once it is kept where it is (theme K01-e's
        # review P0: a waiter given a program traced past refusals its scope holds none of).
        if accepted == reserved:
            pending.set_result(executable)
        else:
            pending.set_exception(_Resettled(accepted))

    def keeps(self) -> bool:
        """Whether it keeps an executable a call over arrays runs (one compiled at a call, or taken from a
        compile over shapes at one): none before its first call and once released (`released`). It says that
        one is kept, not that it is the one a call over given arguments runs: a caller reads it of a program
        called over one shape alone (a kept start's, `dew.training.distributed.start_compiling`)."""
        with self._compiling:
            return any(self._ready.values())

    def guided(self, executable: jax.stages.Compiled, replacement: jax.stages.Compiled) -> None:
        """`executable`, one this program holds, replaced by `replacement`, its compile again guided by a
        profile of its runs (AutoPGLE's, `dew.training.planning.guided_compile`), wherever the program holds
        it (kept, ready to run, or compiled over shapes) under the same key: its calls run the guided one
        from here on. A replacement of other inputs' or outputs' types, placements or layouts, leaf by leaf,
        is refused, as a guided compile is the same program. A collection (`collecting`) holding `executable`
        by its digest keeps it."""
        # Their leaves: a tree's static data may hold a closure each trace makes anew (a forward's pullback,
        # `jax.vjp`'s), the same function of another object.
        def stated(compiled):
            return {name: jax.tree.leaves(tree) for name, tree in (
                ("inputs", compiled.in_avals), ("input formats", compiled.input_formats),
                ("outputs", compiled.out_info), ("output formats", compiled.output_formats))}
        differs = {name: (held, guided) for (name, held), guided in zip(
            stated(executable).items(), stated(replacement).values(), strict=True) if held != guided}
        if differs:
            raise ValueError(f"{self.called}'s guided compile states other {', '.join(differs)} than the "
                             f"executable it replaces: {differs}")
        with self._compiling:
            self._reads[replacement] = self._reads.get(executable, {})
            for key, pending in list(self._executables.items()):
                if pending.done() and pending.exception() is None and pending.result() is executable:
                    settled: Future[jax.stages.Compiled] = Future()
                    settled.set_result(replacement)
                    self._executables[key] = settled
            for kept in (kept for kept_all in self._ready.values() for kept in kept_all):
                if kept.executable is executable:
                    kept.executable = replacement
            for abstract, (shaped, digest, weak_types) in list(self._shaped.items()):
                if shaped is executable:
                    self._shaped[abstract] = (replacement, digest, weak_types)

    def _released(self, key: Hashable, pending: Future[jax.stages.Compiled], failure: BaseException) -> None:
        """Fail the compile reserved under `key` (`_reserved`), the program keeping none."""
        with self._compiling:
            self._executables.pop(key, None)
        if not pending.done():
            pending.set_exception(failure)

    def compiled(self, *args, **kwargs) -> jax.stages.Compiled:
        """The program over these arguments, compiled through the kernel templates' reports, lowered afresh
        where a kernel spills; one a block hands by its fingerprint (`taking`) runs that executable, its
        lowering what proves it the program: a caller that compiles a call itself (a server choosing its
        state's layout) is handed it as a call is; the facts its kept program's trace read read again, a
        compile begun aside (`compiling`) tracing none here."""
        executable, _, _, (_, _, reads) = self._compiled_digested(*args, **kwargs)
        _facts.replay(reads)
        return executable

    def _compiled_digested(self, *args, **kwargs) -> tuple[jax.stages.Compiled, str | None, tuple[bool, ...],
                                                           tuple[jax.stages.Wrapped | None, tuple, Facts]]:
        """`compiled`, the digest its collection holds it by where it compiles within `collecting` (None
        elsewhere, and for a program none can state), and its trace's outputs' weak types: its
        tracing, lowering and compile one compile's work in JAX's events (`telemetry.compiles.spanning`), the
        backend its executable targets stated on the span once it is compiled, which a ledger of what a block
        compiled reads by backend (`dew.training.compiler.HostStartup`); and its kept program's jit, the
        launches refused as it traced (`_accepted`), which its reservation is kept under (`_settle`), and the
        facts its trace read."""
        with spanning(self.called) as span:
            executable, digest, weak_types, settled = self._compiled_spanned(span, args, kwargs)
            span.executable = executable
        return executable, digest, weak_types, settled

    def _compiled_spanned(self, span: Span, args: tuple, kwargs: dict
                          ) -> tuple[jax.stages.Compiled, str | None, tuple[bool, ...],
                                     tuple[jax.stages.Wrapped | None, tuple, Facts]]:
        """`_compiled_digested`'s compile, its `span` told whether it built its executable."""
        from dew.nn.kernels.templates import report

        relowered: Relowered = []
        # A call whose first compile was begun in a thread of its own (`compiling`), the same program over
        # the same abstract inputs, lowers nothing again: the begun lowering is its.
        begun = _ATTEMPTING.get()
        if (begun is not None and begun.program is self and begun.abstract == self.abstract_of(args, kwargs)
                and _refusals_hold(begun.reads)):
            lowered, weak_types = begun.lowered, begun.weak_types
            refused = report.refusal_snapshot() if begun.refused is None else begun.refused
            reads = begun.reads
            begun.taken.set()
        else:
            lowered, weak_types, refused, reads = self._lowered_aside(args, kwargs, relowered)
        # A program compiled ahead or kept for this target runs as it was compiled (`taking`).
        if (ahead := taken(lowered, self.options)) is not None:
            return ahead(), None, weak_types, _accepted([], lowered, relowered, refused, reads)
        # Its first compile from that thread, where it lowers the program begun there.
        first = None
        if begun is not None and (lowered is begun.lowered
                                  or program_key(begun.lowered) == program_key(lowered)):
            first = begun.attempt.result()
        kept_from: list[jax.stages.Lowered] = []
        executable = report.compiled(lowered,
                                     relower=functools.partial(self._relowered, args, kwargs, relowered),
                                     first=first, kept_from=kept_from)
        span.built = True
        # The program kept, a relowering's past the trace its refused config is in, or its own.
        settled = _accepted(kept_from, lowered, relowered, refused, reads)
        lowered = kept_from[-1] if kept_from else lowered
        # A program no file can hold (its trees hold a node no process names: a pullback's closure) is
        # collected by no digest; a call over arrays alike still runs it (`_shaped`).
        digest = None
        if (collected := _COLLECTED.get()) is not None and (
                digest := stated_digest(lowered, self.options)) is not None:
            collected[digest] = executable
        _heard(self, lowered, executable)
        return executable, digest, weak_types, settled

    def lower(self, *args, **kwargs) -> jax.stages.Lowered:
        return self.trace(*args, **kwargs).lower()

    def trace(self, *args, **kwargs) -> jax.stages.Traced:
        return self._snapshot_trace(args, kwargs)[0]

    def _snapshot_trace(self, args: tuple, kwargs: dict) -> tuple[jax.stages.Traced, tuple, Facts]:
        """A call's trace, the launches refused as it traced (`report.refusal_snapshot`) and the facts it
        read, its jit their own call's (`_held_jit`), as JAX's trace cache keys on none of them: a trace
        after a seeded refusal derives past it (row 12165)."""
        traced, refused, made, reads = self._stable_trace(args, kwargs, fresh=False)
        if (key := self._held_key(args, kwargs)) is not None:
            self._validated(key, made)
        return traced, refused, reads

    def _stable_trace(self, args: tuple, kwargs: dict, *, fresh: bool
                      ) -> tuple[jax.stages.Traced, tuple, jax.stages.Wrapped, Facts]:
        """A trace whose launches read one snapshot of the refusals, the jit that made it and the facts it
        read beside it (a held jit's trace JAX's cache answered, which reads none anew, those of the program
        it answered with, `_programs`, read again into every open recording, `_facts.replay`; one answered
        with a program whose facts are not held, or answer otherwise now, traced again by a jit of its own):
        traced again where another thread refused a launch as it traced, until none did, refusals only adding
        of finitely many; a settle's retrace's (`_traced`) its own. Its jit the one validated for this call
        under the snapshot (`_held_jit`), else one of its own, which `fresh` takes always (a relowering's,
        past JAX's trace cache). A raced trace's routes are withdrawn (`withdrawn`), and a validated jit
        whose trace read other refusals let go (`_invalidated`); a held jit's trace that records a refusal
        ends taking nothing (`report.ReadingRefused`), traced again by a jit of its own."""
        from dew.nn.kernels.templates import report

        # Past a held jit's trace that refused, a jit of its own each time, so its retrying ends.
        unheld = fresh
        while True:
            refusals = report.refusal_snapshot()
            key = None if unheld else self._held_key(args, kwargs)
            held = None if key is None else self._held_jit(key)
            marks = route_marks()
            outcome, reads = _read_by(functools.partial(
                self._traced, (self._jitted() if held is None else held, refusals), args, kwargs,
                frozen=held is not None))
            if isinstance(outcome, report.ReadingRefused):
                # A held jit's trace recorded a refusal (its cached program dropped): it took nothing, and
                # the next trace, under the refusals with it, takes a jit of its own.
                withdrawn(marks)
                unheld = True
                continue
            traced, (made, under), program = outcome
            if reads is None:
                with self._compiling:
                    reads = self._programs.get(program)
                if reads is None or not _facts.still(reads):
                    # JAX's trace cache answered the held jit (a jit of its own traces always) with a
                    # program whose facts are not held (another thread's trace not yet returned) or answer
                    # otherwise now: let go, traced by a jit of its own. A trace that ran is taken as it
                    # ran: a fact it read under a context of its own (a GPU it assumes) answers otherwise
                    # outside it.
                    if held is not None:
                        self._invalidated(key, held)
                    withdrawn(marks)
                    unheld = True
                    continue
                _facts.replay(reads)
            else:
                with self._compiling:
                    self._programs[program] = reads
            raced = report.refusal_snapshot() != under
            if held is not None and (raced or under != refusals):
                self._invalidated(key, held)
            if not raced:
                return traced, under, made, reads
            withdrawn(marks)

    def key_of(self, args: tuple, kwargs: dict) -> tuple[Hashable, tuple, dict]:
        """The key of the executables a call runs, and the arguments the executable takes, the static
        ones dropped. It holds what jit keys on: the trace context, with its abstract mesh, and the
        concrete mesh in place (`jax.sharding.get_mesh`), so a mesh of the same shape over other devices
        compiles its own executable. The launches refused are none of it: an executable it keeps holds
        the refusals of the launches its trace read (`_holding`), so a refusal of another launch leaves it
        standing (rows 12311, 12328)."""
        return self._keyed(args, kwargs, _signature)

    def abstract_of(self, args: tuple, kwargs: dict) -> Hashable:
        """What a call over these arguments lowers from, arrays or their shapes alike: `key_of`'s key with
        each array or shape as its shape, dtype, weak type and sharding (`shapes_of`). Two calls of the
        program alike in it lower one program, so a begun compile's lowering serves both (`attempting`)."""
        return self._keyed(args, kwargs, _abstract_signature)[0]

    def _keyed(self, args: tuple, kwargs: dict, signature: Callable[..., Hashable]
               ) -> tuple[Hashable, tuple, dict]:
        """`key_of`'s key with each dynamic leaf as `signature` states it."""
        if not self.numbers and not self.names:
            leaves, structure = jax.tree.flatten((args, kwargs))
            return (len(args), ((), ()), structure, tuple(map(signature, leaves)),
                    jax_config.trace_context(), jax.sharding.get_mesh()), args, kwargs
        numbers = {number % len(args) for number in self.numbers if -len(args) <= number < len(args)}
        dynamic = tuple(arg for number, arg in enumerate(args) if number not in numbers)
        named = {name: value for name, value in kwargs.items() if name not in self.names}
        leaves, structure = jax.tree.flatten((dynamic, named))
        # A static value keys by its type too, as jit's do: True and 1 are two programs.
        statics = (tuple((number, type(args[number]), args[number]) for number in sorted(numbers)),
                   tuple(sorted(((name, type(value), value) for name, value in kwargs.items()
                                 if name in self.names), key=lambda static: static[0])))
        return (len(args), statics, structure, tuple(map(signature, leaves)),
                jax_config.trace_context(), jax.sharding.get_mesh()), dynamic, named


def _given_tuple(given) -> tuple:
    """jit's argnums or argnames as a tuple: none, one (an int or a str) or several."""
    return () if given is None else (given,) if isinstance(given, int | str) else tuple(given)


def trace_context() -> Hashable:
    """What a trace reads of this thread's context beside its function and its arguments' types: the key a
    trace kept outside the compile point is held by (a step's candidates, `dew.training._candidates`; a
    fused region's written program, `dew.algebra.fusion`). What JAX keys a trace on (its configuration's
    traced states, Dew's own run precision among them, its meshes and contexts, and Dew's contexts but those
    a trace records into, `_cache.RECORDING`), and the concrete mesh in place, read as it stands within a
    trace too (where `jax.sharding.get_mesh` refuses); what a model reads from the step's own entry, by value
    (`dew.nn.sharding.trace_context` and the declared axes' table frozen, in place of the variables that
    entry sets anew, `sharding.STATED`); every other context variable set in this context but those no trace
    reads (`_cache.untraced`, and Python's decimal context); the logical axis rules Flax holds; the
    environment's variables of Dew's own, each value by its contents (`_facts.stated`); and whether
    XLA's flags ask for deterministic operations (`_devices.deterministic_ops_requested`), which no `DEW_`
    variable holds and kernel selection reads where it selects (FlashAttention's and cuDNN's eligibility),
    not under the trace's target, so a trace kept under one setting would serve the other. The process's
    facts a trace reads (`_facts.traced_fact`) are no part of it: each is answered where a trace asks
    it, under the trace's own target, and one read outside a trace may have no answer (a compile-only GPU
    mesh's devices hold no memory statistics); a caller whose trace reads one names it beside this key (the
    compiles' refusals, `report.refusal_snapshot`). A value it leaves out costs a trace, never a program
    traced under other state."""
    import decimal
    import os

    from flax import linen as nn
    from jax._src import mesh as mesh_lib

    from dew._devices import deterministic_ops_requested
    from dew.nn import sharding

    # Read as no trace records into (`_cache.unrecorded`, the one owner of `RECORDING`'s defaults), so the key
    # holds alike under every receipt around it.
    with _cache.unrecorded():
        traced = jax_config.trace_context()
    variables = sorted(((variable, stated(value)) for variable, value in contextvars.copy_context().items()
                        if variable not in _cache.UNTRACED | sharding.STATED
                        and not isinstance(value, decimal.Context)),
                       key=lambda pair: (pair[0].name, id(pair[0])))
    entry = sharding.trace_context(), tuple(sorted(sharding.current_boxes().items(), key=repr))
    return (traced, mesh_lib.get_concrete_mesh(), entry, tuple(variables),
            stated(nn.get_logical_axis_rules()),
            tuple(sorted((name, value) for name, value in os.environ.items() if name.startswith("DEW_"))),
            deterministic_ops_requested())


def held_context() -> Hashable:
    """The key a trace kept outside the compile point is held by, beside what it is of: the state a trace
    reads of this thread's context (`trace_context`) and the launches the compiles refused, which a
    template's legality reads as the call traces (`report.refusal_snapshot`, a process's fact a trace reads
    and a caller names beside the trace's state). A step's candidates (`dew.training._candidates`) and the
    algebra's held kernel decisions (`fusion.choice_context`, the derivation's context beside it) are each
    held by it, so a fact added to it reaches both; a launch's derived config is keyed by what its
    derivation reads (`report.derivation_inputs`, `templates.base`)."""
    from dew.nn.kernels.templates import report

    return trace_context(), report.refusal_snapshot()


def _signature(leaf) -> Hashable:
    """What jit keys an argument on: an array's abstract value (shape, dtype, weak type), placement
    and layout (its `format`, which holds its sharding; a key array's sharding), another value's type,
    shape and dtype."""
    if isinstance(leaf, ArrayImpl):
        return leaf.aval, leaf.format, leaf.committed
    if isinstance(leaf, jax.Array):
        # A typed key array states no format: its placement is its sharding.
        return leaf.aval, leaf.sharding, leaf.committed
    return type(leaf), np.shape(leaf), np.result_type(leaf)


def abstract_signature(tree) -> Hashable:
    """`tree`'s arguments as a lowering reads them (`OuterProgram.abstract_of`): its structure and each
    leaf's abstract signature (`_abstract_signature`), arrays and their shapes alike: what a pass over shapes
    and the calls over arrays it compiles for are keyed by (`dew.training.compiler.HostRun`)."""
    leaves, structure = jax.tree.flatten(tree)
    return structure, tuple(_abstract_signature(leaf) for leaf in leaves)


def _abstract_signature(leaf) -> Hashable:
    """What a lowering reads of an argument (`OuterProgram.abstract_of`): an array's or a shape's shape,
    dtype and weak type (its abstract value, a key array's among them), its sharding and its device-local
    layout where it is no default, each as its shape (`shapes_of`), so a call over arrays and one over their
    shapes share it, an array no one committed to its devices lowering as an unplaced shape does, and a
    lowering over one layout is never taken for an array of another (the morning audit of 2026-11-16: a
    begun compile over a default layout taken for a column-major array, which its executable refused);
    another value's type, shape and dtype. A float0 (a zero tangent's) is its shape alone, a host value or
    its shape: no device holds one, and JAX hands a program's float0 result back as a host value, which a
    pass over shapes knows by its shape (a host step's pullback's cotangent, `dew.training.compiler`)."""
    if isinstance(leaf, np.ndarray | jax.ShapeDtypeStruct) and leaf.dtype == jax.dtypes.float0:
        return jax.dtypes.float0, np.shape(leaf)
    if isinstance(leaf, jax.Array | jax.ShapeDtypeStruct):
        shaped = shapes_of(leaf)
        return shaped.shape, shaped.dtype, shaped.weak_type, shaped.sharding, shaped.format.layout
    return type(leaf), np.shape(leaf), np.result_type(leaf)


UNGRAPHED_OPTIONS: Mapping[str, Option] = options_of(UNGRAPHED_FLAG)
"""The XLA options of a program compiled without CUDA command buffers (`program`'s `step`), as a process
given `dew.telemetry.benchmark.UNGRAPHED_FLAG` runs: a timed one's, whose command buffers CUPTI does not read
whole (`dew.telemetry.benchmark.Ungraphed`), and a server's admitting step's, whose fresh inputs update a
command buffer before it can replay (`dew.inference.serving_kernel.decode_programs`)."""


def program[**P, R](function: Callable[P, R], *, step: Mapping[str, Option] | None = None,
                    **arguments: Unpack[JitArguments]) -> OuterProgram:
    """`function` jitted with the XLA options a program compiles with (`compiler_options`), and, where it is
    part of a training step, the step's own beside them (`step`, `dew.training.memory.step_compiler_options`),
    with the rest of `jax.jit`'s `arguments` (its static arguments, the arguments it donates, its placements,
    a tree of formats among them), compiled through the kernel templates' reports (`OuterProgram`), which
    choose each lowering's GEMM backend from its own dots (`report.targeted`): one program for each function,
    options and arguments, which every call with the same reuses. Every outer program Dew compiles is one of
    these, a task's, an autoencoder's, a metric's, a load's, a server's (its steps,
    `dew.inference.serving_kernel.decode_programs`, their own options as `step`) or a simulator's, but those
    whose owner keeps them: an objective's (`Objective.program`, for what runs its modules, a rollout over an
    environment included), and a closure's or a bound method's a run makes (`owned_program`)."""
    options = {**compiler_options(), **(step or {})}
    return _program(
        function, tuple(sorted(options.items())),
        static_argnums=arguments.get("static_argnums"),
        static_argnames=arguments.get("static_argnames"), donate_argnums=arguments.get("donate_argnums"),
        donate_argnames=arguments.get("donate_argnames"),
        in_shardings=_Placed.of(arguments["in_shardings"]) if "in_shardings" in arguments else None,
        out_shardings=_Placed.of(arguments["out_shardings"]) if "out_shardings" in arguments else None)


def _fresh[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """`function` as a callable of its own, which JAX's trace cache keys anew (it keys a trace on the function
    a jit wraps): a jit of it traces afresh where one of `function` would read a trace held from before. Each
    run of it marked where a call reads its facts (`_read_by`), so a trace run is told from one JAX's cache
    answered."""
    @functools.wraps(function)
    def called(*args: P.args, **kwargs: P.kwargs) -> R:
        if (ran := _RAN.get()) is not None:
            ran.append(True)
        return function(*args, **kwargs)
    return called


_RAN: contextvars.ContextVar[list[bool] | None] = _cache.untraced(
    contextvars.ContextVar("outer_program_trace_ran", default=None))
"""Where a call of one of a program's jits marks its function's run (`_fresh`, `_read_by`)."""


def _read_by[R](call: Callable[[], R]) -> tuple[R, Facts | None]:
    """`call()`, a call of a jit of a program's (`_fresh`), and the facts its trace read, recorded as every
    open recording records them; None where JAX's trace cache answered it, its function run by none, whose
    caller reads again those its cached trace read (`_facts.replay`)."""
    ran: list[bool] = []
    token = _RAN.set(ran)
    try:
        with _facts.recording() as reads:
            returned = call()
    finally:
        _RAN.reset(token)
    return returned, reads if ran else None


def _weak_types(traced: jax.stages.Traced) -> tuple[bool, ...]:
    """A trace's outputs' weak types, in their leaves' order: what an executable's `out_info` drops and its
    results over arrays keep (`_outputs`); none for an output of no array (a token's)."""
    return tuple(isinstance(aval, jax_core.ShapedArray) and aval.weak_type
                 for aval in jax.tree.leaves(traced.out_avals))


def _outputs(out_info, weak_types: tuple[bool, ...]):
    """The results' shapes a program's `out_info` states, each with its trace's weak type (`_weak_types`):
    what a pass over shapes returns, so a later call over them keys as one over its arrays does."""
    shapes = shapes_of(out_info)
    leaves = [jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=shape.sharding, weak_type=weak)
              for shape, weak in zip(jax.tree.leaves(shapes), weak_types, strict=True)]
    return jax.tree.unflatten(jax.tree.structure(shapes), leaves)


def owned_program[**P, R](function: Callable[P, R], *, step: Mapping[str, Option] | None = None,
                          **arguments: Unpack[JitArguments]) -> OuterProgram:
    """`function` compiled as `program` compiles it, through Dew's compile point, but kept by no cache of
    Dew's: its caller holds it, and it goes with its caller's last reference. A closure or a bound method
    a run makes (its placement's, its host phases') takes this one, as `program`'s process-wide cache
    would keep it and the run it closes over (a trainer, its objective and their arrays) for good."""
    return OuterProgram(function, compiler_options={**compiler_options(), **(step or {})},
                        **arguments)


def jit[**P, R](function: Callable[P, R], **arguments: Unpack[JitArguments]) -> OuterProgram:
    """`function` compiled through Dew's compile point, as `jax.jit` compiles it (its static arguments, the
    arguments it donates, its placements): a plain JAX or Flax program's products, alone or with their
    elementwise consumers, run the kernel templates where their selectors give them (`dew.algebra.fusion`;
    `dew.kernels.use` names a choice), under the algebra pass, its gradients' too, whichever order
    `jax.grad` and it take. Its caller holds the program (`owned_program`)."""
    return owned_program(function, **arguments)


class Fetched:
    """`program`, of no static argument, as a host loop that calls it every step runs it: the executable of
    its first call in each trace context it is called in (`OuterProgram.executable`; a reference arm's run
    with the algebra pass off is another context), which each call after it in that context runs, so none
    builds the dispatch key a call reads from every argument (`OuterProgram.key_of`, which jit read in C++).
    A step of the loop keeps the first's shapes, dtypes and placements: its executable refuses other
    arguments, where the program would compile for them. A server's step, one executable for each size of its
    admission, runs as this does (`dew.inference.serving_kernel.dispatched`)."""

    def __init__(self, program: OuterProgram):
        if program.numbers or program.names:
            raise ValueError("an executable takes the arguments that are not static: a program with static "
                             "arguments is fetched as `OuterProgram.executable` is, by its caller")
        self.program = program
        self._executables: dict[Hashable, jax.stages.Compiled] = {}

    def __call__(self, *args, **kwargs):
        context = jax_config.trace_context()
        executable = self._executables.get(context)
        if executable is None:
            executable = self._executables[context] = self.program.executable(*args, **kwargs)
        return self.program.invoked(executable, args, kwargs, *args, **kwargs)


_Structure = tuple[jax.tree_util.PyTreeDef]
"""A tree's structure (`jax.tree.flatten`'s second result) in a one-element tuple: JAX declares
`PyTreeDef` as a value, which a type admits only inside another, as `training.distributed.Signature`'s."""


class _Placed(NamedTuple):
    """A tree of placements (`Placements`) as a program's cache holds it: its structure and its
    leaves, each hashable where a mapping of them is not."""
    structure: _Structure
    leaves: tuple

    @classmethod
    def of(cls, tree: Placements) -> _Placed:
        leaves, structure = jax.tree.flatten(tree)
        return cls((structure,), tuple(leaves))

    def tree(self) -> Placements:
        return jax.tree.unflatten(self.structure[0], self.leaves)


@functools.cache
def _program[**P, R](function: Callable[P, R], options: tuple[tuple[str, Option], ...], *,
                     static_argnums: int | tuple[int, ...] | None = None,
                     static_argnames: str | tuple[str, ...] | None = None,
                     donate_argnums: int | tuple[int, ...] | None = None,
                     donate_argnames: str | tuple[str, ...] | None = None,
                     in_shardings: _Placed | None = None, out_shardings: _Placed | None = None
                     ) -> OuterProgram:
    # Each argument `program`'s caller left out is left to jit (an explicit placement of None is a
    # `_Placed`): jit reads static positions from static names only where it is given no positions.
    arguments: JitArguments = {}
    if static_argnums is not None:
        arguments["static_argnums"] = static_argnums
    if static_argnames is not None:
        arguments["static_argnames"] = static_argnames
    if donate_argnums is not None:
        arguments["donate_argnums"] = donate_argnums
    if donate_argnames is not None:
        arguments["donate_argnames"] = donate_argnames
    if in_shardings is not None:
        arguments["in_shardings"] = in_shardings.tree()
    if out_shardings is not None:
        arguments["out_shardings"] = out_shardings.tree()
    return OuterProgram(function, compiler_options=dict(options), **arguments)
