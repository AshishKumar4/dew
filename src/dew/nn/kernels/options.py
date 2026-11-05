"""The XLA options a program compiles with, by the models it runs and the device it runs on, and the
one place a program over them is compiled (`program`).

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
import gc
import threading
import weakref
from collections.abc import Callable, Hashable, Iterator, Mapping
from concurrent.futures import Future
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Protocol, TypedDict, Unpack, runtime_checkable

import jax
import numpy as np
from jax._src import api_util, config as jax_config, core as jax_core
from jax._src.array import ArrayImpl
from jax.experimental.layout import Format
from jax.sharding import Sharding

from dew import algebra
from dew.devices import xla_flag
from dew.nn.kernels.generation import backend, gpu_target
from dew.nn.nested import staged

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from flax import linen as nn


class QuantizedDtypes(Protocol):
    """What the options read of a run's quantization (`dew.quantization.Quantization`, above the kernels):
    the dtypes its forward and backward quantize to."""

    @property
    def dtype(self) -> str: ...

    @property
    def bwd_qtype(self) -> str | None: ...


@runtime_checkable
class Quantized(Protocol):
    """A model class that carries a run's quantization (`dew.quantization.QuantizedClass`), as the options
    read it: its two dtypes."""

    _dew_quantization: ClassVar[QuantizedDtypes]


type Held = nn.Module | DataclassInstance | Mapping[str, Held] | tuple[Held, ...] | list[Held] | None
"""What a program runs that may hold a quantized tower (`quantizations`): a model, a record of one (a
condition's encoder), or a mapping or sequence of them; anything else a record holds is passed over."""


def quantizations(*models: Held) -> list[QuantizedDtypes]:
    """The quantization every model class among `models` carries (`Quantized`), and every tower they
    hold: each module a model, a task's encoder or any record it is handed holds in its fields,
    a tuple's or a mapping's entries, at any depth, so a quantized text encoder inside a condition's
    encoder counts as the denoiser beside it does. Arrays and their trees are not walked."""
    found: list[QuantizedDtypes] = []
    # Each object visited, held until the walk ends: a mapping's values may be made as they are read
    # (a FrozenDict's nested ones), and a freed one's id could name a later, unvisited value.
    seen: dict[int, Held] = {}

    def walk(value: Held) -> None:
        if id(value) in seen or isinstance(value, (jax.Array, np.ndarray, str, bytes, int, float, type)):
            return
        seen[id(value)] = value
        if isinstance(value, Quantized):
            found.append(type(value)._dew_quantization)
        if isinstance(value, Mapping):
            for entry in value.values():
                walk(entry)
        elif isinstance(value, (tuple, list)):
            for entry in value:
                walk(entry)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                if field.name not in ("parent", "name"):
                    walk(getattr(value, field.name, None))

    for model in models:
        walk(model)
    return found


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


def compiler_options(*models: Held) -> dict[str, Option]:
    """The XLA options a program that runs `models` compiles with on the default device, for its
    `jax.jit(..., compiler_options=)`. Every program Dew compiles over a task's model,
    autoencoder or encoders, or over an objective's modules, takes them, for every tower they hold
    (`quantizations`). JAX takes compiler options on a program's outermost jit alone, so a module's
    own jitted method runs under the options of the program that calls it.

    XLA's Triton GEMM fusion computes some int8 dots as two or four times their product
    (openxla/xla#51016), silently, on every GPU measured (`GpuTarget.int8_triton_gemm`). A program
    that runs a model quantized to int8, forward or backward, therefore compiles its GEMMs on
    cuBLAS. With CUDA 13, cuBLAS refuses some int8 shapes (on an A100 a contraction of 16 or 48),
    and then the program does not compile rather than compute wrong. A run that sets
    `xla_gpu_enable_triton_gemm` in XLA_FLAGS keeps its own setting. The options `compiled_with`
    holds open join them.
    """
    int8 = any("int8" in (spec.dtype, spec.bwd_qtype) for spec in quantizations(*models))
    options: dict[str, Option] = {}
    if (int8 and backend() == "gpu" and not gpu_target().int8_triton_gemm
            and xla_flag("xla_gpu_enable_triton_gemm") is None):
        options["xla_gpu_enable_triton_gemm"] = False
    for opened in _COMPILED_WITH.get():
        options.update({name: value for name, value in opened.items() if xla_flag(name) is None})
    return options


TRACING_COLLECTIONS = 50_000
"""The allocations between collections of Python's youngest generation while a plan search runs
(`held_objects`), past CPython's 700: its traces allocate millions of short-lived objects."""

_HELD = 0
"""The blocks of `held_objects` open, across nesting and threads."""

_HELD_GUARD = threading.Lock()

_THRESHOLDS: list[int] = []
"""Python's collection thresholds as the first open block of `held_objects` found them."""


@contextlib.contextmanager
def held_objects() -> Iterator[None]:
    """Within the block, the objects alive when the first open block began are frozen out of Python's
    collector (`gc.freeze`), as a run's model, its state's specs and JAX's own tables are, and the youngest
    generation collects every `TRACING_COLLECTIONS` allocations: a plan search's traces, compiles and
    bookkeeping make millions of short-lived objects, and each full collection walked the long-lived ones
    beside them (row 2440: 59 full collections paused a Qwen3.5-4B step's compile_ahead 73 s of 342,
    FS2440-GC-PLAIN; held over the search, 7 paused it 7 s and it took 279, FS2440-GC-LIBRARY2; held over
    each trace alone, it took 334, FS2440-GC-LIBRARY). Nested and concurrent blocks share one freeze; the
    last to leave restores the thresholds and returns the frozen objects to the collector (`gc.unfreeze`)."""
    global _HELD
    with _HELD_GUARD:
        if not _HELD:
            _THRESHOLDS[:] = gc.get_threshold()
            gc.freeze()
            gc.set_threshold(TRACING_COLLECTIONS, *_THRESHOLDS[1:])
        _HELD += 1
    try:
        yield
    finally:
        with _HELD_GUARD:
            _HELD -= 1
            if not _HELD:
                gc.set_threshold(*_THRESHOLDS)
                gc.unfreeze()


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


_PROGRAMS: weakref.WeakSet[OuterProgram] = weakref.WeakSet()
"""Every program alive (`OuterProgram`), whose executables `release` drops."""


def release() -> None:
    """Drop every live program's compiled executables (`OuterProgram.released`), as `jax.clear_caches`
    drops jit's own: XLA:CPU maps each executable's code into the process, and a program keeps up to
    `OuterProgram.CACHED` of them for as long as it lives, past a cache JAX clears. A process that compiles
    on and on (a test suite) frees the mappings this way; each program compiles again at its next call."""
    for program in list(_PROGRAMS):
        program.released()


class OuterProgram:
    """`function` jitted (`jax.jit`, under `arguments`), every compile of it read through the kernel
    templates' reports (`dew.nn.kernels.templates.report.compiled`): where a launch's kernel spills, its
    config is refused and the function traced afresh, by a new jit of a new callable past JAX's trace
    cache, and compiled once more, the launch's derivation taking the next config. Called as its jit
    is, it runs the executable that what jit keys its own on chooses (`_keyed`): the static arguments,
    the others' structure, shapes, dtypes and placements, and the trace context, the `CACHED` latest compiled;
    `lower` and `trace` are its jit's. Called within another trace (a `jax.grad`, a `jax.vmap`, an
    enclosing jit), it is its jit, which the enclosing program compiles as its own."""

    CACHED: ClassVar[int] = 4096
    """The executables a program keeps, the latest compiled, so its cache is bounded as jit's is: a call of
    one compiled before reads it without refreshing its place (row 3326)."""

    def __init__(self, function: Callable, **arguments):
        self.function, self.arguments = function, arguments
        self.jitted = self._jitted()
        # The static positions and names as jit resolves them, the ones given and those it infers.
        numbers, names = arguments.get("static_argnums"), arguments.get("static_argnames")
        signature = api_util.fun_signature(function)
        if signature is not None:
            numbers, names = api_util.infer_argnums_and_argnames(signature, numbers, names)
        self.numbers = () if numbers is None else (numbers,) if isinstance(numbers, int) else tuple(numbers)
        self.names = frozenset(() if names is None else (names,) if isinstance(names, str) else names)
        self._executables: collections.OrderedDict[Hashable, Future[jax.stages.Compiled]] = (
            collections.OrderedDict())
        # Each compiled executable by its key, read without the lock: a call of a signature compiled before
        # costs its key and one lookup (row 3326).
        self._ready: dict[Hashable, jax.stages.Compiled] = {}
        self._compiling = threading.Lock()
        _PROGRAMS.add(self)

    def released(self) -> None:
        """Drop every executable it keeps (`release`), each compiled again at its next call."""
        with self._compiling:
            self._ready.clear()
            self._executables.clear()

    def _jitted(self) -> jax.stages.Wrapped:
        """Its jit, of a callable of its own, past JAX's trace cache (a relowering's, `compiled`)."""
        @functools.wraps(self.function)
        def traced(*args, **kwargs):
            return self.function(*args, **kwargs)
        return jax.jit(traced, **self.arguments)

    def _traced(self, jitted: jax.stages.Wrapped, args: tuple, kwargs: dict) -> jax.stages.Traced:
        """`jitted`'s trace of a call, its program the algebra pass's (`algebra.program_of`), so its lowering,
        its trace, its executable and every relowering read one program (docs/design/algebra.md §6.7): the
        pass reads the trace jit made, which lowers as it is where no rule changes it; a program a rule
        changes is traced afresh from its rewritten program (`program.replaying`). Its switch is in JAX's
        trace context, so a held program traces afresh as it changes."""
        traced = jitted.trace(*args, **kwargs)
        rewritten = algebra.program_of(traced.jaxpr)
        if rewritten is traced.jaxpr:
            return traced
        replay = algebra.program.replaying(rewritten, jax.tree.structure(traced.out_info), self.numbers,
                                           self.names)
        return jax.jit(functools.wraps(self.function)(replay), **self.arguments).trace(*args, **kwargs)

    @functools.cached_property
    def _eager(self) -> jax.stages.Wrapped:
        """Its jit called under an eager transformation, which traces it where no trace of it can be read
        first: the function rewritten as it traces (`algebra.rewritten`), one program as its compiles'."""
        @functools.wraps(self.function)
        def traced(*args, **kwargs):
            return algebra.rewritten(self.function, static_argnums=self.numbers,
                                     static_argnames=tuple(self.names))(*args, **kwargs)
        return jax.jit(traced, **self.arguments)

    @functools.cached_property
    def _nested(self) -> jax.stages.Wrapped:
        """Its jit within another trace: the enclosing program's own options are the ones it compiles under,
        as JAX takes compiler options on the outermost jit alone."""
        arguments = {name: value for name, value in self.arguments.items() if name != "compiler_options"}
        return jax.jit(self.function, **arguments)

    def __call__(self, *args, **kwargs):
        # Within a staging trace the enclosing program compiles this one as part of its own, under its
        # own options (`_nested`); under an eager transformation alone, or `jax.disable_jit`, its jit
        # compiles itself with its options, or runs the function as Python.
        if staged():
            return self._nested(*args, **kwargs)
        if not jax_core.trace_state_clean() or jax_config.disable_jit.value:
            return self._eager(*args, **kwargs)
        key, dynamic, named = self._keyed(args, kwargs)
        ready = self._ready.get(key)
        return (self._kept(key, args, kwargs) if ready is None else ready)(*dynamic, **named)

    def executable(self, *args, **kwargs) -> jax.stages.Compiled:
        """The executable a call with these arguments runs, compiled at the first: called with the dynamic
        arguments alone, as a caller that calls it every step does, building no dispatch key."""
        key, _, _ = self._keyed(args, kwargs)
        ready = self._ready.get(key)
        return self._kept(key, args, kwargs) if ready is None else ready

    def _kept(self, key: Hashable, args: tuple, kwargs: dict) -> jax.stages.Compiled:
        """The executable kept under `key`, compiled where none is (`compiled`): the map alone is held,
        each key's compile a future its first caller sets, so calls of other keys compile at once."""
        with self._compiling:
            pending = self._executables.get(key)
            owner = pending is None
            if owner:
                pending = self._executables[key] = Future()
                if len(self._executables) > self.CACHED:
                    self._ready.pop(self._executables.popitem(last=False)[0], None)
            self._executables.move_to_end(key)
        if owner:
            try:
                executable = self.compiled(*args, **kwargs)
                pending.set_result(executable)
                with self._compiling:
                    if key in self._executables:
                        self._ready[key] = executable
            except BaseException as error:
                with self._compiling:
                    self._executables.pop(key, None)
                pending.set_exception(error)
        return pending.result()

    def compiled(self, *args, **kwargs) -> jax.stages.Compiled:
        """The program over these arguments compiled through the kernel templates' reports, lowered
        afresh where a kernel spills."""
        from dew.nn.kernels.templates import report

        relowered: list[jax.stages.Wrapped] = []

        def relower() -> jax.stages.Lowered:
            fresh = self._jitted()
            lowered = self._traced(fresh, args, kwargs).lower()
            relowered.append(fresh)
            return lowered
        executable = report.compiled(self.lower(*args, **kwargs), relower=relower)
        # Later calls trace as the relowering did, past the trace its refused config is in.
        if relowered:
            self.jitted = relowered[-1]
        return executable

    def lower(self, *args, **kwargs) -> jax.stages.Lowered:
        return self.trace(*args, **kwargs).lower()

    def trace(self, *args, **kwargs) -> jax.stages.Traced:
        return self._traced(self.jitted, args, kwargs)

    def _keyed(self, args: tuple, kwargs: dict) -> tuple[Hashable, tuple, dict]:
        """The key of the executable a call runs, and the arguments the executable takes, the static
        ones dropped. It holds what jit keys on: the trace context, with its abstract mesh, and the
        concrete mesh in place (`jax.sharding.get_mesh`), so a mesh of the same shape over other devices
        compiles its own executable."""
        if not self.numbers and not self.names:
            leaves, structure = jax.tree.flatten((args, kwargs))
            return (len(args), ((), ()), structure, tuple(map(_signature, leaves)),
                    jax_config.trace_context(), jax.sharding.get_mesh()), args, kwargs
        numbers = {number % len(args) for number in self.numbers if -len(args) <= number < len(args)}
        dynamic = tuple(arg for number, arg in enumerate(args) if number not in numbers)
        named = {name: value for name, value in kwargs.items() if name not in self.names}
        leaves, structure = jax.tree.flatten((dynamic, named))
        # A static value keys by its type too, as jit's do: True and 1 are two programs.
        statics = (tuple((number, type(args[number]), args[number]) for number in sorted(numbers)),
                   tuple(sorted(((name, type(value), value) for name, value in kwargs.items()
                                 if name in self.names), key=lambda static: static[0])))
        return (len(args), statics, structure, tuple(map(_signature, leaves)),
                jax_config.trace_context(), jax.sharding.get_mesh()), dynamic, named


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


def program[**P, R](function: Callable[P, R], *models: Held,
                    step: Mapping[str, Option] | None = None,
                    **arguments: Unpack[JitArguments]) -> OuterProgram:
    """`function` jitted with the XLA options a program over `models` compiles with
    (`compiler_options`), and, where it is part of a training step, the step's own beside them
    (`step`, `dew.training.memory.step_compiler_options`), with the rest of `jax.jit`'s `arguments`
    (its static arguments, the arguments it donates, its placements, a tree of formats among them),
    compiled through the kernel templates' reports (`OuterProgram`): one program for each function,
    options and arguments, which every call with the same reuses, so a quantized model and another never
    share one. Every outer program Dew compiles is one of these, a task's, an autoencoder's, a metric's, a
    load's, a server's (its steps, `dew.inference.serving_kernel.decode_programs`, their own options as
    `step`) or a simulator's, but those whose owner keeps them: an objective's (`Objective.program`, for
    what runs its modules, a rollout over an environment included), and a closure's or a bound method's a
    run makes (`owned_program`)."""
    options = {**compiler_options(*models), **(step or {})}
    return _program(
        function, tuple(sorted(options.items())), static_argnums=arguments.get("static_argnums"),
        static_argnames=arguments.get("static_argnames"), donate_argnums=arguments.get("donate_argnums"),
        donate_argnames=arguments.get("donate_argnames"),
        in_shardings=_Placed.of(arguments["in_shardings"]) if "in_shardings" in arguments else None,
        out_shardings=_Placed.of(arguments["out_shardings"]) if "out_shardings" in arguments else None)


def owned_program[**P, R](function: Callable[P, R], *models: Held,
                          step: Mapping[str, Option] | None = None,
                          **arguments: Unpack[JitArguments]) -> OuterProgram:
    """`function` compiled as `program` compiles it, through Dew's compile point, but kept by no cache of
    Dew's: its caller holds it, and it goes with its caller's last reference. A closure or a bound method
    a run makes (its placement's, its host phases') takes this one, as `program`'s process-wide cache
    would keep it and the run it closes over (a trainer, its objective and their arrays) for good."""
    return OuterProgram(function, compiler_options={**compiler_options(*models), **(step or {})},
                        **arguments)


class Fetched:
    """`program`, of no static argument, as a host loop that calls it every step runs it: the executable of
    its first call in each trace context it is called in (`OuterProgram.executable`; a reference arm's run
    with the algebra pass off is another context), which each call after it in that context runs, so none
    builds the dispatch key a call reads from every argument (`OuterProgram._keyed`, which jit read in C++).
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
        return executable(*args, **kwargs)


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
