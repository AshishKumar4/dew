"""The XLA options a program compiles with, by the models it runs and the device it runs on, and the
one place a program over them is compiled (`program`).

`dew.nn.kernels.generation` stays free of Dew's imports, since benchmark
environments load its target table without Dew or JAX; this reads its
facts with the run's own XLA flags.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import inspect
import threading
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Protocol, TypedDict, Unpack, runtime_checkable

import jax
import numpy as np
from jax._src import config as jax_config
from jax.sharding import Sharding

from dew.devices import xla_flag
from dew.nn.kernels.generation import backend, gpu_target

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


type Placements = Sharding | None | tuple[Placements, ...] | Mapping[str, Placements]
"""Where a program's arguments or results sit, as `jax.jit` takes them: one sharding, none (where
they are), or a tree of either, one for each."""


class JitArguments(TypedDict, total=False):
    """The arguments of `jax.jit` a program takes (`program`): its static arguments, and its placements,
    whose trees `program` holds by their structure and leaves, each hashable."""

    static_argnums: int | tuple[int, ...]
    static_argnames: str | tuple[str, ...]
    in_shardings: Placements
    out_shardings: Placements


class OuterProgram:
    """`function` jitted (`jax.jit`, under `arguments`), every compile of it read through the kernel
    templates' reports (`dew.nn.kernels.templates.report.compiled`): where a launch's kernel spills, its
    config is refused and the function traced afresh, by a new jit of a new callable past JAX's trace
    cache, and compiled once more, the launch's derivation taking the next config. Called as its jit
    is, it runs the executable that what jit keys its own on chooses (`_keyed`): the static arguments,
    the others' structure, shapes, dtypes and placements, and the trace context; `lower` and `trace`
    are its jit's."""

    def __init__(self, function: Callable, **arguments):
        self.function, self.arguments = function, arguments
        self.jitted = self._jitted()
        self.numbers, self.names = _statics(function, arguments.get("static_argnums"),
                                            arguments.get("static_argnames"))
        self._executables: dict[Hashable, jax.stages.Compiled] = {}
        self._compiling = threading.Lock()

    def _jitted(self) -> jax.stages.Wrapped:
        @functools.wraps(self.function)
        def traced(*args, **kwargs):
            return self.function(*args, **kwargs)
        return jax.jit(traced, **self.arguments)

    def __call__(self, *args, **kwargs):
        key, dynamic, named = self._keyed(args, kwargs)
        executable = self._executables.get(key)
        if executable is None:
            with self._compiling:
                executable = self._executables.get(key) or self.compiled(*args, **kwargs)
                self._executables[key] = executable
        return executable(*dynamic, **named)

    def compiled(self, *args, **kwargs) -> jax.stages.Compiled:
        """The program over these arguments compiled through the kernel templates' reports, lowered
        afresh where a kernel spills."""
        from dew.nn.kernels.templates import report

        return report.compiled(self.jitted.lower(*args, **kwargs),
                               relower=lambda: self._jitted().lower(*args, **kwargs))

    def lower(self, *args, **kwargs) -> jax.stages.Lowered:
        return self.jitted.lower(*args, **kwargs)

    def trace(self, *args, **kwargs) -> jax.stages.Traced:
        return self.jitted.trace(*args, **kwargs)

    def _keyed(self, args: tuple, kwargs: dict) -> tuple[Hashable, tuple, dict]:
        """The key of the executable a call runs, and the arguments the executable takes, the static
        ones dropped."""
        numbers = {number % len(args) for number in self.numbers if -len(args) <= number < len(args)}
        dynamic = tuple(arg for number, arg in enumerate(args) if number not in numbers)
        named = {name: value for name, value in kwargs.items() if name not in self.names}
        leaves, structure = jax.tree.flatten((dynamic, named))
        statics = (tuple((number, args[number]) for number in sorted(numbers)),
                   tuple(sorted((name, value) for name, value in kwargs.items() if name in self.names)))
        return (len(args), statics, structure, tuple(map(_signature, leaves)),
                jax_config.trace_context()), dynamic, named


def _statics(function: Callable, numbers: int | tuple[int, ...] | None,
             names: str | tuple[str, ...] | None) -> tuple[tuple[int, ...], frozenset[str]]:
    """The static positions and names of `function` as `jax.jit` reads them: the ones given, and
    where only one kind is, the other's from its signature."""
    numbers = () if numbers is None else (numbers,) if isinstance(numbers, int) else tuple(numbers)
    names = () if names is None else (names,) if isinstance(names, str) else tuple(names)
    if bool(numbers) == bool(names):
        return numbers, frozenset(names)
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return numbers, frozenset(names)
    positional = [parameter.name for parameter in parameters
                  if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
    if names:
        return tuple(positional.index(name) for name in names if name in positional), frozenset(names)
    named = {parameter.name for parameter in parameters if parameter.kind == parameter.POSITIONAL_OR_KEYWORD}
    return numbers, frozenset(positional[number] for number in numbers
                              if -len(positional) <= number < len(positional)
                              and positional[number] in named)


def _signature(leaf) -> Hashable:
    """What jit keys an argument on: an array's abstract value (shape, dtype, weak type) and
    placement, another value's type, shape and dtype."""
    if isinstance(leaf, jax.Array):
        return leaf.aval, leaf.sharding, leaf.committed
    return type(leaf), np.shape(leaf), np.result_type(leaf)


def program[**P, R](function: Callable[P, R], *models: Held,
                    step: Mapping[str, Option] | None = None,
                    **arguments: Unpack[JitArguments]) -> OuterProgram:
    """`function` jitted with the XLA options a program over `models` compiles with
    (`compiler_options`), and, where it is part of a training step, the step's own beside them
    (`step`, `dew.training.memory.step_compiler_options`), with the rest of `jax.jit`'s `arguments`
    (its static arguments, its shardings), compiled through the kernel templates' reports (`OuterProgram`):
    one program for each function, options and arguments,
    which every call with the same reuses, so a quantized model and another never share one. Every
    outer program Dew compiles is one of these, a task's, an autoencoder's, a metric's or a load's,
    but two that own their programs' lifetime: an objective's (`Objective.program`), which its cache
    keeps as long as the objective, and the serving kernel's step programs, which donate the cache
    they step (`dew.inference.serving_kernel.decode_programs`)."""
    options = {**compiler_options(*models), **(step or {})}
    return _program(
        function, tuple(sorted(options.items())), static_argnums=arguments.get("static_argnums"),
        static_argnames=arguments.get("static_argnames"),
        in_shardings=_Placed.of(arguments["in_shardings"]) if "in_shardings" in arguments else None,
        out_shardings=_Placed.of(arguments["out_shardings"]) if "out_shardings" in arguments else None)


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
                     in_shardings: _Placed | None = None, out_shardings: _Placed | None = None
                     ) -> OuterProgram:
    # Each argument `program`'s caller left out is left to jit (an explicit placement of None is a
    # `_Placed`): jit reads static positions from static names only where it is given no positions.
    arguments: JitArguments = {}
    if static_argnums is not None:
        arguments["static_argnums"] = static_argnums
    if static_argnames is not None:
        arguments["static_argnames"] = static_argnames
    if in_shardings is not None:
        arguments["in_shardings"] = in_shardings.tree()
    if out_shardings is not None:
        arguments["out_shardings"] = out_shardings.tree()
    return OuterProgram(function, compiler_options=dict(options), **arguments)
