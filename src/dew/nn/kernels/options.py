"""The XLA options a program compiles with, by the models it runs and the device it runs on, and the
one place a program over them is compiled (`program`).

`dew.nn.kernels.generation` stays free of Dew's imports, since benchmark
environments load its target table without Dew or JAX; this reads its
facts with the run's own XLA flags.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import ClassVar, NamedTuple, Protocol, TypedDict, Unpack, runtime_checkable

import jax
import numpy as np
from jax.sharding import Sharding

from dew.devices import xla_flag
from dew.nn.kernels.generation import backend, gpu_target


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


def quantizations(*models: object) -> list[QuantizedDtypes]:
    """The quantization every model class among `models` carries (`Quantized`), and every tower they
    hold: each module a model, a task's encoder or any record it is handed holds in its fields,
    a tuple's or a mapping's entries, at any depth, so a quantized text encoder inside a condition's
    encoder counts as the denoiser beside it does. Arrays and their trees are not walked."""
    found: list[QuantizedDtypes] = []
    seen: set[int] = set()

    def walk(value: object) -> None:
        if id(value) in seen or isinstance(value, (jax.Array, np.ndarray, str, bytes, int, float, type)):
            return
        seen.add(id(value))
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


def compiler_options(*models: object) -> dict[str, bool]:
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
    `xla_gpu_enable_triton_gemm` in XLA_FLAGS keeps its own setting.
    """
    int8 = any("int8" in (spec.dtype, spec.bwd_qtype) for spec in quantizations(*models))
    if (not int8 or backend() != "gpu" or gpu_target().int8_triton_gemm
            or xla_flag("xla_gpu_enable_triton_gemm") is not None):
        return {}
    return {"xla_gpu_enable_triton_gemm": False}


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


def program[**P, R](function: Callable[P, R], *models: object,
                    step: Mapping[str, bool | int] | None = None,
                    **arguments: Unpack[JitArguments]) -> jax.stages.Wrapped:
    """`function` jitted with the XLA options a program over `models` compiles with
    (`compiler_options`), and, where it is part of a training step, the step's own beside them
    (`step`, `dew.training.memory.step_compiler_options`), with the rest of `jax.jit`'s `arguments`
    (its static arguments, its shardings): one program for each function, options and arguments,
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
def _program[**P, R](function: Callable[P, R], options: tuple[tuple[str, bool | int], ...], *,
                     static_argnums: int | tuple[int, ...] | None = None,
                     static_argnames: str | tuple[str, ...] | None = None,
                     in_shardings: _Placed | None = None, out_shardings: _Placed | None = None
                     ) -> jax.stages.Wrapped:
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
    return jax.jit(function, compiler_options=dict(options), **arguments)
