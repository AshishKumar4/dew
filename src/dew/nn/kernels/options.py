"""The XLA options a program compiles with, by the models it runs and the device it runs on, and the
one place a program over them is compiled (`program`).

`dew.nn.kernels.generation` stays free of Dew's imports, since benchmark
environments load its target table without Dew or JAX; this reads its
facts with the run's own XLA flags.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, ClassVar, Protocol, TypedDict, Unpack, runtime_checkable

import jax
from jax.sharding import Sharding

from dew.devices import xla_flag
from dew.nn.kernels.generation import backend, gpu_target

if TYPE_CHECKING:
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


_COMPILED_WITH: list[dict[str, bool | int | str]] = []
"""The options `compiled_with` holds open, innermost last."""


@contextlib.contextmanager
def compiled_with(options: Mapping[str, bool | int | str]) -> Iterator[None]:
    """Every program Dew compiles while this is open adds `options` to its XLA options
    (`compiler_options`), each but those the run's XLA_FLAGS sets itself: the accuracy gate's seam,
    through which an arm that runs Dew's own compiled programs (a trainer's step, a task's forward)
    compiles them as the gate compiles a Dew arm it compiles itself (tests/reference_error.py,
    `dew_programs`). A program compiled before it opened keeps its own options."""
    _COMPILED_WITH.append(dict(options))
    try:
        yield
    finally:
        _COMPILED_WITH.pop()


def compiler_options(*models: nn.Module) -> dict[str, bool | int | str]:
    """The XLA options a program that runs `models` compiles with on the default device, for its
    `jax.jit(..., compiler_options=)`. Every program Dew compiles over a task's model or
    autoencoder, or over an objective's modules, takes them. JAX takes compiler options on a
    program's outermost jit alone, so a module's own jitted method runs under the options of the
    program that calls it.

    XLA's Triton GEMM fusion computes some int8 dots as two or four times their product
    (openxla/xla#51016), silently, on every GPU measured (`GpuTarget.int8_triton_gemm`). A program
    that runs a model quantized to int8, forward or backward, therefore compiles its GEMMs on
    cuBLAS. With CUDA 13, cuBLAS refuses some int8 shapes (on an A100 a contraction of 16 or 48),
    and then the program does not compile rather than compute wrong. A run that sets
    `xla_gpu_enable_triton_gemm` in XLA_FLAGS keeps its own setting. The options `compiled_with`
    holds open join them.
    """
    specs = [layer._dew_quantization for layer in map(type, models) if isinstance(layer, Quantized)]
    int8 = any("int8" in (spec.dtype, spec.bwd_qtype) for spec in specs)
    options: dict[str, bool | int | str] = {}
    if (int8 and backend() == "gpu" and not gpu_target().int8_triton_gemm
            and xla_flag("xla_gpu_enable_triton_gemm") is None):
        options["xla_gpu_enable_triton_gemm"] = False
    for opened in _COMPILED_WITH:
        options.update({name: value for name, value in opened.items() if xla_flag(name) is None})
    return options


type Placements = Sharding | None | tuple[Placements, ...]
"""Where a program's arguments or results sit, as `jax.jit` takes them: one sharding, none (where
they are), or a tuple of either, one for each."""


class JitArguments(TypedDict, total=False):
    """The arguments of `jax.jit` a program over models takes (`program`), each hashable."""

    static_argnums: int | tuple[int, ...]
    static_argnames: str | tuple[str, ...]
    in_shardings: Placements
    out_shardings: Placements


def program[**P, R](function: Callable[P, R], *models: nn.Module,
                    step: Mapping[str, bool | int] | None = None,
                    **arguments: Unpack[JitArguments]) -> jax.stages.Wrapped:
    """`function` jitted with the XLA options a program over `models` compiles with
    (`compiler_options`), and, where it is part of a training step, the step's own beside them
    (`step`, `dew.training.memory.step_compiler_options`), with the rest of `jax.jit`'s `arguments`
    (its static arguments, its shardings): one program for each function, options and arguments,
    which every call with the same reuses, so a quantized model and another never share one. Every
    program Dew compiles over a task's model, an autoencoder or an objective's modules outside an
    objective's own (`Objective.program`) is one of these."""
    options = {**compiler_options(*models), **(step or {})}
    return _program(function, tuple(sorted(options.items())), **arguments)


@functools.cache
def _program[**P, R](function: Callable[P, R], options: tuple[tuple[str, bool | int | str], ...],
                     **arguments: Unpack[JitArguments]) -> jax.stages.Wrapped:
    return jax.jit(function, compiler_options=dict(options), **arguments)
