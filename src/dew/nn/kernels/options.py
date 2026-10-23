"""The XLA options a program compiles with, by the models it runs and the device it runs on.

`dew.nn.kernels.generation` stays free of Dew's imports, since benchmark
environments load its target table without Dew or JAX; this reads its
facts with the run's own XLA flags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

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


def compiler_options(*models: nn.Module) -> dict[str, bool]:
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
    `xla_gpu_enable_triton_gemm` in XLA_FLAGS keeps its own setting.
    """
    specs = [layer._dew_quantization for layer in map(type, models) if isinstance(layer, Quantized)]
    int8 = any("int8" in (spec.dtype, spec.bwd_qtype) for spec in specs)
    if (not int8 or backend() != "gpu" or gpu_target().int8_triton_gemm
            or xla_flag("xla_gpu_enable_triton_gemm") is not None):
        return {}
    return {"xla_gpu_enable_triton_gemm": False}
