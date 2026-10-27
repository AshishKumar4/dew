"""A training step lowered for a CUDA GPU on a host without one.

Dew's Pallas GPU kernels lower to Triton only for a GPU: on a CPU the suite
interprets them, so a kernel whose Triton lowering fails passes every CPU
test and fails the run's first compile on the GPU (c58 and c57, an A100 and
a G4 session, both ended there). `TrainerConfig.lower_for` names a CUDA
generation, and the run then builds as it would and lowers its first step as
on that GPU: every kernel choice and refusal reads the generation
(`dew.nn.kernels.generation.assumed`), the step traces on an abstract mesh of
that GPU's devices, the mesh JAX's Pallas lowering reads the GPU from, and
lowers for CUDA, which runs every Pallas-Triton lowering in the step. XLA's
own compile of the lowered step, and Triton's of each kernel, still need the
GPU.

A choice keyed on the platform reads `generation.backend()`, the assumed
generation's platform, and takes the GPU's path here. Two read
`generation.gpu_placed()`, this process's own GPU, which an assumed one never
is, so the dry run cannot see their GPU path: attention's cuDNN and
FlashAttention calls (`attention_kernels._hosted`), CUDA kernels and not Pallas ones,
whose trace asks the CUDA plugin for versions and libraries a CPU host lacks
and whose lowering asks for devices an abstract mesh lacks, even on a GPU
host, so attention traces its XLA path.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import jax
from jax.extend.mlir import ir

TRITON = "__gpu$xla.gpu.triton"
"""The custom call Pallas lowers a Triton kernel to."""


@dataclass(frozen=True)
class CustomCall:
    """One custom call of a lowered program: its target and the shapes of the arrays it reads and
    writes."""

    target: str
    operands: tuple[tuple[int, ...], ...]
    results: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TritonCall:
    """One Pallas-Triton call of a lowered program, as Pallas configured it (`mhlo.backend_config`)."""

    name: str
    ir: bytes
    """The kernel's Triton IR, serialized as Pallas wrote it."""
    warps: int
    stages: int


def _read_calls[T](lowered: jax.stages.Lowered, read: Callable[[str, ir.Operation], T | None]) -> list[T]:
    """What `read` makes of each custom call a lowered program makes, given its target and its
    operation, in program order: the calls it makes something of."""
    found: list[T] = []

    def visit(operation: ir.Operation) -> ir.WalkResult:
        if operation.name == "stablehlo.custom_call":
            made = read(ir.StringAttr(operation.attributes["call_target_name"]).value, operation)
            if made is not None:
                found.append(made)
        return ir.WalkResult.ADVANCE

    module = lowered.compiler_ir(dialect="stablehlo")
    assert module is not None, "a lowered program always has its StableHLO module"
    with module.context:
        module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return found


def custom_calls(lowered: jax.stages.Lowered) -> list[CustomCall]:
    """Every custom call a lowered program makes, in program order."""
    def shapes(values) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(value.type.shape) for value in values
                     if isinstance(value.type, ir.RankedTensorType))

    return _read_calls(lowered, lambda target, operation: CustomCall(
        target, shapes(operation.operands), shapes(operation.results)))


def triton_calls(lowered: jax.stages.Lowered) -> list[TritonCall]:
    """Every Pallas-Triton custom call a lowered program makes, in program order."""
    def read(target: str, operation: ir.Operation) -> TritonCall | None:
        if target != TRITON:
            return None
        config = ir.DictAttr(operation.attributes["mhlo.backend_config"])
        return TritonCall(ir.StringAttr(config["name"]).value, ir.StringAttr(config["ir"]).value_bytes,
                          ir.IntegerAttr(config["num_warps"]).value,
                          ir.IntegerAttr(config["num_stages"]).value)

    return _read_calls(lowered, read)


def triton_kernels(lowered: jax.stages.Lowered) -> dict[str, int]:
    """The Pallas-Triton kernels a lowered program calls (`triton_calls`), by name, with the call
    sites of each."""
    return dict(sorted(Counter(call.name for call in triton_calls(lowered)).items()))


class Lowered(SystemExit):
    """Raised by `Trainer.fit` under `lower_for` once the run's first step has lowered for that GPU.

    Lowering was the whole run, so uncaught it ends the program with status 0
    and no traceback. `kernels` names the Pallas-Triton kernels the step
    calls (`triton_kernels`), and `lowered` is the step itself.
    """

    def __init__(self, generation: str, lowered: jax.stages.Lowered):
        super().__init__(0)
        self.generation = generation
        self.lowered = lowered
        self.kernels = triton_kernels(lowered)
