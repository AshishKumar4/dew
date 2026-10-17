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
is, so the dry run cannot see their GPU path: attention's cuDNN (`cudnn_runs`)
and FlashAttention (`flash_refusal`) calls, CUDA kernels and not Pallas ones,
whose trace asks the CUDA plugin for versions and libraries a CPU host lacks
and whose lowering asks for devices an abstract mesh lacks, even on a GPU
host, so attention traces its XLA path.
"""

from __future__ import annotations

from collections import Counter

import jax
from jax.extend.mlir import ir

TRITON = "__gpu$xla.gpu.triton"
"""The custom call Pallas lowers a Triton kernel to, its name in the call's backend config."""


def triton_kernels(lowered: jax.stages.Lowered) -> dict[str, int]:
    """The Pallas-Triton kernels a lowered program calls, by name, with the call sites of each: each
    Triton custom call's `name`, read from the attributes Pallas set (`mhlo.backend_config`)."""
    calls: Counter[str] = Counter()

    def visit(operation: ir.Operation) -> ir.WalkResult:
        attributes = operation.attributes
        if (operation.name == "stablehlo.custom_call"
                and ir.StringAttr(attributes["call_target_name"]).value == TRITON):
            calls[ir.StringAttr(ir.DictAttr(attributes["mhlo.backend_config"])["name"]).value] += 1
        return ir.WalkResult.ADVANCE

    module = lowered.compiler_ir(dialect="stablehlo")
    assert module is not None, "a lowered program always has its StableHLO module"
    with module.context:
        module.operation.walk(visit)
    return dict(sorted(calls.items()))


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
