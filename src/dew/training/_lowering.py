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
generation's platform, and takes the GPU's path here. Two still read
`jax.default_backend()`, so the dry run cannot see their GPU path:
attention's cuDNN (`cudnn_runs`) and FlashAttention (`flash_refusal`) calls,
CUDA kernels and not Pallas ones, whose trace asks the CUDA plugin for
versions and libraries a CPU host lacks, so attention traces its XLA path.
"""

from __future__ import annotations

import re
from collections import Counter

import jax

_TRITON_CALL = re.compile(r'custom_call @__gpu\$xla\.gpu\.triton\(.*?\bname = "([^"]*)"')


def triton_kernels(lowered: jax.stages.Lowered) -> dict[str, int]:
    """The Pallas-Triton kernels a lowered program calls, by name, with the call sites of each."""
    return dict(sorted(Counter(_TRITON_CALL.findall(lowered.as_text(debug_info=False))).items()))


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
