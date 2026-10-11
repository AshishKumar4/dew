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
generation's platform, and takes the GPU's path here. Three still read
`jax.default_backend()`, so the dry run cannot see their GPU path:
attention's cuDNN (`cudnn_runs`) and FlashAttention (`flash_refusal`) calls,
CUDA kernels and not Pallas ones, whose trace asks the CUDA plugin for
versions and libraries a CPU host lacks, so attention traces its XLA path;
and the gated delta rule's explicit 'pallas' on a CPU (dew.nn.linear's
`chunk_states_kernel`), which places the kernel interpreted where a GPU
below sm80 would refuse it. `auto` never asks for that. Nor can it lower
tokamax's Triton ragged dot, the opt-in 'tokamax' grouped matmul's, which
tokamax registers only on a host with a GPU.
"""

from __future__ import annotations

import re
from collections import Counter

import jax
from jax.sharding import AbstractDevice, AbstractMesh, Mesh

DEVICE_KINDS = {
    'sm75': "Tesla T4",
    'sm80': "NVIDIA A100-SXM4-80GB",
    'sm86': "NVIDIA A10",
    'sm89': "NVIDIA L4",
    'sm90': "NVIDIA H100 80GB HBM3",
    'sm120': "NVIDIA RTX PRO 6000 Blackwell Server Edition",
}
"""A `device_kind` of each CUDA generation a step lowers for, as the device reports it: JAX's
Pallas lowering reads the compute capability it targets from the kind (`get_gpu_info`)."""

UNLOWERABLE = {
    ('tokamax_grouped_matmul', 'triton'): (
        "tokamax registers its Triton ragged dot on GPU hosts only, so no CPU lowers it, and the "
        "kernel lab's verify on the GPU is its only check"),
}
"""The Pallas choices a `KERNELS` row can name whose Triton lowering no CPU reaches, and why."""

_TRITON_CALL = re.compile(r'custom_call @__gpu\$xla\.gpu\.triton\(.*?\bname = "([^"]*)"')


def gpu_mesh(mesh: Mesh, generation: str) -> AbstractMesh:
    """`mesh`'s axes over devices of the CUDA `generation` ('sm80'), none of which this process needs."""
    if generation not in DEVICE_KINDS:
        raise ValueError(f"lower_for names {generation!r}; it lowers for one of {', '.join(DEVICE_KINDS)}")
    device = AbstractDevice(device_kind=DEVICE_KINDS[generation], num_cores=None, platform='cuda')
    return AbstractMesh(mesh.axis_sizes, mesh.axis_names, axis_types=mesh.axis_types, abstract_device=device)


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
