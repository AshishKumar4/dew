"""A training step lowered for a CUDA GPU, or compiled for a TPU topology, on a host without one.

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

A TPU topology ('v5e:4x4', as `jax.experimental.topologies` names one) is
compiled too, as MaxText's train_compile does: libtpu compiles for a TPU it
is not running on, so the run's mesh is built over the topology's devices
(`topology_devices`), the step lowers and compiles for them, and its
executable states the memory and work each device's step takes and saves to
a file a host of that topology loads (`Lowered.save`).

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

import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import jax
from jax.extend.mlir import ir

TRITON = "__gpu$xla.gpu.triton"
"""The custom call Pallas lowers a Triton kernel to."""


@dataclass(frozen=True)
class TritonCall:
    """One Pallas-Triton call of a lowered program, as Pallas configured it (`mhlo.backend_config`)."""

    name: str
    ir: bytes
    """The kernel's Triton IR, serialized as Pallas wrote it."""
    warps: int
    stages: int


def triton_calls(lowered: jax.stages.Lowered) -> list[TritonCall]:
    """Every Pallas-Triton custom call a lowered program makes, in program order."""
    calls: list[TritonCall] = []

    def visit(operation: ir.Operation) -> ir.WalkResult:
        attributes = operation.attributes
        if (operation.name == "stablehlo.custom_call"
                and ir.StringAttr(attributes["call_target_name"]).value == TRITON):
            config = ir.DictAttr(attributes["mhlo.backend_config"])
            calls.append(TritonCall(ir.StringAttr(config["name"]).value,
                                    ir.StringAttr(config["ir"]).value_bytes,
                                    ir.IntegerAttr(config["num_warps"]).value,
                                    ir.IntegerAttr(config["num_stages"]).value))
        return ir.WalkResult.ADVANCE

    module = lowered.compiler_ir(dialect="stablehlo")
    assert module is not None, "a lowered program always has its StableHLO module"
    with module.context:
        module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return calls


def triton_kernels(lowered: jax.stages.Lowered) -> dict[str, int]:
    """The Pallas-Triton kernels a lowered program calls (`triton_calls`), by name, with the call
    sites of each."""
    return dict(sorted(Counter(call.name for call in triton_calls(lowered)).items()))


def generation_of(target: str) -> str:
    """The hardware generation `lower_for`'s target names: a CUDA generation ('sm80') as it is, a TPU
    topology's ('v5e:4x4') its chip's ('v5e')."""
    return target.split(":", 1)[0]


def topology_devices(target: str | None) -> list | None:
    """The devices of the TPU topology `target` names ('v5e:4x4', `jax.experimental.topologies`), which
    libtpu compiles for on any host, or None for a target naming none: no target, or a CUDA
    generation's, which lowers on this process's own devices' mesh."""
    if target is None or ":" not in target:
        return None
    from jax.experimental import topologies

    return list(topologies.get_topology_desc(target, "tpu").devices)


class Lowered(SystemExit):
    """Raised by `Trainer.fit` under `lower_for` once the run's first step has lowered for its target.

    Lowering was the whole run, so uncaught it ends the program with status 0
    and no traceback. `kernels` names the Pallas-Triton kernels the step
    calls (`triton_kernels`), and `lowered` is the step itself. For a TPU
    topology the step is compiled for it too (`compiled`): `memory` states
    each device's bytes as XLA placed them, `cost` the work of one device's
    step, and `save` writes the executable a host of the topology loads.
    """

    def __init__(self, generation: str, lowered: jax.stages.Lowered, *, compiled: bool = False):
        super().__init__(0)
        self.generation = generation
        self.lowered = lowered
        self.kernels = triton_kernels(lowered)
        self.compiled = lowered.compile() if compiled else None

    @property
    def memory(self) -> dict[str, int]:
        """Each device's bytes of the compiled step: its arguments, outputs, temporaries, the aliased
        bytes donation reuses, and its code."""
        stats = self._compiled().memory_analysis()
        return {name: int(getattr(stats, f"{name}_size_in_bytes"))
                for name in ("argument", "output", "temp", "alias", "generated_code")}

    @property
    def cost(self) -> dict[str, float]:
        """One device's step as XLA's cost model counts it: its FLOPs and the bytes it accesses."""
        analysis = self._compiled().cost_analysis() or {}
        found = analysis[0] if isinstance(analysis, list) else analysis
        return {name: float(found[key]) for name, key in (("flops", "flops"), ("bytes", "bytes accessed"))
                if key in found}

    def save(self, path: str | Path) -> None:
        """Write the compiled step to `path`, local or remote: its serialized executable and its
        argument and result trees, which `jax.experimental.serialize_executable.deserialize_and_load`
        reads on a host of the topology it was compiled for, and beside it (`path.json`) the target it
        was compiled for and each device's memory and work."""
        from etils import epath
        from jax.experimental.serialize_executable import serialize

        saved = epath.Path(path)
        saved.write_bytes(pickle.dumps(serialize(self._compiled())))
        saved.with_name(saved.name + ".json").write_text(json.dumps(
            {"target": self.generation, "memory": self.memory, "cost": self.cost}, indent=1))

    def _compiled(self) -> jax.stages.Compiled:
        if self.compiled is None:
            raise ValueError(f"a step lowered for {self.generation} is not compiled: a CUDA GPU's compile "
                             "needs the GPU, a TPU topology's ('v5e:4x4') compiles here")
        return self.compiled
