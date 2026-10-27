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

import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import jax
from jax.extend.mlir import ir

from dew.nn.kernels import generation
from dew.nn.kernels.generation import Route
from dew.telemetry.instrumentation import custom_call_kind, sizes

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class KernelCall:
    """One kernel call of a lowered program: its `kind` (`instrumentation.custom_call_kind`: a Pallas
    kernel's compiler, 'triton', 'mosaic_gpu' or 'mosaic_tpu', a library's 'gemm', 'convolution' or
    'attention', or None for a call nothing sizes), its kernel's `name` (Triton's config names it, a
    Mosaic TPU call's `kernel_name`, a Mosaic GPU call's location, the name-stack entry its `mpmd_map`
    lowers under; any other call's target), a Pallas kernel's IR as Pallas serialized it (Triton IR,
    or Mosaic's module bytecode), its `config` (Triton's warps and stages, a Mosaic GPU kernel's hash),
    and the FLOPs its kernel states (its cost estimate), None where it states none."""

    kind: str | None
    name: str
    ir: bytes
    config: str
    flops: float | None


def kernel_calls(lowered: jax.stages.Lowered) -> list[KernelCall]:
    """Every kernel call a lowered program makes, in program order: each custom call but those that run
    no matmul (`instrumentation.custom_call_kind`), the launches one dispatch expands into, forward and
    backward, which only the lowered program shows (a dispatch's `generation.Route` holds its decision
    alone)."""
    calls: list[KernelCall] = []

    def visit(operation: ir.Operation) -> ir.WalkResult:
        if operation.name == "stablehlo.custom_call":
            target = ir.StringAttr(operation.attributes["call_target_name"]).value
            if (kind := custom_call_kind(target)) != "no_matmul":
                calls.append(_kernel_call(kind, target, operation))
        return ir.WalkResult.ADVANCE

    module = lowered.compiler_ir(dialect="stablehlo")
    assert module is not None, "a lowered program always has its StableHLO module"
    with module.context:
        module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return calls


def _kernel_call(kind: str | None, target: str, operation: ir.Operation) -> KernelCall:
    """The `KernelCall` of a custom call `operation` of `kind` to `target`."""
    attributes = operation.attributes
    if kind == "triton":
        config = ir.DictAttr(attributes["mhlo.backend_config"])
        return KernelCall(kind, ir.StringAttr(config["name"]).value, ir.StringAttr(config["ir"]).value_bytes,
                          f"warps={ir.IntegerAttr(config['num_warps']).value} "
                          f"stages={ir.IntegerAttr(config['num_stages']).value}", None)
    if kind == "mosaic_gpu":
        config = ir.DictAttr(attributes["mhlo.backend_config"])
        stated = (json.loads(ir.StringAttr(config["cost_estimate_json"]).value)
                  if "cost_estimate_json" in config else {})
        return KernelCall(kind, _kernel_name(operation.location), ir.StringAttr(config["module"]).value_bytes,
                          ir.StringAttr(config["kernel_hash"]).value_bytes.hex()[:16], stated.get("flops"))
    if kind == "mosaic_tpu":
        config = json.loads(ir.StringAttr(attributes["backend_config"]).value)["custom_call_config"]
        return KernelCall(kind, ir.StringAttr(attributes["kernel_name"]).value, config["body"].encode(), "",
                          config.get("cost_estimate", {}).get("flops"))
    return KernelCall(kind, target, b"", "", None)


def _kernel_name(location: ir.Location) -> str:
    """A Mosaic GPU call's kernel name: its location's name stack ends `<kernel>/mpmd_map`."""
    found = re.match(r'loc\("([^"]+)"', str(location))
    stack = found.group(1).split("/") if found else ["?"]
    return stack[-2] if len(stack) > 1 and stack[-1] == "mpmd_map" else stack[-1]


def triton_calls(lowered: jax.stages.Lowered) -> list[KernelCall]:
    """The Pallas-Triton calls of `kernel_calls`."""
    return [call for call in kernel_calls(lowered) if call.kind == "triton"]


def triton_kernels(lowered: jax.stages.Lowered) -> dict[str, int]:
    """The Pallas-Triton kernels a lowered program calls (`triton_calls`), by name, with the call sites
    of each."""
    return dict(sorted(Counter(call.name for call in triton_calls(lowered)).items()))


@dataclass(frozen=True)
class KernelReport:
    """Which kernels a step runs: each dispatch's decision over the step's trace (`routes`: its choice,
    kernel, the kernel it preferred and why that refused, its site; `generation.routes`) beside the
    launches its lowered program makes (`calls`, `kernel_calls`: how many of each kernel, its config and
    the FLOPs it states). The run log, the run record (`StepCompiled.kernels`, through `Planned`'s) and
    the lowering tool (`Lowered.kernels`, tools/kernel_picks.py) read this one report, and the FLOP count
    sizes each call as its `unsized` does (`instrumentation.sizes`)."""

    routes: tuple[Route, ...]
    calls: tuple[KernelCall, ...]

    def record(self) -> dict:
        """Its JSON: each distinct route with its count, each kernel's launches, configs and stated FLOPs
        (None where any of its launches states none), and `unsized`, the launches whose work the FLOP
        count cannot size (`instrumentation.sizes`)."""
        routes = Counter((route.choice, route.kernel, route.preferred,
                          None if route.refusal is None else route.refusal.detail, route.site)
                         for route in self.routes)
        launches: dict[tuple[str, str], dict] = {}
        for call in self.calls:
            entry = launches.setdefault((call.kind or "", call.name), {
                "kind": call.kind, "name": call.name, "calls": 0, "configs": [], "flops": 0.0})
            entry["calls"] += 1
            if call.config not in entry["configs"]:
                entry["configs"].append(call.config)
            stated = entry["flops"] is not None and call.flops is not None
            entry["flops"] = entry["flops"] + call.flops if stated else None

        def order(counted) -> tuple:
            (choice, kernel, _, _, site), _ = counted
            return choice, site, kernel

        return {"routes": [{"choice": choice, "kernel": kernel, "preferred": preferred, "refusal": refusal,
                            "site": site, "count": count}
                           for (choice, kernel, preferred, refusal, site), count in sorted(routes.items(),
                                                                                         key=order)],
                "launches": [launches[key] for key in sorted(launches)],
                "unsized": sum(not sizes(call.kind, call.flops) for call in self.calls)}

    def summary(self) -> str:
        """One line for the run log: each kernel's launches, and the dispatches off their preferred kernel."""
        launches = ", ".join(f"{entry['name']} x{entry['calls']}" for entry in self.record()["launches"])
        off = sorted({f"{route.choice}: {route.kernel} not {route.preferred}" for route in self.routes
                      if route.kernel != route.preferred})
        return f"kernels: {launches or 'none'}" + (f"; off their preferred: {'; '.join(off)}" if off else "")


def kernel_report(routes, lowered: jax.stages.Lowered) -> KernelReport:
    """The `KernelReport` of a step whose trace reported `routes` and lowered to `lowered`."""
    return KernelReport(tuple(routes), tuple(kernel_calls(lowered)))


def routed[*Arguments](lower: Callable[[*Arguments], jax.stages.Lowered],
                        *arguments: *Arguments) -> tuple[jax.stages.Lowered, list[Route]]:
    """`lower(*arguments)`'s lowered program and the routes its dispatches reported as it traced
    (`generation.routes`)."""
    with generation.routes() as found:
        lowered = lower(*arguments)
    return lowered, found


class Routed:
    """A step's lowering, `lower()`, that keeps the routes and program of its last call (`routed`): a
    tuning run lowers again where a config moved (`tuning.tune`), and the last is the one compiled."""

    def __init__(self, lower: Callable[[], jax.stages.Lowered]):
        self._lower = lower
        self._last: tuple[jax.stages.Lowered, list[Route]] | None = None

    def __call__(self) -> jax.stages.Lowered:
        self._last = routed(self._lower)
        return self._last[0]

    def report(self) -> KernelReport:
        """The last lowering's `KernelReport`."""
        assert self._last is not None, "a report reads a lowering the step made"
        lowered, found = self._last
        return kernel_report(found, lowered)


class Lowered(SystemExit):
    """Raised by `Trainer.fit` under `lower_for` once the run's first step has lowered for that GPU.

    Lowering was the whole run, so uncaught it ends the program with status 0
    and no traceback, the step's kernels logged (`--trainer.lower-for`'s dry run). `kernels` is the step's
    kernel report (`KernelReport`: its dispatches' routes and its lowered launches), and `lowered` is
    the step itself.
    """

    def __init__(self, target: str, lowered: jax.stages.Lowered, routes=()):
        super().__init__(0)
        self.generation = target
        self.lowered = lowered
        self.kernels = kernel_report(routes, lowered)
        _log.info("the step lowered for %s; its %s", target, self.kernels.summary())
