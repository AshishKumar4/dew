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
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace

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
    lowers under, its kernel's hash where its location names none; any other call's target), a Pallas
    kernel's IR as Pallas serialized it (Triton IR, or Mosaic's module bytecode), its `config` (Triton's
    warps and stages, a Mosaic GPU kernel's hash), and the FLOPs its kernel states (its cost estimate), None
    where it states none."""

    kind: str | None
    name: str
    ir: bytes
    config: str
    flops: float | None


@dataclass(frozen=True)
class CustomCall:
    """One custom call of a lowered program: its target, the shapes of the arrays it reads and writes, and
    the element types of those it reads (MLIR's spelling, `i32`, `bf16`)."""

    target: str
    operands: tuple[tuple[int, ...], ...]
    results: tuple[tuple[int, ...], ...]
    operand_types: tuple[str, ...] = ()


def _read[T](lowered: jax.stages.Lowered, read: Callable[[ir.Operation], T | None]) -> list[T]:
    """What `read` makes of each operation of a lowered program's StableHLO module, in program order: the
    operations it makes something of."""
    found: list[T] = []

    def visit(operation: ir.Operation) -> ir.WalkResult:
        made = read(operation)
        if made is not None:
            found.append(made)
        return ir.WalkResult.ADVANCE

    module = lowered.compiler_ir(dialect="stablehlo")
    assert module is not None, "a lowered program always has its StableHLO module"
    with module.context:
        module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return found


def _read_calls[T](lowered: jax.stages.Lowered, read: Callable[[str, ir.Operation], T | None]) -> list[T]:
    """What `read` makes of each custom call a lowered program makes, given its target and its
    operation, in program order: the calls it makes something of."""
    def call(operation: ir.Operation) -> T | None:
        if operation.name != "stablehlo.custom_call":
            return None
        return read(ir.StringAttr(operation.attributes["call_target_name"]).value, operation)

    return _read(lowered, call)


def operations(lowered: jax.stages.Lowered) -> list[str]:
    """The name of every operation a lowered program's StableHLO module holds, in program order (a
    collective a `shard_map` body issues among them, `stablehlo.all_gather`)."""
    return _read(lowered, lambda operation: operation.name)


def custom_calls(lowered: jax.stages.Lowered) -> list[CustomCall]:
    """Every custom call a lowered program makes, in program order."""
    def ranked(values):
        return [ir.RankedTensorType(value.type) for value in values
                if isinstance(value.type, ir.RankedTensorType)]

    def shapes(values) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(kind.shape) for kind in ranked(values))

    return _read_calls(lowered, lambda target, operation: CustomCall(
        target, shapes(operation.operands), shapes(operation.results),
        tuple(str(kind.element_type) for kind in ranked(operation.operands))))


def kernel_calls(lowered: jax.stages.Lowered) -> list[KernelCall]:
    """Every kernel call a lowered program makes, in program order: each custom call but those that run
    no matmul (`instrumentation.custom_call_kind`), the launches one dispatch expands into, forward and
    backward, which only the lowered program shows (a dispatch's `generation.Route` holds its decision
    alone)."""
    def read(target: str, operation: ir.Operation) -> KernelCall | None:
        kind = custom_call_kind(target)
        return None if kind == "no_matmul" else _kernel_call(kind, target, operation)

    return _read_calls(lowered, read)


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
        hashed = ir.StringAttr(config["kernel_hash"]).value_bytes.hex()[:16]
        # A call whose location names no kernel is named by its kernel's hash, apart from every other.
        return KernelCall(kind, _kernel_name(operation.location) or hashed,
                          ir.StringAttr(config["module"]).value_bytes, hashed, stated.get("flops"))
    if kind == "mosaic_tpu":
        config = json.loads(ir.StringAttr(attributes["backend_config"]).value)["custom_call_config"]
        return KernelCall(kind, ir.StringAttr(attributes["kernel_name"]).value, config["body"].encode(), "",
                          config.get("cost_estimate", {}).get("flops"))
    return KernelCall(kind, target, b"", "", None)


def _kernel_name(location: ir.Location) -> str | None:
    """A Mosaic GPU call's kernel name: its location's name stack (`_named`) ends `<kernel>/mpmd_map`; None
    where its location carries no name."""
    named = _named(location)
    if named is None:
        return None
    stack = named.split("/")
    return stack[-2] if len(stack) > 1 and stack[-1] == "mpmd_map" else stack[-1]


def _named(location: ir.Location) -> str | None:
    """The name a location carries, as JAX lowers an equation's (`mlir._source_info_to_location`): a name
    location's own (its name stack), or a call site's callee's (a traceback without a name stack); None
    where it carries none."""
    if location.typeid == ir.NameLoc.static_typeid:
        return ir.NameLoc(location).name_str
    if location.typeid == ir.CallSiteLoc.static_typeid:
        return _named(ir.CallSiteLoc(location).callee)
    return None


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
    """The `KernelReport` of a step whose trace reported `routes` and lowered to `lowered`, its calls without
    their kernels' IR, which nothing that reads a report reads: a run keeps the report after it releases
    the program (`planning.Planned.released`, TightLynx)."""
    return KernelReport(tuple(routes), tuple(replace(call, ir=b"") for call in kernel_calls(lowered)))


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
