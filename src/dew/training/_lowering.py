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
import math
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace

import jax
from jax.extend.mlir import ir

from dew.nn.kernels import generation
from dew.nn.kernels.generation import Route
from dew.nn.kernels.templates import report
from dew.telemetry.instrumentation import call_weights, custom_call_kind, sizes

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


def _read[T](lowered: jax.stages.Lowered, read: Callable[[ir.Operation], T | None]) -> list[tuple[T, float]]:
    """What `read` makes of each operation of a lowered program's StableHLO module, in program order, each
    once with the times it runs a call of the program, as `hlo_flops` weighs an executable's computations
    (`instrumentation.call_weights`): its function's runs from the entry (a nested jit's, a kernel
    `tuning.lowered_once` lowers once for every layer that calls it), times each loop's around it (`_walked`:
    a `stablehlo.while` body its static trip count, a scan of like layers its length); infinite where a loop
    around it states no static count. A function the entry cannot reach runs none and is left out."""
    module = lowered.compiler_ir(dialect="stablehlo")
    assert module is not None, "a lowered program always has its StableHLO module"
    found: list[tuple[T, float]] = []
    with module.context:
        functions = {ir.StringAttr(op.attributes["sym_name"]).value: op.operation
                     for op in module.body.operations if op.operation.name == "func.func"}
        assert "main" in functions, "a lowered program's entry is its module's main"
        callees: dict[str, dict[str, float]] = {name: {} for name in functions}
        for name, function in functions.items():
            for operation, times in _walked(function, 1.0):
                if operation.name == "func.call":
                    callee = ir.FlatSymbolRefAttr(operation.attributes["callee"]).value
                    callees[name][callee] = callees[name].get(callee, 0.0) + times
        runs = call_weights(callees, "main")
        for name, function in functions.items():
            if runs[name]:
                found.extend((made, times) for operation, times in _walked(function, runs[name])
                             if (made := read(operation)) is not None)
    return found


def _walked(operation: ir.Operation, times: float) -> Iterator[tuple[ir.Operation, float]]:
    """`operation`, which runs `times`, and every operation its regions hold, in program order, each with
    the times it runs: a `stablehlo.while`'s condition as often as the loop, its body its trips more
    (`_trips`), as `hlo_flops` weighs a while's condition and body; any other region (a branch, a
    reduction's body) as often as its operation, every branch counted, which bounds the one that runs."""
    yield operation, times
    for index, region in enumerate(operation.regions):
        within = times
        if operation.name == "stablehlo.while" and index == 1:
            trips = _trips(operation)
            within = 0.0 if not trips or not times else times * trips
        for block in region.blocks:
            for nested in block.operations:
                yield from _walked(nested.operation, within)


def _trips(loop: ir.Operation) -> float:
    """The trip count of a `stablehlo.while` that counts one integer it carries from a constant start, by a
    constant positive step, while it is below a constant bound: the loop every `scan` and `fori_loop` with
    static bounds lowers to, read as `instrumentation._counted_trips` reads the HLO loop. Infinite for any
    other loop, and for one whose last step overflows its counter's type, since a wrapped counter need never
    reach the bound."""
    condition, body = (region.blocks[0] for region in loop.regions)
    compare = _result_of(_terminator(condition).operands[0])
    if (compare is None or compare.name != "stablehlo.compare"
            or "LT" not in str(compare.attributes["comparison_direction"])):
        return math.inf
    counter, bound = (operand.maybe_downcast() for operand in compare.operands)
    if not isinstance(counter, ir.BlockArgument):
        return math.inf
    index, limit = counter.arg_number, _constant(bound)
    start = _constant(loop.operands[index].maybe_downcast())
    stepped = _result_of(_terminator(body).operands[index])
    step = None
    if stepped is not None and stepped.name == "stablehlo.add":
        sides = [operand.maybe_downcast() for operand in stepped.operands]
        if any(isinstance(side, ir.BlockArgument) and side.arg_number == index and side.owner == body
               for side in sides):
            step = next((value for value in map(_constant, sides) if value is not None), None)
    kind = ir.RankedTensorType(counter.type).element_type.maybe_downcast()
    if limit is None or start is None or step is None or step <= 0 or not isinstance(kind, ir.IntegerType):
        return math.inf
    trips = max(0, -(-(limit - start) // step))
    # A compare states its type's signedness; one that states none compares signed, StableHLO's default.
    stated = str(compare.attributes["compare_type"]) if "compare_type" in compare.attributes else ""
    signed = "UNSIGNED" not in stated
    if start + trips * step > (1 << (kind.width - signed)) - 1:
        return math.inf
    return float(trips)


def _terminator(block: ir.Block) -> ir.Operation:
    """The last operation of `block`, its terminator (a region's `stablehlo.return`)."""
    return list(block.operations)[-1].operation


def _result_of(value: ir.Value) -> ir.Operation | None:
    """The operation `value` is a result of; None where it is a block's argument."""
    value = value.maybe_downcast()
    return value.owner.operation if isinstance(value, ir.OpResult) else None


def _constant(value: ir.Value) -> int | None:
    """The integer a scalar `stablehlo.constant` holds where `value` is its result; None otherwise."""
    made = _result_of(value)
    if made is None or made.name != "stablehlo.constant":
        return None
    held = made.attributes["value"].maybe_downcast()
    if not isinstance(held, ir.DenseIntElementsAttr) or not held.is_splat:
        return None
    return ir.IntegerAttr(held.get_splat_value()).value


def _read_calls[T](lowered: jax.stages.Lowered,
                   read: Callable[[str, ir.Operation], T | None]) -> list[tuple[T, float]]:
    """What `read` makes of each custom call a lowered program makes, given its target and its
    operation, in program order, each once with the times it runs (`_read`): the calls it makes something
    of."""
    def call(operation: ir.Operation) -> T | None:
        if operation.name != "stablehlo.custom_call":
            return None
        return read(ir.StringAttr(operation.attributes["call_target_name"]).value, operation)

    return _read(lowered, call)


def operations(lowered: jax.stages.Lowered) -> list[str]:
    """The name of every operation a lowered program's StableHLO module holds, in program order, each once
    (a collective a `shard_map` body issues among them, `stablehlo.all_gather`)."""
    return [name for name, _ in _read(lowered, lambda operation: operation.name)]


def custom_calls(lowered: jax.stages.Lowered) -> list[CustomCall]:
    """Every custom call a lowered program makes, in program order, each once."""
    def ranked(values):
        return [ir.RankedTensorType(value.type) for value in values
                if isinstance(value.type, ir.RankedTensorType)]

    def shapes(values) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(kind.shape) for kind in ranked(values))

    return [call for call, _ in _read_calls(lowered, lambda target, operation: CustomCall(
        target, shapes(operation.operands), shapes(operation.results),
        tuple(str(kind.element_type) for kind in ranked(operation.operands))))]


def kernel_calls(lowered: jax.stages.Lowered) -> list[KernelCall]:
    """Every kernel call a lowered program makes, in program order, each lowering once (`kernel_launches`
    with the times each runs): each custom call but those that run no matmul
    (`instrumentation.custom_call_kind`), the launches one dispatch expands into, forward and backward,
    which only the lowered program shows (a dispatch's `generation.Route` holds its decision alone)."""
    return [call for call, _ in kernel_launches(lowered)]


def kernel_launches(lowered: jax.stages.Lowered) -> list[tuple[KernelCall, float]]:
    """Each of `kernel_calls`, once, with the times it runs a call of the program (`_read`): from every
    layer that calls a kernel lowered once, and every trip of a loop around it; infinite where a loop
    around it states no static count."""
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
    # Less the launch the kernel's name carries (`report.named`), so a kernel reads as itself in every call.
    kernel = stack[-2] if len(stack) > 1 and stack[-1] == "mpmd_map" else stack[-1]
    return report.LAUNCH_SUFFIX.sub("", kernel)


def _named(location: ir.Location) -> str | None:
    """The name a location carries, as JAX lowers an equation's (`mlir._source_info_to_location`): a name
    location's own (its name stack), a call site's callee's (a traceback without a name stack), or a fused
    location's first part that carries one (a rewrite that merged operations); None where it carries
    none."""
    if location.typeid == ir.NameLoc.static_typeid:
        return ir.NameLoc(location).name_str
    if location.typeid == ir.CallSiteLoc.static_typeid:
        return _named(ir.CallSiteLoc(location).callee)
    if location.typeid == ir.FusedLoc.static_typeid:
        # Its parts are locations, which the bindings' stubs type as objects.
        parts = (part for part in ir.FusedLoc(location).locations if isinstance(part, ir.Location))
        return next(filter(None, map(_named, parts)), None)
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
    calls: tuple[tuple[KernelCall, float], ...]
    """Each kernel call the lowered program makes, once, with the times it runs (`kernel_launches`)."""

    def record(self) -> dict:
        """Its JSON: each distinct route with its count, each kernel's launches, configs and stated FLOPs
        (None where any of its launches states none), and `unsized`, the launches whose work the FLOP
        count cannot size (`instrumentation.sizes`); a count a loop of no static trip count leaves
        unknown is None, as the FLOP count's is."""
        routes = Counter(self.routes)
        launches: dict[tuple[str, str], dict] = {}
        for call, runs in self.calls:
            entry = launches.setdefault((call.kind or "", call.name), {
                "kind": call.kind, "name": call.name, "calls": 0.0, "configs": [], "flops": 0.0})
            entry["calls"] += runs
            if call.config not in entry["configs"]:
                entry["configs"].append(call.config)
            stated, flops = entry["flops"], call.flops
            entry["flops"] = stated + flops * runs if stated is not None and flops is not None else None
        for entry in launches.values():
            entry["calls"] = int(entry["calls"]) if math.isfinite(entry["calls"]) else None
            if entry["flops"] is not None and not math.isfinite(entry["flops"]):
                entry["flops"] = None

        def order(counted) -> tuple:
            route, _ = counted
            return route.choice, route.site, route.kernel

        return {"routes": [{**asdict(route), "count": count}
                           for route, count in sorted(routes.items(), key=order)],
                "launches": [launches[key] for key in sorted(launches)],
                "unsized": _known(sum(runs for call, runs in self.calls
                                      if not sizes(call.kind, call.flops)))}

    def summary(self) -> str:
        """One line for the run log: each kernel's launches, and the dispatches off their preferred kernel."""
        launches = ", ".join(f"{entry['name']} x{'?' if entry['calls'] is None else entry['calls']}"
                             for entry in self.record()["launches"])
        off = sorted({f"{route.choice}: {route.kernel} not {route.preferred}" for route in self.routes
                      if route.kernel != route.preferred})
        return f"kernels: {launches or 'none'}" + (f"; off their preferred: {'; '.join(off)}" if off else "")


def _known(count: float) -> int | None:
    """A count of launches as a record holds it: None where a loop of no static trip count leaves it
    unknown."""
    return int(count) if math.isfinite(count) else None


def kernel_report(routes, lowered: jax.stages.Lowered) -> KernelReport:
    """The `KernelReport` of a step whose trace reported `routes` and lowered to `lowered`, its calls without
    their kernels' IR, which nothing that reads a report reads: a run keeps the report after it releases
    the program (`planning.Planned.released`, the data lane)."""
    return KernelReport(tuple(routes),
                        tuple((replace(call, ir=b""), runs) for call, runs in kernel_launches(lowered)))


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
    """Raised by `Trainer.fit` under `lower_for` once the run's first step has lowered for its target.

    Lowering was the whole run, so uncaught it ends the program with status 0
    and no traceback, the step's kernels logged (`--trainer.lower-for`'s dry run). `kernels` is the step's
    kernel report (`KernelReport`: its dispatches' routes and its lowered launches), and `lowered` is
    the step itself. A step for devices this host is not compiles ahead (`dew.training.ahead`).
    """

    def __init__(self, target: str, lowered: jax.stages.Lowered, routes=()):
        super().__init__(0)
        self.generation = target
        self.lowered = lowered
        self.kernels = kernel_report(routes, lowered)
        _log.info("the step lowered for %s; its %s", target, self.kernels.summary())
