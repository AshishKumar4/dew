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
a file a host of that topology loads (`Lowered.save`). `fit` starts as it
would on the topology, from shapes: the state's, a resume's rung and window
with it, and the first batch its stream reads at the run's position, marked
as each step's is, and compiles its step there, before anything is placed.

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
import os
import pickle
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import jax
from jax.extend.mlir import ir

from dew.cache import uncached
from dew.nn.kernels import generation
from dew.nn.kernels.generation import Route
from dew.nn.kernels.templates import report
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
    calls: tuple[KernelCall, ...]

    def record(self) -> dict:
        """Its JSON: each distinct route with its count, each kernel's launches, configs and stated FLOPs
        (None where any of its launches states none), and `unsized`, the launches whose work the FLOP
        count cannot size (`instrumentation.sizes`)."""
        routes = Counter(self.routes)
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
            route, _ = counted
            return route.choice, route.site, route.kernel

        return {"routes": [{**asdict(route), "count": count}
                           for route, count in sorted(routes.items(), key=order)],
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


def compiled_name(target: str) -> str:
    """The file a run's executable compiled for the TPU topology `target` is saved as beside its
    checkpoints ('compiled-v5e-4x4'): its name with no colon, which some filesystems and tools read as
    a drive or a host."""
    return f"compiled-{target.replace(':', '-')}"


def compile_opened(trainer, abstract, batch, current: int, *, resumed: bool) -> None:
    """Compile `trainer`'s step for its TPU topology as `fit` opens at step `current`, from the whole
    state's shapes (`abstract`, a resume's window with it) and the first batch its stream read at the
    run's position, marked as every step's is and placed as shapes (`placed_abstract`): its device
    fields, which the objective's inputs are checked against first, at the rung a `resumed`
    checkpoint records, as the step it continues compiles there. `compile` raises `Lowered`."""
    from dew.distributed.batches import split_fields
    from dew.training._fit import check_inputs

    if resumed and trainer.checkpoints is not None:
        trainer._resume_rung(trainer.checkpoints.rung(current))
    device, _ = split_fields(batch)
    check_inputs(trainer, device)
    trainer.compile(abstract, device)


def uncompiled(trainer, current: int, complete: bool) -> ValueError:
    """Why `trainer`'s fit at step `current` compiled no step for its TPU topology, whose devices it
    places nothing on: a rollout's batches are its policy's draws, which no device here draws, and a
    `complete` run has no step left to take."""
    if trainer.rollout is not None:
        return ValueError(
            f"a rollout run's batches are its policy's draws, which the TPU topology {trainer.lower_for} "
            f"this host does not run cannot draw: compile its step with Trainer.compile(state, batch) "
            f"over a batch of its shapes")
    assert complete, "fit compiles a TPU topology's step as its stream opens"
    return ValueError(f"the run is complete at step {current}, its last checkpoint written, so it has no "
                      f"step to compile for the TPU topology {trainer.lower_for}")


class Lowered(SystemExit):
    """Raised by `Trainer.fit` under `lower_for` once the run's first step has lowered for its target.

    Lowering was the whole run, so uncaught it ends the program with status 0
    and no traceback, the step's kernels logged (`--trainer.lower-for`'s dry run). `kernels` is the step's
    kernel report (`KernelReport`: its dispatches' routes and its lowered launches), and `lowered` is
    the step itself. For a TPU topology the step is compiled for it too (`compiled`): `memory` states each
    device's bytes as XLA placed them, `cost` the work of one device's step, and `save` writes the
    executable a host of the topology loads.
    """

    def __init__(self, target: str, lowered: jax.stages.Lowered, routes=(), *, compiled: bool = False):
        super().__init__(0)
        self.generation = target
        self.lowered = lowered
        self.kernels = kernel_report(routes, lowered)
        _log.info("the step lowered for %s; its %s", target, self.kernels.summary())
        self.compiled = None
        if compiled:
            # Compiled for devices this host is not, its executable loads onto no client here: the
            # persistent cache neither reads an entry for it nor writes one (`dew.cache.uncached`).
            with uncached():
                self.compiled = lowered.compile()

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

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the compiled step to `path`, local or remote, and beside it (`path.json`) the target it
        was compiled for and each device's memory and work.

        `path` holds a pickle of `jax.experimental.serialize_executable.serialize`'s triple, the
        serialized executable and its argument and result trees. A host of the topology it was
        compiled for, under the same JAX and libtpu, loads it as

            serialized, inputs, outputs = pickle.loads(epath.Path(path).read_bytes())
            step = deserialize_and_load(serialized, inputs, outputs, backend="tpu")

        Unpickling and deserializing can run arbitrary code, so load only a file this run or another
        trusted one wrote."""
        from etils import epath
        from jax.experimental.serialize_executable import serialize

        saved = epath.Path(path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(pickle.dumps(serialize(self._compiled())))
        saved.with_name(saved.name + ".json").write_text(json.dumps(
            {"target": self.generation, "memory": self.memory, "cost": self.cost}, indent=1))

    def _compiled(self) -> jax.stages.Compiled:
        if self.compiled is None:
            raise ValueError(f"a step lowered for {self.generation} is not compiled: a CUDA GPU's compile "
                             "needs the GPU, a TPU topology's ('v5e:4x4') compiles here")
        return self.compiled
