"""The kernel calls a lowered program makes, read from its StableHLO: each custom call, each kernel launch it
expands into (Pallas-Triton's and Mosaic GPU's, forward and backward, with the config each states), and how
many times each runs a call of the program, every loop around it counted. The kernels layer's own reader of
its launches, which the kernel reports (`templates.report`), the compile gates and the training lowering
(`dew.training._lowering`) read alike."""

from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import jax
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import stablehlo

from dew.nn.kernels.templates import report
from dew.telemetry.instrumentation import MODEL_FLOPS, call_weights, counted_trips, custom_call_kind


@dataclass(frozen=True)
class KernelCall:
    """One kernel call of a lowered program: its `kind` (`instrumentation.custom_call_kind`: a Pallas
    kernel's compiler, 'triton', 'mosaic_gpu' or 'mosaic_tpu', a library's 'gemm', 'convolution' or
    'attention', or None for a call nothing sizes), its kernel's `name` (Triton's config names it, a
    Mosaic TPU call's `kernel_name`, a Mosaic GPU call's location, the name-stack entry its `mpmd_map`
    lowers under, its kernel's hash where its location names none; any other call's target), a Pallas
    kernel's IR as Pallas serialized it (Triton IR, or Mosaic's module bytecode), its `config` (Triton's
    warps and stages, a Mosaic GPU kernel's hash), the FLOPs its kernel states (its cost estimate), None
    where it states none, and the launch that traced a Mosaic GPU kernel, which its location's name carried
    (`report.named`), None where it carried none."""

    kind: str | None
    name: str
    ir: bytes
    config: str
    flops: float | None
    launch: str | None = None
    model_flops: float | None = None
    """The model's share of `flops` a Mosaic GPU kernel declares (`instrumentation.MODEL_FLOPS`, its launch's
    `model_flops`), None where it declares none."""
    operands: tuple[tuple[int, ...], ...] = ()
    """The shapes of the arrays the call reads, as its program reads them: a device's shard's where the
    call runs per shard (within a `shard_map`), the whole array's otherwise."""


@dataclass(frozen=True)
class CustomCall:
    """One custom call of a lowered program: its target, the shapes of the arrays it reads and writes, and
    the element types of those it reads (MLIR's spelling, `i32`, `bf16`)."""

    target: str
    operands: tuple[tuple[int, ...], ...]
    results: tuple[tuple[int, ...], ...]
    operand_types: tuple[str, ...] = ()


def _read[T](lowered: jax.stages.Lowered, read: Callable[[ir.Operation], T | None], *,
             unrun: bool = False) -> list[tuple[T, float]]:
    """What `read` makes of each operation of a lowered program's StableHLO module, each once with the times
    it runs a call of the program, as `hlo_flops` weighs an executable's computations
    (`instrumentation.call_weights`): its function's runs from the entry (a nested jit's, a kernel
    `tuning.lowered_once` lowers once for every layer that calls it), times each loop's around it (`_walked`:
    a `stablehlo.while` body its static trip count, a scan of like layers its length); infinite where a loop
    around it states no static count. In the order JAX lowered them: the entry's operations in turn, a
    function's where its first call is (each lowered once, however many calls reach it), so a kernel's
    compile reports printed as it lowered (Mosaic GPU's resource estimates) come in this order. A function
    no run reaches (behind a loop of no trips, or no call at all) runs none and is left out, or with
    `unrun` given at 0 times, as JAX lowers and compiles it all the same."""
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
        entered: set[str] = set()

        def lower(name: str) -> None:
            entered.add(name)
            if not runs.get(name, 0.0) and not unrun:
                return
            for operation, times in _walked(functions[name], runs.get(name, 0.0)):
                if (made := read(operation)) is not None:
                    found.append((made, times))
                if operation.name == "func.call":
                    callee = ir.FlatSymbolRefAttr(operation.attributes["callee"]).value
                    if callee in functions and callee not in entered:
                        lower(callee)

        lower("main")
        for name in functions:
            if name not in entered:
                lower(name)
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
    static bounds lowers to, by the rule an executable's loop is read by (`instrumentation.counted_trips`).
    Infinite for any other loop."""
    condition, body = (region.blocks[0] for region in loop.regions)
    compare = _result_of(_terminator(condition).operands[0])
    if compare is None or compare.name != "stablehlo.compare" or stablehlo.ComparisonDirectionAttr(
            compare.attributes["comparison_direction"]).value != "LT":
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
    if limit is None or start is None or step is None or not isinstance(kind, ir.IntegerType):
        return math.inf
    # A compare states its type's signedness; one that states none (or NOTYPE) compares as its operands'
    # type is, an unsigned integer's unsigned and a signless or signed one's signed.
    stated = "compare_type" in compare.attributes and stablehlo.ComparisonTypeAttr(
        compare.attributes["compare_type"]).value
    unsigned = stated == "UNSIGNED" or (stated in (False, "NOTYPE") and kind.is_unsigned)
    return counted_trips(start, limit, step, kind.width, signed=not unsigned)


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


def _read_calls[T](lowered: jax.stages.Lowered, read: Callable[[str, ir.Operation], T | None], *,
                   unrun: bool = False) -> list[tuple[T, float]]:
    """What `read` makes of each custom call a lowered program makes, given its target and its
    operation, in the order JAX lowered them, each once with the times it runs (`_read`, `unrun` its):
    the calls it makes something of."""
    def call(operation: ir.Operation) -> T | None:
        if operation.name != "stablehlo.custom_call":
            return None
        return read(ir.StringAttr(operation.attributes["call_target_name"]).value, operation)

    return _read(lowered, call, unrun=unrun)


def operations(lowered: jax.stages.Lowered) -> list[str]:
    """The name of every operation a lowered program's StableHLO module holds, as lowered (`_read`), each once
    (a collective a `shard_map` body issues among them, `stablehlo.all_gather`)."""
    return [name for name, _ in _read(lowered, lambda operation: operation.name)]


def _ranked(values) -> list:
    """The ranked tensor types of `values`, in order."""
    return [ir.RankedTensorType(value.type) for value in values
            if isinstance(value.type, ir.RankedTensorType)]


def _shapes(values) -> tuple[tuple[int, ...], ...]:
    """The shapes of `values`' ranked tensors, in order."""
    return tuple(tuple(kind.shape) for kind in _ranked(values))


def custom_calls(lowered: jax.stages.Lowered) -> list[CustomCall]:
    """Every custom call a lowered program makes, as lowered (`_read`), each once."""
    return [call for call, _ in _read_calls(lowered, lambda target, operation: CustomCall(
        target, _shapes(operation.operands), _shapes(operation.results),
        tuple(str(kind.element_type) for kind in _ranked(operation.operands))))]


def kernel_calls(lowered: jax.stages.Lowered, *, unrun: bool = False) -> list[KernelCall]:
    """Every kernel call a lowered program makes, as lowered (`_read`), each lowering once (`kernel_launches`
    with the times each runs): each custom call but those that run no matmul
    (`instrumentation.custom_call_kind`), the launches one dispatch expands into, forward and backward,
    which only the lowered program shows (a dispatch's `generation.Route` holds its decision alone); with
    `unrun`, those no run reaches too, which JAX compiles all the same (the compile gate's evidence)."""
    return [call for call, _ in kernel_launches(lowered, unrun=unrun)]


def kernel_launches(lowered: jax.stages.Lowered, *, unrun: bool = False) -> list[tuple[KernelCall, float]]:
    """Each of `kernel_calls`, once, with the times it runs a call of the program (`_read`): from every
    layer that calls a kernel lowered once, and every trip of a loop around it; infinite where a loop
    around it states no static count."""
    def read(target: str, operation: ir.Operation) -> KernelCall | None:
        kind = custom_call_kind(target)
        if kind == "no_matmul":
            return None
        called = _kernel_call(kind, target, operation)
        return dataclasses.replace(called, operands=_shapes(operation.operands))

    return _read_calls(lowered, read, unrun=unrun)


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
        named = _kernel_name(operation.location)
        # A call whose location names no kernel is named by its kernel's hash, apart from every other.
        return KernelCall(kind, named[0] if named else hashed, ir.StringAttr(config["module"]).value_bytes,
                          hashed, stated.get("flops"), named[1] if named else None, _model_flops(attributes))
    if kind == "mosaic_tpu":
        config = json.loads(ir.StringAttr(attributes["backend_config"]).value)["custom_call_config"]
        return KernelCall(kind, ir.StringAttr(attributes["kernel_name"]).value, config["body"].encode(), "",
                          config.get("cost_estimate", {}).get("flops"))
    return KernelCall(kind, target, b"", "", None)


def _model_flops(attributes) -> float | None:
    """The model's FLOPs a call's frontend attributes declare (`instrumentation.MODEL_FLOPS`), None where
    they declare none."""
    if "mhlo.frontend_attributes" not in attributes:
        return None
    frontend = ir.DictAttr(attributes["mhlo.frontend_attributes"])
    return float(ir.StringAttr(frontend[MODEL_FLOPS]).value) if MODEL_FLOPS in frontend else None


def _kernel_name(location: ir.Location) -> tuple[str, str | None] | None:
    """A Mosaic GPU call's kernel name and the launch it carries (`report.kernel_of`): its location's name
    stack (`_named`) ends `<kernel>/mpmd_map`; None where its location carries no name."""
    named = _named(location)
    if named is None:
        return None
    stack = named.split("/")
    # Less the launch the kernel's name carries (`report.named`), so a kernel reads as itself in every call.
    return report.kernel_of(stack[-2] if len(stack) > 1 and stack[-1] == "mpmd_map" else stack[-1])


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
