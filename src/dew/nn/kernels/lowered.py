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
import jax.experimental.mosaic.gpu as mgpu
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import stablehlo

from dew.nn.kernels.templates import assembled
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
    where it states none, and the digest its kernel's name carries, a Mosaic GPU call's by its location: its
    kernel's identity within the launch that traced it (`report.named`), None where it carried none."""

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
    caller: str | None = None
    """The launch that traced a Mosaic GPU kernel, as its call's caller states it (`report.launch_metadata`),
    beside its name; None where its call states none."""

    @property
    def launched(self) -> str | None:
        """The digest of the launch that traced the call's kernel: its caller's, else its name's, which is its
        launch's where the launch states no kernel's part (`report.named`; a Triton kernel's name whole)."""
        return self.caller or self.launch or assembled.kernel_of(self.name)[1]


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
        # Each function's stated FLOPs, computed once, which its conditionals' branches are ranked by, and
        # each function's conditionals' largest branches in program order (`_Branches`).
        branches = _Branches(functions)
        callees: dict[str, dict[str, float]] = {name: {} for name in functions}
        for name, function in functions.items():
            for operation, times in _walked(function, 1.0, branches.of(name)):
                if operation.name == "func.call":
                    callee = ir.FlatSymbolRefAttr(operation.attributes["callee"]).value
                    callees[name][callee] = callees[name].get(callee, 0.0) + times
        runs = call_weights(callees, "main")
        entered: set[str] = set()

        def lower(name: str) -> None:
            entered.add(name)
            if not runs.get(name, 0.0) and not unrun:
                return
            for operation, times in _walked(functions[name], runs.get(name, 0.0), branches.of(name)):
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


def _walked(operation: ir.Operation, times: float, largest: Iterator[int]
            ) -> Iterator[tuple[ir.Operation, float]]:
    """`operation`, which runs `times`, and every operation its regions hold, in program order, each with
    the times it runs: a `stablehlo.while`'s condition as often as the loop, its body its trips more
    (`_trips`), as `hlo_flops` weighs a while's condition and body; of a conditional's branches (a
    `stablehlo.case`'s or `stablehlo.if`'s regions) the one `largest` gives next (`_Branches.of`, its
    conditionals' in this order: the branch whose kernels state the most FLOPs) as often as the
    conditional and every other none, as `hlo_flops` counts a conditional's largest branch: it runs one;
    any other region (a reduction's body) as often as its operation."""
    yield operation, times
    chosen = next(largest) if _branched(operation) else None
    for index, region in enumerate(operation.regions):
        within = times
        if operation.name == "stablehlo.while" and index == 1:
            trips = _trips(operation)
            within = 0.0 if not trips or not times else times * trips
        if chosen is not None and index != chosen:
            within = 0.0
        for block in region.blocks:
            for nested in block.operations:
                yield from _walked(nested.operation, within, largest)


def _branched(operation: ir.Operation) -> bool:
    """Whether `operation` is a conditional of more than one branch, which runs one (`_walked`)."""
    return operation.name in ("stablehlo.case", "stablehlo.if") and len(operation.regions) > 1


class _Branches:
    """A module's `functions`' stated FLOPs (`KernelCall.flops`, a library call's none) a run, each
    function's computed once with every function it calls, and each function's conditionals' largest
    branches, in the program order `_walked` meets them: a loop's body counted its trips, a conditional
    its branch of the most stated FLOPs, a call its callee's own."""

    def __init__(self, functions: dict[str, ir.Operation]):
        self.functions = functions
        self.flops: dict[str, float] = {}
        self.chosen: dict[str, list[int]] = {}

    def of(self, name: str) -> Iterator[int]:
        """Function `name`'s conditionals' largest branches, in program order."""
        self._function(name)
        return iter(self.chosen[name])

    def _function(self, name: str) -> float:
        """Function `name`'s stated FLOPs a run, computed once (a recursive call's 0 while it is)."""
        if name not in self.flops:
            self.flops[name] = 0.0
            chosen: list[int] = []
            self.flops[name] = self._stated(self.functions[name], chosen)
            self.chosen[name] = chosen
        return self.flops[name]

    def _stated(self, operation: ir.Operation, chosen: list[int]) -> float:
        """`operation`'s stated FLOPs a run, its conditionals' largest branches appended to `chosen` in
        program order, a conditional's before those its branches hold."""
        total = 0.0
        if operation.name == "stablehlo.custom_call":
            target = ir.StringAttr(operation.attributes["call_target_name"]).value
            kind = custom_call_kind(target)
            if kind != "no_matmul":
                total = _kernel_call(kind, target, operation).flops or 0.0
        elif operation.name == "func.call":
            callee = ir.FlatSymbolRefAttr(operation.attributes["callee"]).value
            if callee in self.functions:
                total = self._function(callee)
        slot = None
        if _branched(operation):
            slot = len(chosen)
            chosen.append(0)
        regions = [sum(self._stated(nested.operation, chosen) for block in region.blocks
                       for nested in block.operations) for region in operation.regions]
        if slot is not None:
            chosen[slot] = regions.index(max(regions))
            return total + max(regions)
        if operation.name == "stablehlo.while":
            trips = _trips(operation)
            body = 0.0 if not trips or not regions[1] else regions[1] * trips
            return total + regions[0] + body
        return total + sum(regions)


def _trips(loop: ir.Operation) -> float:
    """The trip count of a `stablehlo.while` that counts one integer it carries from a constant start, by a
    constant positive step, while it is below a constant bound: the loop every `scan` and `fori_loop` with
    static bounds lowers to, by the rule an executable's loop is read by (`instrumentation.counted_trips`);
    where that bound is one side of an `and`, at most its trips, a bound as a branch's every side counted is
    (a loop of at most one trip, `attention_algebra.once`). Infinite for any other loop."""
    condition, body = (region.blocks[0] for region in loop.regions)
    root = _result_of(_terminator(condition).operands[0])
    sides = ([_result_of(side) for side in root.operands] if root is not None and root.name == "stablehlo.and"
             else [root])
    return min((_compared_trips(loop, body, compare) for compare in sides), default=math.inf)


def _compared_trips(loop: ir.Operation, body: ir.Block, compare: ir.Operation | None) -> float:
    """`_trips` of `loop` by `compare`, its condition's counted bound, or infinite where it is none."""
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


def read_operations[T](lowered: jax.stages.Lowered, read: Callable[[ir.Operation], T | None]) -> list[T]:
    """What `read` makes of each operation of a lowered program's StableHLO module, every region's (a
    reduction's body among them, a function no run reaches too), each once, in the order JAX lowered them
    (`_read`): the operations it makes something of."""
    return [made for made, _ in _read(lowered, read, unrun=True)]


def frontend_attributes(operation: ir.Operation) -> dict[str, str]:
    """An operation's frontend attributes (`mhlo.frontend_attributes`), each its string."""
    return _frontend(operation.attributes)


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
    return _read_calls(lowered, kernel_call, unrun=unrun)


def kernel_call(target: str, operation: ir.Operation) -> KernelCall | None:
    """The `KernelCall` of a custom call `operation` to `target`, None for one that runs no matmul
    (`instrumentation.custom_call_kind`): `kernel_calls`' each, and each a compile compiles alone
    (`report.compiled_alone`)."""
    kind = custom_call_kind(target)
    if kind == "no_matmul":
        return None
    return dataclasses.replace(_kernel_call(kind, target, operation), operands=_shapes(operation.operands))


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
        caller = ir.StringAttr(config["metadata"]).value if "metadata" in config else None
        # A call whose location names no kernel is named by its kernel's hash, apart from every other.
        return KernelCall(kind, named[0] if named else hashed, ir.StringAttr(config["module"]).value_bytes,
                          hashed, stated.get("flops"), named[1] if named else None, _model_flops(attributes),
                          caller=caller)
    if kind == "mosaic_tpu":
        config = json.loads(ir.StringAttr(attributes["backend_config"]).value)["custom_call_config"]
        return KernelCall(kind, ir.StringAttr(attributes["kernel_name"]).value, config["body"].encode(), "",
                          config.get("cost_estimate", {}).get("flops"))
    return KernelCall(kind, target, b"", "", None)


def _frontend(attributes) -> dict[str, str]:
    """A call's frontend attributes, each its string."""
    if "mhlo.frontend_attributes" not in attributes:
        return {}
    return {named.name: ir.StringAttr(named.attr).value
            for named in ir.DictAttr(attributes["mhlo.frontend_attributes"])}


def _model_flops(attributes) -> float | None:
    """The model's FLOPs a call's frontend attributes declare (`instrumentation.MODEL_FLOPS`), None where
    they declare none."""
    declared = _frontend(attributes).get(MODEL_FLOPS)
    return None if declared is None else float(declared)


def _kernel_name(location: ir.Location) -> tuple[str, str | None] | None:
    """A Mosaic GPU call's kernel name and the launch it carries (`assembled.kernel_of`): its location's name
    stack (`_named`) ends `<kernel>/mpmd_map`; None where its location carries no name."""
    named = _named(location)
    if named is None:
        return None
    stack = named.split("/")
    # Less the launch the kernel's name carries (`report.named`), so a kernel reads as itself in every call.
    return assembled.kernel_of(stack[-2] if len(stack) > 1 and stack[-1] == "mpmd_map" else stack[-1])[:2]


def _named(location: ir.Location) -> str | None:
    """The name a location carries, as JAX lowers an equation's (`mlir._source_info_to_location`): a name
    location's own (its name stack), its child's where it is an equation's primitive label (`<primitive>:`,
    which JAX wraps the name stack in: a short conv's kernels under its marked call read 'marked_call:',
    jsf-convloc), a call site's callee's (a traceback without a name stack), or a fused location's first part
    that carries one (a rewrite that merged operations); None where it carries none."""
    if location.typeid == ir.NameLoc.static_typeid:
        named = ir.NameLoc(location)
        if named.name_str.endswith(":") and named.child_loc.typeid == ir.NameLoc.static_typeid:
            return _named(named.child_loc)
        return named.name_str
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


BARRIER_WAIT = "nvvm.mbarrier.try_wait.parity"
"""A thread's wait at a barrier's phase in a Mosaic GPU kernel's module (`BarrierRef.wait_parity`)."""

TENSOR_CORE_COMMIT = "llvm.nvvm.tcgen05.commit"
"""The LLVM intrinsics a tensor-core commit arrives at a barrier by (`llvm.call_intrinsic`'s `intrin`):
`.cg1` and `.cg2`, multicast or not."""

SYNCS = frozenset({"nvvm.barrier", "gpu.barrier"})
"""A sync of a warpgroup's threads at least: its own barrier (`mgpu_utils.warpgroup_barrier`, typed from the
fork's 0.12.0.1.post125), or the block's."""


def unsynced_commits(module: bytes) -> list[str]:
    """Each tensor-core commit (`TENSOR_CORE_COMMIT`) in a Mosaic GPU kernel's `module` (`KernelCall.ir`)
    that arrives at a barrier its thread waited at (`BARRIER_WAIT`) with no warpgroup sync between
    (`SYNCS`): an elected thread commits after its own wait while its warpgroup's other warps may still wait
    at the phase its arrival completes past (row 5671: a B200's tcgen05 chunk issuer). Each warp's path is its
    own (`_Walk`): a branch on the thread's index takes the warp's case, any other branch every case, the
    union after; a loop to its fixed point; a call through its callee. A barrier is its array in shared memory
    (`_barrier`), a wait at one slot and a commit at another counted alike. Each finding names its function,
    warp and barrier."""
    kernel = mgpu.deserialized(module)
    found: dict[str, None] = {}
    with kernel.context:
        functions = {ir.StringAttr(op.attributes["sym_name"]).value: op.operation
                     for op in kernel.body.operations if op.operation.name == "func.func"}
        if not any(map(_commits, functions.values())):
            return []
        for name, function in functions.items():
            for launch in _launches(function):
                view = launch.opview
                sizes = (view.blockSizeX, view.blockSizeY, view.blockSizeZ)
                block = tuple(_value(size, None) for size in sizes)
                if None in block:
                    raise ValueError(f"{name}'s block states no constant size")
                for first in range(0, math.prod(block), WARP_THREADS):
                    where = f"{name}, warp {first // WARP_THREADS}"
                    walk = _Walk(functions, _Thread(first, block), found, where)
                    walk.region(launch.regions[0], frozenset())
    return list(found)


WARP_THREADS = 32
"""The threads of a warp, whose path through a kernel `unsynced_commits` follows (its first lane's)."""


@dataclass(frozen=True)
class _Thread:
    """A thread of a launch: its index (`index`, along x, the block's threads numbered as Mosaic's warps and
    warpgroups are) and the block's sizes (`block`, x, y, z)."""

    index: int
    block: tuple[int, ...]


@dataclass
class _Walk:
    """One warp's walk of a kernel for `unsynced_commits`, from its first lane, `thread`, each finding in
    `found` under `where`."""

    functions: dict[str, ir.Operation]
    thread: _Thread
    found: dict[str, None]
    where: str

    def region(self, body: ir.Region, waited: frozenset) -> frozenset:
        for block in body.blocks:
            for inner in block.operations:
                waited = self.operation(inner.operation, waited)
        return waited

    def operation(self, operation: ir.Operation, waited: frozenset) -> frozenset:
        """The barriers waited at past `operation` and no sync, from `waited` before it."""
        name = operation.name
        if name == BARRIER_WAIT:
            return waited | {_barrier(operation.operands[0])}
        if name in SYNCS:
            return frozenset()
        if (name == "llvm.call_intrinsic"
                and ir.StringAttr(operation.attributes["intrin"]).value.startswith(TENSOR_CORE_COMMIT)):
            if (barrier := _barrier(operation.operands[0])) in waited:
                self.found[f"{self.where}: a tensor-core commit arrives at barrier {barrier} after a wait at "
                           f"it, no warpgroup sync between"] = None
            return waited
        if name == "func.call":
            callee = ir.FlatSymbolRefAttr(operation.attributes["callee"]).value
            called = self.functions.get(callee)
            return waited if called is None else self.region(called.regions[0], waited)
        if name in ("scf.for", "scf.while"):
            # A loop's turn starts where the last left it (a while's condition, then its body): to the fixed
            # point, none run too.
            entry = waited
            while True:
                turned = entry
                for each in operation.regions:
                    turned = self.region(each, turned)
                if (entry | turned) == entry:
                    return entry
                entry |= turned
        if name in ("scf.if", "scf.index_switch"):
            return frozenset().union(*(self.region(each, waited) for each in self.taken(operation)))
        return frozenset().union(waited, *(self.region(each, waited) for each in operation.regions))

    def taken(self, branch: ir.Operation) -> list[ir.Region]:
        """The regions of `branch` the walk's thread can take: the one its value selects where the thread's
        index alone decides it (`_value`), every one otherwise (an absent else among them)."""
        selector = _value(branch.operands[0], self.thread)
        if branch.name == "scf.if":
            then, otherwise = branch.regions
            return [otherwise] if selector == 0 else [then] if selector is not None else [then, otherwise]
        default, *cases = branch.regions
        if selector is None:
            return [default, *cases]
        values = list(ir.DenseI64ArrayAttr(branch.attributes["cases"]))
        return [cases[values.index(selector)] if selector in values else default]


_INTEGER_OPS: dict[str, Callable[[int, int], int]] = {
    "arith.addi": lambda a, b: a + b, "arith.subi": lambda a, b: a - b, "arith.muli": lambda a, b: a * b,
    "arith.divui": lambda a, b: a // b, "arith.divsi": lambda a, b: int(a / b),
    "arith.remui": lambda a, b: a % b,
    "arith.remsi": lambda a, b: int(math.fmod(a, b)), "arith.andi": lambda a, b: a & b,
    "arith.ori": lambda a, b: a | b, "arith.xori": lambda a, b: a ^ b, "arith.shli": lambda a, b: a << b,
    "arith.shrui": lambda a, b: a >> b, "arith.maxsi": max, "arith.minsi": min, "arith.maxui": max,
    "arith.minui": min}
_COMPARISONS = [lambda a, b: a == b, lambda a, b: a != b, lambda a, b: a < b, lambda a, b: a <= b,
                lambda a, b: a > b, lambda a, b: a >= b, lambda a, b: a < b, lambda a, b: a <= b,
                lambda a, b: a > b, lambda a, b: a >= b]
"""`arith.cmpi`'s predicates in their order (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge)."""


def _value(value: ir.Value, thread: _Thread | None) -> int | None:
    """`value` where the constants and `thread`'s index and block (`gpu.thread_id`, `gpu.block_dim`) alone
    make it (a warp's or a warpgroup's index, a role's branch), through integer arithmetic and casts, and a
    lane shuffle's (`nvvm.shfl.sync`, the first lane's own value); None where anything else enters it."""
    owner = value.owner
    if not hasattr(owner, "name"):
        return None
    operation = owner.operation if hasattr(owner, "operation") else owner
    name = operation.name
    if name == "arith.constant":
        try:
            return ir.IntegerAttr(operation.attributes["value"]).value
        except ValueError:  # A float's constant.
            return None
    if name in ("gpu.thread_id", "gpu.block_dim"):
        if thread is None:
            return None
        along = "xyz".index(str(operation.attributes["dimension"])[-2])
        return thread.block[along] if name == "gpu.block_dim" else thread.index if along == 0 else 0
    if name in ("arith.index_cast", "arith.index_castui", "arith.extui", "arith.extsi", "arith.trunci"):
        return _value(operation.operands[0], thread)
    if name == "nvvm.shfl.sync":
        return _value(operation.operands[1], thread)
    if name in _INTEGER_OPS or name == "arith.cmpi":
        a, b = (_value(operand, thread) for operand in operation.operands)
        if a is None or b is None:
            return None
        if name == "arith.cmpi":
            return int(_COMPARISONS[ir.IntegerAttr(operation.attributes["predicate"]).value](a, b))
        return _INTEGER_OPS[name](a, b)
    if name == "arith.select":
        chosen = _value(operation.operands[0], thread)
        return None if chosen is None else _value(operation.operands[1 if chosen else 2], thread)
    return None


def _commits(operation: ir.Operation) -> bool:
    """Whether `operation` holds a tensor-core commit (`TENSOR_CORE_COMMIT`): a kernel with none has none
    to walk to."""
    return any((inner.operation.name == "llvm.call_intrinsic"
               and ir.StringAttr(inner.operation.attributes["intrin"]).value.startswith(TENSOR_CORE_COMMIT))
               or _commits(inner.operation)
               for each in operation.regions for block in each.blocks for inner in block.operations)


def _launches(operation: ir.Operation) -> Iterator[ir.Operation]:
    """The `gpu.launch` operations within `operation`."""
    for each in operation.regions:
        for block in each.blocks:
            for inner in block.operations:
                if inner.operation.name == "gpu.launch":
                    yield inner.operation
                yield from _launches(inner.operation)


def _barrier(value: ir.Value) -> int | str:
    """The barrier array a barrier operand addresses: the byte offset in the kernel's dynamic shared memory of
    the view its address is computed from (each step its first operand: a cast, a pointer's arithmetic, a
    struct's field), the operand's own text where it reads none."""
    while True:
        owner = value.owner
        if not hasattr(owner, "name"):
            return str(value)
        operation = owner.operation if hasattr(owner, "operation") else owner
        if (operation.name == "memref.view"
                and operation.operands[0].owner.name == "gpu.dynamic_shared_memory"):
            return ir.IntegerAttr(operation.operands[1].owner.attributes["value"]).value
        if not operation.operands:
            return str(value)
        value = operation.operands[0]
