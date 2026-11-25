"""What an inlined Mosaic GPU body (`plgpu.inline_mgpu`'s function, opaque to a kernel's jaxpr) issues, read
from the code it emits: the body called under a scratch module on stand-ins of its operands, each op it emits
counted a warpgroup's lanes, the way `emission` counts the rest of the kernel (row 12466)."""

import dataclasses
import functools
import itertools
import math
import re
import weakref
from collections.abc import Callable, Sequence

import numpy as np
from jax._src import core as jax_core
from jax._src.interpreters import mlir as jax_mlir
from jax._src.pallas.mosaic_gpu import primitives as gpu_primitives
from jax._src.state import types as state_types
from jax.experimental.mosaic.gpu import fragmented_array, utils as mgpu_utils
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import arith, func

from dew.nn.kernels import generation
from dew.nn.kernels.templates.cost import Work, most, repeated

LANES = 128
"""The threads of a warpgroup, whose lanes a body's ops issue over: an op issues for each warp a lane of which
runs it, its 32 lanes all."""

WARP = 32

WALKED_TRIPS = 64
"""The most trips of a static loop the count walks one at a time, its induction variable known each trip;
past them it walks the body once, its induction variable unknown, and counts it its trips' times."""

_FLOAT = frozenset({
    "arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.remf", "arith.maximumf", "arith.minimumf",
    "arith.maxnumf", "arith.minnumf", "arith.negf", "arith.cmpf", "math.fma", "math.absf", "math.copysign"})
"""Elementwise float operations, on the fp32 lanes."""

_INTEGER = frozenset({
    "arith.addi", "arith.subi", "arith.divui", "arith.divsi", "arith.remui", "arith.remsi", "arith.andi",
    "arith.ori", "arith.xori", "arith.shli", "arith.shrui", "arith.shrsi", "arith.minsi", "arith.maxsi",
    "arith.minui", "arith.maxui", "arith.cmpi", "arith.ceildivsi", "arith.floordivsi"})
"""Elementwise integer operations, on the integer units where they compute a vector's lanes; a scalar's (an
index, an address) is free, as `emission` takes a kernel's scalar index arithmetic."""

_SPECIAL = frozenset({"math.exp", "math.exp2", "math.log", "math.log2", "math.log1p", "math.tanh",
                      "math.rsqrt", "math.sqrt", "math.sin", "math.cos", "math.erf"})
"""Special functions, on the special-function units (`Work.exponentials`)."""

_CONVERSIONS = frozenset({"arith.extf", "arith.truncf", "arith.fptosi", "arith.fptoui", "arith.sitofp",
                          "arith.uitofp"})

_MEMORY = frozenset({"memref.load", "memref.store", "vector.load", "vector.store", "llvm.load", "llvm.store"})

_FREE_PREFIXES = ("llvm.mlir.", "builtin.unrealized_conversion_cast", "gpu.thread_id",
                  "gpu.block_id", "gpu.block_dim", "gpu.grid_dim", "nvvm.read.ptx.sreg", "nvvm.barrier",
                  "nvvm.bar.", "nvvm.fence", "nvvm.shfl.sync", "nvvm.elect.sync", "nvvm.mbarrier",
                  "nvvm.cp.async.commit", "nvvm.cp.async.wait", "nvvm.cp.async.mbarrier")
"""Ops that move a value between registers or name a thread's place (a shuffle's lanes, a barrier's wait),
which compute nothing a unit's rate prices: register moves, addresses (`llvm.getelementptr`'s arithmetic),
a thread's indices, synchronization and a shuffle's exchange, which `emission` prices for no reduction
either."""

_FREE = frozenset({
    "arith.constant", "arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui", "arith.trunci",
    "arith.bitcast", "scf.yield", "func.return", "gpu.barrier", "llvm.getelementptr", "llvm.ptrtoint",
    "llvm.inttoptr", "llvm.addrspacecast", "llvm.extractvalue", "llvm.insertvalue", "llvm.bitcast",
    "llvm.mul", "llvm.add", "memref.subview", "memref.cast", "memref.reinterpret_cast",
    "memref.collapse_shape", "memref.expand_shape", "llvm.insertelement", "llvm.extractelement",
    "llvm.shufflevector", "llvm.mlir.undef", "llvm.mlir.poison", "llvm.mlir.constant",
    # A vector's registers regrouped: its elements taken, put, broadcast or reshaped, none computed.
    "vector.extract", "vector.insert", "vector.broadcast", "vector.splat", "vector.from_elements",
    "vector.extract_strided_slice", "vector.insert_strided_slice", "vector.shape_cast", "vector.bitcast",
    "vector.shuffle"})


class Uncounted(ValueError):
    """A body's emitted code holds what the count reads no rule for (an op, a trip count its operands state
    only at run time)."""


def body_work(fn: Callable, avals: Sequence, arg_types: Sequence, values: Sequence = (),
              warp: int | None = None) -> Work:
    """What `fn`, an `inline_mgpu` body taking operands of `avals` as `arg_types` states them, issues a
    warpgroup (its `warp`'s 32 lanes where a warp's role runs it, `plgpu.warp_map`'s): the body emitted for
    the target the trace runs for (`generation.gpu_target`), each integer scalar operand whose value the
    count holds (`values`, None where unknown: a block's start along its launch's axis) a constant of it, so
    the lanes its predicates take are read; once a body, target, operands, their layouts, values and warp
    (`_held`). Raises `Uncounted` where its code holds what no rule counts."""
    known = tuple(_scalar(aval, value) for aval, value in itertools.zip_longest(avals, values))
    capability = (generation.capability(generation.gpu_target().generation)
                  or generation.capability(generation.interpreted_target().generation))
    if capability is None:
        raise Uncounted("an inline Mosaic GPU body counted for no CUDA capability, its process's nor the "
                        "interpreted target's")
    key = (capability, tuple(map(str, avals)), tuple(map(str, arg_types)), known, warp)
    found = _held.get(fn, {})
    if key not in found:
        found[key] = _counted(fn, avals, arg_types, known, capability, warp)
        _held[fn] = found
    return found[key]


def _scalar(aval, value) -> int | None:
    """`value` where it is an integer scalar operand's concrete value, else None."""
    if getattr(aval, "shape", None) != () or not np.issubdtype(aval.dtype, np.integer):
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0 and np.issubdtype(value.dtype, np.integer):
        return int(value)
    return int(value) if isinstance(value, int | np.integer) else None


_held: "weakref.WeakKeyDictionary[Callable, dict]" = weakref.WeakKeyDictionary()


class _Launch:
    """A body's launch context as the count stands it in: each TMA copy it asks for (`async_copy`, a
    gathered ring's rows) marked where the body asks for it (`COPIED`), so the walk counts it under the
    branches and loops around it, its shared side's bytes moved between device and shared memory once
    however many lanes issue it; any other use of the context is no rule's."""

    def async_copy(self, *, src_ref, dst_ref, predicate=None, **options) -> None:
        del options
        shared = dst_ref if _shared(dst_ref.type) else src_ref
        moved = _bytes_of(ir.MemRefType(shared.type))
        # Marked under its issuing lanes' guard where it states one.
        mgpu_utils.inline_ptx(f"{COPIED} {moved};", has_side_effects=True, predicate=predicate)

    def __getattr__(self, name: str):
        raise Uncounted(f"a body's use of its launch context's {name}")


COPIED = "// dew.copied"
"""The mark a stood-in launch context leaves where a body asks it for a copy, with its bytes (`_Launch`)."""


def _counted(fn: Callable, avals: Sequence, arg_types: Sequence, known: Sequence, capability: tuple,
             warp: int | None) -> Work:
    with jax_mlir.make_ir_context() as context, ir.Location.unknown():
        from jaxlib.mosaic.python import mosaic_gpu as dialect

        dialect.register_dialect(context)
        module = ir.Module.create()
        i32 = ir.IntegerType.get_signless(32)
        module.operation.attributes["mosaic_gpu.arch_major"] = ir.IntegerAttr.get(i32, capability[0])
        module.operation.attributes["mosaic_gpu.arch_minor"] = ir.IntegerAttr.get(i32, capability[1])
        types = [_stand_in_type(aval, kind) for aval, kind in zip(avals, arg_types, strict=True)]
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("body", ir.FunctionType.get(types, []))
            block = function.add_entry_block()
            launch = _Launch()
            with ir.InsertionPoint(block):
                arguments = [argument if value is None else arith.constant(argument.type, value)
                             for argument, value in zip(block.arguments, known, strict=True)]
                fn(launch, *(_stand_in(aval, kind, argument)
                             for aval, kind, argument in zip(avals, arg_types, arguments, strict=True)))
                func.ReturnOp([])
        active = np.ones(LANES, dtype=bool)
        if warp is not None:
            # A warp's role runs it on its own 32 lanes.
            active = np.arange(LANES) // WARP == warp % (LANES // WARP)
        return _Lanes().region(function.operation.regions[0], active)


def _stand_in_type(aval, kind) -> ir.Type:
    if isinstance(kind, gpu_primitives.RefType):
        # The memref its body takes: the ref's view with its type's own transforms (a tile's tiling and
        # swizzle) applied, the layout it lies in.
        viewed = functools.reduce(lambda viewed, transform: transform.transform_type(viewed),
                                  kind.transforms, aval)
        assert isinstance(viewed, state_types.AbstractRef), viewed
        inner = viewed.inner_aval
        assert isinstance(inner, jax_core.ShapedArray), inner
        shared = "smem" in str(aval.memory_space).lower()
        return ir.MemRefType.get(inner.shape, mgpu_utils.dtype_to_ir_type(inner.dtype),
                                 memory_space=mgpu_utils.smem() if shared else None)
    return mgpu_utils.dtype_to_ir_type(aval.dtype)


def _stand_in(aval, kind, argument):
    """An operand as the body takes it: a ref's memref; an array's registers in its layout, a scalar's of
    none (Lane semantics: every operand not a ref is a `FragmentedArray`), a splat of the argument."""
    if isinstance(kind, gpu_primitives.RefType):
        return argument
    integral = np.issubdtype(aval.dtype, np.integer) or aval.dtype == np.bool_
    signed = bool(np.issubdtype(aval.dtype, np.signedinteger)) if integral else None
    layout = kind.to_mgpu() if aval.shape else None
    return fragmented_array.FragmentedArray.splat(argument, tuple(aval.shape), layout, is_signed=signed)


@dataclasses.dataclass
class _Lanes:
    """A walk of a body's code over a warpgroup's lanes: each scalar its lanes' values where the lanes' place
    and constants decide them (a thread's index, its warp, its lane), None where data does."""

    values: dict = dataclasses.field(default_factory=dict)
    partial: dict = dataclasses.field(default_factory=dict)
    # The values that differ between lanes: a thread's index, what any op computes of one, a region's
    # results where its condition or what it yields does; and of those, each one's lanes up to a term every
    # lane holds alike where its arithmetic states them (an address: a base pointer plus a lane's column).
    varying: set = dataclasses.field(default_factory=set)
    relative: dict = dataclasses.field(default_factory=dict)

    def region(self, region, active: np.ndarray) -> Work:
        work = Work()
        for block in region.blocks:
            for inner in block.operations:
                work = work + self.op(inner.operation, active)
        return work

    def op(self, op, active: np.ndarray) -> Work:
        work = self.walked(op, active)
        if op.results and (any(operand in self.varying for operand in op.operands) or _thread(op.name)
                           or any(operand in self.varying for operand in _yielded(op))):
            self.varying.update(op.results)
        return work

    def walked(self, op, active: np.ndarray) -> Work:
        name = op.name
        if name == "scf.for":
            trips = _trips(*(self.read(op.operands[at]) for at in range(3)))
            if trips is None:
                raise Uncounted("an scf.for whose trips its operands state only at run time")
            # A carried value's argument varies between lanes where its start or what a trip yields does.
            body = op.regions[0].blocks[0]
            carried_values = zip(body.arguments[1:], op.operands[3:], _yielded(op), strict=False)
            for argument, start, carried in carried_values:
                if start in self.varying or carried in self.varying:
                    self.varying.add(argument)
            if trips > WALKED_TRIPS:
                return repeated(self.region(op.regions[0], active), trips)
            return self.trips(op, trips, active)
        if name == "scf.if":
            maybe, surely = self.bounds(op.operands[0])
            if maybe.all() and not surely.any() and op.operands[0] not in self.varying:
                # Data every lane holds alike decides it: one branch runs, each branch's most, as `emission`
                # takes an unknown conditional. Where data a lane holds decides it, a warp's lanes may part
                # and run both, each over every lane (below).
                branches = [self.region(region, active) for region in op.regions if region.blocks]
                return most(branches)
            # Each branch over the lanes that may take it: a lane its place decides takes one, a lane data
            # decides may take either.
            work = self.region(op.regions[0], active & maybe)
            if len(op.regions) > 1 and op.regions[1].blocks:
                work = work + self.region(op.regions[1], active & ~surely)
            return work
        if op.regions:
            raise Uncounted(f"{name}'s regions")
        self.evaluate(op)
        return self.issued(op, active)

    def trips(self, op, trips: int, active: np.ndarray) -> Work:
        """An scf.for of `trips` few trips walked one trip at a time, its induction variable each trip's and
        its carried values each trip's start where known (a row a trip selects, a lane a trip guards); its
        results the last trip's."""
        body = op.regions[0].blocks[0]
        lower, step = self.read(op.operands[0]), self.read(op.operands[2])
        assert lower is not None and step is not None  # `_trips` read them
        carried = [self.read(start) for start in op.operands[3:]]
        work = Work()
        for trip in range(trips):
            self.values[body.arguments[0]] = lower + trip * step
            for argument, lanes in zip(body.arguments[1:], carried, strict=False):
                if lanes is None:
                    self.values.pop(argument, None)
                else:
                    self.values[argument] = lanes
            work = work + self.region(op.regions[0], active)
            carried = [self.read(value) for value in _yielded(op)]
        for returned, lanes in zip(op.results, carried, strict=False):
            # An earlier walk's facts of its results (an outer trip's) dropped before this one's.
            for held in (self.values, self.partial, self.relative):
                held.pop(returned, None)
            if lanes is not None:
                self.values[returned] = lanes
        return work

    def read(self, value) -> np.ndarray | None:
        return self.values.get(value)

    def bounds(self, value) -> tuple[np.ndarray, np.ndarray]:
        """A predicate's lanes that may hold it and that surely do: its lanes where known, else what an
        `and` or `or` of a known predicate bounds (a column's validity and a data-decided slice's: false
        wherever the column is invalid), else every lane may and none surely does."""
        known = self.values.get(value)
        if known is not None:
            return known != 0, known != 0
        return self.partial.get(value, (np.ones(LANES, dtype=bool), np.zeros(LANES, dtype=bool)))

    def evaluate(self, op) -> None:
        """Its results' lanes where its operands' are known."""
        name = op.name
        if not op.results or len(op.results) != 1:
            return
        produced = op.results[0]
        # What an earlier trip of a loop held of it is no longer known (`walked`'s trips).
        for held in (self.values, self.partial, self.relative):
            held.pop(produced, None)
        if isinstance(produced.type, ir.VectorType | ir.MemRefType):
            return
        if name == "arith.constant":
            attribute = op.attributes["value"]
            if isinstance(attribute, ir.IntegerAttr):
                self.values[produced] = np.full(LANES, attribute.value, dtype=np.int64)
            return
        if name in ("gpu.thread_id", "nvvm.read.ptx.sreg.tid.x"):
            dimension = str(op.attributes["dimension"]) if "dimension" in op.attributes else "x"
            self.values[produced] = (np.arange(LANES, dtype=np.int64) if dimension.endswith("x>")
                                   or dimension == "x" else np.zeros(LANES, dtype=np.int64))
            return
        if name == "gpu.block_dim":
            self.values[produced] = np.full(LANES, LANES, dtype=np.int64)
            return
        if name == "nvvm.read.ptx.sreg.laneid":
            self.values[produced] = np.arange(LANES, dtype=np.int64) % WARP
            return
        read = [self.read(operand) for operand in op.operands]
        operands = [lanes for lanes in read if lanes is not None]
        if len(operands) != len(read) and name in ("arith.andi", "arith.ori") and _predicate(produced.type):
            (maybe_a, surely_a), (maybe_b, surely_b) = (self.bounds(operand) for operand in op.operands)
            self.partial[produced] = ((maybe_a & maybe_b, surely_a & surely_b) if name == "arith.andi"
                                      else (maybe_a | maybe_b, surely_a | surely_b))
            return
        if not operands or len(operands) != len(read):
            relative = self.relative_of(name, op)
            if relative is not None:
                self.relative[produced] = relative
            return
        found = _lane_values(name, op, operands)
        if found is not None:
            self.values[produced] = found

    def offsets(self, value) -> np.ndarray | None:
        """`value`'s lanes up to a term every lane holds alike: its lanes where known, none where no lane's
        place decides it (a scalar operand's, a base pointer's), what `relative_of` derived, else None."""
        if (known := self.values.get(value)) is not None:
            return known
        if value not in self.varying:
            return np.zeros(LANES, dtype=np.int64)
        return self.relative.get(value)

    def relative_of(self, name: str, op) -> np.ndarray | None:
        """An address's arithmetic over values known up to a uniform term: sums and differences of them,
        a product by a constant every lane holds, a cast, and an element pointer (`_element_offsets`)."""
        offsets = [self.offsets(operand) for operand in op.operands]
        known = [each for each in offsets if each is not None]
        match name:
            case "arith.addi" | "llvm.add" if len(known) == len(offsets) == 2:
                return known[0] + known[1]
            case "arith.subi" if len(known) == len(offsets) == 2:
                return known[0] - known[1]
            case "arith.muli" | "llvm.mul":
                for constant, other in ((0, 1), (1, 0)):
                    held = self.values.get(op.operands[constant])
                    if held is not None and np.all(held == held[0]) and offsets[other] is not None:
                        return offsets[other] * held[0]
            case "arith.extsi" | "arith.extui" | "arith.index_cast" | "arith.index_castui" | "arith.trunci":
                return offsets[0]
            case "llvm.getelementptr" if all(each is not None for each in offsets):
                index = op.operands[1] if len(op.operands) > 1 else None
                return _element_offsets(op, offsets, index is None or index in self.values
                                        or index not in self.varying)
        return None

    def issued(self, op, active: np.ndarray) -> Work:
        """What `op` issues over the `active` lanes: an op computes for every lane of a warp any lane of which
        runs it; a memory op moves its active lanes' bytes."""
        name = op.name
        warps = active.reshape(-1, WARP).any(axis=1).sum()
        issued, moving = float(warps * WARP), float(active.sum())
        if name in _MEMORY:
            stored = op.operands[0].type if name.endswith("store") and name != "llvm.store" else None
            value = (op.operands[0].type if name == "llvm.store" else stored if stored is not None
                     else op.results[0].type)
            stores = name in ("memref.store", "vector.store", "llvm.store")
            address = op.operands[1] if stores else op.operands[0]
            indices = op.operands[2 if stores else 1:] if name.startswith(("memref.", "vector.")) else ()
            # Each distinct address the active lanes access moves its bytes once, however many lanes access
            # it (a head's scalar every lane reads, a column two lanes hold in a replicated layout); where
            # no lane's address is known up to a common term, each active lane's.
            places = [self.offsets(each) for each in (address, *indices)]
            known = [each for each in places if each is not None]
            distinct = (float(np.unique(np.stack(known)[:, active], axis=1).shape[1])
                        if active.any() and len(known) == len(places) else moving)
            moved = _bytes(value) * distinct
            return Work(shared_bytes=moved) if _shared(address.type) else Work(device_bytes=moved)
        if name in _FREE or name.startswith(_FREE_PREFIXES):
            return Work()
        if name == "llvm.inline_asm":
            # `$N` numbers the block's outputs first (its constraints' `=`), then its operands.
            outputs = sum(each.startswith("=") for each in
                          ir.StringAttr(op.attributes["constraints"]).value.split(","))

            def operand(at: int) -> tuple[np.ndarray | None, tuple[np.ndarray, np.ndarray]]:
                if at < outputs:
                    raise Uncounted(f"a PTX guard or operand on the block's output ${at}")
                value = op.operands[at - outputs]
                return self.read(value), self.bounds(value)

            return _assembly(ir.StringAttr(op.attributes["asm_string"]).value, active, operand)
        if name == "nvvm.cp.async.shared.global":
            # Each active lane's copy of `size` bytes from device into shared memory.
            copied = ir.IntegerAttr(op.attributes["size"]).value * moving
            return Work(device_bytes=copied, shared_bytes=copied)
        typed = op.operands[-1].type if name in ("arith.cmpf", "arith.cmpi", "arith.select") else (
            op.results[0].type if op.results else None)
        if typed is None:
            raise Uncounted(f"{name}, of no result")
        elements = _elements(typed) * issued
        if name == "arith.select":
            if _is_float(typed):
                return Work(operations=elements)
            return Work(integer_operations=elements) if isinstance(typed, ir.VectorType) else Work()
        if name in _FLOAT:
            return Work(operations=elements)
        if name in _SPECIAL:
            return Work(exponentials=elements)
        if name in _CONVERSIONS:
            return Work(conversions=elements)
        if name == "arith.muli":
            return Work(integer_multiply_adds=elements) if isinstance(typed, ir.VectorType) else Work()
        if name in _INTEGER:
            return Work(integer_operations=elements) if isinstance(typed, ir.VectorType) else Work()
        raise Uncounted(f"no rule counts {name}")


_PTX_FLOAT = ("fma.", "mul.", "add.", "sub.", "max.", "min.", "neg.", "abs.")
_PTX_SPECIAL = ("ex2.", "lg2.", "rcp.", "rsqrt.", "sqrt.", "tanh.", "sin.", "cos.")
_PTX_FREE = ("shfl.", "mov.", "bar.", "barrier.", "fence.", "membar.", "elect.", "vote.",
             "activemask.", "mbarrier.", "cp.async.commit_group", "cp.async.wait_group", "cp.async.wait_all",
             "cp.async.mbarrier")
_PTX_INTEGER = ("lop3.", "shf.", "prmt.", "and.", "or.", "xor.", "not.", "shl.", "shr.", "bfe.", "bfi.",
                "add.", "sub.", "min.", "max.", "abs.", "neg.")
_PTX_MULTIPLY = ("mul.", "mad.")


def _assembly(text: str, active: np.ndarray, operand: Callable) -> Work:
    """Inline PTX's work, an instruction at a time: float arithmetic on the fp32 lanes (a packed pair's, `x2`,
    two an instruction), special functions on their units, integer arithmetic and bit manipulation on the
    integer units (a multiply on the FMA pipe; a compare's setp and a select's selp each an issue, ISETP and
    SEL, as `BitOp.issues` prices them; adds accumulating into one register, after an immediate's move, one
    IADD3 of up to three operands, as ptxas fuses them), a conversion at the conversions' rate, a load or
    store its active lanes' bytes; a shuffle, a move and synchronization free (scalar, a lane's exchange); a
    global atomic its word read and written by each lane that runs it. An instruction guarded
    (`@p`, `@!p`) runs on the `active` lanes that may take its guard: an operand's (`$N`, `operand`), or a
    predicate the block sets itself from operands and immediates whose lanes are known (`setp`, and-ed or
    or-ed with another predicate: an elected lane's, a last arriver's among it); a memory instruction whose
    guard's lanes are unknown is refused. Raises `Uncounted` for any other instruction."""
    work = Work()
    predicates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    unknown = (np.ones(LANES, dtype=bool), np.zeros(LANES, dtype=bool))

    def lanes_of(word: str) -> np.ndarray | None:
        # An operand's lanes, an immediate's value on every lane, or None for a register the block sets.
        word = word.strip(",")
        if word.startswith("$") and word[1:].isdigit():
            return operand(int(word[1:]))[0]
        value = _ptx_integer(word)
        return None if value is None else np.full(LANES, value, dtype=np.int64)

    def bounds_of(word: str) -> tuple[np.ndarray, np.ndarray]:
        word = word.strip(",")
        if word.startswith("$") and word[1:].isdigit():
            return operand(int(word[1:]))[1]
        return predicates.get(word, unknown)

    # The three-way add being formed (`IADD3`): its register, its operands so far, whether an add issued it,
    # and the lanes its immediate's move issues it on where no add does.
    adding: tuple[str, int, bool, float] | None = None

    def closed(pending: tuple[str, int, bool, float] | None) -> Work:
        # A three-way add no add issued: its immediate's move materializes the constant, an issue (an IADD3 of
        # its immediate alone, `BitOp.issues`).
        return Work() if pending is None or pending[2] else Work(integer_operations=pending[3])

    for line in _statements(text):
        words = line.split()
        running = active
        guard = None
        if words and words[0].startswith("@"):
            negated = words[0].startswith("@!")
            guard, words = words[0].lstrip("@!"), words[1:]
            maybe, surely = bounds_of(guard)
            if negated:
                maybe, surely = ~surely, ~maybe
            running = active & maybe
            # Its lanes known over the active lanes where one of them surely holds it or surely does not (a
            # ticket's last arriver among its elected lane is bounded by it), or where no lane is active.
            if not active.any() or (active & surely).any() or (active & ~maybe).any():
                guard = None
        issued = float(running.reshape(-1, WARP).any(axis=1).sum() * WARP)
        moving = float(running.sum())
        if words and words[0].startswith("setp."):
            predicates[words[1].strip(",")] = _set(words, lanes_of, bounds_of)
            work, adding = work + closed(adding) + _instruction(words, issued, moving), None
            continue
        if words and words[0].startswith("mov.") and guard is None and len(words) == 3 and (
                _ptx_integer(words[2]) is not None):
            # An immediate's move: an add accumulating into its register issues the IADD3 it starts.
            work, adding = work + closed(adding), (words[1].strip(","), 1, False, issued)
            continue
        if words and words[0].startswith(("add.", "sub.")) and guard is None and len(words) == 4 and not any(
                kind in words[0] for kind in ("f32", "f16", "bf16", "f64")):
            out, first = words[1].strip(","), words[2].strip(",")
            if adding is not None and out == first == adding[0] and adding[1] < 3:
                issued_already, adding = adding[2], (out, adding[1] + 1, True, adding[3])
                if issued_already:
                    continue
            else:
                work, adding = work + closed(adding), (out, 2, True, issued)
        elif words and not words[0].startswith("//"):
            work, adding = work + closed(adding), None
        if " ".join(words).startswith(COPIED):
            # A launch context's copy (`_Launch`): its bytes once, where any lane reaches it and takes
            # its issuer's guard.
            moved = float(words[-1]) * bool(moving)
            work = work + Work(device_bytes=moved, shared_bytes=moved)
            continue
        if words and words[0].startswith("//"):
            continue
        if not words or words[0].startswith(".") or words[0].startswith(_PTX_FREE):
            continue
        if guard is not None and words[0].startswith(("ld.", "st.", "atom.", "red.", "cp.async.")):
            raise Uncounted(f"a memory instruction under a guard of unknown lanes ({guard}): {words[0]}")
        work = work + _instruction(words, issued, moving)
    return work + closed(adding)


def _statements(text: str) -> list[str]:
    """A PTX block's statements: split at their semicolons, a block's braces stripped from their edges (an
    operand list's braces, a vector's `{$0, $1}`, kept)."""
    found = []
    for statement in text.split(";"):
        statement = statement.strip()
        while statement[:1] in ("{", "}"):
            statement = statement[1:].strip()
        while statement[-1:] == "}" and statement.count("{") < statement.count("}"):
            statement = statement[:-1].strip()
        found.append(statement)
    return found


def _instruction(words: list[str], issued: float, moving: float) -> Work:
    """One PTX instruction's work (`_assembly`'s rules), `issued` its issuing warps' lanes and `moving` its
    running lanes."""
    mnemonic = words[0]
    elements = 2 if "x2" in mnemonic else 1
    floating = any(kind in mnemonic for kind in ("f32", "f16", "bf16", "f64"))
    if mnemonic.startswith(("setp.", "selp.")):
        # A compare's or a select's one issue: FSETP or FSEL on the fp32 lanes, ISETP or SEL on the ALU.
        return Work(operations=elements * issued) if floating else Work(integer_operations=elements * issued)
    if mnemonic.startswith(_PTX_FLOAT) and floating:
        return Work(operations=elements * issued)
    if mnemonic.startswith(_PTX_SPECIAL):
        return Work(exponentials=elements * issued)
    if mnemonic.startswith("cvt."):
        return Work(conversions=elements * issued)
    if mnemonic.startswith(_PTX_MULTIPLY):
        return Work(integer_multiply_adds=elements * issued)
    if mnemonic.startswith(_PTX_INTEGER):
        return Work(integer_operations=elements * issued)
    if mnemonic.startswith("mma.") and (shape := re.search(r"\.m(\d+)n(\d+)k(\d+)\.", mnemonic)):
        # A warp's matrix product: 2 M N K operations each warp that issues it, in its operands' mode.
        m, n, k = map(int, shape.groups())
        operations = 2 * m * n * k * issued / WARP
        if ".tf32" in mnemonic:
            return Work(tf32_flops=operations)
        if any(kind in mnemonic for kind in (".e4m3", ".e5m2")):
            return Work(fp8_flops=operations)
        if any(kind in mnemonic for kind in (".bf16", ".f16")):
            return Work(flops=operations)
    elif mnemonic.startswith(("cp.async.ca.", "cp.async.cg.")):
        # A lane's copy of its operand's bytes (16 where it states none) from device into shared memory.
        sizes = [int(word.strip(",")) for word in words[1:] if word.strip(",").isdigit()]
        copied = (sizes[0] if sizes else 16) * moving
        return Work(device_bytes=copied, shared_bytes=copied)
    elif mnemonic.startswith(("atom.", "red.")) and ".global" in mnemonic:
        # Its word read and written, an atomic's and a reduction's alike, at its type's bytes.
        moved = _ptx_bytes(mnemonic) * 2 * moving
        return Work(device_bytes=moved)
    elif mnemonic.startswith(("ld.", "st.")):
        moved = _ptx_bytes(mnemonic) * moving
        return Work(shared_bytes=moved) if ".shared" in mnemonic else Work(device_bytes=moved)
    raise Uncounted(f"no rule counts the PTX instruction {mnemonic}")


def _ptx_bytes(mnemonic: str) -> int:
    """A PTX memory instruction's bytes a lane: its type's (`.u32`, `.f16`, `.b64`, `.b128`, a packed pair's
    `.f16x2` or `.bf16x2`) times its vector's count (`.v2`, `.v4`); refused where it states no type."""
    found = re.search(r"\.(?:[usbf](8|16|32|64|128)|b?f16x2)\b", mnemonic)
    if found is None:
        raise Uncounted(f"a PTX memory instruction of no type the count reads: {mnemonic}")
    bits = int(found.group(1)) if found.group(1) else 32  # a packed pair of 16-bit floats
    vector = re.search(r"\.v(2|4|8)\b", mnemonic)
    return bits // 8 * (int(vector.group(1)) if vector else 1)


def _ptx_integer(word: str) -> int | None:
    """A PTX integer literal's value (hexadecimal `0x`, binary `0b`, octal `0`, decimal, a `U` suffix and a
    sign taken), None where `word` is none or its value lies past a 64-bit lane's."""
    found = re.fullmatch(r"(-?)(0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9][0-9]*)[uU]?", word)
    if found is None:
        return None
    sign, digits = found.groups()
    if digits[:2] in ("0x", "0X"):
        value = int(digits[2:], 16)
    elif digits[:2] in ("0b", "0B"):
        value = int(digits[2:], 2)
    else:
        value = int(digits, 8) if digits.startswith("0") else int(digits)
    value = -value if sign else value
    return value if -(1 << 63) <= value < (1 << 63) else None


_PTX_COMPARES = {"eq": np.equal, "ne": np.not_equal, "lt": np.less, "le": np.less_equal, "gt": np.greater,
                 "ge": np.greater_equal, "lo": np.less, "ls": np.less_equal, "hi": np.greater,
                 "hs": np.greater_equal}


def _set(words: list[str], lanes_of: Callable, bounds_of: Callable) -> tuple[np.ndarray, np.ndarray]:
    """A `setp`'s predicate's lanes that may and that surely hold it (`setp.cmp[.and|.or].type p, a, b[,
    q]`): its comparison's where both sides' lanes are known and its type an integer's (each side read at
    its width and signedness, an unsigned comparison's `lo`, `ls`, `hi`, `hs` unsigned), else any; and-ed or
    or-ed with `q`'s."""
    parts = words[0].split(".")
    compare = _PTX_COMPARES.get(parts[1])
    sides = [lanes_of(word) for word in words[2:4]]
    kind = re.fullmatch(r"([usb])(8|16|32|64)", parts[-1])
    if compare is not None and kind is not None and all(side is not None for side in sides):
        signed = kind.group(1) == "s" and parts[1] not in ("lo", "ls", "hi", "hs")
        first, second = (_at_width(side, int(kind.group(2)), signed) for side in sides)
        held = compare(first, second)
        maybe, surely = held, held
    else:
        maybe, surely = np.ones(LANES, dtype=bool), np.zeros(LANES, dtype=bool)
    if len(parts) > 3 and parts[2] in ("and", "or") and len(words) > 4:
        other_maybe, other_surely = bounds_of(words[4])
        if parts[2] == "and":
            return maybe & other_maybe, surely & other_surely
        return maybe | other_maybe, surely | other_surely
    return maybe, surely


def _lane_values(name: str, op, operands: list[np.ndarray]) -> np.ndarray | None:
    """`name` over its operands' lanes, for the integer arithmetic a lane's place is computed by."""
    first, last = operands[0], operands[-1]
    match name:
        case "arith.index_cast" | "arith.index_castui" | "arith.extsi" | "arith.extui" | "arith.trunci":
            # An extension reads its operand's bits signed (extsi, index_cast's) or unsigned (extui,
            # index_castui's); a truncation keeps its result's bits, unsigned.
            signed = name in ("arith.extsi", "arith.index_cast")
            width = _width(op.results[0].type if name == "arith.trunci" else op.operands[0].type)
            if width is None or width >= 64:
                return None if width is None else first
            return _at_width(first, width, signed=signed and name != "arith.trunci").astype(np.int64)
        case "arith.addi":
            return first + last
        case "arith.subi":
            return first - last
        case "arith.muli":
            return first * last
        case "arith.floordivsi":
            return np.where(last == 0, 0, first // np.where(last == 0, 1, last))
        case "arith.divsi" | "arith.remsi":
            # LLVM's sdiv and srem truncate toward zero, the remainder taking the dividend's sign.
            divisor = np.where(last == 0, 1, last)
            quotient = np.abs(first) // np.abs(divisor) * np.sign(first) * np.sign(divisor)
            return np.where(last == 0, 0, quotient if name == "arith.divsi" else first - divisor * quotient)
        case "arith.divui" | "arith.remui" | "arith.shrui" | "arith.minui" | "arith.maxui":
            return _unsigned(name, first, last, _width(op.results[0].type))
        case "arith.andi":
            return first & last
        case "arith.ori":
            return first | last
        case "arith.xori":
            return first ^ last
        case "arith.shli":
            return first << last
        case "arith.shrsi":
            return first >> last
        case "arith.minsi":
            return np.minimum(first, last)
        case "arith.maxsi":
            return np.maximum(first, last)
        case "arith.cmpi":
            width = _width(op.operands[0].type)
            return None if width is None else _compared(int(op.attributes["predicate"].value), first, last,
                                                        width)
        case "arith.select":
            return np.where(first.astype(bool), operands[1], last)
    return None


def _unsigned(name: str, first: np.ndarray, last: np.ndarray, width: int | None) -> np.ndarray | None:
    """An unsigned integer op over lanes held signed: its operands read unsigned at `width` bits, its result
    held signed at them again; None where the width is unknown."""
    if width is None:
        return None
    first, last = (_at_width(side, width, signed=False) for side in (first, last))
    divisor = np.where(last == 0, 1, last)
    unsigned = {"arith.divui": lambda: np.where(last == 0, 0, first // divisor),
                "arith.remui": lambda: np.where(last == 0, 0, first % divisor),
                "arith.shrui": lambda: first >> last,
                "arith.minui": lambda: np.minimum(first, last),
                "arith.maxui": lambda: np.maximum(first, last)}[name]()
    return _at_width(unsigned, width, signed=True).astype(np.int64)


def _compared(predicate: int, first: np.ndarray, second: np.ndarray, width: int) -> np.ndarray:
    """`arith.cmpi`'s predicate (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge) over lanes of `width`
    bits, each side read signed or unsigned as the predicate reads it."""
    compare = (np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal)
    signed = predicate in (2, 3, 4, 5)
    first, second = (_at_width(side, width, signed) for side in (first, second))
    return compare[predicate if predicate < 6 else predicate - 4](first, second).astype(np.int64)


def _at_width(lanes: np.ndarray, width: int, signed: bool) -> np.ndarray:
    """`lanes` as `width`-bit integers read signed or unsigned (Python's integers, as wide as they need)."""
    held = np.asarray(lanes, dtype=object) % (1 << width)
    return np.where(held >= (1 << (width - 1)), held - (1 << width), held) if signed else held


def _width(value_type: ir.Type) -> int | None:
    """An integer's bits (an index's 64), None for any other type."""
    if isinstance(value_type, ir.IntegerType):
        return value_type.width
    return 64 if isinstance(value_type, ir.IndexType) else None


def _trips(lower, upper, step) -> int | None:
    """An scf.for's trips where its bounds are one constant over every lane; None otherwise."""
    if any(bound is None or np.any(bound != bound[0]) for bound in (lower, upper, step)):
        return None
    return max(0, -(-(int(upper[0]) - int(lower[0])) // int(step[0])))


def _thread(name: str) -> bool:
    """Whether `name` reads a thread's place: its index, its lane."""
    return name in ("gpu.thread_id", "gpu.lane_id") or name.startswith("nvvm.read.ptx.sreg.tid") or (
        name == "nvvm.read.ptx.sreg.laneid")


def _yielded(op) -> list:
    """The values the regions of `op` yield (an scf.if's or scf.for's results)."""
    return [operand for region in op.regions for block in region.blocks for inner in block.operations
            if inner.operation.name == "scf.yield" for operand in inner.operation.operands]


def _predicate(value_type: ir.Type) -> bool:
    return isinstance(value_type, ir.IntegerType) and value_type.width == 1


def _elements(value_type: ir.Type) -> int:
    return math.prod(value_type.shape) if isinstance(value_type, ir.VectorType) else 1


def _is_float(value_type: ir.Type) -> bool:
    element = value_type.element_type if isinstance(value_type, ir.VectorType) else value_type
    return isinstance(element, ir.FloatType)


def _element_offsets(op, offsets: list, known: bool) -> np.ndarray | None:
    """An `llvm.getelementptr`'s address in bytes up to a uniform term, its base's (`offsets[0]`) plus its
    one index's, read signed at its width as LLVM reads it, times its element's stride (a dynamic index
    `offsets[1]`, a constant one a uniform term), so each index of a chain of them keeps its lanes; None for
    several indices (a struct's or array's walk), an element whose stride is no rule's (`_stride`), or a
    narrow index its lanes hold apart known only up to a uniform term (not `known`: its wrap at its width
    depends on that term; an index every lane holds alike moves no lane from another)."""
    indices = ir.DenseI32ArrayAttr(op.attributes["rawConstantIndices"])
    if len(indices) != 1:
        return None
    base = offsets[0]
    if len(offsets) == 1:
        return base
    stride = _stride(ir.TypeAttr(op.attributes["elem_type"]).value)
    index_type = op.operands[1].type
    narrow = isinstance(index_type, ir.IntegerType) and index_type.width < 64
    if stride is None or (narrow and not known):
        return None
    return base + (_at_width(offsets[1], index_type.width, signed=True).astype(np.int64) if narrow
                   else offsets[1]) * stride


def _stride(element: ir.Type) -> int | None:
    """An element's allocation stride in bytes as LLVM lays an array of it out, where it is its payload's: a
    scalar of 8, 16, 32 or 64 bits, or a vector of a power of two of them (its alignment its size); None for
    any other (a vector of three, padded to four; an i24, padded to four bytes; i1s, packed to bits), whose
    layout no rule states."""
    scalar = element.element_type if isinstance(element, ir.VectorType) else element
    if isinstance(scalar, ir.IndexType):
        bits = 64
    elif isinstance(scalar, ir.IntegerType | ir.FloatType):
        bits = scalar.width
    else:
        return None
    if bits not in (8, 16, 32, 64):
        return None
    count = _elements(element)
    return _bytes(element) if count & (count - 1) == 0 else None


def _bytes(value_type: ir.Type) -> int:
    element = value_type.element_type if isinstance(value_type, ir.VectorType) else value_type
    if isinstance(element, ir.IntegerType | ir.FloatType):
        return _elements(value_type) * max(1, element.width // 8)
    if isinstance(element, ir.IndexType):
        return _elements(value_type) * 8
    raise Uncounted(f"a memory op of {value_type}, whose bytes no rule states")


def _bytes_of(memref: ir.MemRefType) -> int:
    """A memref's bytes: its elements at their width."""
    element = memref.element_type
    if not isinstance(element, ir.IntegerType | ir.FloatType):
        raise Uncounted(f"a copy of {memref}, whose bytes no rule states")
    return math.prod(memref.shape) * max(1, element.width // 8)


def _shared(address_type: ir.Type) -> bool:
    """Whether an address is into shared memory: a memref in its space, an LLVM pointer of address space 3."""
    if isinstance(address_type, ir.MemRefType):
        return address_type.memory_space is not None and "workgroup" in str(address_type.memory_space)
    return "ptr<3>" in str(address_type)
