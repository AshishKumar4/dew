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
from dew.nn.kernels.templates.cost import Work

LANES = 128
"""The threads of a warpgroup, whose lanes a body's ops issue over: an op issues for each warp a lane of which
runs it, its 32 lanes all."""

WARP = 32

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
    capability = generation.capability(generation.gpu_target().generation) or (9, 0)
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
            return _repeated(self.region(op.regions[0], active), trips)
        if name == "scf.if":
            maybe, surely = self.bounds(op.operands[0])
            if maybe.all() and not surely.any() and op.operands[0] not in self.varying:
                # Data every lane holds alike decides it: one branch runs, each branch's most, as `emission`
                # takes an unknown conditional. Where data a lane holds decides it, a warp's lanes may part
                # and run both, each over every lane (below).
                branches = [self.region(region, active) for region in op.regions if region.blocks]
                return functools.reduce(_most, branches, Work())
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
        a product by a constant every lane holds, a cast, and an element pointer of one index."""
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
            case "llvm.getelementptr" if len(offsets) <= 2 and all(each is not None for each in offsets):
                return offsets[-1] if len(offsets) == 2 else offsets[0]
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
_PTX_FREE = ("shfl.", "mov.", "setp.", "selp.", "bar.", "barrier.", "fence.", "membar.", "elect.", "vote.",
             "activemask.", "mbarrier.", "cp.async.commit_group", "cp.async.wait_group", "cp.async.wait_all",
             "cp.async.mbarrier")
_PTX_INTEGER = ("lop3.", "shf.", "prmt.", "and.", "or.", "xor.", "not.", "shl.", "shr.", "bfe.", "bfi.",
                "add.", "sub.", "min.", "max.", "abs.", "neg.")
_PTX_MULTIPLY = ("mul.", "mad.")


def _assembly(text: str, active: np.ndarray, operand: Callable) -> Work:
    """Inline PTX's work, an instruction at a time: float arithmetic on the fp32 lanes (a packed pair's, `x2`,
    two an instruction), special functions on their units, integer arithmetic and bit manipulation on the
    integer units (a multiply on the FMA pipe), a conversion at the conversions' rate, a load or store its
    active lanes' bytes; a shuffle, a move, a predicate and synchronization free (scalar, a lane's
    exchange); a global atomic its word read and written by each lane that runs it. An instruction guarded
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
            continue
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
    return work


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
        return Work(device_bytes=8 * moving)
    elif mnemonic.startswith(("ld.", "st.")):
        moved = 4 * moving
        return Work(shared_bytes=moved) if ".shared" in mnemonic else Work(device_bytes=moved)
    raise Uncounted(f"no rule counts the PTX instruction {mnemonic}")


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
    q]`): its comparison's where both sides' lanes are known, else any; and-ed or or-ed with `q`'s."""
    parts = words[0].split(".")
    compare = _PTX_COMPARES.get(parts[1])
    sides = [lanes_of(word) for word in words[2:4]]
    if compare is not None and all(side is not None for side in sides):
        held = compare(sides[0], sides[1])
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
            return first
        case "arith.addi":
            return first + last
        case "arith.subi":
            return first - last
        case "arith.muli":
            return first * last
        case "arith.divui" | "arith.divsi" | "arith.floordivsi":
            return np.where(last == 0, 0, first // np.where(last == 0, 1, last))
        case "arith.remui" | "arith.remsi":
            return np.where(last == 0, 0, first % np.where(last == 0, 1, last))
        case "arith.andi":
            return first & last
        case "arith.ori":
            return first | last
        case "arith.xori":
            return first ^ last
        case "arith.shli":
            return first << last
        case "arith.shrui" | "arith.shrsi":
            return first >> last
        case "arith.minsi" | "arith.minui":
            return np.minimum(first, last)
        case "arith.maxsi" | "arith.maxui":
            return np.maximum(first, last)
        case "arith.cmpi":
            return _compared(int(op.attributes["predicate"].value), first, last)
        case "arith.select":
            return np.where(first.astype(bool), operands[1], last)
    return None


def _compared(predicate: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """`arith.cmpi`'s predicate (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge) over lanes."""
    compare = (np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal)
    return compare[predicate if predicate < 6 else predicate - 4](first, second).astype(np.int64)


def _trips(lower, upper, step) -> int | None:
    """An scf.for's trips where its bounds are one constant over every lane; None otherwise."""
    if any(bound is None or np.any(bound != bound[0]) for bound in (lower, upper, step)):
        return None
    return max(0, -(-(int(upper[0]) - int(lower[0])) // int(step[0])))


def _repeated(work: Work, count: int) -> Work:
    return Work(**{field.name: getattr(work, field.name) * count for field in dataclasses.fields(Work)
                   if field.name not in ("span", "chains")})


def _most(first: Work, second: Work) -> Work:
    return Work(**{field.name: max(getattr(first, field.name), getattr(second, field.name))
                   for field in dataclasses.fields(Work) if field.name not in ("span", "chains")})


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
