"""A register liveness pass over a Mosaic GPU kernel's lowered module (the module bytes its custom call
carries, `dew.training._lowering.KernelCall.ir`): the most 32-bit registers a thread holds live at once, over
structured control flow. Values are tracked as the 32-bit registers they occupy (atoms): an op that computes
makes new ones; one that only regroups (a cast between a warpgroup array and its threads' registers, a
bitcast, an element taken from or put into a vector or struct, an MMA accumulating in place) names its
operands' own. Split into fragment registers (float data) and scalar ones (integers, addresses).

The compile gate compares this pass with the family rules and ptxas. This module never compiles a kernel
or chooses a live run's configuration. Its estimate describes the lowered program; the compiler's
allocation and scheduling constraints are checked separately."""
import collections
import math
import re

from jax._src.interpreters import mlir as jax_mlir
from jax.extend.mlir import ir
from jaxlib.mlir import passmanager
from jaxlib.mlir.dialects import gpu, llvm
from jaxlib.mosaic.python import mosaic_gpu as mosaic_gpu_dialect

FREE = {"arith.constant", "llvm.mlir.constant", "llvm.mlir.undef", "llvm.mlir.poison", "llvm.mlir.zero",
        "gpu.thread_id", "gpu.block_id", "gpu.block_dim", "gpu.grid_dim"}
"""Ops whose results take no register a thread keeps: immediates and special registers. (A1, holding a
thread's id and the arithmetic over it, PAIRS15: sm80's forward 199 against SASS's 193 and its kernel
gradient 171 against 202, but sm90's kernel gradient 112 against 102: open, the per-thread addresses ptxas
keeps on a family without TMA against those it recomputes.)"""

PREFIX = "stable_mosaic_gpu."


def _elements(type_: ir.Type) -> list[tuple[int, bool]] | None:
    """A register value's parts in order, each its bits and whether it is floating-point data. Memrefs
    and other non-register types have no parts; their address calculations have values of their own."""
    if isinstance(type_, ir.VectorType):
        vector = ir.VectorType(type_)
        parts = _elements(vector.element_type)
        return None if parts is None else parts * math.prod(vector.shape)
    if isinstance(type_, llvm.StructType):
        structure = llvm.StructType(type_)
        if structure.opaque:
            return None
        fields = structure.body
        if not isinstance(fields, (list, tuple)):
            raise TypeError(f"LLVM struct fields are not a sequence: {fields!r}")
        parts = []
        for field in fields:
            inner = _elements(field)
            if inner is None:
                return None
            parts.extend(inner)
        return parts
    if isinstance(type_, llvm.PointerType):
        return [(32 if llvm.PointerType(type_).address_space == 3 else 64, False)]
    if isinstance(type_, ir.IntegerType):
        width = ir.IntegerType(type_).width
        return [(0 if width == 1 else width, False)]
    if isinstance(type_, ir.IndexType):
        return [(32, False)]
    if isinstance(type_, ir.FloatType):
        return [(ir.FloatType(type_).width, True)]
    return None


class Atoms:
    """The registers of every value: lists of atom ids, each atom a 32-bit register (`data` whether float)."""

    def __init__(self):
        self.of = {}
        self.data = []
        self.origin = []
        self.making = "arg"

    def fresh(self, type_, unpacked: bool = False) -> list[int]:
        """New registers for a value of `type_`: its sub-word parts packed into 32-bit registers, or, where
        the op that makes it has no packed form (`unpacked`), a register per part."""
        parts = _elements(type_)
        if unpacked and parts is not None:
            parts = [(32 if size else 0, data) for size, data in parts]
        if parts is None:
            return []
        atoms, bits = [], 32
        for size, data in parts:
            for _ in range(max(1, -(-size // 32)) if size else 0):
                atoms.append(len(self.data))
                self.data.append(data)
                self.origin.append(self.making)
            if size and size < 32:  # sub-word parts pack into the previous register while they fit
                if bits + size <= 32 and len(atoms) > 1:
                    self.data.pop()
                    self.origin.pop()
                    atoms.pop()
                    bits += size
                else:
                    bits = size
        return atoms

    def element_atoms(self, value, index: int) -> list[int]:
        """The atoms of part `index` of `value` (an element of a vector, a field of a struct)."""
        parts = _elements(value.type)
        atoms = self.of.get(value, [])
        if parts is None or not atoms:
            return []
        # Map parts to atoms by bit offset.
        offset = sum(size for size, _ in parts[:index])
        size = parts[index][0]
        first, last = offset // 32, max(offset // 32, (offset + size - 1) // 32)
        return atoms[first:last + 1] if last < len(atoms) else []


def _position(op, source) -> int | None:
    """The flat index of the part an extract or insert names statically in `source` (row-major over a
    vector's shape, a struct's fields); None where it is dynamic or unreadable."""
    for name in ("static_position", "position"):
        if name in op.attributes:
            attribute = op.attributes[name]
            position = (list(ir.DenseI64ArrayAttr(attribute)) if isinstance(attribute, ir.DenseI64ArrayAttr)
                        else [ir.IntegerAttr(value).value for value in ir.ArrayAttr(attribute)])
            if not position or any(p < 0 for p in position):
                return None
            vector = isinstance(source.type, ir.VectorType)
            dims = ir.VectorType(source.type).shape if vector else [len(position)]
            flat = 0
            for i, p in enumerate(position):
                inner = 1
                for d in dims[i + 1:]:
                    inner *= d
                flat += p * inner
            return flat if vector or len(position) == 1 else None
    return None


def _constant(value) -> int | None:
    """An integer constant's value, where `value` is one."""
    owner = value.owner
    if isinstance(owner, ir.Block) or owner.operation.name.removeprefix(PREFIX) not in ("arith.constant",
                                                                             "llvm.mlir.constant"):
        return None
    attribute = owner.operation.attributes["value"]
    return ir.IntegerAttr(attribute).value if isinstance(attribute, ir.IntegerAttr) else None


PURE = re.compile(r"(arith|llvm)\.(add|sub|mul|div|rem|shl|shr|and|or|xor|index_cast|ext|trunc|select|cmp"
                  r"|max|min|ptrtoint|inttoptr|getelementptr|zext|sext)")
"""Scalar arithmetic a compiler recomputes where its operands are free (address and index arithmetic)."""


ELEMENTWISE = re.compile(r"(arith\.(add|sub|mul|div|neg|max|min|ext|trunc|select|cmp|fma|bitcast|si|ui|fp)"
                         r"|math\.|llvm\.(fadd|fsub|fmul|fdiv|fneg|fma|bitcast|fpext|fptrunc|select)"
                         r"|llvm\.inline_asm$)")
"""Ops each element of whose result reads only the same element of its operands."""

UNPACKED = re.compile(r"arith\.(select|cmpf|cmpi|maxnumf|minnumf|maximumf|minimumf)$")
"""Ops SASS has no packed 16-bit form of (it has HADD2, HMUL2 and HFMA2, but selects, compares and
min/max of bf16 pairs run a lane at a time): their 16-bit results take a register per element (R3)."""

STREAMS_INTO = re.compile(r"(memref\.store|vector\.store|llvm\.store|nvvm\.stmatrix"
                          r"|vector\.multi_reduction|vector\.reduction|vector\.contract)")
"""Ops that consume a value an element (a register) at a time, so a chain feeding them streams: stores,
and reductions, whose running sums take each element as it comes (R4: a softmax's row sum of its weights
beside their narrowing for the next product)."""


def _within(op, ancestor) -> bool:
    """Whether `op` is `ancestor`'s child or nested in one of its children's regions (R5: a chain consumed in
    a branch or loop of its own block, an index vector a mask compares, streams as one consumed beside it)."""
    while op is not None:
        if op.parent == ancestor:
            return True
        op = op.parent
    return False


def _tied(op) -> list | None:
    """An inline assembly's inputs its outputs are, in output order, where every output is tied to one (an
    MMA accumulating in place); None otherwise."""
    if "constraints" not in op.attributes:
        return None
    constraints = [part.strip() for part in ir.StringAttr(op.attributes["constraints"]).value.split(",")
                   if part.strip()]
    outputs = sum(1 for c in constraints if c.startswith("="))
    if not outputs:
        return None
    inputs = [constraint for constraint in constraints[outputs:] if not constraint.startswith("~")]
    if len(inputs) != len(op.operands):
        # Indirect or read/write constraints are not explicit output-to-input ties. Keep distinct
        # result registers unless every output can be associated with an operand.
        return None
    tied = {int(c): operand for operand, c in zip(op.operands, inputs, strict=True) if c.isdigit()}
    return [tied[i] for i in range(outputs)] if outputs and set(tied) == set(range(outputs)) else None


class Peak:
    def __init__(self):
        self.total, self.fragments, self.scalars, self.where, self.live = 0, 0, 0, "", ()
        self.regions = {}
        self.threads: int | None = None
        self.mma_words = 0

    def see(self, live: set, data: list, where: str, region: str = ""):
        if len(live) > self.regions.get(region, (0,))[0]:
            fragments = sum(1 for atom in live if data[atom])
            self.regions[region] = (len(live), fragments, where)
        if len(live) > self.total:
            fragments = sum(1 for atom in live if data[atom])
            self.total, self.fragments = len(live), fragments
            self.scalars, self.where = len(live) - fragments, where
            self.live = tuple(live)

    def breakdown(self, atoms) -> list:
        return collections.Counter((atoms.origin[a], "F" if atoms.data[a] else "S") for a in self.live
                                   ).most_common(14)


DIVERGENT = re.compile(r"(gpu\.thread_id|gpu\.lane_id|nvvm\.read\.ptx\.sreg\.(tid|laneid)|llvm\.inline_asm"
                       r"|nvvm\.(shfl|ldmatrix|elect|vote)|vector\.|arith\.(cmpf|.*f$))")
"""Ops whose scalar results may differ between a warp's threads: thread and lane ids, shuffles, matrix loads,
inline assembly, anything over fragments."""


def divergent_values(module) -> set:
    """Every value whose threads may disagree within a warp (a fixpoint over the module: a divergent source,
    or an op or a loop carry fed by a divergent value). The rest are warp-uniform: ptxas keeps a uniform
    scalar in the uniform register file (sm90's 63 a warp, `UR`), apart from the registers a thread counts."""
    divergent = set()

    def visit(op) -> bool:
        changed = False
        name = op.name.removeprefix(PREFIX)
        tainted = DIVERGENT.match(name) or any(o in divergent for o in op.operands)
        if tainted:
            for result in op.results:
                if result not in divergent:
                    divergent.add(result)
                    changed = True
        for nested_region in op.regions:
            for block in nested_region.blocks:
                # A region's carried arguments are divergent where the values carried into them are.
                if tainted:
                    for argument in block.arguments:
                        if argument not in divergent:
                            divergent.add(argument)
                            changed = True
                for inner in block.operations:
                    inner = inner.operation
                    if inner.name.removeprefix(PREFIX) in ("scf.yield", "scf.condition") and any(
                            o in divergent for o in inner.operands):
                        for value in (*op.results, *block.arguments):
                            if value not in divergent:
                                divergent.add(value)
                                changed = True
                    changed |= visit(inner)
        return changed

    changed = True
    while changed:
        changed = False
        for op in module.body.operations:
            changed |= visit(op.operation)
    return divergent


class _Pass:
    def __init__(self, divergent: set | None = None):
        self.atoms = Atoms()
        self.free = set()
        self.divergent = divergent
        self.launch_threads: set[int] = set()
        self.mma_words = 0

    @staticmethod
    def _streamed(op, name: str) -> bool:
        if not ELEMENTWISE.match(name) or len(op.results) != 1:
            return False
        if name == "llvm.inline_asm" and (_tied(op) is not None or "has_side_effects" in op.attributes
                                          or not re.fullmatch(r"(vector<\d+x)?f(16|32)>?|bf16",
                                                              str(op.results[0].type))):
            # Only a pure one-result float assembly is an elementwise math op (ex2.approx, rcp.approx).
            return False
        parent = op.parent
        for use in op.results[0].uses:
            user = use.owner.operation
            user_name = user.name.removeprefix(PREFIX)
            if not _within(user, parent) or not (ELEMENTWISE.match(user_name) or STREAMS_INTO.match(user_name)
                                                 or user_name in ("builtin.unrealized_conversion_cast",
                                                                  "vector.bitcast", "llvm.bitcast")):
                return False
        return True

    def _accumulated(self, op) -> list | None:
        """An mma.sync's D registers where its C input dies at it (C2: ptxas writes D over C, `HMMA R88, ...,
        R88`, though the assembly's constraints tie neither): C's registers, else None. C is the last
        operands, one a part of D's struct, each read by this op alone."""
        if "asm_string" not in op.attributes:
            return None
        if "mma.sync" not in ir.StringAttr(op.attributes["asm_string"]).value:
            return None
        if len(op.results) != 1:
            return None
        count = len(_elements(op.results[0].type) or [])
        operands = list(op.operands)
        if not count or len(operands) < count:
            return None
        inputs = operands[-count:]
        if any(len(list(value.uses)) != 1 for value in inputs):
            return None
        atoms = [a for value in inputs for a in self.atoms.of.get(value, ()) if a is not None]
        return atoms if len(atoms) == count else None

    @staticmethod
    def _lanes(type_) -> list:
        """A value's registers, none of them held yet (lanes an insert fills): placeholders."""
        parts = _elements(type_) or []
        return [None] * -(-sum(size for size, _ in parts) // 32)

    def _rematerialized(self, op, name: str) -> bool:
        """Whether `op` is scalar arithmetic over free values only (immediates, special registers and what
        they compute), which the compiler recomputes at its uses rather than keeps."""
        if not op.results:
            return False
        if name in ("llvm.insertelement", "vector.insert", "vector.insertelement", "llvm.insertvalue",
                    "vector.broadcast", "vector.splat", "vector.from_elements", "llvm.bitcast",
                    "vector.bitcast"):
            # A vector or struct assembled from immediates only: rematerialized.
            return all(operand in self.free for operand in op.operands)
        if not PURE.match(name):
            return False
        if any((_elements(r.type) or [(0, True)])[0][1] for r in op.results):
            return False
        return all(operand in self.free for operand in op.operands)

    def define(self, op):
        """The atoms of `op`'s results: its operands' where it only regroups them, else new ones."""
        name = op.name.removeprefix(PREFIX)
        if name == "llvm.inline_asm" and "asm_string" in op.attributes:
            assembly = ir.StringAttr(op.attributes["asm_string"]).value
            if "wgmma.mma_async" in assembly:
                words = sum(-(-size // 32) for result in op.results
                            for size, _ in _elements(result.type) or [])
                self.mma_words = max(self.mma_words, words)
        if isinstance(op.opview, gpu.LaunchOp):
            launch = op.opview
            dimensions = tuple(_constant(size) for size in (launch.blockSizeX, launch.blockSizeY,
                                                             launch.blockSizeZ))
            if any(size is None for size in dimensions):
                raise ValueError("a Mosaic kernel's thread dimensions are not static")
            self.launch_threads.add(math.prod(size for size in dimensions if size is not None))
        self.atoms.making = f"{name}:{str(op.results[0].type)[:28]}" if len(op.results) else name
        results = list(op.results)
        if name in FREE or self._rematerialized(op, name):
            for result in results:
                self.atoms.of[result] = []
                self.free.add(result)
            return
        if name == "llvm.extractelement" and len(results) == 1:
            index = _constant(op.operands[1])
            parts = _elements(op.operands[0].type)
            if index is not None and parts is not None and index < len(parts):
                self.atoms.of[results[0]] = self.atoms.element_atoms(op.operands[0], index)
                return
        if self._streamed(op, name):
            # An elementwise result whose uses are elementwise in its block (or stores): the compiler
            # computes it an element at a time beside its consumers, so its operands stay live in its place.
            self.atoms.of[results[0]] = [a for o in op.operands for a in self.atoms.of.get(o, ())
                                       if a is not None]
            return
        if name == "llvm.insertelement" and len(results) == 1:
            index = _constant(op.operands[2])
            parts = _elements(results[0].type)
            if index is not None and parts is not None and index < len(parts):
                base = list(self.atoms.of.get(op.operands[0], [])) or self._lanes(results[0].type)
                first = sum(size for size, _ in parts[:index]) // 32
                new = self.atoms.of.get(op.operands[1], [])
                base[first:first + len(new)] = new
                self.atoms.of[results[0]] = base
                return
        if self._regrouped(op, name, results):
            return
        unpacked = bool(UNPACKED.match(name))
        for result in results:
            parts = _elements(result.type) or []
            uniform = (self.divergent is not None and result not in self.divergent
                       and not any(data for _, data in parts) and not isinstance(result.type, ir.VectorType))
            self.atoms.of[result] = [] if uniform else self.atoms.fresh(result.type, unpacked=unpacked)

    def _regrouped(self, op, name: str, results) -> bool:
        """Whether this operation only names existing registers in another grouping or order."""
        operand_atoms = [self.atoms.of.get(operand, []) for operand in op.operands]
        if name in ("builtin.unrealized_conversion_cast", "vector.bitcast", "llvm.bitcast"):
            flat = [a for atoms in operand_atoms for a in atoms]
            if len(results) == 1:
                self.atoms.of[results[0]] = flat
                return True
            share = len(flat) // max(1, len(results))
            for i, result in enumerate(results):
                self.atoms.of[result] = flat[i * share:(i + 1) * share]
            return True
        if name in ("vector.extract", "llvm.extractvalue", "vector.extractelement") and len(results) == 1:
            index = _position(op, op.operands[0])
            parts = _elements(op.operands[0].type)
            if index is not None and parts is not None and index < len(parts):
                self.atoms.of[results[0]] = self.atoms.element_atoms(op.operands[0], index)
                return True
        if name in ("vector.insert", "llvm.insertvalue", "vector.insertelement") and len(results) == 1:
            parts = _elements(results[0].type)
            containers = [o for o in op.operands if o.type == results[0].type]
            index = _position(op, containers[0]) if containers else None
            if index is not None and parts is not None and index < len(parts):
                container = containers[0]
                element = next(o for o in op.operands
                               if o != container and not isinstance(o.type, ir.IndexType))
                base = list(self.atoms.of.get(container, [])) or self._lanes(results[0].type)
                first = sum(size for size, _ in parts[:index]) // 32
                new = self.atoms.of.get(element, [])
                base[first:first + len(new)] = new
                self.atoms.of[results[0]] = base
                return True
        if name in ("vector.from_elements", "vector.interleave") and len(results) == 1:
            # A vector assembled from its operands' registers (in order, or interleaved lane by lane).
            if name == "vector.interleave" and len(operand_atoms) == 2:
                first, second = operand_atoms
                self.atoms.of[results[0]] = [a for pair in zip(first, second, strict=False) for a in pair] + \
                    first[len(second):] + second[len(first):]
            else:
                self.atoms.of[results[0]] = [a for atoms in operand_atoms for a in atoms]
            return True
        if name == "vector.extract_strided_slice" and len(results) == 1 and operand_atoms:
            # A 1-D slice of a vector's registers (`offsets`, `sizes`, `strides` by element).
            positions = [ir.ArrayAttr(op.attributes[key]) for key in ("offsets", "sizes", "strides")]
            parts = _elements(op.operands[0].type)
            one_dimension = all(len(position) == 1 for position in positions)
            if one_dimension and parts and all(bits == 32 for bits, _ in parts):
                offset, size, stride = (ir.IntegerAttr(position[0]).value for position in positions)
                self.atoms.of[results[0]] = operand_atoms[0][offset:offset + size * stride:stride]
                return True
        if name == "vector.deinterleave" and len(results) == 2 and operand_atoms:
            whole = operand_atoms[0]
            self.atoms.of[results[0]], self.atoms.of[results[1]] = whole[0::2], whole[1::2]
            return True
        if name == "llvm.getelementptr" and len(results) == 1 and operand_atoms and all(
                _constant(o) is not None or o.owner.operation.name.removeprefix(PREFIX) in FREE
                for o in list(op.operands)[1:] if not isinstance(o.owner, ir.Block)) and all(
                not isinstance(o.owner, ir.Block) for o in list(op.operands)[1:]):
            # A constant offset from a held base folds into the access's address field (LDS, STS, LDSM,
            # STSM and LDG take a register plus an immediate): the base's registers.
            self.atoms.of[results[0]] = list(operand_atoms[0])
            return True
        if name in ("vector.broadcast", "vector.splat") and len(results) == 1:
            self.atoms.of[results[0]] = list(operand_atoms[0]) if operand_atoms else []
            return True
        if name == "llvm.inline_asm":
            tied = _tied(op)
            if tied is not None and len(results) == 1:
                self.atoms.of[results[0]] = [a for operand in tied for a in self.atoms.of.get(operand, [])]
                return True
            accumulated = self._accumulated(op)
            if accumulated is not None:
                self.atoms.of[results[0]] = accumulated
                return True
        return False

    def uses(self, op, into: set):
        extracts = ("vector.extract", "vector.extractelement", "llvm.extractvalue", "llvm.extractelement",
                    "vector.extract_strided_slice")
        if op.name.removeprefix(PREFIX) in extracts and all(result in self.atoms.of for result in op.results):
            extracted = {a for result in op.results for a in self.atoms.of[result] if a is not None}
            inputs = {a for operand in op.operands for a in self.atoms.of.get(operand, ()) if a is not None}
            if extracted <= inputs:
                # A static extraction reads only its part. Reading the whole source here extends
                # every accumulator lane through the last scalar conversion in its epilogue.
                into.update(extracted)
                return
        for operand in op.operands:
            into.update(a for a in self.atoms.of.get(operand, ()) if a is not None)
        for nested_region in op.regions:
            for block in nested_region.blocks:
                for inner in block.operations:
                    self.uses(inner.operation, into)

    def block(self, block, outer: set, peak: Peak, depth: int, region: str = "", preserved=None):
        """The block's ops in order. `preserved` survives the block; other outer atoms are captured values
        whose last use in this block determines when they die."""
        owner = block.owner.operation.name.removeprefix(PREFIX)
        self.atoms.making = f"arg:{owner}"
        for argument in block.arguments:
            uniform = (self.divergent is not None and argument not in self.divergent
                       and not isinstance(argument.type, ir.VectorType)
                       and not any(data for _, data in _elements(argument.type) or []))
            self.atoms.of[argument] = [] if uniform else self.atoms.fresh(argument.type)
        ops = [op.operation for op in block.operations]
        # Define every result first (regions' results included), so uses map to atoms.
        for op in ops:
            self._define_all(op)
        used_by, last = [], {}
        for index, op in enumerate(ops):
            used = set()
            self.uses(op, used)
            used_by.append(used)
            for atom in used:
                last[atom] = index
        preserved = outer if preserved is None else preserved
        local = {a for arg in block.arguments for a in self.atoms.of[arg] if a is not None and a in last}
        local |= {a for a in outer - preserved if a in last}
        outer = preserved
        for index, op in enumerate(ops):
            made = {a for r in op.results for a in self.atoms.of.get(r, ()) if a is not None}
            # A region op's results are born where its operands it consumes have died (a loop's carries
            # leave it as its results): they never coexist.
            edge = index + bool(op.regions)
            live = outer | {a for a in local if last[a] >= edge} | made
            if made and ELEMENTWISE.match(op.name.removeprefix(PREFIX)) and not op.regions:
                # R6: an ALU instruction writes its destination over a source that dies at it, so an
                # elementwise op's result takes the registers of an operand last read there (an output
                # rescaled in place before the product that accumulates into it).
                spent = sorted(a for a in used_by[index] & local if last[a] == index and a not in outer)
                live -= set(spent[:len(made)])
            peak.see(live, self.atoms.data, f"{depth}:{op.name.removeprefix(PREFIX)}", region)
            if op.regions:
                # Live through the regions: what lives past the op and what its regions read (C1: a loop's
                # initial carries are copied into its iteration arguments, so they die at the loop unless
                # its body or a later op reads them).
                survives = outer | {a for a in local if last[a] > index}
                for region_index, nested_region in enumerate(op.regions):
                    for inner in nested_region.blocks:
                        captured = set()
                        for nested in inner.operations:
                            self.uses(nested.operation, captured)
                        across = survives | (captured & local)
                        self.block(inner, across, peak, depth + 1,
                                   f"{region}/{index}:{region_index}", preserved=survives)
            local |= {a for a in made if a in last}
            local = {a for a in local if last[a] > index}

    def _define_all(self, op):
        if op.results and op.results[0] in self.atoms.of:
            return
        self.define(op)


CANONICAL = "builtin.module(canonicalize,cse,loop-invariant-code-motion,canonicalize,cse)"
"""The scalar clean-up a kernel's module takes before the pass (LLVM's own folds, value numbering and
loop-invariant motion run on it before ptxas): constants folded, duplicates merged, invariants hoisted."""


def canonical(module_bytes: bytes, context: ir.Context) -> ir.Module:
    """Decode Mosaic's versioned bytecode through its own serde pass, then simplify scalar expressions
    before counting values. Parsing or verification errors remain gate failures."""
    mosaic_gpu_dialect.register_dialect(context)
    context.allow_unregistered_dialects = True
    module = ir.Module.parse(module_bytes)
    passmanager.PassManager.parse("builtin.module(mosaic_gpu-serde{serialize=false})").run(module.operation)
    passmanager.PassManager.parse(CANONICAL).run(module.operation)
    return module


def kernel_peaks(module_bytes: bytes) -> list[tuple[str, Peak, tuple]]:
    """Each function's peak in a lowered module (its kernel's body among them), the module canonicalized
    first (`canonical`)."""
    found = []
    with jax_mlir.make_ir_context() as context:
        module = canonical(module_bytes, context)
        divergent = divergent_values(module)
        for op in module.body.operations:
            if not op.operation.regions:
                continue
            peak, run = Peak(), _Pass(divergent)
            for nested_region in op.operation.regions:
                for block in nested_region.blocks:
                    run.block(block, set(), peak, 0)
            if len(run.launch_threads) == 1:
                peak.threads = next(iter(run.launch_threads))
            peak.mma_words = run.mma_words
            name = (ir.StringAttr(op.operation.attributes["sym_name"]).value
                    if "sym_name" in op.operation.attributes else op.operation.name)
            regions = sorted(peak.regions.items(), key=lambda kv: -kv[1][0])[:8]
            found.append((str(name), peak, (peak.breakdown(run.atoms), regions)))
    return found
