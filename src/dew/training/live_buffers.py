# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Dew: ported from jax/_src/peak_live.py, read as Dew's own
# (the JAX fork's audit of 2026-11-26, its C01/C02).
"""The buffers a compile holds: those live at the peak of each of its allocations (`peak_live`), and each over
the part of a heap simulation's schedule it lives in (`live_intervals`), each with what made it, the
`checkpoint_name` its value carries and its dimensions' provenance among the call's arguments.

XLA's buffer assignment names them (`BufferAssignmentProto`, its `peak_buffers` and `heap_simulator_traces`);
the compile states its assignment in its memory analysis (XLA:CPU's, and the CUDA plugin's, which serves its
executables' metadata), read here through XLA's own schema (`dew.telemetry.hlo`) beside the compiled module
its buffers are defined in, decoded once (`Assignment`). What a buffer is, its name and its provenance, is
Dew's reading of that dataflow: a memory planner's policy (`dew.training.memory.LiveSet`), which JAX states
none of.
"""

from __future__ import annotations

import dataclasses
import contextlib
import json
from collections.abc import Callable
from typing import Any

import jax

from dew.telemetry import hlo


@dataclasses.dataclass(frozen=True)
class LiveBuffer:
    """A buffer live at the peak of its allocation.

    `bytes` is its size in the buffer assignment, as XLA allocates it; `shape` its dimensions as compiled (a
    device's, after partitioning); `dtype` its element type's XLA name (`f32`, `bf16`); `kind` its
    allocation's: `parameter` (an entry parameter's, aliased with an output or not), `output` or `temporary`.
    `instruction` names the instruction defining it, `op_name` that instruction's (JAX's name stack) and
    `source` its `file:line`. `names` are the `jax.ad_checkpoint.checkpoint_name` names of the value that
    instruction defines where it carries them (a fusion's, its maker's, `_maker`); for a while loop's state
    element an unnamed instruction's buffer holds (XLA's buffer aliases), the names of what the body writes
    into the whole of that element: a scan's stacked output, the residual each step writes a row of
    (`_loop_named`). `parameter` is the entry parameter's number, for a `parameter` buffer. `role` is what it
    is of the instruction making it beyond its value: `workspace` for a cuBLAS call's scratch (`_role`), which
    a recomputation frees none of; empty otherwise. `dims` holds, for each of its dimensions, the (argument,
    axis) pairs it descends from, each argument by its flat index among the call's (`args_info`'s leaves),
    through the dimension maps of the ops that made it (`provenance`): empty where its size descends from
    none, None where an op on its path states no map (a custom call stating no `output_dims`).
    """

    bytes: int
    shape: tuple[int, ...]
    dtype: str
    kind: str
    instruction: str
    op_name: str
    source: str
    names: tuple[str, ...] = ()
    parameter: int | None = None
    role: str = ""
    dims: tuple[frozenset[tuple[int, int]] | None, ...] = ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class LiveInterval(LiveBuffer):
    """A buffer of a compile's allocation over the part of a heap simulation's schedule it lives in.

    Besides `LiveBuffer`'s: `allocation`, the buffer allocation it is assigned to; `trace`, the heap
    simulation that orders it (its index among the assignment's `heap_simulator_traces`): positions compare
    within one trace alone (an allocation simulated per computation has one each, their positions each from 0;
    the pages of one heap each a copy of its one trace, their positions alike). `whole` says whether that
    simulation's schedule is the whole module's (XLA:GPU's), a while body's buffers within its loop's span, or
    one computation's. `complete` says whether every buffer its allocation is assigned lies in one of its
    simulations: XLA places some in an allocation unsimulated (temporary heaps reused, or combined), so a
    planner needing all of an allocation's buffers over one schedule reads it first. `start` is its first
    position among that simulation's events, `end` the position after its last (the simulation's length for
    one never freed); `buffer` its logical buffer's id; `storage` the logical buffer whose memory it lies in,
    as first allocated: its own where it was allocated, else the one the buffer it shares lies in (XLA's
    `SHARE_WITH`: an in-place output over its operand, or memory reused once freed). The memory a trace
    occupies at a position is each storage some interval covering it lies in, once, at the most of their
    `bytes`. `charged` says whether XLA's count of the allocation's peak charges it, from its start to its end
    (`_charged`): in a complete allocation's one trace, the charged intervals covering its peak position are
    `peak_live`'s buffers of it. `bytes` is its slice of the allocation, as the assignment states it.
    """

    allocation: int
    trace: int
    whole: bool
    complete: bool
    start: int
    end: int
    buffer: int
    storage: int
    charged: bool


def _stated(executable: jax.stages.Compiled) -> tuple[bytes | None, Callable[[int], int | None] | None]:
    """`executable`'s serialized buffer assignment, None where the backend states none, and the call's
    argument each of its module's entry parameters holds (`Assignment`'s `argument`).

    JAX 0.12.0.1.post142's surface, its memory analysis's field and its compiled stage's own map; 0.12.2.post1
    states them as `Compiled.buffer_assignment()` and `Compiled.entry_arguments()`, and this reads those in
    the head that moves the pin."""
    analysis = executable.memory_analysis()
    assignment = None if analysis is None else analysis.serialized_buffer_assignment_proto or None
    return assignment, executable._parameter_arguments()


def assignment_of(executable: jax.stages.Compiled) -> Assignment | None:
    """`executable`'s buffer assignment read beside its compiled module, each decoded once; None where the
    backend states no assignment (a plugin serving no executable metadata)."""
    assignment, argument = _stated(executable)
    runtime = executable.runtime_executable()
    if assignment is None or runtime is None:
        return None
    (module,) = runtime.hlo_modules()
    return Assignment(assignment, module.as_serialized_hlo_module_proto(), argument)


def peak_live(executable: jax.stages.Compiled) -> tuple[LiveBuffer, ...] | None:
    """The buffers live at the peak of each of `executable`'s allocations (`Assignment.live`), whether its
    compile ran whole or stopped after buffer assignment (`xla_gpu_experimental_early_exit`); None where the
    backend states no assignment."""
    read = assignment_of(executable)
    return None if read is None else read.live()


def live_intervals(executable: jax.stages.Compiled) -> tuple[LiveInterval, ...] | None:
    """Each buffer of `executable`'s allocations over the part of a heap simulation's schedule it lives in
    (`Assignment.intervals`); None where the backend states no assignment."""
    read = assignment_of(executable)
    return None if read is None else read.intervals()


class Assignment:
    """A serialized `BufferAssignmentProto` read beside the serialized `HloModuleProto` whose buffers it
    assigns, each decoded once; `argument` maps an entry parameter's number to the call's argument it holds
    (None for one that holds none: a closed-over constant, an effect's token, a pruned argument's absence
    shifting the rest), the identity where it is None."""

    def __init__(self, assignment: bytes, module: bytes, argument: Callable[[int], int | None] | None = None):
        self.proto: Any = hlo.message_class("xla.BufferAssignmentProto").FromString(assignment)
        compiled: Any = hlo.message_class("xla.HloModuleProto").FromString(module)
        self.instructions = {instruction.id: instruction for computation in compiled.computations
                             for instruction in computation.instructions}
        self.computations = {computation.id: computation for computation in compiled.computations}
        self.descended = provenance(compiled, argument)
        # Each logical buffer's other positions, as XLA's must-alias analysis states them: a loop state's
        # buffer's among them, at the while's output.
        self.aliases: dict[int, list[Any]] = {}
        for alias in self.proto.buffer_aliases:
            self.aliases.setdefault(alias.source_buffer_id, []).append(alias.location)
        self.types = hlo.pool().FindEnumTypeByName("xla.PrimitiveType")
        self.buffers = {buffer.id: buffer for buffer in self.proto.logical_buffers}
        self.kinds = {allocation.index: _kind(allocation) for allocation in self.proto.buffer_allocations}
        self.parameters = {allocation.index: allocation.parameter_number
                           for allocation in self.proto.buffer_allocations
                           if allocation.is_entry_computation_parameter}

    def live(self) -> tuple[LiveBuffer, ...]:
        """The array buffers live at the peak of each allocation; a constant's and a thread-local
        allocation's are none."""
        return tuple(LiveBuffer(**found) for peak in self.proto.peak_buffers
                     for buffer in peak.logical_buffer_ids
                     if (found := self.described(buffer, peak.index)) is not None)

    def intervals(self) -> tuple[LiveInterval, ...]:
        """Each array buffer over the positions of a heap simulation it lives in (its
        `heap_simulator_traces`' events: allocated at one, or placed in another's memory, and freed at
        another), as `live` describes it, for each allocation it is assigned to whose simulation that is; a
        constant's and a thread-local allocation's are none, as are those of an allocation the assignment
        states no simulation of (XLA:CPU keeps none). In a complete allocation's one trace, the charged
        intervals covering its peak are the buffers `live` names; `complete` is the union of its traces, no
        clock across them."""
        event = hlo.pool().FindMessageTypeByName("xla.HeapSimulatorTrace.Event")
        kinds = {value.name: value.number for value in event.fields_by_name["kind"].enum_type.values}
        # A heap's pages each hold a copy of its one trace, of their own buffers, each its slice's size (as
        # XLA's own count of the peak reads it).
        assigned = {allocation.index: {each.logical_buffer_id: each.size for each in allocation.assigned}
                    for allocation in self.proto.buffer_allocations}
        traced: dict[int, set[int]] = {}
        for simulated in self.proto.heap_simulator_traces:
            traced.setdefault(simulated.buffer_allocation_index, set()).update(
                step.buffer_id for step in simulated.events)
        complete = {allocation: set(sizes) <= traced.get(allocation, set())
                    for allocation, sizes in assigned.items()}
        found = []
        for trace, simulated in enumerate(self.proto.heap_simulator_traces):
            # Each buffer's storage, as first allocated: a share names the buffer it shares, itself perhaps
            # sharing another's, freed or not.
            storages: dict[int, int] = {}
            opened: dict[int, int] = {}
            spans = []
            for position, step in enumerate(simulated.events):
                if step.kind == kinds["FREE"]:
                    spans.append((step.buffer_id, opened.pop(step.buffer_id), position))
                    continue
                shared = step.share_with_canonical_id
                sharing = step.kind == kinds["SHARE_WITH"]
                storages[step.buffer_id] = (int(storages.get(shared, shared)) if sharing
                                            else int(step.buffer_id))
                opened[step.buffer_id] = position
            spans += [(buffer, start, len(simulated.events)) for buffer, start in opened.items()]
            allocation = simulated.buffer_allocation_index
            sizes = assigned.get(allocation, {})
            charged = _charged(simulated.events, sizes, kinds)
            found += [LiveInterval(**{**described, "bytes": sizes[buffer]}, allocation=allocation,
                                   trace=trace, whole=simulated.whole_module_simulation,
                                   complete=complete[allocation], start=start, end=end, buffer=buffer,
                                   storage=storages[buffer], charged=buffer in charged)
                      for buffer, start, end in spans
                      if buffer in sizes and (described := self.described(buffer, allocation)) is not None]
        return tuple(found)

    def described(self, buffer_id: int, allocation: int) -> dict | None:
        """`LiveBuffer`'s fields of logical buffer `buffer_id` of `allocation`; None for a constant's or a
        thread-local allocation's, and for a tuple's index table, no array."""
        if (kind := self.kinds.get(allocation)) is None:
            return None
        buffer = self.buffers[buffer_id]
        instruction = self.instructions[buffer.defined_at.instruction_id]
        index = tuple(buffer.defined_at.shape_index)
        shape, dims = instruction.shape, self.descended.get(instruction.id)
        for position in index:
            shape = shape.tuple_shapes[position]
            dims = None if dims is None else dims[position]
        if shape.tuple_shapes:
            return None
        metadata = instruction.metadata
        named = instruction.frontend_attributes.map.get("checkpoint_name")
        written = None if named else _aliased_named(self.aliases.get(buffer.id, ()), self.instructions,
                                                     self.computations)
        if written is not None:
            named = written.frontend_attributes.map.get("checkpoint_name")
        made, within, getter = _maker(instruction, index, self.instructions, self.computations)
        if not named and getter is not None:
            named = getter.frontend_attributes.map.get("checkpoint_name")
        if not named and _names_output(made, within):
            named = made.frontend_attributes.map.get("checkpoint_name")
        op_name = (metadata.op_name or (written.metadata.op_name if written is not None else "")
                   or made.metadata.op_name)
        return {"bytes": buffer.size, "shape": tuple(shape.dimensions),
                "dtype": self.types.values_by_number[shape.element_type].name.lower(), "kind": kind,
                "instruction": instruction.name, "op_name": op_name,
                "source": f"{metadata.source_file}:{metadata.source_line}",
                "names": tuple(named.split(",")) if named else (),
                "parameter": self.parameters.get(allocation),
                "role": _role(made, within),
                "dims": tuple(None for _ in shape.dimensions) if dims is None else tuple(dims)}


def _charged(events, assigned: dict[int, int], kinds: dict[str, int]) -> set[int]:
    """The buffers of `assigned` (an allocation's, by id) XLA's count of its peak charges over a heap
    simulation's `events`, as it counts them (`ComputePeakMemoryLogicalBuffers`): each allocated one; each
    share whose canonical buffer has no other live instance then, instances counted by the canonical buffer
    it names, each freed one by the buffer it shared; one of no bytes never, as XLA's count names no buffer
    it adds nothing for."""
    instances: dict[int, int] = {}
    canonical: dict[int, int] = {}
    charged = set()
    for step in events:
        if step.buffer_id not in assigned:
            continue  # Another page's, which this allocation's count skips.
        if step.kind == kinds["ALLOC"]:
            instances[step.buffer_id] = 1
            if assigned[step.buffer_id] > 0:
                charged.add(step.buffer_id)
        elif step.kind == kinds["SHARE_WITH"]:
            shared = canonical[step.buffer_id] = int(step.share_with_canonical_id)
            instances[shared] = instances.get(shared, 0) + 1
            if instances[shared] == 1 and assigned[step.buffer_id] > 0:
                charged.add(step.buffer_id)
        else:
            freed = int(canonical.get(step.buffer_id, step.buffer_id))
            instances[freed] = instances.get(freed, 0) - 1
    return charged


def _maker(instruction, index: tuple[int, ...], instructions,
           computations):
    """What makes `instruction`'s output at shape `index`, and where in that maker's own output it is: inside
    a fusion, through its fused computation's tuples and get-tuple-elements to the instruction making it,
    the shape index carried along (XLA:GPU's own fusions carry no op_name of their own, the instruction they
    fuse does: a gemm's fusion, its cuBLAS call and the workspace it holds; a dynamic-slice fusion's root,
    `tuple(updated, get-tuple-element(cublas_call, workspace))`, XLA's
    `dynamic_slice_fusion_rewriter_v2.cc` `CreateFusionBody`); `instruction` itself at `index` for any
    other. Beside them, the nearest get-tuple-element the walk passed that names the value it selects whole
    (a `checkpoint_name` on it, its result the output itself), whose name is that value's own where its
    maker's names its tuple (`_names_output`), or None."""
    if instruction.opcode != "fusion":
        return instruction, index, None
    made = instructions[computations[instruction.called_computation_ids[0]].root_id]
    getter = None
    while True:
        if made.opcode == "tuple" and index:
            made, index = instructions[made.operand_ids[index[0]]], index[1:]
        elif made.opcode == "get-tuple-element":
            if getter is None and not index and made.frontend_attributes.map.get("checkpoint_name"):
                getter = made
            made, index = instructions[made.operand_ids[0]], (made.tuple_index, *index)
        else:
            return made, index, getter


def _cublas(maker) -> bool:
    return maker.opcode == "custom-call" and maker.custom_call_target.startswith("__cublas$")


def _names_output(maker, index: tuple[int, ...]) -> bool:
    """Whether `maker`'s checkpoint name names its output at `index`: its whole output, or a cuBLAS call's
    result, its tuple's first element (XLA's gemm rewriter puts an auxiliary output and the workspace after
    it). Any other element of a tuple an instruction makes is named by none of it: a cuBLAS call's 32 MiB
    workspace is no checkpointed value."""
    return not index or (index == (0,) and _cublas(maker))


def _role(maker, index: tuple[int, ...]) -> str:
    """What the element `index` of `maker`'s output is beyond its value: `workspace` for a cuBLAS call's last
    tuple element past its first, the scratch XLA's gemm rewriter appends after its result (and any
    auxiliary output); "" for any other."""
    last = len(maker.shape.tuple_shapes) - 1
    return "workspace" if _cublas(maker) and last > 0 and index == (last,) else ""


def _aliased_named(locations, instructions, computations):
    """What a while loop's body writes into a state element a buffer is also at (its aliases' `locations`: a
    while's output element), the first found (`_loop_named`); None where it is at none."""
    for location in locations:
        loop = instructions[location.instruction_id]
        if loop.opcode == "while" and len(location.shape_index) == 1:
            written = _loop_named(loop, location.shape_index[0], instructions, computations)
            if written is not None:
                return written
    return None


def _loop_named(loop, element: int, instructions, computations: dict[int, Any]
                ):
    """What while `loop`'s body writes into the whole of its state's `element`: the body's element where it
    is named (the state element is that value); else the row a scan writes into its stacked output each step
    (`_stacked`); None for any other element, a write into part of a buffer naming none of it."""
    # XLA's while names its body first, its condition second.
    root = instructions[computations[loop.called_computation_ids[0]].root_id]
    if root.opcode != "tuple" or element >= len(root.operand_ids):
        return None
    written = instructions[root.operand_ids[element]]
    if written.frontend_attributes.map.get("checkpoint_name"):
        return written
    return _stacked(loop, element, written, instructions, computations)


def _stacked(loop, element: int, written, instructions,
             computations):
    """The row a scan writes into its stacked output each step, state `element` of while `loop`, `written`
    the body's value for it: XLA keeps a loop's state in its initial elements' buffers, so a scan's stack is
    the buffer its broadcast fills, which carries no name, while each step's dynamic-update-slice writes the
    residual into it (row 13304: a scanned stack's residuals, gibibytes each, read unnamed at the step's
    peak). The update, where the loop provably writes every row of the element and nothing else: a broadcast
    fills the element at the loop's start; the body writes one whole row into the element's own state at the
    induction variable i or at n - 1 - i, itself, fused or not, run asynchronously or not (XLA:GPU's async
    fusion: an async-done of an async-start whose computation is the fusion); XLA's trip count annotation
    (WhileLoopBackendConfig) states i runs from 0 by 1 over n steps, n the stack's rows. None otherwise."""
    # Each caller the write is made in, outermost first, with what its called computation holds: an
    # async-start's, then a fusion's.
    callers: list[tuple[Any, frozenset[int]]] = []
    put = written
    if put.opcode == "async-done":
        put = instructions[put.operand_ids[0]]
        if put.opcode != "async-start":
            return None
    while put.opcode in ("async-start", "fusion"):
        computation = computations[put.called_computation_ids[0]]
        callers.append((put, frozenset(instruction.id for instruction in computation.instructions)))
        put = instructions[computation.root_id]
    if put.opcode != "dynamic-update-slice" or not put.shape.dimensions:
        return None
    # A loop's config is XLA's JSON where XLA wrote it, an opaque string where a module's text gave one,
    # whose loop states no trip count to read: its stack is named by none.
    known = None
    with contextlib.suppress(json.JSONDecodeError):
        known = json.loads(loop.backend_config or "{}")
    if not isinstance(known, dict) or not {"known_trip_count", "known_init_step",
                                           "known_induction_variable"} <= known.keys():
        return None  # Proto3's JSON leaves a zero out, so a missing field is 0.
    steps = int(known["known_trip_count"].get("n", 0))
    start = known["known_init_step"]
    counter = int(known["known_induction_variable"].get("tuple_index", 0))
    if int(start.get("init", 0)) != 0 or int(start.get("step", 0)) != 1:
        return None
    # XLA's while names its body first; its one parameter is the loop's state.
    body = computations[loop.called_computation_ids[0]]
    [state_] = [instruction.id for instruction in body.instructions if instruction.opcode == "parameter"]

    def outside(instruction):
        # The body's instruction a called one is: each caller's parameter its operand, innermost first.
        for caller, called in reversed(callers):
            if instruction.opcode == "parameter" and instruction.id in called:
                instruction = instructions[caller.operand_ids[instruction.parameter_number]]
        return instruction

    def state(instruction) -> int | None:
        # The body state's element `instruction` reads, if it reads one.
        if instruction.opcode != "get-tuple-element" or instruction.operand_ids[0] != state_:
            return None
        return instruction.tuple_index

    rows, *row = put.shape.dimensions
    update = instructions[put.operand_ids[1]]
    if (steps != rows or list(update.shape.dimensions) != [1, *row]
            # The element itself, never one laid out again: a bitcast to another layout and a copy back moves
            # its elements (a square's transpose).
            or state(outside(instructions[put.operand_ids[0]])) != element):
        return None
    # The row each step writes, its other starts clamped to 0 as it is whole: the counter itself, never one
    # passed through another op (a bitcast to a signed type of its width takes a large count negative).
    index = outside(instructions[put.operand_ids[2]])
    if index.opcode == "subtract":
        last = outside(instructions[index.operand_ids[0]])
        if last.opcode != "constant" or _scalar(last) != rows - 1:
            return None
        index = outside(instructions[index.operand_ids[1]])
    if state(index) != counter:
        return None
    initial = instructions[loop.operand_ids[0]]
    if initial.opcode != "tuple" or not _spread(instructions[initial.operand_ids[element]], instructions,
                                                computations):
        return None
    # What the row is, through the bitcasts, reshapes and copies on its way.
    return _laid_out(outside(_laid_out(update, instructions)), instructions)


def _laid_out(instruction, instructions):
    """The value `instruction` lays out again (a bitcast, a reshape, a copy, which keep its elements), through
    each; `instruction` where it is none."""
    while instruction.opcode in ("bitcast", "reshape", "copy"):
        instruction = instructions[instruction.operand_ids[0]]
    return instruction


def _spread(value, instructions, computations) -> bool:
    """Whether array `value` is one scalar broadcast, or a constant: a fusion's output or a multi-output
    fusion's element so made too."""
    value = _laid_out(value, instructions)
    element = None
    if value.opcode == "get-tuple-element":
        element, value = value.tuple_index, instructions[value.operand_ids[0]]
    if value.opcode == "fusion":
        value = instructions[computations[value.called_computation_ids[0]].root_id]
    if element is not None:
        if value.opcode != "tuple":
            return False
        value = instructions[value.operand_ids[element]]
    value = _laid_out(value, instructions)
    if value.opcode == "constant":
        return True
    return value.opcode == "broadcast" and not instructions[value.operand_ids[0]].shape.dimensions


def _scalar(constant) -> int | None:
    """Integer scalar `constant`'s value; None for any other."""
    literal = constant.literal
    for values in (literal.s32s, literal.s64s, literal.u32s, literal.u64s):
        if len(values) == 1:
            return int(values[0])
    return None


def _kind(allocation) -> str | None:
    """What `allocation` holds: an entry parameter's, an output's or temporaries; None for a constant's or a
    thread-local one."""
    if allocation.is_constant or allocation.is_thread_local:
        return None
    if allocation.is_entry_computation_parameter:
        return "parameter"
    return "output" if allocation.maybe_live_out else "temporary"


type Dims = tuple[frozenset[tuple[int, int]] | None, ...]
"""An array's provenance: for each of its dimensions the (argument, axis) pairs it descends from, empty where
its size descends from none (an iota's, a constant's, a closed-over constant's), None where no op on its path
states its dimensions (a custom call that states none)."""

_SAME_POSITIONS = frozenset((
    "abs", "acos", "acosh", "add", "and", "asin", "asinh", "atan2", "atanh", "bitcast-convert", "cbrt",
    "ceil", "clamp", "compare", "complex", "convert", "cosh", "cosine", "count-leading-zeros", "divide",
    "erf", "exp2",
    "exponential", "exponential-minus-one", "floor", "imag", "is-finite", "log", "log-plus-one", "log2",
    "logistic", "map", "maximum", "minimum", "mulhi", "multiply", "negate", "not", "or", "popcnt", "power",
    "real", "reduce-precision", "remainder", "round-nearest-afz", "round-nearest-even", "rsqrt", "select",
    "shift-left", "shift-right-arithmetic", "shift-right-logical", "sign", "sine", "sinh", "sqrt",
    "stochastic-convert", "subtract", "tan", "tanh", "xor", "copy", "reverse", "cholesky", "triangular-solve",
    "fft", "slice", "dynamic-slice", "pad", "dynamic-update-slice", "concatenate", "set-dimension-size",
    "add-dependency", "domain", "reduce-window", "select-and-scatter", "all-reduce", "all-reduce-start",
    "all-reduce-done", "all-gather", "reduce-scatter", "all-to-all", "collective-permute",
    "collective-broadcast", "all-gather-done", "collective-permute-done", "copy-done", "async-update",
))
"""Ops each of whose output dimensions descends from its array operands' dimensions at the same position
(every operand of the output's rank): those computing elementwise, slicing, padding, joining, reversing, or
exchanging an array across devices along a dimension it keeps."""

_DESCENDED_FROM_NONE = frozenset((
    "constant", "iota", "rng-bit-generator", "rng-get-and-update-state", "partition-id", "replica-id",
    "get-dimension-size", "after-all", "infeed", "recv", "recv-done", "rng",
))
"""Ops whose outputs' sizes descend from no parameter's axis."""

_PER_OPERAND = frozenset(("sort", "scatter"))
"""Ops whose outputs are their leading operands', one each, as they are."""

_GEMMS = frozenset(("__cublas$gemm", "__cublas$lt$matmul", "__cublas$lt$matmul$f8"))
"""cuBLAS custom calls, whose backend config states their dot's dimension numbers
(`gemm_backend_config.dot_dimension_numbers`)."""


def provenance(module, argument: Callable[[int], int | None] | None = None):
    """Each instruction's provenance in `HloModuleProto` `module`, by its id: an array's `Dims`, a tuple's a
    list of its elements'. Read through the dimension maps each op states (a dot's and a cuBLAS call's
    dimension numbers, a reduce's, broadcast's and transpose's dimensions, a reshape's and bitcast's merges, a
    gather's and a convolution's dimension numbers), into fusion, call and conditional bodies from their
    callers' operands, and over a while loop's body to a fixed point; a custom call states its own in its
    `output_dims` frontend attribute, else its dimensions are unknown. An instruction a computation called
    from several places reaches holds them all. `argument` maps an entry parameter to the call's argument it
    holds (`Assignment`)."""
    computations = {computation.id: computation for computation in module.computations}
    found: dict[int, Any] = {}

    def evaluate(computation, parameters):
        held: dict[int, Any] = {}
        shapes: dict[int, Any] = {}
        for instruction in computation.instructions:
            operands = [held[operand] for operand in instruction.operand_ids]
            operand_shapes = [shapes[operand] for operand in instruction.operand_ids]
            called = [computations[index] for index in instruction.called_computation_ids]
            value = _of(instruction, operands, operand_shapes, parameters, called, evaluate)
            held[instruction.id], shapes[instruction.id] = value, instruction.shape
            known = found.get(instruction.id)
            found[instruction.id] = value if known is None else _joined(known, value)
        return held[computation.root_id]

    entry = computations[module.entry_computation_id]
    evaluate(entry, {instruction.parameter_number: _parameter(
        instruction.parameter_number if argument is None else argument(instruction.parameter_number),
        instruction.shape, ()) for instruction in entry.instructions if instruction.opcode == "parameter"})
    return found


def _of(instruction, operands, operand_shapes, parameters,
        called, evaluate: Callable[[Any, dict[int, Any]], Any]):
    """`instruction`'s provenance from its operands'."""
    opcode, shape = instruction.opcode, instruction.shape
    if opcode == "parameter":
        return parameters[instruction.parameter_number]
    if opcode in _DESCENDED_FROM_NONE:
        return _filled(shape, frozenset())
    if opcode == "tuple":
        return list(operands)
    if opcode == "opt-barrier":
        return operands[0]
    if opcode == "get-tuple-element":
        return operands[0][instruction.tuple_index]
    if opcode in _SAME_POSITIONS:
        return _same_positions(shape, operands)
    if opcode in _PER_OPERAND:
        return operands[0] if not shape.tuple_shapes else list(operands[:len(shape.tuple_shapes)])
    if opcode == "broadcast":
        dims: list[Any] = [frozenset()] * len(shape.dimensions)
        for axis, placed in enumerate(instruction.dimensions):
            dims[placed] = operands[0][axis]
        return tuple(dims)
    if opcode == "transpose":
        return tuple(operands[0][axis] for axis in instruction.dimensions)
    if opcode == "reshape":
        return _merged(operands[0], list(operand_shapes[0].dimensions), list(shape.dimensions))
    if opcode == "bitcast":
        return _bitcast(operands[0], operand_shapes[0], shape)
    if opcode == "dot":
        return _dot(operands[0], operands[1], instruction.dot_dimension_numbers)
    if opcode == "convolution":
        return _convolution(instruction, operands)
    if opcode == "reduce":
        return _reduced(instruction, operands, shape)
    if opcode == "gather":
        return _gathered(instruction, operands)
    if opcode in ("fusion", "call"):
        return evaluate(called[0], dict(enumerate(operands)))
    if opcode == "while":
        return _loop(operands[0], called, evaluate)
    if opcode == "conditional":
        joined = None
        for branch, operand in zip(called, operands[1:], strict=False):
            root = evaluate(branch, {0: operand})
            joined = root if joined is None else _joined(joined, root)
        return joined
    if opcode == "async-start":
        # ((its operands), (its wrapped computation's outputs), its context...).
        return [list(operands), evaluate(called[0], dict(enumerate(operands))),
                *(_filled(element, frozenset()) for element in shape.tuple_shapes[2:])]
    if opcode in ("all-gather-start", "collective-permute-start", "copy-start"):
        # (its operand, its output, its context...), the output its operand's.
        head = operands[0] if len(operands) == 1 else list(operands)
        return [head, head, *(_filled(element, frozenset()) for element in shape.tuple_shapes[2:])]
    if opcode == "async-done":
        return operands[0][1]
    if opcode == "custom-call":
        return _custom(instruction, operands, shape)
    return _filled(shape, None)


def _parameter(number: int | None, shape, index: tuple[int, ...]):
    """An entry parameter's provenance, numbered as the call's argument it is (`number`): each dimension its
    own axis; none for a parameter that is no argument (a closed-over constant, a token), whose sizes descend
    from none; an element of a tuple parameter by its index, which the pairs do not name, unknown."""
    if shape.tuple_shapes:
        return [_parameter(number, element, (*index, position))
                for position, element in enumerate(shape.tuple_shapes)]
    if number is None:
        return tuple(frozenset() for _ in shape.dimensions)
    if index:
        return tuple(None for _ in shape.dimensions)
    return tuple(frozenset({(number, axis)}) for axis in range(len(shape.dimensions)))


def _filled(shape, dimension: frozenset[tuple[int, int]] | None):
    """`shape`'s provenance with every dimension `dimension`."""
    if shape.tuple_shapes:
        return [_filled(element, dimension) for element in shape.tuple_shapes]
    return tuple(dimension for _ in shape.dimensions)


def _union(left: frozenset[tuple[int, int]] | None, right: frozenset[tuple[int, int]] | None
           ) -> frozenset[tuple[int, int]] | None:
    """Two dimensions' provenance together: unknown where either is."""
    return None if left is None or right is None else left | right


def _joined(left, right):
    """Two provenances of one shape together, dimension by dimension."""
    if isinstance(left, list) and isinstance(right, list):
        return [_joined(a, b) for a, b in zip(left, right, strict=True)]
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return tuple(_union(a, b) for a, b in zip(left, right, strict=True))
    raise ValueError(f"provenances of two shapes joined: {left!r}, {right!r}")


def _same_positions(shape, operands):
    """Each output dimension its operands' at its position: a tuple output's elements each their tuple
    operands' element, or the array operand at their index (a variadic collective's, reduce-window's)."""
    if shape.tuple_shapes:
        tuples = [operand for operand in operands if isinstance(operand, list)]
        return [_same_positions(element, [operand[index] for operand in tuples] if tuples
                                else [operands[index]]) for index, element in enumerate(shape.tuple_shapes)]
    rank = len(shape.dimensions)
    dims: list[Any] = [frozenset()] * rank
    for operand in operands:
        if isinstance(operand, tuple) and len(operand) == rank:
            dims = [_union(held, given) for held, given in zip(dims, operand, strict=True)]
    return tuple(dims)


def _merged(operand: Dims, sizes: list[int], merged: list[int]) -> Dims:
    """A reshape's provenance: each output dimension descends from each operand dimension whose span of the
    row-major index it overlaps (a dimension spans the strides from the product of the sizes after it to that
    times its own), so a merged dimension from all it merges and a split one's parts each from the one it
    splits; a dimension of size one spans none."""
    def spans(shape: list[int]) -> list[tuple[int, int]]:
        found, stride = [], 1
        for size in reversed(shape):
            found.append((stride, stride * size))
            stride *= size
        return list(reversed(found))

    given = spans(sizes)
    dims = []
    for low, high in spans(merged):
        descended: frozenset[tuple[int, int]] | None = frozenset()
        for axis, (start, end) in enumerate(given):
            if max(low, start) < min(high, end):
                descended = _union(descended, operand[axis])
        dims.append(descended)
    return tuple(dims)


def _bitcast(operand: Dims, operand_shape, shape) -> Dims:
    """A bitcast's provenance: a reshape of the dimensions as laid out, major to minor (its layout's), so a
    bitcast that transposes is read as one."""
    if (list(operand_shape.dimensions) == list(shape.dimensions)
            and list(operand_shape.layout.minor_to_major) == list(shape.layout.minor_to_major)):
        return operand
    source, target = _major_to_minor(operand_shape), _major_to_minor(shape)
    sizes = [operand_shape.dimensions[axis] for axis in source]
    targets = [shape.dimensions[axis] for axis in target]
    laid = _merged(tuple(operand[axis] for axis in source), sizes, targets)
    dims: list[frozenset[tuple[int, int]] | None] = [frozenset()] * len(shape.dimensions)
    for position, axis in enumerate(target):
        dims[axis] = laid[position]
    return tuple(dims)


def _major_to_minor(shape) -> list[int]:
    order = list(shape.layout.minor_to_major)
    return list(reversed(order)) if order else list(range(len(shape.dimensions)))


def _dot(lhs: Dims, rhs: Dims, numbers) -> Dims:
    """A dot's provenance: its batch dimensions, then the left operand's free ones, then the right's, each
    from those it is made of."""
    lhs_batch, rhs_batch = list(numbers.lhs_batch_dimensions), list(numbers.rhs_batch_dimensions)
    lhs_kept = set(lhs_batch) | set(numbers.lhs_contracting_dimensions)
    rhs_kept = set(rhs_batch) | set(numbers.rhs_contracting_dimensions)
    return (*(_union(lhs[a], rhs[b]) for a, b in zip(lhs_batch, rhs_batch, strict=True)),
            *(lhs[axis] for axis in range(len(lhs)) if axis not in lhs_kept),
            *(rhs[axis] for axis in range(len(rhs)) if axis not in rhs_kept))


def _convolution(instruction, operands) -> Dims:
    """A convolution's provenance: its batch dimension the input's, its feature dimension the kernel's output
    feature (with the input's feature or batch dimension where it groups them), each spatial dimension the
    input's and the kernel's at its place."""
    numbers, (lhs, rhs) = instruction.convolution_dimension_numbers, operands[:2]
    rank = 2 + len(numbers.output_spatial_dimensions)
    dims: list[frozenset[tuple[int, int]] | None] = [frozenset()] * rank
    dims[numbers.output_batch_dimension] = lhs[numbers.input_batch_dimension]
    feature = rhs[numbers.kernel_output_feature_dimension]
    if instruction.feature_group_count > 1:
        feature = _union(feature, lhs[numbers.input_feature_dimension])
    if instruction.batch_group_count > 1:
        feature = _union(feature, lhs[numbers.input_batch_dimension])
    dims[numbers.output_feature_dimension] = feature
    for output, given, kernel in zip(numbers.output_spatial_dimensions, numbers.input_spatial_dimensions,
                                     numbers.kernel_spatial_dimensions, strict=True):
        dims[output] = _union(lhs[given], rhs[kernel])
    return tuple(dims)


def _reduced(instruction, operands, shape):
    """A reduce's provenance: each output its input's dimensions but those it reduces, in order."""
    inputs = operands[:len(operands) // 2]
    removed = set(instruction.dimensions)
    outputs = [tuple(dimension for axis, dimension in enumerate(operand) if axis not in removed)
               for operand in inputs]
    return outputs if shape.tuple_shapes else outputs[0]


def _gathered(instruction, operands) -> Dims:
    """A gather's provenance: its offset dimensions the operand's it keeps, in order; its batch dimensions the
    start indices', in order, with an operand batching dimension's beside its indices' own."""
    numbers, (operand, indices) = instruction.gather_dimension_numbers, operands[:2]
    offsets = list(numbers.offset_dims)
    dropped = set(numbers.collapsed_slice_dims) | set(numbers.operand_batching_dims)
    kept = [axis for axis in range(len(operand)) if axis not in dropped]
    batched = dict(zip(numbers.start_indices_batching_dims, numbers.operand_batching_dims, strict=True))
    indexed = [axis for axis in range(len(indices)) if axis != numbers.index_vector_dim]
    rank = len(offsets) + len(indexed)
    dims: list[frozenset[tuple[int, int]] | None] = [frozenset()] * rank
    batch, offset = iter(indexed), iter(kept)
    for axis in range(rank):
        if axis in offsets:
            dims[axis] = operand[next(offset)]
        else:
            index = next(batch)
            dims[axis] = indices[index]
            if index in batched:
                dims[axis] = _union(dims[axis], operand[batched[index]])
    return tuple(dims)


def _loop(initial, called, evaluate: Callable[[Any, dict[int, Any]], Any]):
    """A while loop's provenance: its state's, its body's output joined into it until the body adds nothing
    (sets only grow, so it ends)."""
    body = called[0]  # XLA's while names its body first, its condition second.
    state = initial
    while True:
        joined = _joined(state, evaluate(body, {0: state}))
        if joined == state:
            return state
        state = joined


def _custom(instruction, operands, shape):
    """A custom call's provenance: as its `output_dims` frontend attribute states it (one list an output, one
    entry a dimension, each a list of [operand, axis] pairs, empty for a size descending from none, null for
    one unknown; `dew.nn.kernels.templates.outputs`); a cuBLAS call's from its dot's dimension numbers, its
    workspace's from none; else unknown."""
    stated = instruction.frontend_attributes.map.get("output_dims")
    if stated:
        elements = list(shape.tuple_shapes) if shape.tuple_shapes else [shape]
        aliased = {tuple(alias.output_shape_index): alias.operand_index
                   for alias in instruction.output_operand_aliasing}
        given = json.loads(stated)
        outputs = []
        for index, element in enumerate(elements):
            if (index,) in aliased:
                # An operand updated in place (a ref a kernel adds into): its own.
                outputs.append(operands[aliased[(index,)]])
            elif index < len(given) and given[index] is not None:
                outputs.append(tuple(None if entry is None else _from_operands(entry, operands)
                                     for entry in given[index]))
            else:
                # Stated unknown (null), or past the outputs stated (a kernel's scratch): unknown.
                outputs.append(_filled(element, None))
        return outputs if shape.tuple_shapes else outputs[0]
    if instruction.custom_call_target in _GEMMS:
        config = json.loads(instruction.backend_config or "{}")
        numbers = config.get("gemm_backend_config", {}).get("dot_dimension_numbers")
        if numbers is not None:
            product = _dot(operands[0], operands[1], _Numbers(numbers))
            if shape.tuple_shapes:
                return [product, *(_filled(element, frozenset()) for element in shape.tuple_shapes[1:])]
            return product
    return _filled(shape, None)


def _from_operands(entry: list[list[int]], operands) -> frozenset[tuple[int, int]] | None:
    descended: frozenset[tuple[int, int]] | None = frozenset()
    for operand, axis in entry:
        descended = _union(descended, operands[operand][axis])
    return descended


class _Numbers:
    """A dot's dimension numbers as a cuBLAS call's backend config states them (JSON: lists of strings), read
    as `DotDimensionNumbers` is."""

    def __init__(self, numbers: dict[str, list[str]]):
        for name in ("lhs_contracting_dimensions", "rhs_contracting_dimensions", "lhs_batch_dimensions",
                     "rhs_batch_dimensions"):
            setattr(self, name, [int(axis) for axis in numbers.get(name, [])])
