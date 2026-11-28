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

import contextlib
import dataclasses
import json
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax

from dew.telemetry import hlo
from dew.telemetry.instrumentation import call_scratch, xla_gemm


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
    none, None where an op on its path states no map (a custom call stating no `output_dims`). `stack` is,
    for a buffer of a scan's stack, the loop writing it and the state element it is (the while's instruction
    id and the element's index), where the loop writes the element a row a step and its body and condition
    read it nowhere else (`_stacked`): each buffer of the element's own values XLA's alias analysis holds
    together (`BufferAssignmentProto.buffer_aliases`), the zeros before the loop, the loop's state, each
    step's write and the loop's output, is that one stack, which a step that saves none of its rows never
    builds; None
    otherwise. `filling` says whether it is one of the values its loop fills the stack through, read by
    nothing but its loop (the zeros, the loop's state, each step's write), beside the loop's output, the
    value what reads the stack past the loop reads.
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
    stack: tuple[int, int] | None = None
    filling: bool = False


UNPLACED = -1
"""A use's position where its trace states none (`LiveInterval.uses`): a reader with no event and none
reading through it."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class LiveInterval(LiveBuffer):
    """A buffer of a compile's allocation over the part of a heap simulation's schedule it lives in.

    Besides `LiveBuffer`'s: `allocation`, the buffer allocation it is assigned to; `trace`, the heap
    simulation that orders it (its index among the assignment's `heap_simulator_traces`): positions compare
    within one trace alone (an allocation simulated per computation has one each, their positions each from 0;
    the pages of one heap each a copy of its one trace, their positions alike). `timeline` is the first trace
    whose events its trace's repeat: a heap's pages share one, their buffers live together at each position,
    and traces of different timelines order their positions apart. `whole` says whether that
    simulation's schedule is the whole module's (XLA:GPU's), a while body's buffers within its loop's span, or
    one computation's. `complete` says whether every array buffer its allocation is assigned lies in one of
    its simulations (a tuple's index table, no array, aside): XLA places some in an allocation unsimulated
    (temporary heaps reused, or combined), so a planner needing all of an allocation's buffers over one
    schedule reads it first. `start` is its first
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
    timeline: int
    uses: tuple[tuple[int, str], ...] = ()
    """The positions of its trace by which each instruction reading the value it holds has run (its last
    event there), each with that instruction's `op_name`: what its value is read by and when, a tuple's
    element through the getter that takes it out, a view's readers through the view; `UNPLACED` where a
    reader's position is stated nowhere."""
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
        self.computation_of = {instruction.id: computation.name for computation in compiled.computations
                               for instruction in computation.instructions}
        self.descended = provenance(compiled, argument)
        # Each logical buffer's other positions, as XLA's must-alias analysis states them: a loop state's
        # buffer's among them, at the while's output.
        self.aliases: dict[int, list[Any]] = {}
        # Each instruction's users with the operand slot each reads it in (`readers`), built where an interval
        # or a loop's stack is read.
        self.uses: dict[int, list[tuple[Any, int]]] | None = None
        # Each computation's parameter ids by number (`parameter`), built where a fusion is walked into.
        self.bound: dict[tuple[int, int], int] | None = None
        # Each loop element's reading (`_Loops`), built where a buffer is described and kept.
        self.loops: _Loops | None = None
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
        # A tuple's index table is no array (`described`), which XLA may place in an allocation past its
        # simulated heap (sm90's gemm fusions' tuples: rows 14089 and 14091's sm90 twins): an allocation is
        # complete where each array it is assigned is simulated.
        table = self.types.values_by_name["TUPLE"].number
        complete = {allocation.index: {each.logical_buffer_id for each in allocation.assigned
                                       if each.element_type != table}
                    <= traced.get(allocation.index, set()) for allocation in self.proto.buffer_allocations}
        found = []
        # A heap's pages each copy one trace's events: the first trace of each sequence of events names it.
        timelines: dict[tuple, int] = {}
        for trace, simulated in enumerate(self.proto.heap_simulator_traces):
            timeline = timelines.setdefault(tuple((step.kind, step.buffer_id, step.share_with_canonical_id)
                                                  for step in simulated.events), trace)
            # Each buffer's storage, as first allocated: a share names the buffer it shares, itself perhaps
            # sharing another's, freed or not.
            storages: dict[int, int] = {}
            opened: dict[int, int] = {}
            spans = []
            for position, step in enumerate(simulated.events):
                if step.kind == kinds["FREE"]:
                    spans.append((step.buffer_id, opened.pop(step.buffer_id), position))
                    continue
                shared: int = int(step.share_with_canonical_id)
                sharing = step.kind == kinds["SHARE_WITH"]
                storages[step.buffer_id] = storages.get(shared, shared) if sharing else int(step.buffer_id)
                opened[step.buffer_id] = position
            spans += [(buffer, start, len(simulated.events)) for buffer, start in opened.items()]
            # Where each instruction runs in this trace: its last output's placing (`ALLOC`, `SHARE_WITH`); a
            # `FREE` names the producer of the value dying, not the instruction running there.
            ran: dict[tuple[str, str], int] = {}
            for position, step in enumerate(simulated.events):
                if step.kind != kinds["FREE"]:
                    ran[(step.computation_name, step.instruction_name)] = position
            allocation = simulated.buffer_allocation_index
            sizes = assigned.get(allocation, {})
            charged = _charged(simulated.events, sizes, kinds)
            found += [LiveInterval(**{**described, "bytes": sizes[buffer]}, allocation=allocation,
                                   trace=trace, timeline=timeline, whole=simulated.whole_module_simulation,
                                   complete=complete[allocation], start=start, end=end, buffer=buffer,
                                   storage=storages[buffer], charged=buffer in charged,
                                   uses=self.used(buffer, ran))
                      for buffer, start, end in spans
                      if buffer in sizes and (described := self.described(buffer, allocation)) is not None]
        return tuple(found)

    def used(self, buffer_id: int, ran: Mapping[tuple[str, str], int]) -> tuple[tuple[int, str], ...]:
        """Where in a trace (`ran`: each instruction's position in it) the instructions reading logical
        buffer `buffer_id`'s value run, with their `op_name`s: its defining instruction's readers, a tuple
        element's through the getter of its index, a view's through its readers; a reader the trace states
        no event of is none of them (`UNPLACED`). A fusion of no op_name of its own reads it, at its own
        position, as the fused ops reading the operand slots it takes the value in: through views, tuple
        slots and nested fusions, a nested fusion's result its caller's where its root passes the value
        on; "" where a path reaches no name, a read no phase is told of (trains 1406 and 1427)."""
        uses = self.readers()
        buffer = self.buffers[buffer_id]
        made = buffer.defined_at.instruction_id
        found: set[tuple[int, str]] = set()
        # Each state the value is at: an instruction whose result holds it, the index within that result,
        # and the fusions (outermost first) whose bodies it is in.
        reading = [(made, tuple(buffer.defined_at.shape_index), ())]
        seen: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
        while reading:
            state = reading.pop()
            if state in seen:
                continue
            seen.add(state)
            value, within, inside = state
            users = uses.get(value, ())
            if not users and value != made:
                found.update(self.left(value, within, inside, ran, reading))
            for user, slot in users:
                found.update(self.read_by(user, slot, within, inside, ran, reading))
        return tuple(sorted(found))

    def readers(self):
        """Each instruction's users across the module by its id, with the operand slot each reads it in,
        built once and kept."""
        if self.uses is None:
            self.uses = {}
            for instruction in self.instructions.values():
                for slot, operand in enumerate(instruction.operand_ids):
                    self.uses.setdefault(operand, []).append((instruction, slot))
        return self.uses

    def read_by(self, user, slot: int, within: tuple[int, ...], inside: tuple[int, ...],
                ran: Mapping[tuple[str, str], int], reading: list) -> list[tuple[int, str]]:
        """What `user`, reading the value at its operand `slot` (`within` its index there), reads it as,
        queuing on `reading` where it passes the value on (`used`)."""
        if within and user.opcode == "get-tuple-element":
            # A getter of the value's slot passes it on, of another reads none of it; a named one inside a
            # fusion a read of its own phase besides, as a named view is (train 1437's review).
            if user.tuple_index != within[0]:
                return []
            reading.append((user.id, within[1:], inside))
            named = inside and user.metadata.op_name
            return [(self.position(inside, ran), user.metadata.op_name)] if named else []
        packed = user.opcode == "tuple"
        if inside:
            # A view, a tuple or a nested fusion inside a fusion passes the value on whatever its name, its
            # readers reading it too: a named one is a read of its own phase besides (a bitcast XLA kept a
            # forward's name on, which feeds the backward's op; train 1437's P0).
            named = [(self.position(inside, ran), user.metadata.op_name)] if user.metadata.op_name else []
            if packed:
                # Packed at its slot: a getter of another reads none of it.
                reading.append((user.id, (slot, *within), inside))
            elif user.opcode in _VIEWS:
                reading.append((user.id, (), inside))
            elif user.opcode == "fusion":
                # Descended whatever its name, which reads nothing of the ops its body names (a forward's
                # name on a nested fusion returning a view to a backward's op; train 1444's P0).
                reading.append((self.parameter(user, slot), within, (*inside, user.id)))
            elif named:
                return named
            else:
                reading.append((user.id, (), inside))
            return named
        key = (self.computation_of[user.id], user.name)
        if key not in ran:
            # A reader the trace places no buffer for (a view of its operand, a bitcast, a tuple packing it at
            # its slot): its readers read the value through it.
            reading.append((user.id, (slot, *within) if packed else (), inside))
            return []
        if user.metadata.op_name or user.opcode != "fusion":
            return [(ran[key], user.metadata.op_name)]
        reading.append((self.parameter(user, slot), within, (user.id,)))
        return []

    def left(self, value: int, within: tuple[int, ...], inside: tuple[int, ...],
             ran: Mapping[tuple[str, str], int], reading: list) -> list[tuple[int, str]]:
        """Where the value a user of no users holds goes (`used`): out of a nested fusion its root returns
        it from, to its caller's users; out of a scheduled one's, or into an unnamed leaf, read by no phase
        told of (""), but for a named view or getter, the read its name already tells (`read_by`; train 1437's
        review); at the top, read where the trace places none (`UNPLACED`)."""
        if not inside:
            return [(UNPLACED, self.instructions[value].metadata.op_name)]
        caller = self.instructions[inside[-1]]
        if len(inside) > 1 and value == self.computations[caller.called_computation_ids[0]].root_id:
            # Returned (its own parameter, or a view of it): read where the caller's result is.
            reading.append((caller.id, within, inside[:-1]))
            return []
        if self.instructions[value].opcode == "parameter" or self.instructions[value].metadata.op_name:
            return []
        return [(self.position(inside, ran), "")]

    def position(self, inside: tuple[int, ...], ran: Mapping[tuple[str, str], int]) -> int:
        """The trace position of the scheduled fusion `inside` begins in."""
        outer = self.instructions[inside[0]]
        return ran[(self.computation_of[outer.id], outer.name)]

    def parameter(self, fusion, slot: int) -> int:
        """The id of `fusion`'s fused computation's parameter its operand `slot` binds, each computation's
        parameters indexed once and kept (`bound`)."""
        if self.bound is None:
            self.bound = {(computation.id, each.parameter_number): each.id
                          for computation in self.computations.values() for each in computation.instructions
                          if each.opcode == "parameter"}
        return self.bound[(fusion.called_computation_ids[0], slot)]

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
        # The defining instruction's name, where it names the element defined (`_names_output`): a tuple
        # call's names its result alone, an unknown call's none of its elements.
        own = (instruction.frontend_attributes.map.get("checkpoint_name")
               if _names_output(instruction, index) else None)
        # A loop's state element: the row its body writes into it, named and scoped as that row, and the
        # stack it is where the loop writes it alone (`_Loops.row`).
        if self.loops is None:
            self.loops = _Loops(_Module(self.instructions, self.computations, self.readers()))
        row = self.loops.row(instruction, index, self.aliases.get(buffer.id, ()))
        named = own or (None if row is None else row.name)
        scoped = None if own else row
        made, within, getter = _maker(instruction, index, self.instructions, self.computations)
        if not named and getter is not None:
            named = getter.frontend_attributes.map.get("checkpoint_name")
        if not named and _names_output(made, within):
            named = made.frontend_attributes.map.get("checkpoint_name")
        # A stack's op is the row its loop writes, which its rung's scope names, however the stack is
        # assigned (row 12463: read as the loop's or its initializer's, a scanned decoder's stacks lay in no
        # rung's scope, so no remat was priced to drop them; train 1474's P-F3).
        op_name = ((scoped.scope if scoped is not None else "") or metadata.op_name or made.metadata.op_name)
        return {"bytes": buffer.size, "shape": tuple(shape.dimensions),
                "dtype": self.types.values_by_number[shape.element_type].name.lower(), "kind": kind,
                "instruction": instruction.name, "op_name": op_name,
                "source": f"{metadata.source_file}:{metadata.source_line}",
                "names": tuple(named.split(",")) if named else (),
                "parameter": self.parameters.get(allocation),
                "role": _role(made, within),
                "dims": tuple(None for _ in shape.dimensions) if dims is None else tuple(dims),
                "stack": None if row is None else row.stack,
                "filling": row is not None and row.filling}


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
            owned: int = int(step.buffer_id)
            freed = canonical.get(owned, owned)
            instances[freed] = instances.get(freed, 0) - 1
    return charged


_VIEWS = frozenset({"bitcast", "bitcast-convert", "reshape", "copy", "transpose"})
"""The ops that pass a value on as it is, laid out or shaped anew: a read through one is its readers' read."""


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
    return _projected(instructions[computations[instruction.called_computation_ids[0]].root_id], index,
                      instructions, computations, entered=frozenset({"fusion"}))


def _projected(made, index: tuple[int, ...], instructions, computations, *, entered: frozenset[str]):
    """The instruction making element `index` of `made`'s output, through tuples, get-tuple-elements and the
    called computations of the ops `entered` names, with where in its own output the element is, and the
    nearest instruction passed that names the value it selects whole, a get-tuple-element or an entered
    call or fusion (`_maker`'s, a stack's row's: `_value_named`)."""
    getter = None
    while True:
        if made.opcode in entered and made.called_computation_ids:
            # A call or fusion that names the value it returns names the element selected whole.
            if getter is None and _names_output(made, index) and made.frontend_attributes.map.get(
                    "checkpoint_name"):
                getter = made
            made = instructions[computations[made.called_computation_ids[0]].root_id]
        elif made.opcode == "tuple" and index:
            made, index = instructions[made.operand_ids[index[0]]], index[1:]
        elif made.opcode == "get-tuple-element":
            if getter is None and not index and made.frontend_attributes.map.get("checkpoint_name"):
                getter = made
            made, index = instructions[made.operand_ids[0]], (made.tuple_index, *index)
        else:
            return made, index, getter


def _cublas(maker) -> bool:
    """Whether `maker` is a custom call whose output is its result first and its scratch last (every cuBLAS
    call of the pinned XLA: `dew.telemetry.instrumentation.call_scratch`, the one record of what a call's
    target is), carrying the frontend attributes of the instruction it replaced."""
    return maker.opcode == "custom-call" and call_scratch(maker.custom_call_target)


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


class _Module(NamedTuple):
    """A compiled module's instructions and computations by id, and each instruction's readers with the
    operand slot each reads it in (`Assignment.readers`): what a loop's state is read through."""

    instructions: Mapping[int, Any]
    computations: Mapping[int, Any]
    readers: Mapping[int, list[tuple[Any, int]]]


class _Row(NamedTuple):
    """The value a loop's body writes into a state element (`_Loops.row`): its checkpoint name and the op
    that names or makes it, the scope a remat rung reads it by, the stack the element is
    (`LiveBuffer.stack`) where the loop writes it a row a step and reads it nowhere else, and whether the
    buffer is one its loop fills it through (`LiveBuffer.filling`)."""

    name: str | None
    scope: str
    stack: tuple[int, int] | None
    filling: bool


class _Written(NamedTuple):
    """What a loop's body writes into a state element (`_loop_named`): the value, and, where it is the row
    of a scan's stack the loop fills alone (`_stacked`), the positions (an instruction's id and the index
    in its output) defining the stack's values its loop fills it through, read by nothing but the loop:
    the zeros it starts from, the body's and condition's state and each step's write; empty otherwise."""

    value: Any
    filling: frozenset[tuple[int, tuple[int, ...]]] = frozenset()


class _Loops:
    """A module's loop state elements as its buffers are read (`row`), each element's reading made once and
    kept: what its body writes into it (`_loop_named`), named and scoped (`_value_named`)."""

    def __init__(self, module: _Module):
        self.module = module
        self.written: dict[tuple[int, int, bool], tuple[_Written, str | None, str] | None] = {}

    def row(self, instruction, index: tuple[int, ...], locations) -> _Row | None:
        """The row a loop's body writes into the state element a buffer holds, named and scoped as that row,
        with the stack the element is: the buffer defined at the loop itself (its state's value past the
        loop, which XLA assigns apart from its initial element's), where the loop is stated to run, or at an
        instruction one of the loop's elements aliases (its `locations`: a scan's stack broadcast before the
        loop, each step's write into it); None where it holds none (row 12463: a scanned decoder's stacks,
        read unnamed and in no rung's scope). Named as the first loop's element it is read as; a stack only
        of an element whose own value the buffer is (the loop's output, or a value it fills it through,
        `_Written.filling`), whichever of the loops its alias set holds that is: XLA's alias set can hold
        other values beside them (a conditional returning one scan's stack from one branch and another's,
        or another array, from the other; review of row 12463)."""
        loops = ([(instruction, index[0], True)] if instruction.opcode == "while" and len(index) == 1
                 else [])
        loops += [(self.module.instructions[location.instruction_id], location.shape_index[0], False)
                  for location in locations if len(location.shape_index) == 1]
        named: tuple[_Written, str | None, str] | None = None
        for loop, element, ran in loops:
            if loop.opcode != "while" or (found := self.loop_named(loop, element, ran=ran)) is None:
                continue
            named = named or found
            written = found[0]
            output = loop.id == instruction.id and ran
            if written.filling and (output or (instruction.id, index) in written.filling):
                return _Row(found[1], found[2], (loop.id, element), filling=not output)
        return None if named is None else _Row(named[1], named[2], None, filling=False)

    def loop_named(self, loop, element: int, *, ran: bool) -> tuple[_Written, str | None, str] | None:
        """`_loop_named`'s reading of `loop`'s `element`, with its value's name and scope (`_value_named`),
        made once; None where it reads none."""
        key = (loop.id, element, ran)
        if key not in self.written:
            written = _loop_named(loop, element, self.module, ran=ran)
            self.written[key] = None if written is None else (
                written, *_value_named(written.value, self.module.instructions, self.module.computations))
        return self.written[key]


class _Trips(NamedTuple):
    """XLA's trip count annotation of a while (WhileLoopBackendConfig): its steps, and where XLA states them
    (its annotator's affine loops alone), its counter's start and stride and the state element holding the
    counter."""

    steps: int
    init: int | None
    step: int | None
    counter: int | None


def _annotation(loop) -> _Trips | None:
    """`loop`'s trip count annotation (`_Trips`), None where it states no trip count: a config XLA's parser
    held as an opaque string, or one of no count. Its counter's fields None where XLA states none (its
    annotator's count of a loop whose counter is no affine one: a counter from -1); a field missing within
    a stated one is 0 (Proto3's JSON leaves a zero out)."""
    known = None
    with contextlib.suppress(ValueError):  # JSON's, or bytes no text decodes
        known = json.loads(loop.backend_config or "{}")
    if not isinstance(known, dict) or "known_trip_count" not in known:
        return None
    start, counter = known.get("known_init_step"), known.get("known_induction_variable")
    return _Trips(int(known["known_trip_count"].get("n", 0)),
                  None if start is None else int(start.get("init", 0)),
                  None if start is None else int(start.get("step", 0)),
                  None if counter is None else int(counter.get("tuple_index", 0)))


def _loop_named(loop, element: int, module: _Module, *, ran: bool = False) -> _Written | None:
    """What while `loop`'s body writes into the whole of its state's `element`: the body's element where it
    is named (the state element is that value); else the row a scan writes into its stacked output each step
    (`_stacked`); None for any other element, a write into part of a buffer naming none of it. With `ran`,
    for the loop's own result, the body's value where its annotation states it runs a step (`_Trips`): a
    loop of no step holds its initial value (review of row 12463)."""
    trips = _annotation(loop)
    if ran and (trips is None or trips.steps <= 0):
        return None
    # XLA's while names its body first, its condition second.
    root = module.instructions[module.computations[loop.called_computation_ids[0]].root_id]
    if root.opcode != "tuple" or element >= len(root.operand_ids):
        return None
    written = module.instructions[root.operand_ids[element]]
    if written.frontend_attributes.map.get("checkpoint_name"):
        return _Written(written)
    return None if trips is None else _stacked(loop, element, written, trips, module)


def _stacked(loop, element: int, written, trips: _Trips, module: _Module) -> _Written | None:
    """The row a scan writes into its stacked output each step, state `element` of while `loop`, `written`
    the body's value for it: XLA keeps a loop's state in its initial elements' buffers, so a scan's stack is
    the buffer its broadcast fills, which carries no name, while each step's dynamic-update-slice writes the
    residual into it (row 13304: a scanned stack's residuals, gibibytes each, read unnamed at the step's
    peak). The update, where the loop provably writes every row of the element and nothing else: a broadcast
    fills the element at the loop's start; the body writes one whole row into the element's own state at the
    induction variable i or at n - 1 - i, itself, fused or not, run asynchronously or not (XLA:GPU's async
    fusion: an async-done of an async-start whose computation is the fusion); XLA's trip count annotation
    (WhileLoopBackendConfig) states i runs from 0 by 1 over n steps, n the stack's rows. Its values the loop
    fills it through (`_Written.filling`), where besides nothing reads them but the loop (`_written_alone`,
    `_started_alone`). None otherwise."""
    instructions, computations = module.instructions, module.computations
    # Each caller the write is made in, outermost first, with what its called computation holds: an
    # async-start's, then a fusion's.
    callers: list[tuple[Any, frozenset[int]]] = []
    # A bitcast moves no byte (XLA's, past layout assignment): the element is the write it views, a scanned
    # decoder's stack its fusion's update viewed as the state's shape (row 12463: a cold 4B start's 167 GiB of
    # stacked residuals read unnamed, no rung's recomputation priced to free them).
    put = _bitcast_of(written, instructions)
    # The positions the write's value is defined at on its way out: each async call's and fusion's, the
    # write's own.
    writes: set[tuple[int, tuple[int, ...]]] = {(put.id, ())}
    if put.opcode == "async-done":
        put = instructions[put.operand_ids[0]]
        if put.opcode != "async-start":
            return None
        writes.add((put.id, (1,)))
    while put.opcode in ("async-start", "fusion"):
        computation = computations[put.called_computation_ids[0]]
        callers.append((put, frozenset(instruction.id for instruction in computation.instructions)))
        put = instructions[computation.root_id]
        writes.add((put.id, ()))
    if put.opcode != "dynamic-update-slice" or not put.shape.dimensions:
        return None
    steps, counter = trips.steps, trips.counter
    if trips.init != 0 or trips.step != 1 or counter is None:
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

    def viewed(instruction):
        # The body's value `instruction` views, through bitcasts in a caller or out of it.
        instruction = outside(instruction)
        while instruction.opcode == "bitcast":
            instruction = outside(instructions[instruction.operand_ids[0]])
        return instruction

    def state(instruction) -> int | None:
        # The body state's element `instruction` reads, if it reads one.
        if instruction.opcode != "get-tuple-element" or instruction.operand_ids[0] != state_:
            return None
        return instruction.tuple_index

    rows, *row = put.shape.dimensions
    update = instructions[put.operand_ids[1]]
    if (steps != rows or list(update.shape.dimensions) != [1, *row]
            # The element itself, viewed through bitcasts alone, never one laid out again: a copy moves its
            # elements (a square's transpose's copy back).
            or state(viewed(instructions[put.operand_ids[0]])) != element):
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
    row_value = _laid_out(outside(_laid_out(update, instructions)), instructions)
    zeros = _started_alone(loop, element, module)
    if zeros is None or not _written_alone(loop, element, written, put, module):
        return _Written(row_value)
    states = {(each.id, (element,)) for index in loop.called_computation_ids[:2]
              for each in computations[index].instructions if each.opcode == "parameter"}
    return _Written(row_value, frozenset({*writes, *states, zeros}))


def _started_alone(loop, element: int, module: _Module) -> tuple[int, tuple[int, ...]] | None:
    """The position defining the value while `loop`'s state `element` starts from (its initial tuple's
    element, through bitcasts; a getter's at its producer's output), where nothing but the loop reads it:
    its producer read through getters alone, each getter of that output and the value through its bitcasts
    read as the tuple's element alone, and the tuple by the loop alone. None where anything else reads it
    (a metric of the zeros, review of row 12463)."""
    initial = module.instructions[loop.operand_ids[0]]
    if [(reader.id, slot) for reader, slot in module.readers.get(initial.id, ())] != [(loop.id, 0)]:
        return None
    value = _bitcast_of(module.instructions[initial.operand_ids[element]], module.instructions)
    ends = {(initial.id, element)}
    if value.opcode != "get-tuple-element":
        return (value.id, ()) if _flows_to(value, ends, module, inward=False) else None
    # The producer's outputs read through getters alone: a reader of the whole tuple (a call taking it) may
    # read the zeros besides (review of row 12463).
    readers = [reader for reader, _ in module.readers.get(value.operand_ids[0], ())]
    if any(reader.opcode != "get-tuple-element" for reader in readers) or not all(
            _flows_to(getter, ends, module, inward=False)
            for getter in readers if getter.tuple_index == value.tuple_index):
        return None
    return value.operand_ids[0], (value.tuple_index,)


def _written_alone(loop, element: int, written, put, module: _Module) -> bool:
    """Whether while `loop` reads its state's `element` only to write it: in its body, the element's state
    reaches no reader but `put`'s target operand (`_stacked`'s dynamic-update-slice), through bitcasts and
    into the fusions and async calls passing it on as their parameters; `written`, the body's value for it,
    reaches none but the body's result at the element; its condition reads it nowhere. Then the loop's
    steps need none of the element's rows: a scan's stack, held for what reads it past the loop alone
    (row 12463: a scanned decoder's forward held 167 GiB of its stacks' in-loop values, which a step that
    saves none of their rows never builds)."""
    body, condition = (module.computations[index] for index in loop.called_computation_ids[:2])
    for computation, ends in ((body, {(put.id, 0)}), (condition, set())):
        # XLA's while body and condition each take the loop's state as their one parameter.
        [state] = [each for each in computation.instructions if each.opcode == "parameter"]
        for reader, _ in module.readers.get(state.id, ()):
            # The state read whole (a call taking the tuple) is read past any element's write.
            if reader.opcode != "get-tuple-element" or (reader.tuple_index == element
                                                         and not _flows_to(reader, ends, module)):
                return False
    # The write's own result, through the bitcasts viewing it, is the body's result at the element alone.
    return _flows_to(_bitcast_of(written, module.instructions), {(body.root_id, element)}, module,
                     inward=False)


def _flows_to(value, ends: set[tuple[int, int]], module: _Module, *, inward: bool = True) -> bool:
    """Whether every reader of `value` reads it as one of `ends` (an instruction's id and the operand slot
    it reads it in) or passes it on to one: through bitcasts, each moving no byte, and, `inward`, into the
    computation of a fusion or an async call taking it, as its parameter at its slot, which that
    computation returns nowhere (a fusion returning its operand passes it to what reads its result)."""
    passing = [value.id]
    seen: set[int] = set()
    returned: set[int] = set()
    while passing:
        if (at := passing.pop()) in seen:
            continue
        if at in returned:
            return False
        seen.add(at)
        for reader, slot in module.readers.get(at, ()):
            if (reader.id, slot) in ends:
                continue
            if reader.opcode == "bitcast":
                passing.append(reader.id)
            elif inward and reader.opcode in ("fusion", "async-start"):
                # An async call's operands are its start's output too: read through it by its own done
                # alone, no update between them (review of row 12463).
                if reader.opcode == "async-start" and [
                        user.opcode for user, _ in module.readers.get(reader.id, ())] != ["async-done"]:
                    return False
                called = module.computations[reader.called_computation_ids[0]]
                returned.add(called.root_id)
                passing.append(next(each.id for each in called.instructions
                                    if each.opcode == "parameter" and each.parameter_number == slot))
            else:
                return False
    return True


def _value_named(instruction, instructions, computations) -> tuple[str | None, str]:
    """The checkpoint name of the value `instruction` makes, and the op that names or makes it: its own,
    else, through tuples, get-tuple-elements, the calls XLA kept and fusions, the nearest instruction
    naming the value whole, or its maker's own (`_projected`), as a buffer it defines is named (a scan's
    row a fusion or a jitted call makes, written into its stack: row 12463). The op is the first of them
    stating one, the instruction's, the naming one's, the maker's: the scope a remat rung reads it by."""
    made, within, namer = _projected(instruction, (), instructions, computations,
                                     entered=frozenset({"fusion", "call"}))
    named = instruction.frontend_attributes.map.get("checkpoint_name")
    if not named and namer is not None:
        named = namer.frontend_attributes.map.get("checkpoint_name")
    if not named and _names_output(made, within):
        named = made.frontend_attributes.map.get("checkpoint_name")
    ops = (instruction, *((namer,) if namer is not None else ()), made)
    return named or None, next((op.metadata.op_name for op in ops if op.metadata.op_name), "")


def _bitcast_of(instruction, instructions):
    """The value `instruction` views through its bitcasts, each moving no byte; `instruction` where it is
    none."""
    while instruction.opcode == "bitcast":
        instruction = instructions[instruction.operand_ids[0]]
    return instruction


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
        called, evaluate):
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


def _loop(initial, called, evaluate):
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
    if xla_gemm(instruction.custom_call_target):
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
