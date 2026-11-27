"""A compiled HLO module read through XLA's own schema: its instructions (`instructions`), and its serialized
records (a module, a buffer assignment) as XLA's messages (`message_class`).

The schema is XLA's hlo.proto and what it imports, at the XLA commit `XLA_COMMIT` names, held as one
serialized FileDescriptorSet beside this module (`xla_protos.binpb`, written by tools/xla_protos.py) in a
descriptor pool of its own (`pool`): another library's copy of these files (xla/xla_data.proto) in protobuf's
default pool does not meet it. A module is decoded once per read; protobuf is imported on first use.
"""

from __future__ import annotations

import functools
import importlib
import importlib.resources
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

if TYPE_CHECKING:
    from google.protobuf.descriptor_pool import DescriptorPool
    from google.protobuf.message import Message

XLA_COMMIT = "2b26bc3432b9e01ba65a66ce2a858031b8a63e8d"
"""The openxla/xla commit whose xla/xla_data.proto, xla/service/metrics.proto and xla/service/hlo.proto
`xla_protos.binpb` describes (tools/xla_protos.py)."""


class HloModule(Protocol):
    """An HLO module as XLA's bindings hold one (a compiled executable's `hlo_modules()`, a lowering's
    `compiler_ir("hlo")`, `module_from_text`'s)."""

    def as_serialized_hlo_module_proto(self) -> bytes: ...


class Instruction(NamedTuple):
    """An instruction of a module (`instructions`): the computation holding it, its name and opcode, its
    custom-call target (None where it is no custom call), its operands and the computations it calls (a
    fusion's, a while loop's body and condition, a conditional's branches, a call's), each by name, whether
    it is its computation's root and whether its computation is the module's entry, and the instruction as
    XLA's schema holds it (`proto`, an `xla.HloInstructionProto`: its shape, dimension numbers and backend
    config, which the fields above do not name)."""

    computation: str
    name: str
    opcode: str
    custom_call_target: str | None
    operands: tuple[str, ...]
    called: tuple[str, ...]
    root: bool
    entry: bool
    proto: Any
    """An `xla.HloInstructionProto`, a message class `pool` makes, which no static type names."""


@functools.cache
def pool() -> DescriptorPool:
    """XLA's schema (`XLA_COMMIT`) in a descriptor pool of its own."""
    from google.protobuf import descriptor_pb2, descriptor_pool

    held = descriptor_pool.DescriptorPool()
    described = importlib.resources.files(__package__).joinpath("xla_protos.binpb").read_bytes()
    for file in descriptor_pb2.FileDescriptorSet.FromString(described).file:
        held.Add(file)
    return held


@functools.cache
def message_class(name: str) -> type[Message]:
    """The message class of XLA's message `name` (`xla.BufferAssignmentProto`), from `pool`."""
    from google.protobuf import message_factory

    return message_factory.GetMessageClass(pool().FindMessageTypeByName(name))


def module_from_text(text: str) -> HloModule:
    """The HLO module `text` writes (an XLA dump file, a test's module), parsed by XLA's own HLO parser, which
    `instructions` reads as it reads a compiled one."""
    # jaxlib's binding of XLA's parser (`jaxlib._hlo`), which no typed module states.
    return importlib.import_module("jaxlib._hlo").hlo_module_from_text(text)


def module_proto(module: HloModule | bytes) -> Message:
    """`module` as XLA's `xla.HloModuleProto`, decoded once: an HLO module (a compiled executable's
    `hlo_modules()`, a lowering's `compiler_ir("hlo")`, `module_from_text`'s) or its serialization."""
    serialized = module if isinstance(module, bytes) else module.as_serialized_hlo_module_proto()
    return message_class("xla.HloModuleProto").FromString(serialized)


def instructions(module: HloModule | bytes) -> Iterator[Instruction]:
    """Each instruction of `module` (`module_proto`'s), computation by computation in the module's order."""
    proto: Any = module_proto(module)
    names = {computation.id: computation.name for computation in proto.computations}
    for computation in proto.computations:
        operands = {instruction.id: instruction.name for instruction in computation.instructions}
        entry = computation.id == proto.entry_computation_id
        for instruction in computation.instructions:
            yield Instruction(
                computation.name, instruction.name, instruction.opcode,
                instruction.custom_call_target if instruction.opcode == "custom-call" else None,
                tuple(operands[operand] for operand in instruction.operand_ids),
                tuple(names[called] for called in instruction.called_computation_ids),
                instruction.id == computation.root_id, entry, instruction)
