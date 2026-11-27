"""What the live-buffer tests share: a module's assignments built by hand and their readings
(tests/test_live_buffers.py, tests/test_gpu_live_buffers.py)."""

import collections
from typing import Any

from google.protobuf.message import Message

from dew.telemetry import hlo
from dew.training.live_buffers import Assignment, provenance


def module_bytes(text: str) -> bytes:
    """HLO module `text`, parsed by XLA's own parser, serialized."""
    return hlo.module_from_text(text).as_serialized_hlo_module_proto()


def instruction_ids(module: bytes, computation: str | None = None) -> dict[str, Any]:
    """Each instruction of serialized `module` (of its `computation` alone where named), by name."""
    proto: Any = hlo.message_class("xla.HloModuleProto").FromString(module)
    return {instruction.name: instruction for each in proto.computations
            if computation is None or each.name == computation for instruction in each.instructions}


def empty_assignment() -> Message:
    """An empty `xla.BufferAssignmentProto` a test fills."""
    return hlo.message_class("xla.BufferAssignmentProto")()


def read(assignment: Message, module: bytes) -> Assignment:
    return Assignment(assignment.SerializeToString(), module)


def descended(text: str) -> dict[str, Any]:
    """The provenance of each instruction of HLO module `text`, by name."""
    module: Any = hlo.module_proto(module_bytes(text))
    found = provenance(module)
    return {instruction.name: found[instruction.id] for computation in module.computations
            for instruction in computation.instructions if instruction.id in found}


def of(*pairs: tuple[int, int]) -> frozenset[tuple[int, int]]:
    return frozenset(pairs)


def held(intervals, at: int) -> int:
    """The bytes `intervals`, of one trace, hold at position `at`: each storage some interval covering it
    lies in, once, at its members' most."""
    storages: dict[int, int] = {}
    for each in intervals:
        if each.start <= at < each.end:
            storages[each.storage] = max(storages.get(each.storage, 0), each.bytes)
    return sum(storages.values())


def counted(intervals, at: int) -> int:
    """The bytes XLA's count of a peak charges `intervals`, of one trace, at position `at`."""
    return sum(each.bytes for each in intervals if each.charged and each.start <= at < each.end)


def temporaries(buffers) -> collections.Counter:
    """`buffers`' temporaries by their instruction and size."""
    return collections.Counter((buffer.instruction, buffer.bytes) for buffer in buffers
                               if buffer.kind == "temporary")
