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
# Modified by Dew: ported from tests/peak_live_test.py, read as Dew's own
# (the JAX fork's audit of 2026-11-26, its C01/C02).
"""What the live-buffer tests share: a module's assignments built by hand and their readings
(tests/test_live_buffers.py, tests/test_gpu_live_buffers.py)."""

import collections
from typing import Any

import jax
import jax.numpy as jnp
from google.protobuf.message import Message
from jax import ad_checkpoint

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


def scanned_hidden(xs, w):
    """A scan over the rows `xs` whose body saves its named residual "hidden", tanh(x @ w), alone, summing
    sin of it into its carry: its gradient's stack of that residual is a loop state element its body writes
    a row a step."""
    def body(carry, x):
        hidden = ad_checkpoint.checkpoint_name(jnp.tanh(x @ w), "hidden")
        return carry + jnp.sum(jnp.sin(hidden)), None

    saved = jax.checkpoint(body, policy=jax.checkpoint_policies.save_only_these_names("hidden"))
    return jax.lax.scan(saved, 0.0, xs)[0]
