"""The seconds a template's launch takes at a config, estimated from the GPU's facts and the launch's
geometry: the cost every template's `cost` returns (`dew.nn.kernels.templates.base`).

A launch runs `programs` programs. Each does `program_flops` tensor-core FLOPs, moves `program_staged`
bytes through shared memory (its operand tiles written there and read back by its matrix
instructions) and evaluates `program_exponentials` exponentials; the launch as a whole reads and writes
`device_bytes` of device memory, each byte once: the programs' reads of one tile again are served by
the L2 cache, whose bandwidth no device or datasheet states, and an H100's grouped launches ranked so
(GMM-h100-3, where counting every reread from device memory tied every tile at the memory bound). A
multiprocessor holds `resident` programs at once, which share its tensor cores, shared memory and
special-function units. So, per multiprocessor,

    program  = max(program_flops / tensor rate, program_staged / shared-memory rate,
                   program_exponentials / special-function rate)
    compute  = (full waves · resident + programs of the last wave per multiprocessor) · program
    memory   = device_bytes / memory bandwidth
    seconds  = max(compute, memory)

with a last wave short of every multiprocessor charging only the programs it has. A persistent launch
is `resident` 1 with its work items as programs. Every rate is a fact of the target
(`dew.nn.kernels.generation.GpuTarget`): its multiprocessors and clock, its memory bandwidth, and its
generation's dense tensor FLOPs per clock per multiprocessor in the launch's arithmetic mode; the
shared-memory and special-function rates and the register file are the CUDA architecture's own
(`SHARED_BYTES_PER_CLOCK`, `EXPONENTIALS_PER_CLOCK`, `REGISTERS`). A fact the target lacks raises,
naming it: no rate is made up.
"""

import dataclasses
import math

from dew.nn.kernels import generation

SHARED_BYTES_PER_CLOCK = 128
"""Shared memory a multiprocessor serves per clock: 32 banks of 4 bytes, on every CUDA generation Dew's
kernels run on (CUDA C++ Programming Guide, "Shared Memory", compute capabilities 5.x to 12.x)."""

EXPONENTIALS_PER_CLOCK = 16
"""Base-2 exponentials a multiprocessor's special-function units return per clock (CUDA C++ Programming
Guide, "Arithmetic Instructions" throughput table, compute capabilities 5.x to 12.x)."""

REGISTERS = 64 * 1024
"""32-bit registers per multiprocessor, the same 64 K from compute capability 5.0 to 12.x (CUDA C++
Programming Guide, "Technical Specifications per Compute Capability")."""

THREAD_REGISTERS = 255
"""32-bit registers one thread may hold, compute capabilities 5.0 to 12.x (the same table); what a
kernel needs past them it spills to memory."""

RESERVED_SHARED = 1024
"""Shared memory the driver reserves for each resident block, from compute capability 8.0 on (CUDA C++
Programming Guide, "Shared Memory" of compute capabilities 8.x and 9.0): a multiprocessor holds the
opted-in block size plus one reservation."""


@dataclasses.dataclass(frozen=True)
class Cost:
    """One launch's work at one config, as the model reads it (the module's docstring). `mode` names
    the tensor cores' arithmetic ('16bit' or 'tf32', `GpuTarget.tensor_rates`) and is read only where
    `program_flops` is nonzero."""
    programs: float
    device_bytes: float
    program_flops: float = 0.0
    program_staged: float = 0.0
    program_exponentials: float = 0.0
    resident: int = 1
    mode: str = '16bit'


def resident(target: generation.GpuTarget, staged: int, threads: int, registers: int) -> int:
    """How many programs, each staging `staged` bytes of shared memory and running `threads` threads of
    `registers` registers each, one multiprocessor of `target` holds at once; at least one, the launch
    being fitted to a block (`Template.fitting`)."""
    shared = generation.shared_budget(target) + RESERVED_SHARED
    by_shared = shared // (staged + RESERVED_SHARED)
    by_registers = REGISTERS // (threads * min(registers, THREAD_REGISTERS))
    return max(1, min(by_shared, by_registers))


def seconds(cost: Cost, target: generation.GpuTarget) -> float:
    """The seconds `cost`'s launch is estimated to take on `target` (the module's docstring)."""
    facts = _facts(target)
    rate = facts.tensor_rate(cost.mode) if cost.program_flops else 1
    named = (('multiprocessors', facts.multiprocessors), ('clock_hz', facts.clock_hz),
             (f'tensor_rates[{cost.mode!r}]', rate), ('memory_bandwidth', facts.memory_bandwidth))
    missing = [name for name, value in named if value is None]
    if missing:
        raise ValueError(f"{facts.generation} ({facts.device_kind}) states no {', '.join(missing)}, "
                         "which the kernel cost model needs")
    clock = facts.clock_hz
    program = max(cost.program_flops / (rate * clock),
                  cost.program_staged / (SHARED_BYTES_PER_CLOCK * clock),
                  cost.program_exponentials / (EXPONENTIALS_PER_CLOCK * clock))
    wave = facts.multiprocessors * cost.resident
    full, rest = divmod(cost.programs, wave)
    waves = full * cost.resident + math.ceil(rest / facts.multiprocessors)
    return max(waves * program, cost.device_bytes / facts.memory_bandwidth)


def _facts(target: generation.GpuTarget) -> generation.GpuTarget:
    """The target whose facts a launch on `target` is costed by: its own on a GPU, and the A100's on a
    CPU, which interprets the kernels as an A100's block (`generation.shared_budget`)."""
    return target if generation.capability(target.generation) is not None else generation.TARGETS['sm80']
