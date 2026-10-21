"""The seconds a template's launch takes at a config, estimated from the GPU's facts and the launch's
geometry: the cost every template's `cost` returns (`dew.nn.kernels.templates.base`).

A launch runs `programs` programs. Each does `program_flops` tensor-core FLOPs, moves `program_staged`
bytes through shared memory (its operand tiles written there and read back by its matrix
instructions) and evaluates `program_exponentials` exponentials; the launch as a whole reads and writes
`device_bytes` of device memory, rereads counted where the launch's order evicts them from the L2
before they come again (each template's `cost` says which). Programs a multiprocessor holds at once
share its tensor cores, shared memory and special-function units, so however many it holds, it does
its programs' work one after another; per multiprocessor,

    program  = max(program_flops / tensor rate, program_staged / shared-memory rate,
                   program_exponentials / special-function rate)
    compute  = ceil(programs / multiprocessors) · program
    memory   = device_bytes / memory bandwidth
    seconds  = max(compute, memory)

A persistent launch's work items are its programs. Every rate is a fact of the target
(`dew.nn.kernels.generation.GpuTarget`): its multiprocessors and clock, its memory bandwidth, and its
generation's dense tensor FLOPs per clock per multiprocessor in the launch's arithmetic mode; the
shared-memory and special-function rates are the CUDA architecture's own (`SHARED_BYTES_PER_CLOCK`,
`EXPONENTIALS_PER_CLOCK`). A fact the target lacks raises,
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

THREAD_REGISTERS = 255
"""32-bit registers one thread may hold, compute capabilities 5.0 to 12.x (CUDA C++ Programming Guide,
"Technical Specifications per Compute Capability"); what a kernel needs past them it spills to memory."""


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
    mode: str = '16bit'


def seconds(cost: Cost, target: generation.GpuTarget) -> float:
    """The seconds `cost`'s launch is estimated to take on `target` (the module's docstring)."""
    facts = _facts(target)
    multiprocessors, clock, bandwidth = facts.multiprocessors, facts.clock_hz, facts.memory_bandwidth
    rate = facts.tensor_rate(cost.mode) if cost.program_flops else 1.0
    if multiprocessors is None or clock is None or bandwidth is None or rate is None:
        named = (('multiprocessors', multiprocessors), ('clock_hz', clock),
                 (f'tensor_rates[{cost.mode!r}]', rate), ('memory_bandwidth', bandwidth))
        raise ValueError(f"{facts.generation} ({facts.device_kind}) states no "
                         f"{', '.join(name for name, value in named if value is None)}, "
                         "which the kernel cost model needs")
    program = max(cost.program_flops / (rate * clock),
                  cost.program_staged / (SHARED_BYTES_PER_CLOCK * clock),
                  cost.program_exponentials / (EXPONENTIALS_PER_CLOCK * clock))
    return max(math.ceil(cost.programs / multiprocessors) * program, cost.device_bytes / bandwidth)


def _facts(target: generation.GpuTarget) -> generation.GpuTarget:
    """The target whose facts a launch on `target` is costed by: its own on a GPU, and the A100's on a
    CPU, which interprets the kernels as an A100's block (`generation.shared_budget`)."""
    return target if generation.capability(target.generation) is not None else generation.TARGETS['sm80']
