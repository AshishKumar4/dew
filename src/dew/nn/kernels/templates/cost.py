"""The seconds a template's launch takes at a config, estimated from the GPU's facts and the launch's
geometry: the cost every template's `cost` returns (`dew.nn.kernels.templates.base`).

A launch runs `programs` programs. Each does `program_flops` tensor-core FLOPs, moves `program_staged`
bytes through shared memory (its operand tiles written there and read back by its matrix
instructions) and evaluates `program_exponentials` exponentials; the launch as a whole reads and writes
`device_bytes` of device memory, rereads counted where the launch's order evicts them from the L2
before they come again (each template's `cost` says which). Programs a multiprocessor holds at once
share its tensor cores, shared memory and special-function units, so it does their work one after
another; and while one waits for its next step's operands, the others' work hides the wait. A program
of `steps` pipeline steps, `stages` of them in flight, issues each step's loads `stages - 1` steps
before it needs them, so each step waits at least the global load latency over `stages - 1`; a
multiprocessor's `resident` programs (`resident`) wait together (Little's law). A mainloop that keeps
`in_flight` steps' products running past their step holds their stages too, so it loads `stages - 1 -
in_flight` ahead, and with none ahead each step's wait and its transfer (its staged bytes at the
program's share of the bandwidth) are serial with its work. Per multiprocessor,

    program  = max(program_flops / tensor rate, program_staged / shared-memory rate,
                   program_exponentials / special-function rate)
             + (serial_staged + steps · spilled_bytes) / shared-memory rate
             + serial_exponentials / special-function rate
    waiting  = steps · load latency / min(stages - 1 - in_flight, steps)
    tile     = program + serial_stored / (memory bandwidth / multiprocessors)
    compute  = ceil(programs / (multiprocessors · resident))
               · max(min(resident, ceil(programs / multiprocessors)) · tile, waiting)
    memory   = device_bytes / memory bandwidth
    l2       = l2_bytes / L2 bandwidth
    seconds  = max(compute, memory, l2) + programs * steps * grid step seconds (TPU's)

`l2_bytes` are the bytes all programs stage from the L2 together: a tile every program re-reads (the
keys and values each query block of attention walks, the operand bands each output tile of a GEMM
reads) is served there, not from device memory, at the L2's rate, so a config of more, smaller programs
stages more. A template states them; one that states none is ranked without the term.

Work a program does between its matrix instructions, which they wait for, is serial: `serial_staged` bytes
and `serial_exponentials` exponentials it adds to its service time rather than overlapping (a Mosaic
mainloop's prologue, rewriting each stage in shared memory before the warpgroup MMA reads it). So are the
`serial_stored` bytes a persistent program stores before its next tile's loads go out, at its
multiprocessor's share of the device's bandwidth: a tile that builds its own pipeline has none of the next
tile's stages in flight while its store drains, which at a kernel gradient's few steps a tile is most of
the tile's time (GMMCOOP-h100's sweep), where a streamed tile loads the next one's stages during its own.
A persistent launch's work items are its programs. Every rate is a fact of the target
(`dew.nn.kernels.generation.GpuTarget`): its multiprocessors and clock, its memory bandwidth, and its
generation's dense tensor FLOPs per clock per multiprocessor in the launch's arithmetic mode; the
shared-memory and special-function rates are the CUDA architecture's own (`SHARED_BYTES_PER_CLOCK`,
`EXPONENTIALS_PER_CLOCK`); the load latency is the generation's, measured (`GpuTarget.load_latency`).
A fact the target lacks raises, naming it: no rate is made up. A generation whose load latency no
measurement states is costed without the waiting term, its launches ranked by their work alone.
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

REGISTERS = 64 * 1024
"""32-bit registers per multiprocessor, compute capabilities 5.0 to 12.x (the same table)."""

RESERVED_SHARED = 1024
"""Shared memory the driver reserves for each resident block, compute capabilities 8.0 to 12.x (CUDA C++
Programming Guide, "Shared Memory" of compute capability 8.x): a multiprocessor holds the opted-in block
size plus one reservation."""


@dataclasses.dataclass(frozen=True)
class Cost:
    """One launch's work at one config, as the model reads it (the module's docstring). `mode` names
    the tensor cores' arithmetic ('16bit' or 'tf32', `GpuTarget.tensor_rates`) and is read only where
    `program_flops` is nonzero. `single_warp`: each scheduler issues one warp's products in order with the
    fragment loads feeding them, which reach the family's `GpuTarget.single_warp_issue` of its rate.
    `in_flight`: the steps whose products a program keeps running past their step (a mainloop's MMAs in
    flight), each holding its stage, so `stages - 1 - in_flight` steps load ahead. `spilled_bytes`: the
    local-memory bytes a program's spilled registers move a step (`base.Template.spilled`), serial with
    its work through the L1, which serves them at shared memory's rate."""
    programs: float
    device_bytes: float
    program_flops: float = 0.0
    program_staged: float = 0.0
    program_exponentials: float = 0.0
    mode: str = '16bit'
    steps: float = 0.0
    stages: int = 2
    resident: int = 1
    serial_staged: float = 0.0
    serial_exponentials: float = 0.0
    l2_bytes: float = 0.0
    serial_stored: float = 0.0
    single_warp: bool = False
    in_flight: int = 0
    spilled_bytes: float = 0.0


def seconds(cost: Cost, target: generation.GpuTarget) -> float:
    """The seconds `cost`'s launch is estimated to take on `target` (the module's docstring)."""
    facts = generation.effective(target)
    multiprocessors, clock, bandwidth = facts.multiprocessors, facts.clock_hz, facts.memory_bandwidth
    rate = facts.tensor_rate(cost.mode) if cost.program_flops else 1.0
    if rate is not None and cost.single_warp and facts.single_warp_issue is not None:
        rate *= facts.single_warp_issue
    if multiprocessors is None or clock is None or bandwidth is None or rate is None:
        named = (('multiprocessors', multiprocessors), ('clock_hz', clock),
                 (f'tensor_rates[{cost.mode!r}]', rate), ('memory_bandwidth', bandwidth))
        raise ValueError(f"{facts.generation} ({facts.device_kind}) states no "
                         f"{', '.join(name for name, value in named if value is None)}, "
                         "which the kernel cost model needs")
    latency = facts.load_latency
    program = program_seconds(cost, rate, clock) + cost.serial_stored * multiprocessors / bandwidth
    waiting = 0.0 if latency is None else waiting_seconds(cost, latency, clock)
    if cost.stages - 1 - cost.in_flight < 1:
        # No step loads ahead: each waits for its own load, its transfer at the program's share of the
        # memory's bandwidth, serial with the products (H100 PCIe sweeps: a two-stage warpgroup-MMA tile,
        # which loads nothing ahead, ran 1.16 to 1.94 times its three-stage twin, R1/R3-sweep-h100).
        program += waiting + cost.program_staged * multiprocessors / bandwidth
        waiting = 0.0
    waves = math.ceil(cost.programs / (multiprocessors * cost.resident))
    # A multiprocessor holds no more programs at once than a wave gives it.
    resident = min(cost.resident, max(math.ceil(cost.programs / multiprocessors), 1))
    l2 = 0.0 if facts.l2_bandwidth is None else cost.l2_bytes / facts.l2_bandwidth
    step = facts.grid_step_seconds
    stepping = 0.0 if step is None else cost.programs * cost.steps * step
    return max(waves * max(resident * program, waiting), cost.device_bytes / bandwidth, l2) + stepping


def program_seconds(cost: Cost, rate: float, clock: float) -> float:
    """One program's service time on a multiprocessor: its tensor-core, shared-memory or special-function
    work, whichever takes longest at `rate` tensor FLOPs a clock and `clock` Hz, plus its serial work (the
    module's docstring)."""
    return (max(cost.program_flops / (rate * clock),
                cost.program_staged / (SHARED_BYTES_PER_CLOCK * clock),
                cost.program_exponentials / (EXPONENTIALS_PER_CLOCK * clock))
            + (cost.serial_staged + cost.steps * cost.spilled_bytes) / (SHARED_BYTES_PER_CLOCK * clock)
            + cost.serial_exponentials / (EXPONENTIALS_PER_CLOCK * clock))


def waiting_seconds(cost: Cost, latency: float, clock: float) -> float:
    """The seconds a program's steps wait on loads `latency` clocks long, the steps loading ahead
    (`stages - 1 - in_flight`) hiding as many steps of it, and never more steps than it takes (Little's
    law, the module's docstring)."""
    ahead = cost.stages - 1 - cost.in_flight
    return cost.steps * latency / max(min(ahead, math.ceil(cost.steps)), 1) / clock


def resident(target: generation.GpuTarget, staged: int, threads: int, registers: int) -> int:
    """How many programs, each staging `staged` bytes of shared memory and running `threads` threads of
    `registers` registers each, one multiprocessor of `target` holds at once: at least one, the launch
    fitted to a block (`Template.fitting`)."""
    facts = generation.effective(target)
    shared = generation.shared_budget(facts) + RESERVED_SHARED
    by_shared = shared // (staged + RESERVED_SHARED)
    by_registers = REGISTERS // (threads * min(registers, THREAD_REGISTERS))
    return max(1, min(by_shared, by_registers))

