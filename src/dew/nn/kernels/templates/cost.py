"""The kernel cost vocabulary (`Work`) and the template's tile walk over it (`walked`), from a GPU's facts
(`generation.GpuTarget`): docs/design/algebra.md §6.5, kernel-templates.md's toolkit. A program of a tile
walk does tensor-core FLOPs, stages bytes through shared memory (its operand tiles written there and read
back by its matrix instructions), evaluates special-function exponentials and does fp32 operations on the
CUDA cores (the algebra between its products); its units overlap, so its time is the longest unit's at its
rate (`walked`, ties broken by the units in series, `dew.algebra.cost.Walk`). The tensor cores' rate is the
generation's (`GpuTarget.tensor_rate`), the shared-memory and special-function rates the CUDA
architecture's own (`SHARED_BYTES_PER_CLOCK`, `EXPONENTIALS_PER_CLOCK`), the CUDA cores' the generation's
fp32 rate. Nothing ranks a launch's configs by it (bible §6: configs are derived); the derivation prices a
form a template's algebra takes by it (`dew.algebra.derive`, `dew.algebra.carry`). The register and
shared-memory constants the resource model reads live here beside it (`resources`)."""


import dataclasses
import functools
import math

from dew.nn.kernels import generation

SHARED_BYTES_PER_CLOCK = 128
"""Shared memory a multiprocessor serves per clock: 32 banks of 4 bytes, on every CUDA generation Dew's
kernels run on (CUDA C++ Programming Guide, "Shared Memory", compute capabilities 5.x to 12.x)."""

EXPONENTIALS_PER_CLOCK = 16
"""Base-2 exponentials a multiprocessor's special-function units return per clock (CUDA C++ Programming
Guide, "Arithmetic Instructions" throughput table, compute capabilities 5.x to 12.x)."""

REGISTER_STEP = 8
"""Registers a thread in one allocation unit (256 registers a warp)."""


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
class Work:
    """The cost vocabulary every price is told in (docs/design/algebra.md §6.5): an operation's work by unit
    (tensor-core FLOPs by mode: the 16-bit mode's, 16-bit operands summed in fp32, and TF32's, fp32 operands
    at the default precision, `GpuTarget.tensor_rate`; CUDA-core fp32 operations, a product at
    `Precision.HIGHEST`'s FLOPs among them; special-function exponentials; conversions, which run on the
    CUDA cores at their rate until a target states one of their own) and its bytes by level (device memory,
    shared memory). The algebra's terms and a template's instructions both price through it; a schedule
    turns it into seconds: XLA's fusion's (`dew.algebra.cost.fused`) or a template's tile walk
    (`walked`)."""

    flops: float = 0.0
    operations: float = 0.0
    exponentials: float = 0.0
    conversions: float = 0.0
    device_bytes: float = 0.0
    shared_bytes: float = 0.0
    tf32_flops: float = 0.0

    def __add__(self, other: "Work") -> "Work":
        return Work(self.flops + other.flops, self.operations + other.operations,
                    self.exponentials + other.exponentials, self.conversions + other.conversions,
                    self.device_bytes + other.device_bytes, self.shared_bytes + other.shared_bytes,
                    self.tf32_flops + other.tf32_flops)


@functools.cache
def rates(target: generation.GpuTarget, *, device: bool) -> tuple[float | None, ...]:
    """`target`'s rate of each unit `units` prices, per second: the tensor cores' 16-bit and TF32 modes, the
    CUDA cores, the special-function units, shared memory and device memory. Each is a multiprocessor's at
    the target's clock, or with `device` the whole device's (its multiprocessors at its clock, the tensor
    cores at its datasheet's dense peak where it states no clock), where a tile walk's device bytes and XLA's
    fusion's staged bytes cost nothing (infinite); None for a unit the target states no rate of (a TPU's
    vector and transcendental units, a datasheet's missing clock)."""
    clock = target.clock_hz
    if not device:
        each = clock
    else:
        each = clock * target.multiprocessors if clock and target.multiprocessors else None
    # The special-function and shared-memory rates are a CUDA multiprocessor's (a TPU has neither).
    cuda = generation.capability(target.generation) is not None

    def rate(per_clock: float | None) -> float | None:
        return None if per_clock is None or each is None else per_clock * each

    tensor = rate(target.tensor_rate("16bit"))
    if tensor is None and device:
        tensor = target.bf16_peak
    return (tensor, rate(target.tensor_rate("tf32")), rate(target.tensor_rate("fp32")),
            rate(EXPONENTIALS_PER_CLOCK if cuda else None),
            math.inf if device else rate(SHARED_BYTES_PER_CLOCK if cuda else None),
            target.memory_bandwidth if device else math.inf)


def units(work: Work, at: tuple[float | None, ...]) -> tuple[list[float], tuple[float, ...]]:
    """`work`'s seconds on each unit at the rates `at` (`rates`'): the tensor cores (their 16-bit and TF32
    modes in turn), the CUDA cores (operations and conversions), the special-function units, shared memory and
    device memory; and, mode by mode, the work at a rate the target states none of, which no time prices (0
    where one does)."""
    tensor, tf32, cuda, special, shared, device = at
    grouped = (((work.flops, tensor), (work.tf32_flops, tf32)), ((work.operations + work.conversions, cuda),),
               ((work.exponentials, special),), ((work.shared_bytes, shared),),
               ((work.device_bytes, device),))
    priced = [sum(amount / per_second for amount, per_second in unit if amount and per_second is not None)
              for unit in grouped]
    unpriced = tuple(amount if per_second is None else 0.0 for unit in grouped for amount, per_second in unit)
    return priced, unpriced


def walked(work: Work, target: generation.GpuTarget) -> float:
    """The seconds a program of a template's tile walk spends on `work`: its units' work overlapped, each at
    its rate on one multiprocessor, the longest unit's time (`units`); its device bytes the launch's, which a
    form of its algebra does not change."""
    return max(units(work, rates(target, device=False))[0])
