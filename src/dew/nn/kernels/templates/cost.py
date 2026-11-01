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
    (tensor-core FLOPs in the 16-bit mode, CUDA-core fp32 operations, special-function exponentials,
    conversions, which run on the CUDA cores at their rate until a target states one of their own) and its
    bytes by level (device memory, shared memory). The algebra's terms and a template's instructions both
    price through it; a schedule turns it into seconds: XLA's fusion's (`dew.algebra.cost.fused`) or a
    template's tile walk (`walked`)."""

    flops: float = 0.0
    operations: float = 0.0
    exponentials: float = 0.0
    conversions: float = 0.0
    device_bytes: float = 0.0
    shared_bytes: float = 0.0

    def __add__(self, other: "Work") -> "Work":
        pairs = zip(dataclasses.astuple(self), dataclasses.astuple(other), strict=True)
        return Work(*(mine + theirs for mine, theirs in pairs))


def walked(work: Work, target: generation.GpuTarget) -> float:
    """The seconds a program of a template's tile walk spends on `work`: its units' work overlapped, each at
    its rate on one multiprocessor, the longest unit's time; its device bytes the launch's, which a form of
    its algebra does not change."""
    clock = target.clock_hz
    tensor = target.tensor_rate("16bit") if work.flops else 1.0
    operations = target.tensor_rate("fp32")
    if clock is None or tensor is None or operations is None:
        named = (("clock_hz", clock), ("tensor_rates['16bit']", tensor), ("tensor_rates['fp32']", operations))
        missing = ", ".join(name for name, value in named if value is None)
        raise ValueError(f"{target.generation} ({target.device_kind}) states no {missing}, which the kernel "
                         "cost model needs")
    return max(work.flops / (tensor * clock), work.shared_bytes / (SHARED_BYTES_PER_CLOCK * clock),
               work.exponentials / (EXPONENTIALS_PER_CLOCK * clock),
               (work.operations + work.conversions) / (operations * clock))
