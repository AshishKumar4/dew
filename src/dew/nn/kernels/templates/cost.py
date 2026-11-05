"""The kernel cost vocabulary (`Work`) and the template's tile walk over it (`walked`), from a GPU's facts
(`generation.GpuTarget`): docs/design/algebra.md §6.5, kernel-templates.md's toolkit. A program of a tile
walk does tensor-core FLOPs, stages bytes through shared memory (its operand tiles written there and read
back by its matrix instructions), evaluates special-function exponentials and does fp32 operations on the
CUDA cores (the algebra between its products); its units overlap, so its time is the longest unit's at its
rate (`walked`, ties broken by the units in series, `dew.algebra.cost.Walk`). The tensor cores' rate is the
generation's (`GpuTarget.tensor_rate`), the shared-memory and special-function rates its architecture's
(`generation.Architecture.shared_bytes_per_clock`, `exponentials_per_clock`), the CUDA cores' the
generation's fp32 rate. Nothing ranks a launch's configs by it (bible §6: configs are derived); the
derivation prices a form a template's algebra takes by it (`dew.algebra.derive`, `dew.algebra.carry`)."""


import dataclasses
import functools
import math

from dew.nn.kernels import generation


@dataclasses.dataclass(frozen=True)
class Work:
    """The cost vocabulary every price is told in (docs/design/algebra.md §6.5): an operation's work by unit
    (tensor-core FLOPs by mode: the 16-bit mode's, 16-bit operands summed in fp32, TF32's, fp32 operands
    at the default precision, and FP8's, `GpuTarget.tensor_rate`; CUDA-core fp32 operations, a product at
    `Precision.HIGHEST`'s FLOPs among them; special-function exponentials; conversions, which run on the
    CUDA cores at their rate until a target states one of their own) and its bytes by level (device memory,
    shared memory), and its span, the dependent steps on its critical path (a loop's iterations, each reading
    the last). The algebra's terms and a template's instructions both price through it; a schedule turns it
    into seconds by the work-span model (Brent's): the larger of its work at its units' rates and its span
    at a dependent step's latency, XLA's fusion's (`dew.algebra.cost.fused`) or a template's tile walk
    (`walked`)."""

    flops: float = 0.0
    operations: float = 0.0
    exponentials: float = 0.0
    conversions: float = 0.0
    device_bytes: float = 0.0
    shared_bytes: float = 0.0
    tf32_flops: float = 0.0
    span: float = 0.0
    fp8_flops: float = 0.0

    def __add__(self, other: "Work") -> "Work":
        # Works summed side by side: their span the longer one's; a chain of works' span is its path's sum,
        # which the term that chains them states (`dew.algebra.linear_scan`).
        return Work(self.flops + other.flops, self.operations + other.operations,
                    self.exponentials + other.exponentials, self.conversions + other.conversions,
                    self.device_bytes + other.device_bytes, self.shared_bytes + other.shared_bytes,
                    self.tf32_flops + other.tf32_flops, max(self.span, other.span),
                    self.fp8_flops + other.fp8_flops)


@functools.cache
def rates(target: generation.GpuTarget, *, device: bool) -> tuple[float | None, ...]:
    """`target`'s rate of each unit `units` prices, per second: the tensor cores' 16-bit, TF32 and FP8 modes
    (FP8's the 16-bit one's where the generation states none, its codes unpacked into 16-bit products), the
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

    def rate(per_clock: float | None) -> float | None:
        return None if per_clock is None or each is None else per_clock * each

    tensor = rate(target.tensor_rate("16bit"))
    if tensor is None and device:
        tensor = target.bf16_peak
    eight = rate(target.tensor_rate("8bit")) or tensor
    return (tensor, rate(target.tensor_rate("tf32")), eight, rate(target.tensor_rate("fp32")),
            rate(target.architecture.exponentials_per_clock),
            math.inf if device else rate(target.architecture.shared_bytes_per_clock),
            target.memory_rate if device else math.inf)


def units(work: Work, at: tuple[float | None, ...]) -> tuple[list[float], tuple[float, ...]]:
    """`work`'s seconds on each unit at the rates `at` (`rates`'): the tensor cores (their 16-bit, TF32 and
    FP8 modes in turn), the CUDA cores (operations and conversions), the special-function units, shared
    memory and device memory; and, mode by mode, the work at a rate the target states none of, which no time
    prices (0 where one does)."""
    tensor, tf32, eight, cuda, special, shared, device = at
    grouped = (((work.flops, tensor), (work.tf32_flops, tf32), (work.fp8_flops, eight)),
               ((work.operations + work.conversions, cuda),),
               ((work.exponentials, special),), ((work.shared_bytes, shared),),
               ((work.device_bytes, device),))
    priced = [sum(amount / per_second for amount, per_second in unit if amount and per_second is not None)
              for unit in grouped]
    unpriced = tuple(amount if per_second is None else 0.0 for unit in grouped for amount, per_second in unit)
    return priced, unpriced


def spanned(work: Work, latency: float | None) -> tuple[float, float]:
    """`work`'s span at a dependent step's `latency` (`Work.span`), and its span at no latency stated
    (`GpuTarget.step_seconds` None), which no time prices (0 where one does)."""
    if not work.span:
        return 0.0, 0.0
    return (0.0, work.span) if latency is None else (work.span * latency, 0.0)


def step_latency(target: generation.GpuTarget, *, device: bool) -> float | None:
    """A dependent step's latency on a schedule's path: XLA's loop iteration (`GpuTarget.step_seconds`) for
    the whole device's; a template's in-kernel step from its instructions' latencies, which no fact states
    yet, so none."""
    return target.step_seconds if device else None


def walked(work: Work, target: generation.GpuTarget) -> float:
    """The seconds a program of a template's tile walk spends on `work` by the work-span model: its units'
    work overlapped, each at its rate on one multiprocessor, the longest unit's time (`units`), or its span
    at an in-kernel step's latency where longer (`spanned`); its device bytes the launch's, which a form of
    its algebra does not change."""
    return max(*units(work, rates(target, device=False))[0],
               spanned(work, step_latency(target, device=False))[0])
