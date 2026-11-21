"""The kernel cost vocabulary (`Work`) and the template's tile walk over it (`walked`), from a GPU's facts
(`generation.GpuTarget`): docs/design/algebra.md §6.5, kernel-templates.md's toolkit. A program of a tile walk
does tensor-core FLOPs, stages bytes through shared memory (its operand tiles written there and read back by
its matrix instructions), evaluates special-function exponentials and does fp32 operations on the vector units
(the CUDA cores', a TPU's VPU: the algebra between its products); its units overlap, so its time is the
longest unit's at its rate (`walked`, ties broken by the units in series, `dew.algebra.cost.Walk`). The tensor
cores' rate is the generation's (`GpuTarget.tensor_rate`), the vector and special-function rates its
architecture's (`generation.Architecture.vector_per_clock`, `exponentials_per_clock`), shared memory's its
multiprocessor's (`generation.Multiprocessor.shared_bytes_per_clock`). Nothing ranks a launch's configs by it
(bible §6: configs are derived); the derivation prices a form a template's algebra takes by it
(`dew.algebra.derive`, `dew.algebra.carry`)."""


import dataclasses
import functools
import math
from typing import NamedTuple

from jax.experimental import pallas as pl

from dew.nn.kernels import generation


def estimate(flops: float, *arrays, transcendentals: float = 0) -> pl.CostEstimate:
    """A launch's cost as it states it to XLA and to Dew's FLOP count (the custom call's cost estimate,
    `dew.telemetry.instrumentation.sizes`): its `flops`, its `transcendentals`, and the bytes each of
    `arrays`, its operands and outputs, takes in memory once (`stored`)."""
    return pl.CostEstimate(flops=int(flops), transcendentals=int(transcendentals),
                           bytes_accessed=stored(*arrays))


def stored(*arrays) -> int:
    """The bytes `arrays` take in memory, each once (`resources.stored_bytes`: a packed dtype's two elements a
    byte)."""
    from dew.nn.kernels.templates import resources

    return int(sum(resources.stored_bytes(array.shape, array.dtype) for array in arrays))


@dataclasses.dataclass(frozen=True)
class Stated:
    """What a launch states of its cost beside its kernel's emission, which counts the rest
    (`emission.costed`): the steps of its walks, every consumer's over the call's batch and heads (`steps`,
    attention's `walks`, the recurrence's `launch_kernel`'s: a laid call's mean chunks a sequence over
    its sequences), the trips its data decides, which the count multiplies a step's work by; and the bytes it
    reads and writes, each of its operands and outputs once (`bytes_accessed`, `estimate`'s); and the most
    steps one of its programs walks (`longest`, its critical path) where its programs walk unequal shares (a
    causal walk's tiles), None where each walks their mean."""

    steps: float
    bytes_accessed: int
    longest: float | None = None


@dataclasses.dataclass(frozen=True)
class Chain:
    """One dependent path through a tile walk, counted, its latencies a part's (`chained`): its dependent
    reductions (`span`, each `GpuTarget.kernel_step_seconds`), the warpgroup products it waits on (`products`)
    and their columns (`columns`, each a 16-bit step of 64 of their depth), its warpgroups' barrier round
    trips (`barriers`), and the tile copies it waits on (`copies`, their `copy_bytes`) through a ring of
    `stages`, whose S - 1 steps ahead hide what of a copy a step outlasts. A walk of K such steps counts K
    times each and keeps its stages."""

    span: float = 0.0
    products: float = 0.0
    columns: float = 0.0
    barriers: float = 0.0
    copies: float = 0.0
    copy_bytes: float = 0.0
    stages: int = 1

    def then(self, other: "Chain") -> "Chain":
        """This path followed by `other`: their events one after another, waited through the one ring a walk's
        loops share (the shallower one's where both wait on copies through rings apart, which hides less)."""
        stages = (min(self.stages, other.stages) if self.copied and other.copied
                  else self.stages if self.copied else other.stages)
        return Chain(**{name: getattr(self, name) + getattr(other, name) for name in _COUNTED}, stages=stages)

    @property
    def copied(self) -> bool:
        """Whether the path waits on a tile copy."""
        return bool(self.copies or self.copy_bytes)

    def times(self, count: float) -> "Chain":
        """This path walked `count` times: each event's count multiplied, its ring the same."""
        return dataclasses.replace(self, **{name: getattr(self, name) * count for name in _COUNTED})

    def outlasts(self, other: "Chain") -> bool:
        """Whether this path takes at least `other`'s time at every part's latencies: at least as many of each
        event, and its copies through no more stages, unless `other` waits on none."""
        counted = all(getattr(self, name) >= getattr(other, name) for name in _COUNTED)
        return counted and (self.stages <= other.stages or not other.copied)


_COUNTED = ("span", "products", "columns", "barriers", "copies", "copy_bytes")


def _kept(chains: tuple[Chain, ...]) -> tuple[Chain, ...]:
    """`chains` but each another outlasts (`Chain.outlasts`), in their order, one of equal chains kept: the
    longest of them at any part's latencies is the longest of these, so the pruning prices nothing less."""
    kept: list[Chain] = []
    for chain in chains:
        if any(held.outlasts(chain) for held in kept):
            continue
        kept = [held for held in kept if not chain.outlasts(held)] + [chain]
    return tuple(kept)


def followed(first: tuple[Chain, ...], then: tuple[Chain, ...]) -> tuple[Chain, ...]:
    """The paths of `first`'s work followed by `then`'s, run one after the other: each of the one's paths
    then each of the other's (`Chain.then`), each another outlasts left out; either's where the other has
    none."""
    if not (first and then):
        return first or then
    return _kept(tuple(one.then(other) for one in first for other in then))


_NOT_ISSUED = {"issued": False}
"""A `Work` field no kernel body issues instruction by instruction (its device bytes, span, dependent paths,
fused kernels and kept bytes): every other field is a unit a body issues (`ISSUED_UNITS`)."""


@dataclasses.dataclass(frozen=True)
class Work:
    """The cost vocabulary every price is told in (docs/design/algebra.md §6.5): an operation's work by unit
    (tensor-core FLOPs by mode: the 16-bit mode's, 16-bit operands summed in fp32, TF32's, fp32 operands at
    the default precision, and FP8's, `GpuTarget.tensor_rate`; CUDA-core fp32 operations, a product at
    `Precision.HIGHEST`'s FLOPs among them; special-function exponentials; conversions, which run on the CUDA
    cores at their rate until a target states one of their own) and its bytes by level (device memory, shared
    memory, and what XLA's fusion keeps on chip between its kernels, `kept_bytes`), its span, the dependent
    steps on its critical path (a loop's iterations, each reading the last), and the kernels XLA's fusion
    makes of it (`kernels`: a reduction's and a product's, each a fusion's root), each paying the part's fixed
    latency in XLA's schedule (`dew.algebra.cost.Fused`). The algebra's terms and a template's instructions
    both price through it; a schedule turns it into seconds by the work-span model (Brent's): the larger of
    its work at its units' rates and its span at a dependent step's latency, XLA's fusion's
    (`dew.algebra.cost.fused`) or a template's tile walk (`walked`). A walk's dependent paths, counted by the
    events they wait on (`chains`, `Chain`), price at the part's latencies where it measured them
    (`chained`)."""

    flops: float = 0.0
    operations: float = 0.0
    exponentials: float = 0.0
    conversions: float = 0.0
    device_bytes: float = dataclasses.field(default=0.0, metadata=_NOT_ISSUED)
    shared_bytes: float = 0.0
    tf32_flops: float = 0.0
    span: float = dataclasses.field(default=0.0, metadata=_NOT_ISSUED)
    fp8_flops: float = 0.0
    kernels: float = dataclasses.field(default=0.0, metadata=_NOT_ISSUED)
    kept_bytes: float = dataclasses.field(default=0.0, metadata=_NOT_ISSUED)
    """The bytes XLA's fusion moves between the vector registers and the on-chip memory its kernels keep their
    operands and results in (`dew.algebra.cost.work`'s: a leaf's read once, a reduction's and a product's
    result written once and read once): a TPU's VMEM, at its rate (`GpuTarget.vmem_rate`); a GPU's on-chip
    copies, which no rate prices (`rates`)."""
    integer_operations: float = 0.0
    """32-bit integer operations a CUDA multiprocessor's ALU issues (add, shift, logic, comparison), beside
    its fp32 lanes at its own rate (`Architecture.integer_per_clock`)."""
    chains: tuple[Chain, ...] = dataclasses.field(default=(), metadata=_NOT_ISSUED)

    def __add__(self, other: "Work") -> "Work":
        # Works summed side by side: their span the longer one's; a chain of works' span is its path's sum,
        # which the term that chains them states (`dew.algebra.linear_scan`).
        summed = {field.name: getattr(self, field.name) + getattr(other, field.name)
                  for field in dataclasses.fields(Work) if field.name not in ("span", "chains")}
        # Each path no other outlasts, the side with none's kept as the other's.
        chains = (_kept(self.chains + other.chains) if self.chains and other.chains
                  else self.chains or other.chains)
        return Work(**summed, span=max(self.span, other.span), chains=chains)


ISSUED_UNITS = tuple(field.name for field in dataclasses.fields(Work) if field.metadata.get("issued", True))
"""The units of `Work` a kernel's body issues, which its emission counts instruction by instruction: every
field not marked otherwise (`_NOT_ISSUED`). A launch's work from its emission takes each
(`emission.launch_work`), and a launch's programs share each alike beside its device bytes (a program's
share of a launch's work), so a unit added to `Work` is counted and shared wherever these are with no list
to update (row 10793)."""


class Rates(NamedTuple):
    """A target's rate of each unit `units` prices, per second (`rates`), read by name: the tensor cores'
    16-bit, TF32 and FP8 modes, the CUDA cores, the special-function units, shared memory, the on-chip memory
    XLA's fusion keeps its kernels' operands and results in, the integer units (a CUDA multiprocessor's
    ALU) and device memory; None where the target states none. Named, so a unit added among them misreads
    no consumer (the review of train 1074, F5)."""

    tensor: float | None
    tf32: float | None
    eight: float | None
    cuda: float | None
    special: float | None
    shared: float | None
    kept: float | None
    integer: float | None
    device: float | None


@functools.cache
def rates(target: generation.GpuTarget, *, device: bool) -> Rates:
    """`target`'s rate of each unit `units` prices, per second: the tensor cores' 16-bit, TF32 and FP8 modes
    (FP8's the 16-bit one's where the generation states none, its codes unpacked into 16-bit products), the
    vector units (the CUDA cores, a TPU's VPU), the special-function units, shared memory, the on-chip memory
    XLA's fusion keeps its kernels' operands and results in (`Work.kept_bytes`), the integer units (a CUDA
    multiprocessor's ALU, `Architecture.integer_per_clock`) and device memory. Each is a multiprocessor's at
    the clock the target holds under a step's tensor-core load (`GpuTarget.loaded_clock_hz`: a B200's
    measured 1.549 GHz, not its 1.965 GHz boost), or with `device` the whole device's (its multiprocessors at
    that clock, the tensor cores at its datasheet's dense peak where it states no clock), where a tile walk's
    device bytes and XLA's fusion's staged bytes cost nothing (infinite). What XLA's fusion keeps on chip is a
    TPU's VMEM traffic, with `device` at its VMEM rate (`GpuTarget.vmem_rate`), and nothing a GPU's or a tile
    walk's (infinite); a TPU tile walk's staged bytes are its VMEM's too, at that rate (its shared memory's
    slot). None for a unit the target states no rate of (a TPU's vector and transcendental units, an
    unmeasured TPU's VMEM, a datasheet's missing clock, an integer rate unread)."""
    clock = target.loaded_clock_hz
    if not device:
        each = clock
    else:
        each = clock * target.multiprocessors if clock and target.multiprocessors else None

    def rate(per_clock: float | None) -> float | None:
        return None if per_clock is None or each is None else per_clock * each

    # A TPU has no CUDA multiprocessor, nor its shared memory's rate: its one VMEM is the on-chip memory a
    # Pallas template stages its tiles in (`Work.shared_bytes`), at the chip's VMEM rate (its one core a tile
    # walk's multiprocessor, `multiprocessors=1`), where it was work no rate priced and a TPU template's
    # forms were compared without it (row 8729); an unmeasured TPU's is none.
    architecture, held, tpu = target.architecture, target.architecture.multiprocessor, generation.tpu(target)
    tensor = rate(target.tensor_rate("16bit"))
    if tensor is None and device:
        tensor = target.bf16_peak
    eight = rate(target.tensor_rate("8bit")) or tensor
    kept = target.vmem_rate if device and tpu else math.inf
    staged = target.vmem_rate if tpu else rate(None if held is None else held.shared_bytes_per_clock)
    return Rates(tensor=tensor, tf32=rate(target.tensor_rate("tf32")), eight=eight,
                 cuda=rate(architecture.vector_per_clock), special=rate(architecture.exponentials_per_clock),
                 shared=math.inf if device else staged, kept=kept,
                 integer=rate(architecture.integer_per_clock),
                 device=target.memory_rate if device else math.inf)


def units(work: Work, at: Rates) -> tuple[list[float], tuple[float, ...]]:
    """`work`'s seconds on each unit at the rates `at` (`rates`'): the tensor cores (their 16-bit, TF32 and
    FP8 modes in turn), the CUDA cores (operations and conversions), the special-function units, on-chip
    memory (shared memory's bytes, then what XLA's fusion keeps there, one memory's two levels, so a TPU's
    one VMEM is one unit), device memory and the integer units; and, level by level, the work at a rate the
    target states none
    of, which no time prices (0 where one does)."""
    grouped = (((work.flops, at.tensor), (work.tf32_flops, at.tf32), (work.fp8_flops, at.eight)),
               ((work.operations + work.conversions, at.cuda),),
               ((work.exponentials, at.special),),
               ((work.shared_bytes, at.shared), (work.kept_bytes, at.kept)),
               ((work.device_bytes, at.device),), ((work.integer_operations, at.integer),))
    priced = [sum((amount / per_second for amount, per_second in unit if amount and per_second is not None),
                  0.0) for unit in grouped]
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
    the whole device's; a template's in-kernel step, a tile's dependent reduction within a program
    (`GpuTarget.kernel_step_seconds`), for a tile walk's. None where the GPU's was not measured."""
    return target.step_seconds if device else target.kernel_step_seconds


def path(work: Work, target: generation.GpuTarget) -> float:
    """The seconds of `work`'s longest dependent path in a template's walk: its span at an in-kernel step's
    latency (`spanned`) or its longest chain at the part's latencies (`chained`), the larger; the one peak
    every price of a walk reads, the greedy (`walked`) and the algebra's exact extractor's
    (`dew.algebra.cost.Walk.spanned`) alike."""
    return max(spanned(work, step_latency(target, device=False))[0], chained(work, target))


def chained(work: Work, target: generation.GpuTarget) -> float:
    """The longest of `work`'s paths (`Work.chains`) at `target`'s part's latencies: a path's events waited
    one after another, C, and its copies, L, waited past what its ring hides, max(C, (L + C) / stages); an
    event the part measured no latency of priced by none (`GpuTarget.product_latency`, `barrier_seconds`,
    `copy_latency`, `kernel_step_seconds`)."""
    step = target.kernel_step_seconds or 0.0
    fixed, per_column = target.product_latency or (0.0, 0.0)
    barrier = target.barrier_seconds or 0.0
    copied, per_byte = target.copy_latency or (0.0, 0.0)

    def seconds(chain: Chain) -> float:
        waited = (chain.span * step + chain.products * fixed + chain.columns * per_column
                  + chain.barriers * barrier)
        return max(waited, (chain.copies * copied + chain.copy_bytes * per_byte + waited) / chain.stages)

    return max(map(seconds, work.chains), default=0.0)


def walked(work: Work, target: generation.GpuTarget) -> float:
    """The seconds a program of a template's tile walk spends on `work` by the work-span model: its units'
    work overlapped, each at its rate on one multiprocessor, the longest unit's time (`units`), or its span
    at an in-kernel step's latency where longer (`spanned`); its device bytes the launch's, which a form of
    its algebra does not change; or its longest dependent path where longer (`path`)."""
    return max(*units(work, rates(target, device=False))[0], path(work, target))


def seconds(work: Work, target: generation.GpuTarget) -> float:
    """The seconds a step of launches one after another spends on `work` on `target` by the work-span model:
    its units' work overlapped at the whole device's rates (`units`), or its span at an in-kernel step's
    latency where longer (`spanned`), and each of its kernels' launch (`GpuTarget.launch_seconds`, none
    where unmeasured): a server's price of a step (`attention_decode.priced_step`), where `walked` is one
    program's."""
    launch = target.launch_seconds or 0.0
    worked = max(*units(work, rates(target, device=True))[0], path(work, target))
    return worked + work.kernels * launch
