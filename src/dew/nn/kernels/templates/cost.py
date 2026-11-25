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
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

from jax.experimental import pallas as pl

from dew.nn.kernels import generation

if TYPE_CHECKING:
    import numpy as np


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
    integer_multiply_adds: float = 0.0
    """32-bit integer multiplies and multiply-adds (IMAD, which a bit form's left shift by a multiply is too,
    `dew.algebra.bits.BitForm.work`), issued on the FMA pipe beside the fp32 lanes' FFMA at their own rate
    (`Architecture.integer_multiply_adds_per_clock`), their seconds the fp32 lanes' unit's (`units`)."""
    unpriced_operations: float = 0.0
    """Elementwise operations no rule prices (`emission.operation_work`: an atan2's, an erf_inv's, which no
    kernel's body holds and no probe has measured), one an output element, at no rate (`units`): a form
    taking more of them is never cheaper (`dew.algebra.cost.cheaper`), its price never a guess."""
    fused_multiply_adds: float = 0.0
    """The adds the assembler fuses into the multiply each alone reads (FFMA, ptxas's --fmad, its default),
    an element each, their multiplies among `operations`: a body's multiply whose one use is an add of its
    shape (`emission`), a product's on the CUDA cores (`Precision.HIGHEST`'s FLOPs, an FFMA two,
    `multiplied`). A CUDA multiprocessor's lanes run each within its multiply's instruction, a TPU's VPU as
    an operation of its own (`Rates.fused`); counted apart, so every field stays work a unit issues and a
    branch's or a role's most of each unit is never another's credit (the review of lane/toolkit
    5cbbe66c586)."""
    issue_excess: float = 0.0
    """The issue slots `issued` counts of a most taken unit by unit (a conditional's branches', `emission`'s
    envelope; its walking roles', a launch's) beyond the most any one of them issues: none of them issues
    another's mix (the review of lane/toolkit c9307c11146), so `issued` takes it off."""
    shuffles: float = 0.0
    """The lane instructions exchanging values among a warp's lanes (SHFL: an inline body's `shfl.sync`, its
    PTX's `shfl.`), each an issue slot a lane (`issued`); the shuffle pipe's own rate, a quarter of a
    multiprocessor's issue (32 lanes a clock), unpriced. A kernel's reductions' butterflies are not counted
    here: the lanes a row spans are its layout's, which its jaxpr does not state (row 12889)."""
    global_reductions: float = 0.0
    """The bytes a body reduces into global memory an element at a time through L2's atomic units
    (`red.global`: mma.sync's dQ adds, `atomic_store_p`), each its own memory instruction a lane, at the
    rate its multiprocessor's requests reach them (`Architecture.global_reduction_bytes_per_clock`); a TMA
    reduction (`copy_smem_to_gmem`'s `add`) is a copy's bytes, not these."""
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
    ALU), its FMA pipe's integer multiply-adds, device memory, the element-wise reductions into global
    memory and a multiprocessor's schedulers' issue slots, and the adds fused into multiplies (none of a CUDA
    multiprocessor's time, infinite; a TPU's at its vector units'); None where the target states none. Named,
    so a unit added among them misreads no consumer (the review of train 1074, F5)."""

    tensor: float | None
    tf32: float | None
    eight: float | None
    cuda: float | None
    special: float | None
    shared: float | None
    kept: float | None
    integer: float | None
    multiply_adds: float | None
    device: float | None
    reductions: float | None
    issue: float | None
    fused: float | None


@functools.cache
def rates(target: generation.GpuTarget, *, device: bool) -> Rates:
    """`target`'s rate of each unit `units` prices, per second: the tensor cores' 16-bit, TF32 and FP8 modes
    (FP8's the 16-bit one's where the generation states none, its codes unpacked into 16-bit products), the
    vector units (the CUDA cores, a TPU's VPU), the special-function units, shared memory, the on-chip memory
    XLA's fusion keeps its kernels' operands and results in (`Work.kept_bytes`), the integer units (a CUDA
    multiprocessor's ALU, `Architecture.integer_per_clock`), its FMA pipe's integer multiply-adds
    (`Architecture.integer_multiply_adds_per_clock`), device memory, the element-wise reductions into
    global memory (`Architecture.global_reduction_bytes_per_clock`) and the instructions its schedulers
    issue (`Multiprocessor.issue_per_clock`). A CUDA multiprocessor's vector units are its fp32 lanes, an
    instruction a lane a clock (`Architecture.vector_per_clock` their FLOPs, two a lane). Each is a
    multiprocessor's at
    the clock the target holds under a step's tensor-core load (`GpuTarget.loaded_clock_hz`: a B200's
    measured 1.549 GHz, not its 1.965 GHz boost), or with `device` the whole device's (its multiprocessors at
    that clock, the tensor cores at its datasheet's dense peak where it states no clock), where a tile walk's
    device bytes and XLA's fusion's staged bytes cost nothing (infinite), and the issue slots (each kernel of
    XLA's fusion runs its units in series, `dew.algebra.cost.Fused`, which no slot shares), as on a target of
    no CUDA multiprocessor. What XLA's fusion keeps on chip is a
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
    vector = architecture.vector_per_clock
    lanes = vector / 2 if vector is not None and held is not None else vector
    issue = math.inf if device or held is None else rate(held.issue_per_clock)
    return Rates(tensor=tensor, tf32=rate(target.tensor_rate("tf32")), eight=eight,
                 cuda=rate(lanes), special=rate(architecture.exponentials_per_clock),
                 shared=math.inf if device else staged, kept=kept,
                 integer=rate(architecture.integer_per_clock),
                 multiply_adds=rate(architecture.integer_multiply_adds_per_clock),
                 device=target.memory_rate if device else math.inf,
                 reductions=rate(architecture.global_reduction_bytes_per_clock), issue=issue,
                 fused=rate(vector) if held is None else math.inf)


def units(work: Work, at: Rates) -> tuple[list[float], tuple[float, ...]]:
    """`work`'s seconds on each unit at the rates `at` (`rates`'): the tensor cores (their 16-bit, TF32 and
    FP8 modes in turn), the CUDA cores (their instructions, operations and conversions; the adds fused into
    a multiply, `Work.fused_multiply_adds`, at their own rate; and the integer multiply-adds their FMA pipe
    issues beside them at its own rate), the special-function units, on-chip memory (shared memory's bytes,
    then what XLA's
    fusion keeps there, one memory's two levels, so a TPU's one VMEM is one unit), device memory, the integer
    units, the element-wise reductions into global memory, the issue slots every instruction of those units
    takes (`issued`) and the operations no rule prices (`Work.unpriced_operations`, at no rate); and, level
    by level, the work at a rate the target states none of, which no time prices (0 where one does)."""
    grouped = (((work.flops, at.tensor), (work.tf32_flops, at.tf32), (work.fp8_flops, at.eight)),
               ((work.operations + work.conversions, at.cuda), (work.fused_multiply_adds, at.fused),
                (work.integer_multiply_adds, at.multiply_adds)),
               ((work.exponentials, at.special),),
               ((work.shared_bytes, at.shared), (work.kept_bytes, at.kept)),
               ((work.device_bytes, at.device),), ((work.integer_operations, at.integer),),
               ((work.global_reductions, at.reductions),), ((issued(work), at.issue),),
               ((work.unpriced_operations, None),))
    priced = [sum((amount / per_second for amount, per_second in unit if amount and per_second is not None),
                  0.0) for unit in grouped]
    unpriced = tuple(amount if per_second is None else 0.0 for unit in grouped for amount, per_second in unit)
    return priced, unpriced



def repeated(work: Work, count: float) -> Work:
    """`work` repeated `count` times in a chain: its every unit's, its span and each of its paths'; its
    kernels its own (a loop's trips run in one), `emission`'s and `inline_work`'s alike."""
    return Work(**{field.name: getattr(work, field.name) * count for field in dataclasses.fields(Work)
                   if field.name not in ("kernels", "chains")}, kernels=work.kernels,
                chains=tuple(chain.times(count) for chain in work.chains))

def most(works: Sequence[Work]) -> Work:
    """The most of each unit any of `works` holds, alternatives of which one runs (a conditional's branches,
    `emission`'s and `inline_work`'s; none, `Work()`): its span the longest, every path none of the others
    outlasts (as works side by side keep them), and its issue slots the most any one of them issues
    (`Work.issue_excess`), none issuing another's mix."""
    if not works:
        return Work()
    taken = Work(**{field.name: max(getattr(work, field.name) for work in works)
                    for field in dataclasses.fields(Work) if field.name not in ("chains", "issue_excess")},
                 chains=sum(works, Work()).chains)
    return dataclasses.replace(taken, issue_excess=max(0.0, issued(taken) - max(map(issued, works))))

def latency_bound(work: Work, at: Rates, target: generation.GpuTarget, *, held: float) -> float:
    """`work`'s issue slots (`issued`) at the rate `at` takes them, when a multiprocessor's schedulers hold
    `held` warps issuing it: a warp issues once a `Multiprocessor.issue_interval` clocks, so a scheduler of
    fewer warps than that interval issues that share of its clocks (`issue_interval / warps` the issue
    slots' seconds over, M-NA-21's one warp a scheduler 4.56); 0 where the target states no interval or no
    warps are held."""
    multiprocessor = target.architecture.multiprocessor
    if (multiprocessor is None or multiprocessor.issue_interval is None or held <= 0
            or at.issue in (None, math.inf)):
        return 0.0
    interval = multiprocessor.issue_interval
    # Each scheduler issues one warp's instruction a clock, 32 threads' (`Multiprocessor.issue_per_clock`).
    schedulers = multiprocessor.issue_per_clock / 32
    return issued(work) / at.issue * max(1.0, interval / (held / schedulers))


def multiplied(field: str, flops: float) -> Work:
    """A product's `flops` in the unit its mode takes (`product.product_mode`'s field): on the CUDA cores
    (`operations`, `Precision.HIGHEST`'s fp32), each multiply an operation and its add fused into it, an FFMA
    a pair (`Work.fused_multiply_adds`)."""
    if field != "operations":
        return dataclasses.replace(Work(), **{field: flops})
    return Work(operations=flops / 2, fused_multiply_adds=flops / 2)

def issued(work: Work) -> float:
    """The instructions `work`'s threads issue on a CUDA multiprocessor's units, each taking one of its
    schedulers' slots (`Multiprocessor.issue_per_clock`): its fp32 lanes' (`units`' CUDA cores'), its integer
    ALU's and FMA pipe's, its special-function units', its shuffles and its reductions into global memory
    (one a lane, four bytes each), less what a most taken unit by unit counts beyond any one alternative's
    (`Work.issue_excess`); its tensor-core and memory instructions, each a warp's work of many elements, are
    none."""
    return (work.operations + work.conversions + work.integer_operations + work.integer_multiply_adds
            + work.exponentials + work.shuffles + work.global_reductions / 4 - work.issue_excess)

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


def wave(multiprocessors: int | None, resident: int) -> int:
    """The programs one wave of a launch runs at once: `resident` on each of `multiprocessors`, one where no
    fact states them (a target's, `GpuTarget.multiprocessors`)."""
    return (multiprocessors or 1) * max(resident, 1)


def spread[Count: (int, "np.ndarray")](programs: Count, over: int) -> Count:
    """How many of `programs` the busiest of `over` runs, as a launch's blocks go to its multiprocessors in
    turn (or its waves, `over` a wave's programs: how many waves it runs); each of an array of counts so."""
    return -(-programs // over)


def sharing(programs: int, resident: int, target: generation.GpuTarget) -> int:
    """The programs of a launch of `programs`, `resident` a multiprocessor at most, each multiprocessor of
    `target` runs at once: no more than its programs spread over the multiprocessors where it knows them
    (`spread`); `resident` where it knows none."""
    if target.multiprocessors is None:
        return resident
    return min(resident, spread(programs, target.multiprocessors))


def busiest_programs(programs: int, wave: int, multiprocessors: int) -> int:
    """The programs the busiest of `multiprocessors` runs over a launch of `programs`, `wave` of them at once:
    each wave's share of a full wave (`spread` of `min(n, wave)`), the last priced at a full wave's sharing.
    Priced at `ceil(n / SMs)` instead, four sm80 kernel gradients' split counts halved and ran 0.6-11.5%
    slower (M-FS-60, row 10865), where this one's ran as priced (M-EM-45b)."""
    count = max(programs, 1)
    return spread(count, wave) * spread(min(count, wave), multiprocessors)


def streamed(per_program_bytes: float, busiest: int, bandwidth: float, multiprocessors: int) -> float:
    """The seconds the busiest multiprocessor streams its `busiest` programs' `per_program_bytes` each from
    device memory at its share of `bandwidth`, its programs' shares in turn: a launch of fewer programs
    than multiprocessors streams each at its multiprocessor's share, not the device's whole rate
    (M-EM-45b-a100: priced at the whole rate, an A100's decode product took a split that ran 1.14x slower)."""
    return busiest * per_program_bytes * multiprocessors / bandwidth


def launch_seconds(work: Work, programs: int, target: generation.GpuTarget, *,
                   resident: int = 1, warps: float = 0.0) -> float:
    """One kernel launch's seconds on `target` for `programs` programs at `resident` programs per
    multiprocessor, each issuing over `warps` warps (`emission.warps_of`; 0 where unknown): the longer of
    the busiest multiprocessor's throughput work (`ceil(n / SMs)` programs sharing one multiprocessor's
    units), its overlapped latency waves (`ceil(n / wave)` waves of `path`), plus launch latency, against
    the whole launch's device bound (`seconds`). Co-resident programs share one multiprocessor's throughput
    units rather than multiplying its peak by `resident`, while overlapping their dependent paths (rows
    6216, 10809, 10810). Its issue slots take a warp's issue interval (`Multiprocessor.issue_interval`) over
    the warps each scheduler holds, where fewer warps than that interval leave it idle between a warp's
    instructions (`latency_bound`: rows 5274, 4123)."""
    count = max(programs, 1)
    shared = {unit: getattr(work, unit) / count for unit in (*ISSUED_UNITS, "device_bytes")}
    per_program = dataclasses.replace(work, **shared, kernels=0)
    per_sm = spread(count, target.multiprocessors or 1)
    waves = spread(count, wave(target.multiprocessors, resident))
    at = rates(target, device=False)
    throughput = max(units(per_program, at)[0])

    def resident_wave(held: int) -> float:
        # `held` programs at once on a multiprocessor: their units shared, their warps hiding each other's
        # issue intervals.
        return held * max(throughput, latency_bound(per_program, at, target, held=held * warps))

    # Its full waves at the residency it holds, then its last at the programs left (the review of
    # lane/toolkit 61dd3c7d2: three programs at two resident are a wave of two and one of one).
    full, left = divmod(per_sm, max(1, min(resident, per_sm)))
    busiest = full * resident_wave(min(resident, per_sm)) + (resident_wave(left) if left else 0.0)
    wave_seconds = max(busiest, waves * path(per_program, target))
    return max(wave_seconds + (target.launch_seconds or 0.0), seconds(work, target))
