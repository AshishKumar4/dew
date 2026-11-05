"""The hardware generation every kernel selector in Dew keys on.

One key, read from the default device: `smXY` from a GPU's compute
capability, `v5e`/`v5p`/`v6e`/`v4` from a TPU's device kind, and the
platform name otherwise. Each generation's facts (`GpuTarget`, its
`Architecture`) are what a selector and a template's derivation read: a GPU
no row names takes its family's facts, so a selector admits an unseen member
of a known family by what its family takes, never by a list of measured
generations.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import logging
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax.sharding import AbstractMesh, Mesh

    from dew.nn.precision import Numerics

    from .selection import Refusal

_log = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class Assumption:
    """The device `assumed` stands in for the default device's: its generation, and its facts (`target`)
    where it was given them. Equal wherever the generation and every fact are (`_facts`), as JAX's trace
    key reads it, not by a target's identity or its partial hash."""
    generation: str
    target: GpuTarget | None = dataclasses.field(default=None, compare=False, hash=False)
    facts: Hashable = None


def assumption() -> Assumption | None:
    """The device `assumed` stands in for in this thread, None outside it: one value, which JAX's trace
    and compile caches key on (`dew.cache.KERNEL_ASSUMED`) and every choice reads."""
    from dew import cache

    return cache.KERNEL_ASSUMED.value


@contextlib.contextmanager
def assumed(generation: str | GpuTarget | None) -> Iterator[None]:
    """Within the block, choose kernels as on a device of `generation` ('sm80'), whatever device this
    process has: `device_generation`, `bf16_dot_runs` and `triton_runs` answer for it, so every
    kernel choice and refusal acts as on that GPU while a step traces and lowers for one on
    a CPU (`dew.training._lowering`). Given a target's facts (a GPU no row names, its multiprocessors,
    shared memory, clock and bandwidth its own), `gpu_target` answers with them, so every launch is sized
    and costed by them alone. None assumes nothing. JAX keys its trace and compile caches on the device
    assumed (`dew.cache.KERNEL_ASSUMED`), so a function jitted for one traces again for another."""
    from dew import cache

    if isinstance(generation, GpuTarget):
        value = Assumption(generation.generation, generation, _facts(generation))
    else:
        value = None if generation is None else Assumption(generation)
    with cache.KERNEL_ASSUMED(value):
        yield


def _facts(value: Hashable | Mapping | list) -> Hashable:
    """`value`'s facts as a hashable tree, equal wherever the facts are: a dataclass's fields by name, a
    mapping's entries sorted by key, a sequence's items in order."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, *((field.name, _facts(getattr(value, field.name)))
                                       for field in dataclasses.fields(value)))
    if isinstance(value, Mapping):
        return tuple(sorted((key, _facts(entry)) for key, entry in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_facts(element) for element in value)
    return value


CHOICES = frozenset({'attention', 'attention_mainloop', 'decode', 'delta_decode', 'delta_gates', 'gate_up',
                     'gated_delta_rule', 'gemm_epilogue', 'gemm_prologue', 'grouped_matmul', 'narrow_copies',
                     'paged_decode', 'rms_norm', 'selective_scan', 'short_conv', 'ssd', 'swiglu'})
"""The choices a dispatch reports (`route`), and so the ones `chosen` may name (`chosen_kernel`): every one a
selector serves (`selection.Op`, which refuses a choice this does not hold as it is built) or a site reads
(`chosen_kernel`, likewise)."""


def known_choices(choices: Iterable[str]) -> None:
    """Refuse any of `choices` that `CHOICES` does not hold, naming every one it holds."""
    if unknown := set(choices) - CHOICES:
        raise ValueError(f"no kernel choice is named {', '.join(sorted(unknown))}; "
                         f"the choices are {', '.join(sorted(CHOICES))}")


@contextlib.contextmanager
def chosen(**kernels: str) -> Iterator[None]:
    """Within the block, a choice named in `kernels` runs that kernel whatever its selector would
    take, for an A/B of two kernels or a test of a kernel the selector does not take on this device.
    Nested blocks keep the outer block's other choices and restore its choices on exit. JAX keys its
    trace and compile caches on the choices (`dew.cache.KERNEL_CHOICES`), so a function jitted
    outside the block traces again within it."""
    from dew import cache

    known_choices(kernels)
    with cache.KERNEL_CHOICES(tuple(sorted({**dict(cache.KERNEL_CHOICES.value), **kernels}.items()))):
        yield


def numerics() -> Numerics:
    """The run's numerics in effect (`run_precision`), the defaults outside any: JAX keys its trace and
    compile caches on them (`dew.cache.NUMERICS`), so a function jitted under one run's traces again
    under another's."""
    from dew import cache
    from dew.nn.precision import Numerics

    held = cache.NUMERICS.value
    return Numerics() if held is None else held


@contextlib.contextmanager
def run_precision(policy: Numerics) -> Iterator[None]:
    """Within the block, programs trace under a run's numerics' products and forms
    (`dew.nn.precision.Numerics`): every product that states no precision at `policy.matmul`
    (`jax.default_matmul_precision` through `configured_precision`; "default" leaves JAX's configuration
    as it stands), and each op's accurate form where its forms are "accurate", the attention template's
    among them, which reads them here (`numerics`, the one seam of a run's forms). The dtypes are the
    model's and the optimizer's own fields, which the run sets from the numerics: a bf16 operand has
    lost its bits before any product reads it, whatever its precision."""
    from dew import cache
    from dew.nn.precision import configured_precision

    with configured_precision(policy.matmul), cache.NUMERICS(policy):
        yield


PALLAS_CHOICES = frozenset({'pallas', 'fla'})
"""The kernel names that run Dew's Pallas or Triton kernels, which lower through JAX's Pallas
Triton backend for a GPU only: what the CPU dry run lowers (`dew.training._lowering`) and the kernel
lab verifies on the GPU. 'flash' and 'cudnn' name CUDA libraries."""


@dataclasses.dataclass(frozen=True)
class RegisterRaises:
    """The register counts a warpgroup's `setmaxnreg` moves a thread's allocation to: from `lowest` to
    `highest` in steps of `step` (PTX ISA, "setmaxnreg": a count in 24 to 256, a multiple of 8, on sm_90a,
    sm_100f and sm_120f)."""

    lowest: int
    highest: int
    step: int


SETMAXNREG = RegisterRaises(24, 256, 8)
"""`setmaxnreg`'s range on every architecture that takes it (PTX ISA, "setmaxnreg")."""


@dataclasses.dataclass(frozen=True)
class Multiprocessor:
    """A CUDA multiprocessor's resources, which the templates' register and resource models read
    (`templates.resources`, `templates.cost`): the most threads and blocks resident at once
    (`resident_threads`, `resident_blocks`), its 32-bit `registers`, the most a thread holds
    (`thread_registers`), the allocator's `register_step` a thread (256 a warp), the shared memory the driver
    reserves for each resident block (`reserved_shared`), the bytes its shared memory serves a clock
    (`shared_bytes_per_clock`), the base-2 exponentials its special-function units return a clock
    (`exponentials_per_clock`), and the range its warpgroups' `setmaxnreg` moves a thread's registers in
    (`register_raises`, None before Hopper). The defaults are the facts every CUDA compute capability from 7.5
    to 12.x shares (CUDA C++ Programming Guide §5.1, Tables 33 and 34; the exponentials the CUDA C++ Best
    Practices Guide 13's §12.1.1 Table 5, `lg2.approx.f32`'s row, where CUDA 13 moved the throughput table,
    15.97 measured on an H200, the kernel challenge's MUFU.EX2; 128 bytes a clock measured 126 to 128 on an
    A100 and an H800, Luo et al. 2025, "Dissecting the NVIDIA Hopper Architecture through Microbenchmarking",
    Table 3) and setmaxnreg's range,
    which a target of no architecture (the CPU interpreting a launch) reads too, under which every family's
    launch interprets."""

    resident_threads: int = 1536
    resident_blocks: int = 16
    registers: int = 64 * 1024
    thread_registers: int = 255
    register_step: int = 8
    reserved_shared: int = 1024
    shared_bytes_per_clock: int = 128
    exponentials_per_clock: int = 16
    register_raises: RegisterRaises | None = SETMAXNREG

    def raises(self) -> RegisterRaises:
        """The range setmaxnreg moves a thread's registers in (`register_raises`); a multiprocessor before
        Hopper's takes none, which no family raising its consumers runs on."""
        if self.register_raises is None:
            raise ValueError("this multiprocessor takes no setmaxnreg: a family raising its consumers' "
                             "registers runs on Hopper's and Blackwell's")
        return self.register_raises


@dataclasses.dataclass(frozen=True)
class Architecture:
    """The facts every device of one instruction family shares, from its compute capability (a TPU's
    generation): the family a template's mainloop claims (`family`), its dense tensor-core FLOPs per clock per
    multiprocessor by arithmetic mode ('16bit': fp16 or bf16 operands summed in fp32; 'tf32'; '8bit': FP8
    operands, E4M3 or E5M2, where the family's tensor instructions multiply them natively; 'fp32': IEEE fp32
    on the CUDA cores, the rate a dot at Precision.HIGHEST takes), Hopper's warpgroup MMA, Blackwell's
    tensor memory columns (PTX ISA, tcgen05.alloc), and on a TPU the scoped VMEM limit at XLA's default and
    the VMEM a kernel may raise its limit to (`vmem_bytes`). `tma`: JAX's Mosaic GPU pipelines move a block
    into shared memory by TMA, which it picks from compute capability 9.0 on (its `emit_pipeline` and
    `copy_gmem_to_smem`), where below it they copy by cp.async. `dependent_launch`: PTX's grid dependency
    control (`griddepcontrol`, programmatic dependent launch), sm_90 on. A family of None is no template's:
    its devices' facts serve the rest of Dew (MFU's peaks). Its CUDA multiprocessor's resources
    (`multiprocessor`), None on a TPU."""

    family: str | None
    tensor_rates: Mapping[str, int] = dataclasses.field(default_factory=dict)
    multiprocessor: Multiprocessor | None = Multiprocessor()
    wgmma: bool = False
    tma: bool = False
    dependent_launch: bool = False
    tmem_columns: int | None = None
    scoped_vmem: int | None = None
    vmem_bytes: int | None = None


# Tensor FLOPs per clock per multiprocessor are a datasheet's dense peak over SMs x boost clock: A100 312 TF
# bf16 -> 2048, TF32 156 -> 1024; A10 125/62.5 -> 1024/512; L4 121/60 -> 1024/512; H100 SXM 989/495 ->
# 4096/2048; RTX PRO 6000 Blackwell Server 500/250 -> 1024/512; B200 2250/1125 over 148 SMs at its
# sustained 1.86 GHz -> 8192/4096, Hopper's doubled (the Blackwell whitepaper). IEEE fp32 runs an FMA a
# CUDA-core lane a clock: 64 lanes an SM on GA100 -> 128 (A100 19.5 TF over 108 SMs x 1410 MHz), 128 on
# TU10x, GA10x, AD10x, GH100 and GB10x/GB20x -> 256. GeForce parts of sm86 and sm89 sum 16-bit products in
# fp32 at half the rate (the GA102 and Ada whitepapers), which their catalogued peaks give
# (`GpuTarget.tensor_rate`). FP8 operands multiply natively at twice the 16-bit rate where the family's
# tensor instructions take them: Ada's and consumer Blackwell's mma.sync (L4 242.5 TF over 58 SMs x
# 2040 MHz -> 2048), Hopper's warpgroup MMA (H100 SXM 1979 -> 8192) and Blackwell's tcgen05 (B200 4500 ->
# 16384). Hopper's mma.sync takes none (an m16n8k32 e4m3 assembles to F2FP unpacks and HMMA.16816 on
# sm_90a; the kernel challenge's judge-facts.md:127), nor does Ampere's: there an FP8 product runs as a
# 16-bit one.
# Resident blocks and threads a multiprocessor holds are the CUDA C++ Programming Guide's (§5.1, Table 33):
# 7.5 16 and 1024, 8.0 32 and 2048, 8.6 16 and 1536, 8.9 24 and 1536, 9.0 and 10.0 32 and 2048, 12.x 24 and
# 1536; setmaxnreg is sm_90a's, sm_100f's and sm_120f's (PTX ISA, `SETMAXNREG`).
_AMPERE = Architecture('mma.sync', {'16bit': 2048, 'tf32': 1024, 'fp32': 128},
                       Multiprocessor(resident_threads=2048, resident_blocks=32, register_raises=None))
_AMPERE_CONSUMER = Architecture('mma.sync', {'16bit': 1024, 'tf32': 512, 'fp32': 256},
                                Multiprocessor(register_raises=None))
_ADA = Architecture('mma.sync', {'16bit': 1024, 'tf32': 512, '8bit': 2048, 'fp32': 256},
                    Multiprocessor(resident_blocks=24, register_raises=None))
# Turing's MMA takes no bf16 operands, which every template computes in: it is outside every family.
_TURING = Architecture(None, {'16bit': 1024, 'fp32': 128},
                       Multiprocessor(resident_threads=1024, register_raises=None))
_HOPPER = Architecture('wgmma', {'16bit': 4096, 'tf32': 2048, '8bit': 8192, 'fp32': 256},
                       Multiprocessor(resident_threads=2048, resident_blocks=32), wgmma=True, tma=True,
                       dependent_launch=True)
# Blackwell's tcgen05 family.
_BLACKWELL = Architecture('tcgen05', {'16bit': 8192, 'tf32': 4096, '8bit': 16384, 'fp32': 256},
                          Multiprocessor(resident_threads=2048, resident_blocks=32), tmem_columns=512,
                          tma=True, dependent_launch=True)
# Consumer Blackwell (RTX 50, RTX PRO 6000) runs sm80's mma.sync forms, its blocks moved by TMA: Ada's rates.
_BLACKWELL_CONSUMER = Architecture('mma.sync', {'16bit': 1024, 'tf32': 512, '8bit': 2048, 'fp32': 256},
                                   Multiprocessor(resident_blocks=24), tma=True, dependent_launch=True)
# A TPU's: its MXU's FLOPs a clock (v5e 197 TF over 1.5 GHz -> 131072) and the scoped VMEM limit at XLA's
# default (v5e 16 MiB, read from a Mosaic kernel's refusal: "limit 16.00M"; v6e's pending GEMMTPU-VMEM's).
# v6e: two 256x256 MXUs, 918 TF over its clock -> 524288. T4: 65 TF fp16 summed in fp32 and 8.1 TF fp32
# over 40 SMs at 1590 MHz -> 1024 and 128.
# VMEM capacity: v6e 128 MiB from an allocation refusal ("would exceed memory (size=134217728)",
# GEMMTPU-GRIDSTEP@v6e1); v5e 128 MiB by Google's specification, pending GEMMTPU-VMEM@v5e1's refusal.
_TPUS = {'v5e': Architecture('tpu', {'16bit': 131072}, None, scoped_vmem=16 * 2 ** 20,
                             vmem_bytes=128 * 2 ** 20),
         'v6e': Architecture('tpu', {'16bit': 524288}, None, vmem_bytes=128 * 2 ** 20)}


def architecture(generation: str) -> Architecture | None:
    """The instruction family `generation` belongs to, from its compute capability alone, one rule a
    capability (a TPU's generation), so a device no row names (B300, sm103) takes its family's facts;
    None outside every family Dew knows (a GPU before Turing, a TPU no rule states, a CPU)."""
    found = capability(generation)
    if found is None:
        return _TPUS.get(generation)
    major, minor = found
    if (major, minor) == (7, 5):
        return _TURING
    if major == 8:
        return {0: _AMPERE, 9: _ADA}.get(minor, _AMPERE_CONSUMER)
    if (major, minor) == (9, 0):
        return _HOPPER
    if major in (10, 11):
        return _BLACKWELL
    if major == 12:
        return _BLACKWELL_CONSUMER
    return None


def family(generation: str) -> str | None:
    """The instruction family a template's mainloop claims `generation` by ('mma.sync', 'wgmma',
    'tcgen05', 'tpu'), seen or unseen: a claim is a family, never a list of devices; None outside every
    family (a Turing GPU, a CPU)."""
    found = architecture(generation)
    return None if found is None else found.family


def issuers(generation: str) -> tuple[str, ...]:
    """The matrix instruction families a device of `generation` issues, its own first (`family`): a GPU of
    a family after Ampere issues mma.sync too, beside Hopper's wgmma or Blackwell's tcgen05; none outside
    every family."""
    own = family(generation)
    if own is None:
        return ()
    return (own,) if own in ('mma.sync', 'tpu') else (own, 'mma.sync')


@dataclasses.dataclass(frozen=True)
class Measured:
    """One part's facts as its kernels meet them, measured (tools/device_facts.py): its device-memory read
    against its bytes, cold (one kernel streaming N bytes after the L2 is evicted, the least-squares line over
    1 MiB to 1 GiB), `fixed` seconds before the stream (launch and first byte) and `bytes_per_second` after,
    the plateau a long transfer reaches, below what the memory clock and bus give; `launch`, the seconds a
    kernel launch adds to a program that runs kernels one after another; and `handoff`, a GPU-scope semaphore
    signal another program of the same launch waits on, where the part's ring families fix tiles up across
    programs (the balanced tail's partials). None where not measured."""

    fixed: float
    bytes_per_second: float
    launch: float | None = None
    handoff: float | None = None


@dataclasses.dataclass(frozen=True)
class GpuTarget:
    """One generation's device prefixes, dense peaks, block memory and measured kernel row.

    `dense_peaks` maps reported device-kind prefixes to precision rates, so
    PCIe/SXM and GeForce/data-center variants retain their own peaks. `device_kind`
    is the canonical device for a generation, or the reported SKU a reader asked
    for. Empty resources remain unknown. The standard-library-only table can
    also be read by a Torch reference's environment without importing JAX.

    `multiprocessors` is the canonical device's streaming multiprocessors, a real
    device's own where it reports them (`gpu_target`), which a launch sizes its
    programs by (`dew.nn.kernels.decode_attention`). `clock_hz` and
    `memory_bandwidth` are the canonical device's boost clock and device-memory
    bytes per second, a real device's own likewise; `measured` maps device-kind
    prefixes to the facts a part was measured to (`part`): its stream, which
    prices its device bytes (`memory_rate`), and its launches' and hand-offs'
    fixed seconds (`launch_seconds`, `handoff_seconds`). Every other fact a kernel's
    cost model reads (`dew.nn.kernels.templates.cost`) is its instruction
    family's (`architecture`): its tensor rates, warpgroup MMA and tensor memory. A row
    states a device's datasheet, its peaks, which MFU reads, and what a compile-only
    target of it needs, beside the few facts measured on the part itself. This
    table and `architecture` are the one
    place a choice is keyed on the hardware; any other code reads their facts, a
    library's support rule the compute capability (`capability`), and a launch
    config its launch's rule from those facts (`dew.nn.kernels.tuning`), as
    tools/lint_slop.py's SLOP016 holds.
    """

    generation: str
    device_kind: str
    shared_limit: int | None = None
    # Out of the hash (still compared), so a target keys a cache: a template's launches lowered once each
    # (`mainloop.lowered_once`).
    dense_peaks: Mapping[str, Mapping[str, float]] = dataclasses.field(default_factory=dict, hash=False)
    aliases: tuple[str, ...] = ()
    multiprocessors: int | None = None
    clock_hz: float | None = None
    memory_bandwidth: float | None = None
    measured: Mapping[str, Measured] = dataclasses.field(default_factory=dict, hash=False)
    int8_triton_gemm: bool = False
    """Whether XLA's Triton GEMM fusion computes int8 dots right on this GPU: on none measured, an
    A100 and an RTX PRO 6000 Blackwell, at jaxlib 0.11.2 (openxla/xla#51016;
    `dew.nn.kernels.options.compiler_options`)."""
    triton_gemm_fusions: bool = True
    """Whether a training step keeps XLA's Triton GEMM fusions on this GPU (`dew.training.memory`): off
    where a step measured cuBLAS's GEMMs faster with them off, an A100's (26a01bc18), an L4's
    (71966bcea) and an H100's (0fb7f065e), unless a module the step runs keeps them (`TritonGemm`)."""
    late_weight_gradients: bool = False
    """Whether XLA's schedule of a training step for this GPU runs its weight gradients after the whole
    backward where the step's head takes its gradient a tile at a time, unfused, holding the products'
    operands to then, as the derivation runs them (`dew.training.footprint.sunk`): an H100's every one, a
    B200's and a B300's some, which running every one there bounds from above (row 3341's calibration,
    compile-only, `TARGETS`' comments)."""
    step_seconds: float | None = None
    """One dependent step's latency on XLA's path: an iteration of a loop XLA lowers, its launches and its
    predicate, each reading the last (the work-span model's step, `templates.cost.Work.span`), measured per
    GPU (FS-STEP-SECONDS1's receipts); None where none was, so a form's span there is priced by no time."""
    kernel_step_seconds: float | None = None
    """One dependent step's latency within a kernel: a tile's 128-lane reduction broadcast into the next
    step's operand, in one program, each reading the last (a template walk's span, `templates.cost.walked`),
    measured per GPU as the slope over 1,024 to 16,384 steps (FS-STEP-SECONDS1's kernel path); None where
    none was."""

    @property
    def architecture(self) -> Architecture:
        """The target's instruction family's facts (`architecture`), none where it belongs to none."""
        return architecture(self.generation) or Architecture('')

    @property
    def tensor_rates(self) -> Mapping[str, int]:
        return self.architecture.tensor_rates

    @property
    def wgmma(self) -> bool:
        """Whether the GPU's matmuls run Hopper's warpgroup MMA, which Pallas Triton pipelines by holding
        every stage of a tile in shared memory (`dew.nn.kernels.grouped_matmul`)."""
        return self.architecture.wgmma

    @property
    def tmem_columns(self) -> int | None:
        """A Blackwell multiprocessor's tensor memory columns, the bound a kernel's tensor memory takes
        (tests/kernel_compile.py); None on a GPU without it."""
        return self.architecture.tmem_columns

    @property
    def scoped_vmem(self) -> int | None:
        """A TPU's scoped VMEM limit in bytes at XLA's default, past which a Mosaic kernel refuses to
        compile (`dew.nn.kernels.splash.block_sizes`); None where no rule states it."""
        return self.architecture.scoped_vmem

    @property
    def vmem_bytes(self) -> int | None:
        """A TPU's VMEM capacity in bytes, the most a Mosaic kernel may raise its scoped limit to
        (`vmem_limit_bytes`); None on GPUs and where no rule states it."""
        return self.architecture.vmem_bytes

    @property
    def multiprocessor(self) -> Multiprocessor:
        """The resources of this target's CUDA multiprocessor (`Architecture.multiprocessor`); a TPU has
        none, which no CUDA launch's derivation reads."""
        if self.architecture.multiprocessor is None:
            raise ValueError(f"a {self.generation} target has no CUDA multiprocessor")
        return self.architecture.multiprocessor

    @property
    def device_kind_prefixes(self) -> tuple[str, ...]:
        return tuple(self.dense_peaks)

    def _part[Fact](self, by_part: Mapping[str, Fact]) -> Fact | None:
        """The fact `by_part` states for this device, under the longest device-kind prefix it starts with."""
        keys = [prefix for prefix in by_part if self.device_kind.startswith(prefix)]
        return by_part[max(keys, key=len)] if keys else None

    @property
    def peaks(self) -> Mapping[str, float]:
        return self._part(self.dense_peaks) or {}

    @property
    def part(self) -> Measured | None:
        """The facts this device's part was measured to (`measured`); None for a part no one measured."""
        return self._part(self.measured)

    @property
    def memory_rate(self) -> float | None:
        """Device-memory bytes a second a long transfer streams at: the part's measured plateau (`part`),
        else its clocks' bandwidth (`memory_bandwidth`)."""
        part = self.part
        return self.memory_bandwidth if part is None else part.bytes_per_second

    @property
    def fixed_seconds(self) -> float | None:
        """The seconds a kernel's stream spends before its bytes arrive, its launch and first byte, measured
        on the part (`Measured.fixed`); None where no one measured it."""
        part = self.part
        return None if part is None else part.fixed

    @property
    def launch_seconds(self) -> float | None:
        """The seconds a kernel launch adds to a program of kernels one after another, measured on the part
        (`Measured.launch`); None where no one measured it."""
        part = self.part
        return None if part is None else part.launch

    @property
    def handoff_seconds(self) -> float | None:
        """A GPU-scope semaphore hand-off's seconds between two programs of one launch, measured on the
        part (`Measured.handoff`); None where no one measured it."""
        part = self.part
        return None if part is None else part.handoff

    @property
    def bf16_peak(self) -> float | None:
        return self.peaks.get('bf16')

    @property
    def fp8_peak(self) -> float | None:
        return self.peaks.get('fp8')

    def tensor_rate(self, mode: str) -> float | None:
        """Dense tensor-core FLOPs per clock per multiprocessor in arithmetic `mode`, or None where
        the generation states none: the generation's, lowered where the SKU's catalogued dense bf16 peak
        is below what its multiprocessors and clock give at that rate, as a GeForce part's half-rate
        fp32 accumulation reads; IEEE fp32 on the CUDA cores keeps its rate. A part reporting fewer
        multiprocessors than its catalogue counts (a MIG slice) keeps the generation's rate: the
        catalogue never raises it."""
        rate = self.tensor_rates.get(mode)
        peak = self.bf16_peak
        if rate is None or mode == 'fp32' or not (peak and self.multiprocessors and self.clock_hz):
            return rate
        return rate * min(1.0, peak / (self.tensor_rates['16bit'] * self.multiprocessors * self.clock_hz))


# CUDA datasheets give sparse tensor peaks; the dense figures below are half.
# A100 bf16 624; H100 SXM bf16/FP8 1979/3958, PCIe 1513/3026, NVL 1671/3341;
# H200 SXM 1979/3958, NVL 1671/3341; GH200 (its H100 die) 1979/3958, its SMs, clock and HBM3 (4 TB/s)
# or HBM3e (4.9) bandwidth a run reads of itself (`runtime_facts`); L4 242/485; RTX PRO 6000 Blackwell
# Server 1000/2000 TF/s.
# Existing rounded bf16 rates remain the baseline; FP8 uses the stated dense half.
# https://i.dell.com/sites/csdocuments/App-Merchandizing_Documents/en/us/h100-gpu.pdf
# https://www.nvidia.com/en-us/data-center/h100/
# https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/NVIDIA_H200_Datasheet.pdf (Jul24)
# https://www.nvidia.com/en-us/data-center/l4/
# https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
# GeForce bf16 is dense fp32-accumulate throughput: RTX4080 97.5 (Ada whitepaper),
# RTX3090 71 (GA102 whitepaper), half the fp16-accumulator rate.
# TPU per-chip bf16: v2/v3 45/123 (two JAX devices/chip, halved); v4 275;
# v5e 197, v5p459, v6e918 from Google's architecture and generation pages.
# Streaming multiprocessors of each canonical device: T4 40, A100 108, A10 72, L4 58, H100 SXM 132,
# RTX PRO 6000 Blackwell Server 188, from NVIDIA's architecture whitepapers and datasheets.
# Boost clock (MHz) and memory bandwidth (GB/s) of each canonical device, from the same datasheets:
# A100 SXM4 80GB 1410/2039, A10 1695/600, L4 2040/300, H100 SXM 1830/3350, RTX PRO 6000 Blackwell
# Server 2617/1597. The rates, latencies and L2 a family shares are `architecture`'s.
# Opted-in shared memory per block, in KiB: Turing64, A100163, Ampere/Ada/RTX
# Blackwell99, Hopper227. The SM reservation is already excluded:
# https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
# https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
# https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
# Attention and grouped products against tokamax 0.0.15's kernels, bf16, causal 2x4096, 16 query over
# 8 key heads, fwd/step ms (bench_tokamax.py; float64-gated, every arm within it). A100: width 128,
# the FA wheel 0.96/4.15, tokamax Triton 1.44/6.73, cuDNN 1.36/5.57; width 64, 0.80/2.44, 0.93/3.55,
# 0.84/3.44; 8 packed documents, the FA wheel's varlen 0.50/1.44 against tokamax Triton's key ranges
# 0.52/7.00. H100: width 128, the FA wheel (FA3) 0.30/1.03, tokamax Mosaic 0.37/1.28, Triton
# 0.55/2.00, cuDNN 0.37/1.28; width 64, 0.25/0.68, 0.30/0.93, 0.40/1.30, 0.30/0.91; packed,
# 0.17/0.60 against Triton's 0.22/1.88. A10 (sm86), causal 2x2048: width 64, the FA wheel
# 0.46/1.32, Triton 0.51/1.51, cuDNN 0.58/1.97; width 32, 0.23/0.82, 0.28/0.99, 0.47/1.67. Grouped
# products, 8192 tokens x 2048 -> 768 over 64 experts: Dew's Pallas grouped matmul 0.60/2.41 on an
# A100 and 0.27/0.86 on an H100, XLA's ragged dot 7.54/33.99 and 2.62/14.51. tokamax 0.0.15's
# ragged_dot ships a Triton backend (TritonRaggedDot: plain, transposed and ragged-contracting products,
# its VJP through them) that needs the triton and jax-triton packages, which Dew's locks do not carry
# (locks/cuda12.txt, cuda13.txt), so it was not measured here; its Mosaic one's backward takes no mixed
# dtypes. Requalifying that Triton backend as a comparator is ledger 153. No tokamax implementation is
# a row's kernel; Dew calls its own through tokamax's Op. Decode (bf16, one query,
# A100-SXM4-40GB, 2026-10-21, 79d8a3abf, synchronized ms by dew.telemetry.benchmark.samples): Qwen3-0.6B's
# 16/8 heads of 128 at 1 to 128 rows over 2048 to 131072 keys, the decode kernel 1.57x faster than
# cuDNN in geometric mean over the 28 shapes, faster in 21, at most 1.19x the fastest arm (cuDNN
# 2.78 ms and the decode kernel 0.42 at one row of 131072 keys; cuDNN's 7 wins at 4, 16 and 128 rows
# by 0.7 to 18%, 0.56 ms against 0.47 at 16 rows of 8192); Qwen3.5-0.8B's 8/2 heads of 256,
# which cuDNN does not take on sm80, the decode kernel at 1.04x the fastest in geometric mean.
TARGETS = {target.generation: target for target in (
    # T4: 40 SMs at a 1590 MHz boost, 320 GB/s (the T4 datasheet).
    GpuTarget('sm75', 'Tesla T4', 64 * 1024, dense_peaks={'Tesla T4': {}, 'NVIDIA Tesla T4': {}},
              aliases=('gpu:t4', 't4'), multiprocessors=40, clock_hz=1590e6, memory_bandwidth=320e9),
    # A dependent step on XLA's path: an A100-SXM4-40GB's loop iteration, 5.96 us (FS-STEP-SECONDS1-a100,
    # its one-element loop's slope over 64 to 1024 iterations, 0.4% from the line).
    GpuTarget('sm80', 'NVIDIA A100-SXM4-80GB', 163 * 1024,
              {'NVIDIA A100': {'bf16': 312e12}}, ('gpu:a100', 'a100'), multiprocessors=108, clock_hz=1410e6,
              memory_bandwidth=2039e9, triton_gemm_fusions=False, step_seconds=5.96e-6,
              # In-kernel: a dependent lane reduction's 225.3 ns (FS-STEP-SECONDS1-a100, its two rounds 225.32
              # and 225.25).
              kernel_step_seconds=225.3e-9,
              # tools/device_facts.py on an A100-SXM4-40GB: its stream FS-DEVICEFACTS2-a100's (1.555 TB/s by
              # its clocks), its launch FS-DEVICEFACTS4-a100's.
              measured={'NVIDIA A100-SXM4-40GB': Measured(6.04e-6, 1.486e12, launch=2.11e-6)}),
    GpuTarget('sm86', 'NVIDIA A10', 99 * 1024,
              # The RTX A6000's 154.8 TF dense (its datasheet's 309.7 with sparsity; the RTX PRO Blackwell
              # architecture whitepaper's Table 4), a real one's other facts its driver's (`runtime_facts`).
              {'NVIDIA A10': {}, 'NVIDIA GeForce RTX 3090': {'bf16': 71e12},
               'NVIDIA RTX A6000': {'bf16': 154.8e12}}, multiprocessors=72,
              clock_hz=1695e6, memory_bandwidth=600e9),
    GpuTarget('sm89', 'NVIDIA L4', 99 * 1024,
              {'NVIDIA L4': {'bf16': 121e12, 'fp8': 242.5e12},
               'NVIDIA GeForce RTX 4080': {'bf16': 97.5e12}}, ('gpu:l4', 'l4'), multiprocessors=58,
              clock_hz=2040e6, memory_bandwidth=300e9, triton_gemm_fusions=False),
    GpuTarget('sm90', 'NVIDIA H100 80GB HBM3', 227 * 1024, {
        'NVIDIA H100': {'bf16': 989e12, 'fp8': 1979e12},
        'NVIDIA H100 PCIe': {'bf16': 756e12, 'fp8': 1513e12},
        'NVIDIA H100 NVL': {'bf16': 835e12, 'fp8': 1670.5e12},
        'NVIDIA H200': {'bf16': 989e12, 'fp8': 1979e12},
        'NVIDIA H200 NVL': {'bf16': 835e12, 'fp8': 1670.5e12},
        'NVIDIA GH200': {'bf16': 989e12, 'fp8': 1979e12}}, multiprocessors=132,
              # A loop iteration on XLA's path, 4.93 us (FS-STEP-SECONDS1-h100's second round, 0.3% from the
              # line; its first's 4.80 us lay 17% from it).
              clock_hz=1830e6, memory_bandwidth=3350e9, triton_gemm_fusions=False, step_seconds=4.93e-6,
              # A step whose head takes its gradient in the backward a tile at a time, unfused, compiled for
              # an H100: Qwen3.5-4B's at 2 x 4096 tokens under 'full' holds 36.6 GiB of temporaries, 29
              # layers' four [8192, 9216] product operands at its peak and no layer's gradient, its weight
              # gradients run after the backward; Qwen3.5-0.8B's at 8 x 3072, 32.1. For an A100 they hold
              # 10.8 and 8.5, each layer's weight gradient run in its backward, and the 0.8B's 8.6 for an
              # RTX PRO 6000 (row 3341's calibration, compile-only; an L4's attention kernel compiles
              # for none).
              late_weight_gradients=True,
              # In-kernel: a dependent lane reduction's 142.7 ns on an H100 80GB HBM3 (FS-STEP-SECONDS1-h100,
              # its two rounds 142.72 and 142.61).
              kernel_step_seconds=142.7e-9,
              # tools/device_facts.py on an H100 SXM and a GH200: their streams FS-DEVICEFACTS2-h100's and
              # -gh200's, their launches and hand-offs FS-DEVICEFACTS4's; the H200's stream the same line
              # through the kernel challenge's best read at each size, 16.8 MB to 805 MB, cold L2 under graph
              # replay (judge-facts.md:3-18).
              # The H100 PCIe's two rounds of tools/device_facts.py agreed to 1% (ET-DEVICEFACTS-h100pcie),
              # its hand-off 0.695 us in both; its launch fit no line (negative slopes).
              measured={'NVIDIA H100 80GB HBM3': Measured(1.87e-6, 3.190e12, launch=1.02e-6, handoff=0.78e-6),
                        'NVIDIA H100 PCIe': Measured(2.11e-6, 1.980e12, handoff=0.695e-6),
                        'NVIDIA GH200 480GB': Measured(2.51e-6, 3.812e12, launch=0.86e-6, handoff=0.71e-6),
                        'NVIDIA H200': Measured(2.81e-6, 4.617e12)}),
    # Datasheet facts alone, no routes: what a compile-only sm100 device needs (`tests/kernel_compile.py`),
    # its peaks the scoreboard's MFU; a B200 run states its own clock and bandwidth (`runtime_facts`).
    GpuTarget('sm100', 'NVIDIA B200', 227 * 1024,
              dense_peaks={'NVIDIA B200': {'bf16': 2.25e15, 'fp8': 4.5e15}}, multiprocessors=148,
              clock_hz=1965e6, memory_bandwidth=8e12,
              # Qwen3.5-0.8B's step at 8 x 3072 tokens under 'full', its head tiled and unfused, compiled for
              # a B200 holds 22.9 GiB of temporaries, against 5.6 with the head fused: some of its weight
              # gradients run after the backward, which the derivation bounds from above by running all
              # there (43.4 derived; 8.5 with each in its layer's backward), row 3341's calibration,
              # compile-only.
              late_weight_gradients=True),
    # Blackwell Ultra, datasheet alone and unmeasured, a compile-only sm103 device's facts: 160 SMs and
    # 8 TB/s (NVIDIA, "Inside NVIDIA Blackwell Ultra"), HGX B300's 2.25/4.5 PF dense bf16/fp8 a GPU (the
    # HGX table's sparse figures halved over 8 GPUs). No source states its clock: B200's, which steers
    # only XLA's own fusions in a compile-only target; a B300 run reads its own (`runtime_facts`).
    GpuTarget('sm103', 'NVIDIA B300', 227 * 1024,
              dense_peaks={'NVIDIA B300': {'bf16': 2.25e15, 'fp8': 4.5e15}}, multiprocessors=160,
              clock_hz=1965e6, memory_bandwidth=8e12,
              # The B200's step above compiled for a B300 holds 13.9 GiB, 5.6 with the head fused: some
              # weight gradients late, bounded from above as on a B200 (row 3341's calibration).
              late_weight_gradients=True),
    GpuTarget('sm120', 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 99 * 1024,
              dense_peaks={'NVIDIA RTX PRO 6000 Blackwell': {'bf16': 500e12, 'fp8': 1000e12}},
              # The Server Edition's 1.597 TB/s (NVIDIA's RTX PRO 6000 Blackwell Server Edition page and
              # datasheet); 1.792 TB/s is the Workstation part's (the RTX PRO Blackwell architecture
              # whitepaper, p.12).
              aliases=('gpu:g4', 'g4'), multiprocessors=188, clock_hz=2617e6, memory_bandwidth=1597e9,
              # In-kernel: a dependent lane reduction's 137.7 ns (FS-STEP-SECONDS1-g4, its two rounds 137.68
              # and 137.66).
              step_seconds=3.68e-6, kernel_step_seconds=137.7e-9),
    GpuTarget('v2', 'TPU v2', dense_peaks={'TPU v2': {'bf16': 22.5e12}}),
    GpuTarget('v3', 'TPU v3', dense_peaks={'TPU v3': {'bf16': 61.5e12}}),
    GpuTarget('v4', 'TPU v4', dense_peaks={'TPU v4': {'bf16': 275e12}}),
    # One TensorCore a chip, of four 128 x 128 MXUs, each 16K multiply-adds a cycle, and 800 GiB/s of HBM
    # (cloud.google.com/tpu/docs/v5e, /tpu/docs/system-architecture-tpu-vm); the clock is 197e12 / 131072,
    # derived from the peak, no stated clock. Its scoped VMEM and MXU rate are its family's (`_TPUS`).
    GpuTarget('v5e', 'TPU v5 lite', dense_peaks={'TPU v5 lite': {'bf16': 197e12}}, multiprocessors=1,
              clock_hz=197e12 / 131072, memory_bandwidth=800 * 2 ** 30),
    GpuTarget('v5p', 'TPU v5', dense_peaks={'TPU v5': {'bf16': 459e12}, 'TPU v5p': {'bf16': 459e12}}),
    GpuTarget('v6e', 'TPU v6 lite', dense_peaks={'TPU v6 lite': {'bf16': 918e12}}),
)}


def capability(generation: str) -> tuple[int, int] | None:
    """The CUDA compute capability a generation names, `(8, 9)` for 'sm89', or None for a TPU's or a
    platform's: what a library states its support by."""
    digits = generation.removeprefix('sm')
    if digits == generation or not digits.isdigit():
        return None
    return int(digits[:-1]), int(digits[-1])


INTERPRETED_GENERATION = "sm90"
"""The generation whose facts rank a call's configs off a GPU of a family a bundle claims (a CPU): the
interpreted bundle's (`templates.bundles.INTERPRETED`, wgmma's); the algebra pass prices a CPU process's
forms by its facts too (`dew.algebra.cost.target`)."""


def gpu_target(identifier: str | None = None) -> GpuTarget:
    """The target for a generation, reported device kind or case-insensitive lab alias.

    None is this process's first device's (or the facts `assumed` was given): its generation is
    `device_generation`'s, the one seam that `assumed` and tests set, and its
    device kind only picks the SKU's peaks within that generation; a real GPU the table names no SKU of
    keeps its own kind, so its peaks stay unknown, while an assumed or patched
    generation takes the generation's canonical device. SKU queries retain
    that SKU's peaks. Unknown kinds have empty resources.
    """
    if identifier is None and (found := assumption()) is not None and found.target is not None:
        return found.target
    if identifier is None:
        import jax
        device = jax.local_devices()[0]
        real = device.platform == 'gpu' and assumption() is None
        if real and gpu_placed():
            _one_kind(jax.devices())
        return _resolved(device, device_generation(), real)
    if identifier in TARGETS:
        return TARGETS[identifier]
    for target in TARGETS.values():
        if identifier.lower() in target.aliases:
            return target
    matches = [(len(prefix), target) for target in TARGETS.values()
               for prefix in target.device_kind_prefixes if identifier.startswith(prefix)]
    if matches:
        return dataclasses.replace(max(matches, key=lambda entry: entry[0])[1], device_kind=identifier)
    return GpuTarget(identifier, identifier)


def built_target(generation: str, multiprocessors: int | None = None) -> GpuTarget:
    """The GPU a program compiled ahead is built for: the CUDA `generation`'s target (`gpu_target`) with the
    `multiprocessors` its build states, where it states them (an H100 PCIe's 114 against the canonical H100's
    132), else the generation's own. What a build's XLA target (`dew.training.ahead.target_config`), its
    launches (`assumed`) and the target its loader holds it to (`dew.training.ahead.BuildTarget`) all read."""
    target = gpu_target(generation)
    if multiprocessors is None:
        return target
    if multiprocessors <= 0:
        raise ValueError(f"a GPU states a positive count of multiprocessors, not {multiprocessors}")
    return dataclasses.replace(target, multiprocessors=multiprocessors)


def gpu_mesh(mesh: Mesh, generation: str) -> AbstractMesh:
    """`mesh`'s axes over abstract devices of the CUDA `generation` ('sm80'), none of which this process
    needs: a program traced on it, under `assumed(generation)`, lowers its Pallas kernels to Triton, and
    its Mosaic GPU kernels to that GPU's instructions, for that GPU on any host (`TrainerConfig.lower_for`;
    the kernel benchmark's oracle; tests/kernel_compile.py). Its devices state their compute capability, which
    Mosaic GPU lowers for (its `utils._infer_arch`; a device that states none it takes for sm90's, and an
    sm80 kernel's cp.async waits do not lower), but an sm90's, which Mosaic's default already is: stated,
    its `9.0` sends a warpgroup-MMA kernel's ptxas a lower default SM than the sm_90a its PTX targets
    ('SM version specified by .target is higher than default SM version assumed', FAM-contract3). JAX
    0.12's abstract device states it (`AbstractDevice.compute_capability`)."""
    from jax.sharding import AbstractDevice, AbstractMesh
    target = gpu_target(generation)
    version = capability(generation)
    if not target.generation.startswith('sm') or target.shared_limit is None or version is None:
        raise ValueError(f"no CUDA target for {generation!r}")
    stated = None if version == (9, 0) else '.'.join(map(str, version))
    device = AbstractDevice(device_kind=target.device_kind, num_cores=None, platform='cuda',
                            compute_capability=stated)
    return AbstractMesh(mesh.axis_sizes, mesh.axis_names, axis_types=mesh.axis_types, abstract_device=device)


# The first GPU generation with bf16 tensor cores (Ampere). Below it XLA
# rejects the BF16_BF16_F32 dot algorithm at run time ("UNIMPLEMENTED:
# Unsupported algorithm on the current device(s): ALG_DOT_BF16_BF16_F32"),
# cuDNN's fused attention refuses bf16 ("SDPA FP16/BF16 requires SM80"), and
# Triton does not compile; measured on a T4 (sm75), 2026-09-22.
BF16_GPU = 80


def _compute_capability(device) -> str | None:
    """`device`'s compute capability, '8.9' for an sm89 card. jax's Device
    declares no such field, and a CUDA device carries it, so it is read at
    this boundary; a device without one gives None."""
    return getattr(device, 'compute_capability', None)


def _multiprocessors(device) -> int | None:
    """`device`'s streaming multiprocessors. jax's Device declares no such
    field, and a CUDA device reports them as its cores, so it is read at this
    boundary; a device without one gives None."""
    return getattr(device, 'core_count', None)


def _shared_memory(device) -> int | None:
    """The shared memory, in bytes, a block of `device` may hold once a kernel opts in. jax's Device
    declares no such field, and a CUDA device reports it among its attributes as XLA describes it
    (`shared_memory_per_block_optin`), so it is read at this boundary; a device without one gives
    None."""
    return getattr(device, 'shared_memory_per_block_optin', None)


@functools.cache
def _clocks(ordinal: int) -> tuple[tuple[str, float], ...]:
    """The visible GPU `ordinal`'s multiprocessor clock and device-memory bandwidth (twice the memory
    clock times the bus's bytes, as CUDA's deviceQuery derives it), which jax's Device does not carry,
    from the CUDA driver (`dew.devices.gpu_attribute`), read once a process; none where no
    driver loads."""
    from dew.devices import CLOCK_RATE, GLOBAL_MEMORY_BUS_WIDTH, MEMORY_CLOCK_RATE, gpu_attribute

    clock, memory_clock, bus = (gpu_attribute(ordinal, attribute)
                                for attribute in (CLOCK_RATE, MEMORY_CLOCK_RATE, GLOBAL_MEMORY_BUS_WIDTH))
    if clock is None or memory_clock is None or bus is None:
        return ()
    return ('clock_hz', clock * 1e3), ('memory_bandwidth', 2 * memory_clock * 1e3 * bus / 8)


def _one_kind(devices) -> None:
    """Refuse a run whose processes read their facts from GPUs of different kinds: every process traces
    the same program, so the kernel choices each makes from its first GPU's facts (`gpu_target`) must be
    the same, which first GPUs of one kind give. A process's other GPUs take no part."""
    first = {}
    for device in sorted(devices, key=lambda device: device.id):
        first.setdefault(device.process_index, device.device_kind)
    kinds = sorted(set(first.values()))
    if len(kinds) > 1:
        raise ValueError(f"the run's processes read their kernel facts from GPUs of {len(kinds)} kinds "
                         f"({', '.join(kinds)}); each traces the same program, so their first GPUs must "
                         "be of one kind")


def runtime_facts(target: GpuTarget, device) -> GpuTarget:
    """`target` with the facts a real CUDA `device` reports of itself in place of the table's: its
    multiprocessors, its block's shared memory, its clock and memory bandwidth (`_clocks`), read free at
    runtime, its family's facts its capability's (`architecture`). A GPU the table does not name so
    keeps every resource a kernel's requirements and cost read."""
    facts = {'multiprocessors': _multiprocessors(device), 'shared_limit': _shared_memory(device),
             **dict(_clocks(device.local_hardware_id))}
    return dataclasses.replace(target, **{name: value for name, value in facts.items() if value is not None})


SHARED_BUDGET = 99 * 1024
"""The shared memory a block holds where a target reports none: the least of any CUDA GPU Dew's Pallas
launches run on (sm86, sm89 and sm120 hold 99 KiB), so a launch sized by it fits wherever it runs."""


def tpu(target: GpuTarget) -> bool:
    """Whether `target` is a TPU's: its device kind names one ('TPU v5 lite')."""
    return target.device_kind.startswith('TPU ')


def shared_budget(target: GpuTarget) -> int:
    """The shared memory, in bytes, a program on `target` may stage: the block's own, which a real
    device reports (`runtime_facts`), else `SHARED_BUDGET`."""
    return target.shared_limit or SHARED_BUDGET


def fp8_tensor_cores(generation: str) -> bool:
    """Whether a CUDA generation's tensor cores multiply FP8: compute capability 8.9 (Ada) and later."""
    found = capability(generation)
    return found is not None and found >= (8, 9)


def device_generation() -> str:
    """Return the hardware generation of the devices a call is placed on.

    The generation is 'sm89' for a GPU of compute capability 8.9, 'v6e' for a
    TPU v6e, and the platform's name otherwise; within `assumed`, the one it
    names. A call placed off the process's default backend (`placed_platform`),
    as a host-layout run's on a GPU host's CPU, takes its platform's name, so
    every choice keyed on the target follows it there.
    """
    import jax
    if (found := assumption()) is not None:
        return found.generation
    if (platform := placed_platform()) != jax.default_backend():
        return platform
    return _generation(jax.local_devices()[0])


def _generation(device) -> str:
    """`device`'s own generation: a CUDA device's by its compute capability, a TPU's by its kind, another
    platform's its name."""
    capability = _compute_capability(device) if device.platform == 'gpu' else None
    if capability:
        return 'sm' + capability.replace('.', '')
    if device.platform == 'tpu':
        kind = device.device_kind or 'tpu'
        return gpu_target(kind).generation
    return device.platform


def device_target(device) -> GpuTarget:
    """`device`'s own target, whatever `assumed` says, as `gpu_target` resolves this process's first device
    unassumed (`_resolved`): its generation (`_generation`), its kind's SKU and, a CUDA device's, the facts it
    reports of itself (`runtime_facts`), its multiprocessors among them. What a program compiled for a target
    is checked against before it loads (`dew.training.ahead.BuildTarget`)."""
    return _resolved(device, _generation(device), device.platform == 'gpu')


def _resolved(device, generation: str, real: bool) -> GpuTarget:
    """`device`'s target within `generation`: its kind picks the SKU's peaks within it, and where the table
    names no SKU of that kind of the generation, a `real` device keeps its own kind, so its peaks stay
    unknown, and another takes the generation's canonical device; a `real` one carries the facts it reports
    (`runtime_facts`)."""
    kind = device.device_kind
    target = gpu_target(kind)
    if target.generation != generation:
        canonical = TARGETS.get(generation, GpuTarget(generation, kind))
        target = dataclasses.replace(canonical, device_kind=kind) if real else canonical
    return runtime_facts(target, device) if real else target


def chosen_kernel(choice: str, default: str) -> str:
    """The kernel `chosen` names for `choice` within its block, else `default`: a choice a site decides
    without a selector (the attention template's form, a step's narrow copies)."""
    from dew import cache

    known_choices((choice,))
    return dict(cache.KERNEL_CHOICES.value).get(choice, default)



INHERENT = 'inherent'
"""The row of a limit no implementation of the kernel lifts (`Refusal.row`)."""


@dataclasses.dataclass(frozen=True)
class Route:
    """One dispatch's decision, made from the target's facts (`device_generation`, the call's shapes
    and dtypes, the binary's architectures), never the host's placement: `kernel` is what the target
    runs for `choice` at `site` (module:function), `preferred` what it runs where nothing refuses (`chosen`,
    or its selector's preference), `refusal` why `preferred` did not take the
    call, and `ran` what this host executed, `kernel` but where the host cannot run it (cuDNN within
    `assumed` on a CPU), and `forced` the kernel the call named or `chosen` named, which forced the choice
    (None where the selector chose). A route is generic exactly where `kernel != preferred`."""

    choice: str
    kernel: str
    preferred: str
    refusal: Refusal | None
    ran: str
    target: str
    site: str
    forced: str | None = None


class _Routes:
    """The routes a `routes` block collected; hashed by identity, as `dew.cache.KERNEL_ROUTES` keys
    on it."""

    def __init__(self) -> None:
        self.found: list[Route] = []


@contextlib.contextmanager
def routes() -> Iterator[list[Route]]:
    """Collect every `Route` a dispatch reports while code runs or traces in the block, in order: what
    a step's trace within `assumed(target)` on a CPU shows of that target's kernels. Each block is a
    fresh collection JAX keys its traces on (`dew.cache.KERNEL_ROUTES`), so a function traced before
    traces, and reports, again; a block nested in another reports to both."""
    from dew import cache

    collected = _Routes()
    with cache.KERNEL_ROUTES((*cache.KERNEL_ROUTES.value, collected)):
        yield collected.found


def _site() -> str:
    """The module and function of the first caller outside this module and the selector."""
    frame = inspect.currentframe()
    while frame is not None and frame.f_globals.get('__name__') in (__name__, 'dew.nn.kernels.selection'):
        frame = frame.f_back
    return '?' if frame is None else f"{frame.f_globals.get('__name__')}:{frame.f_code.co_qualname}"


# Each (choice, kernel, refusal) `route` has logged in this process.
_logged: set[tuple[str, str, str]] = set()


def route(choice: str, *, kernel: str, preferred: str | None = None, refusal: Refusal | None = None,
          ran: str | None = None, forced: str | None = None) -> Route:
    """Report the dispatch of `choice` to `kernel`, the one call every dispatch site makes once it has
    decided (`dew.nn.kernels.selection.select`): `preferred` defaults to the kernel `chosen` names, else
    `kernel`; `ran` is what this host executes, `kernel` by default. Every `routes` block open records it.
    A call off its preferred kernel is logged once a process for each kernel and refusal at INFO, naming
    both and the refusal's reason; one that runs the preferred kernel logs nothing."""
    from dew import cache

    known_choices((choice,))
    generation = device_generation()
    preferred = preferred or dict(cache.KERNEL_CHOICES.value).get(choice) or kernel
    found = Route(choice, kernel, preferred, refusal, ran or kernel, generation, _site(), forced)
    for collected in cache.KERNEL_ROUTES.value:
        collected.found.append(found)
    detail = '' if refusal is None else refusal.detail
    if preferred != kernel and (choice, kernel, detail) not in _logged:
        _logged.add((choice, kernel, detail))
        _log.info("%s runs %r, not %r, on %s: %s", choice, kernel, preferred, generation, detail)
    return found


def linear_tangent(forward: Callable, backward: Callable, portable: Callable, *,
                   live_cotangents: bool = False, primal: Callable | None = None,
                   nested: bool = True) -> Callable:
    """A fused kernel as one function of its operands, differentiated at every order through
    `portable`, the same arithmetic in XLA, while every level of it runs the kernels.
    `forward(*operands)` gives the kernel's output and what its backward reads (`saved`, a tree of
    arrays); `backward(saved, operands, cotangent)` gives the operands' cotangents, a tuple; and
    `portable(*operands)` gives the output.

    The output's derivative is one custom JVP whose tangent is `portable`'s, evaluated as a
    linear map that transposes to `backward` (`linear_map`): forward mode (`jax.jvp`,
    `jax.linearize`) takes the portable tangent however a jit or a scan staged the call, and reverse
    mode runs the backward kernel, through a vjp or through a jvp that a shard_map or a scan then
    transposes. The forward and the backward are custom JVPs of their own whose tangents are
    `portable`'s and its pullback's, so a higher order (`jax.jvp` of `jax.grad`, `jax.grad` of
    `jax.jvp`) differentiates through XLA while each level's value comes from the kernels; what
    `saved` holds is a function of the operands, whose tangents carry it. This composes with
    `jax.vmap` of a gradient, the gradient of a `jax.vmap` and forward over reverse.

    With `live_cotangents`, `backward(saved, operands, cotangent, live)` gives the cotangents of the
    operands at the indices `live` alone, those a reverse pass differentiates, so a frozen operand's
    cotangent is never formed. `primal(*operands)` gives the output where no derivative is taken (a kernel
    that saves nothing), `forward`'s where None. Without `nested`, a derivative's level runs `forward` as
    it is, not as a custom JVP of its own: a value it names (`checkpoint_name`) is a residual a remat
    policy can keep, and no higher order differentiates through it (a kernel's custom VJP takes none)."""
    import jax
    from jax.custom_derivatives import SymbolicZero, zero_from_primal

    @jax.custom_jvp
    def produced(*operands):
        return forward(*operands)

    @produced.defjvp
    def _(primals, tangents):
        output, saved = produced(*primals)
        # `saved` reaches only `pulled`, whose tangent reads the operands' instead.
        return (output, saved), (jax.jvp(portable, primals, tangents)[1], zero_from_primal(saved))

    @functools.partial(jax.custom_jvp, nondiff_argnums=(3,))
    def pulled(saved, operands, cotangent, live):
        arguments = (saved, operands, cotangent) if live is None else (saved, operands, cotangent, live)
        return backward(*arguments)

    @pulled.defjvp
    def _(live, primals, tangents):
        _, operands, cotangent = primals
        _, doperands, dcotangent = tangents

        def pullback(operands, cotangent):
            whole = jax.vjp(portable, *operands)[1](cotangent)
            # `live` indexes the operands' leaves, as the reverse pass's `backward` takes them.
            return whole if live is None else tuple(jax.tree.leaves(whole)[index] for index in live)

        return pulled(*primals, live), jax.jvp(pullback, (operands, cotangent), (doperands, dcotangent))[1]

    @jax.custom_jvp
    def bounded(*operands):
        return forward(*operands)[0] if primal is None else primal(*operands)

    def bounded_jvp(primals, tangents):
        output, saved = (produced if nested else forward)(*primals)
        # Only the perturbed operands are the linear map's: its transpose takes no known zero.
        directions, tree = jax.tree.flatten(tangents, is_leaf=lambda leaf: isinstance(leaf, SymbolicZero))
        live = [index for index, direction in enumerate(directions)
                if not isinstance(direction, SymbolicZero)]
        if not live:
            return output, zero_from_primal(output)

        def tangent(residuals, moving):
            operands, _ = residuals
            whole = [zero_from_primal(operand) for operand in jax.tree.leaves(operands)]
            for index, value in zip(live, moving, strict=True):
                whole[index] = value
            return jax.jvp(portable, operands, jax.tree.unflatten(tree, whole))[1]

        def transposed(residuals, cotangent):
            operands, saved = residuals
            if live_cotangents:
                return list(jax.tree.leaves(pulled(saved, operands, cotangent, tuple(live))))
            pulled_back = jax.tree.leaves(pulled(saved, operands, cotangent, None))
            return [pulled_back[index] for index in live]

        moving = [directions[index] for index in live]
        # The tangent typed as the output it perturbs, its sharding included: traced from `portable`'s
        # tangent, every reverse pass traced the portable math's forward mode for its type alone (a gated
        # delta net layer's training step's trace on an H100's lowering, row 3142).
        typed = jax.tree.map(lambda leaf: jax.typeof(leaf).to_tangent_aval(), output)
        return output, linear_map(tangent, transposed, (primals, saved), moving, out_aval=typed)

    bounded.defjvp(bounded_jvp, symbolic_zeros=True)
    return bounded


def linear_map(apply: Callable, pullback: Callable, residuals, moving, *, out_aval=None):
    """`apply(residuals, moving)`, linear in `moving`, whose transpose is `pullback(residuals, cotangent)`
    (the cotangents of `moving`, a tree like it): one `jax.experimental.hijax.HiPrim`, so reverse mode
    runs `pullback` whether it reaches the map through a vjp or through a jvp that a shard_map, a scan
    or `jax.linear_transpose` transposes, and under `jax.vmap` too, where the map is one of `apply` and
    `pullback` mapped over the batch. Its tangent is the map of the moving tangents plus `apply`'s own
    derivative in `residuals` (a higher order's). `out_aval`, where given, is the output's type (a tree of
    avals like it), which `apply` is then traced for only where forward mode runs it."""
    import jax

    # The output's type as `apply` traces it, its sharding included: a shape and dtype alone
    # (`jax.eval_shape`'s) drop an explicit axis's, and reverse mode would hand the pullback its cotangent
    # unsharded (row 1759).
    if out_aval is None:
        traced, shapes = jax.make_jaxpr(apply, return_shape=True)(residuals, moving)
        out_aval = jax.tree.unflatten(jax.tree.structure(shapes), traced.out_avals)
    in_avals = jax.tree.map(jax.typeof, (residuals, moving))
    return _linear_primitive()(apply, pullback, in_avals, out_aval)(residuals, moving)


@functools.cache
def _linear_primitive() -> type:
    """`linear_map`'s primitive, defined at its first call: this module imports no JAX of its own."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import hijax

    def zero(leaf) -> bool:
        return isinstance(leaf, hijax.Zero)

    def all_zero(tangents) -> bool:
        return all(zero(leaf) for leaf in jax.tree.leaves(tangents, is_leaf=zero))

    def instantiated(tangents):
        """`tangents` with each symbolic zero an array of its tangent type: float0's for an integer leaf."""
        return jax.tree.map(hijax.instantiate_zeros, tangents, is_leaf=zero)

    def unbatched(cotangent):
        """An unbatched operand's cotangent: its batch's sum."""
        if cotangent.dtype == jax.dtypes.float0:
            return np.zeros(cotangent.shape[1:], jax.dtypes.float0)
        return cotangent.sum(0)

    class Linear(hijax.HiPrim):
        apply: Callable
        pullback: Callable
        """`linear_map`'s two functions, `params` that `HiPrim` makes attributes."""

        def __init__(self, apply, pullback, in_avals, out_aval):
            self.in_avals, self.out_aval = in_avals, out_aval
            self.params = {"apply": apply, "pullback": pullback}
            super().__init__()

        def expand(self, *args):
            residuals, moving = args
            return self.apply(residuals, moving)

        def transpose(self, out_ct, *maybe_accums):
            residuals, moving = maybe_accums
            if all_zero(out_ct):
                return
            pulled = self.pullback(residuals, instantiated(out_ct))
            accums = jax.tree.leaves(moving, is_leaf=lambda leaf: isinstance(leaf, hijax.GradAccum))
            for accum, value in zip(accums, jax.tree.leaves(pulled), strict=True):
                if isinstance(accum, hijax.GradAccum):
                    accum.accum(value)

        def jvp(self, primals, tangents):
            residuals, moving = primals
            d_residuals, d_moving = tangents
            terms = []
            if not all_zero(d_moving):
                terms.append(self(residuals, instantiated(d_moving)))
            if not all_zero(d_residuals):
                terms.append(jax.jvp(lambda held: self.apply(held, moving), (residuals,),
                                     (instantiated(d_residuals),))[1])
            if not terms:
                return self(residuals, moving), jax.tree.map(
                    lambda aval: hijax.Zero(aval.to_tangent_aval()), self.out_aval)
            return self(residuals, moving), jax.tree.map(lambda *parts: sum(parts), *terms)

        lin, linearized = hijax.linearize_from_jvp
        vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

        def batch(self, axis_data, args, dims):
            # A linear map again, of `apply` and `pullback` mapped, so a batch transposes to the pullback.
            residual_axes, moving_axes = dims
            mapped = {"axis_size": axis_data.size, "axis_name": axis_data.name,
                      "spmd_axis_name": axis_data.spmd_name or axis_data.explicit_mesh_axis}

            def applied(residuals, moving):
                return jax.vmap(self.apply, in_axes=(residual_axes, moving_axes), **mapped)(residuals, moving)

            def pulled(residuals, cotangent):
                back = jax.vmap(self.pullback, in_axes=(residual_axes, 0), **mapped)(residuals, cotangent)
                leaves, tree = jax.tree.flatten(back)
                axes = jax.tree.leaves(moving_axes, is_leaf=lambda axis: axis is None)
                return jax.tree.unflatten(tree, [
                    unbatched(leaf) if axis is None else jnp.moveaxis(leaf, 0, axis)
                    for leaf, axis in zip(leaves, axes, strict=True)])

            return linear_map(applied, pulled, *args), jax.tree.map(lambda _: 0, self.out_aval)

    return Linear


def bf16_dot_runs() -> bool:
    """Return whether the default device multiplies bf16 operands into an fp32 sum as one dot algorithm.

    That algorithm is BF16_BF16_F32. Every TPU and CPU has it, and so does a
    GPU from generation `BF16_GPU` on.
    """
    generation = device_generation()
    return not generation.startswith('sm') or int(generation[2:]) >= BF16_GPU


def placed_platform() -> str:
    """The platform of the devices the context places a call on: the set mesh's, else the default
    device's (`jax.default_device`), else `jax.default_backend()`, 'gpu' for CUDA's and ROCm's. A
    host-layout run builds and steps its CPU-owned state on a GPU host's CPU, and every choice keyed
    on the platform follows it there."""
    import jax
    device = jax.sharding.get_abstract_mesh().abstract_device or jax.config.jax_default_device
    if device is None:
        return jax.default_backend()
    platform = device if isinstance(device, str) else device.platform
    return 'gpu' if platform in ('cuda', 'rocm') else platform


def gpu_placed() -> bool:
    """Whether a call runs on this process's own GPU: what a CUDA library call needs, which its trace
    asks the installed plugin about and its lowering the devices, where `backend()` also takes an
    assumed GPU. A step traced for an assumed GPU (`lower_for`) lowers without devices, even on a
    GPU host, so it makes none."""
    import jax
    return assumption() is None and jax.default_backend() == 'gpu' and placed_platform() == 'gpu'


def cuda_lowered() -> bool:
    """Whether a library's CUDA call lowers here for the call's target: on this process's own GPU
    (`gpu_placed`), or within `assumed` on a GPU host whose own GPU is the assumed generation's, where a
    compile ahead's program for that GPU's compile-only devices lowers it as for its own (an FFI target
    the library registered, and the facts it reads of the GPU: FlashAttention's, dew_flash_attn)."""
    import jax
    if gpu_placed():
        return True
    found = assumption()
    return (found is not None and jax.default_backend() == 'gpu'
            and _generation(jax.local_devices()[0]) == found.generation)


def backend() -> str:
    """The platform a call's code is chosen for: `placed_platform()`, or within `assumed` the
    assumed generation's, 'gpu' for an `smXY` and 'tpu' for a TPU's, so a choice keyed on the
    platform acts as on that device while a step traces for it on another."""
    if (found := assumption()) is None:
        return placed_platform()
    generation = found.generation
    if generation.startswith('sm'):
        return 'gpu'
    if generation in TARGETS and tpu(TARGETS[generation]):
        return 'tpu'
    return generation


def triton_runs() -> bool:
    """Return whether Dew's Pallas GPU (Triton) kernels can run here.

    This is the one eligibility rule for those kernels: a GPU of compute
    capability 8.0 or later, the same bound JAX's own Pallas lowerings apply.
    A T4 fails to compile them ("Triton support is only enabled for
    cc>=8.0"). Within `assumed`, the GPU is the assumed generation's.
    """
    # JAX's bound is `_backend_supports_triton`.
    return backend() == 'gpu' and bf16_dot_runs()
