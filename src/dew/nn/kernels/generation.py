"""The hardware generation every kernel selector in Dew keys on.

One key, read from the default device: `smXY` from a GPU's compute
capability, `v5e`/`v5p`/`v6e`/`v4` from a TPU's device kind, and the
platform name otherwise. A selector names the generations it was measured on
and sends every other one to its portable path. The measurements are in
docs/performance.md, "Kernel choices per generation".
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import inspect
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax.sharding import AbstractMesh, Mesh

_log = logging.getLogger(__name__)

ASSUMED: contextvars.ContextVar[str | None] = contextvars.ContextVar("assumed_generation", default=None)
"""The generation `assumed` stands in for the default device's, None outside it."""

ASSUMED_TARGET: contextvars.ContextVar[GpuTarget | None] = contextvars.ContextVar("assumed_target",
                                                                                    default=None)
"""The device facts `assumed` stands in for the default device's where it was given them, None otherwise."""


@contextlib.contextmanager
def assumed(generation: str | GpuTarget | None) -> Iterator[None]:
    """Within the block, choose kernels as on a device of `generation` ('sm80'), whatever device this
    process has: `device_generation`, `bf16_dot_runs` and `triton_runs` answer for it, so every
    `KERNELS` choice and kernel refusal acts as on that GPU while a step traces and lowers for one on
    a CPU (`dew.training._lowering`). Given a target's facts (a GPU no row names, its multiprocessors,
    shared memory, clock and bandwidth its own), `gpu_target` answers with them, so every launch is sized
    and costed by them alone. None assumes nothing. JAX keys its trace and compile caches on the device
    assumed (`dew.cache.KERNEL_ASSUMED`), so a function jitted for one traces again for another."""
    from dew import cache

    if isinstance(generation, GpuTarget):
        target, name = generation, generation.generation
    else:
        target, name = None, generation
    token = ASSUMED.set(name)
    facts = ASSUMED_TARGET.set(target)
    try:
        with cache.KERNEL_ASSUMED((name, None if target is None else _facts(target))):
            yield
    finally:
        ASSUMED_TARGET.reset(facts)
        ASSUMED.reset(token)


def _facts(value):
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


CHOICES = frozenset({'attention', 'attention_mainloop', 'attention_narrow', 'attention_packed',
                     'attention_sinks', 'attention_window', 'decode', 'delta_decode', 'delta_gates',
                     'gate_up', 'gemm_epilogue', 'gemm_prologue', 'gated_delta_rule', 'grouped_matmul',
                     'local_attention', 'mla', 'narrow_copies', 'paged_decode', 'rms_norm', 'short_conv',
                     'ssd', 'ssd_decode', 'swiglu'})
"""The choices `measured_kernel` answers, and so the ones `chosen` may name and a dispatch reports
(`route`). A choice no generation's row names runs its site's own best kernel."""


def _named(choices: Iterable[str]) -> None:
    """Refuse any of `choices` that `CHOICES` does not hold, naming every one it holds."""
    if unknown := set(choices) - CHOICES:
        raise ValueError(f"no kernel choice is named {', '.join(sorted(unknown))}; "
                         f"the choices are {', '.join(sorted(CHOICES))}")


@contextlib.contextmanager
def chosen(**kernels: str) -> Iterator[None]:
    """Within the block, a choice named in `kernels` runs that kernel whatever the generation's row
    measured, for an A/B of two kernels or a test of a kernel this device's row does not name.
    Nested blocks keep the outer block's other choices and restore its choices on exit. JAX keys its
    trace and compile caches on the choices (`dew.cache.KERNEL_CHOICES`), so a function jitted
    outside the block traces again within it."""
    from dew import cache

    _named(kernels)
    with cache.KERNEL_CHOICES(tuple(sorted({**dict(cache.KERNEL_CHOICES.value), **kernels}.items()))):
        yield


PALLAS_CHOICES = frozenset({'pallas', 'fla'})
"""The `KERNELS` values that run Dew's Pallas or Triton kernels, which lower through JAX's Pallas
Triton backend for a GPU only: what the CPU dry run lowers (`dew.training._lowering`) and the kernel
lab verifies on the GPU. 'flash' and 'cudnn' name CUDA libraries."""


@dataclasses.dataclass(frozen=True)
class Architecture:
    """The facts every device of one instruction family shares, from its compute capability (a TPU's
    generation): the family a template's mainloop claims (`family`), its dense tensor-core FLOPs per
    clock per multiprocessor by arithmetic mode ('16bit': fp16 or bf16 operands summed in fp32; 'tf32';
    'fp32': IEEE fp32 on the CUDA cores, the rate a dot at Precision.HIGHEST takes), Hopper's warpgroup
    MMA, Blackwell's tensor memory columns (PTX ISA, tcgen05.alloc), and what only a measurement
    states: the global-memory load latency in multiprocessor clocks and the bytes the L2 serves a
    multiprocessor a clock; on a TPU the scoped VMEM limit at XLA's default, the VMEM a kernel may raise
    its limit to (`vmem_bytes`), and a grid step's fixed seconds."""

    family: str
    tensor_rates: Mapping[str, int] = dataclasses.field(default_factory=dict)
    wgmma: bool = False
    tmem_columns: int | None = None
    load_latency: float | None = None
    l2_bytes_per_clock: float | None = None
    scoped_vmem: int | None = None
    vmem_bytes: int | None = None
    grid_step_seconds: float | None = None


# Tensor FLOPs per clock per multiprocessor are a datasheet's dense peak over SMs x boost clock: A100 312 TF
# bf16 -> 2048, TF32 156 -> 1024; A10 125/62.5 -> 1024/512; L4 121/60 -> 1024/512; H100 SXM 989/495 ->
# 4096/2048; RTX PRO 6000 Blackwell Server 500/250 -> 1024/512; B200 2250/1125 over 148 SMs at its
# sustained 1.86 GHz -> 8192/4096, Hopper's doubled (the Blackwell whitepaper). IEEE fp32 runs an FMA a
# CUDA-core lane a clock: 64 lanes an SM on GA100 -> 128 (A100 19.5 TF over 108 SMs x 1410 MHz), 128 on
# TU10x, GA10x, AD10x, GH100 and GB10x/GB20x -> 256. GeForce parts of sm86 and sm89 sum 16-bit products in
# fp32 at half the rate (the GA102 and Ada whitepapers), which their catalogued peaks give
# (`GpuTarget.tensor_rate`). Global-memory load latency in clocks, a pointer chase's (Luo et al.,
# "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", IPDPS 2024, arXiv 2402.13499, Table
# IV): A100 466.3, H800 478.8, RTX 4090 541.5. L2 bytes a multiprocessor a clock, Dew's streaming probe
# (L2PROBE, probe_l2.py) over SMs x clock: A100 5.43 TB/s -> 35.7, A10 4.74 -> 38.8, H100 SXM 13.2 -> 54.6.
# A family no device of which was measured takes its nearest measured family's values, said so below.
_AMPERE = Architecture('mma.sync', {'16bit': 2048, 'tf32': 1024, 'fp32': 128}, load_latency=466.3,
                       l2_bytes_per_clock=35.7)
_AMPERE_CONSUMER = Architecture('mma.sync', {'16bit': 1024, 'tf32': 512, 'fp32': 256}, load_latency=541.5,
                                l2_bytes_per_clock=38.8)
_TURING = Architecture('mma.sync', {'16bit': 1024, 'fp32': 128}, load_latency=541.5, l2_bytes_per_clock=38.8)
_HOPPER = Architecture('wgmma', {'16bit': 4096, 'tf32': 2048, 'fp32': 256}, wgmma=True, load_latency=478.8,
                       l2_bytes_per_clock=54.6)
# Blackwell's tcgen05 family: no member measured, its latency and L2 Hopper's.
_BLACKWELL = Architecture('tcgen05', {'16bit': 8192, 'tf32': 4096, 'fp32': 256}, tmem_columns=512,
                          load_latency=478.8, l2_bytes_per_clock=54.6)
# Consumer Blackwell (RTX 50, RTX PRO 6000) runs sm80's mma.sync forms: Ada's rates and measured values.
_BLACKWELL_CONSUMER = Architecture('mma.sync', {'16bit': 1024, 'tf32': 512, 'fp32': 256}, load_latency=541.5,
                                   l2_bytes_per_clock=38.8)
# A TPU's: its MXU's FLOPs a clock (v5e 197 TF over 1.5 GHz -> 131072), the scoped VMEM limit at XLA's
# default (v5e 16 MiB, read from a Mosaic kernel's refusal: "limit 16.00M"; v6e's pending GEMMTPU-VMEM's),
# and a grid step's fixed seconds, the slope of time over 1 to 65536 Pallas grid steps each issuing one
# small DMA (GEMMTPU-GRIDSTEP: v5e 0.1785 us, v6e 0.2835 us). v6e: two 256x256 MXUs, 918 TF over its
# clock -> 524288. T4: 65 TF fp16 summed in fp32 and 8.1 TF fp32
# over 40 SMs at 1590 MHz -> 1024 and 128.
# VMEM capacity: v6e 128 MiB from an allocation refusal ("would exceed memory (size=134217728)",
# GEMMTPU-GRIDSTEP@v6e1); v5e 128 MiB by Google's specification, pending GEMMTPU-VMEM@v5e1's refusal.
_TPUS = {'v5e': Architecture('tpu', {'16bit': 131072}, scoped_vmem=16 * 2 ** 20, vmem_bytes=128 * 2 ** 20,
                             grid_step_seconds=1.785e-7),
         'v6e': Architecture('tpu', {'16bit': 524288}, vmem_bytes=128 * 2 ** 20, grid_step_seconds=2.835e-7)}


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
        return _AMPERE if minor == 0 else _AMPERE_CONSUMER
    if (major, minor) == (9, 0):
        return _HOPPER
    if major in (10, 11):
        return _BLACKWELL
    if major == 12:
        return _BLACKWELL_CONSUMER
    return None


def family(generation: str) -> str | None:
    """The instruction family a template's mainloop claims `generation` by ('mma.sync', 'wgmma',
    'tcgen05', 'tpu'), seen or unseen: a claim is a family, never a list of devices."""
    found = architecture(generation)
    return None if found is None else found.family


@dataclasses.dataclass(frozen=True)
class GpuTarget:
    """One generation's device prefixes, dense peaks, block memory and measured kernel row.

    `dense_peaks` maps reported device-kind prefixes to precision rates, so
    PCIe/SXM and GeForce/data-center variants retain their own peaks. `device_kind`
    is the canonical device for a generation, or the reported SKU a reader asked
    for. Empty resources remain unknown. The standard-library-only table can
    also be read by a Torch reference's environment without importing JAX.

    `head_tile` is the chunked LM head's `(tokens, columns)` tile where one was
    measured to beat the default (`dew.objectives.lm.chunked.chunked_tile`).
    `multiprocessors` is the canonical device's streaming multiprocessors, a real
    device's own where it reports them (`gpu_target`), which a launch sizes its
    programs by (`dew.nn.kernels.decode_attention`). `clock_hz` and
    `memory_bandwidth` are the canonical device's boost clock and device-memory
    bytes per second, a real device's own likewise. Every other fact a kernel's
    cost model reads (`dew.nn.kernels.templates.cost`) is its instruction
    family's (`architecture`): its tensor rates, warpgroup MMA, tensor memory, and
    the measured values of its load latency and L2 bandwidth. A row
    states a device's datasheet alone: its peaks, which MFU reads, and what a
    compile-only target of it needs. This table and `architecture` are the one
    place a choice is keyed on the hardware; any other code reads their facts, a
    library's support rule the compute capability (`capability`), and a launch
    config its launch's rule from those facts (`dew.nn.kernels.tuning`), as
    tools/lint_slop.py's SLOP016 holds.
    """

    generation: str
    device_kind: str
    shared_limit: int | None = None
    kernels: Mapping[str, str] = dataclasses.field(default_factory=dict)
    dense_peaks: Mapping[str, Mapping[str, float]] = dataclasses.field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    head_tile: tuple[int, int] | None = None
    multiprocessors: int | None = None
    clock_hz: float | None = None
    memory_bandwidth: float | None = None
    int8_triton_gemm: bool = False
    """Whether XLA's Triton GEMM fusion computes int8 dots right on this GPU: on none measured, an
    A100 and an RTX PRO 6000 Blackwell, at jaxlib 0.11.2 (openxla/xla#51016;
    `dew.nn.kernels.options.compiler_options`)."""

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
    def grid_step_seconds(self) -> float | None:
        """A TPU grid step's fixed seconds beside its work (the cost model's step term); None on GPUs,
        whose pipelined steps the model prices by their latency."""
        return self.architecture.grid_step_seconds

    @property
    def load_latency(self) -> float | None:
        """The global-memory load latency in multiprocessor clocks (the cost model's waiting term): its
        family's."""
        return self.architecture.load_latency

    @property
    def l2_bandwidth(self) -> float | None:
        """The bytes a second the L2 serves all multiprocessors' loads together (the cost model's L2
        term): its family's bytes a multiprocessor a clock over its multiprocessors and clock; None where
        any is unstated, the term then dropped."""
        rate = self.architecture.l2_bytes_per_clock
        if rate is None or self.multiprocessors is None or self.clock_hz is None:
            return None
        return rate * self.multiprocessors * self.clock_hz

    @property
    def device_kind_prefixes(self) -> tuple[str, ...]:
        return tuple(self.dense_peaks)

    @property
    def peaks(self) -> Mapping[str, float]:
        keys = [prefix for prefix in self.dense_peaks if self.device_kind.startswith(prefix)]
        return self.dense_peaks[max(keys, key=len)] if keys else {}

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
# Server 2617/1792. The rates, latencies and L2 a family shares are `architecture`'s.
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
# A100-SXM4-40GB, 2026-10-21, ced5d3ad9, synchronized ms by dew.telemetry.benchmark.samples): Qwen3-0.6B's
# 16/8 heads of 128 at 1 to 128 rows over 2048 to 131072 keys, the decode kernel 1.57x faster than
# cuDNN in geometric mean over the 28 shapes, faster in 21, at most 1.19x the fastest arm (cuDNN
# 2.78 ms and the decode kernel 0.42 at one row of 131072 keys; cuDNN's 7 wins at 4, 16 and 128 rows
# by 0.7 to 18%, 0.56 ms against 0.47 at 16 rows of 8192); Qwen3.5-0.8B's 8/2 heads of 256,
# which cuDNN does not take on sm80, the decode kernel at 1.04x the fastest in geometric mean.
TARGETS = {target.generation: target for target in (
    # T4: 40 SMs at a 1590 MHz boost, 320 GB/s (the T4 datasheet).
    GpuTarget('sm75', 'Tesla T4', 64 * 1024, dense_peaks={'Tesla T4': {}, 'NVIDIA Tesla T4': {}},
              aliases=('gpu:t4', 't4'), multiprocessors=40, clock_hz=1590e6, memory_bandwidth=320e9),
    GpuTarget('sm80', 'NVIDIA A100-SXM4-80GB', 163 * 1024,
              {'attention': 'flash', 'attention_narrow': 'flash',
               'attention_window': 'flash', 'attention_sinks': 'flash',
               'attention_packed': 'flash', 'xla_triton_gemm': 'off', 'decode': 'decode',
               'rms_norm': 'pallas', 'swiglu': 'pallas', 'gate_up': 'joined', 'delta_gates': 'joined',
               'gated_delta_rule': 'fla', 'ssd': 'pallas'},
              {'NVIDIA A100': {'bf16': 312e12}}, ('gpu:a100', 'a100'), head_tile=(4096, 8192),
              multiprocessors=108, clock_hz=1410e6, memory_bandwidth=2039e9),
    GpuTarget('sm86', 'NVIDIA A10', 99 * 1024, {'attention_narrow': 'flash'},
              {'NVIDIA A10': {}, 'NVIDIA GeForce RTX 3090': {'bf16': 71e12}}, multiprocessors=72,
              clock_hz=1695e6, memory_bandwidth=600e9),
    GpuTarget('sm89', 'NVIDIA L4', 99 * 1024,
              {'attention': 'flash', 'attention_window': 'flash', 'attention_sinks': 'flash',
               'xla_triton_gemm': 'off', 'gated_delta_rule': 'fla'},
              {'NVIDIA L4': {'bf16': 121e12, 'fp8': 242.5e12},
               'NVIDIA GeForce RTX 4080': {'bf16': 97.5e12}}, ('gpu:l4', 'l4'), head_tile=(4096, 8192),
              multiprocessors=58, clock_hz=2040e6, memory_bandwidth=300e9),
    GpuTarget('sm90', 'NVIDIA H100 80GB HBM3', 227 * 1024,
              {'attention': 'flash', 'attention_narrow': 'flash', 'attention_window': 'flash',
               'attention_sinks': 'flash', 'attention_packed': 'flash', 'xla_triton_gemm': 'off',
               'rms_norm': 'pallas', 'swiglu': 'pallas', 'delta_gates': 'joined', 'gated_delta_rule': 'fla',
               'ssd': 'pallas'}, {
        'NVIDIA H100': {'bf16': 989e12, 'fp8': 1979e12},
        'NVIDIA H100 PCIe': {'bf16': 756e12, 'fp8': 1513e12},
        'NVIDIA H100 NVL': {'bf16': 835e12, 'fp8': 1670.5e12},
        'NVIDIA H200': {'bf16': 989e12, 'fp8': 1979e12},
        'NVIDIA H200 NVL': {'bf16': 835e12, 'fp8': 1670.5e12},
        'NVIDIA GH200': {'bf16': 989e12, 'fp8': 1979e12}}, multiprocessors=132,
              clock_hz=1830e6, memory_bandwidth=3350e9),
    # Datasheet facts alone, no routes: what a compile-only sm100 device needs (`tests/kernel_compile.py`),
    # its peaks the scoreboard's MFU; a B200 run states its own clock and bandwidth (`runtime_facts`).
    GpuTarget('sm100', 'NVIDIA B200', 227 * 1024,
              dense_peaks={'NVIDIA B200': {'bf16': 2.25e15, 'fp8': 4.5e15}}, multiprocessors=148,
              clock_hz=1965e6, memory_bandwidth=8e12),
    # Blackwell Ultra, datasheet alone and unmeasured, a compile-only sm103 device's facts: 160 SMs and
    # 8 TB/s (NVIDIA, "Inside NVIDIA Blackwell Ultra"), HGX B300's 2.25/4.5 PF dense bf16/fp8 a GPU (the
    # HGX table's sparse figures halved over 8 GPUs). No source states its clock: B200's, which steers
    # only XLA's own fusions in a compile-only target; a B300 run reads its own (`runtime_facts`).
    GpuTarget('sm103', 'NVIDIA B300', 227 * 1024,
              dense_peaks={'NVIDIA B300': {'bf16': 2.25e15, 'fp8': 4.5e15}}, multiprocessors=160,
              clock_hz=1965e6, memory_bandwidth=8e12),
    GpuTarget('sm120', 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 99 * 1024,
              {'attention': 'flash', 'gated_delta_rule': 'fla'},
              dense_peaks={'NVIDIA RTX PRO 6000 Blackwell': {'bf16': 500e12, 'fp8': 1000e12}},
              aliases=('gpu:g4', 'g4'), multiprocessors=188, clock_hz=2617e6, memory_bandwidth=1792e9),
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


def gpu_target(identifier: str | None = None) -> GpuTarget:
    """The target for a generation, reported device kind or case-insensitive lab alias.

    None is this process's first device's (or the facts `assumed` was given): its generation is
    `device_generation`'s, the one seam that `assumed` and tests set, and its
    device kind only picks the SKU's peaks within that generation; a real GPU the table names no SKU of
    keeps its own kind, so its peaks stay unknown, while an assumed or patched
    generation takes the generation's canonical device. SKU queries retain
    that SKU's peaks. Unknown kinds have empty resources.
    """
    if identifier is None and (facts := ASSUMED_TARGET.get()) is not None:
        return facts
    if identifier is None:
        import jax
        device = jax.local_devices()[0]
        generation, kind = device_generation(), device.device_kind
        real = device.platform == 'gpu' and ASSUMED.get() is None
        if real and gpu_placed():
            _one_kind(jax.devices())
        target = gpu_target(kind)
        if target.generation != generation:
            canonical = TARGETS.get(generation, GpuTarget(generation, kind))
            target = dataclasses.replace(canonical, device_kind=kind) if real else canonical
        return runtime_facts(target, device) if real else target
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


def gpu_mesh(mesh: Mesh, generation: str) -> AbstractMesh:
    """`mesh`'s axes over abstract devices of the CUDA `generation` ('sm80'), none of which this process
    needs: a program traced on it, under `assumed(generation)`, lowers its Pallas kernels to Triton, and
    its Mosaic GPU kernels to that GPU's instructions, for that GPU on any host (`TrainerConfig.lower_for`;
    the kernel lab's oracle; tests/kernel_compile.py). Its devices state their compute capability, which
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
# Triton does not compile; measured on a T4 (sm75), KernelMatrix 2026-09-22.
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
    every choice a target row measures (`measured_kernel`) follows it there.
    """
    import jax
    generation = ASSUMED.get()
    if generation is not None:
        return generation
    if (platform := placed_platform()) != jax.default_backend():
        return platform
    device = jax.local_devices()[0]
    capability = _compute_capability(device) if device.platform == 'gpu' else None
    if capability:
        return 'sm' + capability.replace('.', '')
    if device.platform == 'tpu':
        kind = device.device_kind or 'tpu'
        return gpu_target(kind).generation
    return device.platform


def measured_kernel(choice: str, default: str, *, fastest: str | None = None) -> str:
    """The kernel `chosen` names for `choice`, else `fastest`, the site's own measurement of a call
    no row measured (an fp32 SiLU product), else `KERNELS` on this generation, else `default`."""
    from dew import cache

    _named((choice,))
    if (kernel := dict(cache.KERNEL_CHOICES.value).get(choice)) is not None:
        return kernel
    return fastest or gpu_target(device_generation()).kernels.get(choice, default)


@dataclasses.dataclass(frozen=True)
class Refusal:
    """Why the kernel a target prefers does not take a call, from the call's own facts on that
    target (its shapes, dtypes, mask): a stable `code`, which a kernel's `SUPPORTS` keys its declared
    limits on, and the `detail` a log names. What the host cannot run is no refusal (`Route.ran`)."""

    code: str
    detail: str
    row: str | None = None
    """The roadmap or ledger row that tracks lifting the limit (`dew.nn.kernels.selection.Requirement`),
    `INHERENT` where no implementation of the kernel lifts it, None for a refusal no requirement gives
    (an analysis's own, which its op's requirements read)."""

    def __str__(self) -> str:
        return self.detail


INHERENT = 'inherent'
"""The row of a limit no implementation of the kernel lifts (`Refusal.row`)."""


@dataclasses.dataclass(frozen=True)
class Route:
    """One dispatch's decision, made from the target's facts (`device_generation`, the call's shapes
    and dtypes, the binary's architectures), never the host's placement: `kernel` is what the target
    runs for `choice` at `site` (module:function), `preferred` what it runs where nothing refuses (the
    generation's row, `chosen`, or the site's own best), `refusal` why `preferred` did not take the
    call, and `ran` what this host executed, `kernel` but where the host cannot run it (cuDNN within
    `assumed` on a CPU). A route is generic exactly where `kernel != preferred`."""

    choice: str
    kernel: str
    preferred: str
    refusal: Refusal | None
    ran: str
    target: str
    site: str


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
          ran: str | None = None) -> Route:
    """Report the dispatch of `choice` to `kernel`, the one call every dispatch site makes once it has
    decided (`dew.nn.kernels.selection.select`): `preferred` defaults to the kernel `chosen` or the
    generation's row names, else `kernel`; `ran` is what this host executes, `kernel` by default.
    Every `routes` block open records it. A call off the measured kernel is the dispatch's own choice,
    logged once a process for each kernel and refusal at INFO, naming both; one that runs the
    preferred kernel, or a choice no row names, logs nothing."""
    from dew import cache

    _named((choice,))
    generation = device_generation()
    picked = None if preferred else dict(cache.KERNEL_CHOICES.value).get(choice)
    measured = preferred or picked or gpu_target(generation).kernels.get(choice)
    found = Route(choice, kernel, measured or kernel, refusal, ran or kernel, generation, _site())
    for collected in cache.KERNEL_ROUTES.value:
        collected.found.append(found)
    detail = '' if refusal is None else refusal.detail
    if measured not in (None, kernel) and (choice, kernel, detail) not in _logged:
        _logged.add((choice, kernel, detail))
        named = "the kernel `chosen` names" if picked else f"the kernel measured fastest on {generation}"
        _log.info("%s runs %r, not %r, %s: %s", choice, kernel, measured, named, detail)
    return found


def linear_tangent(forward: Callable, backward: Callable, portable: Callable, *,
                   live_cotangents: bool = False) -> Callable:
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
    cotangent is never formed."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.custom_derivatives import SymbolicZero

    @jax.custom_jvp
    def produced(*operands):
        return forward(*operands)

    @produced.defjvp
    def _(primals, tangents):
        output, saved = produced(*primals)
        # `saved` reaches only `pulled`, whose tangent reads the operands' instead.
        return (output, saved), (jax.jvp(portable, primals, tangents)[1], jax.tree.map(jnp.zeros_like, saved))

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
        return forward(*operands)[0]

    def bounded_jvp(primals, tangents):
        output, saved = produced(*primals)
        # Only the perturbed operands are the linear map's: its transpose takes no known zero.
        directions, tree = jax.tree.flatten(tangents, is_leaf=lambda leaf: isinstance(leaf, SymbolicZero))
        live = [index for index, direction in enumerate(directions)
                if not isinstance(direction, SymbolicZero)]
        if not live:
            return output, jax.tree.map(jnp.zeros_like, output)

        def tangent(residuals, moving):
            operands, _ = residuals
            # An operand with no tangent space (an index, a length) takes float0's zero.
            whole = [jnp.zeros_like(operand) if jnp.issubdtype(operand.dtype, jnp.inexact)
                     else np.zeros(operand.shape, jax.dtypes.float0)
                     for operand in jax.tree.leaves(operands)]
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
        return output, linear_map(tangent, transposed, (primals, saved), moving)

    bounded.defjvp(bounded_jvp, symbolic_zeros=True)
    return bounded


def linear_map(apply: Callable, pullback: Callable, residuals, moving):
    """`apply(residuals, moving)`, linear in `moving`, whose transpose is `pullback(residuals, cotangent)`
    (the cotangents of `moving`, a tree like it): one `jax.experimental.hijax.HiPrim`, so reverse mode
    runs `pullback` whether it reaches the map through a vjp or through a jvp that a shard_map, a scan
    or `jax.linear_transpose` transposes, and under `jax.vmap` too, where the map is one of `apply` and
    `pullback` mapped over the batch. Its tangent is the map of the moving tangents plus `apply`'s own
    derivative in `residuals` (a higher order's)."""
    import jax

    # The output's type as `apply` traces it, its sharding included: a shape and dtype alone
    # (`jax.eval_shape`'s) drop an explicit axis's, and reverse mode would hand the pullback its cotangent
    # unsharded (row 1759).
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
    return ASSUMED.get() is None and jax.default_backend() == 'gpu' and placed_platform() == 'gpu'


def backend() -> str:
    """The platform a call's code is chosen for: `placed_platform()`, or within `assumed` the
    assumed generation's, 'gpu' for an `smXY` and 'tpu' for a TPU's, so a choice keyed on the
    platform acts as on that device while a step traces for it on another."""
    generation = ASSUMED.get()
    if generation is None:
        return placed_platform()
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
