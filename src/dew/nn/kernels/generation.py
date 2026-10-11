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
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax
    from jax.sharding import AbstractMesh, Mesh

_log = logging.getLogger(__name__)

ASSUMED: contextvars.ContextVar[str | None] = contextvars.ContextVar("assumed_generation", default=None)
"""The generation `assumed` stands in for the default device's, None outside it."""


@contextlib.contextmanager
def assumed(generation: str | None) -> Iterator[None]:
    """Within the block, choose kernels as on a device of `generation` ('sm80'), whatever device this
    process has: `device_generation`, `bf16_dot_runs` and `triton_runs` answer for it, so every
    `KERNELS` choice and kernel refusal acts as on that GPU while a step traces and lowers for one on
    a CPU (`dew.training._lowering`). None assumes nothing."""
    token = ASSUMED.set(generation)
    try:
        yield
    finally:
        ASSUMED.reset(token)


PALLAS_CHOICES = frozenset({'pallas', 'fla', 'triton'})
"""The `KERNELS` values that run Dew's Pallas or Triton kernels, which lower through JAX's Pallas
Triton backend for a GPU only: what the CPU dry run lowers (`dew.training._lowering`) and the kernel
lab verifies on the GPU. 'flash' and 'cudnn' name CUDA libraries."""


UNLOWERABLE = {
    ('tokamax_grouped_matmul', 'triton'): (
        "tokamax registers its Triton ragged dot on GPU hosts only, so no CPU lowers it, and the "
        "kernel lab's verify on the GPU is its only check"),
}
"""The Pallas choices a `KERNELS` row can name whose Triton lowering no CPU reaches, and why."""


@dataclasses.dataclass(frozen=True)
class GpuTarget:
    """One generation's device prefixes, dense peaks, block memory and measured kernel row.

    `dense_peaks` maps reported device-kind prefixes to precision rates, so
    PCIe/SXM and GeForce/data-center variants retain their own peaks. `device_kind`
    is the canonical device for a generation, or the reported SKU a reader asked
    for. Empty resources remain unknown. The standard-library-only table can
    also be read by a Torch reference's environment without importing JAX.
    """

    generation: str
    device_kind: str
    shared_limit: int | None = None
    kernels: Mapping[str, str] = dataclasses.field(default_factory=dict)
    dense_peaks: Mapping[str, Mapping[str, float]] = dataclasses.field(default_factory=dict)
    aliases: tuple[str, ...] = ()

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


# CUDA datasheets give sparse tensor peaks; the dense figures below are half.
# A100 bf16 624; H100 SXM bf16/FP8 1979/3958, PCIe 1513/3026, NVL 1671/3341;
# H200 SXM 1979/3958, NVL 1671/3341; L4 242/485; RTX PRO 6000 Blackwell Server 1000/2000 TF/s.
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
# Opted-in shared memory per block, in KiB: Turing64, A100163, Ampere/Ada/RTX
# Blackwell99, Hopper227. The SM reservation is already excluded:
# https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
# https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
# https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
TARGETS = {target.generation: target for target in (
    GpuTarget('sm75', 'Tesla T4', 64 * 1024, dense_peaks={'Tesla T4': {}, 'NVIDIA Tesla T4': {}},
              aliases=('gpu:t4', 't4')),
    GpuTarget('sm80', 'NVIDIA A100-SXM4-80GB', 163 * 1024,
              {'attention': 'flash', 'grouped_matmul': 'pallas', 'tokamax_grouped_matmul': 'triton',
               'xla_triton_gemm': 'off', 'gated_delta_rule': 'pallas', 'narrow_copies': 'on'},
              {'NVIDIA A100': {'bf16': 312e12}}, ('gpu:a100', 'a100')),
    GpuTarget('sm86', 'NVIDIA A10', 99 * 1024, {'grouped_matmul': 'pallas'},
              {'NVIDIA A10': {}, 'NVIDIA GeForce RTX 3090': {'bf16': 71e12}}),
    GpuTarget('sm89', 'NVIDIA L4', 99 * 1024,
              {'attention': 'flash', 'grouped_matmul': 'pallas', 'tokamax_grouped_matmul': 'triton',
               'xla_triton_gemm': 'off', 'gated_delta_rule': 'pallas', 'narrow_copies': 'on'},
              {'NVIDIA L4': {'bf16': 121e12, 'fp8': 242.5e12},
               'NVIDIA GeForce RTX 4080': {'bf16': 97.5e12}}, ('gpu:l4', 'l4')),
    GpuTarget('sm90', 'NVIDIA H100 80GB HBM3', 227 * 1024, dense_peaks={
        'NVIDIA H100': {'bf16': 989e12, 'fp8': 1979e12},
        'NVIDIA H100 PCIe': {'bf16': 756e12, 'fp8': 1513e12},
        'NVIDIA H100 NVL': {'bf16': 835e12, 'fp8': 1670.5e12},
        'NVIDIA H200': {'bf16': 989e12, 'fp8': 1979e12},
        'NVIDIA H200 NVL': {'bf16': 835e12, 'fp8': 1670.5e12}}),
    GpuTarget('sm120', 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 99 * 1024,
              dense_peaks={'NVIDIA RTX PRO 6000 Blackwell': {'bf16': 500e12, 'fp8': 1000e12}},
              aliases=('gpu:g4', 'g4')),
    GpuTarget('v2', 'TPU v2', dense_peaks={'TPU v2': {'bf16': 22.5e12}}),
    GpuTarget('v3', 'TPU v3', dense_peaks={'TPU v3': {'bf16': 61.5e12}}),
    GpuTarget('v4', 'TPU v4', dense_peaks={'TPU v4': {'bf16': 275e12}}),
    GpuTarget('v5e', 'TPU v5 lite',
              kernels={'grouped_matmul': 'xla', 'tokamax_grouped_matmul': 'mosaic_tpu_v2'},
              dense_peaks={'TPU v5 lite': {'bf16': 197e12}}),
    GpuTarget('v5p', 'TPU v5', dense_peaks={'TPU v5': {'bf16': 459e12}, 'TPU v5p': {'bf16': 459e12}}),
    GpuTarget('v6e', 'TPU v6 lite',
              kernels={'grouped_matmul': 'xla', 'tokamax_grouped_matmul': 'mosaic_tpu_v2'},
              dense_peaks={'TPU v6 lite': {'bf16': 918e12}}),
)}


def gpu_target(identifier: str | None = None) -> GpuTarget:
    """The target for a generation, reported device kind or case-insensitive lab alias.

    None is the default device's: its generation is `device_generation`'s, the
    one seam that `assumed` and tests set, and its device kind only picks the
    SKU's peaks within that generation; a real GPU the table names no SKU of
    keeps its own kind, so its peaks stay unknown, while an assumed or patched
    generation takes the generation's canonical device. SKU queries retain
    that SKU's peaks. Unknown kinds have empty resources.
    """
    if identifier is None:
        import jax
        device = jax.devices()[0]
        generation, kind = device_generation(), device.device_kind
        target = gpu_target(kind)
        if target.generation == generation:
            return target
        canonical = TARGETS.get(generation, GpuTarget(generation, kind))
        real = device.platform == 'gpu' and ASSUMED.get() is None
        return dataclasses.replace(canonical, device_kind=kind) if real else canonical
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
    needs: a program traced on it, under `assumed(generation)`, lowers its Pallas kernels to Triton
    for that GPU on any host (`TrainerConfig.lower_for`; the kernel lab's oracle)."""
    from jax.sharding import AbstractDevice, AbstractMesh
    target = gpu_target(generation)
    if not target.generation.startswith('sm') or target.shared_limit is None:
        raise ValueError(f"no CUDA target for {generation!r}")
    device = AbstractDevice(device_kind=target.device_kind, num_cores=None, platform='cuda')
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


def device_generation() -> str:
    """Return the default device's hardware generation.

    The generation is 'sm89' for a GPU of compute capability 8.9, 'v6e' for a
    TPU v6e, and the platform's name otherwise; within `assumed`, the one it
    names.
    """
    import jax
    generation = ASSUMED.get()
    if generation is not None:
        return generation
    device = jax.devices()[0]
    capability = _compute_capability(device) if device.platform == 'gpu' else None
    if capability:
        return 'sm' + capability.replace('.', '')
    if device.platform == 'tpu':
        kind = device.device_kind or 'tpu'
        return gpu_target(kind).generation
    return device.platform


def measured_kernel(choice: str, default: str) -> str:
    """The kernel `KERNELS` names for `choice` on the default device's generation, else `default`."""
    return gpu_target(device_generation()).kernels.get(choice, default)


# Each (choice, kernel, refusal) `ran_kernel` has logged in this process.
_logged: set[tuple[str, str, str]] = set()


def ran_kernel(choice: str, kernel: str, refusal: str) -> str:
    """Return `kernel`, which a call to `choice` runs because the kernel `KERNELS` measured fastest on
    this generation turned it down for `refusal`. The first such call in a process for each kernel and
    refusal is logged as a warning naming both, so a change that sends calls off the measured kernel
    shows in a run's log, not only in its step time. A generation without a measurement, or a call
    that runs the measured kernel, logs nothing."""
    generation = device_generation()
    measured = gpu_target(generation).kernels.get(choice)
    if measured not in (None, kernel) and (choice, kernel, refusal) not in _logged:
        _logged.add((choice, kernel, refusal))
        _log.warning("%s runs %r, not %r, the kernel measured fastest on %s: %s",
                     choice, kernel, measured, generation, refusal)
    return kernel


def first_refusal(*checks: tuple[bool, str]) -> str | None:
    """The reason beside the first of `checks` that does not hold, or None where all hold: how a
    kernel's eligibility says why it turns a call down (`ran_kernel`)."""
    return next((reason for holds, reason in checks if not holds), None)


def bf16_dot_runs() -> bool:
    """Return whether the default device multiplies bf16 operands into an fp32 sum as one dot algorithm.

    That algorithm is BF16_BF16_F32. Every TPU and CPU has it, and so does a
    GPU from generation `BF16_GPU` on.
    """
    generation = device_generation()
    return not generation.startswith('sm') or int(generation[2:]) >= BF16_GPU


def backend() -> str:
    """The platform a call's code is chosen for: `jax.default_backend()`, or within `assumed` the
    assumed generation's, 'gpu' for an `smXY` and 'tpu' for a TPU's, so a choice keyed on the
    platform acts as on that device while a step traces for it on another."""
    import jax
    generation = ASSUMED.get()
    if generation is None:
        return jax.default_backend()
    if generation.startswith('sm'):
        return 'gpu'
    if generation in TARGETS and TARGETS[generation].device_kind.startswith('TPU '):
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


def strict_dot(a, b, *, exact: bool = False, ieee: bool = False):
    """Same-dtype Pallas dot with fp32 accumulation; reject mismatches before any interpret-mode
    widening, so CPU tests catch dots Triton rejects. Exact interpretation widens bf16/fp16 inputs
    for CPUs without narrow-input/fp32-accumulating dots.

    Each operand dtype names its products, so no global `jax_default_matmul_precision` changes the
    kernel (the suite's 'highest' made every fp32 product IEEE):
    - bf16 or fp16: exact products summed in fp32 (fla's `tl.dot` of narrow operands); no precision.
    - fp32 with `ieee`: IEEE products, fla's `allow_tf32=False` dots (`recompute_w_u`'s `A @ v`).
    - fp32 otherwise: TF32x3 (`DotAlgorithmPreset.TF32_TF32_F32_X3`), three TF32 products near IEEE
      accuracy where fla's own dots take one TF32 (Triton's default `input_precision` for fp32, and
      `SOLVE_TRIL_DOT_PRECISION`, chunk_fwd.py:21). jax 0.11's Pallas lowers `Precision.HIGH` as
      plain TF32 (pallas/triton/lowering.py `_TF32_PRECISIONS`), so the preset is named. IEEE there
      lowers on sm80 to FMA loops ptxas took 3.5 to over 15 minutes to compile for the WY backward;
      TF32x3 compiles in seconds.
    - interpreted: no precision, the CPU's fp32.
    The kkt merge's TF32 products name `Precision.DEFAULT` themselves (fla_gdn `_kkt_solve_kernel`)."""
    import jax
    import jax.numpy as jnp

    if jnp.dtype(a.dtype) != jnp.dtype(b.dtype):
        raise TypeError(f'Pallas dot operands must have the same dtype, got {a.dtype} and {b.dtype}')
    wide = jnp.dtype(a.dtype) == jnp.float32
    if exact:
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        precision = jax.lax.Precision.HIGHEST if ieee and wide else None
    elif wide:
        precision = jax.lax.Precision.HIGHEST if ieee else jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3
    else:
        precision = None
    return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=precision)


COMPILE_SECONDS = 60.0
"""The CPU time one kernel config's Triton compile may take, ptxas's with it: 8 to 40 seconds,
where an IEEE fp32 dot on sm80 took ptxas 209 seconds to past 15 minutes. CPU time, not wall:
a busy host stretches the wall time of every compile, and a gate that measured it would fail
the compile it slowed. A compile still running at ten times the bound is stopped."""

_TRITON_CALL = "__gpu$xla.gpu.triton"
_COMPILE = """import json, resource, sys
import triton
from triton.backends.compiler import GPUTarget

entry = json.loads(sys.argv[1])
kernel = triton.compile(entry["path"], target=GPUTarget("cuda", entry["capability"], 32),
                        options={"num_warps": entry["warps"], "num_stages": entry["stages"]})
used = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
print(json.dumps({"shared": kernel.metadata.shared, "seconds": sum(u.ru_utime + u.ru_stime for u in used)}))
"""


def _triton_calls(lowered: jax.stages.Lowered) -> list[dict]:
    """Each Pallas-Triton call of a lowered program: its Triton IR as text, its warps and stages."""
    # The custom call's Triton IR is reachable only through JAX's MLIR bindings, jaxlib's and its own.
    from jax._src.interpreters import mlir
    from jax._src.lib.triton import dialect
    from jaxlib.mlir import ir

    found = []

    def visit(operation) -> None:
        if (operation.name == "stablehlo.custom_call"
                and ir.StringAttr(operation.attributes["call_target_name"]).value == _TRITON_CALL):
            config = ir.DictAttr(operation.attributes["mhlo.backend_config"])
            found.append({"ir": ir.StringAttr(config["ir"]).value_bytes,
                          "warps": ir.IntegerAttr(config["num_warps"]).value,
                          "stages": ir.IntegerAttr(config["num_stages"]).value})
        for region in operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    visit(inner.operation)

    module = lowered.compiler_ir("stablehlo")
    if module is None:
        raise ValueError("the lowered program has no StableHLO module")
    visit(module.operation)
    context = mlir.make_ir_context()
    dialect.register_dialect(context)
    with context:
        for call in found:
            call["ir"] = ir.Module.parse(call["ir"]).operation.get_asm(enable_debug_info=False)
    return found


def _compiled(entry: dict, seconds: float) -> dict:
    """One Triton call compiled by pip's Triton in a child of its own session: its shared bytes and
    the CPU seconds it and ptxas took, or why it has none. The whole session is stopped at ten
    times `seconds` of wall time."""
    import os
    import signal
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", _COMPILE, json.dumps(entry)], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = child.communicate(timeout=10 * seconds)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.communicate()
        return {"error": f"no compile within {10 * seconds:.0f} s of wall time", "seconds": seconds}
    if child.returncode:
        failed = f"Triton's compile failed ({child.returncode}): {output.strip()[-800:]}"
        return {"error": failed, "seconds": 0.0}
    found = json.loads(output.strip().splitlines()[-1])
    if found["seconds"] > seconds:
        found["error"] = f"its compile took {found['seconds']:.0f} CPU seconds, past {seconds:.0f}"
    return found


def compile_kernels(module, generations: Sequence[str], seconds: float = COMPILE_SECONDS) -> list[dict]:
    """Each of `module`'s declared kernel configs (`kernel_cases`) on each CUDA generation, lowered as
    on that GPU on this host (`assumed`, `gpu_mesh`) and compiled by pip's Triton for its capability,
    never run: its name, the generation's block limit, and the shared bytes Triton allocates (the
    most of any call it makes), or the error. A compile past `seconds` of CPU time fails.

    A cheap first filter, not a GPU's verdict: a kernel that does not compile, compiles past its
    bound, or takes more shared memory than its GPU's block holds fails here, before a GPU is paid
    for. Identical IR compiles once; compiles run one to each CPU this process may use."""
    import concurrent.futures
    import hashlib
    import os
    import tempfile
    from pathlib import Path

    import jax
    import numpy as np
    from jax.sharding import Mesh

    found: list[dict] = []
    calls: dict[str, tuple[dict, list[dict]]] = {}
    with tempfile.TemporaryDirectory(prefix="dew-compile-") as scratch:
        for target in generations:
            mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), target)
            for case in module.kernel_cases(target):
                record: dict = {"name": case.name, "generation": target,
                                "limit": gpu_target(target).shared_limit, "shared": 0, "seconds": 0.0}
                found.append(record)
                fn, args = case.call()
                try:
                    with assumed(target), jax.sharding.use_abstract_mesh(mesh):
                        lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("cuda",))
                except Exception as error:  # A case that does not lower is its own failure.
                    record["error"] = f"it does not lower: {str(error)[-800:]}"
                    continue
                for call in _triton_calls(lowered):
                    named = f"{target}|{call['warps']}|{call['stages']}|{call['ir']}"
                    key = hashlib.sha256(named.encode()).hexdigest()
                    if key not in calls:
                        path = Path(scratch) / f"{len(calls)}.ttir"
                        path.write_text(call["ir"])
                        calls[key] = ({"path": str(path), "capability": int(target.removeprefix("sm")),
                                       "warps": call["warps"], "stages": call["stages"]}, [])
                    calls[key][1].append(record)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
            entries = [entry for entry, _ in calls.values()]
            outcomes = list(pool.map(lambda entry: _compiled(entry, seconds), entries))
    for (_, records), outcome in zip(calls.values(), outcomes, strict=True):
        for record in records:
            record["seconds"] = max(record["seconds"], outcome["seconds"])
            if "error" in outcome:
                record.setdefault("error", outcome["error"])
            else:
                record["shared"] = max(record["shared"], outcome["shared"])
    return found
