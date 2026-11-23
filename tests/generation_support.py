"""Which assumed GPU generations a test may run its kernels under in this process.

`generation.assumed(name)` routes a call as a `name` GPU would. Lowering under it runs anywhere; running
the routed kernels needs a process that can: a CPU interprets any generation's kernels, and a GPU runs its
own generation's and an older one's (Triton's sm80 kernels on an H100), not a newer one's (Mosaic GPU's
Hopper kernels on an A100, which refuse its pre-Hopper pipelines). The Mosaic GPU interpreter itself
runs in a CPU process only: on a GPU host it reads the device and refuses as the compiler does; and
Mosaic GPU's lowering checks a kernel against the process's own device and lowers its copies for it, so
lowering a generation's Mosaic GPU kernels runs on a CPU or on a GPU that copies as it does (`lowers_as`).
"""

import contextlib
import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import tier_wire
from flax.linen.dtypes import canonicalize_dtype

from dew.nn.kernels import gemm, generation


def interprets() -> bool:
    """Whether this process's kernels run interpreted as its run runs them: off a GPU, but not in a GPU tier
    part's prepare, which states them as the node's GPU runs them (`here`), so the run reads its arms from
    the store. Read at import off a GPU alone, a prepare on the node's CPU stated its arm interpreted and the
    run's was one it never drew (row 8750's MissingTruth)."""
    return not tier_wire.preparing() and jax.default_backend() != "gpu"


TOPOLOGY = tier_wire.TOPOLOGY
"""The node's GPUs' serialized topology, the tier's contract's variable (`tier_wire.TOPOLOGY`)."""


def node_target():
    """The node's GPU (`TOPOLOGY`): its generation, its kind's (`generation.gpu_target`), and the facts its
    serialized device states of itself, its multiprocessors and its block's shared memory among them
    (`generation.runtime_facts`), read with no GPU visible, as a prepare runs, so nothing of the CUDA
    driver's: what its run's load holds the programs to (`BuildTarget.held`). `here` reads it as this module
    imports, which gpu_flow imports in turn: it holds nothing of the flows' module, whose import takes
    gpu_flow's (row 9540)."""
    from jax.experimental import topologies

    # The CUDA plugin, which a serialized topology deserializes through, initialized with no GPU and no
    # client (`dew.training.ahead.host_gpus`).
    generation.compile_only_gpus("sm80", 1)
    device = topologies.TopologyDescription.deserialize(Path(os.environ[TOPOLOGY]).read_bytes()).devices[0]
    return generation.runtime_facts(generation.gpu_target(device.device_kind), device,
                                    read=lambda ordinal, attribute: None)


def node_topology(path: Path, gpu: str = "sm80") -> Path:
    """`path` holding a node's serialized topology (`TOPOLOGY`'s, `node_target`'s), one compile-only GPU of
    generation `gpu` serialized as a host serializes its own: a prepare's node in a CPU process."""
    from jax.experimental import topologies

    path.write_bytes(topologies.TopologyDescription(list(generation.compile_only_gpus(gpu, 1))).serialize())
    return path


def here() -> str:
    """This process's device generation as its kernels run: the node's GPU's in a GPU tier part's prepare
    (`node_target`), whose default device, a compile-only GPU, states no generation of its own, so a cell
    pinned to the node's generation prepares there as its run runs it (train 1013's review F4); else its
    device's. The one prepare-aware reading of it: `runnable`, `lowers_as` and every matrix case's pin (a
    `skipif` its module evaluates at import, rows 7868 and 7886) read it."""
    if tier_wire.preparing():
        return node_target().generation
    return generation.device_generation()


_ON_CPU = here() == "cpu"
INTERPRETS = pytest.mark.skipif(not _ON_CPU,
                                reason="interprets the Mosaic GPU kernels, which a CPU process alone does")
COMPILES_ON_CPU = pytest.mark.skipif(
    not _ON_CPU,
    reason="compiles kernels ahead for compile-only GPUs, a CPU process's work (bible §8: no GPU is held "
           "for host compiles), judging each by every CUDA build's ptxas, whose nvcc wheels the CPU "
           "environment alone installs; a GPU process's Mosaic prints a module's PTX at its first compile "
           "alone, so the gate's re-run of a module it compiled for its own device reads none (EM-GAC, rows "
           "7355/7356, 29b3c8921)")
"""A mark placing a resource gate (`kernel_compile.mosaic_failures`) of kernels compiled ahead for
compile-only GPUs in a CPU process, where its every build's ptxas is installed and each module's PTX is
printed for it."""
SUMS_IN_IEEE = pytest.mark.skipif(
    not _ON_CPU,
    reason="compares sums bit for bit as a CPU's IEEE fp32 dots give them, which a GPU's tensor cores, "
           "truncating each add, do not")


def pinned(case_generation: str):
    """A matrix case's pin (`pytest.param(case, marks=pinned(...))`): its cell skips unless this device's
    generation (`here`, the node's in a GPU part's prepare) is the case's, `case_generation`."""
    return pytest.mark.skipif(here() != case_generation,
                              reason=f"a {case_generation} cell runs on its own device alone")


def runnable(name: str) -> bool:
    """Whether this process runs the kernels `generation.assumed(name)` routes to: a CPU any (it
    interprets them); a TPU its own generation's; a GPU Hopper's warpgroup kernels only if it is that
    generation (capability 9.0 alone has the instructions), and any other generation's only if no newer than
    itself and its block
    holds the shared memory `name`'s launches are sized to (an L4's 99 KiB do not an A100's 163)."""
    current = here()
    if not current.startswith("sm") or generation.capability(name) is None:
        # A TPU's launches a CPU interprets, and that TPU runs.
        return current in ("cpu", name)
    if generation.gpu_target(name).wgmma:
        return name == current
    return (generation.capability(name) <= generation.capability(current)
            and generation.shared_budget(generation.gpu_target(name))
            <= generation.shared_budget(generation.gpu_target(current if tier_wire.preparing() else None)))


def runs(name: str):
    """A mark skipping a test that runs kernels under `generation.assumed(name)` where this process
    cannot (`runnable`)."""
    return pytest.mark.skipif(not runnable(name),
                              reason=f"runs {name}'s kernels, which this process's device does not")


def lowers_as(name: str) -> bool:
    """Whether this process runs `name`'s Mosaic GPU kernels as `name` lowers them: where it runs them
    (`runnable`), on a CPU, or on a GPU that copies as `name` does. Mosaic GPU lowers a kernel for the
    process's own device, each copy by its architecture (TMA from Hopper, cp.async before:
    `plgpu.copy_gmem_to_smem`), so an A100's pipelines on an H100 are another kernel, and their cp.async
    waits are refused there ("wait_gmem_to_smem is only supported on pre-Hopper GPUs", main's GH200 tier,
    709ddea7a)."""
    if not runnable(name):
        return False
    current = here()
    return current == "cpu" or generation.architecture(name).tma == generation.architecture(current).tma


def lowers(name: str):
    """A mark skipping a test that lowers `name`'s Mosaic GPU kernels where this process does not run
    them as `name` lowers them (`lowers_as`): Mosaic GPU checks a kernel against the process's own device
    (an A100's 166912 bytes of shared memory refuse an H100's tile) and lowers its copies for it."""
    return pytest.mark.skipif(not lowers_as(name),
                              reason=f"lowers {name}'s Mosaic GPU kernels, which this process's device "
                                     "does not run as they lower")


COMPILES_FOR_TPU = pytest.mark.skipif(
    not (jax.default_backend() == "tpu"
         or (jax.default_backend() == "cpu" and importlib.util.find_spec("libtpu") is not None)),
    reason="compiles for a compile-only TPU, which libtpu does in a CPU process with it installed, or a TPU")
"""A mark skipping a test that compiles kernels for a TPU (`kernel_compile.compile_tpu_kernels`) where this
process cannot: a GPU host's process has no TPU toolchain."""


def generations(*names: str) -> list:
    """`names` as parameters, each marked `runs`, for a test parametrized over assumed generations."""
    return [pytest.param(name, marks=runs(name)) for name in names]


def interpreted_projection(x, kernel, group_sizes, dtype, precision=None):
    """`moe.expert_projection` on the GEMM template's kernels, which a CPU process interprets as the GPU
    `generation.assumed` names (the template's explicit interpreter, `interpret_on_cpu`; product code
    never asks for it): the projection's law, its launches each selected as on that GPU."""
    compute = canonicalize_dtype(x, kernel, dtype=dtype)
    return gemm.gemm(x, kernel, group_sizes, compute=compute, out_dtype=compute,
                         precision=precision, interpret_on_cpu=True)


CUBLAS = {"xla_gpu_enable_triton_gemm": False}
"""XLA's options that make its GEMMs cuBLAS's at any lane flag: its Triton GEMMs off."""


def cublas_compiled(fn, inputs, call: str):
    """`fn`, a program whose dots are a library's GEMMs, compiled as cuBLAS runs each (`CUBLAS`): a library
    arm held to cuBLAS's on the same inputs (docs/design/testing.md). On a GPU its compiled program on
    `inputs` is checked, by its instructions as XLA's own schema reads them
    (`dew.telemetry.instrumentation.hlo_instructions`), to call cuBLAS's `call` and to run no dot of XLA's
    own, fused or not, so no lane flag or fusion choice gives it another GEMM: `__cublas$lt$matmul`, as XLA
    calls a GEMM through cuBLASLt on a Hopper (a deviceless sm90 compile's), cuBLASLt's FP8
    `__cublas$lt$matmul$f8`. On a CPU its dots are XLA's own."""
    from reference_error import without_ynn

    from dew.telemetry.instrumentation import hlo_instructions

    compiled = without_ynn(fn, compiler_options=CUBLAS)
    if jax.default_backend() == "gpu":
        instructions = hlo_instructions(compiled.lower(*inputs).compile()).values()
        targets = {each.target for each in instructions} - {""}
        assert call in targets, f"the library arm calls {sorted(targets)}, no {call}"
        assert not any(each.op == "dot" or "dot" in each.fused for each in instructions), \
            "the library arm runs a dot of XLA's own"
    return compiled


PRODUCTION = ("qwen3.5-27b-gdn", "kimi-linear-kda")
"""The geometries whose 16-bit calls the recurrence resource gate requires to derive tiles on every family,
Dew's production cells (tests/test_recurrence_mosaic.py's `production_refusals`)."""

FAST_BACKWARD_REFUSED = {("qwen3.5-27b-gdn", "sm120"): 10153}
"""The production geometries' 16-bit training whose fast backward does not hold at its derived tiles on a
target (`template.backward_holds`), its gradient the port's or the portable math's there, each by the open
row tracking it. The gate fails a refusal not declared here and a declared one that holds again, so a
declaration retires with its row (`production_refusals`)."""


def production_refusals(target, group: tuple, refused: list[str]) -> list[str]:
    """The recurrence resource gate's failures of `group` (`GATE_GROUPS`') on `target` over the calls it
    refused (`refused`): a production geometry's 16-bit call that derived no tiles, or whose fast backward
    was refused where `FAST_BACKWARD_REFUSED` declares none; and, where it declares one, a training group at
    its derived tiles whose fast backward holds."""
    name, dtype, training, _, variant = group
    if name not in PRODUCTION or dtype != jnp.bfloat16:
        return []
    backward = [case for case in refused if "fast backward" in case]
    failed = [f"{case}: no tiles derived" for case in refused if case not in backward]
    declared = FAST_BACKWARD_REFUSED.get((name, str(target)))
    if declared is None:
        failed += [f"{case}: its fast backward does not hold, no open row declaring it" for case in backward]
    elif training and not isinstance(variant, int) and not backward:
        failed.append(f"{name} on {target}: its fast backward holds; strike it from FAST_BACKWARD_REFUSED "
                      f"(row {declared})")
    return failed


def on_the_host():
    """The host's CPU as the default device: where a cell's interpreted kernels and eager host work run, a
    GPU tier's prepare computing on no device (its default a compile-only GPU), and an interpreter's
    callbacks making no program its store holds (`reference_error.ahead_arm`)."""
    return jax.default_device(jax.devices("cpu")[0])


@contextlib.contextmanager
def on_a_v5e():
    """A TPU v5e's abstract mesh and target, its modules lowered from the host's CPU: a TPU module's lowering
    reads its default device's backend, which a tier's prepare's compile-only GPU is none of, and its
    target, which that prepare assumes its node's GPU for."""
    from jax.sharding import AbstractDevice, AbstractMesh

    device = AbstractDevice(device_kind=generation.TARGETS["v5e"].device_kind, num_cores=1, platform="tpu")
    with on_the_host(), generation.assumed("v5e"), jax.sharding.use_abstract_mesh(
            AbstractMesh((1,), ("data",), abstract_device=device)):
        yield
