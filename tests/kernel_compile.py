"""Compile Pallas kernels for a GPU or a TPU on this CPU with the compiler the device itself runs: the CPU
suite's first filter for declared kernel configs (tests/test_kernel_configs.py) and the template mainloops'
resource gate (Mosaic GPU's shared and tensor memory and spills, `compile_mosaic_kernels`; libtpu's
compile of a Mosaic TPU kernel within its stated VMEM, `compile_tpu_kernels`). XLA's CUDA backend (this
jaxlib's CUDA plugin: the `kernel-compile` extra on a host with no GPU, a GPU host's own CUDA extra)
compiles a program for a compile-only device of the generation, no GPU attached, through the Triton
jaxlib bundles: so a kernel's shared bytes here are the bytes the GPU asks for, the GPU's own refusal
included. A host without the plugin fails here rather than skip. A GPU's tests are the verdict on
everything else.
"""

import dataclasses
import functools
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import jax
import jax.experimental.topologies as topologies
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from dew.nn.kernels.generation import assumed, capability, gpu_mesh, gpu_target
from dew.nn.kernels.templates import report
from dew.training.ahead import compile_only_gpus

REFUSED = re.compile(r"Shared memory size limit exceeded: requested (\d+)")
"""XLA's refusal of a Triton kernel whose shared memory a block of the GPU does not hold."""

KERNEL = re.compile(r"kCustomKernel \[[^\]]*\] (\S+) .*? shared_memory: (\d+) bytes")
"""A custom kernel's line in XLA's thunk sequence, with its name and the shared bytes it launches with: a
Triton call's, or one of XLA's own fusions XLA emits so (`loop_multiply_fusion`)."""

UNIQUIFIED = re.compile(r"__\d+$")
"""The suffix XLA gives a kernel name's later launches in a program (`recurrence_cumulated__1`)."""

TRITON_CALL = re.compile(r'custom_call_target="__gpu\$xla\.gpu\.triton".*?\bname = "([^"]+)"')
"""A Triton kernel's call in XLA's optimized HLO, with the kernel's name, which its launch line carries."""


@functools.cache
def _device(generation: str) -> Mesh:
    """A one-device mesh of a compile-only device of `generation`."""
    return Mesh(np.array(compile_only_gpus(generation)), ("data",))


def refused(record: dict) -> bool:
    """Whether a compiled record failed as a program: its error, or shared bytes past its block's limit
    (a Triton kernel's, or XLA's refusal of a block, which a Mosaic GPU program states only then)."""
    return "error" in record or ("shared" in record and record["shared"] > record["limit"])


def failures(found: list[dict]) -> list[str]:
    """Each compiled record's failure, if any: its error, or its shared bytes past its block's limit."""
    return [f"{record['name']}: {record['error']}" if "error" in record else
            f"{record['name']}: {record['shared']} shared bytes, limit {record['limit']}"
            for record in found if refused(record)]


def compile_kernels(cases, generation: str) -> list[dict]:
    """Each case (`name`, `call()` giving the kernel and abstract operands) on the CUDA `generation`,
    lowered as on that GPU on this host (`assumed`, `gpu_mesh`) and compiled by XLA for a compile-only
    device of it, never run: its name, the generation's block limit, and the shared bytes its Triton
    kernels launch with (the most of any; those XLA refused, where it refused one), or the error.

    A cheap first filter: a kernel that does not compile or takes more shared memory than its GPU's
    block holds fails here as it would there, before a GPU is paid for. A program whose thunk sequence
    launches no Triton kernel fails too: nothing states its kernel's shared bytes."""
    found: list[dict] = []
    for case in cases:
        record: dict = {"name": case.name, "generation": generation,
                        "limit": gpu_target(generation).shared_limit}
        found.append(record)
        sequences = _compiled(case, generation, record)
        if sequences is None:
            continue
        thunks, programs = sequences
        record.update(triton_evidence(programs, thunks))
    return found


def triton_evidence(programs: list[str], thunks: list[str]) -> dict:
    """What XLA's optimized `programs` and thunk sequences (`thunks`) say of a program's Triton kernels:
    the largest shared bytes a call launches with, or why they say nothing. Each Triton call (an HLO
    custom call, named by its kernel) states its shared bytes in a launch line of its own, named as the
    call is, XLA numbering a name's later calls (recurrence_cumulated__1): XLA's own fusions launch as
    custom kernels too, and one kernel's name can begin another's (attention_forward,
    attention_forward_headroom). A call with no launch line of its own, or a program that calls no
    Triton kernel, is no evidence its kernel compiled within its block (row 1381)."""
    calls = Counter(name for text in programs for name in TRITON_CALL.findall(text))
    launches: dict[str, list[int]] = {}
    for text in thunks:
        for name, size in KERNEL.findall(text):
            launches.setdefault(UNIQUIFIED.sub("", name), []).append(int(size))
    if not calls:
        return {"error": "XLA's program calls no Triton kernel"}
    unlaunched = sorted(name for name, count in calls.items() if len(launches.get(name, ())) < count)
    if unlaunched:
        return {"error": f"no launch of its own states the shared bytes of {', '.join(unlaunched)}"}
    return {"shared": max(size for name in calls for size in launches[name])}


def _compiled(case, generation: str, record: dict) -> tuple[list[str], list[str]] | None:
    """`case` lowered as on the CUDA `generation` (`assumed`, `gpu_mesh`) and compiled by XLA for a
    compile-only device of it: XLA's thunk sequences and optimized programs, or None, `record` holding why:
    the error, or the shared bytes of a block XLA refused."""
    mesh = _device(generation)
    replicated = NamedSharding(mesh, PartitionSpec())
    fn, args = case.call()
    placed = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=replicated)
                          if isinstance(leaf, jax.ShapeDtypeStruct) else leaf, args)
    try:
        with assumed(generation), jax.sharding.use_abstract_mesh(gpu_mesh(mesh, generation)):
            lowered = jax.jit(fn).trace(*placed).lower()
    except Exception as error:  # A case that does not lower is its own failure.
        record["error"] = f"it does not lower: {str(error)[-800:]}"
        return None
    # A process whose persistent compilation cache is on keeps it on here (JAX settles that once a
    # process): a compile-only device's entry reads back as unloadable, with a warning, and compiles.
    with tempfile.TemporaryDirectory(prefix="dew-compile-") as dump:
        try:
            lowered.compile(compiler_options={"xla_dump_to": dump})
        except jax.errors.JaxRuntimeError as error:
            refused = REFUSED.search(str(error))
            if refused is None:
                record["error"] = f"XLA does not compile it: {str(error)[-800:]}"
            else:
                record["shared"] = int(refused.group(1))
            return None
        return ([path.read_text() for path in Path(dump).glob("*thunk_sequence.txt")],
                [path.read_text() for path in Path(dump).glob("*after_optimizations.txt")])


def mosaic_kernels(directory: str) -> list[dict]:
    """Each Mosaic GPU kernel's report in `directory` (`report.reports`), as a record a test's process of its
    own writes back as JSON (`compile_mosaic_kernels`)."""
    return [dataclasses.asdict(found) for found in report.reports(directory)]


OWN_PROCESS = "DEW_MOSAIC_RECORDS"
"""Where a test re-run in a process of its own (`compile_mosaic_kernels`) writes its records."""

_BINARY_CAPABILITY: list[tuple[int, int]] = []
"""The compute capability this process's Mosaic GPU binary pass is registered with: the first kernel's it
compiled. jaxlib registers the pass once a process and assembles every later kernel for that capability
(custom_call.cc's GetPassPipeline; an sm_90a kernel after an sm80 one fails in ptxas, 'SM version
specified by .target is higher than default SM version assumed', and an sm80 kernel after an sm90 one is
assembled for sm90). Production compiles for one GPU a process; the gates compile for several."""


def compile_mosaic_kernels(cases, generation: str) -> list[dict]:
    """Each Mosaic GPU case (`name`, `call()` giving the kernel and abstract operands) compiled as
    `compile_kernels` compiles Triton's, for a compile-only device of the CUDA `generation`, with Mosaic's
    own reports on (`report.reported`): its name, the generation's shared-memory and tensor-memory limits
    (`GpuTarget.tmem_columns`, 0 where it has none), and each of its kernels' report (`mosaic_kernels`),
    or the error. In a process that compiled Mosaic kernels for another capability (`_BINARY_CAPABILITY`)
    the calling test runs again in a process of its own, and these are its records there."""
    wanted = capability(generation)
    if _BINARY_CAPABILITY and _BINARY_CAPABILITY[0] != wanted and OWN_PROCESS not in os.environ:
        return compiled_apart()
    _BINARY_CAPABILITY[:] = _BINARY_CAPABILITY or [wanted]
    found = _compiled_here(cases, generation)
    if OWN_PROCESS in os.environ:
        Path(os.environ[OWN_PROCESS]).write_text(json.dumps(found))
    return found


def compiled_apart() -> list[dict]:
    """The records the calling test's `compile_mosaic_kernels` call gives in a pytest process of its own, its
    first Mosaic GPU compile its generation's: read only from a process that passed, as its records are
    written before its test's own checks and teardown, which a nonzero exit says failed (row 1877)."""
    test = os.environ["PYTEST_CURRENT_TEST"].rsplit(" ", 1)[0]
    with tempfile.TemporaryDirectory(prefix="dew-mosaic-apart-") as directory:
        records = Path(directory) / "records.json"
        ran = subprocess.run([sys.executable, "-m", "pytest", test, "-q", "-p", "no:cacheprovider"],
                             cwd=Path(__file__).parent.parent, env={**os.environ, OWN_PROCESS: str(records)},
                             capture_output=True, text=True, check=False)
        if ran.returncode != 0 or not records.exists():
            raise RuntimeError(f"{test} in a process of its own exited {ran.returncode}"
                               f"{'' if records.exists() else ', its records unwritten'}:\n"
                               f"{(ran.stdout + ran.stderr)[-3000:]}")
        return json.loads(records.read_text())


PTXAS_REPORTS: dict[str, tuple[int, int]] = {}
"""Each Mosaic GPU kernel's ptxas report this process read, its registers and spilled bytes, by its module
(the digest of Mosaic's module after layout inference, `mosaic_kernels`): Mosaic compiles a module once a
process, so a later case of the same module dumps the module and its resources, never PTX or ptxas's
report (`inherited`)."""


def inherited(kernel: dict) -> dict:
    """`kernel`'s record (`mosaic_kernels`) with its registers and spills where ptxas reported them: its
    own, which this process keeps by its module (`PTXAS_REPORTS`), or, where Mosaic compiled the same module
    before in this process and so ran no ptxas, that compile's. A kernel whose module no report in this
    process covers keeps them unread."""
    ptxas = {"registers", "spills"}
    if not ptxas & set(kernel["unread"]):
        if kernel["module"] is not None:
            PTXAS_REPORTS[kernel["module"]] = (kernel["registers"], kernel["spilled"])
        return kernel
    if kernel["module"] not in PTXAS_REPORTS:
        return kernel
    registers, spilled = PTXAS_REPORTS[kernel["module"]]
    return {**kernel, "registers": registers, "spilled": spilled, "ptxas": "an earlier compile of its module",
            "unread": [field for field in kernel["unread"] if field not in ptxas]}


def _compiled_here(cases, generation: str) -> list[dict]:
    """`compile_mosaic_kernels`' records, compiled in this process."""
    found: list[dict] = []
    for case in cases:
        # The program's own record (its error, or XLA's refusal of its block), its Mosaic kernels' reports the
        # evidence of their resources: a Triton launch line is none of theirs.
        record: dict = {"name": case.name, "generation": generation,
                        "limit": gpu_target(generation).shared_limit}
        with tempfile.TemporaryDirectory(prefix="dew-mosaic-") as directory, report.reported(directory):
            _compiled(case, generation, record)
            record["kernels"] = [inherited(kernel) for kernel in mosaic_kernels(directory)]
        record["tmem_limit"] = gpu_target(generation).tmem_columns or 0
        found.append(record)
    return found


def mosaic_failures(found: list[dict]) -> list[str]:
    """Each Mosaic GPU record's failure: the program's own (`failures`: its error, or XLA's refusal of a
    block past shared memory), no Mosaic kernel compiled, a kernel whose reports lack a field
    (`mosaic_kernels`), or a kernel past its multiprocessor's shared memory or tensor memory, or
    spilling registers."""
    failed = failures(found)
    for record in found:
        if refused(record):
            continue
        if not record["kernels"]:
            failed.append(f"{record['name']}: no Mosaic GPU kernel compiled")
        for kernel in record.get("kernels", ()):
            # A record's fields past the report's (`inherited`'s `ptxas`, where its registers came from)
            # are its own.
            stated = {field.name: kernel[field.name] for field in dataclasses.fields(report.Report)
                      if field.name in kernel}
            found = report.Report(**{**stated, "unread": tuple(kernel["unread"])})
            if (why := found.failure(record["limit"], record["tmem_limit"])) is not None:
                failed.append(f"{record['name']} {found.kernel}: {why}")
    return failed


TPU_TOPOLOGIES = {"v5e": "v5e:2x2", "v6e": "v6e:2x2"}
"""The compile-only topology of each TPU generation the gate compiles for: libtpu describes a host's
chips (2x2), of which a case takes one."""


@functools.cache
def _tpu_device(generation: str) -> Mesh:
    """A one-device mesh of a compile-only TPU of `generation`."""
    topology = topologies.get_topology_desc(TPU_TOPOLOGIES[generation], "tpu")
    return Mesh(np.array(topology.devices[:1]), ("data",))


def compile_tpu_kernels(cases, generation: str) -> list[dict]:
    """Each case compiled by libtpu for a compile-only TPU of `generation`, traced under its abstract mesh
    so a kernel's `get_tpu_info` reads that TPU: its name and generation, or the error, Mosaic's refusal
    of a kernel past its stated VMEM limit included."""
    mesh = _tpu_device(generation)
    replicated = NamedSharding(mesh, PartitionSpec())
    # The generation assumed as well as its mesh, so a kernel's facts are its target's
    # (`dew.nn.kernels.splash.block_sizes`' scoped VMEM).
    found: list[dict] = []
    for case in cases:
        record: dict = {"name": case.name, "generation": generation}
        found.append(record)
        fn, args = case.call()
        placed = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=replicated)
                              if isinstance(leaf, jax.ShapeDtypeStruct) else leaf, args)
        try:
            with assumed(generation), jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
                lowered = jax.jit(fn).trace(*placed).lower()
            lowered.compile()
        except Exception as error:  # A case that does not lower or compile is its own failure.
            # Mosaic TPU's refusal leads with its reason and ends with diagnostics: both ends kept.
            text = str(error)
            text = text if len(text) <= 1600 else f"{text[:800]} ... {text[-800:]}"
            record["error"] = f"{type(error).__name__}: {text}"
    return found
