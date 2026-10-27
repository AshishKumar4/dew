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

import contextlib
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import jax
import jax.experimental.topologies as topologies
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from dew.nn.kernels.generation import assumed, capability, gpu_mesh, gpu_target

THREADS_PER_MULTIPROCESSOR = {(8, 0): 2048, (9, 0): 2048, (10, 0): 2048}
"""Resident threads a multiprocessor holds by compute capability, 1536 on the others the targets name
(CUDA C++ Programming Guide, "Technical Specifications per Compute Capability")."""

REFUSED = re.compile(r"Shared memory size limit exceeded: requested (\d+)")
"""XLA's refusal of a Triton kernel whose shared memory a block of the GPU does not hold."""

KERNEL = re.compile(r"kCustomKernel \[[^\]]*\] (\S+) .*? shared_memory: (\d+) bytes")
"""A custom kernel's line in XLA's thunk sequence, with its name and the shared bytes it launches with: a
Triton call's, or one of XLA's own fusions XLA emits so (`loop_multiply_fusion`)."""

TRITON_CALL = re.compile(r'custom_call_target="__gpu\$xla\.gpu\.triton".*?\bname = "([^"]+)"')
"""A Triton kernel's call in XLA's optimized HLO, with the kernel's name, which its launch line carries."""


def target_config(generation: str) -> str:
    """The `gpu_device_info` XLA compiles for, from the CUDA `generation`'s target
    (`generation.TARGETS`): its compute capability, the shared memory a block may opt into, its
    multiprocessors, clock and bandwidth; CUDA's per-capability limits for the rest. The fields past
    the capability and shared memory only steer XLA's own fusions, never a Triton kernel's code."""
    target, version = gpu_target(generation), capability(generation)
    facts = (target.shared_limit, target.multiprocessors, target.clock_hz, target.memory_bandwidth)
    if version is None or any(fact is None for fact in facts):
        raise ValueError(f"no CUDA target states {generation!r}'s compile facts")
    major, minor = version
    shared, multiprocessors, clock, bandwidth = facts
    return f"""gpu_device_info {{
  threads_per_block_limit: 1024
  threads_per_warp: 32
  shared_memory_per_block: 49152
  shared_memory_per_core: {shared + 1024}
  threads_per_core_limit: {THREADS_PER_MULTIPROCESSOR.get((major, minor), 1536)}
  core_count: {multiprocessors}
  fpus_per_core: {64 if (major, minor) == (8, 0) else 128}
  block_dim_limit_x: 2147483647
  block_dim_limit_y: 65535
  block_dim_limit_z: 65535
  memory_bandwidth: {int(bandwidth)}
  clock_rate_ghz: {clock / 1e9}
  shared_memory_per_block_optin: {shared}
  cuda_compute_capability {{
    major: {major}
    minor: {minor}
  }}
  registers_per_core_limit: 65536
  registers_per_block_limit: 65536
}}
platform_name: "CUDA"
device_description_str: "{target.device_kind}"
"""


@functools.cache
def _device(generation: str) -> Mesh:
    """A one-device mesh of a compile-only device of `generation`."""
    topology = topologies.get_topology_desc("dew-compile", "cuda", target_config=target_config(generation),
                                            topology="1x1x1")
    return Mesh(np.array(topology.devices[:1]), ("data",))


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
        calls = sorted({name for text in programs for name in TRITON_CALL.findall(text)})
        # Each Triton call's shared bytes from its own launch line, never XLA's fusions': a call none states,
        # or a program that calls no Triton kernel, is no evidence its kernel compiled within its block
        # (row 1381).
        launched = {call: [int(size) for text in thunks for name, size in KERNEL.findall(text)
                           if name.startswith(call)] for call in calls}
        unlaunched = [call for call, sizes in launched.items() if not sizes]
        if not calls or unlaunched:
            record["error"] = (f"no launch states the shared bytes of {', '.join(unlaunched)}" if calls
                               else "XLA's program calls no Triton kernel")
            continue
        record["shared"] = max(max(sizes) for sizes in launched.values())
    return found


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


MOSAIC_DUMPS = {"MOSAIC_GPU_DUMP_PTXAS": "1", "MOSAIC_GPU_DUMP_RESOURCES": "1"}
"""Mosaic GPU's own compile reports (ptxas's register and spill counts, its resource estimate), written
under `MOSAIC_GPU_DUMP_TO` for each kernel compiled while they are set: Mosaic compiles its kernels, to
PTX and through ptxas, when XLA compiles the program around them, device or none."""
DUMPING = threading.RLock()
"""Held while Mosaic's reports are on (`mosaic_dumps`): the dump settings are the process's
environment, so one scope at a time sets and restores them."""
REGISTERS = re.compile(r"Used (\d+) registers")
SPILLS = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
SMEM = re.compile(r"smem_scratch_bytes=(\d+)")
KERNEL_COUNTER = re.compile(r"_\d+$")
"""The counter Mosaic GPU appends to a kernel's name in its reports' file names."""
DUMP_NAME = re.compile(r'mosaic_gpu\.dump_basename = "[^"]*"')
"""The one attribute of Mosaic's module dump that names the dump (its counter) rather than the module."""
TMEM_USED = re.compile(r"tmem_scratch_cols=(\d+), tmem_collective_scratch_cols=(\d+)")


@contextlib.contextmanager
def mosaic_dumps(directory: str):
    """Mosaic GPU's reports (`MOSAIC_DUMPS`) on, into `directory`, for the compiles in the block, under
    `DUMPING` from before the settings are read to after they are restored."""
    with DUMPING:
        before = {name: os.environ.get(name) for name in (*MOSAIC_DUMPS, "MOSAIC_GPU_DUMP_TO")}
        os.environ.update(MOSAIC_DUMPS, MOSAIC_GPU_DUMP_TO=directory)
        try:
            yield
        finally:
            for name, value in before.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def mosaic_kernels(directory: str) -> list[dict]:
    """Each Mosaic GPU kernel's report in `directory`: its shared bytes and tensor memory columns (Mosaic's
    resource estimate), its registers a thread and its spilled bytes (ptxas's), or `unread`, what a
    report lacks: either report without the other, or a field neither states, is no evidence the kernel
    is within its resources. A case's reports are paired by the kernel's name less the counter Mosaic
    appends (`KERNEL_COUNTER`), which JAX 0.12 numbers apart for the two reports; a case names each kernel
    once."""
    kernels = []
    reports = Path(directory)
    named: dict[str, dict[str, Path]] = {}
    for kind, suffix in (("resources", ".resources.txt"), ("ptxas", ".ptxas"),
                         ("module", ".after_layout_inference.txt")):
        for path in reports.glob(f"*{suffix}"):
            found = named.setdefault(KERNEL_COUNTER.sub("", path.name.removesuffix(suffix)), {})
            # Paired by name: a case that compiles a name twice would pair the wrong reports.
            assert kind not in found, f"a case compiled {path.name} and {found[kind].name}: name them apart"
            found[kind] = path
    for name, found in sorted(named.items()):
        text = found["resources"].read_text() if "resources" in found else ""
        log = found["ptxas"].read_text() if "ptxas" in found else ""
        shared, tmem = SMEM.search(text), TMEM_USED.search(text)
        registers, spills = REGISTERS.findall(log), SPILLS.findall(log)
        unread = [field for field, found in (("shared bytes", shared), ("tensor memory columns", tmem),
                                             ("registers", registers), ("spills", spills)) if not found]
        module = found.get("module")
        kernels.append({"kernel": name, "unread": unread,
                        "module": hashlib.sha256(DUMP_NAME.sub("", module.read_text()).encode()).hexdigest()
                        if module else None,
                        "shared": int(shared.group(1)) if shared else None,
                        "tmem": sum(map(int, tmem.groups())) if tmem else None,
                        "registers": max(map(int, registers), default=None),
                        "spilled": max((int(stores) + int(loads) for stores, loads in spills), default=None)})
    return kernels


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
    own reports on (`mosaic_dumps`): its name, the generation's shared-memory and tensor-memory limits
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
        with tempfile.TemporaryDirectory(prefix="dew-mosaic-") as directory, mosaic_dumps(directory):
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
            where = f"{record['name']} {kernel['kernel']}"
            if kernel["unread"]:
                failed.append(f"{where}: its reports state no {', '.join(kernel['unread'])}")
            elif not (kernel["shared"] <= record["limit"] and kernel["tmem"] <= record["tmem_limit"]
                      and kernel["spilled"] == 0):
                failed.append(f"{where}: {kernel['shared']} shared bytes (limit {record['limit']}), "
                              f"{kernel['tmem']} tensor memory columns (limit {record['tmem_limit']}), "
                              f"{kernel['spilled']} bytes spilled")
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
