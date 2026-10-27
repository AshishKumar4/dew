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

from dew.nn.kernels.generation import assumed, gpu_mesh, gpu_target
from dew.training.ahead import compile_only_gpus

REFUSED = re.compile(r"Shared memory size limit exceeded: requested (\d+)")
"""XLA's refusal of a Triton kernel whose shared memory a block of the GPU does not hold."""

KERNEL = re.compile(r"kCustomKernel .*? shared_memory: (\d+) bytes")
"""A custom (Triton) kernel's line in XLA's thunk sequence, with the shared bytes it launches with."""


@functools.cache
def _device(generation: str) -> Mesh:
    """A one-device mesh of a compile-only device of `generation`."""
    return Mesh(np.array(compile_only_gpus(generation)), ("data",))


def failures(found: list[dict]) -> list[str]:
    """Each compiled record's failure, if any: its error, or its shared bytes past its block's limit."""
    return [f"{record['name']}: {record['error']}" if "error" in record else
            f"{record['name']}: {record['shared']} shared bytes, limit {record['limit']}"
            for record in found if "error" in record or not record["shared"] <= record["limit"]]


def compile_kernels(cases, generation: str) -> list[dict]:
    """Each case (`name`, `call()` giving the kernel and abstract operands) on the CUDA `generation`,
    lowered as on that GPU on this host (`assumed`, `gpu_mesh`) and compiled by XLA for a compile-only
    device of it, never run: its name, the generation's block limit, and the shared bytes its Triton
    kernels launch with (the most of any; those XLA refused, where it refused one), or the error.

    A cheap first filter: a kernel that does not compile or takes more shared memory than its GPU's
    block holds fails here as it would there, before a GPU is paid for."""
    mesh = _device(generation)
    replicated = NamedSharding(mesh, PartitionSpec())
    found: list[dict] = []
    for case in cases:
        record: dict = {"name": case.name, "generation": generation,
                        "limit": gpu_target(generation).shared_limit, "shared": 0}
        found.append(record)
        fn, args = case.call()
        placed = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=replicated)
                              if isinstance(leaf, jax.ShapeDtypeStruct) else leaf, args)
        try:
            with assumed(generation), jax.sharding.use_abstract_mesh(gpu_mesh(mesh, generation)):
                lowered = jax.jit(fn).trace(*placed).lower()
        except Exception as error:  # A case that does not lower is its own failure.
            record["error"] = f"it does not lower: {str(error)[-800:]}"
            continue
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
                continue
            sequences = [path.read_text() for path in Path(dump).glob("*thunk_sequence.txt")]
        record["shared"] = max((int(size) for text in sequences for size in KERNEL.findall(text)), default=0)
    return found


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
    for kind, suffix in (("resources", ".resources.txt"), ("ptxas", ".ptxas")):
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
        kernels.append({"kernel": name, "unread": unread,
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
        return _compiled_apart()
    _BINARY_CAPABILITY[:] = _BINARY_CAPABILITY or [wanted]
    found = _compiled_here(cases, generation)
    if OWN_PROCESS in os.environ:
        Path(os.environ[OWN_PROCESS]).write_text(json.dumps(found))
    return found


def _compiled_apart() -> list[dict]:
    """The records the calling test's `compile_mosaic_kernels` call gives in a pytest process of its own, its
    first Mosaic GPU compile its generation's."""
    test = os.environ["PYTEST_CURRENT_TEST"].rsplit(" ", 1)[0]
    with tempfile.TemporaryDirectory(prefix="dew-mosaic-apart-") as directory:
        records = Path(directory) / "records.json"
        ran = subprocess.run([sys.executable, "-m", "pytest", test, "-q", "-p", "no:cacheprovider"],
                             cwd=Path(__file__).parent.parent, env={**os.environ, OWN_PROCESS: str(records)},
                             capture_output=True, text=True, check=False)
        if not records.exists():
            raise RuntimeError(f"{test} gave no records in a process of its own:\n"
                               f"{(ran.stdout + ran.stderr)[-3000:]}")
        return json.loads(records.read_text())


def _compiled_here(cases, generation: str) -> list[dict]:
    """`compile_mosaic_kernels`' records, compiled in this process."""
    found: list[dict] = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="dew-mosaic-") as directory, mosaic_dumps(directory):
            [record] = compile_kernels([case], generation)
            record["kernels"] = mosaic_kernels(directory)
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
        if "error" in record or not record["shared"] <= record["limit"]:
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
