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
import importlib.util
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
from dew.nn.kernels.templates import mosaic, report
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


NEEDS: dict[str, int | dict | None] = {}
"""Each Mosaic GPU kernel's register need this process measured (`register_need`), by its module, as
`PTXAS_REPORTS` keeps its report: a later compile of the same module dumps no PTX."""


def needed(kernel: dict, needs: dict[str, int | dict | None]) -> dict:
    """`kernel`'s record with its register need: from the PTX its compile dumped (`needs`, by kernel name),
    which this process keeps by its module (`NEEDS`), or, where Mosaic compiled the same module before in
    this process and dumped none, that compile's."""
    if kernel["kernel"] in needs:
        if kernel["module"] is not None:
            NEEDS[kernel["module"]] = needs[kernel["kernel"]]
        return {**kernel, "need": needs[kernel["kernel"]]}
    return {**kernel, "need": NEEDS.get(kernel["module"])}


_NVCC = importlib.util.find_spec("nvidia.cuda_nvcc")
PTXAS = Path(next(iter(_NVCC.submodule_search_locations or ()))) / "bin" / "ptxas" if _NVCC else Path("ptxas")
"""The ptxas of the CUDA toolchain this jaxlib's Mosaic GPU compiles with (the nvidia-cuda-nvcc wheel's)."""

ENTRY = re.compile(r"(\.maxntid[^\n]*\n)")
OCCUPANCY = re.compile(r"\.minnctapersm[^\n]*\n")
RAISE = re.compile(r"setmaxnreg\.inc\.sync\.aligned\.u32 \d+;")


def _ptxas(ptx: str, generation: str, *, cap: int, raised: int | None = None) -> tuple[int, int, bool]:
    """ptxas's registers a thread and spilled bytes for the kernel `ptx` on the CUDA `generation`, its entry
    capped at `cap` registers (`.maxnreg`, its occupancy hint dropped) and, where its consumers raise their
    registers (`setmaxnreg.inc`), that raise rewritten to `raised`; and whether ptxas took it."""
    capped = OCCUPANCY.sub("", ENTRY.sub(lambda entry: f"{entry.group(1)}.maxnreg {cap}\n", ptx, count=1))
    if raised is not None:
        capped = RAISE.sub(f"setmaxnreg.inc.sync.aligned.u32 {raised};", capped)
    major, minor = capability(generation) or (0, 0)
    arch = f"sm_{major}{minor}{'a' if major >= 9 else ''}"
    with tempfile.NamedTemporaryFile("w", suffix=".ptx") as source:
        source.write(capped)
        source.flush()
        ran = subprocess.run([str(PTXAS), f"-arch={arch}", "-v", source.name, "-o", os.devnull],
                             capture_output=True, text=True, check=False)
    text = ran.stdout + ran.stderr
    used = [int(count) for count in report.REGISTERS.findall(text)]
    spilled = [int(stores) + int(loads) for stores, loads in report.SPILLS.findall(text)]
    return max(used, default=0), max(spilled, default=0), ran.returncode == 0


def register_need(ptx: str, generation: str) -> int | None:
    """The registers a thread of the kernel `ptx` needs without spilling, by ptxas itself, in the allocator's
    steps of 8: the least cap at which it spills nothing (ptxas's own count fills whatever a cap leaves it,
    201 or 255 at one tile by the output's width, GEMMMODEL2), or, for one whose consumer warpgroups raise
    their registers (warpgroup MMA's `setmaxnreg`), the least raise at which it spills nothing, or its use
    there where ptxas compiles past a raise below it. A cap holds only if its actual use fits too:
    C7507 can ignore a small raise entirely. Scan in order, since ptxas's spill decisions are not
    monotone even once ignored raises are excluded. None where nothing holds it."""
    raising = bool(RAISE.search(ptx))
    steps = [*range(64, 256, 8), *(() if raising else (255,))]

    def holds(step: int) -> bool:
        used, spilled, took = (_ptxas(ptx, generation, cap=255, raised=step) if raising
                               else _ptxas(ptx, generation, cap=step))
        return took and not spilled and used <= step

    # ptxas may ignore a too-small setmaxnreg, and allocation/spill heuristics
    # themselves are not monotone even once that case is excluded.
    least = next((step for step in steps if holds(step)), None)
    if least is None or not raising:
        return least
    return max(least, _ptxas(ptx, generation, cap=255, raised=least)[0])


def register_decisions(ptx: str, generation: str, model: dict) -> dict:
    """Direct assembly evidence at the launch budget and each possible residency's register cap.

    Neither spills nor actual use are monotone in a requested cap: ptxas can ignore a small raise
    (C7507), or spill at a larger cap after fitting at a smaller one. A least spill-free cap therefore
    cannot decide whether the actual launch holds, or how many copies fit a multiprocessor.
    """
    raising = bool(RAISE.search(ptx))
    budget = model["budget"]
    limit = 256 if raising else 255
    step = model.get("register_step", mosaic.Family.register_step)
    checked = {}

    def holds(requested: int) -> bool:
        cap = min(requested, limit)
        if cap <= 0:
            return False
        if cap not in checked:
            used, spilled, took = (_ptxas(ptx, generation, cap=255, raised=cap) if raising
                                   else _ptxas(ptx, generation, cap=cap))
            checked[cap] = {"cap": cap, "registers": used, "spilled": spilled, "compiled": took,
                            "holds": took and not spilled and used <= cap}
        return checked[cap]["holds"]

    admitted = holds(budget)
    refusal_holds = holds(budget - step)
    programs = 0
    refusal_programs = 0
    for count in range(model["most"], 0, -1):
        # Each block's producer keeps its own registers; the remaining words are shared by
        # its consumers in the family's allocation steps.
        cap = min(budget, ((REGISTER_FILE // (128 * count) - model["producer"])
                          // model["consumers"] // step * step))
        if admitted and not programs and holds(cap):
            programs = count
        if refusal_holds and not refusal_programs and holds(cap - step):
            refusal_programs = count
    return {"admitted": admitted, "programs": programs, "budget": budget,
            "refusal_holds": refusal_holds, "refusal_programs": refusal_programs,
            "caps": list(checked.values())}


def _needs(directory: str, generation: str, models: dict | None = None) -> dict[str, int | dict | None]:
    """Each kernel's register need (`register_need`) from the PTX Mosaic dumped into `directory`, by the
    kernel's name less Mosaic's counter (`report.KERNEL_COUNTER`)."""
    found = {}
    for path in Path(directory).glob("*.ptx"):
        name = report.KERNEL_COUNTER.sub("", path.name.removesuffix(".ptx"))
        model = (models or {}).get(name)
        found[name] = (register_decisions(path.read_text(), generation, model) if model is not None
                       else register_need(path.read_text(), generation))
    return found


def _compiled_here(cases, generation: str) -> list[dict]:
    """`compile_mosaic_kernels`' records, compiled in this process, each kernel's register need beside its
    reports (`register_need`, from the PTX Mosaic dumps)."""
    found: list[dict] = []
    for case in cases:
        # The program's own record (its error, or XLA's refusal of its block), its Mosaic kernels' reports the
        # evidence of their resources: a Triton launch line is none of theirs.
        record: dict = {"name": case.name, "generation": generation,
                        "limit": gpu_target(generation).shared_limit, "models": dict(case.models or {})}
        with (tempfile.TemporaryDirectory(prefix="dew-mosaic-") as directory,
              report.reported(directory, ptx=True)):
            _compiled(case, generation, record)
            needs = _needs(directory, generation, record["models"])
            record["kernels"] = [needed(inherited(kernel), needs) for kernel in mosaic_kernels(directory)]
        record["tmem_limit"] = gpu_target(generation).tmem_columns or 0
        found.append(record)
    return found


REGISTER_FILE = 64 * 1024
"""A multiprocessor's 32-bit registers (every CUDA generation the templates claim)."""


def decisions(registers: int | dict | None, model: dict) -> tuple[bool, int]:
    """What a launch decides from a consumer thread's `registers`: whether they fit its budget (admitted, or
    refused for a smaller tile), and the programs a multiprocessor runs by registers, the family's most at
    most, each program's consumers at `registers` in the allocator's steps of 8 beside its producer's.
    None registers (no cap holds the kernel) fit nothing."""
    if isinstance(registers, dict):
        if registers["budget"] != model["budget"]:
            raise ValueError("register evidence belongs to another launch budget")
        if "registers" in model:
            predicted = decisions(model["registers"], model)
            if not predicted[0]:
                return ((True, registers["refusal_programs"]) if registers["refusal_holds"]
                        else predicted)
            if not registers["admitted"]:
                return False, 0
            if registers["refusal_programs"] <= predicted[1] <= registers["programs"]:
                return predicted
            if predicted[1] < registers["refusal_programs"]:
                return True, registers["refusal_programs"]
        return registers["admitted"], registers["programs"]
    if registers is None or registers > model["budget"]:
        return False, 0
    step = model.get("register_step", mosaic.Family.register_step)
    program = (model["consumers"] * -(-registers // step) * step + model["producer"]) * 128
    return True, min(model["most"], REGISTER_FILE // program)


def mosaic_failures(found: list[dict]) -> list[str]:
    """Each Mosaic GPU record's failure: the program's own (`failures`: its error, or XLA's refusal of a
    block past shared memory), no Mosaic kernel compiled, a kernel whose reports lack a field
    (`mosaic_kernels`), a kernel past its multiprocessor's shared memory or tensor memory, or spilling
    registers, or one whose template's model (the case's `models`) decides otherwise than ptxas's register
    need (`register_need`, `decisions`): admitted where the need does not fit its budget or refused where
    it does, or another count of programs a multiprocessor (owner, 10-27: neither spills nor wastes
    occupancy; docs/design/kernel-templates.md), or whose scratch is not the bytes its planner sized it by
    (the model's `shared`, `resources.scratch_bytes`, where it states them)."""
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
            model, need = record.get("models", {}).get(found.kernel), kernel.get("need")
            # A tile its model refuses (the gate compiles those beside a derived one) spills by design: its
            # need alone is checked.
            admitted = model is None or decisions(model["registers"], model)[0]
            if admitted and (why := found.failure(record["limit"], record["tmem_limit"])) is not None:
                failed.append(f"{record['name']} {found.kernel}: {why}")
            if model is not None and model.get("shared") not in (None, found.shared):
                failed.append(f"{record['name']} {found.kernel}: its planner sized {model['shared']} bytes "
                              f"of scratch, the kernel asks for {found.shared}")
            if model is not None and decisions(model["registers"], model) != decisions(need, model):
                failed.append(f"{record['name']} {found.kernel}: the model's {model['registers']} registers "
                              f"decide {decisions(model['registers'], model)}, ptxas's need of {need} "
                              f"{decisions(need, model)}")
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
