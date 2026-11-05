"""A training step compiled ahead of time for GPUs this process does not have.

MaxText's `train_compile` compiles a run's step for a topology it is not running on, from that
topology's compile-only devices, saves the executable and prints what it will cost; this is Dew's
form of it. `compile_only_gpus` gives a CUDA generation's devices with none attached, which XLA's
CUDA backend compiles for on any host that has the CUDA plugin (the `kernel-compile` extra on a host
with no GPU): a `Trainer` whose mesh is built over them (`MeshSpec.build(devices)`) compiles its step
as on those GPUs, the same planning, kernels and collectives, and `compile_ahead` returns it with
XLA's own reading of its cost and of each device's memory, and serialized as JAX serializes an
executable (`jax.experimental.serialize_executable`), for a host of that topology to load and run
(`AheadOfTime.loaded`) or read before booking it.

    devices = compile_only_gpus("h100", 8)
    trainer = Trainer(objective, adamw(3e-4), mesh=MeshSpec(fsdp=8).build(devices))
    ahead = compile_ahead(trainer, batch)
    ahead.peak_bytes, ahead.flops.hardware
    ahead.save("step.dew-aot")

A configured run does both ends: `--trainer.compile-ahead.generation sm90 --trainer.compile-ahead.devices
8 --trainer.compile-ahead.path step.dew-aot` compiles its first step for eight H100s and exits
(`AheadTarget`), and `--trainer.ahead step.dew-aot` on a host of that topology runs it in place of
compiling its first step, where it is the very program the run compiles first (`AheadOfTime.taken`).

A run's plan is the fewest microbatches at the least recomputation that fit, which memory decides; which
plan steps fastest, a rung that recomputes more at fewer microbatches or the plan's own, the GPUs alone
tell. `--trainer.compile-ahead.frontier`, with each GPU's memory stated, lists past the plan each rung
above it that fits at fewer microbatches than every plan before it, at the fewest that fit there, and,
where the ladder's top ('full') is reached, its sibling that offloads the residuals 'minimal' saves
('minimal_offloaded') at the same split (`compile_ahead`'s `frontier`, `FitSearch.above`);
tools/benchmark_step.py times each on the GPUs, a case at its remat and head tile, a plan of s
microbatches as its microbatch's rows accumulated s times (`accumulation`), the time it reports a
microbatch's, s of them an update; and the run names the fastest by its model's remat, `--trainer.split`
and `LMObjective.head_tile`. No run times plans itself.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import pickle
import platform
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.experimental.topologies as topologies
import numpy as np
from jax.experimental import serialize_executable

from dew import communicators
from dew._records import JSON, record
from dew.batch import Batch
from dew.cache import uncached
from dew.nn.kernels.generation import (
    GpuTarget,
    assumed,
    built_target,
    capability,
    device_target,
    gpu_mesh,
    gpu_target,
    routes,
    tpu,
)
from dew.nn.kernels.options import owned_program
from dew.nn.kernels.templates import report
from dew.nn.sharding import Link
from dew.telemetry.instrumentation import StepFlops, step_flops
from dew.training.profiling import pgle_runs

if TYPE_CHECKING:
    from dew.checkpoints import Resume
    from dew.data.dataset import DataPartition, Loader
    from dew.training.trainer import Trainer

_log = logging.getLogger(__name__)

def target_config(target: GpuTarget, capacity: int | None = None) -> str:
    """The `gpu_device_info` XLA compiles for, from the GPU `target` a build states (`built_target`: its
    CUDA generation's, its multiprocessors its own where stated): its compute capability, the shared memory
    a block may opt into, its multiprocessors, clock and bandwidth; CUDA's per-capability limits for the
    rest. The fields past
    the capability and shared memory only steer XLA's own fusions, never a Triton kernel's code. From
    Hopper on, its capability states the architecture-specific features (sm_90a, sm_100a, sm_120a), which
    Mosaic GPU's PTX targets: left unspecified, an sm120's assembled for plain sm_120 and refused its own
    PTX ('.target sm_120a cannot be compiled for architecture sm_120')."""
    generation = target.generation
    version = capability(generation)
    shared, multiprocessors = target.shared_limit, target.multiprocessors
    clock, bandwidth = target.clock_hz, target.memory_bandwidth
    if version is None or shared is None or multiprocessors is None or clock is None or bandwidth is None:
        raise ValueError(f"no CUDA target states {generation!r}'s compile facts")
    major, minor = version
    return f"""gpu_device_info {{
  threads_per_block_limit: 1024
  threads_per_warp: 32
  shared_memory_per_block: 49152
  shared_memory_per_core: {shared + 1024}
  threads_per_core_limit: {target.architecture.resident_threads}
  core_count: {multiprocessors}
  fpus_per_core: {64 if (major, minor) == (8, 0) else 128}
  block_dim_limit_x: 2147483647
  block_dim_limit_y: 65535
  block_dim_limit_z: 65535
  memory_bandwidth: {int(bandwidth)}
  device_memory_size: {0 if capacity is None else capacity}
  clock_rate_ghz: {clock / 1e9}
  shared_memory_per_block_optin: {shared}
  cuda_compute_capability {{
    major: {major}
    minor: {minor}
    feature_extension: {"ACCELERATED_FEATURES" if major >= 9 else "NONE"}
  }}
  registers_per_core_limit: 65536
  registers_per_block_limit: 65536
}}
platform_name: "CUDA"
device_description_str: "{target.device_kind}"
"""


@functools.cache
def compile_only_tpus(topology: str) -> tuple:
    """The devices of the TPU `topology` ('v5e:4x4', as `jax.experimental.topologies` names one), none
    attached: libtpu compiles a program over them on any host that holds it, as MaxText's train_compile
    does, a mesh's collectives included, as for that topology's TPUs."""
    return tuple(topologies.get_topology_desc(topology, "tpu").devices)


def compile_only_gpus(generation: str, count: int = 1, capacity: int | None = None,
                      multiprocessors: int | None = None) -> tuple:
    """`count` compile-only devices of the CUDA `generation`, none attached: XLA's CUDA backend compiles a
    program over them, a mesh's collectives included, as for that many of the generation's GPUs, of the
    `multiprocessors` a build states where it states them (`built_target`). `capacity` states bytes per device
    to XLA's scheduler and rematerialization pass, not only Dew's admission check; absent it, XLA's target
    states zero rather than a hardware capacity."""
    stated = built_target(generation, multiprocessors).multiprocessors
    return _compile_only_gpus(generation, count, capacity, stated)


@functools.cache
def _compile_only_gpus(generation: str, count: int, capacity: int | None,
                       multiprocessors: int | None) -> tuple:
    """`compile_only_gpus`, one topology for each GPU however a caller states it (the generation's own
    multiprocessors as None or as their count)."""
    config = target_config(built_target(generation, multiprocessors), capacity)
    topology = topologies.get_topology_desc("dew-compile", "cuda", target_config=config,
                                            topology=f"1x1x{count}")
    return tuple(topology.devices[:count])


@dataclasses.dataclass(frozen=True)
class Executable:
    """A program compiled ahead, serialized, and the trees of its inputs and outputs, as
    `serialize_executable.serialize` gives them."""

    executable: bytes
    trees: tuple

    @staticmethod
    def of(compiled: jax.stages.Compiled) -> Executable:
        executable, in_tree, out_tree = serialize_executable.serialize(compiled)
        return Executable(executable, (in_tree, out_tree))

    def loaded(self, devices: Sequence) -> jax.stages.Compiled:
        """The executable, loaded onto `devices` (the ones it was compiled for, which its loader checks:
        `BuildTarget.held`) and admitted as a compiled one is (`dew.communicators.admit`)."""
        in_tree, out_tree = self.trees
        loaded = serialize_executable.deserialize_and_load(
            self.executable, in_tree, out_tree, backend=devices[0].client, execution_devices=devices)
        communicators.admit(loaded)
        return loaded


@dataclasses.dataclass(frozen=True)
class AheadProgram:
    """A program compiled ahead beside a step (`AheadOfTime.sampled`): its executable (`Executable`) and the
    program it was compiled from (`program_of`), which a run that takes it lowers too (`taken`)."""

    program: str
    executable: Executable

    @staticmethod
    def of(program: str, compiled: jax.stages.Compiled) -> AheadProgram:
        """`compiled`, the executable `program` (`program_of`) compiled to, as a program compiled ahead."""
        return AheadProgram(program, Executable.of(compiled))

    def taken(self, lowered: jax.stages.Lowered, devices: list) -> jax.stages.Compiled:
        """The executable, loaded onto `devices`, for a run whose program lowered as `lowered`: refused
        where that is another program than this one, and admitted as a compiled one is
        (`dew.communicators.admit`)."""
        if program_of(lowered) != self.program:
            raise ValueError("the sampled step compiled ahead is another program than the one this run "
                             "compiles: compile it ahead from this run's config, with its inspection")
        return self.executable.loaded(devices)


@dataclasses.dataclass(frozen=True)
class AheadOfTime:
    """A training step compiled for `devices` devices of `generation` (`compile_ahead`), compile-only
    GPUs of a CUDA generation or a TPU topology's of its chip ('v5e'): the executable,
    serialized, the program it was compiled from (`program`, `program_of`), and what it costs.
    `flops` are its matmuls' and convolutions' on one device, read off its optimized program as a
    run's are (`dew.telemetry.instrumentation.step_flops`, which
    counts the cuBLAS and cuDNN calls XLA's own cost analysis cannot see into), None where it runs a
    loop of unstated length; the byte counts are XLA's memory analysis of one device: the arguments
    it reads (the state and the batch's share), the outputs it writes, those that alias an argument
    (the donated state), its temporaries and its code."""

    generation: str
    devices: int
    mesh: dict[str, int]
    program: str
    """The step's program as it lowered (`program_of`), which a run that loads it lowers too (`taken`)."""
    executable: Executable
    flops: StepFlops | None
    argument_bytes: int
    output_bytes: int
    alias_bytes: int
    temp_bytes: int
    code_bytes: int
    links: dict[str, Link]
    """Each split axis's link the step decided its projections by (`Trainer.links`), whole: on
    compile-only GPUs none measured, so none spreads, unless the trainer stated one. A run handed the
    step traces with these in place of its own (`Trainer`'s `ahead`), which could spread a projection
    the step does not and lower another program, and its checkpoints record their bandwidths
    (`link_figures`)."""
    target: GpuTarget | None = None
    """The facts of the GPU its kernels' launches were derived for (`gpu_target` within `assumed`), which a
    run handed the step reproduces it under (`handed`, `derived_for`): another SKU of the generation, its
    multiprocessors other, derives other launches, and lowers another program. None for a step compiled
    ahead before steps recorded it, which no run reproduces."""
    plan: JSON = None
    """The plan the run's own fit search makes on those
    GPUs (`Trainer.plan`): every program it compiled, each with the footprint derived for it beside
    XLA's memory of it, and the plan it took."""
    planned_seconds: float | None = None
    """The seconds that search took, its derivations and every compile: the time a run spends planning
    before its first step on those GPUs."""
    capacity: int | None = None
    """The stated bytes per target GPU, or None where no memory capacity was supplied."""
    sampled: AheadProgram | None = None
    """The step that also summarizes what the run inspects (`Candidate.sampled`), compiled beside the step
    for a run that inspects, which a run handed the step takes in place of compiling it
    (`dew.training.planning.FitSearch.measured`); None for a run that inspects nothing."""
    built: BuildTarget | None = None
    """The target its executable was built for (`BuildTarget`), which a process loading it must be
    (`loaded`); None for a step compiled ahead before steps recorded it, which no process loads."""
    refusals: tuple = ()
    """The kernel launch configs its compiles found spilling and refused
    (`dew.nn.kernels.templates.report.refused_launches`), which a run handed the step refuses before it
    lowers (`FitSearch`), so its derivations step down as the compile's did and it lowers this program."""

    @property
    def peak_bytes(self) -> int:
        """The bytes one device holds at the step's peak, as the trainer's own fit check reads a
        step's memory (`dew.training.memory.step_headroom`): its arguments, resident, its outputs that
        alias none of them, and its temporaries."""
        return self.argument_bytes + self.output_bytes - self.alias_bytes + self.temp_bytes

    def over(self, stated: Mapping[str, Link]) -> dict[str, Link]:
        """The links a run handed this step traces with, for every program it compiles: the step's own
        (`links`), which the run's `stated` ones give way to, as it says once where they differ."""
        def figures(link: Link) -> tuple[float | None, float | None]:
            return link.bytes_per_second, link.flops_per_second

        differ = {axis: link for axis, link in stated.items() if axis in self.links
                  and figures(self.links[axis]) != figures(link)}
        if differ:
            _log.warning("this run states links %s, and the step compiled ahead traced with %s; the run "
                         "traces with the step's", differ, {axis: self.links[axis] for axis in differ})
        return self.links

    def save(self, path: str | Path) -> None:
        """Write the step to `path`, for `load` on a host of its topology."""
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> AheadOfTime:
        """The step `save` wrote to `path`."""
        step = _saved(path, AheadOfTime, "step")
        sampled = [] if step.sampled is None else [step.sampled.executable]
        _current(path, "step", [step.executable, *sampled])
        return step

    def loaded(self) -> jax.stages.Compiled:
        """The executable, loaded onto this process's devices, which must be `devices` GPUs or TPUs of
        `generation`: called with the state and batch the step was compiled for, it runs the step. It is
        admitted as a compiled one is (`dew.communicators.admit`)."""
        found = jax.devices()
        if self.built is None:
            raise ValueError("this step was compiled ahead before steps recorded the target they were built "
                             "for, which a process loading it must be; compile it ahead again")
        self.built.held(found)
        return self.executable.loaded(found)


    def derived_for(self, device: GpuTarget) -> GpuTarget:
        """The facts a run on a device of `device`'s facts derives its launches for while it reproduces
        this program (`handed`): the target it was compiled for (`target`), which the device runs where
        it is of its generation and, a GPU, of its multiprocessors (`BuildTarget`) and holds at least the
        shared memory a block of its launches takes, both known. Refused for any other, on which the
        program's launches are not known to run."""
        if self.target is None:
            raise ValueError("this step was compiled ahead before steps recorded the GPU their launches were "
                             "derived for, which a run reproduces it under; compile it ahead again")
        held = tpu(self.target) or (device.shared_limit is not None and self.target.shared_limit is not None
                                    and device.shared_limit >= self.target.shared_limit
                                    and device.multiprocessors == self.target.multiprocessors)
        if device.generation != self.target.generation or not held:
            raise ValueError(f"the step was compiled ahead for {self.target.device_kind} "
                             f"({self.target.generation}, {self.target.multiprocessors} multiprocessors, "
                             f"{self.target.shared_limit} bytes of shared memory a block), whose launches "
                             f"this {device.device_kind} ({device.generation}, {device.multiprocessors}, "
                             f"{device.shared_limit} bytes) does not run (`BuildTarget`); compile it ahead "
                             "for this GPU")
        return self.target

    def taken(self, lowered: jax.stages.Lowered) -> jax.stages.Compiled:
        """The executable, loaded (`loaded`), for a run whose step lowered as `lowered`: refused where
        that is another program than the one compiled ahead, a config, a data shape, a rung or a Dew
        that differs, which the executable would not run as the run's step."""
        if program_of(lowered) != self.program:
            raise ValueError("the step compiled ahead is another program than the one this run compiles "
                             "first: compile it ahead from this run's config and data "
                             "(--trainer.compile-ahead), with this Dew, and no plan recorded for another "
                             "rung")
        return self.loaded()


DUMP_FLAGS = ("--xla_dump", "--xla_gpu_dump")
"""XLA_FLAGS that decide whether and what XLA dumps, which `dumps` sets for a run alone."""
PHASE_FLAGS = (*DUMP_FLAGS, "--xla_gpu_load_autotune_results",
               "--xla_gpu_require_complete_aot_autotune_results", "--xla_gpu_per_fusion_autotune_cache_dir")
"""XLA_FLAGS of one phase's own (where a run dumps, what a build loads), which compute nothing a program
does."""


def computation_flags(flags: str | None = None) -> tuple[str, ...]:
    """The XLA flags, `XLA_FLAGS` by default, that decide what a program computes (Dew's rounding and
    division, dew/__init__.py, among them): every one but a phase's own (`PHASE_FLAGS`), in their order, as
    XLA reads a flag given twice by its last value."""
    given = os.environ.get("XLA_FLAGS", "") if flags is None else flags
    return tuple(setting for setting in given.split() if not setting.startswith(PHASE_FLAGS))


@dataclasses.dataclass(frozen=True)
class BuildTarget:
    """What programs compiled ahead (a step, `compile_ahead`, or named programs, `compile_programs`) were
    built for and with, which a process loading them must be (`held`), the one rule of every loader: the
    generation whose facts XLA compiled for and their launches derived for (`target_config`), its
    multiprocessors among them, which a launch's grid and XLA's fusions read, and how many devices; the
    JAX and jaxlib that compiled them, the XLA flags that decide what they compute (`computation_flags`),
    and the host architecture their host code was built for (a Mosaic kernel carries host code beside its
    cubin). A GPU of the generation (its compute capability) with as many multiprocessors runs them: an
    A100 of 40 GB as one of 80 (2026-11-02, GS-compile-programs3-a100 was refused on the kind's name
    alone); an H100 PCIe (114) does not run an SXM's (132): a balanced tail's grid is the target's
    multiprocessors times its programs (tail.py:62-66, gemm/mosaic_gpu.py:597), and the program starting a
    tile waits on later programs' partials (mosaic.py:1875-1888), so a grid past the GPU's residency can
    wait on a program that never starts. A GPU whose multiprocessors are shared (MPS limits, MIG) holds
    fewer than it states, and runs none of them. The CUDA plugin is not part of it: an H100 ran sm90
    programs a CUDA 12 plugin built, unchanged, under the CUDA 13 plugin of a node's own environment, bit
    for bit as the jitted programs (2026-11-02, GS-compileahead-pilot-h100)."""

    generation: str
    multiprocessors: int | None
    devices: int
    jax: str
    jaxlib: str
    flags: tuple[str, ...]
    host: str

    @staticmethod
    def built(target: GpuTarget, devices: int) -> BuildTarget:
        """The build for `devices` devices of the GPU `target` (its generation and multiprocessors), under
        this process's JAX, jaxlib, computing XLA flags and host: the one place a build's environment is
        read."""
        return BuildTarget(target.generation, target.multiprocessors, devices,
                           importlib.metadata.version("jax"), importlib.metadata.version("jaxlib"),
                           computation_flags(), platform.machine())

    @staticmethod
    def of(devices: Sequence) -> BuildTarget:
        """The target `devices` are, each by its own facts whatever `assumed` says (`device_target`), under
        this process's JAX, XLA flags and host; refused where they are not all of one generation and
        multiprocessors."""
        found = {(target.generation, target.multiprocessors) for target in map(device_target, devices)}
        if len(found) != 1:
            raise ValueError(f"these devices are of several targets, {sorted(found, key=str)}: programs "
                             "compiled ahead load onto devices of one")
        return BuildTarget.built(device_target(devices[0]), len(devices))

    def differences(self, other: BuildTarget) -> list[str]:
        """Each field `other` holds otherwise, said as this one's and other's."""
        named = [field.name for field in dataclasses.fields(self)]
        return [f"{name} {getattr(self, name)!r}, not {getattr(other, name)!r}" for name in named
                if getattr(self, name) != getattr(other, name)]

    def held(self, devices: Sequence) -> None:
        """Refused, naming what differs, where `devices` under this process's JAX, XLA flags and host are not
        this target."""
        if differences := self.differences(BuildTarget.of(devices)):
            raise ValueError(f"these programs were compiled for another target than this process's: "
                             f"{'; '.join(differences)}")


@dataclasses.dataclass(frozen=True)
class AheadPrograms:
    """Named programs compiled ahead for GPUs this process need not have (`compile_programs`): each program's
    executable (`AheadProgram`) and the target they were built for (`BuildTarget`). A process of that target
    loads them (`loaded`) and runs them with no tracing, lowering or compile
    (`dew.telemetry.compiles.executable_only`): a GPU node then spends its time on its device's work
    alone (ledger 2716)."""

    target: BuildTarget
    programs: dict[str, Executable]

    def save(self, path: str | Path) -> None:
        """Write the programs to `path`, for `load` in a process of their target."""
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> AheadPrograms:
        """The programs `save` wrote to `path`."""
        programs = _saved(path, AheadPrograms, "programs")
        _current(path, "programs", list(programs.programs.values()))
        return programs

    def loaded(self) -> dict[str, jax.stages.Compiled]:
        """Each program, by its name, loaded onto this process's first `BuildTarget.devices` devices, the ones
        it was compiled for: each called with the arguments it was compiled for runs it. Refused, naming what
        differs, where those devices, this JAX, its XLA flags or its host are not the target they were built
        for; each is admitted as a compiled program is (`dew.communicators.admit`)."""
        devices = jax.devices()[:self.target.devices]
        self.target.held(devices)
        return {name: program.loaded(devices) for name, program in self.programs.items()}


def _saved[Saved](path: str | Path, kind: type[Saved], what: str) -> Saved:
    """What a `save` of `kind` wrote to `path`, refused where it is not one (`what` it holds)."""
    loaded = pickle.loads(Path(path).read_bytes())
    if not isinstance(loaded, kind):
        raise ValueError(f"{path} holds no {what} compiled ahead of time")
    return loaded


def _current(path: str | Path, what: str, held: Sequence[object]) -> None:
    """Refuse the `what` saved at `path` where its executables `held` are not this Dew's one form
    (`Executable`): saved by an earlier Dew, they are compiled again."""
    if not all(isinstance(executable, Executable) for executable in held):
        raise ValueError(f"{path} holds {what} saved by an earlier Dew, whose executables this one does not "
                         "load: compile them ahead again")


def substituted(taken: Sequence) -> list[str]:
    """Each kernel a compile ahead's routes `taken` (`generation.routes`) lowered another in place of, as this
    host lowers no such kernel (`Route.ran`: FlashAttention without its wheel, cuDNN with no GPU attached),
    named once: the compiled program is not the one the target's GPU runs."""
    return sorted({f"{route.choice}: {route.kernel} ran as {route.ran} at {route.site}" for route in taken
                   if route.ran != route.kernel})


def compile_programs(programs: Mapping[str, tuple[Callable, Sequence[jax.ShapeDtypeStruct]]],
                     generation: str, multiprocessors: int | None = None) -> AheadPrograms:
    """`programs`, each a function and the shapes and dtypes of its arguments, by its name, compiled ahead for
    one GPU of the CUDA `generation`, of the `multiprocessors` it states (`built_target`), on any host holding
    the CUDA plugin (`compile_only_gpus`), its launches derived for that GPU's facts (`assumed`), and
    serialized: a CPU host builds what a GPU node then
    only loads and runs (`AheadPrograms.loaded`, ledger 2716). Arguments and results are placed whole on that
    GPU, whatever the function reads. A function takes and returns arrays (a backward is a program of its own
    over its residuals); one whose program holds arrays it closed over is refused, named, as no executable
    serializes them."""
    target = built_target(generation, multiprocessors)
    devices = compile_only_gpus(generation, multiprocessors=multiprocessors)
    mesh = jax.sharding.Mesh(np.asarray(devices), ("devices",))
    placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    built = {}
    for name, (function, arguments) in programs.items():
        shapes = tuple(jax.ShapeDtypeStruct(argument.shape, argument.dtype, sharding=placed)
                       for argument in arguments)
        # Through Dew's compile point, as a run compiles it: a launch whose kernel spills there is derived
        # again at its next config.
        program = owned_program(function, out_shardings=placed)
        with (assumed(target), jax.sharding.use_abstract_mesh(gpu_mesh(mesh, generation)), uncached(),
              routes() as taken):
            compiled = program.compiled(*shapes)
        if missing := substituted(taken):
            _log.warning("program %r compiled ahead for %s with kernels this host cannot lower: %s", name,
                         generation, "; ".join(missing))
        try:
            built[name] = Executable.of(compiled)
        except (NotImplementedError, ValueError) as refused:
            raise ValueError(f"program {name!r} cannot be compiled ahead (an array it closed over is a "
                             "constant of its program, which no executable serializes: pass it as an "
                             f"argument): {refused}") from refused
    return AheadPrograms(BuildTarget.built(target, 1), built)


def dumps(directory: Path, environment: Mapping[str, str]) -> dict[str, str]:
    """`environment` for a run whose every compile, in it or any process it starts from any directory, its
    compilers write under `directory`: XLA's (`--xla_dump_to`, unfiltered, whatever the environment said of
    XLA's dumping) and Mosaic GPU's own (`MOSAIC_GPU_DUMP_TO`, whose PTX-to-cubin compile writes
    nothing of XLA's). A run of programs compiled ahead leaves both empty (`compiled_in`). Refused where
    either cannot be written, as a compiler that cannot write its dump compiles on."""
    root = directory.resolve()
    for part in ("xla", "mosaic"):
        (root / part).mkdir(parents=True, exist_ok=True)
        if (root / part).is_symlink():
            raise ValueError(f"{root / part} is a link, whose writes compiled_in would not read")
        # A file written and removed, as a compiler's dump would be: a full or read-only filesystem refuses
        # it.
        with tempfile.NamedTemporaryFile(dir=root / part, prefix=".written-"):
            pass
    given = environment.get("XLA_FLAGS", "").split()
    flags = [setting for setting in given if not setting.startswith(DUMP_FLAGS)]
    kept = {name: value for name, value in environment.items() if not name.startswith("MOSAIC_GPU_DUMP")}
    dumping = [*flags, f"--xla_dump_to={root / 'xla'}"]
    return {**kept, "XLA_FLAGS": " ".join(dumping), "MOSAIC_GPU_DUMP_TO": str(root / "mosaic")}


def compiled_in(directory: Path) -> list[Path]:
    """What the compilers of a run given `dumps(directory, ...)` wrote: none where nothing compiled. Refused
    where `directory` is not one `dumps` made, or cannot be read, never read as empty."""
    root = directory.resolve()
    if linked := [part for part in ("xla", "mosaic") if (root / part).is_symlink()]:
        raise ValueError(f"{root}'s {', '.join(linked)} dump directory is a link, whose writes go unread")
    if missing := [part for part in ("xla", "mosaic") if not (root / part).is_dir()]:
        raise FileNotFoundError(f"{root} holds no {', '.join(missing)} dump directory: dumps made none here")

    def unreadable(error: OSError) -> None:
        raise error

    return sorted(Path(top) / name for top, _, names in os.walk(root, onerror=unreadable) for name in names)


@contextlib.contextmanager
def handed(ahead: AheadOfTime | None) -> Iterator[None]:
    """Within the block, a run reproduces the step compiled ahead (`ahead`) as its compile made it: the
    kernel launch configs refused are its compiles' alone (`AheadOfTime.refusals`), in a scope of their
    own (`report.refusals`), whatever this process or its cache refused of other programs, and each
    launch derives for the GPU it was compiled for (`AheadOfTime.derived_for`), whatever this GPU's own
    facts. Without one, the process's own refusals and facts."""
    if ahead is None:
        yield
        return
    with report.refusals(), assumed(ahead.derived_for(gpu_target())):
        report.refuse(ahead.refusals)
        yield


def program_of(lowered: jax.stages.Lowered) -> str:
    """A step's program as `lowered` holds it, its text's digest: two lowerings of one step agree."""
    return hashlib.sha256(lowered.as_text().encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class AheadTarget:
    """Where a configured run compiles its first step ahead and saves it, then exits, training nothing
    (`TrainerConfig.compile_ahead`): for `devices` GPUs of the CUDA `generation` (`compile_only_gpus`),
    or the `devices` TPUs of the TPU topology it names ('v5e:4x4', `compile_only_tpus`), which this host
    need not have, written to `path` (`AheadOfTime.save`). Where `capacity_gib` states each GPU's memory,
    the run's own plan there is made and reported with it (`compile_ahead`): the offline answer to what
    fits, a run with its remat set told of no rung but its own; a TPU states its own. The links it
    compiles with are the run's (`TrainerConfig.links`), which no compile-only device measures."""

    generation: str
    devices: int
    path: str
    capacity_gib: float | None = None
    frontier: bool = False
    multiprocessors: int | None = None
    """The multiprocessors of the GPU of `generation` it compiles for (an H100 PCIe's 114, an L40S's 142),
    where it is not the generation's own (`built_target`): what the step's launches derive for and a GPU
    loading it must have (`BuildTarget`). A TPU topology states its own."""
    """Whether the report lists each rung above the plan that fits at fewer microbatches than every plan
    before it (`compile_ahead`, `FitSearch.above`)."""

    @property
    def capacity(self) -> int | None:
        """The stated bytes per GPU, read both when building its topology and when planning."""
        return None if self.capacity_gib is None else int(self.capacity_gib * 2**30)

    def compile_only(self) -> tuple:
        """The compile-only devices the run's mesh is built over: the TPU topology's, which must hold
        `devices`, or `devices` GPUs of the generation at the stated capacity."""
        if ":" not in self.generation:
            return compile_only_gpus(self.generation, self.devices, self.capacity, self.multiprocessors)
        if self.multiprocessors is not None:
            raise ValueError(f"a TPU topology ({self.generation}) states its own cores; set no "
                             "multiprocessors")
        if self.capacity_gib is not None:
            raise ValueError(f"a TPU topology ({self.generation}) states its own memory; set no capacity")
        devices = compile_only_tpus(self.generation)
        if len(devices) != self.devices:
            raise ValueError(f"the TPU topology {self.generation} holds {len(devices)} devices, not "
                             f"{self.devices}")
        return devices

    def compiled[LossT, EffectsT](self, trainer: Trainer[LossT, EffectsT], loader: Loader,
                                  steps: int | None, examples: int | None = None,
                                  resume: Resume = "auto") -> CompiledAhead:
        """`trainer`, built over this target's devices (`compile_only`), its first step compiled ahead
        (`compile_ahead`) and saved, as what ends the run, over the batch `fit` would start it on
        (`first_step_batch`), at the rung the checkpoint a run resumes under `resume` left it. A run whose
        step reads a rollout's batches, which it makes as it trains, is refused."""
        if trainer.rollout is not None:
            raise ValueError("compile_ahead lowers the step over the loader's batches; this run's step reads "
                             "its rollout's, which it makes only as it trains")
        ahead = compile_ahead(trainer, first_step_batch(trainer, loader, steps, examples, resume),
                              self.capacity, frontier=self.frontier)
        ahead.save(self.path)
        return CompiledAhead(self.path, ahead)


def first_step_batch[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], loader: Loader, steps: int | None,
                                      examples: int | None = None, resume: Resume = "auto") -> Batch:
    """The batch `trainer`'s first step takes from `loader`, as `fit` would start the run: an input its
    objective names alone declared first (`Trainer._declare_input`), the loader read through its device
    steps and the run's clock (its `steps`, or the `examples` it is bounded to, `Trainer.reading`), the
    run checked (`Trainer.validate`), and the first batch of the stream of this process's share fit opens
    (`Trainer._stream_partition`, its ramp's stages beginning where updates do) from the position its
    checkpoint left a run resumed under `resume` at (`Checkpoints.position`), under the layout's rules,
    as the run's resume reads it (`dew.training.startup.resume_admitted`, row 3917). A run its checkpoint
    completes has no step to take, and is refused. What a step compiled ahead (`AheadTarget.compiled`)
    and an offline view of a run (`dew.config.RunConfig.viewed`) start from."""
    from flax import linen as nn

    from dew.data.dataset import DataPartition
    from dew.training._fit import first_batch

    trainer._declare_input(lambda: first_batch(loader, trainer._stream_partition()))
    trainer.reading(loader, examples=examples, steps=steps)
    trainer.validate(steps=steps)
    checkpoints = trainer.checkpoints
    resumed = None if checkpoints is None else checkpoints.resumed(resume)
    if resumed is not None and steps is not None and resumed >= steps:
        raise ValueError(f"the run is complete at step {resumed}, its last checkpoint written, so it has no "
                         "step to take")
    # The shares fit reads by, its layout placing the rows, the saved one's among them.
    with nn.logical_axis_rules(trainer.layout.axis_rules):
        position = None if checkpoints is None or resumed is None else checkpoints.position(
            resumed, DataPartition.of(trainer.device_mesh))
        return first_batch(loader, trainer._stream_partition(), position)


class CompiledAhead(SystemExit):
    """Raised by a configured run under `compile_ahead` once its first step is compiled and saved
    (`AheadTarget.compiled`). Compiling was the whole run, so uncaught it ends the program with status 0
    and no traceback, the step's cost logged; `ahead` is the step and `path` where it was saved."""

    def __init__(self, path: str, ahead: AheadOfTime):
        super().__init__(0)
        self.path, self.ahead = path, ahead
        _log.info("the step compiled ahead for %s %s GPUs holds %s bytes a device at its peak; saved to "
                  "%s", ahead.devices, ahead.generation, ahead.peak_bytes, path)
        if ahead.plan is not None:
            probes = record(ahead.plan, "plan")["probes"]
            assert isinstance(probes, list)
            _log.info("planned on them in %.1f s, %d programs compiled, the plan %s", ahead.planned_seconds,
                      len(probes), json.dumps(ahead.plan))


def step_batch[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch,
                                 partition: DataPartition | None = None) -> Batch:
    """`batch`, the rows a loader gives this process, its share `partition` of the step's (the mesh's,
    `DataPartition.of`, where None), as `fit` hands it to its step: its work counted
    (`Trainer._counted_batch`), each row marked with its place among the step's records
    (`dew.training.distributed.marked`), its host fields set apart (`split_fields`), and on the examples
    clock whether it closes its update window (`closes_window`). A batch of every row, read whole on
    each process, is `DataPartition()`'s."""
    from dew.data.dataset import DataPartition
    from dew.distributed.batches import split_fields
    from dew.training._fit import closes_window
    from dew.training.distributed import marked

    partition = DataPartition.of(trainer.device_mesh) if partition is None else partition
    device, _ = split_fields(marked(partition, trainer._counted_batch)(batch))
    return closes_window(device, trainer._examples)


def compile_ahead[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch,
                                   capacity: int | None = None, *, frontier: bool = False) -> AheadOfTime:
    """`trainer`'s first step over a batch of `batch`'s shapes, compiled for the compile-only devices its
    mesh is built over, GPUs of one CUDA generation (`compile_only_gpus`) or a TPU topology's
    (`compile_only_tpus`), and never run. The planner `Trainer.compile` uses
    (`Trainer._compiled_plan`) selects and compiles the step once; that executable is serialized, not compiled
    again. Its batch is what `fit` hands to its step (`step_batch`), its fresh state abstract, and
    tracing reads the target's kernel choices (`assumed`). The plan reports every compile and the
    time the search took, and with `frontier` each rung above it that fits at fewer microbatches than
    every plan before it (`FitSearch.above`), compiled and not saved: the plans a run's speed is chosen
    among (the module's docstring). `capacity` states bytes per GPU; absent it, memory fit is unknown rather
    than measured. The GPU it compiles for is the one the mesh's compile-only GPUs state, as XLA compiles
    over them: their generation and multiprocessors (`compile_only_gpus`' `multiprocessors`, `built_target`),
    which the step's launches derive for and it records. Profiling, tuning and rollout require execution and
    are refused."""
    if capacity is not None and capacity <= 0:
        raise ValueError("compile-ahead capacity must be positive bytes per GPU")
    if trainer.host_master or trainer.rollout is not None or trainer.lower_for is not None:
        raise ValueError("compile-ahead needs a device-resident step over its loader, with no lower_for")
    if trainer.tune_kernels or pgle_runs():
        raise ValueError("compile-ahead cannot run kernel tuning or AutoPGLE on absent GPUs")
    devices = list(trainer.device_mesh.devices.flat)
    generations = {gpu_target(device.device_kind).generation for device in devices}
    platforms = {device.platform for device in devices}
    if platforms not in ({"gpu"}, {"tpu"}) or len(generations) != 1:
        raise ValueError("compile_ahead compiles for the compile-only devices of one generation a trainer's "
                         "mesh is built over (MeshSpec.build(compile_only_gpus(generation, count)), or a TPU "
                         f"topology's, compile_only_tpus(topology)); this mesh holds "
                         f"{sorted({device.device_kind for device in devices})}")
    (generation,) = generations
    on_tpus = platforms == {"tpu"}
    # The GPU XLA compiles for, as its compile-only devices state it.
    stated = None if on_tpus else built_target(generation, devices[0].core_count)
    if capacity is not None and on_tpus:
        raise ValueError(f"a TPU topology states its own memory; compile it ahead for {generation} with no "
                         "capacity")
    if frontier and capacity is None and not on_tpus:
        raise ValueError("a frontier lists the rungs that fit the GPUs' memory, which compile-only GPUs "
                         "do not state; name it (--trainer.compile-ahead.capacity-gib, compile_ahead's "
                         "capacity)")
    if capacity is not None:
        mesh = trainer.device_mesh
        devices = compile_only_gpus(generation, len(devices), capacity,
                                    None if stated is None else stated.multiprocessors)
        trainer.mesh = trainer.device_mesh = trainer.state_mesh = jax.sharding.Mesh(
            np.asarray(devices).reshape(mesh.devices.shape), mesh.axis_names, axis_types=mesh.axis_types)
    began = time.perf_counter()
    start = trainer._started()
    if start.resume is not None and trainer.checkpoints is not None:
        # A resumed run's step compiles at the rung its checkpoint records, as its resume would.
        trainer._resume_rung(trainer.checkpoints.rung(start.resume))
    # A TPU topology's executable loads onto no client here: the persistent cache neither reads an entry
    # for it nor writes one (`dew.cache.uncached`).
    on = generation if stated is None else stated
    with assumed(on), uncached() if on_tpus else contextlib.nullcontext(), routes() as taken:
        target = gpu_target()
        planned = trainer._compiled_plan(start.abstract, step_batch(trainer, batch), None, capacity, frontier)
    if missing := substituted(taken):
        _log.warning("the step compiled ahead for %s holds kernels this host cannot lower: %s", generation,
                     "; ".join(missing))
    # The sampled step as the search compiled it, its program what it lowered then.
    sampled = None if planned.sampled is None else AheadProgram.of(*planned.sampled)
    seconds = time.perf_counter() - began
    lowered, compiled = planned.candidate.lowered, planned.executable
    memory = compiled.memory_analysis()
    if memory is None:
        raise ValueError(f"XLA states no memory of the step it compiled for {generation}")
    return AheadOfTime(
        generation, len(devices), dict(trainer.device_mesh.shape), program_of(lowered),
        Executable.of(compiled),
        flops=step_flops(compiled),
        argument_bytes=memory.argument_size_in_bytes, output_bytes=memory.output_size_in_bytes,
        alias_bytes=memory.alias_size_in_bytes, temp_bytes=memory.temp_size_in_bytes,
        code_bytes=memory.generated_code_size_in_bytes, plan=trainer.plan, planned_seconds=seconds,
        capacity=capacity, links=dict(trainer.links), target=target,
        built=BuildTarget.built(target, len(devices)),
        refusals=report.refused_launches(), sampled=sampled)
