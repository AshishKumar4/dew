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
import importlib.util
import json
import logging
import os
import pickle
import platform
import tempfile
import time
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.experimental.topologies as topologies
import numpy as np
from jax.experimental import serialize_executable

from dew import communicators
from dew._cache import cuda_libraries, plugin_builds, topology_digest, uncached
from dew._records import JSON, record
from dew.batch import Batch
from dew.nn.kernels import options
from dew.nn.kernels.generation import (
    GpuTarget,
    assumed,
    built_target,
    capability,
    device_target,
    gpu_target,
    tpu,
)
from dew.nn.kernels.options import OuterProgram, collecting, owned_program, program_digest, shapes_of, taking
from dew.nn.kernels.selection import Unlowerable
from dew.nn.kernels.templates import report
from dew.nn.sharding import Link
from dew.telemetry.instrumentation import StepFlops, compiled_memory, step_flops
from dew.training import identity
from dew.training.profiling import pgle_runs

if TYPE_CHECKING:
    from dew.checkpoints import Resume
    from dew.data.compile import Handed
    from dew.data.dataset import DataPartition, Dataset, Loader
    from dew.training._fit import FitArguments, FitPlan, Start
    from dew.training.planning import Planned
    from dew.training.trainer import Trainer

_log = logging.getLogger(__name__)

def target_config(target: GpuTarget, capacity: int | None = None) -> str:
    """The `gpu_device_info` XLA compiles for, from the GPU `target` a build states (`built_target`: its
    CUDA generation's, its multiprocessors its own where stated): its compute capability, the shared memory
    a block may opt into, its multiprocessors, clock and bandwidth, its multiprocessor's and a block's
    limits (`Multiprocessor`) and its fp32 lanes (`Architecture.tensor_rates`); CUDA's grid limits for the
    rest. The fields past the capability and shared memory only steer XLA's own fusions, never a Triton
    kernel's code. From Hopper on, its capability states the architecture-specific features (sm_90a,
    sm_100a, sm_120a), which Mosaic GPU's PTX targets: left unspecified, an sm120's assembled for plain
    sm_120 and refused its own PTX ('.target sm_120a cannot be compiled for architecture sm_120')."""
    generation = target.generation
    version = capability(generation)
    shared, multiprocessors = target.shared_limit, target.multiprocessors
    clock, bandwidth = target.clock_hz, target.memory_bandwidth
    # Its fp32 lanes, two FLOPs a clock each (a fused multiply-add), as its architecture states them.
    fp32 = target.tensor_rates.get("fp32")
    if (version is None or shared is None or multiprocessors is None or clock is None or bandwidth is None
            or fp32 is None):
        raise ValueError(f"no CUDA target states {generation!r}'s compile facts")
    major, minor = version
    multiprocessor = target.multiprocessor
    return f"""gpu_device_info {{
  threads_per_block_limit: {multiprocessor.block_threads}
  threads_per_warp: {multiprocessor.warp_threads}
  shared_memory_per_block: {multiprocessor.block_shared}
  shared_memory_per_core: {shared + multiprocessor.reserved_shared}
  threads_per_core_limit: {multiprocessor.resident_threads}
  core_count: {multiprocessors}
  fpus_per_core: {fp32 // 2}
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
  registers_per_core_limit: {multiprocessor.registers}
  registers_per_block_limit: {multiprocessor.block_registers}
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


_HOSTS_GPUS: set = set()
"""The compile-only GPUs of hosts' own topologies (`host_gpus`): a step compiled over them is that host's
GPUs' program (`compile_ahead`'s `portable` where its mesh holds others)."""


def host_gpus(generation: str, topology: str | Path, count: int) -> tuple:
    """`count` compile-only devices of the CUDA `generation`'s GPUs a host serialized its own topology of
    (`jax.experimental.topologies.get_attached_topology("cuda").serialize()`, written to `topology`), none
    attached: XLA compiles over them as over that host's GPUs, the SKU's memory, clocks and multiprocessors
    its own, so a program compiled here keys the persistent cache as that host's compile of it does (its
    backend's platform version and topology fingerprint are the host's) and the host's run reads it. A
    canonical target (`compile_only_gpus`) keys another entry: the 80 GB A100 it names against a 40 GB
    host's."""
    # The CUDA plugin, which a serialized topology deserializes through, initialized as the generation's
    # compile-only GPUs initialize it: with no client and no GPU.
    compile_only_gpus(generation, 1)
    devices = topologies.TopologyDescription.deserialize(Path(topology).read_bytes()).devices
    _HOSTS_GPUS.update(devices)
    if len(devices) < count:
        raise ValueError(f"the topology at {topology} holds {len(devices)} GPUs, not {count}")
    held = gpu_target(devices[0].device_kind).generation
    if held != gpu_target(generation).generation:
        raise ValueError(f"the topology at {topology} holds {devices[0].device_kind} GPUs ({held}), not "
                         f"{generation}")
    return tuple(devices[:count])


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
        `BuildTarget.held`) and admitted as a compiled one is (`dew.communicators.admit`), the installed
        Dew kernel wheels' FFI targets registered first (`_wheel_targets_registered`)."""
        _wheel_targets_registered()
        in_tree, out_tree = self.trees
        loaded = serialize_executable.deserialize_and_load(
            self.executable, in_tree, out_tree, backend=devices[0].client, execution_devices=devices)
        communicators.admit(loaded)
        return loaded


def _wheel_targets_registered() -> None:
    """Each installed Dew kernel wheel's FFI targets registered with jax: a wheel registers its own at the
    first call that traces one, and a process that only loads a program compiled ahead traces none, so its
    first run of a call to one found no handler (dew-cutlass's grouped GEMM, dew-flash's attention)."""
    if importlib.util.find_spec("dew_cutlass") is not None:
        from dew.nn.kernels import grouped_matmul

        grouped_matmul.cutlass_registered()
    if importlib.util.find_spec("dew_flash_attn") is not None:
        # The wheel's own registration of its targets (`dew_flash_attn.TARGETS`), which its calls run.
        importlib.import_module("dew_flash_attn")._registered()


@dataclasses.dataclass(frozen=True)
class AheadProgram:
    """A program compiled ahead beside a step (`AheadOfTime.sampled`): its executable (`Executable`) and the
    program it was compiled from (`program_digest`), which a run that takes it lowers too (`taken`)."""

    program: str
    executable: Executable

    @staticmethod
    def of(program: str, compiled: jax.stages.Compiled) -> AheadProgram:
        """`compiled`, the executable `program` (`program_digest`) compiled to, as a program compiled
        ahead."""
        return AheadProgram(program, Executable.of(compiled))

    def taken(self, lowered: jax.stages.Lowered, devices: list) -> jax.stages.Compiled:
        """The executable, loaded onto `devices`, for a run whose program lowered as `lowered`: refused
        where that is another program than this one, and admitted as a compiled one is
        (`dew.communicators.admit`)."""
        if program_digest(lowered) != self.program:
            raise ValueError("the sampled step compiled ahead is another program than the one this run "
                             "compiles: compile it ahead from this run's config, with its inspection")
        return self.executable.loaded(devices)


@dataclasses.dataclass(frozen=True)
class AheadOfTime:
    """A training step compiled for `devices` devices of `generation` (`compile_ahead`), compile-only
    GPUs of a CUDA generation or a TPU topology's of its chip ('v5e'): the executable,
    serialized, the program it was compiled from (`program`, `program_digest`), and what it costs.
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
    """The step's program as it lowered (`program_digest`), which a run that loads it lowers too (`taken`)."""
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
    starts: dict[str, Executable] = dataclasses.field(default_factory=dict)
    """The programs a run builds its fresh state with before its first step and that step's own copies of
    its weights (`dew.training.distributed.start_lowered`, the narrowed copies), compiled for the target
    beside the step, by their programs' digests (`program_digest`): a run handed the step runs each of
    these as compiled here (`taking`), so its devices compile nothing before and at its first step, its
    placement included (Main, 11-07: an attention step compiled ahead still compiled its placement on the
    GPU, 5-7 s)."""

    def taking(self) -> contextlib.AbstractContextManager[None]:
        """The block a run handed this step builds its start and its first step's copies in: each program
        of `starts` that it lowers runs as compiled here, loaded onto this process's devices as the step
        is (`loaded`), where it would have compiled it (`dew.nn.kernels.options.taking`)."""
        def load(executable: Executable) -> Callable[[], jax.stages.Compiled]:
            def loaded() -> jax.stages.Compiled:
                found = jax.devices()
                if self.built is not None:
                    self.built.held(found)
                return executable.loaded(found)
            return loaded

        return taking({digest: load(executable) for digest, executable in self.starts.items()})

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
        if "starts" not in vars(step):
            raise ValueError(f"{path} holds a step compiled ahead before steps carried their start's "
                             "programs (`starts`), whose run would compile them on its devices; compile it "
                             "ahead again")
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
        if program_digest(lowered) != self.program:
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
    fewer than it states, and runs none of them. For a CUDA GPU, the plugin that compiled them and the cuBLAS
    and cuDNN the compile loaded are part of it (`plugin`, `libraries`): JAX's own persistent-cache key
    holds the plugin's build, and a compile writes its GEMM algorithms for that cuBLAS and serializes its
    cuDNN graphs, so a program carries both. One run of sm90
    programs a CUDA 12 plugin built, under a CUDA 13 plugin, matched the jitted programs bit for bit
    (2026-11-02, GS-compileahead-pilot-h100), which one program's run does not extend to every graph a
    library serializes. The topology a compile was placed on (`topology`) names the compile, and the
    persistent cache and a shared bundle key by it, but no loader holds it: a program compiled over a
    generation's canonical GPUs runs on a host's own of that generation and multiprocessors."""

    generation: str
    multiprocessors: int | None
    devices: int
    jax: str
    jaxlib: str
    flags: tuple[str, ...]
    host: str
    memory: int | None = None
    """The bytes a device holds that the build's plan held each call to, where the build stated them (its
    capacity: `built_target`'s `memory`): a device holding fewer (its allocator's limit) runs past what the
    plan kept and is refused; a build stating none planned for the least of its target's parts."""
    cpu: str | None = None
    """A CPU target's own processor (`host_cpu`): XLA compiles a CPU's program to the instruction sets of
    the host that compiles it, so its executable runs on a host of the same ones. None for a GPU's or a
    TPU's, whose program is its devices' (a TPU's target is its kind's generation and its devices), and for
    a CPU host that states no processor, which programs kept by their calls refuse (`cpu_stated`)."""
    plugin: str = ""
    """The build of each CUDA PJRT plugin installed where they were compiled (`dew._cache.plugin_builds`),
    empty for a target of no CUDA GPU, a CPU's or a TPU's, whose programs no CUDA plugin compiles."""
    libraries: tuple[tuple[str, int | None], ...] = ()
    """The cuBLAS and cuDNN versions the compile loaded (`dew._cache.cuda_libraries`), None for one it
    could not load, which then wrote none of its state; empty for a target of no CUDA GPU."""
    topology: str | None = None
    """The digest of the GPU topology the programs were compiled over (`dew._cache.topology_digest`): a
    host's own GPUs as a file serialized them (`host_gpus`), None for a generation's canonical ones
    (`compile_only_gpus`), for the GPUs attached and for any other platform's. The attached GPUs'
    serialization differs call to call (two calls on one A100 differ from byte 207 on, probe UB-TOPO,
    2026-11-11), so its digest names nothing. It names the compile, which no loader holds
    (`differences`)."""

    @staticmethod
    def built(target: GpuTarget, devices: int, memory: int | None = None, *,
              topology: bytes | None = None) -> BuildTarget:
        """The build for `devices` devices of the GPU `target` (its generation and multiprocessors), each
        holding the `memory` its plan states, over the serialized `topology` of a host's own GPUs where it
        was given one, under this process's JAX, jaxlib, computing XLA flags and host, and for a CUDA GPU
        (one of a compute capability) its CUDA plugin and libraries: the one place a build's environment is
        read."""
        cuda = capability(target.generation) is not None
        return BuildTarget(target.generation, target.multiprocessors, devices,
                           importlib.metadata.version("jax"), importlib.metadata.version("jaxlib"),
                           computation_flags(), platform.machine(), memory,
                           plugin=plugin_builds() if cuda else "", libraries=cuda_libraries() if cuda else (),
                           topology=None if topology is None else topology_digest(topology))

    @staticmethod
    def of(devices: Sequence) -> BuildTarget:
        """The target `devices` are, each by its own facts whatever `assumed` says (`device_target`), the
        least allocator limit among them its memory where they state one, a CPU's with its host's
        processor (`host_cpu`), under this process's JAX, XLA flags and host; refused where they are not
        all of one generation and multiprocessors."""
        found = {(target.generation, target.multiprocessors) for target in map(device_target, devices)}
        if len(found) != 1:
            raise ValueError(f"these devices are of several targets, {sorted(found, key=str)}: programs "
                             "compiled ahead load onto devices of one")
        limits = [stats["bytes_limit"] for device in devices
                  if "bytes_limit" in (stats := device.memory_stats() or {})]
        own = device_target(devices[0])
        # No topology: the attached GPUs' serialization differs call to call (`topology`).
        built = BuildTarget.built(own, len(devices), min(limits, default=None))
        # A CPU's program is its host's code (`cpu`).
        return dataclasses.replace(built, cpu=host_cpu()) if devices[0].platform == "cpu" else built

    def differences(self, other: BuildTarget) -> list[str]:
        """Each field `other` holds otherwise, said as this one's and other's: memory where other's is
        known and below this one's (a device holding more than a build planned for runs it), a library
        where this build loaded it and other's version is another (a build that loaded none wrote none of
        its state), and never the topology, which names the compile and binds no loader."""
        named = [field.name for field in dataclasses.fields(self)
                 if field.name not in ("memory", "topology", "libraries")]
        found = [f"{name} {getattr(self, name)!r}, not {getattr(other, name)!r}" for name in named
                 if getattr(self, name) != getattr(other, name)]
        theirs = dict(other.libraries)
        found += [f"libraries {name} {version}, not {theirs.get(name)}" for name, version in self.libraries
                  if version is not None and theirs.get(name) != version]
        if self.memory is not None and other.memory is not None and other.memory < self.memory:
            found.append(f"memory {self.memory} bytes, not {other.memory}")
        return found

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
    process: identity.Process | None = None
    """The process that traced `kept` (`dew.training.identity.Process`), None for a file of named
    programs alone."""
    kept: dict[str, KeptProgram] = dataclasses.field(default_factory=dict)
    """The programs a process compiled at Dew's compile point, each by the digest of its call's identity
    (`KeptProgram`, `keeping`): a process that is their target and their `process` runs each for the same
    call, tracing nothing (`kept`)."""

    def save(self, path: str | Path) -> None:
        """Write the programs to `path`, for `load` in a process of their target."""
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> AheadPrograms:
        """The programs `save` wrote to `path`: one saved before a file held programs by their calls
        (`kept`) holds its named programs alone."""
        programs = _saved(path, AheadPrograms, "programs")
        kept = vars(programs).get("kept", {})
        _current(path, "programs", [*programs.programs.values(),
                                    *(each.program.executable for each in kept.values())])
        return programs if "kept" in vars(programs) else AheadPrograms(programs.target, programs.programs)

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


def host_cpu() -> str | None:
    """This host's processor as a CPU target states it (`BuildTarget.cpu`): its model and the instruction
    sets it reports (Linux's `/proc/cpuinfo`), which XLA's CPU compile reads to choose a program's code.
    None on a system that states them nowhere Python reads: no CPU program is kept for such a host or
    handed on one (`cpu_stated`)."""
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    first = cpuinfo.read_text().split("\n\n", 1)[0]
    named = sorted(line for line in first.splitlines()
                   if line.split(":", 1)[0].strip() in ("model name", "flags", "Features", "CPU part"))
    return hashlib.sha256("\n".join(named).encode()).hexdigest()[:16]


def cpu_stated(target: BuildTarget, devices: Sequence) -> None:
    """Refuse programs kept by their calls for a CPU whose host states no processor (`host_cpu`): two
    such hosts of other instruction sets would read as one target."""
    if devices[0].platform == "cpu" and target.cpu is None:
        raise ValueError("this host states no processor features (no /proc/cpuinfo), so a CPU program "
                         "cannot be kept for it or handed on it")


@dataclasses.dataclass(frozen=True)
class KeptProgram:
    """A program a process compiled at Dew's compile point, kept for a later process (`keeping`): the
    identity of the call it compiled for (`dew.training.identity.call_identity`), which a call is handed
    it by, its executable with its lowering's digest (`AheadProgram`), which `kept`'s `verify` proves it
    by, and the ids of the devices it runs on, as its executable states them."""

    call: JSON
    program: AheadProgram
    devices: tuple[int, ...]
    """The ids of the devices it runs on, among the devices of its file's target (`jax.devices()`)."""

    def loaded(self) -> jax.stages.Compiled:
        """The executable, loaded onto this process's devices of the ids it was compiled for; refused,
        naming them, where the process holds none of an id."""
        held = {device.id: device for device in jax.devices()}
        if missing := [number for number in self.devices if number not in held]:
            raise ValueError(f"a kept program runs on devices {list(self.devices)}, and this process holds "
                             f"none of {missing}")
        return self.program.executable.loaded([held[number] for number in self.devices])


class Keeping:
    """The programs Dew's compile point compiles while `keeping` is open, each kept by its call's identity,
    for `save`. `refused` names each it could not keep and why (a call with no stated identity, an
    executable XLA does not serialize): those compile again where the file is handed."""

    def __init__(self) -> None:
        self.kept: dict[str, KeptProgram] = {}
        self.refused: dict[str, str] = {}

    def heard(self, program: OuterProgram, key: Hashable, args: tuple, kwargs: dict,
              lowered: jax.stages.Lowered, executable: jax.stages.Compiled) -> None:
        """Keep `executable`, which `program`'s call under `key` compiled from `lowered`
        (`dew.nn.kernels.options.keeping`)."""
        del args, kwargs
        named = f"{identity.called(program)} ({len(self.refused)})"
        try:
            call = identity.call_identity(program, key)
            serialized = Executable.of(executable)
        except identity.Unestablished as refusal:
            self.refused[named] = str(refusal)
            return
        except (NotImplementedError, ValueError) as refusal:
            # As `compile_programs` reads it: an array a program closed over is a constant of its
            # program, which no executable serializes.
            self.refused[named] = f"its executable does not serialize: {refusal}"
            return
        runtime = executable.runtime_executable()
        assert runtime is not None, "a program compiled on this process's devices is loaded"
        ran = runtime.local_devices()
        if not set(ran) <= set(jax.devices()):
            # The file's target is the process's default backend's (`save`), which this program is none of.
            self.refused[named] = f"it runs on {sorted({device.platform for device in ran})} devices, " \
                                  "outside the process's default backend"
            return
        devices = tuple(device.id for device in ran)
        program_kept = AheadProgram(program_digest(lowered), serialized)
        self.kept[identity.digest(call)] = KeptProgram(call, program_kept, devices)

    def save(self, path: str | Path) -> AheadPrograms:
        """Write the programs kept so far to `path` with this process's target and record, for `kept` in
        a process of both; returns what it wrote. Each program it could not keep is logged with its
        reason (`refused`)."""
        for named, reason in self.refused.items():
            _log.warning("the program %s is not kept, and compiles where the file is handed: %s", named,
                         reason)
        target = BuildTarget.of(jax.devices())
        cpu_stated(target, jax.devices())
        programs = AheadPrograms(target, {}, identity.Process.here(), dict(self.kept))
        programs.save(path)
        return programs


@contextlib.contextmanager
def keeping() -> Iterator[Keeping]:
    """Within the block, every program this thread compiles at Dew's compile point
    (`dew.nn.kernels.options.OuterProgram`: a scoring task's passes, a server's steps, an evaluation's
    programs) is kept by the identity of the call it compiled for, and the `Keeping` it yields saves them:

        with keeping() as programs:
            server = ScoringServer.from_task(task, variables)
        programs.save("scoring.dew-aot")

    A later process of the same target, code and settings hands them back (`kept`) and traces none of
    them. A program compiled before the block, or handed or taken within it, is not among them. The
    block compiles with JAX's persistent cache off (`dew._cache.uncached`): an executable XLA loaded
    from that cache serializes without its kernels (on a CPU it then loads and fails at its first call,
    "Function ... not found"; plain JAX, 2026-11-09), so each kept one is compiled here."""
    kept = Keeping()
    with uncached(), options.keeping(kept.heard):
        yield kept


@contextlib.contextmanager
def kept(path: str | Path, *, verify: bool = False) -> Iterator[AheadPrograms]:
    """Within the block, a call at Dew's compile point whose identity
    (`dew.training.identity.call_identity`) one of the programs kept at `path` was compiled for
    (`keeping`) runs that executable, loaded onto the devices it was compiled for, and traces, lowers and
    compiles nothing; a call the file holds no program for compiles as without it.

    Refused before any call, naming what differs, where this process is not the one the file was kept in:
    its devices, JAX, jaxlib, XLA flags and host (`BuildTarget.held`), or its Python, Dew's source, a
    distribution's version, another module's source, a setting of JAX's, an environment variable of Dew's
    or a fact its traces read (`dew.training.identity.Process.held`). A stale executable is a wrong
    result with nothing to show it, so what an identity cannot state is not kept (`Keeping.refused`); what
    it covers is `dew.training.identity`'s to say. Code that changes what a trace computes by assigning
    module state at run time is outside the identity, as it is outside `jax.jit`'s cache.

    With `verify`, each call handed a program also lowers its own and holds the two to one digest
    (`program_digest`), at the cost of the trace the file saves: the proof, for this file and this
    process, that the identity covered what the trace read (`NotTheKeptProgram` where it did not)."""
    programs = AheadPrograms.load(path)
    if programs.process is None:
        raise ValueError(f"{path} holds programs compiled ahead by name, none kept by its call (`keeping`)")
    cpu_stated(programs.target, jax.devices())
    programs.target.held(jax.devices())
    programs.process.held()

    def give(program: OuterProgram, key: Hashable, args: tuple, kwargs: dict) -> jax.stages.Compiled | None:
        try:
            found = programs.kept.get(identity.digest(identity.call_identity(program, key)))
        except identity.Unestablished as refusal:
            _log.info("a call with no stated identity is handed no kept program, and compiles: %s", refusal)
            return None
        if found is None:
            return None
        if verify and program_digest(program.lower(*args, **kwargs)) != found.program.program:
            raise NotTheKeptProgram(
                f"{identity.called(program)} lowers here to another program than the one kept at "
                f"{path} for the same identity: something its trace reads is outside the identity "
                "(`dew.training.identity`); declare what it reads (`traced_fact`) and keep the programs "
                "again")
        return found.loaded()

    with options.handing(give):
        yield programs


class NotTheKeptProgram(ValueError):
    """A call whose identity a kept program was compiled for lowers to another program (`kept`'s
    `verify`): its trace read something its identity does not state."""


def compile_programs(programs: Mapping[str, tuple[Callable, Sequence[jax.ShapeDtypeStruct]]],
                     generation: str, multiprocessors: int | None = None) -> AheadPrograms:
    """`programs`, each a function and the shapes and dtypes of its arguments, by its name, compiled ahead for
    one GPU of the CUDA `generation`, of the `multiprocessors` it states (`built_target`), on any host holding
    the CUDA plugin (`compile_only_gpus`), its launches derived for that GPU's facts (`assumed`), and
    serialized: a CPU host builds what a GPU node then
    only loads and runs (`AheadPrograms.loaded`, ledger 2716). Arguments and results are placed whole on that
    GPU, whatever the function reads. A function takes and returns arrays (a backward is a program of its own
    over its residuals); one whose program holds arrays it closed over is refused, named, as no executable
    serializes them, as is one holding a kernel this host cannot lower for any GPU of the generation
    (`selection.Unlowerable`: cuDNN's, whose plan holds to its GPU's model). Each is lowered in turn and all
    compile at once (`report.compiled_at_once`)."""
    target = built_target(generation, multiprocessors)
    devices = compile_only_gpus(generation, multiprocessors=multiprocessors)
    mesh = jax.sharding.Mesh(np.asarray(devices), ("devices",))
    placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    lowerings = []
    for name, (function, arguments) in programs.items():
        shapes = tuple(jax.ShapeDtypeStruct(argument.shape, argument.dtype, sharding=placed)
                       for argument in arguments)

        def lowered(function=function, shapes=shapes, name=name) -> jax.stages.Lowered:
            """The program lowered for the target, afresh at each call (a relowering's, past a spill)."""
            try:
                with assumed(target, portable=True), jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
                    return owned_program(function, out_shardings=placed).lower(*shapes)
            except Unlowerable as refused:
                raise refused.refusing(name) from refused
        lowerings.append((lowered(), None, lowered))
    # Through Dew's compile point, as a run compiles them, at once (`report.compiled_at_once`): a launch
    # whose kernel spills there is derived again at its next config.
    executables = report.compiled_at_once(lowerings)
    built = {}
    for name, compiled in zip(programs, executables, strict=True):
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
    """Whether the report lists each rung above the plan that fits at fewer microbatches than every plan
    before it (`compile_ahead`, `FitSearch.above`)."""
    multiprocessors: int | None = None
    """The multiprocessors of the GPU of `generation` it compiles for (an H100 PCIe's 114, an L40S's 142),
    where it is not the generation's own (`built_target`): what the step's launches derive for and a GPU
    loading it must have (`BuildTarget`). A TPU topology states its own."""
    topology: str | None = None
    """A host's own serialized GPU topology (`host_gpus`), which states the GPUs it compiles for whole:
    their memory and multiprocessors, so it sets neither `capacity_gib` nor `multiprocessors`. A dew-ops
    prepare is handed its node's ($DEW_NODE_TOPOLOGY), so its compiles key the cache the node's run reads."""

    @property
    def capacity(self) -> int | None:
        """The stated bytes per GPU, read both when building its topology and when planning."""
        return None if self.capacity_gib is None else int(self.capacity_gib * 2**30)

    def compile_only(self) -> tuple:
        """The compile-only devices the run's mesh is built over: the host's own GPUs where it names their
        `topology` (`host_gpus`), the TPU topology's, which must hold `devices`, or `devices` GPUs of the
        generation at the stated capacity."""
        if self.topology is not None:
            if self.capacity_gib is not None or self.multiprocessors is not None:
                raise ValueError("a host's topology states its GPUs' memory and multiprocessors; set no "
                                 "capacity or multiprocessors with it")
            return host_gpus(self.generation, self.topology, self.devices)
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
                                  resume: Resume = "auto", plan: FitPlan | None = None) -> CompiledAhead:
        """`trainer`, built over this target's devices (`compile_only`), its first step compiled ahead
        (`compile_ahead`) and saved, as what ends the run, over the batch `fit` would start it on
        (`first_step_batch`), at the rung the checkpoint a run resumes under `resume` left it, with its
        evaluations' programs where `plan` is what its `fit` is asked for (`Trainer._fit_plan`). A run whose
        step reads a rollout's batches, which it makes as it trains, is refused."""
        if trainer.rollout is not None:
            raise ValueError("compile_ahead lowers the step over the loader's batches; this run's step reads "
                             "its rollout's, which it makes only as it trains")
        ahead = _compiled_ahead(trainer, first_step_batch(trainer, loader, steps, examples, resume),
                                self.capacity, frontier=self.frontier, plan=plan,
                                topology=None if self.topology is None else Path(self.topology).read_bytes())
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
                                   capacity: int | None = None, *, frontier: bool = False,
                                   fit: FitArguments | None = None, topology: bytes | None = None
                                   ) -> AheadOfTime:
    """`trainer`'s first step over a batch of `batch`'s shapes, compiled for the compile-only devices its
    mesh is built over, GPUs of one CUDA generation (`compile_only_gpus`) or a TPU topology's
    (`compile_only_tpus`), and never run. The planner `Trainer.compile` uses
    (`Trainer._compiled_plan`) selects and compiles the step once; that executable is serialized, not compiled
    again. Its batch is what `fit` hands to its step (`step_batch`), its fresh state abstract, and
    tracing reads the target's kernel choices (`assumed`). The plan reports every compile and the
    time the search took, and with `frontier` each rung above it that fits at fewer microbatches than
    every plan before it (`FitSearch.above`), compiled and not saved: the plans a run's speed is chosen
    among (the module's docstring). With `fit`, what the run's `Trainer.fit` over `batch` is asked for, its
    evaluations' programs compile with its start's (`started`). `capacity` states bytes per GPU;
    absent it, memory fit is unknown rather than measured. The GPU it compiles for is the one the mesh's
    compile-only GPUs state, as XLA compiles over them: their generation and multiprocessors
    (`compile_only_gpus`' `multiprocessors`, `built_target`), which the step's launches derive for and it
    records. `topology` is the serialized topology of a host's own GPUs the mesh's devices were read from
    (`host_gpus`), which the saved build names (`BuildTarget.topology`); None for a generation's canonical
    GPUs. Profiling, tuning and rollout require execution and are refused."""
    return _compiled_ahead(trainer, batch, capacity, frontier=frontier,
                           plan=None if fit is None else fit_plan(trainer, batch, fit), topology=topology)


def fit_plan[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], dataset: Dataset | Loader | Handed,
                              fit: FitArguments) -> FitPlan:
    """What `trainer.fit(dataset, **fit)` runs (`Trainer._fit_plan`, which states `fit`'s defaults)."""
    return trainer._fit_plan(dataset, **fit)


def _compiled_ahead[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch, capacity: int | None,
                                     *, frontier: bool, plan: FitPlan | None, topology: bytes | None = None
                                     ) -> AheadOfTime:
    """`compile_ahead`'s step and programs, the evaluations' with `plan`, what the run's `fit` is asked for
    (`Trainer._fit_plan`), over its own data or a configured run's loader (`AheadTarget.compiled`)."""
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
    stated = None if on_tpus else built_target(generation, devices[0].core_count, capacity)
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
    on = generation if stated is None else stated
    # A canonical target's devices stand for any GPU of the generation; a host's own topology's, for its own.
    portable = not on_tpus and not set(trainer.device_mesh.devices.flat) <= _HOSTS_GPUS
    # The first step's pass count, the phase its batch was read in (`Trainer`'s `phase_passes`).
    trainer.phase_passes.at(batch)
    try:
        with assumed(on, portable=portable), trainer.phase_passes.scoped(trainer):
            target = gpu_target()
            planned = trainer._compiled_plan(start.abstract, step_batch(trainer, batch), None, capacity,
                                             frontier)
            starts = started(trainer, start, planned, plan)
    except Unlowerable as refused:
        raise refused.refusing("step") from refused
    # The sampled step as the search compiled it, its program what it lowered then.
    sampled = None if planned.sampled is None else AheadProgram.of(*planned.sampled)
    seconds = time.perf_counter() - began
    lowered, compiled = planned.candidate.lowered, planned.executable
    memory = compiled_memory(compiled)
    if memory is None:
        raise ValueError(f"XLA states no memory of the step it compiled for {generation}")
    return AheadOfTime(
        generation, len(devices), dict(trainer.device_mesh.shape), program_digest(lowered),
        Executable.of(compiled),
        flops=step_flops(compiled),
        argument_bytes=memory.argument, output_bytes=memory.output, alias_bytes=memory.alias,
        temp_bytes=memory.temp, code_bytes=memory.generated_code, plan=trainer.plan, planned_seconds=seconds,
        capacity=capacity, links=dict(trainer.links), target=target,
        built=BuildTarget.built(target, len(devices), capacity, topology=topology),
        refusals=report.refused_launches(), sampled=sampled, starts=starts)


def started[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], start: Start, planned: Planned,
                             plan: FitPlan | None = None) -> dict[str, Executable]:
    """The programs a run of `trainer` compiles besides its step, compiled for the target over their
    arguments' shapes (`AheadOfTime.starts`): its fresh state's build and placement (`start_lowered`), for
    a run that starts afresh, the first step's narrowed copies of its weights where the plan reads some
    (`narrowed`), and what its steps run on their results, each called as the run calls it, over the
    step's results' shapes (`collecting`): an accumulating window's first allocation (`Trainer._window_of`),
    a split step's whole (`whole_step`), its counters (`bookkeep`), a sampled step's latest sample
    (`latest`), its log's copies (`copied`) and its logged tokens' sum (`added_tokens`), and with the plan
    of its `fit` (`plan`), what its evaluations' warming pass compiles before they run
    (`Trainer._warmed_ahead`; row 4752). Each by its program's digest, through the kernel templates'
    reports as every compile is."""
    from dew.training.distributed import start_lowered
    from dew.training.narrow import narrowed

    lowerings: list[Callable[[], jax.stages.Lowered]] = []
    if start.resume is None and not trainer.host_master:
        key = jax.ShapeDtypeStruct(start.key.shape, start.key.dtype)
        lowerings.append(lambda: start_lowered(trainer._staged_start, start.initializer, key, start.abstract,
                                               start.shardings, start.through, trainer.device_mesh))
    candidate = planned.candidate
    if candidate.copies:
        # The variables as the run's first step holds them, placed as its start placed them.
        variables = jax.tree.map(lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype,
                                                                             sharding=sharding),
                                 start.abstract.variables, start.shardings.variables)
        lowerings.append(lambda: owned_program(lambda variables: narrowed(variables, candidate.copies),
                                               *trainer.objective.towers(),
                                               out_shardings=candidate.shardings.compute).lower(variables))

    def kept(lower: Callable[[], jax.stages.Lowered]) -> tuple[str, Executable]:
        # A program a refused launch's config relowers is keyed by the program it compiled, as the run's
        # compile point lowers it past the refusals the step carries (`OuterProgram.compiled`).
        lowered = [lower()]

        def relower() -> jax.stages.Lowered:
            lowered.append(lower())
            return lowered[-1]
        executable = report.compiled(lowered[0], relower=relower)
        return program_digest(lowered[-1]), Executable.of(executable)
    collected = _results_programs(trainer, start, planned, plan)
    return {**dict(map(kept, lowerings)), **{digest: Executable.of(executable)
                                             for digest, executable in collected.items()}}


def _results_programs[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], start: Start, planned: Planned,
                                       plan: FitPlan | None) -> dict[str, jax.stages.Compiled]:
    """The programs a run's steps run on their results and its state, and with its `fit`'s `plan` its
    evaluations' warming pass over its first state, compiled where the run's own code calls them over
    shapes (`collecting`), by digest: `started`'s last part."""
    from dew.distributed.placement import placed_shapes
    from dew.state import apart, together
    from dew.training.interval import added_tokens, book_shapes, booked, bookkeep, copied, latest
    from dew.training.planning import whole_step

    candidate, split = planned.candidate, planned.split
    placed = placed_shapes(start.abstract, start.shardings)
    window = split * trainer.accumulation
    frozen = apart(placed)[1]
    # A sampled step's results hold its summaries (`Committed.inspected`), which its programs read too.
    steps = [planned.executable] + ([] if planned.sampled is None else [planned.sampled.executable])
    with collecting() as collected:
        if trainer._builds_window(placed, candidate, window):
            trainer._window_of(dataclasses.replace(placed, compute=None), candidate, window)
        for step in steps:
            moving, loss, aux, finite, accepted, committed = shapes_of(step.out_info)
            if split > 1:
                loss, aux, finite, accepted, committed = whole_step(
                    [(loss, aux, finite, accepted, committed)] * split)
            book = book_shapes(loss)
            counted = bookkeep(*booked(book, book, loss, finite, committed, aux or {}))
            if committed is not None and committed.inspected is not None:
                latest((committed.count, committed.inspected), (committed.count, committed.inspected))
            added_tokens(counted.tokens, counted.tokens)
            copied(together(moving, frozen))
        if plan is not None:
            trainer._warmed_ahead(plan, placed, start.shardings)
    return collected
