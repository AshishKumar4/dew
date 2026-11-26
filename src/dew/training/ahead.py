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

    devices = compile_only_gpus("h100", 8, 80 * 2**30)
    trainer = Trainer(objective, adamw(3e-4), mesh=MeshSpec(fsdp=8).build(devices))
    ahead = compile_ahead(trainer, batch)
    ahead.peak_bytes, ahead.flops.hardware
    ahead.save("step.dew-aot")

A configured run does both ends: `dew lower run.json target:ahead --target.generation sm90 --target.devices 8
--target.path step.dew-aot` compiles its first step for eight H100s and exits (`AheadTarget`,
`RunConfig.lower`), and `--trainer.ahead step.dew-aot` on a host of that topology runs it in place of
compiling its first step, where it is the very program the run compiles first (`AheadOfTime.taken`).

A run's plan is the fewest microbatches that fit at the first rung its search admits (the remat rungs at
its own head first, a tight cold start's second compile: docs/design/training.md), which memory decides; which
plan steps fastest, a rung that recomputes more at fewer microbatches or the plan's own, the GPUs alone
tell. `--target.frontier`, with each GPU's memory stated, lists past the plan each rung
above it that fits at fewer microbatches than every plan before it, at the fewest that fit there, and,
where the ladder's top ('full') is reached, its sibling that offloads the residuals 'minimal' saves
('minimal_offloaded') at the same split (`compile_ahead`'s `frontier`, `FitSearch.above`), and below the
plan each lighter rung the run's remat-first search passed over that fits, at the coarsest split it fits at
(the plan's `lighter`, `FitSearch.below`: a head's tile, XLA's defaults or no copies at less recomputation);
tools/benchmark_step.py times each on the GPUs, a case at its remat and head tile, a plan of s
microbatches as its microbatch's rows accumulated s times (`accumulation`), the time it reports a
microbatch's, s of them an update; and the run names the fastest by its model's remat, `--trainer.split`
and `LMObjective.head_tile`. No run times plans itself.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import copyreg
import dataclasses
import functools
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import pickle
import platform
import tempfile
import time
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import jax.experimental.topologies as topologies
import numpy as np
from jax._src import cache_key
from jax.experimental import serialize_executable
from jaxlib import xla_client

from dew import communicators
from dew._cache import carried, cuda_libraries, plugin_builds, topology_digest, uncached
from dew._devices import DEW_SHARE, STATED_GPUS, StatedGpu, device_memory
from dew._records import JSON, boolean, record
from dew._stated import Unestablished, traceback_frames
from dew.batch import Batch, Variables
from dew.nn.kernels import options
from dew.nn.kernels.generation import (
    GpuTarget,
    assumed,
    built_target,
    capability,
    compile_only_gpus,
    device_target,
    gpu_target,
    positive_memory,
    tpu,
)
from dew.nn.kernels.options import (
    OuterProgram,
    attempting,
    collecting,
    owned_program,
    program_digest,
    program_name,
    shapes_of,
    stated_digest,
    taking,
)
from dew.nn.kernels.selection import Unlowerable
from dew.nn.kernels.templates import assembled, report
from dew.nn.sharding import Link
from dew.telemetry.instrumentation import CompiledMemory, StepFlops, compiled_memory, step_flops
from dew.training.profiling import pgle_runs
from dew.training.targets import AheadTarget

if TYPE_CHECKING:
    from jaxlib._jax import Traceback

    from dew.checkpoints import Resume
    from dew.data.dataset import DataPartition, Loader
    from dew.data.handed import Handed
    from dew.data.pipeline import Dataset
    from dew.state import TrainState
    from dew.training._fit import FitArguments, FitPlan, Start
    from dew.training.compiler import HostStartup
    from dew.training.planning import Planned
    from dew.training.trainer import Trainer

_log = logging.getLogger(__name__)

UNTUNED = ("this step was compiled ahead into a file for %s GPUs without their tuned choices, XLA's "
           "untuned program, whose products ran up to 48 times slower on a GH200 (UB-4792-AHEAD-TIME): a run "
           "handed it compiles its step on its GPUs as a run without a file does, and keeps their tuned "
           "choices beside its compilation cache. To train from a file, compile it ahead over the GPU host's "
           "own topology (--target.topology $DEW_NODE_TOPOLOGY) with that host's compilation "
           "cache, once a run there has tuned the step")
"""What a step file compiled without its GPU kind's tuned choices (`AheadOfTime.tuned`) is said as, where it
is saved and where a run is handed it (`dew.training.trainer.Trainer`)."""

AT_RUN = ("this step was compiled ahead into a file for %s GPUs without them, leaving a convolution's "
          "algorithm to run time (dew.nn.kernels.templates.report.DEVICELESS_CONVOLUTION_PICKS): a run "
          "handed it compiles its step on its GPUs as a run without a file does")
"""What a step file leaving a choice to run time (`AheadOfTime.at_run`) is said as, where a run takes it."""

@functools.cache
def compile_only_tpus(topology: str) -> tuple:
    """The devices of the TPU `topology` ('v5e:4x4', as `jax.experimental.topologies` names one), none
    attached: libtpu compiles a program over them on any host that holds it, as MaxText's train_compile
    does, a mesh's collectives included, as for that topology's TPUs."""
    return tuple(topologies.get_topology_desc(topology, "tpu").devices)


@contextlib.contextmanager
def stand_in_gpus(devices: Sequence, memory: int, share: float = DEW_SHARE) -> Iterator[tuple]:
    """`devices`, a host's own CPU devices (`jax.devices("cpu")`), each stated to hold a GPU's `memory`, its
    total bytes as the GPU states them, at `share`, the pool's share in the run they stand in for (Dew's
    own, `DEW_SHARE`, where none is given, as `compile_only_gpus` and `host_gpus` take it, never this
    process's), while the block runs: every
    layout check and admission on them reads the allocator's limit derived from them
    (`dew._devices.allocator_limit`, `StatedGpu`), as `compile_only_gpus`' do, where JAX's CUDA plugin,
    which those need, does not install. `Trainer.validate` checks a run's layout on them against the GPUs'
    limits; a program compiles for the CPU."""
    positive_memory(memory)
    devices = tuple(devices)
    STATED_GPUS.update(dict.fromkeys(devices, StatedGpu(memory, share)))
    try:
        yield devices
    finally:
        for device in devices:
            del STATED_GPUS[device]


_HOSTS_GPUS: set = set()
"""The compile-only GPUs of hosts' own topologies (`host_gpus`): a step compiled over them is that host's
GPUs' program (`compile_ahead`'s `portable` where its mesh holds others)."""


def host_gpus(generation: str, topology: str | Path, count: int, share: float = DEW_SHARE) -> tuple:
    """`count` compile-only devices of the CUDA `generation`'s GPUs a host serialized its own topology of
    (`jax.experimental.topologies.get_attached_topology("cuda").serialize()`, written to `topology`), none
    attached: XLA compiles over them as over that host's GPUs, the SKU's memory, clocks and multiprocessors
    its own, so a program compiled here keys the persistent cache as that host's compile of it does (its
    backend's platform version and topology fingerprint are the host's) and the host's run reads it. A
    canonical target (`compile_only_gpus`) keys another entry: the least memory of the generation's parts
    it states (an A100's 40 GB, `GpuTarget.memory`) against an 80 GB host's. Each is admitted at `share`,
    the pool's share in the host's run (`StatedGpu`)."""
    # The CUDA plugin, which a serialized topology deserializes through, initialized as the generation's
    # compile-only GPUs initialize it: with no client and no GPU.
    compile_only_gpus(generation, 1)
    serialized = Path(topology).read_bytes()
    devices = topologies.TopologyDescription.deserialize(serialized).devices
    _HOSTS_GPUS.update(devices)
    memory = topology_memory(serialized)
    if memory is not None:
        STATED_GPUS.update(dict.fromkeys(devices, StatedGpu(memory, share)))
    if len(devices) < count:
        raise ValueError(f"the topology at {topology} holds {len(devices)} GPUs, not {count}")
    held = gpu_target(devices[0].device_kind).generation
    if held != gpu_target(generation).generation:
        raise ValueError(f"the topology at {topology} holds {devices[0].device_kind} GPUs ({held}), not "
                         f"{generation}")
    return tuple(devices[:count])


def topology_memory(serialized: bytes) -> int | None:
    """The memory each GPU of a serialized GPU topology states, its total bytes, as XLA reads them for its
    target (`device_memory_size`): `PjRtTopologyDescriptionProto.platform_specific_topology` (9,
    xla/pjrt/proto/topology_description.proto), an `Any` whose value (2) is a `GpuTopologyProto`, its
    `gpu_target_config` (8, xla/service/gpu_topology.proto), a `GpuTargetConfigProto`, its `gpu_device_info`
    (1), a `GpuDeviceInfoProto`, its `device_memory_size` (14, xla/stream_executor/device_description.proto),
    read through Dew's one reader of XLA's protos (`proto_fields`); jaxlib's topology hands no accessor of
    it. None where the topology states none."""
    message = serialized
    for field in (9, 2, 8, 1):
        found = assembled.proto_fields(message).get(field, [])
        if len(found) != 1 or not isinstance(found[0], bytes):
            return None
        message = found[0]
    stated = assembled.proto_fields(message).get(14, [])
    return stated[0] if len(stated) == 1 and isinstance(stated[0], int) and stated[0] > 0 else None


@dataclasses.dataclass(frozen=True)
class HeldTraceback:
    """A traceback a program's trees hold, as its file holds it: its frames as text. A checkified program's
    error tree holds each check's (`jax.experimental.checkify`'s `FailedCheckError.traceback_info`), which
    no pickle holds as JAX holds it; its check's message is its own, so a loaded program's failed check
    says what the kept one's would."""

    frames: tuple[str, ...]

    def __str__(self) -> str:
        return "\n".join(self.frames)

    def as_python_traceback(self) -> None:
        """No Python traceback, which a failed check's error is raised with (checkify's `check_impl`): its
        frames are text, which no live frame stands behind; the error says its check's message alone."""


def _held_traceback(traceback: Traceback) -> tuple[type[HeldTraceback], tuple[tuple[str, ...]]]:
    """A traceback of JAX's reduced to its frames (`HeldTraceback`, `dew._stated.traceback_frames`), as a
    program file holds it."""
    return HeldTraceback, (tuple(traceback_frames(traceback)),)


def filed(value: Executable | AheadPrograms | AheadOfTime) -> bytes:
    """`value` as a program file holds it: pickled, a traceback of JAX's as its frames (`HeldTraceback`)."""
    held = io.BytesIO()
    pickler = pickle.Pickler(held, protocol=pickle.HIGHEST_PROTOCOL)
    pickler.dispatch_table = {**copyreg.dispatch_table, xla_client.Traceback: _held_traceback}
    pickler.dump(value)
    return held.getvalue()


@dataclasses.dataclass(frozen=True)
class Executable:
    """A program compiled ahead, serialized, and the trees of its inputs and outputs, as
    `serialize_executable.serialize` gives them, and its memory as XLA analysed it where it was compiled
    (`compiled_memory`), so a run's need is read before any device loads it (the GPU tier's workers sized
    by their parts' programs); None where its backend states none, and for one filed before it was
    recorded."""

    executable: bytes
    trees: tuple
    memory: CompiledMemory | None = None

    @staticmethod
    def of(compiled: jax.stages.Compiled) -> Executable:
        """`compiled` serialized with its trees and its memory, as a file holds them (`filed`); refused
        (ValueError) where a tree holds what no file holds, so a file that would hold it is never written
        short."""
        executable, in_tree, out_tree = serialize_executable.serialize(compiled)
        held = Executable(executable, (in_tree, out_tree), compiled_memory(compiled))
        try:
            filed(held)
        except (TypeError, pickle.PicklingError) as refusal:
            raise ValueError(f"its trees hold what no file holds: {refusal}") from refusal
        return held

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
    """Registers every installed Dew kernel wheel's FFI targets with jax, through the callable each names in
    the `dew.kernel_wheels` entry-point group (dew-cutlass's grouped GEMM, dew-flash-attn's attention). A
    process that loads a program compiled ahead traces no call that would register them."""
    for wheel in importlib.metadata.entry_points(group="dew.kernel_wheels"):
        wheel.load()()


class NotTheStepCompiledAhead(ValueError):
    """A run handed a step compiled ahead (`AheadOfTime`) lowers its step, or its sampled step, to another
    program than the one compiled: another config, data shape, rung or Dew, which the executable would not
    run as the run's step (`AheadOfTime.taken`)."""


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
        if stated_digest(lowered) != self.program:
            raise NotTheStepCompiledAhead(
                "the sampled step compiled ahead is another program than the one this run compiles: compile "
                "it ahead from this run's config, with its inspection")
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
    live_bytes: int | None = None
    """The most one device's buffers hold at once over the step, as XLA's heap simulation states it
    (`CompiledMemory.live`): where a value held too long shows, apart from the room the temporaries'
    packing takes (`temp_bytes`, the one allocation the fit check places). None where the backend states
    none."""
    target: GpuTarget | None = None
    """The facts of the GPU its kernels' launches were derived for (`gpu_target` within `assumed`), which a
    run handed the step reproduces it under (`handed`, `derived_for`): another SKU of the generation, its
    multiprocessors other, derives other launches, and lowers another program. None for a step compiled
    ahead before steps recorded it, which no run reproduces."""
    plan: JSON = None
    """The plan the run's own fit search makes on those
    GPUs (`Trainer.plan`): every program it compiled, each with XLA's memory of it, and the plan it
    took."""
    planned_seconds: float | None = None
    """The seconds that search took, its derivations and every compile: the time a run spends planning
    before its first step on those GPUs."""
    planned_parts: Mapping[str, float] = dataclasses.field(default_factory=dict)
    """The seconds of that search's work by part, summed over its threads (a compile at once runs several):
    the step's JAX traces (`traced`), the algebra's passes over the traced programs (`algebra`,
    `report.passes`), the step's lowerings (`lowered`) and XLA's compiles (`compiled`; the planner's reading,
    `planning.spending`), and a fresh state's start, traced, lowered and compiled beside them (`start`,
    `distributed.START_PROGRAM`'s, row 6740)."""
    capacity: int | None = None
    """The allocator's limit each target GPU holds the plan to, derived from the memory stated for it
    (`dew._devices.allocator_limit`); None for a TPU's, whose topology states its own."""
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
    at_run: bool = False
    """Whether any executable it holds, its step's, its sampled step's or its starts', leaves a
    choice to run time its device's own compile makes (`report.left_at_run`: a convolution compiled for
    compile-only GPUs): a run handed it compiles its step and starts on its devices in their place
    (`runnable`), whatever it tunes."""
    starts: dict[str, Executable] = dataclasses.field(default_factory=dict)
    """The programs a run builds its fresh state with before its first step and that step's own copies of
    its weights (`dew.training.distributed.start_compiled`, the narrowed copies), compiled for the target
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
                found = jax.devices() if self.built is None else self.built.own_devices()
                self.held(found)
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

    @property
    def tuned(self) -> bool:
        """Whether its compile took the GPU kind's tuned choices, as the device's own compile makes them, as
        its plan records it (`autotuned`, `report.autotuned`): a step compiled on compile-only GPUs outside
        a prepare is XLA's untuned program there, which its save says, and a run handed it compiles its
        step on its devices in its place (`runnable`); a TPU's is the device's own."""
        return boolean(record(self.plan, "plan")["autotuned"], "autotuned")

    def runnable(self, devices: Sequence) -> AheadOfTime | None:
        """The step a run on `devices` handed it runs: this one where it is tuned, or where the run's own
        compile would take no tuned choice either (`report.autotuned`: XLA_FLAGS setting the autotuner's
        level 0, as a repeatable lane's do), its products then the run's own; None where its XLA products
        are untuned (`tuned`) and the run's would be tuned, which the run then compiles on its devices as
        without a file, said once (`UNTUNED`). A step of another target is refused first, tuned or not
        (`held`): no run takes it."""
        self.held(devices)
        if report.compiled_as_here(devices, tuned=self.tuned, at_run=self.at_run):
            return self
        _log.warning(UNTUNED if not self.at_run else AT_RUN, self.generation)
        return None

    def save(self, path: str | Path) -> None:
        """Write the step to `path`, for `load` on a host of its topology."""
        if not self.tuned or self.at_run:
            _log.warning(AT_RUN if self.at_run else UNTUNED, self.generation)
        Path(path).write_bytes(filed(self))

    @staticmethod
    def load(path: str | Path) -> AheadOfTime:
        """The step `save` wrote to `path`."""
        step = _saved(path, AheadOfTime, "step")
        if "autotuned" not in record(step.plan, "plan"):
            raise ValueError(f"{path} holds a step compiled ahead before its plan said whether XLA tuned it "
                             "(`autotuned`); compile it ahead again")
        sampled = [] if step.sampled is None else [step.sampled.executable]
        _current(path, "step", [step.executable, *sampled])
        return step

    def loaded(self) -> jax.stages.Compiled:
        """The executable, loaded onto this process's devices, which must be `devices` GPUs or TPUs of
        `generation`: called with the state and batch the step was compiled for, it runs the step. It is
        admitted as a compiled one is (`dew.communicators.admit`). Refused where it would not run as its
        devices' own compile of it (`report.compiled_as_here`: untuned where they tune, or a choice left to
        run time, `at_run`): a load direct or through `taken` admits it as a run handed it does (`runnable`;
        rows 8957, 8986)."""
        found = jax.devices() if self.built is None else self.built.own_devices()
        self.held(found)
        if not report.compiled_as_here(found, tuned=self.tuned, at_run=self.at_run):
            raise ValueError((AT_RUN if self.at_run else UNTUNED) % self.generation)
        return self.executable.loaded(found)

    def held(self, devices: Sequence) -> None:
        """Refused, naming what differs, where `devices` are not the target the step was built for
        (`BuildTarget.held`), or the step recorded none."""
        if self.built is None:
            raise ValueError("this step was compiled ahead before steps recorded the target they were built "
                             "for, which a process loading it must be; compile it ahead again")
        self.built.held(devices)


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
        if stated_digest(lowered) != self.program:
            raise NotTheStepCompiledAhead(
                "the step compiled ahead is another program than the one this run compiles first: compile it "
                "ahead from this run's config and data (dew lower, target:ahead), with this Dew, and no plan "
                "recorded for another rung")
        return self.loaded()


DUMP_FLAGS = ("--xla_dump", "--xla_gpu_dump")
"""XLA_FLAGS that decide whether and what XLA dumps, which `dumps` sets for a run alone."""
PHASE_FLAGS = (*DUMP_FLAGS, "--xla_gpu_load_autotune_results",
               "--xla_gpu_require_complete_aot_autotune_results", "--xla_gpu_per_fusion_autotune_cache_dir")
"""XLA_FLAGS of one phase's own (where a run dumps, what a build loads), which compute nothing a program
does."""
HOST_FLAGS = ("--xla_force_host_platform_device_count",)
"""XLA_FLAGS of a process's own host, which size its CPU client and compute nothing a program compiled in it
does, its own devices compared apart (`BuildTarget`): JAX's compilation cache leaves them out of its key
too (jax/_src/cache_key.py, `xla_flags_to_exclude_from_cache_key`). A GPU tier part's prepare sees no GPU
and its run the GPUs it holds, each its host devices (tests/lane_environment.py)."""


def computation_flags(flags: str | None = None) -> tuple[str, ...]:
    """The XLA flags, `XLA_FLAGS` by default, that decide what a program computes (Dew's rounding and
    division, dew/__init__.py, among them): every one but a phase's own (`PHASE_FLAGS`) and its host's
    (`HOST_FLAGS`), in their order, as XLA reads a flag given twice by its last value."""
    given = os.environ.get("XLA_FLAGS", "") if flags is None else flags
    return tuple(setting for setting in given.split() if not setting.startswith((*PHASE_FLAGS, *HOST_FLAGS)))


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
    """The allocator's limit a device must hold to admit every program built (`programs_need`), each as
    XLA analysed its memory where it compiled it, or, a target's own (`of`), the limit its devices hold: a
    device whose limit is lower than a build's need is refused, naming both. Not the room the plan was
    allowed (`AheadOfTime.capacity`): a step built for a whole GPU runs on a smaller one that admits it
    (the GPU tier's part, its pool sized by its tests' peaks, refused a tiny step built for 80 GiB, rows
    8094 and 8371). A build stating none planned for the least of its target's parts."""
    cpu: str | None = None
    """A CPU target's own processor (`host_cpu`): XLA compiles a CPU's program to the instruction sets of
    the host that compiles it, so its executable runs on a host of the same ones. None for a GPU's or a
    TPU's, whose program is its devices' (a TPU's target is its kind's generation and its devices), and for
    a CPU host that states no processor, for which no program is kept (`cpu_stated`)."""
    plugin: str = ""
    """The build of each CUDA PJRT plugin installed where they were compiled (`dew._cache.plugin_builds`),
    empty for a target of no CUDA GPU, a CPU's or a TPU's, whose programs no CUDA plugin compiles."""
    libraries: tuple[tuple[str, int | None], ...] = ()
    """The cuBLAS and cuDNN versions the compile loaded (`dew._cache.cuda_libraries`), None for one it
    could not load, which then wrote none of its state; empty for a target of no CUDA GPU."""
    chips: str | None = None
    """A TPU target's physical layout (`tpu_chips`): each device's chip coordinates and core, which a TPU's
    executable is partitioned over, so a build for a 2x2 slice is refused on a 1x4 one of as many chips of
    its kind (row 9064). None for a GPU's or a CPU's, and for a build that recorded none."""
    topology: str | None = None
    """The digest of the GPU topology the programs were compiled over (`dew._cache.topology_digest`): a
    host's own GPUs as a file serialized them (`host_gpus`), None for a generation's canonical ones
    (`compile_only_gpus`), for the GPUs attached and for any other platform's. The attached GPUs'
    serialization differs call to call (two calls on one A100 differ from byte 207 on, probe UB-TOPO,
    2026-11-11), so its digest names nothing. It names the compile, which no loader holds
    (`differences`)."""
    device_kind: str | None = None
    """The kind of GPU the programs were compiled for where a build states one (a host's own topology's, a
    GH200's against the canonical H100's), whose own facts its launches derived from (`built_target`): a
    device of another kind is refused (`differences`). None for a generation's canonical GPUs, which stand
    for any of its parts, and for any other platform's (row 13252)."""

    @staticmethod
    def built(target: GpuTarget, devices: int, memory: int | None = None, *,
              topology: bytes | None = None, chips: str | None = None,
              device_kind: str | None = None) -> BuildTarget:
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
                           topology=None if topology is None else topology_digest(topology), chips=chips,
                           device_kind=device_kind)

    @staticmethod
    def of(devices: Sequence) -> BuildTarget:
        """The target `devices` are, each by its own facts whatever `assumed` says (`device_target`), the
        least allocator limit among them its memory where they state one (`device_memory`: an attached
        device's own, a compile-only GPU's derived from its stated memory), a CPU's with its host's
        processor (`host_cpu`), under this process's JAX, XLA flags and host; refused where they are not
        all of one generation and multiprocessors."""
        found = {(target.generation, target.multiprocessors) for target in map(device_target, devices)}
        if len(found) != 1:
            raise ValueError(f"these devices are of several targets, {sorted(found, key=str)}: programs "
                             "compiled ahead load onto devices of one")
        limits = [limit for device in devices
                  if (limit := device_memory(device).get("bytes_limit")) is not None]
        own = device_target(devices[0])
        # No topology: the attached GPUs' serialization differs call to call (`topology`).
        kind = devices[0].device_kind if devices[0].platform == "gpu" else None
        built = BuildTarget.built(own, len(devices), min(limits, default=None), chips=tpu_chips(devices),
                                  device_kind=kind)
        # A CPU's program is its host's code (`cpu`).
        return dataclasses.replace(built, cpu=host_cpu()) if devices[0].platform == "cpu" else built

    def differences(self, other: BuildTarget) -> list[str]:
        """Each field `other` holds otherwise, said as this one's and other's: memory where other's is
        known and below this one's (a device holding more than a build planned for runs it), a library
        where this build loaded it and other's version is another (a build that loaded none wrote none of
        its state), and never the topology, which names the compile and binds no loader."""
        named = [field.name for field in dataclasses.fields(self)
                 if field.name not in ("memory", "topology", "libraries", "chips", "device_kind")]
        found = [f"{name} {getattr(self, name)!r}, not {getattr(other, name)!r}" for name in named
                 if getattr(self, name) != getattr(other, name)]
        theirs = dict(other.libraries)
        found += [f"libraries {name} {version}, not {theirs.get(name)}" for name, version in self.libraries
                  if version is not None and theirs.get(name) != version]
        if self.memory is not None and other.memory is not None and other.memory < self.memory:
            found.append(f"memory {self.memory} bytes, not {other.memory}")
        if self.chips is not None and other.chips != self.chips:
            found.append(f"TPU chips {self.chips}, not {other.chips}")
        # A kind where the build states one (a host's own GPUs'), which a canonical build never does.
        if self.device_kind is not None and other.device_kind != self.device_kind:
            found.append(f"device kind {self.device_kind}, not {other.device_kind}")
        return found

    def own_devices(self) -> list:
        """This process's devices of the build's platform, which its loaders load onto and check (`held`):
        the host's CPUs for a CPU's build (`cpu`), which a GPU's process holds beside its GPUs, else the
        process's own."""
        return jax.devices("cpu") if self.cpu is not None else jax.devices()

    def held(self, devices: Sequence) -> None:
        """Refused, naming what differs, where `devices` under this process's JAX, XLA flags and host are not
        this target."""
        if differences := self.differences(BuildTarget.of(devices)):
            raise ValueError(f"these programs were compiled for another target than this process's: "
                             f"{'; '.join(differences)}")


def tpu_chips(devices: Sequence) -> str | None:
    """`devices`' physical layout where they are TPUs: each device's chip coordinates and its core on the
    chip, in order, as one string a record holds as it is; None for any other platform."""
    if not devices or devices[0].platform != "tpu":
        return None
    return ";".join(f"{','.join(map(str, device.coords))}:{device.core_on_chip}"
                    for device in sorted(devices, key=lambda device: (tuple(device.coords),
                                                                      device.core_on_chip)))


def programs_need(device, programs: Iterable[tuple[CompiledMemory | None, int, int]]) -> int | None:
    """The least allocator limit a GPU like `device` must hold to admit each of `programs`, the most of
    them: each a program's memory as XLA analysed it, the bytes a loop keeps beside it and those of them its
    first call builds, admitted as the planner admits a step (`memory.admission`) and read
    as a limit of that GPU's pool (`least_limit`, whatever share takes it). None for a device of no GPU
    pool (a TPU's, whose topology states its own; a CPU's, but one standing in for a GPU, `stand_in_gpus`)
    and where a program states no memory."""
    from dew.training.memory import admission, least_limit

    programs = list(programs)
    stated = [(memory, held, built) for memory, held, built in programs if memory is not None]
    gpu = device.platform == "gpu" or device in STATED_GPUS
    if not gpu or not programs or len(stated) < len(programs):
        return None
    return least_limit(device, [admission(memory, held, built) for memory, held, built in stated])


@dataclasses.dataclass(frozen=True)
class AheadPrograms:
    """Named programs compiled ahead for GPUs this process need not have (`compile_programs`): each program's
    executable (`AheadProgram`) and the target they were built for (`BuildTarget`). A process of that target
    loads them (`loaded`) and runs them with no tracing, lowering or compile
    (`dew.telemetry.compiles.executable_only`): a GPU node then spends its time on its device's work
    alone (ledger 2716)."""

    target: BuildTarget
    programs: dict[str, Executable]
    tuned: bool
    """Whether their compile took the GPU kind's tuned choices, as the device's own compile makes them
    (`report.autotuned`): one on a device, or for a node's topology within `report.for_devices`; a compile
    for a generation's canonical GPUs is XLA's untuned program (`compile_programs`), which `loaded`
    refuses where the process's own compile would tune (row 8422: untuned convolutions left cuDNN's runtime
    heuristic to pick a non-deterministic algorithm). Stated by every constructor, and a file saved
    without it refused at its load (row 8459)."""
    at_run: tuple[str, ...]
    """The programs leaving a choice to run time their device's own compile makes (`report.left_at_run`: a
    convolution compiled without its GPU, as `compile_programs` reads one for its memory alone), which
    `loaded` refuses on any GPU (row 8688)."""
    kept: dict[str, KeptProgram] | None = None
    """The programs a process compiled at Dew's compile point, each by its fingerprint on its devices
    (`options.placed`; `KeptProgram`, `keeping`): a process of their target runs each for a call that lowers
    to that program under those options over those devices (`kept`); None for a file of named programs
    alone."""

    def save(self, path: str | Path) -> None:
        """Write the programs to `path`, for `load` in a process of their target."""
        Path(path).write_bytes(filed(self))

    @staticmethod
    def load(path: str | Path) -> AheadPrograms:
        """The programs `save` wrote to `path`."""
        programs = _saved(path, AheadPrograms, "programs")
        _current(path, "programs", [*programs.programs.values(),
                                    *(each.executable for each in (programs.kept or {}).values())])
        return programs

    def loaded(self) -> dict[str, jax.stages.Compiled]:
        """Each program, by its name, loaded onto this process's first `BuildTarget.devices` devices of its
        platform (`BuildTarget.own_devices`), the ones it was compiled for: each called with the arguments it
        was compiled for runs it. Refused, naming what differs, where those devices, this JAX, its XLA flags
        or its host are not the target they were built for, and where they would not run as this process's
        own compile of them (`report.compiled_as_here`: untuned where it tunes, or a choice left to run time,
        `at_run`); each is admitted as a compiled program is (`dew.communicators.admit`)."""
        devices = self.target.own_devices()[:self.target.devices]
        self.target.held(devices)
        if not report.compiled_as_here(devices, tuned=self.tuned, at_run=bool(self.at_run)):
            raise ValueError(
                f"programs {', '.join(self.at_run)} were compiled without their GPU and leave a "
                "convolution's algorithm to run time (report.DEVICELESS_CONVOLUTION_PICKS): compile them on "
                "this GPU"
                if self.at_run else
                f"these programs were compiled for {self.target.generation}'s canonical GPUs, untuned (XLA's "
                "defaults where a device's compile tunes its convolutions and GEMMs): compile them for this "
                "node's topology (compile_programs' topology, $DEW_NODE_TOPOLOGY), or run with the autotuner "
                "off (xla_gpu_autotune_level 0)")
        return {name: program.loaded(devices) for name, program in self.programs.items()}


def _saved[Saved: (AheadOfTime, AheadPrograms)](path: str | Path, kind: type[Saved], what: str) -> Saved:
    """What a `save` of `kind` wrote to `path`, refused where it is not one (`what` it holds), or where it
    was saved by a Dew whose `kind` held fewer fields than this one's (a pickle restores those it held
    alone, and a field's class default would stand in for what it never said: rows 8459, 8759)."""
    loaded = pickle.loads(Path(path).read_bytes())
    if not isinstance(loaded, kind):
        raise ValueError(f"{path} holds no {what} compiled ahead of time")
    missing = [field.name for field in dataclasses.fields(kind) if field.name not in vars(loaded)]
    if missing:
        raise ValueError(f"{path} holds {what} compiled ahead by an earlier Dew, before its files held "
                         f"{', '.join(missing)}: compile it ahead again")
    return loaded


def _current(path: str | Path, what: str, held: Sequence[object]) -> None:
    """Refuse the `what` saved at `path` where its executables `held` are not this Dew's one form
    (`Executable`): saved by an earlier Dew, they are compiled again."""
    if not all(isinstance(executable, Executable) for executable in held):
        raise ValueError(f"{path} holds {what} saved by an earlier Dew, whose executables this one does not "
                         "load: compile them ahead again")


def host_cpu() -> str | None:
    """This host's processor as a CPU target states it (`BuildTarget.cpu`): as XLA's CPU compile, which
    chooses a program's code by it, reads it to key its own persistent programs
    (`jax._src.cache_key.host_cpu`), the first processor's vendor, family, model and instruction sets
    (an Arm part's), from Linux's `/proc/cpuinfo`. None where that reading names the platform alone (no
    readable `/proc/cpuinfo`, or one stating neither), which hosts of other instruction sets share: no
    CPU program is kept for such a host or handed on one (`cpu_stated`)."""
    described = cache_key.host_cpu()
    if described == " ".join((platform.system(), platform.machine(), platform.processor())):
        return None
    return hashlib.sha256(described.encode()).hexdigest()[:16]


def cpu_stated(target: BuildTarget, devices: Sequence) -> None:
    """Refuse to keep programs for a CPU whose host states no processor (`host_cpu`): two such hosts of
    other instruction sets would read as one target."""
    if devices[0].platform == "cpu" and target.cpu is None:
        raise ValueError("this host states no processor features (no /proc/cpuinfo), so a CPU program "
                         "cannot be kept for it")


@dataclasses.dataclass(frozen=True)
class KeptProgram:
    """A program a process compiled at Dew's compile point, kept for a later process (`keeping`) by its
    fingerprint (`dew.nn.kernels.options.program_digest` under the options it compiled with): its
    function's name (`OuterProgram.called`), its executable and the ids of the devices it runs on, as its
    executable states them."""

    called: str
    executable: Executable
    devices: tuple[int, ...]
    """The ids of the devices it runs on, among the devices of its file's target (`jax.devices()`)."""

    def loaded(self) -> jax.stages.Compiled:
        """The executable, loaded onto this process's devices of the ids it was compiled for; refused,
        naming them, where the process holds none of an id."""
        held = {device.id: device for device in jax.devices()}
        if missing := [number for number in self.devices if number not in held]:
            raise ValueError(f"a kept program runs on devices {list(self.devices)}, and this process holds "
                             f"none of {missing}")
        return self.executable.loaded([held[number] for number in self.devices])


class Keeping:
    """The programs Dew's compile point compiles while `keeping` is open, each kept by its fingerprint, for
    `save`. `refused` names each it could not keep and why (a tree no file states, an executable XLA does
    not serialize): those compile again where the file is handed."""

    def __init__(self) -> None:
        self.kept: dict[str, KeptProgram] = {}
        self.refused: dict[str, str] = {}

    def heard(self, program: OuterProgram, lowered: jax.stages.Lowered,
              executable: jax.stages.Compiled) -> None:
        """Keep `executable`, which `program` compiled from `lowered`
        (`dew.nn.kernels.options.keeping`)."""
        named = f"{program.called} ({len(self.refused)})"
        try:
            # Stated before it is serialized: a tree of a node no process names (a class made inside a
            # function) is a program no file holds, which pickling its trees refuses by no error of its own
            # (an AttributeError on Python 3.12).
            fingerprint = program_digest(lowered, program.options)
        except Unestablished as refusal:
            self.refused[named] = str(refusal)
            return
        loaded = communicators.loaded_program(executable)
        if loaded is None:
            # Compiled for devices this process has not loaded (a compile-only GPU's), refused before it is
            # serialized: no program of its default backend.
            self.refused[named] = "it was compiled for devices this process has not loaded (a compile-only " \
                                  "target's), outside the process's default backend"
            return
        try:
            serialized = Executable.of(executable)
        except (NotImplementedError, ValueError) as refusal:
            # As `compile_programs` reads it: an array a program closed over is a constant of its
            # program, which no executable serializes.
            self.refused[named] = f"its executable does not serialize: {refusal}"
            return
        ran = loaded.devices
        if not set(ran) <= set(jax.devices()):
            # The file's target is the process's default backend's (`save`), which this program is none of.
            self.refused[named] = f"it runs on {sorted({device.platform for device in ran})} devices, " \
                                  "outside the process's default backend"
            return
        devices = tuple(device.id for device in ran)
        # By its devices too (`options.placed`): one program kept for two of a host's devices is two.
        self.kept[options.placed(fingerprint, devices)] = KeptProgram(program.called, serialized, devices)

    def save(self, path: str | Path) -> AheadPrograms:
        """Write the programs kept so far to `path` with this process's target, for `kept` in a process of
        it; returns what it wrote. Each program it could not keep is logged with its reason (`refused`). The
        target states what its programs need of a device (`programs_need`), not the limit of the GPU that
        kept them, so a smaller GPU that admits them loads them (row 8711, as rows 8094 and 8371 for a
        build compiled ahead)."""
        for named, reason in self.refused.items():
            _log.warning("the program %s is not kept, and compiles where the file is handed: %s", named,
                         reason)
        devices = jax.devices()
        need = programs_need(devices[0], [(kept.executable.memory, 0, 0) for kept in self.kept.values()])
        target = dataclasses.replace(BuildTarget.of(devices), memory=need)
        cpu_stated(target, devices)
        # Each compiled on this process's devices, which made its every choice itself.
        programs = AheadPrograms(target, {}, tuned=True, at_run=(), kept=dict(self.kept))
        programs.save(path)
        return programs


@contextlib.contextmanager
def keeping() -> Iterator[Keeping]:
    """Within the block, every program this thread compiles at Dew's compile point
    (`dew.nn.kernels.options.OuterProgram`: a scoring task's passes, a server's steps, an evaluation's
    programs) is kept by its fingerprint, and the `Keeping` it yields saves them:

        with keeping() as programs:
            server = ScoringServer.from_task(task, variables)
        programs.save("scoring.dew-aot")

    A later process of the same target hands them back (`kept`). A program compiled before the block, or
    taken within it, is not among them. On a CPU the block compiles with JAX's persistent cache off
    (`dew._cache.uncached`): an executable XLA:CPU loaded from that cache serializes without its kernels (it
    then loads and fails at its first call, "Function ... not found"; plain JAX, 2026-11-09), so each kept
    one is compiled here. A GPU's served from the cache serializes whole, its Pallas kernels among it (row
    6641: served, saved and run in a third process on a GH200, M-MC-30b), so a GPU process keeping its
    programs reads the cache as any compile does, where it compiled each again."""
    kept = Keeping()
    served_whole = jax.default_backend() != "cpu"
    with contextlib.nullcontext() if served_whole else uncached(), options.keeping(kept.heard):
        yield kept


@contextlib.contextmanager
def kept(path: str | Path) -> Iterator[AheadPrograms]:
    """Within the block, a call at Dew's compile point runs the executable kept at `path` (`keeping`) for
    the program it lowers to under its options, loaded onto the devices it was compiled for: the call
    traces and lowers as it would, and its fingerprint (`dew.nn.kernels.options.program_digest`) finds the
    kept executable, which runs in place of its compile (`options.taking`). Nothing is trusted but that
    fingerprint and the target: a call whose code, arguments, settings or the facts its trace read lower to
    another program finds none, and compiles as without the file, never running a stale executable. As
    JAX's own persistent cache does, keyed by the same lowered program, options and target; a file carries
    them where no cache is shared.

    A file kept for another target (its devices, JAX, jaxlib, XLA flags or host, `BuildTarget.differences`)
    hands nothing, said once: every call compiles. A file of programs compiled ahead by name, none kept by
    their programs (`compile_programs`), is refused as it is handed."""
    programs = AheadPrograms.load(path)
    if programs.kept is None:
        raise ValueError(f"{path} holds programs compiled ahead by name, none kept by its program "
                         "(`keeping`)")
    here = BuildTarget.of(jax.devices())
    differences = ([] if here.cpu is not None or jax.devices()[0].platform != "cpu"
                   else ["this host states no processor"]) + programs.target.differences(here)
    if differences:
        _log.warning("the programs kept at %s were compiled for another target than this process's, and "
                     "every call compiles: %s", path, "; ".join(differences))
        yield programs
        return
    # Each by its fingerprint on the devices it was compiled for (`options.placed`): loaded onto those, it
    # runs a call lowered over them alone.
    with options.taking({options.placed(key.partition("@")[0], each.devices): each.loaded
                         for key, each in programs.kept.items()}):
        yield programs


class Compile(Protocol):
    """A program's compile, as `compile_programs` makes each (`compiled`): `function` a program whose results
    it places (`out_shardings`), which `lower` lowers: Dew's compile point (`owned_program`), or `jax.jit`
    where the program's run compiles it so."""

    def __call__(self, function: Callable, /, *,
                 out_shardings: jax.sharding.Sharding) -> OuterProgram | jax.stages.Wrapped: ...


def _gpu_mesh(generation: str, multiprocessors: int | None) -> jax.sharding.Mesh:
    """One mesh axis over the compile-only GPUs of `generation` (`compile_only_gpus`)."""
    return jax.sharding.Mesh(np.asarray(compile_only_gpus(generation, multiprocessors=multiprocessors)),
                             ("devices",))


@contextlib.contextmanager
def lowering_for(generation: str, multiprocessors: int | None = None) -> Iterator[None]:
    """Within the block, programs trace and lower as `compile_programs` lowers them for one GPU of the CUDA
    `generation`: their launches derived for that GPU's facts (`assumed`, any GPU of the generation), on the
    compile-only GPUs' abstract mesh, certified on the first (`report.certifying`). A program traced in it
    holds the arrays a compile of it ahead holds (a prepare's, `tests/truths_ahead.py`)."""
    mesh = _gpu_mesh(generation, multiprocessors)
    with (assumed(built_target(generation, multiprocessors), portable=True),
          jax.sharding.use_abstract_mesh(mesh.abstract_mesh), report.certifying(mesh.devices.flat[0])):
        yield


def compile_programs(programs: Mapping[str, tuple[Callable, Sequence[jax.ShapeDtypeStruct]]],
                     generation: str, multiprocessors: int | None = None, *,
                     compiled: Compile = owned_program, topology: str | Path | None = None,
                     memory_only: bool = False) -> AheadPrograms:
    """`programs`, each a function and its arguments' shapes, dtypes and weak types (each a
    `ShapeDtypeStruct` or an abstract value), by its name, compiled ahead for
    one GPU of the CUDA `generation`, of the `multiprocessors` it states (`built_target`), on any host holding
    the CUDA plugin (`compile_only_gpus`), its launches derived for that GPU's facts (`lowering_for`), and
    serialized: a CPU host builds what a GPU node then
    only loads and runs (`AheadPrograms.loaded`, ledger 2716). Arguments and results are placed whole on that
    GPU, whatever the function reads. A function takes and returns arrays (a backward is a program of its own
    over its residuals); one whose program holds arrays it closed over is refused, named, as no executable
    serializes them, as is one holding a kernel this host cannot lower for any GPU of the generation
    (`selection.Unlowerable`: cuDNN's, whose plan holds to its GPU's model). Each is lowered in turn and all
    compile at once (`report.compiled_at_once`). `compiled(function, out_shardings=)` makes each program
    lowered: Dew's compile point (`owned_program`), or the compile the program's run takes where that is
    another (check 1's arms, `jax.jit`). Over a node's own `topology` (`host_gpus`, a job's prepare) the
    compile takes every tuned choice the GPU kind's runs left beside the compilation cache
    (`report.for_devices`), a choice none has tuned refused, named; over the generation's canonical GPUs
    it is XLA's untuned program, which its file says (`AheadPrograms.tuned`) and a tuning process refuses
    to load (row 8422). A program holding a choice only its GPU makes (`report.at_run_time`: a convolution)
    is refused before any compiles (`report.CompiledOnDevice`, naming each): no GPU's load would take it.
    With `memory_only` every program compiles read offline (`report.read_offline`), refused for none, for
    its memory alone (`Executable.memory`, a convolution's workspace XLA's default where its device's
    compile may pick another): programs no process loads, which a prepare sizes a run by."""
    target = built_target(generation, multiprocessors)
    gpus = (compile_only_gpus(generation, multiprocessors=multiprocessors) if topology is None
            else host_gpus(generation, topology, 1))
    mesh = jax.sharding.Mesh(np.asarray(gpus[:1]), ("devices",))
    placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    lowerings = []
    for name, (function, arguments) in programs.items():
        shapes = tuple(jax.ShapeDtypeStruct(argument.shape, argument.dtype, sharding=placed,
                                            weak_type=argument.weak_type)
                       for argument in arguments)

        def lowered(function=function, shapes=shapes, name=name) -> jax.stages.Lowered:
            """The program lowered for the target, afresh at each call (a relowering's, past a spill)."""
            try:
                with lowering_for(generation, multiprocessors):
                    return compiled(function, out_shardings=placed).lower(*shapes)
            except Unlowerable as refused:
                raise refused.refusing(name) from refused
        lowerings.append((lowered(), None, lowered))
    # Through Dew's compile point, as a run compiles them, at once (`report.compiled_at_once`): a launch
    # whose kernel spills there is derived again at its next config; for a node, tuned or refused.
    at_run = tuple(name for name, (lowered, _, _) in zip(programs, lowerings, strict=True)
                   if report.at_run_time(lowered))
    if at_run and not memory_only:
        # Their run compiles them on its GPU, whose choices they would leave to run time here.
        raise report.CompiledOnDevice(lowerings[0][0], at_run)
    with (report.read_offline() if memory_only else report.for_devices() if topology is not None
          else contextlib.nullcontext()):
        tuned = report.autotuned(gpus[:1])
        try:
            executables = report.compiled_at_once(lowerings)
        except report.Untuned as untuned:
            raise ValueError("a program is not preparable for this node until a run on its GPU has tuned "
                             f"it, leaving its choices beside the compilation cache: {untuned}") from untuned
    built = {}
    for name, compiled in zip(programs, executables, strict=True):
        try:
            built[name] = Executable.of(compiled)
        except (NotImplementedError, ValueError) as refused:
            raise ValueError(f"program {name!r} cannot be compiled ahead (an array it closed over is a "
                             "constant of its program, which no executable serializes: pass it as an "
                             f"argument): {refused}") from refused
    need = programs_need(gpus[0], [(executable.memory, 0, 0) for executable in built.values()])
    return AheadPrograms(BuildTarget.built(target, 1, need), built, tuned=tuned, at_run=tuple(
        name for name, compiled in zip(programs, executables, strict=True) if report.left_at_run(compiled)))


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


def target_devices(target: AheadTarget) -> tuple:
    """The compile-only devices the run's mesh is built over: the host's own GPUs where it names their
    `topology` (`host_gpus`), the TPU topology's, which must hold `devices`, or `devices` GPUs of the
    generation at the stated capacity."""
    if target.topology is not None:
        if target.multiprocessors is not None or target.capacity_gib is not None:
            raise ValueError("a host's topology states its GPUs' memory and multiprocessors; set neither "
                             "capacity_gib nor multiprocessors with it")
        return host_gpus(target.generation, target.topology, target.devices, target.share)
    if ":" not in target.generation:
        return compile_only_gpus(target.generation, target.devices, target.memory, target.multiprocessors,
                                 target.share)
    if target.multiprocessors is not None:
        raise ValueError(f"a TPU topology ({target.generation}) states its own cores; set no "
                         "multiprocessors")
    if target.capacity_gib is not None:
        raise ValueError(f"a TPU topology ({target.generation}) states its own memory; set no capacity")
    devices = compile_only_tpus(target.generation)
    if len(devices) != target.devices:
        raise ValueError(f"the TPU topology {target.generation} holds {len(devices)} devices, not "
                         f"{target.devices}")
    return devices


def compiled_for[LossT, EffectsT](target: AheadTarget, trainer: Trainer[LossT, EffectsT], loader: Loader,
                                  steps: int | None, examples: int | None = None,
                                  resume: Resume = "auto", plan: FitPlan | None = None) -> CompiledAhead:
    """`trainer`, built over this target's devices (`compile_only`), its first step compiled ahead
    (`compile_ahead`) and saved, as what ends the run, over the batch `fit` would start it on
    (`first_step_batch`), at the rung the checkpoint a run resumes under `resume` left it, with its
    evaluations' programs where `plan` is what its `fit` is asked for (`Trainer._fit_plan`). A run whose
    step reads a rollout's batches, which it makes as it trains, is refused. A run that keeps its
    variables on the host (`Layout(host=("variables",))`) compiles each program its step calls into the
    persistent cache its run's startup reads (`prepare_host_step`): many programs, saved to no file
    (row 8380: the configured run reached none of them)."""
    if trainer.rollout is not None:
        raise ValueError("compile_ahead lowers the step over the loader's batches; this run's step reads "
                         "its rollout's, which it makes only as it trains")
    if (target.path is None or target.phases) and target.topology is None and ":" not in target.generation:
        # A canonical target keys another entry than a host's run (`host_gpus`): no run reads it.
        raise ValueError(f"steps compiled ahead for {target.generation} GPUs into the persistent cache are "
                         "keyed as a host's run reads them only over that host's own topology: name it "
                         "(--target.topology $DEW_NODE_TOPOLOGY), or save the step to a "
                         "file (--target.path)")
    if trainer.host_master and (target.path is not None or target.phases):
        raise ValueError("a step that keeps its variables on the host compiles its programs into the "
                         "persistent cache its run reads, whole: name no path and no phases")
    # Refused before its first batch is read and its state traced, where its GPUs state no memory.
    stated_limit(list(trainer.device_mesh.devices.flat))
    batch = first_step_batch(trainer, loader, steps, examples, resume)
    if trainer.host_master:
        # Into a cache a device run reads as tuned, as a device step's (below): a program no device run
        # has tuned is counted, its run compiling it on its GPU, never a traceback (row 9693).
        try:
            return CompiledAhead(None, None, startup=prepare_host_step(trainer, batch))
        except report.Untuned as refused:
            # Named as every untuned program is (`program_name`), a host step's VJP results stating no
            # form.
            _log.warning("a host step is not preparable here: %s", refused)
            return CompiledAhead(None, None, (program_name(refused.lowered),))
    topology = None if target.topology is None else Path(target.topology).read_bytes()
    if target.path is not None and not target.phases:
        # Over a host's own GPUs (`topology`), as its run compiles it: every tuned choice the GPU kind's
        # runs left beside the compilation cache, all or none (`report.for_devices`), so its run takes
        # it. Over a generation's canonical GPUs, XLA's untuned program, read for its plan and its
        # kernels, which its record and its save say and a run compiles in its place (`runnable`).
        with report.for_devices() if target.topology is not None else contextlib.nullcontext():
            try:
                ahead = _compiled_ahead(trainer, batch, frontier=target.frontier, plan=plan,
                                        topology=topology)
            except report.CompiledOnDevice as refused:
                raise ValueError(f"this step is compiled on its GPU, not ahead: {refused}") from refused
            except report.Untuned as untuned:
                raise ValueError(
                    "this step is not preparable on this host until a run on such a GPU has tuned it, "
                    "leaving its choices beside the compilation cache: run it there once, then compile "
                    f"it ahead again ({untuned})") from untuned
        ahead.save(target.path)
        return CompiledAhead(target.path, ahead)
    # Into the persistent cache the run reads alone: the steps of the phases named, or the first step
    # and each later phase's, as a job's prepare compiles its node's run (`topology`, row 5460). A
    # phase's step is planned afresh at the split the run was given, as its run plans it at its
    # boundary; the evaluations' programs compile with the first step's.
    batches = ([phase_batch(trainer, batch, phase) for phase in target.phases] if target.phases else
               [batch, *(phase_batch(trainer, batch, phase) for phase in later_phases(trainer, batch))])
    ahead, untuned = None, []
    for index, stepped in enumerate(batches):
        first = not index and not target.phases
        if not first:
            trainer._planned_afresh()
        try:
            # Into a cache a device run reads as tuned: every choice the GPU kind's runs tuned, or none.
            with report.for_devices():
                ahead = _compiled_ahead(trainer, stepped, frontier=False, plan=plan if first else None,
                                        topology=topology)
        except report.Untuned as refused:
            # Its run compiles it on the device, which tunes it (and the next prepare finds it), or makes
            # the choice a compile without it cannot (`report.CompiledOnDevice`).
            _log.warning("a step is not preparable here: %s", refused)
            untuned.append(program_name(refused.lowered))
            continue
        _log.info("a step compiled ahead into the cache in %.1f s", ahead.planned_seconds)
    return CompiledAhead(None, ahead, tuple(untuned))


def stated_limit(devices: Sequence) -> int | None:
    """The allocator's limit a step compiled ahead for `devices` is planned against: the least their stated
    memory gives (`device_memory`); None for a TPU topology's, which states its own. Compile-only GPUs that
    state none are refused, before anything traces, naming how to state it once."""
    if {device.platform for device in devices} == {"tpu"}:
        return None
    limits = [device_memory(device).get("bytes_limit") for device in devices]
    if None in limits:
        generation = gpu_target(devices[0].device_kind).generation
        raise ValueError(f"the {generation} GPUs this step compiles for state no memory, which every plan "
                         "for them is made against: state it once, with each GPU's memory "
                         "(--target.capacity-gib, compile_only_gpus' memory), or a host's own "
                         "serialized topology (--target.topology), which states its GPUs' own")
    return min(limit for limit in limits if limit is not None)


def first_step_batch[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], loader: Loader, steps: int | None,
                                      examples: int | None = None, resume: Resume = "auto") -> Batch:
    """The batch `trainer`'s first step takes from `loader`, as `fit` would start the run: an input its
    objective names alone declared first (`Trainer._declare_input`), the loader read through its device
    steps and the run's clock (its `steps`, or the `examples` it is bounded to, `Trainer.reading`), the
    run checked (`Trainer.validate`), and the first batch of the stream of this process's share fit opens
    (`Trainer._stream_partition`, its ramp's stages beginning where updates do) from the position its
    checkpoint left a run resumed under `resume` at (`Checkpoints.position`), under the layout's rules,
    or started again where its plan changed and the run resumes its data relaxed, as the run's resume
    reads it (`dew.training.startup.resume_admitted`, row 3917). A run its checkpoint
    completes has no step to take, and is refused. What a step compiled ahead (`AheadTarget.compiled`)
    and an offline view of a run (`dew.config.RunConfig.viewed`) start from."""
    from flax import linen as nn

    from dew.data.dataset import DataPartition
    from dew.training._fit import data_resumed, first_batch
    from dew.training.startup import resumed_step

    trainer._declare_input(lambda: first_batch(loader, trainer._stream_partition()))
    trainer.reading(loader, examples=examples, steps=steps)
    trainer.validate(steps=steps)
    checkpoints = trainer.checkpoints
    resumed = resumed_step(trainer, resume)
    if resumed is not None and steps is not None and resumed >= steps:
        raise ValueError(f"the run is complete at step {resumed}, its last checkpoint written, so it has no "
                         "step to take")
    # The shares fit reads by, its layout placing the rows, the saved one's among them, read on from or
    # started again as the run's data resume decides (`Trainer.data_resume`, `data_resumed`).
    with nn.logical_axis_rules(trainer.layout.axis_rules):
        partition = trainer._stream_partition()
        if checkpoints is None or resumed is None:
            return first_batch(loader, partition)
        saved = DataPartition.of(trainer.device_mesh)
        position, _ = data_resumed(loader, partition, lambda: checkpoints.position(resumed, saved),
                                   resume=trainer.data_resume, step=resumed)
        return first_batch(loader, partition, position)


class CompiledAhead(SystemExit):
    """Raised by a configured run under `compile_ahead` once its first step is compiled and saved
    (`AheadTarget.compiled`), or its later phases' steps compiled into its cache (`AheadTarget.phases`).
    Compiling was the whole run, so uncaught it ends the program with status 0 and no traceback, the
    step's cost logged; `ahead` is the step (the last phase's) and `path` where it was saved, if it was;
    `untuned` the digests (`program_digest`) of the programs no device run has tuned yet, each of a step not
    preparable (`report.Untuned`); `startup` what a host step's programs compiled into the cache
    (`prepare_host_step`), its step none."""

    def __init__(self, path: str | None, ahead: AheadOfTime | None, untuned: tuple[str, ...] = (), *,
                 startup: HostStartup | None = None):
        super().__init__(0)
        self.path, self.ahead, self.untuned, self.startup = path, ahead, untuned, startup
        if startup is not None:
            _log.info("a host step's %d programs compiled ahead into the compilation cache in %.1f s (%s)",
                      startup.programs, startup.seconds, startup.by_backend)
            return
        if ahead is None:
            _log.info("no step compiled ahead: none was preparable until a device run has tuned it")
            return
        _log.info("the step compiled ahead for %s %s GPUs holds %s bytes a device at its peak; saved to "
                  "%s", ahead.devices, ahead.generation, ahead.peak_bytes, path or "the compilation cache")
        if ahead.plan is not None:
            probes = record(ahead.plan, "plan")["probes"]
            assert isinstance(probes, list)
            _log.info("planned on them in %.1f s, %d programs compiled, the plan %s", ahead.planned_seconds,
                      len(probes), json.dumps(ahead.plan))


def phase_batch[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch, phase: int) -> Batch:
    """`batch`, the run's first (`first_step_batch`), as its data phase `phase` delivers one: marked read in
    it (`dew.batch.PHASE`), so its step is built at that phase's pass count (`PhasePasses`), and holding that
    phase's rows (`RunShape.rows`, a phase each), each row field's rows the first's taken in turn: the
    shapes and fields that phase's step compiles over, read from no reader. Its other fields, widths and
    dtypes are the first phase's, as one recipe's phases hold them; a phase whose reader delivers others
    lowers another program at its boundary, which compiles there as it would have."""
    from dew.batch import PHASE, Phase

    rows = trainer._shape.rows
    held = next(len(leaf) for name, leaf in batch.items() if name != PHASE and np.ndim(leaf))
    wanted = held if not rows or len(rows) == 1 else rows[phase]

    def resized(leaf):
        return np.resize(leaf, (wanted, *np.shape(leaf)[1:])) if np.ndim(leaf) and len(leaf) == held else leaf

    return {**{name: jax.tree.map(resized, leaf) for name, leaf in batch.items() if name != PHASE},
            PHASE: Phase(phase)}


def later_phases[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch) -> tuple[int, ...]:
    """The data phases after the one `batch`, the run's first step's as its loop read it (its work counted
    and its rows marked, `step_batch`'s but for its host fields), was read in whose steps are other
    programs than its and each other's, each by its first phase: a phase's step is the trainer's own key of
    the batch it delivers (`phase_batch`, `PhasePasses.key_at` its pass count); none for a run of no
    phases. Its work is counted once, as its loop counted it: counted again, its telemetry columns were
    refused as reserved, before a 4B run's first step (M-MC-3gB)."""
    from dew.batch import PHASE
    from dew.distributed.batches import batch_signature, split_fields

    counts = trainer.phase_passes.counts
    if PHASE not in batch or not counts:
        return ()

    def program(phase: int) -> Hashable:
        delivered = split_fields(phase_batch(trainer, batch, phase))[0]
        return trainer.phase_passes.key_at(batch_signature(delivered), counts[phase])

    first = batch[PHASE].index
    firsts: dict[Hashable, int] = {program(first): first}
    for phase in range(first + 1, len(counts)):
        firsts.setdefault(program(phase), phase)
    return tuple(sorted(set(firsts.values()) - {first}))


class LaterPlans:
    """The later data phases' steps of a run (`later_phases`), each planned and compiled on a thread of this
    process once the run's first step is dispatched, over its state's shapes as they stand at an update's end
    and the batch each phase delivers (`phase_batch`), at the split the run was given or the planner's own,
    its objective a copy at its pass count (`PhasePasses.apart`), so the loop's objectives and plan are never
    touched. Each holds every executable it compiled, those that missed on the devices the running phase holds
    among them (`FitSearch.provisional`), and the phase's boundary waits for it (`waited`), whose plan, made
    there against the devices' allocators as they stand, finds each candidate's trace (`candidate_for`) and
    executable (`FitSearch.executables`, the run's compiler's) made: the GPUs no longer idle while it compiles
    (336 s at a 4B decision run's second phase on a GH200, M-MC-3q). One thread, joined: each phase's at its
    boundary, every phase's as the run ends, however it ends. A phase whose plan failed off the loop says so
    at its boundary, which plans it as it would have. A pooled run, a host-master or rollout run, one that
    tunes its kernels or runs PGLE plans its phases at their boundaries."""

    def __init__(self, futures: Mapping[int, Future[Release]]) -> None:
        self.futures = dict(futures)

    @classmethod
    def started[LossT, EffectsT](cls, trainer: Trainer[LossT, EffectsT], batch: Batch,
                                 state: TrainState) -> LaterPlans | None:
        """The thread planning the later phases of `trainer`'s run, begun where they hold steps of other
        programs than the step of `batch`, its first (`later_phases`), over the shapes of `state`, the
        state its first step returned; None where it plans none."""
        from dew.batch import PHASE
        from dew.distributed.batches import split_fields

        if (jax.process_count() > 1 or trainer.host_master or trainer.rollout is not None
                or trainer.tune_kernels or pgle_runs() or state.inner is not None):
            return None
        phases = later_phases(trainer, batch)
        if not phases:
            return None
        from dew.training._candidates import _shapes
        from dew.training.compiler import fresh

        shapes = _shapes(fresh(state))
        counts = trainer.phase_passes.counts
        begun = trainer.phase_passes.begun(int(batch[PHASE].index), phases)
        # Each plan traces under the loop's contexts as it stands, its JAX states carried (`carried`: they
        # hold per thread), its context variables copied.
        entered = carried()

        def planned(phase: int) -> Release:
            with entered():
                return _phase_plan(trainer, shapes, split_fields(phase_batch(trainer, batch, phase))[0],
                                   counts[phase])

        pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="dew-later-phases")
        try:
            futures = {counts[phase]: pool.submit(contextvars.copy_context().run, planned, phase)
                       for phase in begun}
        finally:
            pool.shutdown(wait=False)
        _log.info("planning the steps of data phases %s off the loop", begun)
        return cls(futures)

    def waited(self, passes: int | None) -> Release:
        """The plan of the phase whose step runs `passes` waited for, so the executables it compiled, which
        the run's compiler holds (`FitSearch.executables`), are there for the boundary's plan, and what drops
        every one of them but those the boundary's plan takes, which the boundary calls once it takes one
        (`FitSearch.released_unplanned`); one that failed said, the boundary planning it as it would have."""
        future = self.futures.pop(passes, None) if passes is not None else None
        if future is None:
            return _kept
        try:
            return future.result()
        except Exception as error:  # the boundary plans the phase as it would have
            _log.warning("planning the step of %d passes off the loop failed; its boundary plans it: %r",
                         passes, error)
            return _kept

    def stopped(self) -> None:
        """Every phase's plan waited for as the run ends, a pending one not begun, one that failed said, and
        every program a phase the run never reached compiled dropped (`Release`)."""
        for passes, future in self.futures.items():
            if future.cancel():
                continue
            if (error := future.exception()) is not None:
                _log.warning("planning the step of %s passes off the loop failed: %r", passes, error)
            else:
                future.result()(None)
        self.futures.clear()


type Release = Callable[[Planned | None], None]
"""What drops every program a later phase's plan off the loop held (`FitSearch.released_unplanned`) but the
plan its boundary took, all of them where none (a phase its run never reached)."""


def _kept(planned: Planned | None) -> None:
    """The release of a phase planned off the loop by nothing: nothing to drop."""


def _phase_plan[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], state: TrainState, batch: Batch,
                                 passes: int) -> Release:
    """A plan of `trainer`'s step at `passes` over `state`'s shapes and `batch` (`LaterPlans`), on a compiler
    and objective of its own, its executables held by the run's compiler (`FitSearch.executables`), those it
    compiled before a refusal where no candidate fit the devices as they stand off the loop among them, which
    the boundary judges again."""
    from dew.training.compiler import StepCompiler
    from dew.training.planning import PlanRefused

    phases = trainer.phase_passes.apart(passes)
    compiler = StepCompiler(phases.base, trainer._compiler.links(trainer.device_mesh), None, phases,
                            trainer.fit_policy.remat, executables=trainer._compiler._executables)
    stepping = dataclasses.replace(trainer._stepping(phases.base, compiler), shape=trainer._afresh_shape())
    try:
        compiler._plan(stepping, state, batch, None, provisional=True)
    except PlanRefused as refused:
        _log.info("the step of %d passes fits no plan off the loop; its boundary plans it: %s", passes,
                  refused)
    except BaseException:
        # Its search's programs dropped where it raised otherwise, its error the boundary's to plan past.
        if (settle := compiler._provisional()) is not None:
            settle(None)
        raise
    settle = compiler._provisional()
    return _kept if settle is None else settle

def step_batch[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch,
                                 partition: DataPartition | None = None) -> Batch:
    """`batch`, the rows a loader gives this process, its share `partition` of the step's (the mesh's,
    `DataPartition.of`, where None), as `fit` hands it to its step: its work counted
    (`Trainer._counted_batch`), each row marked with its place among the step's records
    (`dew.training.distributed.marked`), its host fields set apart (`split_fields`), and on the examples
    clock whether it closes its update window (`closes_window`). A batch of every row, read whole on
    each process, is `DataPartition()`'s."""
    from dew.data.dataset import DataPartition
    from dew.training._fit import handed
    from dew.training.distributed import marked

    partition = DataPartition.of(trainer.device_mesh) if partition is None else partition
    return handed(marked(partition, trainer._counted_batch)(batch), trainer._shape.examples)


def compile_ahead[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch, *, frontier: bool = False,
                                   fit: FitArguments | None = None, topology: bytes | None = None
                                   ) -> AheadOfTime:
    """`trainer`'s first step over a batch of `batch`'s shapes, compiled for the compile-only devices its
    mesh is built over, GPUs of one CUDA generation (`compile_only_gpus`) or a TPU topology's
    (`compile_only_tpus`), and never run. The planner `Trainer.compile` uses
    (`Trainer._compiled_plan`) selects and compiles the step once; that executable is serialized, not compiled
    again. Its batch is what `fit` hands to its step (`step_batch`), its fresh state abstract, and
    tracing reads the target's kernel choices (`assumed`). The plan reports every compile and the
    time the search took, and with `frontier` each rung above it that fits at fewer microbatches than
    every plan before it (`FitSearch.above`) and each lighter rung below it its search passed over that fits
    (`lighter`, `FitSearch.below`), compiled and not saved: the plans a run's speed is chosen
    among (the module's docstring). With `fit`, what the run's `Trainer.fit` over `batch` is asked for, its
    evaluations' programs compile with its start's (`started`). The GPUs' memory is the one their
    compile-only devices carry (`compile_only_gpus`' `memory`, a host topology's own), the plan made against
    the allocator's limit derived from it; GPUs that state none are refused before anything traces. The
    GPU it compiles for is the one the mesh's
    compile-only GPUs state, as XLA compiles over them: their generation and multiprocessors
    (`compile_only_gpus`' `multiprocessors`, `built_target`), which the step's launches derive for and it
    records. `topology` is the serialized topology of a host's own GPUs the mesh's devices were read from
    (`host_gpus`), which the saved build names (`BuildTarget.topology`); None for a generation's canonical
    GPUs. Profiling, tuning and rollout require execution and are refused."""
    return _compiled_ahead(trainer, batch, frontier=frontier,
                           plan=None if fit is None else fit_plan(trainer, batch, fit), topology=topology)


def fit_plan[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], dataset: Dataset | Loader | Handed,
                              fit: FitArguments) -> FitPlan:
    """What `trainer.fit(dataset, **fit)` runs (`Trainer._fit_plan`, which states `fit`'s defaults)."""
    return trainer._fit_plan(dataset, **fit)


def _compiled_ahead[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch, *, frontier: bool,
                                     plan: FitPlan | None, topology: bytes | None = None) -> AheadOfTime:
    """`compile_ahead`'s step and programs, the evaluations' with `plan`, what the run's `fit` is asked for
    (`Trainer._fit_plan`), over its own data or a configured run's loader (`AheadTarget.compiled`), with its
    planning's seconds by their parts (`AheadOfTime.planned_parts`)."""
    from dew.telemetry.compiles import Ledger
    from dew.training.distributed import START_PROGRAM
    from dew.training.planning import spending

    with report.recording() as recorded, spending() as spent, Ledger(START_PROGRAM) as start:
        ahead = _planned_ahead(trainer, batch, frontier=frontier, plan=plan, topology=topology)
    # The planning's seconds by its parts, which perf_gate reports beside its whole (row 5339): the step's
    # JAX traces, lowerings and compiles (the planner's own reading of them, `spending`), the fresh state's
    # start compiled beside them (`start`, its own program's; row 6740: the gate read it as the step's) and
    # the algebra's passes.
    started = {kind: sum(made.seconds for made in start.programs if made.kind == kind) for kind in spent}
    stepped = {kind: seconds - started[kind] for kind, seconds in spent.items()}
    return dataclasses.replace(ahead, planned_parts={**stepped, "start": sum(started.values()),
                                                     "algebra": sum(recorded.passes)})


class _Targeted(NamedTuple):
    """What a step compiled ahead is targeted at (`_targeted`): its devices, their generation and the
    allocator limit they state, and the run's start."""

    devices: list
    generation: str
    capacity: int | None
    start: Start


@contextlib.contextmanager
def _targeted[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch, *, what: str,
                               platforms: tuple[frozenset[str], ...]) -> Iterator[_Targeted]:
    """Within the block, `trainer`'s step is traced and compiled for the compile-only devices of one
    generation its mesh is built over (a platform among `platforms`), as a run on them traces it: its start
    (`Trainer._started`), a resumed run's at the rung and split its checkpoint records, its first batch's
    pass count, the target's devices assumed (`assumed`), each launch certified on one of them, and the
    run's phase scoped. The one preamble `compile_ahead` (`_planned_ahead`) and `prepare_host_step` share
    (row 8380: a copy of it drifted, a host step prepared ahead missing a resumed run's rung). `what` names
    the caller in a refusal."""
    if trainer.tune_kernels or pgle_runs():
        raise ValueError(f"{what} cannot run kernel tuning or AutoPGLE on absent GPUs")
    devices = list(trainer.device_mesh.devices.flat)
    generations = {gpu_target(device.device_kind).generation for device in devices}
    found = frozenset(device.platform for device in devices)
    if found not in platforms or len(generations) != 1:
        raise ValueError(f"{what} compiles for the compile-only devices of one generation a trainer's mesh "
                         "is built over (MeshSpec.build(compile_only_gpus(generation, count)), or a TPU "
                         f"topology's, compile_only_tpus(topology)); this mesh holds "
                         f"{sorted({device.device_kind for device in devices})}")
    (generation,) = generations
    on_tpus = found == {"tpu"}
    # The allocator's limit the plan is made against, from the memory the GPUs' devices carry.
    capacity = stated_limit(devices)
    # The GPU XLA compiles for, as its compile-only devices state it.
    # Its kind where the devices are a host's own (`_HOSTS_GPUS`), whose launches derive from that part's
    # facts as its run does (row 13252); a canonical GPU's stands for any of its generation.
    kind = None if on_tpus or not set(devices) <= _HOSTS_GPUS else devices[0].device_kind
    stated = None if on_tpus else built_target(generation, devices[0].core_count, capacity, device_kind=kind)
    on = generation if stated is None else stated
    # A canonical target's devices stand for any GPU of the generation; a host's own topology's, for its own.
    portable = not on_tpus and not set(devices) <= _HOSTS_GPUS
    # Each launch's derived config certified on one of the compile-only devices it compiles for
    # (`report.certified`), as a run on that GPU certifies it: the cache it prepares holds that run's.
    certifying = contextlib.nullcontext() if on_tpus else report.certifying(devices[0])
    # A library that reads the GPU it is traced for (JAX's cuDNN attention gates, row 5894) reads the
    # target's from the concrete mesh the step traces on (`Trainer._traced_on`); made the process's default
    # device besides, the compile-only GPU changed the program a run on that GPU compiles first, which then
    # refused the step (row 6163). The start is traced within it too, as a run on that GPU traces it: traced
    # before, in a build that sees no GPU, its initializer's launches took the CPU's routes and certified
    # nothing, so the run certified them on its GPU (rows 7322, 8299: five init-shape launches).
    # A host step's start runs on its CPU companion (`Trainer._started`), traced as that run traces it.
    start = trainer._started() if trainer.host_master else None
    with assumed(on, portable=portable), certifying:
        start = trainer._started() if start is None else start
        if start.resume is not None and trainer.checkpoints is not None:
            # A resumed run's step compiles at the rung its checkpoint records and the split its state was
            # saved at (a checkpoint keeps a window only part filled, `dew.state.without_empty_window`), as
            # its resume would: a split-2 checkpoint compiled at split 1 was another program than its
            # resume's.
            from dew.training.compiler import resumed_stepping

            trainer._compiler.resume_rung(trainer.checkpoints.rung(start.resume))
            resumed = resumed_stepping(
                trainer._stepping(), trainer.checkpoints.window_size(start.resume),
                held=start.abstract.window is not None, pools=start.abstract.window is not None,
                replicas=start.abstract.inner is not None and start.resume > 0)
            trainer._shape = resumed.shape
        # The first step's pass count, the phase its batch was read in (`Trainer`'s `phase_passes`).
        trainer.phase_passes.at(batch)
        with trainer.phase_passes.scoped(trainer, trainer._compiler):
            yield _Targeted(devices, generation, capacity, start)


def _planned_ahead[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch, *, frontier: bool,
                                    plan: FitPlan | None, topology: bytes | None) -> AheadOfTime:
    """`_compiled_ahead`'s planning and compile: the step's plan, its programs and their record."""
    from dew.distributed.placement import placed_shapes

    if trainer.host_master or trainer.rollout is not None or trainer.lower_for is not None:
        raise ValueError("compile-ahead needs a device-resident step over its loader, with no lower_for")
    began = time.perf_counter()
    try:
        with _targeted(trainer, batch, what="compile_ahead",
                       platforms=(frozenset({"gpu"}), frozenset({"tpu"}))) as targeted:
            devices, generation, capacity, start = targeted
            target = gpu_target()
            # A fresh run's start compiles beside its step's planning, which reads none of it.
            from dew.training.distributed import start_compiling

            begun = None if start.resume is not None or trainer.host_master else start_compiling(
                start.placing, start.initializer, shapes_of(start.key),
                start.abstract, start.shardings, trainer.device_mesh)
            stepped = step_batch(trainer, batch)
            planned = trainer._compiled_plan(start.abstract, stepped, None, frontier)
            with attempting(begun):
                starts = started(trainer, start, planned, plan)
    except Unlowerable as refused:
        raise refused.refusing("step") from refused
    # The sampled step as the search compiled it (`_sampled_ahead`).
    sampled = _sampled_ahead(planned)
    seconds = time.perf_counter() - began
    lowered, compiled = planned.candidate.lowered, planned.executable
    memory = compiled_memory(compiled)
    if memory is None:
        raise ValueError(f"XLA states no memory of the step it compiled for {generation}")
    programs = _step_programs(planned, starts, placed_shapes(start.abstract, start.shardings))
    need = programs_need(devices[0], [(compiled_memory(executable), held, built)
                                      for executable, held, built in programs])
    return AheadOfTime(
        generation, len(devices), dict(trainer.device_mesh.shape), program_digest(lowered),
        Executable.of(compiled),
        flops=step_flops(compiled),
        argument_bytes=memory.argument, output_bytes=memory.output, alias_bytes=memory.alias,
        temp_bytes=memory.temp, code_bytes=memory.generated_code, live_bytes=memory.live, plan=trainer.plan,
        planned_seconds=seconds,
        capacity=capacity, links=dict(trainer.links), target=target,
        built=BuildTarget.built(target, len(devices), need, topology=topology, chips=tpu_chips(devices),
                                device_kind=None if topology is None else devices[0].device_kind),
        refusals=report.refused_launches(), sampled=sampled,
        at_run=any(report.left_at_run(executable) for executable, _, _ in programs),
        starts={digest: Executable.of(executable) for digest, (executable, _) in starts.items()})


def _step_programs(planned: Planned, starts: Started, placed: TrainState
                   ) -> list[tuple[jax.stages.Compiled, int, int]]:
    """Every program a run of a step compiled ahead runs, each with the bytes its run holds beside it and
    those of them its first call builds (`memory.admission`'s `held` and `built`): the one list the build's
    need (`programs_need`) and its choices left to run time (`report.left_at_run`) read (row 8713). Its
    step and sampled step beside what their loop keeps (`Candidate.held`); its fresh state's build beside
    none, run before any state is placed; every other program it runs (`started`) beside what the run holds
    between its steps (the step's arguments, its first call's builds charged once, and what its loop
    keeps), less the bytes of the state among the program's own arguments (row 8710: a program run beside
    the placed state admitted beside none, so a device admitted at the need ran out of memory where the
    run's own admission refused it); and each beside what of the run's placed state (`placed`) the step's
    arguments hold none of (`_unread`). The step's memory is stated (`_planned_ahead`)."""
    candidate = planned.candidate
    memory = compiled_memory(planned.executable)
    assert memory is not None
    unread = _unread(planned.executable, placed, candidate.built)
    resident = memory.argument - candidate.built + candidate.held + unread
    steps = [planned.executable, *([] if planned.sampled is None else [planned.sampled.executable])]
    return [*((step, candidate.held + unread, candidate.built) for step in steps),
            *((executable, 0 if among is None else max(0, resident - among), 0)
              for executable, among in starts.values())]


def _unread(step: jax.stages.Compiled, placed: TrainState, built: int) -> int:
    """The bytes of the run's placed state (`placed`, each leaf in a device's own memory) the step's
    arguments hold none of: its state as its arguments carry it (its moving and frozen trees,
    `step_program`'s first two), each leaf XLA kept there (a leaf it drops holds no sharding,
    `Compiled.input_shardings`), less its first call's builds (`built`), which the placed state holds none
    of. A frozen leaf no step reads is among them, held by the devices all the same (row 8710's review: a
    need read off the step's arguments left out a leaf the run holds, which its evaluation may read)."""
    from dew.distributed.placement import placed_bytes

    (arguments, _), (shardings, _) = step.in_avals, step.input_shardings
    # Each kept leaf beside its sharding, a dropped one beside None, by the arguments' own structure (a field
    # the state holds none of, its window's, a node of neither).
    read = [jax.ShapeDtypeStruct(aval.shape, aval.dtype, sharding=sharding)
            for tree, laid in zip(arguments[:2], shardings[:2], strict=True)
            for aval, sharding in zip(jax.tree.leaves(tree), jax.tree.structure(tree).flatten_up_to(laid),
                                      strict=True)
            if sharding is not None]
    return max(0, placed_bytes(placed, memory="device") - (placed_bytes(read, memory="device") - built))


def _sampled_ahead(planned: Planned) -> AheadProgram | None:
    """A step's sampled step where the run inspects, as a program compiled ahead (`AheadOfTime.sampled`). Its
    program is what the search lowered (the plan unreleased, its lowering held), digested as the step's is:
    a program no file can hold is refused by name."""
    if planned.sampled is None:
        return None
    if planned.sampled.lowered is None:
        raise ValueError("a sampled step's plan was released before it was compiled ahead")
    return AheadProgram.of(program_digest(planned.sampled.lowered), planned.sampled.executable)


def prepare_host_step[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch) -> HostStartup:
    """Compile the programs `trainer`'s host step calls (`Layout(host=("variables",))`) for the GPUs its mesh
    is built over (compile-only GPUs of a generation, or a host's own topology's, `host_gpus`), over its
    fresh state's and `batch`'s shapes as its run's startup does (`StepCompiler._compile_host`), its CPU
    companion this process's CPUs: JAX's persistent cache, keyed by what each program lowers to under its
    options and target, then holds each, and the run's startup on those GPUs reads them, compiling none
    (training.md's warm start; Main's choice for row 7908: no file, the shared cache). Its GPU programs are
    compiled for the devices to run (`report.for_devices`), each taking the tuned choices a run of the GPU
    left, `report.Untuned` where none has and the run tunes. What the pass compiled, by backend
    (`HostStartup`)."""
    if not trainer.host_master:
        raise ValueError("prepare_host_step prepares a step that keeps its variables on the host "
                         "(Layout(host=('variables',))); compile a device step ahead with compile_ahead")
    # For a device to run, as its run compiles them there (`report.for_devices`): every option a device's
    # compile takes, so each program keys the persistent cache alike (rows 8372, 8446: compiled as read
    # offline, its programs took no CUB sort mode, which the run's took, and the run missed them).
    with report.for_devices(), _targeted(trainer, batch, what="prepare_host_step",
                                         platforms=(frozenset({"gpu"}),)) as targeted:
        return trainer._compiler.prepared_host(trainer._stepping(), targeted.start.abstract,
                                               step_batch(trainer, batch))


type Started = dict[str, tuple[jax.stages.Compiled, int | None]]
"""The programs a run compiles besides its step (`started`), each by its digest with the bytes of the run's
state among its arguments, None for one run before any state is placed."""


def started[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], start: Start, planned: Planned,
                             plan: FitPlan | None = None) -> Started:
    """The programs a run of `trainer` compiles besides its step, compiled for the target over their
    arguments' shapes (`AheadOfTime.starts`): its fresh state's build and placement (`start_compiled`), for
    a run that starts afresh, the first step's narrowed copies of its weights where the plan reads some
    (`narrowed`), and what its steps run on their results, each called as the run calls it, over the
    step's results' shapes (`collecting`): an accumulating window's first allocation (`Trainer._window_of`),
    a split step's whole (`whole_step`), its counters (`bookkeep`), a sampled step's latest sample
    (`latest`), its log's copies (`copied`) and its logged tokens' sum (`added_tokens`), and with the plan
    of its `fit` (`plan`), what its evaluations' warming pass compiles before they run
    (`Evaluator.warmed_ahead`; row 4752). Each by its program's digest, through the kernel templates'
    reports as every compile is, with the bytes of the run's state among its arguments, which the run holds
    while it runs (`_step_programs`): None for the fresh state's build, run before any state is placed."""
    from dew.training.distributed import start_compiled
    from dew.training.narrow import narrowing

    candidate = planned.candidate
    programs: Started = {}
    if start.resume is None and not trainer.host_master:
        with _collected(programs, lambda executable: None):
            start_compiled(start.placing, start.initializer,
                           shapes_of(start.key), start.abstract,
                           start.shardings, trainer.device_mesh)
    if candidate.copies:
        # The variables as the run's first step holds them, placed as its start placed them.
        variables = jax.tree.map(lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype,
                                                                             sharding=sharding),
                                 start.abstract.variables, start.shardings.variables)
        with _collected(programs, _arguments):
            narrowing(candidate.copies, candidate.shardings.compute).compiled(variables)
    _results_programs(trainer, start, planned, plan, programs)
    return programs


@contextlib.contextmanager
def _collected(programs: Started, among: Callable[[jax.stages.Compiled], int | None]) -> Iterator[None]:
    """Within the block, each program Dew's compile point compiles (`collecting`) joins `programs` by its
    digest, with the bytes of the run's state among its arguments (`among` of it: None for one run before
    any state is placed); a program two blocks compile, with the fewest, as its run holds the most beside
    it there."""
    with collecting() as collected:
        yield
    for digest, executable in collected.items():
        state = among(executable)
        if digest in programs:
            earlier = programs[digest][1]
            state = earlier if state is None else state if earlier is None else min(state, earlier)
        programs[digest] = (executable, state)


def _arguments(executable: jax.stages.Compiled) -> int:
    """The bytes of the run's state among the arguments of a program over the state alone: its arguments',
    as XLA analysed them, those it reads (`compiled_memory`: a program's unread arguments are no part of
    it). A program XLA states no memory of states no need (`programs_need`), whatever this reads."""
    memory = compiled_memory(executable)
    return 0 if memory is None else memory.argument


def _weights_among(evaluated: Variables) -> Callable[[jax.stages.Compiled], int]:
    """The bytes of the run's state among an evaluation program's arguments: where its first argument is the
    weights it scores by (`evaluated`, as the state places them, `Objective._pipeline_weights`: its variables,
    or the average it keeps beside them), each leaf of them it reads (a leaf it drops holds no sharding,
    `Compiled.input_shardings`) that the state holds in a device's own memory; else none. Its batches and
    other arguments are none of the state, nor is an average the run keeps in host memory, which its
    evaluation fetches to the device beside the live variables (`fetched_state`): new bytes there, counted
    beside the state as its batches are (row 8710: an evaluation counted beside the whole state counted the
    weights it holds on the device twice, and refused a GPU it fitted; its review: a fetched average read as
    the state's)."""
    from dew.distributed.placement import placed_bytes

    signature = _signature(evaluated)

    def among(executable: jax.stages.Compiled) -> int:
        (arguments, _), (shardings, _) = executable.in_avals, executable.input_shardings
        if not arguments or _signature(arguments[0]) != signature:
            return 0
        placed = jax.tree.leaves(shardings[0], is_leaf=lambda leaf: leaf is None)
        read = [leaf for leaf, sharding in zip(jax.tree.leaves(evaluated), placed, strict=True)
                if sharding is not None]
        return placed_bytes(read, memory="device")

    return among


def _signature(tree) -> tuple:
    """`tree`'s structure and each leaf's shape and dtype: what an argument and the state's weights share
    where the one is the other."""
    return jax.tree.structure(tree), tuple((tuple(leaf.shape), leaf.dtype)
                                           for leaf in jax.tree.leaves(tree))


def _results_programs[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], start: Start, planned: Planned,
                                       plan: FitPlan | None, programs: Started) -> None:
    """The programs a run's steps run on their results and its state, and with its `fit`'s `plan` its
    evaluations' warming pass over its first state, compiled where the run's own code calls them over
    shapes (`collecting`), joining `programs` by digest (`_collected`): `started`'s last part. A window's
    allocation and a log's copies take the state alone; the programs over a step's results take none of it,
    and an evaluation's take the weights the run holds beside their batches (`_weights_among`)."""
    from dew.distributed.placement import allocated_bytes, placed_shapes
    from dew.state import apart, taken, together
    from dew.training.interval import added_tokens, book_shapes, booked, bookkeep, copied, latest, streak_of

    candidate, split = planned.candidate, planned.split
    placed = placed_shapes(start.abstract, start.shardings)
    window = split * trainer.accumulation
    frozen = apart(placed)[1]
    # A sampled step's results hold its summaries (`Committed.inspected`), which its programs read too.
    steps = [planned.executable] + ([] if planned.sampled is None else [planned.sampled.executable])
    if trainer._builds_window(placed, candidate, window):
        # Built before the window it builds is: beside the run's state less the window's fresh leaves, which
        # the steady state the need holds keeps (`Candidate.held`) and the builder's own outputs are here,
        # those it takes over from a donor aliasing its buffers (row 8710's review: charged twice).
        fresh = allocated_bytes(taken(placed.window, candidate.prepared.window)[0])
        with _collected(programs, lambda executable: _arguments(executable) + fresh):
            trainer._window_of(dataclasses.replace(placed, compute=None), candidate, window)
    with _collected(programs, lambda executable: 0):
        for step in steps:
            moving, loss, aux, finite, accepted, committed = shapes_of(step.out_info)
            # As the step runs it: its whole where it is split (`StepCompiler.runner`).
            if split > 1:
                loss, aux, finite, accepted, committed = trainer._compiler.whole_program()(
                    [(loss, aux, finite, accepted, committed)] * split)
            book = book_shapes(loss)
            counted = bookkeep(*booked(book, book, loss, finite, committed, aux or {}))
            streak_of(counted)
            if committed is not None and committed.inspected is not None:
                latest((committed.count, committed.inspected), (committed.count, committed.inspected))
            added_tokens(counted.tokens, counted.tokens)
            with _collected(programs, _arguments):
                copied(together(moving, frozen))
    if trainer._staging.stages == 1:
        # One stage's fit unstacks its returned state onto its stored layers (`Trainer.fit`, row 7322).
        moving = shapes_of(planned.executable.out_info)[0]
        with _collected(programs, _arguments):
            trainer._staging.stored(dataclasses.replace(together(moving, frozen), compute=None),
                                    consumed=True)
    if plan is not None:
        with _collected(programs, _weights_among(trainer.objective._pipeline_weights(placed, None))):
            trainer._evaluator().warmed_ahead(plan, placed, start.shardings,
                                              trainer._compiler.links(trainer.device_mesh))
