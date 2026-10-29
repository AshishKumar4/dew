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
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.experimental.topologies as topologies
import numpy as np
from jax.experimental import serialize_executable

from dew._records import JSON, record
from dew.batch import Batch
from dew.nn.kernels.generation import assumed, capability, gpu_target
from dew.nn.kernels.templates import report
from dew.telemetry.instrumentation import StepFlops, step_flops
from dew.training.profiling import pgle_runs

if TYPE_CHECKING:
    from dew.data.dataset import Loader
    from dew.training.trainer import Trainer

_log = logging.getLogger(__name__)

THREADS_PER_MULTIPROCESSOR = {(8, 0): 2048, (9, 0): 2048, (10, 0): 2048}
"""Resident threads a multiprocessor holds by compute capability, 1536 on the others the targets name
(CUDA C++ Programming Guide, "Technical Specifications per Compute Capability")."""


def target_config(generation: str, capacity: int | None = None) -> str:
    """The `gpu_device_info` XLA compiles for, from the CUDA `generation`'s target
    (`generation.TARGETS`): its compute capability, the shared memory a block may opt into, its
    multiprocessors, clock and bandwidth; CUDA's per-capability limits for the rest. The fields past
    the capability and shared memory only steer XLA's own fusions, never a Triton kernel's code."""
    target, version = gpu_target(generation), capability(generation)
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
  threads_per_core_limit: {THREADS_PER_MULTIPROCESSOR.get((major, minor), 1536)}
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
  }}
  registers_per_core_limit: 65536
  registers_per_block_limit: 65536
}}
platform_name: "CUDA"
device_description_str: "{target.device_kind}"
"""


@functools.cache
def compile_only_gpus(generation: str, count: int = 1, capacity: int | None = None) -> tuple:
    """`count` compile-only devices of the CUDA `generation`, none attached: XLA's CUDA backend compiles a
    program over them, a mesh's collectives included, as for that many of the generation's GPUs.
    `capacity` states bytes per device to XLA's scheduler and rematerialization pass, not only Dew's
    admission check; absent it, XLA's target states zero rather than a hardware capacity."""
    config = target_config(generation, capacity)
    topology = topologies.get_topology_desc("dew-compile", "cuda", target_config=config,
                                            topology=f"1x1x{count}")
    return tuple(topology.devices[:count])


@dataclasses.dataclass(frozen=True)
class AheadOfTime:
    """A training step compiled for `devices` GPUs of `generation` (`compile_ahead`): the executable,
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
    executable: bytes
    trees: tuple
    """The trees of the executable's inputs and outputs, as `serialize_executable.serialize` gives them."""
    flops: StepFlops | None
    argument_bytes: int
    output_bytes: int
    alias_bytes: int
    temp_bytes: int
    code_bytes: int
    links: dict[str, float | None]
    """The bandwidth of each split axis's link the step decided its projections by (`Trainer.links`):
    on compile-only GPUs none measured (None), so none spreads. A run handed the step traces with these
    in place of measuring its own (`Trainer`'s `ahead`), which could spread a projection the step does
    not and lower another program, and its checkpoints record them (`link_figures`)."""
    plan: JSON = None
    """The plan the run's own fit search makes on those
    GPUs (`Trainer.plan`): every program it compiled, each with the footprint derived for it beside
    XLA's memory of it, and the plan it took."""
    planned_seconds: float | None = None
    """The seconds that search took, its derivations and every compile: the time a run spends planning
    before its first step on those GPUs."""
    capacity: int | None = None
    """The stated bytes per target GPU, or None where no memory capacity was supplied."""
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

    def save(self, path: str | Path) -> None:
        """Write the step to `path`, for `load` on a host of its topology."""
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> AheadOfTime:
        """The step `save` wrote to `path`."""
        loaded = pickle.loads(Path(path).read_bytes())
        if not isinstance(loaded, AheadOfTime):
            raise ValueError(f"{path} holds no step compiled ahead of time")
        return loaded

    def loaded(self) -> jax.stages.Compiled:
        """The executable, loaded onto this process's devices, which must be `devices` GPUs of
        `generation`: called with the state and batch the step was compiled for, it runs the step."""
        found = jax.devices()
        generations = {gpu_target(device.device_kind).generation for device in found}
        if len(found) != self.devices or generations != {self.generation}:
            raise ValueError(f"this step was compiled for {self.devices} {self.generation} GPUs; this "
                             f"process has {len(found)} of {sorted(generations)}")
        in_tree, out_tree = self.trees
        return serialize_executable.deserialize_and_load(self.executable, in_tree, out_tree,
                                                         backend=found[0].client)


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


def program_of(lowered: jax.stages.Lowered) -> str:
    """A step's program as `lowered` holds it, its text's digest: two lowerings of one step agree."""
    return hashlib.sha256(lowered.as_text().encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class AheadTarget:
    """Where a configured run compiles its first step ahead and saves it, then exits, training nothing
    (`TrainerConfig.compile_ahead`): for `devices` GPUs of the CUDA `generation` (`compile_only_gpus`),
    which this host need not have, written to `path` (`AheadOfTime.save`). Where `capacity_gib` states
    each GPU's memory, the run's own plan there is made and reported with it (`compile_ahead`): the
    offline answer to what fits, a run with its remat set told of no rung but its own."""

    generation: str
    devices: int
    path: str
    capacity_gib: float | None = None

    @property
    def capacity(self) -> int | None:
        """The stated bytes per GPU, read both when building its topology and when planning."""
        return None if self.capacity_gib is None else int(self.capacity_gib * 2**30)

    def compiled[LossT, EffectsT](self, trainer: Trainer[LossT, EffectsT], loader: Loader,
                                  steps: int | None) -> CompiledAhead:
        """`trainer`, built over this target's devices (`compile_only_gpus`), its first step over
        `loader`'s first batch, through the loader's device steps as `fit` reads it
        (`Trainer.reading`), compiled ahead (`compile_ahead`) and saved, as what ends the run. A run
        whose step reads a rollout's batches, which it makes as it trains, is refused."""
        from dew.training._fit import first_batch

        if trainer.rollout is not None:
            raise ValueError("compile_ahead lowers the step over the loader's batches; this run's step reads "
                             "its rollout's, which it makes only as it trains")
        trainer.reading(loader)
        trainer.validate(steps=steps)
        ahead = compile_ahead(trainer, first_batch(loader, trainer.device_mesh), self.capacity)
        ahead.save(self.path)
        return CompiledAhead(self.path, ahead)


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


def step_batch[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch) -> Batch:
    """`batch`, a global batch as a loader gives it, as `fit` hands it to its step: its work counted
    (`Trainer._counted_batch`), each row marked with its place among the step's records
    (`dew.training.distributed.marked`), and its host fields set apart (`split_fields`)."""
    from dew.data.dataset import DataPartition
    from dew.distributed.batches import split_fields
    from dew.training.distributed import marked

    device, _ = split_fields(marked(DataPartition.of(trainer.device_mesh), trainer._counted_batch)(batch))
    return device


def compile_ahead[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch,
                                   capacity: int | None = None) -> AheadOfTime:
    """`trainer`'s first step over a batch of `batch`'s shapes, compiled for the compile-only GPUs its
    mesh is built over (`compile_only_gpus`) and never run. The planner `Trainer.compile` uses
    (`Trainer._compiled_plan`) selects and compiles the step once; that executable is serialized, not compiled
    again. Its batch is what `fit` hands to its step (`step_batch`), its fresh state abstract, and
    tracing reads the target's kernel choices (`assumed`). The plan reports every compile and the
    time the search took. `capacity` states bytes per GPU; absent it, memory fit is unknown rather
    than measured. Profiling, tuning and rollout require execution and are refused."""
    if capacity is not None and capacity <= 0:
        raise ValueError("compile-ahead capacity must be positive bytes per GPU")
    if trainer.host_master or trainer.rollout is not None or trainer.lower_for is not None:
        raise ValueError("compile-ahead needs a device-resident step over its loader, with no lower_for")
    if trainer.tune_kernels or pgle_runs():
        raise ValueError("compile-ahead cannot run kernel tuning or AutoPGLE on absent GPUs")
    devices = list(trainer.device_mesh.devices.flat)
    generations = {gpu_target(device.device_kind).generation for device in devices}
    if {device.platform for device in devices} != {"gpu"} or len(generations) != 1:
        raise ValueError("compile_ahead compiles for the compile-only GPUs of one generation a trainer's "
                         "mesh is built over (MeshSpec.build(compile_only_gpus(generation, count))); this "
                         f"mesh holds {sorted({device.device_kind for device in devices})}")
    (generation,) = generations
    if capacity is not None:
        mesh = trainer.device_mesh
        devices = compile_only_gpus(generation, len(devices), capacity)
        trainer.mesh = trainer.device_mesh = trainer.state_mesh = jax.sharding.Mesh(
            np.asarray(devices).reshape(mesh.devices.shape), mesh.axis_names, axis_types=mesh.axis_types)
    began = time.perf_counter()
    start = trainer._started()
    with assumed(generation):
        planned = trainer._compiled_plan(start.abstract, step_batch(trainer, batch), None, capacity)
    seconds = time.perf_counter() - began
    lowered, compiled = planned.candidate.lowered, planned.executable
    executable, in_tree, out_tree = serialize_executable.serialize(compiled)
    memory = compiled.memory_analysis()
    if memory is None:
        raise ValueError(f"XLA states no memory of the step it compiled for {generation}")
    return AheadOfTime(
        generation, len(devices), dict(trainer.device_mesh.shape), program_of(lowered), executable,
        (in_tree, out_tree),
        flops=step_flops(compiled),
        argument_bytes=memory.argument_size_in_bytes, output_bytes=memory.output_size_in_bytes,
        alias_bytes=memory.alias_size_in_bytes, temp_bytes=memory.temp_size_in_bytes,
        code_bytes=memory.generated_code_size_in_bytes, plan=trainer.plan, planned_seconds=seconds,
        capacity=capacity, links={axis: link.bytes_per_second for axis, link in trainer.links.items()},
        refusals=report.refused_launches())
