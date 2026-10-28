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
A trainer over its own devices compiles ahead too, so a CPU host checks the whole round trip.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.experimental.topologies as topologies
from jax.experimental import serialize_executable

from dew.batch import Batch
from dew.nn.kernels.generation import capability, gpu_target
from dew.telemetry.instrumentation import StepFlops, step_flops

if TYPE_CHECKING:
    from dew.data.dataset import Loader
    from dew.training.trainer import Trainer

_log = logging.getLogger(__name__)

THREADS_PER_MULTIPROCESSOR = {(8, 0): 2048, (9, 0): 2048, (10, 0): 2048}
"""Resident threads a multiprocessor holds by compute capability, 1536 on the others the targets name
(CUDA C++ Programming Guide, "Technical Specifications per Compute Capability")."""


def target_config(generation: str) -> str:
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
def compile_only_gpus(generation: str, count: int = 1) -> tuple:
    """`count` compile-only devices of the CUDA `generation`, none attached: XLA's CUDA backend compiles a
    program over them, a mesh's collectives included, as for that many of the generation's GPUs."""
    topology = topologies.get_topology_desc("dew-compile", "cuda", target_config=target_config(generation),
                                            topology=f"1x1x{count}")
    return tuple(topology.devices[:count])


@dataclasses.dataclass(frozen=True)
class AheadOfTime:
    """A training step compiled for `devices` GPUs of `generation`, or for `devices` of the platform its
    trainer's mesh is built over where `generation` is None (`compile_ahead`): the executable,
    serialized, the program it was compiled from (`program`, `program_of`), and what it costs.
    `flops` are its matmuls' and convolutions' on one device, read off its optimized program as a
    run's are (`dew.telemetry.instrumentation.step_flops`, which
    counts the cuBLAS and cuDNN calls XLA's own cost analysis cannot see into), None where it runs a
    loop of unstated length; the byte counts are XLA's memory analysis of one device: the arguments
    it reads (the state and the batch's share), the outputs it writes, those that alias an argument
    (the donated state), its temporaries and its code."""

    generation: str | None
    devices: int
    mesh: dict[str, int]
    platform: str
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
        `generation`, or `devices` of its platform: called with the state and batch the step was
        compiled for, it runs the step."""
        found = jax.devices()
        here = {device.platform if self.generation is None else gpu_target(device.device_kind).generation
                for device in found}
        wanted = self.platform if self.generation is None else self.generation
        if len(found) != self.devices or here != {wanted}:
            raise ValueError(f"this step was compiled for {self.devices} {wanted} devices; this process "
                             f"has {len(found)} of {sorted(here)}")
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
    which this host need not have, written to `path` (`AheadOfTime.save`)."""

    generation: str
    devices: int
    path: str

    def compiled[LossT, EffectsT](self, trainer: Trainer[LossT, EffectsT], loader: Loader,
                                  steps: int | None) -> CompiledAhead:
        """`trainer`, built over this target's devices (`compile_only_gpus`), its first step over
        `loader`'s first batch compiled ahead (`compile_ahead`) and saved, as what ends the run."""
        from dew.training._fit import first_batch

        trainer.validate(steps=steps)
        ahead = compile_ahead(trainer, first_batch(loader, trainer.device_mesh))
        ahead.save(self.path)
        return CompiledAhead(self.path, ahead)


class CompiledAhead(SystemExit):
    """Raised by a configured run under `compile_ahead` once its first step is compiled and saved
    (`AheadTarget.compiled`). Compiling was the whole run, so uncaught it ends the program with status 0
    and no traceback, the step's cost logged; `ahead` is the step and `path` where it was saved."""

    def __init__(self, path: str, ahead: AheadOfTime):
        super().__init__(0)
        self.path, self.ahead = path, ahead
        _log.info("the step compiled ahead for %s %s devices holds %s bytes a device at its peak; saved "
                  "to %s", ahead.devices, ahead.generation or ahead.platform, ahead.peak_bytes, path)


def step_batch[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch) -> Batch:
    """`batch`, a global batch as a loader gives it, as `fit` hands it to its step: its work counted
    (`Trainer._counted_batch`), each row marked with its place among the step's records
    (`dew.training.distributed.marked`), and its host fields set apart (`split_fields`)."""
    from dew.data.dataset import DataPartition
    from dew.distributed.batches import split_fields
    from dew.training.distributed import marked

    device, _ = split_fields(marked(DataPartition.of(trainer.device_mesh), trainer._counted_batch)(batch))
    return device


def compile_ahead[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], batch: Batch) -> AheadOfTime:
    """`trainer`'s first step over a batch of `batch`'s shapes, compiled for the devices its mesh is
    built over and never run: the step `fit` would compile there, as its plan's first candidate lowers
    it (`Trainer._candidate`), at the base rung and the first split, its state the fresh one the
    trainer builds, abstract and placed by its layout, its batch `batch` as `fit` hands it to the step
    (`step_batch`), compiled with the options the run's compile takes. Over compile-only GPUs of one
    generation (`compile_only_gpus`) it traces as on those GPUs (`dew.nn.kernels.generation.assumed`);
    over the trainer's own devices, as they are."""
    devices = list(trainer.device_mesh.devices.flat)
    platforms = {device.platform for device in devices}
    generations = ({gpu_target(device.device_kind).generation for device in devices} if platforms == {"gpu"}
                   else {None})
    if len(platforms) != 1 or len(generations) != 1:
        raise ValueError("compile_ahead compiles for one platform's devices, one generation of GPUs "
                         "(MeshSpec.build(compile_only_gpus(generation, count))); this mesh holds "
                         f"{sorted({device.device_kind for device in devices})}")
    (generation,), (platform,) = generations, platforms
    start = trainer._started()
    lowered, options = trainer._lowered_for(start.abstract, step_batch(trainer, batch), generation,
                                            trainer.device_mesh)
    compiled = lowered.compile(options)
    executable, in_tree, out_tree = serialize_executable.serialize(compiled)
    memory = compiled.memory_analysis()
    if memory is None:
        raise ValueError(f"XLA states no memory of the step it compiled for {generation or platform}")
    return AheadOfTime(
        generation, len(devices), dict(trainer.device_mesh.shape), platform, program_of(lowered),
        executable, (in_tree, out_tree),
        flops=step_flops(compiled),
        argument_bytes=memory.argument_size_in_bytes, output_bytes=memory.output_size_in_bytes,
        alias_bytes=memory.alias_size_in_bytes, temp_bytes=memory.temp_size_in_bytes,
        code_bytes=memory.generated_code_size_in_bytes)
