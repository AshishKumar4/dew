"""Whether a training step fits its devices, and the rungs that make it fit.

The trainer compiles a step, reads its memory from the executable and each
device's free bytes, and climbs the objective's recomputation ladder until
the step fits everywhere; these are those measurements and the ladder.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from dew.data.dataset import rows_of
from dew.nn.kernels.generation import KERNELS, device_generation
from dew.nn.protocols import Recomputing, TritonGemm
from dew.nn.sharding import mesh_axes
from dew.objectives.base import Batch, Objective, Variables
from dew.records import JSON, integers, json_value
from dew.telemetry.devices import gpu_free_bytes, xla_flag
from dew.training.distributed import PARAMETER_AXES
from dew.training.state import TrainState

_log = logging.getLogger(__name__)


def step_compiler_options(objective, tokens: float, frozen: bool) -> jax.stages.CompilerOptions | None:
    """Return the XLA options for this objective's training step on this device.

    Each option applies unless the run set that flag itself. `tokens` is the
    number of tokens one device steps (`device_tokens`), and `frozen` says
    whether the step trains next to frozen weights.

    On a GPU, dots that share an input (q, k and v; gate and up) run separately.
    XLA's dot merger would run them as one GEMM over their weights, concatenated
    again every step, which costs 4.0 ms of Qwen3-0.6B's step at 1 x 1024. A step
    next to frozen weights (a LoRA adapter's) with 128 tokens or fewer per device
    keeps the merger, as decoding does, because 32-token GEMMs ran 5-15% faster
    merged. On an RTX 4080, bf16, ms per step (docs/performance.md has every row):

        step                       tokens    merged   apart
        Qwen3-0.6B, LoRA r16       1 x 32     15.92   16.48
        Qwen3-0.6B, LoRA r16       4 x 32     18.64   18.99
        Qwen3-0.6B, LoRA r16       2 x 64     18.59   18.89
        Qwen3-0.6B, LoRA r16      1 x 128     20.79   19.18
        Qwen3-0.6B, LoRA r16       8 x 32     24.02   22.97
        Qwen3-0.6B, LoRA r16     1 x 1024     60.61   59.19
        Qwen3-0.6B widths, full    1 x 32     46.83   44.93
        Qwen3-0.6B widths, full  1 x 1024     97.7    94.1
        3-layer decoder, full      1 x 32      5.01    4.89

    At 128 tokens the shapes disagree: 1 x 128 runs faster apart, while 2 x 64
    and 4 x 32 run faster merged. The boundary includes 128, so no step runs
    slower than XLA's default, and 1 x 128 gives up the 1.6 ms it would gain apart.

    Only an LM objective gives the tokens per row, its `seq_len`, so any other
    objective's frozen step runs apart. Triton GEMM fusions are turned off where
    `KERNELS['xla_triton_gemm']` measured a win and no module the step runs
    keeps them (`TritonGemm`).
    """
    generation = device_generation()
    options: dict[str, bool | int] = {}
    small = tokens <= 128
    if (generation.startswith('sm') and xla_flag('xla_gpu_dot_merger_threshold_mb') is None
            and not (frozen and small)):
        options['xla_gpu_dot_merger_threshold_mb'] = 0
    modules = [entry.module for entry in objective.program_key()]
    if (KERNELS['xla_triton_gemm'].get(generation) == 'off' and xla_flag('xla_gpu_enable_triton_gemm') is None
            and modules and not any(isinstance(module, TritonGemm) and module.keeps_triton_gemm
                                    for module in modules)):
        options['xla_gpu_enable_triton_gemm'] = False
    return options or None


def compiled_if_it_fits(program: jax.stages.Lowered, options: jax.stages.CompilerOptions | None,
                        refusals: list[RuntimeError]) -> jax.stages.Compiled | None:
    """Return `program` compiled under `options`, or None if XLA refuses it for memory.

    A refusal is appended to `refusals`. XLA:TPU checks a program's temporaries
    against HBM while it compiles and raises RESOURCE_EXHAUSTED instead of
    returning an executable whose memory `step_fits` could read, so the refusal
    means the step does not fit. For example, on a v6e, Qwen3-0.6B at 16 x 1024
    tokens asked for 38.47G of temporaries next to 31.24G of HBM. Any other
    compile error is raised.
    """
    try:
        return program.compile(options)
    except jax.errors.JaxRuntimeError as error:
        if not str(error).startswith('RESOURCE_EXHAUSTED'):
            raise
        _log.info("XLA refused the step for memory: %s", error)
        refusals.append(error)
        return None


def fitting_default(program: jax.stages.Lowered, executable: jax.stages.Compiled | None,
                    mesh: Mesh, held: int, refusals: list[RuntimeError]
                    ) -> tuple[jax.stages.Compiled | None, bool]:
    """Fall back to the step compiled under XLA's default options if only that one fits.

    It returns the step to run and whether it fits; `held` is as in `step_fits`
    and `refusals` as in `compiled_if_it_fits`. Turning the Triton GEMM fusions
    back on can need fewer temporaries, so XLA's defaults are tried before the
    ladder's first rung.
    """
    default = compiled_if_it_fits(program, None, refusals)
    if not step_fits(default, mesh, held):
        return executable, False
    _log.warning("the step fits the devices only with XLA's default options (Triton GEMM fusions, "
                 "merged dots); compiling it with them")
    return default, True


def split_share(params: Variables) -> float:
    """The share of the bytes of `params`, placed arrays, every collection a
    frozen split keeps among them, that a parameter axis splits
    (`PARAMETER_AXES`): a mesh can name fsdp or tensor and still
    split nothing of a model whose parameters all sit below `Layout`'s
    `min_shard`, and the run's banner says how much it does."""
    total = split = 0
    for leaf in jax.tree.leaves(params):
        total += leaf.nbytes
        sharding = leaf.sharding
        if isinstance(sharding, NamedSharding) and any(
                sharding.mesh.shape[axis] > 1 for assignment in sharding.spec
                for axis in mesh_axes(assignment) if axis in PARAMETER_AXES):
            split += leaf.nbytes
    return split / total if total else 0.0


@runtime_checkable
class _TokenRows(Protocol):
    seq_len: int | None


def device_tokens[LossT, EffectsT](objective: Objective[LossT, EffectsT], batch: Batch, shards: int) -> float:
    """The tokens one device steps of `batch`, split over `shards` row shards,
    or infinity. An `Objective` declares no tokens in a row, since images and
    pairs have no context, so the trainer reads it at this boundary: LM
    objectives name theirs `seq_len`. Rows are counted only then, so another
    objective's batch, one that holds no rows among them, is never asked."""
    per_row = objective.seq_len if isinstance(objective, _TokenRows) else None
    return math.inf if per_row is None else rows_of(batch) // shards * per_row


def climb_to[LossT, EffectsT](objective: Objective[LossT, EffectsT], rung: Mapping[str, object]) -> None:
    """Move `objective` up to the head tile and remat recorded in a checkpoint's `rung` (`Trainer._rung`).

    Each setting is raised only where it is below the recorded one
    (`Recomputing.restore_recompute`). This takes the ladder `recompute_more`
    climbs in one move.
    """
    tile = rung.get('head_tile')
    if tile is not None and head_tile_of(objective) is None:
        rows, columns = integers(tile, 'rung head_tile')
        objective.tile_head((rows, columns))
    entries = objective.program_key()
    trained = [index for index, entry in enumerate(entries) if entry.trained]
    recorded = json_value(rung.get("remat"), "rung remat")
    # One trained module records its rung alone, several a list in program order.
    records = [recorded] if len(trained) == 1 else recorded if isinstance(recorded, list) else []
    rungs = dict(zip(trained, records, strict=False))
    modules = [entry.module.restore_recompute(rungs[index])
               if index in rungs and isinstance(entry.module, Recomputing) else entry.module
               for index, entry in enumerate(entries)]
    if any(module is not entry.module for module, entry in zip(modules, entries, strict=True)):
        objective.substitute(modules)


def head_tile_of[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> tuple[int, int] | None:
    """The tile the objective's vocabulary head computes its backward in
    (`ProgramModule.head_tile`), or None for a head that keeps its whole
    logits or an objective with no head."""
    return next((entry.head_tile for entry in objective.program_key() if entry.head_tile is not None), None)


def recompute_record[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> JSON:
    """What the modules the objective trains recompute in their backward
    pass, as a checkpoint records it (`Recomputing.recompute_record`): one
    module's record, a list of several in program order, None for a module
    that does not say and for an objective that trains none."""
    records = [entry.module.recompute_record() if isinstance(entry.module, Recomputing) else None
               for entry in objective.program_key() if entry.trained]
    return records[0] if len(records) == 1 else records or None


def step_headroom(executable: jax.stages.Compiled, devices: Sequence, held: int = 0) -> int | None:
    """Return the bytes the tightest of `devices` has to spare once the step's memory is placed.

    The step's temporaries and new outputs are placed next to `held` more bytes
    that the loop keeps outside the step. The result is None if the executable or
    a device reports no memory. The arguments (the state and the batch) are
    already resident and counted as in use, and the donated state's buffers are
    reused for the outputs that alias them.

    XLA's GPU step takes all its temporaries in one allocation, so it needs one
    free block that large, not that many free bytes in total (`placeable`). Where
    the allocator `strands_temporaries`, they need room twice.
    """
    stats = executable.memory_analysis()
    memory = [device.memory_stats() or {} for device in devices]
    if stats is None or not all('bytes_limit' in m and 'bytes_in_use' in m for m in memory):
        return None
    beside = stats.output_size_in_bytes - stats.alias_size_in_bytes + held
    return min(placeable(m, unclaimed(d, m)) - beside
               - stats.temp_size_in_bytes * (2 if strands_temporaries(d.platform, m) else 1)
               for d, m in zip(devices, memory, strict=True))


def unclaimed(device, memory: Mapping[str, int]) -> int | None:
    """Return the bytes free on a GPU whose pool grows (`gpu_free_bytes`), or None for any other device.

    That is the most a new region of the pool can take.
    """
    if device.platform != 'gpu' or memory.get('pool_bytes', memory['bytes_limit']) >= memory['bytes_limit']:
        return None
    return gpu_free_bytes(device.local_hardware_id)


def placeable(memory: Mapping[str, int], unclaimed: int | None = None) -> int:
    """Return the largest allocation a device can satisfy, given the `memory` its allocator reports.

    That is the most a step's temporaries, and every buffer next to them, can
    take from one block (`Device.memory_stats`).

    XLA's GPU pool, its BFC allocator, reports its size (pool_bytes) and its
    largest free block. A pool that grows (XLA_PYTHON_CLIENT_PREALLOCATE=false) can
    also take a new block from the part of its limit it has not taken yet, up to
    the GPU's `unclaimed` free bytes. cuda_async and a TPU's allocator report no
    pool, so their free bytes are all there is to read.
    """
    free = memory['bytes_limit'] - memory['bytes_in_use']
    if 'pool_bytes' not in memory:
        return free
    growth = memory['bytes_limit'] - memory['pool_bytes']
    if unclaimed is not None:
        growth = min(growth, unclaimed)
    return min(free, max(memory['largest_free_block_bytes'], growth))


def strands_temporaries(platform: str, memory: Mapping[str, int]) -> bool:
    """Whether a device's allocator can leave a step's temporaries no block to return to.

    If so, the next step needs a second block as large. `platform` and `memory`
    describe the device and what its allocator reports.

    XLA's spatially partitioned BFC pool can (a preallocated pool with
    --xla_gpu_enable_allocator_spatial_partitioning left on, its default): a free
    block below a buffer serves every small allocation before the open space past
    it, so a batch prefetched past the temporaries lets the next step's outputs
    take a few bytes of their block. `prepare_process` turns the partitioning off.
    cuda_async can do the same inside a pool it does not report.
    """
    if 'pool_bytes' not in memory:
        return platform == 'gpu'
    partitioning = xla_flag('xla_gpu_enable_allocator_spatial_partitioning') or 'true'
    return memory['pool_bytes'] >= memory['bytes_limit'] and partitioning.lower() not in ('false', '0')


def fits_everywhere(headroom: int | None) -> bool:
    """Whether every process's headroom is non-negative, with the same answer on every process.

    A process reads only its own devices' memory, so the pool takes the minimum;
    a process that reports none does not affect the result.
    """
    from dew.coordination import from_every_process

    return min((value for value in from_every_process(headroom) if value is not None), default=0) >= 0


def step_fits(executable: jax.stages.Compiled | None, mesh: Mesh, held: int = 0) -> bool:
    """Whether the compiled step fits in the free memory of every device of `mesh`.

    Each device must also hold `held` more bytes, and the processes that hold
    them agree on the answer. A step that XLA refused for memory
    (`compiled_if_it_fits`) does not fit, and its process still takes part in the
    agreement.
    """
    local = [device for device in mesh.devices.flat if device.process_index == jax.process_index()]
    return fits_everywhere(-1 if executable is None else step_headroom(executable, local, held=held))


def refuse_wide_floats(state: TrainState, mesh: Mesh) -> None:
    """Refuse parameters stored in float64 or complex128 on a TPU mesh.

    XLA's TPU backend has no 64-bit floats. It rewrites them into pairs of
    32-bit ones, which are not IEEE doubles (0.02 came back as
    0.01999999999999999 on a v6e), and the rewrite has no case for nextafter,
    which a truncated normal initializer runs. Parameters stored in float64
    ask for exactly the bits that rewrite loses, and their optimizer
    moments and EMA copy follow them. Other state that x64 makes float64,
    such as a dynamic loss scale's power of two, is exact in the pairs and
    trains as it did."""
    if mesh.devices.flat[0].platform != "tpu":
        return
    wide = [jax.tree_util.keystr(path) for path, leaf in jax.tree_util.tree_leaves_with_path(state.variables)
            if leaf.dtype in (jnp.float64, jnp.complex128)]
    if wide:
        others = f" and {len(wide) - 1} more parameters" if len(wide) > 1 else ""
        raise ValueError(
            f"params{wide[0]}{others} are stored in 64-bit floats, and a TPU has no float64: XLA "
            "rewrites it into pairs of float32, which are not IEEE doubles and have no "
            "nextafter. Store the parameters in float32 on a TPU, or train them on a CPU or a GPU.")


def recompute_more[LossT, EffectsT](objective: Objective[LossT, EffectsT]) -> bool:
    """Move the objective one rung up its ladder, and return whether there was a rung to move to.

    A head that keeps its whole logits for the backward pass moves to the
    generation's tile first (`LMObjective.head_tile`); after that, each
    module the objective trains recomputes one rung more where its own
    ladder goes on (`Recomputing.recompute_more`).
    """
    moved = objective.tile_head()
    if moved is not None:
        _log.warning(
            "the step does not fit the devices with the whole logits kept; compiling it again with %s", moved
        )
        return True
    entries = objective.program_key()
    stronger = [entry.module.recompute_more() if entry.trained and isinstance(entry.module, Recomputing)
                else None for entry in entries]
    if all(module is None for module in stronger):
        return False
    current = recompute_record(objective)
    objective.substitute([entry.module if module is None else module
                          for module, entry in zip(stronger, entries, strict=True)])
    _log.warning("the step does not fit the devices under remat %r; compiling it again under %r",
                 current, recompute_record(objective))
    return True
