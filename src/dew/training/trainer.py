"""The trainer: mesh, compiled step, EMA, checkpoints, logging.

What is learned is the objective's business (`dew.objectives.base`). The
trainer materialises the objective's tree on the mesh, compiles one step over
the global batch, and keeps the EMA copy on the optimizer's clock. Effects go
to the capabilities it was given: a `Checkpoints` for disk, a `Tracker` for
numbers and artifacts.

Constructing one opens nothing; the mesh, the compiled step and the
capabilities' resources come into being in `fit`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import functools
import logging
import math
import sys
import time
import types
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from typing_extensions import TypeVar as DefaultTypeVar

from dew.artifacts import agree_process_phase, agreed
from dew.checkpoints import Checkpoints, Ranking
from dew.data.dataset import Checkpointable, Closeable, RampedStream, Reader, rows_of
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES, RematPolicy
from dew.nn.kernels.generation import device_generation
from dew.nn.sharding import (
    BATCH_AXES,
    SEQUENCE_AXIS,
    STAGE_AXIS,
    TENSOR_AXIS,
    LayoutRefused,
    Link,
    Schedule,
    measured_links,
    mesh_axes,
    pipeline_microbatches,
)
from dew.objectives.base import (
    FROZEN,
    Aux,
    Batch,
    Initializer,
    Metric,
    Objective,
    Ratio,
    Shown,
    Step,
    TrainingScalar,
    Variables,
    select,
)
from dew.records import JSON, boolean, integers, json_value, record
from dew.telemetry import profile as telemetry_profile
from dew.telemetry.devices import TRITON_GEMM_OFF_GENERATIONS, gpu_free_bytes, xla_flag
from dew.telemetry.instrumentation import compiled_flops, model_flops_utilization, peak_flops
from dew.telemetry.profile import region
from dew.telemetry.records import (
    AxisLink,
    CheckpointRequested,
    FitEnded,
    FitStarted,
    ProfileWindow as ProfileWindowRecord,
    Record,
    StepCompiled,
)
from dew.training.display import TrainingDisplay
from dew.training.distributed import (
    PARAMETER_AXES,
    PREFETCH_DEPTH,
    DevicePrefetchIterator,
    Layout,
    MeshSpec,
    Placement,
    batch_divisor,
    batch_shardings,
    build_mesh,
    data_partition,
    link_bandwidth,
    shard_batch,
)
from dew.training.evaluation import Evaluation, evaluate
from dew.training.runtime import Preempted, PreemptionNotice
from dew.training.selection import Best
from dew.training.state import Accumulation, TrainState
from dew.training.tracker import Tracker
from dew.training.transaction import Transaction, compact_qk, with_ema

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dew.config import TrainerConfig
    from dew.data import Dataset
    from dew.telemetry.profile import Profiler

# Consecutive non-finite losses that stop a run.
BAD_LOSS_STEPS = 5

StepFn = Callable[[TrainState, Batch], tuple[TrainState, jax.Array, Aux]]
"""A compiled step's body: the state and the global batch in, the new state,
the loss and the objective's report out."""

CompiledStep = Callable[
    [TrainState, Batch],
    tuple[TrainState, jax.Array, dict[str, jax.Array], jax.Array, jax.Array]]
"""A call returns the stepped state, the scalar loss, the objective's whole
report (`Aux.metrics`), whether that loss was finite, and whether the
microbatch was accepted."""

Loss = DefaultTypeVar("Loss", default=Ratio | jax.Array | float)
Effects = DefaultTypeVar("Effects", default=None)

ObjectiveLoss = TypeVar("ObjectiveLoss")
ObjectiveEffects = TypeVar("ObjectiveEffects")
"""The two parameters of the objective `Trainer.from_config` is handed.

A classmethod cannot solve the class's own `Loss` and `Effects` from an
argument, since an unparameterized `Trainer.from_config` binds them to their
defaults. So the factory carries its own pair and names the class it builds."""

Shapes = tuple[tuple[int, ...], ...]
"""A batch's leaf shapes in tree order, the key a compiled step is held
under. A fixed batch has one; a ramped batch has one per stage of its ramp,
each compiled on the first step that reads it."""


def batch_shapes(batch: Batch) -> Shapes:
    """List a batch's leaf shapes in tree order."""
    return tuple(np.shape(leaf) for leaf in jax.tree.leaves(batch))


class Rollout(Protocol):
    """Produces a batch on the host, before the compiled step reads it.

    Sampling is effectful and untraceable, so it lives outside `jit`. The
    trainer calls the rollout with the state, the prefetched batch and a key
    folded from the run key and the step, then reshards what comes back with
    `shard_batch`. The returned batch must hold arrays in fixed shapes, so
    the step still compiles once per run.

    A rollout may hold `metrics`, a mapping of names to floats describing
    its latest call (rewards, lag, truncation); each logging interval sends
    them to the tracker and the display as `rollout/<name>`. It may declare
    how the display shows them in `shown`, as an objective does."""

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...


@dataclasses.dataclass(frozen=True)
class ProfileWindow:
    """Asks for one profiler window per fit: `steps` steps traced into
    `directory` after `warmup` steps have run, so the trace holds the loop
    and not the compile.

    `dew.profile` is the other way to capture one, a context manager around
    any code at all; a fit refuses to schedule a window inside one. The
    window the loop wrote is reported as the `ProfileWindow` record of
    `dew.telemetry.records`."""
    directory: str
    steps: int
    warmup: int = 2


Book = tuple[jax.Array, jax.Array, jax.Array]
"""The loop's device-side counters: the interval's summed loss, the current
streak of non-finite losses, and the longest streak since the last check."""


def fresh_book() -> Book:
    """Return the counters a fresh logging interval starts from."""
    return jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32)


@jax.jit
def bookkeep(book: Book, loss: jax.Array, finite: jax.Array) -> Book:
    """Advance the loop's counters for one step, in one dispatch.

    The same work as five eager ops, the cast, the add, the where, the add
    and the maximum, each dispatching an executable of its own. Those cost
    176 us a step on an i9-12900K against 37 us for this one call, measured
    over 2000 steps on the CPU backend with the result blocked on at the end.
    """
    interval_loss, bad_run, worst_bad_run = book
    bad_run = jnp.where(finite, 0, bad_run + 1)
    dtype = jnp.promote_types(loss.dtype, jnp.float32)
    return interval_loss + loss.astype(dtype), bad_run, jnp.maximum(worst_bad_run, bad_run)


def learning_rate(opt_state: optax.OptState) -> float | None:
    """The learning rate in `opt_state`, where the optimizer carries one:
    under `optax.inject_hyperparams`, the way optax exposes a schedule's
    value. None for a rate the optimizer closes over, or for parameter
    groups that carry several."""
    injected = (optax.InjectHyperparamsState, optax.InjectStatefulHyperparamsState)
    rates = [node.hyperparams["learning_rate"]
             for node in jax.tree.leaves(opt_state, is_leaf=lambda node: isinstance(node, injected))
             if isinstance(node, injected) and "learning_rate" in node.hyperparams]
    # optax types a hyperparameter as any ArrayLike; a learning rate is a real scalar.
    return float(jnp.asarray(rates[0])) if len(rates) == 1 else None


# How the display shows the metrics the trainer itself logs; an objective,
# a rollout and a validation metric declare their own (`Shown`).
TRAINER_SHOWN = {"loss": Shown(better="lower"),
                 "learning_rate": Shown(group="optimizer"), "loss_scale": Shown(group="optimizer"),
                 "accepted": Shown(percent=True, group="optimizer"),
                 "step_time_ms": Shown(better="lower", group="throughput"),
                 "samples_per_sec": Shown(better="higher", group="throughput"),
                 "mfu": Shown(better="higher", percent=True, group="throughput"),
                 "rollout_seconds": Shown(better="lower", group="throughput")}


def goodput(wall: float, first_step: float | None, other: float) -> dict[str, float]:
    """Compute the two goodput numbers from MaxText's report that need no
    cluster telemetry.

    `first_step` is the time from the start of `fit` to the first step's
    result: the placement or restore, the first batch, the compile and the
    step itself. It is None when no step ran. `other` is the time spent
    outside steps after that: evaluations, checkpoint writes and the wait for
    them at the end.

    The step fraction is what is left of `wall`. That counts a step's own
    data stall as step time, as MaxText's start-to-start step time does.
    """
    numbers = {}
    if first_step is not None:
        numbers["goodput/time_to_first_step_s"] = first_step
    steps = wall - (wall if first_step is None else first_step) - other
    numbers["goodput/step_fraction"] = max(steps, 0.0) / wall if wall > 0 else 0.0
    return numbers


def step_compiler_options(objective) -> jax.stages.CompilerOptions | None:
    """XLA options for this objective's training step on this device: Triton
    GEMM fusions off where `TRITON_GEMM_OFF_GENERATIONS` measured a win and
    no mixer of the model keeps them, unless the run set the flag itself."""
    if (device_generation() not in TRITON_GEMM_OFF_GENERATIONS
            or xla_flag('xla_gpu_enable_triton_gemm') is not None):
        return None
    model = _model_of(objective)
    if model is None or _keeps_triton_gemm(model):
        return None
    return {'xla_gpu_enable_triton_gemm': False}


def fitting_default(program: jax.stages.Lowered, executable: jax.stages.Compiled,
                    mesh: Mesh, held: int = 0) -> tuple[jax.stages.Compiled, bool]:
    """Fall back to the step compiled under XLA's default options where it
    fits and the one compiled under `step_compiler_options` does not.
    Returns the step to run and whether it fits; `held` is `step_fits`'s.

    The Triton GEMM fusions can hold fewer temporaries: on an RTX 4080
    (sm89, jax 0.11.2), Qwen3-0.6B's widths at 2 layers and 8 x 1024 tokens
    keep their whole logits in 13.1 GiB with them, inside the 13.24 GiB of an
    0.85 pool, while without them the step does not fit and tiles its head.
    So the fusions come back before the ladder's first rung."""
    default = program.compile()
    if not step_fits(default, mesh, held):
        return executable, False
    _log.warning("the step fits the devices only with XLA's Triton GEMM fusions; compiling it with them")
    return default, True


def _split_share(params: Variables) -> float:
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


def _model_of(objective: Objective[Loss, Effects]) -> nn.Module | None:
    """The one module an objective trains, or None. `Objective` declares no
    model, since some train none (JEPA's pair, the RL actors' wrappers), so
    the trainer reads it at this boundary: LM, diffusion and masked
    objectives name theirs `model`."""
    return getattr(objective, 'model', None)


def _rollout_metrics(rollout: Rollout) -> Mapping[str, float]:
    """The rollout's report of its latest call. A rollout is any callable,
    and a plain function reports nothing, so the trainer reads `metrics` at
    this boundary."""
    return getattr(rollout, 'metrics', {})


def _reported(rollout: Rollout | None, metrics: Sequence[Metric]) -> dict[str, Shown]:
    """How the display shows what the rollout and the validation metrics
    report. Both may declare `shown`, which their protocols leave optional,
    so the trainer reads it at this boundary."""
    return {
        **getattr(rollout, "shown", {}),
        **{metric.name: shown for metric in metrics if (shown := getattr(metric, "shown", None)) is not None},
    }


def _keeps_triton_gemm(model: nn.Module) -> bool:
    """Whether a mixer of the model keeps XLA's Triton GEMM fusions
    (`MixerBase.keeps_triton_gemm`). Any flax module arrives here, so its
    mixer fields are read at this boundary: a decoder names its mixer and its
    layer kinds', and a model without them keeps no mixer."""
    mixers = [getattr(model, 'mixer', None)]
    mixers += [kind.mixer for kind in (getattr(model, 'kinds', None) or {}).values()]
    return any(getattr(mixer, 'keeps_triton_gemm', False) for mixer in mixers)


def climb_to(objective, rung: Mapping[str, object]) -> None:
    """Move `objective` up to the head tile and remat a checkpoint's
    `rung` records (`Trainer._rung`), each where it stands below them: the
    ladder `recompute_more` climbs, taken in one move."""
    tile = rung.get('head_tile')
    if tile is not None and _head_tile_of(objective) is None:
        rows, columns = integers(tile, 'rung head_tile')
        objective.tile_head((rows, columns))
    model = _model_of(objective)
    current = _remat_of(model)
    ladder = (DECODER_REMAT if isinstance(model, CausalTransformer)
              else DIFFUSION_REMAT if isinstance(current, bool | str) else ())
    records = [remat_record(remat) for remat in ladder]
    here, there = (
        remat_record("dots" if current is True else current),
        json_value(rung.get("remat"), "rung remat"),
    )
    if (
        model is not None
        and here in records
        and there in records
        and records.index(there) > records.index(here)
    ):
        objective.model = model.clone(remat=ladder[records.index(there)])


def _head_tile_of(objective: Objective[Loss, Effects]) -> tuple[int, int] | None:
    """The tile an objective's head computes its backward in, read at the
    boundary with any objective: an LM objective's `head_tile`, and None for
    a head that keeps its whole logits or an objective with no head."""
    return getattr(objective, 'head_tile', None)


def _remat_of(model: nn.Module | None) -> RematPolicy | bool | str | None:
    """The model's remat setting, read at the boundary with any flax module:
    a decoder's `RematPolicy`, a diffusion backbone's bool or name, and
    None for a model with no remat field."""
    return getattr(model, 'remat', None)


# What a model recomputes in its backward pass when its step does not fit,
# weakest first. Each rung is slower and holds less: on an NVIDIA L4 (24 GB,
# jax 0.11.2, bf16) the 359.8M-parameter decoder at 4 x 1024 tokens took
# 334.7, 356.4 and 401.3 ms at 9.93, 8.00 and 6.29 GiB, and at 16 x 1024
# only 'full' fits; DiT-L/2 on 64x64 inputs at 16 fits only under 'full'
# (docs/performance.md). A model's own remat is where it starts: the
# trainer moves it up one rung at a time until the compiled step fits the
# devices, and never past a policy the ladder does not name.
DECODER_REMAT = (None, REMAT_POLICIES['minimal'], REMAT_POLICIES['full'])
DIFFUSION_REMAT = (False, 'dots', 'full')


def step_headroom(executable: jax.stages.Compiled, devices: Sequence, held: int = 0) -> int | None:
    """The bytes the tightest of `devices` has to spare once the compiled
    step's temporaries and new outputs are placed beside `held` more bytes
    the loop keeps outside the step, None where the executable or a device
    reports no memory. The arguments, the state and the batch, are already
    resident and counted in use; the donated state's buffers are reused for
    the outputs that alias them.

    XLA's GPU step takes all its temporaries in one allocation, so it needs
    one free block that large, not that many free bytes: on an A100 the
    'minimal' rung of a Qwen3-1.7B fine-tune ran out of memory placing its
    11.68 GiB of temporaries with 16.3 GiB free, split 5.9 GiB below the
    state and 10.4 GiB above it. `placeable` reads the block from the
    allocator. Where it `strands_temporaries`, they need room twice."""
    stats = executable.memory_analysis()
    memory = [device.memory_stats() or {} for device in devices]
    if stats is None or not all('bytes_limit' in m and 'bytes_in_use' in m for m in memory):
        return None
    beside = stats.output_size_in_bytes - stats.alias_size_in_bytes + held
    return min(placeable(m, unclaimed(d, m)) - beside
               - stats.temp_size_in_bytes * (2 if strands_temporaries(d.platform, m) else 1)
               for d, m in zip(devices, memory, strict=True))


def unclaimed(device, memory: Mapping[str, int]) -> int | None:
    """The bytes free on a GPU whose pool grows (`gpu_free_bytes`), the
    most a new region of it can take; None for any other device."""
    if device.platform != 'gpu' or memory.get('pool_bytes', memory['bytes_limit']) >= memory['bytes_limit']:
        return None
    return gpu_free_bytes(device.local_hardware_id)


def placeable(memory: Mapping[str, int], unclaimed: int | None = None) -> int:
    """The bytes one allocation can take from a device whose allocator
    reports `memory` (`Device.memory_stats`), the most a step's temporaries
    and every buffer beside them can need of one block.

    XLA's GPU pool, its BFC allocator, reports its size (pool_bytes) and its
    largest free block; a pool that grows (XLA_PYTHON_CLIENT_PREALLOCATE=false)
    can also take a new block from the part of its limit it has not taken,
    as far as the GPU has `unclaimed` bytes free. cuda_async and a TPU's
    allocator report no pool, and their free bytes are all there is to read."""
    free = memory['bytes_limit'] - memory['bytes_in_use']
    if 'pool_bytes' not in memory:
        return free
    growth = memory['bytes_limit'] - memory['pool_bytes']
    if unclaimed is not None:
        growth = min(growth, unclaimed)
    return min(free, max(memory['largest_free_block_bytes'], growth))


def strands_temporaries(platform: str, memory: Mapping[str, int]) -> bool:
    """Whether the allocator of a device of `platform`, which reports
    `memory`, can leave a step's temporaries no block to return to, so that
    the next step needs a second block as large.

    XLA's spatially partitioned BFC pool can: a preallocated pool with
    --xla_gpu_enable_allocator_spatial_partitioning left on, its default.
    There a free block below a buffer serves every small allocation before
    the open space past it. A batch prefetched while the temporaries are
    placed lands past them; once they are freed the next step's outputs take
    a few bytes of their block. On an RTX 4080 a step with 6.5 GiB of
    temporaries and 3.9 GiB more of its pool to spare failed so in 5 of 16
    runs. With the partitioning off, which `prepare_process` sets, the
    smallest block that fits serves them instead: 4 of 4 runs placed a batch
    past the temporaries 20 to 45 times each and finished.

    cuda_async can too, and reports neither its blocks nor its pool: the
    8192-token step there, 10.3 GiB of temporaries with 10.45 GiB free,
    failed in 1 of 8 runs at a 0.85 pool, 1 of 8 at 0.87 and 1 of 16 at 0.91.
    At the failure its pool held 14.2 GB with 3.0 GB in use, the device had
    1.9 GB free, and neither gave the 11.1 GB the step asked for again."""
    if 'pool_bytes' not in memory:
        return platform == 'gpu'
    partitioning = xla_flag('xla_gpu_enable_allocator_spatial_partitioning') or 'true'
    return memory['pool_bytes'] >= memory['bytes_limit'] and partitioning.lower() not in ('false', '0')


def fits_everywhere(headroom: int | None) -> bool:
    """Whether every process's headroom is non-negative, the same answer on
    every process. A process reads only its own devices' memory, so the pool
    takes the minimum; one that reports none decides nothing."""
    if jax.process_count() == 1:
        return headroom is None or headroom >= 0
    # float32 because a pool without x64 gathers no wider type; its sign is
    # exact, which is all the answer reads.
    gathered = multihost_utils.process_allgather(
        np.asarray(np.inf if headroom is None else headroom, np.float32))
    return bool(np.min(gathered) >= 0)


def step_fits(executable: jax.stages.Compiled, mesh: Mesh, held: int = 0) -> bool:
    """Whether the compiled step fits the free memory of every device of
    `mesh` beside `held` bytes more on each, agreed across the processes that
    hold them."""
    local = [device for device in mesh.devices.flat if device.process_index == jax.process_index()]
    return fits_everywhere(step_headroom(executable, local, held=held))


def prefetched_bytes(batch: Batch, shardings: Placement[Batch]) -> int:
    """The bytes a device holds of the batches `fit` places while a step
    runs, beside the one the step reads: the `PREFETCH_DEPTH` it queues and
    the one it is placing, each laid out as `shardings` places `batch`."""
    shares = jax.tree.map(lambda leaf, sharding: math.prod(sharding.shard_shape(np.shape(leaf)))
                          * np.dtype(leaf.dtype).itemsize, batch, shardings)
    return (PREFETCH_DEPTH + 1) * sum(jax.tree.leaves(shares))


def remat_record(remat: RematPolicy | bool | str | None) -> JSON:
    """A model's remat as a record reads it: a policy by its name in
    `REMAT_POLICIES` where it has one, else its two lists."""
    if isinstance(remat, RematPolicy):
        names = [name for name, policy in REMAT_POLICIES.items() if policy == remat]
        return names[0] if names else {'save': list(remat.save), 'offload': list(remat.offload)}
    return remat


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
    wide = [jax.tree_util.keystr(path) for path, leaf in jax.tree_util.tree_leaves_with_path(state.params)
            if leaf.dtype in (jnp.float64, jnp.complex128)]
    if wide:
        others = f" and {len(wide) - 1} more parameters" if len(wide) > 1 else ""
        raise ValueError(
            f"params{wide[0]}{others} are stored in 64-bit floats, and a TPU has no float64: XLA "
            "rewrites it into pairs of float32, which are not IEEE doubles and have no "
            "nextafter. Store the parameters in float32 on a TPU, or train them on a CPU or a GPU.")


def recompute_more(objective) -> bool:
    """Move the objective one rung up its ladder, and say whether there was a
    rung to move to. A head that keeps its whole logits for the backward
    moves to the generation's tile first (`LMObjective.head_tile`); then the
    model's remat climbs."""
    moved = objective.tile_head()
    if moved is not None:
        _log.warning(
            "the step does not fit the devices with the whole logits kept; compiling it again with %s", moved
        )
        return True
    model = _model_of(objective)
    if model is None:
        return False
    # A decoder always has a remat field; any other model climbs only a
    # bool or named policy, so a missing field and None both stay put.
    current = _remat_of(model)
    ladder = (DECODER_REMAT if isinstance(model, CausalTransformer)
              else DIFFUSION_REMAT if isinstance(current, bool | str) else ())
    current = 'dots' if current is True else current
    if current not in ladder[:-1]:
        return False
    stronger = ladder[ladder.index(current) + 1]
    _log.warning(
        "the step does not fit the devices under remat %r; compiling it again under %r", current, stronger
    )
    objective.model = model.clone(remat=stronger)
    return True


@dataclasses.dataclass(frozen=True)
class Plateau:
    """Stop after `evals` eligible evaluations without an improvement larger than min_delta."""
    metric: Metric | TrainingScalar | types.MethodType
    evals: int = 5
    min_delta: float = 0.0
    mode: Literal['min', 'max'] | None = None
    split: str | None = None

    def __post_init__(self):
        if self.evals < 1 or self.min_delta < 0 or not math.isfinite(self.min_delta):
            raise ValueError("Plateau needs evals >= 1 and min_delta >= 0")


class _MetricValues[Statistics, Additions](Mapping):
    def __init__(
        self,
        metrics: Sequence[Metric],
        scores: Mapping[str, float],
        objective: Objective[Statistics, Additions],
    ):
        self.objective = objective
        self.metrics = tuple(metrics) + tuple(objective.scalars[name] for name in ('loss', *objective.shown)
                                              if name == 'loss' or f'train/{name}' in scores)
        self.scores = scores

    def __getitem__(self, metric):
        split = 'val'
        if isinstance(metric, tuple):
            split, metric = metric
        if isinstance(metric, TrainingScalar):
            if metric.owner is not self.objective:
                raise KeyError("training score belongs to another objective")
            return self.scores[f'train/{metric.name}']
        if (
            isinstance(metric, types.MethodType)
            and metric.__name__ == "loss"
            and metric.__self__ is self.objective
        ):
            return self.scores['train/loss']
        if isinstance(metric, str):
            if any(metric == f'{name}/{declared.name}' for name in {key.split('/')[0] for key in self.scores}
                   for declared in self.metrics):
                raise KeyError("index declared evaluation scores by metric object")
            return self.scores[metric]
        for declared in self.metrics:
            if declared is metric:
                return self.scores[f'{split}/{declared.name}']
        raise KeyError("the score refers to a metric not passed to fit")

    def __iter__(self):
        for name in self.scores:
            split, _, label = name.partition('/')
            metric = next((metric for metric in self.metrics if metric.name == label), None)
            yield (
                name
                if metric is None
                else metric
                if split == "val" or isinstance(metric, TrainingScalar)
                else (split, metric)
            )

    def __len__(self):
        return len(self.scores)


@dataclasses.dataclass(frozen=True)
class _FitPlan:
    """What one `Trainer.fit` call was asked for."""

    dataset: Dataset
    steps: int
    log_every: int
    eval_every: int | None
    checkpoint_every: int | datetime.timedelta | None
    local_every: int | None
    metrics: Sequence[Metric]
    preview: bool
    best: tuple[Best, ...] = ()
    stop: Plateau | None = None
    validation_splits: Mapping[str, Reader] | None = None
    restore_best: bool = False
    validation: bool = False


@dataclasses.dataclass
class _FitRun:
    """One `Trainer.fit` call's progress: what its loop carries from step
    to step and what its cleanup reads however the run ends."""

    started: float
    current: int = 0
    loss: jax.Array | None = None
    evaluation: Evaluation | None = None
    last_checkpoint: float = 0.0
    stop_control: dict = dataclasses.field(default_factory=dict)
    stopped: bool = False
    training: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    source: Iterator[Batch] | None = None
    train: DevicePrefetchIterator | None = None
    tracing: bool = False
    traced: int = 0
    other: float = 0.0
    first_step: float | None = None
    preempted: int | None = None
    notice: PreemptionNotice | None = None


@dataclasses.dataclass
class _Interval:
    """What `fit` sums between logs and between checkpoints.

    `book` is the loss summed since the last checkpoint and both bad-loss
    counters, on device, so the loop never blocks on a result, and they move
    together in one dispatch; the host reads them at the logging cadence.
    `steps` counts since the last checkpoint, the rest since the last log.
    The records and FLOPs are summed per step rather than taken off the
    dataset's batch, so a ramped interval reports the records it read and
    their FLOPs. `rollout_seconds` is the time spent sampling, logged under
    train/rollout_seconds when a rollout is set."""

    book: Book
    last_log_time: float
    last_saved: int | None
    steps: int = 0
    since_log: int = 0
    samples: int = 0
    flops: float | None = 0.0
    rollout_seconds: float = 0.0

    def count(self, batch: Batch, flops: float | None, loss: jax.Array, finite: jax.Array) -> None:
        """Add one step: its records, its FLOPs and its loss."""
        self.since_log += 1
        self.steps += 1
        self.samples += rows_of(batch)
        self.flops = None if self.flops is None or flops is None else self.flops + flops
        self.book = bookkeep(self.book, loss, finite)

    def check_finite(self, step: int, display: TrainingDisplay) -> None:
        """Raise RuntimeError once the loss has been non-finite for
        BAD_LOSS_STEPS steps, and start the longest streak over.

        Deferred to the logging cadence so the step loop never synchronises;
        detection is late by at most that many steps, never missed.
        """
        loss, bad_run, worst_bad_run = self.book
        streak = int(worst_bad_run)
        if streak >= BAD_LOSS_STEPS:
            raise RuntimeError(
                f"Loss has been non-finite for {streak} consecutive steps "
                f"ending near step {step}, stopping")
        if streak:
            display.note(f"Non-finite loss for {streak} step(s) before {step}", style="red")
        self.book = (loss, bad_run, jnp.zeros((), jnp.int32))

    def logged(self, now: float) -> None:
        """Start the next logging interval at `now`."""
        self.last_log_time, self.since_log, self.samples = now, 0, 0
        self.flops, self.rollout_seconds = 0.0, 0.0

    def saved(self, step: int) -> None:
        """Start the next checkpoint interval after the one saved at `step`."""
        loss, bad_run, worst_bad_run = self.book
        self.last_saved, self.steps = step, 0
        self.book = (jnp.zeros_like(loss), bad_run, worst_bad_run)


_DEFAULT_MESH = MeshSpec()
_DEFAULT_LAYOUT = Layout()


class Trainer(Generic[Loss, Effects]):
    """Runs an `Objective`: gradients, sharding, EMA, checkpoints, logging."""

    def __init__(
        self,
        objective: Objective[Loss, Effects],
        optimizer: optax.GradientTransformation,
        *,
        key: int | jax.Array,
        mesh: MeshSpec = _DEFAULT_MESH,
        layout: Layout = _DEFAULT_LAYOUT,
        accumulation: int = 1,
        dynamic_scale: bool = False,
        checkpoints: Checkpoints | None = None,
        tracker: Tracker | None = None,
        step: Callable[[Objective[Loss, Effects], optax.GradientTransformation], StepFn] | None = None,
        rollout: Rollout | None = None,
        profile: ProfileWindow | None = None,
    ):
        """Hold everything a run needs, without opening any of it.

        The mesh, the compiled step and the capabilities' resources come into
        being in `fit`, so constructing a Trainer allocates nothing.

        `accumulation` is how many microbatches pool into one optimizer
        commit. `step` replaces the built-in transaction. A custom step then
        owns the clocks, the scaler, the EMA and the mutable writes, and the
        compiled wrapper owns only the attempted-step counter. `rollout` runs
        once per batch read, before the step and outside replay. `layout` and
        `mesh` say where the state lives.
        """
        if accumulation < 1:
            raise ValueError(f"accumulation must be at least 1, got {accumulation}")
        if step is not None and "params" in layout.host:
            raise ValueError(
                "Parameter-streamed training requires the trainer objective transaction; "
                "custom steps own their execution"
            )
        self.objective = objective
        self.optimizer = optimizer
        from dew.nn.inputs import request_key
        self.seed = int(key) if isinstance(key, (int, np.integer)) and not isinstance(key, bool) else None
        self.key = request_key(key)
        self.mesh = mesh
        self.layout = layout
        self.accumulation = accumulation
        self.dynamic_scale = dynamic_scale
        self.checkpoints = checkpoints
        self.tracker = tracker
        self.step = step
        self.rollout = rollout
        self.profile = profile
        # Set by `compile`, for the batch shape it was called with. A ramped
        # run has one value per stage; `fit` keeps them beside each step. The
        # program is the step as handed to XLA, before GSPMD partitions it,
        # and the executable the step as compiled, whose memory analysis a
        # benchmark reads.
        self.flops_per_step = None
        self.program: jax.stages.Lowered | None = None
        self.executable: jax.stages.Compiled | None = None
        # The links the last compiled step decided its projections by, which
        # say whether one spread over each axis (`dew.nn.sharding.Link`), and
        # each mesh axis's measured bandwidth.
        self.links: dict[str, Link] = {}
        self._bandwidths: dict[tuple[Mesh, str], float | None] = {}
        self._display = TrainingDisplay()
        # The fit ladder's rung beyond the objective's own head and remat: a
        # step that fit only under XLA's default options keeps them for later
        # compiles. A resumed run starts at its checkpoint's rung, and says so
        # if its first compile has to climb past it (`_climb_to`).
        self._xla_defaults = False
        self._resumed_rung: JSON = None

    @classmethod
    def from_config(
        cls, config: TrainerConfig, objective: Objective[ObjectiveLoss, ObjectiveEffects],
        optimizer: optax.GradientTransformation, *, key: int | jax.Array,
        checkpoints: Checkpoints | None = None, tracker: Tracker | None = None,
        step: Callable[[Objective[ObjectiveLoss, ObjectiveEffects],
                        optax.GradientTransformation], StepFn] | None = None,
        rollout: Rollout | None = None,
    ) -> Trainer[ObjectiveLoss, ObjectiveEffects]:
        """Build the trainer a `TrainerConfig` describes.

        The mapping from the config's field names to this constructor's is
        written once, here. `mesh`, `layout`, `accumulation`,
        `dynamic_scale` and `profile` are the config fields a trainer holds.
        `key` is the run key, which `RunConfig.train` draws from
        `config.key`.

        The rest of the config belongs to the capabilities and to the loop,
        and reaches them from their own owners. `checkpoint_dir` and `keep`
        build the `Checkpoints` passed in here, and `wandb` the tracker.
        `xla_flags`, `multi_host` and `compilation_cache_dir` are read by
        `prepare_process` before JAX opens a backend. `batch_ramp` wraps the
        dataset with `dew.data.ramped`. `steps`, `epochs`, `log_every`,
        `eval_every` and `checkpoint_every` are arguments of `fit`. `step`
        and `rollout` are not configurable: they are code a caller hands
        over.

        It builds a `Trainer`, whatever it is called on. The objective's two
        parameters are the factory's own, so a subclass that wants one of
        itself constructs it.
        """
        return Trainer(
            objective, optimizer,
            key=key,
            mesh=config.mesh,
            layout=config.layout,
            accumulation=config.accumulation,
            dynamic_scale=config.dynamic_scale,
            checkpoints=checkpoints,
            tracker=tracker,
            step=step,
            rollout=rollout,
            profile=config.profile,
        )

    # ------------------------------------------------------------------
    # The state
    # ------------------------------------------------------------------

    def initial_state(self, initializer: Initializer | None = None,
                      key: int | jax.Array | None = None) -> TrainState:
        """Build the state a fresh run starts from.

        It is pure, so `fit` traces it once for its shapes and once, sharded,
        for its values.

        Both inputs are the run's own by default, and `place` passes them
        explicitly so that what it compiles takes them as arguments. A held
        checkpoint then reaches the device as an argument instead of as a
        constant embedded in the executable. Passing None means resolve the
        configured input, which is what a no-argument call does. This is the
        one state implementation, so a subclass overrides it here and every
        path sees the override.
        """
        initializer = self.objective.initializer if initializer is None else initializer
        from dew.nn.inputs import request_key
        key = self.key if key is None else request_key(key)
        init_key, run_key = jax.random.split(key)
        params = nn.unbox(initializer(init_key))
        if "params" not in params:
            raise ValueError(
                f"the objective's tree has no params collection, only {sorted(params)}; "
                "the optimizer moves params and treats every other collection as state")
        ema = self.objective.ema
        return TrainState(
            step=jnp.zeros((), jnp.int32),
            microstep=jnp.zeros((), jnp.int32),
            updates=jnp.zeros((), jnp.int32),
            scale=(jax.tree.map(jnp.asarray, dynamic_scale_lib.DynamicScale())
                   if self.dynamic_scale else None),
            window_size=jnp.asarray(self.accumulation, jnp.int32),
            params=params,
            opt_state=self.optimizer.init(params["params"]),
            # The average starts equal to the parameters but as its own
            # buffers: the step donates the state, and a buffer can be
            # donated once.
            ema=None if ema is None else jax.tree.map(jnp.copy, select(params, ema.select)),
            key=run_key,
        )

    @functools.cached_property
    def device_mesh(self) -> Mesh:
        """Build the mesh `MeshSpec` describes over this process pool's devices,
        on first use."""
        return build_mesh(self.mesh)

    @property
    def host_master(self) -> bool:
        return "params" in self.layout.host

    @functools.cached_property
    def state_mesh(self) -> Mesh:
        if not self.host_master:
            return self.device_mesh
        from dew.training.host import companion_mesh
        return companion_mesh(self.device_mesh)

    def shardings(self, state: TrainState) -> Placement[TrainState]:
        """Place every field of `state`, each on the axes its own kind takes.

        Parameter gradients follow parameters, replay records follow batches,
        and the layout's host-resident fields sit in pinned host memory.
        Under a CPU-owned state the frozen collection is the exception: it
        sits where the realization reads it (`execution.resident`) for the
        whole run."""
        mesh = self.state_mesh
        params = dict(state.params)
        frozen = params.pop(FROZEN, None) if self.host_master else None
        placed = self.layout.shardings(mesh, dataclasses.replace(state, params=params, accumulation=None))
        # A root key is one value; a legacy key's uint32 words are not parameter axes.
        placed = dataclasses.replace(placed, key=NamedSharding(mesh, P()))
        placed = dataclasses.replace(placed, **{
            field: jax.tree.map(lambda s: s.with_memory_kind("pinned_host"), getattr(placed, field))
            for field in (() if self.host_master else self.layout.host)})
        if frozen is not None:
            placed = dataclasses.replace(
                placed, params={**placed.params, FROZEN: self._frozen_shardings(state, frozen)}
            )
        accumulation = state.accumulation
        if accumulation is None:
            return placed
        replicated = NamedSharding(mesh, P())
        pending = jax.tree.map(lambda _: replicated, accumulation)
        def buffered_shardings(tree, batches):
            if tree is None:
                return None
            sample = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), tree)
            layout = batch_shardings(mesh, sample) if batches else self.layout.shardings(mesh, sample)
            return jax.tree.map(lambda s: NamedSharding(mesh, P(None, *s.spec)), layout)
        pending = dataclasses.replace(pending,
            gradient=None if accumulation.gradient is None else placed.params["params"],
            batches=buffered_shardings(accumulation.batches, batches=True),
            variables=buffered_shardings(accumulation.variables, batches=False))
        return dataclasses.replace(placed, accumulation=pending)

    def _fetched(self, state: TrainState, shardings: Placement[TrainState]) -> TrainState:
        """Bring the layout's host-resident fields to the device, where a step
        or an evaluation reads them."""
        if self.host_master:
            return state
        return dataclasses.replace(state, **{
            field: jax.device_put(getattr(state, field), jax.tree.map(
                lambda s: s.with_memory_kind("device"), getattr(shardings, field)))
            for field in self.layout.host})

    def place(self) -> tuple[TrainState, Placement[TrainState], bytes | None]:
        """Put the state on the mesh, fresh or restored.

        Returns it with its shardings and the data position a resume
        continues from."""
        # Resolved once, so that the shapes and the values are the same
        # inputs through the same overridable method, and the objective is
        # asked for what it holds exactly once.
        initializer, key = self.objective.initializer, self.key
        if self.host_master:
            from dew.training.host import transfer
            key = transfer(key, NamedSharding(self.state_mesh, P()))
        abstract = jax.eval_shape(self.initial_state, initializer, key)
        refuse_wide_floats(abstract, self.device_mesh)
        shardings = self.shardings(abstract)
        self.layout.check(abstract.params, shardings.params, self.device_mesh)
        checkpoints = self.checkpoints
        resume = None if checkpoints is None else checkpoints.latest
        if checkpoints is None or resume is None:
            if self.host_master:
                return self._placed_host(initializer, key, shardings), shardings, None
            held = self._placed_held(initializer, abstract, shardings)
            with warnings.catch_warnings():
                # A held array no output can take over is freed as the JIT returns.
                warnings.filterwarnings('ignore', 'Some donated buffers were not usable')
                state = jax.jit(self.initial_state, out_shardings=shardings, donate_argnums=0)(held, key)
            return state, shardings, None
        abstract = dataclasses.replace(abstract, accumulation=checkpoints.accumulation_template(resume))
        shardings = self.shardings(abstract)
        if self.host_master and FROZEN in abstract.params:
            abstract = dataclasses.replace(abstract, params={**abstract.params, FROZEN: self._banked_frozen(
                abstract.params[FROZEN],
                lambda rows, path: jax.ShapeDtypeStruct((len(rows), *rows[0].shape), rows[0].dtype))})
        template = jax.tree.map(
            lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
            abstract, shardings)
        template = self._with_drawn_tables(template, shardings, checkpoints.stored(resume),
                                           initializer, key)
        state, position = checkpoints.restore(template, resume,
                                              share=data_partition(self.device_mesh))
        self._climb_to(checkpoints.rung(resume))
        from dew.nn.inputs import request_key
        state = dataclasses.replace(state, key=jax.device_put(request_key(state.key), shardings.key))
        if int(state.window_size) != self.accumulation:
            raise ValueError("checkpoint accumulation window_size differs from this trainer")
        self._display.note(f"Resumed from step {resume} in {checkpoints.source(resume)}")
        return state, shardings, position

    def _rung(self) -> JSON:
        """The fit ladder's rung this trainer's step compiles at: the
        objective's head tile, its model's remat (`remat_record`) and whether
        the step keeps XLA's default options (`fitting_default`). A
        checkpoint records it, since each decides the program a step runs."""
        tile = _head_tile_of(self.objective)
        return {'head_tile': None if tile is None else list(tile),
                'remat': remat_record(_remat_of(_model_of(self.objective))),
                'xla_defaults': self._xla_defaults}

    def _climb_to(self, rung: JSON) -> None:
        """Move the objective up to `rung`, a checkpoint's (`_rung`), where
        it is above where the objective stands, never below it.

        The fit check reads the free memory its own process finds, and one
        that restores a state finds other memory than the one that built it:
        on an A100 a Qwen3-1.7B run trained under remat 'full', and its
        resumed process took 'minimal', ran another program and parted from
        the uninterrupted run at step 70. So the resumed run compiles the
        rung its checkpoint trained on, and climbs further only where that
        does not fit."""
        if rung is None:
            return
        fields = record(rung, 'rung')
        climb_to(self.objective, fields)
        self._xla_defaults = self._xla_defaults or boolean(
            fields.get("xla_defaults", False), "rung xla_defaults"
        )
        self._resumed_rung = self._rung()

    def _placed_held(self, initializer: Initializer, abstract: TrainState,
                     shardings: Placement[TrainState]) -> Initializer:
        """`initializer` with a copy of each array it holds placed where the
        state keeps the variable of its path and shape
        (`Objective.held_variables`), for the state's JIT to take over, the
        rest replicated.

        Handed to the JIT as they are, a loaded checkpoint's arrays were placed
        below the state it built and freed once it was: a hole as large as the
        checkpoint under the state, 5.9 GiB on an A100 for Qwen3-1.7B, whose
        'minimal' rung then found no free block for its temporaries
        (`step_headroom`). Placed where the state keeps them and donated, they
        become the state's buffers, of the variable or of an optimizer moment
        laid out like it. The copy leaves the objective's own arrays alone: a
        checkpoint loaded to the host (`load_pretrained`) leaves no hole, while
        arrays the objective already holds on the devices stay wherever they
        were placed for as long as it holds them."""
        variables = {path: (sharding, leaf.shape) for (path, leaf), sharding in zip(
            jax.tree_util.tree_leaves_with_path(abstract.params), jax.tree.leaves(shardings.params),
            strict=True)}
        replicated = NamedSharding(self.device_mesh, P())

        def placement(path, leaf):
            # An initializer binds the checkpoint as `variables`
            # (`Objective.initializer`), so a leaf's path below that keyword
            # is its variable's.
            keys = [key.key if isinstance(key, jax.tree_util.DictKey) else None for key in path]
            sharding, shape = variables.get(path[keys.index('variables') + 1:] if 'variables' in keys else (),
                                            (replicated, None))
            return sharding if shape == np.shape(leaf) else replicated

        # A copy of an array already on the devices: device_put can reuse a
        # buffer the target shares with the source, which donation would free.
        owned = jax.tree.map(lambda leaf: leaf.copy() if isinstance(leaf, jax.Array) else leaf, initializer)
        return jax.device_put(owned, jax.tree_util.tree_map_with_path(placement, initializer))

    def _with_drawn_tables(self, template: TrainState, shardings: Placement[TrainState],
                           stored: Variables, initializer, key) -> TrainState:
        """`template` with every Fourier table the checkpoint lacks drawn by
        the fresh init, where `restore` keeps it (`is_fourier_table`).

        A checkpoint written before the table became a variable trained
        against the table init draws. Only those leaves are computed: the
        rest of the initial state is dead code to the compiler.
        """
        from dew.checkpoints import absent
        from dew.nn.blocks import is_fourier_table

        drawn = [(field, path) for field in ("params", "ema")
                 if getattr(template, field) is not None and stored.get(field) is not None
                 for path in absent(getattr(template, field), stored[field]) if is_fourier_table(path)]
        if not drawn:
            return template

        def leaf(tree, path):
            for entry in path:
                tree = tree[entry.key]
            return tree

        values = jax.jit(
            lambda initializer, key: [leaf(getattr(self.initial_state(initializer, key), field), path)
                                      for field, path in drawn],
            out_shardings=[leaf(getattr(shardings, field), path) for field, path in drawn],
        )(initializer, key)
        filled = {(field, jax.tree_util.keystr(path)): value
                  for (field, path), value in zip(drawn, values, strict=True)}
        return dataclasses.replace(template, **{
            field: jax.tree_util.tree_map_with_path(
                lambda path, stub, field=field: filled.get((field, jax.tree_util.keystr(path)), stub),
                getattr(template, field))
            for field in {field for field, _ in drawn}})

    def _frozen_shardings(self, state: TrainState, frozen):
        """Return where the frozen collection sits for a CPU-owned run.

        Before placement the layout names each row's shards, and `resident`
        moves the stack's rows to bank memory, a scanned run's shared rows as
        one bank. Once placed, the collection holds those banks, whose
        leading layer axis the layout's rules do not name, and its arrays say
        where they sit.
        """
        leaves = jax.tree.leaves(frozen)
        if leaves and all(isinstance(leaf, jax.Array) for leaf in leaves):
            return jax.tree.map(lambda leaf: leaf.sharding, frozen)
        from dew.training.execution import resident
        rows = self.layout.shardings(
            self.state_mesh, dataclasses.replace(state, params={FROZEN: frozen}, accumulation=None))
        return resident(rows.params[FROZEN], self.bank_sites, self.device_mesh)

    @functools.cached_property
    def bank_sites(self):
        """List the objective's declared layer stacks, which a host layout streams as banks."""
        from dew.inference.banks import bank_sites
        return bank_sites(self.objective) if self.objective.bank_sites else ()

    def _banked_frozen(self, tree, stack, release=None):
        """Stack each scanned run's shared leaves of `tree` into one bank
        (`execution.banked`).

        That is the shape a placement and a checkpoint template take, and the
        arrays the state holds."""
        from dew.training.execution import banked
        return banked(tree, self.bank_sites, stack, release)

    def _placed_host(self, initializer, key, shardings: Placement[TrainState]) -> TrainState:
        """A fresh CPU-owned state, its leaves streamed into place one at a time.

        The tree is built eagerly on the CPU, where the held leaves the
        objective hands over are read by reference and nothing of size is
        computed. Each leaf is then moved to its placement and the source let
        go as it lands (`host.stream`): a moving leaf to the companion, a
        frozen one to where it stays resident (`execution.resident`). The
        transient is one leaf, not the tree, and one JIT could not have
        returned to the two device sets anyway.
        """
        from dew.training.execution import bank_bytes, check_bank_pool
        from dew.training.host import evict, place_leaf, stream, transfer
        held = self.objective.held_variables()
        with jax.default_device(self.state_mesh.local_devices[0]):
            state = self.initial_state(initializer, key)
            if FROZEN in state.params:
                check_bank_pool(bank_bytes(state.params[FROZEN], self.bank_sites), self.device_mesh)
                # A scanned run's frozen rows become its bank here and the
                # bank lands where it stays resident before the next is
                # stacked, so the host holds one bank in transit, not the
                # stack; the rows' pages are let go as it lands and the held
                # tree holds the placed bank where each row was. An
                # objective placed this way holds banks at its rows afterwards
                # and does not seed a second trainer.
                targets = shardings.params[FROZEN]

                def stack(rows, path):
                    target = targets
                    for component in path:
                        target = target[component]
                    bank = place_leaf(jnp.stack(rows), target)
                    for row in rows:
                        evict(np.asarray(row))
                    return bank

                def release(namespace, index, keys, bank):
                    node = held["params"] if held is not None else None
                    for component in (*namespace, f"layers_{index}", *keys[:-1]):
                        node = node.get(component) if isinstance(node, dict) else None
                    if isinstance(node, dict) and keys[-1] in node:
                        node[keys[-1]] = bank
                frozen = self._banked_frozen(state.params[FROZEN], stack, release)
                state = dataclasses.replace(state, params={**state.params, FROZEN: frozen})
        params = stream(state.params, shardings.params, held)
        ema = None if state.ema is None else stream(state.ema, shardings.ema)
        rest = dataclasses.replace(state, params=None, ema=None)
        placed = transfer(rest, dataclasses.replace(shardings, params=None, ema=None))
        return dataclasses.replace(placed, params=params, ema=ema)

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def _loss_shape(self, state: TrainState, batch: Batch):
        # Shapes and dtypes only: a resident frozen leaf sits in another
        # memory space than the moving ones, and the loss the realization
        # runs reads a snapshot in one space. The step's key is drawn inside
        # the trace, so an abstract state compiles a step as a placed one does.
        params = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.params)

        def loss(params, batch, microstep, key, step, ema):
            return self.objective.loss(
                params, batch, Step(microstep, jax.random.fold_in(key, step), with_ema(params, ema)))

        return jax.eval_shape(loss, params, batch, state.microstep, state.key, state.step, state.ema)

    def _initialize_accumulation(self, state: TrainState, batch: Batch, shapes, *, shape_only=False):
        if self.accumulation == 1 or self.step is not None or state.accumulation is not None:
            return state
        stats, aux = shapes
        shared = isinstance(stats, (Ratio, jax.ShapeDtypeStruct))
        mean_dtype = jnp.result_type(jnp.float32, *(x.dtype for x in jax.tree.leaves(stats)))
        slots = self.accumulation - 1
        def shape(leaf: jax.Array) -> jax.ShapeDtypeStruct:
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)

        trainable = jax.tree.map(shape, state.params["params"])
        records = jax.tree.map(shape, batch)
        mutable = (None if shared or aux.variables is None else
                   jax.tree.map(shape, {name: state.params[name] for name in aux.variables}))

        def zeros(leaf):
            dtype = (
                jnp.promote_types(leaf.dtype, jnp.float32)
                if jnp.issubdtype(leaf.dtype, jnp.inexact)
                else leaf.dtype
            )
            return jnp.zeros(leaf.shape, dtype)

        def buffer(tree):
            return jax.tree.map(lambda x: jnp.zeros((slots, *x.shape), x.dtype), tree)

        def allocate():
            return Accumulation(
                gradient=jax.tree.map(zeros, trainable) if shared else None,
                mass=jnp.zeros((), mean_dtype) if shared else None,
                statistics=((jnp.zeros((), mean_dtype),) if shared else
                            tuple(zeros(x) for x in jax.tree.leaves(stats))),
                effects=tuple(zeros(x) for x in jax.tree.leaves(aux.effects)),
                qk_stats=jax.tree.map(
                    lambda x: jnp.full(x.shape, -jnp.inf, jnp.promote_types(x.dtype, jnp.float32))
                    if jnp.issubdtype(x.dtype, jnp.inexact) else zeros(x),
                    compact_qk(jax.tree.map(zeros, aux.qk_stats))),
                batches=None if shared else buffer(records),
                variables=None if mutable is None else buffer(mutable),
                attempts=None if shared else jnp.zeros((slots,), jnp.int32))

        pending = jax.eval_shape(allocate)
        if not shape_only:
            placement = self.shardings(dataclasses.replace(state, accumulation=pending)).accumulation
            # Allocate on the final shards, without a replicated window-sized temporary.
            pending = jax.jit(allocate, out_shardings=placement)()
        return dataclasses.replace(state, accumulation=pending)

    def _links(self, mesh: Mesh) -> dict[str, Link]:
        """The links of `mesh`'s tensor and sequence axes, those it splits,
        for one trace. Every process lists the mesh's devices alike and
        `link_bandwidth` agrees one figure across the pool, so every process
        decides alike and compiles the same program. The peak is the fastest
        device's, on which the FLOPs a spread saves take the least time. Each
        axis's bandwidth is measured once per mesh, and not at all on a CPU
        mesh or for a device the peak table does not name, where it decides
        nothing."""
        devices = list(mesh.devices.flat)
        peaks = [peak for device in devices if (peak := peak_flops(device.device_kind)) is not None]
        peak = max(peaks) if len(peaks) == len(devices) else None
        platform = devices[0].platform
        links = {}
        for axis in (TENSOR_AXIS, SEQUENCE_AXIS):
            if mesh.shape[axis] == 1:
                continue
            if (mesh, axis) not in self._bandwidths:
                self._bandwidths[mesh, axis] = (
                    None if platform == 'cpu' or peak is None else link_bandwidth(mesh, axis))
            links[axis] = Link(self._bandwidths[mesh, axis], peak, platform)
        return links

    @contextlib.contextmanager
    def _traced_on(self, mesh: Mesh, links: Mapping[str, Link] | None = None) -> Iterator[Schedule]:
        """What a traced step reads from context: the mesh, the pipeline's
        microbatch count, the layout's rules, which place the activations
        the model constrains (`dew.nn.sharding.constrain`) as they place the
        parameters, and the links of the axes the step measured, none by
        default."""
        with (jax.set_mesh(mesh), pipeline_microbatches(self.mesh.microbatches) as schedule,
              nn.logical_axis_rules(self.layout.axis_rules), measured_links({} if links is None else links)):
            yield schedule

    def compile(self, state: TrainState, batch: Batch) -> CompiledStep:
        """Compile a transaction over state and one already-produced global batch.

        The step consumes the state it is given. The returned state takes
        over its buffers, so the update runs in place and peak memory holds
        one copy of the parameters and optimizer state, not two. Keep no
        reference to a state after stepping it; `new = step(old, batch)` is
        the whole contract.

        A checkpoint saved before the step is safe. Orbax copies every array
        to the host before `save` returns, as long as `Checkpoints` names no
        prioritized keys and no concurrent transfer limit. The batch is not
        donated; the loader owns it.
        """
        if int(state.window_size) != self.accumulation:
            raise ValueError("checkpoint accumulation window_size differs from this trainer")
        if (state.scale is not None) != self.dynamic_scale:
            raise ValueError("checkpoint dynamic-scaler configuration differs from this trainer")
        if self.host_master:
            return self._compile_host(state, batch)
        mesh = self.device_mesh
        links = self._links(mesh)
        resumed, self._resumed_rung = self._resumed_rung, None
        with self._traced_on(mesh, links) as schedule:
            shapes = None if self.step is not None else self._loss_shape(state, batch)
            if shapes is not None and mesh.shape[STAGE_AXIS] > 1 and not schedule.pipelined:
                raise LayoutRefused(
                    f"the stage axis of {mesh.shape[STAGE_AXIS]} holds a pipeline's stages of "
                    f"a decoder's layer stack, and {type(self.objective).__name__}'s model runs "
                    f"no pipeline, so every stage would compute the whole step; give those "
                    f"devices to the data or fsdp axis")
            prepared = self._initialize_accumulation(state, batch, shapes, shape_only=True)
            shardings = self.shardings(prepared)
            replicated = NamedSharding(mesh, P())
            placement = batch_shardings(mesh, batch)
            held = prefetched_bytes(batch, placement)
            prepared = jax.tree.map(
                lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s), prepared, shardings)
            while True:
                body = (self.step(self.objective, self.optimizer) if self.step is not None else
                        Transaction(self.objective, self.optimizer, self.accumulation, shapes).step())

                def step(current, batch, body=body):
                    # The body sees every field on the device; the out shardings
                    # return the host-resident ones to pinned host memory.
                    advanced, loss, aux = body(self._fetched(current, shardings), batch)
                    return (dataclasses.replace(advanced, step=current.step + 1), loss,
                            aux.metrics, jnp.isfinite(loss), advanced.microstep > current.microstep)

                jitted = jax.jit(step, in_shardings=(shardings, placement),
                                 out_shardings=(shardings, replicated, replicated, replicated,
                                                replicated),
                                 donate_argnums=0)
                self.program = jitted.lower(prepared, batch)
                options = None if self._xla_defaults else step_compiler_options(self.objective)
                self.executable = self.program.compile(options)
                fits = step_fits(self.executable, mesh, held)
                if not fits and options is not None:
                    self.executable, fits = fitting_default(self.program, self.executable, mesh, held)
                    self._xla_defaults = fits
                if not fits and resumed is not None:
                    _log.warning(
                        "the step does not fit the devices at the rung its checkpoint trained on "
                        "(%s); from here the resumed run computes otherwise than the run it continues",
                        resumed,
                    )
                    resumed = None
                if fits or not recompute_more(self.objective):
                    break
            self.flops_per_step = compiled_flops(self.executable)
            self.links = links
        # The program compiled above, not the jit: a call through the jit
        # traces and compiles its own, without the step's compiler options,
        # which was 25 s of an A100's cold start.
        executable = self.executable

        def run(current, batch):
            with self._traced_on(mesh):
                if current.accumulation is None and self.accumulation > 1 and self.step is None:
                    current = self._initialize_accumulation(current, batch, shapes)
                    current = jax.device_put(current, shardings)
                return executable(current, batch)
        return run

    def _compile_host(self, state: TrainState, batch: Batch) -> CompiledStep:
        from dew.training.execution import HostExecution
        from dew.training.host import transfer
        cpu = self.state_mesh
        execution = HostExecution(self.objective, self.layout, self.device_mesh, cpu)
        cpu_batch = transfer(batch, batch_shardings(cpu, batch))
        with self._traced_on(cpu):
            shapes = self._loss_shape(state, cpu_batch)
            prepared = self._initialize_accumulation(state, cpu_batch, shapes, shape_only=True)
            placement = self.shardings(prepared)
            transaction = Transaction(self.objective, self.optimizer, self.accumulation, shapes)
            body = transaction.step(realize=execution.realize, host=True)

        def run(current, batch):
            batch = transfer(batch, batch_shardings(cpu, batch))
            with self._traced_on(cpu):
                if current.accumulation is None and self.accumulation > 1:
                    current = self._initialize_accumulation(current, batch, shapes)
                advanced, loss, aux = body(current, batch)
                advanced = dataclasses.replace(advanced, step=current.step + 1)
                # A leaf already where its placement says stays the array it
                # is: the resident frozen collection crosses no boundary.
                advanced = transfer(advanced, placement)
                return (advanced, loss, aux.metrics, jnp.isfinite(loss),
                        advanced.microstep > current.microstep)
        return run

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: Dataset,
        *,
        steps: int,
        log_every: int = 100,
        eval_every: int | None = None,
        checkpoint_every: int | datetime.timedelta | None = None,
        metrics: Sequence[Metric] = (),
        preview: bool = False,
        best: str | Metric | TrainingScalar | types.MethodType | Best | Sequence[Best] | None = None,
        stop: Plateau | None = None,
        validation: Mapping[str, Reader] | None = None,
        restore_best: bool = False,
    ) -> TrainState:
        """Train to `steps` total steps, resuming from the checkpoints' latest
        step when the directory holds one.

        Every `log_every` steps the tracker receives the loss, the objective's
        metrics and the throughput.

        Every `eval_every` steps, and at the end, the validation split is
        scored. The objective's artifacts go to the tracker and to `metrics`,
        whose reductions are logged as `val/<name>`.

        Every `checkpoint_every` steps, and at the end, the state and the
        data position are written. Every `checkpoints.local_every` steps they
        are written to the local directory as well.

        A preemption notice (a scheduler's SIGTERM; `PreemptionNotice`) stops
        the run at the next step every process agrees on: that step's state
        and data position are written, the final validation is skipped, and
        fit raises `Preempted`, which ends the program with 143 unless caught.
        Run again, fit resumes there.

        Previews are generated only when `preview=True` and a tracker
        receives them; scalar reporting never triggers preview work.
        """
        selection, stop = self._fit_policies(best, stop, metrics, validation, checkpoint_every, restore_best)
        self._preflight(dataset, stop, eval_every, metrics, preview=preview)
        preview = preview and self.tracker is not None
        run = _FitRun(time.perf_counter())
        # A display of this fit's own: one before it may have run other steps.
        self._display = TrainingDisplay()
        profile, checkpoints = self.profile, self.checkpoints
        profiler = self._own_profile_window()
        # One boundary lookup at setup: the prefetch worker and the step
        # scopes share whichever profiler owns the capture.
        tracer = profiler if profiler is not None else telemetry_profile.active_profile()
        try:
            plan = _FitPlan(
                dataset,
                steps,
                log_every,
                eval_every,
                checkpoint_every,
                None if checkpoints is None else checkpoints.local_every,
                metrics,
                preview,
                best=selection,
                stop=stop,
                validation_splits=validation,
                restore_best=restore_best,
                validation=validation is not None
                or (bool(eval_every or metrics) and dataset.val is not None),
            )
            state, shardings, position = self.place()
            run.last_checkpoint = time.perf_counter()
            if checkpoints is not None and checkpoints.latest is not None:
                run.stop_control = checkpoints.control(checkpoints.latest)
            if self._opened(plan, run, state, position):
                return state
            state = self._training_loop(plan, run, state, shardings, position, profiler, tracer,
                                        profile, checkpoints)
        finally:
            primary = sys.exception()
            error = self._closed(run, primary, profiler)
            if primary is None and error is not None:
                raise error
        if run.preempted is not None:
            raise Preempted(run.preempted)
        if restore_best and checkpoints is not None:
            state, _ = checkpoints.restore(
                jax.tree.map(
                    lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding), state
                ),
                step="best",
            )
        return state

    def _training_loop(self, plan: _FitPlan, run: _FitRun, state: TrainState,
                       shardings, position, profiler: Profiler | None, tracer: Profiler | None,
                       profile: ProfileWindow | None, checkpoints: Checkpoints | None) -> TrainState:
        """Dispatch the numerical steps and finish their loop before resource cleanup."""
        compiled: dict[Shapes, tuple[CompiledStep, float | None]] = {}
        interval = _Interval(fresh_book(), time.time(), last_saved=(
            run.current if checkpoints is not None and checkpoints.latest is not None else None))
        seen = 0
        run.notice = PreemptionNotice()
        while run.current < plan.steps:
            # Read through `run` each time: a local alias would keep the
            # closed iterator reachable from a failed run's traceback.
            assert run.train is not None
            # The window's capture opens before this iteration's first
            # read, so the step row records the read it waits on rather
            # than a compile that ran before capture began.
            if (profiler is not None and profile is not None
                    and not run.tracing and run.traced == 0
                    and seen >= profile.warmup):
                self._start_window(profiler)
                run.tracing = True
            capturing = tracer is not None and tracer.running
            step_scope = (jax.profiler.StepTraceAnnotation("train", step_num=run.current)
                          if capturing else contextlib.nullcontext())
            with step_scope:
                with region("input.wait"):
                    batch = next(run.train)
                if self.rollout is not None:
                    batch, sampled = self._rolled_out(state, batch)
                    interval.rollout_seconds += sampled
                first_compile = not compiled
                train_step, measured_flops = self._compiled_for(compiled, state, batch)
                if first_compile:
                    # Rebound once the step is compiled, so the first tick
                    # measures steps, not the compile.
                    interval.last_log_time = time.time()
                with region("train.step"):
                    state, loss, aux, finite, accepted = train_step(state, batch)
                run.loss = loss
                position = run.train.source_state
                run.current += 1
                self._display.step(run.current)
                seen += 1
                interval.count(batch, measured_flops, loss, finite)
                if run.first_step is None:
                    loss.block_until_ready()
                    run.first_step = time.perf_counter() - run.started
                if self._between_steps(plan, run, interval, state, shardings, position,
                                       loss, aux, accepted):
                    break
            # The step row is complete once its scope exits; closing the
            # window here keeps the last iteration inside the capture.
            if run.tracing and profile is not None:
                run.traced += 1
                if run.traced == profile.steps:
                    run.tracing = False
                    assert profiler is not None
                    self._stop_trace(run, profile, profiler)
        self._wind_down(plan, run, interval, state, shardings, position, profiler)
        return state

    def _closed(self, run: _FitRun, primary: BaseException | None,
                profiler: Profiler | None) -> BaseException | None:
        """`fit`'s cleanup however the run ended: the preemption notice, the
        stream, a trace window and the display closed, the checkpoints
        waited on, and the outcome reported. Returns the error to raise when
        the run itself raised none.

        Each teardown step runs even when an earlier one failed, and a later
        failure becomes a note on the first, so one broken sink cannot hide
        the error that ended the run."""
        paused = time.perf_counter()
        profile, checkpoints = self.profile, self.checkpoints
        if run.notice is not None:
            run.notice.close()
        close = (run.train.close if run.train is not None else
                 run.source.close if isinstance(run.source, Closeable) else None)
        stop_trace = None
        if run.tracing and profile is not None:
            run.tracing = False
            assert profiler is not None
            stop_trace = functools.partial(self._stop_trace, run, profile, profiler)
        error = primary
        for label, cleanup in (
            ("Training iterator", close),
            ("Profiler", stop_trace),
            ("Checkpoint wait", None if checkpoints is None else checkpoints.wait),
            ("Display", self._display.close),
        ):
            if cleanup is None:
                continue
            try:
                cleanup()
            except BaseException as failure:
                if error is None:
                    error = failure
                else:
                    error.add_note(f"{label} cleanup failed: {failure!r}")
        # The traceback of a failed run holds fit's frame and this one, and
        # with them whatever these names still point at.
        run.source = run.train = None
        close = stop_trace = cleanup = None
        run.other += time.perf_counter() - paused
        return self._reported_outcome(error, run)

    def _opened(self, plan: _FitPlan, run: _FitRun, state: TrainState, position) -> bool:
        """Start `fit` from the placed state: report it, refuse a run already
        past its steps or a checkpoint cadence with no checkpointer, and open
        the training stream on its prefetch worker. Returns whether the run
        is already complete, a checkpoint at its last step having been
        written, so there is nothing to do."""
        mesh, checkpoints, steps = self.device_mesh, self.checkpoints, plan.steps
        run.current = current = int(state.step)
        started = FitStarted(current, steps,
            checkpoints.source(current) if checkpoints is not None and position is not None else None,
            sum(leaf.size for leaf in jax.tree.leaves(state.params["params"])), mesh.devices.size,
            jax.devices()[0].device_kind, jax.process_count(), dict(mesh.shape), seed=self.seed,
            sharded=_split_share(state.params))
        self._report(started, current)
        if plan.dataset.held_out:
            self._display.note(f"validation: {plan.dataset.held_out} records held out of train")
        if current > steps:
            raise ValueError(f"the run is at step {current}, past the {steps} asked for")

        if plan.checkpoint_every and checkpoints is None:
            raise ValueError(
                "checkpoint_every asks for checkpoints and this trainer has no "
                "checkpointer; pass Checkpoints(directory) to write any")
        if current == steps and checkpoints is not None and checkpoints.latest is not None:
            return True
        if current < steps:
            run.source = plan.dataset.train(data_partition(mesh))
            self._check_stream(run.source, mesh, plan.dataset.batch,
                               checkpointing=bool(plan.checkpoint_every or plan.local_every or (
                                   checkpoints is not None and plan.eval_every and plan.best and
                                   any(not choice.weights_only for choice in plan.best))))
            run.train = DevicePrefetchIterator(run.source, mesh, source_state=position)
            run.source = None  # Lifetime transferred to the prefetch worker.

        model = _model_of(self.objective)
        stored = sorted({str(leaf.dtype) for leaf in jax.tree.leaves(state.params["params"])})
        compute = getattr(model, "dtype", None)
        precision = "/".join(stored)
        if compute is not None and [str(jnp.dtype(compute))] != stored:
            precision += f" parameters, {jnp.dtype(compute)} compute"
        shown = {**TRAINER_SHOWN, **self.objective.shown, **_reported(self.rollout, plan.metrics)}
        agreed("training announcement", functools.partial(
            self._display.start, started, model=type(self.objective if model is None else model).__name__,
            batch=plan.dataset.batch, precision=precision, shown=shown,
            averaged=self.objective.ema is not None and not self.objective._ema_is_reference))
        return False

    def _between_steps(self, plan: _FitPlan, run: _FitRun, interval: _Interval, state: TrainState,
                       shardings: Placement[TrainState], position, loss: jax.Array,
                       aux: dict[str, jax.Array], accepted: jax.Array) -> bool:
        """The cadenced work after one step of `fit`: the log, the validation,
        the checkpoints, and the preemption notice. Returns whether the
        notice stops the run at this step, whose checkpoint is then written."""
        current, steps, checkpoints = run.current, plan.steps, self.checkpoints
        run.training = {'train/loss': loss, **{f'train/{name}': value for name, value in aux.items()}}
        if current % plan.log_every == 0:
            interval.check_finite(current, self._display)
            self._log_interval(current, loss, aux, accepted, state, interval)

        if isinstance(plan.checkpoint_every, datetime.timedelta):
            elapsed = time.perf_counter() - run.last_checkpoint
            due = np.asarray(elapsed >= plan.checkpoint_every.total_seconds())
            checkpoint_due = bool(due) if jax.process_count() == 1 else bool(
                np.asarray(multihost_utils.process_allgather(due)).any())
        else:
            checkpoint_due = bool(plan.checkpoint_every and current % plan.checkpoint_every == 0)
        evaluation_due = bool(plan.eval_every and current % plan.eval_every == 0)
        if current < steps and (
            evaluation_due or ((plan.validation or plan.best or plan.stop) and checkpoint_due)
        ):
            with region("evaluate"):
                run.evaluation = self._evaluation(plan, state, shardings, run.training)
                run.other += run.evaluation.elapsed_seconds
        scores = (
            {} if run.evaluation is None or run.evaluation.step != current else dict(run.evaluation.scores)
        )
        ranking = self._ranking(plan, scores)
        winners = checkpoints._candidates(ranking) if (
            evaluation_due and checkpoints is not None and (plan.checkpoint_every or plan.best)) else ()
        candidate = bool(winners)
        if scores and self._plateau(plan, run, scores):
            run.stopped = True
            run.stop_control['stop_reason'] = 'validation plateau'
            self._display.note("Stopped: validation plateau")
            candidate = checkpoints is not None

        # On its own clock, not the logging one, so that a cadence
        # which does not divide log_every still fires.
        if (plan.checkpoint_every and checkpoints is not None
                and checkpoint_due and current < steps) or (candidate and current < steps):
            assert checkpoints is not None
            run.other += self._saved_checkpoint(
                checkpoints,
                current,
                state,
                position,
                interval,
                scores=scores,
                ranking=ranking,
                training_best=not plan.validation and not plan.best,
                control=run.stop_control,
                weights_only=not checkpoint_due
                and not run.stopped
                and bool(winners)
                and all(rank.weights_only for rank in winners),
            )
            run.last_checkpoint = time.perf_counter()
        if (plan.local_every and checkpoints is not None
                and current % plan.local_every == 0 and current < steps):
            run.other += self._saved_local_checkpoint(
                checkpoints, current, state, position, control=run.stop_control)
        # Asked at every step on every process, as JAX's agreement
        # needs; the last step ends the run on its own.
        if run.stopped:
            return True
        assert run.notice is not None
        if current < steps and run.notice.reached(current):
            if checkpoints is not None and interval.last_saved != current:
                run.other += self._saved_checkpoint(
                    checkpoints,
                    current,
                    state,
                    position,
                    interval,
                    training_best=not plan.validation and not plan.best,
                    control=run.stop_control,
                )
            run.preempted = current
            return True
        return False

    def _wind_down(self, plan: _FitPlan, run: _FitRun, interval: _Interval, state: TrainState,
                   shardings: Placement[TrainState], position, profiler: Profiler | None) -> None:
        """`fit` after its last step: the stream closed, a preemption
        announced, the trace window stopped, the final validation (skipped on
        a preemption) and the checkpoint of the state the run ends on."""
        profile, checkpoints, current, loss = self.profile, self.checkpoints, run.current, run.loss
        if run.notice is not None:
            run.notice.close()
        paused = time.perf_counter()
        if run.train is not None:
            run.train.close()
            run.train = None
        run.other += time.perf_counter() - paused
        if run.preempted is not None:
            stopped_at = run.preempted

            def announce_preemption() -> None:
                self._display.note(f"Preempted at step {stopped_at}: " + (
                    f"its checkpoint and data position go to {checkpoints.directory}, "
                    "where the next run resumes" if checkpoints is not None
                    else "the trainer has no checkpointer, so nothing is written"))

            agreed("preemption announcement", announce_preemption)
        if run.tracing and profile is not None:
            run.tracing = False
            # The window outlived the run, and a trace left running takes the
            # next one down with it.
            assert profiler is not None
            self._stop_trace(run, profile, profiler)
        interval.check_finite(current, self._display)
        if loss is not None:
            # The last step has to land before the wall time is read.
            loss.block_until_ready()
        if (
            (plan.validation or plan.eval_every or plan.best or plan.stop)
            and run.preempted is None
            and not run.stopped
        ):
            run.evaluation = self._evaluation(plan, state, shardings, run.training)
            run.other += run.evaluation.elapsed_seconds
            if self._plateau(plan, run, run.evaluation.scores):
                run.stopped = True
                run.stop_control['stop_reason'] = 'validation plateau'
                self._display.note("Stopped: validation plateau")
        if checkpoints is not None and interval.last_saved != current:
            # The in-loop saves are conditional, so the state the run ends
            # on may never have been written. It goes out under its real
            # step: a step-0 checkpoint holding the final weights would
            # make a resume restart the schedule from the beginning.
            scores = {} if run.evaluation is None or run.evaluation.step != current else run.evaluation.scores
            run.other += self._saved_checkpoint(
                checkpoints,
                current,
                state,
                position,
                interval,
                scores=scores,
                ranking=self._ranking(plan, scores),
                training_best=not plan.validation and not plan.best,
                control=run.stop_control,
            )
        if checkpoints is not None:
            checkpoints.wait()

    def _fit_policies(self, best, stop, metrics, validation, checkpoint_every, restore_best):
        if isinstance(checkpoint_every, datetime.timedelta) and checkpoint_every.total_seconds() <= 0:
            raise ValueError("checkpoint_every duration must be positive")
        splits = ('val',) if validation is None else tuple(validation)
        if not splits:
            raise ValueError("validation must contain a split")
        raw = (
            ()
            if best is None
            else tuple(best)
            if isinstance(best, Sequence) and not isinstance(best, str)
            else (best,)
        )

        def owned(choice):
            if isinstance(choice, types.MethodType):
                if choice.__name__ == 'loss' and choice.__self__ is self.objective:
                    return self.objective.scalars.loss
                raise TypeError("a training selector must be this objective's loss or declared scalar")
            if isinstance(choice, TrainingScalar) and choice.owner is not self.objective:
                raise ValueError("training scalar belongs to a different objective")
            if isinstance(choice, Best):
                source = choice._source
                if source is None and (choice.metric.startswith('train/') or choice.split == 'train'):
                    source = self.objective.scalars[choice.metric.removeprefix('train/')]
                return dataclasses.replace(choice, _source=owned(source) if source is not None else None)
            return choice

        selection = tuple(
            self._best_selection(owned(Best(choice) if isinstance(choice, str) else choice), metrics, splits)
            for choice in raw
        )
        labels = [(choice.split, choice.metric) for choice in selection
                  if isinstance(choice._source, (Metric, TrainingScalar))]
        if len(labels) != len(set(labels)):
            raise ValueError("select each metric once; one Best.top already keeps all its winners")
        if stop is not None:
            metric = owned(stop.metric)
            if not isinstance(metric, (Metric, TrainingScalar)):
                raise TypeError(
                    "Plateau selects a declared metric object, not a Best policy or score function"
                )
            stopping = self._best_selection(Best(metric, mode=stop.mode, split=stop.split), metrics, splits)
            stop = dataclasses.replace(stop, metric=metric, mode=stopping.mode, split=stopping.split)
        if restore_best and any(choice.weights_only for choice in selection):
            raise ValueError("restore_best requires full checkpoints, not weights_only snapshots")
        return selection, stop

    @staticmethod
    def _best_selection(
        best: str | Metric | TrainingScalar | Best,
        metrics: Sequence[Metric],
        splits: Sequence[str] = ("val",),
    ) -> Best:
        selection = best if isinstance(best, Best) else Best(best)
        metric = selection._source
        if metric is None:
            name = selection.metric.removeprefix((selection.split or 'val') + '/')
            metric = next((declared for declared in metrics if declared.name == name), None)
            if metric is None:
                raise ValueError(f"best metric {selection.metric!r} is not among fit's metrics")
            selection = dataclasses.replace(selection, _source=metric)
        if isinstance(metric, TrainingScalar):
            if selection.mode is None:
                if metric.shown.better is None:
                    raise ValueError("training scalar has no declared direction; use Best(scalar, mode=...)")
                selection = dataclasses.replace(
                    selection, mode="max" if metric.shown.better == "higher" else "min"
                )
            return dataclasses.replace(selection, split='train')
        if isinstance(metric, (Metric, TrainingScalar)):
            if not any(metric is declared for declared in metrics):
                raise ValueError("best must be a metric object passed in metrics")
            declaration = _reported(None, metrics).get(metric.name)
            better = declaration.better if isinstance(declaration, Shown) else None
            if selection.mode is None:
                if better not in ('higher', 'lower'):
                    raise ValueError(
                        "best metric has no declared direction; use Best(metric, mode='min' or 'max')"
                    )
                selection = dataclasses.replace(selection, mode='max' if better == 'higher' else 'min')
        if selection.split is None:
            if len(splits) != 1 and isinstance(metric, Metric):
                raise ValueError("Best(metric, split=...) is required with several validation splits")
            selection = dataclasses.replace(selection, split=splits[0] if len(splits) == 1 else None)
        elif selection.split not in splits:
            raise ValueError(f"unknown validation split {selection.split!r}")
        return selection

    def _ranking(self, plan: _FitPlan, scores: Mapping[str, float]) -> tuple[Ranking, ...]:
        if not plan.best:
            name = (
                ("val" if plan.validation_splits is None else next(iter(plan.validation_splits))) + "/loss"
                if plan.validation
                else "train/loss"
            )
            return (Ranking(name, scores[name]),) if name in scores else ()
        ranks = []
        for index, selection in enumerate(plan.best):
            metric = selection._source
            assert metric is not None
            if isinstance(metric, (Metric, TrainingScalar)):
                name = f'{selection.split}/{metric.name}'
                value = scores.get(name, float('nan'))
            else:
                name = f'aggregate:{index}'
                try:
                    value = float(metric(_MetricValues(plan.metrics, scores, self.objective)))
                except KeyError:
                    value = float('nan')
            mode = selection.mode or 'min'
            threshold = selection.threshold
            if threshold is not None and (value >= threshold if mode == 'min' else value <= threshold):
                value = float('nan')
            ranks.append(Ranking(name, value, mode, selection.top, selection.weights_only))
        return tuple(ranks)

    @staticmethod
    def _plateau(plan: _FitPlan, run: _FitRun, scores: Mapping[str, float]) -> bool:
        stop = plan.stop
        if stop is None:
            return False
        assert isinstance(stop.metric, (Metric, TrainingScalar)), (
            "fit binds the stopping selector before stepping"
        )
        name = f'{stop.split}/{stop.metric.name}'
        if name not in scores:
            return False
        value = scores[name] if stop.mode == 'min' else -scores[name]
        if not math.isfinite(value):
            return False
        key = f'plateau:{name}'
        held = run.stop_control.get(key)
        rule = {'mode': stop.mode, 'evals': stop.evals, 'min_delta': stop.min_delta}
        if held is not None and held.get('rule') != rule:
            raise ValueError(
                "Plateau policy differs from the resumed checkpoint; resume with the same stopping rule"
            )
        if held is not None and held.get('step') == run.current:
            return held['bad'] >= stop.evals
        if held is None or value < held['best'] - stop.min_delta:
            run.stop_control[key] = {'best': value, 'bad': 0, 'step': run.current, 'rule': rule}
            return False
        held['bad'] += 1
        held['step'] = run.current
        return held['bad'] >= stop.evals

    def _evaluation(self, plan: _FitPlan, state: TrainState, shardings: Placement[TrainState],
                    training: Mapping[str, jax.Array]) -> Evaluation:
        reports = training if self.checkpoints is not None or plan.best or plan.stop else None
        loss = plan.validation and not plan.best and not any(metric.name == 'loss' for metric in plan.metrics)
        if plan.validation_splits is None:
            return self._evaluate(
                state,
                shardings,
                plan.dataset,
                plan.metrics,
                plan.preview,
                loss=loss and self.checkpoints is not None,
                training=reports,
            )
        evaluations = []
        for split, reader in plan.validation_splits.items():
            evaluations.append(self._evaluate(state, shardings, plan.dataset, plan.metrics, plan.preview,
                                          loss=loss, reader=reader, split=split, training=reports))
        return dataclasses.replace(
            evaluations[0],
            scores={name: value for report in evaluations for name, value in report.scores.items()},
            elapsed_seconds=sum(report.elapsed_seconds for report in evaluations),
        )

    def _preflight(self, dataset: Dataset, stop, eval_every: int | None, metrics: Sequence[Metric], *,
                   preview: bool) -> None:
        """The refusals `fit` makes before it places anything: validation
        nothing reads, and a pipeline's batch, whose placement could take
        minutes or run out of memory before the error. Every other mesh's
        batch is checked with its stream (`_check_stream`), a ramp's at each
        of its stages."""
        if stop is None or not isinstance(stop.metric, TrainingScalar):
            self._check_validation_is_read(eval_every, metrics, preview=preview)
        if self.mesh.stage > 1:
            self._check_batch(dataset.batch, self.device_mesh)

    def _check_validation_is_read(self, eval_every: int | None,
                                  metrics: Sequence[Metric], *, preview: bool) -> None:
        """Refuse a validation cadence whose batches nothing would read.

        A validation pass hands its batches to metrics and to the preview.
        Scheduled with neither, it would open nothing and report nothing, so
        the contradiction is refused before the run does any work.
        """
        if (
            eval_every
            and not metrics
            and self.checkpoints is None
            and not (preview and self.tracker is not None)
        ):
            raise ValueError(
                f"eval_every={eval_every} schedules a validation pass that nothing consumes: "
                + ("preview=True needs a tracker to receive the samples; "
                   if preview else "")
                + "pass metrics to score it, preview=True with a tracker to sample from it, "
                "or leave eval_every unset")

    def _own_profile_window(self) -> Profiler | None:
        """Take ownership of the capture a configured window will write.

        A configured window must own the capture. When another profiler
        already holds it, refuse before the dataset or the mesh do any work,
        so neither trace is silently dropped or cut short. The window also
        resolves its optional dependency here, rather than after warmup steps
        have run.

        Every rank either leaves owning a stopped Profiler or raises
        together; a rank that proceeded alone would deadlock its peers at the
        first collective of training.
        """
        profile = self.profile
        if profile is None:
            return None

        def own_window() -> Profiler:
            if telemetry_profile.active_profile() is not None:
                raise ValueError(
                    "Trainer.profile cannot schedule a window while an explicit "
                    "dew.profile capture is active; drop one or stop the outer "
                    "profiler before fitting")
            telemetry_profile.require_profile_support()
            return telemetry_profile.profile(profile.directory)

        return agreed("profiling window setup", own_window)

    def _check_batch(self, batch: int, mesh: Mesh) -> None:
        """Refuse, before anything is placed, a global batch the mesh cannot
        split (`batch_divisor`): its rows shard over the batch axes as whole
        rows, and a pipeline cuts each device's rows again into M
        microbatches, where a device that holds none of a microbatch's rows
        computes another's again. The message names the row shards, and the
        batches and the microbatch counts that fit. A rollout's rows are its
        own, so a run with one is checked where its step traces."""
        if self.rollout is not None:
            return
        divisor = batch_divisor(mesh, self.mesh)
        if batch % divisor == 0:
            return
        count = self.mesh.microbatches or self.mesh.stage
        shards = divisor // count
        suggestions = [f"a batch that is a multiple of {divisor} rows, {-(-batch // divisor) * divisor} "
                       f"the nearest above {batch}"]
        axes = " x ".join(f"{axis} {mesh.shape[axis]}" for axis in BATCH_AXES if mesh.shape[axis] > 1)
        if count == 1:
            raise LayoutRefused(
                f"a global batch of {batch} rows over the {shards} row shards of {axes} leaves some "
                f"device without whole rows; use {suggestions[0]}, or a mesh whose batch axes "
                f"({' x '.join(BATCH_AXES)}) divide {batch}")
        if batch % shards == 0:
            fitting = [m for m in range(self.mesh.stage, batch // shards + 1, self.mesh.stage)
                       if (batch // shards) % m == 0]
            if fitting:
                suggestions.append(f"microbatches={fitting[-1]}, the most that divide the "
                                   f"{batch // shards} rows a device holds")
        raise LayoutRefused(
            f"a global batch of {batch} rows over {shards} row shards and {count} microbatches "
            f"leaves some microbatch without rows on some device, which then computes another "
            f"microbatch's again; use {' or '.join(suggestions)}")

    def _check_stream(self, source, mesh: Mesh, batch: int, *, checkpointing: bool) -> None:
        """Refuse a training stream this run cannot checkpoint or cannot shard.

        A checkpoint written without the data position would replay the data
        on resume. A `batch` the mesh cannot divide fails where it is placed
        or traced (`_check_batch`), and a ramp stage's, for a later stage,
        an hour in.
        """
        if checkpointing and not isinstance(source, Checkpointable):
            raise ValueError(
                f"checkpoint_every needs a training stream with get_state and "
                f"set_state, and {type(source).__name__} lacks one; a checkpoint "
                f"written without the data position would replay the data on "
                f"resume. Train it with checkpoint_every=None "
                f"(--trainer.checkpoint-every None)")
        if not isinstance(source, RampedStream):
            self._check_batch(batch, mesh)
            return
        if self.accumulation > 1:
            raise ValueError(
                f"a batch ramp grows the records a step reads, and an "
                f"accumulation window of {self.accumulation} pools "
                f"microbatches of one shape into one update; ramp the "
                f"batch or accumulate, not both")
        divisor = batch_divisor(mesh, self.mesh)
        refused = [stage.batch for stage in source.stages if stage.batch % divisor]
        if refused:
            raise ValueError(
                f"the batch ramp reads {refused} records a step at some of "
                f"its stages, and {dict(mesh.shape)} with "
                f"{self.mesh.microbatches or self.mesh.stage} microbatch(es) "
                f"holds a batch that is a multiple of {divisor}")

    def _start_window(self, profiler: Profiler) -> None:
        """Open the profiler's capture on every rank.

        A peer that failed to start raises here. A capture this rank did
        start is closed first, so no live trace outlives the aborted run.
        """
        capturing = False

        def start() -> None:
            nonlocal capturing
            profiler.start()
            capturing = True

        try:
            agreed("profiling window start", start)
        except BaseException as primary:
            if capturing:
                try:
                    profiler.stop()
                except BaseException as failure:
                    primary.add_note(f"Profiler stop failed: {failure!r}")
            raise

    def _rolled_out(self, state: TrainState, batch: Batch) -> tuple[Batch, float]:
        """Sample one batch through the rollout, with the seconds it took.

        Host-side and untraceable: sampling, scoring, advantages. The key
        folds the step key once more, keeping the rollout's draws off the
        step's stream; both are checkpointed, so a resumed run samples
        forward. Fixed shapes mean the step still compiles once.
        """
        began = time.perf_counter()
        assert self.rollout is not None
        key = jax.random.fold_in(jax.random.fold_in(state.key, state.step), 1)
        batch = shard_batch(self.device_mesh, self.rollout(state, batch, key))
        return batch, time.perf_counter() - began

    def _compiled_for(self, compiled: dict[Shapes, tuple[CompiledStep, float | None]],
                      state: TrainState, batch: Batch) -> tuple[CompiledStep, float | None]:
        """Return the step compiled for this batch's shapes, compiling on first sight.

        A ramped run reads a new shape at every stage, so `compiled` keeps
        one step per stage with the FLOPs measured for it.
        """
        shapes = batch_shapes(batch)
        if shapes not in compiled:
            began = time.perf_counter()
            with region("compile"):
                compiled[shapes] = (self.compile(state, batch), self.flops_per_step)
            seconds = time.perf_counter() - began
            remat = _remat_of(_model_of(self.objective))
            links = {axis: AxisLink(link.bytes_per_second, link.spread)
                     for axis, link in self.links.items()}
            self._report(StepCompiled(seconds, remat_record(remat), links), int(state.step))
        return compiled[shapes]

    def _saved_checkpoint(
        self,
        checkpoints: Checkpoints,
        step: int,
        state: TrainState,
        position: bytes | None,
        interval: _Interval,
        *,
        scores: Mapping[str, float] | None = None,
        ranking: Sequence[Ranking] = (),
        training_best: bool = True,
        control: dict | None = None,
        weights_only: bool = False,
    ) -> float:
        """Write one checkpoint with the interval's mean loss, report it and
        start the next interval.

        Returns the seconds it took, the reading of the interval's loss
        included: that read waits on the device, and the wait is time the
        steps did not have."""
        paused = time.perf_counter()
        self._display.status("writing a checkpoint")
        metadata = dict(scores or {})
        if interval.steps:
            metadata.setdefault('train/loss', float(interval.book[0] / interval.steps))
            if not ranking and training_best:
                ranking = (Ranking('train/loss', metadata['train/loss']),)
        checkpoints.save(step, state, position, metadata, share=data_partition(self.device_mesh),
                         ranking=ranking, control=control, weights_only=weights_only, rung=self._rung())
        self._report(CheckpointRequested(checkpoints.directory), step)
        interval.saved(step)
        self._display.status("")
        return time.perf_counter() - paused

    def _saved_local_checkpoint(
        self,
        checkpoints: Checkpoints,
        step: int,
        state: TrainState,
        position: bytes | None,
        *,
        control: dict | None = None,
    ) -> float:
        """Write one checkpoint to the local directory, and report it.

        Returns the seconds it took. The local copy carries no metadata; it
        is the one a restarted node reads back, not the run's record."""
        paused = time.perf_counter()
        self._display.status("writing a local checkpoint")
        if control:
            checkpoints.save_local(
                step,
                state,
                position,
                share=data_partition(self.device_mesh),
                control=control,
                rung=self._rung(),
            )
        else:
            checkpoints.save_local(
                step, state, position, share=data_partition(self.device_mesh), rung=self._rung()
            )
        self._report(CheckpointRequested(str(checkpoints.local_directory), local=True), step)
        self._display.status("")
        return time.perf_counter() - paused

    def _log_interval(self, step: int, loss: jax.Array, aux: dict[str, jax.Array],
                      accepted: jax.Array, state: TrainState, interval: _Interval) -> None:
        """Report one logging interval and start the next.

        Rank zero builds the row; every other rank has nothing to report.
        The report is agreed, so a tracker that failed on rank zero stops its
        peers here rather than at their next collective.
        """
        def report() -> None:
            if jax.process_index() != 0:
                return
            # The interval's numbers need the loss on the host, so this is
            # where the loop waits on the device.
            loss.block_until_ready()
            now = time.time()
            scalars = {"train/loss": float(loss),
                       **{f"train/{k}": float(v) for k, v in aux.items()},
                       **self._throughput(now - interval.last_log_time, interval.since_log,
                                          interval.samples, interval.flops)}
            scalars["train/accepted"] = float(accepted)
            if (rate := learning_rate(state.opt_state)) is not None:
                scalars["train/learning_rate"] = rate
            if state.scale is not None:
                scalars["train/loss_scale"] = float(state.scale.scale)
            if self.rollout is not None:
                scalars["train/rollout_seconds"] = interval.rollout_seconds
                scalars.update({f"rollout/{name}": float(value)
                                for name, value in _rollout_metrics(self.rollout).items()})
            self._display.interval(step, scalars)
            if self.tracker is not None:
                self.tracker.log(scalars, step)
            interval.logged(now)

        with region("log"):
            agreed("training reporting", report)

    def _reported_outcome(self, error: BaseException | None, run: _FitRun) -> BaseException | None:
        """Agree the run's cleanup, report its goodput and its outcome.

        Every rank reaches the cleanup agreement, so a rank that failed alone
        is heard here. Goodput is only reported for a run that reached this
        point without an error, since its numbers describe a run that ran.
        """
        try:
            agree_process_phase(error, phase="fit cleanup")
        except BaseException as failure:
            if error is None:
                error = failure
        if error is None:
            def report_goodput() -> None:
                if jax.process_index() == 0:
                    wall = time.perf_counter() - run.started
                    scalars = goodput(wall, run.first_step, run.other)
                    self._display.summary(run.current, wall, scalars,
                                          None if run.loss is None else float(run.loss))
                    if self.tracker is not None:
                        self.tracker.log(scalars, run.current)

            try:
                agreed("goodput reporting", report_goodput)
            except BaseException as failure:
                error = failure
        try:
            outcome = FitEnded.outcome(time.perf_counter() - run.started, error,
                                       preempted=run.preempted is not None)
            self._report(outcome, run.current)
        except BaseException as failure:
            if error is None:
                error = failure
            else:
                error.add_note(f'Reporting fit outcome failed: {failure!r}')
        return error

    def _report(self, value: Record, step: int) -> None:
        """Hand one record to the tracker on rank zero, then agree with the pool.

        The record's type names the phase, so a rank that failed to report is
        heard about at the report the pool was making, not at the next
        collective."""
        def send() -> None:
            if jax.process_index() == 0 and self.tracker is not None:
                self.tracker.artifact(value, step)

        agreed(type(value).__name__, send)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        state: TrainState,
        shardings: Placement[TrainState],
        dataset: Dataset,
        metrics: Sequence[Metric],
        preview: bool,
        *,
        loss: bool = False,
        reader: Reader | None = None,
        split: str = "val",
        training: Mapping[str, jax.Array] | None = None,
    ) -> Evaluation:
        """Score the validation split with this state's variables and report
        it, returning the seconds it took: the run's, but not its steps',
        which is what the goodput fraction is measured against.

        A CPU-owned run evaluates on the accelerator, over the same snapshot
        a step realizes, so validation reads the weights where the loss
        does."""
        paused = time.perf_counter()
        self._display.status("evaluating")
        mesh, params = self.device_mesh, state.params
        averaged = with_ema(state.params, self._fetched(state, shardings).ema)
        key = state.key
        if self.host_master:
            from dew.training.execution import HostExecution
            execution = HostExecution(self.objective, self.layout, mesh, self.state_mesh)
            with jax.set_mesh(self.state_mesh):
                params, averaged = execution.snapshot(params), execution.snapshot(averaged)
            key = execution.on_accelerator(key)
        assert params is not None, "evaluation always has model variables"
        # The rules and the microbatch count; `evaluate` scores under the
        # mesh itself, and previews decode outside it.
        with (pipeline_microbatches(self.mesh.microbatches),
              nn.logical_axis_rules(self.layout.axis_rules), measured_links(self._links(mesh))):
            evaluation = evaluate(
                self.objective, params, dataset.val if reader is None else reader, metrics=metrics, key=key,
                step=state.step, schedule_step=state.microstep,
                averaged=averaged, preview=preview, mesh=mesh, loss=loss, split=split, training=training)
        self._report_evaluation(evaluation)
        return dataclasses.replace(evaluation, elapsed_seconds=time.perf_counter() - paused)

    def _report_evaluation(self, advanced: Evaluation) -> None:
        """Print one evaluation on rank zero and log its previews and scores."""
        def report() -> None:
            if jax.process_index() != 0:
                return
            self._display.evaluation(advanced)
            if self.tracker is not None:
                for artifact in advanced.previews:
                    self.tracker.artifact(artifact, advanced.step)
                scalars = advanced.scalars
                if scalars:
                    self.tracker.log(scalars, advanced.step)

        agreed("evaluation reporting", report)


    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _stop_trace(self, run: _FitRun, profile: ProfileWindow, profiler: Profiler) -> None:
        """Stop the window's owned capture, then report it on process zero.

        The core Profiler drains the backend and exports the native reports
        on `stop`. The run's last loss is blocked on only to order the
        primary's failure ahead of the profiler's own drain."""
        loss, traced = run.loss, run.traced

        def stop() -> None:
            try:
                if loss is not None:
                    loss.block_until_ready()
            finally:
                primary = sys.exception()
                try:
                    profiler.stop()
                except BaseException as failure:
                    if primary is None:
                        raise
                    primary.add_note(f"Profiler stop failed: {failure!r}")

        def announce() -> None:
            self._display.note(f"Wrote profile for {traced} steps to {profile.directory}")

        agreed("profile stop", stop)
        self._report(ProfileWindowRecord(profile.directory, traced), run.current)
        agreed("profile announcement", announce)

    def _throughput(self, elapsed: float, steps: int, samples: int,
                    flops: float | None) -> dict[str, float]:
        """Compute the interval's rates from the records it read and its FLOPs.

        The FLOPs are the ones the compiler measured for each step's own
        shape; `flops` is None when a shape's measurement was unavailable.

        An interval that spans a stage boundary carries that stage's compile
        in its wall time. MaxText instead hides the performance metrics
        during a ramp (`common/metric_logger.py:166-194`).
        """
        if elapsed <= 0 or steps <= 0:
            return {}
        step_time = elapsed / steps
        scalars = {"train/step_time_ms": step_time * 1000,
                   "train/samples_per_sec": samples / elapsed}
        mfu = None if flops is None else model_flops_utilization(flops / steps, step_time)
        if mfu is not None:
            scalars["train/mfu"] = mfu
        return scalars
