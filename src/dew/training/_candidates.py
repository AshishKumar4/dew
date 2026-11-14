"""The step a fit search lowers at each candidate (`dew.training.planning.FitSearch`): the loss traced
once for its shapes and the narrow copies it reads (`prepared_for`), the state its window of microbatches
pools, its program over one microbatch (`step_program`), lowered, with what the fit check holds beside it
(`candidate_for`), and the same step lowered for a GPU this process lacks (`lowered_for`). The trainer
reads each one's (`StepCompiler._planned`)."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import threading
from collections.abc import Callable, Hashable, Mapping
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import jax.extend.core as jcore
import jax.numpy as jnp
import numpy as np
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P

from dew.batch import CLOSES_WINDOW, Batch
from dew.data.dataset import Stage
from dew.data.pipeline import DeviceStep
from dew.distributed.batches import PREFETCH_DEPTH, batch_shardings, prefetched_bytes
from dew.distributed.placement import Placement, Validation, placed_bytes, placed_shapes
from dew.nn.kernels import tuning
from dew.nn.kernels.generation import assumed, gpu_mesh
from dew.nn.kernels.options import owned_program, trace_context
from dew.nn.kernels.templates import report
from dew.nn.sharding import Link, Schedule, spread_axes, spread_over
from dew.state import Accumulation, TrainState, apart, taken, together
from dew.training._lowering import Routed, kernel_report, routed
from dew.training.diloco import group_of, windowed
from dew.training.distributed import microbatch, microbatch_shapes
from dew.training.model_flops import batch_consumed, batch_tokens
from dew.training.narrow import holds_copies, narrow_copies, narrowed
from dew.training.planning import Candidate, splits
from dew.training.stages import StageLocal
from dew.training.transaction import COMMIT_FIELDS, outcome

if TYPE_CHECKING:
    import optax

    from dew.distributed.placement import Layout, Placement
    from dew.objectives.base import Objective
    from dew.training.diloco import DiLoCo
    from dew.training.inspection import Inspection


class TracedOn(Protocol):
    """What a traced step reads from context, on `mesh`, with the links of the axes it measured, none by
    default (`Trainer._traced_on`)."""

    def __call__(self, mesh: Mesh | AbstractMesh, links: Mapping[str, Link] | None = None, /
                 ) -> contextlib.AbstractContextManager[Schedule]: ...


@dataclasses.dataclass(frozen=True)
class RunShape:
    """How a fit's batches are cut and counted, which every compile of its step reads: the rows its batches
    hold, each phase's (`rows`, `Loader.batches`), the device steps its reader runs (`device`) and the
    batches it prefetches (`prefetch`), the examples clock it runs on (`examples`, `clocked`), the stages of
    its batch ramp and the steps each begins at (`ramp`, `stage_starts`), the microbatches a step's batch is
    read in once decided (`split`: a pooling state's window or the first plan), and the updates its
    optimizer was built for (`updates`). A fit holds one, each part replaced as the run decides it."""

    rows: tuple[int, ...] | None = None
    device: tuple[DeviceStep, ...] = ()
    prefetch: int = PREFETCH_DEPTH
    examples: int | None = None
    ramp: tuple[Stage, ...] = ()
    stage_starts: frozenset[int] = frozenset()
    split: int | None = None
    updates: int | None = None


@dataclasses.dataclass(frozen=True)
class StepSettings[LossT, EffectsT]:
    """What a run states of the step it compiles, as its trainer holds it when it compiles
    (`Trainer._settings`): its objective, its optimizer once built (`optimizer`), its layout and device
    mesh, the steps a commit pools (`accumulation`), a custom step, its loss scaler, DiLoCo, whether it
    tunes its kernels, its inspection, the microbatches a pipelined step feeds, whether its steps record
    what they read, the CUDA generation its first step lowers for in place of compiling it (`lower_for`) and
    where a CPU-owned run keeps its state (`state_mesh`, the device mesh otherwise)."""

    objective: Objective[LossT, EffectsT]
    built_optimizer: optax.GradientTransformation | None
    layout: Layout
    device_mesh: Mesh
    accumulation: int
    step: Callable | None
    dynamic_scale: bool
    diloco: DiLoCo | None
    tune_kernels: bool
    inspection: Inspection | None
    microbatches: int
    record: bool
    lower_for: str | None
    state_mesh: Mesh

    @property
    def optimizer(self) -> optax.GradientTransformation:
        """The optimizer a step commits with, once the run built it over its length."""
        if self.built_optimizer is None:
            raise ValueError("an OptimConfig is built over the run's length, which only fit(steps=...) gives")
        return self.built_optimizer

    @property
    def host_master(self) -> bool:
        """Whether the host holds the variables themselves (a CPU-owned run)."""
        return "variables" in self.layout.host


@dataclasses.dataclass(frozen=True)
class Stepping[LossT, EffectsT]:
    """What a fit search's candidates read of the run, as it stands as it plans (`Trainer._stepping`): the
    run's settings (`StepSettings`), the value its step traces from beside its arguments, read as each
    candidate is made at the rung the search stands at (`step_key`, `Trainer._step_key`), where it places
    a state (`shardings`), its loss's traced shapes (`loss_shape`), where its step keeps a pipelined run
    (`staging`), its window's accumulation (`initialize_accumulation`), its model's batch
    and its shapes (`model_batch`, `model_shapes`), its step's body (`step_body`) and the fields it fetches
    (`fetched`), the mesh it traces on (`traced_on`), how its batches are cut and counted (`shape`), how
    the layout validates a state it holds between steps (`validation`) and whether and how a step lays out
    its window (`builds_window`, `window_of`): read here, none off the trainer itself (row 4555)."""

    settings: StepSettings[LossT, EffectsT]
    step_key: Callable[[], Hashable | None]
    shardings: Callable[[TrainState], Placement[TrainState]]
    loss_shape: Callable[..., tuple]
    staging: StageLocal
    initialize_accumulation: Callable[..., TrainState]
    model_batch: Callable[[Batch], Batch]
    model_shapes: Callable[[Batch], Batch]
    step_body: Callable[..., Callable]
    fetched: Callable[..., TrainState]
    traced_on: TracedOn
    shape: RunShape
    validation: Callable[[TrainState], Validation]
    builds_window: Callable[[TrainState, Candidate, int], bool]
    window_of: Callable[[TrainState, Candidate, int, Accumulation | None], TrainState]


KEPT_CANDIDATES = 4
"""The traced steps a process keeps for a trainer built anew to read (`candidate_for`), the latest: each
holds its lowered module."""

_CANDIDATES: collections.OrderedDict[Hashable, Candidate] = collections.OrderedDict()
_KEEPING = threading.Lock()


def _shaped(tree) -> Hashable:
    """`tree`'s structure and each leaf's abstract value and placement (an array's, a NumPy array's, a
    shape's, a scalar's beside its type, which tells True from 1), or the leaf itself where it is none."""
    def leaf(value) -> Hashable:
        if isinstance(value, jax.Array | jax.ShapeDtypeStruct | np.ndarray | np.generic | bool | int | float
                      | complex):
            aval = jax.typeof(value)
            return type(value), aval.shape, aval.dtype, aval.weak_type, getattr(value, "sharding", None)
        return value
    leaves, structure = jax.tree.flatten(tree)
    return structure, tuple(map(leaf, leaves))


def _shapes(tree):
    """`tree` with each array leaf as its shape, dtype and placement, as a kept step holds a batch it
    traced over and none of its values."""
    return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), leaf.dtype,
                                                          sharding=getattr(leaf, "sharding", None))
                        if hasattr(leaf, "dtype") else leaf, tree)


def candidate_for[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], state: TrainState, batch: Batch,
                                   schedule: Schedule, placement, replicated, split: int, narrow: bool, *,
                                   mesh: Mesh | AbstractMesh | None = None,
                                   platforms: tuple[str, ...] | None = None) -> Candidate:
    """`_candidate`'s step, or the one an equal trainer traced in this process over the same shapes
    (`Trainer._step_key`; row 3873: each fit of a resumed run traced its step again, 6.5 to 12.9 s, and
    lowered it, 9 s), its pipeline's schedule the one its microbatches make; a trainer whose step no value
    states traces its own."""
    # Read as the candidate is made, at the rung the search put the objective at (row 3873).
    stated = stepping.step_key()
    # What the step traces from beside its arguments and the trainer's settings: the compiles' refusals
    # its kernels' derivation reads (a relowering after a spill reads its proposal's), and the context.
    key = None if stated is None else (
        stated, _shaped(state), _shaped(batch), _shaped(placement), replicated, split, narrow, mesh,
        platforms, stepping.shape.rows,
        stepping.shape.split, stepping.shape.prefetch, stepping.shape.examples, report.refused_launches(),
        trace_context())
    if key is not None:
        with _KEEPING:
            if (kept := _CANDIDATES.get(key)) is not None:
                _CANDIDATES.move_to_end(key)
                # Its links decide as its trace decided them: the key holds their figures (`trace_context`).
                spread_over(kept.spread)
                sampled = _sampled(stepping, kept, batch, placement, replicated, split, platforms)
                return dataclasses.replace(kept, sampled=sampled)
    candidate = _candidate(stepping, state, batch, schedule, placement, replicated, split, narrow,
                           mesh=mesh, platforms=platforms)
    if key is not None:
        # Kept detached from this trainer: its batch as shapes, its inspection's step none (`_sampled`
        # builds the reader's own).
        with _KEEPING:
            _CANDIDATES[key] = dataclasses.replace(candidate, micro=_shapes(candidate.micro), sampled=None,
                                                   spread=spread_axes())
            while len(_CANDIDATES) > KEPT_CANDIDATES:
                _CANDIDATES.popitem(last=False)
    return candidate


def _sampled[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], candidate: Candidate, batch: Batch,
                              placement, replicated, split: int, platforms: tuple[str, ...] | None
                              ) -> Callable[[], jax.stages.Lowered] | None:
    """The sampled step `candidate`'s reader lowers (`Candidate.sampled`), its own trainer's, as the step
    lowers, over the same arguments, for the same platforms: none without an inspection."""
    if stepping.settings.inspection is None:
        return None
    watched = step_program(stepping, candidate.shapes, candidate.copies, candidate.shardings, placement,
                           replicated, split, sampled=True)
    part = jax.ShapeDtypeStruct((), jnp.int32, sharding=replicated)
    given = placed_shapes(batch, placement)
    return functools.partial(lambda *arguments: watched.trace(*arguments).lower(lowering_platforms=platforms),
                             *apart(candidate.prepared), given, part)


def _candidate[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], state: TrainState, batch: Batch,
                                schedule: Schedule, placement, replicated, split: int, narrow: bool, *,
                                mesh: Mesh | AbstractMesh | None = None,
                                platforms: tuple[str, ...] | None = None) -> Candidate:
    """The step reading `batch` in `split` microbatches, lowered at the objective's current rung, with what
    the fit check holds beside it. Each launch takes its derived config, but where the run measures configs
    (`tune_kernels`), whose lowering settles them (`tune`). `mesh` and `platforms` lower it for other
    devices, untuned (`lowered_for`)."""
    settings = stepping.settings
    micro = microbatch_shapes(batch, split)
    shapes, copies, prepared, _ = prepared_for(stepping, state, micro, schedule, split, narrow)
    shardings = stepping.shardings(prepared)
    if mesh is not None:
        shardings = jax.tree.map(lambda s: NamedSharding(mesh, s.spec, memory_kind=s.memory_kind),
                                 shardings)
    prepared = placed_shapes(prepared, shardings)
    # The check reads the state's arguments as already on the devices. The copies and accumulators
    # `run` builds on the first call are not yet, so they count with the prefetched batches beside
    # the step: a window it builds over the one the state holds (a split's other than the state's, a ramp
    # stage's of other shapes) donates that one's leaves of its shapes, and builds the rest (`taken`).
    fresh = [] if prepared.window is None else taken(state.window, prepared.window)[0]
    # A step lowered for other devices is checked against none of this process's.
    # The copies the runner makes where the state holds another program's, or none (`holds_copies`).
    made = [] if prepared.compute is None or holds_copies(state.compute, copies) else [prepared.compute]
    built = 0 if mesh is not None else placed_bytes([*made, *fresh])
    held = 0 if mesh is not None else prefetched_bytes(batch, placement, stepping.shape.prefetch) + built
    jitted = step_program(stepping, shapes, copies, shardings, placement, replicated, split)
    part = jax.ShapeDtypeStruct((), jnp.int32, sharding=replicated)
    # The batch as shapes with the placement it is read under, whoever lowers it: a lowering over the host
    # arrays a step compiled ahead reads (`first_step_batch`) and one over the placed arrays a run steps
    # are two texts of one program (`dew.nn.kernels.options.program_digest`), the types of a helper call's
    # arguments, their shardings among them, telling its functions apart in the module.
    given = placed_shapes(batch, placement)
    lower = Routed(functools.partial(
        lambda *arguments: jitted.trace(*arguments).lower(lowering_platforms=platforms),
        *apart(prepared), given, part))
    tuned = settings.tune_kernels and platforms is None
    if tuned:
        lowered, kernels = tuning.tune(lower, measure=True), lower.report()
    else:
        # The tuner is off (bible §6): each launch takes its derived config, nothing settled or read.
        made, found = routed(lambda: jitted.trace(*apart(prepared), given, part))
        lowered = made.lower(lowering_platforms=platforms)
        kernels = kernel_report(found, lowered)
    candidate = Candidate(micro, shapes, copies, shardings, prepared, held, built, lowered, kernels)
    return dataclasses.replace(candidate, sampled=_sampled(stepping, candidate, batch, placement, replicated,
                                                           split, platforms))


class Prepared(NamedTuple):
    """A step's state prepared over microbatches of a batch's shapes (`prepared_for`): the loss's shapes and
    its forward program (`traced`; None under a custom step), the narrow copies, and the state."""

    shapes: tuple | None
    copies: dict
    prepared: TrainState
    traced: jcore.Jaxpr | None


def prepared_for[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], state: TrainState, batch: Batch,
                                  schedule: Schedule, split: int, narrow: bool) -> Prepared:
    """The loss's shapes (None under a custom step), the narrow copies and the state's shapes with the
    accumulation of an update's window, `split` microbatches a step (DiLoCo's groups' `windowed`), and
    the copies the step builds, as it traces them over microbatches of `batch`'s shapes, none where
    `narrow` is off; and the loss's forward program that trace made (None under a custom step)."""
    settings = stepping.settings
    window = split * settings.accumulation
    with group_of(state, batch) as (view, rows):
        traced, shapes = (None, None) if settings.step is not None else stepping.loss_shape(
            view, rows, window, split)
    if shapes is not None:
        stepping.staging.refuse_unpipelined(schedule, settings.objective)
    prepared = windowed(lambda state, rows, *, shape_only: stepping.initialize_accumulation(
        state, rows, shapes, window=window, split=split, shape_only=shape_only), state, batch)
    # DiLoCo's groups read their own parameters, which no narrow copy shadows.
    # A step the trainer composes narrows, from the loss it traced; a custom step's traces none (`traced`
    # None), and a scaled loss's copies none.
    narrows = narrow and not settings.dynamic_scale and settings.diloco is None
    copies = narrow_copies(state, traced) if narrows and traced is not None else {}
    prepared = dataclasses.replace(prepared, compute=jax.eval_shape(
        lambda variables: narrowed(variables, copies), prepared.variables) if copies else None)
    return Prepared(shapes, copies, prepared, traced)


def step_program[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], shapes, copies: dict,
                                  shardings: Placement[TrainState], placement, replicated: NamedSharding,
                                  split: int, sampled: bool = False):
    """The step over one of the `split` microbatches of a batch
    (`microbatch`), which counts the step at its last microbatch. It takes the state apart
    (`apart`): the moving state, which it donates and returns advanced, and the frozen tree, which
    it reads and neither donates nor returns, so the frozen arrays stay the ones the run (and the
    objective's held tree) holds, and no output copies them."""
    settings = stepping.settings
    # The transaction brings the optimizer state and the EMA the layout keeps on the host to the device
    # once its gradient is ready (`Transaction.past`); a custom step, and every other field, reads them
    # from its start.
    late = () if settings.step is not None else tuple(set(COMMIT_FIELDS) & set(settings.layout.host))
    body = stepping.step_body(shapes, copies, split, {field: getattr(shardings, field) for field in late},
                              sampled, shardings.variables["params"])
    body = body if settings.diloco is None else settings.diloco.step(body, settings.device_mesh, split)
    moving, frozen = apart(shardings)

    def step(current, held, batch, part):
        # The body sees every field on the device; the out shardings
        # return the host-resident ones to pinned host memory.
        current = together(current, held)
        micro = microbatch(batch, part, split)
        early = [field for field in settings.layout.host if field not in late]
        has_batch_stats = (split > 1 and current.window is not None
                           and current.window.batch_statistics is not None)
        whole = stepping.model_batch(batch) if has_batch_stats else None
        advanced, loss, aux, committed = body(stepping.fetched(current, shardings, early),
                                              stepping.model_batch(micro), part, batch_tokens(micro),
                                              micro.get(CLOSES_WINDOW), whole)
        stepped, *rest = outcome(current, advanced, loss, aux, committed, batch, part == split - 1,
                                 batch_consumed(micro), split=split)
        return (apart(stepped)[0], *rest)

    returned = (moving, replicated, replicated, replicated, replicated, replicated)
    return owned_program(step, in_shardings=(moving, frozen, placement, replicated),
                         out_shardings=returned, donate_argnums=0)


def lowered_for[LossT, EffectsT](stepping: Stepping[LossT, EffectsT], state: TrainState, batch: Batch,
                                 generation: str, narrow: bool) -> jax.stages.Lowered:
    """The step `compile` builds first, traced as on a GPU of the CUDA `generation` and lowered for one
    (`dew.training._lowering`) on this process's devices. It traces on an abstract mesh of that GPU's
    devices, from which JAX's Pallas lowering reads the GPU, at the base rung its trainer put the
    objective at (`StepCompiler.at_base_rung`), its narrow copies where that rung keeps them (`narrow`), and
    the plan's first split: a run that does not fit splits its batch or recomputes more, no other kernel.
    A CPU-owned state's step realizes its parameters on the host as it runs, and is not lowered."""
    settings = stepping.settings
    if settings.host_master:
        raise ValueError("lower_for lowers a step whose state the devices hold; this layout keeps "
                         "the variables on the host")
    mesh = gpu_mesh(settings.device_mesh, generation)
    with assumed(generation), stepping.traced_on(mesh) as schedule:
        state, batch = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), (state, batch))
        split = splits(stepping.shape.rows, batch, mesh, settings.microbatches, stepping.shape.split)[0]
        placement, replicated = batch_shardings(mesh, batch), NamedSharding(mesh, P())
        return candidate_for(stepping, state, batch, schedule, placement, replicated, split, narrow,
                             mesh=mesh, platforms=('cuda',)).lowered
